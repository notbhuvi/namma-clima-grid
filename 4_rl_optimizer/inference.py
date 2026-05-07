"""
Module 4d — GreenInfraOptimizer
==================================

Inference wrapper for the trained PPO policy.  Used by:

  1. FastAPI (Module 5) — POST /interventions/recommend
     Takes current GNN risk scores → returns a ranked list of
     ward-intervention recommendations within a budget.

  2. Background scheduler (Airflow) — nightly re-planning run
     Loads latest GNN snapshot, runs a full planning episode,
     writes recommendations to PostgreSQL.

Loading priority
─────────────────
  1. Checkpoint path argument / RL_CHECKPOINT env var
  2. checkpoints/ppo_best.pt (module-relative)
  3. MLflow latest run
  4. Untrained fallback (random policy — for integration tests)

Output schema
─────────────
  List of RecommendedIntervention, ordered by expected_benefit DESC:
    ward_id             : int
    intervention_type   : str
    expected_heat_reduction : float  (percentage points)
    expected_flood_reduction: float  (percentage points)
    cost                : int
    priority_rank       : int
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from loguru import logger

_MODULE_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _MODULE_DIR.parent

from environment import (
    BengaluruClimaEnv, INTERVENTIONS, NUM_WARDS,
    OBS_WARD_DIM, N_INTERVENTIONS, EPISODE_LEN,
)
from model import ActorCritic, N_ACTIONS


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class RecommendedIntervention:
    ward_id:                   int
    intervention_type:         str
    expected_heat_reduction:   float
    expected_flood_reduction:  float
    cost:                      int
    priority_rank:             int
    ward_heat_stress:          float
    ward_flood_risk:           float


# ---------------------------------------------------------------------------
# GreenInfraOptimizer
# ---------------------------------------------------------------------------

class GreenInfraOptimizer:
    """
    Wraps the trained PPO policy for production inference.

    Args:
        model:  Trained ActorCritic.
        device: Torch device.
    """

    def __init__(
        self,
        model: ActorCritic,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device).eval()

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: Optional[torch.device] = None,
    ) -> "GreenInfraOptimizer":
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        model = ActorCritic()
        ckpt  = torch.load(path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        ep_rew = ckpt.get("ep_reward_mean", "?")
        logger.info(f"Loaded PPO checkpoint {path} | ep_reward_mean={ep_rew}")
        return cls(model, device)

    @classmethod
    def from_mlflow(
        cls,
        experiment_name: str = "rl_green_infra",
        tracking_uri: str    = "http://localhost:5001",
        device: Optional[torch.device] = None,
    ) -> "GreenInfraOptimizer":
        try:
            import mlflow  # type: ignore
            mlflow.set_tracking_uri(tracking_uri)
            client = mlflow.tracking.MlflowClient()
            exp    = client.get_experiment_by_name(experiment_name)
            if exp is None:
                raise RuntimeError(f"Experiment '{experiment_name}' not found")
            runs = client.search_runs(
                exp.experiment_id,
                order_by=["metrics.best_ep_reward DESC"],
                max_results=1,
            )
            if not runs:
                raise RuntimeError("No runs found")
            run_id = runs[0].info.run_id
            model  = mlflow.pytorch.load_model(f"runs:/{run_id}/model")
            logger.info(f"Loaded PPO from MLflow run {run_id}")
            return cls(model, device)
        except Exception as exc:
            logger.warning(f"MLflow load failed ({exc}) — using untrained model")
            return cls.untrained(device)

    @classmethod
    def untrained(cls, device: Optional[torch.device] = None) -> "GreenInfraOptimizer":
        logger.warning("Using untrained PPO — recommendations are random")
        return cls(ActorCritic(), device)

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------
    def recommend(
        self,
        heat_stress: np.ndarray,
        flood_risk:  np.ndarray,
        budget:      int = 60,
        top_k:       int = 10,
        deterministic: bool = True,
    ) -> List[RecommendedIntervention]:
        """
        Run one planning episode and return top-k intervention recommendations.

        Args:
            heat_stress:   (N,) float32 0–100 per ward (from GNN).
            flood_risk:    (N,) float32 0–100 per ward (from GNN).
            budget:        Total intervention cost budget.
            top_k:         Maximum number of recommendations to return.
            deterministic: If True, policy takes argmax (greedy) actions.

        Returns:
            List of RecommendedIntervention, sorted by expected benefit DESC.
        """
        env = BengaluruClimaEnv(budget=budget, episode_len=EPISODE_LEN)
        obs_np, _ = env.reset()
        env.set_gnn_state(heat_stress, flood_risk)
        obs = torch.from_numpy(env._observe())   # re-observe with updated state

        recommendations: List[dict] = []
        seen_ward_interv: set = set()

        for _ in range(EPISODE_LEN):
            action = self.model.act(
                obs.to(self.device),
                budget_left=env._budget_left,
                deterministic=deterministic,
            )
            ward_idx   = action // N_INTERVENTIONS
            interv_idx = action  % N_INTERVENTIONS
            iv         = INTERVENTIONS[interv_idx]
            ward       = env._wards[ward_idx]
            key        = (ward.ward_id, interv_idx)

            # Cost check
            if iv.cost > env._budget_left:
                break

            if key not in seen_ward_interv:
                benefit = (
                    0.6 * abs(iv.heat_delta) / 100 * ward.heat_stress
                    + 0.4 * abs(iv.flood_delta) / 100 * ward.flood_risk
                )
                recommendations.append({
                    "ward_id":                   ward.ward_id,
                    "intervention_type":         iv.name,
                    "expected_heat_reduction":   round(abs(iv.heat_delta + abs(iv.lst_delta) * 2), 2),
                    "expected_flood_reduction":  round(abs(iv.flood_delta), 2),
                    "cost":                      iv.cost,
                    "benefit_score":             round(benefit, 4),
                    "ward_heat_stress":          round(ward.heat_stress, 1),
                    "ward_flood_risk":           round(ward.flood_risk, 1),
                })
                seen_ward_interv.add(key)

            obs_np, _, done, truncated, _ = env.step(action)
            obs = torch.from_numpy(obs_np)

            if done or truncated:
                break

        # Sort by benefit and assign rank
        recommendations.sort(key=lambda x: x["benefit_score"], reverse=True)
        result = []
        for rank, rec in enumerate(recommendations[:top_k], start=1):
            result.append(RecommendedIntervention(
                ward_id=rec["ward_id"],
                intervention_type=rec["intervention_type"],
                expected_heat_reduction=rec["expected_heat_reduction"],
                expected_flood_reduction=rec["expected_flood_reduction"],
                cost=rec["cost"],
                priority_rank=rank,
                ward_heat_stress=rec["ward_heat_stress"],
                ward_flood_risk=rec["ward_flood_risk"],
            ))

        logger.info(
            f"Recommendations | budget={budget}  generated={len(result)}  "
            f"top_intervention={result[0].intervention_type if result else 'none'}"
        )
        return result

    def recommend_from_dataframe(
        self,
        risk_df: pd.DataFrame,
        budget: int = 60,
        top_k:  int = 10,
    ) -> List[RecommendedIntervention]:
        """
        Convenience wrapper: accepts the DataFrame returned by STGNNInference.

        Args:
            risk_df: DataFrame with columns ward_id, heat_stress_score, flood_risk_score.
            budget:  Intervention budget.
            top_k:   Max recommendations.
        """
        df = risk_df.sort_values("ward_id").reset_index(drop=True)
        heat  = df["heat_stress_score"].to_numpy(dtype=np.float32)
        flood = df["flood_risk_score"].to_numpy(dtype=np.float32)

        # Pad or trim to NUM_WARDS
        N = len(df)
        if N < NUM_WARDS:
            heat  = np.concatenate([heat,  np.full(NUM_WARDS - N, 40.0)])
            flood = np.concatenate([flood, np.full(NUM_WARDS - N, 20.0)])
        elif N > NUM_WARDS:
            heat  = heat[:NUM_WARDS]
            flood = flood[:NUM_WARDS]

        return self.recommend(heat, flood, budget=budget, top_k=top_k)

    def to_dataframe(
        self, recommendations: List[RecommendedIntervention]
    ) -> pd.DataFrame:
        """Convert recommendation list to a pandas DataFrame."""
        return pd.DataFrame([
            {
                "priority_rank":             r.priority_rank,
                "ward_id":                   r.ward_id,
                "intervention_type":         r.intervention_type,
                "expected_heat_reduction":   r.expected_heat_reduction,
                "expected_flood_reduction":  r.expected_flood_reduction,
                "cost":                      r.cost,
                "ward_heat_stress":          r.ward_heat_stress,
                "ward_flood_risk":           r.ward_flood_risk,
            }
            for r in recommendations
        ])


# ---------------------------------------------------------------------------
# Convenience loader for FastAPI
# ---------------------------------------------------------------------------

def load_default_optimizer(
    checkpoint_path: Optional[str] = None,
) -> GreenInfraOptimizer:
    candidates = [
        checkpoint_path,
        os.getenv("RL_CHECKPOINT"),
        str(_MODULE_DIR / "checkpoints" / "ppo_best.pt"),
    ]
    for path in candidates:
        if path and Path(path).exists():
            try:
                return GreenInfraOptimizer.from_checkpoint(path)
            except Exception as exc:
                logger.warning(f"Checkpoint load failed ({exc})")
    try:
        return GreenInfraOptimizer.from_mlflow()
    except Exception:
        pass
    return GreenInfraOptimizer.untrained()


# ---------------------------------------------------------------------------
# __main__ — inference smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from loguru import logger

    logger.info("GreenInfraOptimizer smoke test (untrained policy)")

    optimizer = GreenInfraOptimizer.untrained()

    # Synthetic GNN outputs for 198 wards
    rng = np.random.default_rng(42)
    heat  = rng.uniform(20, 80, NUM_WARDS).astype(np.float32)
    flood = rng.uniform(10, 60, NUM_WARDS).astype(np.float32)

    recs = optimizer.recommend(heat, flood, budget=40, top_k=5)
    df   = optimizer.to_dataframe(recs)
    print(df.to_string(index=False))

    assert len(recs) <= 5
    assert all(r.priority_rank >= 1 for r in recs)
    assert all(0 <= r.ward_id <= NUM_WARDS for r in recs)

    # DataFrame path
    risk_df = pd.DataFrame({
        "ward_id":           range(1, NUM_WARDS + 1),
        "heat_stress_score": heat,
        "flood_risk_score":  flood,
    })
    recs2 = optimizer.recommend_from_dataframe(risk_df, budget=30, top_k=3)
    assert len(recs2) <= 3

    logger.success(f"Top recommendation: Ward {recs[0].ward_id} → {recs[0].intervention_type}")
    logger.success("inference.py smoke test passed.")
