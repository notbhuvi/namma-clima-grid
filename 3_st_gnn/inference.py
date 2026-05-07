"""
Module 3e — STGNNInference
============================

Inference wrapper for the trained SpatioTemporalGNN.  Two consumers:

  1. FastAPI (Module 5) — synchronous call that returns heat_stress and
     flood_risk scores for all 198 wards as a JSON-serialisable dict.

  2. RL optimizer (Module 4) — exposes the ward node_embedding tensor
     so the PPO agent can observe current city-state without re-running
     feature engineering from scratch.

Loading priority
─────────────────
  1. Checkpoint path argument / STGNN_CHECKPOINT env var
  2. checkpoints/gnn_best.pt (relative to this module)
  3. MLflow best run in "st_gnn_ward_risk" experiment
  4. Untrained fallback (random weights — for integration tests)

Prediction input
─────────────────
  predict() accepts either:
    a) A (N, T, F) float32 tensor — direct forward pass
    b) A ward_features DataFrame — converted via the same normalisation
       pipeline used in training so you never hand-roll feature engineering
       in the calling code.

Single-snapshot convenience
────────────────────────────
  predict_snapshot(df) takes a single ward-features snapshot (one row per
  ward) and replicates it T times to fill the temporal window.  Useful for
  REST endpoints where only the latest snapshot is available.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from loguru import logger

_MODULE_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _MODULE_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "8_data"))

from graph_builder import build_ward_graph
from model         import SpatioTemporalGNN, HIDDEN_DIM
from dataset       import (
    FEATURE_COLS, T_WINDOW, NUM_WARDS,
    _normalise_snapshot, _synthetic_history,
)

_GRAPH_CACHE = _MODULE_DIR / "checkpoints" / "ward_graph.pt"


# ---------------------------------------------------------------------------
# STGNNInference
# ---------------------------------------------------------------------------

class STGNNInference:
    """
    Production inference wrapper for the ST-GNN.

    Args:
        model:  Trained SpatioTemporalGNN.
        graph:  Pre-built ward adjacency graph dict (from graph_builder).
        device: Torch device.
    """

    def __init__(
        self,
        model: SpatioTemporalGNN,
        graph: Optional[Dict] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device).eval()
        self.graph = graph or build_ward_graph(cache_path=_GRAPH_CACHE)
        self._ei = self.graph["edge_index"].to(self.device)
        self._ew = self.graph["edge_weight"].to(self.device)
        self._ward_ids: List[int] = self.graph["ward_ids"].tolist()

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: Optional[torch.device] = None,
    ) -> "STGNNInference":
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        model = SpatioTemporalGNN()
        ckpt  = torch.load(path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        epoch = ckpt.get("epoch", "?")
        rmse  = ckpt.get("val_combined_rmse", "?")
        logger.info(f"Loaded GNN checkpoint {path} | epoch={epoch}  combined_RMSE={rmse}")
        return cls(model, device=device)

    @classmethod
    def from_mlflow(
        cls,
        experiment_name: str = "st_gnn_ward_risk",
        tracking_uri: str    = "http://localhost:5001",
        device: Optional[torch.device] = None,
    ) -> "STGNNInference":
        try:
            import mlflow  # type: ignore
            mlflow.set_tracking_uri(tracking_uri)
            client = mlflow.tracking.MlflowClient()
            exp    = client.get_experiment_by_name(experiment_name)
            if exp is None:
                raise RuntimeError(f"Experiment '{experiment_name}' not found")
            runs = client.search_runs(
                exp.experiment_id,
                order_by=["metrics.best_val_combined_rmse ASC"],
                max_results=1,
            )
            if not runs:
                raise RuntimeError("No completed runs found")
            run_id = runs[0].info.run_id
            model  = mlflow.pytorch.load_model(f"runs:/{run_id}/model")
            logger.info(f"Loaded GNN from MLflow run {run_id}")
            return cls(model, device=device)
        except Exception as exc:
            logger.warning(f"MLflow load failed ({exc}) — using untrained model")
            return cls.untrained(device)

    @classmethod
    def untrained(cls, device: Optional[torch.device] = None) -> "STGNNInference":
        logger.warning("Using untrained ST-GNN — predictions are random")
        return cls(SpatioTemporalGNN(), device=device)

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _forward(
        self, x_seq: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """x_seq: (N, T, F)"""
        return self.model(
            x_seq.to(self.device), self._ei, self._ew
        )

    # ------------------------------------------------------------------
    # Gradient-based feature importance
    # ------------------------------------------------------------------
    # Readable feature names for the 12 FEATURE_COLS
    _FEATURE_NAMES = [
        "ndvi", "ndbi", "impervious_pct", "lst", "rainfall",
        "pm25", "population", "area", "industrial_weight",
        "longitude", "latitude", "heat_proxy",
    ]

    def _gradient_importance(
        self, x_seq: torch.Tensor
    ) -> Dict[str, np.ndarray]:
        """
        Compute per-ward, per-feature importance via input-gradient method.

        For each output head (heat, flood), computes
            importance[n, f] = mean_over_T( |x[n,t,f] * grad_x[n,t,f]| )

        Returns dict with:
            heat_importance  : (N, F) float32
            flood_importance : (N, F) float32
        """
        x = x_seq.clone().to(self.device).requires_grad_(True)
        out = self.model(x, self._ei, self._ew)

        # Heat importance
        self.model.zero_grad()
        out["heat_stress"].sum().backward(retain_graph=True)
        heat_grad = (x.grad * x).abs().detach().cpu().numpy()   # (N, T, F)
        heat_imp  = heat_grad.mean(axis=1)                       # (N, F)

        # Flood importance
        x.grad.zero_()
        out["flood_risk"].sum().backward()
        flood_grad = (x.grad * x).abs().detach().cpu().numpy()
        flood_imp  = flood_grad.mean(axis=1)

        return {
            "heat_importance":  heat_imp.astype(np.float32),
            "flood_importance": flood_imp.astype(np.float32),
        }

    def _top_features(
        self,
        importance: np.ndarray,
        top_k: int = 4,
    ) -> List[List[Dict[str, float]]]:
        """
        For each ward, return top-k features sorted by importance.

        Args:
            importance: (N, F) importance scores
            top_k:      number of top features per ward
        Returns:
            List of N lists, each containing top_k dicts:
              {"feature": str, "importance": float}
        """
        N, F = importance.shape
        names = self._FEATURE_NAMES[:F]
        result = []
        for n in range(N):
            row = importance[n]
            total = row.sum() + 1e-12
            normed = row / total  # normalise to sum=1
            idxs = np.argsort(normed)[::-1][:top_k]
            result.append([
                {"feature": names[i], "importance": round(float(normed[i]), 4)}
                for i in idxs
            ])
        return result

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------
    def predict(
        self, x_seq: torch.Tensor, explain: bool = False
    ) -> pd.DataFrame:
        """
        Run inference on a (N, T, F) feature tensor.

        Args:
            x_seq:   (N, T, F) input features.
            explain: If True, add 'top_contributing_features' column.

        Returns a DataFrame with columns:
          ward_id, heat_stress_score, flood_risk_score,
          risk_level  ('low'|'medium'|'high'|'critical')
          [top_contributing_features — if explain=True]
        """
        out = self._forward(x_seq)
        heat_np  = out["heat_stress"].cpu().numpy()
        flood_np = out["flood_risk"].cpu().numpy()
        df = self._build_result_df(heat_np, flood_np)

        if explain:
            imp = self._gradient_importance(x_seq)
            # Combined importance: weighted by task importance
            combined = 0.6 * imp["heat_importance"] + 0.4 * imp["flood_importance"]
            top_feats = self._top_features(combined, top_k=4)
            df["top_contributing_features"] = top_feats

        return df

    def predict_snapshot(
        self, ward_features_df: pd.DataFrame, explain: bool = False
    ) -> pd.DataFrame:
        """
        Predict from a single ward-features snapshot (one row per ward).

        The single snapshot is replicated T times to fill the temporal
        window.  This is the primary endpoint for the FastAPI /predict/risk
        route.
        """
        x = self._df_to_tensor(ward_features_df)   # (N, F)
        x_seq = x.unsqueeze(1).expand(-1, T_WINDOW, -1)   # (N, T, F)
        return self.predict(x_seq, explain=explain)

    def explain(self, x_seq: torch.Tensor) -> Dict[str, object]:
        """
        Full explainability report for a given input.

        Returns:
            {
              "per_ward": [
                {
                  "ward_id": int,
                  "heat_stress": float,
                  "flood_risk": float,
                  "top_contributing_features": [
                    {"feature": str, "importance": float}, ...
                  ]
                }, ...
              ],
              "global_feature_importance": {
                "feature_name": mean_importance_score, ...
              }
            }
        """
        out = self._forward(x_seq)
        heat_np  = out["heat_stress"].cpu().numpy()
        flood_np = out["flood_risk"].cpu().numpy()

        imp = self._gradient_importance(x_seq)
        combined = 0.6 * imp["heat_importance"] + 0.4 * imp["flood_importance"]
        top_feats = self._top_features(combined, top_k=4)

        N = len(heat_np)
        ward_ids = self._ward_ids[:N] if len(self._ward_ids) >= N else list(range(1, N + 1))

        per_ward = []
        for i in range(N):
            per_ward.append({
                "ward_id":     ward_ids[i],
                "heat_stress": round(float(heat_np[i]), 2),
                "flood_risk":  round(float(flood_np[i]), 2),
                "top_contributing_features": top_feats[i],
            })

        # Global feature importance: mean across all wards
        F = combined.shape[1]
        names = self._FEATURE_NAMES[:F]
        global_total = combined.sum() + 1e-12
        global_imp = {
            names[f]: round(float(combined[:, f].sum() / global_total), 4)
            for f in range(F)
        }
        # Sort descending
        global_imp = dict(sorted(global_imp.items(), key=lambda kv: -kv[1]))

        logger.info("Global feature importance distribution:")
        for feat, score in global_imp.items():
            bar = "#" * int(score * 50)
            logger.info(f"  {feat:20s} {score:.4f} {bar}")

        return {
            "per_ward": per_ward,
            "global_feature_importance": global_imp,
        }

    def predict_history(self, ward_features_history_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict from a multi-snapshot history DataFrame (columns include
        snapshot_idx).  Builds the latest T-step window and predicts.
        """
        df = ward_features_history_df.copy()
        if "snapshot_idx" not in df.columns:
            ts = pd.to_datetime(df["snapshot_ts"])
            unique_ts = sorted(ts.unique())
            ts_map = {t: i for i, t in enumerate(unique_ts)}
            df["snapshot_idx"] = ts.map(ts_map)

        df = _normalise_snapshot(df)
        latest_snap = df["snapshot_idx"].max()
        window_snaps = sorted(df["snapshot_idx"].unique())[-T_WINDOW:]

        # Build (N, T, F) tensor
        ward_ids = sorted(df["ward_id"].unique())
        wi_map   = {w: i for i, w in enumerate(ward_ids)}
        si_map   = {s: i for i, s in enumerate(window_snaps)}
        N = len(ward_ids)
        T = len(window_snaps)
        F = len(FEATURE_COLS)
        x_np = np.zeros((N, T, F), dtype=np.float32)

        for _, row in df[df["snapshot_idx"].isin(window_snaps)].iterrows():
            wi = wi_map[row["ward_id"]]
            si = si_map[row["snapshot_idx"]]
            for fi, col in enumerate(FEATURE_COLS):
                x_np[wi, si, fi] = float(row.get(col, 0.0))

        # Pad to T_WINDOW if fewer snapshots are available
        if T < T_WINDOW:
            pad = np.zeros((N, T_WINDOW - T, F), dtype=np.float32)
            x_np = np.concatenate([pad, x_np], axis=1)

        return self.predict(torch.from_numpy(x_np))

    # ------------------------------------------------------------------
    # RL / GNN node embedding
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_node_embeddings(
        self, x_seq: torch.Tensor
    ) -> np.ndarray:
        """
        Return (N, hidden_dim) float32 array of node embeddings.
        Primary interface for the RL optimizer (Module 4).
        """
        out = self._forward(x_seq)
        return out["node_embedding"].cpu().numpy().astype(np.float32)

    def encode_snapshot_embeddings(
        self, ward_features_df: pd.DataFrame
    ) -> np.ndarray:
        """Convenience: single snapshot → (N, hidden_dim) node embeddings."""
        x = self._df_to_tensor(ward_features_df)
        x_seq = x.unsqueeze(1).expand(-1, T_WINDOW, -1)
        return self.encode_node_embeddings(x_seq)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _df_to_tensor(self, df: pd.DataFrame) -> torch.Tensor:
        """Normalise a ward-features DataFrame → (N, F) float32 tensor."""
        normed = _normalise_snapshot(df)
        ward_ids = sorted(normed["ward_id"].unique())
        N = len(ward_ids)
        F = len(FEATURE_COLS)
        x_np = np.zeros((N, F), dtype=np.float32)
        wi_map = {w: i for i, w in enumerate(ward_ids)}
        for _, row in normed.iterrows():
            wi = wi_map[row["ward_id"]]
            for fi, col in enumerate(FEATURE_COLS):
                x_np[wi, fi] = float(row.get(col, 0.0))
        return torch.from_numpy(x_np)

    def _build_result_df(
        self,
        heat: np.ndarray,
        flood: np.ndarray,
    ) -> pd.DataFrame:
        def _level(h: float, f: float) -> str:
            combined = 0.6 * h + 0.4 * f
            if combined >= 75:
                return "critical"
            if combined >= 50:
                return "high"
            if combined >= 25:
                return "medium"
            return "low"

        n = len(heat)
        ward_ids = self._ward_ids[:n] if len(self._ward_ids) >= n else list(range(1, n + 1))
        return pd.DataFrame({
            "ward_id":           ward_ids,
            "heat_stress_score": np.round(heat, 2).tolist(),
            "flood_risk_score":  np.round(flood, 2).tolist(),
            "risk_level":        [_level(h, f) for h, f in zip(heat, flood)],
        }).sort_values("ward_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Convenience loader for FastAPI startup
# ---------------------------------------------------------------------------

def load_default_inference(
    checkpoint_path: Optional[str] = None,
) -> STGNNInference:
    """
    Load with priority chain:
      explicit arg → STGNN_CHECKPOINT env → checkpoints/gnn_best.pt → MLflow → untrained
    """
    candidates = [
        checkpoint_path,
        os.getenv("STGNN_CHECKPOINT"),
        str(_MODULE_DIR / "checkpoints" / "gnn_best.pt"),
    ]
    for path in candidates:
        if path and Path(path).exists():
            try:
                return STGNNInference.from_checkpoint(path)
            except Exception as exc:
                logger.warning(f"Checkpoint load failed ({exc})")
    try:
        return STGNNInference.from_mlflow()
    except Exception:
        pass
    return STGNNInference.untrained()


# ---------------------------------------------------------------------------
# __main__ — inference smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from loguru import logger

    logger.info("STGNNInference smoke test (untrained model)")

    gnn = STGNNInference.untrained()

    # Build synthetic snapshot for all 198 wards
    rng = np.random.default_rng(42)
    sample_df = pd.DataFrame({
        "ward_id":            list(range(1, NUM_WARDS + 1)),
        "ndvi":               rng.uniform(0.1, 0.6, NUM_WARDS).tolist(),
        "ndbi":               rng.uniform(-0.2, 0.4, NUM_WARDS).tolist(),
        "impervious_pct":     rng.uniform(30, 90, NUM_WARDS).tolist(),
        "lst_celsius":        rng.uniform(28, 44, NUM_WARDS).tolist(),
        "rainfall_mm_24h":    rng.exponential(1.5, NUM_WARDS).tolist(),
        "pm25_ugm3":          rng.uniform(20, 80, NUM_WARDS).tolist(),
        "population_density": rng.lognormal(8.5, 0.5, NUM_WARDS).tolist(),
    })

    result = gnn.predict_snapshot(sample_df, explain=True)
    print(result.head(5).to_string(index=False))
    assert "top_contributing_features" in result.columns, "explain column missing"

    emb = gnn.encode_snapshot_embeddings(sample_df)
    assert emb.shape == (NUM_WARDS, HIDDEN_DIM), f"Expected ({NUM_WARDS},{HIDDEN_DIM}), got {emb.shape}"

    # Test full explain() API
    x = gnn._df_to_tensor(sample_df)
    x_seq = x.unsqueeze(1).expand(-1, T_WINDOW, -1)
    report = gnn.explain(x_seq)
    assert "per_ward" in report
    assert "global_feature_importance" in report
    assert len(report["per_ward"]) == NUM_WARDS
    assert "top_contributing_features" in report["per_ward"][0]
    logger.info(f"Top features ward 1: {report['per_ward'][0]['top_contributing_features']}")

    # Test history path
    hist_df = _synthetic_history(n_snapshots=20, n_wards=NUM_WARDS, seed=1)
    result2 = gnn.predict_history(hist_df)
    assert len(result2) == NUM_WARDS

    logger.success(f"Node embeddings shape: {emb.shape}")
    logger.success("inference.py smoke test passed.")
