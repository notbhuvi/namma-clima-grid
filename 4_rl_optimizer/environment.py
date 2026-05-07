"""
Module 4a — BengaluruClimaEnv
================================

Gymnasium environment that simulates Bengaluru's 198 BBMP wards as a city
under climate stress and gives a PPO agent a budget of green infrastructure
interventions to reduce Urban Heat Island intensity and flash-flood risk.

MDP formulation
────────────────
  State  s_t ∈ ℝ^{N × (F + A)}
    Per ward: current heat_stress, flood_risk, ndvi, impervious_pct,
              rainfall_mm, pm25, budget_remaining, plus a one-hot vector
              of which intervention (if any) is currently active.

  Action a_t  — discrete multi-ward budget allocation
    The agent picks ONE ward per step and ONE intervention type to deploy.
    Combined action space: N_wards × N_interventions  (198 × 5 = 990)

  Reward r_t
    r_t = Σ_wards [ w_h·Δheat_reduction + w_f·Δflood_reduction ]
          − cost_penalty(action)
          + budget_efficiency_bonus

    Shaped to encourage spreading interventions and penalise repeated
    coverage of already-treated wards.

  Episode
    T = 30 steps (one "planning season")
    Each step consumes budget; episode ends when budget=0 or T exhausted.

Intervention types (5)
──────────────────────
  0  tree_planting       ndvi ↑0.08  lst ↓1.5°C  cost=2
  1  green_roof          ndvi ↑0.04  lst ↓0.8°C  impervious ↓5  cost=3
  2  permeable_pavement  impervious ↓12  flood_risk ↓15  cost=4
  3  urban_wetland       flood_risk ↓25  ndvi ↑0.05  cost=6
  4  cool_pavement       lst ↓1.0°C  heat_stress ↓8  cost=2

Dynamics
─────────
  After each intervention, ward state is updated via a simple linear
  transition model.  Random climate shocks (heavy rain, heat waves) are
  injected every ~5 steps to prevent over-exploitation of a static policy.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from loguru import logger

NUM_WARDS        = 198
N_INTERVENTIONS  = 5
EPISODE_LEN      = 30
INITIAL_BUDGET   = 60    # total cost units per episode
WARD_FEAT_DIM    = 8     # features per ward in observation
# obs dim per ward = WARD_FEAT_DIM + N_INTERVENTIONS (one-hot active)
OBS_WARD_DIM     = WARD_FEAT_DIM + N_INTERVENTIONS


# ---------------------------------------------------------------------------
# Intervention catalogue
# ---------------------------------------------------------------------------

@dataclass
class Intervention:
    name:           str
    cost:           int
    ndvi_delta:     float = 0.0
    impervious_delta: float = 0.0    # percentage points (negative = reduction)
    lst_delta:      float = 0.0      # Celsius (negative = cooling)
    heat_delta:     float = 0.0      # direct heat_stress change (0–100)
    flood_delta:    float = 0.0      # direct flood_risk change (0–100)

    def effectiveness(self) -> float:
        """Rough scalar effectiveness for reward shaping."""
        return (
            abs(self.heat_delta) * 0.6
            + abs(self.flood_delta) * 0.4
            + abs(self.ndvi_delta) * 20
            + abs(self.lst_delta) * 3
        )


INTERVENTIONS: List[Intervention] = [
    Intervention("tree_planting",       cost=2, ndvi_delta=0.08,  lst_delta=-1.5, heat_delta=-6.0),
    Intervention("green_roof",          cost=3, ndvi_delta=0.04,  lst_delta=-0.8, impervious_delta=-5.0,  heat_delta=-4.0),
    Intervention("permeable_pavement",  cost=4, impervious_delta=-12.0, flood_delta=-15.0),
    Intervention("urban_wetland",       cost=6, ndvi_delta=0.05,  flood_delta=-25.0, heat_delta=-3.0),
    Intervention("cool_pavement",       cost=2, lst_delta=-1.0,   heat_delta=-8.0),
]


# ---------------------------------------------------------------------------
# Ward state
# ---------------------------------------------------------------------------

@dataclass
class WardState:
    ward_id:          int
    heat_stress:      float   # 0–100
    flood_risk:       float   # 0–100
    ndvi:             float   # 0–1
    impervious_pct:   float   # 0–100
    rainfall_mm:      float   # mm
    pm25:             float   # μg/m³
    active_intervention: int  = -1   # index into INTERVENTIONS, -1 = none
    treatment_steps_left: int = 0    # remaining effect duration


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class BengaluruClimaEnv(gym.Env):
    """
    Gymnasium environment for green infrastructure budget allocation.

    observation_space : Box(N × OBS_WARD_DIM,)  — flattened
    action_space      : Discrete(N × N_INTERVENTIONS)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        initial_state: Optional[np.ndarray] = None,
        budget: int = INITIAL_BUDGET,
        episode_len: int = EPISODE_LEN,
        seed: int = 42,
        reward_w_heat: float = 0.6,
        reward_w_flood: float = 0.4,
    ) -> None:
        super().__init__()
        self._base_state   = initial_state    # (N, WARD_FEAT_DIM) optional
        self._budget_init  = budget
        self._ep_len       = episode_len
        self._rng          = np.random.default_rng(seed)
        self._w_heat       = reward_w_heat
        self._w_flood      = reward_w_flood

        # Spaces
        obs_dim = NUM_WARDS * OBS_WARD_DIM
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(NUM_WARDS * N_INTERVENTIONS)

        # State (initialised in reset)
        self._wards: List[WardState] = []
        self._budget_left = 0
        self._step_count  = 0
        self._treated: Dict[int, int] = {}   # ward_id → n_times treated

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._wards       = self._init_wards()
        self._budget_left = self._budget_init
        self._step_count  = 0
        self._treated     = {}
        self._stagnation  = 0          # steps with no meaningful improvement
        self._prev_city_score = self._city_score()

        return self._observe(), {"budget": self._budget_left}

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        ward_idx     = action // N_INTERVENTIONS
        interv_idx   = action  % N_INTERVENTIONS
        intervention = INTERVENTIONS[interv_idx]

        reward        = 0.0
        truncated     = False
        info: dict    = {}

        # ── Budget check ──────────────────────────────────────────────
        if intervention.cost > self._budget_left:
            # Invalid action: penalise and do nothing
            reward = -2.0
            info["invalid"] = True
        else:
            self._budget_left -= intervention.cost
            reward = self._apply_intervention(ward_idx, interv_idx)
            info["ward_id"]   = self._wards[ward_idx].ward_id
            info["interv"]    = intervention.name
            info["cost"]      = intervention.cost

        # ── Random climate shock every ~5 steps ───────────────────────
        self._step_count += 1
        if self._step_count % 5 == 0:
            self._apply_climate_shock()

        # ── Step decay: untreated wards slowly worsen ─────────────────
        self._step_decay()

        # Track stagnation: if city score didn't improve, increment counter
        cur_score = self._city_score()
        if cur_score >= self._prev_city_score - 0.5:
            self._stagnation += 1
        else:
            self._stagnation = 0
        self._prev_city_score = cur_score

        obs       = self._observe()
        done      = (self._budget_left <= 0) or (self._step_count >= self._ep_len) \
                    or (self._stagnation >= 10)
        truncated = (self._step_count >= self._ep_len) or (self._stagnation >= 10)

        info["budget_left"] = self._budget_left
        info["step"]        = self._step_count

        # Terminal bonus: reward total reduction achieved
        if done:
            reward += self._terminal_bonus()

        return obs, float(reward), done, truncated, info

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    def _init_wards(self) -> List[WardState]:
        wards = []
        if self._base_state is not None and len(self._base_state) == NUM_WARDS:
            for i, row in enumerate(self._base_state):
                wards.append(WardState(
                    ward_id=i + 1,
                    heat_stress=float(np.clip(row[0] * 100, 0, 100)),
                    flood_risk=float(np.clip(row[1] * 100, 0, 100)),
                    ndvi=float(np.clip(row[2], 0, 1)),
                    impervious_pct=float(np.clip(row[3] * 100, 0, 100)),
                    rainfall_mm=float(np.clip(row[4] * 20, 0, 50)),
                    pm25=float(np.clip(row[5] * 200, 0, 300)),
                ))
        else:
            # Synthetic initial state
            for i in range(NUM_WARDS):
                cols = 14
                r, c = divmod(i, cols)
                cbd  = math.exp(-((r-7)**2 + (c-7)**2) / 8)
                heat = float(self._rng.uniform(25, 60) + 20 * cbd)
                flood = float(self._rng.uniform(10, 40) + 15 * self._rng.random())
                wards.append(WardState(
                    ward_id=i + 1,
                    heat_stress=float(np.clip(heat, 0, 100)),
                    flood_risk=float(np.clip(flood, 0, 100)),
                    ndvi=float(np.clip(0.45 - 0.30 * cbd + self._rng.uniform(-0.05, 0.05), 0, 1)),
                    impervious_pct=float(np.clip(40 + 40 * cbd + self._rng.uniform(-5, 5), 0, 100)),
                    rainfall_mm=float(self._rng.exponential(2.0)),
                    pm25=float(self._rng.lognormal(3.7 + 0.5 * cbd, 0.3)),
                ))
        return wards

    def _observe(self) -> np.ndarray:
        """
        Flatten all ward states into a (N × OBS_WARD_DIM,) float32 array.
        All values normalised to [0, 1].
        """
        obs = np.zeros((NUM_WARDS, OBS_WARD_DIM), dtype=np.float32)
        budget_norm = self._budget_left / self._budget_init

        for i, w in enumerate(self._wards):
            obs[i, 0] = w.heat_stress    / 100.0
            obs[i, 1] = w.flood_risk     / 100.0
            obs[i, 2] = w.ndvi
            obs[i, 3] = w.impervious_pct / 100.0
            obs[i, 4] = min(w.rainfall_mm / 30.0, 1.0)
            obs[i, 5] = min(w.pm25 / 200.0, 1.0)
            obs[i, 6] = budget_norm
            obs[i, 7] = min(self._treated.get(w.ward_id, 0) / 3.0, 1.0)
            # One-hot: active intervention
            if 0 <= w.active_intervention < N_INTERVENTIONS:
                obs[i, WARD_FEAT_DIM + w.active_intervention] = 1.0

        return obs.reshape(-1)

    def _apply_intervention(self, ward_idx: int, interv_idx: int) -> float:
        """Apply intervention with stochastic transition, compute shaped reward."""
        w = self._wards[ward_idx]
        iv = INTERVENTIONS[interv_idx]

        prev_heat  = w.heat_stress
        prev_flood = w.flood_risk

        # Stochastic effectiveness: ±20% noise on deltas to prevent exploitation
        noise = float(self._rng.uniform(0.8, 1.2))

        w.ndvi            = float(np.clip(w.ndvi + iv.ndvi_delta * noise, 0, 1))
        w.impervious_pct  = float(np.clip(w.impervious_pct + iv.impervious_delta * noise, 0, 100))
        w.heat_stress     = float(np.clip(
            w.heat_stress + iv.heat_delta * noise - abs(iv.lst_delta) * 2 * noise, 0, 100
        ))
        w.flood_risk      = float(np.clip(w.flood_risk + iv.flood_delta * noise, 0, 100))
        w.active_intervention = interv_idx
        w.treatment_steps_left = 5

        # Track repeat treatment
        n_treated = self._treated.get(w.ward_id, 0)
        self._treated[w.ward_id] = n_treated + 1

        # Reward: direct reduction + cost penalty + repeat penalty
        heat_red  = max(0, prev_heat  - w.heat_stress)
        flood_red = max(0, prev_flood - w.flood_risk)
        reward = heat_red * 1.0 + flood_red * 1.0

        # Cost penalty (mild — prevents cost-insensitive exploitation)
        reward -= iv.cost * 0.001

        # Repeat-treatment exponential penalty (escalates with repeats)
        if n_treated > 0:
            reward -= 0.5 * (n_treated ** 1.5)

        # Low-impact action penalty: discourage negligible interventions
        total_reduction = heat_red + flood_red
        if total_reduction < 1.0:
            reward -= 0.3

        return reward

    def _apply_climate_shock(self) -> None:
        """Random heat wave or rainfall event affecting a random cluster of wards."""
        shock_type = self._rng.choice(["heat_wave", "heavy_rain"])
        n_affected = int(self._rng.integers(5, 20))
        affected   = self._rng.choice(NUM_WARDS, size=n_affected, replace=False)

        for idx in affected:
            w = self._wards[idx]
            if shock_type == "heat_wave":
                w.heat_stress = float(np.clip(w.heat_stress + self._rng.uniform(3, 12), 0, 100))
            else:
                w.rainfall_mm += float(self._rng.uniform(5, 25))
                w.flood_risk   = float(np.clip(
                    w.flood_risk + self._rng.uniform(5, 20), 0, 100
                ))

    def _step_decay(self) -> None:
        """Untreated wards slowly drift toward higher stress; treated effects decay."""
        for w in self._wards:
            if w.treatment_steps_left > 0:
                w.treatment_steps_left -= 1
                if w.treatment_steps_left == 0:
                    w.active_intervention = -1
            else:
                # Slow worsening pressure from urbanisation
                w.heat_stress = float(np.clip(
                    w.heat_stress + self._rng.uniform(-0.2, 0.5), 0, 100
                ))

    def _city_score(self) -> float:
        """Aggregate city stress score (lower is better)."""
        return float(
            self._w_heat * np.mean([w.heat_stress for w in self._wards])
            + self._w_flood * np.mean([w.flood_risk for w in self._wards])
        )

    def _terminal_bonus(self) -> float:
        """Bonus at episode end for net city-wide improvement."""
        avg_heat  = np.mean([w.heat_stress  for w in self._wards])
        avg_flood = np.mean([w.flood_risk   for w in self._wards])
        # Reward inversely proportional to remaining average stress
        bonus = self._w_heat * max(0, 50 - avg_heat) * 0.1 \
              + self._w_flood * max(0, 40 - avg_flood) * 0.1
        return float(bonus)

    # ------------------------------------------------------------------
    # Helpers for external callers
    # ------------------------------------------------------------------
    def get_ward_scores(self) -> np.ndarray:
        """Return (N, 2) float32 array of [heat_stress, flood_risk] per ward."""
        return np.array(
            [[w.heat_stress, w.flood_risk] for w in self._wards],
            dtype=np.float32,
        )

    def set_gnn_state(self, heat: np.ndarray, flood: np.ndarray) -> None:
        """
        Override ward heat and flood scores from GNN predictions.
        Called before each episode to ground the RL simulation in real data.

        Args:
            heat:  (N,) float32 0–100
            flood: (N,) float32 0–100
        """
        for i, w in enumerate(self._wards):
            w.heat_stress = float(np.clip(heat[i],  0, 100))
            w.flood_risk  = float(np.clip(flood[i], 0, 100))

    @property
    def obs_dim(self) -> int:
        return NUM_WARDS * OBS_WARD_DIM

    @property
    def act_dim(self) -> int:
        return NUM_WARDS * N_INTERVENTIONS


# ---------------------------------------------------------------------------
# __main__ — env smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from loguru import logger

    env = BengaluruClimaEnv(seed=0)
    obs, info = env.reset()
    assert obs.shape == (NUM_WARDS * OBS_WARD_DIM,)
    logger.info(f"obs shape: {obs.shape}  budget: {info['budget']}")

    total_reward = 0.0
    for step in range(EPISODE_LEN):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        if done:
            break

    scores = env.get_ward_scores()
    logger.info(
        f"Episode done | steps={step+1}  total_reward={total_reward:.2f}  "
        f"final_heat μ={scores[:,0].mean():.1f}  flood μ={scores[:,1].mean():.1f}"
    )
    assert scores.shape == (NUM_WARDS, 2)
    logger.success("environment.py smoke test passed.")
