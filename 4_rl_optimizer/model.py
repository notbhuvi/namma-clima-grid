"""
Module 4b — PPO Actor-Critic Network
=======================================

Architecture
─────────────
  Observation: (N_wards × OBS_WARD_DIM,) = (198 × 13,) = 2574-dim flat vector

  Shared Encoder
  ──────────────
  The observation is first reshaped to (N, D) and passed through a
  per-ward feature extractor (shared MLP), then max-pooled to a global
  city summary, then concatenated back to each ward's features.

    Per-ward MLP : Linear(D→128) → LayerNorm → GELU  → (N, 128)
    Global pool  : max over N                          → (128,)
    Context cat  : cat([ward_feat, global.expand(N)])  → (N, 256)
    Context MLP  : Linear(256→128) → GELU              → (N, 128)
    Flatten      : (N × 128,) = 25344-dim

  Actor Head  (policy)
  ─────────────────────
    FC(25344→1024) → GELU → FC(1024→256) → GELU
    → FC(256→N_acts=990)   — logits (Categorical distribution)

  Critic Head  (value function)
  ──────────────────────────────
    FC(25344→1024) → GELU → FC(1024→256) → GELU
    → FC(256→1)            — scalar state value V(s)

Outputs
────────
  actor_logits : (batch, N_acts)
  value        : (batch, 1)
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from environment import (
    NUM_WARDS, N_INTERVENTIONS, OBS_WARD_DIM,
    INTERVENTIONS,
)

N_ACTIONS   = NUM_WARDS * N_INTERVENTIONS   # 990
WARD_ENC    = 128
CONTEXT_DIM = WARD_ENC * 2                 # 256 (ward + global)
FLAT_DIM    = NUM_WARDS * WARD_ENC         # 25344


# ---------------------------------------------------------------------------
# Shared ward encoder
# ---------------------------------------------------------------------------

class WardEncoder(nn.Module):
    """
    Encodes each ward's local observation + global city context into a
    fixed-dim representation.  Equivariant to ward ordering.
    """

    def __init__(self, ward_dim: int = OBS_WARD_DIM, enc_dim: int = WARD_ENC) -> None:
        super().__init__()
        self.local_mlp = nn.Sequential(
            nn.Linear(ward_dim, enc_dim),
            nn.LayerNorm(enc_dim),
            nn.GELU(),
        )
        self.context_mlp = nn.Sequential(
            nn.Linear(enc_dim * 2, enc_dim),
            nn.GELU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (B, N, D) or (N, D) ward observations.
        Returns:
            (B, N*enc_dim) or (N*enc_dim,) flattened ward encodings.
        """
        squeeze = obs.dim() == 2
        if squeeze:
            obs = obs.unsqueeze(0)

        B, N, D = obs.shape
        local   = self.local_mlp(obs)                             # (B, N, enc_dim)
        global_ = local.max(dim=1).values.unsqueeze(1).expand_as(local)  # (B, N, enc_dim)
        context = torch.cat([local, global_], dim=-1)             # (B, N, 2*enc_dim)
        enc     = self.context_mlp(context)                       # (B, N, enc_dim)
        flat    = enc.reshape(B, -1)                              # (B, N*enc_dim)

        return flat.squeeze(0) if squeeze else flat


# ---------------------------------------------------------------------------
# Actor-Critic
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    """
    PPO Actor-Critic for BengaluruClimaEnv.

    Args:
        ward_dim:   Per-ward observation dimension (default OBS_WARD_DIM=13).
        enc_dim:    Ward encoder output dimension (default 128).
        n_wards:    Number of wards (default 198).
        n_interv:   Number of intervention types (default 5).
    """

    def __init__(
        self,
        ward_dim:  int = OBS_WARD_DIM,
        enc_dim:   int = WARD_ENC,
        n_wards:   int = NUM_WARDS,
        n_interv:  int = N_INTERVENTIONS,
    ) -> None:
        super().__init__()
        self.n_wards   = n_wards
        self.n_interv  = n_interv
        self.n_actions = n_wards * n_interv
        flat_dim       = n_wards * enc_dim

        self.encoder = WardEncoder(ward_dim, enc_dim)

        def _head(out_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(flat_dim, 1024), nn.GELU(),
                nn.Linear(1024, 256),      nn.GELU(),
                nn.Linear(256, out_dim),
            )

        self.actor  = _head(self.n_actions)
        self.critic = _head(1)

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01
                                    if m.out_features == self.n_actions else 1.0)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def _reshape_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Reshape flat obs to (*, N, D)."""
        if obs.dim() == 1:
            return obs.reshape(self.n_wards, -1)
        return obs.reshape(obs.shape[0], self.n_wards, -1)

    # ------------------------------------------------------------------
    def forward(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: (B, N*D) or (N*D,) flat observation.
        Returns:
            actor_logits : (B, N_actions)
            value        : (B, 1)
        """
        x            = self._reshape_obs(obs)
        enc          = self.encoder(x)
        actor_logits = self.actor(enc)
        value        = self.critic(enc)
        return actor_logits, value

    # ------------------------------------------------------------------
    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
        budget_mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Sample action and compute log-prob + entropy + value.

        Args:
            obs:         (B, obs_dim) or (obs_dim,) observation.
            action:      If given, compute log-prob for this specific action
                         (used during PPO update for old-policy evaluation).
            budget_mask: (B, N_actions) bool tensor; True = action infeasible
                         (e.g. cost > remaining budget).  Masked logits → -∞.

        Returns:
            dict with keys: action, log_prob, entropy, value
        """
        logits, value = self.forward(obs)

        if budget_mask is not None:
            logits = logits.masked_fill(budget_mask, float("-inf"))

        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        return {
            "action":   action,
            "log_prob": dist.log_prob(action),
            "entropy":  dist.entropy(),
            "value":    value.squeeze(-1),
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        budget_left: int,
        deterministic: bool = False,
    ) -> int:
        """
        Single-step inference.  Returns an integer action.

        Args:
            obs:          (obs_dim,) flat observation tensor.
            budget_left:  Remaining budget — used to mask out unaffordable actions.
            deterministic: If True, take argmax instead of sampling.
        """
        logits, _ = self.forward(obs.unsqueeze(0))
        logits     = logits.squeeze(0)

        # Mask interventions the agent can't afford
        cost_vec = torch.tensor(
            [iv.cost for iv in INTERVENTIONS] * self.n_wards,
            dtype=torch.float32, device=logits.device,
        )
        mask = cost_vec > budget_left
        logits = logits.masked_fill(mask, float("-inf"))

        if deterministic:
            return int(logits.argmax().item())
        return int(Categorical(logits=logits).sample().item())

    # ------------------------------------------------------------------
    def parameter_count(self) -> Dict[str, int]:
        def _n(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters() if p.requires_grad)
        return {
            "encoder": _n(self.encoder),
            "actor":   _n(self.actor),
            "critic":  _n(self.critic),
            "total":   _n(self),
        }


# ---------------------------------------------------------------------------
# __main__ — architecture smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from loguru import logger

    model  = ActorCritic()
    params = model.parameter_count()
    logger.info(
        f"ActorCritic params | encoder={params['encoder']:,}  "
        f"actor={params['actor']:,}  critic={params['critic']:,}  "
        f"total={params['total']:,}"
    )

    # Batch forward
    obs     = torch.randn(4, NUM_WARDS * OBS_WARD_DIM)
    logits, value = model(obs)
    assert logits.shape == (4, N_ACTIONS), f"logits: {logits.shape}"
    assert value.shape  == (4, 1),         f"value: {value.shape}"

    # Action sampling
    out = model.get_action_and_value(obs)
    assert out["action"].shape   == (4,)
    assert out["log_prob"].shape == (4,)
    assert out["entropy"].shape  == (4,)
    assert out["value"].shape    == (4,)

    # Single-step act
    single_obs = torch.randn(NUM_WARDS * OBS_WARD_DIM)
    action = model.act(single_obs, budget_left=10)
    assert 0 <= action < N_ACTIONS, f"action out of range: {action}"

    logger.success(
        f"logits range=[{logits.min():.2f}, {logits.max():.2f}]  "
        f"value range=[{value.min():.2f}, {value.max():.2f}]"
    )
    logger.success("model.py smoke test passed.")
