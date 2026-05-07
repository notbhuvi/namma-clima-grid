"""
Module 4c — PPO Training Script
==================================

Trains the ActorCritic policy on BengaluruClimaEnv using Proximal Policy
Optimization (PPO-clip).

PPO hyperparameters
────────────────────
  rollout_steps   : 256   (steps collected per update)
  n_epochs        : 4     (PPO epochs per update)
  mini_batch_size : 64
  clip_eps        : 0.2
  entropy_coef    : 0.01  (encourages exploration)
  vf_coef         : 0.5
  gamma           : 0.99
  gae_lambda      : 0.95
  max_grad_norm   : 0.5
  total_timesteps : 500_000

MLflow metrics (per update)
─────────────────────────────
  ep_reward_mean, ep_len_mean,
  policy_loss, value_loss, entropy,
  approx_kl, clip_fraction

Usage
─────
  python 4_rl_optimizer/train.py
  python 4_rl_optimizer/train.py --timesteps 100000 --no-mlflow
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict

_MODULE_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _MODULE_DIR.parent

from environment import BengaluruClimaEnv, NUM_WARDS, OBS_WARD_DIM, N_INTERVENTIONS
from model       import ActorCritic, N_ACTIONS


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
class TrainSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"), extra="ignore"
    )
    mlflow_tracking_uri: str  = "http://localhost:5001"
    mlflow_experiment:   str  = "rl_ppo_optimizer"

    total_timesteps: int   = 300_000
    rollout_steps:   int   = 256
    n_ppo_epochs:    int   = 4
    mini_batch_size: int   = 64
    clip_eps:        float = 0.2
    entropy_coef:    float = 0.01
    vf_coef:         float = 0.5
    gamma:           float = 0.99
    gae_lambda:      float = 0.95
    max_grad_norm:   float = 0.5
    lr:              float = 3e-4
    seed:            int   = 42
    budget:          int   = 60
    checkpoint_dir:  str   = "checkpoints"


_cfg = TrainSettings()


# ---------------------------------------------------------------------------
# GAE advantage computation
# ---------------------------------------------------------------------------

def compute_gae(
    rewards:    torch.Tensor,   # (T,)
    values:     torch.Tensor,   # (T,)
    dones:      torch.Tensor,   # (T,) float
    last_value: torch.Tensor,   # scalar
    gamma:      float = _cfg.gamma,
    lam:        float = _cfg.gae_lambda,
) -> torch.Tensor:
    """Return GAE advantages (T,)."""
    T = len(rewards)
    advantages = torch.zeros(T, dtype=torch.float32)
    last_gae   = 0.0

    for t in reversed(range(T)):
        next_val  = last_value if t == T - 1 else values[t + 1]
        delta     = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        last_gae  = float(delta) + gamma * lam * (1 - float(dones[t])) * last_gae
        advantages[t] = last_gae

    return advantages


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

def ppo_update(
    model:      ActorCritic,
    optimizer:  torch.optim.Optimizer,
    obs_buf:    torch.Tensor,      # (T, obs_dim)
    act_buf:    torch.Tensor,      # (T,)
    logp_buf:   torch.Tensor,      # (T,)
    adv_buf:    torch.Tensor,      # (T,)
    ret_buf:    torch.Tensor,      # (T,) = advantages + values
    clip_eps:   float,
    vf_coef:    float,
    ent_coef:   float,
    n_epochs:   int,
    mb_size:    int,
    max_grad_norm: float,
) -> dict:
    T = obs_buf.shape[0]
    metrics = {k: [] for k in ["policy_loss", "value_loss", "entropy",
                                "approx_kl", "clip_frac"]}

    for _ in range(n_epochs):
        idxs = torch.randperm(T)
        for start in range(0, T, mb_size):
            mb = idxs[start : start + mb_size]
            out = model.get_action_and_value(obs_buf[mb], action=act_buf[mb])

            log_ratio  = out["log_prob"] - logp_buf[mb]
            ratio      = log_ratio.exp()
            adv_mb     = adv_buf[mb]
            adv_mb     = (adv_mb - adv_mb.mean()) / (adv_mb.std() + 1e-8)

            # Policy loss (PPO-clip)
            surr1 = -ratio * adv_mb
            surr2 = -ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv_mb
            pol_loss = torch.max(surr1, surr2).mean()

            # Value loss
            val_loss = 0.5 * (out["value"] - ret_buf[mb]).pow(2).mean()

            # Entropy bonus
            entropy  = out["entropy"].mean()

            loss = pol_loss + vf_coef * val_loss - ent_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                kl    = ((ratio - 1) - log_ratio).mean().item()
                c_frac = ((ratio - 1).abs() > clip_eps).float().mean().item()

            metrics["policy_loss"].append(pol_loss.item())
            metrics["value_loss"].append(val_loss.item())
            metrics["entropy"].append(entropy.item())
            metrics["approx_kl"].append(kl)
            metrics["clip_frac"].append(c_frac)

    return {k: float(np.mean(v)) for k, v in metrics.items()}


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save(model: nn.Module, path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), **meta}, path)


def load_checkpoint(model: nn.Module, path: str | Path) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    return {k: v for k, v in ckpt.items() if k != "state_dict"}


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    total_timesteps: int  = _cfg.total_timesteps,
    rollout_steps:   int  = _cfg.rollout_steps,
    n_ppo_epochs:    int  = _cfg.n_ppo_epochs,
    mini_batch_size: int  = _cfg.mini_batch_size,
    clip_eps:        float = _cfg.clip_eps,
    entropy_coef:    float = _cfg.entropy_coef,
    vf_coef:         float = _cfg.vf_coef,
    gamma:           float = _cfg.gamma,
    gae_lambda:      float = _cfg.gae_lambda,
    max_grad_norm:   float = _cfg.max_grad_norm,
    lr:              float = _cfg.lr,
    seed:            int   = _cfg.seed,
    budget:          int   = _cfg.budget,
    checkpoint_dir:  str   = _cfg.checkpoint_dir,
    use_mlflow:      bool  = True,
) -> Path:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"PPO training on {device}")

    env   = BengaluruClimaEnv(budget=budget, seed=seed)
    model = ActorCritic().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)

    params = model.parameter_count()
    logger.info(f"ActorCritic params: {params['total']:,}")

    # ── MLflow ────────────────────────────────────────────────────────
    mlflow_active = False
    if use_mlflow:
        try:
            import mlflow  # type: ignore
            mlflow.set_tracking_uri(_cfg.mlflow_tracking_uri)
            mlflow.set_experiment(_cfg.mlflow_experiment)
            mlflow.start_run()
            mlflow.log_params({
                "total_timesteps": total_timesteps, "rollout_steps": rollout_steps,
                "clip_eps": clip_eps, "lr": lr, "gamma": gamma,
                "gae_lambda": gae_lambda, "budget": budget,
                "total_params": params["total"],
            })
            mlflow_active = True
        except Exception as exc:
            logger.warning(f"MLflow unavailable ({exc})")

    # ── Buffers ────────────────────────────────────────────────────────
    obs_dim = env.obs_dim
    obs_buf  = torch.zeros(rollout_steps, obs_dim,  dtype=torch.float32)
    act_buf  = torch.zeros(rollout_steps,            dtype=torch.long)
    rew_buf  = torch.zeros(rollout_steps,            dtype=torch.float32)
    done_buf = torch.zeros(rollout_steps,            dtype=torch.float32)
    val_buf  = torch.zeros(rollout_steps,            dtype=torch.float32)
    logp_buf = torch.zeros(rollout_steps,            dtype=torch.float32)

    # ── Training state ─────────────────────────────────────────────────
    obs_np, info = env.reset(seed=seed)
    obs          = torch.from_numpy(obs_np)
    ep_rewards   = deque(maxlen=20)
    ep_lens      = deque(maxlen=20)
    ep_diversity = deque(maxlen=20)   # unique interventions per episode
    ep_rew_cur   = 0.0
    ep_len_cur   = 0
    ep_actions_cur: set = set()       # track (ward, interv) pairs in episode
    global_step  = 0
    update_idx   = 0
    best_ep_rew  = float("-inf")
    best_ckpt    = Path(checkpoint_dir) / "ppo_best.pt"
    t0_total     = time.time()
    reward_acc_10k = 0.0   # accumulator for avg reward per 10k steps
    steps_in_10k   = 0
    last_10k_log   = 0

    while global_step < total_timesteps:
        t0 = time.time()

        # ── Rollout collection ────────────────────────────────────────
        for step in range(rollout_steps):
            with torch.no_grad():
                out = model.get_action_and_value(obs.to(device))

            action  = int(out["action"].item())
            log_prob = out["log_prob"].item()
            value    = out["value"].item()

            obs_buf[step]  = obs
            act_buf[step]  = action
            logp_buf[step] = log_prob
            val_buf[step]  = value

            obs_np, reward, done, truncated, info = env.step(action)
            rew_buf[step]  = reward
            done_buf[step] = float(done or truncated)

            obs = torch.from_numpy(obs_np)
            ep_rew_cur += reward
            ep_len_cur += 1
            ep_actions_cur.add((action // N_INTERVENTIONS, action % N_INTERVENTIONS))
            global_step += 1
            reward_acc_10k += reward
            steps_in_10k += 1

            if done or truncated:
                ep_rewards.append(ep_rew_cur)
                ep_lens.append(ep_len_cur)
                ep_diversity.append(len(ep_actions_cur))
                ep_rew_cur = 0.0
                ep_len_cur = 0
                ep_actions_cur = set()
                obs_np, _ = env.reset()
                obs        = torch.from_numpy(obs_np)

        # ── Compute advantages (GAE) ───────────────────────────────────
        with torch.no_grad():
            _, last_val = model(obs.unsqueeze(0).to(device))
            last_val = last_val.squeeze().item()

        adv_buf = compute_gae(rew_buf, val_buf,
                              done_buf, torch.tensor(last_val),
                              gamma, gae_lambda)
        ret_buf = adv_buf + val_buf

        # ── PPO update ────────────────────────────────────────────────
        update_metrics = ppo_update(
            model, opt,
            obs_buf.to(device), act_buf.to(device),
            logp_buf.to(device), adv_buf.to(device), ret_buf.to(device),
            clip_eps, vf_coef, entropy_coef, n_ppo_epochs, mini_batch_size, max_grad_norm,
        )
        update_idx += 1

        ep_rew_mean  = float(np.mean(ep_rewards))   if ep_rewards   else 0.0
        ep_len_mean  = float(np.mean(ep_lens))     if ep_lens      else 0.0
        ep_div_mean  = float(np.mean(ep_diversity)) if ep_diversity else 0.0
        elapsed      = time.time() - t0
        sps          = rollout_steps / elapsed

        logger.info(
            f"Update {update_idx:4d} | steps={global_step:7d}  "
            f"ep_rew={ep_rew_mean:.2f}  ep_len={ep_len_mean:.0f}  "
            f"diversity={ep_div_mean:.1f}  "
            f"pol_loss={update_metrics['policy_loss']:.4f}  "
            f"val_loss={update_metrics['value_loss']:.4f}  "
            f"entropy={update_metrics['entropy']:.3f}  "
            f"kl={update_metrics['approx_kl']:.4f}  "
            f"sps={sps:.0f}"
        )

        if mlflow_active:
            try:
                import mlflow
                mlflow.log_metrics({
                    "ep_reward_mean": ep_rew_mean,
                    "ep_len_mean":    ep_len_mean,
                    **update_metrics,
                }, step=global_step)
            except Exception:
                pass

        # Log avg reward per 10k steps
        if global_step - last_10k_log >= 10_000 and steps_in_10k > 0:
            avg_rew_10k = reward_acc_10k / steps_in_10k
            logger.info(
                f"  >> avg_reward/10k_steps={avg_rew_10k:.3f} "
                f"(steps {last_10k_log}-{global_step})"
            )
            if mlflow_active:
                try:
                    import mlflow
                    mlflow.log_metric("avg_reward_10k", avg_rew_10k, step=global_step)
                except Exception:
                    pass
            reward_acc_10k = 0.0
            steps_in_10k = 0
            last_10k_log = global_step

        if ep_rew_mean > best_ep_rew and len(ep_rewards) >= 5:
            best_ep_rew = ep_rew_mean
            _save(model, best_ckpt, {
                "global_step": global_step, "ep_reward_mean": best_ep_rew
            })
            logger.success(f"  ↳ best ep_reward={best_ep_rew:.2f} — saved")

    # ── Final checkpoint ──────────────────────────────────────────────
    last_ckpt = Path(checkpoint_dir) / "ppo_last.pt"
    _save(model, last_ckpt, {"global_step": global_step, "ep_reward_mean": ep_rew_mean})
    total_time = time.time() - t0_total

    if mlflow_active:
        try:
            import mlflow
            mlflow.log_metric("best_ep_reward", best_ep_rew)
            mlflow.log_artifact(str(best_ckpt))
            # Register model in MLflow Model Registry
            try:
                active_run = mlflow.active_run()
                if active_run:
                    model_uri = f"runs:/{active_run.info.run_id}/{best_ckpt.name}"
                    mlflow.register_model(model_uri, "rl_model")
                    logger.info("Registered model as 'rl_model' in MLflow Registry")
            except Exception as reg_exc:
                logger.warning(f"RL model registration failed (non-critical): {reg_exc}")
            mlflow.end_run()
        except Exception:
            pass

    logger.success(
        f"Training done | {global_step} steps in {total_time:.0f}s | "
        f"best_ep_reward={best_ep_rew:.2f} | {best_ckpt}"
    )
    return best_ckpt


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NammaClimaGrid — Train RL Green Infra Optimizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--timesteps",   type=int,   default=_cfg.total_timesteps)
    p.add_argument("--lr",          type=float, default=_cfg.lr)
    p.add_argument("--rollout",     type=int,   default=_cfg.rollout_steps)
    p.add_argument("--budget",      type=int,   default=_cfg.budget)
    p.add_argument("--seed",        type=int,   default=_cfg.seed)
    p.add_argument("--checkpoint-dir", type=str, default=_cfg.checkpoint_dir)
    p.add_argument("--no-mlflow",   action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args  = _parse()
    best  = train(
        total_timesteps = args.timesteps,
        lr              = args.lr,
        rollout_steps   = args.rollout,
        budget          = args.budget,
        seed            = args.seed,
        checkpoint_dir  = args.checkpoint_dir,
        use_mlflow      = not args.no_mlflow,
    )
    logger.success(f"Best policy: {best}")
