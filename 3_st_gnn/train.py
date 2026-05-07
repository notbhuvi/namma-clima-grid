"""
Module 3d — ST-GNN Training Script
=====================================

Trains the SpatioTemporalGNN on synthetic (or real) ward time-series data
with the pre-built ward adjacency graph.

Training setup
──────────────
  Dual-task regression:
    Loss = λ_heat × MSE(heat) + λ_flood × MSE(flood)
    Default λ_heat=0.6, λ_flood=0.4  (heat is primary objective)

  Optimizer  : AdamW (lr=5e-4, weight_decay=1e-4)
  Scheduler  : CosineAnnealingLR
  Epochs     : 60
  Early stop : patience=12 on val combined-RMSE

Metrics logged per epoch
─────────────────────────
  train_loss, val_loss,
  val_heat_rmse, val_heat_mae, val_heat_r2,
  val_flood_rmse, val_flood_mae, val_flood_r2

Artifacts
──────────
  checkpoints/gnn_best.pt  — best combined-val-RMSE checkpoint
  checkpoints/gnn_last.pt  — final epoch checkpoint

Usage
─────
  python 3_st_gnn/train.py
  python 3_st_gnn/train.py --parquet data/ward_features_history.parquet
  python 3_st_gnn/train.py --epochs 1 --no-mlflow   # quick smoke test
  python 3_st_gnn/train.py --use-vit                 # fuse CNN-ViT embeddings
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict

_MODULE_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _MODULE_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "8_data"))

from graph_builder import build_ward_graph
from model         import SpatioTemporalGNN, NODE_FEAT_DIM, HIDDEN_DIM, T_WINDOW
from dataset       import make_dataloaders, FEATURE_COLS, NUM_WARDS


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
class TrainSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"), extra="ignore"
    )
    mlflow_tracking_uri: str = "http://localhost:5001"
    mlflow_experiment:   str = "st_gnn_heat_flood"

    epochs:       int   = 100
    batch_size:   int   = 16
    lr:           float = 5e-4
    weight_decay: float = 1e-4
    patience:     int   = 15
    lambda_heat:  float = 0.6
    lambda_flood: float = 0.4
    seed:         int   = 42
    n_snapshots:  int   = 400   # synthetic history length
    checkpoint_dir: str = "checkpoints"


_cfg = TrainSettings()


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def dual_loss(
    pred_heat: torch.Tensor, pred_flood: torch.Tensor,
    tgt_heat: torch.Tensor,  tgt_flood: torch.Tensor,
    lambda_heat: float = _cfg.lambda_heat,
    lambda_flood: float = _cfg.lambda_flood,
) -> torch.Tensor:
    """Weighted sum of MSE losses for heat and flood targets."""
    loss_h = nn.functional.mse_loss(pred_heat,  tgt_heat)
    loss_f = nn.functional.mse_loss(pred_flood, tgt_flood)
    return lambda_heat * loss_h + lambda_flood * loss_f


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _metrics(preds: np.ndarray, targets: np.ndarray, n_wards: int = NUM_WARDS) -> Dict:
    res = preds - targets
    rmse = float(np.sqrt(np.mean(res ** 2)))
    mae  = float(np.mean(np.abs(res)))
    ss_r = np.sum(res ** 2)
    ss_t = np.sum((targets - targets.mean()) ** 2)
    r2   = float(1.0 - ss_r / (ss_t + 1e-8))
    # Per-node RMSE: reshape to (n_samples, n_wards) and compute per-ward
    n_samples = len(preds) // n_wards
    per_node_rmse = None
    if n_samples > 0 and len(preds) == n_samples * n_wards:
        p2d = preds.reshape(n_samples, n_wards)
        t2d = targets.reshape(n_samples, n_wards)
        per_node_rmse = np.sqrt(np.mean((p2d - t2d) ** 2, axis=0))  # (n_wards,)
    return {"rmse": rmse, "mae": mae, "r2": r2, "per_node_rmse": per_node_rmse}


# ---------------------------------------------------------------------------
# Epoch helpers
# ---------------------------------------------------------------------------

def _flatten(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().reshape(-1)


def _train_epoch(
    model: SpatioTemporalGNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_samples  = 0

    ei = edge_index.to(device)
    ew = edge_weight.to(device)

    for x_seq, y_heat, y_flood, _ in loader:
        # x_seq : (B, N, T, F) — batch dim from DataLoader
        # squeeze batch: process each sample individually through graph
        B = x_seq.size(0)
        batch_loss = 0.0

        optimizer.zero_grad()
        for b in range(B):
            xb = x_seq[b].to(device)        # (N, T, F)
            yh = y_heat[b].to(device)        # (N,)
            yf = y_flood[b].to(device)       # (N,)

            out  = model(xb, ei, ew)
            loss = dual_loss(out["heat_stress"], out["flood_risk"], yh, yf)
            batch_loss += loss

        batch_loss = batch_loss / B
        batch_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += batch_loss.item() * B
        n_samples  += B

    return total_loss / max(n_samples, 1)


@torch.no_grad()
def _val_epoch(
    model: SpatioTemporalGNN,
    loader: DataLoader,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    device: torch.device,
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    n_samples  = 0
    ph_all, pf_all, yh_all, yf_all = [], [], [], []

    ei = edge_index.to(device)
    ew = edge_weight.to(device)

    for x_seq, y_heat, y_flood, _ in loader:
        B = x_seq.size(0)
        b_loss = 0.0
        for b in range(B):
            xb = x_seq[b].to(device)
            yh = y_heat[b].to(device)
            yf = y_flood[b].to(device)

            out  = model(xb, ei, ew)
            loss = dual_loss(out["heat_stress"], out["flood_risk"], yh, yf)
            b_loss += loss.item()

            ph_all.append(_flatten(out["heat_stress"]))
            pf_all.append(_flatten(out["flood_risk"]))
            yh_all.append(_flatten(yh))
            yf_all.append(_flatten(yf))

        total_loss += (b_loss / B) * B
        n_samples  += B

    ph = np.concatenate(ph_all)
    pf = np.concatenate(pf_all)
    yh = np.concatenate(yh_all)
    yf = np.concatenate(yf_all)

    return (
        total_loss / max(n_samples, 1),
        _metrics(ph, yh),
        _metrics(pf, yf),
    )


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save(model: nn.Module, path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), **meta}, path)


def load_checkpoint(model: nn.Module, path: str | Path, device=None) -> dict:
    ckpt = torch.load(path, map_location=device or "cpu")
    model.load_state_dict(ckpt["state_dict"])
    return {k: v for k, v in ckpt.items() if k != "state_dict"}


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    parquet_path: Optional[str]   = None,
    epochs: int                   = _cfg.epochs,
    batch_size: int               = _cfg.batch_size,
    lr: float                     = _cfg.lr,
    weight_decay: float           = _cfg.weight_decay,
    patience: int                 = _cfg.patience,
    lambda_heat: float            = _cfg.lambda_heat,
    lambda_flood: float           = _cfg.lambda_flood,
    seed: int                     = _cfg.seed,
    n_snapshots: int              = _cfg.n_snapshots,
    checkpoint_dir: str           = _cfg.checkpoint_dir,
    use_mlflow: bool              = True,
    use_vit_embeddings: bool      = False,
) -> Path:
    """Run the full training loop. Returns path of best checkpoint."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training ST-GNN on {device}")

    # ── Graph ─────────────────────────────────────────────────────────
    graph = build_ward_graph(
        cache_path=Path(checkpoint_dir) / "ward_graph.pt"
    )
    edge_index  = graph["edge_index"]
    edge_weight = graph["edge_weight"]
    logger.info(f"Graph | nodes={graph['num_nodes']}  edges={graph['num_edges']}")

    # ── Data ──────────────────────────────────────────────────────────
    train_loader, val_loader = make_dataloaders(
        parquet_path=parquet_path,
        n_snapshots=n_snapshots,
        batch_size=batch_size,
        seed=seed,
    )

    # ── Model ─────────────────────────────────────────────────────────
    model = SpatioTemporalGNN(
        node_feat_dim=len(FEATURE_COLS),
        use_cnn_vit_embedding=use_vit_embeddings,
    ).to(device)
    params = model.parameter_count()
    logger.info(f"Model params: {params['total']:,} total")

    # ── Optimiser + scheduler ──────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=lr * 0.01
    )

    # ── MLflow ────────────────────────────────────────────────────────
    mlflow_active = False
    if use_mlflow:
        try:
            import mlflow  # type: ignore
            mlflow.set_tracking_uri(_cfg.mlflow_tracking_uri)
            mlflow.set_experiment(_cfg.mlflow_experiment)
            mlflow.start_run()
            mlflow.log_params({
                "epochs": epochs, "batch_size": batch_size, "lr": lr,
                "lambda_heat": lambda_heat, "lambda_flood": lambda_flood,
                "n_snapshots": n_snapshots, "t_window": T_WINDOW,
                "node_feat_dim": len(FEATURE_COLS),
                "use_vit": use_vit_embeddings, "total_params": params["total"],
            })
            mlflow_active = True
        except Exception as exc:
            logger.warning(f"MLflow unavailable ({exc})")

    # ── Training loop ─────────────────────────────────────────────────
    ckpt_dir   = Path(checkpoint_dir)
    best_ckpt  = ckpt_dir / "gnn_best.pt"
    best_rmse  = float("inf")
    no_improve = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss = _train_epoch(
            model, train_loader, optimizer, edge_index, edge_weight, device
        )
        val_loss, heat_m, flood_m = _val_epoch(
            model, val_loader, edge_index, edge_weight, device
        )
        scheduler.step(val_loss)
        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]
        combined_rmse = 0.6 * heat_m["rmse"] + 0.4 * flood_m["rmse"]

        logger.info(
            f"Ep {epoch:3d}/{epochs} | "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"heat_RMSE={heat_m['rmse']:.2f}  flood_RMSE={flood_m['rmse']:.2f}  "
            f"heat_R²={heat_m['r2']:.3f}  flood_R²={flood_m['r2']:.3f}  "
            f"lr={lr_now:.2e}  [{elapsed:.1f}s]"
        )
        # Per-node error distribution (log every 10 epochs)
        if epoch % 10 == 0:
            h_pn = heat_m.get("per_node_rmse")
            f_pn = flood_m.get("per_node_rmse")
            if h_pn is not None:
                logger.info(
                    f"  node-heat RMSE | p10={np.percentile(h_pn,10):.2f} "
                    f"p50={np.percentile(h_pn,50):.2f} p90={np.percentile(h_pn,90):.2f}"
                )
            if f_pn is not None:
                logger.info(
                    f"  node-flood RMSE | p10={np.percentile(f_pn,10):.2f} "
                    f"p50={np.percentile(f_pn,50):.2f} p90={np.percentile(f_pn,90):.2f}"
                )

        if mlflow_active:
            try:
                import mlflow
                combined_mae = 0.6 * heat_m["mae"] + 0.4 * flood_m["mae"]
                combined_r2  = 0.6 * heat_m["r2"]  + 0.4 * flood_m["r2"]
                mlflow.log_metrics({
                    "train_loss":     train_loss,
                    "val_loss":       val_loss,
                    "val_rmse":       combined_rmse,
                    "val_mae":        combined_mae,
                    "val_r2":         combined_r2,
                    "current_lr":     lr_now,
                    "val_heat_rmse":  heat_m["rmse"],
                    "val_heat_r2":    heat_m["r2"],
                    "val_flood_rmse": flood_m["rmse"],
                    "val_flood_r2":   flood_m["r2"],
                }, step=epoch)
            except Exception:
                pass

        if combined_rmse < best_rmse:
            best_rmse  = combined_rmse
            no_improve = 0
            _save(model, best_ckpt, {
                "epoch": epoch, "val_combined_rmse": best_rmse,
                "val_heat_rmse": heat_m["rmse"], "val_flood_rmse": flood_m["rmse"],
            })
            logger.success(f"  ↳ best combined_RMSE={best_rmse:.3f} — saved")
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.warning(f"Early stopping at epoch {epoch}")
                break

    # Final checkpoint
    last_ckpt = ckpt_dir / "gnn_last.pt"
    _save(model, last_ckpt, {"epoch": epoch, "val_combined_rmse": combined_rmse})

    if mlflow_active:
        try:
            import mlflow
            mlflow.log_metric("best_val_combined_rmse", best_rmse)
            mlflow.log_artifact(str(best_ckpt))
            # Register model in MLflow Model Registry
            try:
                active_run = mlflow.active_run()
                if active_run:
                    model_uri = f"runs:/{active_run.info.run_id}/{best_ckpt.name}"
                    mlflow.register_model(model_uri, "stgnn_model")
                    logger.info("Registered model as 'stgnn_model' in MLflow Registry")
            except Exception as reg_exc:
                logger.warning(f"ST-GNN model registration failed (non-critical): {reg_exc}")
            mlflow.end_run()
        except Exception:
            pass

    logger.success(f"Training done | best_combined_RMSE={best_rmse:.3f} | {best_ckpt}")
    return best_ckpt


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NammaClimaGrid — Train ST-GNN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--parquet",       type=str,   default=None)
    p.add_argument("--epochs",        type=int,   default=_cfg.epochs)
    p.add_argument("--batch-size",    type=int,   default=_cfg.batch_size)
    p.add_argument("--lr",            type=float, default=_cfg.lr)
    p.add_argument("--patience",      type=int,   default=_cfg.patience)
    p.add_argument("--n-snapshots",   type=int,   default=_cfg.n_snapshots)
    p.add_argument("--seed",          type=int,   default=_cfg.seed)
    p.add_argument("--checkpoint-dir",type=str,   default=_cfg.checkpoint_dir)
    p.add_argument("--no-mlflow",     action="store_true")
    p.add_argument("--use-vit",       action="store_true",
                   help="Fuse CNN-ViT spatial embeddings (requires Module 2 inference)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    best = train(
        parquet_path       = args.parquet,
        epochs             = args.epochs,
        batch_size         = args.batch_size,
        lr                 = args.lr,
        patience           = args.patience,
        n_snapshots        = args.n_snapshots,
        seed               = args.seed,
        checkpoint_dir     = args.checkpoint_dir,
        use_mlflow         = not args.no_mlflow,
        use_vit_embeddings = args.use_vit,
    )
    logger.success(f"Best model: {best}")
