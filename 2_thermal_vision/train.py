"""
Module 2c — Training Script
============================

Trains the HybridLSTPredictor on synthetic (or real) ward patches with
full MLflow experiment tracking and checkpoint management.

Training setup
──────────────
  Optimizer  : AdamW (weight_decay=1e-4)
  Scheduler  : CosineAnnealingLR (T_max = n_epochs)
  Loss       : 0.7 × MSE  +  0.3 × MAE   (robust to outliers)
  Batch size : 32
  Epochs     : 50 (early stopping with patience=10)

Metrics logged (per epoch)
───────────────────────────
  train_loss, val_loss, val_rmse, val_mae, val_r2

Artifacts saved
────────────────
  checkpoints/best_model.pt   — best val RMSE checkpoint
  checkpoints/last_model.pt   — model after final epoch
  MLflow experiment "thermal_vision_lst"

Usage
─────
  # Train on synthetic data (no real parquet needed):
  python 2_thermal_vision/train.py

  # Use a real ward_features parquet:
  python 2_thermal_vision/train.py --parquet 8_data/ward_features_mock.parquet

  # Quick overfit check (1 epoch, no MLflow):
  python 2_thermal_vision/train.py --epochs 1 --no-mlflow
"""
from __future__ import annotations

import argparse
import os
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

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
_MODULE_DIR  = Path(__file__).resolve().parent
_PROJECT_ROOT = _MODULE_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "8_data"))

from model   import HybridLSTPredictor, N_CHANNELS, PATCH_SIZE, EMBED_DIM
from dataset import make_dataloaders


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
class TrainSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"), extra="ignore"
    )

    mlflow_tracking_uri: str = "http://localhost:5001"
    mlflow_experiment:   str = "thermal_vision_cnn_vit"

    # Training hyperparams (overridable via env vars)
    epochs:     int   = 50
    batch_size: int   = 32
    lr:         float = 1e-4
    weight_decay: float = 1e-4
    val_fraction: float = 0.20
    n_aug_train: int  = 8
    patience:   int   = 5
    seed:       int   = 42

    checkpoint_dir: str = "checkpoints"


_cfg = TrainSettings()


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def combined_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """0.7 × MSE + 0.3 × MAE — balances sensitivity and robustness."""
    mse = nn.functional.mse_loss(pred, target)
    mae = nn.functional.l1_loss(pred, target)
    return 0.7 * mse + 0.3 * mae


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_metrics(
    preds: np.ndarray, targets: np.ndarray
) -> Dict[str, float]:
    residuals = preds - targets
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae  = float(np.mean(np.abs(residuals)))
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    r2   = float(1.0 - ss_res / (ss_tot + 1e-8))
    return {"rmse": rmse, "mae": mae, "r2": r2}


# ---------------------------------------------------------------------------
# One epoch helpers
# ---------------------------------------------------------------------------

def _train_epoch(
    model: HybridLSTPredictor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for patches, targets, _ in loader:
        patches = patches.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        out  = model(patches)
        loss = combined_loss(out["lst_pred"], targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(targets)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def _val_epoch(
    model: HybridLSTPredictor,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    all_preds:   list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for patches, targets, _ in loader:
        patches = patches.to(device)
        targets = targets.to(device)
        out  = model(patches)
        loss = combined_loss(out["lst_pred"], targets)
        total_loss += loss.item() * len(targets)
        all_preds.append(out["lst_pred"].cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    preds   = np.concatenate(all_preds)
    targets_np = np.concatenate(all_targets)
    val_loss = total_loss / len(loader.dataset)
    metrics  = compute_metrics(preds, targets_np)
    return val_loss, metrics


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(model: nn.Module, path: Path, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), **metadata}, path)
    logger.debug(f"Checkpoint saved → {path}")


def load_checkpoint(
    model: nn.Module, path: str | Path, device: Optional[torch.device] = None
) -> dict:
    """Load weights into model from a checkpoint file.  Returns metadata dict."""
    ckpt = torch.load(path, map_location=device or "cpu")
    model.load_state_dict(ckpt["state_dict"])
    meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
    logger.info(f"Checkpoint loaded from {path} | epoch={meta.get('epoch')} "
                f"val_rmse={meta.get('val_rmse', '?'):.3f}" if "val_rmse" in meta else
                f"Checkpoint loaded from {path}")
    return meta


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    parquet_path: Optional[str] = None,
    epochs: int          = _cfg.epochs,
    batch_size: int      = _cfg.batch_size,
    lr: float            = _cfg.lr,
    weight_decay: float  = _cfg.weight_decay,
    val_fraction: float  = _cfg.val_fraction,
    n_aug_train: int     = _cfg.n_aug_train,
    patience: int        = _cfg.patience,
    seed: int            = _cfg.seed,
    checkpoint_dir: str  = _cfg.checkpoint_dir,
    use_mlflow: bool     = True,
    pretrained: bool     = True,
) -> Path:
    """
    Run the full training loop.

    Returns the path of the best checkpoint.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")

    # ── Data ──────────────────────────────────────────────────────────
    train_loader, val_loader = make_dataloaders(
        parquet_path=parquet_path,
        batch_size=batch_size,
        val_fraction=val_fraction,
        n_aug_train=n_aug_train,
        seed=seed,
    )

    # ── Model ─────────────────────────────────────────────────────────
    model = HybridLSTPredictor(pretrained=pretrained).to(device)
    params = model.parameter_count()
    logger.info(f"Model params: {params['total']:,} total "
                f"(CNN={params['cnn']:,} ViT={params['vit']:,})")

    # ── Optimiser + scheduler ──────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )

    # ── MLflow setup ──────────────────────────────────────────────────
    mlflow_run = None
    if use_mlflow:
        try:
            import mlflow  # type: ignore
            mlflow.set_tracking_uri(_cfg.mlflow_tracking_uri)
            mlflow.set_experiment(_cfg.mlflow_experiment)
            mlflow_run = mlflow.start_run()
            mlflow.log_params({
                "epochs": epochs, "batch_size": batch_size, "lr": lr,
                "weight_decay": weight_decay, "n_aug_train": n_aug_train,
                "pretrained": pretrained, "n_channels": N_CHANNELS,
                "patch_size": PATCH_SIZE, "embed_dim": EMBED_DIM,
                "train_wards": len(train_loader.dataset) // n_aug_train,
                "val_wards":   len(val_loader.dataset),
                "total_params": params["total"],
            })
            logger.info(f"MLflow run ID: {mlflow_run.info.run_id}")
        except Exception as exc:
            logger.warning(f"MLflow unavailable ({exc}) — training without tracking")
            mlflow_run = None
            use_mlflow = False

    # ── Training loop ─────────────────────────────────────────────────
    ckpt_dir = Path(checkpoint_dir)
    best_rmse   = float("inf")
    best_ckpt   = ckpt_dir / "best_model.pt"
    no_improve  = 0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = _train_epoch(model, train_loader, optimizer, device)
        val_loss, metrics = _val_epoch(model, val_loader, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]

        row = {
            "epoch":      epoch,
            "train_loss": round(train_loss, 5),
            "val_loss":   round(val_loss, 5),
            **{k: round(v, 4) for k, v in metrics.items()},
            "lr":         round(lr_now, 7),
        }
        history.append(row)

        logger.info(
            f"Ep {epoch:3d}/{epochs} | "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"RMSE={metrics['rmse']:.3f}°C  MAE={metrics['mae']:.3f}°C  "
            f"R²={metrics['r2']:.3f}  lr={lr_now:.2e}  "
            f"[{elapsed:.1f}s]"
        )

        if use_mlflow and mlflow_run:
            try:
                import mlflow
                mlflow.log_metrics(
                    {"train_loss": train_loss, "val_loss": val_loss,
                     "val_rmse": metrics["rmse"], "val_mae": metrics["mae"],
                     "val_r2": metrics["r2"], "lr": lr_now},
                    step=epoch,
                )
            except Exception:
                pass

        # ── Checkpoint + early stopping ────────────────────────────────
        if metrics["rmse"] < best_rmse:
            best_rmse  = metrics["rmse"]
            no_improve = 0
            _save_checkpoint(model, best_ckpt, {
                "epoch": epoch, "val_rmse": best_rmse,
                "val_r2": metrics["r2"],
            })
            logger.success(f"  ↳ new best RMSE={best_rmse:.3f}°C — checkpoint saved")
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.warning(f"Early stopping at epoch {epoch} (patience={patience})")
                break

    # ── Final checkpoint ──────────────────────────────────────────────
    last_ckpt = ckpt_dir / "last_model.pt"
    _save_checkpoint(model, last_ckpt, {
        "epoch": epoch, "val_rmse": metrics["rmse"], "val_r2": metrics["r2"],
    })

    if use_mlflow and mlflow_run:
        try:
            import mlflow
            import mlflow.pytorch
            mlflow.log_metric("best_val_rmse", best_rmse)
            mlflow.log_artifact(str(best_ckpt))
            # Register model in MLflow Model Registry
            try:
                model_uri = f"runs:/{mlflow_run.info.run_id}/{best_ckpt.name}"
                mlflow.register_model(model_uri, "thermal_model")
                logger.info("Registered model as 'thermal_model' in MLflow Registry")
            except Exception as reg_exc:
                logger.warning(f"Model registration failed (non-critical): {reg_exc}")
            mlflow.end_run()
        except Exception:
            pass

    logger.success(
        f"Training complete | best_val_rmse={best_rmse:.3f}°C | "
        f"checkpoint={best_ckpt}"
    )
    return best_ckpt


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NammaClimaGrid — Train Thermal Vision Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--parquet",     type=str,   default=None,
                   help="Path to ward_features.parquet (omit for synthetic)")
    p.add_argument("--epochs",      type=int,   default=_cfg.epochs)
    p.add_argument("--batch-size",  type=int,   default=_cfg.batch_size)
    p.add_argument("--lr",          type=float, default=_cfg.lr)
    p.add_argument("--patience",    type=int,   default=_cfg.patience)
    p.add_argument("--n-aug",       type=int,   default=_cfg.n_aug_train,
                   help="Training augmentation factor per ward")
    p.add_argument("--seed",        type=int,   default=_cfg.seed)
    p.add_argument("--checkpoint-dir", type=str, default=_cfg.checkpoint_dir)
    p.add_argument("--no-mlflow",   action="store_true",
                   help="Disable MLflow tracking")
    p.add_argument("--no-pretrained", action="store_true",
                   help="Train ResNet50 backbone from scratch")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    best = train(
        parquet_path   = args.parquet,
        epochs         = args.epochs,
        batch_size     = args.batch_size,
        lr             = args.lr,
        patience       = args.patience,
        n_aug_train    = args.n_aug,
        seed           = args.seed,
        checkpoint_dir = args.checkpoint_dir,
        use_mlflow     = not args.no_mlflow,
        pretrained     = not args.no_pretrained,
    )
    logger.success(f"Best model: {best}")
