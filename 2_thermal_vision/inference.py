"""
Module 2d — ThermalVisionInference
====================================

Inference wrapper for the trained HybridLSTPredictor.  Designed for two
downstream consumers:

  1. FastAPI backend (Module 5) — synchronous call to predict LST + heat
     stress for all 198 wards from the latest ward_features snapshot.

  2. ST-GNN (Module 3) — batch encode wards to produce the 128-dim spatial
     embedding that becomes the initial node feature.

Interface
─────────
  from inference import ThermalVisionInference

  # Load model (from MLflow run or local checkpoint)
  tvi = ThermalVisionInference.from_checkpoint("checkpoints/best_model.pt")

  # Predict from a ward features DataFrame
  result_df = tvi.predict_dataframe(ward_features_df)
  # Columns: ward_id, lst_pred_celsius, heat_stress_score, ward_embedding

  # Or encode to GNN node features
  node_features = tvi.encode_node_features(ward_features_df)
  # Returns: (198, 128) numpy float32 array, rows sorted by ward_id

Model loading strategy
───────────────────────
  1. Try supplied checkpoint path.
  2. Try latest MLflow model from "thermal_vision_lst" experiment.
  3. Fall back to an untrained (random-weight) model — prediction quality
     is poor but the pipeline never stalls (useful for integration testing).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from loguru import logger

_MODULE_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _MODULE_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "8_data"))

from model   import HybridLSTPredictor, N_CHANNELS, PATCH_SIZE, EMBED_DIM
from dataset import WardPatchDataset, make_dataloaders, ward_row_to_channels

# How many wards to process in one forward pass
_INFERENCE_BATCH = 32


# ---------------------------------------------------------------------------
# ThermalVisionInference
# ---------------------------------------------------------------------------

class ThermalVisionInference:
    """
    Production inference wrapper for the CNN-ViT LST predictor.

    Args:
        model:   Loaded HybridLSTPredictor (call a from_* factory instead).
        device:  Torch device for inference.
    """

    def __init__(
        self,
        model: HybridLSTPredictor,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device).eval()
        self._patch_gen = WardPatchDataset(augment=False)   # just for patch generation

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: Optional[torch.device] = None,
    ) -> "ThermalVisionInference":
        """Load from a .pt checkpoint file saved by train.py."""
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        model = HybridLSTPredictor(pretrained=False)
        ckpt  = torch.load(path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        epoch = ckpt.get("epoch", "?")
        rmse  = ckpt.get("val_rmse", "?")
        logger.info(f"Loaded checkpoint {path} | epoch={epoch}  val_rmse={rmse}")
        return cls(model, device)

    @classmethod
    def from_mlflow(
        cls,
        experiment_name: str = "thermal_vision_lst",
        tracking_uri: str    = "http://localhost:5001",
        device: Optional[torch.device] = None,
    ) -> "ThermalVisionInference":
        """Load the best run's model artifact from MLflow."""
        try:
            import mlflow  # type: ignore
            import mlflow.pytorch  # type: ignore
            mlflow.set_tracking_uri(tracking_uri)
            client   = mlflow.tracking.MlflowClient()
            exp      = client.get_experiment_by_name(experiment_name)
            if exp is None:
                raise RuntimeError(f"MLflow experiment '{experiment_name}' not found")
            runs = client.search_runs(
                exp.experiment_id,
                order_by=["metrics.best_val_rmse ASC"],
                max_results=1,
            )
            if not runs:
                raise RuntimeError("No completed runs found")
            run_id = runs[0].info.run_id
            model  = mlflow.pytorch.load_model(f"runs:/{run_id}/model")
            logger.info(f"Loaded model from MLflow run {run_id}")
            return cls(model, device)
        except Exception as exc:
            logger.warning(f"MLflow load failed ({exc}) — falling back to untrained model")
            return cls.untrained(device)

    @classmethod
    def untrained(
        cls, device: Optional[torch.device] = None
    ) -> "ThermalVisionInference":
        """Return an untrained model — useful for integration tests."""
        logger.warning(
            "Using untrained (random-weight) ThermalVisionInference. "
            "Predictions will be meaningless until a checkpoint is loaded."
        )
        model = HybridLSTPredictor(pretrained=False)
        return cls(model, device)

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _forward_batch(self, patches: torch.Tensor) -> dict[str, np.ndarray]:
        """Run model on a (B, C, H, W) tensor. Returns numpy arrays."""
        patches = patches.to(self.device)
        out     = self.model(patches)
        return {
            "lst_pred":       out["lst_pred"].cpu().numpy(),
            "heat_stress":    out["heat_stress"].cpu().numpy(),
            "ward_embedding": out["ward_embedding"].cpu().numpy(),
        }

    # ------------------------------------------------------------------
    def predict_dataframe(
        self, ward_features_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Predict LST + heat stress + ward embedding for every row in the df.

        Args:
            ward_features_df: DataFrame with at least a 'ward_id' column and
                              the expected feature columns (see dataset.py).
                              Missing columns are filled with defaults.

        Returns:
            DataFrame with columns:
              ward_id, lst_pred_celsius, heat_stress_score,
              ward_embedding  (list of 128 floats per row)
        """
        ds = WardPatchDataset(source=ward_features_df, augment=False)
        all_ids:    list[int]        = []
        all_lst:    list[float]      = []
        all_hs:     list[float]      = []
        all_emb:    list[np.ndarray] = []

        # Process in batches
        patches_buf: list[torch.Tensor] = []
        ids_buf:     list[int]          = []

        def _flush() -> None:
            if not patches_buf:
                return
            batch  = torch.stack(patches_buf)
            result = self._forward_batch(batch)
            all_lst.extend(result["lst_pred"].tolist())
            all_hs.extend(result["heat_stress"].tolist())
            all_emb.extend(result["ward_embedding"])
            all_ids.extend(ids_buf)
            patches_buf.clear()
            ids_buf.clear()

        for i in range(len(ds)):
            patch, _, wid = ds[i]
            patches_buf.append(patch)
            ids_buf.append(wid)
            if len(patches_buf) == _INFERENCE_BATCH:
                _flush()
        _flush()

        result_df = pd.DataFrame({
            "ward_id":           all_ids,
            "lst_pred_celsius":  [round(v, 2) for v in all_lst],
            "heat_stress_score": [round(v, 2) for v in all_hs],
            "ward_embedding":    all_emb,
        }).sort_values("ward_id").reset_index(drop=True)

        logger.info(
            f"Inference complete | wards={len(result_df)} | "
            f"lst_pred μ={result_df['lst_pred_celsius'].mean():.1f}°C  "
            f"heat_stress μ={result_df['heat_stress_score'].mean():.1f}"
        )
        return result_df

    # ------------------------------------------------------------------
    def encode_node_features(
        self, ward_features_df: pd.DataFrame
    ) -> np.ndarray:
        """
        Return a (N_wards, 128) float32 numpy array of ward embeddings,
        ordered by ward_id ascending.  This is the primary interface for
        the ST-GNN in Module 3.

        Args:
            ward_features_df: Ward features DataFrame.

        Returns:
            numpy float32 array of shape (N_wards, EMBED_DIM).
        """
        result_df = self.predict_dataframe(ward_features_df)
        # Stack embedding vectors in ward_id order
        node_features = np.stack(result_df["ward_embedding"].values).astype(np.float32)
        logger.info(f"Node features | shape={node_features.shape}  "
                    f"norm_mean={np.linalg.norm(node_features, axis=1).mean():.3f}")
        return node_features

    # ------------------------------------------------------------------
    def predict_parquet(
        self, parquet_path: str | Path
    ) -> pd.DataFrame:
        """Load a ward_features parquet and run predict_dataframe on it."""
        path = Path(parquet_path)
        if not path.exists():
            raise FileNotFoundError(f"Parquet not found: {path}")
        df = pd.read_parquet(path)
        logger.info(f"Loaded {len(df)} ward rows from {path}")
        return self.predict_dataframe(df)


# ---------------------------------------------------------------------------
# Convenience function for FastAPI
# ---------------------------------------------------------------------------

def load_default_inference(
    checkpoint_path: Optional[str] = None,
) -> ThermalVisionInference:
    """
    Load inference model using a priority chain:
      1. Supplied checkpoint_path argument
      2. THERMAL_VISION_CHECKPOINT env var
      3. checkpoints/best_model.pt (relative to module dir)
      4. MLflow latest run
      5. Untrained fallback

    Designed for FastAPI startup events where failure to load should
    degrade gracefully rather than crash the server.
    """
    import os

    candidates = [
        checkpoint_path,
        os.getenv("THERMAL_VISION_CHECKPOINT"),
        str(_MODULE_DIR / "checkpoints" / "best_model.pt"),
    ]

    for path in candidates:
        if path and Path(path).exists():
            try:
                return ThermalVisionInference.from_checkpoint(path)
            except Exception as exc:
                logger.warning(f"Failed to load checkpoint {path}: {exc}")

    # Try MLflow
    try:
        return ThermalVisionInference.from_mlflow()
    except Exception:
        pass

    return ThermalVisionInference.untrained()


# ---------------------------------------------------------------------------
# __main__ — inference smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from loguru import logger

    logger.info("ThermalVisionInference smoke test (untrained model)")

    tvi = ThermalVisionInference.untrained()

    # Build a tiny synthetic ward features dataframe (5 wards)
    synthetic_df = pd.DataFrame({
        "ward_id":            [1, 2, 3, 4, 5],
        "ndvi":               [0.45, 0.20, 0.35, 0.10, 0.55],
        "ndbi":               [0.05, 0.30, 0.15, 0.45, 0.00],
        "impervious_pct":     [40,   70,   55,   85,   25  ],
        "lst_celsius":        [33.1, 38.4, 35.7, 41.2, 31.5],
        "rainfall_mm_24h":    [1.2,  0.0,  0.5,  0.0,  2.1 ],
        "pm25_ugm3":          [38.0, 55.0, 42.0, 72.0, 29.0],
        "population_density": [5000, 8000, 6500, 9000, 3000],
    })

    result = tvi.predict_dataframe(synthetic_df)
    print(result[["ward_id", "lst_pred_celsius", "heat_stress_score"]].to_string(index=False))

    node_feat = tvi.encode_node_features(synthetic_df)
    assert node_feat.shape == (5, EMBED_DIM), f"Expected (5, {EMBED_DIM}), got {node_feat.shape}"
    assert node_feat.dtype == np.float32

    logger.success(f"Node features shape: {node_feat.shape}  (ready for GNN)")
    logger.success("inference.py smoke test passed.")
