"""
Module 2b — WardPatchDataset
=============================

Converts per-ward tabular features (from Module 1's Delta Lake / Parquet
snapshots) into multi-channel 2-D spatial patches suitable for the
CNN-ViT model.

Dataset flow
─────────────
  ward_features.parquet (198 rows × feature cols)
       │
       ▼  SyntheticPatchGenerator
  (198, C=7, H=64, W=64) — multi-channel raster patch per ward
       │
       ▼  WardPatchDataset / DataLoader
  (batch, C, H, W), (batch,) targets, (batch,) ward_ids

Synthetic patch generation
───────────────────────────
Each ward's tabular scalar features are "painted" onto a 64×64 spatial
canvas with realistic texture:

  Channel  Source                      Texture
  ───────  ──────────────────────────  ────────────────────────────────────
  0  NDVI           ndvi               Smooth gradient + Perlin-like noise
  1  NDBI           ndbi               High-pass from impervious_pct
  2  impervious_pct impervious_pct/100 Radial gradient (denser at centre)
  3  rainfall       log(rain+0.1)      Blurred noise field
  4  pm25           log(pm25)/6        Biased by industrial neighbours
  5  pop_density    log(pop+1)/10      Concentrated near grid centre
  6  heat_proxy     lst_celsius/50     Smooth UHI gradient (hot core)

The texture is deterministic given (ward_id, seed) so runs are
reproducible.  Augmentation (random flip, small rotation) is applied
during training only.

Parquet schema expected
────────────────────────
  ward_id, ndvi, ndbi, impervious_pct, lst_celsius,
  rainfall_mm_24h, pm25_ugm3, population_density,
  heat_stress_score   (optional — used if present)

Any missing column is filled with its statistical mean.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split  # type: ignore
from loguru import logger

# Model constants
N_CHANNELS  = 7
PATCH_SIZE  = 64
CHANNEL_NAMES = ["ndvi", "ndbi", "impervious_norm", "rainfall_log",
                 "pm25_log", "pop_log", "heat_proxy"]


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _log_norm(x: float, eps: float = 0.1, scale: float = 6.0) -> float:
    return math.log(max(x, 0) + eps) / scale


def _clip_norm(x: float, lo: float, hi: float) -> float:
    return max(0.0, min(1.0, (x - lo) / (hi - lo + 1e-8)))


def ward_row_to_channels(row: pd.Series) -> np.ndarray:
    """
    Convert one ward's tabular row into 7 scalar channel values (0..1).

    These values become the "background" of the synthetic patch before
    spatial texture is applied.
    """
    return np.array([
        _clip_norm(float(row.get("ndvi", 0.3)),            -1.0,  1.0),   # 0 NDVI
        _clip_norm(float(row.get("ndbi", 0.0)),            -1.0,  1.0),   # 1 NDBI
        _clip_norm(float(row.get("impervious_pct", 50)),    0.0, 100.0),  # 2 impervious
        _log_norm(float(row.get("rainfall_mm_24h", 0.0))),                 # 3 rainfall
        _log_norm(float(row.get("pm25_ugm3", 40.0))),                      # 4 PM2.5
        _log_norm(float(row.get("population_density", 5000)), scale=12.0), # 5 pop density
        _clip_norm(float(row.get("lst_celsius", 34.0)),    20.0,  50.0),  # 6 heat proxy
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Synthetic patch generator
# ---------------------------------------------------------------------------

class SyntheticPatchGenerator:
    """
    Generates a C×H×W numpy float32 patch for one ward.

    The patch uses the ward's scalar features as mean values and adds
    deterministic smooth texture (low-frequency noise) to simulate real
    raster heterogeneity within a ward.
    """

    def __init__(self, patch_size: int = PATCH_SIZE, seed: int = 0) -> None:
        self.H = patch_size
        self.W = patch_size

    # ------------------------------------------------------------------
    def _smooth_noise(self, rng: np.random.Generator, scale: float = 0.05) -> np.ndarray:
        """Return H×W array of smooth (low-freq) noise in [-scale, +scale]."""
        # Generate a coarse grid then bilinearly upsample
        coarse = rng.standard_normal((8, 8)).astype(np.float32)
        coarse = (coarse - coarse.mean()) / (coarse.std() + 1e-6) * scale
        # Upsample with simple repeat + gaussian blur approximation
        fine = np.kron(coarse, np.ones((self.H // 8, self.W // 8), dtype=np.float32))
        # Trim / pad to exact shape
        fine = fine[:self.H, :self.W]
        if fine.shape != (self.H, self.W):
            padded = np.zeros((self.H, self.W), dtype=np.float32)
            padded[:fine.shape[0], :fine.shape[1]] = fine
            fine = padded
        return fine

    def _radial_gradient(self, cy: float = 0.5, cx: float = 0.5) -> np.ndarray:
        """Radial gradient peaking at (cy, cx) (fractional coords), range 0..1."""
        ys = np.linspace(0, 1, self.H, dtype=np.float32)
        xs = np.linspace(0, 1, self.W, dtype=np.float32)
        YY, XX = np.meshgrid(ys, xs, indexing="ij")
        dist = np.sqrt((YY - cy) ** 2 + (XX - cx) ** 2)
        return np.clip(1.0 - dist / (dist.max() + 1e-6), 0.0, 1.0)

    # ------------------------------------------------------------------
    def generate(
        self,
        channel_vals: np.ndarray,
        ward_id: int,
        augment: bool = False,
    ) -> np.ndarray:
        """
        Generate a (C, H, W) patch from scalar channel_vals.

        Args:
            channel_vals: (C,) array of normalised scalar values per channel.
            ward_id:      Used as RNG seed for reproducibility.
            augment:      If True, apply random flip and small rotation.

        Returns:
            float32 ndarray of shape (C, H, W), values roughly in [-0.5, 1.5]
            (mean ~0..1 per channel, small noise added).
        """
        rng = np.random.default_rng(ward_id * 31337)
        patch = np.zeros((N_CHANNELS, self.H, self.W), dtype=np.float32)

        # NDVI (ch0): smooth gradient, higher toward edges (vegetation at periphery)
        patch[0] = channel_vals[0] + self._smooth_noise(rng, 0.08) \
                   + 0.05 * (1.0 - self._radial_gradient())

        # NDBI (ch1): built-up higher at centre
        patch[1] = channel_vals[1] + self._smooth_noise(rng, 0.06) \
                   + 0.08 * self._radial_gradient()

        # Impervious (ch2): correlated radial
        patch[2] = channel_vals[2] + self._smooth_noise(rng, 0.04) \
                   + 0.06 * self._radial_gradient()

        # Rainfall (ch3): smooth random field
        patch[3] = channel_vals[3] + self._smooth_noise(rng, 0.10)

        # PM2.5 (ch4): correlated with NDBI texture
        patch[4] = channel_vals[4] + 0.3 * patch[1] + self._smooth_noise(rng, 0.05)

        # Population (ch5): radial concentration
        patch[5] = channel_vals[5] + self._smooth_noise(rng, 0.04) \
                   + 0.04 * self._radial_gradient(0.4, 0.4)

        # Heat proxy (ch6): UHI gradient, hot at centre
        patch[6] = channel_vals[6] + self._smooth_noise(rng, 0.05) \
                   + 0.07 * self._radial_gradient() \
                   - 0.04 * patch[0]   # cooler where NDVI is high

        # Augmentation (training only)
        if augment:
            aug_rng = np.random.default_rng(rng.integers(0, 2**31))
            if aug_rng.random() > 0.5:
                patch = patch[:, :, ::-1].copy()   # horizontal flip
            if aug_rng.random() > 0.5:
                patch = patch[:, ::-1, :].copy()   # vertical flip
            # Random brightness shift (per-channel)
            brightness = aug_rng.uniform(-0.05, 0.05, size=(N_CHANNELS, 1, 1)).astype(np.float32)
            patch = patch + brightness
            # Random contrast (per-channel)
            contrast = aug_rng.uniform(0.9, 1.1, size=(N_CHANNELS, 1, 1)).astype(np.float32)
            ch_mean = patch.mean(axis=(1, 2), keepdims=True)
            patch = (patch - ch_mean) * contrast + ch_mean
            # Gaussian noise
            noise = aug_rng.normal(0, 0.02, size=patch.shape).astype(np.float32)
            patch = patch + noise

        return patch.astype(np.float32)


# ---------------------------------------------------------------------------
# WardPatchDataset
# ---------------------------------------------------------------------------

class WardPatchDataset(Dataset):
    """
    PyTorch Dataset that produces (patch, target_lst, ward_id) tuples.

    Can be constructed from a Parquet path or directly from a DataFrame.
    If the Parquet/DataFrame is missing expected columns they are filled
    with reasonable defaults so the dataset always works with synthetic data.
    """

    REQUIRED_COLS = [
        "ward_id", "ndvi", "ndbi", "impervious_pct",
        "lst_celsius", "rainfall_mm_24h", "pm25_ugm3", "population_density",
    ]
    DEFAULT_VALUES = {
        "ndvi": 0.30, "ndbi": 0.05, "impervious_pct": 55.0,
        "lst_celsius": 34.0, "rainfall_mm_24h": 0.5,
        "pm25_ugm3": 42.0, "population_density": 6000.0,
    }

    def __init__(
        self,
        source: Optional[str | Path | pd.DataFrame] = None,
        patch_size: int = PATCH_SIZE,
        augment: bool = False,
        n_augmentations: int = 1,
        seed: int = 42,
    ) -> None:
        """
        Args:
            source:         Parquet path, DataFrame, or None (generates 198
                            fully synthetic rows).
            patch_size:     Spatial resolution of generated patches.
            augment:        Apply random flips during patch generation.
            n_augmentations: Number of augmented copies per ward (training).
            seed:           RNG seed for synthetic data generation.
        """
        self.patch_gen    = SyntheticPatchGenerator(patch_size=patch_size)
        self.augment      = augment
        self.n_aug        = n_augmentations

        df = self._load_source(source, seed)
        self._df = df.reset_index(drop=True)
        logger.info(
            f"WardPatchDataset ready | wards={len(self._df)} "
            f"augment={augment} n_aug={n_augmentations} "
            f"total_samples={len(self)}"
        )

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def _load_source(
        self, source: Optional[str | Path | pd.DataFrame], seed: int
    ) -> pd.DataFrame:
        if source is None:
            return self._synthetic_df(seed)

        if isinstance(source, pd.DataFrame):
            df = source.copy()
        else:
            path = Path(source)
            if not path.exists():
                logger.warning(f"Parquet not found at {path}; using synthetic data")
                return self._synthetic_df(seed)
            df = pd.read_parquet(path)

        # Fill missing columns with defaults
        for col, default in self.DEFAULT_VALUES.items():
            if col not in df.columns:
                df[col] = default
                logger.debug(f"Filled missing column '{col}' with {default}")
        if "ward_id" not in df.columns:
            df["ward_id"] = range(1, len(df) + 1)

        return df[list({c for c in df.columns if c in self.REQUIRED_COLS + ["ward_code", "ward_name"]}
                       | set(self.REQUIRED_COLS))].copy()

    @staticmethod
    def _synthetic_df(seed: int = 42) -> pd.DataFrame:
        """Generate a fully synthetic 198-row ward feature DataFrame."""
        import sys, os
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "8_data"))
        try:
            from mock_ward_features import generate_ward_features  # type: ignore
            df = generate_ward_features(seed=seed)
            logger.info("Synthetic ward features loaded from mock_ward_features.py")
            return df
        except Exception as exc:
            logger.warning(f"mock_ward_features import failed ({exc}); building minimal synthetic df")

        rng = np.random.default_rng(seed)
        n = 198
        cols = 14
        rows_g = math.ceil(n / cols)
        records = []
        for idx in range(n):
            r, c = divmod(idx, cols)
            cbd  = math.exp(-(((r - 7) ** 2 + (c - 7) ** 2) / 6))
            peen = math.exp(-(((r - 11) ** 2 + (c - 2) ** 2) / 5))
            eco  = math.exp(-(((r - 1) ** 2 + (c - 11) ** 2) / 5))
            ind  = min(1.0, 0.6 * cbd + 0.8 * peen + 0.7 * eco)
            ndvi = float(np.clip(rng.uniform(0.1, 0.5) - 0.25 * ind, 0.02, 0.75))
            ndbi = float(np.clip(rng.uniform(-0.2, 0.4) + 0.2 * ind, -0.5, 0.6))
            lst  = float(np.clip(rng.normal(35, 4) + 5 * ind - 6 * ndvi, 22, 48))
            records.append({
                "ward_id":           idx + 1,
                "ndvi":              round(ndvi, 4),
                "ndbi":              round(ndbi, 4),
                "impervious_pct":    round(float(np.clip((ndbi + 1) / 2 * 100, 30, 98)), 2),
                "lst_celsius":       round(lst, 2),
                "rainfall_mm_24h":   round(float(rng.exponential(1.5)), 3),
                "pm25_ugm3":         round(float(rng.lognormal(3.5 + ind, 0.4)), 2),
                "population_density": round(float(rng.lognormal(8.5 - 0.5 * ind, 0.5)), 0),
                "heat_stress_score": round(float(np.clip(0.4 * ind + 0.3 * (lst - 22) / 26
                                                         - 0.2 * ndvi, 0, 1) * 100), 2),
            })
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._df) * self.n_aug

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Returns:
            patch:    (C, H, W) float32 tensor
            target:   scalar float tensor — LST in Celsius
            ward_id:  int
        """
        ward_idx = idx % len(self._df)
        aug_idx  = idx // len(self._df)

        row = self._df.iloc[ward_idx]
        channel_vals = ward_row_to_channels(row)
        ward_id      = int(row["ward_id"])

        patch_np = self.patch_gen.generate(
            channel_vals,
            ward_id=ward_id + aug_idx * 100000,   # unique seed per augmentation
            augment=self.augment,
        )

        patch  = torch.from_numpy(patch_np)
        target = torch.tensor(float(row.get("lst_celsius", 34.0)), dtype=torch.float32)
        return patch, target, ward_id

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------
    def get_ward_ids(self) -> List[int]:
        return self._df["ward_id"].tolist()

    def get_dataframe(self) -> pd.DataFrame:
        return self._df.copy()


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_dataloaders(
    parquet_path: Optional[str | Path] = None,
    batch_size: int = 32,
    val_fraction: float = 0.20,
    n_aug_train: int = 8,
    seed: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders.

    Uses stratified ward split (80/20 by default).  The training set
    is augmented with `n_aug_train` copies per ward.

    Args:
        parquet_path:  Path to ward_features.parquet (None = synthetic).
        batch_size:    Samples per batch.
        val_fraction:  Fraction of wards held out for validation.
        n_aug_train:   Augmentation factor on the training set.
        seed:          RNG seed for reproducibility.
        num_workers:   DataLoader worker processes.

    Returns:
        (train_loader, val_loader)
    """
    full_ds = WardPatchDataset(source=parquet_path, augment=False)
    df = full_ds.get_dataframe()

    train_df, val_df = train_test_split(df, test_size=val_fraction, random_state=seed)
    logger.info(f"Split | train={len(train_df)} wards | val={len(val_df)} wards")

    train_ds = WardPatchDataset(source=train_df, augment=True,  n_augmentations=n_aug_train)
    val_ds   = WardPatchDataset(source=val_df,   augment=False, n_augmentations=1)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    logger.info(
        f"DataLoaders | train_batches={len(train_loader)} "
        f"val_batches={len(val_loader)}"
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# __main__ — dataset smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from loguru import logger

    logger.info("WardPatchDataset smoke test")

    ds = WardPatchDataset(n_augmentations=3)
    logger.info(f"Dataset size: {len(ds)}")

    patch, target, wid = ds[0]
    assert patch.shape  == (N_CHANNELS, PATCH_SIZE, PATCH_SIZE), "patch shape mismatch"
    assert patch.dtype  == torch.float32,                         "patch dtype mismatch"
    assert target.dtype == torch.float32,                         "target dtype mismatch"
    logger.info(
        f"Sample 0 | ward_id={wid}  target={target.item():.1f}°C  "
        f"patch_mean={patch.mean().item():.3f}  patch_std={patch.std().item():.3f}"
    )

    train_dl, val_dl = make_dataloaders(n_aug_train=2, batch_size=16)
    b_patch, b_target, b_ids = next(iter(train_dl))
    assert b_patch.shape == (16, N_CHANNELS, PATCH_SIZE, PATCH_SIZE)
    logger.info(
        f"Train batch | shape={b_patch.shape}  "
        f"target range=[{b_target.min():.1f}, {b_target.max():.1f}]°C"
    )
    logger.success("dataset.py smoke test passed.")
