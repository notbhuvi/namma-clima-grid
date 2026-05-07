"""
Module 7c — Sample Landsat Scene Loader
=======================================

Downloads a Landsat 8 Collection-2 Level-2 scene for the Bengaluru bounding
box using the Google Earth Engine Python API, exporting bands B2..B7 + B10
(optical + thermal) as GeoTIFFs under 8_data/landsat/.

If Earth Engine credentials are not available (no GEE_SERVICE_ACCOUNT /
GEE_KEY_FILE or library missing), the script gracefully falls back to
generating a synthetic 7-band GeoTIFF-like NumPy .npz file that matches the
band count + spatial layout expected by Module 2 (Thermal Vision Engine).
This ensures the training loop always has data to chew on, even in CI or
offline environments.

Usage:
    python 8_data/sample_landsat_loader.py                       # auto mode
    python 8_data/sample_landsat_loader.py --force-fallback      # offline
    python 8_data/sample_landsat_loader.py --size 256            # synth HxW
"""
from __future__ import annotations

import argparse
import os
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np

from _common import BENGALURU_BBOX, MODULE_DIR, logger

# Bands that Module 2 expects:
#   B2..B7 = optical (blue, green, red, NIR, SWIR1, SWIR2)
#   B10    = thermal infrared (LST proxy, label for training)
BAND_NAMES: List[str] = ["B2", "B3", "B4", "B5", "B6", "B7", "B10"]
LANDSAT_DIR: Path = MODULE_DIR / "landsat"


# ---------------------------------------------------------------------------
# Real GEE path
# ---------------------------------------------------------------------------
def _try_gee_download(out_dir: Path) -> bool:
    """
    Attempt a real Landsat download via Google Earth Engine.
    Returns True on success, False on any failure (auth, network, library).
    """
    service_account = os.getenv("GEE_SERVICE_ACCOUNT")
    key_file = os.getenv("GEE_KEY_FILE")
    if not service_account or not key_file or not Path(key_file).exists():
        logger.info("GEE credentials not configured — skipping real download.")
        return False

    try:
        import ee  # type: ignore
    except ImportError:
        logger.warning("earthengine-api not installed — skipping real download.")
        return False

    try:
        logger.info(f"Authenticating to GEE as {service_account}")
        credentials = ee.ServiceAccountCredentials(service_account, key_file)
        ee.Initialize(credentials)

        roi = ee.Geometry.Rectangle([
            BENGALURU_BBOX.west, BENGALURU_BBOX.south,
            BENGALURU_BBOX.east, BENGALURU_BBOX.north,
        ])
        end = date.today()
        start = end - timedelta(days=45)

        collection = (
            ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .filterBounds(roi)
            .filterDate(str(start), str(end))
            .sort("CLOUD_COVER")
        )
        count = collection.size().getInfo()
        if count == 0:
            logger.warning("No Landsat scenes found in the 45-day window.")
            return False

        scene = ee.Image(collection.first())
        scene_id = scene.get("system:index").getInfo()
        logger.success(f"Selected Landsat scene: {scene_id}")

        out_dir.mkdir(parents=True, exist_ok=True)

        # NOTE: ee.Image.getDownloadURL has a 32 MB payload cap which is fine
        # for our small ROI. For larger AOIs you'd use ee.batch.Export instead.
        import urllib.request
        # Map our aliases onto actual Collection-2 band names.
        gee_band_map = {
            "B2":  "SR_B2",  "B3": "SR_B3", "B4": "SR_B4", "B5": "SR_B5",
            "B6":  "SR_B6",  "B7": "SR_B7", "B10": "ST_B10",
        }
        for alias, gee_name in gee_band_map.items():
            url = scene.select(gee_name).getDownloadURL({
                "scale": 30,
                "region": roi,
                "format": "GEO_TIFF",
            })
            out_path = out_dir / f"{scene_id}_{alias}.tif"
            logger.info(f"  downloading {alias} -> {out_path.name}")
            urllib.request.urlretrieve(url, out_path)

        (out_dir / "SCENE_ID.txt").write_text(scene_id)
        logger.success(f"GEE download complete → {out_dir}")
        return True

    except Exception as e:  # noqa: BLE001
        logger.warning(f"GEE download failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Synthetic fallback
# ---------------------------------------------------------------------------
def _synthesize_scene(out_dir: Path, size: int = 256, seed: int = 42) -> Path:
    """
    Create a synthetic 7-band 'Landsat-like' scene as a compressed .npz file.
    Each band has realistic value ranges so downstream normalization works.

    Returns the path to the .npz file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Base spatial structure — a smooth 2D sinusoid modulates vegetation
    y, x = np.mgrid[0:size, 0:size]
    structure = (
        np.sin(2 * np.pi * x / size * 3)
        + np.cos(2 * np.pi * y / size * 2)
    ) * 0.5 + 0.5  # normalized 0..1

    # Optical reflectance bands (SR_B* values are roughly 0..0.4 for vegetation,
    # higher for bare/urban). We produce uint16 like actual Landsat SR products
    # where scale_factor=0.0000275 and add_offset=-0.2.
    bands = {}
    for band in ["B2", "B3", "B4", "B5", "B6", "B7"]:
        noise = rng.normal(0, 0.02, (size, size))
        # Different bands emphasise different spatial patterns
        shift = {"B2": 0.15, "B3": 0.18, "B4": 0.20, "B5": 0.35, "B6": 0.28, "B7": 0.22}[band]
        refl = np.clip(structure * 0.3 + shift + noise, 0.0, 0.6)
        # Convert to Landsat SR DN
        dn = ((refl + 0.2) / 0.0000275).astype(np.uint16)
        bands[band] = dn

    # Thermal band (ST_B10) — Kelvin ×10 in Collection-2 scale: DN ≈ (K / 0.00341802 + 149.0)
    # Target LST range: 295..315 K (≈22..42 °C)
    lst_k = 295 + structure * 20 + rng.normal(0, 0.8, (size, size))
    bands["B10"] = ((lst_k - 149.0) / 0.00341802).astype(np.uint16)

    out_path = out_dir / "synthetic_landsat_scene.npz"
    np.savez_compressed(
        out_path,
        **bands,
        bbox=np.array([
            BENGALURU_BBOX.west, BENGALURU_BBOX.south,
            BENGALURU_BBOX.east, BENGALURU_BBOX.north,
        ]),
        band_names=np.array(BAND_NAMES),
    )

    (out_dir / "SCENE_ID.txt").write_text("SYNTHETIC-FALLBACK")
    logger.success(f"Synthetic Landsat scene written → {out_path} ({size}×{size})")
    return out_path


# ---------------------------------------------------------------------------
# Public loader for downstream modules
# ---------------------------------------------------------------------------
def load_bands(scene_dir: Optional[Path] = None) -> dict[str, np.ndarray]:
    """
    Load Landsat bands from disk. Tries real GeoTIFFs first, then .npz fallback.

    Returns: dict {band_name: np.ndarray}
    """
    scene_dir = scene_dir or LANDSAT_DIR
    if not scene_dir.exists():
        raise FileNotFoundError(
            f"No Landsat scene at {scene_dir}. Run sample_landsat_loader.py first."
        )

    # Try real GeoTIFFs
    tif_files = sorted(scene_dir.glob("*_B*.tif"))
    if tif_files:
        try:
            import rasterio  # type: ignore
            out: dict[str, np.ndarray] = {}
            for band in BAND_NAMES:
                matches = [f for f in tif_files if f.name.endswith(f"_{band}.tif")]
                if not matches:
                    raise FileNotFoundError(f"Missing {band} GeoTIFF in {scene_dir}")
                with rasterio.open(matches[0]) as src:
                    out[band] = src.read(1)
            logger.info(f"Loaded real Landsat bands from {scene_dir}")
            return out
        except ImportError:
            logger.warning("rasterio not installed — falling back to .npz scene.")

    # .npz fallback
    npz_path = scene_dir / "synthetic_landsat_scene.npz"
    if npz_path.exists():
        data = np.load(npz_path)
        out = {band: data[band] for band in BAND_NAMES}
        logger.info(f"Loaded synthetic Landsat scene from {npz_path}")
        return out

    raise FileNotFoundError(f"No Landsat data found in {scene_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download/synthesize a Landsat scene for Bengaluru.")
    p.add_argument("--force-fallback", action="store_true",
                   help="Skip GEE and go straight to synthetic data")
    p.add_argument("--size", type=int, default=256,
                   help="Synthetic scene HxW in pixels (default 256)")
    p.add_argument("--out", type=Path, default=LANDSAT_DIR,
                   help="Output directory for the scene")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if not args.force_fallback and _try_gee_download(args.out):
        logger.success("Real Landsat scene ready for Module 2.")
        return

    logger.info("Using synthetic fallback scene.")
    _synthesize_scene(args.out, size=args.size)
    # Smoke test: re-load what we just wrote.
    bands = load_bands(args.out)
    for name, arr in bands.items():
        logger.info(f"  {name}: shape={arr.shape} dtype={arr.dtype} "
                    f"min={arr.min()} max={arr.max()}")
    logger.success("Synthetic Landsat scene ready for Module 2.")


if __name__ == "__main__":
    main()
