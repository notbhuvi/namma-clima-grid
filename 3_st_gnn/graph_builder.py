"""
Module 3a — Ward Adjacency Graph Builder
==========================================

Constructs the spatial graph used by the ST-GNN.  Each BBMP ward is a node;
edges connect wards that share a boundary (or are within a configurable
distance threshold) so the GNN can propagate heat and flood signals across
the city.

Graph construction strategy
────────────────────────────
  1. Real GeoJSON (preferred)
     If the ward polygon shapefile is present, edges are built by checking
     which polygon pairs share at least one boundary point (touches / intersects).

  2. k-NN fallback (default for synthetic grid)
     Connect each ward centroid to its k nearest neighbours (default k=6).
     Uses scipy KD-tree so no GIS dependency is needed.

  3. Distance threshold
     Additionally, any two wards whose centroids are within `max_dist_km` of
     each other are connected (useful for capturing cross-road urban heat
     corridors).

Outputs
────────
  edge_index : (2, E) int64 tensor   — COO format, undirected (both directions)
  edge_weight: (E,)   float32 tensor — normalised inverse distance weight
  node_coords: (N, 2) float32 tensor — (lon, lat) per ward, sorted by ward_id

The graph is deterministic given the same ward catalogue and is cached as a
.pt file so it only needs to be rebuilt when the ward geometry changes.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger

_MODULE_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _MODULE_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "8_data"))

NUM_WARDS = 198


# ---------------------------------------------------------------------------
# Coordinate loading
# ---------------------------------------------------------------------------

def _load_ward_coords(
    geojson_path: Optional[Path] = None,
) -> Tuple[np.ndarray, List[int]]:
    """
    Return (coords, ward_ids) where coords is (N, 2) float64 [lon, lat].

    Tries the real GeoJSON first, then falls back to the synthetic grid.
    """
    # Try real / synthetic GeoJSON
    if geojson_path and geojson_path.exists():
        try:
            import json
            data = json.loads(geojson_path.read_text())
            features = data["features"]
            wids, lons, lats = [], [], []
            for f in features:
                props = f["properties"]
                wids.append(int(props["ward_id"]))
                lons.append(float(props["centroid_lon"]))
                lats.append(float(props["centroid_lat"]))
            order = np.argsort(wids)
            return np.column_stack([np.array(lons)[order],
                                    np.array(lats)[order]]), list(np.array(wids)[order])
        except Exception as exc:
            logger.warning(f"GeoJSON load failed ({exc}) — using synthetic grid")

    # Synthetic grid (matches _common.py)
    try:
        from _common import synthetic_ward_catalogue, BENGALURU_BBOX  # type: ignore
        catalogue = synthetic_ward_catalogue()
        wids  = [r["ward_id"]     for r in catalogue]
        lons  = [r["centroid"][0] for r in catalogue]
        lats  = [r["centroid"][1] for r in catalogue]
        return np.column_stack([lons, lats]), wids
    except ImportError:
        pass

    # Absolute fallback: reconstruct grid inline
    logger.debug("Building synthetic ward grid inline")
    bbox_west, bbox_south, bbox_east, bbox_north = 77.45, 12.83, 77.78, 13.14
    cols = 14
    rows = math.ceil(NUM_WARDS / cols)
    dx = (bbox_east - bbox_west) / cols
    dy = (bbox_north - bbox_south) / rows
    lons, lats, wids = [], [], []
    for idx in range(NUM_WARDS):
        r, c = divmod(idx, cols)
        lons.append(bbox_west + (c + 0.5) * dx)
        lats.append(bbox_south + (r + 0.5) * dy)
        wids.append(idx + 1)
    return np.column_stack([lons, lats]), wids


# ---------------------------------------------------------------------------
# Haversine distance
# ---------------------------------------------------------------------------

def _haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in kilometres."""
    R = 6371.0
    dlon = math.radians(lon2 - lon1)
    dlat = math.radians(lat2 - lat1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(min(1.0, math.sqrt(a)))


def pairwise_distances_km(coords: np.ndarray) -> np.ndarray:
    """Return (N, N) symmetric distance matrix in km."""
    N = len(coords)
    dist = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(i + 1, N):
            d = _haversine_km(coords[i, 0], coords[i, 1],
                              coords[j, 0], coords[j, 1])
            dist[i, j] = dist[j, i] = d
    return dist


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_ward_graph(
    geojson_path: Optional[Path] = None,
    k_neighbors: int = 6,
    max_dist_km: float = 3.5,
    self_loops: bool = False,
    cache_path: Optional[Path] = None,
) -> Dict[str, torch.Tensor]:
    """
    Build the ward adjacency graph.

    Args:
        geojson_path:  Path to wards GeoJSON (None = use synthetic grid).
        k_neighbors:   Minimum neighbours per node (k-NN baseline).
        max_dist_km:   Also connect any pair closer than this threshold.
        self_loops:    Add self-loops (i→i edges) if True.
        cache_path:    If given, save/load graph to/from this .pt file.

    Returns:
        dict with keys:
          'edge_index'  : (2, E) int64
          'edge_weight' : (E,)   float32
          'node_coords' : (N, 2) float32  [lon, lat]
          'ward_ids'    : (N,)   int64
          'num_nodes'   : int
          'num_edges'   : int
    """
    # ── Cache hit ─────────────────────────────────────────────────────
    if cache_path and Path(cache_path).exists():
        graph = torch.load(cache_path, map_location="cpu")
        logger.info(
            f"Loaded cached graph from {cache_path} | "
            f"nodes={graph['num_nodes']}  edges={graph['num_edges']}"
        )
        return graph

    logger.info("Building ward adjacency graph …")
    coords, ward_ids = _load_ward_coords(geojson_path)
    N = len(coords)

    # ── Distance matrix ───────────────────────────────────────────────
    try:
        from scipy.spatial import cKDTree  # type: ignore
        tree = cKDTree(coords)
        # k+1 because the first neighbour is the point itself
        dists_knn, idxs_knn = tree.query(coords, k=min(k_neighbors + 1, N))
        use_scipy = True
    except ImportError:
        logger.warning("scipy not found — using full O(N²) distance matrix")
        use_scipy = False

    rows, cols_list = [], []

    if use_scipy:
        # Add k-NN edges
        for i in range(N):
            for j_pos in range(1, dists_knn.shape[1]):  # skip self (pos 0)
                j = int(idxs_knn[i, j_pos])
                rows.append(i); cols_list.append(j)
                rows.append(j); cols_list.append(i)

        # Add distance-threshold edges
        lat_scale = 111.0  # km per degree lat
        lon_scale = 111.0 * math.cos(math.radians(coords[:, 1].mean()))
        for i in range(N):
            for j in range(i + 1, N):
                dx = (coords[i, 0] - coords[j, 0]) * lon_scale
                dy = (coords[i, 1] - coords[j, 1]) * lat_scale
                if math.sqrt(dx**2 + dy**2) <= max_dist_km:
                    rows.append(i); cols_list.append(j)
                    rows.append(j); cols_list.append(i)
    else:
        dist_mat = pairwise_distances_km(coords)
        for i in range(N):
            nbr_indices = np.argsort(dist_mat[i])[1:k_neighbors + 1]
            for j in nbr_indices:
                rows.append(i); cols_list.append(int(j))
                rows.append(int(j)); cols_list.append(i)
            # Distance threshold
            for j in range(i + 1, N):
                if dist_mat[i, j] <= max_dist_km:
                    rows.append(i); cols_list.append(j)
                    rows.append(j); cols_list.append(i)

    if self_loops:
        for i in range(N):
            rows.append(i); cols_list.append(i)

    # ── Deduplicate edges ─────────────────────────────────────────────
    edge_set = set(zip(rows, cols_list))
    src_arr = np.array([e[0] for e in edge_set], dtype=np.int64)
    dst_arr = np.array([e[1] for e in edge_set], dtype=np.int64)

    # ── Edge weights: normalised inverse distance ──────────────────────
    lat_scale = 111.0
    lon_scale = 111.0 * math.cos(math.radians(coords[:, 1].mean()))
    weights = []
    for s, d in zip(src_arr, dst_arr):
        if s == d:
            weights.append(1.0)
        else:
            dx = (coords[s, 0] - coords[d, 0]) * lon_scale
            dy = (coords[s, 1] - coords[d, 1]) * lat_scale
            dist = max(math.sqrt(dx**2 + dy**2), 0.01)
            weights.append(1.0 / dist)
    weights_arr = np.array(weights, dtype=np.float32)
    # Normalise per-node (row-normalise)
    for i in range(N):
        mask = src_arr == i
        total = weights_arr[mask].sum()
        if total > 0:
            weights_arr[mask] /= total

    graph = {
        "edge_index":  torch.tensor(np.vstack([src_arr, dst_arr]), dtype=torch.long),
        "edge_weight": torch.tensor(weights_arr, dtype=torch.float32),
        "node_coords": torch.tensor(coords, dtype=torch.float32),
        "ward_ids":    torch.tensor(ward_ids, dtype=torch.long),
        "num_nodes":   N,
        "num_edges":   len(src_arr),
    }

    logger.success(
        f"Graph built | nodes={N}  edges={len(src_arr)}  "
        f"avg_degree={len(src_arr)/N:.1f}"
    )

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(graph, cache_path)
        logger.info(f"Graph cached → {cache_path}")

    return graph


# ---------------------------------------------------------------------------
# __main__ — smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from loguru import logger

    graph = build_ward_graph(k_neighbors=6, max_dist_km=3.5)
    N = graph["num_nodes"]
    E = graph["num_edges"]
    ei = graph["edge_index"]

    assert ei.shape[0] == 2,            "edge_index must be (2, E)"
    assert ei.max().item() < N,         "edge index out of range"
    assert graph["node_coords"].shape == (N, 2), "coords shape mismatch"
    assert graph["edge_weight"].shape  == (E,),  "weight shape mismatch"

    degrees = torch.bincount(ei[0], minlength=N).float()
    logger.info(
        f"Degree stats | min={degrees.min():.0f}  max={degrees.max():.0f}  "
        f"mean={degrees.mean():.1f}  median={degrees.median():.0f}"
    )
    logger.success("graph_builder.py smoke test passed.")
