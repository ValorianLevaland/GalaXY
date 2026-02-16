"""analysis_core.py

Shared analysis core for membrane band / distance-to-ZC analysis.

This module is geometry-engine agnostic. It expects the caller to provide:
- a membrane MembraneModel (skeleton graph + dist-to-ZC)
- distance bands (BandSpec list)
- localization points (x,y)

It computes:
- per-band density metrics
- DBSCAN cluster stats (nano-clusters)
- optional hierarchical clustering on cluster centroids
- optional Ripley K / Besag L per band (2D) using the existing ripley_backend

Additionally, it supports an optional **band_callback** that receives
(point subset, DBSCAN labels, distance-to-ZC) for each band. This enables the
GUI/worker to export point-level tables and/or generate QC figures *without*
re-running DBSCAN.

License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from scipy.spatial import cKDTree

from sklearn.cluster import DBSCAN

try:
    import shapely.geometry as shgeom
    import shapely.ops as shops
except Exception:  # pragma: no cover
    shgeom = None
    shops = None

from .geometry_base import BandSpec, MembraneModel
from .ripley_backend import (
    RipleyParams,
    CSRParams,
    make_radii,
    shapely_to_parts,
    mask_points_in_poly,
    distance_to_boundary,
    estimate_K_L,
    csr_envelope_LminusR,
)


@dataclass(frozen=True)
class DBSCANParams:
    eps: float = 30.0  # units (e.g., nm)
    min_samples: int = 5
    use_z: bool = False  # If True and Z coordinates are available, cluster in (x,y,z).


@dataclass(frozen=True)
class HierarchicalDBSCANParams:
    enabled: bool = False
    eps: float = 200.0
    min_samples: int = 3


@dataclass
class BandAnalysisResult:
    band_name: str
    d_start: float
    d_end: float
    n_points: int
    area: float
    sk_length: float
    density_area: float
    density_length: float

    # DBSCAN summary
    n_clusters: int
    frac_points_in_clusters: float
    mean_cluster_size: float
    median_cluster_size: float

    # Hierarchical
    n_superclusters: Optional[int] = None

    # Ripley summary
    ripley_r_peak: Optional[float] = None
    ripley_lmr_peak: Optional[float] = None
    ripley_auc_pos: Optional[float] = None


@dataclass
class FullAnalysisOutput:
    band_summary: pd.DataFrame
    clusters: pd.DataFrame
    superclusters: Optional[pd.DataFrame]
    ripley_curves: Dict[str, pd.DataFrame]  # key=band_name
    cluster_ripley_summary: Optional[pd.DataFrame] = None
    cluster_ripley_curves: Optional[Dict[str, pd.DataFrame]] = None  # key=band__cluster


# Callback type: (band spec, points in band, dbscan labels, dist-to-zc for those points)
BandCallback = Callable[[BandSpec, np.ndarray, np.ndarray, np.ndarray], None]


def _ensure_shapely() -> None:
    if shgeom is None or shops is None:
        raise ImportError("shapely is required for cluster geometry metrics.")


def _cluster_table(
    points_xy: np.ndarray,
    labels: np.ndarray,
    band_name: str,
    d_start: float,
    d_end: float,
    dist_to_zc: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Return per-cluster metrics table."""
    _ensure_shapely()

    rows = []
    if points_xy.shape[0] == 0:
        return pd.DataFrame(
            columns=[
                "band",
                "d_start",
                "d_end",
                "cluster_id",
                "n_points",
                "centroid_x",
                "centroid_y",
                "area_hull",
                "radius_eq",
                "bbox_w",
                "bbox_h",
                "mean_d_to_zc",
            ]
        )

    for cid in sorted(set(int(x) for x in labels.tolist() if int(x) >= 0)):
        idx = np.where(labels == cid)[0]
        pts = points_xy[idx]
        n = int(len(idx))
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))

        minx, miny = float(np.min(pts[:, 0])), float(np.min(pts[:, 1]))
        maxx, maxy = float(np.max(pts[:, 0])), float(np.max(pts[:, 1]))
        bbox_w = maxx - minx
        bbox_h = maxy - miny

        # Convex hull area
        if n >= 3:
            hull = shgeom.MultiPoint([(float(x), float(y)) for x, y in pts]).convex_hull
            area = float(hull.area)
        else:
            area = 0.0
        radius_eq = float(np.sqrt(area / np.pi)) if area > 0 else 0.0

        md = float(np.mean(dist_to_zc[idx])) if dist_to_zc is not None else np.nan

        rows.append(
            {
                "band": band_name,
                "d_start": float(d_start),
                "d_end": float(d_end),
                "cluster_id": int(cid),
                "n_points": n,
                "centroid_x": cx,
                "centroid_y": cy,
                "area_hull": area,
                "radius_eq": radius_eq,
                "bbox_w": float(bbox_w),
                "bbox_h": float(bbox_h),
                "mean_d_to_zc": md,
            }
        )

    return pd.DataFrame(rows)


def run_dbscan(points: np.ndarray, params: DBSCANParams) -> np.ndarray:
    """Run DBSCAN on an (N,D) array (D>=2).

    Notes
    -----
    - scikit-learn DBSCAN works in any dimension; GalaXY_2 may pass (x,y) or (x,y,z).
    - Rows containing NaNs/Infs are automatically labeled as noise (-1) instead of crashing.
    """

    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return np.zeros((0,), dtype=int)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError("points must be an array of shape (N,D) with D>=2")

    if float(params.eps) <= 0:
        raise ValueError("DBSCAN eps must be > 0")
    if int(params.min_samples) <= 0:
        raise ValueError("DBSCAN min_samples must be > 0")

    finite = np.all(np.isfinite(pts), axis=1)
    if not np.any(finite):
        return np.full((pts.shape[0],), -1, dtype=int)

    model = DBSCAN(eps=float(params.eps), min_samples=int(params.min_samples))
    labels_finite = model.fit_predict(pts[finite]).astype(int)

    labels = np.full((pts.shape[0],), -1, dtype=int)
    labels[finite] = labels_finite
    return labels


def run_hierarchical_dbscan(cluster_table: pd.DataFrame, params: HierarchicalDBSCANParams) -> Optional[pd.DataFrame]:
    if not params.enabled:
        return None
    if cluster_table.empty:
        return pd.DataFrame(columns=["band", "supercluster_id", "n_clusters", "centroid_x", "centroid_y"])

    # Build per-band hierarchical clustering.
    out_rows = []
    for band, g in cluster_table.groupby("band"):
        centroids = g[["centroid_x", "centroid_y"]].to_numpy(dtype=float)
        if centroids.shape[0] < max(2, params.min_samples):
            continue
        labels = DBSCAN(eps=float(params.eps), min_samples=int(params.min_samples)).fit_predict(centroids)
        for sid in sorted(set(int(x) for x in labels.tolist() if int(x) >= 0)):
            idx = np.where(labels == sid)[0]
            sub = centroids[idx]
            out_rows.append(
                {
                    "band": str(band),
                    "supercluster_id": int(sid),
                    "n_clusters": int(len(idx)),
                    "centroid_x": float(np.mean(sub[:, 0])),
                    "centroid_y": float(np.mean(sub[:, 1])),
                }
            )

    return pd.DataFrame(out_rows)


def _ripley_for_window(
    points_xy: np.ndarray,
    window_geom,
    ripley: RipleyParams,
    csr: Optional[CSRParams],
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Compute L(r)-r curve and summary metrics for a given polygon window."""
    if points_xy.shape[0] < 2:
        radii = make_radii(ripley.r_min, ripley.r_max, ripley.dr)
        curve = pd.DataFrame(
            {
                "r": radii,
                "K": np.nan,
                "L": np.nan,
                "L_minus_r": np.nan,
                "n_eligible": np.zeros_like(radii, dtype=int),
                "env_LminusR_lo": np.nan,
                "env_LminusR_hi": np.nan,
                "env_LminusR_mean": np.nan,
            }
        )
        return curve, {"r_peak": np.nan, "lmr_peak": np.nan, "auc_pos": np.nan}

    parts = tuple(shapely_to_parts(window_geom))
    area = float(window_geom.area)

    radii = make_radii(ripley.r_min, ripley.r_max, ripley.dr)

    bd = None
    if ripley.edge_correction == "border":
        bd = distance_to_boundary(parts, points_xy)

    est = estimate_K_L(points_xy, area, radii, edge_correction=ripley.edge_correction, boundary_dist=bd)

    env_lo = np.full_like(radii, np.nan, dtype=float)
    env_hi = np.full_like(radii, np.nan, dtype=float)
    env_mean = np.full_like(radii, np.nan, dtype=float)

    if csr is not None and csr.n_simulations > 0 and points_xy.shape[0] >= 2:
        env = csr_envelope_LminusR(
            window_parts=parts,
            window_area=area,
            n_points=int(points_xy.shape[0]),
            radii=radii,
            edge_correction=ripley.edge_correction,
            n_simulations=csr.n_simulations,
            alpha=csr.alpha,
            seed=csr.seed,
        )
        env_lo = env["Lmr_lo"]
        env_hi = env["Lmr_hi"]
        env_mean = env["Lmr_mean"]

    lmr = est["L_minus_r"]
    if np.all(np.isnan(lmr)):
        r_peak = np.nan
        lmr_peak = np.nan
        auc_pos = np.nan
    else:
        kmax = int(np.nanargmax(lmr))
        r_peak = float(radii[kmax])
        lmr_peak = float(lmr[kmax])
        auc_pos = float(np.trapz(np.maximum(0.0, lmr), radii))

    curve = pd.DataFrame(
        {
            "r": radii,
            "K": est["K"],
            "L": est["L"],
            "L_minus_r": lmr,
            "n_eligible": est["n_eligible"],
            "env_LminusR_lo": env_lo,
            "env_LminusR_hi": env_hi,
            "env_LminusR_mean": env_mean,
        }
    )

    return curve, {"r_peak": r_peak, "lmr_peak": lmr_peak, "auc_pos": auc_pos}



def _ripley_for_cluster(
    points_xy: np.ndarray,
    band_geom,
    dbscan_eps: float,
    ripley: RipleyParams,
    csr: Optional[CSRParams],
) -> Tuple[Optional[pd.DataFrame], Dict[str, float], float]:
    """Compute Ripley on a DBSCAN cluster using a local window defined by its (buffered) hull.

    Window = convex hull(points) buffered by dbscan_eps, then intersected with the parent band geometry.
    Returns (curve_df_or_None, summary_dict, window_area).
    """
    if points_xy.shape[0] < 2:
        return None, {"r_peak": np.nan, "lmr_peak": np.nan, "auc_pos": np.nan}, float("nan")

    # Build a local support window around the cluster
    mp = shgeom.MultiPoint([(float(x), float(y)) for x, y in points_xy])
    hull = mp.convex_hull
    # Buffer to avoid degenerate hulls (Point/LineString) and to approximate eps-neighborhood
    try:
        local = hull.buffer(float(dbscan_eps))
    except Exception:
        local = hull

    try:
        local = local.intersection(band_geom)
    except Exception:
        pass

    if local is None or getattr(local, "is_empty", True):
        return None, {"r_peak": np.nan, "lmr_peak": np.nan, "auc_pos": np.nan}, float("nan")

    area = float(local.area) if hasattr(local, "area") else float("nan")
    if not np.isfinite(area) or area <= 0:
        return None, {"r_peak": np.nan, "lmr_peak": np.nan, "auc_pos": np.nan}, area

    curve, summ = _ripley_for_window(points_xy, local, ripley, csr)
    return curve, summ, area

def analyze_bands(
    *,
    points_xy: np.ndarray,
    model: MembraneModel,
    bands: List[BandSpec],
    dbscan: DBSCANParams,
    hierarchical: HierarchicalDBSCANParams,
    ripley: Optional[RipleyParams] = None,
    csr: Optional[CSRParams] = None,
    band_callback: Optional[BandCallback] = None,
    compute_cluster_ripley: bool = False,
    cluster_ripley_min_points: int = 6,
) -> FullAnalysisOutput:
    """Analyze clustering in each distance band.

    Parameters
    ----------
    points_xy:
        All localizations (x,y). Caller may pass full dataset; we filter to membrane/bands.
    model:
        MembraneModel with membrane geometry and dist-to-ZC defined on skeleton nodes.
    bands:
        List of BandSpec (band polygons) typically from geometry_base.build_distance_bands().
    dbscan:
        Nano-cluster DBSCAN parameters.
    hierarchical:
        Optional second-level DBSCAN on nano-cluster centroids.
    ripley/csr:
        Optional Ripley K/L per band.
    band_callback:
        If provided, called once per band with (band, pts_in_band, dbscan_labels, dist_to_zc).
        This enables streaming export of per-point labels and QC figures.

    Returns
    -------
    FullAnalysisOutput with per-band summary and per-cluster tables.
    """
    _ensure_shapely()

    # Filter points to membrane first.
    mem_parts = tuple(shapely_to_parts(model.membrane_geom))
    in_mem = mask_points_in_poly(mem_parts, points_xy)
    pts_mem = points_xy[in_mem]

    # Project membrane points to nearest skeleton node to attach distance-to-ZC.
    tree = cKDTree(model.node_xy)
    _, nn = tree.query(pts_mem, k=1)
    nn = nn.astype(int)
    d_pts = model.node_dist_to_zc[nn]

    band_rows: List[Dict] = []
    cluster_tables: List[pd.DataFrame] = []
    ripley_curves: Dict[str, pd.DataFrame] = {}
    cluster_ripley_rows: List[Dict] = []
    cluster_ripley_curves: Dict[str, pd.DataFrame] = {}

    for band in bands:
        band_parts = tuple(shapely_to_parts(band.geom))
        in_band = mask_points_in_poly(band_parts, pts_mem)
        pts_b = pts_mem[in_band]
        d_b = d_pts[in_band]

        labels = run_dbscan(pts_b, dbscan)

        # Optional per-band callback (for exporting point labels, figures, etc.)
        if band_callback is not None:
            band_callback(band, pts_b, labels, d_b)

        n_points = int(pts_b.shape[0])
        n_clusters = int(len(set(labels.tolist())) - (1 if -1 in labels else 0))
        frac_in = float(np.mean(labels >= 0)) if n_points > 0 else 0.0
        sizes = [int(np.sum(labels == cid)) for cid in set(labels.tolist()) if cid >= 0]
        mean_sz = float(np.mean(sizes)) if sizes else 0.0
        med_sz = float(np.median(sizes)) if sizes else 0.0

        density_area = float(n_points / band.area) if band.area > 0 else float("nan")
        density_len = float(n_points / band.skeleton_length) if band.skeleton_length > 0 else float("nan")

        ct = _cluster_table(
            pts_b,
            labels,
            band_name=band.name,
            d_start=band.d_start,
            d_end=band.d_end,
            dist_to_zc=d_b,
        )
        cluster_tables.append(ct)

        r_peak = lmr_peak = auc_pos = None
        if ripley is not None:
            curve, summ = _ripley_for_window(pts_b, band.geom, ripley, csr)
            ripley_curves[band.name] = curve
            # Always record region-level Ripley summary stats (even if we don't
            # compute per-cluster Ripley).
            r_peak = float(summ.get("r_peak", np.nan))
            lmr_peak = float(summ.get("lmr_peak", np.nan))
            auc_pos = float(summ.get("auc_pos", np.nan))
        # Optional: Ripley per DBSCAN cluster within this band
        if (ripley is not None) and compute_cluster_ripley and (pts_b.shape[0] >= 2):
            # Iterate each cluster id >= 0
            for cid in sorted({int(x) for x in set(labels.tolist()) if int(x) >= 0}):
                idx_c = labels == cid
                pts_c = pts_b[idx_c]
                if pts_c.shape[0] < int(cluster_ripley_min_points):
                    continue
                curve_c, summ_c, area_c = _ripley_for_cluster(
                    pts_c, band.geom, float(dbscan.eps), ripley, csr
                )
                if curve_c is None:
                    continue
                key = f"{band.name}__cluster_{cid}"
                cluster_ripley_curves[key] = curve_c
                cluster_ripley_rows.append(
                    {
                        "band": band.name,
                        "cluster_id": int(cid),
                        "n_points": int(pts_c.shape[0]),
                        "window_area": float(area_c),
                        "dbscan_eps": float(dbscan.eps),
                        "ripley_r_peak": float(summ_c.get("r_peak", np.nan)),
                        "ripley_lmr_peak": float(summ_c.get("lmr_peak", np.nan)),
                        "ripley_auc_pos": float(summ_c.get("auc_pos", np.nan)),
                    }
                )

            # (lmr_peak/auc_pos already populated above for the band-level curve)

        band_rows.append(
            {
                "band": band.name,
                "d_start": float(band.d_start),
                "d_end": float(band.d_end),
                "n_points": n_points,
                "area": float(band.area),
                "skeleton_length": float(band.skeleton_length),
                "density_area": density_area,
                "density_length": density_len,
                "dbscan_eps": float(dbscan.eps),
                "dbscan_min_samples": int(dbscan.min_samples),
                "n_clusters": n_clusters,
                "frac_points_in_clusters": frac_in,
                "mean_cluster_size": mean_sz,
                "median_cluster_size": med_sz,
                "ripley_r_peak": r_peak,
                "ripley_lmr_peak": lmr_peak,
                "ripley_auc_pos": auc_pos,
            }
        )

    band_summary = pd.DataFrame(band_rows)
    clusters = pd.concat(cluster_tables, ignore_index=True) if cluster_tables else pd.DataFrame()

    superclusters = run_hierarchical_dbscan(clusters, hierarchical)

    cluster_ripley_summary = pd.DataFrame(cluster_ripley_rows) if cluster_ripley_rows else None
    cluster_ripley_curves_out = cluster_ripley_curves if cluster_ripley_curves else None

    return FullAnalysisOutput(
        band_summary=band_summary,
        clusters=clusters,
        superclusters=superclusters,
        ripley_curves=ripley_curves,
        cluster_ripley_summary=cluster_ripley_summary,
        cluster_ripley_curves=cluster_ripley_curves_out,
    )
