"""galaxy.analysis

Generic analysis backend for GalaXY.

This module performs per-window analyses given:
- point coordinates
- a list of RegionSpec windows (polygon subregions)
- optional per-point reference values (distance-to-something)

For each window it computes:
- intensity / density (per area; per length if provided)
- DBSCAN nano-clusters + per-cluster geometry metrics
- optional hierarchical DBSCAN on nano-cluster centroids
- optional Ripley's K / Besag's L(r)-r with CSR envelopes

The module is intentionally UI-agnostic.

License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from scipy.spatial import cKDTree

try:
    import shapely.geometry as shgeom
    import shapely.ops as shops
except Exception:  # pragma: no cover
    shgeom = None
    shops = None

from .tiling import RegionSpec
from .analysis_core import (
    DBSCANParams,
    HierarchicalDBSCANParams,
    run_dbscan,
    run_hierarchical_dbscan,
)
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


# ----------------------------
# Dataclasses
# ----------------------------

@dataclass
class RegionAnalysisOutput:
    region_summary: pd.DataFrame
    clusters: pd.DataFrame
    superclusters: Optional[pd.DataFrame]
    ripley_curves: Dict[str, pd.DataFrame]
    cluster_ripley_summary: Optional[pd.DataFrame] = None
    cluster_ripley_curves: Optional[Dict[str, pd.DataFrame]] = None


# Callback type: (region, pts_in_region, dbscan_labels, ref_values)
RegionCallback = Callable[[RegionSpec, np.ndarray, np.ndarray, Optional[np.ndarray]], None]


# ----------------------------
# Internal helpers
# ----------------------------

def _ensure_shapely() -> None:
    if shgeom is None or shops is None:
        raise ImportError("shapely is required for cluster geometry metrics")


def _cluster_table_generic(
    *,
    points_xy: np.ndarray,
    points_z: Optional[np.ndarray] = None,
    labels: np.ndarray,
    region_name: str,
    axis: str,
    d_start: float,
    d_end: float,
    ref_values: Optional[np.ndarray] = None,
    ref_name: str = "ref",
) -> pd.DataFrame:
    """Return per-cluster metrics table.

    Notes
    -----
    - This is robust to empty inputs.
    - `ref_values` can be any scalar per point (distance, intensity, ...).
    """

    _ensure_shapely()

    base_cols = [
        "region",
        "axis",
        "d_start",
        "d_end",
        "cluster_id",
        "n_points",
        "centroid_x",
        "centroid_y",
        "centroid_z",
        "area_hull",
        "radius_eq",
        "bbox_w",
        "bbox_h",
        "bbox_dz",
    ]

    ref_col = f"mean_{ref_name}" if ref_values is not None else None

    cols = base_cols + ([ref_col] if ref_col else [])

    if points_xy.size == 0:
        return pd.DataFrame(columns=cols)

    pts = np.asarray(points_xy, dtype=float)
    lab = np.asarray(labels, dtype=int)
    if pts.shape[0] != lab.shape[0]:
        raise ValueError("points_xy and labels must have same length")

    rows: List[Dict] = []
    for cid in sorted(set(int(x) for x in lab.tolist() if int(x) >= 0)):
        idx = np.where(lab == cid)[0]
        p = pts[idx]
        n = int(len(idx))
        cx = float(np.mean(p[:, 0]))
        cy = float(np.mean(p[:, 1]))
        if points_z is not None:
            z = np.asarray(points_z, dtype=float)
            if z.shape[0] != pts.shape[0]:
                raise ValueError("points_z must have same length as points_xy")
            cz = float(np.nanmean(z[idx])) if n > 0 else np.nan
        else:
            cz = np.nan

        minx, miny = float(np.min(p[:, 0])), float(np.min(p[:, 1]))
        maxx, maxy = float(np.max(p[:, 0])), float(np.max(p[:, 1]))
        bbox_w = float(maxx - minx)
        bbox_h = float(maxy - miny)
        if points_z is not None:
            z = np.asarray(points_z, dtype=float)
            zc = z[idx]
            # robust to NaNs: dz is computed on finite entries only
            zc_fin = zc[np.isfinite(zc)]
            bbox_dz = float(np.max(zc_fin) - np.min(zc_fin)) if zc_fin.size else np.nan
        else:
            bbox_dz = np.nan

        # Convex hull area
        if n >= 3:
            hull = shgeom.MultiPoint([(float(x), float(y)) for x, y in p]).convex_hull
            area = float(hull.area)
        else:
            area = 0.0
        radius_eq = float(np.sqrt(area / np.pi)) if area > 0 else 0.0

        row = {
            "region": str(region_name),
            "axis": str(axis),
            "d_start": float(d_start),
            "d_end": float(d_end),
            "cluster_id": int(cid),
            "n_points": n,
            "centroid_x": cx,
            "centroid_y": cy,
            "centroid_z": cz,
            "area_hull": area,
            "radius_eq": radius_eq,
            "bbox_w": bbox_w,
            "bbox_h": bbox_h,
            "bbox_dz": bbox_dz,
        }

        if ref_values is not None:
            rv = np.asarray(ref_values, dtype=float)
            if rv.shape[0] != pts.shape[0]:
                raise ValueError("ref_values must have same length as points_xy")
            row[ref_col] = float(np.nanmean(rv[idx])) if n > 0 else np.nan

        rows.append(row)

    return pd.DataFrame(rows, columns=cols)


def _ripley_for_region(
    *,
    points_xy: np.ndarray,
    region: RegionSpec,
    ripley: RipleyParams,
    csr: Optional[CSRParams],
    logger=None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Compute L(r)-r curve and summary metrics for one region."""

    radii = make_radii(ripley.r_min, ripley.r_max, ripley.dr)

    if points_xy.shape[0] < 2 or region.area <= 0:
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

    bd = None
    if ripley.edge_correction == "border":
        bd = distance_to_boundary(region.parts, points_xy)

    est = estimate_K_L(
        points_xy,
        float(region.area),
        radii,
        edge_correction=ripley.edge_correction,
        boundary_dist=bd,
        logger=logger,
    )

    env_lo = np.full_like(radii, np.nan, dtype=float)
    env_hi = np.full_like(radii, np.nan, dtype=float)
    env_mean = np.full_like(radii, np.nan, dtype=float)

    if csr is not None and int(csr.n_simulations) > 0 and points_xy.shape[0] >= 2:
        env = csr_envelope_LminusR(
            window_parts=region.parts,
            window_area=float(region.area),
            n_points=int(points_xy.shape[0]),
            radii=radii,
            edge_correction=ripley.edge_correction,
            n_simulations=int(csr.n_simulations),
            alpha=float(csr.alpha),
            seed=csr.seed,
            logger=logger,
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


# ----------------------------
# Public API
# ----------------------------


def _ripley_for_cluster_region(
    *,
    points_xy: np.ndarray,
    parent_region: RegionSpec,
    dbscan_eps: float,
    ripley: RipleyParams,
    csr: Optional[CSRParams],
    logger=None,
) -> Tuple[Optional[pd.DataFrame], Dict[str, float], float]:
    """Ripley for one DBSCAN cluster inside a region.

    Window = convex hull(points) buffered by dbscan_eps, intersected with the parent region geometry.
    This gives a *local* interaction diagnostic at the cluster support scale.
    """
    if points_xy.shape[0] < 2:
        return None, {"r_peak": np.nan, "lmr_peak": np.nan, "auc_pos": np.nan}, float("nan")

    mp = shgeom.MultiPoint([(float(x), float(y)) for x, y in points_xy])
    hull = mp.convex_hull
    try:
        local = hull.buffer(float(dbscan_eps))
    except Exception:
        local = hull

    try:
        local = local.intersection(parent_region.geom)
    except Exception:
        pass

    if local is None or getattr(local, "is_empty", True):
        return None, {"r_peak": np.nan, "lmr_peak": np.nan, "auc_pos": np.nan}, float("nan")

    area = float(local.area) if hasattr(local, "area") else float("nan")
    if not np.isfinite(area) or area <= 0:
        return None, {"r_peak": np.nan, "lmr_peak": np.nan, "auc_pos": np.nan}, area

    # Reuse region-level function but with the local window
    parts = tuple(shapely_to_parts(local))
    radii = make_radii(ripley.r_min, ripley.r_max, ripley.dr)

    bd = None
    if ripley.edge_correction == "border":
        bd = distance_to_boundary(parts, points_xy)

    est = estimate_K_L(points_xy, area, radii, edge_correction=ripley.edge_correction, boundary_dist=bd)

    env_lo = np.full_like(radii, np.nan, dtype=float)
    env_hi = np.full_like(radii, np.nan, dtype=float)
    env_mean = np.full_like(radii, np.nan, dtype=float)

    if csr is not None and csr.n_simulations > 0 and points_xy.shape[0] >= 2:
        try:
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
        except Exception:
            if logger is not None:
                try:
                    logger.warning("CSR envelope failed for a cluster; continuing without envelope.")
                except Exception:
                    pass

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
    return curve, {"r_peak": r_peak, "lmr_peak": lmr_peak, "auc_pos": auc_pos}, area

def analyze_regions(
    *,
    points_xy: np.ndarray,
    points_z: Optional[np.ndarray] = None,
    regions: List[RegionSpec],
    dbscan: DBSCANParams,
    hierarchical: HierarchicalDBSCANParams,
    ripley: Optional[RipleyParams] = None,
    csr: Optional[CSRParams] = None,
    ref_values: Optional[np.ndarray] = None,
    ref_name: str = "ref",
    region_callback: Optional[RegionCallback] = None,
    analysis_mode: str = "full",
    compute_cluster_ripley: bool = False,
    cluster_ripley_min_points: int = 6,
    logger=None,
) -> RegionAnalysisOutput:
    """Analyze clustering in each region.

    Parameters
    ----------
    points_xy:
        All points (x,y) in world units.
    regions:
        List of RegionSpec windows.
    dbscan/hierarchical:
        DBSCAN parameters.
    ripley/csr:
        Optional Ripley analysis (2D) + CSR envelopes.
    ref_values:
        Optional scalar value per point (same length as points_xy). Use NaN for points that
        should not contribute.
    ref_name:
        Label for the scalar (used in output column names).
    region_callback:
        If provided, called for each region with (region, pts_in_region, labels, ref_in_region).
    analysis_mode:
        "full" (default) runs DBSCAN (+ optional hierarchical + optional cluster-Ripley).
        "ripley_only" skips DBSCAN/hierarchical and computes only region-level Ripley (if enabled).

    Returns
    -------
    RegionAnalysisOutput
    """

    mode = str(analysis_mode or "full").strip().lower().replace("-", "_")
    if mode in ("full", "dbscan", "default"):
        run_dbscan_mode = True
    elif mode in ("ripley_only", "ripley"):
        run_dbscan_mode = False
    else:
        raise ValueError(f"Unknown analysis_mode: {analysis_mode!r} (expected 'full' or 'ripley_only')")

    # NOTE: Ripley-only mode intentionally skips any shapely-heavy cluster geometry
    # computations. The worker/tiling still requires shapely upstream to construct
    # valid RegionSpec windows.
    if run_dbscan_mode:
        _ensure_shapely()

    pts_all = np.asarray(points_xy, dtype=float)
    if pts_all.ndim != 2 or pts_all.shape[1] != 2:
        raise ValueError("points_xy must be (N,2)")

    if ref_values is not None:
        rv_all = np.asarray(ref_values, dtype=float)
        if rv_all.shape[0] != pts_all.shape[0]:
            raise ValueError("ref_values must have the same length as points_xy")
    else:
        rv_all = None

    if points_z is not None:
        z_all = np.asarray(points_z, dtype=float)
        if z_all.shape[0] != pts_all.shape[0]:
            raise ValueError("points_z must have the same length as points_xy")
    else:
        z_all = None

    summary_rows: List[Dict] = []
    cluster_tables: List[pd.DataFrame] = []
    ripley_curves: Dict[str, pd.DataFrame] = {}
    cluster_ripley_rows: List[Dict] = []
    cluster_ripley_curves: Dict[str, pd.DataFrame] = {}

    # Stable empty cluster table (ensures output CSV has columns even when clustering is skipped).
    def _empty_cluster_table() -> pd.DataFrame:
        z_empty = None
        if points_z is not None and bool(getattr(dbscan, "use_z", False)):
            z_empty = np.zeros((0,), dtype=float)
        rv_empty = np.zeros((0,), dtype=float) if ref_values is not None else None
        return _cluster_table_generic(
            points_xy=np.zeros((0, 2), dtype=float),
            points_z=z_empty,
            labels=np.zeros((0,), dtype=int),
            region_name="",
            axis="",
            d_start=float("nan"),
            d_end=float("nan"),
            ref_values=rv_empty,
            ref_name=ref_name,
        ).iloc[0:0]

    for region in regions:
        if region is None or region.area <= 0 or region.parts is None:
            continue

        in_region = mask_points_in_poly(region.parts, pts_all)
        pts_r = pts_all[in_region]
        rv_r = rv_all[in_region] if rv_all is not None else None
        z_r = z_all[in_region] if z_all is not None else None

        # Guard: remove non-finite coordinates and, if provided, non-finite ref values.
        keep = np.all(np.isfinite(pts_r), axis=1)
        if run_dbscan_mode and z_r is not None and bool(getattr(dbscan, "use_z", False)):
            keep &= np.isfinite(z_r)
        if rv_r is not None:
            keep &= np.isfinite(rv_r)

        pts_r = pts_r[keep]
        if rv_r is not None:
            rv_r = rv_r[keep]
        if z_r is not None:
            z_r = z_r[keep]

        if run_dbscan_mode:
            # Cluster in 2D or 2.5D depending on dbscan.use_z and availability of z
            if z_r is not None and bool(getattr(dbscan, "use_z", False)):
                feats = np.column_stack([pts_r, z_r])
            else:
                feats = pts_r

            labels = run_dbscan(feats, dbscan)

            ct = _cluster_table_generic(
                points_xy=pts_r,
                points_z=z_r,
                labels=labels,
                region_name=region.name,
                axis=region.axis,
                d_start=region.d_start,
                d_end=region.d_end,
                ref_values=rv_r,
                ref_name=ref_name,
            )
            cluster_tables.append(ct)

        else:
            # Ripley-only mode: no clustering performed. We still provide a labels
            # array (all noise) for downstream exports (e.g. points_labeled.csv).
            labels = np.full((pts_r.shape[0],), -1, dtype=int)

        if region_callback is not None:
            try:
                # Backward-compatible callback signature:
                #   old: cb(region, pts_xy, labels, ref_values)
                #   new: cb(region, pts_xy, labels, ref_values, z_values)
                try:
                    region_callback(region, pts_r, labels, rv_r, z_r)
                except TypeError:
                    region_callback(region, pts_r, labels, rv_r)
            except Exception:
                # Never let plotting/export crash the analysis.
                if logger is not None:
                    try:
                        logger.warning(f"Region callback failed for {region.name}; continuing.")
                    except Exception:
                        pass

        n_points = int(pts_r.shape[0])
        if run_dbscan_mode:
            n_clusters = int(len(set(labels.tolist())) - (1 if -1 in labels else 0))
            frac_in = float(np.mean(labels >= 0)) if n_points > 0 else 0.0
            sizes = [int(np.sum(labels == cid)) for cid in set(labels.tolist()) if cid >= 0]
            mean_sz = float(np.mean(sizes)) if sizes else 0.0
            med_sz = float(np.median(sizes)) if sizes else 0.0
        else:
            # Prefer NaNs over zeros to avoid misleading plots.
            n_clusters = np.nan
            frac_in = np.nan
            mean_sz = np.nan
            med_sz = np.nan

        density_area = float(n_points / region.area) if region.area > 0 else float("nan")
        density_len = float(n_points / region.skeleton_length) if np.isfinite(region.skeleton_length) and region.skeleton_length > 0 else float("nan")

        r_peak = lmr_peak = auc_pos = None
        if ripley is not None:
            curve, summ = _ripley_for_region(points_xy=pts_r, region=region, ripley=ripley, csr=csr, logger=logger)
            ripley_curves[str(region.name)] = curve
            # Always record summary stats for the region-level Ripley curve.
            r_peak = float(summ.get("r_peak", np.nan))
            lmr_peak = float(summ.get("lmr_peak", np.nan))
            auc_pos = float(summ.get("auc_pos", np.nan))

        # Optional: Ripley per DBSCAN cluster inside this region (local-window hull buffered by eps)
        if run_dbscan_mode and (ripley is not None) and compute_cluster_ripley and (pts_r.shape[0] >= 2):
            for cid in sorted({int(x) for x in set(labels.tolist()) if int(x) >= 0}):
                idx_c = labels == cid
                pts_c = pts_r[idx_c]
                if pts_c.shape[0] < int(cluster_ripley_min_points):
                    continue
                curve_c, summ_c, area_c = _ripley_for_cluster_region(
                    points_xy=pts_c,
                    parent_region=region,
                    dbscan_eps=float(dbscan.eps),
                    ripley=ripley,
                    csr=csr,
                    logger=logger,
                )
                if curve_c is None:
                    continue
                key = f"{region.name}__cluster_{cid}"
                cluster_ripley_curves[key] = curve_c
                cluster_ripley_rows.append(
                    {
                        "region": str(region.name),
                        "axis": str(region.axis),
                        "d_start": float(region.d_start),
                        "d_end": float(region.d_end),
                        "cluster_id": int(cid),
                        "n_points": int(pts_c.shape[0]),
                        "window_area": float(area_c),
                        "dbscan_eps": float(dbscan.eps),
                        "ripley_r_peak": float(summ_c.get("r_peak", np.nan)),
                        "ripley_lmr_peak": float(summ_c.get("lmr_peak", np.nan)),
                        "ripley_auc_pos": float(summ_c.get("auc_pos", np.nan)),
                    }
                )



        row = {
            "region": str(region.name),
            "axis": str(region.axis),
            "d_start": float(region.d_start),
            "d_end": float(region.d_end),
            "n_points": n_points,
            "area": float(region.area),
            "skeleton_length": float(region.skeleton_length),
            "density_area": density_area,
            "density_length": density_len,
            "dbscan_eps": float(dbscan.eps),
            "dbscan_min_samples": int(dbscan.min_samples),
            "dbscan_ran": bool(run_dbscan_mode),
            "n_clusters": n_clusters,
            "frac_points_in_clusters": frac_in,
            "mean_cluster_size": mean_sz,
            "median_cluster_size": med_sz,
            "ripley_r_peak": r_peak,
            "ripley_lmr_peak": lmr_peak,
            "ripley_auc_pos": auc_pos,
        }

        if rv_r is not None:
            row[f"mean_{ref_name}"] = float(np.nanmean(rv_r)) if rv_r.size else np.nan
            row[f"median_{ref_name}"] = float(np.nanmedian(rv_r)) if rv_r.size else np.nan

        summary_rows.append(row)

    region_summary = pd.DataFrame(summary_rows)

    clusters = pd.concat(cluster_tables, ignore_index=True) if cluster_tables else _empty_cluster_table()

    superclusters = None
    if run_dbscan_mode:
        superclusters = run_hierarchical_dbscan(clusters.rename(columns={"region": "band"}), hierarchical)
        # run_hierarchical_dbscan expects column 'band'; we map back.
        if superclusters is not None and not superclusters.empty:
            superclusters = superclusters.rename(columns={"band": "region"})

    cluster_ripley_summary = pd.DataFrame(cluster_ripley_rows) if cluster_ripley_rows else None
    cluster_ripley_curves_out = cluster_ripley_curves if cluster_ripley_curves else None

    return RegionAnalysisOutput(
        region_summary=region_summary,
        clusters=clusters,
        superclusters=superclusters,
        ripley_curves=ripley_curves,
        cluster_ripley_summary=cluster_ripley_summary,
        cluster_ripley_curves=cluster_ripley_curves_out,
    )
