"""galaxy.worker

Qt worker thread for running GalaXY analyses without freezing the UI.

This follows the same pattern as the previous membrane app:
- The GUI collects inputs (points, ROIs, parameters)
- The worker performs CPU-heavy geometry + clustering + figure export
- Results are written to an output directory with a full audit trail

License: MIT
"""

from __future__ import annotations



import os
import json
import traceback
import datetime as _dt
import itertools
from dataclasses import asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from qtpy import QtCore

try:
    import shapely.geometry as shgeom
    import shapely.ops as shops
except Exception:  # pragma: no cover
    shgeom = None
    shops = None

from .audit import save_run_config, QtLogHandler
from .profiles import get_profile
from .tiling import (
    WindowBinningParams,
    build_domain_geometry,
    build_regions_for_profile,
    union_geoms,
)
from .reference import (
    ref_distance_to_centroid,
    ref_distance_to_domain_boundary,
    ref_distance_to_outer_boundary,
    ref_distance_to_nucleus,
    ref_distance_to_seed,
    ref_geodesic_from_model,
)
from .analysis import analyze_regions
from .analysis_core import DBSCANParams, HierarchicalDBSCANParams
from .ripley_backend import (
    RipleyParams,
    CSRParams,
    make_radii,
    shapely_to_parts,
    mask_points_in_poly,
    distance_to_boundary as _ripley_distance_to_boundary,
    estimate_K_L,
    csr_envelope_LminusR,
    estimate_cross_K_L,
    csr_envelope_cross_LminusR,
)
from .geometry_base import GeometryParams, BandParams

from .figures import plot_overview_galaxy, plot_summary_vs_axis, plot_ripley_summary_vs_axis, plot_band_points_only
from .figures_legacy import plot_band_dbscan


def _ensure_shapely() -> None:
    if shgeom is None or shops is None:
        raise ImportError("shapely is required for GalaXY")


def _sanitize_filename(s: str) -> str:
    keep = []
    for ch in str(s):
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep).strip("_")
    return out[:200] if out else "item"


def _centroid_xy(geom) -> Tuple[float, float]:
    c = geom.centroid
    return float(c.x), float(c.y)


def _maybe_downsample_points(points_xy: np.ndarray, *, max_points: int, random_downsample: Optional[int], seed: int = 0) -> np.ndarray:
    """Downsample points for expensive statistics (Ripley) to keep runtime bounded."""
    pts = np.asarray(points_xy, dtype=float)
    n = int(pts.shape[0])
    if n <= 0:
        return pts

    target = None
    if random_downsample is not None and int(random_downsample) > 0:
        target = int(random_downsample)
    elif max_points is not None and int(max_points) > 0 and n > int(max_points):
        target = int(max_points)

    if target is None or n <= target:
        return pts

    rng = np.random.default_rng(int(seed))
    idx = rng.choice(n, size=int(target), replace=False)
    return pts[idx]


def _compute_ripley_curve_df(
    *,
    points_xy: np.ndarray,
    window_geom,
    ripley: RipleyParams,
    csr: Optional[CSRParams],
    logger: Optional[QtLogHandler] = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Compute Ripley K(r) and Besag L(r)-r for points in a given window geometry.

    Notes
    -----
    - `window_geom` must be a shapely Polygon/MultiPolygon (holes allowed).
    - Returns a dataframe with columns: r, K, L, L_minus_r, n_eligible, env_* (optional).
    """
    _ensure_shapely()

    pts = _maybe_downsample_points(
        np.asarray(points_xy, dtype=float),
        max_points=int(getattr(ripley, "max_points", 200_000)),
        random_downsample=getattr(ripley, "random_downsample", None),
        seed=int(seed),
    )

    radii = make_radii(float(ripley.r_min), float(ripley.r_max), float(ripley.dr))

    # Build window parts (needed for border correction + CSR)
    if window_geom is None or getattr(window_geom, "is_empty", True):
        return pd.DataFrame(
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

    try:
        wgeom = window_geom.buffer(0)
    except Exception:
        wgeom = window_geom

    area = float(getattr(wgeom, "area", 0.0))
    if area <= 0 or pts.shape[0] < 2:
        return pd.DataFrame(
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

    parts = tuple(shapely_to_parts(wgeom))

    bd = None
    if str(ripley.edge_correction).lower().strip() == "border":
        bd = _ripley_distance_to_boundary(parts, pts)

    est = estimate_K_L(
        pts,
        float(area),
        radii,
        edge_correction=str(ripley.edge_correction),
        boundary_dist=bd,
        logger=logger,
    )

    env_lo = np.full_like(radii, np.nan, dtype=float)
    env_hi = np.full_like(radii, np.nan, dtype=float)
    env_mean = np.full_like(radii, np.nan, dtype=float)

    if csr is not None and int(csr.n_simulations) > 0 and pts.shape[0] >= 2:
        env = csr_envelope_LminusR(
            window_parts=parts,
            window_area=float(area),
            n_points=int(pts.shape[0]),
            radii=radii,
            edge_correction=str(ripley.edge_correction),
            n_simulations=int(csr.n_simulations),
            alpha=float(csr.alpha),
            seed=csr.seed,
            logger=logger,
        )
        env_lo = env["Lmr_lo"]
        env_hi = env["Lmr_hi"]
        env_mean = env["Lmr_mean"]

    return pd.DataFrame(
        {
            "r": radii,
            "K": est["K"],
            "L": est["L"],
            "L_minus_r": est["L_minus_r"],
            "n_eligible": est["n_eligible"],
            "env_LminusR_lo": env_lo,
            "env_LminusR_hi": env_hi,
            "env_LminusR_mean": env_mean,
        }
    )


def _plot_ripley_curve_png(
    *,
    ripley_df: pd.DataFrame,
    out_path: str,
    title: str = "",
    dpi: int = 220,
):
    """Save a Ripley curve figure (K(r) + L(r)-r) to PNG."""
    import matplotlib.pyplot as plt

    if ripley_df is None or ripley_df.empty:
        return

    r = ripley_df["r"].to_numpy(dtype=float)
    K = ripley_df["K"].to_numpy(dtype=float) if "K" in ripley_df.columns else None
    lmr = ripley_df["L_minus_r"].to_numpy(dtype=float) if "L_minus_r" in ripley_df.columns else None

    lo = ripley_df["env_LminusR_lo"].to_numpy(dtype=float) if "env_LminusR_lo" in ripley_df.columns else None
    hi = ripley_df["env_LminusR_hi"].to_numpy(dtype=float) if "env_LminusR_hi" in ripley_df.columns else None
    mean = ripley_df["env_LminusR_mean"].to_numpy(dtype=float) if "env_LminusR_mean" in ripley_df.columns else None

    fig = plt.figure(figsize=(7.2, 6.2))
    ax0 = fig.add_subplot(211)
    ax1 = fig.add_subplot(212, sharex=ax0)

    if K is not None:
        ax0.plot(r, K, linewidth=1.6)
        ax0.set_ylabel("K(r)")
        ax0.grid(True, alpha=0.25)
    else:
        ax0.axis("off")

    if lmr is not None:
        ax1.plot(r, lmr, linewidth=1.6)
        if lo is not None and hi is not None and np.any(np.isfinite(lo)) and np.any(np.isfinite(hi)):
            ax1.fill_between(r, lo, hi, alpha=0.18)
        if mean is not None and np.any(np.isfinite(mean)):
            ax1.plot(r, mean, linewidth=1.2, linestyle="--")
        ax1.axhline(0.0, linewidth=1.0, alpha=0.6)
        ax1.set_xlabel("r")
        ax1.set_ylabel("L(r)-r")
        ax1.grid(True, alpha=0.25)

    if title:
        fig.suptitle(title)

    fig.tight_layout(rect=[0, 0, 1, 0.96] if title else None)
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)


def _summarize_ripley_lmr(r: np.ndarray, lmr: np.ndarray) -> Dict[str, float]:
    """Return summary metrics for a Ripley/Besag L(r)-r curve.

    Metrics
    -------
    - r_peak: radius at which L(r)-r is maximal
    - lmr_peak: maximal L(r)-r value
    - auc_pos: area under the *positive* part of L(r)-r
    """

    r = np.asarray(r, dtype=float)
    y = np.asarray(lmr, dtype=float)
    m = np.isfinite(r) & np.isfinite(y)
    if r.size == 0 or not np.any(m):
        return {"r_peak": float("nan"), "lmr_peak": float("nan"), "auc_pos": float("nan")}

    rr = r[m]
    yy = y[m]
    k = int(np.argmax(yy))
    r_peak = float(rr[k])
    lmr_peak = float(yy[k])
    auc_pos = float(np.trapz(np.maximum(yy, 0.0), rr))
    return {"r_peak": r_peak, "lmr_peak": lmr_peak, "auc_pos": auc_pos}


def _compute_cross_ripley_curve_df(
    *,
    points1_xy: np.ndarray,
    points2_xy: np.ndarray,
    window_geom,
    ripley: RipleyParams,
    csr: Optional[CSRParams],
    logger: Optional[QtLogHandler] = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Compute cross-Ripley K12(r) + L12(r)-r for two point sets in a window."""

    _ensure_shapely()

    # Downsample (independently) to keep runtime bounded.
    max_pts = int(getattr(ripley, "max_points", 200_000))
    rand_ds = getattr(ripley, "random_downsample", None)

    p1 = _maybe_downsample_points(np.asarray(points1_xy, dtype=float), max_points=max_pts, random_downsample=rand_ds, seed=int(seed))
    p2 = _maybe_downsample_points(np.asarray(points2_xy, dtype=float), max_points=max_pts, random_downsample=rand_ds, seed=int(seed) + 1)

    radii = make_radii(float(ripley.r_min), float(ripley.r_max), float(ripley.dr))

    # Empty / invalid window
    if window_geom is None or getattr(window_geom, "is_empty", True):
        return pd.DataFrame(
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

    try:
        wgeom = window_geom.buffer(0)
    except Exception:
        wgeom = window_geom

    area = float(getattr(wgeom, "area", 0.0))
    if area <= 0 or p1.shape[0] < 1 or p2.shape[0] < 1:
        return pd.DataFrame(
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

    parts = tuple(shapely_to_parts(wgeom))

    bd1 = None
    if str(ripley.edge_correction).lower().strip() == "border":
        bd1 = _ripley_distance_to_boundary(parts, p1)

    est = estimate_cross_K_L(
        p1,
        p2,
        float(area),
        radii,
        edge_correction=str(ripley.edge_correction),
        boundary_dist1=bd1,
        logger=logger,
    )

    env_lo = np.full_like(radii, np.nan, dtype=float)
    env_hi = np.full_like(radii, np.nan, dtype=float)
    env_mean = np.full_like(radii, np.nan, dtype=float)

    if csr is not None and int(csr.n_simulations) > 0 and p1.shape[0] >= 1 and p2.shape[0] >= 1:
        env = csr_envelope_cross_LminusR(
            window_parts=parts,
            window_area=float(area),
            points1_xy=p1,
            n_points2=int(p2.shape[0]),
            radii=radii,
            edge_correction=str(ripley.edge_correction),
            n_simulations=int(csr.n_simulations),
            alpha=float(csr.alpha),
            seed=csr.seed,
            logger=logger,
        )
        env_lo = env["Lmr_lo"]
        env_hi = env["Lmr_hi"]
        env_mean = env["Lmr_mean"]

    return pd.DataFrame(
        {
            "r": radii,
            "K": est["K"],
            "L": est["L"],
            "L_minus_r": est["L_minus_r"],
            "n_eligible": est["n_eligible"],
            "env_LminusR_lo": env_lo,
            "env_LminusR_hi": env_hi,
            "env_LminusR_mean": env_mean,
        }
    )


def _assign_children_to_domains(
    domains: List[dict],
    children: List[dict],
    *,
    one_to_all_if_single_child: bool,
) -> Dict[int, List[dict]]:
    """Assign child ROIs (holes or seeds) to domains.

    Strategy
    --------
    1) If one child and multiple domains and one_to_all_if_single_child: reuse it for all.
    2) Else, assign by containment of child centroid in domain.
    3) Else, nearest domain centroid.

    Returns
    -------
    mapping: domain_index -> list of child dicts
    """

    mapping: Dict[int, List[dict]] = {i: [] for i in range(len(domains))}
    if not domains or not children:
        return mapping

    if one_to_all_if_single_child and len(children) == 1 and len(domains) >= 1:
        for i in range(len(domains)):
            mapping[i].append(children[0])
        return mapping

    dom_centroids = [
        _centroid_xy(d["geom"]) if d.get("geom") is not None else (np.nan, np.nan)
        for d in domains
    ]

    for child in children:
        cg = child.get("geom")
        if cg is None or getattr(cg, "is_empty", True):
            continue

        cc = cg.centroid
        assigned = None
        for i, d in enumerate(domains):
            dg = d.get("geom")
            if dg is None or getattr(dg, "is_empty", True):
                continue
            try:
                if dg.contains(cc):
                    assigned = i
                    break
            except Exception:
                pass

        if assigned is None:
            # Nearest domain centroid
            cx, cy = float(cc.x), float(cc.y)
            d2 = []
            for (dx, dy) in dom_centroids:
                if not np.isfinite(dx) or not np.isfinite(dy):
                    d2.append(np.inf)
                else:
                    d2.append((cx - dx) ** 2 + (cy - dy) ** 2)
            assigned = int(np.argmin(d2)) if d2 else 0

        mapping[assigned].append(child)

    return mapping


def _axis_for_binning_mode(profile_id: str, tiling_mode: str) -> Tuple[str, str]:
    """Return (axis_name, ref_name) given a profile and tiling mode."""

    m = str(tiling_mode).lower().strip()
    if m in ("radial", "radial_shells"):
        return "r_centroid", "r_centroid"
    if m in ("boundary", "boundary_shells"):
        return "d_boundary", "d_boundary"
    if m in ("outer", "outer_shells"):
        return "d_outer", "d_outer"
    if m in ("perinuclear", "perinuclear_shells"):
        return "d_nucleus", "d_nucleus"
    if m in ("seed", "seed_bands"):
        return "d_seed", "d_seed"
    if m in ("geodesic_bands",):
        return "d_geodesic", "d_geodesic"
    if m in ("grid",):
        return "tile", "tile"
    # fallback
    return "distance", "distance"


class GalaXYWorker(QtCore.QObject):
    """Run one GalaXY analysis batch (potentially multiple domains/cells)."""

    finished = QtCore.Signal(str)
    failed = QtCore.Signal(str)
    progress = QtCore.Signal(int)

    def __init__(
        self,
        *,
        points_xy: np.ndarray,
        points_z: Optional[np.ndarray],
        channel_labels: Optional[np.ndarray],
        channels_to_use: Optional[List[str]],
        channel_alias_map: Optional[Dict[str, str]],
        domains: List[dict],
        holes: List[dict],
        seeds: List[dict],
        profile_id: str,
        tiling_mode: str,
        geom_backend: str,
        geom_params: GeometryParams,
        band_params: BandParams,
        binning_params: WindowBinningParams,
        dbscan_params: DBSCANParams,
        hier_params: HierarchicalDBSCANParams,
        do_ripley: bool,
        ripley_params: RipleyParams,
        csr_params: Optional[CSRParams],
        do_cross_ripley: bool,
        out_dir: str,
        input_csv: str,
        x_col: str,
        y_col: str,
        z_col: str,
        channel_col: str,
        logger: QtLogHandler,
        save_overview_figures: bool,
        save_region_figures: bool,
        save_region_ripley_figures: bool,
        save_band_isolation_figures: bool,
        compute_cluster_ripley: bool,
        save_cluster_ripley_figures: bool,
        cluster_ripley_min_points: int,
        export_region_points: bool,
        fig_max_points: int,
        run_mode: str = "full",
    ):
        super().__init__()
        self.points_xy = np.asarray(points_xy, dtype=float)
        self.points_z = None if points_z is None else np.asarray(points_z, dtype=float)
        self.channel_labels = None if channel_labels is None else np.asarray(channel_labels, dtype=str)
        self.channels_to_use = None if channels_to_use is None else [str(x) for x in list(channels_to_use)]
        self.channel_alias_map = {str(k): str(v) for k, v in (channel_alias_map or {}).items()}

        if self.points_z is not None and len(self.points_z) != len(self.points_xy):
            raise ValueError("points_z must have the same length as points_xy")
        if self.channel_labels is not None and len(self.channel_labels) != len(self.points_xy):
            raise ValueError("channel_labels must have the same length as points_xy")

        # Channels we will actually analyze.
        if self.channel_labels is None:
            self.channels_to_analyze = ["all"]
        else:
            ch_list = self.channels_to_use
            if ch_list is None:
                ch_list = sorted({str(v) for v in self.channel_labels.tolist()})
            # Deduplicate while preserving order.
            seen = set()
            out = []
            for c in ch_list:
                c = str(c)
                if c not in seen:
                    out.append(c)
                    seen.add(c)
            self.channels_to_analyze = out
        self.domains = list(domains)
        self.holes = list(holes)
        self.seeds = list(seeds)
        self.profile_id = str(profile_id)
        self.tiling_mode = str(tiling_mode)
        self.geom_backend = str(geom_backend)
        self.geom_params = geom_params
        self.band_params = band_params
        self.binning_params = binning_params
        self.dbscan_params = dbscan_params
        self.hier_params = hier_params
        self.do_ripley = bool(do_ripley)
        self.ripley_params = ripley_params
        self.csr_params = csr_params
        self.do_cross_ripley = bool(do_cross_ripley)
        self.out_dir = str(out_dir)
        self.input_csv = str(input_csv)
        self.x_col = str(x_col)
        self.y_col = str(y_col)
        self.z_col = str(z_col or "")
        self.channel_col = str(channel_col or "")
        self.logger = logger
        self.save_overview_figures = bool(save_overview_figures)
        self.save_region_figures = bool(save_region_figures)
        self.save_region_ripley_figures = bool(save_region_ripley_figures)
        self.save_band_isolation_figures = bool(save_band_isolation_figures)
        self.compute_cluster_ripley = bool(compute_cluster_ripley)
        self.save_cluster_ripley_figures = bool(save_cluster_ripley_figures)
        self.cluster_ripley_min_points = int(cluster_ripley_min_points)
        self.export_region_points = bool(export_region_points)
        self.fig_max_points = int(fig_max_points)
        self.run_mode = str(run_mode or "full")
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    @QtCore.Slot()
    def run(self) -> None:
        try:
            self._run_impl()
        except Exception:
            self.failed.emit(traceback.format_exc())

    def _run_impl(self) -> None:
        _ensure_shapely()
        os.makedirs(self.out_dir, exist_ok=True)

        profile = get_profile(self.profile_id)

        # Resolve tiling mode once (used for region construction). The GUI may pass an empty string.
        tiling_mode = self.tiling_mode.strip() if (self.tiling_mode and self.tiling_mode.strip()) else profile.default_tiling
        axis_name, ref_name = _axis_for_binning_mode(profile.id, tiling_mode)

        # ---- Pipeline mode (full vs Ripley-only)
        run_mode = str(getattr(self, "run_mode", "full") or "full").strip().lower().replace("-", "_")
        if run_mode in ("full", "default", "dbscan"):
            run_dbscan = True
        elif run_mode in ("ripley_only", "ripley"):
            run_dbscan = False
        else:
            raise ValueError(f"Unknown run_mode: {self.run_mode!r} (expected 'full' or 'ripley_only')")

        # Effective flags (what we will actually execute)
        do_ripley = bool(self.do_ripley) or (not run_dbscan)
        do_cross_ripley = bool(self.do_cross_ripley) and do_ripley

        save_region_dbscan_figures = bool(self.save_region_figures) and run_dbscan
        compute_cluster_ripley = bool(self.compute_cluster_ripley) and run_dbscan and do_ripley
        save_cluster_ripley_figures = bool(self.save_cluster_ripley_figures) and compute_cluster_ripley

        # Hierarchical clustering only makes sense if nano-clusters exist.
        hier_params = self.hier_params
        if not run_dbscan:
            hier_params = HierarchicalDBSCANParams(enabled=False, eps=float(self.hier_params.eps), min_samples=int(self.hier_params.min_samples))

        # Helpful warning: user enabled 2.5D DBSCAN but no Z column was provided.
        if run_dbscan and bool(getattr(self.dbscan_params, "use_z", False)) and self.points_z is None:
            try:
                self.logger.warning(
                    "DBSCAN 'Use Z (2.5D)' is enabled, but no Z values were provided; clustering will fall back to 2D."
                )
            except Exception:
                pass

        # ---- Save run config (top-level)
        cfg = {
            "app": "GalaXY_2",
            "version": "0.2.1",
            "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
            "run_mode": str(run_mode),
            "pipeline": {
                "run_dbscan": bool(run_dbscan),
                "run_hierarchical": bool(run_dbscan and bool(getattr(hier_params, "enabled", False))),
                "do_ripley": bool(do_ripley),
                "do_cross_ripley": bool(do_cross_ripley),
                "compute_cluster_ripley": bool(compute_cluster_ripley),
            },
            "profile": asdict(profile),
            "tiling_mode": str(tiling_mode),
            "axis": axis_name,
            "input_csv": self.input_csv,
            "data_columns": {
                "x_col": self.x_col,
                "y_col": self.y_col,
                "z_col": self.z_col,
                "channel_col": self.channel_col,
            },
            "geometry_backend": self.geom_backend,
            "geometry_params": asdict(self.geom_params),
            "band_params": asdict(self.band_params),
            "binning_params": asdict(self.binning_params),
            "dbscan_params": asdict(self.dbscan_params),
            "hierarchical_dbscan": asdict(hier_params),
            "do_ripley": bool(do_ripley),
            "do_cross_ripley": bool(do_cross_ripley),
            "ripley_params": asdict(self.ripley_params),
            "csr_params": asdict(self.csr_params) if self.csr_params else None,
            "channels": {
                "enabled": self.channel_labels is not None,
                "channels_to_use": self.channels_to_use,
                "channels_analyzed": list(self.channels_to_analyze),
                "alias_map": dict(self.channel_alias_map),
            },
            "z": {
                "enabled": self.points_z is not None,
            },
            "rois": {
                "domains": [
                    {
                        "name": d.get("name"),
                        "vertices_xy": d.get("vertices_xy"),
                        "wkt": getattr(d.get("geom"), "wkt", None),
                    }
                    for d in self.domains
                ],
                "holes": [
                    {
                        "name": h.get("name"),
                        "type": h.get("type"),
                        "vertices_xy": h.get("vertices_xy"),
                        "wkt": getattr(h.get("geom"), "wkt", None),
                    }
                    for h in self.holes
                ],
                "seeds": [
                    {
                        "name": s.get("name"),
                        "vertices_xy": s.get("vertices_xy"),
                        "wkt": getattr(s.get("geom"), "wkt", None),
                    }
                    for s in self.seeds
                ],
            },
            "outputs": {
                "save_overview_figures": bool(self.save_overview_figures),
                "save_region_dbscan_figures": bool(save_region_dbscan_figures),
                "save_region_ripley_figures": bool(self.save_region_ripley_figures),
                "compute_cluster_ripley": bool(compute_cluster_ripley),
                "save_cluster_ripley_figures": bool(save_cluster_ripley_figures),
                "cluster_ripley_min_points": int(self.cluster_ripley_min_points),
                "export_region_points": bool(self.export_region_points),
                "fig_max_points": int(self.fig_max_points),
                "dpi": 220,
            },
        }

        save_run_config(os.path.join(self.out_dir, "run_config.json"), cfg)
        try:
            self.logger.info(f"Saved run_config.json in {self.out_dir}")
        except Exception:
            pass

        if not self.domains:
            try:
                self.logger.warning("No domain ROIs selected. Nothing to do.")
            except Exception:
                pass
            self.finished.emit(self.out_dir)
            return

        # ---- Assign holes/seeds to domains
        holes_by_domain = _assign_children_to_domains(self.domains, self.holes, one_to_all_if_single_child=False)
        seeds_by_domain = _assign_children_to_domains(self.domains, self.seeds, one_to_all_if_single_child=True)

        all_region_rows: List[pd.DataFrame] = []
        all_cluster_rows: List[pd.DataFrame] = []
        all_supercluster_rows: List[pd.DataFrame] = []
        all_cross_rows: List[pd.DataFrame] = []

        n_domains = len(self.domains)
        n_channels = len(getattr(self, "channels_to_analyze", [])) or 1
        tasks_per_domain = n_channels + (1 if (do_ripley and do_cross_ripley) else 0)
        total_tasks = max(1, n_domains * tasks_per_domain)
        done_tasks = 0

        for idx, dom in enumerate(self.domains, start=1):
            if self._cancel:
                try:
                    self.logger.warning("Analysis canceled by user.")
                except Exception:
                    pass
                break

            roi_raw_name = str(dom.get("name", f"domain_{idx:02d}"))
            roi_slug = _sanitize_filename(roi_raw_name)
            roi_dirname = f"roi_{idx:02d}__{roi_slug}"
            cell_dir = os.path.join(self.out_dir, roi_dirname)
            os.makedirs(cell_dir, exist_ok=True)

            try:
                self.logger.info(f"--- ROI {idx}/{n_domains}: {roi_raw_name} ---")
            except Exception:
                pass

            outer_geom = dom.get("geom")
            if outer_geom is None or getattr(outer_geom, "is_empty", True):
                try:
                    self.logger.warning(f"Skipping ROI '{roi_raw_name}': empty geometry")
                except Exception:
                    pass
                done_tasks += tasks_per_domain
                self.progress.emit(int(100 * done_tasks / total_tasks))
                continue

            # ----- Holes/seeds assigned to this ROI
            holes_assigned = holes_by_domain.get(idx - 1, [])
            seeds_assigned = seeds_by_domain.get(idx - 1, [])

            hole_geoms = [h.get("geom") for h in holes_assigned if h.get("geom") is not None]
            seed_geoms = [s.get("geom") for s in seeds_assigned if s.get("geom") is not None]

            # Typed holes: nucleus-only union (fallback to all holes if none typed)
            nuc_geoms = []
            for h in holes_assigned:
                t = str(h.get("type", "")).strip().lower()
                if t in ("nucleus", "nuc", "nuclear", "nuclei"):
                    g = h.get("geom")
                    if g is not None and not getattr(g, "is_empty", True):
                        nuc_geoms.append(g)

            outer_union, holes_union, domain_geom = build_domain_geometry(outer_geoms=[outer_geom], hole_geoms=hole_geoms)
            seed_union = union_geoms(seed_geoms) if seed_geoms else shgeom.GeometryCollection([])
            nucleus_union = union_geoms(nuc_geoms) if nuc_geoms else holes_union

            # Validate profile requirements
            if profile.requires_holes and (holes_union is None or getattr(holes_union, "is_empty", True)):
                raise ValueError(
                    f"Profile '{profile.name}' requires holes (e.g., nucleus), but none were assigned to ROI '{roi_raw_name}'."
                )
            if profile.requires_seeds and (seed_union is None or getattr(seed_union, "is_empty", True)):
                raise ValueError(
                    f"Profile '{profile.name}' requires seeds, but no seeds were assigned to ROI '{roi_raw_name}'."
                )

            # Some profiles/tilings specifically want nucleus holes for the reference axis.
            holes_for_profile = holes_union
            if profile.id == "perinuclear" or str(tiling_mode).lower().startswith("perinuclear"):
                holes_for_profile = nucleus_union if nucleus_union is not None else holes_union

            # ---- Build regions/windows (shared across channels)
            binning = self.binning_params
            if binning is None:
                binning = WindowBinningParams(mode=str(tiling_mode))
            else:
                # Clone with updated mode
                binning = WindowBinningParams(**{**asdict(binning), "mode": str(tiling_mode)})

            regions, ctx = build_regions_for_profile(
                profile=profile,
                outer_geom=outer_union,
                holes_geom=holes_for_profile,
                domain_geom=domain_geom,
                seed_geom=seed_union,
                geom_backend=self.geom_backend,
                geom_params=self.geom_params,
                band_params=self.band_params,
                binning=binning,
                axis_override=axis_name,
                logger=self.logger,
            )

            if not regions:
                try:
                    self.logger.warning(f"No non-empty regions produced for ROI '{roi_raw_name}'.")
                except Exception:
                    pass
                done_tasks += tasks_per_domain
                self.progress.emit(int(100 * done_tasks / total_tasks))
                continue

            # Save region geometry WKT once per ROI
            try:
                rows = []
                for r in regions:
                    rows.append(
                        {
                            "region": r.name,
                            "axis": r.axis,
                            "d_start": r.d_start,
                            "d_end": r.d_end,
                            "area": r.area,
                            "skeleton_length": r.skeleton_length,
                            "wkt": getattr(r.geom, "wkt", ""),
                        }
                    )
                pd.DataFrame(rows).to_csv(os.path.join(cell_dir, "regions_wkt.csv"), index=False)
            except Exception:
                pass

            # Pre-filter points to this domain to speed up per-region masking.
            try:
                dom_parts = tuple(shapely_to_parts(domain_geom))
                in_dom = mask_points_in_poly(dom_parts, self.points_xy)
            except Exception:
                in_dom = np.ones((self.points_xy.shape[0],), dtype=bool)

            pts_dom = self.points_xy[in_dom]
            z_dom = self.points_z[in_dom] if self.points_z is not None else None
            ch_dom = self.channel_labels[in_dom] if self.channel_labels is not None else None

            # Compute reference values for *domain-filtered* points (used for export + axis plots)
            ref_dom = None
            if axis_name == "r_centroid":
                ref_dom = ref_distance_to_centroid(domain_geom=domain_geom, points_xy=pts_dom, mask_to_domain=False)
            elif axis_name == "d_boundary":
                ref_dom = ref_distance_to_domain_boundary(domain_geom=domain_geom, points_xy=pts_dom, mask_to_domain=False)
            elif axis_name == "d_outer":
                ref_dom = ref_distance_to_outer_boundary(
                    outer_geom=outer_union, domain_geom=domain_geom, points_xy=pts_dom, mask_to_domain=False
                )
            elif axis_name == "d_nucleus":
                ref_dom = ref_distance_to_nucleus(
                    nucleus_geom=holes_for_profile, domain_geom=domain_geom, points_xy=pts_dom, mask_to_domain=False
                )
            elif axis_name == "d_seed":
                ref_dom = ref_distance_to_seed(seed_geom=seed_union, domain_geom=domain_geom, points_xy=pts_dom, mask_to_domain=False)
            elif axis_name == "d_geodesic":
                model = ctx.get("model")
                if model is None:
                    raise RuntimeError("Geodesic profile produced no model in context")
                ref_dom = ref_geodesic_from_model(model=model, points_xy=pts_dom, mask_to_domain=False)
            else:
                ref_dom = None

            # ROI-level overview figures (all points, regardless of channel)
            if self.save_overview_figures:
                try:
                    fig_dir = os.path.join(cell_dir, "figures")
                    os.makedirs(fig_dir, exist_ok=True)

                    plot_overview_galaxy(
                        points_xy=pts_dom,
                        domain_geom=domain_geom,
                        seed_geom=seed_union,
                        regions=regions,
                        out_path=os.path.join(fig_dir, "overview_all.png"),
                        title=f"{roi_raw_name} — {profile.name}",
                        show_points=True,
                        max_points=self.fig_max_points,
                        seed=0,
                        dpi=220,
                    )
                    plot_overview_galaxy(
                        points_xy=np.zeros((0, 2), dtype=float),
                        domain_geom=domain_geom,
                        seed_geom=seed_union,
                        regions=regions,
                        out_path=os.path.join(fig_dir, "overview_regions.png"),
                        title=f"{roi_raw_name} — regions",
                        show_points=False,
                        max_points=self.fig_max_points,
                        seed=0,
                        dpi=220,
                    )
                except Exception:
                    pass

            # ---- Per-channel analysis
            channel_points: Dict[str, np.ndarray] = {}

            for raw_ch in list(self.channels_to_analyze):
                if self._cancel:
                    try:
                        self.logger.warning("Analysis canceled by user.")
                    except Exception:
                        pass
                    break

                # Determine alias + output dir name
                if self.channel_labels is None:
                    alias = "all"
                    raw_label = "all"
                    mask_ch = np.ones((pts_dom.shape[0],), dtype=bool)
                else:
                    raw_label = str(raw_ch)
                    alias = self.channel_alias_map.get(raw_label, raw_label)
                    mask_ch = (ch_dom == raw_label) if ch_dom is not None else np.ones((pts_dom.shape[0],), dtype=bool)

                alias_slug = _sanitize_filename(alias)
                ch_dir = os.path.join(cell_dir, f"channel_{alias_slug}")
                os.makedirs(ch_dir, exist_ok=True)

                pts_ch = pts_dom[mask_ch]
                z_ch = z_dom[mask_ch] if z_dom is not None else None
                ref_ch = ref_dom[mask_ch] if ref_dom is not None else None

                channel_points[str(raw_label)] = np.asarray(pts_ch, dtype=float)

                # ---- Region callback for figures + point export (per-channel)
                points_export_path = os.path.join(ch_dir, "points_labeled.csv")
                if self.export_region_points and os.path.exists(points_export_path):
                    try:
                        os.remove(points_export_path)
                    except Exception:
                        pass

                def _callback(region, pts_r, labels, rv_r, z_r=None, *, _ch_dir=ch_dir, _roi=roi_raw_name, _alias=alias, _raw=raw_label):
                    # Points export (append per region)
                    if self.export_region_points:
                        try:
                            cols = {
                                "x": np.asarray(pts_r[:, 0], dtype=float),
                                "y": np.asarray(pts_r[:, 1], dtype=float),
                                "region": str(region.name),
                                "label": np.asarray(labels, dtype=int),
                            }
                            if z_r is not None:
                                cols["z"] = np.asarray(z_r, dtype=float)
                            if self.channel_labels is not None:
                                cols["channel_raw"] = np.full((len(pts_r),), str(_raw), dtype=object)
                                cols["channel"] = np.full((len(pts_r),), str(_alias), dtype=object)
                            cols[ref_name] = rv_r if rv_r is not None else np.full((len(pts_r),), np.nan, dtype=float)
                            dfp = pd.DataFrame(cols)
                            dfp.to_csv(
                                points_export_path,
                                mode="a",
                                header=not os.path.exists(points_export_path),
                                index=False,
                            )
                        except Exception:
                            pass

                    # Per-band isolation PNG (DBSCAN-agnostic):
                    # polygon outline + points that belong to this band/region.
                    if self.save_band_isolation_figures:
                        try:
                            fig_dir = os.path.join(_ch_dir, "figures", "band_points")
                            os.makedirs(fig_dir, exist_ok=True)
                            out_png = os.path.join(fig_dir, f"band_{_sanitize_filename(region.name)}.png")
                            plot_band_points_only(
                                title=f"{_roi} — {_alias} — {region.name}",
                                band_geom=region.geom,
                                points_xy=pts_r,
                                out_path=out_png,
                                seed=0,
                            )
                        except Exception:
                            pass

                    # Per-region DBSCAN figure
                    if save_region_dbscan_figures:
                        try:
                            fig_dir = os.path.join(_ch_dir, "figures", "region_dbscan")
                            os.makedirs(fig_dir, exist_ok=True)
                            out_png = os.path.join(fig_dir, f"{_sanitize_filename(region.name)}.png")
                            plot_band_dbscan(
                                band_name=str(region.name),
                                band_geom=region.geom,
                                points_xy=pts_r,
                                labels=labels,
                                out_path=out_png,
                                title=f"{_roi} — {_alias} — {region.name}",
                                background_xy=None,
                                max_points=self.fig_max_points,
                                seed=0,
                                dpi=220,
                            )
                        except Exception:
                            pass

                    # Per-cluster Ripley (per DBSCAN cluster inside this region)
                    if do_ripley and compute_cluster_ripley:
                        try:
                            min_pts = int(self.cluster_ripley_min_points)
                            cluster_ids = sorted(set(int(x) for x in labels.tolist() if int(x) >= 0))

                            if cluster_ids:
                                cl_rip_dir = os.path.join(_ch_dir, "ripley_clusters")
                                cl_fig_dir = os.path.join(_ch_dir, "figures", "ripley_clusters")
                                made_rip_dir = False
                                made_fig_dir = False

                                n_saved = 0
                                n_skipped_small = 0
                                n_skipped_badwin = 0

                                for cid in cluster_ids:
                                    idx_c = np.where(np.asarray(labels, dtype=int) == int(cid))[0]
                                    if idx_c.size < min_pts:
                                        n_skipped_small += 1
                                        continue

                                    pts_c = np.asarray(pts_r[idx_c], dtype=float)
                                    if pts_c.shape[0] < 2:
                                        n_skipped_small += 1
                                        continue

                                    # Local window: convex hull buffered by eps, intersected with parent region
                                    try:
                                        hull = shgeom.MultiPoint([(float(x), float(y)) for x, y in pts_c]).convex_hull
                                    except Exception:
                                        n_skipped_badwin += 1
                                        continue

                                    try:
                                        w = hull.buffer(float(self.dbscan_params.eps)).intersection(region.geom)
                                    except Exception:
                                        w = hull.buffer(float(self.dbscan_params.eps))

                                    try:
                                        w = w.buffer(0)
                                    except Exception:
                                        pass

                                    if w is None or getattr(w, "is_empty", True) or float(getattr(w, "area", 0.0)) <= 0:
                                        n_skipped_badwin += 1
                                        continue

                                    curve = _compute_ripley_curve_df(
                                        points_xy=pts_c,
                                        window_geom=w,
                                        ripley=self.ripley_params,
                                        csr=self.csr_params if do_ripley else None,
                                        logger=self.logger,
                                        seed=0,
                                    )

                                    base = f"{_sanitize_filename(region.name)}__cluster_{int(cid):03d}"

                                    # Only create output dirs when we actually have something to write
                                    if not made_rip_dir:
                                        os.makedirs(cl_rip_dir, exist_ok=True)
                                        made_rip_dir = True

                                    csv_path = os.path.join(cl_rip_dir, f"ripley_{base}.csv")
                                    curve.to_csv(csv_path, index=False)

                                    if save_cluster_ripley_figures:
                                        if not made_fig_dir:
                                            os.makedirs(cl_fig_dir, exist_ok=True)
                                            made_fig_dir = True
                                        png_path = os.path.join(cl_fig_dir, f"ripley_{base}.png")
                                        _plot_ripley_curve_png(
                                            ripley_df=curve,
                                            out_path=png_path,
                                            title=f"Ripley — {_roi} — {_alias} — {region.name} — cluster {int(cid)}",
                                            dpi=220,
                                        )

                                    n_saved += 1

                                if n_saved == 0:
                                    try:
                                        self.logger.warning(
                                            f"Cluster Ripley: no outputs for ROI '{_roi}' channel '{_alias}' region '{region.name}' "
                                            f"(clusters={len(cluster_ids)}, skipped_small={n_skipped_small}, skipped_badwin={n_skipped_badwin}). "
                                            f"Tip: lower 'Cluster Ripley min points' (currently {min_pts})."
                                        )
                                    except Exception:
                                        pass
                        except Exception as e:
                            try:
                                self.logger.warning(f"Cluster Ripley failed for region {region.name}: {e}")
                            except Exception:
                                pass

                # ---- Run analysis for this channel
                out = analyze_regions(
                    points_xy=pts_ch,
                    points_z=z_ch,
                    regions=regions,
                    dbscan=self.dbscan_params,
                    hierarchical=hier_params,
                    ripley=self.ripley_params if do_ripley else None,
                    csr=self.csr_params if do_ripley else None,
                    ref_values=ref_ch,
                    ref_name=ref_name,
                    region_callback=_callback,
                    analysis_mode=("full" if run_dbscan else "ripley_only"),
                    logger=self.logger,
                )

                # ---- Save tables (per-channel)
                out.region_summary.to_csv(os.path.join(ch_dir, "region_summary.csv"), index=False)

                # Optional convenience: a slim Ripley-only table (keeps only Ripley-relevant columns)
                if (not run_dbscan) and do_ripley:
                    try:
                        keep_cols = [
                            "region",
                            "axis",
                            "d_start",
                            "d_end",
                            "n_points",
                            "area",
                            "skeleton_length",
                            "density_area",
                            "density_length",
                            "ripley_r_peak",
                            "ripley_lmr_peak",
                            "ripley_auc_pos",
                        ]
                        cols = [c for c in keep_cols if c in out.region_summary.columns]
                        out.region_summary.loc[:, cols].to_csv(os.path.join(ch_dir, "ripley_summary.csv"), index=False)
                    except Exception:
                        pass
                out.clusters.to_csv(os.path.join(ch_dir, "clusters.csv"), index=False)
                if out.superclusters is not None:
                    out.superclusters.to_csv(os.path.join(ch_dir, "superclusters.csv"), index=False)

                # Ripley curves (per region)
                if do_ripley and out.ripley_curves:
                    rip_dir = os.path.join(ch_dir, "ripley")
                    os.makedirs(rip_dir, exist_ok=True)
                    for rname, curve in out.ripley_curves.items():
                        curve.to_csv(os.path.join(rip_dir, f"ripley_{_sanitize_filename(rname)}.csv"), index=False)

                # Ripley figures per region (PNG)
                if do_ripley and out.ripley_curves and self.save_region_ripley_figures:
                    try:
                        fig_dir = os.path.join(ch_dir, "figures", "region_ripley")
                        os.makedirs(fig_dir, exist_ok=True)
                        for rname, curve in out.ripley_curves.items():
                            out_png = os.path.join(fig_dir, f"ripley_{_sanitize_filename(rname)}.png")
                            _plot_ripley_curve_png(
                                ripley_df=curve,
                                out_path=out_png,
                                title=f"Ripley — {roi_raw_name} — {alias} — {rname}",
                                dpi=220,
                            )
                    except Exception as e:
                        try:
                            self.logger.warning(f"Region Ripley figure export failed: {e}")
                        except Exception:
                            pass

                # Channel-level figures: overview + summary
                if self.save_overview_figures:
                    try:
                        fig_dir = os.path.join(ch_dir, "figures")
                        os.makedirs(fig_dir, exist_ok=True)
                        plot_overview_galaxy(
                            points_xy=pts_ch,
                            domain_geom=domain_geom,
                            seed_geom=seed_union,
                            regions=regions,
                            out_path=os.path.join(fig_dir, "overview.png"),
                            title=f"{roi_raw_name} — {alias}",
                            show_points=True,
                            max_points=self.fig_max_points,
                            seed=0,
                            dpi=220,
                        )
                        if run_dbscan:
                            plot_summary_vs_axis(
                                summary_df=out.region_summary,
                                out_path=os.path.join(fig_dir, "summary_vs_axis.png"),
                                axis_label=axis_name,
                                title=f"{roi_raw_name} — {alias} — summary",
                                dpi=220,
                            )
                        elif do_ripley:
                            plot_ripley_summary_vs_axis(
                                summary_df=out.region_summary,
                                out_path=os.path.join(fig_dir, "ripley_vs_axis.png"),
                                axis_label=axis_name,
                                title=f"{roi_raw_name} — {alias} — Ripley summary",
                                dpi=220,
                            )
                    except Exception:
                        pass

                # Collect aggregated
                try:
                    rs = out.region_summary.copy()
                    rs.insert(0, "roi", roi_dirname)
                    rs.insert(1, "roi_name", roi_raw_name)
                    rs.insert(2, "channel", alias)
                    rs.insert(3, "channel_raw", raw_label)
                    all_region_rows.append(rs)
                except Exception:
                    pass

                try:
                    cl = out.clusters.copy()
                    cl.insert(0, "roi", roi_dirname)
                    cl.insert(1, "roi_name", roi_raw_name)
                    cl.insert(2, "channel", alias)
                    cl.insert(3, "channel_raw", raw_label)
                    all_cluster_rows.append(cl)
                except Exception:
                    pass

                if out.superclusters is not None and not out.superclusters.empty:
                    try:
                        sc = out.superclusters.copy()
                        sc.insert(0, "roi", roi_dirname)
                        sc.insert(1, "roi_name", roi_raw_name)
                        sc.insert(2, "channel", alias)
                        sc.insert(3, "channel_raw", raw_label)
                        all_supercluster_rows.append(sc)
                    except Exception:
                        pass

                done_tasks += 1
                self.progress.emit(int(100 * done_tasks / total_tasks))

            # ---- Cross-Ripley across channels (ROI-level)
            if do_ripley and do_cross_ripley and (self.channel_labels is not None):
                try:
                    cross_channels = list(self.channels_to_use) if self.channels_to_use is not None else list(self.channels_to_analyze)
                    cross_channels = [str(c) for c in cross_channels]
                    # Need at least two channels
                    if len(cross_channels) >= 2:
                        cross_rows = []
                        cross_dir = os.path.join(cell_dir, "cross")
                        os.makedirs(cross_dir, exist_ok=True)

                        for a_raw, b_raw in itertools.combinations(cross_channels, 2):
                            if self._cancel:
                                break

                            pts_a_all = channel_points.get(str(a_raw), np.zeros((0, 2), dtype=float))
                            pts_b_all = channel_points.get(str(b_raw), np.zeros((0, 2), dtype=float))

                            a_alias = self.channel_alias_map.get(str(a_raw), str(a_raw))
                            b_alias = self.channel_alias_map.get(str(b_raw), str(b_raw))
                            a_slug = _sanitize_filename(a_alias)
                            b_slug = _sanitize_filename(b_alias)

                            pair_dir = os.path.join(cross_dir, f"{a_slug}__{b_slug}")

                            # Two directed estimates (A->B and B->A)
                            for direction, (src_raw, dst_raw, src_alias, dst_alias, pts1_all, pts2_all) in (
                                (
                                    "A_to_B",
                                    (str(a_raw), str(b_raw), str(a_alias), str(b_alias), pts_a_all, pts_b_all),
                                ),
                                (
                                    "B_to_A",
                                    (str(b_raw), str(a_raw), str(b_alias), str(a_alias), pts_b_all, pts_a_all),
                                ),
                            ):
                                if self._cancel:
                                    break

                                dir_dir = os.path.join(pair_dir, direction)
                                os.makedirs(dir_dir, exist_ok=True)

                                for region in regions:
                                    if self._cancel:
                                        break

                                    # Mask each channel to the current window
                                    m1 = mask_points_in_poly(region.parts, pts1_all) if pts1_all.size else np.zeros((0,), dtype=bool)
                                    m2 = mask_points_in_poly(region.parts, pts2_all) if pts2_all.size else np.zeros((0,), dtype=bool)
                                    pts1 = pts1_all[m1] if pts1_all.size else np.zeros((0, 2), dtype=float)
                                    pts2 = pts2_all[m2] if pts2_all.size else np.zeros((0, 2), dtype=float)

                                    n1 = int(pts1.shape[0])
                                    n2 = int(pts2.shape[0])

                                    # Compute only when both patterns are non-empty.
                                    curve = None
                                    summ = {"r_peak": float("nan"), "lmr_peak": float("nan"), "auc_pos": float("nan")}
                                    if n1 >= 1 and n2 >= 1:
                                        curve = _compute_cross_ripley_curve_df(
                                            points1_xy=pts1,
                                            points2_xy=pts2,
                                            window_geom=region.geom,
                                            ripley=self.ripley_params,
                                            csr=self.csr_params if do_ripley else None,
                                            logger=self.logger,
                                            seed=0,
                                        )
                                        try:
                                            summ = _summarize_ripley_lmr(curve["r"].to_numpy(dtype=float), curve["L_minus_r"].to_numpy(dtype=float))
                                        except Exception:
                                            summ = {"r_peak": float("nan"), "lmr_peak": float("nan"), "auc_pos": float("nan")}

                                        # Persist curve
                                        csv_path = os.path.join(dir_dir, f"cross_ripley_{_sanitize_filename(region.name)}.csv")
                                        curve.to_csv(csv_path, index=False)

                                        # Optional figure
                                        if self.save_region_ripley_figures:
                                            png_path = os.path.join(dir_dir, f"cross_ripley_{_sanitize_filename(region.name)}.png")
                                            _plot_ripley_curve_png(
                                                ripley_df=curve,
                                                out_path=png_path,
                                                title=f"Cross Ripley — {roi_raw_name} — {src_alias}→{dst_alias} — {region.name}",
                                                dpi=220,
                                            )

                                    cross_rows.append(
                                        {
                                            "roi": roi_dirname,
                                            "roi_name": roi_raw_name,
                                            "pair": f"{a_alias}__{b_alias}",
                                            "direction": str(direction),
                                            "channel1": str(src_alias),
                                            "channel2": str(dst_alias),
                                            "channel1_raw": str(src_raw),
                                            "channel2_raw": str(dst_raw),
                                            "region": str(region.name),
                                            "axis": str(region.axis),
                                            "d_start": float(region.d_start),
                                            "d_end": float(region.d_end),
                                            "area": float(region.area),
                                            "n_points1": n1,
                                            "n_points2": n2,
                                            "ripley_r_peak": float(summ.get("r_peak", np.nan)),
                                            "ripley_lmr_peak": float(summ.get("lmr_peak", np.nan)),
                                            "ripley_auc_pos": float(summ.get("auc_pos", np.nan)),
                                        }
                                    )

                        if cross_rows:
                            cross_df = pd.DataFrame(cross_rows)
                            cross_df.to_csv(os.path.join(cell_dir, "cross_ripley_summary.csv"), index=False)
                            all_cross_rows.append(cross_df)
                except Exception as e:
                    try:
                        self.logger.warning(f"Cross-Ripley failed for ROI '{roi_raw_name}': {e}")
                    except Exception:
                        pass

                done_tasks += 1
                self.progress.emit(int(100 * done_tasks / total_tasks))

        # ---- Aggregated outputs
        if all_region_rows:
            pd.concat(all_region_rows, ignore_index=True).to_csv(
                os.path.join(self.out_dir, "all_rois__region_summary.csv"), index=False
            )
        if all_cluster_rows:
            pd.concat(all_cluster_rows, ignore_index=True).to_csv(
                os.path.join(self.out_dir, "all_rois__clusters.csv"), index=False
            )
        if all_supercluster_rows:
            pd.concat(all_supercluster_rows, ignore_index=True).to_csv(
                os.path.join(self.out_dir, "all_rois__superclusters.csv"), index=False
            )
        if all_cross_rows:
            pd.concat(all_cross_rows, ignore_index=True).to_csv(
                os.path.join(self.out_dir, "all_rois__cross_ripley_summary.csv"), index=False
            )

        # Ensure the GUI reaches 100% even if we skipped some tasks.
        try:
            self.progress.emit(100)
        except Exception:
            pass

        self.finished.emit(self.out_dir)
