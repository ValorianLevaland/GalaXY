"""galaxy_posthoc_report.py

Post-hoc reporting for GalaXY_2 outputs.

This script reads a GalaXY_2 *output folder* (the folder that contains
`run_config.json` and the aggregated CSVs such as `all_rois__clusters.csv`).

It generates:
  - enriched tables (with derived metrics like equivalent diameter, NN distances)
  - per-channel and per-region summary CSVs (median/IQR/quantiles)
  - histogram PNGs for:
      * #clusters per region
      * cluster size (#points per cluster)
      * cluster equivalent diameter (2 * radius_eq)
      * nearest-neighbor distances between cluster centroids
      * region-vs-global NN distance ratios

GUI
---
The default entrypoint launches a small GUI (PyQt5 if available; otherwise a
tkinter fallback) to pick the output folder and run the report.

Notes
-----
"Global" spacing comparison is computed **within each ROI and channel** by
pooling all clusters in that ROI+channel, then comparing each region's median
NN distance to that ROI+channel global median.

Dependencies
------------
This file is intentionally independent from the GalaXY_2 GUI/Napari stack.
It only requires: numpy, pandas, scipy, matplotlib.

License: MIT
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import sys
import traceback
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from scipy.spatial import cKDTree


# ----------------------------
# Small utilities
# ----------------------------


def _now_slug() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _read_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _is_finite_series(s: pd.Series) -> pd.Series:
    a = pd.to_numeric(s, errors="coerce")
    return np.isfinite(a.to_numpy(dtype=float, copy=False))


def _sanitize_filename(name: str) -> str:
    # Keep it close to what worker.py does, but do not import anything.
    name = str(name)
    name = name.strip().replace(" ", "_")
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name[:200] if len(name) > 200 else name


def _quantiles(x: np.ndarray, qs: Sequence[float]) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"q{int(q * 100):02d}": float("nan") for q in qs}
    out = {}
    for q in qs:
        out[f"q{int(q * 100):02d}"] = float(np.quantile(x, q))
    return out


def _summary_stats(series: pd.Series) -> Dict[str, float]:
    x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float, copy=False)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "median": float("nan"),
            "max": float("nan"),
            "iqr": float("nan"),
            **_quantiles(x, (0.05, 0.25, 0.75, 0.95)),
        }

    q25, q50, q75 = np.quantile(x, [0.25, 0.50, 0.75])
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if x.size >= 2 else float("nan"),
        "min": float(np.min(x)),
        "median": float(q50),
        "max": float(np.max(x)),
        "iqr": float(q75 - q25),
        **_quantiles(x, (0.05, 0.25, 0.75, 0.95)),
    }


def _try_import_matplotlib():
    import matplotlib

    # Use non-interactive backend for robustness (headless runs)
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _auto_bins(x: np.ndarray, *, max_bins: int = 80) -> int:
    """Reasonable histogram bin count with a cap (Freedman–Diaconis-ish)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return 10
    q25, q75 = np.quantile(x, [0.25, 0.75])
    iqr = float(q75 - q25)
    if iqr <= 0:
        return min(max_bins, 30)
    bw = 2 * iqr * (x.size ** (-1 / 3))
    if bw <= 0:
        return min(max_bins, 30)
    nb = int(np.ceil((np.max(x) - np.min(x)) / bw))
    return int(np.clip(nb, 10, max_bins))


def _plot_hist(
    *,
    series: pd.Series,
    out_png: str,
    title: str,
    xlabel: str,
    ylabel: str = "Count",
    max_bins: int = 80,
) -> None:
    plt = _try_import_matplotlib()

    x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float, copy=False)
    x = x[np.isfinite(x)]
    if x.size == 0:
        # Still emit an empty figure to avoid breaking pipelines
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        fig.savefig(out_png, dpi=220)
        plt.close(fig)
        return

    bins = _auto_bins(x, max_bins=max_bins)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(x, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


# ----------------------------
# Loading GalaXY outputs
# ----------------------------


@dataclass(frozen=True)
class GalaxyTables:
    out_dir: str
    run_config: Optional[Dict]
    region_summary: pd.DataFrame
    clusters: pd.DataFrame
    superclusters: Optional[pd.DataFrame] = None


def _load_aggregated_tables(out_dir: str) -> Tuple[Optional[Dict], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    cfg = None
    cfg_path = os.path.join(out_dir, "run_config.json")
    if os.path.exists(cfg_path):
        try:
            cfg = _read_json(cfg_path)
        except Exception:
            cfg = None

    rs_path = os.path.join(out_dir, "all_rois__region_summary.csv")
    cl_path = os.path.join(out_dir, "all_rois__clusters.csv")
    sc_path = os.path.join(out_dir, "all_rois__superclusters.csv")

    rs = _read_csv(rs_path) if os.path.exists(rs_path) else None
    cl = _read_csv(cl_path) if os.path.exists(cl_path) else None
    sc = _read_csv(sc_path) if os.path.exists(sc_path) else None

    return cfg, rs, cl, sc


def load_galaxy_output(out_dir: str) -> GalaxyTables:
    """Load GalaXY_2 output tables.

    Prefers the top-level aggregated CSVs. If they do not exist, scans ROI/channel
    subfolders and concatenates per-channel tables.
    """
    out_dir = os.path.abspath(out_dir)
    if not os.path.isdir(out_dir):
        raise FileNotFoundError(f"Not a folder: {out_dir}")

    cfg, rs, cl, sc = _load_aggregated_tables(out_dir)
    if rs is not None and cl is not None:
        return GalaxyTables(out_dir=out_dir, run_config=cfg, region_summary=rs, clusters=cl, superclusters=sc)

    # Fallback: scan per-ROI / per-channel
    roi_dirs = [
        os.path.join(out_dir, d)
        for d in os.listdir(out_dir)
        if os.path.isdir(os.path.join(out_dir, d)) and d.startswith("roi_")
    ]
    if not roi_dirs:
        raise FileNotFoundError(
            "Could not find aggregated tables (all_rois__*.csv) and no roi_* subfolders were found. "
            "Please point to the GalaXY_2 output folder."
        )

    all_rs: List[pd.DataFrame] = []
    all_cl: List[pd.DataFrame] = []
    all_sc: List[pd.DataFrame] = []

    for roi_path in sorted(roi_dirs):
        roi_dirname = os.path.basename(roi_path)
        roi_name = roi_dirname
        # roi_XX__name pattern
        m = re.match(r"^(roi_[^_]+)__?(.*)$", roi_dirname)
        if m and m.group(2):
            roi_name = m.group(2)

        ch_dirs = [
            os.path.join(roi_path, d)
            for d in os.listdir(roi_path)
            if os.path.isdir(os.path.join(roi_path, d)) and d.startswith("channel_")
        ]
        if not ch_dirs:
            # single-channel runs might still create channel_all
            continue

        for ch_path in sorted(ch_dirs):
            ch_dirname = os.path.basename(ch_path)
            ch_alias = ch_dirname.replace("channel_", "")
            rs_path = os.path.join(ch_path, "region_summary.csv")
            cl_path = os.path.join(ch_path, "clusters.csv")
            sc_path = os.path.join(ch_path, "superclusters.csv")

            if os.path.exists(rs_path):
                df = _read_csv(rs_path)
                df.insert(0, "roi", roi_dirname)
                df.insert(1, "roi_name", roi_name)
                df.insert(2, "channel", ch_alias)
                df.insert(3, "channel_raw", ch_alias)
                all_rs.append(df)

            if os.path.exists(cl_path):
                df = _read_csv(cl_path)
                df.insert(0, "roi", roi_dirname)
                df.insert(1, "roi_name", roi_name)
                df.insert(2, "channel", ch_alias)
                df.insert(3, "channel_raw", ch_alias)
                all_cl.append(df)

            if os.path.exists(sc_path):
                df = _read_csv(sc_path)
                df.insert(0, "roi", roi_dirname)
                df.insert(1, "roi_name", roi_name)
                df.insert(2, "channel", ch_alias)
                df.insert(3, "channel_raw", ch_alias)
                all_sc.append(df)

    rs_out = pd.concat(all_rs, ignore_index=True) if all_rs else pd.DataFrame()
    cl_out = pd.concat(all_cl, ignore_index=True) if all_cl else pd.DataFrame()
    sc_out = pd.concat(all_sc, ignore_index=True) if all_sc else None

    return GalaxyTables(out_dir=out_dir, run_config=cfg, region_summary=rs_out, clusters=cl_out, superclusters=sc_out)


# ----------------------------
# Derived metrics
# ----------------------------


def enrich_region_summary(region_summary: pd.DataFrame) -> pd.DataFrame:
    rs = region_summary.copy()

    # Convenience densities for clusters, not points
    if "n_clusters" in rs.columns and "area" in rs.columns:
        rs["cluster_density_area"] = pd.to_numeric(rs["n_clusters"], errors="coerce") / pd.to_numeric(rs["area"], errors="coerce")
    if "n_clusters" in rs.columns and "skeleton_length" in rs.columns:
        rs["cluster_density_length"] = pd.to_numeric(rs["n_clusters"], errors="coerce") / pd.to_numeric(rs["skeleton_length"], errors="coerce")

    return rs


def enrich_clusters(clusters: pd.DataFrame) -> pd.DataFrame:
    cl = clusters.copy()
    if "radius_eq" in cl.columns:
        cl["diameter_eq"] = 2.0 * pd.to_numeric(cl["radius_eq"], errors="coerce")
    if "bbox_w" in cl.columns and "bbox_h" in cl.columns:
        w = pd.to_numeric(cl["bbox_w"], errors="coerce")
        h = pd.to_numeric(cl["bbox_h"], errors="coerce")
        cl["bbox_area"] = w * h
        # aspect ratio >= 1
        mn = np.minimum(w.to_numpy(dtype=float, copy=False), h.to_numpy(dtype=float, copy=False))
        mx = np.maximum(w.to_numpy(dtype=float, copy=False), h.to_numpy(dtype=float, copy=False))
        with np.errstate(divide="ignore", invalid="ignore"):
            ar = mx / mn
        cl["bbox_aspect"] = ar
    if "area_hull" in cl.columns and "n_points" in cl.columns:
        a = pd.to_numeric(cl["area_hull"], errors="coerce")
        n = pd.to_numeric(cl["n_points"], errors="coerce")
        with np.errstate(divide="ignore", invalid="ignore"):
            cl["hull_point_density"] = n / a
    return cl


def _compute_group_nn(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    *,
    x_col: str = "centroid_x",
    y_col: str = "centroid_y",
    out_col: str = "nn_dist",
) -> pd.Series:
    """Return a Series aligned to df.index with per-row NN distances within group."""
    if df.empty:
        return pd.Series([], dtype=float)
    if x_col not in df.columns or y_col not in df.columns:
        return pd.Series(np.full((len(df),), np.nan), index=df.index, dtype=float)

    out = pd.Series(np.full((len(df),), np.nan), index=df.index, dtype=float)

    for _, g in df.groupby(list(group_cols), sort=False):
        if g.shape[0] < 2:
            continue
        xy = g[[x_col, y_col]].to_numpy(dtype=float)
        finite = np.all(np.isfinite(xy), axis=1)
        if np.sum(finite) < 2:
            continue
        xy_f = xy[finite]
        idx_f = g.index.to_numpy()[finite]

        tree = cKDTree(xy_f)
        # k=2: [self, nearest other]
        d, _ = tree.query(xy_f, k=2)
        nn = d[:, 1]
        out.loc[idx_f] = nn

    return out


def compute_nn_distances(clusters: pd.DataFrame) -> pd.DataFrame:
    """Add NN distance columns to clusters table.

    Adds:
      - nn_region: nearest-neighbor distance within ROI+channel+region
      - nn_roi_channel: nearest-neighbor distance within ROI+channel (all regions pooled)
    """
    cl = clusters.copy()

    required = ["roi", "channel", "region", "centroid_x", "centroid_y"]
    missing = [c for c in required if c not in cl.columns]
    if missing:
        # Still return without crashing.
        cl["nn_region"] = np.nan
        cl["nn_roi_channel"] = np.nan
        return cl

    cl["nn_region"] = _compute_group_nn(cl, ["roi", "channel", "region"], out_col="nn_region")
    cl["nn_roi_channel"] = _compute_group_nn(cl, ["roi", "channel"], out_col="nn_roi_channel")
    return cl


# ----------------------------
# Summaries
# ----------------------------


def summarize_by_group(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    metrics: Sequence[Tuple[str, str]],
    *,
    prefix: str = "",
) -> pd.DataFrame:
    """Summarize numerical columns by group.

    Parameters
    ----------
    df:
        Input dataframe.
    group_cols:
        Columns to group by.
    metrics:
        List of (output_name, source_column).
    prefix:
        Optional prefix added to each output_name.
    """
    rows = []
    if df.empty:
        return pd.DataFrame()

    for keys, g in df.groupby(list(group_cols), dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(group_cols, keys)}

        for out_name, src_col in metrics:
            if src_col not in g.columns:
                stats = _summary_stats(pd.Series([], dtype=float))
            else:
                stats = _summary_stats(g[src_col])
            for k, v in stats.items():
                row[f"{prefix}{out_name}__{k}"] = v

        rows.append(row)

    return pd.DataFrame(rows)


def compute_region_nn_vs_global(
    clusters_nn: pd.DataFrame,
    region_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-(ROI, channel, region) NN distance summaries and ratios.

    Returns a table with:
      - region-level median NN distance (within-region)
      - ROI+channel global median NN distance (all regions pooled)
      - ratio (region/global)
    """
    cl = clusters_nn.copy()
    if cl.empty or "nn_region" not in cl.columns or "nn_roi_channel" not in cl.columns:
        return pd.DataFrame()

    # Region-level NN stats: aggregate NN distances per region
    reg = (
        cl.groupby(["roi", "roi_name", "channel", "region"], dropna=False)
        .agg(
            n_clusters=("cluster_id", "count"),
            nn_region_median=("nn_region", "median"),
            nn_region_mean=("nn_region", "mean"),
            nn_roi_channel_median=("nn_roi_channel", "median"),
            nn_roi_channel_mean=("nn_roi_channel", "mean"),
        )
        .reset_index()
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        reg["nn_median_ratio_region_vs_global"] = reg["nn_region_median"] / reg["nn_roi_channel_median"]
        reg["nn_mean_ratio_region_vs_global"] = reg["nn_region_mean"] / reg["nn_roi_channel_mean"]

    # Bring axis/d_start/d_end and n_clusters from region_summary if available
    if not region_summary.empty and "region" in region_summary.columns:
        rs_cols = [c for c in ["roi", "roi_name", "channel", "region", "axis", "d_start", "d_end", "n_clusters", "area", "skeleton_length"] if c in region_summary.columns]
        rs_sub = region_summary[rs_cols].drop_duplicates()
        reg = reg.merge(rs_sub, on=[c for c in ["roi", "roi_name", "channel", "region"] if c in rs_sub.columns], how="left")

    return reg


# ----------------------------
# Report generation
# ----------------------------


@dataclass(frozen=True)
class ReportOptions:
    report_name: str = "posthoc_report"
    per_region_name_figures: bool = True
    max_region_names: int = 80
    per_channel_figures: bool = True
    include_nn_analysis: bool = True
    max_bins: int = 80


def generate_posthoc_report(out_dir: str, *, options: Optional[ReportOptions] = None, log_fn=None) -> str:
    """Generate the report and return the report folder path."""
    if options is None:
        options = ReportOptions()

    def log(msg: str) -> None:
        if log_fn is not None:
            try:
                log_fn(msg)
            except Exception:
                pass
        else:
            print(msg)

    log(f"Loading GalaXY_2 outputs from: {out_dir}")
    tables = load_galaxy_output(out_dir)

    rs = enrich_region_summary(tables.region_summary)
    cl = enrich_clusters(tables.clusters)

    if options.include_nn_analysis:
        log("Computing nearest-neighbor distances between cluster centroids…")
        cl = compute_nn_distances(cl)

    # Report folder
    report_dir = os.path.join(tables.out_dir, f"{options.report_name}__{_now_slug()}")
    fig_dir = os.path.join(report_dir, "figures")
    tab_dir = os.path.join(report_dir, "tables")
    _safe_mkdir(fig_dir)
    _safe_mkdir(tab_dir)

    # Save enriched tables
    log("Writing enriched tables…")
    rs.to_csv(os.path.join(tab_dir, "region_summary__enriched.csv"), index=False)
    cl.to_csv(os.path.join(tab_dir, "clusters__enriched.csv"), index=False)
    if tables.superclusters is not None:
        try:
            tables.superclusters.to_csv(os.path.join(tab_dir, "superclusters.csv"), index=False)
        except Exception:
            pass
    if tables.run_config is not None:
        try:
            with open(os.path.join(report_dir, "run_config.json"), "w", encoding="utf-8") as f:
                json.dump(tables.run_config, f, indent=2)
        except Exception:
            pass

    # Global summaries
    log("Computing summary tables…")
    summary_global = pd.DataFrame(
        [
            {
                "scope": "global",
                "n_rois": int(rs["roi"].nunique()) if "roi" in rs.columns else np.nan,
                "n_channels": int(rs["channel"].nunique()) if "channel" in rs.columns else np.nan,
                "n_region_rows": int(len(rs)),
                "n_cluster_rows": int(len(cl)),
            }
        ]
    )
    summary_global.to_csv(os.path.join(tab_dir, "summary_global.csv"), index=False)

    # Per-channel summaries
    if "channel" in rs.columns:
        per_channel_rs = summarize_by_group(
            rs,
            ["channel"],
            metrics=[
                ("n_clusters_per_region", "n_clusters"),
                ("cluster_density_area", "cluster_density_area"),
                ("cluster_density_length", "cluster_density_length"),
                ("frac_points_in_clusters", "frac_points_in_clusters"),
            ],
        )
    else:
        per_channel_rs = pd.DataFrame()

    if "channel" in cl.columns:
        per_channel_cl = summarize_by_group(
            cl,
            ["channel"],
            metrics=[
                ("cluster_size", "n_points"),
                ("diameter_eq", "diameter_eq"),
                ("area_hull", "area_hull"),
                ("bbox_aspect", "bbox_aspect"),
                ("nn_region", "nn_region"),
                ("nn_roi_channel", "nn_roi_channel"),
            ],
        )
    else:
        per_channel_cl = pd.DataFrame()

    if not per_channel_rs.empty and not per_channel_cl.empty:
        per_channel = per_channel_rs.merge(per_channel_cl, on=["channel"], how="outer")
    else:
        per_channel = per_channel_rs if not per_channel_rs.empty else per_channel_cl

    if not per_channel.empty:
        per_channel.to_csv(os.path.join(tab_dir, "summary_by_channel.csv"), index=False)

    # Per-region-name summaries (across ROIs)
    # This matches the user's "per regions" request.
    group_cols = [c for c in ["channel", "region"] if c in rs.columns]
    if group_cols:
        by_region_rs = summarize_by_group(
            rs,
            group_cols,
            metrics=[
                ("n_clusters_per_roi", "n_clusters"),
                ("cluster_density_area", "cluster_density_area"),
                ("cluster_density_length", "cluster_density_length"),
                ("frac_points_in_clusters", "frac_points_in_clusters"),
            ],
        )
    else:
        by_region_rs = pd.DataFrame()

    group_cols_cl = [c for c in ["channel", "region"] if c in cl.columns]
    if group_cols_cl:
        by_region_cl = summarize_by_group(
            cl,
            group_cols_cl,
            metrics=[
                ("cluster_size", "n_points"),
                ("diameter_eq", "diameter_eq"),
                ("nn_region", "nn_region"),
            ],
        )
    else:
        by_region_cl = pd.DataFrame()

    if not by_region_rs.empty and not by_region_cl.empty:
        by_region = by_region_rs.merge(by_region_cl, on=group_cols, how="outer")
    else:
        by_region = by_region_rs if not by_region_rs.empty else by_region_cl

    if not by_region.empty:
        by_region.to_csv(os.path.join(tab_dir, "summary_by_region.csv"), index=False)

    # NN comparisons: region vs ROI+channel pooled
    nn_comp = pd.DataFrame()
    if options.include_nn_analysis and ("nn_region" in cl.columns) and ("nn_roi_channel" in cl.columns):
        nn_comp = compute_region_nn_vs_global(cl, rs)
        if not nn_comp.empty:
            nn_comp.to_csv(os.path.join(tab_dir, "nn_region_vs_global_by_roi.csv"), index=False)

    # ----------------------------
    # Figures
    # ----------------------------
    log("Generating figures…")

    # Helper to make per-channel figure dirs
    def ch_figdir(ch: str) -> str:
        d = os.path.join(fig_dir, f"channel_{_sanitize_filename(ch)}")
        _safe_mkdir(d)
        return d

    channels: List[str]
    if "channel" in rs.columns:
        channels = sorted({str(x) for x in rs["channel"].dropna().unique().tolist()})
    elif "channel" in cl.columns:
        channels = sorted({str(x) for x in cl["channel"].dropna().unique().tolist()})
    else:
        channels = ["all"]

    for ch in channels:
        if options.per_channel_figures and ("channel" in rs.columns or "channel" in cl.columns):
            rs_ch = rs[rs["channel"].astype(str) == str(ch)] if "channel" in rs.columns else rs
            cl_ch = cl[cl["channel"].astype(str) == str(ch)] if "channel" in cl.columns else cl
        else:
            rs_ch = rs
            cl_ch = cl

        dch = ch_figdir(ch) if len(channels) > 1 else fig_dir

        # 1) Number of clusters per (ROI, region)
        if "n_clusters" in rs_ch.columns:
            _plot_hist(
                series=rs_ch["n_clusters"],
                out_png=os.path.join(dch, "hist__n_clusters_per_region_row.png"),
                title=f"# clusters per region (rows) — channel={ch}",
                xlabel="# clusters",
                max_bins=options.max_bins,
            )

        # 2) Cluster size
        if "n_points" in cl_ch.columns:
            _plot_hist(
                series=cl_ch["n_points"],
                out_png=os.path.join(dch, "hist__cluster_size_n_points.png"),
                title=f"Cluster size (#points) — channel={ch}",
                xlabel="# points per cluster",
                max_bins=options.max_bins,
            )

        # 3) Equivalent diameter
        if "diameter_eq" in cl_ch.columns:
            _plot_hist(
                series=cl_ch["diameter_eq"],
                out_png=os.path.join(dch, "hist__cluster_diameter_eq.png"),
                title=f"Cluster equivalent diameter (2×radius_eq) — channel={ch}",
                xlabel="Equivalent diameter (same units as x/y)",
                max_bins=options.max_bins,
            )

        # 4) NN distances
        if options.include_nn_analysis and ("nn_region" in cl_ch.columns):
            _plot_hist(
                series=cl_ch["nn_region"],
                out_png=os.path.join(dch, "hist__nn_distance_within_region.png"),
                title=f"Cluster centroid NN distance (within region) — channel={ch}",
                xlabel="NN distance",
                max_bins=options.max_bins,
            )
        if options.include_nn_analysis and ("nn_roi_channel" in cl_ch.columns):
            _plot_hist(
                series=cl_ch["nn_roi_channel"],
                out_png=os.path.join(dch, "hist__nn_distance_within_roi_channel.png"),
                title=f"Cluster centroid NN distance (ROI+channel pooled) — channel={ch}",
                xlabel="NN distance",
                max_bins=options.max_bins,
            )

        # 5) Region-vs-global NN ratio
        if options.include_nn_analysis and (not nn_comp.empty):
            sub = nn_comp[nn_comp["channel"].astype(str) == str(ch)] if "channel" in nn_comp.columns else nn_comp
            if "nn_median_ratio_region_vs_global" in sub.columns:
                _plot_hist(
                    series=sub["nn_median_ratio_region_vs_global"],
                    out_png=os.path.join(dch, "hist__nn_ratio_region_vs_global_median.png"),
                    title=f"NN distance ratio: region/global (median) — channel={ch}",
                    xlabel="median(NN_region) / median(NN_global)",
                    max_bins=options.max_bins,
                )

        # Per region name histograms (cluster size + diameter + nn)
        if options.per_region_name_figures and ("region" in rs_ch.columns or "region" in cl_ch.columns):
            # Determine region name order by frequency
            if "region" in rs_ch.columns:
                reg_names = rs_ch["region"].astype(str).value_counts().index.tolist()
            else:
                reg_names = cl_ch["region"].astype(str).value_counts().index.tolist()
            reg_names = reg_names[: int(options.max_region_names)]

            per_reg_dir = os.path.join(dch, "per_region")
            _safe_mkdir(per_reg_dir)

            for rname in reg_names:
                # --- n_clusters per ROI for that region
                if "region" in rs_ch.columns and "n_clusters" in rs_ch.columns:
                    rs_r = rs_ch[rs_ch["region"].astype(str) == str(rname)]
                    if not rs_r.empty:
                        _plot_hist(
                            series=rs_r["n_clusters"],
                            out_png=os.path.join(per_reg_dir, f"hist__n_clusters__{_sanitize_filename(rname)}.png"),
                            title=f"# clusters per ROI — region={rname} — channel={ch}",
                            xlabel="# clusters",
                            max_bins=options.max_bins,
                        )

                # --- cluster sizes / diameters / NN within that region
                if "region" in cl_ch.columns:
                    cl_r = cl_ch[cl_ch["region"].astype(str) == str(rname)]
                    if not cl_r.empty:
                        if "n_points" in cl_r.columns:
                            _plot_hist(
                                series=cl_r["n_points"],
                                out_png=os.path.join(per_reg_dir, f"hist__cluster_size__{_sanitize_filename(rname)}.png"),
                                title=f"Cluster size (#points) — region={rname} — channel={ch}",
                                xlabel="# points per cluster",
                                max_bins=options.max_bins,
                            )
                        if "diameter_eq" in cl_r.columns:
                            _plot_hist(
                                series=cl_r["diameter_eq"],
                                out_png=os.path.join(per_reg_dir, f"hist__diameter_eq__{_sanitize_filename(rname)}.png"),
                                title=f"Cluster diameter_eq — region={rname} — channel={ch}",
                                xlabel="Equivalent diameter",
                                max_bins=options.max_bins,
                            )
                        if options.include_nn_analysis and ("nn_region" in cl_r.columns):
                            _plot_hist(
                                series=cl_r["nn_region"],
                                out_png=os.path.join(per_reg_dir, f"hist__nn_region__{_sanitize_filename(rname)}.png"),
                                title=f"Cluster NN distance within region — region={rname} — channel={ch}",
                                xlabel="NN distance",
                                max_bins=options.max_bins,
                            )

    # Minimal HTML index for convenience
    try:
        _write_index_html(report_dir)
    except Exception:
        pass

    log(f"Done. Report written to: {report_dir}")
    return report_dir


def _write_index_html(report_dir: str) -> None:
    fig_dir = os.path.join(report_dir, "figures")
    tab_dir = os.path.join(report_dir, "tables")
    if not os.path.isdir(fig_dir):
        return

    # Collect assets
    pngs: List[str] = []
    for root, _, files in os.walk(fig_dir):
        for fn in sorted(files):
            if fn.lower().endswith(".png"):
                rel = os.path.relpath(os.path.join(root, fn), report_dir)
                pngs.append(rel.replace(os.sep, "/"))

    csvs: List[str] = []
    if os.path.isdir(tab_dir):
        for fn in sorted(os.listdir(tab_dir)):
            if fn.lower().endswith(".csv"):
                rel = os.path.relpath(os.path.join(tab_dir, fn), report_dir)
                csvs.append(rel.replace(os.sep, "/"))

    html = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>GalaXY_2 posthoc report</title>",
        "<style>body{font-family:Arial,Helvetica,sans-serif;margin:24px} img{max-width:900px;width:100%;height:auto;border:1px solid #ddd} .card{margin:18px 0;padding:12px;border:1px solid #eee;border-radius:8px} code{background:#f7f7f7;padding:2px 4px;border-radius:4px}</style>",
        "</head><body>",
        f"<h1>GalaXY_2 posthoc report</h1>",
        f"<p>Generated: {_dt.datetime.now().isoformat()}</p>",
        "<h2>Tables</h2>",
        "<ul>",
    ]
    for rel in csvs:
        html.append(f"<li><a href='{_html.escape(rel)}'>{_html.escape(rel)}</a></li>")
    html += ["</ul>", "<h2>Figures</h2>"]

    for rel in pngs:
        html += [
            "<div class='card'>",
            f"<div><code>{_html.escape(rel)}</code></div>",
            f"<div><img src='{_html.escape(rel)}'></div>",
            "</div>",
        ]

    html += ["</body></html>"]
    with open(os.path.join(report_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write("\n".join(html))


# ----------------------------
# GUI wrappers
# ----------------------------


def _launch_gui_qt() -> None:
    """PyQt5 GUI (preferred if installed)."""
    from PyQt5 import QtCore, QtWidgets

    class Window(QtWidgets.QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("GalaXY_2 — Posthoc report")
            self.setMinimumWidth(720)

            self.path_edit = QtWidgets.QLineEdit()
            self.path_edit.setPlaceholderText("Select GalaXY_2 output folder (contains run_config.json / all_rois__*.csv)")

            browse_btn = QtWidgets.QPushButton("Browse…")
            browse_btn.clicked.connect(self._browse)

            self.chk_per_region = QtWidgets.QCheckBox("Generate per-region-name histograms")
            self.chk_per_region.setChecked(True)

            self.chk_nn = QtWidgets.QCheckBox("Compute NN distances + region-vs-global comparisons")
            self.chk_nn.setChecked(True)

            self.max_regions = QtWidgets.QSpinBox()
            self.max_regions.setRange(1, 500)
            self.max_regions.setValue(80)
            self.max_regions.setSingleStep(10)
            max_regions_row = QtWidgets.QHBoxLayout()
            max_regions_row.addWidget(QtWidgets.QLabel("Max region names:"))
            max_regions_row.addWidget(self.max_regions)
            max_regions_row.addStretch(1)

            run_btn = QtWidgets.QPushButton("Generate report")
            run_btn.clicked.connect(self._run)

            self.log = QtWidgets.QPlainTextEdit()
            self.log.setReadOnly(True)
            self.log.setMaximumBlockCount(2000)

            top = QtWidgets.QHBoxLayout()
            top.addWidget(self.path_edit, 1)
            top.addWidget(browse_btn)

            layout = QtWidgets.QVBoxLayout(self)
            layout.addLayout(top)
            layout.addWidget(self.chk_per_region)
            layout.addWidget(self.chk_nn)
            layout.addLayout(max_regions_row)
            layout.addWidget(run_btn)
            layout.addWidget(self.log, 1)

        def _append(self, s: str) -> None:
            self.log.appendPlainText(str(s))

        def _browse(self) -> None:
            p = QtWidgets.QFileDialog.getExistingDirectory(self, "Select GalaXY_2 output folder")
            if p:
                self.path_edit.setText(p)

        def _run(self) -> None:
            out_dir = self.path_edit.text().strip()
            if not out_dir:
                QtWidgets.QMessageBox.warning(self, "Missing folder", "Please select a GalaXY_2 output folder.")
                return
            if not os.path.isdir(out_dir):
                QtWidgets.QMessageBox.critical(self, "Invalid folder", f"Not a folder: {out_dir}")
                return

            opts = ReportOptions(
                per_region_name_figures=bool(self.chk_per_region.isChecked()),
                max_region_names=int(self.max_regions.value()),
                include_nn_analysis=bool(self.chk_nn.isChecked()),
            )
            self._append("—" * 60)
            try:
                report_dir = generate_posthoc_report(out_dir, options=opts, log_fn=self._append)
                QtWidgets.QMessageBox.information(
                    self,
                    "Done",
                    f"Report generated in:\n{report_dir}\n\nOpen index.html for a quick overview.",
                )
            except Exception as e:
                tb = traceback.format_exc()
                self._append(tb)
                QtWidgets.QMessageBox.critical(self, "Report failed", f"{e}\n\nSee log for details.")

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    w = Window()
    w.show()
    app.exec_()


def _launch_gui_tk() -> None:
    """tkinter fallback GUI (in case Qt is not available)."""
    import tkinter as tk
    from tkinter import filedialog, messagebox

    root = tk.Tk()
    root.title("GalaXY_2 — Posthoc report")

    frm = tk.Frame(root, padx=10, pady=10)
    frm.pack(fill="both", expand=True)

    path_var = tk.StringVar(value="")

    def browse():
        p = filedialog.askdirectory(title="Select GalaXY_2 output folder")
        if p:
            path_var.set(p)

    tk.Label(frm, text="GalaXY_2 output folder:").grid(row=0, column=0, sticky="w")
    tk.Entry(frm, textvariable=path_var, width=70).grid(row=1, column=0, sticky="ew")
    tk.Button(frm, text="Browse…", command=browse).grid(row=1, column=1, padx=(8, 0))

    per_region_var = tk.BooleanVar(value=True)
    nn_var = tk.BooleanVar(value=True)
    max_regions_var = tk.IntVar(value=80)

    tk.Checkbutton(frm, text="Generate per-region-name histograms", variable=per_region_var).grid(row=2, column=0, sticky="w", pady=(8, 0))
    tk.Checkbutton(frm, text="Compute NN distances + region-vs-global comparisons", variable=nn_var).grid(row=3, column=0, sticky="w")

    mr_row = tk.Frame(frm)
    mr_row.grid(row=4, column=0, sticky="w", pady=(6, 0))
    tk.Label(mr_row, text="Max region names:").pack(side="left")
    tk.Spinbox(mr_row, from_=1, to=500, increment=10, textvariable=max_regions_var, width=6).pack(side="left", padx=(6, 0))

    log = tk.Text(frm, height=18)
    log.grid(row=6, column=0, columnspan=2, sticky="nsew", pady=(10, 0))
    frm.grid_columnconfigure(0, weight=1)
    frm.grid_rowconfigure(6, weight=1)

    def append(s: str):
        log.insert("end", str(s) + "\n")
        log.see("end")

    def run():
        out_dir = path_var.get().strip()
        if not out_dir:
            messagebox.showwarning("Missing folder", "Please select a GalaXY_2 output folder.")
            return
        if not os.path.isdir(out_dir):
            messagebox.showerror("Invalid folder", f"Not a folder: {out_dir}")
            return
        opts = ReportOptions(
            per_region_name_figures=bool(per_region_var.get()),
            max_region_names=int(max_regions_var.get()),
            include_nn_analysis=bool(nn_var.get()),
        )
        append("—" * 60)
        try:
            report_dir = generate_posthoc_report(out_dir, options=opts, log_fn=append)
            messagebox.showinfo("Done", f"Report generated in:\n{report_dir}\n\nOpen index.html for a quick overview.")
        except Exception as e:
            append(traceback.format_exc())
            messagebox.showerror("Report failed", str(e))

    tk.Button(frm, text="Generate report", command=run).grid(row=5, column=0, sticky="w", pady=(10, 0))

    root.mainloop()


def main() -> None:
    """Launch GUI (Qt preferred, tkinter fallback)."""
    try:
        _launch_gui_qt()
    except Exception:
        # If Qt isn't available, still provide a GUI.
        _launch_gui_tk()


if __name__ == "__main__":
    main()
