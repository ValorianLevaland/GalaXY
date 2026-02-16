"""galaxy.figures

Matplotlib figure helpers for GalaXY.

We reuse the figure primitives from the original membrane app but adapt them for
arbitrary topologies and distance axes.

All functions are pure (no napari dependency).

License: MIT
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np

import matplotlib.pyplot as plt

from .figures_legacy import (
    add_shapely_geom,
    plot_band_dbscan,
)

try:
    import shapely.geometry as shgeom
except Exception:  # pragma: no cover
    shgeom = None


def _ensure_shapely() -> None:
    if shgeom is None:
        raise ImportError("shapely is required for figure export")


def plot_overview_galaxy(
    *,
    points_xy: np.ndarray,
    domain_geom,
    seed_geom,
    regions: Sequence,
    out_path: str,
    title: str = "",
    show_points: bool = True,
    max_points: int = 200_000,
    seed: int = 0,
    dpi: int = 220,
):
    """Overview plot for any profile: domain + seeds + regions + points."""

    _ensure_shapely()

    pts = np.asarray(points_xy, dtype=float) if points_xy is not None else np.zeros((0, 2), dtype=float)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    # Domain outline
    add_shapely_geom(ax, domain_geom, facecolor=None, edgecolor="black", alpha=1.0, linewidth=1.0, zorder=2, fill=False)

    # Regions (semi-transparent)
    cmap = plt.cm.get_cmap("tab20")
    for k, r in enumerate(regions):
        geom = getattr(r, "geom", r)
        color = cmap(k % cmap.N)
        add_shapely_geom(ax, geom, facecolor=color, edgecolor=color, alpha=0.18, linewidth=0.8, zorder=3, fill=True)

    # Seeds
    if seed_geom is not None and (not getattr(seed_geom, "is_empty", True)):
        add_shapely_geom(ax, seed_geom, facecolor="red", edgecolor="red", alpha=0.35, linewidth=1.2, zorder=4, fill=True)

    # Points
    if show_points and pts.size > 0:
        n = int(pts.shape[0])
        if max_points is not None and max_points > 0 and n > max_points:
            rng = np.random.default_rng(int(seed))
            idx = rng.choice(n, size=int(max_points), replace=False)
            pts = pts[idx]
        ax.scatter(pts[:, 0], pts[:, 1], s=1.0, c="0.5", alpha=0.55, linewidths=0, rasterized=True, zorder=1)

    if title:
        ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")

    # Tight bounds around domain
    try:
        minx, miny, maxx, maxy = domain_geom.bounds
        pad = 0.02 * max(maxx - minx, maxy - miny)
        ax.set_xlim(minx - pad, maxx + pad)
        ax.set_ylim(miny - pad, maxy + pad)
    except Exception:
        pass

    fig.tight_layout()
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)


def plot_summary_vs_axis(
    *,
    summary_df,
    out_path: str,
    axis_label: str,
    title: str = "",
    dpi: int = 220,
):
    """Plot a compact set of summary metrics against the bin midpoints.

    This function is topology-agnostic:
    - If density_length is mostly NaN, it falls back to density_area.
    """

    import pandas as pd

    df = summary_df.copy()
    if df is None or len(df) == 0:
        return

    d0 = df["d_start"].to_numpy(dtype=float)
    d1 = df["d_end"].to_numpy(dtype=float)
    dmid = 0.5 * (d0 + d1)

    df = df.assign(_dmid=dmid)
    df = df.sort_values("_dmid")

    density_len = df.get("density_length")
    density_area = df.get("density_area")

    use_len = False
    if density_len is not None:
        arr = np.asarray(density_len, dtype=float)
        use_len = np.isfinite(arr).sum() >= max(2, int(0.5 * len(arr)))

    y_density = np.asarray(density_len if use_len else density_area, dtype=float)
    y_density_label = "density / length" if use_len else "density / area"

    fig = plt.figure(figsize=(9, 7))

    ax1 = fig.add_subplot(311)
    ax1.plot(df["_dmid"], y_density, marker="o", linewidth=1.5)
    ax1.set_ylabel(y_density_label)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(312, sharex=ax1)
    ax2.plot(df["_dmid"], df["frac_points_in_clusters"].to_numpy(dtype=float), marker="o", linewidth=1.5)
    ax2.set_ylabel("fraction in clusters")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(313, sharex=ax1)
    ax3.plot(df["_dmid"], df["mean_cluster_size"].to_numpy(dtype=float), marker="o", linewidth=1.5)
    ax3.set_xlabel(axis_label)
    ax3.set_ylabel("mean cluster size")
    ax3.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title)

    fig.tight_layout(rect=[0, 0, 1, 0.96] if title else None)
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)


def plot_ripley_curve(
    *,
    curve_df: pd.DataFrame,
    out_path: str,
    title: str = "",
    dpi: int = 220,
    show_envelope: bool = True,
    axis_label: str = "r",
    y_label: str = "L(r)-r",
) -> None:
    """Plot Ripley/Besag L(r)-r for a single region.

    The input ``curve_df`` is expected to contain at least:
      - r
      - L_minus_r

    Optional envelope columns (if present) are:
      - env_LminusR_lo
      - env_LminusR_hi
      - env_LminusR_mean
    """

    if curve_df is None or len(curve_df) == 0:
        return

    if "r" not in curve_df.columns or "L_minus_r" not in curve_df.columns:
        raise ValueError("curve_df must contain columns: 'r' and 'L_minus_r'")

    r = curve_df["r"].to_numpy(dtype=float)
    y = curve_df["L_minus_r"].to_numpy(dtype=float)

    m = np.isfinite(r) & np.isfinite(y)
    if not np.any(m):
        return
    r = r[m]
    y = y[m]

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))

    # Main curve
    ax.plot(r, y, label=y_label)

    # Optional CSR envelope
    if show_envelope and ("env_LminusR_lo" in curve_df.columns) and ("env_LminusR_hi" in curve_df.columns):
        lo = curve_df["env_LminusR_lo"].to_numpy(dtype=float)[m]
        hi = curve_df["env_LminusR_hi"].to_numpy(dtype=float)[m]
        if np.any(np.isfinite(lo)) and np.any(np.isfinite(hi)):
            ax.fill_between(r, lo, hi, alpha=0.2, label="CSR envelope")
        if "env_LminusR_mean" in curve_df.columns:
            mu = curve_df["env_LminusR_mean"].to_numpy(dtype=float)[m]
            if np.any(np.isfinite(mu)):
                ax.plot(r, mu, linestyle="--", linewidth=1.5, label="CSR mean")

    ax.axhline(0.0, linewidth=1.0)

    ax.set_xlabel(axis_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)

    ax.grid(True, linestyle=":", linewidth=0.8)
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

