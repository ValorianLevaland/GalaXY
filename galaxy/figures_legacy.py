"""figures.py

Matplotlib figure export helpers.

This module is used by the GUI worker thread to create QC figures that make
analysis reproducible and easy to audit.

Figures produced (typical):
- overview_points.png: membrane outline + ZC + distance bands + point cloud
- overview_bands.png: membrane outline + ZC + distance bands (no points)
- per-band DBSCAN overlays: points colored by cluster label within each band
- summary_vs_distance.png: band metrics vs distance from ZC

All plotting functions are pure (no napari dependency).

License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
# NOTE: Backend is controlled by the main GUI (Qt5Agg). We avoid changing it here.
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

try:
    import shapely.geometry as shgeom
except Exception:  # pragma: no cover
    shgeom = None


def _ensure_shapely() -> None:
    if shgeom is None:
        raise ImportError("shapely is required for figure export")


def _iter_polygons(geom) -> Iterable["shgeom.Polygon"]:
    """Yield Polygon(s) from shapely geometry."""
    _ensure_shapely()
    if geom is None or geom.is_empty:
        return
    if isinstance(geom, shgeom.Polygon):
        yield geom
    elif isinstance(geom, shgeom.MultiPolygon):
        for g in geom.geoms:
            yield g
    else:
        # Try geometry collections
        try:
            for g in geom.geoms:
                if isinstance(g, shgeom.Polygon):
                    yield g
                elif isinstance(g, shgeom.MultiPolygon):
                    for gg in g.geoms:
                        yield gg
        except Exception:
            return


def _ring_to_path_vertices(coords: Sequence[Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (vertices, codes) for a ring."""
    c = np.asarray(coords, dtype=float)
    if c.ndim != 2 or c.shape[0] < 3:
        raise ValueError("Ring must have at least 3 points")
    # Ensure closed for Path.CLOSEPOLY usage.
    if not np.allclose(c[0], c[-1]):
        c = np.vstack([c, c[0]])
    n = int(c.shape[0])
    codes = np.full((n,), Path.LINETO, dtype=np.uint8)
    codes[0] = Path.MOVETO
    codes[-1] = Path.CLOSEPOLY
    return c, codes


def _polygon_to_path(poly: "shgeom.Polygon") -> Path:
    """Convert a Polygon (with holes) into a Matplotlib Path."""
    _ensure_shapely()

    vertices_list: List[np.ndarray] = []
    codes_list: List[np.ndarray] = []

    v, c = _ring_to_path_vertices(poly.exterior.coords)
    vertices_list.append(v)
    codes_list.append(c)

    for interior in poly.interiors:
        v, c = _ring_to_path_vertices(interior.coords)
        vertices_list.append(v)
        codes_list.append(c)

    vertices = np.concatenate(vertices_list, axis=0)
    codes = np.concatenate(codes_list, axis=0)
    return Path(vertices, codes)


def add_shapely_geom(
    ax,
    geom,
    *,
    facecolor: Optional[str] = None,
    edgecolor: str = "k",
    alpha: float = 0.25,
    linewidth: float = 1.0,
    zorder: int = 2,
    fill: bool = True,
):
    """Add Polygon/MultiPolygon geometry to axes as a patch."""
    _ensure_shapely()
    if geom is None or geom.is_empty:
        return

    for poly in _iter_polygons(geom):
        path = _polygon_to_path(poly)
        fc = facecolor if (fill and facecolor is not None) else "none"
        patch = PathPatch(
            path,
            facecolor=fc,
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=alpha,
            zorder=zorder,
        )
        # Make holes transparent when filled.
        try:
            patch.set_fillrule("evenodd")
        except Exception:
            pass
        ax.add_patch(patch)


def _downsample(points_xy: np.ndarray, max_points: int, seed: int = 0) -> np.ndarray:
    n = int(points_xy.shape[0])
    if max_points is None or max_points <= 0 or n <= max_points:
        return points_xy
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(n, size=int(max_points), replace=False)
    return points_xy[idx]


def plot_overview(
    *,
    points_xy: np.ndarray,
    membrane_geom,
    zc_geom,
    bands: Sequence,
    out_path: str,
    title: str = "",
    show_points: bool = True,
    max_points: int = 200_000,
    seed: int = 0,
    dpi: int = 220,
):
    """Global overview plot with membrane + ZC + distance bands."""
    _ensure_shapely()

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    # Polygons first
    add_shapely_geom(ax, membrane_geom, facecolor=None, edgecolor="black", alpha=1.0, linewidth=1.0, zorder=2, fill=False)

    # Bands (semi-transparent)
    cmap = plt.cm.get_cmap("tab20")
    for k, b in enumerate(bands):
        try:
            geom = b.geom
        except Exception:
            geom = b
        color = cmap(k % cmap.N)
        add_shapely_geom(ax, geom, facecolor=color, edgecolor=color, alpha=0.18, linewidth=0.8, zorder=3, fill=True)

    # ZC on top
    if zc_geom is not None and (not getattr(zc_geom, "is_empty", True)):
        add_shapely_geom(ax, zc_geom, facecolor="red", edgecolor="red", alpha=0.35, linewidth=1.2, zorder=4, fill=True)

    # Points
    if show_points and points_xy is not None and points_xy.size > 0:
        pts = _downsample(np.asarray(points_xy, dtype=float), int(max_points), seed=seed)
        ax.scatter(pts[:, 0], pts[:, 1], s=1.0, c="0.5", alpha=0.55, linewidths=0, rasterized=True, zorder=1)

    if title:
        ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")

    # Tight bounds around membrane if available
    try:
        minx, miny, maxx, maxy = membrane_geom.bounds
        pad = 0.02 * max(maxx - minx, maxy - miny)
        ax.set_xlim(minx - pad, maxx + pad)
        ax.set_ylim(miny - pad, maxy + pad)
    except Exception:
        pass

    fig.tight_layout()
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)


def plot_band_dbscan(
    *,
    band_name: str,
    band_geom,
    points_xy: np.ndarray,
    labels: np.ndarray,
    out_path: str,
    title: str = "",
    background_xy: Optional[np.ndarray] = None,
    max_points: int = 200_000,
    seed: int = 0,
    dpi: int = 220,
):
    """Per-band plot: DBSCAN clusters colored, noise grey."""
    _ensure_shapely()

    pts = np.asarray(points_xy, dtype=float)
    lab = np.asarray(labels, dtype=int)
    if pts.shape[0] != lab.shape[0]:
        raise ValueError("points_xy and labels must have same length")

    # Downsample consistently
    n = int(pts.shape[0])
    if max_points is not None and max_points > 0 and n > max_points:
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(n, size=int(max_points), replace=False)
        pts = pts[idx]
        lab = lab[idx]

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)

    # Optional background (for context)
    if background_xy is not None and background_xy.size > 0:
        bg = _downsample(np.asarray(background_xy, dtype=float), int(max_points), seed=seed)
        ax.scatter(bg[:, 0], bg[:, 1], s=1.0, c="0.85", alpha=0.35, linewidths=0, rasterized=True, zorder=1)

    # Noise
    noise = (lab < 0)
    if np.any(noise):
        ax.scatter(pts[noise, 0], pts[noise, 1], s=2.0, c="0.7", alpha=0.35, linewidths=0, rasterized=True, zorder=2)

    # Clusters
    in_cl = ~noise
    if np.any(in_cl):
        ax.scatter(
            pts[in_cl, 0],
            pts[in_cl, 1],
            s=3.0,
            c=lab[in_cl].astype(float),
            cmap="nipy_spectral",
            alpha=0.9,
            linewidths=0,
            rasterized=True,
            zorder=3,
        )

    # Band outline
    add_shapely_geom(ax, band_geom, facecolor=None, edgecolor="red", alpha=0.9, linewidth=1.4, zorder=4, fill=False)

    if title:
        ax.set_title(title)
    else:
        ax.set_title(band_name)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")

    try:
        minx, miny, maxx, maxy = band_geom.bounds
        pad = 0.05 * max(maxx - minx, maxy - miny)
        ax.set_xlim(minx - pad, maxx + pad)
        ax.set_ylim(miny - pad, maxy + pad)
    except Exception:
        pass

    fig.tight_layout()
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)


def plot_summary_vs_distance(
    *,
    band_summary_df,
    out_path: str,
    title: str = "",
    dpi: int = 220,
):
    """Plot key metrics vs distance to ZC."""
    import pandas as pd

    df = band_summary_df.copy()
    if df.empty:
        return

    # distance midpoints
    d0 = df["d_start"].to_numpy(dtype=float)
    d1 = df["d_end"].to_numpy(dtype=float)
    dmid = 0.5 * (d0 + d1)
    # ensure ZC at 0
    if "band" in df.columns:
        zc = df["band"].astype(str).str.upper().eq("ZC")
        dmid = np.where(zc.to_numpy(), 0.0, dmid)

    df = df.assign(_dmid=dmid)
    df = df.sort_values("_dmid")

    fig = plt.figure(figsize=(9, 7))

    ax1 = fig.add_subplot(311)
    ax1.plot(df["_dmid"], df["density_length"], marker="o", linewidth=1.5)
    ax1.set_ylabel("density / length")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(312, sharex=ax1)
    ax2.plot(df["_dmid"], df["frac_points_in_clusters"], marker="o", linewidth=1.5)
    ax2.set_ylabel("fraction in clusters")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(313, sharex=ax1)
    ax3.plot(df["_dmid"], df["mean_cluster_size"], marker="o", linewidth=1.5)
    ax3.set_xlabel("distance to ZC")
    ax3.set_ylabel("mean cluster size")
    ax3.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title)

    fig.tight_layout(rect=[0, 0, 1, 0.96] if title else None)
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)
