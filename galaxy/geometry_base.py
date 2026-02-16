"""geometry_base.py

Shared dataclasses + helper utilities for membrane distance models.

This module defines the *common contract* between the GUI/worker and the
geometry engines (Ring / Skeleton) as well as the band-construction utilities.

Key concepts
------------
- MembraneModel: a membrane band (Polygon-with-hole) + its skeleton graph and
  per-node geodesic distance-to-ZC (contact zone).
- Bands: polygons derived from the skeleton, used as analysis windows.

Banding strategies
------------------
Two strategies are supported (single codebase; selectable in the GUI):

1) fixed_distance
   - classic bands with a constant distance width Δd (e.g. 500 nm)
   - easy to interpret as "at distance d ± Δd/2"
   - but can yield unequal membrane length/area per band for exotic shapes

2) equal_length  (recommended)
   - bands defined by equal *geodesic membrane length* along the skeleton
   - distance ranges are adaptive, but each band has comparable statistical power
   - best for comparing clustering metrics across distance

License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import shapely.geometry as shgeom
    import shapely.ops as shops
except Exception:  # pragma: no cover
    shgeom = None
    shops = None


# -----------------------------
# Public dataclasses
# -----------------------------

@dataclass(frozen=True)
class GeometryParams:
    """Parameters controlling skeletonization/graph construction."""

    pixel_size: float = 50.0  # coordinate units per raster pixel (e.g., nm)
    prune_spurs: bool = True
    prune_spur_length: float = 200.0  # units; spurs shorter than this are removed
    thickness_mode: str = "auto"  # auto|manual
    thickness_manual: float = 200.0


@dataclass(frozen=True)
class BandParams:
    """Band construction parameters.

    Notes
    -----
    - `mode` selects the slicing strategy.
    - Only the parameters relevant to the selected mode are used.
    """

    mode: str = "equal_length"  # equal_length | fixed_distance
    band_width: float = 500.0   # used if mode == fixed_distance
    band_length: float = 5000.0  # used if mode == equal_length (target skeleton length per band)
    max_distance: Optional[float] = None  # units; if None -> engine default
    subtract_zc: bool = True


@dataclass(frozen=True)
class BandSpec:
    name: str
    d_start: float
    d_end: float
    geom: object  # shapely Polygon/MultiPolygon
    area: float
    skeleton_length: float


@dataclass
class MembraneModel:
    """Common representation returned by geometry engines."""

    membrane_geom: object
    zc_geom: object

    # Skeleton graph representation
    graph: object  # networkx.Graph[int]
    node_xy: np.ndarray  # (N,2) float, world coords (x,y)
    node_dist_to_zc: np.ndarray  # (N,) float (world units), np.inf if unreachable
    skeleton_length: float
    thickness: float

    # Ring extras (optional)
    is_ring: bool = False
    ring_perimeter: Optional[float] = None
    ring_s: Optional[np.ndarray] = None  # per node, arc-length coordinate in [0,P)


# -----------------------------
# Shapely helpers
# -----------------------------

def ensure_shapely() -> None:
    if shgeom is None or shops is None:
        raise ImportError("shapely is required for membrane geometry operations.")


def clean_geometry(geom):
    """Attempt to fix invalid geometries (self-intersections etc.)."""
    ensure_shapely()
    if geom is None:
        raise ValueError("Geometry is None")
    if geom.is_empty:
        return geom
    g = geom
    try:
        # buffer(0) is a common fix for self-intersections.
        if not g.is_valid:
            g = g.buffer(0)
    except Exception:
        pass
    return g


def geometry_area(geom) -> float:
    ensure_shapely()
    if geom is None or geom.is_empty:
        return 0.0
    return float(geom.area)


def geometry_bounds(geom) -> Tuple[float, float, float, float]:
    ensure_shapely()
    if geom is None or geom.is_empty:
        return (0.0, 0.0, 0.0, 0.0)
    minx, miny, maxx, maxy = geom.bounds
    return float(minx), float(miny), float(maxx), float(maxy)


def iter_polygons(geom) -> Iterable[object]:
    """Yield shapely Polygon(s) from Polygon/MultiPolygon."""
    ensure_shapely()
    if geom is None or geom.is_empty:
        return
    if isinstance(geom, shgeom.Polygon):
        yield geom
    elif isinstance(geom, shgeom.MultiPolygon):
        for g in geom.geoms:
            yield g
    else:
        raise TypeError(f"Unsupported geometry type: {type(geom)}")


def merge_geoms(geoms: Sequence[object]):
    ensure_shapely()
    if not geoms:
        return shgeom.GeometryCollection([])
    return shops.unary_union(list(geoms))


# -----------------------------
# Band utilities
# -----------------------------

def default_max_distance(model: MembraneModel) -> float:
    """Engine-independent default max distance."""
    if model.is_ring and model.ring_perimeter is not None and np.isfinite(model.ring_perimeter):
        return float(model.ring_perimeter) / 2.0
    d = model.node_dist_to_zc
    finite = d[np.isfinite(d)]
    if finite.size == 0:
        return 0.0
    return float(np.max(finite))


def _edges_with_mid_distance(model: MembraneModel) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (edges, mid_dist, edge_len) for all finite-distance edges."""
    G = model.graph
    edges = []
    edge_mid_d = []
    edge_len = []
    for u, v, data in G.edges(data=True):
        du = model.node_dist_to_zc[int(u)]
        dv = model.node_dist_to_zc[int(v)]
        if not (np.isfinite(du) and np.isfinite(dv)):
            continue
        dm = 0.5 * (float(du) + float(dv))
        w = float(data.get("weight", np.linalg.norm(model.node_xy[int(u)] - model.node_xy[int(v)])))
        edges.append((int(u), int(v)))
        edge_mid_d.append(dm)
        edge_len.append(w)

    if not edges:
        return np.zeros((0, 2), dtype=int), np.zeros((0,), dtype=float), np.zeros((0,), dtype=float)

    return (
        np.asarray(edges, dtype=int),
        np.asarray(edge_mid_d, dtype=float),
        np.asarray(edge_len, dtype=float),
    )


def _zc_clip_for_subtraction(model: MembraneModel, subtract_zc: bool):
    """Return ZC geometry clipped to membrane, or None."""
    ensure_shapely()
    if not subtract_zc:
        return None
    zc_clip = model.zc_geom
    if zc_clip is None or zc_clip.is_empty:
        return None
    try:
        zc_clip = zc_clip.intersection(model.membrane_geom)
    except Exception:
        pass
    return zc_clip


def _buffer_edges_to_band_poly(model: MembraneModel, sel_edges: np.ndarray, buf: float):
    """Convert selected graph edges to a (multi)polygon band window."""
    ensure_shapely()
    lines = []
    for u, v in sel_edges:
        p = tuple(map(float, model.node_xy[int(u)]))
        q = tuple(map(float, model.node_xy[int(v)]))
        lines.append(shgeom.LineString([p, q]))

    if not lines:
        return shgeom.GeometryCollection([])

    multi = shgeom.MultiLineString(lines) if len(lines) > 1 else lines[0]

    try:
        poly = multi.buffer(float(buf), cap_style=1, join_style=1)
    except Exception:
        poly = multi.buffer(float(buf))

    # Restrict to membrane area.
    try:
        poly = poly.intersection(model.membrane_geom)
    except Exception:
        pass

    # Optional simplification to reduce vertex count (keeps exports smaller and
    # plotting faster). The original code referenced a non-existent variable
    # (`parames.pixel_size`), which would crash band construction at runtime.
    #
    # We tie the tolerance to the effective band radius (`buf`) so the
    # simplification scale remains sensible across datasets and backends.
    try:
        tol = 0.25 * float(buf)
        if np.isfinite(tol) and tol > 0:
            poly = poly.simplify(tolerance=tol, preserve_topology=True)
    except Exception:
        pass

    return clean_geometry(poly)


def _format_distance(d: float) -> str:
    """Human-friendly distance formatting for band names."""
    d = float(d)
    if not np.isfinite(d):
        return "nan"
    if abs(d - round(d)) < 1e-6:
        return str(int(round(d)))
    # Large values: keep 1 decimal.
    if abs(d) >= 1000:
        s = f"{d:.1f}"
    else:
        s = f"{d:.3g}"
    return s.rstrip("0").rstrip(".")


def build_fixed_distance_bands(model: MembraneModel, params: BandParams) -> List[BandSpec]:
    """Build band polygons by graph-distance to ZC using a fixed distance width Δd.

    Bands are created by selecting skeleton edges with midpoint distance in [d0, d1),
    buffering those edges by ~thickness/2, then intersecting with membrane_geom.

    Notes
    -----
    - Band polygons may be disconnected (MultiPolygon).
    - If params.subtract_zc is True, ZC∩membrane is removed from all bands.
    """
    ensure_shapely()

    bw = float(params.band_width)
    if bw <= 0:
        raise ValueError("band_width must be > 0")

    max_d = params.max_distance
    if max_d is None:
        max_d = default_max_distance(model)
    max_d = float(max_d)

    if max_d <= 0:
        return []

    edges, edge_mid_d, edge_len = _edges_with_mid_distance(model)
    if edges.shape[0] == 0:
        return []

    # Optionally limit to max distance
    keep = edge_mid_d < max_d
    edges = edges[keep]
    edge_mid_d = edge_mid_d[keep]
    edge_len = edge_len[keep]
    if edges.shape[0] == 0:
        return []

    zc_clip = _zc_clip_for_subtraction(model, params.subtract_zc)

    # Buffer radius
    buf = float(model.thickness) / 2.0
    if not np.isfinite(buf) or buf <= 0:
        buf = bw / 4.0

    bands: List[BandSpec] = []

    n_bins = int(np.ceil(max_d / bw))
    for k in range(n_bins):
        d0 = k * bw
        d1 = min((k + 1) * bw, max_d)
        sel = (edge_mid_d >= d0) & (edge_mid_d < d1)
        if not np.any(sel):
            continue

        poly = _buffer_edges_to_band_poly(model, edges[sel], buf=buf)

        if zc_clip is not None and not zc_clip.is_empty:
            try:
                poly = poly.difference(zc_clip)
            except Exception:
                pass
            poly = clean_geometry(poly)

        area = geometry_area(poly)
        if area <= 0:
            continue

        sk_len = float(np.sum(edge_len[sel]))
        bands.append(
            BandSpec(
                name=f"band_d{_format_distance(d0)}-{_format_distance(d1)}",
                d_start=float(d0),
                d_end=float(d1),
                geom=poly,
                area=area,
                skeleton_length=sk_len,
            )
        )

    return bands


def build_equal_length_bands(model: MembraneModel, params: BandParams) -> List[BandSpec]:
    """Build bands with equal *geodesic membrane length* (along the skeleton).

    Strategy
    --------
    1) compute midpoint distance-to-ZC for each skeleton edge
    2) sort edges by this distance (ascending)
    3) accumulate edge length until reaching params.band_length
    4) cut a band (at the end of a same-distance group) and repeat

    This yields bands with comparable statistical power even for:
    - curved rings
    - strongly concave outlines
    - branched/exotic shapes

    The distance ranges [d_start, d_end] are *adaptive* and reported per band.
    """
    ensure_shapely()

    target_len = float(params.band_length)
    if target_len <= 0:
        raise ValueError("band_length must be > 0 for equal_length mode")

    max_d = params.max_distance
    if max_d is None:
        max_d = default_max_distance(model)
    max_d = float(max_d)

    if max_d <= 0:
        return []

    edges, edge_mid_d, edge_len = _edges_with_mid_distance(model)
    if edges.shape[0] == 0:
        return []

    # Limit by max distance
    keep = edge_mid_d < max_d
    edges = edges[keep]
    edge_mid_d = edge_mid_d[keep]
    edge_len = edge_len[keep]
    if edges.shape[0] == 0:
        return []

    # Sort by distance
    order = np.argsort(edge_mid_d, kind="mergesort")
    edges = edges[order]
    edge_mid_d = edge_mid_d[order]
    edge_len = edge_len[order]

    zc_clip = _zc_clip_for_subtraction(model, params.subtract_zc)

    # Buffer radius based on thickness estimate
    buf = float(model.thickness) / 2.0
    if not np.isfinite(buf) or buf <= 0:
        # fall back to something conservative based on typical edge length
        med_edge = float(np.median(edge_len)) if edge_len.size else 1.0
        buf = max(med_edge, 1.0)

    bands: List[BandSpec] = []

    cur_edges_idx: List[int] = []
    cur_len = 0.0
    cur_d0: Optional[float] = None

    def flush_band(d_end: float) -> None:
        nonlocal cur_edges_idx, cur_len, cur_d0, bands
        if not cur_edges_idx:
            return
        sel_edges = edges[np.asarray(cur_edges_idx, dtype=int)]
        poly = _buffer_edges_to_band_poly(model, sel_edges, buf=buf)

        if zc_clip is not None and not zc_clip.is_empty:
            try:
                poly = poly.difference(zc_clip)
            except Exception:
                pass
            poly = clean_geometry(poly)

        area = geometry_area(poly)
        if area > 0:
            d0 = float(cur_d0) if cur_d0 is not None else float(np.min(edge_mid_d[cur_edges_idx]))
            d1 = float(d_end)
            bands.append(
                BandSpec(
                    name=f"band_d{_format_distance(d0)}-{_format_distance(d1)}",
                    d_start=d0,
                    d_end=d1,
                    geom=poly,
                    area=float(area),
                    skeleton_length=float(cur_len),
                )
            )

        cur_edges_idx = []
        cur_len = 0.0
        cur_d0 = None

    # Iterate edges, accumulating length.
    n = int(edges.shape[0])
    for i in range(n):
        dm = float(edge_mid_d[i])
        if cur_d0 is None:
            cur_d0 = dm
        cur_edges_idx.append(i)
        cur_len += float(edge_len[i])

        # Decide if we should close the band here:
        # - reached target length AND
        # - end of a same-distance group (to avoid overlapping distance ranges)
        next_dm = float(edge_mid_d[i + 1]) if (i + 1) < n else float("inf")
        end_of_group = (not np.isfinite(next_dm)) or (next_dm > dm)

        if cur_len >= target_len and end_of_group:
            flush_band(d_end=dm)

    # Last partial band
    if cur_edges_idx:
        flush_band(d_end=float(edge_mid_d[cur_edges_idx[-1]]))

    # Optional: merge a very small tail band into previous to avoid tiny last bin
    if len(bands) >= 2:
        last = bands[-1]
        if last.skeleton_length < 0.25 * target_len:
            prev = bands[-2]

            # Merge edges by union of polygons; distances become [prev.d_start, last.d_end]
            try:
                merged_geom = clean_geometry(merge_geoms([prev.geom, last.geom]))
            except Exception:
                merged_geom = prev.geom

            bands[-2] = BandSpec(
                name=f"band_d{_format_distance(prev.d_start)}-{_format_distance(last.d_end)}",
                d_start=float(prev.d_start),
                d_end=float(last.d_end),
                geom=merged_geom,
                area=float(geometry_area(merged_geom)),
                skeleton_length=float(prev.skeleton_length + last.skeleton_length),
            )
            bands.pop(-1)

    return bands


def build_bands(model: MembraneModel, params: BandParams) -> List[BandSpec]:
    """Build bands according to params.mode."""
    mode = str(params.mode).lower().strip()
    if mode in {"equal_length", "length", "equal_membrane_length"}:
        return build_equal_length_bands(model, params)
    # default/fallback
    return build_fixed_distance_bands(model, params)


# Backwards-compatibility name (older code might still import this)
def build_distance_bands(model: MembraneModel, params: BandParams) -> List[BandSpec]:
    """Alias for build_bands (kept for compatibility)."""
    return build_bands(model, params)
