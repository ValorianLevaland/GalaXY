"""galaxy.tiling

Topology-agnostic helpers to build analysis windows ("regions") from a domain ROI.

The goal is to map a user-selected *topology profile* to a list of RegionSpec
objects, each with:
- a polygon window geometry (shapely)
- vectorized parts representation (ripley_backend.PolyPart) for fast point masking
- bin limits (d_start, d_end) for plotting summaries against an axis

This module intentionally does NOT perform clustering; it only constructs regions.

License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import shapely.geometry as shgeom
    import shapely.ops as shops
except Exception:  # pragma: no cover
    shgeom = None
    shops = None

from .profiles import TopologyProfile
from .geometry_base import (
    BandParams,
    GeometryParams,
    BandSpec,
    MembraneModel,
    build_bands,
    clean_geometry,
)
from .geometry_ring import build_model as build_ring_model
from .geometry_skeleton import build_model as build_skeleton_model
from .ripley_backend import (
    PolyPart,
    SubdivisionParams,
    WindowSpec,
    build_windows_from_roi,
    shapely_to_parts,
    mask_points_in_poly,
    distance_to_boundary,
    _uniform_points_in_window,
)


# ----------------------------
# Dataclasses
# ----------------------------

@dataclass(frozen=True)
class WindowBinningParams:
    """Parameters controlling non-geodesic subdivisions."""

    mode: str = "none"  # none | radial_shells | boundary_shells | outer_shells | perinuclear_shells | seed_bands | grid

    # Common
    n_bins: int = 6
    max_distance: Optional[float] = None
    include_seed_as_bin0: bool = True

    # Grid
    grid_rows: int = 3
    grid_cols: int = 3

    # For seed bands
    seed_band_width: float = 500.0

    # For fixed-width shells
    shell_width: Optional[float] = None


@dataclass(frozen=True)
class RegionSpec:
    """One analysis region (window/subwindow)."""

    name: str
    geom: object  # shapely Polygon/MultiPolygon
    parts: Tuple[PolyPart, ...]
    area: float

    # Axis / binning metadata
    axis: str
    d_start: float
    d_end: float

    # Optional for geodesic profiles
    skeleton_length: float = float("nan")


# ----------------------------
# Shapely helpers
# ----------------------------

def _ensure_shapely() -> None:
    if shgeom is None or shops is None:
        raise ImportError("shapely is required for ROI operations")


def union_geoms(geoms: Sequence[object]):
    _ensure_shapely()
    if not geoms:
        return shgeom.GeometryCollection([])
    try:
        return shops.unary_union(list(geoms))
    except Exception:
        # fallback: iterative union
        g = geoms[0]
        for h in geoms[1:]:
            try:
                g = g.union(h)
            except Exception:
                pass
        return g


def build_domain_geometry(
    *,
    outer_geoms: Sequence[object],
    hole_geoms: Sequence[object],
) -> Tuple[object, object, object]:
    """Return (outer_union, holes_union, domain=outer-holes) as shapely geometries."""
    _ensure_shapely()

    outer = union_geoms([clean_geometry(g) for g in outer_geoms if g is not None])
    holes = union_geoms([clean_geometry(g) for g in hole_geoms if g is not None]) if hole_geoms else shgeom.GeometryCollection([])

    if outer.is_empty:
        return outer, holes, outer

    try:
        domain = outer.difference(holes) if (holes is not None and (not holes.is_empty)) else outer
    except Exception:
        domain = outer

    domain = clean_geometry(domain)
    return outer, holes, domain


def _regions_from_window_specs(
    windows: List[Tuple[str, object]],
    *,
    axis: str,
    d_edges: Optional[np.ndarray] = None,
    skeleton_lengths: Optional[np.ndarray] = None,
) -> List[RegionSpec]:
    """Convert a list of (name, geom) into RegionSpec list."""
    out: List[RegionSpec] = []
    for k, (name, geom) in enumerate(windows):
        if geom is None or getattr(geom, "is_empty", True):
            continue
        try:
            geom = geom.buffer(0)
        except Exception:
            pass
        area = float(getattr(geom, "area", 0.0))
        if area <= 0:
            continue
        parts = tuple(shapely_to_parts(geom))
        if d_edges is not None and len(d_edges) >= 2 and k < len(d_edges) - 1:
            d0 = float(d_edges[k])
            d1 = float(d_edges[k + 1])
        else:
            d0 = float(k)
            d1 = float(k + 1)
        sk = float("nan")
        if skeleton_lengths is not None and k < len(skeleton_lengths):
            sk = float(skeleton_lengths[k])
        out.append(RegionSpec(name=str(name), geom=geom, parts=parts, area=area, axis=str(axis), d_start=d0, d_end=d1, skeleton_length=sk))
    return out


# ----------------------------
# Non-geodesic tiling builders
# ----------------------------

def build_radial_shells(domain_geom, *, n_bins: int, axis: str) -> List[RegionSpec]:
    """Centroid-based radial shells (intersected with domain)."""
    _ensure_shapely()
    sub = SubdivisionParams(mode="radial", radial_shells=int(max(1, n_bins)))
    # build_windows_from_roi returns WindowSpec without geometry; so we re-create shells ourselves
    # for full reproducibility (WKT export).
    # We'll mirror its logic using shapely buffers.

    if domain_geom is None or domain_geom.is_empty:
        return []

    domain_geom = domain_geom.buffer(0)
    minx, miny, maxx, maxy = domain_geom.bounds
    c = domain_geom.centroid
    cx, cy = float(c.x), float(c.y)
    r_max = max(
        np.hypot(minx - cx, miny - cy),
        np.hypot(minx - cx, maxy - cy),
        np.hypot(maxx - cx, miny - cy),
        np.hypot(maxx - cx, maxy - cy),
    )
    edges = np.linspace(0.0, r_max, int(max(1, n_bins)) + 1, dtype=float)
    windows: List[Tuple[str, object]] = []
    for k in range(len(edges) - 1):
        r0 = float(edges[k])
        r1 = float(edges[k + 1])
        outer = shgeom.Point(cx, cy).buffer(r1, resolution=64)
        if r0 <= 0:
            ring = outer
        else:
            inner = shgeom.Point(cx, cy).buffer(r0, resolution=64)
            ring = outer.difference(inner)
        win = domain_geom.intersection(ring)
        windows.append((f"radial_{k+1}_r{r0:g}-{r1:g}", win))
    return _regions_from_window_specs(windows, axis=axis, d_edges=edges)


def build_boundary_shells(domain_geom, *, n_bins: int, axis: str) -> List[RegionSpec]:
    """Distance-to-boundary shells using morphological erosion of the full domain boundary.

    Notes
    -----
    If the domain has holes, this considers *all* boundaries (outer + hole boundaries).
    This is sometimes desirable (e.g., exclusion zones), but for cortex analysis users
    typically want distance to the *outer* boundary only; use build_outer_shells.
    """
    _ensure_shapely()

    if domain_geom is None or domain_geom.is_empty:
        return []

    # Use ripley_backend's d_max estimation logic (sampling) by reconstructing it here.
    domain_geom = domain_geom.buffer(0)
    parts = tuple(shapely_to_parts(domain_geom))
    area = float(domain_geom.area)
    if area <= 0:
        return []

    minx, miny, maxx, maxy = domain_geom.bounds
    rng = np.random.default_rng(0)
    probe_n = int(min(5000, max(1000, area / max(1.0, (maxx - minx) * (maxy - miny)) * 5000)))
    probe = _uniform_points_in_window(parts, probe_n, rng)
    bd_probe = distance_to_boundary(parts, probe)
    d_max = float(np.max(bd_probe)) if bd_probe.size else 0.0
    if d_max <= 0:
        return []

    edges = np.linspace(0.0, d_max, int(max(1, n_bins)) + 1, dtype=float)
    windows: List[Tuple[str, object]] = []
    for k in range(len(edges) - 1):
        d0 = float(edges[k])
        d1 = float(edges[k + 1])
        outer = domain_geom.buffer(-d0)
        inner = domain_geom.buffer(-d1)
        if outer.is_empty:
            continue
        if inner.is_empty:
            shell_geom = outer
        else:
            shell_geom = outer.difference(inner)
        if shell_geom.is_empty:
            continue
        windows.append((f"boundary_{k+1}_d{d0:g}-{d1:g}", shell_geom))
    return _regions_from_window_specs(windows, axis=axis, d_edges=edges)


def build_outer_shells(
    *,
    outer_geom,
    holes_geom,
    domain_geom,
    n_bins: int,
    axis: str,
    max_distance: Optional[float] = None,
) -> List[RegionSpec]:
    """Shells based on distance to the *outer* boundary, ignoring hole boundaries.

    Implementation
    --------------
    We erode the *outer* polygon and take differences to build shells, then intersect
    with the final domain to enforce holes/exclusions.
    """
    _ensure_shapely()

    if outer_geom is None or outer_geom.is_empty:
        return []

    outer_geom = clean_geometry(outer_geom)
    domain_geom = clean_geometry(domain_geom)

    outer_parts = tuple(shapely_to_parts(outer_geom))
    if max_distance is None:
        # Estimate max distance to outer boundary using probe sampling inside the domain.
        dom_parts = tuple(shapely_to_parts(domain_geom))
        area = float(domain_geom.area)
        if area <= 0:
            return []
        minx, miny, maxx, maxy = domain_geom.bounds
        rng = np.random.default_rng(0)
        probe_n = int(min(5000, max(1000, area / max(1.0, (maxx - minx) * (maxy - miny)) * 5000)))
        probe = _uniform_points_in_window(dom_parts, probe_n, rng)
        bd_probe = distance_to_boundary(outer_parts, probe)
        d_max = float(np.max(bd_probe)) if bd_probe.size else 0.0
    else:
        d_max = float(max_distance)

    if d_max <= 0:
        return []

    edges = np.linspace(0.0, d_max, int(max(1, n_bins)) + 1, dtype=float)
    windows: List[Tuple[str, object]] = []
    for k in range(len(edges) - 1):
        d0 = float(edges[k])
        d1 = float(edges[k + 1])
        outer_eroded = outer_geom.buffer(-d0)
        inner_eroded = outer_geom.buffer(-d1)
        if outer_eroded.is_empty:
            continue
        if inner_eroded.is_empty:
            shell = outer_eroded
        else:
            shell = outer_eroded.difference(inner_eroded)
        # enforce holes/exclusions
        try:
            shell = shell.intersection(domain_geom)
        except Exception:
            pass
        if shell.is_empty:
            continue
        windows.append((f"outer_{k+1}_d{d0:g}-{d1:g}", shell))

    return _regions_from_window_specs(windows, axis=axis, d_edges=edges)


def build_perinuclear_shells(
    *,
    nucleus_geom,
    domain_geom,
    n_bins: int,
    axis: str,
    max_distance: Optional[float] = None,
) -> List[RegionSpec]:
    """Shells based on distance to nucleus boundary (outward bands)."""
    _ensure_shapely()

    if nucleus_geom is None or nucleus_geom.is_empty:
        return []
    if domain_geom is None or domain_geom.is_empty:
        return []

    nucleus_geom = clean_geometry(nucleus_geom)
    domain_geom = clean_geometry(domain_geom)

    # Estimate max distance: farthest point in domain from nucleus.
    if max_distance is None:
        dom_parts = tuple(shapely_to_parts(domain_geom))
        area = float(domain_geom.area)
        if area <= 0:
            return []
        rng = np.random.default_rng(0)
        probe = _uniform_points_in_window(dom_parts, min(4000, max(1000, int(area / max(1.0, area) * 4000))), rng)
        nuc_parts = tuple(shapely_to_parts(nucleus_geom))
        # probe points are outside nucleus by construction; boundary distance is polygon distance.
        d_probe = distance_to_boundary(nuc_parts, probe)
        d_max = float(np.max(d_probe)) if d_probe.size else 0.0
    else:
        d_max = float(max_distance)

    if d_max <= 0:
        return []

    edges = np.linspace(0.0, d_max, int(max(1, n_bins)) + 1, dtype=float)

    windows: List[Tuple[str, object]] = []
    for k in range(len(edges) - 1):
        d0 = float(edges[k])
        d1 = float(edges[k + 1])
        outer = nucleus_geom.buffer(d1)
        inner = nucleus_geom.buffer(d0)
        shell = outer.difference(inner) if d0 > 0 else outer
        try:
            shell = shell.intersection(domain_geom)
        except Exception:
            pass
        if shell.is_empty:
            continue
        windows.append((f"perinuclear_{k+1}_d{d0:g}-{d1:g}", shell))

    return _regions_from_window_specs(windows, axis=axis, d_edges=edges)


def build_seed_bands(
    *,
    seed_geom,
    domain_geom,
    n_bins: int,
    axis: str,
    band_width: float,
    include_seed: bool,
    max_distance: Optional[float] = None,
) -> List[RegionSpec]:
    """Euclidean distance bands around seed ROI(s), intersected with the domain."""
    _ensure_shapely()

    if seed_geom is None or seed_geom.is_empty:
        return []
    if domain_geom is None or domain_geom.is_empty:
        return []

    seed_geom = clean_geometry(seed_geom)
    domain_geom = clean_geometry(domain_geom)

    if band_width <= 0:
        raise ValueError("band_width must be > 0")

    if max_distance is None:
        # estimate by sampling domain points and taking max distance to seed
        dom_parts = tuple(shapely_to_parts(domain_geom))
        area = float(domain_geom.area)
        if area <= 0:
            return []
        rng = np.random.default_rng(0)
        probe = _uniform_points_in_window(dom_parts, min(5000, max(1000, int(area / max(1.0, area) * 5000))), rng)
        seed_parts = tuple(shapely_to_parts(seed_geom))
        inside = mask_points_in_poly(seed_parts, probe)
        d_boundary = distance_to_boundary(seed_parts, probe)
        d = np.where(inside, 0.0, d_boundary)
        d_max = float(np.max(d)) if d.size else 0.0
    else:
        d_max = float(max_distance)

    if d_max <= 0:
        d_max = float(band_width)

    # edges based on fixed band_width, but capped at d_max
    n_bins = int(max(1, n_bins))
    if n_bins is None or n_bins <= 0:
        n_bins = int(np.ceil(d_max / float(band_width)))

    edges = np.linspace(0.0, d_max, n_bins + 1, dtype=float)

    windows: List[Tuple[str, object]] = []

    if include_seed:
        try:
            seed_inside = seed_geom.intersection(domain_geom)
        except Exception:
            seed_inside = seed_geom
        if not seed_inside.is_empty:
            windows.append(("seed", seed_inside))

    for k in range(len(edges) - 1):
        d0 = float(edges[k])
        d1 = float(edges[k + 1])
        if d0 == 0 and include_seed:
            # band 1 begins outside seed; exclude seed area explicitly
            outer = seed_geom.buffer(d1)
            ring = outer.difference(seed_geom)
        else:
            outer = seed_geom.buffer(d1)
            inner = seed_geom.buffer(d0)
            ring = outer.difference(inner) if d0 > 0 else outer
        try:
            ring = ring.intersection(domain_geom)
        except Exception:
            pass
        if ring.is_empty:
            continue
        windows.append((f"seedband_{k+1}_d{d0:g}-{d1:g}", ring))

    # Build RegionSpec. If include_seed, the first window has no natural d_start/d_end; set to 0.
    regions: List[RegionSpec] = []
    if include_seed and windows:
        # Seed bin
        name0, g0 = windows[0]
        if g0 is not None and (not g0.is_empty) and float(g0.area) > 0:
            regions.append(
                RegionSpec(
                    name=str(name0),
                    geom=g0,
                    parts=tuple(shapely_to_parts(g0)),
                    area=float(g0.area),
                    axis=axis,
                    d_start=0.0,
                    d_end=0.0,
                )
            )
        # The rest correspond to edges
        rest = windows[1:]
        regions.extend(_regions_from_window_specs(rest, axis=axis, d_edges=edges))
        return regions

    return _regions_from_window_specs(windows, axis=axis, d_edges=edges)


def build_grid(domain_geom, *, rows: int, cols: int, axis: str) -> List[RegionSpec]:
    _ensure_shapely()
    if domain_geom is None or domain_geom.is_empty:
        return []

    domain_geom = domain_geom.buffer(0)
    minx, miny, maxx, maxy = domain_geom.bounds
    rows = int(max(1, rows))
    cols = int(max(1, cols))
    dx = (maxx - minx) / cols
    dy = (maxy - miny) / rows

    windows: List[Tuple[str, object]] = []
    for rr in range(rows):
        for cc in range(cols):
            x0 = minx + cc * dx
            x1 = minx + (cc + 1) * dx
            y0 = miny + rr * dy
            y1 = miny + (rr + 1) * dy
            tile = shgeom.Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
            win = domain_geom.intersection(tile)
            if win.is_empty:
                continue
            windows.append((f"tile_r{rr+1}_c{cc+1}", win))
    return _regions_from_window_specs(windows, axis=axis)


# ----------------------------
# Geodesic model builder
# ----------------------------

def build_geodesic_model(
    *,
    domain_geom,
    seed_geom,
    geom_backend: str,
    geom_params: GeometryParams,
    logger=None,
) -> Tuple[MembraneModel, str]:
    """Build a MembraneModel-like skeleton model for geodesic profiles."""

    mode = str(geom_backend).lower().strip()
    if mode == "ring":
        return build_ring_model(domain_geom, seed_geom, geom_params), "ring"
    if mode == "skeleton":
        return build_skeleton_model(domain_geom, seed_geom, geom_params), "skeleton"

    # Auto
    try:
        model = build_ring_model(domain_geom, seed_geom, geom_params)
        return model, "ring"
    except Exception as e_ring:
        if logger is not None:
            try:
                logger.warning(f"Auto backend: ring failed ({type(e_ring).__name__}: {e_ring}); using skeleton backend.")
            except Exception:
                pass
        model = build_skeleton_model(domain_geom, seed_geom, geom_params)
        return model, "skeleton"


def build_geodesic_bands(
    *,
    model: MembraneModel,
    band_params: BandParams,
    axis: str,
) -> List[RegionSpec]:
    """Build geodesic distance bands (equal-length or fixed-distance)."""

    bands: List[BandSpec] = build_bands(model, band_params)
    out: List[RegionSpec] = []
    for b in bands:
        if b.geom is None or getattr(b.geom, "is_empty", True):
            continue
        parts = tuple(shapely_to_parts(b.geom))
        out.append(
            RegionSpec(
                name=str(b.name),
                geom=b.geom,
                parts=parts,
                area=float(b.area),
                axis=str(axis),
                d_start=float(b.d_start),
                d_end=float(b.d_end),
                skeleton_length=float(b.skeleton_length),
            )
        )
    return out


# ----------------------------
# Profile dispatcher
# ----------------------------

def build_regions_for_profile(
    *,
    profile: TopologyProfile,
    outer_geom,
    holes_geom,
    domain_geom,
    seed_geom,
    geom_backend: str,
    geom_params: GeometryParams,
    band_params: BandParams,
    binning: WindowBinningParams,
    axis_override: Optional[str] = None,
    logger=None,
) -> Tuple[List[RegionSpec], Dict[str, object]]:
    """Return (regions, context) for the selected profile.

    context contains optional extra objects (e.g., geodesic model) needed downstream.
    """

    ctx: Dict[str, object] = {}
    axis = str(axis_override) if axis_override is not None else profile.default_axis

    # Geodesic profiles
    if profile.uses_geodesic:
        if seed_geom is None or getattr(seed_geom, "is_empty", True):
            raise ValueError("This profile requires a seed ROI, but no seeds were provided.")
        model, engine = build_geodesic_model(
            domain_geom=domain_geom,
            seed_geom=seed_geom,
            geom_backend=geom_backend,
            geom_params=geom_params,
            logger=logger,
        )
        ctx["model"] = model
        ctx["engine"] = engine
        regions = build_geodesic_bands(model=model, band_params=band_params, axis=axis)
        return regions, ctx

    # Non-geodesic profiles
    mode = str(binning.mode or profile.default_tiling).lower().strip()

    if mode in ("radial", "radial_shells"):
        regions = build_radial_shells(domain_geom, n_bins=int(max(1, binning.n_bins)), axis=axis)
        return regions, ctx

    if mode in ("boundary", "boundary_shells"):
        regions = build_boundary_shells(domain_geom, n_bins=int(max(1, binning.n_bins)), axis=axis)
        return regions, ctx

    if mode in ("outer", "outer_shells"):
        regions = build_outer_shells(
            outer_geom=outer_geom,
            holes_geom=holes_geom,
            domain_geom=domain_geom,
            n_bins=int(max(1, binning.n_bins)),
            axis=axis,
            max_distance=binning.max_distance,
        )
        return regions, ctx

    if mode in ("perinuclear", "perinuclear_shells"):
        if holes_geom is None or getattr(holes_geom, "is_empty", True):
            raise ValueError("Perinuclear profile requires nucleus holes (Holes layer) but none were provided.")
        regions = build_perinuclear_shells(
            nucleus_geom=holes_geom,
            domain_geom=domain_geom,
            n_bins=int(max(1, binning.n_bins)),
            axis=axis,
            max_distance=binning.max_distance,
        )
        return regions, ctx

    if mode in ("seed", "seed_bands"):
        if seed_geom is None or getattr(seed_geom, "is_empty", True):
            raise ValueError("Seed-centered profile requires seeds but none were provided.")
        regions = build_seed_bands(
            seed_geom=seed_geom,
            domain_geom=domain_geom,
            n_bins=int(max(1, binning.n_bins)),
            axis=axis,
            band_width=float(binning.seed_band_width),
            include_seed=bool(binning.include_seed_as_bin0),
            max_distance=binning.max_distance,
        )
        return regions, ctx

    if mode in ("grid",):
        regions = build_grid(domain_geom, rows=int(binning.grid_rows), cols=int(binning.grid_cols), axis=axis)
        return regions, ctx

    # none
    if domain_geom is None or getattr(domain_geom, "is_empty", True):
        return [], ctx

    parts = tuple(shapely_to_parts(domain_geom))
    reg = RegionSpec(
        name="ROI",
        geom=domain_geom,
        parts=parts,
        area=float(getattr(domain_geom, "area", 0.0)),
        axis=axis,
        d_start=0.0,
        d_end=0.0,
    )
    return [reg], ctx
