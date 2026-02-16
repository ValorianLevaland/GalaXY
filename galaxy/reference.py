"""galaxy.reference

Vectorized computation of reference scalar fields used to bin/stratify points.

In GalaXY, *distance axis* values are treated as optional per-point metadata.
They are not required for clustering, but they enable:
- per-window summaries vs distance
- per-cluster mean distance

This module provides robust, topology-agnostic helpers.

License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from scipy.spatial import cKDTree

try:
    import shapely.geometry as shgeom
except Exception:  # pragma: no cover
    shgeom = None

from .ripley_backend import shapely_to_parts, mask_points_in_poly, distance_to_boundary
from .geometry_base import MembraneModel


def _ensure_shapely() -> None:
    if shgeom is None:
        raise ImportError("shapely is required for reference fields")


def ref_distance_to_centroid(
    *,
    domain_geom,
    points_xy: np.ndarray,
    mask_to_domain: bool = True,
) -> np.ndarray:
    """Euclidean distance to the domain centroid.

    Parameters
    ----------
    domain_geom:
        Shapely geometry (Polygon/MultiPolygon).
    points_xy:
        (N,2) array.
    mask_to_domain:
        If True, values for points outside the domain are set to NaN.
    """

    _ensure_shapely()
    pts = np.asarray(points_xy, dtype=float)
    c = domain_geom.centroid
    cx, cy = float(c.x), float(c.y)
    d = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
    if mask_to_domain:
        parts = tuple(shapely_to_parts(domain_geom))
        inside = mask_points_in_poly(parts, pts)
        d = d.astype(float)
        d[~inside] = np.nan
    return d


def ref_distance_to_domain_boundary(
    *,
    domain_geom,
    points_xy: np.ndarray,
    mask_to_domain: bool = True,
) -> np.ndarray:
    """Distance to the boundary of the domain (outer + hole boundaries)."""

    _ensure_shapely()
    pts = np.asarray(points_xy, dtype=float)
    parts = tuple(shapely_to_parts(domain_geom))
    d = distance_to_boundary(parts, pts)
    if mask_to_domain:
        inside = mask_points_in_poly(parts, pts)
        d = d.astype(float)
        d[~inside] = np.nan
    return d


def ref_distance_to_outer_boundary(
    *,
    outer_geom,
    domain_geom,
    points_xy: np.ndarray,
    mask_to_domain: bool = True,
) -> np.ndarray:
    """Distance to the outer boundary only (ignores holes).

    outer_geom:
        The outer cell boundary polygon(s) *without* holes.
    domain_geom:
        Final domain geometry (outer - holes). Used only for masking.
    """

    _ensure_shapely()
    pts = np.asarray(points_xy, dtype=float)
    outer_parts = tuple(shapely_to_parts(outer_geom))
    d = distance_to_boundary(outer_parts, pts)
    if mask_to_domain:
        dom_parts = tuple(shapely_to_parts(domain_geom))
        inside = mask_points_in_poly(dom_parts, pts)
        d = d.astype(float)
        d[~inside] = np.nan
    return d


def ref_distance_to_seed(
    *,
    seed_geom,
    domain_geom,
    points_xy: np.ndarray,
    mask_to_domain: bool = True,
) -> np.ndarray:
    """Distance to a seed ROI (0 inside the seed)."""

    _ensure_shapely()
    pts = np.asarray(points_xy, dtype=float)
    seed_parts = tuple(shapely_to_parts(seed_geom))
    inside_seed = mask_points_in_poly(seed_parts, pts)
    d_boundary = distance_to_boundary(seed_parts, pts)
    d = np.where(inside_seed, 0.0, d_boundary)
    if mask_to_domain:
        dom_parts = tuple(shapely_to_parts(domain_geom))
        inside = mask_points_in_poly(dom_parts, pts)
        d = d.astype(float)
        d[~inside] = np.nan
    return d


def ref_distance_to_nucleus(
    *,
    nucleus_geom,
    domain_geom,
    points_xy: np.ndarray,
    mask_to_domain: bool = True,
) -> np.ndarray:
    """Distance to nucleus boundary (0 at the nucleus boundary; >0 in cytoplasm).

    Notes
    -----
    The nucleus is expected to be excluded from the domain; points in the nucleus
    should normally not exist after masking.
    """

    _ensure_shapely()
    pts = np.asarray(points_xy, dtype=float)
    nuc_parts = tuple(shapely_to_parts(nucleus_geom))
    d = distance_to_boundary(nuc_parts, pts)
    if mask_to_domain:
        dom_parts = tuple(shapely_to_parts(domain_geom))
        inside = mask_points_in_poly(dom_parts, pts)
        d = d.astype(float)
        d[~inside] = np.nan
    return d


def ref_geodesic_from_model(
    *,
    model: MembraneModel,
    points_xy: np.ndarray,
    mask_to_domain: bool = True,
) -> np.ndarray:
    """Project points to nearest skeleton node and return geodesic distance-to-seed.

    This is the general mechanism behind "membrane geodesic" and "neurite" profiles.
    """

    pts = np.asarray(points_xy, dtype=float)

    # Optionally restrict to the domain first.
    if mask_to_domain:
        parts = tuple(shapely_to_parts(model.membrane_geom))
        inside = mask_points_in_poly(parts, pts)
    else:
        inside = np.ones(len(pts), dtype=bool)

    out = np.full(len(pts), np.nan, dtype=float)
    if not np.any(inside):
        return out

    pts_in = pts[inside]
    tree = cKDTree(np.asarray(model.node_xy, dtype=float))
    _, nn = tree.query(pts_in, k=1)
    nn = nn.astype(int)
    d = np.asarray(model.node_dist_to_zc, dtype=float)[nn]

    out[inside] = d
    return out
