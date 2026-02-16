"""
ripley_backend.py

Backend utilities for spatial point-pattern clustering analysis (Ripley's K / Besag's L)
on 2D localization point clouds (e.g., ThunderSTORM CSV exports).

Key features:
- Works with arbitrary polygonal ROIs (including holes / multipolygons).
- Supports "border" (reduced sample) edge correction (robust, interpretable).
- Optional ROI subdivision (grid / radial shells / boundary-distance shells).
- Monte Carlo CSR (complete spatial randomness) envelopes.
- Rich logging suitable for scientific audit trails.

Author: ChatGPT (GPT-5.2 Pro)
License: MIT (you may adapt freely; keep attribution if you like)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import math
import json
import numpy as np
import pandas as pd

from matplotlib.path import Path
from scipy.spatial import cKDTree

try:
    import shapely.geometry as _shgeom
    import shapely.ops as _shops
except Exception:  # pragma: no cover
    _shgeom = None
    _shops = None


ArrayLike = Union[np.ndarray, Sequence[Sequence[float]]]


# ----------------------------
# Configuration dataclasses
# ----------------------------

@dataclass(frozen=True)
class RipleyParams:
    r_min: float = 0.0
    r_max: float = 500.0
    dr: float = 5.0
    edge_correction: str = "border"  # "border" or "none"
    # Performance / safety
    max_points: int = 200_000  # soft limit (you can raise)
    random_downsample: Optional[int] = None  # set e.g. 50_000 if needed


@dataclass(frozen=True)
class CSRParams:
    n_simulations: int = 199
    alpha: float = 0.05  # pointwise envelope (alpha/2, 1-alpha/2)
    seed: Optional[int] = 0


@dataclass(frozen=True)
class SubdivisionParams:
    mode: str = "none"  # none|grid|radial|boundary_shells
    # grid
    grid_rows: int = 1
    grid_cols: int = 1
    # radial (centroid-based)
    radial_shells: int = 1
    # boundary-distance shells (distance to boundary)
    boundary_shells: int = 1


# ----------------------------
# Logging helper (backend-safe)
# ----------------------------

class SimpleLogger:
    """Minimal logger interface used by backend. GUI can provide a compatible object."""
    def info(self, msg: str) -> None:  # pragma: no cover
        print(msg)

    def warning(self, msg: str) -> None:  # pragma: no cover
        print("WARNING:", msg)

    def error(self, msg: str) -> None:  # pragma: no cover
        print("ERROR:", msg)


# ----------------------------
# CSV helpers
# ----------------------------

def auto_detect_xy_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Attempt to detect x/y localization columns from common ThunderSTORM exports.

    Returns
    -------
    (x_col, y_col)

    Raises
    ------
    ValueError if no reasonable guess.
    """
    cols = list(df.columns)

    # Canonical ThunderSTORM style
    candidates = [
        ("x [nm]", "y [nm]"),
        ("x_nm", "y_nm"),
        ("x (nm)", "y (nm)"),
        ("x", "y"),
        ("X", "Y"),
        ("x [px]", "y [px]"),
        ("x_px", "y_px"),
    ]
    for x, y in candidates:
        if x in cols and y in cols:
            return x, y

    # Heuristic: choose first pair among columns containing 'x' and 'y'
    x_like = [c for c in cols if str(c).strip().lower().startswith("x")]
    y_like = [c for c in cols if str(c).strip().lower().startswith("y")]
    if x_like and y_like:
        return x_like[0], y_like[0]

    raise ValueError("Could not auto-detect x/y columns. Please select columns manually.")


# ----------------------------
# Geometry (vectorized masks + distances)
# ----------------------------

def _close_ring(ring_xy: np.ndarray) -> np.ndarray:
    ring_xy = np.asarray(ring_xy, dtype=float)
    if ring_xy.ndim != 2 or ring_xy.shape[1] != 2:
        raise ValueError("Ring must be an array of shape (N, 2) in (x, y).")
    if len(ring_xy) < 3:
        raise ValueError("Ring must have at least 3 vertices.")
    if not np.allclose(ring_xy[0], ring_xy[-1]):
        ring_xy = np.vstack([ring_xy, ring_xy[0]])
    return ring_xy


def polygon_area_xy(exterior_xy: np.ndarray) -> float:
    """Signed area (absolute value is polygon area). Expects closed or open ring."""
    ring = np.asarray(exterior_xy, dtype=float)
    if not np.allclose(ring[0], ring[-1]):
        ring = np.vstack([ring, ring[0]])
    x = ring[:, 0]
    y = ring[:, 1]
    return 0.5 * float(np.abs(np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:])))


def polygon_centroid_xy(exterior_xy: np.ndarray) -> Tuple[float, float]:
    """Centroid of a simple polygon (exterior only)."""
    ring = np.asarray(exterior_xy, dtype=float)
    if not np.allclose(ring[0], ring[-1]):
        ring = np.vstack([ring, ring[0]])
    x = ring[:, 0]
    y = ring[:, 1]
    a = (np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:]))
    A = 0.5 * a
    if abs(A) < 1e-12:
        # Degenerate, fallback to mean
        return float(np.mean(x[:-1])), float(np.mean(y[:-1]))
    cx = (1.0 / (6.0 * A)) * np.sum((x[:-1] + x[1:]) * (x[:-1] * y[1:] - x[1:] * y[:-1]))
    cy = (1.0 / (6.0 * A)) * np.sum((y[:-1] + y[1:]) * (x[:-1] * y[1:] - x[1:] * y[:-1]))
    return float(cx), float(cy)


@dataclass(frozen=True)
class PolyPart:
    """One polygon component (exterior + holes)."""
    exterior: np.ndarray  # (N,2) (x,y), closed
    holes: Tuple[np.ndarray, ...] = ()  # each (M,2) (x,y), closed


def _path_from_ring(ring_xy: np.ndarray) -> Path:
    ring = np.asarray(ring_xy, dtype=float)
    if not np.allclose(ring[0], ring[-1]):
        ring = np.vstack([ring, ring[0]])
    codes = np.full(len(ring), Path.LINETO, dtype=np.uint8)
    codes[0] = Path.MOVETO
    return Path(ring, codes)


def _ring_distance(points_xy: np.ndarray, ring_xy: np.ndarray) -> np.ndarray:
    """
    Minimum distance from each point to a closed polyline (ring boundary).
    Vectorized across points. Complexity O(N * E) where E ~ #edges.
    """
    P = np.asarray(points_xy, dtype=float)
    ring = np.asarray(ring_xy, dtype=float)
    if not np.allclose(ring[0], ring[-1]):
        ring = np.vstack([ring, ring[0]])

    # Drop duplicate last point for edge iteration
    v = ring[:-1]
    w = ring[1:]
    min_dist2 = np.full(P.shape[0], np.inf, dtype=float)

    for a, b in zip(v, w):
        ab = b - a
        ap = P - a
        denom = float(np.dot(ab, ab))
        if denom <= 0:
            # Degenerate edge
            d2 = np.sum((P - a) ** 2, axis=1)
        else:
            t = (ap @ ab) / denom
            t = np.clip(t, 0.0, 1.0)
            proj = a + np.outer(t, ab)
            d2 = np.sum((P - proj) ** 2, axis=1)
        min_dist2 = np.minimum(min_dist2, d2)

    return np.sqrt(min_dist2)


def mask_points_in_poly(parts: Sequence[PolyPart], points_xy: np.ndarray) -> np.ndarray:
    """
    Vectorized point-in-polygon for Polygon/MultiPolygon (supports holes).
    A point is inside if inside any part's exterior and not inside its holes.
    """
    pts = np.asarray(points_xy, dtype=float)
    inside_any = np.zeros(len(pts), dtype=bool)

    for part in parts:
        ext_path = _path_from_ring(part.exterior)
        inside = ext_path.contains_points(pts)
        if part.holes:
            for h in part.holes:
                hole_path = _path_from_ring(h)
                inside &= ~hole_path.contains_points(pts)
        inside_any |= inside

    return inside_any


def distance_to_boundary(parts: Sequence[PolyPart], points_xy: np.ndarray) -> np.ndarray:
    """Minimum distance from points to any boundary ring (exterior or holes)."""
    pts = np.asarray(points_xy, dtype=float)
    dmin = np.full(len(pts), np.inf, dtype=float)
    for part in parts:
        dmin = np.minimum(dmin, _ring_distance(pts, part.exterior))
        for h in part.holes:
            dmin = np.minimum(dmin, _ring_distance(pts, h))
    return dmin


def shapely_to_parts(geom) -> List[PolyPart]:
    """
    Convert shapely Polygon or MultiPolygon into PolyPart list.
    Holes are kept.
    """
    if _shgeom is None:
        raise ImportError("shapely is required for this conversion but is not available.")

    if geom.is_empty:
        return []

    parts: List[PolyPart] = []
    if isinstance(geom, _shgeom.Polygon):
        ext = np.asarray(geom.exterior.coords, dtype=float)
        holes = tuple(np.asarray(r.coords, dtype=float) for r in geom.interiors)
        parts.append(PolyPart(exterior=_close_ring(ext), holes=tuple(_close_ring(h) for h in holes)))
    elif isinstance(geom, _shgeom.MultiPolygon):
        for g in geom.geoms:
            parts.extend(shapely_to_parts(g))
    else:
        raise TypeError(f"Unsupported shapely geometry type: {type(geom)}")

    return parts


def vertices_to_shapely_polygon(vertices_xy: np.ndarray):
    if _shgeom is None:
        raise ImportError("shapely is required but is not available.")
    v = np.asarray(vertices_xy, dtype=float)
    if v.shape[0] < 3:
        raise ValueError("Polygon needs >=3 vertices.")
    if not np.allclose(v[0], v[-1]):
        v = np.vstack([v, v[0]])
    poly = _shgeom.Polygon(v)
    # Clean potential self-intersections
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


# ----------------------------
# Ripley K / Besag L
# ----------------------------

def make_radii(r_min: float, r_max: float, dr: float) -> np.ndarray:
    if dr <= 0:
        raise ValueError("dr must be > 0.")
    if r_max <= r_min:
        raise ValueError("r_max must be > r_min.")
    # include r_max if divisible-ish
    n = int(math.floor((r_max - r_min) / dr)) + 1
    radii = r_min + dr * np.arange(n, dtype=float)
    # Ensure last isn't above r_max due to floating error
    radii = radii[radii <= (r_max + 1e-12)]
    return radii


def estimate_K_L(
    points_xy: np.ndarray,
    window_area: float,
    radii: np.ndarray,
    edge_correction: str = "border",
    boundary_dist: Optional[np.ndarray] = None,
    logger: Optional[SimpleLogger] = None,
) -> Dict[str, np.ndarray]:
    """
    Estimate Ripley's K(r) and Besag's L(r) for a 2D point pattern.

    Estimator used:
      - edge_correction="border": reduced-sample (border) correction using boundary_dist.
      - edge_correction="none": uncorrected.

    Parameters
    ----------
    points_xy : (N,2) array
        Coordinates in same units as radii.
    window_area : float
        Area of the observation window (ROI).
    radii : (M,) array
        Radii at which to evaluate.
    edge_correction : str
        "border" or "none"
    boundary_dist : (N,) array
        Distance to ROI boundary for each point (required for border correction).

    Returns
    -------
    dict with keys: r, K, L, L_minus_r, n_points, n_eligible, intensity
    """
    if logger is None:
        logger = SimpleLogger()

    pts = np.asarray(points_xy, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points_xy must be (N,2).")
    n = pts.shape[0]
    if n < 2:
        K = np.full_like(radii, np.nan, dtype=float)
        L = np.full_like(radii, np.nan, dtype=float)
        return {
            "r": radii,
            "K": K,
            "L": L,
            "L_minus_r": L - radii,
            "n_points": np.array([n], dtype=int),
            "n_eligible": np.zeros_like(radii, dtype=int),
            "intensity": np.array([np.nan], dtype=float),
        }

    if window_area <= 0:
        raise ValueError("window_area must be > 0.")

    if edge_correction not in ("border", "none"):
        raise ValueError("edge_correction must be 'border' or 'none'.")

    if edge_correction == "border":
        if boundary_dist is None:
            raise ValueError("boundary_dist is required for border correction.")
        bd = np.asarray(boundary_dist, dtype=float)
        if bd.shape[0] != n:
            raise ValueError("boundary_dist must have same length as points.")
    else:
        bd = None

    r_max = float(np.max(radii))
    logger.info(f"Building KD-tree for N={n} points; querying pairs within r_max={r_max:g} ...")
    tree = cKDTree(pts)

    pairs = tree.query_pairs(r_max, output_type="ndarray")
    if pairs.size == 0:
        logger.warning("No point pairs found within r_max. K will be ~0 for all r.")
        degrees = np.zeros(n, dtype=np.int32)
        K = np.zeros_like(radii, dtype=float)
        n_eligible = np.zeros_like(radii, dtype=int)
        lam = n / window_area
        for k, r in enumerate(radii):
            if edge_correction == "border":
                eligible = bd >= r
            else:
                eligible = np.ones(n, dtype=bool)
            n_el = int(np.sum(eligible))
            n_eligible[k] = n_el
            if n_el == 0:
                K[k] = np.nan
            else:
                K[k] = 0.0
        L = np.sqrt(K / math.pi)
        return {
            "r": radii,
            "K": K,
            "L": L,
            "L_minus_r": L - radii,
            "n_points": np.array([n], dtype=int),
            "n_eligible": n_eligible,
            "intensity": np.array([lam], dtype=float),
        }

    i = pairs[:, 0]
    j = pairs[:, 1]
    d = np.linalg.norm(pts[i] - pts[j], axis=1)
    order = np.argsort(d)
    i = i[order]
    j = j[order]
    d = d[order]

    logger.info(f"Found {len(d):,} pairs <= r_max. Estimating K(r) on {len(radii)} radii ...")

    degrees = np.zeros(n, dtype=np.int32)
    K = np.full_like(radii, np.nan, dtype=float)
    n_eligible = np.zeros_like(radii, dtype=int)

    lam = n / window_area
    idx = 0
    for k, r in enumerate(radii):
        # Increment degrees with newly included pairs
        while idx < len(d) and d[idx] <= r:
            degrees[i[idx]] += 1
            degrees[j[idx]] += 1
            idx += 1

        if edge_correction == "border":
            eligible = bd >= r
        else:
            eligible = np.ones(n, dtype=bool)

        n_el = int(np.sum(eligible))
        n_eligible[k] = n_el
        if n_el == 0:
            K[k] = np.nan
        else:
            K[k] = float(np.sum(degrees[eligible])) / (lam * n_el)

    L = np.sqrt(K / math.pi)
    return {
        "r": radii,
        "K": K,
        "L": L,
        "L_minus_r": L - radii,
        "n_points": np.array([n], dtype=int),
        "n_eligible": n_eligible,
        "intensity": np.array([lam], dtype=float),
    }



# ----------------------------
# Cross (bivariate) Ripley K / L
# ----------------------------

def estimate_cross_K_L(
    points1_xy: np.ndarray,
    points2_xy: np.ndarray,
    window_area: float,
    radii: np.ndarray,
    edge_correction: str = "border",
    boundary_dist1: Optional[np.ndarray] = None,
    logger: Optional[SimpleLogger] = None,
) -> Dict[str, np.ndarray]:
    """Estimate bivariate (cross) Ripley's K12(r) and Besag's L12(r).

    The implementation mirrors :func:`estimate_K_L` but counts *cross* pairs
    between two point sets (1 -> 2).

    Border correction
    -----------------
    If ``edge_correction == 'border'``, a reduced-sample estimator is used:
    only points in set 1 that are at least distance ``r`` from the window
    boundary are considered eligible. The normalization uses the intensity of
    set 2 (lambda2 = n2 / area) and the number of eligible points in set 1.

    Returns
    -------
    dict with keys: r, K, L, L_minus_r, n_points1, n_points2, n_eligible, intensity2
    """

    if logger is None:
        logger = SimpleLogger()

    p1 = np.asarray(points1_xy, dtype=float)
    p2 = np.asarray(points2_xy, dtype=float)

    if p1.ndim != 2 or p1.shape[1] != 2:
        raise ValueError("points1_xy must be (N1,2).")
    if p2.ndim != 2 or p2.shape[1] != 2:
        raise ValueError("points2_xy must be (N2,2).")

    n1 = int(p1.shape[0])
    n2 = int(p2.shape[0])

    if n1 < 1 or n2 < 1:
        K = np.full_like(radii, np.nan, dtype=float)
        L = np.full_like(radii, np.nan, dtype=float)
        return {
            "r": radii,
            "K": K,
            "L": L,
            "L_minus_r": L - radii,
            "n_points1": np.array([n1], dtype=int),
            "n_points2": np.array([n2], dtype=int),
            "n_eligible": np.zeros_like(radii, dtype=int),
            "intensity2": np.array([np.nan], dtype=float),
        }

    if window_area <= 0:
        raise ValueError("window_area must be > 0.")

    if edge_correction not in ("border", "none"):
        raise ValueError("edge_correction must be 'border' or 'none'.")

    if edge_correction == "border":
        if boundary_dist1 is None:
            raise ValueError("boundary_dist1 is required for border correction.")
        bd1 = np.asarray(boundary_dist1, dtype=float)
        if bd1.shape[0] != n1:
            raise ValueError("boundary_dist1 must have same length as points1.")
    else:
        bd1 = None

    r_max = float(np.max(radii))

    logger.info(f"Building KD-trees for cross-Ripley: N1={n1}, N2={n2}, r_max={r_max:g} ...")
    t1 = cKDTree(p1)
    t2 = cKDTree(p2)

    # Compute all cross distances <= r_max.
    # SciPy's sparse_distance_matrix is efficient and returns a sparse COO matrix.
    sdm = t1.sparse_distance_matrix(t2, r_max, output_type="coo_matrix")
    if sdm.nnz == 0:
        logger.warning("No cross pairs found within r_max. K12 will be ~0 for all r.")
        K = np.zeros_like(radii, dtype=float)
        n_eligible = np.zeros_like(radii, dtype=int)
        lam2 = n2 / window_area
        for k, r in enumerate(radii):
            eligible = (bd1 >= r) if (edge_correction == "border") else np.ones(n1, dtype=bool)
            n_el = int(np.sum(eligible))
            n_eligible[k] = n_el
            K[k] = np.nan if n_el == 0 else 0.0
        L = np.sqrt(K / math.pi)
        return {
            "r": radii,
            "K": K,
            "L": L,
            "L_minus_r": L - radii,
            "n_points1": np.array([n1], dtype=int),
            "n_points2": np.array([n2], dtype=int),
            "n_eligible": n_eligible,
            "intensity2": np.array([lam2], dtype=float),
        }

    i = np.asarray(sdm.row, dtype=np.int32)
    d = np.asarray(sdm.data, dtype=float)

    order = np.argsort(d)
    i = i[order]
    d = d[order]

    logger.info(f"Found {len(d):,} cross pairs <= r_max. Estimating K12(r) on {len(radii)} radii ...")

    # degree per point in set 1 (number of set-2 neighbors <= current r)
    deg = np.zeros(n1, dtype=np.int32)

    K = np.full_like(radii, np.nan, dtype=float)
    n_eligible = np.zeros_like(radii, dtype=int)

    lam2 = n2 / window_area
    idx = 0
    for k, r in enumerate(radii):
        while idx < len(d) and d[idx] <= r:
            deg[i[idx]] += 1
            idx += 1

        if edge_correction == "border":
            eligible = bd1 >= r
        else:
            eligible = np.ones(n1, dtype=bool)

        n_el = int(np.sum(eligible))
        n_eligible[k] = n_el
        if n_el == 0:
            K[k] = np.nan
        else:
            K[k] = float(np.sum(deg[eligible])) / (lam2 * n_el)

    L = np.sqrt(K / math.pi)
    return {
        "r": radii,
        "K": K,
        "L": L,
        "L_minus_r": L - radii,
        "n_points1": np.array([n1], dtype=int),
        "n_points2": np.array([n2], dtype=int),
        "n_eligible": n_eligible,
        "intensity2": np.array([lam2], dtype=float),
    }


def csr_envelope_cross_LminusR(
    window_parts: Sequence[PolyPart],
    window_area: float,
    points1_xy: np.ndarray,
    n_points2: int,
    radii: np.ndarray,
    edge_correction: str,
    n_simulations: int,
    alpha: float,
    seed: Optional[int],
    logger: Optional[SimpleLogger] = None,
) -> Dict[str, np.ndarray]:
    """Monte Carlo CSR envelopes for cross L12(r)-r (1 fixed, 2 simulated).

    Interpretation
    -------------
    This tests the deviation from *independence* between pattern 1 and a uniform
    (CSR) pattern 2, while keeping the number of points in pattern 2 fixed.

    Returns
    -------
    dict: Lmr_lo, Lmr_hi, Lmr_mean
    """

    if logger is None:
        logger = SimpleLogger()

    if n_simulations <= 0:
        raise ValueError("n_simulations must be > 0.")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0,1).")
    if n_points2 < 0:
        raise ValueError("n_points2 must be >= 0")

    p1 = np.asarray(points1_xy, dtype=float)
    if p1.ndim != 2 or p1.shape[1] != 2:
        raise ValueError("points1_xy must be (N1,2).")

    rng = np.random.default_rng(seed)
    lmr_sims = np.zeros((n_simulations, len(radii)), dtype=float)

    bd1 = None
    if edge_correction == "border":
        bd1 = distance_to_boundary(window_parts, p1)

    logger.info(f"Cross-CSR envelope: {n_simulations} simulations; alpha={alpha:g}; seed={seed}")
    for s in range(n_simulations):
        pts2 = _uniform_points_in_window(window_parts, int(n_points2), rng)
        est = estimate_cross_K_L(
            p1,
            pts2,
            window_area,
            radii,
            edge_correction=edge_correction,
            boundary_dist1=bd1,
            logger=logger,
        )
        lmr_sims[s] = est["L_minus_r"]

        if (s + 1) % max(1, n_simulations // 10) == 0:
            logger.info(f"  Cross CSR progress: {s+1}/{n_simulations}")

    lo_q = alpha / 2.0
    hi_q = 1.0 - alpha / 2.0

    return {
        "Lmr_lo": np.quantile(lmr_sims, lo_q, axis=0),
        "Lmr_hi": np.quantile(lmr_sims, hi_q, axis=0),
        "Lmr_mean": np.mean(lmr_sims, axis=0),
    }


# ----------------------------
# CSR simulation and envelopes
# ----------------------------

def _uniform_points_in_window(parts: Sequence[PolyPart], n: int, rng: np.random.Generator,
                              bbox: Optional[Tuple[float,float,float,float]] = None,
                              max_batches: int = 10_000) -> np.ndarray:
    """
    Rejection sampling of uniform points in an arbitrary polygonal window (with holes).
    Vectorized using matplotlib.path for speed.

    Parameters
    ----------
    parts : list of PolyPart
    n : number of points
    rng : numpy Generator
    bbox : (xmin, xmax, ymin, ymax) optional
        If None, computed from parts.
    """
    if n <= 0:
        return np.zeros((0,2), dtype=float)

    # Compute bounding box from exteriors
    if bbox is None:
        xs = []
        ys = []
        for part in parts:
            xs.append(part.exterior[:,0])
            ys.append(part.exterior[:,1])
        xmin = float(np.min(np.concatenate(xs)))
        xmax = float(np.max(np.concatenate(xs)))
        ymin = float(np.min(np.concatenate(ys)))
        ymax = float(np.max(np.concatenate(ys)))
    else:
        xmin, xmax, ymin, ymax = map(float, bbox)

    out = np.zeros((n, 2), dtype=float)
    filled = 0

    # Adaptive batch size
    batch = max(10_000, n * 5)
    batches_used = 0

    while filled < n:
        batches_used += 1
        if batches_used > max_batches:
            raise RuntimeError("Rejection sampling did not converge. Window may be too small or bbox too large.")

        cand = np.column_stack([
            rng.uniform(xmin, xmax, size=batch),
            rng.uniform(ymin, ymax, size=batch),
        ])
        mask = mask_points_in_poly(parts, cand)
        kept = cand[mask]
        if kept.size == 0:
            continue
        take = min(len(kept), n - filled)
        out[filled:filled+take] = kept[:take]
        filled += take

    return out


def csr_envelope_LminusR(
    window_parts: Sequence[PolyPart],
    window_area: float,
    n_points: int,
    radii: np.ndarray,
    edge_correction: str,
    n_simulations: int,
    alpha: float,
    seed: Optional[int],
    logger: Optional[SimpleLogger] = None,
) -> Dict[str, np.ndarray]:
    """
    Monte Carlo CSR envelopes for L(r)-r.

    Returns dict: Lmr_lo, Lmr_hi, Lmr_mean
    """
    if logger is None:
        logger = SimpleLogger()

    if n_simulations <= 0:
        raise ValueError("n_simulations must be > 0.")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0,1).")

    rng = np.random.default_rng(seed)

    lmr_sims = np.zeros((n_simulations, len(radii)), dtype=float)

    logger.info(f"CSR envelope: {n_simulations} simulations; alpha={alpha:g}; seed={seed}")
    for s in range(n_simulations):
        pts = _uniform_points_in_window(window_parts, n_points, rng)
        bd = None
        if edge_correction == "border":
            bd = distance_to_boundary(window_parts, pts)

        est = estimate_K_L(pts, window_area, radii, edge_correction=edge_correction, boundary_dist=bd, logger=logger)
        lmr_sims[s] = est["L_minus_r"]

        if (s + 1) % max(1, n_simulations // 10) == 0:
            logger.info(f"  CSR progress: {s+1}/{n_simulations}")

    lo = np.quantile(lmr_sims, alpha / 2.0, axis=0)
    hi = np.quantile(lmr_sims, 1.0 - alpha / 2.0, axis=0)
    mean = np.mean(lmr_sims, axis=0)
    return {"Lmr_lo": lo, "Lmr_hi": hi, "Lmr_mean": mean}


# ----------------------------
# Subdivision windows (grid / shells)
# ----------------------------

@dataclass(frozen=True)
class WindowSpec:
    """
    Defines an analysis window (ROI or subregion).

    geometry: shapely Polygon/MultiPolygon (preferred) OR (parts, area) representation.
    parts: list of PolyPart used for vectorized contains/distance.
    area: float window area
    name: human-readable
    """
    name: str
    parts: Tuple[PolyPart, ...]
    area: float


def build_windows_from_roi(
    roi_geom,  # shapely geometry
    subdivision: SubdivisionParams,
    logger: Optional[SimpleLogger] = None,
) -> List[WindowSpec]:
    """
    Create a list of WindowSpec objects to analyze:
    - If subdivision.mode == "none": returns [ROI]
    - grid: ROI intersected with each grid tile
    - radial: ROI intersected with centroid-based annuli
    - boundary_shells: shells based on distance-to-boundary bands

    All returned windows use shapely-derived geometry; holes are kept.

    Notes
    -----
    This requires shapely.
    """
    if logger is None:
        logger = SimpleLogger()
    if _shgeom is None:
        raise ImportError("shapely is required for subdivision but is not available.")

    if roi_geom.is_empty:
        return []

    roi_geom = roi_geom.buffer(0)  # clean
    roi_area = float(roi_geom.area)
    if roi_area <= 0:
        return []

    roi_parts = tuple(shapely_to_parts(roi_geom))
    out: List[WindowSpec] = []

    mode = subdivision.mode.lower().strip()
    if mode == "none":
        out.append(WindowSpec(name="ROI", parts=roi_parts, area=roi_area))
        return out

    # Compute ROI bbox for grid
    minx, miny, maxx, maxy = roi_geom.bounds

    if mode == "grid":
        rows = max(1, int(subdivision.grid_rows))
        cols = max(1, int(subdivision.grid_cols))
        logger.info(f"Building grid subdivision: {rows}x{cols}")

        dx = (maxx - minx) / cols
        dy = (maxy - miny) / rows

        for rr in range(rows):
            for cc in range(cols):
                x0 = minx + cc * dx
                x1 = minx + (cc + 1) * dx
                y0 = miny + rr * dy
                y1 = miny + (rr + 1) * dy
                tile = _shgeom.Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
                win = roi_geom.intersection(tile)
                if win.is_empty:
                    continue
                area = float(win.area)
                if area <= 0:
                    continue
                parts = tuple(shapely_to_parts(win))
                out.append(WindowSpec(name=f"tile_r{rr+1}_c{cc+1}", parts=parts, area=area))

        if not out:
            logger.warning("Grid subdivision produced no non-empty subwindows; falling back to whole ROI.")
            out.append(WindowSpec(name="ROI", parts=roi_parts, area=roi_area))
        return out

    if mode == "radial":
        shells = max(1, int(subdivision.radial_shells))
        # Centroid: use shapely centroid (handles holes)
        c = roi_geom.centroid
        cx, cy = float(c.x), float(c.y)

        # Maximum radius: farthest vertex of ROI bbox corner
        r_max = max(
            math.hypot(minx - cx, miny - cy),
            math.hypot(minx - cx, maxy - cy),
            math.hypot(maxx - cx, miny - cy),
            math.hypot(maxx - cx, maxy - cy),
        )
        edges = np.linspace(0.0, r_max, shells + 1, dtype=float)

        logger.info(f"Building radial shells around centroid ({cx:g},{cy:g}) with {shells} bins up to r={r_max:g}")

        for k in range(shells):
            r0 = float(edges[k])
            r1 = float(edges[k + 1])
            outer = _shgeom.Point(cx, cy).buffer(r1, resolution=64)
            if r0 <= 0:
                ring = outer
            else:
                inner = _shgeom.Point(cx, cy).buffer(r0, resolution=64)
                ring = outer.difference(inner)
            win = roi_geom.intersection(ring)
            if win.is_empty:
                continue
            area = float(win.area)
            if area <= 0:
                continue
            parts = tuple(shapely_to_parts(win))
            out.append(WindowSpec(name=f"radial_{k+1}_r{r0:g}-{r1:g}", parts=parts, area=area))

        if not out:
            logger.warning("Radial subdivision produced no non-empty subwindows; falling back to whole ROI.")
            out.append(WindowSpec(name="ROI", parts=roi_parts, area=roi_area))
        return out

    if mode in ("boundary_shells", "boundary", "boundaryshells"):
        shells = max(1, int(subdivision.boundary_shells))
        # Distance-to-boundary max is approximated by the inradius proxy:
        # Use maximum distance of a dense random sample to boundary? That's expensive.
        # Instead, use shapely's "maximum inscribed circle" is not available in 1.7.
        # We'll approximate using the maximum of distance-to-boundary over ROI centroid and bbox midpoints,
        # then refine with a small Monte Carlo sample.
        logger.info(f"Building boundary-distance shells (bands) with {shells} bins")

        # Prepare some probe points inside ROI
        rng = np.random.default_rng(0)
        probe = _uniform_points_in_window(roi_parts, min(5000, int(max(1000, roi_area / (max(1.0, (maxx-minx)*(maxy-miny))) * 5000))), rng)
        bd_probe = distance_to_boundary(roi_parts, probe)
        d_max = float(np.max(bd_probe)) if bd_probe.size else 0.0
        if d_max <= 0:
            logger.warning("Could not estimate interior distance scale; falling back to whole ROI.")
            out.append(WindowSpec(name="ROI", parts=roi_parts, area=roi_area))
            return out

        edges = np.linspace(0.0, d_max, shells + 1, dtype=float)

        for k in range(shells):
            d0 = float(edges[k])
            d1 = float(edges[k + 1])

            # Shell = erosion by d0 minus erosion by d1
            outer = roi_geom.buffer(-d0)
            inner = roi_geom.buffer(-d1)
            if outer.is_empty:
                continue
            if inner.is_empty:
                shell_geom = outer
            else:
                shell_geom = outer.difference(inner)

            if shell_geom.is_empty:
                continue
            area = float(shell_geom.area)
            if area <= 0:
                continue
            parts = tuple(shapely_to_parts(shell_geom))
            out.append(WindowSpec(name=f"boundary_{k+1}_d{d0:g}-{d1:g}", parts=parts, area=area))

        if not out:
            logger.warning("Boundary-shell subdivision produced no non-empty subwindows; falling back to whole ROI.")
            out.append(WindowSpec(name="ROI", parts=roi_parts, area=roi_area))
        return out

    logger.warning(f"Unknown subdivision mode '{subdivision.mode}', using whole ROI.")
    out.append(WindowSpec(name="ROI", parts=roi_parts, area=roi_area))
    return out


# ----------------------------
# High-level analysis helper
# ----------------------------

@dataclass
class AnalysisResult:
    roi_name: str
    window_name: str
    n_points: int
    area: float
    intensity: float
    r_peak: float
    lmr_max: float
    auc_pos: float
    curve: pd.DataFrame  # r, K, L, L_minus_r, env_lo, env_hi, n_eligible


def analyze_points_in_windows(
    points_xy: np.ndarray,
    roi_name: str,
    windows: Sequence[WindowSpec],
    ripley: RipleyParams,
    csr: Optional[CSRParams],
    logger: Optional[SimpleLogger] = None,
) -> List[AnalysisResult]:
    """
    Run Ripley/Besag analysis for each WindowSpec on the same parent point cloud.

    points_xy are assumed to already be in the same coordinate system as windows.
    Each window filters points inside itself and computes K/L.

    CSR envelope is computed per window (conditional on observed n in that window).
    """
    if logger is None:
        logger = SimpleLogger()

    radii = make_radii(ripley.r_min, ripley.r_max, ripley.dr)

    all_results: List[AnalysisResult] = []

    for win in windows:
        mask = mask_points_in_poly(win.parts, points_xy)
        pts_w = points_xy[mask]
        n = int(pts_w.shape[0])

        logger.info(f"[{roi_name} | {win.name}] Points in window: {n:,} ; Area={win.area:g}")

        # Downsample if requested
        if ripley.random_downsample is not None and n > ripley.random_downsample:
            rng = np.random.default_rng(0)
            idx = rng.choice(n, size=int(ripley.random_downsample), replace=False)
            pts_w = pts_w[idx]
            n = int(pts_w.shape[0])
            logger.warning(f"[{roi_name} | {win.name}] Downsampled to {n:,} points for analysis.")

        bd = None
        if ripley.edge_correction == "border":
            bd = distance_to_boundary(win.parts, pts_w)

        est = estimate_K_L(
            pts_w, win.area, radii,
            edge_correction=ripley.edge_correction,
            boundary_dist=bd,
            logger=logger,
        )

        env_lo = np.full_like(radii, np.nan, dtype=float)
        env_hi = np.full_like(radii, np.nan, dtype=float)
        env_mean = np.full_like(radii, np.nan, dtype=float)

        if csr is not None and csr.n_simulations > 0 and n >= 2:
            env = csr_envelope_LminusR(
                window_parts=win.parts,
                window_area=win.area,
                n_points=n,
                radii=radii,
                edge_correction=ripley.edge_correction,
                n_simulations=csr.n_simulations,
                alpha=csr.alpha,
                seed=csr.seed,
                logger=logger,
            )
            env_lo = env["Lmr_lo"]
            env_hi = env["Lmr_hi"]
            env_mean = env["Lmr_mean"]

        lmr = est["L_minus_r"]
        # Summary metrics
        if np.all(np.isnan(lmr)):
            r_peak = float("nan")
            lmr_max = float("nan")
            auc_pos = float("nan")
        else:
            kmax = int(np.nanargmax(lmr))
            r_peak = float(radii[kmax])
            lmr_max = float(lmr[kmax])
            auc_pos = float(np.trapz(np.maximum(0.0, lmr), radii))

        curve = pd.DataFrame({
            "r": radii,
            "K": est["K"],
            "L": est["L"],
            "L_minus_r": lmr,
            "n_eligible": est["n_eligible"],
            "env_LminusR_lo": env_lo,
            "env_LminusR_hi": env_hi,
            "env_LminusR_mean": env_mean,
        })

        all_results.append(AnalysisResult(
            roi_name=roi_name,
            window_name=win.name,
            n_points=n,
            area=float(win.area),
            intensity=float(est["intensity"][0]) if est["intensity"].size else float("nan"),
            r_peak=r_peak,
            lmr_max=lmr_max,
            auc_pos=auc_pos,
            curve=curve,
        ))

    return all_results


def results_to_summary_df(results: Sequence[AnalysisResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "roi": r.roi_name,
            "window": r.window_name,
            "n_points": r.n_points,
            "area": r.area,
            "intensity": r.intensity,
            "r_peak": r.r_peak,
            "LminusR_max": r.lmr_max,
            "AUC_pos_LminusR": r.auc_pos,
        })
    return pd.DataFrame(rows)
