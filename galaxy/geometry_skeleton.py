"""geometry_skeleton.py

General membrane geometry backend based on skeletonization and shortest-path
geodesic distances on a graph.

This backend is designed to work on "exotic" cell morphologies: elongated,
concave, partially open, or branched membrane bands.

Inputs
------
- membrane_geom: shapely Polygon/MultiPolygon representing the membrane band.
- zc_geom: shapely Polygon/MultiPolygon representing the contact zone.

Outputs
-------
A MembraneModel (see geometry_base.py) containing:
- skeleton graph (networkx)
- per-node distance-to-ZC (multi-source Dijkstra)
- thickness estimate (area / skeleton_length) unless manual.

License: MIT
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import networkx as nx
from scipy.spatial import cKDTree

from skimage.draw import polygon as sk_polygon
from skimage.morphology import skeletonize

from .geometry_base import (
    BandParams,
    GeometryParams,
    MembraneModel,
    clean_geometry,
    ensure_shapely,
    geometry_area,
    geometry_bounds,
    iter_polygons,
)

try:
    import shapely.geometry as shgeom
except Exception:  # pragma: no cover
    shgeom = None


def _poly_to_mask(membrane_geom, pixel_size: float, pad: int = 5) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Rasterize a Polygon/MultiPolygon to a binary mask.

    Returns
    -------
    mask : (H,W) bool
    origin : (xmin, ymin) in world coords

    Notes
    -----
    - pixel_size defines world units per pixel.
    - pad adds a margin in pixels.
    """
    ensure_shapely()

    minx, miny, maxx, maxy = geometry_bounds(membrane_geom)
    if maxx <= minx or maxy <= miny:
        raise ValueError("Invalid geometry bounds")

    ps = float(pixel_size)
    if ps <= 0:
        raise ValueError("pixel_size must be > 0")

    # Expand bounds by pad pixels.
    xmin = minx - pad * ps
    ymin = miny - pad * ps
    xmax = maxx + pad * ps
    ymax = maxy + pad * ps

    W = int(np.ceil((xmax - xmin) / ps)) + 1
    H = int(np.ceil((ymax - ymin) / ps)) + 1

    mask = np.zeros((H, W), dtype=bool)

    def world_to_rc(xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = (xy[:, 0] - xmin) / ps
        y = (xy[:, 1] - ymin) / ps
        rr = np.clip(np.round(y).astype(int), 0, H - 1)
        cc = np.clip(np.round(x).astype(int), 0, W - 1)
        return rr, cc

    for poly in iter_polygons(membrane_geom):
        ext = np.asarray(poly.exterior.coords, dtype=float)
        rr, cc = world_to_rc(ext)
        rri, cci = sk_polygon(rr, cc, shape=mask.shape)
        mask[rri, cci] = True

        # Holes
        for hole in poly.interiors:
            h = np.asarray(hole.coords, dtype=float)
            rrh, cch = world_to_rc(h)
            rrih, ccih = sk_polygon(rrh, cch, shape=mask.shape)
            mask[rrih, ccih] = False

    return mask, (xmin, ymin)


def _skeleton_to_graph(skel: np.ndarray, pixel_size: float, origin_xy: Tuple[float, float]) -> Tuple[nx.Graph, np.ndarray]:
    """Convert skeleton pixel mask to a weighted undirected graph.

    Nodes are integer ids. node_xy is (N,2) in world coords.
    """
    ps = float(pixel_size)
    xmin, ymin = map(float, origin_xy)

    # Collect skeleton pixels
    coords_rc = np.argwhere(skel)
    if coords_rc.size == 0:
        return nx.Graph(), np.zeros((0, 2), dtype=float)

    # Map (r,c) -> node id
    rc_to_id: Dict[Tuple[int, int], int] = {}
    node_xy = np.zeros((coords_rc.shape[0], 2), dtype=float)
    for idx, (r, c) in enumerate(coords_rc):
        rc_to_id[(int(r), int(c))] = idx
        x = xmin + float(c) * ps
        y = ymin + float(r) * ps
        node_xy[idx] = (x, y)

    G = nx.Graph()
    G.add_nodes_from(range(coords_rc.shape[0]))

    # 8-connectivity
    nbrs = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    for (r, c), u in rc_to_id.items():
        for dr, dc in nbrs:
            rr = r + dr
            cc = c + dc
            v = rc_to_id.get((rr, cc))
            if v is None:
                continue
            if v <= u:
                continue  # add each edge once
            w = ps * float(np.hypot(dr, dc))
            G.add_edge(u, v, weight=w)

    return G, node_xy


def _prune_short_spurs(G: nx.Graph, node_xy: np.ndarray, max_len: float) -> Tuple[nx.Graph, np.ndarray]:
    """Remove degree-1 spur chains shorter than max_len.

    This is a conservative, topology-preserving cleaning step.
    """
    if G.number_of_nodes() == 0:
        return G, node_xy

    max_len = float(max_len)
    if max_len <= 0:
        return G, node_xy

    # Precompute weights lookup
    def edge_w(a: int, b: int) -> float:
        return float(G.edges[a, b].get("weight", np.linalg.norm(node_xy[a] - node_xy[b])))

    to_remove = set()

    # We iterate because removing spurs can create new endpoints.
    changed = True
    while changed:
        changed = False
        endpoints = [n for n, deg in G.degree() if deg == 1 and n not in to_remove]
        for ep in endpoints:
            if ep in to_remove:
                continue

            # Walk until branchpoint (deg != 2) or back to endpoint.
            path = [ep]
            length = 0.0
            prev = None
            cur = ep
            while True:
                nbrs = [x for x in G.neighbors(cur) if x not in to_remove]
                if prev is not None:
                    nbrs = [x for x in nbrs if x != prev]
                if len(nbrs) == 0:
                    break
                nxt = nbrs[0]
                length += edge_w(cur, nxt)
                path.append(nxt)
                prev, cur = cur, nxt
                deg = G.degree(cur)
                if deg != 2:
                    break
                if length > max_len:
                    break

            if length <= max_len:
                # Remove all nodes in the spur path except the terminal node if it is a branchpoint.
                # If the last node is a branchpoint (deg>2), keep it.
                last = path[-1]
                if G.degree(last) > 2:
                    rm = path[:-1]
                else:
                    rm = path
                if rm:
                    to_remove.update(rm)
                    changed = True

        if changed and to_remove:
            G.remove_nodes_from(list(to_remove))
            # Relabel nodes to keep ids compact and update node_xy accordingly
            mapping = {old: new for new, old in enumerate(G.nodes())}
            G = nx.relabel_nodes(G, mapping, copy=True)
            node_xy = node_xy[list(mapping.keys())]
            to_remove.clear()

    return G, node_xy


def build_model(membrane_geom, zc_geom, params: GeometryParams) -> MembraneModel:
    ensure_shapely()

    mem = clean_geometry(membrane_geom)
    zc = clean_geometry(zc_geom) if zc_geom is not None else shgeom.GeometryCollection([])

    if mem.is_empty:
        raise ValueError("Membrane geometry is empty")

    # Rasterize + skeletonize
    mask, origin = _poly_to_mask(mem, pixel_size=params.pixel_size)
    if mask.sum() == 0:
        raise RuntimeError("Rasterization produced an empty mask. Try smaller pixel_size or verify ROI.")

    skel = skeletonize(mask)
    G, node_xy = _skeleton_to_graph(skel, pixel_size=params.pixel_size, origin_xy=origin)

    if params.prune_spurs and G.number_of_nodes() > 0:
        G, node_xy = _prune_short_spurs(G, node_xy, max_len=params.prune_spur_length)

    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        raise RuntimeError("Skeleton graph is empty. ROI may be too thin or pixel_size too large.")

    # Keep the largest connected component (membrane should be one component).
    components = list(nx.connected_components(G))
    if len(components) > 1:
        largest = max(components, key=len)
        G = G.subgraph(largest).copy()
        # remap node ids
        mapping = {old: new for new, old in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping, copy=True)
        node_xy = node_xy[list(mapping.keys())]

    # Skeleton length
    sk_len = float(sum(float(d.get("weight", 0.0)) for _, _, d in G.edges(data=True)))

    # Thickness
    mem_area = geometry_area(mem)
    if params.thickness_mode.lower().strip() == "manual":
        thickness = float(params.thickness_manual)
    else:
        thickness = float(mem_area / sk_len) if sk_len > 0 else float("nan")

    # Determine skeleton nodes belonging to ZC: buffer by thickness/2 to be tolerant.
    zc_clip = zc
    try:
        zc_clip = zc_clip.intersection(mem)
    except Exception:
        pass

    zc_nodes: List[int] = []
    if zc_clip is not None and not zc_clip.is_empty:
        try:
            region = zc_clip.buffer(max(params.pixel_size, thickness) / 2.0)
        except Exception:
            region = zc_clip

        # vectorized contains using prepared geometry
        try:
            from shapely.prepared import prep

            reg_p = prep(region)
            for i, (x, y) in enumerate(node_xy):
                if reg_p.contains(shgeom.Point(float(x), float(y))):
                    zc_nodes.append(i)
        except Exception:
            # fallback: distance to region
            for i, (x, y) in enumerate(node_xy):
                if region.distance(shgeom.Point(float(x), float(y))) <= params.pixel_size * 1.5:
                    zc_nodes.append(i)

    # Fallback: pick closest skeleton nodes to ZC boundary if none found
    if not zc_nodes:
        # Use centroids of ZC polygons
        zc_pts = []
        for poly in iter_polygons(zc_clip) if zc_clip is not None else []:
            c = poly.centroid
            zc_pts.append((float(c.x), float(c.y)))
        if not zc_pts:
            c = zc.centroid
            zc_pts.append((float(c.x), float(c.y)))
        tree = cKDTree(node_xy)
        _, idxs = tree.query(np.asarray(zc_pts, dtype=float), k=min(10, len(node_xy)))
        idxs = np.atleast_1d(idxs).ravel().tolist()
        zc_nodes = list(sorted(set(int(i) for i in idxs)))

    if not zc_nodes:
        raise RuntimeError("Could not map ZC onto the membrane skeleton. Ensure ZC overlaps membrane band.")

    # Multi-source Dijkstra distances from ZC nodes.
    dist_map = nx.multi_source_dijkstra_path_length(G, sources=zc_nodes, weight="weight")
    dist = np.full((node_xy.shape[0],), np.inf, dtype=float)
    for nid, d in dist_map.items():
        dist[int(nid)] = float(d)

    model = MembraneModel(
        membrane_geom=mem,
        zc_geom=zc_clip,
        graph=G,
        node_xy=node_xy,
        node_dist_to_zc=dist,
        skeleton_length=sk_len,
        thickness=thickness,
        is_ring=False,
        ring_perimeter=None,
        ring_s=None,
    )

    return model

# --- Skeleton preview helpers (Napari QC) ------------------------------------

from dataclasses import dataclass

@dataclass
class SkeletonPreview:
    """Convenience container for skeleton preview layers."""
    nodes_xy: np.ndarray              # (N,2) x,y in world coords (raw graph nodes)
    edges_xy: np.ndarray              # (E,2,2) each edge as [[x1,y1],[x2,y2]]
    median_xy: np.ndarray             # (M,2) x,y resampled points along edges
    overlaps_xy: np.ndarray           # (K,2) x,y points flagged as too close
    nn_median: float
    nn_iqr: Tuple[float, float]
    nn_min: float
    overlap_rate: float


def _resample_segment(p0: np.ndarray, p1: np.ndarray, step: float) -> np.ndarray:
    """Uniformly resample a single segment [p0->p1] including p0, excluding p1."""
    v = (p1 - p0).astype(float)
    L = float(np.hypot(v[0], v[1]))
    if L <= 0 or step <= 0:
        return p0[None, :]
    n = max(1, int(np.floor(L / step)))
    ts = (np.arange(n + 1, dtype=float) * (step / L))
    ts = ts[ts < 1.0]  # exclude p1 to reduce duplicates at junctions
    return p0[None, :] + ts[:, None] * v[None, :]


def make_skeleton_preview(
    model: MembraneModel,
    *,
    step: Optional[float] = None,
    overlap_thresh: Optional[float] = None,
    dedup_epsilon: Optional[float] = None,
) -> SkeletonPreview:
    """
    Build a QC preview of the geodesic skeleton.

    Parameters
    ----------
    model : MembraneModel
        Output of geometry_skeleton.build_model().
    step : float, optional
        Resampling step for the "median skeleton". Defaults to model raster pixel_size
        (estimated from graph edge weights / node spacing if not available).
    overlap_thresh : float, optional
        Flag points with nearest-neighbor distance < overlap_thresh.
        Defaults to 0.75 * step.
    dedup_epsilon : float, optional
        Spatial tolerance for de-duplicating resampled points (default 0.1 * step).

    Returns
    -------
    SkeletonPreview
    """
    G = model.graph
    nodes_xy = np.asarray(model.node_xy, dtype=float)
    if G is None or G.number_of_nodes() == 0:
        raise ValueError("Model has no skeleton graph to preview.")

    # Infer a sane default step.
    # If edge weights exist, use median edge length as a proxy for pixel_size sampling.
    if step is None:
        w = []
        for _, _, d in G.edges(data=True):
            if "weight" in d:
                w.append(float(d["weight"]))
        if len(w) > 0:
            step = float(np.median(w))
        else:
            # fallback: use NN spacing
            if nodes_xy.shape[0] >= 2:
                tree = cKDTree(nodes_xy)
                dnn, _ = tree.query(nodes_xy, k=2)
                step = float(np.median(dnn[:, 1]))
            else:
                step = 1.0

    step = float(step)
    if step <= 0:
        raise ValueError("step must be > 0")

    if overlap_thresh is None:
        overlap_thresh = 0.6 * step
    overlap_thresh = float(overlap_thresh)

    if dedup_epsilon is None:
        dedup_epsilon = 0.10 * step
    dedup_epsilon = float(dedup_epsilon)

    # Edges as segments (E,2,2) in xy
    edges = []
    for u, v, d in G.edges(data=True):
        p0 = nodes_xy[int(u)]
        p1 = nodes_xy[int(v)]
        edges.append([p0, p1])
    edges_xy = np.asarray(edges, dtype=float) if edges else np.zeros((0, 2, 2), dtype=float)

    # Resample along edges to get a "median" skeleton point cloud.
    pts = []
    for seg in edges_xy:
        p0 = seg[0]
        p1 = seg[1]
        pts.append(_resample_segment(p0, p1, step=step))
    median_xy = np.vstack(pts) if pts else np.zeros((0, 2), dtype=float)

    # Deduplicate median points (grid-hash with epsilon).
    if median_xy.shape[0] > 0:
        q = np.round(median_xy / dedup_epsilon).astype(np.int64)
        _, keep = np.unique(q, axis=0, return_index=True)
        median_xy = median_xy[np.sort(keep)]

    # NN stats + overlap detection on the *median* skeleton (more relevant to spacing QC).
    overlaps_xy = np.zeros((0, 2), dtype=float)
    nn_median = float("nan")
    nn_iqr = (float("nan"), float("nan"))
    nn_min = float("nan")
    overlap_rate = float("nan")

    if median_xy.shape[0] >= 2:
        tree = cKDTree(median_xy)
        dnn, idx = tree.query(median_xy, k=2)  # k=1 is self, k=2 gives NN
        nn = dnn[:, 1].astype(float)

        nn_median = float(np.median(nn))
        q25 = float(np.percentile(nn, 25))
        q75 = float(np.percentile(nn, 75))
        nn_iqr = (q25, q75)
        nn_min = float(np.min(nn))

        bad = nn < overlap_thresh
        overlaps_xy = median_xy[bad]
        overlap_rate = float(np.mean(bad)) if nn.size else 0.0
    else:
        overlap_rate = 0.0

    return SkeletonPreview(
        nodes_xy=nodes_xy,
        edges_xy=edges_xy,
        median_xy=median_xy,
        overlaps_xy=overlaps_xy,
        nn_median=nn_median,
        nn_iqr=nn_iqr,
        nn_min=nn_min,
        overlap_rate=overlap_rate,
    )
