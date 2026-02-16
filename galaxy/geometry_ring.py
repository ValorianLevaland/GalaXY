"""geometry_ring.py

"Ring" membrane geometry backend.

This backend is optimized for the common case where the membrane ROI is a
single closed loop (topologically a cycle): e.g., a roughly circular cell
outline or a clean closed membrane band.

Implementation strategy
-----------------------
We reuse the robust skeletonization/graph construction from geometry_skeleton,
then enforce a ring topology (connected, all degrees==2 after optional spur
pruning). We additionally compute a periodic arc-length coordinate `s` for each
node, enabling convenient perimeter-based plots.

The distance-to-ZC used for banding is still the shortest path distance on the
cycle, which is equivalent to the periodic geodesic distance.

License: MIT
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import networkx as nx

from .geometry_base import GeometryParams, MembraneModel
from .geometry_skeleton import build_model as _build_skeleton_model


def _is_single_cycle(G: nx.Graph) -> bool:
    """Return True if graph is connected and every node has degree 2."""
    if G.number_of_nodes() == 0:
        return False
    if not nx.is_connected(G):
        return False
    degs = [deg for _, deg in G.degree()]
    return all(d == 2 for d in degs)


def _order_cycle_nodes(G: nx.Graph) -> List[int]:
    """Return nodes in traversal order around the cycle."""
    if G.number_of_nodes() == 0:
        return []

    start = next(iter(G.nodes()))
    nbrs = list(G.neighbors(start))
    if len(nbrs) != 2:
        raise ValueError("Graph is not a simple cycle")

    order: List[int] = [start]
    prev = start
    curr = nbrs[0]

    while curr != start:
        order.append(curr)
        nbrs = list(G.neighbors(curr))
        if len(nbrs) != 2:
            raise ValueError("Graph is not a simple cycle")
        nxt = nbrs[0] if nbrs[0] != prev else nbrs[1]
        prev, curr = curr, nxt

        if len(order) > G.number_of_nodes() + 2:
            raise RuntimeError("Cycle traversal did not terminate")

    return order


def _compute_ring_s(G: nx.Graph, order: List[int]) -> Tuple[np.ndarray, float]:
    """Compute per-node arc-length coordinate s and perimeter."""
    if not order:
        return np.zeros((0,), dtype=float), 0.0

    s = np.zeros((G.number_of_nodes(),), dtype=float)
    cum = 0.0
    for a, b in zip(order[:-1], order[1:]):
        w = float(G.edges[a, b].get("weight", 0.0))
        cum += w
        s[b] = cum
    # closing edge
    w_close = float(G.edges[order[-1], order[0]].get("weight", 0.0))
    perimeter = cum + w_close
    # Wrap to [0,P)
    s = np.mod(s, perimeter) if perimeter > 0 else s
    return s, perimeter


def build_model(membrane_geom, zc_geom, params: GeometryParams) -> MembraneModel:
    """Build a MembraneModel under the assumption of a single closed loop."""
    model = _build_skeleton_model(membrane_geom, zc_geom, params)
    G = model.graph

    if not _is_single_cycle(G):
        # Provide a more helpful message for the GUI.
        raise RuntimeError(
            "Ring backend requires a single-cycle skeleton (no branches). "
            "If your cell is elongated/branched, use the Skeleton backend or Auto."
        )

    order = _order_cycle_nodes(G)
    ring_s, P = _compute_ring_s(G, order)

    model.is_ring = True
    model.ring_s = ring_s
    model.ring_perimeter = float(P)

    return model
