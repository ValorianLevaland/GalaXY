"""galaxy.profiles

Defines the set of supported topology *profiles* (aka presets).

A profile is a thin configuration layer over the same core primitives:
- Domain polygon (optionally with holes)
- Reference field (centroid, boundary, seed, geodesic)
- Subdivision strategy (shells/bands/grid)

The GUI uses this module to:
- populate the profile dropdown
- decide which ROI layers are required
- decide which parameter groups should be visible

License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class TopologyProfile:
    """A named analysis preset.

    Notes
    -----
    - `id` is used internally and in run_config.json.
    - `requires_seeds`: seeds are mandatory (e.g., ZC / patch centered).
    - `requires_holes`: holes are mandatory (e.g., perinuclear needs a nucleus).
    - `uses_geodesic`: uses a skeleton/graph model and geodesic distance.
    - `default_axis`: semantic name for the distance axis.
    - `default_tiling`: string identifying the default tiling mode.
    """

    id: str
    name: str
    description: str

    requires_seeds: bool = False
    requires_holes: bool = False
    uses_geodesic: bool = False

    # Axis names are for labeling plots/outputs.
    default_axis: str = "distance"

    # Tiling mode: geodesic_bands | radial_shells | boundary_shells | outer_shells | perinuclear_shells | seed_bands | grid | none
    default_tiling: str = "none"

    # Whether the profile should be hidden unless Expert mode is enabled.
    expert_only: bool = False


# ---- Profile registry ----

PROFILES: List[TopologyProfile] = [
    TopologyProfile(
        id="membrane",
        name="Membrane (Ring/Branched) — geodesic from seeds",
        description=(
            "Thin membrane band (outer minus inner). Distance axis is geodesic distance from seed ROI(s) "
            "(e.g., contact zone). Bands can be equal-length or fixed-distance."
        ),
        requires_seeds=True,
        requires_holes=True,
        uses_geodesic=True,
        default_axis="d_geodesic",
        default_tiling="geodesic_bands",
        expert_only=False,
    ),
    TopologyProfile(
        id="cytoplasm",
        name="Cytoplasm (Blob) — radial or boundary shells",
        description=(
            "2D area domain (cell polygon minus nucleus holes). Typical use: cytoplasmic clusters. "
            "Supports centroid-radial shells and boundary-distance shells."
        ),
        requires_seeds=False,
        requires_holes=False,
        uses_geodesic=False,
        default_axis="r_centroid",
        default_tiling="radial_shells",
        expert_only=False,
    ),
    TopologyProfile(
        id="perinuclear",
        name="Perinuclear — shells from nucleus boundary",
        description=(
            "Cytoplasm domain (cell minus nucleus). Distance axis is distance to nucleus boundary; "
            "windows are shells around nucleus."
        ),
        requires_seeds=False,
        requires_holes=True,
        uses_geodesic=False,
        default_axis="d_nucleus",
        default_tiling="perinuclear_shells",
        expert_only=False,
    ),
    TopologyProfile(
        id="cortex",
        name="Cortex / Peripheral shell — shells from outer boundary",
        description=(
            "Peripheral enrichment analysis. Distance axis is distance to the *outer* cell boundary "
            "(membrane-in). Windows are shells based on that distance (ignores nucleus holes)."
        ),
        requires_seeds=False,
        requires_holes=False,
        uses_geodesic=False,
        default_axis="d_outer",
        default_tiling="outer_shells",
        expert_only=False,
    ),
    TopologyProfile(
        id="patch",
        name="Patch-centered — distance bands around seed ROI(s)",
        description=(
            "Generalization of synapse/contact/adhesion patches. Seed ROI(s) define the center; "
            "windows are Euclidean distance bands around seeds, intersected with the domain."
        ),
        requires_seeds=True,
        requires_holes=False,
        uses_geodesic=False,
        default_axis="d_seed",
        default_tiling="seed_bands",
        expert_only=False,
    ),

    # Expert-only profiles
    TopologyProfile(
        id="neurite",
        name="Neuron/Process (Graph) — geodesic from soma/seed",
        description=(
            "Branched process morphology (neurons/astrocytes) treated as a skeleton graph. "
            "Geodesic distance from a seed ROI (e.g., soma) is used for binning."
        ),
        requires_seeds=True,
        requires_holes=False,
        uses_geodesic=True,
        default_axis="d_geodesic",
        default_tiling="geodesic_bands",
        expert_only=True,
    ),
    TopologyProfile(
        id="grid",
        name="Grid (Heatmap windows) — descriptive",
        description=(
            "Domain subdivided into a grid of tiles. Useful for exploratory heterogeneity maps, "
            "but easy to overinterpret."
        ),
        requires_seeds=False,
        requires_holes=False,
        uses_geodesic=False,
        default_axis="tile",
        default_tiling="grid",
        expert_only=True,
    ),
]


def list_profiles(*, expert_mode: bool) -> List[TopologyProfile]:
    """Return profiles appropriate for the requested UI mode."""
    if expert_mode:
        return list(PROFILES)
    return [p for p in PROFILES if not p.expert_only]


def get_profile(profile_id: str) -> TopologyProfile:
    for p in PROFILES:
        if p.id == profile_id:
            return p
    raise KeyError(f"Unknown profile id: {profile_id}")


def profile_by_name(name: str) -> Optional[TopologyProfile]:
    for p in PROFILES:
        if p.name == name:
            return p
    return None


def profile_tooltip(profile: TopologyProfile) -> str:
    """Small helper for GUI tooltips."""
    req = []
    if profile.requires_seeds:
        req.append("Seeds")
    if profile.requires_holes:
        req.append("Holes")
    if profile.uses_geodesic:
        req.append("Geodesic")
    req_txt = ", ".join(req) if req else "None"
    return f"{profile.description}\n\nRequirements: {req_txt}".strip()
