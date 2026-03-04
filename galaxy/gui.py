"""galaxy.gui

GalaXY Napari + Qt GUI.

This is the user-facing application.

Run
---
    python run_galaxy.py

or (from the package root):
    python -m galaxy.gui

License: MIT
"""

from __future__ import annotations

import os
import datetime as _dt
from typing import List, Optional

import numpy as np
import pandas as pd

from qtpy import QtCore, QtWidgets

from .audit import QtLogHandler
from .profiles import list_profiles, profile_tooltip, profile_by_name, get_profile
from .tiling import WindowBinningParams
from .analysis_core import DBSCANParams, HierarchicalDBSCANParams
from .ripley_backend import auto_detect_xy_columns, RipleyParams, CSRParams, vertices_to_shapely_polygon
from .geometry_base import GeometryParams, BandParams
from .worker import GalaXYWorker

try:
    import shapely.geometry as shgeom
except Exception:  # pragma: no cover
    shgeom = None


# ----------------------------
# Napari/shapely conversions
# ----------------------------

def _ensure_shapely() -> None:
    if shgeom is None:
        raise ImportError("shapely is required")


def _napari_shape_to_polygon(verts_yx: np.ndarray):
    """Convert Napari (y,x) vertices to shapely Polygon."""
    _ensure_shapely()
    v = np.asarray(verts_yx, dtype=float)
    if v.ndim != 2 or v.shape[1] != 2:
        raise ValueError("Expected vertices as (N,2) array in (y,x)")
    verts_xy = np.column_stack([v[:, 1], v[:, 0]])
    return vertices_to_shapely_polygon(verts_xy)


def _geom_to_napari_rings(geom) -> List[np.ndarray]:
    """Convert shapely Polygon/MultiPolygon to a list of rings as Napari polygons (y,x)."""
    _ensure_shapely()

    rings: List[np.ndarray] = []
    if geom is None or geom.is_empty:
        return rings

    def _add_ring(coords):
        c = np.asarray(coords, dtype=float)
        if c.shape[0] > 1 and np.allclose(c[0], c[-1]):
            c = c[:-1]
        if c.shape[0] < 3:
            return
        rings.append(np.column_stack([c[:, 1], c[:, 0]]))  # (y,x)

    if isinstance(geom, shgeom.Polygon):
        _add_ring(geom.exterior.coords)
        for hole in geom.interiors:
            _add_ring(hole.coords)
    elif isinstance(geom, shgeom.MultiPolygon):
        for g in geom.geoms:
            rings.extend(_geom_to_napari_rings(g))
    else:
        try:
            for g in geom.geoms:
                rings.extend(_geom_to_napari_rings(g))
        except Exception:
            pass

    return rings


def _sanitize_filename(s: str) -> str:
    keep = []
    for ch in str(s):
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep).strip("_")
    return out[:200] if out else "item"


# ----------------------------
# Main dock widget
# ----------------------------


class GalaXYDock(QtWidgets.QWidget):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.logger = QtLogHandler()
        self.logger.sig.connect(self._append_log)

        self.df: Optional[pd.DataFrame] = None
        self.points_xy: Optional[np.ndarray] = None
        self.points_z: Optional[np.ndarray] = None
        self.channel_labels: Optional[np.ndarray] = None
        self.points_layer = None

        self.domain_layer = None
        self.holes_layer = None
        self.seeds_layer = None
        self.overlay_layer = None

        self._csv_path: str = ""
        self.worker_thread: Optional[QtCore.QThread] = None
        self.worker: Optional[GalaXYWorker] = None

        self._build_ui()
        self._refresh_profiles()

    # -----------------
    # UI
    # -----------------

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        self.tabs = QtWidgets.QTabWidget()

        # Napari dock widgets do not auto-scroll. We wrap each tab content
        # into a QScrollArea so long panels remain accessible.
        def _wrap_scroll_tab(w: QtWidgets.QWidget) -> QtWidgets.QScrollArea:
            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
            scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
            scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
            scroll.setWidget(w)
            return scroll

        layout.addWidget(self.tabs)

        # === Data tab ===
        tab_data = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(tab_data)

        self.btn_load = QtWidgets.QPushButton("Load localization CSV")
        self.btn_load.clicked.connect(self._on_load_csv)
        v.addWidget(self.btn_load)

        grid = QtWidgets.QGridLayout()
        v.addLayout(grid)
        self.cmb_x = QtWidgets.QComboBox()
        self.cmb_y = QtWidgets.QComboBox()
        self.cmb_z = QtWidgets.QComboBox()
        self.cmb_channel = QtWidgets.QComboBox()
        self.cmb_z.addItem("<None>")
        self.cmb_channel.addItem("<None>")
        grid.addWidget(QtWidgets.QLabel("X column:"), 0, 0)
        grid.addWidget(self.cmb_x, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Y column:"), 1, 0)
        grid.addWidget(self.cmb_y, 1, 1)
        grid.addWidget(QtWidgets.QLabel("Z column (optional):"), 2, 0)
        grid.addWidget(self.cmb_z, 2, 1)
        grid.addWidget(QtWidgets.QLabel("Channel column (optional):"), 3, 0)
        grid.addWidget(self.cmb_channel, 3, 1)

        self.btn_show = QtWidgets.QPushButton("Show points")
        self.btn_show.clicked.connect(self._on_show_points)
        v.addWidget(self.btn_show)

        self.lbl_datainfo = QtWidgets.QLabel("No data loaded.")
        self.lbl_datainfo.setWordWrap(True)
        v.addWidget(self.lbl_datainfo)

        self.grp_channels = QtWidgets.QGroupBox("Channels")
        vc = QtWidgets.QVBoxLayout(self.grp_channels)
        self.tbl_channels = QtWidgets.QTableWidget(0, 3)
        self.tbl_channels.setHorizontalHeaderLabels(["Use", "Channel", "Alias"])
        self.tbl_channels.horizontalHeader().setStretchLastSection(True)
        vc.addWidget(self.tbl_channels)
        self.grp_channels.setVisible(False)
        v.addWidget(self.grp_channels)

        self.tabs.addTab(_wrap_scroll_tab(tab_data), "Data")

        # === ROIs tab ===
        tab_roi = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(tab_roi)

        self.btn_domain_layer = QtWidgets.QPushButton("Create/Select Domain Shapes layer ('Domain')")
        self.btn_domain_layer.clicked.connect(self._on_create_domain_layer)
        v.addWidget(self.btn_domain_layer)

        self.btn_holes_layer = QtWidgets.QPushButton("Create/Select Holes Shapes layer ('Holes')")
        self.btn_holes_layer.clicked.connect(self._on_create_holes_layer)
        v.addWidget(self.btn_holes_layer)

        self.btn_seeds_layer = QtWidgets.QPushButton("Create/Select Seeds Shapes layer ('Seeds')")
        self.btn_seeds_layer.clicked.connect(self._on_create_seeds_layer)
        v.addWidget(self.btn_seeds_layer)

        self.btn_refresh_rois = QtWidgets.QPushButton("Refresh ROI lists")
        self.btn_refresh_rois.clicked.connect(self._refresh_roi_tables)
        v.addWidget(self.btn_refresh_rois)

        self.btn_save_rois = QtWidgets.QPushButton("Save ROIs…")
        self.btn_save_rois.clicked.connect(self._on_save_rois)
        v.addWidget(self.btn_save_rois)

        self.btn_load_rois = QtWidgets.QPushButton("Load ROIs…")
        self.btn_load_rois.clicked.connect(self._on_load_rois)
        v.addWidget(self.btn_load_rois)

        v.addWidget(QtWidgets.QLabel("Domain ROIs (outer boundaries):"))
        self.tbl_domains = QtWidgets.QTableWidget(0, 3)
        self.tbl_domains.setHorizontalHeaderLabels(["Use", "Name", "Vertices"])
        self.tbl_domains.horizontalHeader().setStretchLastSection(True)
        v.addWidget(self.tbl_domains)

        v.addWidget(QtWidgets.QLabel("Holes ROIs (nucleus/exclusions):"))
        self.tbl_holes = QtWidgets.QTableWidget(0, 4)
        self.tbl_holes.setHorizontalHeaderLabels(["Use", "Name", "Type", "Vertices"])
        self.tbl_holes.horizontalHeader().setStretchLastSection(True)
        v.addWidget(self.tbl_holes)

        v.addWidget(QtWidgets.QLabel("Seeds ROIs (ZC/patch/soma):"))
        self.tbl_seeds = QtWidgets.QTableWidget(0, 3)
        self.tbl_seeds.setHorizontalHeaderLabels(["Use", "Name", "Vertices"])
        self.tbl_seeds.horizontalHeader().setStretchLastSection(True)
        v.addWidget(self.tbl_seeds)

        self.lbl_roi_hint = QtWidgets.QLabel("")
        self.lbl_roi_hint.setWordWrap(True)
        v.addWidget(self.lbl_roi_hint)

        self.tabs.addTab(_wrap_scroll_tab(tab_roi), "ROIs")

        # === Settings tab ===
        tab_set = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(tab_set)

        self.chk_expert = QtWidgets.QCheckBox("Expert mode")
        self.chk_expert.setChecked(False)
        self.chk_expert.stateChanged.connect(self._refresh_profiles)
        v.addWidget(self.chk_expert)

        v.addWidget(QtWidgets.QLabel("Topology profile:"))
        self.cmb_profile = QtWidgets.QComboBox()
        self.cmb_profile.currentIndexChanged.connect(self._on_profile_changed)
        v.addWidget(self.cmb_profile)

        self.lbl_profile_desc = QtWidgets.QLabel("")
        self.lbl_profile_desc.setWordWrap(True)
        v.addWidget(self.lbl_profile_desc)

        # Tiling
        self.grp_tiling = QtWidgets.QGroupBox("Subdivision (tiling)")
        gt = QtWidgets.QGridLayout(self.grp_tiling)

        gt.addWidget(QtWidgets.QLabel("Tiling mode:"), 0, 0)
        self.cmb_tiling = QtWidgets.QComboBox()
        gt.addWidget(self.cmb_tiling, 0, 1)

        self.spn_bins = QtWidgets.QSpinBox(); self.spn_bins.setRange(1, 10_000); self.spn_bins.setValue(6)
        gt.addWidget(QtWidgets.QLabel("# bins/shells:"), 1, 0)
        gt.addWidget(self.spn_bins, 1, 1)

        self.spn_seed_bw = QtWidgets.QDoubleSpinBox(); self.spn_seed_bw.setRange(1e-9, 1e12); self.spn_seed_bw.setValue(500.0)
        self.spn_seed_bw.setSuffix(" units")
        gt.addWidget(QtWidgets.QLabel("Seed band width:"), 2, 0)
        gt.addWidget(self.spn_seed_bw, 2, 1)

        self.chk_include_seed = QtWidgets.QCheckBox("Include seed as bin 0")
        self.chk_include_seed.setChecked(True)
        gt.addWidget(self.chk_include_seed, 3, 0, 1, 2)

        self.spn_grid_rows = QtWidgets.QSpinBox(); self.spn_grid_rows.setRange(1, 999); self.spn_grid_rows.setValue(3)
        self.spn_grid_cols = QtWidgets.QSpinBox(); self.spn_grid_cols.setRange(1, 999); self.spn_grid_cols.setValue(3)
        gt.addWidget(QtWidgets.QLabel("Grid rows:"), 4, 0)
        gt.addWidget(self.spn_grid_rows, 4, 1)
        gt.addWidget(QtWidgets.QLabel("Grid cols:"), 5, 0)
        gt.addWidget(self.spn_grid_cols, 5, 1)

        v.addWidget(self.grp_tiling)

        # Geodesic geometry params
        self.grp_geo = QtWidgets.QGroupBox("Geodesic model (skeleton graph)")
        gg = QtWidgets.QGridLayout(self.grp_geo)

        self.cmb_backend = QtWidgets.QComboBox(); self.cmb_backend.addItems(["Auto", "Ring", "Skeleton"])
        gg.addWidget(QtWidgets.QLabel("Geometry backend:"), 0, 0)
        gg.addWidget(self.cmb_backend, 0, 1)

        self.spn_px = QtWidgets.QDoubleSpinBox(); self.spn_px.setRange(1e-6, 1e12); self.spn_px.setValue(50.0)
        self.spn_px.setSuffix(" units/pixel")
        gg.addWidget(QtWidgets.QLabel("Raster pixel size:"), 1, 0)
        gg.addWidget(self.spn_px, 1, 1)

        self.chk_prune = QtWidgets.QCheckBox("Prune short spurs")
        self.chk_prune.setChecked(True)
        gg.addWidget(self.chk_prune, 2, 0, 1, 2)

        self.spn_prune_len = QtWidgets.QDoubleSpinBox(); self.spn_prune_len.setRange(0.0, 1e12); self.spn_prune_len.setValue(200.0)
        self.spn_prune_len.setSuffix(" units")
        gg.addWidget(QtWidgets.QLabel("Spur length threshold:"), 3, 0)
        gg.addWidget(self.spn_prune_len, 3, 1)

        self.cmb_thick_mode = QtWidgets.QComboBox(); self.cmb_thick_mode.addItems(["auto", "manual"])
        gg.addWidget(QtWidgets.QLabel("Thickness mode:"), 4, 0)
        gg.addWidget(self.cmb_thick_mode, 4, 1)

        self.spn_thick = QtWidgets.QDoubleSpinBox(); self.spn_thick.setRange(0.0, 1e12); self.spn_thick.setValue(200.0)
        self.spn_thick.setSuffix(" units")
        gg.addWidget(QtWidgets.QLabel("Thickness (manual):"), 5, 0)
        gg.addWidget(self.spn_thick, 5, 1)

        v.addWidget(self.grp_geo)

        # Geodesic band params
        self.grp_bands = QtWidgets.QGroupBox("Geodesic banding")
        gb = QtWidgets.QGridLayout(self.grp_bands)

        self.cmb_band_mode = QtWidgets.QComboBox(); self.cmb_band_mode.addItems(["equal_length", "fixed_distance"])
        gb.addWidget(QtWidgets.QLabel("Banding strategy:"), 0, 0)
        gb.addWidget(self.cmb_band_mode, 0, 1)

        self.spn_band_len = QtWidgets.QDoubleSpinBox(); self.spn_band_len.setRange(1e-9, 1e12); self.spn_band_len.setValue(5000.0)
        self.spn_band_len.setSuffix(" units")
        gb.addWidget(QtWidgets.QLabel("Target skeleton length / band:"), 1, 0)
        gb.addWidget(self.spn_band_len, 1, 1)

        self.spn_band_w = QtWidgets.QDoubleSpinBox(); self.spn_band_w.setRange(1e-9, 1e12); self.spn_band_w.setValue(500.0)
        self.spn_band_w.setSuffix(" units")
        gb.addWidget(QtWidgets.QLabel("Fixed distance band width:"), 2, 0)
        gb.addWidget(self.spn_band_w, 2, 1)

        self.chk_subtract_seed = QtWidgets.QCheckBox("Subtract seed region (exclude ZC/patch) from bins")
        self.chk_subtract_seed.setChecked(True)
        gb.addWidget(self.chk_subtract_seed, 3, 0, 1, 2)

        v.addWidget(self.grp_bands)

        self.btn_preview_regions = QtWidgets.QPushButton("Build regions overlay (preview)")
        self.btn_preview_regions.clicked.connect(self._on_preview_regions)
        v.addWidget(self.btn_preview_regions)

        self.btn_preview_skeleton = QtWidgets.QPushButton("Preview skeleton (nodes/edges)")
        self.btn_preview_skeleton.clicked.connect(self._on_preview_skeleton)
        v.addWidget(self.btn_preview_skeleton)

        self.tabs.addTab(_wrap_scroll_tab(tab_set), "Settings")

        # === Run tab ===
        tab_run = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(tab_run)

        # Output
        self.grp_out = QtWidgets.QGroupBox("Output")
        go = QtWidgets.QGridLayout(self.grp_out)
        self.txt_outdir = QtWidgets.QLineEdit("")
        self.btn_outdir = QtWidgets.QPushButton("Browse")
        self.btn_outdir.clicked.connect(self._on_choose_outdir)
        go.addWidget(QtWidgets.QLabel("Output base directory:"), 0, 0)
        go.addWidget(self.txt_outdir, 0, 1)
        go.addWidget(self.btn_outdir, 0, 2)

        self.chk_fig_overview = QtWidgets.QCheckBox("Save overview figures")
        self.chk_fig_overview.setChecked(True)
        go.addWidget(self.chk_fig_overview, 1, 0, 1, 3)

        self.chk_fig_region = QtWidgets.QCheckBox("Save per-region DBSCAN figures")
        self.chk_fig_region.setChecked(True)
        go.addWidget(self.chk_fig_region, 2, 0, 1, 3)

        self.chk_fig_cluster_hists = QtWidgets.QCheckBox("Save cluster histogram figures")
        self.chk_fig_cluster_hists.setChecked(True)
        go.addWidget(self.chk_fig_cluster_hists, 3, 0, 1, 3)

        self.chk_fig_cluster_hists_region = QtWidgets.QCheckBox("Save per-region cluster histogram figures")
        self.chk_fig_cluster_hists_region.setChecked(True)
        go.addWidget(self.chk_fig_cluster_hists_region, 4, 0, 1, 3)

        self.chk_fig_ripley_region = QtWidgets.QCheckBox("Save per-region Ripley figures")
        self.chk_fig_ripley_region.setChecked(False)
        go.addWidget(self.chk_fig_ripley_region, 5, 0, 1, 3)

        self.chk_ripley_per_cluster = QtWidgets.QCheckBox("Compute Ripley per DBSCAN cluster (local window)")
        self.chk_ripley_per_cluster.setChecked(True)
        go.addWidget(self.chk_ripley_per_cluster, 6, 0, 1, 3)

        self.chk_fig_ripley_cluster = QtWidgets.QCheckBox("Save per-cluster Ripley figures")
        self.chk_fig_ripley_cluster.setChecked(True)
        go.addWidget(self.chk_fig_ripley_cluster, 7, 0, 1, 3)

        lab = QtWidgets.QLabel("Cluster Ripley min points:")
        self.spin_cluster_ripley_minpts = QtWidgets.QSpinBox()
        self.spin_cluster_ripley_minpts.setRange(2, 999999)
        self.spin_cluster_ripley_minpts.setValue(6)
        hh = QtWidgets.QHBoxLayout()
        hh.addWidget(lab)
        hh.addWidget(self.spin_cluster_ripley_minpts)
        w = QtWidgets.QWidget()
        w.setLayout(hh)
        go.addWidget(w, 8, 0, 1, 3)

        self.chk_export_points = QtWidgets.QCheckBox("Export region-labeled points")
        self.chk_export_points.setChecked(False)
        go.addWidget(self.chk_export_points, 9, 0, 1, 3)

        self.spn_fig_maxpts = QtWidgets.QSpinBox(); self.spn_fig_maxpts.setRange(1000, 2_000_000); self.spn_fig_maxpts.setValue(200_000)
        go.addWidget(QtWidgets.QLabel("Max points in figures:"), 10, 0)
        go.addWidget(self.spn_fig_maxpts, 10, 1)

        v.addWidget(self.grp_out)

        # DBSCAN
        self.grp_db = QtWidgets.QGroupBox("DBSCAN")
        gd = QtWidgets.QGridLayout(self.grp_db)
        self.spn_eps = QtWidgets.QDoubleSpinBox(); self.spn_eps.setRange(1e-9, 1e12); self.spn_eps.setValue(30.0)
        self.spn_eps.setSuffix(" units")
        self.spn_min = QtWidgets.QSpinBox(); self.spn_min.setRange(1, 10_000); self.spn_min.setValue(5)
        gd.addWidget(QtWidgets.QLabel("eps:"), 0, 0)
        gd.addWidget(self.spn_eps, 0, 1)
        gd.addWidget(QtWidgets.QLabel("min_samples:"), 1, 0)
        gd.addWidget(self.spn_min, 1, 1)

        self.chk_use_z = QtWidgets.QCheckBox("Use Z for DBSCAN (2.5D)")
        self.chk_use_z.setChecked(False)
        gd.addWidget(self.chk_use_z, 2, 0, 1, 2)

        v.addWidget(self.grp_db)

        # Hierarchical
        self.grp_hier = QtWidgets.QGroupBox("Hierarchical clustering (clusters of clusters)")
        gh = QtWidgets.QGridLayout(self.grp_hier)
        self.chk_hier = QtWidgets.QCheckBox("Enable")
        self.chk_hier.setChecked(False)
        gh.addWidget(self.chk_hier, 0, 0, 1, 2)
        self.spn_eps2 = QtWidgets.QDoubleSpinBox(); self.spn_eps2.setRange(1e-9, 1e12); self.spn_eps2.setValue(200.0)
        self.spn_eps2.setSuffix(" units")
        self.spn_min2 = QtWidgets.QSpinBox(); self.spn_min2.setRange(1, 10_000); self.spn_min2.setValue(3)
        gh.addWidget(QtWidgets.QLabel("eps:"), 1, 0)
        gh.addWidget(self.spn_eps2, 1, 1)
        gh.addWidget(QtWidgets.QLabel("min_samples:"), 2, 0)
        gh.addWidget(self.spn_min2, 2, 1)
        v.addWidget(self.grp_hier)

        # Ripley
        self.grp_ripley = QtWidgets.QGroupBox("Ripley / Besag L(r)-r")
        gr = QtWidgets.QGridLayout(self.grp_ripley)
        self.chk_ripley = QtWidgets.QCheckBox("Enable")
        self.chk_ripley.setChecked(False)
        gr.addWidget(self.chk_ripley, 0, 0, 1, 2)

        self.spn_rmin = QtWidgets.QDoubleSpinBox(); self.spn_rmin.setRange(0.0, 1e12); self.spn_rmin.setValue(0.0)
        self.spn_rmax = QtWidgets.QDoubleSpinBox(); self.spn_rmax.setRange(1e-9, 1e12); self.spn_rmax.setValue(500.0)
        self.spn_dr = QtWidgets.QDoubleSpinBox(); self.spn_dr.setRange(1e-9, 1e12); self.spn_dr.setValue(5.0)
        self.cmb_edge = QtWidgets.QComboBox(); self.cmb_edge.addItems(["border", "none"])
        gr.addWidget(QtWidgets.QLabel("r_min:"), 1, 0)
        gr.addWidget(self.spn_rmin, 1, 1)
        gr.addWidget(QtWidgets.QLabel("r_max:"), 2, 0)
        gr.addWidget(self.spn_rmax, 2, 1)
        gr.addWidget(QtWidgets.QLabel("dr:"), 3, 0)
        gr.addWidget(self.spn_dr, 3, 1)
        gr.addWidget(QtWidgets.QLabel("edge correction:"), 4, 0)
        gr.addWidget(self.cmb_edge, 4, 1)

        self.chk_csr = QtWidgets.QCheckBox("CSR envelope")
        self.chk_csr.setChecked(False)
        gr.addWidget(self.chk_csr, 5, 0, 1, 2)

        self.spn_sims = QtWidgets.QSpinBox(); self.spn_sims.setRange(0, 10_000); self.spn_sims.setValue(199)
        self.spn_alpha = QtWidgets.QDoubleSpinBox(); self.spn_alpha.setRange(0.001, 0.999); self.spn_alpha.setValue(0.05)
        self.spn_seed = QtWidgets.QSpinBox(); self.spn_seed.setRange(-1, 1_000_000); self.spn_seed.setValue(0)
        gr.addWidget(QtWidgets.QLabel("# simulations:"), 6, 0)
        gr.addWidget(self.spn_sims, 6, 1)
        gr.addWidget(QtWidgets.QLabel("alpha:"), 7, 0)
        gr.addWidget(self.spn_alpha, 7, 1)
        gr.addWidget(QtWidgets.QLabel("seed (-1=none):"), 8, 0)
        gr.addWidget(self.spn_seed, 8, 1)

        self.chk_cross_ripley = QtWidgets.QCheckBox("Cross-Ripley between selected channels (expert)")
        self.chk_cross_ripley.setChecked(False)
        gr.addWidget(self.chk_cross_ripley, 9, 0, 1, 2)

        v.addWidget(self.grp_ripley)

        # Run controls
        self.btn_run = QtWidgets.QPushButton("Run analysis")
        self.btn_run.clicked.connect(self._on_run)
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self._on_cancel)
        self.btn_cancel.setEnabled(False)

        h = QtWidgets.QHBoxLayout()
        h.addWidget(self.btn_run)
        h.addWidget(self.btn_cancel)
        v.addLayout(h)

        self.progress = QtWidgets.QProgressBar(); self.progress.setRange(0, 100)
        v.addWidget(self.progress)

        self.lbl_runinfo = QtWidgets.QLabel("")
        self.lbl_runinfo.setWordWrap(True)
        v.addWidget(self.lbl_runinfo)

        self.txt_log = QtWidgets.QPlainTextEdit(); self.txt_log.setReadOnly(True)
        v.addWidget(self.txt_log)

        self.tabs.addTab(_wrap_scroll_tab(tab_run), "Run")

    # -----------------
    # Logging
    # -----------------

    def _append_log(self, line: str) -> None:
        self.txt_log.appendPlainText(line)

    # -----------------
    # Profile handling
    # -----------------

    def _refresh_profiles(self) -> None:
        expert = bool(self.chk_expert.isChecked())
        profiles = list_profiles(expert_mode=expert)

        # Expert-only widgets
        try:
            self.chk_cross_ripley.setVisible(expert)
        except Exception:
            pass

        current = self.cmb_profile.currentText() if self.cmb_profile.count() > 0 else None

        self.cmb_profile.blockSignals(True)
        self.cmb_profile.clear()
        for p in profiles:
            self.cmb_profile.addItem(p.name)
            self.cmb_profile.setItemData(self.cmb_profile.count() - 1, profile_tooltip(p), QtCore.Qt.ToolTipRole)
        self.cmb_profile.blockSignals(False)

        # restore selection if possible
        if current:
            idx = self.cmb_profile.findText(current)
            if idx >= 0:
                self.cmb_profile.setCurrentIndex(idx)
            else:
                self.cmb_profile.setCurrentIndex(0)
        else:
            self.cmb_profile.setCurrentIndex(0)

        self._on_profile_changed()

    def _on_profile_changed(self) -> None:
        prof = profile_by_name(self.cmb_profile.currentText())
        if prof is None:
            return

        self.lbl_profile_desc.setText(prof.description)

        # ROI hint
        req = []
        if prof.requires_holes:
            req.append("Holes (e.g., nucleus)")
        if prof.requires_seeds:
            req.append("Seeds (e.g., ZC / patch)")
        if prof.uses_geodesic:
            req.append("Geodesic skeleton model")
        req_txt = ", ".join(req) if req else "Domain only"
        self.lbl_roi_hint.setText(f"Requirements for this profile: {req_txt}.")

        # Populate tiling options
        self.cmb_tiling.blockSignals(True)
        self.cmb_tiling.clear()

        if prof.uses_geodesic:
            self.cmb_tiling.addItems(["geodesic_bands"])
        else:
            if prof.id == "cytoplasm":
                self.cmb_tiling.addItems(["radial_shells", "boundary_shells", "grid"])
            elif prof.id == "perinuclear":
                self.cmb_tiling.addItems(["perinuclear_shells"])
            elif prof.id == "cortex":
                self.cmb_tiling.addItems(["outer_shells", "boundary_shells", "grid"])
            elif prof.id == "patch":
                self.cmb_tiling.addItems(["seed_bands", "grid"])
            elif prof.id == "grid":
                self.cmb_tiling.addItems(["grid"])
            else:
                self.cmb_tiling.addItems([prof.default_tiling])

        self.cmb_tiling.blockSignals(False)

        # Enable/disable groups
        self.grp_geo.setVisible(bool(prof.uses_geodesic))
        self.grp_bands.setVisible(bool(prof.uses_geodesic))

        # Non-geodesic tiling params
        self.grp_tiling.setVisible(True)

        # Force sensible defaults
        self.spn_bins.setValue(6)

    # -----------------
    # Data IO
    # -----------------

    def _on_load_csv(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select localization CSV", "", "CSV (*.csv);;All (*.*)")
        if not path:
            return
        try:
            df = pd.read_csv(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Failed to load CSV", str(e))
            return

        self.df = df
        self._csv_path = path

        self.cmb_x.clear(); self.cmb_y.clear()
        self.cmb_z.blockSignals(True)
        self.cmb_channel.blockSignals(True)
        self.cmb_z.clear(); self.cmb_channel.clear()
        self.cmb_z.addItem("<None>")
        self.cmb_channel.addItem("<None>")
        self.cmb_z.blockSignals(False)
        self.cmb_channel.blockSignals(False)

        for c in df.columns:
            self.cmb_x.addItem(str(c))
            self.cmb_y.addItem(str(c))
            self.cmb_z.addItem(str(c))
            self.cmb_channel.addItem(str(c))

        # Hide channel selection UI until points are instantiated
        try:
            self.grp_channels.setVisible(False)
            self.tbl_channels.setRowCount(0)
        except Exception:
            pass

        try:
            x, y = auto_detect_xy_columns(df)
            self.cmb_x.setCurrentText(x)
            self.cmb_y.setCurrentText(y)
        except Exception:
            pass

        # Heuristic defaults for optional Z / channel columns
        try:
            cols_l = [str(c).strip().lower() for c in df.columns]
            # Z column
            z_candidates = []
            for raw, cl in zip(df.columns, cols_l):
                if cl in ("z", "z_nm", "z(nm)", "z [nm]", "z-coordinate", "z_coordinate"):
                    z_candidates.append(str(raw))
            if not z_candidates:
                for raw, cl in zip(df.columns, cols_l):
                    if cl.startswith("z") or " z" in cl or cl.endswith(" z"):
                        z_candidates.append(str(raw))
            if z_candidates:
                self.cmb_z.setCurrentText(z_candidates[0])

            # Channel column
            ch_candidates = []
            for raw, cl in zip(df.columns, cols_l):
                if cl in ("channel", "ch", "chan", "label", "protein", "species"):
                    ch_candidates.append(str(raw))
            if not ch_candidates:
                for raw, cl in zip(df.columns, cols_l):
                    if "channel" in cl or cl.startswith("ch") or cl.endswith("_ch"):
                        ch_candidates.append(str(raw))
            if ch_candidates:
                self.cmb_channel.setCurrentText(ch_candidates[0])
        except Exception:
            pass

        self.lbl_datainfo.setText(f"Loaded: {os.path.basename(path)}\nRows: {len(df):,}  Cols: {df.shape[1]}")
        self.logger.info(f"Loaded CSV: {path}")

    def _on_show_points(self):
        if self.df is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load a CSV first.")
            return

        x_col = self.cmb_x.currentText()
        y_col = self.cmb_y.currentText()
        z_col = self.cmb_z.currentText() if hasattr(self, "cmb_z") else "<None>"
        ch_col = self.cmb_channel.currentText() if hasattr(self, "cmb_channel") else "<None>"

        # Load coordinates
        try:
            x = self.df[x_col].to_numpy(dtype=float)
            y = self.df[y_col].to_numpy(dtype=float)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Invalid columns", str(e))
            return

        pts = np.column_stack([x, y])

        mask = np.all(np.isfinite(pts), axis=1)

        z_raw = None
        if z_col and z_col != "<None>":
            try:
                z_raw = self.df[z_col].to_numpy(dtype=float)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Invalid Z column", str(e))
                return
            mask &= np.isfinite(z_raw)

        ch_raw = None
        if ch_col and ch_col != "<None>":
            try:
                ch_raw = self.df[ch_col].fillna("<NA>").astype(str).to_numpy()
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Invalid Channel column", str(e))
                return

        n_drop = int(np.sum(~mask))
        if n_drop > 0:
            self.logger.warning(f"Dropping {n_drop:,} rows with non-finite coordinates (X/Y or Z).")

        pts = pts[mask]
        z = z_raw[mask].astype(float) if z_raw is not None else None
        ch = ch_raw[mask].astype(str) if ch_raw is not None else None

        self.points_xy = pts
        self.points_z = z
        self.channel_labels = ch

        info = f"Points: {pts.shape[0]:,} (x={x_col}, y={y_col})"
        if z is not None:
            info += f", z={z_col}"
        if ch is not None:
            info += f", channel={ch_col}"
        self.logger.info(info)

        # Channel table population
        if ch is not None:
            unique = sorted({str(v) for v in ch.tolist()})
            self._populate_channels_table(unique)
            self.grp_channels.setVisible(True)
        else:
            try:
                self.tbl_channels.setRowCount(0)
                self.grp_channels.setVisible(False)
            except Exception:
                pass

        try:
            import napari  # noqa: F401
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Napari missing", f"Napari could not be imported.\n\n{e}")
            return

        # Napari uses (y,x) for 2D
        pts_yx = np.column_stack([pts[:, 1], pts[:, 0]])

        properties = {}
        if z is not None:
            properties["z"] = z
        if ch is not None:
            properties["channel"] = ch

        if self.points_layer is None or self.points_layer not in self.viewer.layers:
            self.points_layer = self.viewer.add_points(
                pts_yx,
                name="Points",
                size=2,
                opacity=0.6,
                properties=properties if properties else None,
            )
        else:
            self.points_layer.data = pts_yx
            try:
                self.points_layer.properties = properties if properties else {}
            except Exception:
                pass

    def _snapshot_channels_table(self) -> dict:
        """Return {raw_label: (use_bool, alias_str)} from the channels table."""
        out = {}
        if not hasattr(self, "tbl_channels"):
            return out
        try:
            for r in range(self.tbl_channels.rowCount()):
                chk = self.tbl_channels.item(r, 0)
                if chk is None:
                    continue
                raw = self.tbl_channels.item(r, 1).text().strip() if self.tbl_channels.item(r, 1) is not None else ""
                alias = self.tbl_channels.item(r, 2).text().strip() if self.tbl_channels.item(r, 2) is not None else raw
                use = bool(chk.checkState() == QtCore.Qt.Checked)
                if raw:
                    out[str(raw)] = (use, alias)
        except Exception:
            pass
        return out

    def _populate_channels_table(self, labels: List[str]) -> None:
        """Populate the channels table from a list of raw labels."""
        if not hasattr(self, "tbl_channels"):
            return
        prev = self._snapshot_channels_table()
        self.tbl_channels.setRowCount(0)
        for lab in labels:
            row = self.tbl_channels.rowCount()
            self.tbl_channels.insertRow(row)

            chk = QtWidgets.QTableWidgetItem("")
            chk.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            chk.setCheckState(QtCore.Qt.Checked)
            self.tbl_channels.setItem(row, 0, chk)

            raw_item = QtWidgets.QTableWidgetItem(str(lab))
            raw_item.setFlags(QtCore.Qt.ItemIsEnabled)
            self.tbl_channels.setItem(row, 1, raw_item)

            alias_default = str(lab)
            if str(lab) in prev:
                use, alias = prev[str(lab)]
                chk.setCheckState(QtCore.Qt.Checked if use else QtCore.Qt.Unchecked)
                alias_default = alias or alias_default

            alias_item = QtWidgets.QTableWidgetItem(alias_default)
            self.tbl_channels.setItem(row, 2, alias_item)

        try:
            self.tbl_channels.resizeColumnsToContents()
        except Exception:
            pass

    # -----------------
    # ROI layer creation
    # -----------------

    def _get_or_create_shapes_layer(self, name: str):
        try:
            layer = self.viewer.layers[name]
            return layer
        except Exception:
            pass

        # Create new
        layer = self.viewer.add_shapes(name=name, shape_type="polygon", edge_width=2)
        return layer

    def _get_or_create_points_layer(self, name: str):
        try:
            layer = self.viewer.layers[name]
            return layer
        except Exception:
            # napari add_points expects (N, D). We'll set later.
            return self.viewer.add_points(np.zeros((0, 2), dtype=float), name=name)

    def _on_create_domain_layer(self):
        self.domain_layer = self._get_or_create_shapes_layer("Domain")
        self.viewer.layers.selection.active = self.domain_layer

    def _on_create_holes_layer(self):
        self.holes_layer = self._get_or_create_shapes_layer("Holes")
        self.viewer.layers.selection.active = self.holes_layer

    def _on_create_seeds_layer(self):
        self.seeds_layer = self._get_or_create_shapes_layer("Seeds")
        self.viewer.layers.selection.active = self.seeds_layer

    # -----------------
    # ROI table refresh
    # -----------------

    def _populate_table_from_layer(self, table: QtWidgets.QTableWidget, layer, default_prefix: str):
        """Populate a ROI table from a Napari Shapes layer.

        The method preserves the previous per-ROI table state (Use/Name/Type) when possible,
        keyed by the original shape index stored in Qt.UserRole.
        """

        # Snapshot previous state
        prev = {}
        try:
            for r in range(table.rowCount()):
                chk_item = table.item(r, 0)
                if chk_item is None:
                    continue
                idx = chk_item.data(QtCore.Qt.UserRole)
                if idx is None:
                    continue
                idx = int(idx)
                prev_name = table.item(r, 1).text().strip() if table.item(r, 1) is not None else ""
                prev_checked = bool(chk_item.checkState() == QtCore.Qt.Checked)

                prev_type = ""
                if table.columnCount() >= 4:
                    w = table.cellWidget(r, 2)
                    if isinstance(w, QtWidgets.QComboBox):
                        prev_type = w.currentText().strip()
                    else:
                        it = table.item(r, 2)
                        prev_type = it.text().strip() if it is not None else ""

                prev[idx] = {"checked": prev_checked, "name": prev_name, "type": prev_type}
        except Exception:
            prev = {}

        table.setRowCount(0)
        if layer is None:
            return
        try:
            data = list(layer.data)
        except Exception:
            return

        has_type = (table.columnCount() >= 4)

        # Hole type presets (editable combo for "best compromise")
        hole_type_presets = ["nucleus", "exclusion", "inner", "other"]

        for i, verts in enumerate(data):
            row = table.rowCount()
            table.insertRow(row)

            chk = QtWidgets.QTableWidgetItem("")
            chk.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            chk.setData(QtCore.Qt.UserRole, int(i))

            # restore previous checked state if available
            if i in prev and prev[i].get("checked") is False:
                chk.setCheckState(QtCore.Qt.Unchecked)
            else:
                chk.setCheckState(QtCore.Qt.Checked)
            table.setItem(row, 0, chk)

            # Name
            default_name = f"{default_prefix}_{i+1}"
            if i in prev and prev[i].get("name"):
                default_name = prev[i]["name"]
            name_item = QtWidgets.QTableWidgetItem(default_name)
            table.setItem(row, 1, name_item)

            # Type (only for tables with 4 columns, currently Holes)
            if has_type:
                combo = QtWidgets.QComboBox()
                combo.setEditable(True)
                combo.addItems(hole_type_presets)
                # default type: nucleus (common), but preserve previous choice
                t = prev.get(i, {}).get("type") or "nucleus"
                # Ensure current text is set even if not in presets
                if t not in hole_type_presets:
                    combo.addItem(t)
                combo.setCurrentText(t)
                table.setCellWidget(row, 2, combo)

                nverts = int(np.asarray(verts).shape[0])
                table.setItem(row, 3, QtWidgets.QTableWidgetItem(str(nverts)))
            else:
                nverts = int(np.asarray(verts).shape[0])
                table.setItem(row, 2, QtWidgets.QTableWidgetItem(str(nverts)))

    def _refresh_roi_tables(self):
        # Resolve layers if user created them manually
        if self.domain_layer is None:
            try:
                self.domain_layer = self.viewer.layers["Domain"]
            except Exception:
                pass
        if self.holes_layer is None:
            try:
                self.holes_layer = self.viewer.layers["Holes"]
            except Exception:
                pass
        if self.seeds_layer is None:
            try:
                self.seeds_layer = self.viewer.layers["Seeds"]
            except Exception:
                pass

        self._populate_table_from_layer(self.tbl_domains, self.domain_layer, "domain")
        self._populate_table_from_layer(self.tbl_holes, self.holes_layer, "hole")
        self._populate_table_from_layer(self.tbl_seeds, self.seeds_layer, "seed")

    def _collect_shapes(self, table: QtWidgets.QTableWidget, layer, kind: str) -> List[dict]:
        out: List[dict] = []
        if layer is None:
            return out

        for row in range(table.rowCount()):
            chk = table.item(row, 0)
            if chk is None or chk.checkState() != QtCore.Qt.Checked:
                continue
            idx = chk.data(QtCore.Qt.UserRole)
            if idx is None:
                continue
            idx = int(idx)

            name_item = table.item(row, 1)
            name = name_item.text().strip() if name_item is not None else f"{kind}_{idx+1}"

            roi_type = None
            if kind == "hole" and table.columnCount() >= 4:
                w = table.cellWidget(row, 2)
                if isinstance(w, QtWidgets.QComboBox):
                    roi_type = w.currentText().strip()
                else:
                    it = table.item(row, 2)
                    roi_type = it.text().strip() if it is not None else ""
                roi_type = roi_type or "nucleus"

            try:
                verts_yx = np.asarray(layer.data[idx], dtype=float)
                geom = _napari_shape_to_polygon(verts_yx)
                verts_xy = np.column_stack([verts_yx[:, 1], verts_yx[:, 0]])
            except Exception as e:
                self.logger.warning(f"Failed to read {kind} ROI #{idx+1}: {e}")
                continue

            d = {"name": name, "geom": geom, "vertices_xy": verts_xy.tolist()}
            if roi_type is not None:
                d["type"] = roi_type
            out.append(d)

        return out

    # -----------------
    # ROI Save / Load
    # -----------------

    def _export_rois_from_table_and_layer(
        self,
        table: QtWidgets.QTableWidget,
        layer,
        kind: str,
        *,
        include_unchecked: bool = True
    ) -> List[dict]:
        """Export ROIs using the table state (names/types) and the layer geometry (vertices)."""
        out: List[dict] = []
        if layer is None:
            return out

        try:
            layer_data = list(layer.data)
        except Exception:
            return out

        for row in range(table.rowCount()):
            chk = table.item(row, 0)
            if chk is None:
                continue
            if (not include_unchecked) and (chk.checkState() != QtCore.Qt.Checked):
                continue

            idx = chk.data(QtCore.Qt.UserRole)
            if idx is None:
                continue
            idx = int(idx)
            if idx < 0 or idx >= len(layer_data):
                continue

            name_item = table.item(row, 1)
            name = name_item.text().strip() if name_item is not None else f"{kind}_{idx+1}"

            roi = {
                "name": name,
                "vertices_yx": np.asarray(layer_data[idx], dtype=float).tolist(),
            }

            if kind == "hole" and table.columnCount() >= 4:
                w = table.cellWidget(row, 2)
                if isinstance(w, QtWidgets.QComboBox):
                    roi_type = w.currentText().strip()
                else:
                    it = table.item(row, 2)
                    roi_type = it.text().strip() if it is not None else ""
                roi["type"] = roi_type or "nucleus"

            out.append(roi)

        return out

    def _apply_roi_table_state(self, table: QtWidgets.QTableWidget, entries: List[dict], kind: str) -> None:
        """After loading layer.data + calling _refresh_roi_tables(), overwrite saved names/types."""
        n = min(table.rowCount(), len(entries))
        for r in range(n):
            entry = entries[r]
            name = str(entry.get("name") or "").strip()
            if name:
                table.item(r, 1).setText(name)

            if kind == "hole" and table.columnCount() >= 4:
                roi_type = str(entry.get("type") or "nucleus").strip()
                w = table.cellWidget(r, 2)
                if isinstance(w, QtWidgets.QComboBox):
                    w.setCurrentText(roi_type)

    def _on_save_rois(self):
        from .roi_io import save_rois_json

        # Ensure tables reflect latest layer shapes
        self._refresh_roi_tables()

        default_dir = self.txt_outdir.text().strip() or os.getcwd()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save ROIs",
            os.path.join(default_dir, "rois_galaxy.json"),
            "JSON (*.json)"
        )
        if not path:
            return

        rois = {
            "domains": self._export_rois_from_table_and_layer(self.tbl_domains, self.domain_layer, "domain", include_unchecked=True),
            "holes": self._export_rois_from_table_and_layer(self.tbl_holes, self.holes_layer, "hole", include_unchecked=True),
            "seeds": self._export_rois_from_table_and_layer(self.tbl_seeds, self.seeds_layer, "seed", include_unchecked=True),
        }

        try:
            save_rois_json(path, rois, app="GalaXY")
            self.logger.info(f"Saved ROIs to: {path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save ROIs", f"Failed to save ROIs:\n{e}")

    def _on_load_rois(self):
        from .roi_io import load_rois_json

        default_dir = self.txt_outdir.text().strip() or os.getcwd()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load ROIs",
            default_dir,
            "JSON (*.json)"
        )
        if not path:
            return

        try:
            rois = load_rois_json(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load ROIs", f"Failed to read ROI file:\n{e}")
            return

        # Create/select layers
        self.domain_layer = self._get_or_create_shapes_layer("Domain")
        self.holes_layer = self._get_or_create_shapes_layer("Holes")
        self.seeds_layer = self._get_or_create_shapes_layer("Seeds")

        # Replace layer geometries
        try:
            self.domain_layer.data = [np.asarray(r["vertices_yx"], dtype=float) for r in rois.get("domains", [])]
            self.holes_layer.data = [np.asarray(r["vertices_yx"], dtype=float) for r in rois.get("holes", [])]
            self.seeds_layer.data = [np.asarray(r["vertices_yx"], dtype=float) for r in rois.get("seeds", [])]
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load ROIs", f"Failed to apply ROIs to Napari layers:\n{e}")
            return

        # Refresh tables then apply saved names/types
        self._refresh_roi_tables()
        self._apply_roi_table_state(self.tbl_domains, rois.get("domains", []), "domain")
        self._apply_roi_table_state(self.tbl_holes, rois.get("holes", []), "hole")
        self._apply_roi_table_state(self.tbl_seeds, rois.get("seeds", []), "seed")

        self.logger.info(f"Loaded ROIs from: {path}")


    # -----------------
    # Preview regions overlay
    # -----------------

    def _on_preview_regions(self) -> None:
        # Lightweight preview: uses first domain ROI only.
        try:
            from .tiling import build_domain_geometry, build_regions_for_profile, union_geoms
            from .profiles import get_profile
            from .reference import ref_geodesic_from_model
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Import error", str(e))
            return

        doms = self._collect_shapes(self.tbl_domains, self.domain_layer, "domain")
        holes = self._collect_shapes(self.tbl_holes, self.holes_layer, "hole")
        seeds = self._collect_shapes(self.tbl_seeds, self.seeds_layer, "seed")
        if not doms:
            QtWidgets.QMessageBox.warning(self, "No domains", "Draw/select at least one Domain polygon.")
            return

        prof = profile_by_name(self.cmb_profile.currentText())
        if prof is None:
            return

        outer_geom = doms[0]["geom"]
        # holes/seeds currently unassigned; in preview we just union them
        hole_geoms = [h["geom"] for h in holes]
        outer_union, holes_union, domain_geom = build_domain_geometry(outer_geoms=[outer_geom], hole_geoms=hole_geoms)

        # Typed holes: for perinuclear analyses we prefer nucleus-only holes
        nuc_geoms = []
        for h in holes:
            t = str(h.get("type", "")).strip().lower()
            if t in ("nucleus", "nuc", "nuclear", "nuclei"):
                nuc_geoms.append(h["geom"])
        nucleus_union = union_geoms(nuc_geoms) if nuc_geoms else holes_union

        seed_union = union_geoms([s["geom"] for s in seeds]) if seeds else shgeom.GeometryCollection([])

        tiling_mode = self.cmb_tiling.currentText().strip() if self.cmb_tiling.count() else prof.default_tiling

        geom_params = GeometryParams(
            pixel_size=float(self.spn_px.value()),
            prune_spurs=bool(self.chk_prune.isChecked()),
            prune_spur_length=float(self.spn_prune_len.value()),
            thickness_mode=str(self.cmb_thick_mode.currentText()),
            thickness_manual=float(self.spn_thick.value()),
        )
        band_params = BandParams(
            mode=str(self.cmb_band_mode.currentText()),
            band_width=float(self.spn_band_w.value()),
            band_length=float(self.spn_band_len.value()),
            subtract_zc=bool(self.chk_subtract_seed.isChecked()),
        )
        binning = WindowBinningParams(
            mode=str(tiling_mode),
            n_bins=int(self.spn_bins.value()),
            seed_band_width=float(self.spn_seed_bw.value()),
            include_seed_as_bin0=bool(self.chk_include_seed.isChecked()),
            grid_rows=int(self.spn_grid_rows.value()),
            grid_cols=int(self.spn_grid_cols.value()),
        )

        from .worker import _axis_for_binning_mode
        axis_name, _ = _axis_for_binning_mode(prof.id, tiling_mode)

        try:
            regions, _ctx = build_regions_for_profile(
                profile=get_profile(prof.id),
                outer_geom=outer_union,
                holes_geom=(nucleus_union if (str(tiling_mode).lower().startswith("perinuclear") or prof.id == "perinuclear") else holes_union),
                domain_geom=domain_geom,
                seed_geom=seed_union,
                geom_backend=str(self.cmb_backend.currentText()),
                geom_params=geom_params,
                band_params=band_params,
                binning=binning,
                axis_override=axis_name,
                logger=self.logger,
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Preview failed", str(e))
            return

        # Create/update overlay shapes layer
        self.overlay_layer = self._get_or_create_shapes_layer("RegionsOverlay")
        rings = []
        for r in regions:
            rings.extend(_geom_to_napari_rings(getattr(r, "geom", None)))
        self.overlay_layer.data = rings
        try:
            self.overlay_layer.face_color = "transparent"
            self.overlay_layer.edge_color = "yellow"
        except Exception:
            pass
        self.viewer.layers.selection.active = self.overlay_layer
        self.logger.info(f"Previewed {len(regions)} regions on RegionsOverlay")

    def _on_preview_skeleton(self) -> None:
        """
        Preview the geodesic skeleton graph in Napari with:
        - SkeletonEdges (shapes)
        - SkeletonNodes (points)
        - SkeletonMedian (points)
        - SkeletonOverlaps (points)
        """
        try:
            from .tiling import build_domain_geometry, build_regions_for_profile, union_geoms
            from .profiles import get_profile
            from .geometry_base import GeometryParams, BandParams
            from .geometry_skeleton import make_skeleton_preview
            from .worker import _axis_for_binning_mode
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Import error", str(e))
            return

        doms = self._collect_shapes(self.tbl_domains, self.domain_layer, "domain")
        holes = self._collect_shapes(self.tbl_holes, self.holes_layer, "hole")
        seeds = self._collect_shapes(self.tbl_seeds, self.seeds_layer, "seed")
        if not doms:
            QtWidgets.QMessageBox.warning(self, "No domains", "Draw/select at least one Domain polygon.")
            return
        if not seeds:
            QtWidgets.QMessageBox.warning(self, "No seeds", "Geodesic preview requires a Seed ROI (contact zone).")
            return

        prof = profile_by_name(self.cmb_profile.currentText())
        if prof is None:
            return

        # This preview is meaningful only for geodesic workflows.
        if not getattr(prof, "uses_geodesic", False):
            QtWidgets.QMessageBox.information(
                self,
                "Not geodesic",
                "Current profile does not use the geodesic skeleton model.\n"
                "Choose a Membrane/Geodesic profile to preview the skeleton.",
            )
            return

        # Encourage skeleton backend (but still allow Auto if it resolves to Skeleton internally).
        backend = str(self.cmb_backend.currentText()).strip()
        if backend.lower() == "ring":
            QtWidgets.QMessageBox.information(
                self,
                "Ring backend selected",
                "Geometry backend is set to Ring.\n"
                "Switch to Skeleton (or Auto) to preview the geodesic skeleton graph.",
            )
            return

        outer_geom = doms[0]["geom"]
        hole_geoms = [h["geom"] for h in holes]
        outer_union, holes_union, domain_geom = build_domain_geometry(outer_geoms=[outer_geom], hole_geoms=hole_geoms)

        # Typed nucleus holes (same logic as regions preview)
        nuc_geoms = []
        for h in holes:
            t = str(h.get("type", "")).strip().lower()
            if t in ("nucleus", "nuc", "nuclear", "nuclei"):
                nuc_geoms.append(h["geom"])
        nucleus_union = union_geoms(nuc_geoms) if nuc_geoms else holes_union

        seed_union = union_geoms([s["geom"] for s in seeds])

        tiling_mode = self.cmb_tiling.currentText().strip() if self.cmb_tiling.count() else prof.default_tiling

        geom_params = GeometryParams(
            pixel_size=float(self.spn_px.value()),
            prune_spurs=bool(self.chk_prune.isChecked()),
            prune_spur_length=float(self.spn_prune_len.value()),
            thickness_mode=str(self.cmb_thick_mode.currentText()),
            thickness_manual=float(self.spn_thick.value()),
        )
        band_params = BandParams(
            mode=str(self.cmb_band_mode.currentText()),
            band_width=float(self.spn_band_w.value()),
            band_length=float(self.spn_band_len.value()),
            subtract_zc=bool(self.chk_subtract_seed.isChecked()),
        )
        binning = WindowBinningParams(
            mode=str(tiling_mode),
            n_bins=int(self.spn_bins.value()),
            seed_band_width=float(self.spn_seed_bw.value()),
            include_seed_as_bin0=bool(self.chk_include_seed.isChecked()),
            grid_rows=int(self.spn_grid_rows.value()),
            grid_cols=int(self.spn_grid_cols.value()),
        )

        axis_name, _ = _axis_for_binning_mode(prof.id, tiling_mode)

        # Build regions to get the geodesic model in ctx["model"]
        try:
            _regions, ctx = build_regions_for_profile(
                profile=get_profile(prof.id),
                outer_geom=outer_union,
                holes_geom=(nucleus_union if (str(tiling_mode).lower().startswith(
                    "perinuclear") or prof.id == "perinuclear") else holes_union),
                domain_geom=domain_geom,
                seed_geom=seed_union,
                geom_backend=backend,
                geom_params=geom_params,
                band_params=band_params,
                binning=binning,
                axis_override=axis_name,
                logger=self.logger,
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Skeleton preview failed", str(e))
            return

        model = ctx.get("model", None)
        if model is None:
            QtWidgets.QMessageBox.critical(self, "Skeleton preview failed", "No geodesic model returned in context.")
            return

        # Compute preview (median step ~= pixel_size)
        try:
            prev = make_skeleton_preview(
                model,
                step=float(geom_params.pixel_size),
                overlap_thresh=0.75 * float(geom_params.pixel_size),
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Skeleton preview failed", str(e))
            return

        # ---- Render in Napari ----
        # Edges (Shapes) in yx order
        edges_yx = []
        for seg in prev.edges_xy:
            # seg: [[x1,y1],[x2,y2]] -> [[y1,x1],[y2,x2]]
            edges_yx.append(np.asarray([[seg[0, 1], seg[0, 0]], [seg[1, 1], seg[1, 0]]], dtype=float))

        edges_layer = self._get_or_create_shapes_layer("SkeletonEdges")
        edges_layer.data = edges_yx
        try:
            edges_layer.shape_type = ["line"] * len(edges_yx)
            edges_layer.face_color = "transparent"
            edges_layer.edge_color = "cyan"
        except Exception:
            pass

        # Raw nodes
        nodes_yx = np.column_stack([prev.nodes_xy[:, 1], prev.nodes_xy[:, 0]]) if prev.nodes_xy.size else np.zeros(
            (0, 2))
        nodes_layer = self._get_or_create_points_layer("SkeletonNodes")
        nodes_layer.data = nodes_yx
        try:
            nodes_layer.size = float(geom_params.pixel_size) * 0.5
            nodes_layer.opacity = 0.8
        except Exception:
            pass

        # Median/resampled skeleton
        med_yx = np.column_stack([prev.median_xy[:, 1], prev.median_xy[:, 0]]) if prev.median_xy.size else np.zeros(
            (0, 2))
        med_layer = self._get_or_create_points_layer("SkeletonMedian")
        med_layer.data = med_yx
        try:
            med_layer.size = float(geom_params.pixel_size) * 0.8
            med_layer.opacity = 0.9
        except Exception:
            pass

        # Overlaps
        ov_yx = np.column_stack(
            [prev.overlaps_xy[:, 1], prev.overlaps_xy[:, 0]]) if prev.overlaps_xy.size else np.zeros((0, 2))
        ov_layer = self._get_or_create_points_layer("SkeletonOverlaps")
        ov_layer.data = ov_yx
        try:
            ov_layer.size = float(geom_params.pixel_size) * 1.2
            ov_layer.opacity = 1.0
        except Exception:
            pass

        # Log QC stats
        self.logger.info(
            "Skeleton preview: "
            f"median NN={prev.nn_median:.3g}, "
            f"IQR=[{prev.nn_iqr[0]:.3g},{prev.nn_iqr[1]:.3g}], "
            f"min NN={prev.nn_min:.3g}, "
            f"overlap_rate={100.0 * prev.overlap_rate:.2f}% "
            f"(step={float(geom_params.pixel_size):.3g})"
        )

        # Focus selection
        try:
            self.viewer.layers.selection.active = med_layer
        except Exception:
            pass

    # -----------------
    # Output dir
    # -----------------

    def _on_choose_outdir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output directory")
        if d:
            self.txt_outdir.setText(d)

    # -----------------
    # Run analysis
    # -----------------

    def _on_run(self):
        if self.points_xy is None:
            QtWidgets.QMessageBox.warning(self, "No points", "Load data and show points first.")
            return

        prof = profile_by_name(self.cmb_profile.currentText())
        if prof is None:
            QtWidgets.QMessageBox.warning(self, "No profile", "Select a profile.")
            return

        domains = self._collect_shapes(self.tbl_domains, self.domain_layer, "domain")
        holes = self._collect_shapes(self.tbl_holes, self.holes_layer, "hole")
        seeds = self._collect_shapes(self.tbl_seeds, self.seeds_layer, "seed")

        if not domains:
            QtWidgets.QMessageBox.warning(self, "No domains", "Draw/select at least one Domain polygon.")
            return

        if prof.requires_holes and not holes:
            QtWidgets.QMessageBox.warning(self, "Missing holes", "This profile requires Holes (e.g. nucleus).")
            return

        if prof.requires_seeds and not seeds:
            QtWidgets.QMessageBox.warning(self, "Missing seeds", "This profile requires Seeds (e.g. ZC/patch).")
            return

        out_base = self.txt_outdir.text().strip()
        if not out_base:
            QtWidgets.QMessageBox.warning(self, "Output", "Choose an output directory.")
            return

        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_name = os.path.splitext(os.path.basename(self._csv_path or "dataset"))[0]
        out_dir = os.path.join(out_base, f"{csv_name}__GalaXY_2_{prof.id}_{ts}")
        os.makedirs(out_dir, exist_ok=True)

        self.logger.set_log_file(os.path.join(out_dir, "run.log"))
        self.logger.info(f"Output directory: {out_dir}")

        tiling_mode = self.cmb_tiling.currentText().strip() if self.cmb_tiling.count() else prof.default_tiling

        geom_params = GeometryParams(
            pixel_size=float(self.spn_px.value()),
            prune_spurs=bool(self.chk_prune.isChecked()),
            prune_spur_length=float(self.spn_prune_len.value()),
            thickness_mode=str(self.cmb_thick_mode.currentText()),
            thickness_manual=float(self.spn_thick.value()),
        )
        band_params = BandParams(
            mode=str(self.cmb_band_mode.currentText()),
            band_width=float(self.spn_band_w.value()),
            band_length=float(self.spn_band_len.value()),
            subtract_zc=bool(self.chk_subtract_seed.isChecked()),
        )
        binning = WindowBinningParams(
            mode=str(tiling_mode),
            n_bins=int(self.spn_bins.value()),
            seed_band_width=float(self.spn_seed_bw.value()),
            include_seed_as_bin0=bool(self.chk_include_seed.isChecked()),
            grid_rows=int(self.spn_grid_rows.value()),
            grid_cols=int(self.spn_grid_cols.value()),
        )

        # DBSCAN
        use_z = bool(getattr(self, "chk_use_z", None) is not None and self.chk_use_z.isChecked() and self.points_z is not None)
        dbscan_params = DBSCANParams(
            eps=float(self.spn_eps.value()),
            min_samples=int(self.spn_min.value()),
            use_z=use_z,
        )

        hier_params = HierarchicalDBSCANParams(
            enabled=bool(self.chk_hier.isChecked()),
            eps=float(self.spn_eps2.value()),
            min_samples=int(self.spn_min2.value()),
        )

        # Ripley
        do_ripley = bool(self.chk_ripley.isChecked())
        ripley_params = RipleyParams(
            r_min=float(self.spn_rmin.value()),
            r_max=float(self.spn_rmax.value()),
            dr=float(self.spn_dr.value()),
            edge_correction=str(self.cmb_edge.currentText()),
        )

        csr_params = None
        if do_ripley and bool(self.chk_csr.isChecked()) and int(self.spn_sims.value()) > 0:
            seed = int(self.spn_seed.value())
            if seed < 0:
                seed = None
            csr_params = CSRParams(
                n_simulations=int(self.spn_sims.value()),
                alpha=float(self.spn_alpha.value()),
                seed=seed,
            )
        save_overview_figures = bool(self.chk_fig_overview.isChecked())
        save_region_figures = bool(self.chk_fig_region.isChecked())
        save_cluster_hist_figures = bool(getattr(self, "chk_fig_cluster_hists", None) is not None and self.chk_fig_cluster_hists.isChecked())
        save_region_cluster_hist_figures = bool(getattr(self, "chk_fig_cluster_hists_region", None) is not None and self.chk_fig_cluster_hists_region.isChecked())
        save_region_ripley_figures = bool(getattr(self, "chk_fig_ripley_region", None) is not None and self.chk_fig_ripley_region.isChecked())

        compute_cluster_ripley = bool(getattr(self, "chk_ripley_per_cluster", None) is not None and self.chk_ripley_per_cluster.isChecked())
        save_cluster_ripley_figures = bool(getattr(self, "chk_fig_ripley_cluster", None) is not None and self.chk_fig_ripley_cluster.isChecked())
        cluster_ripley_min_points = int(getattr(self, "spin_cluster_ripley_minpts", None).value() if getattr(self, "spin_cluster_ripley_minpts", None) is not None else 6)

        export_region_points = bool(self.chk_export_points.isChecked())
        fig_max_points = int(self.spn_fig_maxpts.value())

        # Channel selection (optional)
        channel_labels = self.channel_labels
        channels_to_use = None
        channel_alias_map = None
        if channel_labels is not None and hasattr(self, "tbl_channels") and self.tbl_channels.rowCount() > 0:
            channels_to_use = []
            channel_alias_map = {}
            for r in range(self.tbl_channels.rowCount()):
                chk = self.tbl_channels.item(r, 0)
                raw = self.tbl_channels.item(r, 1).text().strip() if self.tbl_channels.item(r, 1) is not None else ""
                alias = self.tbl_channels.item(r, 2).text().strip() if self.tbl_channels.item(r, 2) is not None else raw
                if raw:
                    channel_alias_map[raw] = alias
                    if chk is None or chk.checkState() == QtCore.Qt.Checked:
                        channels_to_use.append(raw)
            if channels_to_use is not None and len(channels_to_use) == 0:
                channels_to_use = None  # fall back to all

        do_cross_ripley = bool(getattr(self, "chk_cross_ripley", None) is not None and self.chk_cross_ripley.isChecked())
        if not do_ripley:
            do_cross_ripley = False
        if channel_labels is None or channels_to_use is None or len(channels_to_use) < 2:
            do_cross_ripley = False

        # Column names for audit
        z_col = self.cmb_z.currentText() if hasattr(self, "cmb_z") else "<None>"
        ch_col = self.cmb_channel.currentText() if hasattr(self, "cmb_channel") else "<None>"
        if z_col == "<None>":
            z_col = ""
        if ch_col == "<None>":
            ch_col = ""

        # Disable run, enable cancel
        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress.setValue(0)

        # Start worker
        self.worker_thread = QtCore.QThread()
        self.worker = GalaXYWorker(
            points_xy=self.points_xy,
            points_z=self.points_z,
            channel_labels=channel_labels,
            channels_to_use=channels_to_use,
            channel_alias_map=channel_alias_map,
            domains=domains,
            holes=holes,
            seeds=seeds,
            profile_id=prof.id,
            tiling_mode=str(tiling_mode),
            geom_backend=str(self.cmb_backend.currentText()),
            geom_params=geom_params,
            band_params=band_params,
            binning_params=binning,
            dbscan_params=dbscan_params,
            hier_params=hier_params,
            do_ripley=do_ripley,
            ripley_params=ripley_params,
            csr_params=csr_params,
            do_cross_ripley=do_cross_ripley,
            out_dir=out_dir,
            input_csv=self._csv_path,
            x_col=self.cmb_x.currentText(),
            y_col=self.cmb_y.currentText(),
            z_col=z_col,
            channel_col=ch_col,
            logger=self.logger,
            save_overview_figures=save_overview_figures,
            save_region_figures=save_region_figures,
            save_cluster_hist_figures=save_cluster_hist_figures,
            save_region_cluster_hist_figures=save_region_cluster_hist_figures,
            save_region_ripley_figures=save_region_ripley_figures,
            compute_cluster_ripley=compute_cluster_ripley,
            save_cluster_ripley_figures=save_cluster_ripley_figures,
            cluster_ripley_min_points=cluster_ripley_min_points,
            export_region_points=export_region_points,
            fig_max_points=fig_max_points,
            run_mode=run_mode,
        )
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)

        self.worker_thread.start()
        self.logger.info("Analysis started.")

    def _on_cancel(self) -> None:
        """Request cancellation of the running analysis.

        Cancellation is cooperative: the worker stops after finishing the
        current ROI/channel unit.
        """
        if self.worker is None:
            return

        try:
            self.worker.cancel()
            self.logger.warning(
                "Cancel requested. The analysis will stop after the current ROI/channel finishes."
            )
        except Exception as e:
            self.logger.error(f"Cancel request failed: {e}")

    def _on_finished(self, out_dir: str) -> None:
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress.setValue(100)
        self.lbl_runinfo.setText(f"Finished. Outputs saved in:\n{out_dir}")
        self.logger.info("Analysis finished.")

    def _on_failed(self, tb: str) -> None:
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress.setValue(0)
        self.logger.error("Analysis failed:\n" + tb)
        QtWidgets.QMessageBox.critical(self, "Analysis failed", tb)


# ----------------------------
# Entry point
# ----------------------------


def main():
    try:
        import napari
    except Exception as e:
        QtWidgets.QMessageBox.critical(None, "Missing dependency: napari", f"Napari could not be imported.\n\n{e}")
        return

    viewer = napari.Viewer(title="GalaXY_2")
    dock = GalaXYDock(viewer)
    viewer.window.add_dock_widget(dock, area="right", name="GalaXY_2")
    napari.run()


if __name__ == "__main__":
    main()
