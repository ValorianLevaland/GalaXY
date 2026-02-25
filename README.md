# GalaXY

GalaXY is a **topology-aware** 2D (and optional **2.5D**) cluster analysis application for spatial point patterns (e.g., SMLM localizations).

It generalizes “membrane + contact zone (ZC)” workflows into a single robust backend that can handle:

- **Membrane / cortex rings** (thin bands) with **geodesic distance** from seed ROIs
- **Cytoplasm “blobs”** with centroid-radial shells or boundary shells
- **Perinuclear shells** (distance from nucleus)
- **Seed-centered bands** (distance around patches/adhesions)
- **Grid tiling** (exploratory maps; expert mode)
- (Expert) **branched neurites/processes** via skeleton-geodesic distance

### What’s new in GalaXY_2

- **Multi-channel point sets**: load one CSV containing multiple “channels” (e.g., proteins) and run the full analysis per channel.
- **Optional Z-aware (2.5D) DBSCAN**: cluster in (x,y,z) while keeping the geometry/tiling strictly 2D.
- **Typed hole ROIs**: holes can be tagged as **nucleus**, **exclusion**, **inner** (etc.).  
  *Nucleus-typed* holes are used for perinuclear axes; all hole types are still removed from the domain.
- **Optional cross-Ripley** (expert): compute bivariate Ripley L12(r) between channel pairs.
- **Optional per-region Ripley figures**: one PNG per region in addition to the CSV curves.
- **Ripley-only mode**: skip DBSCAN/hierarchical clustering and run only per-region Ripley first (useful to pick a sensible DBSCAN eps).

The GUI is implemented as a Napari dock widget (Qt).

---

## Quick start

From the `GalaXY_2/` folder:

```bash
python run_galaxy.py
```

---

## Inputs

### CSV

A CSV file containing at least:

- **x** and **y** columns

Optional:

- a **z** column (for 2.5D DBSCAN only; all geometry remains 2D)
- a **channel** column (for multi-channel analysis)

### ROIs (Napari Shapes layers)

- `Domain`: outer boundary for each ROI (each polygon is analyzed as one ROI/cell)
- `Holes` (optional): nucleus/exclusions/inner boundary polygons  
  *In GalaXY_2, holes can be typed in the table (“nucleus”, “exclusion”, “inner”…).*
- `Seeds` (optional/required depending on the profile): ZC / patches / soma

---

## Outputs

For each ROI (in a subfolder `roi_XX__name/`):

- `regions_wkt.csv` — window geometries as WKT for full reproducibility
- `figures/overview_all.png`, `figures/overview_regions.png`

For each **channel** inside each ROI (subfolder `channel_<alias>/`):

- `region_summary.csv`
- `clusters.csv` (empty in Ripley-only mode)
- `superclusters.csv` (if hierarchical clustering enabled)
- `ripley/` (if Ripley enabled): one CSV per region
- `figures/overview.png`, `figures/summary_vs_axis.png` (full mode)
- `figures/ripley_vs_axis.png` + `ripley_summary.csv` (Ripley-only mode)
- optional `figures/region_dbscan/*.png`
- optional `figures/region_ripley/*.png`
- optional `points_labeled.csv` (region labels and cluster labels)

Cross-channel (expert), inside each ROI:

- `cross/<channelA>__<channelB>/<A_to_B|B_to_A>/...`
  - per-region cross-Ripley CSVs and optional PNGs
  - `cross_ripley_summary.csv`

At the **top-level output folder**:

- `run_config.json` (parameters + ROIs + package versions)
- `all_rois__region_summary.csv`
- `all_rois__clusters.csv`
- `all_rois__superclusters.csv` (if any)
- `all_rois__cross_ripley_summary.csv` (if any)

---

## Notes on robustness

- All runs write a `run_config.json` capturing:
  - parameters
  - ROI vertices
  - WKT geometries
  - package versions
  - channel selection / aliasing
- Worker thread isolates CPU-heavy processing from the GUI thread.
- Export callbacks are guarded so figure/IO failures do not crash the analysis.

---

## License

MIT
