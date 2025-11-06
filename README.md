# L2 Event Selection and Analysis

## Overview

This project provides a Python-based framework to analyze L2 events stored in ROOT files. The main purpose is to:

- Apply trigger and base selection masks to events.
- Compute event-level quantities such as track angles, cluster counts, residuals, and derived metrics.
- Check detector acceptance and whether the track intersects the TR layers.
- Produce detailed TXT dumps and ROOT histograms.
- Optionally, store selected information in a new ROOT TTree for further analysis.

The project is modular and can be extended to handle different selection criteria or geometries.

## Features

- Event selection based on trigger flags and cluster multiplicity.
- Track-level analysis including theta, phi, residuals, and Dsum.
- Acceptance check based on detector geometry.
- Handling of 2-cluster tracks with interpolation of missing layers.
- Output:
  - TXT dump with event and cluster details, including issues.
  - ROOT histograms for quick inspection of key quantities.
  - Optional ROOT TTree with selected branches including `track_hit_TR` and `in_acceptance`.

## File Structure

- `check_extra_tracks.py`: Main script for event selection, analysis, and output.
- `geometry_utils.py`: Geometry-related functions for detector acceptance and TR hit checks.
- `geometry/Geometry.hh`: Header file defining detector constants.
- `output/`: Directory where TXT dumps and ROOT files will be saved.

## Usage

Run the main script with:

```bash
python check_extra_tracks.py --input <input_root_file> --output-dir <output_dir> [--save-tree]
```

Options:

- `--input`: Path to the input ROOT file containing L2 events.
- `--output-dir`: Directory to save TXT and ROOT outputs (default: `./output`).
- `--save-tree`: Include this flag to save a ROOT TTree with selected event branches.

Example:

```bash
python check_extra_tracks.py --input data/L2_sample.root --output-dir results --save-tree
```

## Outputs

1. **TXT dump**: Contains event-by-event details including:
   - `x0`, `y0`, `theta`, `phi`
   - Number of clusters (`n_cls`)
   - Dsum of residuals
   - Track TR hit and acceptance flags
   - Warnings for anomalous values (e.g., `mean_x=-999`, `theta>72°`, clusters sharing the same Z)

2. **ROOT histograms**:
   - `h_ncls`: Number of clusters per event
   - `h_samez`: Clusters with same Z per event
   - `h_resx`, `h_resy`: Residual distributions
   - `h_dsum_vs_ncls`: 2D histogram of Dsum vs number of clusters

3. **Optional ROOT TTree** (`SelectedEvents`):
   - `x0`, `y0`, `theta`, `phi`, `n_cls`
   - `track_hit_TR`, `in_acceptance`

## Notes

- Geometry-related functions have been modularized into `geometry_utils.py`. Modify the `Geometry.hh` constants if the detector geometry changes.
- The code handles tracks with 2 clusters specially by interpolating missing layers and checking acceptance.
- Make sure the input ROOT files contain the required branches listed in the script:
  - `x0`, `y0`, `theta`, `phi`
  - `cls_mean_x`, `cls_mean_y`, `cls_mean_z`, `cls_size`, `cls_res_x`, `cls_res_y`
  - `trig_conf_flag[6]` for trigger selection
