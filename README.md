Limadou Tracking Event Analysis

This repository contains a Python-based analysis framework for L2 tracking events from Limadou detector ROOT files. It selects events based on trigger and cluster requirements, computes event-level metrics, checks detector acceptance, and optionally saves results into ROOT TTrees and text dumps.

⸻

Features
  •  Load ROOT files using uproot and awkward-array.
  •  Apply trigger and base masks to select valid events.
  •  Compute event-level metrics:
  •  Track coordinates (x0, y0, theta, phi)
  •  Cluster information (mean_x, mean_y, mean_z, residuals, size)
  •  Flags for common issues:
    •  mean_x = -999
    •  n_cls > 3
    •  theta > 72°
    •  Clusters sharing the same Z
  •  Acceptance checks: whether the track intersects the detector acceptance and TR1/TR2.
  •  Optional creation of a ROOT TTree with selected event branches.
  •  Output histograms for cluster counts, residuals, and Dsum.

⸻

Project Structure

limadou_tracking_test/
├── check_extra_tracks.py      # Main analysis script
├── geometry_utils.py          # Geometry-related functions
├── geometry/Geometry.hh       # ROOT C++ geometry constants
├── output/                    # Default output directory
└── README.md                  # This file


⸻

Installation
  1.  Install Python 3.10+ (recommended).
  2.  Install dependencies:

pip install uproot awkward numpy

  3.  Ensure ROOT is installed and accessible from Python:

# For ROOT 6+
python -m pip install uproot

  4.  Make sure geometry/Geometry.hh exists and is correctly configured.

⸻

Usage

python check_extra_tracks.py --input path/to/input.root --output-dir path/to/output --save-tree

Arguments
  •  --input: Path to the input ROOT file containing L2 events.
  •  --output-dir: Directory to save the output TXT and ROOT files. Default is ./output.
  •  --save-tree: Optional flag. If provided, a ROOT TTree with selected event-level branches is created.

Outputs
  1.  TXT Dump: <input_basename>_selected.txt
Contains per-event information, cluster details, and issue flags.
  2.  ROOT File: <input_basename>_selected.root
  •  Histograms:
  •  h_ncls – Number of clusters per event
  •  h_samez – Number of clusters sharing the same Z
  •  h_resx, h_resy – Residual distributions
  •  h_dsum_vs_ncls – Dsum of residuals vs number of clusters
  •  Optional TTree SelectedEvents with branches:
  •  x0, y0, theta, phi, n_cls, track_hit_TR, in_acceptance

⸻

Example

python check_extra_tracks.py --input data/run001.root --output-dir output/run001 --save-tree

Console Output:

🔍 Opening ROOT file: data/run001.root
✅ Selected 152 events (mask + trigger).
📝 Writing detailed event dump to output/run001/run001_selected.txt
✅ TTree 'SelectedEvents' written successfully with track_hit_TR and in_acceptance.
📊 Summary:
  Tracks with mean_x = -999: 3
  Tracks with theta > 72°: 5
  Tracks with ≥2 clusters sharing the same z: 2
  n_cls=2 tracks not in acceptance: 10
  Tracks with n_cls > 3: 8
✅ Done.


⸻

Notes
  •  All angles in the output are in degrees, but internal calculations convert to radians where needed.
  •  The geometry_utils.py module contains all geometry-related functions:
  •  load_geometry()
  •  track_hit_TR()
  •  is_in_acceptance()
  •  handle_two_cluster_track()
  •  The analysis assumes 2-cluster tracks as the standard case; single-cluster and multi-cluster tracks are handled accordingly.

⸻

License

MIT License – Free to use and modify for scientific research.
