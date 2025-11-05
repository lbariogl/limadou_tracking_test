import ROOT
import uproot
import awkward as ak
import math
import argparse
import os
import numpy as np
from geometry_utils import (
    load_geometry,
    track_hit_TR,
    is_in_acceptance,
    handle_two_cluster_track,
)


# ============================================================
# Utility functions
# ============================================================
def safe_first(arr):
    """Safely return the first element of an array, or NaN if empty."""
    return float(arr[0]) if len(arr) > 0 else float("nan")


# ============================================================
# Data loading
# ============================================================
def load_and_select_events(input_file):
    """Open ROOT file, apply trigger and base mask, and return selected arrays."""
    print(f"🔍 Opening ROOT file: {input_file}")
    with uproot.open(input_file) as f:
        tree = f["L2"]

        trig_conf_flag = tree["L2Event/trig_conf_flag[6]"].array(library="np")
        if getattr(trig_conf_flag, "ndim", None) == 2:
            trig_mask = trig_conf_flag[:, 1] == 1
            print("Using trigger mask (ndim=2)")
        else:
            trig_mask = np.array([evt[1] for evt in trig_conf_flag]) == 1
            print("Using trigger mask (object-array)")

        x0 = tree["L2Event/x0"].array(library="ak")
        x0_m2 = tree["L2Event/x0_m2"].array(library="ak")
        base_mask = (ak.num(x0) == 1) & (ak.num(x0_m2) == 0)
        mask = base_mask & trig_mask

        branches = [
            "L2Event/x0",
            "L2Event/y0",
            "L2Event/theta",
            "L2Event/phi",
            "L2Event/cls_mean_x",
            "L2Event/cls_mean_y",
            "L2Event/cls_mean_z",
            "L2Event/cls_size",
            "L2Event/cls_res_x",
            "L2Event/cls_res_y",
        ]
        arrays = tree.arrays(branches, library="ak")[mask]

    print(f"✅ Selected {len(arrays)} events (mask + trigger).")
    return arrays


# ============================================================
# Event-level analysis
# ============================================================
def analyze_event(evt):
    """Compute key metrics, flags, and acceptance for one event."""
    x0_val = safe_first(evt["L2Event/x0"])
    y0_val = safe_first(evt["L2Event/y0"])
    theta_val = safe_first(evt["L2Event/theta"])
    phi_val = safe_first(evt["L2Event/phi"])

    # Build cluster structures including residuals and size
    cls_structs = [
        {
            "mean_x": float(mx),
            "mean_y": float(my),
            "mean_z": float(mz),
            "size": int(sz),
            "res_x": float(rx),
            "res_y": float(ry),
        }
        for mx, my, mz, sz, rx, ry in zip(
            evt["L2Event/cls_mean_x"],
            evt["L2Event/cls_mean_y"],
            evt["L2Event/cls_mean_z"],
            evt["L2Event/cls_size"],
            evt["L2Event/cls_res_x"],
            evt["L2Event/cls_res_y"],
        )
    ]

    n_cls = len(cls_structs)
    z_values = [c["mean_z"] for c in cls_structs]
    same_z_count = len(z_values) - len(set(z_values))

    has_bad_meanx = any(c["mean_x"] == -999 for c in cls_structs)
    bad_ncls = n_cls > 3
    bad_theta = theta_val > 72

    issues = {
        "mean_x": has_bad_meanx,
        "n_cls": bad_ncls,
        "theta": bad_theta,
        "same_z_count": same_z_count,
    }

    hit_tr = False
    in_acc = False

    if n_cls == 2:
        # Compute acceptance and TR hit for 2-cluster tracks
        result = handle_two_cluster_track(
            cls_structs, math.radians(theta_val), math.radians(phi_val), dist_z=3.5
        )
        if result:
            in_acc = result["in_acceptance"]
            hit_tr = result["hit_TR"]
    else:
        # For other tracks, only check TR hit
        hit_tr = track_hit_TR(
            x0_val, y0_val, math.radians(theta_val), math.radians(phi_val)
        )

    return (
        x0_val,
        y0_val,
        theta_val,
        phi_val,
        cls_structs,
        n_cls,
        issues,
        hit_tr,
        in_acc,
    )


# ============================================================
# Text output and histogram filling
# ============================================================
def write_txt_dump(
    txt_output,
    arrays,
    h_ncls,
    h_samez,
    h_resx,
    h_resy,
    h_dsum_vs_ncls,
    counters,
):
    """Write event details into a TXT file and fill histograms."""
    print(f"📝 Writing detailed event dump to {txt_output}")
    with open(txt_output, "w") as f:
        for i, evt in enumerate(arrays):
            (
                x0_val,
                y0_val,
                theta_val,
                phi_val,
                cls_structs,
                n_cls,
                issues,
                hit_tr,
                in_acc,
            ) = analyze_event(evt)

            # --- Fill histograms ---
            h_ncls.Fill(n_cls)
            # For demonstration, we fill h_samez with 0 as placeholder
            h_samez.Fill(0)

            # Compute Dsum for residuals if available
            Dsum = 0.0
            cls_structs = [
                {
                    "mean_x": float(mx),
                    "mean_y": float(my),
                    "mean_z": float(mz),
                    "res_x": float(rx),
                    "res_y": float(ry),
                    "size": int(sz),
                }
                for mx, my, mz, rx, ry, sz in zip(
                    evt["L2Event/cls_mean_x"],
                    evt["L2Event/cls_mean_y"],
                    evt["L2Event/cls_mean_z"],
                    evt["L2Event/cls_res_x"],
                    evt["L2Event/cls_res_y"],
                    evt["L2Event/cls_size"],
                )
            ]

            for c in cls_structs:
                h_resx.Fill(c["res_x"])
                h_resy.Fill(c["res_y"])
                Dsum += c["res_x"] ** 2 + c["res_y"] ** 2

            h_dsum_vs_ncls.Fill(Dsum, n_cls)

            # --- Update counters ---
            if issues["mean_x"]:
                counters["bad_meanx"] += 1
            if issues["theta"]:
                counters["high_theta"] += 1
            if issues["same_z_count"] >= 2:
                counters["same_z_tracks"] += 1

            # --- Write event details ---
            f.write(f"Event {i}\n")
            f.write(
                f"  x0={x0_val:.5f}, y0={y0_val:.5f}, "
                f"theta={theta_val:.5f}, phi={phi_val:.5f}\n"
            )
            f.write(
                f"  n_cls={n_cls}, Dsum={Dsum:.5f}, track_hit_TR={int(hit_tr)}, in_acceptance={int(in_acc)}\n"
            )

            if any([issues["mean_x"], issues["n_cls"], issues["theta"]]):
                f.write("  ⚠️ Issues:\n")
                if issues["mean_x"]:
                    f.write("    - mean_x = -999\n")
                if issues["n_cls"]:
                    f.write("    - n_cls > 3\n")
                if issues["theta"]:
                    f.write("    - theta > 72°\n")

            for j, c in enumerate(cls_structs):
                f.write(
                    f"    cls[{j}] -> mean_x={c['mean_x']:.5f}, "
                    f"mean_y={c['mean_y']:.5f}, "
                    f"mean_z={c['mean_z']:.5f}, "
                    f"size={c['size']}, "
                    f"res_x={c['res_x']:.5f}, "
                    f"res_y={c['res_y']:.5f}\n"
                )

            f.write("\n")


# ============================================================
# ROOT tree output
# ============================================================
def create_ttree(root_file, arrays):
    """Create and write a ROOT TTree with selected event-level info."""
    root_file.cd()
    tree_out = ROOT.TTree("SelectedEvents", "Selected Events after filtering")

    x0_val = np.zeros(1, dtype=np.float32)
    y0_val = np.zeros(1, dtype=np.float32)
    theta_val = np.zeros(1, dtype=np.float32)
    phi_val = np.zeros(1, dtype=np.float32)
    n_cls = np.zeros(1, dtype=np.int32)
    hit_tr = np.zeros(1, dtype=np.int32)
    in_acc = np.zeros(1, dtype=np.int32)

    tree_out.Branch("x0", x0_val, "x0/F")
    tree_out.Branch("y0", y0_val, "y0/F")
    tree_out.Branch("theta", theta_val, "theta/F")
    tree_out.Branch("phi", phi_val, "phi/F")
    tree_out.Branch("n_cls", n_cls, "n_cls/I")
    tree_out.Branch("track_hit_TR", hit_tr, "track_hit_TR/I")
    tree_out.Branch("in_acceptance", in_acc, "in_acceptance/I")

    for evt in arrays:
        (
            x0_val[0],
            y0_val[0],
            theta_val[0],
            phi_val[0],
            _,
            n_cls[0],
            _,
            hit_tr[0],
            in_acc[0],
        ) = analyze_event(evt)
        tree_out.Fill()

    tree_out.Write()
    print(
        "✅ TTree 'SelectedEvents' written successfully with track_hit_TR and in_acceptance."
    )


# ============================================================
# Main orchestrator
# ============================================================
def extract_selected_info(input_file, output_dir, save_tree=False):
    load_geometry()
    arrays = load_and_select_events(input_file)

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    txt_output = os.path.join(output_dir, f"{base_name}_selected.txt")
    root_output = os.path.join(output_dir, f"{base_name}_selected.root")

    # --- Histograms ---
    h_ncls = ROOT.TH1F(
        "h_ncls", "Number of Clusters per Event;N_{cls};Entries", 16, -0.5, 15.5
    )
    h_samez = ROOT.TH1F(
        "h_samez", "Clusters with same Z per Event;Count;Entries", 16, -0.5, 15.5
    )
    h_resx = ROOT.TH1F("h_resx", "Residual X;res_x;Entries", 100, -0.5, 0.5)
    h_resy = ROOT.TH1F("h_resy", "Residual Y;res_y;Entries", 100, -0.5, 0.5)

    # Variable binning for Dsum
    dsum_edges = np.concatenate(
        [
            np.linspace(0, 50, 20, endpoint=False),
            np.linspace(50, 200, 15, endpoint=False),
            np.linspace(200, 1000, 10, endpoint=False),
            np.linspace(1000, 2000, 6),
        ]
    ).astype(np.float64)

    h_dsum_vs_ncls = ROOT.TH2F(
        "h_dsum_vs_ncls",
        "Dsum vs Ncls;Dsum;N_{cls}",
        len(dsum_edges) - 1,
        dsum_edges,
        16,
        -0.5,
        15.5,
    )

    counters = {"bad_meanx": 0, "high_theta": 0, "same_z_tracks": 0}

    write_txt_dump(
        txt_output, arrays, h_ncls, h_samez, h_resx, h_resy, h_dsum_vs_ncls, counters
    )

    print(f"💾 Saving output ROOT file: {root_output}")
    root_file = ROOT.TFile(root_output, "RECREATE")
    h_ncls.Write()
    h_samez.Write()
    h_resx.Write()
    h_resy.Write()
    h_dsum_vs_ncls.Write()
    if save_tree:
        create_ttree(root_file, arrays)
    root_file.Close()

    print("📊 Summary:")
    print(f"  Tracks with mean_x = -999: {counters['bad_meanx']}")
    print(f"  Tracks with theta > 72°: {counters['high_theta']}")
    print(f"  Tracks with ≥2 clusters sharing the same z: {counters['same_z_tracks']}")
    ncls2_not_in_acc = sum(
        1 for evt in arrays if analyze_event(evt)[5] == 2 and not analyze_event(evt)[8]
    )
    print(f"  n_cls=2 tracks not in acceptance: {ncls2_not_in_acc}")
    ncls_gt3 = sum(1 for evt in arrays if analyze_event(evt)[5] > 3)
    print(f"  Tracks with n_cls > 3: {ncls_gt3}")
    print("✅ Done.\n")


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and analyze L2 events.")
    parser.add_argument("--input", required=True, help="Input ROOT file path")
    parser.add_argument(
        "--output-dir", default="./output", help="Directory to save outputs"
    )
    parser.add_argument(
        "--save-tree",
        action="store_true",
        help="Save a ROOT TTree with selected branches",
    )
    args = parser.parse_args()

    extract_selected_info(args.input, args.output_dir, args.save_tree)
