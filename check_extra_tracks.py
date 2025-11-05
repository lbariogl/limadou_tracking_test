import ROOT
import uproot
import awkward as ak
import math
import argparse
import os
import numpy as np


# -------------------- Utility functions -------------------- #
def safe_first(arr):
    """Safely return the first element of an array, or NaN if empty."""
    return float(arr[0]) if len(arr) > 0 else float("nan")


# -------------------- Data loading -------------------- #
def load_and_select_events(input_file):
    """Open ROOT file, apply trigger and base mask, and return selected arrays."""
    print(f"🔍 Opening ROOT file: {input_file}")
    with uproot.open(input_file) as f:
        tree = f["L2"]

        # --- Trigger mask ---
        trig_conf_flag = tree["L2Event/trig_conf_flag[6]"].array(library="np")
        if getattr(trig_conf_flag, "ndim", None) == 2:
            trig_mask = trig_conf_flag[:, 1] == 1
            print("Using trigger mask (ndim=2)")
        else:
            trig_mask = np.array([evt[1] for evt in trig_conf_flag]) == 1
            print("Using trigger mask (object-array)")

        # --- Selection mask ---
        x0 = tree["L2Event/x0"].array(library="ak")
        x0_m2 = tree["L2Event/x0_m2"].array(library="ak")
        base_mask = (ak.num(x0) == 1) & (ak.num(x0_m2) == 0)
        mask = base_mask & trig_mask

        # --- Load relevant branches ---
        branches = [
            "L2Event/x0",
            "L2Event/y0",
            "L2Event/theta",
            "L2Event/phi",
            "L2Event/cls_mean_x",
            "L2Event/cls_mean_y",
            "L2Event/cls_mean_z",
            "L2Event/cls_size",
        ]
        arrays = tree.arrays(branches, library="ak")[mask]

    print(f"✅ Selected {len(arrays)} events (mask + trigger).")
    return arrays


# -------------------- Event-level analysis -------------------- #
def analyze_event(evt):
    """Compute key metrics and flags for one event."""
    x0_val = safe_first(evt["L2Event/x0"])
    y0_val = safe_first(evt["L2Event/y0"])
    theta_val = safe_first(evt["L2Event/theta"])
    phi_val = safe_first(evt["L2Event/phi"])

    cls_structs = [
        {
            "mean_x": float(mx),
            "mean_y": float(my),
            "mean_z": float(mz),
            "size": int(sz),
        }
        for mx, my, mz, sz in zip(
            evt["L2Event/cls_mean_x"],
            evt["L2Event/cls_mean_y"],
            evt["L2Event/cls_mean_z"],
            evt["L2Event/cls_size"],
        )
    ]

    n_cls = len(cls_structs)
    z_values = [c["mean_z"] for c in cls_structs]
    same_z_count = len(z_values) - len(set(z_values))

    has_bad_meanx = any(c["mean_x"] == -999 for c in cls_structs)
    bad_ncls = n_cls != 2
    bad_theta = theta_val > 72

    issues = {
        "mean_x": has_bad_meanx,
        "n_cls": bad_ncls,
        "theta": bad_theta,
        "same_z_count": same_z_count,
    }

    return x0_val, y0_val, theta_val, phi_val, cls_structs, n_cls, issues


# -------------------- Text output -------------------- #
def write_txt_dump(txt_output, arrays, h_ncls, h_samez, counters):
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
            ) = analyze_event(evt)

            h_ncls.Fill(n_cls)
            h_samez.Fill(issues["same_z_count"])

            if issues["mean_x"]:
                counters["bad_meanx"] += 1
            if issues["theta"]:
                counters["high_theta"] += 1
            if issues["same_z_count"] >= 2:
                counters["same_z_tracks"] += 1

            # --- Write to TXT file ---
            f.write(f"Event {i}\n")
            f.write(
                f"  x0={x0_val:.5f}, y0={y0_val:.5f}, "
                f"theta={theta_val:.5f}, phi={phi_val:.5f}\n"
            )
            f.write(f"  n_cls={n_cls}, same_z_clusters={issues['same_z_count']}\n")

            if any([issues["mean_x"], issues["n_cls"], issues["theta"]]):
                f.write("  ⚠️ Issues:\n")
                if issues["mean_x"]:
                    f.write("    - mean_x = -999\n")
                if issues["n_cls"]:
                    f.write("    - n_cls != 2\n")
                if issues["theta"]:
                    f.write("    - theta > 72°\n")

            for j, c in enumerate(cls_structs):
                f.write(
                    f"    cls[{j}] -> mean_x={c['mean_x']:.5f}, "
                    f"mean_y={c['mean_y']:.5f}, "
                    f"mean_z={c['mean_z']:.5f}, "
                    f"size={c['size']}\n"
                )

            if len(cls_structs) == 3:
                for i1, i2 in [(0, 1), (1, 2)]:
                    dx = cls_structs[i2]["mean_x"] - cls_structs[i1]["mean_x"]
                    dy = cls_structs[i2]["mean_y"] - cls_structs[i1]["mean_y"]
                    dz = cls_structs[i2]["mean_z"] - cls_structs[i1]["mean_z"]
                    r = math.sqrt(dx * dx + dy * dy + dz * dz)
                    phi = math.atan2(dy, dx)
                    theta = math.acos(dz / r) if r != 0 else float("nan")
                    f.write(
                        f"    Δ(#{i1}->{i2}): "
                        f"r={r:.5f}, phi={phi:.5f}, theta={theta:.5f}\n"
                    )
            f.write("\n")


# -------------------- ROOT output -------------------- #
def create_ttree(root_file, arrays):
    """Create a real ROOT TTree with selected branches."""
    root_file.cd()
    tree_out = ROOT.TTree("SelectedEvents", "Selected Events after filtering")

    x0_val = np.zeros(1, dtype=np.float32)
    y0_val = np.zeros(1, dtype=np.float32)
    theta_val = np.zeros(1, dtype=np.float32)
    phi_val = np.zeros(1, dtype=np.float32)
    n_cls = np.zeros(1, dtype=np.int32)

    tree_out.Branch("x0", x0_val, "x0/F")
    tree_out.Branch("y0", y0_val, "y0/F")
    tree_out.Branch("theta", theta_val, "theta/F")
    tree_out.Branch("phi", phi_val, "phi/F")
    tree_out.Branch("n_cls", n_cls, "n_cls/I")

    for evt in arrays:
        x0_val[0] = safe_first(evt["L2Event/x0"])
        y0_val[0] = safe_first(evt["L2Event/y0"])
        theta_val[0] = safe_first(evt["L2Event/theta"])
        phi_val[0] = safe_first(evt["L2Event/phi"])
        n_cls[0] = len(evt["L2Event/cls_mean_x"])
        tree_out.Fill()

    tree_out.Write()
    print("✅ TTree 'SelectedEvents' written successfully.")


# -------------------- Main orchestrator -------------------- #
def extract_selected_info(input_file, output_dir, save_tree=False):
    arrays = load_and_select_events(input_file)

    # Prepare output
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    txt_output = os.path.join(output_dir, f"{base_name}_selected.txt")
    root_output = os.path.join(output_dir, f"{base_name}_selected.root")

    # Create histograms
    h_ncls = ROOT.TH1F(
        "h_ncls", "Number of Clusters per Event;N_{cls};Entries", 16, -0.5, 15.5
    )
    h_samez = ROOT.TH1F(
        "h_samez", "Clusters with same Z per Event;Count;Entries", 16, -0.5, 15.5
    )

    counters = {"bad_meanx": 0, "high_theta": 0, "same_z_tracks": 0}

    # Write text and fill histograms
    write_txt_dump(txt_output, arrays, h_ncls, h_samez, counters)

    # Save to ROOT file
    print(f"💾 Saving output ROOT file: {root_output}")
    root_file = ROOT.TFile(root_output, "RECREATE")
    h_ncls.Write()
    h_samez.Write()
    if save_tree:
        create_ttree(root_file, arrays)
    root_file.Close()

    # Summary
    print("📊 Summary:")
    print(f"  Tracks with mean_x = -999: {counters['bad_meanx']}")
    print(f"  Tracks with theta > 72°: {counters['high_theta']}")
    print(f"  Tracks with ≥2 clusters sharing the same z: {counters['same_z_tracks']}")
    print("✅ Done.\n")


# -------------------- Entry point -------------------- #
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
