import ROOT
import uproot
import awkward as ak
import math
import argparse
import os
import numpy as np


def safe_first(arr):
    return float(arr[0]) if len(arr) > 0 else float("nan")


def extract_selected_info(input_file, output_dir):
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

    # --- Prepare output paths ---
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

    # --- Counters ---
    n_bad_meanx = 0
    n_theta_high = 0
    n_same_z_tracks = 0

    # --- Output ---
    print(f"📝 Writing detailed event dump to {txt_output}")

    with open(txt_output, "w") as f:
        for i, evt in enumerate(arrays):
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
            h_ncls.Fill(n_cls)

            # --- Check for clusters with the same z ---
            z_values = [c["mean_z"] for c in cls_structs]
            same_z_count = len(z_values) - len(set(z_values))
            h_samez.Fill(same_z_count)

            has_bad_meanx = any(c["mean_x"] == -999 for c in cls_structs)
            bad_ncls = n_cls != 2
            bad_theta = theta_val > 72

            if has_bad_meanx:
                n_bad_meanx += 1
            if bad_theta:
                n_theta_high += 1
            if same_z_count >= 2:
                n_same_z_tracks += 1  # count events with ≥2 same-z clusters

            f.write(f"Event {i}\n")
            f.write(
                f"  x0={x0_val:.5f}, y0={y0_val:.5f}, "
                f"theta={theta_val:.5f}, phi={phi_val:.5f}\n"
            )
            f.write(f"  n_cls={n_cls}, same_z_clusters={same_z_count}\n")

            if has_bad_meanx or bad_ncls or bad_theta:
                f.write("  ⚠️ Issues:\n")
                if has_bad_meanx:
                    f.write("    - mean_x = -999\n")
                if bad_ncls:
                    f.write("    - n_cls != 2\n")
                if bad_theta:
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

            else:
                f.write(f"    ⚠️ {len(cls_structs)} clusters found — special case.\n")

            f.write("\n")

    # --- Save histograms ---
    print(f"💾 Saving histograms to {root_output}")
    root_file = ROOT.TFile(root_output, "RECREATE")
    h_ncls.Write()
    h_samez.Write()
    root_file.Close()

    # --- Final summary ---
    print("📊 Summary:")
    print(f"  Tracks with mean_x = -999: {n_bad_meanx}")
    print(f"  Tracks with theta > 72°: {n_theta_high}")
    print(f"  Tracks with ≥2 clusters sharing the same z: {n_same_z_tracks}")
    print("✅ Done.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract events with duplicated z clusters and trigger mask."
    )
    parser.add_argument("--input", required=True, help="Input ROOT file path")
    parser.add_argument(
        "--output-dir",
        required=False,
        default="./output",
        help="Directory to save the output files (will be created if missing)",
    )

    args = parser.parse_args()
    extract_selected_info(args.input, args.output_dir)
