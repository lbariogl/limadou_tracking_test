import ROOT
import uproot
import awkward as ak
import math
import argparse
import os


def safe_first(arr):
    return float(arr[0]) if len(arr) > 0 else float("nan")


def extract_selected_info(input_file, txt_output):
    print(f"🔍 Opening ROOT file: {input_file}")
    with uproot.open(input_file) as f:
        # ✅ Correct tree name
        tree = f["L2"]

        # --- Apply your existing mask logic ---
        x0 = tree["L2Event/x0"].array(library="ak")
        x0_m2 = tree["L2Event/x0_m2"].array(library="ak")
        mask = (ak.num(x0) == 1) & (ak.num(x0_m2) == 0)

        # --- Load only required branches ---
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

        arrays = tree.arrays(branches, library="ak")
        arrays = arrays[mask]

    print(f"✅ Selected {len(arrays)} events matching criteria.")

    # --- Write output to text ---
    print(f"📝 Writing detailed event dump to {txt_output}")
    os.makedirs(os.path.dirname(txt_output), exist_ok=True)

    with open(txt_output, "w") as f:
        for i, evt in enumerate(arrays):

            x0_val = safe_first(evt["L2Event/x0"])
            y0_val = safe_first(evt["L2Event/y0"])
            theta_val = safe_first(evt["L2Event/theta"])
            phi_val = safe_first(evt["L2Event/phi"])

            f.write(f"Event {i}\n")
            f.write(
                f"  x0={x0_val:.5f}, y0={y0_val:.5f}, "
                f"theta={theta_val:.5f}, phi={phi_val:.5f}\n"
            )

            # --- Build array of cluster structs (array of dicts) ---
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

            f.write(f"  n_cls={len(cls_structs)}\n")
            for j, c in enumerate(cls_structs):
                f.write(
                    f"    cls[{j}] -> mean_x={c['mean_x']:.5f}, "
                    f"mean_y={c['mean_y']:.5f}, "
                    f"mean_z={c['mean_z']:.5f}, "
                    f"size={c['size']}\n"
                )

            # --- Special handling: 3 clusters ---
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

            elif len(cls_structs) == 4:
                f.write("    ⚠️ 4 clusters found — special case.\n")

            f.write("\n")

    print("✅ Done. Events successfully written.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract events with 1 x0 and 0 x0_m2, print detailed info."
    )
    parser.add_argument("input", help="Input ROOT file")
    parser.add_argument("--txt", required=True, help="Output TXT file path")

    args = parser.parse_args()
    extract_selected_info(args.input, args.txt)
