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

ROOT.gStyle.SetOptStat(0)


from utils import safe_first, load_and_select_events, analyze_event


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
                missing_in_acc,
            ) = analyze_event(evt)

            # --- Fill histograms ---
            h_ncls.Fill(n_cls)

            # Compute Dsum for residuals if available
            Dsum = sum(c["res_x"] ** 2 + c["res_y"] ** 2 for c in cls_structs)
            h_dsum_vs_ncls.Fill(Dsum, n_cls)

            for c in cls_structs:
                h_resx.Fill(c["res_x"])
                h_resy.Fill(c["res_y"])

            # --- Update counters ---
            if issues["mean_x"]:
                counters["bad_meanx"] += 1
            if issues["no_TR_hit"]:
                counters["no_TR_hit"] += 1
            if issues["same_z_count"] >= 2:
                counters["same_z_tracks"] += 1

            # --- Write event details ---
            f.write(f"Event {i}\n")
            f.write(
                f"  x0={x0_val:.5f}, y0={y0_val:.5f}, "
                f"theta={theta_val:.5f}, phi={phi_val:.5f}\n"
            )
            f.write(
                f"  n_cls={n_cls}, Dsum={Dsum:.5f}, track_hit_TR={int(hit_tr)}, missing_in_acceptance={int(missing_in_acc)}\n"
            )

            if any([issues["mean_x"], issues["n_cls"], issues["no_TR_hit"]]):
                f.write("  ⚠️ Issues:\n")
                if issues["mean_x"]:
                    f.write("    - mean_x = -999\n")
                if issues["n_cls"]:
                    f.write("    - n_cls > 3\n")
                if issues["no_TR_hit"]:
                    f.write("    - no_TR_hit\n")

            for j, c in enumerate(cls_structs):
                f.write(
                    f"  Cluster {j}: "
                    f"mean_x={c['mean_x']:.3f}, mean_y={c['mean_y']:.3f}, mean_z={c['mean_z']:.3f}, "
                    f"size={c['size']}, res_x={c['res_x']:.3f}, res_y={c['res_y']:.3f}, "
                    f"track_idx={c['track_idx']}\n"
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
    tree_out.Branch("missing_in_acceptance", in_acc, "missing_in_acceptance/I")

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
        "✅ TTree 'SelectedEvents' written successfully with track_hit_TR and missing_in_acceptance."
    )


# ============================================================
# Summary histogram creation (updated with "good tracks" count)
# ============================================================
def make_summary_hist(arrays, output_file, output_dir):
    """
    Create and fill a ROOT histogram summarizing event counts.

    Categories:
      1. Total tracks after masks
      2. Tracks with n_cls = 2
      3. Tracks with n_cls > 3
      4. Tracks with track_hit_TR
      5. Tracks with n_cls=2 & track_hit_TR & !missing_in_acc
      6. Tracks with mean_x = -999
      7. Tracks with Dsum > 10
      8. Tracks with ≥2 clusters having the same z
      9. "Good" tracks passing all quality conditions
    """

    print("\n📊 Building summary histogram...")

    # --- Define histogram ---
    h_summary = ROOT.TH1F("h_summary", "Event summary counts;;Counts", 9, 0.5, 9.5)
    labels = [
        "Tracks after masks",
        "n_cls = 2",
        "n_cls > 3",
        "track_hit_TR",
        "n_cls = 2 + TR + out of acc",
        "mean_x = -999",
        "Dsum > 10",
        "clusters same z",
        "Good tracks",
    ]
    for i, label in enumerate(labels, start=1):
        h_summary.GetXaxis().SetBinLabel(i, label)

    # --- Counters ---
    total_after_masks = len(arrays)
    ncls_eq2 = ncls_gt3 = track_hit_tr_count = ncls2_tr_missing_out_of_acc = 0
    bad_meanx = dsum_gt10 = samez_ge2 = good_tracks = 0

    for evt in arrays:
        (
            x0_val,
            y0_val,
            theta_val,
            phi_val,
            cls_structs,
            n_cls,
            issues,
            hit_tr,
            missing_in_acc,
        ) = analyze_event(evt)

        # (1) n_cls categories
        if n_cls == 2:
            ncls_eq2 += 1
        if n_cls > 3:
            ncls_gt3 += 1

        # (2) TR hit and missing out of acceptance
        if hit_tr:
            track_hit_tr_count += 1
        if n_cls == 2 and hit_tr and not missing_in_acc:
            ncls2_tr_missing_out_of_acc += 1

        # (3) mean_x = -999 (taken directly from issues)
        if issues["mean_x"]:
            bad_meanx += 1

        # (4) Dsum
        Dsum = sum(c["res_x"] ** 2 + c["res_y"] ** 2 for c in cls_structs)
        if Dsum > 10:
            dsum_gt10 += 1

        # (5) Same-z clusters (taken directly from issues)
        if issues["same_z_count"] >= 2:
            samez_ge2 += 1

        # (6) "Good track" selection
        # Conditions:
        # - n_cls < 4
        # - not bad_mean_x
        # - hit_TR
        # - if n_cls == 2: not in_acc ; if n_cls == 3: automatically ok
        # - Dsum < 10
        # - no same-z clusters
        if (
            n_cls < 4
            and not issues["mean_x"]
            and hit_tr
            and (n_cls != 2 or not missing_in_acc)
            and Dsum < 10
            and issues["same_z_count"] == 0
        ):
            good_tracks += 1

    # --- Fill histogram bins ---
    values = [
        total_after_masks,
        ncls_eq2,
        ncls_gt3,
        track_hit_tr_count,
        ncls2_tr_missing_out_of_acc,
        bad_meanx,
        dsum_gt10,
        samez_ge2,
        good_tracks,
    ]
    for i, v in enumerate(values, start=1):
        h_summary.SetBinContent(i, v)

    # --- Print text summary ---
    print("\n===== Summary Counts =====")
    for label, value in zip(labels, values):
        perc = (value / total_after_masks * 100) if total_after_masks > 0 else 0
        print(f"{label:<45}: {value:6d} ({perc:5.2f}%)")

    # --- Save histogram ---
    output_file.cd()
    h_summary.Write()

    # --- Draw and save canvas ---
    # --- Draw and save canvas ---
    c_summary = ROOT.TCanvas("c_summary", "Summary", 1000, 600)
    c_summary.SetBottomMargin(0.28)

    # Style
    h_summary.SetFillColor(ROOT.kAzure - 4)
    h_summary.SetLineColor(ROOT.kAzure - 4)
    h_summary.SetStats(0)

    # Force order + angled labels
    h_summary.GetXaxis().SetRangeUser(0.5, 9.5)
    h_summary.GetXaxis().SetCanExtend(False)
    h_summary.GetXaxis().LabelsOption("v")  # or "a" for 45°
    h_summary.GetXaxis().SetLabelSize(0.035)
    h_summary.GetXaxis().SetTitle("")

    # Draw as continuous histogram
    h_summary.Draw("hist text0")

    # Save as PDF and also inside ROOT file
    pdf_path = os.path.join(output_dir, "summary_counts.pdf")
    c_summary.SaveAs(pdf_path)
    c_summary.Write()

    print(f"\n💾 Saved summary canvas to: {pdf_path}")
    print("📂 Canvas and histogram also stored inside ROOT file.")
    print("✅ Summary histogram creation complete.\n")


# ============================================================
# Main orchestrator
# ============================================================
def extract_selected_info(input_file, output_dir, save_tree=False):
    load_geometry()

    # --- Define which masks to apply ---
    masks_to_apply = {
        'trig': False,
        'trig_count': True,
        'one_hough_zero_comb': True,
    }

    arrays, counters = load_and_select_events(input_file, masks_to_apply=masks_to_apply)

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

    counters = {"bad_meanx": 0, "no_TR_hit": 0, "same_z_tracks": 0}

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

    # --- Summary histogram and PDF ---
    make_summary_hist(arrays, root_file, output_dir)
    root_file.Close()

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
