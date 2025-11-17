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
    """
    Open ROOT file, apply:
      - trigger mask (trig_conf_flag)
      - trigger-count mask (scintillator sums TR1 & TR2)
      - base_mask (x0/x0_m2 multiplicity)
    and return selected arrays and counters.
    """
    print(f"🔍 Opening ROOT file: {input_file}")
    with uproot.open(input_file) as f:
        tree = f["L2"]

        # --- Total number of events ---
        total_events = tree.num_entries
        print(f"Total events in file: {total_events}")

        # --- Trigger mask ---
        trig_conf_flag = tree["L2Event/trig_conf_flag[6]"].array(library="np")
        if getattr(trig_conf_flag, "ndim", None) == 2:
            trig_mask = trig_conf_flag[:, 1] == 1
            print("Using trigger mask (ndim=2)")
        else:
            trig_mask = np.array([evt[1] for evt in trig_conf_flag]) == 1
            print("Using trigger mask (object-array)")

        n_after_trig = np.count_nonzero(trig_mask)
        print(
            f"Events after trig_mask: {n_after_trig} ({n_after_trig / total_events * 100:.2f}%)"
        )

        # --- Trigger-count mask based on scintillator sums (TR1 & TR2) ---
        # Read arrays with awkward
        # Keys assumed: "L2Event/scint_raw_counts.scint_raw_counts.TR1_HG" and TR2_HG
        # Each event: TR1_HG -> (5,2), TR2_HG -> (4,2)
        try:
            tr1_hg = tree[
                "L2Event/scint_raw_counts/scint_raw_counts.TR1_HG[5][2]"
            ].array(library="ak")
            tr2_hg = tree[
                "L2Event/scint_raw_counts/scint_raw_counts.TR2_HG[4][2]"
            ].array(library="ak")
        except Exception as e:
            # If keys differ, try alternative naming (compatibility)
            tr1_key = "L2Event/scint_raw_counts/scint_raw_counts.TR1_HG[5][2]"
            tr2_key = "L2Event/scint_raw_counts/scint_raw_counts.TR2_HG[4][2]"
            if tr1_key in tree.keys() and tr2_key in tree.keys():
                tr1_hg = tree[tr1_key].array(library="ak")
                tr2_hg = tree[tr2_key].array(library="ak")
            else:
                raise RuntimeError(
                    "Could not find TR1/TR2 scintillator branches in tree."
                ) from e

        # Sum the pairs along last axis -> shape becomes (Nevents, 5) and (Nevents, 4)
        # If arrays are not exactly shaped (e.g. missing), ak.sum still works elementwise.
        tr1_sums = ak.sum(tr1_hg, axis=2)
        tr2_sums = ak.sum(tr2_hg, axis=2)

        # Condition: at least one pair sum > 50
        tr1_counts_good = ak.any(tr1_sums > 50, axis=1)
        tr2_counts_good = ak.any(tr2_sums > 50, axis=1)

        # Final trigger-count mask = AND (both TR1 and TR2 must have at least one pair > 50)
        trig_count_mask = tr1_counts_good & tr2_counts_good

        # Count events surviving trig + trig_count
        n_after_trig_count = np.count_nonzero(trig_mask & trig_count_mask)
        print(
            f"Events after trig + scintillator-count mask: {n_after_trig_count} ({n_after_trig_count / total_events * 100:.2f}%)"
        )

        # --- Base mask (multiplicity condition on full arrays) ---
        x0_full = tree["L2Event/x0"].array(library="ak")
        x0_m2_full = tree["L2Event/x0_m2"].array(library="ak")
        base_mask = (ak.num(x0_full) == 1) & (ak.num(x0_m2_full) == 0)
        n_after_base = np.count_nonzero(base_mask)
        print(
            f"Events after base_mask (x0==1 & x0_m2==0): {n_after_base} ({n_after_base / total_events * 100:.2f}%)"
        )

        # --- Combined mask (final selection) ---
        mask = trig_mask & trig_count_mask & base_mask
        n_after_combined = np.count_nonzero(mask)
        print(
            f"Events after all masks (trig & scint counts & base): {n_after_combined} ({n_after_combined / total_events * 100:.2f}%)"
        )

        # --- Branch loading (only for selected events) ---
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
            "L2Event/cls_track_idx",
        ]
        arrays = tree.arrays(branches, library="ak")[mask]

    # --- Summary printout ---
    print("\n✅ Event selection summary:")
    print(f"  Total events:                               {total_events}")
    print(f"  After trig_mask:                            {n_after_trig}")
    print(f"  After trig+scintillator-count mask:         {n_after_trig_count}")
    print(f"  After base_mask (x0==1 & x0_m2==0):         {n_after_base}")
    print(f"  After all masks combined:                    {len(arrays)}")
    print("--------------------------------------------------")

    # Return arrays and counters (in the order used downstream)
    return arrays, n_after_trig, n_after_trig_count, n_after_base


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
            "track_idx": int(ti),
        }
        for mx, my, mz, sz, rx, ry, ti in zip(
            evt["L2Event/cls_mean_x"],
            evt["L2Event/cls_mean_y"],
            evt["L2Event/cls_mean_z"],
            evt["L2Event/cls_size"],
            evt["L2Event/cls_res_x"],
            evt["L2Event/cls_res_y"],
            evt["L2Event/cls_track_idx"],
        )
        if ti not in (-1, -999)  # Filter: exclude clusters without valid track-idx
    ]

    if len(cls_structs) == 0:
        print("⚠️ Warning: event with all clusters invalid (track_idx = -1 or -999)")

    n_cls = len(cls_structs)
    z_values = [c["mean_z"] for c in cls_structs]
    same_z_count = len(z_values) - len(set(z_values))

    has_bad_meanx = any(c["mean_x"] == -999 for c in cls_structs)
    bad_ncls = n_cls > 3

    hit_tr = False
    missing_in_acc = False

    if n_cls == 2:
        # Compute acceptance and TR hit for 2-cluster tracks
        result = handle_two_cluster_track(
            cls_structs, math.radians(theta_val), math.radians(phi_val), dist_z=3.5
        )
        if result:
            missing_in_acc = result["missing_in_acceptance"]
            hit_tr = result["hit_TR"]
    else:
        # For other tracks, only check TR hit
        hit_tr = track_hit_TR(
            x0_val, y0_val, math.radians(theta_val), math.radians(phi_val)
        )

    # Replace "theta > 72" issue with "not track_hit_TR"
    bad_trhit = not hit_tr

    issues = {
        "mean_x": has_bad_meanx,
        "n_cls": bad_ncls,
        "no_TR_hit": bad_trhit,
        "same_z_count": same_z_count,
    }

    return (
        x0_val,
        y0_val,
        theta_val,
        phi_val,
        cls_structs,
        n_cls,
        issues,
        hit_tr,
        missing_in_acc,
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
    arrays, n_after_trig, n_after_cls, n_after_base = load_and_select_events(input_file)

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
