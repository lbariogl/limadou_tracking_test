import os
import numpy as np
import uproot
import ROOT
import glob
import sys
import argparse
import awkward as ak

# === CONFIGURAZIONE ===
input_directory = "/home/lbariogl/limadou/test/data/"  # Directory containing ROOT files
output_directory = "/home/lbariogl/limadou/test/output/"  # Directory to save output histograms and plots
tree_name = "L2"

# coppie, range e label X (ROOT style)
branch_pairs = {
    ("x0", "x0_m2"): {"range": (-90, 90), "xlabel": "x_{0} (mm)"},
    ("y0", "y0_m2"): {"range": (-90, 90), "xlabel": "y_{0} (mm)"},
    ("phi", "phi_m2"): {"range": (-180, 180), "xlabel": "#phi (#circ)"},
    ("theta", "theta_m2"): {
        "range": (0, 90),
        "xlabel": "#theta (#circ)",
    },
}

nbins = 80  # bins per histogram

# === GET LIST OF ROOT FILES ===
root_files = glob.glob(os.path.join(input_directory, "*.root"))
if not root_files:
    raise FileNotFoundError(f"No ROOT files found in: {input_directory}")

print(f"Found {len(root_files)} ROOT files to process")


def return_output_name(output_dir, input_file, suffix):
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    return os.path.join(output_dir, f"{base_name}{suffix}")


# Keep the helper functions
def make_hist_from_jagged(name, arr_jagged, nbins, xmin, xmax):
    h = ROOT.TH1F(name, name, nbins, xmin, xmax)
    for evt in arr_jagged:
        for val in evt:
            h.Fill(float(val))
    return h


# Function to process a single file
def process_root_file(input_file_name):

    print(f"\nProcessing: {input_file_name}")

    output_file_name = return_output_name(
        output_directory, input_file_name, "_distributions.root"
    )

    out_file = ROOT.TFile(output_file_name, "RECREATE")

    # === READ TREE ===
    input_file = uproot.open(input_file_name)
    if tree_name not in input_file:
        print(f"⚠️ TTree '{tree_name}' not found in {input_file_name}, skipping...")
        return

    tree = input_file[tree_name]
    total_events = tree.num_entries
    print(f"✅ Read TTree '{tree_name}' with {total_events} events.")

    # === TRIGGER MASK ===
    trig_conf_flag = tree["L2Event/trig_conf_flag[6]"].array(library="np")
    if getattr(trig_conf_flag, "ndim", None) == 2:
        trig_mask = trig_conf_flag[:, 1] == 1
        print("Using trigger mask (ndim=2)")
    else:
        trig_mask = np.array([evt[1] for evt in trig_conf_flag]) == 1
        print("Using trigger mask (object-array)")

    n_after_trig = int(np.sum(trig_mask))
    print(
        f"Events after trig_mask: {n_after_trig} ({n_after_trig / total_events * 100:.2f}%)"
    )

    if n_after_trig == 0:
        print("❌ No events passed trigger mask — skipping file.")
        return

    # === CLUSTER MASK: events with at least one cluster (non-empty cls_mean_x) ===
    cls_mean_x = tree["L2Event/cls_mean_x"].array(library="ak")
    cls_mask = ak.num(cls_mean_x) > 0

    n_after_cls = int(np.sum(cls_mask & trig_mask))
    print(
        f"Events after cls_mask (non-empty cls_mean_x): {n_after_cls} ({n_after_cls / total_events * 100:.2f}%)"
    )

    if n_after_cls == 0:
        print("❌ No events have non-empty clusters — skipping file.")
        return

    # === FINAL MASK COMBINATION ===
    mask = trig_mask & cls_mask

    nsel = int(np.sum(mask))
    print(f"Selected {nsel} events (trigger + cluster mask)")

    if nsel == 0:
        print("❌ No events after combined masks — skipping file.")
        return

    # === Distribuzione del numero di clusters per evento ===
    # Get cluster_x0 array for selected events and calculate cluster_multiplicities
    cluster_x0_array = tree["L2Event/cls_size"].array(library="np")[mask]
    cluster_multiplicities = [len(evt) for evt in cluster_x0_array]

    # Create and fill multiplicity histogram
    cClsMult = ROOT.TCanvas("cClsMult", "cClsMult", 900, 700)
    h_cls_mult = ROOT.TH1F(
        "h_cls_mult",
        "; clusters per event; Entries",
        max(cluster_multiplicities) - min(cluster_multiplicities) + 1,
        min(cluster_multiplicities) - 0.5,
        max(cluster_multiplicities) + 0.5,
    )
    for mult in cluster_multiplicities:
        h_cls_mult.Fill(mult)

    # Print statistics
    print(f"Cluster multiplicity stats:")
    print(f"  Mean: {np.mean(cluster_multiplicities):.2f}")
    print(f"  Std Dev: {np.std(cluster_multiplicities):.2f}")
    print(f"  Min: {min(cluster_multiplicities)}")
    print(f"  Max: {max(cluster_multiplicities)}")

    cClsMult.cd()
    h_cls_mult.Draw("HIST")
    pave_cls = ROOT.TPaveText(0.65, 0.80, 0.88, 0.88, "brNDC")
    pave_cls.AddText(f"Selected events: {nsel}")
    pave_cls.AddText(f"Events with no clusters: {h_cls_mult.GetBinContent(1)}")
    pave_cls.SetFillColor(0)
    pave_cls.Draw()
    out_file.cd()
    h_cls_mult.Write()  # Save cluster multiplicity histogram

    cluster_multiplicity_pdf = return_output_name(
        output_directory, input_file_name, "_cluster_multiplicity.pdf"
    )

    cClsMult.Print(cluster_multiplicity_pdf)
    print(f"✅ Cluster multiplicity plot saved in '{cluster_multiplicity_pdf}'")

    # === Distribuzione del numero di tracce per evento ===

    # Get x0 array for selected events and calculate track_multiplicities
    x0_array = tree["L2Event/x0"].array(library="np")[mask]
    track_multiplicities = [len(evt) for evt in x0_array]

    # Create and fill multiplicity histogram
    cTrackMult = ROOT.TCanvas("cTrackMult", "cTrackMult", 900, 700)
    h_trk_mult = ROOT.TH1F(
        "h_trk_mult",
        "; tracks per event; Entries",
        max(track_multiplicities) - min(track_multiplicities) + 1,
        min(track_multiplicities) - 0.5,
        max(track_multiplicities) + 0.5,
    )
    h_trk_mult.SetLineColor(ROOT.kBlue)
    h_trk_mult.SetLineWidth(2)

    for mult in track_multiplicities:
        h_trk_mult.Fill(mult)

    # Print statistics
    print(f"Track multiplicity stats:")
    print(f"  Mean: {np.mean(track_multiplicities):.2f}")
    print(f"  Std Dev: {np.std(track_multiplicities):.2f}")
    print(f"  Min: {min(track_multiplicities)}")
    print(f"  Max: {max(track_multiplicities)}")

    cTrackMult.cd()
    h_trk_mult.Draw("HIST")
    pave_trk = ROOT.TPaveText(0.65, 0.80, 0.88, 0.88, "brNDC")
    pave_trk.AddText(f"Selected events: {nsel}")
    pave_trk.AddText(f"Events with no tracks: {h_trk_mult.GetBinContent(1)}")
    pave_trk.AddText(f"Events with tracks: {nsel - h_trk_mult.GetBinContent(1)}")
    pave_trk.SetFillColor(0)
    pave_trk.Draw()
    out_file.cd()
    h_trk_mult.Write()  # Save track multiplicity histogram

    track_multiplicity_pdf = return_output_name(
        output_directory, input_file_name, "_track_multiplicity.pdf"
    )
    cTrackMult.Print(track_multiplicity_pdf)
    print(f"✅ Track multiplicity plot saved in '{track_multiplicity_pdf}'")

    # === Distribuzione del numero di tracce per evento (usando x0_m2) ===
    x0_m2_array = tree["L2Event/x0_m2"].array(library="np")[mask]
    track_multiplicities_m2 = [len(evt) for evt in x0_m2_array]

    # Create and fill multiplicity histogram
    cTrackMultM2 = ROOT.TCanvas("cTrackMultM2", "cTrackMultM2", 900, 700)
    h_trk_mult_m2 = ROOT.TH1F(
        "h_trk_mult_m2",
        "; tracks per event (M2); Entries",
        max(track_multiplicities_m2) - min(track_multiplicities_m2) + 1,
        min(track_multiplicities_m2) - 0.5,
        max(track_multiplicities_m2) + 0.5,
    )
    h_trk_mult_m2.SetLineColor(ROOT.kRed)
    h_trk_mult_m2.SetLineWidth(2)

    for mult in track_multiplicities_m2:
        h_trk_mult_m2.Fill(mult)

    # Print statistics
    print(f"Track multiplicity M2 stats:")
    print(f"  Mean: {np.mean(track_multiplicities_m2):.2f}")
    print(f"  Std Dev: {np.std(track_multiplicities_m2):.2f}")
    print(f"  Min: {min(track_multiplicities_m2)}")
    print(f"  Max: {max(track_multiplicities_m2)}")

    cTrackMultM2.cd()
    h_trk_mult_m2.Draw("HIST")
    pave_trk_m2 = ROOT.TPaveText(0.65, 0.80, 0.88, 0.88, "brNDC")
    pave_trk_m2.AddText(f"Selected events: {nsel}")
    pave_trk_m2.AddText(f"Events with no tracks: {h_trk_mult_m2.GetBinContent(1)}")
    pave_trk_m2.AddText(f"Events with tracks: {nsel - h_trk_mult_m2.GetBinContent(1)}")
    pave_trk_m2.SetFillColor(0)
    pave_trk_m2.Draw()
    out_file.cd()
    h_trk_mult_m2.Write()  # Save track multiplicity M2 histogram

    track_multiplicity_m2_pdf = return_output_name(
        output_directory, input_file_name, "_track_multiplicity_m2.pdf"
    )
    cTrackMultM2.Print(track_multiplicity_m2_pdf)
    print(f"✅ Track multiplicity M2 plot saved in '{track_multiplicity_m2_pdf}'")

    # === 2D histograms: track multiplicity vs cluster multiplicity ===

    # Standard tracks vs clusters
    c2D = ROOT.TCanvas("c2D", "c2D", 900, 700)
    h2D_trk_cls = ROOT.TH2F(
        "h2D_trk_cls",
        "Track multiplicity vs Cluster multiplicity;Clusters per event;Tracks per event",
        max(cluster_multiplicities) - min(cluster_multiplicities) + 1,
        min(cluster_multiplicities) - 0.5,
        max(cluster_multiplicities) + 0.5,
        max(track_multiplicities) - min(track_multiplicities) + 1,
        min(track_multiplicities) - 0.5,
        max(track_multiplicities) + 0.5,
    )

    for cls_mult, trk_mult in zip(cluster_multiplicities, track_multiplicities):
        h2D_trk_cls.Fill(cls_mult, trk_mult)

    c2D.cd()
    h2D_trk_cls.Draw("COLZ")
    out_file.cd()
    h2D_trk_cls.Write()
    c2D.Print(
        return_output_name(output_directory, input_file_name, "_trk_vs_cls_2D.pdf")
    )

    # M2 tracks vs clusters
    c2D_m2 = ROOT.TCanvas("c2D_m2", "c2D_m2", 900, 700)
    h2D_trk_m2_cls = ROOT.TH2F(
        "h2D_trk_m2_cls",
        "Track multiplicity (M2) vs Cluster multiplicity;Clusters per event;Tracks per event (M2)",
        max(cluster_multiplicities) - min(cluster_multiplicities) + 1,
        min(cluster_multiplicities) - 0.5,
        max(cluster_multiplicities) + 0.5,
        max(track_multiplicities_m2) - min(track_multiplicities_m2) + 1,
        min(track_multiplicities_m2) - 0.5,
        max(track_multiplicities_m2) + 0.5,
    )

    for cls_mult, trk_mult in zip(cluster_multiplicities, track_multiplicities_m2):
        h2D_trk_m2_cls.Fill(cls_mult, trk_mult)

    c2D_m2.cd()
    h2D_trk_m2_cls.Draw("COLZ")
    out_file.cd()
    h2D_trk_m2_cls.Write()
    c2D_m2.Print(
        return_output_name(output_directory, input_file_name, "_trk_m2_vs_cls_2D.pdf")
    )

    # === INIZIALIZZAZIONE ROOT ===
    c = ROOT.TCanvas("c", "c", 900, 700)

    # === CREAZIONE ISTOGRAMMI ===
    histos = {}
    counts = {}
    for (b1, b2), info in branch_pairs.items():
        xmin, xmax = info["range"]

        for b in [b1, b2]:
            full_name = f"L2Event/{b}"
            if full_name in tree.keys():
                arr_jagged = tree[full_name].array(library="np")[
                    mask
                ]  # Apply mask here
                histos[b] = make_hist_from_jagged(b, arr_jagged, nbins, xmin, xmax)
                try:
                    counts[b] = int(sum(len(evt) for evt in arr_jagged))
                except Exception:
                    counts[b] = int(sum(len(evt) for evt in list(arr_jagged)))
            else:
                print(f"⚠️ Branch {full_name} not found. Skipped.")

    # Verifica coerenza numero totale di tracce
    n_tracks_from_mult = sum(
        i * h_trk_mult.GetBinContent(i + 1) for i in range(int(h_trk_mult.GetNbinsX()))
    )
    print("Total tracks (from multiplicity hist):", n_tracks_from_mult)
    print("Total tracks (from x0 histogram):", counts["x0"])

    # === DISEGNO E SALVATAGGIO ===
    c.Print(
        return_output_name(output_directory, input_file_name, "_distributions.pdf")
        + "["
    )  # apre multipagina

    for (b1, b2), info in branch_pairs.items():
        xlabel = info["xlabel"]
        h1 = histos.get(b1, None)
        h2 = histos.get(b2, None)

        if h1 is None and h2 is None:
            continue

        # Stile
        if h1:
            h1.SetLineColor(ROOT.kBlue)
            h1.SetLineWidth(2)
        if h2:
            h2.SetLineColor(ROOT.kRed)
            h2.SetLineWidth(2)

        # Normalizzazione
        h1.Scale(1.0 / h1.Integral(), "width") if h1 else None
        h2.Scale(1.0 / h2.Integral(), "width") if h2 else None

        # Massimo Y
        max_y = 0
        if h1:
            max_y = max(max_y, h1.GetMaximum())
        if h2:
            max_y = max(max_y, h2.GetMaximum())
        if max_y <= 0:
            max_y = 1.0
        if h1:
            h1.SetMaximum(max_y * 1.2)

        # Titolo
        base_hist = h1 if h1 else h2
        if base_hist:
            base_hist.SetTitle(f"Confronto: {b1} vs {b2};{xlabel};Normalised counts")

        # Disegno
        c.cd()
        if base_hist:
            base_hist.Draw("HIST")
        if h1 and h2:
            if base_hist is h1:
                h2.Draw("HIST SAME")
            else:
                h1.Draw("HIST SAME")

        # Legenda
        legend = ROOT.TLegend(0.65, 0.70, 0.88, 0.88)
        if h1:
            label1 = f"{b1} (N={counts.get(b1, 0)})"
            legend.AddEntry(h1, label1, "l")
        if h2:
            label2 = f"{b2} (N={counts.get(b2, 0)})"
            legend.AddEntry(h2, label2, "l")
        legend.SetBorderSize(0)
        legend.Draw()

        c.Print(
            return_output_name(output_directory, input_file_name, "_distributions.pdf")
        )

        # Save histograms to ROOT file
        if h1:
            out_file.cd()
            h1.Write()
        if h2:
            out_file.cd()
            h2.Write()

    c.Print(
        return_output_name(output_directory, input_file_name, "_distributions.pdf")
        + "]"
    )

    print(
        f"✅ Plots saved in '{return_output_name(output_directory, input_file_name, '_distributions.pdf')}'"
    )
    print(f"✅ Histograms saved in ROOT file: {output_file_name}")

    # === Energy deposit distributions ===
    # First track
    edep1_array = tree["L2Event/Edep_stdAlone/Edep_stdAlone.tr1_tot_HG"].array(
        library="np"
    )[mask]
    h_edep1 = ROOT.TH1F(
        "h_edep1",
        "Energy deposit - Track 1;E_{dep} (ADC);Entries",
        1000,
        0,
        1000,  # adjust binning as needed
    )
    for evt in edep1_array:
        if isinstance(evt, (np.ndarray, list)):
            for val in evt:
                h_edep1.Fill(float(val))
        else:
            h_edep1.Fill(float(evt))

    # Second track
    edep2_array = tree["L2Event/Edep_stdAlone/Edep_stdAlone.tr2_tot_HG"].array(
        library="np"
    )[mask]
    h_edep2 = ROOT.TH1F(
        "h_edep2",
        "Energy deposit - Track 2;E_{dep} (ADC);Entries",
        1000,
        0,
        1000,  # adjust binning as needed
    )
    for evt in edep2_array:
        if isinstance(evt, (np.ndarray, list)):
            for val in evt:
                h_edep2.Fill(float(val))
        else:
            h_edep2.Fill(float(evt))

    # Save histograms
    out_file.cd()
    h_edep1.Write()
    h_edep2.Write()


if __name__ == "__main__":
    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0)

    parser = argparse.ArgumentParser(
        description="Process ROOT files from L2 tree to produce histograms and plots."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all ROOT files in the input directory",
    )
    parser.add_argument(
        "--file", type=str, help="Process a single ROOT file (full path)"
    )

    args = parser.parse_args()

    if args.all and args.file:
        print("⚠️ Please specify only one option: either --all or --file.")
        sys.exit(1)

    if not args.all and not args.file:
        print("❗ Please specify an option: --all or --file <path>")
        sys.exit(1)

    # Se l’utente sceglie --all
    if args.all:
        root_files = glob.glob(os.path.join(input_directory, "*.root"))
        if not root_files:
            raise FileNotFoundError(f"No ROOT files found in: {input_directory}")
        print(f"Found {len(root_files)} ROOT files to process")
        for root_file in root_files:
            process_root_file(root_file)

    # Se l’utente sceglie un singolo file
    elif args.file:
        if not os.path.exists(args.file):
            raise FileNotFoundError(f"File not found: {args.file}")
        process_root_file(args.file)
