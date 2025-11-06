#!/usr/bin/env python3
import ROOT
import glob
import os
import sys
import argparse
import re


def compare_track_multiplicity(input_dir, output_dir):
    """Compare track multiplicity histograms across ROOT files."""
    print(f"🔍 Input directory: {input_dir}")
    print(f"📤 Output directory: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)
    os.chdir(input_dir)

    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetPalette(ROOT.kBird)

    # Find input files
    files = glob.glob("*_R*d*_distributions.root")
    if not files:
        print("❌ Error: No files found matching '*_R*d*_distributions.root'")
        print("Current directory:", os.getcwd())
        return

    print(f"✅ Found {len(files)} files:")
    for f in files:
        print(f"  - {f}")

    palette = ROOT.TColor.GetPalette()
    palette_size = palette.GetSize()

    # Output ROOT file
    out_file_path = os.path.join(output_dir, "track_multiplicity_comparison.root")
    out_file = ROOT.TFile(out_file_path, "RECREATE")

    # === 1. MULTIPLICITY COMPARISON ===
    cMultComp = ROOT.TCanvas("cMultComp", "cMultComp", 800, 600)
    cMultComp.SetLogy()

    frame = cMultComp.DrawFrame(-0.5, 0.1, 20.5, 1e5)
    frame.SetTitle(";Track multiplicity;Entries")

    legend = ROOT.TLegend(0.44, 0.65, 0.86, 0.81)
    legend.SetNColumns(3)

    # Reference histogram
    first_file = ROOT.TFile(files[0], "READ")
    h_trk_mult = first_file.Get("h_trk_mult")
    if not h_trk_mult:
        print(f"❌ Error: histogram 'h_trk_mult' not found in {files[0]}")
        sys.exit(1)
    h_trk_mult.SetDirectory(0)
    h_trk_mult.SetLineColor(ROOT.kRed)
    h_trk_mult.SetLineWidth(2)
    h_trk_mult.Draw("SAME")
    legend.AddEntry(h_trk_mult, "Hough", "l")

    # Loop over all histograms
    hist_info = []
    for fname in files:
        f = ROOT.TFile(fname, "READ")
        h = f.Get("h_trk_mult_m2")
        if not h:
            print(f"⚠️ Warning: histogram 'h_trk_mult_m2' not found in {fname}")
            f.Close()
            continue

        h.SetDirectory(0)
        match = re.search(r"R(\d+)d(\d+)", fname)
        if not match:
            print(f"⚠️ Warning: could not parse region/detector from filename {fname}")
            continue
        r, d = map(int, match.groups())
        hist_info.append({"hist": h, "r": r, "d": d, "value": float(f"{r}.{d}")})
        f.Close()

    hist_info.sort(key=lambda x: x["value"])

    for i, info in enumerate(hist_info):
        h = info["hist"]
        r, d = info["r"], info["d"]
        h.SetName(f"h_trk_mult_m2_R{r}d{d}")
        color_index = palette[int(i * (palette_size - 1) / (len(hist_info) - 1))]
        h.SetLineColor(color_index)
        h.SetLineWidth(2)
        h.SetMarkerColor(color_index)
        h.Draw("SAME")
        legend.AddEntry(h, f"R = {r}.{d} mm", "l")
        out_file.cd()
        h.Write()

    legend.Draw()

    pdf_path = os.path.join(output_dir, "track_multiplicity_comparison.pdf")
    cMultComp.SaveAs(pdf_path)
    out_file.cd()
    cMultComp.Write("canvas")
    h_trk_mult.Write()
    print(f"✅ Saved track multiplicity comparison: {pdf_path}")

    # === 2. INTEGRAL COMPARISON ===
    cIntComp = ROOT.TCanvas("cIntComp", "cIntComp", 800, 600)
    out_file.cd()

    h_int = ROOT.TH1F(
        "h_int", "Integral comparison;Number of tracks;Number of events", 5, 0.5, 5.5
    )
    h_int.SetDirectory(out_file)
    h_int.SetStats(0)

    ranges = [(2, 2), (2, 3), (2, 4), (2, 5), (2, 6)]
    labels = ["1 track", "1-2 tracks", "1-3 tracks", "1-4 tracks", "1-5 tracks"]

    for i, (low, high) in enumerate(ranges, 1):
        h_int.SetBinContent(i, h_trk_mult.Integral(low, high))

    h_int.SetLineColor(ROOT.kRed)
    h_int.SetLineWidth(2)
    h_int.Draw()

    legend_int = ROOT.TLegend(0.39, 0.65, 0.86, 0.81)
    legend_int.SetNColumns(3)
    legend_int.AddEntry(h_int, "Hough", "l")

    integral_histos = []
    for i, info in enumerate(hist_info):
        h = info["hist"]
        r, d = info["r"], info["d"]
        h_int_rd = ROOT.TH1F(
            f"h_int_R{r}d{d}",
            h_int.GetTitle(),
            h_int.GetNbinsX(),
            h_int.GetXaxis().GetXmin(),
            h_int.GetXaxis().GetXmax(),
        )
        h_int_rd.SetDirectory(out_file)

        for j, (low, high) in enumerate(ranges, 1):
            h_int_rd.SetBinContent(j, h.Integral(low, high))

        color_index = palette[int(i * (palette_size - 1) / (len(hist_info) - 1))]
        h_int_rd.SetLineColor(color_index)
        h_int_rd.SetLineWidth(2)
        h_int_rd.Draw("SAME")
        legend_int.AddEntry(h_int_rd, f"R = {r}.{d} mm", "l")
        integral_histos.append(h_int_rd)

    for i, label in enumerate(labels, 1):
        h_int.GetXaxis().SetBinLabel(i, label)
    h_int.GetYaxis().SetRangeUser(0.0, h_int.GetMaximum() * 1.1)
    legend_int.Draw()

    pdf_int_path = os.path.join(
        output_dir, "track_multiplicity_integral_comparison.pdf"
    )
    cIntComp.SaveAs(pdf_int_path)
    out_file.cd()
    cIntComp.Write("canvas_integral")
    print(f"✅ Saved integral comparison: {pdf_int_path}")

    integral_histos.clear()
    out_file.Close()
    print(f"🏁 Output ROOT file saved in: {out_file_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare track multiplicity histograms across ROOT files."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input directory containing *_R*d*_distributions.root files",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output directory where results will be stored",
    )

    args = parser.parse_args()

    input_dir = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output)

    compare_track_multiplicity(input_dir, output_dir)


if __name__ == "__main__":
    main()
