import ROOT
import glob
import os
import sys

# Cerca file con suffisso problematico "_histos_histos.root" e segnala se presenti
_bad_files = glob.glob("*_histos_histos.root")
if _bad_files:
    print(
        "Warning: found files ending with '_histos_histos.root' (check for duplicated suffix):"
    )
    for _bf in _bad_files:
        print(f"  - {_bf}")
    # Rimuovi i file trovati
    for _bf in _bad_files:
        try:
            os.remove(_bf)
            print(f"Removed {_bf}")
        except Exception as e:
            print(f"Failed to remove {_bf}: {e}")

ROOT.gStyle.SetOptStat(0)

# Get all files matching the pattern
files = glob.glob("*_R*d*_histos.root")  # Changed pattern here
if not files:
    print(
        "Error: No files found matching pattern '*_R*d*_histos.root'"
    )  # Updated error message
    print("Current directory:", os.getcwd())
    print("Files in current directory:", os.listdir())
    sys.exit(1)

n_files = len(files)
print(f"Found {n_files} files:")
for f in files:
    print(f"  - {f}")

# Set up a nice palette (kBird is a good choice for visibility)
ROOT.gStyle.SetPalette(ROOT.kBird)
palette = ROOT.TColor.GetPalette()
palette_size = palette.GetSize()

out_file = ROOT.TFile("track_multiplicity_comparison.root", "RECREATE")

# Create canvas
cMultComp = ROOT.TCanvas("cMultComp", "cMultComp", 800, 600)
cMultComp.SetLogy()

# Create frame with specific binning
frame = cMultComp.DrawFrame(
    -0.5, 0.1, 20.5, 1e5
)  # y-range from 0.1 to 1e5 due to log scale
frame.SetTitle(";Track multiplicity;Entries")

# Create legend with 3 columns
legend = ROOT.TLegend(0.44, 0.65, 0.86, 0.81)
legend.SetNColumns(3)

# Get and store first histogram (h_trk_mult)
first_file = ROOT.TFile(files[0], "READ")
h_trk_mult = first_file.Get("h_trk_mult")
if not h_trk_mult:
    print(f"Error: histogram 'h_trk_mult' not found in {files[0]}")
    print("Available histograms in the file:")
    for key in first_file.GetListOfKeys():
        print(f"  - {key.GetName()}")
    sys.exit(1)
h_trk_mult.SetDirectory(0)
h_trk_mult.SetLineColor(ROOT.kRed)
h_trk_mult.SetLineWidth(2)

# Loop over files to collect all histograms
hist_info = []
for fname in files:
    f = ROOT.TFile(fname, "READ")
    h = f.Get("h_trk_mult_m2")
    if not h:
        print(f"Error: histogram 'h_trk_mult_m2' not found in {fname}")
        print("Available histograms in the file:")
        for key in f.GetListOfKeys():
            print(f"  - {key.GetName()}")
        f.Close()
        continue
    h.SetDirectory(0)

    # Extract R and d values from filename
    R = int(fname.split("_R")[1].split("d")[0])
    d = int(fname.split("d")[1].split("_histos")[0])

    # Store histogram and its properties
    hist_info.append(
        {
            "hist": h,
            "R": R,
            "d": d,
            "value": float(f"{R}.{d}"),  # Convert R.d to float for sorting
        }
    )
    f.Close()

# Sort histograms by R.d value
hist_info.sort(key=lambda x: x["value"])

# Draw reference histogram first
h_trk_mult.Draw("SAME")
legend.AddEntry(h_trk_mult, "Hough", "l")

# Draw sorted histograms
for i, info in enumerate(hist_info):
    h = info["hist"]
    R = info["R"]
    d = info["d"]

    # Set properties
    h.SetName(f"h_trk_mult_m2_R{R}d{d}")
    color_index = palette[int(i * (palette_size - 1) / (len(hist_info) - 1))]
    h.SetLineColor(color_index)
    h.SetLineWidth(2)
    h.SetMarkerColor(color_index)

    # Draw histogram
    h.Draw("SAME")

    # Add to legend
    legend.AddEntry(h, f"R = {R}.{d} mm", "l")

    # Save to output file
    out_file.cd()
    h.Write()

# Draw legend
legend.Draw()

# Save canvas as PDF and ROOT file
cMultComp.SaveAs("track_multiplicity_comparison.pdf")
out_file.cd()
cMultComp.Write("canvas")
h_trk_mult.Write()  # save reference histogram

# Create new canvas for integral comparison
cIntComp = ROOT.TCanvas("cIntComp", "cIntComp", 800, 600)
out_file.cd()  # Make sure we're in the output file directory

# Create histogram for integrals
h_int = ROOT.TH1F(
    "h_int", "Integral comparison;Number of tracks;Number of events", 5, 0.5, 5.5
)
h_int.SetDirectory(out_file)
h_int.SetStats(0)

# Define bin ranges for integration
ranges = [(2, 2), (2, 3), (2, 4), (2, 5), (2, 6)]
labels = ["1 track", "1-2 tracks", "1-3 tracks", "1-4 tracks", "1-5 tracks"]

# Calculate integrals for reference histogram (absolute counts)
for i, (low, high) in enumerate(ranges, 1):
    integral = h_trk_mult.Integral(low, high)
    h_int.SetBinContent(i, integral)

h_int.SetLineColor(ROOT.kRed)  # Changed from kBlack to kRed
h_int.SetLineWidth(2)
h_int.Draw()

# Create legend with 3 columns
legend_int = ROOT.TLegend(0.39, 0.65, 0.86, 0.81)
legend_int.SetNColumns(3)
legend_int.AddEntry(h_int, "Hough", "l")

# List to keep references to created histograms
integral_histos = []

# Create histograms for each R.d value
for i, info in enumerate(hist_info):
    h = info["hist"]
    R = info["R"]
    d = info["d"]

    h_int_rd = ROOT.TH1F(
        f"h_int_R{R}d{d}",
        h_int.GetTitle(),
        h_int.GetNbinsX(),
        h_int.GetXaxis().GetXmin(),
        h_int.GetXaxis().GetXmax(),
    )
    h_int_rd.SetDirectory(out_file)

    # Calculate integrals (absolute counts)
    for j, (low, high) in enumerate(ranges, 1):
        integral = h.Integral(low, high)
        h_int_rd.SetBinContent(j, integral)

    # Set style
    color_index = palette[int(i * (palette_size - 1) / (len(hist_info) - 1))]
    h_int_rd.SetLineColor(color_index)
    h_int_rd.SetLineWidth(2)

    h_int_rd.Draw("SAME")
    legend_int.AddEntry(h_int_rd, f"R = {R}.{d} mm", "l")
    integral_histos.append(h_int_rd)  # Keep reference

# Set bin labels
for i, label in enumerate(labels, 1):
    h_int.GetXaxis().SetBinLabel(i, label)

h_int.GetYaxis().SetRangeUser(0.0, h_int.GetMaximum() * 1.1)  # Adjust y-axis range

legend_int.Draw()

# Save canvas
cIntComp.Update()
cIntComp.SaveAs("track_multiplicity_integral_comparison.pdf")
out_file.cd()
cIntComp.Write("canvas_integral")

# Clean up
integral_histos.clear()
