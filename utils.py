import uproot
import awkward as ak
import numpy as np
import math
from geometry_utils import handle_two_cluster_track, track_hit_TR


def safe_first(arr):
    """Safely return the first element of an array, or NaN if empty."""
    return float(arr[0]) if len(arr) > 0 else float("nan")


def compute_trig_mask(tree):
    """
    Compute trigger mask from trig_conf_flag[6][:, 1].

    Returns:
      (trig_mask, count)
    """
    trig_conf_flag = tree["L2Event/trig_conf_flag[6]"].array(library="np")
    if getattr(trig_conf_flag, "ndim", None) == 2:
        trig_mask = trig_conf_flag[:, 1] == 1
        print("Using trigger mask (ndim=2)")
    else:
        trig_mask = np.array([evt[1] for evt in trig_conf_flag]) == 1
        print("Using trigger mask (object-array)")
    count = np.count_nonzero(trig_mask)
    return trig_mask, count


def compute_trig_count_mask(tree):
    """
    Compute trigger-count mask based on scintillator sums (TR1 & TR2).
    Condition: at least one pair sum > 50 in both TR1 and TR2.

    Returns:
      (trig_count_mask, count)
    """
    try:
        tr1_hg = tree[
            "L2Event/scint_raw_counts/scint_raw_counts.TR1_HG[5][2]"
        ].array(library="ak")
        tr2_hg = tree[
            "L2Event/scint_raw_counts/scint_raw_counts.TR2_HG[4][2]"
        ].array(library="ak")
    except Exception as e:
        tr1_key = "L2Event/scint_raw_counts/scint_raw_counts.TR1_HG[5][2]"
        tr2_key = "L2Event/scint_raw_counts/scint_raw_counts.TR2_HG[4][2]"
        if tr1_key in tree.keys() and tr2_key in tree.keys():
            tr1_hg = tree[tr1_key].array(library="ak")
            tr2_hg = tree[tr2_key].array(library="ak")
        else:
            raise RuntimeError(
                "Could not find TR1/TR2 scintillator branches in tree."
            ) from e

    tr1_sums = ak.sum(tr1_hg, axis=2)
    tr2_sums = ak.sum(tr2_hg, axis=2)

    tr1_counts_good = ak.any(tr1_sums > 50, axis=1)
    tr2_counts_good = ak.any(tr2_sums > 50, axis=1)

    trig_count_mask = tr1_counts_good & tr2_counts_good
    count = np.count_nonzero(trig_count_mask)
    return trig_count_mask, count


def compute_one_hough_zero_comb_mask(tree):
    """
    Compute one_hough_zero_comb mask based on multiplicity: x0==1 & x0_m2==0.

    Returns:
      (one_hough_zero_comb_mask, count)
    """
    x0_full = tree["L2Event/x0"].array(library="ak")
    x0_m2_full = tree["L2Event/x0_m2"].array(library="ak")
    one_hough_zero_comb_mask = (ak.num(x0_full) == 1) & (ak.num(x0_m2_full) == 0)
    count = np.count_nonzero(one_hough_zero_comb_mask)
    return one_hough_zero_comb_mask, count


def load_and_select_events(input_file, masks_to_apply=None):
    """
    Open ROOT file and apply selection masks.

    Parameters:
      - input_file: path to ROOT file
      - masks_to_apply: optional dict with boolean flags for which masks to apply.
                        Keys: 'trig', 'trig_count', 'one_hough_zero_comb' (default: all True)
                        Example: {'trig': True, 'trig_count': False, 'one_hough_zero_comb': True}

    Returns:
      (arrays, counters_dict)
      where counters_dict = {'total': N, 'trig': n_trig or None, 'trig_count': n_tc or None, 'one_hough_zero_comb': n_base or None, 'final': n_final}
    """
    if masks_to_apply is None:
        masks_to_apply = {'trig': True, 'trig_count': True, 'one_hough_zero_comb': True}

    print(f"🔍 Opening ROOT file: {input_file}")
    with uproot.open(input_file) as f:
        tree = f["L2"]

        total_events = tree.num_entries
        print(f"Total events in file: {total_events}")

        mask = np.ones(total_events, dtype=bool)
        counters = {'total': total_events, 'trig': None, 'trig_count': None, 'one_hough_zero_comb': None}

        # --- Trigger mask ---
        if masks_to_apply.get('trig', True):
            trig_mask, n_trig = compute_trig_mask(tree)
            counters['trig'] = n_trig
            print(
                f"Events after trig_mask: {n_trig} ({n_trig / total_events * 100:.2f}%)"
            )
            mask = mask & trig_mask

        # --- Trigger-count mask ---
        if masks_to_apply.get('trig_count', True):
            trig_count_mask, n_trig_count = compute_trig_count_mask(tree)
            counters['trig_count'] = n_trig_count
            print(
                f"Events after trig_count_mask: {n_trig_count} ({n_trig_count / total_events * 100:.2f}%)"
            )
            mask = mask & trig_count_mask

        # --- one_hough_zero_comb mask ---
        if masks_to_apply.get('one_hough_zero_comb', True):
            one_hough_zero_comb_mask, n_base = compute_one_hough_zero_comb_mask(tree)
            counters['one_hough_zero_comb'] = n_base
            print(
                f"Events after one_hough_zero_comb_mask (x0==1 & x0_m2==0): {n_base} ({n_base / total_events * 100:.2f}%)"
            )
            mask = mask & one_hough_zero_comb_mask

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
        n_final = len(arrays)
        counters['final'] = n_final

    # --- Summary printout ---
    print("\n✅ Event selection summary:")
    print(f"  Total events:                               {total_events}")
    if counters['trig'] is not None:
        print(f"  After trig_mask:                            {counters['trig']}")
    if counters['trig_count'] is not None:
        print(f"  After trig_count_mask:                      {counters['trig_count']}")
    if counters['one_hough_zero_comb'] is not None:
        print(f"  After one_hough_zero_comb_mask (x0==1 & x0_m2==0):         {counters['one_hough_zero_comb']}")
    print(f"  After all masks combined:                    {n_final}")
    print("--------------------------------------------------")

    return arrays, counters


def analyze_event(evt):
    """Compute key metrics, flags, and acceptance for one event.

    This function mirrors the behavior previously defined in the main script.
    It expects an event-like mapping with the same branch keys as returned
    by `tree.arrays([...], library='ak')`.
    """
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
        if ti not in (-1, -999)
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
        result = handle_two_cluster_track(
            cls_structs, math.radians(theta_val), math.radians(phi_val), dist_z=3.5
        )
        if result:
            missing_in_acc = result["missing_in_acceptance"]
            hit_tr = result["hit_TR"]
    else:
        hit_tr = track_hit_TR(
            x0_val, y0_val, math.radians(theta_val), math.radians(phi_val)
        )

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
