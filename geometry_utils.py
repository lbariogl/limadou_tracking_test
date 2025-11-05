import ROOT
import math


# -------------------- Load Geometry -------------------- #
def load_geometry(header_path="geometry/Geometry.hh"):
    """Load the Geometry.hh constants into the ROOT interpreter."""
    ROOT.gInterpreter.Declare(f'#include "{header_path}"')
    print(f"✅ Geometry constants loaded from {header_path}")


# -------------------- Track hits TR -------------------- #
def track_hit_TR(x1, y1, theta, phi, hits_TR1andTR2=True):
    """Check whether the track hits both TR1 and TR2 based on geometry constants."""
    if not hits_TR1andTR2:
        return True

    G = ROOT.Geometry

    xTR1t = x1 - (G.StaveZ[1] - G.TR1Thickness / 2) * math.tan(theta) * math.cos(phi)
    yTR1t = y1 - (G.StaveZ[1] - G.TR1Thickness / 2) * math.tan(theta) * math.sin(phi)
    xTR2b = x1 + (-G.StaveZ[1] + G.TR2CenterZ - G.TR2Thickness / 2) * math.tan(
        theta
    ) * math.cos(phi)
    yTR2b = y1 + (-G.StaveZ[1] + G.TR2CenterZ - G.TR2Thickness / 2) * math.tan(
        theta
    ) * math.sin(phi)

    cond_TR1 = (xTR1t < G.TR1Size[0] / 2 and xTR1t > -G.TR1Size[0] / 2) and (
        (
            (yTR1t < (2.5 * G.TR1Size[1] + 2 * G.TR1GapY))
            and (yTR1t > (1.5 * G.TR1Size[1] + 2 * G.TR1GapY))
        )
        or (
            (yTR1t < (1.5 * G.TR1Size[1] + 1 * G.TR1GapY))
            and (yTR1t > (0.5 * G.TR1Size[1] + 1 * G.TR1GapY))
        )
        or (
            (yTR1t < (0.5 * G.TR1Size[1] + 0 * G.TR1GapY))
            and (yTR1t > -(0.5 * G.TR1Size[1] + 0 * G.TR1GapY))
        )
        or (
            (yTR1t < -(0.5 * G.TR1Size[1] + 1 * G.TR1GapY))
            and (yTR1t > -(1.5 * G.TR1Size[1] + 1 * G.TR1GapY))
        )
        or (
            (yTR1t < -(1.5 * G.TR1Size[1] + 2 * G.TR1GapY))
            and (yTR1t > -(2.5 * G.TR1Size[1] + 2 * G.TR1GapY))
        )
    )

    cond_TR2 = (yTR2b < G.TR2Size[1] / 2 and yTR2b > -G.TR2Size[1] / 2) and (
        (
            (xTR2b < (2 * G.TR2Size[0] + 1.5 * G.TR2GapX))
            and (xTR2b > (1 * G.TR2Size[0] + 1.5 * G.TR2GapX))
        )
        or (
            (xTR2b < (1 * G.TR2Size[0] + 0.5 * G.TR2GapX))
            and (xTR2b > (0 * G.TR2Size[0] + 0.5 * G.TR2GapX))
        )
        or (
            (xTR2b < -(0 * G.TR2Size[0] + 0.5 * G.TR2GapX))
            and (xTR2b > -(1 * G.TR2Size[0] + 0.5 * G.TR2GapX))
        )
        or (
            (xTR2b < -(1 * G.TR2Size[0] + 1.5 * G.TR2GapX))
            and (xTR2b > -(2 * G.TR2Size[0] + 1.5 * G.TR2GapX))
        )
    )

    return cond_TR1 and cond_TR2


# -------------------- Acceptance -------------------- #
def is_in_acceptance(x, y):
    """Check if a point (x, y) is within the detector acceptance."""
    G = ROOT.Geometry

    insideX = (
        (
            (x < G.ChipSizeX * 2.5 + G.ChipDistanceX * 2)
            and (x > G.ChipSizeX * 1.5 + G.ChipDistanceX * 2)
        )
        or (
            (x < G.ChipSizeX * 1.5 + G.ChipDistanceX * 1)
            and (x > G.ChipSizeX * 0.5 + G.ChipDistanceX * 1)
        )
        or (
            (x < G.ChipSizeX * 0.5 + G.ChipDistanceX * 0)
            and (x > -(G.ChipSizeX * 0.5 + G.ChipDistanceX * 0))
        )
        or (
            (x > -(G.ChipSizeX * 1.5 + G.ChipDistanceX * 1))
            and (x < -(G.ChipSizeX * 0.5 + G.ChipDistanceX * 1))
        )
        or (
            (x > -(G.ChipSizeX * 2.5 + G.ChipDistanceX * 2))
            and (x < -(G.ChipSizeX * 1.5 + G.ChipDistanceX * 2))
        )
    )

    insideY = (
        (
            (y < G.ChipSizeY * 5 + G.ChipStaveDistanceY * 2 + G.ChipDistanceY * 2.5)
            and (y > G.ChipSizeY * 4 + G.ChipStaveDistanceY * 2 + G.ChipDistanceY * 2.5)
        )
        or (
            (y < G.ChipSizeY * 4 + G.ChipStaveDistanceY * 2 + G.ChipDistanceY * 1.5)
            and (y > G.ChipSizeY * 3 + G.ChipStaveDistanceY * 2 + G.ChipDistanceY * 1.5)
        )
        or (
            (y < G.ChipSizeY * 3 + G.ChipStaveDistanceY * 1 + G.ChipDistanceY * 1.5)
            and (y > G.ChipSizeY * 2 + G.ChipStaveDistanceY * 1 + G.ChipDistanceY * 1.5)
        )
        or (
            (y < G.ChipSizeY * 2 + G.ChipStaveDistanceY * 1 + G.ChipDistanceY * 0.5)
            and (y > G.ChipSizeY * 1 + G.ChipStaveDistanceY * 1 + G.ChipDistanceY * 0.5)
        )
        or (
            (y < G.ChipSizeY * 1 + G.ChipStaveDistanceY * 0 + G.ChipDistanceY * 0.5)
            and (y > G.ChipSizeY * 0 + G.ChipStaveDistanceY * 0 + G.ChipDistanceY * 0.5)
        )
    )

    return insideX and insideY


# -------------------- Handle 2-cluster track -------------------- #
def handle_two_cluster_track(cls_structs, theta, phi, dist_z):
    """Handle the case of two clusters: identify missing layer, compute intersection, and check acceptance."""
    if len(cls_structs) != 2:
        return None

    G = ROOT.Geometry
    z1, z2 = cls_structs[0]["mean_z"], cls_structs[1]["mean_z"]

    missing_layer = None
    for i, z_ref in enumerate(G.StaveZ):
        if all(abs(z_ref - z_evt) > 1e-3 for z_evt in (z1, z2)):
            missing_layer = i
            break

    if missing_layer is None:
        print("⚠️ Warning: could not identify missing layer for 2-cluster track")
        return None

    if missing_layer == 0:
        x_miss = cls_structs[1]["mean_x"] - dist_z * math.tan(theta) * math.cos(phi)
        y_miss = cls_structs[1]["mean_y"] - dist_z * math.tan(theta) * math.sin(phi)
    elif missing_layer == 1:
        x_miss = cls_structs[0]["mean_x"] + dist_z * math.tan(theta) * math.cos(phi)
        y_miss = cls_structs[0]["mean_y"] + dist_z * math.tan(theta) * math.sin(phi)
    else:
        x_miss = cls_structs[0]["mean_x"] + 2 * dist_z * math.tan(theta) * math.cos(phi)
        y_miss = cls_structs[0]["mean_y"] + 2 * dist_z * math.tan(theta) * math.sin(phi)

    in_acc = is_in_acceptance(x_miss, y_miss)
    hit_tr = track_hit_TR(x_miss, y_miss, theta, phi)

    return {"missing_layer": missing_layer, "in_acceptance": in_acc, "hit_TR": hit_tr}
