"""
95th Percentile Hausdorff Distance (HD95) for BraTS 2021.
Computes symmetric boundary surface distances with standard challenge penalty conventions.
"""

from typing import Union, Tuple, Dict
import numpy as np
import scipy.ndimage as ndi


def compute_hd95(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    penalty_distance: float = 373.13  # Max diagonal for 240x240x155 at 1mm spacing
) -> float:
    """
    Compute 95th percentile Hausdorff distance between two binary 3D masks.
    """
    p = np.asarray(pred_mask, dtype=bool)
    g = np.asarray(gt_mask, dtype=bool)

    p_sum = np.sum(p)
    g_sum = np.sum(g)

    if p_sum == 0 and g_sum == 0:
        return 0.0
    if p_sum == 0 or g_sum == 0:
        return float(penalty_distance)

    # Extract 3D surface boundary voxels (xor with erosion)
    struct = ndi.generate_binary_structure(3, 1)
    p_border = p ^ ndi.binary_erosion(p, structure=struct)
    g_border = g ^ ndi.binary_erosion(g, structure=struct)

    # Compute Euclidean distance transform
    dt_gt = ndi.distance_transform_edt(~g_border, sampling=voxel_spacing)
    dt_pred = ndi.distance_transform_edt(~p_border, sampling=voxel_spacing)

    # Distances from pred border to nearest GT border
    distances_p_to_g = dt_gt[p_border]
    # Distances from GT border to nearest pred border
    distances_g_to_p = dt_pred[g_border]

    if len(distances_p_to_g) == 0 or len(distances_g_to_p) == 0:
        return float(penalty_distance)

    hd95_p_to_g = np.percentile(distances_p_to_g, 95)
    hd95_g_to_p = np.percentile(distances_g_to_p, 95)

    return float(max(hd95_p_to_g, hd95_g_to_p))


def compute_brats_hd95s(
    pred_regions: Tuple[np.ndarray, np.ndarray, np.ndarray],
    gt_regions: Tuple[np.ndarray, np.ndarray, np.ndarray],
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> Dict[str, float]:
    """
    Compute WT, TC, and ET HD95 scores.
    """
    p_wt, p_tc, p_et = pred_regions
    g_wt, g_tc, g_et = gt_regions

    hd_wt = compute_hd95(p_wt, g_wt, voxel_spacing)
    hd_tc = compute_hd95(p_tc, g_tc, voxel_spacing)
    hd_et = compute_hd95(p_et, g_et, voxel_spacing)
    hd_avg = (hd_wt + hd_tc + hd_et) / 3.0

    return {
        "hd95_wt": hd_wt,
        "hd95_tc": hd_tc,
        "hd95_et": hd_et,
        "hd95_avg": hd_avg
    }
