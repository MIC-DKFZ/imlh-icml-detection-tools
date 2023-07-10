import numpy as np
from typing import Tuple

def compute_weighted_froc_curve_data(
    fp_probs: np.ndarray,
    fp_weights: np.ndarray,
    tp_probs: np.ndarray,
    tp_weights: np.ndarray,
    target_weights: np.ndarray,
    num_images: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """This function is modified from the official evaluation code of
    `CAMELYON 16 Challenge <https://camelyon16.grand-challenge.org/>`_, and used to compute
    the required data for plotting the Free Response Operating Characteristic (FROC) curve.
    
    From https://docs.monai.io/en/stable/_modules/monai/metrics/froc.html
    Adapted to allow for weights and to also return thresholds.
    Weights must be in [0,1].

    Args:
        fp_probs (np.ndarray): an array that contains the probabilities of the false positive detections for all
            images under evaluation. [N]
        fp_weights (np.ndarray): array that contains the associated weight of each FP. [N]
        tp_probs (np.ndarray): an array that contains the probabilities of the True positive detections for all
            images under evaluation. [M]
        fp_weights (np.ndarray): array that contains the associated weight of each TP. [M]
        target_weights (np.ndarray): the weight of each targets for all images under evaluation. [K]
        num_images (int): the number of images under evaluation.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: FPs per image, sensitivities, thresholds
    """
    assert len(fp_weights) == 0 or fp_weights.max() < 1.0 + 1e-12
    assert len(tp_weights) == 0 or tp_weights.max() < 1.0 + 1e-12
    assert target_weights.max() < 1.0 + 1e-12
    
    fp_probs = fp_probs[fp_weights > 1e-12]
    fp_weights = fp_weights[fp_weights > 1e-12]
    tp_probs = tp_probs[tp_weights > 1e-12]
    tp_weights = tp_weights[tp_weights > 1e-12]
    target_weights = target_weights[target_weights > 1e-12]

    ### Find thresholds
    all_probs = sorted(set(list(fp_probs) + list(tp_probs)), reverse=True)
    ### Find TPs, FPs for each threshold
    total_fps, total_tps = [0], [0]
    for thresh in all_probs:
        mask_fp = np.where((fp_probs >= thresh), True, False)
        mask_tp = np.where((tp_probs >= thresh), True, False)
        valid_fp_weights = fp_weights[mask_fp]
        valid_tp_weights = tp_weights[mask_tp]
        total_fps.append(valid_fp_weights.sum())
        total_tps.append(valid_tp_weights.sum())
    ### fps, sens
    fps_per_image = np.asarray(total_fps) / float(num_images)
    total_sensitivity = np.asarray(total_tps) / target_weights.sum()
    return fps_per_image, total_sensitivity, np.array([0] + all_probs)


if __name__=='__main__':
    target_weights = np.array([0.5, 1.0])
    num_images = 1
    fpi_thresholds = [1/8, 1/4, 1/2, 1, 1*2, 1*4, 1*8]

    ## 2 TP, 2 FP
    print("\n---- 2 TP, 2 FP (2 targets) (lowest thres -> TP) ----")
    fp_probs   = np.array([0.6, 0.9])
    fp_weights = np.array([0.4, 0.8])
    tp_probs   = np.array([0.5, 0.7])
    tp_weights = np.array([1.0, 0.5])
    fps, sens, thresholds = compute_weighted_froc_curve_data(
        fp_probs, fp_weights, tp_probs, tp_weights, target_weights, num_images
    )
    print(f"fps:\n{fps}")
    print(f"sens:\n{sens}")
    print(f"thresholds:\n{thresholds}")
    curve = np.interp(fpi_thresholds, fps, sens)
    print(f"curve: {curve}")
