import numpy as np
from nndet.core.boxes import box_iou_np
from typing import Tuple

def __duplicate_nonzero_in_rows_or_cols(arr: np.array) -> bool:
    """Check whether there are rows or columns with multiple non-zero values in a 2D array

    Args:
        arr (np.array): input 2D array

    Returns:
        bool: Whether there are non-zero rows or cols
    """
    for row in arr:
        if len(row[row>0.0])>1:
            return True
    for column in arr.T:
        if len(column[column>0.0])>1:
            return True
    return False

def preprocess_ious(ious: np.array, pred_scores: np.array, froc_threshold: float = 0.0) -> np.array:
    """Preprocess IoUs so that there is 1 value left per row/column, with the standard method

    Args:
        ious (np.array): input 2D array with the IoUs
        pred_scores (np.array): Score (confidence/prob) for each prediction
        froc_threshold (float): Threshold for IoU array. Defaults to 0.0.

    Returns:
        np.array: Preprocessed IoUs, with only 1 value per row/col.
    """
    ious[ious < froc_threshold+1e-12] = 0.0 # thresold
    xs = pred_scores.argsort() # sorted indexes of scores/probs
    while __duplicate_nonzero_in_rows_or_cols(ious):
        x, xs = xs[-1], xs[:-1] # pop highest score/prob
        y = ious[x].argmax() # GT index with highest IoU for this prediction
        value = ious[x][y]
        if value > 0+1e-12: # if there is indeed a successful prediction, remove others
            ious[x] = np.zeros(ious.shape[1])
            ious[:, y] = np.zeros(ious.shape[0])
            ious[x][y] = value
    return ious

def matcher(p_ious: np.array, pred_array: np.array) -> Tuple[np.array, np.array]:
    """Find prediction scores of false positives and true positives.

    Args:
        p_ious (np.array): input 2D array with the preprocessed IoUs (max 1 value per row/col)
        pred_array (np.array): any array that has the same size/order as the predictions. Like pred_scores.

    Returns:
        Tuple[np.array, np.array]: false positive values, true positive values
    """
    if len(pred_array) == 0:
        return np.array([]), np.array([])
    tp_idxs = np.sum(p_ious, axis=1) > 1e-12
    return pred_array[~tp_idxs], pred_array[tp_idxs]