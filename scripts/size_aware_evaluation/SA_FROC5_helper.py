from typing import Tuple
import io
import numpy as np
from IoUPreprocessAndMatching import preprocess_ious, matcher
from nndet.core.boxes.size_to_weight import boxes_size_weight_combine
from nndet.core.boxes import box_iou_np
import PIL
from loguru import logger
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

def sa_froc5_probs_and_weights(
    pred_boxes: np.array, 
    pred_scores: np.array, 
    gt_boxes: np.array, 
    size_method_name,
    size_to_weight_function_name_config_tuples,
    threshold: float = 0.1,
    only_pred_size: bool = False,
    x_spacing: float = 1.0,
    y_spacing: float = 1.0,
    z_spacing: float = 1.0,
    
)-> Tuple[np.array, np.array, np.array, np.array, np.array]:
    """Get probabilities and weights for a single case

    Args:
        pred_boxes (np.array): Prediction boxes (x1,y1,x2,y2,z1,z2). [Nx6]
        pred_scores (np.array): Prediction scores for each box. [N]
        gt_boxes (np.array): Groung truth boxes (x1,y1,x2,y2,z1,z2). [Mx6]
        size_method_name (str, optional): Method to be used for size calculation. Defaults to 'area'.
        size_to_weight_function_name_config_tuples (List[Tuple[str, Dict]], optional):
            List of size to weight function names+configs to apply.
            Defaults to [('linear', {})].
        threshold (float, optional): IoU threshold. Defaults to 0.1.
        only_pred_size(bool, optional): Don't use GT box size when a DT matches. Defaults to False.
        x_spacing (float, optional): Image characteristic. Defaults to 1.0.
        y_spacing (float, optional): Image characteristic. Defaults to 1.0.
        z_spacing (float, optional): Image characteristic. Defaults to 1.0.

    Returns:
        Tuple[np.array, np.array, np.array, np.array, np.array]: 
            fp_probs, fp_weights, tp_probs, tp_weights, gt_weights
            for sa_froc calculation
    """
    ### Calculate weights
    pred_weights = boxes_size_weight_combine(
        pred_boxes, 
        size_method_name=size_method_name, 
        size_to_weight_function_name_config_tuples=size_to_weight_function_name_config_tuples,
        x_spacings=np.full(pred_boxes.shape[0], x_spacing),
        y_spacings=np.full(pred_boxes.shape[0], y_spacing),
        z_spacings=np.full(pred_boxes.shape[0], z_spacing),
    )
    gt_weights = boxes_size_weight_combine(
        gt_boxes, 
        size_method_name=size_method_name, 
        size_to_weight_function_name_config_tuples=size_to_weight_function_name_config_tuples,
        x_spacings=np.full(gt_boxes.shape[0], x_spacing),
        y_spacings=np.full(gt_boxes.shape[0], y_spacing),
        z_spacings=np.full(gt_boxes.shape[0], z_spacing),
    )

    if only_pred_size:
        raise NotImplementedError # Maybe below is fine, but not needed now
        ### If we don't want to use matching GT weights, remove 0 weight predictions early.
        gt_boxes = gt_boxes[gt_weights > 0.0+1e-12]
        gt_weights = gt_weights[gt_weights > 0.0+1e-12]
        pred_boxes = pred_boxes[pred_weights > 0.0+1e-12]
        pred_scores = pred_scores[pred_weights > 0.0+1e-12]
        pred_weights = pred_weights[pred_weights > 0.0+1e-12]

    ### IoUs
    ious = box_iou_np(pred_boxes, gt_boxes)
    ious = preprocess_ious(ious, pred_scores, threshold)

    if not only_pred_size:
        ### If we want to use matching GT weights, set weight of prediction to weight of GT if matching.
        for pred_idx in range(len(pred_boxes)):
            if sum(ious[pred_idx]) > 0:
                gt_idx = ious[pred_idx].argmax()
                pred_weights[pred_idx] = gt_weights[gt_idx]

    fp_probs, tp_probs = matcher(ious, pred_scores)
    fp_weights, tp_weights = matcher(ious, pred_weights)

    ## Change for FPs
    fp_weights = 1 - fp_weights

    return fp_probs, fp_weights, tp_probs, tp_weights, gt_weights



def score_and_plot_image(
    fps: np.array, 
    sens: np.array, 
    show: bool = False,
) -> Tuple[float, PIL.Image.Image]:
    """Calculate the SA FROC score and a plot for better visualization.

    Args:
        fps (np.array): false positives [N]
        sens (np.array): sensitivities [N]
        show (bool, optional): whether to pop a window with the image of the plot. 
            Defaults to False.

    Returns:
        Tuple[float, PIL.Image.Image]: score (float), PIL image
    """
    fpi_thresholds = [1/8, 1/4, 1/2, 1, 1*2, 1*4, 1*8]
    curve = np.interp(fpi_thresholds, fps, sens)
    logger.info(f'SA_FROC3.score_and_plot_image: Finished.\ncurve={curve}')

    ### Score
    score = np.mean(curve)

    ### Plot
    fig, ax = plt.subplots()
    ## y-axis
    ax.set_ylim([0, 1.02])
    plt.gca().yaxis.grid(True)
    plt.yticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    ## x-axis
    ax.set_xscale("log", base=2)
    ax.plot(fpi_thresholds, curve, "o-")
    ax.set_title(f"SA FROC curve (score={score:.2f})")
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    plt.xticks(fpi_thresholds, rotation=33)
    
    ### PIL Image from plot
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    image = PIL.Image.open(buf)
    plt.close(fig)
    if show:
        image.show()

    ### Result
    return score, image, curve