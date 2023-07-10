"""Results same as nndet
"""

import os, argparse, yaml, json
import shutil
from typing import Sequence, Dict
from tqdm import tqdm
import numpy as np
import SimpleITK as sitk

from nndet.io.load import load_pickle, save_pickle
from SA_FROC5_helper import sa_froc5_probs_and_weights, score_and_plot_image
from monai_froc_weighted import compute_weighted_froc_curve_data


### ---- Actual runner ----
def runner(
    pred_boxes_dir, 
    gt_boxes_dir, 
    nii_images_dir, 
    nii_labels_dir, 
    output_dir, 
    config: Dict,
    classes: Sequence[int] = None, 
    only_pred_size = False,
    threshold = 0.1,
    prefer_nii_images_dir: bool = False,
):
    ## Collect case IDs
    case_ids = [
        '_'.join(x.split('_')[:-1]) 
        for x in os.listdir(pred_boxes_dir) 
        if x.endswith('_boxes.pkl')
    ]

    size_method_name = config['size_aware_loss_config']['size_method_name']
    size_to_weight_function_name_config_tuples = [
        (x['name'], x['config'])
        for x in config['size_aware_loss_config']['size_to_weight_combine']
    ]
    max_w = config['size_aware_loss_config']['max_w']

    num_images = 0
    num_targets = 0
    fp_probs = np.array([])
    fp_weights = np.array([])
    tp_probs = np.array([])
    tp_weights = np.array([])
    target_weights = np.array([])

    ### Iterate over cases
    for case_id in (pbar := tqdm(case_ids)):
        pbar.set_description(f"> Processing {case_id}")

        ### Get spacing
        if not prefer_nii_images_dir:
            spath = os.path.join(nii_labels_dir, case_id+'.nii.gz')
        else:
            spath = os.path.join(nii_images_dir, case_id+'_0000.nii.gz')
        x_spacing, y_spacing, z_spacing = sitk.ReadImage(spath).GetSpacing()

        ### Get boxes, classes, scores
        gt = np.load(os.path.join(gt_boxes_dir, f"{case_id}_boxes_gt.npz"), allow_pickle=True)
        pred = load_pickle(os.path.join(pred_boxes_dir, f"{case_id}_boxes.pkl"))
        gt_boxes = gt["boxes"] if gt["boxes"].shape[-1] != 0 else np.array([])
        gt_classes = gt["classes"] if gt["classes"].shape[-1] != 0 else np.array([])
        pred_boxes = pred["pred_boxes"] if pred["pred_boxes"].shape[-1] != 0 else np.array([])
        pred_classes = pred["pred_labels"] if pred["pred_labels"].shape[-1] != 0 else np.array([])
        pred_scores = pred["pred_scores"] if pred["pred_scores"].shape[-1] != 0 else np.array([])

        ### Filter for classes, if needed
        if classes is not None:
            pred_boxes = pred_boxes[np.isin(pred_classes, classes)]
            pred_scores = pred_scores[np.isin(pred_classes, classes)]
            gt_boxes = gt_boxes[np.isin(gt_classes, classes)]

        num_images += 1
        num_targets += len(gt_boxes)

        _fp_probs, _fp_weights, _tp_probs, _tp_weights, _gt_weights = sa_froc5_probs_and_weights(
            pred_boxes, pred_scores, gt_boxes,
            size_method_name, size_to_weight_function_name_config_tuples,
            threshold, only_pred_size, x_spacing, y_spacing, z_spacing,
        )

        fp_probs = np.append(fp_probs, _fp_probs)
        fp_weights = np.append(fp_weights, _fp_weights)
        tp_probs = np.append(tp_probs, _tp_probs)
        tp_weights = np.append(tp_weights, _tp_weights)
        target_weights = np.append(target_weights, _gt_weights)
    
    sum_target_weights = sum(target_weights)
    num_nonzero_target_weights = len(target_weights[target_weights > 1e-12])
    sum_fp_weights = sum(fp_weights)
    num_nonzero_fp_weights = len(fp_weights[fp_weights > 1e-12])
    _a = fp_probs[fp_weights > 1e-12]
    num_nonzero_fp_weights_score0p1 = len(_a[_a > 0.1-1e-12])
    sum_tp_weights = sum(tp_weights)
    num_nonzero_tp_weights = len(tp_weights[tp_weights > 1e-12])
    _a = tp_probs[tp_weights > 1e-12]
    num_nonzero_tp_weights_score0p1 = len(_a[_a > 0.1-1e-12])

    fp_weights /= max_w
    tp_weights /= max_w
    target_weights /= max_w

    fps, sens, _ = compute_weighted_froc_curve_data(
        fp_probs, fp_weights, 
        tp_probs, tp_weights,
        target_weights, 
        num_images,
    )

    score, image, curve = score_and_plot_image(fps, sens)
    print(f"Score: {score}")
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(config, f)
    with open(os.path.join(output_dir, 'SA_FROC5f.json'), 'w') as f:
        json.dump({
            'SA_FROC_0.1': score,
            'num_images': num_images,
            'num_targets': num_targets,
            'sum_target_weights': sum_target_weights,
            'sum_fp_weights': sum_fp_weights,
            'sum_tp_weights': sum_tp_weights,
            'num_nonzero_target_weights': num_nonzero_target_weights,
            'num_nonzero_fp_weights': num_nonzero_fp_weights,
            'num_nonzero_tp_weights': num_nonzero_tp_weights,
            'num_nonzero_fp_weights_score0p1': num_nonzero_fp_weights_score0p1,
            'num_nonzero_tp_weights_score0p1': num_nonzero_tp_weights_score0p1,
            'sens@0.125FPs/case': curve[0],
            'sens@0.25FPs/case': curve[1],
            'sens@0.5FPs/case': curve[2],
            'sens@1FP/case': curve[3],
            'sens@2FP/case': curve[4],
            'sens@4FP/case': curve[5],
            'sens@8FP/case': curve[6],
        }, f, indent=4)
    image.save(os.path.join(output_dir, 'SA_FROC5f_0.1.png'))



### ---- CLI ----
if __name__=='__main__':
    ### Args
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "task",
        type=str,
        help="If Task123X_Name give '123X'",
    )
    parser.add_argument(
        "model",
        type=str,
        help="Name of the subfolder under the task name in $det_models"
    )
    parser.add_argument(
        "fold",
        type=int,
        help="Which fold (give -1 for consolidated)"
    )
    parser.add_argument(
        "configs",
        type=str,
        help="Which configs to use (comma separated). Name in config/ directory, without .yaml",
    )
    parser.add_argument(
        "-c", "--classes",
        type=str,
        help="Which classes to work on. All if not provided. Comma separated if many",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--test",
        help="Evaluate test predictions -> uses different folder",
        action="store_true",
    )
    parser.add_argument(
        "-p", "--predonly",
        help="Only use pred boxes, no gt box when it matches",
        action="store_true",
    )
    parser.add_argument(
        "-s", "--spacingfromimages",
        help="Get spacing information for each case from images dir instead of labels dir (because labels might be wrong)",
        action="store_true",
    )

    parser.add_argument(
        "-pt", "--parenttask",
        type=str,
        help="Get spacing information from nifti directory of another task. Use if raw_splitted/ does not exist for this task.",
        default=None,
        required=False,
    )

    script_dir_path = os.path.dirname(os.path.abspath(__file__))

    args = parser.parse_args()
    task_id = args.task
    model = args.model
    fold = args.fold
    config_names = args.configs.split(',')
    configs = []
    for config_name in config_names:
        with open(os.path.join(script_dir_path, 'configs', config_name + '.yaml'), 'r') as f:
            configs.append( yaml.safe_load(f) )
    classes = [int(x) for x in args.classes.split(',')] if args.classes is not None else None
    test = args.test
    only_pred_size = args.predonly
    spacing_from_images = args.spacingfromimages
    parent_task_id = args.parenttask

    ### Paths
    det_data = os.environ['det_data']
    det_models = os.environ['det_models']
    ## Task dirs
    _possible_task_full_names = [x for x in os.listdir(det_data) if x.startswith(f"Task{task_id}_")]
    if len(_possible_task_full_names) > 1:
        raise ValueError(f"Task ID Ambiguity: There must be only 1 directory in det_data nndet dir which starts with 'Task{task_id}_'")
    if len(_possible_task_full_names) == 0:
        raise ValueError(f"Not task found that starts with 'Task{task_id}_'")
    task_full_name = _possible_task_full_names[0]
    task_data_dir = os.path.join(det_data, task_full_name)
    task_models_dir = os.path.join(det_models, task_full_name)
    ## Subdirs in det_models
    fold_str = f"fold{fold}" if fold != -1 else "consolidated"
    fold_dir = os.path.join(task_models_dir, model, fold_str)

    ### Dirs needed for runner
    pred_boxes_dir = (
        os.path.join(fold_dir, 'test_predictions')
        if test else
        os.path.join(fold_dir, 'val_predictions')
    )
    gt_boxes_dir = (
        os.path.join(task_data_dir, 'preprocessed', 'labelsTs')
        if test else
        os.path.join(task_data_dir, 'preprocessed', 'labelsTr')
    )
    nii_parent_dir = (
        task_data_dir 
        if parent_task_id is None else
        os.path.join(
            det_data,
            [x for x in os.listdir(det_data) if x.startswith(f"Task{parent_task_id}_")][0]
        )
    )
    nii_images_dir = os.path.join(nii_parent_dir, 'raw_splitted', 
        ('imagesTs' if test else 'imagesTr')
    )
    nii_labels_dir = os.path.join(nii_parent_dir, 'raw_splitted', 
        ('labelsTs' if test else 'labelsTr')
    )
    cl_str = 'all' if classes is None else ('cl' + '-'.join([str(x) for x in classes]))
    prefix = 'P_' if only_pred_size else ''
    
    for idx, (config_name, config) in enumerate(zip(config_names, configs)):
        if len(configs)>1:
            print('\n---')
            print(f">>> {idx+1}/{len(configs)} Running config: {config_name}")
            print('---\n')
        output_dir = os.path.join(fold_dir, 'SA_FROC5f_'+('test' if test else 'train'), prefix + config_name + '__' + cl_str)
        runner(
            pred_boxes_dir, gt_boxes_dir, nii_images_dir, nii_labels_dir, output_dir, 
            config, classes, only_pred_size, 0.1, spacing_from_images
        )
