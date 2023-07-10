from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from loguru import logger

from nndet.arch.heads.classifier.dense import DenseClassifierType
from nndet.arch.heads.comb.base import AnchorHead
from nndet.arch.heads.comb.anchor_all import BoxHeadAll
from nndet.arch.heads.comb.anchor_sampled import BoxHeadHNM
from nndet.arch.heads.regressor.dense_single import DenseRegressorType
from nndet.core.boxes.coder import BoxCoderND
from nndet.core.boxes.sampler import SamplerType
from nndet.core.boxes.size_to_weight import boxes_size_weight_combine

class BoxHeadSAComputeBase:
    @staticmethod
    def compute_size_weights(
        boxes: Tensor,
        target_boxes: Tensor,
        conf: Dict,
    ) -> Tuple[Tensor, Tensor]:
        """Size-aware weights. 
        Helper function to calculate weights needed based on the configuration provided.
        Target boxes and labels refer to GT boxes that the respective box might predict.
        Target box is "random" when target label is 0 (background class)

        Args:
            boxes (Tensor): Boxes (x1,y1,x2,y2,z1,z2) [N]
            target_boxes (Tensor): GT boxes, containing unusual things when boxes don't match [N]
            conf (Dict): The contents of 'size_aware_loss_config' key of 'head_kwargs' of config.

        Returns:
            Tuple[Tensor, Tensor]: Weight for each prediction box, Weight for each target box
        """
        ### Spacing if provided (otherwise provide values adjusted to training spacing ('target_spacing' in the plan.pkl))
        x_spacings = (conf['x_training_spacing'] if 'x_training_spacing' in conf.keys() else None)
        y_spacings = (conf['y_training_spacing'] if 'y_training_spacing' in conf.keys() else None)
        z_spacings = (conf['z_training_spacing'] if 'z_training_spacing' in conf.keys() else None)
        ### Actually calculate weights
        size_method_name = conf['size_method_name']
        size_to_weight_function_name_config_tuples = [
            (x['name'], x['config'])
            for x in conf['size_to_weight_combine']
        ]
        weights_pred = boxes_size_weight_combine(
            boxes, 
            size_method_name, 
            size_to_weight_function_name_config_tuples,
            x_spacings, y_spacings, z_spacings,
        )
        weights_target = boxes_size_weight_combine(
            target_boxes, 
            size_method_name, 
            size_to_weight_function_name_config_tuples,
            x_spacings, y_spacings, z_spacings,
        )
        return weights_pred, weights_target


class BoxHeadSAHNM(BoxHeadHNM, BoxHeadSAComputeBase):
    """
    Size-aware (SA) adaptation of BoxHead for Hard Negative Mining loss
    """
    def __init__(self, size_aware_loss_config: Dict, **kwargs):
        super().__init__(**kwargs)
        self.size_aware_loss_config = size_aware_loss_config

    def compute_loss(self, 
        prediction: Dict[str, Tensor], target_labels: List[Tensor], 
        matched_gt_boxes: List[Tensor], anchors: List[Tensor]
    ) -> Tuple[Dict[str, Tensor], torch.Tensor, torch.Tensor]:
        """Override of BoxHeadHNM"""
        conf = self.size_aware_loss_config
        ### ---- Calculate normal loss
        losses, sampled_pos_inds, sampled_neg_inds = super().compute_loss(
            prediction, target_labels, matched_gt_boxes, anchors
        )
        ### ---- Parameters needed that are found from the arguments
        sampled_inds   = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)
        boxes = self.coder.decode_single(prediction["box_deltas"], torch.cat(anchors, dim=0))[sampled_inds]
        target_boxes = torch.cat(matched_gt_boxes, dim=0)[sampled_inds]
        target_labels = torch.cat(target_labels, dim=0)[sampled_inds]
        ### ---- Calculate "risk"
        risk_pred, risk_target = BoxHeadSAComputeBase.compute_size_weights(
            boxes, target_boxes, conf
        )
        ## For TP
        if conf['target_weight_if_available']:
            weight_p = torch.where(target_labels==conf['target_class']+1, risk_target.to(torch.float16), risk_pred.to(torch.float16))
        else:
            weight_p = risk_pred.to(torch.float16)
        ## For FP
        if conf['weigh_fp']:
            weight_n = (1 - weight_p) if conf['invert_fp'] else weight_p
        else:
            weight_n = torch.ones(weight_p.shape, dtype=weight_p.dtype, device=weight_p.device)
        ## For FP in boxes predicting other classes
        if conf['consider_box_predicting_other_foreground_class_as_fp']:
            weight_o = weight_n
        else:
            weight_o = torch.ones(weight_p.shape, dtype=weight_p.dtype, device=weight_p.device)
        ### ---- Apply min/max
        min_w = conf['loss_weight_min']
        max_w = conf['loss_weight_max']
        weight_p = min_w+(max_w-min_w)*weight_p
        weight_n = min_w+(max_w-min_w)*weight_n
        weight_o = min_w+(max_w-min_w)*weight_o
        ### ---- Make them apply only in certain cases
        weight_p[target_labels != conf['target_class']+1] = 1.0
        weight_n[target_labels != 0] = 1.0
        weight_o[target_labels == 0] = 1.0
        weight_o[target_labels == conf['target_class']+1] = 1.0
        w = weight_p * weight_n * weight_o
        if w.min().item() < min_w - 1e-12 or w.max().item() > max_w + 1e-12:
            raise ValueError('Unexpected weight off min/max limits')
        ### ----- Apply scale to loss and then average it
        losses["cls"][:, conf['target_class']] *= w
        ## Non-target foreground classes weight
        for cl in [cl for cl in range(losses["cls"].shape[1]) if cl != conf['target_class']]:
            losses["cls"][:, cl] *= conf['other_foreground_classes_weight']
        losses["cls"] = torch.nan_to_num( losses["cls"], nan=1e-12 ) # making sure
        losses["cls"] = torch.sum(losses["cls"]) # IT IS ALREADY DIVIDED: / max(1, sampled_pos_inds.numel())
        ### ---- Sum regression loss because of none reduction
        if "reg" in losses.keys() and len(losses["reg"].shape) > 0:
            losses["reg"] = torch.nan_to_num( losses["reg"], nan=1e-12 ) # making sure
            losses["reg"] = torch.sum(losses["reg"]) # IT IS ALREADY DIVIDED: / max(1, sampled_pos_inds.numel())
        ### ---- Return normally
        return losses, sampled_pos_inds, sampled_neg_inds



class BoxHeadSAFocal(BoxHeadAll, BoxHeadSAComputeBase):
    """
    Size-aware (SA) adaptation of BoxHead for Focal loss
    """
    def __init__(self, size_aware_loss_config: Dict, **kwargs):
        super().__init__(**kwargs)
        self.size_aware_loss_config = size_aware_loss_config

    def compute_loss(self, 
        prediction: Dict[str, Tensor], target_labels: List[Tensor], 
        matched_gt_boxes: List[Tensor], anchors: List[Tensor]
    ) -> Tuple[Dict[str, Tensor], torch.Tensor, Optional[torch.Tensor]]:
        """Override of BoxHeadAll"""
        conf = self.size_aware_loss_config
        ### ---- Calculate normal loss
        losses, sampled_pos_inds, sampled_neg_inds = super().compute_loss(
            prediction, target_labels, matched_gt_boxes, anchors
        )
        ### ---- Parameters needed that are found from the arguments
        boxes = self.coder.decode_single(prediction["box_deltas"], torch.cat(anchors, dim=0))
        target_boxes = torch.cat(matched_gt_boxes, dim=0)
        target_labels = torch.cat(target_labels, dim=0)
        ### ---- Calculate "risk"
        risk_pred, risk_target = BoxHeadSAComputeBase.compute_size_weights(
            boxes, target_boxes, conf
        )
        ## For TP
        if conf['target_weight_if_available']:
            weight_p = torch.where(target_labels==conf['target_class']+1, risk_target.to(torch.float16), risk_pred.to(torch.float16))
        else:
            weight_p = risk_pred.to(torch.float16)
        ## For FP
        if conf['weigh_fp']:
            weight_n = (1 - weight_p) if conf['invert_fp'] else weight_p
        else:
            weight_n = torch.ones(weight_p.shape, dtype=weight_p.dtype, device=weight_p.device)
        ## For FP in boxes predicting other classes
        if conf['consider_box_predicting_other_foreground_class_as_fp']:
            weight_o = weight_n
        else:
            weight_o = torch.ones(weight_p.shape, dtype=weight_p.dtype, device=weight_p.device)
        ### ---- Apply min/max
        min_w = conf['loss_weight_min']
        max_w = conf['loss_weight_max']
        weight_p = min_w+(max_w-min_w)*weight_p
        weight_n = min_w+(max_w-min_w)*weight_n
        weight_o = min_w+(max_w-min_w)*weight_o
        ### ---- Make them apply only in certain cases
        weight_p[target_labels != conf['target_class']+1] = 1.0
        weight_n[target_labels != 0] = 1.0
        weight_o[target_labels == 0] = 1.0
        weight_o[target_labels == conf['target_class']+1] = 1.0
        w = weight_p * weight_n * weight_o
        if w.min().item() < min_w - 1e-12 or w.max().item() > max_w + 1e-12:
            raise ValueError('Unexpected weight off min/max limits')
        ### ----- Apply scale to loss and then average it
        losses["cls"][:, conf['target_class']] *= w
        ## Non-target foreground classes weight
        for cl in [cl for cl in range(losses["cls"].shape[1]) if cl != conf['target_class']]:
            losses["cls"][:, cl] *= conf['other_foreground_classes_weight']
        losses["cls"] = torch.nan_to_num( losses["cls"], nan=1e-12 ) # making sure
        losses["cls"] = torch.sum(losses["cls"]) # IT IS ALREADY DIVIDED: / max(1, sampled_pos_inds.numel())
        ### ---- Sum regression loss because of none reduction
        if "reg" in losses.keys() and len(losses["reg"].shape) > 0:
            losses["reg"] = torch.nan_to_num( losses["reg"], nan=1e-12 ) # making sure
            losses["reg"] = torch.sum(losses["reg"]) # IT IS ALREADY DIVIDED: / max(1, sampled_pos_inds.numel())
        ### ---- Return normally
        return losses, sampled_pos_inds, sampled_neg_inds
