# imlh-icml-detection-tools

Copyright © German Cancer Research Center (DKFZ), [Division of Medical Image Computing (MIC)](https://www.dkfz.de/en/mic/index.php). Please make sure that your usage of this code is in compliance with the code [license](./LICENSE). 

-----

Requires [nndetection](https://github.com/MIC-DKFZ/nnDetection) to run.

* Adaptation to focal loss is in `nndet/`. This files should be added/replaced in regular version of nndetection and are then used during training. It also needs changes to the nndetection yaml configuration

```
  head_kwargs: # keyword arguments to passed to head
    size_aware_loss_config: # SA
      size_method_name: 'max-axial-diameter'
      x_training_spacing: 1.4 # replace with your data
      y_training_spacing: 1.4 # replace with your data
      z_training_spacing: 2.0 # replace with your data
      target_class: 0 # replace with your target class
      target_weight_if_available: True # only for target class
      weigh_fp: True # Whether to apply weighing when the box predicts nothing
      consider_box_predicting_other_foreground_class_as_fp: True # -//- non-target foreground class
      invert_fp: True # Low risk --> high weight for FPs. Make sure weights go up to 1.0 when inverting is set to True
      other_foreground_classes_weight: 1.0
      loss_weight_max: 2.0
      loss_weight_min: 1.0
      size_to_weight_combine:
      - name: 'breast-mortality-risk-sopik'
        config: {}
```

* Adaptation to metric is in `scripts/` (`SA_FROC5_runner.py`). It is a regular script meant to run independently of nndetection. Requires extra package `pillow`. For config in the argument choose `erlangen_mort_sopik_mad`.