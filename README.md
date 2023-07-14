# Detection tools for IMLH-ICML 2023

Copyright © German Cancer Research Center (DKFZ), [Division of Medical Image Computing (MIC)](https://www.dkfz.de/en/mic/index.php). Please make sure that your usage of this code is in compliance with the code [license](./LICENSE). 

-----

This repository contains code required to run risk-adjusted training and evaluation. It is complementary to the following work, accepted at ICML-IMLH 2023:

```
Risk-adjusted Training and Evaluation for Medical Object Detection in Breast Cancer MRI, Bounias et al., Workshop on Interpretable ML in Healthcare at International Conference on Machine Learning (ICML), Honolulu, Hawaii, USA. 2023.
```

```
1 Division of Medical Image Computing (MIC), German Cancer Research Center (DKFZ), Heidelberg, Germany 2 Medical Faculty, University of Heidelberg, Heidelberg, Germany 3 Faculty of Mathematics and Computer Science, University of Heidelberg, Heidelberg, Germany 4 Helmholtz Imaging, German Cancer Research Center (DKFZ), Heidelberg, Germany 5 German Cancer Consortium (DKTK), Partner Site Heidelberg, Germany 6 Pattern Analysis and Learning Group, Department of Radiation Oncology, Heidelberg University Hospital, Heidelberg, Germany 7 Heidelberg Institute of Radiation Oncology (HIRO), National Center for Radiation Research in Oncology (NCRO), Heidelberg, Germany 8 Interactive Machine Learning Group, German Cancer Research Center (DKFZ), Heidelberg, Germany 9 Institute of Radiology, Uniklinikum Erlangen, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU), Germany 10 National Center for Tumor Diseases (NCT), Heidelberg University Hospital (UKHD) and German Cancer Research Center (DKFZ), Heidelberg, Germany.
```


The code is meant to allow introducing risk functions into the calculations of medical object detection (currently through usage of object size), both in training (by adaptation of focal loss) and during evaluation (by adaptation of the FROC metric). It is still work-in-progress.

Requires [nndetection](https://github.com/MIC-DKFZ/nnDetection) to run.

* **Adaptation to loss** is in `nndet/`. This folder should be added/replaced in regular version of nndetection (root directory) and are then used during training. It also needs changes to the nndetection yaml configuration:

  1. `module` should be replaced with `RetinaUNetC011FocalSA`.

  2. The following should be added to the `head_kwargs` key:

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

* **Adaptation to metric** is in `scripts/` (`SA_FROC5_runner.py`). It is a regular script meant to run independently of nndetection. Requires extra package `pillow`. For config in the argument choose `erlangen_mort_sopik_mad`.
