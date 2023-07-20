# Neural Extended Kalman Filters (Neural EKF)
 
This repository contains codes and data for the following publication:
* Wei Liu, Zhilu Lai, Kiran Bacsa, and Eleni Chatzi (2023). [Neural extended Kalman filters for learning and predicting dynamics of structural systems](https://doi.org/10.1177/14759217231179912). *Structural Health Monitoring*.

The Neural EKF is a generalized version of the conventional EKF, where the modeling of process dynamics and sensory observations can be parameterized by neural networks, therefore learned by end-to-end training. The method is implemented under the variational inference framework with the EKF conducting inference from sensing measurements.

## Repository Overview
 * `networks` - Neural EKF model.
   * `core.py` - Key functions for the Neural EKF.
   * `ekf.py` - Functions for EKF computation.
   * `models.py` - Architectures of neural networks.
 * `utils` - Utility functions.
   * `data_utils.py` - Functions for loading and processing raw data.
   * `log_utils.py` - Functions for logging metrics and plots.
   * `net_utils.py` - Funcionts for convenience of some computation.
   * `plot_utils.py` - Plotting functions.
 * `datasets.py` - Dataset configuration.
 * `duffing_data_preprocess.ipynb` - Transform .mat data into .pickle data.
 * `experiments.py` - Manage training and evaluation of models.
 * `train.py` - Configuration for training Neural EKF model.

## Citation
Please cite the following paper if you find the work relevant and useful in your research:
```
@article{doi:10.1177/14759217231179912,
author = {Wei Liu and Zhilu Lai and Kiran Bacsa and Eleni Chatzi},
title ={Neural extended Kalman filters for learning and predicting dynamics of structural systems},
journal = {Structural Health Monitoring},
pages = {14759217231179912},
year = {2023},
doi = {10.1177/14759217231179912}
}
```
