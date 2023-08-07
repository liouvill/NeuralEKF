# Neural Extended Kalman Filters (Neural EKF)
 
This repository contains codes and data for the following publication:
* Wei Liu, Zhilu Lai, Kiran Bacsa, and Eleni Chatzi (2023). [Neural extended Kalman filters for learning and predicting dynamics of structural systems](https://doi.org/10.1177/14759217231179912). *Structural Health Monitoring*.

The Neural EKF is a generalized version of the conventional EKF, where the modeling of process dynamics and sensory observations can be parameterized by neural networks, therefore learned by end-to-end training. The method is implemented under the variational inference framework with the EKF conducting inference from sensing measurements.

## Setup
python>=3.10 is recommended.

To install the required packages to run this repository, run:
```
pip install -r requirements.txt
```

## Repository Overview
 * `duffing_data` - Simulated data and saved results for Duffing oscillators with observation noises 0.1, 0.01 and 0.001, repectively. The data and checkpoint in the root folder are for the 0.1 noise case, same as those in the subfolder `duffing_data/0.1`.
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
 * `experiments.py` - Manages training and evaluation of models.
 * `plot_duffing.py` - Visualizes results as in the paper.
 * `train.py` - Configuration for training Neural EKF model.
### Training and evaluation
Training is loaded and resumed from a saved checkpoint (or set ``load=False`` in the training configuration to start from scratch). Due to different randomization of the resumed training process, it takes a while to converge and reproduce results as in the paper.

[Tensorboard](https://www.tensorflow.org/tensorboard) is used to track model metrics and visualizations, and its logs will be automatically created and saved during training in the `logs` folder.

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
