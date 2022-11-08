# Neural Extended Kalman Filters (Neural EKF)
 
This repository contains codes and data for the following publication:
* Wei Liu, Zhilu Lai, Kiran Bacsa, and Eleni Chatzi (2022). [Neural Extended Kalman Filters for Learning and Predicting Dynamics of Structural Systems](https://arxiv.org/abs/2210.04165).

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
@article{liu2022neural,
  title={Neural Extended Kalman Filters for Learning and Predicting Dynamics of Structural Systems},
  author={Liu, Wei and Lai, Zhilu and Bacsa, Kiran and Chatzi, Eleni},
  journal={arXiv preprint arXiv:2210.04165},
  year={2022}
}
```
