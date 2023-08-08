from collections import namedtuple

from torch import nn as nn

from datasets import (
    OfflineDatasetConfig,
)
from networks.core import KalmanEstimatorConfig
from networks.ekf import EKFEstimatorConfig
from experiments import ExpConfig
from experiments import train

model_config: KalmanEstimatorConfig
exp_config: ExpConfig


# Configure hyper parameters
hyperparameter_defaults = dict(batch_size=200, learning_rate=3e-3, epochs=100000, latent_dim=4)

HyperParameterConfig = namedtuple(
    "HyperParameterConfig", list(hyperparameter_defaults.keys())
)
hy_config = HyperParameterConfig(**hyperparameter_defaults)

dataset_config = OfflineDatasetConfig(
    traj_len=51,
    num_viz_trajectories=1,
    paths=['./duffing_train.pickle','./duffing_test.pickle'],
)

# EKF settings
model_config = EKFEstimatorConfig(
    is_smooth=True,
    latent_dim=hy_config.latent_dim,
    ctrl_dim=0,
    dataset=dataset_config,
    dyn_hidden_units=64,
    dyn_layers=3,
    dyn_nonlinearity=nn.Softplus(beta=2, threshold=20),
    obs_hidden_units=64,
    obs_layers=3,
    obs_nonlinearity=nn.Softplus(beta=2, threshold=20),
    ramp_iters=200,
    burn_in=100,
    dkl_anneal_iter=1000,
    alpha=0.5,
    beta=1.0,
    atol=1e-9,  # default: 1e-9
    rtol=1e-7,  # default: 1e-7
    z_pred=False,
)

def lr1(step: int, base_lr: float) -> float:
    lr = base_lr
    _lr = lr * 0.975 ** (step // 100)
    return max(_lr, lr * 1e-2)  # default

# experiment settings
exp_config = ExpConfig(
    model=model_config,
    ramp_iters=model_config.ramp_iters,
    batch_size=hy_config.batch_size,
    epochs=hy_config.epochs,
    log_iterations_simple=100,
    log_iterations_images=100,
    base_learning_rate=hy_config.learning_rate,
    learning_rate_function=lr1,
    gradient_clip_max_norm=500,
)
train(exp_config, deterministic=True, load=True)  # train the model
