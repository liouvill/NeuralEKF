import datetime
import random
import tempfile
import time
from contextlib import nullcontext
from dataclasses import replace
from typing import Tuple

import numpy as np
import torch
from fannypack.utils import Buddy
from torch.utils.data import DataLoader

from utils.log_utils import log_scalars
from utils.net_utils import gradient_norm

from dataclasses import dataclass
from typing import Callable, Optional

from networks.core import KalmanEstimatorConfig

LearningRateScheduler = Callable[[int, float], float]


@dataclass(frozen=True)
class ExpConfig:
    """All parameters for training.

    This must be serializable.
    """

    model: KalmanEstimatorConfig
    ramp_iters: int
    batch_size: int
    epochs: int
    base_learning_rate: float
    gradient_clip_max_norm: Optional[float] = None
    const_var: bool = False
    log_iterations_simple: int = 50
    log_iterations_images: int = 100
    git_commit_hash: Optional[str] = None
    learning_rate_function: Optional[LearningRateScheduler] = None
    name: Optional[str] = None


def default_lr_scheduler(step: int, base_lr: float) -> float:
    lr = base_lr
    _lr = lr * 0.99 ** (step // 10)
    return max(_lr, 1e-4)  # default


def runtime_data_preprocess(
    exp_config: ExpConfig,
    iteration: int,
    batch_t: torch.Tensor,
    batch_y: torch.Tensor,
    batch_u: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Modify training inputs."""
    # scaled traj-length curriculum
    train_length = min((iteration // exp_config.ramp_iters) + 2, len(batch_t))
    filter_length = min(
        (iteration // exp_config.ramp_iters) + 2, exp_config.model.dataset.traj_len
    )
    if train_length < len(batch_t):
        # extract random subset of data
        start_ind = np.random.randint(0, len(batch_t) - train_length)
        batch_t = batch_t[start_ind : (start_ind + train_length)]
        batch_y = batch_y[start_ind : (start_ind + train_length)]
        batch_u = batch_u[start_ind : (start_ind + train_length)]
    return batch_t, batch_y, batch_u, filter_length


def train(
    exp_config: ExpConfig,
    debug: bool = False,
    test: bool = False,
    save: bool = True,
    load: bool = True,
    deterministic: bool = True,
) -> None:
    """Trains a network.

    Parameters
    ----------
    exp_config : ExpConfig
        Specifies all information regarding the experiment.
    debug : bool, default=False
        Flag for debugging mode.
    test : bool, default=False
        Flag for testing.
    save : bool, default=True
        Flag for saving model after training.
    deterministic: bool, default=True
        Fix the seed so experiments are deterministic.
    """
    if deterministic:
        np.random.seed(0)
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(0)

    if debug:
        cm = torch.autograd.detect_anomaly
    else:
        cm = nullcontext  # type: ignore

    # loading custom lr and loss schedulers
    learning_rate = (
        default_lr_scheduler
        if exp_config.learning_rate_function is None
        else exp_config.learning_rate_function
    )
    estimator = exp_config.model.create()

    # set up buddy
    # some hacks to get around serialization
    # TODO figure out a better way to do this
    metadata_dir = "metadata"
    dir = tempfile.gettempdir()
    if exp_config.name is None:
        name = (
            f"{exp_config.model.dataset.__class__.__name__}_{exp_config.model.__class__.__name__}_"
            + f"{datetime.datetime.now().strftime('%a-%H-%M-%S')}"
        )
        exp_config = replace(exp_config, name=name)
    assert exp_config.name is not None
    buddy = Buddy(exp_config.name, estimator, optimizer_type="adam", metadata_dir=dir)
    buddy._metadata_dir = metadata_dir
    # hack, replace input metadata with current metadata
    buddy._metadata = exp_config  # type: ignore
    buddy._write_metadata()  # Write to disk
    buddy.set_learning_rate(lambda x: learning_rate(x, exp_config.base_learning_rate))

    # process data
    batch_size = 1 if debug else exp_config.batch_size
    dataset = exp_config.model.dataset.create()
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    vis_data = dataset.get_viz_data(buddy.device)  # type: ignore
    if load:
        buddy.load_checkpoint(path="checkpoints/DynamicSystemDatasetConfig_EKFEstimatorConfig_Fri-11-32-10-0000000000384795.ckpt")

    # training loop
    iteration = 0
    for e in range(exp_config.epochs):
        for batch in dataloader:
            start_time = time.time()
            #print(estimator._modules)
            batch_t, batch_y, batch_u = dataset.preprocess_data(batch, buddy.device)
            batch_t, batch_y, batch_u, filter_length = runtime_data_preprocess(
                exp_config, iteration, batch_t, batch_y, batch_u
            )

            # compute loss
            with cm():
                loss = estimator.loss(batch_t, batch_y, batch_u, iteration, avg=True)
            buddy.minimize(loss, clip_grad_max_norm=exp_config.gradient_clip_max_norm)

            # logging
            print(f"Iteration: {iteration} \t Loss: {loss.item():.2f}")
            if iteration % exp_config.log_iterations_simple == 0:
                log_scalars(
                    buddy,
                    {
                        "Total_Loss": loss.item(),
                        "Learning_Rate": buddy.get_learning_rate(),
                        "Iteration_Wall_Time": time.time() - start_time,
                        "Gradient_Norm": gradient_norm(estimator),
                    },
                    scope="Train",
                )
            if iteration % exp_config.log_iterations_images == 0:
                with torch.no_grad():
                    estimator.eval()
                    filter_length = min(filter_length, vis_data.t.shape[0] // 3)
                    estimator.log(buddy, vis_data, filter_length=2)
                    estimator.train()
            iteration += 1

            if test:
                break

    # save model after training
    if save:
        buddy.save_checkpoint(label="final")
