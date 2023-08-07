from dataclasses import replace
from typing import Dict

import numpy as np
import torch
from fannypack.utils import Buddy, to_numpy

from datasets import VisData
#from networks.core import KalmanEstimator
from utils.net_utils import reparameterize_gauss
from utils.plot_utils import PlotHandler as ph

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from matplotlib import pyplot as plt

DIM_TO_NAME = {0: "x1", 1: "x2", 2: "x3", 3: "x4", 4: "x5", 5: "x6", 6: "x7", 7: "x8"}


def _log_timeseries(
    buddy: Buddy,
    viz: VisData,
    y_f: torch.Tensor,
    y_p: torch.Tensor,
    cov_f: torch.Tensor,
    cov_p: torch.Tensor,
    filter_length: int,
    dim: int,
    name: str,
) -> None:
    """Helper to log timeseries compared data.

    Parameters
    ----------
    buddy : Buddy
        Buddy helper.
    viz : VisData
        Visualization data.
    y_f : torch.Tensor, shape=(T, B, p)
        Filtered samples.
    y_p : torch.Tensor, shape=(T, B, p)
        Predicted samples.
    filter_length : int
        Number of points to filter over (rest are prediction baselines).
    dim : int
        Dimension to plot (x or y for 2D data).
    name : str
        Name of plot.
    """
    pred_len = len(y_p)
    with ph.plot_context(viz.plot_settings, size=[8,8]) as (fig, ax):
        ph.plot_timeseries_compare(
            [
                viz.np_t[:filter_length],
                viz.np_t[(filter_length - 1) : (pred_len + filter_length - 1)],
                viz.np_t[:filter_length],
                viz.np_t[(filter_length - 1) : (pred_len + filter_length - 1)],
            ],
            [
                viz.np_y[:filter_length, :, dim],
                viz.np_y[(filter_length - 1) : (pred_len + filter_length - 1), :, dim],
                y_f[..., dim],
                y_p[..., dim],
            ],
            [
                np.zeros_like(viz.np_y[:filter_length, :, dim]),
                np.zeros_like(viz.np_y[(filter_length - 1) : (pred_len + filter_length - 1), :, dim]),
                cov_f[..., dim, dim],
                cov_p[..., dim, dim],
            ],
            DIM_TO_NAME[dim],
            style_list=["-", "-", "-", "--"],
            startmark_list=["o", None, "o", None],
            endmark_list=[None, "x", None, "x"],
            color_list=["silver", "silver", "b", "b"],
            linewidth_list=[3, 3, 1, 1],
            label_list=[None, "ground-truth", None, "prediction"]
        )
        p_img = ph.plot_as_image(fig)

    _name = f"{name}-prediction-trajectory-{DIM_TO_NAME[dim]}t"
    buddy.log_image(_name, p_img)


# =============================================================================
# def _log_timeseries_subplots(
#     buddy: Buddy,
#     viz: VisData,
#     y_f: torch.Tensor,
#     y_p: torch.Tensor,
#     cov_f: torch.Tensor,
#     cov_p: torch.Tensor,
#     filter_length: int,
#     name: str,
# ) -> None:
#     """Helper to log timeseries compared data.
# 
#     Parameters
#     ----------
#     buddy : Buddy
#         Buddy helper.
#     viz : VisData
#         Visualization data.
#     y_f : torch.Tensor, shape=(T, B, p)
#         Filtered samples.
#     y_p : torch.Tensor, shape=(T, B, p)
#         Predicted samples.
#     filter_length : int
#         Number of points to filter over (rest are prediction baselines).
#     dim : int
#         Dimension to plot (x or y for 2D data).
#     name : str
#         Name of plot.
#     """
#     pred_len = len(y_p)
#     with ph.plot_context(sp_shape=[1,2], size=[24,6]) as (fig, axs):
#         plt.rc('font', family='Times New Roman')
#         for i in range(2):
#             #axs[0].set_title(f"$a_i$}")
#             axs[i].set_ylabel(f"$x_{i+1}$", fontdict={'family' : 'Times New Roman', 'size' : 12})
#             axs[i].plot(viz.np_t[:filter_length], viz.np_y[:filter_length,:,i], color="black", linestyle="-", label="true response", lw=1)
#             axs[i].plot(viz.np_t[filter_length-1:filter_length-1+pred_len], viz.np_y[filter_length-1:filter_length-1+pred_len,:,i], color="black", linestyle="-", label="true response", lw=1)
#             axs[i].plot(viz.np_t[:filter_length], y_f[...,i], color="red", linestyle="--", label="prediction", lw=1)
#             axs[i].plot(viz.np_t[filter_length-1:filter_length-1+pred_len], y_p[...,i], color="red", linestyle="--", label="prediction", lw=1)
#             axs[i].set_xlabel("Time [sec]", fontdict={'family' : 'Times New Roman', 'size' : 12})
# 
#             handles, labels = axs[i].get_legend_handles_labels()
#             by_label = dict(zip(labels, handles))
#             if i == 0:
#                 axs[i].legend(by_label.values(), by_label.keys(), loc="lower left", bbox_to_anchor= (0.0, 1.01), ncol= 2, prop={'family' : 'Times New Roman', 'size' : 12})
#             #axs[i].legend(loc="upper right", ncol= 1, prop={'family' : 'Times New Roman', 'size' : 60})
#             #axs[i].tick_params(axis="x", labelsize=60)
#             #axs[i].set_xticks([0,2,4,6,8,10])#axs[i].get_xticks())
#             #axs[i].set_yticks(axs[i].get_yticks())
#             #axs[i].set_xticklabels(axs[i].get_xticks(), fontdict={'family' : 'Times New Roman', 'size' : 120})
#             #axs[i].set_yticklabels(axs[i].get_yticks(), fontdict={'family' : 'Times New Roman', 'size' : 120})
#             #temp = np.max(np.abs(viz.np_y[:pred_len,:,i]))
#             #axs[i].set_ylim((-1.05*temp, 0.85*temp))
# 
#         p_img = ph.plot_as_image(fig)
# 
#     _name = f"{name}-prediction-trajectory-t"
#     buddy.log_image(_name, p_img)
# =============================================================================
def _log_timeseries_subplots(
    buddy: Buddy,
    viz: VisData,
    y_f: torch.Tensor,
    y_p: torch.Tensor,
    cov_f: torch.Tensor,
    cov_p: torch.Tensor,
    filter_length: int,
    name: str,
) -> None:
    """Helper to log timeseries compared data.

    Parameters
    ----------
    buddy : Buddy
        Buddy helper.
    viz : VisData
        Visualization data.
    y_f : torch.Tensor, shape=(T, B, p)
        Filtered samples.
    y_p : torch.Tensor, shape=(T, B, p)
        Predicted samples.
    filter_length : int
        Number of points to filter over (rest are prediction baselines).
    dim : int
        Dimension to plot (x or y for 2D data).
    name : str
        Name of plot.
    """
    pred_len = len(y_p)
    with ph.plot_context(sp_shape=[2,1], size=[70,50]) as (fig, axs):
        plt.rc('font', family='Times New Roman')
        plt.rcParams["mathtext.fontset"] = "cm"
        for i in range(2):
            #axs[0].set_title(f"$a_i$}")
            axs[i].set_ylabel(f"$x_{i+1}$", fontdict={'family' : 'Times New Roman', 'size' : 180})
            axs[i].plot(viz.np_t[:filter_length], viz.np_y[:filter_length,:,i], color="black", linestyle="-", label="true response", lw=20)
            axs[i].plot(viz.np_t[filter_length-1:filter_length-1+pred_len], viz.np_y[filter_length-1:filter_length-1+pred_len,:,i], color="black", linestyle="-", label="true response", lw=20)
            axs[i].plot(viz.np_t[:filter_length], y_f[...,i], color="red", linestyle="--", label="prediction", lw=20)
            axs[i].plot(viz.np_t[filter_length-1:filter_length-1+pred_len], y_p[...,i], color="red", linestyle="--", label="prediction", lw=20)

            handles, labels = axs[i].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            if i == 0:
                axs[i].legend(by_label.values(), by_label.keys(), loc="lower left", bbox_to_anchor= (0.0, 1.01), ncol= 2, prop={'family' : 'Times New Roman', 'size' : 180})
            if i == 1:
                axs[i].set_xlabel("Time [sec]", fontdict={'family' : 'Times New Roman', 'size' : 180})
            #axs[i].legend(loc="upper right", ncol= 1, prop={'family' : 'Times New Roman', 'size' : 60})
            axs[i].set_xticks([0,2,4,6,8,10])#axs[i].get_xticks())
            axs[i].set_yticks([-3,-2,-1,0,1,2])#axs[i].get_yticks())
            axs[i].set_xticklabels(axs[i].get_xticks(), fontdict={'family' : 'Times New Roman', 'size' : 180})
            axs[i].set_yticklabels(axs[i].get_yticks(), fontdict={'family' : 'Times New Roman', 'size' : 180})
            temp = np.max(np.abs(viz.np_y[:pred_len,:,i]))
            axs[i].set_ylim((-1.1*temp, 1.1*temp))

        p_img = ph.plot_as_image(fig)

    _name = f"{name}-prediction-trajectory-t"
    buddy.log_image(_name, p_img)


def _log_filter(
    buddy: Buddy,
    viz: VisData,
    y_mu_f: torch.Tensor,
    filter_length: int,
    dim: int,
    name: str,
) -> None:
    """Helper to log filter timeseries compared data.

    Parameters
    ----------
    See _log_timeseries().
    """
    with ph.plot_context(viz.plot_settings) as (fig, ax):
        ph.plot_timeseries_compare(
            [viz.np_t, viz.np_t[:filter_length]],
            [viz.np_y[..., dim], y_mu_f[..., dim]],
            DIM_TO_NAME[dim],
            style_list=["-", "-"],
            startmark_list=["o", "o"],
            endmark_list=["x", "x"],
            color_list=["silver", "b"],
            linewidth_list=[3, 1],
        )
        p_img = ph.plot_as_image(fig)
    _name = f"{name}-{DIM_TO_NAME[dim]}"
    buddy.log_image(_name, p_img)


def log_scalars(buddy: Buddy, scalar_dict: Dict[str, float], scope=None):
    if scope is not None:
        buddy.log_scope_push(scope)
    for name, value in scalar_dict.items():
        buddy.log_scalar(name, value)
    if scope is not None:
        buddy.log_scope_pop(scope)


def log_image(buddy: Buddy, image: np.ndarray, name: str, scope=None):
    if scope is not None:
        buddy.log_scope_push(scope)
    buddy.log_image(name, image)
    if scope is not None:
        buddy.log_scope_pop(scope)


def log_basic(
    estimator,
    buddy: Buddy,
    viz: VisData,
    filter_length: int = 1,
    smooth: bool = False,
    ramp_pred: bool = False,
) -> None:
    """Log basic visual information for Gaussian estimators with filter ONLY.

    Parameters
    ----------
    estimator : Estimator
        The estimator.
    buddy : Buddy
        Buddy helper for training.
    viz : VisData
        Visualization data.
    filter_length : int, default=1
        Length of data to provide for filtering during prediction runs.
    smooth : bool, default=False
        Flag indicating whether estimator should smooth.
    ramp_pred : bool, default=False
        Flag indicating whether to ramp pred horizon. Used for pred visualizations early
        on in KF training when it is numerically unstable.
    """
    assert filter_length >= 1
    filter_length = min(filter_length, len(viz.np_t))
    data_var = 0.0  # variance of independent Gaussian injected noise

    # ramp the visualizations
    if ramp_pred and hasattr(estimator, "_ramp_iters"):
        it = buddy.optimizer_steps
        idx_p = min((it // estimator._ramp_iters) + filter_length + 1, len(viz.t))  # type: ignore
    else:
        idx_p = None

    # ---------- #
    # PREDICTION #
    # ---------- #

    # filtered portion
    z0 = estimator.get_initial_hidden_state(viz.y0.shape[0])
    noise = torch.randn_like(viz.y[0:filter_length]) * np.sqrt(data_var)
    z_mu_f, z_cov_f = estimator(
        viz.t[:filter_length],
        viz.y[:filter_length] + noise,
        viz.u[:filter_length],
        z0,
        return_hidden=True,
    )
    y_mu_f, y_cov_f = estimator(
        viz.t[:filter_length], viz.y[:filter_length] + noise, viz.u[:filter_length], z0,
    )

    # smooth if possible
    if smooth:
        z_mu_s, z_cov_s = estimator.get_smooth()  # type: ignore
        y_mu_s, y_cov_s = estimator.latent_to_observation(viz.t[:filter_length], z_mu_s, z_cov_s)
    else:
        z_mu_s = z_mu_f
        z_cov_s = z_cov_f
        y_mu_s = y_mu_f
        y_cov_s = y_cov_f

    # predicting
    y_mu_p, y_cov_p = estimator.predict(
        viz.t[(filter_length - 1) : idx_p],
        z_mu_s[-1],
        z_cov_s[-1],
        viz.t[(filter_length - 1) : idx_p],
        viz.u[(filter_length - 1) : idx_p],
    )

    z_mu_p, z_cov_p = estimator.predict(
        viz.t[(filter_length - 1) : idx_p],
        z_mu_s[-1],
        z_cov_s[-1],
        viz.t[(filter_length - 1) : idx_p],
        viz.u[(filter_length - 1) : idx_p],
        return_hidden = True,
    )

    # sampling from observation distributions
    y_mu_s = to_numpy(y_mu_s)
    y_mu_p = to_numpy(y_mu_p)
    z_mu_s = to_numpy(z_mu_s)
    z_mu_p = to_numpy(z_mu_p)
    y_cov_s = to_numpy(y_cov_s)
    y_cov_p = to_numpy(y_cov_p)
    z_cov_s = to_numpy(z_cov_s)
    z_cov_p = to_numpy(z_cov_p)

    np.save("y_s", y_mu_s)
    np.save("y_p", y_mu_p)
    np.save("z_s", z_mu_s)
    np.save("z_p", z_mu_p)

    # log RMSE
    pred_len = len(y_mu_p[:, :, 0])
    RMSE_1 = np.sqrt(mean_squared_error(viz.np_y[0 : pred_len, :, 0], y_mu_p[:, :, 0]))
    RMSE_2 = np.sqrt(mean_squared_error(viz.np_y[0 : pred_len, :, 1], y_mu_p[:, :, 1]))

    with buddy.log_scope("metrics"):
        buddy.log_scalar("RMSE1", RMSE_1)
        buddy.log_scalar("RMSE2", RMSE_2)

    # log prediction vs. ground truth
    with buddy.log_scope("0_predict"):
        # plotting means versus ground truth
        _log_timeseries_subplots(buddy, viz, y_mu_s, y_mu_p, y_cov_s, y_cov_p, filter_length, "means")
