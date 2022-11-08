import pickle
from abc import ABC
from dataclasses import dataclass
from typing import List, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.data_utils import prep_batch
from utils.plot_utils import Axis, PlotSettings

# ------------------------- #
# VISUALIZATION DATACLASSES #
# ------------------------- #


@dataclass(frozen=True)
class VisData:
    """Struct to hold data for plotting."""

    t: torch.Tensor
    y0: torch.Tensor
    y: torch.Tensor
    u: torch.Tensor
    x: torch.Tensor
    v: torch.Tensor
    np_t: np.ndarray
    np_y: np.ndarray
    np_u: np.ndarray
    np_x: np.ndarray
    np_v: np.ndarray
    plot_settings: PlotSettings


# ------- #
# HELPERS #
# ------- #


def _get_viz_data_basic(dataset, device: torch.device) -> VisData:
    """Helper for returning only times and associated data, no extra info."""
    assert hasattr(dataset, "_viz_data")
    assert callable(getattr(dataset, "get_default_plot_settings", None))

    t_list = [t for t, x, u in dataset._viz_data]  # type: ignore
    x_list = [x for t, x, u in dataset._viz_data]  # type: ignore
    u_list = [u for t, x, u in dataset._viz_data]  # type: ignore

    t = torch.tensor(t_list, dtype=torch.float, device=device)[0, :]
    y = torch.tensor(x_list, dtype=torch.float, device=device).transpose(0, 1)
    u = torch.tensor(u_list, dtype=torch.float, device=device).transpose(0, 1)
    x = y[...,[0,1]]
    v = y[...,[0,1]]
    y = y[...,[0,1]]

    return VisData(
        t=t,
        y0=y[0],
        y=y,
        u=u,
        x=x,
        v=v,
        np_t=t.clone().cpu().numpy(),
        np_y=y.clone().cpu().numpy(),
        np_u=u.clone().cpu().numpy(),
        np_x=x.clone().cpu().numpy(),
        np_v=v.clone().cpu().numpy(),
        plot_settings=dataset.get_default_plot_settings(),
    )


# -------- #
# DATASETS #
# -------- #


@dataclass
class DatasetConfig:
    """Configuration for a Dataset for training."""

    traj_len: int
    num_viz_trajectories: int

    @property
    def obs_shape(self) -> Tuple[int, ...]:
        """Observation dimension."""
        raise NotImplementedError

    def create(self):
        """Create a `DynamicsDataset`."""
        raise NotImplementedError


@dataclass
class OfflineDatasetConfig(DatasetConfig):
    """Dataset for training that reads from a saved pickle file."""

    paths: Tuple[str, str]

    @property
    def obs_shape(self) -> Tuple[int, ...]:
        """Observation dimension."""
        return (2,)

    def create(self):
        """Create a `DynamicsDataset`."""
        dataset = OfflineDataset(
            self.traj_len, self.num_viz_trajectories, self.paths[0], self.paths[1],
        )
        return dataset


class OfflineDataset(ABC, Dataset):
    """Dataset for DE-defined continuous dynamical systems."""

    def __init__(
        self,
        traj_len: int,
        num_viz_trajectories: int,
        train_data_path: str,
        val_data_path: str,
    ) -> None:
        """Initalize an offline saved dataset."""
        with open(train_data_path, "rb") as handle:
            self._data = pickle.load(handle)
        assert len(self._data[0][0]) >= traj_len
        self._data = [
            (t[:traj_len], x[:traj_len], c[:traj_len]) for (t, x, c) in self._data
        ]
        with open(val_data_path, "rb") as handle:
            self._viz_data = pickle.load(handle)[:num_viz_trajectories]

    def __len__(self) -> int:
        """Length of dataset."""
        return len(self._data)

    def __getitem__(self, idx: Union[int, torch.Tensor, List[int]]) -> torch.Tensor:
        """Get specific datapoint."""
        if torch.is_tensor(idx):
            assert isinstance(idx, torch.Tensor)  # mypy
            idx = idx.tolist()
        # TODO fix
        return self._data[idx]  # type: ignore

    def get_viz_data(self, device: torch.device) -> VisData:
        """Get a VisData object of the viz dataset."""
        return _get_viz_data_basic(self, device)

    def get_default_plot_settings(self) -> PlotSettings:
        """Get plot settings for each system."""
        return PlotSettings(axis=Axis(xlim=(-np.pi - 0.1, np.pi + 0.1), ylim=(-7, 7),))

    def preprocess_data(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """See parent."""
        _, batch_t, batch_y, batch_u = prep_batch(batch, device)
        return batch_t, batch_y, batch_u

