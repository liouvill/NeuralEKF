from ..data import _trajectories_file
from ._buddy import Buddy
from ._conversions import to_device, to_numpy, to_torch
from ._deprecation import deprecation_wrapper, new_name_wrapper
from ._git import get_git_commit_hash
from ._math import (
    cholesky_inverse,
    cholupdate,
    gaussian_log_prob,
    matrix_dim_from_tril_count,
    quadratic_matmul,
    tril_count_from_matrix_dim,
    tril_from_vector,
    tril_inverse,
    vector_from_tril,
)
from ._module_freezing import freeze_module, unfreeze_module
from ._pdb_safety_net import pdb_safety_net
from ._slice_wrapper import SliceWrapper
from ._squeeze import squeeze
from ._stopwatch import stopwatch

DictIterator = new_name_wrapper(
    "fannypack.utils.DictIterator", "fannypack.utils.SliceWrapper", SliceWrapper
)

TrajectoriesFile = new_name_wrapper(
    "fannypack.utils.TrajectoriesFiles",
    "fannypack.data.TrajectoriesFile",
    _trajectories_file.TrajectoriesFile,
)

__all__ = [
    "Buddy",
    "to_device",
    "to_numpy",
    "to_torch",
    "deprecation_wrapper",
    "new_name_wrapper",
    "get_git_commit_hash",
    "cholesky_inverse",
    "cholupdate",
    "gaussian_log_prob",
    "matrix_dim_from_tril_count",
    "quadratic_matmul",
    "tril_count_from_matrix_dim",
    "tril_from_vector",
    "tril_inverse",
    "vector_from_tril",
    "freeze_module",
    "unfreeze_module",
    "pdb_safety_net",
    "SliceWrapper",
    "squeeze",
    "stopwatch",
]
