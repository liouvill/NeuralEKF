from typing import Optional, Union

import numpy as np
import torch


def cholesky_inverse(u: torch.Tensor, upper: bool = False) -> torch.Tensor:
    """Alternative to `torch.cholesky_inverse()`, with support for batch dimensions.

    Relevant issue tracker: https://github.com/pytorch/pytorch/issues/7500

    Args:
        u (torch.Tensor): Triangular Cholesky factor. Shape should be `(*, N, N)`.
        upper (bool, optional): Whether to consider the Cholesky factor as a lower or
            upper triangular matrix.

    Returns:
        torch.Tensor:
    """
    if u.dim() == 2 and not u.requires_grad:
        return torch.cholesky_inverse(u, upper=upper)
    return torch.cholesky_solve(torch.eye(u.size(-1)).expand(u.size()), u, upper=upper)


def cholupdate(
    L: torch.Tensor,
    x: torch.Tensor,
    weight: Optional[Union[torch.Tensor, float]] = None,
) -> torch.Tensor:
    """Batched rank-1 Cholesky update.

    Computes the Cholesky decomposition of `RR^T + weight * xx^T`.

    Args:
        L (torch.Tensor): Lower triangular Cholesky decomposition of a PSD matrix. Shape
            should be `(*, matrix_dim, matrix_dim)`.
        x (torch.Tensor): Rank-1 update vector. Shape should be `(*, matrix_dim)`.
        weight (torch.Tensor or float, optional): Set to -1 for "downdate". Shape must
            be broadcastable with `(*, matrix_dim)`.

    Returns:
        torch.Tensor: New L matrix. Same shape as L.
    """
    # Expected shapes: (*, dim, dim) and (*, dim)
    batch_dims = L.shape[:-2]
    matrix_dim = x.shape[-1]
    assert x.shape[:-1] == batch_dims
    assert matrix_dim == L.shape[-1] == L.shape[-2]

    # Flatten batch dimensions, and clone for tensors we need to mutate
    L = L.reshape((-1, matrix_dim, matrix_dim))
    x = x.reshape((-1, matrix_dim)).clone()
    L_out_cols = []

    sign: Union[float, torch.Tensor]
    if weight is None:
        sign = L.new_ones((1,))
    elif isinstance(weight, float):
        x = x * np.sqrt(np.abs(weight))
        sign = float(np.sign(weight))
    else:
        x = x * torch.sqrt(torch.abs(weight))
        sign = torch.sign(weight)

    # Cholesky update; mostly copied from Wikipedia:
    # https://en.wikipedia.org/wiki/Cholesky_decomposition
    for k in range(matrix_dim):
        r = torch.sqrt(L[:, k, k] ** 2 + sign * x[:, k] ** 2)
        c = (r / L[:, k, k])[:, None]
        s = (x[:, k] / L[:, k, k])[:, None]

        # We build output column-by-column to avoid in-place modification errors
        L_out_col = torch.zeros_like(L[:, :, k])
        L_out_col[:, k] = r
        L_out_col[:, k + 1 :] = (L[:, k + 1 :, k] + sign * s * x[:, k + 1 :]) / c
        L_out_cols.append(L_out_col)

        # We clone x at each iteration, also to avoid in-place modification errors
        x_next = x.clone()
        x_next[:, k + 1 :] = c * x[:, k + 1 :] - s * L_out_col[:, k + 1 :]
        x = x_next

    # Stack columns together
    L_out = torch.stack(L_out_cols, dim=2)

    # Unflatten batch dimensions and return
    return L_out.reshape(batch_dims + (matrix_dim, matrix_dim))


def quadratic_matmul(x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    r"""Computes $x^\top A x$, with support for arbitrary batch axes.

    Stolen from @alberthli/@wuphilipp.

    Args:
        x (torch.Tensor): Vectors. Shape should be `(*, D)`.
        A (torch.Tensor): Matrices. Shape should be `(*, D, D)`.

    Returns:
        torch.Tensor: Batched output of multiplication. Shape should be `(*)`.
    """
    assert x.shape[-1] == A.shape[-1] == A.shape[-2]

    x_T = x.unsqueeze(-2)  # shape=(*, 1, X)
    x_ = x.unsqueeze(-1)  # shape(*, X, 1)
    quadratic = x_T @ A @ x_  # shape=(*, 1, 1)
    return quadratic.squeeze(-1).squeeze(-1)


def gaussian_log_prob(
    mean: torch.Tensor, covariance: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    """Computes log probabilities under multivariate Gaussian distributions,
    with support for arbitrary batch axes.

    Naive version of...
    ```
    torch.distributions.MultivariateNormal(
        mean, covariance
    ).log_prob(value)
    ```
    that avoids some Cholesky-related CUDA errors.
    https://discuss.pytorch.org/t/cuda-illegal-memory-access-when-using-batched-torch-cholesky/51624

    Stolen from @alberthli/@wuphilipp.

    Args:
        mean (torch.Tensor): Means vectors. Shape should be `(*, D)`.
        covariance (torch.Tensor): Covariances matrices. Shape should be `(*, D, D)`.
        value (torch.Tensor): State vectors. Shape should be `(*, D)`.

    Returns:
        torch.Tensor: Batched log probabilities. Shape should be `(*)`.
    """
    D = mean.shape[-1]
    assert covariance.shape[:-1] == mean.shape == value.shape

    exponential = quadratic_matmul(value - mean, torch.inverse(covariance))
    other_terms = torch.logdet(covariance) + D * np.log(2.0 * np.pi)
    log_p = -0.5 * (exponential + other_terms)
    return log_p


def matrix_dim_from_tril_count(tril_count: int):
    """Computes the dimension of a lower triangular square matrix given a count of its
    lower-triangular components.

    Args:
        tril_count (int): Count of lower-triangular terms.
    Returns:
        int: Dimension of square matrix.
    """
    matrix_dim = int(0.5 * (1 + 8 * tril_count) ** 0.5 - 0.5)
    return matrix_dim


def tril_count_from_matrix_dim(matrix_dim: int):
    """Computes the number of lower triangular terms in a square matrix of a given
    dimension `(matrix_dim, matrix_dim)`.

    Args:
        matrix_dim (int): Dimension of square matrix.
    Returns:
        int: Count of lower-triangular terms.
    """
    tril_count = (matrix_dim ** 2 - matrix_dim) // 2 + matrix_dim
    return tril_count


def tril_from_vector(lower_vector: torch.Tensor) -> torch.Tensor:
    """Computes lower-triangular square matrices from a flattened vector of nonzero
    terms. Supports arbitrary batch dimensions.

    Args:
        lower_vector (torch.Tensor): Vectors containing the nonzero terms of a
            square lower-triangular matrix. Shape should be `(*, tril_count)`.
    Returns:
        torch.Tensor: Square matrices. Shape should be `(*, matrix_dim, matrix_dim)`.
    """
    batch_dims = lower_vector.shape[:-1]
    tril_count = lower_vector.shape[-1]
    matrix_dim = matrix_dim_from_tril_count(tril_count)

    output = torch.zeros(batch_dims + (matrix_dim, matrix_dim))
    tril_indices = torch.tril_indices(matrix_dim, matrix_dim)
    output[..., tril_indices[0], tril_indices[1]] = lower_vector
    return output


def vector_from_tril(tril_matrix: torch.Tensor) -> torch.Tensor:
    """Retrieves the lower triangular terms of square matrices as vectors. Supports
    arbitrary batch dimensions.

    Args:
        tril_matrix (torch.Tensor): Square matrices. Shape should be
            `(*, matrix_dim, matrix_dim)`
    Returns:
        torch.Tensor: Flattened vectors. Shape should be `(*, tril_count)`.
    """
    matrix_dim = tril_matrix.shape[-1]
    assert tril_matrix.shape[-2] == matrix_dim

    tril_indices = torch.tril_indices(matrix_dim, matrix_dim)
    return tril_matrix[..., tril_indices[0], tril_indices[1]]


def tril_inverse(tril_matrix: torch.Tensor) -> torch.Tensor:
    """Invert a lower-triangular matrix.

    Args:
        tril_matrix (torch.Tensor): Lower-triangular matrix to invert. Shape should be
            `(*, matrix_dim, matrix_dim)`.

    Returns:
        torch.Tensor: Inverted matrix. Shape should be `(*, matrix_dim, matrix_dim)`.
    """
    assert tril_matrix.shape[-1] == tril_matrix.shape[-2], "Input must be square!"
    matrix_dim = tril_matrix.shape[-1]
    batch_dims = tril_matrix.shape[:-2]

    # Get an identity matrix
    identity = torch.eye(tril_matrix.shape[-1], device=tril_matrix.device)

    # View identity w/ batch dimensions
    identity = identity.reshape(
        (1,) * len(batch_dims) + (matrix_dim, matrix_dim)
    ).expand(tril_matrix.shape)

    # Invert with triangular solve
    inverse = torch.triangular_solve(identity, tril_matrix, upper=False).solution
    assert inverse.shape == tril_matrix.shape
    return inverse
