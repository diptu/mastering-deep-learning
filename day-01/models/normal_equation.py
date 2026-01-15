"""
Linear Regression using the Normal Equation.

Given the cost function:
    J(θ) = (1 / m) * Σ (Xθ - y)²

At the minimum:
    ∂J(θ) / ∂θ = 0

Closed-form solution:
    θ = (XᵀX)⁻¹Xᵀy
"""

from typing import Sequence, Union

import numpy as np


ArrayLike = Union[Sequence[float], np.ndarray]


def normal_equation(X: ArrayLike, y: ArrayLike) -> np.ndarray:
    """
    Compute linear regression parameters using the normal equation.

    Args:
        X: Feature values of shape (n_samples,) or (n_samples, 1).
        y: Target values of shape (n_samples,).

    Returns:
        Parameter vector θ of shape (2, 1) → [bias, weight].

    Raises:
        ValueError: If input shapes are incompatible.
    """
    X_arr: np.ndarray = np.asarray(X, dtype=float)
    y_arr: np.ndarray = np.asarray(y, dtype=float)

    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)

    if y_arr.ndim != 1 or X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("X and y must have compatible shapes")

    # Add bias term (column of ones)
    ones: np.ndarray = np.ones((X_arr.shape[0], 1))
    X_b: np.ndarray = np.hstack((ones, X_arr))

    # Normal Equation (numerically safer than explicit inverse)
    theta: np.ndarray = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y_arr

    return theta
