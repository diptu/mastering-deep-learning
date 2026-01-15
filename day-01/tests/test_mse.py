from typing import List

import numpy as np
import pytest

from loss.mse import mse


def test_mse_basic() -> None:
    """Test MSE with simple list inputs."""
    y_true: List[float] = [3.0, -0.5, 2.0, 7.0]
    y_pred: List[float] = [2.5, 0.0, 2.0, 8.0]

    result: float = mse(y_true, y_pred)

    assert result == pytest.approx(0.375)


def test_mse_zero_error() -> None:
    """Test MSE when predictions perfectly match ground truth."""
    y_true: List[int] = [1, 2, 3]
    y_pred: List[int] = [1, 2, 3]

    result: float = mse(y_true, y_pred)

    assert result == 0.0


def test_mse_numpy_arrays() -> None:
    """Test MSE with NumPy array inputs."""
    y_true: np.ndarray = np.array([1.0, 2.0, 3.0], dtype=float)
    y_pred: np.ndarray = np.array([2.0, 2.0, 4.0], dtype=float)

    result: float = mse(y_true, y_pred)

    expected: float = 2.0 / 3.0
    assert result == pytest.approx(expected)


def test_mse_shape_mismatch() -> None:
    """Test MSE raises ValueError when shapes do not match."""
    y_true: List[int] = [1, 2, 3]
    y_pred: List[int] = [1, 2]

    with pytest.raises(ValueError):
        mse(y_true, y_pred)
