from typing import List

import numpy as np
import pytest

from models.normal_equation import normal_equation


def test_normal_equation_basic() -> None:
    """Test normal equation with simple 1D input."""
    X: List[float] = [1.0, 2.0, 3.0, 4.0]
    y: List[float] = [2.0, 4.0, 5.0, 7.0]

    theta: np.ndarray = normal_equation(X, y)

    assert theta.shape == (2,)
    assert theta[0] == pytest.approx(0.5, rel=1e-2)
    assert theta[1] == pytest.approx(1.6, rel=1e-2)


def test_normal_equation_numpy_input() -> None:
    """Test normal equation with NumPy arrays."""
    X: np.ndarray = np.array([1, 2, 3, 4], dtype=float)
    y: np.ndarray = np.array([2, 4, 5, 7], dtype=float)

    theta: np.ndarray = normal_equation(X, y)

    assert theta.shape == (2,)
    assert isinstance(theta, np.ndarray)


def test_normal_equation_zero_variance_feature() -> None:
    """Test behavior when feature has zero variance."""
    X = [1.0, 1.0, 1.0, 1.0]
    y = [2.0, 2.0, 2.0, 2.0]

    theta = normal_equation(X, y)

    assert theta.shape == (2,)
    assert theta[0] + theta[1] == pytest.approx(2.0)


def test_normal_equation_shape_mismatch() -> None:
    """Test error raised when X and y shapes do not match."""
    X: List[float] = [1.0, 2.0, 3.0]
    y: List[float] = [1.0, 2.0]

    with pytest.raises(ValueError):
        normal_equation(X, y)
