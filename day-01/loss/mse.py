from typing import Sequence, Union

import numpy as np


ArrayLike = Union[Sequence[float], np.ndarray]


def mse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Compute Mean Squared Error (MSE).

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        Mean squared error as a float.

    Raises:
        ValueError: If input shapes do not match.
    """
    y_true_arr: np.ndarray = np.asarray(y_true, dtype=float)
    y_pred_arr: np.ndarray = np.asarray(y_pred, dtype=float)

    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    errors: np.ndarray = y_true_arr - y_pred_arr
    squared_errors: np.ndarray = errors**2
    mse_value: float = float(np.mean(squared_errors))

    return mse_value
