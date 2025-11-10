"""Single neuron model with sigmoid activation and MSE computation.
https://www.deep-ml.com/problems/24
"""

from __future__ import annotations
from typing import List, Tuple
import numpy as np


class SingleNeuron: # pylint:disable=too-few-public-methods
    """Implements a single neuron model with sigmoid activation."""

    def single_neuron_model(
        self,
        features: List[List[float]],
        labels: List[int],
        weights: List[float],
        bias: float
    ) -> Tuple[List[float], float]:
        """
        Compute the single neuron output (sigmoid) and mean squared error.

        Args:
            features: 2D list of input features (n_samples x n_features)
            labels: 1D list of target labels (0 or 1)
            weights: 1D list of neuron weights
            bias: float bias term

        Returns:
            Tuple containing:
            - List of predicted probabilities for each sample
            - Mean squared error between predictions and labels
        """
        x_mat: np.ndarray = np.asarray(features, dtype=float)
        w_vec: np.ndarray = np.asarray(weights, dtype=float)
        y_vec: np.ndarray = np.asarray(labels, dtype=float)

        z_vec: np.ndarray = x_mat @ w_vec + bias
        probs_arr: np.ndarray = 1.0 / (1.0 + np.exp(-z_vec))
        probs_arr = np.round(probs_arr, 4)

        mse: float = float(np.round(np.mean((probs_arr - y_vec) ** 2), 4))
        return probs_arr.tolist(), mse


if __name__ == "__main__":
    sol = SingleNeuron()
    result: Tuple[List[float], float] = sol.single_neuron_model(
        features=[[0.5, 1.0], [-1.5, -2.0], [2.0, 1.5]],
        labels=[0, 1, 0],
        weights=[0.7, -0.4],
        bias=-0.1
    )
    assert result == ([0.4626, 0.4134, 0.6682], 0.3349)
    print("✅ All tests passed successfully!")
