import numpy as np
from visual import visualize_features


class Dense:
    def __init__(self, in_features: int, out_features: int):
        self.weights = 0.01 * np.random.randn(in_features, out_features)
        self.biases = np.zeros((1, out_features))
        self.output = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.output = np.dot(x, self.weights) + self.biases
        return self.output
