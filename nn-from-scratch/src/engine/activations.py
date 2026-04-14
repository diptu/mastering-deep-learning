import numpy as np


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))
