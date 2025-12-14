import numpy as np


class ReLU:
    def __init__(self):
        self.output = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.output = np.maximum(0, x)
        return self.output


class SoftMax:
    def forward(self, x):
        shiftx = x - x.max(axis=1, keepdims=True)

        # 2. Standard Softmax calculation using the shifted values.
        exp_values = np.exp(shiftx)

        # 3. Normalization: Sum the exponentials along the class dimension (axis=1)
        norm_val = np.sum(exp_values, axis=1, keepdims=True)

        # Optional: Print for debugging (removed in production code)
        # print(f"norm_val: {norm_val} ")

        prob = exp_values / norm_val
        return prob
