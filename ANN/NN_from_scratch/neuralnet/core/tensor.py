# Tensor + gradient
import numpy as np


class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=float)
        self.grad = None
        self.requires_grad = requires_grad

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)
