import numpy as np
from neuralnet.core.module import Module
from neuralnet.core.parameter import Parameter


class Dense(Module):
    def __init__(self, in_features, out_features):
        self.W = Parameter(np.random.randn(in_features, out_features) * 0.01)
        self.b = Parameter(np.zeros(out_features))

    def forward(self, x):
        self.x = x
        return x @ self.W.data + self.b.data

    def backward(self, grad):
        self.W.grad += self.x.T @ grad
        self.b.grad += grad.sum(axis=0)
        return grad @ self.W.data.T

    def parameters(self):
        return [self.W, self.b]
