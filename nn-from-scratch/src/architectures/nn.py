import numpy as np
from engine.activations import sigmoid


class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, x):
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, x) + b
            a = sigmoid(z)
            return a


if __name__ == "__main__":
    net = Network([2, 3, 1])
