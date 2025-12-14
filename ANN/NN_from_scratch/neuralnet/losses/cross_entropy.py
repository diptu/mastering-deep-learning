import numpy as np
from neuralnet.activations.softmax import softmax


class SparseCrossEntropy:
    def forward(self, logits, y):
        self.p = softmax(logits)
        self.y = y
        return -np.mean(np.log(self.p[np.arange(len(y)), y]))

    def backward(self):
        grad = self.p
        grad[np.arange(len(self.y)), self.y] -= 1
        return grad / len(self.y)
