import numpy as np
import random
from engine.activations import sigmoid


def SGD(self, training_data, epochs, batch_size, eta, test_data=None):
    if test_data:
        test_size = len(test_data)
        train_size = len(training_data)

        for epoch in epochs:
            random.shuffle(test_data)
            batches = [
                training_data[k : k + batch_size]
                for k in range(0, train_size, batch_size)
            ]
            for batch in batches:
                self.update_mini_batch(batch, eta)

            if test_data:
                print(f"Epoch {epoch} {self.evaluate(test_data)}/ {test_size}")
            else:
                print("Epoch {epoch} complete")


def update_mini_batch(self, batch, eta):
    b = [np.zeros(b.shape) for b in self.biases]
    w = [np.zeros(w.shape) for w in self.weights]

    for x, y in batch:
        delta_b, delta_w = self.backprop(x, y)
        b = [b + db for b, db in zip(b, delta_b)]
        w = [w + dw for w, dw in zip(w, delta_w)]
    self.weights = [w - (eta / len(batch)) * nw for w, nw in zip(self.weights, w)]
    self.biases = [b - (eta / len(batch)) * nb for b, nb in zip(self.biases, b)]


def backprop(self, x, y):
    b = [np.zeros(b.shape) for b in self.biases]
    w = [np.zeros(w.shape) for w in self.weights]

    # feedforward
    activation = x
    activations = [x]
    zs = []
    for b, w in zip(self.biases, self.weights):
        z = np.dot(w, activation) + b
        zs.append(z)
        activation = sigmoid(z)
        activations.append(activation)

    # TODO backward pass
