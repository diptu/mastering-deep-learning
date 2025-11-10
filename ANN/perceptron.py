import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = np.zeros(input_size)
        self.bias = 0.0
        self.loss_history = []

    def activation(self, x):
        """Step activation function"""
        return np.where(x >= 0, 1, 0)

    def predict(self, X):
        """Calculate linear combination and apply activation"""
        linear_output = np.dot(X, self.weights) + self.bias
        return self.activation(linear_output)

    def perceptron_loss(self, X, y):
        """Count of misclassified samples"""
        predictions = self.predict(X)
        loss = np.sum(predictions != y)
        return loss

    def fit(self, X, y):
        """Train the perceptron using misclassification loss"""
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                prediction = self.predict(x_i)
                update = self.lr * (y[idx] - prediction)
                self.weights += update * x_i
                self.bias += update
            
            # Track loss for this iteration
            loss = self.perceptron_loss(X, y)
            self.loss_history.append(loss)
            print(f"Iteration {_+1}/{self.n_iters}, Misclassified samples: {loss}")

# --- Example Usage ---
if __name__ == "__main__":
    # AND logic gate dataset
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([0, 0, 0, 1])

    perceptron = Perceptron(input_size=2, learning_rate=0.1, n_iters=10)
    perceptron.fit(X, y)

    print("\nFinal Weights:", perceptron.weights)
    print("Final Bias:", perceptron.bias)
    print("Predictions:", perceptron.predict(X))
