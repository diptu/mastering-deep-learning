import numpy as np


class DenseLayer:
    """
    Represents a single fully connected (Dense) layer in a neural network.
    """

    def __init__(self, n_inputs, n_neurons):
        """
        Initializes weights (randomly) and biases (to zeros).

        :param n_inputs: The number of input features (or neurons from the previous layer).
        :param n_neurons: The number of neurons in this layer.
        """
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)

        self.biases = np.zeros((1, n_neurons))

        self.inputs = None
        self.output = None

    def forward(self, inputs):
        """
        Performs the forward pass calculation: output = dot(inputs, weights) + biases.

        :param inputs: A NumPy array of input data/activations.
                       Shape: (n_samples, n_inputs)
        :return: The output activation of this layer.
                 Shape: (n_samples, n_neurons)
        """
        self.inputs = inputs

        self.output = np.dot(inputs, self.weights) + self.biases

        return self.output


def main():
    X = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 8, 7, 6], [5, 4, 3, 2]])
    np.random.seed(42)
    layer_1 = DenseLayer(4, 3)
    layer_2 = DenseLayer(3, 5)

    layer_1_out = layer_1.forward(X)
    layer_2_out = layer_2.forward(layer_1_out)
    print("Layer Output:")
    print(layer_2_out)

    print("\nOutput Shape:", layer_2_out.shape)


if __name__ == "__main__":
    main()
