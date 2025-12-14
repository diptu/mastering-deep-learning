import numpy as np
import matplotlib.pyplot as plt


def visualize_features(X: np.ndarray):
    """
    Line plot where each feature has a different color.
    X shape: (samples, features)
    """
    n_features = X.shape[1]
    colors = plt.cm.tab10(np.linspace(0, 1, n_features))

    plt.figure()
    for i in range(n_features):
        plt.plot(X[:, i], color=colors[i], label=f"Feature {i}")

    plt.title("Generated Data")
    plt.xlabel("Sample Index")
    plt.ylabel("Feature Value")
    plt.legend()
    plt.show()
