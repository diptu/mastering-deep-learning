import numpy as np
from visual import visualize_features
from activation import ReLU, SoftMax
from Layers import Dense


def generate_data(n_samples=100, n_features=5, seed=42):
    np.random.seed(seed)
    return np.random.randn(n_samples, n_features)


def main():
    # Data
    X = generate_data(n_samples=100, n_features=3)
    n_samples, n_features = X.shape

    print("Input Shape:", X.shape)

    # visualize_features(X)

    # Model config
    H1_NEURONS = 32
    H2_NEURONS = 16
    O_NEURONS = n_features
    relu = ReLU()
    soft_max = SoftMax()

    # Layer 1
    dense_1 = Dense(n_features, H1_NEURONS)
    z1 = dense_1.forward(X)
    a1 = relu.forward(z1)

    # Layer 2
    dense_2 = Dense(H1_NEURONS, H2_NEURONS)
    z2 = dense_2.forward(a1)
    a2 = soft_max.forward(z2)

    print("\nSample Output:")
    print(a2[:5])
    print("Final Output Shape:", a2.shape)


if __name__ == "__main__":
    main()
