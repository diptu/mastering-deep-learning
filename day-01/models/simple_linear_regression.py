import numpy as np


class LinearRegression:
    """
    Simple Linear Regression (1 feature)

    y_hat = w * x + b
    """

    def __init__(self, lr: float = 0.01, epochs: int = 1000):
        self.lr = lr
        self.epochs = epochs

        self.w: float | None = None
        self.b: float | None = None

        # normalization params
        self.x_mean: float | None = None
        self.x_std: float | None = None
        self.y_mean: float | None = None
        self.y_std: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train using gradient descent
        """
        # Ensure shapes (m,)
        X = X.reshape(-1)
        y = y.reshape(-1)
        m = X.shape[0]

        # Normalize (VERY IMPORTANT)
        self.x_mean = X.mean()
        self.x_std = X.std() + 1e-8
        self.y_mean = y.mean()
        self.y_std = y.std() + 1e-8

        Xn = (X - self.x_mean) / self.x_std
        yn = (y - self.y_mean) / self.y_std

        # Initialize parameters
        self.w = 0.0
        self.b = 0.0

        # Gradient Descent
        for _ in range(self.epochs):
            y_hat = self.w * Xn + self.b
            error = y_hat - yn

            dw = (2 / m) * np.sum(error * Xn)
            db = (2 / m) * np.sum(error)

            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using trained model
        """
        X = X.reshape(-1)
        Xn = (X - self.x_mean) / self.x_std
        y_norm = self.w * Xn + self.b
        return y_norm * self.y_std + self.y_mean
