# app/training/train.py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from app.core.logger import get_logger
from sklearn.metrics import accuracy_score

logger = get_logger(__name__)


def build_model(input_dim):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(64, activation="relu", input_dim=input_dim),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def plot_loss(history):
    """Plot training vs validation loss curves."""
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Binary Crossentropy Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_accuracy(history):
    """Plot training vs validation accuracy curves."""
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()


def evaluate_model(model, X_test, y_test):
    """Run prediction → threshold → accuracy score."""
    y_log = model.predict(X_test)
    y_pred = np.where(y_log > 0.5, 1, 0)

    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"✔ Test Accuracy: {accuracy:.4f}")
    return accuracy
