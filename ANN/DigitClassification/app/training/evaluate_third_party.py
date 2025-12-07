# app/utils/evaluate_third_party.py
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from app.core.config import settings
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_model(model_name: str = "tf_digit_classification_model.h5") -> tf.keras.Model:
    """
    Load the trained TensorFlow model from the models directory.
    """
    model_path = settings.MODELS_DIR / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    model = tf.keras.models.load_model(model_path)
    print(f"✔ Model loaded successfully from {model_path}")
    return model


def load_3p_dataset(base_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a third-party dataset organized by class folders (0-9).

    Args:
        base_dir (str): Path to dataset folder

    Returns:
        X (np.ndarray): Array of images (N, 28, 28)
        y (np.ndarray): Array of labels
    """
    X, y = [], []
    base_path = Path(base_dir)

    for class_folder in sorted(base_path.iterdir()):  # folders 0-9
        if not class_folder.is_dir():
            continue
        label = int(class_folder.name)
        for img_file in class_folder.iterdir():
            if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
                continue
            img = Image.open(img_file).convert("L")  # grayscale
            img = img.resize((28, 28))
            arr = np.array(img, dtype=np.float32) / 255.0
            X.append(arr)
            y.append(label)

    X = np.stack(X)
    y = np.array(y)
    print(f"✔ Loaded {len(y)} images from {base_dir}")
    return X, y


def evaluate_third_party_dataset(
    model: tf.keras.Model, X: np.ndarray, y: np.ndarray, plot_cm: bool = True
):
    """
    Evaluate a model on given dataset and print metrics.

    Args:
        model (tf.keras.Model): Trained TensorFlow model
        X (np.ndarray): Images array (N, 28, 28)
        y (np.ndarray): Labels array
        plot_cm (bool): Whether to plot the confusion matrix
    """
    # Predict
    y_prob = model.predict(X)
    y_pred = np.argmax(y_prob, axis=1)

    # Accuracy
    acc = accuracy_score(y, y_pred)
    print(f"✅ Test Accuracy: {acc * 100:.4f} %")

    # Confusion matrix
    if plot_cm:
        cm = confusion_matrix(y, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.show()

    # Classification report
    report = classification_report(y, y_pred)
    print(report)


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    print(f"RAW_TEST_DIR:{settings.RAW_TEST_DIR}")
    dataset_dir = settings.RAW_TEST_DIR

    model = load_model()
    X_test, y_test = load_dataset(dataset_dir)
    evaluate_third_party_dataset(model, X_test, y_test)
