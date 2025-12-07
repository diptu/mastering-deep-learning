# app/inference/predictor.py
from io import BytesIO
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from app.core.config import settings
from app.core.logger import get_logger
from app.inference.preprocess import Preprocessor

logger = get_logger(__name__)


class Predictor:
    def __init__(self):
        model_path: Path = settings.MODELS_DIR / "tf_digit_classification_model.h5"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        logger.info(f"✔ Loaded model from {model_path}")
        self.pre = Preprocessor()

    # -----------------------------
    # IMAGE PREPROCESSING
    # -----------------------------
    def preprocess_image(self, img: Image.Image) -> np.ndarray:
        """
        Convert PIL image → grayscale → 28x28 → flatten → normalize
        Returns: (1, 784) array ready for model.predict
        """
        img = img.convert("L")  # grayscale
        img = img.resize((28, 28))
        arr = np.array(img, dtype=np.float32)
        arr = arr.reshape(1, -1)  # flatten
        arr = self.pre.normalize_for_inference(arr)
        return arr

    def predict_from_file_bytes(self, image_bytes: bytes) -> dict:
        """
        Accept raw image bytes, preprocess, predict digit
        """
        try:
            img = Image.open(BytesIO(image_bytes))
        except Exception as e:
            logger.error(f"Failed to open image: {e}")
            raise ValueError("Invalid image file")

        X = self.preprocess_image(img)
        probs = self.model.predict(X, verbose=0).squeeze()
        pred = int(np.argmax(probs))
        confidence = float(np.max(probs))

        return {
            "prediction": pred,
            "probability": confidence,
            "probabilities": probs.tolist(),
        }
