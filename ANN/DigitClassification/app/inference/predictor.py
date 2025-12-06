import joblib
import pandas as pd
import tensorflow

from app.core.config import settings
from app.core.logger import get_logger
from app.inference.preprocess import Preprocessor

logger = get_logger(__name__)


class Predictor:
    def __init__(self):
        """
        Load trained model, scaler, and metadata for inference.
        """
        # Paths
        model_path = settings.MODELS_DIR / "tf_digit_classification_model.h5.h5"
        scaler_path = settings.MODELS_DIR / "scaler.pkl"
        meta_path = settings.MODELS_DIR / "meta.pkl"

        # Load model
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        self.model = tensorflow.keras.models.load_model(model_path)
        logger.info(f"✔ Loaded model from {model_path}")

        # Load fitted scaler
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
        self.scaler = joblib.load(scaler_path)
        logger.info(f"✔ Loaded scaler from {scaler_path}")

        # Load expected features
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {meta_path}")
        meta = joblib.load(meta_path)
        self.expected_features = meta["expected_features"]
        logger.info(f"✔ Loaded expected features: {self.expected_features}")

        # Preprocessor instance for encoding
        self.pre = Preprocessor(target_col="Exited")
        # Assign the loaded scaler to preprocessor
        self.pre.scaler = self.scaler
        self.pre.expected_features = self.expected_features

    # ---------------------------------------------------------
    # PREDICTION
    # ---------------------------------------------------------
    def predict(self, payload: dict) -> dict:
        """
        Predict churn probability for a single input row.

        Args:
            payload (dict): Raw input (from API/Pydantic)

        Returns:
            dict: {"prediction": 0/1, "probability": float}
        """
        # Convert dict to single-row DataFrame
        df = pd.DataFrame([payload])

        # Encode categorical + match training features
        df_enc = self.pre.encode_for_inference(df)

        # Scale using fitted scaler
        X = self.pre.scale_for_inference(df_enc)

        # Predict probability
        prob = self.model.predict(X, verbose=0)[0][0]
        pred = int(prob > 0.5)

        return {"probability": float(prob), "prediction": pred}
