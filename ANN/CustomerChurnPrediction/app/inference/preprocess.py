from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from app.core.config import settings
from app.core.logger import get_logger
from app.inference.encoders.categorical import encode_one_hot
from app.inference.feature_selection import select_features
from app.inference.load_data import load_dataset

logger = get_logger(__name__)


class Preprocessor:
    def __init__(self, target_col="Exited"):
        self.target_col = target_col
        self.scaler = StandardScaler()
        self.expected_features = None  # Will be determined after training

    # -------- Training-time pipeline --------
    def preprocess(self, input_file: str, output_file: str) -> pd.DataFrame:
        """
        Load raw CSV → feature selection → encode categorical → save processed CSV.
        Also sets self.expected_features dynamically.
        """
        raw_file_path = settings.RAW_DATA_DIR / input_file
        df = load_dataset(raw_file_path)

        # Drop identifiers
        drop_cols = ["Surname", "RowNumber", "CustomerId"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

        # Feature selection
        df = select_features(df, self.target_col)

        # Encode categorical
        df = encode_one_hot(df, cols=["Geography", "Gender"])

        # Ensure processed directory exists
        settings.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        processed_path = settings.PROCESSED_DATA_DIR / output_file
        df.to_csv(processed_path, index=False)
        logger.info(f"✔ Processed data saved to {processed_path}")

        # Determine expected features dynamically
        self.expected_features = [c for c in df.columns if c != self.target_col]

        return df

    def split_and_scale(self, df: pd.DataFrame, test_size=0.2, random_state=42):
        """
        Split into train/test and fit scaler on training data.
        """
        if self.expected_features is None:
            self.expected_features = [c for c in df.columns if c != self.target_col]

        X = df[self.expected_features]
        y = df[self.target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Fit scaler on training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, y_train, y_test

    # -------- Inference-time helpers --------
    def encode_for_inference(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode raw JSON / DataFrame to match training features and column order.
        """
        df = df.copy()

        # One-hot encode categorical features
        df["Geography_Germany"] = (df.get("Geography") == "Germany").astype(int)
        df["Geography_Spain"] = (df.get("Geography") == "Spain").astype(int)
        df["Gender_Male"] = (df.get("Gender") == "Male").astype(int)

        # Drop original categorical columns
        df = df.drop(columns=["Geography", "Gender"], errors="ignore")

        # Fill missing expected columns
        for c in self.expected_features:
            if c not in df.columns:
                df[c] = 0

        # Enforce column order
        return df[self.expected_features]

    def scale_for_inference(self, df: pd.DataFrame):
        """
        Scale input using the fitted scaler. Automatically loads scaler from disk if needed.
        """
        if not hasattr(self.scaler, "mean_"):
            scaler_path = settings.MODELS_DIR / "scaler.pkl"
            if scaler_path.exists():
                self.load_fitted_scaler(scaler_path)
            else:
                raise RuntimeError("Scaler is not fitted and no scaler.pkl found.")
        return self.scaler.transform(df)

    # -------- Save / Load helpers --------
    def save_fitted_scaler(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, path)
        logger.info(f"✔ Saved scaler to {path}")

    def load_fitted_scaler(self, path: Path):
        self.scaler = joblib.load(path)
        logger.info(f"✔ Loaded scaler from {path}")

    def save_metadata(self, path: Path):
        """
        Save expected features for inference.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"expected_features": self.expected_features}, path)
        logger.info(f"✔ Saved metadata to {path}")

    def load_metadata(self, path: Path):
        meta = joblib.load(path)
        self.expected_features = meta["expected_features"]
        logger.info(f"✔ Loaded metadata from {path}")
