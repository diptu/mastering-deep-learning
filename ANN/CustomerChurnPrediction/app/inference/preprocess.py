# app/inference/preprocessor.py
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

    def preprocess(self, input_file: str, output_file: str) -> pd.DataFrame:
        raw_file_path = settings.RAW_DATA_DIR / input_file
        df = load_dataset(raw_file_path)

        # Drop identifiers
        drop_cols = ["Surname", "RowNumber", "CustomerId"]
        df = df.drop(columns=drop_cols)

        # Feature selection
        df = select_features(df, self.target_col)

        # Encode categorical features
        df = encode_one_hot(df, cols=["Geography", "Gender"])

        # Save processed data
        processed_dir = settings.PROCESSED_DATA_DIR
        processed_dir.mkdir(parents=True, exist_ok=True)
        processed_path = processed_dir / output_file
        df.to_csv(processed_path, index=False)
        logger.info(f"✔ Processed data saved to {processed_path}")

        return df

    def split_and_scale(self, df: pd.DataFrame):
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, y_train, y_test


if __name__ == "__main__":
    pre = Preprocessor()
    df = pre.preprocess("churn.csv", "processed_churn.csv")
    X_train, X_test, y_train, y_test = pre.split_and_scale(df)
    print(X_train.shape, X_test.shape)
