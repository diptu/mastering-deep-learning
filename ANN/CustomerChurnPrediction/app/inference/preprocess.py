"""Full sklearn/Tensorflow preprocessing pipeline."""

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from app.core.config import settings
from app.core.logger import get_logger
from app.inference.encoders.categorical import encode_one_hot
from app.inference.feature_selection import select_features
from app.inference.load_data import load_dataset

logger = get_logger(__name__)


def preprocess_pipeline(
    input_filename: str, output_filename: str, target_col: str = "Exited"
):
    """
    Load raw dataset → run feature selection → save processed dataset.

    Args:
        input_filename (str): CSV filename inside RAW_DATA_DIR.
        output_filename (str): Output CSV filename saved in PROCESSED_DATA_DIR.
    """

    # Build full input file path
    raw_file_path = settings.RAW_DATA_DIR / input_filename
    logger.debug(f"raw_file_path: {raw_file_path}")
    # Load dataset
    df = load_dataset(raw_file_path)

    drop_cols = ["Surname", "RowNumber", "CustomerId"]  # identifiers
    df = df.drop(columns=drop_cols)

    # Apply feature selection
    processed_df = select_features(df, target_col)
    logger.debug(
        f"Data after dropping weak features has {processed_df.shape[1]} columns"
    )

    # one hot Encode categorical feature
    processed_df = encode_one_hot(processed_df, cols=["Geography", "Gender"])

    # Ensure output directory exists
    processed_dir: Path = settings.PROCESSED_DATA_DIR
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Save processed file
    processed_file_path = processed_dir / output_filename
    processed_df.to_csv(processed_file_path, index=False)

    logger.info(f"✔ Processed data saved to: {processed_file_path}")

    return processed_df


if __name__ == "__main__":
    target_col = "Exited"
    input_data = "churn"
    output_data = f"processed_{input_data}"

    # # ----------------
    # # # EDA
    # # --------------
    # path = settings.RAW_DATA_DIR / f"{input_data}.csv"
    # df = pd.read_csv(path)

    # data_analysis(df)

    df = preprocess_pipeline(
        input_filename=f"{input_data}.csv",
        output_filename=f"{output_data}.csv",
        target_col=target_col,
    )
    # # ----------------
    # #Split data
    # # --------------
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # # ----------------
    # # Standaralize Data
    # # --------------
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)
    print(X_train_scaled)
