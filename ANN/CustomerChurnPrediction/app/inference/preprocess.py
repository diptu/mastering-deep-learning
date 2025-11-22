"""Full sklearn/Tensorflow preprocessing pipeline."""

import logging
from pathlib import Path

from app.core.config import settings
from app.inference.feature_selection import select_features
from app.inference.load_data import load_dataset

# Configure basic logging to console
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Get a logger instance
logger = logging.getLogger(__name__)


def preprocess_pipeline(input_filename: str, output_filename: str):
    """
    Load raw dataset → run feature selection → save processed dataset.

    Args:
        input_filename (str): CSV filename inside RAW_DATA_DIR.
        output_filename (str): Output CSV filename saved in PROCESSED_DATA_DIR.
    """
    target_col = "Exited"

    # Build full input file path
    raw_file_path = settings.RAW_DATA_DIR / input_filename
    logger.debug(f"raw_file_path: {raw_file_path}")
    # Load dataset
    df = load_dataset(raw_file_path)

    # Apply feature selection
    processed_df = select_features(df, target_col)
    print(f"Data after dropping weak features has {processed_df.shape[1]} columns")

    # Ensure output directory exists
    processed_dir: Path = settings.PROCESSED_DATA_DIR
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Save processed file
    processed_file_path = processed_dir / output_filename
    processed_df.to_csv(processed_file_path, index=False)

    print(f"✔ Processed data saved to: {processed_file_path}")

    return processed_df


if __name__ == "__main__":
    input_data = "churn"
    output_data = f"processed_{input_data}"

    final_df = preprocess_pipeline(
        input_filename=f"{input_data}.csv", output_filename=f"{output_data}.csv"
    )

    print(final_df.head())
