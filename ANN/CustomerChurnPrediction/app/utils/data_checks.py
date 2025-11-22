# app/utils/data_checks.py
import pandas as pd
from app.core.logger import get_logger

logger = get_logger(__name__)


def check_class_balance(
    df: pd.DataFrame, target_col: str, imbalanced_threshold: float = 0.3
) -> None:
    """
    Logs class distribution and checks if the target column is balanced.

    Args:
        df (pd.DataFrame): Dataset containing the target column.
        target_col (str): Name of the target column.
        imbalanced_threshold (float): Minimum ratio for each class to consider balanced.
    """
    if target_col not in df.columns:
        logger.error(f"Target column '{target_col}' not found in DataFrame.")
        return

    class_counts = df[target_col].value_counts()
    logger.info(f"Class distribution:\n{class_counts}")

    total = len(df)
    ratios = class_counts / total

    if (ratios < imbalanced_threshold).any():
        logger.warning(f"⚠ Data is imbalanced. Class ratios:\n{ratios.to_dict()}")
    else:
        logger.info(f"✔ {target_col} column is reasonably balanced.")


def log_missing_values(df: pd.DataFrame) -> None:
    """
    Logs the number of missing values per column in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to check.
    """
    missing_counts = df.isnull().sum()
    total_missing = missing_counts.sum()

    if total_missing == 0:
        logger.info("✔ No missing values in the dataset.")
    else:
        logger.warning(
            f"⚠ Missing values detected:\n{missing_counts[missing_counts > 0]}"
        )
