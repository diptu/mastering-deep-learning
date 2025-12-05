from typing import List

import pandas as pd
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder

from app.core.logger import get_logger

logger = get_logger(__name__)


def get_categorical_chi2(
    df: pd.DataFrame, target_col: str, pval_threshold: float = 0.05
) -> List[str]:
    """
    Select categorical features based on chi-squared test and log dropped features.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target_col (str): Name of the target column.
        pval_threshold (float): Max p-value to consider feature significant.

    Returns:
        List[str]: Selected categorical features.
    """
    categorical_cols = df.select_dtypes(include="object").columns
    if len(categorical_cols) == 0:
        logger.info("No categorical columns found.")
        return []

    # Encode categorical features
    X_cat = df[categorical_cols].apply(lambda col: LabelEncoder().fit_transform(col))
    y = df[target_col]

    _, p_values = chi2(X_cat, y)

    # Selected features
    selected = [
        col for col, p in zip(categorical_cols, p_values) if p <= pval_threshold
    ]

    # Log dropped features
    dropped = [col for col, p in zip(categorical_cols, p_values) if p > pval_threshold]
    if dropped:
        logger.warning(
            f"⚠ Dropped categorical columns (p-value > {pval_threshold}): {dropped}"
        )
    else:
        logger.info("✔ No categorical columns were dropped.")

    # Optional: log all p-values
    pval_dict = dict(zip(categorical_cols, p_values))
    logger.debug(f"Categorical feature p-values: {pval_dict}")

    return selected
