"""Select Feature"""

import pandas as pd

from app.core.logger import get_logger
from app.inference.categorical_features import get_categorical_chi2
from app.inference.numeric_features import get_numeric_correlation

logger = get_logger(__name__)


def select_features(df: pd.DataFrame, target_col: str):
    numeric_selected = get_numeric_correlation(df, target_col)
    categorical_selected = get_categorical_chi2(df, target_col, 0.1)

    final_features = numeric_selected + categorical_selected
    logger.info(f"Final selected features: {final_features}")
    return df[final_features + [target_col]]
