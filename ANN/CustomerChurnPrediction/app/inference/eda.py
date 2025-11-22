"""Select Feature"""

import pandas as pd

from app.core.logger import get_logger
from app.utils.data_checks import check_class_balance, log_missing_values

logger = get_logger(__name__)


def data_analysis(df: pd.DataFrame):
    # Log missing value counts
    log_missing_values(df)
    # Log class counts
    check_class_balance(df, target_col="Exited", imbalanced_threshold=0.3)
    check_class_balance(df, target_col="Geography", imbalanced_threshold=0.3)
    check_class_balance(df, target_col="Gender", imbalanced_threshold=0.3)

    logger.info(df.head())
