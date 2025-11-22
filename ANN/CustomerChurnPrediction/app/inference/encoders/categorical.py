"""Select Feature"""

import pandas as pd
from app.core.logger import get_logger

logger = get_logger(__name__)


def encode_one_hot(df: pd.DataFrame, cols: list):
    return pd.get_dummies(df, columns=cols, drop_first=True)
