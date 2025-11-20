from typing import List

import pandas as pd
from scipy.stats import pointbiserialr


def get_numeric_correlation(
    df: pd.DataFrame, target_col: str, threshold: float = 0.05
) -> List[str]:
    numeric_cols = df.select_dtypes(include="number").columns.drop(target_col)
    correlations = {}
    for col in numeric_cols:
        corr, _ = pointbiserialr(df[col], df[target_col])
        correlations[col] = corr
    selected = [col for col, corr in correlations.items() if abs(corr) >= threshold]
    return selected
