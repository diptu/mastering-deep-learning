from typing import List

import pandas as pd
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder


def get_categorical_chi2(
    df: pd.DataFrame, target_col: str, pval_threshold: float = 0.05
) -> List[str]:
    categorical_cols = df.select_dtypes(include="object").columns
    if len(categorical_cols) == 0:
        return []
    X_cat = df[categorical_cols].apply(LabelEncoder().fit_transform)
    y = df[target_col]
    _, p_values = chi2(X_cat, y)
    selected = [
        col for col, p in zip(categorical_cols, p_values) if p <= pval_threshold
    ]
    return selected
