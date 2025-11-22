import pandas as pd

from app.core.config import settings


def load_dataset(path: str = settings.RAW_DATA_DIR) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns")
    return df
