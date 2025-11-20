import pandas as pd
from app.core.config import settings


def load_dataset() -> pd.DataFrame:
    csv_path = settings.DATA_PATH
    df = pd.read_csv(csv_path)
    print(f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns")
    return df
