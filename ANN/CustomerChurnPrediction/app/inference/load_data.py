# app/inference/load_data.py
import pandas as pd

from app.core.logger import get_logger

logger = get_logger(__name__)


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    logger.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df
