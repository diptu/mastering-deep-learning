import pandas as pd

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


def load_dataset(path: str = settings.RAW_DATA_DIR) -> pd.DataFrame:
    df = pd.read_csv(path)
    logger.info(f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns")
    return df
