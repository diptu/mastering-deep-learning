# app/core/config.py
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"


class Settings(BaseSettings):
    """App settings loaded from environment variables."""

    # General App Config
    APP_NAME: str = "churn prediction"
    APP_ENV: str = "development"
    VERSION: str = "0.0.1"
    RAW_DIR: Optional[str] = None  # default relative folder
    PROCESSED_DIR: Optional[str] = None
    MODEL_DIR: Optional[str] = None
    DEBUG: bool = True

    EPOCHS: int = 15
    BATCH: int = 32

    # Database / API
    KAGGLE_USERNAME: Optional[str] = None
    KAGGLE_KEY: Optional[str] = None

    # Computed property
    @property
    def RAW_DATA_DIR(self) -> Path:
        return BASE_DIR / self.RAW_DIR

    @property
    def PROCESSED_DATA_DIR(self) -> Path:
        return BASE_DIR / self.PROCESSED_DIR

    @property
    def MODELS_DIR(self) -> Path:
        return BASE_DIR / self.MODEL_DIR

    @property
    def BASE_DIR(self) -> Path:
        return BASE_DIR

    # Pydantic v2 config
    model_config = SettingsConfigDict(env_file=str(ENV_FILE), extra="ignore")


# Instantiate settings
settings = Settings()

# Runtime validation
if not settings.KAGGLE_USERNAME:
    raise ValueError("KAGGLE_USERNAME must be set in environment variables")

if not settings.KAGGLE_KEY:
    raise ValueError("KAGGLE_KEY must be set in environment variables")
