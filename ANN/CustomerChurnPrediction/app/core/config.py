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
    FILE_PATH: Optional[str] = "data"  # default relative folder
    DEBUG: bool = True

    # Database / API
    USER_NAME: Optional[str] = None
    API_KEY: Optional[str] = None

    # Computed property
    @property
    def DATA_PATH(self) -> Path:
        return BASE_DIR / self.FILE_PATH

    # Pydantic v2 config
    model_config = SettingsConfigDict(env_file=str(ENV_FILE), extra="ignore")


# Instantiate settings
settings = Settings()

# Runtime validation
if not settings.USER_NAME:
    raise ValueError("USER_NAME must be set in environment variables")

if not settings.API_KEY:
    raise ValueError("API_KEY must be set in environment variables")
