import argparse
import os
from pathlib import Path

# Set Kaggle credentials first
from app.core.config import settings

os.environ["KAGGLE_USERNAME"] = settings.USER_NAME
os.environ["KAGGLE_KEY"] = settings.API_KEY

from kaggle.api.kaggle_api_extended import KaggleApi  # now reads env


def download_dataset(dataset: str):
    out_dir = "app/data"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()  # reads from env
    print(f"⬇ Downloading dataset: {dataset} ...")
    api.dataset_download_files(dataset, path=out_dir, unzip=True)
    print(f"✔ Dataset downloaded & extracted to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Kaggle dataset using API credentials."
    )
    parser.add_argument("-d", "--dataset", required=True, help="Kaggle dataset slug")
    args = parser.parse_args()
    download_dataset(args.dataset)
