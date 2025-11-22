# app/utils/download_kaggle_dataset.py
import os
from pathlib import Path

from app.core.config import settings
from kaggle.api.kaggle_api_extended import KaggleApi

os.environ["KAGGLE_USERNAME"] = settings.USER_NAME
os.environ["KAGGLE_KEY"] = settings.API_KEY


def download_dataset(dataset_slug: str, target_csv="churn.csv"):
    out_dir = Path(settings.RAW_DATA_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    print(f"⬇ Downloading {dataset_slug} ...")
    api.dataset_download_files(dataset_slug, path=out_dir, unzip=True)

    # Rename first CSV
    csv_files = list(out_dir.glob("*.csv"))
    if not csv_files:
        print("⚠ No CSV found!")
        return

    source_file = csv_files[0]
    target_file = out_dir / target_csv
    if target_file.exists():
        target_file.unlink()
    source_file.rename(target_file)
    print(f"✔ Dataset ready at {target_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", required=True)
    args = parser.parse_args()
    download_dataset(args.dataset)
