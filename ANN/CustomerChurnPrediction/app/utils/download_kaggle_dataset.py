import argparse
import os
from pathlib import Path

# Set Kaggle credentials first
from app.core.config import settings

os.environ["KAGGLE_USERNAME"] = settings.USER_NAME
os.environ["KAGGLE_KEY"] = settings.API_KEY

from kaggle.api.kaggle_api_extended import KaggleApi  # now reads env


def download_dataset(dataset: str, file: str = None):
    out_dir = Path(settings.RAW_DATA_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()  # reads from env

    print(f"⬇ Downloading dataset: {dataset} ...")
    api.dataset_download_files(dataset, path=out_dir, unzip=True)
    print(f"✔ Dataset downloaded & extracted to: {out_dir}")

    # --- Rename CSV to churn.csv ---
    csv_files = list(out_dir.glob("*.csv"))

    if not csv_files:
        print("⚠ No CSV files found in the downloaded dataset.")
        return

    source_file = csv_files[0]  # pick the first CSV
    target_file = out_dir / file

    # Remove existing churn.csv to avoid conflicts
    if target_file.exists():
        target_file.unlink()

    source_file.rename(target_file)

    print(f"✔ Renamed dataset file to: {target_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Kaggle dataset and rename CSV to churn.csv."
    )
    parser.add_argument("-d", "--dataset", required=True, help="Kaggle dataset slug")
    args = parser.parse_args()
    download_dataset(args.dataset, "churn.csv")
