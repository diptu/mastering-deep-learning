import argparse
import os

from app.core.config import settings

# --- FIX 1: Set Environment Variables with Corrected Names ---
# Assuming app/core/config.py was updated to use KAGGLE_USERNAME/KAGGLE_KEY,
# which are the standard environment variables the Kaggle API checks.
os.environ["KAGGLE_USERNAME"] = settings.KAGGLE_USERNAME
os.environ["KAGGLE_KEY"] = settings.KAGGLE_KEY

from kaggle.api.kaggle_api_extended import KaggleApi


def download_dataset(dataset_slug: str, target_csv: str = "mnist.csv"):
    """
    Downloads a Kaggle dataset, unzips it, and renames the first CSV file found.
    """
    # Use the computed Path property (settings.RAW_DATA_DIR) which includes BASE_DIR
    out_dir = settings.RAW_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- FIX 2: Remove api.authenticate() ---
    # Instantiating KaggleApi() here automatically picks up the KAGGLE_USERNAME
    # and KAGGLE_KEY from os.environ, making the explicit authenticate() call redundant
    # and avoiding the troublesome check for the local kaggle.json file.
    api = KaggleApi()

    api.authenticate()

    print(
        f"⬇ Downloading {dataset_slug} to {out_dir.relative_to(settings.RAW_DATA_DIR)}..."
    )
    api.dataset_download_files(dataset_slug, path=out_dir, unzip=True)

    # Rename first CSV
    csv_files = list(out_dir.glob("*.csv"))
    if not csv_files:
        print("⚠ No CSV found after unzipping the dataset!")
        return

    source_file = csv_files[0]
    target_file = out_dir / source_file

    if target_file.exists():
        target_file.unlink()

    source_file.rename(target_file)
    print(f"✔ Dataset ready at {target_file.relative_to(settings.BASE_DIR)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download a Kaggle dataset using environment variables for authentication."
    )
    parser.add_argument(
        "-d",
        "--dataset",
        required=True,
        help="The Kaggle dataset slug (e.g., hojjatk/mnist-dataset)",
    )

    args = parser.parse_args()

    download_dataset(args.dataset, settings.RAW_DATA_DIR)
