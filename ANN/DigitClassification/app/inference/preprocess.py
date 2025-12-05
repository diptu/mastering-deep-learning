import struct
from pathlib import Path
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

# --------------------------------------------------------------------
#                     SIMPLE IDX FILE LOADERS
# --------------------------------------------------------------------


def _read_uints(f, fmt: str):
    """Small helper that reads and unpacks a struct format."""
    size = struct.calcsize(fmt)
    return struct.unpack(fmt, f.read(size))


def load_images(path: Path) -> np.ndarray:
    """Load MNIST images (IDX-3-UBYTE). Returns shape: (N, 28, 28)."""  # 🚨 UPDATED DOCSTRING
    try:
        with path.open("rb") as f:
            # magic: 2051, num: 60000, rows: 28, cols: 28 (for training set)
            magic, num, rows, cols = _read_uints(f, ">IIII")
            if magic != 2051:
                raise ValueError(f"Invalid magic number for images: {magic}")

            data = np.frombuffer(f.read(), dtype=np.uint8)

            # 🚨 FIX: Reshape to (num, rows, cols) instead of (num, rows * cols)
            return data.reshape(num, rows, cols)  # (60000, 28, 28) format

    except Exception as e:
        logger.error(f"Failed loading images: {path} → {e}")
        return np.array([])


def load_labels(path: Path) -> np.ndarray:
    """Load MNIST labels (IDX-1-UBYTE)."""
    try:
        with path.open("rb") as f:
            magic, num = _read_uints(f, ">II")
            if magic != 2049:
                raise ValueError(f"Invalid magic number for labels: {magic}")

            return np.frombuffer(f.read(), dtype=np.uint8)

    except Exception as e:
        logger.error(f"Failed loading labels: {path} → {e}")
        return np.array([])


# --------------------------------------------------------------------
#                        PREPROCESSOR
# --------------------------------------------------------------------


class Preprocessor:
    """Loads, normalizes, and prepares MNIST data."""

    def _resolve_path(self, raw_dir: Path, filename_base: str, is_image: bool) -> Path:
        """
        Attempts to resolve the correct path for an MNIST file, checking both
        the standard flat name (with '.') and the potentially nested directory
        structure found in the 'tree' output.
        """
        # 1. Standard/Correct Path (Flat file with '.')
        extension = "idx3-ubyte" if is_image else "idx1-ubyte"
        flat_filename = f"{filename_base}.{extension}"
        path_flat = raw_dir / flat_filename

        if path_flat.is_file():
            return path_flat

        # 2. Path pointing to a directory that contains the file (as seen in the tree output)
        # This is a less common but necessary check based on your specific directory structure.
        dir_name = f"{filename_base}-{extension}"
        path_nested = raw_dir / dir_name / dir_name

        if path_nested.is_file():
            logger.warning(
                f"Using nested path for {filename_base}: {path_nested.relative_to(settings.BASE_DIR)}"
            )
            return path_nested

        # 3. Path pointing to the directory itself (which was causing the error)
        # We check this last, as it's typically the source of error, but if it exists
        # and we can't find the file, we return it to let the caller logging the error.
        path_dir_error = raw_dir / dir_name

        # If the file is not found in the standard locations, return the directory path
        # that was originally used, so the error handling in load_images/load_labels
        # can log the appropriate error (which will likely still be "Is a directory").
        return path_dir_error

    def preprocess(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raw = settings.RAW_DATA_DIR

        logger.info(f"Loading MNIST from {raw.relative_to(settings.BASE_DIR)}")

        # Use the path resolver for robustness
        train_images_path = self._resolve_path(raw, "train-images", is_image=True)
        train_labels_path = self._resolve_path(raw, "train-labels", is_image=False)
        test_images_path = self._resolve_path(raw, "t10k-images", is_image=True)
        test_labels_path = self._resolve_path(raw, "t10k-labels", is_image=False)

        X_train = load_images(train_images_path)
        y_train = load_labels(train_labels_path)
        X_test = load_images(test_images_path)
        y_test = load_labels(test_labels_path)

        if any(arr.size == 0 for arr in [X_train, y_train, X_test, y_test]):
            # This will be triggered if load_images/load_labels failed and returned np.array([])
            raise RuntimeError("Failed to load one or more MNIST files.")

        logger.info(f"Train: X={X_train.shape}, y={y_train.shape}")
        logger.info(f"Test:  X={X_test.shape}, y={y_test.shape}")

        # Normalize the images before returning (standard practice)
        # X_train_norm, X_test_norm = self.normalize(X_train, X_test)

        return X_train, y_train, X_test, y_test

    # ----------------------------------------------------------------

    def normalize(self, X_train: np.ndarray, X_test: np.ndarray):
        """Normalize pixel values from [0,255] → [0,1]."""
        logger.info("Normalizing MNIST images...")
        return X_train.astype(np.float32) / 255.0, X_test.astype(np.float32) / 255.0

    # ----------------------------------------------------------------

    def normalize_for_inference(self, X_new: Union[np.ndarray, list]) -> np.ndarray:
        """Normalize single or batch of 784-sized flattened images."""
        X_new = np.asarray(X_new)

        if X_new.ndim == 1:  # single image
            X_new = X_new.reshape(1, -1)

        if X_new.shape[-1] != 784:
            raise ValueError(f"Expected shape (N, 784), got {X_new.shape}")

        return X_new.astype(np.float32) / 255.0

    def plot_sample_images(
        self, X: np.ndarray, y: np.ndarray, num_samples: int = 10
    ) -> None:
        if num_samples > X.shape[0]:
            num_samples = X.shape[0]

        grid_size = int(np.sqrt(num_samples))
        if grid_size * grid_size != num_samples:
            num_samples = grid_size * grid_size
            logger.warning(
                f"Adjusted num_samples to {num_samples} for square grid plotting."
            )

        indices = np.random.choice(X.shape[0], num_samples, replace=False)

        # Create the figure and axes
        fig, axes = plt.subplots(
            grid_size, grid_size, figsize=(grid_size * 2, grid_size * 2)
        )
        fig.suptitle(f"Sample MNIST Images (N={num_samples})", fontsize=16)

        # 🚨 ULTIMATE FIX: Safely convert axes to a 1D array of Axes objects.
        # np.ravel() works correctly whether 'axes' is a single object or a 2D array.
        axes_flat = np.ravel(axes)

        for i, idx in enumerate(indices):
            ax = axes_flat[i]  # Index into the safely flattened array

            image = X[idx].reshape(28, 28)

            ax.imshow(image, cmap="gray")
            ax.set_title(f"Label: {y[idx]}", fontsize=10)
            ax.axis("off")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
