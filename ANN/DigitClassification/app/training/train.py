from app.core.logger import get_logger
from app.inference.preprocess import Preprocessor

logger = get_logger(__name__)


def run_data_preparation(normalize: bool = True):
    """
    Loads MNIST raw images/labels, normalizes them, and returns all four arrays.

    Returns:
        X_train_norm, y_train, X_test_norm, y_test
    """
    pre = Preprocessor()

    try:
        # Load raw pixel arrays
        X_train, y_train, X_test, y_test = pre.preprocess()
        print(X_train.shape)

        # Plot the first test image (X_test[0]) and its label (y_test[0])
        # We use [0:1] to select the first row but KEEP the shape as (1, 28*28),
        # instead of [0], which returns a (28*28,) 1D array.

        # test_image_sample = X_train[0:1]  # Shape: (1, 28*28)
        # test_label_sample = y_train[0:1]  # Shape: (1,)

        # print(f"test_image_sample:{test_image_sample.shape}")
        # pre.plot_sample_images(test_image_sample, test_label_sample, num_samples=1)

    except RuntimeError:
        logger.error("FATAL: MNIST preprocessing failed. Check file locations.")
        raise  # allow upstream system to fail cleanly

    if normalize:
        # Normalize pixel values
        X_train_norm, X_test_norm = pre.normalize(X_train, X_test)
        logger.info("--- Data Preparation Complete ---")
        logger.info(f"X_train_norm: {X_train_norm.shape} | dtype={X_train_norm.dtype}")
        logger.info(f"X_test_norm:  {X_test_norm.shape} | dtype={X_test_norm.dtype}")
        logger.info(f"y_train: {y_train.shape} | y_test: {y_test.shape}")

        return X_train_norm, y_train, X_test_norm, y_test

    # Summary logging
    logger.info("--- Data Preparation Complete ---")
    logger.info(f"X_train: {X_train.shape} | dtype={X_train.dtype}")
    logger.info(f"X_test:  {X_test.shape} | dtype={X_test.dtype}")
    logger.info(f"y_train: {y_train.shape} | y_test: {y_test.shape}")

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = run_data_preparation(normalize=False)
    N = 5
    print(f"\nFirst {N} normalized pixel values of image 0:")
    print(f"{X_train[0][:N]}")
    print(f"Label of the first image: {y_train[0]}")
