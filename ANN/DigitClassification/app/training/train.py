from app.core.logger import get_logger
from app.inference.preprocess import Preprocessor

logger = get_logger(__name__)


def train():
    """
    Loads MNIST raw images/labels, normalizes them, and returns all four arrays.

    Returns:
        X_train_norm, y_train, X_test_norm, y_test
    """
    pre = Preprocessor()

    try:
        # Load raw pixel arrays
        X_train, y_train, X_test, y_test = pre.preprocess()
        # print(X_train.shape)

    except RuntimeError:
        logger.error("FATAL: MNIST preprocessing failed. Check file locations.")
        raise  # allow upstream system to fail cleanly

    # Normalize pixel values
    X_train_norm, X_test_norm = pre.normalize(X_train, X_test)
    logger.info("--- Data Preparation Complete ---")
    logger.info(f"X_train_norm: {X_train_norm.shape} | dtype={X_train_norm.dtype}")
    logger.info(f"X_test_norm:  {X_test_norm.shape} | dtype={X_test_norm.dtype}")
    logger.info(f"y_train: {y_train.shape} | y_test: {y_test.shape}")


if __name__ == "__main__":
    train()
