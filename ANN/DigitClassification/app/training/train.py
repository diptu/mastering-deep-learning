import numpy as np
from app.core.config import settings
from app.core.logger import get_logger
from app.inference.preprocess import Preprocessor
from app.models.tf_model import build_model, evaluate_model, plot_accuracy, plot_loss
from matplotlib import pyplot as plt

logger = get_logger(__name__)


def data_preprocess():
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
    return X_train_norm, X_test_norm, y_train, y_test


if __name__ == "__main__":
    logger.info("🚀 Starting training pipeline...")

    # ---------------------------------------------------------
    # 1️⃣ Load + Preprocess + Normalize
    # ---------------------------------------------------------
    X_train, X_test, y_train, y_test = data_preprocess()

    # ---------------------------------------------------------
    # 2️⃣ Build + Train Model
    # ---------------------------------------------------------
    HEIGHT = X_train.shape[1]
    WIDTH = X_train.shape[2]

    input_dim = (HEIGHT, WIDTH)
    logger.debug(f"input_dim:{input_dim}")

    model = build_model(input_dim)
    logger.debug(model.summary())

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=settings.EPOCHS,
        batch_size=settings.BATCH,
        verbose=1,
    )

    # # ---------------------------------------------------------
    # # 3️⃣ Save Model + Scaler + Metadata
    # # ---------------------------------------------------------
    # settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # model_path = settings.MODELS_DIR / "tf_digit_classification_model.h5"
    # scaler_path = settings.MODELS_DIR / "scaler.pkl"
    # meta_path = settings.MODELS_DIR / "meta.pkl"

    # # # save model
    # model.save(model_path)
    # logger.info(f"✔ Model saved to {model_path}")

    # # save scaler
    # pre.save_fitted_scaler(scaler_path)

    # # save expected feature columns
    # joblib.dump({"expected_features": pre.expected_features}, meta_path)
    # logger.info(f"✔ Metadata saved to {meta_path}")

    # # ---------------------------------------------------------
    # # 4️⃣ Plot Training Curves
    # # ---------------------------------------------------------
    plot_loss(history)
    plot_accuracy(history)

    # # ---------------------------------------------------------
    # # 5️⃣ Evaluate Model
    # # ---------------------------------------------------------
    evaluate_model(model, X_test, y_test)

    # # ---------------------------------------------------------
    # # 6. Test Single image prediction
    # # ---------------------------------------------------------
    # 3. Random Selection
    import random

    num_test_samples = X_test.shape[0]
    random_index = random.randint(0, num_test_samples - 1)

    # Retrieve the raw image and its true label
    raw_image = X_test[random_index]
    true_label = y_test[random_index]

    # Add the Channel dimension: (28, 28) -> (1,28, 28)
    prepared_image = np.expand_dims(raw_image, axis=0)
    logger.info(f"Shape of prepared_image : {prepared_image.shape}")

    pred_label = model.predict(prepared_image).argmax(axis=1)
    print(pred_label)

    image_for_plot = np.squeeze(prepared_image)
    plt.imshow(image_for_plot, cmap="gray")
    plt.xlabel(f"true_label;{true_label}")
    plt.show()
    logger.info("🎉 Training pipeline completed successfully!")
