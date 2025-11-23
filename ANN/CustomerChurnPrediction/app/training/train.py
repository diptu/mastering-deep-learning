# app/training/train.py
import joblib
from app.core.config import settings
from app.core.logger import get_logger
from app.inference.preprocess import Preprocessor
from app.models.tf_model import build_model, evaluate_model, plot_accuracy, plot_loss

logger = get_logger(__name__)

if __name__ == "__main__":
    logger.info("🚀 Starting training pipeline...")

    # ---------------------------------------------------------
    # 1️⃣ Load + Preprocess + Scale
    # ---------------------------------------------------------
    pre = Preprocessor(target_col="Exited")

    df = pre.preprocess("churn.csv", "processed_churn.csv")

    X_train, X_test, y_train, y_test = pre.split_and_scale(df)

    # ---------------------------------------------------------
    # 2️⃣ Build + Train Model
    # ---------------------------------------------------------
    model = build_model(X_train.shape[1])

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=settings.EPOCHS,
        batch_size=settings.BATCH,
        verbose=1,
    )

    # ---------------------------------------------------------
    # 3️⃣ Save Model + Scaler + Metadata
    # ---------------------------------------------------------
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = settings.MODELS_DIR / "tf_churn_model.h5"
    scaler_path = settings.MODELS_DIR / "scaler.pkl"
    meta_path = settings.MODELS_DIR / "meta.pkl"

    # save model
    model.save(model_path)
    logger.info(f"✔ Model saved to {model_path}")

    # save scaler
    pre.save_fitted_scaler(scaler_path)

    # save expected feature columns
    joblib.dump({"expected_features": pre.expected_features}, meta_path)
    logger.info(f"✔ Metadata saved to {meta_path}")

    # ---------------------------------------------------------
    # 4️⃣ Plot Training Curves
    # ---------------------------------------------------------
    plot_loss(history)
    plot_accuracy(history)

    # ---------------------------------------------------------
    # 5️⃣ Evaluate Model
    # ---------------------------------------------------------
    evaluate_model(model, X_test, y_test)

    logger.info("🎉 Training pipeline completed successfully!")
