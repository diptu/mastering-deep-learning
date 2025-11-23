from app.inference.predictor import Predictor
from app.schemas.predict import ChurnRequest
from fastapi import APIRouter

router = APIRouter(prefix="/v1/churn", tags=["Churn Prediction"])

# Initialize Predictor once
predictor = Predictor()


@router.post("/predict")
async def predict_churn(payload: ChurnRequest):
    """
    Predict customer churn and return a user-friendly response.
    """
    input_data = payload.model_dump()
    result = predictor.predict(input_data)

    prediction_label = "Churn" if result["prediction"] == 1 else "No Churn"
    probability_pct = round(result["probability"] * 100, 2)

    return {
        "success": True,
        "prediction": prediction_label,
        "probability": f"{probability_pct}%",
        "message": f"The model predicts '{prediction_label}' with probability {probability_pct}%.",
    }
