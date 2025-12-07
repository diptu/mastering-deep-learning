# app/api/v1/predict.py
from app.core.logger import get_logger
from app.inference.predictor import Predictor
from app.schemas.predict import DigitPredictResponse
from fastapi import APIRouter, File, HTTPException, UploadFile

logger = get_logger(__name__)
router = APIRouter(prefix="/predict", tags=["Prediction"])

# Load model once at startup
predictor = Predictor()


@router.post("/", response_model=DigitPredictResponse)
async def predict_digit(file: UploadFile = File(...)):
    """
    Accept any image file (PNG/JPG), preprocess to 28x28, and predict MNIST digit.
    """
    try:
        # Read file content
        image_bytes = await file.read()

        # Use Predictor to preprocess and predict
        result = predictor.predict_from_file_bytes(image_bytes)
        return result

    except ValueError as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
