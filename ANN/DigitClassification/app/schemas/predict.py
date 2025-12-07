# app/schemas/predict.py
from typing import List

from fastapi import File, UploadFile
from pydantic import BaseModel


# -----------------------------
# INPUT SCHEMA
# -----------------------------
class DigitPredictRequest(BaseModel):
    """
    Request schema for MNIST digit prediction via image file upload.
    """

    file: UploadFile = File(...)


# -----------------------------
# RESPONSE SCHEMA
# -----------------------------
class DigitPredictResponse(BaseModel):
    """
    Response schema for MNIST digit prediction.
    """

    prediction: int  # Predicted digit (0-9)
    probability: float  # Probability/confidence of predicted class
    probabilities: List[float]  # Probabilities for all 10 classes (0-9)
