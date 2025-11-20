from pydantic import BaseModel


class InferenceRequest(BaseModel):
    data: float


class InferenceResponse(BaseModel):
    prediction: float
