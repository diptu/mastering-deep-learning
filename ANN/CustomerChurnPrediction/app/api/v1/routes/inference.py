from fastapi import APIRouter

router = APIRouter()


@router.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest):
    prediction = run_model(request.data)
    return InferenceResponse(prediction=prediction)
