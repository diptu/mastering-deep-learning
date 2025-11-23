from app.api.v1.routes.churn import router as churn_router
from fastapi import FastAPI

app = FastAPI(title="Customer Churn Prediction API")

app.include_router(churn_router)
