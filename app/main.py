from fastapi import FastAPI
from app.api.predict import router as predict_router

app = FastAPI(title="Crop Disease Detection API")

app.include_router(predict_router)
