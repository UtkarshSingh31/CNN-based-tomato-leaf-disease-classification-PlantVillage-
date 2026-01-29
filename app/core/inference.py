from fastapi import FastAPI

from app.core.model import load_model, DEVICE
from app.core.gradcam import GradCAM
from app.config.settings import CLASS_NAMES, MODEL_PATH

app = FastAPI(title="Crop Disease Detection API")

model = load_model(len(CLASS_NAMES), MODEL_PATH)
gradcam = GradCAM(model, model.layer4[-1].conv2)
