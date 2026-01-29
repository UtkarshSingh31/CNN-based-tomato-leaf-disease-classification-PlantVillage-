from fastapi import APIRouter, File, UploadFile
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

from app.core.inference import model, gradcam
from app.core.model import DEVICE
from app.core.severity import compute_severity_topk
from app.config.settings import CLASS_NAMES, TOPK_PERCENT

router = APIRouter()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    rgb = np.array(image)

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    input_tensor.requires_grad_(True)

    logits = model(input_tensor)
    probs = torch.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)

    cam = gradcam.generate(input_tensor, pred.item())
    severity = compute_severity_topk(cam, rgb, TOPK_PERCENT)

    return {
        "disease": CLASS_NAMES[pred.item()],
        "confidence": round(conf.item(), 3),
        "severity_percent": round(severity, 2)
    }
