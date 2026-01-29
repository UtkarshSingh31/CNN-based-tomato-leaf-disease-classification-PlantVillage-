import torch
import torch.nn as nn
from torchvision import models

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(num_classes, model_path):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    for p in model.parameters():
        p.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)


    # Grad-CAM needs gradients here
    for p in model.layer4.parameters():
        p.requires_grad = True

    model.eval()
    model.to(DEVICE)
    return model
