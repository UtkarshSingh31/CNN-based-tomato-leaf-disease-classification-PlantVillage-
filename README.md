🌿 Crop Disease Detection & Severity Estimation

An end-to-end deep learning system for tomato leaf disease classification and severity estimation, deployed as a Hugging Face Space using a hybrid FastAPI + Gradio architecture.

This project goes beyond simple classification by estimating disease severity using Grad-CAM–based spatial analysis, making predictions more interpretable and actionable.

🚀 Live Demo

👉 Hugging Face Space
https://huggingface.co/spaces/<your-username>/crop-disease-detection

Upload a tomato leaf image

Get:

Disease name

Prediction confidence

Estimated severity (%)

📌 Problem Statement

Most crop disease models only answer:

“What disease is this?”

But in real agriculture, farmers also need to know:

“How severe is it?”

Severity determines:

whether treatment is needed

urgency of action

potential yield loss

This project addresses both classification and severity estimation in a single pipeline.

🧠 Solution Overview
What the system does

Classifies tomato leaf diseases using a CNN

Explains predictions using Grad-CAM

Estimates severity based on spatial activation

Serves results via API and interactive UI

🧩 Architecture
User (Browser)
 ├── Gradio UI (Hugging Face Space)
 │     └── Image Upload
 │
 └── FastAPI Backend (/api/predict)
        ├── ResNet18 classifier
        ├── Grad-CAM localization
        ├── Leaf extraction (GrabCut)
        └── Top-K severity computation


Gradio → Human-friendly UI

FastAPI → Clean, reusable API

Docker → Reproducible deployment

🧪 Model Details

Backbone: ResNet18 (ImageNet weights)

Dataset: PlantVillage (Tomato subset, 10 classes)

Training strategy:

Frozen backbone

Trainable classifier head

Input size: 224 × 224

Output:

Disease label

Confidence score

🟠 Disease Severity Estimation (Key Contribution)
Why confidence ≠ severity

High confidence does not mean high damage

A small lesion can be classified confidently

How severity is estimated

Grad-CAM highlights regions responsible for prediction

Leaf-only masking removes background influence (GrabCut)

Top-K CAM analysis:

Only strongest activation regions are considered

Avoids inflated severity from diffuse attention

Severity definition

Severity = percentage of leaf area belonging to the top-K disease-relevant regions

This produces realistic, visually consistent estimates.

📊 Output Example
{
  "disease": "Tomato_Early_blight",
  "confidence": 0.93,
  "severity_percent": 20.0
}

🖥️ User Interface (Gradio)

The Gradio UI provides:

Image upload

One-click prediction

Clear numerical outputs

Designed for:

Demonstrations

Academic evaluation

Non-technical users

🔌 API Endpoint (FastAPI)

The backend exposes a clean API:

POST /api/predict

Input

Image file (.jpg, .png)

Response

{
  "disease": "Tomato_Late_blight",
  "confidence": 0.91,
  "severity_percent": 27.5
}


This allows:

Mobile apps

Web apps

Integration with other systems

📁 Project Structure
crop-disease-detection/
│
├── app/
│   ├── main.py            # FastAPI + Gradio entrypoint
│   ├── api/               # API routes
│   ├── core/              # ML logic (model, Grad-CAM, severity)
│   ├── ui/                # Gradio UI
│   └── config/            # Settings
│
├── models/
│   └── best_resnet.pth    # Trained model weights
│
├── Dockerfile
├── requirements.txt
└── README.md

🐳 Deployment (Why Docker)

This project uses Docker because it requires:

OpenCV system dependencies

FastAPI backend

Gradio UI

Torch + torchvision compatibility

Docker ensures:

Reproducibility

Environment consistency

Production-like behavior

⚠️ Limitations & Notes

Severity is attention-based, not pixel-perfect segmentation

Grad-CAM highlights model evidence, not medical ground truth

Best suited for:

Demos

Academic projects

Early-stage decision support

Future upgrades:

True lesion segmentation

Multi-crop support

Yield-loss estimation

🧠 Key Learnings

Classification alone is insufficient for real-world ML systems

Explainability helps validate and debug models

Severity is a designed metric, not a free by-product

Proper backend architecture matters as much as model accuracy

📜 License

MIT License — free to use, modify, and distribute.

🙌 Acknowledgements

PlantVillage dataset

PyTorch & TorchVision

Hugging Face Spaces

Gradio & FastAPI
