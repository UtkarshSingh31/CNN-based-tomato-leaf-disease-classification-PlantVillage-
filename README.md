```markdown
# 🌿 Crop Disease Detection & Severity Estimation

An end-to-end **deep learning system** for **tomato leaf disease classification and severity estimation**, deployed as a **Hugging Face Space** using a **hybrid FastAPI + Gradio architecture**.

This project goes beyond basic classification by estimating **disease severity** using **Grad-CAM–based spatial analysis**, making predictions more interpretable and practically useful.

---

## 🚀 Live Demo

**Hugging Face Space**  
👉 https://huggingface.co/spaces/utkarshsingh0013/crop-detection-detection

Upload a tomato leaf image to get:
- Disease name
- Prediction confidence
- Estimated severity (%)

---

## 📌 Problem Statement

Most crop disease models only answer:

> *“What disease is this?”*

In real agricultural settings, farmers also need to know:

> *“How severe is it?”*

Severity determines:
- urgency of treatment
- choice of intervention
- potential yield impact

This project addresses both **disease identification** and **severity estimation** in a single pipeline.

---

## 🧠 Solution Overview

The system performs:

1. **Disease Classification** using a CNN
2. **Model Explainability** using Grad-CAM
3. **Severity Estimation** using spatial activation analysis
4. **Deployment** via API and interactive UI

---

## 🧩 System Architecture

```
User (Browser)
├── Gradio UI (Hugging Face Space)
│     └── Image Upload
│
└── FastAPI Backend (/api/predict)
├── ResNet18 classifier
├── Grad-CAM localization
├── Leaf extraction (GrabCut)
└── Top-K severity computation
```

- **Gradio** → interactive user interface  
- **FastAPI** → reusable backend API  
- **Docker** → reproducible deployment environment  

---

## 🧪 Model Details

- **Architecture**: ResNet18
- **Pretraining**: ImageNet weights
- **Dataset**: PlantVillage (Tomato subset)
- **Number of classes**: 10
- **Input size**: 224 × 224
- **Training strategy**:
  - Frozen backbone
  - Trainable classification head

---

## 🟠 Disease Severity Estimation

### Why confidence is not severity

- A model can be **highly confident** about a disease
- The actual infected area may still be small

Hence, confidence ≠ severity.

---

### Severity Estimation Method

1. **Grad-CAM** highlights regions responsible for the prediction
2. **Leaf-only masking** removes background influence (GrabCut)
3. **Top-K CAM strategy**:
   - Only the strongest activation regions are considered
   - Prevents severity inflation due to diffuse attention

### Severity Definition

> **Severity = percentage of leaf area belonging to the top-K disease-relevant regions**

This produces visually consistent and interpretable estimates.

---

## 📊 Example Output

```json
{
  "disease": "Tomato_Early_blight",
  "confidence": 0.93,
  "severity_percent": 20.0
}
````

---

## 🖥️ User Interface (Gradio)

The Gradio UI provides:

* Image upload
* One-click prediction
* Clear numerical outputs

Designed for:

* Demonstrations
* Academic evaluation
* Non-technical users

---

## 🔌 API Endpoint (FastAPI)

### `POST /api/predict`

**Input**

* Image file (`.jpg`, `.png`)

**Response**

```json
{
  "disease": "Tomato_Late_blight",
  "confidence": 0.91,
  "severity_percent": 27.5
}
```

This API can be consumed by:

* Web applications
* Mobile apps
* External services

---

## 📁 Project Structure

```
crop-disease-detection/
│
├── app/
│   ├── main.py            # FastAPI + Gradio entrypoint
│   ├── api/               # API routes
│   ├── core/              # ML logic (model, Grad-CAM, severity)
│   ├── ui/                # Gradio UI
│   └── config/            # Configuration
│
├── models/
│   └── best_resnet.pth    # Trained model weights
│
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🐳 Deployment (Why Docker)

Docker is used to ensure:

* consistent runtime environment
* OpenCV system dependency support
* reproducibility across machines
* production-like deployment

This is especially important for FastAPI + ML systems.

---

## ⚠️ Limitations

* Severity is **attention-based**, not pixel-level ground truth
* Grad-CAM highlights **model evidence**, not medical diagnosis
* Best suited for:

  * academic projects
  * demonstrations
  * early-stage decision support

---

## 🔮 Future Improvements

* True lesion segmentation
* Multi-crop support
* Severity calibration per disease
* Yield-loss estimation

---

## 🧠 Key Learnings

* Classification alone is insufficient for real-world ML systems
* Explainability improves trust and debugging
* Severity is a **designed metric**, not a free by-product
* Backend architecture is as important as model accuracy

---

## 📜 License

MIT License

---

## 🙌 Acknowledgements

* PlantVillage Dataset
* PyTorch & TorchVision
* Hugging Face Spaces
* Gradio & FastAPI

```
