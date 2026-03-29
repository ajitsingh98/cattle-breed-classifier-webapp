# 🐄 Indigenous Cattle Breed Classifier

AI-powered classification system for **26 indigenous Indian cattle and buffalo breeds**. Trains and compares 4 deep learning models, deploys the best behind a FastAPI API, and serves a modern React web app.

![Python](https://img.shields.io/badge/Python-3.11-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red) ![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green) ![React](https://img.shields.io/badge/React-18-61DAFB) ![License](https://img.shields.io/badge/License-MIT-yellow)

## Features

- **4 Model Architectures**: MLP, CNN (scratch), ResNet50 (transfer), ViT-B/16 (transfer)
- **Weighted Model Selection**: 50% F1 + 20% Accuracy + 15% Latency + 10% Size + 5% Calibration
- **3 Input Modes**: Image upload, camera capture, URL
- **Breed Metadata**: Region, milk yield, lifespan, primary use for all 26 breeds
- **FastAPI Backend**: `/predict/file`, `/predict/url`, `/predict/base64`, `/breeds`
- **React Frontend**: Premium dark UI with glassmorphism, micro-animations, responsive layout
- **Docker Deployment**: Backend + Frontend via `docker-compose`

## Architecture

```
React App ──→ FastAPI /predict/* ──→ Inference Service
                                       ├── Image Preprocessor
                                       ├── Best Model Checkpoint
                                       ├── Breed Metadata Lookup
                                       └── Response Formatter

Training Pipeline ──→ Shared ML Package
                       ├── 4 Model Architectures
                       ├── Unified Trainer
                       ├── Evaluation (metrics, plots)
                       └── Comparison & Selection
```

## Quick Start

### 1. Backend

```bash
cd cattle-breed-classifier-webapp
pip install -r backend/requirements.txt
PYTHONPATH=. uvicorn backend.app.main:app --reload --port 8000
```

API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### 2. Frontend

```bash
cd frontend
npm install
VITE_API_URL=http://localhost:8000 npm run dev
```

Open [http://localhost:5173](http://localhost:5173)

### 3. Docker

```bash
docker-compose up --build
# Backend: http://localhost:8000
# Frontend: http://localhost:3000
```

## Training

### Prepare Dataset

```bash
PYTHONPATH=. python ml/src/data/prepare_dataset.py
```

### Run Notebooks

Each notebook in `ml/notebooks/` trains one model and saves artifacts:

| Notebook | Model | Purpose |
|----------|-------|---------|
| `01_mlp_baseline.ipynb` | MLP | Weak baseline |
| `02_cnn_from_scratch.ipynb` | CNN (5 blocks) | From-scratch comparison |
| `03_resnet_transfer_learning.ipynb` | ResNet50 | Transfer learning |
| `04_vit_transfer_learning.ipynb` | ViT-B/16 | Transformer approach |
| `05_model_comparison.ipynb` | — | Weighted comparison & selection |

### Configuration

YAML configs in `ml/configs/`:
- `base.yaml` — shared settings (image size, augmentation, data paths)
- `mlp.yaml`, `cnn.yaml`, `resnet.yaml`, `vit.yaml` — per-model hyperparameters

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/version` | GET | Model + API version |
| `/predict/file` | POST | Classify uploaded image |
| `/predict/url` | POST | Classify from image URL |
| `/predict/base64` | POST | Classify from base64 |
| `/breeds` | GET | List all breeds |
| `/breeds/{name}` | GET | Breed detail |

### Sample Response

```json
{
  "predicted_breed": "Gir Cow",
  "confidence": 0.91,
  "top_k": [
    {"breed": "Gir Cow", "confidence": 0.91},
    {"breed": "Sahiwal Cow", "confidence": 0.05}
  ],
  "breed_info": {
    "region": "Gujarat, India",
    "avg_milk_liters_per_day": "6-10",
    "lifespan_years": "12-15",
    "primary_use": "Dairy"
  },
  "inference_time_ms": 45.2
}
```

## Project Structure

```
├── backend/              # FastAPI inference service
│   ├── app/
│   │   ├── api/routes/   # health, predict, metadata
│   │   ├── core/         # config, logging
│   │   ├── schemas/      # Pydantic models
│   │   └── services/     # inference, image_loader, breed_info
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/             # React + Vite
│   ├── src/
│   │   ├── pages/        # Home, Predict, BreedExplorer, About
│   │   ├── services/     # API client
│   │   └── hooks/        # usePredictionHistory
│   └── Dockerfile
├── ml/                   # Machine learning package
│   ├── src/
│   │   ├── data/         # prepare_dataset, transforms, metadata
│   │   ├── models/       # mlp, cnn, resnet, vit
│   │   ├── training/     # trainer, engine, callbacks, factories
│   │   ├── evaluation/   # metrics, plots, compare_models
│   │   └── inference/    # predictor, preprocess
│   ├── configs/          # YAML configs per model
│   └── notebooks/        # Training notebooks
├── data/
│   └── breed_metadata.csv
├── Cattle_Resized/       # Raw image dataset (26 breeds)
├── models/               # Saved checkpoints
├── docker-compose.yml
└── README.md
```

## Breeds Supported (26)

**Cows (21)**: Alambadi, Amritmahal, Bargur, Dangi, Deoni, Gir, Hallikar, Kangayam, Kankrej, Kasaragod, Kenkatha, Kherigarh, Malnad Gidda, Nagori, Nimari, Pulikulam, Rathi, Sahiwal, Tharparkar, Umblachery

**Buffaloes (5)**: Banni, Jaffrabadi, Mehsana, Nagpuri, Nili Ravi, Shurti

## Built By

[Ajit Kumar Singh](https://www.linkedin.com/in/sajit9285/) — [GitHub](https://github.com/sajit9285/cattle-breed-classifier-webapp)
