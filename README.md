# FastAPI ML Serving 🚀

Production-ready ML model serving framework with FastAPI.

## Features
- **Multi-model support** — serve multiple models with versioning
- **Dynamic batching** — automatic request batching for GPU efficiency
- **A/B testing** — traffic splitting between model versions
- **Health checks** — liveness and readiness probes
- **Metrics** — Prometheus-compatible metrics endpoint
- **Graceful shutdown** — clean request draining

## Architecture
```
app/
├── main.py              # FastAPI application
├── routers/
│   ├── predict.py       # Prediction endpoints
│   ├── models.py        # Model management
│   └── health.py        # Health check endpoints
├── services/
│   ├── model_loader.py  # Dynamic model loading (ONNX, PyTorch)
│   ├── batcher.py       # Request batching engine
│   └── ab_router.py     # A/B test traffic routing
├── middleware/
│   ├── metrics.py       # Prometheus metrics
│   └── logging.py       # Request logging
└── config.py            # Configuration
```

## Quick Start
```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API
```bash
# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"model": "resnet50", "version": "v2", "inputs": [[1.0, 2.0, 3.0]]}'

# Batch prediction
curl -X POST http://localhost:8000/predict/batch \
  -d '{"model": "resnet50", "inputs": [[1,2,3], [4,5,6], [7,8,9]]}'

# Model management
curl http://localhost:8000/models          # List loaded models
curl http://localhost:8000/models/resnet50 # Model info
curl http://localhost:8000/health          # Health check
curl http://localhost:8000/metrics         # Prometheus metrics
```
