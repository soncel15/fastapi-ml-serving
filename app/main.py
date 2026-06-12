"""
FastAPI ML Serving Application
"""
from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.routers import predict, models, health
from app.services.model_loader import ModelRegistry
from app.middleware.metrics import MetricsMiddleware
from app.config import Settings

settings = Settings()
registry = ModelRegistry()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load models
    registry.load_from_config(settings.model_config)
    yield
    # Shutdown: cleanup
    registry.cleanup()


app = FastAPI(
    title="ML Model Server",
    description="Production ML model serving with FastAPI",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(MetricsMiddleware)
app.state.registry = registry

app.include_router(predict.router, prefix="/predict", tags=["prediction"])
app.include_router(models.router, prefix="/models", tags=["models"])
app.include_router(health.router, tags=["health"])
