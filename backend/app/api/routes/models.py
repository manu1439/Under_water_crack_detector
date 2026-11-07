"""
AquaGuard AI — Model Management API Endpoints.
Handles model status checking, dataset verification queries,
manual training triggers, and model validation evaluations.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, status

from backend.app.config import (
    DATASET_ROOT,
    DEFAULT_BASE_MODEL,
    DEVICE,
    MODELS_DIR,
    MODEL_WEIGHTS,
)
from backend.app.schemas.models import (
    DatasetInfo,
    ModelStatus,
    TrainingConfig,
    TrainingStatus,
    ValidationMetrics,
)
from backend.app.services.crack_detection import get_detector
from backend.training.train import train_crack_model, verify_dataset
from backend.training.validate import validate_crack_model

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/models", tags=["Model Management"])

_train_executor = ThreadPoolExecutor(max_workers=1)

# Training status tracking
_current_training_status = TrainingStatus(is_training=False)


@router.get("/status", response_model=ModelStatus)
async def get_model_status():
    """
    Check availability, path, size, and device for the local YOLO26 model.
    """
    weights_path = Path(MODEL_WEIGHTS)
    exists = weights_path.exists()
    size_mb = 0.0
    if exists:
        try:
            size_mb = round(weights_path.stat().st_size / (1024 * 1024), 2)
        except Exception:
            pass

    detector = get_detector()

    device_str = detector.device
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        device_str = f"{gpu_name} (CUDA:0)"

    return ModelStatus(
        model_loaded=detector.is_loaded,
        model_path=str(weights_path),
        model_exists=exists,
        model_size_mb=size_mb,
        base_model=DEFAULT_BASE_MODEL,
        device=device_str,
        classes=detector.class_names if detector.is_loaded else ["crack"],
        inference_img_size=detector.img_size,
    )


@router.get("/dataset", response_model=DatasetInfo)
async def get_dataset_info():
    """
    Inspect the local dataset directory and return verified image, label, and split counts.
    """
    dataset_dir = DATASET_ROOT
    if not dataset_dir.exists():
        return DatasetInfo(exists=False)

    try:
        stats = verify_dataset(dataset_dir)
        splits = stats.get("splits", {})

        return DatasetInfo(
            exists=True,
            train_images=splits.get("train", {}).get("images", 0),
            train_labels=splits.get("train", {}).get("labels", 0),
            valid_images=splits.get("valid", {}).get("images", 0),
            valid_labels=splits.get("valid", {}).get("labels", 0),
            test_images=splits.get("test", {}).get("images", 0),
            test_labels=splits.get("test", {}).get("labels", 0),
            classes=stats.get("classes", ["crack"]),
            num_classes=stats.get("num_classes", 1),
            data_yaml_path=str(dataset_dir / "data.yaml"),
        )
    except Exception as e:
        logger.error(f"Error querying dataset info: {e}")
        return DatasetInfo(exists=False)


def _run_training_worker(cfg: TrainingConfig):
    """Worker executed in background thread."""
    global _current_training_status
    _current_training_status = TrainingStatus(
        is_training=True,
        current_epoch=0,
        total_epochs=cfg.epochs,
        progress_percent=0.0,
    )

    def epoch_progress(ep: int, tot: int, pct: float):
        global _current_training_status
        _current_training_status.current_epoch = ep
        _current_training_status.total_epochs = tot
        _current_training_status.progress_percent = pct

    try:
        train_crack_model(
            base_model=cfg.base_model,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            img_size=cfg.img_size,
            workers=cfg.workers,
            device=cfg.device,
            resume=cfg.resume,
            epoch_callback=epoch_progress,
        )
        _current_training_status = TrainingStatus(
            is_training=False,
            current_epoch=cfg.epochs,
            total_epochs=cfg.epochs,
            progress_percent=100.0,
            metrics={"status": "completed"},
        )
    except Exception as e:
        logger.error(f"Training worker error: {e}", exc_info=True)
        _current_training_status = TrainingStatus(
            is_training=False,
            error=str(e),
        )


@router.post("/train", response_model=TrainingStatus)
async def trigger_training(
    cfg: TrainingConfig = TrainingConfig(),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """
    Manually trigger YOLO26 fine-tuning.
    Will NEVER be called automatically on startup.
    """
    global _current_training_status
    if _current_training_status.is_training:
        return _current_training_status

    _current_training_status = TrainingStatus(
        is_training=True,
        total_epochs=cfg.epochs,
        progress_percent=1.0,
    )

    loop = asyncio.get_running_loop()
    loop.run_in_executor(_train_executor, _run_training_worker, cfg)

    return _current_training_status


@router.get("/train/status", response_model=TrainingStatus)
async def get_training_status():
    """
    Poll training progress.
    """
    return _current_training_status


@router.post("/validate", response_model=ValidationMetrics)
async def validate_model(
    split: str = "val",
    imgsz: int = 640,
    conf: float = 0.25,
):
    """
    Run validation against validation or test split and return precision, recall, mAP50, mAP50-95.
    """
    weights_path = Path(MODEL_WEIGHTS)
    if not weights_path.exists():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Trained model weights (best.pt) do not exist yet. Train a model first.",
        )

    loop = asyncio.get_running_loop()
    try:
        metrics = await loop.run_in_executor(
            _train_executor,
            validate_crack_model,
            weights_path,
            DATASET_ROOT,
            split,
            imgsz,
            conf,
        )
        return ValidationMetrics(
            precision=metrics.get("precision", 0.0),
            recall=metrics.get("recall", 0.0),
            mAP50=metrics.get("mAP50", 0.0),
            mAP50_95=metrics.get("mAP50_95", 0.0),
        )
    except Exception as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=500, detail=f"Validation execution failed: {str(e)}")
