"""
AquaGuard AI — YOLO26 Model Validation and Evaluation Script.

Evaluates trained model weights against the validation or test split,
computes precision, recall, mAP50, mAP50-95, and exports performance metrics.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.app.config import (
    DATASET_ROOT,
    DEFAULT_IOU,
    DEFAULT_CONFIDENCE,
    DEVICE,
    MODELS_DIR,
    TRAIN_IMG_SIZE,
)
from backend.training.train import prepare_data_yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("aquaguard.validate")


def validate_crack_model(
    model_path: Path | str | None = None,
    dataset_dir: Path | None = None,
    split: str = "val",
    img_size: int = TRAIN_IMG_SIZE,
    conf_thresh: float = DEFAULT_CONFIDENCE,
    iou_thresh: float = DEFAULT_IOU,
    device: str = DEVICE,
) -> dict:
    """
    Validate the crack detection model on the specified split.
    """
    model_path = Path(model_path) if model_path else MODELS_DIR / "best.pt"
    dataset_dir = Path(dataset_dir) if dataset_dir else DATASET_ROOT

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model weights not found at {model_path}. Train a model first or check path."
        )

    resolved_yaml = prepare_data_yaml(dataset_dir)

    if device == "auto":
        import torch
        device = "0" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading weights from: {model_path}")
    logger.info(f"Evaluating on split: '{split}' | Device: {device} | ImgSz: {img_size}")

    model = YOLO(str(model_path))

    metrics = model.val(
        data=str(resolved_yaml),
        split=split,
        imgsz=img_size,
        conf=conf_thresh,
        iou=iou_thresh,
        device=device,
        save_json=True,
        plots=True,
        verbose=True,
    )

    results = {
        "model_path": str(model_path),
        "split": split,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "fitness": float(metrics.fitness),
    }

    logger.info("=" * 50)
    logger.info("VALIDATION RESULTS SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Precision:   {results['precision']:.4f}")
    logger.info(f"Recall:      {results['recall']:.4f}")
    logger.info(f"mAP@50:      {results['mAP50']:.4f}")
    logger.info(f"mAP@50-95:   {results['mAP50_95']:.4f}")
    logger.info("=" * 50)

    # Save validation output summary next to weights
    metrics_path = model_path.parent / f"val_metrics_{split}.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AquaGuard AI — Model Validation")
    parser.add_argument("--model", type=str, default=str(MODELS_DIR / "best.pt"), help="Path to weights file")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--imgsz", type=int, default=TRAIN_IMG_SIZE, help="Image resolution")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONFIDENCE, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU, help="IoU threshold")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu, auto, 0)")

    args = parser.parse_args()

    validate_crack_model(
        model_path=args.model,
        split=args.split,
        img_size=args.imgsz,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        device=args.device,
    )
