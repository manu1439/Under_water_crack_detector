"""
AquaGuard AI — YOLO26 Training Pipeline for Underwater Crack Detection.

Validates the local dataset, configures Ultralytics YOLO26 fine-tuning,
executes training, saves model weights, logs metrics, and exports validation plots.
"""

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.app.config import (
    DATASET_ROOT,
    DEFAULT_BASE_MODEL,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_WORKERS,
    DEVICE,
    MODELS_DIR,
    TRAIN_IMG_SIZE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("aquaguard.train")


def verify_dataset(dataset_dir: Path) -> dict:
    """
    Thoroughly inspect and verify the dataset structure, splits, and annotations.
    Returns dataset statistics dictionary.
    """
    logger.info("=" * 60)
    logger.info("DATASET VERIFICATION SUMMARY")
    logger.info("=" * 60)

    data_yaml_path = dataset_dir / "data.yaml"
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found at {data_yaml_path}")

    with open(data_yaml_path, "r") as f:
        data_cfg = yaml.safe_load(f)

    logger.info(f"Dataset root: {dataset_dir}")
    logger.info(f"Declared classes ({data_cfg.get('nc', 1)}): {data_cfg.get('names')}")

    stats = {
        "classes": data_cfg.get("names", ["crack"]),
        "num_classes": data_cfg.get("nc", 1),
        "splits": {},
    }

    total_images = 0
    total_labels = 0
    total_annotations = 0

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for split in ["train", "valid", "test"]:
        img_dir = dataset_dir / split / "images"
        lbl_dir = dataset_dir / split / "labels"

        if not img_dir.exists():
            logger.warning(f"Split '{split}' images folder missing: {img_dir}")
            continue

        images = [f for f in img_dir.iterdir() if f.suffix.lower() in valid_extensions]
        labels = [f for f in lbl_dir.iterdir() if f.suffix.lower() == ".txt"] if lbl_dir.exists() else []

        # Count individual bounding boxes
        split_annotations = 0
        corrupt_labels = 0
        for lf in labels:
            try:
                with open(lf, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split()
                            if len(parts) >= 5:
                                split_annotations += 1
            except Exception:
                corrupt_labels += 1

        stats["splits"][split] = {
            "images": len(images),
            "labels": len(labels),
            "annotations": split_annotations,
            "corrupt_labels": corrupt_labels,
        }

        total_images += len(images)
        total_labels += len(labels)
        total_annotations += split_annotations

        logger.info(
            f"Split [{split:5s}]: {len(images):6d} images | "
            f"{len(labels):6d} label files | {split_annotations:6d} boxes"
        )

    logger.info("-" * 60)
    logger.info(
        f"TOTAL: {total_images} images | {total_labels} label files | "
        f"{total_annotations} annotated crack boxes"
    )
    logger.info("=" * 60)

    if total_images == 0:
        raise ValueError(f"No images found in dataset splits at {dataset_dir}")

    return stats


def prepare_data_yaml(dataset_dir: Path) -> Path:
    """
    Ensure data.yaml has absolute paths to avoid Ultralytics path resolution issues.
    Returns path to the resolved YAML file.
    """
    original_yaml = dataset_dir / "data.yaml"
    with open(original_yaml, "r") as f:
        data_cfg = yaml.safe_load(f)

    # Convert paths to absolute
    data_cfg["path"] = str(dataset_dir.resolve()).replace("\\", "/")
    data_cfg["train"] = "train/images"
    data_cfg["val"] = "valid/images"
    data_cfg["test"] = "test/images"

    resolved_yaml = dataset_dir / "data_resolved.yaml"
    with open(resolved_yaml, "w") as f:
        yaml.dump(data_cfg, f, default_flow_style=False)

    return resolved_yaml


def train_crack_model(
    base_model: str = DEFAULT_BASE_MODEL,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    img_size: int = TRAIN_IMG_SIZE,
    workers: int = DEFAULT_WORKERS,
    device: str = DEVICE,
    resume: bool = False,
    dataset_dir: Path | None = None,
    output_dir: Path | None = None,
    epoch_callback: None | object = None,
) -> Path:
    """
    Execute YOLO26 fine-tuning on the underwater crack dataset.
    Returns path to the best trained weights (best.pt).
    """
    dataset_dir = dataset_dir or DATASET_ROOT
    output_dir = output_dir or MODELS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Verify dataset
    stats = verify_dataset(dataset_dir)

    # 2. Prepare resolved YAML
    resolved_yaml = prepare_data_yaml(dataset_dir)

    # 3. Resolve device
    if device == "auto":
        import torch
        device = "0" if torch.cuda.is_available() else "cpu"

    logger.info(f"Starting training on device: {device.upper()}")
    logger.info(
        f"Configuration: model={base_model}, epochs={epochs}, "
        f"batch={batch_size}, imgsz={img_size}, workers={workers}"
    )

    # 4. Initialize YOLO model
    # If base model doesn't exist locally, Ultralytics will automatically fetch base architecture weights
    model = YOLO(base_model)

    if epoch_callback and callable(epoch_callback):
        def on_train_epoch_end(trainer):
            ep = trainer.epoch + 1
            tot = trainer.epochs
            pct = round((ep / tot) * 100, 1)
            try:
                epoch_callback(ep, tot, pct)
            except Exception:
                pass
        model.add_callback("on_train_epoch_end", on_train_epoch_end)

    # 5. Execute training run
    project_run_dir = dataset_dir.parent / "runs" / "train"
    results = model.train(
        data=str(resolved_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        workers=workers,
        resume=resume,
        project=str(project_run_dir),
        name="aquaguard_crack",
        exist_ok=True,
        save=True,
        plots=True,
        verbose=True,
    )

    # 6. Locate best weights
    best_weights = project_run_dir / "aquaguard_crack" / "weights" / "best.pt"
    target_weights = output_dir / "best.pt"

    if best_weights.exists():
        shutil.copy2(best_weights, target_weights)
        logger.info(f"Successfully copied best weights to: {target_weights}")
    else:
        last_weights = project_run_dir / "aquaguard_crack" / "weights" / "last.pt"
        if last_weights.exists():
            shutil.copy2(last_weights, target_weights)
            logger.warning(f"best.pt not found; copied last.pt to: {target_weights}")
        else:
            raise FileNotFoundError(f"No trained weights found in {project_run_dir}")

    # 7. Save metadata
    meta = {
        "base_model": base_model,
        "epochs": epochs,
        "batch_size": batch_size,
        "img_size": img_size,
        "device": device,
        "dataset_stats": stats,
        "weights_path": str(target_weights),
    }
    with open(output_dir / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Training successfully completed. Model ready at: {target_weights}")
    return target_weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AquaGuard AI — YOLO26 Crack Detector Training")
    parser.add_argument("--model", type=str, default=DEFAULT_BASE_MODEL, help="Base model (e.g. yolo26n.pt, yolo26s.pt)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=TRAIN_IMG_SIZE, help="Image resolution")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="DataLoader workers")
    parser.add_argument("--device", type=str, default="auto", help="Device: 'auto', 'cpu', or GPU index '0'")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted training")
    parser.add_argument("--verify-only", action="store_true", help="Only verify dataset without training")

    args = parser.parse_args()

    if args.verify_only:
        verify_dataset(DATASET_ROOT)
    else:
        train_crack_model(
            base_model=args.model,
            epochs=args.epochs,
            batch_size=args.batch,
            img_size=args.imgsz,
            workers=args.workers,
            device=args.device,
            resume=args.resume,
        )
