"""
AquaGuard AI — Application configuration.
All settings are configurable via environment variables or .env file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(os.getenv("AQUAGUARD_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
BACKEND_ROOT = PROJECT_ROOT / "backend"
DATASET_ROOT = Path(os.getenv("AQUAGUARD_DATASET_ROOT", PROJECT_ROOT / "dataset"))
MODELS_DIR = Path(os.getenv("AQUAGUARD_MODELS_DIR", PROJECT_ROOT / "models"))
UPLOAD_DIR = Path(os.getenv("AQUAGUARD_UPLOAD_DIR", BACKEND_ROOT / "uploads"))
OUTPUT_DIR = Path(os.getenv("AQUAGUARD_OUTPUT_DIR", BACKEND_ROOT / "outputs"))
DATABASE_URL = os.getenv("AQUAGUARD_DATABASE_URL", f"sqlite+aiosqlite:///{BACKEND_ROOT / 'aquaguard.db'}")

# Ensure directories exist
for d in [MODELS_DIR, UPLOAD_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Model settings ───────────────────────────────────────────────────────
MODEL_WEIGHTS = os.getenv("AQUAGUARD_MODEL_WEIGHTS", str(MODELS_DIR / "best.pt"))
DEFAULT_BASE_MODEL = os.getenv("AQUAGUARD_BASE_MODEL", "yolo26n.pt")
INFERENCE_IMG_SIZE = int(os.getenv("AQUAGUARD_INFERENCE_IMG_SIZE", "640"))
DEFAULT_CONFIDENCE = float(os.getenv("AQUAGUARD_DEFAULT_CONFIDENCE", "0.25"))
DEFAULT_IOU = float(os.getenv("AQUAGUARD_DEFAULT_IOU", "0.45"))
DEVICE = os.getenv("AQUAGUARD_DEVICE", "auto")  # auto, cpu, cuda, 0, 1, ...

# ── Frame extraction settings ────────────────────────────────────────────
DEFAULT_FRAME_INTERVAL = int(os.getenv("AQUAGUARD_FRAME_INTERVAL", "2"))  # every N frames
BLUR_THRESHOLD = float(os.getenv("AQUAGUARD_BLUR_THRESHOLD", "20.0"))
BRIGHTNESS_MIN = float(os.getenv("AQUAGUARD_BRIGHTNESS_MIN", "15.0"))
BRIGHTNESS_MAX = float(os.getenv("AQUAGUARD_BRIGHTNESS_MAX", "240.0"))
CONTRAST_MIN = float(os.getenv("AQUAGUARD_CONTRAST_MIN", "5.0"))
DUPLICATE_HASH_THRESHOLD = int(os.getenv("AQUAGUARD_DUPLICATE_HASH_THRESHOLD", "8"))

# ── Enhancement settings ─────────────────────────────────────────────────
CLAHE_CLIP_LIMIT = float(os.getenv("AQUAGUARD_CLAHE_CLIP_LIMIT", "3.0"))
CLAHE_TILE_GRID = int(os.getenv("AQUAGUARD_CLAHE_TILE_GRID", "8"))
DEFAULT_GAMMA = float(os.getenv("AQUAGUARD_DEFAULT_GAMMA", "1.5"))

# ── Training settings ────────────────────────────────────────────────────
DEFAULT_EPOCHS = int(os.getenv("AQUAGUARD_EPOCHS", "50"))
DEFAULT_BATCH_SIZE = int(os.getenv("AQUAGUARD_BATCH_SIZE", "16"))
DEFAULT_WORKERS = int(os.getenv("AQUAGUARD_WORKERS", "4"))
TRAIN_IMG_SIZE = int(os.getenv("AQUAGUARD_TRAIN_IMG_SIZE", "640"))

# ── Upload limits ─────────────────────────────────────────────────────────
MAX_UPLOAD_SIZE_MB = int(os.getenv("AQUAGUARD_MAX_UPLOAD_SIZE_MB", "500"))
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# ── Server ────────────────────────────────────────────────────────────────
API_HOST = os.getenv("AQUAGUARD_API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("AQUAGUARD_API_PORT", "8000"))
CORS_ORIGINS = os.getenv("AQUAGUARD_CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
