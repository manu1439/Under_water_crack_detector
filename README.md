# AquaGuard AI — Underwater Infrastructure Crack Detection System

**AquaGuard AI** is an end-to-end, locally executable computer vision and machine learning application engineered for inspecting submerged concrete infrastructure such as bridge piers, dams, retaining walls, and water reservoirs.

The system extracts frames from inspection footage, removes blurry and dark frames, enhances underwater visibility using CLAHE, runs custom YOLO26 crack detection, estimates visual crack severity, and presents real-time telemetry through a high-performance React dashboard.

---

## 1. System Architecture

```text
Underwater Video / Image Upload
         │
         ▼
[Frame Extraction] ──▶ Samples footage at configurable intervals
         │
         ▼
[Intelligent Quality Filter] ──▶ Laplacian variance blur rejection, brightness check, perceptual hash dedup
         │
         ▼
[Underwater Enhancement] ──▶ CLAHE (L-channel adaptive lightness), Gamma, or Retinex
         │
         ▼
[Fine-Tuned YOLO26] ──▶ Local custom crack object detection (class: 'crack')
         │
         ▼
[Severity Assessment] ──▶ Heuristic visual extent, edge density, darkness, aspect ratio (LOW / MED / HIGH)
         │
         ▼
[Annotated Video & PDF Report] ──▶ Compiles annotated frames and exports engineering PDF
         │
         ▼
[AquaGuard Web Dashboard] ──▶ Real-time monitoring, inspection replay, telemetry analytics
```

---

## 2. Key Features

- **100% Local Inference**: Zero cloud API dependencies, zero Roboflow API reliance.
- **Intelligent Frame Filtering**: Automatically discards low-quality underwater frames (murky, dark, or duplicates).
- **Underwater Visibility Restoration**: CLAHE in LAB space, Gamma correction, and Retinex illumination correction.
- **Explainable Severity Estimation**: Separates detector confidence from crack severity; returns human-readable diagnostic reasons.
- **Interactive Inspection Studio**: Click on any detected crack in the results table to instantly jump to and highlight that frame.
- **Automated Engineering Reports**: Export inspection summaries to structured JSON and styled PDF.

---

## 3. Technology Stack

- **Backend**: Python 3.11, FastAPI, OpenCV, Ultralytics YOLO26, PyTorch, SQLAlchemy, aiosqlite, fpdf2, imagehash
- **Frontend**: React 18, Vite, Tailwind CSS, Recharts, Lucide Icons, Axios
- **Storage**: SQLite local database for inspection telemetry and metadata

---

## 4. Quick Start (Windows PowerShell)

### Step 1: Virtual Environment Setup

Open PowerShell in `c:\majorproj`:

```powershell
# Activate existing virtual environment
.\aquaguard_env\Scripts\Activate.ps1

# (Optional) Verify installed packages
pip list
```

### Step 2: Start Backend API Server

```powershell
cd c:\majorproj
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

The FastAPI backend will start at:
- **API Base**: `http://localhost:8000`
- **Interactive Swagger Docs**: `http://localhost:8000/docs`

### Step 3: Start Frontend Dashboard

Open a second PowerShell terminal:

```powershell
cd c:\majorproj\frontend
npm run dev
```

Open your browser at:
- **Dashboard UI**: `http://localhost:5173`

---

## 5. Dataset & YOLO26 Training

The local dataset contains:
- **11,460 Training images** (25,278 crack bounding boxes)
- **368 Validation images** (808 crack bounding boxes)
- **184 Testing images** (368 crack bounding boxes)
- **Single class**: `crack`

### Verify Dataset Integrity

```powershell
python backend/training/train.py --verify-only
```

### Fine-Tune YOLO26 Locally

To manually trigger YOLO26 training on your GPU or CPU:

```powershell
# Train on GPU (CUDA 0)
python backend/training/train.py --model yolo26n.pt --epochs 50 --batch 16 --device 0

# Train on CPU
python backend/training/train.py --model yolo26n.pt --epochs 20 --batch 8 --device cpu
```

The resulting weights are saved directly to `models/best.pt`.

### Evaluate Validation Split

```powershell
python backend/training/validate.py --model models/best.pt --split val
```

---

## 6. Understanding Confidence vs. Severity

- **Detection Confidence (%)**: Indicates how certain the YOLO model is that the detected feature is a physical crack.
- **Estimated Severity (LOW / MEDIUM / HIGH)**: Indicates the apparent physical severity of the crack based on:
  - Surface area ratio relative to the image
  - Canny edge density (crack pattern complexity and branching)
  - Relative pixel darkness
  - Crack elongation and aspect ratio

*Note: Visual severity estimates are designed for inspection triage and do not replace a licensed structural engineer's certified stamp.*

---

## 7. Running Tests

```powershell
pytest backend/tests/test_pipeline.py -v
```
