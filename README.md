# AquaGuard AI — Underwater Infrastructure Crack Detection & Assessment System

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6%20%7C%20CUDA%2012.4-EE4C2C.svg)](https://pytorch.org/)
[![YOLO26](https://img.shields.io/badge/Model-Fine--Tuned%20YOLO26-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![React 18](https://img.shields.io/badge/Frontend-React%2018%20%7C%20Tailwind-61DAFB.svg)](https://react.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**AquaGuard AI** is a locally deployable, modular computer vision and machine learning application engineered for inspecting submerged concrete infrastructure—including bridge piers, dams, retaining walls, caissons, and sea water intakes.

The system ingests inspection videos or high-resolution still images, filters out turbid or blurry frames, enhances underwater visibility using CLAHE in LAB color space, detects concrete cracks with a custom fine-tuned **YOLO26** model, evaluates physical defect severity (`LOW`, `MEDIUM`, `HIGH`) with explainable heuristics, and generates comprehensive engineering **PDF** and **JSON** inspection reports.

---

## 1. Key Capabilities

- **100% Local Execution**: Completely offline, self-contained inference without cloud dependencies or external API tokens.
- **Pre-Trained Deep Learning Weights**: Includes fine-tuned weights (`models/best.pt`) trained for 30 epochs on 12,012 underwater images achieving **92.8% Precision** and **82.8% mAP@50**.
- **Dual-Domain Detection**: Concurrently runs inference across both raw underwater frames and CLAHE contrast-enhanced frames, merging detections via IoU spatial matching to avoid color-shift false negatives.
- **Intelligent Frame Quality Selector**: Evaluates Laplacian variance for blur, filters low-contrast murky frames, and eliminates perceptual duplicates using pHash.
- **Physical Severity Diagnostics**: Classifies defects into `LOW`, `MEDIUM`, or `HIGH` severity using crack surface area ratio, Canny edge density, pixel darkness, and aspect ratio.
- **Automated Engineering Reports**: One-click download of multi-page PDF inspection reports with keyframe snapshots, defect dimensions, severity diagnostics, and metadata.
- **Single-Port Architecture**: The FastAPI backend directly serves the pre-compiled React 18 production build on port 8000.

---

## 2. System Requirements

| Component | Minimum Requirement | Recommended |
| :--- | :--- | :--- |
| **Operating System** | Windows 10/11, Ubuntu 20.04+, or macOS 12+ | Windows 11 (64-bit) / Ubuntu 22.04 |
| **Python** | Python 3.10 or 3.11 | Python 3.11.8 |
| **Node.js** *(Optional)* | Node.js 18+ (only needed if recompiling frontend) | Node.js 20 LTS |
| **RAM** | 8 GB System Memory | 16 GB+ |
| **GPU Acceleration** | CPU inference supported | NVIDIA GPU (RTX 2000 Ada / RTX 3060+ with 4GB+ VRAM) |
| **Storage** | 3 GB available disk space | 10 GB (for video processing outputs) |

---

## 3. Step-by-Step Installation on Any Machine

Follow these exact steps to set up and run AquaGuard AI on a fresh machine.

### Step 1: Open Terminal in Project Root
Navigate to the root directory where the project is cloned or extracted:
```bash
cd c:\majorproj
```
*(On Linux/macOS, navigate to `/path/to/majorproj`).*

---

### Step 2: Create and Activate a Python Virtual Environment

#### On Windows (PowerShell):
```powershell
# If script execution is restricted on your machine, enable it:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned

# Create virtual environment
python -m venv aquaguard_env

# Activate virtual environment
.\aquaguard_env\Scripts\Activate.ps1
```

#### On Windows (Command Prompt `cmd`):
```cmd
python -m venv aquaguard_env
aquaguard_env\Scripts\activate.bat
```

#### On Linux / macOS (Bash / Zsh):
```bash
python3 -m venv aquaguard_env
source aquaguard_env/bin/activate
```

---

### Step 3: Install Python Dependencies

1. Upgrade `pip` to the latest version:
   ```bash
   python -m pip install --upgrade pip
   ```

2. Install all core application dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. **(Recommended for NVIDIA GPU owners)** Install PyTorch with CUDA acceleration:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   ```

4. Verify your environment:
   ```bash
   python -c "import torch; print('PyTorch Version:', torch.__version__); print('CUDA Available:', torch.cuda.is_available())"
   ```
   *If `CUDA Available: True` appears, GPU acceleration is active. If `False`, AquaGuard AI will smoothly run on CPU.*

---

### Step 4: Frontend Verification (No Node.js Required)

AquaGuard AI comes with the production React frontend **already compiled** inside [`frontend/dist/`](file:///c:/majorproj/frontend/dist/). 

You do **not** need to install Node.js or run npm commands just to use the application. The FastAPI backend automatically serves the pre-built web interface at `http://localhost:8000`.

*(If you plan to modify the React frontend code, install Node.js and run `cd frontend && npm install && npm run build`).*

---

## 4. How to Run AquaGuard AI

### Method 1: The One-Click Windows Launcher (Easiest)
On Windows, simply double-click:
```cmd
start_aquaguard.bat
```
This automatically finds your Python virtual environment, launches the backend server, and opens **`http://localhost:8000`** in your default web browser.

---

### Method 2: Manual Terminal Launch (Cross-Platform)

Make sure your virtual environment is activated, then run:

```bash
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser and navigate to:
- **Application Dashboard**: [http://localhost:8000](http://localhost:8000)
- **Interactive REST API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

### Method 3: Frontend Development Mode (For UI Developers)

If you wish to edit React UI components with live Hot Module Replacement (HMR):

1. **Terminal 1 (Backend API):**
   ```bash
   python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Terminal 2 (Vite Frontend):**
   ```bash
   cd frontend
   npm run dev
   ```
   Open [http://localhost:5173](http://localhost:5173) in your browser.

---

### Method 4: Docker Deployment

AquaGuard AI can also be run entirely inside an isolated container:

```bash
docker-compose up --build
```
Access the application at `http://localhost:8000`.

---

## 5. How to Run an Underwater Inspection

1. **Access the Dashboard**: Open `http://localhost:8000` and click the **"New Inspection"** tab.
2. **Select Footage**: Upload underwater inspection media (MP4, AVI, MOV, or JPG/PNG photos). A sample underwater video is provided at:
   ```
   input/sample_inspection.mp4
   ```
3. **Configure Vision Pipeline Parameters**:
   - **Frame Extraction Interval**: Set to `1` or `2` (extracts every frame for complete coverage).
   - **Frame Selection Mode**: Click **"Analyze All Frames"** (ensures murky or turbid underwater frames are not prematurely discarded).
   - **Underwater Enhancement**: Keep **CLAHE (Adaptive Lightness - Default)** selected.
   - **Detection Confidence**: Default `25%` (standard for YOLO26 crack detection).
4. **Initiate Analysis**: Click **"Start AI Inspection"**.
5. **Inspect Findings**:
   - Examine the **Annotated Keyframe Viewer** displaying crack bounding boxes, confidence tags, and severity color badges.
   - Click any row in the **Consolidated Frame-Level Crack Detections** table to jump directly to that defect and read its physical diagnostic reasons.
   - Toggle **"Enhancement Comparison (Original vs CLAHE)"** to inspect restored contrast.
   - Click **"Download PDF Report"** for an engineering document or **"JSON Report"** for raw data integration.

---

## 6. Model Training & Evaluation (Optional)

The trained model is already present at [`models/best.pt`](file:///c:/majorproj/models/best.pt). You only need to run training if you wish to fine-tune on additional custom datasets.

### Model Telemetry
- **Base Architecture**: YOLO26 Nano (`yolo26n.pt`)
- **Trained Weights**: `models/best.pt` (5.45 MB)
- **Class Label**: `crack` (Single Class)
- **Metrics on Validation Split**:
  - Precision: **92.8%**
  - Recall: **74.5%**
  - mAP@50: **82.8%**
  - mAP@50-95: **62.0%**
  - Inference Latency: **1.7 ms** per image on GPU

### Re-Training via Batch Script (Windows)
Double-click:
```cmd
train_model.bat
```

### Re-Training via Python CLI
```bash
# Auto-detects GPU (CUDA) or falls back to CPU
python backend/training/train.py --model yolo26n.pt --epochs 30 --batch 16 --device auto

# Evaluate model performance against the validation split
python backend/training/validate.py --model models/best.pt --split val
```

---

## 7. Automated Test Suite

To verify that all components (enhancement, detection, severity assessment, PDF generation) are operating correctly:

```bash
pytest backend/tests/test_pipeline.py -v
```

All 4 integration tests will run and validate the complete pipeline.

---

## 8. Project Directory Structure

```text
majorproj/
├── backend/
│   ├── app/
│   │   ├── main.py                     # FastAPI entrypoint & SPA static router
│   │   ├── config.py                   # Global configuration & directory paths
│   │   ├── api/routes/
│   │   │   ├── inspection.py           # Upload, start inspection, polling API
│   │   │   ├── models.py               # Model telemetry & training API
│   │   │   └── reports.py              # PDF/JSON report download handlers
│   │   ├── database/models.py          # Async SQLAlchemy SQLite ORM models
│   │   ├── schemas/models.py           # Pydantic data schemas
│   │   └── services/
│   │       ├── video_pipeline.py       # End-to-end inspection orchestrator
│   │       ├── frame_selection.py      # Frame extractor, blur/contrast/pHash filter
│   │       ├── image_enhancement.py    # CLAHE, Gamma, Retinex algorithms
│   │       ├── crack_detection.py      # YOLO26 inference & bounding box annotator
│   │       ├── severity_assessment.py  # Multi-factor physical severity calculator
│   │       └── report_generator.py     # PDF & JSON report generator
│   ├── tests/
│   │   └── test_pipeline.py            # Pytest test suite
│   ├── training/
│   │   ├── train.py                    # Ultralytics model training pipeline
│   │   └── validate.py                 # Model evaluation script
│   └── requirements.txt                # Backend dependencies
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── DashboardPage.jsx       # Overview telemetry & recent runs
│   │   │   ├── NewInspectionPage.jsx   # Upload & pipeline parameter controls
│   │   │   ├── ProcessingPage.jsx      # Live progress animation & stage status
│   │   │   ├── ResultsPage.jsx         # Annotated defect viewer & report download
│   │   │   ├── HistoryPage.jsx         # Searchable inspection archive
│   │   │   └── ModelManagementPage.jsx # Model telemetry & training trigger
│   │   ├── App.jsx                     # Route layout & header navigation
│   │   └── index.css                   # Tailwind CSS styles
│   └── dist/                           # Compiled production build served by FastAPI
├── models/
│   ├── best.pt                         # Fine-tuned 30-epoch YOLO26 weights (5.45 MB)
│   └── model_meta.json                 # Model performance metrics & training info
├── input/
│   └── sample_inspection.mp4           # 6-second underwater concrete crack test video
├── outputs/                            # Inspection runs, annotated images, reports
├── start_aquaguard.bat                 # One-click Windows application launcher
├── train_model.bat                     # Standalone model training execution script
├── requirements.txt                    # Root Python dependencies
├── Dockerfile                          # Multi-stage Docker container specification
├── docker-compose.yml                  # Docker Compose orchestrator
├── PROJECT_HANDOVER_SUMMARY.md         # Comprehensive technical handover documentation
└── README.md                           # This setup and operational guide
```

---

## 9. Troubleshooting & Common Questions

#### 1. "Script execution is disabled on this system" (Windows PowerShell)
Run PowerShell as Administrator or in user scope:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

#### 2. "ModuleNotFoundError: No module named 'greenlet'"
SQLAlchemy async SQLite requires the `greenlet` C-extension. Run:
```bash
pip install greenlet
```

#### 3. "Port 8000 is already in use"
Another process is running on port 8000. Either terminate the process or specify a different port:
```bash
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8080
```
Then visit `http://localhost:8080`.

#### 4. "PyTorch reports CUDA Available: False"
Verify that you have an NVIDIA GPU and driver installed, then install the CUDA 12.4 PyTorch wheel:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

#### 5. "Why were 0 cracks detected on my underwater video?"
Underwater footage often has green/blue color cast and low contrast:
- In the **New Inspection** tab, set **Frame Selection Mode** to **"Analyze All Frames"**.
- Set **Frame Extraction Interval** to `1` or `2`.
- Keep **Underwater Enhancement** set to **CLAHE**.
- The built-in **Dual-Domain Detection** will analyze both the raw frame and the CLAHE frame to guarantee faint underwater cracks are identified.

---

## 10. License & Disclaimers

- **License**: MIT License.
- **Engineering Notice**: Severity ratings produced by AquaGuard AI are automated visual estimates designed for inspection triage and prioritized maintenance planning. They do not substitute for a certified structural engineer's on-site physical evaluation.
