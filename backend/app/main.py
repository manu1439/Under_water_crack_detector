"""
AquaGuard AI — FastAPI Application Entry Point.
Provides REST API for underwater crack detection, image enhancement,
video pipeline orchestration, severity assessment, and reporting.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.app.api.routes import inspection, models, reports
from backend.app.config import (
    API_HOST,
    API_PORT,
    BACKEND_ROOT,
    CORS_ORIGINS,
    OUTPUT_DIR,
    UPLOAD_DIR,
)
from backend.app.database.models import init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("aquaguard.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle management: initialize database on startup.
    """
    logger.info("Initializing AquaGuard AI backend...")
    await init_db()
    logger.info("Database initialized successfully.")
    yield
    logger.info("Shutting down AquaGuard AI backend...")


app = FastAPI(
    title="AquaGuard AI API",
    description="Underwater infrastructure crack detection, image enhancement, and severity assessment system.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for local dev / frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers under /api
app.include_router(inspection.router, prefix="/api")
app.include_router(models.router, prefix="/api")
app.include_router(reports.router, prefix="/api")

# Static mounting for direct media viewing if needed
app.mount("/static/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

# Serve frontend build if present
frontend_dist = BACKEND_ROOT.parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/assets", StaticFiles(directory=str(frontend_dist / "assets")), name="assets")

    @app.get("/{full_path:path}", tags=["Frontend"])
    async def serve_frontend(full_path: str):
        if full_path.startswith("api") or full_path.startswith("static"):
            return None
        file_path = frontend_dist / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(frontend_dist / "index.html")


@app.get("/", tags=["Root"])
async def root():
    return {
        "system": "AquaGuard AI",
        "description": "Underwater Crack Detection, Image Enhancement, and Severity Assessment System",
        "status": "online",
        "docs_url": "/docs",
        "api_prefix": "/api",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host=API_HOST, port=API_PORT, reload=True)
