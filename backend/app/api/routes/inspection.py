"""
AquaGuard AI — Inspection API Endpoints.
Handles file uploads, pipeline triggering, progress polling, results retrieval,
annotated video streaming, and inspection history.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.config import (
    ALLOWED_IMAGE_EXTENSIONS,
    ALLOWED_VIDEO_EXTENSIONS,
    MAX_UPLOAD_SIZE_MB,
    OUTPUT_DIR,
    UPLOAD_DIR,
)
from backend.app.database.models import Inspection, get_session
from backend.app.schemas.models import (
    ErrorResponse,
    InspectionConfig,
    InspectionProgress,
    InspectionResultFull,
    InspectionStatus,
    InspectionSummary,
    ProcessResponse,
    UploadResponse,
)
from backend.app.services.video_pipeline import run_pipeline_async
from backend.app.services.report_generator import ReportGenerator

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/inspection", tags=["Inspection"])

# In-memory progress cache for fast polling
_progress_tracker: dict[str, InspectionProgress] = {}


@router.post(
    "/upload",
    response_model=UploadResponse,
    responses={400: {"model": ErrorResponse}},
)
async def upload_file(
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
):
    """
    Upload an inspection video or single image for processing.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a valid filename.",
        )

    ext = Path(file.filename).suffix.lower()
    if ext in ALLOWED_VIDEO_EXTENSIONS:
        file_type = "video"
    elif ext in ALLOWED_IMAGE_EXTENSIONS:
        file_type = "image"
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{ext}'. Supported videos: {ALLOWED_VIDEO_EXTENSIONS}, images: {ALLOWED_IMAGE_EXTENSIONS}",
        )

    inspection_id = str(uuid.uuid4())
    save_dir = UPLOAD_DIR / inspection_id
    save_dir.mkdir(parents=True, exist_ok=True)
    destination = save_dir / file.filename

    # Save uploaded file
    file_size_bytes = 0
    try:
        with open(destination, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                file_size_bytes += len(chunk)
                if file_size_bytes > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
                    buffer.close()
                    if destination.exists():
                        destination.unlink()
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File exceeds maximum allowed limit of {MAX_UPLOAD_SIZE_MB}MB.",
                    )
                buffer.write(chunk)
    except Exception as e:
        if destination.exists():
            destination.unlink()
        if isinstance(e, HTTPException):
            raise e
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {str(e)}")

    file_size_mb = round(file_size_bytes / (1024 * 1024), 2)

    # Initialize inspection record in DB
    inspection = Inspection(
        id=inspection_id,
        filename=file.filename,
        file_type=file_type,
        file_path=str(destination),
        file_size_mb=file_size_mb,
        status=InspectionStatus.PENDING.value,
        output_dir=str(OUTPUT_DIR / inspection_id),
    )
    session.add(inspection)
    await session.commit()

    _progress_tracker[inspection_id] = InspectionProgress(
        inspection_id=inspection_id,
        status=InspectionStatus.PENDING,
        progress_percent=0.0,
        current_stage="File uploaded, waiting to start",
    )

    return UploadResponse(
        inspection_id=inspection_id,
        filename=file.filename,
        file_type=file_type,
        file_size_mb=file_size_mb,
        message="File uploaded successfully. Ready for inspection.",
    )


async def _execute_inspection_task(
    inspection_id: str,
    file_path: str,
    file_type: str,
    config: InspectionConfig,
):
    """
    Background worker that runs the full CV and ML pipeline and persists the outcomes.
    """
    from backend.app.database.models import async_session

    output_base = OUTPUT_DIR / inspection_id
    output_base.mkdir(parents=True, exist_ok=True)

    def progress_callback(prog: InspectionProgress):
        _progress_tracker[inspection_id] = prog

    try:
        is_image = file_type == "image"
        result: InspectionResultFull = await run_pipeline_async(
            video_path=file_path,
            inspection_id=inspection_id,
            config=config,
            output_base=output_base,
            progress_callback=progress_callback,
            is_image=is_image,
        )

        # Generate Reports (JSON and PDF)
        report_gen = ReportGenerator(output_base)
        json_path = report_gen.generate_json_report(result)

        annotated_frames = list((output_base / "annotated").glob("*.jpg")) + list(
            (output_base / "annotated").glob("*.png")
        )
        pdf_path = report_gen.generate_pdf_report(result, annotated_frames)

        result.report_json_path = str(json_path)
        result.report_pdf_path = str(pdf_path)

        # Update DB record
        async with async_session() as session:
            stmt = select(Inspection).where(Inspection.id == inspection_id)
            db_res = await session.execute(stmt)
            insp = db_res.scalar_one_or_none()
            if insp:
                insp.status = InspectionStatus.COMPLETED.value
                insp.video_duration = result.summary.video_duration
                insp.extracted_frames = result.summary.total_frames_extracted
                insp.selected_frames = result.summary.selected_frames
                insp.rejected_frames = result.summary.rejected_frames
                insp.processed_frames = result.summary.processed_frames
                insp.total_detections = result.summary.total_detections
                insp.avg_confidence = result.summary.avg_confidence
                insp.severity_distribution = result.summary.severity_distribution
                insp.config = config.model_dump()
                insp.annotated_video_path = result.annotated_video_path
                insp.report_json_path = str(json_path)
                insp.report_pdf_path = str(pdf_path)
                insp.progress_percent = 100.0
                insp.current_stage = "Completed"
                insp.results = result.model_dump()
                await session.commit()

        _progress_tracker[inspection_id] = InspectionProgress(
            inspection_id=inspection_id,
            status=InspectionStatus.COMPLETED,
            progress_percent=100.0,
            current_stage="Completed",
            processed_frames=result.summary.processed_frames,
            detection_count=result.summary.total_detections,
        )

    except Exception as e:
        logger.error(f"Inspection pipeline failure for {inspection_id}: {e}", exc_info=True)
        async with async_session() as session:
            stmt = select(Inspection).where(Inspection.id == inspection_id)
            db_res = await session.execute(stmt)
            insp = db_res.scalar_one_or_none()
            if insp:
                insp.status = InspectionStatus.FAILED.value
                insp.error_message = str(e)
                insp.current_stage = "Failed"
                await session.commit()

        _progress_tracker[inspection_id] = InspectionProgress(
            inspection_id=inspection_id,
            status=InspectionStatus.FAILED,
            progress_percent=0.0,
            current_stage="Failed",
            error_message=str(e),
        )


@router.post("/{inspection_id}/process", response_model=ProcessResponse)
async def start_processing(
    inspection_id: str,
    config: InspectionConfig = InspectionConfig(),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    session: AsyncSession = Depends(get_session),
):
    """
    Start the pipeline for an uploaded inspection file in the background.
    """
    stmt = select(Inspection).where(Inspection.id == inspection_id)
    res = await session.execute(stmt)
    inspection = res.scalar_one_or_none()

    if not inspection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inspection '{inspection_id}' not found.",
        )

    if inspection.status in [InspectionStatus.EXTRACTING_FRAMES.value, InspectionStatus.DETECTING.value]:
        return ProcessResponse(
            inspection_id=inspection_id,
            message="Inspection is already being processed.",
            status=InspectionStatus(inspection.status),
        )

    inspection.status = InspectionStatus.EXTRACTING_FRAMES.value
    inspection.config = config.model_dump()
    await session.commit()

    _progress_tracker[inspection_id] = InspectionProgress(
        inspection_id=inspection_id,
        status=InspectionStatus.EXTRACTING_FRAMES,
        progress_percent=5.0,
        current_stage="Starting video processing pipeline",
    )

    background_tasks.add_task(
        _execute_inspection_task,
        inspection_id,
        inspection.file_path,
        inspection.file_type,
        config,
    )

    return ProcessResponse(
        inspection_id=inspection_id,
        message="Inspection processing started in background.",
        status=InspectionStatus.EXTRACTING_FRAMES,
    )


@router.get("/{inspection_id}/status", response_model=InspectionProgress)
async def get_inspection_status(
    inspection_id: str,
    session: AsyncSession = Depends(get_session),
):
    """
    Poll processing progress and status.
    """
    if inspection_id in _progress_tracker:
        return _progress_tracker[inspection_id]

    stmt = select(Inspection).where(Inspection.id == inspection_id)
    res = await session.execute(stmt)
    insp = res.scalar_one_or_none()

    if not insp:
        raise HTTPException(status_code=404, detail="Inspection not found.")

    return InspectionProgress(
        inspection_id=insp.id,
        status=InspectionStatus(insp.status) if insp.status in InspectionStatus._value2member_map_ else InspectionStatus.PENDING,
        progress_percent=insp.progress_percent or 0.0,
        current_stage=insp.current_stage or "",
        total_frames=insp.total_video_frames or 0,
        extracted_frames=insp.extracted_frames or 0,
        selected_frames=insp.selected_frames or 0,
        processed_frames=insp.processed_frames or 0,
        detection_count=insp.total_detections or 0,
        error_message=insp.error_message,
    )


@router.get("/{inspection_id}/results", response_model=InspectionResultFull)
async def get_inspection_results(
    inspection_id: str,
    session: AsyncSession = Depends(get_session),
):
    """
    Retrieve full inspection results (detections, severities, frames, metrics).
    """
    stmt = select(Inspection).where(Inspection.id == inspection_id)
    res = await session.execute(stmt)
    insp = res.scalar_one_or_none()

    if not insp:
        raise HTTPException(status_code=404, detail="Inspection not found.")

    if insp.results:
        return insp.results

    # Fallback to results.json on disk if present
    json_path = OUTPUT_DIR / inspection_id / "results.json"
    if json_path.exists():
        with open(json_path, "r") as f:
            data = json.load(f)
            return data

    if insp.status == InspectionStatus.FAILED.value:
        raise HTTPException(
            status_code=400,
            detail=f"Inspection failed: {insp.error_message or 'Unknown error'}",
        )

    raise HTTPException(
        status_code=202,
        detail="Inspection results are not ready yet. Current status: " + insp.status,
    )


@router.get("/{inspection_id}/video")
async def get_annotated_video(
    inspection_id: str,
    session: AsyncSession = Depends(get_session),
):
    """
    Stream or download the annotated inspection video.
    """
    stmt = select(Inspection).where(Inspection.id == inspection_id)
    res = await session.execute(stmt)
    insp = res.scalar_one_or_none()

    if not insp:
        raise HTTPException(status_code=404, detail="Inspection not found.")

    video_path = OUTPUT_DIR / inspection_id / "annotated_video.mp4"
    if not video_path.exists():
        if insp.annotated_video_path and Path(insp.annotated_video_path).exists():
            video_path = Path(insp.annotated_video_path)
        else:
            raise HTTPException(status_code=404, detail="Annotated video not available for this inspection.")

    return FileResponse(
        str(video_path),
        media_type="video/mp4",
        filename=f"annotated_{insp.filename}",
    )


@router.get("/{inspection_id}/frames")
async def list_inspection_frames(
    inspection_id: str,
    session: AsyncSession = Depends(get_session),
):
    """
    Return directory list of extracted, selected, enhanced, and annotated frame URLs.
    """
    output_base = OUTPUT_DIR / inspection_id
    if not output_base.exists():
        raise HTTPException(status_code=404, detail="Inspection directory not found.")

    def get_urls(folder: str):
        p = output_base / folder
        if not p.exists():
            return []
        files = sorted(list(p.glob("*.jpg")) + list(p.glob("*.png")))
        return [f"/api/inspection/{inspection_id}/static/{folder}/{f.name}" for f in files]

    return {
        "inspection_id": inspection_id,
        "extracted": get_urls("extracted"),
        "selected": get_urls("selected"),
        "enhanced": get_urls("enhanced"),
        "annotated": get_urls("annotated"),
    }


@router.get("/{inspection_id}/static/{category}/{filename}")
async def get_frame_image(
    inspection_id: str,
    category: str,
    filename: str,
):
    """
    Serve static frame images (extracted, selected, enhanced, annotated).
    """
    allowed_categories = {"extracted", "selected", "enhanced", "annotated"}
    if category not in allowed_categories:
        raise HTTPException(status_code=400, detail="Invalid frame category.")

    file_path = OUTPUT_DIR / inspection_id / category / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found.")

    return FileResponse(str(file_path), media_type="image/jpeg")


@router.get("/{inspection_id}/report")
async def get_inspection_report(
    inspection_id: str,
    format: str = Query("pdf", pattern="^(pdf|json)$"),
    session: AsyncSession = Depends(get_session),
):
    """
    Download or view the inspection report in PDF or JSON format.
    """
    stmt = select(Inspection).where(Inspection.id == inspection_id)
    res = await session.execute(stmt)
    insp = res.scalar_one_or_none()

    if not insp:
        raise HTTPException(status_code=404, detail="Inspection not found.")

    output_base = OUTPUT_DIR / inspection_id
    if format == "pdf":
        file_path = output_base / "inspection_report.pdf"
        media_type = "application/pdf"
        download_name = f"AquaGuard_Report_{insp.filename.rsplit('.', 1)[0]}.pdf"
    else:
        file_path = output_base / "report.json"
        media_type = "application/json"
        download_name = f"AquaGuard_Report_{insp.filename.rsplit('.', 1)[0]}.json"

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Report in format '{format}' is not available for this inspection.",
        )

    return FileResponse(
        str(file_path),
        media_type=media_type,
        filename=download_name,
    )


@router.get("/history", response_model=list[InspectionSummary])
async def get_inspection_history(
    limit: int = Query(20, ge=1, le=100),
    session: AsyncSession = Depends(get_session),
):
    """
    Fetch history of previous inspections with summary metrics.
    """
    stmt = select(Inspection).order_by(desc(Inspection.created_at)).limit(limit)
    res = await session.execute(stmt)
    records = res.scalars().all()

    summaries = []
    for r in records:
        dist = r.severity_distribution or {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "UNKNOWN": 0}
        method = "clahe"
        if r.config and isinstance(r.config, dict):
            method = r.config.get("enhancement_method", "clahe")

        summaries.append(
            InspectionSummary(
                inspection_id=r.id,
                filename=r.filename,
                created_at=r.created_at.strftime("%Y-%m-%d %H:%M:%S") if r.created_at else "",
                status=InspectionStatus(r.status) if r.status in InspectionStatus._value2member_map_ else InspectionStatus.PENDING,
                video_duration=r.video_duration or 0.0,
                total_frames_extracted=r.extracted_frames or 0,
                selected_frames=r.selected_frames or 0,
                rejected_frames=r.rejected_frames or 0,
                processed_frames=r.processed_frames or 0,
                total_detections=r.total_detections or 0,
                severity_distribution=dist,
                avg_confidence=r.avg_confidence or 0.0,
                enhancement_method=str(method),
            )
        )

    return summaries
