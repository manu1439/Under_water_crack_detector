"""
AquaGuard AI — Report API Endpoints.
Serves generated JSON and PDF inspection reports.
"""

from __future__ import annotations

import logging
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.config import OUTPUT_DIR
from backend.app.database.models import Inspection, get_session

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/reports", tags=["Reports"])


@router.get("/{inspection_id}/download")
async def download_report(
    inspection_id: str,
    format: str = Query("pdf", pattern="^(pdf|json)$"),
    session: AsyncSession = Depends(get_session),
):
    """
    Download the inspection report in PDF or JSON format.
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
