"""
AquaGuard AI — SQLAlchemy database models and session management.
"""

from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, Float, Text, DateTime, JSON, Boolean
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

from backend.app.config import DATABASE_URL


class Base(DeclarativeBase):
    pass


class Inspection(Base):
    __tablename__ = "inspections"

    id = Column(String, primary_key=True)
    filename = Column(String, nullable=False)
    file_type = Column(String, nullable=False)  # "video" or "image"
    file_path = Column(String, nullable=False)
    file_size_mb = Column(Float, default=0.0)
    status = Column(String, default="pending")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                        onupdate=lambda: datetime.now(timezone.utc))

    # Video info
    video_duration = Column(Float, default=0.0)
    video_fps = Column(Float, default=0.0)
    total_video_frames = Column(Integer, default=0)

    # Processing stats
    extracted_frames = Column(Integer, default=0)
    selected_frames = Column(Integer, default=0)
    rejected_frames = Column(Integer, default=0)
    processed_frames = Column(Integer, default=0)
    total_detections = Column(Integer, default=0)
    avg_confidence = Column(Float, default=0.0)

    # Severity distribution as JSON
    severity_distribution = Column(JSON, default=dict)

    # Config as JSON
    config = Column(JSON, default=dict)

    # Paths
    output_dir = Column(String, nullable=True)
    annotated_video_path = Column(String, nullable=True)
    report_json_path = Column(String, nullable=True)
    report_pdf_path = Column(String, nullable=True)

    # Progress
    progress_percent = Column(Float, default=0.0)
    current_stage = Column(String, default="")
    error_message = Column(Text, nullable=True)

    # Results as JSON (frame-level detection results)
    results = Column(JSON, nullable=True)


# ── Engine and session factory ────────────────────────────────────────────

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"timeout": 60.0},
)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db():
    """Create all tables and configure WAL mode for concurrent access."""
    async with engine.begin() as conn:
        from sqlalchemy import text
        await conn.execute(text("PRAGMA journal_mode=WAL;"))
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncSession:
    """Yield an async database session."""
    async with async_session() as session:
        yield session
