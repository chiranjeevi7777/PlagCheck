"""
FastAPI API routes for PlagCheck AI.

All existing routes are preserved with identical paths, request/response shapes,
and behaviour. Only the background workers and imports are updated to use the
new RAG pipeline.

Routes:
  POST /upload                   — Upload PDF/DOCX (≤50 MB)
  POST /analyze                  — Start combined plagiarism + AI analysis
  POST /analyze-ai               — Start standalone AI pattern analysis
  GET  /analyze/status/{task_id} — Poll task progress
  GET  /report                   — Fetch JSON report
  GET  /ai-report                — Fetch AI analysis section
  GET  /report/pdf               — Serve combined PDF inline
  GET  /download                 — Download combined PDF
  GET  /download-ai-report       — Download standalone AI PDF
"""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.core.config import settings
from app.core.logging import get_logger
from app.services.reporting import PlagiarismReporter
from app.workers.pipeline import PROGRESS_STORE, run_ai_only_task, run_analysis_task

logger = get_logger(__name__)
router = APIRouter()

_MAX_SIZE = 50 * 1024 * 1024
_ALLOWED_EXTS = {".pdf", ".docx"}


# ── Request schemas ───────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    file_path: str
    filename: str
    search_query: Optional[str] = None


class AIAnalyzeRequest(BaseModel):
    file_path: str
    filename: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/upload")
def upload_file(
    file: UploadFile = File(...),
    search_query: Optional[str] = Form(None),
) -> Dict[str, Any]:
    """Upload a single PDF or DOCX document (≤ 50 MB)."""
    ext = Path(file.filename).suffix.lower()
    if ext not in _ALLOWED_EXTS:
        raise HTTPException(400, "Only PDF and DOCX files are supported.")

    file.file.seek(0, 2)
    size = file.file.tell()
    file.file.seek(0)
    if size > _MAX_SIZE:
        raise HTTPException(400, "File exceeds the 50 MB limit.")

    session_dir = settings.upload_path / str(uuid.uuid4())
    session_dir.mkdir(parents=True, exist_ok=True)
    dest = session_dir / f"uploaded_{file.filename}"

    try:
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        logger.error(f"Error saving upload: {e}")
        raise HTTPException(500, "Failed to save uploaded document.")

    return {
        "file_path": str(dest),
        "filename": file.filename,
        "search_query": search_query or "",
    }


@router.post("/analyze")
def analyze_documents(
    request: AnalyzeRequest, background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """Start combined plagiarism + AI writing pattern analysis (background)."""
    task_id = str(uuid.uuid4())
    PROGRESS_STORE[task_id] = {
        "status": "queued",
        "progress": 0.0,
        "message": "Analysis queued…",
        "report_id": None,
        "error": None,
    }
    background_tasks.add_task(
        run_analysis_task,
        task_id=task_id,
        file_path=request.file_path,
        filename=request.filename,
        search_query=request.search_query,
    )
    return {"task_id": task_id, "status": "queued"}


@router.post("/analyze-ai")
def analyze_ai_only(
    request: AIAnalyzeRequest, background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """Start standalone AI writing pattern analysis (background)."""
    task_id = str(uuid.uuid4())
    PROGRESS_STORE[task_id] = {
        "status": "queued",
        "progress": 0.0,
        "message": "AI analysis queued…",
        "report_id": None,
        "error": None,
    }
    background_tasks.add_task(
        run_ai_only_task,
        task_id=task_id,
        file_path=request.file_path,
        filename=request.filename,
    )
    return {"task_id": task_id, "status": "queued"}


@router.get("/analyze/status/{task_id}")
def get_analysis_status(task_id: str) -> Dict[str, Any]:
    """Poll the progress of any analysis task."""
    if task_id not in PROGRESS_STORE:
        raise HTTPException(404, "Task not found.")
    return PROGRESS_STORE[task_id]


@router.get("/report")
def get_report(report_id: str = Query(...)) -> Any:
    """Return combined JSON report (plagiarism + AI analysis)."""
    for prefix in ("", "ai_"):
        path = settings.report_path / f"{prefix}{report_id}.json"
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                raise HTTPException(500, f"Failed to load report: {e}") from e
    raise HTTPException(404, "Report not found.")


@router.get("/ai-report")
def get_ai_report(report_id: str = Query(...)) -> Any:
    """Return the ai_analysis section of a combined report, or a standalone AI report."""
    combined = settings.report_path / f"{report_id}.json"
    if combined.exists():
        with open(combined, encoding="utf-8") as f:
            return json.load(f).get("ai_analysis", {})
    ai_only = settings.report_path / f"ai_{report_id}.json"
    if ai_only.exists():
        with open(ai_only, encoding="utf-8") as f:
            return json.load(f)
    raise HTTPException(404, "AI report not found.")


@router.get("/report/pdf")
def get_report_pdf(report_id: str = Query(...)) -> FileResponse:
    """Serve the combined PDF report inline."""
    pdf_path = settings.report_path / f"{report_id}.pdf"
    if not pdf_path.exists():
        _regenerate_pdf(report_id, pdf_path)
    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=f"plagcheck_report_{report_id[:8]}.pdf",
    )


@router.get("/download")
def download_report(report_id: str = Query(...)) -> FileResponse:
    """Trigger download of the combined PDF report."""
    pdf_path = settings.report_path / f"{report_id}.pdf"
    if not pdf_path.exists():
        _regenerate_pdf(report_id, pdf_path)
    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=f"plagcheck_report_{report_id[:8]}.pdf",
        headers={"Content-Disposition": f"attachment; filename=plagcheck_report_{report_id[:8]}.pdf"},
    )


@router.get("/download-ai-report")
def download_ai_report(report_id: str = Query(...)) -> FileResponse:
    """Download standalone AI analysis PDF (regenerated from JSON if needed)."""
    pdf_path = settings.report_path / f"ai_{report_id}.pdf"
    if not pdf_path.exists():
        ai_data = _load_ai_data(report_id)
        try:
            PlagiarismReporter.generate_ai_pdf_report(ai_data, pdf_path)
        except Exception as e:
            raise HTTPException(500, f"Failed to generate AI PDF: {e}") from e
    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=f"ai_analysis_{report_id[:8]}.pdf",
        headers={"Content-Disposition": f"attachment; filename=ai_analysis_{report_id[:8]}.pdf"},
    )


# ── Private helpers ───────────────────────────────────────────────────────────

def _regenerate_pdf(report_id: str, pdf_path: Path) -> None:
    json_path = settings.report_path / f"{report_id}.json"
    if not json_path.exists():
        raise HTTPException(404, "Report data not found.")
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        PlagiarismReporter.generate_combined_pdf_report(data, pdf_path)
    except Exception as e:
        logger.error(f"PDF regeneration failed: {e}")
        raise HTTPException(500, "Failed to generate PDF report.") from e


def _load_ai_data(report_id: str) -> Dict[str, Any]:
    ai_path = settings.report_path / f"ai_{report_id}.json"
    if ai_path.exists():
        with open(ai_path, encoding="utf-8") as f:
            return json.load(f)
    combined = settings.report_path / f"{report_id}.json"
    if combined.exists():
        with open(combined, encoding="utf-8") as f:
            return json.load(f).get("ai_analysis", {})
    raise HTTPException(404, "AI report not found.")
