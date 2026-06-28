"""
Routes module for PlagCheck AI.
Handles file upload, combined plagiarism + AI writing pattern analysis, and report serving.
"""

import json
import uuid
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import APIRouter, File, UploadFile, BackgroundTasks, HTTPException, Query, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel

from config import settings
from utils import logger
from extractor import TextExtractor
from chunker import DocumentChunker
from groq_client import GroqPlagiarismClient
from comparator import PlagiarismComparator
from report import PlagiarismReporter
from semanticscholar import SemanticScholarClient
from ai_analyzer import AIWritingPatternService

router = APIRouter()

# ──────────────────────────────────────────────────────────────────────────────
# In-memory progress store  {task_id: {...}}
# ──────────────────────────────────────────────────────────────────────────────
PROGRESS_STORE: Dict[str, Dict[str, Any]] = {}


# ──────────────────────────────────────────────────────────────────────────────
# Request models
# ──────────────────────────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    file_path: str
    filename: str
    search_query: Optional[str] = None


class AIAnalyzeRequest(BaseModel):
    """Standalone AI-only analysis (no plagiarism check)."""
    file_path: str
    filename: str


# ──────────────────────────────────────────────────────────────────────────────
# Background worker — combined plagiarism + AI pattern analysis
# ──────────────────────────────────────────────────────────────────────────────
def run_analysis_task(
    task_id: str,
    file_path: str,
    filename: str,
    search_query: Optional[str] = None,
) -> None:
    """
    Full pipeline:
      1. Extract text
      2. Determine search query (auto or manual)
      3. Query Semantic Scholar / Groq fallback
      4. Chunk document
      5. Plagiarism comparison against papers
      6. AI writing pattern analysis (concurrent)
      7. Aggregate & persist JSON report
      8. Generate combined PDF
    """
    try:
        _update(task_id, 5.0, "processing", "Extracting text from uploaded document…")

        susp_file = Path(file_path)
        susp_text = TextExtractor.detect_and_extract(susp_file)
        susp_word_count = len(susp_text.split())

        if not susp_text.strip():
            raise ValueError("The uploaded document yielded no extractable text.")

        groq_client = GroqPlagiarismClient()

        # ── Step 2: Search query ────────────────────────────────────────────
        _update(task_id, 12.0, "processing", "Determining search keywords…")
        query = (search_query or "").strip()
        if not query:
            query = groq_client.generate_search_query(susp_text)
            logger.info(f"Task {task_id}: Auto-generated query: '{query}'")

        # ── Step 3: Literature search ───────────────────────────────────────
        _update(task_id, 18.0, "processing", f"Searching literature databases for '{query}'…")
        scholar_client = SemanticScholarClient(groq_client)
        papers = scholar_client.search_papers(query, limit=5)
        if not papers:
            raise ValueError("No reference literature could be retrieved for the query.")

        # ── Step 4: Chunk document ──────────────────────────────────────────
        _update(task_id, 27.0, "processing", "Chunking document…")
        chunker = DocumentChunker(chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)
        susp_chunks = chunker.split_into_chunks(susp_text)
        if not susp_chunks:
            raise ValueError("Could not build chunks from the document. Document may be too short.")

        orig_meta = {
            "query": query,
            "paper_count": len(papers),
            "filename": "Semantic Scholar References",
            "word_count": sum(len(p.get("abstract", "").split()) for p in papers),
            "chunk_count": len(papers),
        }
        susp_meta = {
            "filename": filename,
            "word_count": susp_word_count,
            "chunk_count": len(susp_chunks),
        }

        # ── Step 5: Plagiarism comparison ───────────────────────────────────
        _update(task_id, 30.0, "processing", "Starting plagiarism comparison…")
        comparator = PlagiarismComparator(groq_client)
        N_chunks = len(susp_chunks)

        def plag_progress(current: int, total: int, message: str) -> None:
            # Scale 30 → 60
            pct = 30.0 + (current / total) * 30.0
            _update(task_id, round(pct, 1), "processing", message)

        plag_results = comparator.compare_against_papers(
            suspected_chunks=susp_chunks,
            papers=papers,
            progress_callback=plag_progress,
        )

        # ── Step 6: AI writing pattern analysis ────────────────────────────
        _update(task_id, 62.0, "processing", "Running AI Writing Pattern Analysis…")
        ai_service = AIWritingPatternService(groq_client, max_workers=3)

        def ai_progress(current: int, total: int, message: str) -> None:
            # Scale 62 → 90
            pct = 62.0 + (current / total) * 28.0
            _update(task_id, round(pct, 1), "processing", message)

        ai_report = ai_service.analyze_document(susp_chunks, progress_callback=ai_progress)

        # ── Step 7: Aggregate plagiarism report ─────────────────────────────
        _update(task_id, 91.0, "processing", "Aggregating results…")
        plag_report = PlagiarismReporter.aggregate_results(
            results=plag_results,
            orig_meta=orig_meta,
            susp_meta=susp_meta,
        )

        # Merge AI report into combined structure
        combined_report = {
            **plag_report,
            "ai_analysis": ai_report,
        }

        # ── Step 8: Persist JSON ────────────────────────────────────────────
        report_id = str(uuid.uuid4())
        report_json_path = settings.report_path / f"{report_id}.json"
        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(combined_report, f, indent=2, ensure_ascii=False)

        # ── Step 9: Generate PDF ────────────────────────────────────────────
        _update(task_id, 95.0, "processing", "Generating combined PDF report…")
        pdf_path = settings.report_path / f"{report_id}.pdf"
        PlagiarismReporter.generate_combined_pdf_report(combined_report, pdf_path)

        _update(task_id, 100.0, "completed", "Analysis completed successfully!", report_id)
        logger.info(f"Task {task_id} completed. Report ID: {report_id}")

    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}", exc_info=True)
        PROGRESS_STORE[task_id].update({
            "status": "failed",
            "progress": 100.0,
            "message": f"Analysis failed: {e}",
            "error": str(e),
        })


def run_ai_only_task(task_id: str, file_path: str, filename: str) -> None:
    """Standalone AI writing pattern analysis (no plagiarism check)."""
    try:
        _update(task_id, 5.0, "processing", "Extracting text…")
        text = TextExtractor.detect_and_extract(Path(file_path))
        if not text.strip():
            raise ValueError("No extractable text found.")

        _update(task_id, 15.0, "processing", "Chunking document…")
        chunker = DocumentChunker(chunk_size=300, overlap=50)
        chunks = chunker.split_into_chunks(text)
        if not chunks:
            raise ValueError("Document is too short to analyse.")

        groq_client = GroqPlagiarismClient()
        ai_service = AIWritingPatternService(groq_client, max_workers=3)

        def ai_progress(current: int, total: int, message: str) -> None:
            pct = 20.0 + (current / total) * 70.0
            _update(task_id, round(pct, 1), "processing", message)

        ai_report = ai_service.analyze_document(chunks, progress_callback=ai_progress)
        ai_report["metadata"] = {
            "filename": filename,
            "word_count": len(text.split()),
            "chunk_count": len(chunks),
        }

        _update(task_id, 93.0, "processing", "Saving AI report…")
        report_id = str(uuid.uuid4())
        json_path = settings.report_path / f"ai_{report_id}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(ai_report, f, indent=2, ensure_ascii=False)

        _update(task_id, 100.0, "completed", "AI analysis complete!", report_id)

    except Exception as e:
        logger.error(f"AI-only task {task_id} failed: {e}", exc_info=True)
        PROGRESS_STORE[task_id].update({
            "status": "failed",
            "progress": 100.0,
            "message": f"AI analysis failed: {e}",
            "error": str(e),
        })


# ──────────────────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────────────────
def _update(
    task_id: str,
    progress: float,
    status: str,
    message: str,
    report_id: Optional[str] = None,
) -> None:
    entry = PROGRESS_STORE.setdefault(task_id, {"error": None, "report_id": None})
    entry.update({"status": status, "progress": progress, "message": message})
    if report_id is not None:
        entry["report_id"] = report_id


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/upload")
def upload_file(
    file: UploadFile = File(...),
    search_query: Optional[str] = Form(None),
):
    """Upload a single document (PDF/DOCX ≤ 50 MB)."""
    MAX_SIZE = 50 * 1024 * 1024
    ALLOWED_EXTS = {".pdf", ".docx"}

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(400, "Only PDF and DOCX files are supported.")

    file.file.seek(0, 2)
    size = file.file.tell()
    file.file.seek(0)
    if size > MAX_SIZE:
        raise HTTPException(400, f"File exceeds 50 MB limit.")

    session_dir = settings.upload_path / str(uuid.uuid4())
    session_dir.mkdir(parents=True, exist_ok=True)
    file_path = session_dir / f"uploaded_{file.filename}"

    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        logger.error(f"Error saving upload: {e}")
        raise HTTPException(500, "Failed to save uploaded document.")

    return {
        "file_path": str(file_path),
        "filename": file.filename,
        "search_query": search_query or "",
    }


@router.post("/analyze")
def analyze_documents(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    """
    Start combined plagiarism + AI writing pattern analysis in background.
    Returns task_id for progress polling.
    """
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
def analyze_ai_only(request: AIAnalyzeRequest, background_tasks: BackgroundTasks):
    """
    Start a standalone AI writing pattern analysis (no plagiarism check).
    Returns task_id for progress polling.
    """
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
def get_analysis_status(task_id: str):
    """Poll progress of any analysis task."""
    if task_id not in PROGRESS_STORE:
        raise HTTPException(404, "Task not found.")
    return PROGRESS_STORE[task_id]


@router.get("/report")
def get_report(report_id: str = Query(...)):
    """Return combined JSON report."""
    # Support both regular and AI-only prefix
    for prefix in ("", "ai_"):
        path = settings.report_path / f"{prefix}{report_id}.json"
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                raise HTTPException(500, f"Failed to load report: {e}")
    raise HTTPException(404, "Report not found.")


@router.get("/ai-report")
def get_ai_report(report_id: str = Query(...)):
    """Return the ai_analysis section of a combined report, or a standalone AI report."""
    # Try combined report first
    combined_path = settings.report_path / f"{report_id}.json"
    if combined_path.exists():
        with open(combined_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("ai_analysis", {})

    # Try standalone AI report
    ai_path = settings.report_path / f"ai_{report_id}.json"
    if ai_path.exists():
        with open(ai_path, "r", encoding="utf-8") as f:
            return json.load(f)

    raise HTTPException(404, "AI report not found.")


@router.get("/report/pdf")
def get_report_pdf(report_id: str = Query(...)):
    """Serve the combined PDF report inline."""
    pdf_path = settings.report_path / f"{report_id}.pdf"

    if not pdf_path.exists():
        json_path = settings.report_path / f"{report_id}.json"
        if not json_path.exists():
            raise HTTPException(404, "Report data not found.")
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                report_data = json.load(f)
            PlagiarismReporter.generate_combined_pdf_report(report_data, pdf_path)
        except Exception as e:
            logger.error(f"PDF re-generation failed: {e}")
            raise HTTPException(500, "Failed to generate PDF report.")

    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=f"plagcheck_report_{report_id[:8]}.pdf",
    )


@router.get("/download")
def download_report(report_id: str = Query(...)):
    """Trigger download of the combined PDF report."""
    pdf_path = settings.report_path / f"{report_id}.pdf"
    if not pdf_path.exists():
        get_report_pdf(report_id)

    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=f"plagcheck_report_{report_id[:8]}.pdf",
        headers={"Content-Disposition": f"attachment; filename=plagcheck_report_{report_id[:8]}.pdf"},
    )


@router.get("/download-ai-report")
def download_ai_report(report_id: str = Query(...)):
    """Trigger download of the standalone AI analysis PDF (generates from JSON)."""
    pdf_path = settings.report_path / f"ai_{report_id}.pdf"

    if not pdf_path.exists():
        json_path = settings.report_path / f"ai_{report_id}.json"
        if not json_path.exists():
            # Fallback: try the combined report's AI section
            combined = settings.report_path / f"{report_id}.json"
            if combined.exists():
                with open(combined, "r", encoding="utf-8") as f:
                    data = json.load(f)
                ai_data = data.get("ai_analysis", {})
            else:
                raise HTTPException(404, "AI report not found.")
        else:
            with open(json_path, "r", encoding="utf-8") as f:
                ai_data = json.load(f)

        try:
            PlagiarismReporter.generate_ai_pdf_report(ai_data, pdf_path)
        except Exception as e:
            raise HTTPException(500, f"Failed to generate AI PDF: {e}")

    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=f"ai_analysis_{report_id[:8]}.pdf",
        headers={"Content-Disposition": f"attachment; filename=ai_analysis_{report_id[:8]}.pdf"},
    )
