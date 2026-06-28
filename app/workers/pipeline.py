"""
RAG Analysis Pipeline.

Full pipeline orchestration for combined plagiarism + AI writing pattern analysis.
Each stage is independent and failure-isolated.

Pipeline stages:
  1. Text extraction
  2. Document chunking
  3. Query engineering (4 variants, 1 Groq call)
  4. Multi-source concurrent retrieval (up to 5 sources × 4 queries = 20 searches)
  5. Embedding + hybrid reranking → top-K candidates
  6. Lexical pre-filtering per chunk (no LLM cost on clearly original chunks)
  7. Groq verification (only for chunks that pass pre-filter, 1 call each)
  8. AI writing pattern analysis (parallel, ThreadPoolExecutor)
  9. Aggregate plagiarism report
 10. Generate PDF
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from app.core.config import settings
from app.core.logging import get_logger
from app.embedding.bge_embedder import BGEEmbedder
from app.llm.groq_client import GroqAPIClient
from app.llm.query_engineer import QueryEngineer
from app.llm.verifier import GroqVerifier
from app.reranking.cross_encoder import CrossEncoderReranker
from app.retrieval.manager import RetrievalManager
from app.schemas.retrieval import CandidatePaper
from app.services.ai_analysis import AIWritingPatternService
from app.services.chunking import DocumentChunker
from app.services.extraction import TextExtractor
from app.services.reporting import PlagiarismReporter

logger = get_logger(__name__)

# ── Shared singletons (created once per process) ──────────────────────────────
_embedder = BGEEmbedder()
_reranker = CrossEncoderReranker()
_retrieval_manager = RetrievalManager(embedder=_embedder, reranker=_reranker)

# ── Stop words for lexical pre-filter (preserved from original) ───────────────
_STOP_WORDS: Set[str] = {
    "the", "and", "that", "for", "with", "this", "from", "are", "was", "were",
    "have", "has", "had", "been", "they", "their", "there", "then", "about",
    "would", "could", "should", "your", "them", "some", "other", "into",
    "than", "its", "also", "these", "those", "such", "only", "over", "more",
}

# ── Progress store (shared with API routes) ───────────────────────────────────
PROGRESS_STORE: Dict[str, Dict[str, Any]] = {}


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


def _content_words(text: str) -> Set[str]:
    words = re.findall(r"\b[a-z0-9]{3,}\b", text.lower())
    return {w for w in words if w not in _STOP_WORDS}


# ── Full combined analysis pipeline ──────────────────────────────────────────

def run_analysis_task(
    task_id: str,
    file_path: str,
    filename: str,
    search_query: Optional[str] = None,
) -> None:
    """
    Full RAG plagiarism + AI writing pattern pipeline.
    Runs synchronously as a FastAPI BackgroundTask.
    Async retrieval is executed via asyncio.run() in a fresh event loop.
    """
    try:
        # ── 1. Extract ────────────────────────────────────────────────────────
        _update(task_id, 5.0, "processing", "Extracting text from document…")
        susp_text = TextExtractor.detect_and_extract(Path(file_path))
        if not susp_text.strip():
            raise ValueError("No extractable text found in uploaded document.")
        susp_word_count = len(susp_text.split())

        # ── 2. Chunk ──────────────────────────────────────────────────────────
        _update(task_id, 10.0, "processing", "Chunking document…")
        chunker = DocumentChunker(
            chunk_size=settings.chunk_size, overlap=settings.chunk_overlap
        )
        susp_chunks = chunker.split(susp_text)
        if not susp_chunks:
            raise ValueError("Document too short to analyse.")

        # ── 3. Query engineering ──────────────────────────────────────────────
        _update(task_id, 15.0, "processing", "Generating search query variants…")
        groq_client = GroqAPIClient()
        engineer = QueryEngineer(groq_client)

        if search_query and search_query.strip():
            # User-supplied query → build a degenerate bundle with all slots the same
            from app.schemas.retrieval import QueryBundle
            query_bundle = QueryBundle(
                keyword_query=search_query.strip(),
                semantic_query=search_query.strip(),
                expanded_query=search_query.strip(),
                academic_query=search_query.strip(),
            )
            logger.info(f"Using user-supplied query: '{search_query}'")
        else:
            query_bundle = engineer.generate(susp_text)

        # ── 4. Multi-source retrieval (async) ─────────────────────────────────
        _update(task_id, 20.0, "processing", "Searching academic literature (multi-source)…")
        loop = asyncio.new_event_loop()
        try:
            retrieval_result = loop.run_until_complete(
                _retrieval_manager.retrieve(
                    query_bundle=query_bundle,
                    chunk_text=susp_text[:1000],
                    papers_per_source=8,
                )
            )
        finally:
            loop.close()

        candidates: List[CandidatePaper] = retrieval_result.candidates
        logger.info(
            f"Retrieved {retrieval_result.total_unique} unique candidates "
            f"from {retrieval_result.sources_used} in "
            f"{retrieval_result.retrieval_time_ms:.0f}ms"
        )

        # Fallback to mock papers if all sources failed
        if not candidates:
            logger.warning("All retrieval sources returned empty. Using Groq fallback.")
            candidates = _retrieval_manager.generate_mock_papers(
                query_bundle.keyword_query, groq_client
            )
        if not candidates:
            raise ValueError("No reference papers could be retrieved for this document.")

        _update(task_id, 38.0, "processing",
                f"Retrieved {len(candidates)} candidates from "
                f"{', '.join(retrieval_result.sources_used)}.")

        # ── 5. Verification per chunk ─────────────────────────────────────────
        _update(task_id, 42.0, "processing", "Starting plagiarism verification…")
        verifier = GroqVerifier(groq_client)
        N = len(susp_chunks)

        # Build candidate word sets for pre-filtering
        cand_word_sets = [_content_words(c.abstract) for c in candidates]

        plag_results: List[Dict[str, Any]] = []
        for idx, chunk in enumerate(susp_chunks):
            pct = 42.0 + (idx / N) * 33.0
            msg = f"Verifying chunk {idx + 1}/{N}…"
            _update(task_id, round(pct, 1), "processing", msg)

            chunk_words = _content_words(chunk["text"])
            max_overlap = max(
                (len(chunk_words & cw) for cw in cand_word_sets), default=0
            )

            if max_overlap < 3:
                # Pre-filter: clearly original — skip LLM call
                logger.info(f"Chunk {idx + 1}: pre-filtered (overlap={max_overlap}), skipping LLM.")
                plag_results.append(_no_match_result(chunk, "Insufficient lexical overlap with any reference."))
                continue

            # Select best-matching candidate by combined_score
            best_candidate = max(
                candidates,
                key=lambda c: c.combined_score + _content_words(c.abstract).__and__(chunk_words).__len__() * 0.05,
            )

            result = verifier.verify(chunk["text"], best_candidate)
            plag_results.append({
                "suspected_chunk_id": chunk["id"],
                "suspected_text": chunk["text"],
                "original_chunk_id": f"cand_{candidates.index(best_candidate)}",
                "original_text": best_candidate.abstract,
                "original_idx": candidates.index(best_candidate),
                "original_title": best_candidate.title,
                "original_authors": ", ".join(best_candidate.authors),
                "original_url": best_candidate.url or "",
                "original_year": best_candidate.year or "N/A",
                # LLM scores
                "semantic_similarity": result.semantic_similarity,
                "exact_copy": result.exact_copy,
                "paraphrase": result.paraphrase,
                "classification": result.classification,
                "confidence": result.confidence,
                "reason": result.reason,
                "sentence_matches": [m.model_dump() for m in result.sentence_matches],
                # Retrieval provenance
                "source": result.source,
                "source_url": result.source_url,
                "citation_count": result.citation_count,
                "embedding_score": result.embedding_score,
                "reranker_score": result.reranker_score,
                "keyword_overlap": result.keyword_overlap,
                "combined_score": result.combined_score,
            })

        # ── 6. AI writing pattern analysis ────────────────────────────────────
        _update(task_id, 77.0, "processing", "Running AI writing pattern analysis…")
        ai_service = AIWritingPatternService(groq_client, max_workers=3)

        def _ai_progress(current: int, total: int, message: str) -> None:
            pct = 77.0 + (current / total) * 15.0
            _update(task_id, round(pct, 1), "processing", message)

        ai_report = ai_service.analyze_document(susp_chunks, progress_callback=_ai_progress)

        # ── 7. Aggregate ──────────────────────────────────────────────────────
        _update(task_id, 93.0, "processing", "Aggregating results…")
        orig_meta = {
            "query": query_bundle.keyword_query,
            "paper_count": len(candidates),
            "filename": "Multi-source Academic References",
            "word_count": sum(len(c.abstract.split()) for c in candidates),
            "chunk_count": len(candidates),
            "sources": retrieval_result.sources_used,
            "retrieval_time_ms": retrieval_result.retrieval_time_ms,
        }
        susp_meta = {
            "filename": filename,
            "word_count": susp_word_count,
            "chunk_count": N,
        }
        plag_report = PlagiarismReporter.aggregate_results(
            results=plag_results, orig_meta=orig_meta, susp_meta=susp_meta
        )
        combined_report = {**plag_report, "ai_analysis": ai_report}

        # ── 8. Persist + PDF ──────────────────────────────────────────────────
        report_id = str(uuid.uuid4())
        json_path = settings.report_path / f"{report_id}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(combined_report, f, indent=2, ensure_ascii=False)

        _update(task_id, 96.0, "processing", "Generating PDF report…")
        pdf_path = settings.report_path / f"{report_id}.pdf"
        PlagiarismReporter.generate_combined_pdf_report(combined_report, pdf_path)

        _update(task_id, 100.0, "completed", "Analysis completed successfully!", report_id)
        logger.info(f"Task {task_id} completed — report_id={report_id}")

    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}", exc_info=True)
        PROGRESS_STORE.setdefault(task_id, {}).update({
            "status": "failed",
            "progress": 100.0,
            "message": f"Analysis failed: {e}",
            "error": str(e),
        })


# ── Standalone AI-only pipeline ───────────────────────────────────────────────

def run_ai_only_task(task_id: str, file_path: str, filename: str) -> None:
    """Standalone AI writing pattern analysis — no plagiarism check."""
    try:
        _update(task_id, 5.0, "processing", "Extracting text…")
        text = TextExtractor.detect_and_extract(Path(file_path))
        if not text.strip():
            raise ValueError("No extractable text found.")

        _update(task_id, 15.0, "processing", "Chunking document…")
        chunker = DocumentChunker(chunk_size=300, overlap=50)
        chunks = chunker.split(text)
        if not chunks:
            raise ValueError("Document too short to analyse.")

        groq_client = GroqAPIClient()
        ai_service = AIWritingPatternService(groq_client, max_workers=3)

        def _progress(current: int, total: int, message: str) -> None:
            pct = 20.0 + (current / total) * 70.0
            _update(task_id, round(pct, 1), "processing", message)

        ai_report = ai_service.analyze_document(chunks, progress_callback=_progress)
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
        logger.info(f"AI-only task {task_id} complete — report_id={report_id}")

    except Exception as e:
        logger.error(f"AI-only task {task_id} failed: {e}", exc_info=True)
        PROGRESS_STORE.setdefault(task_id, {}).update({
            "status": "failed",
            "progress": 100.0,
            "message": f"AI analysis failed: {e}",
            "error": str(e),
        })


# ── Helper ────────────────────────────────────────────────────────────────────

def _no_match_result(chunk: Dict[str, Any], reason: str) -> Dict[str, Any]:
    return {
        "suspected_chunk_id": chunk["id"],
        "suspected_text": chunk["text"],
        "original_chunk_id": "N/A",
        "original_text": "N/A",
        "original_idx": -1,
        "original_title": "N/A",
        "original_authors": "N/A",
        "original_url": "",
        "original_year": "N/A",
        "semantic_similarity": 0,
        "exact_copy": 0,
        "paraphrase": 0,
        "classification": "Original",
        "confidence": 100,
        "reason": reason,
        "sentence_matches": [],
        "source": "",
        "source_url": "",
        "citation_count": 0,
        "embedding_score": 0.0,
        "reranker_score": 0.0,
        "keyword_overlap": 0.0,
        "combined_score": 0.0,
    }
