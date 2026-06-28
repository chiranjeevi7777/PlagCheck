"""
Document Enhancement Package.

Contains services for paragraph classification, revision planning,
validation, rewriting, metrics analysis, diff highlight, and export writing.
"""

from __future__ import annotations

from app.enhancement.metrics import DocumentMetricsCalculator
from app.enhancement.classifier import ParagraphClassifier
from app.enhancement.planner import RevisionPlanner
from app.enhancement.validator import RevisionValidator
from app.enhancement.rewriter import ParagraphRewriter
from app.enhancement.diff_engine import DifferenceEngine
from app.enhancement.document_writer import DocumentWriter
from app.enhancement.comparison import DocumentVersionManager, DocumentVersion, VersionHistoryStore
from app.enhancement.report import EnhancementReportCompiler

__all__ = [
    "DocumentMetricsCalculator",
    "ParagraphClassifier",
    "RevisionPlanner",
    "RevisionValidator",
    "ParagraphRewriter",
    "DifferenceEngine",
    "DocumentWriter",
    "DocumentVersionManager",
    "DocumentVersion",
    "VersionHistoryStore",
    "EnhancementReportCompiler",
]
