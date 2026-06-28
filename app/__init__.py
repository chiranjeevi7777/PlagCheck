"""
PlagCheck AI — Retrieval-Augmented Plagiarism Detection Platform.

This __init__.py exposes the FastAPI `app` instance at the package level
so that `uvicorn app:app` resolves correctly when the `app/` package
shadows the root `app.py` shim.
"""

from app.main import app  # noqa: F401 — uvicorn entry point

__all__ = ["app"]
