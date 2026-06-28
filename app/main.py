"""
PlagCheck AI — FastAPI application entry point.

This module creates and configures the FastAPI app instance.
It is imported by the root app.py shim so uvicorn can start with:
  uvicorn app:app --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.routes import router
from app.api.enhancement_routes import router as enhancement_router
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_BASE_DIR = Path(__file__).resolve().parent.parent  # project root


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Pre-warm NLTK tokenisers on server startup."""
    import nltk
    for resource in ("tokenizers/punkt", "tokenizers/punkt_tab"):
        try:
            nltk.data.find(resource)
        except LookupError:
            try:
                nltk.download(resource.split("/")[-1], quiet=True)
                logger.info(f"NLTK '{resource}' downloaded.")
            except Exception as e:
                logger.warning(f"NLTK download failed for '{resource}': {e}")
    logger.info("PlagCheck AI startup complete.")
    yield
    logger.info("PlagCheck AI shutting down.")


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="PlagCheck AI",
        description=(
            "Retrieval-Augmented Plagiarism Detection Platform — "
            "Multi-source retrieval, embedding reranking, and Groq verification."
        ),
        version="2.0.0",
        lifespan=_lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Ensure storage dirs exist
    settings.upload_path  # noqa: B018 — property creates dir
    settings.report_path  # noqa: B018

    # Static files + templates
    static_dir = _BASE_DIR / "static"
    templates_dir = _BASE_DIR / "templates"
    (static_dir / "css").mkdir(parents=True, exist_ok=True)
    (static_dir / "js").mkdir(parents=True, exist_ok=True)
    templates_dir.mkdir(parents=True, exist_ok=True)

    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    templates = Jinja2Templates(directory=str(templates_dir))

    # Include API router
    app.include_router(router)
    app.include_router(enhancement_router)

    # SPA home page
    @app.get("/", response_class=HTMLResponse)
    def home(request: Request) -> HTMLResponse:
        """Serve the main SPA dashboard."""
        return templates.TemplateResponse(request, "index.html")

    @app.head("/")
    def home_head() -> HTMLResponse:
        """Endpoint for health check probes (prevents 405 Method Not Allowed)."""
        return HTMLResponse(content="", status_code=200)

    # Global exception handler
    @app.exception_handler(Exception)
    def global_error(request: Request, exc: Exception) -> JSONResponse:
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal Server Error: {exc}"},
        )

    return app


app = create_app()
