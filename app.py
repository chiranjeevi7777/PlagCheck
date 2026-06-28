import os
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from routes import router
from utils import logger

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm or download NLTK tokenizers on start
    from utils import init_nltk
    init_nltk()
    yield

app = FastAPI(
    title="PlagCheck AI",
    description="Groq-powered Plagiarism Detection System",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure folders exist
settings.upload_path
settings.report_path

# Mount static files and templates
BASE_DIR = Path(__file__).resolve().parent

# Make sure directories exist
(BASE_DIR / "static" / "css").mkdir(parents=True, exist_ok=True)
(BASE_DIR / "static" / "js").mkdir(parents=True, exist_ok=True)
(BASE_DIR / "templates").mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Include standard routes
app.include_router(router)

@app.get("/", response_class=HTMLResponse)
def get_home_page(request: Request):
    """Serves the main application SPA dashboard."""
    return templates.TemplateResponse(request, "index.html")

# Global error handler for generic exceptions
@app.exception_handler(Exception)
def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal Server Error: {str(exc)}"}
    )

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting PlagCheck server on http://localhost:8000")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
