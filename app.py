"""
PlagCheck AI — Root entry point shim.

Uvicorn start command (unchanged):
  uvicorn app:app --host 0.0.0.0 --port $PORT

This file is intentionally minimal. All application logic lives in app/main.py.
The shim preserves backward compatibility with Render's start command.
"""

from app.main import app  # noqa: F401 — re-exported for uvicorn

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
