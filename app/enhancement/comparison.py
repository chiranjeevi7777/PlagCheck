"""
Version Control & Comparison Service.

Tracks multiple versions of enhanced documents (Version 1, Version 2, etc.),
allowing users to compare changes, restore previous versions, and download historical versions.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class DocumentVersion(BaseModel):
    """Pydantic model representing a specific document version's metadata."""
    version_num: int = Field(..., description="Version number (e.g. 1, 2, 3)")
    timestamp: str = Field(..., description="ISO formatting timestamp")
    file_path: str = Field(..., description="Path to the document file at this version")
    paragraphs: List[str] = Field(default_factory=list, description="All paragraphs text of this version")
    replacements: Dict[str, str] = Field(default_factory=dict, description="Replacements applied to get to this version")
    report_id: Optional[str] = Field(None, description="Associated plagiarism/AI analysis report ID")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Writing analytics metrics for this version")


class VersionHistoryStore(BaseModel):
    """Pydantic model holding full revision history for a document."""
    document_id: str = Field(..., description="Unique document session ID")
    original_filename: str = Field(..., description="Original filename uploaded")
    original_file_path: str = Field(..., description="Original file path")
    current_version: int = Field(1, description="Current active version number")
    history: Dict[str, DocumentVersion] = Field(default_factory=dict, description="Mapping of version_num strings to DocumentVersion")


class DocumentVersionManager:
    """Manages document versions, persistence, and state transitions."""

    def __init__(self, storage_dir: Optional[Path] = None) -> None:
        self.storage_dir = storage_dir or (settings.report_path / "enhancements")
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _get_history_path(self, document_id: str) -> Path:
        return self.storage_dir / f"{document_id}_history.json"

    def get_history(self, document_id: str) -> Optional[VersionHistoryStore]:
        """Load document version history from file store."""
        path = self._get_history_path(document_id)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return VersionHistoryStore.model_validate(data)
        except Exception as e:
            logger.error(f"Failed to load version history for {document_id}: {e}")
            return None

    def save_history(self, store: VersionHistoryStore) -> None:
        """Persist document version history to file store."""
        path = self._get_history_path(store.document_id)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(store.model_dump_json(indent=2))
        except Exception as e:
            logger.error(f"Failed to save version history for {store.document_id}: {e}")

    def create_initial_version(
        self, document_id: str, filename: str, file_path: Path, paragraphs: List[str], metrics: Dict[str, Any]
    ) -> VersionHistoryStore:
        """Create version 1 (the original document representation)."""
        logger.info(f"Creating initial version for document: {document_id}")
        
        # Save a copy of the original file into the version folder
        version_file_path = self.storage_dir / f"{document_id}_v1{file_path.suffix}"
        import shutil
        shutil.copy(str(file_path), str(version_file_path))

        v1 = DocumentVersion(
            version_num=1,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            file_path=str(version_file_path),
            paragraphs=paragraphs,
            replacements={},
            metrics=metrics
        )

        store = VersionHistoryStore(
            document_id=document_id,
            original_filename=filename,
            original_file_path=str(file_path),
            current_version=1,
            history={"1": v1}
        )
        self.save_history(store)
        return store

    def add_version(
        self,
        document_id: str,
        file_path: Path,
        paragraphs: List[str],
        replacements: Dict[str, str],
        metrics: Dict[str, Any],
        report_id: Optional[str] = None
    ) -> DocumentVersion:
        """Create a new version (e.g. Version 2, 3) in the document's history."""
        store = self.get_history(document_id)
        if not store:
            raise ValueError(f"History not found for document ID: {document_id}")

        new_version_num = store.current_version + 1
        
        # Save file copy
        version_file_path = self.storage_dir / f"{document_id}_v{new_version_num}{file_path.suffix}"
        import shutil
        shutil.copy(str(file_path), str(version_file_path))

        new_version = DocumentVersion(
            version_num=new_version_num,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            file_path=str(version_file_path),
            paragraphs=paragraphs,
            replacements=replacements,
            metrics=metrics,
            report_id=report_id
        )

        store.history[str(new_version_num)] = new_version
        store.current_version = new_version_num
        self.save_history(store)
        
        logger.info(f"Added version {new_version_num} for document: {document_id}")
        return new_version

    def restore_version(self, document_id: str, version_num: int) -> DocumentVersion:
        """Restore document state to a previous version number."""
        store = self.get_history(document_id)
        if not store:
            raise ValueError(f"History not found for document ID: {document_id}")

        v_str = str(version_num)
        if v_str not in store.history:
            raise ValueError(f"Version {version_num} not found in history.")

        target_version = store.history[v_str]
        
        # Copy the target version file to be the current active version file
        active_path = Path(store.original_file_path)
        import shutil
        shutil.copy(target_version.file_path, str(active_path))

        # We increase the version number of the current state by adding a duplicate version
        return self.add_version(
            document_id=document_id,
            file_path=active_path,
            paragraphs=target_version.paragraphs,
            replacements=target_version.replacements,
            metrics=target_version.metrics,
            report_id=target_version.report_id
        )
