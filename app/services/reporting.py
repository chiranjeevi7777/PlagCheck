"""
Reporting service — thin re-export of the root report.py module.

The ReportLab PDF compiler lives in the project-root report.py to avoid
duplicating ~40 KB of generation code. All application code should import
PlagiarismReporter from here rather than directly from the root module.
"""

from __future__ import annotations

# Root-level report.py is always importable when running from project root
from report import PlagiarismReporter  # noqa: F401 – re-exported

__all__ = ["PlagiarismReporter"]
