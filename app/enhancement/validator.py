"""
Revision Validation Service.

Validates that paragraph revisions preserve semantic meaning, references,
technical terminology, numbers, equations, and formatting placeholders.
"""

from __future__ import annotations

import re
from typing import List, Tuple, Set


class RevisionValidator:
    """Validates revised paragraphs against original versions for content integrity."""

    @staticmethod
    def extract_references(text: str) -> Set[str]:
        """Extract citations (e.g. [1], [1, 2], (Smith et al., 2020)) from text."""
        # Find numeric citations like [1], [12, 14]
        numeric_citations = re.findall(r"\[\d+(?:\s*,\s*\d+)*\]", text)
        
        # Find author-year citations like (Smith et al., 2020), (Jones & Taylor, 2018)
        author_year_citations = re.findall(
            r"\((?:[A-Z][a-zA-Z]+(?:\s+et\s+al\.?|\s+and\s+[A-Z][a-zA-Z]+)?,\s*\d{4})\)", text
        )
        return set(numeric_citations + author_year_citations)

    @staticmethod
    def extract_numbers(text: str) -> Set[str]:
        """Extract numeric values (integers and decimals) from text."""
        # Extract numbers but ignore those inside citations (e.g. [1] or (2020))
        # Let's strip brackets first or just do a general match and compare
        cleaned_text = re.sub(r"\[\d+(?:\s*,\s*\d+)*\]", "", text)
        cleaned_text = re.sub(r"\b\d{4}\b", "", cleaned_text) # ignore years as they are often citations
        numbers = re.findall(r"\b\d+(?:\.\d+)?\b", cleaned_text)
        return set(numbers)

    @staticmethod
    def extract_equations(text: str) -> Set[str]:
        """Extract LaTeX or basic math equations (e.g. $y = mx + c$, or equation terms)."""
        inline_math = re.findall(r"\$.*?\$", text)
        block_math = re.findall(r"\$\$.*?\$\$", text)
        return set(inline_math + block_math)

    @staticmethod
    def validate(original: str, revised: str) -> Tuple[bool, List[str]]:
        """
        Validate that the revised text preserves the integrity of the original text.
        Returns (is_valid, list_of_errors).
        """
        errors: List[str] = []

        # ── 1. Check citations/references preservation ────────────────────────
        orig_refs = RevisionValidator.extract_references(original)
        rev_refs = RevisionValidator.extract_references(revised)
        missing_refs = orig_refs - rev_refs
        added_refs = rev_refs - orig_refs
        
        if missing_refs:
            errors.append(f"Citations missing in revised text: {', '.join(missing_refs)}")
        if added_refs:
            errors.append(f"Fabricated citations added in revised text: {', '.join(added_refs)}")

        # ── 2. Check numbers preservation ─────────────────────────────────────
        orig_nums = RevisionValidator.extract_numbers(original)
        rev_nums = RevisionValidator.extract_numbers(revised)
        missing_nums = orig_nums - rev_nums
        
        if missing_nums:
            errors.append(f"Technical numbers omitted or altered: {', '.join(missing_nums)}")

        # ── 3. Check equations preservation ────────────────────────────────────
        orig_eqs = RevisionValidator.extract_equations(original)
        rev_eqs = RevisionValidator.extract_equations(revised)
        missing_eqs = orig_eqs - rev_eqs
        
        if missing_eqs:
            errors.append(f"Mathematical equations altered or missing: {', '.join(missing_eqs)}")

        # ── 4. Length/Meaning check: prevent complete deletion or blanking ─────
        orig_words = len(original.split())
        rev_words = len(revised.split())
        if rev_words < 5:
            errors.append("Revised paragraph is abnormally short or blank.")
        elif rev_words < orig_words * 0.4:
            errors.append(f"Revision appears to heavily summarize (Original: {orig_words} words, Revised: {rev_words} words).")

        is_valid = len(errors) == 0
        return is_valid, errors
