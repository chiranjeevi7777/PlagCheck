"""
Difference Engine.

Generates word-level, sentence-level, and paragraph-level diff structures
with HTML-highlighted visualizations for comparison dashboards.
"""

from __future__ import annotations

import difflib
import re
from typing import Any, Dict, List


class DifferenceEngine:
    """Computes exact modifications between original and revised document segments."""

    @staticmethod
    def get_word_diff_html(original: str, revised: str) -> str:
        """
        Generate inline HTML highlighted text showing added/deleted words.
        Deleted text wrapped in <del class="diff-del">...</del>
        Added text wrapped in <ins class="diff-add">...</ins>
        """
        if not original and not revised:
            return ""
        if not original:
            return f'<ins class="diff-add">{revised}</ins>'
        if not revised:
            return f'<del class="diff-del">{original}</del>'

        # Split text into words/tokens including punctuation
        original_tokens = re.findall(r"\w+|\s+|[^\w\s]", original)
        revised_tokens = re.findall(r"\w+|\s+|[^\w\s]", revised)

        matcher = difflib.SequenceMatcher(None, original_tokens, revised_tokens)
        html_parts: List[str] = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            orig_sub = "".join(original_tokens[i1:i2])
            rev_sub = "".join(revised_tokens[j1:j2])

            if tag == "equal":
                html_parts.append(orig_sub)
            elif tag == "delete":
                html_parts.append(f'<del class="diff-del">{orig_sub}</del>')
            elif tag == "insert":
                html_parts.append(f'<ins class="diff-add">{rev_sub}</ins>')
            elif tag == "replace":
                html_parts.append(f'<del class="diff-del">{orig_sub}</del><ins class="diff-add">{rev_sub}</ins>')

        # Merge spaces and tidy up formatting
        combined_html = "".join(html_parts)
        return combined_html

    @staticmethod
    def get_sentence_diffs(original: str, revised: str) -> Dict[str, Any]:
        """
        Break text into sentences and compare them to classify added/removed/modified.
        """
        orig_sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', original) if s.strip()]
        rev_sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', revised) if s.strip()]

        matcher = difflib.SequenceMatcher(None, orig_sents, rev_sents)
        diff_sents: List[Dict[str, Any]] = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                for s in orig_sents[i1:i2]:
                    diff_sents.append({"status": "unchanged", "text": s})
            elif tag == "delete":
                for s in orig_sents[i1:i2]:
                    diff_sents.append({"status": "deleted", "text": s})
            elif tag == "insert":
                for s in rev_sents[j1:j2]:
                    diff_sents.append({"status": "added", "text": s})
            elif tag == "replace":
                # Match them up to see word differences
                o_s = orig_sents[i1:i2]
                r_s = rev_sents[j1:j2]
                max_len = max(len(o_s), len(r_s))
                for idx in range(max_len):
                    if idx < len(o_s) and idx < len(r_s):
                        diff_sents.append({
                            "status": "modified",
                            "original": o_s[idx],
                            "revised": r_s[idx],
                            "html": DifferenceEngine.get_word_diff_html(o_s[idx], r_s[idx])
                        })
                    elif idx < len(o_s):
                        diff_sents.append({"status": "deleted", "text": o_s[idx]})
                    else:
                        diff_sents.append({"status": "added", "text": r_s[idx]})

        return {
            "sentences": diff_sents,
            "original_count": len(orig_sents),
            "revised_count": len(rev_sents),
        }

    @staticmethod
    def compare_documents(original_paras: List[str], revised_paras: List[str]) -> List[Dict[str, Any]]:
        """
        Compare paragraph lists and highlight differences.
        """
        matcher = difflib.SequenceMatcher(None, original_paras, revised_paras)
        para_diffs: List[Dict[str, Any]] = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                for s in original_paras[i1:i2]:
                    para_diffs.append({
                        "status": "unchanged",
                        "original": s,
                        "revised": s,
                        "html": s
                    })
            elif tag == "delete":
                for s in original_paras[i1:i2]:
                    para_diffs.append({
                        "status": "deleted",
                        "original": s,
                        "revised": "",
                        "html": f'<del class="diff-del">{s}</del>'
                    })
            elif tag == "insert":
                for s in revised_paras[j1:j2]:
                    para_diffs.append({
                        "status": "added",
                        "original": "",
                        "revised": s,
                        "html": f'<ins class="diff-add">{s}</ins>'
                    })
            elif tag == "replace":
                o_p = original_paras[i1:i2]
                r_p = revised_paras[j1:j2]
                max_len = max(len(o_p), len(r_p))
                for idx in range(max_len):
                    if idx < len(o_p) and idx < len(r_p):
                        para_diffs.append({
                            "status": "modified",
                            "original": o_p[idx],
                            "revised": r_p[idx],
                            "html": DifferenceEngine.get_word_diff_html(o_p[idx], r_p[idx])
                        })
                    elif idx < len(o_p):
                        para_diffs.append({
                            "status": "deleted",
                            "original": o_p[idx],
                            "revised": "",
                            "html": f'<del class="diff-del">{o_p[idx]}</del>'
                        })
                    else:
                        para_diffs.append({
                            "status": "added",
                            "original": "",
                            "revised": r_p[idx],
                            "html": f'<ins class="diff-add">{r_p[idx]}</ins>'
                        })

        return para_diffs
