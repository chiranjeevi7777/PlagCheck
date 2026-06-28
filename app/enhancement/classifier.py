"""
Paragraph Classification Service.

Categorises document paragraphs into originality, readability, style,
and tone issue categories using hybrid local metrics and LLM verification.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.core.logging import get_logger
from app.llm.groq_client import GroqAPIClient
from app.enhancement.metrics import WritingAnalytics

logger = get_logger(__name__)


class ParagraphClassificationResult(BaseModel):
    """Pydantic model representing LLM classification output."""
    categories: List[str] = Field(default_factory=list, description="List of identified issue categories")
    issues: List[str] = Field(default_factory=list, description="Specific details of issues found")
    weaknesses: List[str] = Field(default_factory=list, description="General structural or style weaknesses")
    is_fine: bool = Field(default=True, description="True if paragraph requires no revisions")


class ParagraphClassifier:
    """Classifies a text paragraph to detect style, originality, and structure issues."""

    def __init__(self, groq_client: Optional[GroqAPIClient] = None) -> None:
        self.groq_client = groq_client or GroqAPIClient()

    def classify(self, text: str, similarity_score: float = 0.0) -> ParagraphClassificationResult:
        """Classify a paragraph using a hybrid approach: local metrics + LLM verification."""
        text = text.strip()
        if not text:
            return ParagraphClassificationResult(is_fine=True)

        categories: List[str] = []
        issues: List[str] = []

        # ── 1. Local Heuristics & Writing Metrics ─────────────────────────────
        metrics = WritingAnalytics.analyze(text)
        
        # Readability check
        fre = metrics["readability"]["flesch_reading_ease"]
        if fre < 45.0:
            categories.append("Low Readability")
            issues.append(f"Low readability score (Flesch Ease: {fre}). Needs simpler sentence structures.")
            
        # Passive Voice
        pv_pct = metrics["lexical"]["passive_voice_percentage"]
        if pv_pct > 25.0:
            categories.append("Passive Voice")
            issues.append(f"High passive voice usage ({pv_pct}%). Should use active voice verbs.")
            
        # Repetitive Writing
        reps = metrics["lexical"]["repeated_phrases"]
        if reps:
            categories.append("Repetitive Writing")
            rep_desc = ", ".join(f"'{k}' ({v}x)" for k, v in list(reps.items())[:3])
            issues.append(f"Repeated word groups found: {rep_desc}")
            
        # Vocabulary Diversity
        ttr = metrics["lexical"]["type_token_ratio"]
        if ttr < 0.45 and len(text.split()) > 40:
            categories.append("Low Vocabulary Diversity")
            issues.append(f"Low vocabulary variation (TTR: {ttr:.2f}). Needs word variety.")

        # Similarity overlap checks
        if similarity_score >= 0.70:
            categories.append("High Similarity")
            issues.append(f"High plagiarism match detected (similarity: {similarity_score * 100:.1f}%).")
        elif similarity_score >= 0.35:
            categories.append("Needs Better Citation")
            issues.append(f"Moderate match detected (similarity: {similarity_score * 100:.1f}%). Attribution or citation enhancement needed.")
        elif similarity_score >= 0.15:
            categories.append("Low Originality")
            issues.append(f"Minor overlap with external literature (similarity: {similarity_score * 100:.1f}%). Recommend paraphrasing.")

        # ── 2. LLM Style and Tone Verification ─────────────────────────────────
        # We query the LLM to analyze nuance (Academic tone, mechanical writing style, grammar, flow)
        system_prompt = (
            "You are a professional academic editor. Analyze the provided paragraph for structural, "
            "stylistic, and grammatical issues. Respond ONLY with a raw JSON object containing the "
            "classification results. Do not include markdown formatting or backticks."
        )

        user_prompt = (
            f"Analyze this paragraph and determine if any of these issues are present:\n"
            f"- 'Mechanical Writing Style' (extremely predictable or robotic)\n"
            f"- 'Weak Academic Tone' (too informal, vague phrasing)\n"
            f"- 'Grammar Issues' (syntax, spelling, agreement errors)\n"
            f"- 'Sentence Flow Problems' (abrupt transitions, awkward phrasing)\n\n"
            f"Paragraph:\n\"{text}\"\n\n"
            f"Return JSON shape:\n"
            f'{{\n'
            f'  "categories": ["CategoryName1", ...],\n'
            f'  "weaknesses": ["description of weakness 1", ...],\n'
            f'  "is_fine": true/false\n'
            f'}}'
        )

        try:
            response_text = self.groq_client.chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_format={"type": "json_object"}
            )
            data = json.loads(response_text)
            
            llm_categories = data.get("categories", [])
            for cat in llm_categories:
                if cat in {"Mechanical Writing Style", "Weak Academic Tone", "Grammar Issues", "Sentence Flow Problems"}:
                    if cat not in categories:
                        categories.append(cat)
                        
            weaknesses = data.get("weaknesses", [])
            for w in weaknesses:
                issues.append(f"LLM Editor: {w}")
                
            is_fine = data.get("is_fine", True) and (not categories)
        except Exception as e:
            logger.error(f"LLM paragraph classification failed, relying on local: {e}")
            is_fine = not categories

        return ParagraphClassificationResult(
            categories=categories,
            issues=issues,
            weaknesses=issues,
            is_fine=is_fine
        )

    @staticmethod
    def classify_paragraph(text: str, metrics: Dict[str, Any] = None) -> Dict[str, Any]:
        """Classify paragraph and return a dictionary for routing layer."""
        classifier = ParagraphClassifier()
        res = classifier.classify(text)
        
        category = "Standard"
        if not res.is_fine and res.categories:
            category = res.categories[0]
            
        return {
            "category": category,
            "categories": res.categories,
            "issues": res.issues,
            "weaknesses": res.weaknesses,
            "is_fine": res.is_fine
        }

