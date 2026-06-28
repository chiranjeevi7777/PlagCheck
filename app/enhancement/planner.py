"""
Enhancement Planning Service.

Formulates a detailed step-by-step revision plan for flagged paragraphs,
including priorities, estimated benefit, and confidence ratings.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.core.logging import get_logger
from app.llm.groq_client import GroqAPIClient
from app.enhancement.classifier import ParagraphClassificationResult

logger = get_logger(__name__)


class ParagraphEnhancementPlan(BaseModel):
    """Pydantic model representing a structured enhancement plan for a paragraph."""
    issues_found: List[str] = Field(..., description="List of specific issues flagged")
    priority: str = Field("medium", description="Priority level: high, medium, low")
    improvement_plan: str = Field(..., description="Step-by-step plan for revising the paragraph")
    estimated_benefit: str = Field(..., description="Expected improvement in readability, originality, or style")
    confidence: float = Field(0.9, description="Confidence rating of the plan (0.0 to 1.0)")


class ParagraphPlanner:
    """Generates revision plans for paragraphs requiring improvement."""

    def __init__(self, groq_client: Optional[GroqAPIClient] = None) -> None:
        self.groq_client = groq_client or GroqAPIClient()

    def generate_plan(
        self, text: str, classification: ParagraphClassificationResult
    ) -> ParagraphEnhancementPlan:
        """Create a revision plan for the paragraph."""
        if classification.is_fine or not classification.categories:
            return ParagraphEnhancementPlan(
                issues_found=[],
                priority="low",
                improvement_plan="No improvements required. The paragraph already meets quality standards.",
                estimated_benefit="None",
                confidence=1.0,
            )

        # Determine priority based on categories
        priority = "low"
        if any(c in {"High Similarity", "Needs Better Citation", "Grammar Issues"} for c in classification.categories):
            priority = "high"
        elif any(c in {"Low Readability", "Low Originality", "Mechanical Writing Style"} for c in classification.categories):
            priority = "medium"

        system_prompt = (
            "You are an academic writing director. Given a paragraph and its identified issues, "
            "formulate a detailed improvement plan. Respond ONLY with a raw JSON object matching the "
            "requested shape. Do not wrap in markdown code blocks."
        )

        user_prompt = (
            f"Paragraph:\n\"{text}\"\n\n"
            f"Issues Flagged:\n" + "\n".join(f"- {i}" for i in classification.issues) + "\n\n"
            f"Categories:\n" + "\n".join(f"- {c}" for c in classification.categories) + "\n\n"
            f"Create a revision plan in JSON format:\n"
            f'{{\n'
            f'  "issues_found": ["issue 1", "issue 2", ...],\n'
            f'  "priority": "high/medium/low",\n'
            f'  "improvement_plan": "step-by-step revision strategy description",\n'
            f'  "estimated_benefit": "what will improve (e.g. increase clarity, fix grammar)",\n'
            f'  "confidence": 0.95\n'
            f'}}'
        )

        try:
            response_text = self.groq_client.chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_format={"type": "json_object"}
            )
            data = json.loads(response_text)
            
            # Fill with defaults if keys are missing
            return ParagraphEnhancementPlan(
                issues_found=data.get("issues_found", classification.issues),
                priority=data.get("priority", priority),
                improvement_plan=data.get("improvement_plan", "Paraphrase paragraph to improve sentence flow and academic tone."),
                estimated_benefit=data.get("estimated_benefit", "Improve readability and original phrasing."),
                confidence=float(data.get("confidence", 0.90))
            )
        except Exception as e:
            logger.error(f"Failed to generate enhancement plan via LLM: {e}")
            # Fallback to local heuristic planner
            plan_desc = f"Paraphrase text to resolve: {', '.join(classification.categories)}. Focus on active voice and sentence structure."
            benefit_desc = "Enhances readability score and increases originality index."
            return ParagraphEnhancementPlan(
                issues_found=classification.issues,
                priority=priority,
                improvement_plan=plan_desc,
                estimated_benefit=benefit_desc,
                confidence=0.80
            )


class RevisionPlanner:
    """Compatibility wrapper for ParagraphPlanner."""

    @staticmethod
    def generate_revision_plan(text: str, classification: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a revision plan dictionary for the routing layer."""
        from app.enhancement.classifier import ParagraphClassificationResult
        cls_obj = ParagraphClassificationResult(
            categories=classification.get("categories", []),
            issues=classification.get("issues", []),
            weaknesses=classification.get("weaknesses", []),
            is_fine=classification.get("is_fine", True)
        )
        planner = ParagraphPlanner()
        plan_obj = planner.generate_plan(text, cls_obj)
        return {
            "strategy": plan_obj.improvement_plan,
            "improvement_plan": plan_obj.improvement_plan,
            "issues_found": plan_obj.issues_found,
            "issues": plan_obj.issues_found,
            "priority": plan_obj.priority,
            "estimated_benefit": plan_obj.estimated_benefit,
            "expected_benefit": plan_obj.estimated_benefit,
            "confidence": plan_obj.confidence
        }


