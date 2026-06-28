"""
Paragraph Rewriter Service.

Implements the paragraph-level LLM revision engine with in-line integrity validation
and single-attempt retry correction loops.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from app.core.logging import get_logger
from app.llm.groq_client import GroqAPIClient
from app.enhancement.planner import ParagraphEnhancementPlan
from app.enhancement.validator import RevisionValidator

logger = get_logger(__name__)


class ParagraphRewriter:
    """Revises a paragraph to resolve flagged issues while maintaining technical facts and references."""

    def __init__(self, groq_client: Optional[GroqAPIClient] = None) -> None:
        self.groq_client = groq_client or GroqAPIClient()

    def rewrite(
        self, text: str, plan: ParagraphEnhancementPlan, attempt: int = 1
    ) -> str:
        """
        Revises the text using Groq and validates the output.
        Retries once if validation fails.
        """
        system_prompt = (
            "You are a Senior Academic Editor. Your goal is to improve the academic quality of the text.\n"
            "CRITICAL RULES:\n"
            "1. You MUST preserve all references, citations (e.g. [1], (Smith, 2020)), and numbers exactly as written.\n"
            "2. Preserve all LaTeX or mathematical equations (e.g. $y=mx+c$) exactly.\n"
            "3. Maintain technical terminology and core factual meaning.\n"
            "4. NEVER invent new facts or add references that were not in the original text.\n"
            "5. Do NOT summarize or shorten the text. Rewrite at similar length with improved tone and flow.\n"
            "6. Output ONLY the revised paragraph text. Do not write intros like 'Here is the revised paragraph' or markdown fences."
        )

        user_prompt = (
            f"Original Paragraph:\n\"{text}\"\n\n"
            f"Revision Instructions:\n" + "\n".join(f"- {i}" for i in plan.issues_found) + "\n"
            f"Improvement Plan: {plan.improvement_plan}\n\n"
            f"Write the revised academic paragraph now:"
        )

        try:
            revised_text = self.groq_client.chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt
            ).strip()

            # Clean any quotes the model might have added
            if revised_text.startswith('"') and revised_text.endswith('"'):
                revised_text = revised_text[1:-1].strip()

            # ── Validation ────────────────────────────────────────────────────
            is_valid, errors = RevisionValidator.validate(text, revised_text)
            if is_valid:
                return revised_text

            # If invalid and first attempt, retry once with the error feedback
            if attempt == 1:
                logger.warning(
                    f"First revision attempt failed validation: {errors}. Retrying with feedback..."
                )
                retry_user_prompt = (
                    f"{user_prompt}\n\n"
                    f"CRITICAL FEEDBACK: Your previous revision failed validation because:\n"
                    + "\n".join(f"- {e}" for e in errors) + "\n\n"
                    f"Please rewrite the paragraph again, correcting these issues:"
                )
                
                revised_text_retry = self.groq_client.chat(
                    system_prompt=system_prompt,
                    user_prompt=retry_user_prompt
                ).strip()

                if revised_text_retry.startswith('"') and revised_text_retry.endswith('"'):
                    revised_text_retry = revised_text_retry[1:-1].strip()

                is_valid_retry, errors_retry = RevisionValidator.validate(text, revised_text_retry)
                if is_valid_retry:
                    return revised_text_retry
                else:
                    logger.error(
                        f"Second revision attempt failed validation: {errors_retry}. Falling back to original paragraph."
                    )
            else:
                logger.error("Validation failed on retry. Falling back to original.")

        except Exception as e:
            logger.error(f"Error during paragraph revision: {e}", exc_info=True)

        # Fallback: return the original paragraph if all else fails
        return text

    @staticmethod
    def rewrite_paragraph(
        original_text: str, plan_strategy: str, focus_area: str
    ) -> str:
        """Compatibility helper method for routing layer."""
        from app.enhancement.planner import ParagraphEnhancementPlan
        plan = ParagraphEnhancementPlan(
            issues_found=[f"Improve writing quality in focus area: {focus_area}"],
            priority="medium",
            improvement_plan=plan_strategy,
            estimated_benefit=f"Enhanced {focus_area}",
            confidence=0.9
        )
        rewriter = ParagraphRewriter()
        return rewriter.rewrite(original_text, plan)

