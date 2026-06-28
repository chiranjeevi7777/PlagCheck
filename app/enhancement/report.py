"""
Enhancement Report Service.

Compiles "Before vs After" JSON and PDF reports comparing writing metrics,
originality gains, and paragraph-level changes.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

from app.core.logging import get_logger
from app.enhancement.comparison import DocumentVersion

logger = get_logger(__name__)


class EnhancementReportCompiler:
    """Compiles analytic and visual reports highlighting document enhancements."""

    @staticmethod
    def compile_metrics_comparison(v1: DocumentVersion, v2: DocumentVersion) -> Dict[str, Any]:
        """Compute differences in readability, passive voice, lexical density, and tone."""
        m1 = v1.metrics
        m2 = v2.metrics

        def safe_diff(val2: Any, val1: Any) -> float:
            try:
                return round(float(val2) - float(val1), 2)
            except (ValueError, TypeError):
                return 0.0

        comparison = {
            "readability": {
                "flesch_reading_ease": {
                    "before": m1.get("readability", {}).get("flesch_reading_ease", 0),
                    "after": m2.get("readability", {}).get("flesch_reading_ease", 0),
                    "diff": safe_diff(m2.get("readability", {}).get("flesch_reading_ease", 0), m1.get("readability", {}).get("flesch_reading_ease", 0)),
                },
                "flesch_kincaid_grade": {
                    "before": m1.get("readability", {}).get("flesch_kincaid_grade", 0),
                    "after": m2.get("readability", {}).get("flesch_kincaid_grade", 0),
                    "diff": safe_diff(m2.get("readability", {}).get("flesch_kincaid_grade", 0), m1.get("readability", {}).get("flesch_kincaid_grade", 0)),
                },
                "gunning_fog": {
                    "before": m1.get("readability", {}).get("gunning_fog", 0),
                    "after": m2.get("readability", {}).get("gunning_fog", 0),
                    "diff": safe_diff(m2.get("readability", {}).get("gunning_fog", 0), m1.get("readability", {}).get("gunning_fog", 0)),
                }
            },
            "lexical": {
                "lexical_density": {
                    "before": m1.get("lexical", {}).get("lexical_density", 0),
                    "after": m2.get("lexical", {}).get("lexical_density", 0),
                    "diff": safe_diff(m2.get("lexical", {}).get("lexical_density", 0), m1.get("lexical", {}).get("lexical_density", 0)),
                },
                "type_token_ratio": {
                    "before": m1.get("lexical", {}).get("type_token_ratio", 0),
                    "after": m2.get("lexical", {}).get("type_token_ratio", 0),
                    "diff": safe_diff(m2.get("lexical", {}).get("type_token_ratio", 0), m1.get("lexical", {}).get("type_token_ratio", 0)),
                }
            },
            "grammar": {
                "passive_voice_pct": {
                    "before": m1.get("grammar", {}).get("passive_voice_pct", 0),
                    "after": m2.get("grammar", {}).get("passive_voice_pct", 0),
                    "diff": safe_diff(m2.get("grammar", {}).get("passive_voice_pct", 0), m1.get("grammar", {}).get("passive_voice_pct", 0)),
                }
            },
            "structure": {
                "word_count": {
                    "before": m1.get("structure", {}).get("word_count", 0),
                    "after": m2.get("structure", {}).get("word_count", 0),
                    "diff": safe_diff(m2.get("structure", {}).get("word_count", 0), m1.get("structure", {}).get("word_count", 0)),
                },
                "paragraph_count": {
                    "before": m1.get("structure", {}).get("paragraph_count", 0),
                    "after": m2.get("structure", {}).get("paragraph_count", 0),
                    "diff": safe_diff(m2.get("structure", {}).get("paragraph_count", 0), m1.get("structure", {}).get("paragraph_count", 0)),
                }
            }
        }
        return comparison

    @classmethod
    def generate_comparison_pdf(
        cls,
        v1: DocumentVersion,
        v2: DocumentVersion,
        output_path: Path,
        diff_html_paras: List[Dict[str, Any]]
    ) -> None:
        """Generates a professional PDF comparison report."""
        logger.info(f"Generating PDF comparison report at {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=letter,
            rightMargin=54, leftMargin=54, topMargin=54, bottomMargin=54
        )

        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            'CompTitle', parent=styles['Heading1'], fontSize=22, leading=26,
            textColor=colors.HexColor('#1E1E2F'), spaceAfter=12
        )
        section_style = ParagraphStyle(
            'CompSection', parent=styles['Heading2'], fontSize=15, leading=19,
            textColor=colors.HexColor('#2A2D34'), spaceBefore=18, spaceAfter=10
        )
        body_style = ParagraphStyle(
            'CompBody', parent=styles['Normal'], fontSize=10, leading=14,
            textColor=colors.HexColor('#4A4A4A')
        )
        label_style = ParagraphStyle(
            'CompLabel', parent=body_style, fontName='Helvetica-Bold',
            textColor=colors.HexColor('#2E2E2E')
        )
        
        card_title_style = ParagraphStyle(
            'CompCardTitle', parent=body_style, fontName='Helvetica-Bold',
            fontSize=9, leading=11, alignment=1, textColor=colors.HexColor('#6B7280')
        )
        card_val_lg = ParagraphStyle(
            'CompCardValLg', parent=body_style, fontName='Helvetica-Bold',
            fontSize=18, leading=22, alignment=1
        )

        story = []

        # Title
        story.append(Paragraph("Document Originality & Quality Enhancement Report", title_style))
        story.append(Paragraph(f"Compiled on: {time.strftime('%Y-%m-%d %H:%M:%S')}", body_style))
        story.append(Spacer(1, 15))

        # Metrics Compilation
        comp_metrics = cls.compile_metrics_comparison(v1, v2)

        # Overview Table / Dashboard Card
        ease_before = comp_metrics["readability"]["flesch_reading_ease"]["before"]
        ease_after = comp_metrics["readability"]["flesch_reading_ease"]["after"]
        grade_before = comp_metrics["readability"]["flesch_kincaid_grade"]["before"]
        grade_after = comp_metrics["readability"]["flesch_kincaid_grade"]["after"]
        ttr_before = comp_metrics["lexical"]["type_token_ratio"]["before"]
        ttr_after = comp_metrics["lexical"]["type_token_ratio"]["after"]

        # Color highlight
        ease_color = "#34C759" if ease_after > ease_before else "#FF9500"

        kpi_data = [
            [
                [
                    Paragraph("Flesch Reading Ease", card_title_style),
                    Spacer(1, 4),
                    Paragraph(f"<font color='{ease_color}'>{ease_before} &rarr; {ease_after}</font>", card_val_lg)
                ],
                [
                    Paragraph("Flesch Grade Level", card_title_style),
                    Spacer(1, 4),
                    Paragraph(f"{grade_before} &rarr; {grade_after}", card_val_lg)
                ],
                [
                    Paragraph("Lexical Diversity (TTR)", card_title_style),
                    Spacer(1, 4),
                    Paragraph(f"{ttr_before} &rarr; {ttr_after}", card_val_lg)
                ],
            ]
        ]
        
        kpi_table = Table(kpi_data, colWidths=[170, 170, 170])
        kpi_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8F9FA')),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#E5E7EB')),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E5E7EB')),
            ('TOPPADDING', (0,0), (-1,-1), 10),
            ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ]))

        story.append(Paragraph("Enhancement Performance Dashboard", section_style))
        story.append(kpi_table)
        story.append(Spacer(1, 15))

        # Full Metrics Comparison table
        metrics_table_rows = [
            [Paragraph("Category / Metric", label_style), Paragraph("Original (v1)", label_style), Paragraph("Enhanced (v2)", label_style), Paragraph("Change", label_style)],
            [Paragraph("Flesch Reading Ease", body_style), Paragraph(str(ease_before), body_style), Paragraph(str(ease_after), body_style), Paragraph(f"+{comp_metrics['readability']['flesch_reading_ease']['diff']}", body_style)],
            [Paragraph("Flesch Kincaid Grade", body_style), Paragraph(str(grade_before), body_style), Paragraph(str(grade_after), body_style), Paragraph(str(comp_metrics['readability']['flesch_kincaid_grade']['diff']), body_style)],
            [Paragraph("Gunning Fog Index", body_style), Paragraph(str(comp_metrics['readability']['gunning_fog']['before']), body_style), Paragraph(str(comp_metrics['readability']['gunning_fog']['after']), body_style), Paragraph(str(comp_metrics['readability']['gunning_fog']['diff']), body_style)],
            [Paragraph("Type-Token Ratio", body_style), Paragraph(str(ttr_before), body_style), Paragraph(str(ttr_after), body_style), Paragraph(f"+{comp_metrics['lexical']['type_token_ratio']['diff']}", body_style)],
            [Paragraph("Passive Voice (%)", body_style), Paragraph(f"{comp_metrics['grammar']['passive_voice_pct']['before']}%", body_style), Paragraph(f"{comp_metrics['grammar']['passive_voice_pct']['after']}%", body_style), Paragraph(f"{comp_metrics['grammar']['passive_voice_pct']['diff']}%", body_style)],
            [Paragraph("Total Word Count", body_style), Paragraph(str(comp_metrics['structure']['word_count']['before']), body_style), Paragraph(str(comp_metrics['structure']['word_count']['after']), body_style), Paragraph(str(comp_metrics['structure']['word_count']['diff']), body_style)],
        ]
        
        metrics_table = Table(metrics_table_rows, colWidths=[200, 100, 100, 110])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F2F2F7')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#D1D1D6')),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        
        story.append(Paragraph("Detailed Analytical Breakdown", section_style))
        story.append(metrics_table)
        story.append(Spacer(1, 15))
        story.append(PageBreak())

        # Side-by-side or Stacked paragraph changes
        story.append(Paragraph("Paragraph-by-Paragraph Revisions", section_style))
        
        highlight_style = ParagraphStyle(
            'CompHighlightText', parent=body_style, fontSize=9, leading=13
        )

        for idx, diff in enumerate(diff_html_paras):
            status = diff.get("status", "unchanged")
            if status == "unchanged":
                continue

            status_color = "#34C759" if status == "added" else ("#FF3B30" if status == "deleted" else "#FF9500")
            
            chunk_header = [
                [
                    Paragraph(f"<b>Segment {idx + 1} ({status.capitalize()})</b>", label_style),
                    Paragraph(f"<font color='{status_color}'><b>Action: {status.upper()}</b></font>", label_style)
                ]
            ]
            header_table = Table(chunk_header, colWidths=[250, 260])
            header_table.setStyle(TableStyle([
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.HexColor('#D1D1D6')),
                ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ]))

            text_side_by_side = [
                [
                    Paragraph("<b>Original Version</b>", body_style),
                    Paragraph("<b>Enhanced Version</b>", body_style)
                ],
                [
                    Paragraph(diff.get("original", "") or "(Empty)", highlight_style),
                    Paragraph(diff.get("html", "") or diff.get("revised", "") or "(Empty)", highlight_style)
                ]
            ]
            
            side_table = Table(text_side_by_side, colWidths=[250, 260])
            side_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F9FAFB')),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#E5E7EB')),
                ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E5E7EB')),
                ('TOPPADDING', (0,0), (-1,-1), 6),
                ('BOTTOMPADDING', (0,0), (-1,-1), 6),
                ('LEFTPADDING', (0,0), (-1,-1), 6),
                ('RIGHTPADDING', (0,0), (-1,-1), 6),
            ]))

            block = []
            block.append(header_table)
            block.append(Spacer(1, 4))
            block.append(side_table)
            block.append(Spacer(1, 15))

            story.append(KeepTogether(block))

        doc.build(story)
        logger.info("PDF comparison report compiled successfully.")
