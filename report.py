import time
import xml.sax.saxutils as saxutils
from pathlib import Path
from typing import List, Dict, Any
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from utils import logger
from config import settings

class PlagiarismReporter:
    """Aggregates results and generates analytical reports (JSON and PDF)."""

    @staticmethod
    def aggregate_results(
        results: List[Dict[str, Any]], 
        orig_meta: Dict[str, Any], 
        susp_meta: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Aggregates chunk-level comparison results into an overall report."""
        if not results:
            return {
                "overall_similarity": 0,
                "overall_exact_copy": 0,
                "overall_paraphrase": 0,
                "average_confidence": 0,
                "flagged_chunks_count": 0,
                "highest_similarity": 0,
                "lowest_similarity": 0,
                "average_similarity": 0,
                "classification_counts": {},
                "chunks": []
            }

        total_chunks = len(results)
        similarities = [r["semantic_similarity"] for r in results]
        exact_copies = [r["exact_copy"] for r in results]
        paraphrases = [r["paraphrase"] for r in results]
        confidences = [r["confidence"] for r in results]

        avg_similarity = sum(similarities) / total_chunks
        avg_exact_copy = sum(exact_copies) / total_chunks
        avg_paraphrase = sum(paraphrases) / total_chunks
        avg_confidence = sum(confidences) / total_chunks
        highest_similarity = max(similarities)
        lowest_similarity = min(similarities)

        # Count classifications
        classification_counts = {}
        flagged_chunks_count = 0
        
        for r in results:
            cls = r["classification"]
            classification_counts[cls] = classification_counts.get(cls, 0) + 1
            if r["semantic_similarity"] >= 30:  # Flagged threshold
                flagged_chunks_count += 1

        # Heuristic for overall document classification
        if avg_similarity >= 80:
            overall_classification = "Near Duplicate"
        elif avg_similarity >= 60:
            overall_classification = "Heavy Paraphrasing"
        elif avg_similarity >= 40:
            overall_classification = "Heavy Rewrite"
        elif avg_similarity >= 20:
            overall_classification = "Light Rewrite"
        elif avg_similarity >= 10:
            overall_classification = "Minor Similarity"
        else:
            overall_classification = "Original"

        return {
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "query": orig_meta.get("query", ""),
                "paper_count": orig_meta.get("paper_count", 0),
                "original_filename": orig_meta.get("filename", "N/A"),
                "original_word_count": orig_meta.get("word_count", 0),
                "original_chunk_count": orig_meta.get("chunk_count", 0),
                "suspected_filename": susp_meta.get("filename", "N/A"),
                "suspected_word_count": susp_meta.get("word_count", 0),
                "suspected_chunk_count": susp_meta.get("chunk_count", 0),
            },
            "overall_similarity": round(avg_similarity),
            "overall_exact_copy": round(avg_exact_copy),
            "overall_paraphrase": round(avg_paraphrase),
            "overall_classification": overall_classification,
            "average_confidence": round(avg_confidence),
            "flagged_chunks_count": flagged_chunks_count,
            "highest_similarity": highest_similarity,
            "lowest_similarity": lowest_similarity,
            "average_similarity": round(avg_similarity),
            "classification_counts": classification_counts,
            "chunks": results
        }

    @staticmethod
    def _escape(text: str) -> str:
        """Escape text for ReportLab paragraph compatibility (prevent XML parsing errors)."""
        return saxutils.escape(text)

    @classmethod
    def _highlight_text(cls, chunk_text: str, sentence_matches: List[Dict[str, Any]], is_suspected: bool) -> str:
        """Wraps matched sentences in HTML-like font tags for ReportLab Paragraph highlighting."""
        # Split text into sentences (crude boundary splitting for display)
        # We can reconstruct the chunk by matching sentences
        escaped_chunk = cls._escape(chunk_text)
        
        if not sentence_matches:
            return escaped_chunk

        # Sort matches by length of suspected_sentence descending to avoid nested replacement errors
        sorted_matches = sorted(sentence_matches, key=lambda x: len(x["suspected_sentence"]), reverse=True)
        
        highlighted = escaped_chunk
        for match in sorted_matches:
            target_sentence = match["suspected_sentence"] if is_suspected else match["original_sentence"]
            escaped_target = cls._escape(target_sentence)
            
            if not escaped_target or escaped_target not in highlighted:
                continue

            color = "#ff4d4d" if match["match_type"] == "exact_copy" else "#ffa333"
            replacement = f"<b><font color='{color}'>{escaped_target}</font></b>"
            
            # Simple replace
            highlighted = highlighted.replace(escaped_target, replacement)
            
        return highlighted

    @classmethod
    def generate_pdf_report(cls, report_data: Dict[str, Any], output_path: Path) -> None:
        """Generates a professional PDF report from the aggregated plagiarism metrics."""
        logger.info(f"Generating PDF report at {output_path}")
        
        # Page dimensions: letter (8.5 x 11 inches)
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=letter,
            rightMargin=54, leftMargin=54, topMargin=54, bottomMargin=54
        )

        styles = getSampleStyleSheet()
        
        # Define Custom Styles
        title_style = ParagraphStyle(
            'ReportTitle',
            parent=styles['Heading1'],
            fontSize=24,
            leading=28,
            textColor=colors.HexColor('#1E1E2F'),
            spaceAfter=15
        )
        
        section_style = ParagraphStyle(
            'ReportSection',
            parent=styles['Heading2'],
            fontSize=16,
            leading=20,
            textColor=colors.HexColor('#2A2D34'),
            spaceBefore=15,
            spaceAfter=10
        )
        
        body_style = ParagraphStyle(
            'ReportBody',
            parent=styles['Normal'],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor('#4A4A4A')
        )

        meta_label_style = ParagraphStyle(
            'MetaLabel',
            parent=body_style,
            fontName='Helvetica-Bold',
            textColor=colors.HexColor('#2E2E2E')
        )

        highlight_style = ParagraphStyle(
            'HighlightText',
            parent=body_style,
            fontSize=9,
            leading=13
        )

        # Dashboard Card Styles to prevent font-leading overlap
        card_title_style = ParagraphStyle(
            'CardTitle',
            parent=body_style,
            fontName='Helvetica-Bold',
            fontSize=9,
            leading=11,
            alignment=1,  # Center
            textColor=colors.HexColor('#6B7280')
        )
        
        card_value_large_style = ParagraphStyle(
            'CardValueLarge',
            parent=body_style,
            fontName='Helvetica-Bold',
            fontSize=22,
            leading=26,
            alignment=1,  # Center
        )
        
        card_value_small_style = ParagraphStyle(
            'CardValueSmall',
            parent=body_style,
            fontName='Helvetica-Bold',
            fontSize=13,
            leading=16,
            alignment=1,  # Center
            textColor=colors.HexColor('#111827')
        )

        story = []

        # 1. Header Block / Title
        story.append(Paragraph("AI Plagiarism Analysis Report", title_style))
        story.append(Paragraph(f"Generated on: {report_data['metadata']['timestamp']}", body_style))
        story.append(Spacer(1, 15))

        # 2. Document Metadata Table
        meta = report_data["metadata"]
        if meta.get("query"):
            meta_data = [
                [Paragraph("Metric", meta_label_style), Paragraph("Analysis Scope", meta_label_style)],
                [Paragraph("Uploaded Filename", body_style), Paragraph(cls._escape(meta["suspected_filename"]), body_style)],
                [Paragraph("Word Count", body_style), Paragraph(str(meta["suspected_word_count"]), body_style)],
                [Paragraph("Document Chunks", body_style), Paragraph(str(meta["suspected_chunk_count"]), body_style)],
                [Paragraph("Scholar Search Query", body_style), Paragraph(cls._escape(meta["query"]), body_style)],
                [Paragraph("Reference Papers Checked", body_style), Paragraph(str(meta["paper_count"]), body_style)],
            ]
            meta_table = Table(meta_data, colWidths=[200, 310])
        else:
            meta_data = [
                [Paragraph("Metric", meta_label_style), Paragraph("Original Document", meta_label_style), Paragraph("Suspected Document", meta_label_style)],
                [Paragraph("Filename", body_style), Paragraph(cls._escape(meta["original_filename"]), body_style), Paragraph(cls._escape(meta["suspected_filename"]), body_style)],
                [Paragraph("Word Count", body_style), Paragraph(str(meta["original_word_count"]), body_style), Paragraph(str(meta["suspected_word_count"]), body_style)],
                [Paragraph("Semantic Chunks", body_style), Paragraph(str(meta["original_chunk_count"]), body_style), Paragraph(str(meta["suspected_chunk_count"]), body_style)],
            ]
            meta_table = Table(meta_data, colWidths=[150, 180, 180])
        
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F2F2F7')),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#D1D1D6')),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 20))

        # 3. Overall Performance Dashboard Cards
        sim_score = report_data["overall_similarity"]
        exact_score = report_data["overall_exact_copy"]
        para_score = report_data["overall_paraphrase"]
        classification = report_data["overall_classification"]
        
        # Color definitions based on similarity score
        score_color = "#34C759"  # Green
        if sim_score >= 60:
            score_color = "#FF3B30"  # Red
        elif sim_score >= 30:
            score_color = "#FF9500"  # Orange

        card_data = [
            [
                [
                    Paragraph("Overall Similarity", card_title_style),
                    Spacer(1, 4),
                    Paragraph(f"<font color='{score_color}'>{sim_score}%</font>", card_value_large_style)
                ],
                [
                    Paragraph("Exact Copying", card_title_style),
                    Spacer(1, 4),
                    Paragraph(f"<font color='#1F2937'>{exact_score}%</font>", card_value_large_style)
                ],
                [
                    Paragraph("Paraphrasing", card_title_style),
                    Spacer(1, 4),
                    Paragraph(f"<font color='#1F2937'>{para_score}%</font>", card_value_large_style)
                ],
            ],
            [
                [
                    Paragraph("Classification", card_title_style),
                    Spacer(1, 4),
                    Paragraph(classification, card_value_small_style)
                ],
                [
                    Paragraph("Confidence Score", card_title_style),
                    Spacer(1, 4),
                    Paragraph(f"{report_data['average_confidence']}%", card_value_small_style)
                ],
                [
                    Paragraph("Flagged Chunks", card_title_style),
                    Spacer(1, 4),
                    Paragraph(f"{report_data['flagged_chunks_count']} / {len(report_data['chunks'])}", card_value_small_style)
                ],
            ]
        ]
        
        card_table = Table(card_data, colWidths=[170, 170, 170])
        card_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8F9FA')),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#E5E7EB')),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E5E7EB')),
            ('TOPPADDING', (0,0), (-1,-1), 10),
            ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ]))
        story.append(Paragraph("Similarity Summary", section_style))
        story.append(card_table)
        story.append(Spacer(1, 20))

        # 4. Recommendation Block
        rec_title_style = ParagraphStyle('RecTitle', fontName='Helvetica-Bold', fontSize=11, textColor=colors.HexColor('#1E1E2F'))
        rec_text_style = ParagraphStyle('RecText', parent=body_style, fontSize=9.5, leading=13)
        
        # Recommendations heuristics
        if sim_score >= 60:
            rec_content = "Critical Alert: High degree of similarity detected. The suspected document contains major direct copying or heavy paraphrasing from the original. Immediate review is recommended."
        elif sim_score >= 30:
            rec_content = "Warning: Moderate similarity detected. Significant sections of paraphrasing and light rewriting are present. Review required to ensure proper citation."
        elif sim_score >= 10:
            rec_content = "Notice: Minor similarity detected. Text overlaps are small, possibly common phrases or citations. Verify the flagged segments."
        else:
            rec_content = "Clean: The document shows negligible similarity to the original text. No plagiarized chunks detected."

        rec_table_data = [[
            Paragraph("System Recommendation:", rec_title_style),
            Paragraph(rec_content, rec_text_style)
        ]]
        rec_table = Table(rec_table_data, colWidths=[150, 360])
        rec_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#EBF5FF')),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#BFDBFE')),
            ('TOPPADDING', (0,0), (-1,-1), 8),
            ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ]))
        story.append(rec_table)
        story.append(Spacer(1, 20))
        story.append(PageBreak())

        # 5. Highlighted Similar Sections (Detailed Chunks)
        story.append(Paragraph("Flagged Chunk Details", section_style))
        
        flagged_count = 0
        for idx, chunk in enumerate(report_data["chunks"]):
            # Only list chunks with similarity >= 10% to keep report concise
            if chunk["semantic_similarity"] < 20:
                continue

            flagged_count += 1
            
            # Format the text with highlighting tags
            susp_highlighted = cls._highlight_text(chunk["suspected_text"], chunk["sentence_matches"], is_suspected=True)
            orig_highlighted = cls._highlight_text(chunk["original_text"], chunk["sentence_matches"], is_suspected=False)

            # Check if there is reference paper info
            paper_info = ""
            if "original_title" in chunk and chunk["original_title"] != "N/A":
                paper_info = f"<br/><b>Matched Paper:</b> {cls._escape(chunk['original_title'])} ({chunk.get('original_year', 'N/A')})"
                if chunk.get("original_authors") and chunk["original_authors"] != "N/A":
                    paper_info += f"<br/><b>Authors:</b> {cls._escape(chunk['original_authors'])}"
                if chunk.get("original_url"):
                    paper_info += f"<br/><b>Link:</b> <font color='blue'>{cls._escape(chunk['original_url'])}</font>"

            chunk_meta_data = [
                [
                    Paragraph(f"<b>Suspected Chunk {idx+1} (ID: {chunk['suspected_chunk_id']})</b>", meta_label_style),
                    Paragraph(f"<b>Similarity: {chunk['semantic_similarity']}%</b> | <b>Class: {chunk['classification']}</b> | <b>Confidence: {chunk['confidence']}%</b>", meta_label_style)
                ],
                [
                    Paragraph(f"<b>Reason:</b> {cls._escape(chunk['reason'])}{paper_info}", body_style),
                    ""
                ]
            ]
            
            chunk_meta_table = Table(chunk_meta_data, colWidths=[250, 260])
            chunk_meta_table.setStyle(TableStyle([
                ('SPAN', (0,1), (1,1)),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('LINEBELOW', (0,1), (-1,1), 0.5, colors.HexColor('#D1D1D6')),
                ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ]))

            text_side_by_side = [
                [
                    Paragraph("<b>Suspected Version:</b>", body_style),
                    Paragraph("<b>Original Version:</b>", body_style)
                ],
                [
                    Paragraph(susp_highlighted, highlight_style),
                    Paragraph(orig_highlighted, highlight_style)
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

            chunk_block = []
            chunk_block.append(chunk_meta_table)
            chunk_block.append(Spacer(1, 5))
            chunk_block.append(side_table)
            chunk_block.append(Spacer(1, 15))
            
            # Keep each block together so it doesn't break awkwardly across pages
            story.append(KeepTogether(chunk_block))

        if flagged_count == 0:
            story.append(Paragraph("No significant similarities detected in individual chunks.", body_style))

        # Build Document
        doc.build(story)
        logger.info("PDF report generated successfully.")

    @classmethod
    def generate_combined_pdf_report(cls, report_data: Dict[str, Any], output_path: Path) -> None:
        """
        Generate a combined PDF with Plagiarism Analysis + AI Writing Pattern Analysis sections.
        Falls back gracefully if ai_analysis key is missing.
        """
        logger.info(f"Generating combined PDF report at {output_path}")

        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=letter,
            rightMargin=54, leftMargin=54, topMargin=54, bottomMargin=54
        )
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle('CTTitle', parent=styles['Heading1'], fontSize=22, leading=26,
                                     textColor=colors.HexColor('#1E1E2F'), spaceAfter=12)
        section_style = ParagraphStyle('CTSection', parent=styles['Heading2'], fontSize=15, leading=19,
                                       textColor=colors.HexColor('#2A2D34'), spaceBefore=18, spaceAfter=10)
        sub_style = ParagraphStyle('CTSub', parent=styles['Heading3'], fontSize=12, leading=15,
                                   textColor=colors.HexColor('#374151'), spaceBefore=10, spaceAfter=6)
        body_style = ParagraphStyle('CTBody', parent=styles['Normal'], fontSize=10, leading=14,
                                    textColor=colors.HexColor('#4A4A4A'))
        label_style = ParagraphStyle('CTLabel', parent=body_style, fontName='Helvetica-Bold',
                                     textColor=colors.HexColor('#2E2E2E'))
        card_title_style = ParagraphStyle('CTCardTitle', parent=body_style, fontName='Helvetica-Bold',
                                          fontSize=9, leading=11, alignment=1, textColor=colors.HexColor('#6B7280'))
        card_val_lg = ParagraphStyle('CTCardValLg', parent=body_style, fontName='Helvetica-Bold',
                                     fontSize=20, leading=24, alignment=1)
        card_val_sm = ParagraphStyle('CTCardValSm', parent=body_style, fontName='Helvetica-Bold',
                                     fontSize=12, leading=15, alignment=1, textColor=colors.HexColor('#111827'))
        disclaimer_style = ParagraphStyle('CTDisclaim', parent=body_style, fontSize=9, leading=13,
                                          textColor=colors.HexColor('#6B7280'),
                                          backColor=colors.HexColor('#FEF9C3'),
                                          borderPadding=6)

        story = []
        meta = report_data.get("metadata", {})
        ai = report_data.get("ai_analysis", {})
        sim_score = report_data.get("overall_similarity", 0)
        ai_score = ai.get("overall_ai_score", 0)

        # ── Header ───────────────────────────────────────────────────────────
        story.append(Paragraph("PlagCheck AI — Combined Analysis Report", title_style))
        story.append(Paragraph(f"Generated: {meta.get('timestamp', time.strftime('%Y-%m-%d %H:%M:%S'))}", body_style))
        story.append(Spacer(1, 12))

        # ── Document Info ─────────────────────────────────────────────────────
        story.append(Paragraph("Document Information", section_style))
        doc_info = [
            [Paragraph("Field", label_style), Paragraph("Value", label_style)],
            [Paragraph("Filename", body_style), Paragraph(cls._escape(meta.get("suspected_filename", "N/A")), body_style)],
            [Paragraph("Word Count", body_style), Paragraph(str(meta.get("suspected_word_count", 0)), body_style)],
            [Paragraph("Document Chunks", body_style), Paragraph(str(meta.get("suspected_chunk_count", 0)), body_style)],
            [Paragraph("Search Query", body_style), Paragraph(cls._escape(meta.get("query", "N/A")), body_style)],
            [Paragraph("Reference Papers", body_style), Paragraph(str(meta.get("paper_count", 0)), body_style)],
        ]
        info_table = Table(doc_info, colWidths=[160, 350])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F2F2F7')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#D1D1D6')),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(info_table)
        story.append(Spacer(1, 18))

        # ═══════════════════════════════════════════════════════════════════
        # SECTION 1 — PLAGIARISM ANALYSIS
        # ═══════════════════════════════════════════════════════════════════
        story.append(Paragraph("═" * 60, body_style))
        story.append(Paragraph("PLAGIARISM ANALYSIS", section_style))

        sim_color = "#34C759" if sim_score < 30 else ("#FF9500" if sim_score < 60 else "#FF3B30")
        plag_cards = [[
            [Paragraph("Overall Similarity", card_title_style), Spacer(1, 4),
             Paragraph(f"<font color='{sim_color}'>{sim_score}%</font>", card_val_lg)],
            [Paragraph("Exact Copying", card_title_style), Spacer(1, 4),
             Paragraph(f"{report_data.get('overall_exact_copy', 0)}%", card_val_lg)],
            [Paragraph("Paraphrasing", card_title_style), Spacer(1, 4),
             Paragraph(f"{report_data.get('overall_paraphrase', 0)}%", card_val_lg)],
        ], [
            [Paragraph("Classification", card_title_style), Spacer(1, 4),
             Paragraph(report_data.get("overall_classification", "N/A"), card_val_sm)],
            [Paragraph("Avg Confidence", card_title_style), Spacer(1, 4),
             Paragraph(f"{report_data.get('average_confidence', 0)}%", card_val_sm)],
            [Paragraph("Flagged Chunks", card_title_style), Spacer(1, 4),
             Paragraph(f"{report_data.get('flagged_chunks_count', 0)} / {len(report_data.get('chunks', []))}", card_val_sm)],
        ]]
        pt = Table(plag_cards, colWidths=[170, 170, 170])
        pt.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8F9FA')),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#E5E7EB')),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E5E7EB')),
            ('TOPPADDING', (0, 0), (-1, -1), 9), ('BOTTOMPADDING', (0, 0), (-1, -1), 9),
        ]))
        story.append(pt)
        story.append(Spacer(1, 14))

        # Flagged chunks summary
        flagged = [c for c in report_data.get("chunks", []) if c.get("semantic_similarity", 0) >= 20]
        story.append(Paragraph(f"Flagged Sections: {len(flagged)}", sub_style))
        for idx, chunk in enumerate(flagged[:8]):  # Cap at 8 in combined report
            title_info = ""
            if chunk.get("original_title", "N/A") != "N/A":
                title_info = f" | Matched: {cls._escape(chunk['original_title'])}"
            story.append(Paragraph(
                f"<b>Chunk {idx + 1}</b> — Similarity: {chunk['semantic_similarity']}% "
                f"({chunk['classification']}){title_info}<br/>"
                f"<i>{cls._escape(chunk.get('reason', ''))[:200]}</i>",
                body_style
            ))
            story.append(Spacer(1, 5))

        story.append(PageBreak())

        # ═══════════════════════════════════════════════════════════════════
        # SECTION 2 — AI WRITING PATTERN ANALYSIS
        # ═══════════════════════════════════════════════════════════════════
        story.append(Paragraph("═" * 60, body_style))
        story.append(Paragraph("AI WRITING PATTERN ANALYSIS", section_style))

        if ai:
            ai_color = "#34C759" if ai_score <= 40 else ("#FF9500" if ai_score <= 70 else "#FF3B30")
            ai_chunks = ai.get("chunk_results", [])
            ai_cards = [[
                [Paragraph("AI Pattern Score", card_title_style), Spacer(1, 4),
                 Paragraph(f"<font color='{ai_color}'>{ai_score}%</font>", card_val_lg)],
                [Paragraph("Avg Confidence", card_title_style), Spacer(1, 4),
                 Paragraph(f"{ai.get('average_confidence', 0)}%", card_val_lg)],
                [Paragraph("Total Chunks", card_title_style), Spacer(1, 4),
                 Paragraph(str(ai.get("total_chunks", 0)), card_val_lg)],
            ], [
                [Paragraph("Very High Prob.", card_title_style), Spacer(1, 4),
                 Paragraph(str(ai.get("very_high_probability_chunks", 0)), card_val_sm)],
                [Paragraph("High Prob.", card_title_style), Spacer(1, 4),
                 Paragraph(str(ai.get("high_probability_chunks", 0)), card_val_sm)],
                [Paragraph("Moderate Prob.", card_title_style), Spacer(1, 4),
                 Paragraph(str(ai.get("moderate_probability_chunks", 0)), card_val_sm)],
            ]]
            at = Table(ai_cards, colWidths=[170, 170, 170])
            at.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8F9FA')),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#E5E7EB')),
                ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E5E7EB')),
                ('TOPPADDING', (0, 0), (-1, -1), 9), ('BOTTOMPADDING', (0, 0), (-1, -1), 9),
            ]))
            story.append(at)
            story.append(Spacer(1, 12))

            story.append(Paragraph(f"Overall Classification: {ai.get('overall_classification', 'N/A')}", sub_style))

            # Top features
            top_feats = ai.get("top_features", [])
            if top_feats:
                story.append(Paragraph("Most Detected Writing Features:", sub_style))
                feat_text = ", ".join(f"{f['feature']} ({f['count']}x)" for f in top_feats[:8])
                story.append(Paragraph(feat_text, body_style))
                story.append(Spacer(1, 10))

            # Highest/Lowest sections
            hc = ai.get("highest_chunk", {})
            lc = ai.get("lowest_chunk", {})
            if hc:
                story.append(Paragraph(
                    f"<b>Highest AI Pattern Section:</b> {hc.get('chunk_id', 'N/A')} "
                    f"— {hc.get('ai_probability', 0)}% ({hc.get('classification', '')})", body_style))
            if lc:
                story.append(Paragraph(
                    f"<b>Lowest AI Pattern Section:</b> {lc.get('chunk_id', 'N/A')} "
                    f"— {lc.get('ai_probability', 0)}% ({lc.get('classification', '')})", body_style))
            story.append(Spacer(1, 12))

            # Per-chunk highlights
            story.append(Paragraph("Highlighted AI Sections (Top 8):", sub_style))
            sorted_chunks = sorted(ai_chunks, key=lambda x: -x.get("ai_probability", 0))
            for cidx, cr in enumerate(sorted_chunks[:8]):
                prob = cr.get("ai_probability", 0)
                chunk_color = "#34C759" if prob <= 20 else ("#8BC34A" if prob <= 40 else
                              ("#FFC107" if prob <= 60 else ("#FF9800" if prob <= 80 else "#F44336")))
                feats = ", ".join(cr.get("features", [])[:4])
                story.append(Paragraph(
                    f"<font color='{chunk_color}'><b>{cr.get('chunk_id', '')}</b></font> "
                    f"— AI Probability: <b>{prob}%</b> | Confidence: {cr.get('confidence', 0)}% "
                    f"| {cr.get('classification', '')}<br/>"
                    f"<i>{cls._escape(cr.get('reason', '')[:200])}</i><br/>"
                    f"Features: {cls._escape(feats)}",
                    body_style
                ))
                story.append(Spacer(1, 6))

            # Disclaimer
            story.append(Spacer(1, 14))
            story.append(Paragraph(
                f"⚠ Disclaimer: {ai.get('disclaimer', '')}",
                disclaimer_style
            ))
        else:
            story.append(Paragraph("AI Writing Pattern Analysis was not performed for this report.", body_style))

        story.append(PageBreak())

        # ═══════════════════════════════════════════════════════════════════
        # SECTION 3 — COMBINED SUMMARY & RECOMMENDATIONS
        # ═══════════════════════════════════════════════════════════════════
        story.append(Paragraph("═" * 60, body_style))
        story.append(Paragraph("COMBINED DOCUMENT SUMMARY", section_style))

        combined_rows = [
            [Paragraph("Metric", label_style), Paragraph("Value", label_style)],
            [Paragraph("Plagiarism Similarity Score", body_style), Paragraph(f"{sim_score}%", body_style)],
            [Paragraph("Plagiarism Classification", body_style), Paragraph(report_data.get("overall_classification", "N/A"), body_style)],
            [Paragraph("Estimated AI Pattern Score", body_style), Paragraph(f"{ai_score}%", body_style)],
            [Paragraph("AI Classification", body_style), Paragraph(ai.get("overall_classification", "N/A"), body_style)],
            [Paragraph("Total Chunks Analysed", body_style), Paragraph(str(meta.get("suspected_chunk_count", 0)), body_style)],
            [Paragraph("Flagged Plagiarism Chunks", body_style), Paragraph(str(report_data.get("flagged_chunks_count", 0)), body_style)],
            [Paragraph("High AI Pattern Chunks", body_style), Paragraph(str(ai.get("high_probability_chunks", 0) + ai.get("very_high_probability_chunks", 0)), body_style)],
        ]
        cs_table = Table(combined_rows, colWidths=[280, 230])
        cs_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F2F2F7')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#D1D1D6')),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(cs_table)
        story.append(Spacer(1, 14))

        # Recommendation
        if sim_score >= 60:
            rec = "Critical: High plagiarism detected. Review and cite all matched sources immediately."
        elif sim_score >= 30:
            rec = "Warning: Moderate similarity found. Ensure all paraphrased sections are properly attributed."
        else:
            rec = "Plagiarism: The document appears largely original relative to the searched literature."

        if ai_score >= 70:
            ai_rec = " AI Pattern: Very high AI writing patterns detected. Manual review of highlighted sections is strongly recommended."
        elif ai_score >= 40:
            ai_rec = " AI Pattern: Moderate AI writing patterns present. Some sections may have been AI-assisted."
        else:
            ai_rec = " AI Pattern: Writing style appears predominantly human-authored."

        rec_data = [[Paragraph("Recommendation:", label_style), Paragraph(rec + ai_rec, body_style)]]
        rec_table = Table(rec_data, colWidths=[130, 380])
        rec_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#EBF5FF')),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#BFDBFE')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(rec_table)

        doc.build(story)
        logger.info("Combined PDF report generated successfully.")

    @classmethod
    def generate_ai_pdf_report(cls, ai_data: Dict[str, Any], output_path: Path) -> None:
        """Generate a standalone PDF for AI writing pattern analysis results."""
        logger.info(f"Generating AI-only PDF at {output_path}")

        doc = SimpleDocTemplate(str(output_path), pagesize=letter,
                                rightMargin=54, leftMargin=54, topMargin=54, bottomMargin=54)
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('AITitle', parent=styles['Heading1'], fontSize=22, leading=26,
                                     textColor=colors.HexColor('#1E1E2F'), spaceAfter=12)
        section_style = ParagraphStyle('AISection', parent=styles['Heading2'], fontSize=15, leading=19,
                                       textColor=colors.HexColor('#2A2D34'), spaceBefore=16, spaceAfter=8)
        body_style = ParagraphStyle('AIBody', parent=styles['Normal'], fontSize=10, leading=14,
                                    textColor=colors.HexColor('#4A4A4A'))
        label_style = ParagraphStyle('AILabel', parent=body_style, fontName='Helvetica-Bold')
        card_title_style = ParagraphStyle('AICardTitle', parent=body_style, fontName='Helvetica-Bold',
                                          fontSize=9, leading=11, alignment=1, textColor=colors.HexColor('#6B7280'))
        card_val_lg = ParagraphStyle('AICardValLg', parent=body_style, fontName='Helvetica-Bold',
                                     fontSize=20, leading=24, alignment=1)

        story = []
        ai_score = ai_data.get("overall_ai_score", 0)
        meta = ai_data.get("metadata", {})
        chunk_results = ai_data.get("chunk_results", [])

        story.append(Paragraph("AI Writing Pattern Analysis Report", title_style))
        story.append(Paragraph(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", body_style))
        if meta:
            story.append(Paragraph(
                f"Document: {cls._escape(meta.get('filename', 'N/A'))} | "
                f"Words: {meta.get('word_count', 0)} | Chunks: {meta.get('chunk_count', 0)}",
                body_style))
        story.append(Spacer(1, 14))

        ai_color = "#34C759" if ai_score <= 40 else ("#FF9500" if ai_score <= 70 else "#FF3B30")
        kpi_data = [[
            [Paragraph("Overall AI Score", card_title_style), Spacer(1, 4),
             Paragraph(f"<font color='{ai_color}'>{ai_score}%</font>", card_val_lg)],
            [Paragraph("Avg Confidence", card_title_style), Spacer(1, 4),
             Paragraph(f"{ai_data.get('average_confidence', 0)}%", card_val_lg)],
            [Paragraph("Total Chunks", card_title_style), Spacer(1, 4),
             Paragraph(str(ai_data.get("total_chunks", 0)), card_val_lg)],
        ]]
        kt = Table(kpi_data, colWidths=[170, 170, 170])
        kt.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8F9FA')),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#E5E7EB')),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E5E7EB')),
            ('TOPPADDING', (0, 0), (-1, -1), 9), ('BOTTOMPADDING', (0, 0), (-1, -1), 9),
        ]))
        story.append(kt)
        story.append(Spacer(1, 12))

        story.append(Paragraph(f"Classification: {ai_data.get('overall_classification', 'N/A')}", section_style))

        # Per-chunk table
        if chunk_results:
            story.append(Paragraph("Chunk-level Results", section_style))
            table_data = [[
                Paragraph("Chunk", label_style),
                Paragraph("AI Prob.", label_style),
                Paragraph("Confidence", label_style),
                Paragraph("Classification", label_style),
            ]]
            for cr in chunk_results:
                prob = cr.get("ai_probability", 0)
                chunk_color = ("#34C759" if prob <= 20 else "#8BC34A" if prob <= 40 else
                               "#FFC107" if prob <= 60 else "#FF9800" if prob <= 80 else "#F44336")
                table_data.append([
                    Paragraph(cr.get("chunk_id", ""), body_style),
                    Paragraph(f"<font color='{chunk_color}'>{prob}%</font>", body_style),
                    Paragraph(f"{cr.get('confidence', 0)}%", body_style),
                    Paragraph(cr.get("classification", ""), body_style),
                ])
            ct = Table(table_data, colWidths=[100, 80, 80, 250])
            ct.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F2F2F7')),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#D1D1D6')),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ]))
            story.append(ct)
            story.append(Spacer(1, 12))

        # Disclaimer
        story.append(Paragraph(
            f"⚠ Disclaimer: {ai_data.get('disclaimer', '')}",
            ParagraphStyle('Disc', parent=body_style, fontSize=9, leading=13,
                           textColor=colors.HexColor('#6B7280'))
        ))

        doc.build(story)
        logger.info("AI-only PDF report generated.")
