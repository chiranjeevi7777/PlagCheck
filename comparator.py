import re
from typing import List, Dict, Any, Callable, Optional
from groq_client import GroqPlagiarismClient, ChunkComparisonResult
from utils import logger

class PlagiarismComparator:
    """Coordinates the document comparison using a sliding-window chunk alignment strategy."""

    def __init__(self, groq_client: GroqPlagiarismClient):
        self.groq_client = groq_client

    def _clean_and_tokenize(self, text: str) -> set:
        """Extract unique content words from text for lexical comparison."""
        # Find all words of length 3 or more (alphanumeric)
        words = re.findall(r'\b[a-z0-9]{3,}\b', text.lower())
        # A standard set of common stop words to filter out
        stop_words = {
            "the", "and", "that", "for", "with", "this", "from", "are", "was", "were", 
            "have", "has", "had", "been", "they", "their", "there", "then", "about", 
            "would", "could", "should", "your", "them", "some", "other", "into", 
            "than", "its", "also", "these", "those", "such", "only", "over", "more", 
            "most", "both", "each", "under", "between", "through", "during", "before", 
            "after", "above", "below", "did", "does", "doesnt", "didnt", "hasnt", 
            "havent", "hadnt", "arent", "isnt", "wasnt", "werent"
        }
        return set(w for w in words if w not in stop_words)

    def compare_documents(
        self, 
        original_chunks: List[Dict[str, Any]], 
        suspected_chunks: List[Dict[str, Any]], 
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Compares suspected chunks against original chunks using a sliding window.
        
        For each suspected chunk S_i:
          1. Calculate matching original chunk index j = int(i * len(O) / len(S)).
          2. Compare S_i with O_{j-1}, O_j, O_{j+1} (bounded by index range).
          3. Select the best match (highest semantic similarity) as the final result for S_i.
        """
        N = len(suspected_chunks)
        M = len(original_chunks)

        if N == 0 or M == 0:
            logger.warning("Empty chunks list received.")
            return []

        # 1. Plan comparison tasks
        comparison_tasks = []
        for i in range(N):
            # Mapped index in original chunks
            j = int(i * M / N)
            
            # Candidate indices in original chunks
            candidates = {j - 1, j, j + 1}
            # Keep only valid indices in original doc
            valid_candidates = [idx for idx in candidates if 0 <= idx < M]
            
            for o_idx in valid_candidates:
                comparison_tasks.append((i, o_idx))

        total_tasks = len(comparison_tasks)
        logger.info(f"Planned {total_tasks} pairwise comparisons (Suspected Chunks: {N}, Original Chunks: {M}).")

        # 2. Run comparisons
        all_results = {}  # Map: suspected_idx -> List of results
        for idx, (s_idx, o_idx) in enumerate(comparison_tasks):
            s_chunk = suspected_chunks[s_idx]
            o_chunk = original_chunks[o_idx]
            
            status_msg = f"Comparing Suspected Chunk {s_idx + 1}/{N} against Original Chunk {o_idx + 1}/{M}..."
            logger.info(f"[{idx + 1}/{total_tasks}] {status_msg}")
            
            if progress_callback:
                progress_callback(idx + 1, total_tasks, status_msg)

            try:
                # Call Groq API
                comp_result: ChunkComparisonResult = self.groq_client.compare_chunks(
                    original_text=o_chunk["text"],
                    suspected_text=s_chunk["text"]
                )
                
                result_data = {
                    "original_chunk_id": o_chunk["id"],
                    "original_text": o_chunk["text"],
                    "original_idx": o_idx,
                    "semantic_similarity": comp_result.semantic_similarity,
                    "exact_copy": comp_result.exact_copy,
                    "paraphrase": comp_result.paraphrase,
                    "classification": comp_result.classification,
                    "confidence": comp_result.confidence,
                    "reason": comp_result.reason,
                    "sentence_matches": [m.model_dump() for m in comp_result.sentence_matches]
                }
            except Exception as e:
                logger.error(f"Error in comparison task ({s_idx}, {o_idx}): {e}")
                # Log a default zero-similarity entry if it fails to avoid breaking the entire run
                result_data = {
                    "original_chunk_id": o_chunk["id"],
                    "original_text": o_chunk["text"],
                    "original_idx": o_idx,
                    "semantic_similarity": 0,
                    "exact_copy": 0,
                    "paraphrase": 0,
                    "classification": "Original",
                    "confidence": 0,
                    "reason": f"API Error during analysis: {str(e)}",
                    "sentence_matches": []
                }
            
            if s_idx not in all_results:
                all_results[s_idx] = []
            all_results[s_idx].append(result_data)

        # 3. Aggregate best matches for each suspected chunk
        final_aligned_results = []
        for s_idx in range(N):
            s_chunk = suspected_chunks[s_idx]
            results_for_chunk = all_results.get(s_idx, [])
            
            if not results_for_chunk:
                # Fallback if no task succeeded
                best_match = {
                    "suspected_chunk_id": s_chunk["id"],
                    "suspected_text": s_chunk["text"],
                    "original_chunk_id": "N/A",
                    "original_text": "N/A",
                    "original_idx": -1,
                    "semantic_similarity": 0,
                    "exact_copy": 0,
                    "paraphrase": 0,
                    "classification": "Original",
                    "confidence": 0,
                    "reason": "No successful comparison performed.",
                    "sentence_matches": []
                }
            else:
                # Find the comparison with highest semantic similarity
                best_match_raw = max(results_for_chunk, key=lambda x: x["semantic_similarity"])
                best_match = {
                    "suspected_chunk_id": s_chunk["id"],
                    "suspected_text": s_chunk["text"],
                    "original_chunk_id": best_match_raw["original_chunk_id"],
                    "original_text": best_match_raw["original_text"],
                    "original_idx": best_match_raw["original_idx"],
                    "semantic_similarity": best_match_raw["semantic_similarity"],
                    "exact_copy": best_match_raw["exact_copy"],
                    "paraphrase": best_match_raw["paraphrase"],
                    "classification": best_match_raw["classification"],
                    "confidence": best_match_raw["confidence"],
                    "reason": best_match_raw["reason"],
                    "sentence_matches": best_match_raw["sentence_matches"]
                }
            
            final_aligned_results.append(best_match)

        return final_aligned_results

    def compare_against_papers(
        self, 
        suspected_chunks: List[Dict[str, Any]], 
        papers: List[Dict[str, Any]], 
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Compares suspected chunks against 5 paper abstracts.
        First screens each suspected chunk to find the single best-matching paper (if any).
        Then does the detailed pairwise sentence-level comparison for matches.
        """
        N = len(suspected_chunks)
        if N == 0 or not papers:
            logger.warning("Empty chunks or papers received.")
            return []

        # Convert papers to original chunks
        original_chunks = []
        paper_word_sets = []
        for p_idx, paper in enumerate(papers):
            author_names = ", ".join([a.get("name", "") for a in paper.get("authors", [])]) if paper.get("authors") else "Unknown Author"
            original_chunks.append({
                "id": f"paper_{p_idx}",
                "text": paper["abstract"],
                "title": paper["title"],
                "authors": author_names,
                "url": paper.get("url", ""),
                "year": paper.get("year", "N/A"),
                "word_count": len(paper["abstract"].split())
            })
            paper_word_sets.append(self._clean_and_tokenize(paper["abstract"]))

        final_aligned_results = []

        for idx, s_chunk in enumerate(suspected_chunks):
            status_msg = f"Screening Suspected Chunk {idx + 1}/{N} against reference papers..."
            logger.info(status_msg)
            
            if progress_callback:
                progress_callback(idx + 1, N, status_msg)

            # Pre-filter: calculate lexical content word overlap
            s_words = self._clean_and_tokenize(s_chunk["text"])
            max_overlap = 0
            for p_words in paper_word_sets:
                overlap = len(s_words & p_words)
                if overlap > max_overlap:
                    max_overlap = overlap

            logger.info(f"Chunk {idx + 1}/{N} max content word overlap: {max_overlap} words.")

            if max_overlap < 3:
                logger.info(f"Skipping LLM screening for Chunk {idx + 1}/{N} due to low content overlap ({max_overlap} < 3).")
                best_match = {
                    "suspected_chunk_id": s_chunk["id"],
                    "suspected_text": s_chunk["text"],
                    "original_chunk_id": "N/A",
                    "original_text": "N/A",
                    "original_idx": -1,
                    "original_title": "N/A",
                    "original_authors": "N/A",
                    "original_url": "",
                    "original_year": "N/A",
                    "semantic_similarity": 0,
                    "exact_copy": 0,
                    "paraphrase": 0,
                    "classification": "Original",
                    "confidence": 100,
                    "reason": "This chunk does not show similarity to any of the retrieved reference papers (insufficient lexical overlap).",
                    "sentence_matches": []
                }
                final_aligned_results.append(best_match)
                continue

            # Step 1: Screen to find if it matches any of the papers
            matched_idx = self.groq_client.find_best_matching_paper(s_chunk["text"], papers)

            if matched_idx >= 0 and matched_idx < len(original_chunks):
                # Step 2: Perform detailed comparison
                o_chunk = original_chunks[matched_idx]
                logger.info(f"Detailed comparison: Suspected Chunk {idx + 1} with Paper Index {matched_idx}")
                try:
                    comp_result = self.groq_client.compare_chunks(
                        original_text=o_chunk["text"],
                        suspected_text=s_chunk["text"]
                    )
                    best_match = {
                        "suspected_chunk_id": s_chunk["id"],
                        "suspected_text": s_chunk["text"],
                        "original_chunk_id": o_chunk["id"],
                        "original_text": o_chunk["text"],
                        "original_idx": matched_idx,
                        "original_title": o_chunk["title"],
                        "original_authors": o_chunk["authors"],
                        "original_url": o_chunk["url"],
                        "original_year": o_chunk["year"],
                        "semantic_similarity": comp_result.semantic_similarity,
                        "exact_copy": comp_result.exact_copy,
                        "paraphrase": comp_result.paraphrase,
                        "classification": comp_result.classification,
                        "confidence": comp_result.confidence,
                        "reason": comp_result.reason,
                        "sentence_matches": [m.model_dump() for m in comp_result.sentence_matches]
                    }
                except Exception as e:
                    logger.error(f"Error comparing chunk {idx} to paper {matched_idx}: {e}")
                    best_match = {
                        "suspected_chunk_id": s_chunk["id"],
                        "suspected_text": s_chunk["text"],
                        "original_chunk_id": o_chunk["id"],
                        "original_text": o_chunk["text"],
                        "original_idx": matched_idx,
                        "original_title": o_chunk["title"],
                        "original_authors": o_chunk["authors"],
                        "original_url": o_chunk["url"],
                        "original_year": o_chunk["year"],
                        "semantic_similarity": 0,
                        "exact_copy": 0,
                        "paraphrase": 0,
                        "classification": "Original",
                        "confidence": 0,
                        "reason": f"Error running detailed comparison: {e}",
                        "sentence_matches": []
                    }
            else:
                # No match found - original content
                best_match = {
                    "suspected_chunk_id": s_chunk["id"],
                    "suspected_text": s_chunk["text"],
                    "original_chunk_id": "N/A",
                    "original_text": "N/A",
                    "original_idx": -1,
                    "original_title": "N/A",
                    "original_authors": "N/A",
                    "original_url": "",
                    "original_year": "N/A",
                    "semantic_similarity": 0,
                    "exact_copy": 0,
                    "paraphrase": 0,
                    "classification": "Original",
                    "confidence": 100,
                    "reason": "This chunk does not show similarity to any of the retrieved reference papers.",
                    "sentence_matches": []
                }
            final_aligned_results.append(best_match)

        return final_aligned_results
