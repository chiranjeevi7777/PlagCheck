"""
Linguistic Metrics and Writing Analytics.

Computes readability scores, vocabulary diversity, passive voice,
sentence length distributions, and repetition metrics.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Set

# Lightweight stop words list for lexical density
_STOP_WORDS: Set[str] = {
    "the", "and", "that", "for", "with", "this", "from", "are", "was", "were",
    "have", "has", "had", "been", "they", "their", "there", "then", "about",
    "would", "could", "should", "your", "them", "some", "other", "into",
    "than", "its", "also", "these", "those", "such", "only", "over", "more",
    "a", "an", "in", "on", "at", "to", "of", "or", "as", "by", "it", "is", "be"
}

# Syllables estimation vowels pattern
_VOWELS = re.compile(r'[aeiouy]+', re.IGNORECASE)


def count_syllables_word(word: str) -> int:
    """Estimate syllable count in a single word using heuristic rules."""
    word = word.lower().strip()
    if not word:
        return 0
    # Clean non-alphabetic
    word = re.sub(r'[^a-z]', '', word)
    if not word:
        return 0
    
    # Exceptions
    if word.endswith('e'):
        # Silent e unless word is very short
        if word.endswith('le') and len(word) > 2 and word[-3] not in 'aeiou':
            pass  # -le counts as a syllable (e.g. table, candle)
        else:
            word = word[:-1]
            
    # Count vowel groups
    vowel_groups = _VOWELS.findall(word)
    count = len(vowel_groups)
    
    # Ad-hoc adjustments
    if word.endswith('es') or word.endswith('ed'):
        # usually silent e in plural/past tense
        if count > 1:
            count -= 1
            
    # Guarantee at least 1 syllable
    return max(1, count)


def detect_passive_voice(text: str) -> int:
    """
    Count instances of passive voice constructions.
    Looks for forms of 'to be' followed by past participles.
    """
    be_verbs = r"\b(am|is|are|was|were|be|been|being)\b"
    # Match words ending in 'ed' or common irregular past participles
    past_participles = (
        r"\s+(?:[a-z]+ed|done|written|given|seen|taken|chosen|known|shown|held|"
        r"made|built|kept|run|said|told|heard|begun|broken|brought|cut|drawn|"
        r"drunk|eaten|fallen|found|forgotten|frozen|gone|grown|lost|met|paid|"
        r"read|sent|shaken|spent|spread|understood|won|worn)\b"
    )
    pattern = be_verbs + past_participles
    matches = re.findall(pattern, text, re.IGNORECASE)
    return len(matches)


def get_repeated_phrases(text: str, n: int = 3, min_count: int = 2) -> Dict[str, int]:
    """Find repeated n-grams in the text."""
    words = re.findall(r"\b[a-z]{3,}\b", text.lower())
    if len(words) < n:
        return {}
    
    ngrams: Dict[str, int] = {}
    for i in range(len(words) - n + 1):
        gram = " ".join(words[i:i+n])
        ngrams[gram] = ngrams.get(gram, 0) + 1
        
    return {k: v for k, v in ngrams.items() if v >= min_count}


class WritingAnalytics:
    """Engine to compute exhaustive readability and lexical metrics for a document or paragraph."""

    @staticmethod
    def analyze(text: str) -> Dict[str, Any]:
        """Compute metrics for the given text segment."""
        # Sanitise line endings
        text_clean = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Split sentences
        # Simple rule-based splitter to avoid external dependencies
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text_clean) if s.strip()]
        if not sentences:
            sentences = [text_clean] if text_clean.strip() else []
            
        # Split words
        words = re.findall(r"\b[a-zA-Z']+\b", text_clean)
        words_lower = [w.lower() for w in words]
        
        word_count = len(words)
        sentence_count = len(sentences)
        character_count = sum(len(w) for w in words)
        
        if word_count == 0 or sentence_count == 0:
            return {
                "word_count": word_count,
                "sentence_count": sentence_count,
                "readability": {
                    "flesch_reading_ease": 0.0,
                    "flesch_kincaid_grade": 0.0,
                    "gunning_fog": 0.0,
                    "smog_index": 0.0,
                    "coleman_liau_index": 0.0,
                    "automated_readability_index": 0.0
                },
                "lexical": {
                    "lexical_density": 0.0,
                    "type_token_ratio": 0.0,
                    "passive_voice_percentage": 0.0,
                    "repeated_phrases": {}
                },
                "sentence_stats": {
                    "avg_length": 0.0,
                    "std_dev": 0.0
                }
            }
            
        # Syllables and Complex Words
        syllable_counts = [count_syllables_word(w) for w in words]
        total_syllables = sum(syllable_counts)
        complex_words = sum(1 for c in syllable_counts if c >= 3)
        
        # Readability Indexes
        # Flesch Reading Ease
        fre = 206.835 - 1.015 * (word_count / sentence_count) - 84.6 * (total_syllables / word_count)
        fre = max(0.0, min(100.0, fre))
        
        # Flesch-Kincaid Grade
        fkg = 0.39 * (word_count / sentence_count) + 11.8 * (total_syllables / word_count) - 15.59
        fkg = max(0.0, fkg)
        
        # Gunning Fog
        complex_pct = (complex_words / word_count) * 100.0
        gfi = 0.4 * ((word_count / sentence_count) + complex_pct)
        gfi = max(0.0, gfi)
        
        # SMOG Index
        if sentence_count >= 1:
            smog = 1.0430 * math.sqrt(complex_words * (30 / sentence_count)) + 3.1291
        else:
            smog = 0.0
            
        # Coleman-Liau
        # L = average letters per 100 words
        # S = average sentences per 100 words
        L = (character_count / word_count) * 100.0
        S = (sentence_count / word_count) * 100.0
        cli = 0.0588 * L - 0.296 * S - 15.8
        cli = max(0.0, cli)
        
        # Automated Readability Index (ARI)
        ari = 4.71 * (character_count / word_count) + 0.5 * (word_count / sentence_count) - 21.43
        ari = max(0.0, ari)
        
        # Lexical density
        content_words = [w for w in words_lower if w not in _STOP_WORDS]
        lexical_density = len(content_words) / word_count if word_count > 0 else 0.0
        
        # Vocabulary Diversity (TTR)
        ttr = len(set(words_lower)) / word_count if word_count > 0 else 0.0
        
        # Passive Voice
        passive_count = detect_passive_voice(text_clean)
        passive_pct = (passive_count / sentence_count) * 100.0
        
        # Sentence stats
        sent_lengths = [len(s.split()) for s in sentences]
        avg_sent_len = sum(sent_lengths) / len(sent_lengths)
        variance = sum((x - avg_sent_len) ** 2 for x in sent_lengths) / len(sent_lengths)
        std_dev = math.sqrt(variance)
        
        # Repetition check (3-grams)
        repeated = get_repeated_phrases(text_clean, n=3, min_count=2)
        
        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "readability": {
                "flesch_reading_ease": round(fre, 2),
                "flesch_kincaid_grade": round(fkg, 2),
                "gunning_fog": round(gfi, 2),
                "smog_index": round(smog, 2),
                "coleman_liau_index": round(cli, 2),
                "automated_readability_index": round(ari, 2)
            },
            "lexical": {
                "lexical_density": round(lexical_density, 3),
                "type_token_ratio": round(ttr, 3),
                "passive_voice_percentage": round(passive_pct, 1),
                "repeated_phrases": repeated
            },
            "sentence_stats": {
                "avg_length": round(avg_sent_len, 2),
                "std_dev": round(std_dev, 2)
            }
        }


class DocumentMetricsCalculator:
    """Compatibility wrapper for WritingAnalytics."""

    @staticmethod
    def calculate_all_metrics(text: str) -> Dict[str, Any]:
        """Compute all readability and lexical metrics."""
        return WritingAnalytics.analyze(text)

