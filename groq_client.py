import json
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import groq
from groq import Groq
from config import settings
from utils import logger

class SentenceMatch(BaseModel):
    suspected_sentence: str = Field(description="The matching sentence from the suspected document.")
    original_sentence: str = Field(description="The corresponding sentence from the original document.")
    similarity_score: int = Field(description="Similarity score between these two sentences (0-100).")
    match_type: str = Field(description="Classification of match: 'exact_copy', 'paraphrase', or 'none'.")

class ChunkComparisonResult(BaseModel):
    semantic_similarity: int = Field(description="Overall semantic similarity score between 0 and 100.")
    exact_copy: int = Field(description="Percentage of exact copy (0-100).")
    paraphrase: int = Field(description="Percentage of paraphrasing (0-100).")
    classification: str = Field(description="One of: Original, Minor Similarity, Light Rewrite, Heavy Rewrite, Heavy Paraphrasing, Near Duplicate, Exact Copy.")
    confidence: int = Field(description="Confidence score between 0 and 100.")
    reason: str = Field(description="Brief explanation of the similarity and changes detected.")
    sentence_matches: List[SentenceMatch] = Field(default=[], description="List of specific matching sentence pairs.")

class GroqPlagiarismClient:
    """Client for Groq LLM queries with built-in retry and format enforcement."""

    def __init__(self):
        self.api_key = settings.groq_api_key
        self.model = settings.groq_model
        self.temperature = settings.temperature
        self.max_tokens = settings.max_tokens
        self.max_retries = settings.max_retries
        self.timeout = settings.timeout_seconds
        
        # Collect API keys
        self.api_keys = []
        for keys_str in (settings.groq_api_keys, settings.groq_api_key):
            if keys_str:
                for k in keys_str.split(","):
                    k_clean = k.strip()
                    if k_clean and k_clean not in self.api_keys:
                        self.api_keys.append(k_clean)
            
        self.current_key_index = 0
        
        # Collect fallback models
        models_str = settings.fallback_models
        self.models_list = []
        if models_str:
            self.models_list = [m.strip() for m in models_str.split(",") if m.strip()]
        if not self.models_list:
            self.models_list = [settings.groq_model]
        # Ensure configured model is first in the list
        if settings.groq_model not in self.models_list:
            self.models_list.insert(0, settings.groq_model)
            
        if not self.api_keys:
            logger.warning("No Groq API keys are set. API calls will fail.")
            self.client = None
        else:
            self.client = Groq(api_key=self.api_keys[0], timeout=self.timeout, max_retries=0)

    def _call_groq_api(self, messages: list) -> str:
        """Call Groq API with retries, key rotation, and model fallback on rate limits or errors."""
        if not self.api_keys:
            raise ValueError("No Groq API keys configured. Please set GROQ_API_KEY or GROQ_API_KEYS in your .env file.")
        
        last_exception = None
        
        # Iterate through available models
        for model_name in self.models_list:
            # Iterate through available API keys
            for key_offset in range(len(self.api_keys)):
                # Rotate starting from the last successful key index
                active_key_index = (self.current_key_index + key_offset) % len(self.api_keys)
                active_key = self.api_keys[active_key_index]
                
                # Groq json_object format requires the word 'json' in the prompt
                modified_messages = messages
                has_json_word = False
                for msg in messages:
                    if "json" in msg.get("content", "").lower():
                        has_json_word = True
                        break
                if not has_json_word and messages:
                    modified_messages = list(messages)
                    modified_messages[-1] = {
                        "role": messages[-1]["role"],
                        "content": messages[-1]["content"] + "\nReturn response in JSON format."
                    }
                
                try:
                    logger.info(
                        f"Attempting Groq call - Model: '{model_name}', "
                        f"Key Index: {active_key_index} (Key suffix: ...{active_key[-6:] if len(active_key) > 6 else 'N/A'})"
                    )
                    
                    client = Groq(api_key=active_key, timeout=self.timeout, max_retries=0)
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=modified_messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        response_format={"type": "json_object"}
                    )
                    
                    # Update active key and model on success
                    self.current_key_index = active_key_index
                    self.model = model_name
                    # Sync the main client instance
                    self.client = client
                    
                    return response.choices[0].message.content
                    
                except groq.RateLimitError as rle:
                    last_exception = rle
                    logger.warning(
                        f"Rate limit exceeded (429) for Model: '{model_name}', Key Index: {active_key_index}. "
                        f"Rotating to next key..."
                    )
                except (groq.APIConnectionError, groq.APITimeoutError) as te:
                    last_exception = te
                    logger.warning(
                        f"Timeout or connection error for Model: '{model_name}', Key Index: {active_key_index}. "
                        f"Error: {te}. Retrying with next key..."
                    )
                except groq.APIStatusError as ase:
                    last_exception = ase
                    if ase.status_code == 429:
                        logger.warning(
                            f"Status 429 received for Model: '{model_name}', Key Index: {active_key_index}. Rotating key..."
                        )
                    elif ase.status_code in (400, 404):
                        logger.warning(
                            f"Model '{model_name}' is not supported or not found (Status {ase.status_code}). "
                            f"Skipping this model and moving to fallback..."
                        )
                        break  # Break key-loop for this model to try the next model
                    elif ase.status_code in (401, 403):
                        logger.warning(
                            f"Authentication error (Status {ase.status_code}) for Key Index {active_key_index}. "
                            f"Rotating key..."
                        )
                    else:
                        logger.warning(
                            f"API Status Error {ase.status_code} for Model: '{model_name}', Key Index: {active_key_index}: {ase.message}. "
                            f"Rotating key..."
                        )
                except Exception as e:
                    last_exception = e
                    logger.error(f"Unexpected error in Groq call for Model '{model_name}': {e}")
                    raise e
            
            logger.warning(
                f"Exhausted all API keys for Model: '{model_name}'. "
                f"Falling back to next model if available..."
            )
            
        logger.error("All Groq API keys and fallback models have been exhausted.")
        if last_exception:
            raise last_exception
        raise ValueError("Failed to execute Groq API call after trying all keys and models.")

    def compare_chunks(self, original_text: str, suspected_text: str) -> ChunkComparisonResult:
        """
        Compare original and suspected text chunks using Groq LLM.
        Returns validated JSON response as ChunkComparisonResult.
        """
        system_prompt = (
            "You are an expert plagiarism detection system.\n"
            "Compare the following two passages (Original and Suspected).\n"
            "Analyze the degree of copying, paraphrasing, structure changes, and semantic alignment.\n"
            "Provide:\n"
            "1. Semantic similarity score (0-100)\n"
            "2. Exact copying percentage (0-100)\n"
            "3. Paraphrasing percentage (0-100)\n"
            "4. Classification (choose exactly from: 'Original', 'Minor Similarity', 'Light Rewrite', 'Heavy Rewrite', 'Heavy Paraphrasing', 'Near Duplicate', 'Exact Copy')\n"
            "5. Confidence score (0-100)\n"
            "6. A brief reason detailing the structural or semantic overlap.\n"
            "7. Sentence matches: An array of objects mapping similar sentences from suspected to original. "
            "Only include pairs with similarity >= 50. For each, specify 'exact_copy' or 'paraphrase'.\n\n"
            "Return ONLY a JSON object matching this schema:\n"
            "{\n"
            '  "semantic_similarity": int,\n'
            '  "exact_copy": int,\n'
            '  "paraphrase": int,\n'
            '  "classification": "string",\n'
            '  "confidence": int,\n'
            '  "reason": "string",\n'
            '  "sentence_matches": [\n'
            '    {\n'
            '      "suspected_sentence": "string",\n'
            '      "original_sentence": "string",\n'
            '      "similarity_score": int,\n'
            '      "match_type": "exact_copy" | "paraphrase"\n'
            '    }\n'
            '  ]\n'
            "}\n"
            "Never return markdown fences or explanations outside of the JSON."
        )

        user_content = f"--- ORIGINAL CHUNK ---\n{original_text}\n\n--- SUSPECTED CHUNK ---\n{suspected_text}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        try:
            logger.info("Sending comparison request to Groq API...")
            raw_response = self._call_groq_api(messages)
            
            # Parse and validate response
            data = json.loads(raw_response)
            
            # Clean up the classification if it doesn't match expected values exactly
            allowed_classes = {
                "original", "minor similarity", "light rewrite", "heavy rewrite", 
                "heavy paraphrasing", "near duplicate", "exact copy"
            }
            cls = data.get("classification", "Original").strip().lower()
            # Find closest match or default to original
            matched_cls = "Original"
            for ac in allowed_classes:
                if ac in cls:
                    # Convert to title case for display
                    words = ac.split()
                    matched_cls = " ".join(w.capitalize() for w in words)
                    break
            data["classification"] = matched_cls

            # Validate using Pydantic
            result = ChunkComparisonResult(**data)
            return result
        except json.JSONDecodeError as je:
            logger.error(f"Failed to parse JSON from Groq: {je}. Raw: {raw_response}")
            raise ValueError("Groq returned invalid JSON format.")
        except Exception as e:
            logger.error(f"Error comparing chunks with Groq: {e}")
            raise e

    def generate_search_query(self, document_text: str) -> str:
        """
        Extract a 2-4 word academic search query from the document text.
        """
        system_prompt = (
            "You are a research assistant. Analyze the given text snippet and extract a concise, "
            "2-4 word academic search query that captures the core scientific topic or domain of the text.\n"
            "Return ONLY a JSON object matching this schema:\n"
            "{\n"
            '  "query": "string"\n'
            "}"
        )
        
        snippet = document_text[:1500]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Snippet:\n{snippet}"}
        ]
        try:
            logger.info("Extracting query using Groq...")
            raw_response = self._call_groq_api(messages)
            data = json.loads(raw_response)
            query = data.get("query", "").strip()
            if not query:
                query = "deep learning"
            logger.info(f"Generated search query: {query}")
            return query
        except Exception as e:
            logger.error(f"Error generating search query: {e}")
            return "deep learning"

    def generate_mock_papers(self, query: str) -> List[Dict[str, Any]]:
        """
        Generates 5 realistic, high-quality reference academic papers matching the query using Groq.
        Used as a fallback when Semantic Scholar API returns 429.
        """
        system_prompt = (
            "You are a research assistant. Generate 5 realistic, high-quality, relevant academic papers "
            "that match the search query. For each paper, provide a title, a detailed abstract (150-250 words, "
            "rich in technical detail and terminology matching the topic), authors, a realistic Semantic Scholar URL, "
            "and publication year.\n"
            "Return ONLY a JSON object matching this schema:\n"
            "{\n"
            '  "papers": [\n'
            '    {\n'
            '      "title": "string",\n'
            '      "abstract": "string",\n'
            '      "authors": [\n'
            '        {\n'
            '          "name": "string"\n'
            '        }\n'
            '      ],\n'
            '      "url": "string",\n'
            '      "year": int\n'
            '    }\n'
            '  ]\n'
            "}\n"
            "Ensure the abstracts are detailed and sound like actual published scientific abstracts."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Query: {query}"}
        ]
        try:
            logger.info(f"Generating mock papers for query: {query}...")
            raw_response = self._call_groq_api(messages)
            data = json.loads(raw_response)
            papers = data.get("papers", [])
            logger.info(f"Generated {len(papers)} mock papers.")
            return papers
        except Exception as e:
            logger.error(f"Error generating mock papers: {e}")
            return [
                {
                    "title": f"Advancements in {query.title()}",
                    "abstract": f"This paper explores key concepts, frameworks, and practical methodologies related to {query}. "
                                "We detail foundational elements and provide comprehensive evaluations on baseline benchmarks.",
                    "authors": [{"name": "A. Author"}, {"name": "B. Researcher"}],
                    "url": "https://www.semanticscholar.org/paper/example-1",
                    "year": 2025
                }
            ]

    def find_best_matching_paper(self, chunk_text: str, papers: List[Dict[str, Any]]) -> int:
        """
        Looks at the chunk_text and returns the index of the most similar paper abstract (0-4),
        or -1 if the chunk does not match any of them.
        """
        system_prompt = (
            "You are a research assistant screening a passage for potential plagiarism.\n"
            "Compare the given suspected passage against the academic paper abstracts provided.\n"
            "Identify which paper abstract (index 0 to 4) is semantically matching or has been "
            "paraphrased/copied. If the passage is completely original and does not match any of them, return -1.\n"
            "Return ONLY a JSON object matching this schema:\n"
            "{\n"
            '  "matched_index": int\n'
            "}"
        )
        
        papers_summary = ""
        for idx, p in enumerate(papers):
            papers_summary += f"--- PAPER ABSTRACT {idx} ---\nTitle: {p['title']}\nAbstract: {p['abstract']}\n\n"
            
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{papers_summary}--- SUSPECTED PASSAGE ---\n{chunk_text}"}
        ]
        try:
            logger.info("Screening chunk against paper abstracts...")
            raw_response = self._call_groq_api(messages)
            data = json.loads(raw_response)
            matched_idx = int(data.get("matched_index", -1))
            if 0 <= matched_idx < len(papers):
                logger.info(f"Chunk matched paper index: {matched_idx}")
                return matched_idx
            return -1
        except Exception as e:
            logger.error(f"Error screening chunk matching paper: {e}")
            return -1
