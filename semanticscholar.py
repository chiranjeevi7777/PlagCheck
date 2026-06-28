import requests
from typing import List, Dict, Any
from utils import logger
from groq_client import GroqPlagiarismClient

class SemanticScholarClient:
    """Client for searching academic literature using Semantic Scholar with an LLM fallback."""

    def __init__(self, groq_client: GroqPlagiarismClient):
        self.groq_client = groq_client
        self.search_url = "https://api.semanticscholar.org/graph/v1/paper/search"

    def search_papers(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Queries Semantic Scholar for relevant papers.
        Falls back to Groq-generated papers if rate-limited (429) or on network failure.
        """
        params = {
            "query": query,
            "limit": limit,
            "fields": "title,abstract,authors,url,year"
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        
        try:
            logger.info(f"Querying Semantic Scholar API for: '{query}'")
            response = requests.get(self.search_url, params=params, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                papers = data.get("data", [])
                
                # Filter out papers without abstracts
                valid_papers = []
                for p in papers:
                    if p.get("abstract") and p.get("abstract").strip():
                        valid_papers.append(p)
                
                logger.info(f"Retrieved {len(valid_papers)} papers with abstracts from Semantic Scholar.")
                
                # If we retrieved enough papers, return them
                if len(valid_papers) >= 2:
                    return valid_papers
                else:
                    logger.warning("Fewer than 2 papers with abstracts returned. Supplementing/falling back to LLM generation.")
            
            elif response.status_code == 429:
                logger.warning("Semantic Scholar API returned 429 (Too Many Requests). Falling back to Groq LLM paper generation.")
            else:
                logger.warning(f"Semantic Scholar API returned status {response.status_code}. Falling back to Groq LLM paper generation.")
                
        except Exception as e:
            logger.error(f"Semantic Scholar query failed: {e}. Falling back to Groq LLM paper generation.")
            
        # Fallback: Generate realistic papers using Groq LLM
        return self.groq_client.generate_mock_papers(query)
