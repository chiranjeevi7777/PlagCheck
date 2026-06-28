# PlagCheck AI - Retrieval-Augmented Plagiarism Detection Platform

An enterprise-ready, retrieval-augmented plagiarism detection and AI writing pattern analysis system. Built with **FastAPI** (Python 3.11+) and the **Groq Cloud API**, PlagCheck AI performs hybrid semantic search, multi-source academic retrieval, two-stage candidate reranking, and segment-level LLM verification without vector database overhead.

---

## 🚀 Key Features

*   **Retrieval-Augmented Plagiarism Detection (RAG)**: Uses a unified `RetrievalManager` to search concurrently across 5 academic databases: **Semantic Scholar**, **OpenAlex**, **arXiv**, **Crossref**, and **CORE**.
*   **Intelligent Query Engineering**: Automatically expands and reformulates raw document segments into **four distinct search queries** (keyword, semantic, expanded, and academic) to maximize recall.
*   **Two-Stage Ranking Pipeline**:
    *   *First Stage (Dense Embeddings)*: Compares document chunks with candidate papers using semantic text embeddings.
    *   *Second Stage (Cross-Encoder Reranking)*: Evaluates candidate abstracts against document chunks using a cross-encoder model to surface the most relevant reference papers.
*   **Linear-Scaled Comparison Engine**: Performs target verification of suspected document chunks against reference papers using a sliding-window mapping layout to keep LLM calls at $O(N)$ linear complexity.
*   **AI Writing Pattern Service**: Evaluates predictability, structure homogeneity, and lexical density (repetition of transition words) in parallel across a multi-threaded worker pool.
*   **Robust Groq Client**:
    *   *Global Round-Robin Key Rotation*: Distributes API requests dynamically across a list of comma-separated keys to avoid TPM/RPM rate limits.
    *   *Flexible Model Fallback*: Orderly transitions across multiple models (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, `mixtral-8x7b-32768`, `gemma2-9b-it`, etc.) on decommissioning or outages.
    *   *Prompt Optimization*: Explicitly instructs models to avoid markdown fence wrapping (such as ` ```json `), preventing `json_validate_failed` HTTP 400 errors.
*   **Render Free-Tier Optimization**: Leverages the **Hugging Face Inference API** for embedding and cross-encoder tasks (falling back to a lightweight local TF-IDF signed hashing system when offline). This limits active memory usage to **<150MB RAM**, easily running under Render's 512MB limits.
*   **Intelligent Document Revision & Originality Enhancement Engine**:
    *   *Linguistic Analytics & Issues Classification*: Uses hybrid metrics and LLM classification to detect readability issues, passive voice, mechanical tone, weak style, and plagiarism matches.
    *   *Strict Integrity Rewriting Loop*: Paraphrases paragraphs to resolve issues while strictly keeping all original citations, equations, numbers, and technical terminology in place, with automatic verification and retry fallback checks.
    *   *Style-Preserving Document Export*: Applies chosen revisions directly to the uploaded Word document structure (.docx), outputting a modified file with original layouts, margins, and headings preserved.
    *   *Side-by-Side Visual Diff Editor*: Displays color-coded additions and deletions side-by-side on an interactive glassmorphic dashboard tab.
*   **Interactive Glassmorphic Dashboard**: A modern, dark-mode dashboard with side-by-side plagiarism comparisons, sentence-level matching, and Chart.js metrics.
*   **Comprehensive PDF Reports**: Generates professional PDF summaries using ReportLab, highlighting match percentages, source citations, and AI probability.

---

## 📁 System Directory Structure

```text
PlagCheck/
├── app/
│   ├── api/
│   │   ├── enhancement_routes.py # Document enhancement and revision endpoints
│   │   └── routes.py             # Main plagiarism & AI pattern API endpoints
│   ├── core/
│   │   ├── config.py             # Environment configuration and settings loading
│   │   └── logging.py            # Centralized application logging
│   ├── embedding/
│   │   └── bge_embedder.py       # Embedding generator (HF API + local TF-IDF fallback)
│   ├── enhancement/              # Originality Enhancement Engine package
│   │   ├── classifier.py         # Paragraph issue classifier (Linguistic + LLM)
│   │   ├── comparison.py         # Version history manager & track changes
│   │   ├── diff_engine.py        # HTML-highlighted character visual diff generator
│   │   ├── document_writer.py    # Style-preserving Word document writer
│   │   ├── metrics.py            # Readable metrics & lexical complexity evaluator
│   │   ├── planner.py            # Paragraph-level step-by-step revision planner
│   │   ├── report.py             # Enhancement report compiler and analyzer
│   │   ├── rewriter.py           # Integrity-preserving LLM rewrite controller
│   │   └── validator.py          # Academic facts, equations & citation validator
│   ├── llm/
│   │   ├── groq_client.py        # Robust Groq client with rotation and fallbacks
│   │   ├── query_engineer.py     # Document query expansion / rewriting service
│   │   └── verifier.py           # Segment comparison verifier (Groq)
│   ├── reranking/
│   │   └── cross_encoder.py      # Cross-Encoder reranker service (HF API + fallback)
│   ├── retrieval/
│   │   ├── clients/              # API clients (arXiv, Crossref, OpenAlex, Semantic Scholar, CORE)
│   │   └── manager.py            # Unified academic retrieval coordinator
│   ├── schemas/
│   │   ├── analysis.py           # Pydantic schemas for verification and style reports
│   │   └── retrieval.py          # Pydantic schemas for query bundles and papers
│   ├── services/
│   │   ├── ai_analyzer.py        # Parallelized AI writing style evaluator
│   │   ├── chunking.py           # Sentence-aware document segmentation
│   │   ├── comparison.py         # Sliding-window comparison manager
│   │   ├── extraction.py         # PDF & Word text extractor
│   │   └── report.py             # ReportLab PDF report generation engine
│   ├── workers/
│   │   └── pipeline.py           # Asynchronous document analysis coordinator
│   └── main.py                   # FastAPI application setup and lifespans
├── static/                       # CSS and JS dashboard resources
│   ├── css/
│   │   └── style.css             # Glassmorphic style definitions
│   └── js/
│       ├── app.js                # Core dashboard JS and plotting logic
│       ├── ai_app.js             # Standalone AI analysis logic
│       └── enhancement_app.js    # Interactive diff editor and version controller
├── templates/                    # Jinja2 HTML layout file
├── app.py                        # Root entrypoint shim (Uvicorn target)
├── requirements.txt              # Python dependencies
├── .env                          # Application credentials (local template)
└── README.md                     # Project documentation
```

---

## 🛠️ Installation & Local Setup

### 1. Clone & Install Dependencies
Ensure you have Python 3.11+ installed. Clone the repository, create a virtual environment, and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment variables
Create a `.env` file in the root directory:
```env
# Groq LLM Configuration (Multiple keys comma-separated)
GROQ_API_KEY=gsk_keyA,gsk_keyB,gsk_keyC
GROQ_MODEL=llama-3.3-70b-versatile
FALLBACK_MODELS=llama-3.3-70b-versatile,llama-3.1-8b-instant,mixtral-8x7b-32768,gemma2-9b-it,qwen/qwen3-32b,qwen/qwen3.6-27b
TEMPERATURE=0.0
MAX_TOKENS=1024

# Academic Sources Features
ENABLE_OPENALEX=true
ENABLE_ARXIV=true
ENABLE_CROSSREF=true
ENABLE_CORE=true
CORE_API_KEY=your_core_api_key_here

# Embedding & Reranking Tuning
ENABLE_EMBEDDING=true
ENABLE_RERANKING=true
USE_HUGGINGFACE_API=true
HF_API_TOKEN=your_huggingface_api_token_here
MAX_RETRIEVED_PAPERS=30
MAX_RERANKED_PAPERS=10
RETRIEVAL_TIMEOUT=12.0
```

### 3. Run Development Server
Start the local server using the root shim:
```bash
uvicorn app:app --reload
```
Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

---

## 🌐 Production Deployment on Render

To deploy the platform as a Web Service on **Render** (or equivalent cloud hosts):

1.  **Connect Repository**: Link your GitHub repository to your Render dashboard.
2.  **Configuration Settings**:
    *   **Environment**: `Python`
    *   **Build Command**: `pip install -r requirements.txt`
    *   **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
3.  **Environment Variables**:
    *   Add your `GROQ_API_KEY` (or multiple comma-separated keys to avoid rate limits).
    *   Add `HF_API_TOKEN` to enable high-quality semantic embeddings via the Hugging Face Inference API.
    *   (Optional) Add `CORE_API_KEY` for access to the CORE academic database.
