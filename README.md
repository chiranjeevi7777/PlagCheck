# PlagCheck AI - Groq-Powered Plagiarism & AI Writing Pattern Detector

An enterprise-ready plagiarism detection and AI writing pattern analysis system built with **FastAPI** (Python 3.12) and the **Groq Cloud API**. It performs structural, lexical, and semantic comparison of uploaded documents against literature without vector database overhead.

---

## Key Features

- **Document Parsing:** Supports `.pdf` and `.docx` uploads up to 50MB.
- **Intelligent Chunking:** Automatically segments text into sentence-aware blocks of 300–500 words with a 50-word overlap.
- **Linear-Scaled Comparison:** Matches text using a mapped sliding-window alignment strategy to keep LLM calls at $O(N)$ linear scale.
- **AI Writing Pattern Analyzer:** Analyzes chunks in parallel via a multi-threaded worker pool to detect predictability, structure regularity, and lack of sentence length variation.
- **Interactive UI Dashboard:** A glassmorphic dark-mode single page application (SPA) featuring analysis grids, Chart.js visualizations, and side-by-side matching overlays.
- **PDF Report Downloads:** Generates comprehensive PDF reports with highlighted matches, detailed scores, and system warnings.

---

## Performance & Reliability Optimizations (New)

### 1. Lexical Pre-filtering (Token Savings)
To prevent unnecessary API costs and speed up comparisons, the comparator implements a content-word overlap screening:
* Suspected document chunks are compared against reference abstracts using tokenized content-words (excluding common English stop words).
* If the maximum overlap is **less than 3 content words**, the LLM call is bypassed. The chunk is marked **Original**, reducing API token usage by up to **90%** for original papers.

### 2. Multi-Key API Rotation
* The system accepts a comma-separated list of keys in `GROQ_API_KEY` (or `GROQ_API_KEYS`).
* If a key hits a rate limit (HTTP 429), the client automatically rotates to the next key.

### 3. Graceful Model Fallback
* If a model is deprecated, decommissioned (like `gemma2-9b-it`), or unsupported, the client catches the HTTP 400/404 error, logs a warning, skips that model, and falls back to the next model in the priority list:
  1. `llama-3.3-70b-versatile` (Primary)
  2. `llama-3.1-8b-instant`
  3. `qwen/qwen3-32b`
  4. `qwen/qwen3.6-27b`

### 4. Fast Failover (No SDK Retries)
* Disabled the default Groq SDK exponential backoff (which blocks requests for 50+ seconds on rate limits) by setting `max_retries=0`, ensuring the failover to the next key or model triggers instantly.

---

## Folder Structure

```
PlagCheck/
├── app.py                  # FastAPI server with lifespan handler
├── config.py               # Settings and Environment configuration
├── routes.py               # API routes (upload, status, reports)
├── extractor.py            # PDF / Word text extraction
├── chunker.py              # Paragraph-based sentence chunking
├── groq_client.py          # Groq SDK Client with rotation and fallback
├── comparator.py           # Plagiarism comparison with lexical pre-filtering
├── ai_analyzer.py          # Multi-threaded AI pattern analysis
├── report.py               # ReportLab PDF compilation engine
├── utils.py                # Logging and NLTK pre-warming
├── requirements.txt        # Backend dependencies
├── .env                    # Environment credentials (ignored by Git)
├── .gitignore              # Files ignored by Git
└── static/
    ├── css/
    │   ├── style.css       # SPA theme styling
    │   └── ai_style.css    # AI writing pattern styling
    ├── js/
    │   ├── app.js          # Plagiarism charting & visualizer
    │   └── ai_app.js       # AI metrics visualizer
    └── templates/
        └── index.html      # Glassmorphic UI Dashboard
```

---

## Installation & Local Execution

### 1. Install Requirements
Create a virtual environment and run:
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=gsk_yourKey1,gsk_yourKey2
GROQ_MODEL=llama-3.3-70b-versatile
```

### 3. Run the Development Server
```bash
uvicorn app:app --reload
```
Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your web browser.

---

## Production Deployment on Render

Render automatically detects the repository and deploys the app as a standard Web Service.

1. **Connect Repository:** Connect your GitHub repository to Render.
2. **Build Command:** `pip install -r requirements.txt`
3. **Start Command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
4. **Environment Variables:** Add `GROQ_API_KEY` (comma-separated list) to your Render environment variables.
