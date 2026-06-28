# PlagCheck AI - Groq-Powered Plagiarism Detection System

An enterprise-ready plagiarism detection system built with **FastAPI** (Python 3.12) and **Groq Cloud LLM** (using `llama-3.3-70b-versatile`). It performs structural and semantic comparison between original and suspected documents without database overhead or external vector embeddings.

---

## Features
- **File support:** Handles `.pdf` and `.docx` uploads up to 50MB.
- **Intelligent chunking:** Groups text into sentence-aware blocks of 300–500 words with a 50-word overlap.
- **Linear-scaled comparisons:** Utilizes a mapped sliding-window alignment strategy to reduce LLM calls from $O(N \times M)$ to $O(N)$ linear scale.
- **Detailed JSON responses:** Evaluates semantic similarity, exact copy %, paraphrase %, change reasoning, and sentence-level matches.
- **Interactive UI Dashboard:** Stunning glassmorphic dark-mode SPA containing progress bars, Chart.js diagrams, sortable tables, and side-by-side highlighting.
- **PDF Report Downloads:** Standardized ReportLab-generated PDF files with analysis cards, system recommendations, and sentence highlights.
- **Fail-safe:** Custom retry handlers with exponential backoff for Groq rate limits, and fallback extraction algorithms.

---

## Folder Structure
```
PlagCheck/
├── app.py                  # FastAPI Entrypoint
├── config.py               # Config & Dotenv Loader
├── routes.py               # API Router Endpoints
├── extractor.py            # PDF / DOCX Text Extractor Heuristics
├── chunker.py              # Sentence Chunker
├── groq_client.py          # Groq SDK Wrapper & JSON validation
├── comparator.py           # Sliding Window Comparator Engine
├── report.py               # Metrics aggregator & ReportLab PDF Generator
├── utils.py                # Logging & NLTK loader
├── requirements.txt        # Backend dependencies
├── .env                    # System variables (API Keys)
├── README.md               # User documentation
├── static/
│   ├── css/
│   │   └── style.css       # Premium styles
│   └── js/
│       └── app.js          # Chart & interactive UI logic
└── templates/
    └── index.html          # SPA HTML Structure
```

---

## Installation & Setup

### 1. Clone & Navigate to Workspace
Ensure you are inside the `PlagCheck` folder:
```bash
cd PlagCheck
```

### 2. Install Requirements
Create a virtual environment (optional but recommended) and run:
```bash
pip install -r requirements.txt
```

### 3. Setup Groq API Key
Open `.env` and fill in your Groq API Key:
```env
GROQ_API_KEY=your_actual_groq_api_key_here
```

### 4. Run the Server
Launch the development server using Uvicorn:
```bash
uvicorn app:app --reload
```
Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your web browser to run the application.
