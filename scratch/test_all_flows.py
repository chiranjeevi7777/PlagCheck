import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from app import app
from extractor import TextExtractor
from chunker import DocumentChunker
from semanticscholar import SemanticScholarClient
from groq_client import GroqPlagiarismClient
from ai_analyzer import AIPatternAnalyzer, AIWritingPatternService
from report import PlagiarismReporter
from config import settings

def run_tests():
    print("==================================================")
    print("RUNNING PLAGCHECK SYSTEM VERIFICATION TESTS")
    print("==================================================")
    
    # 1. Test Text Extractor
    print("\n--- 1. Testing TextExtractor ---")
    pdf_path = Path("../test_doc.pdf")
    if not pdf_path.exists():
        pdf_path = Path("test_doc.pdf")
    
    if not pdf_path.exists():
        print("ERROR: test_doc.pdf not found. Please run create_test_pdf.py first.")
        sys.exit(1)
        
    text = TextExtractor.detect_and_extract(pdf_path)
    print(f"Extracted Text Length: {len(text)} characters")
    print(f"Sample Text:\n{text[:200]}...")
    assert len(text) > 0, "Failed to extract text from PDF"
    print("PASS: TextExtractor successfully extracted text from PDF")

    # 2. Test Document Chunker
    print("\n--- 2. Testing DocumentChunker ---")
    chunker = DocumentChunker()
    chunks = chunker.split_into_chunks(text)
    print(f"Number of chunks generated: {len(chunks)}")
    for idx, chunk in enumerate(chunks):
        print(f"  Chunk {idx}: ID={chunk['id']}, Words={chunk['word_count']}, Text Length={len(chunk['text'])}")
    assert len(chunks) > 0, "Chunker returned 0 chunks"
    print("PASS: DocumentChunker successfully split text into semantic chunks")

    # 3. Test Groq Connection and Client
    print("\n--- 3. Testing Groq API Client and Semantic Scholar ---")
    groq_client = GroqPlagiarismClient()
    
    # Verify we can make a simple call
    print("Testing connection to Groq API...")
    test_messages = [
        {"role": "system", "content": "You are a helper."},
        {"role": "user", "content": "Say hello in one word."}
    ]
    try:
        response = groq_client._call_groq_api(test_messages)
        print(f"Groq Response: {response.strip()}")
        print("PASS: Groq API client connection works")
    except Exception as e:
        print(f"WARNING: Groq API client call failed: {e}")

    # 4. Test Semantic Scholar Search
    print("\n--- 4. Testing Semantic Scholar Search / Mock Fallback ---")
    ss_client = SemanticScholarClient(groq_client)
    papers = ss_client.search_papers("deep learning", limit=3)
    print(f"Retrieved {len(papers)} papers")
    for idx, p in enumerate(papers):
        print(f"  Paper {idx+1}: '{p.get('title')}' ({p.get('year', 'N/A')})")
    assert len(papers) > 0, "Semantic Scholar failed to return any papers/mock papers"
    print("PASS: Semantic Scholar / Fallback paper discovery works")

    # 5. Test AI Pattern Analyzer
    print("\n--- 5. Testing AI Writing Pattern Analysis ---")
    ai_service = AIWritingPatternService(groq_client, max_workers=2)
    print("Running AI Pattern analysis on extracted document chunks...")
    ai_report = ai_service.analyze_document(chunks)
    print(f"Overall AI Score: {ai_report['overall_ai_score']}%")
    print(f"Classification: {ai_report['overall_classification']}")
    print(f"Average Confidence: {ai_report['average_confidence']}%")
    print(f"Detected Features: {[f['feature'] for f in ai_report['top_features']]}")
    assert "overall_ai_score" in ai_report, "AI report missing overall score"
    print("PASS: AIWritingPatternService analyzed document successfully")

    # 6. Test PDF Report Generation
    print("\n--- 6. Testing PDF Report Generation ---")
    # Make a dummy combined report dict
    combined_report_data = {
        "metadata": {
            "timestamp": "2026-06-28 12:00:00",
            "query": "deep learning",
            "paper_count": len(papers),
            "suspected_filename": "test_doc.pdf",
            "suspected_word_count": sum(c["word_count"] for c in chunks),
            "suspected_chunk_count": len(chunks)
        },
        "overall_similarity": 15,
        "overall_exact_copy": 5,
        "overall_paraphrase": 10,
        "overall_classification": "Light Rewrite",
        "average_confidence": 90,
        "flagged_chunks_count": 0,
        "chunks": [
            {
                "suspected_chunk_id": "chunk_0",
                "suspected_text": chunks[0]["text"],
                "original_chunk_id": "ref_chunk_0",
                "original_text": "Deep learning is a subfield of machine learning inspired by neural networks.",
                "semantic_similarity": 15,
                "exact_copy": 5,
                "paraphrase": 10,
                "classification": "Light Rewrite",
                "confidence": 90,
                "reason": "Text shows slight stylistic and structure overlap.",
                "sentence_matches": [],
                "original_title": papers[0]["title"] if papers else "Deep Neural Networks Study",
                "original_authors": "Jane Doe et al.",
                "original_year": "2023",
                "original_url": "https://example.org/paper1"
            }
        ],
        "ai_analysis": ai_report
    }
    
    test_pdf_path = Path("test_combined_report.pdf")
    if test_pdf_path.exists():
        test_pdf_path.unlink()
        
    PlagiarismReporter.generate_combined_pdf_report(combined_report_data, test_pdf_path)
    assert test_pdf_path.exists(), "Combined PDF report was not created"
    assert test_pdf_path.stat().st_size > 0, "Combined PDF report is empty"
    print(f"Generated PDF size: {test_pdf_path.stat().st_size} bytes")
    print("PASS: PlagiarismReporter generated combined PDF report successfully")
    
    # 7. Test FastAPI Routes via TestClient
    print("\n--- 7. Testing FastAPI Endpoints ---")
    client = TestClient(app)
    
    # Test Root / Index
    resp = client.get("/")
    assert resp.status_code == 200, f"Root endpoint returned {resp.status_code}"
    print("PASS: GET / is successful")
    
    # Test Upload
    with open(pdf_path, "rb") as f:
        resp = client.post("/upload", files={"file": ("test_doc.pdf", f, "application/pdf")}, data={"search_query": "deep learning"})
    assert resp.status_code == 200, f"Upload endpoint returned {resp.status_code}"
    upload_json = resp.json()
    print(f"Upload success! Path: {upload_json['file_path']}")
    
    # Test Analyze-AI trigger (standalone AI task)
    resp = client.post("/analyze-ai", json={
        "file_path": upload_json["file_path"],
        "filename": upload_json["filename"]
    })
    assert resp.status_code == 200, f"Analyze-AI endpoint returned {resp.status_code}"
    task_json = resp.json()
    task_id = task_json["task_id"]
    print(f"Analyze-AI Triggered Task ID: {task_id}")
    
    # Poll status
    import time
    print("Polling task status...")
    for _ in range(15):
        resp = client.get(f"/analyze/status/{task_id}")
        assert resp.status_code == 200, f"Status check failed: {resp.status_code}"
        status_data = resp.json()
        print(f"  Status: {status_data['status']}, Progress: {status_data['progress']}%")
        if status_data["status"] == "completed":
            break
        time.sleep(1)
        
    assert status_data["status"] == "completed", f"Task did not complete, status is {status_data['status']}"
    print("PASS: Standalone AI analysis task completed successfully via background workers")
    
    # Test AI report details
    report_id = status_data["report_id"]
    resp = client.get(f"/ai-report?report_id={report_id}")
    assert resp.status_code == 200, f"AI report endpoint returned {resp.status_code}"
    report_json = resp.json()
    print(f"AI Overall Score from endpoint: {report_json['overall_ai_score']}%")
    assert isinstance(report_json["overall_ai_score"], (int, float)), "AI score must be a number"
    
    # Clean up generated files
    if test_pdf_path.exists():
        test_pdf_path.unlink()
    print("PASS: All FastAPI endpoints verified successfully")
    
    print("\n==================================================")
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    run_tests()
