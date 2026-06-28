"""Quick sanity test — run with: python test_sanity.py"""

print("=== TEST 2: Schema Validation ===")
from app.schemas.retrieval import CandidatePaper, QueryBundle
from app.schemas.analysis import ChunkComparisonResult, classify_ai_probability

qb = QueryBundle(
    keyword_query="machine learning neural networks",
    semantic_query="deep learning models for classification",
    expanded_query="ML AI deep learning artificial neural network",
    academic_query="gradient descent backpropagation convolutional networks",
)
queries = qb.all_queries()
assert len(queries) == 4
print(f"  [OK] QueryBundle.all_queries(): {len(queries)} unique queries")

p = CandidatePaper(title="Test Paper", abstract="This is a test abstract about AI.", source="openalex")
p.embedding_score = 0.87
p.reranker_score = 0.92
p.keyword_overlap = 0.45
p.combined_score = 0.80
print(f"  [OK] CandidatePaper: title={p.title!r}, combined_score={p.combined_score}")

for prob, expected in [(10, "Very Low"), (35, "Low"), (55, "Moderate"), (75, "High"), (90, "Very High")]:
    cls = classify_ai_probability(prob)
    assert expected in cls, f"{prob} -> {cls}"
print("  [OK] classify_ai_probability(): all 5 bands correct")

cr = ChunkComparisonResult(
    semantic_similarity=80, exact_copy=10, paraphrase=30,
    classification="Heavy Paraphrasing", confidence=85,
    reason="Test", sentence_matches=[],
)
print(f"  [OK] ChunkComparisonResult: {cr.classification}")


print("\n=== TEST 3: Chunker ===")
from app.services.chunking import DocumentChunker

text = (
    "Artificial intelligence (AI) is intelligence demonstrated by machines. "
    "Machine learning is a subset of artificial intelligence. "
    "Deep learning uses neural networks with many layers. "
    "Natural language processing enables computers to understand text. "
    "Computer vision allows machines to interpret visual information. "
    "Reinforcement learning trains agents through reward signals. "
    "Transfer learning applies knowledge from one domain to another. "
    "Generative models can create new data similar to training data."
)

chunker = DocumentChunker(chunk_size=50, overlap=10, max_chunk_size=80, min_chunk_size=20)
chunks = chunker.split(text)
assert len(chunks) >= 1
print(f"  [OK] Chunked into {len(chunks)} chunks")
for c in chunks:
    print(f"       {c['id']}: {c['word_count']} words")


print("\n=== TEST 4: TextExtractor ===")
from app.services.extraction import TextExtractor
from pathlib import Path

pdf_path = Path("test_doc.pdf")
if pdf_path.exists():
    text_out = TextExtractor.detect_and_extract(pdf_path)
    wc = len(text_out.split())
    print(f"  [OK] Extracted {wc} words from {pdf_path.name}")
    assert wc > 0
else:
    print("  [SKIP] test_doc.pdf not found")


print("\n=== TEST 5: QueryBundle dedup ===")
qb2 = QueryBundle(
    keyword_query="neural networks",
    semantic_query="neural networks",
    expanded_query="neural networks deep learning",
    academic_query="artificial neural network architectures",
)
queries2 = qb2.all_queries()
print(f"  [OK] Deduped to {len(queries2)} unique queries (3 expected from 4 slots with 1 duplicate)")
assert len(queries2) == 3


print("\n=== TEST 6: BGEEmbedder graceful fallback ===")
from app.embedding.bge_embedder import BGEEmbedder
import numpy as np

emb = BGEEmbedder()
vec = emb.encode("This is a test sentence for embedding.")
print(f"  [OK] encode() returned shape={vec.shape}, dtype={vec.dtype}")
assert vec.shape[0] in (384, 768), f"Unexpected dim: {vec.shape[0]}"


print("\n=== TEST 7: CrossEncoderReranker keyword fallback ===")
from app.reranking.cross_encoder import CrossEncoderReranker

reranker = CrossEncoderReranker()
candidates = [
    CandidatePaper(
        title="Neural Networks Survey",
        abstract="A comprehensive survey of deep neural network architectures and training methods.",
        source="openalex",
    ),
    CandidatePaper(
        title="Cooking Recipes",
        abstract="This paper discusses the best recipes for Italian cuisine including pasta and pizza.",
        source="arxiv",
    ),
    CandidatePaper(
        title="Deep Learning Applications",
        abstract="Applications of deep learning in computer vision and natural language processing tasks.",
        source="semantic_scholar",
    ),
]
ranked = reranker.rerank(
    query="deep neural network architectures training", candidates=candidates, top_k=2
)
print(f"  [OK] Reranked {len(candidates)} -> top {len(ranked)}")
for r in ranked:
    print(f"       {r.title}: combined={r.combined_score:.3f}")
assert len(ranked) == 2
assert ranked[0].title != "Cooking Recipes", "Ranking error: cooking paper ranked first"
print("  [OK] Cooking paper correctly ranked last")


print("\n=== TEST 8: RetrievalManager dedup ===")
from app.retrieval.manager import RetrievalManager

mgr = RetrievalManager()
dupes = [
    CandidatePaper(title="ML Fundamentals", abstract="An intro to ML.", source="ss", doi="10.1234/ml"),
    CandidatePaper(title="ML Fundamentals", abstract="Duplicate.", source="oa", doi="10.1234/ml"),
    CandidatePaper(title="Deep Learning Survey", abstract="Overview of DL.", source="arxiv"),
    CandidatePaper(title="Deep Learning Survey", abstract="Another duplicate.", source="crossref"),
]
unique = mgr._deduplicate(dupes)
print(f"  [OK] Dedup: {len(dupes)} raw -> {len(unique)} unique (removed {len(dupes)-len(unique)})")
assert len(unique) == 2


print("\n=== TEST 9: Live retrieval (SemanticScholar async) ===")
import asyncio
from app.retrieval.clients.semantic_scholar import SemanticScholarClient

async def test_ss():
    client = SemanticScholarClient(timeout=15.0)
    papers = await client.search("deep learning image classification", limit=3)
    return papers

papers = asyncio.run(test_ss())
print(f"  [OK] SemanticScholar returned {len(papers)} papers")
if papers:
    print(f"       First: {papers[0].title[:60]!r}")
    print(f"       Source: {papers[0].source}, Year: {papers[0].year}")


print("\n=== TEST 10: Live retrieval (OpenAlex async) ===")
from app.retrieval.clients.openalex import OpenAlexClient

async def test_oa():
    client = OpenAlexClient(timeout=15.0)
    papers = await client.search("transformer attention mechanism NLP", limit=3)
    return papers

papers_oa = asyncio.run(test_oa())
print(f"  [OK] OpenAlex returned {len(papers_oa)} papers")
if papers_oa:
    print(f"       First: {papers_oa[0].title[:60]!r}")


print("\n=== TEST 11: Live retrieval (arXiv async) ===")
from app.retrieval.clients.arxiv import ArxivClient

async def test_arxiv():
    client = ArxivClient(timeout=15.0)
    papers = await client.search("convolutional neural network image recognition", limit=3)
    return papers

papers_ax = asyncio.run(test_arxiv())
print(f"  [OK] arXiv returned {len(papers_ax)} papers")
if papers_ax:
    print(f"       First: {papers_ax[0].title[:60]!r}")


print("\n=== TEST 12: Live retrieval (Crossref async) ===")
from app.retrieval.clients.crossref import CrossrefClient

async def test_cr():
    client = CrossrefClient(timeout=15.0)
    papers = await client.search("machine learning plagiarism detection", limit=3)
    return papers

papers_cr = asyncio.run(test_cr())
print(f"  [OK] Crossref returned {len(papers_cr)} papers")
if papers_cr:
    print(f"       First: {papers_cr[0].title[:60]!r}")


print("\n=== TEST 13: Full RetrievalManager concurrent fetch ===")
from app.schemas.retrieval import QueryBundle

async def test_manager():
    qb = QueryBundle(
        keyword_query="plagiarism detection machine learning",
        semantic_query="automated text similarity academic integrity",
        expanded_query="plagiarism detection NLP text reuse similarity",
        academic_query="computational methods for academic plagiarism identification",
    )
    result = await mgr.retrieve(query_bundle=qb, chunk_text="Plagiarism detection using machine learning.", papers_per_source=3)
    return result

import time
t0 = time.monotonic()
result = asyncio.run(test_manager())
elapsed = time.monotonic() - t0
print(f"  [OK] RetrievalManager: {result.total_raw} raw -> {result.total_unique} unique -> {len(result.candidates)} ranked")
print(f"       Sources: {result.sources_used}")
print(f"       Time: {elapsed:.1f}s (concurrent across {len(result.sources_used)} sources)")
if result.candidates:
    top = result.candidates[0]
    print(f"       Top candidate: {top.title[:55]!r}")
    print(f"       Scores: emb={top.embedding_score:.3f} rerank={top.reranker_score:.3f} kw={top.keyword_overlap:.3f} combined={top.combined_score:.3f}")


print("\n=== TEST 14: FastAPI app routes ===")
from fastapi.testclient import TestClient
from app.main import app as fastapi_app

client = TestClient(fastapi_app)

# Health check via home route
resp = client.get("/")
print(f"  [OK] GET / -> HTTP {resp.status_code}")
assert resp.status_code == 200

# Status for non-existent task
resp = client.get("/analyze/status/nonexistent-task-id")
print(f"  [OK] GET /analyze/status/nonexistent -> HTTP {resp.status_code} (expected 404)")
assert resp.status_code == 404

# Report for non-existent ID
resp = client.get("/report?report_id=does-not-exist")
print(f"  [OK] GET /report?report_id=... -> HTTP {resp.status_code} (expected 404)")
assert resp.status_code == 404


print("\n============================")
print("ALL TESTS PASSED")
print("============================")
