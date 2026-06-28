import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from comparator import PlagiarismComparator

# Mock a simple GroqPlagiarismClient
class MockGroqClient:
    def __init__(self):
        self.calls = 0

    def find_best_matching_paper(self, text, papers):
        self.calls += 1
        return 0  # Always matches the first paper

    def compare_chunks(self, original_text, suspected_text):
        class MockResult:
            semantic_similarity = 80
            exact_copy = 10
            paraphrase = 70
            classification = "Light Rewrite"
            confidence = 90
            reason = "Test"
            sentence_matches = []
        return MockResult()

# 1. Setup mock papers and suspected chunks
papers = [
    {
        "title": "Quantum Computing",
        "abstract": "This abstract discusses quantum superposition, entanglement, qubits, and quantum algorithms such as Shor's algorithm.",
        "authors": [{"name": "A. Physicist"}],
        "url": "https://example.com/quantum",
        "year": 2024
    }
]

# Chunk 1: shares lots of words with the abstract (quantum, superposition, entanglement, qubits, algorithms)
chunk_high_overlap = {
    "id": "chunk_high",
    "text": "Quantum computing uses superposition and entanglement to manipulate qubits for running advanced quantum algorithms.",
}

# Chunk 2: shares almost no words with the abstract (deep, learning, models, GPU, backpropagation)
chunk_low_overlap = {
    "id": "chunk_low",
    "text": "Deep learning models are trained using backpropagation on large graphical processors to recognize complex patterns in image datasets.",
}

client = MockGroqClient()
comparator = PlagiarismComparator(client)

results = comparator.compare_against_papers([chunk_high_overlap, chunk_low_overlap], papers)

print(f"Total Groq Client find_best_matching_paper calls: {client.calls}")
print(f"Chunk High similarity: {results[0]['semantic_similarity']}")
print(f"Chunk Low similarity: {results[1]['semantic_similarity']}")
print(f"Chunk Low reason: {results[1]['reason']}")

assert client.calls == 1, f"Expected exactly 1 call (for high overlap), got {client.calls}"
assert results[0]['semantic_similarity'] == 80, "Expected similarity 80 for high overlap"
assert results[1]['semantic_similarity'] == 0, "Expected similarity 0 for low overlap"
print("SUCCESS: Lexical pre-filtering successfully verified!")
