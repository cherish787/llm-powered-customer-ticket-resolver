from src.ingestion import load_data
from src.embeddings import EmbeddingEngine
from src.retriever import VectorStore
from src.llm_pipeline import LLMPipeline
from src.evaluation import Evaluator
import os

def test_pipeline():
    print("--- Starting Verification ---")
    
    # 1. Ingestion
    print("1. Testing Ingestion...")
    df = load_data("data/tickets.csv")
    assert not df.empty, "Ingestion failed: DataFrame is empty"
    print(f"Loaded {{len(df)}} tickets.")

    # 2. Embeddings
    print("2. Testing Embeddings...")
    engine = EmbeddingEngine()
    emb = engine.get_embeddings(["How do I reset my password?"])
    assert emb.shape[1] > 0, "Embedding generation failed"
    print(f"Embedding dimension: {{emb.shape[1]}}")

    # 3. Retriever
    print("3. Testing Retriever...")
    store = VectorStore(emb.shape[1])
    all_embs = engine.get_embeddings(df["cleaned_query"].tolist())
    store.add_documents(all_embs, df)
    
    query = "Forgot password"
    results = store.search(engine.get_embeddings([query]), k=2)
    assert len(results) > 0, "Search returned no results"
    print(f"Top result: {{results[0]['query']}}")

    # 4. LLM Pipeline (Mock)
    print("4. Testing LLM Pipeline (Mock mode)...")
    pipeline = LLMPipeline(use_mock=True)
    response = pipeline.query(query, context=results)
    assert "intent" in response, "LLM response missing intent"
    print(f"LLM Prediction: {{response['intent']}}")

    # 5. Evaluation
    print("5. Testing Evaluation...")
    evaluator = Evaluator(engine)
    sim = evaluator.get_similarity(query, results[0]['query'])
    print(f"Cosine Similarity: {{sim:.4f}}")
    hallucination = evaluator.detect_hallucination(response['response'], results)
    print(f"Hallucination Detected: {{hallucination}}")

    print("\n--- All tests passed! ---")

if __name__ == "__main__":
    test_pipeline()
