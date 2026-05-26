from fastapi import FastAPI, Query
from src.ingestion import load_data
from src.embeddings import EmbeddingEngine
from src.retriever import VectorStore
from src.llm_pipeline import LLMPipeline
import os

app = FastAPI(title="LLM-Powered Support Query API")

# Global instances (simplified for this demo)
DATA_PATH = "data/tickets.csv"
INDEX_PATH = "data/ticket_index"

embedding_engine = EmbeddingEngine()
vector_store = None
llm_pipeline = LLMPipeline()

@app.on_event("startup")
def startup_event():
    global vector_store
    df = load_data(DATA_PATH)
    if df.empty:
        print("Dataset not found or empty.")
        return

    # Initialize VectorStore
    sample_emb = embedding_engine.get_embeddings(["test"])
    dimension = sample_emb.shape[1]
    vector_store = VectorStore(dimension)

    # Build index if metadata doesn't exists
    if not os.path.exists(f"{{INDEX_PATH}}.metadata"):
        print("Building FAISS index...")
        embeddings = embedding_engine.get_embeddings(df["cleaned_query"].tolist())
        vector_store.add_documents(embeddings, df)
        vector_store.save(INDEX_PATH)
    else:
        print("Loading FAISS index...")
        vector_store.load(INDEX_PATH)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Support Query Understanding API"}

@app.get("/retrieve")
def retrieve_tickets(q: str, k: int = 3):
    """Retrieve top-k similar tickets."""
    if not vector_store:
        return {"error": "Vector store not initialized"}
    
    query_emb = embedding_engine.get_embeddings([q])
    results = vector_store.search(query_emb, k=k)
    return {"query": q, "results": results}

@app.get("/predict")
def predict_intent(q: str, strategy: str = "few-shot"):
    """Predict intent, summary, and response using RAG."""
    # 1. Retrieve
    query_emb = embedding_engine.get_embeddings([q])
    context = vector_store.search(query_emb, k=2)
    
    # 2. Query LLM
    prediction = llm_pipeline.query(q, context=context, strategy=strategy)
    
    return {
        "query": q,
        "strategy": strategy,
        "prediction": prediction,
        "retrieved_context": context
    }
