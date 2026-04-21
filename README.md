LLM-Powered Support Query Understanding (RAG)
This project is a lightweight implementation of a Retrieval-Augmented Generation (RAG) pipeline for handling customer support queries. The goal was to explore how combining semantic search with LLMs can improve intent detection, summarization, and response generation.

Overview
The system takes a user query, retrieves similar past tickets using embeddings, and uses that context to generate structured outputs like intent, summary, and a suggested response.
It’s intentionally kept simple—no heavy infra—just enough to understand how RAG systems work end-to-end.

How it works
	•	Data ingestion Loads a small dataset of support tickets from data/tickets.csv (synthetic for now)
	•	Embeddings Uses Sentence Transformers (all-MiniLM-L6-v2) to convert text into vectors
	•	Retrieval Stores embeddings in a FAISS index and retrieves top-k similar tickets for a given query
	•	Prompting Supports:
	◦	Zero-shot (direct query to LLM)
	◦	Few-shot (injecting retrieved examples into the prompt)
	•	LLM layer Generates:
	◦	intent
	◦	short summary
	◦	response suggestion
	•	Uses OpenAI API if available, otherwise falls back to a mock mode
	•	Evaluation (basic) Includes simple checks like:
	◦	cosine similarity between outputs
	◦	heuristic-based hallucination detection

Project structure

data/        # dataset + FAISS index
src/         # core logic (embeddings, retrieval, prompts, etc.)
app/         # FastAPI app



Running the project
Install dependencies:

pip install -r requirements.txt

Start the server:

uvicorn app.main:app --reload


API
Retrieve similar tickets

GET /retrieve?q=your_query&k=3

Run full pipeline (intent + summary + response)

GET /predict?q=your_query&strategy=few-shot


Notes 
	•	Using all-MiniLM-L6-v2 keeps things fast and lightweight, but it’s not as expressive as larger embedding models
	•	FAISS works well for this scale, but for dynamic or large datasets, a managed vector DB would make more sense
	•	Prompt design has a noticeable impact—few-shot generally performs better, but adds latency

Example
Input

GET /predict?q=I can't log in

Output

{
  "intent": "Account Access",
  "summary": "User is unable to log into their account.",
  "response": "Try resetting your password or contact support if the issue continues."
}

