LLM-Powered Support Query Understanding (RAG)

A lightweight Retrieval-Augmented Generation (RAG) project built to explore how semantic search and LLMs can work together for customer support automation.

The system retrieves similar historical support tickets using embeddings and uses that context to generate structured outputs such as:
	•	Intent classification
	•	Query summarization
	•	Suggested support responsess

This project focuses on understanding the full RAG workflow end-to-end without introducing unnecessary infrastructure or complexity.



What This Project Does

Given a customer support query, the pipeline:
	1.	Converts the query into embeddings
	2.	Retrieves the most relevant past tickets using vector similarity
	3.	Injects retrieved context into an LLM prompt
	4.	Generates:
	•	User intent
	•	Short issue summary
	•	Suggested responses

The implementation is intentionally lightweight and designed for experimentation, learning, and rapid iteration.



Features
	•	Semantic search using Sentence Transformers
	•	FAISS-based vector retrieval
	•	Zero-shot and few-shot prompting support
	•	Structured response generation
	•	FastAPI API endpoints
	•	OpenAI integration with fallback mock mode
	•	Basic evaluation utilities for output quality checks



Tech Stack

Component	Technology
Embeddings	Sentence Transformers (all-MiniLM-L6-v2)
Vector Search	FAISS
LLM	OpenAI API / Mock fallback
Backend	FastAPI
Language	Python




How the Pipeline Works

1. Data Ingestion

Loads support tickets from:

data/tickets.csv

The dataset is synthetic for experimentation purposes.



2. Embedding Generation

Each ticket is converted into vector embeddings using:

all-MiniLM-L6-v2

This model is lightweight, fast, and works well for semantic similarity tasks.



3. Retrieval

Embeddings are stored in a FAISS index.

For every incoming query:
	•	The query is embedded
	•	Top-k similar tickets are retrieved
	•	Retrieved examples are passed into the prompt context



4. Prompting Strategies

The project supports two prompting approaches:

Zero-Shot

Directly sends the query to the LLM.

Few-Shot

Adds retrieved support examples into the prompt before generation.

Few-shot prompting generally improves output quality, especially for intent detection and response consistency.



5. Response Generation

The LLM generates:
	•	Intent
	•	Summary
	•	Suggested Support Response

Example:

{
  "intent": "Account Access",
  "summary": "User is unable to log into their account.",
  "response": "Try resetting your password or contact support if the issue continues."
}




Project Structure

.
├── data/          # Dataset + FAISS index
├── src/           # Core RAG pipeline logic
├── app/           # FastAPI application
├── requirements.txt
└── README.md




Installation

Clone the repository and install dependencies:

pip install -r requirements.txt




Running the Application

Start the FastAPI server:

uvicorn app.main:app --reload

Server will run locally at:

http://127.0.0.1:8000




API Endpoints

Retrieve Similar Tickets

Returns top-k semantically similar support tickets.

GET /retrieve?q=your_query&k=3

Example

GET /retrieve?q=password reset&k=3




Run Full RAG Pipeline

Generates intent, summary, and response.

GET /predict?q=your_query&strategy=few-shot

Example

GET /predict?q=I can't log in&strategy=few-shot




Evaluation

The project includes lightweight evaluation utilities such as:
	•	Cosine similarity checks
	•	Simple hallucination heuristics
	•	Output consistency validation

These are mainly intended for experimentation rather than production benchmarking.



Design Decisions

Why all-MiniLM-L6-v2?

Chosen because it is:
	•	Fast
	•	Lightweight
	•	Easy to run locally
	•	Good enough for semantic retrieval tasks

Larger embedding models may improve retrieval quality but increase latency and resource usage.



Why FAISS?

FAISS is simple and efficient for small-to-medium scale vector search.

For production-scale systems, a managed vector database such as:
	•	Pinecone
	•	Weaviate
	•	Qdrant
	•	Milvus

would likely be more suitable.



Future Improvements

Possible next steps:
	•	Better evaluation metrics
	•	Streaming responses
	•	Hybrid search (BM25 + embeddings)
	•	Reranking models
	•	Conversation memory
	•	Real support ticket datasets
	•	Docker deployment
	•	Vector DB integration



Learning Goals Behind This Project

This project was primarily built to understand:
	•	How RAG pipelines work internally
	•	Embedding-based retrieval
	•	Prompt engineering tradeoffs
	•	Few-shot vs zero-shot generation
	•	LLM orchestration patterns

The focus is more on clarity and experimentation than production optimization.



Example Workflow

Input

GET /predict?q=I can't log in

Output

{
  "intent": "Account Access",
  "summary": "User is unable to log into their account.",
  "response": "Try resetting your password or contact support if the issue continues."
}


⸻

License

MIT License
