# LLM-Powered Support Query Understanding (RAG)

A minimal but realistic implementation of a Retrieval-Augmented Generation (RAG) pipeline for customer support tickets using Python, FAISS, and OpenAI.

## Architecture
The system follows a classic RAG architecture:
1.  **Ingestion**: Synthetic support tickets are loaded from `data/tickets.csv`.
2.  **Embeddings**: Text is vectorized using `sentence-transformers` (`all-MiniLM-L6-v2`).
3.  **Retrieval**: Vectors are stored in a `FAISS` index for fast similarity search.
4.  **Prompt Engineering**: Supports Zero-shot and Few-shot (retrieved examples) strategies.
5.  **LLM Pipeline**: Generates intent, summary, and response suggestions.
6.  **Evaluation**: Includes cosine similarity and basic hallucination detection.

## Project Structure
- `data/`: CSV data and FAISS index storage.
- `src/`: Moduloar logic for embeddings, retrieval, prompts, etc.
- `app/`: FastAPI application.
- `requirements.txt`: Project dependencies.

## Setup
1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Set up environment variables (Optional):
    Create a `.env` file and add:
    ```env
    OPENAI_API_KEY=your_key_here
    ```
    *If no key is provided, the system defaults to a Mock mode for demonstration.*

3.  Run the API:
    ```bash
    uvicorn app.main:app --reload
    ```

## API Endpoints
- `GET /retrieve?q=query&k=3`: Returns top-k similar tickets.
- `GET /predict?q=query&strategy=few-shot`: Returns intent, summary, and response suggestion.

## Tradeoffs
- **Latency vs Accuracy**: Using a local small embedding model (`all-MiniLM-L6-v2`) provides low latency but may be less nuanced than larger models (e.g., `text-embedding-3-small`).
- **FAISS (CPU)**: extremely fast for small datasets, but doesn't handle real-time metadata updates as easily as managed vector databases.

## Sample Input/Output
**Input**: `GET /predict?q=I can't log in`
**Output** (Structured JSON):
```json
{
  "query": "I can't log in",
  "prediction": {
    "intent": "Account Access",
    "summary": "User is unable to log into their account.",
    "response": "Reset your password via the login page or contact support if the issue persists."
  }
}
```
