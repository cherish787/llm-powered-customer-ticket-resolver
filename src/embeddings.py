from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingEngine:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the sentence transformer model."""
        self.model = SentenceTransformer(model_name)

    def get_embeddings(self, texts: list) -> np.ndarray:
        """Generate embeddings for a list of strings."""
        return self.model.encode(texts, convert_to_numpy=True)

if __name__ == "__main__":
    # Test embeddings
    engine = EmbeddingEngine()
    test_texts = ["How to reset password?", "I want a refund"]
    embeddings = engine.get_embeddings(test_texts)
    print(f"Embedding shape: {embeddings.shape}")
