from sklearn.metrics.pairwise import cosine_similarity
from src.embeddings import EmbeddingEngine

class Evaluator:
    def __init__(self, engine: EmbeddingEngine):
        self.engine = engine

    def get_similarity(self, text_a: str, text_b: str) -> float:
        """Calculate cosine similarity between two strings."""
        emb_a = self.engine.get_embeddings([text_a])
        emb_b = self.engine.get_embeddings([text_b])
        return float(cosine_similarity(emb_a, emb_b)[0][0])

    def detect_hallucination(self, generated_response: str, context_tickets: list) -> bool:
        """
        Basic heuristic for hallucination detection.
        Checks if the response length is excessive or if it lacks keywords from the context.
        """
        if not context_tickets:
            return False
        
        # Heuristic 1: Length check (response shouldn't be 3x longer than context snippets)
        context_text = " ".join([c.get("resolution", "") for c in context_tickets])
        if len(generated_response) > len(context_text) * 3:
            return True
        
        # Heuristic 2: Keyword overlap (very basic)
        keywords = set(context_text.lower().split())
        gen_words = set(generated_response.lower().split())
        overlap = gen_words.intersection(keywords)
        
        if len(keywords) > 0 and len(overlap) == 0:
            return True
            
        return False
