import faiss
import numpy as np
import pandas as pd
import os

class VectorStore:
    def __init__(self, dimension: int):
        """Initialize FAISS index."""
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata = pd.DataFrame()

    def add_documents(self, embeddings: np.ndarray, metadata: pd.DataFrame):
        """Add embeddings and metadata to the store."""
        self.index.add(embeddings.astype("float32"))
        self.metadata = pd.concat([self.metadata, metadata], ignore_index=True)

    def search(self, query_embedding: np.ndarray, k: int = 3):
        """Search for top-k similar documents."""
        distances, indices = self.index.search(query_embedding.astype("float32"), k)
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx != -1:
                item = self.metadata.iloc[idx].to_dict()
                item["distance"] = float(distances[0][i])
                results.append(item)
        return results

    def save(self, path: str):
        """Save the FAISS index."""
        faiss.write_index(self.index, f"{path}.index")
        self.metadata.to_pickle(f"{path}.metadata")

    def load(self, path: str):
        """Load the FAISS index."""
        if os.path.exists(f"{path}.index"):
            self.index = faiss.read_index(f"{path}.index")
            self.metadata = pd.read_pickle(f"{path}.metadata")
