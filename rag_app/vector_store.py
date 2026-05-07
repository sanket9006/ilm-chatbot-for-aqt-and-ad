import faiss
import numpy as np
from typing import List, Dict

class VectorStore:
    def __init__(self, dimension: int):
        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(dimension)
        # Store metadata separately to retrieve after search
        self.metadata = []

    def add_chunks(self, embeddings: np.ndarray, chunks: List[Dict]):
        """
        Adds embeddings and their corresponding chunk data to the store.
        """
        self.index.add(embeddings)
        self.metadata.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Dict]:
        """
        Searches for the most similar chunks.
        """
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx != -1:  # FAISS returns -1 if no result found
                results.append(self.metadata[idx])
                
        return results
