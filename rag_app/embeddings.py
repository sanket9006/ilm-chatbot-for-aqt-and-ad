from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class EmbeddingModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Load the pre-trained model
        self.model = SentenceTransformer(model_name)

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Converts a list of strings into embeddings (vectors).
        """
        embeddings = self.model.encode(texts)
        return np.array(embeddings).astype('float32')

    def get_query_embedding(self, query: str) -> np.ndarray:
        """
        Converts a single query string into an embedding.
        """
        embedding = self.model.encode([query])
        return np.array(embedding).astype('float32')
