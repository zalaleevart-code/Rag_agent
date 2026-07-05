import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List
from src.config import EMBEDDING_MODEL

class Embedder:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if Embedder._model is None:
            print(f"Loading {EMBEDDING_MODEL}...")
            Embedder._model = SentenceTransformer(EMBEDDING_MODEL)
            print("Ready")

    @property
    def model(self):
        return Embedder._model

    def encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def encode_query(self, query: str) -> np.ndarray:
        return self.encode([query])