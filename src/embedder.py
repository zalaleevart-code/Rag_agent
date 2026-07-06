import numpy as np
import ollama
from typing import List, Dict
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
            print(f"Используется модель эмбеддингов Ollama: {EMBEDDING_MODEL}")
            Embedder._model = EMBEDDING_MODEL
            try:
                ollama.list()
                print("Ollama готов к работе")
            except Exception as e:
                print(f"Ошибка Ollama: {e}")

    @property
    def model(self):
        return Embedder._model

    def encode(self, texts: List[str]) -> np.ndarray:
        embeddings = []
        for text in texts:
            try:
                response = ollama.embeddings(
                    model=self.model,
                    prompt=text
                )
                embedding = response['embedding']
                embeddings.append(embedding)
            except Exception as e:
                print(f"Ошибка кодирования текста: {e}")
                embeddings.append([0.0] * 768)

        return np.array(embeddings, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        result = self.encode([query])
        return result[0]

    def encode_chunks(self, chunks: List[Dict]) -> np.ndarray:
        texts = [chunk['text'] for chunk in chunks]
        return self.encode(texts)

    def get_dimension(self) -> int:
        test = self.encode(["тест"])
        return test.shape[1]