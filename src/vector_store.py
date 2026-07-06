import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict
from src.config import FAISS_INDEX_DIR


class VectorStore:
    def __init__(self, dimension: int = None):
        self.index = None
        self.chunks = []
        self.dimension = dimension

    def build_index(self, embeddings: np.ndarray, chunks: List[Dict]):
        self.chunks = chunks
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings.astype(np.float32))
        print(f"Индекс построен. Векторов: {self.index.ntotal}")

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        if self.index is None:
            return []

        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        distances, indices = self.index.search(query_embedding.astype(np.float32), k)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.chunks):
                result = self.chunks[idx].copy()
                result["score"] = 1.0 / (1.0 + float(dist))
                results.append(result)
        return results

    def save_index(self, path: Path = None):
        path = path or FAISS_INDEX_DIR
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        with open(path / "chunks.pkl", 'wb') as f:
            pickle.dump(self.chunks, f)
        with open(path / "metadata.pkl", 'wb') as f:
            pickle.dump({"dimension": self.dimension}, f)

    def load_index(self, path: Path = None):
        path = path or FAISS_INDEX_DIR
        self.index = faiss.read_index(str(path / "index.faiss"))
        with open(path / "chunks.pkl", 'rb') as f:
            self.chunks = pickle.load(f)
        try:
            with open(path / "metadata.pkl", 'rb') as f:
                meta = pickle.load(f)
                self.dimension = meta.get("dimension")
        except:
            self.dimension = self.index.d
        print(f"Индекс загружен. Векторов: {self.index.ntotal}")
        return self.chunks