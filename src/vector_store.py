import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict
from src.config import FAISS_INDEX_DIR

class VectorStore:
    def __init__(self):
        self.index = None
        self.chunks = []

    def build_index(self, embeddings: np.ndarray, chunks: List[Dict]):
        self.chunks = chunks
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings.astype(np.float32))
        print(f"Index built. Vectors: {self.index.ntotal}")

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        if self.index is None:
            return []
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

    def load_index(self, path: Path = None):
        path = path or FAISS_INDEX_DIR
        self.index = faiss.read_index(str(path / "index.faiss"))
        with open(path / "chunks.pkl", 'rb') as f:
            self.chunks = pickle.load(f)
        print(f"Index loaded. Vectors: {self.index.ntotal}")
        return self.chunks