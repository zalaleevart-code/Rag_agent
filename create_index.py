import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.chunker import Chunker
from src.embedder import Embedder
from src.vector_store import VectorStore
from src.config import PDF_PATH, OUTPUT_DIR, FAISS_INDEX_DIR


def create_index():
    print("=" * 60)
    print("Создание индекса для Agentic RAG")
    print("=" * 60)

    if not PDF_PATH.exists():
        print(f"PDF не найден: {PDF_PATH}")
        print("Положите PDF в папку data/")
        return

    print("\n1. Разбиение PDF на чанки...")
    chunker = Chunker()
    chunks = chunker.create_chunks()
    chunker.save_chunks(chunks)

    print("\n2. Векторизация чанков...")
    embedder = Embedder()
    embeddings = embedder.encode_chunks(chunks)

    print("\n3. Построение FAISS индекса...")
    dimension = embedder.get_dimension()
    vector_store = VectorStore(dimension)
    vector_store.build_index(embeddings, chunks)
    vector_store.save_index()

    print("\n" + "=" * 60)
    print("ГОТОВО!")
    print(f"  Чанков: {len(chunks)}")
    print(f"  Размерность: {dimension}")
    print(f"  Индекс: {FAISS_INDEX_DIR / 'index.faiss'}")
    print("=" * 60)


if __name__ == "__main__":
    create_index()