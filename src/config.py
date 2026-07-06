from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
FAISS_INDEX_DIR = OUTPUT_DIR / "faiss_index"

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
FAISS_INDEX_DIR.mkdir(exist_ok=True)

PDF_PATH = DATA_DIR / "Руководство_пользователя_GPSS_STUDIO_v2025.pdf"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5:7b"

MAX_ITERATIONS = 2
CHUNKS_PER_SEARCH = 5
MAX_CHUNKS_TOTAL = 10

TOP_K = 5