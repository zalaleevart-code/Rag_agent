import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.agentic_rag import AgenticRAG
from src.config import PDF_PATH, LLM_MODEL, FAISS_INDEX_DIR, OUTPUT_DIR, EMBEDDING_MODEL

USER_ID = "main"

def main():
    print("=" * 50)
    print("ИИ-консультант по ALINA GPSS")
    print("=" * 50)

    try:
        import ollama
        ollama.list()
        print("Ollama запущен")
    except:
        print("Ollama не запущен!")
        print("Запустите: ollama serve")
        return

    if not PDF_PATH.exists():
        print(f"PDF не найден: {PDF_PATH}")
        print("Положите PDF в папку data/")
        return

    if not (FAISS_INDEX_DIR / "index.faiss").exists():
        print("Индекс не найден. Запустите create_index.py сначала!")
        return

    print(f"Модель эмбеддингов: {EMBEDDING_MODEL}")
    print(f"LLM модель: {LLM_MODEL}")

    try:
        rag = AgenticRAG()
        rag.load_index()
    except Exception as e:
        print(f"Ошибка загрузки индекса: {e}")
        return

    print("\nВведите 'exit' для выхода, '/clear' для очистки памяти\n")

    while True:
        try:
            q = input("\nВопрос: ").strip()
            if q.lower() in ['exit', 'quit']:
                break
            if q == '/clear':
                rag.memory.clear_user(USER_ID)
                rag.memory.save_to_file()
                print("Память очищена")
                continue
            if not q:
                continue

            answer = rag.answer(q, USER_ID)

            print("\n" + "=" * 50)
            print(answer)
            print("=" * 50)

        except KeyboardInterrupt:
            rag.memory.save_to_file()
            print("\nДо свидания!")
            break
        except Exception as e:
            print(f"Ошибка: {e}")

if __name__ == "__main__":
    main()