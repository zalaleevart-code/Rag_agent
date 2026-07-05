import sys
import ollama
from src.agentic_rag import AgenticRAG
from src.config import PDF_PATH, LLM_MODEL, FAISS_INDEX_DIR, OUTPUT_DIR

USER_ID = "main"

def main():
    print("=" * 50)
    print("ИИ-консультант по ALINA GPSS")
    print("=" * 50)

    try:
        ollama.list()
    except:
        print("Ollama не запущен!")
        return

    if not PDF_PATH.exists():
        print(f"PDF не найден: {PDF_PATH}")
        return

    if not (FAISS_INDEX_DIR / "index.faiss").exists() or not (OUTPUT_DIR / "chunks.json").exists():
        print("Запустите create_index.py сначала!")
        return

    rag = AgenticRAG()
    rag.load_index()

    print(f"\nМодель: {LLM_MODEL}")
    print("Введите 'exit' для выхода, '/clear' для очистки памяти\n")

    while True:
        try:
            q = input("\nВопрос: ").strip()
            if q.lower() in ['exit','quit']:
                break
            if q == '/clear':
                rag.memory.clear_user(USER_ID)
                print("Память очищена")
                continue
            if not q:
                continue

            answer = rag.answer(q, USER_ID)

            print("\n" + "=" * 50)
            print(answer)
            print("=" * 50)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Ошибка: {e}")

if __name__ == "__main__":
    main()