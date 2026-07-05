import ollama
import re
from typing import List, Dict
from src.embedder import Embedder
from src.vector_store import VectorStore
from src.memory import MemoryManager
from src.config import LLM_MODEL, MAX_ITERATIONS, CHUNKS_PER_SEARCH, MAX_CHUNKS_TOTAL


class AgenticRAG:
    def __init__(self):
        self.embedder = None
        self.vector_store = None
        self.llm_model = LLM_MODEL
        self.max_iterations = MAX_ITERATIONS
        self.chunks_per_search = CHUNKS_PER_SEARCH
        self.max_chunks_total = MAX_CHUNKS_TOTAL
        self.context = []
        self.memory = MemoryManager()
        self.memory.load_from_file()

    def _get_embedder(self):
        if self.embedder is None:
            self.embedder = Embedder()
        return self.embedder

    def load_index(self):
        self.vector_store = VectorStore()
        self.vector_store.load_index()
        print("Готов к работе")

    def _search(self, query: str) -> List[Dict]:
        embedder = self._get_embedder()
        q_emb = embedder.encode_query(query)
        return self.vector_store.search(q_emb, self.chunks_per_search)

    def _keyword_search(self, query: str) -> List[Dict]:
        words = re.findall(r'[а-яa-z0-9]{3,}', query.lower())
        stop = {'это', 'что', 'как', 'для', 'на', 'в', 'с', 'по', 'из', 'от', 'до', 'за'}
        keywords = [w for w in words if w not in stop]

        if not keywords:
            return []

        scored = []
        for chunk in self.vector_store.chunks:
            text = chunk['text'].lower()
            score = sum(text.count(kw) for kw in keywords)
            if score:
                c = chunk.copy()
                c['score'] = min(0.5 + score * 0.01, 1.0)
                scored.append(c)

        scored.sort(key=lambda x: x['score'], reverse=True)
        return scored[:self.chunks_per_search]

    def _hybrid_search(self, query: str) -> List[Dict]:
        sem = self._search(query)
        kw = self._keyword_search(query)

        merged = {c['id']: c for c in sem}
        for c in kw:
            if c['id'] in merged:
                merged[c['id']]['score'] = min(merged[c['id']].get('score', 0) + 0.2, 1.0)
            else:
                merged[c['id']] = c

        return sorted(merged.values(), key=lambda x: x.get('score', 0), reverse=True)[:self.chunks_per_search]

    def _add_context(self, chunks: List[Dict]) -> int:
        existing = {c['id'] for c in self.context}
        added = 0
        for c in chunks:
            if c['id'] not in existing and len(self.context) < self.max_chunks_total:
                self.context.append(c)
                existing.add(c['id'])
                added += 1
        return added

    def _decide(self, query: str, step: int) -> Dict:
        ctx = ""
        for i, c in enumerate(self.context[:3], 1):
            ctx += f"{i}. {c['text'][:80]}...\n"

        prompt = f"""Ты агент поиска по GPSS Studio.
Вопрос: {query}
Найдено: {len(self.context)} чанков, шаг {step}/{self.max_iterations}
Контекст: {ctx}
Решение: STOP если достаточно, REFINE если уточнить, SEARCH_MORE если искать другое.
Формат: РЕШЕНИЕ: [STOP|REFINE|SEARCH_MORE]
ЗАПРОС: [новый запрос]
ПРИЧИНА: [почему]"""

        try:
            resp = ollama.chat(model=self.llm_model, messages=[{"role": "user", "content": prompt}])
            ans = resp['message']['content']

            dec, q, reason = "STOP", query, ""
            for line in ans.split('\n'):
                line = line.strip()
                if line.startswith('РЕШЕНИЕ:'):
                    dec = line.replace('РЕШЕНИЕ:', '').strip()
                elif line.startswith('ЗАПРОС:'):
                    q = line.replace('ЗАПРОС:', '').strip()
                elif line.startswith('ПРИЧИНА:'):
                    reason = line.replace('ПРИЧИНА:', '').strip()

            return {"decision": dec if dec in ['STOP','REFINE','SEARCH_MORE'] else 'STOP',
                    "new_query": q if q and len(q) < 200 else query,
                    "reason": reason}
        except:
            return {"decision": "STOP", "new_query": query, "reason": "error"}

    def _generate(self, query: str, user_id: str = None) -> str:
        hist = ""
        if user_id:
            h = self.memory.get_history(user_id)
            if h:
                hist = "\n".join([f"{m['role']}: {m['content'][:200]}" for m in h[-4:]])

        ctx = ""
        for c in self.context[:8]:
            ctx += c['text'] + "\n"

        prompt = f"""Ты эксперт ALINA GPSS. Отвечай ТОЛЬКО на русском.

ПРАВИЛА:
1. Если даешь инструкцию — ОБЯЗАТЕЛЬНО нумеруй шаги (ШАГ 1, ШАГ 2, ШАГ 3...)
2. Если пользователь спрашивает "шаг N" — используй ИСТОРИЮ, чтобы понять, о какой инструкции идет речь

История:
{hist}

Информация из документации:
{ctx}

Вопрос: {query}

Ответ:"""

        try:
            resp = ollama.chat(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "Ты русскоязычный эксперт. Отвечай только на русском. Если даешь инструкцию — нумеруй шаги."},
                    {"role": "user", "content": prompt}
                ]
            )
            return resp['message']['content']
        except Exception as e:
            return f"Ошибка: {e}"

    def answer(self, query: str, user_id: str = "default") -> str:
        self.memory.add_message(user_id, "user", query)
        self.context = []
        current = query

        for step in range(1, self.max_iterations + 1):
            chunks = self._hybrid_search(current)
            self._add_context(chunks)

            if not self.context:
                break

            dec = self._decide(query, step)
            if dec['decision'] == 'STOP':
                break
            elif dec['decision'] in ['REFINE', 'SEARCH_MORE']:
                if dec['new_query'] == current:
                    break
                current = dec['new_query']

        ans = self._generate(query, user_id)
        self.memory.add_message(user_id, "assistant", ans)
        self.memory.save_to_file()

        return ans