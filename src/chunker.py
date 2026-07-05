import json
import re
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import CHUNK_SIZE, CHUNK_OVERLAP, OUTPUT_DIR
from src.pdf_processor import PDFProcessor

class Chunker:
    def __init__(self, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def detect_chunk_type(self, text: str) -> str:
        patterns = {
            'concept': r'(термин|определение|понятие|называется|это)',
            'instruction': r'(как|создать|сделать|шаг|последовательность|алгоритм)',
            'example': r'(пример|допустим|рассмотрим|рис\.|табл\.)',
            'reference': r'(состоит|включает|содержит|представляет|является)'
        }
        for t, p in patterns.items():
            if re.search(p, text.lower(), re.I):
                return t
        return 'reference'

    def extract_keywords(self, text: str, chapter: str = "", section: str = "") -> List[str]:
        keywords = set()
        if chapter:
            keywords.add(chapter.replace("Глава", "").strip())
        if section:
            keywords.add(section.strip())

        patterns = [
            r'\b(ТЭБ|транзакт|блок|модель|схема)\b',
            r'\b(GPSS|GPSS Studio|GPSS World Core)\b',
            r'\b(композитный|элементарный|иерархия)\b',
            r'\b(отладка|отладчик|ошибка)\b',
            r'\b(проект|анимация|отчет|фактор|показатель)\b',
        ]
        for p in patterns:
            matches = re.findall(p, text, re.I)
            keywords.update([m.lower() for m in matches])

        blocks = ['GENERATE', 'ADVANCE', 'SEIZE', 'RELEASE', 'QUEUE',
                  'DEPART', 'ENTER', 'LEAVE', 'TERMINATE', 'TRANSFER']
        for b in blocks:
            if b in text.upper():
                keywords.add(b.lower())

        return list(keywords)

    def create_chunks(self, pdf_path=None) -> List[Dict]:
        processor = PDFProcessor(pdf_path)
        doc = processor.doc
        if not doc:
            processor.load_pdf()
            doc = processor.doc

        chunks = []
        chapter = "Введение"
        section = "Общее"

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if not text.strip():
                continue

            lines = text.split('\n')
            for line in lines[:10]:
                if re.search(r'^Глава\s+\d+\.', line):
                    chapter = line.strip()
                elif re.search(r'^\d+\.\d+\.', line):
                    section = line.strip()

            page_chunks = self.text_splitter.split_text(text)
            for i, chunk_text in enumerate(page_chunks):
                if len(chunk_text.strip()) < 50:
                    continue

                chunks.append({
                    "id": f"ch_{page_num+1:03d}_{i:03d}",
                    "text": chunk_text,
                    "metadata": {
                        "chapter": chapter,
                        "section": section,
                        "page": page_num + 1,
                        "type": self.detect_chunk_type(chunk_text),
                        "keywords": self.extract_keywords(chunk_text, chapter, section),
                    }
                })

        processor.close()
        print(f"Created {len(chunks)} chunks")
        return chunks

    def save_chunks(self, chunks: List[Dict], output_path=None):
        if output_path is None:
            output_path = OUTPUT_DIR / "chunks.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        print(f"Saved chunks to {output_path}")
        return output_path

    def load_chunks(self, input_path=None):
        if input_path is None:
            input_path = OUTPUT_DIR / "chunks.json"
        if not input_path.exists():
            return []
        with open(input_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        print(f"Loaded {len(chunks)} chunks")
        return chunks