import fitz
from pathlib import Path
from src.config import PDF_PATH

class PDFProcessor:
    def __init__(self, pdf_path=None):
        if pdf_path is None:
            self.pdf_path = PDF_PATH
        else:
            self.pdf_path = Path(pdf_path) if isinstance(pdf_path, (str, Path)) else pdf_path
        self.doc = None

    def load_pdf(self):
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF не найден: {self.pdf_path}")
        self.doc = fitz.open(str(self.pdf_path))
        print(f"Загружено {len(self.doc)} страниц")
        return self.doc

    def get_page_count(self):
        if not self.doc:
            self.load_pdf()
        return len(self.doc)

    def extract_text(self, page_num=None):
        if not self.doc:
            self.load_pdf()
        if page_num is not None:
            return self.doc[page_num].get_text()
        full_text = ""
        for p in range(len(self.doc)):
            text = self.doc[p].get_text()
            if text.strip():
                full_text += f"\n--- Страница {p + 1} ---\n"
                full_text += text
        return full_text

    def close(self):
        if self.doc:
            self.doc.close()