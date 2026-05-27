from pathlib import Path
from typing import List, Dict

from pypdf import PdfReader
import docx


def load_pdf(file_path: Path) -> List[Dict]:
    documents = []
    reader = PdfReader(file_path)

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            documents.append({
                "text": text,
                "source": file_path.name,
                "page": page_num + 1
            })
    return documents


def load_docx(file_path: Path) -> List[Dict]:
    documents = []
    doc = docx.Document(file_path)

    full_text = []
    for para in doc.paragraphs:
        if para.text.strip():
            full_text.append(para.text)

    documents.append({
        "text": "\n".join(full_text),
        "source": file_path.name,
        "page": None
    })
    return documents


def load_documents(data_dir: Path) -> List[Dict]:
    all_docs = []

    for file_path in data_dir.iterdir():
        if file_path.suffix.lower() == ".pdf":
            all_docs.extend(load_pdf(file_path))

        elif file_path.suffix.lower() == ".docx":
            all_docs.extend(load_docx(file_path))

    return all_docs


if __name__ == "__main__":
    docs = load_documents(Path("data/raw_docs"))
    print(f"Loaded {len(docs)} document sections")
