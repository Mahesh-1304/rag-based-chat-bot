# ingestion/document_loader.py
"""
Loads PDF and DOCX documents from a directory.
Returns a tuple (documents, errors) so callers can handle per-file failures.
"""

import logging
from pathlib import Path
from typing import List, Dict, Tuple

from pypdf import PdfReader
import docx

logger = logging.getLogger(__name__)


def load_pdf(file_path: Path) -> List[Dict]:
    """Extract text page-by-page from a PDF file."""
    documents = []
    reader = PdfReader(file_path)

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            documents.append({
                "text": text,
                "source": file_path.name,
                "page": page_num + 1,
            })
    return documents


def load_docx(file_path: Path) -> List[Dict]:
    """Extract full text from a DOCX file as a single document section."""
    doc = docx.Document(file_path)
    full_text = [para.text for para in doc.paragraphs if para.text.strip()]

    if not full_text:
        return []

    return [{
        "text": "\n".join(full_text),
        "source": file_path.name,
        "page": None,
    }]


def load_documents(data_dir: Path) -> Tuple[List[Dict], List[Dict]]:
    """
    Load all PDF and DOCX files from a directory.

    Returns:
        (documents, errors)
        - documents: list of {text, source, page} dicts
        - errors:    list of {file, error} dicts for files that failed
    """
    data_dir = Path(data_dir)
    all_docs: List[Dict] = []
    errors: List[Dict] = []

    supported_files = [
        f for f in data_dir.iterdir()
        if f.suffix.lower() in (".pdf", ".docx")
    ]

    if not supported_files:
        logger.warning(f"No PDF or DOCX files found in {data_dir}")
        return [], []

    for file_path in supported_files:
        try:
            if file_path.suffix.lower() == ".pdf":
                docs = load_pdf(file_path)
            else:
                docs = load_docx(file_path)

            if docs:
                all_docs.extend(docs)
                logger.info(f"Loaded {len(docs)} section(s) from {file_path.name}")
            else:
                logger.warning(f"No extractable text in {file_path.name}")

        except Exception as e:
            logger.error(f"Failed to load {file_path.name}: {e}")
            errors.append({"file": file_path.name, "error": str(e)})

    return all_docs, errors


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    docs, errs = load_documents(Path("data/raw_docs"))
    print(f"Loaded {len(docs)} document sections, {len(errs)} errors")