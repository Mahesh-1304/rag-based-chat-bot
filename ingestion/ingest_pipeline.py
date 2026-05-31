# ingestion/ingest_pipeline.py
"""
Simple ingestion entry point: Load → Clean → Chunk → Save.
Run from the project root: python -m ingestion.ingest_pipeline
"""

import json
import logging
from pathlib import Path

from ingestion.document_loader import load_documents
from ingestion.text_cleaner import clean_text
from ingestion.chunker import chunk_text

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_ingestion(
    input_dir: str = "data/raw_docs",
    output_file: str = "data/processed_docs/chunks.json",
):
    Path("data/processed_docs").mkdir(parents=True, exist_ok=True)

    # 1. Load documents
    raw_docs, errors = load_documents(Path(input_dir))
    if errors:
        logger.warning(f"{len(errors)} file(s) failed to load: {errors}")
    if not raw_docs:
        logger.error("No documents loaded. Aborting.")
        return
    logger.info(f"Loaded {len(raw_docs)} raw document sections")

    # 2. Clean text
    cleaned_docs = clean_text(raw_docs)
    logger.info(f"Cleaned {len(cleaned_docs)} document sections")

    # 3. Chunk text
    chunks = chunk_text(cleaned_docs)
    logger.info(f"Created {len(chunks)} chunks")

    # 4. Save to JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved chunks to {output_file}")


if __name__ == "__main__":
    run_ingestion()