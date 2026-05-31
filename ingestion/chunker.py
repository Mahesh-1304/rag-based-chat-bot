# ingestion/chunker.py
"""
Text chunking using tiktoken for accurate token counting.
Falls back to whitespace splitting if tiktoken is unavailable.
"""

import uuid
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    # Verify it actually loaded (requires network on first use)
    _enc.encode("test")

    def _tokenize(text: str) -> List[int]:
        return _enc.encode(text)

    def _detokenize(tokens: List[int]) -> str:
        return _enc.decode(tokens)

    logger.info("chunker: using tiktoken for token counting")

except Exception:
    logger.warning("tiktoken unavailable — falling back to whitespace splitting")

    def _tokenize(text: str):   # type: ignore[misc]
        return text.split()     # words act as pseudo-tokens

    def _detokenize(tokens) -> str:   # type: ignore[misc]
        return " ".join(tokens)


def chunk_text(
    documents: List[Dict],
    chunk_size: int = 400,
    overlap: int = 50,
) -> List[Dict]:
    """
    Split documents into overlapping token-based chunks.

    Args:
        documents:  List of {text, source, page} dicts.
        chunk_size: Maximum tokens per chunk.
        overlap:    Number of tokens shared between consecutive chunks.

    Returns:
        List of chunk dicts with chunk_id, text, source, page.
    """
    all_chunks: List[Dict] = []

    for doc in documents:
        tokens = _tokenize(doc["text"])
        start = 0

        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_str = _detokenize(chunk_tokens)

            chunk_id = (
                f"{doc['source']}_{doc.get('page', 'NA')}_{uuid.uuid4().hex[:6]}"
            )
            all_chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_str,
                "source": doc["source"],
                "page": doc.get("page"),
            })

            if end == len(tokens):
                break  # avoid infinite loop on last chunk

            start += chunk_size - overlap

    return all_chunks


if __name__ == "__main__":
    import logging
    from pathlib import Path
    logging.basicConfig(level=logging.INFO)
    from ingestion.text_cleaner import clean_text
    from ingestion.document_loader import load_documents

    docs, _ = load_documents(Path("data/raw_docs"))
    cleaned = clean_text(docs)
    chunks = chunk_text(cleaned)
    print(f"Created {len(chunks)} chunks")
    if chunks:
        print(chunks[0])