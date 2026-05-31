# ingestion/text_cleaner.py
"""
Text cleaning utilities for RAG ingestion pipeline.
"""

import re
from typing import List, Dict


# Common PDF encoding artifacts
REPLACEMENTS = {
    "â€™": "'",
    "â€˜": "'",
    "â€œ": '"',
    "â€\x9d": '"',
    "â€“": "-",
    "â€”": "--",
    "Â ": " ",
}


def clean_text(documents: List[Dict]) -> List[Dict]:
    """
    Cleans text from loaded documents.

    - Fixes common PDF encoding artifacts
    - Removes 'Page N' and 'Page N of M' patterns
    - Normalizes whitespace
    - Preserves dates, fractions, and version numbers
    """

    cleaned_docs = []

    for doc in documents:
        text = doc.get("text", "")

        # Fix PDF encoding artifacts
        for bad, good in REPLACEMENTS.items():
            text = text.replace(bad, good)

        # Remove page markers
        text = re.sub(
            r"\bPage\s*\d+(\s*of\s*\d+)?\b",
            "",
            text,
            flags=re.IGNORECASE,
        )

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove leading/trailing spaces
        text = text.strip()

        if text:
            cleaned_docs.append({
                **doc,
                "text": text,
            })

    return cleaned_docs


if __name__ == "__main__":
    import logging
    from pathlib import Path

    logging.basicConfig(level=logging.INFO)

    from ingestion.document_loader import load_documents

    docs, _ = load_documents(Path("data/raw_docs"))

    print(f"Loaded {len(docs)} documents")

    cleaned = clean_text(docs)

    print(f"Cleaned {len(cleaned)} documents")

    if cleaned:
        print("\nSample text:")
        print(cleaned[0]["text"][:300])