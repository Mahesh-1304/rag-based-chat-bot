import re
from typing import List, Dict

def clean_text(documents: List[Dict]) -> List[Dict]:
    """
    Cleans the text in a list of document dictionaries.
    Removes headers, footers, page numbers, extra whitespace.
    """
    cleaned_docs = []

    for doc in documents:
        text = doc["text"]

        # Remove page numbers like 'Page 1 of 10' or '1/10'
        text = re.sub(r'Page\s*\d+\s*(of\s*\d+)?', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\b\d+/\d+\b', '', text)

        # Remove multiple newlines/tabs and normalize spaces
        text = re.sub(r'\s+', ' ', text)

        # Strip leading/trailing spaces
        text = text.strip()

        # Optional: lowercase
        # text = text.lower()

        cleaned_docs.append({
            **doc,
            "text": text
        })

    return cleaned_docs


if __name__ == "__main__":
    # Quick test
    from document_loader import load_documents
    docs = load_documents("data/raw_docs")
    cleaned = clean_text(docs)
    print(f"Cleaned {len(cleaned)} documents")
    print(cleaned[0]["text"][:300])  # preview first 300 chars
