import uuid
from typing import List, Dict

# Simple tokenizer example using whitespace
def tokenize(text: str) -> List[str]:
    return text.split()

def detokenize(tokens: List[str]) -> str:
    return " ".join(tokens)

def chunk_text(documents: List[Dict], chunk_size: int = 400, overlap: int = 50) -> List[Dict]:
    """
    Splits documents into chunks of `chunk_size` tokens with `overlap`.
    """
    all_chunks = []

    for doc in documents:
        tokens = tokenize(doc["text"])
        start = 0
        while start < len(tokens):
            end = start + chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text_str = detokenize(chunk_tokens)

            chunk_id = f"{doc['source']}_{doc.get('page', 'NA')}_{uuid.uuid4().hex[:6]}"
            all_chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text_str,
                "source": doc["source"],
                "page": doc.get("page")
            })

            start += chunk_size - overlap  # move forward with overlap

    return all_chunks


if __name__ == "__main__":
    from text_cleaner import clean_text
    from document_loader import load_documents

    docs = load_documents("data/raw_docs")
    cleaned = clean_text(docs)
    chunks = chunk_text(cleaned)
    print(f"Created {len(chunks)} chunks")
    print(chunks[0])
