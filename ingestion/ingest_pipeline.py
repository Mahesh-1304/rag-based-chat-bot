import json
from pathlib import Path
from document_loader import load_documents
from text_cleaner import clean_text
from chunker import chunk_text

def run_ingestion(input_dir: str = "data/raw_docs",
                  output_file: str = "data/processed_docs/chunks.json"):
    Path("data/processed_docs").mkdir(exist_ok=True)

    # 1️⃣ Load documents
    raw_docs = load_documents(Path(input_dir))
    print(f"Loaded {len(raw_docs)} raw document sections")

    # 2️⃣ Clean text
    cleaned_docs = clean_text(raw_docs)
    print(f"Cleaned {len(cleaned_docs)} document sections")

    # 3️⃣ Chunk text
    chunks = chunk_text(cleaned_docs)
    print(f"Created {len(chunks)} chunks")

    # 4️⃣ Save to JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved chunks to {output_file}")


if __name__ == "__main__":
    run_ingestion()
