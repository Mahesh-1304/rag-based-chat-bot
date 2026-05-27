import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_DIM = 384  # MiniLM output size

CHUNKS_PATH = "data/processed_docs/chunks.json"
VECTOR_STORE_PATH = "embeddings/vector_store"


def load_chunks():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    Path(VECTOR_STORE_PATH).mkdir(parents=True, exist_ok=True)

    chunks = load_chunks()
    texts = [c["text"] for c in chunks]

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True
    )

    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(VECTOR_DIM)
    index.add(embeddings)

    faiss.write_index(index, f"{VECTOR_STORE_PATH}/index.faiss")

    # save metadata separately
    with open(f"{VECTOR_STORE_PATH}/metadata.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print(f"✅ Stored {len(chunks)} chunks in FAISS")


if __name__ == "__main__":
    main()
