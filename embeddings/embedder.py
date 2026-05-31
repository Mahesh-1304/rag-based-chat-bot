# embeddings/embedder.py
"""
Generates embeddings and builds a FAISS index.
Respects VECTOR_STORE_TYPE setting: flat | hnsw | ivf
"""

import json
import logging
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

from config.settings import settings

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CHUNKS_PATH = settings.PROCESSED_DOCS_DIR / "chunks.json"
VECTOR_STORE_PATH = settings.VECTOR_STORE_DIR


def load_chunks():
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Chunks file not found at {CHUNKS_PATH}. "
            "Run the ingestion pipeline first."
        )
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS index according to settings.VECTOR_STORE_TYPE.
    Supported types: flat, hnsw, ivf
    """
    dim = embeddings.shape[1]
    store_type = settings.VECTOR_STORE_TYPE.lower()

    if store_type == "hnsw":
        index = faiss.IndexHNSWFlat(dim, settings.HNSW_M)
        logger.info(f"Building HNSW index (M={settings.HNSW_M})")
    elif store_type == "ivf":
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, settings.IVF_NLIST)
        logger.info(f"Building IVF index (nlist={settings.IVF_NLIST})")
        index.train(embeddings)
    else:
        # Default: flat L2
        index = faiss.IndexFlatL2(dim)
        logger.info("Building Flat L2 index")

    return index


def main():
    VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)

    chunks = load_chunks()
    texts = [c["text"] for c in chunks]
    logger.info(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")

    logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)

    logger.info("Generating embeddings...")
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype="float32")
    logger.info(f"Embeddings shape: {embeddings.shape}")

    index = build_index(embeddings)
    index.add(embeddings)

    index_file = VECTOR_STORE_PATH / "index.faiss"
    meta_file = VECTOR_STORE_PATH / "metadata.json"

    faiss.write_index(index, str(index_file))
    logger.info(f"FAISS index saved to {index_file}")

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)
    logger.info(f"Metadata saved to {meta_file}")

    print(f"✅ Stored {len(chunks)} chunks in FAISS ({settings.VECTOR_STORE_TYPE} index)")


if __name__ == "__main__":
    main()