# retrieval/retriever.py
"""
Vector-based document retriever using FAISS and Sentence Transformers.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np

try:
    import faiss
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise ImportError(
        "Required packages not installed. "
        "Run: pip install faiss-cpu sentence-transformers"
    ) from e

logger = logging.getLogger(__name__)


class Retriever:
    """
    Retrieves relevant document chunks based on semantic similarity.

    Uses FAISS for efficient vector search and Sentence Transformers
    for embedding generation.
    """

    def __init__(
        self,
        index_path: str,
        metadata_path: str,
        model_name: str = "all-MiniLM-L6-v2",
        top_k: int = 5,
        score_threshold: float = 0.5,
    ):
        """
        Initialize the retriever.

        Args:
            index_path:      Path to FAISS index file.
            metadata_path:   Path to metadata JSON file.
            model_name:      Name of the embedding model.
            top_k:           Number of results to retrieve.
            score_threshold: Minimum similarity score (0-1).

        Raises:
            FileNotFoundError: If index or metadata files don't exist.
        """
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.top_k = top_k
        self.score_threshold = score_threshold

        if not Path(index_path).exists():
            raise FileNotFoundError(f"FAISS index not found at {index_path}")
        if not Path(metadata_path).exists():
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

        # Load embedding model
        logger.info(f"Loading embedding model: {model_name}")
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

        # Load FAISS index
        logger.info(f"Loading FAISS index from {index_path}")
        try:
            self.index: faiss.Index = faiss.read_index(index_path)
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            raise

        # Load metadata
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.metadata: List[Dict] = json.load(f)
            logger.info(f"Loaded {len(self.metadata)} document chunks")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid metadata JSON: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            raise

        if len(self.metadata) != self.index.ntotal:
            logger.warning(
                f"Metadata count ({len(self.metadata)}) != "
                f"Index size ({self.index.ntotal}). Data may be inconsistent."
            )

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: The search query.
            top_k: Number of results to return (uses default if None).

        Returns:
            List of chunk dicts with 'similarity_score' added, sorted by relevance.

        Raises:
            ValueError: If query is empty.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        query = query.strip()
        k = top_k or self.top_k

        # Generate query embedding
        logger.debug(f"Encoding query: {query[:60]}...")
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        query_embedding = np.array(query_embedding, dtype="float32")

        # Search FAISS index
        logger.debug(f"Searching FAISS index for top {k} results")
        distances, indices = self.index.search(query_embedding, k)

        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx >= len(self.metadata):
                continue

            # Convert L2 distance to a 0-1 similarity score
            similarity_score = 1.0 / (1.0 + float(distance))

            if similarity_score >= self.score_threshold:
                # Copy chunk to avoid mutating the stored metadata
                chunk = dict(self.metadata[idx])
                chunk["similarity_score"] = similarity_score
                chunk["distance"] = float(distance)
                results.append(chunk)

        logger.info(
            f"Retrieved {len(results)} chunks "
            f"(threshold: {self.score_threshold:.2f})"
        )
        return results

    def retrieve_with_scores(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> Tuple[List[Dict], List[float]]:
        """
        Retrieve chunks and return similarity scores separately.

        Returns:
            Tuple of (chunks_without_score_field, scores).
        """
        chunks = self.retrieve(query, top_k)
        # Use .get() and build new dicts — don't mutate the originals
        scores = [c.get("similarity_score", 0.0) for c in chunks]
        clean_chunks = [
            {k: v for k, v in c.items() if k != "similarity_score"}
            for c in chunks
        ]
        return clean_chunks, scores

    def get_stats(self) -> Dict:
        """Get statistics about the retriever and index."""
        try:
            model_name = self.model[0].auto_model.config.name_or_path
        except Exception:
            model_name = "unknown"

        return {
            "total_chunks": len(self.metadata),
            "index_size": self.index.ntotal,
            "index_type": type(self.index).__name__,
            "embedding_dim": self.model.get_embedding_dimension(),
            "model_name": model_name,
            "top_k": self.top_k,
            "score_threshold": self.score_threshold,
        }


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    try:
        retriever = Retriever(
            index_path="embeddings/vector_store/index.faiss",
            metadata_path="embeddings/vector_store/metadata.json",
            top_k=3,
        )
        print("Stats:", retriever.get_stats())

        results = retriever.retrieve("What skills does Mahesh have?")
        for i, result in enumerate(results, 1):
            print(f"\n--- Result {i} ---")
            print(f"Source: {result['source']} (Page {result.get('page')})")
            print(f"Score: {result['similarity_score']:.2%}")
            print(f"Text: {result['text'][:200]}...")

    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)