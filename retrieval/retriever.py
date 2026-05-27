# retrieval/retriever.py
"""
Vector-based document retriever using FAISS and Sentence Transformers.
Includes error handling, logging, and proper type hints.
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
        "Required packages not installed. Run: pip install faiss-cpu sentence-transformers"
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
        top_k: int = 3,
        score_threshold: float = 0.5
    ):
        """
        Initialize the retriever.
        
        Args:
            index_path: Path to FAISS index file
            metadata_path: Path to metadata JSON file
            model_name: Name of the embedding model
            top_k: Number of results to retrieve
            score_threshold: Minimum similarity score (0-1)
            
        Raises:
            FileNotFoundError: If index or metadata files don't exist
            ValueError: If configuration is invalid
        """
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.top_k = top_k
        self.score_threshold = score_threshold
        
        if not Path(index_path).exists():
            raise FileNotFoundError(f"FAISS index not found at {index_path}")
        if not Path(metadata_path).exists():
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
        
        # Load model
        try:
            logger.info(f"Loading embedding model: {model_name}")
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
        
        # Load FAISS index
        try:
            logger.info(f"Loading FAISS index from {index_path}")
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
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: The search query
            top_k: Number of results to return (uses default if None)
            
        Returns:
            List of relevant chunks with metadata, sorted by relevance
            
        Raises:
            ValueError: If query is empty or invalid
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        query = query.strip()
        if len(query) > 2000:
            logger.warning(f"Query is very long ({len(query)} chars)")
        
        k = top_k or self.top_k
        
        try:
            # Generate query embedding
            logger.debug(f"Encoding query: {query[:50]}...")
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            query_embedding = np.array(query_embedding, dtype="float32")
            
            # Search FAISS index
            logger.debug(f"Searching FAISS index for top {k} results")
            distances, indices = self.index.search(query_embedding, k)
            
            # Process results
            results = []
            for distance, idx in zip(distances[0], indices[0]):
                if idx == -1:
                    # Invalid index
                    continue
                
                if idx >= len(self.metadata):
                    logger.warning(f"Invalid index {idx}, skipping")
                    continue
                
                # Convert L2 distance to similarity score (0-1 range)
                # Lower distance = higher similarity
                similarity_score = 1 / (1 + distance)
                
                if similarity_score >= self.score_threshold:
                    chunk = self.metadata[idx].copy()
                    chunk["similarity_score"] = float(similarity_score)
                    chunk["distance"] = float(distance)
                    results.append(chunk)
            
            logger.info(
                f"Retrieved {len(results)} chunks for query "
                f"(threshold: {self.score_threshold:.2f})"
            )
            return results
        
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            raise
    
    def retrieve_with_scores(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> Tuple[List[Dict], List[float]]:
        """
        Retrieve chunks and return with explicit similarity scores.
        
        Args:
            query: The search query
            top_k: Number of results to return
            
        Returns:
            Tuple of (chunks, similarity_scores)
        """
        chunks = self.retrieve(query, top_k)
        scores = [c.pop("similarity_score") for c in chunks]
        return chunks, scores
    
    def get_stats(self) -> Dict:
        """Get statistics about the retriever and index."""
        return {
            "total_chunks": len(self.metadata),
            "index_size": self.index.ntotal,
            "index_type": self.index.__class__.__name__,
            "embedding_dim": self.model.get_sentence_embedding_dimension(),
            "model_name": self.model.modules()[0].config.model_name_or_path if hasattr(self.model.modules()[0], 'config') else "unknown",
            "top_k": self.top_k,
            "score_threshold": self.score_threshold,
        }


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    try:
        retriever = Retriever(
            index_path="embeddings/vector_store/index.faiss",
            metadata_path="embeddings/vector_store/metadata.json",
            top_k=3
        )
        
        print("Stats:", retriever.get_stats())
        
        results = retriever.retrieve("What skills does Mahesh have?")
        
        for i, result in enumerate(results, 1):
            print(f"\n--- Result {i} ---")
            print(f"Source: {result['source']} (Page {result['page']})")
            print(f"Score: {result['similarity_score']:.2%}")
            print(f"Text: {result['text'][:200]}...")
    
    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)
