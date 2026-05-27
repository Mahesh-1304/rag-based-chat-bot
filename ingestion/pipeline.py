# ingestion/pipeline.py
"""
Document ingestion pipeline with error handling and logging.
Orchestrates the flow: Load → Clean → Chunk → Embed.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class IngestionStats:
    """Statistics from ingestion process."""
    total_files_found: int = 0
    files_loaded: int = 0
    files_failed: int = 0
    total_documents: int = 0
    total_chunks: int = 0
    total_errors: List[Dict] = None
    
    def __post_init__(self):
        if self.total_errors is None:
            self.total_errors = []


class IngestionPipeline:
    """
    Orchestrates the document ingestion pipeline.
    
    Pipeline steps:
    1. Load documents from files
    2. Clean text
    3. Split into chunks
    4. Generate embeddings (optional)
    5. Store in vector database
    """
    
    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        chunk_size: int = 400,
        chunk_overlap: int = 50
    ):
        """Initialize the pipeline."""
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Import here to avoid circular dependencies
        from .document_loader import load_documents
        from .text_cleaner import clean_text
        from .chunker import chunk_text
        
        self.load_documents = load_documents
        self.clean_text = clean_text
        self.chunk_text = chunk_text
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run(self) -> Tuple[List[Dict], IngestionStats]:
        """
        Run the complete ingestion pipeline.
        
        Returns:
            Tuple of (chunks, statistics)
        """
        stats = IngestionStats()
        
        logger.info("=" * 60)
        logger.info("Starting Document Ingestion Pipeline")
        logger.info("=" * 60)
        
        # Step 1: Load documents
        logger.info(f"\n[1/4] Loading documents from {self.input_dir}")
        raw_docs = self._load_documents_safe(stats)
        
        if not raw_docs:
            logger.error("No documents loaded. Pipeline failed.")
            return [], stats
        
        stats.total_documents = len(raw_docs)
        logger.info(f"✓ Loaded {len(raw_docs)} document sections")
        
        # Step 2: Clean text
        logger.info(f"\n[2/4] Cleaning text")
        try:
            cleaned_docs = self.clean_text(raw_docs)
            logger.info(f"✓ Cleaned {len(cleaned_docs)} documents")
        except Exception as e:
            logger.error(f"Text cleaning failed: {e}")
            stats.total_errors.append({
                "step": "cleaning",
                "error": str(e)
            })
            return [], stats
        
        # Step 3: Chunk text
        logger.info(f"\n[3/4] Chunking text (size={self.chunk_size}, overlap={self.chunk_overlap})")
        try:
            chunks = self.chunk_text(
                cleaned_docs,
                chunk_size=self.chunk_size,
                overlap=self.chunk_overlap
            )
            stats.total_chunks = len(chunks)
            logger.info(f"✓ Created {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"Text chunking failed: {e}")
            stats.total_errors.append({
                "step": "chunking",
                "error": str(e)
            })
            return [], stats
        
        # Step 4: Save chunks
        logger.info(f"\n[4/4] Saving chunks to disk")
        try:
            self._save_chunks(chunks)
            logger.info(f"✓ Saved to {self.output_dir / 'chunks.json'}")
        except Exception as e:
            logger.error(f"Failed to save chunks: {e}")
            stats.total_errors.append({
                "step": "saving",
                "error": str(e)
            })
            return [], stats
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("Pipeline Complete")
        logger.info("=" * 60)
        logger.info(f"Documents loaded: {stats.total_documents}")
        logger.info(f"Chunks created: {stats.total_chunks}")
        if stats.total_errors:
            logger.warning(f"Errors: {len(stats.total_errors)}")
        
        return chunks, stats
    
    def _load_documents_safe(self, stats: IngestionStats) -> List[Dict]:
        """Load documents with error tracking."""
        raw_docs = []
        errors = []
        
        # Count total files
        pdf_files = list(self.input_dir.glob("*.pdf"))
        docx_files = list(self.input_dir.glob("*.docx"))
        stats.total_files_found = len(pdf_files) + len(docx_files)
        
        if stats.total_files_found == 0:
            logger.warning(f"No PDF or DOCX files found in {self.input_dir}")
            return []
        
        logger.info(f"Found {stats.total_files_found} files to process")
        
        # Load documents, tracking errors per file
        try:
            loaded, file_errors = self.load_documents(self.input_dir)
            raw_docs.extend(loaded)
            errors.extend(file_errors)
            stats.files_loaded = len(loaded) if loaded else 0
            stats.files_failed = len(file_errors)
        except Exception as e:
            logger.error(f"Document loading failed: {e}")
            stats.total_errors.append({
                "step": "loading",
                "error": str(e)
            })
            return []
        
        # Log per-file errors
        if errors:
            logger.warning(f"\nEncountered {len(errors)} file loading errors:")
            for error in errors:
                logger.warning(f"  • {error['file']}: {error['error']}")
            stats.total_errors.extend(errors)
        
        return raw_docs
    
    def _save_chunks(self, chunks: List[Dict]) -> None:
        """Save chunks to JSON file."""
        output_file = self.output_dir / "chunks.json"
        
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
            
            # Verify file was written
            if not output_file.exists():
                raise IOError(f"File was not created: {output_file}")
            
            file_size_mb = output_file.stat().st_size / (1024 * 1024)
            logger.info(f"File size: {file_size_mb:.2f} MB")
        
        except Exception as e:
            logger.error(f"Failed to save chunks file: {e}")
            raise


def main(
    input_dir: str = "data/raw_docs",
    output_dir: str = "data/processed_docs",
    chunk_size: int = 400,
    chunk_overlap: int = 50
):
    """
    Main entry point for ingestion pipeline.
    
    Args:
        input_dir: Directory containing raw documents
        output_dir: Directory to save processed chunks
        chunk_size: Size of text chunks (in tokens)
        chunk_overlap: Overlap between chunks
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    pipeline = IngestionPipeline(
        input_dir=input_dir,
        output_dir=output_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunks, stats = pipeline.run()
    
    # Return stats for monitoring/logging
    return {
        "success": len(chunks) > 0,
        "chunks_created": stats.total_chunks,
        "documents_processed": stats.total_documents,
        "files_processed": stats.files_loaded,
        "files_failed": stats.files_failed,
        "errors": stats.total_errors
    }


if __name__ == "__main__":
    import sys
    
    # Optional: Accept command-line arguments
    result = main(
        input_dir=sys.argv[1] if len(sys.argv) > 1 else "data/raw_docs",
        output_dir=sys.argv[2] if len(sys.argv) > 2 else "data/processed_docs"
    )
    
    # Exit with error code if pipeline failed
    sys.exit(0 if result["success"] else 1)
