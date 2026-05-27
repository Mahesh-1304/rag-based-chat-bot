# IMPLEMENTATION GUIDE - Step-by-Step Improvements

This document provides a step-by-step guide to implement all the architectural improvements.

## Phase 1: Critical Fixes (Must Do - 2-3 hours)

### 1.1 Fix Directory Structure

```bash
# Current state
project/
├── embedder.py
├── chunker.py
├── document_loader.py
├── text_cleaner.py
├── ingest_pipeline.py
├── retriever.py
├── context_builder.py
├── llm_client.py
├── response_generator.py
└── prompt_templates.py

# Target state
project/
├── ingestion/
│   ├── __init__.py
│   ├── document_loader.py
│   ├── text_cleaner.py
│   ├── chunker.py
│   └── pipeline.py
├── retrieval/
│   ├── __init__.py
│   ├── embedder.py
│   ├── retriever.py
│   └── context_builder.py
├── llm/
│   ├── __init__.py
│   ├── llm_client.py
│   ├── response_generator.py
│   └── prompt_templates.py
└── config/
    ├── __init__.py
    └── settings.py
```

**Implementation Steps:**

```bash
# 1. Create directories
mkdir -p ingestion retrieval llm config api tests utils

# 2. Create __init__.py files
touch ingestion/__init__.py
touch retrieval/__init__.py
touch llm/__init__.py
touch config/__init__.py
touch api/__init__.py
touch utils/__init__.py

# 3. Move files
mv document_loader.py ingestion/
mv text_cleaner.py ingestion/
mv chunker.py ingestion/
mv ingest_pipeline.py ingestion/pipeline.py

mv embedder.py retrieval/
mv retriever.py retrieval/
mv context_builder.py retrieval/

mv llm_client.py llm/
mv response_generator.py llm/
mv prompt_templates.py llm/

# 4. Verify structure
tree -I '__pycache__'
```

### 1.2 Add Configuration Management

**File**: `config/settings.py` (use provided example)

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Paths
    DATA_DIR: str = "data"
    RAW_DOCS_DIR: str = "data/raw_docs"
    # ... etc
    
    class Config:
        env_file = ".env"

settings = Settings()
```

**Usage in code**:

```python
# Old way (❌ don't do this)
CHUNKS_PATH = "data/processed_docs/chunks.json"
INDEX_PATH = "embeddings/vector_store/index.faiss"

# New way (✅ do this)
from config.settings import settings

chunks_path = settings.PROCESSED_DOCS_DIR / "chunks.json"
index_path = settings.VECTOR_STORE_DIR / "index.faiss"
```

### 1.3 Create .env File

```bash
# Copy template
cp .env.example .env

# Edit with your values
nano .env  # or use your editor
```

### 1.4 Update requirements.txt

Replace the minimal requirements with the complete one (see `requirements_complete.txt`).

**Key additions**:
- `pydantic-settings` (for config)
- `langchain-community` (missing from original)
- `sentence-transformers` (missing from original)
- `pytest` (for testing)
- Testing and logging utilities

```bash
# Update requirements
pip install -r requirements_complete.txt
```

### 1.5 Update All Imports

**Old imports (❌)**:
```python
from retrieval.retriever import Retriever
from retrieval.context_builder import build_context
from llm.prompt_templates import SYSTEM_PROMPT
```

**New imports (✅)** - These should now work after moving files:
```python
from retrieval.retriever import Retriever
from retrieval.context_builder import build_context
from llm.prompt_templates import SYSTEM_PROMPT
from config.settings import settings
```

**Update these files**:
- `ingestion/pipeline.py` - Add config imports
- `retrieval/retriever.py` - Add config imports
- `llm/response_generator.py` - Add config imports
- `api/main.py` - Add config and logger imports

---

## Phase 2: Error Handling & Logging (4-5 hours)

### 2.1 Add Logging Configuration

**File**: `config/logging_config.py`

```python
import logging
import logging.config
from config.settings import settings

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": settings.LOG_FORMAT
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": "INFO",
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": str(settings.LOGS_DIR / "app.log"),
            "formatter": "standard",
            "level": "DEBUG",
        },
    },
    "root": {
        "level": settings.LOG_LEVEL,
        "handlers": ["console", "file"],
    },
}

logging.config.dictConfig(LOGGING_CONFIG)
```

### 2.2 Add Error Handling to Each Module

**Pattern to follow**:

```python
import logging

logger = logging.getLogger(__name__)

def load_chunks(path: str) -> list:
    """Load chunks with error handling."""
    try:
        if not Path(path).exists():
            raise FileNotFoundError(f"Chunks file not found: {path}")
        
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in chunks file: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to load chunks: {e}")
        raise
```

**Update these files**:

1. `retrieval/retriever.py` - ✅ Use provided version
2. `ingestion/pipeline.py` - ✅ Use provided version
3. `ingestion/document_loader.py` - Add try-catch per file
4. `retrieval/embedder.py` - Add error handling
5. `llm/llm_client.py` - Add error handling

### 2.3 Add Type Hints

**Pattern**:

```python
# Before (❌)
def chunk_text(documents, chunk_size=400, overlap=50):
    all_chunks = []
    for doc in documents:
        ...

# After (✅)
from typing import List, Dict

def chunk_text(
    documents: List[Dict[str, str]],
    chunk_size: int = 400,
    overlap: int = 50
) -> List[Dict[str, str]]:
    """Split documents into chunks with overlap.
    
    Args:
        documents: List of document dictionaries
        chunk_size: Number of tokens per chunk
        overlap: Number of overlapping tokens
        
    Returns:
        List of chunk dictionaries
    """
    all_chunks = []
    ...
```

**Tools to help**:
```bash
# Check type hints coverage
pip install pyright
pyright ingestion/

# Auto-format code
black ingestion/
```

---

## Phase 3: Performance Improvements (3-4 hours)

### 3.1 Upgrade Vector Store Index

**Old (slow - O(n))**:
```python
# embedder.py
index = faiss.IndexFlatL2(VECTOR_DIM)
index.add(embeddings)
```

**New (fast - O(log n))**:
```python
# embedder.py
def create_index(embeddings: np.ndarray) -> faiss.Index:
    """Create HNSW index for fast search."""
    d = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(d, 32)  # 32 neighbors
    index.add(embeddings)
    return index
```

### 3.2 Implement Caching

**File**: `utils/cache.py`

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_cached_embedding(query: str):
    """Cache embeddings for repeated queries."""
    return model.encode([query])
```

Usage:
```python
# In retriever.py
from utils.cache import get_cached_embedding

class Retriever:
    def retrieve(self, query: str):
        embedding = get_cached_embedding(query)
        # ... rest of code
```

### 3.3 Better Text Splitting

**Replace naive tokenizer**:

```python
# Old (❌)
def tokenize(text: str):
    return text.split()

# New (✅)
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)

splits = splitter.split_text(text)
```

---

## Phase 4: API & Testing (4-5 hours)

### 4.1 Implement FastAPI Endpoints

Use the provided `api/main.py` file.

```bash
# Test if it works
python -m api.main

# Check endpoint docs
# Visit: http://localhost:8000/docs
```

### 4.2 Add Unit Tests

**File**: `tests/test_retriever.py`

```python
import pytest
from retrieval.retriever import Retriever

@pytest.fixture
def retriever():
    return Retriever(
        index_path="embeddings/vector_store/index.faiss",
        metadata_path="embeddings/vector_store/metadata.json"
    )

def test_retrieve_returns_list(retriever):
    results = retriever.retrieve("test query")
    assert isinstance(results, list)

def test_retrieve_empty_query_raises(retriever):
    with pytest.raises(ValueError):
        retriever.retrieve("")

def test_retrieve_returns_similarity_scores(retriever):
    results = retriever.retrieve("test")
    for r in results:
        assert "similarity_score" in r
        assert 0 <= r["similarity_score"] <= 1
```

Run tests:
```bash
pytest tests/ -v
pytest tests/ --cov=ingestion --cov=retrieval --cov=llm
```

---

## Phase 5: Documentation & Polish (2-3 hours)

### 5.1 Add Docstrings

Use Google-style docstrings:

```python
def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
    """
    Retrieve relevant chunks for a query.
    
    Uses semantic similarity search to find the most relevant
    document chunks based on the query embedding.
    
    Args:
        query: The search query string
        top_k: Number of results to return (uses default if None)
        
    Returns:
        List of relevant chunks with metadata, sorted by relevance
        
    Raises:
        ValueError: If query is empty or invalid
        FileNotFoundError: If vector store files not found
        
    Examples:
        >>> retriever = Retriever("index.faiss", "metadata.json")
        >>> results = retriever.retrieve("What is AI?")
        >>> print(f"Found {len(results)} results")
    """
```

### 5.2 Add Module Documentation

Add at top of each file:

```python
"""
Module: Text Cleaner

Provides text cleaning and normalization functions for document processing.
Handles removal of common artifacts like page numbers and excess whitespace.

Functions:
    clean_text: Clean a list of documents
    
Example:
    >>> docs = [{"text": "Hello  World", "source": "doc.pdf"}]
    >>> cleaned = clean_text(docs)
"""
```

### 5.3 Create Architecture Diagram

Document in README:

```
User Query
    ↓
[API Endpoint]
    ↓
[Query Embedding]
    ↓
[FAISS Index Search] → [Vector Store]
    ↓
[Retrieve Top-K Chunks]
    ↓
[Context Builder]
    ↓
[LLM Generation]
    ↓
[Response]
```

---

## Implementation Checklist

### Phase 1: Critical (✓ Must Complete)
- [ ] Create directory structure
- [ ] Create `__init__.py` files in each package
- [ ] Move files to correct directories
- [ ] Create `config/settings.py`
- [ ] Create `.env` file
- [ ] Update `requirements.txt`
- [ ] Verify all imports work: `python -c "from ingestion.pipeline import *"`

### Phase 2: Error Handling
- [ ] Create `config/logging_config.py`
- [ ] Add logging to each module
- [ ] Add error handling to document loader
- [ ] Add error handling to embedder
- [ ] Add error handling to retriever
- [ ] Add error handling to LLM client
- [ ] Add type hints throughout

### Phase 3: Performance
- [ ] Upgrade FAISS index to HNSW
- [ ] Implement caching layer
- [ ] Replace naive tokenizer with RecursiveCharacterTextSplitter
- [ ] Test performance improvements

### Phase 4: API & Testing
- [ ] Implement `api/main.py`
- [ ] Test API endpoints with curl
- [ ] Create unit tests
- [ ] Create integration tests
- [ ] Achieve 80%+ code coverage

### Phase 5: Documentation
- [ ] Add docstrings to all functions
- [ ] Add module-level documentation
- [ ] Create API documentation
- [ ] Update README with examples
- [ ] Add architecture diagrams

---

## Testing Each Phase

### Phase 1 - Test imports:
```bash
python -c "
from ingestion.pipeline import main as ingest
from retrieval.retriever import Retriever
from llm.response_generator import answer_query
print('✓ All imports work')
"
```

### Phase 2 - Test logging:
```bash
python -c "
import logging
from config.logging_config import LOGGING_CONFIG
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)
logger.info('✓ Logging works')
"
```

### Phase 3 - Test performance:
```bash
python -c "
import time
from retrieval.retriever import Retriever

r = Retriever('embeddings/vector_store/index.faiss', 'embeddings/vector_store/metadata.json')
start = time.time()
results = r.retrieve('test query')
print(f'✓ Search took {(time.time()-start)*1000:.2f}ms')
"
```

### Phase 4 - Test API:
```bash
# In one terminal
python -m api.main

# In another terminal
curl http://localhost:8000/health
pytest tests/ -v
```

---

## Timeline Estimate

| Phase | Tasks | Hours | Total |
|-------|-------|-------|-------|
| 1 | Structure & Config | 2-3 | 2-3h |
| 2 | Error Handling & Logging | 4-5 | 6-8h |
| 3 | Performance | 3-4 | 9-12h |
| 4 | API & Testing | 4-5 | 13-17h |
| 5 | Documentation | 2-3 | 15-20h |

**Total: 15-20 hours for complete refactoring**

Or implement in phases:
- **Quick Win** (Phase 1): 2-3 hours - makes code runnable
- **Robust** (Phases 1-2): 6-8 hours - makes code production-ready
- **Professional** (All phases): 15-20 hours - production-grade system

---

## After Implementation Verification

```bash
# 1. All tests pass
pytest tests/ -v --cov

# 2. No import errors
python -c "from ingestion.pipeline import *; from retrieval.retriever import *; from llm.response_generator import *"

# 3. API starts cleanly
python -m api.main  # Should start without errors

# 4. Type checking passes
mypy ingestion/ retrieval/ llm/

# 5. Code formatting
black ingestion/ retrieval/ llm/ --check

# 6. Sample ingestion works
python -m ingestion.pipeline

# 7. Sample query works
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "test"}'
```

All green? 🎉 You're production-ready!
