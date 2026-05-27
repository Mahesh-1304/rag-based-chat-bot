# RAG Document Chatbot - Complete Setup Guide

A production-ready Retrieval Augmented Generation (RAG) system for asking questions about documents. This system combines semantic search with large language models to provide accurate, context-aware answers.

## 📋 Table of Contents

- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Improvements Made](#improvements-made)
- [Troubleshooting](#troubleshooting)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG PIPELINE                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  [1] INGESTION PHASE                                         │
│  ├─ Load Documents (PDF, DOCX)                              │
│  ├─ Clean Text (remove headers, normalize)                  │
│  ├─ Chunk Text (semantic chunks)                            │
│  └─ Generate Embeddings (Sentence Transformers)             │
│              ↓                                               │
│  [2] STORAGE PHASE                                           │
│  └─ Store in FAISS Vector Database                          │
│       (Fast similarity search)                               │
│              ↓                                               │
│  [3] QUERY PHASE                                             │
│  ├─ Embed User Query (same model)                           │
│  ├─ Semantic Search (retrieve top-k similar chunks)         │
│  └─ Build Context                                           │
│              ↓                                               │
│  [4] GENERATION PHASE                                        │
│  └─ LLM Answer Generation (Llama2 or GPT)                   │
│       (Using retrieved context)                             │
│              ↓                                               │
│  [5] SERVING PHASE                                           │
│  └─ FastAPI REST Endpoints                                  │
│     (Health checks, Query, Retrieve, Stats)                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Purpose | Technology |
|-----------|---------|-----------|
| **Document Loader** | Extract text from PDFs & DOCX | pypdf, python-docx |
| **Text Cleaner** | Remove headers, normalize spaces | regex, NLP preprocessing |
| **Chunker** | Split into semantically meaningful chunks | LangChain RecursiveCharacterTextSplitter |
| **Embedder** | Generate semantic embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| **Vector Store** | Store & search embeddings | FAISS (IndexHNSWFlat) |
| **Retriever** | Find relevant chunks for queries | Semantic similarity search |
| **Context Builder** | Format chunks into LLM context | String formatting |
| **LLM Client** | Generate answers from context | Llama2 (local) or OpenAI API |
| **API Server** | REST endpoints for querying | FastAPI |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- 8GB+ RAM (for embeddings & LLM)
- ~5GB disk space (for models)

### 1. Clone & Setup

```bash
# Clone repository
git clone <repo-url>
cd rag-document-chatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy example config
cp .env.example .env

# Edit .env with your paths (use absolute or relative paths)
nano .env
```

### 3. Prepare Documents

```bash
# Create data directory
mkdir -p data/raw_docs

# Add your PDF and DOCX files to data/raw_docs/
cp your_documents/*.pdf data/raw_docs/
cp your_documents/*.docx data/raw_docs/
```

### 4. Ingest Documents

```bash
# Run the ingestion pipeline
python -m ingestion.pipeline

# This will:
# 1. Load all documents from data/raw_docs/
# 2. Clean and chunk text
# 3. Generate embeddings
# 4. Create FAISS index
# 5. Save to data/processed_docs/chunks.json
```

### 5. Start API Server

```bash
# Start FastAPI server
python -m api.main

# Server will be available at http://localhost:8000
# API docs at http://localhost:8000/docs (Swagger UI)
```

### 6. Query Your Documents

```bash
# Using curl
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What skills does Mahesh have?"}'

# Or use the Python client
python -c "
from requests import post
response = post('http://localhost:8000/query', json={
    'question': 'What skills does Mahesh have?'
})
print(response.json())
"
```

---

## 📁 Project Structure

```
rag-document-chatbot/
├── ingestion/                    # Document processing
│   ├── __init__.py
│   ├── document_loader.py        # Load PDF, DOCX
│   ├── text_cleaner.py           # Clean and normalize
│   ├── chunker.py                # Split into chunks
│   └── pipeline.py               # Orchestrate ingestion
│
├── retrieval/                    # Semantic search
│   ├── __init__.py
│   ├── embedder.py               # Generate embeddings
│   ├── retriever.py              # Search vector store
│   └── context_builder.py        # Format for LLM
│
├── llm/                          # Language model
│   ├── __init__.py
│   ├── llm_client.py             # LLM inference
│   ├── response_generator.py     # Orchestrate QA
│   └── prompt_templates.py       # System prompts
│
├── api/                          # REST API
│   ├── __init__.py
│   └── main.py                   # FastAPI server
│
├── config/                       # Configuration
│   ├── __init__.py
│   ├── settings.py               # Environment & defaults
│   └── logging_config.py         # Logger setup
│
├── utils/                        # Helper utilities
│   ├── __init__.py
│   ├── validators.py             # Input validation
│   ├── metrics.py                # Performance tracking
│   └── cache.py                  # Caching
│
├── tests/                        # Unit & integration tests
│   ├── __init__.py
│   ├── test_loader.py
│   ├── test_chunker.py
│   └── test_retriever.py
│
├── data/                         # Data directories
│   ├── raw_docs/                 # Input documents
│   └── processed_docs/           # Processed chunks
│
├── embeddings/                   # Vector store
│   └── vector_store/
│       ├── index.faiss           # FAISS index
│       └── metadata.json         # Chunk metadata
│
├── models/                       # Language models
│   └── llama-2-7b.Q5_K_M.gguf    # Llama2 model
│
├── logs/                         # Application logs
│
├── .env.example                  # Configuration template
├── .gitignore                    # Git ignore rules
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── main.py                       # Entry point
```

---

## ⚙️ Setup & Installation

### System Requirements

```bash
# Check Python version
python --version  # Should be 3.10 or higher

# Check available RAM
# Linux/Mac: free -h
# Windows: wmic OS get TotalVisibleMemorySize

# Check disk space
df -h  # or use Windows File Explorer
```

### Install Dependencies

The system requires several key packages:

```bash
# Core dependencies
langchain>=0.1.0                  # LLM orchestration
langchain-community>=0.0.10       # Additional integrations
langchain-huggingface>=0.0.1      # HuggingFace models

# Vector search
faiss-cpu>=1.7.4                  # Fast similarity search
sentence-transformers>=2.2.2      # Embedding generation

# Document processing
pypdf>=3.0.0                      # PDF extraction
python-docx>=0.8.11               # DOCX extraction

# LLM (choose one)
llama-cpp-python>=0.2.0           # For local Llama2
openai>=1.0.0                     # For OpenAI API

# Web framework
fastapi>=0.104.0                  # REST API
uvicorn>=0.24.0                   # ASGI server

# UI
streamlit>=1.28.0                 # Web dashboard (optional)

# Configuration & utilities
pydantic>=2.0.0                   # Data validation
pydantic-settings>=2.0.0          # Settings management
python-dotenv>=1.0.0              # .env file support
numpy>=1.24.0                     # Numerical computing

# Testing
pytest>=7.0.0                     # Test framework
pytest-asyncio>=0.21.0            # Async test support
```

### Create Virtual Environment

```bash
# Create venv
python -m venv venv

# Activate venv
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Verify activation (should show (venv) in prompt)
which python  # Linux/Mac
where python  # Windows
```

### Install Requirements

```bash
# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install requirements
pip install -r requirements.txt

# Verify installation
python -c "import faiss, sentence_transformers, fastapi; print('✓ All packages installed')"
```

---

## 🔧 Configuration

### Environment Variables (.env)

Create a `.env` file from `.env.example`:

```bash
cp .env.example .env
```

Key settings:

```env
# Paths (relative to project root)
DATA_DIR=./data
RAW_DOCS_DIR=./data/raw_docs
VECTOR_STORE_DIR=./embeddings/vector_store
MODEL_PATH=./models/llama-2-7b.Q5_K_M.gguf

# Embedding settings
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
CHUNK_SIZE=400
CHUNK_OVERLAP=50

# Retriever settings
RETRIEVER_TOP_K=3
RETRIEVER_SCORE_THRESHOLD=0.5

# LLM settings
LLM_TEMPERATURE=0.0           # Factual, not creative
LLM_MAX_TOKENS=256

# API settings
API_HOST=0.0.0.0
API_PORT=8000

# Logging
LOG_LEVEL=INFO
```

### Programmatic Configuration

Access settings in code:

```python
from config.settings import settings

# Use settings
print(settings.DATA_DIR)
print(settings.RETRIEVER_TOP_K)
print(settings.API_PORT)

# Settings are loaded from:
# 1. .env file (priority)
# 2. Environment variables
# 3. Defaults in settings.py
```

---

## 💻 Usage

### Command Line Interface

#### 1. Ingest Documents

```bash
# Basic ingestion
python -m ingestion.pipeline

# Custom paths
python -m ingestion.pipeline data/my_docs data/output

# With custom chunk size
python -m ingestion.pipeline \
    data/raw_docs \
    data/processed_docs \
    --chunk-size 512 \
    --chunk-overlap 100
```

#### 2. Query Documents (Command Line)

```bash
python -c "
from retrieval.retriever import Retriever
from config.settings import settings

retriever = Retriever(
    index_path=str(settings.VECTOR_STORE_DIR / 'index.faiss'),
    metadata_path=str(settings.VECTOR_STORE_DIR / 'metadata.json')
)

results = retriever.retrieve('Your question here')
for r in results:
    print(f'Source: {r[\"source\"]}')
    print(f'Content: {r[\"text\"][:200]}...')
    print('---')
"
```

### Python API

#### Retrieve Documents

```python
from retrieval.retriever import Retriever

retriever = Retriever(
    index_path="embeddings/vector_store/index.faiss",
    metadata_path="embeddings/vector_store/metadata.json",
    top_k=5
)

# Retrieve similar chunks
chunks = retriever.retrieve("What skills does Mahesh have?")

for chunk in chunks:
    print(f"Score: {chunk['similarity_score']:.2%}")
    print(f"Source: {chunk['source']}")
    print(f"Text: {chunk['text']}\n")
```

#### Generate Answers

```python
from response_generator import answer_query

# Full pipeline: retrieve + generate
answer = answer_query("What is your name?")
print(answer)
```

### REST API

Start the server:

```bash
python -m api.main
```

API is available at `http://localhost:8000`

---

## 📚 API Documentation

### Endpoints

#### 1. Health Check

```bash
GET /health

Response:
{
  "status": "healthy",
  "initialized": true,
  "retriever_stats": {
    "total_chunks": 150,
    "index_size": 150,
    "embedding_dim": 384,
    "model_name": "all-MiniLM-L6-v2"
  },
  "timestamp": "2024-01-15T10:30:00"
}
```

#### 2. Query Documents (Main Endpoint)

```bash
POST /query

Request:
{
  "question": "What skills does Mahesh have?",
  "top_k": 3,
  "include_scores": true
}

Response:
{
  "question": "What skills does Mahesh have?",
  "answer": "Mahesh has skills in Python, Data Analysis, and Business Strategy...",
  "retrieved_chunks": [
    {
      "chunk_id": "resume_1_a2b3c4",
      "text": "Proficient in Python, SQL, and Data Analysis...",
      "source": "resume.pdf",
      "page": 1,
      "similarity_score": 0.92
    }
  ],
  "timestamp": "2024-01-15T10:30:00",
  "processing_time_ms": 245.3
}
```

#### 3. Retrieve Without Generating Answer

```bash
GET /retrieve?query=skills&top_k=5

Response:
[
  {
    "chunk_id": "resume_1_a2b3c4",
    "text": "...",
    "source": "resume.pdf",
    "page": 1,
    "similarity_score": 0.92
  },
  ...
]
```

#### 4. Get Statistics

```bash
GET /stats

Response:
{
  "total_chunks": 150,
  "index_size": 150,
  "index_type": "IndexHNSWFlat",
  "embedding_dim": 384,
  "model_name": "all-MiniLM-L6-v2",
  "top_k": 3,
  "score_threshold": 0.5
}
```

### Interactive API Documentation

Visit `http://localhost:8000/docs` for Swagger UI where you can test endpoints directly.

---

## ✨ Improvements Made

### From Original to Enhanced Version

| Issue | Before | After |
|-------|--------|-------|
| **Import Errors** | Mismatched directory structure | Proper package organization with `__init__.py` |
| **Hardcoded Paths** | Absolute Windows paths scattered | Configuration management with `.env` |
| **Error Handling** | No error handling | Try-catch blocks, logging throughout |
| **Logging** | `print()` statements | Proper `logging` module with levels |
| **Type Hints** | Missing | Complete type annotations |
| **API** | FastAPI in requirements only | Full FastAPI implementation |
| **Vector Search** | IndexFlatL2 (O(n) search) | HNSW index (O(log n) search) |
| **Documentation** | None | Comprehensive docstrings & README |
| **Testing** | Basic import test | Unit test structure ready |
| **Async Support** | Blocking I/O | Async endpoints in FastAPI |

### Performance Improvements

```
Before:
- Vector search: O(n) - 10,000+ items = slow
- Hardcoded paths: breaks on different machines
- No caching: repeated queries are slow
- No error recovery: single failure stops everything

After:
- Vector search: O(log n) - 10,000+ items = fast
- Configuration-driven: works everywhere
- Caching ready: repeated queries instant
- Error recovery: graceful degradation
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Module Not Found Errors

```
ModuleNotFoundError: No module named 'retrieval'
```

**Solution**: Ensure directory structure is correct and `__init__.py` files exist:

```bash
touch retrieval/__init__.py
touch ingestion/__init__.py
touch llm/__init__.py
touch api/__init__.py
touch config/__init__.py
```

#### 2. FAISS Index Not Found

```
FileNotFoundError: FAISS index not found at embeddings/vector_store/index.faiss
```

**Solution**: Run ingestion pipeline first:

```bash
python -m ingestion.pipeline
```

#### 3. Out of Memory

If process crashes with "killed" or "out of memory":

**Solution**: Use smaller batch size or reduce top_k:

```env
CHUNK_SIZE=200        # Reduce from 400
RETRIEVER_TOP_K=1     # Reduce from 3
```

#### 4. API Port Already in Use

```
OSError: [Errno 48] Address already in use
```

**Solution**: Change port in `.env`:

```env
API_PORT=8001  # or any available port
```

#### 5. Slow Retrieval

**Solution**: Check if using HNSW index:

```env
VECTOR_STORE_TYPE=hnsw  # Fast
# NOT: flat (slow) or ivf (complex)
```

#### 6. Wrong Answers from LLM

**Solution**: Improve prompt and check retrieved chunks:

1. Use `/retrieve` endpoint to see what's being retrieved
2. Check if documents are relevant
3. Adjust RETRIEVER_SCORE_THRESHOLD in `.env`
4. Improve prompt in `llm/prompt_templates.py`

---

## 📈 Next Steps

### For Production Deployment:

1. **Database**: Add PostgreSQL for metadata
2. **Caching**: Add Redis for query caching
3. **Monitoring**: Add Prometheus metrics
4. **Containerization**: Create Docker image
5. **Load Testing**: Stress test with locust
6. **Authentication**: Add API key validation
7. **Rate Limiting**: Prevent abuse

### For Better Performance:

1. Use quantized embeddings (reduce memory)
2. Add re-ranking stage (improve quality)
3. Implement query expansion (find more results)
4. Use multi-hop retrieval (chain queries)
5. Add caching layer (faster responses)

### For Better Quality:

1. Fine-tune embedding model (domain-specific)
2. Improve chunking strategy
3. Add metadata filtering
4. Use better LLM (GPT-4, Claude)
5. Implement human feedback loop

---

## 📞 Support

For issues or questions:

1. Check this README
2. Review code comments and docstrings
3. Check API docs at `http://localhost:8000/docs`
4. Review logs in `logs/` directory
5. Open an issue on GitHub

---

## 📄 License

[Your License Here]

---

## 👤 Author

Created as a production-ready RAG system. See ARCHITECTURE_ANALYSIS.md for detailed technical analysis.
#   r a g - b a s e d - c h a t - b o t  
 