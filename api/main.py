# api/main.py
import logging
import time
import shutil
import json
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from config.settings import settings
from retrieval.retriever import Retriever
from llm.response_generator import answer_query

logging.basicConfig(level=settings.LOG_LEVEL, format=settings.LOG_FORMAT)
logger = logging.getLogger(__name__)


class AppState:
    retriever: Optional[Retriever] = None
    initialized: bool = False


app_state = AppState()


def try_init_retriever():
    index_path = settings.VECTOR_STORE_DIR / "index.faiss"
    meta_path = settings.VECTOR_STORE_DIR / "metadata.json"
    if index_path.exists() and meta_path.exists():
        app_state.retriever = Retriever(
            index_path=str(index_path),
            metadata_path=str(meta_path),
            top_k=settings.RETRIEVER_TOP_K,
            score_threshold=settings.RETRIEVER_SCORE_THRESHOLD,
        )
        stats = app_state.retriever.get_stats()
        logger.info(f"Retriever initialized with {stats['total_chunks']} chunks")
        app_state.initialized = True
    else:
        logger.warning("No vector store found — upload documents to get started.")
        app_state.initialized = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application starting up...")
    try:
        try_init_retriever()
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        app_state.initialized = False
    yield
    logger.info("Shutdown complete")


app = FastAPI(
    title="RAG Document Chatbot API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files if folder exists
frontend_dir = Path("frontend")
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory="frontend"), name="static")


# ── Pydantic Models ──────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    top_k: Optional[int] = Field(None, ge=1, le=10)
    include_scores: Optional[bool] = Field(False)

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Question cannot be empty")
        return v.strip()


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    source: str
    page: Optional[int]
    similarity_score: Optional[float] = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    retrieved_chunks: List[RetrievedChunk]
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    processing_time_ms: Optional[float] = None


class HealthResponse(BaseModel):
    status: str
    initialized: bool
    retriever_stats: Optional[dict] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


# ── Routes ───────────────────────────────────────────────────

@app.get("/")
async def serve_frontend():
    index = Path("frontend/index.html")
    if index.exists():
        return FileResponse(str(index))
    return {"message": "RAG Chatbot API is running. Place index.html in the frontend/ folder."}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    allowed = {".pdf", ".docx"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported.")

    settings.RAW_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    dest = settings.RAW_DOCS_DIR / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    logger.info(f"Saved uploaded file: {dest}")

    try:
        import numpy as np
        import faiss
        from sentence_transformers import SentenceTransformer
        from ingestion.document_loader import load_documents
        from ingestion.text_cleaner import clean_text
        from ingestion.chunker import chunk_text

        docs, errors = load_documents(settings.RAW_DOCS_DIR)
        if not docs:
            raise ValueError("No text could be extracted from the uploaded file.")

        cleaned = clean_text(docs)
        chunks = chunk_text(cleaned)

        settings.PROCESSED_DOCS_DIR.mkdir(parents=True, exist_ok=True)
        with open(settings.PROCESSED_DOCS_DIR / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        texts = [c["text"] for c in chunks]
        embeddings = model.encode(texts, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype="float32")

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        settings.VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(settings.VECTOR_STORE_DIR / "index.faiss"))
        with open(settings.VECTOR_STORE_DIR / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2)

        try_init_retriever()

        return {
            "filename": file.filename,
            "chunks_created": len(chunks),
            "status": "ready",
            "message": f"Successfully processed {file.filename} into {len(chunks)} chunks."
        }

    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.get("/documents")
async def list_documents():
    raw_dir = settings.RAW_DOCS_DIR
    if not raw_dir.exists():
        return {"documents": []}
    files = [
        {
            "name": f.name,
            "size_kb": round(f.stat().st_size / 1024, 1),
            "type": f.suffix.upper().lstrip(".")
        }
        for f in raw_dir.iterdir()
        if f.suffix.lower() in (".pdf", ".docx")
    ]
    return {"documents": files}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        retriever_stats = None
        if app_state.retriever:
            retriever_stats = app_state.retriever.get_stats()
        return HealthResponse(
            status="healthy" if app_state.initialized else "degraded",
            initialized=app_state.initialized,
            retriever_stats=retriever_stats,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    if not app_state.initialized or not app_state.retriever:
        raise HTTPException(status_code=503, detail="No documents loaded yet. Please upload a document first.")

    start_time = time.time()
    try:
        k = request.top_k or settings.RETRIEVER_TOP_K
        retrieved = app_state.retriever.retrieve(request.question, top_k=k)

        if not retrieved:
            return QueryResponse(
                question=request.question,
                answer="I could not find relevant information in the documents to answer your question.",
                retrieved_chunks=[],
            )

        context = "\n\n---\n\n".join(
            f"[Source: {c['source']}, Page: {c.get('page')}]\n{c['text']}"
            for c in retrieved
        )
        answer = answer_query(query=request.question, context=context)

        chunks = [
            RetrievedChunk(
                chunk_id=c.get("chunk_id", "unknown"),
                text=c["text"],
                source=c["source"],
                page=c.get("page"),
                similarity_score=c.get("similarity_score") if request.include_scores else None,
            )
            for c in retrieved
        ]

        processing_time = (time.time() - start_time) * 1000
        return QueryResponse(
            question=request.question,
            answer=answer,
            retrieved_chunks=chunks,
            processing_time_ms=processing_time,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/retrieve", response_model=List[RetrievedChunk])
async def retrieve_endpoint(
    query: str = Query(..., min_length=3, max_length=500),
    top_k: int = Query(3, ge=1, le=10),
):
    if not app_state.initialized or not app_state.retriever:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    chunks = app_state.retriever.retrieve(query, top_k=top_k)
    return [
        RetrievedChunk(
            chunk_id=c.get("chunk_id", "unknown"),
            text=c["text"],
            source=c["source"],
            page=c.get("page"),
            similarity_score=c.get("similarity_score"),
        )
        for c in chunks
    ]


@app.get("/stats")
async def get_stats():
    if not app_state.retriever:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    return app_state.retriever.get_stats()


@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        workers=1,
        log_level=settings.LOG_LEVEL.lower(),
        reload=False,
    )