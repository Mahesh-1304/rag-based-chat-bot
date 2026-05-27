# api/main.py
"""
FastAPI server for the RAG Document Chatbot.
Provides REST endpoints for document retrieval and answer generation.
"""

import logging
from typing import Optional, List
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn

from config.settings import settings
from retrieval.retriever import Retriever
from llm.response_generator import answer_query

# ============================================================================
# Setup Logging
# ============================================================================

logging.basicConfig(
    level=settings.LOG_LEVEL,
    format=settings.LOG_FORMAT
)
logger = logging.getLogger(__name__)

# ============================================================================
# Global State
# ============================================================================

class AppState:
    """Global application state."""
    retriever: Optional[Retriever] = None
    initialized: bool = False


app_state = AppState()

# ============================================================================
# Lifespan Management (FastAPI 0.93+)
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    
    # Startup
    logger.info("Application starting up...")
    try:
        # Initialize retriever
        logger.info("Loading vector store and retriever...")
        app_state.retriever = Retriever(
            index_path=str(settings.VECTOR_STORE_DIR / "index.faiss"),
            metadata_path=str(settings.VECTOR_STORE_DIR / "metadata.json"),
            top_k=settings.RETRIEVER_TOP_K,
            score_threshold=settings.RETRIEVER_SCORE_THRESHOLD
        )
        
        stats = app_state.retriever.get_stats()
        logger.info(f"✓ Retriever initialized with {stats['total_chunks']} chunks")
        app_state.initialized = True
    
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        app_state.initialized = False
        raise
    
    yield  # Application runs here
    
    # Shutdown
    logger.info("Application shutting down...")
    # Cleanup if needed
    logger.info("✓ Application shutdown complete")

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="RAG Document Chatbot API",
    description="Retrieve information from documents using semantic search and LLM",
    version="1.0.0",
    lifespan=lifespan
)

# ============================================================================
# Pydantic Models
# ============================================================================

class QueryRequest(BaseModel):
    """Request model for document queries."""
    
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="The question to ask about the documents"
    )
    top_k: Optional[int] = Field(
        None,
        ge=1,
        le=10,
        description="Number of documents to retrieve (optional)"
    )
    include_scores: Optional[bool] = Field(
        False,
        description="Include similarity scores in response"
    )
    
    @validator('question')
    def question_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Question cannot be empty or whitespace")
        return v.strip()


class RetrievedChunk(BaseModel):
    """A single retrieved document chunk."""
    
    chunk_id: str
    text: str
    source: str
    page: Optional[int]
    similarity_score: Optional[float] = None


class QueryResponse(BaseModel):
    """Response model for queries."""
    
    question: str
    answer: str
    retrieved_chunks: List[RetrievedChunk]
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    processing_time_ms: Optional[float] = None


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    initialized: bool
    retriever_stats: Optional[dict] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    Returns status of application and vector store.
    """
    try:
        retriever_stats = None
        if app_state.retriever:
            retriever_stats = app_state.retriever.get_stats()
        
        return HealthResponse(
            status="healthy" if app_state.initialized else "degraded",
            initialized=app_state.initialized,
            retriever_stats=retriever_stats
        )
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Main query endpoint.
    Retrieves relevant documents and generates an answer using the LLM.
    
    Args:
        request: QueryRequest containing the question
        
    Returns:
        QueryResponse with answer and retrieved chunks
        
    Raises:
        HTTPException: If retriever not initialized or query fails
    """
    import time
    
    if not app_state.initialized or not app_state.retriever:
        raise HTTPException(
            status_code=503,
            detail="Application not initialized. Vector store may be missing."
        )
    
    start_time = time.time()
    
    try:
        logger.info(f"Processing query: {request.question[:50]}...")
        
        # Step 1: Retrieve relevant chunks
        top_k = request.top_k or settings.RETRIEVER_TOP_K
        retrieved = app_state.retriever.retrieve(request.question, top_k=top_k)
        
        if not retrieved:
            logger.warning(f"No relevant documents found for query")
            return QueryResponse(
                question=request.question,
                answer="I could not find relevant information in the documents to answer your question.",
                retrieved_chunks=[]
            )
        
        # Step 2: Build context from retrieved chunks
        logger.info(f"Retrieved {len(retrieved)} chunks")
        
        # Step 3: Generate answer using LLM
        logger.info("Generating answer with LLM...")
        context = "\n\n---\n\n".join([
            f"[Source: {c['source']}, Page: {c['page']}]\n{c['text']}"
            for c in retrieved
        ])
        
        answer = answer_query(request.question, context)
        
        # Step 4: Prepare response
        chunks = [
            RetrievedChunk(
                chunk_id=c.get("chunk_id", "unknown"),
                text=c["text"],
                source=c["source"],
                page=c.get("page"),
                similarity_score=c.get("similarity_score") if request.include_scores else None
            )
            for c in retrieved
        ]
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        response = QueryResponse(
            question=request.question,
            answer=answer,
            retrieved_chunks=chunks,
            processing_time_ms=processing_time
        )
        
        logger.info(f"✓ Query processed in {processing_time:.2f}ms")
        return response
    
    except ValueError as e:
        logger.warning(f"Invalid query: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"Query processing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal error: {str(e)}"
        )


@app.get("/retrieve", response_model=List[RetrievedChunk])
async def retrieve_endpoint(
    query: str = Query(..., min_length=3, max_length=500, description="Search query"),
    top_k: int = Query(3, ge=1, le=10, description="Number of results")
):
    """
    Retrieve documents without generating an answer.
    Useful for exploring what's in the vector store.
    
    Args:
        query: The search query
        top_k: Number of results to return
        
    Returns:
        List of retrieved chunks with metadata
    """
    if not app_state.initialized or not app_state.retriever:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    
    try:
        logger.info(f"Retrieve-only query: {query[:50]}...")
        chunks = app_state.retriever.retrieve(query, top_k=top_k)
        
        return [
            RetrievedChunk(
                chunk_id=c.get("chunk_id", "unknown"),
                text=c["text"],
                source=c["source"],
                page=c.get("page"),
                similarity_score=c.get("similarity_score")
            )
            for c in chunks
        ]
    
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get statistics about the vector store and retriever."""
    if not app_state.retriever:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    
    return app_state.retriever.get_stats()


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle ValueError exceptions."""
    logger.warning(f"Value error: {exc}")
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    logger.info("Starting RAG Document Chatbot API server...")
    logger.info(f"Configuration: {settings}")
    
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        workers=settings.API_WORKERS,
        log_level=settings.LOG_LEVEL.lower(),
        reload=False  # Set to True for development
    )
