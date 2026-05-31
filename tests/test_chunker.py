# tests/test_chunker.py
import pytest
from ingestion.chunker import chunk_text


SAMPLE_DOCS = [
    {"text": " ".join([f"word{i}" for i in range(500)]), "source": "test.pdf", "page": 1},
    {"text": "Short sentence.", "source": "test.pdf", "page": 2},
]


def test_chunk_text_returns_list():
    chunks = chunk_text(SAMPLE_DOCS, chunk_size=100, overlap=10)
    assert isinstance(chunks, list)
    assert len(chunks) > 0


def test_chunks_have_required_keys():
    chunks = chunk_text(SAMPLE_DOCS, chunk_size=100, overlap=10)
    for chunk in chunks:
        assert "chunk_id" in chunk
        assert "text" in chunk
        assert "source" in chunk
        assert "page" in chunk


def test_chunk_ids_are_unique():
    chunks = chunk_text(SAMPLE_DOCS, chunk_size=100, overlap=10)
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids)), "Chunk IDs must be unique"


def test_short_doc_produces_one_chunk():
    docs = [{"text": "Hello world", "source": "a.pdf", "page": 1}]
    chunks = chunk_text(docs, chunk_size=400, overlap=50)
    assert len(chunks) == 1


def test_empty_documents_list():
    chunks = chunk_text([], chunk_size=400, overlap=50)
    assert chunks == []
