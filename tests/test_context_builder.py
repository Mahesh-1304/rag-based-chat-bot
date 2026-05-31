# tests/test_context_builder.py
from retrieval.context_builder import build_context


def test_build_context_single_chunk():
    chunks = [{"source": "resume.pdf", "page": 1, "text": "Mahesh is a developer."}]
    ctx = build_context(chunks)
    assert "resume.pdf" in ctx
    assert "Mahesh is a developer." in ctx


def test_build_context_separator():
    chunks = [
        {"source": "a.pdf", "page": 1, "text": "First"},
        {"source": "b.pdf", "page": 2, "text": "Second"},
    ]
    ctx = build_context(chunks)
    assert "---" in ctx


def test_build_context_empty():
    assert build_context([]) == ""
