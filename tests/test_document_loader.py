# tests/test_document_loader.py
import pytest
from pathlib import Path
from ingestion.document_loader import load_documents


def test_load_documents_empty_dir(tmp_path):
    docs, errors = load_documents(tmp_path)
    assert docs == []
    assert errors == []


def test_load_documents_returns_tuple(tmp_path):
    result = load_documents(tmp_path)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_load_documents_unsupported_files_ignored(tmp_path):
    (tmp_path / "notes.txt").write_text("Hello")
    docs, errors = load_documents(tmp_path)
    assert docs == []
    assert errors == []
