# tests/test_text_cleaner.py
import pytest
from ingestion.text_cleaner import clean_text


def _doc(text):
    return {"text": text, "source": "test.pdf", "page": 1}


def test_removes_page_headers():
    docs = [_doc("Page 1 of 10 Some content here")]
    cleaned = clean_text(docs)
    assert "Page 1 of 10" not in cleaned[0]["text"]
    assert "Some content here" in cleaned[0]["text"]


def test_preserves_dates():
    docs = [_doc("Invoice date: 01/01/2024")]
    cleaned = clean_text(docs)
    assert "01/01/2024" in cleaned[0]["text"]


def test_preserves_fractions():
    docs = [_doc("Mix 1/2 cup of flour")]
    cleaned = clean_text(docs)
    assert "1/2" in cleaned[0]["text"]


def test_collapses_whitespace():
    docs = [_doc("Hello   world\n\nfoo")]
    cleaned = clean_text(docs)
    assert cleaned[0]["text"] == "Hello world foo"


def test_empty_text_excluded():
    docs = [_doc("   ")]
    cleaned = clean_text(docs)
    assert cleaned == []


def test_passthrough_metadata():
    docs = [{"text": "Some text", "source": "resume.pdf", "page": 3}]
    cleaned = clean_text(docs)
    assert cleaned[0]["source"] == "resume.pdf"
    assert cleaned[0]["page"] == 3
