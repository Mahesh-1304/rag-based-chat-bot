# retrieval/context_builder.py
"""
Builds a formatted context string from retrieved document chunks.
"""
from typing import List, Dict


def build_context(chunks: List[Dict]) -> str:
    """
    Format a list of retrieved chunks into a single context string for the LLM.

    Args:
        chunks: List of chunk dicts with 'source', 'page', and 'text' keys.

    Returns:
        Multi-block context string separated by '---', or empty string if no chunks.
    """
    if not chunks:
        return ""

    blocks = [
        f"[Source: {c['source']}, Page: {c.get('page')}]\n{c['text']}"
        for c in chunks
    ]
    return "\n\n---\n\n".join(blocks)


if __name__ == "__main__":
    sample = [
        {"source": "resume.pdf", "page": 1, "text": "Mahesh Ubarhande is an aspiring business analyst..."}
    ]
    print(build_context(sample))