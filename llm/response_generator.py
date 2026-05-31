# llm/response_generator.py
"""
Response generator that uses pre-retrieved context and the LLM client.
Called by api/main.py which handles retrieval separately.
"""

import logging
from llm.llm_client import generate_answer
from llm.prompt_templates import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def answer_query(query: str, context: str) -> str:
    """
    Generate an answer for a query given pre-built context.

    Args:
        query:   The user's question.
        context: Retrieved document chunks already formatted as a string.

    Returns:
        A string answer from the LLM (or fallback).
    """
    if not context or not context.strip():
        logger.warning("answer_query called with empty context")
        return "Not found in documents."

    logger.info(f"Generating answer for query: {query[:60]}...")
    return generate_answer(context=context, query=query)