import logging
from llm.prompt_templates import SYSTEM_PROMPT
from config.settings import settings

logger = logging.getLogger(__name__)


def generate_answer(context: str, query: str) -> str:
    try:
        from openai import OpenAI

        client = OpenAI(
            base_url=settings.OLLAMA_BASE_URL,
            api_key="ollama",
        )

        user_message = (
            f"Use the following context to answer the question.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {query}\n\n"
            f"ANSWER (based only on the context above):"
        )

        response = client.chat.completions.create(
            model=settings.OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=settings.LLM_MAX_TOKENS,
            temperature=settings.LLM_TEMPERATURE,
        )

        answer = response.choices[0].message.content.strip()
        logger.info(f"Ollama responded: {answer[:80]}...")
        return answer

    except Exception as e:
        logger.error(f"Ollama request failed: {e}")
        return _fallback_answer(context, query)


def _fallback_answer(context: str, query: str) -> str:
    if not context or not context.strip():
        return "Not found in documents."
    query_words = set(query.lower().split())
    matches = sum(1 for word in query_words if word in context.lower())
    if matches > 0:
        return f"Based on the documents: {context[:400]}..."
    return "Not found in documents."