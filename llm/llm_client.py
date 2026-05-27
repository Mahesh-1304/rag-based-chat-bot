import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def generate_answer(context: str, query: str) -> str:
    """
    Generate answer using OpenAI API.
    
    Falls back to a simple rule-based answer if API key is not configured.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        logger.warning("OPENAI_API_KEY not set. Using fallback answer.")
        return fallback_answer(context, query)
    
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a document-based assistant. Answer ONLY using the provided context. If the answer is not in the context, say 'Not found in documents.'"
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
                }
            ],
            max_tokens=256,
            temperature=0.0
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return fallback_answer(context, query)


def fallback_answer(context: str, query: str) -> str:
    """
    Fallback answer when OpenAI API is not available.
    Returns a simple response based on context.
    """
    if not context or not context.strip():
        return "Not found in documents."
    
    # Simple heuristic: if query words are in context, return it
    query_words = set(query.lower().split())
    context_lower = context.lower()
    
    # Count matching words
    matches = sum(1 for word in query_words if word in context_lower)
    
    if matches > 0:
        return f"Based on the documents: {context[:300]}..."
    else:
        return "Not found in documents."
