from retrieval.retriever import Retriever
from retrieval.context_builder import build_context
from llm.llm_client import generate_answer


def answer_query(query: str):
    retriever = Retriever(k=3)
    chunks = retriever.retrieve(query)

    if not chunks:
        return "Not found in documents."

    context = build_context(chunks)
    answer = generate_answer(context, query)

    return answer


if __name__ == "__main__":
    q = "What skills does Mahesh have?"
    print(answer_query(q))
