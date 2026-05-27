def build_context(chunks: list) -> str:
    context_blocks = []

    for c in chunks:
        block = (
            f"[Source: {c['source']}, Page: {c['page']}]\n"
            f"{c['text']}"
        )
        context_blocks.append(block)

    return "\n\n---\n\n".join(context_blocks)


if __name__ == "__main__":
    sample_chunks = [
        {
            "source": "resume.pdf",
            "page": 1,
            "text": "Mahesh Ubarhande is an aspiring business analyst..."
        }
    ]

    print(build_context(sample_chunks))
