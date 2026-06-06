"""
Constructs the prompt sent to the LLM.

This is where grounding happens — the system prompt explicitly instructs the
model to only use the provided context. Understanding this module deeply is
what separates RAG engineers from people who just call LangChain.

Key concepts:
- System prompt:  Sets behaviour for the entire conversation.
- Context block:  The retrieved chunks, formatted clearly with source info.
- User question:  The original query.

Interview question: "How do you prevent hallucination in RAG?"
Answer lives here: the system prompt + the fallback behaviour when context is weak.
"""


SYSTEM_PROMPT = """You are a precise document assistant. Your job is to answer questions
using ONLY the context passages provided below. 

Rules you must follow:
1. If the answer is clearly in the context, answer directly and cite the source.
2. If the context is partially relevant, use what is there and acknowledge limits.
3. If the context contains no relevant information, say exactly:
   "I could not find relevant information in the provided documents."
   Do NOT guess, infer beyond the text, or use prior knowledge.
4. Always mention which document and page your answer comes from.
"""


def build_prompt(query: str, chunks: list[dict]) -> str:
    """
    Assemble the full prompt from retrieved chunks + user query.

    Args:
        query:  The user's question.
        chunks: Retrieved chunks from Retriever.retrieve()

    Returns:
        A single formatted string ready to send to the LLM.
    """
    if not chunks:
        context_block = "No relevant context was retrieved."
    else:
        context_parts = []
        for i, chunk in enumerate(chunks, start=1):
            context_parts.append(
                f"[Source {i}: {chunk['source']}, Page {chunk['page']} | Similarity: {chunk['score']}]\n"
                f"{chunk['text']}"
            )
        context_block = "\n\n---\n\n".join(context_parts)

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"=== CONTEXT ===\n{context_block}\n\n"
        f"=== QUESTION ===\n{query}\n\n"
        f"=== ANSWER ==="
    )
    return prompt


def build_messages(query: str, chunks: list[dict]) -> list[dict]:
    """
    Build chat-style messages (system/user) for models that use message format.
    Used with HuggingFace chat-template models.
    """
    context_block = _format_context(chunks)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context:\n{context_block}\n\nQuestion: {query}",
        },
    ]


def _format_context(chunks: list[dict]) -> str:
    if not chunks:
        return "No relevant context was retrieved."
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(f"[{chunk['source']} | p{chunk['page']}]\n{chunk['text']}")
    return "\n\n".join(parts)
