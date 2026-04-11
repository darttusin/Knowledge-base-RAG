from collections.abc import Sequence

from rag.models import RetrievedChunk

SYSTEM_INSTRUCTIONS = (
    "You are an assistant that answers questions about PyTorch 2.x and its ecosystem.\n"
    "You must only use the context snippets below (PyTorch docs + curated StackOverflow answers).\n"
    "If the context is not enough to answer, say you don't know and suggest where to look "
    "in the official docs.\n"
    "Never invent APIs, arguments, or behavior that are not supported by the context."
)

ANSWER_INSTRUCTIONS = (
    "Answer format:\n"
    "1) First, give a concise direct answer in 2-4 sentences.\n"
    "2) Then provide a bullet list with details and short code examples if helpful.\n"
    "3) Each bullet with factual claims must end with a citation in the form "
    "[§N], where N is the context id.\n"
    "4) If multiple snippets support the same point, you can use [§1, §3].\n"
    "5) After the bullets, add a small 'Where to read more' list with file paths or URLs."
)


def render_context(
    chunks: Sequence[RetrievedChunk],
    max_chars: int = 14000,
) -> str:
    parts: list[str] = []
    total_len = 0

    for ch in chunks:
        header = f"[{ch.id}] {ch.source} (score={ch.score:.4f})"
        body = ch.text.strip()
        block = header + "\n" + body + "\n"
        if total_len + len(block) > max_chars:
            break
        parts.append(block)
        total_len += len(block)

    return "\n\n".join(parts)


def build_rag_prompt(
    question: str,
    chunks: Sequence[RetrievedChunk],
) -> str:
    context_block = render_context(chunks)
    joined_sources = "\n".join(f"[§{c.id}] {c.source}" for c in chunks)

    return (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"{ANSWER_INSTRUCTIONS}\n\n"
        f"User question:\n{question}\n\n"
        f"Context snippets:\n{context_block}\n\n"
        "Remember:\n"
        "- Base the answer only on the context snippets above.\n"
        "- If the answer is not in the context, say that you don't know and suggest "
        "checking the relevant section in the PyTorch docs.\n"
        "- Use the citation format [§N] that corresponds to the snippet ids.\n\n"
        f"List of snippet ids and sources:\n{joined_sources}\n"
    )
