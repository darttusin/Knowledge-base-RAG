from dataclasses import dataclass
from typing import Any, Sequence

from google import genai
from settings import settings


@dataclass
class RetrievedChunk:
    id: int
    text: str
    source: str
    score: float


SYSTEM_INSTRUCTIONS = """You are an assistant that answers questions about PyTorch 2.x and its ecosystem.
You must only use the context snippets below (PyTorch docs + curated StackOverflow answers).
If the context is not enough to answer, say you don't know and suggest where to look in the official docs.
Never invent APIs, arguments, or behavior that are not supported by the context."""

ANSWER_INSTRUCTIONS = """Answer format:
1) First, give a concise direct answer in 2–4 sentences.
2) Then provide a bullet list with details and short code examples if helpful.
3) Each bullet with factual claims must end with a citation in the form [§N], where N is the context id.
4) If multiple snippets support the same point, you can use [§1, §3].
5) After the bullets, add a small 'Where to read more' list with file paths or URLs."""


def render_context(chunks: Sequence[RetrievedChunk], max_chars: int = 14000) -> str:
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


def build_rag_prompt(question: str, chunks: Sequence[RetrievedChunk]) -> str:
    context_block = render_context(chunks)
    joined_sources = "\n".join(f"[§{c.id}] {c.source}" for c in chunks)

    prompt = f"""{SYSTEM_INSTRUCTIONS}

{ANSWER_INSTRUCTIONS}

User question:
{question}

Context snippets:
{context_block}

Remember:
- Base the answer only on the context snippets above.
- If the answer is not in the context, say that you don't know and suggest checking the relevant section in the PyTorch docs.
- Use the citation format [§N] that corresponds to the snippet ids.

List of snippet ids and sources:
{joined_sources}
"""
    return prompt


class GeminiGenerator:
    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        api_key: str | None = None,
        temperature: float = 0.1,
        max_output_tokens: int = 1024,
    ) -> None:
        if api_key is None:
            self.client = genai.Client()
        else:
            self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def generate_answer(
        self,
        question: str,
        chunks: Sequence[RetrievedChunk],
    ) -> str:
        prompt = build_rag_prompt(question, chunks)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_output_tokens,
            },
        )

        return response.text  # type: ignore


gemini = GeminiGenerator(api_key=settings.GEMINI_API_KEY, model_name="gemini-2.5-flash")


def rag_answer(query: str, documents: dict[str, list[Any]]) -> str:
    docs = documents["documents"][0]
    metas = documents["metadatas"][0]
    dists = documents["distances"][0]

    chunks: list[RetrievedChunk] = []
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
        chunks.append(
            RetrievedChunk(
                id=i + 1,
                text=doc,
                source=meta.get("source", "N/A"),
                score=1.0 - float(dist),
            )
        )

    if not chunks:
        return "I couldn't retrieve any relevant context for this question from the knowledge base."

    return gemini.generate_answer(query, chunks)
