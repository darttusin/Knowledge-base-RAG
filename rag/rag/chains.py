from collections.abc import Sequence

from rag.llm import ChatModel
from rag.models import RetrievedChunk
from rag.prompts import SYSTEM_INSTRUCTIONS, build_rag_prompt


def answer(
    model: ChatModel,
    question: str,
    chunks: Sequence[RetrievedChunk],
) -> str:
    prompt = build_rag_prompt(question, chunks)
    return model.invoke(
        [
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": prompt},
        ]
    )


def answer_without_context(model: ChatModel, question: str) -> str:
    return model.invoke(
        [
            {
                "role": "system",
                "content": "You are a helpful AI assistant specializing in PyTorch.",
            },
            {"role": "user", "content": question},
        ]
    )


def answer_stream(
    model: ChatModel,
    question: str,
    chunks: Sequence[RetrievedChunk],
):
    prompt = build_rag_prompt(question, chunks)
    return model.stream(
        [
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": prompt},
        ]
    )
