from collections.abc import Sequence

from rag.llm import ChatModel
from rag.models import RetrievedChunk
from rag.prompts import SYSTEM_INSTRUCTIONS, build_rag_prompt


def answer(
    model: ChatModel,
    question: str,
    chunks: Sequence[RetrievedChunk],
    history: list[dict] | None = None,
) -> str:
    prompt = build_rag_prompt(question, chunks)

    # Build message list: system + history + current question
    messages = [{"role": "system", "content": SYSTEM_INSTRUCTIONS}]

    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": prompt})

    return model.invoke(messages)


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
    history: list[dict] | None = None,
):
    prompt = build_rag_prompt(question, chunks)

    # Build message list: system + history + current question
    messages = [{"role": "system", "content": SYSTEM_INSTRUCTIONS}]

    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": prompt})

    return model.stream(messages)
