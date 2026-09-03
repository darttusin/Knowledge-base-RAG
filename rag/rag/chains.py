from collections.abc import Sequence
from dataclasses import asdict

from prompt_contract import PromptContract

from rag.llm import ChatModel
from rag.models import RetrievedChunk
from rag.prompts import SYSTEM_INSTRUCTIONS, build_rag_prompt


def _build_messages(
    question: str,
    chunks: Sequence[RetrievedChunk],
    history: list[dict] | None,
    contract: PromptContract | None,
) -> list[dict]:
    """Assemble chat messages, honouring a prompt contract when given.

    Without a contract the legacy `build_rag_prompt` format is used, so
    existing runs stay reproducible. A LoRA adapter must be served with the
    contract it was trained under — pass the one saved next to its weights.
    """
    if contract is not None:
        context = contract.render_context([asdict(c) for c in chunks])
        messages = contract.build_messages(question, context)
        if history:
            messages[1:1] = history
        return messages

    messages: list[dict] = [{"role": "system", "content": SYSTEM_INSTRUCTIONS}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": build_rag_prompt(question, chunks)})
    return messages


def answer(
    model: ChatModel,
    question: str,
    chunks: Sequence[RetrievedChunk],
    history: list[dict] | None = None,
    contract: PromptContract | None = None,
) -> str:
    return model.invoke(_build_messages(question, chunks, history, contract))


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
    contract: PromptContract | None = None,
):
    return model.stream(_build_messages(question, chunks, history, contract))
