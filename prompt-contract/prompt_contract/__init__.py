"""Versioned prompt-format primitives for training and opt-in inference.

A LoRA adapter is only valid under the exact prompt format it was trained
on. Change the system prompt, swap the order of question and context, or
render a chunk differently, and the adapter is being asked to do a task it
never saw — the degradation is silent, because nothing crashes.

This module turns that format into an explicit, versioned, hashable object:

    dataset-synth  → stores structured chunks and shared refusal texts
    lora-train     → builds the chat messages it trains on
    rag            → can build the inference prompt when passed a contract

`fingerprint()` is stored next to the adapter, so a mismatch between the
contract an adapter was trained under and the one it is served with can be
detected by an integration. The current backend and eval-runner do not load
or validate the adapter contract automatically.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

CONTRACT_FILENAME = "prompt_contract.json"

GROUNDED_SYSTEM_PROMPT = (
    "You are an expert assistant for the provided documentation. Answer the "
    "user's question using ONLY the information in the Context. If the Context "
    "does not contain enough information to answer the question reliably, say so "
    "explicitly instead of guessing. When showing code, use fenced code blocks "
    "with the appropriate language tag."
)

SOURCED_SYSTEM_PROMPT = (
    "You are an expert assistant for the provided documentation. Answer the "
    "user's question using ONLY the numbered Context snippets. Every factual "
    "claim must end with a citation of the form [§N], where N is the snippet id. "
    "If the Context does not contain enough information to answer reliably, say "
    "so explicitly and point at where to look instead of guessing. When showing "
    "code, use fenced code blocks with the appropriate language tag."
)

DEFAULT_REFUSALS = (
    "Based on the provided context, I cannot answer this question — it does not "
    "contain the relevant information.",
    "The provided context does not contain information to answer this question.",
    "I don't have enough information in the provided context to answer this reliably.",
    "This question cannot be answered from the given context.",
    "The context provided is not relevant to this question, so I cannot answer it.",
)

# Fields that actually change the token sequence the model sees. Only these
# go into the fingerprint — bumping `name` or tweaking `refusal_answers`
# should not invalidate an adapter, but reordering the template must.
_FINGERPRINT_FIELDS = (
    "system_prompt",
    "user_template",
    "chunk_template",
    "chunk_joiner",
    "context_chunks",
)


@dataclass(frozen=True)
class PromptContract:
    """The exact prompt format an adapter is trained and served under.

    `user_template` must contain both `{context}` and `{question}` — their
    order inside the template is part of the contract, because it is one of
    the things that silently breaks an adapter when it changes.

    `chunk_template` may reference `{id}`, `{source}`, `{score}` and `{text}`.
    `context_chunks` records how many chunks were rendered into one training
    example; it should match the retriever's `top_k` at serving time.
    """

    name: str = "grounded"
    version: str = "1"
    system_prompt: str = GROUNDED_SYSTEM_PROMPT
    user_template: str = "Context:\n{context}\n\nQuestion: {question}"
    chunk_template: str = "{text}"
    chunk_joiner: str = "\n\n"
    context_chunks: int = 1
    max_context_chars: int = 14000
    refusal_answers: tuple[str, ...] = field(default=DEFAULT_REFUSALS)

    def __post_init__(self) -> None:
        for placeholder in ("{context}", "{question}"):
            if placeholder not in self.user_template:
                raise ValueError(f"user_template must contain {placeholder}")
        if "{text}" not in self.chunk_template:
            raise ValueError("chunk_template must contain {text}")
        if self.context_chunks < 1:
            raise ValueError("context_chunks must be >= 1")

    # --- rendering -----------------------------------------------------

    def render_chunk(self, index: int, chunk: Mapping[str, object] | str) -> str:
        """Render one chunk. Accepts a bare string or a mapping with metadata."""
        if isinstance(chunk, str):
            fields: dict[str, object] = {"text": chunk}
        else:
            fields = dict(chunk)
        score = fields.get("score")
        return self.chunk_template.format(
            id=fields.get("id", index),
            source=fields.get("source", "N/A"),
            score=f"{float(score):.4f}" if isinstance(score, (int, float)) else "N/A",
            text=str(fields.get("text", "")).strip(),
        )

    def render_context(self, chunks: Sequence[Mapping[str, object] | str]) -> str:
        """Join rendered chunks, truncating at `max_context_chars`.

        Truncation drops whole chunks from the tail rather than cutting a
        chunk mid-sentence — a half-chunk teaches the model to answer from
        fragments.
        """
        parts: list[str] = []
        total = 0
        for i, chunk in enumerate(chunks, start=1):
            block = self.render_chunk(i, chunk)
            if parts and total + len(block) + len(self.chunk_joiner) > self.max_context_chars:
                break
            total += len(block) + (len(self.chunk_joiner) if parts else 0)
            parts.append(block)
        return self.chunk_joiner.join(parts)

    def render_user(self, question: str, context: str) -> str:
        return self.user_template.format(context=context, question=question)

    def build_messages(
        self,
        question: str,
        context: str,
        answer: str | None = None,
    ) -> list[dict[str, str]]:
        """Chat messages for training (with `answer`) or inference (without)."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.render_user(question, context)},
        ]
        if answer is not None:
            messages.append({"role": "assistant", "content": answer})
        return messages

    # --- identity + persistence ----------------------------------------

    def fingerprint(self) -> str:
        """Stable short hash over the format-defining fields only."""
        payload = {k: getattr(self, k) for k in _FINGERPRINT_FIELDS}
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def with_context_chunks(self, n: int) -> PromptContract:
        """Copy of this contract rendering `n` chunks per example."""
        return replace(self, context_chunks=n)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["refusal_answers"] = list(self.refusal_answers)
        data["fingerprint"] = self.fingerprint()
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> PromptContract:
        known = {f for f in cls.__dataclass_fields__}  # noqa: C416
        kwargs = {k: v for k, v in data.items() if k in known}
        if "refusal_answers" in kwargs:
            kwargs["refusal_answers"] = tuple(kwargs["refusal_answers"])  # type: ignore[arg-type]
        return cls(**kwargs)  # type: ignore[arg-type]

    def save(self, directory: Path | str) -> Path:
        path = Path(directory) / CONTRACT_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return path

    @classmethod
    def load(cls, path: Path | str) -> PromptContract:
        p = Path(path)
        if p.is_dir():
            p = p / CONTRACT_FILENAME
        with p.open(encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


#: Reproduces the format `lora-train` used before contracts existed, so
#: existing runs stay bit-identical when they adopt this module.
GROUNDED_CONTRACT = PromptContract()

#: Numbered chunks with sources and mandatory `[§N]` citations — use this
#: when the answer must be traceable back to a specific document.
SOURCED_CONTRACT = PromptContract(
    name="sourced",
    system_prompt=SOURCED_SYSTEM_PROMPT,
    user_template="Context snippets:\n{context}\n\nQuestion: {question}",
    chunk_template="[§{id}] {source}\n{text}",
)

CONTRACTS: dict[str, PromptContract] = {
    "grounded": GROUNDED_CONTRACT,
    "sourced": SOURCED_CONTRACT,
}


def get_contract(name: str) -> PromptContract:
    """Look up a built-in contract by name, or load one from a JSON path."""
    if name in CONTRACTS:
        return CONTRACTS[name]
    path = Path(name)
    if path.exists():
        return PromptContract.load(path)
    known = ", ".join(sorted(CONTRACTS))
    raise ValueError(f"unknown contract {name!r} (known: {known}; or pass a JSON path)")


__all__ = [
    "CONTRACTS",
    "CONTRACT_FILENAME",
    "DEFAULT_REFUSALS",
    "GROUNDED_CONTRACT",
    "SOURCED_CONTRACT",
    "PromptContract",
    "get_contract",
]
