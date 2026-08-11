"""Context assembly is where a grounded dataset is won or lost.

If the gold chunk is always first, or the distractors come from an
unrelated part of the corpus, the model learns a shortcut instead of
learning to read the context it is given.
"""

from __future__ import annotations

import json
import sys
import types

import pytest

# `dataset_synth.chunks` imports chromadb only to read a collection, which
# these tests never touch.
sys.modules.setdefault("chromadb", types.ModuleType("chromadb"))

from dataset_synth.chunks import Chunk  # noqa: E402
from dataset_synth.config import SynthConfig  # noqa: E402
from dataset_synth.pipeline import (  # noqa: E402
    ADVERSARIAL_ANSWERS,
    Record,
    _add_adversarial,
    _assemble_contexts,
    _split_and_write,
)

K = 5


@pytest.fixture
def chunks() -> list[Chunk]:
    return [
        Chunk(text=f"Body {i} about api_{i}.", source=f"docs/mod{i % 3}/f{i}.md")
        for i in range(30)
    ]


@pytest.fixture
def records(chunks: list[Chunk]) -> list[Record]:
    return [
        Record(
            question=f"What does api_{i} do?",
            answer=f"api_{i} does thing {i}.",
            context=chunks[i].text,
            is_adversarial=False,
            source=chunks[i].source,
        )
        for i in range(20)
    ]


@pytest.fixture
def cfg() -> SynthConfig:
    return SynthConfig(context_chunks=K, adversarial_fraction=0.25, seed=42)


def test_every_example_gets_k_unique_chunks(records, chunks, cfg):
    _assemble_contexts(records, chunks, cfg)
    for r in records:
        assert len(r.chunks) == K
        assert len({c["text"] for c in r.chunks}) == K


def test_gold_chunk_is_always_present(records, chunks, cfg):
    _assemble_contexts(records, chunks, cfg)
    for r in records:
        assert any(c["text"] == r.context for c in r.chunks)


def test_gold_position_varies(records, chunks, cfg):
    _assemble_contexts(records, chunks, cfg)
    positions = {
        next(i for i, c in enumerate(r.chunks) if c["text"] == r.context) for r in records
    }
    assert len(positions) > 1, "gold chunk always lands in the same slot"


def test_chunk_ids_match_render_order(records, chunks, cfg):
    _assemble_contexts(records, chunks, cfg)
    for r in records:
        assert [c["id"] for c in r.chunks] == list(range(1, K + 1))


def test_adversarial_examples_exclude_their_own_gold(records, chunks, cfg):
    _assemble_contexts(records, chunks, cfg)
    gold_by_question = {r.question: r.context for r in records}

    combined = _add_adversarial(records, chunks, cfg)
    adversarial = [r for r in combined if r.is_adversarial]

    assert adversarial
    for r in adversarial:
        assert len(r.chunks) == K
        assert r.answer in ADVERSARIAL_ANSWERS
        assert all(c["text"] != gold_by_question[r.question] for c in r.chunks)


def test_written_rows_carry_structured_chunks(records, chunks, cfg, tmp_path):
    _assemble_contexts(records, chunks, cfg)
    combined = _add_adversarial(records, chunks, cfg)

    write_cfg = SynthConfig(output_dir=str(tmp_path), val_fraction=0.1, seed=42)
    summary = _split_and_write(combined, write_cfg)
    assert summary["train"] > 0

    rows = [json.loads(ln) for ln in (tmp_path / "train.jsonl").open(encoding="utf-8")]
    for row in rows:
        assert {"question", "answer", "chunks", "is_adversarial", "source"} <= set(row)
        assert row["chunks"], "chunks must not be empty"
