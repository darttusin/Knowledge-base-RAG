from __future__ import annotations

import pytest
from prompt_contract import (
    CONTRACT_FILENAME,
    GROUNDED_CONTRACT,
    SOURCED_CONTRACT,
    PromptContract,
    get_contract,
)

CHUNKS = [
    {"id": 1, "source": "docs/a.md", "text": "Alpha body", "score": 0.5},
    {"id": 2, "source": "docs/b.md", "text": "Beta body"},
]


def test_user_template_must_carry_both_placeholders():
    with pytest.raises(ValueError, match="context"):
        PromptContract(user_template="Question: {question}")
    with pytest.raises(ValueError, match="question"):
        PromptContract(user_template="Context:\n{context}")


def test_chunk_template_must_carry_text():
    with pytest.raises(ValueError, match="text"):
        PromptContract(chunk_template="[{id}] {source}")


def test_render_context_joins_every_chunk():
    rendered = GROUNDED_CONTRACT.render_context(CHUNKS)
    assert "Alpha body" in rendered
    assert "Beta body" in rendered


def test_sourced_contract_renders_ids_and_sources():
    rendered = SOURCED_CONTRACT.render_context(CHUNKS)
    assert "[§1] docs/a.md" in rendered
    assert "[§2] docs/b.md" in rendered


def test_render_context_drops_whole_chunks_when_truncating():
    contract = PromptContract(max_context_chars=15)
    rendered = contract.render_context([{"text": "x" * 10}, {"text": "y" * 10}])
    # the second chunk would overflow, so it is dropped rather than cut
    assert rendered == "x" * 10


def test_render_context_keeps_first_chunk_even_when_oversized():
    contract = PromptContract(max_context_chars=5)
    assert contract.render_context([{"text": "x" * 50}]) == "x" * 50


def test_build_messages_shape():
    ctx = GROUNDED_CONTRACT.render_context(CHUNKS)
    train = GROUNDED_CONTRACT.build_messages("Q?", ctx, answer="A.")
    assert [m["role"] for m in train] == ["system", "user", "assistant"]
    assert "Q?" in train[1]["content"]

    infer = GROUNDED_CONTRACT.build_messages("Q?", ctx)
    assert [m["role"] for m in infer] == ["system", "user"]


def test_fingerprint_tracks_format_not_labels():
    renamed = PromptContract(name="something-else", version="9")
    assert renamed.fingerprint() == GROUNDED_CONTRACT.fingerprint()

    reordered = PromptContract(user_template="Question: {question}\n\nContext:\n{context}")
    assert reordered.fingerprint() != GROUNDED_CONTRACT.fingerprint()

    assert (
        GROUNDED_CONTRACT.with_context_chunks(5).fingerprint()
        != GROUNDED_CONTRACT.fingerprint()
    )


def test_save_load_round_trip(tmp_path):
    contract = GROUNDED_CONTRACT.with_context_chunks(3)
    path = contract.save(tmp_path)
    assert path.name == CONTRACT_FILENAME

    assert PromptContract.load(tmp_path) == contract
    assert PromptContract.load(path) == contract
    assert get_contract(str(path)).fingerprint() == contract.fingerprint()


def test_get_contract_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown contract"):
        get_contract("no-such-contract")
