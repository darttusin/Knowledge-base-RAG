"""Orchestration tests: stage order, artifact reuse and recorded metadata.

The stages themselves are replaced with fakes — what matters here is that
the pipeline runs them in the right order, honours skips, and records selected
config/contract/stage fields. These tests do not prove full reproducibility.
"""

from __future__ import annotations

import json

import pytest
from lora_pipeline import pipeline as pipe_mod
from lora_pipeline.__main__ import _build_parser, _config_from_args
from prompt_contract import CONTRACT_FILENAME, PromptContract


@pytest.fixture
def docs_dir(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "a.md").write_text("# Alpha\nAlpha does things.")
    return d


def _cfg(docs_dir, tmp_path, *extra):
    argv = [
        "--docs-dir", str(docs_dir),
        "--output-dir", str(tmp_path / "run"),
        "--teacher-api-url", "http://localhost:9/v1",
        "--teacher-model", "my-teacher",
        "--context-chunks", "4",
        *extra,
    ]
    return _config_from_args(_build_parser().parse_args(argv))


@pytest.fixture
def calls(monkeypatch):
    seen: list[str] = []

    def fake(name):
        def _f(cfg):
            seen.append(name)
            if name == "synth":
                cfg.dataset_dir.mkdir(parents=True, exist_ok=True)
                cfg.train_jsonl.write_text('{"question":"q","answer":"a","chunks":[]}\n')
            return {"stage": name}
        return _f

    monkeypatch.setattr(pipe_mod, "step_ingest", fake("ingest"))
    monkeypatch.setattr(pipe_mod, "step_synth", fake("synth"))
    monkeypatch.setattr(pipe_mod, "step_train", fake("train"))
    monkeypatch.setattr(pipe_mod, "preflight_teacher", lambda c: seen.append("preflight"))
    return seen


# --- CLI wiring ---------------------------------------------------------

def test_list_arguments_are_parsed(docs_dir, tmp_path):
    cfg = _cfg(docs_dir, tmp_path, "--ext", "md,txt", "--lora-targets", "q_proj,v_proj")
    assert cfg.extensions == ("md", "txt")
    assert cfg.lora_targets == ("q_proj", "v_proj")


def test_all_linear_stays_a_string(docs_dir, tmp_path):
    # PEFT reads "all-linear" as a directive; a list would target characters
    assert _cfg(docs_dir, tmp_path).lora_targets == "all-linear"


def test_context_chunks_reaches_the_contract(docs_dir, tmp_path):
    assert _cfg(docs_dir, tmp_path).resolved_contract().context_chunks == 4


# --- validation ---------------------------------------------------------

def test_missing_docs_dir_is_rejected(tmp_path):
    cfg = _config_from_args(_build_parser().parse_args(
        ["--docs-dir", str(tmp_path / "nope"), "--output-dir", str(tmp_path / "r"),
         "--teacher-api-url", "http://x/v1"]
    ))
    with pytest.raises(FileNotFoundError):
        cfg.validate()


def test_teacher_url_required_only_when_generating(docs_dir, tmp_path):
    cfg = _config_from_args(_build_parser().parse_args(
        ["--docs-dir", str(docs_dir), "--output-dir", str(tmp_path / "r")]
    ))
    with pytest.raises(ValueError, match="teacher_api_url"):
        cfg.validate()

    cfg.skip_synth = True
    cfg.validate()


# --- orchestration ------------------------------------------------------

def test_stages_run_in_order(docs_dir, tmp_path, calls):
    stages = pipe_mod.run_pipeline(_cfg(docs_dir, tmp_path))
    assert calls == ["preflight", "ingest", "synth", "train"]
    assert set(stages) == {"ingest", "synth", "train"}
    assert all("duration_sec" in s for s in stages.values())


def test_skip_train_stops_after_the_dataset(docs_dir, tmp_path, calls):
    pipe_mod.run_pipeline(_cfg(docs_dir, tmp_path, "--skip-train"))
    assert "train" not in calls


def test_skipped_stages_still_allow_training(docs_dir, tmp_path, calls):
    seeded = _cfg(docs_dir, tmp_path)
    seeded.dataset_dir.mkdir(parents=True, exist_ok=True)
    seeded.train_jsonl.write_text("{}\n")

    pipe_mod.run_pipeline(_cfg(docs_dir, tmp_path, "--skip-ingest", "--skip-synth"))
    assert calls == ["train"]


def test_training_without_a_dataset_fails_loudly(docs_dir, tmp_path, calls):
    cfg = _cfg(docs_dir, tmp_path, "--skip-ingest", "--skip-synth")
    with pytest.raises(FileNotFoundError, match="no training data"):
        pipe_mod.run_pipeline(cfg)


# --- provenance ---------------------------------------------------------

def test_contract_is_saved_beside_the_artifacts(docs_dir, tmp_path, calls):
    cfg = _cfg(docs_dir, tmp_path)
    pipe_mod.run_pipeline(cfg)

    saved = PromptContract.load(cfg.output_dir)
    assert (cfg.output_dir / CONTRACT_FILENAME).exists()
    assert saved.fingerprint() == cfg.resolved_contract().fingerprint()
    assert saved.context_chunks == 4


def test_manifest_records_the_run(docs_dir, tmp_path, calls):
    cfg = _cfg(docs_dir, tmp_path)
    pipe_mod.run_pipeline(cfg)

    manifest = json.loads(cfg.manifest_path.read_text())
    assert manifest["teacher_model"] == "my-teacher"
    assert manifest["base_model"] == cfg.base_model
    assert manifest["contract"]["fingerprint"] == cfg.resolved_contract().fingerprint()
    assert set(manifest["stages"]) == {"ingest", "synth", "train"}
    assert manifest["config"]["docs_dir"] == str(docs_dir)
