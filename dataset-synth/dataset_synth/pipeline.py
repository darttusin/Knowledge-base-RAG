"""End-to-end synthetic dataset pipeline.

chunks → teacher Q&A (concurrent) → dedup → adversarial → optional SO mix
→ train/val split → JSONL.

Output schema matches dataset-prep / what lora-train expects:
    {"question": ..., "answer": ..., "context": ..., "is_adversarial": bool, "source": ...}
"""

from __future__ import annotations

import json
import random
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from dataset_synth.chunks import Chunk, load_chunks
from dataset_synth.config import SynthConfig
from dataset_synth.teacher import Teacher

ADVERSARIAL_ANSWERS = (
    "Based on the provided context, I cannot answer this question — it does not "
    "contain the relevant information.",
    "The provided context does not contain information to answer this question.",
    "I don't have enough information in the provided context to answer this reliably.",
    "This question cannot be answered from the given context.",
    "The context provided is not relevant to this question, so I cannot answer it.",
)

_NORM_RE = re.compile(r"[^a-z0-9\s]+")
_WS_RE = re.compile(r"\s+")


@dataclass
class Record:
    question: str
    answer: str
    context: str
    is_adversarial: bool
    source: str

    def to_json(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "context": self.context,
            "is_adversarial": self.is_adversarial,
            "source": self.source,
        }


def _normalize_q(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).lower()
    text = _NORM_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def _generate_all(chunks: list[Chunk], teacher: Teacher, max_workers: int) -> list[Record]:
    """Concurrent teacher generation over all chunks."""
    records: list[Record] = []

    def _work(chunk: Chunk) -> list[Record]:
        pairs = teacher.generate(chunk.text)
        return [
            Record(
                question=p.question,
                answer=p.answer,
                context=chunk.text,
                is_adversarial=False,
                source=chunk.source,
            )
            for p in pairs
        ]

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_work, c): c for c in chunks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="teacher gen"):
            records.extend(fut.result())

    logger.info("generated {n} raw pairs from {c} chunks", n=len(records), c=len(chunks))
    return records


def _dedup(records: list[Record]) -> list[Record]:
    seen: set[str] = set()
    out: list[Record] = []
    for r in records:
        key = _normalize_q(r.question)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(r)
    logger.info("dedup: {a} → {b} unique questions", a=len(records), b=len(out))
    return out


def _add_adversarial(records: list[Record], chunks: list[Chunk], cfg: SynthConfig) -> list[Record]:
    rng = random.Random(cfg.seed)
    n_adv = int(len(records) * cfg.adversarial_fraction)
    if n_adv <= 0 or len(chunks) < 2:
        return records

    chunk_texts = [c.text for c in chunks]
    sampled = rng.sample(records, n_adv)
    adversarial: list[Record] = []
    for r in sampled:
        # pick a context chunk that is NOT the one that produced this question
        wrong = rng.choice(chunk_texts)  # noqa: S311
        for _ in range(3):
            if wrong != r.context:
                break
            wrong = rng.choice(chunk_texts)  # noqa: S311
        adversarial.append(
            Record(
                question=r.question,
                answer=rng.choice(ADVERSARIAL_ANSWERS),  # noqa: S311
                context=wrong,
                is_adversarial=True,
                source="adversarial",
            )
        )

    combined = records + adversarial
    rng.shuffle(combined)
    logger.info("added {n} adversarial ({p:.0f}%); total={t}",
                n=n_adv, p=100 * n_adv / len(combined), t=len(combined))
    return combined


def _load_mix(cfg: SynthConfig, synth_count: int) -> list[Record]:
    """Optionally blend in prepared StackOverflow rows for a hybrid dataset."""
    if not cfg.mix_jsonl or cfg.mix_fraction <= 0:
        return []
    path = Path(cfg.mix_jsonl)
    if not path.exists():
        logger.warning("mix_jsonl not found, skipping: {p}", p=path)
        return []

    rows: list[Record] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            rows.append(
                Record(
                    question=d.get("question", ""),
                    answer=d.get("answer", ""),
                    context=d.get("context", ""),
                    is_adversarial=bool(d.get("is_adversarial", False)),
                    source=d.get("source", "stackoverflow"),
                )
            )

    n_take = int(synth_count * cfg.mix_fraction)
    rng = random.Random(cfg.seed)
    if n_take < len(rows):
        rows = rng.sample(rows, n_take)
    logger.info("mixing in {n} StackOverflow rows from {p}", n=len(rows), p=path)
    return rows


def _split_and_write(records: list[Record], cfg: SynthConfig) -> dict[str, int]:
    rng = random.Random(cfg.seed)
    rng.shuffle(records)
    n_val = max(1, int(len(records) * cfg.val_fraction))
    val, train = records[:n_val], records[n_val:]

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        path = out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                json.dump(r.to_json(), f, ensure_ascii=False)
                f.write("\n")
        logger.info("wrote {n} rows → {p}", n=len(rows), p=path)

    return {"train": len(train), "val": len(val)}


def run_synth(cfg: SynthConfig) -> dict[str, int]:
    """Run the full synthetic-generation pipeline. Returns summary counts."""
    chunks = load_chunks(cfg)
    if not chunks:
        raise RuntimeError("no chunks after filtering — check chroma_path / filters")

    teacher = Teacher(cfg)
    records = _generate_all(chunks, teacher, cfg.max_workers)
    if not records:
        raise RuntimeError("teacher produced 0 pairs — check teacher endpoint / model")

    records = _dedup(records)
    synth_count = len(records)
    records = _add_adversarial(records, chunks, cfg)
    records.extend(_load_mix(cfg, synth_count))

    summary = _split_and_write(records, cfg)
    summary.update({
        "chunks_used": len(chunks),
        "synth_pairs": synth_count,
        "total_rows": summary["train"] + summary["val"],
    })
    logger.info("synth pipeline done: {s}", s=summary)
    return summary
