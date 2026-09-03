"""Recompute the composite RAG score from logged RAGAS components.

The composite `score/rag_score` is a weighted mix of faithfulness,
answer_relevancy and context_recall. Because the weights are a value
judgement (how much you care about not-hallucinating vs answering), the
honest way to report results is a weight-sensitivity analysis rather
than a single cherry-picked number.

This reads the per-run `eval done. summary: {...}` line from each log,
re-derives rag_score under several weight schemes WITHOUT re-running the
(expensive) eval, and prints base-vs-v2 pairwise winners.

Usage:
    uv run --locked --package eval-runner python \\
        eval-runner/scripts/recompute_score.py
    uv run --locked --package eval-runner python \\
        eval-runner/scripts/recompute_score.py --weights 0.6 0.2 0.2
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Default base-vs-v2 pairs for the v2 experiment matrix.
PAIRS = [
    ("vanilla", "base-vanilla-k5", "v2-vanilla-k5"),
    ("rerank", "base-rerank-k5", "v2-rerank-k5"),
    ("qt", "base-qt-k5", "v2-qt-k5"),
]

# Weight schemes for the sensitivity table: (faithfulness, answer_relevancy, context_recall)
SCHEMES = {
    "0.4/0.4/0.2 (baseline notebook)": (0.4, 0.4, 0.2),
    "0.5/0.3/0.2 (moderate)": (0.5, 0.3, 0.2),
    "0.6/0.2/0.2 (faith-priority)": (0.6, 0.2, 0.2),
    "0.5/0.5/0.0 (no ctx)": (0.5, 0.5, 0.0),
    "0.7/0.3/0.0 (faith+rel)": (0.7, 0.3, 0.0),
}

SUMMARY_RE = re.compile(r"eval done\. summary: (\{.*\})")


def load_components(logs_dir: Path) -> dict[str, tuple[float, float, float]]:
    """Map run name -> (faithfulness, answer_relevancy, context_recall)."""
    out: dict[str, tuple[float, float, float]] = {}
    for log in sorted(logs_dir.glob("*.log")):
        txt = log.read_text()
        matches = SUMMARY_RE.findall(txt)
        if not matches:
            continue
        d = json.loads(matches[-1].replace("'", '"'))
        if "ragas/faithfulness" not in d:
            continue
        out[log.stem] = (
            d.get("ragas/faithfulness", float("nan")),
            d.get("ragas/answer_relevancy", float("nan")),
            d.get("ragas/context_recall", float("nan")),
        )
    return out


def score(components: tuple[float, float, float], weights: tuple[float, float, float]) -> float:
    return sum(c * w for c, w in zip(components, weights, strict=True))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--logs-dir", type=Path, default=Path("eval-runner/logs"))
    p.add_argument(
        "--weights",
        type=float,
        nargs=3,
        default=None,
        metavar=("FAITH", "REL", "CTX"),
        help="If given, print a single scheme with these weights instead of the full table.",
    )
    args = p.parse_args()

    comp = load_components(args.logs_dir)
    if not comp:
        print(f"no run summaries found in {args.logs_dir}")
        return

    schemes = {f"{args.weights[0]}/{args.weights[1]}/{args.weights[2]} (custom)": tuple(args.weights)} \
        if args.weights else SCHEMES

    for sname, w in schemes.items():
        print("=" * 64)
        print(f"SCHEME {sname}")
        wins_v2 = 0
        n_pairs = 0
        for label, b, v in PAIRS:
            if b not in comp or v not in comp:
                continue
            n_pairs += 1
            sb, sv = score(comp[b], w), score(comp[v], w)
            winner = "v2" if sv > sb else "base"
            wins_v2 += sv > sb
            print(f"  {label:8} base={sb:.3f}  v2={sv:.3f}  -> {winner}")
        if n_pairs:
            print(f"  v2 wins {wins_v2}/{n_pairs} pairs")


if __name__ == "__main__":
    main()
