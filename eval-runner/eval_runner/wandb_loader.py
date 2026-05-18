"""Pull eval-runner results from wandb for cross-run comparison.

Wraps `wandb.Api` to flatten finished runs into a pandas DataFrame
where each row is one run, `cfg.*` columns hold original RunConfig
fields, and the rest are logged scalar metrics
(`lexical/`, `ragas/`, `latency/`, `score/`, …).

Typical use from a notebook:

    from eval_runner.wandb_loader import fetch_runs_as_df, has_tag

    df = fetch_runs_as_df("pytorch-rag-eval", entity="ooovotetoda")
    phase1 = df[df["tags"].apply(lambda ts: has_tag(ts, "phase:1"))]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import wandb


@dataclass
class RunRecord:
    """Flat snapshot of one wandb run."""

    name: str
    id: str
    url: str
    state: str  # "finished" / "running" / "crashed" / …
    tags: list[str]
    config: dict[str, Any]
    summary: dict[str, Any]

    def to_row(self) -> dict[str, Any]:
        """Flatten into a single dict suitable for `pd.DataFrame([...])`.

        Config keys are prefixed with `cfg.` to avoid colliding with metric
        names. Non-scalar summary values (lists, dicts, wandb artifact refs)
        are dropped — DataFrame columns should be plot-friendly.
        """
        row: dict[str, Any] = {
            "run_name": self.name,
            "run_id": self.id,
            "run_url": self.url,
            "state": self.state,
            "tags": self.tags,
        }
        for k, v in self.config.items():
            row[f"cfg.{k}"] = v
        for k, v in self.summary.items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                row[k] = v
        return row


def fetch_runs(
    project: str,
    entity: str | None = None,
    tags: list[str] | None = None,
    exclude_name_prefix: str | None = None,
    only_finished: bool = True,
) -> list[RunRecord]:
    """Fetch runs from a wandb project.

    Args:
        project: wandb project name (e.g. "pytorch-rag-eval").
        entity: wandb entity / team. If None, uses the default logged-in entity.
        tags: optional list of tags — keep runs that have ANY of these.
        exclude_name_prefix: drop runs whose name starts with this string
            (default usage: exclude `_sanity-check` debug runs).
        only_finished: skip runs that aren't in state="finished".
    """
    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    filters: dict[str, Any] = {}
    if tags:
        filters["tags"] = {"$in": tags}

    runs = api.runs(path, filters=filters or None)
    records: list[RunRecord] = []
    for r in runs:
        if only_finished and getattr(r, "state", None) != "finished":
            continue
        if exclude_name_prefix and r.name.startswith(exclude_name_prefix):
            continue
        records.append(
            RunRecord(
                name=r.name,
                id=r.id,
                url=r.url,
                state=str(getattr(r, "state", "unknown")),
                tags=list(r.tags or []),
                config=dict(r.config or {}),
                summary=dict(r.summary or {}),
            )
        )
    return records


def fetch_runs_as_df(
    project: str,
    entity: str | None = None,
    tags: list[str] | None = None,
    exclude_name_prefix: str | None = "_",
    only_finished: bool = True,
) -> pd.DataFrame:
    """Convenience wrapper: fetch_runs(...) -> DataFrame.

    Default `exclude_name_prefix='_'` drops debug runs whose names start
    with underscore (the sanity check pattern).
    """
    records = fetch_runs(
        project=project,
        entity=entity,
        tags=tags,
        exclude_name_prefix=exclude_name_prefix,
        only_finished=only_finished,
    )
    df = pd.DataFrame([r.to_row() for r in records])
    if not df.empty:
        df = df.sort_values("run_name").reset_index(drop=True)
    return df


def has_tag(tags: list[str] | None, tag: str) -> bool:
    """Predicate helper for DataFrame filtering on the `tags` list column."""
    return tag in (tags or [])
