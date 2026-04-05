from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from typing import Any

import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from rag.metrics import (
    embedding_similarity,
    squad_f1,
    squad_precision,
    squad_recall,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ground-truth evaluation (lexical + semantic)
# ---------------------------------------------------------------------------


def evaluate_with_ground_truth(
    eval_df: pd.DataFrame,
    rag_fn: Callable[[str], str],
    pure_fn: Callable[[str], str],
    eval_embed_model: SentenceTransformer,
    max_samples: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if max_samples is not None and max_samples < len(eval_df):
        data = eval_df.sample(max_samples, random_state=42).reset_index(drop=True)
    else:
        data = eval_df.reset_index(drop=True)

    records: list[dict[str, Any]] = []

    for row in tqdm(data.itertuples(index=False), total=len(data)):
        q = row.question
        gold = row.answer

        rag_answer = rag_fn(q)
        pure_answer = pure_fn(q)

        records.append(
            {
                "question": q,
                "gold": gold,
                "rag_answer": rag_answer,
                "pure_answer": pure_answer,
                "rag_precision": squad_precision(rag_answer, gold),
                "rag_recall": squad_recall(rag_answer, gold),
                "rag_f1": squad_f1(rag_answer, gold),
                "pure_precision": squad_precision(pure_answer, gold),
                "pure_recall": squad_recall(pure_answer, gold),
                "pure_f1": squad_f1(pure_answer, gold),
                "answer_similarity": embedding_similarity(
                    rag_answer, pure_answer, eval_embed_model
                ),
            }
        )

    results_df = pd.DataFrame(records)

    summary = {
        "rag_mean_f1": results_df["rag_f1"].mean(),
        "pure_mean_f1": results_df["pure_f1"].mean(),
        "answer_similarity": results_df["answer_similarity"].mean(),
        "rag_better_count": int((results_df["rag_f1"] > results_df["pure_f1"]).sum()),
        "pure_better_count": int((results_df["pure_f1"] > results_df["rag_f1"]).sum()),
    }

    return results_df, summary


# ---------------------------------------------------------------------------
# RAGAS evaluation (runs in isolated venv)
# ---------------------------------------------------------------------------

_RAGAS_SCRIPT = """\
import os, json, sys, logging

logging.basicConfig(level=logging.DEBUG, stream=sys.stderr)

import pandas as pd
from datasets import Dataset
from ragas import evaluate, RunConfig
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

input_file = sys.argv[1]
output_file = sys.argv[2]
api_url = sys.argv[3]
api_key = sys.argv[4]
judge_model_name = sys.argv[5]
embed_model_name = sys.argv[6]

with open(input_file, "r") as f:
    data = json.load(f)

dataset = Dataset.from_list(data)
llm = ChatOpenAI(
    model=judge_model_name,
    base_url=api_url,
    api_key=api_key,
    temperature=0,
    timeout=120,
)

print(">>> Smoke test LLM...", flush=True)
test_resp = llm.invoke("Reply with exactly: OK")
print(f">>> LLM responded: {test_resp.content!r}", flush=True)

embeddings = HuggingFaceEmbeddings(model_name=embed_model_name)

run_config = RunConfig(
    timeout=180,
    max_retries=5,
    max_wait=90,
    max_workers=4,
)

print(">>> Ragas: Start Evaluation...", flush=True)
results = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_recall],
    llm=llm,
    embeddings=embeddings,
    run_config=run_config,
)

results_df = results.to_pandas()
print(">>> NaN counts per column:", flush=True)
print(
    results_df[["faithfulness", "answer_relevancy", "context_recall"]]
    .isna().sum().to_string(),
    flush=True,
)

with open(output_file, "w") as f:
    json.dump(results_df.to_dict(orient="records"), f)
print(">>> Ragas: Done.", flush=True)
"""

_RAGAS_VENV_DEPS = [
    "pandas",
    "wrapt",
    "ragas",
    "langchain-openai",
    "langchain-huggingface",
    "sentence-transformers",
    "datasets",
]


def _setup_ragas_venv(venv_path: str) -> str:
    python_exec = f"{venv_path}/bin/python"
    clean_env = os.environ.copy()
    clean_env.pop("PYTHONPATH", None)
    clean_env.pop("PYTHONHOME", None)

    if os.path.exists(venv_path):
        shutil.rmtree(venv_path)

    logger.info("Creating isolated ragas venv at %s", venv_path)
    subprocess.run(  # noqa: S603
        [sys.executable, "-m", "pip", "install", "virtualenv"],
        check=True,
    )
    subprocess.run(  # noqa: S603
        [sys.executable, "-m", "virtualenv", venv_path],
        check=True,
    )
    subprocess.run(  # noqa: S603
        [
            python_exec,
            "-m",
            "pip",
            "install",
            "-q",
            "--no-cache-dir",
            *_RAGAS_VENV_DEPS,
        ],
        check=True,
        env=clean_env,
    )
    return python_exec


def run_ragas_evaluation(
    df_prepared: pd.DataFrame,
    api_url: str,
    api_key: str,
    judge_model: str,
    embed_model: str = "BAAI/bge-base-en-v1.5",
    venv_path: str = "./ragas_venv",
) -> pd.DataFrame:
    script_path = os.path.join(venv_path, "run_eval.py")
    input_path = os.path.join(venv_path, "temp_input.json")
    output_path = os.path.join(venv_path, "temp_output.json")

    try:
        python_exec = _setup_ragas_venv(venv_path)
    except Exception:
        logger.exception("Failed to setup ragas venv")
        return pd.DataFrame()

    with open(script_path, "w") as f:
        f.write(_RAGAS_SCRIPT)

    records = df_prepared.to_dict(orient="records")
    with open(input_path, "w") as f:
        json.dump(records, f)

    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)

    logger.info("Running ragas subprocess...")
    try:
        result = subprocess.run(  # noqa: S603
            [
                python_exec,
                script_path,
                input_path,
                output_path,
                api_url,
                api_key,
                judge_model,
                embed_model,
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        logger.info(result.stdout)
        if result.stderr:
            for line in result.stderr.splitlines():
                if any(
                    k in line
                    for k in [
                        "ERROR",
                        "WARNING",
                        "NaN",
                        "parse",
                        "failed",
                        "exception",
                        "Traceback",
                    ]
                ):
                    logger.warning("STDERR: %s", line)

        if os.path.exists(output_path):
            with open(output_path) as f:
                return pd.DataFrame(json.load(f))

        logger.error("Ragas output file not created")
        logger.error("Full STDERR: %s", result.stderr[-3000:])
        return pd.DataFrame()

    except Exception:
        logger.exception("Ragas subprocess failed")
        return pd.DataFrame()
