"""Upload a trained LoRA adapter directory to the Hugging Face Hub.

This is an external publishing operation: it creates or reuses a remote model
repository and uploads the selected directory. Unless ``--no-model-card`` is
passed, it first overwrites ``README.md`` inside that local adapter directory.

The generated card assumes ``BASE_MODEL`` below and reads only a few LoRA fields
from ``adapter_config.json``. It cannot infer the dataset, model revision,
quantization, prompt contract actually used at serving time, or evaluation
results; review the card and adapter contents before publishing.

Use a cached Hugging Face login or ``HF_TOKEN``. Avoid ``--token`` when possible
because command-line arguments can be visible to other local processes.

Usage from the workspace root:
    uv run --locked --package lora-train python \\
        lora-train/scripts/upload_to_hf.py \\
        --adapter-dir /absolute/path/to/adapter/final \\
        --repo-id <your-username>/pytorch-rag-lora-r16

    # private repo:
    ... --private
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo

BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"

MODEL_CARD_TEMPLATE = """\
---
base_model: {base_model}
library_name: peft
license: apache-2.0
tags:
- lora
- peft
- rag
- pytorch-docs
- text-generation
---

# PyTorch-RAG LoRA adapter

LoRA adapter for [`{base_model}`]({base_url}), fine-tuned for a
retrieval-augmented QA experiment.

Part of an HSE university project on RAG over a private knowledge base.

## Adapter metadata

- **Assumed base model:** `{base_model}`
- **Method:** LoRA (PEFT)
- **Rank / alpha / dropout:** {r} / {alpha} / {dropout}
- **Target modules:** {targets}

This uploader does not reconstruct training provenance. Record the exact base
and dataset revisions, preprocessing, quantization, optimizer, seed, Git SHA and
prompt-contract fingerprint separately. If `prompt_contract.json` is present in
the adapter directory, serving must load and apply that same contract explicitly.

## Loading

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained(
    "{base_model}", dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(base, "{repo_id}")
tok = AutoTokenizer.from_pretrained("{repo_id}")
```

Loading succeeds independently of prompt compatibility or answer quality.

### Serving with vLLM

```bash
vllm serve {base_model} \\
    --enable-lora \\
    --lora-modules pytorch-rag={repo_id} \\
    --max-lora-rank {r}
```

## Evaluation status

This is a research artifact. The uploader does not run evaluation and this model
card makes no quality claim. Publish separately versioned results with corpus,
retrieval, generator, judge, metric weights, sample and seed provenance before
using the adapter beyond an experiment.
"""


def _read_adapter_config(adapter_dir: Path) -> dict:
    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"adapter_config.json not found in {adapter_dir}")
    with cfg_path.open() as f:
        return json.load(f)


def _build_model_card(adapter_dir: Path, repo_id: str) -> str:
    cfg = _read_adapter_config(adapter_dir)
    targets = cfg.get("target_modules", [])
    if isinstance(targets, list):
        targets_str = ", ".join(f"`{t}`" for t in sorted(targets))
    else:
        targets_str = str(targets)

    return MODEL_CARD_TEMPLATE.format(
        base_model=BASE_MODEL,
        base_url=f"https://huggingface.co/{BASE_MODEL}",
        repo_id=repo_id,
        r=cfg.get("r", "?"),
        alpha=cfg.get("lora_alpha", "?"),
        dropout=cfg.get("lora_dropout", "?"),
        targets=targets_str,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Upload a LoRA adapter to HuggingFace Hub")
    p.add_argument("--adapter-dir", type=Path, required=True, help="Path to the adapter folder (the 'final/' dir)")
    p.add_argument("--repo-id", type=str, required=True, help="HF repo id, e.g. username/pytorch-rag-lora")
    p.add_argument("--private", action="store_true", help="Create the repo as private")
    p.add_argument("--token", type=str, default=None, help="HF token (else uses cached login / HF_TOKEN)")
    p.add_argument(
        "--no-model-card",
        action="store_true",
        help="Don't overwrite README.md with a generated model card",
    )
    args = p.parse_args()

    if not args.adapter_dir.exists():
        print(f"ERROR: adapter dir not found: {args.adapter_dir}", file=sys.stderr)
        sys.exit(1)
    if not (args.adapter_dir / "adapter_model.safetensors").exists():
        print(
            f"ERROR: no adapter_model.safetensors in {args.adapter_dir} — is this the right folder?",
            file=sys.stderr,
        )
        sys.exit(1)

    api = HfApi(token=args.token)

    print(f"Creating repo {args.repo_id} (private={args.private})...")
    create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
        token=args.token,
    )

    if not args.no_model_card:
        card = _build_model_card(args.adapter_dir, args.repo_id)
        readme_path = args.adapter_dir / "README.md"
        readme_path.write_text(card, encoding="utf-8")
        print(f"Wrote model card → {readme_path}")

    print(f"Uploading {args.adapter_dir} → {args.repo_id} ...")
    api.upload_folder(
        folder_path=str(args.adapter_dir),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Upload PyTorch-RAG LoRA adapter (r=16)",
    )

    url = f"https://huggingface.co/{args.repo_id}"
    print(f"\nDone. Adapter available at: {url}")
    print(f"Pull it with: PeftModel.from_pretrained(base, '{args.repo_id}')")


if __name__ == "__main__":
    main()
