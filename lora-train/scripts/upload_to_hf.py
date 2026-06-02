"""Upload a trained LoRA adapter to the HuggingFace Hub.

Creates (or reuses) a model repo, writes a proper model card with the
base-model tag so HF shows the adapter↔base relationship, and uploads
the adapter folder via `upload_folder` (handles the 154 MB safetensors
through the Hub's LFS automatically).

Auth: run `huggingface-cli login` once (or set HF_TOKEN env var) before
running this.

Usage:
    cd lora-train
    uv run python scripts/upload_to_hf.py \\
        --adapter-dir runs/qwen25-coder-7b-lora-r16-v1/final \\
        --repo-id <your-username>/pytorch-rag-lora-r16

    # private repo:
    uv run python scripts/upload_to_hf.py ... --private
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder

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
retrieval-augmented QA system over PyTorch documentation + StackOverflow.

Part of an HSE university project on RAG over a private knowledge base.

## Training summary

- **Base model:** `{base_model}`
- **Method:** LoRA (PEFT), bf16, RAG-aware SFT
- **Rank / alpha / dropout:** {r} / {alpha} / {dropout}
- **Target modules:** {targets}
- **Trainable params:** {trainable}
- **Dataset:** ~1.8k StackOverflow PyTorch Q&A pairs, each enriched with
  top-k retrieved documentation chunks as context, plus ~15% adversarial
  "cannot answer from context" examples.

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained(
    "{base_model}", dtype="bfloat16", device_map="auto"
)
model = PeftModel.from_pretrained(base, "{repo_id}")
tok = AutoTokenizer.from_pretrained("{repo_id}")

messages = [
    {{"role": "system", "content": "You are an expert PyTorch assistant. "
      "Answer using ONLY the provided Context."}},
    {{"role": "user", "content": "Context:\\n<retrieved chunks>\\n\\nQuestion: <q>"}},
]
inputs = tok.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(model.device)
out = model.generate(inputs, max_new_tokens=400)
print(tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True))
```

### Serving with vLLM

```bash
vllm serve {base_model} \\
    --enable-lora \\
    --lora-modules pytorch-rag={repo_id} \\
    --max-lora-rank {r}
```

## Note on results

This is a research artifact. In our evaluation the v1 adapter did **not**
outperform the base model on the RAG task (composite RAG score dropped vs
the `base-vanilla` baseline) — most likely due to a stylistic shift toward
terse StackOverflow answers and limited training data. See the project
report for the full analysis. Use as a baseline / starting point.
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

    trainable = "~40M (0.53% of 7.66B)"  # from training log; informational

    return MODEL_CARD_TEMPLATE.format(
        base_model=BASE_MODEL,
        base_url=f"https://huggingface.co/{BASE_MODEL}",
        repo_id=repo_id,
        r=cfg.get("r", "?"),
        alpha=cfg.get("lora_alpha", "?"),
        dropout=cfg.get("lora_dropout", "?"),
        targets=targets_str,
        trainable=trainable,
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
