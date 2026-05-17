"""Head-to-head smoke test: base model vs LoRA adapter through vLLM.

Sends the same questions to both `--base-model` and `--lora-model` over
the vLLM OpenAI-compatible endpoint, prints answers side by side. Used
to quickly confirm a freshly-deployed LoRA adapter actually loads and
behaves differently from the base model.

Three test cases, designed to surface different aspects of the LoRA:
  1. RELEVANT context     — both should answer; LoRA should look more
                             concise / formatted per training prompt.
  2. IRRELEVANT context   — LoRA should *refuse* (the adversarial
                             examples from dataset-prep trained it for
                             this); base will likely hallucinate.
  3. NO context           — both fall back on parametric knowledge;
                             LoRA may still echo trained style.

Run on the vast.ai server:
    python scripts/smoke_test.py

Run from the Mac against a remote endpoint:
    python scripts/smoke_test.py --base-url http://193.222.57.16:44090/v1
"""

from __future__ import annotations

import argparse
import sys

from openai import OpenAI

DEFAULT_BASE_URL = "http://localhost:18000/v1"
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_LORA_MODEL = "pytorch-rag"

SYSTEM_PROMPT = (
    "You are an expert PyTorch assistant. Answer the user's question using ONLY "
    "the information provided in the Context. If the Context does not contain "
    "enough information to answer the question reliably, say so explicitly "
    "instead of guessing. When showing code, use fenced code blocks with the "
    "`python` language tag."
)

TESTS: list[dict] = [
    {
        "label": "1. RELEVANT context — both should answer correctly",
        "context": (
            "torch.cuda.is_available()\n"
            "Return a bool indicating if CUDA is currently available.\n"
            "Returns: bool. True if CUDA is available, False otherwise.\n\n"
            "Example:\n"
            "    if torch.cuda.is_available():\n"
            "        device = torch.device('cuda')\n"
            "    else:\n"
            "        device = torch.device('cpu')\n"
        ),
        "question": "How can I check if CUDA is available before moving my tensor to GPU?",
    },
    {
        "label": "2. IRRELEVANT context — LoRA should refuse, base will hallucinate",
        "context": (
            "torch.nn.functional.gelu(input, approximate='none')\n"
            "Applies the Gaussian Error Linear Units function.\n"
            "GELU(x) = x * Phi(x) where Phi(x) is the standard Gaussian CDF.\n"
            "When the approximate argument is 'tanh', Gelu is estimated with the\n"
            "tanh approximation.\n"
        ),
        "question": "How do I save and load a PyTorch model checkpoint to disk?",
    },
    {
        "label": "3. NO context — both fall back on parametric knowledge",
        "context": "(no relevant context provided)",
        "question": "What is the difference between torch.no_grad() and torch.inference_mode()?",
    },
]


def call_model(client: OpenAI, model: str, system: str, user: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        max_tokens=400,
    )
    return response.choices[0].message.content or "(empty)"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--lora-model", default=DEFAULT_LORA_MODEL)
    p.add_argument("--api-key", default="EMPTY", help="vLLM ignores; placeholder.")
    args = p.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    print(f"Endpoint: {args.base_url}")
    print(f"Base model: {args.base_model}")
    print(f"LoRA model: {args.lora_model}")

    # Confirm both models are actually served before sending any prompts
    try:
        served = {m.id for m in client.models.list().data}
    except Exception as exc:
        print(f"FAILED to query /v1/models: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"Models served: {sorted(served)}\n")
    for required in (args.base_model, args.lora_model):
        if required not in served:
            print(f"FAILED: model {required!r} not served by vLLM", file=sys.stderr)
            sys.exit(1)

    for test in TESTS:
        user_msg = f"Context:\n{test['context']}\n\nQuestion: {test['question']}"
        print("=" * 90)
        print(f"TEST {test['label']}")
        print("=" * 90)
        print(f"Q: {test['question']}\n")

        for label, model in (("BASE", args.base_model), ("LORA", args.lora_model)):
            print(f"--- {label}: {model} ---")
            try:
                print(call_model(client, model, SYSTEM_PROMPT, user_msg))
            except Exception as exc:
                print(f"FAILED: {exc}", file=sys.stderr)
            print()


if __name__ == "__main__":
    main()
