"""Обучение TopicClassifier на вопросах StackOverflow по PyTorch.

Предполагаемая форма запуска из корня workspace:
    uv run --locked --package outlier-detection python \\
        outlier-detection/scripts/train.py \\
        --csv data/stackoverflow-pytorch.csv \\
        --output /tmp/pytorch_topic_classifier.joblib

В clean environment команда сейчас не является рабочим baseline: package manifest
не перечисляет pandas, а скрипт использует stale top-level import
``from topic_classifier`` вместо установленного package path. Сначала исправьте
import/dependencies либо воспроизведите окружение осознанно. Default output указывает
на tracked model и перезаписывается; всегда задавайте новый temporary/ignored
``--output``. Joblib-файл не хранит полный provenance корпуса, split и версий.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

MODULE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MODULE_ROOT))

from topic_classifier import TopicClassifier

PROJECT_ROOT = MODULE_ROOT.parent
DEFAULT_CSV = PROJECT_ROOT / "data" / "stackoverflow-pytorch.csv"
DEFAULT_OUTPUT = MODULE_ROOT / "models" / "pytorch_topic_classifier.joblib"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PyTorch topic classifier")
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Path to stackoverflow-pytorch.csv (default: data/stackoverflow-pytorch.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output .joblib path (default: models/pytorch_topic_classifier.joblib)",
    )
    parser.add_argument("--nu", type=float, default=0.05, help="OneClassSVM nu parameter")
    parser.add_argument("--max-features", type=int, default=5000, help="TF-IDF max features")
    parser.add_argument("--kernel", type=str, default="rbf", help="SVM kernel")
    parser.add_argument("--gamma", type=str, default="scale", help="SVM gamma")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"ERROR: CSV not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("Training TopicClassifier")
    print("=" * 60)
    print(f"CSV:          {args.csv}")
    print(f"Output:       {args.output}")
    print(f"nu:           {args.nu}")
    print(f"max_features: {args.max_features}")
    print(f"kernel:       {args.kernel}")
    print(f"gamma:        {args.gamma}")
    print("-" * 60)

    print("Loading CSV...")
    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows")

    if "question_body" not in df.columns:
        print(f"ERROR: 'question_body' column not found. Columns: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    texts = df["question_body"].dropna().astype(str).tolist()
    print(f"Non-empty questions: {len(texts)}")

    train_texts, test_texts = train_test_split(
        texts,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print(f"Train: {len(train_texts)} samples")
    print(f"Test:  {len(test_texts)} samples")
    print("-" * 60)

    print("Fitting model...")
    clf = TopicClassifier(
        max_features=args.max_features,
        nu=args.nu,
        kernel=args.kernel,
        gamma=args.gamma,
    )
    clf.fit(train_texts)

    print("-" * 60)
    train_result = clf.predict(train_texts)
    train_inliers = (train_result.labels == 1).sum()
    train_outliers = (train_result.labels == -1).sum()
    print(
        f"Train: inliers {train_inliers} ({train_inliers / len(train_result) * 100:.1f}%), "
        f"outliers {train_outliers} ({train_outliers / len(train_result) * 100:.1f}%)"
    )

    test_result = clf.predict(test_texts)
    test_inliers = (test_result.labels == 1).sum()
    test_outliers = (test_result.labels == -1).sum()
    print(
        f"Test:  inliers {test_inliers} ({test_inliers / len(test_result) * 100:.1f}%), "
        f"outliers {test_outliers} ({test_outliers / len(test_result) * 100:.1f}%)"
    )

    print("-" * 60)
    print("Sanity check:")
    samples = [
        "How to create a tensor and move it to GPU in PyTorch",
        "Training a neural network with torch.nn.Module",
        "Best recipe for chocolate cake with strawberries",
        "How to change a car tire",
    ]
    print(clf.predict(samples))
    print("-" * 60)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    saved_path = clf.save(args.output)
    size_mb = saved_path.stat().st_size / (1024 * 1024)
    print(f"Saved: {saved_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
