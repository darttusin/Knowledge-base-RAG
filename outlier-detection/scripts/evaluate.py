"""Оценка One-Class SVM детектора off-topic вопросов.

Позитивный класс для метрик детекции — «outlier» (вопрос не по теме).

Eval-набор:
  - in-topic: часть PyTorch-вопросов из CSV, восстановленная по default
    ``test_size=0.2`` и ``random_state=42`` scripts/train.py; это не доказывает,
    что конкретная загруженная модель их не видела, если она обучалась иначе
  - off-topic: встроенный размеченный набор из 90 вопросов в трёх корзинах
    сложности (easy: бытовые темы; medium: программирование без ML;
    hard: соседние ML-фреймворки — TensorFlow/Keras/JAX/sklearn/XGBoost)

Считает: precision/recall/F1 (outlier), PR-AUC, ROC-AUC, FPR,
inlier acceptance, detection rate по корзинам. Рисует PR-кривую,
ROC-кривую, confusion matrix и бар-чарт по корзинам; сохраняет PNG
локально и, если не указан ``--no-wandb``, отправляет metrics, figures и примеры
ошибок в W&B.

Локальный пример из корня workspace:
    uv run --locked --package outlier-detection python \\
        outlier-detection/scripts/evaluate.py \\
        --csv data/stackoverflow-pytorch.csv \\
        --model outlier-detection/models/pytorch_topic_classifier.joblib \\
        --out /tmp/outlier-eval \\
        --no-wandb

Текущий package manifest не перечисляет pandas, matplotlib и W&B, используемые
этим скриптом. Default ``--out`` указывает на tracked results и перезаписывается.
До исправления manifest команда зависит от уже наполненной общей ``.venv``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

MODULE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MODULE_ROOT))
sys.path.insert(0, str(MODULE_ROOT / "outlier_detection"))

try:
    from outlier_detection.topic_classifier import TopicClassifier
except ImportError:
    from topic_classifier import TopicClassifier

PROJECT_ROOT = MODULE_ROOT.parent
DEFAULT_CSV = PROJECT_ROOT / "data" / "stackoverflow-pytorch.csv"
DEFAULT_MODEL = MODULE_ROOT / "models" / "pytorch_topic_classifier.joblib"
DEFAULT_OUT = MODULE_ROOT / "eval_results"

EASY_NEGATIVES = [
    "What is the best recipe for borscht?",
    "How do I improve my chess opening repertoire?",
    "What are the top tourist attractions in Rome?",
    "How long should I marinate chicken for grilling?",
    "What is the capital of Australia?",
    "How do I train for a marathon in 12 weeks?",
    "What movies won the Oscar for best picture in 2020?",
    "How can I improve my credit score quickly?",
    "What plants grow well in low light apartments?",
    "How do I change a flat tire on a highway?",
    "What is the difference between espresso and ristretto?",
    "How should I store fresh basil to keep it longer?",
    "What are good exercises for lower back pain?",
    "When was the Eiffel Tower built and why?",
    "How do I negotiate a higher salary at a new job?",
    "What breed of dog is best for apartment living?",
    "How do I remove red wine stains from a carpet?",
    "What is the plot of Crime and Punishment?",
    "Which guitar strings are best for beginners?",
    "How much water should I drink per day?",
    "What are the rules of cricket for newcomers?",
    "How do I apply for a Schengen visa?",
    "What is a good skincare routine for dry skin?",
    "How do I make sourdough starter from scratch?",
    "What are the symptoms of vitamin D deficiency?",
    "How do mortgage interest rates work?",
    "What is the history of the Silk Road?",
    "How do I teach my cat to use a scratching post?",
    "What should I pack for a two-week hiking trip?",
    "How do tides work and why are there two per day?",
]

MEDIUM_NEGATIVES = [
    "How do I write a LEFT JOIN across three tables in SQL?",
    "What is the difference between let and const in JavaScript?",
    "How do I center a div horizontally and vertically in CSS?",
    "How do I revert the last commit in git without losing changes?",
    "What is the difference between a Docker image and a container?",
    "How do I parse JSON in Java with Jackson?",
    "How do I set up nginx as a reverse proxy?",
    "What is a foreign key constraint in PostgreSQL?",
    "How do async and await work in C#?",
    "How do I make an HTTP POST request with curl?",
    "What is the event loop in Node.js?",
    "How do I write unit tests with JUnit 5?",
    "What is the difference between TCP and UDP?",
    "How do I schedule a cron job to run every Monday?",
    "How do I handle CORS errors in a React app?",
    "What are Kubernetes pods and deployments?",
    "How do I create a virtual host in Apache?",
    "What is the difference between REST and GraphQL?",
    "How do I use flexbox to build a responsive navbar?",
    "How do I read environment variables in a bash script?",
    "What is dependency injection in Spring Boot?",
    "How do I optimize a slow MySQL query with indexes?",
    "What is the difference between stack and heap memory?",
    "How do I implement OAuth2 login in a web app?",
    "How do I merge two branches and resolve conflicts in git?",
    "What does the volatile keyword do in C++?",
    "How do I rate-limit requests in an Express server?",
    "What is the difference between threads and processes?",
    "How do I deploy a static site to AWS S3 and CloudFront?",
    "How do I validate an email address with a regular expression?",
]

HARD_NEGATIVES = [
    "How do I create a constant tensor in TensorFlow 2?",
    "What is the difference between Sequential and Functional API in Keras?",
    "How do I use tf.data.Dataset for batching and shuffling?",
    "How do I save and restore a model checkpoint in TensorFlow?",
    "What does model.compile do in Keras and which optimizers are available?",
    "How do I freeze layers during transfer learning in Keras?",
    "How do I use jax.jit to speed up a numerical function?",
    "What is the difference between jax.grad and jax.vjp?",
    "How do I vectorize a function with jax.vmap?",
    "How do I tune hyperparameters with GridSearchCV in scikit-learn?",
    "What is the difference between fit and fit_transform in sklearn?",
    "How do I handle class imbalance with sklearn class_weight?",
    "How do I plot a confusion matrix from a sklearn classifier?",
    "What is early stopping in XGBoost and how do I configure it?",
    "How do I interpret feature importance in a random forest?",
    "How do I use cross_val_score with a custom scoring function?",
    "What is the difference between bagging and boosting?",
    "How do I export a Keras model to TensorFlow Lite?",
    "How do I use tf.GradientTape for custom training loops?",
    "What is the difference between LightGBM and XGBoost?",
    "How do I standardize features with StandardScaler before SVM?",
    "How do I implement k-means clustering in scikit-learn?",
    "What is PCA and how do I choose the number of components?",
    "How do I do a stratified train test split in sklearn?",
    "How does dropout work in Keras layers?",
    "How do I run distributed training with TensorFlow MirroredStrategy?",
    "What is the Keras EarlyStopping callback patience parameter?",
    "How do I convert a pandas DataFrame to a tf.data pipeline?",
    "How do I use Optuna to tune LightGBM hyperparameters?",
    "What is the difference between SAME and VALID padding in TensorFlow conv layers?",
]

BUCKETS = {
    "easy (бытовые)": EASY_NEGATIVES,
    "medium (прогр., не ML)": MEDIUM_NEGATIVES,
    "hard (другие ML-фреймворки)": HARD_NEGATIVES,
}


def load_held_out_inliers(csv_path: Path, n_samples: int, seed: int) -> list[str]:
    """Тот же сплит, что в scripts/train.py → реально отложенные вопросы."""
    df = pd.read_csv(csv_path)
    texts = df["question_body"].dropna().astype(str).tolist()
    _, test_texts = train_test_split(texts, test_size=0.2, random_state=42)
    rng = np.random.default_rng(seed)
    if n_samples < len(test_texts):
        idx = rng.choice(len(test_texts), size=n_samples, replace=False)
        return [test_texts[i] for i in idx]
    return test_texts


def fig_pr_curve(y_true: np.ndarray, y_score: np.ndarray, pr_auc: float):
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, lw=2)
    ax.fill_between(rec, prec, alpha=0.15)
    ax.set_xlabel("Recall (доля пойманных off-topic)")
    ax.set_ylabel("Precision")
    ax.set_title(f"PR-кривая · детекция off-topic · PR-AUC = {pr_auc:.3f}")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_roc_curve(y_true: np.ndarray, y_score: np.ndarray, roc_auc: float):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2)
    ax.plot([0, 1], [0, 1], ls="--", c="gray", lw=1)
    ax.set_xlabel("FPR (по-теме вопросы, ошибочно отклонённые)")
    ax.set_ylabel("TPR (пойманные off-topic)")
    ax.set_title(f"ROC-кривая · ROC-AUC = {roc_auc:.3f}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_confusion(cm: np.ndarray):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v}", ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black", fontsize=14)
    ax.set_xticks([0, 1], ["in-topic", "off-topic"])
    ax.set_yticks([0, 1], ["in-topic", "off-topic"])
    ax.set_xlabel("Предсказание")
    ax.set_ylabel("Истина")
    ax.set_title("Confusion matrix")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    return fig


def fig_buckets(rates: dict[str, float], acceptance: float):
    labels = ["in-topic\n(acceptance)"] + list(rates.keys())
    values = [acceptance] + list(rates.values())
    colors = ["#1D9E75"] + ["#D85A30"] * len(rates)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar(range(len(values)), values, color=colors)
    ax.set_xticks(range(len(labels)), labels, fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Доля")
    ax.set_title("Acceptance по-теме и detection rate off-topic по корзинам")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate One-Class SVM off-topic detector")
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--n-inliers", type=int, default=400)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb-project", type=str, default="pytorch-rag-eval")
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()

    print(f"Загрузка модели: {args.model}")
    clf = TopicClassifier.load(args.model)

    inliers = load_held_out_inliers(args.csv, args.n_inliers, args.seed)
    negatives = [q for bucket in BUCKETS.values() for q in bucket]
    print(f"Eval-набор: {len(inliers)} in-topic (held-out) + {len(negatives)} off-topic")

    texts = inliers + negatives
    y_true = np.array([0] * len(inliers) + [1] * len(negatives))  # 1 = off-topic

    result = clf.predict(texts)
    y_pred = (result.labels == -1).astype(int)
    y_score = -result.scores  # выше = более off-topic

    metrics = {
        "precision_outlier": precision_score(y_true, y_pred),
        "recall_outlier": recall_score(y_true, y_pred),
        "f1_outlier": f1_score(y_true, y_pred),
        "pr_auc": average_precision_score(y_true, y_score),
        "roc_auc": roc_auc_score(y_true, y_score),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "inlier_acceptance": float((y_pred[: len(inliers)] == 0).mean()),
        "fpr": float((y_pred[: len(inliers)] == 1).mean()),
        "n_inliers": len(inliers),
        "n_outliers": len(negatives),
    }

    bucket_rates: dict[str, float] = {}
    offset = len(inliers)
    for name, bucket in BUCKETS.items():
        pred = y_pred[offset : offset + len(bucket)]
        bucket_rates[name] = float(pred.mean())
        offset += len(bucket)

    print("\n=== Метрики (positive = off-topic) ===")
    for k, v in metrics.items():
        print(f"  {k:22s} {v:.4f}" if isinstance(v, float) else f"  {k:22s} {v}")
    print("\n=== Detection rate по корзинам ===")
    for name, rate in bucket_rates.items():
        print(f"  {name:32s} {rate:.2f}")

    args.out.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    figures = {
        "pr_curve": fig_pr_curve(y_true, y_score, metrics["pr_auc"]),
        "roc_curve": fig_roc_curve(y_true, y_score, metrics["roc_auc"]),
        "confusion_matrix": fig_confusion(cm),
        "buckets": fig_buckets(bucket_rates, metrics["inlier_acceptance"]),
    }
    for name, fig in figures.items():
        path = args.out / f"{name}.png"
        fig.savefig(path, dpi=150)
        print(f"Сохранено: {path}")

    with (args.out / "metrics.json").open("w") as f:
        json.dump({**metrics, "bucket_detection": bucket_rates}, f, indent=2, ensure_ascii=False)

    misses = [t for t, yt, yp in zip(texts, y_true, y_pred) if yt == 1 and yp == 0]
    false_rejects = [t for t, yt, yp in zip(texts, y_true, y_pred) if yt == 0 and yp == 1]
    print(f"\nПропущенные off-topic ({len(misses)}):")
    for t in misses[:10]:
        print(f"  - {t[:90]}")

    if not args.no_wandb:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            name="outlier-svm-eval",
            tags=["outlier-detection", "one-class-svm"],
            config={
                "model": "TF-IDF + OneClassSVM",
                "nu": clf.nu,
                "kernel": clf.kernel,
                "max_features": clf.max_features,
                "n_inliers": len(inliers),
                "n_outliers": len(negatives),
            },
        )
        wandb.log({f"outlier/{k}": v for k, v in metrics.items()})
        wandb.log({f"outlier/detect_{i}": r for i, r in enumerate(bucket_rates.values())})
        for name, fig in figures.items():
            wandb.log({f"outlier/{name}": wandb.Image(fig)})
        if misses:
            wandb.log({"outlier/missed_offtopic": wandb.Table(
                columns=["question"], data=[[t] for t in misses])})
        if false_rejects:
            wandb.log({"outlier/false_rejects": wandb.Table(
                columns=["question"], data=[[t[:300]] for t in false_rejects])})
        print(f"\nwandb: {run.url}")
        run.finish()

    plt.close("all")


if __name__ == "__main__":
    main()
