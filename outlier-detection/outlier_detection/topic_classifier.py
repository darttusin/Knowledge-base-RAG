"""
One-Class SVM классификатор тематики текстов.

Обучается на текстах одной темы и определяет,
относится ли новый текст к этой теме или нет.

Пример использования:
    from outlier_detection import TopicClassifier

    clf = TopicClassifier()
    clf.fit(texts)                      # список строк по теме
    clf.predict("How to use DataLoader in PyTorch?")
    clf.save("pytorch_model.joblib")

    clf2 = TopicClassifier.load("pytorch_model.joblib")
    clf2.predict(["some text", "another text"])
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import OneClassSVM


@dataclass
class PredictionResult:
    """Результат предсказания для одного или нескольких текстов."""

    labels: np.ndarray
    """1 — текст по теме, -1 — выброс."""

    scores: np.ndarray
    """Decision function score. Чем выше — тем увереннее модель, что текст по теме."""

    texts: list[str]
    """Исходные тексты."""

    def __repr__(self) -> str:
        lines = []
        for text, label, score in zip(self.texts, self.labels, self.scores):
            tag = "inlier" if label == 1 else "outlier"
            lines.append(f"[{tag}] (score={score:+.3f}) {text[:80]}")
        return "\n".join(lines)

    def __iter__(self):
        yield from zip(self.texts, self.labels, self.scores)

    def __len__(self) -> int:
        return len(self.labels)


class TopicClassifier:
    """One-Class SVM классификатор тематики текстов.

    Parameters
    ----------
    max_features : int
        Максимальное количество признаков TF-IDF.
    nu : float
        Верхняя граница доли выбросов (от 0 до 1).
        Например, 0.05 означает ≈5% выбросов в обучающих данных.
    kernel : str
        Ядро SVM: 'rbf', 'linear', 'poly', 'sigmoid'.
    gamma : str | float
        Коэффициент ядра. 'scale' или 'auto' — автоматический подбор.
    strip_html : bool
        Удалять ли HTML-теги из текстов.
    """

    def __init__(
        self,
        max_features: int = 5000,
        nu: float = 0.05,
        kernel: str = "rbf",
        gamma: Union[str, float] = "scale",
        strip_html: bool = True,
    ) -> None:
        self.max_features = max_features
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.strip_html = strip_html

        self._vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
        )
        self._model = OneClassSVM(kernel=kernel, gamma=gamma, nu=nu)
        self._is_fitted = False

    @staticmethod
    def _clean_html(text: str) -> str:
        return re.sub(r"<[^>]+>", " ", text).strip()

    def _preprocess(self, texts: list[str]) -> list[str]:
        if self.strip_html:
            return [self._clean_html(t) for t in texts]
        return list(texts)

    def fit(self, texts: list[str]) -> "TopicClassifier":
        """Обучить модель на текстах одной темы.

        Parameters
        ----------
        texts : list[str]
            Тексты, представляющие целевую тему.

        Returns
        -------
        self
        """
        cleaned = self._preprocess(texts)
        X = self._vectorizer.fit_transform(cleaned)
        self._model.fit(X)
        self._is_fitted = True
        return self

    def predict(self, texts: Union[str, list[str]]) -> PredictionResult:
        """Классифицировать текст(ы): по теме (1) или выброс (-1).

        Parameters
        ----------
        texts : str | list[str]
            Один текст или список текстов.

        Returns
        -------
        PredictionResult
        """
        if not self._is_fitted:
            raise RuntimeError("Модель не обучена. Сначала вызовите fit().")

        single = isinstance(texts, str)
        if single:
            texts = [texts]

        cleaned = self._preprocess(texts)
        X = self._vectorizer.transform(cleaned)

        labels = self._model.predict(X)
        scores = self._model.decision_function(X)

        # Пустой TF-IDF вектор → выброс
        norms = X.power(2).sum(axis=1).A1
        labels[norms == 0] = -1
        scores[norms == 0] = -1.0

        return PredictionResult(labels=labels, scores=scores, texts=texts)

    def save(self, path: Union[str, Path]) -> Path:
        """Сохранить обученную модель на диск.

        Parameters
        ----------
        path : str | Path
            Путь к файлу (рекомендуется расширение .joblib).

        Returns
        -------
        Path  — путь к сохранённому файлу.
        """
        path = Path(path)
        dump(
            {
                "vectorizer": self._vectorizer,
                "model": self._model,
                "params": {
                    "max_features": self.max_features,
                    "nu": self.nu,
                    "kernel": self.kernel,
                    "gamma": self.gamma,
                    "strip_html": self.strip_html,
                },
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TopicClassifier":
        """Загрузить ранее сохранённую модель.

        Parameters
        ----------
        path : str | Path
            Путь к .joblib файлу.

        Returns
        -------
        TopicClassifier
        """
        data = load(path)
        params = data["params"]
        instance = cls(**params)
        instance._vectorizer = data["vectorizer"]
        instance._model = data["model"]
        instance._is_fitted = True
        return instance
