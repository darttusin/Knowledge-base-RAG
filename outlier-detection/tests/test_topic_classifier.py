import numpy as np
import pytest

from topic_classifier import PredictionResult, TopicClassifier

PYTORCH_TEXTS = [
    "How to create a tensor in PyTorch",
    "Using DataLoader for batch processing",
    "Training a neural network with torch.nn",
    "Gradient computation with autograd",
    "How to save and load PyTorch models",
    "Implementing custom Dataset class in PyTorch",
    "Using GPU acceleration with CUDA in PyTorch",
    "Understanding torch.optim optimizers",
    "Batch normalization in PyTorch layers",
    "Transfer learning with pretrained models in torchvision",
]


@pytest.fixture
def fitted_classifier():
    clf = TopicClassifier(nu=0.1)
    clf.fit(PYTORCH_TEXTS)
    return clf


class TestTopicClassifier:
    def test_fit_marks_as_fitted(self):
        clf = TopicClassifier()
        assert clf._is_fitted is False
        clf.fit(PYTORCH_TEXTS)
        assert clf._is_fitted is True

    def test_predict_before_fit_raises(self):
        clf = TopicClassifier()
        with pytest.raises(RuntimeError, match="не обучена"):
            clf.predict("any text")

    def test_predict_single_string(self, fitted_classifier):
        result = fitted_classifier.predict("How to use DataLoader in PyTorch?")
        assert len(result) == 1
        assert result.labels[0] in (1, -1)

    def test_predict_list(self, fitted_classifier):
        result = fitted_classifier.predict(["text one", "text two"])
        assert len(result) == 2

    def test_inlier_detection(self, fitted_classifier):
        result = fitted_classifier.predict("Training a neural network with PyTorch autograd")
        assert result.labels[0] == 1

    def test_outlier_detection(self, fitted_classifier):
        result = fitted_classifier.predict("Best recipe for chocolate cake with strawberries")
        assert result.labels[0] == -1

    def test_empty_tfidf_vector_is_outlier(self, fitted_classifier):
        result = fitted_classifier.predict("абвгдежз")
        assert result.labels[0] == -1
        assert result.scores[0] == -1.0

    def test_html_stripping(self):
        clf = TopicClassifier(strip_html=True)
        clf.fit(["<p>PyTorch tensor operations</p>"] * 10)
        result = clf.predict("<b>tensor</b> computation")
        assert len(result) == 1

    def test_no_html_stripping(self):
        clf = TopicClassifier(strip_html=False)
        clf.fit(["PyTorch tensor operations"] * 10)
        result = clf.predict("tensor computation")
        assert len(result) == 1

    def test_save_and_load(self, fitted_classifier, tmp_path):
        model_path = tmp_path / "model.joblib"
        fitted_classifier.save(model_path)
        assert model_path.exists()

        loaded = TopicClassifier.load(model_path)
        assert loaded._is_fitted is True
        assert loaded.nu == fitted_classifier.nu
        assert loaded.kernel == fitted_classifier.kernel

        original = fitted_classifier.predict(PYTORCH_TEXTS)
        restored = loaded.predict(PYTORCH_TEXTS)
        np.testing.assert_array_equal(original.labels, restored.labels)
        np.testing.assert_array_almost_equal(original.scores, restored.scores)

    def test_custom_params(self):
        clf = TopicClassifier(max_features=100, nu=0.1, kernel="linear", gamma="auto")
        clf.fit(PYTORCH_TEXTS)
        result = clf.predict("PyTorch tensor")
        assert len(result) == 1


class TestPredictionResult:
    def test_repr(self):
        result = PredictionResult(
            labels=np.array([1, -1]),
            scores=np.array([0.5, -0.3]),
            texts=["on topic", "off topic"],
        )
        text = repr(result)
        assert "[inlier]" in text
        assert "[outlier]" in text

    def test_iter(self):
        result = PredictionResult(
            labels=np.array([1, -1]),
            scores=np.array([0.5, -0.3]),
            texts=["a", "b"],
        )
        items = list(result)
        assert len(items) == 2
        assert items[0] == ("a", 1, 0.5)

    def test_len(self):
        result = PredictionResult(
            labels=np.array([1, -1, 1]),
            scores=np.array([0.5, -0.3, 0.8]),
            texts=["a", "b", "c"],
        )
        assert len(result) == 3
