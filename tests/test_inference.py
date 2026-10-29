"""Offline unit tests plus one opt-in, cache-only model integration test."""

import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from src.inference import FinBertAnalyzer, MODEL_ID, MODEL_REVISION, SENTIMENT_LABELS


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        # Deliberately reorder labels to detect hardcoded class indices.
        self.config = SimpleNamespace(
            id2label={0: "neutral", 1: "positive", 2: "negative"},
            max_position_embeddings=512,
        )

    def forward(self, **inputs):
        assert not self.training
        assert not torch.is_grad_enabled()
        assert all(t.device.type == "cpu" for t in inputs.values())
        return SimpleNamespace(logits=torch.tensor([[0.5, 2.0, -1.0]]))


@pytest.fixture
def analyzer(monkeypatch):
    tokenizer = Mock(return_value={"input_ids": torch.tensor([[101, 102]])})
    tokenizer_loader = Mock(return_value=tokenizer)
    model_loader = Mock(return_value=FakeModel())
    monkeypatch.setattr("src.inference.AutoTokenizer.from_pretrained", tokenizer_loader)
    monkeypatch.setattr("src.inference.AutoModelForSequenceClassification.from_pretrained", model_loader)
    instance = FinBertAnalyzer(local_files_only=True)
    instance.predict("Revenue rose.")
    instance.predict("Costs fell.")
    tokenizer_loader.assert_called_once_with(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True
    )
    model_loader.assert_called_once_with(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True, use_safetensors=True
    )
    return instance


def check_prediction(result, sentence):
    assert set(result) == {"sentence", "label", "probabilities"}
    assert result["sentence"] == sentence
    assert result["label"] in SENTIMENT_LABELS
    scores = result["probabilities"]
    assert set(scores) == SENTIMENT_LABELS
    assert all(isinstance(p, float) and 0 <= p <= 1 for p in scores.values())
    assert sum(scores.values()) == pytest.approx(1, abs=1e-6)
    assert result["label"] == max(scores, key=scores.__getitem__)


def test_prediction_and_label_mapping(analyzer):
    sentence = "Revenue rose."
    result = analyzer.predict(sentence)
    check_prediction(result, sentence)
    assert result["label"] == "positive"
    analyzer.tokenizer.assert_called_with(
        sentence, return_tensors="pt", truncation=True, max_length=512
    )


@pytest.mark.parametrize("sentence", ["", " ", "\n\t"])
def test_blank_input(analyzer, sentence):
    with pytest.raises(ValueError, match="empty"):
        analyzer.predict(sentence)


@pytest.mark.parametrize("sentence", [None, 42, ["Revenue rose."]])
def test_non_string_input(analyzer, sentence):
    with pytest.raises(TypeError, match="string"):
        analyzer.predict(sentence)


def test_cpu_and_repeatability(analyzer):
    assert analyzer.device.type == "cpu"
    assert all(p.device.type == "cpu" for p in analyzer.model.parameters())
    first = analyzer.predict("Revenue rose.")
    second = analyzer.predict("Revenue rose.")
    assert first["label"] == second["label"]
    assert first["probabilities"] == pytest.approx(second["probabilities"], abs=1e-6)


def test_invalid_model_mapping(analyzer, monkeypatch):
    model = FakeModel()
    model.config.id2label = {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}
    monkeypatch.setattr(
        "src.inference.AutoModelForSequenceClassification.from_pretrained", Mock(return_value=model)
    )
    with pytest.raises(ValueError, match="labels"):
        FinBertAnalyzer(local_files_only=True)


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("RUN_FINBERT_INTEGRATION") != "1",
    reason="Set RUN_FINBERT_INTEGRATION=1 after caching the model.",
)
def test_real_finbert_cpu_inference(cached_analyzer):
    # One load for all real-model assertions; never download during tests.
    analyzer = cached_analyzer
    assert not analyzer.model.training
    assert all(p.device.type == "cpu" for p in analyzer.model.parameters())
    sentence = "The company reported strong revenue growth."
    first = analyzer.predict(sentence)
    second = analyzer.predict(sentence)
    check_prediction(first, sentence)
    assert first["label"] == second["label"]
    assert first["probabilities"] == pytest.approx(second["probabilities"], abs=1e-6)
