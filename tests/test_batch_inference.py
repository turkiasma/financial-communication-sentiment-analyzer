"""Batch contracts, CPU tensor execution and opt-in offline model integration."""

from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from src.batch_inference import main, predict_records, read_records, validate_record
from src.ingestion import load_document
from src.inference import FinBertAnalyzer
from src.model_config import MODEL_ID, MODEL_REVISION
from src.preprocessing import preprocess_document, write_jsonl

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_earnings.json"


class FakeTokenizer:
    def encode(self, text, **options):
        assert options["truncation"] is False
        return [101] + [42] * len(text.split()) + [102]

    def __call__(self, sentences, **options):
        assert options == {"return_tensors": "pt", "padding": True, "truncation": False}
        rows = [self.encode(text, truncation=False) for text in sentences]
        width = max(map(len, rows))
        return {
            "input_ids": torch.tensor([row + [0] * (width - len(row)) for row in rows]),
            "attention_mask": torch.tensor([[1] * len(row) + [0] * (width - len(row)) for row in rows]),
        }


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(
            id2label={0: "neutral", 1: "positive", 2: "negative"}, max_position_embeddings=512
        )
        self.batch_sizes = []

    def forward(self, **inputs):
        assert not self.training
        assert not torch.is_grad_enabled()
        assert all(t.device.type == "cpu" for t in inputs.values())
        size = inputs["input_ids"].shape[0]
        self.batch_sizes.append(size)
        return SimpleNamespace(logits=torch.tensor([[0.5, 2.0, -1.0]]).repeat(size, 1))


@pytest.fixture
def analyzer(monkeypatch):
    model_loader = Mock(return_value=FakeModel())
    tokenizer_loader = Mock(return_value=FakeTokenizer())
    monkeypatch.setattr("src.inference.AutoModelForSequenceClassification.from_pretrained", model_loader)
    monkeypatch.setattr("src.inference.AutoTokenizer.from_pretrained", tokenizer_loader)
    result = FinBertAnalyzer(local_files_only=True)
    yield result
    model_loader.assert_called_once_with(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True, use_safetensors=True
    )
    tokenizer_loader.assert_called_once()


@pytest.fixture
def records():
    return preprocess_document(load_document(FIXTURE), token_counter=lambda text: len(text.split()) + 2)


def test_read_jsonl(records, tmp_path):
    path = tmp_path / "input.jsonl"
    write_jsonl(records, path)
    assert read_records(path) == records


def test_missing_input(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_records(tmp_path / "missing.jsonl")


@pytest.mark.parametrize("payload", ["{bad}\n", "[]\n", "42\n", '{"a":1,"a":2}\n',
                                     '{"extra":NaN}\n', '{"extra":Infinity}\n'])
def test_malformed_jsonl(tmp_path, payload):
    path = tmp_path / "bad.jsonl"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        read_records(path)


def test_empty_input(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("\n \n", encoding="utf-8")
    with pytest.raises(ValueError, match="no sentence records"):
        read_records(path)


def test_missing_field(records):
    del records[0]["source_sha256"]
    with pytest.raises(ValueError, match="source_sha256"):
        predict_records(records)


@pytest.mark.parametrize("field,value", [
    ("text", ""), ("text", " \n"), ("text", 1), ("original_text", ""),
    ("sentence_index", True), ("sentence_index", -1), ("sentence_id", "wrong"),
    ("source_char_start", -1), ("source_char_end", 0), ("source_sha256", "bad"),
    ("exceeds_token_limit", "false"), ("token_count", False), ("token_count", 0),
    ("max_tokens", 513), ("tokenizer_revision", "wrong"), ("tokenizer_model", "other/model"),
    ("long_sentence_policy", []), ("preprocessing_version", ""), ("company", ""),
])
def test_invalid_records(records, field, value):
    records[0][field] = value
    with pytest.raises(ValueError):
        validate_record(records[0])


def test_duplicate_or_existing_predictions(records):
    with pytest.raises(ValueError, match="Duplicate sentence_id"):
        predict_records([records[0], records[0]])
    records[0]["predicted_sentiment"] = "positive"
    with pytest.raises(ValueError, match="already contains"):
        predict_records(records)


@pytest.mark.parametrize("size", [0, -1, True, 1.5, "8"])
def test_invalid_batch_size(records, size):
    with pytest.raises(ValueError, match="positive integer"):
        predict_records(records, batch_size=size)


def check_predictions(source, predictions):
    assert len(source) == len(predictions)
    for original, predicted in zip(source, predictions):
        assert all(predicted[key] == value for key, value in original.items())
        assert predicted["model_name"] == MODEL_ID
        assert predicted["model_revision"] == MODEL_REVISION
        assert predicted["inference_device"] == "cpu"
        scores = [predicted[f"prob_{label}"] for label in ("positive", "negative", "neutral")]
        assert all(0 <= score <= 1 for score in scores)
        assert sum(scores) == pytest.approx(1, abs=1e-6)
    assert len({r["sentence_id"] for r in predictions}) == len(source)


def test_multiple_batches_mapping_order_and_metadata(records, analyzer):
    original = deepcopy(records)
    result = predict_records(records, analyzer=analyzer, batch_size=2)
    assert result.loaded_count == 5
    assert not result.skipped
    check_predictions(records, result.predictions)
    assert analyzer.model.batch_sizes == [2, 2, 1]
    assert all(r["predicted_sentiment"] == "positive" for r in result.predictions)
    assert records == original
    repeated = predict_records(records, analyzer=analyzer, batch_size=3)
    assert repeated == result


def test_unknown_source_fields_preserved(records, analyzer):
    records[0]["source_section"] = {"name": "test section"}
    result = predict_records(records, analyzer=analyzer)
    assert result.predictions[0]["source_section"] == records[0]["source_section"]


def test_flagged_only_avoids_model_load(records, monkeypatch):
    for record in records:
        record["exceeds_token_limit"] = True
    loader = Mock(side_effect=AssertionError("Model should not load for flagged-only input"))
    monkeypatch.setattr("src.batch_inference.FinBertAnalyzer", loader)
    result = predict_records(records)
    assert result.loaded_count == 5
    assert not result.predictions
    assert len(result.skipped) == 5
    assert all("reprocess" in r["skip_reason"] for r in result.skipped)
    assert all("predicted_sentiment" not in r for r in result.skipped)
    loader.assert_not_called()


def test_actual_length_rechecked_and_skips_ordered(records, analyzer):
    records[1]["exceeds_token_limit"] = True
    records[0]["text"] = "Revenue " * 520  # False flag and stale short count must not cause truncation.
    result = predict_records(records, analyzer=analyzer, batch_size=2)
    assert [r["sentence_id"] for r in result.skipped] == [records[0]["sentence_id"], records[1]["sentence_id"]]
    assert "Actual token count" in result.skipped[0]["skip_reason"]
    assert len(result.predictions) == 3
    all_ids = [r["sentence_id"] for r in result.predictions + result.skipped]
    assert len(set(all_ids)) == result.loaded_count == 5


def test_stale_token_count_rejected(records, analyzer):
    records[0]["token_count"] += 1
    with pytest.raises(ValueError, match="reprocess"):
        predict_records(records, analyzer=analyzer)


def test_cpu_required(records, analyzer):
    analyzer.device = torch.device("cuda")  # No CUDA tensor is constructed.
    with pytest.raises(ValueError, match="CPU"):
        predict_records(records, analyzer=analyzer)


def test_direct_batch_refuses_truncation(analyzer):
    with pytest.raises(ValueError, match="token limit"):
        analyzer.predict_batch(["revenue " * 520])
    assert analyzer.model.batch_sizes == []
    assert analyzer.predict_batch([]) == []


@pytest.mark.parametrize("sentences,error", [("not a list", TypeError), ([None], TypeError), ([" "], ValueError)])
def test_direct_batch_invalid_input(analyzer, sentences, error):
    with pytest.raises(error):
        analyzer.predict_batch(sentences)


def test_invalid_prediction_count(records, analyzer, monkeypatch):
    monkeypatch.setattr(analyzer, "predict_batch", lambda texts: [])
    with pytest.raises(ValueError, match="number of predictions"):
        predict_records(records, analyzer=analyzer)


def test_invalid_probabilities(records, analyzer, monkeypatch):
    monkeypatch.setattr(analyzer, "predict_batch", lambda texts: [
        {"sentence": text, "label": "positive", "probabilities": {"positive": 2., "neutral": 0., "negative": -1.}}
        for text in texts
    ])
    with pytest.raises(ValueError, match="invalid probabilities"):
        predict_records(records, analyzer=analyzer)


def test_cli_summary_and_skip_file(records, analyzer, tmp_path, monkeypatch, capsys):
    loader = Mock(return_value=analyzer)
    monkeypatch.setattr("src.batch_inference.FinBertAnalyzer", loader)
    records[1]["exceeds_token_limit"] = True
    source, output = tmp_path / "input.jsonl", tmp_path / "predictions.jsonl"
    write_jsonl(records, source)
    main(["--input", str(source), "--output", str(output), "--batch-size", "2", "--local-files-only"])
    captured = capsys.readouterr()
    assert "Loaded: 5 | Classified: 4 | Skipped: 1" in captured.out
    assert "Device: cpu" in captured.out
    assert records[1]["sentence_id"] in captured.err
    predictions = [json.loads(line) for line in output.read_text().splitlines()]
    check_predictions([r for i, r in enumerate(records) if i != 1], predictions)
    skips = [json.loads(line) for line in Path(str(output) + ".skipped.jsonl").read_text().splitlines()]
    assert len(skips) == 1 and "skip_reason" in skips[0]
    loader.assert_called_once_with(local_files_only=True)
    original = output.read_bytes()
    with pytest.raises(SystemExit) as error:
        main(["--input", str(source), "--output", str(output)])
    assert error.value.code == 2 and output.read_bytes() == original


def test_cli_bad_input_does_not_write(tmp_path, capsys):
    output = tmp_path / "out.jsonl"
    with pytest.raises(SystemExit) as error:
        main(["--input", str(tmp_path / "missing.jsonl"), "--output", str(output)])
    assert error.value.code == 2
    assert "Error:" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("RUN_FINBERT_INTEGRATION") != "1", reason="Requires cached FinBERT.")
def test_real_batch_offline_multiple_batches(cached_analyzer, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    analyzer = cached_analyzer
    records = preprocess_document(
        load_document(FIXTURE), token_counter=lambda text: len(analyzer.tokenizer.encode(text, truncation=False))
    )
    first = predict_records(records, analyzer=analyzer, batch_size=2)
    second = predict_records(records, analyzer=analyzer, batch_size=3)
    assert not first.skipped and not second.skipped
    check_predictions(records, first.predictions)
    assert not analyzer.model.training
    assert all(p.device.type == "cpu" for p in analyzer.model.parameters())
    for left, right in zip(first.predictions, second.predictions):
        assert left["sentence_id"] == right["sentence_id"]
        assert left["predicted_sentiment"] == right["predicted_sentiment"]
        for label in ("positive", "negative", "neutral"):
            assert left[f"prob_{label}"] == pytest.approx(right[f"prob_{label}"], abs=1e-6)
    single = analyzer.predict(records[0]["text"])
    assert first.predictions[0]["predicted_sentiment"] == single["label"]
    assert first.predictions[0]["prob_positive"] == pytest.approx(single["probabilities"]["positive"], abs=1e-6)
