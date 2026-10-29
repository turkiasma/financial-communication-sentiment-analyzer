"""Known metric arithmetic and model-independent evaluation contracts."""

from copy import deepcopy
import csv
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.evaluation import (
    _digest, classification_metrics, error_tags, export_evaluation, infer_examples,
    main, score_evaluation,
)
from src.evaluation_data import EvaluationDataset, LABELS, load_phrasebank
from src.model_config import MODEL_ID, MODEL_REVISION


@pytest.fixture
def cohort():
    texts = ["Costs did not improve.", "Statement number two.", "Profit grew.", "Revenue declined."]
    truth = ["negative", "neutral", "positive", "positive"]
    records = [{"example_id": f"synthetic_{i}", "text": text, "original_text": text,
                "true_label": label, "original_label": label, "source_dataset": "synthetic",
                "split": "test", "source_type": "headline" if i < 2 else "post"}
               for i, (text, label) in enumerate(zip(texts, truth))]
    dataset = EvaluationDataset(records, {"loaded_rows": 4, "retained_rows": 4, "excluded": []},
                                {"dataset": "synthetic", "split": "test", "revision": "synthetic-test-only",
                                 "warning": "Synthetic metric fixture; not an actual benchmark result."})
    return dataset


@pytest.fixture
def cache(cohort):
    predicted = ["negative", "positive", "positive", "neutral"]
    rows = []
    for i, (record, label) in enumerate(zip(cohort.records, predicted)):
        confidence = .95 if i == 1 else .8
        probabilities = {sentiment: confidence if sentiment == label else (1 - confidence) / 2 for sentiment in LABELS}
        rows.append({"example_id": record["example_id"], "predicted_label": label,
                     "probabilities": probabilities, "token_count": len(record["text"].split()) + 2})
    return {"evaluation_version": "phase5_v1", "model_name": MODEL_ID, "model_revision": MODEL_REVISION,
            "device": "cpu", "cohort_sha256": _digest(cohort.records), "predictions": rows, "excluded": []}


def test_known_metrics():
    metrics = classification_metrics(["negative", "neutral", "positive", "positive"],
                                     ["negative", "positive", "positive", "neutral"])
    assert metrics["sample_size"] == 4 and metrics["accuracy"] == .5
    assert metrics["macro_precision"] == metrics["macro_recall"] == metrics["macro_f1"] == .5
    assert metrics["weighted_f1"] == .5
    assert metrics["per_class"]["negative"] == {"precision": 1., "recall": 1., "f1": 1., "support": 1}
    assert metrics["per_class"]["neutral"]["recall"] == 0
    assert metrics["confusion_matrix"] == [[1, 0, 0], [0, 0, 1], [0, 1, 1]]
    assert metrics["prediction_distribution"] == {"negative": 1, "neutral": 1, "positive": 2}


def test_imbalance_weighting_and_zero_division():
    metrics = classification_metrics(["positive", "positive", "positive", "negative"], ["positive"] * 4)
    assert metrics["accuracy"] == .75
    assert metrics["macro_f1"] == pytest.approx((6 / 7) / 3)
    assert metrics["weighted_f1"] == pytest.approx((6 / 7) * .75)
    single = classification_metrics(["neutral", "neutral"], ["neutral", "neutral"])
    assert single["accuracy"] == single["weighted_f1"] == 1
    assert single["macro_f1"] == pytest.approx(1 / 3)
    assert single["per_class"]["negative"]["support"] == 0
    assert single["per_class"]["negative"]["precision"] == 0


@pytest.mark.parametrize("truth,predicted", [([], []), (["positive"], []), (["unknown"], ["neutral"])])
def test_invalid_metric_inputs(truth, predicted):
    with pytest.raises(ValueError):
        classification_metrics(truth, predicted)


def test_error_confidence_and_baseline_same_cohort(cohort, cache):
    result = score_evaluation(cohort, cache)
    assert result["models"]["finbert"]["sample_size"] == result["models"]["keyword_v1"]["sample_size"] == 4
    assert result["models"]["keyword_v1"]["accuracy"] == .75
    assert len(result["errors"]) == 2
    assert result["errors"][0]["example_id"] == "synthetic_1"
    assert result["errors"][0]["error_type"] == "neutral_to_positive"
    assert all(not row["text"] for row in result["errors"])
    confidence = result["confidence_summary"]
    assert confidence["high_confidence_errors_at_least_0_9"] == 1
    assert confidence["mean_correct"] == pytest.approx(.8)
    assert confidence["mean_incorrect"] == pytest.approx(.875)
    local = score_evaluation(cohort, cache, include_text=True)
    assert local["errors"][0]["text"] == cohort.records[1]["text"]


def test_objective_error_pattern_tags():
    tags = error_tags("Expenses increased, revenue declined, and guidance was reduced; no EPS loss is certain, but risks may remain.")
    assert {"negation_present", "expense_and_increase_terms", "revenue_and_decrease_terms",
            "guidance_and_reduction_terms", "loss_term", "financial_abbreviation", "caution_term"} <= set(tags)


def test_subgroup_counts_and_warnings(cohort, cache):
    result = score_evaluation(cohort, cache)
    rows = result["robustness"]
    for model in ("finbert", "keyword_v1"):
        for dimension in ("source_type", "true_class", "word_length", "negation", "finbert_confidence"):
            assert sum(row["sample_size"] for row in rows if row["model"] == model and row["dimension"] == dimension) == 4
    headline = next(r for r in rows if r["model"] == "finbert" and r["dimension"] == "source_type" and r["group"] == "headline")
    assert headline["accuracy"] == .5 and headline["sample_size"] == 2
    assert "small subgroup" in headline["warning"]
    assert headline["support_negative"] == 1 and headline["support_neutral"] == 1


def test_cache_missing_duplicate_bad_probability_or_cohort(cohort, cache):
    for malformed in ([], None, {"predictions": [], "excluded": None}):
        with pytest.raises(ValueError, match="Prediction cache"):
            score_evaluation(cohort, malformed)
    bad = deepcopy(cache);bad["predictions"].pop()
    with pytest.raises(ValueError, match="missing"):
        score_evaluation(cohort, bad)
    bad = deepcopy(cache);bad["predictions"].append(bad["predictions"][0])
    with pytest.raises(ValueError, match="Duplicate"):
        score_evaluation(cohort, bad)
    bad = deepcopy(cache);bad["predictions"][0]["probabilities"]["negative"] = float("nan")
    with pytest.raises(ValueError, match="probability"):
        score_evaluation(cohort, bad)
    bad = deepcopy(cache);bad["cohort_sha256"] = "wrong"
    with pytest.raises(ValueError, match="does not match"):
        score_evaluation(cohort, bad)


def test_cannot_score_unlabelled_or_duplicate_texts(cohort, cache):
    cohort.records[0]["true_label"] = ""
    with pytest.raises(ValueError, match="unlabelled"):
        score_evaluation(cohort, cache)
    cohort.records[0]["true_label"] = "negative"
    cohort.records[1]["text"] = cohort.records[0]["text"]
    with pytest.raises(ValueError, match="Duplicate evaluation texts"):
        score_evaluation(cohort, cache)


def test_long_exclusion_applies_to_both_models(cohort, cache):
    row = cache["predictions"].pop()
    cache["excluded"] = [{"example_id": row["example_id"], "reason": "over_model_token_limit", "token_count": 513}]
    result = score_evaluation(cohort, cache)
    assert result["coverage"]["evaluated_examples"] == 3
    assert all(metrics["sample_size"] == 3 for metrics in result["models"].values())


def test_inference_uses_analyzer_batches_once(cohort, cache):
    predictions_by_text = {record["text"]: row for record, row in zip(cohort.records, cache["predictions"])}
    batches = []
    def predict_batch(texts):
        batches.append(len(texts))
        return [{"sentence": text, "label": predictions_by_text[text]["predicted_label"],
                 "probabilities": predictions_by_text[text]["probabilities"]} for text in texts]
    analyzer = SimpleNamespace(device=SimpleNamespace(type="cpu"), predict_batch=predict_batch,
                               tokenizer=SimpleNamespace(encode=lambda text, **kwargs: list(range(len(text.split()) + 2))))
    assert infer_examples(cohort, analyzer=analyzer, batch_size=2) == cache
    assert batches == [2, 2]
    with pytest.raises(ValueError):
        infer_examples(cohort, analyzer=analyzer, batch_size=0)


def test_deterministic_all_artifacts(cohort, cache, tmp_path):
    result = score_evaluation(cohort, cache)
    export_evaluation(result, cache, tmp_path / "first")
    export_evaluation(score_evaluation(cohort, cache), cache, tmp_path / "second")
    for path in (tmp_path / "first").iterdir():
        assert path.read_bytes() == (tmp_path / "second" / path.name).read_bytes()
    data = json.loads((tmp_path / "first" / "evaluation_metrics.json").read_text())
    assert data == result
    with (tmp_path / "first" / "evaluation_metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2 and {row["sample_size"] for row in rows} == {"4"}
    with (tmp_path / "first" / "confusion_matrix.csv").open() as stream:
        matrices = list(csv.DictReader(stream))
    assert len(matrices) == 18
    assert sum(int(row["count"]) for row in matrices) == 8
    assert (tmp_path / "first" / "evaluation_report.md").exists()
    with pytest.raises(ValueError):
        export_evaluation(result, cache, tmp_path / "first")


def test_cache_cli_avoids_loading_model(cohort, cache, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("src.evaluation.load_phrasebank", lambda *args, **kwargs: cohort)
    def no_model(*args, **kwargs):
        raise AssertionError("Rescoring a cache must not load a model")
    monkeypatch.setattr("src.evaluation.infer_examples", no_model)
    path = tmp_path / "cache.json";path.write_text(json.dumps(cache))
    main(["--dataset", "phrasebank", "--predictions-input", str(path), "--output-dir", str(tmp_path / "out")])
    assert "N=4" in capsys.readouterr().out


def test_abandoned_earnings_options_removed(tmp_path, capsys):
    for args in (["--dataset", "earnings"], ["--dataset", "fiqa", "--annotations", "obsolete.csv"]):
        with pytest.raises(SystemExit) as error:
            main([*args, "--output-dir", str(tmp_path / "out")])
        assert error.value.code == 2
    capsys.readouterr()


def test_external_artifacts_reuse_existing_metrics(cohort, cache, tmp_path):
    result = score_evaluation(cohort, cache)
    result["dataset"]["role"] = "primary_external_sample"  # Original synthetic export contract only.
    export_evaluation(result, cache, tmp_path)
    assert (tmp_path / "external_evaluation_metrics.csv").read_bytes() == (tmp_path / "evaluation_metrics.csv").read_bytes()
    assert (tmp_path / "external_evaluation_metrics.json").read_bytes() == (tmp_path / "evaluation_metrics.json").read_bytes()
    assert (tmp_path / "external_evaluation_report.md").exists()


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("RUN_FINBERT_INTEGRATION") != "1", reason="Requires cached FinBERT.")
def test_cached_model_labelled_evaluation(cached_analyzer, cohort, tmp_path):
    cache = infer_examples(cohort, analyzer=cached_analyzer, batch_size=2)
    result = score_evaluation(cohort, cache)
    assert cache["device"] == "cpu"
    assert all(0 <= p <= 1 for row in cache["predictions"] for p in row["probabilities"].values())
    assert all(abs(sum(row["probabilities"].values()) - 1) < 1e-6 for row in cache["predictions"])
    assert result["coverage"]["evaluated_examples"] == 4
    export_evaluation(result, cache, tmp_path / "integration")


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("RUN_FINBERT_INTEGRATION") != "1", reason="Requires cached FinBERT and local genuine labels.")
def test_small_genuine_labelled_sample(cached_analyzer):
    if not Path("data/evaluation/phrasebank/FinancialPhraseBank-v1.0.zip").exists():
        pytest.skip("Genuine dataset is local; use explicit evaluation_data fetch first.")
    original = load_phrasebank("data/evaluation")
    dataset = EvaluationDataset(original.records[:6], {"loaded_rows": 6, "excluded": []}, original.provenance)
    cache = infer_examples(dataset, analyzer=cached_analyzer, batch_size=2)
    result = score_evaluation(dataset, cache)
    assert result["models"]["finbert"]["sample_size"] == 6
    assert sum(map(sum, result["models"]["finbert"]["confusion_matrix"])) == 6


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("RUN_FINBERT_INTEGRATION") != "1", reason="Requires cached FinBERT and local external source files.")
def test_genuine_sentfin_adapter_through_external_reports(cached_analyzer, tmp_path):
    from src.evaluation_data import load_sentfin
    required = [Path("data/evaluation/sentfin/SEntFiN.csv"), Path("data/evaluation/sentfin/LICENSE"),
                Path("data/evaluation/phrasebank/FinancialPhraseBank-v1.0.zip")]
    required += [Path("data/evaluation/fiqa") / f"task1_{kind}_ABSA_{split}.json"
                 for kind in ("headline", "post") for split in ("train", "test")]
    if any(not path.exists() for path in required):
        pytest.skip("Explicitly fetch SEntFiN and reference corpora first; tests never download data.")
    data = load_sentfin("data/evaluation", sample_size=6)
    cache = infer_examples(data, analyzer=cached_analyzer, batch_size=2)
    result = score_evaluation(data, cache)
    assert result["models"]["finbert"]["sample_size"] == result["models"]["keyword_v1"]["sample_size"] == 6
    export_evaluation(result, cache, tmp_path / "external")
    assert (tmp_path / "external" / "external_evaluation_metrics.json").exists()
