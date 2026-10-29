"""Known arithmetic, provenance/coverage contracts and offline end-to-end checks."""

from copy import deepcopy
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from src.analysis import analyze_predictions, compare_quarters, export_analysis, main, render_report
from src.ingestion import FinancialDocument, load_document
from src.model_config import MODEL_ID, MODEL_REVISION
from src.preprocessing import preprocess_document, write_jsonl

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_earnings.json"


def sample(labels, *, ticker="TEST", company="Fictional Company", year=2025, quarter="Q1", suffix=""):
    metadata = json.loads(FIXTURE.read_text())
    metadata.update(document_id=f"{ticker.lower()}_{year}_{quarter.lower()}{suffix}",
                    ticker=ticker, company=company, fiscal_year=year, fiscal_quarter=quarter)
    metadata["extra"].update(selection_policy="synthetic_test_v1", comparison_profile="synthetic_test_v1")
    text = "\n\n".join(f"Financial statement number {i + 1} was provided." for i in range(len(labels)))
    source = preprocess_document(
        FinancialDocument(metadata, text, hashlib.sha256(text.encode()).hexdigest()),
        token_counter=lambda value: len(value.split()) + 2,
    )
    probabilities = {"positive": (.8, .1, .1), "negative": (.1, .8, .1), "neutral": (.1, .2, .7)}
    predictions = []
    for record, label in zip(source, labels):
        scores = probabilities[label]
        predictions.append({**record, "predicted_sentiment": label, "prob_positive": scores[0],
                            "prob_negative": scores[1], "prob_neutral": scores[2],
                            "model_name": MODEL_ID, "model_revision": MODEL_REVISION,
                            "inference_device": "cpu", "batch_inference_version": "1"})
    return source, predictions


@pytest.fixture
def known():
    return sample(["positive", "positive", "negative", "neutral"])


def test_counts_percentages_net_and_probability_means(known):
    source, predictions = known
    original = deepcopy(predictions)
    result = analyze_predictions(predictions, source_records=source)
    row = result["periods"][0]["summary"]
    assert (row["positive_count"], row["negative_count"], row["neutral_count"]) == (2, 1, 1)
    assert (row["positive_pct"], row["negative_pct"], row["neutral_pct"]) == (50, 25, 25)
    assert sum(row[label + "_pct"] for label in ("positive", "negative", "neutral")) == pytest.approx(100)
    assert row["net_sentiment"] == .25  # Neutral is included: (2 - 1) / 4.
    assert row["mean_prob_positive"] == pytest.approx(.45)
    assert row["mean_prob_negative"] == pytest.approx(.3)
    assert row["mean_prob_neutral"] == pytest.approx(.25)
    assert row["total_sentences"] == row["classified_sentences"] == 4
    assert row["skipped_sentences"] == 0 and row["coverage_pct"] == 100
    assert result["company_summaries"][0]["classified_sentences"] == 4
    assert result["processing_provenance"]["preprocessing_version"] == "2"
    assert predictions == original


def test_consecutive_changes_and_company_grouping(known):
    source, predictions = known
    next_source, next_predictions = sample(["negative", "neutral"], quarter="Q2")
    other_source, other_predictions = sample(["neutral"], ticker="OTHER", company="Other Fictional Company")
    result = analyze_predictions(predictions + next_predictions + other_predictions,
                                 source_records=source + next_source + other_source)
    assert result["coverage"]["companies"] == 2
    assert len(result["periods"]) == 3 and len(result["comparisons"]) == 1
    change = result["comparisons"][0]
    assert change["ticker"] == "TEST"
    assert change["change_positive_pp"] == -50
    assert change["change_negative_pp"] == 25
    assert change["change_neutral_pp"] == 25
    assert change["change_net_sentiment"] == -.75
    assert change["previous_classified_sentences"] == 4 and change["current_classified_sentences"] == 2


def test_year_rollover_and_gap():
    source1, predictions1 = sample(["positive"], year=2024, quarter="Q4")
    source2, predictions2 = sample(["neutral"], year=2025, quarter="Q1")
    source3, predictions3 = sample(["negative"], year=2025, quarter="Q3")
    result = analyze_predictions(predictions3 + predictions2 + predictions1,
                                 source_records=source3 + source1 + source2)
    summaries = [period["summary"] for period in result["periods"]]
    assert [(s["fiscal_year"], s["fiscal_quarter"]) for s in summaries] == [(2024, "Q4"), (2025, "Q1"), (2025, "Q3")]
    assert len(result["comparisons"]) == 1
    assert result["comparisons"][0]["change_net_sentiment"] == -1
    assert any("Missing fiscal quarters" in note for note in result["limitations"])


def test_multiple_documents_in_same_period():
    first_source, first_predictions = sample(["positive"])
    second_source, second_predictions = sample(["negative"], suffix="_second")
    result = analyze_predictions(first_predictions + second_predictions,
                                 source_records=first_source + second_source)
    row = result["periods"][0]["summary"]
    assert row["documents_analyzed"] == 2 and row["classified_sentences"] == 2
    assert row["net_sentiment"] == 0


def test_skips_and_zero_classified_group():
    first_source, predictions = sample(["positive"])
    second_source, unused = sample(["neutral"], quarter="Q2")
    second_source[0]["exceeds_token_limit"] = True
    skipped = [{**second_source[0], "skip_reason": "Overlong input retained for review."}]
    result = analyze_predictions(predictions, skipped_records=skipped, source_records=first_source + second_source)
    row = result["periods"][1]["summary"]
    assert row["classified_sentences"] == 0 and row["skipped_sentences"] == row["total_sentences"] == 1
    assert row["coverage_pct"] == 0
    assert row["neutral_count"] == 0
    assert all(row[key] is None for key in ("net_sentiment", "positive_pct", "negative_pct", "neutral_pct", "mean_prob_positive"))
    assert result["comparisons"][0]["change_net_sentiment"] is None
    assert "N/A" in render_report(result)
    all_skipped = analyze_predictions([], skipped_records=skipped, source_records=second_source)
    assert all_skipped["coverage"]["classified_sentences"] == 0


def test_unverified_coverage_is_explicit(known):
    _, predictions = known
    result = analyze_predictions(predictions)
    row = result["periods"][0]["summary"]
    assert row["total_sentences"] is None and row["coverage_pct"] is None
    assert row["coverage_verified"] is False
    assert any("coverage are unknown" in note for note in result["limitations"])


def test_unaccounted_source_record_is_error(known):
    source, predictions = known
    with pytest.raises(ValueError, match="Unaccounted source"):
        analyze_predictions(predictions[:-1], source_records=source)


def test_duplicate_ids_across_predictions_and_skips(known):
    source, predictions = known
    with pytest.raises(ValueError, match="Duplicate"):
        analyze_predictions(predictions + [predictions[0]])
    with pytest.raises(ValueError, match="Duplicate"):
        analyze_predictions(predictions, skipped_records=[{**source[0], "skip_reason": "Test"}])


def test_empty_input():
    with pytest.raises(ValueError, match="No classified or skipped"):
        analyze_predictions([])


@pytest.mark.parametrize("field,value", [
    ("predicted_sentiment", "happy"), ("prob_positive", -1), ("prob_negative", 1.5),
    ("prob_neutral", float("nan")), ("prob_positive", float("inf")), ("prob_positive", True),
    ("prob_positive", "0.8"), ("prob_positive", .2), ("model_revision", "other"),
    ("inference_device", "cuda"), ("batch_inference_version", ""),
    ("exceeds_token_limit", True), ("source_sha256", "wrong"), ("segmentation_unit", []),
])
def test_malformed_predictions(known, field, value):
    _, predictions = known
    predictions[0][field] = value
    with pytest.raises(ValueError):
        analyze_predictions(predictions)


def test_missing_probability_and_wrong_winner(known):
    _, predictions = known
    del predictions[0]["prob_neutral"]
    with pytest.raises(ValueError, match="Missing"):
        analyze_predictions(predictions)
    _, predictions = sample(["positive"])
    predictions[0]["predicted_sentiment"] = "negative"
    with pytest.raises(ValueError, match="largest"):
        analyze_predictions(predictions)


@pytest.mark.parametrize("field,value", [
    ("company", "Other Company"), ("fiscal_quarter", "Q2"), ("source_sha256", "a" * 64),
])
def test_inconsistent_document_metadata(known, field, value):
    _, predictions = known
    predictions[1][field] = value
    with pytest.raises(ValueError, match="Inconsistent"):
        analyze_predictions(predictions)


def test_mixed_processing_or_extraction_policies(known):
    _, predictions = known
    predictions[0]["preprocessing_version"] = "other"
    with pytest.raises(ValueError, match="Incompatible"):
        analyze_predictions(predictions)
    _, first = sample(["positive"])
    _, second = sample(["neutral"], quarter="Q2")
    second[0]["extra"]["selection_policy"] = "different_scope"
    with pytest.raises(ValueError, match="extraction policies"):
        analyze_predictions(first + second)


def test_mixed_model_pipeline_and_section_scopes():
    _, first = sample(["positive"])
    _, second = sample(["neutral"], quarter="Q2")
    second[0]["batch_inference_version"] = "other"
    with pytest.raises(ValueError, match="Incompatible model"):
        analyze_predictions(first + second)
    second[0]["batch_inference_version"] = "1"
    second[0]["extra"]["selected_sections"] = ["different section"]
    with pytest.raises(ValueError, match="extraction policies"):
        analyze_predictions(first + second)


def test_comparison_rejects_duplicate_groups(known):
    source, predictions = known
    row = analyze_predictions(predictions, source_records=source)["periods"][0]["summary"]
    with pytest.raises(ValueError, match="Duplicate company-period"):
        compare_quarters([row, row])


def test_changed_source_metadata_rejected(known):
    source, predictions = known
    predictions[0]["text"] = "Altered text."
    with pytest.raises(ValueError, match="changed"):
        analyze_predictions(predictions, source_records=source)


def test_evidence_references_and_absent_categories(known):
    source, predictions = known
    result = analyze_predictions(predictions, source_records=source)
    evidence = result["periods"][0]["evidence"]["positive"][0]
    assert evidence["sentence_id"] == predictions[0]["sentence_id"]  # Stable tie-break.
    for field in ("text", "original_text", "source_url", "document_id", "source_sha256"):
        assert evidence[field] == predictions[0][field]
    one_source, one_prediction = sample(["positive"])
    report = render_report(analyze_predictions(one_prediction, source_records=one_source))
    assert "Negative: no classified sentences" in report
    assert "Neutral: no classified sentences" in report


def test_deterministic_exports_and_schema(known, tmp_path):
    source, predictions = known
    first = analyze_predictions(predictions, source_records=source)
    second = analyze_predictions(list(reversed(predictions)), source_records=list(reversed(source)))
    assert first == second
    export_analysis(first, tmp_path / "one")
    export_analysis(second, tmp_path / "two")
    for path in (tmp_path / "one").iterdir():
        assert path.read_bytes() == (tmp_path / "two" / path.name).read_bytes()
    with (tmp_path / "one" / "quarterly_sentiment_summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["classified_sentences"] == "4" and rows[0]["net_sentiment"] == "0.25"
    assert json.loads((tmp_path / "one" / "sentiment_analysis.json").read_text()) == first
    assert (tmp_path / "one" / "quarterly_comparison.csv").read_text().startswith("company,ticker,")
    with pytest.raises(ValueError, match="already exists"):
        export_analysis(first, tmp_path / "one")


def test_cli_and_automatic_skip_discovery(tmp_path, capsys):
    source, predictions = sample(["positive", "neutral"])
    skip = {**source[1], "skip_reason": "Explicitly skipped for test."}
    input_path, source_path = tmp_path / "predictions.jsonl", tmp_path / "source.jsonl"
    write_jsonl(predictions[:1], input_path)
    write_jsonl(source, source_path)
    write_jsonl([skip], Path(str(input_path) + ".skipped.jsonl"))
    main(["--input", str(input_path), "--source-input", str(source_path), "--output-dir", str(tmp_path / "report")])
    assert "Classified: 1 | Known skips: 1" in capsys.readouterr().out
    output = json.loads((tmp_path / "report" / "sentiment_analysis.json").read_text())
    assert output["coverage"]["coverage_verified"] is True
    assert len(output["input_files"]) == 3


def test_analysis_imports_no_model_framework():
    script = "import src.analysis, sys; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
    subprocess.run([sys.executable, "-c", script], check=True, cwd=Path(__file__).parents[1])


def test_cli_malformed_input(tmp_path, capsys):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"prob_positive":NaN}\n')
    with pytest.raises(SystemExit) as error:
        main(["--input", str(path), "--output-dir", str(tmp_path / "report")])
    assert error.value.code == 2 and "Error:" in capsys.readouterr().err
    assert not (tmp_path / "report").exists()


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("RUN_FINBERT_INTEGRATION") != "1", reason="Requires cached FinBERT.")
def test_synthetic_complete_pipeline(cached_analyzer, tmp_path):
    from src.batch_inference import predict_records
    analyzer = cached_analyzer
    source = preprocess_document(load_document(FIXTURE), token_counter=lambda text: len(analyzer.tokenizer.encode(text, truncation=False)))
    batch = predict_records(source, analyzer=analyzer, batch_size=2)
    result = analyze_predictions(batch.predictions, skipped_records=batch.skipped, source_records=source)
    export_analysis(result, tmp_path / "report")
    assert result["coverage"]["classified_sentences"] == len(source)
    assert sum(p["summary"]["classified_sentences"] for p in result["periods"]) == len(batch.predictions)


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("RUN_FINBERT_INTEGRATION") != "1", reason="Requires cached FinBERT and local real TXT.")
def test_real_eight_release_pipeline(cached_analyzer, tmp_path):
    from src.batch_inference import predict_records
    manifests = [Path("data/metadata") / f"{ticker}_fy2025_q{quarter}.json"
                 for ticker in ("msft", "aapl") for quarter in range(1, 5)]
    if any(not path.exists() or not (path.parent / json.loads(path.read_text())["file_path"]).exists() for path in manifests):
        pytest.skip("Real TXT is intentionally local; follow docs/data_plan.md to prepare all eight releases.")
    analyzer = cached_analyzer
    source = []
    for path in manifests:
        document = load_document(path)
        assert document.source_sha256 == document.metadata["extra"]["extracted_txt_sha256"]
        records = preprocess_document(document, token_counter=lambda text: len(analyzer.tokenizer.encode(text, truncation=False)),
                                      long_sentence_policy="keep")
        for record in records:
            assert record["original_text"] == document.text[record["source_char_start"]:record["source_char_end"]]
        source.extend(records)
    batch = predict_records(source, analyzer=analyzer, batch_size=8)
    result = analyze_predictions(batch.predictions, skipped_records=batch.skipped, source_records=source)
    assert result["coverage"]["documents"] == result["coverage"]["company_periods"] == 8
    assert result["coverage"]["companies"] == 2 and len(result["comparisons"]) == 6
    assert sum(p["summary"]["classified_sentences"] for p in result["periods"]) == len(batch.predictions)
    assert sum(p["summary"]["skipped_sentences"] for p in result["periods"]) == len(batch.skipped)
    grouped = [sentence_id for p in result["periods"] for sentence_id in p["classified_sentence_ids"]]
    assert len(grouped) == len(set(grouped)) == len(batch.predictions)
    export_analysis(result, tmp_path / "real_report")
