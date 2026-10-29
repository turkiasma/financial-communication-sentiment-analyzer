"""Original small synthetic data for dataset schema and exclusion contracts."""

import csv
import json
import hashlib
from pathlib import Path
import zipfile

import pytest

from src.evaluation_data import (
    deduplicate, fiqa_label, load_fiqa, load_phrasebank,
    normalize_label, validate_examples,
    parse_sentfin_decisions, load_sentfin,
    filter_reference_overlaps, near_duplicate_audit, stratified_sample,
)


def example(identifier="one", text="Original financial statement.", label="neutral"):
    return {"example_id": identifier, "text": text, "true_label": label,
            "original_label": label, "source_dataset": "synthetic", "split": "test"}


@pytest.mark.parametrize("score,label", [(-1, "negative"), (-.100001, "negative"), (-.1, "neutral"),
                                        (0, "neutral"), (.099999, "neutral"), (.1, "positive"), (1, "positive")])
def test_fiqa_boundaries(score, label):
    assert fiqa_label(score) == label


@pytest.mark.parametrize("score", [None, True, "bad", -1.1, 1.1, float("inf"), float("nan")])
def test_invalid_fiqa_scores(score):
    with pytest.raises(ValueError):
        fiqa_label(score)


def test_named_and_explicit_numeric_mapping():
    assert normalize_label(" Positive ") == "positive"
    assert normalize_label(0, {0: "negative", 1: "neutral", 2: "positive"}) == "negative"
    for value in (0, True, "unknown"):
        with pytest.raises(ValueError):
            normalize_label(value)


def test_unknown_numeric_mapping():
    with pytest.raises(ValueError):
        normalize_label(9, {0: "negative"})


@pytest.mark.parametrize("change", [{"text": ""}, {"true_label": ""}, {"original_label": None},
                                    {"source_dataset": ""}, {"example_id": ""}, {"true_label": "unknown"}])
def test_invalid_and_unlabelled_records(change):
    with pytest.raises(ValueError):
        validate_examples([{**example(), **change}])


def test_empty_duplicate_id_and_mixed_splits():
    with pytest.raises(ValueError):
        validate_examples([])
    with pytest.raises(ValueError, match="Duplicate"):
        validate_examples([example(), example()])
    with pytest.raises(ValueError, match="one dataset and split"):
        validate_examples([example(), {**example("two"), "split": "train"}])


def test_duplicates_and_conflicts_are_audited():
    rows = [example("b", "Revenue grew.", "positive"), example("a", "REVENUE  grew.", "positive"),
            example("c", "Costs rose.", "negative"), example("d", "Costs rose.", "positive")]
    retained, excluded = deduplicate(rows)
    assert [r["example_id"] for r in retained] == ["a"]
    assert [r["reason"] for r in excluded] == ["duplicate_text", "conflicting_reference_labels", "conflicting_reference_labels"]
    assert excluded[0]["duplicate_of"] == "a"


def phrasebank_fixture(root, rows=None):
    folder = root / "phrasebank"
    folder.mkdir()
    archive = folder / "FinancialPhraseBank-v1.0.zip"
    content = rows or "Revenue grew.@positive\nThe board met.@neutral\nCosts harmed margins.@negative\nRevenue grew.@positive\n"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("FinancialPhraseBank-v1.0/Sentences_AllAgree.txt", content.encode("iso-8859-1"))
        output.writestr("FinancialPhraseBank-v1.0/Sentences_50Agree.txt", "Known overlap.@neutral\n")
    return archive


def test_phrasebank_loading_and_counts(tmp_path):
    phrasebank_fixture(tmp_path)
    dataset = load_phrasebank(tmp_path)
    assert len(dataset.records) == 3
    assert dataset.audit["loaded_rows"] == 4
    assert dataset.audit["class_counts"] == {"negative": 1, "neutral": 1, "positive": 1}
    assert "not independent" in dataset.provenance["warning"]
    assert all(record["split"] == "allagree" for record in dataset.records)
    with pytest.raises(ValueError):
        load_phrasebank(tmp_path, split="50agree")


def test_phrasebank_unknown_label(tmp_path):
    phrasebank_fixture(tmp_path, "Original sentence.@unknown\n")
    with pytest.raises(ValueError, match="Unknown"):
        load_phrasebank(tmp_path)


def fiqa_fixture(root):
    folder = root / "fiqa"
    folder.mkdir()
    annotation = lambda score: {"target": "Synthetic", "sentiment_score": score, "aspects": "original aspect"}
    files = {
        "headline_train": {
            "1": {"sentence": "Revenue improved.", "info": [annotation(".5")]},
            "2": {"sentence": "Mixed targets.", "info": [annotation(".2"), annotation("-.2")]},
            "3": {"sentence": "Test overlap.", "info": [annotation(".2")]},
            "4": {"sentence": "Known overlap.", "info": [annotation(".2")]},
        },
        "post_train": {"5": {"sentence": "Costs damaged margins.", "info": [annotation("-.5")]},
                       "6": {"sentence": "A statement was issued.", "info": [annotation("0")] }},
        "headline_test": {"7": {"sentence": "Test overlap.", "info": [{"target": "Synthetic"}]}},
        "post_test": {},
    }
    for name, value in files.items():
        kind, split = name.split("_")
        (folder / f"task1_{kind}_ABSA_{split}.json").write_text(json.dumps(value))
    return folder


def test_fiqa_loading_original_labels_and_exclusions(tmp_path):
    reference = phrasebank_fixture(tmp_path)
    fiqa_fixture(tmp_path)
    dataset = load_fiqa(tmp_path, phrasebank_reference=reference)
    assert len(dataset.records) == 3
    assert dataset.audit["loaded_rows"] == 6
    assert dataset.audit["class_counts"] == {"negative": 1, "neutral": 1, "positive": 1}
    assert {r["reason"] for r in dataset.audit["excluded"]} == {"multiple_targets", "text_overlaps_unlabelled_test", "text_overlaps_phrasebank_50agree"}
    assert dataset.records[0]["original_label"] == ".5"
    assert dataset.records[0]["target"] == "Synthetic"
    assert dataset.audit["phrasebank_overlap_checked"] is True
    with pytest.raises(ValueError, match="no reference scores"):
        load_fiqa(tmp_path, split="test")


def test_missing_data_does_not_download(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_phrasebank(tmp_path)


@pytest.mark.parametrize("invalid", [[], {"x": None}, {"x": {"sentence": "", "info": []}}])
def test_invalid_fiqa_structure(tmp_path, invalid):
    folder = fiqa_fixture(tmp_path)
    (folder / "task1_headline_ABSA_train.json").write_text(json.dumps(invalid))
    with pytest.raises(ValueError):
        load_fiqa(tmp_path)


@pytest.mark.parametrize("raw,expected,fallback", [
    ("{'Example': 'positive'}", {"Example": "positive"}, False),
    ("{'O'Brien': 'negative'}", {"O'Brien": "negative"}, True),
    ('{"Dr Reddy\'s": \'neutral\', \'Example\': \'positive\'}', {"Dr Reddy's": "neutral", "Example": "positive"}, False),
])
def test_actual_decisions_grammar(raw, expected, fallback):
    assert parse_sentfin_decisions(raw) == (expected, fallback)


@pytest.mark.parametrize("raw", ["{}", "[]", "{'Example': 'unknown'}", "{'Example': 1}",
                                "{'A': 'positive', 'A': 'negative'}", "{'': 'positive'}",
                                "{'Example': __import__('os').getcwd()}", "{'O'Brien': 'positive' trailing}"])
def test_invalid_decisions_never_guessed_or_executed(raw):
    with pytest.raises(ValueError):
        parse_sentfin_decisions(raw)


def test_exact_and_normalized_reference_overlap():
    records = [example("one", "Revenue grew."), example("two", "  REVENUE  grew. "), example("three", "Different statement.")]
    retained, excluded, counts = filter_reference_overlaps(records, [{"example_id": "ref", "text": "Revenue grew."}])
    assert [row["example_id"] for row in retained] == ["three"]
    assert counts["exact_matches"] == counts["normalized_only_matches"] == 1
    assert [row["reason"] for row in excluded] == ["exact_reference_overlap", "normalized_reference_overlap"]


def test_near_duplicate_screen_reports_without_changing_text():
    records = [example("one", "Revenue grew strongly in Europe."),
               example("two", "Revenue strongly grew in Europe."),
               example("three", "Revenue did not grow in Europe.")]
    audit = near_duplicate_audit(records)
    assert audit["pairs"] == [{"example_id": "two", "reference_id": "one", "word_set_jaccard": 1.0}]
    assert len(records) == 3 and "not" in records[2]["text"]
    assert near_duplicate_audit(records) == audit


def test_stratified_sampling_preserves_proportions_and_order():
    records = [example(f"id{i:02}", f"Original unique statement {i}.", label)
               for i, label in enumerate(["negative"] * 3 + ["neutral"] * 6 + ["positive"] * 3)]
    sample = stratified_sample(records, 8, seed=42)
    assert {label: sum(r["true_label"] == label for r in sample) for label in ("negative", "neutral", "positive")} == {
        "negative": 2, "neutral": 4, "positive": 2}
    assert stratified_sample(list(reversed(records)), 8, seed=42) == sample
    with pytest.raises(ValueError):
        stratified_sample(records, 13)


def sentfin_fixture(root, monkeypatch):
    phrasebank_fixture(root);fiqa_fixture(root)
    folder = root / "sentfin";folder.mkdir()
    rows = [
        (1, "Revenue grew.", "{'Example': 'positive'}"),
        (2, "The board met.", "{'Example': 'neutral'}"),
        (3, "Expenses harmed margins.", "{'Example': 'negative'}"),
        (4, "Mixed targets.", "{'Example': 'positive', 'Regulator': 'neutral'}"),
        (5, "REVENUE  grew.", "{'Example': 'positive'}"),
        (6, "Known overlap.", "{'Example': 'neutral'}"),
        (7, "Income fell sharply.", "{'Example': 'negative'}"),
        (8, "A new filing appeared.", "{'Example': 'neutral'}"),
    ]
    path = folder / "SEntFiN.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream);writer.writerow(["S No.", "Title", "Decisions", "Words"])
        writer.writerows((i, text, decisions, len(text.split())) for i, text, decisions in rows)
    (folder / "LICENSE").write_text("Synthetic fixture licence text; not a real dataset.")
    monkeypatch.setattr("src.evaluation_data.SENTFIN_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    return path


def test_sentfin_adapter_audit_and_provenance(tmp_path, monkeypatch):
    sentfin_fixture(tmp_path, monkeypatch)
    data = load_sentfin(tmp_path, sample_size=3)
    assert len(data.records) == 3 and data.audit["loaded_rows"] == 8
    assert data.audit["single_entity_rows"] == 7 and data.audit["filtered_eligible_rows"] == 5
    assert data.audit["class_counts"] == {"negative": 1, "neutral": 1, "positive": 1}
    assert data.provenance["role"] == "primary_external_sample"
    assert data.provenance["revision"] and data.provenance["files"][0]["sha256"]
    assert data.audit["overlap"]["exact_matches"] == 1
    assert all(record["original_decisions"] and record["target"] for record in data.records)
    assert load_sentfin(tmp_path, sample_size=3).records == data.records


def test_sentfin_requires_pinned_dataset_identity(tmp_path, monkeypatch):
    path = sentfin_fixture(tmp_path, monkeypatch)
    path.write_text(path.read_text() + "invalid row\n")
    with pytest.raises(ValueError, match="pinned"):
        load_sentfin(tmp_path, sample_size=3)


@pytest.mark.parametrize("field,value", [("Title", ""), ("Decisions", ""), ("Decisions", "{'Example': ''}")])
def test_sentfin_rejects_missing_text_or_reference_labels(tmp_path, monkeypatch, field, value):
    path = sentfin_fixture(tmp_path, monkeypatch)
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    rows[0][field] = value
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    monkeypatch.setattr("src.evaluation_data.SENTFIN_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError):
        load_sentfin(tmp_path, sample_size=3)
