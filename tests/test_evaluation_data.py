"""Original small synthetic data for dataset schema and exclusion contracts."""

import csv
import json
from pathlib import Path
import zipfile

import pytest

from src.evaluation_data import (
    deduplicate, fiqa_label, load_annotations, load_fiqa, load_phrasebank,
    normalize_label, prepare_annotations, validate_examples,
)
from src.ingestion import load_document
from src.preprocessing import preprocess_document, write_jsonl


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


def test_blind_annotation_sample_pending_and_review_gate(tmp_path):
    fixture = Path(__file__).parent / "fixtures" / "synthetic_earnings.json"
    records = preprocess_document(load_document(fixture), token_counter=lambda text: len(text.split()) + 2)
    source = tmp_path / "source.jsonl"
    write_jsonl(records, source)
    first, second = tmp_path / "first.csv", tmp_path / "second.csv"
    audit = prepare_annotations(source, first, size=3, seed=42)
    prepare_annotations(source, second, size=3, seed=42)
    assert first.read_bytes() == second.read_bytes()
    assert audit["selected"] == 3 and audit["annotation_status"] == "pending"
    with first.open() as stream:
        rows = list(csv.DictReader(stream))
    assert all(not row["human_label"] and row["review_status"] == "pending" for row in rows)
    assert not ({"predicted_sentiment", "prob_positive", "prob_negative", "prob_neutral"} & rows[0].keys())
    with pytest.raises(ValueError, match="pending"):
        load_annotations(first)
    for row in rows:  # Synthetic contract test; no real human annotations are claimed.
        row.update(human_label="neutral", annotator_id="synthetic_test", review_status="reviewed")
    reviewed = tmp_path / "reviewed.csv"
    with reviewed.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    assert len(load_annotations(reviewed).records) == 3
    with pytest.raises(FileExistsError):
        prepare_annotations(source, first, size=3)
