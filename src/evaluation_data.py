"""Small local evaluation datasets, explicit retrieval and blind human-annotation samples."""

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import random
import unicodedata
from urllib.request import Request, urlopen
import zipfile

from src.records import read_records

LABELS = ("negative", "neutral", "positive")
PHRASEBANK_REVISION = "8d3fe0c36d5feec6b3cc5e455b0fcb4820fb9964"
FIQA_REVISION = "acd0f4cfafa6421774bc8f009e8fa3e0ec441357"
FIQA_MAPPING = "negative: score < -0.1; neutral: -0.1 <= score < 0.1; positive: score >= 0.1"
PHRASEBANK_WARNING = "Financial PhraseBank overlaps FinBERT supervised training; this is not independent held-out performance."
FIQA_WARNING = "Original labelled training partition, evaluated without fitting; not held-out test performance or guaranteed unseen data. The original FinBERT study also used FiQA for regression experiments; the released classifier's FiQA exposure is unverified. Target/aspect labels are compared to a sentence-level model."
SOURCES = {
    "phrasebank": {"FinancialPhraseBank-v1.0.zip":
        f"https://huggingface.co/datasets/takala/financial_phrasebank/resolve/{PHRASEBANK_REVISION}/data/FinancialPhraseBank-v1.0.zip"},
    "fiqa": {f"task1_{kind}_ABSA_{split}.json":
        f"https://raw.githubusercontent.com/dayanfcosta/fiqa-2018-task1/{FIQA_REVISION}/datasets/task1_{kind}_ABSA_{split}.json"
        for kind in ("headline", "post") for split in ("train", "test")},
}


@dataclass
class EvaluationDataset:
    records: list[dict]
    audit: dict
    provenance: dict


def text_key(text: str) -> str:
    """Case/Unicode/whitespace normalization used only to identify duplicates."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def normalize_label(value, mapping: dict | None = None) -> str:
    if isinstance(value, bool):
        raise ValueError("Boolean values are not sentiment labels.")
    if mapping is None:
        if not isinstance(value, str):
            raise ValueError("Numeric sentiment labels require an explicit mapping.")
        label = value.strip().lower()
    else:
        try:
            label = mapping[value]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Unknown sentiment label: {value!r}") from exc
    if label not in LABELS:
        raise ValueError(f"Unknown sentiment label: {value!r}")
    return label


def fiqa_label(score) -> str:
    if isinstance(score, bool):
        raise ValueError("FiQA score must be numeric, not boolean.")
    try:
        number = float(score)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid FiQA score: {score!r}") from exc
    if not math.isfinite(number) or not -1 <= number <= 1:
        raise ValueError("FiQA scores must be finite and in [-1, 1].")
    return "negative" if number < -.1 else "neutral" if number < .1 else "positive"


def validate_examples(records: list[dict]) -> None:
    if not records:
        raise ValueError("No labelled evaluation examples.")
    seen = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Evaluation examples must be objects.")
        for field in ("example_id", "text", "true_label", "source_dataset", "split"):
            if not isinstance(record.get(field), str) or not record[field].strip():
                raise ValueError(f"Evaluation example requires non-empty {field}; unlabelled examples cannot be scored.")
        if "original_label" not in record or record["original_label"] is None:
            raise ValueError("Original reference label must be preserved.")
        if record["true_label"] not in LABELS:
            raise ValueError("Unknown true_label in evaluation data.")
        if record["example_id"] in seen:
            raise ValueError(f"Duplicate example_id: {record['example_id']}")
        seen.add(record["example_id"])
    if len({(r["source_dataset"], r["split"]) for r in records}) != 1:
        raise ValueError("Evaluate one dataset and split at a time; do not silently pool partitions.")


def deduplicate(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Keep lowest stable ID for same-label duplicates; exclude all label conflicts."""
    validate_examples(records)
    groups = defaultdict(list)
    for record in sorted(records, key=lambda r: r["example_id"]):
        groups[text_key(record["text"])].append(record)
    retained, excluded = [], []
    for members in groups.values():
        if len({r["true_label"] for r in members}) > 1:
            excluded.extend({"example_id": r["example_id"], "reason": "conflicting_reference_labels"} for r in members)
        else:
            retained.append(members[0])
            excluded.extend({"example_id": r["example_id"], "reason": "duplicate_text",
                             "duplicate_of": members[0]["example_id"]} for r in members[1:])
    return sorted(retained, key=lambda r: r["example_id"]), sorted(excluded, key=lambda r: r["example_id"])


def _file_info(path: Path) -> dict:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}


def _retrieval_date(directory: Path) -> str:
    path = directory / "retrieval_manifest.json"
    return json.loads(path.read_text())["retrieval_date"] if path.exists() else "not_recorded"


def fetch_dataset(dataset: str, data_dir: str | Path, *, retrieval_date: str) -> dict:
    """Explicitly fetch only pinned small dataset files; never called by evaluation."""
    date.fromisoformat(retrieval_date)
    if dataset not in SOURCES:
        raise ValueError("Retrievable datasets are phrasebank and fiqa.")
    directory = Path(data_dir) / dataset
    directory.mkdir(parents=True, exist_ok=True)
    files = []
    for filename, url in SOURCES[dataset].items():
        destination = directory / filename
        if not destination.exists():
            with urlopen(Request(url, headers={"User-Agent": "FinSentiment/0.5 personal research"}), timeout=30) as response:
                content = response.read(2_000_001)
            if len(content) > 2_000_000:
                raise ValueError("Unexpected dataset size above 2 MB; retrieval stopped.")
            with destination.open("xb") as output:
                output.write(content)
        files.append({**_file_info(destination), "url": url})
    manifest = {"dataset": dataset, "retrieval_date": retrieval_date, "files": files}
    path = directory / "retrieval_manifest.json"
    if not path.exists():
        with path.open("x") as output:
            json.dump(manifest, output, indent=2)
    return manifest


def _phrasebank_rows(archive: Path, subset: str) -> list[dict]:
    with zipfile.ZipFile(archive) as source:
        matches = [name for name in source.namelist() if name.endswith(f"Sentences_{subset}.txt")]
        if len(matches) != 1:
            raise ValueError(f"Expected one Sentences_{subset}.txt in PhraseBank archive.")
        content = source.read(matches[0]).decode("iso-8859-1")
    records = []
    for index, line in enumerate(content.splitlines(), 1):
        try:
            text, label = line.rsplit("@", 1)
        except ValueError as exc:
            raise ValueError(f"Invalid PhraseBank row {index}.") from exc
        records.append({"example_id": f"phrasebank_{subset}_{index:05d}", "text": text.strip(),
                        "original_text": text, "true_label": normalize_label(label), "original_label": label,
                        "source_dataset": "financial_phrasebank", "split": subset.lower(),
                        "source_type": "financial_news", "source_row": index})
    validate_examples(records)
    return records


def load_phrasebank(data_dir: str | Path, *, split: str = "allagree") -> EvaluationDataset:
    if split != "allagree":
        raise ValueError("This protocol evaluates only the allagree subset; do not combine agreement subsets.")
    archive = Path(data_dir) / "phrasebank" / "FinancialPhraseBank-v1.0.zip"
    raw = _phrasebank_rows(archive, "AllAgree")
    records, excluded = deduplicate(raw)
    return EvaluationDataset(records, {
        "loaded_rows": len(raw), "retained_rows": len(records), "excluded": excluded,
        "class_counts": {label: sum(r["true_label"] == label for r in records) for label in LABELS},
    }, {"dataset": "financial_phrasebank", "split": "allagree", "revision": PHRASEBANK_REVISION,
        "source_url": "https://huggingface.co/datasets/takala/financial_phrasebank", "license": "CC-BY-NC-SA-3.0",
        "retrieval_date": _retrieval_date(archive.parent),
        "role": "training_overlap_descriptive_benchmark", "warning": PHRASEBANK_WARNING,
        "label_mapping": "Original named labels: negative, neutral, positive", "files": [_file_info(archive)]})


def _fiqa_object(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or any(not isinstance(record, dict) for record in raw.values()):
        raise ValueError(f"FiQA file must map source IDs to record objects: {path}")
    return raw


def load_fiqa(data_dir: str | Path, *, split: str = "train", phrasebank_reference: str | Path | None = None) -> EvaluationDataset:
    if split != "train":
        raise ValueError("The accessible original FiQA test files have no reference scores. Only train is eligible; this is not held-out test evaluation.")
    directory = Path(data_dir) / "fiqa"
    test_keys, test_count, files = set(), 0, []
    for kind in ("headline", "post"):
        path = directory / f"task1_{kind}_ABSA_test.json"
        raw = _fiqa_object(path)
        for record in raw.values():
            if not isinstance(record.get("sentence"), str) or not record["sentence"].strip():
                raise ValueError("Invalid text in unlabelled FiQA overlap reference.")
            test_keys.add(text_key(record["sentence"]))
        test_count += len(raw)
        files.append(_file_info(path))
    overlap_keys = set()
    if phrasebank_reference is not None:
        archive = Path(phrasebank_reference)
        overlap_keys = {text_key(r["text"]) for r in _phrasebank_rows(archive, "50Agree")}
        files.append(_file_info(archive))
    records, excluded, loaded = [], [], 0
    for kind in ("headline", "post"):
        path = directory / f"task1_{kind}_ABSA_train.json"
        raw = _fiqa_object(path)
        files.append(_file_info(path))
        for source_id, record in sorted(raw.items()):
            loaded += 1
            example_id = f"fiqa_train_{kind}_{source_id}"
            text, annotations = record.get("sentence"), record.get("info")
            if not isinstance(text, str) or not text.strip() or not isinstance(annotations, list) or not annotations:
                raise ValueError(f"Invalid FiQA record: {example_id}")
            if len(annotations) != 1:
                excluded.append({"example_id": example_id, "reason": "multiple_targets", "targets": len(annotations)})
                continue
            annotation = annotations[0]
            if (not isinstance(annotation, dict) or "sentiment_score" not in annotation
                    or not isinstance(annotation.get("target"), str) or not annotation["target"].strip()):
                raise ValueError(f"Missing reference score/target: {example_id}")
            label = fiqa_label(annotation["sentiment_score"])
            if text_key(text) in test_keys:
                excluded.append({"example_id": example_id, "reason": "text_overlaps_unlabelled_test"})
                continue
            if text_key(text) in overlap_keys:
                excluded.append({"example_id": example_id, "reason": "text_overlaps_phrasebank_50agree"})
                continue
            records.append({"example_id": example_id, "text": text, "original_text": text,
                            "true_label": label, "original_label": annotation["sentiment_score"],
                            "source_dataset": "fiqa_original", "split": "train", "source_type": kind,
                            "target": annotation["target"], "aspect": annotation.get("aspects"), "source_id": source_id})
    if not records:
        raise ValueError("No eligible single-target labelled FiQA records.")
    retained, duplicate_exclusions = deduplicate(records)
    excluded.extend(duplicate_exclusions)
    return EvaluationDataset(retained, {
        "loaded_rows": loaded, "unlabelled_test_rows_inspected": test_count,
        "retained_rows": len(retained), "excluded": sorted(excluded, key=lambda r: r["example_id"]),
        "class_counts": {label: sum(r["true_label"] == label for r in retained) for label in LABELS},
        "phrasebank_overlap_checked": phrasebank_reference is not None,
    }, {"dataset": "fiqa_original", "split": "train", "revision": FIQA_REVISION,
        "source_url": "https://github.com/dayanfcosta/fiqa-2018-task1",
        "license": "Organizer restricts data to noncommercial use; no extra redistribution permission assumed",
        "role": "external_candidate_original_train_diagnostic", "warning": FIQA_WARNING,
        "retrieval_date": _retrieval_date(directory),
        "label_mapping": FIQA_MAPPING, "files": files})


ANNOTATION_COLUMNS = ["example_id", "sentence_id", "document_id", "company", "ticker", "fiscal_year",
                      "fiscal_quarter", "source_url", "source_sha256", "text", "human_label", "annotator_id",
                      "review_status", "annotation_notes"]


def prepare_annotations(source_path: str | Path, output_path: str | Path, *, size: int = 80, seed: int = 42) -> dict:
    source = sorted(read_records(source_path), key=lambda r: r["sentence_id"])
    unique = {}
    for record in source:
        unique.setdefault(text_key(record["text"]), record)
    if type(size) is not int or size <= 0 or size > len(unique):
        raise ValueError(f"Annotation size must be from 1 to {len(unique)} distinct source texts.")
    groups = defaultdict(list)
    for record in unique.values():
        groups[record["document_id"]].append(record)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    selected = []
    while len(selected) < size:
        for doc_id in sorted(groups):
            if groups[doc_id] and len(selected) < size:
                selected.append(groups[doc_id].pop())
    random.Random(seed + 1).shuffle(selected)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=ANNOTATION_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for record in selected:
            writer.writerow({**{k: record[k] for k in ANNOTATION_COLUMNS if k in record},
                             "example_id": f"earnings_human_{record['sentence_id']}", "human_label": "",
                             "annotator_id": "", "review_status": "pending", "annotation_notes": ""})
    return {"seed": seed, "source_rows": len(source), "distinct_texts": len(unique), "selected": size,
            "duplicates_not_sampled": len(source) - len(unique),
            "per_document": dict(sorted(Counter(r["document_id"] for r in selected).items())),
            "annotation_status": "pending", "source_file": _file_info(Path(source_path))}


def load_annotations(path: str | Path) -> EvaluationDataset:
    with Path(path).open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    records = []
    for row in rows:
        if row.get("review_status") != "reviewed" or not row.get("annotator_id", "").strip() or not row.get("human_label", "").strip():
            raise ValueError("Human annotation is pending: every example needs a reviewed human label and annotator_id.")
        records.append({**row, "true_label": normalize_label(row["human_label"]), "original_label": row["human_label"],
                        "source_dataset": "earnings_human", "split": "annotation_sample", "source_type": "earnings"})
    validate_examples(records)
    records, excluded = deduplicate(records)
    return EvaluationDataset(records, {"loaded_rows": len(rows), "retained_rows": len(records), "excluded": excluded,
                                      "class_counts": {label: sum(r["true_label"] == label for r in records) for label in LABELS}},
                             {"dataset": "earnings_human", "split": "annotation_sample", "revision": "user-reviewed-local-file",
                              "warning": "Small human-reviewed in-domain sample; not a population accuracy estimate.",
                              "files": [_file_info(Path(path))]})


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fetch = subparsers.add_parser("fetch", help="Explicit small pinned-file download; never downloads a model.")
    fetch.add_argument("--dataset", choices=tuple(SOURCES), required=True)
    fetch.add_argument("--data-dir", type=Path, default=Path("data/evaluation"))
    fetch.add_argument("--retrieval-date", default=date.today().isoformat())
    annotations = subparsers.add_parser("prepare-annotations")
    annotations.add_argument("--input", type=Path, required=True)
    annotations.add_argument("--output", type=Path, required=True)
    annotations.add_argument("--size", type=int, default=80)
    annotations.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        if args.command == "fetch":
            result = fetch_dataset(args.dataset, args.data_dir, retrieval_date=args.retrieval_date)
        else:
            result = prepare_annotations(args.input, args.output, size=args.size, seed=args.seed)
    except (ValueError, OSError, zipfile.BadZipFile) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
