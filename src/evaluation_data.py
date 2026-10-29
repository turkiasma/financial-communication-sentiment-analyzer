"""Small local evaluation datasets, explicit retrieval and external benchmark audits."""

import argparse
import ast
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import random
import re
import unicodedata
from urllib.request import Request, urlopen
import zipfile


LABELS = ("negative", "neutral", "positive")
PHRASEBANK_REVISION = "8d3fe0c36d5feec6b3cc5e455b0fcb4820fb9964"
FIQA_REVISION = "acd0f4cfafa6421774bc8f009e8fa3e0ec441357"
SENTFIN_REVISION = "eba43610fe3cb57a3e5773cd6c037da1f8992ad5"
SENTFIN_SHA256 = "570af7ba58f724f3fcc414dd5ba80da5b1a3b342134d76c6ebdf53a32c54a92f"
FIQA_MAPPING = "negative: score < -0.1; neutral: -0.1 <= score < 0.1; positive: score >= 0.1"
PHRASEBANK_WARNING = "Financial PhraseBank overlaps FinBERT supervised training; this is not independent held-out performance."
FIQA_WARNING = "Original labelled training partition, evaluated without fitting; not held-out test performance or guaranteed unseen data. The original FinBERT study also used FiQA for regression experiments; the released classifier's FiQA exposure is unverified. Target/aspect labels are compared to a sentence-level model."
SOURCES = {
    "sentfin": {filename: f"https://raw.githubusercontent.com/pyRis/SEntFiN/{SENTFIN_REVISION}/{filename}"
                for filename in ("SEntFiN.csv", "LICENSE")},
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
        raise ValueError("Retrievable datasets are phrasebank, fiqa and sentfin.")
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


def parse_sentfin_decisions(raw: str) -> tuple[dict[str, str], bool]:
    """Parse only a dictionary of string targets/labels, never executable code.

    Thirteen published rows have unescaped apostrophes in entity names. For those
    rows only, a restricted, full-coverage string-pair grammar reads the original
    target verbatim. Unknown labels or unparsed trailing content are errors.
    """
    if not isinstance(raw, str) or not raw.startswith("{") or not raw.endswith("}"):
        raise ValueError("SEntFiN Decisions must be a string dictionary.")
    fallback = False
    try:
        node = ast.parse(raw, mode="eval").body
        if not isinstance(node, ast.Dict):
            raise ValueError("Decisions must be a dictionary.")
        pairs = [(ast.literal_eval(key), ast.literal_eval(value)) for key, value in zip(node.keys, node.values)]
    except SyntaxError:
        fallback = True
        pattern = re.compile(r"\s*'(.+?)'\s*:\s*'([^']+)'\s*(?:,\s*|$)")
        inner, position, pairs = raw[1:-1], 0, []
        while position < len(inner):
            match = pattern.match(inner, position)
            if match is None:
                raise ValueError("Malformed SEntFiN Decisions string; no labels inferred.")
            pairs.append(match.groups())
            position = match.end()
    if not pairs:
        raise ValueError("SEntFiN requires at least one target/reference label.")
    decisions = {}
    for target, label in pairs:
        if not isinstance(target, str) or not target.strip() or not isinstance(label, str):
            raise ValueError("SEntFiN targets and reference labels must be strings.")
        if target in decisions:
            raise ValueError("Duplicate target in SEntFiN Decisions.")
        normalize_label(label)
        decisions[target] = label
    return decisions, fallback


def filter_reference_overlaps(records: list[dict], references: list[dict]) -> tuple[list[dict], list[dict], dict]:
    exact = {record["text"] for record in references}
    normalized = {text_key(record["text"]) for record in references}
    retained, excluded = [], []
    for record in records:
        reason = ("exact_reference_overlap" if record["text"] in exact else
                  "normalized_reference_overlap" if text_key(record["text"]) in normalized else None)
        if reason:
            excluded.append({"example_id": record["example_id"], "reason": reason})
        else:
            retained.append(record)
    return retained, excluded, {"reference_rows": len(references),
                                "exact_matches": sum(row["reason"] == "exact_reference_overlap" for row in excluded),
                                "normalized_only_matches": sum(row["reason"] == "normalized_reference_overlap" for row in excluded)}


def near_duplicate_audit(records: list[dict], references: list[dict] | None = None) -> dict:
    """Bounded lexical similarity screen, not semantic deduplication or a proof of no overlap.

    Word-set Jaccard >=0.9, >=4 tokens; five rarest reference tokens generate up
    to 200 candidates per row, with 2,000 reported pairs maximum. No text is
    discarded: a changed number/negation can change financial meaning.
    """
    token_sets, postings, ids, normalized = [], defaultdict(list), [], []
    def add(record):
        tokens = set(re.findall(r"\w+", text_key(record["text"])))
        index = len(ids)
        ids.append(record["example_id"])
        token_sets.append(tokens)
        normalized.append(text_key(record["text"]))
        for token in tokens:
            postings[token].append(index)
    if references is not None:
        for record in references:
            add(record)
    pairs, checked = [], 0
    for record in sorted(records, key=lambda row: row["example_id"]):
        tokens = set(re.findall(r"\w+", text_key(record["text"])))
        if len(tokens) >= 4:
            candidates = set()
            rare = sorted((token for token in tokens if 0 < len(postings[token]) <= 200),
                          key=lambda token: (len(postings[token]), token))[:5]
            for token in rare:
                candidates.update(postings[token])
            for index in sorted(candidates)[:200]:
                if normalized[index] == text_key(record["text"]):
                    continue  # Exact/normalized equality has a separate complete audit.
                checked += 1
                other = token_sets[index]
                similarity = len(tokens & other) / len(tokens | other)
                if similarity >= .9:
                    pairs.append({"example_id": record["example_id"], "reference_id": ids[index], "word_set_jaccard": similarity})
                    if len(pairs) == 2000:
                        return {"pairs": pairs, "truncated": True, "candidate_comparisons": checked,
                                "method": "bounded word-set Jaccard >=0.9; report only, not exhaustive"}
        if references is None:
            add(record)
    return {"pairs": pairs, "truncated": False, "candidate_comparisons": checked,
            "method": "bounded word-set Jaccard >=0.9; report only, not exhaustive"}


def stratified_sample(records: list[dict], size: int, seed: int = 42) -> list[dict]:
    """Largest-remainder class quotas preserve proportions, without using predictions."""
    validate_examples(records)
    if type(size) is not int or not 1 <= size <= len(records):
        raise ValueError(f"Sample size must be from 1 to {len(records)}.")
    groups = {label: sorted((r for r in records if r["true_label"] == label), key=lambda r: r["example_id"]) for label in LABELS}
    raw = {label: size * len(groups[label]) / len(records) for label in LABELS}
    quotas = {label: math.floor(raw[label]) for label in LABELS}
    for label in sorted(LABELS, key=lambda label: (-(raw[label] - quotas[label]), label))[:size - sum(quotas.values())]:
        quotas[label] += 1
    selected, rng = [], random.Random(seed)
    for label in LABELS:
        rng.shuffle(groups[label])
        selected.extend(groups[label][:quotas[label]])
    return sorted(selected, key=lambda row: row["example_id"])


def _external_references(data_dir: Path) -> tuple[list[dict], list[dict]]:
    archive = data_dir / "phrasebank" / "FinancialPhraseBank-v1.0.zip"
    references = [{"example_id": "reference_" + r["example_id"], "text": r["text"]}
                  for r in _phrasebank_rows(archive, "50Agree")]
    files = [_file_info(archive)]
    for kind in ("headline", "post"):
        for split in ("train", "test"):
            path = data_dir / "fiqa" / f"task1_{kind}_ABSA_{split}.json"
            raw = _fiqa_object(path)
            for source_id, record in raw.items():
                if not isinstance(record.get("sentence"), str) or not record["sentence"].strip():
                    raise ValueError("Invalid reference text in overlap audit.")
                references.append({"example_id": f"reference_fiqa_{kind}_{split}_{source_id}", "text": record["sentence"]})
            files.append(_file_info(path))
    return references, files


def load_sentfin(data_dir: str | Path, *, sample_size: int = 2000, seed: int = 42) -> EvaluationDataset:
    root = Path(data_dir)
    path = root / "sentfin" / "SEntFiN.csv"
    if _file_info(path)["sha256"] != SENTFIN_SHA256:
        raise ValueError("SEntFiN content does not match the pinned author-published dataset; fetch the exact snapshot.")
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != ["S No.", "Title", "Decisions", "Words"]:
            raise ValueError("Unexpected author-published SEntFiN columns.")
        rows = list(reader)
    records, excluded, seen_ids, fallbacks = [], [], set(), 0
    for row in rows:
        if (not isinstance(row.get("S No."), str) or not row["S No."].isdigit()
                or row["S No."] in seen_ids or not isinstance(row.get("Title"), str) or not row["Title"].strip()):
            raise ValueError("Invalid or duplicate SEntFiN source ID/text.")
        seen_ids.add(row["S No."])
        example_id = f"sentfin_{int(row['S No.']):05d}"
        decisions, fallback = parse_sentfin_decisions(row["Decisions"])
        fallbacks += fallback
        if len(decisions) != 1:
            excluded.append({"example_id": example_id, "reason": "multiple_entities", "targets": len(decisions)})
            continue
        target, label = next(iter(decisions.items()))
        records.append({"example_id": example_id, "text": row["Title"], "original_text": row["Title"],
                        "true_label": normalize_label(label), "original_label": label, "target": target,
                        "original_decisions": row["Decisions"], "source_dataset": "sentfin", "split": "single_entity",
                        "source_type": "financial_headline", "source_row": int(row["S No."])})
    single_entity_count = len(records)
    records, duplicate_exclusions = deduplicate(records)
    excluded.extend(duplicate_exclusions)
    references, reference_files = _external_references(root)
    records, overlaps, overlap_counts = filter_reference_overlaps(records, references)
    excluded.extend(overlaps)
    within_near = near_duplicate_audit(records)
    cross_near = near_duplicate_audit(records, references)
    eligible_count = len(records)
    eligible_counts = dict(Counter(row["true_label"] for row in records))
    selected = stratified_sample(records, sample_size, seed)
    selected_ids = {row["example_id"] for row in selected}
    excluded.extend({"example_id": row["example_id"], "reason": "outside_fixed_stratified_sample"}
                    for row in records if row["example_id"] not in selected_ids)
    for row in selected:
        row["split"] = f"single_entity_sample_{sample_size}_seed{seed}"
    return EvaluationDataset(selected, {
        "loaded_rows": len(rows), "single_entity_rows": single_entity_count, "filtered_eligible_rows": eligible_count,
        "retained_rows": len(selected), "eligible_class_counts": eligible_counts,
        "class_counts": dict(Counter(row["true_label"] for row in selected)), "sample_size": sample_size, "seed": seed,
        "excluded": sorted(excluded, key=lambda row: row["example_id"]), "overlap": overlap_counts,
        "restricted_decisions_parser_rows": fallbacks, "within_near_duplicates": within_near, "cross_reference_near_duplicates": cross_near,
        "sample_near_duplicate_pairs": sum(pair["example_id"] in selected_ids and pair["reference_id"] in selected_ids
                                           for pair in within_near["pairs"]),
    }, {"dataset": "sentfin", "split": f"single_entity_sample_{sample_size}_seed{seed}", "revision": SENTFIN_REVISION,
        "source_url": "https://github.com/pyRis/SEntFiN", "paper_url": "https://arxiv.org/html/2305.12257",
        "license": "MIT in author repository; original news rights remain relevant; raw text kept local",
        "role": "primary_external_sample", "annotation_method": "Three human MBA annotators, consensus and curator adjudication, documented by authors",
        "label_mapping": "Original entity labels negative/neutral/positive; only single-entity headlines",
        "collection_period": "2011-2015 (authors' annotated corpus; not the separate 2002-2017 news-flow corpus)",
        "warning": f"A fixed {sample_size:,}-example external sample of human-labelled single-entity headlines, not the full dataset or an author-defined test split. Exact/normalized reference overlaps are excluded; pretraining/near-duplicate independence is not guaranteed.",
        "retrieval_date": _retrieval_date(path.parent),
        "sampling": "Proportional largest-remainder allocation by true class, seeded shuffle; no outcome-based selection",
        "files": [_file_info(path), _file_info(root / "sentfin" / "LICENSE")] + reference_files})


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fetch = subparsers.add_parser("fetch", help="Explicit small pinned-file download; never downloads a model.")
    fetch.add_argument("--dataset", choices=tuple(SOURCES), required=True)
    fetch.add_argument("--data-dir", type=Path, default=Path("data/evaluation"))
    fetch.add_argument("--retrieval-date", default=date.today().isoformat())
    args = parser.parse_args(argv)
    try:
        result = fetch_dataset(args.dataset, args.data_dir, retrieval_date=args.retrieval_date)
    except (ValueError, OSError, zipfile.BadZipFile) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
