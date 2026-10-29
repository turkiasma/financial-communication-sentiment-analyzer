"""CPU evaluation against genuine references, a frozen baseline and descriptive subgroups."""

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import fmean

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from src.evaluation_baseline import BASELINE_VERSION, predict_baseline
from src.evaluation_data import (
    EvaluationDataset, LABELS, load_fiqa, load_phrasebank, load_sentfin,
    text_key, validate_examples,
)
from src.model_config import MAX_INPUT_TOKENS, MODEL_ID, MODEL_REVISION

EVALUATION_VERSION = "phase5_v1"


def classification_metrics(truth: list[str], predicted: list[str]) -> dict:
    """Fixed three-label metrics, including zero-support classes with zero division=0."""
    if not truth or len(truth) != len(predicted):
        raise ValueError("Metrics require a non-empty, equally sized truth/prediction pair.")
    if any(label not in LABELS for label in truth + predicted):
        raise ValueError("Metrics received an unknown sentiment label.")
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=list(LABELS), zero_division=0
    )
    return {
        "sample_size": len(truth), "accuracy": float(accuracy_score(truth, predicted)),
        "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()), "weighted_f1": float(sum(f1 * support) / len(truth)),
        "per_class": {label: {"precision": float(precision[i]), "recall": float(recall[i]),
                              "f1": float(f1[i]), "support": int(support[i])} for i, label in enumerate(LABELS)},
        "prediction_distribution": {label: predicted.count(label) for label in LABELS},
        "label_order": list(LABELS), "confusion_matrix": confusion_matrix(truth, predicted, labels=list(LABELS)).tolist(),
        "zero_division": 0,
    }


def _digest(records: list[dict]) -> str:
    ordered = sorted(records, key=lambda record: record["example_id"])
    return hashlib.sha256(json.dumps(ordered, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _check_prediction(row: dict) -> None:
    if not isinstance(row, dict) or row.get("predicted_label") not in LABELS:
        raise ValueError("Invalid predicted label in evaluation cache.")
    scores = row.get("probabilities")
    if not isinstance(scores, dict) or set(scores) != set(LABELS):
        raise ValueError("Prediction must include all three probabilities.")
    if any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in scores.values()):
        raise ValueError("Invalid sentiment probability.")
    if not math.isclose(sum(scores.values()), 1, rel_tol=0, abs_tol=1e-6):
        raise ValueError("Prediction probabilities must sum to 1.")
    if scores[row["predicted_label"]] != max(scores.values()):
        raise ValueError("Predicted label does not match maximum probability.")


def infer_examples(dataset: EvaluationDataset, *, analyzer=None, batch_size: int = 8, local_files_only: bool = True) -> dict:
    """Use one existing FinBERT analyzer, without truncation or fabricated references."""
    validate_examples(dataset.records)
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if analyzer is None:
        from src.inference import FinBertAnalyzer
        analyzer = FinBertAnalyzer(local_files_only=local_files_only)
    if analyzer.device.type != "cpu":
        raise ValueError("Evaluation must use CPU inference.")
    eligible, excluded = [], []
    for record in sorted(dataset.records, key=lambda row: row["example_id"]):
        count = len(analyzer.tokenizer.encode(record["text"], add_special_tokens=True, truncation=False, verbose=False))
        if count > MAX_INPUT_TOKENS:
            excluded.append({"example_id": record["example_id"], "reason": "over_model_token_limit", "token_count": count})
        else:
            eligible.append((record, count))
    predictions = []
    for start in range(0, len(eligible), batch_size):
        chunk = eligible[start:start + batch_size]
        batch = analyzer.predict_batch([record["text"] for record, count in chunk])
        if len(batch) != len(chunk):
            raise ValueError("FinBERT returned a different number of predictions than examples.")
        for (record, count), prediction in zip(chunk, batch):
            if prediction["sentence"] != record["text"]:
                raise ValueError("FinBERT prediction order differs from source examples.")
            row = {"example_id": record["example_id"], "predicted_label": prediction["label"],
                   "probabilities": prediction["probabilities"], "token_count": count}
            _check_prediction(row)
            predictions.append(row)
    return {"evaluation_version": EVALUATION_VERSION, "model_name": MODEL_ID, "model_revision": MODEL_REVISION,
            "device": "cpu", "cohort_sha256": _digest(dataset.records), "predictions": predictions, "excluded": excluded}


def error_tags(text: str) -> list[str]:
    """Objective lexical co-occurrences for review, not asserted causes of errors."""
    lowered = text.lower()
    patterns = {
        "negation_present": bool(re.search(r"\b(no|not|never|without|neither|nor)\b|n't", lowered)),
        "expense_and_increase_terms": bool(re.search(r"\b(expenses?|costs?)\b", lowered) and re.search(r"\b(increas\w*|higher|rose|rising)\b", lowered)),
        "revenue_and_decrease_terms": bool(re.search(r"\b(revenue|sales)\b", lowered) and re.search(r"\b(decreas\w*|declin\w*|lower|fell|fall\w*)\b", lowered)),
        "loss_term": bool(re.search(r"\b(loss|losses)\b", lowered)),
        "guidance_and_reduction_terms": bool(re.search(r"\b(guidance|forecast\w*|outlook)\b", lowered) and re.search(r"\b(cut|reduc\w*|lower\w*)\b", lowered)),
        "caution_term": bool(re.search(r"\b(may|might|uncertain\w*|challeng\w*|cautious|risk\w*)\b", lowered)),
        "financial_abbreviation": bool(re.search(r"\b(EPS|EBITDA|EBIT|GAAP|ROE|ROI)\b", text)),
    }
    return [name for name, present in patterns.items() if present]


def subgroup_metrics(records: list[dict], rows: list[dict], model_predictions: dict[str, list[str]]) -> list[dict]:
    """Conditional performance slices; confidence slices are defined by FinBERT."""
    dimensions = {"true_class": {}, "source_type": {}, "word_length": {}, "negation": {}, "finbert_confidence": {}}
    for index, (record, prediction) in enumerate(zip(records, rows)):
        confidence = max(prediction["probabilities"].values())
        values = {"true_class": record["true_label"], "source_type": record.get("source_type", "unspecified"),
                  "word_length": "<=20_words" if len(record["text"].split()) <= 20 else ">20_words",
                  "negation": "present" if "negation_present" in error_tags(record["text"]) else "absent",
                  "finbert_confidence": "<0.6" if confidence < .6 else "0.6_to_<0.9" if confidence < .9 else ">=0.9"}
        for dimension, value in values.items():
            dimensions[dimension].setdefault(value, []).append(index)
    output = []
    for model, predictions in model_predictions.items():
        overall = classification_metrics([r["true_label"] for r in records], predictions)
        for dimension, groups in dimensions.items():
            for group, indices in sorted(groups.items()):
                metrics = classification_metrics([records[i]["true_label"] for i in indices], [predictions[i] for i in indices])
                warnings = []
                if len(indices) < 30:
                    warnings.append("small subgroup (<30); descriptive only")
                if any(metrics["per_class"][label]["support"] == 0 for label in LABELS):
                    warnings.append("absent true class; fixed-three-class macro F1 includes zero-support classes")
                output.append({"model": model, "dimension": dimension, "group": group,
                               "sample_size": len(indices), "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
                               "accuracy_difference_from_overall": metrics["accuracy"] - overall["accuracy"],
                               "macro_f1_difference_from_overall": metrics["macro_f1"] - overall["macro_f1"],
                               **{f"support_{label}": metrics["per_class"][label]["support"] for label in LABELS},
                               "warning": "; ".join(warnings)})
    return output


def score_evaluation(dataset: EvaluationDataset, cache: dict, *, include_text: bool = False) -> dict:
    """Score both models on exactly the same reference cohort; no inference here."""
    validate_examples(dataset.records)
    if not isinstance(cache, dict) or not isinstance(cache.get("predictions"), list) or not isinstance(cache.get("excluded"), list):
        raise ValueError("Prediction cache must contain prediction and exclusion lists.")
    if len({text_key(record["text"]) for record in dataset.records}) != len(dataset.records):
        raise ValueError("Duplicate evaluation texts must be audited/deduplicated before scoring.")
    if (cache.get("cohort_sha256") != _digest(dataset.records) or cache.get("model_name") != MODEL_ID
            or cache.get("model_revision") != MODEL_REVISION or cache.get("device") != "cpu"
            or cache.get("evaluation_version") != EVALUATION_VERSION):
        raise ValueError("Prediction cache does not match the labelled cohort, model or protocol.")
    by_id = {record["example_id"]: record for record in dataset.records}
    seen = set()
    for row in cache.get("predictions", []):
        _check_prediction(row)
        if row.get("example_id") not in by_id or row["example_id"] in seen:
            raise ValueError("Duplicate or unknown evaluation prediction ID.")
        if type(row.get("token_count")) is not int or not 2 <= row["token_count"] <= MAX_INPUT_TOKENS:
            raise ValueError("Invalid token count in eligible prediction cache.")
        seen.add(row["example_id"])
    for row in cache.get("excluded", []):
        if not isinstance(row, dict):
            raise ValueError("Excluded evaluation entries must be objects.")
        if row.get("example_id") not in by_id or row["example_id"] in seen:
            raise ValueError("Duplicate or unknown excluded evaluation ID.")
        if row.get("reason") != "over_model_token_limit" or type(row.get("token_count")) is not int or row["token_count"] <= MAX_INPUT_TOKENS:
            raise ValueError("Unjustified evaluation exclusion in cache.")
        seen.add(row["example_id"])
    if seen != by_id.keys():
        raise ValueError("Prediction cache is missing evaluation examples.")
    rows = sorted(cache["predictions"], key=lambda r: r["example_id"])
    records = [by_id[row["example_id"]] for row in rows]
    if not records:
        raise ValueError("No token-compatible labelled examples; no metrics were fabricated.")
    truth = [record["true_label"] for record in records]
    predictions = {"finbert": [row["predicted_label"] for row in rows],
                   BASELINE_VERSION: [predict_baseline(record["text"]) for record in records]}
    metrics = {model: classification_metrics(truth, labels) for model, labels in predictions.items()}
    errors, correct_confidence, wrong_confidence = [], [], []
    for record, row in zip(records, rows):
        confidence = max(row["probabilities"].values())
        if record["true_label"] == row["predicted_label"]:
            correct_confidence.append(confidence)
            continue
        wrong_confidence.append(confidence)
        errors.append({"example_id": record["example_id"], "source_dataset": record["source_dataset"],
                       "split": record["split"], "source_type": record.get("source_type", "unspecified"),
                       "original_label": record["original_label"], "true_label": record["true_label"],
                       "predicted_label": row["predicted_label"], "confidence": confidence,
                       **{f"prob_{label}": row["probabilities"][label] for label in LABELS},
                       "error_type": f"{record['true_label']}_to_{row['predicted_label']}",
                       "pattern_tags": ";".join(error_tags(record["text"])),
                       "text": record["text"] if include_text else "",
                       "original_text": record.get("original_text", record["text"]) if include_text else "",
                       "text_sha256": hashlib.sha256(record["text"].encode()).hexdigest(),
                       "source_url": record.get("source_url", dataset.provenance.get("source_url", "")),
                       "target": record.get("target", ""), "aspect": record.get("aspect", "")})
    errors.sort(key=lambda row: (-row["confidence"], row["example_id"]))
    limitations = [dataset.provenance.get("warning", ""),
                   "No model/baseline fitting or threshold tuning on these labels; no claim that source text was absent from pretraining.",
                   "Reference annotation task and selected text domain may differ from the quarterly earnings application.",
                   "Subgroup differences are descriptive, with no significance/causal or demographic-fairness claim; absent classes affect fixed-label macro averages.",
                   "Softmax confidence is not calibrated correctness. Pattern tags are co-occurrences, not established error causes.",
                   "Quarterly earnings predictions remain unlabelled and are not used as evaluation ground truth."]
    return {"evaluation_version": EVALUATION_VERSION, "dataset": dataset.provenance, "dataset_audit": dataset.audit,
            "coverage": {"eligible_labelled_examples": len(dataset.records), "evaluated_examples": len(records),
                         "model_length_exclusions": cache.get("excluded", [])},
            "models": metrics, "baseline_version": BASELINE_VERSION,
            "baseline_source_sha256": hashlib.sha256(Path(__file__).with_name("evaluation_baseline.py").read_bytes()).hexdigest(),
            "model_name": MODEL_ID, "model_revision": MODEL_REVISION, "device": "cpu",
            "confidence_summary": {"correct_count": len(correct_confidence), "incorrect_count": len(wrong_confidence),
                                   "mean_correct": fmean(correct_confidence) if correct_confidence else None,
                                   "mean_incorrect": fmean(wrong_confidence) if wrong_confidence else None,
                                   "high_confidence_errors_at_least_0_9": sum(p >= .9 for p in wrong_confidence)},
            "errors": errors, "robustness": subgroup_metrics(records, rows, predictions),
            "text_included": include_text, "limitations": [note for note in limitations if note]}


def render_evaluation_report(result: dict) -> str:
    dataset = result["dataset"]
    lines = ["# FinSentiment model evaluation", "", "## Dataset and protocol", "",
             f"Dataset: **{dataset['dataset']}**; split/subset: **{dataset['split']}**; revision: `{dataset.get('revision', 'not supplied')}`.",
             f"Source: {dataset.get('source_url', 'not supplied')}.",
             f"License/access: {dataset.get('license', 'Original source restrictions apply')}.", "",
             f"**{dataset.get('warning', '')}**", "",
             f"Mapping: {dataset.get('label_mapping', 'Original named reference labels')}.",
             f"Evaluation role: {dataset.get('role', 'not supplied')}. Annotation provenance: {dataset.get('annotation_method', 'See dataset source/protocol')}.",
             f"Protocol: `{result['evaluation_version']}`; model `{MODEL_ID}` at `{MODEL_REVISION}`, CPU; baseline `{BASELINE_VERSION}`.",
             "The baseline vocabulary and thresholds were fixed before benchmark predictions; no fine-tuning or test-label fitting occurred.", "",
             f"Raw rows: {result['dataset_audit'].get('loaded_rows')}; retained after dataset exclusions: {result['coverage']['eligible_labelled_examples']}; evaluated by both models: {result['coverage']['evaluated_examples']}; overlong exclusions: {len(result['coverage']['model_length_exclusions'])}.",
             f"Dataset exclusion reasons: {dict(Counter(row['reason'] for row in result['dataset_audit'].get('excluded', [])))}.", "",
             "## FinBERT and baseline", "",
             "Macro F1 is the primary metric. Macro precision/recall/F1 use all three fixed classes; zero divisions are 0. Confusion matrices have true-label rows and predicted-label columns.", "",
             "| Model | N | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for model, metrics in result["models"].items():
        lines.append(f"| {model} | {metrics['sample_size']} | {metrics['accuracy']:.4f} | {metrics['macro_precision']:.4f} | {metrics['macro_recall']:.4f} | {metrics['macro_f1']:.4f} | {metrics['weighted_f1']:.4f} |")
    lines += ["", "Negative-class recall: " + "; ".join(
        f"{model}={metrics['per_class']['negative']['recall']:.4f} (support={metrics['per_class']['negative']['support']})"
        for model, metrics in result["models"].items()) + ".", ""]
    if "overlap" in result["dataset_audit"]:
        lines += ["Reference overlap audit: " + json.dumps(result["dataset_audit"]["overlap"], sort_keys=True) + ".",
                  "Near-duplicate screens are bounded word-set similarity checks, not exhaustive or semantic guarantees. They flag candidates for review and do not remove sentences based on model outcomes.", ""]
    for model, metrics in result["models"].items():
        lines += ["", f"### {model}: class results", "",
                  "| True class | Support | Precision | Recall | F1 | Predicted count |", "| --- | --- | --- | --- | --- | --- |"]
        for label in LABELS:
            row = metrics["per_class"][label]
            lines.append(f"| {label} | {row['support']} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {metrics['prediction_distribution'][label]} |")
        lines += ["", "| Truth / prediction | negative | neutral | positive |", "| --- | --- | --- | --- |"]
        for label, counts in zip(LABELS, metrics["confusion_matrix"]):
            lines.append(f"| {label} | " + " | ".join(map(str, counts)) + " |")
        lines += [""]
    lines += ["## Confidence and error review", "", json.dumps(result["confidence_summary"], sort_keys=True), "",
              "Observed FinBERT error transitions: " + json.dumps(dict(Counter(row["error_type"] for row in result["errors"])), sort_keys=True) + ".",
              "Observed error-tag counts (tags can overlap): " + json.dumps(dict(Counter(
                  tag for row in result["errors"] for tag in row["pattern_tags"].split(";") if tag)), sort_keys=True) + ".", "",
              "Pattern tags identify lexical co-occurrences for review; no causal error explanation is inferred. Text is omitted unless explicitly requested for local review.", "",
              "| Example ID | Truth | Prediction | Confidence | Pattern tags |", "| --- | --- | --- | --- | --- |"]
    for row in result["errors"][:10]:
        lines.append(f"| {row['example_id']} | {row['true_label']} | {row['predicted_label']} | {row['confidence']:.4f} | {row['pattern_tags']} |")
        if row["text"]:
            lines += ["", "> " + row["text"].replace("\n", "\n> "), ""]
    if not result["errors"]:
        lines.append("No incorrect predictions in this evaluated cohort; this does not prove general accuracy.")
    lines += ["", "## Descriptive robustness slices", "",
              "Length uses <=20 versus >20 whitespace words. Negation is lexical. Confidence strata are defined by FinBERT, including for baseline comparisons. Small or class-incomplete groups are not reliable estimates of general performance; this is not demographic fairness analysis.", "",
              "| Model | Dimension | Group | N | Accuracy | Macro F1 | Warning |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in result["robustness"]:
        lines.append(f"| {row['model']} | {row['dimension']} | {row['group']} | {row['sample_size']} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} | {row['warning']} |")
    lines += ["", "No company/quarter accuracy is inferred from unlabelled earnings predictions.", "", "## Limitations", ""]
    lines += [f"- {note}" for note in result["limitations"]]
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("x", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def export_evaluation(result: dict, cache: dict, output_dir: str | Path) -> None:
    directory = Path(output_dir)
    names = ("evaluation_metrics.json", "evaluation_metrics.csv", "confusion_matrix.csv", "error_analysis.csv",
             "robustness_summary.csv", "evaluation_report.md", "model_predictions.json", "excluded_examples.csv")
    if result["dataset"].get("role") == "primary_external_sample":
        names += ("external_evaluation_metrics.csv", "external_evaluation_metrics.json", "external_evaluation_report.md")
    if any((directory / name).exists() for name in names):
        raise ValueError("Evaluation artifacts already exist; choose a new output directory.")
    directory.mkdir(parents=True, exist_ok=True)
    for name, value in ((names[0], result), (names[6], cache)):
        with (directory / name).open("x", encoding="utf-8") as output:
            output.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")
    dataset, split = result["dataset"]["dataset"], result["dataset"]["split"]
    metric_rows, confusion_rows = [], []
    for model, metrics in result["models"].items():
        metric_rows.append({"dataset": dataset, "split": split, "model": model,
                            **{key: metrics[key] for key in ("sample_size", "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")},
                            **{f"f1_{label}": metrics["per_class"][label]["f1"] for label in LABELS},
                            **{f"support_{label}": metrics["per_class"][label]["support"] for label in LABELS},
                            **{f"predicted_{label}": metrics["prediction_distribution"][label] for label in LABELS}})
        for i, truth in enumerate(LABELS):
            for j, predicted in enumerate(LABELS):
                confusion_rows.append({"dataset": dataset, "split": split, "model": model, "true_label": truth,
                                       "predicted_label": predicted, "count": metrics["confusion_matrix"][i][j]})
    _write_csv(directory / names[1], metric_rows, list(metric_rows[0]))
    _write_csv(directory / names[2], confusion_rows, ["dataset", "split", "model", "true_label", "predicted_label", "count"])
    _write_csv(directory / names[3], result["errors"], [
        "example_id", "source_dataset", "split", "source_type", "original_label", "true_label", "predicted_label",
        "confidence", "prob_negative", "prob_neutral", "prob_positive", "error_type", "pattern_tags", "text",
        "original_text", "text_sha256", "source_url", "target", "aspect"])
    robust = [{"dataset": dataset, "split": split, **row} for row in result["robustness"]]
    _write_csv(directory / names[4], robust, list(robust[0]))
    exclusions = result["dataset_audit"].get("excluded", []) + result["coverage"]["model_length_exclusions"]
    _write_csv(directory / names[7], exclusions, ["example_id", "reason", "duplicate_of", "targets", "token_count"])
    with (directory / names[5]).open("x", encoding="utf-8") as output:
        output.write(render_evaluation_report(result))
    if len(names) > 8:
        for source_name, destination_name in ((names[1], names[8]), (names[0], names[9]), (names[5], names[10])):
            with (directory / destination_name).open("xb") as output:
                output.write((directory / source_name).read_bytes())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("sentfin", "fiqa", "phrasebank"), required=True)
    parser.add_argument("--split", help="allagree for PhraseBank; train for FiQA; single_entity for SEntFiN (no official test split).")
    parser.add_argument("--sample-size", type=int, help="SEntFiN stratified sample size; default 2000.")
    parser.add_argument("--seed", type=int, default=42, help="SEntFiN sampling seed; fixed protocol uses 42.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/evaluation"))
    parser.add_argument("--phrasebank-reference", type=Path, help="Optional 50Agree text-overlap reference archive for FiQA.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--include-text", action="store_true", help="Include reference/error text only in local ignored reports.")
    parser.add_argument("--predictions-input", type=Path, help="Validate and rescore a saved model_predictions.json without model loading.")
    args = parser.parse_args(argv)
    try:
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise ValueError("Evaluation output directory is not empty; choose a new directory before inference.")
        if args.sample_size is not None and args.dataset != "sentfin":
            raise ValueError("--sample-size applies only to SEntFiN; original benchmark cohorts are unchanged.")
        if args.dataset == "phrasebank":
            dataset = load_phrasebank(args.data_dir, split=args.split or "allagree")
        elif args.dataset == "fiqa":
            dataset = load_fiqa(args.data_dir, split=args.split or "train", phrasebank_reference=args.phrasebank_reference)
        else:
            if args.split not in (None, "single_entity"):
                raise ValueError("SEntFiN has no published test partition; use split single_entity.")
            dataset = load_sentfin(args.data_dir, sample_size=args.sample_size if args.sample_size is not None else 2000, seed=args.seed)
        if args.predictions_input is not None:
            cache = json.loads(args.predictions_input.read_text(encoding="utf-8"))
        else:
            cache = infer_examples(dataset, batch_size=args.batch_size, local_files_only=args.local_files_only)
        result = score_evaluation(dataset, cache, include_text=args.include_text)
        export_evaluation(result, cache, args.output_dir)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(dataset.provenance["warning"])
    for model, metrics in result["models"].items():
        print(f"{dataset.provenance['dataset']} / {dataset.provenance['split']} | {model} | N={metrics['sample_size']} | accuracy={metrics['accuracy']:.4f} | macro F1={metrics['macro_f1']:.4f}")
    print(f"CPU | Output: {args.output_dir}")


if __name__ == "__main__":
    main()
