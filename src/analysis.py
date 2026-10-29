"""Descriptive fiscal-quarter sentiment analysis of saved predictions; no model loading."""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any

from src.ingestion import REQUIRED_FIELDS
from src.model_config import MODEL_ID, MODEL_REVISION
from src.records import PREDICTION_FIELDS, read_jsonl, validate_record, validate_records

ANALYSIS_VERSION = "1"
LABELS = ("positive", "negative", "neutral")


def sentiment_distribution(labels: list[str]) -> dict[str, Any]:
    """Shared label-count definition for quarterly analysis and document uploads."""
    if any(label not in LABELS for label in labels):
        raise ValueError("Unknown sentiment label.")
    count = len(labels)
    result = {f"{label}_count": labels.count(label) for label in LABELS}
    result.update({f"{label}_pct": 100 * result[f"{label}_count"] / count if count else None
                   for label in LABELS})
    result["net_sentiment"] = (result["positive_count"] - result["negative_count"]) / count if count else None
    return result


SUMMARY_COLUMNS = [
    "company", "ticker", "fiscal_year", "fiscal_quarter", "documents_analyzed",
    "total_sentences", "classified_sentences", "skipped_sentences", "observed_sentences",
    "coverage_verified", "coverage_pct", "positive_count", "negative_count", "neutral_count",
    "positive_pct", "negative_pct", "neutral_pct", "mean_prob_positive", "mean_prob_negative",
    "mean_prob_neutral", "net_sentiment", "document_type", "selection_policy", "comparison_profile",
]
COMPARISON_COLUMNS = ["company", "ticker", "previous_fiscal_year", "previous_fiscal_quarter",
                      "current_fiscal_year", "current_fiscal_quarter"]
for _label in LABELS:
    COMPARISON_COLUMNS += [f"previous_{_label}_pct", f"current_{_label}_pct", f"change_{_label}_pp"]
COMPARISON_COLUMNS += ["previous_net_sentiment", "current_net_sentiment", "change_net_sentiment"]
for _period in ("previous", "current"):
    COMPARISON_COLUMNS += [f"{_period}_{field}" for field in (
        "classified_sentences", "skipped_sentences", "total_sentences", "coverage_pct", "coverage_verified")]

LIMITATIONS = [
    "Results describe selected corporate language, not underlying financial health or future stock performance; this is not investment advice.",
    "These are small, selected samples from corporate earnings releases; management language may be promotional and sentences within a release are not independent observations.",
    "No statistical significance, causality or model accuracy is established by quarterly differences.",
    "FinBERT probabilities are uncalibrated model outputs, and high-probability classifications can be wrong.",
    "Company-specific reporting structures and fiscal calendars differ; comparisons are within each company, not a ranking of companies.",
    "Sentence segmentation is heuristic. Preprocessing version 2 also retains unpunctuated narrative paragraphs/bullets as text segments.",
    "Coverage measures supplied extracted text, not the complete original earnings release. Skipped text is never counted as neutral.",
    "Net sentiment includes neutral classifications in its denominator; a score near zero can hide a large neutral share.",
    "Supporting sentences and derived outputs are for local review; public readability does not establish redistribution permission.",
]


def _base(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key not in PREDICTION_FIELDS}


def validate_prediction(record: Any) -> None:
    """Validate existing output without changing source facts or recalculating inference."""
    if not isinstance(record, dict):
        raise ValueError("Prediction must be a JSON object.")
    required = PREDICTION_FIELDS - {"skip_reason"}
    if required - record.keys():
        raise ValueError(f"Missing prediction fields: {', '.join(sorted(required - record.keys()))}")
    if "skip_reason" in record:
        raise ValueError("A skipped record must not have sentiment predictions.")
    validate_record(_base(record))
    label = record["predicted_sentiment"]
    if label not in LABELS:
        raise ValueError("predicted_sentiment must be positive, negative or neutral.")
    scores = {}
    for sentiment in LABELS:
        value = record[f"prob_{sentiment}"]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"prob_{sentiment} must be a finite probability from 0 to 1.")
        scores[sentiment] = value
    if not math.isclose(sum(scores.values()), 1, rel_tol=0, abs_tol=1e-6):
        raise ValueError("Sentiment probabilities must sum to 1.")
    if scores[label] < max(scores.values()):
        raise ValueError("Predicted sentiment does not match the largest probability.")
    if record["exceeds_token_limit"] or record["token_count"] > record["max_tokens"]:
        raise ValueError("An overlong/flagged record must not have a sentiment prediction.")
    if record["model_name"] != MODEL_ID or record["model_revision"] != MODEL_REVISION:
        raise ValueError("Prediction does not use the pinned FinBERT model/revision.")
    if record["inference_device"] != "cpu":
        raise ValueError("Expected CPU inference provenance.")
    if not isinstance(record["batch_inference_version"], str) or not record["batch_inference_version"].strip():
        raise ValueError("batch_inference_version must be a non-empty string.")


def _quarter_number(year: int, quarter: str) -> int:
    return year * 4 + int(quarter[1]) - 1


def compare_quarters(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare only adjacent fiscal quarters of the same company, including Q4→Q1."""
    ordered = sorted(summaries, key=lambda row: (
        row["ticker"], _quarter_number(row["fiscal_year"], row["fiscal_quarter"])))
    keys = [(row["ticker"], row["fiscal_year"], row["fiscal_quarter"]) for row in ordered]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate company-period summary.")
    comparisons = []
    for previous, current in zip(ordered, ordered[1:]):
        if previous["ticker"] != current["ticker"]:
            continue
        if any(previous.get(field) != current.get(field) for field in (
                "company", "document_type", "selection_policy", "comparison_profile")):
            raise ValueError("Incompatible company or extraction scope in quarterly summaries.")
        if _quarter_number(current["fiscal_year"], current["fiscal_quarter"]) != _quarter_number(
                previous["fiscal_year"], previous["fiscal_quarter"]) + 1:
            continue
        row = {"company": current["company"], "ticker": current["ticker"]}
        for prefix, summary in (("previous", previous), ("current", current)):
            for field in ("fiscal_year", "fiscal_quarter", "net_sentiment", "classified_sentences",
                          "skipped_sentences", "total_sentences", "coverage_pct", "coverage_verified"):
                row[f"{prefix}_{field}"] = summary[field]
            for label in LABELS:
                row[f"{prefix}_{label}_pct"] = summary[f"{label}_pct"]
        for label in LABELS:
            before, after = previous[f"{label}_pct"], current[f"{label}_pct"]
            row[f"change_{label}_pp"] = None if before is None or after is None else after - before
        before, after = previous["net_sentiment"], current["net_sentiment"]
        row["change_net_sentiment"] = None if before is None or after is None else after - before
        comparisons.append(row)
    return comparisons


def analyze_predictions(
    predictions: list[dict[str, Any]],
    *,
    skipped_records: list[dict[str, Any]] | None = None,
    source_records: list[dict[str, Any]] | None = None,
    evidence_per_class: int = 1,
) -> dict[str, Any]:
    """Aggregate validated predictions, reconciling skips against Phase 2 if supplied.

    Without source_records, total source size and coverage are explicitly unknown.
    Zero classified groups have null percentages/means/net, never fabricated zeros.
    """
    if type(evidence_per_class) is not int or not 1 <= evidence_per_class <= 3:
        raise ValueError("evidence_per_class must be an integer from 1 to 3.")
    skipped = skipped_records or []
    for prediction in predictions:
        validate_prediction(prediction)
    for record in skipped:
        if not isinstance(record, dict) or not isinstance(record.get("skip_reason"), str) or not record["skip_reason"].strip():
            raise ValueError("Skipped records must include a non-empty skip_reason.")
        if (PREDICTION_FIELDS - {"skip_reason"}) & record.keys():
            raise ValueError("Skipped records must not contain sentiment/model prediction fields.")
        validate_record(_base(record))
    observed = [_base(record) for record in predictions + skipped]
    validate_records(observed)  # Reject duplicates within/across classified and skipped inputs.
    if not observed:
        raise ValueError("No classified or skipped sentences to analyze.")
    coverage_verified = source_records is not None
    if source_records is not None:
        validate_records(source_records)
        source_by_id = {record["sentence_id"]: record for record in source_records}
        observed_by_id = {record["sentence_id"]: record for record in observed}
        if source_by_id.keys() != observed_by_id.keys():
            missing = sorted(source_by_id.keys() - observed_by_id.keys())
            extra = sorted(observed_by_id.keys() - source_by_id.keys())
            raise ValueError(f"Unaccounted source sentences {missing[:5]} or unexpected outputs {extra[:5]}; supply complete predictions and skips.")
        for sentence_id, source in source_by_id.items():
            if any(observed_by_id[sentence_id].get(key) != value for key, value in source.items()):
                raise ValueError(f"Source metadata/text changed for {sentence_id}.")
        canonical = source_records
    else:
        canonical = observed

    # These processing versions must be uniform even across different companies.
    processing_versions = {
        tuple(record.get(field, "legacy") for field in (
            "preprocessing_version", "tokenizer_model", "tokenizer_revision", "max_tokens", "segmentation_unit"))
        for record in canonical
    }
    model_versions = {(r["model_name"], r["model_revision"], r["batch_inference_version"]) for r in predictions}
    if len(processing_versions) != 1 or len(model_versions) > 1:
        raise ValueError("Incompatible model or preprocessing versions; reprocess consistently.")

    documents, company_names, company_tickers, profiles = {}, {}, {}, {}
    for record in canonical:
        ticker, company = record["ticker"], record["company"]
        if ticker in company_names and company_names[ticker] != company:
            raise ValueError("Inconsistent company name for a ticker.")
        if company in company_tickers and company_tickers[company] != ticker:
            raise ValueError("Inconsistent ticker for a company.")
        company_names[ticker], company_tickers[company] = company, ticker
        metadata = {field: record[field] for field in REQUIRED_FIELDS | {"extra"} if field in record}
        metadata["source_sha256"] = record["source_sha256"]
        doc_id = record["document_id"]
        if doc_id in documents and documents[doc_id] != metadata:
            raise ValueError(f"Inconsistent metadata/quarter/source hash within document {doc_id}.")
        documents[doc_id] = metadata
        extra = record.get("extra", {})
        policy = extra.get("selection_policy", "unspecified")
        profile = extra.get("comparison_profile", policy)
        if not isinstance(policy, str) or not isinstance(profile, str) or not policy or not profile:
            raise ValueError("Selection policy and comparison profile must be non-empty strings.")
        sections = extra.get("selected_sections", [])
        if not isinstance(sections, list) or any(not isinstance(section, str) or not section for section in sections):
            raise ValueError("selected_sections must be a list of non-empty section names.")
        signature = (record["document_type"], policy, profile, tuple(sections))
        if ticker in profiles and profiles[ticker] != signature:
            raise ValueError(f"Incompatible extraction policies/document types for {ticker}; do not mix scopes across quarters.")
        profiles[ticker] = signature

    groups = defaultdict(list)
    for record in canonical:
        groups[(record["ticker"], record["fiscal_year"], record["fiscal_quarter"])].append(record)
    predicted_by_id = {record["sentence_id"]: record for record in predictions}
    skipped_ids = {record["sentence_id"] for record in skipped}
    periods = []
    for key in sorted(groups, key=lambda k: (k[0], _quarter_number(k[1], k[2]))):
        members = sorted(groups[key], key=lambda r: (r["document_id"], r["sentence_index"]))
        classified = [predicted_by_id[r["sentence_id"]] for r in members if r["sentence_id"] in predicted_by_id]
        count = len(classified)
        first = members[0]
        doc_ids = sorted({r["document_id"] for r in members})
        row = {field: first[field] for field in ("company", "ticker", "fiscal_year", "fiscal_quarter")}
        row.update({
            "documents_analyzed": len(doc_ids), "total_sentences": len(members) if coverage_verified else None,
            "classified_sentences": count, "skipped_sentences": sum(r["sentence_id"] in skipped_ids for r in members),
            "observed_sentences": len(members), "coverage_verified": coverage_verified,
            "coverage_pct": 100 * count / len(members) if coverage_verified else None,
            "document_type": profiles[key[0]][0], "selection_policy": profiles[key[0]][1],
            "comparison_profile": profiles[key[0]][2],
        })
        evidence = {}
        for label in LABELS:
            candidates = [r for r in classified if r["predicted_sentiment"] == label]
            row[f"{label}_count"] = len(candidates)
            row[f"{label}_pct"] = 100 * len(candidates) / count if count else None
            row[f"mean_prob_{label}"] = fmean(r[f"prob_{label}"] for r in classified) if count else None
            ranked = sorted(candidates, key=lambda r: (-r[f"prob_{label}"], r["sentence_id"]))[:evidence_per_class]
            evidence[label] = [{
                **{field: r[field] for field in (
                    "sentence_id", "document_id", "company", "ticker", "fiscal_year", "fiscal_quarter",
                    "source_url", "text", "original_text", "source_sha256", "source_char_start", "source_char_end")},
                "predicted_sentiment": label, "probability": r[f"prob_{label}"],
            } for r in ranked]
        row.update(sentiment_distribution([r["predicted_sentiment"] for r in classified]))
        periods.append({
            "summary": row, "documents": [documents[doc_id] for doc_id in doc_ids], "evidence": evidence,
            "classified_sentence_ids": [r["sentence_id"] for r in classified],
            "skipped_sentence_ids": [r["sentence_id"] for r in members if r["sentence_id"] in skipped_ids],
        })
    summaries = [period["summary"] for period in periods]
    comparisons = compare_quarters(summaries)
    limitations = list(LIMITATIONS)
    if not coverage_verified:
        limitations.append("Original Phase 2 source records were not supplied: total source size and coverage are unknown; skipped counts include supplied skips only.")
    if any(profile[1] == "unspecified" for profile in profiles.values()):
        limitations.append("Some inputs have no extraction policy; source-scope comparability has not been established.")
    for previous, current in zip(summaries, summaries[1:]):
        if previous["ticker"] == current["ticker"] and _quarter_number(current["fiscal_year"], current["fiscal_quarter"]) > _quarter_number(previous["fiscal_year"], previous["fiscal_quarter"]) + 1:
            limitations.append(f"Missing fiscal quarters for {current['ticker']} between FY{previous['fiscal_year']} {previous['fiscal_quarter']} and FY{current['fiscal_year']} {current['fiscal_quarter']}; no comparison bridges the gap.")
    company_summaries = []
    for ticker in sorted(company_names):
        selected = [row for row in summaries if row["ticker"] == ticker]
        company_summaries.append({
            "company": company_names[ticker], "ticker": ticker,
            "fiscal_periods": [f"FY{row['fiscal_year']} {row['fiscal_quarter']}" for row in selected],
            "documents": sum(row["documents_analyzed"] for row in selected),
            "classified_sentences": sum(row["classified_sentences"] for row in selected),
            "skipped_sentences": sum(row["skipped_sentences"] for row in selected),
            "selection_policy": profiles[ticker][1], "comparison_profile": profiles[ticker][2],
        })
    return {
        "analysis_version": ANALYSIS_VERSION, "model_name": MODEL_ID, "model_revision": MODEL_REVISION,
        "net_sentiment_definition": "(positive_count - negative_count) / classified_sentences; neutral sentences are included in the denominator",
        "percentage_denominator": "classified_sentences", "mean_probability_denominator": "all classified sentences in the period",
        "processing_provenance": {field: canonical[0].get(field, "legacy") for field in (
            "preprocessing_version", "tokenizer_model", "tokenizer_revision", "max_tokens", "segmentation_unit")},
        "batch_inference_versions": sorted({r["batch_inference_version"] for r in predictions}),
        "company_summaries": company_summaries,
        "coverage": {"companies": len(company_names), "company_periods": len(periods), "documents": len(documents),
                     "total_sentences": len(canonical) if coverage_verified else None,
                     "classified_sentences": len(predictions), "skipped_sentences": len(skipped),
                     "coverage_verified": coverage_verified},
        "periods": periods, "comparisons": comparisons, "limitations": limitations,
    }


def _display(value: Any, digits: int = 1) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def render_report(result: dict[str, Any]) -> str:
    """Render measured values and exact supporting text, without invented interpretation."""
    coverage = result["coverage"]
    lines = ["# Financial communication sentiment report", "",
             "Descriptive analysis of saved FinBERT predictions. Sentiment describes selected corporate language; it is not investment advice or evidence of financial performance.", "",
             f"Companies: {coverage['companies']}; company-periods: {coverage['company_periods']}; documents: {coverage['documents']}; classified text units: {coverage['classified_sentences']}; known skips: {coverage['skipped_sentences']}.",
             f"Source coverage verified against Phase 2: {coverage['coverage_verified']}.", "",
             f"Model: `{result['model_name']}` at `{result['model_revision']}`; preprocessing version `{result['processing_provenance']['preprocessing_version']}`; analysis version `{result['analysis_version']}`.", "",
             "## Method and distributions", "",
             result["net_sentiment_definition"] + ". Percentages and mean probabilities use all classified units. Undefined metrics appear as N/A, not zero. Raw CSV/JSON retain arithmetic precision; displayed values are rounded.", "",
             "| Company | Fiscal period | Classified / source | Skipped | Positive % | Negative % | Neutral % | Net |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for period in result["periods"]:
        row = period["summary"]
        total = row["total_sentences"] if row["total_sentences"] is not None else "unknown"
        lines.append(f"| {row['company']} | FY{row['fiscal_year']} {row['fiscal_quarter']} | {row['classified_sentences']} / {total} | {row['skipped_sentences']} | {_display(row['positive_pct'])} | {_display(row['negative_pct'])} | {_display(row['neutral_pct'])} | {_display(row['net_sentiment'], 2)} |")
    lines += ["", "## Consecutive-quarter changes within each company", "",
              "Changes in class percentages are percentage points (pp). Net changes use the -1 to +1 scale. No statistical significance is asserted.", "",
              "| Company | Previous → current | Δ positive pp | Δ negative pp | Δ neutral pp | Δ net | Classified sample sizes |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in result["comparisons"]:
        lines.append(f"| {row['company']} | FY{row['previous_fiscal_year']} {row['previous_fiscal_quarter']} → FY{row['current_fiscal_year']} {row['current_fiscal_quarter']} | {_display(row['change_positive_pp'])} | {_display(row['change_negative_pp'])} | {_display(row['change_neutral_pp'])} | {_display(row['change_net_sentiment'], 2)} | {row['previous_classified_sentences']} → {row['current_classified_sentences']} |")
    if not result["comparisons"]:
        lines += ["", "No consecutive fiscal-quarter pairs are available."]
    lines += ["", "## Source documents, selection and evidence", "",
              "Each example is the actual extracted input text selected by highest predicted-class probability, with sentence ID as a deterministic tie-breaker. High probability does not prove correctness. Absent categories are reported explicitly."]
    for period in result["periods"]:
        row = period["summary"]
        lines += ["", f"### {row['company']} FY{row['fiscal_year']} {row['fiscal_quarter']}", "",
                  f"Policy: `{row['selection_policy']}`; within-company comparison profile: `{row['comparison_profile']}`.", ""]
        for doc in period["documents"]:
            extra = doc.get("extra", {})
            lines.append(f"- [{doc['document_id']}]({doc['source_url']}) — {doc['document_type']}; published {doc['publication_date']}; quarter ended {extra.get('quarter_end', 'not supplied')}; TXT SHA-256 `{doc['source_sha256']}`.")
            lines.append(f"  Scope: {extra.get('content_scope', 'not supplied')}")
            lines.append(f"  Extraction: {extra.get('extraction_method', 'not supplied')}")
            if extra.get("extraction_limitations"):
                lines.append(f"  Limitations: {extra['extraction_limitations']}")
        for label in LABELS:
            examples = period["evidence"][label]
            if not examples:
                lines += ["", f"{label.capitalize()}: no classified sentences in this category."]
            for example in examples:
                lines += ["", f"{label.capitalize()} — probability {example['probability']:.3f}; [{example['sentence_id']}]({example['source_url']}):", "",
                          "> " + example["text"].replace("\n", "\n> ")]
    lines += ["", "## Limitations", ""] + [f"- {note}" for note in result["limitations"]]
    return "\n".join(lines) + "\n"


def export_analysis(result: dict[str, Any], output_dir: str | Path) -> None:
    """Create the four requested artifacts without overwriting existing outputs."""
    directory = Path(output_dir)
    names = ("quarterly_sentiment_summary.csv", "quarterly_comparison.csv",
             "sentiment_analysis.json", "financial_sentiment_report.md")
    if any((directory / name).exists() for name in names):
        raise ValueError("Analysis output already exists; choose a new output directory.")
    report = render_report(result)
    serialized = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    directory.mkdir(parents=True, exist_ok=True)
    for name, rows, fields in (
        (names[0], [p["summary"] for p in result["periods"]], SUMMARY_COLUMNS),
        (names[1], result["comparisons"], COMPARISON_COLUMNS),
    ):
        with (directory / name).open("x", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    with (directory / names[2]).open("x", encoding="utf-8") as output:
        output.write(serialized)
    with (directory / names[3]).open("x", encoding="utf-8") as output:
        output.write(report)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Phase 3 predictions JSONL.")
    parser.add_argument("--source-input", type=Path, help="Complete Phase 2 JSONL for verified coverage.")
    parser.add_argument("--skipped-input", type=Path, help="Phase 3 skipped JSONL; otherwise discover <input>.skipped.jsonl.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        skipped_path = args.skipped_input
        if skipped_path is None and Path(str(args.input) + ".skipped.jsonl").exists():
            skipped_path = Path(str(args.input) + ".skipped.jsonl")
        predictions = read_jsonl(args.input, allow_empty=True)
        skipped = read_jsonl(skipped_path, allow_empty=True) if skipped_path is not None else []
        sources = read_jsonl(args.source_input) if args.source_input is not None else None
        result = analyze_predictions(predictions, skipped_records=skipped, source_records=sources)
        input_paths = [path for path in (args.input, args.source_input, skipped_path) if path is not None]
        result["input_files"] = [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in input_paths]
        export_analysis(result, args.output_dir)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    coverage = result["coverage"]
    print(f"Companies: {coverage['companies']} | Company-periods: {coverage['company_periods']} | Documents: {coverage['documents']} | Classified: {coverage['classified_sentences']} | Known skips: {coverage['skipped_sentences']} | Consecutive comparisons: {len(result['comparisons'])}\nCoverage verified: {coverage['coverage_verified']} | Output: {args.output_dir}")
    if not coverage["coverage_verified"]:
        print("Warning: source totals/coverage are unknown; supply --source-input and all skipped records.")


if __name__ == "__main__":
    main()
