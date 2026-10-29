"""Reports contain real processing output and use request-owned files."""

import csv
import pymupdf
from pathlib import Path

from src.document_processing import analyze_document
from src.reporting import AnalysisDownloads, CSV_COLUMNS, markdown_report
from tests.test_document_processing import MockAnalyzer


def test_csv_report_order_metadata_and_cleanup(tmp_path, pdf_text):
    path = tmp_path / "source.txt"
    path.write_text("Revenue grew. The board met. A loss occurred.\n\n123 456")
    result = analyze_document(path, {"company": "Example, Inc", "fiscal_year": "2025"}, analyzer=MockAnalyzer())
    first = AnalysisDownloads(result, parent=tmp_path)
    second = AnalysisDownloads(result, parent=tmp_path)
    assert first.csv.parent != second.csv.parent
    with first.csv.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == CSV_COLUMNS
        rows = list(reader)
    assert len(rows) == result["classified_units"] == 3
    assert [r["text"] for r in rows] == [r["text"] for r in result["predictions"]]
    assert rows[0]["company"] == "Example, Inc" and rows[0]["ticker"] == ""
    for row in rows:
        assert sum(float(row[f"prob_{label}"]) for label in ("positive", "negative", "neutral")) == 1
    with first.skipped.open(newline="") as stream:
        omitted = list(csv.DictReader(stream))
    assert len(omitted) == 1 and omitted[0]["text"] == "123 456"
    report = pdf_text(first.report)
    assert "Net sentiment: 0.0000" in report and "classified segments: 3" in report
    assert "not classification accuracy" in report and "investment" in report
    assert first.csv.read_bytes() == second.csv.read_bytes()
    first.cleanup()
    assert not first.csv.exists() and second.csv.exists()
    second.cleanup()


def test_absent_categories_and_literal_markdown(tmp_path):
    path = tmp_path / "<script>.txt"
    path.write_text("The board met.")
    result = analyze_document(path, {"company": "[click](file:///secret)<script>"}, analyzer=MockAnalyzer())
    report = markdown_report(result)
    assert "No segments classified as positive" in report and "No segments classified as negative" in report
    assert "<script>" not in report and "[click](" not in report
    assert result["summary"]["neutral_pct"] == 100


def test_pdf_unicode_summary_source_statements_and_no_markdown_download(tmp_path, pdf_text):
    path = tmp_path / "financial_€_report.txt"
    path.write_text("Revenue grew to €12.5 million — up 8.2%. Management’s outlook did not change. Cash includes £4.0 million and ¥500 million.", encoding="utf-8")
    result = analyze_document(path, {"company": "Example & Co", "ticker": "EX", "fiscal_year": 2025, "fiscal_quarter": "Q2"}, analyzer=MockAnalyzer())
    downloads = AnalysisDownloads(result, parent=tmp_path)
    try:
        assert downloads.report.name == "financial_document_report.pdf"
        assert downloads.report.read_bytes().startswith(b"%PDF-")
        assert not list(downloads.report.parent.glob("*.md"))
        text = pdf_text(downloads.report)
        assert "Financial Document Analysis Report" in text and path.name in text
        assert all(value in text for value in ("Example & Co", "Ticker: EX", "Fiscal Year: 2025", "Fiscal Quarter: Q2", "ProsusAI/finbert"))
        assert all(value in text for value in ("€12.5", "£4.0", "¥500", "—", "Management’s"))
        assert f"Total analyzed segments: {result['classified_units']}" in text
        for label in ("positive", "negative", "neutral"):
            assert f"{label.title()} {result['summary'][label + '_count']} {result['summary'][label + '_pct']:.2f}%" in text
        assert f"Net sentiment: {result['summary']['net_sentiment']:.4f}" in text
        assert "high-confidence predictions" in text and "not investment advice" in text
        with pymupdf.open(downloads.report) as document:
            assert not document.is_repaired and not document.needs_pass
            assert all(abs(page.rect.width - 595) < 1 and abs(page.rect.height - 842) < 1 for page in document)
            assert all(f"Page {i + 1} of {len(document)}" in page.get_text() for i, page in enumerate(document))
    finally:
        downloads.cleanup()


def test_multi_page_pdf_wraps_all_long_statements_without_clipping(tmp_path, pdf_text):
    from types import SimpleNamespace
    class FormattingAnalyzer(MockAnalyzer):
        """Fictional fixed outputs solely for a report-layout contract test."""
        model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=512))
        def predict_batch(self, texts):
            return [{"sentence": text, "label": "positive", "probabilities": {"positive": .8, "negative": .1, "neutral": .1}} for text in texts]
    source = ("Revenue grew despite expenses of €12.5 million and 7.5% inflation; " * 100).strip() + " Final evidence marker."
    path = tmp_path / "long_original.txt"
    path.write_text(source)
    result = analyze_document(path, analyzer=FormattingAnalyzer())
    downloads = AnalysisDownloads(result, parent=tmp_path)
    try:
        assert "Final evidence marker" in pdf_text(downloads.report)
        with pymupdf.open(downloads.report) as document:
            assert len(document) >= 3
            combined = " ".join(" ".join(page.get_text().split()) for page in document)
            assert combined.count("€12.5") == 100
            for page in document:
                for block in page.get_text("dict")["blocks"]:
                    for line in block.get("lines", []):
                        for span in line["spans"]:
                            x0, y0, x1, y1 = span["bbox"]
                            assert x0 >= 47 and x1 <= page.rect.width - 47
                            assert y0 >= 42 and y1 <= page.rect.height - 14
        assert downloads.report.stat().st_size < 500_000
    finally:
        downloads.cleanup()
