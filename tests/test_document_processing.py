"""Upload adapter contracts; fictional data and deterministic mock predictions."""

from types import SimpleNamespace
from pathlib import Path
import os

import pytest

from src.document_processing import analyze_document, token_safe_segments, upload_metadata
from src.ingestion import FinancialDocument
from src.preprocessing import preprocess_document


class MockTokenizer:
    def encode(self, text, **kwargs):
        assert kwargs.get("truncation") is False
        return [0] * (len(text.split()) + 2)


class MockAnalyzer:
    tokenizer = MockTokenizer()
    device = SimpleNamespace(type="cpu")
    model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=12))

    def predict_batch(self, texts):
        assert all(len(self.tokenizer.encode(t, truncation=False)) <= 12 for t in texts)
        output = []
        for text in texts:
            label = "negative" if "loss" in text.lower() else "positive" if "grew" in text.lower() else "neutral"
            output.append({"sentence": text, "label": label,
                           "probabilities": {name: .8 if name == label else .1 for name in ("positive", "negative", "neutral")}})
        return output


@pytest.fixture
def upload(tmp_path):
    path = tmp_path / "financial.txt"
    path.write_text("Revenue grew 12.5%. We did not report a loss. The board met.", encoding="utf-8")
    return path


def test_txt_optional_metadata_order_mapping_and_aggregation(upload):
    result = analyze_document(upload, analyzer=MockAnalyzer())
    assert result["metadata"] == dict(company=None, ticker=None, fiscal_year=None, fiscal_quarter=None)
    assert result["classified_units"] == result["total_units"] == 3
    assert result["skipped_units"] == 0 and result["summary"]["net_sentiment"] == 0
    assert all(result["summary"][f"{label}_pct"] == pytest.approx(100 / 3) for label in ("positive", "negative", "neutral"))
    assert [r["sentence_index"] for r in result["predictions"]] == [0, 1, 2]
    assert analyze_document(upload, analyzer=MockAnalyzer()) == result
    text = upload.read_text()
    for row in result["predictions"]:
        assert text[row["source_char_start"]:row["source_char_end"]] == row["original_text"]
        assert sum(row[f"prob_{label}"] for label in ("positive", "negative", "neutral")) == pytest.approx(1)


@pytest.mark.parametrize("metadata", [{"fiscal_year": True}, {"fiscal_year": 0}, {"fiscal_year": 2025.5},
                                     {"fiscal_year": "abc"}, {"fiscal_quarter": "Q5"}, {"company": 123},
                                     {"source_url": "invented"}, {"ticker": "bad\nname"}])
def test_invalid_optional_metadata(metadata):
    with pytest.raises(ValueError):
        upload_metadata(metadata)


def test_supplied_metadata_preserved(upload):
    facts = {"company": "Example Inc", "ticker": "EX", "fiscal_year": "2025", "fiscal_quarter": "Q2"}
    result = analyze_document(upload, facts, analyzer=MockAnalyzer())
    assert result["metadata"]["fiscal_year"] == 2025
    assert result["predictions"][0]["company"] == "Example Inc"
    assert "source_url" not in result["predictions"][0]


@pytest.mark.parametrize("filename,content,error", [("x.txt", b"", "empty"), ("x.txt", b" \n\t", "empty"),
    ("x.txt", b"\xff\xfe", "UTF-8"), ("x.txt", b"Revenue\x00grew", "control"), ("x.docx", b"text", "Upload")])
def test_invalid_document(tmp_path, filename, content, error):
    path = tmp_path / filename
    path.write_bytes(content)
    with pytest.raises(ValueError, match=error):
        analyze_document(path, analyzer=MockAnalyzer())


def test_size_unit_and_character_limits(upload, monkeypatch):
    monkeypatch.setattr("src.document_processing.MAX_UPLOAD_BYTES", 10)
    with pytest.raises(ValueError, match="upload limit"):
        analyze_document(upload, analyzer=MockAnalyzer())
    monkeypatch.setattr("src.document_processing.MAX_UPLOAD_BYTES", 1024)
    monkeypatch.setattr("src.document_processing.MAX_EXTRACTED_CHARS", 10)
    with pytest.raises(ValueError, match="character"):
        analyze_document(upload, analyzer=MockAnalyzer())
    monkeypatch.setattr("src.document_processing.MAX_EXTRACTED_CHARS", 1024)
    monkeypatch.setattr("src.document_processing.MAX_SEGMENTS", 2)
    with pytest.raises(ValueError, match="segment limit"):
        analyze_document(upload, analyzer=MockAnalyzer())


@pytest.mark.parametrize("words", [9, 10, 11, 27, 80])
def test_token_boundaries_lossless_financial_unicode_and_negation(tmp_path, words):
    tokens = (["Revenue", "€12.5m", "increased", "12.5%", "but", "did", "not", "offset", "café", "losses"] * 9)[:words]
    text = "  " + "\t ".join(tokens) + "\n"
    path = tmp_path / "long.txt"
    path.write_text(text, encoding="utf-8")
    result = analyze_document(path, analyzer=MockAnalyzer())
    rows = result["predictions"]
    assert " ".join(row["text"] for row in rows) == " ".join(tokens)
    assert all(row["token_count"] <= 12 for row in rows)
    assert len({row["sentence_id"] for row in rows}) == len(rows)
    assert result["original_text_units"] == 1
    assert result["split_original_units"] == (1 if words > 10 else 0)
    for row in rows:
        assert text[row["source_char_start"]:row["source_char_end"]] == row["original_text"]
    assert len(result["predictions"]) == (words + 9) // 10


def test_paragraph_order_and_numeric_exclusions(tmp_path):
    path = tmp_path / "paragraphs.txt"
    path.write_text("Revenue grew\n\n123 456 78%\n\nWe did not make a loss\n\nThe board met.")
    result = analyze_document(path, analyzer=MockAnalyzer())
    assert [r["text"] for r in result["predictions"]] == ["Revenue grew", "We did not make a loss", "The board met."]
    assert result["skipped_units"] == 1 and result["total_units"] == 4
    assert result["skipped"][0]["text"] == "123 456 78%"


def test_cpu_batch_and_prediction_contract(upload):
    analyzer = MockAnalyzer()
    analyzer.device = SimpleNamespace(type="cuda")
    with pytest.raises(ValueError, match="CPU"):
        analyze_document(upload, analyzer=analyzer)
    for size in (0, True, 33):
        with pytest.raises(ValueError, match="Batch size"):
            analyze_document(upload, analyzer=MockAnalyzer(), batch_size=size)
    analyzer = MockAnalyzer()
    analyzer.predict_batch = lambda values: []
    with pytest.raises(ValueError, match="count"):
        analyze_document(upload, analyzer=analyzer)
    analyzer.predict_batch = lambda values: [{"sentence": text, "label": "positive", "probabilities": {"positive": 3, "negative": 0, "neutral": 0}} for text in values]
    with pytest.raises(ValueError, match="probabilities"):
        analyze_document(upload, analyzer=analyzer)


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("RUN_FINBERT_INTEGRATION") != "1", reason="Requires cached CPU model.")
def test_real_tokenizer_long_upload_and_model(cached_analyzer, tmp_path):
    text = ("Revenue was not lower despite expenses of €12.50 million and 7.5% inflation; " * 55).strip()
    path = tmp_path / "long.txt"
    path.write_text(text)
    result = analyze_document(path, analyzer=cached_analyzer)
    assert len(result["predictions"]) > 1
    assert " ".join(row["text"] for row in result["predictions"]) == text
    assert all(row["token_count"] <= 512 and row["inference_device"] == "cpu" for row in result["predictions"])


def test_negation_at_chunk_boundary_stays_with_following_word(tmp_path):
    path = tmp_path / "negation.txt"
    words = ["Revenue"] + ["remained"] * 8 + ["not", "lower"]
    path.write_text(" ".join(words))
    result = analyze_document(path, analyzer=MockAnalyzer())
    assert result["predictions"][-1]["text"] == "not lower"
    assert " ".join(r["text"] for r in result["predictions"]) == " ".join(words)


def test_invalid_input_does_not_load_model(tmp_path, monkeypatch):
    def unexpected():
        raise AssertionError("Invalid input must fail before model loading")
    monkeypatch.setattr("src.document_processing.get_analyzer", unexpected)
    path = tmp_path / "empty.txt"
    path.write_text("")
    with pytest.raises(ValueError, match="empty"):
        analyze_document(path)


def test_single_lazy_model_across_concurrent_requests(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from src import document_processing
    calls = []
    def load(**kwargs):
        assert kwargs == {"local_files_only": True}
        calls.append(True)
        return MockAnalyzer()
    monkeypatch.setattr(document_processing, "_ANALYZER", None)
    monkeypatch.setattr(document_processing, "FinBertAnalyzer", load)
    with ThreadPoolExecutor(max_workers=3) as executor:
        models = list(executor.map(lambda _: document_processing.get_analyzer(), range(6)))
    assert len(calls) == 1 and all(model is models[0] for model in models)


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("RUN_FINBERT_INTEGRATION") != "1", reason="Requires cached CPU model.")
@pytest.mark.parametrize("name", ["positive", "negative", "mixed"])
def test_original_financial_acceptance_fixtures(cached_analyzer, name, pdf_text):
    """Fixture names describe content, never expected model percentages or gold labels."""
    import csv
    from src.reporting import AnalysisDownloads
    path = Path(__file__).parent / "fixtures" / "phase6" / f"{name}.txt"
    result = analyze_document(path, analyzer=cached_analyzer)
    rows = result["predictions"]
    summary = result["summary"]
    labels = ("positive", "negative", "neutral")
    n = len(rows)
    assert n == result["classified_units"] > 0
    assert sum(summary[f"{label}_count"] for label in labels) == n
    assert sum(summary[f"{label}_pct"] for label in labels) == pytest.approx(100)
    assert summary["net_sentiment"] == (summary["positive_count"] - summary["negative_count"]) / n
    source = path.read_text()
    for row in rows:
        assert source[row["source_char_start"]:row["source_char_end"]] == row["original_text"]
        scores = [row[f"prob_{label}"] for label in labels]
        assert all(0 <= score <= 1 for score in scores) and sum(scores) == pytest.approx(1)
    for examples in result["evidence"].values():
        assert all(row in rows for row in examples)
    downloads = AnalysisDownloads(result)
    try:
        with downloads.csv.open(newline="", encoding="utf-8") as stream:
            exported = list(csv.DictReader(stream))
        assert len(exported) == n
        assert [row["text"] for row in exported] == [row["text"] for row in rows]
        assert f"Net sentiment: {summary['net_sentiment']:.4f}" in pdf_text(downloads.report)
        with downloads.skipped.open(newline="", encoding="utf-8") as stream:
            assert len(list(csv.DictReader(stream))) == result["skipped_units"]
    finally:
        downloads.cleanup()
