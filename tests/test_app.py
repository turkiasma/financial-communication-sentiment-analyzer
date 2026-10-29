"""Gradio wiring, upload state isolation and clearing of failed analyses."""

from pathlib import Path
import os

import pytest

from src import app
from src.document_processing import analyze_document
from tests.test_document_processing import MockAnalyzer


def assert_cleared(result):
    assert result[:3] == ("", "", [])
    for button in result[3:5]:
        assert button["value"] is None and button["interactive"] is False


def test_app_import_and_components_wired():
    demo = app.create_app()
    config = demo.config
    assert config["title"] == "Financial Document Analyzer"
    event = next(d for d in config["dependencies"] if d["api_name"] == "analyze_document")
    assert len(event["inputs"]) == 6 and len(event["outputs"]) == 13
    assert any(c["type"] == "file" for c in config["components"])
    assert any(c["type"] == "state" for c in config["components"])
    primary = [c for c in config["components"] if c["props"].get("elem_id") in {"download-csv", "download-report"}]
    assert [c["props"]["label"] for c in primary] == ["Download CSV", "Download PDF Report"]
    assert primary[1]["props"]["min_width"] == 180
    for component in primary:
        assert component["type"] == "downloadbutton"
        assert component["props"]["size"] == "sm" and component["props"]["scale"] == 0
        assert component["props"]["icon"] and component["props"].get("value") is None
        assert component["props"]["interactive"] is False
    table = next(c for c in config["components"] if c["type"] == "dataframe")
    assert table["props"]["column_widths"] == ["110px", "90px", "520px"]
    # The upload is the only File component: report preview cards are gone.
    assert sum(c["type"] == "file" for c in config["components"]) == 1
    assert event["outputs"][3:5] == [c["id"] for c in primary]
    assert sum(c["type"] == "downloadbutton" for c in config["components"]) == 2
    assert not any(c["type"] == "accordion" for c in config["components"])


def test_callback_generates_current_downloads_and_clears_failure(tmp_path, monkeypatch):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("Revenue grew.")
    second.write_text("A loss occurred.")
    monkeypatch.setattr(app, "analyze_document", lambda path, metadata: analyze_document(path, metadata, analyzer=MockAnalyzer()))
    events = list(app.analyze_upload(str(first), "", "", "", None))
    assert_cleared(events[0])
    success = events[-1]
    assert "100.00%" in success[1] and "Revenue grew." in success[2][0]
    assert success[3]["interactive"] is True and success[4]["interactive"] is True
    owner = success[6]
    assert Path(success[3]["value"]).exists()
    other = list(app.analyze_upload(str(second), "", "", "", None, owner))[-1]
    assert not Path(success[3]["value"]).exists()
    assert other[6] is not owner and "A loss occurred." in other[2][0]
    assert "Revenue grew." not in str(other[:3])
    failure = list(app.analyze_upload(None, "", "", "", None, other[6]))[-1]
    assert_cleared(failure)
    assert failure[6] is None
    assert "failed" in failure[5] and not Path(other[3]["value"]).exists()


def test_callback_does_not_expose_server_paths(monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("Cannot read /secret/local/file.txt")
    monkeypatch.setattr(app, "analyze_document", fail)
    result = list(app.analyze_upload("uploaded.txt", "", "", "", None))[-1]
    assert "/secret" not in str(result) and result[3]["value"] is None


def test_scanned_pdf_error_remains_actionable(monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("Scanned/image-only PDFs need OCR, which is not supported.")
    monkeypatch.setattr(app, "analyze_document", fail)
    result = list(app.analyze_upload("scanned.pdf", "", "", "", None))[-1]
    assert "OCR" in result[5] and result[3]["value"] is None


def test_simple_results_hide_skip_counts_but_warn_on_excluded_content(tmp_path, monkeypatch, pdf_text):
    path = tmp_path / "partly_usable.txt"
    path.write_text("Revenue grew. The board met.\n\n123 456")
    monkeypatch.setattr(app, "analyze_document", lambda path, metadata: analyze_document(path, metadata, analyzer=MockAnalyzer()))
    result = list(app.analyze_upload(str(path), "", "", "", None))[-1]
    try:
        assert "Total analyzed segments:** 2" in result[0]
        assert "skipped" not in (result[0] + result[1]).lower()
        assert "original narrative units" not in result[0].lower()
        assert "Warning: Some content could not be analyzed" in result[5]
        assert all(row[0] in ("positive", "negative") for row in result[2])
        assert result[6].skipped.exists()
        assert "skipped units/blocks: 1" in pdf_text(result[6].report)
    finally:
        result[6].cleanup()


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("RUN_FINBERT_INTEGRATION") != "1", reason="Requires the cached CPU model and loopback HTTP.")
def test_ui_samples_through_actual_gradio_http(cached_analyzer, monkeypatch, tmp_path, pdf_text):
    import csv
    import socket
    from gradio_client import Client, handle_file
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(tmp_path / "gradio-cache"))
    monkeypatch.setattr("src.document_processing._ANALYZER", cached_analyzer)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    demo = app.create_app()
    try:
        _, url, _ = demo.launch(server_name="127.0.0.1", server_port=port, share=False,
                                prevent_thread_lock=True, quiet=True, max_file_size=5 * 1024 * 1024)
        client = Client(url, verbose=False)
        fixtures = Path(__file__).parent / "fixtures" / "ui_samples"
        previous_id = None
        for filename in ("positive_earnings.txt", "negative_earnings.txt", "mixed_earnings.txt", "sample_earnings.pdf"):
            response = client.predict(handle_file(str(fixtures / filename)), "", "", "", None,
                                      api_name="/analyze_document")
            assert response[5].startswith("Analysis complete.")
            csv_value = response[3].get("value") if isinstance(response[3], dict) else response[3]
            report_value = response[4].get("value") if isinstance(response[4], dict) else response[4]
            with Path(csv_value).open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            assert rows and f"Total analyzed segments:** {len(rows)}" in response[0]
            assert rows[0]["document_id"] != previous_id
            previous_id = rows[0]["document_id"]
            labels = ("positive", "negative", "neutral")
            counts = {label: sum(r["predicted_sentiment"] == label for r in rows) for label in labels}
            net = (counts["positive"] - counts["negative"]) / len(rows)
            assert f"Net sentiment: {net:.4f}" in response[0]
            assert f"Net sentiment: {net:.4f}" in pdf_text(report_value)
            for label in labels:
                assert f"{label.title()}: {100 * counts[label] / len(rows):.2f}%" in response[1]
            for row in rows:
                assert sum(float(row[f"prob_{label}"]) for label in labels) == pytest.approx(1)
            assert "skipped" not in response[0].lower()
        failed = client.predict(None, "", "", "", None, api_name="/analyze_document")
        assert failed[0] == "" and failed[1] == "" and "Choose" in failed[5]
        assert all((v.get("value") if isinstance(v, dict) else v) is None for v in failed[3:5])
    finally:
        demo.close()
