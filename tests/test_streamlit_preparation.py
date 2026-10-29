"""Streamlit adapter contracts and UI wiring without installing Streamlit."""

from contextlib import nullcontext
import csv
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
import tomllib
from types import ModuleType, SimpleNamespace

import pymupdf
import pytest

from src import document_processing, streamlit_support as support
from src.reporting import AnalysisDownloads

ENTRY = Path("deploy/streamlit/app.py").resolve()
FIXTURES = Path("tests/fixtures/ui_samples")


class Tokenizer:
    def encode(self, text, **kwargs):
        assert kwargs["truncation"] is False
        return [0] * (len(text.split()) + 2)


class Analyzer:
    tokenizer = Tokenizer()
    device = SimpleNamespace(type="cpu")
    model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=12))

    def predict_batch(self, texts):
        assert all(len(text.split()) <= 10 for text in texts)
        labels = ["positive" if "grew" in t else "negative" if "loss" in t else "neutral" for t in texts]
        return [{"sentence": text, "label": label,
                 "probabilities": {name: .8 if name == label else .1 for name in ("positive", "negative", "neutral")}}
                for text, label in zip(texts, labels)]


def upload(name, content):
    stream = io.BytesIO(content)
    stream.name = name
    return stream


def pdf_text(content):
    with pymupdf.open(stream=content, filetype="pdf") as pdf:
        return " ".join(" ".join(page.get_text().split()) for page in pdf)


@pytest.mark.parametrize("name", ["positive_earnings.txt", "sample_earnings.pdf"])
def test_adapter_reuses_exact_csv_and_pdf_content(name):
    path = FIXTURES / name
    metadata = {"company": "Example Inc", "ticker": "EX", "fiscal_year": "2025", "fiscal_quarter": "Q2"}
    baseline = document_processing.analyze_document(path, metadata, analyzer=Analyzer())
    reports = AnalysisDownloads(baseline)
    try:
        actual = support.analyze_uploaded_bytes(name, path.read_bytes(), metadata, analyzer=Analyzer())
        assert actual["csv_bytes"] == reports.csv.read_bytes()
        assert pdf_text(actual["pdf_bytes"]) == pdf_text(reports.report.read_bytes())
        assert actual["summary"] == baseline["summary"]
        assert {row["Predicted sentiment"].lower() for row in actual["evidence"]} == {
            label for label, rows in baseline["evidence"].items() if rows}
    finally:
        reports.cleanup()


def test_session_isolation_consecutive_uploads_and_failed_analysis():
    first, second = {}, {}
    content = b"Revenue grew. We reported a loss. The board met."
    result = support.process_for_session(first, "report.txt", content, analyzer=Analyzer())
    assert second == {} and support.RESULT_KEY in first
    assert result["summary"]["net_sentiment"] == 0
    assert all(result["summary"][f"{label}_pct"] == pytest.approx(100/3) for label in ("positive", "negative", "neutral"))
    identity = support.upload_identity("report.txt", content, {})
    first[support.INPUT_KEY] = identity
    support.synchronize_input(first, identity)
    assert first[support.RESULT_KEY] is result
    support.synchronize_input(first, support.upload_identity("report.txt", content, {"ticker": "NEW"}))
    assert support.RESULT_KEY not in first
    support.process_for_session(first, "new.txt", b"We reported a loss.", analyzer=Analyzer())
    assert first[support.RESULT_KEY]["summary"]["net_sentiment"] == -1
    with pytest.raises(ValueError, match="UTF-8"):
        support.process_for_session(first, "invalid.txt", b"\xff", analyzer=Analyzer())
    assert support.RESULT_KEY not in first and second == {}


def test_upload_and_report_paths_are_unique_and_cleaned(monkeypatch):
    paths, reports = [], []
    original_analysis, original_reports = support.analyze_document, support.AnalysisDownloads
    def analyze(path, *args, **kwargs):
        paths.append(path)
        return original_analysis(path, *args, **kwargs)
    def export(result):
        owner = original_reports(result)
        reports.append(owner.csv.parent)
        return owner
    monkeypatch.setattr(support, "analyze_document", analyze)
    monkeypatch.setattr(support, "AnalysisDownloads", export)
    for _ in range(2):
        support.analyze_uploaded_bytes("../../report.txt", b"Revenue grew.", analyzer=Analyzer())
    with pytest.raises(ValueError):
        support.analyze_uploaded_bytes("broken.pdf", b"invalid", analyzer=Analyzer())
    assert paths[0].name == "report.txt"
    assert len(set(paths)) == 3 and len(set(reports)) == 2
    assert all(not path.parent.exists() for path in paths)
    assert all(not path.exists() for path in reports)


@pytest.mark.parametrize("name,content,match", [
    ("empty.txt", b"", "empty"), ("document.docx", b"data", "Upload"),
    ("file\x00.txt", b"data", "filename"), ("document.txt", b"\x00", "control"),
    ("huge.txt", b"x" * (5 * 1024 * 1024 + 1), "5 MiB"),
    ("broken.pdf", b"not a PDF", "Cannot open"),
])
def test_validation_limits_and_errors(name, content, match):
    with pytest.raises(ValueError, match=match):
        support.analyze_uploaded_bytes(name, content, analyzer=Analyzer())


def test_pdf_and_backend_limits_and_long_input(monkeypatch):
    with pytest.raises(ValueError, match="text|scanned|image"):
        support.analyze_uploaded_bytes("image.pdf", (FIXTURES/"image_only_earnings.pdf").read_bytes(), analyzer=Analyzer())
    with pymupdf.open() as pdf:
        for _ in range(31):
            pdf.new_page()
        with pytest.raises(ValueError, match="30"):
            support.analyze_uploaded_bytes("pages.pdf", pdf.tobytes(), analyzer=Analyzer())
    with pytest.raises(ValueError, match="year"):
        support.analyze_uploaded_bytes("a.txt", b"Revenue grew.", {"fiscal_year": "bad"}, analyzer=Analyzer())
    content = b"Revenue " + b"remained " * 70 + b"not lower"
    result = support.analyze_uploaded_bytes("long.txt", content, analyzer=Analyzer())
    rows = list(csv.DictReader(io.StringIO(result["csv_bytes"].decode())))
    assert " ".join(row["text"] for row in rows) == content.decode()
    assert result["split_original_units"] == 1 and all(int(row["token_count"]) <= 12 for row in rows)
    monkeypatch.setattr(document_processing, "MAX_EXTRACTED_CHARS", 10)
    with pytest.raises(ValueError, match="character"):
        support.analyze_uploaded_bytes("a.txt", b"Revenue grew.", analyzer=Analyzer())
    monkeypatch.setattr(document_processing, "MAX_EXTRACTED_CHARS", 500_000)
    monkeypatch.setattr(document_processing, "MAX_SEGMENTS", 1)
    with pytest.raises(ValueError, match="segment"):
        support.analyze_uploaded_bytes("a.txt", b"Revenue grew. We reported a loss.", analyzer=Analyzer())


def test_read_upload_checks_size_before_copy_and_hides_errors():
    with pytest.raises(ValueError, match="Choose"):
        support.read_upload(None)
    with pytest.raises(ValueError, match="empty"):
        support.read_upload(upload("empty.txt", b""))
    with pytest.raises(ValueError, match="5 MiB"):
        support.read_upload(upload("huge.txt", b"x" * (5 * 1024 * 1024 + 1)))
    assert support.user_error(ValueError("TXT documents must use UTF-8 encoding.")) == "TXT documents must use UTF-8 encoding."
    for error in (ValueError("failure /home/private/cache"), OSError("private secret"), RuntimeError("private traceback")):
        assert "private" not in support.user_error(error)


class FakeStreamlit(ModuleType):
    """Small widget test double; does not claim real Streamlit/browser coverage."""
    def __init__(self):
        super().__init__("streamlit")
        self.session_state, self.calls, self.values = {}, [], {}
        self.upload, self.pressed = None, False

    def cache_resource(self, **options):
        def decorate(function):
            cache = []
            def cached():
                if not cache:
                    cache.append(function())
                return cache[0]
            return cached
        return decorate

    def file_uploader(self, label, **kwargs):
        self.calls.append(("file_uploader", label, kwargs))
        return self.upload

    def text_input(self, label, **kwargs):
        self.calls.append(("text_input", label, kwargs))
        return self.values.get(kwargs["key"], "")

    def selectbox(self, label, choices, **kwargs):
        self.calls.append(("selectbox", label, kwargs))
        return self.values.get(kwargs["key"])

    def button(self, label, **kwargs):
        self.calls.append(("button", label, kwargs))
        return self.pressed and not kwargs["disabled"]

    def columns(self, count):
        return [self] * count

    def spinner(self, text):
        self.calls.append(("spinner", text, {}))
        return nullcontext()

    def __getattr__(self, name):
        if name not in {"set_page_config", "title", "caption", "error", "metric", "subheader", "warning", "download_button", "dataframe"}:
            raise AttributeError(name)
        def record(*args, **kwargs):
            self.calls.append((name, args, kwargs))
        return record


@pytest.fixture
def frontend(monkeypatch):
    fake = FakeStreamlit()
    monkeypatch.setitem(sys.modules, "streamlit", fake)
    spec = importlib.util.spec_from_file_location("streamlit_frontend_test", ENTRY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "configure_runtime", lambda: None)
    return module, fake


def test_initial_ui_import_safe_and_no_model_load(frontend, monkeypatch):
    module, fake = frontend
    def unexpected():
        raise AssertionError("No model should load before Analyze")
    monkeypatch.setattr(document_processing, "get_analyzer", unexpected)
    assert fake.calls == []
    module.main()
    assert ("title", ("Financial Document Analyzer",), {}) in fake.calls
    assert next(row for row in fake.calls if row[0] == "button")[2]["disabled"]
    uploader = next(row for row in fake.calls if row[0] == "file_uploader")[2]
    assert uploader["type"] == ["txt", "pdf"] and uploader["max_upload_size"] == 5
    assert not any(row[0] in {"metric", "download_button", "error"} for row in fake.calls)


def test_ui_metadata_metrics_downloads_reuse_and_stale_clearing(frontend, monkeypatch):
    module, fake = frontend
    loads = []
    def load():
        loads.append(True)
        return Analyzer()
    monkeypatch.setattr(document_processing, "get_analyzer", load)
    fake.upload = upload("report.txt", b"Revenue grew. We reported a loss. The board met.")
    fake.values = {"company": "Example Inc", "ticker": "EX", "fiscal_year": "2025", "fiscal_quarter": "Q2"}
    fake.pressed = True
    module.main()
    downloads = [row for row in fake.calls if row[0] == "download_button"]
    assert len(downloads) == 2
    assert [row[1][0] for row in downloads] == ["Download CSV", "Download PDF Report"]
    assert all(row[2]["icon"] == ":material/download:" and row[2]["on_click"] == "ignore" for row in downloads)
    csv_rows = list(csv.DictReader(io.StringIO(downloads[0][2]["data"].decode())))
    assert csv_rows[0]["company"] == "Example Inc" and csv_rows[0]["fiscal_year"] == "2025"
    assert "Net sentiment: 0.0000" in pdf_text(downloads[1][2]["data"])
    assert [row[1] for row in fake.calls if row[0] == "metric"] == [
        ("Positive", "33.33%"), ("Negative", "33.33%"), ("Neutral", "33.33%"), ("Net sentiment", "0.0000")]
    fake.calls.clear()
    fake.pressed = False
    module.main()
    assert loads == [True] and not any(row[0] == "spinner" for row in fake.calls)
    fake.values["ticker"] = "NEW"
    fake.calls.clear()
    module.main()
    assert not any(row[0] == "download_button" for row in fake.calls)
    fake.upload = upload("new.txt", b"We reported a loss.")
    fake.pressed = True
    module.main()
    assert loads == [True] and fake.session_state[support.RESULT_KEY]["summary"]["net_sentiment"] == -1
    fake.upload = upload("invalid.txt", b"\xff")
    fake.calls.clear()
    module.main()
    assert support.RESULT_KEY not in fake.session_state
    assert any(row[0] == "error" for row in fake.calls)
    assert not any(row[0] in {"download_button", "metric"} for row in fake.calls)


def test_ui_model_load_failure_and_upload_callback_clear_results(frontend, monkeypatch):
    module, fake = frontend
    fake.upload = upload("report.txt", b"Revenue grew.")
    fake.pressed = True
    fake.session_state[support.RESULT_KEY] = {"old": "report"}
    def fail():
        raise OSError("private /home/user/secret")
    monkeypatch.setattr(document_processing, "get_analyzer", fail)
    module.main()
    assert support.RESULT_KEY not in fake.session_state
    errors = [row for row in fake.calls if row[0] == "error"]
    assert errors and "private" not in str(errors)
    fake.session_state[support.RESULT_KEY] = {"old": "report"}
    uploader = next(row for row in fake.calls if row[0] == "file_uploader")
    uploader[2]["on_change"]()
    assert support.RESULT_KEY not in fake.session_state


def test_entry_import_does_not_import_backend_or_gradio():
    program = """
import runpy,sys,types
st=types.ModuleType('streamlit')
st.cache_resource=lambda **kw: lambda function: function
sys.modules['streamlit']=st
runpy.run_path('deploy/streamlit/app.py',run_name='import_test')
assert 'src.app' not in sys.modules and 'gradio' not in sys.modules
assert 'src.document_processing' not in sys.modules and 'torch' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", program], check=True, capture_output=True, text=True)


def test_configuration_runtime_defaults_and_requirements(monkeypatch):
    fake = FakeStreamlit()
    monkeypatch.setitem(sys.modules, "streamlit", fake)
    spec = importlib.util.spec_from_file_location("streamlit_runtime_config_test", ENTRY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    environment = {"HF_HUB_OFFLINE": "1", "FINSENTIMENT_ALLOW_MODEL_DOWNLOAD": "0"}
    monkeypatch.setattr(module.os, "environ", environment)
    module.configure_runtime()
    assert environment["FINSENTIMENT_ALLOW_MODEL_DOWNLOAD"] == "0" and environment["HF_HUB_OFFLINE"] == "1"
    assert environment["OMP_NUM_THREADS"] == environment["MKL_NUM_THREADS"] == "2"
    del environment["FINSENTIMENT_ALLOW_MODEL_DOWNLOAD"]
    module.configure_runtime()
    assert environment["FINSENTIMENT_ALLOW_MODEL_DOWNLOAD"] == "1"
    requirements = ENTRY.with_name("requirements.txt").read_text()
    assert "streamlit==1.66.1" in requirements and "torch==2.14.1+cpu" in requirements
    assert "--extra-index-url https://download.pytorch.org/whl/cpu" in requirements
    assert not any(line.startswith(("pytest", "scikit-learn", "gradio", "nvidia")) for line in requirements.splitlines())
    # No higher-priority dependency file masks the requirements beside the entrypoint.
    assert not any(ENTRY.with_name(name).exists() for name in ("uv.lock", "Pipfile", "environment.yml"))
    config = tomllib.loads(Path(".streamlit/config.toml").read_text())
    assert config["server"]["maxUploadSize"] == 5 and config["client"]["showErrorDetails"] is False


def test_optional_real_streamlit_apptest_without_loading_model(monkeypatch):
    pytest.importorskip("streamlit", minversion="1.66.1")
    from streamlit.testing.v1 import AppTest
    def unexpected():
        raise AssertionError("Widget edits must not initialize FinBERT")
    monkeypatch.setattr(document_processing, "get_analyzer", unexpected)
    monkeypatch.setenv("FINSENTIMENT_ALLOW_MODEL_DOWNLOAD", "0")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "TOKENIZERS_PARALLELISM", "HF_HUB_DISABLE_TELEMETRY"):
        monkeypatch.setenv(key, os.environ.get(key, "2" if key.endswith("NUM_THREADS") else "false"))
    app = AppTest.from_file(str(ENTRY)).run(timeout=15)
    assert not app.exception and app.title[0].value == "Financial Document Analyzer"
    app.text_input(key="company").set_value("Example Inc").run()
    assert not app.exception and not app.metric and app.button[0].disabled


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("RUN_FINBERT_INTEGRATION") != "1", reason="Requires cached CPU model.")
@pytest.mark.parametrize("name", ["positive_earnings.txt", "sample_earnings.pdf"])
def test_cached_finbert_streamlit_adapter_matches_existing_backend(cached_analyzer, name):
    assert cached_analyzer.device.type == "cpu"
    path = FIXTURES / name
    actual = support.process_for_session({}, name, path.read_bytes(), analyzer=cached_analyzer)
    baseline = document_processing.analyze_document(path, analyzer=cached_analyzer)
    reports = AnalysisDownloads(baseline)
    try:
        assert actual["csv_bytes"] == reports.csv.read_bytes()
        assert pdf_text(actual["pdf_bytes"]) == pdf_text(reports.report.read_bytes())
        assert actual["summary"] == baseline["summary"]
    finally:
        reports.cleanup()
