"""Offline native-Space configuration, runtime mode and repeatable packaging."""

from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest
import yaml

from deploy.huggingface.build_bundle import build_bundle, verify_bundle, RUNTIME_FILES
from src import app, document_processing


def test_bundle_repeatability_inventory_and_metadata(tmp_path):
    first = build_bundle(tmp_path / "first")
    second = build_bundle(tmp_path / "second")
    names = sorted(p.relative_to(first) for p in first.rglob("*") if p.is_file())
    assert all((first / p).read_bytes() == (second / p).read_bytes() for p in names)
    assert "src/evaluation.py" not in {str(p) for p in names}
    assert not any(str(p).startswith(("tests/", "data/", "outputs/", ".venv")) for p in names)
    card = (first / "README.md").read_text().split("---", 2)[1]
    config = yaml.safe_load(card)
    assert config["sdk"] == "gradio" and config["app_file"] == "app.py"
    assert config["sdk_version"] == "6.31.0" and config["python_version"] == "3.14.7"
    assert config["suggested_hardware"] == "cpu-basic"
    requirements = (first / "requirements.txt").read_text()
    assert f"gradio=={config['sdk_version']}" in requirements
    assert "torch==2.14.1+cpu" in requirements and "https://download.pytorch.org/whl/cpu" in requirements
    assert "pytest" not in requirements and "scikit-learn" not in requirements and "nvidia" not in requirements.lower()
    for name in RUNTIME_FILES:
        assert (first / name).read_bytes() == Path(name).read_bytes()
    assert verify_bundle(first)["files"]


def test_bundle_refuses_overwrite_and_detects_missing_source(tmp_path):
    directory = build_bundle(tmp_path / "bundle")
    with pytest.raises(ValueError, match="never overwritten"):
        build_bundle(directory)
    (directory / "src/inference.py").unlink()
    with pytest.raises(ValueError, match="inventory"):
        verify_bundle(directory)
    with pytest.raises(ValueError, match="separate"):
        build_bundle(Path("src"))


@pytest.mark.parametrize("allow,offline,local_only", [(None,None,True),("1",None,False),("1","1",True),("1","TRUE",True),("0",None,True)])
def test_model_download_opt_in_and_offline_precedence(monkeypatch, allow, offline, local_only):
    for key, value in (("FINSENTIMENT_ALLOW_MODEL_DOWNLOAD", allow),("HF_HUB_OFFLINE", offline)):
        if value is None:monkeypatch.delenv(key, raising=False)
        else:monkeypatch.setenv(key, value)
    monkeypatch.setattr(document_processing, "_ANALYZER", None)
    calls = []
    model = object()
    def load(**options):
        calls.append(options)
        return model
    monkeypatch.setattr(document_processing, "FinBertAnalyzer", load)
    assert document_processing.get_analyzer() is document_processing.get_analyzer() is model
    assert calls == [{"local_files_only":local_only}]


def test_model_error_is_clear_and_does_not_contain_private_path(monkeypatch):
    monkeypatch.setattr(document_processing,"_ANALYZER",None)
    def fail(**kwargs):raise OSError("private /home/user/cache path")
    monkeypatch.setattr(document_processing,"FinBertAnalyzer",fail)
    with pytest.raises(ValueError,match="FinBERT could not load") as error:
        document_processing.get_analyzer()
    assert "/home" not in str(error.value)


@pytest.mark.parametrize("hosted,host",[(False,"127.0.0.1"),(True,"0.0.0.0")])
def test_shared_launch_binding_and_limits(monkeypatch,hosted,host):
    monkeypatch.delenv("GRADIO_SERVER_NAME",raising=False)
    calls=[]
    def create(**options):
        calls.append(options)
        return SimpleNamespace(launch=lambda **kwargs: kwargs)
    monkeypatch.setattr(app,"create_app",create)
    options=app.launch_app(hosted=hosted)
    assert calls==[{"hosted":hosted}]
    assert options["server_name"]==host and options["share"] is False
    assert options["max_file_size"]==5*1024*1024


def test_hosted_notice_local_ui_unchanged():
    def notices(demo):return [c["props"].get("value","") for c in demo.config["components"] if c["type"]=="markdown"]
    assert any("Hosted demo privacy" in text for text in notices(app.create_app(hosted=True)))
    assert not any("Hosted demo privacy" in text for text in notices(app.create_app()))


def test_entrypoint_is_import_safe_and_launches_once(monkeypatch,tmp_path):
    bundle=build_bundle(tmp_path/"bundle")
    namespace=runpy.run_path(str(bundle/"app.py"),run_name="space_entry_test")
    assert "main" in namespace
    for key in ("FINSENTIMENT_ALLOW_MODEL_DOWNLOAD","OMP_NUM_THREADS","MKL_NUM_THREADS","TOKENIZERS_PARALLELISM","GRADIO_ANALYTICS_ENABLED","HF_HUB_DISABLE_TELEMETRY"):
        monkeypatch.delenv(key,raising=False)
    calls=[]
    monkeypatch.setattr(document_processing,"get_analyzer",lambda:calls.append("load"))
    monkeypatch.setattr(app,"launch_app",lambda **kwargs:calls.append(kwargs))
    namespace["main"]()
    assert calls==["load",{"hosted":True}]
    import os
    assert os.environ["FINSENTIMENT_ALLOW_MODEL_DOWNLOAD"]=="1"
    monkeypatch.setenv("FINSENTIMENT_ALLOW_MODEL_DOWNLOAD","0")
    namespace["configure_runtime"]()
    assert os.environ["FINSENTIMENT_ALLOW_MODEL_DOWNLOAD"]=="0"
