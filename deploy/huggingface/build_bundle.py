"""Copy an explicit runtime allowlist into a new, self-contained Space directory.

Standard library only. No network, authentication, installation or deployment.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil

REPOSITORY = Path(__file__).resolve().parents[2]
TEMPLATES = Path(__file__).resolve().parent
RUNTIME_FILES = (
    "src/__init__.py", "src/app.py", "src/document_processing.py", "src/pdf_extraction.py",
    "src/reporting.py", "src/inference.py", "src/model_config.py", "src/preprocessing.py",
    "src/ingestion.py", "src/analysis.py", "src/records.py", "src/assets/download.svg",
)
TEMPLATE_FILES = ("README.md", "app.py", "requirements.txt")


def verify_bundle(directory):
    directory = Path(directory)
    expected = set(RUNTIME_FILES) | set(TEMPLATE_FILES)
    manifest = json.loads((directory / "bundle_manifest.json").read_text())
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    if actual != expected | {"bundle_manifest.json"} or set(manifest["files"]) != expected:
        raise ValueError("Bundle file inventory differs from the runtime allowlist.")
    for name in sorted(expected):
        path = directory / name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != manifest["files"][name]:
            raise ValueError(f"Bundle content changed: {name}")
        if path.suffix == ".py":
            for node in ast.walk(ast.parse(path.read_text(), filename=name)):
                if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("src."):
                    dependency = node.module.replace(".", "/") + ".py"
                    if dependency not in expected:
                        raise ValueError(f"Missing runtime import {dependency} from {name}")
    return manifest


def build_bundle(output):
    output = Path(output).resolve()
    if (output == REPOSITORY or REPOSITORY.is_relative_to(output)
            or output.is_relative_to(REPOSITORY / "src") or output.is_relative_to(TEMPLATES)):
        raise ValueError("Choose a separate output directory outside src/ and deployment templates.")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output must be a new or empty directory; existing files are never overwritten.")
    sources = {name: REPOSITORY / name for name in RUNTIME_FILES}
    sources.update({name: TEMPLATES / name for name in TEMPLATE_FILES})
    for name, source in sources.items():
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"Missing or linked runtime source: {name}")
    output.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, source in sorted(sources.items()):
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        hashes[name] = hashlib.sha256(destination.read_bytes()).hexdigest()
    manifest = {"format": "finsentiment_native_gradio_bundle_v1", "files": hashes}
    (output / "bundle_manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    verify_bundle(output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPOSITORY / "outputs" / "huggingface-space")
    args = parser.parse_args(argv)
    try:
        output = build_bundle(args.output)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(f"Prepared local bundle: {output} (no upload or deployment performed)")


if __name__ == "__main__":
    main()
