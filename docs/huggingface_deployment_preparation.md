# Native Gradio Space preparation — no deployment performed

Prepared 2026-10-29, after Phase 6 was committed and pushed separately as
`6515e872ced66250398abb5ddb7448cc4b8a542c`. This preparation is separate from Phase 6. No
Space, Hub remote, login, upload, hardware change or deployment was performed.
No new virtual environment, dependency installation or model download was used.

## Official runtime requirements reviewed

The [Gradio Space guide](https://huggingface.co/docs/hub/spaces-sdks-gradio) and
[configuration reference](https://huggingface.co/docs/hub/spaces-config-reference)
use README YAML for `sdk: gradio`, `sdk_version`, `python_version` and `app_file`.
`app_file` is relative to the Space repository root. This template selects native
Gradio **6.31.0**, Python **3.14.7**, and root **app.py**, matching the tested local
Python/package versions. It does not use Docker or Docker-only `app_port` settings.
`suggested_hardware: cpu-basic` is a suggestion, not an allocation or permission.

According to the current [Spaces overview](https://huggingface.co/docs/hub/spaces-overview),
creating a Gradio/Docker compute Space requires PRO for personal accounts or a
qualifying organization plan. CPU Basic has no hourly hardware charge, but that
does not waive account eligibility. A documented free ZeroGPU exception is not
the CPU-only configuration prepared here. Your account, available hardware and
any subscription/payment prerequisites have **not** been checked or assumed.
Verify them manually before deciding to publish.

[Dependency documentation](https://huggingface.co/docs/hub/spaces-dependencies)
installs root `requirements.txt` in the Space's build environment. The SDK version
and the requirements Gradio pin agree. Optional packages.txt/pre-requirements.txt
are unnecessary for this prepared wheel-based runtime.

## Source templates and generated bundle

```text
deploy/huggingface/
  README.md           Space card and YAML metadata
  app.py              Thin entry point, no UI duplication
  requirements.txt    Six pinned runtime packages and CPU wheel index
  build_bundle.py     Standard-library-only explicit copy/verification script

outputs/huggingface-space/       Generated, ignored, self-contained artifact
  README.md
  app.py
  requirements.txt
  bundle_manifest.json
  src/
    __init__.py
    app.py
    document_processing.py
    pdf_extraction.py
    reporting.py
    inference.py
    model_config.py
    preprocessing.py
    ingestion.py
    analysis.py
    records.py
    assets/download.svg
```

Generate from the repository root using any installed Python 3.9+ interpreter:

```bash
python deploy/huggingface/build_bundle.py
```

The output must be new or empty. Choose another path with `--output` if the default
already exists. It never overwrites an existing bundle. The allowlist excludes
all datasets, tests, financial releases, reports, caches, environments, model
weights, Git metadata and offline evaluation modules. The full `analysis.py`
module is retained because its shared sentiment-distribution helper imports the
record/metadata contracts; this does not require scikit-learn or any benchmark data.

The bundle contains byte-identical copies of the current runtime source. Edit
`src/` or these templates, then regenerate into a fresh directory; do not edit the
generated copies independently. The manifest records file SHA-256 values, with
no timestamps or absolute paths. Two builds from the same inputs are byte-identical.
The script verifies inventory, hashes and local `src.*` import closure. Build a
fresh clean bundle for later upload; never upload the whole GitHub checkout.

## Entry point and model behaviour

Root `app.py` is import-safe: importing it neither starts a server nor downloads
weights. Its `main()` sets conservative environment defaults before importing
Torch/Hub libraries, initializes the model once and calls the shared
`src.app.launch_app(hosted=True)` exactly once. The UI is imported from the same
source as local development. Hosted binding defaults to `0.0.0.0`, with Gradio's
normal port selection (7860 by default) and `share=False`. There is no public tunnel.
For a local preview only, `GRADIO_SERVER_NAME=127.0.0.1` overrides hosted binding.

Local `python -m src.app` remains loopback-only and cache-only by default.
`FINSENTIMENT_ALLOW_MODEL_DOWNLOAD=1` permits the existing analyzer to retrieve the
pinned public checkpoint. The prepared Space entry sets that default; an explicit
`0` is respected. `HF_HUB_OFFLINE=1` always takes precedence. Model/tokenizer
revision remains `0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031`, with safetensors and CPU
inference unchanged. Public Hub metadata verified that this revision and its
`model.safetensors`, configuration and tokenizer files exist; no weights were fetched.

Startup may take time to download/initialize the checkpoint in a fresh authorized
Space. A single locked process-local analyzer is reused across requests; weights
are not downloaded per request. Failures produce a clear model/cache/network
message without exposing local paths to the UI. A failed initialization can be
retried; offline smoke testing does not prove a fresh online cold start.

## Minimal runtime requirements and CPU resources

| Package | Pin | Runtime reason |
| --- | --- | --- |
| Gradio | 6.31.0 | Existing UI and HTTP server; same as SDK YAML |
| torch | 2.14.1+cpu | Existing CPU inference |
| transformers | 5.20.2 | Existing pinned tokenizer and classifier |
| numpy | 2.5.4 | Gradio/Torch runtime array dependency |
| pandas | 3.1.0 | Gradio Dataframe processing; pinned to the tested version |
| PyMuPDF | 1.26.7 | Existing PDF extraction and PDF reports, without layout/OCR models |

No pytest, scikit-learn, dataset package, torchvision, torchaudio or GPU package
is required. Transitive dependencies are resolved by these releases: this is not
a full hash-locked environment. NumPy requires Python >=3.12, so the default Space
Python 3.10 would not match these pins; the template explicitly sets 3.14.7.
PyPI metadata was checked for the pinned releases, and the official CPU index
contains `torch-2.14.1+cpu-cp314-cp314-manylinux_2_28_x86_64.whl`. Its small metadata
sidecar declares no NVIDIA dependencies. No wheel payload was downloaded.

[PyTorch's official installation guide](https://pytorch.org/get-started/locally/)
provides separate CPU wheels. The prepared requirements use the official CPU
extra index and the explicit `+cpu` version; a standard CUDA-enabled PyPI torch
wheel cannot satisfy that version. Do not replace it with an unqualified torch
pin or reinstall Torch on this laptop. A clean native-Space build must still
confirm wheel/Python availability and all resolved transitive dependencies.

The documented default [CPU Basic resources](https://huggingface.co/docs/hub/spaces-overview)
are 2 vCPUs, 16 GB RAM and 50 GB ephemeral disk. FinBERT has roughly 110 million
parameters, about 440 MB for float32 parameter storage alone; Python, Torch,
tokenizer, activations, PDF parsing and dataframes add memory. This is a sizing
estimate, not measured hosted RSS or a capacity guarantee. CPU inference batch
size stays 8; OMP/MKL defaults are 2 threads. Existing queue concurrency stays 1,
with at most 8 queued analyses and the existing model lock. No algorithm, model,
segmentation, CSV or PDF content was changed for resource preparation.

Limits remain 5 MiB, 30 PDF pages, 500,000 extracted characters and 2,000 analysis
units. These bound ordinary workload but are not a sandbox for pathological PDFs.
No GPU, OCR model, persistent document store or database is added.

## Storage, privacy and environment configuration

[Space disk is ephemeral](https://huggingface.co/docs/hub/spaces-storage): cache and
runtime files can disappear when stopped/restarted. This preparation uses no
bucket, volume or persistent storage. Do not assume a `/data` mount or the older
persistent-storage tiers. Re-fetching the model after cache loss is expected.

The hosted UI and Space card explicitly say that files are processed on the hosted
server, not on the visitor's own device. Use public/non-confidential demonstration
materials only. Temporary reports have per-request directories and per-session
state, fixed generated basenames and sanitized document display names. A different
request cannot accidentally reuse an earlier request's state. Failed analysis
clears displayed predictions/downloads; no raw text is printed by the application.
Static assets/files use Gradio's normal file-access controls, without broad allowed
server paths. No authentication or confidential-processing guarantee is offered.

Request-owned generated reports are cleaned on replacement or state expiry (one hour).
Gradio removes cached uploads/downloads older than an hour on a ten-minute cycle;
cleanup can be delayed, and prior known cache URLs are not promised immediate
revocation. This is temporary retention, not guaranteed secure erasure. All local
raw data, uploads and generated reports remain excluded from Git.

| Variable | Prepared/default behaviour |
| --- | --- |
| FINSENTIMENT_ALLOW_MODEL_DOWNLOAD | Space entry defaults 1; normal local app defaults 0 |
| HF_HUB_OFFLINE | Set 1 for existing-cache-only verification; unset for a fresh hosted download |
| OMP_NUM_THREADS / MKL_NUM_THREADS | Space entry defaults 2 |
| TOKENIZERS_PARALLELISM | Defaults false in Space entry |
| GRADIO_SERVER_NAME | Hosted default 0.0.0.0; use 127.0.0.1 only for local preview |
| GRADIO_SERVER_PORT | Optional local test port; native SDK normally uses 7860 |
| GRADIO_TEMP_DIR | Optional writable ephemeral cache path; no persistence required |
| HF_HOME / HF_HUB_CACHE | Optional writable model cache paths; default library locations work |
| GRADIO_ANALYTICS_ENABLED / HF_HUB_DISABLE_TELEMETRY | False / 1 by default in Space entry |

[Hub environment variables](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables)
are read at import time; the entry sets defaults first. The public pinned model
needs no hardcoded token. No account authentication was performed. If a later
private model requires credentials, use a Space Secret rather than source code
or public Variables, as described in [Space settings](https://huggingface.co/docs/hub/spaces-overview#managing-secrets-and-environment-variables).
Do not paste tokens into README, requirements, manifests or Git.

Review intended-use licensing before publication: the pinned model card declares
[CC-BY-NC-SA-4.0](https://huggingface.co/api/models/ProsusAI/finbert/revision/0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031),
and PyMuPDF is [AGPL/commercial](https://pypi.org/project/pymupdf/1.26.7/).
No new project/model/software licence was invented in YAML. Public access is not
a substitute for reviewing those terms for the eventual demonstration.

## Local verification without a fresh installation

Reuse the existing venv and cached model. From the repository root, after building:

```bash
source .venv-finsentiment/bin/activate
HF_HUB_OFFLINE=1 FINSENTIMENT_ALLOW_MODEL_DOWNLOAD=0 \
  GRADIO_SERVER_NAME=127.0.0.1 GRADIO_SERVER_PORT=7862 \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
  python outputs/huggingface-space/app.py
python -m pytest -q
RUN_FINBERT_INTEGRATION=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q
```

Use a free local port. `PYTHONDONTWRITEBYTECODE=1` keeps the generated directory
clean for inventory checks. No local absolute paths are encoded in the entry or
source bundle. The offline smoke test also launches via the actual root entry
from a bundle working directory, proves imports come from that bundle, uploads
an original TXT and text PDF, validates CSV/PDF, and checks failed-upload clearing.
The small test fixtures are read from outside the bundle and are not copied into it.

Actual verification on 2026-10-29 used Python 3.14.7 and the existing environment:

- Two final bundles were byte-identical: 16 files including the manifest, 92,647
  bytes, with no data, weights, caches, tests or local absolute paths.
- Import resolution came from the generated bundle, not the repository's `src/`.
  The real root entry launched once, reused one CPU analyzer, and respected
  explicit offline/cache-only settings. The local HTTP server closed normally.
- `positive_earnings.txt`: 10 classified segments, net sentiment 1.0000;
  `sample_earnings.pdf`: 7 classified segments, net sentiment 0.1429.
- Both actual downloaded CSV files were byte-identical to Phase 6 exports.
  Both downloaded PDF reports opened successfully and contained the source name
  and matching net sentiment. Failed empty uploads cleared prior downloads.
- The default suite passed **285 tests, 13 skipped** (4.95 seconds). The complete
  offline integration suite passed **298 tests** (23.73 seconds), with 25 upstream
  Gradio/Pandas deprecation warnings and no failures.

The local smoke record is `outputs/space_bundle_verification.json` (ignored).
Packaging/metadata, first-download opt-in and offline precedence, import-safe
startup, single launch, model-load errors, binding/limits and the hosted notice
are also covered by `tests/test_huggingface_preparation.py`.

A second full environment or fresh dependency installation was deliberately not
attempted because local free space is only a few hundred MiB. Remote build,
online cold start, account permissions, bandwidth, hardware allocation and actual
hosted latency/memory remain unverified. Do not describe this as deployed or
fresh-install validated.

## Steps you will complete manually later

1. Confirm account/organization eligibility for a native Gradio CPU compute Space,
   available hardware, subscription/billing conditions and intended-use licences.
2. Review and later commit this preparation separately if satisfied. Regenerate a
   fresh bundle from the reviewed sources; inspect its manifest and contents.
3. When you explicitly decide to deploy, create a Space manually with the **Gradio**
   SDK and suitable visibility/CPU hardware. This step has not been performed.
4. Put the generated bundle's **contents** at the Space repository root, not nested
   inside `deploy/huggingface/` or `outputs/`. Do not upload the GitHub repository,
   datasets, test artifacts, caches or weights.
5. Keep README YAML and runtime pins aligned. Permit first-start network access;
   do not set HF_HUB_OFFLINE=1 on an uncached Space. No token is normally required.
6. Inspect the native build/runtime logs for Python/wheel compatibility, one CPU
   model load, successful launch on the runtime port and no accidental CUDA wheels.
7. Test an original public TXT/PDF, metadata, current CSV/PDF downloads, long inputs,
   failure clearing and the privacy notice. Measure hosted resource use/latency.
8. Keep uploads non-confidential; verify temporary-file cleanup and the actual
   account/hardware costs. Pause or remove the Space yourself when appropriate.

No Hugging Face remote was added and no Space was created, configured or uploaded.
