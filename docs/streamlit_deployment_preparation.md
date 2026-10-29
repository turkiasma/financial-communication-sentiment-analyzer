# Streamlit Community Cloud preparation — no deployment performed

Prepared after Hugging Face configuration was committed and pushed separately:
`89a7a9573b3dae0367f3be13b6a475d08cd4bdff`
(`chore: prepare native Hugging Face Gradio Space deployment`). Streamlit
preparation is tracked separately from deployment. No Cloud app, new repository, remote,
OAuth connection or deployment was created. FinSight was not accessed or changed.

## Architecture and entrypoint

```text
.streamlit/config.toml             Upload bound, no detailed errors/static serving
deploy/streamlit/
  app.py                          Small Streamlit UI and cached model wrapper
  requirements.txt                Runtime-only Python 3.12 package pins
  README.md                       Entrypoint and preparation instructions
src/streamlit_support.py           Session/upload/export adapter
tests/test_streamlit_preparation.py
```

Select `deploy/streamlit/app.py` as the future Community Cloud entrypoint in the
existing `turkiasma/financial-communication-sentiment-analyzer` repository. The
entry dynamically resolves the checkout root from its own location; no laptop
path or venv is required. It does not import or launch `src.app` or Gradio.
The complete repository provides `src/`; copying only `deploy/streamlit/` will
not provide the backend. No benchmark datasets are required at runtime.

The frontend contains one uploader, four optional metadata fields, one primary
Analyze button, three sentiment metrics, net sentiment, a small supporting-segment
table and two download buttons with native download icons. It shows up to three
segments for each of positive, negative and neutral, using backend evidence.
It adds no stock data, chatbot, dashboard, pages, APIs or alternative model.

The adapter reuses these actual modules:

- `document_processing.py`: validation, optional metadata, token-safe segments,
  prediction assembly, shared model singleton and serialized processing.
- `pdf_extraction.py`, `preprocessing.py`, `ingestion.py`: existing bounded
  text-PDF extraction and source-preserving preprocessing.
- `inference.py`, `model_config.py`: CPU-only FinBERT and pinned revision.
- `analysis.py`, `records.py`: shared sentiment distribution and metadata contracts.
- `reporting.py`: unchanged `AnalysisDownloads`, CSV columns and A4 PDF report.

There are no changes to these existing backend modules, the Gradio app, model
configuration, quarterly comparisons, evaluation code or benchmark evidence.
The new adapter handles upload streams and session presentation only.

## Cloud Python, dependencies and discovery

The official [deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
says Community Cloud supports released Python versions receiving security
updates, defaults to **3.12**, and offers a Python dropdown under Advanced
settings. Recommend selecting **3.12 explicitly**. The actual account dropdown
has not been inspected; this is documentation verification, not an account check.
There is no invented runtime.txt/python-version file controlling that selection.
Local Python remains 3.14.7; it has not been replaced or claimed to validate
execution under 3.12.

According to [dependency discovery](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies),
Cloud searches the entrypoint directory first, then the repository root. It
selects the first supported dependency file in priority order: uv.lock, Pipfile,
environment.yml, requirements.txt, pyproject.toml. There is no higher-priority
file beside this entry. Therefore `deploy/streamlit/requirements.txt` takes
precedence over the full root development requirements and the unrelated
Hugging Face requirements. Leave the root requirements unchanged.

Cloud may process requirements.txt with uv and fall back to pip. An extra CPU
index can cause uv index-selection conflicts for unrelated package names; if
the future build falls back, inspect the pip resolution and confirm CPU Torch.
No package installation or resolver run was performed locally. There is no
packages.txt requirement for the selected PyMuPDF wheel or an OCR system package.

| Runtime package | Pin | Python 3.12 compatibility evidence |
| --- | --- | --- |
| Streamlit | 1.66.1 | PyPI Requires-Python >=3.11; universal wheel |
| Torch | 2.14.1+cpu | Official Linux x86_64 cp312 CPU wheel and metadata |
| Transformers | 5.20.2 | PyPI >=3.10; universal wheel |
| NumPy | 2.5.4 | PyPI >=3.12; Linux cp312 wheel |
| pandas | 3.1.0 | PyPI >=3.11; Linux cp312 wheel |
| PyMuPDF | 1.26.7 | PyPI >=3.10; Linux cp310-abi3 wheel compatible with cp312 |

Sources: [Streamlit release metadata](https://pypi.org/pypi/streamlit/1.66.1/json),
[Transformers](https://pypi.org/pypi/transformers/5.20.2/json),
[NumPy](https://pypi.org/pypi/numpy/2.5.4/json),
[pandas](https://pypi.org/pypi/pandas/3.1.0/json),
[PyMuPDF](https://pypi.org/pypi/PyMuPDF/1.26.7/json), and the
[official CPU wheel index](https://download.pytorch.org/whl/cpu/torch/).
Streamlit's declared NumPy >=1.25,<3 and pandas >=1.5.3,<4 constraints accept
these pins. Transitive dependencies such as PyArrow/Altair are required by
Streamlit and resolved by its release; these six pins are not a full hash lock.

The requirements use `--extra-index-url https://download.pytorch.org/whl/cpu`
and `torch==2.14.1+cpu`, following the official
[CPU wheel installation approach](https://pytorch.org/get-started/locally/).
The `+cpu` suffix cannot be satisfied by the ordinary CUDA-enabled PyPI Torch
version. The exact official file
`torch-2.14.1+cpu-cp312-cp312-manylinux_2_28_x86_64.whl` was found, and its small
metadata sidecar declares no NVIDIA dependencies. No wheel payload was fetched.
Community Cloud's documented Debian 11 environment has a suitable glibc baseline
for manylinux_2_28. This checks package/platform metadata; it is not a clean
installation or a guarantee that every future Cloud build will resolve.

There is no Gradio, scikit-learn, pytest, dataset loader, torchvision, torchaudio
or GPU package in the Streamlit requirements. No root requirement was changed.

## Lazy model loading, cache and CPU use

The entry sets defaults before importing the backend: model download allowed,
OMP/MKL two threads, tokenizer parallelism disabled, Hub telemetry disabled.
Explicit environment settings win. Importing the entry creates no model and
does not launch a server. Only clicking Analyze calls `cached_analyzer()` inside
a loading spinner. It uses supported
[`st.cache_resource`](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_resource)
for a process-wide resource and delegates to the existing locked backend
`get_analyzer()`. Reruns and other users reuse that model; the backend lock
serializes preprocessing/inference, with batch size 8 and CPU placement intact.
No document/result is placed in st.cache_data or st.cache_resource.

`ProsusAI/finbert` revision remains
`0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031`, using safetensors. A fresh cloud
instance may download the public checkpoint on its first analysis and reuse
the normal Hub disk cache thereafter. A restart/cache loss may require another
download. No hardcoded token or new authentication is needed for this model.
Do not set offline mode on an uncached Cloud app. Local verification uses the
existing checkpoint and never downloads a second copy.

| Variable | Behaviour |
| --- | --- |
| FINSENTIMENT_ALLOW_MODEL_DOWNLOAD | Streamlit defaults 1; explicit 0 forces cache-only |
| HF_HUB_OFFLINE | Explicit 1 takes precedence and prevents Hub downloads |
| OMP_NUM_THREADS / MKL_NUM_THREADS | Default 2 each |
| TOKENIZERS_PARALLELISM | Default false |
| HF_HUB_DISABLE_TELEMETRY | Default 1 |
| HF_HOME / HF_HUB_CACHE | Optional writable cache locations; library defaults work |

Cloud configuration is not forced offline, and no model or data is loaded at
startup. Model initialization errors become controlled messages rather than raw
tracebacks in the interface. The original Gradio local workflow is unchanged.

## Memory and workload limits

FinBERT has approximately 110 million parameters (roughly 440 MB in float32),
before Torch/Transformers, Streamlit/PyArrow, tokenizer, model-loading buffers,
PDF parsing, activations and per-session downloads. One shared model and batch
size 8 reduce duplicate memory, but cannot guarantee a fit on Community Cloud.
No universal available-RAM quota is assumed. The official
[resource-limit guidance](https://docs.streamlit.io/knowledge-base/deploy/resource-limits)
recommends resource caching and profiling; the actual deployment allocation and
peak RSS must be measured. Concurrent sessions still consume memory for uploads
and downloads even though model work is serialized.

A separate offline local smoke process loaded exactly one cached CPU analyzer
and processed the original TXT/PDF fixtures. It measured **870.48 MiB peak RSS**
on this Linux/Python 3.14.7 laptop. The checkpoint has 109,484,547 parameters,
437,938,188 parameter bytes. This process did **not** import Streamlit or Gradio;
Streamlit/PyArrow and additional sessions would add overhead. This measurement
is a local backend reference, not hosted memory validation or a maximum bound.

The uploader and root `.streamlit/config.toml` cap uploads at 5 MiB; the adapter
checks the buffer before copying it and the backend enforces the same limit.
The unchanged limits are 30 PDF pages, 500,000 extracted characters and 2,000
analysis units. Long input is split using the original token-aware algorithm,
without changing probabilities, labels or source text. No OCR or second model
is added. These are ordinary workload bounds, not a sandbox for malformed PDFs.

If the unchanged model cannot fit the actual cloud allocation, document that
deployment limitation and choose an appropriate host later. Do not silently
quantize, substitute a model or alter inference to make it fit.

## Sessions, uploads, downloads and privacy

[`st.file_uploader`](https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader)
provides a memory-backed upload. The adapter strips directory components from
the client name and rejects invalid/control-character/oversized names. It writes
one unique temporary upload directory so the existing path-based backend can run.
The unchanged report exporter creates another unique per-analysis directory.
Both directories are removed on success/failure after CSV/PDF bytes are read.
No upload, source text, generated report or model enters Git, application logs,
a shared user-result cache or a persistent document store.

Only a lightweight input fingerprint, presentation summary/evidence and the two
download byte strings are associated with the current user's session state.
Full prediction records are not retained separately. Upload/metadata callbacks
clear results; the input fingerprint also catches changed content/names/metadata
on rerun. Analyze clears prior results before model loading or processing, and
any failure leaves no old metrics or downloads. Normal widget reruns do not infer.
Separate sessions never select reports through shared filenames or a global
document lookup. Downloads use fixed safe filenames and correct CSV/PDF MIME
types, with [`on_click="ignore"`](https://docs.streamlit.io/develop/api-reference/widgets/st.download_button)
to avoid an unnecessary rerun.

Removing results clears application session references. Streamlit owns its
upload buffers/media-download retention; disconnect, browser history, an already
issued download URL or a running interrupted session may outlive visible state.
Immediate URL revocation and secure erasure are not promised. The hosted UI
states that processing happens on the service's server and asks for **public,
non-confidential documents only**. There is no new authentication, database or
persistent storage. Cloud can override some configuration, including usage
statistics; see its [configuration limitations](https://docs.streamlit.io/deploy/streamlit-community-cloud/status#configuration).

Benchmark evidence is unchanged. Headline benchmark accuracy does not measure
uploaded earnings-report accuracy. Review the existing documented model/software
licences before a later publication; this preparation grants no new rights.

## Local verification and known gaps

The laptop uses Python 3.14.7 with the existing CPU model cache. Streamlit is not
installed. With only a few hundred MiB free, no fresh venv, dependency stack,
Torch reinstall, large wheel or model download was attempted. A real Streamlit
server/browser/AppTest run and a clean Python 3.12 installation remain unverified.

Run the available tests from the repository root:

```bash
source .venv-finsentiment/bin/activate
python -m pytest -q
RUN_FINBERT_INTEGRATION=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q
```

The new tests use deterministic mock predictions for most cases: TXT/PDF,
metadata, exact CSV/PDF content versus existing backend, sentiment aggregation,
long inputs, upload/PDF/text limits, unique paths/cleanup, session isolation,
consecutive uploads and failures. A small Streamlit test double verifies the real
UI function's widget wiring, lazy cache use, download data and invalidation; this
is not claimed as a real Streamlit renderer/browser run. A fresh-process import
test proves the entry does not load Torch/backend/Gradio or launch a server.
Optional AppTest coverage is skipped when Streamlit is absent. Optional integration
tests use the existing cached FinBERT for real TXT/PDF CSV/PDF parity.

Actual complete-suite outcomes:

- Default: **302 passed, 16 skipped** (6.54 seconds, commit review rerun).
- Offline integration enabled: **317 passed, 1 skipped**, with 25 upstream
  Gradio/Pandas deprecation warnings (28.35 seconds, commit review rerun).
- The integration run's sole remaining skip is the real Streamlit AppTest test.
  The default also skips the 13 existing model integrations and two new cached
  TXT/PDF parity cases. No failures occurred.

Actual local smoke results are recorded in the ignored
`outputs/streamlit_preparation/verification.json`: CPU inference and one shared
model, 10 TXT segments (net 1.0000), 7 PDF segments (net 0.1429), valid PDF reports
with source name/net sentiment, and both CSV downloads byte-identical to the
saved Phase 6 exports. A subsequent invalid UTF-8 upload cleared previous output.
Gradio was not imported by this process. No model weights were downloaded.

On a suitably provisioned environment where the runtime dependencies are already
installed, preview locally from the repository root:

```bash
HF_HUB_OFFLINE=1 FINSENTIMENT_ALLOW_MODEL_DOWNLOAD=0 \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -m streamlit run deploy/streamlit/app.py --server.address 127.0.0.1
```

For an uncached, authorized environment, omit offline mode and permit download.
Do not install this deployment stack into the low-disk laptop solely to preview it.

## Exact manual deployment steps remaining

1. Review the committed preparation and the remaining verification gaps. On a
   machine with sufficient storage, install the pinned runtime in Python 3.12,
   verify CPU Torch and run the real Streamlit/AppTest/TXT/PDF/download checks.
2. Confirm the reviewed preparation commit is available on the intended branch
   of the existing GitHub repository before selecting it for deployment.
3. Later, sign into Community Cloud through your GitHub account and verify
   repository administrative access, workspace/app availability and resource
   terms. Do not edit or replace your FinSight app.
4. Choose Create app → existing GitHub app; select
   `turkiasma/financial-communication-sentiment-analyzer`, the reviewed branch
   (normally main), and entrypoint **deploy/streamlit/app.py**.
5. In Advanced settings select **Python 3.12**. No token/secrets are normally
   needed for the public checkpoint. Leave HF_HUB_OFFLINE unset and ensure
   FINSENTIMENT_ALLOW_MODEL_DOWNLOAD is not explicitly disabled.
6. Only when you decide to publish, deploy. Inspect logs to confirm the selected
   entrypoint requirements, CPU Torch, compatible wheels and successful startup.
   An empty app should not initialize FinBERT until Analyze.
7. Test one public TXT and text PDF, optional metadata, sentiment metrics/evidence,
   both downloads, long inputs, consecutive uploads, failures and separate sessions.
   Verify CSV equivalence and PDF content. Measure cold-load/inference peak memory
   and latency; confirm the unchanged model fits before treating it as usable.
8. Check privacy/cleanup and document any Cloud resource/build/network limitation.
   If memory is insufficient, stop and choose hosting separately; do not alter
   the model or predictions without approval.

No deployment, new remote, new repository or FinSight change was performed.
