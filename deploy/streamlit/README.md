# Financial Document Analyzer — Streamlit preparation

Alternative frontend for the existing CPU FinBERT document analyzer. This
preparation is tracked separately from deployment. The working Gradio app
remains available through `python -m src.app`.

Future Community Cloud settings:

- Repository: `turkiasma/financial-communication-sentiment-analyzer`
- Branch: the reviewed branch containing this preparation, normally `main`
- Entrypoint: **`deploy/streamlit/app.py`**
- Advanced settings → Python: **3.12**
- Dependencies: **`deploy/streamlit/requirements.txt`**, beside the entrypoint;
  keep this file in place so Cloud chooses it before root development requirements.

This app imports the shared modules from the existing repository. Do not upload
this directory alone: it needs `src/`. Do not create another repository, replace
the Gradio app, or select an unrelated existing Streamlit app such as FinSight.

The six runtime pins are Streamlit 1.66.1, Torch 2.14.1+cpu, Transformers 5.20.2,
NumPy 2.5.4, pandas 3.1.0 and PyMuPDF 1.26.7. The official PyTorch CPU index and
explicit `+cpu` suffix exclude NVIDIA/CUDA Torch builds. Python 3.12 compatibility
was checked using release/wheel metadata, not a clean local installation.

When Streamlit is available in a suitably provisioned environment, run from the
repository root:

```bash
HF_HUB_OFFLINE=1 FINSENTIMENT_ALLOW_MODEL_DOWNLOAD=0 \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -m streamlit run deploy/streamlit/app.py --server.address 127.0.0.1
```

The current laptop does not have Streamlit installed. No environment, Torch,
Streamlit dependency stack or model weights were installed/downloaded for this
preparation. Backend exports and mocked widget behavior are tested; real
Streamlit/AppTest/browser behavior still requires verification.

The model loads lazily after Analyze and is shared through `st.cache_resource`
and the existing locked backend singleton. The cloud entry permits first-use
download of the pinned public checkpoint; explicit offline/cache-only settings
are respected. Do not enable offline mode on an uncached cloud instance.

Uploads are English UTF-8 TXT or text-based PDF only: 5 MiB, 30 PDF pages,
500,000 extracted characters and 2,000 analysis units. Scanned/image-only PDFs
need unsupported OCR. Percentages are predicted segment labels, not measured
upload accuracy. No benchmark accuracy is claimed for uploaded documents.

Files are processed on the hosted server. Use public, non-confidential documents
only. Reports belong to session state; user documents are never globally cached.
Unique temporary upload/report directories are removed after processing. Session
download bytes and Streamlit-managed files can outlive a visible result; this
does not promise immediate URL revocation or secure erasure.

FinBERT's approximately 440 MB of float32 parameters are only part of process
memory. Cloud capacity, cold-start network access and a clean Python 3.12 build
remain unverified. See the complete [preparation and manual deployment checklist](../../docs/streamlit_deployment_preparation.md)
before deciding to publish.
