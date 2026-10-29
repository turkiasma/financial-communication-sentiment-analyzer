# Financial Communication Sentiment Analyzer

A Python backend NLP project exploring how financial-domain NLP can help analyze
and compare corporate communication sentiment across reporting periods.
**Only Phase 1 is implemented:** local, pretrained FinBERT inference on individual
English financial sentences. This is a portfolio/research project, not a trading,
stock-prediction or financial recommendation system.

## Roadmap

1. **Project setup and FinBERT baseline — implemented.**
2. Financial document ingestion and preprocessing — planned.
3. Batch sentiment inference and structured prediction export — planned.
4. Reporting-period sentiment comparison — planned.
5. Dataset-based evaluation and robustness analysis — planned.
6. Final research reports and portfolio documentation — planned.

## Structure

```text
src/__init__.py          Package marker
src/inference.py         Reusable CPU classifier and five-sentence terminal demo
tests/test_inference.py  Offline unit tests and an opt-in model integration test
pytest.ini              Test discovery and integration marker
requirements.txt        Direct dependencies pinned to the verified environment
.gitignore              Environment, cache and generated-file exclusions
```

## Environment

Run commands from the repository root on Linux:

```bash
source .venv-finsentiment/bin/activate
python --version
python -c "import torch, transformers; print(torch.__version__, transformers.__version__); print('CUDA build:', torch.version.cuda)"
```

The existing environment was verified with Python 3.14.7, PyTorch 2.14.1+cpu,
Transformers 5.20.2, NumPy 2.5.4 and pytest 9.1.1. It also contains pandas,
scikit-learn and NLTK; Phase 1 does not need them. No packages need to be installed
in the existing environment. The separate `.venv` directory is not used.

For a **new environment only**, install the pinned dependencies after activating
it (do not recreate or overwrite an existing environment):

```bash
python -m pip install -r requirements.txt
```

The requirements explicitly select the `+cpu` PyTorch build from the official
CPU wheel index. Do not replace it with `transformers[torch]` or a generic GPU
installation. These direct dependency pins describe the tested environment,
not a complete transitive lockfile; wheel availability varies by Python/platform.

## Run sample inference

```bash
python -m src.inference
```

The demo loads the model once and prints each of five sentences, its predicted
label, and positive, negative and neutral probabilities. The sentences are
demonstrations, not labelled evaluation data; no particular prediction is required.

For reuse in Python:

```python
from src.inference import FinBertAnalyzer

analyzer = FinBertAnalyzer()  # Reuse this instance for subsequent sentences.
result = analyzer.predict("The company reported strong revenue growth.")
print(result["label"])
print(result["probabilities"])
```

The result has `sentence`, `label`, and `probabilities` keys. All three lowercase
labels are obtained from the model's `id2label` configuration. Softmax converts
logits into probabilities. The model and input tensors are explicitly placed on
CPU, with evaluation mode and gradients disabled. Blank strings raise `ValueError`;
non-string inputs raise `TypeError`.

## Model and cache

We use [`ProsusAI/finbert`](https://huggingface.co/ProsusAI/finbert), pinned to
revision `0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031` for a repeatable baseline.
See the [official implementation](https://github.com/ProsusAI/finBERT) and
[Transformers cache documentation](https://huggingface.co/docs/transformers/installation).
No training, fine-tuning or paid API is involved.

The first run downloads the tokenizer, configuration and a single safetensors
checkpoint (approximately 438 MB) into the Hugging Face cache, normally
`~/.cache/huggingface/hub`. It does not download TensorFlow/Flax weights or datasets.
Later runs reuse these files. Hugging Face honors `HF_HOME`/`HF_HUB_CACHE` if a
different cache location is needed; keep it outside tracked project files.
To require the cached snapshot without network access:

```bash
python -m src.inference --local-files-only
```

An uncached offline run fails with a model-loading error; run once with network
access and enough storage first. Ordinary tests never download a model.
CPU inference may be slow on some machines. If CPU thread contention causes
slowdowns, try `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m src.inference`.

The model card identifies the weights license as **CC BY-NC-SA 4.0**. Retain
attribution and review its linked license/usage terms before redistribution or
commercial use. The model is credited to ProsusAI; see Araci (2019),
*FinBERT: Financial Sentiment Analysis with Pre-trained Language Models*.

## Tests

```bash
python -m pytest -q
# After the first successful model run; cached files required:
RUN_FINBERT_INTEGRATION=1 python -m pytest -q
```

Default tests mock model loading while exercising real PyTorch softmax, label
mapping (with reordered indices), output structure, probability range/sum,
input validation, CPU placement, evaluation mode, disabled gradients and
repeatability. They also verify that repeated predictions do not reload the model.
The opt-in integration test loads the real cached model exactly once and checks
its outputs, CPU parameters and repeated predictions with absolute tolerance
`1e-6`. A missing cache causes that opt-in test to fail, not silently download.

Verified in the environment above: default tests **9 passed, 1 skipped**;
integration enabled **10 passed**. The five-sentence demo also succeeded with
`HF_HUB_OFFLINE=1` and `--local-files-only`. Its observed labels, in source order,
were positive, positive, neutral, positive and negative. In particular, the
operating-expenses sentence received a positive score of about 0.899; this is an
observed model output, not an endorsed interpretation or a correctness assertion.

## Current limitations

- Single English sentences only; no ingestion, sentence segmentation, batching,
  exports, quarterly comparisons or dataset evaluation yet.
- Inputs exceeding the model's 512-token context (including special tokens) are
  truncated. Later preprocessing must handle long documents explicitly.
- Softmax scores are model outputs, not calibrated certainty or financial advice.
  Demonstration results do not establish accuracy or real-world reliability.
- Repeatability is checked within numerical tolerance on this environment;
  different hardware or dependency versions can produce small differences.
- Financial PhraseBank was used to fine-tune FinBERT, so a future evaluation on
  that dataset must not be described as independent unseen-data evaluation.
  FiQA also needs dataset inspection and careful label mapping before evaluation.

Next: Phase 2 should ingest a small set of financial communications, clean and
segment their text, and preserve company, reporting period and source metadata.
