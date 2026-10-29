# Financial Communication Sentiment Analyzer

A Python backend NLP project exploring how financial-domain NLP can help analyze
and compare corporate communication sentiment across reporting periods.
**Phases 1 and 2 are implemented:** local FinBERT sentence inference, plus local
English TXT ingestion and sentence preprocessing with provenance. This is a portfolio/research project, not a trading,
stock-prediction or financial recommendation system.

## Roadmap

1. **Project setup and FinBERT baseline — implemented.**
2. **Financial document ingestion and preprocessing — implemented.**
3. Batch sentiment inference and structured prediction export — planned.
4. Reporting-period sentiment comparison — planned.
5. Dataset-based evaluation and robustness analysis — planned.
6. Final research reports and portfolio documentation — planned.

## Structure

```text
src/__init__.py          Package marker
src/inference.py         Reusable CPU classifier and five-sentence terminal demo
src/model_config.py      Shared pinned model/tokenizer settings
src/ingestion.py         JSON metadata validation and UTF-8 TXT loading
src/preprocessing.py     Sentence records, token-length checks and JSONL CLI
tests/test_inference.py  Offline unit tests and an opt-in model integration test
tests/test_ingestion.py  Ingestion and metadata validation tests
tests/test_preprocessing.py  Segmentation, provenance, length policy and CLI tests
tests/fixtures/          Original fictional earnings text and matching metadata
docs/data_plan.md        Verified source shortlist and future evaluation input plan
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

Phase 1 baseline verification, before Phase 2: default tests **9 passed, 1 skipped**;
integration enabled **10 passed**. The five-sentence demo also succeeded with
`HF_HUB_OFFLINE=1` and `--local-files-only`. Its observed labels, in source order,
were positive, positive, neutral, positive and negative. In particular, the
operating-expenses sentence received a positive score of about 0.899; this is an
observed model output, not an endorsed interpretation or a correctness assertion.

## Phase 2: local document preprocessing

The pipeline is `metadata JSON + local UTF-8 TXT → validated document → normalized
sentences → token-length checks → JSONL`. It performs no sentiment prediction.
No new dependencies, NLP data downloads or model weights are needed. The tokenizer
is loaded from the existing Phase 1 cache with network access disabled.

Prepare a narrative `.txt` file and a JSON object with these required fields:

```json
{
  "document_id": "synthetic_q1_2025",
  "company": "Example Analytics Ltd.",
  "ticker": "SYNTHETIC",
  "fiscal_year": 2025,
  "fiscal_quarter": "Q1",
  "document_type": "earnings_release",
  "publication_date": "2025-05-01",
  "source_url": "https://example.com/synthetic-earnings",
  "language": "en",
  "file_path": "synthetic_earnings.txt",
  "extra": {"synthetic": true}
}
```

This example is fictional. `file_path` is resolved **relative to the metadata
JSON's directory**; absolute paths also work. Fiscal year and quarter must be
supplied from the company's fiscal calendar, separately from the valid ISO
publication date. Required strings must be nonblank; IDs use letters, digits,
hyphens and underscores. Quarters are Q1–Q4, the year is an integer and language
must be `en`. No missing financial facts are inferred. Additional provenance
belongs in the optional `extra` object; unknown top-level fields are rejected to
catch typos. Read/encoding/metadata errors produce a CLI message and exit code 2.

Try the supplied original synthetic fixture:

```bash
source .venv-finsentiment/bin/activate
python -m src.preprocessing \
  --metadata tests/fixtures/synthetic_earnings.json \
  --text tests/fixtures/synthetic_earnings.txt \
  --output outputs/synthetic_sentences.jsonl
```

`--text` is optional; when provided it must match metadata `file_path`. Output
directories are created as needed. Existing output files are never overwritten;
choose a new filename for another run. Store real inputs in ignored `data/raw/`
and generated records in `outputs/` or `data/processed/`, both ignored by Git.

### Output schema and provenance

Each JSONL line is a JSON object containing **all input metadata**, plus:

| Field | Meaning |
| --- | --- |
| `sentence_id` | Document ID plus a one-based, at-least-four-digit suffix, e.g. `synthetic_q1_2025_0001` |
| `sentence_index` | Zero-based index in emitted order |
| `text` | Normalized sentence for later inference |
| `original_text` | Exact corresponding substring from the original local TXT |
| `source_char_start`, `source_char_end` | Zero-based Python character offsets, end exclusive, in decoded TXT; not byte offsets |
| `source_sha256` | SHA-256 of original TXT bytes, including its newline encoding/BOM |
| `token_count` | Full WordPiece length, including special tokens, without truncation |
| `exceeds_token_limit`, `max_tokens` | Whether the sentence exceeds the 512-token model input limit |
| `tokenizer_model`, `tokenizer_revision` | Pinned FinBERT tokenizer identity |
| `long_sentence_policy`, `preprocessing_version` | Processing policy/version for reproducibility |

JSONL preserves Unicode and nested metadata without CSV escaping or flattening,
and supports later sequential reading. IDs and order are deterministic for the
same document, settings and code version; editing the document may change IDs.
Use unique document IDs across the collection. No sentiment fields are added.

### Cleaning, segmentation and long sentences

Whitespace and line breaks collapse to single spaces. Formatting-only separator
lines (`---`, `===`, `___`, `***`), unnumbered bullet prefixes and BOM artifacts
are removed; zero-width spaces become whitespace to preserve word boundaries.
Financial symbols, decimals, percentages, dates,
case, negation and other words remain intact. There is no stemming or stopword
removal. Punctuation-only and numeric-only fragments are discarded; short
alphabetic fragments and an unterminated final sentence are retained.

A small standard-library rule set handles `.?!`, closing quotes, decimals, common
English titles/month abbreviations and dotted acronyms such as `U.S.`. Company
suffixes before lowercase continuations are protected. It needs no Punkt data or
additional NLP framework. Unpunctuated bullets, headings, unusual abbreviations,
tables and acronyms at sentence ends can be ambiguous: review sentence boundaries
before research use and prepare narrative sections rather than financial tables.

**Default long-sentence policy is `error`:** a sentence over 512 tokens stops the
document before output is opened. To preserve the entire sentence for review:

```bash
python -m src.preprocessing \
  --metadata tests/fixtures/synthetic_earnings.json \
  --output outputs/synthetic_flagged.jsonl \
  --long-sentence-policy keep
```

`keep` sets `exceeds_token_limit=true` on overlong records and prints their count.
It does not make them model-ready. No text is silently truncated or automatically
split into chunks that could lose sentiment context. Phase 3 must handle flagged
records explicitly before calling Phase 1 inference, whose standalone interface
still truncates overlong inputs. For programmatic reuse:

```python
from src.ingestion import load_document
from src.preprocessing import load_token_counter, preprocess_document

document = load_document("tests/fixtures/synthetic_earnings.json")
counter = load_token_counter()  # Cache-only tokenizer; no model weights.
records = preprocess_document(document, token_counter=counter)
# Future Phase 3 can pass each eligible record['text'] to the inference layer.
```

The shared model settings were extracted into `src/model_config.py`; Phase 1
inference behavior is unchanged. Phase 2 unit tests use an injected token counter;
the opt-in integration suite also verifies the actual cached tokenizer and long
sentence policy. Run both phases with the test commands above.

Combined verification on 2026-10-29: **63 passed, 2 skipped** by default;
**65 passed** with `RUN_FINBERT_INTEGRATION=1 HF_HUB_OFFLINE=1` (including the
unchanged Phase 1 tests). The offline CLI emitted five synthetic records; their
metadata, IDs, source spans and token counts were checked against the input.

### Financial sources and future evaluation

See [the data plan](docs/data_plan.md) for verified Microsoft and Apple FY2025
Q1–Q4 releases, exact source links, fiscal/publication dates, access terms,
provenance recommendations and future labelled-data input planning. No third-party
financial text or evaluation data was downloaded into the repository. Publicly
readable documents are not assumed redistributable. The supplied FiQA URL now
redirects to a gated aspect-based dataset; its actual schema and mapping must be
inspected in Phase 5. No evaluation adapter or metrics are implemented.

## Current limitations

- Local English TXT only, with manually prepared metadata; no PDF/HTML parsing,
  automatic language verification, scraping, batch inference, quarterly comparisons
  or dataset evaluation. JSONL preprocessing export is implemented.
- Segmentation is heuristic and requires review. Long sentences are rejected or
  explicitly flagged by preprocessing; standalone Phase 1 inference still truncates.
- Softmax scores are model outputs, not calibrated certainty or financial advice.
  Demonstration results do not establish accuracy or real-world reliability.
- Repeatability is checked within numerical tolerance on this environment;
  different hardware or dependency versions can produce small differences.
- Financial PhraseBank was used to fine-tune FinBERT, so a future evaluation on
  that dataset must not be described as independent unseen-data evaluation.
  FiQA also needs dataset inspection and careful label mapping before evaluation.

Next: Phase 3 can read these sentence records, handle flagged long inputs, run CPU
FinBERT inference in batches and preserve metadata alongside all probabilities.
That work is not implemented here.
