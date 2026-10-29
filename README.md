# Financial Communication Sentiment Analyzer

A Python backend NLP project exploring how financial-domain NLP can help analyze
and compare corporate communication sentiment across reporting periods.
**Phases 1–5 are implemented:** local FinBERT inference, English TXT ingestion,
sentence preprocessing with provenance, CPU batch predictions and descriptive
quarterly sentiment comparison on eight real earnings-release excerpts, plus
labelled-data evaluation, a fixed baseline and descriptive robustness analysis.
Independent earnings annotation and a verified FiQA held-out test remain pending.
This is a portfolio/research project, not a trading,
stock-prediction or financial recommendation system.

## Roadmap

1. **Project setup and FinBERT baseline — implemented.**
2. **Financial document ingestion and preprocessing — implemented.**
3. **Batch sentiment inference and structured prediction export — implemented.**
4. **Reporting-period sentiment aggregation and comparison — implemented.**
5. **Dataset-based evaluation and robustness analysis — implemented, with access/annotation limitations below.**
6. Final research reports and portfolio documentation — planned.

## Structure

```text
src/__init__.py          Package marker
src/inference.py         Reusable CPU classifier and five-sentence terminal demo
src/batch_inference.py   Validated JSONL input, CPU batches and prediction JSONL CLI
src/analysis.py          Model-free quarterly summaries, comparisons and reports
src/records.py           Shared record validation and strict JSONL reader
src/evaluation_data.py   Pinned-file retrieval, labelled loaders/audits and blind annotation sample
src/evaluation.py        Three-class metrics, cache-only CPU evaluation, errors and subgroups
src/evaluation_baseline.py  Original frozen keyword baseline, no fitting/download
src/model_config.py      Shared pinned model/tokenizer settings
src/ingestion.py         JSON metadata validation and UTF-8 TXT loading
src/preprocessing.py     Sentence records, token-length checks and JSONL CLI
tests/test_inference.py  Offline unit tests and an opt-in model integration test
tests/test_ingestion.py  Ingestion and metadata validation tests
tests/test_preprocessing.py  Segmentation, provenance, length policy and CLI tests
tests/test_batch_inference.py  Batch validation, metadata, long inputs and integration tests
tests/test_analysis.py   Known arithmetic, coverage, fiscal ordering, exports and end-to-end checks
tests/test_evaluation*.py  Dataset, baseline, metrics, cache, subgroup and integration tests
tests/conftest.py        Single shared cached model for opt-in integration tests
tests/fixtures/          Original fictional earnings text and matching metadata
data/metadata/          Eight verified real-release metadata files; no release text
docs/data_plan.md        Verified source shortlist and future evaluation input plan
docs/evaluation_methodology.md  Fixed protocol, mappings, contamination/access limitations
docs/annotation_guide.md Human reference-label instructions; annotation remains pending
docs/evaluation_findings.md  Actual executed benchmark results and limitations
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
scikit-learn 1.9.1 and NLTK; Phase 5 reuses scikit-learn for metrics. No packages need to be installed
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

Within each paragraph, whitespace and line breaks collapse to single spaces. Formatting-only separator
lines (`---`, `===`, `___`, `***`), unnumbered bullet prefixes and BOM artifacts
are removed; zero-width spaces become whitespace to preserve word boundaries.
Financial symbols, decimals, percentages, dates,
case, negation and other words remain intact. There is no stemming or stopword
removal. Punctuation-only and numeric-only fragments are discarded; short
alphabetic fragments and an unterminated final sentence are retained.

Preprocessing **version 2**, introduced in Phase 4, honors blank-line paragraph
boundaries before segmentation. Unpunctuated narrative bullets remain separate
text segments; no punctuation is invented. Single line wraps within a paragraph
still collapse to whitespace. Records include
`segmentation_unit=sentence_or_paragraph_fragment`. Source offsets remain exact.
All eight Phase 4 releases were reprocessed with version 2; do not mix them with
old version-1 predictions. The term "sentences" in output column names includes
these documented paragraph fragments.

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
split into chunks that could lose sentiment context. Phase 3 skips flagged records
with reasons, while the standalone Phase 1 `predict` interface retains its original
truncation behavior. For programmatic reuse:

```python
from src.ingestion import load_document
from src.preprocessing import load_token_counter, preprocess_document

document = load_document("tests/fixtures/synthetic_earnings.json")
counter = load_token_counter()  # Cache-only tokenizer; no model weights.
records = preprocess_document(document, token_counter=counter)
# These records can be supplied to Phase 3 predict_records.
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
provenance recommendations and future labelled-data input planning. Phase 3 real
source text and predictions are local, ignored files; no third-party text is
tracked in Git. Publicly readable documents are not assumed redistributable. The supplied FiQA URL now
redirects to a gated aspect-based dataset; its actual schema and mapping must be
inspected before any gated-version evaluation. Phase 5 below documents the
accessible original-format alternative and its implemented evaluation framework.

## Phase 3: CPU batch sentiment inference

The pipeline is `Phase 2 JSONL → record validation → length checks → padded CPU
batches → prediction JSONL`. `FinBertAnalyzer.predict_batch` reuses the same model,
label mapping and softmax formatting as Phase 1. The model runs in evaluation mode
under `torch.inference_mode()`. It is loaded once per CLI invocation, and the
Python API can reuse one analyzer across documents. Default batch size is **8**.
Padding is to the longest sentence in each batch; truncation is disabled.

```bash
# Run Phase 2 first if this input does not exist; choose fresh output filenames.
python -m src.preprocessing \
  --metadata tests/fixtures/synthetic_earnings.json \
  --output outputs/synthetic_batch_input.jsonl
python -m src.batch_inference \
  --input outputs/synthetic_batch_input.jsonl \
  --output outputs/synthetic_batch_predictions.jsonl \
  --batch-size 8 \
  --local-files-only
```

Use `HF_HUB_OFFLINE=1` as well for an entirely offline run. No model download is
needed in the existing environment. Without `--local-files-only`, the CLI permits
normal Hugging Face cache/network behavior, still using the pinned model revision.

### Input validation and output schema

Input must contain the complete Phase 2 record schema above, including required
document metadata, source text/spans/hash, and tokenization provenance. Blank JSONL
lines are ignored. Malformed JSON, duplicate JSON keys or sentence IDs, missing
fields, blank text, invalid metadata and existing prediction fields are rejected
with an error before output is written. Extra source fields are preserved.
The input tokenizer identity must match the pinned FinBERT revision.

Every successful output retains **every input field unchanged** and adds:

| Field | Meaning |
| --- | --- |
| `predicted_sentiment` | `positive`, `negative` or `neutral` from the model label mapping |
| `prob_positive`, `prob_negative`, `prob_neutral` | Actual softmax probabilities; sum approximately 1 |
| `model_name`, `model_revision` | `ProsusAI/finbert` and the pinned checkpoint revision |
| `inference_device` | `cpu` |
| `batch_inference_version` | Prediction pipeline version, currently `1` |

Successful predictions remain in input order with unique source sentence IDs.
There are no timestamps in prediction records. Numerical reproducibility is
checked to absolute tolerance `1e-6`; hardware, batch padding or dependency changes
may cause small differences. Neither the JSONL output nor demo counts establish
classification accuracy.

**Long inputs are skipped, never truncated.** Phase 2 flags and stored over-limit
counts cause immediate skips. For remaining records, actual token lengths are
recomputed using the loaded tokenizer (including special tokens). Over-limit
texts are skipped even if their stored flag is false. A stale short token count
raises an error requesting preprocessing again, rather than silently rewriting
provenance. There is no automatic chunking in Phase 3.

The CLI prints loaded/classified/skipped counts, model name, CPU device and output
location. If anything is skipped, it also prints each reason and writes a companion
file named `<output>.skipped.jsonl`. Each skipped record preserves its input plus
`skip_reason`, with **no sentiment assigned**. An all-skipped run writes an empty
prediction file and does not load model weights. Existing prediction/companion
files are never overwritten. Invalid inputs return exit code 2; a completed run
with documented skips returns 0, so check the summary and companion file.

For reuse without reloading the model:

```python
from src.inference import FinBertAnalyzer
from src.batch_inference import predict_records, read_records

analyzer = FinBertAnalyzer(local_files_only=True)
result = predict_records(read_records("outputs/synthetic_batch_input.jsonl"),
                         analyzer=analyzer, batch_size=8)
print(len(result.predictions), len(result.skipped))
```

### Authentic financial-document demonstration

The original Phase 3 demonstration used four official releases and their complete opening CEO and CFO
statement paragraphs extracted locally using one consistent selection policy:
Microsoft FY2025 Q1–Q2 and Apple FY2025 Q1–Q2. All sentences in the selected
paragraphs were included, including product commentary. These are **excerpts**,
not full releases or an exhaustive financial corpus. No sentiment-based selection
or quarter-over-quarter calculation was performed.

See [the data plan](docs/data_plan.md) for official links, fiscal dates, exact
extraction scope, access/redistribution notes and a manual preparation workflow.
The metadata files are suitable for Git; Phase 4 has updated these manifests to
the broader policy below and added four more. TXT files and original HTML
snapshots remain under ignored `data/raw/`. On another
checkout, prepare the TXT files first. For an existing local real document:

```bash
python -m src.preprocessing \
  --metadata data/metadata/msft_fy2025_q1.json \
  --output outputs/msft_q1_sentences.jsonl
python -m src.batch_inference \
  --input outputs/msft_q1_sentences.jsonl \
  --output outputs/msft_q1_predictions.jsonl \
  --batch-size 8 --local-files-only
```

User-provided releases use the same workflow: place a UTF-8 narrative TXT in
`data/raw/`, create a matching metadata JSON with accurate source facts, then run
these two CLIs. HTML/PDF ingestion and automated downloading are not implemented.

Phase 3 baseline verification: **112 passed, 3 skipped** by default and **115 passed**
with `RUN_FINBERT_INTEGRATION=1 HF_HUB_OFFLINE=1`. Integration tests share one
cached model and cover real multi-batch inference, CPU placement, all probabilities,
metadata preservation and agreement across batch sizes and single inference.

The local real-document run classified **18 of 18 sentences across four releases**
in batches of 8, with **0 skips and 0 processing errors**. The measured pipeline
time was about **1.16 seconds**, including about 0.26 seconds to load cached model
weights, excluding Python import/startup and source acquisition; timings are
machine/cache-specific. Predictions and the execution summary are in ignored
`outputs/phase3/`. These counts describe software execution, not labelled model
evaluation. Synthetic and real-data CLI runs also succeeded offline.

## Phase 4: sentiment aggregation and fiscal-quarter comparison

`src.analysis` reads saved predictions, validates their probabilities and source
metadata, groups by company and fiscal period, and produces the four artifacts
below. It uses only the standard library and imports neither PyTorch nor
Transformers. FinBERT is not loaded during aggregation. Shared schema validation
was moved into `src.records`; Phase 3's public reader/validator imports still work.

For a complete workflow on a user-provided financial TXT, first prepare metadata
as described above, run preprocessing and batch inference, then analyze saved
predictions. The expanded eight-document run is already available locally:

```bash
source .venv-finsentiment/bin/activate
python -m src.analysis \
  --input outputs/phase4/real_predictions.jsonl \
  --source-input outputs/phase4/real_sentences.jsonl \
  --output-dir outputs/phase4/my_report
```

Use a new output directory: existing artifacts are never overwritten. The
`--source-input` is the complete Phase 2 JSONL. The CLI automatically discovers
`<input>.skipped.jsonl`, or accepts an explicit `--skipped-input`. Reconciliation
requires every source ID to occur exactly once in classified or skipped outputs,
with every source field unchanged. Missing outputs cause an error, not inferred
neutral predictions. If source records are omitted, the report marks coverage as
unverified and source totals/coverage as unknown; supplied skips are known skips
only. Supply source records for trustworthy coverage reporting.

### Metrics and fiscal ordering

- Positive/negative/neutral counts and percentages use **classified sentences**.
- Mean probability for each class averages that class's probability over **all**
  classified sentences, not just those assigned to the class.
- **Net sentiment = (positive count − negative count) / classified count.**
  Neutral sentences remain in the denominator. This is a transparent descriptive
  indicator on [-1, +1], not an established financial risk metric.
- Zero-classified groups have counts of zero but **null** percentages, means and
  net sentiment; CSV uses blank cells and Markdown uses N/A. Skips remain skips.
- Comparisons use consecutive fiscal quarters **within each company**, including
  FY2024 Q4 → FY2025 Q1. Gaps are flagged and are not bridged. Distribution changes
  are **percentage points**, while net changes use the original [-1, +1] scale.

Duplicate IDs, conflicting company/document metadata, nonfinite or invalid
probabilities, wrong winning labels, and mixed model/preprocessing versions are
rejected. Each company must use consistent document type, extraction policy,
comparison profile and selected section names across its periods. Different
companies may have different profiles and are never compared as equivalent text.

### Artifacts

| File | Contents |
| --- | --- |
| `quarterly_sentiment_summary.csv` | Company/quarter, document count, total/classified/skipped units, verified coverage, three class counts/percentages, three mean probabilities, net score and extraction policy/profile |
| `quarterly_comparison.csv` | Previous/current fiscal periods, distributions, percentage-point changes, net scores/change, sample sizes and coverage for both periods |
| `sentiment_analysis.json` | Company summaries, processing versions, source documents/hashes, period summaries, evidence, classified/skipped ID membership, comparisons, coverage, input-file hashes and limitations |
| `financial_sentiment_report.md` | Readable methodology, distribution/comparison tables, official sources, exact supporting text and interpretation limits |

Evidence is the highest predicted-class-probability sentence in each available
category per period, with sentence ID as tie-breaker. An absent class is stated
explicitly. Confidence is not proof of objective sentiment. Exact text, source
URL, document/period, offsets and hash remain traceable. Evidence/report files
contain third-party excerpts and stay in ignored output directories.
CSV/JSON retain arithmetic precision; Markdown rounds percentages to one decimal
and net scores to two. Outputs contain no timestamps and are deterministic for
the same validated inputs. No significance testing or accuracy metrics are run.

### Expanded real-document collection and actual results

All eight FY2025 releases were re-extracted under `financial_narrative_v2`, including
the original four, then preprocessed with version 2 and inferred on CPU. The
selection includes quarterly opening overview, CEO/CFO statements and capital
return. Microsoft also supplies segment/business financial highlights and an
outlook paragraph; Apple's opening narrative lacks equivalent sections. The
metadata records company-specific profiles and limitations. Tables, navigation,
headings, webcast logistics, standalone annual-results sections, duplicate blocks
and legal boilerplate are excluded. Every chosen block is included regardless of
sentiment. Opening Q4 management statements can still discuss full-year results.

See [the eight-source inventory and extraction instructions](docs/data_plan.md).
Microsoft has 22, 23, 22 and 22 units in Q1–Q4; Apple has 10, 10, 8 and 11.
That is **128 classified units, 8 documents, 2 companies, 8 company-periods,
0 skipped units and 6 consecutive-quarter comparisons**. Coverage is 100% of
the extracted local text units, not 100% of either complete earnings release.

| Company | FY2025 quarter | Classified | Positive / negative / neutral counts | Net sentiment |
| --- | --- | --- | --- | --- |
| Microsoft | Q1 | 22 | 17 / 0 / 5 | 0.77 |
| Microsoft | Q2 | 23 | 18 / 1 / 4 | 0.74 |
| Microsoft | Q3 | 22 | 18 / 0 / 4 | 0.82 |
| Microsoft | Q4 | 22 | 18 / 0 / 4 | 0.82 |
| Apple | Q1 | 10 | 6 / 0 / 4 | 0.60 |
| Apple | Q2 | 10 | 8 / 0 / 2 | 0.80 |
| Apple | Q3 | 8 | 5 / 0 / 3 | 0.62 |
| Apple | Q4 | 11 | 7 / 0 / 4 | 0.64 |

For example, Microsoft Q1→Q2 changes were +1.0 pp positive, +4.3 pp negative and
−5.3 pp neutral; net sentiment changed by about −0.034. The sample sizes are only
22 and 23 units. These are descriptive differences with no significance claim.
The broader scope increased the original 18-unit demonstration to 128 units, but
the samples remain small, company-selected and potentially promotional.

The executed reports are in ignored `outputs/phase4/final_report/`; source-unit
and prediction JSONL plus an execution summary are in `outputs/phase4/`. The
complete eight-document run took about 6.3 seconds in the measured environment,
excluding Python imports and source acquisition. This is not a performance benchmark.

Run all checks with the test commands above. Phase 4 verification: **149 passed,
5 skipped** by default; **154 passed** with cached integration enabled. Tests
cover known arithmetic, neutral denominators, fiscal rollover/gaps, duplicate and
incompatible inputs, skipped/zero-classified groups, deterministic exports and
synthetic plus real eight-release workflows. Real-source tests skip explicitly on
a fresh clone lacking the ignored TXT files; no automated test accesses the web.

## Phase 5: labelled evaluation and robustness

`src.evaluation_data` retrieves small pinned files explicitly and loads/audits
genuine references. `src.evaluation` reuses the existing strict CPU classifier,
scores FinBERT and the fixed `keyword_v1` baseline on identical examples, and
exports metrics, errors and subgroups. Saved predictions can be rescored without
loading a model. No fitting, extra model, paid API or package installation was used.
Existing scikit-learn 1.9.1 supplies metrics; `datasets` and `pyarrow` are not needed.

Read [the protocol](docs/evaluation_methodology.md),
[executed findings](docs/evaluation_findings.md), and
[annotation guide](docs/annotation_guide.md). Source revisions, retrieval dates,
file hashes and baseline/protocol hashes are in
`data/evaluation_manifests/phase5_sources.json`. Raw data, human annotation text
and generated artifacts remain ignored by Git.

### Datasets, mappings and independence limitations

| Cohort actually evaluated | N | Negative / neutral / positive support |
| --- | --- | --- |
| Financial PhraseBank AllAgree | 2,259 | 303 / 1,386 / 570 |
| FiQA original labelled train, single-target | 1,052 | 331 / 74 / 647 |

PhraseBank's original archive has 2,264 AllAgree rows; five duplicates were removed.
**It overlaps FinBERT training and is not an independent held-out benchmark.**
No overlapping agreement subsets were pooled into the evaluation.

The requested FiQA repository is access-gated, and original organizer downloads
returned 404. The actual accessible files come from a pinned FiQA participant
repository: labelled training data and 192 **unlabelled** test inputs. The latter
have no sentiment scores; this run is an original-train diagnostic, not the gated
publisher's 234-item test split. The original FinBERT paper also used FiQA
regression experiments, so the current classifier's exact FiQA exposure is unknown.
**Proven independent unseen-data performance has not been established.**

FiQA's continuous scores use the published PIXIU mapping, fixed before results:
**score < -0.1 → negative; -0.1 <= score < 0.1 → neutral; score >= 0.1 → positive**.
Sixty multi-target sentences are excluded because FinBERT is sentence-level;
one train/test text overlap is excluded. No eligible overlap was found against
PhraseBank 50Agree text. These checks do not rule out near-duplicates, pretraining
contamination or prior research exposure. This classification diagnostic differs
from the original target/aspect regression task.

Records preserve `example_id`, `text`, `original_text`, `true_label`,
`original_label`, `source_dataset`, `split` and available source/target/aspect
metadata. Unknown labels and missing text/references cause errors. Duplicate
detection uses case/Unicode/whitespace normalization only; inference uses the
original text. Same-label duplicates keep the lowest stable ID; conflicting
references exclude all copies with recorded reasons. Overlong inputs are excluded
from both models without truncation; neither actual run had length exclusions.

### Obtain data and run

Retrieval is explicit, about 1.1 MB total, with no model download. Skip fetch on
this laptop when files already exist; loaders never download automatically:

```bash
source .venv-finsentiment/bin/activate
python -m src.evaluation_data fetch --dataset phrasebank
python -m src.evaluation_data fetch --dataset fiqa

python -m src.evaluation --dataset phrasebank --split allagree \
  --output-dir outputs/phase5/my_phrasebank --local-files-only

python -m src.evaluation --dataset fiqa --split train \
  --phrasebank-reference data/evaluation/phrasebank/FinancialPhraseBank-v1.0.zip \
  --output-dir outputs/phase5/my_fiqa --local-files-only
```

Use fresh output directories. Add `HF_HUB_OFFLINE=1` for offline inference.
Batch size defaults to 8, configurable with `--batch-size`; the Python API accepts
one existing analyzer for multiple datasets, as used in the actual combined run.

### Metrics and actual outcomes

Metrics include accuracy, class precision/recall/F1/support, macro precision/
recall/F1, weighted F1, predicted distributions and confusion matrices. Fixed
label order is **negative, neutral, positive**; matrix rows are truth and columns
prediction. Macro averages include all three labels, with zero division set to 0.
The original keyword baseline has a fixed vocabulary, three-word negation scope
and neutral ties; it was not tuned on results and is not a state-of-the-art model.

| Cohort | Model | Accuracy | Macro F1 | Weighted F1 |
| --- | --- | --- | --- | --- |
| PhraseBank AllAgree (N=2,259) | FinBERT | 0.9717 | 0.9625 | 0.9720 |
| PhraseBank AllAgree (N=2,259) | keyword_v1 | 0.8039 | 0.7223 | 0.7925 |
| FiQA original train (N=1,052) | FinBERT | 0.4990 | 0.4765 | 0.5804 |
| FiQA original train (N=1,052) | keyword_v1 | 0.3051 | 0.3188 | 0.3883 |

On FiQA, 322 positive references became neutral predictions; neutral F1 is 0.1789.
There were 125 incorrect predictions with confidence >=0.9. Headline accuracy was
0.6233 (N=377), versus 0.4296 for posts (N=675). Subgroups also cover true class,
<=20/>20 words, lexical negation and fixed confidence bins, with sizes, supports
and warnings for small/class-incomplete groups. Differences are descriptive, not
significance tests, causal explanations or demographic fairness measurements.
High confidence is not proof of correctness. The operating-expenses demo remains
unlabelled and is not counted as measured error evidence.

### Reports, reproduction and human labels

Actual artifacts are in `outputs/phase5/phrasebank/` and `outputs/phase5/fiqa/`:
`evaluation_metrics.json`, `evaluation_metrics.csv`, `confusion_matrix.csv`,
`error_analysis.csv`, `robustness_summary.csv`, `evaluation_report.md`, plus
`excluded_examples.csv` and the cohort-bound `model_predictions.json` cache.
Error text is omitted by default; `--include-text` is for local review subject to
source terms. Objective error-pattern tags are co-occurrences, not proven causes.

All eight artifacts per dataset reproduced byte-for-byte by rescoring saved
predictions against the same references. This mode does not load FinBERT:

```bash
python -m src.evaluation --dataset fiqa --split train \
  --phrasebank-reference data/evaluation/phrasebank/FinancialPhraseBank-v1.0.zip \
  --predictions-input outputs/phase5/fiqa/model_predictions.json \
  --output-dir outputs/phase5/my_fiqa_reproduced
```

Default tests: **209 passed, 7 skipped**; offline integration enabled: **216
passed**. Tests cover dataset/mapping validation, duplicates, metric arithmetic,
zero division, single-class data, baseline rules, errors/subgroups, deterministic
exports, cache integrity, human-label gating and real cached-model integration.
Tests never download datasets. Model repeatability is checked on small integration
cases, and a separate full FiQA CLI inference rerun reproduced all eight artifacts
byte-for-byte. PhraseBank's full metric replay uses the saved prediction cache.

The blind local template `data/evaluation/annotations/earnings_sample.csv` contains
**80 blank reference labels**, sampled from 119 distinct Phase 4 texts with seed
42 and document-stratified round-robin selection. It contains source metadata but
no FinBERT labels/confidence. Every row needs a genuine `human_label`,
`annotator_id` and `review_status=reviewed`; the loader rejects pending reviews.
No earnings accuracy, annotation agreement or labelled company/quarter subgroup
performance is claimed yet.

## Current limitations

- Local English TXT only, with manually prepared metadata; no PDF/HTML ingestion,
  automatic language verification or scraping. Dataset evaluation, quarterly
  summaries/comparisons and CSV/JSON/Markdown descriptive reports are implemented.
- All JSON records/results are held in memory; model tensors are limited to one
  configurable batch. The implementation targets small collections, not large
  corpus streaming. Batch prediction CSV export is not implemented; analysis and
  evaluation do export CSV reports.
- Segmentation is heuristic and requires review. Long sentences are rejected or
  explicitly flagged by preprocessing; standalone Phase 1 inference still truncates.
- Softmax scores are model outputs, not calibrated certainty or financial advice.
  Demonstration results do not establish accuracy or real-world reliability.
- Repeatability is checked within numerical tolerance on this environment;
  different hardware or dependency versions can produce small differences.
- Financial PhraseBank was used to fine-tune FinBERT, so a future evaluation on
  that dataset must not be described as independent unseen-data evaluation.
  FiQA also needs dataset inspection and careful label mapping before evaluation.

Next: Phase 6 can strengthen research presentation and, after human annotation
and evaluation review, add a lightweight Gradio upload interface and potential
Hugging Face Spaces deployment. These are planned only. No interface, PDF
extraction, deployment, model training or demographic fairness metrics were
implemented during Phase 5.
