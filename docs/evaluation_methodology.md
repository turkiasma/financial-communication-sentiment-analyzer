# Evaluation protocol: Phase 5 and Phase 5B

Protocol fixed before running benchmark inference on 2026-10-29. Neither model
weights, score thresholds, keyword vocabulary nor subgroup cutoffs are fitted to
evaluation outcomes. The baseline vocabulary is fixed in
`src/evaluation_baseline.py`; its version is `keyword_v1`.

## Dataset roles and access

Only SEntFiN, FiQA and PhraseBank are supported by retrieval/evaluation CLIs.
Their metrics are never combined into an overall score. Rejected candidates remain
in the historical audit only; no runtime integrations or local downloads remain.

- **Primary external benchmark — SEntFiN:** documented human entity labels from the author repository,
  MIT licence, restricted to single-entity headlines. The pinned source, exact
  schema, exclusions, reference overlap checks, bounded near-duplicate screen and
  **2,000-example proportional sample, seed 42** were fixed before inference in
  the [external audit](external_benchmark_audit.md). No official held-out partition
  exists. This is primary external sample evidence, with unknown complete model
  exposure and a headline/entity versus earnings/sentence task difference.
- **Secondary diagnostic — FiQA:** the requested [TheFinAI dataset](https://huggingface.co/datasets/TheFinAI/fiqa-sentiment-classification)
  redirects to gated `TheFinAI/en-fiqa-absa`; it was not accessed through the gate.
  The [original organizer](https://sites.google.com/view/fiqa/home) defines target/
  aspect scores in [-1, 1] and restricts data to noncommercial use. Its train/test
  download links returned 404. A [FiQA participant repository](https://github.com/dayanfcosta/fiqa-2018-task1)
  supplies original-format labelled **training** files and unlabelled test inputs,
  pinned at `acd0f4cfafa6421774bc8f009e8fa3e0ec441357`. Use the labelled training
  partition for a frozen-model external **diagnostic**, without fitting or tuning.
  This is not the gated publisher's 234-item test split or verified held-out test
  performance. The participant copy has no organizer checksum to authenticate it.
  No extra redistribution permission is assumed.
  The [original FinBERT paper](https://arxiv.org/html/1908.10063) also trained and
  evaluated regression variants on FiQA. The current Hub classifier's exact FiQA
  exposure is not established by its PhraseBank-focused model card. Consequently
  this is an external-source **candidate diagnostic**, not a proven independent
  generalization benchmark; prior research/model-selection exposure is possible.
- **Training-overlap diagnostic — Financial PhraseBank:** the `Sentences_AllAgree.txt` agreement subset only,
  from [takala/financial_phrasebank](https://huggingface.co/datasets/takala/financial_phrasebank),
  revision `8d3fe0c36d5feec6b3cc5e455b0fcb4820fb9964`. This is a descriptive
  pipeline benchmark, **not independent held-out evidence**: the
  [FinBERT authors](https://github.com/ProsusAI/finBERT) identify PhraseBank as
  supervised training data. Do not pool overlapping agreement subsets. The
  50-agreement subset is used as overlap-reference text for FiQA and SEntFiN.
  The dataset identifies CC BY-NC-SA 3.0; source text stays local.
- **Earnings:** descriptive quarterly inference only. The abandoned annotation
  workflow has been removed; no user-created or AI-generated earnings gold labels
  enter evaluation. Existing real releases, metadata and Phase 4 outputs are retained.

## FiQA mapping fixed before model results

Use the published [PIXIU datasheet](https://papers.nips.cc/paper_files/paper/2023/file/6a386d703b50f1cf1f61ab02a15967bb-Supplemental-Datasets_and_Benchmarks.pdf),
section B.4, as precedent: negative for score < -0.1; neutral for
-0.1 <= score < 0.1; positive for score >= 0.1. Reject scores outside [-1, 1].
Boundary rules are explicit and never adjusted after observing results. This is
a derived three-class diagnostic, not the original FiQA regression task.

Keep only sentences with exactly one annotated target. Exclude multi-target
sentences before prediction, regardless of scores, because FinBERT has no
target-conditioned head. Preserve original target, aspect and score. Source type
is headline or post, as identified by the source file. Exclude labelled training
text appearing in unlabelled test inputs. Report known text overlap with the
PhraseBank 50-agreement subset when that reference archive is supplied, and
exclude such FiQA matches before inference. None of these checks establishes
absence of pretraining contamination (including Reuters text) or near-duplicates.

## Validation, duplicates and exclusions

Keep original text/labels, source revision, split, IDs and file hashes. Normalize
case, Unicode NFKC and whitespace **only for duplicate detection**; infer on the
original text. Identical text with matching labels keeps the lowest stable ID;
conflicting labels exclude all copies, with each decision recorded. Invalid or
missing text/labels cause errors, not guessed labels. Multi-target, overlap and
duplicate exclusions are recorded with IDs and counts. Evaluate each dataset
separately; no pooled score or merging of train/validation/test partitions.

Overlong (>512 tokens including special tokens) examples are excluded from **both**
models' metric cohorts, with explicit reasons; never truncate or compare different
cohorts. FinBERT uses the cached pinned checkpoint and strict `predict_batch` CPU
execution, loaded once. No training or additional model is involved.

## Metrics, baseline and robustness

Report sample size and class supports with accuracy, per-class precision/recall/
F1, macro precision/recall/F1, weighted F1, prediction distribution and confusion
matrix. Label order is **negative, neutral, positive**; rows are truth, columns
prediction. Macro averages always use these three labels, including zero support;
zero-division values are 0. Unsupported-class effects are explicitly noted.

The original keyword baseline counts fixed positive/negative words, flips their
sign under a three-word negation scope, and resolves ties to neutral. Punctuation
ends the scope. It has no fitting, pretrained weights, downloaded lexicon or
probability output. Expense/revenue context, sarcasm, target and numeric direction
are weaknesses by design. Both models use exactly the same eligible examples.

Subgroups are descriptive: true class; supplied headline/post/news source type;
whitespace word count <=20 versus >20; lexical negation present/absent; and
FinBERT maximum probability <0.6, [0.6, 0.9), or >=0.9. Thresholds are fixed here.
Report group sizes/supports and warnings below 30 examples or when a true class
is absent. Subgroup accuracy differences from the overall cohort are descriptive,
not significance tests, causal findings or demographic fairness measurements.
No protected attributes are inferred.

Incorrect predictions retain IDs, truth, predictions, probabilities, confidence
and objective text-pattern tags. Tags (negation, expense increase, revenue decline,
loss, guidance reduction, caution, abbreviation) are hypotheses for manual review,
not explanations of model behavior. Compare confidence on correct/incorrect
predictions; high confidence is not guaranteed correctness. The original
"Operating expenses increased significantly." diagnostic remains unlabelled and
is not counted as measured accuracy.

Raw data, annotation text and generated artifacts remain ignored by Git. Report
text is omitted from error artifacts by default; `--include-text` is for local
review subject to source terms. No in-domain earnings accuracy is established.
Access-limited test-set evaluation remains unavailable. External-source results do not prove absence of
pretraining exposure or validate all earnings-report uses.
