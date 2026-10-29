# Evaluation findings: three distinct evidence levels

Executed on 2026-10-29 using CPU-only `ProsusAI/finbert` revision
`0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031` and the unchanged `keyword_v1`
baseline. All results come from genuine local labelled records. SEntFiN is the
primary external benchmark; FiQA and PhraseBank are separate diagnostics. **No
combined overall accuracy is computed.** See the [fixed protocol](evaluation_methodology.md)
and preserved [candidate investigation](external_benchmark_audit.md).

SEntFiN provides our strongest external evaluation. The sample contains
human-annotated financial headlines and underwent exact and normalized-text
overlap checks against accessible reference datasets. Nevertheless, the absence
of all prior training or pretraining exposure cannot be established. These
headline results do not directly measure accuracy on Microsoft and Apple earnings
releases. No official held-out SEntFiN partition is claimed.

## A. Primary external benchmark: SEntFiN

The [candidate audit](external_benchmark_audit.md) examined exactly four sources.
ARSD's records and explicit dataset licence could not be found; Mendeley's labels
are model votes with generated paraphrases; TFNS's source card does not establish
human annotation. None was scored. Author-published **SEntFiN** was accepted under
its MIT licence and documented three-MBA-annotator procedure, with explicit scope
and exposure limitations. This adds external human-reference evidence beyond the
known-overlap PhraseBank score and uncertain-exposure FiQA diagnostic.

The actual CSV contains **10,753 rows**. Excluding 2,847 multi-entity rows, 44
same-label duplicate extras and all 16 conflicting-reference rows leaves **7,846**.
No exact/normalized match to 6,151 accessible PhraseBank 50Agree and FiQA train/test
reference rows was found. A bounded lexical near-duplicate screen found 15
internal candidate pairs and zero cross-reference candidates; neither pair member
was jointly selected in the final sample. This screen is not exhaustive semantic
or pretraining-overlap detection. No official held-out partition exists.

Before inference, a proportional **2,000-row sample, seed 42**, was fixed: negative
602, neutral 677, positive 721. The other 5,846 valid rows are logged as unsampled,
separately from quality/task exclusions. Both frozen models evaluated all 2,000;
there were **zero token-limit exclusions**. This is a sample result, not a score
for the full corpus, a publisher-defined test partition or annual reports.

| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 |
| --- | --- | --- | --- | --- | --- |
| Unchanged ProsusAI/finbert | 0.7475 | 0.7579 | 0.7485 | 0.7496 | 0.7495 |
| Frozen keyword_v1 | 0.5715 | 0.6816 | 0.5665 | 0.5612 | 0.5632 |

| True class | Support | FinBERT precision / recall / F1 | Baseline precision / recall / F1 |
| --- | --- | --- | --- |
| Negative | 602 | 0.7774 / 0.7542 / 0.7656 | 0.8358 / 0.3721 / 0.5149 |
| Neutral | 677 | 0.6534 / 0.7770 / 0.7099 | 0.4539 / 0.8656 / 0.5955 |
| Positive | 721 | 0.8429 / 0.7143 / 0.7733 | 0.7551 / 0.4619 / 0.5731 |

Confusion matrices: rows are true negative/neutral/positive; columns are predicted
negative/neutral/positive. Each matrix totals 2,000.

| Model / true class | Predicted negative | Predicted neutral | Predicted positive |
| --- | --- | --- | --- |
| FinBERT / negative | 454 | 121 | 27 |
| FinBERT / neutral | 82 | 526 | 69 |
| FinBERT / positive | 48 | 158 | 515 |
| Baseline / negative | 224 | 336 | 42 |
| Baseline / neutral | 25 | 586 | 66 |
| Baseline / positive | 19 | 369 | 333 |

FinBERT predicts 584 negative, 805 neutral and 611 positive; the baseline predicts
268, 1,291 and 441. FinBERT's **negative recall is 454/602 = 75.42%**, versus
224/602 = 37.21% for the baseline. Its 148 missed negative references include 121
neutral predictions and 27 positive predictions. These observations do not
establish negative recall on the eight unlabelled earnings releases.

### Observed errors and interpretation

There are **505 FinBERT errors**. Transitions are positive→neutral 158,
negative→neutral 121, neutral→negative 82, neutral→positive 69, positive→negative 48,
and negative→positive 27. Of the errors, 124 have maximum probability >=0.9.
Average confidence is 0.8734 on correct predictions and 0.7740 on errors; confidence
is not proof of correctness or a calibration study.

The following are paraphrases of genuine local source records, with stable IDs;
full original text is omitted from Git. Labels are the published human references,
not newly assigned judgements. These are selected error illustrations, not an
estimate of how often a semantic failure occurs.

| Source ID | Published truth → prediction | Observed content (paraphrase) |
| --- | --- | --- |
| sentfin_04389 | negative → positive (0.6745) | Deccan Chronicle's quarterly net loss becomes larger. |
| sentfin_09031 | negative → positive (0.5741) | Henkel India's quarterly loss widens. |
| sentfin_08434 | positive → negative (0.9758) | TV18 reports a substantial reduction in annual net loss. |
| sentfin_06518 | neutral → positive (0.8803) | Hindalco faces rising input costs alongside potentially helpful demand. |
| sentfin_09957 | negative → neutral (0.9459) | A commentator recommends against entering oil-marketing stocks at current levels. |
| sentfin_00686 | neutral → negative (0.9589) | Reliance may report a flat first-quarter performance. |
| sentfin_08921 | negative → positive (0.9153) | Soyabean oil prices rise with retailer demand, illustrating target-dependent commodity polarity. |

Fixed lexical tags occur in 32 caution-related, 27 loss-related, 24 negation-related
and one expense/increase-related errors; tags can overlap. No error matches the
fixed revenue-decrease or guidance-reduction tag. This does not prove an absence
of those semantic categories. No separate margin-decline rate was established.
Mixed clauses, loss direction, negation and target interpretation are plausible
review hypotheses, not demonstrated causal explanations. Published neutral labels
on some loss announcements also reveal a task/annotation difference; no references
were relabelled after looking at model output.

All sampled headlines have <=20 whitespace words, so no long-text comparison is
available. Negation-present examples number 56 (18 negative, 32 neutral, 6 positive),
with accuracy 0.5714 and macro F1 0.4474; the remaining 1,944 have accuracy 0.7526
and macro F1 0.7544. Different supports and sparse positive examples limit this
comparison. It is descriptive, not a significance test or causal robustness claim.

### Artifacts and limits

The ignored `outputs/phase5b/sentfin/` directory contains the required
`external_evaluation_metrics.csv`, `external_evaluation_metrics.json` and
`external_evaluation_report.md`, alongside the shared framework's confusion,
error, subgroup, exclusions and prediction-cache files. Error text is omitted by
default; source IDs and hashes support local review. The metadata-only
`data/evaluation_manifests/phase5b_sources.json` records all four candidates,
source hashes, fixed selection and audit counts. Model and baseline are unchanged.

SEntFiN supplies useful additional external human-labelled evidence, with no
identified exact/normalized overlap against the available references. Its
single-entity financial headlines differ from quarterly earnings prose, its
original study trained a different finBERT variant, and complete historical
checkpoint exposure cannot be proven. No official test split or verified earnings
accuracy is claimed. Phase 6 presentation can proceed with these caveats; no UI,
PDF extraction or deployment was implemented.

Pre-finalization Phase 5B verification: **231 passed, 8 skipped** by default and **239 passed** with
`RUN_FINBERT_INTEGRATION=1 HF_HUB_OFFLINE=1` (CPU thread limits set to 2).
A cache replay **and a second complete 2,000-row offline CPU inference** each
reproduced all **11 artifacts byte-for-byte**, including model probabilities,
metrics, reports and exclusions. `outputs/phase5b/run_verification.json` records
artifact hashes and consistency checks. No new packages, model weights or CUDA
were installed. Finalization retains these measured results.

## B. Secondary diagnostic: FiQA original labelled train

The pinned original-format participant repository supplies labelled **training**
files and 192 unlabelled test inputs. The requested publisher's files were gated
and organizer download links returned 404. The participant test inputs contain no
sentiment scores; this is not the publisher's 234-example test split or an
independent held-out test. The released FinBERT classifier's exact FiQA exposure
is unverified; the original study used FiQA for regression experiments.

Of 1,113 training sentences, 60 multi-target examples and one train/test text
overlap are excluded, leaving **1,052** single-target examples: 331 negative,
74 neutral and 647 positive. No eligible normalized overlap was identified against
PhraseBank 50Agree, and no token-limit exclusions occurred. These checks do not
rule out semantic near-duplicates, pretraining or prior research exposure.

The mapping, fixed before results, is negative for score < -0.1, neutral for
-0.1 <= score < 0.1, positive for score >= 0.1. Original scores, targets and aspects
are preserved. Continuous target/aspect annotations differ from sentence-level
classification. Thus **49.90% accuracy is a diagnostic cohort score, not held-out
generalization performance**. Loader, baseline comparisons and reports remain.

| Model | N | Accuracy | Macro F1 | Weighted F1 | Negative / neutral / positive F1 |
| --- | --- | --- | --- | --- | --- |
| FinBERT | 1,052 | 0.4990 | 0.4765 | 0.5804 | 0.6700 / 0.1789 / 0.5804 |
| keyword_v1 | 1,052 | 0.3051 | 0.3188 | 0.3883 | 0.4152 / 0.1378 / 0.4033 |

FinBERT confusion: true rows and predicted columns, negative/neutral/positive:

```text
199  113   19
 15   50    9
 49  322  276
```

FiQA has 647 positive references, but FinBERT predicts positive only 304 times.
It predicts neutral 485 times although only 74 references map to neutral. The
largest confusion is **322 positive references predicted neutral**. Positive
precision is 0.9079 but recall is 0.4266; neutral precision is only 0.1031. These
are cohort-specific observations, not evidence that earnings labels are wrong.

Inspected genuine cases, with original text retained only in local source files:

- `fiqa_train_headline_199`: reference positive (score 0.252), prediction negative,
  confidence 0.9565. The sentence combines improved profit with contract problems;
  mixed content is an observed feature, not a proven cause of the prediction.
- `fiqa_train_headline_991`: reference positive (0.154), prediction neutral,
  confidence 0.9514. The source concerns board appointments; its mapped reference
  and the sentence-level model disagree about direction.
- `fiqa_train_post_16347`: reference neutral (-0.011), prediction negative,
  confidence 0.9537. The microblog uses technical-trading terminology.
- `fiqa_train_post_14959`: reference positive (0.65), prediction neutral,
  confidence 0.9440. The input is a terse trade-position statement.

No causal explanation is asserted. Objective tags identify lexical co-occurrences
for manual investigation. For example, 40 FiQA errors contain lexical negation;
this is an error count, not a negation error rate without its denominator. The
earlier operating-expenses demo has no verified reference label and is not
included as a measured error or accuracy example.

## Confidence and robustness

FiQA has 527 errors, including **125 high-confidence errors**. Of 364 FiQA predictions at
confidence >=0.9, accuracy is 0.6566. Mean FiQA confidence is 0.8234 on correct
predictions and 0.7852 on incorrect predictions. No formal calibration claim is
made; high confidence clearly does not guarantee correctness in this cohort.

| FiQA FinBERT subgroup | N | Accuracy | Macro F1 |
| --- | --- | --- | --- |
| Headlines | 377 | 0.6233 | 0.5987 |
| Posts | 675 | 0.4296 | 0.4092 |
| <=20 whitespace words | 958 | 0.5084 | 0.4824 |
| >20 whitespace words | 94 | 0.4043 | 0.4065 |
| Lexical negation present | 63 | 0.3651 | 0.3801 |
| Lexical negation absent | 989 | 0.5076 | 0.4800 |

These groups have different class supports and source characteristics. Differences
are descriptive, without a significance test or causal interpretation. Fixed-label macro F1 includes zero-support classes as zero;
affected groups carry warnings. This is classification consistency analysis,
not demographic fairness.

## C. Training-overlap diagnostic: Financial PhraseBank

The original AllAgree file has 2,264 rows. Removing five same-label duplicate
extras leaves **2,259**: 303 negative, 1,386 neutral and 570 positive. No other
overlapping agreement subset is scored and no token-limit exclusions occurred.

PhraseBank was used for FinBERT fine-tuning. Known training overlap can inflate
its high score, so it is **excluded from independent validation claims**. It
remains useful for checking the inference, labels, metrics, baseline and export
pipeline. It cannot validate accuracy on unseen earnings reports.

| Model | N | Accuracy | Macro F1 | Weighted F1 | Negative / neutral / positive F1 |
| --- | --- | --- | --- | --- | --- |
| FinBERT | 2,259 | 0.9717 | 0.9625 | 0.9720 | 0.9430 / 0.9824 / 0.9620 |
| keyword_v1 | 2,259 | 0.8039 | 0.7223 | 0.7925 | 0.5784 / 0.8696 / 0.7189 |

FinBERT confusion: true rows and predicted columns, negative/neutral/positive:

```text
298     1     4
 19  1340    27
 12     1   557
```

There are 64 FinBERT errors, 14 with confidence >=0.9. Genuine example
`phrasebank_AllAgree_01991` has neutral reference, negative prediction and 0.6341
confidence: a factual agreement-expiration statement is assigned negative
sentiment. Among 90 negation examples, 85 are neutral, three negative and two
positive; subgroup macro F1 0.7567 is especially support-sensitive. Fixed-label
macro F1 includes zero-support classes as zero, with warnings in exports.

The large differences between these datasets have no single demonstrated cause:
training overlap, annotation tasks, score discretization, class distributions and
source language differ. No cross-dataset ranking or pooled accuracy is inferred.

## Shared integrity, historical findings and reproducibility

Both models always score identical eligible examples within each dataset. The
baseline is a fixed original keyword rule set with negation handling, not a tuned
financial lexicon or a trained state-of-the-art comparator. Metrics are point
estimates, without a significance claim. Sources, revisions, dates and hashes are
in the metadata-only manifests under `data/evaluation_manifests/`; third-party
texts and generated reports remain ignored by Git.

The unfinished earnings annotation workflow has been removed. No reference labels
were produced, copied from model predictions or presented as expert earnings
labels. No earnings accuracy or annotation agreement is claimed. Actual releases,
metadata, Phase 4 outputs and previous commits remain intact.

Historical Phase 5 reports are preserved under `outputs/phase5/fiqa/` and
`outputs/phase5/phrasebank/`. Each has eight artifacts: metrics JSON/CSV, confusion,
error and subgroup CSVs, Markdown report, exclusions and prediction cache. Original
Phase 5 cache replay reproduced these byte-for-byte; a separate full FiQA inference
also reproduced probabilities. Original inference timings were about 132 seconds
for PhraseBank and 44 seconds for FiQA, excluding acquisition/import startup;
these are environment-specific timings, not comparative performance benchmarks.
Historical Phase 5 tests were 209 passed / 7 skipped by default and 216 passed
with offline integration. Finalization verifies current metrics against these
saved results; historical report wording is retained as a record of its run.

ARSD, Mendeley and TFNS remain rejected candidates documented in the original
audit and metadata. Their unused runtime registry/CLI options, rejection-specific
tests and local investigation downloads were removed. No loaders existed for them.
Shared overlap, sampling, metrics and error-analysis logic is retained.

Phase 6 presentation can proceed with these evidence levels and limitations.
No Gradio, PDF extraction, deployment or other Phase 6 implementation is included.

Finalization verification: **229 passed, 8 skipped** by default and **237 passed**
with offline cached-model integration (CPU thread limits 2). The two fewer tests
reflect removal of the unused candidate registry and its rejection-specific CLI
path. All three actual CLI cache evaluations completed: SEntFiN 2,000, FiQA 1,052,
and PhraseBank 2,259. Both models' complete metrics, dataset audits, coverage,
confidence, errors and subgroup results exactly match the saved runs; prediction
caches are byte-identical. Baseline source hash is unchanged. Evidence is local at
`outputs/phase5b/finalization/verification.json`. No dependencies or model weights
were installed or downloaded during finalization.
