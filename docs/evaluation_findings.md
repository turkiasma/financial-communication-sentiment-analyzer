# Phase 5: executed evaluation findings

Executed on 2026-10-29 with the existing CPU-only environment, pinned
`ProsusAI/finbert` revision `0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031`, and the
unchanged `keyword_v1` baseline. All numbers below come from the local executed
evaluations, not synthetic test cases. The two cohorts were evaluated separately.

**These are diagnostic results, not proof of independent unseen-data accuracy.**
PhraseBank is known FinBERT fine-tuning data. The FiQA files used here are the
original labelled training partition from a participant repository, not the gated
publisher's held-out test. The original FinBERT paper also used FiQA regression
experiments; the released classifier's exact FiQA exposure is not established.

## Sources, class supports and exclusions

| Cohort | Raw source sentences | Evaluated distinct examples | Negative | Neutral | Positive |
| --- | --- | --- | --- | --- | --- |
| Financial PhraseBank AllAgree | 2,264 | 2,259 | 303 | 1,386 | 570 |
| FiQA original train, single-target | 1,113 | 1,052 | 331 | 74 | 647 |

PhraseBank: five same-label duplicate texts were removed, retaining stable first
IDs. No other agreement subset was scored. FiQA: 60 multi-target sentences and
one training sentence overlapping an unlabelled test input were excluded. The 192
participant test inputs have **zero** reference sentiment scores, so no test-set
accuracy is claimed. An overlap check against PhraseBank's 50-agreement text found
no eligible FiQA match; near-duplicates and pretraining contamination remain
unresolved. No examples exceeded the model's token limit in either evaluation.

FiQA reference mapping was fixed before model results using the PIXIU datasheet:
score < -0.1 is negative; -0.1 <= score < 0.1 is neutral; score >= 0.1 is positive.
Original scores, targets and aspects are retained. This derived classification
task differs from FiQA's original target/aspect regression task.

Sources and revisions, original file URLs, retrieval dates and SHA-256 hashes are
recorded in `data/evaluation_manifests/phase5_sources.json`. See the
[methodology](evaluation_methodology.md) for licensing, source links and the frozen
protocol. Third-party text is not committed.

## Actual FinBERT and baseline results

| Dataset | Model | N | Accuracy | Macro F1 | Weighted F1 |
| --- | --- | --- | --- | --- | --- |
| PhraseBank AllAgree | FinBERT | 2,259 | 0.9717 | 0.9625 | 0.9720 |
| PhraseBank AllAgree | keyword_v1 | 2,259 | 0.8039 | 0.7223 | 0.7925 |
| FiQA original train diagnostic | FinBERT | 1,052 | 0.4990 | 0.4765 | 0.5804 |
| FiQA original train diagnostic | keyword_v1 | 1,052 | 0.3051 | 0.3188 | 0.3883 |

Both models use exactly the same eligible examples in each dataset. Macro F1
weights the three classes equally and is the primary metric. The baseline is a
small original fixed keyword rule set with negation handling; it is not a tuned
financial lexicon or a strong trained classifier. These point estimates establish
performance against that baseline on these cohorts, not state-of-the-art ranking
or a statistically tested improvement.

| Dataset/model | Negative F1 | Neutral F1 | Positive F1 |
| --- | --- | --- | --- |
| PhraseBank FinBERT | 0.9430 | 0.9824 | 0.9620 |
| PhraseBank keyword_v1 | 0.5784 | 0.8696 | 0.7189 |
| FiQA FinBERT | 0.6700 | 0.1789 | 0.5804 |
| FiQA keyword_v1 | 0.4152 | 0.1378 | 0.4033 |

The large difference between the two datasets must not be attributed to one
proven cause: training overlap, annotation/task differences, score discretization,
class distribution and source language all differ. It does show why the high
PhraseBank number cannot validate the earnings application by itself.

## Confusion and genuine error review

Confusion matrices below use true rows and predicted columns in the order
negative, neutral, positive:

```text
PhraseBank FinBERT       FiQA FinBERT
298     1     4          199   113    19
 19  1340    27           15    50     9
 12     1   557           49   322   276
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
- `phrasebank_AllAgree_01991`: reference neutral, prediction negative, confidence
  0.6341. A factual agreement-expiration statement is assigned negative sentiment.

No causal explanation is asserted. Objective tags identify lexical co-occurrences
for manual investigation. For example, 40 FiQA errors contain lexical negation;
this is an error count, not a negation error rate without its denominator. The
earlier operating-expenses demo has no verified reference label and is not
included as a measured error or accuracy example.

## Confidence and robustness

PhraseBank has 64 errors, including 14 with maximum probability >=0.9. FiQA has
527 errors, including **125 high-confidence errors**. Of 364 FiQA predictions at
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
are descriptive, without a significance test or causal interpretation. In
PhraseBank's 90 negation examples, 85 are neutral, only three are negative and
two positive: its subgroup macro F1 of 0.7567 is especially support-sensitive.
Fixed-label macro F1 includes zero-support classes as zero; affected groups carry
warnings. This is classification consistency analysis, not demographic fairness.

## Human annotation and reproducibility status

The blind earnings template has **80 pending examples** sampled from 119 distinct
texts among the 128 Phase 4 units, with seed 42 and round-robin document coverage.
No reference labels were assigned or copied from FinBERT. All 80 labels remain
blank and the loader correctly refuses evaluation until human review is complete.
No earnings accuracy, annotation agreement or company/quarter labelled subgroup
performance is reported.

Local artifacts are under `outputs/phase5/phrasebank/` and `outputs/phase5/fiqa/`.
Each contains metrics JSON/CSV, confusion CSV, error CSV, subgroup CSV, a readable
report, exclusions CSV and saved model predictions. Error text is omitted by
default; IDs/hashes reference local data. All eight artifacts reproduced
byte-for-byte by rescoring the saved predictions against the same labelled source
cohorts; this verification does not rerun the full model inference. Model
repeatability/offline CPU behavior is separately checked by integration tests.
A separate fresh full FiQA CLI inference run also reproduced all eight artifacts
byte-for-byte, including the complete saved model probabilities.

The executed inference/evaluation runs took about 132 seconds for PhraseBank and
44 seconds for FiQA in this environment. Timings exclude source acquisition and
Python import startup and are not a benchmark comparison between datasets.

Default tests: **209 passed, 7 skipped**. Offline integration enabled: **216
passed**. Confusion totals, supports, distributions, error counts and CSV/JSON
metrics agree. No new packages, GPUs, model downloads or fine-tuning were used.

Remaining limitations: gated/missing genuine FiQA test references, unestablished
independence of FiQA from prior FinBERT research, known PhraseBank training overlap,
task/label differences, simple baseline, uncalibrated scores and pending human
earnings labels. Do not present this work as complete independent generalization
validation. Phase 6 can package the reproducible research and, after human review,
add a modest upload interface; no interface or deployment was implemented here.
