# Phase 5B external benchmark audit

Checked 2026-10-29. Four candidates were investigated: the two requested sources
and **only two alternatives**. No candidate was accepted based on a newer date
alone. FinBERT weights and the Phase 5 keyword baseline remain unchanged.

## Eligibility decisions

| Candidate | Access / licence | Reference provenance | Decision |
| --- | --- | --- | --- |
| [ARSD](https://wp2025.cs.hku.hk/fyp25036/datasets/) | Description/report accessible; no record download or explicit dataset licence found | Authors claim five finance students and majority voting; records cannot be checked | Unavailable for evaluation; promising but not accepted |
| [Mendeley financial headlines](https://data.mendeley.com/datasets/fk37gb6dmp/1) | Files publicly downloadable, CC BY 4.0 with third-party caveat | Model-generated voting labels and generated paraphrases | Not suitable as independent human ground truth; not scored |
| [Twitter Financial News Sentiment](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment) | CSVs public, source card MIT | Annotated, but annotation procedure/people not established; metadata says `annotations_creators: other` | Diagnostic-only provenance; not selected or scored |
| [SEntFiN 1.0](https://github.com/pyRis/SEntFiN) | Author CSV and MIT licence public | Human entity-sentiment labels documented in the [paper](https://arxiv.org/html/2305.12257), section 3.1 | Accepted **primary external sample**, restricted to single-entity headlines and screened against known references |

An accepted external sample is not mathematically guaranteed free from all
pretraining or downstream research exposure. The sentence-level restriction and
sampling policy are limitations, not a claim of the authors' full benchmark score.

## ARSD: requested highest-priority source

The official dataset page describes 3,000 annual-report sentences. Its acquisition
flowchart and [final report](https://wp2025.cs.hku.hk/fyp25036/wp-content/uploads/sites/37/Final-Report.pdf),
section 3.2.1, describe five finance students, majority labels, and extraction from
annualreports.com (Items 1, 1A and 7). These are publisher claims, not annotations
verified at record level. The project report is dated 2026-04-20; source-report
years and a dataset publication date were not established.

The dataset page, project home, final-report page/report, and results page were
inspected. Their accessible links offered external PhraseBank/FiQA sources,
figures and reports, but no ARSD CSV/JSON/archive, source repository or explicit
data licence. The report contains no identified GitHub link or data-use licence.
No access control was bypassed, account created or author contacted. Thus actual
labels/columns, class counts, duplicates, overlap and permitted local use cannot
be verified. No adapter, substitute text or performance metrics were fabricated.

## Mendeley: annotation provenance fails the gold-reference requirement

DOI `10.17632/fk37gb6dmp.1`, version 1, published **2024-08-22**. The public file
listing is available at
`https://data.mendeley.com/public-api/datasets/fk37gb6dmp/files?folder_id=root&version=1`.
Initial guessed version-route endpoints returned 404; the actual public listing
then provided the three advertised files. Downloads were small and their SHA-256
hashes matched the listing. Notebooks were **read as JSON, never executed**.

`Data Labeling Code.ipynb` loads `yiyanghkust/finbert-tone`,
`nickmuchi/finbert-tone-finetuned-fintwitter-classification`, and
`mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis`. Code cells
55–59 count votes and select the largest count, resolving ties to neutral. The
notebook also contains generative-model experiments and uses
`humarin/chatgpt_paraphraser_on_T5_base` to augment positive/negative rows while
retaining assigned labels. The exact string `ProsusAI/finbert` was **not** found
in the labelling code: the circularity finding concerns model-generated reference
labels including other FinBERT variants, not a claim that our exact checkpoint
generated every label. No additional models were downloaded or executed.

The spreadsheet `final_sent_headline_df_Balanced.xlsx` has columns `Sno`,
`Headlines`, `sentiment_label` and **18,617 rows**: 6,130 negative, 6,123 neutral,
6,364 positive. There are 6,423 extra exact-duplicate rows, 6,433 extra normalized
duplicate rows, and 2,937 normalized-text groups with conflicting labels. The
source is financialexpress.com/market; exact collection dates are not supplied
as spreadsheet metadata. Zero normalized unique matches were found against
available PhraseBank 50Agree and original FiQA text. That does not remedy label
provenance. T5 augmentation establishes a near-duplicate risk; exhaustive semantic
pair counts were not computed for this rejected dataset. No accuracy is reported.

## Alternative 1: Twitter Financial News Sentiment

Pinned Hugging Face revision `ccbe24de388e287beb92dd393a335c376b350ac3`.
The CSV schema is `text,label`, with 0=bearish/negative, 1=bullish/positive,
2=neutral. Actual file counts, duplicates, cross-split overlap and reference
overlap are recorded in the structured audit manifest. The card's stated split
sizes do not match the actual files. It mentions Twitter API collection but does
not document an annotation procedure or collection period. Other models' claims
that this corpus is human-labelled do not establish original annotation provenance.
It is not accepted as independent human gold or assigned an evaluation score.

## Alternative 2: SEntFiN, accepted with explicit scope

Pinned author repository revision `eba43610fe3cb57a3e5773cd6c037da1f8992ad5`;
CSV SHA-256 `570af7ba58f724f3fcc414dd5ba80da5b1a3b342134d76c6ebdf53a32c54a92f`.
The repository's MIT licence names the four authors. The paper documents three
MBA annotators, consensus and curator adjudication. The annotated headlines are
from Economic Times **2011–2015**, distinct from the paper's separate 2002–2017
news-flow corpus. The CSV lacks per-headline dates or an official train/test split.
The public repository existed in 2021; the arXiv paper was posted in 2023. These
dates alone are not evidence of independence.

Schema: `S No.`, `Title`, `Decisions`, `Words`. The original 10,753 rows include
2,847 multi-entity rows, excluded before prediction. Named reference labels are
positive/negative/neutral. Thirteen decision strings contain unescaped apostrophes
in entity keys; a strict full-coverage string-pair parser reads those original
names without executing code or guessing labels. Unknown labels or malformed
records cause errors. Original decision strings and target names are preserved.

There are 7,906 single-entity rows. Duplicate/conflict filtering retains 7,846:
44 same-label extra rows and all 16 rows in conflicting-reference groups are
excluded. Before sampling, supports are negative 2,363, neutral 2,654, positive
2,829. All eligible text is checked for exact and NFKC/case/whitespace-normalized
matches against **all accessible PhraseBank 50Agree and original FiQA train/test
texts** (6,151 reference rows): zero matches were found.

A bounded word-set Jaccard >=0.9 screen checks five rare reference tokens, up to
200 candidates per input, and caps reporting at 2,000 pairs. It finds 15 internal
candidate pairs and no cross-reference candidate pairs. This is not exhaustive
semantic or pretraining overlap detection. Candidates are reported, not deleted:
changing a financial number or negation can change meaning. No independence claim
rests solely on that screen. The original SEntFiN study trained its own finBERT
variant; we use the unchanged pinned ProsusAI classifier, not that model. Its
complete training/pretraining history cannot be audited from text matches alone.

## Protocol fixed before external predictions

Use **2,000 examples**, proportionally stratified over the 7,846 eligible records
using largest-remainder quotas and seed **42**. Allocation is negative **602**,
neutral **677**, positive **721**. Selection uses original reference classes only,
never model outputs. The 5,846 eligible but unsampled records are logged separately
from invalid/duplicate exclusions. Results describe this sample, not all 10,753
rows, all 7,846 eligible rows, annual reports, or an author-defined held-out test.

The frozen Phase 5 `keyword_v1` baseline, metric definitions, label order,
subgroup cutoffs and cached ProsusAI weights are unchanged. Token-limit exclusions
apply equally to both models. No tuning or fine-tuning is performed. Negative
recall and class supports receive explicit attention. All raw content stays in
ignored `data/evaluation/`, and generated artifacts in `outputs/phase5b/`.

The abandoned earnings-template workflow, guide, CLI options, loader and tests
were removed. The local file was checked to be entirely blank/pending before
deletion. No earnings gold labels were ever produced. Historical Phase 5 results
remain intact; no earlier Git commit, real release, earnings-release metadata or Phase 4
artifact was rewritten or deleted.

## Execution outcome

Both unchanged models evaluated all 2,000 selected records with zero overlong
exclusions. FinBERT accuracy/macro F1 are 0.7475/0.7496; the baseline's are
0.5715/0.5612. Negative recall is 454/602 versus 224/602. No flagged internal
near-duplicate pair has both members in the selected sample. The full counts,
confusions, actual error examples and limitations are in
[findings](evaluation_findings.md); machine-readable source metadata and audit
counts are in `data/evaluation_manifests/phase5b_sources.json`.

## Finalization: supported datasets and local cleanup

The original investigation above and metadata-only candidate file hashes are
preserved. The supported retrieval/evaluation datasets are now **SEntFiN**
(primary external sample), **FiQA** (original-train secondary diagnostic) and
**PhraseBank** (known training-overlap diagnostic). No combined accuracy is used.

Rejected-candidate summary: **ARSD** lacked downloadable records and a verified
dataset licence; **Mendeley** used classifier-voted labels and generated
paraphrases; **TFNS** lacked verified human annotation provenance. None was scored.
Their runtime eligibility registry/CLI entries and rejection-specific tests were
removed; no dedicated loaders existed. Unused local ARSD figures, Mendeley
notebooks/spreadsheet and TFNS files were deleted, freeing 4,237,098 bytes.
Historical file paths/hashes in the manifest record the investigation, not an
expectation that rejected-candidate downloads remain installed. The three retained
datasets, overlap-reference files and all shared evaluation logic remain intact.
