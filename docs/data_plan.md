# Financial document and evaluation data plan

Source pages checked on **2026-10-29**. This contains the wider source shortlist
and the expanded eight-release Phase 4 collection (with the original Phase 3
demonstration retained below as history). Source text, snapshots and generated
predictions are kept locally in ignored directories; no third-party document text
or evaluation dataset is tracked in Git. Links and metadata below were checked
against official pages.

## Candidate earnings releases

Use English quarterly **earnings releases** (`document_type=earnings_release`)
from one company over four consecutive fiscal periods. Both candidates have
official releases accessible without a login. Apple Investor Relations links
to its Newsroom releases; Microsoft hosts its releases directly on Investor
Relations. Fiscal quarters and publication dates are distinct:

| Company (ticker) | Fiscal period | Quarter ended | Published | Official release |
| --- | --- | --- | --- | --- |
| Microsoft (MSFT) | FY2025 Q1 | 2024-09-30 | 2024-10-30 | [FY25 Q1](https://www.microsoft.com/en-us/investor/earnings/fy-2025-q1/press-release-webcast) |
| Microsoft (MSFT) | FY2025 Q2 | 2024-12-31 | 2025-01-29 | [FY25 Q2](https://www.microsoft.com/en-us/investor/earnings/fy-2025-q2/press-release-webcast) |
| Microsoft (MSFT) | FY2025 Q3 | 2025-03-31 | 2025-04-30 | [FY25 Q3](https://www.microsoft.com/en-us/investor/earnings/fy-2025-q3/press-release-webcast) |
| Microsoft (MSFT) | FY2025 Q4 | 2025-06-30 | 2025-07-30 | [FY25 Q4](https://www.microsoft.com/en-us/investor/earnings/fy-2025-q4/press-release-webcast) |
| Apple (AAPL) | FY2025 Q1 | 2024-12-28 | 2025-01-30 | [Q1 results](https://www.apple.com/newsroom/2025/01/apple-reports-first-quarter-results/) |
| Apple (AAPL) | FY2025 Q2 | 2025-03-29 | 2025-05-01 | [Q2 results](https://www.apple.com/newsroom/2025/05/apple-reports-second-quarter-results/) |
| Apple (AAPL) | FY2025 Q3 | 2025-06-28 | 2025-07-31 | [Q3 results](https://www.apple.com/newsroom/2025/07/apple-reports-third-quarter-results/) |
| Apple (AAPL) | FY2025 Q4 | 2025-09-27 | 2025-10-30 | [Q4 results](https://www.apple.com/newsroom/2025/10/apple-reports-fourth-quarter-results/) |

The [Apple Investor Relations directory](https://investor.apple.com/investor-relations/)
provides the quarterly report links. Microsoft Q4 also contains annual results;
keep that annual section separate from quarterly narrative if making a later
comparison. Compare the same types of narrative sections across periods: opening
quarterly overview and management remarks, for example. Do not mix whole
transcripts with short announcements, or table rows with narrative sentences.

## Access, provenance and redistribution

Public readability does not establish redistribution permission. This shortlist
does not assert permission to publish full text or derived sentence collections.

- [Microsoft terms, Documents section](https://www.microsoft.com/en-us/legal/terms-of-use)
  describe conditional informational/noncommercial or personal use, preservation
  of notices, no modification, and restrictions on network posting. These terms
  do not give this repository a general license to redistribute extracted text.
- [Apple website terms](https://www.apple.com/legal/internet-services/terms/site.html)
  restrict copying/publication/distribution and automated page acquisition, with
  specific exceptions. No general redistribution license for these releases was
  identified in the pages reviewed.

For either candidate, check the applicable terms and permissions for the intended
acquisition, extraction and publication before assembling a corpus. Keep notices
with originals; keep third-party text and processed text out of Git. Commit only
links, independently recorded metadata, code and original synthetic fixtures
unless redistribution permission is established. No scraper, crawler or downloader
is part of the application.

When lawful local text is prepared, place it in ignored `data/raw/`. Record the
official URL, exact fiscal period, publication date and document type. In `extra`,
record retrieval date, quarter-end date, section scope, extraction method,
permission/terms notes, any removed material and an original source hash if
available. The generated `source_sha256` identifies the local TXT bytes, **not**
the original HTML/PDF or the authenticity of its financial claims. Character
offsets refer to this TXT. Retain the original local input to reproduce them.
Keep UTF-8 encoding and preprocessing version/tokenizer revision with results.

The supplied fixture is original fictional text and explicitly marked synthetic.
Its example.com URL is a placeholder, not financial evidence.

## Phase 3: original four-release local demonstration

The first two fiscal periods for each company were rechecked and processed:

| Document ID / metadata | Fiscal period | Published | Selected sentences classified |
| --- | --- | --- | --- |
| [msft_fy2025_q1](../data/metadata/msft_fy2025_q1.json) | Microsoft FY2025 Q1 | 2024-10-30 | 3 |
| [msft_fy2025_q2](../data/metadata/msft_fy2025_q2.json) | Microsoft FY2025 Q2 | 2025-01-29 | 4 |
| [aapl_fy2025_q1](../data/metadata/aapl_fy2025_q1.json) | Apple FY2025 Q1 | 2025-01-30 | 6 |
| [aapl_fy2025_q2](../data/metadata/aapl_fy2025_q2.json) | Apple FY2025 Q2 | 2025-05-01 | 5 |

These counts describe the Phase 3 `opening_ceo_cfo_statements_v1` run; current
metadata files now describe the broader Phase 4 extraction below. Original Phase 3
TXT/predictions remain local, and its metadata can be recovered from the Phase 3
Git commit. Official source URLs and exact quarter-end dates are in the table
above and each metadata file. These are fiscal periods; Microsoft FY2025 Q1 was published in
calendar 2024. None of the required source metadata was missing or uncertain.

### Consistent extraction scope

Policy `opening_ceo_cfo_statements_v1`: select the **complete first CEO statement
paragraph and complete first CFO statement paragraph** from the opening results
narrative, in source order. Include every sentence in those paragraphs, including
product commentary and any challenging outlook they contain. Do not filter on
sentiment, product name, financial number or predicted label. Exclude other
paragraphs uniformly: quantitative bullet/table summaries, business highlights,
dividend notices, separate outlook sections, webcast logistics, annual summaries,
corporate descriptions, legal boilerplate, navigation and notices.

This is a deliberately narrow, comparable management-commentary demonstration.
It does **not** claim to cover the full earnings release, all risks, or all guidance.
A future broader scope should be defined explicitly and applied consistently
before period comparisons. The different companies' fiscal calendars also differ.

Acquisition was a one-time retrieval of the four explicitly selected official
pages, with no crawling or discovery automation. Both sites returned accessible
HTML without authentication or access bypass. Complete source snapshots remain
in ignored `data/raw/source_html/` with their notices intact. The two statement
blocks were selected from those snapshots, HTML entities decoded, whitespace
collapsed and a blank line placed between paragraphs. The local TXT contains only
those two original statement paragraphs; no text was generated or paraphrased.
No acquisition script or third-party text was added to tracked project files.

Microsoft statement blocks were HTML `p` elements, zero-based block indices 6 and
7 in each saved page. Apple statement blocks were `div.pagebody-copy`, indices 1
and 2 in each saved page. These positions describe the saved snapshots and should
not be assumed stable if websites change. Each manifest records the source HTML
hash, extracted TXT hash, retrieval date, quarter-end date, block selection,
selection policy, extraction exclusions and rights notes in `extra`.

The local end-to-end run produced 18 sentence records and 18 CPU FinBERT
predictions, with no skips or errors. All metadata and source spans were checked,
and prediction probability sums were validated. Results are in ignored
`outputs/phase3/real_predictions.jsonl`, per-document JSONL files and a small
`execution_summary.json`. This is an execution demonstration, not an accuracy
evaluation or quarterly sentiment comparison.

### Historical Phase 3 manual preparation

This describes the original narrow extraction. For the current manifests, use
the Phase 4 instructions below and their broader selection policy.

1. Open the four official URLs using the links above and check applicable terms
   for your intended use. Preserve original notices with any retained originals.
2. Identify the first CEO and first CFO statement paragraphs in the opening
   narrative. Include the full paragraphs without selecting individual sentences.
3. Save the selected text as UTF-8 in `data/raw/<document_id>.txt`, keeping the
   paragraphs in source order, separated by a blank line. Do not include headings
   or tables. Metadata `file_path` is relative to `data/metadata/`.
4. Confirm source facts in the metadata against the release. For your own
   extraction, update `extra.retrieved_at`, extraction method, source/TXT hashes,
   selection details and rights notes to match the files you actually obtained;
   do not reuse this run's hashes if your bytes differ. Keep document IDs unique.
5. Run `python -m src.preprocessing --metadata data/metadata/<document_id>.json
   --output outputs/<document_id>_sentences.jsonl`, then
   `python -m src.batch_inference --input outputs/<document_id>_sentences.jsonl
   --output outputs/<document_id>_predictions.jsonl --batch-size 8 --local-files-only`.
6. Inspect sentence boundaries, CLI skip counts and any skipped-record companion
   file. Confirm `original_text` and offsets against the local TXT. Never claim
   classification for a sentence that was skipped.

Real-text availability is intentionally local: a fresh clone contains metadata
and instructions, not the source text or derived predictions. Public access did
not establish redistribution permission. No text or model weights need to be
downloaded again on this laptop; no additional dependencies were installed.

## Phase 4: eight-release consistent financial narrative collection

All eight official releases in the first table were verified and processed.
Four new releases (each company's Q3 and Q4) were acquired as small, individually
selected HTML snapshots. Existing Q1/Q2 snapshots were reused. No broad scraper,
bulk downloader, model download or dataset download was added. Source dates and
fiscal periods match the official pages; no required source metadata was missing.

| Manifest (official URL is also in the inventory above) | Classified units | Positive / negative / neutral |
| --- | --- | --- |
| [Microsoft FY2025 Q1](../data/metadata/msft_fy2025_q1.json) | 22 | 17 / 0 / 5 |
| [Microsoft FY2025 Q2](../data/metadata/msft_fy2025_q2.json) | 23 | 18 / 1 / 4 |
| [Microsoft FY2025 Q3](../data/metadata/msft_fy2025_q3.json) | 22 | 18 / 0 / 4 |
| [Microsoft FY2025 Q4](../data/metadata/msft_fy2025_q4.json) | 22 | 18 / 0 / 4 |
| [Apple FY2025 Q1](../data/metadata/aapl_fy2025_q1.json) | 10 | 6 / 0 / 4 |
| [Apple FY2025 Q2](../data/metadata/aapl_fy2025_q2.json) | 10 | 8 / 0 / 2 |
| [Apple FY2025 Q3](../data/metadata/aapl_fy2025_q3.json) | 8 | 5 / 0 / 3 |
| [Apple FY2025 Q4](../data/metadata/aapl_fy2025_q4.json) | 11 | 7 / 0 / 4 |

Policy **`financial_narrative_v2`** was applied to every release, including the four
old ones; no version-1 excerpt was appended to the new collection. All complete
selected blocks are retained without sentiment filtering. Duplicate blocks within
a document are removed; repeated commentary across different quarters remains
part of each source. The current manifests record the original HTML hash,
extracted TXT hash, retrieval date, quarter-end, selected blocks/sections,
company-specific comparison profile and extraction limitations.

### Selection boundaries and differing company formats

- **Microsoft:** select the datelined quarterly overview through the business
  outlook paragraph immediately preceding `Quarterly Highlights, Product Releases,
  and Enhancements`. This includes opening revenue/income/EPS text bullets,
  complete CEO/CFO statements, business-segment narrative and all its financial
  bullets, shareholder-return commentary, and the outlook paragraph. Remove
  headings. For Q4, omit the separate `Fiscal Year 2025 Results` section up to
  `Business Outlook`, so annual summary bullets are not mistaken for quarterly
  results. Opening Q4 CEO/CFO remarks can still include annual figures; they are
  kept in context and this limitation is recorded. The selected HTML `p` indices
  are in each manifest; they total 20 blocks per quarter.
- **Apple:** select the opening quarterly results overview, complete CEO/CFO
  paragraphs, and dividend/share-repurchase narrative. Stop before the webcast
  logistics paragraph. These are the first four `div.pagebody-copy` blocks in each
  saved source. The source's dateline location label remains with the overview.
  Q4 management remarks also refer to full-year performance and are retained.
- **Both:** exclude numerical tables, navigation, headings, webcast logistics,
  unrelated product-release lists, repeated corporate descriptions and legal
  boilerplate. No stand-alone legal risk list is treated as operational narrative.
  Relevant challenges or outlook language inside selected sections is kept.

The common top-level policy has distinct comparison profiles:
`msft_quarterly_financial_narrative_v2` and
`aapl_quarterly_financial_narrative_v2`. Microsoft has additional segment highlights
and an outlook paragraph that refers to guidance on a separate call; Apple has no
equivalent sections in the selected opening release body. No call transcript or
PDF was substituted. The reports compare quarters **within a company**, not the
two companies' non-equivalent structures as if they were identical.

### Manual extraction and reproduction

1. Open each of the eight official links in the inventory. Check applicable use
   terms; retain original notices with originals. Current local snapshots are in
   ignored `data/raw/source_html/`; no new acquisition is needed on this laptop.
2. Select the complete blocks defined above. Decode HTML entities and collapse
   whitespace within each block, without changing its words, numbers, negation or
   punctuation. Join blocks in source order with a blank line and end with a newline.
   Do **not** add periods to unpunctuated financial bullets.
3. Save UTF-8 text at `data/raw/phase4/<document_id>.txt`. Review source order,
   duplicate removal and section exclusions. Keep originals and extracted text
   separate. On another machine, update retrieval/method/selection notes and
   hashes in the manifests to match the bytes actually prepared; source HTML may
   change. Do not reuse a hash from this run for a different local extraction.
4. Run Phase 2 preprocessing version 2 on each manifest with
   `--long-sentence-policy keep`. This preserves blank-line paragraph boundaries,
   so unpunctuated financial bullets remain separate text segments with exact
   source offsets. The output column name `classified_sentences` includes these
   documented units; this is not an assertion that each bullet is a grammatical
   sentence. Reprocess **all** quarters when changing scope or preprocessing.
5. Combine the sentence JSONL in a fixed document order if running one batch job,
   preserving IDs, then run Phase 3 with `--batch-size 8 --local-files-only`.
   Keep its skipped companion file if created. Predictions already exist locally
   at `outputs/phase4/real_predictions.jsonl`; Phase 2 input is
   `outputs/phase4/real_sentences.jsonl`.
6. Run `python -m src.analysis --input outputs/phase4/real_predictions.jsonl
   --source-input outputs/phase4/real_sentences.jsonl --output-dir outputs/phase4/my_report`.
   Use a fresh output directory. The CLI discovers `<input>.skipped.jsonl`, or
   accepts `--skipped-input`. Full source-ID reconciliation is required for verified
   coverage; absent outputs are errors. Without Phase 2 input, total source size
   and coverage are explicitly unknown, not assumed complete.

The executed collection has **128 classified units, 0 skips, 8 documents, 8
company-periods and 6 adjacent-quarter comparisons**, with source-ID accounting
verified. Reports are local under `outputs/phase4/final_report/`; complete source
and derived text remain ignored by Git. Corporate language is predominantly
positive/neutral in this selected sample; missing negative categories are reported
as absent rather than supplied with fabricated evidence.

Samples of 8–23 units per release are small and dependent within documents.
Broadening extraction improves coverage over the original 18-unit demonstration
but does not establish representativeness, statistical significance, financial
performance, stock-price effects or model accuracy. Copyright/access restrictions
above still apply; reports containing exact excerpts are not approved public
redistribution artifacts merely because they were generated locally.

## Phase 5 labelled evaluation — implemented with limitations

See the executed [evaluation protocol](evaluation_methodology.md),
[findings](evaluation_findings.md), and [human annotation guide](annotation_guide.md).
The evaluation schema preserves `example_id`, `text`, `original_text`,
`true_label`, `original_label`, `source_dataset`, `split` and available target/aspect
and source-type metadata. Gold references remain separate from predictions.

- [Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank):
  the pinned original ZIP is read directly, avoiding legacy dataset-script or
  Arrow dependencies. Only AllAgree is scored: 2,264 original rows, 2,259 after
  five same-label duplicates. CC BY-NC-SA 3.0 applies. Known FinBERT fine-tuning
  overlap is prominently disclosed; no independent held-out claim is made.
- The requested [FiQA URL](https://huggingface.co/datasets/TheFinAI/fiqa-sentiment-classification)
  redirects to gated [TheFinAI/en-fiqa-absa](https://huggingface.co/datasets/TheFinAI/en-fiqa-absa).
  Those gated files were not accessed, and original organizer downloads returned
  404. A pinned original-format participant repository supplies labelled training
  files and unlabelled test inputs. The implemented diagnostic uses 1,052
  single-target train examples after 60 multi-target and one train/test overlap
  exclusions. Published PIXIU thresholds were fixed before inference; original
  scores/targets are preserved. No official test accuracy is fabricated.
- The [original FinBERT paper](https://arxiv.org/html/1908.10063) also used FiQA
  regression experiments. The released classifier's exact exposure is unknown;
  this remains an external-source candidate diagnostic, not guaranteed unseen
  generalization evidence. A check against PhraseBank 50Agree found no eligible
  FiQA text match; it cannot rule out near-duplicates or pretraining contamination.
- Eighty original earnings units were sampled without consulting predictions,
  but all human labels remain blank. No in-domain accuracy or annotation agreement
  is claimed. The loader rejects pending annotations.

Raw data, annotation text and generated reports stay in ignored
`data/evaluation/` and `outputs/phase5/`. Only the metadata-only
`data/evaluation_manifests/phase5_sources.json`, code and documentation are intended
for Git. It records source URLs/revisions/hashes, retrieval dates and the pending
annotation protocol. No dataset package, additional model or GPU dependency was
installed. Demographic fairness metrics remain out of scope.
