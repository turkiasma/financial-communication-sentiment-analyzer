# Financial document and evaluation data plan

Source pages checked on **2026-10-29**. This contains the wider source shortlist
and a four-release Phase 3 demonstration. Source text, snapshots and generated
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

## Phase 3: four-release local demonstration

The first two fiscal periods for each company were rechecked and processed:

| Document ID / metadata | Fiscal period | Published | Selected sentences classified |
| --- | --- | --- | --- |
| [msft_fy2025_q1](../data/metadata/msft_fy2025_q1.json) | Microsoft FY2025 Q1 | 2024-10-30 | 3 |
| [msft_fy2025_q2](../data/metadata/msft_fy2025_q2.json) | Microsoft FY2025 Q2 | 2025-01-29 | 4 |
| [aapl_fy2025_q1](../data/metadata/aapl_fy2025_q1.json) | Apple FY2025 Q1 | 2025-01-30 | 6 |
| [aapl_fy2025_q2](../data/metadata/aapl_fy2025_q2.json) | Apple FY2025 Q2 | 2025-05-01 | 5 |

Official source URLs and exact quarter-end dates are in the table above and each
metadata file. These are fiscal periods; Microsoft FY2025 Q1 was published in
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

### Manual preparation on another checkout

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

## Future labelled evaluation input — Phase 5 only

The existing `normalize_text` function and token counter can be reused on already
labelled sentences, without re-segmenting them or assigning invented company and
quarter metadata. Plan a **separate** evaluation-record schema with `dataset_id`,
`dataset_revision`, `split`, `sample_id`, `text`, `original_text`, `original_label`
and a documented `mapped_label`. Preserve target/aspect, source category and
agreement information where supplied. Gold labels must remain separate from
model predictions. This adapter and all evaluation metrics remain unimplemented.

- [Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank):
  inspect the selected annotator-agreement configuration, label representation,
  duplicates and data availability before defining a split. The card identifies
  CC BY-NC-SA 3.0 and currently reports a legacy dataset-script compatibility
  issue. [FinBERT's model card](https://huggingface.co/ProsusAI/finbert) identifies
  PhraseBank as fine-tuning data; results must not be presented as independent
  unseen-data evaluation.
- The requested [FiQA URL](https://huggingface.co/datasets/TheFinAI/fiqa-sentiment-classification)
  currently redirects to [TheFinAI/en-fiqa-absa](https://huggingface.co/datasets/TheFinAI/en-fiqa-absa).
  The visible card requires an access form, describes aspect-based sentiment,
  and lists sentence, target, aspect, score and headline/post type fields. The
  actual gated files and label values have **not** been inspected. Confirm the
  exact dataset/version, permitted use, split definitions, target semantics and
  score/label representation before deciding a three-class mapping. Do not
  assume its numerical indices match FinBERT, and do not invent score thresholds.

Earnings releases have no gold sentiment labels by default. They can support
later communication-tone comparisons but are not automatically evaluation data.
No dataset download, label conversion, metrics or fairness analysis is part of
Phase 2.
