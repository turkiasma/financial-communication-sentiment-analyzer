# Financial document and evaluation data plan

Source pages checked on **2026-10-29**. This is a source shortlist, not an ingested
dataset. No third-party financial documents or evaluation datasets are included
in the repository. Links and metadata below were checked against official pages.

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
unless redistribution permission is established. There is no scraper in Phase 2.

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
