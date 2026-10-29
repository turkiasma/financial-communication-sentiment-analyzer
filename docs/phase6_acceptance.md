# Phase 6 final UI cleanup and end-to-end acceptance

Checked 2026-10-29 with the existing cached CPU-only FinBERT model. No dependencies,
model weights, CUDA packages or third-party documents were downloaded for this
cleanup. Earlier small-fixture acceptance records remain local in
`outputs/phase6/acceptance/`; this record describes the final UI and larger samples.

## Final visible interface

The heading and browser title are exactly **Financial Document Analyzer**. The
repository and internal FinSentiment identifiers are unchanged. Results show total
analyzed segments, positive/negative/neutral percentages, net sentiment, and actual
positive/negative supporting statements with confidence. Neutral statements are
still tracked and can appear in the technical report, but are not displayed as
supporting statements in the UI.

The only two visible download controls are **Download CSV** and **Download PDF Report**,
using native Gradio DownloadButton components with local SVG icons and small sizes.
Both sit below the sentiment percentages; each downloads the current analysis.
There are no report File preview cards, skipped-content counter, skipped-content
section, accordion or third export button in the visible UI.

Internal skipped/rejected content records, accounting, tests and the reporting
API's skipped CSV are preserved. Downloaded PDF retains processing counts
and limitations. If any input content is excluded or PDF pages lack extractable
text, the UI gives a short warning: results cover only processed text and the
user should review the report. Long-input splitting is also disclosed briefly.
No label, score, denominator, baseline or evaluation methodology was changed.

## Original files available for upload

All fixtures are small original software-testing materials under
`tests/fixtures/ui_samples/`:

- `positive_earnings.txt`: ten financial sentences about growth, margins, income,
  cash flow, customer demand and outlook.
- `negative_earnings.txt`: ten sentences about declining revenue, widening losses,
  margins, weak demand, lower guidance and cash difficulties.
- `mixed_earnings.txt`: twelve positive, negative and neutral statements with
  monetary amounts, percentages, negation, FY2025, EPS and a company abbreviation.
- `sample_earnings.pdf`: a one-page original text PDF with six financial sentences
  and an introductory paragraph fragment, giving seven analysis units.
- `image_only_earnings.pdf`: a one-page generated solid image with no text, for the
  unsupported-OCR error case.

The PDFs were generated with the existing PyMuPDF installation. Fixtures are not
reference sentiment labels and must not be used to claim model accuracy.
Authentic Microsoft/Apple texts remain in ignored `data/raw/`; no real release
was invented, recreated or copied into the test fixtures.

## Observed application results

Each category was uploaded through the actual Gradio HTTP client on loopback
port **7860**, with Analyze Document triggered and both exports downloaded.
All cases used the same client session to check independent consecutive uploads.

| Input | Analyzed segments | Positive / negative / neutral count | Positive / negative / neutral % | Net |
| --- | --- | --- | --- | --- |
| Positive earnings TXT | 10 | 10 / 0 / 0 | 100 / 0 / 0 | 1.0000 |
| Negative earnings TXT | 10 | 3 / 7 / 0 | 30 / 70 / 0 | -0.4000 |
| Mixed earnings TXT | 12 | 5 / 3 / 4 | 41.67 / 25 / 33.33 | 0.1667 |
| Microsoft FY2025 Q1 selected narrative TXT | 22 | 17 / 0 / 5 | 77.27 / 0 / 22.73 | 0.7727 |
| Original text PDF | 7 | 3 / 2 / 2 | 42.86 / 28.57 / 28.57 | 0.1429 |

The negative-content fixture produced three positive predictions. Filenames
never determined expected classifications or percentages. These are workflow
observations, not measured accuracy or independent benchmarks.

For every successful response, counts, percentages and net were independently
recomputed from the downloaded CSV. Counts sum to analyzed segments; percentages
sum approximately to 100; net equals `(positive - negative) / analyzed`, with
neutral predictions in the denominator. CSV rows match the analyzed count, remain
in source order, and have actual probabilities in [0,1] summing approximately to
1. Source hashes/IDs and character slices identify the current input. Supporting
statements match current CSV text, labels and confidence. Report summary, CSV
and UI agree, and no previous document's exports are reused.

Mixed content preserved `6.5%`, `$1.35`, `did not`, FY2025, EPS, `$1.28` and `$480`.
Microsoft used the same local selected narrative and policy as Phase 4. All 22
texts and labels match in order; probabilities agree within `1e-6`. This does not
imply that whole-release PDF results would match curated TXT excerpts. It is not
an earnings-specific accuracy measurement.

## Errors and partial/long inputs

Actual app checks covered no file, empty TXT, corrupted PDF, image-only PDF and
invalid fiscal year. Each returned a clear error and empty results with no active
downloads. Unsupported extensions and uploads above 5 MiB were rejected by upload
validation before inference. Normal file replacement first clears the previous
upload; clearing also removes its results and disables exports.

A partly usable TXT (one narrative sentence plus a numeric-only paragraph)
produced one analyzed segment and one internally logged exclusion. The UI omitted
all skipped counts/details and showed the short incomplete-processing warning;
the report retained the actual accounting. A long financial unit yielded three
token-safe segments with no overlap, loss of normalized source text or truncation.
Its CSV/report/UI agreed and the UI disclosed that long text was split.

## Browser rendering and real button clicks

Existing Firefox 156.0.1 was run headlessly in an isolated temporary profile via
WebDriver BiDi, using the already installed Node runtime. No browser automation
package was installed. Rendered screenshots were visually inspected at desktop
1440-pixel width and mobile widths 390 and 320. There is no page-level horizontal
overflow. The two icon buttons measure **140×30** and **180×30** pixels; they align
side by side when space allows and wrap at the narrower width. Supporting text
uses a contained horizontal table scroll rather than an extremely narrow column.
Long-input output was also checked at desktop and mobile widths.

Browser automation uploaded the original positive sample, clicked Analyze Document,
inspected results, and clicked both rendered download buttons. Firefox confirmed
both downloads complete; actual saved files match the HTTP CSV/report for the
same current input. Clearing the upload removed the previous summary/evidence and
disabled the two downloads. Corrupted/image-only PDFs returned visible errors with
no stale results or enabled links. No skipped-content section or counter appeared.
These were automated browser interactions and screenshot inspection, not manual
pointer/touch testing or physical-device testing.

Local ignored evidence is under `outputs/phase6/final_ui/`: `http_acceptance.json`,
`browser-acceptance.json`, per-case downloads, actual browser downloads and desktop/
mobile screenshots. Only the temporary Firefox profile created for these checks
is removed afterwards; unrelated files and caches are untouched.

## Automated verification and remaining limits

Default suite: **273 passed, 13 skipped**. Offline cached-model integration:
**286 passed**, with 25 upstream Gradio/Pandas deprecation warnings. No test failed.
The new loopback HTTP integration test processes all three original TXT samples
and the text PDF through the actual Gradio app, validates exports/results and
checks successful requests followed by a failed one. Its temporary Gradio cache is
isolated from the running app so test cleanup cannot remove live icon/export assets.
Browser checks also verify loaded icon dimensions after analysis. Unit tests explicitly verify
that skipped UI details are absent while internal tracking and warnings remain.

```bash
source .venv-finsentiment/bin/activate
python -m pytest -q
RUN_FINBERT_INTEGRATION=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q
```

TXT/PDF extraction, heuristic sentence boundaries, cross-segment context and
uncalibrated-confidence limitations remain. Scanned PDFs need unsupported OCR;
unusual tables and multi-column reading order require review. Synthetic test
content is not gold-labelled financial evaluation. SEntFiN, FiQA and PhraseBank
results and caveats remain unchanged. No deployment or subsequent-phase work is included in this Phase 6 implementation.


## PDF-only report download update

The report control now reads **Download PDF Report** and returns
`financial_document_report.pdf`; no Markdown file is offered by the UI. CSV bytes,
analysis results, optional metadata, warnings, upload limits and both control
positions/styles are unchanged. The existing PyMuPDF dependency generates A4 pages
using measured text wrapping, embedded Unicode-capable fonts, consistent margins,
a summary table and page numbers. No additional dependency was installed.

The positive fixture and authentic Microsoft FY2025 Q1 release both produced valid
two-page PDFs (approximately 38 KiB and 43 KiB). Header metadata, all class counts/
percentages, net score, supporting text/confidence and limitations were checked
against the current CSV/UI. CSV downloads match the previous Markdown-era files
byte-for-byte. A longer mock formatting fixture spans at least three pages; text
bounds, complete evidence, common Unicode financial symbols and page numbers are
checked automatically. PDF pages were rendered and visually inspected. The actual
Gradio CSV/PDF controls were clicked in Firefox, saved files were validated, and
failure cleared both downloads. New evidence stays ignored under
`outputs/phase6/pdf_reports/`; earlier acceptance artifacts are preserved.

Only report generation, its download label, relevant tests and documentation were
changed. Markdown generation remains available internally. Font coverage includes
common financial symbols and punctuation; unsupported rare glyphs produce a clear
report error rather than silently substituting text. The prior extraction and
classification limitations remain, and no uploaded-document accuracy is claimed.
