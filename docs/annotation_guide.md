# Human annotation of earnings statements

Status: **pending human annotation/review**. No earnings reference labels have
been assigned by the pipeline or copied from FinBERT.

The local template `data/evaluation/annotations/earnings_sample.csv` contains 80
distinct original financial text units from the eight existing earnings releases.
Sampling uses Phase 2 records, not predictions: normalize Unicode/case/whitespace
only to deduplicate, retain the lowest stable sentence ID, shuffle with seed 42,
and sample round-robin across documents until 80 units are selected. Shuffle the
presentation order with seed 43. This balances document coverage subject to each
document's available distinct text; it is not a random estimate of the full corpus
or a sentiment-balanced dataset. Source/quarter identifiers remain visible, but
FinBERT labels and confidence are absent. The template and source text stay local.

## Instructions for the human reviewer

Read the **entire** text unit and judge its explicitly stated financial meaning
for the named company, rather than writing style, enthusiasm or predicted stock
returns. Use source context only to resolve references, and note when you do so.

- **positive:** a favourable financial/operating development, improved results,
  growth, stronger margins, or an explicitly favourable outlook for the target.
- **negative:** an adverse development, weakening results, higher harmful costs,
  losses, deteriorating margins, reduced guidance or an explicitly adverse outlook.
- **neutral:** factual/administrative information without an explicit favourable
  or adverse implication; also genuinely balanced financial statements when no
  direction predominates. Do not use neutral merely because you are unsure.

Preserve negation, comparisons, quantities and which entity is affected. Higher
expenses are not automatically positive because they "increased"; increased
investment is not automatically negative either. Classify what the statement
actually establishes and record ambiguity in notes. Corporate promotional words
and high confidence from a model are not reference evidence.

Fill `human_label` with exactly `positive`, `negative` or `neutral`, supply your
`annotator_id`, and set `review_status` to `reviewed` only after checking the label.
Use `annotation_notes` for ambiguity, mixed targets, source context, or a fragment
whose meaning cannot be judged alone. Leave unresolved items `pending`; do not
guess a label to complete the file. Keep IDs, text and source columns unchanged.
The evaluation loader rejects pending or unlabelled rows rather than silently
excluding them. If irreducible exclusions are needed, define and record that
policy before producing an evaluation cohort.

Do not consult FinBERT predictions until your independent labels are recorded.
If a second reviewer becomes available, annotate independently before resolving
disagreements. No annotation agreement has been measured or claimed here.

## Prepare and eventually evaluate

```bash
python -m src.evaluation_data prepare-annotations \
  --input outputs/phase4/real_sentences.jsonl \
  --output data/evaluation/annotations/earnings_sample.csv --size 80 --seed 42

# Only after every reference label has genuinely been reviewed:
python -m src.evaluation --dataset earnings \
  --annotations data/evaluation/annotations/earnings_sample.csv \
  --output-dir outputs/phase5/earnings --local-files-only
```

Preparation refuses to overwrite an existing template, protecting user edits.
No in-domain accuracy, subgroup performance or company/quarter evaluation should
be reported until these human references exist. The 128 model-classified earnings
units from Phase 4 are not a labelled test set.
