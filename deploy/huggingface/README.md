---
title: Financial Document Analyzer
emoji: 📄
sdk: gradio
sdk_version: "6.31.0"
python_version: "3.14.7"
app_file: app.py
suggested_hardware: cpu-basic
models:
  - ProsusAI/finbert
short_description: Financial language sentiment with TXT/PDF uploads and CSV/PDF reports
---

# Financial Document Analyzer

Analyze English financial communications with the pinned `ProsusAI/finbert` model
on CPU. Upload UTF-8 TXT or a text-based PDF; optionally supply company and fiscal
metadata. View predicted sentiment percentages, net sentiment and supporting
statements, then download sentence-level CSV and an A4 PDF report.

**Privacy:** This hosted service processes files on its server, not on your device.
Use public, non-confidential demonstration documents only. Uploads and generated
reports are temporarily cached; this is not a confidential document-processing or
secure-erasure service. Files may remain until scheduled cleanup or restart.
No authentication or persistent document storage is provided.

Limits: 5 MiB per document, 30 PDF pages, 500,000 extracted characters and 2,000
analysis units. Scanned/image-only PDFs need unsupported OCR. Extraction order,
tables and sentence/segment boundaries can be imperfect. Long units are split
without truncation; percentages count the resulting classified segments.

Percentages describe predicted text labels, **not classification accuracy**.
Confidence is uncalibrated and predictions can be wrong. Sentiment does not
measure financial health or investment value and is not investment advice.

The strongest external evaluation was a 2,000-example human-labelled SEntFiN
headline sample (74.75% accuracy, macro F1 0.7496). It is not an official held-out
partition, all prior model exposure cannot be excluded, and its results do not
measure accuracy on uploaded earnings releases. FiQA remains an original-train
diagnostic; PhraseBank has known fine-tuning overlap. No pooled accuracy is used.

The first startup may download and initialize the pinned public checkpoint.
Subsequent requests reuse one CPU model. Model/cache files can be lost with the
Space's ephemeral runtime storage. The model is not included in this repository.

Before publishing, review the pinned model card's declared licence
([CC-BY-NC-SA-4.0](https://huggingface.co/ProsusAI/finbert/blob/0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031/README.md))
and PyMuPDF's AGPL/commercial terms for your intended use. This template does not
assign a new licence to third-party software or model weights.
