"""Local-only Gradio document analyzer. Run with python -m src.app."""

import os
import re
from pathlib import Path
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr

from src.document_processing import analyze_document, MAX_UPLOAD_BYTES
from src.reporting import AnalysisDownloads


def clear_results(previous=None):
    if previous is not None:
        previous.cleanup()
    return "", "", [], *[gr.update(value=None, interactive=False) for _ in range(2)], "Ready for a new document.", None


def analyze_upload(upload, company, ticker, fiscal_year, fiscal_quarter, previous=None):
    """Clear stale outputs before work; a failure returns no old downloads."""
    cleared = clear_results(previous)
    yield (*cleared[:5], "Analyzing on CPU…", None, *[gr.update(interactive=False) for _ in range(6)])
    downloads = None
    try:
        if not upload:
            raise ValueError("Choose a TXT or text-based PDF document first.")
        result = analyze_document(upload, {"company": company, "ticker": ticker,
                                          "fiscal_year": fiscal_year, "fiscal_quarter": fiscal_quarter})
        downloads = AnalysisDownloads(result)
        summary = result["summary"]
        text = (f"**Total analyzed segments:** {result['classified_units']}\n\n"
                f"**Net sentiment: {summary['net_sentiment']:.4f}**\n\n"
                "Net = (positive − negative) / analyzed segments, including neutral segments. "
                "Percentages show predicted segment labels, not prediction accuracy.")
        bars = ""
        for label, color in (("positive", "#27835e"), ("negative", "#bd4747"), ("neutral", "#69788d")):
            pct = summary[f"{label}_pct"]
            bars += (f'<div style="margin:12px 0"><span>{label.title()}: {pct:.2f}% '
                     f'</span><div style="background:#e5e7eb;border-radius:5px">'
                     f'<div style="height:12px;width:{pct:.6f}%;background:{color};border-radius:5px"></div></div></div>')
        rows = [[r["predicted_sentiment"], round(r[f"prob_{r['predicted_sentiment']}"] , 4), r["text"]]
                for label in ("positive", "negative") for r in result["evidence"][label]]
        absent = [label for label in ("positive", "negative") if not result["evidence"][label]]
        status = "Analysis complete. "
        if absent:
            status += "No segments classified as " + ", ".join(absent) + ". "
        if result["skipped_units"] or any(note.startswith("Pages without extractable text") for note in result["warnings"]):
            status += "Warning: Some content could not be analyzed. Results cover only processed text; review the downloaded report's processing notes. "
        elif result["page_count"] is not None:
            status += "PDF extraction can affect reading order; review the report's processing notes. "
        if result["split_original_units"]:
            status += "Long text was split into token-safe segments; percentages count those segments."
        yield (text, bars, rows, *[gr.update(value=str(path), interactive=True) for path in (downloads.csv, downloads.report)],
               status, downloads, *[gr.update(interactive=True) for _ in range(6)])
    except Exception as exc:
        if downloads is not None:
            downloads.cleanup()
        message = str(exc) if isinstance(exc, ValueError) and not re.search(r"(?:^|[\s'\"])(?:/|[A-Za-z]:\\)", str(exc)) else "Analysis could not complete. Check the document and the local cached-model setup."
        yield (*cleared[:5], "Analysis failed: " + message, None, *[gr.update(interactive=True) for _ in range(6)])


def create_app():
    with gr.Blocks(title="Financial Document Analyzer", analytics_enabled=False,
                   delete_cache=(600, 3600)) as demo:
        gr.Markdown("# Financial Document Analyzer\nAnalyze financial communications using FinBERT and explore sentence-level sentiment.")
        gr.Markdown("Results reflect FinBERT's classification of financial language and should not be interpreted as investment advice.")
        gr.Markdown("## 1. Upload & configuration\nEnglish UTF-8 TXT or text-based PDF · maximum 5 MiB · PDFs up to 30 pages · no scanned-PDF OCR.")
        upload = gr.File(label="Financial document", file_types=[".txt", ".pdf"], type="filepath", elem_id="document-upload")
        with gr.Row():
            company = gr.Textbox(label="Company (optional)")
            ticker = gr.Textbox(label="Ticker (optional)")
        with gr.Row():
            year = gr.Textbox(label="Fiscal year (optional)", placeholder="2025")
            quarter = gr.Dropdown(label="Fiscal quarter (optional)", choices=["Q1", "Q2", "Q3", "Q4"], value=None)
        button = gr.Button("Analyze Document", variant="primary")
        status = gr.Markdown("Ready for a document.")
        gr.Markdown("## 2. Sentiment results")
        summary = gr.Markdown()
        bars = gr.HTML()
        icon = str(Path(__file__).with_name("assets") / "download.svg")
        with gr.Row():
            csv = gr.DownloadButton("Download CSV", size="sm", scale=0, min_width=140,
                                    icon=icon, interactive=False, elem_id="download-csv")
            report = gr.DownloadButton("Download PDF Report", size="sm", scale=0, min_width=180,
                                       icon=icon, interactive=False, elem_id="download-report")
        gr.Markdown("## 3. Supporting segments\nUp to three positive and three negative segments, ranked by confidence. Confidence is uncalibrated and can be wrong.")
        evidence = gr.Dataframe(headers=["Predicted sentiment", "Confidence", "Text"],
                                datatype=["str", "number", "str"], interactive=False, wrap=True,
                                column_widths=[110, 90, 520])
        state = gr.State(None, time_to_live=3600, delete_callback=lambda owner: owner.cleanup() if owner else None)
        results = [summary, bars, evidence, csv, report, status, state]
        button.click(analyze_upload, inputs=[upload, company, ticker, year, quarter, state],
                     outputs=results + [upload, company, ticker, year, quarter, button],
                     concurrency_limit=1, concurrency_id="finbert_cpu", api_name="analyze_document", trigger_mode="once")
        gr.on(triggers=[upload.upload, upload.clear], fn=clear_results, inputs=[state],
              outputs=results, queue=False, api_name=False)
    return demo.queue(default_concurrency_limit=1, max_size=8)


def main():
    create_app().launch(server_name="127.0.0.1", share=False, max_file_size=MAX_UPLOAD_BYTES,
                        inbrowser=False, show_error=False)


if __name__ == "__main__":
    main()
