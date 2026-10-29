"""Per-analysis CSV and PDF downloads; no shared writable report filenames."""

import csv
from pathlib import Path
from tempfile import TemporaryDirectory

CSV_COLUMNS = ["document_id", "company", "ticker", "fiscal_year", "fiscal_quarter",
               "sentence_id", "sentence_index", "text", "predicted_sentiment",
               "prob_positive", "prob_negative", "prob_neutral", "model_name", "model_revision",
               "token_count", "parent_sentence_id", "parent_sentence_index", "segment_index",
               "segment_count", "split_from_long_unit", "original_text", "source_char_start",
               "source_char_end", "source_pages", "source_sha256"]


def plain_markdown(value):
    """Render untrusted document content as literal Markdown text, never HTML."""
    text = str(value).replace("\n", " ").replace("\r", " ")
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    for character in "\\`*_{}[]()#+-.!|":
        text = text.replace(character, "\\" + character)
    return text


def markdown_report(result):
    summary = result["summary"]
    lines = ["# FinSentiment — Document Analysis", "", "Document: " + plain_markdown(result["document_name"]),
             "", f"Original narrative units: {result['original_text_units']}; split original units: {result['split_original_units']}.",
             f"Identified analysis segments / excluded blocks: {result['total_units']}; classified segments: {result['classified_units']}; skipped units/blocks: {result['skipped_units']}.",
             "Skips include numeric-only fragments and PDF table/repeated-margin blocks; image-only pages are listed separately in extraction warnings.", ""]
    for field, value in result["metadata"].items():
        lines.append(f"{field.replace('_', ' ').title()}: {plain_markdown(value) if value is not None else 'Unspecified'}")
    lines += ["", "| Predicted label | Count | Percentage of classified segments |", "| --- | --- | --- |"]
    for label in ("positive", "negative", "neutral"):
        value = summary[f"{label}_pct"]
        lines.append(f"| {label.title()} | {summary[f'{label}_count']} | {value:.2f}% |" if value is not None else f"| {label.title()} | 0 | Unavailable |")
    net = summary["net_sentiment"]
    lines += ["", f"Net sentiment: {net:.4f}" if net is not None else "Net sentiment: unavailable",
              "Net = (positive count − negative count) / classified segments; neutral segments remain in the denominator.",
              "", "Percentages describe predicted language labels, not prediction accuracy.", "", "## Supporting segments", ""]
    for label, rows in result["evidence"].items():
        lines.append(f"### {label.title()}")
        lines.append("")
        if not rows:
            lines += [f"No segments classified as {label}.", ""]
        for row in rows:
            lines += [f"- {plain_markdown(row['text'])} — {label}, confidence {row[f'prob_{label}']:.4f} ({row['sentence_id']}).", ""]
    model = result["predictions"][0]["model_name"] if result["predictions"] else "ProsusAI/finbert"
    lines += ["Model: " + model + " (CPU).", "", "## Processing limitations", ""]
    lines += ["- " + plain_markdown(note) for note in result["warnings"] + result["limitations"]]
    return "\n".join(lines) + "\n"


def write_pdf_report(result, output_path):
    """Write A4 pages using embedded fonts and measured, non-truncating wrapping."""
    import pymupdf

    regular, bold = pymupdf.Font("helv"), pymupdf.Font("hebo")
    fallback = None
    page_rect = pymupdf.paper_rect("a4")
    left, right, top, bottom = 48, page_rect.width - 48, 44, page_rect.height - 48
    document = pymupdf.open()
    page, y = None, top

    def select_font(text, heading=False):
        nonlocal fallback
        font, name = (bold, "report_bold") if heading else (regular, "report_body")
        if all(char.isspace() or font.has_glyph(ord(char)) for char in text):
            return font, name
        if fallback is None:
            fallback = pymupdf.Font("cjk")
        if not all(char.isspace() or fallback.has_glyph(ord(char)) for char in text):
            raise ValueError("The PDF report contains characters unavailable in the embedded fonts.")
        return fallback, "report_unicode"

    def ensure_space(height):
        nonlocal page, y
        if page is None or y + height > bottom:
            page = document.new_page(width=page_rect.width, height=page_rect.height)
            y = top

    def line(text, size=10.5, heading=False, color=(.15, .20, .26), x=None):
        nonlocal y
        font, name = select_font(text, heading)
        height = max(1.45, font.ascender - font.descender + .1) * size
        ensure_space(height)
        page.insert_font(fontname=name, fontbuffer=font.buffer)
        page.insert_text((left if x is None else x, y + font.ascender * size), text,
                         fontsize=size, fontname=name, color=color)
        y += height

    def paragraph(text, size=10.5, heading=False, gap=7):
        nonlocal y
        text = str(text).replace("\t", "    ")
        font, _ = select_font(text, heading)
        available = right - left
        for original_line in text.splitlines() or [""]:
            current = ""
            for word in original_line.split():
                candidate = current + (" " if current else "") + word
                if font.text_length(candidate, fontsize=size) <= available:
                    current = candidate
                    continue
                if current:
                    line(current, size, heading)
                    current = ""
                # An unbroken token is wrapped by character without losing text.
                for char in word:
                    if current and font.text_length(current + char, fontsize=size) > available:
                        line(current, size, heading)
                        current = ""
                    current += char
            line(current, size, heading)
        y += gap

    def section(title):
        nonlocal y
        ensure_space(52)
        y += 9
        paragraph(title, size=14, heading=True, gap=7)

    try:
        paragraph("Financial Document Analysis Report", size=21, heading=True, gap=12)
        paragraph("Document: " + result["document_name"])
        for field, value in result["metadata"].items():
            if value is not None:
                paragraph(f"{field.replace('_', ' ').title()}: {value}", gap=3)
        model = result["predictions"][0]["model_name"] if result["predictions"] else "ProsusAI/finbert"
        paragraph(f"Model: {model} (CPU)")
        section("Sentiment summary")
        paragraph(f"Total analyzed segments: {result['classified_units']}", heading=True)
        summary = result["summary"]
        ensure_space(4 * 28 + 10)
        table = [("Predicted sentiment", "Count", "Percentage")]
        for label in ("positive", "negative", "neutral"):
            pct = summary[f"{label}_pct"]
            table.append((label.title(), str(summary[f"{label}_count"]), f"{pct:.2f}%" if pct is not None else "Unavailable"))
        columns = (left, left + 250, left + 350, right)
        for index, row in enumerate(table):
            for col, text in enumerate(row):
                cell = pymupdf.Rect(columns[col], y, columns[col + 1], y + 28)
                page.draw_rect(cell, color=(.78, .82, .87), fill=(.93, .95, .97) if index == 0 else None, width=.5)
                font, name = select_font(text, index == 0)
                page.insert_font(fontname=name, fontbuffer=font.buffer)
                page.insert_text((cell.x0 + 8, y + 7 + font.ascender * 10), text, fontsize=10, fontname=name, color=(.15, .20, .26))
            y += 28
        y += 12
        net = summary["net_sentiment"]
        paragraph(f"Net sentiment: {net:.4f}" if net is not None else "Net sentiment: unavailable", size=13, heading=True)
        paragraph("Net = (positive count − negative count) / classified segments. Neutral segments remain in the denominator. Percentages describe predicted text labels, not classification accuracy.", size=9)
        section("Supporting statements")
        for label, rows in result["evidence"].items():
            ensure_space(45)
            paragraph(label.title(), size=11, heading=True, gap=4)
            if not rows:
                paragraph(f"No segments classified as {label}.")
            for row in rows:
                paragraph(row.get("original_text", row["text"]), gap=4)
                paragraph(f"Predicted sentiment: {label} · Confidence: {row[f'prob_{label}']:.4f}\nSegment: {row['sentence_id']}", size=9, gap=12)
        section("Processing notes and limitations")
        paragraph(f"Original narrative units: {result['original_text_units']}; split original units: {result['split_original_units']}.\nIdentified units / excluded blocks: {result['total_units']}; classified segments: {result['classified_units']}; skipped units/blocks: {result['skipped_units']}.")
        paragraph("Exclusions may include numeric-only fragments and detected PDF table or repeated-margin blocks. Image-only pages are described in extraction notes.")
        notes = result["warnings"] + result["limitations"] + [
            "Predictions can be incorrect, including high-confidence predictions.",
            "Long sentences may be split into multiple text segments; segments are not independent original sentences.",
            "PDF extraction can be imperfect, including reading order and table handling.",
        ]
        for note in dict.fromkeys(notes):
            paragraph("• " + note)
        for index, output_page in enumerate(document):
            output_page.insert_textbox(pymupdf.Rect(left, page_rect.height - 35, right, page_rect.height - 15),
                                       f"Page {index + 1} of {len(document)}", fontsize=8, fontname="helv", align=2, color=(.4, .45, .5))
        document.set_metadata({"title": "Financial Document Analysis Report", "creator": "Financial Document Analyzer"})
        document.subset_fonts()
        document.save(output_path, garbage=4, deflate=True)
    finally:
        document.close()


class AnalysisDownloads:
    """Own only this request's directory; explicit cleanup also works on failures."""
    def __init__(self, result, *, parent=None):
        self._temporary = TemporaryDirectory(prefix="finsentiment-", dir=parent)
        folder = Path(self._temporary.name)
        self.csv = folder / "finsentiment_predictions.csv"
        self.report = folder / "financial_document_report.pdf"
        self.skipped = folder / "finsentiment_skipped.csv"
        try:
            with self.csv.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
                writer.writeheader()
                for row in result["predictions"]:
                    writer.writerow({key: row.get(key) for key in CSV_COLUMNS})
            write_pdf_report(result, self.report)
            fields = ["document_id", "skip_index", "skip_reason", "source_pages", "source_char_start", "source_char_end", "text"]
            with self.skipped.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for index, row in enumerate(result["skipped"]):
                    writer.writerow({**{key: row.get(key) for key in fields}, "document_id": result["document_id"], "skip_index": index})
        except Exception:
            self.cleanup()
            raise

    def cleanup(self):
        self._temporary.cleanup()
