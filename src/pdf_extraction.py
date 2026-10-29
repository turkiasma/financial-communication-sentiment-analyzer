"""Bounded text-PDF extraction without OCR; every omitted block is reported."""

from collections import Counter
import math

MAX_PDF_PAGES = 30
MAX_EXTRACTED_CHARS = 500_000


def extract_pdf(content: bytes) -> dict:
    import pymupdf

    try:
        document = pymupdf.open(stream=content, filetype="pdf")
    except Exception as exc:
        raise ValueError("Cannot open this PDF; it may be corrupted or invalid.") from exc
    with document:
        if document.needs_pass:
            raise ValueError("Password-protected PDFs are not supported. Upload an unlocked copy.")
        if not 1 <= len(document) <= MAX_PDF_PAGES:
            raise ValueError(f"PDF must contain 1–{MAX_PDF_PAGES} pages.")
        blocks, omissions, empty_pages = [], [], []
        total = 0
        try:
            for page_index, page in enumerate(document):
                text_blocks = [b for b in page.get_text("blocks", sort=True) if b[6] == 0 and b[4].strip()]
                if not text_blocks:
                    empty_pages.append(page_index + 1)
                tables = [pymupdf.Rect(t.bbox) for t in page.find_tables().tables]
                for index, block in enumerate(text_blocks):
                    rect = pymupdf.Rect(block[:4])
                    text = block[4].strip()
                    total += len(text)
                    if total > MAX_EXTRACTED_CHARS:
                        raise ValueError("PDF text exceeds the 500,000-character processing limit.")
                    item = {"page": page_index + 1, "block": index, "text": text,
                            "margin": rect.y1 < page.rect.height * .07 or rect.y0 > page.rect.height * .93}
                    if any((rect & table).get_area() / max(rect.get_area(), 1) > .5 for table in tables):
                        omissions.append({**item, "skip_reason": "detected_table_block"})
                    else:
                        blocks.append(item)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("PDF text/layout extraction failed; try a UTF-8 TXT export.") from exc
        if not blocks and not omissions:
            raise ValueError("PDF has no extractable text. Scanned/image-only PDFs need OCR, which is not supported.")
        # Remove only short, exactly repeated margin blocks across >=3 and >=80% of pages.
        repeats = Counter()
        for page in range(1, len(document) + 1):
            repeats.update({b["text"] for b in blocks if b["page"] == page and b["margin"] and len(b["text"]) <= 120})
        repeated = {text for text, n in repeats.items() if n >= max(3, math.ceil(len(document) * .8))}
        pieces, page_spans, offset = [], [], 0
        for block in blocks:
            if block["margin"] and block["text"] in repeated:
                omissions.append({**block, "skip_reason": "repeated_margin_block"})
                continue
            text = block["text"]
            pieces.append(text)
            page_spans.append({"start": offset, "end": offset + len(text), "page": block["page"], "block": block["block"]})
            offset += len(text) + 2
        text = "\n\n".join(pieces)
        if not text.strip():
            raise ValueError("No narrative text remains after PDF table/header filtering; review a TXT export instead.")
        warnings = ["PDF blocks are ordered top-to-bottom then left-to-right. Multi-column reading order and table detection can be imperfect; review extracted text before interpreting results.",
                    "PDF character offsets refer to extracted text, not PDF bytes. No OCR is performed."]
        if omissions:
            warnings.append(f"Excluded {len(omissions)} table/repeated-margin blocks; their text and page numbers are available in the skipped-content CSV.")
        if empty_pages:
            warnings.append(f"Pages without extractable text: {', '.join(map(str, empty_pages))}. Image content on these pages was not analyzed.")
        return {"text": text, "page_spans": page_spans, "omissions": omissions,
                "warnings": warnings, "page_count": len(document), "empty_pages": empty_pages}
