"""Small, generated PDF fixtures, with no copyrighted source text."""

import pymupdf
import pytest

from src.pdf_extraction import extract_pdf
from src.document_processing import analyze_document
from tests.test_document_processing import MockAnalyzer


def pdf_bytes(text="Revenue grew 12.5%. We did not report a loss.", pages=1, encrypted=False):
    with pymupdf.open() as doc:
        for _ in range(pages):
            page = doc.new_page()
            if text:
                page.insert_text((72, 100), text)
        return doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="password") if encrypted else doc.tobytes()


def test_text_pdf_end_to_end(tmp_path):
    content = pdf_bytes()
    extraction = extract_pdf(content)
    assert "12.5%" in extraction["text"] and "not" in extraction["text"]
    path = tmp_path / "financial.pdf"
    path.write_bytes(content)
    result = analyze_document(path, analyzer=MockAnalyzer())
    assert result["page_count"] == 1 and result["classified_units"] == 2
    assert all(row["source_pages"] == [1] for row in result["predictions"])


@pytest.mark.parametrize("content,error", [(b"", "corrupted"), (b"%PDF broken", "corrupted"),
    (pdf_bytes(text=""), "no extractable text"), (pdf_bytes(encrypted=True), "Password")])
def test_unusable_pdf(content, error):
    with pytest.raises(ValueError, match=error):
        extract_pdf(content)


def test_scanned_and_mixed_pages():
    with pymupdf.open() as doc:
        page = doc.new_page()
        pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 10, 10), False)
        pixmap.clear_with(255)
        page.insert_image(pymupdf.Rect(50, 50, 150, 150), pixmap=pixmap)
        with pytest.raises(ValueError, match="Scanned/image-only"):
            extract_pdf(doc.tobytes())
        doc.new_page().insert_text((72, 100), "Revenue was not lower.")
        result = extract_pdf(doc.tobytes())
        assert result["empty_pages"] == [1] and "Pages without" in " ".join(result["warnings"])


def test_page_and_character_limits(monkeypatch):
    monkeypatch.setattr("src.pdf_extraction.MAX_PDF_PAGES", 1)
    with pytest.raises(ValueError, match="pages"):
        extract_pdf(pdf_bytes(pages=2))
    monkeypatch.setattr("src.pdf_extraction.MAX_EXTRACTED_CHARS", 5)
    with pytest.raises(ValueError, match="character"):
        extract_pdf(pdf_bytes())


def test_repeated_margin_blocks_and_body_retained():
    with pymupdf.open() as doc:
        for i in range(3):
            page = doc.new_page()
            page.insert_text((40, 25), "EXAMPLE REPORT")
            page.insert_text((72, 150), f"Revenue did not decline in period {i}.")
        result = extract_pdf(doc.tobytes())
    assert len(result["omissions"]) == 3
    assert all(item["skip_reason"] == "repeated_margin_block" for item in result["omissions"])
    assert result["text"].count("Revenue") == 3
    assert "EXAMPLE REPORT" not in result["text"]


def test_detected_table_excluded_with_text_accounted():
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((50, 70), "Revenue grew despite uncertain demand.")
        for x in (50, 200, 350):
            page.draw_line((x, 150), (x, 230))
        for y in (150, 190, 230):
            page.draw_line((50, y), (350, y))
        for x, y, text in ((60, 175, "Metric"), (210, 175, "Value"), (60, 215, "Revenue"), (210, 215, "$10")):
            page.insert_text((x, y), text)
        result = extract_pdf(doc.tobytes())
    assert result["omissions"] and all(r["skip_reason"] == "detected_table_block" for r in result["omissions"])
    assert "grew" in result["text"]
    assert any("Revenue" in r["text"] for r in result["omissions"])
