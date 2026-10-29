"""Streamlit frontend. Run from the repository root with streamlit run."""

import os
from pathlib import Path
import sys

# Resolve shared modules from this checkout, regardless of the launch directory.
REPOSITORY = str(Path(__file__).resolve().parents[2])
if REPOSITORY not in sys.path:
    sys.path.insert(0, REPOSITORY)

import streamlit as st


def configure_runtime():
    for key, value in {"FINSENTIMENT_ALLOW_MODEL_DOWNLOAD": "1", "OMP_NUM_THREADS": "2",
                       "MKL_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false",
                       "HF_HUB_DISABLE_TELEMETRY": "1"}.items():
        os.environ.setdefault(key, value)


@st.cache_resource(show_spinner=False)
def cached_analyzer():
    # Only the model is global. User documents and report bytes are never cached.
    from src.document_processing import get_analyzer
    return get_analyzer()


def clear_session_result():
    st.session_state.pop("finsentiment_result", None)


def main():
    configure_runtime()  # Before importing Torch/Hub or the shared backend.
    from src.streamlit_support import (clear_result, synchronize_input, upload_identity,
                                       read_upload, process_for_session, user_error, RESULT_KEY)

    st.set_page_config(page_title="Financial Document Analyzer", page_icon="📄", layout="centered")
    st.title("Financial Document Analyzer")
    st.caption("Analyze financial communications with FinBERT and explore sentence-level sentiment.")
    st.caption("Files are processed on this service's server. Upload public, non-confidential documents only. "
               "Predictions describe financial language and are not investment advice.")

    upload = st.file_uploader("Financial document", type=["txt", "pdf"], max_upload_size=5,
                              key="document_upload", on_change=clear_session_result,
                              help="English UTF-8 TXT or text-based PDF · 5 MiB · up to 30 PDF pages · no OCR.")
    left, right = st.columns(2)
    company = left.text_input("Company (optional)", key="company", on_change=clear_session_result)
    ticker = right.text_input("Ticker (optional)", key="ticker", on_change=clear_session_result)
    year = left.text_input("Fiscal year (optional)", key="fiscal_year", on_change=clear_session_result)
    quarter = right.selectbox("Fiscal quarter (optional)", [None, "Q1", "Q2", "Q3", "Q4"],
                              format_func=lambda value: value or "Unspecified", key="fiscal_quarter",
                              on_change=clear_session_result)
    metadata = {"company": company, "ticker": ticker, "fiscal_year": year, "fiscal_quarter": quarter}
    try:
        name, content = read_upload(upload) if upload is not None else (None, None)
        identity = upload_identity(name, content, metadata) if upload is not None else None
        synchronize_input(st.session_state, identity)
    except Exception as exc:
        clear_result(st.session_state)
        st.error(user_error(exc))
        return

    if st.button("Analyze Document", type="primary", disabled=upload is None):
        clear_result(st.session_state)
        try:
            with st.spinner("Loading FinBERT and analyzing on CPU…"):
                process_for_session(st.session_state, name, content, metadata, analyzer=cached_analyzer())
        except Exception as exc:
            clear_result(st.session_state)
            st.error(user_error(exc))

    result = st.session_state.get(RESULT_KEY)
    if result is None:
        return
    st.subheader("Sentiment results")
    summary = result["summary"]
    for column, label in zip(st.columns(3), ("positive", "negative", "neutral")):
        column.metric(label.title(), f"{summary[f'{label}_pct']:.2f}%")
    st.metric("Net sentiment", f"{summary['net_sentiment']:.4f}")
    st.caption(f"{result['classified_units']} classified segments. "
               "Net = (positive − negative) / all classified segments, including neutral. "
               "Percentages show predicted labels, not measured accuracy.")
    if result["skipped_units"] or any(note.startswith("Pages without extractable text") for note in result["warnings"]):
        st.warning("Some content could not be analyzed. Results cover processed text only; review the PDF report's processing notes.")
    elif result["page_count"] is not None:
        st.caption("PDF extraction can affect reading order; review the report's processing notes.")
    if result["split_original_units"]:
        st.caption("Long text was split into token-safe segments; percentages count those segments.")
    csv_column, pdf_column = st.columns(2)
    csv_column.download_button("Download CSV", data=result["csv_bytes"], file_name="predictions.csv",
                               mime="text/csv", icon=":material/download:", on_click="ignore", width="content")
    pdf_column.download_button("Download PDF Report", data=result["pdf_bytes"], file_name="financial_document_report.pdf",
                               mime="application/pdf", icon=":material/download:", on_click="ignore", width="content")
    st.subheader("Supporting segments")
    st.caption("Up to three segments per label, ranked by confidence. Confidence is uncalibrated and can be wrong.")
    st.dataframe(result["evidence"], hide_index=True, width="stretch")


if __name__ == "__main__":
    main()
