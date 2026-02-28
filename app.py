# app.py
# AI Legal Assistant — Main Streamlit Application
# Production build — fully local, no external calls

import os
import logging
import tempfile
from pathlib import Path
from modules.ingestion import ingest_document, get_ingested_documents, clear_all_documents

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

import streamlit as st
from dotenv import load_dotenv

from modules.ingestion import ingest_document, get_ingested_documents, clear_all_documents
from modules.redaction import redact_document
from modules.llm import rag_query, summarize_documents, test_connection

load_dotenv()

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Legal AI Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* ── Base ──────────────────────────────────────────────── */
    html, body, [class*="css"] { font-size: 18px !important; }
    .main, .stApp { background-color: #0D1117; color: #E6EDF3; }

    /* ── Header strip — deep teal ──────────────────────────── */
    header[data-testid="stHeader"] {
        background-color: #0A3D3D !important;
        border-bottom: 1px solid #1F5F5F !important;
    }

    /* ── Sidebar ────────────────────────────────────────────── */
    [data-testid="stSidebar"] { background-color: #5A86AD !important; }
    [data-testid="stSidebar"] * { color: #2C2C2C !important; background-color: transparent !important; }
    [data-testid="stSidebar"] .stMarkdown p { color: #2C2C2C !important; font-size: 18px !important; }
    [data-testid="stSidebar"] code {
        color: #1A1A1A !important;
        background-color: rgba(0,0,0,0.15) !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        max-width: 200px !important;
        display: inline-block !important;
    }
    [data-testid="stSidebar"] h2 { font-size: 26px !important; color: #1A1A1A !important; }
    [data-testid="stSidebar"] h2:first-of-type { font-size: 32px !important; }
    [data-testid="stSidebar"] h3 { font-size: 20px !important; color: #1A1A1A !important; }

    /* ── Sidebar buttons — transparent, charcoal border ────── */
    [data-testid="stSidebar"] .stButton > button {
        background-color: transparent !important;
        color: #2C2C2C !important;
        border: 1.5px solid #2C2C2C !important;
        border-radius: 6px !important;
        font-size: 17px !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background-color: rgba(0,0,0,0.1) !important;
        color: #1A1A1A !important;
        border-color: #1A1A1A !important;
    }
    
    /* ── Sidebar selectbox ──────────────────────────────────── */
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background-color: transparent !important;
        border: 1.5px solid #2C2C2C !important;
        border-radius: 6px !important;
    }
    
    [data-testid="stSidebar"] [data-baseweb="select"] * {
        color: #2C2C2C !important;
        background-color: transparent !important;
    }

    /* ── Sidebar slider ─────────────────────────────────────── */
    [data-testid="stSidebar"] [data-testid="stSlider"] > div,
    [data-testid="stSidebar"] [data-testid="stSlider"] > div > div,
    [data-testid="stSidebar"] [data-testid="stSlider"] > div > div > div {
        background-color: transparent !important;
    }
    [data-testid="stSidebar"] [data-baseweb="slider"] > div {
        background-color: #3D6B8A !important;
        height: 4px !important;
    }
    [data-testid="stSidebar"] [data-baseweb="slider"] > div > div {
        background-color: #2EA043 !important;
    }
    [data-testid="stSidebar"] [role="slider"] {
        background-color: #2EA043 !important;
        border: 3px solid #2EA043 !important;
        box-shadow: none !important;
    }
    [data-testid="stSidebar"] [data-testid="stSlider"] span { color: #2C2C2C !important; }

    [data-testid="stSidebar"] [role="slider"] {
        background-color: #2EA043 !important;
        border-color: #2EA043 !important;
        box-shadow: none !important;
    }

    /* ── Main title ─────────────────────────────────────────── */
    .stApp h1 { font-size: 42px !important; color: #E6EDF3 !important; }

    /* ── Main area buttons — green ──────────────────────────── */
    section[data-testid="stMain"] .stButton > button {
        background-color: #238636 !important;
        color: #FFFFFF !important;
        border: 2px solid #2EA043 !important;
        font-size: 18px !important;
        border-radius: 6px !important;
        transition: all 0.2s ease !important;
    }
    section[data-testid="stMain"] .stButton > button:hover {
        background-color: #FFFFFF !important;
        color: #238636 !important;
        border-color: #238636 !important;
    }

    /* ── Browse Files button ────────────────────────────────── */
    [data-testid="stFileUploadDropzone"] button,
    [data-testid="stBaseButton-secondary"] {
        background-color: transparent !important;
        color: #2C2C2C !important;
        border: 1.5px solid #2C2C2C !important;
        border-radius: 6px !important;
        font-size: 18px !important;
    }
    [data-testid="stFileUploadDropzone"] button:hover,
    [data-testid="stBaseButton-secondary"]:hover {
        background-color: rgba(0,0,0,0.1) !important;
        color: #1A1A1A !important;
        border-color: #1A1A1A !important;
    }

    /* ── Tabs ───────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab"] {
        font-size: 18px !important;
        color: #E6EDF3 !important;
        background-color: transparent !important;
        border: none !important;
    }
    .stTabs [data-baseweb="tab"]:hover { color: #58A6FF !important; }
    .stTabs [aria-selected="true"] {
        color: #2EA043 !important;
        border-bottom: 3px solid #2EA043 !important;
        background-color: transparent !important;
    }
    .stTabs [data-baseweb="tab-highlight"] { background-color: #2EA043 !important; }
    .stTabs [data-baseweb="tab-border"] { background-color: #2EA043 !important; }

    .stTabs [data-baseweb="tab"] span {
        color: #E6EDF3 !important;
    }
    .stTabs [aria-selected="true"] span {
        color: #2EA043 !important;
    }
    .stTabs [data-baseweb="tab"]:hover span {
        color: #58A6FF !important;
    }

    /* ── Chat messages ──────────────────────────────────────── */
    [data-testid="stChatMessage"] .stMarkdown p,
    [data-testid="stChatMessage"] .stMarkdown li,
    [data-testid="stChatMessage"] .stMarkdown ol li,
    [data-testid="stChatMessage"] .stMarkdown ul li,
    [data-testid="stChatMessage"] .stMarkdown h1,
    [data-testid="stChatMessage"] .stMarkdown h2,
    [data-testid="stChatMessage"] .stMarkdown h3,
    [data-testid="stChatMessage"] .stMarkdown strong,
    [data-testid="stChatMessage"] .stMarkdown em,
    [data-testid="stChatMessage"] .stMarkdown span {
        font-size: 23px !important;
        color: #D0E8FF !important;
        line-height: 1.7 !important;
    }
    [data-testid="stChatMessage"] ol li,
    [data-testid="stChatMessage"] ul li { color: #D0E8FF !important; font-size: 23px !important; }

    /* ── Citation & answer boxes ────────────────────────────── */
    .citation-box {
        background-color: #161B22;
        border-left: 3px solid #238636;
        padding: 10px 15px;
        border-radius: 4px;
        margin: 5px 0;
        font-size: 16px !important;
        font-family: monospace;
        color: #A8D8A8 !important;
    }
    .answer-box {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 20px;
        margin: 10px 0;
        font-size: 23px !important;
        color: #D0E8FF !important;
        line-height: 1.7 !important;
    }

    /* ── Chat input ─────────────────────────────────────────── */
    .stChatInput input { font-size: 18px !important; }

    /* ── Status labels ──────────────────────────────────────── */
    .status-success { color: #2EA043 !important; font-weight: bold !important; font-size: 18px !important; }
    .status-skip    { color: #5A3E00 !important; font-weight: bold; }
    .status-error   { color: #7A0000 !important; font-weight: bold; }
    [data-testid="stSidebar"] #ollama-status,
    [data-testid="stSidebar"] #ollama-status * {
        color: #2EA043 !important;
        background-color: transparent !important;
    }

    /* ── Redaction checkboxes ───────────────────────────────── */
    [data-testid="stCheckbox"] label,
    [data-testid="stCheckbox"] p { color: #D0E8FF !important; font-size: 18px !important; }

    /* ── Download buttons ───────────────────────────────────── */
    [data-testid="stDownloadButton"] button {
        background-color: transparent !important;
        color: #D0E8FF !important;
        border: 1.5px solid #D0E8FF !important;
        font-size: 18px !important;
        border-radius: 6px !important;
    }
    [data-testid="stDownloadButton"] button:hover {
        background-color: rgba(208,232,255,0.1) !important;
        color: #FFFFFF !important;
        border-color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "ingested_docs" not in st.session_state:
    st.session_state.ingested_docs = get_ingested_documents()
if "confirm_clear" not in st.session_state:
    st.session_state.confirm_clear = False

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚖️ Amicus Ai")
    st.markdown("---")

    # Connection status
    st.markdown("### System Status")
    if test_connection():
        st.markdown(
            '<p id="ollama-status" style="color:#2EA043 !important; '
            'font-weight:900 !important; font-size:24px !important; '
            'background-color:transparent !important; '
            'text-shadow: 1px 1px 3px rgba(0,0,0,0.9) !important;">'
            '● Ollama LLM Connected</p>',
            unsafe_allow_html=True
        )
    else:
        st.markdown('<p class="status-error">● Ollama Not Running</p>',
                   unsafe_allow_html=True)
        st.warning("Start Ollama: open Terminal and run `ollama serve`")

    st.markdown("---")

    # Document upload
    st.markdown("### Upload Documents")
    uploaded_files = st.file_uploader(
        "Drop files here",
        type=["pdf", "docx", "xlsx", "xls", "txt", "csv"],
        accept_multiple_files=True,
        help="Supported: PDF, Word, Excel, TXT, CSV"
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.ingested_docs:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=Path(uploaded_file.name).suffix
                    ) as tmp:
                        tmp.write(uploaded_file.getbuffer())
                        tmp_path = tmp.name

                    result = ingest_document(tmp_path, original_name=uploaded_file.name)
                    os.unlink(tmp_path)

                    if result["status"] == "success":
                        st.markdown(
                            f'<p class="status-success">✓ {uploaded_file.name}'
                            f' ({result["chunks"]} chunks)</p>',
                            unsafe_allow_html=True
                        )
                        st.session_state.ingested_docs = get_ingested_documents()
                    else:
                        st.markdown(
                            f'<p class="status-skip">⚠ {uploaded_file.name}'
                            f' — {result["status"]}</p>',
                            unsafe_allow_html=True
                        )

    st.markdown("---")

    # Document inventory
    st.markdown("### Documents in Store")
    docs = get_ingested_documents()
    if docs:
        for doc in docs:
            st.markdown(f"📄 `{doc}`")

        st.markdown("---")
        if st.button("🗑️ Clear All Documents", key="clear_docs"):
            st.session_state.confirm_clear = True

        if st.session_state.get("confirm_clear"):
            st.warning("⚠️ This will remove all documents from the vector store. Are you sure?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Yes, Clear All", key="confirm_yes"):
                    clear_all_documents()
                    st.session_state.ingested_docs = []
                    st.session_state.confirm_clear = False
                    st.session_state.messages = []
                    st.rerun()
            with col2:
                if st.button("Cancel", key="confirm_no"):
                    st.session_state.confirm_clear = False
                    st.rerun()
    else:
        st.markdown("*No documents uploaded yet*")

    # Settings
    st.markdown("### Settings")
    top_k = st.slider("Chunks to retrieve", min_value=3, max_value=10, value=5)
    st.caption("Higher = more context, slower response")
    st.markdown("**Retrieval Strategy**")
    retrieval_mode = st.selectbox(
        "Mode",
        options=["rerank", "hybrid", "vector", "hyde", "bm25"],
        index=0,
        help=(
            "Rerank: Best overall strategy for complex questions — reranks chunks using cross-encoder\n"
            "Hybrid: Best for general questions — combines semantic + keyword search\n"
            "Vector: Fast semantic similarity search\n"
            "HyDE: Best for conceptual queries — generates hypothetical answer first\n"
            "BM25: Best for exact legal terms, statute numbers, case citations"
        ),
        key="retrieval_mode"
    )


# ── Main Area ─────────────────────────────────────────────────────────────────
st.markdown("# ⚖️ Document Analysis")
st.markdown("*Fully local · Air-gapped · Attorney-client privilege protected*")
tab1, tab2, tab3 = st.tabs(["💬 Query Documents", "📋 Summarize", "🛡️ Redact"])

# ── TAB 1: Query ──────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Ask a Question About Your Documents")

    # Chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                st.markdown("**Sources:**")
                for source in message["sources"]:
                    st.markdown(
                        f'<div class="citation-box">'
                        f'📄 {source["file"]} — Page {source["page"]} '
                        f'(relevance: {source["score"]})'
                        f'</div>',
                        unsafe_allow_html=True
                    )

    # Query input
    if query := st.chat_input("e.g. Who are the parties in this case?"):
        docs = get_ingested_documents()
        if not docs:
            st.warning("Please upload documents first.")
        else:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            # Get RAG response
            with st.chat_message("assistant"):
                with st.spinner("Searching documents and generating response..."):
                    result = rag_query(
                                        user_input,
                                        top_k=st.session_state.get("top_k", 5),
                                        mode=st.session_state.get("retrieval_mode", "rerank")
                                    )

                st.markdown(result["answer"])

                # Show citations
                if result["sources"]:
                    st.markdown("**Sources:**")
                    for source in result["sources"]:
                        st.markdown(
                            f'<div class="citation-box">'
                            f'📄 {source["file"]} — Page {source["page"]} '
                            f'(relevance: {source["score"]})'
                            f'</div>',
                            unsafe_allow_html=True
                        )

            # Save to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"]
            })

# ── TAB 2: Summarize ──────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Generate Document Summary")
    st.markdown("Produces a structured executive summary of all uploaded documents.")

    docs = get_ingested_documents()
    if not docs:
        st.warning("Please upload documents first.")
    else:
        st.markdown("**Documents to summarize:**")
        for doc in docs:
            st.markdown(f"📄 `{doc}`")

        if st.button("Generate Summary", type="primary"):
            with st.spinner("Analyzing documents... this may take 30-60 seconds."):
                summary = summarize_documents()

            st.markdown("### Summary")
            st.markdown(
                f'<div class="answer-box">{summary}</div>',
                unsafe_allow_html=True
            )

            # Download summary as TXT
            st.download_button(
                label="Download Summary as TXT",
                data=summary,
                file_name="case_summary.txt",
                mime="text/plain"
            )
# ── TAB 3: Redact ─────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### 🛡️ Document Redaction & Template Generator")
    st.markdown("Upload a document to automatically detect and replace PII with placeholders.")

    all_categories = {
        "[PARTY_NAME]":       "Person names",
        "[ORGANIZATION]":     "Companies & courts",
        "[LOCATION]":         "Cities & states",
        "[DATE]":             "All dates",
        "[CASE_NO]":          "Case numbers",
        "[SSN]":              "Social security numbers",
        "[PHONE]":            "Phone numbers",
        "[EMAIL]":            "Email addresses",
        "[FINANCIAL_AMOUNT]": "Dollar amounts",
        "[BAR_NO]":           "Attorney bar numbers",
        "[ZIP_CODE]":         "ZIP codes",
        "[DOCKET_NO]":        "Docket numbers",
    }

    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.markdown("**Select Categories to Redact**")
        st.caption("Nothing is redacted unless explicitly selected.")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("☑️ Select All", key="select_all",
                        use_container_width=True):
                for placeholder in all_categories:
                    st.session_state[f"redact_{placeholder}"] = True
                st.rerun()
        with col_b:
            if st.button("☐ Clear All", key="clear_all_redact",
                        use_container_width=True):
                for placeholder in all_categories:
                    st.session_state[f"redact_{placeholder}"] = False
                st.rerun()

        selected_categories = []
        for placeholder, description in all_categories.items():
            if st.checkbox(
                f"{placeholder} — {description}",
                value=False,
                key=f"redact_{placeholder}"
            ):
                selected_categories.append(placeholder)

        st.markdown("---")
        aggressive = st.toggle("Aggressive Mode", value=False)
        st.caption("Also redacts legal references and cardinal numbers.")
    
    with col1:
        redact_file = st.file_uploader(
            "Upload document to redact",
            type=["pdf", "docx", "xlsx", "txt", "csv"],
            key="redact_uploader",
            help="Document will be processed locally — no data leaves this machine"
        )

        if redact_file:
            if not selected_categories:
                st.info("☝️ Select at least one category above to enable redaction.")

            if st.button("🛡️ Redact Document", type="primary",
                        disabled=len(selected_categories) == 0):
                with st.spinner(f"Redacting {redact_file.name}..."):
                    # Save to temp file
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=Path(redact_file.name).suffix
                    ) as tmp:
                        tmp.write(redact_file.getbuffer())
                        tmp_path = tmp.name

                    # Redact
                    output_path = tmp_path + "_REDACTED.txt"
                    report = redact_document(
                        tmp_path,
                        output_path=output_path,
                        aggressive=aggressive,
                        categories=selected_categories if selected_categories else None
                    )
                    os.unlink(tmp_path)

                if report["status"] == "success":
                    st.success(f"✅ Redaction complete — {report['total_redactions']} items redacted")

                    # Show placeholder counts
                    st.markdown("**Redaction Summary:**")
                    cols = st.columns(3)
                    for i, (placeholder, count) in enumerate(report['placeholder_counts'].items()):
                        with cols[i % 3]:
                            st.metric(placeholder, count)

                    # Read and show preview
                    with open(output_path, "r", encoding="utf-8") as f:
                        redacted_content = f.read()

                    st.markdown("**Preview (first 1000 characters):**")
                    st.markdown(
                        f'<div class="answer-box" style="font-size:16px !important;">'
                        f'{redacted_content[:1000]}...'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                    # Export options
                    st.markdown("**Export Redacted Document As:**")
                    export_col1, export_col2, export_col3 = st.columns(3)

                    with export_col1:
                        st.download_button(
                            label="📄 Export as TXT",
                            data=redacted_content,
                            file_name=f"{Path(redact_file.name).stem}_REDACTED.txt",
                            mime="text/plain",
                            use_container_width=True
                        )

                    with export_col2:
                        # Generate DOCX in memory
                        from docx import Document as DocxDocument
                        from io import BytesIO
                        doc = DocxDocument()
                        doc.add_heading("REDACTED DOCUMENT", level=1)
                        doc.add_paragraph(f"Source: {redact_file.name}")
                        doc.add_paragraph(f"Redactions: {report['total_redactions']}")
                        doc.add_paragraph("─" * 50)
                        for para in redacted_content.split("\n"):
                            if para.strip():
                                doc.add_paragraph(para)
                        docx_buffer = BytesIO()
                        doc.save(docx_buffer)
                        docx_buffer.seek(0)
                        st.download_button(
                            label="📝 Export as DOCX",
                            data=docx_buffer.getvalue(),
                            file_name=f"{Path(redact_file.name).stem}_REDACTED.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            use_container_width=True
                        )

                    with export_col3:
                        # Generate PDF in memory
                        from reportlab.lib.pagesizes import letter
                        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
                        from reportlab.lib.styles import getSampleStyleSheet
                        from io import BytesIO as BytesIO2
                        pdf_buffer = BytesIO2()
                        pdf_doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
                        styles = getSampleStyleSheet()
                        story = []
                        story.append(Paragraph("REDACTED DOCUMENT", styles['Title']))
                        story.append(Paragraph(f"Source: {redact_file.name}", styles['Normal']))
                        story.append(Paragraph(f"Total redactions: {report['total_redactions']}", styles['Normal']))
                        story.append(Spacer(1, 12))
                        for para in redacted_content.split("\n"):
                            if para.strip():
                                try:
                                    story.append(Paragraph(para, styles['Normal']))
                                    story.append(Spacer(1, 6))
                                except Exception:
                                    pass
                        pdf_doc.build(story)
                        pdf_buffer.seek(0)
                        st.download_button(
                            label="📋 Export as PDF",
                            data=pdf_buffer.getvalue(),
                            file_name=f"{Path(redact_file.name).stem}_REDACTED.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )

                    os.unlink(output_path)
                else:
                    st.error(f"Redaction failed: {report['status']}")