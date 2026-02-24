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

from modules.ingestion import ingest_document, get_ingested_documents
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
    /* Base font size ~15% larger */
    html, body, [class*="css"] {
        font-size: 18px !important;
    }

    .main { background-color: #0D1117; }
    .stApp { background-color: #0D1117; color: #E6EDF3; }

    /* Sidebar dusty blue background */
    [data-testid="stSidebar"] {
        background-color: #5A86AD !important;
    }

    /* Sidebar font — medium charcoal grey, no white bubbles */
    [data-testid="stSidebar"] * {
        color: #2C2C2C !important;
        background-color: transparent !important;
    }
    [data-testid="stSidebar"] .stMarkdown p {
        color: #2C2C2C !important;
        font-size: 18px !important;
        background-color: transparent !important;
    }
    [data-testid="stSidebar"] code {
        color: #1A1A1A !important;
        background-color: rgba(0,0,0,0.15) !important;
    }

    /* Sidebar title ~10% larger */
    [data-testid="stSidebar"] h2 {
        font-size: 26px !important;
        color: #1A1A1A !important;
    }

    /* Sidebar Amicus AI title ~25% larger */
    [data-testid="stSidebar"] h2:first-of-type {
        font-size: 32px !important;
        color: #1A1A1A !important;
    }

    /* Main title ~10% larger */
    .stApp h1 {
        font-size: 42px !important;
        color: #E6EDF3 !important;
    }

    /* Response text — 30% larger, light blue, ALL elements */
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

    /* Numbered and indented lists */
    [data-testid="stChatMessage"] ol,
    [data-testid="stChatMessage"] ul {
        color: #D0E8FF !important;
    }
    [data-testid="stChatMessage"] ol li,
    [data-testid="stChatMessage"] ul li {
        color: #D0E8FF !important;
        font-size: 23px !important;
    }

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

    /* ALL buttons — friendly green with border */
    .stButton > button {
        background-color: #238636 !important;
        color: #FFFFFF !important;
        border: 2px solid #2EA043 !important;
        font-size: 18px !important;
        border-radius: 6px !important;
        transition: all 0.2s ease !important;
    }
    /* Hover — white background, green text */
    .stButton > button:hover {
        background-color: #FFFFFF !important;
        color: #238636 !important;
        border: 2px solid #238636 !important;
    }

    /* Clear All Documents — transparent with charcoal border */
    [data-testid="stSidebar"] .stButton > button {
        background-color: transparent !important;
        color: #2C2C2C !important;
        border: 1.5px solid #2C2C2C !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background-color: rgba(0,0,0,0.1) !important;
        color: #1A1A1A !important;
        border: 1.5px solid #1A1A1A !important;
    }

    /* Browse Files button — match Clear All style */
    [data-testid="stFileUploadDropzone"] button {
        background-color: transparent !important;
        color: #2C2C2C !important;
        border: 1.5px solid #2C2C2C !important;
        border-radius: 6px !important;
        font-size: 18px !important;
    }
    [data-testid="stFileUploadDropzone"] button:hover {
        background-color: rgba(0,0,0,0.1) !important;
        color: #1A1A1A !important;
        border: 1.5px solid #1A1A1A !important;
    }

    /* Tab buttons — override completely */
    .stTabs [data-baseweb="tab"] {
        font-size: 18px !important;
        color: #E6EDF3 !important;
        background-color: transparent !important;
        border: none !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #58A6FF !important;
        background-color: transparent !important;
    }
    .stTabs [aria-selected="true"] {
        color: #2EA043 !important;
        border-bottom: 3px solid #2EA043 !important;
        background-color: transparent !important;
    }

    /* Tab underline indicators — force green, kill red */
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #2EA043 !important;
    }
    .stTabs [data-baseweb="tab-border"] {
        background-color: #30363D !important;
    }

    //* Slider — kill green container box */
    [data-testid="stSlider"] > div > div > div {
        background-color: transparent !important;
    }
    /* Slider filled track — green */
    [data-testid="stSlider"] [data-baseweb="slider"] > div > div {
        background-color: #238636 !important;
    }
    /* Slider thumb — green, no red */
    [data-testid="stSlider"] [role="slider"] {
        background-color: #2EA043 !important;
        border: 3px solid #2EA043 !important;
        box-shadow: none !important;
    }
    /* Slider unfilled track */
    [data-testid="stSlider"] [data-baseweb="slider"] > div {
        background-color: #4A7A99 !important;
    }
    [data-testid="stSlider"] span {
        color: #2C2C2C !important;
    }

    /* Chat input */
    .stChatInput input {
        font-size: 18px !important;
    }

    /* Status text */
    .status-success {
        color: #2EA043 !important;
        font-weight: bold !important;
        font-size: 18px !important;
    }
    .status-skip { color: #5A3E00 !important; font-weight: bold; }
    .status-error { color: #7A0000 !important; font-weight: bold; }

    /* Force Ollama status green — must be last rule */
    #ollama-status {
        color: #2EA043 !important;
    }

    /* Browse Files — target Streamlit's internal upload button */
    [data-testid="stFileUploaderDropzoneInstructions"] ~ div button,
    [data-testid="stBaseButton-secondary"] {
        background-color: transparent !important;
        color: #2C2C2C !important;
        border: 1.5px solid #2C2C2C !important;
        border-radius: 6px !important;
        font-size: 18px !important;
    }
    [data-testid="stBaseButton-secondary"]:hover {
        background-color: rgba(0,0,0,0.1) !important;
        color: #1A1A1A !important;
        border: 1.5px solid #1A1A1A !important;
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


# ── Main Area ─────────────────────────────────────────────────────────────────
st.markdown("# ⚖️ Document Analysis")
st.markdown("*Fully local · Air-gapped · Attorney-client privilege protected*")
st.markdown("---")

tab1, tab2 = st.tabs(["💬 Query Documents", "📋 Summarize"])

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
                    result = rag_query(query, top_k=top_k)

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

            # Download button
            st.download_button(
                label="Download Summary as TXT",
                data=summary,
                file_name="case_summary.txt",
                mime="text/plain"
            )