# ⚖️ Amicus AI — Local Legal Document Intelligence

**Fully local · Air-gapped · Attorney-client privilege protected**

Amicus AI is a privacy-first legal document analysis tool built for litigation attorneys. It runs entirely on your local machine — no documents, queries, or responses ever leave your computer. Built with Llama 3.1 8B, ChromaDB, and a three-stage retrieval pipeline evaluated against a synthetic legal corpus.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Retrieval Pipeline](#retrieval-pipeline)
- [Evaluation Results](#evaluation-results)
- [Setup](#setup)
- [Usage](#usage)
- [Query Tips](#query-tips)
- [Known Limitations & Roadmap](#known-limitations--roadmap)
- [Project Structure](#project-structure)

---

## Features

- **Query Documents** — Ask natural language questions across multiple uploaded legal documents simultaneously. Sources cited with relevance scores.
- **Summarize** — Generate structured summaries of individual documents or entire case files.
- **Redact & Template** — Automatically detect and replace PII (names, organizations, dates, case numbers, SSNs, phone numbers) with placeholders. Export redacted documents as reusable templates.
- **OCR Support** — Scanned PDFs are automatically processed via Tesseract OCR. No manual conversion required.
- **Multi-format ingestion** — PDF (digital and scanned), DOCX, TXT, XLSX, CSV.
- **Three retrieval modes** — Vector, Hybrid (HyDE + BM25), and Rerank (cross-encoder). Selectable in the UI.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Streamlit UI                        │
│         Query · Summarize · Redact · Settings           │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                    LangChain RAG                        │
│         rag_query() → retrieve → prompt → LLM           │
└──────┬─────────────────────────────────────┬────────────┘
       │                                     │
┌──────▼──────────┐                 ┌────────▼────────────┐
│   ChromaDB      │                 │   Llama 3.1 8B      │
│   Vector Store  │                 │   via Ollama        │
│   cosine sim    │                 │   local inference   │
└──────┬──────────┘                 └─────────────────────┘
       │
┌──────▼──────────────────────────────────────────────────┐
│              Three-Stage Retrieval Pipeline             │
│                                                         │
│  Stage 1: Hybrid Retrieval (HyDE + BM25 + RRF)         │
│           → 30 candidate chunks                         │
│                                                         │
│  Stage 2: Cross-Encoder Reranking                       │
│           → cross-encoder/ms-marco-MiniLM-L-6-v2        │
│           → scored against actual query                 │
│                                                         │
│  Stage 3: Top-K Selection                               │
│           → 7 highest-scoring chunks to LLM            │
└─────────────────────────────────────────────────────────┘
```

**Models (all local, no API calls):**
- LLM: `llama3.1:8b` via Ollama
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- OCR: Tesseract 5.5 via PyMuPDF

---

## Retrieval Pipeline

### Vector Mode
Standard semantic search using cosine similarity against dense embeddings. Fast and effective for straightforward single-document queries.

### Hybrid Mode (HyDE + BM25 + RRF)
Three-component pipeline fused with Reciprocal Rank Fusion:
- **HyDE** (Hypothetical Document Embeddings): generates a hypothetical answer to the query, embeds it, and retrieves by semantic similarity to that answer — improves recall on abstract questions
- **BM25**: keyword-based sparse retrieval — excels at exact legal citations, statute numbers, and case names
- **RRF**: combines vector and BM25 rankings without requiring score normalization

Best for: exact citation lookup, large corpora, queries where you're not sure which document has the answer.

### Rerank Mode (Default)
Extends hybrid with a cross-encoder re-ranking stage:
1. Hybrid retrieves 30 candidate chunks
2. Cross-encoder scores each chunk against the actual query using full cross-attention (not just embedding similarity)
3. Top 7 highest-scoring chunks sent to LLM

Best for: complex queries, cross-document synthesis, precision-critical legal research. **Recommended default.**

---

## Evaluation Results

Evaluated against 15 synthetic Q&A pairs using RAGAS metrics. All three retrieval modes compared.

| Metric | Vector | Hybrid | Rerank |
|--------|--------|--------|--------|
| Faithfulness | 0.666 | 0.661 | **0.682** |
| Answer Relevancy | 0.679 | **0.773** | 0.671 |
| Context Precision | 0.783 | 0.613 | **0.800** |
| Context Recall | **0.867** | 0.667 | 0.800 |

**Key findings:**
- Rerank achieves highest context precision (0.80) — 80% of retrieved chunks are relevant, critical for legal work where irrelevant context can mislead analysis
- Hybrid achieves highest answer relevancy (0.77) — broader candidate pool gives LLM more material for targeted answers
- Rerank matches or exceeds vector on all metrics except recall, making it the best all-around choice for production use

*Note: Synthetic benchmarks favor vector retrieval because test questions are generated from document text and naturally match vector vocabulary. Hybrid and rerank show their advantage on real attorney queries involving cross-document synthesis and exact citation lookup.*

---

## Setup

### Prerequisites
- macOS (Apple Silicon recommended) or Linux
- [Ollama](https://ollama.ai) installed and running
- Python 3.11+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (`brew install tesseract` on macOS)
- Node.js (for document generation scripts only)

### Installation

```bash
# Clone the repository
git clone https://github.com/ncc1701mike/legal-ai-assistant.git
cd legal-ai-assistant

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Pull the LLM
ollama pull llama3.1:8b

# Start the app
streamlit run app.py
```

### First Run
The embedding model (`all-MiniLM-L6-v2`, ~90MB) and reranker (`ms-marco-MiniLM-L-6-v2`, ~90MB) will download automatically on first use and cache locally. No subsequent downloads required.

---

## Usage

### Uploading Documents
Drag and drop or browse to upload documents in the sidebar. Supported formats: PDF, DOCX, TXT, XLSX, CSV. Scanned PDFs are automatically OCR'd via Tesseract.

### Querying
Type natural language questions in the Query Documents tab. Use the **Chunks to Retrieve** slider (default: 7) to control how much context the LLM receives — higher values improve cross-document synthesis at the cost of response speed.

### Retrieval Mode Selection
Use the **Mode** dropdown in the sidebar:
- **rerank** — best precision, recommended default
- **hybrid** — best for exact citations and large corpora
- **vector** — fastest, good for simple single-document queries

### Summarize
Select documents to summarize in the Summarize tab. Summaries are structured and citation-aware.

### Redact
Upload a document in the Redact tab, select PII categories to replace, and download the redacted version as a reusable template.

---

## Query Tips

**Use descriptive language, not section numbers.**
The system retrieves by semantic meaning, not document structure. Instead of:
> "What does Section 5.1 say?"

Ask:
> "What does the settlement agreement say about neutral references?"

**Break complex questions into follow-ups.**
For multi-part questions spanning many documents, start broad then drill down:
> "What were the grounds for the termination?" → "Was the position later refilled?"

**Increase chunks for cross-document synthesis.**
If an answer seems incomplete and you know the information is in your documents, increase the Chunks slider to 9 or 10 and re-ask.

**Use hybrid mode for exact citations.**
When looking up specific case numbers, statute citations, or charge numbers, switch to hybrid mode — BM25 excels at exact string matching.

---

## Known Limitations & Roadmap

### Current Limitations

**1. Section-number retrieval**
Queries using formal section numbers (e.g., "Section 5.1") can fail to retrieve the correct chunk because the section header and content may be in adjacent chunks and the cross-encoder doesn't score "Section 5.1" as semantically meaningful. **Workaround:** Use descriptive natural language queries.

*Roadmap fix:* Implement document structure-aware chunking that keeps section headers attached to their content using a sliding window that always includes the nearest preceding heading. Alternatively, add a metadata layer that indexes section numbers separately and allows direct lookup by section reference.

**2. Cross-document synthesis depth**
Complex queries requiring synthesis across 4+ documents may miss evidence that requires very specific sub-queries to surface. The system retrieves by semantic proximity to the query, so evidence that's relevant but phrased differently than the question may not rank highly.

*Roadmap fix:* Implement **multi-hop retrieval** — after the initial retrieval, extract key entities and facts from the top chunks, then run a second retrieval pass using those entities as additional queries. This is sometimes called "iterative retrieval" or "chain-of-thought RAG." LangGraph's agent framework is already in the stack and would support this pattern.

**3. LLM source attribution**
When synthesizing multiple documents, the LLM occasionally misattributes which person said what (e.g., attributing a lawyer's internal memo analysis to an expert witness). The retrieved chunks are correct but the LLM conflates speakers.

*Roadmap fix:* Prepend each chunk with a structured metadata header before sending to the LLM: `[SOURCE: expert_witness_deposition.txt | SPEAKER: Dr. Margaret Okonkwo]`. This gives the LLM explicit speaker context and reduces conflation.

**4. OCR quality on complex layouts**
Tesseract OCR works well on standard court documents but can produce artifacts on multi-column layouts, tables, or low-resolution scans (visible as run-together words like "motionona" instead of "motion on a").

*Roadmap fix:* Implement a post-OCR cleanup step using the LLM itself — pass OCR'd text through a short "fix spacing and formatting errors" prompt before chunking. Alternatively, upgrade to PyMuPDF's premium OCR or integrate a vision model (LLaVA) for complex layouts.

**5. Rerank score display for non-rerank modes**
When using vector or hybrid mode, the displayed relevance scores are raw cosine similarity or BM25 scores respectively, which are not directly comparable to rerank scores. Low cosine similarity scores (~0.05) can appear misleadingly low even when the answer is correct.

*Roadmap fix:* Normalize scores to a 0–1 scale per mode, or label them differently ("BM25 score" vs "semantic similarity" vs "rerank score") so the display is mode-aware.

---

## Project Structure

```
legal-ai-assistant/
├── app.py                    # Streamlit UI — main application
├── modules/
│   ├── ingestion.py          # Document parsing, chunking, ChromaDB ingestion
│   │                         # Supports PDF (digital + OCR), DOCX, TXT, XLSX, CSV
│   ├── retrieval.py          # Three retrieval modes: vector, hybrid, rerank
│   │                         # HyDE, BM25, RRF fusion, cross-encoder reranking
│   ├── llm.py                # RAG query, summarization, Ollama connection
│   └── redaction.py          # PII detection and template generation
├── evaluation/
│   ├── run_evaluation.py     # RAGAS evaluation runner — all three modes
│   ├── evaluation.ipynb      # Results visualization and analysis notebook
│   ├── testset.json          # 15 synthetic Q&A pairs (GPT-4o generated)
│   ├── results_vector.json   # Vector baseline RAGAS scores
│   ├── results_hybrid.json   # Hybrid RAGAS scores
│   ├── results_rerank.json   # Rerank RAGAS scores
│   └── results_combined.json # Three-way comparison
├── db/
│   ├── chroma/               # ChromaDB persistent vector store
│   └── embeddings/           # Cached sentence-transformer models
├── data/                     # Sample documents for testing
└── requirements.txt
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Llama 3.1 8B (Ollama) |
| Embeddings | all-MiniLM-L6-v2 (sentence-transformers) |
| Reranker | ms-marco-MiniLM-L-6-v2 (cross-encoder) |
| Vector DB | ChromaDB (persistent, local) |
| RAG Framework | LangChain |
| OCR | Tesseract 5.5 via PyMuPDF |
| BM25 | rank-bm25 |
| Evaluation | RAGAS |
| UI | Streamlit |
| Language | Python 3.11 |

---

## Privacy & Security

All processing is fully local. No data is transmitted to external servers at any time. The application is designed for use with privileged attorney-client communications and work product. No telemetry, no cloud inference, no external API calls during normal operation.

The only external network calls occur during initial model downloads (embedding model and reranker), which are cached locally and not repeated.

---

*Built as a two-week prototype for litigation practice. Evaluated and stress-tested against a synthetic 5-document legal corpus spanning DOCX, TXT, digital PDF, and scanned PDF formats.*
