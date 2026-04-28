# Amicus AI — Local Legal Document Intelligence

[![CI Eval](https://github.com/ncc1701mike/legal-ai-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/ncc1701mike/legal-ai-assistant/actions/workflows/ci.yml)

**Fully local · Air-gapped · Attorney-client privilege protected**

Amicus AI is a privacy-first legal document analysis tool built for litigation attorneys. It runs entirely on your local machine — no documents, queries, or responses ever leave your computer. Built with Llama 3.1 8B, ChromaDB, and a six-stage agentic retrieval pipeline evaluated against a synthetic legal corpus.

## Demo & Deliverable

- [5-minute Loom demo](https://www.loom.com/share/26a183025d574435bc41a3c34d036c60)
- [Project Deliverable — Tasks 1–7](docs/Amicus_AI_Project_Deliverable.docx)

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Retrieval Pipeline](#retrieval-pipeline)
- [Evaluation Results](#evaluation-results)
- [Setup](#setup)
- [Usage](#usage)
- [Development Workflow & CI](#development-workflow--ci)
- [Query Tips](#query-tips)
- [Project Structure](#project-structure)
- [Known Limitations](#known-limitations)

---

## Features

- **Query Documents** — Ask natural language questions across multiple uploaded legal documents simultaneously. Sources cited with relevance scores.
- **Analysis Engine** — Four hardware profiles (Standard / Enhanced / Professional / Enterprise) that select the best local model for your machine. No model IDs exposed to attorneys.
- **Agentic Mode** — Six-node LangGraph pipeline with query planning, multi-pass retrieval, attribution-aware synthesis, and self-critique loop. Automatically routed for complex contradiction and cross-document questions.
- **Summarize** — Generate structured summaries of individual documents or entire case files.
- **Redact & Template** — Automatically detect and replace PII (names, organizations, dates, case numbers, SSNs, phone numbers, addresses) with placeholders. Export redacted documents as reusable templates.
- **OCR Support** — Scanned PDFs are automatically processed via Tesseract OCR. No manual conversion required.
- **Multi-format ingestion** — PDF (digital and scanned), DOCX, TXT, XLSX, CSV.
- **Case Management** — Isolated ChromaDB collections per case; switch cases without re-ingesting.
- **Windows & macOS** — Fully cross-platform via pathlib; launch scripts for both.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Streamlit UI                               │
│        Query · Summarize · Redact · Case Mgmt · Analysis Engine    │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                ┌───────────────▼───────────────┐
                │   Complexity Router (app.py)  │
                │   HARD/CONTRADICTION→agentic  │
                │   others → rerank             │
                └───────┬───────────────┬───────┘
                        │               │
          ┌─────────────▼──┐     ┌──────▼──────────────────────────┐
          │  Rerank Mode   │     │  Agentic RAG (LangGraph 6-node) │
          │                │     │                                  │
          │ 1. HyDE embed  │     │ 1. query_planning               │
          │ 2. BM25 sparse │     │ 2. multi_retrieve               │
          │ 3. RRF fusion  │     │ 3. rerank_filter                │
          │ 4. Cross-enc   │     │ 4. attribute_sources            │
          │    reranking   │     │ 5. synthesize                   │
          │ 5. Top-7 → LLM │     │ 6. self_critique  → LLM        │
          └────────────────┘     └─────────────────────────────────┘
                        │               │
          ┌─────────────▼───────────────▼──────────────────────────┐
          │                   ChromaDB                              │
          │        Persistent per-case vector collections          │
          └─────────────────────────────────────────────────────────┘
                                │
          ┌─────────────────────▼──────────────────────────────────┐
          │               Ollama (local inference)                 │
          │  Standard: llama3.1:8b  (8GB RAM)                     │
          │  Enhanced: llama3.3:8b  (16GB RAM)                    │
          │  Professional: mistral-nemo:12b  (16GB RAM)           │
          │  Enterprise: llama3.1:70b  (32GB+ RAM)                │
          └────────────────────────────────────────────────────────┘
```

**Shared models (all local, no API calls):**
| Role | Model |
|------|-------|
| LLM | Configured via Analysis Engine (default: `llama3.1:8b`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| OCR | Tesseract 5.5 via PyMuPDF |

---

## Retrieval Pipeline

### Rerank Mode (recommended default)
1. **HyDE** — generate a hypothetical answer, embed it, retrieve by semantic similarity
2. **BM25** — keyword-based sparse retrieval for exact citations and statute numbers
3. **RRF fusion** — combine vector and BM25 rankings without score normalization
4. **Cross-encoder reranking** — score all 30 candidates against the real query
5. **Top-7 selection** — send highest-scoring chunks to LLM

### Agentic Mode (auto-routed for complex queries)
A six-node LangGraph pipeline that goes beyond single-pass retrieval:
1. **query_planning** — decompose the question into sub-queries
2. **multi_retrieve** — run each sub-query independently through the rerank pipeline
3. **rerank_filter** — deduplicate and re-score across all retrieved chunks
4. **attribute_sources** — tag each chunk with document and speaker metadata
5. **synthesize** — generate a structured, attributed answer
6. **self_critique** — evaluate the answer against the retrieved evidence; revise if needed

### Multi-hop Mode
Between rerank and agentic in complexity: runs initial retrieval, extracts gaps using the LLM, runs targeted follow-up queries for referenced but unexplained entities, and merges all chunks before synthesis.

---

## Evaluation Results

### LLM-Judge Eval — Chen v. Nexagen corpus (30 questions)

Evaluated using an LLM-based judge against 30 human-authored questions spanning five categories: FACTUAL, CROSS-DOC, CONTRADICTION, TEMPORAL, and TRAP (hallucination guards). Routing: HARD and CONTRADICTION questions → agentic; all others → rerank.

| Category | Pass | Total | Rate |
|----------|------|-------|------|
| FACTUAL | — | 8 | — |
| CROSS-DOC | — | 8 | — |
| CONTRADICTION | — | 6 | — |
| TEMPORAL | — | 4 | — |
| TRAP | — | 4 | — |
| **Overall** | **19** | **30** | **63%** |

Baseline (pre-agentic, multihop mode): 16/30 (53%). The LangGraph agentic pipeline added +3 questions (+10 percentage points).

### RAGAS Eval — Three retrieval modes

Evaluated against 15 synthetic Q&A pairs using RAGAS metrics.

| Metric | Vector | Hybrid | Rerank |
|--------|--------|--------|--------|
| Faithfulness | 0.666 | 0.661 | **0.682** |
| Answer Relevancy | 0.679 | **0.773** | 0.671 |
| Context Precision | 0.783 | 0.613 | **0.800** |
| Context Recall | **0.867** | 0.667 | 0.800 |

Rerank achieves the highest context precision (0.80) — 80% of retrieved chunks are relevant, the most important metric for legal work where irrelevant context can mislead analysis.

---

## Setup

For step-by-step instructions, see the platform-specific guide:

- **macOS:** [install/mac_setup.md](install/mac_setup.md)
- **Windows:** [install/windows_setup.md](install/windows_setup.md)

### Quick start (macOS)

```bash
git clone https://github.com/ncc1701mike/legal-ai-assistant.git
cd legal-ai-assistant

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

ollama pull llama3.1:8b

./launch.sh          # macOS
# or: launch.bat     # Windows
```

The embedding model (`all-MiniLM-L6-v2`, ~90MB) and reranker (`ms-marco-MiniLM-L-6-v2`, ~90MB) download automatically on first use and cache locally.

---

## Usage

### Uploading Documents
Drag and drop or browse to upload documents in the sidebar. Supported formats: PDF, DOCX, TXT, XLSX, CSV. Scanned PDFs are automatically OCR'd.

### Case Management
Create named cases (e.g., `chen_v_nexagen`) to isolate document collections. Switch cases in the sidebar without re-ingesting.

### Querying
Type natural language questions in the Query Documents tab. The system automatically routes to agentic mode for complex questions and rerank mode for simpler ones. Use the **Chunks** slider (default: 7) to control context depth.

### Analysis Engine (sidebar)
Select your hardware profile — Standard, Enhanced, Professional, or Enterprise. The system detects your RAM and shows only profiles your hardware can run. IT admins can see model IDs and pull commands in the collapsed expander.

### Redact
Upload a document in the Redact tab, select PII categories, and download the redacted version as a reusable template.

---

## Development Workflow & CI

### CI Pipeline

Every push to `main` triggers a GitHub Actions workflow that:
1. Starts an Ollama Docker service and pulls `llama3.1:8b`
2. Installs Python dependencies
3. Runs a 10-question diagnostic eval subset (`eval/eval_ci.py`)
4. Fails the workflow if pass rate drops below 70%

The 10 questions are selected to cover each failure mode: simple fact retrieval (FACTUAL), cross-document synthesis (CROSS-DOC), contradiction detection (CONTRADICTION), date reasoning (TEMPORAL), and hallucination resistance (TRAP).

```
push to main
    │
    ▼
GitHub Actions: ci.yml
    ├── Start Ollama service (Docker)
    ├── Pull llama3.1:8b
    ├── pip install -r requirements.txt
    ├── python eval/eval_ci.py
    │       ├── 10 questions × LLM-judge scoring
    │       └── exit 0 (≥70%) or exit 1 (<70%)
    └── Upload eval/results_history.jsonl as artifact
```

### Regression Tracking

Every eval run (CI or full) appends a record to `eval/results_history.jsonl`:

```bash
# Run the full 30-question eval and record history
python eval/eval_batch1.py

# View score progression over time
python eval/show_history.py

# Filter to CI runs only
python eval/show_history.py --ci
```

### Pre-commit Smoke Test (optional)

A 3-question smoke test catches regressions before they reach CI:

```bash
# Install
cp scripts/pre_commit_check.py .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit

# Bypass for a single commit
git commit --no-verify
```

The hook passes if at least 2 of 3 questions pass (Q01 FACTUAL, Q17 CONTRADICTION, Q27 TRAP). Infrastructure failures (Ollama not running) are treated as pass so the hook never blocks work on non-eval machines.

### Running the Full Eval Locally

```bash
# Activate venv and ensure Ollama is running
source venv/bin/activate
ollama serve &

# 10-question CI subset (~8 minutes)
python eval/eval_ci.py

# Full 30-question eval (~45 minutes)
python eval/eval_batch1.py

# Score history
python eval/show_history.py
```

---

## Query Tips

**Use descriptive language, not section numbers.**
The system retrieves by semantic meaning. Instead of "What does Section 5.1 say?", ask "What does the settlement agreement say about neutral references?"

**Break complex questions into follow-ups.**
Start broad, then drill down: "What were the grounds for the termination?" → "Was the position later refilled?"

**Increase chunks for cross-document synthesis.**
If an answer seems incomplete, increase the Chunks slider to 9 or 10 and re-ask.

**Use hybrid mode for exact citations.**
When looking up specific case numbers or statute citations, switch to hybrid mode — BM25 excels at exact string matching.

**For contradiction questions, use agentic mode.**
Questions that ask "what does each party claim?" or "what is the dispute about X?" benefit most from the self-critique loop.

---

## Project Structure

```
legal-ai-assistant/
├── app.py                         # Streamlit UI — main application
├── modules/
│   ├── ingestion.py               # Document parsing, chunking, ChromaDB ingestion
│   ├── retrieval.py               # Vector, hybrid, rerank retrieval modes
│   ├── llm.py                     # RAG query, streaming, model config (get/set_primary_model)
│   ├── agentic_rag.py             # LangGraph 6-node agentic pipeline
│   ├── multihop.py                # Multi-hop retrieval (gap extraction + follow-up)
│   ├── case_manager.py            # Per-case ChromaDB collection management
│   ├── redaction.py               # PII detection and template generation
│   ├── hardware_detect.py         # RAM detection, hardware profiles, model recommendations
│   ├── update_checker.py          # Ollama model update checking (offline-safe)
│   ├── feedback.py                # Query feedback logging
│   ├── cache.py                   # Query result caching (diskcache)
│   ├── citation_verifier.py       # Citation grounding verification
│   └── search.py                  # Optional: case law search (requires internet)
├── eval/
│   ├── eval_batch1.py             # Full 30-question eval runner
│   ├── eval_ci.py                 # 10-question CI subset eval
│   ├── eval_regression.py         # append_result() — regression history writer
│   ├── show_history.py            # ASCII table of score progression
│   └── results_history.jsonl      # Persistent eval history (one JSON per line)
├── evaluation/
│   ├── run_evaluation.py          # RAGAS evaluation runner
│   ├── evaluation.ipynb           # Results visualization notebook
│   └── testset.json               # 15 synthetic Q&A pairs
├── tests/
│   ├── test_case_manager.py
│   ├── test_citation_verifier.py
│   ├── test_hardware_detect.py
│   └── test_update_checker.py
├── install/
│   ├── mac_setup.md               # Step-by-step macOS setup guide
│   └── windows_setup.md           # Step-by-step Windows setup guide
├── scripts/
│   └── pre_commit_check.py        # Optional pre-commit 3-question smoke test
├── .github/workflows/
│   └── ci.yml                     # GitHub Actions CI workflow
├── db/
│   ├── chroma/                    # ChromaDB persistent vector store (gitignored)
│   └── embeddings/                # Cached sentence-transformer models (gitignored)
├── data/                          # Corpus documents (gitignored — attorney-client privilege)
├── launch.sh                      # macOS launch script
├── launch.bat                     # Windows launch script
└── requirements.txt
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Llama 3.1 8B / 3.3 8B / Mistral Nemo 12B / Llama 3.1 70B (Ollama) |
| Embeddings | all-MiniLM-L6-v2 (sentence-transformers) |
| Reranker | ms-marco-MiniLM-L-6-v2 (cross-encoder) |
| Vector DB | ChromaDB (persistent, local, per-case) |
| Agentic Pipeline | LangGraph StateGraph (6 nodes) |
| RAG Framework | LangChain |
| OCR | Tesseract 5.5 via PyMuPDF |
| BM25 | rank-bm25 |
| Evaluation | LLM judge + RAGAS |
| UI | Streamlit |
| Language | Python 3.11 |
| CI | GitHub Actions + Ollama Docker |

---

## Privacy & Security

All processing is fully local. No data is transmitted to external servers at any time. The application is designed for use with privileged attorney-client communications and work product. No telemetry, no cloud inference, no external API calls during normal operation.

The only external network calls occur during:
- Initial model downloads (embedding model and reranker, cached locally after first use)
- Ollama model pulls (`ollama pull <model>`) initiated by the user
- Optional model update checks (update checker in Analysis Engine, user-initiated, fails gracefully offline)

Case documents in `data/` and the ChromaDB store in `db/chroma/` are gitignored and never committed.

---

*Built as a two-week prototype for litigation practice. Evaluated and stress-tested against a synthetic 5-document legal corpus spanning DOCX, TXT, digital PDF, and scanned PDF formats. Batch eval: 19/30 (63%) on Chen v. Nexagen, up from 16/30 (53%) baseline.*
