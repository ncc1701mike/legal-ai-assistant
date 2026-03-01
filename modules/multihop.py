# modules/multihop.py
# Multi-hop Retrieval via LangGraph StateGraph
#
# Architecture:
#   initial_retrieve → extract_gaps → followup_retrieve → synthesize
#
# Stage 1 (initial_retrieve): Standard rerank retrieval — top 7 chunks
# Stage 2 (extract_gaps):     LLM reads initial chunks, extracts key entities
#                             and identifies what's referenced but not explained
# Stage 3 (followup_retrieve): Runs targeted sub-queries for each gap (max 3)
# Stage 4 (synthesize):       Merges all chunks, deduplicates, returns final context
#
# Adds ~5-10 seconds vs single-pass rerank. Worth it for complex cross-document
# queries. Simple queries benefit less — use rerank mode for those.

import logging
from typing import List, Dict, Any, TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from modules.retrieval import rerank_retrieve, format_context
from modules.llm import PRIMARY_MODEL


# ── State Definition ──────────────────────────────────────────────────────────
class MultiHopState(TypedDict):
    query:           str
    initial_chunks:  List[Dict[str, Any]]
    gap_queries:     List[str]
    followup_chunks: List[Dict[str, Any]]
    all_chunks:      List[Dict[str, Any]]
    context:         str
    top_k:           int


# ── Node 1: Initial Retrieval ─────────────────────────────────────────────────
def initial_retrieve(state: MultiHopState) -> MultiHopState:
    """
    Standard rerank retrieval — same as rerank mode.
    Fetches top_k chunks as the starting context.
    """
    logging.info(f"[MultiHop] Stage 1: Initial retrieval for: {state['query'][:80]}")
    chunks = rerank_retrieve(state["query"], top_k=state["top_k"])
    return {**state, "initial_chunks": chunks}


# ── Node 2: Extract Gaps ──────────────────────────────────────────────────────
def extract_gaps(state: MultiHopState) -> MultiHopState:
    """
    LLM reads the initial chunks and identifies:
    - Key entities mentioned but not fully explained (people, dates, events)
    - Facts that are referenced but need corroboration from other documents
    - Follow-up sub-queries that would produce a more complete answer

    Returns up to 3 targeted sub-queries.
    """
    logging.info("[MultiHop] Stage 2: Extracting gaps from initial context")

    if not state["initial_chunks"]:
        return {**state, "gap_queries": []}

    # Build a brief summary of what we have
    initial_context = format_context(state["initial_chunks"])

    llm = ChatOllama(
        model=PRIMARY_MODEL,
        temperature=0.0,
        base_url="http://localhost:11434"
    )

    messages = [
        SystemMessage(content=(
            "You are a legal research assistant. Your job is to identify gaps in "
            "retrieved document context that would help answer a legal question more completely. "
            "Be concise and specific. Output ONLY a numbered list of follow-up search queries, "
            "one per line, maximum 3 queries. No explanations, no preamble. "
            "Each query should be 5-15 words targeting a specific missing fact or entity. "
            "If the context already fully answers the question, output: NONE"
        )),
        HumanMessage(content=(
            f"ORIGINAL QUESTION: {state['query']}\n\n"
            f"CONTEXT RETRIEVED SO FAR:\n{initial_context[:3000]}\n\n"
            "What specific follow-up searches would find missing evidence needed "
            "to answer the original question more completely? "
            "Focus on: people mentioned but not explained, events referenced but undated, "
            "facts in one document that need corroboration from another document."
        ))
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()

    # Parse the numbered list into individual queries
    gap_queries = []
    if raw.upper() != "NONE":
        for line in raw.split("\n"):
            line = line.strip()
            # Strip leading numbers/bullets
            for prefix in ["1.", "2.", "3.", "- ", "• "]:
                if line.startswith(prefix):
                    line = line[len(prefix):].strip()
                    break
            if line and len(line) > 10:
                gap_queries.append(line)

    gap_queries = gap_queries[:3]  # Hard cap at 3
    logging.info(f"[MultiHop] Gap queries identified: {gap_queries}")
    return {**state, "gap_queries": gap_queries}


# ── Node 3: Follow-up Retrieval ───────────────────────────────────────────────
def followup_retrieve(state: MultiHopState) -> MultiHopState:
    """
    Runs rerank retrieval for each gap query.
    Fetches 3 chunks per sub-query — targeted, not broad.
    """
    logging.info(f"[MultiHop] Stage 3: Running {len(state['gap_queries'])} follow-up queries")

    followup_chunks = []
    for sub_query in state["gap_queries"]:
        logging.info(f"[MultiHop]   Sub-query: {sub_query}")
        chunks = rerank_retrieve(sub_query, top_k=3)
        followup_chunks.extend(chunks)

    return {**state, "followup_chunks": followup_chunks}


# ── Node 4: Merge and Deduplicate ─────────────────────────────────────────────
def merge_and_format(state: MultiHopState) -> MultiHopState:
    """
    Merges initial + follow-up chunks, deduplicates by content fingerprint,
    re-sorts by rerank score, and formats final context.

    Initial chunks are kept in full. Follow-up chunks are deduplicated against
    initial set and against each other. Final result capped at top_k * 2.
    """
    logging.info("[MultiHop] Stage 4: Merging and deduplicating chunks")

    seen = set()
    merged = []

    # Initial chunks first — they're already scored
    for chunk in state["initial_chunks"]:
        key = chunk["text"][:120]
        if key not in seen:
            merged.append(chunk)
            seen.add(key)

    # Add follow-up chunks not already covered
    for chunk in state["followup_chunks"]:
        key = chunk["text"][:120]
        if key not in seen:
            # Tag as follow-up so we can distinguish in logs
            chunk = {**chunk, "method": chunk.get("method", "") + "_followup"}
            merged.append(chunk)
            seen.add(key)

    # Re-sort by rerank score — best evidence first regardless of retrieval stage
    merged.sort(
        key=lambda x: x.get("rerank_score", x.get("rrf_score", x.get("score", 0))),
        reverse=True
    )

    # Cap at top_k * 2 to avoid overwhelming the LLM context window
    max_chunks = state["top_k"] * 2
    merged = merged[:max_chunks]

    context = format_context(merged)

    logging.info(
        f"[MultiHop] Final context: {len(merged)} chunks "
        f"({len(state['initial_chunks'])} initial + "
        f"{len(merged) - len(state['initial_chunks'])} follow-up)"
    )

    return {**state, "all_chunks": merged, "context": context}


# ── Graph Assembly ────────────────────────────────────────────────────────────
def build_multihop_graph() -> StateGraph:
    """Build and compile the multi-hop retrieval graph."""
    graph = StateGraph(MultiHopState)

    graph.add_node("initial_retrieve",  initial_retrieve)
    graph.add_node("extract_gaps",      extract_gaps)
    graph.add_node("followup_retrieve", followup_retrieve)
    graph.add_node("merge_and_format",  merge_and_format)

    graph.set_entry_point("initial_retrieve")
    graph.add_edge("initial_retrieve",  "extract_gaps")
    graph.add_edge("extract_gaps",      "followup_retrieve")
    graph.add_edge("followup_retrieve", "merge_and_format")
    graph.add_edge("merge_and_format",  END)

    return graph.compile()


# ── Public Interface ──────────────────────────────────────────────────────────
_graph = None  # Module-level cache — build once, reuse

def multihop_retrieve_and_format(
    query: str,
    top_k: int = 7
) -> tuple[str, List[Dict[str, Any]]]:
    """
    Multi-hop retrieval pipeline.
    Drop-in replacement for retrieve_and_format(mode="rerank").

    Returns:
        (formatted_context_string, raw_chunks_list)
    """
    global _graph
    if _graph is None:
        _graph = build_multihop_graph()

    initial_state: MultiHopState = {
        "query":           query,
        "initial_chunks":  [],
        "gap_queries":     [],
        "followup_chunks": [],
        "all_chunks":      [],
        "context":         "",
        "top_k":           top_k,
    }

    result = _graph.invoke(initial_state)
    return result["context"], result["all_chunks"]


# ── CLI Test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
        "What did the company's own lawyer recommend, and did the company follow that advice?"

    print(f"\nQuery: {query}")
    print("=" * 70)
    context, chunks = multihop_retrieve_and_format(query, top_k=7)
    print(f"\nChunks retrieved: {len(chunks)}")
    for c in chunks:
        method = c.get("method", "")
        score = c.get("rerank_score", c.get("score", 0))
        print(f"  [{score:.3f}] {c['source']} p{c['page']} ({method})")
    print(f"\nContext preview:\n{context[:1000]}...")
