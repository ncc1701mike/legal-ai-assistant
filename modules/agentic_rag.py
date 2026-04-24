# modules/agentic_rag.py
# Agentic RAG — Full end-to-end LangGraph pipeline for Expert mode
#
# Architecture:
#   query_planner → multi_pass_retrieve → attribution_synthesis → self_critique
#
# Node 1 (query_planner):        Decomposes complex query into 2-4 targeted sub-queries
# Node 2 (multi_pass_retrieve):  Runs each sub-query independently, deduplicates, re-ranks
# Node 3 (attribution_synthesis): Builds speaker→document map, generates attributed response
# Node 4 (self_critique):        Checks attribution, hallucination, source hierarchy
#                                 Conditional edge: revise if fails, return if passes
#
# Replaces multihop as Expert mode. Returns same dict format as rag_query().

import logging
import re
from typing import List, Dict, Any, TypedDict, Optional

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langsmith import traceable

from modules.retrieval import rerank_retrieve, format_context
from modules.llm import PRIMARY_MODEL, SYSTEM_PROMPT

logger = logging.getLogger(__name__)


# ── State Definition ──────────────────────────────────────────────────────────
class AgenticState(TypedDict):
    query:           str
    sub_queries:     List[str]
    all_chunks:      List[Dict[str, Any]]
    speaker_map:     Dict[str, List[str]]   # speaker → [source_file, ...]
    draft_response:  str
    final_response:  str
    critique_notes:  str
    revision_count:  int
    top_k:           int
    case_id:         Optional[str]


# ── Helpers ───────────────────────────────────────────────────────────────────
def _get_llm() -> ChatOllama:
    return ChatOllama(
        model=PRIMARY_MODEL,
        temperature=0.0,
        base_url="http://localhost:11434"
    )


def _parse_numbered_list(text: str, max_items: int = 4) -> List[str]:
    """Parse a numbered or bulleted list from LLM output."""
    items = []
    for line in text.strip().split("\n"):
        line = line.strip()
        for prefix in ["1.", "2.", "3.", "4.", "- ", "• "]:
            if line.startswith(prefix):
                line = line[len(prefix):].strip()
                break
        if line and len(line) > 8:
            items.append(line)
    return items[:max_items]


def _dedup_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate chunks by content fingerprint, keep highest scored."""
    seen = {}
    for chunk in chunks:
        key = chunk["text"][:120]
        score = chunk.get("rerank_score", chunk.get("rrf_score", chunk.get("score", 0)))
        if key not in seen or score > seen[key][0]:
            seen[key] = (score, chunk)
    return [v[1] for v in seen.values()]


def _build_speaker_map(chunks: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Build a map of speaker/witness name → source documents.
    Looks for deposition document descriptions to identify witnesses.
    """
    speaker_map: Dict[str, List[str]] = {}
    depo_pattern = re.compile(
        r'(?:deposition|testimony|testified|witness)\s+(?:of\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})',
        re.IGNORECASE
    )
    name_pattern = re.compile(r'\b([A-Z][a-z]+\s+(?:[A-Z]\.\s+)?[A-Z][a-z]+)\b')

    for chunk in chunks:
        source = chunk.get("source", "")
        desc = chunk.get("description", "").lower()
        text = chunk.get("text", "")

        # Primary: deposition docs — extract witness name from description or filename
        if "deposition" in desc or "testimony" in desc:
            # Try to extract name from description
            matches = depo_pattern.findall(desc + " " + text[:200])
            for name in matches:
                name = name.strip()
                if name and len(name) > 5:
                    if name not in speaker_map:
                        speaker_map[name] = []
                    if source not in speaker_map[name]:
                        speaker_map[name].append(source)

        # Secondary: scan chunk text for capitalized names near attribution verbs
        attr_verbs = r'(?:testified|stated|said|wrote|emailed|replied|confirmed|denied|admitted)'
        attr_pattern = re.compile(
            rf'([A-Z][a-z]+(?:\s+[A-Z][a-z]+){{1,2}})\s+{attr_verbs}',
            re.IGNORECASE
        )
        for match in attr_pattern.finditer(text):
            name = match.group(1).strip()
            if name and len(name) > 4:
                if name not in speaker_map:
                    speaker_map[name] = []
                if source not in speaker_map[name]:
                    speaker_map[name].append(source)

    return speaker_map


# ── Node 1: Query Planner ─────────────────────────────────────────────────────
@traceable(name="agentic_query_planner")
def query_planner(state: AgenticState) -> AgenticState:
    """
    Decomposes the user query into 2-4 targeted sub-queries.
    Performance questions always generate exactly 3 sub-queries targeting:
    IT/badge logs, historical reviews, and witness testimony — ensuring all
    three independent evidence categories are retrieved rather than just the
    chunks most semantically similar to the top-level question.
    """
    logger.info(f"[AgenticRAG] Node 1: Planning sub-queries for: {state['query'][:80]}")

    llm = _get_llm()
    messages = [
        SystemMessage(content=(
            "You are a legal research query planner. Your job is to decompose a complex "
            "legal question into 2-4 targeted sub-queries that together will retrieve all "
            "evidence needed to answer the original question completely.\n\n"
            "Rules:\n"
            "- Each sub-query should target a DIFFERENT aspect of the question\n"
            "- Sub-queries should be 5-15 words, specific and searchable\n"
            "- If the question is simple and single-topic, output just 1-2 sub-queries\n"
            "- Output ONLY a numbered list of sub-queries, no preamble or explanation\n"
            "- Focus on: specific people, specific events, specific dates, specific documents\n\n"
            "PERFORMANCE QUESTIONS — MANDATORY FAN-OUT:\n"
            "If the question involves performance decline, work quality, productivity, "
            "missed deadlines, or whether an employee's performance changed, you MUST "
            "generate exactly 3 sub-queries, one per evidence category, in this order:\n"
            "  1. IT system logs, badge access records, login timestamps, hours worked\n"
            "  2. Historical performance reviews, prior evaluations, supervisor ratings\n"
            "  3. Witness testimony about work quality and conduct from colleagues or observers\n"
            "Do NOT generate a single broad sub-query for the whole performance question. "
            "Fan out across all three evidence categories even if the original question "
            "does not name them — the goal is to retrieve from three distinct document types "
            "that each provide independent evidence about whether performance actually changed."
        )),
        HumanMessage(content=(
            f"Decompose this legal question into targeted sub-queries:\n\n{state['query']}"
        ))
    ]

    response = llm.invoke(messages)
    sub_queries = _parse_numbered_list(response.content, max_items=4)

    # Always include the original query as a fallback
    if not sub_queries or state["query"] not in sub_queries:
        sub_queries = [state["query"]] + sub_queries
    sub_queries = sub_queries[:4]

    logger.info(f"[AgenticRAG] Sub-queries: {sub_queries}")
    return {**state, "sub_queries": sub_queries}


# ── Node 2: Multi-Pass Retrieval ──────────────────────────────────────────────
@traceable(name="agentic_multi_pass_retrieve")
def multi_pass_retrieve(state: AgenticState) -> AgenticState:
    """
    Runs each sub-query independently through rerank retrieval.
    Deduplicates and re-ranks the combined result set.
    """
    logger.info(f"[AgenticRAG] Node 2: Running {len(state['sub_queries'])} sub-queries")

    all_chunks = []
    for sub_query in state["sub_queries"]:
        logger.info(f"[AgenticRAG]   Retrieving: {sub_query[:60]}")
        chunks = rerank_retrieve(sub_query, top_k=state["top_k"], case_id=state.get("case_id"))
        all_chunks.extend(chunks)

    # Deduplicate and re-rank
    deduped = _dedup_chunks(all_chunks)
    deduped.sort(
        key=lambda x: x.get("rerank_score", x.get("rrf_score", x.get("score", 0))),
        reverse=True
    )

    # Cap at top_k * 3 to give synthesis node rich context without overwhelming it
    max_chunks = state["top_k"] * 3
    deduped = deduped[:max_chunks]

    # Build speaker map from retrieved chunks
    speaker_map = _build_speaker_map(deduped)

    logger.info(
        f"[AgenticRAG] Retrieved {len(deduped)} unique chunks, "
        f"speakers identified: {list(speaker_map.keys())}"
    )
    return {**state, "all_chunks": deduped, "speaker_map": speaker_map}


# ── Node 3: Attribution-Aware Synthesis ───────────────────────────────────────
@traceable(name="agentic_attribution_synthesis")
def attribution_synthesis(state: AgenticState) -> AgenticState:
    """
    Generates a response with strict per-speaker attribution.
    Injects speaker→document map into the prompt to enforce sourcing discipline.
    """
    logger.info("[AgenticRAG] Node 3: Generating attribution-aware response")

    context = format_context(state["all_chunks"])

    # Build speaker attribution instructions from the map
    speaker_instructions = ""
    if state["speaker_map"]:
        lines = []
        for speaker, sources in state["speaker_map"].items():
            lines.append(f"  - {speaker}: statements must be cited from {', '.join(sources)}")
        speaker_instructions = (
            "\n\nIDENTIFIED SPEAKERS IN RETRIEVED DOCUMENTS — STRICT ATTRIBUTION REQUIRED:\n"
            + "\n".join(lines)
            + "\n\nFor each speaker above, cite ONLY their own deposition/document. "
            "NEVER attribute one speaker's words to another document."
        )

    agentic_system = SYSTEM_PROMPT + speaker_instructions + """

AGENTIC RAG MODE — ADDITIONAL REQUIREMENTS:
- This response was generated after multi-pass retrieval across sub-queries
- You have richer context than standard mode — use it to answer ALL parts of the question
- If the question has multiple parts (A, B, C), answer each part explicitly
- Identify and explicitly flag ANY contradictions between documents
- Do not collapse multi-part answers into a single vague response
- If a sub-question cannot be answered from the retrieved context, say so explicitly"""

    llm = _get_llm()
    messages = [
        SystemMessage(content=agentic_system),
        HumanMessage(content=(
            f"CONTEXT FROM DOCUMENTS:\n{context}\n\n"
            f"QUESTION: {state['query']}\n\n"
            "Answer completely, with strict per-speaker attribution and explicit "
            "citations for every factual claim."
        ))
    ]

    response = llm.invoke(messages)
    draft = response.content.strip()

    logger.info(f"[AgenticRAG] Draft response length: {len(draft)} chars")
    return {**state, "draft_response": draft}


# ── Node 4: Self-Critique Loop ────────────────────────────────────────────────
@traceable(name="agentic_self_critique")
def self_critique(state: AgenticState) -> AgenticState:
    """
    Reviews the draft response for:
    1. Attribution errors — wrong speaker cited for a statement
    2. Hallucination — facts not present in retrieved chunks
    3. Source hierarchy violations — complaint used as primary source over deposition
    4. Multi-part collapse — question had multiple parts but only one was answered

    Sets critique_notes. Conditional edge checks if revision is needed.
    """
    logger.info(f"[AgenticRAG] Node 4: Self-critique (revision #{state['revision_count']})")

    context = format_context(state["all_chunks"])
    llm = _get_llm()

    messages = [
        SystemMessage(content=(
            "You are a quality control reviewer for legal document analysis. "
            "Your job is to identify specific errors in a draft response.\n\n"
            "Check for these failure modes:\n"
            "1. ATTRIBUTION ERROR: A statement is attributed to the wrong speaker or document\n"
            "2. HALLUCINATION: A fact, date, or quote appears in the response but NOT in the context\n"
            "3. SOURCE HIERARCHY: The complaint or answer is cited as primary evidence when "
            "a deposition or internal document covering the same fact is available\n"
            "4. MULTI-PART COLLAPSE: The original question had multiple sub-questions "
            "but the response only addresses some of them\n\n"
            "Output format:\n"
            "PASS — if no significant errors found\n"
            "FAIL: [list specific errors found, one per line]\n\n"
            "Be strict but fair. Minor wording issues are not failures. "
            "Only flag clear factual or attribution errors."
        )),
        HumanMessage(content=(
            f"ORIGINAL QUESTION: {state['query']}\n\n"
            f"RETRIEVED CONTEXT (ground truth):\n{context[:4000]}\n\n"
            f"DRAFT RESPONSE TO REVIEW:\n{state['draft_response']}\n\n"
            "Does this response pass quality control? List any specific errors."
        ))
    ]

    response = llm.invoke(messages)
    critique = response.content.strip()

    logger.info(f"[AgenticRAG] Critique result: {critique[:100]}")
    return {**state, "critique_notes": critique}


# ── Conditional Edge: Should We Revise? ───────────────────────────────────────
def should_revise(state: AgenticState) -> str:
    """
    Returns 'revise' if critique found failures and we haven't hit the revision limit.
    Returns 'accept' otherwise.
    Max 1 revision to avoid infinite loops.
    """
    if state["revision_count"] >= 1:
        logger.info("[AgenticRAG] Max revisions reached — accepting response")
        return "accept"

    critique = state["critique_notes"].upper()
    if critique.startswith("FAIL") or "ATTRIBUTION ERROR" in critique or "HALLUCINATION" in critique:
        logger.info("[AgenticRAG] Critique failed — revising response")
        return "revise"

    logger.info("[AgenticRAG] Critique passed — accepting response")
    return "accept"


# ── Node 5: Revision ──────────────────────────────────────────────────────────
@traceable(name="agentic_revise")
def revise_response(state: AgenticState) -> AgenticState:
    """
    Revises the draft response based on critique notes.
    Only runs if self_critique returned FAIL.
    """
    logger.info("[AgenticRAG] Revising response based on critique")

    context = format_context(state["all_chunks"])
    llm = _get_llm()

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=(
            f"CONTEXT FROM DOCUMENTS:\n{context}\n\n"
            f"ORIGINAL QUESTION: {state['query']}\n\n"
            f"PREVIOUS DRAFT (with errors):\n{state['draft_response']}\n\n"
            f"ERRORS IDENTIFIED:\n{state['critique_notes']}\n\n"
            "Please rewrite the response correcting ALL identified errors. "
            "Maintain strict per-speaker attribution and cite every factual claim."
        ))
    ]

    response = llm.invoke(messages)
    revised = response.content.strip()

    return {
        **state,
        "draft_response": revised,
        "revision_count": state["revision_count"] + 1
    }


# ── Node 6: Finalize ──────────────────────────────────────────────────────────
def finalize(state: AgenticState) -> AgenticState:
    """Moves draft_response to final_response."""
    return {**state, "final_response": state["draft_response"]}


# ── Graph Assembly ────────────────────────────────────────────────────────────
def build_agentic_graph() -> StateGraph:
    """Build and compile the agentic RAG graph."""
    graph = StateGraph(AgenticState)

    graph.add_node("query_planner",          query_planner)
    graph.add_node("multi_pass_retrieve",    multi_pass_retrieve)
    graph.add_node("attribution_synthesis",  attribution_synthesis)
    graph.add_node("self_critique",          self_critique)
    graph.add_node("revise_response",        revise_response)
    graph.add_node("finalize",               finalize)

    graph.set_entry_point("query_planner")
    graph.add_edge("query_planner",         "multi_pass_retrieve")
    graph.add_edge("multi_pass_retrieve",   "attribution_synthesis")
    graph.add_edge("attribution_synthesis", "self_critique")

    graph.add_conditional_edges(
        "self_critique",
        should_revise,
        {
            "revise": "revise_response",
            "accept": "finalize",
        }
    )

    graph.add_edge("revise_response", "self_critique")
    graph.add_edge("finalize",        END)

    return graph.compile()


# ── Module-level graph cache ──────────────────────────────────────────────────
_graph = None


# ── Public Interface ──────────────────────────────────────────────────────────
@traceable(name="agentic_rag_query")
def agentic_rag_query(
    question: str,
    top_k: int = 7,
    case_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Full agentic RAG pipeline.
    Drop-in replacement for rag_query() when mode="agentic".

    Returns same dict format as rag_query():
        answer, sources, chunks, chunks_used, question
    """
    global _graph
    if _graph is None:
        _graph = build_agentic_graph()

    initial_state: AgenticState = {
        "query":           question,
        "sub_queries":     [],
        "all_chunks":      [],
        "speaker_map":     {},
        "draft_response":  "",
        "final_response":  "",
        "critique_notes":  "",
        "revision_count":  0,
        "top_k":           top_k,
        "case_id":         case_id,
    }

    result = _graph.invoke(initial_state)

    # Format sources — same structure as rag_query
    sources = []
    seen = set()
    for chunk in result["all_chunks"]:
        key = f"{chunk['source']}|{chunk['page']}"
        if key not in seen:
            sources.append({
                "file":         chunk["source"],
                "page":         chunk["page"],
                "score":        chunk.get("rrf_score", chunk.get("score", 0)),
                "rerank_score": chunk.get("rerank_score") or chunk.get("rrf_score") or chunk.get("score")
            })
            seen.add(key)

    return {
        "question":    question,
        "answer":      result["final_response"],
        "sources":     sources,
        "chunks":      result["all_chunks"],
        "chunks_used": len(result["all_chunks"]),
        "sub_queries": result["sub_queries"],
        "speaker_map": result["speaker_map"],
        "critique":    result["critique_notes"],
        "revision_count": result["revision_count"],
    }


# ── Streaming public interface ────────────────────────────────────────────────
def stream_agentic_rag_query(
    question: str,
    top_k: int = 7,
    case_id: Optional[str] = None,
    progress_callback=None,
    result_holder: dict = None,
):
    """
    Streaming version of agentic_rag_query for the Streamlit UI.

    Yields tokens from the final synthesis step so the UI can display the
    response as it is generated rather than waiting ~77 s for the full pipeline.

    Progress phases are reported via ``progress_callback(message: str)`` so the
    caller can update a status placeholder for each pipeline stage:
        "🔍 Planning query..."
        "📚 Retrieving documents (N sub-queries)..."
        "✍️ Generating response..."
        "🔄 Revising response..."   (only when self-critique fails)

    After the generator is exhausted, ``result_holder`` is populated with the
    same dict shape as ``agentic_rag_query()`` so the caller can read sources,
    sub_queries, critique, etc.

    The LangGraph pipeline (``agentic_rag_query``) is left untouched and
    continues to be used by the eval runner and CLI.
    """
    from modules.llm import _build_sources, SYSTEM_PROMPT
    from modules.cache import set_cached_query, get_cached_query

    def _progress(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)

    # ── Cache hit — replay char by char ──────────────────────────────────────
    cached = get_cached_query(question, "agentic", top_k)
    if cached is not None:
        if result_holder is not None:
            result_holder.update({**cached, "from_cache": True})
        yield from cached["answer"]
        return

    llm = _get_llm()

    # ── Stage 1: Query planning ───────────────────────────────────────────────
    _progress("🔍 Planning query...")
    state: AgenticState = {
        "query":          question,
        "sub_queries":    [],
        "all_chunks":     [],
        "speaker_map":    {},
        "draft_response": "",
        "final_response": "",
        "critique_notes": "",
        "revision_count": 0,
        "top_k":          top_k,
        "case_id":        case_id,
    }
    state = query_planner(state)

    # ── Stage 2: Multi-pass retrieval ─────────────────────────────────────────
    n_sub = len(state["sub_queries"])
    _progress(f"📚 Retrieving documents ({n_sub} sub-quer{'y' if n_sub == 1 else 'ies'})...")
    state = multi_pass_retrieve(state)

    # ── Stage 3: Stream attribution synthesis ─────────────────────────────────
    _progress("✍️ Generating response...")
    context = format_context(state["all_chunks"])

    speaker_instructions = ""
    if state["speaker_map"]:
        lines = [
            f"  - {sp}: statements must be cited from {', '.join(srcs)}"
            for sp, srcs in state["speaker_map"].items()
        ]
        speaker_instructions = (
            "\n\nIDENTIFIED SPEAKERS IN RETRIEVED DOCUMENTS — STRICT ATTRIBUTION REQUIRED:\n"
            + "\n".join(lines)
            + "\n\nFor each speaker above, cite ONLY their own deposition/document. "
            "NEVER attribute one speaker's words to another document."
        )

    agentic_system = SYSTEM_PROMPT + speaker_instructions + """

AGENTIC RAG MODE — ADDITIONAL REQUIREMENTS:
- This response was generated after multi-pass retrieval across sub-queries
- You have richer context than standard mode — use it to answer ALL parts of the question
- If the question has multiple parts (A, B, C), answer each part explicitly
- Identify and explicitly flag ANY contradictions between documents
- Do not collapse multi-part answers into a single vague response
- If a sub-question cannot be answered from the retrieved context, say so explicitly"""

    messages = [
        SystemMessage(content=agentic_system),
        HumanMessage(content=(
            f"CONTEXT FROM DOCUMENTS:\n{context}\n\n"
            f"QUESTION: {state['query']}\n\n"
            "Answer completely, with strict per-speaker attribution and explicit "
            "citations for every factual claim."
        )),
    ]

    full_answer = ""
    for chunk in llm.stream(messages):
        token = chunk.content
        if token:
            full_answer += token
            yield token

    state["draft_response"] = full_answer

    # ── Stage 4: Self-critique (silent — runs after last yield) ───────────────
    state = self_critique(state)

    # ── Stage 5: Optional revision ────────────────────────────────────────────
    if should_revise(state) == "revise":
        _progress("🔄 Revising response...")
        context2 = format_context(state["all_chunks"])
        messages2 = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=(
                f"CONTEXT FROM DOCUMENTS:\n{context2}\n\n"
                f"ORIGINAL QUESTION: {state['query']}\n\n"
                f"PREVIOUS DRAFT (with errors):\n{state['draft_response']}\n\n"
                f"ERRORS IDENTIFIED:\n{state['critique_notes']}\n\n"
                "Please rewrite the response correcting ALL identified errors. "
                "Maintain strict per-speaker attribution and cite every factual claim."
            )),
        ]
        revised = ""
        for chunk in llm.stream(messages2):
            token = chunk.content
            if token:
                revised += token
        state["draft_response"] = revised
        state["revision_count"] = 1

    state["final_response"] = state["draft_response"]

    # ── Populate result_holder and cache ─────────────────────────────────────
    result = {
        "question":       question,
        "answer":         state["final_response"],
        "sources":        _build_sources(state["all_chunks"]),
        "chunks":         state["all_chunks"],
        "chunks_used":    len(state["all_chunks"]),
        "sub_queries":    state["sub_queries"],
        "speaker_map":    state["speaker_map"],
        "critique":       state["critique_notes"],
        "revision_count": state["revision_count"],
    }
    if result_holder is not None:
        result_holder.update(result)
    set_cached_query(question, "agentic", top_k, result)


# ── CLI Test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
        "What did Marcus Webb and Richard Calloway each say about Chen's performance, and do their accounts contradict each other?"

    print(f"\nQuery: {query}")
    print("=" * 70)
    result = agentic_rag_query(query, top_k=7)
    print(f"Sub-queries used: {result['sub_queries']}")
    print(f"Speakers identified: {list(result['speaker_map'].keys())}")
    print(f"Chunks retrieved: {result['chunks_used']}")
    print(f"Revisions: {result['revision_count']}")
    print(f"Critique: {result['critique'][:200]}")
    print(f"\nAnswer:\n{result['answer']}")