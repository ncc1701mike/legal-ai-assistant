# modules/llm.py
# Local LLM interface via Ollama + RAG query engine

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable
from typing import List, Dict, Any
from modules.cache import get_cached_query, set_cached_query



# ── Model Configuration ───────────────────────────────────────────────────────
PRIMARY_MODEL = "llama3.1:8b"
FALLBACK_MODEL = "phi4:14b"

SYSTEM_PROMPT = """You are an exceptionally precise legal document analyst helping a litigation \
attorney during the discovery phase of a case. You analyze documents with the rigor of a \
senior associate preparing for trial. Your primary obligations are accuracy, strict source \
attribution, and intellectual honesty about the limits of the available evidence.

CRITICAL RULES FOR USING DOCUMENT CONTEXT:

1. DOCUMENT IDENTIFICATION
   Each chunk is labeled [DOCUMENT N] with FILE, DESCRIPTION, and PAGE.
   Use the DESCRIPTION field to understand what kind of document you are citing:
   - "Deposition / Witness Testimony" = one witness's sworn account under oath
   - "Court Order / Judicial Ruling" = the judge's analysis and holdings
   - "Internal Memorandum / Email" = internal communications, potentially privileged
   - "Legal Complaint" = allegations made by plaintiff's counsel — not proven facts
   - "Answer / Affirmative Defense" = defendant's denials — not admissions unless explicit
   - "Performance Review / HR Document" = employer's contemporaneous record
   - "Accommodation Request / Denial" = formal HR correspondence

2. STRICT WITNESS ATTRIBUTION — THIS IS NON-NEGOTIABLE
   When multiple witnesses address the same event, you MUST:
   a) Identify each witness by name and document source separately
   b) Quote or closely paraphrase their specific words — do not merge accounts
   c) Never attribute one witness's statement to another witness
   d) Never attribute a complaint's characterization to a witness's own testimony
   e) If Webb said X in his deposition, cite the Webb deposition — not the complaint's
      description of what Webb said
   EXAMPLE OF CORRECT ATTRIBUTION:
   "Chen testified that Webb stated '...' (02_chen_deposition.txt). Webb, in his own
   deposition, stated '...' (03_webb_deposition.txt). These accounts differ in that..."

3. CONTRADICTION HANDLING
   When documents contain conflicting accounts of the same event:
   a) Present each party's version separately and completely before drawing conclusions
   b) Identify the specific factual point of disagreement with precision
   c) Note what documentary evidence exists to support each version
   d) Do NOT resolve contradictions by picking a side — present both fully
   e) Do NOT assume the complaint's characterization is accurate — it is advocacy
   f) The deposition transcript is the primary source for a witness's account,
      not a summary in a complaint or answer

4. DOCUMENT HIERARCHY FOR ATTRIBUTION
   When the same fact appears in multiple documents, prioritize in this order:
   PRIMARY:   Deposition transcripts (witness's own sworn words)
   PRIMARY:   Internal emails and memos (contemporaneous records)
   SECONDARY: Expert reports (opinion based on primary sources)
   TERTIARY:  Complaints and answers (advocacy documents — characterizations only)
   Use primary sources whenever available. If you only have a complaint's description
   of a deposition but not the deposition itself, say so explicitly.

5. HANDLING MISSING INFORMATION
   If the retrieved context does not contain enough information to fully answer:
   a) Answer what you can from the available evidence
   b) Explicitly state what information is missing or not found in the documents
   c) Do NOT fill gaps with inferences, assumptions, or general legal knowledge
   d) Do NOT confabulate facts, dates, amounts, or outcomes not in the documents
   e) Saying "the documents do not address this" is correct and valuable

6. NUMERICAL AND DATE PRECISION
   When citing salaries, damages, dates, or case numbers:
   a) State the exact figure from the document — do not approximate
   b) If multiple figures appear, explain what each represents
   c) Distinguish between gross and net figures when both are present
   d) Always state the source document for any specific number

7. CASE OUTCOME AWARENESS
   Do not state or imply outcomes that are not documented:
   - Do not state settlement amounts unless a settlement document is in the corpus
   - Do not state trial outcomes unless a verdict or judgment is in the corpus
   - Do not state that claims succeeded or failed unless a court order says so
   - Active litigation with a future trial date means no outcome has occurred yet

8. SOURCE CITATION FORMAT
   Cite every factual claim: (Source: filename.txt, Page X)
   For contradictions cite both sources: (Source: filename_A.txt vs filename_B.txt)
   For synthesized cross-document findings, cite all contributing sources."""
   
# ── Core LLM Interface ────────────────────────────────────────────────────────
def get_llm(model: str = PRIMARY_MODEL, temperature: float = 0.0) -> ChatOllama:
    """Returns a ChatOllama instance connected to the local Ollama server."""
    return ChatOllama(
        model=model,
        temperature=temperature,
        base_url="http://localhost:11434"
    )


@traceable(name="query_llm")
def query_llm(prompt: str, context: str = "", model: str = PRIMARY_MODEL,
              multihop: bool = False) -> str:
    """
    Send a query to the local LLM with optional RAG context.
    Returns the model's response as a string.
    """
    llm = get_llm(model=model)
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    if context:
        if multihop:
            instruction = """You have been provided with evidence gathered across \
multiple retrieval passes from several legal documents. Some chunks are from the \
initial retrieval, others are from targeted follow-up searches designed to fill \
specific gaps.

Your task is to SYNTHESIZE across all documents and DRAW INFERENCES where the \
evidence permits. Specifically:
- If Document A recommends action X and Document B shows action Y occurred instead, \
explicitly state that the recommendation was not followed and cite both documents.
- If a timeline can be reconstructed across documents, reconstruct it.
- If the corpus does not contain evidence of an outcome, say so clearly — but first \
check whether the outcome can be INFERRED from what IS present.
- Do not summarize each document separately. Synthesize them into a single coherent answer.

Cite every factual claim with (Source: filename, Page X)."""
        else:
            instruction = """Use the following document excerpts to answer the \
question. Cite the source filename and page number for each piece of information \
you use. Provide a clear, structured answer with citations."""

        messages.append(HumanMessage(content=f"""{instruction}

DOCUMENT CONTEXT:
{context}

QUESTION:
{prompt}"""))
    else:
        messages.append(HumanMessage(content=prompt))
    response = llm.invoke(messages)
    return response.content


# ── RAG Query Engine ──────────────────────────────────────────────────────────

@traceable(name="rag_query")
def rag_query(question: str, top_k: int = 5, 
              mode: str = "hybrid") -> Dict[str, Any]:
    """
    Full RAG pipeline in one call:
    1. Retrieve relevant chunks from ChromaDB
    2. Format as context
    3. Query local LLM with context
    4. Return answer + source citations

    Returns dict with answer, sources, and raw chunks.
    """
    from modules.retrieval import retrieve_and_format

    # Check cache first
    cached = get_cached_query(question, mode, top_k)
    if cached is not None:
        cached["from_cache"] = True
        return cached

    from modules.retrieval import retrieve_and_format
    context, chunks = retrieve_and_format(question, top_k=top_k, mode=mode)
    answer = query_llm(question, context=context, multihop=(mode == "multihop"))
    sources = []
    seen = set()
    for chunk in chunks:
        key = f"{chunk['source']}|{chunk['page']}"
        if key not in seen:
            sources.append({
                "file": chunk["source"],
                "page": chunk["page"],
                "score": chunk.get("rrf_score", chunk.get("score", 0)),
                "rerank_score": chunk.get("rerank_score") or chunk.get("rrf_score") or chunk.get("score")
            })
            seen.add(key)
    result = {
        "question": question,
        "answer": answer,
        "sources": sources,
        "chunks": chunks,
        "chunks_used": len(chunks)
    }
    set_cached_query(question, mode, top_k, result)
    return result

@traceable(name="summarize_documents")
def summarize_documents(document_names: List[str] = None) -> str:
    """
    Generate an executive summary of all ingested documents or a subset.
    """
    from modules.retrieval import retrieve_and_format

    summary_query = "parties names dates evidence testimony exhibits claims jurisdiction"

    context, chunks = retrieve_and_format(summary_query, top_k=15)

    if document_names:
        filtered = [c for c in chunks if c["source"] in document_names]
        from modules.retrieval import format_context
        context = format_context(filtered) if filtered else context

    prompt = f"""Based on the following legal documents, provide a structured 
executive summary that covers:
1. Key parties involved
2. Main legal claims or issues
3. Important dates and deadlines
4. Critical facts established
5. Any notable gaps or areas requiring further review

{context}"""

    return query_llm(prompt, model=PRIMARY_MODEL)


def test_connection() -> bool:
    """Quick health check — confirms Ollama is running and model is responsive."""
    try:
        result = query_llm("Respond with exactly three words: Connection is confirmed.")
        return len(result) > 0
    except Exception as e:
        print(f"Ollama connection failed: {e}")
        return False


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"\nRAG Query: {query}")
        print("=" * 60)
        result = rag_query(query)
        print(f"\nANSWER:\n{result['answer']}")
        print(f"\nSOURCES:")
        for s in result["sources"]:
            print(f"  • {s['file']} — Page {s['page']} (relevance: {s['score']})")
    else:
        print("Testing connection...")
        if test_connection():
            print("✅ LLM connection confirmed.")
        else:
            print("❌ Connection failed.")