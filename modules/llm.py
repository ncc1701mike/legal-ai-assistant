# modules/llm.py
# Local LLM interface via Ollama + RAG query engine

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from typing import List, Dict, Any

# ── Model Configuration ───────────────────────────────────────────────────────
PRIMARY_MODEL = "llama3.1:8b"
FALLBACK_MODEL = "phi4:14b"

SYSTEM_PROMPT = """You are a very bright and amazing legal document analysis assistant helping a \
litigation attorney during the discovery phase of a case. You are precise, \
thorough, and always cite your sources. 

When referencing information from documents, always include the document name \
and page number in your response using this format: (Source: filename, Page X)

Never fabricate information. If the context provided does not contain enough \
information to answer the question, say so clearly and explicitly. Do not guess."""


# ── Core LLM Interface ────────────────────────────────────────────────────────
def get_llm(model: str = PRIMARY_MODEL, temperature: float = 0.0) -> ChatOllama:
    """Returns a ChatOllama instance connected to the local Ollama server."""
    return ChatOllama(
        model=model,
        temperature=temperature,
        base_url="http://localhost:11434"
    )


def query_llm(prompt: str, context: str = "", model: str = PRIMARY_MODEL) -> str:
    """
    Send a query to the local LLM with optional RAG context.
    Returns the model's response as a string.
    """
    llm = get_llm(model=model)

    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    if context:
        messages.append(HumanMessage(content=f"""Use the following document \
excerpts to answer the question. Cite the source filename and page number for \
each piece of information you use.

DOCUMENT CONTEXT:
{context}

QUESTION:
{prompt}

Provide a clear, structured answer with citations."""))
    else:
        messages.append(HumanMessage(content=prompt))

    response = llm.invoke(messages)
    return response.content


# ── RAG Query Engine ──────────────────────────────────────────────────────────
def rag_query(query: str, top_k: int = 5, model: str = PRIMARY_MODEL) -> Dict[str, Any]:
    """
    Full RAG pipeline in one call:
    1. Retrieve relevant chunks from ChromaDB
    2. Format as context
    3. Query local LLM with context
    4. Return answer + source citations

    Returns dict with answer, sources, and raw chunks.
    """
    from modules.retrieval import retrieve_and_format

    context, chunks = retrieve_and_format(query, top_k=top_k)
    answer = query_llm(query, context=context, model=model)

    # Build clean citations list
    sources = []
    seen = set()
    for chunk in chunks:
        key = f"{chunk['source']}|{chunk['page']}"
        if key not in seen:
            sources.append({
                "file": chunk["source"],
                "page": chunk["page"],
                "score": chunk["score"]
            })
            seen.add(key)

    return {
        "query": query,
        "answer": answer,
        "sources": sources,
        "chunks_used": len(chunks)
    }


def summarize_documents(document_names: List[str] = None) -> str:
    """
    Generate an executive summary of all ingested documents or a subset.
    """
    from modules.retrieval import retrieve_and_format

    summary_query = "Provide a comprehensive summary of all key facts, \
parties, claims, dates, and legal issues present in these documents."

    context, chunks = retrieve_and_format(summary_query, top_k=10)

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