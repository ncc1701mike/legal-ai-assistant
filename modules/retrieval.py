# modules/retrieval.py
# Semantic search and retrieval from ChromaDB vector store
# Enhanced with HyDE, BM25, and Hybrid (RRF) retrieval strategies

import os
import logging
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import chromadb
from rank_bm25 import BM25Okapi

from modules.ingestion import CHROMA_PATH, COLLECTION_NAME, EMBEDDING_MODEL

# ── Initialize Components ─────────────────────────────────────────────────────
embedding_model = SentenceTransformer(EMBEDDING_MODEL, cache_folder="./db/embeddings")

# At module level, after embedding_model initialization
_cross_encoder = None

def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
    return _cross_encoder

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)


# ── Strategy 1: Standard Vector Retrieval ────────────────────────────────────
def retrieve(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Embed the query locally and search ChromaDB for the most relevant chunks.
    Returns a list of results with text, source, page, and similarity score.
    """
    query_embedding = embedding_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text":     results["documents"][0][i],
            "source":   results["metadatas"][0][i]["source"],
            "page":     results["metadatas"][0][i]["page"],
            "score":    round(1 - results["distances"][0][i], 4),
            "method":   "vector"
        })

    return chunks


# ── Strategy 2: HyDE Retrieval ────────────────────────────────────────────────
def hyde_retrieve(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    hypothetical = _generate_hypothetical_answer(query)
    hyde_embedding = embedding_model.encode(hypothetical).tolist()

    # Fetch large pool to ensure diversity across documents
    fetch_n = min(top_k * 6, collection.count())

    results = collection.query(
        query_embeddings=[hyde_embedding],
        n_results=fetch_n,
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text":         results["documents"][0][i],
            "source":       results["metadatas"][0][i]["source"],
            "page":         results["metadatas"][0][i]["page"],
            "score":        round(1 - results["distances"][0][i], 4),
            "method":       "hyde",
            "hypothetical": hypothetical
        })

    return diversify_results(chunks, top_k=top_k, max_per_source=2)

# ── Generate Hypothetical Answer ─────────────────────────────────────────────
def _generate_hypothetical_answer(query: str) -> str:
    from langchain_ollama import ChatOllama
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = ChatOllama(model="llama3.1:8b", temperature=0.3)

    messages = [
        SystemMessage(content=(
            "You are a legal document assistant. Given a question about legal "
            "documents, write a short hypothetical passage (4-6 sentences) that "
            "directly answers the question. The passage should mention multiple "
            "types of legal proceedings, documents, or concepts that would appear "
            "across different legal documents such as transcripts, complaints, "
            "depositions, settlements, and memos. Use formal legal language. "
            "Write only the passage with no preamble or explanation."
        )),
        HumanMessage(content=f"Question: {query}")
    ]

    response = llm.invoke(messages)
    return response.content.strip()

# ── Strategy 3: BM25 Keyword Retrieval ───────────────────────────────────────
def bm25_retrieve(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    all_chunks = _get_all_chunks()
    if not all_chunks:
        return []

    tokenized_corpus = [doc["text"].lower().split() for doc in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    import numpy as np
    # Fetch larger pool before diversifying
    fetch_n = min(top_k * 6, len(all_chunks))
    top_indices = np.argsort(scores)[::-1][:fetch_n]

    chunks = []
    for idx in top_indices:
        if scores[idx] > 0:
            chunks.append({
                "text":   all_chunks[idx]["text"],
                "source": all_chunks[idx]["source"],
                "page":   all_chunks[idx]["page"],
                "score":  round(float(scores[idx]), 4),
                "method": "bm25"
            })

    return diversify_results(chunks, top_k=top_k, max_per_source=2)


def _get_all_chunks() -> List[Dict[str, Any]]:
    """Fetch all chunks from ChromaDB for BM25 indexing."""
    try:
        count = collection.count()
        if count == 0:
            return []

        results = collection.get(
            limit=count,
            include=["documents", "metadatas"]
        )

        chunks = []
        for i in range(len(results["documents"])):
            chunks.append({
                "text":   results["documents"][i],
                "source": results["metadatas"][i]["source"],
                "page":   results["metadatas"][i]["page"],
            })
        return chunks

    except Exception as e:
        logging.error(f"Error fetching chunks for BM25: {e}")
        return []


# ── Strategy 4: Hybrid RRF Retrieval ─────────────────────────────────────────
def hybrid_retrieve(query: str, top_k: int = 5,
                    use_hyde: bool = True) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval using Reciprocal Rank Fusion (RRF).
    
    Combines:
    - HyDE or standard vector retrieval (semantic understanding)
    - BM25 retrieval (exact keyword matching)
    
    RRF formula: score(d) = Σ 1 / (k + rank(d))
    where k=60 is a constant that reduces the impact of high rankings.
    
    This is the recommended strategy for legal documents — it catches
    both conceptual queries (HyDE) and exact legal terms (BM25).
    """
    # Fetch from both strategies with larger pools
    fetch_k = top_k * 2

    if use_hyde:
        try:
            semantic_chunks = hyde_retrieve(query, top_k=fetch_k)
            
        except Exception as e:
            print(f"DEBUG HyDE failed: {e}, falling back to vector")
            semantic_chunks = retrieve(query, top_k=fetch_k)
    else:
        semantic_chunks = retrieve(query, top_k=fetch_k)

    bm25_chunks = bm25_retrieve(query, top_k=fetch_k)

    # Reciprocal Rank Fusion
    k = 60  # RRF constant
    rrf_scores = {}
    chunk_map = {}

    # Score semantic results
    for rank, chunk in enumerate(semantic_chunks):
        key = f"{chunk['source']}|{chunk['page']}|{chunk['text'][:50]}"
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (k + rank + 1)
        chunk_map[key] = chunk

    # Score BM25 results
    for rank, chunk in enumerate(bm25_chunks):
        key = f"{chunk['source']}|{chunk['page']}|{chunk['text'][:50]}"
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (k + rank + 1)
        if key not in chunk_map:
            chunk_map[key] = chunk

    # Score semantic results — weighted 2x vs BM25
    for rank, chunk in enumerate(semantic_chunks):
        key = f"{chunk['source']}|{chunk['page']}|{chunk['text'][:50]}"
        rrf_scores[key] = rrf_scores.get(key, 0) + 2 / (k + rank + 1)
        chunk_map[key] = chunk

    # Score BM25 results — standard weight
    for rank, chunk in enumerate(bm25_chunks):
        key = f"{chunk['source']}|{chunk['page']}|{chunk['text'][:50]}"
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (k + rank + 1)
        if key not in chunk_map:
            chunk_map[key] = chunk

    # Sort by RRF score and return top_k
    sorted_keys = sorted(rrf_scores.keys(),
                         key=lambda x: rrf_scores[x], reverse=True)

    results = []
    for key in sorted_keys[:top_k]:
        chunk = chunk_map[key].copy()
        chunk["rrf_score"] = round(rrf_scores[key], 6)
        chunk["method"] = "hybrid_rrf"
        results.append(chunk)

    return results


# ── Strategy 5: Hybrid + Cross-Encoder Reranker ───────────────────────────
def rerank_retrieve(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Three-stage retrieval pipeline:
    1. Hybrid RRF fetches top 20 candidates (broad recall)
    2. Cross-encoder reranker scores all 20 against the actual query
    3. Return top_k reranked results (precision)
    
    The cross-encoder reads query+chunk together using full cross-attention,
    producing a much more accurate relevance score than cosine similarity.
    Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (runs locally, ~80MB)
    """
    from sentence_transformers import CrossEncoder

    # Stage 1: Get broad candidate pool from hybrid
    candidates = hybrid_retrieve(query, top_k=20, use_hyde=True)

    if not candidates:
        return retrieve(query, top_k=top_k)

    # Stage 2: Rerank with cross-encoder
    cross_encoder = _get_cross_encoder()

    pairs = [(query, chunk["text"]) for chunk in candidates]
    scores = cross_encoder.predict(pairs)

    # Attach rerank scores
    for chunk, score in zip(candidates, scores):
        chunk["rerank_score"] = round(float(score), 6)
        chunk["method"] = "hybrid_rerank"

    # Stage 3: Sort by rerank score, return top_k
    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]

# ── Diversify Results ───────────────────────────────────────────────────────
def diversify_results(chunks: List[Dict[str, Any]], 
                      top_k: int = 5,
                      max_per_source: int = 2) -> List[Dict[str, Any]]:
    """
    Ensure no single document dominates results.
    Limits each source to max_per_source chunks in the final result set.
    Critical when documents have very different chunk counts.
    """
    seen_sources = {}
    diversified = []
    
    for chunk in chunks:
        source = chunk["source"]
        count = seen_sources.get(source, 0)
        if count < max_per_source:
            diversified.append(chunk)
            seen_sources[source] = count + 1
        if len(diversified) >= top_k:
            break
    
    return diversified
# ── Formatting ────────────────────────────────────────────────────────────────
def format_context(chunks: List[Dict[str, Any]]) -> str:
    """
    Format retrieved chunks into a clean context block for the LLM.
    Includes source, page, and citation for every chunk.
    """
    if not chunks:
        return "No relevant documents found."

    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        score = chunk.get("rrf_score", chunk.get("score", 0))
        method = chunk.get("method", "vector")
        context_parts.append(
            f"[{i}] Source: {chunk['source']} | Page: {chunk['page']} "
            f"| Relevance: {score} | Method: {method}\n{chunk['text']}"
        )

    return "\n\n---\n\n".join(context_parts)


def retrieve_and_format(query: str, top_k: int = 5,
                        mode: str = "hybrid") -> Tuple[str, List[Dict]]:
    """
    Convenience function — retrieves and formats in one call.
    
    Args:
        query: Search query
        top_k: Number of results to return
        mode: "vector" | "hyde" | "bm25" | "hybrid"
    
    Returns:
        (formatted_context_string, raw_chunks_list)
    """
    
    if mode == "hyde":
        chunks = hyde_retrieve(query, top_k=top_k * 2)
    elif mode == "bm25":
        chunks = bm25_retrieve(query, top_k=top_k * 2)
    elif mode == "hybrid":
        chunks = hybrid_retrieve(query, top_k=top_k * 2, use_hyde=True)
    elif mode == "rerank":
        chunks = rerank_retrieve(query, top_k=top_k)
        return format_context(chunks), chunks  # skip diversify — reranker handles it
    else:
        chunks = retrieve(query, top_k=top_k * 2)

    # Prevent any single document from dominating
    chunks = diversify_results(chunks, top_k=top_k, max_per_source=2)
    context = format_context(chunks)
    return context, chunks


# ── CLI Test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is this case about?"

    print(f"\nQuery: {query}")
    print("=" * 60)

    print("\n--- VECTOR ---")
    for chunk in retrieve(query, top_k=3):
        print(f"  [{chunk['score']}] {chunk['source']} p{chunk['page']}: "
              f"{chunk['text'][:120]}...")

    print("\n--- BM25 ---")
    for chunk in bm25_retrieve(query, top_k=3):
        print(f"  [{chunk['score']}] {chunk['source']} p{chunk['page']}: "
              f"{chunk['text'][:120]}...")

    print("\n--- HYBRID RRF ---")
    for chunk in hybrid_retrieve(query, top_k=3):
        print(f"  [{chunk['rrf_score']}] {chunk['source']} p{chunk['page']} "
              f"({chunk['method']}): {chunk['text'][:120]}...")