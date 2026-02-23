# modules/retrieval.py
# Semantic search and retrieval from ChromaDB vector store

from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import chromadb

from modules.ingestion import CHROMA_PATH, COLLECTION_NAME, EMBEDDING_MODEL

# ── Initialize Components ─────────────────────────────────────────────────────
embedding_model = SentenceTransformer(EMBEDDING_MODEL, cache_folder="./db/embeddings")

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)


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
            "score":    round(1 - results["distances"][0][i], 4)  # cosine similarity
        })

    return chunks


def format_context(chunks: List[Dict[str, Any]]) -> str:
    """
    Format retrieved chunks into a clean context block for the LLM.
    Includes source and page citation for every chunk.
    """
    if not chunks:
        return "No relevant documents found."

    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        context_parts.append(
            f"[{i}] Source: {chunk['source']} | Page: {chunk['page']} "
            f"| Relevance: {chunk['score']}\n{chunk['text']}"
        )

    return "\n\n---\n\n".join(context_parts)


def retrieve_and_format(query: str, top_k: int = 5) -> tuple[str, List[Dict]]:
    """
    Convenience function — retrieves and formats in one call.
    Returns (formatted_context_string, raw_chunks_list)
    """
    chunks = retrieve(query, top_k=top_k)
    context = format_context(chunks)
    return context, chunks


if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is this case about?"

    print(f"\nQuery: {query}")
    print("=" * 60)

    chunks = retrieve(query, top_k=3)

    if not chunks:
        print("No results found — is the vector store populated?")
    else:
        for i, chunk in enumerate(chunks, start=1):
            print(f"\n[{i}] {chunk['source']} | Page {chunk['page']} | Score: {chunk['score']}")
            print(f"    {chunk['text'][:200]}...")