# modules/llm.py
# Local LLM interface via Ollama (OpenAI-compatible endpoint)

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# ── Model Configuration ──────────────────────────────────────────────────────
PRIMARY_MODEL = "llama3.1:8b"
FALLBACK_MODEL = "phi4:14b"  # benchmark candidate

SYSTEM_PROMPT = """You are a legal document analysis assistant helping a 
litigation attorney during the discovery phase of a case. You are precise, 
thorough, and always cite your sources. When referencing information from 
documents, always include the document name and page number and any date information. Never fabricate 
information. If the context provided does not contain enough information to 
answer the question, say so clearly."""


def get_llm(model: str = PRIMARY_MODEL, temperature: float = 0.0) -> ChatOllama:
    """
    Returns a ChatOllama instance connected to the local Ollama server.
    temperature=0.0 for maximum consistency in legal analysis.
    """
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

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
    ]

    if context:
        messages.append(HumanMessage(content=f"""Use the following document 
excerpts to answer the question. Cite the source and page number for each 
piece of information you use.

DOCUMENT CONTEXT:
{context}

QUESTION:
{prompt}"""))
    else:
        messages.append(HumanMessage(content=prompt))

    response = llm.invoke(messages)
    return response.content


def test_connection() -> bool:
    """
    Quick health check — confirms Ollama is running and model is responsive.
    """
    try:
        result = query_llm("Respond with exactly three words: Connection is confirmed.")
        return len(result) > 0
    except Exception as e:
        print(f"Ollama connection failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing Ollama connection...")
    if test_connection():
        print("✅ LLM connection confirmed — Llama 3.1 8B is ready.")
    else:
        print("❌ Connection failed — is Ollama running?")