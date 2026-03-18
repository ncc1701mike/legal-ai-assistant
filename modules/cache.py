# modules/cache.py
# Persistent disk-based query cache using diskcache
# Survives between sessions — invalidated when document store is cleared

import hashlib
import json
import logging
import diskcache

logger = logging.getLogger(__name__)

CACHE_DIR = "./db/cache"
CACHE_TTL = 60 * 60 * 24 * 7  # 7 days

_cache = diskcache.Cache(CACHE_DIR)

def _make_key(prefix: str, **kwargs) -> str:
    """Generate a stable cache key from function arguments."""
    payload = json.dumps(kwargs, sort_keys=True)
    digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return f"{prefix}:{digest}"

def get_cached_query(question: str, mode: str, top_k: int):
    """Retrieve cached rag_query result if available."""
    key = _make_key("rag_query", question=question, mode=mode, top_k=top_k)
    result = _cache.get(key)
    if result is not None:
        logger.info(f"Cache HIT — {key}")
    return result

def set_cached_query(question: str, mode: str, top_k: int, result: dict):
    """Store rag_query result in persistent cache."""
    key = _make_key("rag_query", question=question, mode=mode, top_k=top_k)
    _cache.set(key, result, expire=CACHE_TTL)
    logger.info(f"Cache SET — {key}")

def get_cached_retrieval(query: str, mode: str, top_k: int):
    """Retrieve cached retrieve_and_format result if available."""
    key = _make_key("retrieval", query=query, mode=mode, top_k=top_k)
    return _cache.get(key)

def set_cached_retrieval(query: str, mode: str, top_k: int, result: tuple):
    """Store retrieve_and_format result in persistent cache."""
    key = _make_key("retrieval", query=query, mode=mode, top_k=top_k)
    _cache.set(key, result, expire=CACHE_TTL)

def clear_cache():
    """Clear all cached results — call when document store is cleared."""
    _cache.clear()
    logger.info("Cache cleared")

def cache_stats() -> dict:
    """Return basic cache statistics."""
    return {
        "size": len(_cache),
        "volume_mb": round(_cache.volume() / 1024 / 1024, 2)
    }