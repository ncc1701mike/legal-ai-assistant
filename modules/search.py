# modules/search.py
# Case Law Search — CourtListener API (Free, No Key Required)
#
# ⚠️  DEPLOYMENT NOTE — READ BEFORE ENABLING  ⚠️
# This module makes outbound HTTPS requests to api.courtlistener.com.
# It is DISABLED by default in Shenelle's local deployment to preserve
# air-gap security. To enable, uncomment the import in app.py (Tab 4).
#
# Architecture rationale: Built as a self-contained bolt-on module with
# zero coupling to the RAG pipeline. Enabling or disabling it has no
# effect on document ingestion, retrieval, or redaction.

import requests
import logging
from typing import List, Dict, Any, Optional

COURTLISTENER_BASE = "https://www.courtlistener.com/api/rest/v4"
REQUEST_TIMEOUT = 10  # seconds


def search_case_law(
    query: str,
    court: Optional[str] = None,
    max_results: int = 5
) -> List[Dict[str, Any]]:
    """
    Search CourtListener for relevant case law opinions.

    Args:
        query:       Natural language or keyword search query
        court:       Optional court filter e.g. "scotus", "ca2", "nyed"
                     Full list: https://www.courtlistener.com/api/rest/v4/courts/
        max_results: Number of results to return (max 20)

    Returns:
        List of dicts with keys: case_name, citation, court, date_filed,
        summary, url, download_url
    """
    params = {
        "q": query,
        "type": "o",           # opinions only
        "order_by": "score desc",
        "page_size": max_results,
        "format": "json",
    }

    if court:
        params["court"] = court

    try:
        response = requests.get(
            f"{COURTLISTENER_BASE}/search/",
            params=params,
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": "AmicusAI/1.0 (Legal Research Tool)"}
        )
        response.raise_for_status()
        data = response.json()

    except requests.exceptions.ConnectionError:
        return [{"error": "No internet connection. Case law search requires network access."}]
    except requests.exceptions.Timeout:
        return [{"error": "CourtListener API timed out. Try again in a moment."}]
    except requests.exceptions.HTTPError as e:
        return [{"error": f"CourtListener API error: {e}"}]
    except Exception as e:
        logging.error(f"Case law search failed: {e}")
        return [{"error": f"Search failed: {str(e)}"}]

    results = []
    for hit in data.get("results", []):
        # Extract citation — CourtListener returns a list
        citations = hit.get("citation", [])
        citation_str = citations[0] if citations else "Citation unavailable"

        results.append({
            "case_name":    hit.get("caseName", "Unknown"),
            "citation":     citation_str,
            "court":        hit.get("court", "Unknown court"),
            "date_filed":   hit.get("dateFiled", "Unknown date"),
            "summary":      _truncate(hit.get("snippet", "No summary available"), 400),
            "url":          f"https://www.courtlistener.com{hit.get('absolute_url', '')}",
            "status":       hit.get("status", ""),
        })

    return results


def lookup_citation(citation: str) -> Dict[str, Any]:
    """
    Look up a specific case by citation string.
    e.g. lookup_citation("737 F.3d 834")

    Returns case details or an error dict.
    """
    params = {
        "q": f'"{citation}"',
        "type": "o",
        "page_size": 1,
        "format": "json",
    }

    try:
        response = requests.get(
            f"{COURTLISTENER_BASE}/search/",
            params=params,
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": "AmicusAI/1.0 (Legal Research Tool)"}
        )
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])

        if not results:
            return {"error": f"No case found for citation: {citation}"}

        hit = results[0]
        citations = hit.get("citation", [])
        return {
            "case_name":  hit.get("caseName", "Unknown"),
            "citation":   citations[0] if citations else citation,
            "court":      hit.get("court", "Unknown court"),
            "date_filed": hit.get("dateFiled", "Unknown date"),
            "summary":    _truncate(hit.get("snippet", ""), 600),
            "url":        f"https://www.courtlistener.com{hit.get('absolute_url', '')}",
        }

    except requests.exceptions.ConnectionError:
        return {"error": "No internet connection."}
    except Exception as e:
        return {"error": str(e)}


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "…"
