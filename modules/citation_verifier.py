"""
modules/citation_verifier.py
Citation verification layer for the legal AI assistant.

Checks every source reference in a generated response against the actual
retrieved chunks before the response is shown to the user.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class CitationReport:
    verified_citations: List[Dict]    # citations found in both response and chunks
    unverified_citations: List[Dict]  # citations in response but NOT in retrieved chunks
    missing_citations: List[str]      # specific factual claims with no citation found
    overall_confidence_score: float   # 0.0 – 1.0


# ── Regex patterns ────────────────────────────────────────────────────────────

# (Source: filename.txt, Page 5)  or  (Source: filename.txt, Page 5-7)
_SOURCE_PAGE_RE = re.compile(
    r'\(Source:\s*([^,)]+?)\s*,\s*Page\s+(\d+)(?:-\d+)?\)',
    re.IGNORECASE,
)

# (Source: file_a.txt vs file_b.txt)  — contradiction citations
_SOURCE_VS_RE = re.compile(
    r'\(Source:\s*([^\s)]+)\s+vs\.?\s+([^\s)]+)\)',
    re.IGNORECASE,
)

# Specific factual claim heuristic patterns
_DATE_RE = re.compile(
    r'\b(?:January|February|March|April|May|June|July|August|'
    r'September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
    r'|\b\d{1,2}/\d{1,2}/\d{2,4}\b',
    re.IGNORECASE,
)
_DOLLAR_RE = re.compile(
    r'\$[\d,]+(?:\.\d{1,2})?(?:\s*(?:million|billion|thousand))?',
    re.IGNORECASE,
)
_NAMED_ACTION_RE = re.compile(
    r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s+'
    r'(?:stated|testified|said|wrote|emailed|signed|approved|denied|filed|reported|confirmed)\b'
)

_CLAIM_PATTERNS = [_DATE_RE, _DOLLAR_RE, _NAMED_ACTION_RE]


# ── Core functions ────────────────────────────────────────────────────────────

def parse_citations(text: str) -> List[Dict]:
    """
    Extract all citation references from response text.

    Recognises:
      - (Source: filename.txt, Page 5)
      - (Source: filename.txt, Page 5-7)
      - (Source: file_a.txt vs file_b.txt)

    Returns a list of dicts:
        {'filename': str, 'page': Optional[int], 'raw': str}
    """
    citations: List[Dict] = []

    for match in _SOURCE_PAGE_RE.finditer(text):
        citations.append({
            'filename': match.group(1).strip(),
            'page': int(match.group(2)),
            'raw': match.group(0),
        })

    for match in _SOURCE_VS_RE.finditer(text):
        for fname in (match.group(1).strip(), match.group(2).strip()):
            citations.append({
                'filename': fname,
                'page': None,
                'raw': match.group(0),
            })

    return citations


def verify_single_citation(
    filename: str,
    page: Optional[int],
    chunks: List[Dict],
) -> bool:
    """
    Check whether a specific citation exists in the retrieved chunk list.

    Matching strategy:
      1. Exact filename + page match.
      2. Case-insensitive substring match on filename (handles path prefixes,
         minor naming differences, etc.).
      3. If page is None (e.g. vs-format citations), match on filename alone.

    Returns True if the citation is traceable to at least one retrieved chunk.
    """
    if not chunks:
        return False

    fn_lower = filename.lower().strip()

    for chunk in chunks:
        chunk_source = str(chunk.get('source', '')).lower().strip()

        # Filename match (exact or substring in either direction)
        filename_matches = (
            chunk_source == fn_lower
            or fn_lower in chunk_source
            or chunk_source in fn_lower
        )

        if not filename_matches:
            continue

        # No page required — filename match is sufficient
        if page is None:
            return True

        # Page match — compare as strings to handle int/str variance
        if str(chunk.get('page', '')) == str(page):
            return True

    return False


def flag_unsupported_claims(response_text: str, chunks: List[Dict]) -> List[str]:
    """
    Use heuristics to identify specific factual claims (dates, dollar amounts,
    named actions) that appear in the response but cannot be traced to any
    retrieved chunk by substring search.

    Returns a deduplicated list of claim strings that appear unsupported.
    Note: substring absence is a signal, not a guarantee — use as a prompt
    for attorney review, not as a definitive hallucination finding.
    """
    # Build a single lowercase corpus from all chunk content for fast search
    chunk_corpus = ' '.join(
        chunk.get('content', '') or chunk.get('text', '') or ''
        for chunk in chunks
    ).lower()

    unsupported: List[str] = []
    seen: set = set()

    for pattern in _CLAIM_PATTERNS:
        for match in pattern.finditer(response_text):
            claim = match.group(0).strip()
            claim_lower = claim.lower()

            if claim_lower in seen:
                continue

            if claim_lower not in chunk_corpus:
                unsupported.append(claim)

            seen.add(claim_lower)

    return unsupported


def verify_citations(response_text: str, chunks: List[Dict]) -> CitationReport:
    """
    Main entry point. Check every source reference in the response against
    the actual retrieved chunks and flag unsupported factual claims.

    Args:
        response_text: The full LLM answer string.
        chunks: The raw retrieved chunks list from rag_query() / stream_rag_query().

    Returns:
        CitationReport with verified, unverified, missing, and a confidence score.
    """
    all_citations = parse_citations(response_text)
    verified: List[Dict] = []
    unverified: List[Dict] = []

    for citation in all_citations:
        if verify_single_citation(citation['filename'], citation['page'], chunks):
            verified.append(citation)
        else:
            unverified.append(citation)

    missing = flag_unsupported_claims(response_text, chunks)

    # Confidence score
    # - Start from 1.0
    # - Each unverified citation costs up to 0.6 total weight
    # - Each unsupported claim costs up to 0.4 total weight (capped at 10 claims)
    total_citations = len(all_citations)
    total_claims = max(len(missing), 0)

    if total_citations == 0 and total_claims == 0:
        # Nothing to verify and no violations — clean slate
        score = 1.0
    else:
        citation_penalty = (len(unverified) / max(total_citations, 1)) * 0.6
        claim_penalty = min(total_claims / 10.0, 1.0) * 0.4
        score = max(0.0, 1.0 - citation_penalty - claim_penalty)

    return CitationReport(
        verified_citations=verified,
        unverified_citations=unverified,
        missing_citations=missing,
        overall_confidence_score=round(score, 3),
    )
