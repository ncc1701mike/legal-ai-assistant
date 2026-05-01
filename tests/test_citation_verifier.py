"""
tests/test_citation_verifier.py
Unit tests for modules/citation_verifier.py
"""

import pytest
from modules.citation_verifier import (
    parse_citations,
    verify_single_citation,
    flag_unsupported_claims,
    verify_citations,
    CitationReport,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _chunk(source: str, page: int, content: str = "") -> dict:
    return {"source": source, "page": page, "content": content}


# ── parse_citations ───────────────────────────────────────────────────────────

class TestParseCitations:
    def test_single_page_citation(self):
        text = "She denied it. (Source: depo_chen.txt, Page 12)"
        result = parse_citations(text)
        assert len(result) == 1
        assert result[0]["filename"] == "depo_chen.txt"
        assert result[0]["page"] == 12
        assert "(Source: depo_chen.txt, Page 12)" in result[0]["raw"]

    def test_page_range_citation(self):
        text = "Relevant passage (Source: exhibit_a.txt, Page 5-7)"
        result = parse_citations(text)
        assert len(result) == 1
        assert result[0]["page"] == 5  # takes start of range

    def test_vs_citation(self):
        text = "Accounts differ (Source: file_a.txt vs file_b.txt)"
        result = parse_citations(text)
        assert len(result) == 2
        filenames = {r["filename"] for r in result}
        assert "file_a.txt" in filenames
        assert "file_b.txt" in filenames
        for r in result:
            assert r["page"] is None

    def test_multiple_citations(self):
        text = (
            "First fact (Source: doc1.txt, Page 3). "
            "Second fact (Source: doc2.txt, Page 7). "
            "Conflict (Source: doc1.txt vs doc3.txt)"
        )
        result = parse_citations(text)
        # 2 page-citations + 2 from vs-citation
        assert len(result) == 4

    def test_case_insensitive(self):
        text = "(source: myfile.txt, page 9)"
        result = parse_citations(text)
        assert len(result) == 1
        assert result[0]["page"] == 9

    def test_no_citations(self):
        text = "No citations in this text at all."
        assert parse_citations(text) == []

    def test_filename_with_spaces_trimmed(self):
        text = "(Source:  spaced_file.txt , Page 1)"
        result = parse_citations(text)
        assert result[0]["filename"] == "spaced_file.txt"


# ── verify_single_citation ─────────────────────────────────────────────────────

class TestVerifySingleCitation:
    def test_exact_match(self):
        chunks = [_chunk("depo_webb.txt", 4)]
        assert verify_single_citation("depo_webb.txt", 4, chunks) is True

    def test_wrong_page(self):
        chunks = [_chunk("depo_webb.txt", 4)]
        assert verify_single_citation("depo_webb.txt", 5, chunks) is False

    def test_substring_filename_match(self):
        # chunk has full path; citation has just filename
        chunks = [_chunk("cases/2024/depo_webb.txt", 4)]
        assert verify_single_citation("depo_webb.txt", 4, chunks) is True

    def test_chunk_filename_substring_of_citation(self):
        # The matcher is bidirectional: chunk_source in fn OR fn in chunk_source.
        # "webb.txt" is a substring of "depo_webb.txt", so this DOES match.
        chunks = [_chunk("webb.txt", 4)]
        assert verify_single_citation("depo_webb.txt", 4, chunks) is True
        assert verify_single_citation("webb.txt", 4, chunks) is True

    def test_no_page_required_filename_match(self):
        chunks = [_chunk("exhibit_a.txt", 2)]
        assert verify_single_citation("exhibit_a.txt", None, chunks) is True

    def test_no_page_required_filename_mismatch(self):
        chunks = [_chunk("exhibit_b.txt", 2)]
        assert verify_single_citation("exhibit_a.txt", None, chunks) is False

    def test_empty_chunks(self):
        assert verify_single_citation("any.txt", 1, []) is False

    def test_case_insensitive_filename(self):
        chunks = [_chunk("Depo_Webb.TXT", 1)]
        assert verify_single_citation("depo_webb.txt", 1, chunks) is True

    def test_page_as_string_in_chunk(self):
        # chunk stores page as string (e.g. from metadata)
        chunks = [{"source": "doc.txt", "page": "3", "content": ""}]
        assert verify_single_citation("doc.txt", 3, chunks) is True


# ── flag_unsupported_claims ───────────────────────────────────────────────────

class TestFlagUnsupportedClaims:
    def test_date_found_in_chunks(self):
        text = "The meeting occurred on March 15, 2023."
        chunks = [_chunk("doc.txt", 1, "The meeting occurred on March 15, 2023.")]
        result = flag_unsupported_claims(text, chunks)
        assert "March 15, 2023" not in result

    def test_date_not_in_chunks(self):
        text = "Terminated on January 5, 2022."
        chunks = [_chunk("doc.txt", 1, "no dates here")]
        result = flag_unsupported_claims(text, chunks)
        assert any("January 5, 2022" in c for c in result)

    def test_dollar_amount_found(self):
        text = "Salary was $85,000 per year."
        chunks = [_chunk("doc.txt", 1, "salary was $85,000 per year")]
        result = flag_unsupported_claims(text, chunks)
        assert result == []

    def test_dollar_amount_not_found(self):
        text = "Damages claimed at $2.5 million."
        chunks = [_chunk("doc.txt", 1, "no amounts here")]
        result = flag_unsupported_claims(text, chunks)
        assert any("$2.5 million" in c.lower() or "2.5" in c for c in result)

    def test_named_action_found(self):
        text = "Smith testified that the policy was outdated."
        chunks = [_chunk("doc.txt", 1, "Smith testified that the policy was outdated")]
        result = flag_unsupported_claims(text, chunks)
        assert result == []

    def test_named_action_not_found(self):
        text = "Johnson signed the agreement."
        chunks = [_chunk("doc.txt", 1, "no relevant content here")]
        result = flag_unsupported_claims(text, chunks)
        assert any("Johnson" in c for c in result)

    def test_deduplication(self):
        text = "January 5, 2022 ... January 5, 2022 again."
        chunks = [_chunk("doc.txt", 1, "")]
        result = flag_unsupported_claims(text, chunks)
        count = sum(1 for c in result if "january 5, 2022" in c.lower())
        assert count == 1

    def test_empty_chunks(self):
        text = "Settlement was $500,000."
        result = flag_unsupported_claims(text, [])
        assert len(result) >= 1

    def test_no_claims_in_text(self):
        text = "The sky is blue. Contracts exist in many forms."
        chunks = [_chunk("doc.txt", 1, "")]
        result = flag_unsupported_claims(text, chunks)
        assert result == []


# ── verify_citations (integration) ───────────────────────────────────────────

class TestVerifyCitations:
    def test_all_verified(self):
        text = "Key fact. (Source: doc.txt, Page 2)"
        chunks = [_chunk("doc.txt", 2, "Key fact.")]
        report = verify_citations(text, chunks)
        assert len(report.verified_citations) == 1
        assert len(report.unverified_citations) == 0
        assert report.overall_confidence_score > 0.5

    def test_unverified_citation_penalises_score(self):
        text = "Some claim. (Source: ghost.txt, Page 99)"
        chunks = [_chunk("real.txt", 1, "other content")]
        report = verify_citations(text, chunks)
        assert len(report.unverified_citations) == 1
        assert report.overall_confidence_score < 0.9

    def test_no_citations_returns_clean(self):
        text = "No source references anywhere."
        chunks = [_chunk("doc.txt", 1, "No source references anywhere.")]
        report = verify_citations(text, chunks)
        assert report.overall_confidence_score == 1.0
        assert report.verified_citations == []
        assert report.unverified_citations == []

    def test_mixed_verified_unverified(self):
        text = (
            "Fact A. (Source: real.txt, Page 1) "
            "Fact B. (Source: fake.txt, Page 50)"
        )
        chunks = [_chunk("real.txt", 1, "Fact A.")]
        report = verify_citations(text, chunks)
        assert len(report.verified_citations) == 1
        assert len(report.unverified_citations) == 1
        # 1 unverified out of 2 total → citation_penalty = 0.5 * 0.6 = 0.3 → score ≤ 0.7
        assert report.overall_confidence_score <= 0.7

    def test_score_bounded_zero_to_one(self):
        # Many unverified + many unsupported claims should not go below 0.0
        text = " ".join(
            [f"(Source: ghost{i}.txt, Page {i})" for i in range(20)]
            + ["Terminated on January 5, 2022.", "Damages were $9 million."]
        )
        chunks = []
        report = verify_citations(text, chunks)
        assert 0.0 <= report.overall_confidence_score <= 1.0

    def test_score_rounded_to_three_decimals(self):
        text = "(Source: doc.txt, Page 1)"
        chunks = [_chunk("doc.txt", 1)]
        report = verify_citations(text, chunks)
        assert report.overall_confidence_score == round(report.overall_confidence_score, 3)

    def test_returns_citation_report_type(self):
        report = verify_citations("", [])
        assert isinstance(report, CitationReport)


# ── Confidence score stability ────────────────────────────────────────────────

class TestConfidenceScoreStability:
    """Score must be identical whether or not a preamble opener is prepended."""

    def test_score_unchanged_after_preamble_prepended(self):
        clean = (
            "Chen testified that Webb used racial slurs on March 3rd "
            "(Source: chen_deposition.txt, Page 12)."
        )
        # Preamble contains no (Source:…) citation and no _NAMED_ACTION_RE match
        preambled = (
            "Based on the provided context, I will answer the question. " + clean
        )
        chunks = [_chunk(
            "chen_deposition.txt", 12,
            "Chen testified that Webb used racial slurs on March 3rd",
        )]
        assert (
            verify_citations(clean, chunks).overall_confidence_score
            == verify_citations(preambled, chunks).overall_confidence_score
        )

    def test_zero_citation_zero_claim_score_is_1(self):
        text = "No source references anywhere."
        chunks = [_chunk("doc.txt", 1, text)]
        assert verify_citations(text, chunks).overall_confidence_score == 1.0

    def test_empty_response_score_is_1(self):
        assert verify_citations("", []).overall_confidence_score == 1.0
