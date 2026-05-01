"""
Tests for modules/llm.strip_preamble()

No network or model calls — purely string transformation.
"""

import pytest
from modules.llm import strip_preamble


ANSWER = "Chen testified that Webb used racial slurs on March 3rd (chen_deposition.txt, Page 12)."


class TestStripPreamble:
    # ── Patterns that SHOULD be stripped ─────────────────────────────────────

    def test_provided_document_excerpts_strict_attribution(self):
        text = (
            "Based on the provided document excerpts, I will answer the question about "
            "the termination with strict source attribution. " + ANSWER
        )
        assert strip_preamble(text) == ANSWER

    def test_provided_document_excerpts_as_follows(self):
        text = (
            "Based on the provided document excerpts, I can answer the question as follows: "
            + ANSWER
        )
        assert strip_preamble(text) == ANSWER

    def test_provided_document_excerpts_following_answer(self):
        text = (
            "Based on the provided document excerpts, the following answer can be given: "
            + ANSWER
        )
        assert strip_preamble(text) == ANSWER

    def test_provided_document_excerpts_critical_rules(self):
        text = (
            "Based on the provided document excerpts, I will answer the question about "
            "race discrimination while following the critical rules for using document context. "
            + ANSWER
        )
        assert strip_preamble(text) == ANSWER

    def test_provided_context_short_form(self):
        text = "Based on the provided context, I will now answer the question. " + ANSWER
        assert strip_preamble(text) == ANSWER

    def test_i_will_answer_this_question_strict_attribution(self):
        text = (
            "I will answer this question about the HR complaint with strict source attribution. "
            + ANSWER
        )
        assert strip_preamble(text) == ANSWER

    def test_i_will_answer_the_question(self):
        text = "I will answer the question about damages as follows: " + ANSWER
        assert strip_preamble(text) == ANSWER

    def test_based_on_documents_period(self):
        text = "Based on the documents provided. " + ANSWER
        assert strip_preamble(text) == ANSWER

    def test_based_on_context_colon(self):
        text = "Based on context: " + ANSWER
        assert strip_preamble(text) == ANSWER

    def test_based_on_with_long_clause(self):
        text = (
            "Based on the provided document excerpts, which include depositions, emails, "
            "and HR records, I will now provide a comprehensive answer. " + ANSWER
        )
        assert strip_preamble(text) == ANSWER

    def test_case_insensitive_based_on(self):
        text = "based on the provided documents, here is the answer: " + ANSWER
        assert strip_preamble(text) == ANSWER

    def test_strips_trailing_whitespace_after_preamble(self):
        text = "Based on the provided document excerpts, I will answer.   " + ANSWER
        result = strip_preamble(text)
        assert result == ANSWER

    def test_preamble_ends_with_colon(self):
        text = "Based on the provided context, the answer is as follows: " + ANSWER
        assert strip_preamble(text) == ANSWER

    # ── Text that should NOT be stripped ─────────────────────────────────────

    def test_no_preamble_returned_unchanged(self):
        assert strip_preamble(ANSWER) == ANSWER

    def test_empty_string_unchanged(self):
        assert strip_preamble("") == ""

    def test_based_on_in_middle_not_stripped(self):
        text = ANSWER + " Based on the provided context, this is additional analysis."
        assert strip_preamble(text) == text

    def test_based_on_in_second_paragraph_not_stripped(self):
        text = ANSWER + "\n\nBased on the provided documents, here is more context."
        assert strip_preamble(text) == text

    def test_legitimate_answer_starting_with_the_is_unchanged(self):
        text = "The termination occurred on June 1st (hr_record.txt, Page 3)."
        assert strip_preamble(text) == text

    def test_legitimate_answer_starting_with_witness_name_unchanged(self):
        text = "Webb testified that he did not witness the incident (webb_depo.txt, Page 7)."
        assert strip_preamble(text) == text

    def test_based_on_evidence_in_body_not_stripped(self):
        # "Based on" appearing mid-sentence after real content is NOT a preamble
        text = "Webb was terminated. Based on the deposition evidence, the termination was retaliatory."
        assert strip_preamble(text) == text

    # ── Idempotency and edge cases ────────────────────────────────────────────

    def test_idempotent_on_clean_text(self):
        assert strip_preamble(ANSWER) == strip_preamble(strip_preamble(ANSWER))

    def test_idempotent_on_preamble_text(self):
        text = "Based on the provided context, the following answer can be given: " + ANSWER
        once = strip_preamble(text)
        twice = strip_preamble(once)
        assert once == twice

    def test_only_first_preamble_removed(self):
        # Two "Based on" openers back to back — only the first should go
        text = (
            "Based on the provided context, here is the answer: "
            "Based on the provided context, here is the answer: "
            + ANSWER
        )
        result = strip_preamble(text)
        # First preamble stripped; second remains (it's not at position 0 after strip)
        assert result.startswith("Based on the provided context")
