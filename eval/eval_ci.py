#!/usr/bin/env python3
"""
CI Eval — 10-question diagnostic subset of the Chen v. Nexagen corpus.
Designed to run on every commit. Target: ~8 minutes on llama3.1:8b.

Question selection (10 of 30):
  FACTUAL      Q01, Q08  — salary and diagnosis date (basic fact retrieval)
  CROSS-DOC    Q10, Q14  — disputed deliverable + WFH comparison (multi-doc)
  CONTRADICTION Q17, Q20 — Webb's first-meeting claim + Calloway phrase (hard)
  TEMPORAL     Q24, Q25  — back-pay period + HR-to-PIP gap (date reasoning)
  TRAP         Q27, Q29  — settlement confabulation + Title VII outcome (hallucination)

Exit 0: pass rate >= 70%
Exit 1: pass rate <  70%
"""

import sys
import time
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
_EVAL_DIR = Path(__file__).parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_EVAL_DIR))

from modules.llm import rag_query, get_llm, get_primary_model
from langchain_core.messages import HumanMessage

PASS_THRESHOLD = 0.70

CI_QUESTIONS = [
    # FACTUAL — 2 questions
    {
        "id": "Q01", "type": "FACTUAL", "difficulty": "EASY",
        "question": "What was Diana Chen's salary at the time of her termination?",
        "key_facts": ["124,500", "highest", "grade band"],
        "trap": None,
    },
    {
        "id": "Q08", "type": "FACTUAL", "difficulty": "EASY",
        "question": "What medical diagnosis did Diana Chen receive in early 2021, and what was the specific date?",
        "key_facts": ["lupus", "february", "2021"],
        "trap": None,
    },
    # CROSS-DOC — 2 questions
    {
        "id": "Q10", "type": "CROSS-DOC", "difficulty": "HARD",
        "question": "What is the dispute about whether Chen submitted the August 3 deliverable on time?",
        "key_facts": ["August 3", "4:47", "email", "IT logs", "not received"],
        "trap": None,
    },
    {
        "id": "Q14", "type": "CROSS-DOC", "difficulty": "MEDIUM",
        "question": "How many other employees had work from home approved, and how does this compare to Chen's denial?",
        "key_facts": ["fourteen", "comparable", "operational necessity"],
        "trap": None,
    },
    # CONTRADICTION — 2 questions (routed to agentic)
    {
        "id": "Q17", "type": "CONTRADICTION", "difficulty": "HARD",
        "question": "What is the specific factual dispute between Chen and Webb about what Webb said during their January 17 2022 first meeting?",
        "key_facts": ["good fit", "Chen", "Webb", "denies", "journal"],
        "trap": None,
    },
    {
        "id": "Q20", "type": "CONTRADICTION", "difficulty": "HARD",
        "question": "What does Calloway say his December 15 email phrase timing is clean meant, and what alternative interpretation is presented?",
        "key_facts": ["timing is clean", "process", "retaliatory", "Calloway"],
        "trap": None,
    },
    # TEMPORAL — 2 questions
    {
        "id": "Q24", "type": "TEMPORAL", "difficulty": "MEDIUM",
        "question": "From Chen's termination to the trial date, how long was the period, and what was the total back pay estimated?",
        "key_facts": ["December 19, 2022", "January 13, 2025", "25.5", "310,749"],
        "trap": None,
    },
    {
        "id": "Q25", "type": "TEMPORAL", "difficulty": "MEDIUM",
        "question": "How many days elapsed between Chen's internal HR complaint and the PIP, and why is that significant legally?",
        "key_facts": ["July 15", "September 12", "58 days", "retaliation"],
        "trap": None,
    },
    # TRAP — 2 questions (hallucination guards)
    {
        "id": "Q27", "type": "TRAP", "difficulty": "EASY",
        "question": "What was the settlement amount in Chen v. Nexagen?",
        "key_facts": ["no settlement", "trial", "45,000"],
        "trap": "confabulate_settlement",
    },
    {
        "id": "Q29", "type": "TRAP", "difficulty": "MEDIUM",
        "question": "Was Chen's Title VII racial discrimination claim successful at trial?",
        "key_facts": ["dismissed", "summary judgment", "May", "2024", "ADA"],
        "trap": "confabulate_trial_outcome",
    },
]


def get_mode(q: dict) -> str:
    if q["difficulty"] == "HARD" or q["type"] == "CONTRADICTION":
        return "agentic"
    return "rerank"


def score_response(question: str, response: str, key_facts: list, trap: str) -> tuple[bool, str]:
    """LLM judge — identical prompt to eval_batch1.py for consistency."""
    facts_formatted = "\n".join(f"  - {f}" for f in key_facts)

    trap_instruction = ""
    if trap == "confabulate_settlement":
        trap_instruction = (
            "\nTRAP CHECK: This case has no settlement. If the answer states or implies a "
            "specific settlement amount (other than the $45,000 severance offer at termination), "
            "return FAIL regardless of other facts covered."
        )
    elif trap == "confabulate_trial_outcome":
        trap_instruction = (
            "\nTRAP CHECK: Chen's Title VII racial discrimination claim was dismissed at "
            "summary judgment — it was not successful at trial. If the answer states or implies "
            "the Title VII claim succeeded at trial, return FAIL regardless of other facts covered."
        )

    eval_prompt = f"""You are a strict legal eval judge. Determine whether the system answer substantively covers each required key fact.

QUESTION:
{question}

REQUIRED KEY FACTS (substance, not exact wording):
{facts_formatted}

SYSTEM ANSWER:
{response}
{trap_instruction}
INSTRUCTIONS:
- Evaluate SUBSTANCE, not exact keyword presence. Paraphrasing, synonyms, and equivalent numeric representations all count.
- A fact like "124,500" is covered if the answer mentions $124,500 or "approximately $124k" in context.
- A fact like "declined" is covered if the answer says "rejected", "refused", "did not accept", etc.
- A fact like "five" is covered if the answer says "5" or "five consecutive".
- DATE CONTAINMENT: A key fact like "May" is fully covered if the answer contains a full date that includes that month, e.g. "May 15, 2024" satisfies the key fact "May". Do not require the bare month word to appear separately.
- NO-SETTLEMENT TRAP: The key fact "no settlement" is satisfied if the answer correctly states that the case went to trial AND correctly identifies the $45,000 figure as a severance/separation offer — not a lawsuit settlement.
- Return PASS only if ALL key facts are substantively covered AND no trap condition is triggered.
- Return FAIL if ANY key fact is missing from the substance of the answer.

Respond in EXACTLY this format (two lines only):
VERDICT: PASS
REASON: one sentence"""

    try:
        llm = get_llm()
        msg = llm.invoke([HumanMessage(content=eval_prompt)])
        text = msg.content.strip()
        passed = "VERDICT: PASS" in text
        reason = next(
            (ln.replace("REASON:", "").strip() for ln in text.splitlines() if ln.startswith("REASON:")),
            "no reason provided",
        )
        return passed, reason
    except Exception as e:
        return False, f"evaluator error: {e}"


def _count_corpus() -> int:
    try:
        from modules.ingestion import chroma_client, COLLECTION_NAME
        col = chroma_client.get_or_create_collection(COLLECTION_NAME)
        return col.count()
    except Exception:
        return -1


def run_ci_eval() -> int:
    """Run the 10-question CI eval. Returns exit code 0 (PASS) or 1 (FAIL)."""
    print(f"{'='*70}")
    print(f"AMICUS CI EVAL — {len(CI_QUESTIONS)}-question diagnostic subset")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}   Threshold: {PASS_THRESHOLD:.0%}")
    print(f"{'='*70}")

    results = []
    passed = 0

    for q in CI_QUESTIONS:
        mode = get_mode(q)
        t0 = time.time()
        try:
            result = rag_query(q["question"], top_k=7, mode=mode)
            answer = result.get("answer", "")
            elapsed = time.time() - t0

            ok, reason = score_response(q["question"], answer, q["key_facts"], q["trap"])
            status = "PASS" if ok else "FAIL"
            tag = f"[{q['id']}] {q['type']:<13} {q['difficulty']:<7} {mode:<7}"
            print(f"  {status}  {tag}  {elapsed:.1f}s  {reason[:58]}")
            if ok:
                passed += 1
            results.append({"id": q["id"], "passed": ok, "mode": mode, "elapsed": round(elapsed, 2)})

        except Exception as e:
            elapsed = time.time() - t0
            print(f"  FAIL  [{q['id']}] ERROR: {e}")
            results.append({"id": q["id"], "passed": False, "mode": mode, "elapsed": round(elapsed, 2)})

    total = len(CI_QUESTIONS)
    pass_rate = passed / total
    verdict = "PASS" if pass_rate >= PASS_THRESHOLD else "FAIL"

    print(f"{'='*70}")
    print(f"  Score:   {passed}/{total} ({pass_rate:.0%})   Threshold: {PASS_THRESHOLD:.0%}")
    print(f"  Verdict: {verdict}")
    print(f"{'='*70}")

    # Append to regression history (best-effort, never blocks CI)
    try:
        from eval_regression import append_result
        append_result(
            eval_type="ci",
            passed=passed,
            total=total,
            model_used=get_primary_model(),
            corpus_size=_count_corpus(),
            question_results=results,
        )
    except Exception:
        pass

    return 0 if pass_rate >= PASS_THRESHOLD else 1


if __name__ == "__main__":
    sys.exit(run_ci_eval())
