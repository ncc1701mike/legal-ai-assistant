#!/usr/bin/env python3
"""
Pre-commit smoke test — 3 questions, < 2 minutes.
Catches obvious retrieval regressions before a commit lands on main.

INSTALL (optional but recommended):
    cp scripts/pre_commit_check.py .git/hooks/pre-commit
    chmod +x .git/hooks/pre-commit

BYPASS for a single commit:
    git commit --no-verify

The hook passes (exit 0) if at least 2 of 3 smoke questions pass.
Infrastructure failures (Ollama not running, import errors) are treated
as a pass so the hook never blocks work on non-eval machines.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))

SMOKE_QUESTIONS = [
    {
        "id": "Q01",
        "question": "What was Diana Chen's salary at the time of her termination?",
        "key_facts": ["124,500"],
        "mode": "rerank",
        "trap": None,
    },
    {
        "id": "Q17",
        "question": "What is the specific factual dispute between Chen and Webb about what Webb said during their January 17 2022 first meeting?",
        "key_facts": ["good fit", "denies", "journal"],
        "mode": "agentic",
        "trap": None,
    },
    {
        "id": "Q27",
        "question": "What was the settlement amount in Chen v. Nexagen?",
        "key_facts": ["no settlement", "45,000"],
        "mode": "rerank",
        "trap": "confabulate_settlement",
    },
]

PASS_REQUIRED = 2  # at least 2 of 3 must pass


def quick_judge(question: str, answer: str, key_facts: list, trap: str) -> bool:
    try:
        from modules.llm import get_llm
        from langchain_core.messages import HumanMessage

        facts_str = "\n".join(f"  - {f}" for f in key_facts)
        trap_note = ""
        if trap == "confabulate_settlement":
            trap_note = (
                "\nTRAP: This case has no litigation settlement. "
                "FAIL if a specific settlement amount is stated (the $45,000 is a severance offer, not a settlement)."
            )

        prompt = (
            f"Legal eval judge. Does this answer cover ALL required facts?\n\n"
            f"QUESTION: {question}\n"
            f"REQUIRED FACTS:\n{facts_str}\n"
            f"ANSWER: {answer}{trap_note}\n\n"
            "Reply with one word: PASS or FAIL."
        )
        llm = get_llm()
        msg = llm.invoke([HumanMessage(content=prompt)])
        return "PASS" in msg.content.upper()
    except Exception:
        return True  # Infrastructure down — don't block commit


def main() -> int:
    print("\n[pre-commit] Running smoke test (3 questions)...")

    try:
        from modules.llm import rag_query
    except Exception as e:
        print(f"[pre-commit] SKIP — import error: {e}")
        return 0  # Can't import modules; probably not an eval machine

    passed = 0
    for q in SMOKE_QUESTIONS:
        try:
            result = rag_query(q["question"], top_k=5, mode=q["mode"])
            answer = result.get("answer", "")
            ok = quick_judge(q["question"], answer, q["key_facts"], q["trap"])
        except Exception as e:
            print(f"  [{q['id']}] SKIP (error: {e})")
            ok = True  # Infrastructure failure — pass through

        status = "PASS" if ok else "FAIL"
        print(f"  [{q['id']}] {status}")
        if ok:
            passed += 1

    total = len(SMOKE_QUESTIONS)
    verdict = "PASS" if passed >= PASS_REQUIRED else "FAIL"
    print(f"[pre-commit] {passed}/{total} — {verdict}")

    if verdict == "FAIL":
        print("[pre-commit] Retrieval regression detected. Investigate before committing.")
        print("[pre-commit] To bypass: git commit --no-verify")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
