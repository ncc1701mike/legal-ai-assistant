#!/usr/bin/env python3
"""
Batch 1 Eval Runner — Chen v. Nexagen
Complexity-routed: HARD/CONTRADICTION → agentic, others → rerank
Baseline: 53% pass rate (16/30) from pre-agentic multihop mode.
"""

import sys
import time
import json
from datetime import datetime

sys.path.insert(0, "/Users/michaeldoran/AIE9/legal-ai-assistant")
from modules.llm import rag_query, get_llm
from langchain_core.messages import HumanMessage

QUESTIONS = [
    # FACTUAL — Easy
    {
        "id": "Q01", "type": "FACTUAL", "difficulty": "EASY",
        "question": "What was Diana Chen's salary at the time of her termination?",
        "key_facts": ["124,500", "highest", "grade band"],
        "trap": None
    },
    {
        "id": "Q02", "type": "FACTUAL", "difficulty": "EASY",
        "question": "What date was Chen placed on the Performance Improvement Plan?",
        "key_facts": ["September 12, 2022", "90", "December"],
        "trap": None
    },
    {
        "id": "Q03", "type": "FACTUAL", "difficulty": "EASY",
        "question": "Who was Diana Chen's treating rheumatologist?",
        "key_facts": ["Okonkwo", "Bay Area Rheumatology"],
        "trap": None
    },
    {
        "id": "Q04", "type": "FACTUAL", "difficulty": "EASY",
        "question": "What was the EEOC charge number assigned to Chen's complaint?",
        "key_facts": ["556-2023-00847"],
        "trap": None
    },
    {
        "id": "Q05", "type": "FACTUAL", "difficulty": "EASY",
        "question": "What severance package did Nexagen offer Chen at the time of her termination, and did she accept it?",
        "key_facts": ["45", "health", "declined"],
        "trap": None
    },
    {
        "id": "Q06", "type": "FACTUAL", "difficulty": "EASY",
        "question": "What were the three accommodations Chen requested in her February 2022 formal accommodation request?",
        "key_facts": ["9:30", "work from home", "fluorescent lighting"],
        "trap": None
    },
    {
        "id": "Q07", "type": "FACTUAL", "difficulty": "EASY",
        "question": "How many consecutive Exceeds Expectations performance reviews did Chen receive before Webb became her supervisor?",
        "key_facts": ["five", "exceeds expectations", "reviews"],
        "trap": None
    },
    {
        "id": "Q08", "type": "FACTUAL", "difficulty": "EASY",
        "question": "What medical diagnosis did Diana Chen receive in early 2021, and what was the specific date?",
        "key_facts": ["lupus", "february", "2021"],
        "trap": None
    },
    # CROSS-DOC — Medium
    {
        "id": "Q09", "type": "CROSS-DOC", "difficulty": "MEDIUM",
        "question": "What did Patricia Huang recommend in her July 22 memo, and did Nexagen follow her advice?",
        "key_facts": ["Huang", "July 22", "no policy violations", "interactive process"],
        "trap": None
    },
    {
        "id": "Q10", "type": "CROSS-DOC", "difficulty": "HARD",
        "question": "What is the dispute about whether Chen submitted the August 3 deliverable on time?",
        "key_facts": ["August 3", "4:47", "email", "IT logs", "not received"],
        "trap": None
    },
    {
        "id": "Q11", "type": "CROSS-DOC", "difficulty": "MEDIUM",
        "question": "What accommodations did Chen's predecessor supervisor Forsythe approve, and how does that compare to what Webb approved?",
        "key_facts": ["Forsythe", "flexible", "Webb", "denied"],
        "trap": None
    },
    {
        "id": "Q12", "type": "CROSS-DOC", "difficulty": "MEDIUM",
        "question": "What did Calloway tell Webb to do in their November 2022 email exchange, and what was the stated reason?",
        "key_facts": ["document", "every missed meeting", "FMLA", "termination"],
        "trap": None
    },
    {
        "id": "Q13", "type": "CROSS-DOC", "difficulty": "MEDIUM",
        "question": "What does the IT log evidence show about Chen's work hours during the PIP period?",
        "key_facts": ["47.3", "hours", "badge", "83"],
        "trap": None
    },
    {
        "id": "Q14", "type": "CROSS-DOC", "difficulty": "MEDIUM",
        "question": "How many other employees had work from home approved, and how does this compare to Chen's denial?",
        "key_facts": ["fourteen", "comparable", "operational necessity"],
        "trap": None
    },
    {
        "id": "Q15", "type": "CROSS-DOC", "difficulty": "MEDIUM",
        "question": "What did witness Torres observe about Chen's conduct at the August 24 team meeting?",
        "key_facts": ["Torres", "legitimate", "professional", "not aggressive"],
        "trap": None
    },
    {
        "id": "Q16", "type": "CROSS-DOC", "difficulty": "MEDIUM",
        "question": "What is the total back pay calculation and how was it derived?",
        "key_facts": ["310,749", "248,149", "62,600", "25.5"],
        "trap": None
    },
    # CONTRADICTION — Hard → agentic mode
    {
        "id": "Q17", "type": "CONTRADICTION", "difficulty": "HARD",
        "question": "What is the specific factual dispute between Chen and Webb about what Webb said during their January 17 2022 first meeting?",
        "key_facts": ["good fit", "Chen", "Webb", "denies", "journal"],
        "trap": None
    },
    {
        "id": "Q18", "type": "CONTRADICTION", "difficulty": "HARD",
        "question": "The PIP alleges Chen missed the June 15 deadline. What does each party say about this deadline, and what evidence supports each position?",
        "key_facts": ["June 15", "June 21", "June 22", "verbal extension", "dependency"],
        "trap": None
    },
    {
        "id": "Q19", "type": "CONTRADICTION", "difficulty": "HARD",
        "question": "How do Webb and Torres characterize the August 24 2022 team meeting differently, and who else was present?",
        "key_facts": ["Webb", "confrontational", "Torres", "professional", "Marsh"],
        "trap": None
    },
    {
        "id": "Q20", "type": "CONTRADICTION", "difficulty": "HARD",
        "question": "What does Calloway say his December 15 email phrase timing is clean meant, and what alternative interpretation is presented?",
        "key_facts": ["timing is clean", "process", "retaliatory", "Calloway"],
        "trap": None
    },
    {
        "id": "Q21", "type": "CONTRADICTION", "difficulty": "HARD",
        "question": "Did Chen's performance actually decline in 2022 as Webb claimed? What does the documentary evidence show on both sides?",
        "key_facts": ["IT logs", "Torres", "five", "Forsythe", "PIP"],
        "trap": None
    },
    {
        "id": "Q22", "type": "CONTRADICTION", "difficulty": "HARD",
        "question": "The PIP cites three specific missed deadlines. For each one, what is the factual dispute and what evidence exists?",
        "key_facts": ["June 15", "July 8", "August 3", "verbal", "IT logs"],
        "trap": None
    },
    # TEMPORAL — Medium
    {
        "id": "Q23", "type": "TEMPORAL", "difficulty": "MEDIUM",
        "question": "How long after Chen's lupus diagnosis did she submit her formal ADA accommodation request, and what happened in between?",
        "key_facts": ["February", "2021", "2022", "Forsythe", "Webb"],
        "trap": None
    },
    {
        "id": "Q24", "type": "TEMPORAL", "difficulty": "MEDIUM",
        "question": "From Chen's termination to the trial date, how long was the period, and what was the total back pay estimated?",
        "key_facts": ["December 19, 2022", "January 13, 2025", "25.5", "310,749"],
        "trap": None
    },
    {
        "id": "Q25", "type": "TEMPORAL", "difficulty": "MEDIUM",
        "question": "How many days elapsed between Chen's internal HR complaint and the PIP, and why is that significant legally?",
        "key_facts": ["July 15", "September 12", "58 days", "retaliation"],
        "trap": None
    },
    {
        "id": "Q26", "type": "TEMPORAL", "difficulty": "MEDIUM",
        "question": "At what point in the PIP did Chen take FMLA leave, and how quickly did Webb respond?",
        "key_facts": ["November", "six days", "warning", "FMLA"],
        "trap": None
    },
    # TRAP
    {
        "id": "Q27", "type": "TRAP", "difficulty": "EASY",
        "question": "What was the settlement amount in Chen v. Nexagen?",
        "key_facts": ["no settlement", "trial", "45,000"],
        "trap": "confabulate_settlement"
    },
    {
        "id": "Q28", "type": "TRAP", "difficulty": "EASY",
        "question": "Did the court grant Nexagen's Motion to Dismiss?",
        "key_facts": ["denied", "November", "2023", "discovery"],
        "trap": None
    },
    {
        "id": "Q29", "type": "TRAP", "difficulty": "MEDIUM",
        "question": "Was Chen's Title VII racial discrimination claim successful at trial?",
        "key_facts": ["dismissed", "summary judgment", "May", "2024", "ADA"],
        "trap": "confabulate_trial_outcome"
    },
    {
        "id": "Q30", "type": "TRAP", "difficulty": "EASY",
        "question": "What year was Chen hired at Nexagen, and what was her salary at that time?",
        "key_facts": ["2016", "87,000", "Senior Research Analyst"],
        "trap": None
    },
]


def get_mode(q: dict) -> str:
    """Route to agentic for HARD contradiction/cross-doc, rerank for everything else."""
    if q["difficulty"] == "HARD" or q["type"] == "CONTRADICTION":
        return "agentic"
    return "rerank"


def score_response(question: str, response: str, key_facts: list, trap: str) -> tuple:
    """LLM-based evaluator — checks substantive coverage of key facts, not exact keywords."""
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
- Check whether the SUBSTANCE of each key fact is present in the answer. Paraphrasing, synonyms, and equivalent numeric representations all count.
- A fact like "124,500" is covered if the answer mentions $124,500 or "approximately $124k" in context.
- A fact like "declined" is covered if the answer says "rejected", "refused", "did not accept", etc.
- A fact like "five" is covered if the answer says "5" or "five consecutive".
- Return PASS only if ALL key facts are substantively covered AND no trap condition is triggered.
- Return FAIL if ANY key fact is missing from the substance of the answer.

Respond in EXACTLY this format (two lines only):
VERDICT: PASS
REASON: one sentence"""

    try:
        llm = get_llm()
        msg = llm.invoke([HumanMessage(content=eval_prompt)])
        verdict_text = msg.content.strip()

        passed = "VERDICT: PASS" in verdict_text
        reason_line = next(
            (ln for ln in verdict_text.splitlines() if ln.startswith("REASON:")),
            "REASON: no reason provided"
        )
        reason = reason_line.replace("REASON:", "").strip()

        score = 1.0 if passed else 0.0
        hits = ["LLM: all key facts covered"] if passed else []
        misses = [] if passed else [reason]
        return passed, score, hits, misses

    except Exception as e:
        return False, 0.0, [], [f"evaluator error: {e}"]


def run_eval():
    print(f"\n{'='*70}")
    print(f"BATCH 1 EVAL — COMPLEXITY-ROUTED (agentic/rerank)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    results = []
    passed = 0
    failed = 0
    total_time = 0

    for q in QUESTIONS:
        mode = get_mode(q)
        print(f"[{q['id']}] {q['type']} / {q['difficulty']} → {mode}")
        print(f"  Q: {q['question'][:75]}...")

        start = time.time()
        try:
            result = rag_query(q["question"], top_k=7, mode=mode)
            answer = result.get("answer", "")
            elapsed = time.time() - start
            total_time += elapsed

            ok, score, hits, misses = score_response(q["question"], answer, q["key_facts"], q["trap"])

            status = "✅ PASS" if ok else "❌ FAIL"
            print(f"  {status} | Score: {score:.0%} | Time: {elapsed:.1f}s")
            if misses:
                print(f"  Missing: {misses}")

            results.append({
                "id": q["id"], "type": q["type"], "difficulty": q["difficulty"],
                "mode": mode, "passed": ok, "score": score,
                "hits": hits, "misses": misses, "elapsed": elapsed,
                "answer_preview": answer[:300]
            })
            if ok:
                passed += 1
            else:
                failed += 1

        except Exception as e:
            elapsed = time.time() - start
            print(f"  ❌ ERROR: {e}")
            results.append({
                "id": q["id"], "type": q["type"], "difficulty": q["difficulty"],
                "mode": mode, "passed": False, "score": 0,
                "hits": [], "misses": q["key_facts"],
                "elapsed": elapsed, "answer_preview": f"ERROR: {e}"
            })
            failed += 1
        print()

    # Summary
    total = passed + failed
    pass_rate = passed / total if total else 0
    delta = passed - 16

    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  Pass rate:  {passed}/{total} ({pass_rate:.0%})")
    print(f"  Baseline:   16/30 (53%)")
    print(f"  Delta:      {'+' if delta >= 0 else ''}{delta} ({pass_rate - 0.53:+.0%})")
    print(f"  Total time: {total_time:.0f}s ({total_time/total:.1f}s avg)")

    print(f"\n  By type:")
    for qtype in ["FACTUAL", "CROSS-DOC", "CONTRADICTION", "TEMPORAL", "TRAP"]:
        tr = [r for r in results if r["type"] == qtype]
        tp = sum(1 for r in tr if r["passed"])
        print(f"    {qtype:15s}: {tp}/{len(tr)}")

    print(f"\n  By difficulty:")
    for diff in ["EASY", "MEDIUM", "HARD"]:
        dr = [r for r in results if r["difficulty"] == diff]
        dp = sum(1 for r in dr if r["passed"])
        print(f"    {diff:8s}: {dp}/{len(dr)}")

    failed_qs = [r for r in results if not r["passed"]]
    if failed_qs:
        print(f"\n  Failed questions:")
        for r in failed_qs:
            print(f"    ❌ {r['id']} ({r['type']}/{r['difficulty']}/{r['mode']}) missing: {r['misses'][:3]}")

    out_path = "/tmp/batch1_eval_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "mode": "complexity_routed",
            "passed": passed, "total": total, "pass_rate": pass_rate,
            "baseline": 0.53, "results": results
        }, f, indent=2)
    print(f"\n  Results saved: {out_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    run_eval()