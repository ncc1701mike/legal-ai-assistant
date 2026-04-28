#!/usr/bin/env python3
"""
Print an ASCII table of Amicus eval score history from results_history.jsonl.

Usage:
    python eval/show_history.py
"""

import json
import sys
from pathlib import Path

HISTORY_PATH = Path(__file__).parent / "results_history.jsonl"


def load_history() -> list:
    if not HISTORY_PATH.exists():
        return []
    records = []
    for line in HISTORY_PATH.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def type_breakdown(question_results: list) -> str:
    """One-line summary like  F:2/2 X:1/2 C:2/2 T:2/2 TR:1/2"""
    categories = {
        "FACTUAL": ("F", []),
        "CROSS-DOC": ("X", []),
        "CONTRADICTION": ("C", []),
        "TEMPORAL": ("T", []),
        "TRAP": ("TR", []),
    }
    # question_results doesn't carry type — just show pass/fail counts
    passed = sum(1 for r in question_results if r.get("passed"))
    total = len(question_results)
    return f"{passed}/{total}"


def print_table(records: list) -> None:
    if not records:
        print("No eval history found.")
        print(f"Run eval/eval_ci.py or eval/eval_batch1.py to generate records.")
        return

    COL = {"#": 4, "Date": 19, "Type": 5, "Score": 7, "P/T": 6, "Delta": 7, "Model": 22, "Chunks": 7}
    sep = "  "
    header = sep.join(k.ljust(v) for k, v in COL.items())
    rule = "-" * len(header)

    print(f"\n{'='*len(header)}")
    print("AMICUS EVAL HISTORY")
    print(f"{'='*len(header)}")
    print(header)
    print(rule)

    prev_rate = None
    for i, r in enumerate(records, 1):
        ts = r.get("timestamp", "")[:16].replace("T", " ")
        etype = r.get("eval_type", "?")[:5]
        rate = r.get("pass_rate", 0.0)
        passed = r.get("passed", 0)
        total = r.get("total", 0)
        model = r.get("model_used", "?")[:21]
        corpus = str(r.get("corpus_size", "?"))

        if prev_rate is not None:
            delta = rate - prev_rate
            delta_str = f"{delta:+.0%}"
        else:
            delta_str = "—"
        prev_rate = rate

        row = sep.join([
            str(i).ljust(COL["#"]),
            ts.ljust(COL["Date"]),
            etype.ljust(COL["Type"]),
            f"{rate:.0%}".ljust(COL["Score"]),
            f"{passed}/{total}".ljust(COL["P/T"]),
            delta_str.ljust(COL["Delta"]),
            model.ljust(COL["Model"]),
            corpus.ljust(COL["Chunks"]),
        ])
        print(row)

    print(rule)

    latest = records[-1]
    best = max(records, key=lambda r: r.get("pass_rate", 0))
    print(f"  Latest : {latest.get('pass_rate', 0):.0%}  ({latest.get('eval_type', '?')} — {latest.get('timestamp', '')[:10]})")
    print(f"  Best   : {best.get('pass_rate', 0):.0%}  ({best.get('eval_type', '?')} — {best.get('timestamp', '')[:10]})")
    print(f"  Runs   : {len(records)} total  ({sum(1 for r in records if r.get('eval_type')=='ci')} CI,  {sum(1 for r in records if r.get('eval_type')=='full')} full)")
    print(f"{'='*len(header)}\n")


if __name__ == "__main__":
    records = load_history()
    # Optional: filter by type with --ci or --full flag
    if "--ci" in sys.argv:
        records = [r for r in records if r.get("eval_type") == "ci"]
    elif "--full" in sys.argv:
        records = [r for r in records if r.get("eval_type") == "full"]
    print_table(records)
