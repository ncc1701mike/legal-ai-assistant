#!/usr/bin/env python3
"""
Regression history tracker for Amicus eval runs.

append_result() is called by eval_ci.py and eval_batch1.py after each run.
Records are appended to eval/results_history.jsonl — one JSON object per line.

Record schema:
  timestamp        ISO-8601 run time
  eval_type        "ci" (10-question subset) or "full" (all 30 questions)
  pass_rate        float 0.0-1.0
  passed           int
  total            int
  model_used       Ollama model ID at time of run
  corpus_size      ChromaDB chunk count (-1 if unavailable)
  question_results list of {id, passed, mode, elapsed}
"""

import json
from datetime import datetime
from pathlib import Path

HISTORY_PATH = Path(__file__).parent / "results_history.jsonl"


def append_result(
    eval_type: str,
    passed: int,
    total: int,
    model_used: str,
    corpus_size: int,
    question_results: list,
) -> None:
    """Append one run record to results_history.jsonl. Never raises."""
    try:
        pass_rate = round(passed / total, 4) if total else 0.0
        record = {
            "timestamp":        datetime.now().isoformat(),
            "eval_type":        eval_type,
            "pass_rate":        pass_rate,
            "passed":           passed,
            "total":            total,
            "model_used":       model_used,
            "corpus_size":      corpus_size,
            "question_results": question_results,
        }
        with HISTORY_PATH.open("a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass
