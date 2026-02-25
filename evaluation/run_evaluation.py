# evaluation/run_evaluation.py
# Runs RAGAS evaluation on both vector and hybrid retrieval pipelines
# Reads testset.json, queries both pipelines, scores with RAGAS
# Saves results to results_vector.json and results_hybrid.json

import os
import json
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from datasets import Dataset

from modules.retrieval import retrieve_and_format
from modules.llm import rag_query

# ── Configuration ─────────────────────────────────────────────────────────────
TESTSET_PATH = Path(__file__).parent / "testset.json"
RESULTS_DIR = Path(__file__).parent
MODES = ["vector", "hybrid"]

# ── Load Testset ──────────────────────────────────────────────────────────────
def load_testset() -> list[dict]:
    with open(TESTSET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"]


# ── Run Pipeline ──────────────────────────────────────────────────────────────
def run_pipeline(question: str, mode: str) -> dict:
    """Run a question through the RAG pipeline and return answer + contexts."""
    try:
        result = rag_query(question, top_k=5, mode=mode)
        answer = result.get("answer", "")
        chunks = result.get("chunks", [])
        contexts = [c["text"] for c in chunks]
        return {
            "answer": answer,
            "contexts": contexts if contexts else ["No context retrieved"]
        }
    except Exception as e:
        print(f"    ⚠️  Pipeline error: {e}")
        return {
            "answer": "Error generating answer",
            "contexts": ["No context retrieved"]
        }


# ── Run RAGAS Evaluation ──────────────────────────────────────────────────────
def run_ragas_evaluation(questions: list[dict], mode: str) -> dict:
    """
    Run all questions through the pipeline and evaluate with RAGAS.
    Returns dict with scores and per-question results.
    """
    print(f"\n  Running {len(questions)} questions through {mode} pipeline...")

    ragas_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }

    per_question_results = []

    for i, qa in enumerate(questions, 1):
        question = qa["question"]
        ground_truth = qa["ground_truth"]
        source_doc = qa.get("source_doc", "unknown")

        print(f"  [{i:02d}/{len(questions)}] {question[:60]}...")

        result = run_pipeline(question, mode=mode)

        ragas_data["question"].append(question)
        ragas_data["answer"].append(result["answer"])
        ragas_data["contexts"].append(result["contexts"])
        ragas_data["ground_truth"].append(ground_truth)

        per_question_results.append({
            "question": question,
            "answer": result["answer"],
            "ground_truth": ground_truth,
            "source_doc": source_doc,
            "contexts_retrieved": len(result["contexts"]),
            "question_type": qa.get("question_type", "unknown")
        })

    # Build RAGAS dataset
    dataset = Dataset.from_dict(ragas_data)

    # Configure RAGAS to use GPT-4o
    from openai import OpenAI
    from ragas.llms import llm_factory
    from langchain_openai import OpenAIEmbeddings as LangchainOpenAIEmbeddings

    openai_client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
    llm = llm_factory("gpt-4o", client=openai_client)
    
    # Use LangChain embeddings directly — RAGAS AnswerRelevancy calls embed_query internally
    lc_embeddings = LangchainOpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.environ['OPENAI_API_KEY']
    )

    metrics = [Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()]
    for metric in metrics:
        metric.llm = llm
        if hasattr(metric, 'embeddings'):
            metric.embeddings = lc_embeddings
        if hasattr(metric, 'embedding'):
            metric.embedding = lc_embeddings
    

    print(f"\n  Scoring with RAGAS (faithfulness, answer_relevancy, "
          f"context_precision, context_recall)...")

    scores = evaluate(dataset, metrics=metrics)
    df = scores.to_pandas()
    numeric_cols = df.select_dtypes(include='number').columns
    scores_dict = df[numeric_cols].mean().to_dict()

    return {
        "mode": mode,
        "scores": {k: round(float(v), 4) for k, v in scores_dict.items()},
        "per_question": per_question_results
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n🔬 RAGAS Evaluation — Legal AI Assistant")
    print("=" * 60)

    questions = load_testset()
    print(f"Loaded {len(questions)} questions from testset.json")
    print(f"Evaluating modes: {MODES}")
    print("=" * 60)

    all_results = {}

    for mode in MODES:
        print(f"\n{'─' * 60}")
        print(f"📊 Evaluating: {mode.upper()} retrieval")
        print(f"{'─' * 60}")

        results = run_ragas_evaluation(questions, mode=mode)
        all_results[mode] = results

        # Save individual results
        output_path = RESULTS_DIR / f"results_{mode}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n  ✅ Saved to {output_path}")

        # Print scores
        print(f"\n  Scores for {mode.upper()}:")
        for metric, score in results["scores"].items():
            if score != score:  # NaN check
                print(f"    {metric:<25}    NaN  (scoring error)")
            else:
                bar = "█" * int(score * 20)
                print(f"    {metric:<25} {score:.4f}  {bar}")

    # Print comparison table
    print(f"\n{'=' * 60}")
    print("📈 COMPARISON TABLE")
    print(f"{'=' * 60}")
    print(f"{'Metric':<25} {'Vector':>10} {'Hybrid':>10} {'Delta':>10}")
    print(f"{'─' * 55}")

    vector_scores = all_results.get("vector", {}).get("scores", {})
    hybrid_scores = all_results.get("hybrid", {}).get("scores", {})

    for metric in vector_scores:
        v = vector_scores.get(metric, 0)
        h = hybrid_scores.get(metric, 0)
        delta = h - v
        delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        print(f"  {metric:<23} {v:>10.4f} {h:>10.4f} {delta_str:>10}")

    # Save combined results
    combined_path = RESULTS_DIR / "results_combined.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Combined results saved to: {combined_path}")
    print(f"Next step: Open evaluation/evaluation.ipynb")


if __name__ == "__main__":
    main()