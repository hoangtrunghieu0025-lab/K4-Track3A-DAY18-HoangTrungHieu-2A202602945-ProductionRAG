from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset
        import pandas as pd

        dataset = Dataset.from_dict({
            "question": questions, "answer": answers,
            "contexts": contexts, "ground_truth": ground_truths,
        })
        embeddings = None
        try:
            from langchain_openai import OpenAIEmbeddings
            embeddings = OpenAIEmbeddings()
        except Exception:
            pass

        eval_kwargs = {"metrics": [faithfulness, answer_relevancy, context_precision, context_recall]}
        if embeddings is not None:
            eval_kwargs["embeddings"] = embeddings

        result = evaluate(dataset, **eval_kwargs)
        df = result.to_pandas()
        
        def safe_float(val):
            try:
                f_val = float(val)
                import math
                if math.isnan(f_val):
                    return 0.0
                return f_val
            except (ValueError, TypeError):
                return 0.0

        per_question = []
        for idx, row in df.iterrows():
            q = row.get("question") or row.get("user_input") or (questions[idx] if idx < len(questions) else "")
            ans = row.get("answer") or row.get("response") or (answers[idx] if idx < len(answers) else "")
            ctx = row.get("contexts") or row.get("retrieved_contexts") or (contexts[idx] if idx < len(contexts) else [])
            gt = row.get("ground_truth") or row.get("reference") or (ground_truths[idx] if idx < len(ground_truths) else "")
            per_question.append(EvalResult(
                question=str(q),
                answer=str(ans),
                contexts=list(ctx) if isinstance(ctx, (list, tuple)) else [str(ctx)],
                ground_truth=str(gt),
                faithfulness=safe_float(row.get("faithfulness", 0.0)),
                answer_relevancy=safe_float(row.get("answer_relevancy", 0.0)),
                context_precision=safe_float(row.get("context_precision", 0.0)),
                context_recall=safe_float(row.get("context_recall", 0.0))
            ))
        
        def get_agg_score(metric_name):
            if metric_name in df.columns:
                vals = [safe_float(v) for v in df[metric_name] if pd.notna(v)]
                return sum(vals) / len(vals) if vals else 0.0
            return 0.0

        return {
            "faithfulness": get_agg_score("faithfulness"),
            "answer_relevancy": get_agg_score("answer_relevancy"),
            "context_precision": get_agg_score("context_precision"),
            "context_recall": get_agg_score("context_recall"),
            "per_question": per_question
        }
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        return {"faithfulness": 0.0, "answer_relevancy": 0.0,
                "context_precision": 0.0, "context_recall": 0.0, "per_question": []}


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature"),
        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
        "answer_relevancy": ("Answer doesn't match question", "Improve prompt template"),
    }
    
    analyzed = []
    for r in eval_results:
        metrics = {
            "faithfulness": r.faithfulness,
            "context_recall": r.context_recall,
            "context_precision": r.context_precision,
            "answer_relevancy": r.answer_relevancy
        }
        avg_score = sum(metrics.values()) / 4.0
        worst_metric = min(metrics, key=metrics.get)
        
        analyzed.append({
            "question": r.question,
            "avg_score": avg_score,
            "worst_metric": worst_metric,
            "score": metrics[worst_metric],
            "diagnosis": diagnostic_tree[worst_metric][0],
            "suggested_fix": diagnostic_tree[worst_metric][1]
        })
        
    sorted_analyzed = sorted(analyzed, key=lambda x: x["avg_score"])
    return sorted_analyzed[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
