"""Eval harness: run the question set, score retrieval + answers, dump a table.

Usage:  python -m evals.run_eval [--no-hybrid] [--rerank]

Retrieval metrics (Hit@k, MRR) are computed against each question's expected
source file. Answer-quality metrics via ragas (faithfulness, answer relevance,
context precision) plug in on day 2 — see the TODO in score_answers().
Every run appends a markdown file under evals/results/ for the README table.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from agentic_rag.graph import ask
from agentic_rag.retriever import HybridRetriever

QUESTIONS = Path("evals/questions.jsonl")
RESULTS_DIR = Path("evals/results")


def load_questions() -> list[dict]:
    with open(QUESTIONS, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def retrieval_metrics(retriever: HybridRetriever, questions: list[dict], **flags) -> dict:
    hits, reciprocal_ranks = 0, []
    for q in questions:
        chunks = retriever.retrieve(q["question"], **flags)
        sources = [Path(c["source"]).name.lower() for c in chunks]
        expected = q["expected_source"].lower()
        if expected in sources:
            hits += 1
            reciprocal_ranks.append(1.0 / (sources.index(expected) + 1))
        else:
            reciprocal_ranks.append(0.0)
    n = len(questions)
    return {"hit_at_k": hits / n, "mrr": sum(reciprocal_ranks) / n}


def score_answers(rows: list[dict]) -> dict:
    # TODO(day2): run ragas evaluate() over {question, answer, contexts,
    # ground_truth} and return faithfulness / answer_relevancy /
    # context_precision means.
    return {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-hybrid", action="store_true", help="dense-only baseline")
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--skip-answers", action="store_true",
                        help="retrieval metrics only (no LLM calls)")
    args = parser.parse_args()

    questions = load_questions()
    retriever = HybridRetriever()
    flags = {"use_hybrid": not args.no_hybrid, "use_rerank": args.rerank}

    metrics = retrieval_metrics(retriever, questions, **flags)
    print(f"Retrieval: Hit@k={metrics['hit_at_k']:.3f}  MRR={metrics['mrr']:.3f}")

    rows = []
    if not args.skip_answers:
        for q in questions:
            result = ask(q["question"])
            keyword_hit = all(
                kw.lower() in result["answer"].lower() for kw in q["answer_keywords"]
            )
            rows.append({**q, **result, "keyword_hit": keyword_hit})
            print(f"  [{'ok' if keyword_hit else 'MISS'}] {q['question'][:70]}")
        metrics["keyword_accuracy"] = sum(r["keyword_hit"] for r in rows) / len(rows)
        metrics.update(score_answers(rows))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    mode = "baseline" if args.no_hybrid else ("hybrid+rerank" if args.rerank else "hybrid")
    out = RESULTS_DIR / f"{stamp}-{mode}.md"
    lines = [f"# Eval run {stamp} — mode: {mode}", "", "| Metric | Value |", "|---|---|"]
    lines += [f"| {k} | {v:.3f} |" if isinstance(v, float) else f"| {k} | {v} |"
              for k, v in metrics.items()]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
