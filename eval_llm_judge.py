"""
LLM-Judge evaluation for TRACE.

Protocol: judge model (default gpt-4o-mini), temperature=0.0, JSON output,
3 judge runs, skips category 5 (adversarial) in LLM-judge aggregation.

This script runs LLM-judge on existing result files (no re-inference needed).

Usage:
    # Judge all week1 results (3 judge runs, report mean ± std)
    python eval_llm_judge.py --result_dir results/week1 --judge_runs 3

    # Judge a single result file
    python eval_llm_judge.py --result results/week1/amem_1_20260316_142404.json
"""

import os
import json
import argparse
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, asdict
import statistics

from openai import OpenAI
from tqdm import tqdm

logger = logging.getLogger("llm_judge")

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

LOCOMO_ACCURACY_PROMPT = """
Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
    (1) a question (posed by one user to another user),
    (2) a 'gold' (ground truth) answer,
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
"""

SYSTEM_MESSAGE = "You are an expert grader that determines if answers to questions match a gold standard answer."

# Category names for LoCoMo
CATEGORY_NAMES = {
    1: "Multi-Hop",
    2: "Temporal",
    3: "Open Domain",
    4: "Single-Hop",
    5: "Adversarial",
}

# Skip category 5 (adversarial) in LLM-judge evaluation
SKIP_CATEGORIES = {5}


@dataclass
class JudgeResult:
    sample_id: int
    question: str
    prediction: str
    reference: str
    category: int
    judgment: str  # "CORRECT" or "WRONG" or "SKIPPED" or "ERROR"
    score: int     # 1 or 0


def get_judge_client(api_base: Optional[str] = None, api_key: Optional[str] = None) -> OpenAI:
    """Create OpenAI client for judge model."""
    kwargs = {}
    if api_base:
        kwargs["base_url"] = api_base
    if api_key:
        kwargs["api_key"] = api_key
    else:
        kwargs["api_key"] = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    return OpenAI(**kwargs)


def judge_single(
    client: OpenAI,
    model: str,
    question: str,
    prediction: str,
    reference: str,
    category: int,
    max_retries: int = 3,
) -> Tuple[str, int]:
    """Judge a single QA pair.
    Returns (judgment_label, score)."""
    # Skip category 5 (adversarial)
    if category in SKIP_CATEGORIES:
        return "SKIPPED", 0

    prompt = LOCOMO_ACCURACY_PROMPT.format(
        question=question,
        gold_answer=reference,
        generated_answer=prediction,
    )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            content = response.choices[0].message.content
            label = json.loads(content).get("label", "WRONG").strip().upper()
            score = 1 if label == "CORRECT" else 0
            return label, score
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                logger.warning(f"Judge call failed (attempt {attempt+1}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                logger.error(f"Judge call failed after {max_retries} attempts: {e}")
                return "ERROR", 0


def judge_result_file(
    result_path: str,
    client: OpenAI,
    judge_model: str,
    judge_run_id: int = 1,
    batch_delay: float = 0.1,
) -> Dict:
    """Judge all QA pairs in a single result file."""
    with open(result_path, encoding="utf-8") as f:
        data = json.load(f)

    results = data["individual_results"]
    run_label = data.get("run_label", Path(result_path).stem)
    logger.info(f"Judging {len(results)} questions from {run_label} (judge run {judge_run_id})")

    judgments: List[JudgeResult] = []
    skipped = 0
    for item in tqdm(results, desc=f"Judging {run_label}"):
        label, score = judge_single(
            client=client,
            model=judge_model,
            question=item["question"],
            prediction=item["prediction"],
            reference=item["reference"],
            category=item["category"],
        )
        if label == "SKIPPED":
            skipped += 1
            continue
        judgments.append(JudgeResult(
            sample_id=item["sample_id"],
            question=item["question"],
            prediction=item["prediction"],
            reference=item["reference"],
            category=item["category"],
            judgment=label,
            score=score,
        ))
        if batch_delay > 0:
            time.sleep(batch_delay)

    # Aggregate
    all_scores = [j.score for j in judgments]
    category_scores = defaultdict(list)
    for j in judgments:
        category_scores[j.category].append(j.score)

    overall_score = statistics.mean(all_scores) if all_scores else 0.0

    per_category = {}
    for cat in sorted(category_scores.keys()):
        scores = category_scores[cat]
        cat_name = CATEGORY_NAMES.get(cat, f"cat_{cat}")
        per_category[cat_name] = {
            "score": statistics.mean(scores),
            "count": len(scores),
            "correct": sum(scores),
        }

    return {
        "source_file": str(result_path),
        "run_label": run_label,
        "judge_model": judge_model,
        "judge_run_id": judge_run_id,
        "timestamp": datetime.now().isoformat(),
        "total_questions": len(results),
        "judged_questions": len(judgments),
        "skipped_questions": skipped,
        "skipped_categories": sorted(SKIP_CATEGORIES),
        "overall_llm_score": overall_score,
        "per_category_llm_score": per_category,
        "errors": sum(1 for j in judgments if j.judgment == "ERROR"),
        "individual_judgments": [asdict(j) for j in judgments],
    }


def aggregate_judge_runs(judge_results: List[Dict]) -> Dict:
    """Aggregate multiple judge runs (for mean ± std reporting)."""
    if not judge_results:
        return {}

    overall_scores = [r["overall_llm_score"] for r in judge_results]

    # Collect per-category scores across runs
    all_categories = set()
    for r in judge_results:
        all_categories.update(r["per_category_llm_score"].keys())

    per_category_agg = {}
    for cat in sorted(all_categories):
        cat_scores = []
        for r in judge_results:
            if cat in r["per_category_llm_score"]:
                cat_scores.append(r["per_category_llm_score"][cat]["score"])
        per_category_agg[cat] = {
            "mean": statistics.mean(cat_scores) if cat_scores else 0.0,
            "std": statistics.stdev(cat_scores) if len(cat_scores) > 1 else 0.0,
            "values": cat_scores,
        }

    return {
        "num_judge_runs": len(judge_results),
        "overall_llm_score_mean": statistics.mean(overall_scores),
        "overall_llm_score_std": statistics.stdev(overall_scores) if len(overall_scores) > 1 else 0.0,
        "overall_llm_score_values": overall_scores,
        "per_category_llm_score": per_category_agg,
    }


def main():
    parser = argparse.ArgumentParser(description="LLM-Judge evaluation for TRACE")
    parser.add_argument("--result", type=str, required=True,
                        help="Path to a single result JSON file (output of eval_locomo.py / eval_longmemeval.py)")
    parser.add_argument("--judge_model", type=str, default="openai/gpt-4o-mini",
                        help="Model to use as judge (default: openai/gpt-4o-mini)")
    parser.add_argument("--judge_runs", type=int, default=3,
                        help="Number of judge runs for stability (default: 3)")
    parser.add_argument("--api_base", type=str, default="https://openrouter.ai/api/v1",
                        help="API base URL")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for judge JSON files (default: same as --result)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")

    client = get_judge_client(api_base=args.api_base)

    # Judge files are written next to the source result by default.
    output_dir = args.output_dir or str(Path(args.result).parent)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    judge_results = []
    for run_id in range(1, args.judge_runs + 1):
        r = judge_result_file(args.result, client, args.judge_model, run_id, batch_delay=0.05)
        judge_results.append(r)
        out_path = Path(output_dir) / f"judge_{Path(args.result).stem}_run{run_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(r, f, indent=2, ensure_ascii=False)
        print(f"Run {run_id}: LLM Score = {r['overall_llm_score']:.4f}")

    agg = aggregate_judge_runs(judge_results)
    print(f"\nAggregated ({args.judge_runs} runs):")
    print(f"  Overall LLM Score: {agg['overall_llm_score_mean']:.4f} ± {agg['overall_llm_score_std']:.4f}")
    for cat, vals in agg["per_category_llm_score"].items():
        print(f"  {cat}: {vals['mean']:.4f} ± {vals['std']:.4f}")


if __name__ == "__main__":
    main()
