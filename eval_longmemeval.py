"""Evaluate TRACE on LongMemEval with query-time session filtering."""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import statistics
import sys
import time
from collections import OrderedDict, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_graph_longmemeval import load_config
from eval_locomo import (
    TRACEGraphAgent,
    expand_pipeline_config,
    expand_reasoner_config,
    DEFAULT_TOKEN_OPT,
)


# LongMemEval-specific TokenOpt profile: same shape as LoCoMo's defaults but
# with ``enabled=False``. The minimal-format token-truncation pass does not
# transfer to LongMemEval, so the LongMemEval main + ablation runs use the
# origin pipeline (no truncation). Configs may opt back in via
# ``"longmemeval_token_opt": {"enabled": true}``.
LME_DEFAULT_TOKEN_OPT = {**DEFAULT_TOKEN_OPT, "enabled": False}


def expand_lme_token_opt_config(user_token_opt):
    """Merge user token_opt overrides into the LongMemEval default
    (which keeps the entire schema but disables the truncation pass)."""
    out = dict(LME_DEFAULT_TOKEN_OPT)
    if user_token_opt:
        out.update(user_token_opt)
    return out
from ingest_longmemeval import DEFAULT_DATASET, artifact_paths, default_memories_dir, safe_model_name
from trace.llm_text_io import parse_plain_text_answer
from trace.causal_graph import CausalGraph
from trace.dataset_adapter import LongMemEvalAdapter
from trace.event_schema import TypedEdge
from trace.prompts.longmemeval_judge import (
    SYSTEM_MESSAGE as JUDGE_SYSTEM_MESSAGE,
    format_longmemeval_judge_prompt,
)
from utils import aggregate_metrics, calculate_metrics

logger = logging.getLogger("eval_longmemeval")


class FilteredEmbeddingRetriever:
    """Retriever view over a subset of an existing embedding matrix."""

    def __init__(self, base_retriever, global_indices: List[int]):
        self.base_retriever = base_retriever
        self.model = base_retriever.model
        self.global_indices = list(global_indices)
        base_embeddings = getattr(base_retriever, "embeddings", None)
        if base_embeddings is None or not self.global_indices:
            self.embeddings = np.empty((0, 0))
        else:
            self.embeddings = base_embeddings[self.global_indices]
        self.corpus = [
            base_retriever.corpus[i]
            for i in self.global_indices
            if i < len(base_retriever.corpus)
        ]
        self.document_ids = {doc: i for i, doc in enumerate(self.corpus)}

    def search(self, query: str, k: int = 5):
        if self.embeddings is None or len(self.embeddings) == 0:
            return np.array([], dtype=int)
        query_embedding = self.model.encode([query])[0]
        denom = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        denom[denom == 0] = 1.0
        sims = self.embeddings @ query_embedding / denom
        k = min(k, len(sims))
        return np.argsort(sims)[-k:][::-1]


def load_longmemeval_memories(memories_dir: Path, tag: str, agent: TRACEGraphAgent):
    paths = artifact_paths(memories_dir, tag)
    missing = [
        p for p in [paths["memory_cache"], paths["retriever_cache"], paths["retriever_embeddings"], paths["session_note_map"]]
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing memory artifacts. Run ingest_longmemeval.py first:\n" + "\n".join(map(str, missing)))

    with paths["memory_cache"].open("rb") as f:
        memories = pickle.load(f)
    agent.memory_system.memories = memories
    agent.memory_system.retriever = agent.memory_system.retriever.load(
        str(paths["retriever_cache"]),
        str(paths["retriever_embeddings"]),
    )
    with paths["session_note_map"].open("r", encoding="utf-8") as f:
        session_note_map = json.load(f)
    return memories, agent.memory_system.retriever, session_note_map


def make_memory_view(base_memories: OrderedDict, base_retriever, allowed_note_ids: Set[str]):
    filtered = OrderedDict()
    global_indices = []
    global_to_local = {}
    for idx, (note_id, note) in enumerate(base_memories.items()):
        if note_id in allowed_note_ids:
            local_idx = len(filtered)
            global_to_local[idx] = local_idx
            filtered[note_id] = note
            global_indices.append(idx)

    # Remap note.links (global indices) to local indices via shallow copy
    # so neighborhood expansion indexes into the filtered list correctly.
    if global_to_local:
        import copy
        remapped = OrderedDict()
        for note_id, note in filtered.items():
            if getattr(note, 'links', None):
                note_copy = copy.copy(note)
                note_copy.links = [
                    global_to_local[g] for g in note.links if g in global_to_local
                ]
                remapped[note_id] = note_copy
            else:
                remapped[note_id] = note
        filtered = remapped

    return filtered, FilteredEmbeddingRetriever(base_retriever, global_indices)


def filter_graph_by_note_ids(graph: Optional[CausalGraph], allowed_note_ids: Set[str]) -> Optional[CausalGraph]:
    if graph is None:
        return None

    filtered = CausalGraph()
    allowed_event_ids = set()

    for event in graph.get_all_events():
        if any(note_id in allowed_note_ids for note_id in event.source_note_ids):
            event_copy = type(event).from_dict(event.to_dict())
            filtered.add_event(event_copy)
            allowed_event_ids.add(event.event_id)

    for u, v, data in graph.graph.edges(data=True):
        if u not in allowed_event_ids or v not in allowed_event_ids:
            continue
        edge_obj = data.get("data")
        if edge_obj is not None:
            edge = TypedEdge.from_dict(edge_obj.to_dict())
        else:
            edge = TypedEdge(
                source_event_id=u,
                target_event_id=v,
                edge_type=data.get("edge_type", "temporal_before"),
                confidence=data.get("confidence", 0.0),
                reason=data.get("reason", ""),
            )
        filtered.add_edge(edge)

    included_sessions = set()
    for sid, sdata in graph._sessions.items():
        note_ids = set(sdata.get("note_ids", []))
        if not note_ids & allowed_note_ids:
            continue
        session_copy = dict(sdata)
        session_copy["event_ids"] = [
            eid for eid in sdata.get("event_ids", [])
            if eid in allowed_event_ids
        ]
        if not session_copy["event_ids"]:
            continue
        filtered.add_session(session_copy)
        included_sessions.add(sid)

    for tid, tdata in graph._topics.items():
        session_ids = [
            sid for sid in tdata.get("session_ids", [])
            if sid in included_sessions
        ]
        if not session_ids:
            continue
        topic_copy = dict(tdata)
        topic_copy["session_ids"] = session_ids
        filtered.add_topic(topic_copy)

    return filtered


def allowed_notes_for_question(session_note_map: Dict[str, List[str]], haystack_session_ids: Iterable[str]) -> Set[str]:
    allowed = set()
    for sid in haystack_session_ids:
        allowed.update(session_note_map.get(sid, []))
    return allowed


def get_judge_client(api_base: Optional[str], api_key: Optional[str]):
    from openai import OpenAI
    kwargs = {"api_key": api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")}
    if api_base:
        kwargs["base_url"] = api_base
    return OpenAI(**kwargs)


def judge_single(client, model: str, item: Dict, max_retries: int = 3) -> Tuple[str, int]:
    prompt = format_longmemeval_judge_prompt(
        question=item.get("raw_question") or item["question"],
        gold_answer=item["reference"],
        generated_answer=item["prediction"],
        question_type=item.get("original_question_type", "unknown"),
    )
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            content = response.choices[0].message.content
            label = json.loads(content).get("label", "WRONG").strip().upper()
            return label, 1 if label == "CORRECT" else 0
        except Exception as exc:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                logger.warning("Judge failed (%s); retrying in %ss", exc, wait)
                time.sleep(wait)
            else:
                logger.error("Judge failed after retries: %s", exc)
    return "ERROR", 0


def run_judge(result_path: str, judge_model: str, judge_runs: int, api_base: Optional[str], api_key: Optional[str]) -> Dict:
    with open(result_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    client = get_judge_client(api_base, api_key)
    all_runs = []

    for run_id in range(1, judge_runs + 1):
        judgments = []
        for item in tqdm(data.get("individual_results", []), desc=f"Judge run {run_id}"):
            label, score = judge_single(client, judge_model, item)
            judgments.append({
                "question_id": item.get("question_id"),
                "question": item.get("raw_question") or item["question"],
                "prediction": item["prediction"],
                "reference": item["reference"],
                "original_question_type": item.get("original_question_type"),
                "judgment": label,
                "score": score,
            })

        scores = [j["score"] for j in judgments]
        per_type = defaultdict(list)
        for j in judgments:
            per_type[j["original_question_type"]].append(j["score"])
        run_payload = {
            "source_file": result_path,
            "judge_model": judge_model,
            "judge_run_id": run_id,
            "overall_llm_score": statistics.mean(scores) if scores else 0.0,
            "per_type_llm_score": {
                qtype: {
                    "score": statistics.mean(vals),
                    "count": len(vals),
                    "correct": sum(vals),
                }
                for qtype, vals in sorted(per_type.items())
            },
            "individual_judgments": judgments,
        }
        all_runs.append(run_payload)
        out_path = Path(result_path).with_name(f"judge_{Path(result_path).stem}_run{run_id}.json")
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(run_payload, f, indent=2, ensure_ascii=False)
        print(f"Judge run {run_id}: {run_payload['overall_llm_score']:.4f}")

    overall_values = [r["overall_llm_score"] for r in all_runs]
    return {
        "num_judge_runs": judge_runs,
        "overall_llm_score_mean": statistics.mean(overall_values) if overall_values else 0.0,
        "overall_llm_score_std": statistics.stdev(overall_values) if len(overall_values) > 1 else 0.0,
        "overall_llm_score_values": overall_values,
    }


def evaluate(
    dataset_path: str,
    config: Dict,
    memories_dir: Path,
    graph_dir: Path,
    tag: str,
    output_path: str,
    sample_idx: Optional[int] = None,
    limit: Optional[int] = None,
    lite: bool = False,
    exclude_adversarial: bool = True,
    question_ids: Optional[List[str]] = None,
):
    model = config.get("model", "openai/gpt-4o-mini")
    backend = config.get("backend", "openai")
    api_base = config.get("api_base", "https://openrouter.ai/api/v1")
    api_key = config.get("api_key") or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    retrieve_k = int(config.get("retrieve_k", 10))

    adapter = LongMemEvalAdapter(dataset_path, sample_idx=sample_idx, limit=limit)
    if lite:
        graph = None
        logger.info("lite=True: TRACElite mode — skipping graph load; "
                    "L1 hybrid + L3 prompt + TokenOpt only")
    else:
        graph_file = graph_dir / f"event_graph_longmemeval_{tag}.json"
        graph = CausalGraph.load(str(graph_file)) if graph_file.exists() else None
        if graph is None:
            logger.warning("Graph not found at %s; evaluation will use A-Mem fallback", graph_file)

    agent = TRACEGraphAgent(
        model=model,
        backend=backend,
        retrieve_k=retrieve_k,
        temperature_c5=float(config.get("temperature_c5", 0.5)),
        api_key=api_key,
        api_base=api_base,
        skip_evolution=True,
        graph=None,
        reasoner_config=expand_reasoner_config(config.get("reasoner")),
        pipeline_config=expand_pipeline_config(config.get("pipeline")),
        token_opt_config=expand_lme_token_opt_config(config.get("longmemeval_token_opt")),
        lite_mode=lite,
    )
    base_memories, base_retriever, session_note_map = load_longmemeval_memories(memories_dir, tag, agent)

    results = []
    all_metrics = []
    all_categories = []
    category_counts = defaultdict(int)
    type_counts = defaultdict(int)
    path_stats = {"with_paths": 0, "without_paths": 0, "fallback": 0}

    qas = adapter.get_all_qa_pairs()
    if exclude_adversarial:
        kept = [qa for qa in qas if not str(qa.get("question_id", "")).endswith("_abs")]
        dropped = len(qas) - len(kept)
        logger.info(
            "exclude_adversarial=True: dropped %d adversarial questions (question_id endswith '_abs'); %d remaining",
            dropped, len(kept),
        )
        qas = kept
    else:
        logger.info("exclude_adversarial=False: evaluating all %d questions (including adversarial)", len(qas))
    if question_ids:
        wanted = set(question_ids)
        kept = [qa for qa in qas if qa.get("question_id") in wanted]
        logger.info("question_ids filter: kept %d/%d (requested %d)", len(kept), len(qas), len(wanted))
        qas = kept
    for q_idx, qa in enumerate(tqdm(qas, desc=f"LongMemEval {tag}")):
        allowed_note_ids = allowed_notes_for_question(session_note_map, qa["haystack_session_ids"])
        if not allowed_note_ids:
            logger.warning("Question %s has no allowed notes", qa.get("question_id"))

        filtered_memories, filtered_retriever = make_memory_view(base_memories, base_retriever, allowed_note_ids)
        agent.memory_system.memories = filtered_memories
        agent.memory_system.retriever = filtered_retriever
        filtered_graph = filter_graph_by_note_ids(graph, allowed_note_ids)
        if filtered_graph is not None:
            agent.set_graph(filtered_graph, sample=None)
        elif lite:
            # TRACElite: empty graph + init pipeline so format_context can run
            # with L1 hybrid; retrieve()/Phase 2 are bypassed in answer_question.
            agent.graph = CausalGraph()
            agent._init_pipeline()
        else:
            agent.graph = None
            agent.pipeline = None

        try:
            raw_prediction, prompt_used, raw_context = agent.answer_question(
                qa["question"],
                int(qa["category"]),
                str(qa["answer"] or ""),
                question_type=qa.get("original_question_type"),
            )
        except Exception as exc:
            logger.warning("Question %s failed: %s", qa.get("question_id"), exc)
            raw_prediction, prompt_used, raw_context = "", "", "[ERROR]"

        prediction = parse_plain_text_answer(raw_prediction or "")
        if qa.get("original_question_type") == "temporal-reasoning":
            from trace.llm_text_io import parse_temporal_choice_answer
            prediction = parse_temporal_choice_answer(
                raw=raw_prediction or "",
                question=qa.get("raw_question") or qa.get("question") or "",
                fallback=prediction,
            )
        metrics = calculate_metrics(prediction, qa["answer"]) if qa.get("answer") else {
            "exact_match": 0,
            "f1": 0.0,
            "rouge1_f": 0.0,
            "rouge2_f": 0.0,
            "rougeL_f": 0.0,
            "bleu1": 0.0,
            "bleu2": 0.0,
            "bleu3": 0.0,
            "bleu4": 0.0,
            "bert_f1": 0.0,
            "meteor": 0.0,
            "sbert_similarity": 0.0,
        }

        all_metrics.append(metrics)
        all_categories.append(int(qa["category"]))
        category_counts[int(qa["category"])] += 1
        type_counts[qa["original_question_type"]] += 1
        if agent.pipeline is None:
            path_stats["fallback"] += 1
        elif raw_context and raw_context not in ("[NO_PATHS_FOUND]", "[ERROR]"):
            path_stats["with_paths"] += 1
        else:
            path_stats["without_paths"] += 1

        results.append({
            "sample_id": q_idx,
            "question_id": qa.get("question_id"),
            "question": qa["question"],
            "raw_question": qa["raw_question"],
            "question_date": qa.get("question_date"),
            "reference": qa["answer"],
            "prediction": prediction,
            "raw_response": raw_prediction,
            "category": int(qa["category"]),
            "original_question_type": qa["original_question_type"],
            "haystack_session_ids": qa["haystack_session_ids"],
            "answer_session_ids": qa["answer_session_ids"],
            "metrics": metrics,
            "path_explanation": raw_context[:2000] if raw_context else "",
        })

    aggregate = aggregate_metrics(all_metrics, all_categories)

    output = {
        "model": model,
        "dataset": dataset_path,
        "backend": backend,
        "api_base": api_base,
        "run_label": config.get("run_label", "trace_longmemeval"),
        "tag": tag,
        "timestamp": datetime.now().isoformat(),
        "total_questions": len(results),
        "exclude_adversarial": exclude_adversarial,
        "category_distribution": dict(category_counts),
        "question_type_distribution": dict(type_counts),
        "path_stats": path_stats,
        "reasoner_config": config.get("reasoner", {}),
        "pipeline_config": config.get("pipeline", {}),
        "longmemeval_token_opt_config": config.get("longmemeval_token_opt", {}),
        "aggregate_metrics": aggregate,
        "individual_results": results,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info("Saved %s", output_path)
    return output


def main():
    parser = argparse.ArgumentParser(description="Evaluate TRACE on LongMemEval")
    parser.add_argument("--config", default="configs/longmemeval_main.json")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--memories-dir", default=None)
    parser.add_argument("--graph_dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--result", default=None, help="Judge an existing result JSON instead of running inference")
    parser.add_argument("--judge_model", default="openai/gpt-4o-mini")
    parser.add_argument("--judge_runs", type=int, default=3,
                        help="Number of LLM-judge runs after eval (default: 3, set 0 to skip).")
    parser.add_argument("--skip_graph_load", "--lite", dest="skip_graph_load", action="store_true",
                        help="TRACElite mode: skip graph load and BFS/Phase 2; "
                             "use only L1 hybrid context + L3 prompt.")
    parser.add_argument("--include_adversarial", action="store_true",
                        help="Evaluate all 500 questions including adversarial (question_id endswith '_abs'). "
                             "Default (this flag absent) excludes 30 adversarial to align with the n=470 cohort.")
    parser.add_argument("--question_ids", default=None,
                        help="Comma-separated question_ids to filter to (smoke runs). Applied AFTER "
                             "exclude_adversarial. Use to run a stratified subset for token-budget probes.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    config = load_config(args.config)
    # Allow skip_graph_load to be set via either CLI --skip_graph_load (alias --lite) or config "lite": true
    if config.get("lite") is True:
        args.skip_graph_load = True
    dataset_path = args.dataset or config.get("dataset", DEFAULT_DATASET)
    model = config.get("model", "openai/gpt-4o-mini")
    backend = config.get("backend", "openai")
    tag = LongMemEvalAdapter.subset_tag(args.sample, args.limit)

    if args.result:
        summary = run_judge(
            result_path=args.result,
            judge_model=args.judge_model,
            judge_runs=max(1, args.judge_runs),
            api_base=config.get("api_base", "https://openrouter.ai/api/v1"),
            api_key=config.get("api_key"),
        )
        print(json.dumps(summary, indent=2))
        return

    memories_dir = Path(args.memories_dir or default_memories_dir(backend, model))
    graph_dir = Path(args.graph_dir or f"cached_graphs_longmemeval_{backend}_{safe_model_name(model)}")
    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        # All LongMemEval eval outputs (main + ablations) land in results/longmemeval/.
        out_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "results" / "longmemeval"
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(out_dir / f"trace_longmemeval_{tag}_{ts}.json")

    evaluate(
        dataset_path=dataset_path,
        config=config,
        memories_dir=memories_dir,
        graph_dir=graph_dir,
        tag=tag,
        output_path=args.output,
        sample_idx=args.sample,
        limit=args.limit,
        lite=args.skip_graph_load,
        exclude_adversarial=not args.include_adversarial,
        question_ids=([q.strip() for q in args.question_ids.split(",") if q.strip()]
                       if args.question_ids else None),
    )

    if args.judge_runs > 0:
        summary = run_judge(
            result_path=args.output,
            judge_model=args.judge_model,
            judge_runs=args.judge_runs,
            api_base=config.get("api_base", "https://openrouter.ai/api/v1"),
            api_key=config.get("api_key"),
        )
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
