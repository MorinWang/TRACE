"""TRACE evaluation harness for LoCoMo with graph-augmented path reasoning.

Extends ``trace.agent.TRACEAgent`` with ``TRACEGraphAgent`` that uses the
causal-temporal event graph for path-grounded retrieval and explanation.
Supports pipeline-layer ablation toggles.

Usage:
    python eval_locomo.py --config configs/locomo_main.json
    python eval_locomo.py --config configs/locomo_main.json --sample 0
"""

import argparse
import hashlib
import json
import logging
import os
import pickle
import sys
from collections import defaultdict
from datetime import datetime
from typing import Optional

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trace.agent import (
    TRACEAgent,
    compute_prompt_hash,
    ANSWER_PROMPT_CATEGORY_2,
    ANSWER_PROMPT_DEFAULT,
)
from load_dataset import load_locomo_dataset
from memory_layer_robust import RobustLLMController, RobustAgenticMemorySystem
from utils import calculate_metrics, aggregate_metrics
from trace.causal_graph import CausalGraph
from trace.llm_text_io import parse_plain_text_answer
from trace.path_reasoner import PathReasoner
from trace.trace_pipeline import TRACEPipeline, PipelineConfig, RetrievalResult, TokenOptConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("eval_locomo")


# ---------------------------------------------------------------------------
# Default config blocks for LoCoMo
#
# Configs in configs/locomo_*.json carry only the *ablation-controlled flags*;
# everything else (token_opt budgets, default reasoner weights, fanout/hops
# derived from session/topic_augmentation) is filled in here. CLI flags
# (--api_base, --model, ...) override these defaults; per-config JSON values
# override the CLI defaults.
# ---------------------------------------------------------------------------

DEFAULT_REASONER = {
    "alpha": 0.4,
    "beta": 0.3,
    "gamma": 0.15,
    "delta": 0.15,
    "max_depth": 4,
    "top_k": 10,
    "neural_sim_aggregator": "mean",
}

DEFAULT_TOKEN_OPT = {
    "enabled": True,
    "notes_budget": 6000,
    "path_budget": 3000,
    "total_cap": 10000,
    "dedup_by_event_id": True,
    "dedup_section1": "note_id",
    "min_path_score": 0.3,
    "source_note_query_filter": 0.4,
    "note_format_mode": "minimal",
    "use_llmlingua": False,
    "tiktoken_encoding": "o200k_base",
}

DEFAULT_PIPELINE = {
    "layer1_hybrid_context": True,
    "layer1_neighborhood_expansion": True,
    "layer3_enhanced_qa_prompt": True,
    "layer4_graph_expansion": True,
    "layer5_query_entity_extraction": True,
    "session_augmentation": True,
    "topic_augmentation": True,
    "hierarchical_seed_selection": True,
    "answer_aware_paths": True,
    "answer_aware_top_k": 5,
}


def expand_pipeline_config(user_pipeline: Optional[dict]) -> dict:
    """Merge user pipeline overrides into the default ablation skeleton.

    Auto-derives fanout/hops fields from the corresponding boolean toggles so
    ablation configs only need to specify the high-level flag (e.g.
    ``session_augmentation: false`` automatically zeros out
    ``max_session_fanout`` and ``max_session_hops``).
    """
    pipeline = dict(DEFAULT_PIPELINE)
    if user_pipeline:
        pipeline.update(user_pipeline)

    # Derive: when an augmentation is OFF, its fanout/hops collapse to 0.
    pipeline["max_session_fanout"] = 3 if pipeline.get("session_augmentation") else 0
    pipeline["max_session_hops"] = 1 if pipeline.get("session_augmentation") else 0
    pipeline["max_topic_fanout"] = 2 if pipeline.get("topic_augmentation") else 0
    pipeline["max_topic_hops"] = 1 if pipeline.get("topic_augmentation") else 0
    if not pipeline.get("answer_aware_paths"):
        pipeline["answer_aware_top_k"] = 0
    return pipeline


def expand_reasoner_config(user_reasoner: Optional[dict]) -> dict:
    """Merge user reasoner overrides into the default reasoner config."""
    out = dict(DEFAULT_REASONER)
    if user_reasoner:
        out.update(user_reasoner)
    return out


def expand_token_opt_config(user_token_opt: Optional[dict]) -> dict:
    """Merge user token_opt overrides into the default TokenOpt config."""
    out = dict(DEFAULT_TOKEN_OPT)
    if user_token_opt:
        out.update(user_token_opt)
    return out


# ---------------------------------------------------------------------------
# TRACE Graph Agent
# ---------------------------------------------------------------------------

class TRACEGraphAgent(TRACEAgent):
    """Agent that uses graph-augmented path reasoning for retrieval."""

    def __init__(
        self,
        model, backend, retrieve_k, temperature_c5,
        api_key=None, api_base=None,
        skip_evolution=False,
        graph: Optional[CausalGraph] = None,
        reasoner_config: Optional[dict] = None,
        pipeline_config: Optional[dict] = None,
        token_opt_config: Optional[dict] = None,
        embedding_model: str = 'all-MiniLM-L6-v2',
        lite_mode: bool = False,
    ):
        super().__init__(
            model, backend, retrieve_k, temperature_c5,
            api_key=api_key, api_base=api_base,
            skip_evolution=skip_evolution,
            embedding_model=embedding_model,
        )
        self.graph = graph
        self.lite_mode = lite_mode
        reasoner_config = reasoner_config or {}
        self.reasoner = PathReasoner(**reasoner_config)

        # Build PipelineConfig from dict
        p_cfg = pipeline_config or {}
        self.pipeline_config = PipelineConfig(**{
            k: v for k, v in p_cfg.items()
            if k in PipelineConfig.__dataclass_fields__
        })

        # Build TokenOptConfig from dict (defaults: disabled)
        tok_cfg = token_opt_config or {}
        self.token_opt = TokenOptConfig(**{
            k: v for k, v in tok_cfg.items()
            if k in TokenOptConfig.__dataclass_fields__
        })

        self.entity_extractor = None

        if self.pipeline_config.layer5_query_entity_extraction:
            try:
                from trace.query_entity_extractor import QueryEntityExtractor
                self.entity_extractor = QueryEntityExtractor(
                    llm=self.memory_system.llm_controller.llm if hasattr(self, 'memory_system') else None
                )
            except Exception as e:
                logger.warning(f"Could not init QueryEntityExtractor: {e}")

        self.pipeline = None
        if graph is not None:
            self._init_pipeline()

    def _init_pipeline(self):
        """Initialize pipeline with current graph and config."""
        if self.pipeline_config.layer5_query_entity_extraction and self.entity_extractor is None:
            try:
                from trace.query_entity_extractor import QueryEntityExtractor
                self.entity_extractor = QueryEntityExtractor(
                    llm=self.memory_system.llm_controller.llm
                )
            except Exception as e:
                logger.warning(f"Could not init QueryEntityExtractor: {e}")

        self.pipeline = TRACEPipeline(
            graph=self.graph,
            reasoner=self.reasoner,
            config=self.pipeline_config,
            context_filter=None,
            entity_extractor=self.entity_extractor,
            token_opt=self.token_opt,
        )

    def set_graph(self, graph: CausalGraph, sample=None):
        """Set the event graph (called per-sample after loading).

        If session_augmentation is enabled and sample is provided,
        injects session auxiliary nodes into the graph.
        """
        self.graph = graph
        if (self.pipeline_config.session_augmentation
                and sample is not None
                and hasattr(sample, 'session_summary')
                and sample.session_summary):
            # Skip if graph already has sessions (built by hierarchical builder)
            if not graph._sessions:
                from trace.session_injector import inject_session_nodes
                inject_session_nodes(
                    graph=graph,
                    memories=self.memory_system.memories,
                    session_summaries=sample.session_summary,
                    conversation_sessions=(
                        sample.conversation.sessions
                        if hasattr(sample, 'conversation') and hasattr(sample.conversation, 'sessions')
                        else {}
                    ),
                    sample_idx=int(getattr(sample, 'sample_id', '0').replace('conv-', '')),
                )
        # Method C: Topic clustering (after session injection)
        if (self.pipeline_config.topic_augmentation
                and graph._sessions):
            # Skip if graph already has topics (built by hierarchical builder)
            if not graph._topics:
                from trace.session_injector import inject_topic_nodes
                inject_topic_nodes(
                    graph=graph,
                    llm=self.memory_system.llm_controller.llm,
                    sample_idx=int(getattr(sample, 'sample_id', '0').replace('conv-', '')),
                )
        self._init_pipeline()

    def answer_question(self, question: str, category: int, answer: str,
                        context_override: Optional[str] = None,
                        question_type: Optional[str] = None) -> tuple:
        """Override: inject graph path reasoning between retrieval and answering."""
        if context_override is not None:
            return super().answer_question(question, category, answer, context_override)

        if self.pipeline is None:
            return super().answer_question(question, category, answer)

        # 1. Neural retrieval
        keywords = self.generate_query_llm(question)

        # 2. Get raw indices
        indices = self.memory_system.retriever.search(keywords, k=self.retrieve_k)
        if not isinstance(indices, np.ndarray):
            indices = np.array(indices)

        # 3. Get query embedding
        query_embedding = self.memory_system.retriever.model.encode([keywords])[0]

        if self.lite_mode:
            # TRACElite: skip BFS / Phase 2; format_context with empty top_paths
            # still runs L1 hybrid section1 (notes + neighborhood) and TokenOpt
            # truncation, then L3 enhanced prompt.
            context = self.pipeline.format_context(
                top_paths=[],
                memory_system=self.memory_system,
                fallback_indices=indices,
                query=question,
                final_k=self.retrieve_k,
                question_category=category,
            )
            raw_context = "[LITE_NO_GRAPH]"
            if self.pipeline_config.layer3_enhanced_qa_prompt:
                user_prompt, temperature = self._build_enhanced_prompt(
                    question, category, answer, context, question_type=question_type
                )
            else:
                user_prompt, temperature = self._build_legacy_prompt(
                    question, category, answer, context
                )
            try:
                response = self.memory_system.llm_controller.llm.get_completion(
                    user_prompt, temperature=temperature,
                )
            except Exception as e:
                logger.warning("answer_question (lite) failed: %s - returning empty", e)
                response = ""
            return response, user_prompt, raw_context

        # 4. TRACE pipeline
        result = self.pipeline.retrieve(
            query=question,
            retrieval_indices=indices,
            memory_system=self.memory_system,
            query_embedding=query_embedding,
            final_k=self.retrieve_k,
            question_category=category,
        )

        # 5. Context and explanation
        context = result.context
        raw_context = result.explanation if result.explanation else "[NO_PATHS_FOUND]"

        # 6. Generate answer - Layer 3: enhanced prompts or legacy
        if self.pipeline_config.layer3_enhanced_qa_prompt:
            user_prompt, temperature = self._build_enhanced_prompt(
                question, category, answer, context, question_type=question_type
            )
        else:
            user_prompt, temperature = self._build_legacy_prompt(
                question, category, answer, context
            )

        try:
            response = self.memory_system.llm_controller.llm.get_completion(
                user_prompt, temperature=temperature,
            )
        except Exception as e:
            logger.warning("answer_question failed: %s - returning empty", e)
            response = ""

        # Phase 2: Answer-aware support path search (Method D)
        if (self.pipeline_config.answer_aware_paths
                and self.pipeline is not None
                and response):
            try:
                model = self.memory_system.retriever.model
                answer_emb = model.encode([response])[0]
                q_type = self.detect_question_type(question)
                support_result = self.pipeline.retrieve_with_support(
                    query=question,
                    answer=response,
                    retrieval_indices=indices,
                    memory_system=self.memory_system,
                    query_embedding=query_embedding,
                    answer_embedding=answer_emb,
                    question_type=q_type,
                )
                if support_result.explanation:
                    raw_context = support_result.explanation
                    logger.info(
                        f"Phase 2: found {support_result.num_paths_found} support paths"
                        f" (type={q_type})"
                    )
            except Exception as e:
                logger.warning("Phase 2 answer-aware path search failed: %s", e)

        return response, user_prompt, raw_context

    def _build_legacy_prompt(self, question, category, answer, context):
        """Fallback A-Mem prompt templates (used when L3 enhanced QA prompt is off)."""
        if category == 2:
            prompt = ANSWER_PROMPT_CATEGORY_2.format(context=context, question=question)
            return prompt, 0.7
        else:
            prompt = ANSWER_PROMPT_DEFAULT.format(context=context, question=question)
            return prompt, 0.7

    @staticmethod
    def detect_question_type(question: str) -> str:
        """Content-based question type detection (dataset-agnostic)."""
        q = question.lower().strip()
        temporal_starters = [
            "when ", "how long ", "how many years", "how many months",
            "what time ", "what date ", "since when", "how old ",
        ]
        temporal_keywords = [
            "how long ago", "what year", "what month",
        ]
        if any(q.startswith(p) for p in temporal_starters):
            return "temporal"
        if any(k in q for k in temporal_keywords):
            return "temporal"

        return "default"

    def _build_enhanced_prompt(self, question, category, answer, context,
                               question_type: Optional[str] = None):
        """Layer 3: Enhanced QA prompts with CoT reasoning.

        Adapts to three non-adversarial question shapes:
          - cat 3 (Open Domain) → QA_PROMPT_OD (relaxed answer suffix)
          - LongMemEval SSP     → QA_PROMPT_PREFERENCE (preference profiling)
          - temporal phrasing   → QA_PROMPT_TEMPORAL (absolute-date discipline)
          - everything else     → QA_PROMPT_DEFAULT (cat 1 / cat 4 / generic)

        Cat 5 (Adversarial) is excluded from the QA loop by default
        (`--skip_cat5`); evaluation aggregation drops it per the n=1540 cohort
        convention. With `--include_cat5`, cat 5 questions fall through to
        QA_PROMPT_DEFAULT — they are not part of any reported metric.
        """
        try:
            from trace.prompts.qa_prompt import (
                QA_PROMPT_DEFAULT, QA_PROMPT_TEMPORAL,
                QA_PROMPT_OD, QA_PROMPT_PREFERENCE,
            )
        except ImportError:
            return self._build_legacy_prompt(question, category, answer, context)

        if category == 3:
            # A4: OD questions get relaxed answer suffix
            prompt = QA_PROMPT_OD.format(context=context, question=question)
            return prompt, 0.7
        elif question_type == "single-session-preference":
            # SSP: profile user preferences instead of answering recommendation directly
            prompt = QA_PROMPT_PREFERENCE.format(context=context, question=question)
            return prompt, 0.7
        elif self.detect_question_type(question) == "temporal":
            prompt = QA_PROMPT_TEMPORAL.format(context=context, question=question)
            return prompt, 0.7
        else:
            prompt = QA_PROMPT_DEFAULT.format(context=context, question=question)
            return prompt, 0.7

    @staticmethod
    def _is_abstention(response: str) -> bool:
        """Check if the response is an 'I don't know' abstention."""
        lower = response.lower()
        abstention_phrases = [
            "not mentioned", "no information", "i don't know",
            "cannot determine", "not enough information",
            "no evidence", "not provided", "unclear from",
        ]
        return any(p in lower for p in abstention_phrases)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return json.load(f)


def evaluate_trace_dataset(
    dataset_path: str,
    model: str,
    output_path: Optional[str] = None,
    backend: str = "openai",
    temperature_c5: float = 0.5,
    retrieve_k: int = 10,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    graph_dir: str = "cached_graphs_openai_openai_gpt-4o-mini",
    reasoner_config: Optional[dict] = None,
    pipeline_config: Optional[dict] = None,
    token_opt_config: Optional[dict] = None,
    sample_idx: Optional[int] = None,
    run_label: str = "locomo_main",
    embedding_model: str = 'all-MiniLM-L6-v2',
    memories_dir: Optional[str] = None,
    skip_graph_load: bool = False,
    skip_cat5: bool = True,
):
    """Evaluate TRACE with graph-augmented retrieval.

    skip_graph_load=True activates "TRACElite" mode: no event graph is loaded,
    BFS / Phase 2 are skipped, and answers come from L1 hybrid context (A-Mem
    notes + neighborhood expansion) + L3 enhanced QA prompt + TokenOpt
    truncation. Used to evaluate the Table A "w/o (Graph + Hier + D)" row as a
    standalone deployable system rather than an ablation cell.

    skip_cat5 defaults to True (post-2026-05-24): cat 5 (Adversarial) questions
    are excluded from the question loop entirely. Cat 5 is already excluded from
    overall_llm_score and weighted F1 aggregation in `utils.aggregate_metrics`,
    so this only saves the per-question abstain cost (~50 tokens/Q + 1 LLM
    call). Output result.json will have category_5 absent from
    `aggregate_metrics`. Pass `--include_cat5` to opt back in for
    reproductions of pre-2026-05-22 numbers.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    safe_model = model.replace("/", "_")

    # Setup logging
    log_filename = f"eval_locomo_{safe_model}_{run_label}_{timestamp}.log"
    log_path = os.path.join(os.path.dirname(__file__), "logs", log_filename)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    eval_logger = logging.getLogger('eval_locomo')
    eval_logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    eval_logger.addHandler(fh)

    eval_logger.info(f"Dataset: {dataset_path}")
    eval_logger.info(f"Model: {model}, Backend: {backend}")
    eval_logger.info(f"Graph dir: {graph_dir}")
    eval_logger.info(f"Reasoner config: {reasoner_config}")
    eval_logger.info(f"Pipeline config: {pipeline_config}")
    eval_logger.info(f"Token opt config: {token_opt_config}")

    samples = load_locomo_dataset(dataset_path)
    eval_logger.info(f"Loaded {len(samples)} samples")

    # Determine which samples to process
    if sample_idx is not None:
        sample_indices = [sample_idx]
    else:
        sample_indices = list(range(len(samples)))

    memories_dir = memories_dir or os.path.join(
        os.path.dirname(__file__),
        f"cached_memories_{backend}_{safe_model}",
    )

    results = []
    all_metrics = []
    all_categories = []
    total_questions = 0
    category_counts = defaultdict(int)
    path_stats = {"with_paths": 0, "without_paths": 0, "fallback": 0}
    allow_categories = [1, 2, 3, 4] if skip_cat5 else [1, 2, 3, 4, 5]
    if skip_cat5:
        eval_logger.info("--skip_cat5: cat 5 (Adversarial) excluded from QA loop")

    for idx in sample_indices:
        if idx >= len(samples):
            continue
        sample = samples[idx]

        # Load graph (skipped in TRACElite mode — no graph file required)
        if skip_graph_load:
            graph = None
            eval_logger.info(f"Sample {idx}: TRACElite mode, skipping graph load")
        else:
            graph_file = os.path.join(graph_dir, f"event_graph_sample_{idx}.json")
            if os.path.exists(graph_file):
                graph = CausalGraph.load(graph_file)
                eval_logger.info(f"Sample {idx}: loaded graph with {graph.num_events()} events, {graph.num_edges()} edges")
            else:
                eval_logger.warning(f"Sample {idx}: no graph file at {graph_file}, using A-Mem fallback")
                graph = None

        # Create agent
        agent = TRACEGraphAgent(
            model, backend, retrieve_k, temperature_c5,
            api_key=api_key, api_base=api_base,
            skip_evolution=True,
            graph=graph,
            reasoner_config=reasoner_config,
            pipeline_config=pipeline_config,
            token_opt_config=token_opt_config,
            embedding_model=embedding_model,
            lite_mode=skip_graph_load,
        )

        # Load cached memories
        memory_cache_file = os.path.join(memories_dir, f"memory_cache_sample_{idx}.pkl")
        retriever_cache_file = os.path.join(memories_dir, f"retriever_cache_sample_{idx}.pkl")
        retriever_emb_file = os.path.join(memories_dir, f"retriever_cache_embeddings_sample_{idx}.npy")

        if os.path.exists(memory_cache_file) and os.path.exists(retriever_cache_file):
            with open(memory_cache_file, "rb") as f:
                agent.memory_system.memories = pickle.load(f)
            agent.memory_system.retriever = agent.memory_system.retriever.load(
                retriever_cache_file, retriever_emb_file
            )
            eval_logger.info(f"Sample {idx}: loaded {len(agent.memory_system.memories)} cached memories")

            # Re-init pipeline now that memory_system is loaded
            if graph is not None:
                agent.set_graph(graph, sample=sample)
                # Load session summaries for direct evidence
                if hasattr(sample, 'session_summary') and sample.session_summary:
                    agent.pipeline._session_summaries = sample.session_summary
            elif skip_graph_load:
                # TRACElite: build empty graph + init pipeline so format_context can run
                # with TokenOpt, while retrieve()/Phase 2 are bypassed in answer_question.
                agent.graph = CausalGraph()
                agent._init_pipeline()
        else:
            eval_logger.error(f"Sample {idx}: memory cache not found, skipping")
            continue

        # Process QA pairs
        for qa in tqdm(sample.qa, desc=f"Sample {idx}"):
            cat = int(qa.category)
            if cat not in allow_categories:
                continue

            try:
                prediction, prompt_used, raw_ctx = agent.answer_question(
                    qa.question, cat, qa.final_answer,
                )
            except Exception as e:
                logger.warning("answer_question crashed for '%s': %s - skipping", qa.question[:50], e)
                prediction, prompt_used, raw_ctx = "", "", "[ERROR]"

            if prediction is None:
                logger.warning("answer_question returned None for '%s' - treating as empty", qa.question[:50])
                prediction = ""

            raw_prediction = prediction  # Save raw LLM response before extraction
            prediction = parse_plain_text_answer(prediction)

            metrics = calculate_metrics(prediction, qa.final_answer) if qa.final_answer else {
                "exact_match": 0, "f1": 0.0, "rouge1_f": 0.0, "rouge2_f": 0.0,
                "rougeL_f": 0.0, "bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0,
                "bleu4": 0.0, "bert_f1": 0.0, "meteor": 0.0, "sbert_similarity": 0.0,
            }
            all_metrics.append(metrics)
            all_categories.append(cat)
            total_questions += 1
            category_counts[cat] += 1

            # Track path stats
            if hasattr(agent, 'pipeline') and agent.pipeline is not None:
                # Check last retrieval result via context content
                if raw_ctx and raw_ctx != "[NO_PATHS_FOUND]":
                    path_stats["with_paths"] += 1
                else:
                    path_stats["without_paths"] += 1
            else:
                path_stats["fallback"] += 1

            results.append({
                "sample_id": idx,
                "question": qa.question,
                "reference": qa.final_answer,
                "prediction": prediction,
                "raw_response": raw_prediction,
                "category": cat,
                "metrics": metrics,
                "path_explanation": raw_ctx[:2000] if raw_ctx else "",
            })

    # Aggregate metrics
    agg = aggregate_metrics(all_metrics, all_categories)

    output = {
        "model": model,
        "dataset": dataset_path,
        "backend": backend,
        "api_base": api_base,
        "run_label": run_label,
        "prompt_hash": compute_prompt_hash(),
        "timestamp": timestamp,
        "total_questions": total_questions,
        "category_distribution": dict(category_counts),
        "path_stats": path_stats,
        "reasoner_config": reasoner_config or {},
        "pipeline_config": pipeline_config or {},
        "token_opt_config": token_opt_config or {},
        "aggregate_metrics": agg,
        "individual_results": results,
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        eval_logger.info(f"Results saved to {output_path}")

    # Print summary
    eval_logger.info("=== TRACE Evaluation Summary ===")
    eval_logger.info(f"Total questions: {total_questions}")
    eval_logger.info(f"Path stats: {path_stats}")
    for key, val in agg.items():
        eval_logger.info(f"  {key}:")
        for metric_name, metric_val in val.items():
            if isinstance(metric_val, dict) and 'mean' in metric_val:
                eval_logger.info(f"    {metric_name}: mean={metric_val['mean']:.4f}")

    return output


def main():
    parser = argparse.ArgumentParser(description="TRACE graph-augmented evaluation")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model", type=str, default="openai/gpt-4o-mini")
    parser.add_argument("--backend", type=str, default="openai")
    parser.add_argument("--api_base", type=str, default="https://openrouter.ai/api/v1")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="data/locomo10.json")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override the directory auto-generated results land in "
                             "when --output is not given. Default: results/locomo.")
    parser.add_argument("--temperature_c5", type=float, default=0.5)
    parser.add_argument("--retrieve_k", type=int, default=10)
    parser.add_argument("--graph_dir", type=str, default="cached_graphs_openai_openai_gpt-4o-mini")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--run_label", type=str, default="locomo_main")
    parser.add_argument("--memories_dir", type=str, default=None,
                        help="Override default cached_memories_<backend>_<model> dir")
    parser.add_argument("--skip_graph_load", action="store_true",
                        help="TRACElite mode: skip event graph load and BFS/Phase 2; "
                             "use only L1 hybrid context + L3 prompt + TokenOpt.")
    parser.add_argument("--skip_cat5", dest="skip_cat5", action="store_true", default=True,
                        help="[default since 2026-05-24] Skip cat 5 (Adversarial) questions entirely. "
                             "Saves ~22%% wall-clock + tokens. Cat 5 is already excluded from "
                             "overall_llm_score and weighted F1, so this only removes the per-question "
                             "abstain cost (~50 tokens/Q + 1 LLM call). Kept as an explicit flag for "
                             "backward-compat with existing scripts (e.g. scripts/run_ablation_queue.py).")
    parser.add_argument("--include_cat5", dest="skip_cat5", action="store_false",
                        help="Opt back into running cat 5 (Adversarial) questions. Use only when "
                             "reproducing pre-2026-05-22 numbers — cat 5 is excluded from main metrics "
                             "aggregation regardless.")
    parser.add_argument("--judge_runs", type=int, default=3,
                        help="Number of LLM-judge runs after eval completes (default: 3, set 0 to skip).")
    parser.add_argument("--judge_model", type=str, default="openai/gpt-4o-mini",
                        help="Judge model (default: openai/gpt-4o-mini).")
    args = parser.parse_args()

    reasoner_config = {}
    pipeline_config = {}
    token_opt_config = {}

    if args.config:
        cfg = load_config(args.config)
        for key, val in cfg.items():
            if key == "reasoner":
                reasoner_config = val
            elif key == "pipeline":
                pipeline_config = val
            elif key == "token_opt":
                token_opt_config = val
            elif hasattr(args, key):
                setattr(args, key, val)

    # Expand sparse ablation configs to full config dicts using the
    # default values declared at the top of this module.
    reasoner_config = expand_reasoner_config(reasoner_config)
    pipeline_config = expand_pipeline_config(pipeline_config)
    token_opt_config = expand_token_opt_config(token_opt_config)

    if args.api_key is None:
        args.api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")

    dataset_path = os.path.join(os.path.dirname(__file__), args.dataset)

    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        # All LoCoMo eval outputs (main + ablations) land in results/locomo/.
        # --output_dir overrides the default if explicitly set.
        out_dir = args.output_dir or os.path.join("results", "locomo")
        out_dir = os.path.join(os.path.dirname(__file__), out_dir) if not os.path.isabs(out_dir) else out_dir
        os.makedirs(out_dir, exist_ok=True)
        if args.sample is not None:
            args.output = os.path.join(out_dir, f"trace_{args.run_label}_sample_{args.sample}_{ts}.json")
        else:
            args.output = os.path.join(out_dir, f"trace_{args.run_label}_{ts}.json")

    evaluate_trace_dataset(
        dataset_path=dataset_path,
        model=args.model,
        output_path=args.output,
        backend=args.backend,
        temperature_c5=args.temperature_c5,
        retrieve_k=args.retrieve_k,
        api_key=args.api_key,
        api_base=args.api_base,
        graph_dir=os.path.join(os.path.dirname(__file__), args.graph_dir),
        reasoner_config=reasoner_config,
        pipeline_config=pipeline_config,
        token_opt_config=token_opt_config,
        sample_idx=args.sample,
        run_label=args.run_label,
        embedding_model='all-MiniLM-L6-v2',
        memories_dir=(
            os.path.join(os.path.dirname(__file__), args.memories_dir)
            if args.memories_dir and not os.path.isabs(args.memories_dir)
            else args.memories_dir
        ),
        skip_graph_load=args.skip_graph_load,
        skip_cat5=args.skip_cat5,
    )

    # Post-eval LLM-judge (default 3 runs; set --judge_runs 0 to skip).
    if args.judge_runs > 0:
        import subprocess
        judge_cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "eval_llm_judge.py"),
            "--result", args.output,
            "--judge_runs", str(args.judge_runs),
            "--judge_model", args.judge_model,
        ]
        if args.api_base:
            judge_cmd += ["--api_base", args.api_base]
        logger.info("Auto-running LLM judge (%d runs) on %s", args.judge_runs, args.output)
        subprocess.run(judge_cmd, check=False)


if __name__ == "__main__":
    main()
