"""Prepare LongMemEval artifacts for TRACE.

Step 0: de-duplicate and globally sort sessions.
Step 1: ingest turns into A-Mem-compatible memory caches.
Step 2: generate/cache LoCoMo-style session summaries.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, Optional

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory_layer_robust import RobustAgenticMemorySystem, RobustLLMController
from trace.dataset_adapter import LongMemEvalAdapter, format_turn_for_memory
from trace.session_summarizer import SessionSummarizer

logger = logging.getLogger("ingest_longmemeval")


DEFAULT_DATASET = "data/longmemeval_s_cleaned.json"


def load_config(config_path: Optional[str]) -> Dict:
    if not config_path:
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_model_name(model: str) -> str:
    return model.replace("/", "_")


def default_memories_dir(backend: str, model: str) -> str:
    return f"cached_memories_longmemeval_{backend}_{safe_model_name(model)}"


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def artifact_paths(memories_dir: Path, tag: str) -> Dict[str, Path]:
    return {
        "memory_cache": memories_dir / f"memory_cache_{tag}.pkl",
        "retriever_cache": memories_dir / f"retriever_cache_{tag}.pkl",
        "retriever_embeddings": memories_dir / f"retriever_cache_embeddings_{tag}.npy",
        "session_note_map": memories_dir / f"longmemeval_session_note_map_{tag}.json",
        "session_index": memories_dir / f"longmemeval_session_index_{tag}.json",
    }


def ingest_memories(
    adapter: LongMemEvalAdapter,
    config: Dict,
    memories_dir: Path,
    tag: str,
    force: bool = False,
):
    paths = artifact_paths(memories_dir, tag)
    if paths["memory_cache"].exists() and paths["session_note_map"].exists() and not force:
        logger.info("Memory artifacts already exist for %s; use --force to rebuild", tag)
        return

    memories_dir.mkdir(parents=True, exist_ok=True)
    api_key = config.get("api_key") or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    memory_system = RobustAgenticMemorySystem(
        model_name=config.get("embedding_model", "all-MiniLM-L6-v2"),
        llm_backend=config.get("backend", "openai"),
        llm_model=config.get("model", "openai/gpt-4o-mini"),
        api_key=api_key,
        api_base=config.get("api_base", "https://openrouter.ai/api/v1"),
        skip_evolution=True,  # LongMemEval default: store notes without A-Mem evolution
    )

    session_note_map = defaultdict(list)
    session_index = []

    for sid, session in tqdm(list(adapter.iter_sorted_sessions()), desc=f"Ingest {tag}"):
        note_ids = []
        for turn in session.get("turns", []):
            content = format_turn_for_memory(turn)
            note_id = memory_system.add_note(content, time=session.get("date", ""))
            note_ids.append(note_id)
            session_note_map[sid].append(note_id)
        session_index.append({
            "session_id": sid,
            "session_key": adapter.summary_key_for_sid(sid),
            "session_node_id": adapter.session_node_id_for_sid(sid),
            "date_time": session.get("date", ""),
            "note_ids": note_ids,
            "num_turns": len(session.get("turns", [])),
        })

    with paths["memory_cache"].open("wb") as f:
        pickle.dump(memory_system.memories, f)
    memory_system.retriever.save(str(paths["retriever_cache"]), str(paths["retriever_embeddings"]))
    write_json(paths["session_note_map"], dict(session_note_map))
    write_json(paths["session_index"], session_index)
    logger.info("Saved %d memories for %s", len(memory_system.memories), tag)


def build_summaries(
    adapter: LongMemEvalAdapter,
    summaries_dir: Path,
    tag: str,
    batch_size: int = 20,
):
    if adapter.summarizer is not None:
        adapter.summarizer.batch_summarize(adapter.unique_sessions, batch_size=batch_size)

    by_key = OrderedDict()
    metadata = OrderedDict()
    for sid, session in tqdm(list(adapter.iter_sorted_sessions()), desc=f"Summaries {tag}"):
        key = adapter.summary_key_for_sid(sid)
        by_key[key] = adapter._summary_for_session(sid, session)
        metadata[key] = adapter._metadata_for_sid(sid)

    summaries_dir.mkdir(parents=True, exist_ok=True)
    write_json(summaries_dir / f"longmemeval_session_summaries_{tag}.json", by_key)
    write_json(summaries_dir / f"longmemeval_session_metadata_{tag}.json", metadata)
    logger.info("Saved %d session summaries for %s", len(by_key), tag)


def main():
    parser = argparse.ArgumentParser(description="Ingest LongMemEval for TRACE")
    parser.add_argument("--config", default="configs/longmemeval_main.json")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--sample", type=int, default=None, help="Only use one LongMemEval question's haystack sessions")
    parser.add_argument("--limit", type=int, default=None, help="Use the first N LongMemEval questions")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--memories-dir", default=None)
    parser.add_argument("--summaries-dir", default="cached_summaries")
    parser.add_argument("--summary-batch-size", type=int, default=20)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")

    config = load_config(args.config if os.path.exists(args.config) else None)
    dataset_path = args.dataset or config.get("dataset", DEFAULT_DATASET)
    model = config.get("model", "openai/gpt-4o-mini")
    backend = config.get("backend", "openai")
    tag = LongMemEvalAdapter.subset_tag(args.sample, args.limit)
    memories_dir = Path(args.memories_dir or default_memories_dir(backend, model))
    summaries_dir = Path(args.summaries_dir)

    api_key = config.get("api_key") or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    llm = RobustLLMController(
        backend=config.get("backend", "openai"),
        model=config.get("model", "openai/gpt-4o-mini"),
        api_key=api_key,
        api_base=config.get("api_base", "https://openrouter.ai/api/v1"),
    ).llm
    summarizer = SessionSummarizer(
        llm=llm,
        cache_path=str(summaries_dir / "longmemeval_session_summaries_cache.json"),
    )
    adapter = LongMemEvalAdapter(
        dataset_path=dataset_path,
        summarizer=summarizer,
        sample_idx=args.sample,
        limit=args.limit,
    )

    logger.info("Subset %s: %d questions, %d unique sessions", tag, adapter.get_num_samples(), len(adapter.unique_sessions))

    ingest_memories(
        adapter=adapter,
        config=config,
        memories_dir=memories_dir,
        tag=tag,
        force=args.force,
    )
    build_summaries(adapter, summaries_dir, tag, batch_size=args.summary_batch_size)


if __name__ == "__main__":
    main()

