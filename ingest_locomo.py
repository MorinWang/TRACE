"""Build LoCoMo A-Mem memory caches for TRACE evaluation.

Run this once before `python run_eval.py`. It populates
``cached_memories_<backend>_<safe_model>/`` with per-sample memory and
retriever caches that ``eval_locomo.py`` and ``build_graph_locomo.py``
both consume read-only.

Usage:

    python ingest_locomo.py --config configs/locomo_main.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Optional

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trace.agent import TRACEAgent
from load_dataset import load_locomo_dataset

logger = logging.getLogger("ingest_locomo")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_config(path: Optional[str]) -> dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_model(model: str) -> str:
    return model.replace("/", "_")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LoCoMo memory caches for TRACE")
    parser.add_argument("--config", type=str, default=None,
                        help="Experiment config JSON (used for model/backend/dataset paths)")
    parser.add_argument("--dataset", type=str, default="data/locomo10.json")
    parser.add_argument("--model", type=str, default="openai/gpt-4o-mini")
    parser.add_argument("--backend", type=str, default="openai")
    parser.add_argument("--api_base", type=str, default="https://openrouter.ai/api/v1")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--retrieve_k", type=int, default=10)
    parser.add_argument("--temperature_c5", type=float, default=0.5)
    parser.add_argument("--memories_dir", type=str, default=None,
                        help="Override default cached_memories_<backend>_<model> dir")
    parser.add_argument("--sample", type=int, default=None,
                        help="Build cache for a single sample only (0-9). Default: all.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    for key, val in cfg.items():
        if hasattr(args, key) and getattr(args, key) in (None, parser.get_default(key)):
            setattr(args, key, val)

    if args.api_key is None:
        args.api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")

    here = Path(__file__).resolve().parent
    dataset_path = here / args.dataset
    memories_dir = (
        Path(args.memories_dir) if args.memories_dir
        else here / f"cached_memories_{args.backend}_{safe_model(args.model)}"
    )
    memories_dir.mkdir(parents=True, exist_ok=True)

    samples = load_locomo_dataset(str(dataset_path))
    logger.info(f"Loaded {len(samples)} samples from {dataset_path}")
    logger.info(f"Memory cache directory: {memories_dir}")

    indices = [args.sample] if args.sample is not None else list(range(len(samples)))

    for idx in indices:
        sample = samples[idx]
        memory_cache_file = memories_dir / f"memory_cache_sample_{idx}.pkl"
        retriever_cache_file = memories_dir / f"retriever_cache_sample_{idx}.pkl"
        retriever_emb_file = memories_dir / f"retriever_cache_embeddings_sample_{idx}.npy"

        if memory_cache_file.exists() and retriever_cache_file.exists():
            logger.info(f"Sample {idx}: cache exists, skipping")
            continue

        logger.info(f"Sample {idx}: ingesting {sum(len(s.turns) for s in sample.conversation.sessions.values())} turns")
        agent = TRACEAgent(
            model=args.model,
            backend=args.backend,
            retrieve_k=args.retrieve_k,
            temperature_c5=args.temperature_c5,
            api_key=args.api_key,
            api_base=args.api_base,
        )

        for _, session in sample.conversation.sessions.items():
            for turn in tqdm(session.turns, desc=f"sample {idx} session", leave=False):
                msg = f"Speaker {turn.speaker} says: {turn.text}"
                agent.add_memory(msg, time=session.date_time)

        with open(memory_cache_file, "wb") as f:
            pickle.dump(agent.memory_system.memories, f)
        agent.memory_system.retriever.save(str(retriever_cache_file), str(retriever_emb_file))
        logger.info(f"Sample {idx}: cached {len(agent.memory_system.memories)} memories -> {memory_cache_file.name}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
