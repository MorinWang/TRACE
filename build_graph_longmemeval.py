"""Build a TRACE hierarchical graph for LongMemEval artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import pickle
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ingest_longmemeval import artifact_paths, default_memories_dir, safe_model_name
from memory_layer_robust import RobustLLMController
from trace.causal_graph import CausalGraph
from trace.dataset_adapter import LongMemEvalAdapter
from trace.event_extractor import EventExtractor, build_nickname_map
from trace.prompts.cross_note import PROMPT_VERSION as CROSS_NOTE_VERSION
from trace.prompts.extraction import PROMPT_VERSION as EXTRACTION_VERSION
from trace.prompts.update import PROMPT_VERSION as UPDATE_VERSION
from trace.topic_clusterer import TopicNode, cluster_sessions_llm
from trace.update_detector import UpdateDetector
from trace.validity_propagator import ValidityPropagator

logger = logging.getLogger("build_graph_longmemeval")


def load_config(config_path: Optional[str]) -> Dict:
    if not config_path or not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_prompt_hash() -> str:
    combined = f"{EXTRACTION_VERSION}|{CROSS_NOTE_VERSION}|{UPDATE_VERSION}|longmemeval_hierarchical"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def make_llm(config: Dict):
    api_key = config.get("api_key") or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    return RobustLLMController(
        backend=config.get("backend", "openai"),
        model=config.get("model", "openai/gpt-4o-mini"),
        api_key=api_key,
        api_base=config.get("api_base", "https://openrouter.ai/api/v1"),
    ).llm


class SessionSummaryNote:
    def __init__(self, note_id: str, content: str, timestamp: str):
        self.id = note_id
        self.content = content
        self.context = "LongMemEval session summary"
        self.timestamp = timestamp
        self.keywords = []


def load_artifacts(
    memories_dir: Path,
    summaries_dir: Path,
    tag: str,
):
    paths = artifact_paths(memories_dir, tag)
    required = [
        paths["memory_cache"],
        paths["session_note_map"],
        paths["session_index"],
        summaries_dir / f"longmemeval_session_summaries_{tag}.json",
        summaries_dir / f"longmemeval_session_metadata_{tag}.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing LongMemEval artifacts. Run ingest_longmemeval.py first:\n" + "\n".join(missing))

    with paths["memory_cache"].open("rb") as f:
        memories = pickle.load(f)
    return {
        "memories": memories,
        "session_note_map": load_json(paths["session_note_map"]),
        "session_index": load_json(paths["session_index"]),
        "summaries": load_json(summaries_dir / f"longmemeval_session_summaries_{tag}.json"),
        "metadata": load_json(summaries_dir / f"longmemeval_session_metadata_{tag}.json"),
    }


def build_topics(
    sessions: Dict[str, dict],
    llm,
    tag: str,
    llm_max_sessions: int = 120,
    chunk_size: int = 50,
) -> List[TopicNode]:
    """Create topic nodes without sending huge global prompts."""
    if not sessions:
        return []
    if len(sessions) <= llm_max_sessions:
        return cluster_sessions_llm(sessions, llm, tag)

    sorted_ids = sorted(sessions.keys(), key=lambda sid: sessions[sid].get("date_time", ""))
    topics = []
    for offset in range(0, len(sorted_ids), chunk_size):
        chunk_ids = sorted_ids[offset:offset + chunk_size]
        topics.append(TopicNode(
            topic_id=f"topic_{tag}_{offset // chunk_size}",
            label=f"Temporal window {offset // chunk_size + 1}",
            session_ids=chunk_ids,
            description="Chronological LongMemEval session window used to bound cross-session edge inference.",
        ))
    logger.info("Large LongMemEval subset: using %d chronological topic windows", len(topics))
    return topics


def build_graph(
    config: Dict,
    memories_dir: Path,
    summaries_dir: Path,
    output_dir: Path,
    tag: str,
    force: bool = False,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_file = output_dir / f"event_graph_longmemeval_{tag}.json"
    log_file = output_dir / f"extraction_log_longmemeval_{tag}.jsonl"
    prompt_hash = compute_prompt_hash()

    if graph_file.exists() and not force:
        try:
            meta = load_json(graph_file).get("metadata", {})
            if meta.get("prompt_hash") == prompt_hash:
                graph = CausalGraph.load(str(graph_file))
                logger.info("%s cache valid: %s", tag, graph.summary())
                return graph
        except Exception:
            pass

    artifacts = load_artifacts(memories_dir, summaries_dir, tag)
    memories = artifacts["memories"]
    session_index = artifacts["session_index"]
    summaries = artifacts["summaries"]

    llm = make_llm(config)
    extractor = EventExtractor(
        llm=llm,
        extraction_temperature=config.get("extraction_temperature", 0.0),
        cross_note_temperature=config.get("cross_note_temperature", 0.1),
        nickname_map=build_nickname_map(memories),
    )
    update_detector = UpdateDetector(llm=llm, temperature=config.get("update_detection_temperature", 0.0))
    propagator = ValidityPropagator()
    graph = CausalGraph()

    sessions: Dict[str, dict] = {}
    for rec in session_index:
        key = rec["session_key"]
        node_id = rec["session_node_id"]
        sessions[node_id] = {
            "session_id": node_id,
            "original_session_id": rec["session_id"],
            "session_key": key,
            "summary": summaries.get(key, ""),
            "date_time": rec.get("date_time", ""),
            "note_ids": rec.get("note_ids", []),
            "event_ids": [],
        }

    topics = build_topics(
        sessions,
        llm,
        tag,
        llm_max_sessions=config.get("longmemeval_topic_llm_max_sessions", 120),
        chunk_size=config.get("longmemeval_topic_chunk_size", 50),
    )
    session_to_topic = {}
    topic_to_sessions = {}
    for topic in topics:
        topic_to_sessions[topic.topic_id] = sorted(
            topic.session_ids,
            key=lambda sid: sessions.get(sid, {}).get("date_time", ""),
        )
        for sid in topic.session_ids:
            session_to_topic[sid] = topic.topic_id

    stats = {
        "total_sessions": len(sessions),
        "sessions_with_events": 0,
        "total_events_extracted": 0,
        "intra_session_edges": 0,
        "cross_note_edges": 0,
        "auto_temporal_edges": 0,
        "update_edges": 0,
        "contradiction_edges": 0,
        "topics": len(topics),
    }
    start_time = time.time()
    event_to_session = {}
    session_events = defaultdict(list)
    all_new_events = []

    with log_file.open("w", encoding="utf-8") as extraction_log:
        for rec in tqdm(session_index, desc=f"{tag} Phase 1"):
            key = rec["session_key"]
            node_id = rec["session_node_id"]
            summary = summaries.get(key, "")
            if not summary:
                continue

            note = SessionSummaryNote(key, summary, rec.get("date_time", ""))
            try:
                events, intra_edges = extractor.extract_from_note(note)
            except Exception as exc:
                logger.error("Extraction failed for %s: %s", key, exc)
                continue

            extraction_log.write(json.dumps({
                "session_key": key,
                "session_node_id": node_id,
                "original_session_id": rec["session_id"],
                "summary_len": len(summary),
                "num_events": len(events),
                "num_intra_edges": len(intra_edges),
            }, ensure_ascii=False) + "\n")

            if not events:
                continue

            stats["sessions_with_events"] += 1
            stats["total_events_extracted"] += len(events)
            stats["intra_session_edges"] += len(intra_edges)

            note_ids = rec.get("note_ids", [])
            for event in events:
                event.source_note_ids = note_ids
                if not event.time_anchor or event.time_anchor == "unknown":
                    event.time_anchor = rec.get("date_time", "")
                graph.add_event(event)
                all_new_events.append(event)
                event_to_session[event.event_id] = node_id
                session_events[node_id].append(event.event_id)
            for edge in intra_edges:
                graph.add_edge(edge)

    logger.info("Phase 1 done: %d events", graph.num_events())

    topic_event_ids = defaultdict(set)
    for eid, sid in event_to_session.items():
        topic_event_ids[session_to_topic.get(sid)].add(eid)

    topic_session_pos = {}
    for tid, sids in topic_to_sessions.items():
        for pos, sid in enumerate(sids):
            topic_session_pos[sid] = (tid, pos)

    cross_session_window = int(config.get("longmemeval_cross_session_window", 3))

    def allowed_event_ids_for_session(session_id: str) -> Optional[Set[str]]:
        topic_pos = topic_session_pos.get(session_id)
        if topic_pos is None:
            return None
        tid, pos = topic_pos
        ordered = topic_to_sessions.get(tid, [])
        nearby = ordered[max(0, pos - cross_session_window):pos + cross_session_window + 1]
        allowed = set()
        for sid in nearby:
            allowed.update(session_events.get(sid, []))
        return allowed

    sorted_session_ids = sorted(sessions.keys(), key=lambda sid: sessions[sid].get("date_time", ""))
    for sid in tqdm(sorted_session_ids, desc=f"{tag} Phase 2"):
        eids = session_events.get(sid, [])
        if not eids:
            continue
        events_in_session = [graph.get_event(eid) for eid in eids if graph.get_event(eid)]
        cross_edges = extractor.infer_cross_note_edges(
            events_in_session,
            graph,
            min_jaccard=config.get("cross_note_min_jaccard", 0.5),
            max_candidates=config.get("cross_note_max_candidates", 5),
            allowed_event_ids=allowed_event_ids_for_session(sid),
        )
        for edge in cross_edges:
            if edge.edge_type == "temporal_before":
                stats["auto_temporal_edges"] += 1
            else:
                stats["cross_note_edges"] += 1
            graph.add_edge(edge)

    logger.info("Phase 2 done: %d cross, %d temporal", stats["cross_note_edges"], stats["auto_temporal_edges"])

    if not config.get("longmemeval_skip_update_detection", False):
        for event in tqdm(all_new_events, desc=f"{tag} Phase 3"):
            sid = event_to_session.get(event.event_id)
            same_session_ids = set(session_events.get(sid, [])) if sid else {event.event_id}
            update_edges = update_detector.detect(event, graph, exclude_ids=same_session_ids)
            for edge in update_edges:
                if edge.edge_type == "updates":
                    stats["update_edges"] += 1
                elif edge.edge_type == "contradicts":
                    stats["contradiction_edges"] += 1
                graph.add_edge(edge)
            propagator.process_edges(update_edges, graph)

    for sid, sdata in sessions.items():
        sdata["event_ids"] = session_events.get(sid, [])

    prev_sid = None
    for sid in sorted_session_ids:
        if not sessions[sid]["event_ids"]:
            continue
        graph.add_session(sessions[sid])
        if prev_sid is not None:
            graph.graph.add_edge(prev_sid, sid, edge_type="temporal_before", confidence=1.0)
        prev_sid = sid

    for topic in topics:
        graph.add_topic(topic.to_dict())

    elapsed = time.time() - start_time
    metadata = {
        "builder": "longmemeval_hierarchical",
        "tag": tag,
        "prompt_hash": prompt_hash,
        "prompt_versions": {
            "extraction": EXTRACTION_VERSION,
            "cross_note": CROSS_NOTE_VERSION,
            "update": UPDATE_VERSION,
        },
        "config": config,
        "built_at": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "stats": stats,
    }
    graph.save(str(graph_file), metadata=metadata)
    print(graph.summary())
    print(json.dumps(stats, indent=2))
    return graph


def main():
    parser = argparse.ArgumentParser(description="Build TRACE graph for LongMemEval")
    parser.add_argument("--config", default="configs/longmemeval_main.json")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--memories-dir", default=None)
    parser.add_argument("--summaries-dir", default="cached_summaries")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")

    config = load_config(args.config)
    model = config.get("model", "openai/gpt-4o-mini")
    backend = config.get("backend", "openai")
    tag = LongMemEvalAdapter.subset_tag(args.sample, args.limit)
    memories_dir = Path(args.memories_dir or default_memories_dir(backend, model))
    output_dir = Path(args.output_dir or f"cached_graphs_longmemeval_{backend}_{safe_model_name(model)}")

    build_graph(
        config=config,
        memories_dir=memories_dir,
        summaries_dir=Path(args.summaries_dir),
        output_dir=output_dir,
        tag=tag,
        force=args.force,
    )


if __name__ == "__main__":
    main()
