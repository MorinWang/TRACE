"""Build the TRACE topology-aware hierarchical hypergraph for LoCoMo.

Builds a 3-level nested hypergraph (Event -> Session -> Topic) with
topic-guided cross-note edge inference: cross-note edges are only inferred
between events in the same topic, reducing noise and improving edge quality.

Usage:
    python build_graph_locomo.py --sample 0
    python build_graph_locomo.py --all
    python build_graph_locomo.py --sample 0 --force
"""

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

from memory_layer_robust import RobustLLMController
from load_dataset import load_locomo_dataset
from trace.parsing_utils import extract_session_num
from trace.event_schema import EventNode
from trace.causal_graph import CausalGraph
from trace.event_extractor import EventExtractor, build_nickname_map
from trace.update_detector import UpdateDetector
from trace.validity_propagator import ValidityPropagator
from trace.topic_clusterer import cluster_sessions_llm
from trace.prompts.extraction import PROMPT_VERSION as EXTRACTION_VERSION
from trace.prompts.cross_note import PROMPT_VERSION as CROSS_NOTE_VERSION
from trace.prompts.update import PROMPT_VERSION as UPDATE_VERSION

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("build_graph_hier")


def compute_prompt_hash() -> str:
    combined = f"{EXTRACTION_VERSION}|{CROSS_NOTE_VERSION}|{UPDATE_VERSION}|hierarchical"
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return json.load(f)


def load_cached_memories(sample_idx: int, memories_dir: str) -> Optional[dict]:
    cache_file = os.path.join(memories_dir, f"memory_cache_sample_{sample_idx}.pkl")
    if not os.path.exists(cache_file):
        logger.error(f"Memory cache not found: {cache_file}")
        return None
    with open(cache_file, "rb") as f:
        memories = pickle.load(f)
    logger.info(f"Loaded {len(memories)} cached memories from {cache_file}")
    return memories


def sort_notes_by_timestamp(memories: dict) -> list:
    notes = list(memories.values())
    notes.sort(key=lambda n: getattr(n, 'timestamp', '') or '')
    return notes


def group_notes_by_session(memories: dict) -> Dict[str, List]:
    """Group notes by timestamp (= session). Returns {timestamp: [note_objects]}."""
    groups = defaultdict(list)
    for note in memories.values():
        ts = getattr(note, 'timestamp', '') or ''
        if ts:
            groups[ts].append(note)
    return dict(groups)


def build_session_mapping(memories, session_summaries, sample):
    """Build session metadata: align timestamps with session summaries.

    Returns:
        sessions: dict of session_id -> {summary, date_time, note_ids, notes}
        note_to_session: dict of note_id -> session_id
    """
    ts_groups = defaultdict(list)
    for note_id, note in memories.items():
        ts = getattr(note, 'timestamp', '') or ''
        if ts:
            ts_groups[ts].append(note_id)

    sorted_timestamps = sorted(ts_groups.keys())
    sorted_keys = sorted(session_summaries.keys(), key=extract_session_num)
    n_sessions = min(len(sorted_timestamps), len(sorted_keys))

    if n_sessions == 0:
        return {}, {}

    sessions = {}
    note_to_session = {}
    conv_sessions = getattr(sample.conversation, 'sessions', {}) if hasattr(sample, 'conversation') else {}

    for i in range(n_sessions):
        ts = sorted_timestamps[i]
        sess_key = sorted_keys[i]
        sess_num = extract_session_num(sess_key)
        session_id = f"sess_{int(getattr(sample, 'sample_id', '0').replace('conv-', ''))}_{sess_num}"

        date_time = ts
        if conv_sessions and sess_num in conv_sessions:
            date_time = getattr(conv_sessions[sess_num], 'date_time', ts)

        note_ids = ts_groups[ts]
        sessions[session_id] = {
            "session_id": session_id,
            "session_key": sess_key,
            "summary": session_summaries[sess_key],
            "date_time": date_time,
            "note_ids": note_ids,
            "event_ids": [],  # filled during Phase 1
        }
        for nid in note_ids:
            note_to_session[nid] = session_id

    logger.info(f"Built {len(sessions)} session mappings")
    return sessions, note_to_session


def build_graph_for_sample(
    sample_idx: int,
    config: dict,
    memories_dir: str,
    output_dir: str,
    dataset_path: str,
    force: bool = False,
):
    """Build topology-aware hierarchical hypergraph for one sample."""
    graph_file = os.path.join(output_dir, f"event_graph_sample_{sample_idx}.json")
    log_file = os.path.join(output_dir, f"extraction_log_sample_{sample_idx}.jsonl")
    prompt_hash = compute_prompt_hash()

    # Cache check
    if os.path.exists(graph_file) and not force:
        try:
            with open(graph_file, "r") as f:
                meta = json.load(f).get("metadata", {})
            if meta.get("prompt_hash") == prompt_hash:
                logger.info(f"Sample {sample_idx}: cache valid, skipping")
                graph = CausalGraph.load(graph_file)
                print(f"\nSample {sample_idx} (cached):\n{graph.summary()}")
                return graph
        except (json.JSONDecodeError, KeyError):
            pass

    # Load memories
    memories = load_cached_memories(sample_idx, memories_dir)
    if memories is None:
        return None

    # Load dataset for session summaries
    dataset = load_locomo_dataset(dataset_path)
    sample = dataset[sample_idx]
    session_summaries = getattr(sample, 'session_summary', {}) or {}
    if not session_summaries:
        logger.warning(f"Sample {sample_idx}: no session summaries, falling back to flat build")

    # Init components
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    llm_ctrl = RobustLLMController(
        backend=config.get("backend", "openai"),
        model=config.get("model", "openai/gpt-4o-mini"),
        api_key=api_key,
        api_base=config.get("api_base", "https://openrouter.ai/api/v1"),
    )
    llm = llm_ctrl.llm

    nickname_map = build_nickname_map(memories)
    extractor = EventExtractor(
        llm=llm,
        extraction_temperature=config.get("extraction_temperature", 0.0),
        cross_note_temperature=config.get("cross_note_temperature", 0.1),
        nickname_map=nickname_map,
    )
    graph = CausalGraph()
    update_detector = UpdateDetector(llm=llm, temperature=config.get("update_detection_temperature", 0.0))
    propagator = ValidityPropagator()

    # ================================================================
    # Phase 0: Preparation — session mapping + topic clustering
    # ================================================================
    phase_timings = {}
    overall_start = time.time()
    phase0_start = overall_start
    sessions, note_to_session = build_session_mapping(memories, session_summaries, sample)

    # Topic clustering (1 LLM call)
    temp_session_data = {sid: sdata for sid, sdata in sessions.items()}
    topics = cluster_sessions_llm(temp_session_data, llm, sample_idx)

    # Build topic → session → event mappings
    session_to_topic = {}
    for topic in topics:
        for sid in topic.session_ids:
            session_to_topic[sid] = topic.topic_id

    phase_timings["phase0_preparation_s"] = round(time.time() - phase0_start, 2)
    logger.info(f"Phase 0: {len(sessions)} sessions, {len(topics)} topics ({phase_timings['phase0_preparation_s']:.1f}s)")
    for t in topics:
        logger.info(f"  Topic '{t.label}': {len(t.session_ids)} sessions")

    # ================================================================
    # Phase 1: Event extraction (from session summaries)
    # ================================================================
    os.makedirs(output_dir, exist_ok=True)
    extraction_log = open(log_file, "w", encoding="utf-8")

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

    # Track event → session mapping
    event_to_session: Dict[str, str] = {}
    session_events: Dict[str, List[str]] = defaultdict(list)
    all_new_events: List[EventNode] = []

    start_time = time.time()
    phase1_start = start_time

    sorted_sess_keys = sorted(session_summaries.keys(), key=extract_session_num)

    for sess_key in tqdm(sorted_sess_keys, desc=f"Sample {sample_idx} Phase 1"):
        summary_text = session_summaries[sess_key]
        sess_num = extract_session_num(sess_key)

        # Find matching session_id
        session_id = None
        for sid, sdata in sessions.items():
            if sdata.get("session_key") == sess_key:
                session_id = sid
                break
        if session_id is None:
            continue

        sdata = sessions[session_id]
        note_ids = sdata.get("note_ids", [])
        session_date = sdata.get("date_time", "")

        # Create a fake "note" for the extractor interface
        class SessionNote:
            pass
        note = SessionNote()
        note.content = summary_text
        note.context = f"Session {sess_num} conversation summary"
        note.timestamp = session_date
        note.keywords = []
        note.id = f"session_{sample_idx}_{sess_num}"

        try:
            events, intra_edges = extractor.extract_from_note(note)
        except Exception as e:
            logger.error(f"Extraction failed for {sess_key}: {e}")
            continue

        log_entry = {
            "session_key": sess_key,
            "session_id": session_id,
            "summary_len": len(summary_text),
            "num_events": len(events),
            "num_intra_edges": len(intra_edges),
        }
        extraction_log.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        if not events:
            continue

        stats["sessions_with_events"] += 1
        stats["total_events_extracted"] += len(events)
        stats["intra_session_edges"] += len(intra_edges)

        # Set source_note_ids to all notes in this session (broad mapping for context)
        for event in events:
            event.source_note_ids = note_ids
            if not event.time_anchor or event.time_anchor == "unknown":
                event.time_anchor = session_date
            graph.add_event(event)
            all_new_events.append(event)
            event_to_session[event.event_id] = session_id
            session_events[session_id].append(event.event_id)

        for edge in intra_edges:
            graph.add_edge(edge)

    extraction_log.close()
    phase_timings["phase1_extraction_s"] = round(time.time() - phase1_start, 2)
    logger.info(f"Phase 1 done: {graph.num_events()} events, {stats['intra_session_edges']} intra edges ({phase_timings['phase1_extraction_s']:.1f}s)")

    # ================================================================
    # Phase 2: Cross-session edges (topic-guided)
    # ================================================================
    phase2_start = time.time()
    # Build topic → event_ids mapping
    topic_event_ids: Dict[str, Set[str]] = defaultdict(set)
    for eid, sid in event_to_session.items():
        tid = session_to_topic.get(sid)
        if tid:
            topic_event_ids[tid].add(eid)

    # Process sessions chronologically for cross-note edges
    sorted_session_ids = sorted(sessions.keys(), key=lambda s: sessions[s]["date_time"])

    for sid in tqdm(sorted_session_ids, desc=f"Sample {sample_idx} Phase 2"):
        eids = session_events.get(sid, [])
        if not eids:
            continue

        events_in_session = [graph.get_event(eid) for eid in eids if graph.get_event(eid)]
        if not events_in_session:
            continue

        # Get same-topic event IDs for topic-guided filtering
        tid = session_to_topic.get(sid)
        allowed_ids = topic_event_ids.get(tid) if tid else None

        cross_edges = extractor.infer_cross_note_edges(
            events_in_session, graph,
            min_jaccard=config.get("cross_note_min_jaccard", 0.5),
            max_candidates=config.get("cross_note_max_candidates", 5),
            allowed_event_ids=allowed_ids,
        )
        for edge in cross_edges:
            if edge.edge_type == "temporal_before":
                stats["auto_temporal_edges"] += 1
            else:
                stats["cross_note_edges"] += 1
            graph.add_edge(edge)

    phase_timings["phase2_cross_session_s"] = round(time.time() - phase2_start, 2)
    logger.info(f"Phase 2 done: {stats['cross_note_edges']} cross-note, {stats['auto_temporal_edges']} temporal ({phase_timings['phase2_cross_session_s']:.1f}s)")

    # ================================================================
    # Phase 3: Evolution — update detection + validity propagation
    # ================================================================
    phase3_start = time.time()
    # Group events by session for proper exclude_ids
    for event in tqdm(all_new_events, desc=f"Sample {sample_idx} Phase 3"):
        # Exclude events from the same session (not all events)
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

    phase_timings["phase3_update_detection_s"] = round(time.time() - phase3_start, 2)
    logger.info(f"Phase 3 done: {stats['update_edges']} updates, {stats['contradiction_edges']} contradictions ({phase_timings['phase3_update_detection_s']:.1f}s)")

    # ================================================================
    # Phase 4: Hierarchy injection — session + topic nodes
    # ================================================================
    phase4_start = time.time()
    # Update session event_ids with actual extracted events
    for sid, sdata in sessions.items():
        sdata["event_ids"] = session_events.get(sid, [])

    # Inject session nodes
    prev_sid = None
    for sid in sorted_session_ids:
        sdata = sessions[sid]
        if not sdata["event_ids"]:
            continue
        graph.add_session(sdata)
        if prev_sid is not None:
            graph.graph.add_edge(prev_sid, sid, edge_type="temporal_before", confidence=1.0)
        prev_sid = sid

    # Inject topic nodes
    for topic in topics:
        graph.add_topic(topic.to_dict())

    phase_timings["phase4_hierarchy_injection_s"] = round(time.time() - phase4_start, 2)
    logger.info(f"Phase 4 done: {len(graph._sessions)} sessions, {len(graph._topics)} topics ({phase_timings['phase4_hierarchy_injection_s']:.1f}s)")

    # ================================================================
    # Phase 5: Save
    # ================================================================
    phase5_start = time.time()
    elapsed = time.time() - overall_start
    metadata = {
        "sample_idx": sample_idx,
        "prompt_hash": prompt_hash,
        "builder": "hierarchical_session_topic",
        "prompt_versions": {
            "extraction": EXTRACTION_VERSION,
            "cross_note": CROSS_NOTE_VERSION,
            "update": UPDATE_VERSION,
        },
        "config": config,
        "built_at": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "phase_timings": phase_timings,
        "stats": stats,
    }
    graph.save(graph_file, metadata=metadata)
    # Phase 5 (save) timing is logged but not persisted to graph metadata
    # (the metadata snapshot is taken before save completes; circular dependency).
    phase_timings["phase5_save_s"] = round(time.time() - phase5_start, 2)
    logger.info(f"Phase 5 done: graph saved ({phase_timings['phase5_save_s']:.1f}s)")
    logger.info(f"Phase breakdown: {phase_timings}")

    print(f"\nSample {sample_idx} ({elapsed:.1f}s):")
    print(graph.summary())
    print(f"Stats: {json.dumps(stats, indent=2)}")

    return graph


def main():
    parser = argparse.ArgumentParser(description="Build topology-aware hierarchical hypergraph")
    parser.add_argument("--config", default="configs/locomo_main.json")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--memories-dir", default="cached_memories_openai_openai_gpt-4o-mini")
    parser.add_argument("--output-dir", default="cached_graphs_openai_openai_gpt-4o-mini")
    parser.add_argument("--dataset", default="data/locomo10.json")
    args = parser.parse_args()

    config = load_config(args.config)
    logger.info(f"Config: {args.config}")
    logger.info(f"Prompt hash: {compute_prompt_hash()}")

    if args.sample is not None:
        build_graph_for_sample(
            args.sample, config, args.memories_dir, args.output_dir,
            args.dataset, args.force,
        )
    elif args.all:
        for i in range(10):
            build_graph_for_sample(
                i, config, args.memories_dir, args.output_dir,
                args.dataset, args.force,
            )
    else:
        parser.print_help()
        return


if __name__ == "__main__":
    main()
