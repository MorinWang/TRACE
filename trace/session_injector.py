"""Session hyperedge injection for TRACE causal graph (Method A).

Adds session auxiliary nodes via star expansion: each session becomes a node
connected bidirectionally to all events from that session. This provides
structural retrieval paths that don't depend on embedding quality.

No LLM calls — pure Python, runs in milliseconds.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

from trace.parsing_utils import extract_session_num

logger = logging.getLogger(__name__)


@dataclass
class SessionNode:
    """A session auxiliary node for star expansion."""
    session_id: str         # "sess_0_1"
    session_key: str        # "session_1_summary"
    summary: str            # session summary text
    date_time: str          # session timestamp
    event_ids: List[str] = field(default_factory=list)
    note_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SessionNode":
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)


def _group_notes_by_timestamp(memories: dict) -> Dict[str, List[str]]:
    """Group A-Mem note IDs by their timestamp (= session)."""
    session_notes = defaultdict(list)
    for note_id, note in memories.items():
        ts = getattr(note, 'timestamp', '') or ''
        if ts:
            session_notes[ts].append(note_id)
    return dict(session_notes)


def inject_session_nodes(
    graph,
    memories: dict,
    session_summaries: Dict[str, str],
    conversation_sessions: dict,
    sample_idx: int,
) -> int:
    """Inject session auxiliary nodes into an existing CausalGraph.

    Args:
        graph: CausalGraph instance (modified in place).
        memories: dict of note_id -> note objects (A-Mem memories).
        session_summaries: {"session_N_summary": "text..."} from LoCoMo.
        conversation_sessions: sample.conversation.sessions dict.
        sample_idx: sample index for generating session IDs.

    Returns:
        Number of session nodes injected.
    """
    if not session_summaries:
        logger.warning("No session summaries available, skipping injection")
        return 0

    # Group notes by timestamp to reconstruct sessions
    ts_groups = _group_notes_by_timestamp(memories)
    sorted_timestamps = sorted(ts_groups.keys())

    # Sort session summary keys numerically
    sorted_keys = sorted(session_summaries.keys(), key=_extract_session_num)

    # Align timestamps with session keys (same order = same session)
    # Handle mismatch: use min of both lengths
    n_sessions = min(len(sorted_timestamps), len(sorted_keys))
    if n_sessions == 0:
        logger.warning("Could not align timestamps with session summaries")
        return 0

    if len(sorted_timestamps) != len(sorted_keys):
        logger.warning(
            f"Timestamp groups ({len(sorted_timestamps)}) != "
            f"session summaries ({len(sorted_keys)}), using first {n_sessions}"
        )

    injected = 0
    prev_session_id = None

    for i in range(n_sessions):
        ts = sorted_timestamps[i]
        sess_key = sorted_keys[i]
        summary = session_summaries[sess_key]
        note_ids = ts_groups[ts]

        # Collect event IDs from notes in this session
        event_ids = []
        seen_eids = set()
        for nid in note_ids:
            for event in graph.get_events_by_note(nid):
                if event.event_id not in seen_eids:
                    seen_eids.add(event.event_id)
                    event_ids.append(event.event_id)

        if not event_ids:
            continue

        # Get session date_time from conversation if available
        sess_num = extract_session_num(sess_key)
        date_time = ts  # fallback to note timestamp
        if conversation_sessions and sess_num in conversation_sessions:
            session_obj = conversation_sessions[sess_num]
            date_time = getattr(session_obj, 'date_time', ts)

        session_id = f"sess_{sample_idx}_{sess_num}"

        session_data = SessionNode(
            session_id=session_id,
            session_key=sess_key,
            summary=summary,
            date_time=date_time,
            event_ids=event_ids,
            note_ids=note_ids,
        ).to_dict()

        graph.add_session(session_data)
        injected += 1

        # Temporal edge between consecutive sessions
        if prev_session_id is not None:
            graph.graph.add_edge(
                prev_session_id, session_id,
                edge_type="temporal_before",
                confidence=1.0,
            )

        prev_session_id = session_id

    logger.info(
        f"Injected {injected} session nodes for sample {sample_idx} "
        f"(total graph nodes: {graph.graph.number_of_nodes()}, "
        f"edges: {graph.graph.number_of_edges()})"
    )
    return injected


def inject_topic_nodes(
    graph,
    llm,
    sample_idx: int,
) -> int:
    """Inject topic nodes into a graph that already has session nodes.

    Must be called AFTER inject_session_nodes(). Uses LLM to cluster
    sessions into topics (1 API call per sample).

    Args:
        graph: CausalGraph with sessions already injected.
        llm: LLM controller with get_completion() method.
        sample_idx: sample index.

    Returns:
        Number of topic nodes injected.
    """
    if not graph._sessions:
        logger.warning("No sessions in graph, skipping topic injection")
        return 0

    # Skip if topics already loaded from cache
    if graph._topics:
        logger.info(f"Topics already present ({len(graph._topics)}), skipping clustering")
        return len(graph._topics)

    from trace.topic_clusterer import cluster_sessions_llm

    topics = cluster_sessions_llm(
        session_summaries=graph._sessions,
        llm=llm,
        sample_idx=sample_idx,
    )

    for topic in topics:
        graph.add_topic(topic.to_dict())

    logger.info(
        f"Injected {len(topics)} topic nodes for sample {sample_idx} "
        f"(total graph nodes: {graph.graph.number_of_nodes()}, "
        f"edges: {graph.graph.number_of_edges()})"
    )
    return len(topics)
