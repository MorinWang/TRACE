"""TRACE Causal Graph: NetworkX-based typed directed event graph.

Wraps nx.DiGraph with typed node/edge storage and efficient indices
for participant-based lookup.
"""

import json
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

from trace.event_schema import EventNode, TypedEdge, HyperedgeNode, VALID_EDGE_TYPES

logger = logging.getLogger(__name__)


class CausalGraph:
    """Typed causal-temporal event graph built on NetworkX."""

    def __init__(self):
        self.graph = nx.DiGraph()
        self._events: Dict[str, EventNode] = {}
        self._sessions: Dict[str, dict] = {}           # session_id -> SessionNode.to_dict()
        self._topics: Dict[str, dict] = {}             # topic_id -> TopicNode.to_dict()
        self._hyperedges: Dict[str, HyperedgeNode] = {}  # hyperedge view of _sessions ∪ _topics
        self._note_to_events: Dict[str, List[str]] = defaultdict(list)
        self._note_to_session: Dict[str, str] = {}     # note_id -> session_id
        self._participant_index: Dict[str, Set[str]] = defaultdict(set)

    # ------------------------------------------------------------------
    # Add operations
    # ------------------------------------------------------------------

    def add_event(self, event: EventNode):
        """Add an event node to the graph and update all indices."""
        if event.event_id in self._events:
            logger.warning(f"Event {event.event_id} already exists, skipping")
            return

        self._events[event.event_id] = event
        self.graph.add_node(event.event_id)

        for note_id in event.source_note_ids:
            self._note_to_events[note_id].append(event.event_id)

        for p in event.participants:
            self._participant_index[p.lower()].add(event.event_id)

    def remove_event(self, event_id: str):
        """Remove an event node and all its edges from the graph."""
        if event_id not in self._events:
            return
        event = self._events.pop(event_id)
        if self.graph.has_node(event_id):
            self.graph.remove_node(event_id)  # also removes all edges
        for note_id in event.source_note_ids:
            ids = self._note_to_events.get(note_id, [])
            if event_id in ids:
                ids.remove(event_id)
        for p in event.participants:
            s = self._participant_index.get(p.lower())
            if s:
                s.discard(event_id)

    def add_edge(self, edge: TypedEdge):
        """Add a typed edge between two events."""
        if edge.source_event_id not in self._events:
            logger.warning(f"Source event {edge.source_event_id} not in graph, skipping edge")
            return
        if edge.target_event_id not in self._events:
            logger.warning(f"Target event {edge.target_event_id} not in graph, skipping edge")
            return

        self.graph.add_edge(
            edge.source_event_id,
            edge.target_event_id,
            edge_type=edge.edge_type,
            confidence=edge.confidence,
            reason=edge.reason,
            data=edge,
        )

    def add_session(self, session_data: dict):
        """Add a session auxiliary node with star-expansion edges.

        session_data must have: session_id, summary, date_time, event_ids, note_ids.
        """
        sid = session_data["session_id"]
        if sid in self._sessions:
            return

        self._sessions[sid] = session_data
        self.graph.add_node(sid, node_type="session")

        # Star expansion: bidirectional edges between session and its events
        for eid in session_data.get("event_ids", []):
            if eid in self._events:
                self.graph.add_edge(eid, sid, edge_type="belongs_to", confidence=1.0)
                self.graph.add_edge(sid, eid, edge_type="contains", confidence=1.0)

        # Index note -> session
        for nid in session_data.get("note_ids", []):
            self._note_to_session[nid] = sid

        # Register hyperedge view (bipartite-incidence representation)
        self._hyperedges[sid] = HyperedgeNode(
            h_id=sid,
            edge_type='session',
            member_ids=list(session_data.get("event_ids", [])),
            confidence=1.0,
        )

    def get_session(self, session_id: str) -> Optional[dict]:
        return self._sessions.get(session_id)

    def is_session_node(self, node_id: str) -> bool:
        return node_id in self._sessions

    def add_topic(self, topic_data: dict):
        """Add a topic auxiliary node with star-expansion edges to sessions.

        topic_data must have: topic_id, label, session_ids, description.
        """
        tid = topic_data["topic_id"]
        if tid in self._topics:
            return

        self._topics[tid] = topic_data
        self.graph.add_node(tid, node_type="topic")

        # Star expansion: bidirectional edges between topic and its sessions
        for sid in topic_data.get("session_ids", []):
            if sid in self._sessions:
                self.graph.add_edge(sid, tid, edge_type="belongs_to_topic", confidence=1.0)
                self.graph.add_edge(tid, sid, edge_type="topic_contains", confidence=1.0)

        # Register hyperedge view (bipartite-incidence representation)
        self._hyperedges[tid] = HyperedgeNode(
            h_id=tid,
            edge_type='topic',
            member_ids=list(topic_data.get("session_ids", [])),
            confidence=1.0,
        )

    def get_topic(self, topic_id: str) -> Optional[dict]:
        return self._topics.get(topic_id)

    def is_topic_node(self, node_id: str) -> bool:
        return node_id in self._topics

    def get_hyperedge(self, h_id: str) -> Optional[HyperedgeNode]:
        """Return HyperedgeNode for a given session/topic id, or None."""
        return self._hyperedges.get(h_id)

    def is_hyperedge_node(self, node_id: str) -> bool:
        """True if node_id is registered as a hyperedge (session or topic)."""
        return node_id in self._hyperedges

    def get_all_hyperedges(self) -> List[HyperedgeNode]:
        """Return all hyperedges (sessions + topics) in insertion order."""
        return list(self._hyperedges.values())

    # ------------------------------------------------------------------
    # Query operations
    # ------------------------------------------------------------------

    def get_event(self, event_id: str) -> Optional[EventNode]:
        return self._events.get(event_id)

    def get_events_by_note(self, note_id: str) -> List[EventNode]:
        ids = self._note_to_events.get(note_id, [])
        return [self._events[eid] for eid in ids if eid in self._events]

    def get_events_by_participant_overlap(
        self,
        participants: List[str],
        min_jaccard: float = 0.5,
        max_k: int = 5,
        exclude_ids: Optional[Set[str]] = None,
    ) -> List[EventNode]:
        """Find events whose participants overlap with the given list.

        Returns up to max_k events sorted by descending Jaccard similarity.
        """
        p_set = {p.lower() for p in participants}
        if not p_set:
            return []

        exclude = exclude_ids or set()

        # Collect candidate event IDs that share at least one participant
        candidate_ids: Set[str] = set()
        for p in p_set:
            candidate_ids |= self._participant_index.get(p, set())
        candidate_ids -= exclude

        scored: List[Tuple[EventNode, float]] = []
        for eid in candidate_ids:
            event = self._events[eid]
            e_set = {p.lower() for p in event.participants}
            union = p_set | e_set
            jaccard = len(p_set & e_set) / len(union) if union else 0.0
            if jaccard >= min_jaccard:
                scored.append((event, jaccard))

        scored.sort(key=lambda x: -x[1])
        return [e for e, _ in scored[:max_k]]

    def get_all_events(self) -> List[EventNode]:
        return list(self._events.values())

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def num_events(self) -> int:
        return len(self._events)

    def num_edges(self) -> int:
        return self.graph.number_of_edges()

    def edge_type_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = defaultdict(int)
        for _, _, data in self.graph.edges(data=True):
            counts[data.get("edge_type", "unknown")] += 1
        return dict(counts)

    def summary(self) -> str:
        edge_counts = self.edge_type_counts()
        lines = [
            f"Events: {self.num_events()}",
            f"Edges: {self.num_edges()}",
        ]
        for etype in sorted(VALID_EDGE_TYPES):
            if etype in edge_counts:
                lines.append(f"  {etype}: {edge_counts[etype]}")
        # Count invalidated events
        invalidated = sum(1 for e in self._events.values() if e.valid_until is not None)
        if invalidated:
            lines.append(f"Invalidated events: {invalidated}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Serialization (JSON for human readability and audit)
    # ------------------------------------------------------------------

    def save(self, filepath: str, metadata: Optional[dict] = None):
        """Save the graph to a JSON file (schema 1.1 with hyperedges field)."""
        edges = []
        for u, v, data in self.graph.edges(data=True):
            edge_obj = data.get("data")
            if isinstance(edge_obj, TypedEdge):
                edges.append(edge_obj.to_dict())
            else:
                edges.append({
                    "source_event_id": u,
                    "target_event_id": v,
                    "edge_type": data.get("edge_type", "unknown"),
                    "confidence": data.get("confidence", 0.0),
                    "reason": data.get("reason", ""),
                })

        # Copy caller metadata so we don't mutate the caller's dict; ensure
        # schema_version is set without overriding a caller-provided value.
        meta = dict(metadata or {})
        meta.setdefault("schema_version", "1.1")
        payload = {
            "metadata": meta,
            "events": {eid: e.to_dict() for eid, e in self._events.items()},
            "edges": edges,
            "sessions": list(self._sessions.values()),
            "topics": list(self._topics.values()),
        }
        if self._hyperedges:
            payload["hyperedges"] = [h.to_dict() for h in self._hyperedges.values()]
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved graph to {filepath}: {self.num_events()} events, {self.num_edges()} edges")

    @classmethod
    def load(cls, filepath: str) -> "CausalGraph":
        """Load a graph from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            payload = json.load(f)

        graph = cls()

        for eid, edict in payload.get("events", {}).items():
            event = EventNode.from_dict(edict)
            graph.add_event(event)

        for edict in payload.get("edges", []):
            edge = TypedEdge.from_dict(edict)
            graph.add_edge(edge)

        # Restore session nodes (also populates _hyperedges index for sessions)
        for sdata in payload.get("sessions", []):
            graph.add_session(sdata)

        # Restore topic nodes (also populates _hyperedges index for topics)
        for tdata in payload.get("topics", []):
            graph.add_topic(tdata)

        # NOTE: payload.get("hyperedges", []) is informational only — the
        # _hyperedges index has already been built from sessions/topics above.
        # See spec §3.2.4: sessions/topics are the JSON source of truth.

        return graph

    # ------------------------------------------------------------------
    # BFS path search
    # ------------------------------------------------------------------

    def bfs_paths(
        self,
        seed_ids: List[str],
        max_depth: int = 3,
        max_paths: int = 30,
        max_neighbors_per_node: int = 5,
        max_session_fanout: int = 3,
        max_session_hops: int = 1,
        max_topic_fanout: int = 2,
        max_topic_hops: int = 1,
    ) -> List[List[Tuple[str, Optional[str]]]]:
        """BFS path search from seed events.

        Handles 3 node types: event, session, topic.
        Each type has independent fanout and hop limits.
        """
        from collections import deque

        paths: List[List[Tuple[str, Optional[str]]]] = []
        valid_seeds = [sid for sid in seed_ids if sid in self._events]

        if not valid_seeds:
            return paths

        # BFS queue: (current_path, visited, session_hops, topic_hops)
        queue: deque = deque()

        for sid in valid_seeds:
            initial_path = [(sid, None)]
            queue.append((initial_path, {sid}, 0, 0))

        while queue:
            current_path, visited, session_hops, topic_hops = queue.popleft()
            current_id = current_path[-1][0]
            current_depth = len(current_path) - 1

            if current_depth >= max_depth:
                continue

            current_is_session = self.is_session_node(current_id)
            current_is_topic = self.is_topic_node(current_id)

            # Bidirectional: follow both forward and backward edges
            neighbors = []
            for successor_id in self.graph.successors(current_id):
                if successor_id not in visited:
                    edge_data = self.graph.edges[current_id, successor_id]
                    edge_obj = edge_data.get("data")
                    if edge_obj and getattr(edge_obj, "t_invalid_at", None) is not None:
                        continue
                    neighbors.append((successor_id, edge_data, False))
            for predecessor_id in self.graph.predecessors(current_id):
                if predecessor_id not in visited:
                    edge_data = self.graph.edges[predecessor_id, current_id]
                    edge_obj = edge_data.get("data")
                    if edge_obj and getattr(edge_obj, "t_invalid_at", None) is not None:
                        continue
                    neighbors.append((predecessor_id, edge_data, True))

            # Hub limiting: different limits per node type
            if current_is_topic:
                fanout_limit = max_topic_fanout
            elif current_is_session:
                fanout_limit = max_session_fanout
            else:
                fanout_limit = max_neighbors_per_node
            if len(neighbors) > fanout_limit:
                _UPDATE_TYPES = {"updates", "contradicts"}
                neighbors.sort(
                    key=lambda x: (
                        1 if x[1].get("edge_type") in _UPDATE_TYPES else 0,
                        x[1].get("confidence", 0.5),
                    ),
                    reverse=True,
                )
                neighbors = neighbors[:fanout_limit]

            for neighbor_id, edge_data, is_backward in neighbors:
                neighbor_is_session = self.is_session_node(neighbor_id)
                neighbor_is_topic = self.is_topic_node(neighbor_id)

                # Enforce hop limits
                if neighbor_is_session and session_hops >= max_session_hops:
                    continue
                if neighbor_is_topic and topic_hops >= max_topic_hops:
                    continue

                edge_type = edge_data.get("edge_type", "unknown")
                if is_backward:
                    if edge_type == "temporal_before":
                        edge_type = "temporal_after"
                    elif edge_type == "temporal_after":
                        edge_type = "temporal_before"

                new_path = list(current_path)
                new_path[-1] = (current_id, edge_type)
                new_path.append((neighbor_id, None))

                paths.append(new_path)
                if len(paths) >= max_paths:
                    return paths

                new_visited = visited | {neighbor_id}
                new_session_hops = session_hops + (1 if neighbor_is_session else 0)
                new_topic_hops = topic_hops + (1 if neighbor_is_topic else 0)
                queue.append((new_path, new_visited, new_session_hops, new_topic_hops))

        return paths
