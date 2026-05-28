"""TRACE Path Reasoner: BFS path search + fusion scoring + explanation formatting.

Implements pruning, scoring, and explanation generation for support paths
through the causal-temporal event graph.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from trace.event_schema import EventNode
from trace.causal_graph import CausalGraph

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ScoredPath:
    """A scored support path through the event graph."""

    events: List[EventNode] = field(default_factory=list)
    edge_types: List[str] = field(default_factory=list)
    edge_confidences: List[float] = field(default_factory=list)
    neural_sim: float = 0.0
    path_conf: float = 0.0
    temporal_cons: float = 1.0       # binary {0, 1}
    update_val: float = 1.0          # 1.0 / 0.5 / 0.0
    total_score: float = 0.0
    explanation: str = ""
    session_summaries: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# PathReasoner
# ---------------------------------------------------------------------------

class PathReasoner:
    """BFS path search + fusion scoring + pruning.

    Retrieves paths from the causal graph, applies pruning rules,
    computes fusion scores, and formats explanations.
    """

    def __init__(
        self,
        alpha: float = 0.4,     # neural_sim weight
        beta: float = 0.3,      # path_conf weight
        gamma: float = 0.15,    # temporal_cons weight
        delta: float = 0.15,    # update_val weight
        max_depth: int = 3,
        top_k: int = 10,
        neural_sim_aggregator: str = "mean",   # NEW v4 (B1): "max" (legacy) | "mean"
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.max_depth = max_depth
        self.top_k = top_k
        self.neural_sim_aggregator = neural_sim_aggregator

    def find_and_score_paths(
        self,
        seed_event_ids: List[str],
        graph: CausalGraph,
        query_embedding: Optional[np.ndarray] = None,
        note_id_to_idx: Optional[Dict[str, int]] = None,
        all_embeddings: Optional[np.ndarray] = None,
    ) -> List[ScoredPath]:
        """Main entry: BFS from seeds -> prune -> score -> sort -> top-K.

        Also supplements BFS paths with direct evidence events — query-relevant
        events found via embedding similarity that BFS couldn't reach (e.g.
        isolated nodes or events in disconnected components).
        """
        # Over-generate by 5x to account for pruning and seed-only paths
        raw_paths = graph.bfs_paths(
            seed_ids=seed_event_ids,
            max_depth=self.max_depth,
            max_paths=self.top_k * 5,
        )

        scored_paths: List[ScoredPath] = []

        for raw_path in raw_paths:
            # Skip length-1 paths (seed-only) -- they provide no reasoning chain
            if len(raw_path) <= 1:
                continue

            # Apply pruning
            if not self._prune_path(raw_path, graph):
                continue

            # Build ScoredPath
            sp = self._build_scored_path(raw_path, graph)
            if sp is None:
                continue

            # Compute scoring components
            sp.path_conf = self._compute_path_conf(raw_path, graph)
            sp.temporal_cons = 1.0 if self._check_temporal_consistency(raw_path, graph) else 0.0
            sp.update_val = self._compute_update_val(raw_path, graph)

            # Neural similarity (if embeddings available)
            if query_embedding is not None and note_id_to_idx is not None and all_embeddings is not None:
                sp.neural_sim = self._compute_neural_sim(
                    raw_path, graph, query_embedding, note_id_to_idx, all_embeddings
                )

            # Total score
            sp.total_score = self._score_path(sp)

            # Format explanation
            sp.explanation = self.format_explanation(sp)

            scored_paths.append(sp)

        # Sort descending by total_score
        scored_paths.sort(key=lambda sp: -sp.total_score)

        # Supplement with direct evidence events that BFS couldn't reach
        direct = []
        if query_embedding is not None:
            direct = self._find_direct_evidence(
                graph, query_embedding, scored_paths, seed_event_ids,
            )

        # Merge: keep top BFS paths + always include direct evidence
        if direct:
            bfs_top = scored_paths[: max(1, self.top_k - len(direct))]
            return bfs_top + direct
        return scored_paths[: self.top_k]

    def _find_direct_evidence(
        self,
        graph: CausalGraph,
        query_embedding: np.ndarray,
        existing_paths: List[ScoredPath],
        seed_event_ids: List[str],
        max_direct: int = 3,
        min_sim: float = 0.45,
    ) -> List[ScoredPath]:
        """Find query-relevant events that BFS couldn't reach.

        Searches all events by embedding similarity to the query. Events already
        covered by BFS paths or seeds are skipped. Returns single-event
        ScoredPaths marked as direct evidence.
        """
        # Collect event IDs already in BFS paths
        covered_ids = set(seed_event_ids)
        for sp in existing_paths:
            for ev in sp.events:
                covered_ids.add(ev.event_id)

        # Get cached event description embeddings from the pipeline
        # (stored on the graph object during pipeline.retrieve)
        desc_cache = getattr(graph, '_desc_embedding_cache', None)
        if desc_cache is None:
            return []

        cache_eids = desc_cache['event_ids']
        cache_embs = desc_cache['embeddings']
        cache_norms = desc_cache['norms']

        q_norm = np.linalg.norm(query_embedding)
        if q_norm == 0:
            return []

        sims = cache_embs @ query_embedding / (cache_norms * q_norm)

        # Find top candidates not already covered
        top_indices = np.argsort(sims)[::-1]
        direct_paths = []

        for idx in top_indices:
            if len(direct_paths) >= max_direct:
                break
            sim = float(sims[idx])
            if sim < min_sim:
                break

            eid = cache_eids[idx]
            if eid in covered_ids:
                continue

            event = graph.get_event(eid)
            if event is None:
                continue

            # Create a single-event ScoredPath as direct evidence
            sp = ScoredPath(
                events=[event],
                edge_types=[],
                edge_confidences=[],
                neural_sim=sim,
                path_conf=1.0,
                temporal_cons=1.0,
                update_val=event.update_val if hasattr(event, 'update_val') else 1.0,
                total_score=sim * 0.9,  # slightly below equivalent BFS path
            )
            sp.explanation = self._format_direct_evidence(event, sim)
            direct_paths.append(sp)
            covered_ids.add(eid)

        if direct_paths:
            logger.info(
                f"Added {len(direct_paths)} direct evidence events "
                f"(sims: {[f'{sp.neural_sim:.2f}' for sp in direct_paths]})"
            )
        return direct_paths

    @staticmethod
    def _format_direct_evidence(event: EventNode, sim: float) -> str:
        """Format a direct evidence event."""
        return (
            f"[Direct Evidence] {event.state_change} "
            f"({event.time_anchor}, {event.event_type})\n"
            f"Score: {sim:.2f} (direct match)"
        )

    # ------------------------------------------------------------------
    # Build ScoredPath from raw path
    # ------------------------------------------------------------------

    def _build_scored_path(
        self,
        raw_path: List[Tuple[str, Optional[str]]],
        graph: CausalGraph,
    ) -> Optional[ScoredPath]:
        """Convert a raw BFS path into a ScoredPath with events and edges.

        Session nodes are skipped in the events list but recorded as
        'session_link' edges. Their summaries are collected separately.
        """
        events = []
        edge_types = []
        edge_confidences = []
        session_summaries = []

        for i, (node_id, edge_type) in enumerate(raw_path):
            # Check if this is a topic node
            if graph.is_topic_node(node_id):
                topic_data = graph.get_topic(node_id)
                if topic_data:
                    label = topic_data.get("label", "")
                    if label:
                        session_summaries.append(f"[Topic: {label}]")
                if edge_type is not None:
                    edge_types.append("topic_link")
                    edge_confidences.append(0.5)
                continue

            # Check if this is a session node
            if graph.is_session_node(node_id):
                session_data = graph.get_session(node_id)
                if session_data:
                    summary = session_data.get("summary", "")
                    if summary:
                        session_summaries.append(summary[:200])
                if edge_type is not None:
                    edge_types.append("session_link")
                    edge_confidences.append(0.7)
                continue

            event = graph.get_event(node_id)
            if event is None:
                return None
            events.append(event)

            if edge_type is not None:
                next_id = raw_path[i + 1][0] if i + 1 < len(raw_path) else None
                if next_id and (graph.is_session_node(next_id) or graph.is_topic_node(next_id)):
                    link_type = "topic_link" if graph.is_topic_node(next_id) else "session_link"
                    link_conf = 0.5 if graph.is_topic_node(next_id) else 0.7
                    edge_types.append(link_type)
                    edge_confidences.append(link_conf)
                else:
                    edge_types.append(edge_type)
                    if next_id and graph.graph.has_edge(node_id, next_id):
                        conf = graph.graph.edges[node_id, next_id].get("confidence", 1.0)
                        edge_confidences.append(conf)
                    else:
                        edge_confidences.append(1.0)

        if not events:
            return None

        return ScoredPath(
            events=events,
            edge_types=edge_types,
            edge_confidences=edge_confidences,
            session_summaries=session_summaries,
        )

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------

    def _prune_path(
        self,
        raw_path: List[Tuple[str, Optional[str]]],
        graph: CausalGraph,
    ) -> bool:
        """Return True if path should be KEPT (passes all pruning checks)."""
        if not self._check_temporal_consistency(raw_path, graph):
            return False
        if not self._check_no_consecutive_contradicts(raw_path):
            return False
        return True

    def _check_temporal_consistency(
        self,
        raw_path: List[Tuple[str, Optional[str]]],
        graph: CausalGraph,
    ) -> bool:
        """Check that temporal edges are not violated. Skips session edges."""
        for i, (event_id, edge_type) in enumerate(raw_path):
            if edge_type not in ("temporal_before", "temporal_after"):
                continue
            if i + 1 >= len(raw_path):
                continue
            # Skip if either end is a session or topic node
            next_id = raw_path[i + 1][0]
            if (graph.is_session_node(event_id) or graph.is_topic_node(event_id)
                    or graph.is_session_node(next_id) or graph.is_topic_node(next_id)):
                continue

            source_event = graph.get_event(event_id)
            target_event = graph.get_event(next_id)

            if source_event is None or target_event is None:
                continue

            source_time = self._parse_date(source_event.time_anchor)
            target_time = self._parse_date(target_event.time_anchor)

            if source_time is not None and target_time is not None:
                if edge_type == "temporal_before" and source_time > target_time:
                    return False
                if edge_type == "temporal_after" and source_time < target_time:
                    return False

        return True

    def _check_no_consecutive_contradicts(
        self,
        raw_path: List[Tuple[str, Optional[str]]],
    ) -> bool:
        """No two consecutive contradicts edges allowed."""
        edge_types = [et for _, et in raw_path if et is not None]
        for i in range(len(edge_types) - 1):
            if edge_types[i] == "contradicts" and edge_types[i + 1] == "contradicts":
                return False
        return True

    @staticmethod
    def _parse_date(time_anchor: str) -> Optional[int]:
        """Try to parse a date string into an ordinal for comparison.

        Returns None if the date cannot be parsed (e.g. relative anchors).
        """
        if not time_anchor:
            return None
        import re
        # Match YYYY-MM-DD or YYYY/MM/DD
        match = re.match(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", time_anchor)
        if match:
            try:
                from datetime import date
                return date(int(match.group(1)), int(match.group(2)), int(match.group(3))).toordinal()
            except ValueError:
                return None
        return None

    # ------------------------------------------------------------------
    # Scoring components
    # ------------------------------------------------------------------

    def _compute_path_conf(
        self,
        raw_path: List[Tuple[str, Optional[str]]],
        graph: CausalGraph,
    ) -> float:
        """Product of edge confidences along the path.

        Session edges (belongs_to/contains) apply a 0.7 discount.
        Enables edges apply a 0.5 discount.
        Contradiction penalty: if any edge is 'contradicts', multiply by 0.5.
        """
        confidences = []
        has_contradiction = False

        for i, (node_id, edge_type) in enumerate(raw_path):
            if edge_type is None:
                continue

            # Skip session-internal edges
            if edge_type in ("belongs_to", "contains"):
                confidences.append(0.7)
                continue

            # Skip topic-internal edges
            if edge_type in ("belongs_to_topic", "topic_contains"):
                confidences.append(0.5)
                continue

            next_id = raw_path[i + 1][0] if i + 1 < len(raw_path) else None
            if next_id:
                # Check both directions — path may traverse backward edges
                if graph.graph.has_edge(node_id, next_id):
                    conf = graph.graph.edges[node_id, next_id].get("confidence", 1.0)
                elif graph.graph.has_edge(next_id, node_id):
                    conf = graph.graph.edges[next_id, node_id].get("confidence", 1.0)
                else:
                    conf = 1.0
                # Discount weak "enables" edges — thematic associations, not causal
                if edge_type == "enables":
                    conf *= 0.5
                confidences.append(conf)
            else:
                confidences.append(1.0)

            if edge_type == "contradicts":
                has_contradiction = True

        if not confidences:
            return 1.0

        product = 1.0
        for c in confidences:
            product *= c

        if has_contradiction:
            product *= 0.5

        return product

    def _compute_update_val(
        self,
        raw_path: List[Tuple[str, Optional[str]]],
        graph: CausalGraph,
    ) -> float:
        """Minimum update_val across all events in the path (skips session nodes)."""
        if not raw_path:
            return 1.0

        min_val = 1.0
        for event_id, _ in raw_path:
            if graph.is_session_node(event_id) or graph.is_topic_node(event_id):
                continue
            event = graph.get_event(event_id)
            if event is not None:
                min_val = min(min_val, event.update_val)
        return min_val

    def _compute_neural_sim(
        self,
        raw_path: List[Tuple[str, Optional[str]]],
        graph: CausalGraph,
        query_embedding: np.ndarray,
        note_id_to_idx: Dict[str, int],
        all_embeddings: np.ndarray,
    ) -> float:
        """Aggregate per-event cosine sim along path.

        aggregator='max' (legacy A-Mem-style): max over any event's source_note in path
        aggregator='mean' (default v4): mean over per-event maxes (penalizes走偏 events)

        Skips session and topic nodes (they have no note embeddings).
        """
        if not raw_path:
            return 0.0

        sims: List[float] = []
        for event_id, _ in raw_path:
            if graph.is_session_node(event_id) or graph.is_topic_node(event_id):
                continue
            event = graph.get_event(event_id)
            if event is None:
                continue
            ev_max = 0.0
            for note_id in event.source_note_ids:
                idx = note_id_to_idx.get(note_id)
                if idx is not None and idx < len(all_embeddings):
                    sim = self._cosine_similarity(query_embedding, all_embeddings[idx])
                    if sim > ev_max:
                        ev_max = sim
            if ev_max > 0:
                sims.append(ev_max)

        if not sims:
            return 0.0
        if self.neural_sim_aggregator == "max":
            return max(sims)
        return sum(sims) / len(sims)  # mean

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _score_path(self, sp: ScoredPath) -> float:
        """Linear fusion: alpha*neural_sim + beta*path_conf + gamma*temporal_cons + delta*update_val."""
        return (
            self.alpha * sp.neural_sim
            + self.beta * sp.path_conf
            + self.gamma * sp.temporal_cons
            + self.delta * sp.update_val
        )

    # ------------------------------------------------------------------
    # Explanation formatting
    # ------------------------------------------------------------------

    def format_explanation(self, sp: ScoredPath) -> str:
        """Format a ScoredPath into human-readable explanation string."""
        if not sp.events:
            return ""

        lines = []
        session_idx = 0
        for i, event in enumerate(sp.events):
            lines.append(
                f"E{i+1} ({event.state_change}, {event.time_anchor}, {event.event_type})"
            )
            if i < len(sp.edge_types):
                et = sp.edge_types[i]
                if et == "session_link" and session_idx < len(sp.session_summaries):
                    snippet = sp.session_summaries[session_idx][:80]
                    lines.append(f"  ==[Session: \"{snippet}...\"]==>")
                    session_idx += 1
                else:
                    lines.append(f"  --[{et}]-->")

        lines.append(
            f"Score: {sp.total_score:.2f} "
            f"(neural={sp.neural_sim:.2f}, conf={sp.path_conf:.2f}, "
            f"temporal={sp.temporal_cons:.0f}, update={sp.update_val:.1f})"
        )
        return "\n".join(lines)
