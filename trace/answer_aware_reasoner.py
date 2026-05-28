"""Answer-Aware Path Reasoner for TRACE (Method D).

Two-phase path search: after generating an answer (Phase 1), uses the answer
to find support paths that faithfully explain the reasoning (Phase 2).

Key insight: knowing the answer lets us search for paths that actually contain
the relevant facts, dramatically improving path faithfulness.
"""

import logging
import re
from typing import Dict, List, Optional

import numpy as np

from trace.causal_graph import CausalGraph
from trace.path_reasoner import PathReasoner, ScoredPath

logger = logging.getLogger(__name__)

_STOP_WORDS = frozenset({
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'can', 'to', 'of', 'in', 'for', 'on',
    'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
    'before', 'after', 'between', 'out', 'off', 'over', 'under',
    'again', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
    'other', 'some', 'such', 'no', 'not', 'only', 'own', 'same', 'so',
    'than', 'too', 'very', 'just', 'because', 'but', 'and', 'or', 'if',
    'while', 'about', 'up', 'it', 'its', 'he', 'she', 'they', 'them',
    'his', 'her', 'their', 'what', 'which', 'who', 'whom', 'this',
    'that', 'these', 'those', 'i', 'me', 'my', 'we', 'our', 'you',
    'your', 'also', 'many', 'much', 'well', 'like', 'even', 'still',
    'already', 'really', 'new', 'get', 'got', 'go', 'going', 'went',
    'come', 'came', 'make', 'made', 'take', 'took', 'know', 'knew',
    'think', 'thought', 'see', 'saw', 'want', 'said', 'tell', 'told',
})


class AnswerAwarePathReasoner:
    """Find support paths that faithfully connect query to answer."""

    def __init__(
        self,
        alpha_answer: float = 0.35,
        alpha_query: float = 0.25,
        alpha_coherence: float = 0.25,
        alpha_temporal: float = 0.15,
        max_depth: int = 4,
        top_k: int = 5,
    ):
        self.alpha_answer = alpha_answer
        self.alpha_query = alpha_query
        self.alpha_coherence = alpha_coherence
        self.alpha_temporal = alpha_temporal
        self.max_depth = max_depth
        self.top_k = top_k
        self._base = PathReasoner(max_depth=max_depth, top_k=top_k)

    @staticmethod
    def _extract_key_terms(text: str) -> set:
        """Extract key terms from text (lowercase, stop words removed, len>1)."""
        tokens = re.sub(r'[^\w\s]', ' ', text.lower()).split()
        return {t for t in tokens if t not in _STOP_WORDS and len(t) > 1}

    def _grounding_score(self, sp: ScoredPath, answer: str, query: str) -> float:
        """Token overlap between answer+query key terms and event descriptions."""
        key_terms = self._extract_key_terms(answer) | self._extract_key_terms(query)
        if not key_terms:
            return 0.5  # neutral for empty/stopword-only answers
        event_text = " ".join(ev.state_change for ev in sp.events)
        event_tokens = self._extract_key_terms(event_text)
        if not event_tokens:
            return 0.0
        overlap = len(key_terms & event_tokens)
        return overlap / len(key_terms)

    @staticmethod
    def _find_event_session(event_id: str, graph: CausalGraph) -> Optional[str]:
        """Find the session node that contains this event via graph edges."""
        try:
            for neighbor in graph.graph.predecessors(event_id):
                if graph.is_session_node(neighbor):
                    return neighbor
            for neighbor in graph.graph.successors(event_id):
                if graph.is_session_node(neighbor):
                    return neighbor
        except Exception:
            pass
        return None

    @staticmethod
    def _find_relevant_sentences(text: str, key_terms: set, max_sentences: int = 2) -> str:
        """Find sentences in text that contain any key terms. Returns joined string."""
        sentences = re.split(r'[.!?]+', text)
        scored = []
        for sent in sentences:
            sent = sent.strip()
            if not sent or len(sent) < 10:
                continue
            sent_lower = sent.lower()
            overlap = sum(1 for t in key_terms if t in sent_lower)
            if overlap > 0:
                scored.append((sent, overlap))
        scored.sort(key=lambda x: -x[1])
        return "; ".join(s for s, _ in scored[:max_sentences])

    def _enrich_with_session_detail(
        self, sp: ScoredPath, graph: CausalGraph, answer: str, query: str
    ):
        """Inject session-level detail into path explanation after event lines."""
        key_terms = self._extract_key_terms(answer) | self._extract_key_terms(query)
        if not key_terms:
            return

        # Pre-compute which events have enrichments
        enrichments = {}
        for i, ev in enumerate(sp.events):
            sid = self._find_event_session(ev.event_id, graph)
            if not sid:
                continue
            session = graph.get_session(sid)
            if not session:
                continue
            detail = self._find_relevant_sentences(
                session.get('summary', ''), key_terms
            )
            if detail:
                enrichments[i] = detail[:150]

        if not enrichments:
            return

        # Inject [Detail: ...] lines after matching E1, E2, ... lines
        new_lines = []
        event_counter = 0
        for line in sp.explanation.split('\n'):
            new_lines.append(line)
            marker = f'E{event_counter + 1} '
            if line.strip().startswith(marker):
                if event_counter in enrichments:
                    new_lines.append(f'  [Detail: "{enrichments[event_counter]}"]')
                event_counter += 1
        sp.explanation = '\n'.join(new_lines)

    def find_support_paths(
        self,
        query: str,
        answer: str,
        graph: CausalGraph,
        query_embedding: np.ndarray,
        answer_embedding: np.ndarray,
        note_id_to_idx: Optional[Dict[str, int]] = None,
        all_embeddings: Optional[np.ndarray] = None,
        question_type: str = "default",
    ) -> List[ScoredPath]:
        """Find paths that support the given answer to the query."""
        desc_cache = getattr(graph, '_desc_embedding_cache', None)
        if desc_cache is None:
            logger.info("No event description cache, falling back to base reasoner")
            return []

        cache_eids = desc_cache['event_ids']
        cache_embs = desc_cache['embeddings']
        cache_norms = desc_cache['norms']

        # Find answer-relevant events (top-5)
        answer_event_ids = self._find_relevant_events(
            answer_embedding, cache_eids, cache_embs, cache_norms, graph, top_k=5
        )

        # Find query-relevant events as seeds (top-5)
        query_event_ids = self._find_relevant_events(
            query_embedding, cache_eids, cache_embs, cache_norms, graph, top_k=5
        )

        if not query_event_ids:
            logger.info("No query-relevant events found for support paths")
            return []

        # Combine seeds: query events + answer events
        all_seeds = list(query_event_ids)
        seen = set(all_seeds)
        for eid in answer_event_ids:
            if eid not in seen:
                seen.add(eid)
                all_seeds.append(eid)

        # BFS from combined seeds
        raw_paths = graph.bfs_paths(
            seed_ids=all_seeds,
            max_depth=self.max_depth,
            max_paths=self.top_k * 8,
        )

        answer_event_set = set(answer_event_ids)

        scored_paths: List[ScoredPath] = []
        for raw_path in raw_paths:
            if len(raw_path) <= 1:
                continue

            if not self._base._prune_path(raw_path, graph):
                continue

            sp = self._base._build_scored_path(raw_path, graph)
            if sp is None or not sp.events:
                continue

            # Compute answer-aware scoring
            sp.neural_sim = self._compute_answer_sim(
                sp, answer_embedding, desc_cache, graph
            )
            query_sim = self._compute_query_sim(
                sp, query_embedding, desc_cache, graph
            )
            sp.path_conf = self._base._compute_path_conf(raw_path, graph)
            sp.temporal_cons = 1.0 if self._base._check_temporal_consistency(raw_path, graph) else 0.0

            # Bonus: path passes through an answer-relevant event
            answer_bonus = 0.0
            for ev in sp.events:
                if ev.event_id in answer_event_set:
                    answer_bonus = 0.15
                    break

            sp.total_score = (
                self.alpha_answer * sp.neural_sim
                + self.alpha_query * query_sim
                + self.alpha_coherence * sp.path_conf
                + self.alpha_temporal * sp.temporal_cons
                + answer_bonus
            )
            sp.update_val = query_sim

            # Grounding hard-filter: discard paths with zero keyword overlap
            grounding = self._grounding_score(sp, answer, query)
            if grounding == 0:
                continue

            sp.explanation = self._base.format_explanation(sp)
            scored_paths.append(sp)

        scored_paths.sort(key=lambda sp: -sp.total_score)
        top_paths = scored_paths[:self.top_k]

        # Enrich top paths with session-level detail
        for sp in top_paths:
            self._enrich_with_session_detail(sp, graph, answer, query)

        return top_paths

    def _find_relevant_events(
        self,
        embedding: np.ndarray,
        cache_eids: list,
        cache_embs: np.ndarray,
        cache_norms: np.ndarray,
        graph: CausalGraph,
        top_k: int = 5,
        min_sim: float = 0.35,
    ) -> List[str]:
        """Find events most similar to the given embedding."""
        q_norm = np.linalg.norm(embedding)
        if q_norm == 0:
            return []

        sims = cache_embs @ embedding / (cache_norms * q_norm)
        top_indices = np.argsort(sims)[::-1]

        result = []
        for idx in top_indices:
            if len(result) >= top_k:
                break
            sim = float(sims[idx])
            if sim < min_sim:
                break
            eid = cache_eids[idx]
            if graph.get_event(eid) is not None:
                result.append(eid)

        return result

    def _compute_answer_sim(
        self,
        sp: ScoredPath,
        answer_embedding: np.ndarray,
        desc_cache: dict,
        graph: CausalGraph,
    ) -> float:
        """Max cosine similarity between answer and any event's state_change."""
        eid_to_idx = {eid: i for i, eid in enumerate(desc_cache['event_ids'])}
        a_norm = np.linalg.norm(answer_embedding)
        if a_norm == 0:
            return 0.0

        max_sim = 0.0
        for ev in sp.events:
            ci = eid_to_idx.get(ev.event_id)
            if ci is not None:
                sim = float(
                    desc_cache['embeddings'][ci] @ answer_embedding
                    / (desc_cache['norms'][ci] * a_norm)
                )
                if sim > max_sim:
                    max_sim = sim
        return max_sim

    def _compute_query_sim(
        self,
        sp: ScoredPath,
        query_embedding: np.ndarray,
        desc_cache: dict,
        graph: CausalGraph,
    ) -> float:
        """Max cosine similarity between query and any event's state_change."""
        eid_to_idx = {eid: i for i, eid in enumerate(desc_cache['event_ids'])}
        q_norm = np.linalg.norm(query_embedding)
        if q_norm == 0:
            return 0.0

        max_sim = 0.0
        for ev in sp.events:
            ci = eid_to_idx.get(ev.event_id)
            if ci is not None:
                sim = float(
                    desc_cache['embeddings'][ci] @ query_embedding
                    / (desc_cache['norms'][ci] * q_norm)
                )
                if sim > max_sim:
                    max_sim = sim
        return max_sim
