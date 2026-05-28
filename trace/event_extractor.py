"""TRACE Event Extractor: LLM-based event and edge extraction from memory notes.

Orchestrates intra-note extraction (one LLM call per note) and cross-note
edge inference (batched candidates).
"""

import logging
import re
from typing import Dict, List, Optional, Set, Tuple

from trace.event_schema import EventNode, TypedEdge
from trace.parsing_utils import try_parse_date
from trace.causal_graph import CausalGraph
from trace.prompts.extraction import (
    format_extraction_prompt,
    parse_extraction_response,
)
from trace.prompts.cross_note import (
    format_cross_note_prompt,
    parse_cross_note_response,
)

logger = logging.getLogger(__name__)


def build_nickname_map(memories: dict) -> Dict[str, str]:
    """Build a nickname → canonical name mapping from memory note contents.

    Scans all "Speaker X says:" patterns to extract canonical names,
    then generates common nickname variants (first 3+ char prefixes).
    """
    canonical_names = set()
    for note in memories.values():
        content = getattr(note, 'content', '')
        match = re.match(r'^Speaker\s+(\S+)', content)
        if match:
            canonical_names.add(match.group(1))

    nickname_map: Dict[str, str] = {}
    for name in canonical_names:
        name_lower = name.lower()
        nickname_map[name_lower] = name
        # Generate prefix-based nicknames (3+ chars)
        for length in range(3, len(name)):
            prefix = name_lower[:length]
            # Only add if unambiguous (no other canonical name shares this prefix)
            conflicts = [n for n in canonical_names if n.lower().startswith(prefix) and n != name]
            if not conflicts:
                nickname_map[prefix] = name

    return nickname_map


def normalize_participants(participants: List[str], nickname_map: Dict[str, str]) -> List[str]:
    """Normalize participant names using the nickname map."""
    normalized = []
    for p in participants:
        canonical = nickname_map.get(p.lower())
        if canonical:
            normalized.append(canonical)
        else:
            normalized.append(p)
    return sorted(set(normalized))


class EventExtractor:
    """Extracts events and typed edges from memory notes using LLM."""

    def __init__(self, llm, extraction_temperature: float = 0.0,
                 cross_note_temperature: float = 0.1,
                 nickname_map: Optional[Dict[str, str]] = None):
        """
        Args:
            llm: A RobustBaseLLMController instance (has .get_completion(prompt, temperature)).
            extraction_temperature: Temperature for intra-note extraction.
            cross_note_temperature: Temperature for cross-note edge inference.
            nickname_map: Optional mapping of lowercase nickname → canonical name.
        """
        self.llm = llm
        self.extraction_temperature = extraction_temperature
        self.cross_note_temperature = cross_note_temperature
        self.nickname_map = nickname_map or {}

    def extract_from_note(self, note) -> Tuple[List[EventNode], List[TypedEdge]]:
        """Extract events and intra-note edges from a single memory note.

        Args:
            note: A RobustMemoryNote object with .content, .context, .timestamp,
                  .keywords, .id attributes.

        Returns:
            (events, edges) extracted from this note.
        """
        prompt = format_extraction_prompt(
            content=note.content,
            context=getattr(note, 'context', 'General'),
            timestamp=getattr(note, 'timestamp', 'unknown'),
            keywords=getattr(note, 'keywords', []),
        )

        try:
            response = self.llm.get_completion(prompt, temperature=self.extraction_temperature)
        except Exception as e:
            logger.error(f"LLM call failed for note {note.id}: {e}")
            return [], []

        events, edges = parse_extraction_response(
            response=response,
            note_id=note.id,
            note_content=note.content,
        )

        # Normalize participant nicknames to canonical names
        if self.nickname_map:
            for event in events:
                event.participants = normalize_participants(event.participants, self.nickname_map)

        if events:
            logger.debug(f"Note {note.id}: extracted {len(events)} events, {len(edges)} edges")

        return events, edges

    def infer_cross_note_edges(
        self,
        new_events: List[EventNode],
        graph: CausalGraph,
        min_jaccard: float = 0.5,
        max_candidates: int = 5,
        allowed_event_ids: Optional[Set[str]] = None,
    ) -> List[TypedEdge]:
        """Infer edges between new events and existing graph events.

        For each new event:
        1. Auto-infer temporal_before from absolute timestamps (no LLM call).
        2. Find candidates by participant Jaccard overlap.
        3. One LLM call per new event with batched candidates.

        Args:
            new_events: Events just extracted from a note.
            graph: The existing causal graph.
            min_jaccard: Minimum Jaccard similarity for candidate selection.
            max_candidates: Maximum candidates per new event.
            allowed_event_ids: If provided, only consider candidates in this set
                (for topic-guided edge inference).

        Returns:
            List of new TypedEdge objects.
        """
        all_edges: List[TypedEdge] = []
        new_event_ids = {e.event_id for e in new_events}

        for new_event in new_events:
            # 1. Auto-infer temporal_before from absolute timestamps
            auto_edges = self._auto_temporal_edges(
                new_event, graph, new_event_ids,
                allowed_event_ids=allowed_event_ids,
            )
            all_edges.extend(auto_edges)

            # 2. Find candidates by participant overlap
            candidates = graph.get_events_by_participant_overlap(
                participants=new_event.participants,
                min_jaccard=min_jaccard,
                max_k=max_candidates,
                exclude_ids=new_event_ids,
            )

            # Topic-guided filtering
            if allowed_event_ids is not None:
                candidates = [c for c in candidates if c.event_id in allowed_event_ids]

            if not candidates:
                continue

            # 3. LLM call for cross-note edges
            prompt = format_cross_note_prompt(new_event, candidates)
            try:
                response = self.llm.get_completion(
                    prompt, temperature=self.cross_note_temperature
                )
            except Exception as e:
                logger.error(f"Cross-note LLM call failed for event {new_event.event_id}: {e}")
                continue

            candidate_ids = [c.event_id for c in candidates]
            edges = parse_cross_note_response(response, new_event.event_id, candidate_ids)
            all_edges.extend(edges)

        return all_edges

    def _auto_temporal_edges(
        self,
        new_event: EventNode,
        graph: CausalGraph,
        exclude_ids: set,
        allowed_event_ids: Optional[Set[str]] = None,
    ) -> List[TypedEdge]:
        """Auto-create temporal_before edges from parseable absolute timestamps.

        No LLM call needed — pure date comparison.
        """
        new_date = try_parse_date(new_event.time_anchor)
        if new_date is None:
            return []

        edges = []
        # Check events with overlapping participants
        candidates = graph.get_events_by_participant_overlap(
            participants=new_event.participants,
            min_jaccard=0.3,
            max_k=20,
            exclude_ids=exclude_ids,
        )

        # Topic-guided filtering
        if allowed_event_ids is not None:
            candidates = [c for c in candidates if c.event_id in allowed_event_ids]

        for candidate in candidates:
            cand_date = try_parse_date(candidate.time_anchor)
            if cand_date is None:
                continue

            if cand_date < new_date:
                # candidate happened before new event
                edge = TypedEdge(
                    source_event_id=candidate.event_id,
                    target_event_id=new_event.event_id,
                    edge_type="temporal_before",
                    confidence=1.0,
                    reason=f"Auto-inferred: {candidate.time_anchor} < {new_event.time_anchor}",
                )
                edges.append(edge)
            elif new_date < cand_date:
                edge = TypedEdge(
                    source_event_id=new_event.event_id,
                    target_event_id=candidate.event_id,
                    edge_type="temporal_before",
                    confidence=1.0,
                    reason=f"Auto-inferred: {new_event.time_anchor} < {candidate.time_anchor}",
                )
                edges.append(edge)

        return edges
