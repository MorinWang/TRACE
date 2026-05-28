"""TRACE Update Detector: Detects update and contradiction relationships.

Screens candidate events by participant overlap and event type, then uses
LLM to classify as update/contradiction/independent.
"""

import logging
from typing import List, Set

from trace.event_schema import EventNode, TypedEdge
from trace.causal_graph import CausalGraph
from trace.prompts.update import format_update_prompt, parse_update_response

logger = logging.getLogger(__name__)


class UpdateDetector:
    """Detects update and contradiction relationships between events."""

    def __init__(self, llm, temperature: float = 0.0):
        """
        Args:
            llm: A RobustBaseLLMController instance.
            temperature: LLM temperature for update detection.
        """
        self.llm = llm
        self.temperature = temperature

    def detect(
        self,
        new_event: EventNode,
        graph: CausalGraph,
        min_jaccard: float = 0.5,
        exclude_ids: Set[str] = None,
    ) -> List[TypedEdge]:
        """Screen candidates and classify update/contradiction relationships.

        Candidate screening:
        - Participants Jaccard ≥ min_jaccard
        - Same event_type
        - Not already invalidated (valid_until is None)

        Args:
            new_event: The newly extracted event.
            graph: The current causal graph.
            min_jaccard: Minimum Jaccard for candidate selection.
            exclude_ids: Event IDs to exclude from candidates.

        Returns:
            List of TypedEdge objects (updates or contradicts only).
        """
        exclude = exclude_ids or set()
        exclude.add(new_event.event_id)

        # Get candidates with participant overlap
        candidates = graph.get_events_by_participant_overlap(
            participants=new_event.participants,
            min_jaccard=min_jaccard,
            max_k=10,
            exclude_ids=exclude,
        )

        # Filter: not already invalidated (removed same event_type constraint
        # to allow cross-type updates, e.g. action updating a state_change)
        candidates = [
            c for c in candidates
            if c.valid_until is None
        ]

        if not candidates:
            return []

        edges: List[TypedEdge] = []
        for candidate in candidates:
            prompt = format_update_prompt(new_event, candidate)

            try:
                response = self.llm.get_completion(prompt, temperature=self.temperature)
            except Exception as e:
                logger.error(
                    f"Update detection LLM call failed for "
                    f"{new_event.event_id} vs {candidate.event_id}: {e}"
                )
                continue

            relationship, reason = parse_update_response(response)

            if relationship in ("update", "contradiction"):
                edge_type = "updates" if relationship == "update" else "contradicts"
                edge = TypedEdge(
                    source_event_id=new_event.event_id,
                    target_event_id=candidate.event_id,
                    edge_type=edge_type,
                    confidence=0.9,
                    reason=reason,
                )
                edges.append(edge)
                logger.info(
                    f"Detected {edge_type}: '{new_event.state_change}' "
                    f"→ '{candidate.state_change}'"
                )

        return edges
