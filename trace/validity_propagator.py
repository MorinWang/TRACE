"""TRACE Validity Propagator: Propagates validity changes through the event graph.

When an event is updated or contradicted, propagates validity status to
downstream events in the causal chain.
"""

import logging
from typing import List

from trace.event_schema import EventNode, TypedEdge
from trace.causal_graph import CausalGraph

logger = logging.getLogger(__name__)


class ValidityPropagator:
    """Propagates validity changes when updates/contradictions are detected."""

    def propagate_update(
        self,
        new_event: EventNode,
        old_event: EventNode,
        graph: CausalGraph,
    ):
        """Handle an update relationship: new_event updates old_event.

        Steps:
        1. Mark old_event.valid_until = new_event.time_anchor
        2. Set old_event.update_val = 0.0
        3. Downstream successors via causes/enables → update_val = 0.5
        """
        old_event.valid_until = new_event.time_anchor
        old_event.update_val = 0.0

        logger.info(
            f"Updated event {old_event.event_id}: "
            f"valid_until={old_event.valid_until}, update_val=0.0"
        )

        # P1-2: Invalidate strong causal edges connected to old_event
        # Only causes/prevents — enables ("makes possible") and temporal edges preserved
        invalidate_types = {"causes", "prevents"}
        invalidated_count = 0
        for successor_id in graph.graph.successors(old_event.event_id):
            edge_data = graph.graph.edges[old_event.event_id, successor_id]
            edge_obj = edge_data.get("data")
            if (edge_obj and edge_obj.t_invalid_at is None
                    and edge_data.get("edge_type") in invalidate_types):
                edge_obj.t_invalid_at = new_event.time_anchor
                invalidated_count += 1
        for predecessor_id in graph.graph.predecessors(old_event.event_id):
            edge_data = graph.graph.edges[predecessor_id, old_event.event_id]
            edge_obj = edge_data.get("data")
            if (edge_obj and edge_obj.t_invalid_at is None
                    and edge_data.get("edge_type") in invalidate_types):
                edge_obj.t_invalid_at = new_event.time_anchor
                invalidated_count += 1
        if invalidated_count:
            logger.info(f"P1-2: Invalidated {invalidated_count} causal edges connected to {old_event.event_id}")

        # Propagate to downstream events (successors via causal edges)
        causal_edge_types = {"causes", "enables"}
        for successor_id in graph.graph.successors(old_event.event_id):
            edge_data = graph.graph.edges[old_event.event_id, successor_id]
            if edge_data.get("edge_type") in causal_edge_types:
                successor = graph.get_event(successor_id)
                if successor and successor.update_val > 0.5:
                    successor.update_val = 0.5
                    logger.debug(
                        f"Propagated partial validity to {successor_id}: "
                        f"update_val=0.5"
                    )

    def propagate_contradiction(
        self,
        event_a: EventNode,
        event_b: EventNode,
        graph: CausalGraph,
    ):
        """Handle a contradiction relationship.

        - Neither event is invalidated (no valid_until change)
        - The contradiction edge in the graph serves as the signal
        - Path scoring applies a penalty (path_conf × 0.5)
        """
        logger.info(
            f"Contradiction noted between {event_a.event_id} and "
            f"{event_b.event_id} — no validity change, penalty at path scoring"
        )

    def process_edges(self, edges: List[TypedEdge], graph: CausalGraph):
        """Process a batch of update/contradiction edges.

        Args:
            edges: TypedEdge objects with edge_type "updates" or "contradicts".
            graph: The causal graph containing the events.
        """
        for edge in edges:
            if edge.edge_type == "updates":
                new_event = graph.get_event(edge.source_event_id)
                old_event = graph.get_event(edge.target_event_id)
                if new_event and old_event:
                    self.propagate_update(new_event, old_event, graph)

            elif edge.edge_type == "contradicts":
                event_a = graph.get_event(edge.source_event_id)
                event_b = graph.get_event(edge.target_event_id)
                if event_a and event_b:
                    self.propagate_contradiction(event_a, event_b, graph)
