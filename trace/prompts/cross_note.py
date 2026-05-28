"""Prompt template and parser for cross-note edge inference.

Given a new event and candidate existing events, determines typed edges
between them via LLM.
"""

import json
import logging
from typing import List, Tuple

from trace.event_schema import EventNode, TypedEdge, validate_edge_type
from trace.parsing_utils import strip_markdown_fences, extract_json_block

logger = logging.getLogger(__name__)

PROMPT_VERSION = "cross_note_v2.0"

CROSS_NOTE_EDGE_PROMPT = """\
Given a new event and existing events from the memory graph, identify any \
causal, temporal, update, or contradiction relationships between them.

## New Event
- Type: {new_type}
- Participants: {new_participants}
- Time: {new_time}
- Description: {new_state_change}

## Existing Events
{candidates_block}

## Instructions
For each relationship found between the new event and any existing event, output:
- existing_event_idx: index of the existing event (0-based)
- direction: "new_to_existing" (new event affects existing) or "existing_to_new" (existing affects new)
- edge_type: one of "causes", "enables", "prevents", "temporal_before", "updates", "contradicts"
- confidence: float 0.0-1.0
- reason: one brief sentence

Edge type definitions (choose carefully):
- causes: A DIRECTLY caused B. Without A, B would not have happened.
- enables: A is a NECESSARY precondition for B — B could NOT have happened without A. \
Do NOT use "enables" for vague thematic connections, shared topics, or events that merely \
involve the same person. Two events about the same person are NOT automatically related.
- prevents: A prevents or blocks B from happening.
- temporal_before: A happened before B. Only use when there is a clear, specific time ordering.
- updates: A provides newer information that replaces or supersedes B's state. \
Look for the same attribute (location, job, preference, status) changing over time.
- contradicts: A and B assert incompatible facts about the same thing.

IMPORTANT:
- Most event pairs have NO relationship. Return empty edges if no clear relationship exists.
- Do NOT create edges between events just because they share participants or topics.
- "updates" and "contradicts" are high-value — actively look for these when the same \
attribute appears in both events with different values.
- Only report relationships you are confident about (confidence >= 0.7).

## Output Format
Respond with ONLY a JSON object:
{{
  "edges": [
    {{
      "existing_event_idx": 0,
      "direction": "existing_to_new",
      "edge_type": "causes",
      "confidence": 0.85,
      "reason": "The job change caused the relocation"
    }}
  ]
}}

If no relationships exist, return: {{"edges": []}}
"""


def format_candidates_block(candidates: List[EventNode]) -> str:
    lines = []
    for i, c in enumerate(candidates):
        lines.append(
            f"[{i}] Type: {c.event_type} | Participants: {', '.join(c.participants)} "
            f"| Time: {c.time_anchor} | Description: {c.state_change}"
        )
    return "\n".join(lines)


def format_cross_note_prompt(new_event: EventNode, candidates: List[EventNode]) -> str:
    return CROSS_NOTE_EDGE_PROMPT.format(
        new_type=new_event.event_type,
        new_participants=", ".join(new_event.participants),
        new_time=new_event.time_anchor,
        new_state_change=new_event.state_change,
        candidates_block=format_candidates_block(candidates),
    )


def parse_cross_note_response(
    response: str,
    new_event_id: str,
    candidate_ids: List[str],
) -> List[TypedEdge]:
    """Parse cross-note edge inference response.

    Args:
        response: Raw LLM response.
        new_event_id: ID of the new event.
        candidate_ids: IDs of candidate events (indexed by position).

    Returns:
        List of TypedEdge objects.
    """
    if not response:
        logger.warning(f"Empty/None response for cross-note event {new_event_id}")
        return []

    data = None
    try:
        cleaned = strip_markdown_fences(response)
        data = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        try:
            block = extract_json_block(response)
            data = json.loads(block)
        except (json.JSONDecodeError, ValueError):
            logger.warning(f"Failed to parse cross-note response for event {new_event_id}")
            return []

    if not isinstance(data, dict):
        return []

    raw_edges = data.get("edges", [])
    if not isinstance(raw_edges, list):
        return []

    edges: List[TypedEdge] = []
    for raw in raw_edges:
        if not isinstance(raw, dict):
            continue

        idx = raw.get("existing_event_idx")
        if not isinstance(idx, int) or idx < 0 or idx >= len(candidate_ids):
            continue

        direction = raw.get("direction", "existing_to_new")
        edge_type = raw.get("edge_type", "")
        if not validate_edge_type(edge_type):
            continue

        confidence = raw.get("confidence", 0.5)
        try:
            confidence = float(confidence)
        except (ValueError, TypeError):
            confidence = 0.5

        if confidence < 0.7:
            continue

        existing_id = candidate_ids[idx]

        if direction == "new_to_existing":
            src, tgt = new_event_id, existing_id
        else:
            src, tgt = existing_id, new_event_id

        edge = TypedEdge(
            source_event_id=src,
            target_event_id=tgt,
            edge_type=edge_type,
            confidence=confidence,
            reason=str(raw.get("reason", "")),
        )
        edges.append(edge)

    return edges
