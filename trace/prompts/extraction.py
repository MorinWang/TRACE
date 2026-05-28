"""Prompt template and parser for intra-note event + edge extraction.

Extracts structured events and typed edges from a single memory note via LLM.
"""

import json
import logging
from typing import List, Tuple

from trace.parsing_utils import strip_markdown_fences, extract_json_block
from trace.event_schema import (
    EventNode, TypedEdge,
    VALID_EVENT_TYPES, VALID_EDGE_TYPES,
    validate_event_type, validate_edge_type,
)

logger = logging.getLogger(__name__)

# Prompt version for cache invalidation tracking
PROMPT_VERSION = "extraction_v1.0"

EXTRACTION_PROMPT = """\
Extract structured events from the following memory note. An event is a discrete \
action, state change, preference expression, or plan mentioned in the text.

## Memory Note
- Content: {content}
- Context: {context}
- Timestamp: {timestamp}
- Keywords: {keywords}

## Instructions
For each distinct event mentioned, extract:
- event_type: MUST be exactly one of these four values: "action", "state_change", "preference", "plan". Do NOT use any other type (e.g., do not use "inquiry", "question", "observation"). Map questions/inquiries to "action"
- participants: list of person names involved (use full canonical names, resolve pronouns like "he/she" to actual names if possible)
- time_anchor: when the event occurred or will occur. Use the absolute date if mentioned, otherwise use the note timestamp or a relative expression
- state_change: a concise one-sentence description of what happened or changed

If multiple events are found in the same note, also identify causal/temporal relationships between them:
- edge_type: one of "causes", "enables", "prevents", "temporal_before"
- confidence: float 0.0-1.0 indicating how confident you are about this relationship
- reason: one brief sentence explaining why this relationship holds

## Output Format
Respond with ONLY a JSON object in this exact format (no other text):
{{
  "events": [
    {{
      "event_type": "action",
      "participants": ["Alice"],
      "time_anchor": "2024-03-15",
      "state_change": "Alice started a new job at Google"
    }}
  ],
  "edges": [
    {{
      "source_event_idx": 0,
      "target_event_idx": 1,
      "edge_type": "causes",
      "confidence": 0.85,
      "reason": "Getting the job offer caused the relocation"
    }}
  ]
}}

If no clear events can be extracted, return: {{"events": [], "edges": []}}

## Example
Input note content: "Speaker Alice says: I finally moved to Shanghai last month! The new job at Tencent starts next week."
Output:
{{
  "events": [
    {{
      "event_type": "action",
      "participants": ["Alice"],
      "time_anchor": "last month",
      "state_change": "Alice moved to Shanghai"
    }},
    {{
      "event_type": "plan",
      "participants": ["Alice"],
      "time_anchor": "next week",
      "state_change": "Alice will start a new job at Tencent"
    }}
  ],
  "edges": [
    {{
      "source_event_idx": 0,
      "target_event_idx": 1,
      "edge_type": "enables",
      "confidence": 0.8,
      "reason": "Moving to Shanghai enables starting the new job there"
    }}
  ]
}}
"""


def format_extraction_prompt(
    content: str,
    context: str,
    timestamp: str,
    keywords: List[str],
) -> str:
    return EXTRACTION_PROMPT.format(
        content=content,
        context=context or "General",
        timestamp=timestamp or "unknown",
        keywords=", ".join(keywords) if keywords else "none",
    )


def parse_extraction_response(
    response: str,
    note_id: str,
    note_content: str = "",
) -> Tuple[List[EventNode], List[TypedEdge]]:
    """Parse LLM extraction response into EventNode and TypedEdge lists.

    Args:
        response: Raw LLM response text.
        note_id: The source memory note ID.
        note_content: Source note content for provenance.

    Returns:
        Tuple of (events, edges). On parse failure, returns ([], []).
    """
    if not response:
        logger.warning(f"Empty/None response for note {note_id}")
        return [], []

    # Try parsing JSON
    data = None
    try:
        cleaned = strip_markdown_fences(response)
        data = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        try:
            block = extract_json_block(response)
            data = json.loads(block)
        except (json.JSONDecodeError, ValueError):
            logger.warning(f"Failed to parse extraction response for note {note_id}")
            return [], []

    if not isinstance(data, dict):
        logger.warning(f"Extraction response is not a dict for note {note_id}")
        return [], []

    # Parse events
    raw_events = data.get("events", [])
    if not isinstance(raw_events, list):
        raw_events = []

    events: List[EventNode] = []
    for i, raw in enumerate(raw_events):
        if not isinstance(raw, dict):
            continue

        event_type = raw.get("event_type", "action")
        if not validate_event_type(event_type):
            # Try to map common variants
            event_type_lower = event_type.lower().replace(" ", "_")
            if event_type_lower in VALID_EVENT_TYPES:
                event_type = event_type_lower
            else:
                logger.warning(f"Invalid event_type '{event_type}' in note {note_id}, defaulting to 'action'")
                event_type = "action"

        participants = raw.get("participants", [])
        if isinstance(participants, str):
            participants = [participants]
        participants = [str(p).strip() for p in participants if p]

        event = EventNode(
            event_id=EventNode.generate_id(),
            event_type=event_type,
            participants=participants,
            time_anchor=str(raw.get("time_anchor", "unknown")),
            state_change=str(raw.get("state_change", "")),
            provenance=note_content[:200] if note_content else "",
            source_note_ids=[note_id],
        )
        events.append(event)

    # Parse edges (intra-note only)
    raw_edges = data.get("edges", [])
    if not isinstance(raw_edges, list):
        raw_edges = []

    edges: List[TypedEdge] = []
    for raw in raw_edges:
        if not isinstance(raw, dict):
            continue

        src_idx = raw.get("source_event_idx")
        tgt_idx = raw.get("target_event_idx")
        edge_type = raw.get("edge_type", "")

        # Validate indices
        if not isinstance(src_idx, int) or not isinstance(tgt_idx, int):
            continue
        if src_idx < 0 or src_idx >= len(events) or tgt_idx < 0 or tgt_idx >= len(events):
            continue
        if src_idx == tgt_idx:
            continue

        if not validate_edge_type(edge_type):
            continue

        confidence = raw.get("confidence", 0.5)
        try:
            confidence = float(confidence)
        except (ValueError, TypeError):
            confidence = 0.5

        edge = TypedEdge(
            source_event_id=events[src_idx].event_id,
            target_event_id=events[tgt_idx].event_id,
            edge_type=edge_type,
            confidence=confidence,
            reason=str(raw.get("reason", "")),
        )
        edges.append(edge)

    return events, edges
