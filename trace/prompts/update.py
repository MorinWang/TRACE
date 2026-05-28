"""Prompt template and parser for update/contradiction detection.

Determines whether a new event updates, contradicts, or is independent of
an existing event.
"""

import logging
import re
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

PROMPT_VERSION = "update_v1.0"

UPDATE_DETECTION_PROMPT = """\
Determine the relationship between a new event and an existing event that \
involve the same participants.

## New Event (more recent)
- Type: {new_type}
- Participants: {new_participants}
- Time: {new_time}
- Description: {new_state_change}

## Existing Event (older)
- Type: {old_type}
- Participants: {old_participants}
- Time: {old_time}
- Description: {old_state_change}

## Instructions
Classify the relationship as one of:
- "update": The new event updates or replaces the state described in the existing event \
(e.g., moved from city A to city B, changed job, revised a plan)
- "contradiction": The two events describe contradictory information but it's unclear which \
is more authoritative (e.g., different speakers disagree about a fact)
- "independent": The events describe different aspects or topics despite sharing participants

## Output Format
Respond with exactly two lines:
RELATIONSHIP: update
REASON: The new event explicitly states a change from the previous state

(Replace "update" with the appropriate classification.)
"""


def format_update_prompt(new_event, old_event) -> str:
    return UPDATE_DETECTION_PROMPT.format(
        new_type=new_event.event_type,
        new_participants=", ".join(new_event.participants),
        new_time=new_event.time_anchor,
        new_state_change=new_event.state_change,
        old_type=old_event.event_type,
        old_participants=", ".join(old_event.participants),
        old_time=old_event.time_anchor,
        old_state_change=old_event.state_change,
    )


def parse_update_response(response: str) -> Tuple[Optional[str], str]:
    """Parse update detection response.

    Returns:
        (relationship_type, reason) where relationship_type is
        "update", "contradiction", "independent", or None on parse failure.
    """
    if not response:
        return None, ""

    relationship = None
    reason = ""

    for line in response.strip().split("\n"):
        line = line.strip()

        # Match RELATIONSHIP: <value>
        rel_match = re.match(r'^RELATIONSHIP:\s*(.+)$', line, re.IGNORECASE)
        if rel_match:
            rel_val = rel_match.group(1).strip().lower()
            if rel_val in ("update", "contradiction", "independent"):
                relationship = rel_val

        # Match REASON: <value>
        reason_match = re.match(r'^REASON:\s*(.+)$', line, re.IGNORECASE)
        if reason_match:
            reason = reason_match.group(1).strip()

    if relationship is None:
        # Fallback: search for keywords in the response
        resp_lower = response.lower()
        if "update" in resp_lower and "independent" not in resp_lower:
            relationship = "update"
        elif "contradict" in resp_lower:
            relationship = "contradiction"
        elif "independent" in resp_lower:
            relationship = "independent"
        else:
            logger.warning(f"Could not parse update detection response: {response[:200]}")

    return relationship, reason
