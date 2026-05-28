"""Topic clustering for TRACE nested hypergraph (Method C).

Groups sessions into topical clusters using LLM-based classification.
One API call per sample during injection. Results cached in graph JSON.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TopicNode:
    """A topic auxiliary node grouping related sessions."""
    topic_id: str           # "topic_0_1"
    label: str              # "LGBTQ advocacy and support"
    session_ids: List[str] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TopicNode":
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)


TOPIC_CLUSTERING_PROMPT = """You are given summaries of conversation sessions between two people. Group these sessions by topic/theme.

Sessions:
{session_list}

Instructions:
- Group sessions that discuss the same topic or theme together.
- Each session should belong to exactly one group.
- Create 3-8 groups (fewer if sessions are very similar, more if diverse).
- Give each group a short label (2-5 words) and a one-sentence description.

Output JSON format:
{{"groups": [{{"label": "short topic label", "description": "one sentence describing this topic", "session_ids": ["sess_0_1", "sess_0_5"]}}]}}

Output ONLY valid JSON, no markdown fences."""


def cluster_sessions_llm(
    session_summaries: Dict[str, dict],
    llm,
    sample_idx: int,
    temperature: float = 0.0,
) -> List[TopicNode]:
    """Cluster sessions into topics using a single LLM call.

    Args:
        session_summaries: dict of session_id -> session_data dict
            (from graph._sessions after session injection).
        llm: LLM controller with get_completion() method.
        sample_idx: sample index for topic ID generation.
        temperature: LLM temperature.

    Returns:
        List of TopicNode objects.
    """
    if not session_summaries:
        return []

    # Build session list for prompt
    lines = []
    for sid in sorted(session_summaries.keys()):
        sdata = session_summaries[sid]
        summary = sdata.get("summary", "")[:300]
        date = sdata.get("date_time", "")
        lines.append(f"- {sid} [{date}]: {summary}")
    session_list = "\n".join(lines)

    prompt = TOPIC_CLUSTERING_PROMPT.format(session_list=session_list)

    try:
        response = llm.get_completion(prompt, temperature=temperature)
        topics = _parse_clustering_response(response, sample_idx)
        if topics:
            logger.info(
                f"Topic clustering: {len(session_summaries)} sessions -> "
                f"{len(topics)} topics"
            )
            return topics
    except Exception as e:
        logger.warning(f"Topic clustering failed: {e}")

    return _fallback_single_topic(session_summaries, sample_idx)


def _parse_clustering_response(
    response: str, sample_idx: int
) -> List[TopicNode]:
    """Parse LLM JSON response into TopicNode list."""
    text = response.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    data = json.loads(text)
    groups = data.get("groups", [])

    topics = []
    for i, group in enumerate(groups):
        session_ids = group.get("session_ids", [])
        if not session_ids:
            continue
        topics.append(TopicNode(
            topic_id=f"topic_{sample_idx}_{i}",
            label=group.get("label", f"Topic {i}"),
            session_ids=session_ids,
            description=group.get("description", ""),
        ))
    return topics


def _fallback_single_topic(
    session_summaries: Dict[str, dict], sample_idx: int
) -> List[TopicNode]:
    """Fallback: put all sessions in one topic."""
    return [TopicNode(
        topic_id=f"topic_{sample_idx}_0",
        label="General",
        session_ids=list(session_summaries.keys()),
        description="All sessions (clustering fallback)",
    )]
