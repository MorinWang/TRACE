"""TRACE Event Schema: EventNode and TypedEdge dataclasses.

Defines the core data structures for the symbolic causal-temporal event graph.
"""

import uuid
from dataclasses import dataclass, field, asdict
from typing import List, Optional


# ---------------------------------------------------------------------------
# Valid type enums
# ---------------------------------------------------------------------------

VALID_EVENT_TYPES = {"action", "state_change", "preference", "plan"}

VALID_EDGE_TYPES = {
    "causes",           # A causes B
    "enables",          # A makes B possible
    "prevents",         # A prevents B
    "temporal_before",  # A happens before B
    "updates",          # A updates/replaces B's state
    "contradicts",      # A contradicts B
    "belongs_to",       # event -> session node (star expansion)
    "contains",         # session node -> event (star expansion)
    "belongs_to_topic", # session -> topic node (nested hypergraph)
    "topic_contains",   # topic node -> session (nested hypergraph)
}


# ---------------------------------------------------------------------------
# EventNode
# ---------------------------------------------------------------------------

@dataclass
class EventNode:
    """A structured event extracted from a memory note."""
    event_id: str
    event_type: str                     # action | state_change | preference | plan
    participants: List[str]             # canonical names, sorted alphabetically
    time_anchor: str                    # absolute ("2024-03-15") or relative ("last week")
    state_change: str                   # human-readable description of what happened
    provenance: str                     # source note content snippet (for audit)
    source_note_ids: List[str]          # memory note IDs this event was extracted from
    valid_until: Optional[str] = None   # set when a newer event updates this one
    update_val: float = 1.0             # 1.0=valid, 0.5=partial, 0.0=expired

    def __post_init__(self):
        # Sort participants for deterministic comparison
        self.participants = sorted(self.participants)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EventNode":
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)

    @staticmethod
    def generate_id() -> str:
        return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# TypedEdge
# ---------------------------------------------------------------------------

@dataclass
class TypedEdge:
    """A typed directed edge between two events in the causal-temporal graph.

    Edge types are drawn from ``VALID_EDGE_TYPES`` (causes / enables / prevents /
    temporal_before / updates / contradicts, plus the four star-expansion types).
    """
    source_event_id: str
    target_event_id: str
    edge_type: str          # one of VALID_EDGE_TYPES
    confidence: float       # 0.0 - 1.0
    reason: str             # brief justification (for debugging/audit)
    t_invalid_at: Optional[str] = None  # set when edge is invalidated by an update

    def __post_init__(self):
        self.confidence = max(0.0, min(1.0, self.confidence))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TypedEdge":
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_event_type(event_type: str) -> bool:
    return event_type in VALID_EVENT_TYPES


def validate_edge_type(edge_type: str) -> bool:
    return edge_type in VALID_EDGE_TYPES


# ---------------------------------------------------------------------------
# HyperedgeNode (bipartite-incidence representation of TRACE's hierarchical
# hypergraph; star expansion form per Berge 1973; Zhou et al. 2006)
# ---------------------------------------------------------------------------

VALID_HYPEREDGE_TYPES = {"session", "topic"}


@dataclass
class HyperedgeNode:
    """A hyperedge in the bipartite incidence representation of TRACE's
    hierarchical hypergraph.

    Each hyperedge groups >=2 events (or, for topic hyperedges, >=2 sessions)
    under a typed membership relation. Realized via star expansion: `h_id` is
    the auxiliary vertex shared with the corresponding SessionNode/TopicNode
    in CausalGraph (i.e. SessionNode/TopicNode are the realizations of
    HyperedgeNode in the bipartite incidence form).
    """
    h_id: str
    edge_type: str
    member_ids: List[str]
    confidence: float = 1.0

    def __post_init__(self):
        if self.edge_type not in VALID_HYPEREDGE_TYPES:
            raise ValueError(f"Invalid hyperedge type: {self.edge_type}")
        self.confidence = max(0.0, min(1.0, self.confidence))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "HyperedgeNode":
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)
