"""Incremental ingestion of a newly arrived session into an existing evidence graph.

How a newly arrived session is *inserted*, *linked*, and *propagated* through the
validity annotations.

Relationship to `build_graph_locomo.py`: that builder runs Phase 1 over
every session, then Phase 2 over every session, then Phase 3 -- so by the time
supersession is detected the graph already holds events from sessions that had
not yet arrived. This module runs the same four steps for a *single* arriving
session against a graph that holds only the ingested history. The components,
thresholds and caps are the ones the batch builder uses; `scripts/
verify_incremental_parity.py` checks that correspondence.

Scope: sessions arrive in chronological order and are appended. Out-of-order and
backfilled arrival are out of scope.

Topic assignment is the one component that is approximated rather than
reproduced: global clustering has no incremental form, so an arriving session
joins the nearest existing topic centroid and repeated drift triggers a periodic
recluster. See `TopicCentroidAssigner`.
"""

from __future__ import annotations

import logging
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Set

import numpy as np

from trace.causal_graph import CausalGraph
from trace.event_schema import EventNode, TypedEdge

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gate parameters
#
# These mirror the batch builder. They live here as named constants so that the
# parity checker can compare them against `UpdateDetector.detect`'s defaults and
# against the values `build_graph_locomo` passes, rather than relying on
# the two code paths happening to agree.
# ---------------------------------------------------------------------------

#: participant Jaccard threshold admitting a supersession candidate
SUPERSESSION_MIN_JACCARD = 0.5
#: hard per-event cap on supersession candidates (the binding constraint)
SUPERSESSION_MAX_CANDIDATES = 10
#: participant Jaccard threshold for cross-session support linking
SUPPORT_MIN_JACCARD = 0.5
#: per-event cap for cross-session support linking
SUPPORT_MAX_CANDIDATES = 5

# ---------------------------------------------------------------------------
# Topic assignment parameters
#
# Unlike the four gates above, these have no batch counterpart to mirror:
# global LLM clustering has no incremental form.
# ---------------------------------------------------------------------------

#: target sessions per topic, used to size k when reclustering. The batch graphs
#: run at 272 sessions / 50 topics = 5.4, with 3-7 topics per conversation.
TOPIC_TARGET_SESSIONS = 5
#: a topic holding more than this many times the target has outgrown what one
#: centroid can describe, and forces a recluster
TOPIC_OVERSIZE_FACTOR = 2
#: drift arrivals that trigger a recluster. Three sessions in a row that fit
#: their topic worse than any current member means the centroids have stopped
#: describing the conversation.
TOPIC_RECLUSTER_AFTER_DRIFTS = 3
#: sentence encoder backing the centroids -- the model the rest of the pipeline
#: already uses (`trace/embedding_retriever.py`)
TOPIC_ENCODER_MODEL = "all-MiniLM-L6-v2"


@dataclass
class IngestReport:
    """What ingesting one session cost and did."""

    session_id: str
    #: position in arrival order
    arrival_index: int = -1
    topic_id: str = ""
    #: cosine to the centroid of the topic the session joined; -1.0 for the
    #: first session of a conversation
    topic_cosine: float = -1.0
    #: the session opened the conversation's first topic
    topic_opened: bool = False
    #: the session fit its topic worse than every member already in it
    topic_drifted: bool = False
    #: this arrival triggered a recluster of the topic layer
    topic_reclustered: bool = False
    events_added: int = 0
    intra_edges: int = 0
    support_edges: int = 0
    update_edges: int = 0
    contradiction_edges: int = 0
    #: events passing the participant predicate, before the per-event cap.
    #: This is the quantity that grows with history length.
    candidate_pool: int = 0
    #: pairs actually handed to the pairwise classifier, after the cap.
    #: This is the quantity that stays flat -- the marginal cost of the session.
    candidates_screened: int = 0
    #: invariants breached during this ingest; empty means the run was clean
    violations: List[str] = field(default_factory=list)


class _SessionNote:
    """The note shim `EventExtractor.extract_from_note` expects.

    Constructed exactly as `build_graph_locomo` Phase 1 constructs it, so
    the extractor sees a byte-identical input for the same session.
    """

    __slots__ = ("content", "context", "timestamp", "keywords", "id")

    def __init__(self, content: str, context: str, timestamp: str, note_id: str):
        self.content = content
        self.context = context
        self.timestamp = timestamp
        self.keywords = []
        self.id = note_id


@dataclass
class TopicAssignment:
    """Where an arriving session was filed, and what it cost the topic layer."""

    topic_id: str
    #: cosine to the winning centroid; -1.0 when no topic existed yet
    cosine: float = -1.0
    #: the session opened the conversation's first topic
    opened: bool = False
    #: the session fit its topic worse than every member already in it
    drifted: bool = False
    #: this arrival triggered a recluster of the whole topic layer
    reclustered: bool = False


@lru_cache(maxsize=1)
def _default_encoder():
    """The pipeline's sentence encoder, loaded once per process."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(TOPIC_ENCODER_MODEL)


class TopicCentroidAssigner:
    """Files an arriving session under the nearest existing topic centroid.

    Topic assignment is the one step of the batch builder with no incremental
    form: `trace/topic_clusterer.py` clusters by putting every session of the
    conversation into a single LLM prompt, which is not available when only the
    history so far has arrived. The incremental stand-in keeps a centroid per
    topic -- the mean of its member session-summary embeddings -- and admits an
    arriving session to the nearest one. Periodically the whole layer is
    re-derived from every session embedding seen so far.

    There is deliberately **no absolute cosine floor** on admission. Within one
    conversation the session summaries share speakers, register and domain, so
    their pairwise cosines sit in a narrow band, and any floor low enough to
    admit a genuine continuation admits everything. So admission is
    unconditional -- the arriving session always joins its nearest centroid --
    and the correction is deferred to the recluster, which is triggered by two
    signals that need no constant of their own:

    * **semantic drift** -- the arriving session fits its topic worse than every
      current member does. Measured against the topic's own spread, so it scales
      with whatever band that conversation happens to occupy.
    * **structural drift** -- a topic has grown past `oversize_factor` times the
      target size, i.e. one centroid is being asked to describe more than a
      topic's worth of sessions.

    Two properties matter for the rest of the ingest:

    * No LLM call, and deterministic. The encoder is local and the k-means is
      seeded, so replaying one arrival order twice yields the same topic layer.
    * The rewrite is confined to Level 2. A recluster re-files sessions and
      rewrites `belongs_to_topic` / `topic_contains` edges; it never touches an
      event, an event-level edge, or a validity annotation. Topic membership is
      read by the cross-session *support* gate only, so a rewrite changes what
      later sessions may link to and nothing already asserted.
    """

    def __init__(
        self,
        encoder=None,
        target_sessions_per_topic: int = TOPIC_TARGET_SESSIONS,
        recluster_after_drifts: int = TOPIC_RECLUSTER_AFTER_DRIFTS,
        oversize_factor: int = TOPIC_OVERSIZE_FACTOR,
    ):
        self._encoder = encoder
        self.target_sessions_per_topic = target_sessions_per_topic
        self.recluster_after_drifts = recluster_after_drifts
        self.oversize_factor = oversize_factor

        #: session_id -> unit-norm summary embedding, in arrival order
        self._embeddings: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._centroids: Dict[str, np.ndarray] = {}
        self._members: Dict[str, List[str]] = {}
        self._drifts = 0
        self._generation = 0
        self._opened = 0

    # -- encoding ----------------------------------------------------------

    def _embed(self, text: str) -> np.ndarray:
        encoder = self._encoder or _default_encoder()
        vec = np.asarray(encoder.encode([text or ""])[0], dtype=float)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm else vec

    def _centroid_of(self, session_ids: Sequence[str]) -> np.ndarray:
        vecs = [self._embeddings[s] for s in session_ids if s in self._embeddings]
        mean = np.mean(np.vstack(vecs), axis=0)
        norm = float(np.linalg.norm(mean))
        return mean / norm if norm else mean

    # -- entry point -------------------------------------------------------

    def assign(
        self,
        graph: CausalGraph,
        sample_idx: int,
        session_id: str,
        summary_text: str,
    ) -> TopicAssignment:
        """File one arriving session. Modifies `graph`'s topic layer."""
        emb = self._embed(summary_text)
        self._embeddings[session_id] = emb

        if not self._centroids:
            return TopicAssignment(
                topic_id=self._open(graph, sample_idx, session_id), opened=True
            )

        best_tid, best_cos = "", -1.0
        for tid, centroid in self._centroids.items():
            cos = float(np.dot(emb, centroid))
            if cos > best_cos:
                best_tid, best_cos = tid, cos

        # Semantic drift is judged before admission, against the topic as it
        # stands: does the arrival fit it worse than every session already in it?
        drifted = best_cos < self._cohesion(best_tid)
        self._admit(graph, best_tid, session_id)

        if drifted:
            self._drifts += 1
        oversized = (
            len(self._members.get(best_tid, ()))
            > self.oversize_factor * self.target_sessions_per_topic
        )

        result = TopicAssignment(topic_id=best_tid, cosine=best_cos, drifted=drifted)
        if oversized or self._drifts >= self.recluster_after_drifts:
            self._recluster(graph, sample_idx)
            result.reclustered = True
            result.topic_id = self._topic_of(session_id) or best_tid
        return result

    def _cohesion(self, topic_id: str) -> float:
        """How well the *weakest* current member fits this topic's centroid.

        The admission bar, expressed in the topic's own terms rather than as a
        global constant. A single-member topic has no spread to compare against,
        so it admits anything (-1.0).
        """
        members = [s for s in self._members.get(topic_id, ()) if s in self._embeddings]
        if len(members) < 2:
            return -1.0
        centroid = self._centroids[topic_id]
        return min(float(np.dot(self._embeddings[s], centroid)) for s in members)

    def _topic_of(self, session_id: str) -> Optional[str]:
        for tid, members in self._members.items():
            if session_id in members:
                return tid
        return None

    # -- the three mutations -----------------------------------------------

    def _admit(self, graph: CausalGraph, topic_id: str, session_id: str) -> None:
        """Extend an existing topic by one session.

        `CausalGraph.add_topic` returns early for a topic that already exists, so
        the per-session effects it would have applied -- star edges in both
        directions, hyperedge member list -- are applied here instead.
        `causal_graph.py` is untouched.
        """
        topic = graph._topics.get(topic_id)
        if topic is None:
            return
        members = self._members.setdefault(topic_id, list(topic.get("session_ids", [])))
        if session_id in members:
            return

        members.append(session_id)
        topic["session_ids"] = list(members)
        if session_id in graph._sessions:
            graph.graph.add_edge(
                session_id, topic_id, edge_type="belongs_to_topic", confidence=1.0
            )
            graph.graph.add_edge(
                topic_id, session_id, edge_type="topic_contains", confidence=1.0
            )
        hyper = graph._hyperedges.get(topic_id)
        if hyper is not None and session_id not in hyper.member_ids:
            hyper.member_ids.append(session_id)

        self._centroids[topic_id] = self._centroid_of(members)
        topic["label"] = self._label_for(graph, members)

    def _open(self, graph: CausalGraph, sample_idx: int, session_id: str) -> str:
        """Start a new topic holding just this session."""
        tid = f"topic_{sample_idx}_{self._generation}_{self._opened}"
        self._opened += 1
        self._members[tid] = [session_id]
        graph.add_topic(
            {
                "topic_id": tid,
                "label": self._label_for(graph, [session_id]),
                "session_ids": [session_id],
                "description": "Incremental topic: sessions admitted by centroid "
                               "similarity to their session summaries.",
            }
        )
        self._centroids[tid] = self._centroid_of([session_id])
        return tid

    def _recluster(self, graph: CausalGraph, sample_idx: int) -> List[str]:
        """Re-derive the topic layer from every session embedding seen so far.

        k comes from the target sessions-per-topic ratio rather than a search,
        and the k-means is seeded, so the rewrite is reproducible.
        """
        session_ids = [s for s in self._embeddings if s in graph._sessions]
        if len(session_ids) < 2:
            self._drifts = 0
            return []

        from sklearn.cluster import KMeans

        k = max(
            1,
            min(
                len(session_ids),
                round(len(session_ids) / self.target_sessions_per_topic),
            ),
        )
        matrix = np.vstack([self._embeddings[s] for s in session_ids])
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(matrix)

        for tid in list(self._centroids):
            self._detach(graph, tid)
        self._generation += 1

        grouped: Dict[int, List[str]] = defaultdict(list)
        for sid, label in zip(session_ids, labels):
            grouped[int(label)].append(sid)  # arrival order kept within a cluster

        new_ids: List[str] = []
        for label in sorted(grouped):
            members = grouped[label]
            tid = f"topic_{sample_idx}_{self._generation}_{label}"
            self._members[tid] = list(members)
            graph.add_topic(
                {
                    "topic_id": tid,
                    "label": self._label_for(graph, members),
                    "session_ids": list(members),
                    "description": "Incremental topic, re-derived at recluster "
                                   f"{self._generation}.",
                }
            )
            self._centroids[tid] = self._centroid_of(members)
            new_ids.append(tid)

        # keep `_open`'s counter clear of the ids this recluster just claimed
        self._opened = len(new_ids)
        self._drifts = 0
        logger.info(
            "Reclustered %d sessions into %d topics (generation %d)",
            len(session_ids),
            len(new_ids),
            self._generation,
        )
        return new_ids

    def _detach(self, graph: CausalGraph, topic_id: str) -> None:
        """Drop a topic node, its star edges, and its hyperedge view.

        `CausalGraph` has no `remove_topic` and is a read-only dependency here,
        so the removal goes through the same three structures `add_topic` writes:
        the `_topics` dict, the `_hyperedges` index, and the NetworkX node --
        whose removal takes the incident star edges with it.
        """
        graph._topics.pop(topic_id, None)
        graph._hyperedges.pop(topic_id, None)
        self._centroids.pop(topic_id, None)
        self._members.pop(topic_id, None)
        if graph.graph.has_node(topic_id):
            graph.graph.remove_node(topic_id)

    # -- labelling ---------------------------------------------------------

    def _label_for(self, graph: CausalGraph, session_ids: Sequence[str]) -> str:
        """A deterministic stand-in for the batch builder's LLM topic label.

        The member closest to the centroid names the topic. Topic scoring in
        `trace/trace_pipeline.py` reads `label` together with `description` and
        the member summaries, so this is a weaker label than the LLM's phrase
        but not a missing one.
        """
        known = [s for s in session_ids if s in self._embeddings]
        if not known:
            return "Incremental topic"
        centroid = self._centroid_of(known)
        medoid = max(known, key=lambda s: float(np.dot(self._embeddings[s], centroid)))
        summary = (graph._sessions.get(medoid) or {}).get("summary", "") or ""
        return summary.strip().split(". ")[0][:80].strip() or "Incremental topic"


class IncrementalIngestor:
    """Ingests one session at a time into a graph that already holds the history."""

    def __init__(
        self,
        extractor,
        update_detector,
        propagator,
        topic_assigner: Optional[TopicCentroidAssigner] = None,
    ):
        self.extractor = extractor
        self.update_detector = update_detector
        self.propagator = propagator
        self.topic_assigner = topic_assigner or TopicCentroidAssigner()

    # -- step 1: insert ----------------------------------------------------

    def _insert(
        self,
        graph: CausalGraph,
        session_data: dict,
        summary_text: str,
        sample_idx: int,
        sess_num: int,
        report: IngestReport,
    ) -> List[EventNode]:
        """Distil the arriving session and add its events and intra edges.

        Local by construction: the extractor sees this session's summary only.
        No previously ingested session is re-read and no previously extracted
        event is recomputed.
        """
        note = _SessionNote(
            content=summary_text,
            context=f"Session {sess_num} conversation summary",
            timestamp=session_data.get("date_time", ""),
            note_id=f"session_{sample_idx}_{sess_num}",
        )

        try:
            events, intra_edges = self.extractor.extract_from_note(note)
        except Exception as exc:  # mirrors the batch builder: skip, do not crash
            logger.error("Extraction failed for %s: %s", session_data.get("session_id"), exc)
            return []

        note_ids = session_data.get("note_ids", [])
        session_date = session_data.get("date_time", "")
        for event in events:
            event.source_note_ids = note_ids
            if not event.time_anchor or event.time_anchor == "unknown":
                event.time_anchor = session_date
            graph.add_event(event)
        for edge in intra_edges:
            graph.add_edge(edge)

        report.events_added = len(events)
        report.intra_edges = len(intra_edges)
        return events

    # -- step 2: link (support) -------------------------------------------

    def _link_support(
        self,
        graph: CausalGraph,
        new_events: Sequence[EventNode],
        topic_id: str,
        report: IngestReport,
    ) -> None:
        """Infer cross-session support edges under the participant + topic gate.

        The topic half of Eq. 4 applies here, and only here.
        """
        allowed = self._topic_event_ids(graph, topic_id)
        allowed.update(e.event_id for e in new_events)

        edges = self.extractor.infer_cross_note_edges(
            list(new_events),
            graph,
            min_jaccard=SUPPORT_MIN_JACCARD,
            max_candidates=SUPPORT_MAX_CANDIDATES,
            allowed_event_ids=allowed,
        )
        for edge in edges:
            graph.add_edge(edge)
        report.support_edges = len(edges)

    # -- step 3: link (evolution) + propagate ------------------------------

    def _link_evolution(
        self,
        graph: CausalGraph,
        new_events: Sequence[EventNode],
        session_event_ids: Set[str],
        ingested_before: Set[str],
        report: IngestReport,
    ) -> None:
        """Detect supersession against ingested history, then propagate validity.

        The candidate admission is participant overlap + the per-event cap +
        `valid_until is None` + same-session exclusion. There is no topic filter
        on this path -- that is the batch builder's behaviour too.
        """
        for event in new_events:
            # Observed copy of the screening the detector performs internally.
            # Recomputed here rather than inferred, so the invariants below are
            # checked against the same predicate the classifier sees.
            pool = graph.get_events_by_participant_overlap(
                participants=event.participants,
                min_jaccard=SUPERSESSION_MIN_JACCARD,
                max_k=10 ** 9,  # uncapped, to observe the pool the cap acts on
                exclude_ids=set(session_event_ids),
            )
            pool = [c for c in pool if c.valid_until is None]
            report.candidate_pool += len(pool)

            screened = graph.get_events_by_participant_overlap(
                participants=event.participants,
                min_jaccard=SUPERSESSION_MIN_JACCARD,
                max_k=SUPERSESSION_MAX_CANDIDATES,
                exclude_ids=set(session_event_ids),
            )
            screened = [c for c in screened if c.valid_until is None]
            report.candidates_screened += len(screened)

            # Invariant: every comparison is against already-ingested history,
            # so no candidate slot is ever spent on a pair whose direction is
            # temporally inadmissible.
            for cand in screened:
                if cand.event_id not in ingested_before:
                    report.violations.append(
                        f"candidate {cand.event_id} for {event.event_id} is not "
                        f"from already-ingested history"
                    )

            edges = self.update_detector.detect(
                event, graph, exclude_ids=set(session_event_ids)
            )
            for edge in edges:
                if edge.edge_type == "updates":
                    report.update_edges += 1
                elif edge.edge_type == "contradicts":
                    report.contradiction_edges += 1
                graph.add_edge(edge)
            self.propagator.process_edges(edges, graph)

    # -- step 4: hierarchy -------------------------------------------------

    def _attach_hierarchy(
        self,
        graph: CausalGraph,
        session_data: dict,
        new_events: Sequence[EventNode],
        summary_text: str,
        prev_session_id: Optional[str],
        sample_idx: int,
    ) -> TopicAssignment:
        """Add the session node, its chronological link, and file it under a topic.

        Mirrors `build_graph_locomo` Phase 4 for a single session, with
        global topic clustering replaced by centroid assignment.
        """
        sid = session_data["session_id"]
        payload = dict(session_data)
        payload["event_ids"] = [e.event_id for e in new_events]
        graph.add_session(payload)

        if prev_session_id is not None and prev_session_id in graph._sessions:
            graph.graph.add_edge(
                prev_session_id, sid, edge_type="temporal_before", confidence=1.0
            )

        return self.topic_assigner.assign(graph, sample_idx, sid, summary_text)

    # -- topic membership --------------------------------------------------

    def _topic_event_ids(self, graph: CausalGraph, topic_id: str) -> Set[str]:
        """Event ids of every session already filed under `topic_id`."""
        out: Set[str] = set()
        topic = graph._topics.get(topic_id)
        if not topic:
            return out
        for sid in topic.get("session_ids", []):
            session = graph._sessions.get(sid) or {}
            out.update(session.get("event_ids", []))
        return out

    # -- entry point -------------------------------------------------------

    def ingest_session(
        self,
        graph: CausalGraph,
        *,
        sample_idx: int,
        session_data: dict,
        summary_text: str,
        sess_num: int,
        session_index: int,
        prev_session_id: Optional[str] = None,
    ) -> IngestReport:
        """Insert, link, and propagate one arriving session. Modifies `graph`.

        `session_index` is the position in arrival order, recorded on the report.
        `prev_session_id` is the previously ingested session, used for the
        chronological session-level edge.
        """
        report = IngestReport(
            session_id=session_data["session_id"], arrival_index=session_index
        )
        ingested_before = set(graph._events.keys())

        new_events = self._insert(
            graph, session_data, summary_text, sample_idx, sess_num, report
        )
        if not new_events:
            return report

        session_event_ids = {e.event_id for e in new_events}

        # Hierarchy first: the support gate needs this session filed under a
        # topic before it can ask which events share that topic.
        assignment = self._attach_hierarchy(
            graph, session_data, new_events, summary_text, prev_session_id, sample_idx
        )
        report.topic_id = assignment.topic_id
        report.topic_cosine = assignment.cosine
        report.topic_opened = assignment.opened
        report.topic_drifted = assignment.drifted
        report.topic_reclustered = assignment.reclustered

        self._link_support(graph, new_events, assignment.topic_id, report)
        self._link_evolution(
            graph, new_events, session_event_ids, ingested_before, report
        )
        return report
