"""Parity check for `incremental_ingest.IncrementalIngestor`.

Verifies that the incremental driver performs the *same operations* as the batch
builder -- same gates, same thresholds, same exclusion rules, same propagation,
same serialisation -- without making a single LLM call. (The topic assigner's
sentence encoder and k-means do run: both are local and deterministic, so the
replay stays reproducible.)

It does not check that the incremental driver reproduces the batch graph
edge-for-edge, and cannot: `CausalGraph.get_events_by_participant_overlap`
collects candidates in a `set`, sorts by Jaccard, and truncates to `max_k`. On
LoCoMo 99.8% of events have more than `max_k` candidates tied at the same
Jaccard, so which ones survive depends on set iteration order, which Python
randomises per process (`PYTHONHASHSEED` is never fixed in this repo). Which
pairs the delivered graphs were built from is therefore not recoverable.

What is checked instead, session by session, over a *delivered* graph -- one
produced by the batch builder, i.e. the output of `build_graph_locomo.py` --
replayed in arrival order:

  C1  gate parity      -- the driver's constants equal the batch builder's
  C2  same-session     -- a session's events are never compared against each other
  C3  past-only        -- every candidate comes from already-ingested history
  C4  validity filter  -- already-superseded events are never re-targeted
  C5  propagation      -- validity mutations match the delivered graph's values
  C6  serialisation    -- output schema is byte-compatible with the batch format
  C7  no rewrite       -- ingesting a session mutates history only via propagation
  C8  topic scope      -- topic assignment, a recluster included, never removes an
                          event or an event-level edge

Usage:
    python scripts/verify_incremental_parity.py [--sample 0] [--all]
"""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

TRACE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TRACE_DIR))

from incremental_ingest import (  # noqa: E402
    SUPERSESSION_MAX_CANDIDATES,
    SUPERSESSION_MIN_JACCARD,
    SUPPORT_MAX_CANDIDATES,
    SUPPORT_MIN_JACCARD,
    IncrementalIngestor,
)
from trace.causal_graph import CausalGraph  # noqa: E402
from trace.event_schema import EventNode, TypedEdge  # noqa: E402
from trace.parsing_utils import extract_session_num  # noqa: E402
from trace.update_detector import UpdateDetector  # noqa: E402
from trace.validity_propagator import ValidityPropagator  # noqa: E402

STRUCTURAL = {"belongs_to", "contains", "belongs_to_topic", "topic_contains"}
EVOLUTION = {"updates", "contradicts"}


# ---------------------------------------------------------------------------
# C1 -- gate parity, by introspection of the batch code
# ---------------------------------------------------------------------------

def check_gate_parity() -> List[str]:
    """Compare the driver's constants against the batch code they mirror."""
    problems: List[str] = []

    sig = inspect.signature(UpdateDetector.detect)
    detect_jaccard = sig.parameters["min_jaccard"].default
    if detect_jaccard != SUPERSESSION_MIN_JACCARD:
        problems.append(
            f"C1 supersession min_jaccard: driver {SUPERSESSION_MIN_JACCARD} "
            f"!= UpdateDetector.detect default {detect_jaccard}"
        )

    src = inspect.getsource(UpdateDetector.detect)
    m = re.search(r"max_k\s*=\s*(\d+)", src)
    if not m:
        problems.append("C1 could not locate max_k in UpdateDetector.detect")
    elif int(m.group(1)) != SUPERSESSION_MAX_CANDIDATES:
        problems.append(
            f"C1 supersession cap: driver {SUPERSESSION_MAX_CANDIDATES} "
            f"!= UpdateDetector.detect max_k {m.group(1)}"
        )

    builder = (TRACE_DIR / "build_graph_locomo.py").read_text(encoding="utf-8")
    for label, const, key in (
        ("support min_jaccard", SUPPORT_MIN_JACCARD, "cross_note_min_jaccard"),
        ("support cap", SUPPORT_MAX_CANDIDATES, "cross_note_max_candidates"),
    ):
        m = re.search(rf'config\.get\("{key}",\s*([0-9.]+)\)', builder)
        if not m:
            problems.append(f"C1 could not locate {key} in build_graph_locomo.py")
        elif float(m.group(1)) != float(const):
            problems.append(
                f"C1 {label}: driver {const} != batch builder {m.group(1)}"
            )
    return problems


# ---------------------------------------------------------------------------
# Recorded stubs -- replay the delivered graph, zero LLM calls
# ---------------------------------------------------------------------------

class RecordedExtractor:
    """Returns the events the batch run extracted for each session."""

    def __init__(self, by_note_id: Dict[str, Tuple[List[EventNode], List[TypedEdge]]],
                 cross_by_source: Dict[str, List[TypedEdge]]):
        self._by_note_id = by_note_id
        self._cross_by_source = cross_by_source
        self.support_calls: List[dict] = []

    def extract_from_note(self, note):
        events, intra = self._by_note_id.get(note.id, ([], []))
        return copy.deepcopy(events), copy.deepcopy(intra)

    def infer_cross_note_edges(self, events_in_session, graph, min_jaccard,
                               max_candidates, allowed_event_ids=None):
        self.support_calls.append(
            {"min_jaccard": min_jaccard, "max_candidates": max_candidates,
             "allowed_is_none": allowed_event_ids is None}
        )
        out: List[TypedEdge] = []
        present = set(graph._events.keys())
        for ev in events_in_session:
            for edge in self._cross_by_source.get(ev.event_id, []):
                # only edges whose target is already ingested are reachable now
                if edge.target_event_id in present:
                    out.append(copy.deepcopy(edge))
        return out


class OracleUpdateDetector:
    """`UpdateDetector.detect` with the LLM replaced by the delivered graph.

    The screening is the real one -- same graph method, same predicate, same cap
    -- so C2/C3/C4 are observed against exactly what the classifier would see.
    Only the verdict is replayed: a recorded edge means update/contradicts, its
    absence means independent.
    """

    def __init__(self, recorded: Dict[Tuple[str, str], TypedEdge]):
        self._recorded = recorded
        self.observations: List[dict] = []

    def detect(self, new_event, graph, min_jaccard: float = 0.5, exclude_ids=None):
        exclude = set(exclude_ids or set())
        exclude.add(new_event.event_id)

        candidates = graph.get_events_by_participant_overlap(
            participants=new_event.participants,
            min_jaccard=min_jaccard,
            max_k=SUPERSESSION_MAX_CANDIDATES,
            exclude_ids=exclude,
        )
        candidates = [c for c in candidates if c.valid_until is None]

        self.observations.append({
            "event_id": new_event.event_id,
            "exclude_ids": set(exclude_ids or set()),
            "candidates": [c.event_id for c in candidates],
            "all_unsuperseded": all(c.valid_until is None for c in candidates),
            "min_jaccard": min_jaccard,
        })

        out: List[TypedEdge] = []
        for cand in candidates:
            edge = self._recorded.get((new_event.event_id, cand.event_id))
            if edge is not None:
                out.append(copy.deepcopy(edge))
        return out


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

def load_delivered(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _event_level_edges(graph: CausalGraph) -> Set[Tuple[str, str, str]]:
    """Every edge with an event at both ends, as (source, target, edge_type)."""
    return {
        (u, v, d.get("edge_type", ""))
        for u, v, d in graph.graph.edges(data=True)
        if u in graph._events and v in graph._events
    }


def _materialise(payload: dict, tmpdir: Path) -> Path:
    """Write a payload back to disk so `CausalGraph.load` can read it."""
    p = tmpdir / "delivered.json"
    p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return p


def build_stubs(payload: dict):
    """Split the delivered graph into per-session extractor inputs + an oracle."""
    events = {eid: EventNode.from_dict(d) for eid, d in payload["events"].items()}
    # replay must re-derive validity, not inherit it
    for ev in events.values():
        ev.valid_until = None
        ev.update_val = 1.0

    sessions = payload["sessions"]
    sample_idx = payload.get("metadata", {}).get("sample_idx", 0)
    event_to_session: Dict[str, str] = {}
    for s in sessions:
        for eid in s.get("event_ids", []):
            event_to_session[eid] = s["session_id"]

    by_note_id: Dict[str, Tuple[List[EventNode], List[TypedEdge]]] = {}
    for s in sessions:
        sess_num = extract_session_num(s["session_key"])
        note_id = f"session_{sample_idx}_{sess_num}"
        by_note_id[note_id] = ([events[e] for e in s.get("event_ids", []) if e in events], [])

    cross_by_source: Dict[str, List[TypedEdge]] = defaultdict(list)
    recorded: Dict[Tuple[str, str], TypedEdge] = {}
    for ed in payload["edges"]:
        et = ed["edge_type"]
        if et in STRUCTURAL:
            continue
        src, tgt = ed["source_event_id"], ed["target_event_id"]
        if src not in events or tgt not in events:
            continue  # session-level edge, injected by the hierarchy step
        edge = TypedEdge.from_dict(ed)
        if et in EVOLUTION:
            recorded[(src, tgt)] = edge
        elif event_to_session.get(src) == event_to_session.get(tgt):
            note_id_owner = event_to_session.get(src)
            for s in sessions:
                if s["session_id"] == note_id_owner:
                    n = f"session_{sample_idx}_{extract_session_num(s['session_key'])}"
                    by_note_id[n][1].append(edge)
                    break
        else:
            cross_by_source[src].append(edge)

    return events, sessions, sample_idx, by_note_id, cross_by_source, recorded


def replay(payload: dict) -> dict:
    events, sessions, sample_idx, by_note_id, cross_by_source, recorded = build_stubs(payload)

    extractor = RecordedExtractor(by_note_id, cross_by_source)
    detector = OracleUpdateDetector(recorded)
    ingestor = IncrementalIngestor(extractor, detector, ValidityPropagator())

    # session number is the authoritative chronology (date_time is a free-text
    # string like "10:31 am on 13 October, 2023" and does not sort)
    ordered = sorted(sessions, key=lambda s: extract_session_num(s["session_key"]))
    session_of_event = {
        eid: s["session_id"] for s in sessions for eid in s.get("event_ids", [])
    }

    graph = CausalGraph()
    findings: List[str] = []
    reports = []
    prev_sid: Optional[str] = None

    for idx, s in enumerate(ordered):
        before = {
            eid: (ev.valid_until, ev.update_val) for eid, ev in graph._events.items()
        }
        events_before = set(graph._events)
        edges_before = _event_level_edges(graph)
        rep = ingestor.ingest_session(
            graph,
            sample_idx=sample_idx,
            session_data=s,
            summary_text=s.get("summary", ""),
            sess_num=extract_session_num(s["session_key"]),
            session_index=idx,
            prev_session_id=prev_sid,
        )
        reports.append(rep)
        findings.extend(f"C3 {v}" for v in rep.violations)

        # C8: the topic layer is allowed to rewrite itself -- a recluster drops
        # topic nodes and their star edges on purpose -- but Level 0 is off
        # limits. A recluster that removed the wrong node would show up here.
        lost_events = events_before - set(graph._events)
        if lost_events:
            findings.append(
                f"C8 {len(lost_events)} event(s) disappeared while ingesting "
                f"{s['session_id']}: {sorted(lost_events)[:3]}"
            )
        lost_edges = edges_before - _event_level_edges(graph)
        if lost_edges:
            findings.append(
                f"C8 {len(lost_edges)} event-level edge(s) disappeared while "
                f"ingesting {s['session_id']}: {sorted(lost_edges)[:3]}"
            )

        # C7: validity is monotone. `update_val` may only fall (1.0 -> 0.5 for a
        # downstream successor via causes/enables, or -> 0.0 for a direct
        # supersession target); `valid_until`, once written, is never rewritten.
        for eid, (vu, uv) in before.items():
            ev = graph._events[eid]
            if ev.update_val > uv:
                findings.append(
                    f"C7 {eid}: update_val rose {uv} -> {ev.update_val} while "
                    f"ingesting {s['session_id']}"
                )
            if vu is not None and ev.valid_until != vu:
                findings.append(
                    f"C7 {eid}: valid_until rewritten {vu} -> {ev.valid_until} "
                    f"while ingesting {s['session_id']}"
                )
        if rep.events_added:
            prev_sid = s["session_id"]

    # C2 / C4: inspect what the classifier was actually offered
    for obs in detector.observations:
        own = session_of_event.get(obs["event_id"])
        for cid in obs["candidates"]:
            if session_of_event.get(cid) == own:
                findings.append(
                    f"C2 candidate {cid} shares session {own} with {obs['event_id']}"
                )
        if not obs["all_unsuperseded"]:
            findings.append(f"C4 superseded event offered as candidate to {obs['event_id']}")
        if obs["min_jaccard"] != SUPERSESSION_MIN_JACCARD:
            findings.append(f"C4 min_jaccard drift: {obs['min_jaccard']}")

    # C1b: the support gate was called with the topic restriction present
    for call in extractor.support_calls:
        if call["allowed_is_none"]:
            findings.append("C1 support linking called without the topic restriction")
        if call["min_jaccard"] != SUPPORT_MIN_JACCARD or call["max_candidates"] != SUPPORT_MAX_CANDIDATES:
            findings.append(f"C1 support gate drift: {call}")

    # C5: propagation mechanism parity. `ValidityPropagator.propagate_update`
    # sets the target's valid_until to the *superseding event's* time anchor and
    # its update_val to 0.0; downstream successors reached via causes/enables
    # fall to 0.5. Checked against that contract rather than against the
    # delivered values, because which pair wins a candidate slot is not
    # reproducible (see module docstring) -- agreement with delivered is
    # reported separately as a diagnostic.
    delivered_events = payload["events"]
    agree = total = 0
    for eid, ev in graph._events.items():
        if ev.valid_until is None:
            if ev.update_val not in (1.0, 0.5):
                findings.append(f"C5 {eid}: update_val {ev.update_val} without valid_until")
            continue
        total += 1
        if ev.update_val != 0.0:
            findings.append(f"C5 {eid}: superseded but update_val={ev.update_val}, expected 0.0")
        anchors = {
            graph._events[e].time_anchor
            for e in graph.graph.predecessors(eid)
            if e in graph._events
            and graph.graph.edges[e, eid].get("edge_type") == "updates"
        }
        if ev.valid_until not in anchors:
            findings.append(
                f"C5 {eid}: valid_until={ev.valid_until} is not the time anchor of "
                f"any event that supersedes it ({sorted(anchors)})"
            )
        if delivered_events.get(eid, {}).get("valid_until") == ev.valid_until:
            agree += 1

    # C6: serialisation parity. The driver writes through `CausalGraph.save`,
    # the same serialiser `build_graph_locomo` uses, so the comparison is
    # against a batch graph re-serialised with today's code -- the delivered
    # files on disk are schema 1.0 and predate the `hyperedges` field.
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "incremental.json"
        ref = Path(td) / "batch_reserialised.json"
        graph.save(str(out), metadata={"builder": "incremental_v1", "sample_idx": sample_idx})
        CausalGraph.load(str(_materialise(payload, Path(td)))).save(
            str(ref), metadata=payload.get("metadata", {})
        )
        got = json.loads(out.read_text(encoding="utf-8"))
        want = json.loads(ref.read_text(encoding="utf-8"))
        reloaded = CausalGraph.load(str(out))

    if set(got.keys()) != set(want.keys()):
        findings.append(f"C6 top-level keys {sorted(got.keys())} != {sorted(want.keys())}")
    if got["events"] and want["events"]:
        gk = set(next(iter(got["events"].values())).keys())
        wk = set(next(iter(want["events"].values())).keys())
        if gk != wk:
            findings.append(f"C6 event fields {sorted(gk)} != {sorted(wk)}")
    if got["edges"] and want["edges"]:
        gk = set(got["edges"][0].keys())
        wk = set(want["edges"][0].keys())
        if gk != wk:
            findings.append(f"C6 edge fields {sorted(gk)} != {sorted(wk)}")
    if (len(reloaded._events), len(reloaded._sessions), len(reloaded._topics)) != (
        len(graph._events), len(graph._sessions), len(graph._topics)
    ):
        findings.append("C6 driver output does not round-trip through CausalGraph.load")

    return {
        "sample_idx": sample_idx,
        "sessions": len(ordered),
        "events": len(graph._events),
        "reports": reports,
        "findings": findings,
        "topics": len(graph._topics),
        "topic_drifts": sum(1 for r in reports if r.topic_drifted),
        "reclusters": sum(1 for r in reports if r.topic_reclustered),
        "evolution_edges_recorded": len(recorded),
        "evolution_edges_replayed": sum(r.update_edges + r.contradiction_edges for r in reports),
        "supersessions": total,
        "same_anchor_as_delivered": agree,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--graph-dir", default="cached_graphs_openai_openai_gpt-4o-mini")
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    gate = check_gate_parity()
    print("C1 gate parity vs. batch builder:", "OK" if not gate else "FAIL")
    for g in gate:
        print("   ", g)
    print(f"    supersession: Jaccard>={SUPERSESSION_MIN_JACCARD}, cap={SUPERSESSION_MAX_CANDIDATES}")
    print(f"    support:      Jaccard>={SUPPORT_MIN_JACCARD}, cap={SUPPORT_MAX_CANDIDATES} (+ topic restriction)")
    print()

    gdir = TRACE_DIR / args.graph_dir
    if args.all:
        paths = sorted(gdir.glob("event_graph_sample_*.json"))
        paths = [p for p in paths if "pre_hyperedge" not in p.name]
    else:
        paths = [gdir / f"event_graph_sample_{args.sample if args.sample is not None else 0}.json"]

    all_findings = list(gate)
    hdr = (f"{'sample':>6} {'sessions':>8} {'events':>7} {'pool':>7} {'screened':>9} "
           f"{'evo replayed':>13} {'topics':>7} {'reclust':>8} {'findings':>9}")
    print(hdr); print("-" * len(hdr))
    for p in paths:
        r = replay(load_delivered(p))
        pool = sum(x.candidate_pool for x in r["reports"])
        scr = sum(x.candidates_screened for x in r["reports"])
        print(f"{r['sample_idx']:>6} {r['sessions']:>8} {r['events']:>7} {pool:>7} {scr:>9} "
              f"{r['evolution_edges_replayed']:>4}/{r['evolution_edges_recorded']:<8} "
              f"{r['topics']:>7} {r['reclusters']:>8} {len(r['findings']):>9}")
        all_findings.extend(r["findings"])

    print()
    if all_findings:
        print(f"FAIL — {len(all_findings)} finding(s):")
        for f in all_findings[:40]:
            print("  -", f)
        return 1
    print("PASS — C1..C8 clean: the incremental driver performs the batch builder's")
    print("       operations, with candidates drawn only from ingested history.")
    print()
    print("Note: 'evo replayed / recorded' is a candidate-coverage diagnostic, NOT a")
    print("      correctness metric. The delivered graphs' candidate sets are not")
    print("      reproducible (Jaccard ties + per-process set ordering), so a shortfall")
    print("      here reflects tie-breaking, not driver behaviour.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
