"""Marginal supersession-screening cost of ingesting one more session.

Answers: as ingested history grows, how much work does linking a *new* event
cost? Measured on the batch-built graphs, in the incremental regime -- candidates
are drawn only from strictly earlier sessions -- with zero LLM calls and no
graph rebuild.

The screening predicate mirrors `CausalGraph.get_events_by_participant_overlap`
(participants lowercased, Jaccard over the two participant sets, kept when
>= min_jaccard, truncated to max_k). Thresholds are imported from
`incremental_ingest` rather than restated, so this measurement cannot drift
away from the gates the driver actually applies.

Two quantities per new event, bucketed by how many sessions were already
ingested when it arrived:

  candidate pool -- events passing the participant predicate, before the cap
  screening cost -- min(pool, max_k), i.e. pairs actually handed to the
                    pairwise classifier

The pool is what grows with history. The cost is what the system pays.

Same-session events are never candidates for one another (the parity check's
C2): history is extended only once a session has been fully ingested, which is
what the driver does and what makes the arriving session's events invisible to
each other.

Usage:
    python scripts/measure_incremental_marginal_cost.py
    python scripts/measure_incremental_marginal_cost.py --graph-dir <dir> --out <file>
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

TRACE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TRACE_DIR))

from incremental_ingest import (  # noqa: E402
    SUPERSESSION_MAX_CANDIDATES,
    SUPERSESSION_MIN_JACCARD,
)
from trace.parsing_utils import extract_session_num  # noqa: E402

# Reporting buckets. Declared here, not chosen after seeing the numbers:
# "early history" is the first few sessions, "settled history" is past the
# point where the pool has clearly outgrown the cap.
EARLY = (1, 3)
LATE_FROM = 10


def jaccard(a: Set[str], b: Set[str]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def measure_sample(payload: dict) -> Dict[int, List[Tuple[int, int]]]:
    """Replay one delivered graph in arrival order.

    Returns {prior_session_count: [(pool, cost), ...]} over the events that
    arrived in the session at that position.
    """
    events = payload["events"]
    parts = {
        eid: {p.lower() for p in (e.get("participants") or [])}
        for eid, e in events.items()
    }
    sessions = sorted(
        payload["sessions"], key=lambda s: extract_session_num(s["session_key"])
    )

    rows: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    history: List[str] = []
    for t, session in enumerate(sessions):
        arriving = [eid for eid in session.get("event_ids", []) if eid in parts]
        for eid in arriving:
            p = parts[eid]
            if not p:
                # No participants -> the predicate admits nothing. Still an
                # arriving event, so it is counted, at zero cost.
                rows[t].append((0, 0))
                continue
            pool = sum(
                1
                for h in history
                if jaccard(p, parts[h]) >= SUPERSESSION_MIN_JACCARD
            )
            rows[t].append((pool, min(pool, SUPERSESSION_MAX_CANDIDATES)))
        # Only now does the arriving session become history (C2).
        history.extend(arriving)
    return rows


def aggregate(rows: Dict[int, List[Tuple[int, int]]], lo: int, hi: int = None) -> dict:
    sel = [
        pair
        for t, pairs in rows.items()
        if t >= lo and (hi is None or t <= hi)
        for pair in pairs
    ]
    if not sel:
        return {"n_events": 0, "mean_candidate_pool": 0.0, "mean_screening_cost": 0.0}
    return {
        "n_events": len(sel),
        "mean_candidate_pool": round(statistics.mean(x[0] for x in sel), 2),
        "mean_screening_cost": round(statistics.mean(x[1] for x in sel), 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--graph-dir", default="cached_graphs_openai_openai_gpt-4o-mini")
    ap.add_argument("--out", default="results/incremental/marginal_cost_locomo.json")
    args = ap.parse_args()

    gdir = Path(args.graph_dir)
    if not gdir.is_absolute():
        gdir = TRACE_DIR / gdir
    paths = sorted(
        p
        for p in gdir.glob("event_graph_sample_*.json")
        if "pre_hyperedge" not in p.name
    )
    if not paths:
        print(f"no delivered graphs under {gdir}", file=sys.stderr)
        return 1

    merged: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for p in paths:
        payload = json.loads(p.read_text(encoding="utf-8"))
        for t, pairs in measure_sample(payload).items():
            merged[t].extend(pairs)

    by_t = {
        t: {
            "n_events": len(pairs),
            "mean_candidate_pool": round(statistics.mean(x[0] for x in pairs), 2),
            "mean_screening_cost": round(statistics.mean(x[1] for x in pairs), 2),
        }
        for t, pairs in sorted(merged.items())
    }
    early = aggregate(merged, EARLY[0], EARLY[1])
    late = aggregate(merged, LATE_FROM)
    pool_growth = (
        late["mean_candidate_pool"] / early["mean_candidate_pool"]
        if early["mean_candidate_pool"]
        else None
    )
    cost_growth = (
        late["mean_screening_cost"] / early["mean_screening_cost"]
        if early["mean_screening_cost"]
        else None
    )

    report = {
        "_meta": {
            "measurement": "incremental marginal supersession-screening cost",
            "graph_dir": gdir.name,
            "graphs": [p.name for p in paths],
            "n_events": sum(len(v) for v in merged.values()),
            "min_jaccard": SUPERSESSION_MIN_JACCARD,
            "max_candidates": SUPERSESSION_MAX_CANDIDATES,
            "regime": "candidates drawn only from strictly earlier sessions",
            "llm_calls": 0,
            "early_bucket": {"prior_sessions_from": EARLY[0], "to": EARLY[1]},
            "late_bucket": {"prior_sessions_from": LATE_FROM},
        },
        "by_prior_session_count": by_t,
        "early": early,
        "late": late,
        "pool_growth": round(pool_growth, 2) if pool_growth else None,
        "cost_growth": round(cost_growth, 2) if cost_growth else None,
    }

    out = Path(args.out)
    if not out.is_absolute():
        out = TRACE_DIR / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    hdr = f"{'prior sessions':>14} {'events':>7} {'cand pool':>10} {'screening cost':>15}"
    print(hdr)
    print("-" * len(hdr))
    for t, row in by_t.items():
        print(
            f"{t:>14} {row['n_events']:>7} {row['mean_candidate_pool']:>10.1f} "
            f"{row['mean_screening_cost']:>15.2f}"
        )
    print()
    print(
        f"early (t={EARLY[0]}..{EARLY[1]}): pool {early['mean_candidate_pool']:.1f}  "
        f"cost {early['mean_screening_cost']:.2f}  (n={early['n_events']})"
    )
    print(
        f"late  (t>={LATE_FROM}) : pool {late['mean_candidate_pool']:.1f}  "
        f"cost {late['mean_screening_cost']:.2f}  (n={late['n_events']})"
    )
    print()
    print(f"candidate pool grows {pool_growth:.2f}x; screening cost grows {cost_growth:.2f}x")
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
