#!/usr/bin/env bash
# Run TRACE LoCoMo main from scratch and reproduce the paper's numbers.
#
# Pipeline:
#   1. ingest_locomo.py        — A-Mem note memory for all 10 samples
#   2. build_graph_locomo.py   — hierarchical hypergraph for all 10 samples
#   3. eval_locomo.py          — TRACE QA over n=1540 cat 1-4 questions
#                                (auto-runs 3 LLM-judge runs at the end)
#
# Usage (from the repository root):
#   export OPENROUTER_API_KEY=sk-or-v1-...
#   conda activate trace
#   bash run_locomo_main.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "ERROR: OPENROUTER_API_KEY is not set." >&2
    echo "       Get one at https://openrouter.ai/ and run:" >&2
    echo "       export OPENROUTER_API_KEY=sk-or-v1-..." >&2
    exit 1
fi

CONFIG="configs/locomo_main.json"
DATASET="data/locomo10.json"

if [[ ! -f "${CONFIG}" ]]; then
    echo "ERROR: ${CONFIG} not found. Run from the repository root." >&2
    exit 1
fi
if [[ ! -f "${DATASET}" ]]; then
    echo "ERROR: ${DATASET} not found. Run from the repository root." >&2
    exit 1
fi

echo "============================================================"
echo "TRACE LoCoMo main — full pipeline from scratch"
echo "============================================================"
echo "Config:  ${CONFIG}"
echo "Dataset: ${DATASET}"
echo "Started: $(date)"
echo

# ---------------------------------------------------------------------------
# Stage 1 — A-Mem ingest
# ---------------------------------------------------------------------------

echo "[$(date +%H:%M:%S)] Stage 1/3: ingest_locomo.py (10 samples)"
python ingest_locomo.py --config "${CONFIG}"

# ---------------------------------------------------------------------------
# Stage 2 — Hierarchical hypergraph
# ---------------------------------------------------------------------------

echo
echo "[$(date +%H:%M:%S)] Stage 2/3: build_graph_locomo.py --all --force"
python build_graph_locomo.py --all --dataset "${DATASET}" --force

# ---------------------------------------------------------------------------
# Stage 3 — TRACE QA + auto-judge
# ---------------------------------------------------------------------------

echo
echo "[$(date +%H:%M:%S)] Stage 3/3: eval_locomo.py + 3 LLM-judge runs"
python eval_locomo.py --config "${CONFIG}"

echo
echo "============================================================"
echo "Done. Results in:  results/locomo/"
echo "Finished: $(date)"
echo "============================================================"
