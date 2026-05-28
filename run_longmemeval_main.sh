#!/usr/bin/env bash
# Run TRACE LongMemEval main from scratch and reproduce the paper's numbers.
#
# Pipeline:
#   1. ingest_longmemeval.py   — A-Mem note memory + LoCoMo-style session summaries
#                                for the full 470-question (non-adversarial) cohort
#   2. build_graph_longmemeval.py — hierarchical hypergraph from session summaries
#                                   (LongMemEval skips Phase 3 update detection per
#                                   `longmemeval_skip_update_detection: true` in the config)
#   3. eval_longmemeval.py     — TRACE QA over n=470 non-adversarial questions
#                                (auto-runs 3 LLM-judge runs at the end)
#
# Usage (from the repository root):
#   export OPENROUTER_API_KEY=sk-or-v1-...
#   conda activate trace
#   bash run_longmemeval_main.sh

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

CONFIG="configs/longmemeval_main.json"
DATASET="data/longmemeval_s_cleaned.json"

if [[ ! -f "${CONFIG}" ]]; then
    echo "ERROR: ${CONFIG} not found. Run from the repository root." >&2
    exit 1
fi
if [[ ! -f "${DATASET}" ]]; then
    echo "ERROR: ${DATASET} not found. Run from the repository root." >&2
    echo "       Run 'bash data/download_longmemeval.sh' to fetch it." >&2
    exit 1
fi

echo "============================================================"
echo "TRACE LongMemEval main — full pipeline from scratch"
echo "============================================================"
echo "Config:  ${CONFIG}"
echo "Dataset: ${DATASET}"
echo "Started: $(date)"
echo

# ---------------------------------------------------------------------------
# Stage 1 — A-Mem ingest + session summaries
# ---------------------------------------------------------------------------

echo "[$(date +%H:%M:%S)] Stage 1/3: ingest_longmemeval.py"
echo "                  (memory cache + ~19,195 session summaries)"
python ingest_longmemeval.py --config "${CONFIG}" --force

# ---------------------------------------------------------------------------
# Stage 2 — Hierarchical hypergraph
# ---------------------------------------------------------------------------

echo
echo "[$(date +%H:%M:%S)] Stage 2/3: build_graph_longmemeval.py --force"
python build_graph_longmemeval.py --config "${CONFIG}" --force

# ---------------------------------------------------------------------------
# Stage 3 — TRACE QA + auto-judge
# ---------------------------------------------------------------------------

echo
echo "[$(date +%H:%M:%S)] Stage 3/3: eval_longmemeval.py + 3 LLM-judge runs"
python eval_longmemeval.py --config "${CONFIG}"

echo
echo "============================================================"
echo "Done. Results in:  results/longmemeval/"
echo "Finished: $(date)"
echo "============================================================"
