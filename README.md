# TRACE — Temporal Reasoning with Augmented Causal Evidence

TRACE extends an A-Mem-style note-memory layer with a hierarchical hypergraph (Event → Session → Topic) and graph-guided retrieval for long-conversation QA. This release ships TRACE on two long-conversation benchmarks: **LoCoMo** and **LongMemEval-S**, both running on `openai/gpt-4o-mini` via OpenRouter. Each benchmark provides a main config plus three leave-one-out ablation configs (`no_topic`, `no_hier`, `no_L3`) and a runtime `--skip_graph_load` flag for the graph-free baseline.

For metrics, methodology, and the headline results, see the paper.

## Setup

```bash
conda create -n trace python=3.12 -y
conda activate trace
pip install -r requirements.txt

export OPENROUTER_API_KEY=sk-or-v1-...   # https://openrouter.ai/
```

On Windows: `set OPENROUTER_API_KEY=sk-or-v1-...`. The engines read this env var directly; no `.env` file is required.

NLTK assets `punkt` and `wordnet` are auto-downloaded on first run.

## Data

LoCoMo (`data/locomo10.json`) is bundled in this release. LongMemEval-S cleaned is fetched on demand:

```bash
bash data/download_longmemeval.sh
```

This clones the upstream LongMemEval repository and copies `data/longmemeval_s_cleaned.json` into `data/`. If the script fails (upstream layout drift), download `longmemeval_s_cleaned.json` from the official LongMemEval release manually and place it at `data/longmemeval_s_cleaned.json`.

## Running the experiments

All commands are run from the repository root.

### LoCoMo

One-shot reproduce the paper's numbers (full pipeline from scratch):

```bash
bash run_locomo_main.sh
```

Or run the three stages manually:

```bash
# 1. Ingest LoCoMo into A-Mem note memory (10 samples)
python ingest_locomo.py --config configs/locomo_main.json

# 2. Build the hierarchical hypergraph
python build_graph_locomo.py --all --dataset data/locomo10.json --force

# 3. Run TRACE QA (n=1540, cat 1-4; auto-runs 3 LLM-judge runs)
python eval_locomo.py --config configs/locomo_main.json
```

Smoke test on a single sample:

```bash
python eval_locomo.py --config configs/locomo_main.json --sample 0
```

LOO ablations (after the main ingest + graph build):

```bash
python eval_locomo.py --config configs/locomo_no_topic.json
python eval_locomo.py --config configs/locomo_no_hier.json
python eval_locomo.py --config configs/locomo_main.json --skip_graph_load --run_label locomo_graph_off
python eval_locomo.py --config configs/locomo_no_L3.json
```

Each run auto-runs 3 LLM-judge runs and writes everything to `results/locomo/`.

### LongMemEval

One-shot reproduce the paper's numbers (full pipeline from scratch):

```bash
bash run_longmemeval_main.sh
```

Or run the three stages manually:

```bash
# 1. Ingest LongMemEval: build A-Mem memory caches AND LoCoMo-style session
#    summaries (~500 questions × union of haystacks ≈ 19k unique sessions).
#    Outputs:  cached_memories_longmemeval_*/  +  cached_summaries/
python ingest_longmemeval.py --config configs/longmemeval_main.json

# 2. Build the hierarchical hypergraph from session summaries
python build_graph_longmemeval.py --config configs/longmemeval_main.json

# 3. Run TRACE QA (n=470 non-adversarial; auto-runs 3 LLM-judge runs)
python eval_longmemeval.py --config configs/longmemeval_main.json
```

LOO ablations:

```bash
python eval_longmemeval.py --config configs/longmemeval_no_topic.json
python eval_longmemeval.py --config configs/longmemeval_no_hier.json
python eval_longmemeval.py --config configs/longmemeval_main.json --skip_graph_load --run_label longmemeval_graph_off
python eval_longmemeval.py --config configs/longmemeval_no_L3.json
```

Results land in `results/longmemeval/`.

### Re-judging

If you want to re-score an existing result file with a different judge model or more runs:

```bash
python eval_llm_judge.py --result results/locomo/<file>.json --judge_runs 3
```

Pass `--judge_runs 0` to either `eval_locomo.py` or `eval_longmemeval.py` to skip the auto-judge step.

## Repository layout

```
TRACE_release/
├── README.md
├── LICENSE
├── requirements.txt
├── run_locomo_main.sh                one-shot LoCoMo pipeline (ingest + graph + eval + judge)
├── run_longmemeval_main.sh           one-shot LongMemEval pipeline (ingest + summaries + graph + eval + judge)
├── eval_locomo.py                    LoCoMo QA engine
├── eval_longmemeval.py               LongMemEval QA engine
├── eval_llm_judge.py                 LLM-judge scoring
├── ingest_locomo.py                  LoCoMo memory ingest
├── ingest_longmemeval.py             LongMemEval memory ingest + session summaries
├── build_graph_locomo.py             LoCoMo hierarchical graph builder
├── build_graph_longmemeval.py        LongMemEval hierarchical graph builder
├── memory_layer_robust.py            A-Mem memory system
├── load_dataset.py                   LoCoMo dataset parser
├── utils.py                          evaluation metrics
├── trace/                            core TRACE library (17 modules)
│   └── prompts/                      6 LLM prompt templates (extraction, cross-note, update, QA, entity extraction, LongMemEval judge)
├── configs/                          8 experiment configs (4 LoCoMo + 4 LongMemEval)
└── data/
    ├── locomo10.json
    └── download_longmemeval.sh        fetches LongMemEval-S into data/longmemeval_s_cleaned.json
```

## License

MIT. See `LICENSE`.
