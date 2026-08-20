#!/usr/bin/env bash
#
# Run the SemanticFlow evaluation end to end: baseline + multi-agent (no-HITL) +
# multi-agent (force-ask HITL), then the uncertainty/effort/comparison analysis.
#
# Each experiment writes experiments/<id>/summary.json with accuracy AND a cost block.
#
# Usage:
#   ./scripts/run_full_experiment.sh                 # core 35-task benchmark
#   TASKS=data/all_tasks.jsonl ./scripts/run_full_experiment.sh   # full 85 tasks
#
# Override anything via env, e.g.:
#   PROVIDER=openai K=1 REPEATS=1 PREFIX=smoke ./scripts/run_full_experiment.sh
#
# COST: spend = providers x arms x K x REPEATS x tasks. Reserve the full config (3
# providers, K=3, REPEATS=3) for FINAL reported numbers only. For development:
#   - K=1 loses nothing here (the aleatoric signal it powers is flat, AUROC ~0.50).
#   - ENSEMBLE=anthropic keeps it single-provider ($0 openai).
#   - SEMANTICFLOW_LLM_CACHE=1 (safe at K=1) makes warm re-runs $0.
#   - ANTHROPIC_MODEL=claude-haiku-4-5-20251001 is several x cheaper for pipeline debugging.
# Cheap dev sweep (cache on, single provider, haiku, 5 tasks):
#   SEMANTICFLOW_LLM_CACHE=1 ANTHROPIC_MODEL=claude-haiku-4-5-20251001 \
#     TASKS=data/_subset.jsonl K=1 REPEATS=1 PREFIX=dev ./scripts/run_full_experiment.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."

# --- Config (override via env) ----------------------------------------------
TASKS="${TASKS:-data/real_tasks.jsonl}"   # core 35; set data/all_tasks.jsonl for full 85
PROVIDER="${PROVIDER:-anthropic}"          # real single provider for the baseline
ENSEMBLE="${ENSEMBLE:-anthropic}"          # multi-agent ensemble providers (comma-sep). Single
                                           # provider -> no cross-model epistemic signal; the
                                           # hybrid trigger uses aleatoric+structural only.
                                           # Set "openai,anthropic,gemini" for the 3-model ensemble.
K="${K:-3}"                                # self-consistency samples -> aleatoric signal
REPEATS="${REPEATS:-3}"                    # full-pipeline repeats -> accuracy_std
PREFIX="${PREFIX:-exp}"                    # experiment-id prefix
SIGNAL="${SIGNAL:-total_uncertainty}"      # signal to sweep (pick best via validate-uncertainty)

# --- Load .env (keys) FIRST, so the script's tunables below win over any
#     SEMANTICFLOW_*/LLM_PROVIDER values committed in .env (e.g. K=1, REPEATS=1).
if [[ -f .env ]]; then
  set -a; source .env; set +a
fi

export LLM_PROVIDER="$PROVIDER"
export SEMANTICFLOW_ENSEMBLE_PROVIDERS="$ENSEMBLE"
export SEMANTICFLOW_SELF_CONSISTENCY_K="$K"
export SEMANTICFLOW_EXPERIMENT_REPEATS="$REPEATS"

run() { echo; echo ">>> $*"; uv run python -m semanticflow.cli "$@"; }

# --- Preflight: refuse to "succeed" silently on the mock client -------------
if [[ "$PROVIDER" == "mock" ]]; then
  echo "ERROR: PROVIDER=mock produces placeholder output — set a real provider." >&2
  exit 1
fi
echo "Tasks=$TASKS  provider=$PROVIDER  ensemble=$ENSEMBLE  K=$K  repeats=$REPEATS  prefix=$PREFIX"
echo "Smoke-testing one task before the full run..."
# Use no-HITL: multi_agent_hitl prompts interactively for clarifications, which hangs/
# aborts when run non-interactively. The experiments below run HITL with a SIMULATED
# human, so this still validates the full real-API pipeline.
uv run python -m semanticflow.cli run-task --task-id real_001 --mode multi_agent_no_hitl >/dev/null
echo "Smoke test OK."

mkdir -p figures

# --- 1. Experiments ----------------------------------------------------------
run run-experiment --mode baseline_single_agent --tasks-path "$TASKS" --experiment-id "${PREFIX}-baseline"
run run-experiment --mode multi_agent_no_hitl   --tasks-path "$TASKS" --experiment-id "${PREFIX}-nohitl"
SEMANTICFLOW_AGREEMENT_THRESHOLD=1.01 \
  run run-experiment --mode multi_agent_hitl --tasks-path "$TASKS" --experiment-id "${PREFIX}-hitl"

# --- 2. Analysis -------------------------------------------------------------
run experiment-cost --experiment "${PREFIX}-baseline"
run experiment-cost --experiment "${PREFIX}-nohitl"
run experiment-cost --experiment "${PREFIX}-hitl"

run validate-uncertainty --experiment "${PREFIX}-nohitl" --output "figures/${PREFIX}-uncertainty.json"

run sweep-threshold --no-hitl "${PREFIX}-nohitl" --hitl "${PREFIX}-hitl" \
  --signal "$SIGNAL" --output "figures/${PREFIX}-sweep.json"

run compare-experiments --baseline "${PREFIX}-baseline" --treatment "${PREFIX}-hitl" \
  --output "figures/${PREFIX}-compare.json"

# --- 3. Figures (matplotlib optional; writes CSVs regardless) ----------------
uv run python scripts/plot_results.py --no-hitl "${PREFIX}-nohitl" \
  --sweep "figures/${PREFIX}-sweep.json" --compare "figures/${PREFIX}-compare.json" \
  --outdir figures || echo "(plot_results skipped/failed — install matplotlib: uv sync)"

echo; echo "DONE. Summaries: experiments/${PREFIX}-{baseline,nohitl,hitl}/summary.json ; figures in figures/"
