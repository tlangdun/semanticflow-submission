# SemanticFlow

**Generating dbt Semantic Layer artifacts from natural language, with hybrid uncertainty
quantification and a slot-scoped human-in-the-loop.**

SemanticFlow turns a plain-English analytics request such as *"Show me orders per day for
the last 30 days"* into dbt semantic models, MetricFlow metrics, and an executed MetricFlow
query. It is the research artifact for an MSc dissertation (University of Bath). The
manuscript itself is not distributed with this repository.

---

## Contents

1. [What the system does](#what-the-system-does)
2. [Repository layout](#repository-layout)
3. [Step-by-step: getting it running](#step-by-step-getting-it-running)
4. [Step-by-step: reproducing the evaluation](#step-by-step-reproducing-the-evaluation)
5. [Running on the second schema (TPC-H)](#running-on-the-second-schema-tpc-h)
6. [Configuration reference](#configuration-reference)
7. [Tests](#tests)
8. [Further documentation](#further-documentation)

---

## What the system does

- **Multi-provider consensus.** Three LLM providers (OpenAI, Anthropic, Gemini) each parse
  the request into a semantic spec; the specs are reconciled into one consensus spec.
- **Hybrid uncertainty.** Total uncertainty is decomposed into *aleatoric* (within-model
  K-sample self-consistency), *epistemic* (cross-provider disagreement) and *structural*
  (field-set Jaccard) components, optionally sharpened by execution-equivalence clustering
  of each provider's result table.
- **Slot-scoped human-in-the-loop.** Clarifying questions fire only when the uncertainty
  signal trips. A *slot-scoped guard* reverts an individual clarified slot to provider
  consensus only when every provider already agreed on it and the human answer did not
  address that slot.
- **Self-healing execution.** Automatic recovery from common dbt/MetricFlow errors, plus
  execution-feedback repair driven by the query result itself.
- **Execution-grounded evaluation.** A generated query counts as correct only if its result
  table matches the task's validated gold SQL (numeric tolerance, column order- and
  name-agnostic) — not if its YAML looks similar to a reference.
- **Two model tiers, two schemas.** The same pipeline is evaluated on a *cheap* and a
  *frontier* model tier, over the Jaffle Shop demo warehouse and a TPC-H second schema.

```
"Count orders by day for the last 30 days"
            |
      3 providers parse  ->  consensus spec
            |
      hybrid uncertainty  (aleatoric + epistemic + structural)
            |
      trip?  ->  clarifying questions  ->  slot-scoped guard
            |
      generate semantic_models.yml + metrics.yml
            |
      dbt parse -> dbt build -> mf validate -> mf query
            |
      execution-equivalence score vs gold SQL
```

---

## Repository layout

```
semanticflow/
  agents/              multi-agent pipeline
    orchestrator.py      pipeline coordination, uncertainty + guard logic
    semantic_mapper.py   cross-provider consensus, hybrid uncertainty
    ambiguity_analyzer.py clarifying-question generation
    designer.py          entity / measure inference
    codegen.py           MetricFlow YAML generation
    executor.py          dbt + MetricFlow invocation
    error_recovery.py    self-healing on dbt/MetricFlow errors
    execution_repair.py  execution-feedback repair
    result_validator.py  result sanity checks
    evaluator.py         execution-equivalence scoring
  evaluation/          scoring, uncertainty validation, conformal calibration
  dbt_integration/     dbt / MetricFlow runners and YAML handling
  llm/                 provider clients, routing, caching, cost accounting
  cli.py               command-line interface

data/
  all_tasks.jsonl        85-task Jaffle Shop benchmark (35 real + 30 synthetic + 20 extended)
  real_tasks.jsonl       35 hand-authored tasks
  synthetic_tasks.jsonl  30 generated tasks (25 gold-backed + 5 unanswerable HITL probes)
  ext_tasks.jsonl        20 extended tasks
  tpch_tasks.jsonl       40-task TPC-H second-schema benchmark

scripts/
  author_gold.py         authors and verifies gold SQL for the hand-authored tasks
  build_synthetic.py     builds and verifies the synthetic benchmark
  direct_sql_baseline.py naive single-call text-to-SQL baseline arm
  parallel_run.py        shards one experiment across isolated dbt project copies
  plot_results.py        produces the three thesis figures
  llm_smoke_test.py      one-shot provider connectivity check
  run_full_experiment.sh end-to-end three-arm run plus analysis

third_party/
  jaffle_shop_duckdb/    primary dbt project (Jaffle Shop demo warehouse)
  tpch_duckdb/           second-schema dbt project (TPC-H)

figures/                 result JSON/CSV and the three figures reported in the write-up
docs/reference.md        detailed reference documentation
```

---

## Step-by-step: getting it running

### Prerequisites

- **Python 3.12** (pinned in `.python-version`)
- **[uv](https://docs.astral.sh/uv/)** for dependency management
- At least one LLM API key. All three are needed to reproduce the multi-provider arms.

### Step 1 — Install dependencies

```bash
uv sync
source .venv/bin/activate
```

This installs `dbt-core`, `dbt-duckdb` and `dbt-metricflow` into the virtualenv, so the
`dbt` and `mf` executables the pipeline shells out to are on `PATH` once it is activated.

### Step 2 — Provide API keys

Create a `.env` file in the repository root:

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...
```

Then load it into the shell:

```bash
set -a; source .env; set +a
```

Confirm the providers answer:

```bash
python scripts/llm_smoke_test.py
```

> Without keys the pipeline falls back to a mock client and logs a warning. Those runs
> exercise the plumbing but produce no meaningful accuracy numbers.

### Step 3 — Build the Jaffle Shop warehouse

The pipeline reads and writes a DuckDB warehouse inside the dbt project directory:

```bash
cd third_party/jaffle_shop_duckdb
dbt seed
dbt build
cd ../..
```

Gold-SQL scoring reads a warehouse at the repository root. A prebuilt copy is committed as
`jaffle_shop.duckdb`; to refresh it after rebuilding, either copy the project's warehouse
up or point the scorer at it explicitly:

```bash
cp third_party/jaffle_shop_duckdb/jaffle_shop.duckdb jaffle_shop.duckdb
# or, per run:
export DUCKDB_PATH=third_party/jaffle_shop_duckdb/jaffle_shop.duckdb
```

### Step 4 — Select a real provider for the baseline arm

`config.toml` ships with `llm_provider = "mock"` so a fresh clone never spends credits by
accident. Set it to a real provider before running anything you intend to measure:

```bash
export LLM_PROVIDER=anthropic     # or: openai | gemini
```

Environment variables override `config.toml`, so no file edit is required.

### Step 5 — Run a single task

```bash
# Inspect the benchmark
uv run python main.py list-tasks --tasks-path data/all_tasks.jsonl

# Run one task end to end, with clarifying questions enabled
uv run python main.py run-task --task-id real_001 --mode multi_agent_hitl
```

Every run writes its artifacts — generated YAML, dbt and MetricFlow logs, the semantic
spec, the LLM transcript and the evaluation record — to
`experiments/<experiment_id>/<run_id>/`.

### Step 6 — Run a small experiment

```bash
uv run python main.py run-experiment \
  --mode multi_agent_no_hitl \
  --tasks-path data/real_tasks.jsonl \
  --experiment-id trial-nohitl

uv run python main.py experiment-cost --experiment trial-nohitl
```

`experiment-cost` reports token usage and estimated spend, which is worth checking before
launching the full matrix below.

---

## Step-by-step: reproducing the evaluation

Accuracy is measured by execution equivalence against each task's validated gold SQL. The
primary benchmark is the **85-task** Jaffle Shop set (`data/all_tasks.jsonl`, 26 low / 21
medium / 38 high ambiguity; 78 carry gold SQL). High-ambiguity tasks are scored against
their intended post-clarification interpretation, since they have no single valid answer.

### Step 1 — Enable the hybrid signal

```bash
export SEMANTICFLOW_SELF_CONSISTENCY_K=3    # per-provider samples -> aleatoric signal
export SEMANTICFLOW_EXPERIMENT_REPEATS=3    # pipeline repeats per task -> accuracy_std
```

Repeats multiply cost by three; use `1` for a first pass and `3` for final numbers.

### Step 2 — Run the three arms

```bash
# Arm 1: single-provider baseline
uv run python main.py run-experiment --mode baseline_single_agent \
  --tasks-path data/all_tasks.jsonl --experiment-id exp-baseline

# Arm 2: multi-provider consensus, no clarification
uv run python main.py run-experiment --mode multi_agent_no_hitl \
  --tasks-path data/all_tasks.jsonl --experiment-id exp-nohitl

# Arm 3: clarification. AGREEMENT_THRESHOLD=1.01 forces every task to ask, which is what
# gives each task an "if-asked" score for the offline threshold sweep in Step 4.
SEMANTICFLOW_AGREEMENT_THRESHOLD=1.01 \
  uv run python main.py run-experiment --mode multi_agent_hitl \
  --tasks-path data/all_tasks.jsonl --experiment-id exp-hitl
```

To run the frontier tier instead of the cheap tier, override the model names:

```bash
export OPENAI_MODEL=gpt-5.4 ANTHROPIC_MODEL=claude-sonnet-4-6 GEMINI_MODEL=gemini-3.1-pro-preview
```

Arms are independent and can be run concurrently, but each mutates the shared dbt project
directory, so concurrent runs must be isolated. `scripts/parallel_run.py` shards one
experiment across private copies of the project and merges the results:

```bash
uv run python scripts/parallel_run.py --workers 4 --mode multi_agent_no_hitl \
  --tasks-path data/all_tasks.jsonl --experiment-id exp-nohitl
```

### Step 3 — Is the uncertainty signal real?

```bash
uv run python main.py validate-uncertainty --experiment exp-nohitl
```

Reports AUROC of each uncertainty component against execution correctness.

### Step 4 — Effort–accuracy trade-off

```bash
uv run python main.py sweep-threshold \
  --no-hitl exp-nohitl --hitl exp-hitl \
  --signal total_uncertainty --output figures/sweep.json
```

Sweeps the clarification trigger threshold offline, and also reports the oracle upper
bound and the cheapest question budget reaching a target accuracy.

### Step 5 — Conformal calibration

```bash
uv run python main.py conformal-threshold \
  --experiment exp-nohitl --signal total_uncertainty --alpha 0.3
```

Calibrates the trigger with a finite-sample risk guarantee over the auto-answered tasks.

### Step 6 — Paired comparison

```bash
uv run python main.py compare-experiments \
  --baseline exp-baseline --treatment exp-hitl \
  --tasks-path data/all_tasks.jsonl --output figures/compare.json
```

Reports the accuracy delta with a **bootstrap 95% CI** (the primary read) alongside
McNemar's exact p-value (**exploratory** — n is small), stratified into overall,
high-ambiguity, and low/medium. This is a design-science feasibility study, not a powered
randomised trial: read the effect sizes and the effort–accuracy curve, not the
significance stars.

### Step 7 — Class-aware scorecard and the direct-SQL arm

```bash
uv run python main.py score-by-class \
  --baseline exp-baseline --no-hitl exp-nohitl --hitl exp-hitl \
  --tasks-path data/all_tasks.jsonl

uv run python scripts/direct_sql_baseline.py \
  --tasks-path data/all_tasks.jsonl --experiment-id exp-direct-sql
```

The direct-SQL arm asks a single LLM call for SQL straight from the question and schema,
scored by the same execution equivalence — the control for whether the semantic-layer
apparatus earns its keep.

### Step 8 — Figures

```bash
uv run python scripts/plot_results.py --no-hitl exp-nohitl \
  --sweep figures/sweep.json --compare figures/compare.json --outdir figures
```

Writes the ROC, effort–accuracy and accuracy-by-ambiguity series as CSV always, and as PNG
when matplotlib is installed (`uv sync` includes it in the dev group).

`scripts/run_full_experiment.sh` wraps steps 2–6 into a single invocation with sensible
defaults if you would rather not run them one at a time.

---

## Running on the second schema (TPC-H)

The TPC-H warehouse is not committed. Regenerate it, then point both the dbt project and
the gold scorer at it:

```bash
duckdb third_party/tpch_duckdb/tpch.duckdb <<'SQL'
INSTALL tpch; LOAD tpch;
CALL dbgen(sf = 0.05);
CREATE TABLE raw_lineitem AS SELECT * FROM lineitem;
CREATE TABLE raw_orders   AS SELECT * FROM orders;
CREATE TABLE raw_customer AS SELECT * FROM customer;
CREATE TABLE raw_nation   AS SELECT * FROM nation;
SQL

cd third_party/tpch_duckdb && dbt build && cd ../..

DBT_PROJECT_DIR=third_party/tpch_duckdb DUCKDB_PATH=third_party/tpch_duckdb/tpch.duckdb \
  uv run python main.py run-experiment --mode multi_agent_hitl \
  --tasks-path data/tpch_tasks.jsonl --experiment-id tpch-hitl
```

Both environment variables are required for a non-Jaffle run: `DBT_PROJECT_DIR` drives
column-type enrichment, `DUCKDB_PATH` is the gold-scoring warehouse. See
`third_party/tpch_duckdb/README.md` for the schema conventions the designer relies on.

---

## Configuration reference

Tunables live in [`config.toml`](config.toml) under a documented `[semanticflow]` table;
secrets live in `.env`. Environment variables override both.

| Variable | Meaning |
| --- | --- |
| `LLM_PROVIDER` | Single provider for the baseline arm. Must be real, not `mock`. |
| `OPENAI_MODEL` / `ANTHROPIC_MODEL` / `GEMINI_MODEL` | Model per provider; how the tier is selected. |
| `SEMANTICFLOW_MODE` | `baseline_single_agent` \| `multi_agent_no_hitl` \| `multi_agent_hitl` |
| `SEMANTICFLOW_AGREEMENT_THRESHOLD` | Clarification fires below this consensus level; `1.01` forces every task to ask. |
| `SEMANTICFLOW_SELF_CONSISTENCY_K` | Samples per provider; `>1` enables the aleatoric and total signals. |
| `SEMANTICFLOW_EXPERIMENT_REPEATS` | Full-pipeline repeats per task; yields `accuracy_std`. |
| `SEMANTICFLOW_USE_OPENROUTER` | Route all providers via OpenRouter (steadier under load). |
| `SEMANTICFLOW_LLM_CACHE` | Disk cache for LLM responses. Only valid at `K=1` — see below. |
| `DBT_PROJECT_DIR` / `DUCKDB_PATH` | Target dbt project and gold-scoring warehouse. |
| `SEMANTICFLOW_LOG_LEVEL` | `INFO`, or `DEBUG` for the full transcript. |

Default cheap tier: `gpt-5.4-mini` / `claude-haiku-4-5` / `gemini-2.5-flash`.
Frontier tier: `gpt-5.4` / `claude-sonnet-4-6` / `gemini-3.1-pro-preview`.

> **Cache caveat.** The disk cache is force-disabled in code whenever
> `SEMANTICFLOW_SELF_CONSISTENCY_K > 1`: identical cached responses would collapse the
> aleatoric signal to zero. Use it only for `K=1` debugging runs.

---

## Tests

The suite is 312 tests and runs offline — no API keys and no warehouse required.

```bash
python -m pytest                          # full suite
python -m pytest tests/test_evaluation.py # execution scoring, uncertainty, statistics
```

The `-m` form puts the repository root on `sys.path` so the `semanticflow` package imports
cleanly.

---

## Further documentation

- [`docs/reference.md`](docs/reference.md) — configuration options, the JSONL task schema,
  per-agent descriptions, MetricFlow conventions, troubleshooting.
The full method, the reported results and their limitations are documented in the
dissertation manuscript, which is submitted separately and not included here.

## License

MIT. The vendored dbt projects under `third_party/` retain their own licences.
