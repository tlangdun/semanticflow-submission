# SemanticFlow — Reference Documentation

## Overview
SemanticFlow Eval turns natural language analytics requests into dbt Semantic Layer artifacts and validates them end to end. It loads tasks from JSONL files, uses a multi-agent pipeline to interpret intent, generates MetricFlow YAML and supporting SQL, runs dbt + MetricFlow validation/query, and (optionally) compares results to gold outputs.

Core goals:
- Convert NL requests into a structured semantic spec.
- Generate dbt semantic models and metrics for MetricFlow.
- Validate with dbt parse/build and MetricFlow CLI.
- Evaluate semantic accuracy and execution accuracy (when gold SQL/results exist).
- Log every run with artifacts for auditability.

Key features:
- **Multi-agent consensus**: 3 LLMs (OpenAI, Anthropic, Gemini) vote on interpretation
- **Human-in-the-loop (HITL)**: Smart clarifying questions when models disagree
- **Self-healing**: Auto-recovery from common dbt/MetricFlow errors
- **Result validation**: Sanity checks on query outputs (nulls, outliers, duplicates)
- **Iterative refinement**: Improve specs based on feedback loops

## Requirements
- Python 3.12 (see `.python-version`).
- dbt-core + dbt-duckdb.
- dbt-metricflow (MetricFlow CLI).
- LLM SDKs as needed: openai, anthropic, google-genai.
- LangGraph, Typer, Pydantic, DuckDB.

Install deps with `uv sync` after setting up your environment.

## Project layout
- `semanticflow/`: main library
  - `agents/`: multi-agent pipeline
    - `semantic_mapper.py`: LLM consensus + spec generation
    - `ambiguity_analyzer.py`: conflict detection + smart clarifying questions
    - `designer.py`: generic entity/measure/dimension inference from schema
    - `codegen.py`: YAML generation (consolidated files)
    - `executor.py`: dbt + MetricFlow execution
    - `evaluator.py`: semantic + execution accuracy
    - `error_recovery.py`: **NEW** - self-healing error analysis and auto-fix
    - `result_validator.py`: **NEW** - query result sanity checks
    - `refinement.py`: **NEW** - iterative spec improvement from feedback
    - `orchestrator.py`: LangGraph pipeline coordination
  - `dbt_integration/`: dbt/MetricFlow integration
    - `runner.py`: dbt commands (parse, build, seed)
    - `metricflow_runner.py`: mf validate/query with path resolution fixes
    - `semantic_yaml.py`: YAML spec models
    - `filter_utils.py`: filter translation (abstract → MetricFlow Jinja)
    - `project_inspect.py`: schema extraction
  - `evaluation/`: gold SQL runner and result comparison
  - `llm/`: LLM provider clients
    - `router.py`: OpenAI, Anthropic, Gemini clients
    - `cache.py`: **NEW** - LLM response caching for dev efficiency
    - `heuristics.py`: fallback spec generation
  - `utils/`: **NEW** - utility modules
    - `naming.py`: metric name normalization and suggestion
  - `log.py`: structured logging (text/JSON formats)
  - `tasks/`: JSONL task loader and Task model
  - `cli.py`: Typer CLI
  - `experiment.py`: batch runner and artifact logging
- `data/`: task JSONL files (`real_tasks.jsonl`, `synthetic_tasks.jsonl`)
- `third_party/jaffle_shop_duckdb/`: dbt project
- `experiments/`: per-run artifacts and results
- `tests/`: unit tests
  - `test_filter_translation.py`: filter translation tests
  - `test_designer.py`: entity/measure inference tests
  - `test_new_features.py`: error recovery, validation, refinement tests

## Configuration
Settings are loaded from (in order):
1) `config.toml` (or `SEMANTICFLOW_CONFIG` path)
2) Environment variables

This project does not auto-load `.env`; use `source .env` or your shell tool of choice.

### Using .env for LLM keys
The CLI does not load `.env` automatically. If your keys live in `.env`, run:
```
source .env
```

If you are using a file without `export` statements, use:
```
set -a
source .env
set +a
```

Notes:
- `LLM_PROVIDER` defaults to `mock`, so baseline runs will not call real APIs unless you set it.
- Multi-agent runs attempt OpenAI/Anthropic/Gemini in parallel, but missing keys or missing SDKs
  fall back to a mock client, which produces no API usage.

### Environment variables
DBT:
- `DBT_PROJECT_DIR` (default: `third_party/jaffle_shop_duckdb`)
- `DBT_PROFILE_NAME`
- `DUCKDB_PATH` (optional; defaults to `jaffle_shop.duckdb` if present)

LLMs:
- `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_BASE_URL`
- `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`
- `GEMINI_API_KEY`, `GEMINI_MODEL`
- `LLM_PROVIDER` (single-provider mode for baseline)

SemanticFlow:
- `SEMANTICFLOW_MODE` (baseline_single_agent | multi_agent_no_hitl | multi_agent_hitl)
- `SEMANTICFLOW_AGREEMENT_THRESHOLD` (default 0.75)
- `SEMANTICFLOW_UNCERTAINTY_THRESHOLD` (default 0.5)
- `SEMANTICFLOW_RUN_DBT` (true/false)
- `SEMANTICFLOW_MAX_LLM_REPAIR` (default 1; also controls max refinement iterations)
- `SEMANTICFLOW_LLM_TIMEOUT_SEC` (default 60; per-provider total budget incl. repair)
- `SEMANTICFLOW_LOG_LLM_RAW` (true/false; writes raw LLM responses to artifacts only)
- `SEMANTICFLOW_NORMALIZE_METRIC_NAMES` (true/false; normalize metric names to snake_case)
- `SEMANTICFLOW_METRIC_NAME_PREFIX` (optional prefix for metric names)
- `SEMANTICFLOW_LOG_LEVEL` (DEBUG | INFO | WARNING | ERROR)
- `SEMANTICFLOW_LOG_FORMAT` (text | json)
- `SEMANTICFLOW_LLM_CACHE` (true/false; cache LLM responses for dev efficiency)

MetricFlow:
- `MF_CLI_PATH` (default: `mf`; use `./.venv/bin/mf` if needed)
- `MF_CLI_MODE` (`mf` or `dbt_sl`)
- `MF_VALIDATE_WAREHOUSE` (true/false)
- `MF_QUERY_OUTPUT` (any non-empty value enables CSV capture)

### config.toml
You can also use a `config.toml` with a `[semanticflow]` section:

```
[semanticflow]
dbt_project_dir = "third_party/jaffle_shop_duckdb"
llm_provider = "openai"
openai_model = "gpt-4o"
```

## Tasks format (JSONL)
Each line is one task. Required:
- `task_id`: unique string
- `nl_request`: natural language request
- `schema_context`: schema hints (models, columns)

Optional:
- `expected_semantic`: gold semantic spec for evaluation
- `gold_sql`: SQL to compute gold results
- `gold_query_result`: precomputed result rows
- `result_schema`: expected columns/types
- `compare_rules`: comparison rules (abs_tol, rel_tol, ordering, nulls_equal)
- `difficulty`, `ambiguity`, `dbt_project`

### Task examples
Minimal task (semantic only):
```
{"task_id":"real_001","nl_request":"Show orders per day for the last 30 days.","schema_context":{"models":["orders"],"columns":{"orders":["order_id","order_date","customer_id"]}},"expected_semantic":{"metric":"orders_per_day","base_measure":"order_count","group_by":["order_date"],"time_granularity":"day","filters":["last_30_days"]},"difficulty":"simple","ambiguity":"low"}
```

Task with gold SQL (compute on the fly):
```
{"task_id":"gold_001","nl_request":"Show orders per day for the last 30 days.","schema_context":{"models":["orders"],"columns":{"orders":["order_id","order_date","customer_id"]}},"expected_semantic":{"metric":"orders_per_day","base_measure":"order_count","group_by":["order_date"],"time_granularity":"day","filters":["last_30_days"]},"gold_sql":"select date_trunc('day', order_date) as order_date, count(*) as order_count from orders where order_date >= current_date - interval 30 day group by 1 order by 1","result_schema":[{"name":"order_date","type":"date"},{"name":"order_count","type":"integer"}],"compare_rules":{"ordering":"ordered","abs_tol":0.0,"rel_tol":0.0,"nulls_equal":true},"difficulty":"simple","ambiguity":"low"}
```

Task with stored gold results:
```
{"task_id":"gold_002","nl_request":"List top 2 customers by lifetime value.","schema_context":{"models":["customers"],"columns":{"customers":["customer_id","customer_lifetime_value"]}},"expected_semantic":{"metric":"customer_lifetime_value","group_by":["customer"],"order_by":[{"field":"customer_lifetime_value","direction":"desc"}],"limit":2},"gold_query_result":[{"customer_id":1,"customer_lifetime_value":450.0},{"customer_id":2,"customer_lifetime_value":320.0}],"result_schema":[{"name":"customer_id","type":"integer"},{"name":"customer_lifetime_value","type":"float"}],"compare_rules":{"ordering":"ordered","abs_tol":0.01,"rel_tol":0.0,"nulls_equal":true},"difficulty":"simple","ambiguity":"low"}
```

## Orchestration pipeline
Implemented with LangGraph in `semanticflow/agents/orchestrator.py`.

Modes:
- `baseline_single_agent`: single LLM spec only
- `multi_agent_no_hitl`: 3 LLMs + consensus, no human loop
- `multi_agent_hitl`: 3 LLMs + consensus + clarifying questions

### SemanticMapperAgent
- Calls OpenAI, Anthropic, and Gemini in parallel.
- Expects strict JSON and attempts a single repair if invalid (this is a second LLM call).
- Computes Jensen-Shannon divergence (JSD) to estimate epistemic uncertainty.
- Produces a weighted-vote consensus spec.
- Logs: `model_outputs`, `model_confidences`, `epistemic_uncertainty`, `semantic_agreement_score`.
  - Debug: `llm_debug` includes per-provider status, attempts, and error details.
  - If `SEMANTICFLOW_LOG_LLM_RAW=true`, raw responses are saved to artifacts (`llm_raw.json`).

### AmbiguityAnalyzerAgent
- Runs when agreement < threshold.
- Generates **specific, actionable** clarifying questions based on detected conflicts:
  - Analyzes disagreements between model outputs (metrics, measures, dimensions, granularity)
  - Questions include concrete options: "Do you want COUNT or SUM?"
  - References actual schema columns available
  - Avoids vague questions like "Can you clarify?"
- Rule-based fallback when LLM unavailable
- Questions are validated to filter out overly vague ones

### DesignerAgent
- Maps semantic specs to dbt semantic model + metric specs.
- **Generic inference** from schema context:
  - Entities: Infers primary/foreign keys from column naming conventions (`*_id`)
  - Measures: Infers aggregation (COUNT/SUM) from column names and data types
  - Dimensions: Detects time dimensions from names (`*_date`, `*_time`) and SQL types
- Metric name normalization to snake_case
- Auto-suggests metric names from measures + dimensions if not provided
- Determines when a time spine is required.

### CodegenAgent
- Writes/merges YAML files into the dbt project (**consolidated files** - dbt best practice):
  - All semantic models: `models/semantic/semantic_models.yml`
  - All metrics: `models/semantic/metrics.yml`
- Creates backups with `.bak.<timestamp>` on merge.
- Writes/merges tests into `models/schema.yml`.
- Creates a time spine SQL model when needed:
  - `models/semantic/metricflow_time_spine.sql`

### ExecutorAgent
- Runs `dbt parse` and `dbt build` (full build by default).
- Auto-runs `dbt seed` if build errors indicate missing raw tables.
- Runs MetricFlow validation (`mf validate-configs` or `dbt sl validate`).
- Runs MetricFlow query (`mf query`) with:
  - Metric name
  - Group-by mapping to MetricFlow canonical names
  - Query-time filters (relative time windows like `last_30_days`)

### EvaluatorAgent
- Semantic accuracy vs `expected_semantic`.
- Execution accuracy when `gold_sql` or `gold_query_result` is present.
- Uses `evaluation/result_compare.py` for tolerant, order-aware comparison.

### ErrorRecoveryAgent
- **Self-healing** capability for common dbt/MetricFlow errors:
  - Missing column → suggests similar column from schema
  - Invalid dimension name format → auto-fix to `metric_time`
  - Missing time spine → triggers creation
  - Duplicate metric → updates existing
- Pattern-based error detection with confidence scoring
- Auto-applies high-confidence fixes (>70%)

### ResultValidatorAgent
- **Sanity checks** on query results:
  - Empty results warning (filters too restrictive?)
  - Null value detection in key columns
  - Numeric outlier detection (IQR-based)
  - Negative values in count/amount columns
  - Duplicate row detection
  - Time series gap analysis
- Generates actionable suggestions for each issue

### RefinementEngine
- **Iterative improvement** based on multiple feedback types:
  - Human clarifications (HITL answers)
  - Execution errors
  - Validation issues
  - Direct user corrections
  - Result mismatches
- Rule-based fallback when LLM unavailable
- Max iteration limit to prevent infinite loops
- Tracks refinement history

## Human-in-the-Loop (HITL) Workflow
When running in `multi_agent_hitl` mode:

1. **Consensus Detection**: Three LLMs interpret the request in parallel
2. **Ambiguity Detection**: If models disagree (agreement < threshold), ambiguity is flagged
3. **Smart Questions**: The AmbiguityAnalyzer generates specific questions:
   - "Which metric do you want? Options: 'order_count', 'revenue'. Is this COUNT or SUM?"
   - "What time granularity? Options: day, week, month."
   - "What time period? e.g., 'last 30 days', 'this month'"
4. **User Response**: User answers the questions
5. **Refinement**: SemanticMapper refines the spec based on answers
6. **Execution**: Pipeline continues with refined spec

The `OrchestrationResult` now includes:
- `auto_fixes_applied`: List of automatic error fixes applied
- `validation_issues`: Query result warnings/errors
- `refinement_iterations`: Number of refinement cycles

## MetricFlow rules and conventions
- Relative time windows (e.g., `last_30_days`) are translated to valid MetricFlow Jinja syntax:
  - `{{ TimeDimension('metric_time', 'day') }} >= current_date - interval '30 days'`
- Metric YAML only includes static filters that define the metric itself.
- Group-by mapping uses MetricFlow names such as `metric_time__day` or `order__order_date__day`.
- Dimension names must be either `metric_time` or `entity__dimension` format.

## CLI usage
List tasks:
```
python -m semanticflow.cli list-tasks --tasks-path data/real_tasks.jsonl
```

Run one task:
```
python -m semanticflow.cli run-task --task-id real_001
```

Options:
- `--mode`, `--hitl`, `--skip-dbt`, `--tasks-path`
- `--experiment-id`, `--results-path` (to control logging destination)

Run batch experiment:
```
python -m semanticflow.cli run-experiment --mode multi_agent_no_hitl --tasks-path data/real_tasks.jsonl
```

### Uncertainty + selective-HITL analysis (offline, no API calls)
These read a saved experiment's `results.jsonl` and need no LLM credits:
```
# Is any uncertainty signal real? AUROC vs execution-correctness.
python -m semanticflow.cli validate-uncertainty --experiment exp-nohitl

# Effort-accuracy sweep: now also reports the ORACLE upper bound (what a perfect
# signal could buy) and questions-at-fixed-accuracy (cheapest point reaching a target).
python -m semanticflow.cli sweep-threshold --no-hitl exp-nohitl --hitl exp-hitl \
    --signal total_uncertainty

# Conformal Layer 3: calibrate the HITL trigger with a finite-sample risk guarantee.
# alpha = target error rate among auto-answered (un-asked) tasks.
python -m semanticflow.cli conformal-threshold --experiment exp-nohitl \
    --signal total_uncertainty --alpha 0.3
```

### Execution-based consistency signal (strongest black-box UQ for text-to-SQL)
Set `execution_consistency = true` (env `SEMANTICFLOW_EXECUTION_CONSISTENCY=true`) to
execute each provider's representative spec and cluster the result tables by
execution-equivalence, logging `execution_uncertainty` / `total_uncertainty_exec`. Costs
extra dbt/mf runs (default off).

## Experiments and artifacts
Each run is logged to `experiments/<experiment_id>/<run_id>/` and appended to a JSONL results file.

Artifacts include:
- Copies of generated YAML/SQL
- dbt logs
- MetricFlow validate/query logs
- Final semantic spec and model outputs
- Optional raw LLM responses (`llm_raw.json` when enabled)

Experiment summary statistics are written to:
- `experiments/<experiment_id>/summary.json`

## Notes and known warnings
- dbt may warn about time spine YAML configuration (non-fatal).
- MetricFlow CLI versions vary. If `mf` is not on PATH, set `MF_CLI_PATH=./.venv/bin/mf`.
- If you see empty metric results, it may be due to relative time filters vs sample data recency.
- If LLM calls are slow, reduce retries (`SEMANTICFLOW_MAX_LLM_REPAIR=0`) or lower `SEMANTICFLOW_LLM_TIMEOUT_SEC`.
- Enable LLM caching during development with `SEMANTICFLOW_LLM_CACHE=true` to speed up iteration.

## Running Experiments

### Experiment Modes
- `baseline_single_agent`: Single LLM, no consensus, no HITL
- `multi_agent_no_hitl`: 3 LLMs with consensus voting, no human clarification
- `multi_agent_hitl`: 3 LLMs + consensus + human clarifying questions

### Step 1: Run Baseline Experiment (Single Agent)
```bash
python -m semanticflow.cli run-experiment \
  --mode baseline_single_agent \
  --tasks-path data/real_tasks.jsonl \
  --experiment-id exp-baseline
```

### Step 2: Run Multi-Agent with HITL
```bash
python -m semanticflow.cli run-experiment \
  --mode multi_agent_hitl \
  --tasks-path data/real_tasks.jsonl \
  --experiment-id exp-multiagent-hitl
```

### Step 3: Compare Results
```bash
python -m semanticflow.cli compare-experiments \
  --baseline exp-baseline \
  --treatment exp-multiagent-hitl
```

Example output (illustrative format only — the reported results are in the
dissertation manuscript, which is not distributed with this repository):
```
============================================================
EXPERIMENT COMPARISON: baseline vs multi_agent_hitl
============================================================

Metric                                Baseline   Treatment          Δ
-----------------------------------------------------------------
execution_accuracy_avg                   27.00%      45.70%    +20.00pp   (95% CI [+5.7, +34.3], McNemar p=0.039)
dbt_build_success_rate                  100.00%     100.00%     +0.00%
mf_query_success_rate                    74.30%      65.70%     -8.60%

-----------------------------------------------------------------
Tasks compared: 35

By Ambiguity Level (execution accuracy):
  low/medium: baseline=27.80%, treatment=27.80%, Δ=+0.00%
  high:       baseline=23.50%, treatment=64.70%, Δ=+41.20pp  (CI [+17.6, +64.7], p=0.016, 7 gains / 0 regressions)

Winner: TREATMENT (gain concentrated entirely in the high-ambiguity stratum)
============================================================
```

### Save Comparison Report
```bash
python -m semanticflow.cli compare-experiments \
  --baseline exp-baseline \
  --treatment exp-multiagent-hitl \
  --output comparison_report.json
```

### Evaluation Metrics
| Metric | Description |
|--------|-------------|
| `semantic_accuracy_avg` | How well the generated spec matches `expected_semantic` |
| `dbt_parse_success_rate` | % of tasks that parse without errors |
| `dbt_build_success_rate` | % of tasks that build successfully |
| `mf_validate_success_rate` | % that pass MetricFlow validation |
| `mf_query_success_rate` | % that execute queries successfully |
| `execution_accuracy_avg` | How well results match gold SQL (if provided) |

### Per-Task Scoring
- `overall_score`: Average of all semantic checks (0-1)
- `metric_name_match`: Did it pick the right metric?
- `measures_f1`: F1 score for base measures
- `dimensions_f1`: F1 score for group_by dimensions
- `filters_accuracy`: Did filters match expected?

## Running Tests
Run all tests:
```bash
python -m tests.test_filter_translation
python -m tests.test_designer
python -m tests.test_new_features
```

Current test counts:
- Filter translation: 19 tests
- Designer/inference: 13 tests
- New features (error recovery, validation, refinement, caching): 15 tests
- **Total: 47 tests**

## Windows Support
- The system automatically handles Unicode encoding for `mf` CLI output on Windows by setting `PYTHONUTF8=1` and `PYTHONIOENCODING=utf-8` in subprocess calls.
- Path resolution for `dbt` and `MetricFlow` integration has been patched to ensure both tools execute from the project directory and share the same DuckDB database file (`jaffle_shop.duckdb`), resolving potential catalog errors.

## Quickstart (Git Bash)
To start running tasks immediately using Git Bash on Windows:

1. **Load Environment Variables**:
   ```bash
   # Load .env variables (exporting them)
   set -a
   source .env
   set +a
   ```

2. **Activate Virtual Environment**:
   ```bash
   # Activate the uv-managed venv
   source .venv/Scripts/activate
   or
   source .venv/bin/activate

   ```

3. **Run a Task**:
   ```bash
   # Run the example task in HITL mode
   python -m semanticflow.cli run-task --task-id real_001 --mode multi_agent_hitl

   # Count orders by order_date at day granularity, where order_date is within the last 30 days
   ```
