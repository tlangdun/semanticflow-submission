# TPC-H second-schema benchmark (SemanticFlow generalisation check)

Second dbt/MetricFlow schema used to test whether SemanticFlow's high-ambiguity
clarification lift generalises beyond Jaffle Shop (dissertation §"Does it generalise?").
40 NL tasks live in `data/tpch_tasks.jsonl`.

## Regenerating the warehouse (`tpch.duckdb`, gitignored)

The DuckDB file (~26 MB at scale factor 0.05) is not committed; regenerate it natively:

```sql
-- in: duckdb third_party/tpch_duckdb/tpch.duckdb
INSTALL tpch; LOAD tpch;
CALL dbgen(sf = 0.05);
-- the dbt models read raw_-prefixed tables:
CREATE TABLE raw_lineitem AS SELECT * FROM lineitem;
CREATE TABLE raw_orders   AS SELECT * FROM orders;
CREATE TABLE raw_customer AS SELECT * FROM customer;
CREATE TABLE raw_nation   AS SELECT * FROM nation;
```

Then `dbt build` (run from inside this dir) materialises `orders` + `line_items`, and the
SemanticFlow pipeline writes `models/semantic/semantic_models.yml` + `metrics.yml` per run.

## Running the pipeline on this schema

```bash
DBT_PROJECT_DIR=third_party/tpch_duckdb DUCKDB_PATH=third_party/tpch_duckdb/tpch.duckdb \
  uv run python main.py run-experiment --mode multi_agent_hitl \
  --tasks-path data/tpch_tasks.jsonl --experiment-id tpch-guard-hitl
```

Both env vars are required for a non-Jaffle run (project dir drives type enrichment;
`DUCKDB_PATH` is the global gold-scoring path). The profile is named `jaffle_shop` so it
matches `settings.dbt_profile_name`. Models carry `{model}_id` key columns so the designer
can infer the MetricFlow primary entity.
