"""Naive direct-text-to-SQL baseline: a single LLM call writes SQL straight from the
question + schema, we execute it against DuckDB, and score by the SAME execution-equivalence
the full pipeline uses. No multi-agent, no dbt/MetricFlow, no clarification.

This answers "is the semantic-layer apparatus worth it vs just asking an LLM for SQL?"
Output rows are compatible with `compare-experiments` (accuracy_basis='execution',
execution_match, execution_accuracy), so it slots in as another arm.

Usage:
  PYTHONPATH=. uv run python scripts/direct_sql_baseline.py \
      --tasks-path data/all_tasks.jsonl --experiment-id cheap-direct-sql --provider openai
  # frontier: OPENAI_MODEL=gpt-5.4 ... --experiment-id frontier-direct-sql
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import duckdb

from semanticflow.config import load_settings
from semanticflow.evaluation import compare_results, run_gold_sql
from semanticflow.llm import UsageTracker, build_provider_client
from semanticflow.tasks.loader import load_tasks

DUCKDB = "jaffle_shop.duckdb"  # repo-root warehouse; same one the evaluator's gold uses

SYS = ("You are an expert data analyst. Given a question and a database schema, output a "
       "SINGLE DuckDB SQL SELECT query that answers it. Output ONLY the SQL — no markdown "
       "fences, no commentary. All tables are in the 'main' schema (e.g. main.orders).")


def _schema_text(ctx: dict) -> str:
    cols = ctx.get("columns", {})
    return "\n".join(f"  main.{m}({', '.join(cols.get(m, []))})" for m in ctx.get("models", []))


def _extract_sql(raw: str) -> str:
    raw = (raw or "").strip()
    m = re.search(r"```(?:sql)?\s*(.+?)```", raw, re.DOTALL | re.IGNORECASE)
    if m:
        raw = m.group(1).strip()
    # take from the first SELECT/WITH onward
    m = re.search(r"\b(WITH|SELECT)\b", raw, re.IGNORECASE)
    return raw[m.start():].strip().rstrip(";") if m else raw


def _run(sql: str, con) -> tuple[bool, list[dict]]:
    try:
        cur = con.execute(sql)
        cols = [d[0] for d in cur.description]
        return True, [dict(zip(cols, r)) for r in cur.fetchall()]
    except Exception:
        return False, []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-path", required=True)
    ap.add_argument("--experiment-id", required=True)
    ap.add_argument("--provider", default="openai")
    args = ap.parse_args()

    settings = load_settings()
    client = build_provider_client(args.provider, settings, UsageTracker())
    tasks = load_tasks(args.tasks_path)
    con = duckdb.connect(DUCKDB, read_only=True)
    out_dir = Path("experiments") / args.experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for t in tasks:
        prompt = f"Question: {t.nl_request}\n\nSchema:\n{_schema_text(t.schema_context)}\n\nSQL:"
        try:
            raw = client.complete(SYS, prompt, temperature=0.0)
        except Exception as e:
            raw = ""
        sql = _extract_sql(raw)
        ok, pred = _run(sql, con) if sql else (False, [])
        if t.gold_sql:
            gok, gold, _ = run_gold_sql(t.gold_sql, duckdb_path=DUCKDB)
            cmp = compare_results(gold, pred, result_schema=t.result_schema,
                                  compare_rules=t.compare_rules)
            match = bool(cmp.match and ok)
            acc = cmp.accuracy if ok else 0.0
            basis = "execution"
        else:
            match, acc, basis = False, 0.0, "none"  # no gold SQL -> not exec-scorable here
        rows.append({
            "experiment_id": args.experiment_id, "task_id": t.task_id, "mode": "direct_sql",
            "llm_provider": args.provider, "ambiguity": t.ambiguity,
            "accuracy_basis": basis, "execution_accuracy": acc, "execution_match": match,
            "semantic_accuracy": acc, "mf_query_success": ok, "gold_sql_present": bool(t.gold_sql),
            "generated_sql": sql,
        })
        print(f"  {t.task_id:<10} {basis:<9} acc={acc:.2f} match={match} ok={ok}")
    (out_dir / "results.jsonl").write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
    ex = [r for r in rows if r["accuracy_basis"] == "execution"]
    corr = sum(1 for r in ex if r["execution_match"])
    print(f"\n{args.experiment_id}: {corr}/{len(ex)} execution-correct ({len(rows)} total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
