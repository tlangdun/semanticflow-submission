"""Expand the synthetic benchmark with answerable, execution-scorable tasks.

The original synth_001..005 are deliberately unanswerable (they reference concepts
not in the schema — region, segment, churn) and have no gold; they are kept as pure
HITL-ambiguity probes. This script ADDS synth_006.. over the real jaffle_shop schema,
each with verified non-empty gold SQL, stratified by ambiguity/difficulty, to raise n
for the effort-accuracy / power analysis. Idempotent: re-running rewrites synth_006+.

Run: python scripts/build_synthetic.py [--check]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "data" / "synthetic_tasks.jsonl"
DB = ROOT / "jaffle_shop.duckdb"
PROJECT = "third_party/jaffle_shop_duckdb"

COLS = {
    "orders": ["order_id", "customer_id", "order_date", "amount", "status",
               "credit_card_amount", "coupon_amount"],
    "customers": ["customer_id", "first_order", "number_of_orders", "customer_lifetime_value"],
    "stg_payments": ["payment_id", "order_id", "payment_method", "amount"],
}
TOL = {"rel_tol": 0.01, "abs_tol": 1e-6}


def _sc(*models: str) -> dict:
    return {"models": list(models), "columns": {m: COLS[m] for m in models}}


def _agnostic(ordering: str = "set", tol: bool = False) -> dict:
    rules = {"ordering": ordering, "column_agnostic": True}
    if tol:
        rules.update(TOL)
    return rules


# (id, nl, models, ambiguity, difficulty, measure, group_by, gold_sql, ordered, tol)
SPECS = [
    # --- low ambiguity (clear, deterministic) ---
    ("synth_006", "Count orders by status.", ["orders"], "low", "simple",
     "order_count", ["status"], "SELECT status, COUNT(*) AS order_count FROM main.orders GROUP BY 1 ORDER BY 1", False, False),
    ("synth_007", "What is total revenue across all orders?", ["orders"], "low", "simple",
     "gross_revenue", [], "SELECT SUM(amount) AS gross_revenue FROM main.orders", False, True),
    ("synth_008", "How many customers are there in total?", ["customers"], "low", "simple",
     "customer_count", [], "SELECT COUNT(*) AS customer_count FROM main.customers", False, False),
    ("synth_009", "Show the number of orders placed in February 2018.", ["orders"], "low", "simple",
     "order_count", [], "SELECT COUNT(*) AS order_count FROM main.orders WHERE order_date BETWEEN '2018-02-01' AND '2018-02-28'", False, False),
    ("synth_010", "What is the average order amount?", ["orders"], "low", "simple",
     "average_order_value", [], "SELECT AVG(amount) AS avg_order_value FROM main.orders", False, True),
    ("synth_011", "What is the largest single order amount?", ["orders"], "low", "simple",
     "max_order_amount", [], "SELECT MAX(amount) AS max_order_amount FROM main.orders", False, True),
    ("synth_012", "Show total payment amount by payment method.", ["stg_payments"], "low", "simple",
     "payment_amount", ["payment_method"], "SELECT payment_method, SUM(amount) AS payment_amount FROM main.stg_payments GROUP BY 1 ORDER BY 1", False, True),
    ("synth_013", "How many completed orders are there?", ["orders"], "low", "simple",
     "order_count", [], "SELECT COUNT(*) AS order_count FROM main.orders WHERE status = 'completed'", False, False),
    ("synth_014", "Count orders per month.", ["orders"], "low", "medium",
     "order_count", ["order_date"], "SELECT date_trunc('month', order_date) AS m, COUNT(*) AS order_count FROM main.orders GROUP BY 1 ORDER BY 1", False, False),
    ("synth_015", "Total revenue by order status.", ["orders"], "low", "medium",
     "gross_revenue", ["status"], "SELECT status, SUM(amount) AS gross_revenue FROM main.orders GROUP BY 1 ORDER BY 1", False, True),

    # --- medium ambiguity ---
    ("synth_016", "Revenue from completed orders only.", ["orders"], "medium", "medium",
     "gross_revenue", [], "SELECT SUM(amount) AS gross_revenue FROM main.orders WHERE status = 'completed'", False, True),
    ("synth_017", "Top 3 customers by total revenue.", ["orders"], "medium", "medium",
     "gross_revenue", ["customer_id"], "SELECT customer_id, SUM(amount) AS gross_revenue FROM main.orders GROUP BY 1 ORDER BY 2 DESC, 1 LIMIT 3", True, True),
    ("synth_018", "How many orders used a coupon?", ["orders"], "medium", "medium",
     "order_count", [], "SELECT COUNT(*) AS order_count FROM main.orders WHERE coupon_amount > 0", False, False),
    ("synth_019", "What is the average customer lifetime value?", ["customers"], "medium", "simple",
     "avg_clv", [], "SELECT AVG(customer_lifetime_value) AS avg_clv FROM main.customers", False, True),
    ("synth_020", "How many customers placed more than 2 orders?", ["customers"], "medium", "medium",
     "customer_count", [], "SELECT COUNT(*) AS customer_count FROM main.customers WHERE number_of_orders > 2", False, False),
    ("synth_021", "Count of payments per method.", ["stg_payments"], "medium", "simple",
     "payment_count", ["payment_method"], "SELECT payment_method, COUNT(*) AS payment_count FROM main.stg_payments GROUP BY 1 ORDER BY 1", False, False),
    ("synth_022", "Credit card revenue from orders.", ["orders"], "medium", "medium",
     "credit_card_amount", [], "SELECT SUM(credit_card_amount) AS credit_card_revenue FROM main.orders", False, True),

    # --- high ambiguity (scored against the intended interpretation) ---
    ("synth_023", "How many orders?", ["orders"], "high", "simple",
     "order_count", [], "SELECT COUNT(*) AS order_count FROM main.orders", False, False),
    ("synth_024", "Sales by month.", ["orders"], "high", "medium",
     "gross_revenue", ["order_date"], "SELECT date_trunc('month', order_date) AS m, SUM(amount) AS gross_revenue FROM main.orders GROUP BY 1 ORDER BY 1", False, True),
    ("synth_025", "Customer numbers.", ["customers"], "high", "simple",
     "customer_count", [], "SELECT COUNT(*) AS customer_count FROM main.customers", False, False),
    ("synth_026", "What's the order situation?", ["orders"], "high", "medium",
     "order_count", ["status"], "SELECT status, COUNT(*) AS order_count FROM main.orders GROUP BY 1 ORDER BY 1", False, False),
    ("synth_027", "Give me payment info.", ["stg_payments"], "high", "medium",
     "payment_amount", ["payment_method"], "SELECT payment_method, SUM(amount) AS payment_amount FROM main.stg_payments GROUP BY 1 ORDER BY 1", False, True),
    ("synth_028", "Revenue overview.", ["orders"], "high", "simple",
     "gross_revenue", [], "SELECT SUM(amount) AS gross_revenue FROM main.orders", False, True),
    ("synth_029", "Show recent orders.", ["orders"], "high", "medium",
     "order_count", ["order_date"], "SELECT order_date, COUNT(*) AS order_count FROM main.orders WHERE order_date BETWEEN '2018-04-01' AND '2018-04-09' GROUP BY 1 ORDER BY 1", False, False),
    ("synth_030", "Who are the top customers?", ["orders"], "high", "medium",
     "gross_revenue", ["customer_id"], "SELECT customer_id, SUM(amount) AS gross_revenue FROM main.orders GROUP BY 1 ORDER BY 2 DESC, 1 LIMIT 5", True, True),
]


def _record(spec) -> dict:
    tid, nl, models, amb, diff, measure, group_by, sql, ordered, tol = spec
    return {
        "task_id": tid,
        "nl_request": nl,
        "dbt_project": PROJECT,
        "schema_context": _sc(*models),
        "expected_semantic": {"metric": tid, "base_measure": measure, "group_by": group_by},
        "gold_sql": sql,
        "compare_rules": _agnostic("ordered" if ordered else "set", tol),
        "ambiguity": amb,
        "difficulty": diff,
    }


def main() -> int:
    check_only = "--check" in sys.argv
    con = duckdb.connect(str(DB), read_only=True)

    new = {s[0] for s in SPECS}
    existing = []
    if TASKS.exists():
        existing = [json.loads(l) for l in TASKS.read_text().splitlines() if l.strip()]
    kept = [t for t in existing if t.get("task_id") not in new]

    records, failures = [], []
    for spec in SPECS:
        rec = _record(spec)
        try:
            cur = con.execute(rec["gold_sql"])
            cols = [c[0] for c in cur.description] if cur.description else []
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        except Exception as exc:
            failures.append(f"{rec['task_id']}: SQL error: {exc}")
            continue
        if not rows:
            failures.append(f"{rec['task_id']}: gold SQL returned EMPTY")
            continue
        print(f"  {rec['task_id']:<11} {rec['ambiguity']:<6} {len(rows):>3} rows  sample={rows[0]}")
        records.append(rec)
    con.close()

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("  -", f)
        return 1

    print(f"\nOK: {len(records)} answerable synthetic tasks verified; "
          f"{len(kept)} original (unanswerable) tasks kept.")
    if not check_only:
        with TASKS.open("w", encoding="utf-8") as fh:
            for t in kept + records:
                fh.write(json.dumps(t) + "\n")
        print(f"Wrote {TASKS} ({len(kept) + len(records)} tasks total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
