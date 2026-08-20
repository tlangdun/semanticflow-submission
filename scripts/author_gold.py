"""Author + verify executable gold SQL for the deterministic (low/medium-ambiguity)
benchmark tasks, and rewrite relative-time prompts to absolute 2018 windows (option A).

High-ambiguity tasks are intentionally left without gold_sql: they have no single
valid answer and are scored post-HITL, not on autonomous execution match.

Run: python scripts/author_gold.py        (verifies, prints report, rewrites jsonl)
     python scripts/author_gold.py --check (verify only, no write)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "data" / "real_tasks.jsonl"
DB = ROOT / "jaffle_shop.duckdb"

# Numeric comparisons get a relative tolerance; pure-integer/label results are exact.
TOL = {"rel_tol": 0.01, "abs_tol": 1e-6}

# task_id -> spec. `sql` is ground truth. `rules` -> compare_rules.
# `nl`/`filters` overwrite the prompt / expected filters for rewritten time tasks.
GOLD: dict[str, dict] = {
    "real_001": {
        "nl": "Show me the number of orders per day in March 2018.",
        "filters": ["order_date >= '2018-03-01'", "order_date <= '2018-03-31'"],
        "sql": "SELECT order_date, COUNT(*) AS orders FROM main.orders "
               "WHERE order_date BETWEEN '2018-03-01' AND '2018-03-31' "
               "GROUP BY order_date ORDER BY order_date",
        "rules": {"ordering": "set", "column_agnostic": True},
    },
    "real_002": {
        "sql": "SELECT customer_id, customer_lifetime_value FROM main.customers "
               "ORDER BY customer_lifetime_value DESC, customer_id LIMIT 10",
        "rules": {"ordering": "ordered", "column_agnostic": True, **TOL},
    },
    "real_003": {
        "sql": "SELECT customer_id, SUM(amount) AS gross_revenue FROM main.orders "
               "GROUP BY customer_id ORDER BY customer_id",
        "rules": {"ordering": "set", "column_agnostic": True, **TOL},
    },
    "real_004": {
        "sql": "SELECT customer_id, COUNT(*) AS number_of_orders FROM main.orders "
               "GROUP BY customer_id ORDER BY customer_id",
        "rules": {"ordering": "set", "column_agnostic": True},
    },
    "real_005": {
        "sql": "SELECT payment_method, COUNT(*) AS payment_count FROM main.stg_payments "
               "GROUP BY payment_method ORDER BY payment_method",
        "rules": {"ordering": "set", "column_agnostic": True},
    },
    "real_006": {
        "sql": "SELECT date_trunc('week', order_date) AS order_week, SUM(amount) AS gross_revenue "
               "FROM main.orders GROUP BY 1 ORDER BY 1",
        "rules": {"ordering": "set", "column_agnostic": True, **TOL},
    },
    "real_007": {
        "sql": "SELECT order_date, COUNT(*) AS order_count FROM main.orders "
               "WHERE status = 'completed' GROUP BY order_date ORDER BY order_date",
        "rules": {"ordering": "set", "column_agnostic": True},
    },
    "real_008": {  # credit card revenue vs all other payment types (from orders columns)
        "sql": "SELECT SUM(credit_card_amount) AS credit_card_revenue, "
               "SUM(coupon_amount + bank_transfer_amount + gift_card_amount) AS other_revenue "
               "FROM main.orders",
        "rules": {"ordering": "set", "column_agnostic": True, **TOL},
    },
    "real_009": {
        "nl": "How many customers placed their first order in Q1 2018 (Jan-Mar)?",
        "filters": ["first_order >= '2018-01-01'", "first_order <= '2018-03-31'"],
        "sql": "SELECT COUNT(*) AS new_customers FROM main.customers "
               "WHERE first_order BETWEEN '2018-01-01' AND '2018-03-31'",
        "rules": {"ordering": "set", "column_agnostic": True},
    },
    "real_010": {
        "sql": "SELECT date_trunc('month', order_date) AS order_month, AVG(amount) AS avg_order_value "
               "FROM main.orders GROUP BY 1 ORDER BY 1",
        "rules": {"ordering": "set", "column_agnostic": True, **TOL},
    },
    "real_011": {
        "nl": "Which customers have a lifetime value over $30?",
        "filters": ["customer_lifetime_value > 30"],
        "sql": "SELECT customer_id FROM main.customers "
               "WHERE customer_lifetime_value > 30 ORDER BY customer_id",
        "rules": {"ordering": "set", "column_agnostic": True},
    },
    "real_014": {
        "nl": "Count of returned and return-pending orders by month for Q1 2018.",
        "filters": ["status IN ('returned', 'return_pending')",
                    "order_date >= '2018-01-01'", "order_date <= '2018-03-31'"],
        "sql": "SELECT date_trunc('month', order_date) AS order_month, status, COUNT(*) AS order_count "
               "FROM main.orders WHERE status IN ('returned', 'return_pending') "
               "AND order_date BETWEEN '2018-01-01' AND '2018-03-31' "
               "GROUP BY 1, 2 ORDER BY 1, 2",
        "rules": {"ordering": "set", "column_agnostic": True},
    },
    "real_015": {
        "sql": "SELECT p.payment_method, SUM(p.amount) AS revenue "
               "FROM main.stg_payments p JOIN main.orders o ON p.order_id = o.order_id "
               "WHERE o.status = 'completed' GROUP BY 1 ORDER BY 1",
        "rules": {"ordering": "set", "column_agnostic": True, **TOL},
    },
    "real_017": {
        "nl": "Top 5 customers by number of orders in March 2018.",
        "filters": ["order_date >= '2018-03-01'", "order_date <= '2018-03-31'"],
        "sql": "SELECT customer_id, COUNT(*) AS order_count FROM main.orders "
               "WHERE order_date BETWEEN '2018-03-01' AND '2018-03-31' "
               "GROUP BY customer_id ORDER BY order_count DESC, customer_id LIMIT 5",
        "rules": {"ordering": "ordered", "column_agnostic": True},
    },
    "real_018": {
        "sql": "SELECT order_date, COUNT(*) AS order_count FROM main.orders "
               "WHERE status != 'returned' GROUP BY order_date ORDER BY order_date",
        "rules": {"ordering": "set", "column_agnostic": True},
    },
    "real_019": {
        "sql": "SELECT 100.0 * SUM(CASE WHEN coupon_amount > 0 THEN 1 ELSE 0 END) / COUNT(*) "
               "AS coupon_usage_pct FROM main.orders",
        "rules": {"ordering": "set", "column_agnostic": True, **TOL},
    },
    "real_020": {
        "sql": "SELECT date_trunc('month', first_order) AS acquisition_month, COUNT(*) AS new_customers "
               "FROM main.customers WHERE first_order IS NOT NULL GROUP BY 1 ORDER BY 1",
        "rules": {"ordering": "set", "column_agnostic": True},
    },
    "real_026": {
        "sql": "SELECT customer_id, COUNT(*) AS order_count FROM main.orders "
               "GROUP BY customer_id ORDER BY customer_id",
        "rules": {"ordering": "set", "column_agnostic": True},
    },
}

# High-ambiguity tasks scored against the INTENDED interpretation (from expected_semantic).
# These are the post-HITL gold: autonomously the model may pick a different valid reading and
# fail; HITL should recover the intended one. This makes the HITL stratum measurable.
TOL_R = {"rel_tol": 0.01, "abs_tol": 1e-6}
GOLD_HIGH: dict[str, dict] = {
    "real_012": {"sql": "SELECT SUM(amount) AS total_sales FROM main.orders",
                 "rules": {"ordering": "set", "column_agnostic": True, **TOL_R}},
    "real_013": {"sql": "SELECT order_date, COUNT(*) AS order_count FROM main.orders GROUP BY 1 ORDER BY 1",
                 "rules": {"ordering": "set", "column_agnostic": True}},
    "real_016": {"sql": "SELECT date_trunc('month', order_date) AS m, SUM(amount) AS rev FROM main.orders "
                        "WHERE order_date BETWEEN '2018-03-01' AND '2018-04-30' GROUP BY 1 ORDER BY 1",
                 "rules": {"ordering": "set", "column_agnostic": True, **TOL_R}},
    "real_021": {"sql": "SELECT SUM(amount) AS gross_revenue FROM main.orders",
                 "rules": {"ordering": "set", "column_agnostic": True, **TOL_R}},
    "real_022": {"sql": "SELECT COUNT(*) AS order_count FROM main.orders",
                 "rules": {"ordering": "set", "column_agnostic": True}},
    "real_023": {"sql": "SELECT customer_id, COUNT(*) AS order_count FROM main.orders GROUP BY 1 ORDER BY 1",
                 "rules": {"ordering": "set", "column_agnostic": True}},
    "real_024": {"sql": "SELECT order_date, SUM(amount) AS gross_revenue FROM main.orders GROUP BY 1 ORDER BY 1",
                 "rules": {"ordering": "set", "column_agnostic": True, **TOL_R}},
    "real_025": {"sql": "SELECT order_date, SUM(amount) AS gross_revenue FROM main.orders GROUP BY 1 ORDER BY 1",
                 "rules": {"ordering": "set", "column_agnostic": True, **TOL_R}},
    "real_027": {"sql": "SELECT order_date, COUNT(*) AS order_count FROM main.orders "
                        "WHERE order_date BETWEEN '2018-04-01' AND '2018-04-09' GROUP BY 1 ORDER BY 1",
                 "rules": {"ordering": "set", "column_agnostic": True}},
    "real_028": {"sql": "SELECT payment_method, SUM(amount) AS payment_amount FROM main.stg_payments GROUP BY 1 ORDER BY 1",
                 "rules": {"ordering": "set", "column_agnostic": True, **TOL_R}},
    "real_029": {"sql": "SELECT customer_id, COUNT(*) AS order_count FROM main.orders GROUP BY 1 ORDER BY 1",
                 "rules": {"ordering": "set", "column_agnostic": True}},
    "real_030": {"sql": "SELECT COUNT(*) AS order_count FROM main.orders",
                 "rules": {"ordering": "set", "column_agnostic": True}},
    "real_031": {"sql": "SELECT COUNT(*) AS order_count FROM main.orders",
                 "rules": {"ordering": "set", "column_agnostic": True}},
    "real_032": {"sql": "SELECT status, COUNT(*) AS order_count FROM main.orders GROUP BY 1 ORDER BY 1",
                 "rules": {"ordering": "set", "column_agnostic": True}},
    "real_033": {"sql": "SELECT date_trunc('month', order_date) AS m, SUM(amount) AS rev FROM main.orders GROUP BY 1 ORDER BY 1",
                 "rules": {"ordering": "set", "column_agnostic": True, **TOL_R}},
    "real_034": {"sql": "SELECT customer_id, SUM(amount) AS gross_revenue FROM main.orders GROUP BY 1 ORDER BY 1",
                 "rules": {"ordering": "set", "column_agnostic": True, **TOL_R}},
    "real_035": {"sql": "SELECT order_date, SUM(amount) AS gross_revenue FROM main.orders GROUP BY 1 ORDER BY 1",
                 "rules": {"ordering": "set", "column_agnostic": True, **TOL_R}},
}
GOLD.update(GOLD_HIGH)


def main() -> int:
    check_only = "--check" in sys.argv
    con = duckdb.connect(str(DB), read_only=True)

    tasks = [json.loads(line) for line in TASKS.read_text().splitlines() if line.strip()]
    by_id = {t["task_id"]: t for t in tasks}

    failures = []
    for tid, spec in GOLD.items():
        if tid not in by_id:
            failures.append(f"{tid}: not found in tasks file")
            continue
        try:
            cur = con.execute(spec["sql"])
            cols = [c[0] for c in cur.description] if cur.description else []
            rows = [dict(zip(cols, rec)) for rec in cur.fetchall()]
        except Exception as exc:
            failures.append(f"{tid}: SQL error: {exc}")
            continue
        if not rows:
            failures.append(f"{tid}: gold SQL returned EMPTY (empty-result trap)")
            continue
        print(f"  {tid}: {len(rows):>3} rows  sample={rows[0]}")

        if not check_only:
            t = by_id[tid]
            t["gold_sql"] = spec["sql"]
            t["compare_rules"] = spec["rules"]
            if "nl" in spec:
                t["nl_request"] = spec["nl"]
            if "filters" in spec and isinstance(t.get("expected_semantic"), dict):
                t["expected_semantic"]["filters"] = spec["filters"]

    con.close()

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("  -", f)
        return 1

    with_gold = len(GOLD)
    high_intended = len(GOLD_HIGH)
    print(f"\nOK: {with_gold} tasks have verified non-empty gold SQL "
          f"({with_gold - high_intended} deterministic + {high_intended} high-ambiguity scored "
          f"against the intended/post-HITL interpretation).")

    if not check_only:
        with TASKS.open("w", encoding="utf-8") as fh:
            for t in tasks:
                fh.write(json.dumps(t) + "\n")
        print(f"Wrote {TASKS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
