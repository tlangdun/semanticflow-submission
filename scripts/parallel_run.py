"""Run one SemanticFlow experiment FASTER by sharding its tasks across N isolated
worker copies of the dbt project, then merging the per-shard results.

Why this is needed: every run mutates one shared dbt project dir — the generated
semantic-layer YAML (models/semantic/*.yml), the dbt target/, and the DuckDB warehouse
(profiles.yml uses a RELATIVE path, so each project copy gets its own warehouse). Two runs
over the same dir clobber each other. Giving each worker its own copy removes all shared
mutable state, so shards run truly concurrently. Each task's `dbt_project` is rewritten to
the worker's copy, and DBT_PROJECT_DIR is set per worker (for the type-enrichment lookup
that reads settings.dbt_project_dir).

This also lets you run DIFFERENT experiments at once: launch two of these with different
--experiment-id (they pick disjoint worker dirs), e.g. a cheap and a frontier arm.

Usage:
  PYTHONPATH=. uv run python scripts/parallel_run.py \
      --experiment-id cheap-fast-nohitl --mode multi_agent_no_hitl \
      --tasks-path data/all_tasks.jsonl --workers 4

Pass experiment knobs via the ENVIRONMENT (inherited by every shard), e.g.:
  SEMANTICFLOW_EXPERIMENT_REPEATS=3 SEMANTICFLOW_AGREEMENT_THRESHOLD=1.01 \
  SEMANTICFLOW_TASK_TIMEOUT_SEC=400 OPENAI_MODEL=gpt-5.4 ... <the command above>

Caveats:
  - Workers multiply the API call rate (N x providers x K). Keep --workers modest
    (3-4) to avoid provider rate limits; raising it past (cpu_cores - 1) won't help
    because each task is dbt/mf-subprocess bound.
  - Each worker copy is a few MB (images/logs/target/.git excluded). Cleaned up at the end
    unless --keep is passed.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_PROJECT = ROOT / "third_party" / "jaffle_shop_duckdb"


def _shard(ids: list, n: int) -> list[list]:
    buckets: list[list] = [[] for _ in range(n)]
    for i, x in enumerate(ids):
        buckets[i % n].append(x)
    return [b for b in buckets if b]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment-id", required=True)
    ap.add_argument("--mode", required=True)
    ap.add_argument("--tasks-path", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--keep", action="store_true", help="keep worker dirs + shard exps")
    args = ap.parse_args()

    tasks = [json.loads(l) for l in Path(args.tasks_path).read_text().splitlines() if l.strip()]
    if not tasks:
        print("no tasks"); return 1
    n = max(1, min(args.workers, len(tasks)))
    shards = _shard(tasks, n)
    work = Path("/tmp/sf_workers") / args.experiment_id
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    print(f"sharding {len(tasks)} tasks across {len(shards)} workers under {work}")

    procs = []
    shard_exp_ids = []
    for k, shard in enumerate(shards):
        wdir = work / f"w{k}"
        # Lean copy: skip the heavy/irrelevant dirs; keep models, dbt_project.yml,
        # profiles.yml, the warehouse, seeds, macros.
        subprocess.run(
            ["rsync", "-a",
             "--exclude=images", "--exclude=logs", "--exclude=target",
             "--exclude=.git", "--exclude=.github",
             f"{BASE_PROJECT}/", f"{wdir}/"],
            check=True,
        )
        wdir_abs = str(wdir.resolve())
        # Rewrite each task's dbt_project to this worker's copy.
        shard_tasks = []
        for t in shard:
            t = dict(t); t["dbt_project"] = wdir_abs; shard_tasks.append(t)
        shard_file = work / f"shard_{k}.jsonl"
        shard_file.write_text("".join(json.dumps(t) + "\n" for t in shard_tasks))

        exp_id = f"{args.experiment_id}-shard{k}"
        shard_exp_ids.append(exp_id)
        env = os.environ.copy()
        env["DBT_PROJECT_DIR"] = wdir_abs
        env["PYTHONPATH"] = str(ROOT)
        cmd = ["uv", "run", "python", "main.py", "run-experiment",
               "--mode", args.mode, "--tasks-path", str(shard_file),
               "--experiment-id", exp_id]
        log = open(work / f"w{k}.log", "w")
        procs.append((k, subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)))
        print(f"  worker {k}: {len(shard)} tasks -> exp {exp_id}  (log {work}/w{k}.log)")

    # Wait for all shards.
    failed = []
    for k, p in procs:
        if p.wait() != 0:
            failed.append(k)
    if failed:
        print(f"WARNING: workers {failed} exited non-zero; merging what completed")

    # Merge shard results.jsonl -> experiments/<experiment-id>/results.jsonl
    out_dir = ROOT / "experiments" / args.experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)
    merged = out_dir / "results.jsonl"
    rows = []
    for exp_id in shard_exp_ids:
        f = ROOT / "experiments" / exp_id / "results.jsonl"
        if f.exists():
            rows.extend(l for l in f.read_text().splitlines() if l.strip())
    merged.write_text("\n".join(rows) + ("\n" if rows else ""))
    print(f"merged {len(rows)} task rows -> {merged}")

    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)
        for exp_id in shard_exp_ids:
            shutil.rmtree(ROOT / "experiments" / exp_id, ignore_errors=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
