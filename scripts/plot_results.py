"""Turn SemanticFlow result JSON into the three thesis figures.

Figures (each produced only if its input is given):
  1. ROC curves  — uncertainty signal vs execution-correctness   (from a no-HITL run)
  2. Effort-accuracy curve                                        (from sweep-threshold JSON)
  3. Accuracy by ambiguity, baseline vs treatment                (from compare-experiments JSON)

matplotlib is optional: PNGs are written when it is installed; the underlying series are
ALWAYS written as CSV so the figures can be drawn elsewhere.

Examples:
  python -m semanticflow.cli sweep-threshold --no-hitl exp-nohitl --hitl exp-hitl \
      --signal total_uncertainty --output figures/sweep.json
  python -m semanticflow.cli compare-experiments --baseline exp-baseline \
      --treatment exp-hitl --output figures/compare.json
  python scripts/plot_results.py --no-hitl exp-nohitl --sweep figures/sweep.json \
      --compare figures/compare.json --outdir figures
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

SIGNALS = ("epistemic_uncertainty", "structural_uncertainty",
           "aleatoric_uncertainty", "total_uncertainty")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False


def _load_jsonl(exp_id: str) -> list[dict]:
    path = Path("experiments") / exp_id / "results.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _is_correct(row: dict) -> bool | None:
    if row.get("accuracy_basis") != "execution":
        return None
    if row.get("execution_match") is not None:
        return bool(row["execution_match"])
    acc = row.get("semantic_accuracy")
    return None if acc is None else float(acc) >= 1.0


def _roc_points(scores: list[float], is_error: list[bool]) -> list[tuple[float, float]]:
    """(FPR, TPR) for 'high signal => error'. Includes (0,0) and (1,1)."""
    P = sum(is_error)
    N = len(is_error) - P
    if P == 0 or N == 0:
        return [(0.0, 0.0), (1.0, 1.0)]
    pts = [(0.0, 0.0)]
    for thr in sorted(set(scores), reverse=True):
        tp = sum(1 for s, e in zip(scores, is_error) if s >= thr and e)
        fp = sum(1 for s, e in zip(scores, is_error) if s >= thr and not e)
        pts.append((fp / N, tp / P))
    pts.append((1.0, 1.0))
    return pts


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def fig_roc(exp_id: str, outdir: Path) -> None:
    rows = _load_jsonl(exp_id)
    labelled = [(r, c) for r in rows if (c := _is_correct(r)) is not None]
    series = {}
    for sig in SIGNALS:
        pairs = [(float(r[sig]), not c) for r, c in labelled if r.get(sig) is not None]
        if len({e for _, e in pairs}) < 2:
            continue
        series[sig] = _roc_points([s for s, _ in pairs], [e for _, e in pairs])
    if not series:
        print("  ROC: no signal had both correct and incorrect labelled tasks — skipped")
        return
    for sig, pts in series.items():
        _write_csv(outdir / f"roc_{sig}.csv", ["fpr", "tpr"], [[a, b] for a, b in pts])
    def _auc(pts):
        return sum((x2 - x1) * (y1 + y2) / 2 for (x1, y1), (x2, y2) in zip(pts, pts[1:]))
    if HAVE_MPL:
        fig, ax = plt.subplots(figsize=(5.2, 5.2))
        for sig, pts in sorted(series.items(), key=lambda kv: -_auc(kv[1])):
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker=".", markersize=4, linewidth=1.3,
                    label=f"{sig.replace('_uncertainty','').replace('_',' ')} (AUC {_auc(pts):.2f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="chance")
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title("Uncertainty signals vs. execution error", fontsize=11)
        ax.grid(linestyle=":", alpha=0.5); ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        fig.tight_layout(); fig.savefig(outdir / "fig1_roc.png", dpi=200); plt.close(fig)
    print(f"  ROC: {len(series)} signal(s) -> {outdir}/roc_*.csv" + (" + fig1_roc.png" if HAVE_MPL else ""))


def fig_sweep(sweep_json: Path, outdir: Path) -> None:
    data = json.loads(sweep_json.read_text())
    curve = data.get("curve", [])
    if not curve:
        print("  Sweep: empty curve — skipped"); return
    _write_csv(outdir / "sweep.csv", ["threshold", "trigger_fraction", "questions", "mean_accuracy"],
               [[c["threshold"], c["trigger_fraction"], c["questions_asked"], c["mean_accuracy"]] for c in curve])
    if HAVE_MPL:
        qs = [c["questions_asked"] for c in curve]
        acc = [c["mean_accuracy"] * 100 for c in curve]
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        ax.plot(qs, acc, marker="o", color="#3182bd", linewidth=1.5, markersize=5)
        if data.get("best_point"):
            b = data["best_point"]
            ax.scatter([b["questions_asked"]], [b["mean_accuracy"] * 100], color="#e6550d", zorder=5, s=60,
                       label=f"Selective optimum ($\\tau={b['threshold']}$)")
        # annotate never-ask (max questions) and always-ask (0 questions) endpoints
        ax.annotate("never-ask", (qs[-1], acc[-1]), textcoords="offset points", xytext=(-6, -12), fontsize=8, ha="right")
        ax.annotate("always-ask", (qs[0], acc[0]), textcoords="offset points", xytext=(6, -12), fontsize=8)
        ax.set_xlabel("Clarifying questions asked (effort)"); ax.set_ylabel("Mean execution accuracy (%)")
        ax.set_title("Effort-accuracy frontier (cheap tier, total_uncertainty trigger)")
        ax.grid(linestyle=":", alpha=0.5); ax.set_axisbelow(True)
        if data.get("best_point"): ax.legend(frameon=False, loc="lower right")
        fig.tight_layout(); fig.savefig(outdir / "fig2_effort_accuracy.png", dpi=200); plt.close(fig)
    print(f"  Sweep: {len(curve)} points -> {outdir}/sweep.csv" + (" + fig2_effort_accuracy.png" if HAVE_MPL else ""))


def fig_ambiguity(compare_json: Path, outdir: Path) -> None:
    data = json.loads(compare_json.read_text())
    by = data.get("by_ambiguity", {})
    if not by:
        print("  Ambiguity: no by_ambiguity block — skipped"); return
    levels = [l for l in ("low", "medium", "high") if l in by]
    _write_csv(outdir / "by_ambiguity.csv", ["level", "baseline", "treatment", "delta"],
               [[l, by[l]["baseline"], by[l]["treatment"], by[l]["delta"]] for l in levels])
    if HAVE_MPL:
        x = list(range(len(levels)))
        w = 0.38
        base = [by[l]["baseline"] * 100 for l in levels]
        treat = [by[l]["treatment"] * 100 for l in levels]
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        b1 = ax.bar([i - w/2 for i in x], base, w, label="No-HITL ensemble", color="#9ecae1", edgecolor="black", linewidth=0.6)
        b2 = ax.bar([i + w/2 for i in x], treat, w, label="Guard-HITL", color="#3182bd", edgecolor="black", linewidth=0.6)
        for bars in (b1, b2):
            for r in bars:
                ax.annotate(f"{r.get_height():.0f}", (r.get_x() + r.get_width()/2, r.get_height()),
                            textcoords="offset points", xytext=(0, 2), ha="center", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels([l.capitalize() for l in levels])
        ax.set_xlabel("Ambiguity stratum"); ax.set_ylabel("Execution accuracy (%)")
        ax.set_ylim(0, 118)
        ax.set_title("Execution accuracy by ambiguity stratum (cheap tier)")
        ax.grid(axis="y", linestyle=":", alpha=0.5); ax.set_axisbelow(True)
        ax.legend(frameon=False, loc="upper center", ncol=2)
        fig.tight_layout(); fig.savefig(outdir / "fig3_accuracy_by_ambiguity.png", dpi=200); plt.close(fig)
    print(f"  Ambiguity: {len(levels)} levels -> {outdir}/by_ambiguity.csv" + (" + fig3_accuracy_by_ambiguity.png" if HAVE_MPL else ""))


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot SemanticFlow result figures.")
    ap.add_argument("--no-hitl", dest="no_hitl", help="no-HITL experiment id (for ROC, Fig 1)")
    ap.add_argument("--sweep", help="sweep-threshold --output JSON (Fig 2)")
    ap.add_argument("--compare", help="compare-experiments --output JSON (Fig 3)")
    ap.add_argument("--outdir", default="figures", help="output directory")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if not HAVE_MPL:
        print("matplotlib not installed — writing CSVs only (uv add matplotlib for PNGs).")
    if not any([args.no_hitl, args.sweep, args.compare]):
        ap.error("provide at least one of --no-hitl / --sweep / --compare")

    if args.no_hitl:
        fig_roc(args.no_hitl, outdir)
    if args.sweep:
        fig_sweep(Path(args.sweep), outdir)
    if args.compare:
        fig_ambiguity(Path(args.compare), outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
