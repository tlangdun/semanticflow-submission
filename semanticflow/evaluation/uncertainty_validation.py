"""Validate whether an uncertainty signal actually discriminates correct from
incorrect specs, before investing in a richer signal.

Reads a no-HITL experiment's results.jsonl and, for each candidate signal column
(e.g. ``epistemic_uncertainty`` = bag-of-tokens JSD, ``structural_uncertainty`` =
field-set distance), computes the AUROC of "signal predicts error" along with the
mean signal on correct vs incorrect tasks and accuracy by uncertainty tercile.

AUROC interpretation (positive class = INCORRECT task): 0.5 = signal is noise;
>0.5 = higher uncertainty really does flag wrong outputs; <0.5 = anti-correlated.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

# Below this many tasks in the smaller class, an AUROC is too unstable to report.
_MIN_CLASS = 5

DEFAULT_SIGNALS = (
    "epistemic_uncertainty",        # cross-model, bag-of-tokens JSD ([0, ln2])
    "epistemic_uncertainty_norm",   # same, normalized to [0,1] (JSD / ln2)
    "structural_uncertainty",       # cross-model, field-set Jaccard distance ([0,1])
    "aleatoric_uncertainty",        # within-model self-consistency (needs self_consistency_k > 1)
    "total_uncertainty",            # aleatoric + structural-epistemic (hybrid, option C)
    "execution_uncertainty",        # execution-equivalence clustering (needs execution_consistency)
    "total_uncertainty_exec",       # total_uncertainty + execution_uncertainty (full hybrid)
)


def _auroc_with_ci(
    scores: list[float], is_positive: list[bool], n_boot: int = 2000, seed: int = 12345
) -> dict[str, Any] | None:
    """AUROC of the signal predicting the positive (incorrect) class, with a bootstrap
    95% CI. Returns None when either class is empty."""
    y = np.array(is_positive, dtype=bool)
    s = np.array(scores, dtype=float)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    point = float(roc_auc_score(y, s))
    out: dict[str, Any] = {"auroc": round(point, 4)}
    if min(n_pos, n_neg) < _MIN_CLASS:
        out["unstable"] = True
        out["note"] = f"min class size {min(n_pos, n_neg)} < {_MIN_CLASS}; AUROC unreliable"
        return out
    rng = np.random.default_rng(seed)
    n = len(y)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yi, si = y[idx], s[idx]
        pos = int(yi.sum())
        if pos == 0 or pos == n:
            continue  # degenerate resample with only one class present
        boot.append(roc_auc_score(yi, si))
    if boot:
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out["ci95"] = [round(float(lo), 4), round(float(hi), 4)]
    return out


def _is_correct(row: dict[str, Any]) -> bool | None:
    """A task counts as correct only when scored by execution-equivalence and matched.
    Returns None when there is no trustworthy (execution) label."""
    if row.get("accuracy_basis") != "execution":
        return None
    if row.get("execution_match") is not None:
        return bool(row["execution_match"])
    acc = row.get("semantic_accuracy")
    return None if acc is None else float(acc) >= 1.0


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def validate_uncertainty(
    experiment_id: str,
    signals: tuple[str, ...] = DEFAULT_SIGNALS,
) -> dict[str, Any]:
    results_file = Path("experiments") / experiment_id / "results.jsonl"
    if not results_file.exists():
        return {"error": f"results not found: {results_file}"}
    rows = [json.loads(line) for line in results_file.read_text().splitlines() if line.strip()]

    labelled = [(r, _is_correct(r)) for r in rows]
    usable = [(r, c) for r, c in labelled if c is not None]
    n_correct = sum(1 for _, c in usable if c)
    n_incorrect = sum(1 for _, c in usable if not c)

    report: dict[str, Any] = {
        "experiment_id": experiment_id,
        "n_rows": len(rows),
        "n_execution_labelled": len(usable),
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "signals": {},
    }
    if n_correct == 0 or n_incorrect == 0:
        report["warning"] = ("need both correct and incorrect execution-labelled tasks "
                             "to compute AUROC; run more tasks or check the gold benchmark.")
        return report

    for sig in signals:
        pairs = [(float(r[sig]), not c) for r, c in usable if r.get(sig) is not None]
        if not pairs:
            report["signals"][sig] = {"error": "signal absent from results"}
            continue
        scores = [s for s, _ in pairs]
        is_error = [p for _, p in pairs]
        # accuracy by tercile of the signal — np.array_split makes near-equal-size bins
        # even when the count isn't divisible by 3 (the old floor-third left the high bin
        # oversized and broke for n < 3).
        ordered = sorted(zip(scores, [not e for e in is_error]), key=lambda x: x[0])
        correct_flags = [1.0 if corr else 0.0 for _, corr in ordered]
        labels = ("low_uncertainty", "mid_uncertainty", "high_uncertainty")
        thirds = np.array_split(np.array(correct_flags), 3) if correct_flags else []
        bins = {
            label: (float(part.mean()) if len(part) else None)
            for label, part in zip(labels, thirds)
        }
        report["signals"][sig] = {
            "auroc_predicts_error": _auroc_with_ci(scores, is_error),
            "mean_on_correct": _mean([s for s, e in pairs if not e]),
            "mean_on_incorrect": _mean([s for s, e in pairs if e]),
            "accuracy_by_uncertainty_tercile": bins,
            "n": len(pairs),
        }
    return report
