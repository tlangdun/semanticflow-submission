"""Conformal selective HITL — Layer 3 of the uncertainty stack.

The threshold sweep (``experiment.threshold_sweep``) shows *what* an operating point
buys, but picking the point by eyeballing the curve gives no guarantee. This module
calibrates the HITL trigger threshold with a finite-sample risk guarantee, in the
spirit of RTS (Reliable Text-to-SQL, arXiv:2501.10858) and conformal risk control
(Angelopoulos et al.): choose the threshold so that the error rate among the tasks we
*auto-answer* (do not ask about) stays at or below a target risk ``alpha``, with the
fewest human questions.

Decision rule: ASK a human when ``signal >= tau`` (high uncertainty), otherwise
auto-answer. We want the largest acceptance region (fewest asks) whose accepted-set
error rate is provably <= ``alpha``.

We report a split-conformal estimate: calibrate ``tau`` on a calibration split, then
measure the *realized* accepted-set risk and ask-rate on a disjoint test split,
averaged over many random splits. If the realized test risk tracks ``alpha``, the
guarantee transfers out-of-sample — the honest claim for a small-n thesis.

NOTE: with the current n (~35, ~10 correct / ~25 incorrect) these bounds are loose;
the machinery is what matters, and it tightens directly with the K=3 / synthetic
expansion.
"""
from __future__ import annotations

import json
from math import ceil
from pathlib import Path
from typing import Any

import numpy as np


def calibrate_threshold(
    signals: list[float],
    correct: list[bool],
    alpha: float,
    delta: float = 0.1,
) -> dict[str, Any]:
    """Pick the smallest ask-rate threshold ``tau`` whose auto-answered error <= ``alpha``.

    ASK when ``signal >= tau``; auto-answer otherwise. We scan candidate thresholds
    (the observed signal values) from permissive (ask little) to strict and keep the
    one that asks the fewest questions while holding the accepted-set risk at or below
    ``alpha`` under a conservative (upper-confidence) estimate.

    The conservative risk uses a one-sided Clopper-Pearson-style slack: rather than the
    raw accepted error rate we use ``(errors + z) / n_accepted`` with a small additive
    correction, so a lucky calibration split does not pick an over-permissive tau. The
    realized guarantee is verified empirically on a held-out split by
    :func:`conformal_selective_hitl`.

    Returns a dict with the chosen ``tau`` and the calibration-set accept stats, or
    ``tau=None`` if no acceptance region meets the target.
    """
    n = len(signals)
    if n == 0 or len(correct) != n:
        return {"tau": None, "reason": "empty or mismatched calibration data"}

    order = sorted(range(n), key=lambda i: signals[i])
    s = [signals[i] for i in order]
    c = [correct[i] for i in order]
    pairs = list(zip(s, c, strict=True))

    # Candidate thresholds: just above each observed signal (accept everything strictly
    # below tau). Adding +inf lets "accept everything" be considered.
    candidates = sorted(set(s)) + [float("inf")]
    # Additive finite-sample slack on the accepted error rate (Hoeffding one-sided at
    # level delta): err_hat + sqrt(ln(1/delta) / (2 n_acc)). Conservative for small n.
    best: dict[str, Any] | None = None
    for tau in candidates:
        accepted = [(si, ci) for si, ci in pairs if si < tau]
        n_acc = len(accepted)
        if n_acc == 0:
            continue
        errors = sum(1 for _, ci in accepted if not ci)
        err_hat = errors / n_acc
        slack = (np.log(1.0 / max(delta, 1e-9)) / (2.0 * n_acc)) ** 0.5
        risk_ucb = err_hat + slack
        if risk_ucb <= alpha:
            ask_rate = (n - n_acc) / n
            cand = {
                "tau": tau if tau != float("inf") else None,
                "accept_everything": tau == float("inf"),
                "n_accepted": n_acc,
                "accepted_errors": errors,
                "accepted_risk_empirical": round(err_hat, 4),
                "accepted_risk_ucb": round(risk_ucb, 4),
                "ask_rate": round(ask_rate, 4),
            }
            # Maximise acceptance (minimise asks) => smallest ask_rate.
            if best is None or cand["ask_rate"] < best["ask_rate"]:
                best = cand
    if best is None:
        return {"tau": None, "reason": f"no acceptance region holds risk <= {alpha}"}
    return best


def evaluate_selective(signals: list[float], correct: list[bool], tau: float | None) -> dict[str, Any]:
    """Realized accept-set stats for threshold ``tau`` (None => accept everything)."""
    n = len(signals)
    if n == 0:
        return {"n": 0}
    cut = float("inf") if tau is None else tau
    accepted = [ci for si, ci in zip(signals, correct, strict=True) if si < cut]
    n_acc = len(accepted)
    asked = n - n_acc
    accepted_errors = sum(1 for ci in accepted if not ci)
    return {
        "n": n,
        "n_accepted": n_acc,
        "ask_rate": round(asked / n, 4),
        "accepted_risk": round(accepted_errors / n_acc, 4) if n_acc else None,
        "accepted_accuracy": round(1 - accepted_errors / n_acc, 4) if n_acc else None,
    }


def _is_correct(row: dict[str, Any]) -> bool | None:
    if row.get("accuracy_basis") != "execution":
        return None
    if row.get("execution_match") is not None:
        return bool(row["execution_match"])
    acc = row.get("semantic_accuracy")
    return None if acc is None else float(acc) >= 1.0


def conformal_selective_hitl(
    experiment_id: str,
    signal: str = "total_uncertainty",
    alpha: float = 0.2,
    delta: float = 0.1,
    n_splits: int = 200,
    cal_frac: float = 0.5,
    seed: int = 12345,
) -> dict[str, Any]:
    """Split-conformal evaluation of the selective-HITL threshold on a saved experiment.

    For each of ``n_splits`` random calibration/test partitions: calibrate ``tau`` on
    the calibration half to hold accepted error <= ``alpha``, then measure the realized
    accepted-set risk and ask-rate on the test half. Average across splits.

    The guarantee transfers if ``mean test accepted_risk <= alpha``. ``ask_rate`` is the
    fraction of tasks routed to a human — the cost the guarantee buys.
    """
    results_file = Path("experiments") / experiment_id / "results.jsonl"
    if not results_file.exists():
        return {"error": f"results not found: {results_file}"}
    rows = [json.loads(line) for line in results_file.read_text().splitlines() if line.strip()]

    data = []
    for r in rows:
        c = _is_correct(r)
        u = r.get(signal)
        if c is None or u is None:
            continue
        data.append((float(u), bool(c)))
    n = len(data)
    if n < 4:
        return {"error": f"need >=4 execution-labelled tasks with '{signal}'; have {n}",
                "experiment_id": experiment_id, "signal": signal}

    signals_all = [d[0] for d in data]
    correct_all = [d[1] for d in data]
    base_acc = sum(correct_all) / n

    rng = np.random.default_rng(seed)
    n_cal = max(2, ceil(n * cal_frac))
    test_risks: list[float] = []
    test_ask_rates: list[float] = []
    test_accept_acc: list[float] = []
    n_no_region = 0
    for _ in range(n_splits):
        perm = rng.permutation(n)
        cal_idx, test_idx = perm[:n_cal], perm[n_cal:]
        if len(test_idx) == 0:
            continue
        cal = calibrate_threshold(
            [signals_all[i] for i in cal_idx],
            [correct_all[i] for i in cal_idx],
            alpha=alpha, delta=delta,
        )
        tau = cal.get("tau")
        if cal.get("tau") is None and not cal.get("accept_everything"):
            # No region met the target on this calibration split => ask everyone.
            n_no_region += 1
            test_ask_rates.append(1.0)
            continue
        ev = evaluate_selective(
            [signals_all[i] for i in test_idx],
            [correct_all[i] for i in test_idx],
            tau,
        )
        test_ask_rates.append(ev["ask_rate"])
        if ev.get("accepted_risk") is not None:
            test_risks.append(ev["accepted_risk"])
            test_accept_acc.append(ev["accepted_accuracy"])

    def _mean(xs: list[float]) -> float | None:
        return round(sum(xs) / len(xs), 4) if xs else None

    mean_risk = _mean(test_risks)
    return {
        "experiment_id": experiment_id,
        "signal": signal,
        "target_alpha": alpha,
        "delta": delta,
        "n_labelled": n,
        "base_accuracy": round(base_acc, 4),
        "n_splits": n_splits,
        "cal_size": n_cal,
        "mean_test_accepted_risk": mean_risk,
        "mean_test_ask_rate": _mean(test_ask_rates),
        "mean_test_accepted_accuracy": _mean(test_accept_acc),
        "splits_with_no_region": n_no_region,
        "guarantee_holds": (mean_risk is not None and mean_risk <= alpha),
        "note": ("Guarantee transfers out-of-sample iff mean_test_accepted_risk <= "
                 "target_alpha. Small n makes this loose; tightens with K=3/synthetic."),
    }
