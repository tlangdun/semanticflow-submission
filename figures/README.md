# Figures and result artifacts

The three figures reported in the write-up, and the analysis outputs they are drawn from.
All of it is regenerable — see "Step-by-step: reproducing the evaluation" in the repository
README.

| File | Produced by | Contents |
| --- | --- | --- |
| `compare.json` | `compare-experiments` | Paired comparison, `cheap-fixed-nohitl` vs `cheap-slotguard-hitl`, 85 tasks: per-task outcomes, ambiguity strata, bootstrap CIs, McNemar. |
| `sweep.json`, `sweep.csv` | `sweep-threshold` | Offline effort–accuracy sweep of the clarification trigger on `total_uncertainty`, `cheap-fixed-nohitl` vs `cheap-fixed-hitl`. |
| `roc_*.csv` | `plot_results.py` | ROC series for each uncertainty component against execution correctness. |
| `by_ambiguity.csv` | `plot_results.py` | Accuracy by ambiguity stratum, baseline vs treatment. |
| `fig1_roc.png` | `plot_results.py` | ROC curves of the uncertainty signals. |
| `fig2_effort_accuracy.png` | `plot_results.py` | Effort–accuracy curve with the oracle upper bound. |
| `fig3_accuracy_by_ambiguity.png` | `plot_results.py` | Accuracy by ambiguity, baseline vs treatment. |

The manuscript's LaTeX source resolves `\graphicspath` to this directory.
