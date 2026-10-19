# R2 analyses (HEALTH-D-26-00180, second revision)

Additional analyses added in the second revision. All scripts use the frozen configuration in
`results/raw/protocol/` and the saved outputs in `results/raw/`; the locked test interval is not used for any selection.

Run from this folder (`cd r2_analysis`) in the environment of `requirements.txt`:

| Order | Script | Purpose | Output |
|---|---|---|---|
| 1 | `oat.py` | One-at-a-time sensitivity of lambda_G, lambda_W, lambda_r, lambda_geo on validation seeds 1101-1105 (Table 6). Requires new validation-only fits. | `oat_seed_metrics.csv`, `oat_graphs.npz` |
| 2 | `cases.py` | Recomputes locked-test reconstructions of comparator methods for 20 seeds (inputs to Table 16). | `baseline_xhat.npz` |
| 3 | `final_dec.py` | Decision-oriented 7-day incidence metrics with per-seed SDs (Table 16) and Fig. 17; uses `decision.py`. | `decision_table.csv`, `fig_decision_case.png` |
| 4 | `colorfigs.py` | Figures 1-15 from saved outputs. | `figures/` |
| 5 | `wf_bw.py` | Workflow diagram (Fig. 16). | `figures/fig_operational_workflow.png` |
| - | `regen.py` | Earlier relabelling script (superseded by `colorfigs.py`). | |

`outputs/` contains the CSV files used in the manuscript; `figures/` contains the published figures.

## Method-name mapping (raw output identifiers -> manuscript labels)
- `Proposed revised` -> Proposed method
- `Original proposed ablation` -> Hard-mask variant
- `Known-delay proposed oracle` -> Known-delay oracle

Legacy identifiers are kept in the raw result files for traceability.
