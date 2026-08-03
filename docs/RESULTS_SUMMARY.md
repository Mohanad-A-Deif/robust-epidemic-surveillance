# Results summary and scientific interpretation

## Retrospective reconstruction

Across 20 locked-test seeds, mean RMSE was:

- No delay: `0.3007 ± 0.0506`
- Original proposed version: `0.3097 ± 0.0515`
- Revised proposed version: `0.3197 ± 0.0515`
- Fixed geography: `0.3230 ± 0.0240`

The revised method was competitive but was not the lowest-error method. Multiplicity-corrected comparisons did not justify a universal superiority claim.

## Causal rolling nowcasting

- Kalman/RTS: RMSE `0.8683`
- Delay-aware augmented state-space model: RMSE `1.0343`
- Revised proposed method: RMSE `1.0857`

The proposed method retained high mean temporal correlation (`r≈0.869`) but showed substantial level/scale bias. Reconstruction performance must not be described as causal nowcasting performance.

## Robustness

Robust fitting reduced sensitivity to injected outliers relative to the no-robustness ablation. Explicit delay modeling was most useful under severe missingness and delay-prior mismatch, but the no-delay ablation remained stronger in several moderate/high-contamination settings. No baseline was removed when it outperformed the proposed method.

## Graph evaluation

On synthetic graphs, support F1 was approximately `0.28–0.31`, with AUROC near chance. State RMSE remained low, showing that accurate state reconstruction does not imply identifiable directed edges. The RKI learned graph is therefore reported as an exploratory association/propagation graph only.

## Scope

The results demonstrate methodological feasibility and reveal failure modes. They do not establish a true regional transmission network, 6G deployment readiness, or universal superiority.
