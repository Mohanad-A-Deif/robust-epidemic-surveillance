# Concise Revision Log

## Methodology
- Replaced ambiguous `W >= 0`/PSD notation with elementwise non-negativity and a zero diagonal.
- Kept direction in the dynamic transition matrix, but replaced the invalid asymmetric-Laplacian smoothness term with a positive-semidefinite Laplacian built from `(W + W.T)/2`.
- Added the full off-diagonal graph gradient and aligned the manuscript equation with the implementation.
- Replaced hard geographic masking with a soft geographic sparsity prior; non-geographic edges remain admissible.
- Added delay-backprojection initialization, optional tempered delay responsibilities, optional temporal delay consistency, state warm-up, graph staging, and joint refinement.
- Added adaptive row-sum regularization, weighted graph sparsity, backtracking, and explicit convergence diagnostics.
- Retained the previous proposed implementation as an ablation.

## Leakage control and evaluation
- Added chronological train/validation/test splits (219/73/73 days).
- Selected hyperparameters, graph threshold, stopping rules, and configuration using train/validation only.
- Wrote a locked-test protocol before final test evaluation.
- Separated retrospective reconstruction from causal rolling nowcasting; causal day `t` uses only messages arriving by `t`.
- Used paired corruption seeds and fixed nested corruption templates so one-factor sweeps do not redraw unrelated corruption locations.

## Real and semi-real data
- Added real RKI symptom-onset expected-case trajectories for six German states over 365 days.
- Converted counts to daily incidence per 100,000 using documented 31 December 2020 population denominators, then applied `log1p`.
- Kept real trajectories, injected missingness, injected delays, and injected outliers as separate documented objects.
- Added source manifests, checksums, data dictionary, population table, preprocessing outputs, and reproducible message files.

## Experiments
- Added arrival interpolation, timestamp oracle, delay backprojection, Kalman/RTS, delay-aware state space, graph-temporal reconstruction, robust median, robust low-rank completion, fixed geography, and no-delay/no-robustness/no-graph ablations.
- Added 20-seed reference, causal, robustness, and synthetic test evaluations.
- Added model-mismatch tests for nonlinear dynamics, abrupt interventions, nonstationary delays, observation mismatch, and time-varying graphs.
- Added separate sparse, dense, and time-varying directed graph-recovery experiments without giving the true support to the estimator.
- Added one-factor scalability experiments for nodes, time points, maximum delay, and graph density.
- Added runtime, absolute peak RSS memory, normalized-objective convergence, state intervals, edge selection frequency, and graph stability.

## Statistics and presentation
- Added mean/SD, median/IQR, bootstrap 95% confidence intervals, paired t tests, Wilcoxon signed-rank tests, effect sizes, Holm correction, Friedman tests, and average ranks using seeds as blocks.
- Rebuilt the primary figures as 600-dpi grayscale PNGs with white backgrounds, serif typography, inward ticks, dotted grids, and marker/line distinctions.
- Rewrote the manuscript for Healthcare Analytics, added a healthcare workflow and use case, reduced the 6G claim, and expanded limitations.
- Rewrote the response to reviewers point by point.

## Scientific conclusions changed
- The revised method is not claimed to be uniformly best in retrospective reconstruction.
- A simpler Kalman/RTS implementation had lower mean causal RMSE; the proposed causal estimate showed level/scale bias.
- Directed graph recovery was weak despite accurate state reconstruction; RKI graph outputs are therefore described only as learned association/propagation structure and stability, not a true transmission network.
- No claim of real-world deployment readiness remains.
