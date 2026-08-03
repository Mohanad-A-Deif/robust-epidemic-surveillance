# Results Output Catalog

## EDA tables

- Dataset overview and temporal coverage.
- Descriptive statistics on raw, log-transformed, and standardized scales.
- Missing-value and zero-value fractions.
- Train/validation/test distribution statistics.
- Linear trends, ADF and KPSS stationarity tests.
- Lag-1 and lag-7 autocorrelation.
- Peak dates and values.
- Pearson and Spearman interregional correlation.
- Peak lagged cross-correlation matrices.
- Reference-network node and global descriptors.
- Scenario integrity: realized delay, drop, outlier, censoring, and arrival collisions.

## EDA figures

1. Raw epidemic trajectories.
2. Log-transformed trajectories.
3. Leakage-safe standardized trajectories.
4. Per-node distribution boxplots.
5. Pearson correlation heatmap.
6. Peak cross-correlation lag heatmap.
7. Autocorrelation panels.
8. Weekly reporting pattern.
9. Reference adjacency heatmap.
10. Reference network diagram.
11. Scenario delay/drop/outlier characteristics.

## Seed-level result metrics

### State recovery

- MSE, RMSE, MAE, median AE, maximum AE.
- NRMSE by range and standard deviation.
- Normalized MAE.
- MAPE and sMAPE.
- R², Pearson correlation, Spearman correlation, and bias.
- Overall and held-out test-period values.
- Per-node versions of all principal state metrics.

### Graph recovery when a ground truth graph exists

- Frobenius and normalized Frobenius errors.
- Support precision, recall, and F1.
- AUROC and AUPRC.
- Structural Hamming distance.
- Edge-weight correlation.
- Graph-density error.

### Real-data graph analysis

- Agreement with geographic reference support.
- Stability of weights across seeds.
- Edge-support Jaccard similarity.
- Edge-weight coefficient of variation.
- Learned density and maximum row sum.

### Delay recovery

- Delay MAE and RMSE.
- Delay bias.
- Exact-delay and within-one-day accuracy.
- Posterior probability assigned to the true delay.
- Posterior entropy.

### Decision relevance and computation

- Outbreak precision, recall, and F1.
- Runtime, peak-memory increment, iterations, and convergence status.
- Objective monotonicity.

## Statistical outputs

- Mean, SD, median, IQR, and bootstrap 95% CI.
- Paired t-tests.
- Wilcoxon signed-rank tests.
- Holm-adjusted p-values.
- Cohen's dz and rank-biserial effects.
- Friedman omnibus test.
- Average method ranks.

## Result figures

1. Main benchmark: RMSE, MAE, and R².
2. Runtime–accuracy trade-off.
3. Missingness robustness.
4. Delay-severity robustness.
5. Outlier robustness.
6. Delay-prior misspecification.
7. Objective convergence.
8. Average ranks.
9. Pairwise significance matrix.
10. Ground-truth versus recovered trajectories.
11. Ground-truth/reference versus learned graph heatmaps.

Additional sensitivity, model-mismatch, uncertainty, and scalability figures are generated when their corresponding seed-level fields are supplied.
