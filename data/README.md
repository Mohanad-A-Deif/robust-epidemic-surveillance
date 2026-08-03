# Data package

## Real epidemic trajectories

The raw source is the RKI StopptCOVID symptom-onset expected-case dataset, pinned to DOI `10.5281/zenodo.10888033`. The bundled source file is verified against MD5 `e59953809e7a8050bb40045c6172ee30`.

## Latent epidemic state

For each state and date:

1. Sum the three age groups.
2. Divide by the documented state population denominator.
3. Multiply by 100,000.
4. Apply `log1p`.

The primary state is therefore:

```text
log1p(daily expected symptom-onset incidence per 100,000 population)
```

`x_standardized.csv` is an auxiliary training-only z-scored representation. Its means and standard deviations are fitted using the training interval only.

## Real versus injected components

- Real: RKI epidemic trajectories.
- Injected: message missingness, transport delays, and outlier additions.
- Geographic adjacency: reference/soft prior, not transmission ground truth.

## Files

- `raw/`: pinned RKI source CSV.
- `processed/`: raw cases, incidence per 100k, log1p state, optional standardized state, splits, nodes, and reference adjacency.
- `metadata/`: source, population, node, adjacency, and scenario metadata.
- `scenarios/`: 280 compressed message-level scenario files, 20 corruption templates, per-file summaries, and combined manifests.

Every generated message remains a separate row, including multiple reports arriving at the same node and time.
