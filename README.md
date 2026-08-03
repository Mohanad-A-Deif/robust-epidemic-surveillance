# Robust Epidemic Surveillance under Missing and Delayed Data

[![Reproducibility](https://img.shields.io/badge/reproducibility-complete-brightgreen)](#reproducibility-status)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](requirements.txt)
[![Code license](https://img.shields.io/badge/code-MIT-blue)](LICENSES/MIT.txt)
[![Data license](https://img.shields.io/badge/data-CC%20BY%204.0-lightgrey)](LICENSES/RKI_DATA_NOTICE.txt)

This repository is the supplementary and reproducibility package for the manuscript:

> **A Robust Analytics Approach to Graph-Based Epidemic Surveillance Under Missing and Delayed Data**

It combines the real RKI epidemic trajectories, controlled semi-real communication corruptions, the revised estimator, external/internal baselines, locked-test protocol, seed-level outputs, statistical analyses, publication tables, and publication figures.

## What is real and what is injected?

- **Real component:** expected daily COVID-19 cases by symptom-onset date from the Robert Koch Institute (RKI).
- **Population normalization:** state-level incidence per 100,000 population using documented population denominators dated 31 December 2020.
- **Primary latent state:** `log1p(incidence per 100,000 population)`.
- **Injected components:** message missingness, transport delay, and additive outlier contamination.
- **Geographic adjacency:** a soft structural reference only; it is **not** a ground-truth transmission network.

The source RKI file is pinned to DOI `10.5281/zenodo.10888033` and verified against MD5 `e59953809e7a8050bb40045c6172ee30`.

## Repository structure

```text
.
├── code/
│   ├── rki_data_pipeline/       # Source preparation and nested corruption generation
│   └── model_pipeline/          # Proposed model, baselines, statistics, and figures
├── configs/                     # Data, results, and locked-test configurations
├── data/
│   ├── raw/                     # Pinned RKI source CSV
│   ├── processed/               # Incidence, log1p state, splits, and metadata
│   ├── scenarios/               # 14 scenarios × 20 fixed seeds = 280 message files
│   └── metadata/                # Nodes, population, scenarios, source manifest
├── results/
│   ├── tables/                  # Main publication-ready CSV tables
│   ├── figures/                 # 600-DPI grayscale PNG figures
│   └── raw/                     # Seed-level metrics, tests, arrays, and manifests
├── supplementary_material/      # Revised LaTeX methodology and results sections
├── docs/                        # Data dictionary, protocol, tests, and package notes
└── scripts/                     # One-command validation and reproduction helpers
```

## Quick start

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
python scripts/run_tests.py
python scripts/validate_repository.py
```

## Reproduce the dataset

The complete data products are already included. To rebuild them from the bundled, checksum-verified RKI source:

```bash
python scripts/reproduce_data.py
```

The full generation creates 280 compressed message-level scenario files. For a faster smoke run:

```bash
python scripts/reproduce_data.py --quick
```

## Run a software smoke test

```bash
python scripts/run_quick_smoke.py
```

This smoke run verifies the end-to-end software path using a small synthetic dataset. Its outputs are not manuscript results.

## Main empirical findings

- Retrospective reconstruction: the revised proposed method obtained mean RMSE `0.3197 ± 0.0515`; the no-delay ablation obtained `0.3007 ± 0.0506`.
- Causal rolling nowcasting: Kalman/RTS obtained the lowest mean RMSE (`0.8683`); the proposed method obtained `1.0857`, retained high temporal correlation, and showed scale bias.
- Robustness: the robust loss protected against contamination relative to the no-robustness ablation, while the benefit of explicit delay modeling depended on the corruption regime.
- Directed graph recovery: state reconstruction was accurate in the synthetic study, but directed-edge recovery was weak (support F1 roughly `0.28–0.31`, AUROC near chance). The learned RKI graph is therefore interpreted as an association/propagation graph, not a biological transmission network.

See [`docs/RESULTS_SUMMARY.md`](docs/RESULTS_SUMMARY.md) for the complete cautious interpretation.

## Train/validation/test protocol

Hyperparameters, graph threshold, stopping rules, and model settings were selected using training and validation data only. The test interval was locked before final evaluation. Retrospective reconstruction and causal rolling nowcasting are reported as separate tasks. The protocol is recorded in [`configs/locked_test_protocol.json`](configs/locked_test_protocol.json).

## Reproducibility status

- 365 daily time points and 6 states.
- 20 fixed seeds (`1101`–`1120`).
- 14 controlled scenarios.
- 280 compressed message-level files.
- Corruption templates fixed per seed and reused across scenario levels.
- Complete seed-level results, statistical tests, publication tables, and figures included.
- SHA-256 manifest supplied for all package files.

## Citation and licenses

Use [`CITATION.cff`](CITATION.cff) and [`CITATION.md`](CITATION.md). Code is released under the MIT License. The RKI source and derived data retain CC BY 4.0 attribution requirements. See [`LICENSE`](LICENSE) and [`LICENSES/`](LICENSES/).

## Important scientific limitations

This repository does not claim real 6G telemetry, a known RKI transmission graph, deployment readiness, or universal superiority of the proposed method. External comparison methods are identified as simplified internal implementations unless an exact published implementation is explicitly stated.
