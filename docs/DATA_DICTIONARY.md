# Data dictionary

## Processed trajectory files

| File | Definition |
|---|---|
| `daily_state_cases_long.csv` | Long-format expected cases, population, and incidence per 100,000. |
| `x_raw.csv` | Alias of the raw daily expected-case matrix. |
| `x_raw_cases.csv` | Daily expected symptom-onset cases, dates × states. |
| `x_incidence_per_100k.csv` | Daily expected cases per 100,000 population. |
| `x_log1p.csv` | `log1p(incidence per 100,000)`; primary latent-state scale. |
| `x_standardized.csv` | Per-state z-score of `x_log1p`, fitted on training dates only. |
| `normalization_parameters.csv` | Training means and standard deviations. |
| `splits.csv` | Chronological train, validation, and locked-test membership. |
| `nodes.csv` | Node IDs, codes, state names, slice labels, population, and population date. |
| `adjacency_reference.csv` | Symmetric geographic reference adjacency; not directed ground truth. |
| `dataset.npz` | Compact model-ready NumPy archive. |
| `dataset_manifest.json` | Provenance, transformation, split, and interpretation metadata. |

## Message-level scenario fields

| Field | Meaning |
|---|---|
| `message_id` | Unique report identifier. |
| `template_id` | Fixed corruption template used for the seed. |
| `scenario`, `family`, `seed` | Scenario metadata. |
| `node_id`, `node_code`, `state`, `slice_id` | Reporting-node metadata. |
| `generation_index`, `generation_date` | Time at which the clean report was generated. |
| `clean_value` | Clean `x_log1p` state. |
| `delay` | Injected transport delay. |
| `arrival_index`, `arrival_date` | Arrival time when received within the horizon. |
| `dropped` | Injected permanent message loss. |
| `right_censored` | Message scheduled after the study horizon. |
| `received` | Message available within the observation horizon. |
| `is_outlier` | Whether an additive outlier was injected. |
| `outlier_addition` | Additive corruption. |
| `observed_value` | Received clean/corrupted value. |
| `true_delay_probability` | Probability under the generation delay distribution. |
| `inference_delay_probability` | Probability provided to the inference algorithm. |
| `delay_uniform_template` | Seed-fixed uniform variable used for nested delay draws. |
| `drop_uniform_template` | Seed-fixed uniform variable used for nested drop decisions. |
| `outlier_uniform_template` | Seed-fixed uniform variable used for nested outlier selection. |

## Evaluation interpretation

RKI graph outputs are association/propagation graphs evaluated by geographic agreement, stability, edge uncertainty, density, and row sums. Ground-truth directed graph recovery is evaluated only in separate synthetic experiments.
