# Test and validation report

Date: 2026-08-03

## Environment

- Python: 3.13.5
- Platform used for package validation: Linux container

## Completed checks

1. **RKI offline integration test:** passed.
   - Source parsing and header normalization.
   - Age-group aggregation.
   - Incidence per 100,000 calculation.
   - `log1p` transformation.
   - Chronological train/validation/test split.
   - Training-only standardization.
   - Fixed corruption-template generation.
   - Message arrival/drop/censoring integrity.

2. **Model core tests:** 6 passed.
   - State and graph metrics.
   - End-to-end proposed-model fit.
   - Positive-semidefinite symmetrized smoothness Laplacian.
   - Geographic adjacency used as a soft prior.
   - Causal filtering excludes future arrivals.

3. **Minimal end-to-end smoke test:** 1 passed.

4. **Full repository data validation:** passed.
   - Processed matrix: 365 dates × 6 states.
   - Scenario files checked: 280.
   - Seeds: 20.
   - Scenarios: 14.
   - RKI raw-file MD5: `e59953809e7a8050bb40045c6172ee30`.

5. **Publication figures:** 20 PNG files found; 20 contain approximately 600-DPI metadata.

## Commands

```bash
python scripts/run_tests.py
python scripts/run_quick_smoke.py
python scripts/validate_repository.py
```

The frozen manuscript results were not regenerated during this final packaging pass; they were copied from the completed locked-test runs and verified for presence, structure, and provenance.
