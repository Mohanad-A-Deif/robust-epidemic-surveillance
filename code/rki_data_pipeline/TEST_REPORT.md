# Offline test report

Date: 2026-07-21

The offline integration test completed successfully using genuine RKI fixture rows.
The following checks passed:

- UTF-8 semicolon CSV parsing.
- Normalization of the source header `Altersgrupppe`.
- Aggregation of three age groups to state-day totals.
- Exact numeric comparison against independently calculated totals.
- Chronological train/validation/test labelling.
- Training-only log/z-score transformation.
- Reference adjacency construction.
- Message-level delay, drop, right-censoring, and outlier generation.
- Preservation of arrival arithmetic and separate same-time report records.
- Output schema and integrity validation.

Command:

```bash
python tests/test_pipeline.py
```
