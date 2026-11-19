# Erratum: finalized-run exclusion counts

**Date:** 2026-11-18

The frozen `artifacts/runs/frozen-test-2023-24/run_manifest.json` records
`row_counts.test_excluded: 6`. That historical value is mislabeled. The six excluded
targets are 2016-01-02 through 2016-01-07, all in the training period; they are the
initial feature-window warm-up rows. The frozen manifest, its artifacts, and their
hashes are preserved unchanged.

Newly finalized runs classify excluded targets using their configured target-date
split. Their manifests retain eligible train, validation, and test row counts, record
exclusion counts under `excluded_rows_by_split`, and record the corresponding
issue-date, target-date, and missing/nonfinite field details under
`excluded_dates_by_split`. For the current 2016–2024 data, the six warm-up rows count
as training exclusions and test exclusions are zero.
