# Release readiness

**Status at the 2026-11-19 pre-integration checkpoint:** PR #1 was merged to `main` at
`2b746109f090c6f520e025c1ceb5a7083b36e8d3`. Hosted branch `codex/sunreserve-stage1-demo`
at `3c0b56533a582962a135e4f274148cdaac52eda1` had successful CI run `44093878149`. At that
checkpoint, local branch `codex/sunreserve-post-submission-p0` was at that commit with
uncommitted, unpushed changes and no candidate PR. The successful hosted run did not cover that
dirty tree. The submission receipt was confirmed; the exact submitted snapshot remained
unresolved. No release tag or published hosted app was recorded at capture. The package
metadata version was `0.1.0`; no release had been tagged. The pre-integration default local
full suite reported 287 passed and 4 skipped in 104.98 seconds. The four opt-in Streamlit
lifecycle tests were skipped by default; when run separately with authorized loopback, they
passed 4 tests in 5.29 seconds. Those local results did not replace hosted checks on the
candidate at that checkpoint.

## Architecture

- `app.py` provides the local Streamlit research explorer and fresh prediction flow.
- `src/sunreserve/` contains source retrieval and validation, date-aligned feature building,
  model-bundle inference, reserve decisions, validation/finalization, artifact loading, and
  the CLI (`python -m sunreserve`).
- `data/`, `models/`, and `artifacts/` hold the cached observations, prepared dataset,
  numeric Ridge bundle, development outputs, and frozen historical results.
- NASA POWER values are gridded historical estimates. Historical target observations may
  have been published after issue time; the replay is ex-post and does not prove operational
  forecast availability.

## Local setup and checks

Use Python 3.11–3.13. From the repository root:

```powershell
py -3.11 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -e ".[dev]"
& .\.venv\Scripts\python.exe -m pip check
& .\.venv\Scripts\python.exe -m sunreserve --help
& .\.venv\Scripts\python.exe -m pytest
& .\.venv\Scripts\ruff.exe check .
& .\.venv\Scripts\ruff.exe format --check .
& .\.venv\Scripts\python.exe -m streamlit run app.py
```

For Linux/macOS, create the virtual environment with `python3.11 -m venv .venv`, activate
it, and run the same Python module commands through that environment.

## Reproducing a run

The default workflow uses the validated local NASA POWER cache; `prepare` does not need a
network request. A deliberate source refresh is separate and should be reviewed against its
new provenance before preparation:

```powershell
python -m sunreserve fetch --refresh
```

1. Run `python -m sunreserve prepare` to build the daily calendar from the verified raw
   response and save the quality report.
2. Run `python -m sunreserve validate --run-id <unique-id>`. Inspect the selection manifest
   and validation results. Keep the printed SHA-256 digest with the reviewed manifest.
3. After freezing that selection, finalize it with the reviewed digest and a new run ID. The
   finalized run uses the saved numeric model bundle:

   ```powershell
   python -m sunreserve finalize --experiment-id <unique-id> --manifest-sha256 <digest> --run-id <new-unique-run-id>
   ```
4. Run `python -m sunreserve report --run-id <new-unique-run-id>` to render a report from
   the finalized outputs.

Use new run IDs; preserve the shipped historical outputs and never tune or select on the
2025 holdout. Exact snapshot and source commands are documented in
[`implementation_plan.md`](implementation_plan.md) and [`data_sources.md`](data_sources.md).

## Artifact inventory

| Path | Purpose |
| --- | --- |
| `data/raw/` | Cached NASA POWER response and provenance sidecars, including the separate 2025 holdout source. |
| `data/processed/` | Prepared daily data and source-quality report. |
| `models/ridge_model.json` | Numeric Ridge inference bundle and stored model identity. |
| `artifacts/validation/` | Development validation predictions, metrics, and selection manifest. |
| `artifacts/runs/frozen-test-2023-24/` | Preserved historical test replay and its manifest/report. |
| `artifacts/runs/frozen-test-2025/` | Preserved separate 2025 replay and its manifest/report. |
| `artifacts/experiments/` | Local validation experiments and finalized outputs created with unique IDs. |

Do not overwrite frozen run folders. Check manifests and file digests when copying or
reviewing result files; do not treat a screenshot or exported document as a substitute for
the underlying prediction and metric artifacts.

## Limits, recovery, and contribution notes

- The data is not measured school demand, rooftop generation, or a verified real-time input.
  Demand, PV conversion, and reserve scenarios are hypothetical; results are not operational
  savings, reliability, or dispatch evidence.
- One held-out year does not establish future performance. The 2025 period is reserved for
  evaluation and must not be used for model fitting, tuning, or selection.
- Recommendation: MIT for repo-owned code and documentation, if the user chooses to license
  them. Keep NASA POWER data attribution and source terms separate from the project license.
  No root `LICENSE` has been added; wait for the user's choice before adding one.
- If installation is inconsistent, recreate the virtual environment, install
  `.[dev]`, and run `pip check`. If cache validation fails, do not overwrite it with an
  unverified response; inspect the error and provenance, then refresh only deliberately.
  If an artifact fails checksum/schema validation, preserve the file and compare it with a
  trusted frozen copy before recovery. Use a new run ID rather than replacing output.
- Contributions should use a feature branch, focused tests, `ruff check .`, and
  `ruff format --check .`. Reproduction or model changes should record source/model/config
  identity and keep the holdout boundary intact. See
  [`dependency-security.md`](dependency-security.md) for actionable advisory handling.

## Proposed CI checks and hosted limits

For `main`, recommended branch rules are PR-only changes and no force-push or branch deletion.
Require the `checks` job for Python 3.11, 3.12, and 3.13, plus `minimum-dependencies`,
`streamlit-lifecycle`, and `windows-smoke`. After this candidate is pushed, confirm the exact
hosted status-context strings in GitHub before configuring the required checks; this document
does not guess context display names. Any human-review count remains a user/governance choice.

The `checks` job covers installation, pip compatibility, vulnerability audit, full tests,
offline frozen-artifact checks, Ruff lint, and format on Ubuntu. Additional jobs cover
selected minimum dependency versions, a real Streamlit session lifecycle on Linux, and a
pinned Windows Python snapshot with tests, an HTTP health smoke, offline artifact checks,
and dependency audit.

These are workflow checks, not evidence that branch protection or required-check settings
are enabled. No hosted settings had been applied at the pre-integration checkpoint. The cited
successful CI run belongs to the hosted Stage 1 commit above; it was not a check of the then-local
candidate changes. That pre-integration review batch did not create a release tag or publish an
app or package. The README screenshot remains a prior captured state; this document does not
claim visual/mobile acceptance for the candidate.
