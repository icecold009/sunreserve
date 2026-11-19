# SunReserve implementation plan

## Delivery status at pre-integration checkpoint (2026-11-19)

At the 2026-11-19 pre-integration capture, the organizer receipt confirmed that the entry had been received; the exact submitted snapshot remained unresolved. PR #1 had been merged to `main` at `2b746109f090c6f520e025c1ceb5a7083b36e8d3`. Hosted branch `codex/sunreserve-stage1-demo` at `3c0b56533a582962a135e4f274148cdaac52eda1` had successful CI run `44093878149`. At that capture, local branch `codex/sunreserve-post-submission-p0` was at the same commit with uncommitted and unpushed work and no candidate PR. The hosted CI result applied to its hosted commit, not to the then-dirty working tree. No hosted app publication or release tag was recorded at capture. See [post-submission evidence](post_submission_evidence.md) for scope and snapshot limits.

Earlier local verification is retained below as dated evidence. It was not acceptance evidence for the candidate working tree at the 2026-11-19 pre-integration checkpoint.

## 1. Outcome and boundaries

Build an AI + Climate research prototype for a hypothetical school facilities manager in Bengaluru. Ask: **Does a small next-day solar model improve a simulated backup-energy decision compared with assuming tomorrow equals today?**

The minimum complete solution is a reproducible historical experiment, two simple baselines, one trained model, a transparent reserve simulation, an offline demonstration, and an honest report. A second model is bounded comparison work, not a prerequisite to a complete project.

- Predict daily solar radiation, not measured rooftop generation.
- Describe the school, demand, panel conversion, and backup capacity as hypothetical assumptions. No school interviews or measured school data are established.
- Historical weather observations may have been published after the forecast issue time. This is an ex-post research replay, not an operational day-ahead forecast or a deployable controller.
- Do not claim real savings, emissions reductions, battery optimization, or demonstrated benefit to a school.
- Build new hackathon work. Do not extend or copy implementation from existing personal projects.
- No accounts, database, authentication, paid API, live control, hosted deployment, or multi-city scope is needed.

## 2. Current baseline

Observed baseline at project start: `feature/veritas-sunreserve`, an editable Python package, isolated `.venv`, dependency declarations, Ruff/pytest configuration, and data/documentation folders. The scaffold was initially committed and pushed as project work; the current checkout has Git history. This describes only the project-start baseline; later implementation and delivery state are recorded in the dated status and checkpoint sections below.

Verified on 2026-11-14: Stage A data ingestion, daily preparation, supervised rows, chronological splits, and two simple baselines are implemented. NASA POWER supplied 3,288 complete daily rows (2016–2024); the raw response SHA-256 is `925dde6e86766df7f0f65c91d77947e7306068892fd02ddec0c5b891d28c908`. The supervised builder reports 3,287 candidate target rows, six initial-window exclusions, and 3,281 eligible rows (2,185 train, 365 validation, 731 test). Forty focused tests and global Ruff checks passed. No Ridge model, validation selection, reserve evaluation, or final-test performance result is included in this Stage A evidence.

Stage B verified on 2026-11-14: cached `validate` reads prepared daily rows only through 2022-12-31, fits Ridge preprocessing and coefficients on the fixed 2016–2021 training partition, and selects among Ridge and both baselines on 365 validation targets by MAE. Validation selected Ridge at MAE 0.6249 kWh/m²/day; persistence MAE was 0.6767 and seasonal-mean MAE was 0.8365. Two offline runs produced byte-identical validation prediction and metric files. The training-only low-solar threshold was 4.5967 kWh/m²/day. These are validation results only; 2023–2024 outcomes were not loaded, predicted, scored, or reported.

Preserve this starting work. Keep implementation on a feature branch. Use the GitHub connector first for hosted branch, PR, review, check, and merge operations. Do not commit or merge into `main` without explicit approval. Luna performs future coding under the user's standing preference. Jev reviews are advisory and do not substitute for deterministic verification.

## 3. Execution order and cut points

Checked items record completed work with evidence in their entries; unchecked delivery and submission tasks remain pending. Stage time ranges are original estimates, not measured usage.

| Stage | Depends on | Budget | Exit condition |
| --- | --- | --- | --- |
| A. Data and baselines | Existing scaffold | 2–3 h | Provenance, clean daily series, chronological split, and both baselines work |
| B. Features and model | A | 3–4 h | Leakage checks pass; validation comparison selects a frozen approach |
| C. Reserve experiment | B | 3–4 h | Identical scenarios evaluate all approaches with correct units |
| D. Demonstration | C | 3–4 h | Cached historical results render and failure examples are inspectable |
| E. Final evaluation and delivery | A–D | 5–7 h | One reproducible run, final test results, write-up, credits, and submission checks |

Do not use the final test period to steer stages B–D. During development use validation predictions and small synthetic fixtures. Reserve the final test run for the frozen experiment in stage E.

Cut order if time tightens: extra tree model, additional feature variants, extra charts, optional video polish. Keep provenance, baselines, held-out evaluation, reserve arithmetic, a usable demo, and the write-up. If the model does not beat the baselines, report the negative result and analyze why; do not keep tuning on test outcomes.

## 4. Configuration and executable entry point

Primary files: new `src/sunreserve/config.py`, `src/sunreserve/__main__.py`, and configuration documentation.

- [x] Define one serializable experiment configuration: Bengaluru reference coordinate (12.9716, 77.5946), date interval, source variables, time standard, split boundaries, random seed, allowed models, primary metric, and simulation assumptions. Evidence: JSON round-trip and required defaults are covered in `tests/test_config.py`.
- [x] Validate coordinate ranges, date ordering, positive areas/demand, efficiency/derating in (0, 1], nonnegative reserve capacity, and a disjoint chronological split. Evidence: invalid configuration cases fail in `tests/test_config.py`.
- [x] Resolve repository data/artifact paths explicitly; create directories only when an invoked command needs them. Evidence: `ProjectPaths` anchors paths at the checkout; fetch/prepare create only their output directories.
- [x] Provide CLI subcommands `fetch`, `prepare`, `validate`, `finalize`, and `report`. Each returns a nonzero status with an actionable error on invalid or missing input. Evidence: CLI failure/contract tests and completed cached command sequence.
- [x] Make network access explicit in `fetch`; all later commands work from cached data. Prevent an accidental download from importing a module or starting the interface. Evidence: offline CLI and app checks; app launch reads only cached artifacts.
- [x] Record the resolved configuration in every experiment manifest. Do not silently switch sources or locations. Evidence: frozen run manifests include resolved settings and configuration hash.

Historical acceptance criteria: invalid settings fail before work starts, and local fixture commands work without network access. Implementation and completed command evidence are recorded in sections 5–10 below.

## 5. Stage A: source ingestion and quality checks

Primary files: `src/sunreserve/data.py`, `docs/data_sources.md`, new `tests/test_data.py`.

- [x] Verify current NASA POWER parameter availability, terms, attribution, and time semantics before selecting the full dataset. Start with daily `ALLSKY_SFC_SW_DWN`, `T2M`, and `RH2M` for 2016–2024. Evidence: current API docs and terms are linked in `docs/data_sources.md`; the retrieved response reports the expected RE units and UTC. The AWS Registry's no-restrictions statement and citation request are recorded. The referencing guide's notification request for data transmission is documented as a request, not a legal requirement; no notification was sent.
- [x] Use the standard-library HTTP client, or explicitly declare a direct dependency if another client is chosen. Use timeouts and bounded retries for transient failures; do not repeatedly retry invalid requests. Evidence: `urllib` fetch; tests verify retry bounds and no retry for HTTP 422.
- [x] Save the untouched response under ignored `data/raw/` and a provenance sidecar containing request URL/parameters, retrieval UTC time, coordinate, dates, units, source names, time standard, checksum, and missing-value marker. Evidence: local raw response and sidecar were inspected; checksum linkage is recorded in the quality report.
- [x] Reuse a matching cached response by default; require an explicit refresh to overwrite it. Never replace a good cache with an error response. Evidence: live cache reuse and corrupt-cache/invalid-refresh fixtures passed.
- [x] Parse and validate the response schema, reported units, requested location, and variable lengths before transformation. Do not assume an HTTP-success response contains valid observations. Evidence: full POWER response validated; schema, coordinate, date, time, and unit failure fixtures pass.
- [x] Produce one sorted row per calendar date with explicit units: radiation in kWh/m²/day, temperature in °C, and relative humidity in percent. Evidence: prepared CSV contains 3,288 unique dates from 2016-01-01 through 2024-12-31; units appear in the quality report.
- [x] Convert source fill values such as -999 to missing values. Reject duplicate or malformed dates; flag impossible negative radiation and out-of-range humidity. Evidence: each condition has focused fixtures in `tests/test_data.py`.
- [x] Reindex to a complete daily calendar. Report missing coverage by variable/year and the exact exclusions. A missing day must never make a next-row shift mean "tomorrow." Evidence: full calendar preparation and a dropped-day next-calendar-day fixture passed; the real source has zero missing values.
- [x] Use a documented exclusion policy for unavailable target/predictor rows. Avoid interpolating across gaps using later observations; fit any imputation only on training data. Evidence: missingness exclusions are documented and tested; Ridge imputer/scaler statistics match training rows even when validation features contain extreme values or missing entries.
- [x] Save prepared data under ignored `data/processed/`, with a quality report and raw-data checksum linkage. Evidence: `daily.csv` and `quality_report.json` were generated and their SHA-256 linkage inspected.

Acceptance: fixture coverage includes fill values, missing days, duplicate dates, wrong units, malformed payloads, and failed downloads. Full-history coverage was verified for 2016–2024 as recorded above. If a future refresh fails, retain the validated cache and report that retrieval failure rather than replacing it with an error response.

## 6. Stage A/B: supervised rows and chronological splits

Primary files: `src/sunreserve/features.py`, new `tests/test_features.py` and `tests/test_splits.py`.

- [x] Define issue date t and target date t+1 explicitly. A row's target is radiation on that next calendar day; predictor observations must be dated no later than t. Evidence: exact issue/target alignment and year-boundary fixtures pass.
- [x] Start with today's radiation, a few prior radiation lags, a trailing seven-day radiation summary, today's weather, and known calendar seasonality. Keep the first feature set small and fixed. Evidence: fixed feature schema and hand-calculated lag/window values are asserted in `tests/test_features.py`.
- [x] Document lag and rolling-window alignment. Never use centered windows, tomorrow's observed weather, backfilled future values, or the target as a feature. Evidence: `docs/data_sources.md` describes right-aligned windows; future-observation mutation leaves earlier features unchanged in tests.
- [x] Attach `issue_date`, `target_date`, and feature-availability metadata to each row. Describe availability as historical observation dates, not verified release timestamps. Evidence: hand-built row checks dates, seven-day window, availability limit, and release-time caveat.
- [x] Split by **target date**: 2016–2021 training; 2022 validation; 2023–2024 final test. Earlier observations may supply lag context without moving later labels into training. Evidence: boundary and non-overlap tests pass for all three partitions.
- [x] Keep target values for the final test out of selection/validation reports until `finalize` is invoked. Evidence: `validate` reads only the development prefix through 2022-12-31, applies a validation target-date cutoff, and writes train/validation counts and validation-only predictions/metrics; the offline integration fixture places sentinel outcomes after the cutoff and confirms they do not appear in outputs.
- [x] Report usable row counts after lags/missingness exclusions for every partition. Compare all approaches on the same eligible target dates. Evidence: builder reports counts and exclusions; persistence and seasonal predictions both cover the same 365 validation targets.

Acceptance: a hand-built daily sequence yields exactly the expected next-day labels and lag values. Tests cover missing dates, year boundaries, no target overlap, train-only transformations, and the invariant that changing later observations cannot change earlier feature rows.

## 7. Stage B: baselines, models, and selection

Primary files: `src/sunreserve/train.py`, `src/sunreserve/evaluate.py`, new `tests/test_models.py` and `tests/test_metrics.py`.

- [x] Implement persistence: predict tomorrow's radiation as today's radiation. Evidence: next-day alignment and zero clipping are covered by focused tests.
- [x] Implement a seasonal baseline: training-only mean radiation grouped by target calendar month. Define a training-global fallback for an absent month. Evidence: fitting rejects non-training labels; monthly means and missing-month global fallback are tested.
- [x] Fit Ridge regression through a preprocessing pipeline. Estimate scaling/imputation from training observations only. Evidence: fixed `alpha=1.0`, `seed=42`, median imputer and scaler; tests inspect fit statistics and prove validation values do not alter them.
- [ ] If time permits, fit one small tree-based regressor with fixed seed and bounded settings. Avoid broad parameter searches.
- [x] Clip physically invalid negative predictions consistently before evaluation and record the rule. Apply the same rule to validation and final test. Evidence: persistence, seasonal mean, Ridge, and the scoring function all use the same zero lower bound; the clip rule is saved in the manifest.
- [x] Calculate validation MAE as the primary selection metric. Specify a fixed tie-breaking order favoring the simpler approach before seeing results. Evidence: MAE selection and the fixed persistence → seasonal mean → Ridge tie order are tested with a tied synthetic case.
- [x] Select between candidate approaches using validation only; preserve baseline results regardless of which method wins. Keep training years fixed for the final experiment rather than silently refitting with validation labels. Evidence: train and validation partitions are enforced separately; test partitions are refused by selection; configuration rejects missing or unsupported model names and validation predictions include the fixed three-model comparison; the model is not refit with validation.
- [x] Calculate secondary metrics: RMSE, seasonal MAE, bias, sample counts, and errors on low-solar days. Freeze the low-solar threshold using training data, such as its 25th percentile. Evidence: hand-computed metric tests pass; actual validation threshold is the training 25th percentile (4.5967 kWh/m²/day).
- [x] Record configurations, validation scores, selected approach, exclusions, environment versions, seeds, and input hashes in a selection manifest. Evidence: versioned `artifacts/validation/selection_manifest.json` records train/validation counts, config/input/development-code hashes, package versions, fixed settings, exclusions, selected method, and validation metrics.
- [x] Test constant/degenerate targets, empty partitions, nonfinite predictions, baseline calculations, and metric values on small hand-checkable examples. Evidence: focused config/model/metric/validation suite passed; configuration tests reject NaN/infinite numeric settings before range/sign checks.

Acceptance: validation reports are reproducible and identify a selected method without consulting test performance. Evidence: actual cached validation selected Ridge consistently in two identical offline runs; the final-test partition was excluded by both loader and selector guards. A weak model is a reportable result, not a reason to invent improvement.

## 8. Stage C: transparent reserve simulation

Primary files: `src/sunreserve/decision.py`, `docs/assumptions.md`, new `tests/test_decision.py`.

This is a daily energy-balance scenario, not a physical battery simulator. It assumes prepared backup energy is fully available that day, with no inter-day state, power limits, tariffs, charging losses, or outages. Those assumptions must appear in the interface and report.

- [x] Define an illustrative generation proxy G = radiation × area × efficiency × derating. Proposed starting values: 20 m², 0.20, and 0.80. This simplified horizontal-radiation conversion is not a validated rooftop PV yield model. Evidence: formula and explicit limits are documented in `docs/assumptions.md`; hand-calculated generation tests pass.
- [x] Set an illustrative daily demand D = 15 kWh and reserve cap C = 15 kWh. These are experiment settings, not measurements. Freeze them before final testing. Evidence: default primary validation scenario reports the configured hypothetical 15 kWh demand/cap and zero buffer; a nonzero configured primary buffer is also persisted in both scenario rows and the manifest; assumptions label these unmeasured.
- [x] For each approach, predict generation Ghat and compute prepared reserve R = min(C, max(0, D − Ghat + b)). The default primary buffer b is zero; an explicit validated config value may override it. Evidence: the default cached validation artifact records 0.0 kWh in `decision_predictions.csv` and `selection_manifest.json`; `tests/test_validation.py::test_validate_writes_only_training_and_validation_outputs` verifies a 2.5 kWh override in both outputs. The fixed sensitivity grid remains demand/capacity 10/15/20 kWh and buffers 0/10/20% of demand.
- [x] From actual historical radiation, compute simulated shortfall = max(0, D − G − R). Evidence: zero-sun, ample-sun, exact-balance, and cap-exhaustion fixtures validate the balance; output stores `simulated_shortfall_kwh`.
- [x] Compute excess prepared reserve = max(0, R − max(0, D − G)). Keep this separate from surplus solar generation. Evidence: hand-calculated excess fixture and separate `excess_prepared_reserve_kwh` output.
- [x] Report total/mean shortfall, number/share of shortfall days, total/mean excess reserve, and their paired differences from both baselines. Evidence: 112 scenario/approach summaries record totals, means, counts/shares, and paired daily differences versus persistence and seasonal mean on common dates.
- [x] Include an always-prepare-backup reference R = min(D, C) to expose the reliability/over-preparation trade-off. Evidence: validation output includes this reference; test verifies its simulated shortfall equals the capacity-limited floor.
- [x] Check a small predeclared sensitivity grid, for example demand 10/15/20 kWh, cap 10/15/20 kWh, and buffers 0/10/20% of demand. Apply identical settings to every approach. Do not pick a favorable scenario after seeing test results. Evidence: fixed grid has 27 settings plus primary; output has 28 scenarios, each with 4 approaches and all 365 validation dates.
- [x] Show when even full reserve cannot cover demand; distinguish unavoidable capacity limits from forecast errors. Evidence: `capacity_limited_shortfall_kwh` is separately reported; incremental shortfall above that floor is explicitly labeled descriptive, not a pure causal forecast-error estimate.
- [x] Do not collapse shortfall and excess into a claimed monetary benefit without a justified cost model. Present the trade-off directly. Evidence: assumptions document reports energy quantities only and makes no claimed savings, cost, or emissions benefit.

Acceptance: hand-calculated tests cover zero sun, ample sun, exact balance, negative forecast clipping, exhausted reserve cap, invalid inputs, and no scenario advantages from mismatched dates/settings. Changing the model must not change the observed radiation or assumed demand. Verified: `tests/test_decision.py` plus validation/model/metric tests pass (33 tests); actual cached validation emitted 40,880 per-date rows for 28 scenarios × 4 approaches × 365 dates and 112 summaries, all dated 2022 only. In the primary scenario, persistence/Ridge/seasonal mean simulated shortfall totals were 171.01/230.86/376.99 kWh and excess prepared reserve totals were 177.16/76.40/17.21 kWh; capacity-limited shortfall was zero for each. These are descriptive validation outputs, not savings or test-period results.

## 9. Stage D: cached Streamlit demonstration

Primary files: `app.py`, optional new `src/sunreserve/presentation.py`, and focused rendering checks.

- [x] Load saved reports/predictions without training or fetching data. During development load validation artifacts; clearly label which partition is displayed. Evidence: `app.py` loads only checked cached validation artifacts through the offline presentation loader; the view prominently labels 2022 validation. Missing-artifact view explains local recovery commands without running them.
- [x] Show a historical-date selector and radiation predictions beside NASA POWER target values, with units and model names. Evidence: recording-UI test verifies date/model selectors and chart series labels in radiation units; the UI identifies the target as NASA POWER model/satellite-derived radiation, not station data.
- [x] Show reserve, shortfall, and excess reserve for the same date/scenario. Label all converted energy as simulated. Evidence: rendering test selects a saved date/scenario and checks same-row simulated energy metrics and assumed settings.
- [x] Add one readable actual-versus-predicted chart and a compact baseline comparison table. Evidence: offline rendering test checks chart columns and four-row comparison table for the selected scenario.
- [x] Add failure examples chosen by a documented rule, such as the largest overprediction and underprediction. Do not present only favorable days. Evidence: `failure_examples` deterministically picks the largest positive and largest negative 2022 radiation errors; rendering test checks both.
- [x] Display provenance, date range, assumptions, missingness, historical-data availability limitation, and the fact that no real school deployment is demonstrated. Evidence: rendered expander includes NASA POWER provenance/units, coverage and missingness, historical-availability caveat, hypothetical assumptions, and no-deployment statement.
- [x] If scenario controls are included, label them exploratory and do not overwrite frozen experiment results. Evidence: controls only select saved model/date/scenario rows; no scenario settings are edited or artifacts written.
- [x] Handle absent, corrupt, incompatible, or stale artifacts with an explanatory screen and instructions; do not display empty or fabricated results. Evidence: loader tests exercise all four failure categories; missing-artifact rendering test checks guidance; integrity checks reject incompatible/stale matrices and hashes.
- [x] Manually verify launch, date selection, units, charts, missing-artifact behavior, small-screen readability, and operation without network access. Evidence: live browser acceptance at 433×938 CSS px (responsive breakpoint ≤640 px) visibly checked date/model/scenario controls, complete metric values and units, charts, assumptions/source, failure examples, and missing-artifact instructions. Offline browser interaction uses cached artifacts; app does not fetch/train at launch. Streamlit metric values were shortened to fit its truncating component; the focused presentation tests assert the unit and 13-character display budget (7 passed after this change). The local AppTest harness failed even for a title-only app, so offline loader and recording-UI tests provide automated UI coverage.

Acceptance: a viewer can understand the user, input, forecast, decision, and limitation within two minutes. A fresh launch with valid cached artifacts requires no API key, network request, or retraining.

## 10. Stage E: reproducible final run and artifacts

Primary files: CLI entry point, report serialization, `README.md`, and new integration tests.

- [x] Make `validate` generate development predictions/reports using training and validation only. Evidence: synthetic end-to-end integration creates 2022-only predictions and confirms the development manifest/artifact hashes.
- [x] Freeze the selected model, feature schema, preprocessing, thresholds, scenario grid, and reporting definitions in a manifest before `finalize` evaluates 2023–2024. Evidence: `validate` writes the versioned selection manifest and prints its digest; integration checks reject a wrong digest/configuration before the held-out file is parsed.
- [x] Make `finalize` verify the manifest/input hashes, write to a new run directory, and refuse silent overwrite. Record that a final test has been inspected. The guard is provenance tracking, not a guarantee against deliberate retuning. Evidence: frozen run `artifacts/runs/frozen-test-2023-24/` records `test_inspected: true`, its inspection time, code/data/config hashes, and the required test period. Integrity/report checks passed.
- [x] Save `predictions.csv`: issue date, target date, split, actual radiation, and one prediction column per approach, with documented units. Evidence: synthetic integration verifies the required dated test rows, columns, units, and 2023–2024 boundaries.
- [x] Save `forecast_metrics.json`, `decision_metrics.csv`, `data_quality.json`, `selection_manifest.json`, and `run_manifest.json`. Reject nonfinite/invalid JSON values and version each artifact schema. Evidence: strict serializers and finalized-run loader validate all required schemas/checksums; synthetic artifact tests pass.
- [x] Include environment versions, seed, source checksums, configuration hash, code identifiers, and row counts. Record "uncommitted" plus relevant file hashes when no commit exists; never fabricate a commit SHA. Evidence: synthetic run manifest tests check the recorded environment, seed, row counts, input/code hashes, and `uncommitted` identity.
- [x] Produce a report with the comparison, seasonal behavior, unfavorable examples, scenario sensitivity, and limitations. Compare identical target dates across approaches. Evidence: synthetic report assertions cover these sections; report loader checks common dates and settings for all approaches.
- [x] Add an offline integration test using a tiny synthetic fixture for preparation → validation → final evaluation → report loading. It checks contracts and reproducibility, not whether a model happens to win. Evidence: `tests/test_finalize.py` runs the pipeline twice and checks identical prediction/decision output, freeze/hash failures, schema and overwrite behavior; full suite passes.
- [x] Add only necessary new dependencies as direct declarations. Record the resolved environment before final delivery; do not upgrade packages during the final experiment without a reason and re-verification. No new dependencies were needed; clean-environment `pip check` passed. Frozen run environment: Python 3.13.12, NumPy 2.5.4, pandas 3.1.1, scikit-learn 1.9.1, Streamlit 1.67.0.
- [x] Document and verify a clean-environment installation and full CLI sequence. Evidence: fresh install and cached `fetch` → `prepare` → `validate` → `finalize` → `report` passed; final artifacts and integrity checks passed. Setup commands are in README.

Frozen final-run evidence (`artifacts/runs/frozen-test-2023-24/`): 731 eligible targets from 2023-01-01 through 2024-12-31, with six warmup exclusions; Ridge was frozen from validation only. Forecast MAE/RMSE/bias in radiation `kWh/m²/day`: persistence 0.5809/0.7836/+0.0002; training-only seasonal mean 0.7283/0.9297/+0.0822; Ridge 0.5455/0.7140/+0.0186. Primary hypothetical scenario totals in simulated kWh: persistence shortfall/excess 274.00/274.00; seasonal 570.10/22.76; Ridge 385.00/111.35; always-prepare 0/10344.50. Capacity-floor shortfall is zero for all four. Across 27 fixed sensitivity scenarios, Ridge had lower mean shortfall than persistence in 2/27 and lower excess in 25/27 (never both); against seasonal mean, lower shortfall in 27/27 and lower excess in 7/27. Forecast improvement is modest and does not establish reserve reliability improvement over persistence. All demand/capacity/generation settings remain hypothetical; historical observations' operational availability is unverified.

The run manifest records schema-v1 artifact hashes, input/config/code checksums, environment, seed 42, row counts (2185 train, 365 validation, 731 test, 6 test exclusions), validation-selection manifest SHA-256 `d4e4d2f86ba3ac795d5e33adb313a0bb8fc64b438aaafe4a86d6af4670e92653`, and code identity `uncommitted`. Raw POWER response SHA-256: `925dde6e86766df7f0f65c91d77947e7306068892fd02ddec0c5b891d28c908`. The report, `predictions.csv`, `forecast_metrics.json`, `decision_metrics.csv`, `decision_predictions.csv`, `data_quality.json`, `selection_manifest.json`, and `run_manifest.json` are present with recorded hashes; integrity checks passed.

Planned command contract after implementation:

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe -m sunreserve fetch
& .\.venv\Scripts\python.exe -m sunreserve prepare
& .\.venv\Scripts\python.exe -m sunreserve validate
$manifestHash = (Get-FileHash .\artifacts\validation\selection_manifest.json -Algorithm SHA256).Hash.ToLower()
# Inspect the manifest and validation results before running the held-out test.
& .\.venv\Scripts\python.exe -m sunreserve finalize --manifest-sha256 $manifestHash --run-id frozen-test-2023-24
& .\.venv\Scripts\python.exe -m sunreserve report --run-id frozen-test-2023-24
& .\.venv\Scripts\python.exe -m pytest
& .\.venv\Scripts\ruff.exe check .
& .\.venv\Scripts\ruff.exe format --check .
& .\.venv\Scripts\python.exe -m streamlit run app.py
```

The command block and `/artifacts/validation` paths above document the previously verified historical baseline sequence. Current validation uses `sunreserve validate --run-id <unique-id>` and publishes under `artifacts/experiments/<id>/`. To finalize that experiment, inspect its selection manifest and run `sunreserve finalize --experiment-id <id> --manifest-sha256 <digest> --run-id <unique-run-id>`; finalized artifacts are published under `artifacts/runs/<unique-run-id>/`. Omitting `--experiment-id` retains the historical baseline finalize inputs and freeze guard.

`finalize` requires the inspected selection-manifest digest and an optional safe run ID; `report` requires that exact run ID. Raw data, prepared data, model objects, and generated experiment runs remain ignored local files. The source repository is public, but there is no packaged public demo bundle or hosted app; a reproducible download-and-run path is documented in the README.

## 11. Report and hackathon delivery

Primary files: `docs/writeup.md`, `README.md`, source/assumption documentation, and optional demo script.

- [x] Write the research write-up: specific user/problem, prototype, data/model choices, baseline and held-out results, failures, next steps, limitations, and source/AI-tool credits. Evidence: `docs/writeup.md` reports the verified frozen run and reproducing artifacts.
- [x] Explain that forecast improvement and decision improvement are different outcomes. State explicitly if either fails to improve. Evidence: the write-up reports modest forecast gains and the less favorable Ridge-versus-persistence shortfall trade-off without claiming reliability improvement.
- [x] Cite NASA POWER, libraries/models/tutorials, and every coding assistant used. Verify data-use terms and current parameter documentation during ingestion. Evidence: NASA POWER docs/terms were checked during Stage A and the write-up credits the data, direct software/model sources, Luna, Codex, and advisory Jev review. The POWER referencing guide's requested notification is recorded; no email/notification was sent. No root `LICENSE`, `COPYING`, or `NOTICE` file exists and no license was added, so this project patch grants no project reuse license.
- [ ] Prepare an optional video of at most two minutes: problem, real historical example, baseline comparison, and one limitation. Do not script numerical claims until measured results exist.
- [x] Ensure the public source project and README assets are usable without private local files. Evidence: the pushed package commit `33c650bca3cbc82e365914792603773e7f02ce42` is publicly accessible; the connector verified the exact hosted tree and changed-file blobs, and signed-out Edge checks verified the README and rendered assets. Reproduction is documented in the README. At the 2026-11-19 pre-integration checkpoint, no hosted app or packaged demo bundle was recorded.
- [ ] Check public links while signed out, saved notebook outputs if a notebook is used, team names/emails/schools, and the latest submission version.
- [ ] Preserve a submission snapshot before the deadline: Monday, 16 November 2026, 10:29 AM IST, based on the organizer's project briefs.

The visual README package and submission materials were committed and pushed to the authorized feature branch at package commit `33c650bca3cbc82e365914792603773e7f02ce42`. This paragraph records the state of that historical checkpoint only; later receipt and PR evidence is summarized in the current delivery status above.

## 12. Completion criteria

- [x] One documented location and source, with cached provenance and quality checks.
- [x] Chronological, leakage-tested forecasting with both simple baselines.
- [x] Frozen validation-selected approach evaluated on unseen final dates.
- [x] Correct, clearly hypothetical reserve simulation with fair comparisons.
- [x] Reproducible artifacts, focused tests, passing lint/format, and an offline demo.
- [x] Honest results and failure examples included where applicable.
- [x] Complete write-up, citations, and setup instructions. The public feature branch contains the reviewed write-up, README/assets, submission write-up, demo script, and reproducible source package at commit `33c650bca3cbc82e365914792603773e7f02ce42`.

Optional follow-up work only after this checklist: quantile forecasts with calibrated uncertainty, measured school demand, a proper PV model, verified real-time data availability, or field testing. Each would need its own scope and evidence; none is required for the hackathon minimum.

## 13. Stage F: fresh trained-model inference

This improvement adds a separate fresh prediction mode while retaining the cached historical
replay. Prediction mode accepts seven consecutive daily rows ending on an issue date, runs a
Ridge model trained on the existing training split, and applies the existing reserve formula
without claiming a realized outcome. It does not establish that historical or user-entered
observations were available at forecast issue time.

- [x] Reuse one shared feature builder for batch rows and a validated seven-day inference window. Evidence: feature-parity, date/gap/future-date, field, range, and input-sensitivity tests in `tests/test_inference.py` and `tests/test_features.py`.
- [x] Export/load a versioned numeric model bundle with feature schema, training dates/count, training-fitted imputer/scaler statistics, Ridge coefficients, seasonal baseline, and manifest checksum. Evidence: `validate` wrote `models/ridge_model.json`; bundle checksum is bound in the validation manifest, and tests compare prediction against the fitted sklearn pipeline and reject missing, corrupt, mismatched, mixed-partition, and post-training cases.
- [x] Provide CLI `predict` with strict JSON observation validation, model identity, radiation/predicted-proxy/reserve outputs, assumptions, and simple baselines. Evidence: end-to-end CLI test and 32 focused tests passed.
- [x] Define the development reserve selection objective before computing new candidate scores: mean simulated shortfall plus mean excess prepared reserve, equal 1:1 energy weights on the fixed primary scenario. Evidence: selection definition/weights/settings/scores and model hash are stored in the versioned validation manifest; only the 2022 validation partition was scored. Current score: persistence 0.953888, Ridge 0.841804, seasonal mean 1.080013 kWh/day; Ridge remains the development selection. This is not a new final-period claim.
- [x] Compute planned reserve for forecast inputs without requiring or fabricating next-day actual radiation. Evidence: hand-calculated tests verify predicted generation/reserve, and assert no realized shortfall/excess columns are emitted.
- [x] Add a distinct default Streamlit prediction mode, editable observation input, units, model identity, baseline comparison, assumptions, and historical-data availability cautions; replace deprecated dataframe width arguments. Evidence: fresh-mode browser acceptance confirmed changed predictions and planned reserves, units/model/baselines, and the historical-data caveat.
- [x] Complete browser acceptance of both modes. Evidence: in the fresh mode, changing daily radiation from 5.7787 to 7.7787 changed the prediction; changing hypothetical demand from 15 to 25 changed planned reserve. Historical date selection from 2022-01-01 to 2022-01-02 worked. Charts, units, model identity, baseline comparison, assumptions, failure examples, and missing-artifact behavior were checked in the browser. Invalid-input rejection was verified by automated tests, not directly in the browser.
- [x] Run full test/lint/format and clean-environment dependency checks after the improvement; inspect outputs and preserve the 2023–2024 run. Evidence: 133 tests passed; Ruff check was clean; Ruff format reported 33 files already formatted; `pip check` was clean. The copied selection manifest and eight companion files in `frozen-test-2023-24` are unchanged. The current validation selection manifest was intentionally updated; the other five validation files are unchanged.
- [x] Evaluate only the frozen selection on a genuinely untouched later period before making any new final-performance claim. Evidence: the separately sourced 2025 holdout in Stage 14 was evaluated once after selection was frozen on 2022 validation; no fitting, tuning, or selection used 2025.
- [x] Complete-diff review. Evidence: 33 successful, untruncated Jev 1.14.0 calls (524,777 input / 16,467 output) covered all application/source files, all 12 tests, docs, packaging/model files, both selection-manifest contents (112 metric records in six chunks; the copied 2025 manifest was hash-identical), all 2025 prediction/decision rows (two plus six chunks), and the final `.gitattributes`, `pyproject.toml`, and sidecar review; `evaluate.py` was covered in two exact line-range excerpts. Exclusions were confirmed unchanged pre-existing 2023–2024/validation/raw/processed artifacts and empty placeholders. One initial full-predictions CSV call returned non-JSON; two successful chunks then covered and reconstructed the exact rows with newline normalization only. Generic incomplete-context/correctness-concern gates were checked against source, tests, and artifact hashes; no concrete file/line findings remained. The staged sidecar blob SHA-256 matches the 2025 run manifest after raw-byte attributes were applied, and `git diff --cached --check` is clean.
- [x] Complete source and license audit. Evidence: the current AWS Registry and NASA POWER referencing guidance are cited in `docs/data_sources.md`; their requests and the fact that no notification was sent are stated plainly. No root `LICENSE`, `COPYING`, or `NOTICE` file was found and no project license was added; this is documented as a reuse limitation, not a code defect.
- [x] Commit and push the reviewed feature-branch package, then verify hosted files through the GitHub connector. Evidence: package commit `33c650bca3cbc82e365914792603773e7f02ce42`; recursive hosted tree is untruncated with 79 blobs and the exact local/remote path set, and all 14 changed-file blob SHAs match. This is the package commit, not a claim about the final HEAD after this plan/write-up follow-up. The later receipt and PR state is recorded above.

## 14. Separate untouched 2025 evaluation path

Implement a distinct one-time runner for the separately sourced 2025 year. This path applies
the frozen Ridge bundle and fixed persistence/seasonal baselines on identical eligible target
dates; 2016–2024 observations are permitted only as earlier feature context. The runner must
verify both frozen SHA-256 digests, source/provenance schema and units, and a new immutable run
ID before writing versioned predictions, metrics, report, quality data, and hashes. It must
not fit, tune, or select on 2025, and does not replace or rewrite the 2023–2024 run.

- [x] Add an isolated evaluation module and explicit CLI entry point without changing code files bound by the frozen development-manifest code hashes. Evidence: the module verifies the exact manifest/model digests and source sidecar; the frozen hashes remain unchanged.
- [x] Score only 2025 target dates and compare Ridge, persistence, and training-only seasonal mean on common eligible dates; calculate the fixed hypothetical reserve score. Evidence: synthetic integration checks issue/target alignment, 2025-only unique targets, equal baseline sample dates, and mean-shortfall-plus-mean-excess arithmetic.
- [x] Persist versioned output artifacts, source/model/config/code hashes, environment metadata, and refuse run-ID overwrite. Evidence: offline artifact checksums and overwrite rejection passed in `tests/test_holdout_2025.py`.
- [x] Add synthetic tests for CLI digest failure, model/manifest mismatches, date alignment, common-date baselines, arithmetic, artifact hashes, and overwrite. Evidence: `.venv/Scripts/python.exe -m pytest tests/test_holdout_2025.py -q` passed 2 tests; Ruff check and format check passed for the new module, CLI, and test.
- [x] Apply the frozen Ridge and two fixed baselines to the separate actual 2025 snapshot once, after freezing model and selection digests. Evidence: `artifacts/runs/frozen-test-2025/` records 365 common targets (2025-01-01–2025-12-31), zero exclusions, zero missing source days, and zero 2023–2024 targets scored. Ridge (`sunreserve.ridge.train-only.v1`, 2,185 training rows through 2021-12-31; bundle SHA-256 `40a11e5119b7838d10a4a7ef7cc5bbf564998759b3954d4b865fdfe3f392b61f`) had radiation MAE/RMSE 0.5387915/0.7247856 kWh/m²/day versus persistence 0.5885584/0.8145780 and training-only seasonal mean 0.7119433/0.9344102. On the predeclared equal 1:1 energy score, mean shortfall + mean excess, Ridge scored 0.787398 kWh/day (0.607825 shortfall + 0.179572 excess), persistence 0.871433 (0.437777 + 0.433656), and seasonal mean 0.927848 (0.881887 + 0.045961). Thus Ridge radiation MAE beats both baselines and its score is lower under that hypothetical formula, but its simulated shortfall exceeds persistence. These are descriptive simulated outcomes, not cost, measured generation, or operational reliability results; NASA POWER solar latency prevents a verified live next-day availability claim. Selection manifest SHA-256: `9cd60c7ab8b4e41078c1bf44d6698178334e6b9b6dba3d01a88f4e9a9d097fff`.
- [x] Preserve the earlier final-period replay and accurately document regeneration of validation selection. Evidence: the copied selection manifest and its eight companion files in `artifacts/runs/frozen-test-2023-24/` are unchanged. The current `artifacts/validation/selection_manifest.json` was intentionally regenerated to bind the new model and decision objective; the other five validation files remain unchanged. Do not describe all 14 unchanged files plus the regenerated manifest as unchanged.

## 15. Visual README and public-source delivery checkpoint

The visual README package was pushed as commit `33c650bca3cbc82e365914792603773e7f02ce42` to `feature/veritas-sunreserve`. The GitHub connector confirmed an untruncated recursive tree with 79 blobs, the exact local/remote path set, and matching blob SHAs for all 14 changed files. This identifies the package commit only; a later plan/write-up follow-up is not included in that SHA.

Signed-out Edge 157 checks covered the GitHub README in wide light, wide dark, and 390 px narrow layouts. All eight images decoded in every layout; the narrow document width remained 390 px, dark body/text colors were legible, and the Mermaid frame rendered light labels on dark nodes. The chart generator ran twice with stable identical hashes for its three PNGs, derived from 365 frozen prediction rows and three decision-metric rows. Earlier root verification passed 136 tests in 34.85 seconds, Ruff check, 38-file Ruff format check, and `pip check`. On a clean clone of the exact pushed package commit, Python 3.13.12 `pip install -e ".[dev]"` succeeded and installed `sunreserve` from that clone with Pillow 13.0.0; `pip check` was clean, 136 tests passed in 45.54 seconds, Ruff check passed, and Ruff format reported 38 files already formatted. The first non-elevated install attempt had hit DNS while resolving setuptools; the later clean-clone install passed. No frozen 2025 run artifact changed. These visual and runtime checks describe the recorded package snapshot only; they do not establish the visual/mobile acceptance for the candidate at this checkpoint.

## References

- [Organizer project briefs](https://drive.google.com/file/d/15tYD48XRmCgWbk1_lLDtc8lebi6qm-4-/view): Climate requirements, judging, and submission rules reviewed in this chat.
- [NASA POWER daily API documentation](https://power.larc.nasa.gov/docs/services/api/temporal/daily/): source interface; verify current details during ingestion.

Jev plan/diff judgments are advisory inputs. No probability from those reviews is evidence of scientific performance or competition outcome.
