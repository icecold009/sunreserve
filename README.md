# SunReserve

SunReserve pairs fresh trained-model inference from a user-supplied seven-day observation
window with a reproducible historical replay. It asks whether a compact next-day
**solar-radiation** model can improve a hypothetical school's daily backup-energy
preparation over simple baselines. It is a research prototype, not a live weather feed,
rooftop-generation measurement, operational controller, or demonstrated school deployment.

NASA POWER daily radiation is model/satellite-derived `kWh/m²/day`. The reserve
experiment converts it to a hypothetical electricity proxy and reports prepared reserve,
shortfall, and excess in simulated `kWh/day`. Demand, panel area, efficiency, derating,
capacity, and buffer are assumptions, not measurements.

## Install

Python 3.11–3.13 is supported. From the repository root in PowerShell:

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe -m pip check
```

The clean-environment installation and dependency check passed with Python 3.13.12,
NumPy 2.5.4, pandas 3.1.1, scikit-learn 1.9.1, and Streamlit 1.67.0. `fetch` uses the
standard-library HTTP client. NumPy, pandas, scikit-learn, and Streamlit are runtime
dependencies; pytest and Ruff are in the development extra. Generated data and experiment
artifacts are local and ignored by Git.

## Reproduce the experiment

Download is explicit; every command after `fetch` uses cached files only.

```powershell
& .\.venv\Scripts\python.exe -m sunreserve fetch
& .\.venv\Scripts\python.exe -m sunreserve prepare
& .\.venv\Scripts\python.exe -m sunreserve validate
$manifestHash = (Get-FileHash .\artifacts\validation\selection_manifest.json -Algorithm SHA256).Hash.ToLower()
# Inspect the selection manifest and validation outputs before final evaluation.
& .\.venv\Scripts\python.exe -m sunreserve finalize --manifest-sha256 $manifestHash --run-id frozen-test-2023-24
& .\.venv\Scripts\python.exe -m sunreserve report --run-id frozen-test-2023-24
```

`validate` fits preprocessing and models on 2016–2021 training targets and selects among
Ridge, persistence, and training-only seasonal mean using 2022 validation MAE. The final
test period is not read until `finalize`, which verifies the frozen manifest and input/code
hashes, writes a new run directory, and refuses overwrite. The completed run is
`artifacts/runs/frozen-test-2023-24/`; its run manifest records an `uncommitted` code
identity rather than an invented commit. Its copied historical selection manifest SHA-256 is
`d4e4d2f86ba3ac795d5e33adb313a0bb8fc64b438aaafe4a86d6af4670e92653`; this is intentionally
distinct from the regenerated current validation manifest SHA-256
`9cd60c7ab8b4e41078c1bf44d6698178334e6b9b6dba3d01a88f4e9a9d097fff`, which freezes the
fresh model and reserve-selection objective. The raw POWER response SHA-256 is
`925dde6e86766df7f0f65c91d77947e7306068892fd02ddec0c5b891d28c908`.

To open the cached demonstration after validation:

```powershell
& .\.venv\Scripts\python.exe -m streamlit run app.py
```

The app opens in **Next-day AI prediction** mode. Edit the seven consecutive daily rows, then
run fresh Ridge inference. The model bundle is trained on 2016–2021 training targets only;
its imputer and scaler use training-only fitted statistics. Prediction mode computes a new
forecast from the entered values and never substitutes a cached prediction. It compares the
forecast and planned reserve with persistence and seasonal-mean baselines. The next day's
actual radiation is not supplied, so prediction mode does not report realized shortfall or
excess reserve. The app also keeps the separate 2022 cached historical replay mode; replay
loads saved artifacts and neither fetches nor trains.

Prediction mode requires a validation-generated `models/ridge_model.json` bundle that
matches the checksum in `artifacts/validation/selection_manifest.json`. To run it from a JSON
file, use seven records with the exact keys `date`, `ALLSKY_SFC_SW_DWN`, `T2M`, and `RH2M`:

```powershell
& .\.venv\Scripts\python.exe -m sunreserve predict --observations .\observations.json
```

Dates must be seven consecutive ISO calendar dates ending on the issue date; future-dated
observations are rejected. All fields must be finite, radiation nonnegative, and humidity
between 0 and 100 percent. Model or input
errors fail with an actionable message. NASA POWER historical observation dates do not
establish release-time availability, and solar data latency makes these results unsuitable
for an operational tomorrow forecast. All electricity and reserve assumptions remain
hypothetical. The final run is local; no public demo bundle or deployment was created.

Development model choice uses a predeclared primary-scenario score on the 2022 validation
period: mean simulated shortfall plus mean excess prepared reserve, with equal 1:1 weights
on the two energy quantities. This is a research ranking, not a cost function. The frozen
validation manifest stores the score definition, weights, candidate scores, selected
approach, scenario assumptions, and fitted model checksum. No new final-period claim is
made from these validation scores.

## Frozen 2023–2024 test results

The frozen selection was **Ridge**, chosen on validation data only. The 2023-01-01 through
2024-12-31 test has 731 eligible targets; six initial-window rows were excluded. Every
approach is scored on the same dates. Forecast errors are solar radiation in
`kWh/m²/day`:

| Approach | MAE | RMSE | Bias (prediction − actual) |
| --- | ---: | ---: | ---: |
| Persistence | 0.5809 | 0.7836 | +0.0002 |
| Training-only seasonal mean | 0.7283 | 0.9297 | +0.0822 |
| Ridge (frozen on validation) | 0.5455 | 0.7140 | +0.0186 |

Ridge has a modest forecast-error improvement against both radiation baselines. The
primary hypothetical reserve scenario shows a different trade-off; the totals below are
simulated energy over the 731-day test period, not savings or causal effects:

| Approach | Shortfall (kWh) | Days with shortfall | Excess prepared reserve (kWh) |
| --- | ---: | ---: | ---: |
| Persistence | 274.00 | 150 | 274.00 |
| Training-only seasonal mean | 570.10 | 219 | 22.76 |
| Ridge | 385.00 | 178 | 111.35 |
| Always prepare backup | 0.00 | 0 | 10,344.50 |

No approach had a capacity-floor shortfall in the primary scenario. Ridge had more
shortfall than persistence, while preparing less excess reserve, so the experiment does
not show a reliability improvement over persistence. Across the fixed 27 sensitivity
scenarios, Ridge's mean shortfall was lower than persistence in 2/27 and its mean excess
was lower in 25/27; it was never lower on both measures. Against seasonal mean, Ridge's
mean shortfall was lower in 27/27 and excess was lower in 7/27. See [the full write-up](docs/writeup.md)
and [assumptions](docs/assumptions.md) for the interpretation and limitations.

The report, prediction/decision tables, quality report, selection manifest, and run
manifest are in `artifacts/runs/frozen-test-2023-24/`. The run manifest records artifact
SHA-256 hashes, schema versions, environment, seed, counts, configuration and code hashes,
source checksums, and that the test was inspected. In particular, the prediction file
hash is `215f076fae47f558ad625c918cb72d4057e6d815f8244229e88232ab4acd079b`, and the
decision-metrics hash is
`f674ddccdb0336dd6a1147de5cc800395eee82d8c11a25a8e07f224dd63d2f28`.

## Checks

```powershell
& .\.venv\Scripts\python.exe -m pytest
& .\.venv\Scripts\ruff.exe check .
& .\.venv\Scripts\ruff.exe format --check .
```

Current clean-environment verification used Python 3.13.12: installation and `pip check`
were clean, the full suite passed 133 tests in 44.05s, Ruff check was clean, and Ruff format
reported 33 files already formatted. Browser acceptance covered both modes. In fresh
prediction, changing daily radiation from 5.7787 to 7.7787 changed the prediction, and
changing hypothetical demand from 15 to 25 changed planned reserve. Historical date selection
from 2022-01-01 to 2022-01-02 worked. Charts, units, model identity, baseline comparisons,
assumptions, failure examples, and missing-artifact behavior were verified. Invalid-input
rejection is covered by automated tests; it was not directly exercised in the browser. The
older replay milestone had 113 passing tests before fresh inference was added. The copied
selection manifest and eight companion files in `frozen-test-2023-24` remain unchanged; the
current validation manifest was intentionally regenerated, and the other five validation
files remain unchanged.

## Separate 2025 holdout replay

The opt-in command `python -m sunreserve.holdout_2025_cli evaluate-2025` applies the frozen
training-only Ridge bundle and compares it with persistence and the training-only seasonal
mean on common 2025 targets. It refuses a mismatched selection-manifest or model digest and
never overwrites an existing run ID. It records the 2025-only predictions, radiation metrics,
hypothetical reserve arithmetic, source quality/provenance, hashes, and a descriptive report.
The `2023–2024` rows are only prior-date feature context and are not scored. No model is fit,
selected, or tuned on the 2025 period.

This is the command used for the one-time frozen replay. To reproduce it in a clean copy,
first verify the exact selection and model digests and confirm that the destination run ID is
absent. The current workspace already contains `frozen-test-2025`; the runner refuses to
overwrite it.

```powershell
& .\.venv\Scripts\python.exe -m sunreserve.holdout_2025_cli evaluate-2025 `
  --manifest-sha256 9cd60c7ab8b4e41078c1bf44d6698178334e6b9b6dba3d01a88f4e9a9d097fff `
  --model-sha256 40a11e5119b7838d10a4a7ef7cc5bbf564998759b3954d4b865fdfe3f392b61f `
  --run-id untouched-2025-reviewed
```

Synthetic-only integration coverage exercises the command's digest and overwrite guards,
date alignment, common baseline dates, reserve-score arithmetic, and artifact checksums. The
one-time evaluation has been run and is saved under
`artifacts/runs/frozen-test-2025/`. It contains 365/365 common targets from 2025-01-01 through
2025-12-31, with zero exclusions and zero missing source days. The run scored no 2023–2024
targets; those dates were used only as earlier feature context. The frozen model is
`sunreserve.ridge.train-only.v1` (2,185 training rows through 2021-12-31; model SHA-256
`40a11e5119b7838d10a4a7ef7cc5bbf564998759b3954d4b865fdfe3f392b61f`). The frozen validation
selection manifest SHA-256 is
`9cd60c7ab8b4e41078c1bf44d6698178334e6b9b6dba3d01a88f4e9a9d097fff`; model selection used
2022 validation only.

| Approach | Radiation MAE (kWh/m²/day) | Radiation RMSE (kWh/m²/day) | Mean simulated shortfall (kWh/day) | Mean excess reserve (kWh/day) | Equal-weight score (kWh/day) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ridge | 0.5387915 | 0.7247856 | 0.607825 | 0.179572 | 0.787398 |
| Persistence | 0.5885584 | 0.8145780 | 0.437777 | 0.433656 | 0.871433 |
| Training-only seasonal mean | 0.7119433 | 0.9344102 | 0.881887 | 0.045961 | 0.927848 |

Ridge had lower radiation MAE than both simple baselines. Its simulated mean shortfall was
higher than persistence, while its predeclared score—mean shortfall plus mean excess prepared
reserve with equal 1:1 energy weights—was lower than both baselines. This ranking is only under
that explicitly hypothetical energy score, not a cost or reliability guarantee. Reserve values
are simulated electricity quantities from a radiation proxy and hypothetical panel area,
efficiency, derating, demand, capacity, and buffer. The replay is descriptive historical
research; NASA POWER solar latency means these inputs are not available as verified next-day
live data. Do not rerun this immutable run ID or tune against the holdout.

## Data and credits

The source is the [NASA POWER Daily API](https://power.larc.nasa.gov/docs/services/api/temporal/daily/),
Renewable Energy community, at the Bengaluru reference coordinate. The 2016–2024 development
and historical replay source covers 3,288 calendar days; the three variables have no missing
values. A separate 2025 snapshot supports the untouched replay. See [data sources](docs/data_sources.md)
for units, provenance, terms, and why historical availability does not establish day-ahead
operational availability.

The project uses pandas, NumPy, scikit-learn Ridge and preprocessing tools, and Streamlit.
AI assistance: Luna implemented the project; Codex planned, coordinated, and independently
reviewed and validated it; Jev provided advisory plan and code reviews. Committing,
publishing, public links, video upload, and hackathon submission were not part of this
implementation; no deployment or public demo is claimed.
