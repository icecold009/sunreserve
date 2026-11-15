# SunReserve: a historical solar-radiation replay

## Problem and prototype

A hypothetical school facilities manager must decide how much backup energy to prepare
when tomorrow's solar contribution is uncertain. SunReserve combines fresh Ridge
inference from a user-supplied seven-day observation window with a cached historical
replay. It provides source validation, chronological forecasting, a fixed reserve
sensitivity grid, reproducible model and result artifacts, and a Streamlit demonstration.
This is a historical research prototype, not an operational controller or a validated
deployment.

## Data and method

The input is NASA POWER Daily API data for a Bengaluru reference coordinate (12.9716° N,
77.5946° E) from 2016-01-01 through 2024-12-31 in UTC: all-sky surface shortwave downward
radiation (`kWh/m²/day`), 2-meter temperature (`°C`), and relative humidity (`%`). POWER
provides model/satellite-derived estimates, not measurements of school demand or rooftop
electricity generation. The source contains 3,288 daily records with no missing values in
these variables. The supervised builder excludes six initial-window rows, then uses
chronological target dates: 2016–2021 training, 2022 validation, and 2023–2024 test.
Each row predicts the next calendar day; missing calendar dates are not skipped by a row
shift.

Persistence predicts tomorrow's radiation from today's observation. The seasonal-mean
baseline uses training-target months only. Ridge preprocessing and coefficients are fit
on training rows only. Ridge was selected by validation MAE in 2022 and frozen before
final evaluation. Final test results did not inform model, feature, preprocessing,
threshold, or reserve-setting choices. All three forecasting approaches are compared on
the same 731 test targets. The test contains 731 eligible targets from 2023-01-01 through
2024-12-31; six initial-window exclusions are recorded in the run manifest.

The added prediction flow accepts seven consecutive user-entered daily observations and
uses the same shared feature builder, training-fitted median imputation/scaling, and
training-only Ridge coefficients. Its model is a versioned numeric JSON bundle whose hash
is bound to the validation manifest. The flow computes fresh inference; it does not look up
saved validation or test predictions. Persistence and training-only seasonal-mean forecasts
are shown beside Ridge. It reports a planned reserve only because no next-day actual is
available to calculate realized shortfall or excess.

To make development model choice reflect the simulated decision, we predeclared the
primary-scenario score `mean shortfall + mean excess prepared reserve`, with equal 1:1
weights on the two `kWh/day` quantities. This is not a cost function. On the 2022 validation
period, the scores were 0.953888 for persistence, 0.841804 for Ridge, and 1.080013 for the
seasonal mean; Ridge therefore remained the development choice. These are selection scores
on development data, not new final-period results. The already-inspected 2023–2024 test is
not reused to tune this flow.

For the decision replay, the simplified electricity proxy is radiation × hypothetical
panel area × conversion efficiency × derating. The primary assumptions are 20 m², 0.20,
0.80, daily demand 15 kWh, reserve cap 15 kWh, and buffer zero. These values are
illustrative, not observed. The experiment reports prepared reserve, shortfall, and excess
as simulated `kWh/day` quantities. It does not estimate money, emissions, or causal
benefit.

## Frozen 2023–2024 test results

Forecast errors measure solar radiation, in `kWh/m²/day`:

| Approach | MAE | RMSE | Bias (prediction − actual) | Low-solar MAE |
| --- | ---: | ---: | ---: | ---: |
| Persistence | 0.5809 | 0.7836 | +0.0002 | 0.7442 |
| Training-only seasonal mean | 0.7283 | 0.9297 | +0.0822 | 1.0554 |
| Ridge, frozen on validation | 0.5455 | 0.7140 | +0.0186 | 0.7418 |

Ridge modestly improves radiation MAE and RMSE against both baselines. That forecast
result does not imply a reserve reliability improvement. In the primary hypothetical
scenario, cumulative simulated outcomes over the same test dates are:

| Approach | Total shortfall (kWh) | Shortfall days | Total excess prepared reserve (kWh) | Capacity-floor shortfall (kWh) |
| --- | ---: | ---: | ---: | ---: |
| Persistence | 274.00 | 150 | 274.00 | 0.00 |
| Training-only seasonal mean | 570.10 | 219 | 22.76 | 0.00 |
| Ridge | 385.00 | 178 | 111.35 | 0.00 |
| Always prepare backup | 0.00 | 0 | 10,344.50 | 0.00 |

Ridge had 111 kWh more simulated shortfall than persistence, alongside about 163 kWh
less excess prepared reserve. The primary scenario therefore shows a trade-off and no
reliability improvement over persistence. The always-prepare reference has no shortfall
at the cost of much more prepared reserve; it is an illustrative bound, not a recommended
policy.

The 27 fixed sensitivity scenarios also show no universal winner. Ridge had lower mean
shortfall than persistence in 2/27 and lower mean excess in 25/27, never both at once.
Against the seasonal-mean baseline, Ridge had lower mean shortfall in 27/27 and lower
mean excess in 7/27. These counts describe only the predeclared scenarios and historical
replay. The detailed report includes seasonal metrics and the largest positive and
negative Ridge errors, selected by error magnitude rather than favorable outcome.

## Separate untouched 2025 replay

The later holdout applies the already-frozen training-only Ridge bundle and the two simple
baselines without refitting, tuning, or selecting on 2025. Development model selection used
the 2022 validation period only. The frozen model is
`sunreserve.ridge.train-only.v1`, trained on 2,185 rows through 2021-12-31, with bundle
SHA-256 `40a11e5119b7838d10a4a7ef7cc5bbf564998759b3954d4b865fdfe3f392b61f`. The bound
selection-manifest SHA-256 is
`9cd60c7ab8b4e41078c1bf44d6698178334e6b9b6dba3d01a88f4e9a9d097fff`.

All approaches were compared on the same 365 targets from 2025-01-01 through 2025-12-31.
The source had zero missing days and the run excluded zero targets; no 2023–2024 targets
were scored. Forecast errors are the NASA POWER solar-radiation estimate, not electricity:

| Approach | Radiation MAE (kWh/m²/day) | Radiation RMSE (kWh/m²/day) |
| --- | ---: | ---: |
| Ridge | 0.5387915 | 0.7247856 |
| Persistence | 0.5885584 | 0.8145780 |
| Training-only seasonal mean | 0.7119433 | 0.9344102 |

Ridge's radiation MAE is lower than both baselines. The reserve replay gives a different
comparison. Under the predeclared primary scenario and score—mean simulated shortfall plus
mean excess prepared reserve, with equal 1:1 energy weights—the results are:

| Approach | Mean shortfall (kWh/day) | Mean excess prepared reserve (kWh/day) | Score (kWh/day) |
| --- | ---: | ---: | ---: |
| Ridge | 0.607825 | 0.179572 | 0.787398 |
| Persistence | 0.437777 | 0.433656 | 0.871433 |
| Training-only seasonal mean | 0.881887 | 0.045961 | 0.927848 |

Ridge's score is lower than both simple baselines under this predeclared equal-energy
formula, while its simulated shortfall is higher than persistence. The score is a hypothetical
energy ranking, not a monetary objective or a reliability guarantee. The reserve arithmetic
uses simulated electricity from a solar-radiation proxy and hypothetical facility assumptions;
it does not report measured generation, demand, savings, or real reserve performance. This is
a descriptive historical replay, not a validated operational controller. NASA POWER's
approximately 5–7 day solar-data latency means these observations do not support a verified
next-day live-input claim.

## Reproducibility and verification

The frozen run is `artifacts/runs/frozen-test-2023-24/`. Its versioned manifests record
the configuration and validation-selection hashes, source and prepared-data checksums,
code-file hashes, environment, seed, test inspection, artifact schemas, and SHA-256 for
each report artifact. The test run records code identity as `uncommitted`; no commit SHA
was fabricated. The environment was Python 3.13.12, NumPy 2.5.4, pandas 3.1.1,
scikit-learn 1.9.1, and Streamlit 1.67.0.

The separate 2025 run is saved under `artifacts/runs/frozen-test-2025/`; its manifest records
365 eligible targets, zero exclusions, and zero 2023–2024 targets scored. To keep the earlier
evaluation frozen, the copied selection manifest and its eight companion files in
`frozen-test-2023-24` were left unchanged. The current validation selection manifest was
intentionally regenerated to bind the fresh model and predeclared reserve objective; the
other five validation files remain unchanged.

In the clean Python 3.13.12 environment, installation and `pip check` were clean; the full
suite passed 133 tests in 44.05s; Ruff check was clean; and Ruff format reported 33 files
already formatted. Browser acceptance covered both modes: editing daily radiation from
5.7787 to 7.7787 changed the fresh prediction, changing hypothetical demand from 15 to 25
changed planned reserve, and historical date selection from 2022-01-01 to 2022-01-02 worked.
Charts, units, model identity, baseline comparisons, assumptions, failure examples, and
missing-artifact behavior were verified in the browser. Invalid-input rejection is covered
by automated tests, not a direct browser interaction. The app does not fetch data or train
at launch.

## Limitations and next steps

NASA POWER's documentation reports approximate latency of 5–7 days for solar parameters
and 2–3 days for meteorology, so these observations are not verified next-day operational
inputs. User-supplied rows may have different provenance, but this prototype does not verify
their forecast-time availability. The radiation-to-electricity proxy is
not a validated photovoltaic yield model. Demand, array, conversion efficiency, derating,
reserve capacity, and buffer are hypothetical. The daily balance omits inter-day battery
state, power limits, outages, tariffs, charging losses, and dispatch constraints. The
results are neither a causal savings analysis nor evidence of real facility benefit.
Further claims would require forecast-available weather data, measured site demand and
generation, an appropriate physical model, and prospective field validation.

No public demo bundle, deployment, video upload, or hackathon submission was created.
Those publication and submission actions were outside the authorized implementation.
The local reproduction instructions are in the [README](../README.md).

## Sources and credits

- NASA POWER, [Daily API documentation](https://power.larc.nasa.gov/docs/services/api/temporal/daily/),
  [data FAQ and latency](https://power.larc.nasa.gov/docs/faqs/data/#latency),
  [referencing and redistribution notice](https://power.larc.nasa.gov/docs/referencing/), and
  [AWS Open Data Registry entry](https://registry.opendata.aws/nasa-power/). Cite the NASA
  POWER Project and follow its request for notice when transmitting data to other researchers.
- pandas ([documentation](https://pandas.pydata.org/docs/)), NumPy
  ([documentation](https://numpy.org/doc/)), scikit-learn Ridge regression and preprocessing
  ([Ridge documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)),
  and Streamlit ([documentation](https://docs.streamlit.io/)).
- AI assistance: Luna (OpenAI) implemented the project; Codex (OpenAI) planned,
  coordinated, reviewed, and validated it; Jev provided advisory plan and code reviews.
