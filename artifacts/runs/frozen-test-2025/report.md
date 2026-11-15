# SunReserve untouched 2025 holdout · frozen-test-2025

This run applies the frozen model and validation-selected decision contract to the separate 2025 historical period. It does not refit, tune, or select on 2025. Targets cover 2025 only; 2023–2024 are not scored in this run.

Common eligible targets: 365; excluded 2025 target dates: 0. Frozen model: `sunreserve.ridge.train-only.v1`.

Radiation metrics use `kWh/m²/day`. Reserve outcomes use hypothetical simulated `kWh/day` assumptions and are not real facility performance. The decision score is `mean simulated shortfall kWh/day + mean excess prepared reserve kWh/day; equal 1:1 energy weights, not monetary costs`. The development-selected approach was `ridge` using 2022 validation only.

| Approach | Radiation MAE | Radiation RMSE | Mean shortfall (kWh/day) | Mean excess reserve (kWh/day) | Mean shortfall + excess (kWh/day) |
| --- | ---: | ---: | ---: | ---: | ---: |
| ridge | 0.5388 | 0.7248 | 0.6078 | 0.1796 | 0.7874 |
| persistence | 0.5886 | 0.8146 | 0.4378 | 0.4337 | 0.8714 |
| seasonal_mean | 0.7119 | 0.9344 | 0.8819 | 0.0460 | 0.9278 |

These are historical estimates. NASA POWER publication latency and user-input provenance do not establish forecast-time availability. The electricity conversion, demand, capacity, and buffer remain hypothetical; this is not an operational controller, cost model, or causal analysis. See `docs/assumptions.md` and `docs/data_sources.md`.
