# SunReserve final historical replay

This report evaluates the frozen validation selection on the explicitly finalized held-out test period. It is a historical research replay, not a validated operational controller.

- Run: `frozen-test-2023-24`
- Test period: 2023-01-01 through 2024-12-31 (731 eligible targets)
- Model selected using validation only: **ridge**
- Test inspection recorded: 2026-11-14T17:04:50.482358+00:00
- Identity: `uncommitted`; code identity is recorded as source file SHA-256 values.

## Forecast comparison

Solar-radiation units are kWh/m²/day. Model selection was frozen before these held-out metrics were calculated. Persistence and the training-only seasonal mean remain included.

| Approach | MAE | RMSE | Bias (prediction − target) | Low-solar MAE | n |
|---|---:|---:|---:|---:|---:|
| persistence | 0.5809 | 0.7836 | 0.0002 | 0.7442 | 731 |
| seasonal_mean | 0.7283 | 0.9297 | 0.0822 | 1.0554 | 731 |
| ridge | 0.5455 | 0.7140 | 0.0186 | 0.7418 | 731 |

## Seasonal behavior

Seasonal MAE by approach (kWh/m²/day):

| Approach | DJF | JJA | MAM | SON |
|---|---:|---:|---:|---:|
| persistence | 0.6310 | 0.6020 | 0.5048 | 0.5867 |
| seasonal_mean | 0.7690 | 0.8014 | 0.7138 | 0.6285 |
| ridge | 0.6076 | 0.5664 | 0.4774 | 0.5315 |

## Reserve comparison

All reserve and electricity values are hypothetical simulated kWh/day. Shortfall and excess reserve are separate quantities; no monetary or emissions savings are inferred.

### Primary hypothetical scenario

| Approach | Total simulated shortfall (kWh) | Shortfall days | Total excess prepared reserve (kWh) | Capacity-limited floor (kWh) |
|---|---:|---:|---:|---:|
| persistence | 274.00 | 150 | 274.00 | 0.00 |
| seasonal_mean | 570.10 | 219 | 22.76 | 0.00 |
| ridge | 385.00 | 178 | 111.35 | 0.00 |
| always_prepare_backup | 0.00 | 0 | 10344.50 | 0.00 |

### Fixed sensitivity grid

The report includes 27 predeclared demand/capacity/buffer settings for every forecast approach and the always-prepare reference.

| Approach | Mean daily shortfall range (kWh/day) | Mean excess reserve range (kWh/day) |
|---|---:|---:|
| persistence | 0.0391–0.8242 | 0.0655–3.5868 |
| seasonal_mean | 0.0978–1.2229 | 0.0000–3.4700 |
| ridge | 0.0764–0.8459 | 0.0040–3.5327 |
| always_prepare_backup | 0.0000–0.0978 | 6.5528–16.4550 |

## Unfavorable forecast examples

For the validation-selected `ridge` approach, examples are selected by the largest positive and largest negative test prediction errors; this rule does not filter for favorable days.

| Example | Target date | Prediction minus target (kWh/m²/day) |
|---|---|---:|
| Largest overprediction | 2024-11-12 | 3.1610 |
| Largest underprediction | 2023-02-04 | -2.3436 |

## Limitations

- NASA POWER supplies model/satellite-derived daily solar radiation; it is not measured electricity generation.
- Demand, array area, conversion efficiency, derating, reserve capacity, and buffer are hypothetical assumptions.
- The generation proxy is not a validated PV yield model; the reserve calculation has no inter-day battery state, costs, outages, or dispatch constraints.
- Forecast-error metrics and reserve metrics answer different questions. The replay is not a causal savings or emissions analysis.
- Historical data availability does not establish operational forecast availability. No real facility deployment is demonstrated.

The full run settings are recorded in `selection_manifest.json`, `run_manifest.json`, and `assumptions.md`.
