# SunReserve

**Track:** Climate
**Team:** Shaurya Saria

## The problem

A hypothetical school facilities manager must decide how much backup energy to prepare when tomorrow’s solar contribution is uncertain. Over-preparing can waste capacity, while under-preparing can leave a shortfall; the project asks whether a small radiation forecast changes that simulated trade-off compared with simple rules.

## What we built

SunReserve is a Python research prototype with a Streamlit interface. A user edits seven consecutive daily radiation, temperature, and humidity observations; the app validates them and runs a fresh trained Ridge prediction for the next date. It displays that estimate beside persistence and seasonal-mean baselines and translates each into a planned reserve under editable hypothetical facility assumptions. A separate historical replay shows saved evaluation results. The app does not control equipment or claim measured electricity production.

## How it works

**Data.** We used NASA POWER Daily API estimates for one Bengaluru reference point (12.9716° N, 77.5946° E), with radiation, 2 m temperature, and relative humidity from 2016–2024: 3,288 calendar days. The source variables had no missing values. We kept calendar gaps explicit, built next-calendar-day targets, and excluded six initial rows without a complete seven-day feature window. Chronological target splits were 2,185 training rows through 2021, 365 validation rows in 2022, and 731 test rows in 2023–2024. A separately sourced, untouched 2025 holdout contains 365 common evaluation dates. POWER values are gridded historical estimates, not school or rooftop measurements.

**Model.** The selected `sunreserve.ridge.train-only.v1` pipeline fits median imputation, scaling, and Ridge regression on training rows only; validation data selected it, and 2025 was not used for fitting, tuning, or selection. We compare it with persistence (tomorrow equals today) and a training-only month-of-year mean.

## Results

On the frozen 2025 holdout, radiation MAE was 0.539 kWh/m²/day for Ridge, 0.589 for persistence, and 0.712 for the seasonal mean. In the primary hypothetical reserve replay (15 kWh/day demand, 15 kWh capacity, 20 m², 20% conversion, and 0.80 derating), mean simulated shortfall was 0.608 kWh/day for Ridge, 0.438 for persistence, and 0.882 for the seasonal mean. Thus Ridge improved radiation MAE but had higher simulated shortfall than persistence.

The equal-weight score is mean shortfall plus mean excess reserve: 0.787 kWh/day for Ridge, 0.871 for persistence, and 0.928 for the seasonal mean. It is a hypothetical comparison with equal energy weights, not a validated cost function, reliability measure, or operational recommendation. The simulation converts radiation to an electricity proxy using assumed panel area, efficiency, and derating. In the 124-row low-solar slice (threshold 4.5967 kWh/m²/day, frozen from training), Ridge MAE was 0.724, persistence 0.722, and the seasonal mean 1.106; Ridge did not beat persistence there. The demo screenshot separately uses a hypothetical 25 kWh/day demand; its 7.75 kWh planned Ridge reserve is not a measured outcome or the 15 kWh/day test scenario.

## What we'd do next

First, evaluate with weather inputs whose availability at each forecast issue time is verified. Then, if a school partner and measured data become available, compare against site generation and demand and agree on a reserve criterion before any field trial.

## Sources

- Dataset: [NASA POWER Daily API](https://power.larc.nasa.gov/docs/services/api/temporal/daily/) and [data provenance](data_sources.md).
- Model and software: [scikit-learn Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html), plus [pandas](https://pandas.pydata.org/docs/), [NumPy](https://numpy.org/doc/), and [Streamlit](https://docs.streamlit.io/).
- Code and frozen outputs: [SunReserve public repository](https://github.com/icecold009/sunreserve/tree/feature/veritas-sunreserve).
- Format: [Veritas AI Hacker Guide](https://drive.google.com/file/d/1_XIisoApHfiuIzkPev6hKJ4SlViGH4pm/view).
- AI tools: OpenAI Codex/Luna assisted with implementation and documentation; Jev provided advisory review. These tools did not supply the NASA observations or independently validate the model’s real-world performance.
