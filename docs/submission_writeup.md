# SunReserve

**Track:** AI + Climate | **Team:** Shaurya Saria

## The problem

A hypothetical Bengaluru school facilities manager must plan backup energy before tomorrow's solar contribution is known. Preparing too little may leave demand unmet; preparing too much may waste capacity. We test whether a next-day solar-radiation forecast changes that simulated trade-off.

## What we built

SunReserve is a Python research prototype with a Streamlit interface. A user edits seven consecutive days of radiation, 2 m temperature, and humidity. The app validates the inputs, runs a fresh trained Ridge prediction for the next date, compares it with persistence and a training-only monthly-mean baseline, and converts each forecast into a planned reserve under editable hypothetical facility assumptions. A separate panel shows frozen historical evaluation results. The app does not control equipment or report measured electricity production.

## What we added

We started from the brief's next-day solar forecast and added **fresh trained-model inference** from user-edited inputs, **reserve decision comparisons** across Ridge and two simple baselines, and a **separate 2025 evaluation** on an untouched holdout. We also frame the output for a hypothetical Bengaluru school facilities manager.

## How it works

**Data.** NASA POWER Daily gridded estimates for one Bengaluru point (12.9716° N, 77.5946° E), 2016-2024: 3,288 calendar days with no missing source values. Chronological targets include 2,185 training rows through 2021, 365 validation rows in 2022, and 731 test rows in 2023-2024. A separate 2025 holdout contains 365 dates. These are not rooftop measurements.

**Model.** Scikit-learn Ridge with median imputation and scaling fitted on training data only; validation selected the model. The baselines are yesterday's reading and training-only monthly means. The 2025 holdout was not used for fitting, tuning, or selection.

## Results

On the 365-date 2025 holdout, radiation MAE was **0.539 kWh/m²/day** for Ridge, **0.589** for persistence, and **0.712** for the seasonal mean. In the primary hypothetical reserve replay (15 kWh/day demand, 15 kWh capacity, 20 m² panels, 20% conversion, 0.80 derating), mean simulated shortfall was **0.608 kWh/day** for Ridge, above persistence at **0.438** and below the seasonal mean at **0.882**. The equal-weight shortfall-plus-excess score was 0.787, 0.871, and 0.928 kWh/day, respectively; it is an illustrative comparison, not a validated cost or reliability target. In the 124-date low-solar slice, Ridge MAE was 0.724 versus 0.722 for persistence, so Ridge did not win there. Reserve values are simulated, not measured outcomes.

## What we'd do next

Verify that forecast inputs are available at the time a decision would be made. With a school partner and measured site data, compare against actual generation and demand and agree on a reserve criterion before any field trial.

## Sources and credits

Dataset: [NASA POWER Daily API](https://power.larc.nasa.gov/docs/services/api/temporal/daily/) and [data provenance](data_sources.md). Model and libraries: [scikit-learn Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html), [pandas](https://pandas.pydata.org/docs/), [NumPy](https://numpy.org/doc/), and [Streamlit](https://docs.streamlit.io/). Code: [public GitHub repository](https://github.com/icecold009/sunreserve/tree/feature/veritas-sunreserve). No pretrained model or tutorial code was used. Luna (OpenAI) implemented the project and visuals; [Codex](https://openai.com/codex/) planned, coordinated, reviewed, and validated it; [Jev by TypeSafe AI](https://typesafe.ai/) provided advisory reviews. These tools did not supply the data or validate real-world performance.

