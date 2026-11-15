# Reserve simulation assumptions

SunReserve is a historical research replay. These calculations do not validate an
operational controller, rooftop system, school demand estimate, battery, or dispatch
policy. All demand, PV capacity, and reserve settings below are hypothetical experiment
assumptions; they are not measured site data.

## Solar-radiation-to-electricity proxy

The NASA POWER `ALLSKY_SFC_SW_DWN` value is daily solar radiation in
`kWh/m²/day`, not generated electricity. For an illustrative array, the simulator
converts radiation to a simplified daily electricity proxy:

```text
G_proxy (kWh/day) = radiation (kWh/m²/day)
                    × panel area (m²)
                    × conversion efficiency
                    × derating factor
```

The primary scenario uses 20 m², efficiency 0.20, and derating 0.80. This is not a
validated PV yield model: it does not model panel orientation, shading, temperature,
inverter behavior, hourly irradiance, curtailment, or system availability.

## Daily energy balance

For each target date, the selected forecast approach predicts radiation, which is
converted to predicted proxy generation `G_hat`. Negative predicted radiation is
clipped to zero before conversion. The simulator assumes the prepared reserve is
fully available that day:

```text
R = min(C, max(0, D - G_hat + b))
simulated shortfall = max(0, D - G_actual - R)
excess prepared reserve = max(0, R - max(0, D - G_actual))
capacity-limited shortfall = max(0, D - G_actual - C)
additional shortfall above capacity floor = max(0, simulated shortfall - capacity-limited shortfall)
```

Here `D` is assumed daily demand, `C` is the assumed reserve capacity, and `b` is
an assumed reserve buffer. Simulated shortfall includes both capacity limits and
forecast-dependent reserve preparation. Capacity-limited shortfall is reported
separately because even a perfectly sized forecast cannot cover demand beyond the
reserve capacity. Additional shortfall above that capacity floor is descriptive; it
is not a pure causal estimate of forecast error.
Excess prepared reserve is backup energy prepared beyond the realized residual need;
it is not surplus solar generation. The simulator has no inter-day state, state of
charge, charge/discharge power limits, efficiency loss, backup fuel use, prices,
outages, or grid behavior.

The always-prepare-backup reference prepares `min(D, C)` each day. It exposes the
reliability/over-preparation trade-off and does not use a forecast.

## Frozen development scenarios

The default primary hypothetical scenario is 20 m², 0.20 efficiency, 0.80 derating,
15 kWh daily demand, 15 kWh reserve capacity, and zero buffer. If the active
experiment configuration sets `reserve_buffer_kwh` to another nonnegative value,
that value is used and recorded for the primary scenario. The sensitivity grid is
unchanged: the fixed Cartesian grid of 10/15/20 kWh demand, 10/15/20 kWh reserve
capacity, and buffers of 0/10/20% of demand. The 27 sensitivity settings and primary
scenario are applied to persistence, seasonal-mean, and Ridge predictions on the same
validation target dates. The always-prepare-backup reference is included for each
setting.

Validation summaries include total/mean simulated shortfall, shortfall-day count/share,
total/mean excess prepared reserve, and capacity-limited shortfall. They include
additional shortfall above the capacity floor, and paired mean daily simulated
shortfall and excess differences versus persistence and seasonal
mean on the identical dates. These descriptive differences are not claimed savings,
cost reductions, emissions reductions, or causal effects. No favorable scenario is
selected using final-test results.

All reserve outputs are hypothetical simulated `kWh/day` quantities. They are
historical estimates using model-derived/satellite-derived daily radiation and do
not demonstrate deployment, real-time data availability, or reserve performance at
a real facility.
