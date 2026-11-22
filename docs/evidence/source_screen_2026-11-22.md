# Bengaluru radiation-outcome source screen — 2026-11-22

This is a bounded source-qualification note, not a data acquisition. The source
screen records the following findings. No requests were sent to IMD, NIWE, or
NASA POWER. NOAA was queried only through the read-only NOMADS GETs documented in
the capture receipt; no provider was otherwise contacted.

- IMD Greater Bengaluru AWS documentation lists temperature, relative humidity,
  rainfall, and wind, but not irradiance.
- IMD Bengaluru ARU online solar-radiation values are 1991–2020 normals, not
  daily observed outcomes for the target interval.
- NIWE reports quality-controlled ground solar measurements. Its listed
  Karnataka stations are outside Bengaluru; raw time-series are sold and carry
  restrictive no-redistribution terms.
- NASA POWER remains a retrospective proxy. Its daily radiation product is a
  satellite/radiative-transfer estimate with the documented 5–7 day solar
  latency, not a Bengaluru site measurement or a qualified next-day outcome
  ([project source record](../data_sources.md)).

**Decision:** A separately measured radiation outcome aligned to the approved
target, together with its access terms and quality-control procedure, remains
unqualified. Site-specific and operational evaluation must wait for that evidence.
The listed-source descriptions do not establish that a measurement series is
accessible or available for Bengaluru.
