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

## NWIC Karnataka hourly radiation candidate — 2026-11-22

The [NWIC Karnataka dataset listing](https://nwdp.nwic.gov.in/dataset/solar-radiation-telemetry-hourly-karnataka-department)
advertises hourly telemetric solar-radiation resources for 1991–2020,
2021–2025, and 2026–2030. Its public description says records include a date,
station identifier/geographic hierarchy, and solar-radiation values. It does
not identify a Bengaluru station or expose, in the listing, the exact measured
channel and units, acquisition-time timezone/bin convention, averaging rule,
missingness/QC flags, or license for the current resource periods.

Read-only attempts to open the 2021–2025 and 2026–2030 resource previews and
the 2021–2025 download timed out on 2026-11-22. No CSV bytes or sample rows
were obtained, and no data producer was contacted. Metadata from another
state's resource or the separate 1991–2020 Karnataka resource is not treated as
evidence for these resources. NWIC says producers are responsible for validating
and documenting data quality, and its [disclaimer](https://nwdp.nwic.gov.in/footer/disclaimer)
disclaims accuracy or completeness.

**Decision: NO-GO as a target-aligned observed outcome.** Until the actual
resource establishes a Bengaluru-area station and coordinates, measured
quantity/units, timestamp and averaging semantics that cover the approved
08:30–08:30 IST interval, missingness/QC treatment, and resource-specific
reuse terms, do not ingest it as an outcome or substitute it for a measurement
source. The NWIC listing remains a candidate for a later qualification attempt.
