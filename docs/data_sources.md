# Data source and provenance

## Selected source

SunReserve uses the single-point NASA POWER Daily API at
[`/api/temporal/daily/point`](https://power.larc.nasa.gov/docs/services/api/temporal/daily/).
The request is explicit about the Renewable Energy community (`community=RE`), the
Bengaluru reference coordinates (latitude 12.9716, longitude 77.5946), date interval
(2016-01-01 through 2024-12-31), JSON format, and UTC time standard. The daily API
otherwise defaults to Local Solar Time; the request explicitly sets
`time-standard=UTC`. NASA documents daily availability beginning in 1981.

| POWER parameter | Meaning | Expected RE daily unit |
| --- | --- | --- |
| `ALLSKY_SFC_SW_DWN` | All-sky surface shortwave downward irradiance | kWh/m²/day |
| `T2M` | Temperature at 2 meters | °C |
| `RH2M` | Relative humidity at 2 meters | % |

The daily API documentation lists these as available parameters. POWER API units are
community-specific; the Renewable Energy community uses kWh/m²/day for the solar
parameter. The ingestion code checks each unit string in the response metadata before
transforming values. It also checks requested coordinates, date bounds, and the time
standard.

POWER describes solar data as inferred from satellite observations with radiative-transfer
models; its meteorological data comes from NASA assimilation products. These are gridded
historical estimates, not measurements from a Bengaluru school, rooftop, or local weather
station. Historical observations may also have been published after the simulated forecast
issue date. The experiment is therefore an ex-post historical replay, not an operational
day-ahead forecast.

NASA POWER's [data FAQ latency section](https://power.larc.nasa.gov/docs/faqs/data/#latency)
reports approximate latency of 5–7 days for solar parameters and 2–3 days for meteorology.
Consequently, even the latest NASA POWER solar value is not suitable as a verified same-day
input to an operational next-day forecast. The interactive prediction mode accepts user
observations and runs fresh inference, but it does not connect to a live or verified
forecast-available feed.

## Terms and acknowledgement

The [NASA POWER data FAQ](https://power.larc.nasa.gov/docs/faqs/data/) describes source
data and API-specific units. The [NASA POWER record in the AWS Open Data Registry](https://registry.opendata.aws/nasa-power/)
states that there are no restrictions on use, access, or download and asks users to cite the
NASA POWER Project. The [POWER referencing guide](https://power.larc.nasa.gov/docs/referencing/)
also requests notification when POWER data is transmitted to other researchers. The project
write-up acknowledges NASA Langley Research Center's POWER Project and links to the API
documentation and the applicable data-use notes. The notification language is recorded as
the guide's request, not stated as a legal requirement; no email or notification was sent.

The NASA POWER data terms do not provide a license for this project's code or documentation.
No root `LICENSE`, `COPYING`, or `NOTICE` file was found, and no project reuse license was
added. This implementation therefore grants no license to reuse the project itself; that
remains an explicit delivery limitation.

## Local files

An explicit `python -m sunreserve fetch` command saves the original response bytes under
`data/raw/` and a JSON provenance sidecar beside it. The sidecar records the complete
request URL and parameters, UTC retrieval time, requested/reported location, units, fill
marker, and SHA-256 checksum. A matching checksum and request are required to reuse the
cache. Refresh downloads are parsed and validated before either cached file is replaced.
Normal imports, preparation, validation, and app startup do not make network requests.

`python -m sunreserve prepare` uses only this cache. It writes a full daily calendar to
`data/processed/daily.csv`, preserving gaps and fill values as missing rather than
interpolating, plus `data/processed/quality_report.json` with per-variable and per-year
missing counts, exact missing dates, and complete-case exclusions.

Validation manifests bind the full prepared CSV bytes and the exact persisted CSV prefix
through `validation_end`. New manifests mark prefix digests as `prepared-csv-prefix-v1` and
hash the header plus exactly the configured development-row count, independent of pandas CSV
parsing or formatting. Markerless legacy manifests remain usable only when the full prepared
CSV bytes match the checksums recorded by both the manifest and quality report.

## Supervised row timing and coverage

Each issue date `t` targets the radiation observation on the next calendar date
`t + 1`; the builder reindexes dates before shifting, so a missing day cannot be skipped.
Features use radiation on `t`, radiation lags for the three prior calendar days, a
right-aligned trailing seven-day mean through `t`, temperature and humidity on `t`, and
known target-month seasonality. The row records its issue date, target date, earliest
seven-day feature window date, and latest feature availability date. POWER release
timestamps were not verified, so this availability is by observation date only.

Splits are assigned by target date: 2016–2021 training, 2022 validation, and 2023–2024
test. The source interval begins on 2016-01-01, so it cannot create a 2016-01-01 target row
without 2015-12-31 issue-date inputs. The seven-day feature window excludes targets
2016-01-02 through 2016-01-07. The inspected source yields 3,281 eligible rows: 2,185
training, 365 validation, and 731 test. The validation-only run selected Ridge by MAE:
0.6249 kWh/m²/day on 365 validation targets, versus 0.6767 for persistence and 0.8365 for
the seasonal mean. These are validation results only; out-of-sample results are reported
separately below. The cached validation manifest contains the full metrics and provenance.

## Retrieval status

The cached response header identifies NASA POWER Daily API version `v2.10.3`; it was
retrieved on 2026-11-14. This identifies the saved response and must not be conflated with
the current documentation status. NASA POWER's current documentation describes POWER Data
v10 and was checked separately on 2026-11-19; the API migration notice says the endpoints
continue to work as expected. The cached response was not regenerated as part of that
documentation check or this documentation update.

When citing the cached data, identify the NASA POWER Project, response API version and
retrieval date, requested date interval, reference coordinates, Renewable Energy community,
UTC time standard, parameter identifiers, and units. Link the [daily API documentation](https://power.larc.nasa.gov/docs/services/api/temporal/daily/)
and [POWER referencing guide](https://power.larc.nasa.gov/docs/referencing/); include the
raw-response SHA-256 when referring to these exact local bytes. Cite the [current POWER
Data v10 documentation](https://power.larc.nasa.gov/docs/) separately from the version and
provenance of the cached response. No notification was sent in this documentation batch.

The exact raw response is cached locally with SHA-256
`925dde6e86766df7f0f65c91d77947e7306068892fd02ddec0dc5b891d28c908`. Preparation found
3,288 consecutive calendar days from 2016-01-01 through 2024-12-31, with no absent dates or
fill values in the three requested variables. This is source coverage, not a claim of
on-site measurement quality or forecast availability. The offline fixtures separately
exercise missing values, missing days, malformed payloads, and cache failures.

## Separate 2025 holdout snapshot

The untouched evaluation period uses separate source files
`data/raw/power_daily_sunreserve_2025_holdout.json` and its
`power_daily_sunreserve_2025_holdout.meta.json` provenance sidecar. This is the same
NASA POWER Daily API, location, three parameters, and UTC time standard, requested for
2025-01-01 through 2025-12-31. The raw response SHA-256 is
`c679c980786c4dc26a69153513a76eeba116d2148cbc728390278a204c10af41`. Validation found 365
calendar days and no missing source values. The one-time run under
`artifacts/runs/frozen-test-2025/` had 365 eligible common targets and zero exclusions;
2023–2024 target rows scored in that run: zero. Earlier observations were used only to
construct seven-day lag/window features for 2025 issue dates. See its data-quality and run
manifests for request and artifact hashes.

This is still a historical replay. Solar radiation is the NASA POWER radiation estimate in
`kWh/m²/day`; it is not measured electricity generation. The radiation-to-electricity
conversion and reserve assumptions are hypothetical, and source-date history does not prove
that those observations were available on their modeled issue dates. In particular, NASA
POWER's cited 5–7 day solar-parameter latency rules it out as a verified live next-day input.
