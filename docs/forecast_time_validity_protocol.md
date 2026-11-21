# Forecast-time-valid inputs and evaluation protocol

**Status:** P4 protocol draft; the NOAA GFS screen below is a no-go for a historical issue-time-valid backtest. Study configuration and source qualification remain open.

**Reviewed:** 2026-11-21.

**Scope:** define the evidence required before SunReserve may describe a forecast as
using inputs available at forecast issue time and define a chronological evaluation
for a future study.

## Current evidence boundary

The existing SunReserve runs remain historical replays. The 2016–2024 development
data and the separately frozen 2025 snapshot are unchanged. Their date-indexed
features do not establish when each value was released, and the 2025 evaluation has
already been run once under its frozen protocol. Do not regenerate, retune, or
reinterpret those artifacts as an issue-time-valid backtest.

NASA POWER reports approximate latency of 2–3 days for meteorological parameters
and 5–7 days for solar parameters. Its daily solar data are therefore not eligible
as verified next-day solar inputs. Current POWER observations may remain useful as
historical context or outcome labels, but their source-date timestamps do not prove
that they were available at a simulated issue time. See the official [POWER latency
FAQ](https://power.larc.nasa.gov/docs/faqs/data/) and the project’s
[data-source record](data_sources.md).

No source is qualified for an operational-validity claim. The NOAA screen below
checked official product/access documentation and a sample catalog entry only; no
forecast bytes were downloaded and no tape/archive order was placed. The historical
archive's sample-byte retrieval, full revision policy, and any order or cloud
processing costs remain unverified.

## Temporal contract

Before sourcing or scoring data, record the following in an approved study
configuration:

| Field | Required definition |
| --- | --- |
| Site and timezone | Coordinates and an IANA timezone for the site; retain UTC timestamps as the canonical machine representation. |
| Issue schedule | Exact local issue time or times and the latest time by which the forecast must be usable. Do not infer it from a dataset’s cycle time. |
| Target interval | Exact start and end instants, timezone, horizon, and aggregation rule. A local civil day and a meteorological day are different targets. |
| Target quantity | Measured radiation or electricity outcome, parameter, units, sensor or provider, and quality-control procedure. Do not treat radiation as generated electricity. |
| Input sources | Provider, product, version, variables, grid or station, resolution, units, terms, and archive access route. |
| Vintage policy | Which eligible forecast cycle is selected when multiple cycles exist, how revisions are handled, and how source availability is proven. |
| Training cutoff | At each simulated issue, the last time at which training data and labels may enter the fitted model. |
| Exclusions | Missing-value, late-publication, revision, spatial-coverage, and incomplete-target rules. Apply these rules before comparing models. |

The site issue schedule, target interval, and reserve-cost assumptions remain open
decisions. This protocol does not assume a school dispatch deadline or select a
reserve objective. The current Bengaluru demo and its UTC display are not evidence
for an operating schedule.

### Required clocks and provenance

Each forecast-input record must be traceable to an immutable source vintage and
include, where applicable:

- forecast issue time and forecast target valid start/end;
- provider run identifier and model initialization time;
- provider publication or dissemination time, and the evidence supporting it;
- time the forecast system could actually retrieve the value;
- retrieval timestamp, source/product version, parameter, location, units, and
  aggregation/time standard;
- revision policy, raw-response or file checksum, and transformation/version
  identifiers.

For observations used as predictors or training labels, record the observation
period and the first verified publication or availability time. Retrieval time
alone is not proof of historical availability: downloading a historical value
today does not show that it existed at the simulated issue time.

Define a record’s usable availability time as the latest relevant time among source
publication, delivery to the forecasting system, and any required processing step.
An input is eligible only when that time is at or before the predeclared issue
time and all transformations use only eligible records. Apply valid-time rules by
input role: forecast covariates must cover or otherwise align with the target
interval under the declared horizon and aggregation; lagged observations must lie
within their declared historical feature window, ending no later than the issue
cutoff, and need not overlap the target interval; static and calendar features
must follow their declared semantics and be knowable at issue time. Model
forecasts also require both an initialization time and proof that the selected
run was disseminated by issue time; initialization alone is insufficient.

Daily or other aggregated observations become available only after the aggregation
period ends and the resulting product is published. If source publication time or
a conservative documented upper bound cannot be established, mark availability
unknown and exclude the record from claims of operational forecast validity.

Training follows the same as-of rule. At every simulated origin, fit and select
using only training data whose labels and relevant versions were available by that
origin’s training cutoff. A final or revised historical dataset without archived
vintages may be used as an evaluation outcome if its target definition is sound,
but not as evidence that historical model training had access to that version.

## Source qualification gate

The following assessments are based on official documentation checked on
2026-11-19, except for the NOAA GFS screen checked on 2026-11-21. They are
screening results, not approval to use a source.

| Candidate | What the documentation establishes | Current decision |
| --- | --- | --- |
| NASA POWER Daily | POWER reports 5–7 day solar latency and 2–3 day meteorological latency. Its historical daily records are not an as-issued archive of next-day forecasts. | Reject as verified next-day solar predictors. Preserve existing data and frozen results; use future POWER values as labels or context only where their revision and label-timing treatment is explicit. |
| ERA5 | ECMWF describes ERA5 as reanalysis, with analyses and short forecasts initialized twice daily from analyses. Forecast initialization, step, and valid time must be interpreted separately. | At most a clearly labeled retrospective proxy/reforecast study. It does not by itself demonstrate an operational run that users actually received at the historical issue time. See [ERA5 data documentation](https://confluence.ecmwf.int/pages/viewpage.action?pageId=388500357) and [time/step semantics](https://confluence.ecmwf.int/pages/viewpage.action?pageId=85416081). |
| ECMWF Open Data | The current service offers real-time IFS/AIFS forecasts and retains the latest 12 runs, about 2–3 days. Its documentation points to separate archive access for full historical data; archive access can require a service agreement. | Possible forward data-capture source or recent-cycle check, not a demonstrated multi-year historical backtest archive. Verify variable coverage, historical access, cost, terms, and exact publication evidence before selection. See [Open Data](https://www.ecmwf.int/en/forecasts/datasets/open-data), [archive access](https://www.ecmwf.int/en/forecasts/access-forecasts/access-archive-datasets), and [ordering historical datasets](https://www.ecmwf.int/en/forecasts/accessing-forecasts/order-historical-datasets). |
| IMD forecast bulletins | The IMD standard operating procedure specifies issue and validity times and defines the meteorological day as 0830 IST to 0830 IST. | Timing semantics are useful, but a suitable archived Bengaluru point series, forecast variables, machine-readable vintages, availability evidence, and reuse terms have not been established. See the [IMD forecasting SOP](https://mausam.imd.gov.in/imd_latest/contents/pdf/forecasting_sop.pdf). |
| Site or nearby measurements | Needed to support site-specific outcome claims and to characterize local error. | No site meter or station dataset has been selected or verified. Confirm sampling, calibration, missingness, location, rights, and publication times before use. |
| NOAA/NCEP GFS 0.5° archive | Four daily UTC cycles, 3-hourly output through +192h in the long-term 0.5° archive; surface DSWRF is a downward shortwave flux field, with 0–3h average values in the 3-hour product. NCEI documents an approximately two-year online window, with 2006–present data available for tape order. | **No-go for historical issue-time-valid evaluation on current evidence.** The archive catalog identifies a run and forecast hour but does not give the sample’s original publication/availability time. NCEP’s current production page gives 30-day mean production timing, not a historical per-run publication log. The illustrative IMD meteorological-day target aligns with the 3-hour grid; a midnight-to-midnight local civil-day target would not. Keep as a prospective capture candidate only; see the bounded screen below. |

### Bounded NOAA GFS access and sample-vintage screen

This screen was completed on 2026-11-21 using official documentation and catalog
metadata. It did not download forecast data, place an archive order, or approve a
study configuration. The following values are a comparison example based on the
existing demo point, not an operating schedule or an approved study contract:

| Field | Provisional comparison value | Status |
| --- | --- | --- |
| Site and timezone | Existing demo point, 12.9716° N, 77.5946° E; `Asia/Kolkata` | Candidate only; site is not confirmed for a study. |
| Issue cutoff | 18:00 local time (12:30 UTC) on day D | Candidate only; no operator deadline has been supplied. |
| Target interval | Next IMD meteorological day, 08:30 IST on D+1 through 08:30 IST on D+2, equivalent to `[03:00 UTC on D+1, 03:00 UTC on D+2)` | Candidate only; not a local civil day or confirmed operating interval. This 24-hour interval aligns to complete 3-hour GFS bins. |
| Target quantity | Daily integrated downward shortwave radiation at the model grid cell, expressed as kWh/m² | Radiation, not electricity; the measured outcome source and QC procedure remain unselected. |
| Vintage rule | Latest 00/06/12/18 UTC GFS run whose required fields are proven disseminated and retrievable by cutoff | This rule cannot be reconstructed for historical sample dates from evidence found in this screen. |

**Archive and sample record.** NCEI documents the GFS 0.5° forecast archive from
2006 to present, approximately two years online, with 3-hourly outputs through
forecast hour +192; records beyond the online window may require a tape order.
The NCEI THREDDS catalog advertises a sample record for the 2022-12-31 00:00 UTC
run at forecast hour +006 (`gfs_3_20221231_0000_006.grb2`, catalog size
44,681,004 bytes). This establishes that a run/lead-time catalog entry exists,
not that its bytes were retrieved in this task or when the field first became
available to a forecaster. NCEI says its cloud migration runs through Spring
2027 and may delay archive and access services. See the [GFS archive description](https://www.ncei.noaa.gov/products/weather-climate-models/global-forecast),
the [sample catalog entry](https://www.ncei.noaa.gov/thredds/catalog/model-gfs-004-files/202212/20221231/catalog.html?dataset=gfs-g4-files%2F202212%2F20221231%2Fgfs_3_20221231_0000_006.grb2),
and [NCEI's migration notice](https://www.ncei.noaa.gov/news/cloud-migration).

**Forecast content and timing.** NCEP's product inventory lists surface DSWRF
for the 0.5° product as a 0–3-hour average in W/m². Under the illustrative IMD
meteorological-day target, eight complete 3-hour intervals span 03:00 UTC to
03:00 UTC the following day, so the integrated irradiation can be computed by
weighting each interval mean by three hours and converting Wh/m² to kWh/m². This
is a radiation estimate at the model grid, not measured site electricity. If the
study instead requires midnight-to-midnight Asia/Kolkata, its boundaries fall
at 18:30 UTC and bisect the GFS 3-hour bins; that target needs a finer qualified
source or a predeclared partial-bin treatment. The target choice remains open.
See the [GFS 0.5° product inventory](https://www.nco.ncep.noaa.gov/pmb/products/gfs/gfs.t12z.pgrb2full.0p50.f003.shtml),
[GRIB shortwave units](https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_table4-2-0-4.shtml),
and the [IMD forecasting SOP](https://mausam.imd.gov.in/imd_latest/contents/pdf/forecasting_sop.pdf)
for its meteorological-day definition.

NCEP's live production-status page reports 30-day average job times. For the
illustrative issue cutoff, the latest plausible cycle is 06 UTC on D. The target
then spans forecast leads +021 to +045; its eight DSWRF interval means end at
+024, +027, +030, +033, +036, +039, +042, and +045. The 06 UTC 48-hour product
group was averaging a 09:50:29 UTC finish, about 2 hours 40 minutes before the
12:30 UTC cutoff; the full +000-to-+384 forecast averaged an 11:09:45 UTC finish,
about 80 minutes before cutoff. The 12 UTC 48-hour product group was averaging a
15:51:27 UTC finish, after the illustrative cutoff. These are current 30-day
averages, not historical per-vintage release receipts or guarantees. They do not establish that those
fields were published before any simulated issue time. No per-vintage publication
log or first-seen receipt was located. See [NCEP production status](https://www.nco.ncep.noaa.gov/pmb/nwprod/prodstat/index.html),
which states that the times are 30-day running averages.

**Access, revisions, and decision.** NCEI lists HTTPS, THREDDS, and archive
ordering routes. NOAA says data available through NODD are free to access, have
no use restrictions or registration requirement, and have no data egress charge;
cloud compute services may still have their own charges. The NODD GFS feed is a
trailing 30-day window, while older data use NCEI's separate archive routes. No
source-specific estimate for tape/archive orders or cloud processing was checked.
See [NOAA's NODD access and cost FAQ](https://www.noaa.gov/big-data-project-frequently-asked-questions).
The archive catalog entry was discoverable, but a sample GRIB byte retrieval was
not verified. GFS changes over time, and the
catalog record does not establish an immutable model-version or revision
history. Prospective use would need to capture the raw bytes, provider/run
metadata, first retrieval and publication evidence, version notices, and a
checksum at receipt.

**Decision: fail the historical source-qualification gate for operationally
framed evaluation.** The run/lead metadata, global coverage, radiation field,
and approximate archive depth make GFS worth a prospective trial. The missing
historical publication evidence is decisive; sample-byte retrievability,
revision history, the site/outcome, and full access costs also remain unresolved.
The proposed IMD meteorological day makes the 3-hour aggregation arithmetically
feasible, but must be approved as the study target. Do not treat the catalogued
2022 sample as an as-issued vintage. A prospective archive is the fallback:
first approve the site, timezone, issue cutoff, target interval and quantity,
then retain every eligible cycle's raw file, receipt times, source notices,
checksums, missing-cycle log, and transformation identity. Accumulate enough
verified history for a predeclared chronological evaluation; do not backfill
its availability from files downloaded later.

A source can pass only if the study team can obtain the required historical
vintages, establish issue-time availability, define valid-time alignment, confirm
access terms and costs, and reproduce the same eligible records from preserved
bytes. An operational forecast archive and a reanalysis/reforecast dataset must
not be combined under one operational-validity claim.

## Evaluation design

1. **Freeze the question first.** Predeclare the site, issue schedule, target
   interval/horizon, outcome, forecast sources, model-selection rules, primary
   forecast metric, coverage policy, and exclusions before looking at the final
   evaluation period. For a daily radiation study, keep the radiation outcome and
   units explicit; do not silently substitute a daily average for an energy sum or
   a local-day target for a UTC-day target.
2. **Construct as-of examples.** For each issue time, select only the latest
   eligible vintage under the locked vintage policy. Store the exact input bytes,
   eligibility evidence, selected cycle, and target interval. Apply publication
   cutoffs to labels and training data as well as predictors.
3. **Use chronological origins.** Train on prior eligible observations, tune only
   on earlier validation origins, and advance through rolling or expanding origins
   without using later labels. Do not use random row splits for this time-series
   evaluation.
4. **Reserve a future holdout.** After all source, feature, model, and metric
   choices are fixed, bind one future evaluation period to immutable source,
   model, configuration, code, and environment identities. Do not reuse the
   already-scored 2025 holdout as a new untouched test.
5. **Compare on common targets.** Every candidate and baseline must score the same
   eligible target intervals. Report coverage, exclusions, missingness, and
   failures alongside results; do not improve a score by silently dropping harder
   dates for one method.
6. **Use timing-valid baselines.** Include a training-only seasonal climatology
   and persistence only when the last observation was verified available by the
   issue time. Other operational forecast baselines must use the same issue-time
   information budget. Existing historical baselines remain historical
   comparisons until their input timing is verified.
7. **Report forecast performance.** Report MAE, RMSE, bias, and paired skill
   differences against each timing-valid baseline on common targets. Also show
   counts and error summaries by predeclared season or lead where coverage permits.
   Preserve the per-issue forecasts and target joins so every metric can be
   recomputed. Do not call simulated reserve outcomes measured reliability.

Calibration of predictive intervals, reserve sizing under asymmetric shortage and
over-preparation costs, and any declared reserve objective belong to the subsequent
uncertainty/decision phase. P4 does not choose those costs or change the current
validation objective.

## Go/no-go and next evidence

Proceed to an operationally framed backtest only when all of these conditions are
met:

- a usable historical archive preserves the exact forecast vintages and their
  publication/availability evidence;
- site timezone, issue cutoff, target valid interval, and outcome are declared;
- predictor, label, and training cutoffs pass the as-of rules above;
- spatial/temporal resolution, missingness, source revisions, terms, and costs are
  documented;
- validation and future-holdout boundaries, baseline rules, metrics, and
  exclusions are locked before scoring the holdout.

If a required condition fails, stop the operational-validity claim. A separate
retrospective proxy study may proceed only with its data class, timing limitations,
and non-operational interpretation stated plainly. The first follow-up is a
source-access and sample-vintage audit; it is not a model rerun. Keep the existing
frozen artifacts byte-identical throughout this work.
