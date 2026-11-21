# Forecast-time-valid inputs and evaluation protocol

**Status:** P4 protocol draft; NOAA GFS has not passed the historical issue-time-valid source gate. The bounded Bengaluru comparison contract below was approved on 2026-11-21; operational suitability and source qualification remain open.

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
checked official product documentation and catalog metadata, inspected message
interval metadata for one forecast cycle, and captured all six NCSS CSV point
responses needed for the approved comparison target. A reproducible local check
demonstrates a complete daily aggregation of 8.154 kWh/m² for this sample,
including nested-window reconstruction at the target boundary. The separate +063
CSV is a structural sample outside that target; the checker validates it and
excludes it from the integration. These retrievals establish neither historical
publication by the issue cutoff nor an outcome at the requested site. Historical
issue-time availability remains **UNVERIFIED**, and operational evaluation is
**NOT PERMITTED** on this evidence. No raw GRIB forecast file was retrieved: saved
SHA-256 values identify CSV representations, not raw GRIB bytes. Structured
excerpts of source XML metadata are retained, but no full XML response body is
preserved. Exact client receipt timestamps were not retained for the five newly
added target CSV responses. No tape/archive order was placed; historical raw-byte
retrievability, full revision policy, and order or cloud-processing costs remain
unverified.

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

The issue cutoff and target interval are approved for this bounded comparison.
An operating schedule, deployment site, and reserve-cost objective remain open;
this protocol does not assume a school dispatch deadline or select a reserve
objective. The current Bengaluru demo and its UTC display are not evidence for an
operating schedule.

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
| NOAA/NCEP GFS 0.5° archive | NCEI documents four daily UTC cycles and 3-hourly forecast outputs through +192h in the long-term 0.5° archive, with an approximately two-year online window. DSWRF is a downward shortwave flux in W/m². Metadata excerpts for one six-record +021 to +045 sample identify mixed 3- and 6-hour averages; its saved point responses pass daily aggregation at 8.154 kWh/m². | **No-go for historical issue-time-valid evaluation on current evidence.** The bounded sample establishes aggregation for that cycle and cell only. It does not establish original publication/availability, full archive coverage, revision history, or raw GRIB identity. NCEP's current production page gives 30-day mean production timing, not a historical per-run publication log. Decode metadata for each selected record; do not infer its averaging interval from file cadence. See the bounded screen below. |

### Bounded NOAA GFS access and sample-vintage screen

This screen was updated on 2026-11-21 using official product documentation,
catalog metadata, and a bounded point-sample access attempt. The user approved the
following comparison contract for research qualification. It does not establish an
operating schedule, site authorization, or source qualification:

| Field | Approved comparison value | Status |
| --- | --- | --- |
| Site and timezone | Existing demo point, 12.9716° N, 77.5946° E; `Asia/Kolkata` | Approved for this bounded comparison only; not a confirmed installation or study site. |
| Issue cutoff | 18:00 IST (12:30 UTC) on day D | Approved comparison cutoff; no operator deadline has been supplied. |
| Target interval | Next IMD meteorological day, 08:30 IST on D+1 through 08:30 IST on D+2, equivalent to `[03:00 UTC on D+1, 03:00 UTC on D+2)` | Approved comparison target. It is not a midnight-to-midnight local civil day. The interval spans 24 hours; matching three-hour forecast cadence alone does not establish valid three-hour radiation intervals. |
| Target quantity | Daily integrated downward shortwave radiation at the model grid cell for the approved point, expressed as kWh/m² | Approved comparison quantity; radiation, not electricity. The grid-cell selection/interpolation rule, observed outcome source, and QC procedure remain unselected. |
| Vintage rule | Latest 00/06/12/18 UTC GFS run whose required fields are proven disseminated and retrievable by the 12:30 UTC cutoff | Historical publication/availability evidence for a qualifying vintage was not found. |

**Archive and sample access record.** NCEI documents Grid 004 as the 0.5° GFS
forecast archive, with 3-hourly leads through +192 and only about two years online;
older records may require an archive order. The previously cited 2022 catalog
record named `gfs_3_20221231_0000_006.grb2` is identified as Grid 3 in its catalog
metadata and must not be used as evidence for the 0.5° product.

A source-side NCSS metadata description for the historical 0.5° file
`gfs_4_20200412_1800_063.grb2` reports forecast reference time
2020-04-12 18:00 UTC, a contiguous valid interval from lead +060 to +063
(2020-04-15 06:00–09:00 UTC), and surface Downward Short-Wave Radiation Flux as a
3-hour average in W/m². Its GRIB metadata identifies parameter 0/4/192, surface
level, and statistical process Average. A separate +384 file describes a
378–384-hour interval and a 6-hour average, demonstrating that interval length
depends on the message and lead; file cadence cannot substitute for interval
metadata.

A target-window lead was also retrieved from the plausible 06 UTC cycle on day D:
`gfs_4_20200412_0600_024.grb2`. Its provider-side dataset description reports
reference time 2020-04-12 06:00 UTC and the surface DSWRF field (parameter
0/4/192, W/m², statistical process Average) on a 6-hour interval from lead +018
to +024, timestamped at +021. The approved target begins at +021, so this mean
alone cannot be multiplied by only the three in-target hours. A separate +021
file supplies a 3-hour mean over +018 to +021 for the same run, grid cell,
parameter, surface level, units, and statistical process. The 3-hour interval is
nested in the 6-hour interval and shares its start. Subtracting the shorter
interval's integrated energy from the 6-hour integrated energy reconstructs
exactly the remaining +021 to +024 energy. The subsequent +030, +036, +042, and
+045 records provide contiguous 6-, 6-, 6-, and 3-hour intervals to the target
end. Source metadata excerpts, request URLs, saved CSVs, and checksums are in the
[bounded sample evidence manifest](evidence/noaa_gfs_grid4_20200412_0600_target_evidence.json).
See the [Grid 4 +021 metadata](https://www.ncei.noaa.gov/thredds/ncss/grid/model-gfs-004-files-old/202004/20200412/gfs_4_20200412_0600_021.grb2/dataset.xml)
and [Grid 4 +024 metadata](https://www.ncei.noaa.gov/thredds/ncss/grid/model-gfs-004-files-old/202004/20200412/gfs_4_20200412_0600_024.grb2/dataset.xml).

An NCSS grid-as-point request at the approved coordinates returned a single CSV
record for the +063 sample: 1040.0 W/m² at grid point 13.000° N, 77.500° E. The
dataset interval is 06:00–09:00 UTC; the CSV timestamp is 07:30 UTC, its midpoint.
This is a structural access sample at lead +060 to +063, not one of the approved
contract's +021 to +045 target intervals and not an as-issued forecast vintage.
Its CSV hash and metadata excerpt are recorded in the manifest as a supplemental
out-of-target record; the checker validates its identity and content, then excludes
it from the target integration. Successful retrieval of this sample or the six
target CSVs proves neither historical publication by the cutoff nor an observed
outcome at the requested site.
The returned CSV is preserved as
[`noaa_gfs_grid4_20200412_1800_f063_point_sample.csv`](evidence/noaa_gfs_grid4_20200412_1800_f063_point_sample.csv)
(UTF-8, no BOM, LF line endings; 227 bytes). Its SHA-256 is
`57ebf6a13d5c14cbbeda8b659fb27aa5529b4a2ebb69c4445aaf6abcfa4a5af1`. The local
client observed the successful response by 2026-11-21 05:37:48 UTC. This checksum
identifies the preserved CSV representation, not the original GRIB file; no raw
GRIB bytes or GRIB checksum were obtained. A bounded grid-region request and a
2019 +024 target-lead metadata request returned “FileNotFound”; a 2022 Grid 4
analysis-set request returned an S3 403 response. Catalog “Last Modified” times
are catalog/storage metadata, not proof of first dissemination to forecasters.
For the +024 target-window sample, the point response returned 527.0 W/m² at the
interval midpoint, 2020-04-13 03:00 UTC; it is preserved as
[`noaa_gfs_grid4_20200412_0600_f024_target_point_sample.csv`](evidence/noaa_gfs_grid4_20200412_0600_f024_target_point_sample.csv).
The client observed that response by 2026-11-21 05:46:02 UTC. Its checksum is
`31ee618b9d13104c8bf7cce711c2ea0f7229ddea12875066d711f53d8342ea52`. Both
checksums identify saved CSV responses, not the raw source GRIB bytes or first
provider publication.
See the [NCEI GFS archive description](https://www.ncei.noaa.gov/products/weather-climate-models/global-forecast),
[Grid 3 sample catalog record](https://www.ncei.noaa.gov/thredds/catalog/model-gfs-004-files/202212/20221231/catalog.html?dataset=gfs-g4-files%2F202212%2F20221231%2Fgfs_3_20221231_0000_006.grb2),
[Grid 4 +063 metadata](https://www.ncei.noaa.gov/thredds/ncss/grid/model-gfs-004-files-old/202004/20200412/gfs_4_20200412_1800_063.grb2/dataset.html),
[Grid 4 +384 metadata](https://www.ncei.noaa.gov/thredds/ncss/grid/model-gfs-004-files-old/201908/20190831/gfs_4_20190831_1800_384.grb2/dataset.html),
[+063 point CSV request](https://www.ncei.noaa.gov/thredds/ncss/grid/model-gfs-004-files-old/202004/20200412/gfs_4_20200412_1800_063.grb2?var=Downward_Short-Wave_Radiation_Flux_surface_3_Hour_Average&latitude=12.9716&longitude=77.5946&time=2020-04-15T09%3A00%3A00Z&accept=csv),
[+024 target-lead point CSV request](https://www.ncei.noaa.gov/thredds/ncss/grid/model-gfs-004-files-old/202004/20200412/gfs_4_20200412_0600_024.grb2?var=Downward_Short-Wave_Radiation_Flux_surface_6_Hour_Average&latitude=12.9716&longitude=77.5946&time=2020-04-13T06%3A00%3A00Z&accept=csv),
and [NCEI's migration notice](https://www.ncei.noaa.gov/news/cloud-migration).

**Forecast content and interval reconstruction.** The current NCEP 0.5° product
inventory describes a surface DSWRF product as a 0–3-hour average in W/m². This
does not prove the interval semantics of historical Grid 4 sample messages. Before
integrating any daily quantity, decode every selected GRIB message and record its
parameter and surface level, units, statistical-processing type, forecast
reference time, valid time, and explicit interval start/end (including step-range
and time-range metadata). Accept the 24-hour target only if the decoded records
show interval means that form a complete, contiguous, nonoverlapping partition of
`[03:00 UTC on D+1, 03:00 UTC on D+2)`. For such means in W/m², compute energy as
`sum(mean_flux × interval_seconds) / 3,600,000` kWh/m². Do not infer interval
duration from three-hour file spacing. If a mean overlaps the target boundary,
allow reconstruction only when compatible nested averages share the same exact
start or end, and their energy difference yields the needed nonoverlapping
segment. Verify the run, cell, parameter, level, units, statistical process, and
explicit bounds for both records. Otherwise, reject the integration. The sample
checker at [`scripts/check_gfs_target_aggregation.py`](../scripts/check_gfs_target_aggregation.py)
validates these conditions for this one response set and calculates 8.154 kWh/m²
for the approved target. This is a model-grid radiation estimate, not measured
site electricity. See the [current GFS 0.5° product
inventory](https://www.nco.ncep.noaa.gov/pmb/products/gfs/gfs.t12z.pgrb2full.0p50.f003.shtml),
[GRIB shortwave units](https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_table4-2-0-4.shtml),
and the [IMD forecasting SOP](https://mausam.imd.gov.in/imd_latest/contents/pdf/forecasting_sop.pdf)
for its meteorological-day definition.

NCEP's live production-status page reports 30-day average job times. For the
approved comparison cutoff, the latest plausible cycle is 06 UTC on D. The target
then spans forecast leads +021 to +045. A hypothetical complete 3-hour partition
would end at +024, +027, +030, +033, +036, +039, +042, and +045. The retrieved
sample instead has a nested +018-to-+021 3-hour mean and +018-to-+024 6-hour mean,
then 6-hour intervals through +042 and a final 3-hour interval. The checker uses
those decoded interval bounds rather than assuming the hypothetical cadence. The
06 UTC 48-hour product
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
Catalog entries and six NCSS dataset descriptions and bounded CSV point responses
were inspected for one 06 UTC cycle and one returned grid cell. The saved records
cover the complete +021 to +045 target after the nested-window reconstruction.
The executable check passes daily aggregation and computes 8.154 kWh/m² from the
sampled means. This does not qualify the archive for historical operational
evaluation. No raw GRIB checksum, immutable model-version or revision history, or
historical per-vintage publication/receipt evidence was established. Exact client
receipt clocks were not retained for the five new CSV files, and the structured
metadata excerpts are not full raw XML response bodies.
Prospective use would need to capture the raw bytes, provider/run metadata, first
retrieval and publication evidence, version notices, and a checksum at receipt.

**Decision: daily aggregation passes for this one retrieved sample; historical
source qualification for operationally framed evaluation still fails.** NCEI
documents a long-term 0.5° forecast archive. The +021/+024 nested means support a
mathematically exact +021-to-+024 energy difference because the decoded intervals
share their start and compatible DSWRF metadata. The +030 through +045 records
complete the target with contiguous intervals. The checker returns an integrated
value of 8.154 kWh/m² for the selected grid point and cycle. The earlier +063
sample remains an out-of-window three-hour field, and the separate +384 metadata
record shows a six-hour interval, confirming that message intervals vary with
lead. No raw GRIB checksum or historical per-vintage publication/receipt
evidence was found; exact receipt timestamps for the five added CSVs were not
retained.
Keep historical comparisons explicitly retrospective unless archived publication
evidence is obtained. A prospective archive is the fallback: retain the raw source
bytes at first receipt, receipt timestamp with timezone, source/run identifiers,
publication/availability evidence, checksums, decoded message metadata, missing
cycle log, and transformation identity. Accumulate enough verified history for a
predeclared chronological evaluation; do not backfill availability from files
downloaded later. The approved comparison interval is settled for this bounded
qualification task; target-site validation and the observed outcome remain open.

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
