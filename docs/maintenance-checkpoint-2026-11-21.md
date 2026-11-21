# Maintenance checkpoint — 2026-11-21

This note captures the state reviewed on 2026-11-21. Hosted settings and PR
statuses can change after this checkpoint.

## P4 evidence handoff

The local P4 evidence branch `codex/sunreserve-p4-forecast-time-validity` was at
`b1b9d5d`; its full suite had **306 passed, 4 skipped**. CI run `44907836922`
succeeded on that exact commit with all six jobs passing. PR #18 is open and ready
for review (not draft, unmerged); its body was updated, and no reviewer was
requested or named. The archived NOAA historical route remains unqualified for
issue-time-valid evaluation because historical publication and receipt evidence
are missing.

Research direction: use prospective GFS forecast-vintage capture. NCEI describes
the recent 0.25° GFS forecast stream as retained in a trailing 30-day window, with
four cycles per day (00, 06, 12, and 18 UTC) and three-hourly output through
+384 hours ([NCEI GFS product page](https://www.ncei.noaa.gov/products/weather-climate-models/global-forecast)).
No prospective data has been captured in this checkpoint. A capture should retain
raw bytes and checksums, exact requests, response metadata, and client receipt
timestamps. The Bengaluru comparison site is selected; evaluation remains blocked
until a target-aligned observed-radiation source and quality-control procedure are
selected. IMD's 2025
annual report states that it operates 47 Solar Radiation Stations; that count does
not establish data access or the availability of a Bengaluru station
([IMD Annual Report 2025](https://metnet.imd.gov.in/docs/imdnews/ANNUAL_REPORT2025English.pdf)).

## Dependabot PR triage

All 15 Dependabot PRs #3–#17 remain open and none is merged as of this review.
None was considered merge-ready; they await explicit maintainer disposition.

| PR | Update | Triage |
| --- | --- | --- |
| [#3](https://github.com/icecold009/sunreserve/pull/3), [#4](https://github.com/icecold009/sunreserve/pull/4) | scikit-learn 1.4.0 → 1.9.1 | Duplicate version updates generated for overlapping requirements locations. |
| [#5](https://github.com/icecold009/sunreserve/pull/5) | python-multipart 0.0.33 → 0.0.34 | Snapshot-only change; regenerate and review the captured environment before considering it. |
| [#6](https://github.com/icecold009/sunreserve/pull/6), [#16](https://github.com/icecold009/sunreserve/pull/16) | Streamlit 1.49.0 → 1.67.0 | Duplicate version updates from overlapping requirements locations. |
| [#7](https://github.com/icecold009/sunreserve/pull/7), [#14](https://github.com/icecold009/sunreserve/pull/14) | Ruff 0.5.0 → 0.17.5 | Duplicate version updates from overlapping requirements locations. |
| [#8](https://github.com/icecold009/sunreserve/pull/8) | Pillow 10.1.0 → 13.0.0 | Incompatible with the Streamlit 1.49 requirement `Pillow<12`. |
| [#9](https://github.com/icecold009/sunreserve/pull/9) | numpy 2.5.4 → 2.5.5 | Snapshot-only change; regenerate and review the captured environment. |
| [#10](https://github.com/icecold009/sunreserve/pull/10) | pandas 2.1.0 → 3.1.1 | Fails the Python 3.11 minimum-dependency install. |
| [#11](https://github.com/icecold009/sunreserve/pull/11) | numpy 1.26.0 → 2.5.5 | Fails the Python 3.11 minimum-dependency install. |
| [#12](https://github.com/icecold009/sunreserve/pull/12) | pytest 9.0.3 → 9.2.0 | Snapshot-only change; regenerate and review the captured environment. |
| [#13](https://github.com/icecold009/sunreserve/pull/13) | pytest 9.0.3 → 9.2.0 | A compatibility-minimum pin update alone does not change the lower bound declared in `pyproject.toml`. |
| [#15](https://github.com/icecold009/sunreserve/pull/15), [#17](https://github.com/icecold009/sunreserve/pull/17) | pip 26.2.0 → 26.3 | Duplicate snapshot-only changes; regenerate the environment before reconsideration. |

For the compatibility-minimum-only PRs [#3](https://github.com/icecold009/sunreserve/pull/3), [#4](https://github.com/icecold009/sunreserve/pull/4), [#6](https://github.com/icecold009/sunreserve/pull/6), [#7](https://github.com/icecold009/sunreserve/pull/7), [#13](https://github.com/icecold009/sunreserve/pull/13), [#14](https://github.com/icecold009/sunreserve/pull/14), and [#16](https://github.com/icecold009/sunreserve/pull/16), editing `requirements/compat-minimums.txt` alone does not raise the lower bounds declared in `pyproject.toml`; none is merge-ready without a deliberate bound change.

## Branch protection and CI

The connector's read of `GET /branches/main` returned `protected=false` and
`protection.enabled=false`; the repository rulesets listing was empty. The
dedicated protection endpoint returned HTTP 403, `Resource not accessible by
integration`, and the connector exposed no write operation for this setting.
Protection is therefore unapplied. The recommendation is to require a pull
request, one approval, and these six exact checks:

- `checks (3.11)`
- `checks (3.12)`
- `checks (3.13)`
- `minimum-dependencies`
- `streamlit-lifecycle`
- `windows-smoke`

The CI workflow's full test suite already includes the frozen-artifact tests. The
repeated focused invocations have been removed from the matrix and Windows jobs;
the distinct lifecycle, compatibility, lint, formatting, scanner, and Windows
health checks remain.

## Open decisions and follow-ups

- Bind the submitted snapshot to an exact artifact or commit.
- Decide the repository license.
- Complete browser and accessibility acceptance.
- Establish the exact file and check coverage of PR #2 review.
- Prospective GFS capture is a research direction, not an external-data capture
  already performed; complete site, observed-target, and quality-control choices
  before any evaluation.
