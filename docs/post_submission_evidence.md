# Post-submission evidence inventory

**Submission receipt: confirmed. Submitted snapshot: unresolved.** The organizer receipt confirms that the entry was received, but the exact submitted bytes/version are not bound to a verified commit in this checkout.

**Hosted delivery:** PR #1 was merged to `main` at `2b746109f090c6f520e025c1ceb5a7083b36e8d3`. Hosted branch `codex/sunreserve-stage1-demo` at `3c0b56533a582962a135e4f274148cdaac52eda1` had successful CI run `44093878149`. This confirms the recorded branch/check state; it does not establish which snapshot the event platform received or evaluated.

**Local worktree at the 2026-11-19 pre-integration checkpoint:** branch `codex/sunreserve-post-submission-p0` was at `3c0b56533a582962a135e4f274148cdaac52eda1` and contained uncommitted, unpushed changes with no candidate PR. The cited hosted CI result did not cover that working tree.

## Scope and method

The inventory below was captured 2026-11-18 (Asia/Calcutta) from branch `codex/sunreserve-post-submission-p0`, HEAD `3c0b56533a582962a135e4f274148cdaac52eda1`. The baseline is commit `2b746109f090c6f520e025c1ceb5a7083b36e8d3` (`main`). The receipt, PR, and hosted CI status above was reconciled on 2026-11-19; the inventory remains a byte comparison against the stated baseline.

The inventory covers all 32 files tracked in the current checkout under `data/`, `models/`, and `artifacts/`. SHA-256 was computed over each file's working-tree bytes. Each file was also read as a blob from the baseline commit; the working bytes and baseline bytes were compared directly. **All 32 files match the baseline byte-for-byte.** The hash shown for each path is therefore both its working-byte SHA-256 and its baseline-blob SHA-256.

## SHA-256 inventory

| Tracked path | SHA-256 | Baseline comparison |
| --- | --- | --- |
| `artifacts/.gitkeep` | `01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b` | Match |
| `artifacts/runs/frozen-test-2023-24/assumptions.md` | `fa1cea1a8ed42f9ea0c3129ba33e14698cb58500bac26f144c04132830867688` | Match |
| `artifacts/runs/frozen-test-2023-24/data_quality.json` | `e63b8b7ab16cc312723c0470c8ee78f686c3059a86dbfbbde4469ceb0c60c199` | Match |
| `artifacts/runs/frozen-test-2023-24/decision_metrics.csv` | `f674ddccdb0336dd6a1147de5cc800395eee82d8c11a25a8e07f224dd63d2f28` | Match |
| `artifacts/runs/frozen-test-2023-24/decision_predictions.csv` | `660e35c7dd26d3ef890638a8c352f4a2276940d9e3468f89f9ba4467e3e193a3` | Match |
| `artifacts/runs/frozen-test-2023-24/forecast_metrics.json` | `b2756239b54dbbf02384229b4c383557e39b65f053368e6fc34a90e1c9cb0c79` | Match |
| `artifacts/runs/frozen-test-2023-24/predictions.csv` | `215f076fae47f558ad625c918cb72d4057e6d815f8244229e88232ab4acd079b` | Match |
| `artifacts/runs/frozen-test-2023-24/report.md` | `a3f1d716425e34700a7342d3c9964c58f072c0c4807162fb387781d54935d617` | Match |
| `artifacts/runs/frozen-test-2023-24/run_manifest.json` | `f784601d5b7ba7e2f4e870f7969d274f41ba66f158e0afcbbf5aabd8d4bd018e` | Match |
| `artifacts/runs/frozen-test-2023-24/selection_manifest.json` | `d4e4d2f86ba3ac795d5e33adb313a0bb8fc64b438aaafe4a86d6af4670e92653` | Match |
| `artifacts/runs/frozen-test-2025/data_quality.json` | `2dbeae8b915411675625e6960457a39d3dc085c9b2cc4a84e5f1e8be67d1961c` | Match |
| `artifacts/runs/frozen-test-2025/decision_metrics.csv` | `9b7565b6890f69378be8c9137387e26a1974691eee7b61fd9bd8e789840badf3` | Match |
| `artifacts/runs/frozen-test-2025/decision_predictions.csv` | `6e0eade4560fb8b039e6839f02ceab1904beae0234822cf09f6ae9d7948441c3` | Match |
| `artifacts/runs/frozen-test-2025/forecast_metrics.json` | `2822a00741a5a97c95020bc5e5299208a4f6b8e663b987e840d7a98d5258448c` | Match |
| `artifacts/runs/frozen-test-2025/predictions.csv` | `8f1881a57cd1fcf639435b187580a5162851a3a37040541c297a240a85c9c4af` | Match |
| `artifacts/runs/frozen-test-2025/report.md` | `c09128555f824147699334c7c19210290d92e15f12ff64c31689fbb4df30cbaa` | Match |
| `artifacts/runs/frozen-test-2025/run_manifest.json` | `f75b0077ca64e621e0a7097f95b56d362ede584d3cc99acae48c1b96fc59623c` | Match |
| `artifacts/runs/frozen-test-2025/selection_manifest.json` | `9cd60c7ab8b4e41078c1bf44d6698178334e6b9b6dba3d01a88f4e9a9d097fff` | Match |
| `artifacts/validation/decision_metrics.csv` | `11a165db4cae04efdddcacc6c30f203cba26613df5f90c5e8cffb1a4db080743` | Match |
| `artifacts/validation/decision_predictions.csv` | `1cf64b9d25339f8830c7ce16942bbf957f5400b0866ac3d7cce8c0e505b6f04f` | Match |
| `artifacts/validation/selection_manifest.json` | `9cd60c7ab8b4e41078c1bf44d6698178334e6b9b6dba3d01a88f4e9a9d097fff` | Match |
| `artifacts/validation/validation_metrics.json` | `8668d40066e88002207b24d4c43a118dd17d048eaa5649598ec05c6b91513e20` | Match |
| `artifacts/validation/validation_predictions.csv` | `767724222a93dedb2f93cfd812ce1af61816cfca004b2fd20ec3c82e01ac1b35` | Match |
| `data/processed/.gitkeep` | `01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b` | Match |
| `data/processed/daily.csv` | `b68d96ba070429161be7139d6a12e5f34004c5cc7778bd23bb820d51e555bb64` | Match |
| `data/processed/quality_report.json` | `e63b8b7ab16cc312723c0470c8ee78f686c3059a86dbfbbde4469ceb0c60c199` | Match |
| `data/raw/.gitkeep` | `01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b` | Match |
| `data/raw/power_daily_sunreserve.json` | `925dde6e86766df7f0f65c91d77947e7306068892fd02ddec0dc5b891d28c908` | Match |
| `data/raw/power_daily_sunreserve.meta.json` | `dbd33eb5d7c5b58b060f81a0588604c8112823235bc64cd1b28e8aa070fc7141` | Match |
| `data/raw/power_daily_sunreserve_2025_holdout.json` | `c679c980786c4dc26a69153513a76eeba116d2148cbc728390278a204c10af41` | Match |
| `data/raw/power_daily_sunreserve_2025_holdout.meta.json` | `4c5545975aede101c3fbe9e5d3b19ee20c692505d564cd72f86196549040bf08` | Match |
| `models/ridge_model.json` | `40a11e5119b7838d10a4a7ef7cc5bbf564998759b3954d4b865fdfe3f392b61f` | Match |

## Boundaries

- This is a local working-tree and commit-blob comparison. It does not prove which bytes were submitted, received, or evaluated by an event platform.
- No test suite or artifact-generation command was run for this inventory.
- At the 2026-11-18 inventory capture, this evidence inventory was the only change in that
  task; no data, model, or evaluation artifact was modified then.

## Current P4 forecast-time evidence status (2026-11-21)

The bounded comparison schedule remains the one approved in the forecast-time
protocol: an 18:00 IST issue cutoff and the following IMD meteorological-day
target, represented as the half-open UTC interval `[03:00 on D+1, 03:00 on D+2)`.
This is a comparison contract only. No operating schedule, installation site,
target-site outcome source, or outcome quality procedure has been selected.

Six saved NOAA NCEI Grid 4 NCSS point CSV responses for the 2020-04-12 06 UTC run
cover the target after the exact nested-window reconstruction. The checker reports
8.154 kWh/m² from those six target records. The separate +063 CSV is recorded as
a supplemental structural sample and checked for its saved CSV hash, row identity,
and metadata consistency; its +060 to +063 interval falls outside the target and
is excluded from the integration. These hashes identify CSV response bytes, not
raw GRIB. No raw GRIB or complete XML response body was captured.

Historical first-publication and issue-time availability remain **UNVERIFIED**;
exact client receipt timestamps are unavailable for five of the six target CSVs.
The archived CSV retrievals do not demonstrate publication by the 2020 issue
cutoff, and the model-grid sample does not establish any observed site outcome.
Operational evaluation is **NOT PERMITTED** on this evidence. The historical
inventory and its byte-comparison statements above retain their original scope.
