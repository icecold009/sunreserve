# Demo acceptance checklist

This checklist separates observed UI behavior from checks that still need a direct browser observation. It describes the local research/demo prototype only; passing it does not establish site performance or operating readiness.

## Recorded browser checks — 2026-11-23

| Flow | Status | Evidence |
| --- | --- | --- |
| Save the blank CSV template | **Passed** | Downloaded the template and checked its header and seven blank observation rows. |
| Import a valid seven-row CSV | **Passed** | User confirmed import succeeded. Displayed rows matched the valid fixture, which is identical to the historical example. |
| Reject invalid CSV input | **Passed** | User confirmed the invalid CSV rejection was visible. |
| Reset after import and reimport valid CSV | **Passed** | User confirmed reset and reimport succeeded. |
| Run a fresh prediction and save JSON whose values match the displayed result | **Passed with source limitation** | Saved JSON matched the displayed fresh result: target 2025-01-01; Ridge 5.390630517794998, persistence 5.7787, seasonal 5.257658659217877; each prepared reserve output was 0. JSON source kind was `local_historical_example`, so this does not establish that the result came from the imported CSV. |
| Select replay model and saved scenario | **Passed** | Keyboard selection changed the approach to Persistence baseline and selected a saved sensitivity scenario. |
| Operate replay controls with the keyboard | **Passed** | Tab moved focus from the scenario selector to the validation-date year control; Enter activated the fresh-prediction action. |
| Check chart and app warnings | **No warning observed** | No Vega or Streamlit warning/error appeared in the available browser console and local server log during this check. |
| Launch and stop from a clean checkout | **Launch/stop passed; dependency install blocked** | A clean clone imported the source and Streamlit started at canonical port 8501, then stopped with Ctrl+C. The CUA browser network could not reach the process; exact-head Windows CI health check passed. A fresh `pip install -e ".[dev]"` was blocked after one attempt by restricted DNS resolving `setuptools>=84.1.0`; no retry was made. |

## Manual acceptance procedure

1. Follow [Quick start](../README.md#quick-start) from a clean checkout. Confirm the local app opens without a source-data download or model fit; stop it with **Ctrl+C** in the same PowerShell window.
2. In **Next-day AI prediction**, download the blank CSV template. Confirm the columns are `date,ALLSKY_SFC_SW_DWN,T2M,RH2M` and it contains seven blank observation rows.
3. Import a valid CSV with the exact four columns and seven consecutive valid observations. Confirm the app reports a successful import and the displayed rows match the file.
4. Import an invalid CSV containing an invalid observation value such as `-999`. Confirm the app reports an error and does not silently accept the bad rows. Use **Reset to historical example**, then import the valid file again and confirm success.
5. Run a fresh prediction and save the complete JSON result. Compare the displayed target date, forecast values, and reserve outputs with the corresponding exported values; confirm the export includes the input rows, assumptions, and model identity.
6. Open **Historical replay**. Use the keyboard to select a different forecast approach and saved reserve scenario. Confirm the selection and displayed replay respond. Tab through the replay controls and activate a control with Enter or Space.
7. Check the browser console and local server log for Vega, Streamlit, or app warnings/errors during the flows above. Record the outcome, rather than inferring it from unit tests.

Update the recorded status table only after observing each browser result. Keep failed or inaccessible checks pending and record the limitation and any user-assisted action separately.
