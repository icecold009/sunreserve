# Demo acceptance checklist

This checklist separates observed UI behavior from checks that still need a direct browser observation. It describes the local research/demo prototype only; passing it does not establish site performance or operating readiness.

## Recorded browser checks — 2026-11-23

| Flow | Status | Evidence |
| --- | --- | --- |
| Save the blank CSV template | **Passed** | Downloaded the template and checked its header and seven blank observation rows. |
| Import a valid seven-row CSV | **Pending** | Native file picker was not accessible through the browser automation session. |
| Reject invalid CSV input | **Pending** | Native file picker was not accessible through the browser automation session. |
| Reset after import and reimport valid CSV | **Pending** | Requires the file-picker actions above and a visible reset/reimport result. |
| Run a fresh prediction and save JSON whose values match the displayed result | **Pending** | A fresh prediction was displayed, but the current download and its saved contents were not verified. |
| Select replay model and saved scenario | **Passed** | Keyboard selection changed the approach to Persistence baseline and selected a saved sensitivity scenario. |
| Operate replay controls with the keyboard | **Passed** | Tab moved focus from the scenario selector to the validation-date year control; Enter activated the fresh-prediction action. |
| Check chart and app warnings | **No warning observed** | No Vega or Streamlit warning/error appeared in the available browser console and local server log during this check. |
| Launch and stop from a clean checkout | **Pending** | Verify the README commands from a clean checkout and stop with Ctrl+C in the launching PowerShell session. |

## Manual acceptance procedure

1. Follow [Quick start](../README.md#quick-start) from a clean checkout. Confirm the local app opens without a source-data download or model fit; stop it with **Ctrl+C** in the same PowerShell window.
2. In **Next-day AI prediction**, download the blank CSV template. Confirm the columns are `date,ALLSKY_SFC_SW_DWN,T2M,RH2M` and it contains seven blank observation rows.
3. Import a valid CSV with the exact four columns and seven consecutive valid observations. Confirm the app reports a successful import and the displayed rows match the file.
4. Import an invalid CSV containing an invalid observation value such as `-999`. Confirm the app reports an error and does not silently accept the bad rows. Use **Reset to historical example**, then import the valid file again and confirm success.
5. Run a fresh prediction and save the complete JSON result. Compare the displayed target date, forecast values, and reserve outputs with the corresponding exported values; confirm the export includes the input rows, assumptions, and model identity.
6. Open **Historical replay**. Use the keyboard to select a different forecast approach and saved reserve scenario. Confirm the selection and displayed replay respond. Tab through the replay controls and activate a control with Enter or Space.
7. Check the browser console and local server log for Vega, Streamlit, or app warnings/errors during the flows above. Record the outcome, rather than inferring it from unit tests.

Update the recorded status table only after observing each browser result. Keep failed or inaccessible checks pending and record the limitation and any user-assisted action separately.
