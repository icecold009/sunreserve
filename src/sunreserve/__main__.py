"""Command-line interface for explicit local data actions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sunreserve.config import ExperimentConfig
from sunreserve.data import DataError, ProjectPaths, fetch_raw, prepare_daily
from sunreserve.decision import plan_daily_reserve
from sunreserve.evaluate import (
    _safe_run_id,
    _sha256_file,
    finalize_run,
    render_final_report,
    run_validation,
)
from sunreserve.inference import (
    ModelArtifactError,
    load_model_bundle,
    load_observation_file,
    load_project_model,
    predict_next_day,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sunreserve")
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="project root (defaults to the checked-out repository containing this package)",
    )
    parser.add_argument("--config", type=Path, help="optional JSON experiment configuration")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch", help="explicitly download or reuse NASA POWER data")
    fetch.add_argument("--refresh", action="store_true", help="replace the cache after validation")
    commands.add_parser("prepare", help="prepare the cached raw response without network access")
    commands.add_parser(
        "validate",
        help="fit on training rows and select on validation rows using local cached data only",
    )
    finalize = commands.add_parser(
        "finalize",
        help="evaluate the held-out test after explicitly freezing a validated manifest digest",
    )
    finalize.add_argument(
        "--manifest-sha256",
        required=True,
        help="SHA-256 printed by validate after you inspect and freeze its selection manifest",
    )
    finalize.add_argument(
        "--run-id", help="optional safe run identifier; an existing run is never overwritten"
    )
    report = commands.add_parser(
        "report", help="write a report from one already finalized run only"
    )
    report.add_argument("--run-id", required=True, help="exact run directory under artifacts/runs")
    predict = commands.add_parser(
        "predict", help="run fresh next-day Ridge inference from seven observed daily rows"
    )
    predict.add_argument(
        "--observations", type=Path, required=True, help="JSON file with seven daily observations"
    )
    predict.add_argument(
        "--model", type=Path, help="fitted model bundle (defaults to models/ridge_model.json)"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        paths = ProjectPaths(args.root)
        if args.command == "report":
            report_path = render_final_report(paths.runs_dir / _safe_run_id(args.run_id))
            print(f"Finalized-run report: {report_path}")
            return 0
        config = ExperimentConfig.load(args.config) if args.config else ExperimentConfig()
        if args.command == "predict":
            if args.model:
                model = load_model_bundle(args.model.resolve())
            else:
                model, _ = load_project_model(paths.root)
            observations = load_observation_file(args.observations)
            result = predict_next_day(model, observations)
            forecasts = [
                result.radiation_kwh_m2_day,
                result.persistence_kwh_m2_day,
                result.seasonal_mean_kwh_m2_day,
            ]
            plans = plan_daily_reserve(
                forecasts,
                demand_kwh=config.daily_demand_kwh,
                capacity_kwh=config.reserve_cap_kwh,
                buffer_kwh=config.reserve_buffer_kwh,
                panel_area_m2=config.panel_area_m2,
                efficiency=config.conversion_efficiency,
                derating=config.derating_factor,
            )
            labels = ("ridge", "persistence", "seasonal_mean")
            summary = {
                "issue_date": result.issue_date.isoformat(),
                "target_date": result.target_date.isoformat(),
                "target": "NASA POWER daily solar radiation",
                "radiation_units": "kWh/m^2/day",
                "model_id": result.model_id,
                "model_sha256": model.artifact_sha256,
                "training_target_period": [
                    model.training_start.isoformat(),
                    model.training_end.isoformat(),
                ],
                "prediction_and_reserve_plan": [
                    {
                        "approach": label,
                        "predicted_radiation_kwh_m2_day": float(value),
                        "predicted_generation_proxy_kwh_day": float(
                            plans.loc[index, "predicted_generation_proxy_kwh_day"]
                        ),
                        "prepared_reserve_kwh": float(plans.loc[index, "prepared_reserve_kwh"]),
                    }
                    for index, (label, value) in enumerate(zip(labels, forecasts, strict=True))
                ],
                "assumptions": {
                    "daily_demand_kwh": config.daily_demand_kwh,
                    "reserve_capacity_kwh": config.reserve_cap_kwh,
                    "buffer_kwh": config.reserve_buffer_kwh,
                    "panel_area_m2": config.panel_area_m2,
                    "conversion_efficiency": config.conversion_efficiency,
                    "derating_factor": config.derating_factor,
                    "availability_note": (
                        "Historical observation dates only; release timestamps and operational "
                        "data availability were not verified. Not an operational controller."
                    ),
                },
            }
            print(json.dumps(summary, indent=2, allow_nan=False))
            return 0
        if args.command == "fetch":
            path, digest, reused = fetch_raw(config, paths, refresh=args.refresh)
            action = "Reused" if reused else "Saved"
            print(f"{action} validated NASA POWER response: {path}")
            print(f"SHA-256: {digest}")
            print(f"Provenance: {paths.sidecar_path}")
            return 0
        if args.command == "prepare":
            frame, quality = prepare_daily(config, paths)
            print(f"Prepared {len(frame)} calendar days: {paths.daily_path}")
            print(f"Complete-case days: {quality['complete_case_days']}")
            print(f"Excluded days with missing required data: {quality['excluded_days']}")
            print(f"Quality report: {paths.quality_path}")
            return 0
        if args.command == "validate":
            selection = run_validation(paths, config)
            print("Validation only; the final test period was not loaded or evaluated.")
            print(f"Selected approach by validation MAE: {selection.selected_model}")
            print(f"Eligible rows: {selection.eligible_rows}")
            for name, values in selection.metrics.items():
                print(
                    f"{name}: MAE={values['mae']:.4f}, RMSE={values['rmse']:.4f}, "
                    f"bias={values['bias_prediction_minus_actual']:.4f} "
                    f"(n={values['sample_count']})"
                )
            print(f"Validation artifacts: {paths.validation_dir}")
            print(f"Training-only inference model: {paths.model_path}")
            selection_manifest = json.loads(
                (paths.validation_dir / "selection_manifest.json").read_text(encoding="utf-8")
            )
            decision_selection = selection_manifest["decision_selection"]
            print("2022 validation decision scores (mean shortfall + mean excess, kWh/day):")
            for approach, score in decision_selection["scores"].items():
                print(f"{approach}: {score:.6f}")
            print(f"Selected by development reserve score: {decision_selection['selected_model']}")
            print(
                "Frozen selection manifest SHA-256: "
                f"{_sha256_file(paths.validation_dir / 'selection_manifest.json')}"
            )
            return 0
        if args.command == "finalize":
            finalized = finalize_run(
                paths,
                manifest_sha256=args.manifest_sha256,
                run_id=args.run_id,
                expected_config=config if args.config else None,
            )
            print("Final test inspected after frozen manifest and input checks passed.")
            print(f"Frozen selected approach: {finalized.selected_model}")
            print(f"Eligible row counts: {finalized.row_counts}")
            print(f"Finalized run: {finalized.run_dir}")
            return 0
        raise DataError(f"unsupported command {args.command!r}")
    except (DataError, ModelArtifactError, ValueError, OSError) as error:
        print(f"sunreserve: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
