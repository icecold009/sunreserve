"""Check one NOAA GFS daily-aggregation sample using saved CSV evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

SECONDS_PER_KWH_M2 = 3_600_000
EXPECTED_RECORDS = ("f021", "f024", "f030", "f036", "f042", "f045")
EXPECTED_SUPPLEMENTAL_RECORDS = ("f063",)
SUPPLEMENTAL_FORECAST_REFERENCE_UTC = "2020-04-12T18:00:00Z"


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError(f"timestamp must be UTC with an explicit offset: {value}")
    return parsed.astimezone(UTC)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_point_value(evidence_dir: Path, record: dict[str, Any], manifest: dict[str, Any]) -> float:
    filename = record["csv_response_file"]
    require(Path(filename).name == filename, f"unsafe evidence filename: {filename}")
    csv_path = evidence_dir / filename
    payload = csv_path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    require(
        digest == record["csv_response_sha256"],
        f"SHA-256 mismatch for {filename}: {digest}",
    )

    text = payload.decode("utf-8-sig")
    rows = list(csv.DictReader(text.splitlines()))
    require(len(rows) == 1, f"expected one point row in {filename}, found {len(rows)}")
    row = rows[0]
    fields = list(row)
    require("time" in row, f"missing time column in {filename}")
    flux_fields = [field for field in fields if field.startswith("Downward_Short-")]
    require(len(flux_fields) == 1, f"expected one DSWRF field in {filename}")
    flux_field = flux_fields[0]
    require(
        flux_field.split('[unit="', maxsplit=1)[0] == record["field_name"],
        f"CSV field disagrees with metadata in {filename}",
    )
    require('[unit="W.m-2"]' in flux_field, f"unexpected CSV units in {filename}")
    require(
        row["time"] == record["interval_midpoint_utc"],
        f"CSV timestamp is not the metadata interval midpoint in {filename}",
    )

    grid = manifest["study_contract"]["returned_grid_cell"]
    latitude = float(row['latitude[unit="degrees_north"]'])
    require(
        math.isclose(latitude, grid["latitude_degrees_north"], abs_tol=1e-9),
        f"unexpected returned latitude in {filename}",
    )
    longitude = float(row['longitude[unit="degrees_east"]'])
    require(
        math.isclose(longitude, grid["longitude_degrees_east"], abs_tol=1e-9),
        f"unexpected returned longitude in {filename}",
    )
    require(
        row["station"] == manifest["source"]["returned_grid_point_station"],
        f"unexpected station label in {filename}",
    )
    value = float(row[flux_field])
    require(math.isfinite(value) and value >= 0, f"invalid flux in {filename}")
    return value


def validate_manifest_shape(manifest: Any) -> None:
    require(isinstance(manifest, dict), "manifest must be a JSON object")
    for key in (
        "study_contract",
        "source",
        "capture_and_availability",
        "records",
        "supplemental_out_of_target_records",
    ):
        require(key in manifest, f"manifest missing {key}")
    require(isinstance(manifest["study_contract"], dict), "study_contract must be an object")
    require(isinstance(manifest["source"], dict), "source must be an object")
    require(
        isinstance(manifest["capture_and_availability"], dict),
        "capture_and_availability must be an object",
    )
    require(isinstance(manifest["records"], list), "records must be an array")
    require(
        isinstance(manifest["supplemental_out_of_target_records"], list),
        "supplemental_out_of_target_records must be an array",
    )


def validate_unique_record_ids(manifest: dict[str, Any]) -> None:
    target_ids = [record["id"] for record in manifest["records"]]
    supplemental_ids = [record["id"] for record in manifest["supplemental_out_of_target_records"]]
    require(
        len(target_ids) == len(set(target_ids)),
        "duplicate record ID in target records",
    )
    require(
        len(supplemental_ids) == len(set(supplemental_ids)),
        "duplicate record ID in supplemental records",
    )
    collisions = set(target_ids).intersection(supplemental_ids)
    require(
        not collisions,
        f"record ID appears in both target and supplemental records: {sorted(collisions)}",
    )


def validate_record_metadata(record: dict[str, Any], source: dict[str, Any]) -> None:
    metadata = record["metadata_response_excerpt"]
    require(
        metadata["forecast_reference_time_utc"] == source["forecast_reference_time_utc"],
        f"forecast run mismatch in {record['id']}",
    )
    require(
        metadata["parameter"] == "0/4/192",
        f"parameter mismatch in {record['id']}",
    )
    require(
        metadata["level"] == "surface, ground or water",
        f"level mismatch in {record['id']}",
    )
    require(metadata["units"] == "W.m-2", f"unit mismatch in {record['id']}")
    require(
        metadata["statistical_processing"] == "Average",
        f"statistical process mismatch in {record['id']}",
    )
    require(
        metadata["field_name"] == record["field_name"],
        f"field-name mismatch in {record['id']}",
    )

    start = parse_utc(record["interval_start_utc"])
    end = parse_utc(record["interval_end_utc"])
    midpoint = parse_utc(record["interval_midpoint_utc"])
    duration_seconds = (end - start).total_seconds()
    require(duration_seconds > 0, f"empty interval in {record['id']}")
    require(
        duration_seconds == record["interval_hours"] * 3600,
        f"interval duration disagrees with metadata in {record['id']}",
    )
    require(
        record["field_name"].endswith(f"_{record['interval_hours']}_Hour_Average"),
        f"field averaging period disagrees with interval in {record['id']}",
    )
    require(midpoint - start == end - midpoint, f"midpoint mismatch in {record['id']}")
    reference = parse_utc(source["forecast_reference_time_utc"])
    lead_end = (end - reference).total_seconds() / 3600
    lead_midpoint = (midpoint - reference).total_seconds() / 3600
    require(
        lead_end == record["forecast_lead_end_hours"],
        f"lead time disagrees with interval in {record['id']}",
    )
    require(
        metadata["time_axis_start_lead_hours"] == (start - reference).total_seconds() / 3600,
        f"metadata interval start disagrees in {record['id']}",
    )
    require(
        metadata["time_axis_end_lead_hours"] == lead_end,
        f"metadata interval end disagrees in {record['id']}",
    )
    require(
        metadata["time_axis_midpoint_lead_hours"] == lead_midpoint,
        f"metadata interval midpoint disagrees in {record['id']}",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("docs/evidence/noaa_gfs_grid4_20200412_0600_target_evidence.json"),
        help="path to the saved point-sample evidence manifest",
    )
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_manifest_shape(manifest)
        validate_unique_record_ids(manifest)
        evidence_dir = manifest_path.parent
        study = manifest["study_contract"]
        source = manifest["source"]
        records = {record["id"]: record for record in manifest["records"]}
        require(
            tuple(records) == EXPECTED_RECORDS,
            "record set/order does not match test contract",
        )

        values: dict[str, float] = {}
        for record_id in EXPECTED_RECORDS:
            record = records[record_id]
            validate_record_metadata(record, source)
            values[record_id] = load_point_value(evidence_dir, record, manifest)

        supplemental = {
            record["id"]: record for record in manifest["supplemental_out_of_target_records"]
        }
        require(
            tuple(supplemental) == EXPECTED_SUPPLEMENTAL_RECORDS,
            "supplemental record set/order does not match contract",
        )
        supplemental_record = supplemental["f063"]
        require(
            supplemental_record["forecast_reference_time_utc"]
            == SUPPLEMENTAL_FORECAST_REFERENCE_UTC
            and supplemental_record["interval_start_utc"] == "2020-04-15T06:00:00Z"
            and supplemental_record["interval_end_utc"] == "2020-04-15T09:00:00Z"
            and supplemental_record["interval_midpoint_utc"] == "2020-04-15T07:30:00Z"
            and supplemental_record["forecast_lead_end_hours"] == 63,
            "supplemental f063 identity or interval does not match the documented sample",
        )
        require(
            supplemental_record["metadata_response_excerpt"]["forecast_reference_time_utc"]
            == supplemental_record["forecast_reference_time_utc"],
            "supplemental f063 forecast reference time disagrees with metadata",
        )
        validate_record_metadata(
            supplemental_record,
            {"forecast_reference_time_utc": SUPPLEMENTAL_FORECAST_REFERENCE_UTC},
        )
        supplemental_value = load_point_value(evidence_dir, supplemental_record, manifest)
        target_start = parse_utc(manifest["study_contract"]["target_start_utc"])
        target_end = parse_utc(manifest["study_contract"]["target_end_utc"])
        supplemental_start = parse_utc(supplemental_record["interval_start_utc"])
        supplemental_end = parse_utc(supplemental_record["interval_end_utc"])
        require(
            supplemental_end <= target_start or supplemental_start >= target_end,
            "supplemental record overlaps target interval",
        )

        target_start = parse_utc(study["target_start_utc"])
        target_end = parse_utc(study["target_end_utc"])
        require(
            target_end - target_start == timedelta(hours=24),
            "target interval is not 24 hours",
        )

        parent = records["f024"]
        child = records["f021"]
        parent_start = parse_utc(parent["interval_start_utc"])
        parent_end = parse_utc(parent["interval_end_utc"])
        child_start = parse_utc(child["interval_start_utc"])
        child_end = parse_utc(child["interval_end_utc"])
        require(child_start == parent_start, "nested means do not share an interval start")
        require(child_end == target_start, "nested shorter mean does not end at target start")
        require(parent_end > target_start, "parent mean does not straddle target start")
        require(
            parent_end - target_start == child_end - child_start,
            "nested difference does not reconstruct an equal-length target segment",
        )

        def energy_kwh_m2(flux: float, start: datetime, end: datetime) -> float:
            return flux * (end - start).total_seconds() / SECONDS_PER_KWH_M2

        parent_energy = energy_kwh_m2(values["f024"], parent_start, parent_end)
        child_energy = energy_kwh_m2(values["f021"], child_start, child_end)
        reconstructed_energy = parent_energy - child_energy
        require(reconstructed_energy >= 0, "nested subtraction produced negative target energy")

        segments: list[dict[str, Any]] = [
            {
                "interval_start_utc": study["target_start_utc"],
                "interval_end_utc": parent["interval_end_utc"],
                "energy_kwh_m2": reconstructed_energy,
                "method": "nested-window energy subtraction: f024 minus f021",
            }
        ]
        for record_id in ("f030", "f036", "f042", "f045"):
            record = records[record_id]
            segments.append(
                {
                    "interval_start_utc": record["interval_start_utc"],
                    "interval_end_utc": record["interval_end_utc"],
                    "energy_kwh_m2": energy_kwh_m2(
                        values[record_id],
                        parse_utc(record["interval_start_utc"]),
                        parse_utc(record["interval_end_utc"]),
                    ),
                    "method": f"mean flux integration: {record_id}",
                }
            )

        cursor = target_start
        for segment in segments:
            start = parse_utc(segment["interval_start_utc"])
            end = parse_utc(segment["interval_end_utc"])
            require(start == cursor, f"gap or overlap before segment starting {start.isoformat()}")
            require(end > start, "empty reconstructed segment")
            cursor = end
        require(cursor == target_end, "segments do not cover the complete target interval")

        total = sum(segment["energy_kwh_m2"] for segment in segments)
        capture = manifest["capture_and_availability"]
        result = {
            "aggregation_status": "PASS",
            "forecast_reference_time_utc": source["forecast_reference_time_utc"],
            "grid_point": study["returned_grid_cell"],
            "target_interval_utc": [study["target_start_utc"], study["target_end_utc"]],
            "daily_integrated_downward_shortwave_kwh_m2": round(total, 9),
            "segments": segments,
            "historical_issue_time_availability": capture["historical_issue_time_availability"],
            "exact_receipt_timestamps_preserved_for_all_records": capture[
                "exact_receipt_timestamps_preserved_for_all_records"
            ],
            "full_raw_metadata_xml_preserved": False,
            "raw_grib_checksum_preserved": capture["raw_grib_bytes_or_checksum_preserved"],
            "supplemental_out_of_target_record": {
                "id": "f063",
                "csv_flux_w_m2": supplemental_value,
                "excluded_from_target_integration": True,
                "reason": (
                    "Its [2020-04-15T06:00:00Z, 2020-04-15T09:00:00Z) interval "
                    "is outside the target."
                ),
            },
            "operational_evaluation": "NOT PERMITTED on this evidence",
            "interpretation": (
                "One retrieved model-grid sample passes interval reconstruction; "
                "this is not an operational-validity or site-outcome result."
            ),
        }
        print(json.dumps(result, indent=2))
        return 0
    except (
        KeyError,
        OSError,
        UnicodeError,
        ValueError,
        TypeError,
        AttributeError,
        OverflowError,
        IndexError,
    ) as exc:
        print(f"AGGREGATION FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
