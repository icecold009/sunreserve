"""Capture and verify one bounded prospective NOAA GFS vintage offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

CYCLE = "2026112112"
CYCLE_START = datetime(2026, 11, 21, 12, tzinfo=UTC)
TARGET_START = datetime(2026, 11, 23, 3, tzinfo=UTC)
TARGET_END = datetime(2026, 11, 24, 3, tzinfo=UTC)
REQUESTED_POINT = {"latitude_degrees_north": 12.9716, "longitude_degrees_east": 77.5946}
GRID_POINT = {"latitude_degrees_north": 13.0, "longitude_degrees_east": 77.5}
BOUNDS = {"leftlon": "77.375", "rightlon": "77.625", "toplat": "13.125", "bottomlat": "12.875"}
LEAD_INTERVALS = {
    39: (36, 39),
    42: (36, 42),
    45: (42, 45),
    48: (42, 48),
    51: (48, 51),
    54: (48, 54),
    57: (54, 57),
    60: (54, 60),
    63: (60, 63),
}
LEADS = tuple(LEAD_INTERVALS)
EXPECTED_SEGMENTS = (
    {
        "start_lead_hours": 39,
        "end_lead_hours": 42,
        "method": "f042 minus f039",
        "source_leads": [42, 39],
    },
    {
        "start_lead_hours": 42,
        "end_lead_hours": 45,
        "method": "f045 direct interval",
        "source_leads": [45],
    },
    {
        "start_lead_hours": 45,
        "end_lead_hours": 48,
        "method": "f048 minus f045",
        "source_leads": [48, 45],
    },
    {
        "start_lead_hours": 48,
        "end_lead_hours": 51,
        "method": "f051 direct interval",
        "source_leads": [51],
    },
    {
        "start_lead_hours": 51,
        "end_lead_hours": 54,
        "method": "f054 minus f051",
        "source_leads": [54, 51],
    },
    {
        "start_lead_hours": 54,
        "end_lead_hours": 57,
        "method": "f057 direct interval",
        "source_leads": [57],
    },
    {
        "start_lead_hours": 57,
        "end_lead_hours": 60,
        "method": "f060 minus f057",
        "source_leads": [60, 57],
    },
    {
        "start_lead_hours": 60,
        "end_lead_hours": 63,
        "method": "f063 direct interval",
        "source_leads": [63],
    },
)
DEFAULT_OUTPUT = Path("docs/evidence/noaa_gfs_20261121_12z_bengaluru")
NOMADS_BASE = "https://nomads.ncep.noaa.gov"
PRODUCT_DIR = "/gfs.20261121/12/atmos"
FILTER_ENDPOINT = f"{NOMADS_BASE}/cgi-bin/filter_gfs_0p25.pl"
USER_AGENT = "SunReserve bounded prospective GFS capture/1.0"
INTERVAL_PATTERN = re.compile(r":DSWRF:surface:(\d+)-(\d+)\s+hour\s+ave\s+fcst\b", re.IGNORECASE)


class CaptureError(ValueError):
    """Raised when the captured response does not match the frozen request contract."""


def timestamp_utc(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError("capture clock must return a timezone-aware UTC datetime")
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def utc_now() -> datetime:
    return datetime.now(UTC)


def build_index_request(lead: int) -> dict[str, Any]:
    filename = f"gfs.t12z.pgrb2.0p25.f{lead:03d}"
    url = f"{NOMADS_BASE}/pub/data/nccf/com/gfs/prod/gfs.20261121/12/atmos/{filename}.idx"
    return {"method": "GET", "url": url, "parameters": {}}


def build_grib_request(lead: int) -> dict[str, Any]:
    filename = f"gfs.t12z.pgrb2.0p25.f{lead:03d}"
    parameters = {
        "file": filename,
        "var_DSWRF": "on",
        "lev_surface": "on",
        "subregion": "",
        **BOUNDS,
        "dir": PRODUCT_DIR,
    }
    query = urllib.parse.urlencode(parameters)
    return {"method": "GET", "url": f"{FILTER_ENDPOINT}?{query}", "parameters": parameters}


def fetch_response(
    url: str,
    *,
    opener: Callable[..., Any] | None = None,
    clock: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    open_url = opener or urllib.request.urlopen
    now = clock or utc_now
    started = timestamp_utc(now())
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT},
        method="GET",
    )
    with open_url(request, timeout=60) as response:
        body = response.read()
        received = timestamp_utc(now())
        return {
            "body": body,
            "request_started_utc": started,
            "body_received_utc": received,
            "http_status": int(response.status),
            "response_headers": {str(key): str(value) for key, value in response.headers.items()},
            "final_url": response.geturl(),
        }


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def save_exclusive(path: Path, payload: bytes) -> dict[str, Any]:
    with path.open("xb") as output:
        output.write(payload)
    return {"file": path.name, "size_bytes": len(payload), "sha256": sha256(payload)}


def validate_index(payload: bytes, lead: int) -> dict[str, int]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CaptureError(f"f{lead:03d} index response is not UTF-8") from exc
    candidates = [
        (line, match)
        for line in text.splitlines()
        if ":DSWRF:surface:" in line
        for match in [INTERVAL_PATTERN.search(line)]
        if match is not None
    ]
    if len(candidates) != 1:
        raise CaptureError(
            f"f{lead:03d} index must contain exactly one DSWRF surface average interval; "
            f"found {len(candidates)}"
        )
    line, match = candidates[0]
    if f":d={CYCLE}:" not in line:
        raise CaptureError(f"f{lead:03d} DSWRF index row has the wrong forecast reference time")
    start, end = (int(value) for value in match.groups())
    expected = LEAD_INTERVALS[lead]
    if (start, end) != expected:
        raise CaptureError(
            f"f{lead:03d} DSWRF interval is {start}-{end}h, expected {expected[0]}-{expected[1]}h"
        )
    return {"start_lead_hours": start, "end_lead_hours": end}


def validate_grib2_message(payload: bytes, lead: int) -> dict[str, Any]:
    if len(payload) < 20:
        raise CaptureError(f"f{lead:03d} GRIB2 response is shorter than one message")
    if payload[:4] != b"GRIB":
        raise CaptureError(f"f{lead:03d} response does not begin with GRIB magic")
    if payload[7] != 2:
        raise CaptureError(f"f{lead:03d} response is not GRIB edition 2")
    encoded_length = struct.unpack(">Q", payload[8:16])[0]
    if encoded_length != len(payload):
        raise CaptureError(
            f"f{lead:03d} GRIB2 length field says {encoded_length}, "
            f"response has {len(payload)} bytes"
        )
    if payload[-4:] != b"7777":
        raise CaptureError(f"f{lead:03d} GRIB2 response lacks the 7777 message trailer")

    sections: dict[int, list[bytes]] = {}
    offset = 16
    while offset < len(payload) - 4:
        if offset + 5 > len(payload) - 4:
            raise CaptureError(f"f{lead:03d} GRIB2 has a truncated section header")
        section_length = int.from_bytes(payload[offset : offset + 4], "big")
        section_number = payload[offset + 4]
        if section_length < 5 or offset + section_length > len(payload) - 4:
            raise CaptureError(f"f{lead:03d} GRIB2 has an invalid section length")
        sections.setdefault(section_number, []).append(payload[offset : offset + section_length])
        offset += section_length
    if offset != len(payload) - 4:
        raise CaptureError(f"f{lead:03d} GRIB2 sections do not end at the message trailer")
    for section_number in (1, 3, 4, 5, 6, 7):
        if len(sections.get(section_number, [])) != 1:
            raise CaptureError(
                f"f{lead:03d} GRIB2 must contain exactly one section {section_number}"
            )
    identification = sections[1][0]
    if len(identification) < 21:
        raise CaptureError(f"f{lead:03d} GRIB2 identification section is truncated")
    try:
        reference_time = datetime(
            int.from_bytes(identification[12:14], "big"),
            identification[14],
            identification[15],
            identification[16],
            identification[17],
            identification[18],
            tzinfo=UTC,
        )
    except ValueError as exc:
        raise CaptureError(f"f{lead:03d} GRIB2 has an invalid reference timestamp") from exc
    if reference_time != CYCLE_START:
        raise CaptureError(
            f"f{lead:03d} GRIB2 reference time {timestamp_utc(reference_time)} "
            f"does not match capture cycle {timestamp_utc(CYCLE_START)}"
        )

    product = sections[4][0]
    if len(product) < 58 or int.from_bytes(product[7:9], "big") != 8:
        raise CaptureError(f"f{lead:03d} GRIB2 does not use product definition template 4.8")
    expected_interval = LEAD_INTERVALS[lead]
    if payload[6] != 0 or product[9:11] != bytes((4, 192)):
        raise CaptureError(f"f{lead:03d} GRIB2 product metadata is not the expected DSWRF field")
    if (
        product[22] != 1
        or product[23] != 0
        or int.from_bytes(product[24:28], "big") != 0
        or product[28] != 255
    ):
        raise CaptureError(
            f"f{lead:03d} GRIB2 fixed surfaces do not encode the expected surface level"
        )
    forecast_unit = product[17]
    forecast_start = int.from_bytes(product[18:22], "big")
    if forecast_unit != 1 or forecast_start != expected_interval[0]:
        raise CaptureError(
            f"f{lead:03d} GRIB2 forecast period starts at {forecast_start} in unit "
            f"{forecast_unit}; expected hour {expected_interval[0]}"
        )
    try:
        interval_end = datetime(
            int.from_bytes(product[34:36], "big"),
            product[36],
            product[37],
            product[38],
            product[39],
            product[40],
            tzinfo=UTC,
        )
    except ValueError as exc:
        raise CaptureError(f"f{lead:03d} GRIB2 has an invalid interval-end timestamp") from exc
    expected_end = CYCLE_START + timedelta(hours=expected_interval[1])
    if interval_end != expected_end:
        raise CaptureError(
            f"f{lead:03d} GRIB2 interval ends at {timestamp_utc(interval_end)}, "
            f"expected {timestamp_utc(expected_end)}"
        )
    if (
        product[41] != 1
        or product[46] != 0
        or product[47] != 2
        or product[48] != 1
        or int.from_bytes(product[49:53], "big") != expected_interval[1] - expected_interval[0]
        or product[53] != 255
        or int.from_bytes(product[54:58], "big") != 0
    ):
        raise CaptureError(
            f"f{lead:03d} GRIB2 time-range metadata disagrees with expected interval "
            f"{expected_interval[0]}-{expected_interval[1]}h"
        )

    grid = sections[3][0]
    if len(grid) < 72 or int.from_bytes(grid[12:14], "big") != 0:
        raise CaptureError(
            f"f{lead:03d} GRIB2 does not use regular latitude/longitude grid template 3.0"
        )
    point_count = int.from_bytes(grid[6:10], "big")
    ni = int.from_bytes(grid[30:34], "big")
    nj = int.from_bytes(grid[34:38], "big")
    if point_count != 1 or ni != 1 or nj != 1:
        raise CaptureError(
            f"f{lead:03d} filtered GRIB2 is not one grid point: "
            f"points={point_count}, Ni={ni}, Nj={nj}"
        )
    basic_angle = int.from_bytes(grid[38:42], "big")
    subdivisions = int.from_bytes(grid[42:46], "big")
    coordinate_unit = (
        1e-6 if basic_angle == 0 or subdivisions == 0xFFFFFFFF else basic_angle / subdivisions
    )

    def coordinate(start: int) -> float:
        encoded = int.from_bytes(grid[start : start + 4], "big", signed=True)
        return encoded * coordinate_unit

    observed_grid = {
        "number_of_data_points": point_count,
        "Ni": ni,
        "Nj": nj,
        "latitude_first_degrees_north": coordinate(46),
        "longitude_first_degrees_east": coordinate(50),
        "latitude_last_degrees_north": coordinate(55),
        "longitude_last_degrees_east": coordinate(59),
    }
    for key, expected in (
        ("latitude_first_degrees_north", GRID_POINT["latitude_degrees_north"]),
        ("latitude_last_degrees_north", GRID_POINT["latitude_degrees_north"]),
        ("longitude_first_degrees_east", GRID_POINT["longitude_degrees_east"]),
        ("longitude_last_degrees_east", GRID_POINT["longitude_degrees_east"]),
    ):
        if abs(observed_grid[key] - expected) > 1e-6:
            raise CaptureError(
                f"f{lead:03d} GRIB2 {key}={observed_grid[key]} disagrees with "
                f"expected point {expected}"
            )
    return observed_grid


def target_segments() -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for segment in EXPECTED_SEGMENTS:
        start = CYCLE_START + timedelta(hours=segment["start_lead_hours"])
        end = CYCLE_START + timedelta(hours=segment["end_lead_hours"])
        segments.append(
            {
                **segment,
                "interval_start_utc": timestamp_utc(start),
                "interval_end_utc": timestamp_utc(end),
            }
        )
    return segments


def validate_segment_coverage(segments: list[dict[str, Any]]) -> None:
    if len(segments) != len(EXPECTED_SEGMENTS):
        raise CaptureError("reconstructed segments do not match the eight-part target contract")
    cursor = TARGET_START
    for segment, expected in zip(segments, EXPECTED_SEGMENTS, strict=True):
        if segment != {
            **expected,
            "interval_start_utc": timestamp_utc(
                CYCLE_START + timedelta(hours=expected["start_lead_hours"])
            ),
            "interval_end_utc": timestamp_utc(
                CYCLE_START + timedelta(hours=expected["end_lead_hours"])
            ),
        }:
            raise CaptureError("reconstructed segment metadata disagrees with target contract")
        start = CYCLE_START + timedelta(hours=segment["start_lead_hours"])
        end = CYCLE_START + timedelta(hours=segment["end_lead_hours"])
        if start != cursor or end <= start:
            raise CaptureError("reconstructed segments contain a gap, overlap, or empty interval")
        cursor = end
    if cursor != TARGET_END:
        raise CaptureError("reconstructed segments do not cover the complete target interval")


def response_record(
    request: dict[str, Any], response: dict[str, Any], file_info: dict[str, Any]
) -> dict[str, Any]:
    return {
        "request": request,
        "request_started_utc": response["request_started_utc"],
        "body_received_utc": response["body_received_utc"],
        "http_status": response["http_status"],
        "response_headers": response["response_headers"],
        "final_url": response["final_url"],
        **file_info,
    }


def base_receipt(started_utc: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "capture_status": "INCOMPLETE",
        "capture_started_utc": started_utc,
        "capture_completed_utc": None,
        "purpose": "one exploratory prospective GFS forecast vintage",
        "latest_cycle_claim": False,
        "issue_cutoff_availability": "NOT ESTABLISHED",
        "protocol_issue_cutoff_local": "2026-11-22T18:00:00+05:30",
        "operational_result_computed": False,
        "forecast_reference_time_utc": timestamp_utc(CYCLE_START),
        "target_interval_utc": [timestamp_utc(TARGET_START), timestamp_utc(TARGET_END)],
        "target_interval_local": ["2026-11-23T08:30:00+05:30", "2026-11-24T08:30:00+05:30"],
        "target_interval_convention": "half-open [start, end)",
        "source": {
            "provider": "NOAA NOMADS",
            "product": "GFS 0.25 degree pgrb2",
            "variable": "DSWRF",
            "level": "surface",
            "units": "W m-2",
            "grid_point_selection": "single point returned by the specified bounds",
        },
        "requested_point": REQUESTED_POINT,
        "expected_grid_point": GRID_POINT,
        "bounds": BOUNDS,
        "expected_forecast_lead_intervals": [
            {
                "lead_end_hours": lead,
                "interval_start_lead_hours": interval[0],
                "interval_end_lead_hours": interval[1],
            }
            for lead, interval in LEAD_INTERVALS.items()
        ],
        "target_reconstruction_segments": target_segments(),
        "records": [],
        "limitations": [
            "This capture does not establish that the selected run is the latest cycle.",
            "This capture does not establish availability by the 18:00 IST protocol issue cutoff.",
            "No radiation values are integrated and no operational evaluation result is computed.",
        ],
    }


def write_receipt(output_dir: Path, receipt: dict[str, Any]) -> None:
    path = output_dir / "capture_receipt.json"
    with path.open("xb") as output:
        output.write((json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def capture(
    output_dir: Path = DEFAULT_OUTPUT,
    *,
    opener: Callable[..., Any] | None = None,
    clock: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"capture output already exists; refusing overwrite: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    now = clock or utc_now
    receipt = base_receipt(timestamp_utc(now()))
    records: dict[int, dict[str, Any]] = {}

    try:
        for lead in LEADS:
            request = build_index_request(lead)
            response = fetch_response(request["url"], opener=opener, clock=now)
            file_info = save_exclusive(output_dir / f"f{lead:03d}.idx", response["body"])
            record = {
                "lead_end_hours": lead,
                "idx_response": response_record(request, response, file_info),
            }
            receipt["records"].append(record)
            records[lead] = record
            if response["http_status"] != 200:
                raise CaptureError(f"f{lead:03d} index HTTP status {response['http_status']}")

        for lead in LEADS:
            records[lead]["decoded_dswrf_interval"] = validate_index(
                (output_dir / f"f{lead:03d}.idx").read_bytes(), lead
            )

        segments = target_segments()
        validate_segment_coverage(segments)

        for lead in LEADS:
            request = build_grib_request(lead)
            response = fetch_response(request["url"], opener=opener, clock=now)
            file_info = save_exclusive(output_dir / f"f{lead:03d}.grib2", response["body"])
            records[lead]["grib2_response"] = response_record(request, response, file_info)
            if response["http_status"] != 200:
                raise CaptureError(f"f{lead:03d} GRIB2 HTTP status {response['http_status']}")
            records[lead]["grib2_grid_definition"] = validate_grib2_message(response["body"], lead)

        receipt["capture_status"] = "COMPLETE"
        receipt["capture_completed_utc"] = timestamp_utc(now())
        write_receipt(output_dir, receipt)
        return receipt
    except Exception as exc:
        receipt["capture_status"] = "INCOMPLETE"
        receipt["capture_completed_utc"] = timestamp_utc(now())
        receipt["capture_error"] = f"{type(exc).__name__}: {exc}"
        write_receipt(output_dir, receipt)
        raise


def _validate_file_record(directory: Path, info: dict[str, Any], lead: int, suffix: str) -> bytes:
    filename = info["file"]
    if Path(filename).name != filename or filename != f"f{lead:03d}.{suffix}":
        raise CaptureError(f"unsafe or unexpected evidence filename for f{lead:03d}.{suffix}")
    path = directory / filename
    payload = path.read_bytes()
    actual_hash = sha256(payload)
    if len(payload) != info["size_bytes"] or actual_hash != info["sha256"]:
        raise CaptureError(f"evidence size/hash mismatch for {filename}")
    return payload


def _validate_receipt_timestamps(info: dict[str, Any], lead: int, response_name: str) -> None:
    if not isinstance(info, dict):
        raise CaptureError(f"f{lead:03d} {response_name} receipt is not an object")
    parsed: list[datetime] = []
    for field in ("request_started_utc", "body_received_utc"):
        value = info.get(field)
        if not isinstance(value, str):
            raise CaptureError(f"f{lead:03d} {response_name} receipt is missing valid {field}")
        try:
            timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise CaptureError(
                f"f{lead:03d} {response_name} receipt has invalid {field}: {value!r}"
            ) from exc
        if timestamp.utcoffset() != timedelta(0):
            raise CaptureError(f"f{lead:03d} {response_name} {field} must have a UTC timezone")
        parsed.append(timestamp.astimezone(UTC))
    if parsed[1] < parsed[0]:
        raise CaptureError(f"f{lead:03d} {response_name} body receipt precedes request start")


def verify_capture(output_dir: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    receipt_path = output_dir / "capture_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("capture_status") != "COMPLETE":
        raise CaptureError("receipt does not mark the capture complete")
    if receipt.get("purpose") != "one exploratory prospective GFS forecast vintage":
        raise CaptureError("receipt purpose does not match the exploratory-capture contract")
    if (
        receipt.get("latest_cycle_claim") is not False
        or receipt.get("operational_result_computed") is not False
    ):
        raise CaptureError("receipt makes an unsupported latest-cycle or operational-result claim")
    if receipt.get("issue_cutoff_availability") != "NOT ESTABLISHED":
        raise CaptureError("receipt must leave issue-cutoff availability unestablished")
    if receipt.get("forecast_reference_time_utc") != timestamp_utc(CYCLE_START):
        raise CaptureError("receipt has an unexpected forecast reference time")
    if receipt.get("target_interval_utc") != [
        timestamp_utc(TARGET_START),
        timestamp_utc(TARGET_END),
    ]:
        raise CaptureError("receipt has an unexpected target interval")
    if receipt.get("target_reconstruction_segments") != target_segments():
        raise CaptureError("receipt reconstruction segments do not match target contract")
    validate_segment_coverage(receipt["target_reconstruction_segments"])
    records = receipt.get("records")
    if not isinstance(records, list) or [item.get("lead_end_hours") for item in records] != list(
        LEADS
    ):
        raise CaptureError("receipt records do not contain exactly the nine ordered forecast leads")

    observed_grids: list[dict[str, Any]] = []
    for record in records:
        lead = record["lead_end_hours"]
        idx_request = build_index_request(lead)
        idx_response = record["idx_response"]
        _validate_receipt_timestamps(idx_response, lead, "index")
        if idx_response["request"] != idx_request or idx_response["http_status"] != 200:
            raise CaptureError(
                f"f{lead:03d} index request/status does not match the capture contract"
            )
        idx_payload = _validate_file_record(output_dir, idx_response, lead, "idx")
        decoded = validate_index(idx_payload, lead)
        if record.get("decoded_dswrf_interval") != decoded:
            raise CaptureError(f"f{lead:03d} receipt interval disagrees with its saved index")

        grib_request = build_grib_request(lead)
        grib_response = record.get("grib2_response")
        if not isinstance(grib_response, dict):
            raise CaptureError(f"f{lead:03d} has no saved filtered GRIB2 response record")
        _validate_receipt_timestamps(grib_response, lead, "GRIB2")
        if grib_response["request"] != grib_request or grib_response["http_status"] != 200:
            raise CaptureError(
                f"f{lead:03d} GRIB2 request/status does not match the capture contract"
            )
        grib_payload = _validate_file_record(output_dir, grib_response, lead, "grib2")
        observed_grids.append(validate_grib2_message(grib_payload, lead))
        if (
            "grib2_grid_definition" in record
            and record["grib2_grid_definition"] != observed_grids[-1]
        ):
            raise CaptureError(f"f{lead:03d} receipt grid definition disagrees with saved GRIB2")

    return {
        "verification_status": "PASS",
        "capture_receipt": str(receipt_path),
        "verified_records": len(records),
        "verified_target_segments": len(EXPECTED_SEGMENTS),
        "observed_grid_definitions": observed_grids,
        "operational_result_computed": False,
        "issue_cutoff_availability": "NOT ESTABLISHED",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture_parser = subparsers.add_parser("capture", help="make the one bounded NOAA GET capture")
    capture_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    verify_parser = subparsers.add_parser(
        "verify", help="verify a saved capture without network access"
    )
    verify_parser.add_argument("--directory", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        if args.command == "capture":
            receipt = capture(args.output)
            print(
                json.dumps(
                    {
                        "capture_status": receipt["capture_status"],
                        "receipt": str(args.output / "capture_receipt.json"),
                    }
                )
            )
        else:
            print(json.dumps(verify_capture(args.directory), indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"{args.command.upper()} FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
