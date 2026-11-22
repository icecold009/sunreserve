"""Capture and verify one bounded prospective NOAA GFS vintage offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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


def build_cycle_index_request(lead: int, cycle: str) -> dict[str, Any]:
    cycle_start = parse_cycle(cycle)
    day = cycle_start.strftime("%Y%m%d")
    hour = cycle_start.strftime("%H")
    filename = f"gfs.t{hour}z.pgrb2.0p25.f{lead:03d}"
    url = f"{NOMADS_BASE}/pub/data/nccf/com/gfs/prod/gfs.{day}/{hour}/atmos/{filename}.idx"
    return {"method": "GET", "url": url, "parameters": {}}


def build_cycle_grib_request(
    lead: int, cycle: str, bounds: dict[str, str] = BOUNDS
) -> dict[str, Any]:
    cycle_start = parse_cycle(cycle)
    day = cycle_start.strftime("%Y%m%d")
    hour = cycle_start.strftime("%H")
    filename = f"gfs.t{hour}z.pgrb2.0p25.f{lead:03d}"
    parameters = {
        "file": filename,
        "var_DSWRF": "on",
        "lev_surface": "on",
        "subregion": "",
        **{key: bounds[key] for key in BOUNDS},
        "dir": f"/gfs.{day}/{hour}/atmos",
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


def validate_grib2_message(
    payload: bytes,
    lead: int,
    *,
    cycle_start: datetime = CYCLE_START,
    expected_interval: tuple[int, int] | None = None,
    expected_grid_point: dict[str, float] = GRID_POINT,
) -> dict[str, Any]:
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
    if reference_time != cycle_start:
        raise CaptureError(
            f"f{lead:03d} GRIB2 reference time {timestamp_utc(reference_time)} "
            f"does not match capture cycle {timestamp_utc(cycle_start)}"
        )

    product = sections[4][0]
    if len(product) < 58 or int.from_bytes(product[7:9], "big") != 8:
        raise CaptureError(f"f{lead:03d} GRIB2 does not use product definition template 4.8")
    expected_interval = expected_interval or LEAD_INTERVALS[lead]
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
    expected_end = cycle_start + timedelta(hours=expected_interval[1])
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
        ("latitude_first_degrees_north", expected_grid_point["latitude_degrees_north"]),
        ("latitude_last_degrees_north", expected_grid_point["latitude_degrees_north"]),
        ("longitude_first_degrees_east", expected_grid_point["longitude_degrees_east"]),
        ("longitude_last_degrees_east", expected_grid_point["longitude_degrees_east"]),
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


def _capture_legacy(
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


def verify_legacy_capture(output_dir: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
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


def parse_cycle(value: str) -> datetime:
    if not isinstance(value, str) or re.fullmatch(r"\d{10}", value) is None:
        raise CaptureError("cycle must use YYYYMMDDHH format")
    try:
        result = datetime.strptime(value, "%Y%m%d%H").replace(tzinfo=UTC)
    except ValueError as exc:
        raise CaptureError(f"cycle is not a valid UTC date/hour: {value!r}") from exc
    if result.hour not in {0, 6, 12, 18}:
        raise CaptureError("GFS cycle hour must be one of 00, 06, 12, or 18 UTC")
    return result


def parse_utc_datetime(value: str | datetime, name: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise CaptureError(f"{name} must be an ISO-8601 timezone-aware timestamp") from exc
    else:
        raise CaptureError(f"{name} must be an ISO-8601 timezone-aware timestamp")
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise CaptureError(f"{name} must use a UTC timezone")
    return parsed.astimezone(UTC)


def _target_offsets(
    cycle_start: datetime, target_start: datetime, target_end: datetime
) -> tuple[int, int]:
    if target_end <= target_start:
        raise CaptureError("target interval must be a non-empty half-open interval")
    start_seconds = (target_start - cycle_start).total_seconds()
    end_seconds = (target_end - cycle_start).total_seconds()
    if start_seconds < 0 or end_seconds > 384 * 3600:
        raise CaptureError(
            "target interval must follow the cycle and end within 384 forecast hours"
        )
    if start_seconds % 3600 or end_seconds % 3600:
        raise CaptureError("target boundaries must align to whole UTC hours from the cycle")
    return int(start_seconds // 3600), int(end_seconds // 3600)


def decode_dswrf_interval(payload: bytes, cycle: str, lead: int) -> dict[str, int]:
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
    if f":d={cycle}:" not in line:
        raise CaptureError(f"f{lead:03d} DSWRF index row has the wrong forecast reference time")
    start, end = (int(value) for value in match.groups())
    if start < 0 or end <= start or end > 384 or end != lead:
        raise CaptureError(f"f{lead:03d} DSWRF interval has invalid lead bounds {start}-{end}")
    return {"start_lead_hours": start, "end_lead_hours": end}


def probe_leads(target_start_offset: int, target_end_offset: int) -> tuple[int, ...]:
    # A six-hour lookback finds the preceding accumulated window used by nested differencing.
    first = max(0, target_start_offset - 6)
    return tuple(range(first, target_end_offset + 1))


def _derive_segments(
    intervals: list[dict[str, int]],
    cycle_start: datetime,
    target_start_offset: int,
    target_end_offset: int,
) -> list[dict[str, Any]]:
    by_interval: dict[tuple[int, int], list[int]] = {}
    for item in intervals:
        start = item["start_lead_hours"]
        end = item["end_lead_hours"]
        if end > start:
            by_interval.setdefault((start, end), []).append(item["lead_end_hours"])
    normalized = [
        {"start": start, "end": end, "lead": min(leads)}
        for (start, end), leads in sorted(by_interval.items())
    ]

    def walk(cursor: int, seen: frozenset[int]) -> list[dict[str, Any]] | None:
        if cursor == target_end_offset:
            return []
        if cursor in seen:
            return None
        options: list[tuple[int, list[int], str]] = []
        for item in normalized:
            if item["start"] == cursor and cursor < item["end"] <= target_end_offset:
                options.append(
                    (item["end"], [item["lead"]], f"f{item['lead']:03d} direct interval")
                )
        for baseline in normalized:
            if baseline["end"] != cursor:
                continue
            for outer in normalized:
                if (
                    outer["start"] == baseline["start"]
                    and outer["end"] > cursor
                    and outer["end"] <= target_end_offset
                ):
                    options.append(
                        (
                            outer["end"],
                            [outer["lead"], baseline["lead"]],
                            f"f{outer['lead']:03d} minus f{baseline['lead']:03d}",
                        )
                    )
        for end, source_leads, method in sorted(
            options, key=lambda option: (option[0], len(option[1]))
        ):
            rest = walk(end, seen | {cursor})
            if rest is None:
                continue
            start_time = cycle_start + timedelta(hours=cursor)
            end_time = cycle_start + timedelta(hours=end)
            segment = {
                "start_lead_hours": cursor,
                "end_lead_hours": end,
                "interval_start_utc": timestamp_utc(start_time),
                "interval_end_utc": timestamp_utc(end_time),
                "source_leads": source_leads,
                "method": method,
            }
            return [segment, *rest]
        return None

    segments = walk(target_start_offset, frozenset())
    if segments is None:
        raise CaptureError(
            "decoded DSWRF intervals cannot reconstruct the complete target interval"
        )
    return segments


def _attempt_request(
    directory: Path,
    kind: str,
    lead: int,
    cycle: str,
    cutoff: datetime,
    *,
    opener: Callable[..., Any] | None,
    clock: Callable[[], datetime] | None,
    bounds: dict[str, str] = BOUNDS,
) -> dict[str, Any]:
    request = (
        build_cycle_index_request(lead, cycle)
        if kind == "index"
        else build_cycle_grib_request(lead, cycle, bounds)
    )
    now = clock or utc_now
    checked_at = parse_utc_datetime(now(), "capture clock")
    record: dict[str, Any] = {
        "kind": kind,
        "lead_end_hours": lead,
        "request": request,
        "request_started_utc": None,
        "body_received_utc": None,
        "http_status": None,
        "response_headers": None,
        "final_url": None,
        "attempt_checked_utc": timestamp_utc(checked_at),
        "error": None,
    }
    if checked_at > cutoff:
        record["error"] = "cutoff reached before request; no request sent"
        return record

    started = parse_utc_datetime(now(), "request start clock")
    if started > cutoff:
        record["attempt_checked_utc"] = timestamp_utc(started)
        record["error"] = "cutoff reached before request; no request sent"
        return record
    record["attempt_checked_utc"] = timestamp_utc(started)
    record["request_started_utc"] = timestamp_utc(started)
    req = urllib.request.Request(request["url"], headers={"User-Agent": USER_AGENT}, method="GET")
    open_url = opener or urllib.request.urlopen
    try:
        with open_url(req, timeout=60) as response:
            body = response.read()
            received = parse_utc_datetime(now(), "response body clock")
            record.update(
                {
                    "body_received_utc": timestamp_utc(received),
                    "http_status": int(response.status),
                    "response_headers": {
                        str(key): str(value) for key, value in response.headers.items()
                    },
                    "final_url": response.geturl(),
                }
            )
            suffix = "idx" if kind == "index" else "grib2"
            record.update(save_exclusive(directory / f"f{lead:03d}.{suffix}", body))
            if int(response.status) != 200:
                record["error"] = f"HTTP status {response.status}"
            elif received > cutoff:
                record["error"] = "response body received after cutoff"
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def write_json_exclusive(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        output.write((json.dumps(data, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def capture_candidate(
    cycle: str,
    cutoff: str | datetime,
    target_start: str | datetime,
    target_end: str | datetime,
    output_dir: Path,
    *,
    opener: Callable[..., Any] | None = None,
    clock: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    cycle_start = parse_cycle(cycle)
    cutoff_time = parse_utc_datetime(cutoff, "cutoff")
    start = parse_utc_datetime(target_start, "target_start")
    end = parse_utc_datetime(target_end, "target_end")
    start_offset, end_offset = _target_offsets(cycle_start, start, end)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=False)
    started = parse_utc_datetime((clock or utc_now)(), "capture clock")
    receipt: dict[str, Any] = {
        "schema_version": 2,
        "record_type": "gfs_candidate",
        "capture_status": "INCOMPLETE",
        "cycle": cycle,
        "forecast_reference_time_utc": timestamp_utc(cycle_start),
        "capture_started_utc": timestamp_utc(started),
        "cutoff_utc": timestamp_utc(cutoff_time),
        "target_interval_utc": [timestamp_utc(start), timestamp_utc(end)],
        "target_interval_convention": "half-open [start, end)",
        "source": {
            "provider": "NOAA NOMADS",
            "product": "GFS 0.25 degree pgrb2",
            "variable": "DSWRF",
            "level": "surface",
            "units": "W m-2",
        },
        "requested_point": REQUESTED_POINT,
        "expected_grid_point": GRID_POINT,
        "bounds": BOUNDS,
        "probe_leads": list(probe_leads(start_offset, end_offset)),
        "attempts": [],
        "required_leads": [],
        "target_reconstruction_segments": [],
        "eligible_before_cutoff": False,
        "failure_reason": None,
    }
    intervals: list[dict[str, int]] = []
    for lead in receipt["probe_leads"]:
        attempt = _attempt_request(
            output_dir, "index", lead, cycle, cutoff_time, opener=opener, clock=clock
        )
        if attempt["http_status"] == 200 and "file" in attempt:
            try:
                interval = decode_dswrf_interval(
                    (output_dir / attempt["file"]).read_bytes(), cycle, lead
                )
                attempt["decoded_interval"] = interval
                if (
                    interval["end_lead_hours"] < start_offset
                    or interval["start_lead_hours"] >= end_offset
                ):
                    attempt["error"] = "decoded interval is outside the target and boundary windows"
                else:
                    intervals.append({"lead_end_hours": lead, **interval})
            except CaptureError as exc:
                attempt["error"] = str(exc)
        elif attempt["error"] is None:
            attempt["error"] = "index response unavailable"
        receipt["attempts"].append(attempt)

    try:
        segments = _derive_segments(intervals, cycle_start, start_offset, end_offset)
        required_leads = sorted({lead for segment in segments for lead in segment["source_leads"]})
        receipt["target_reconstruction_segments"] = segments
        receipt["required_leads"] = required_leads
    except CaptureError as exc:
        receipt["failure_reason"] = str(exc)
        required_leads = []

    grib_ok: set[int] = set()
    for lead in required_leads:
        attempt = _attempt_request(
            output_dir,
            "grib2",
            lead,
            cycle,
            cutoff_time,
            opener=opener,
            clock=clock,
            bounds=receipt["bounds"],
        )
        index_interval = next(item for item in intervals if item["lead_end_hours"] == lead)
        if attempt["http_status"] == 200 and "file" in attempt:
            try:
                grid = validate_grib2_message(
                    (output_dir / attempt["file"]).read_bytes(),
                    lead,
                    cycle_start=cycle_start,
                    expected_interval=(
                        index_interval["start_lead_hours"],
                        index_interval["end_lead_hours"],
                    ),
                    expected_grid_point=receipt["expected_grid_point"],
                )
                attempt["decoded_interval"] = index_interval
                attempt["grid_definition"] = grid
                if attempt["error"] is None:
                    grib_ok.add(lead)
            except (CaptureError, KeyError, ValueError) as exc:
                attempt["error"] = str(exc)
        elif attempt["error"] is None:
            attempt["error"] = "GRIB2 response unavailable"
        receipt["attempts"].append(attempt)

    all_bodies_timely = all(
        parse_utc_datetime(attempt["body_received_utc"], "body_received_utc") <= cutoff_time
        for attempt in receipt["attempts"]
        if attempt["body_received_utc"] is not None
    )
    all_requests_timely = all(
        parse_utc_datetime(attempt["request_started_utc"], "request_started_utc") <= cutoff_time
        for attempt in receipt["attempts"]
        if attempt["request_started_utc"] is not None
    )
    receipt["all_response_bodies_received_by_cutoff"] = all_bodies_timely and all_requests_timely
    receipt["capture_completed_utc"] = timestamp_utc(
        parse_utc_datetime((clock or utc_now)(), "capture clock")
    )
    if required_leads and set(required_leads) == grib_ok:
        receipt["capture_status"] = "COMPLETE"
        if not receipt["all_response_bodies_received_by_cutoff"]:
            receipt["failure_reason"] = "one or more response bodies arrived after cutoff"
    elif receipt["failure_reason"] is None:
        receipt["failure_reason"] = "one or more required GRIB2 responses are missing or invalid"
    receipt["eligible_before_cutoff"] = (
        receipt["capture_status"] == "COMPLETE"
        and receipt["all_response_bodies_received_by_cutoff"]
    )
    write_json_exclusive(output_dir / "candidate_receipt.json", receipt)
    return receipt


def capture(
    cycle: str,
    cutoff: str | datetime,
    target_start: str | datetime,
    target_end: str | datetime,
    output_dir: Path,
    *,
    opener: Callable[..., Any] | None = None,
    clock: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Capture one cycle candidate under an explicit target and UTC cutoff contract."""
    return capture_candidate(
        cycle,
        cutoff,
        target_start,
        target_end,
        output_dir,
        opener=opener,
        clock=clock,
    )


def _validate_attempt_time(attempt: dict[str, Any], cutoff: datetime, lead: int) -> None:
    checked = attempt.get("attempt_checked_utc")
    if checked is None:
        raise CaptureError(f"f{lead:03d} attempt lacks a clock check timestamp")
    parse_utc_datetime(checked, "attempt_checked_utc")
    started = attempt.get("request_started_utc")
    received = attempt.get("body_received_utc")
    if started is None:
        if received is not None or attempt.get("error") is None:
            raise CaptureError(
                f"f{lead:03d} skipped request has inconsistent timestamp/error fields"
            )
        if any(
            attempt.get(field) is not None
            for field in ("http_status", "response_headers", "final_url")
        ):
            raise CaptureError(f"f{lead:03d} skipped request has response metadata")
        if parse_utc_datetime(checked, "attempt_checked_utc") <= cutoff:
            raise CaptureError(f"f{lead:03d} request was skipped before cutoff")
        return
    request_time = parse_utc_datetime(started, "request_started_utc")
    checked_time = parse_utc_datetime(checked, "attempt_checked_utc")
    if request_time > cutoff:
        raise CaptureError(f"f{lead:03d} request started after cutoff")
    if checked_time != request_time:
        raise CaptureError(f"f{lead:03d} request start disagrees with attempt clock")
    if received is None:
        if attempt.get("error") is None:
            raise CaptureError(f"f{lead:03d} failed response lacks an explicit error")
        return
    body_time = parse_utc_datetime(received, "body_received_utc")
    if body_time < request_time:
        raise CaptureError(f"f{lead:03d} body receipt precedes request start")
    if (
        not isinstance(attempt.get("http_status"), int)
        or isinstance(attempt.get("http_status"), bool)
        or not isinstance(attempt.get("response_headers"), dict)
        or not isinstance(attempt.get("final_url"), str)
    ):
        raise CaptureError(f"f{lead:03d} response receipt lacks HTTP metadata")


def _validate_candidate_attempt_file(
    output_dir: Path, attempt: dict[str, Any], kind: str, lead: int
) -> bytes | None:
    if "file" not in attempt:
        if attempt.get("body_received_utc") is not None:
            raise CaptureError(f"f{lead:03d} {kind} response timestamp has no saved body")
        return None
    suffix = "idx" if kind == "index" else "grib2"
    expected_name = f"f{lead:03d}.{suffix}"
    filename = attempt["file"]
    if Path(filename).name != filename or filename != expected_name:
        raise CaptureError(f"unsafe or unexpected candidate evidence filename {filename!r}")
    payload = (output_dir / filename).read_bytes()
    if len(payload) != attempt.get("size_bytes") or sha256(payload) != attempt.get("sha256"):
        raise CaptureError(f"candidate evidence size/hash mismatch for {filename}")
    if attempt.get("body_received_utc") is None:
        raise CaptureError(f"{filename} has no response body receipt timestamp")
    return payload


def verify_candidate(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "candidate_receipt.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("schema_version") != 2 or receipt.get("record_type") != "gfs_candidate":
        raise CaptureError("not a version-2 GFS candidate receipt")
    cycle = receipt.get("cycle")
    cycle_start = parse_cycle(cycle)
    cutoff = parse_utc_datetime(receipt.get("cutoff_utc"), "cutoff_utc")
    target_start, target_end = (
        parse_utc_datetime(value, "target_interval_utc")
        for value in receipt.get("target_interval_utc", [])
    )
    start_offset, end_offset = _target_offsets(cycle_start, target_start, target_end)
    expected_probes = list(probe_leads(start_offset, end_offset))
    if receipt.get("forecast_reference_time_utc") != timestamp_utc(cycle_start):
        raise CaptureError("candidate reference time disagrees with cycle")
    if receipt.get("target_interval_convention") != "half-open [start, end)":
        raise CaptureError("candidate target interval must use half-open semantics")
    if receipt.get("requested_point") != REQUESTED_POINT:
        raise CaptureError("candidate requested point does not match the fixed Bengaluru contract")
    if receipt.get("source") != {
        "provider": "NOAA NOMADS",
        "product": "GFS 0.25 degree pgrb2",
        "variable": "DSWRF",
        "level": "surface",
        "units": "W m-2",
    }:
        raise CaptureError("candidate source metadata disagrees with the capture contract")
    capture_started = parse_utc_datetime(receipt.get("capture_started_utc"), "capture_started_utc")
    capture_completed = parse_utc_datetime(
        receipt.get("capture_completed_utc"), "capture_completed_utc"
    )
    if capture_completed < capture_started:
        raise CaptureError("candidate completion precedes capture start")
    bounds = receipt.get("bounds")
    expected_point = receipt.get("expected_grid_point")
    if bounds != BOUNDS:
        raise CaptureError("candidate bounds do not match the fixed Bengaluru contract")
    if expected_point != GRID_POINT:
        raise CaptureError(
            "candidate expected grid point does not match the fixed Bengaluru contract"
        )
    if (
        not isinstance(bounds, dict)
        or set(bounds) != set(BOUNDS)
        or not all(isinstance(value, str) for value in bounds.values())
    ):
        raise CaptureError("candidate bounds do not match the recorded request contract")
    if (
        not isinstance(expected_point, dict)
        or set(expected_point) != set(GRID_POINT)
        or not all(
            isinstance(value, (int, float)) and math.isfinite(value)
            for value in expected_point.values()
        )
    ):
        raise CaptureError("candidate expected grid point is invalid")
    if receipt.get("probe_leads") != expected_probes:
        raise CaptureError("candidate probe leads disagree with cycle and target")
    attempts = receipt.get("attempts")
    if not isinstance(attempts, list):
        raise CaptureError("candidate attempts must be a list")
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    intervals: list[dict[str, int]] = []
    all_responses_timely = True
    for attempt in attempts:
        kind = attempt.get("kind")
        lead = attempt.get("lead_end_hours")
        if kind not in {"index", "grib2"} or not isinstance(lead, int):
            raise CaptureError("candidate attempt has invalid kind or lead")
        key = (kind, lead)
        if key in by_key:
            raise CaptureError(f"duplicate candidate attempt {key}")
        by_key[key] = attempt
        expected_request = (
            build_cycle_index_request(lead, cycle)
            if kind == "index"
            else build_cycle_grib_request(lead, cycle, receipt["bounds"])
        )
        if attempt.get("request") != expected_request:
            raise CaptureError(f"f{lead:03d} {kind} request disagrees with candidate contract")
        _validate_attempt_time(attempt, cutoff, lead)
        payload = _validate_candidate_attempt_file(output_dir, attempt, kind, lead)
        if attempt.get("request_started_utc") is None:
            continue
        if (
            attempt.get("body_received_utc") is not None
            and parse_utc_datetime(attempt["body_received_utc"], "body_received_utc") > cutoff
        ):
            all_responses_timely = False
        if payload is None:
            continue
        if attempt.get("http_status") != 200:
            if attempt.get("error") is None:
                raise CaptureError(f"f{lead:03d} non-200 response lacks explicit error")
            continue
        if kind == "index":
            try:
                decoded = decode_dswrf_interval(payload, cycle, lead)
            except CaptureError:
                if attempt.get("error") is None:
                    raise
            else:
                if attempt.get("decoded_interval") != decoded:
                    raise CaptureError(f"f{lead:03d} index interval disagrees with saved body")
                if (
                    decoded["end_lead_hours"] >= start_offset
                    and decoded["start_lead_hours"] < end_offset
                ):
                    intervals.append({"lead_end_hours": lead, **decoded})

    for lead in expected_probes:
        if ("index", lead) not in by_key:
            raise CaptureError(f"candidate lacks index attempt for probe lead f{lead:03d}")
    try:
        segments = _derive_segments(intervals, cycle_start, start_offset, end_offset)
        required_leads = sorted({lead for segment in segments for lead in segment["source_leads"]})
    except CaptureError:
        segments = []
        required_leads = []
    if receipt.get("target_reconstruction_segments") != segments:
        raise CaptureError("candidate target reconstruction disagrees with decoded index intervals")
    if receipt.get("required_leads") != required_leads:
        raise CaptureError("candidate required leads disagree with decoded index intervals")
    expected_attempt_keys = {("index", lead) for lead in expected_probes} | {
        ("grib2", lead) for lead in required_leads
    }
    if set(by_key) != expected_attempt_keys:
        raise CaptureError("candidate attempt set disagrees with probes and required records")
    expected_files = {"candidate_receipt.json"} | {
        attempt["file"] for attempt in attempts if "file" in attempt
    }
    actual_entries = {item.name for item in output_dir.iterdir()}
    if actual_entries != expected_files:
        raise CaptureError("candidate directory contains missing or unrecorded evidence files")

    grib_ok: set[int] = set()
    for lead in required_leads:
        attempt = by_key.get(("grib2", lead))
        if attempt is None:
            continue
        payload = _validate_candidate_attempt_file(output_dir, attempt, "grib2", lead)
        if payload is None or attempt.get("http_status") != 200:
            continue
        interval = next(item for item in intervals if item["lead_end_hours"] == lead)
        grid = validate_grib2_message(
            payload,
            lead,
            cycle_start=cycle_start,
            expected_interval=(interval["start_lead_hours"], interval["end_lead_hours"]),
            expected_grid_point=receipt["expected_grid_point"],
        )
        if attempt.get("decoded_interval") != interval or attempt.get("grid_definition") != grid:
            raise CaptureError(f"f{lead:03d} GRIB2 metadata disagrees with saved body/index")
        if (
            attempt.get("error") is None
            and parse_utc_datetime(attempt["body_received_utc"], "body_received_utc") <= cutoff
        ):
            grib_ok.add(lead)
    complete = bool(required_leads) and grib_ok == set(required_leads)
    if (receipt.get("capture_status") == "COMPLETE") != complete:
        raise CaptureError("candidate status disagrees with verified required records")
    if receipt.get("capture_status") not in {"COMPLETE", "INCOMPLETE"}:
        raise CaptureError("candidate has an unknown capture status")
    if receipt.get("all_response_bodies_received_by_cutoff") is not all_responses_timely:
        raise CaptureError("candidate response timing disagrees with cutoff")
    expected_eligibility = all_responses_timely and complete
    if receipt.get("eligible_before_cutoff") is not expected_eligibility:
        raise CaptureError("candidate eligibility disagrees with verified cutoff and data status")
    return {
        "verification_status": "PASS",
        "candidate_status": receipt["capture_status"],
        "eligible_before_cutoff": expected_eligibility,
        "cycle": cycle,
        "required_leads": required_leads,
        "target_reconstruction_segments": segments,
    }


def verify_capture(output_dir: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    candidate_receipt = output_dir / "candidate_receipt.json"
    if candidate_receipt.exists():
        return verify_candidate(output_dir)
    receipt = json.loads((output_dir / "capture_receipt.json").read_text(encoding="utf-8"))
    if receipt.get("schema_version") == 1:
        return verify_legacy_capture(output_dir)
    raise CaptureError("unrecognized capture receipt contract")


def _selection_evaluation(
    root: Path,
    issue_day: str,
    cutoff: datetime,
    target_start: datetime,
    target_end: datetime,
) -> dict[str, Any]:
    if re.fullmatch(r"\d{8}", issue_day or "") is None:
        raise CaptureError("issue day must use YYYYMMDD format")
    try:
        issue_date = datetime.strptime(issue_day, "%Y%m%d").date()
    except ValueError as exc:
        raise CaptureError("issue day must use YYYYMMDD format") from exc
    if cutoff.date() != issue_date:
        raise CaptureError("cutoff must fall on the selected issue day in UTC")
    scheduled_cycles = [
        datetime.combine(issue_date, datetime.min.time(), tzinfo=UTC) + timedelta(hours=hour)
        for hour in (0, 6, 12, 18)
    ]
    candidate_results: list[dict[str, Any]] = []
    eligible: list[tuple[datetime, Path, dict[str, Any]]] = []
    for receipt_path in sorted(root.glob("*/candidate_receipt.json")):
        candidate_dir = receipt_path.parent
        try:
            result = verify_candidate(candidate_dir)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            cycle_start = parse_cycle(receipt["cycle"])
            candidate_valid = (
                cycle_start in scheduled_cycles
                and cycle_start <= cutoff
                and receipt.get("cutoff_utc") == timestamp_utc(cutoff)
                and receipt.get("target_interval_utc")
                == [timestamp_utc(target_start), timestamp_utc(target_end)]
            )
            is_eligible = candidate_valid and result["eligible_before_cutoff"]
            candidate_results.append(
                {
                    "candidate_directory": candidate_dir.name,
                    "cycle": receipt.get("cycle"),
                    "capture_status": result["candidate_status"],
                    "eligible": is_eligible,
                    "contract_match": candidate_valid,
                }
            )
            if is_eligible:
                eligible.append((cycle_start, candidate_dir, result))
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            candidate_results.append(
                {
                    "candidate_directory": candidate_dir.name,
                    "eligible": False,
                    "verification_error": f"{type(exc).__name__}: {exc}",
                }
            )
    selected = max(eligible, key=lambda item: item[0]) if eligible else None
    return {
        "selected": selected,
        "candidate_results": candidate_results,
        "scheduled_cycles": [timestamp_utc(item) for item in scheduled_cycles],
    }


def select_latest_candidate(
    root: Path,
    issue_day: str,
    cutoff: str | datetime,
    target_start: str | datetime,
    target_end: str | datetime,
    output_path: Path,
) -> dict[str, Any]:
    cutoff_time = parse_utc_datetime(cutoff, "cutoff")
    start = parse_utc_datetime(target_start, "target_start")
    end = parse_utc_datetime(target_end, "target_end")
    if end <= start:
        raise CaptureError("target interval must be a non-empty half-open interval")
    if output_path.parent.resolve() != root.resolve():
        raise CaptureError("selection receipt output must be directly inside candidate root")
    evaluated = _selection_evaluation(root, issue_day, cutoff_time, start, end)
    selected = evaluated["selected"]
    receipt = {
        "schema_version": 2,
        "record_type": "gfs_selection",
        "selection_status": "SELECTED" if selected else "CUTOFF_MISS",
        "issue_day": issue_day,
        "cutoff_utc": timestamp_utc(cutoff_time),
        "target_interval_utc": [timestamp_utc(start), timestamp_utc(end)],
        "scheduled_cycles": evaluated["scheduled_cycles"],
        "candidate_results": evaluated["candidate_results"],
        "selected_cycle": selected[0].strftime("%Y%m%d%H") if selected else None,
        "selected_candidate_directory": selected[1].name if selected else None,
        "failure_reason": None
        if selected
        else "no complete eligible candidate was received by cutoff",
    }
    write_json_exclusive(output_path, receipt)
    return receipt


def verify_selection(selection_path: Path) -> dict[str, Any]:
    receipt = json.loads(selection_path.read_text(encoding="utf-8"))
    if receipt.get("schema_version") != 2 or receipt.get("record_type") != "gfs_selection":
        raise CaptureError("not a version-2 GFS selection receipt")
    cutoff = parse_utc_datetime(receipt.get("cutoff_utc"), "cutoff_utc")
    start, end = (
        parse_utc_datetime(value, "target_interval_utc")
        for value in receipt.get("target_interval_utc", [])
    )
    evaluated = _selection_evaluation(
        selection_path.parent,
        receipt.get("issue_day"),
        cutoff,
        start,
        end,
    )
    selected = evaluated["selected"]
    expected_status = "SELECTED" if selected else "CUTOFF_MISS"
    expected_cycle = selected[0].strftime("%Y%m%d%H") if selected else None
    expected_dir = selected[1].name if selected else None
    if (
        receipt.get("selection_status") != expected_status
        or receipt.get("selected_cycle") != expected_cycle
        or receipt.get("selected_candidate_directory") != expected_dir
        or receipt.get("scheduled_cycles") != evaluated["scheduled_cycles"]
        or receipt.get("candidate_results") != evaluated["candidate_results"]
        or receipt.get("failure_reason")
        != (None if selected else "no complete eligible candidate was received by cutoff")
    ):
        raise CaptureError("selection receipt disagrees with saved eligible candidate receipts")
    return {
        "verification_status": "PASS",
        "selection_status": expected_status,
        "selected_cycle": expected_cycle,
        "candidate_count": len(evaluated["candidate_results"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture_parser = subparsers.add_parser(
        "capture", help="capture one cycle candidate without sending requests after cutoff"
    )
    capture_parser.add_argument("--cycle", required=True, help="GFS cycle as YYYYMMDDHH UTC")
    capture_parser.add_argument("--cutoff", required=True, help="ISO-8601 UTC receipt cutoff")
    capture_parser.add_argument("--target-start", required=True, help="ISO-8601 UTC start")
    capture_parser.add_argument("--target-end", required=True, help="ISO-8601 UTC end")
    capture_parser.add_argument(
        "--output", type=Path, required=True, help="new candidate directory"
    )

    select_parser = subparsers.add_parser(
        "select", help="select latest eligible issue-day cycle from saved candidate receipts"
    )
    select_parser.add_argument("--issue-day", required=True, help="issue date as YYYYMMDD UTC")
    select_parser.add_argument("--cutoff", required=True, help="ISO-8601 UTC receipt cutoff")
    select_parser.add_argument("--target-start", required=True, help="ISO-8601 UTC start")
    select_parser.add_argument("--target-end", required=True, help="ISO-8601 UTC end")
    select_parser.add_argument("--root", type=Path, required=True, help="candidate root directory")
    select_parser.add_argument("--output", type=Path, required=True, help="new selection receipt")

    verify_parser = subparsers.add_parser("verify", help="verify saved receipt(s) without network")
    verify_target = verify_parser.add_mutually_exclusive_group(required=True)
    verify_target.add_argument(
        "--directory", type=Path, help="candidate or legacy capture directory"
    )
    verify_target.add_argument("--selection", type=Path, help="selection receipt path")
    args = parser.parse_args(argv)
    try:
        if args.command == "capture":
            receipt = capture(
                args.cycle,
                args.cutoff,
                args.target_start,
                args.target_end,
                args.output,
            )
            print(
                json.dumps(
                    {
                        "capture_status": receipt["capture_status"],
                        "eligible_before_cutoff": receipt["eligible_before_cutoff"],
                        "receipt": str(args.output / "candidate_receipt.json"),
                    }
                )
            )
            return (
                0
                if receipt["capture_status"] == "COMPLETE" and receipt["eligible_before_cutoff"]
                else 2
            )
        if args.command == "select":
            receipt = select_latest_candidate(
                args.root,
                args.issue_day,
                args.cutoff,
                args.target_start,
                args.target_end,
                args.output,
            )
            print(json.dumps(receipt, indent=2))
            return 0 if receipt["selection_status"] == "SELECTED" else 2
        if args.selection is not None:
            print(json.dumps(verify_selection(args.selection), indent=2))
        else:
            print(json.dumps(verify_capture(args.directory), indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"{args.command.upper()} FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
