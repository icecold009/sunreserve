from __future__ import annotations

import json
import re
import struct
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest
from scripts import capture_gfs_prospective as capture_script


def fake_index(lead: int, *, interval: tuple[int, int] | None = None) -> bytes:
    start, end = interval or capture_script.LEAD_INTERVALS[lead]
    return f"1:0:d=2026112112:DSWRF:surface:{start}-{end} hour ave fcst:\n".encode()


def _section(number: int, length: int, *, points: int | None = None) -> bytes:
    section = bytearray(length)
    section[:4] = length.to_bytes(4, "big")
    section[4] = number
    if number == 3:
        section[6:10] = (points or 1).to_bytes(4, "big")
        section[12:14] = (0).to_bytes(2, "big")
        section[14] = 6
        section[30:34] = (1).to_bytes(4, "big")
        section[34:38] = (1).to_bytes(4, "big")
        section[46:50] = (13_000_000).to_bytes(4, "big", signed=True)
        section[50:54] = (77_500_000).to_bytes(4, "big", signed=True)
        section[54] = 0x30
        section[55:59] = (13_000_000).to_bytes(4, "big", signed=True)
        section[59:63] = (77_500_000).to_bytes(4, "big", signed=True)
        section[63:67] = (250_000).to_bytes(4, "big")
        section[67:71] = (250_000).to_bytes(4, "big")
        section[71] = 0x40
    return bytes(section)


def fake_grib2(*, length_delta: int = 0, trailer: bytes = b"7777", point_count: int = 1) -> bytes:
    sections = b"".join(
        (
            _section(1, 21),
            _section(3, 72, points=point_count),
            _section(4, 58),
            _section(5, 21),
            _section(6, 6),
            _section(7, 5),
        )
    )
    payload = bytearray(b"GRIB" + b"\x00\x00\x00\x02" + b"\x00" * 8 + sections + trailer)
    payload[8:16] = struct.pack(">Q", len(payload) + length_delta)
    return bytes(payload)


class FakeResponse:
    status = 200
    headers = {"Content-Type": "application/octet-stream", "ETag": '"fixture"'}

    def __init__(self, url: str, body: bytes) -> None:
        self.url = url
        self.body = body

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body

    def geturl(self) -> str:
        return self.url


def fake_opener(*, bad_index_lead: int | None = None, bad_grib_lead: int | None = None):
    calls: list[str] = []

    def open_url(request, *, timeout: int) -> FakeResponse:
        assert timeout == 60
        url = request.full_url
        calls.append(url)
        parsed = urlparse(url)
        if parsed.path.endswith(".idx"):
            match = re.search(r"\.f(\d{3})\.idx$", parsed.path)
            assert match is not None
            lead = int(match.group(1))
            interval = (36, 39) if lead == bad_index_lead else None
            return FakeResponse(url, fake_index(lead, interval=interval))
        query = parse_qs(parsed.query)
        match = re.search(r"\.f(\d{3})$", query["file"][0])
        assert match is not None
        lead = int(match.group(1))
        payload = fake_grib2(length_delta=1) if lead == bad_grib_lead else fake_grib2()
        return FakeResponse(url, payload)

    return open_url, calls


def fixed_clock() -> datetime:
    return datetime(2026, 11, 22, 0, 0, tzinfo=UTC)


def test_idx_intervals_and_target_reconstruction_contract() -> None:
    for lead, interval in capture_script.LEAD_INTERVALS.items():
        assert capture_script.validate_index(fake_index(lead), lead) == {
            "start_lead_hours": interval[0],
            "end_lead_hours": interval[1],
        }
    segments = capture_script.target_segments()
    capture_script.validate_segment_coverage(segments)
    assert len(segments) == 8
    assert segments[0]["source_leads"] == [42, 39]


def test_index_with_wrong_actual_interval_is_rejected() -> None:
    with pytest.raises(capture_script.CaptureError, match="expected 36-42h"):
        capture_script.validate_index(fake_index(42, interval=(36, 39)), 42)


def test_grib2_requires_one_complete_message() -> None:
    grid = capture_script.validate_grib2_message(fake_grib2(), 39)
    assert grid["number_of_data_points"] == 1
    assert grid["latitude_first_degrees_north"] == 13.0
    assert grid["longitude_first_degrees_east"] == 77.5
    with pytest.raises(capture_script.CaptureError, match="length field"):
        capture_script.validate_grib2_message(fake_grib2(length_delta=1), 39)
    with pytest.raises(capture_script.CaptureError, match="7777"):
        capture_script.validate_grib2_message(fake_grib2(trailer=b"NOPE"), 39)
    with pytest.raises(capture_script.CaptureError, match="not one grid point"):
        capture_script.validate_grib2_message(fake_grib2(point_count=2), 39)


def test_capture_and_offline_verification(tmp_path: Path) -> None:
    output_dir = tmp_path / "capture"
    opener, calls = fake_opener()
    receipt = capture_script.capture(output_dir, opener=opener, clock=fixed_clock)

    assert receipt["capture_status"] == "COMPLETE"
    assert len(calls) == 18
    assert len(receipt["records"]) == 9
    assert receipt["latest_cycle_claim"] is False
    assert receipt["issue_cutoff_availability"] == "NOT ESTABLISHED"
    assert receipt["operational_result_computed"] is False
    assert capture_script.verify_capture(output_dir)["verification_status"] == "PASS"

    with pytest.raises(FileExistsError, match="refusing overwrite"):
        capture_script.capture(output_dir, opener=opener, clock=fixed_clock)
    assert len(calls) == 18


def test_capture_rejects_bad_index_before_requesting_grib(tmp_path: Path) -> None:
    output_dir = tmp_path / "bad-index"
    opener, calls = fake_opener(bad_index_lead=42)
    with pytest.raises(capture_script.CaptureError, match="expected 36-42h"):
        capture_script.capture(output_dir, opener=opener, clock=fixed_clock)

    assert all(urlparse(url).path.endswith(".idx") for url in calls)
    receipt = json.loads((output_dir / "capture_receipt.json").read_text(encoding="utf-8"))
    assert receipt["capture_status"] == "INCOMPLETE"


def test_offline_verifier_detects_tampered_saved_response(tmp_path: Path) -> None:
    output_dir = tmp_path / "capture"
    opener, _ = fake_opener()
    capture_script.capture(output_dir, opener=opener, clock=fixed_clock)
    grib_path = output_dir / "f039.grib2"
    grib_path.write_bytes(grib_path.read_bytes() + b"tampered")

    with pytest.raises(capture_script.CaptureError, match="size/hash mismatch"):
        capture_script.verify_capture(output_dir)
