from __future__ import annotations

import io
import json
import re
import struct
import urllib.error
from datetime import UTC, datetime, timedelta
from email.message import Message
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import pytest
from scripts import capture_gfs_prospective as capture_script


def fake_index(
    lead: int,
    *,
    interval: tuple[int, int] | None = None,
    cycle: str = capture_script.CYCLE,
) -> bytes:
    start, end = interval or capture_script.LEAD_INTERVALS[lead]
    return f"1:0:d={cycle}:DSWRF:surface:{start}-{end} hour ave fcst:\n".encode()


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


def fake_grib2(
    lead: int = 39,
    *,
    length_delta: int = 0,
    trailer: bytes = b"7777",
    point_count: int = 1,
    first_surface_type: int = 1,
    increment_type: int = 2,
    cycle_start: datetime = capture_script.CYCLE_START,
    interval: tuple[int, int] | None = None,
) -> bytes:
    identification = bytearray(_section(1, 21))
    identification[12:14] = cycle_start.year.to_bytes(2, "big")
    identification[14:19] = bytes(
        (
            cycle_start.month,
            cycle_start.day,
            cycle_start.hour,
            cycle_start.minute,
            cycle_start.second,
        )
    )

    start_hour, end_hour = interval or capture_script.LEAD_INTERVALS[lead]
    interval_end = cycle_start + timedelta(hours=end_hour)
    product = bytearray(_section(4, 58))
    product[7:9] = (8).to_bytes(2, "big")
    product[9:11] = bytes((4, 192))
    product[22] = first_surface_type
    product[28] = 255
    product[17] = 1
    product[18:22] = start_hour.to_bytes(4, "big")
    product[34:36] = interval_end.year.to_bytes(2, "big")
    product[36:41] = bytes(
        (
            interval_end.month,
            interval_end.day,
            interval_end.hour,
            interval_end.minute,
            interval_end.second,
        )
    )
    product[41] = 1
    product[46] = 0
    product[47] = increment_type
    product[48] = 1
    product[49:53] = (end_hour - start_hour).to_bytes(4, "big")
    product[53] = 255
    product[54:58] = (0).to_bytes(4, "big")
    sections = b"".join(
        (
            bytes(identification),
            _section(3, 72, points=point_count),
            bytes(product),
            _section(5, 21),
            _section(6, 6),
            _section(7, 5),
        )
    )
    payload = bytearray(b"GRIB" + b"\x00\x00\x00\x02" + b"\x00" * 8 + sections + trailer)
    payload[8:16] = struct.pack(">Q", len(payload) + length_delta)
    return bytes(payload)


class FakeResponse:
    headers = {"Content-Type": "application/octet-stream", "ETag": '"fixture"'}

    def __init__(self, url: str, body: bytes, status: int = 200) -> None:
        self.url = url
        self.body = body
        self.status = status

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
        payload = fake_grib2(lead, length_delta=1) if lead == bad_grib_lead else fake_grib2(lead)
        return FakeResponse(url, payload)

    return open_url, calls


def interval_map(start_offset: int, end_offset: int) -> dict[int, tuple[int, int]]:
    assert (end_offset - start_offset) % 3 == 0
    intervals = {
        start_offset: (start_offset - 3, start_offset),
        start_offset + 3: (start_offset - 3, start_offset + 3),
    }
    steps = (end_offset - start_offset) // 3
    for step in range(1, steps):
        lead = start_offset + 3 * (step + 1)
        if step % 2:
            intervals[lead] = (start_offset + 3 * step, lead)
        else:
            intervals[lead] = (start_offset + 3 * (step - 1), lead)
    return intervals


def candidate_opener(cycle: str, intervals: dict[int, tuple[int, int]]):
    calls: list[str] = []
    cycle_start = capture_script.parse_cycle(cycle)

    def open_url(request, *, timeout: int) -> FakeResponse:
        assert timeout == 60
        url = request.full_url
        calls.append(url)
        parsed = urlparse(url)
        if parsed.path.endswith(".idx"):
            match = re.search(r"\.f(\d{3})\.idx$", parsed.path)
            assert match is not None
            lead = int(match.group(1))
            interval = intervals.get(lead)
            if interval is None:
                return FakeResponse(url, b"not available", status=404)
            return FakeResponse(url, fake_index(lead, interval=interval, cycle=cycle))
        query = parse_qs(parsed.query)
        match = re.search(r"\.f(\d{3})$", query["file"][0])
        assert match is not None
        lead = int(match.group(1))
        return FakeResponse(
            url,
            fake_grib2(lead, cycle_start=cycle_start, interval=intervals[lead]),
        )

    return open_url, calls


class SequenceClock:
    def __init__(self, values: list[datetime], fallback: datetime) -> None:
        self.values = values
        self.fallback = fallback

    def __call__(self) -> datetime:
        return self.values.pop(0) if self.values else self.fallback


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
    grid = capture_script.validate_grib2_message(fake_grib2(39), 39)
    assert grid["number_of_data_points"] == 1
    assert grid["latitude_first_degrees_north"] == 13.0
    assert grid["longitude_first_degrees_east"] == 77.5
    with pytest.raises(capture_script.CaptureError, match="length field"):
        capture_script.validate_grib2_message(fake_grib2(39, length_delta=1), 39)
    with pytest.raises(capture_script.CaptureError, match="7777"):
        capture_script.validate_grib2_message(fake_grib2(39, trailer=b"NOPE"), 39)
    with pytest.raises(capture_script.CaptureError, match="not one grid point"):
        capture_script.validate_grib2_message(fake_grib2(39, point_count=2), 39)


def test_grib2_in_file_interval_must_match_record_lead() -> None:
    with pytest.raises(capture_script.CaptureError, match="forecast period starts at 60"):
        capture_script.validate_grib2_message(fake_grib2(63), 39)


def test_grib2_in_file_surface_must_match_surface_level_contract() -> None:
    with pytest.raises(capture_script.CaptureError, match="fixed surfaces"):
        capture_script.validate_grib2_message(fake_grib2(39, first_surface_type=100), 39)


def test_grib2_in_file_increment_descriptor_must_match_interval_contract() -> None:
    with pytest.raises(capture_script.CaptureError, match="time-range metadata"):
        capture_script.validate_grib2_message(fake_grib2(39, increment_type=1), 39)


def test_capture_and_offline_verification(tmp_path: Path) -> None:
    output_dir = tmp_path / "capture"
    opener, calls = fake_opener()
    receipt = capture_script._capture_legacy(output_dir, opener=opener, clock=fixed_clock)

    assert receipt["capture_status"] == "COMPLETE"
    assert len(calls) == 18
    assert len(receipt["records"]) == 9
    assert receipt["latest_cycle_claim"] is False
    assert receipt["issue_cutoff_availability"] == "NOT ESTABLISHED"
    assert receipt["operational_result_computed"] is False
    assert capture_script.verify_capture(output_dir)["verification_status"] == "PASS"

    with pytest.raises(FileExistsError, match="refusing overwrite"):
        capture_script._capture_legacy(output_dir, opener=opener, clock=fixed_clock)
    assert len(calls) == 18


def test_capture_rejects_bad_index_before_requesting_grib(tmp_path: Path) -> None:
    output_dir = tmp_path / "bad-index"
    opener, calls = fake_opener(bad_index_lead=42)
    with pytest.raises(capture_script.CaptureError, match="expected 36-42h"):
        capture_script._capture_legacy(output_dir, opener=opener, clock=fixed_clock)

    assert all(urlparse(url).path.endswith(".idx") for url in calls)
    receipt = json.loads((output_dir / "capture_receipt.json").read_text(encoding="utf-8"))
    assert receipt["capture_status"] == "INCOMPLETE"


def test_offline_verifier_detects_tampered_saved_response(tmp_path: Path) -> None:
    output_dir = tmp_path / "capture"
    opener, _ = fake_opener()
    capture_script._capture_legacy(output_dir, opener=opener, clock=fixed_clock)
    grib_path = output_dir / "f039.grib2"
    grib_path.write_bytes(grib_path.read_bytes() + b"tampered")

    with pytest.raises(capture_script.CaptureError, match="size/hash mismatch"):
        capture_script.verify_capture(output_dir)


def test_offline_verifier_rejects_swapped_lead_after_hash_update(tmp_path: Path) -> None:
    output_dir = tmp_path / "capture"
    opener, _ = fake_opener()
    capture_script._capture_legacy(output_dir, opener=opener, clock=fixed_clock)
    receipt_path = output_dir / "capture_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    f039 = output_dir / "f039.grib2"
    f063_bytes = (output_dir / "f063.grib2").read_bytes()
    f039.write_bytes(f063_bytes)
    f039_record = receipt["records"][0]["grib2_response"]
    f039_record["size_bytes"] = len(f063_bytes)
    f039_record["sha256"] = capture_script.sha256(f063_bytes)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(capture_script.CaptureError, match="forecast period starts at 60"):
        capture_script.verify_capture(output_dir)


@pytest.mark.parametrize(
    ("started", "received", "expected_error"),
    [
        (None, "2026-11-22T00:00:00Z", "missing valid request_started_utc"),
        (
            "2026-11-22T05:30:00+05:30",
            "2026-11-22T00:00:00Z",
            "must have a UTC timezone",
        ),
        (
            "2026-11-22T00:00:01Z",
            "2026-11-22T00:00:00Z",
            "body receipt precedes request start",
        ),
    ],
)
def test_offline_verifier_rejects_invalid_receipt_timestamps(
    tmp_path: Path, started: str | None, received: str, expected_error: str
) -> None:
    output_dir = tmp_path / "capture"
    opener, _ = fake_opener()
    capture_script._capture_legacy(output_dir, opener=opener, clock=fixed_clock)
    receipt_path = output_dir / "capture_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    idx_record = receipt["records"][0]["idx_response"]
    if started is None:
        idx_record.pop("request_started_utc")
    else:
        idx_record["request_started_utc"] = started
    idx_record["body_received_utc"] = received
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(capture_script.CaptureError, match=expected_error):
        capture_script.verify_capture(output_dir)


def test_candidate_capture_derives_leads_for_arbitrary_cycle_and_target(tmp_path: Path) -> None:
    cycle = "2026112206"
    target_start = datetime(2026, 11, 23, 3, tzinfo=UTC)
    target_end = datetime(2026, 11, 24, 3, tzinfo=UTC)
    start_offset, end_offset = 21, 45
    intervals = interval_map(start_offset, end_offset)
    opener, calls = candidate_opener(cycle, intervals)
    clock_value = datetime(2026, 11, 22, 7, tzinfo=UTC)
    output_dir = tmp_path / "candidate-06"

    receipt = capture_script.capture(
        cycle,
        "2026-11-22T12:30:00Z",
        target_start,
        target_end,
        output_dir,
        opener=opener,
        clock=lambda: clock_value,
    )

    assert receipt["capture_status"] == "COMPLETE"
    assert receipt["cycle"] == cycle
    assert receipt["target_interval_utc"] == [
        "2026-11-23T03:00:00.000000Z",
        "2026-11-24T03:00:00.000000Z",
    ]
    assert receipt["required_leads"] == sorted(intervals)
    assert receipt["target_reconstruction_segments"][0]["method"] == "f024 minus f021"
    assert capture_script.verify_candidate(output_dir)["verification_status"] == "PASS"
    assert len(calls) == len(receipt["probe_leads"]) + len(intervals)
    assert all("gfs.20261122/06/atmos" in unquote(url) for url in calls)
    with pytest.raises(FileExistsError):
        capture_script.capture(
            cycle,
            "2026-11-22T12:30:00Z",
            target_start,
            target_end,
            output_dir,
            opener=opener,
            clock=lambda: clock_value,
        )


def test_candidate_logs_late_response_and_sends_no_later_request(tmp_path: Path) -> None:
    cycle = "2026112206"
    intervals = interval_map(21, 45)
    opener, calls = candidate_opener(cycle, intervals)
    cutoff = datetime(2026, 11, 22, 12, 30, tzinfo=UTC)
    before = datetime(2026, 11, 22, 12, 29, tzinfo=UTC)
    after = datetime(2026, 11, 22, 12, 31, tzinfo=UTC)
    clock = SequenceClock([before, before, before, after], after)
    output_dir = tmp_path / "late-candidate"

    receipt = capture_script.capture_candidate(
        cycle,
        cutoff,
        datetime(2026, 11, 23, 3, tzinfo=UTC),
        datetime(2026, 11, 24, 3, tzinfo=UTC),
        output_dir,
        opener=opener,
        clock=clock,
    )

    assert len(calls) == 1
    assert receipt["capture_status"] == "INCOMPLETE"
    assert receipt["eligible_before_cutoff"] is False
    assert receipt["attempts"][0]["body_received_utc"] == "2026-11-22T12:31:00.000000Z"
    assert receipt["attempts"][0]["file"] == "f015.idx"
    assert all(
        attempt["request_started_utc"] is None and "no request sent" in attempt["error"]
        for attempt in receipt["attempts"][1:]
    )
    assert capture_script.verify_candidate(output_dir)["verification_status"] == "PASS"


def test_candidate_logs_unavailable_responses_and_selector_writes_cutoff_miss(
    tmp_path: Path,
) -> None:
    cycle = "2026112206"

    def unavailable(request, *, timeout: int) -> FakeResponse:
        return FakeResponse(request.full_url, b"temporary error", status=503)

    output_dir = tmp_path / "candidates" / "candidate-06"
    receipt = capture_script.capture_candidate(
        cycle,
        "2026-11-22T12:30:00Z",
        "2026-11-23T03:00:00Z",
        "2026-11-24T03:00:00Z",
        output_dir,
        opener=unavailable,
        clock=lambda: datetime(2026, 11, 22, 7, tzinfo=UTC),
    )
    assert receipt["capture_status"] == "INCOMPLETE"
    assert receipt["attempts"][0]["error"] == "HTTP status 503"
    assert "file" in receipt["attempts"][0]
    assert capture_script.verify_candidate(output_dir)["candidate_status"] == "INCOMPLETE"

    selection_path = output_dir.parent / "selection.json"
    selection = capture_script.select_latest_candidate(
        output_dir.parent,
        "20261122",
        "2026-11-22T12:30:00Z",
        "2026-11-23T03:00:00Z",
        "2026-11-24T03:00:00Z",
        selection_path,
    )
    assert selection["selection_status"] == "CUTOFF_MISS"
    assert selection["failure_reason"]
    assert capture_script.verify_selection(selection_path)["verification_status"] == "PASS"
    with pytest.raises(FileExistsError):
        capture_script.select_latest_candidate(
            output_dir.parent,
            "20261122",
            "2026-11-22T12:30:00Z",
            "2026-11-23T03:00:00Z",
            "2026-11-24T03:00:00Z",
            selection_path,
        )


def test_selector_freezes_latest_complete_scheduled_cycle(tmp_path: Path) -> None:
    root = tmp_path / "candidates"
    target_start = datetime(2026, 11, 23, 3, tzinfo=UTC)
    target_end = datetime(2026, 11, 24, 3, tzinfo=UTC)
    cutoff = "2026-11-22T12:30:00Z"
    for cycle in ("2026112200", "2026112206", "2026112212", "2026112218"):
        cycle_start = capture_script.parse_cycle(cycle)
        start_offset = int((target_start - cycle_start).total_seconds() // 3600)
        end_offset = int((target_end - cycle_start).total_seconds() // 3600)
        intervals = interval_map(start_offset, end_offset)
        opener, _ = candidate_opener(cycle, intervals)
        clock_value = (
            datetime(2026, 11, 22, 18, tzinfo=UTC)
            if cycle.endswith("18")
            else datetime(2026, 11, 22, 12, 5, tzinfo=UTC)
        )
        capture_script.capture_candidate(
            cycle,
            cutoff,
            target_start,
            target_end,
            root / f"cycle-{cycle}",
            opener=opener,
            clock=lambda clock_value=clock_value: clock_value,
        )

    selection_path = root / "selection.json"
    receipt = capture_script.select_latest_candidate(
        root,
        "20261122",
        cutoff,
        target_start,
        target_end,
        selection_path,
    )
    assert receipt["selection_status"] == "SELECTED"
    assert receipt["selected_cycle"] == "2026112212"
    assert capture_script.verify_selection(selection_path)["selected_cycle"] == "2026112212"


def test_candidate_logs_unavailable_network_response_without_inventing_receipt(
    tmp_path: Path,
) -> None:
    def unavailable(request, *, timeout: int) -> FakeResponse:
        raise OSError("connection unavailable")

    output_dir = tmp_path / "unavailable"
    receipt = capture_script.capture(
        "2026112206",
        "2026-11-22T12:30:00Z",
        "2026-11-23T03:00:00Z",
        "2026-11-24T03:00:00Z",
        output_dir,
        opener=unavailable,
        clock=lambda: datetime(2026, 11, 22, 7, tzinfo=UTC),
    )
    attempt = receipt["attempts"][0]
    assert receipt["capture_status"] == "INCOMPLETE"
    assert attempt["request_started_utc"] == "2026-11-22T07:00:00.000000Z"
    assert attempt["body_received_utc"] is None
    assert attempt["http_status"] is None
    assert "OSError: connection unavailable" == attempt["error"]
    assert "file" not in attempt
    assert capture_script.verify_candidate(output_dir)["candidate_status"] == "INCOMPLETE"


def test_candidate_logs_http_error_response_body_and_metadata(tmp_path: Path) -> None:
    cycle = "2026112206"
    intervals = interval_map(21, 45)
    delegate, _ = candidate_opener(cycle, intervals)
    body = b"NOMADS unavailable"
    headers = Message()
    headers["Content-Type"] = "text/plain"
    headers["Retry-After"] = "60"
    http_error_url = "https://nomads.ncep.noaa.gov/error/f015.idx"
    clock_value = datetime(2026, 11, 22, 7, tzinfo=UTC)
    failed = False

    def opener(request, *, timeout: int) -> FakeResponse:
        nonlocal failed
        if not failed:
            failed = True
            raise urllib.error.HTTPError(
                http_error_url,
                503,
                "Service Unavailable",
                headers,
                io.BytesIO(body),
            )
        return delegate(request, timeout=timeout)

    output_dir = tmp_path / "http-error"
    receipt = capture_script.capture(
        cycle,
        "2026-11-22T12:30:00Z",
        "2026-11-23T03:00:00Z",
        "2026-11-24T03:00:00Z",
        output_dir,
        opener=opener,
        clock=lambda: clock_value,
    )

    attempt = receipt["attempts"][0]
    saved = output_dir / attempt["file"]
    assert attempt["http_status"] == 503
    assert attempt["response_headers"]["Retry-After"] == "60"
    assert attempt["final_url"] == http_error_url
    assert attempt["request_started_utc"] == "2026-11-22T07:00:00.000000Z"
    assert attempt["body_received_utc"] == "2026-11-22T07:00:00.000000Z"
    assert saved.read_bytes() == body
    assert attempt["size_bytes"] == len(body)
    assert attempt["sha256"] == capture_script.sha256(body)
    assert attempt["error"] == "HTTP status 503"
    verified = capture_script.verify_candidate(output_dir)
    assert verified["candidate_status"] == "COMPLETE"
    assert verified["eligible_before_cutoff"] is True


@pytest.mark.parametrize(
    ("field", "expected_error"),
    [
        ("requested_point", "requested point"),
        ("expected_grid_point", "expected grid point"),
        ("bounds", "bounds do not match"),
    ],
)
def test_candidate_verifier_rejects_changes_to_fixed_site_contract(
    tmp_path: Path, field: str, expected_error: str
) -> None:
    cycle = "2026112206"
    intervals = interval_map(21, 45)
    opener, _ = candidate_opener(cycle, intervals)
    output_dir = tmp_path / "candidate"
    capture_script.capture(
        cycle,
        "2026-11-22T12:30:00Z",
        "2026-11-23T03:00:00Z",
        "2026-11-24T03:00:00Z",
        output_dir,
        opener=opener,
        clock=lambda: datetime(2026, 11, 22, 7, tzinfo=UTC),
    )
    receipt_path = output_dir / "candidate_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if field == "requested_point":
        receipt[field]["latitude_degrees_north"] += 0.1
    elif field == "expected_grid_point":
        receipt[field]["longitude_degrees_east"] += 0.1
    else:
        receipt[field]["leftlon"] = "0"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(capture_script.CaptureError, match=expected_error):
        capture_script.verify_candidate(output_dir)


@pytest.mark.parametrize(("eligible", "expected_exit"), [(False, 2), (True, 0)])
def test_cli_capture_reports_and_returns_cutoff_eligibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    eligible: bool,
    expected_exit: int,
) -> None:
    monkeypatch.setattr(
        capture_script,
        "capture",
        lambda *args, **kwargs: {
            "capture_status": "COMPLETE",
            "eligible_before_cutoff": eligible,
        },
    )
    result = capture_script.main(
        [
            "capture",
            "--cycle",
            "2026112212",
            "--cutoff",
            "2026-11-22T12:30:00Z",
            "--target-start",
            "2026-11-23T03:00:00Z",
            "--target-end",
            "2026-11-24T03:00:00Z",
            "--output",
            str(tmp_path / "candidate"),
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert output["capture_status"] == "COMPLETE"
    assert output["eligible_before_cutoff"] is eligible
    assert result == expected_exit


def test_saved_nov21_legacy_capture_remains_verifiable() -> None:
    capture_dir = Path(__file__).parents[1] / "docs/evidence/noaa_gfs_20261121_12z_bengaluru"
    assert capture_script.verify_capture(capture_dir)["verification_status"] == "PASS"
