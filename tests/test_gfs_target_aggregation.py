"""Regression tests for the saved NOAA GFS point-sample evidence check."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "docs/evidence/noaa_gfs_grid4_20200412_0600_target_evidence.json"
CHECKER = REPO_ROOT / "scripts/check_gfs_target_aggregation.py"


def copied_evidence(tmp_path: Path) -> Path:
    source_manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    for record in (
        source_manifest["records"] + source_manifest["supplemental_out_of_target_records"]
    ):
        shutil.copy2(
            MANIFEST.parent / record["csv_response_file"],
            evidence_dir / record["csv_response_file"],
        )
    manifest_path = evidence_dir / "manifest.json"
    manifest_path.write_text(json.dumps(source_manifest), encoding="utf-8")
    return manifest_path


def run_checker(manifest_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECKER), "--manifest", str(manifest_path)],
        check=False,
        capture_output=True,
        text=True,
    )


def test_saved_sample_reconstructs_target_and_excludes_supplemental(tmp_path: Path) -> None:
    result = run_checker(copied_evidence(tmp_path))
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["aggregation_status"] == "PASS"
    assert report["daily_integrated_downward_shortwave_kwh_m2"] == 8.154
    assert report["target_interval_utc"] == ["2020-04-13T03:00:00Z", "2020-04-14T03:00:00Z"]
    assert report["historical_issue_time_availability"] == "UNVERIFIED"
    assert report["operational_evaluation"] == "NOT PERMITTED on this evidence"
    assert report["supplemental_out_of_target_record"]["id"] == "f063"
    assert report["supplemental_out_of_target_record"]["excluded_from_target_integration"] is True


def test_supplemental_csv_tampering_fails_integrity_check(tmp_path: Path) -> None:
    manifest_path = copied_evidence(tmp_path)
    record = json.loads(manifest_path.read_text(encoding="utf-8"))[
        "supplemental_out_of_target_records"
    ][0]
    csv_path = manifest_path.parent / record["csv_response_file"]
    csv_path.write_bytes(csv_path.read_bytes() + b"\n")
    result = run_checker(manifest_path)
    assert result.returncode == 1
    assert "SHA-256 mismatch" in result.stderr


def test_tampered_manifest_hash_fails_closed(tmp_path: Path) -> None:
    manifest_path = copied_evidence(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["records"][0]["csv_response_sha256"] = hashlib.sha256(b"wrong").hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = run_checker(manifest_path)
    assert result.returncode == 1
    assert "SHA-256 mismatch" in result.stderr


def test_malformed_manifest_reports_cli_error(tmp_path: Path) -> None:
    manifest_path = tmp_path / "bad.json"
    manifest_path.write_text("[]", encoding="utf-8")
    result = run_checker(manifest_path)
    assert result.returncode == 1
    assert "AGGREGATION FAIL: manifest must be a JSON object" in result.stderr
