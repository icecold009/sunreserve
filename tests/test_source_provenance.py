from __future__ import annotations

import subprocess

from sunreserve import source_provenance as provenance


def _completed(command: list[str], stdout: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")


def test_source_provenance_reports_clean_checkout_without_filenames(monkeypatch) -> None:
    revision = "a1" * 20
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        return _completed(command, f"{revision}\n" if "rev-parse" in command else "")

    monkeypatch.setattr(provenance.subprocess, "run", fake_run)

    result = provenance.source_provenance()

    assert result == {"code_revision": revision, "working_tree_dirty": False}
    assert calls[0][0:3] == ["git", "-C", str(provenance.SOURCE_CHECKOUT_ROOT)]
    assert calls[1][-3:] == ["status", "--porcelain=v1", "--untracked-files=normal"]


def test_source_provenance_reports_dirty_without_retaining_status_paths(monkeypatch) -> None:
    revision = "b2" * 20

    def fake_run(command, **_kwargs):
        output = f"{revision}\n" if "rev-parse" in command else " M private/path.txt\n"
        return _completed(command, output)

    monkeypatch.setattr(provenance.subprocess, "run", fake_run)

    result = provenance.source_provenance()

    assert result == {"code_revision": revision, "working_tree_dirty": True}
    assert "private/path.txt" not in str(result)


def test_source_provenance_marks_unavailable_git_metadata(monkeypatch) -> None:
    def unavailable(*_args, **_kwargs):
        raise FileNotFoundError("git executable unavailable")

    monkeypatch.setattr(provenance.subprocess, "run", unavailable)

    assert provenance.source_provenance() == {
        "code_revision": "unavailable",
        "working_tree_dirty": None,
    }


def test_source_provenance_preserves_revision_when_status_is_unavailable(monkeypatch) -> None:
    revision = "c3" * 20

    def fake_run(command, **_kwargs):
        if "rev-parse" in command:
            return _completed(command, f"{revision}\n")
        raise subprocess.CalledProcessError(128, command)

    monkeypatch.setattr(provenance.subprocess, "run", fake_run)

    assert provenance.source_provenance() == {
        "code_revision": revision,
        "working_tree_dirty": None,
    }
