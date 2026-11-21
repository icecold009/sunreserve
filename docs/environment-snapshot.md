# Tested environment snapshot

The application declares broad compatibility ranges in `pyproject.toml` for
CPython 3.11–3.13. The exact Windows reference snapshot is recorded in
[`requirements/snapshots/windows-cpython-3.13.12-amd64.txt`](../requirements/snapshots/windows-cpython-3.13.12-amd64.txt).

## Clean reference environment

On 2026-11-21, a clean Windows AMD64 virtual environment was created with CPython
3.13.12 from current `main` at `51e9f34`, then installed from the current project
and development extras with `python -m pip install -e ".[dev]"`. The resulting
candidate `pip freeze --all` contained 47 third-party package pins. It was captured
after the candidate upgrades and before the separately pinned `pip-audit` scanner
was installed. The editable `sunreserve @ git+...` line is intentionally omitted.

The refreshed candidate set includes pip 26.3, numpy 2.5.5, pytest 9.2.0,
python-multipart 0.0.34, and setuptools 84.2.1. The pinned scanner remains in
`requirements/security-tools.txt` and is not part of the application environment
snapshot.

Validation of this clean environment completed with:

- Full test suite: **306 passed, 4 skipped**.
- `pip check`: no dependency conflicts.
- Ruff lint and formatting checks: passed.
- `pip-audit --no-deps -r requirements/snapshots/windows-cpython-3.13.12-amd64.txt`:
  no known vulnerabilities reported.

Recreate the recorded package environment from PowerShell:

```powershell
py -3.13 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements\snapshots\windows-cpython-3.13.12-amd64.txt
& .\.venv\Scripts\python.exe -m pip install --no-deps -e ".[dev]"
& .\.venv\Scripts\python.exe -m pip check
```

The snapshot records one observed package resolution; it is not a source-code
snapshot, a minimal application lock, or a guarantee that identical wheels exist
on every platform. It does not freeze Python itself or replace the compatibility
ranges. CI resolves compatible current releases on Ubuntu and separately tests
selected declared minimums on Python 3.11; it does not cover a full package-version
cross product.

The selected declared minimums in `requirements/compat-minimums.txt` are installed
together on Python 3.11 in CI. The supported-version CI matrix otherwise resolves
current compatible releases, so the project checks a small lower-bound sample and
a small set of current resolutions.
