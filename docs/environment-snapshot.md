# Tested environment snapshot

The application declares compatibility separately in `pyproject.toml`: CPython 3.11–3.13, with bounded but non-exact runtime and development dependency ranges. CI exercises those Python versions on Ubuntu and adds one Windows x64 PowerShell smoke on Python 3.13.12.

[`requirements/snapshots/windows-cpython-3.13.12-amd64.txt`](../requirements/snapshots/windows-cpython-3.13.12-amd64.txt) records the exact third-party package versions resolved in the local reference environment on 2026-11-19:

- CPython 3.13.12, 64-bit AMD64
- Windows 11, build 10.0.26300
- 46 exact package versions, including development tools, transitive dependencies, and packages already present in the reference environment
- The editable `sunreserve` checkout itself is omitted so the snapshot does not depend on a local or branch-specific source path

The snapshot was refreshed after dependency-audit findings to use pip 26.2.0 and pytest 9.0.3. The other package versions are preserved, and the separately installed `pip-audit` scanner remains outside the snapshot.

At the 2026-11-19 reference capture, the checkout was based on `3c0b56533a582962a135e4f274148cdaac52eda1` with uncommitted P0/P1 work in progress. This file captures package versions only; it is not a source-code snapshot.

Recreate that reference environment from PowerShell:

```powershell
py -3.13 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements\snapshots\windows-cpython-3.13.12-amd64.txt
& .\.venv\Scripts\python.exe -m pip install --no-deps -e ".[dev]"
& .\.venv\Scripts\python.exe -m pip check
```

The snapshot records one observed environment, including any ambient packages visible to `pip freeze`; it is not a minimal application lock. It does not replace the compatibility ranges, claim that the same wheels exist on every platform, or freeze Python itself. CI updates are proposals; review and refresh the snapshot deliberately when accepting dependency changes.

The selected declared minimums in `requirements/compat-minimums.txt` are installed together on Python 3.11 in CI. The supported-version CI matrix otherwise resolves the current compatible releases, so the project checks a small lower-bound sample and a small set of current resolutions without a full package-version cross product.
