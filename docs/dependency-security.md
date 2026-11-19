# Dependency and workflow security

CI installs the pinned `pip-audit` scanner from `requirements/security-tools.txt` and audits supported-version environments with `pip-audit --local` against the current Python Packaging Advisory Database. The Windows snapshot job audits the exact locked package list with `pip-audit --no-deps -r requirements/snapshots/windows-cpython-3.13.12-amd64.txt`. The scan checks known package vulnerabilities; `pip check` separately checks installed dependency compatibility and does not detect vulnerabilities. The scanner needs its advisory service, so this check runs online in CI.

Dependabot proposes weekly updates for Python packages and GitHub Actions. Workflow actions are pinned to full commit SHAs with release comments. Review an action update by checking the upstream release/tag and SHA together, then run the workflow checks before accepting it.

## Handling an actionable finding

1. Confirm the affected package, advisory identifier, affected version range, and whether the installed dependency is reachable in this project. Check the advisory's fixed version and upstream release notes.
2. Prefer upgrading to a supported fixed release within the declared `pyproject.toml` range. If the range prevents a safe fix, change the range and tests in the same reviewed update.
3. Run the full tests, `pip check`, and `pip-audit`; regenerate the tested environment snapshot when the accepted resolution changes.
4. If no safe fix is available, document the advisory identifier, affected path, exposure assessment, compensating action, owner, and a review date in the change proposal. Do not silently suppress the scanner or treat an unpatched finding as resolved.
5. Address confirmed critical or high-severity reachable findings before routine feature work. For suspected vulnerabilities in SunReserve itself, check the repository's configured security policy and reporting route. If none is configured, request a private contact from a maintainer; never put exploit details in a public issue.

The scanner reports known package advisories. It does not establish that a package is benign, catch every supply-chain compromise, inspect native system libraries, or replace code review.
