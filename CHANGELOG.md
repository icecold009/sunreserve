# Changelog

Changes are recorded here before a versioned release. At the 2026-11-19 pre-integration
checkpoint, no release tag had been created.

## Unreleased — candidate at 2026-11-19 checkpoint

- Added a shared JSON prediction export contract for the Streamlit app and CLI, including
  observation/assumption provenance, result freshness, model identity, package-source and
  path-free entrypoint digests.
- Expanded the local research explorer with clear labels for historical examples, validation
  replay, frozen 2025 results, and user-entered observations; frozen outputs are read from
  checksum-validated stored artifacts.
- Hardened observation validation, model inference, reserve arithmetic, and artifact loading;
  finalization uses the saved numeric model bundle rather than fitting a new model.
- Added pinned environment snapshots, bounded CI checks, dependency-audit guidance, and
  weekly dependency/action update proposals.
- Raised the setuptools minimum to 84.1.0 in build and development environments after audit
  findings; pinned that lower bound in the compatibility check. The captured Windows snapshot
  remains unchanged because setuptools was not present in that environment.
- At the 2026-11-19 pre-integration checkpoint, this candidate was in the uncommitted working
  tree and had not received hosted CI or a candidate pull request; no release tag or publication
  was recorded at capture.
