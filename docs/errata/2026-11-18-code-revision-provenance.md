# Erratum: frozen run revision provenance

**Date:** 2026-11-18

The `code_revision: "uncommitted"` fields in the frozen
`artifacts/runs/frozen-test-2023-24/run_manifest.json` and
`artifacts/runs/frozen-test-2025/run_manifest.json` are historical placeholders. They do
not identify the source commit that produced either run, or whether that checkout had
uncommitted changes. The original commit and dirty state cannot be inferred from those
fields. Both manifests and all associated frozen artifacts and hashes remain unchanged.

Newly generated validation, experiment, finalized-run, and 2025 holdout manifests record
the full source-checkout Git HEAD when available and a working-tree dirty state. If HEAD
cannot be read or is not a full SHA, `code_revision` is `unavailable` and
`working_tree_dirty` is `null`. If HEAD is readable but `git status` is unavailable,
`code_revision` retains that full SHA and only `working_tree_dirty` is `null`. These
fields do not change configuration, input-data, or model identity hashes.
