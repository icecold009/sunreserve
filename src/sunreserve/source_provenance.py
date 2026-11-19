"""Git provenance for the source checkout that produced SunReserve artifacts."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

SOURCE_CHECKOUT_ROOT = Path(__file__).resolve().parents[2]
UNAVAILABLE_REVISION = "unavailable"


def source_provenance() -> dict[str, str | bool | None]:
    """Return source-checkout HEAD and dirty state without retaining status filenames."""
    command = ["git", "-C", str(SOURCE_CHECKOUT_ROOT)]
    try:
        head = subprocess.run(
            [*command, "rev-parse", "--verify", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError, UnicodeError):
        return {"code_revision": UNAVAILABLE_REVISION, "working_tree_dirty": None}
    if re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", head) is None:
        return {"code_revision": UNAVAILABLE_REVISION, "working_tree_dirty": None}

    try:
        status = subprocess.run(
            [*command, "status", "--porcelain=v1", "--untracked-files=normal"],
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        ).stdout
        dirty: bool | None = bool(status.strip())
    except (OSError, subprocess.SubprocessError, UnicodeError):
        dirty = None
    return {"code_revision": head.lower(), "working_tree_dirty": dirty}
