# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt)
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""The Lean toolchain for tests that kernel-check generated Lean files."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from proof_export.live import find_lake  # noqa: E402

# The tracked Lake project: its `lean-toolchain` pins the toolchain that the
# elan launcher selects when `lean` runs inside it.
LEAN_PROJECT = ROOT / "lean_export"


def lean_executable() -> str:
    """The `lean` beside the Lake executable the Lean export would use.

    Discovery is the export's own (`proof_export.live.find_lake`: PATH first,
    then `ELAN_HOME/bin`, default `~/.elan/bin`, with the platform's file
    names). Without a toolchain the calling test is skipped, since the Lean
    export is optional; a toolchain without `lean` beside its Lake fails.
    """
    lake = find_lake()
    if lake is None:
        raise unittest.SkipTest("Lean toolchain not found: no lake on PATH or under ELAN_HOME/bin")
    lean = lake.with_name("lean" + lake.suffix)
    assert lean.is_file(), f"Lean toolchain without lean beside {lake}"
    return str(lean)


def lean_environment() -> dict[str, str]:
    """The kernel check's child environment: a configured ELAN_HOME is kept, else `~/.elan`."""
    environment = dict(os.environ)
    environment.setdefault("ELAN_HOME", str(Path.home() / ".elan"))
    return environment
