# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt)
# Dual-licensed under the GNU Affero General Public License v3 or later
# and a commercial license — see https://generative-logic.com/license.
# Contributions require CLA — see CONTRIBUTING.md.

"""Direct orchestration regression tests for the accelerated frame."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run_modes


def run_tests() -> None:
    calls: list[str] = []
    original_reader = run_modes.configuration_reader
    original_find = run_modes._find_gl_quick_exe
    original_subprocess_run = run_modes.subprocess.run
    original_seed = run_modes._seed_per_batch_binary
    original_native = run_modes.run_gl_quick
    original_merge = run_modes._merge_into_shared
    original_sleep = run_modes.time.sleep
    try:
        run_modes.configuration_reader = lambda _path: SimpleNamespace(
            parameters=SimpleNamespace(simple_facts_parameters=[])
        )
        run_modes._find_gl_quick_exe = lambda: run_modes.PROJECT_ROOT / "fake_gl_quick"
        run_modes.subprocess.run = lambda *_args, **_kwargs: calls.append("conjecturer")
        run_modes._seed_per_batch_binary = lambda _tag: calls.append("seed")
        run_modes.run_gl_quick = lambda _tag: calls.append("native")
        run_modes._merge_into_shared = lambda _tag: calls.append("merge")

        def reject_sleep(_seconds: float) -> None:
            raise AssertionError("_run_batch must not insert a fixed filesystem wait")

        run_modes.time.sleep = reject_sleep
        run_modes._run_batch("Peano", [], add_cross_anchor=False)
        assert calls == ["conjecturer", "seed", "native", "merge"]
    finally:
        run_modes.configuration_reader = original_reader
        run_modes._find_gl_quick_exe = original_find
        run_modes.subprocess.run = original_subprocess_run
        run_modes._seed_per_batch_binary = original_seed
        run_modes.run_gl_quick = original_native
        run_modes._merge_into_shared = original_merge
        run_modes.time.sleep = original_sleep


if __name__ == "__main__":
    run_tests()
    print("run_modes frame tests passed")
