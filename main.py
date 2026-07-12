# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# ------------------------------------------------------------------------------
#
# This software is also available under a commercial license. For details,
# see: https://generative-logic.com/license
#
# Contributions to this project must be made under the terms of the
# Contributor License Agreement (CLA). See the project's CONTRIBUTING.md file.




import sys
import time

import multiprocessing

# Force UTF-8 for stdout/stderr so non-ASCII glyphs (em-dash, ASCII-art tail
# characters) survive shell redirection on Windows. Without this, Python falls
# back to the console's locale (often cp1252) and writes 0x97 instead of the
# UTF-8 em-dash, which then renders as a black-block in editors that read the
# log as UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import os
import subprocess

import run_modes
from frame_timing import (
    StageTimer,
    configure_frame_timing,
    record_frame_timing,
)


# Path to the native gl_quick binary. Resolved relative to this file so
# the gate works identically when main.py is invoked from a worktree
# (`.worktree/<name>/`) or from the main repo. Windows-only suffix on
# Windows; POSIX paths elsewhere.
_GL_QUICK_EXE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "GL_Quick_VS", "GL_Quick",
    "gl_quick.exe" if os.name == "nt" else "gl_quick",
)


def _run_unit_test_gate():
    """Run the in-tree C++ unit-test harness; abort on any failure.

    Invokes ``gl_quick --unit-tests``. The harness is fast (sub-second
    today, target <30 s for the full suite) and runs entirely on
    synthetic in-memory inputs — no config-file reads, no real proof
    runs. First failure aborts main.py before any pipeline work, so
    regressions are caught before the 10–60-minute prover invocation
    starts.

    Skipped silently when the binary does not exist yet (typical fresh
    clone before first build) so the bootstrap-cycle isn't blocked.
    """
    if not os.path.exists(_GL_QUICK_EXE):
        print(
            f"[main.py] {_GL_QUICK_EXE} not found; skipping unit-test gate.",
            file=sys.stderr,
        )
        return
    proc = subprocess.run([_GL_QUICK_EXE, "--unit-tests"], check=False)
    if proc.returncode != 0:
        print(
            f"[main.py] Unit tests failed (exit {proc.returncode}). Aborting.",
            file=sys.stderr,
        )
        sys.exit(proc.returncode)


# Path to the Python verifier unit-test harness. Resolved relative to this
# file so the gate works identically from a worktree (`.worktree/<name>/`)
# and the main repo.
_VERIFIER_TEST_HARNESS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "tests", "test_harness.py",
)


def _run_verifier_unit_test_gate():
    """Run the Python verifier-side unit-test harness; abort on any failure.

    Invokes ``python tests/test_harness.py`` which imports every sibling
    ``test_verifier_<group>.py`` module and runs the full suite against the
    live ``verifier.py`` checkers: mostly subtle-error tests, plus positive
    sanity tests that prove the fixture rig is intact. Each failure test
    feeds a malformed ``ProofLine`` into the relevant ``TAG_CHECKERS`` entry
    (or ``verify_chapter`` for chapter-level meta-checks) and asserts the
    verifier reports a failure. Runtime is <5 s — comparable to the C++
    harness — so a regression in any checker surfaces before any pipeline
    work rather than after a 10–60-minute prover run.

    Skipped silently when ``tests/test_harness.py`` is absent (matches the
    fresh-clone fall-through used by ``_run_unit_test_gate``). Black-box:
    the harness imports ``verifier.py`` symbols and calls them as-is — zero
    modifications to verifier logic, in keeping with invariant I-16.
    """
    if not os.path.exists(_VERIFIER_TEST_HARNESS):
        print(
            f"[main.py] {_VERIFIER_TEST_HARNESS} not found; "
            "skipping verifier unit-test gate.",
            file=sys.stderr,
        )
        return
    # Flush the parent's buffered stdout/stderr before launching the harness
    # subprocess so its output does not interleave with anything still sitting
    # unflushed in the parent buffer (stdout is block-buffered when redirected
    # to a file).
    sys.stdout.flush()
    sys.stderr.flush()
    proc = subprocess.run(
        [sys.executable, _VERIFIER_TEST_HARNESS], check=False)
    if proc.returncode != 0:
        print(
            f"[main.py] Verifier unit tests failed "
            f"(exit {proc.returncode}). Aborting.",
            file=sys.stderr,
        )
        sys.exit(proc.returncode)


_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def _read_version() -> str:
    """Return the release-version string from the `VERSION` file at the repo root.

    Falls back to ``unknown`` when the file is absent (out-of-tree invocation
    or unstamped working copy).
    """
    path = os.path.join(_REPO_ROOT, "VERSION")
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            return text if text else "unknown"
    except OSError:
        return "unknown"


def main():

    run_started_ns = time.perf_counter_ns()

    if "--version" in sys.argv[1:]:
        print(f"Generative Logic {_read_version()}")
        return

    descriptor_positions = [
        index for index, argument in enumerate(sys.argv)
        if argument == "--run-descriptor"
    ]
    assert len(descriptor_positions) <= 1, \
        "--run-descriptor must be supplied at most once"
    if descriptor_positions:
        descriptor_position = descriptor_positions[0]
        assert descriptor_position + 1 < len(sys.argv), \
            "--run-descriptor requires a value"
        run_descriptor = sys.argv[descriptor_position + 1]
    else:
        run_descriptor = time.strftime("%Y%m%d_%H%M%S")
    assert run_descriptor and all(
        character.isalnum() or character in "-_"
        for character in run_descriptor
    ), "run descriptor must contain only letters, digits, hyphens, or underscores"

    start_time = time.time()

    # Wipe .debug/hashburst_trace.txt once per main.py session so each
    # full run starts with a clean trace. The C++ side (prover.cpp:2384)
    # only truncates conditionally on the first burst of a matching LB,
    # which would let stale traces from prior sessions survive when the
    # current session's target batch (e.g. IncubatorGauss1) is not run.
    # Truncating here gives "clean per main.py" semantics while still
    # preserving the trace across sibling gl_quick.exe invocations
    # within the same session (the previous truncate-at-process-start
    # in main.cpp wiped IncubatorGauss1's trace as soon as the
    # following Gauss-main batch started).
    os.makedirs(".debug", exist_ok=True)
    frame_timing_path = os.path.abspath(
        os.path.join(".debug", f"frame_timing_{run_descriptor}.jsonl")
    )
    configure_frame_timing(frame_timing_path)
    print(f"[frame-timing] {frame_timing_path}")

    startup_started_ns = time.perf_counter_ns()
    memory_log_path = os.path.abspath(
        os.path.join(".debug", f"memory_{run_descriptor}.log")
    )
    os.environ["GL_MEMORY_LOG_PATH"] = memory_log_path
    with open(memory_log_path, "w", encoding="utf-8") as memory_log:
        memory_log.write("# GL static-pool memory report\n\n")
        memory_log.write(f"Run descriptor: `{run_descriptor}`\n\n")
        memory_log.write(
            "Each peak is the independent high-water mark of one physical "
            "pool during one batch. Pool peaks are not additive.\n"
        )
    print(f"[memory-log] {memory_log_path}")
    open(".debug/hashburst_trace.txt", "w").close()
    open(".debug/trap.log", "w").close()

    # Wipe `.rt/*.log` once per main.py session so each full run starts
    # with a clean view of the per-LB runtime-measurement table. A
    # non-empty `.rt/` after a run is then unambiguous: every file
    # represents a pathological LB from THIS run. The C++ side never
    # wipes the folder; it only ever rewrites the per-LB file in place.
    # When RT_MEASUREMENT == 0 (the default) no files are ever written
    # and this wipe is a no-op. See `docs/_meta/rt_measurement.md`.
    os.makedirs(".rt", exist_ok=True)
    for _stale in os.listdir(".rt"):
        _stale_path = os.path.join(".rt", _stale)
        if os.path.isfile(_stale_path):
            os.remove(_stale_path)
    record_frame_timing(
        "python.startup",
        (time.perf_counter_ns() - startup_started_ns) / 1_000_000_000.0,
        parent="run.total",
    )

    # C++ unit-test gate. Synthetic in-memory inputs, no real proof runs;
    # safe to run pre-pipeline. First failure aborts main.py before any
    # 10–60-minute proof work. See _run_unit_test_gate's docstring for budget.
    with StageTimer("python.unit_tests.cpp", parent="run.total"):
        _run_unit_test_gate()

    # Python verifier-side unit-test gate. Runs with the C++ gate, before the
    # pipeline: its fixtures are the frozen, test-owned tests/fixtures/ copies
    # (independent of the live files/GL_binaries/ the pipeline regenerates), so
    # the gate no longer waits on a populated files/GL_binaries/. First failure
    # aborts main.py before any 10-60-minute proof work.
    with StageTimer("python.unit_tests.verifier", parent="run.total"):
        _run_verifier_unit_test_gate()

    with StageTimer("python.pipeline", parent="run.total"):
        run_modes.full_run()

    end_time = time.time()
    record_frame_timing(
        "run.total",
        (time.perf_counter_ns() - run_started_ns) / 1_000_000_000.0,
        parent="root",
    )

    print(f"Overall runtime: {end_time - start_time:.5f} seconds")




if __name__ == "__main__":


    multiprocessing.freeze_support()
    main()

