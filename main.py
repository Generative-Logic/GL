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

import run_modes


def main():

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
    open(".debug/hashburst_trace.txt", "w").close()

    run_modes.full_run()

    end_time = time.time()

    print(f"Overall runtime: {end_time - start_time:.5f} seconds")




if __name__ == "__main__":


    multiprocessing.freeze_support()
    main()

