/*
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU Affero General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option) any
later version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more
details.

You should have received a copy of the GNU Affero General Public License along
with this program. If not, see <https://www.gnu.org/licenses/>.

Commercial licensing without the AGPLv3 obligations is available:
https://generative-logic.com/license
*/

#pragma once

#include <cassert>
#include <filesystem>
#include <fstream>
#include <ostream>

namespace gl {

    /// @brief The per-run diagnostics stream for observation-only telemetry
    ///        lines that would otherwise flood the main run log.
    ///
    /// @details
    /// Returns one process-wide append-mode stream backed by
    /// `.debug/run_diagnostics.log`, opened lazily on first use with the
    /// `.debug` directory created if absent. The Phase 2 backend telemetry
    /// (`[GPU-PACKING]`, `[GPU-PHASE2]`, `[GPU-PHASE2-MEMORY]`), the phase
    /// wall-time lines (`[PHASE2-TIMING]`, `[PHASE13-TIMING]`,
    /// `[PHASE13-DETAIL]`), and the deload telemetry (`[DELOAD]`) write here
    /// so the main run log keeps only the proof-relevant narrative. The
    /// pipeline driver truncates the file at run start, so one file holds one
    /// pipeline run's telemetry across all of its prover processes.
    ///
    /// Every emission site is a single-threaded seam (post-join barrier code
    /// and phase-window closes), so the stream needs no lock. The lines are
    /// observation only — no prover decision reads them (I-44 discipline).
    ///
    /// @return The open append-mode diagnostics stream.
    /// @invariant Opening the stream never fails silently: an unopenable file
    ///            asserts (Rule 19), never degrades to a discarded stream.
    inline std::ostream& diagnosticsLog() {
        static std::ofstream stream = [] {
            std::filesystem::create_directories(".debug");
            std::ofstream s(".debug/run_diagnostics.log",
                            std::ios::out | std::ios::app);
            assert(s.is_open()
                && "diagnostics log .debug/run_diagnostics.log must open");
            return s;
        }();
        return stream;
    }

}  // namespace gl
