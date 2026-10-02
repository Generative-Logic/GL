/* Generative Logic : A deterministic reasoning and knowledge generation engine.
 Copyright(C) 2025-2026 Generative Logic UG(haftungsbeschränkt)

 This program is free software : you can redistribute it and /or modify
 it under the terms of the GNU Affero General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the
 GNU Affero General Public License for more details.

 You should have received a copy of the GNU Affero General Public License
 along with this program.If not, see < https://www.gnu.org/licenses/>.

 ------------------------------------------------------------------------------

 This software is also available under a commercial license.For details,
 see: https://generative-logic.com/license

 Contributions to this project must be made under the terms of the
 Contributor License Agreement(CLA).See the project's CONTRIBUTING.md file.*/

#pragma once
#include <cstdint>

namespace gl {

    struct Phase2ThreadCpu {
        uint64_t userNanoseconds{0};
        uint64_t systemNanoseconds{0};
        uint64_t cycles{0};
        bool cyclesAvailable{false};
    };

    /// @brief Read operating-system CPU accounting for the current thread.
    /// @details User and system times exclude periods when this thread is not
    /// executing (Windows `GetThreadTimes`, macOS Mach `thread_info`, Linux
    /// `getrusage(RUSAGE_THREAD)`). Windows additionally supplies scheduled CPU cycles, which
    /// must not be converted to elapsed seconds. Accounting resolution belongs
    /// to the operating system; nanosecond units do not imply that precision.
    /// @return Cumulative thread CPU counters; failed system calls assert.
    Phase2ThreadCpu readPhase2ThreadCpu();

    /// @brief Read the process-wide opt-in for Phase 2 profiling ranges.
    /// @details GL_PHASE2_RT accepts absent, 0 or 1. The setting is frozen on
    /// first use; it controls observations only and never proof execution.
    /// @return True exactly when GL_PHASE2_RT is 1.
    bool phase2TraceEnabled();

    /// @brief Inspect the current thread's balanced observation nesting.
    /// @details Each enabled range contributes one level until it closes.
    /// Disabled ranges do not change nesting or call the profiling interface.
    /// @return The number of open enabled ranges on the current thread.
    uint32_t phase2TraceDepth();

    /// @brief Scope a host operation and its asynchronous CUDA launches.
    /// @details NVTX records host range boundaries and correlates enclosed
    /// launches with the GPU timeline. The scope neither synchronizes CUDA nor
    /// treats host range duration as kernel time. Labels are consumed at entry.
    class Phase2TraceScope {
    public:
        /// @brief Open an optional observation range with a numeric identity.
        /// @details Stack-owned state preserves strict thread-local nesting.
        /// No prover state, interner, storage capacity or execution policy changes.
        /// @param label Nonempty operation label, consumed synchronously.
        /// @param payload Numeric iteration, task or chunk identifier.
        /// @param enabled Whether this range emits observations.
        /// @param measureCpu Whether to collect thread CPU counters for this scope.
        /// @return A scope closed by finish or destruction.
        Phase2TraceScope(const char* label, uint64_t payload = 0,
            bool enabled = phase2TraceEnabled(), bool measureCpu = false);
        /// @brief Close a range still owned by this object.
        /// @details Explicitly finished ranges require no second close.
        /// @return No value.
        ~Phase2TraceScope();
        Phase2TraceScope(const Phase2TraceScope&) = delete;
        Phase2TraceScope& operator=(const Phase2TraceScope&) = delete;
        /// @brief End this observation at an existing algorithm boundary.
        /// @details The active range must be the current thread's innermost
        /// range. Disabled scopes and already-finished scopes remain closed.
        /// @return No value.
        void finish();
    private:
        uint32_t depth_{0};
        bool measureCpu_{false};
        Phase2ThreadCpu cpuStart_{};
    };
}
