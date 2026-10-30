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

#include <cassert>

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX  // keep windows.h from clobbering std::min / std::max
#  endif
#  include <windows.h>
#endif

// Process-wide scheduling settings of gl_quick (Windows only).

namespace gl {

    /// @brief Ask Windows to run this process ahead of normal-priority
    ///        foreign load and never to slow it down as background work.
    ///
    /// @details
    /// Phases 1 and 3 of every iteration wait for the slowest logic block,
    /// and the tail of a long batch runs on a single thread, so whatever
    /// slows that thread slows the whole run. The call makes two
    /// process-wide settings that every thread of the process inherits:
    ///
    /// - priority class `ABOVE_NORMAL_PRIORITY_CLASS`: when the processors
    ///   are busy, the prover's threads run before the threads of
    ///   normal-priority processes;
    /// - execution-speed power throttling switched off
    ///   (`ProcessPowerThrottling` with the execution-speed bit set in the
    ///   control mask and cleared in the state mask): a run started without
    ///   a visible window is otherwise eligible for the efficiency mode that
    ///   Windows applies to background work, with lower clock speeds and,
    ///   on hybrid processors, the efficiency cores.
    ///
    /// Call once at process start. On every other system the call does
    /// nothing: raising the priority needs administrator rights on Linux
    /// and macOS, Linux has no per-process power throttling, and macOS
    /// chooses cores by per-thread quality-of-service classes.
    ///
    /// @return Nothing.
    /// @invariant On Windows, after the call the process priority class is
    ///            `ABOVE_NORMAL_PRIORITY_CLASS` and execution-speed power
    ///            throttling is explicitly off.
    inline void configureProcessScheduling() {
#ifdef _WIN32
        const HANDLE process = GetCurrentProcess();
        const BOOL priorityRaised =
            SetPriorityClass(process, ABOVE_NORMAL_PRIORITY_CLASS);
        assert(priorityRaised != 0
               && "configureProcessScheduling: SetPriorityClass failed");
        PROCESS_POWER_THROTTLING_STATE throttling{};
        throttling.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
        throttling.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
        throttling.StateMask = 0;
        const BOOL throttlingOff = SetProcessInformation(
            process, ProcessPowerThrottling, &throttling, sizeof(throttling));
        assert(throttlingOff != 0
               && "configureProcessScheduling: SetProcessInformation failed");
#endif
    }

} // namespace gl
