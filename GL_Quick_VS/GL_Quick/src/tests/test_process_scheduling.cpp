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

/// @file
/// @brief Unit test for the process-wide scheduling settings.
///
/// @details
/// Covers `gl::configureProcessScheduling`: on Windows, after the call the
/// process runs in the above-normal priority class with execution-speed
/// power throttling explicitly off, read back through the same Windows
/// interfaces. The harness process runs `main`, which applies the settings
/// before any test; applying them again must leave the same state. On other
/// systems the call does nothing and the test checks nothing.

#include "test_harness.hpp"

#include "../process_scheduling.hpp"

TEST(process_scheduling, above_normal_priority_without_power_throttling) {
    gl::configureProcessScheduling();
#ifdef _WIN32
    const HANDLE process = GetCurrentProcess();
    ASSERT_EQ(GetPriorityClass(process),
              static_cast<DWORD>(ABOVE_NORMAL_PRIORITY_CLASS));
    PROCESS_POWER_THROTTLING_STATE throttling{};
    throttling.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
    ASSERT_TRUE(GetProcessInformation(process, ProcessPowerThrottling,
                                      &throttling, sizeof(throttling)) != 0);
    ASSERT_EQ(throttling.ControlMask & PROCESS_POWER_THROTTLING_EXECUTION_SPEED,
              static_cast<ULONG>(PROCESS_POWER_THROTTLING_EXECUTION_SPEED));
    ASSERT_EQ(throttling.StateMask & PROCESS_POWER_THROTTLING_EXECUTION_SPEED,
              static_cast<ULONG>(0));
#endif
}
