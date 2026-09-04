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
/// @brief Direct unit tests for the Windows CUDA build and runtime contract.
///
/// @details
/// The first test validates the exact device properties required by the first
/// Phase 2 port. The second launches native device code and verifies its copied
/// result, proving more than enumeration or successful linking. The third uses
/// the reusable CUDA-event owner around that real device work and requires a
/// positive device interval. Any CUDA failure asserts in production; there is no
/// processor fallback to conceal it.

#include "test_harness.hpp"

#include "../gpu/phase2_cuda.hpp"

#include <cstdint>

TEST(phase2_cuda, device_contract_is_supported) {
    const gl::gpu::CudaDeviceContract contract =
        gl::gpu::queryCudaDeviceContract();
    ASSERT_GE(contract.deviceOrdinal, 0);
    ASSERT_TRUE(contract.computeMajor > 8
        || (contract.computeMajor == 8 && contract.computeMinor >= 9));
    ASSERT_TRUE(contract.totalGlobalBytes > 0);
    ASSERT_TRUE(contract.multiprocessorCount > 0);
    ASSERT_TRUE(contract.maximumThreadsPerBlock >= 256);
}

TEST(phase2_cuda, native_kernel_launches_and_round_trips) {
    constexpr uint32_t input = 4070u;
    constexpr uint32_t expected = input * 1664525u + 1013904223u;
    ASSERT_EQ(gl::gpu::launchCudaContractProbe(input), expected);
}

TEST(phase2_cuda, reusable_device_timer_measures_native_work) {
    gl::gpu::CudaPhase2DeviceTimer timer;
    timer.start();
    constexpr uint32_t input = 89u;
    constexpr uint32_t expected = input * 1664525u + 1013904223u;
    ASSERT_EQ(gl::gpu::launchCudaContractProbe(input), expected);
    ASSERT_TRUE(timer.stopSeconds() > 0.0);
}
