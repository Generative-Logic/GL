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
#include "test_harness.hpp"

#include <cstddef>
#include <vector>

TEST(unit_harness, cuda_route_gate_reserves_worker_zero_for_cuda_tests) {
    // Every CUDA-route test runs on worker 0; the others go round-robin to
    // workers 1..K-1 in registration order.
    for (std::size_t ordinal = 0; ordinal < 5; ++ordinal)
        ASSERT_EQ(gl::tests::assignedWorker(true, ordinal, 4, true), 0u);
    const std::vector<unsigned> expected = { 1, 2, 3, 1, 2, 3, 1 };
    for (std::size_t ordinal = 0; ordinal < expected.size(); ++ordinal)
        ASSERT_EQ(gl::tests::assignedWorker(false, ordinal, 4, true), expected[ordinal]);
    // Two workers: one holds the device, the other runs everything else.
    ASSERT_EQ(gl::tests::assignedWorker(false, 7, 2, true), 1u);
    ASSERT_EQ(gl::tests::assignedWorker(true, 0, 2, true), 0u);
}

TEST(unit_harness, processor_route_gate_deals_every_worker) {
    const std::vector<unsigned> expected = { 0, 1, 2, 0, 1, 2 };
    for (std::size_t ordinal = 0; ordinal < expected.size(); ++ordinal)
        ASSERT_EQ(gl::tests::assignedWorker(false, ordinal, 3, false), expected[ordinal]);
}

TEST(unit_harness, a_single_worker_runs_every_test) {
    ASSERT_EQ(gl::tests::assignedWorker(true, 0, 1, true), 0u);
    ASSERT_EQ(gl::tests::assignedWorker(false, 41, 1, true), 0u);
    ASSERT_EQ(gl::tests::assignedWorker(false, 41, 1, false), 0u);
}

TEST(unit_harness, default_worker_count_is_half_the_threads_within_one_to_twelve) {
    ASSERT_EQ(gl::tests::defaultWorkerCount(0), 1u);
    ASSERT_EQ(gl::tests::defaultWorkerCount(1), 1u);
    ASSERT_EQ(gl::tests::defaultWorkerCount(2), 1u);
    ASSERT_EQ(gl::tests::defaultWorkerCount(8), 4u);
    ASSERT_EQ(gl::tests::defaultWorkerCount(24), 12u);
    ASSERT_EQ(gl::tests::defaultWorkerCount(32), 12u);
    ASSERT_EQ(gl::tests::defaultWorkerCount(128), 12u);
}
