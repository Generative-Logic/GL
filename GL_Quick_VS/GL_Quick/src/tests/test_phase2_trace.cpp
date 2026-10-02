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
#include "../infra/phase2_trace.hpp"
#include <thread>

TEST(phase2_trace, enabled_and_disabled_ranges_preserve_nesting) {
    ASSERT_EQ(gl::phase2TraceDepth(), uint32_t(0));
    gl::Phase2TraceScope outer("test.outer", 123, true);
    ASSERT_EQ(gl::phase2TraceDepth(), uint32_t(1));
    {
        gl::Phase2TraceScope disabled("test.disabled", 456, false);
        ASSERT_EQ(gl::phase2TraceDepth(), uint32_t(1));
        gl::Phase2TraceScope inner("test.inner", 789, true);
        ASSERT_EQ(gl::phase2TraceDepth(), uint32_t(2));
    }
    ASSERT_EQ(gl::phase2TraceDepth(), uint32_t(1));
    outer.finish();
    ASSERT_EQ(gl::phase2TraceDepth(), uint32_t(0));
    outer.finish();
    ASSERT_EQ(gl::phase2TraceDepth(), uint32_t(0));
    ASSERT_EQ(gl::phase2TraceEnabled(), gl::phase2TraceEnabled());
}

TEST(phase2_trace, independent_thread_nesting) {
    uint32_t observed = 0;
    gl::Phase2TraceScope parent("test.parent", 0, true);
    std::thread worker([&] {
        gl::Phase2TraceScope first("test.worker", 0, true);
        observed = gl::phase2TraceDepth();
    });
    worker.join();
    ASSERT_EQ(observed, uint32_t(1));
    ASSERT_EQ(gl::phase2TraceDepth(), uint32_t(1));
    parent.finish();
    ASSERT_EQ(gl::phase2TraceDepth(), uint32_t(0));
}


TEST(phase2_trace, current_thread_cpu_counters_are_monotonic) {
    const auto before = gl::readPhase2ThreadCpu();
    volatile uint64_t work = 1;
    {
        gl::Phase2TraceScope measured("test.cpu", 0, true, true);
        for (uint64_t i = 0; i < 100000; ++i) work = work * 3 + i;
    }
    const auto after = gl::readPhase2ThreadCpu();
    ASSERT_TRUE(after.userNanoseconds >= before.userNanoseconds);
    ASSERT_TRUE(after.systemNanoseconds >= before.systemNanoseconds);
    ASSERT_EQ(after.cyclesAvailable, before.cyclesAvailable);
    ASSERT_TRUE(after.cycles >= before.cycles);
    if (after.cyclesAvailable) ASSERT_TRUE(after.cycles > before.cycles);
    ASSERT_EQ(gl::phase2TraceDepth(), uint32_t(0));
    gl::Phase2TraceScope disabled("test.disabled.cpu", 0, false, true);
    ASSERT_EQ(gl::phase2TraceDepth(), uint32_t(0));
}
