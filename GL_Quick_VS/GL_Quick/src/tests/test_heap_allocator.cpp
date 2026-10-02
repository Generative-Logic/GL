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
/// @brief Unit tests for the process-wide heap allocator configuration.
///
/// @details
/// Covers `gl::configureHeapAllocator` (D-427):
/// the arena size mimalloc reports after the call, and the behaviour the
/// configuration exists for — blocks above 64 MiB, sized like the Phase 2 host
/// projection's large columns, are found again once freed and collected
/// instead of leaking (G-78). Whether the allocator returns them to the
/// operating system at the collect or keeps them for reuse depends on its
/// version: mimalloc 3.5.1 on macOS keeps the blocks of up to 112 MiB. The
/// harness process runs `main`, which configures the allocator before any
/// test. Without mimalloc both tests check nothing.

#include "test_harness.hpp"

#include "../memory_infra/heap_allocator.hpp"

#include <cstddef>
#include <cstdint>
#include <new>

#ifdef USE_MIMALLOC
#include <mimalloc.h>

namespace {
    // The process commit charge in bytes, as mimalloc reads it from the
    // operating system.
    int64_t processCommitBytes() {
        std::size_t elapsed = 0, user = 0, system = 0, rss = 0, peakRss = 0;
        std::size_t commit = 0, peakCommit = 0, faults = 0;
        mi_process_info(&elapsed, &user, &system, &rss, &peakRss, &commit,
                        &peakCommit, &faults);
        return static_cast<int64_t>(commit);
    }
}
#endif

TEST(heap_allocator, configured_arenas_fill_whole_bitmap_fields) {
    gl::configureHeapAllocator();
#ifdef USE_MIMALLOC
    ASSERT_EQ(mi_option_get_size(mi_option_arena_reserve), std::size_t(2) << 30);
#endif
}

TEST(heap_allocator, blocks_above_64_mib_are_reused_after_free) {
#ifdef USE_MIMALLOC
    constexpr int64_t kMiB = int64_t(1) << 20;
    // The six large projection columns of IncubatorPeano1's second Phase 2
    // (census of 2026-09-28), each above the 64 MiB that 1 GiB arenas could
    // still place.
    const int64_t sizes[] = { 72 * kMiB, 88 * kMiB, 112 * kMiB,
                              96 * kMiB, 320 * kMiB, 456 * kMiB };

    // Two rounds of allocate, free and collect. The second round finds the
    // memory the first one freed, returned to the operating system or kept
    // by the allocator; a leaked block makes it commit fresh memory instead.
    int64_t allocatedCommit[2] = {};
    for (int round = 0; round < 2; ++round) {
        mi_collect(true);
        void* blocks[6] = {};
        for (int index = 0; index < 6; ++index)
            blocks[index] = ::operator new(static_cast<std::size_t>(sizes[index]));
        allocatedCommit[round] = processCommitBytes();
        for (int index = 0; index < 6; ++index) ::operator delete(blocks[index]);
        mi_collect(true);
    }

    // The second round commits no more than the first (one 32 MiB segment
    // of slack for allocator bookkeeping).
    ASSERT_GE(allocatedCommit[0] + 32 * kMiB, allocatedCommit[1]);
#endif
}
