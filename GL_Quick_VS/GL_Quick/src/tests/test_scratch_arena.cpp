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
/// @brief Unit tests for the per-worker scratch string arenas (string
///        statification, scratch / cold half).
///
/// @details
/// Covers the COLD byte-bump scratch surface of `ScratchArena` (= `LbArena`
/// bound cold via `bind`): lazy acquisition, exact-length bump allocation,
/// block advance with no-straddle padding, mark / rewind with poisoning (cold
/// `kArenaPoisonByte`), wholesale reset with generation bump, and the per-task
/// `releaseAll` (blocks RETURNED to the pool, generation bumped, no retention).
/// Plus the `ScratchArenaRegistry` lifecycle: per-slot distinct / stable
/// arenas, no reserve cap, idempotent same-shape re-init. Scratch fill rides
/// the BYTE tier (`allocBytes` = a resolved `alloc`), never the page tier, so
/// it never collides with `allocPage` containers on the same arena.

#include "test_harness.hpp"

#include "../memory_infra/scratch_arena.hpp"

#include <cstring>

namespace {
    // 1 MiB pool / 256 KiB block — 4 blocks, as in the static-memory suite.
    const gl::StaticMemoryConfig kScratchTestCfg{ 1 << 20, 1 << 18 };
}

TEST(scratch_arena, lazy_acquisition_and_exact_length_bump) {
    gl::GlobalMemoryManager m;
    m.init(kScratchTestCfg);
    gl::ScratchArena a;
    a.bind(&m);
    // Idle arena holds nothing — blocks acquire lazily on first fill.
    ASSERT_EQ(a.blocksHeld(), 0);
    ASSERT_EQ(m.blocksInUse(), static_cast<int64_t>(0));
    char* p1 = a.allocBytes(16);
    char* p2 = a.allocBytes(7);
    char* p3 = a.allocBytes(1);
    // One byte-bump block; the page tier is untouched.
    ASSERT_EQ(a.blocksHeld(), 1);
    ASSERT_EQ(m.blocksInUse(), static_cast<int64_t>(1));
    // Exact-length bump: consecutive, non-overlapping, writable.
    ASSERT_EQ(p2, p1 + 16);
    ASSERT_EQ(p3, p2 + 7);
    std::memset(p1, 0xA5, 16);
    std::memset(p2, 0x5A, 7);
    ASSERT_EQ(p1[15], static_cast<char>(0xA5));
    ASSERT_EQ(p2[6], static_cast<char>(0x5A));
    ASSERT_EQ(a.usedBytes(), static_cast<int64_t>(24));
    ASSERT_EQ(a.peakUsedBytes(), static_cast<int64_t>(24));
}

TEST(scratch_arena, block_advance_no_straddle) {
    gl::GlobalMemoryManager m;
    m.init(kScratchTestCfg);
    gl::ScratchArena a;
    a.bind(&m);
    const int32_t bb = m.blockBytes();
    // Fill block 0 to 32 bytes short of full, then ask for 64 — the request
    // cannot fit the 32-byte remainder, so the byte-bump pads to the next
    // block rather than straddle. The page tier is never touched.
    char* first = a.allocBytes(bb - 32);     // block 0
    char* second = a.allocBytes(64);         // pads to block 1
    ASSERT_EQ(a.blocksHeld(), 2);
    ASSERT_EQ(m.blocksInUse(), static_cast<int64_t>(2));
    ASSERT_EQ(a.pageHighWater(), 0);
    // usedBytes is the byte cursor: block 0 padded full (bb) plus the 64.
    ASSERT_EQ(a.usedBytes(), static_cast<int64_t>(bb) + 64);
    // The two runs are in different blocks — writing one does not touch the
    // other (no straddle, no overlap).
    std::memset(first, 0xA5, static_cast<size_t>(bb - 32));
    std::memset(second, 0x5A, 64);
    ASSERT_EQ(first[bb - 33], static_cast<char>(0xA5));
    ASSERT_EQ(second[63], static_cast<char>(0x5A));
}

TEST(scratch_arena, mark_rewind_reuses_and_poisons) {
    gl::GlobalMemoryManager m;
    m.init(kScratchTestCfg);
    gl::ScratchArena a;
    a.bind(&m);
    char* keep = a.allocBytes(8);
    std::memset(keep, 0x11, 8);
    const gl::ScratchArena::Mark mk = a.mark();
    char* scratch = a.allocBytes(32);
    std::memset(scratch, 0x22, 32);
    const uint64_t genBefore = a.generation();
    a.rewind(mk);
    // Rewind does not bump the generation (same scope) and poisons the
    // freed span with the COLD poison byte; the kept prefix is untouched.
    ASSERT_EQ(a.generation(), genBefore);
    ASSERT_EQ(scratch[0], static_cast<char>(gl::kArenaPoisonByte));
    ASSERT_EQ(scratch[31], static_cast<char>(gl::kArenaPoisonByte));
    ASSERT_EQ(keep[7], static_cast<char>(0x11));
    // The next allocation reuses the rewound space.
    char* again = a.allocBytes(32);
    ASSERT_EQ(again, scratch);
}

TEST(scratch_arena, rewind_across_blocks_keeps_retained_blocks) {
    gl::GlobalMemoryManager m;
    m.init(kScratchTestCfg);
    gl::ScratchArena a;
    a.bind(&m);
    const int32_t bb = m.blockBytes();
    const gl::ScratchArena::Mark start = a.mark();
    a.allocBytes(bb);                        // block 0 (full)
    char* inSecond = a.allocBytes(128);      // block 1
    std::memset(inSecond, 0x33, 128);
    ASSERT_EQ(a.blocksHeld(), 2);
    a.rewind(start);
    // The cursor rolls back across the block boundary; both carved blocks stay
    // retained (popTo keeps blocks), the freed span is poisoned.
    ASSERT_EQ(a.blocksHeld(), 2);
    ASSERT_EQ(m.blocksInUse(), static_cast<int64_t>(2));
    ASSERT_EQ(a.usedBytes(), static_cast<int64_t>(0));
    ASSERT_EQ(inSecond[0], static_cast<char>(gl::kArenaPoisonByte));
    // Refilling advances through the SAME retained blocks, acquiring nothing
    // new.
    a.allocBytes(bb);
    char* refill = a.allocBytes(128);
    ASSERT_EQ(refill, inSecond);
    ASSERT_EQ(m.blocksInUse(), static_cast<int64_t>(2));
}

TEST(scratch_arena, reset_bumps_generation_retains_blocks) {
    gl::GlobalMemoryManager m;
    m.init(kScratchTestCfg);
    gl::ScratchArena a;
    a.bind(&m);
    char* p = a.allocBytes(64);
    std::memset(p, 0x44, 64);
    const uint64_t genBefore = a.generation();
    a.reset();
    ASSERT_EQ(a.generation(), genBefore + 1);
    ASSERT_EQ(a.usedBytes(), static_cast<int64_t>(0));
    ASSERT_EQ(a.blocksHeld(), 1);                     // reset retains blocks
    ASSERT_EQ(p[0], static_cast<char>(gl::kArenaPoisonByte));
    ASSERT_EQ(p[63], static_cast<char>(gl::kArenaPoisonByte));
    // Peak survives the reset (lifetime telemetry).
    ASSERT_EQ(a.peakUsedBytes(), static_cast<int64_t>(64));
    // The cursor restarts at the front of the retained storage.
    char* q = a.allocBytes(64);
    ASSERT_EQ(q, p);
}

TEST(scratch_arena, release_all_returns_blocks_per_task) {
    gl::GlobalMemoryManager m;
    m.init(kScratchTestCfg);
    gl::ScratchArena a;
    a.bind(&m);
    a.allocBytes(32);
    ASSERT_EQ(a.blocksHeld(), 1);
    ASSERT_EQ(m.blocksInUse(), static_cast<int64_t>(1));
    const uint64_t genBefore = a.generation();
    // The per-task release: blocks go BACK to the pool (no retention) and the
    // generation bumps so an escaped ScratchString from the finished task
    // asserts.
    a.releaseAll();
    ASSERT_EQ(a.blocksHeld(), 0);
    ASSERT_EQ(m.blocksInUse(), static_cast<int64_t>(0));
    ASSERT_EQ(a.generation(), genBefore + 1);
    // Still bound: the next task re-acquires lazily.
    char* p = a.allocBytes(16);
    std::memset(p, 0x55, 16);
    ASSERT_EQ(a.blocksHeld(), 1);
    a.releaseAll();
    ASSERT_EQ(m.blocksInUse(), static_cast<int64_t>(0));
}

TEST(scratch_arena, registry_slots_are_distinct_and_stable) {
    gl::GlobalMemoryManager m;
    m.init(kScratchTestCfg);
    gl::ScratchArenaRegistry reg;
    ASSERT_FALSE(reg.initialized());
    reg.init(&m, 3);                                 // no reserve cap
    ASSERT_TRUE(reg.initialized());
    ASSERT_EQ(reg.slotCount(), 3u);
    gl::ScratchArena* s0 = &reg.forSlot(0);
    gl::ScratchArena* s1 = &reg.forSlot(1);
    gl::ScratchArena* s2 = &reg.forSlot(2);
    ASSERT_NE(s0, s1);
    ASSERT_NE(s1, s2);
    // Slots allocate independently.
    char* a = s0->allocBytes(8);
    char* b = s1->allocBytes(8);
    ASSERT_NE(a, b);
    ASSERT_EQ(s2->blocksHeld(), 0);
    // Per-task release on one slot leaves the others untouched.
    s0->releaseAll();
    ASSERT_EQ(s0->blocksHeld(), 0);
    ASSERT_EQ(s1->blocksHeld(), 1);
    // Idempotent same-shape re-init; addresses stay stable.
    reg.init(&m, 3);
    ASSERT_EQ(&reg.forSlot(0), s0);
    ASSERT_EQ(&reg.forSlot(2), s2);
    s1->releaseAll();
}
