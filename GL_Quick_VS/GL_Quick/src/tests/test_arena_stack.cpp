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
/// @brief Unit tests for `ArenaStack<T>` — the LIFO-with-popTo stack on an
///        `LbArena` byte-bump tier.
///
/// @details
/// Covers the contract the depth-first frontier relies on: LIFO push / back /
/// pop ordering, empty / size, the popTo reclaim that returns the byte cursor
/// exactly to a previous position (so the footprint tracks the live frontier,
/// not total pushes), correctness when frames span more than one pool block,
/// and the grow / backtrack / regrow cycle.

#include "test_harness.hpp"

#include "../memory_infra/arena_stack.hpp"
#include "../memory_infra/lb_arena.hpp"
#include "../memory_infra/global_memory_manager.hpp"

#include <cstdint>

namespace {
    // 1 MiB pool / 256 KiB block — 4 blocks, as in the lb_arena suite.
    const gl::StaticMemoryConfig kStackTestCfg{ 1 << 20, 1 << 18 };

    // A small POD with an array member, to exercise alignment + a frame larger
    // than a scalar (mirrors the prover's StackItem shape).
    struct Item {
        int32_t a;
        int16_t b[4];
    };
}

TEST(arena_stack, push_back_pop_lifo) {
    gl::GlobalMemoryManager m;
    m.init(kStackTestCfg);
    gl::LbArena arena(&m);
    gl::ArenaStack<int> s(arena);

    ASSERT_TRUE(s.empty());
    ASSERT_EQ(s.size(), 0);

    s.push(10);
    s.push(20);
    s.push(30);
    ASSERT_FALSE(s.empty());
    ASSERT_EQ(s.size(), 3);
    ASSERT_EQ(s.back(), 30);

    s.pop();
    ASSERT_EQ(s.back(), 20);
    ASSERT_EQ(s.size(), 2);
    s.pop();
    ASSERT_EQ(s.back(), 10);
    s.pop();
    ASSERT_TRUE(s.empty());
    ASSERT_EQ(s.size(), 0);
}

TEST(arena_stack, popto_reclaims_cursor_exactly) {
    gl::GlobalMemoryManager m;
    m.init(kStackTestCfg);
    gl::LbArena arena(&m);
    gl::ArenaStack<int> s(arena);

    const gl::ArenaOffset c0 = arena.cursor();
    ASSERT_EQ(c0, static_cast<gl::ArenaOffset>(0));
    for (int i = 0; i < 100; ++i) s.push(i);
    ASSERT_TRUE(arena.cursor() > c0);
    for (int i = 0; i < 100; ++i) s.pop();
    ASSERT_TRUE(s.empty());
    // LIFO popTo returns the byte cursor exactly to the start.
    ASSERT_EQ(arena.cursor(), c0);
}

TEST(arena_stack, values_survive_block_spanning) {
    gl::GlobalMemoryManager m;
    m.init(kStackTestCfg);
    gl::LbArena arena(&m);
    gl::ArenaStack<Item> s(arena);

    // Frame ~ ArenaOffset(4) + Item(12) = 16 bytes; 256 KiB / 16 = 16384 frames
    // per block, so 20000 pushes spans at least two blocks.
    const int N = 20000;
    for (int i = 0; i < N; ++i) {
        Item it{ i, { static_cast<int16_t>(i & 7),
                      static_cast<int16_t>((i >> 1) & 7), 0, 0 } };
        s.push(it);
    }
    ASSERT_EQ(s.size(), N);
    ASSERT_TRUE(arena.blocksHeld() >= static_cast<int64_t>(2));

    // Pop all in LIFO order, verifying every value survived the spanning.
    for (int i = N - 1; i >= 0; --i) {
        const Item it = s.back();
        ASSERT_EQ(it.a, i);
        ASSERT_EQ(it.b[0], static_cast<int16_t>(i & 7));
        ASSERT_EQ(it.b[1], static_cast<int16_t>((i >> 1) & 7));
        s.pop();
    }
    ASSERT_TRUE(s.empty());
    ASSERT_EQ(arena.cursor(), static_cast<gl::ArenaOffset>(0));
}

TEST(arena_stack, grow_backtrack_regrow) {
    // The depth-first pattern: grow, backtrack partway, grow again. popTo
    // reclaim means equal live depth lands the cursor at the same position.
    gl::GlobalMemoryManager m;
    m.init(kStackTestCfg);
    gl::LbArena arena(&m);
    gl::ArenaStack<int> s(arena);

    for (int i = 0; i < 50; ++i) s.push(i);
    const gl::ArenaOffset peak = arena.cursor();

    for (int i = 0; i < 30; ++i) s.pop();
    ASSERT_TRUE(arena.cursor() < peak);
    ASSERT_EQ(s.size(), 20);
    ASSERT_EQ(s.back(), 19);

    for (int i = 0; i < 30; ++i) s.push(100 + i);
    ASSERT_EQ(s.size(), 50);
    ASSERT_EQ(s.back(), 129);
    // 50 equally-sized live frames again -> cursor back at the same peak.
    ASSERT_EQ(arena.cursor(), peak);
}
