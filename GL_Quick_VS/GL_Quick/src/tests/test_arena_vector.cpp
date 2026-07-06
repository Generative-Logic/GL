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
/// @brief Unit tests for the arena-backed vector (`ArenaVector`, the
///        scattered-elements-plus-offset-index cold container).
///
/// @details
/// Covers push_back / index across spine growth, element-reference stability
/// across growth (the offset values survive spine reallocation), the
/// `std::vector::erase` shift contract against an oracle, the wipeSubtree
/// back-to-front filter-erase pattern, clear / release reuse, cross-arena
/// copy-assign (the LB-clone path), assign-from-`std::vector` (scope
/// teardown), and the canonical element-stream deload round-trip
/// (`appendSpanBytes` -> `bulkAppendBytes` byte-identical).

#include "test_harness.hpp"

#include "../memory_infra/arena_vector.hpp"
#include "../memory_infra/lb_arena.hpp"
#include "../parameters.hpp"

#include <cstddef>
#include <cstring>
#include <vector>

namespace {
    // 1 MiB pool / 256 KiB block — 4 blocks.
    const gl::StaticMemoryConfig kAVTestCfg{ 1 << 20, 1 << 18 };

    // 6-byte trivially-copyable element, as in the paged-vector suite.
    struct Pod6 {
        int16_t a;
        int16_t b;
        int16_t c;
    };
}

TEST(arena_vector, push_back_size_and_index) {
    gl::GlobalMemoryManager g;
    g.init(kAVTestCfg);
    gl::LbArena arena(&g);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::ArenaVector<Pod6> v(&arena, &dirty);
    ASSERT_TRUE(v.empty());
    for (int32_t i = 0; i < 2000; ++i)
        v.push_back(Pod6{ static_cast<int16_t>(i & 0x7FFF),
                          static_cast<int16_t>((i + 1) & 0x7FFF),
                          static_cast<int16_t>((i + 2) & 0x7FFF) });
    ASSERT_EQ(v.size(), 2000);
    ASSERT_FALSE(v.empty());
    ASSERT_EQ(v[0].a, static_cast<int16_t>(0));
    ASSERT_EQ(v[1999].a, static_cast<int16_t>(1999));
    ASSERT_EQ(v[1999].c, static_cast<int16_t>(2001));
    // The spine grew many times (8 -> 16 -> ... -> 2048); every element
    // still reads back, scattered through the arena and reassembled by it.
    ASSERT_NE(static_cast<int>(dirty), static_cast<int>(gl::DirtyState::Clean));
}

TEST(arena_vector, element_references_stable_across_growth) {
    gl::GlobalMemoryManager g;
    g.init(kAVTestCfg);
    gl::LbArena arena(&g);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::ArenaVector<Pod6> v(&arena, &dirty);
    v.push_back(Pod6{ 7, 8, 9 });
    const Pod6* first = &v[0];
    for (int32_t i = 0; i < 3000; ++i) v.push_back(Pod6{ 1, 2, 3 });
    // An element never moves (the arena only bumps forward); a spine growth
    // relocates the offset run but copies the offset VALUES verbatim, so the
    // resolved element address is unchanged.
    ASSERT_EQ(&v[0], first);
    ASSERT_EQ(v[0].a, static_cast<int16_t>(7));
    ASSERT_EQ(v[0].c, static_cast<int16_t>(9));
}

TEST(arena_vector, interleaved_vectors_scatter_and_read_back) {
    gl::GlobalMemoryManager g;
    g.init(kAVTestCfg);
    gl::LbArena arena(&g);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::ArenaVector<Pod6> a(&arena, &dirty);
    gl::ArenaVector<Pod6> b(&arena, &dirty);
    // Interleave so the two vectors' elements are scattered through the
    // shared arena — the whole point of the model.
    for (int16_t i = 0; i < 1500; ++i) {
        a.push_back(Pod6{ i, 0, 0 });
        b.push_back(Pod6{ static_cast<int16_t>(-i), 1, 1 });
    }
    ASSERT_EQ(a.size(), 1500);
    ASSERT_EQ(b.size(), 1500);
    for (int32_t i = 0; i < 1500; ++i) {
        ASSERT_EQ(a[i].a, static_cast<int16_t>(i));
        ASSERT_EQ(b[i].a, static_cast<int16_t>(-i));
        ASSERT_EQ(b[i].b, static_cast<int16_t>(1));
    }
}

TEST(arena_vector, erase_matches_std_vector_oracle) {
    gl::GlobalMemoryManager g;
    g.init(kAVTestCfg);
    gl::LbArena arena(&g);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::ArenaVector<Pod6> v(&arena, &dirty);
    std::vector<Pod6> oracle;
    uint32_t seed = 12345u;
    auto next = [&seed]() {
        seed = seed * 1664525u + 1013904223u;
        return seed;
    };
    for (int op = 0; op < 4000; ++op) {
        if (!oracle.empty() && next() % 3 == 0) {
            const int32_t i = static_cast<int32_t>(next() % oracle.size());
            v.erase(i);
            oracle.erase(oracle.begin() + i);
        }
        else {
            const Pod6 p{ static_cast<int16_t>(next() & 0x7FFF),
                          static_cast<int16_t>(next() & 0x7FFF),
                          static_cast<int16_t>(next() & 0x7FFF) };
            v.push_back(p);
            oracle.push_back(p);
        }
    }
    ASSERT_EQ(v.size(), static_cast<int32_t>(oracle.size()));
    for (int32_t i = 0; i < v.size(); ++i) {
        ASSERT_EQ(v[i].a, oracle[i].a);
        ASSERT_EQ(v[i].b, oracle[i].b);
        ASSERT_EQ(v[i].c, oracle[i].c);
    }
}

TEST(arena_vector, back_to_front_filter_erase) {
    // The wipeSubtree pattern: back-to-front erase of matching elements.
    gl::GlobalMemoryManager g;
    g.init(kAVTestCfg);
    gl::LbArena arena(&g);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::ArenaVector<Pod6> v(&arena, &dirty);
    for (int32_t i = 0; i < 100; ++i)
        v.push_back(Pod6{ static_cast<int16_t>(i),
                          static_cast<int16_t>(i % 2), 0 });
    for (int32_t i = v.size(); i-- > 0;)
        if (v[i].b == 1) v.erase(i);
    ASSERT_EQ(v.size(), 50);
    for (int32_t i = 0; i < v.size(); ++i) {
        ASSERT_EQ(v[i].a, static_cast<int16_t>(2 * i));
        ASSERT_EQ(v[i].b, static_cast<int16_t>(0));
    }
}

TEST(arena_vector, clear_and_release_reuse) {
    gl::GlobalMemoryManager g;
    g.init(kAVTestCfg);
    gl::LbArena arena(&g);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::ArenaVector<Pod6> v(&arena, &dirty);
    for (int32_t i = 0; i < 100; ++i) v.push_back(Pod6{ 1, 2, 3 });
    v.clear();
    ASSERT_EQ(v.size(), 0);
    ASSERT_TRUE(v.empty());
    for (int32_t i = 0; i < 50; ++i)
        v.push_back(Pod6{ static_cast<int16_t>(i), 0, 0 });
    ASSERT_EQ(v.size(), 50);
    ASSERT_EQ(v[0].a, static_cast<int16_t>(0));
    ASSERT_EQ(v[49].a, static_cast<int16_t>(49));
    v.release();
    ASSERT_EQ(v.size(), 0);
    for (int32_t i = 0; i < 10; ++i) v.push_back(Pod6{ 5, 5, 5 });
    ASSERT_EQ(v.size(), 10);
    ASSERT_EQ(v[9].a, static_cast<int16_t>(5));
}

TEST(arena_vector, copy_assign_across_arenas) {
    gl::GlobalMemoryManager g;
    g.init(kAVTestCfg);
    gl::LbArena arenaA(&g);
    gl::LbArena arenaB(&g);
    gl::DirtyState dirtyA = gl::DirtyState::Clean;
    gl::DirtyState dirtyB = gl::DirtyState::Clean;
    gl::ArenaVector<Pod6> a(&arenaA, &dirtyA);
    gl::ArenaVector<Pod6> b(&arenaB, &dirtyB);
    for (int32_t i = 0; i < 1100; ++i)
        b.push_back(Pod6{ static_cast<int16_t>(i), 0, 0 });
    a = b;  // cross-arena deep copy (the LB-clone path)
    ASSERT_EQ(a.size(), b.size());
    ASSERT_NE(&a[0], &b[0]);  // distinct storage in distinct arenas
    for (int32_t i = 0; i < a.size(); ++i) ASSERT_EQ(a[i].a, b[i].a);
    b.clear();
    ASSERT_EQ(a.size(), 1100);  // source teardown leaves the copy intact
    ASSERT_EQ(a[1099].a, static_cast<int16_t>(1099));
}

TEST(arena_vector, assign_from_std_vector) {
    gl::GlobalMemoryManager g;
    g.init(kAVTestCfg);
    gl::LbArena arena(&g);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::ArenaVector<Pod6> v(&arena, &dirty);
    v.push_back(Pod6{ 9, 9, 9 });  // pre-existing content must vanish
    std::vector<Pod6> src;
    for (int32_t i = 0; i < 50; ++i)
        src.push_back(Pod6{ static_cast<int16_t>(i), 1, 2 });
    v = src;
    ASSERT_EQ(v.size(), 50);
    ASSERT_EQ(v[0].a, static_cast<int16_t>(0));
    ASSERT_EQ(v[49].a, static_cast<int16_t>(49));
}

TEST(arena_vector, deload_stream_round_trip) {
    gl::GlobalMemoryManager g;
    g.init(kAVTestCfg);
    gl::LbArena arena(&g);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::ArenaVector<Pod6> v(&arena, &dirty);
    for (int32_t i = 0; i < 777; ++i)
        v.push_back(Pod6{ static_cast<int16_t>(i),
                          static_cast<int16_t>(i * 3),
                          static_cast<int16_t>(i * 7) });
    std::vector<char> buf;
    v.appendSpanBytes(buf, 0);
    ASSERT_EQ(static_cast<int32_t>(buf.size()),
              static_cast<int32_t>(777 * sizeof(Pod6)));
    // Rebuild element-by-element into a fresh vector on a second arena; the
    // canonical stream carries logical content only, never offsets.
    gl::LbArena arena2(&g);
    gl::DirtyState dirty2 = gl::DirtyState::Clean;
    gl::ArenaVector<Pod6> w(&arena2, &dirty2);
    w.bulkAppendBytes(buf.data(), 777);
    ASSERT_EQ(w.size(), 777);
    for (int32_t i = 0; i < 777; ++i) {
        ASSERT_EQ(w[i].a, v[i].a);
        ASSERT_EQ(w[i].b, v[i].b);
        ASSERT_EQ(w[i].c, v[i].c);
    }
    // Partial dump from a start row (the tail-delta window shape).
    std::vector<char> buf2;
    v.appendSpanBytes(buf2, 700);
    ASSERT_EQ(static_cast<int32_t>(buf2.size()),
              static_cast<int32_t>((777 - 700) * sizeof(Pod6)));
}

TEST(arena_vector, exceeds_single_block_spine) {
    // A vector with more than 65536 elements — past one block of 4-byte
    // spine offsets — must be indexed by the two-level spine (the cap that
    // crashed the Gauss batch on the single-level spine). 16 MiB pool.
    const gl::StaticMemoryConfig kBigCfg{ 16 << 20, 1 << 18 };
    gl::GlobalMemoryManager g;
    g.init(kBigCfg);
    gl::LbArena arena(&g);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::ArenaVector<Pod6> v(&arena, &dirty);
    const int32_t n = 70000;
    for (int32_t i = 0; i < n; ++i)
        v.push_back(Pod6{ static_cast<int16_t>(i & 0x7FFF),
                          static_cast<int16_t>((i >> 1) & 0x7FFF), 0 });
    ASSERT_EQ(v.size(), n);
    // Spot-check across many chunk boundaries (chunk = 1024 elements).
    for (int32_t i = 0; i < n; i += 257) {
        ASSERT_EQ(v[i].a, static_cast<int16_t>(i & 0x7FFF));
        ASSERT_EQ(v[i].b, static_cast<int16_t>((i >> 1) & 0x7FFF));
    }
    ASSERT_EQ(v[n - 1].a, static_cast<int16_t>((n - 1) & 0x7FFF));
    // Erase near the front shifts the spine across every chunk.
    v.erase(100);
    ASSERT_EQ(v.size(), n - 1);
    ASSERT_EQ(v[100].a, static_cast<int16_t>(101 & 0x7FFF));
    ASSERT_EQ(v[n - 2].a, static_cast<int16_t>((n - 1) & 0x7FFF));
}
