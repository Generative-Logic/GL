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
/// @brief Unit tests for `PagedVector` — the contiguous-within-page statified
///        container that backs the paged cold containers.
///
/// @details
/// Covers the adaptive in-arena directory (empty -> single inline data-page vid
/// -> directory page) across the page boundary, element-reference stability
/// across growth, the `std::vector::erase` shift-left contract (against a live
/// oracle), back-to-front filter erase, clear/tail-shrink page reclamation with
/// directory demotion, cross-arena deep copy, assign-from-`std::vector`, and the
/// bulk reload path. The load-bearing test is `deload_bytes_match_arena_vector`:
/// a `PagedVector` and an `ArenaVector` with identical content must serialise to
/// byte-identical canonical streams, so swapping a container's element type
/// leaves its deload image — and the proof output — unchanged.

#include "test_harness.hpp"

#include "../memory_infra/arena_vector.hpp"
#include "../memory_infra/paged_vector.hpp"
#include "../memory_infra/lb_arena.hpp"
#include "../parameters.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace {
    // 1 MiB pool / 256 KiB block / 8 KiB page (the default) — 4 blocks,
    // 32 pages per block.
    const gl::StaticMemoryConfig kPvCfg{ 1 << 20, 1 << 18 };

    // 6-byte trivially-copyable element. At an 8 KiB page that is
    // floor(8192 / 6) = 1365 -> 1024 elements per page (largest power of two),
    // so the page boundary lands at index 1024.
    struct PvPod {
        int16_t a;
        int16_t b;
        int16_t c;
        bool operator==(const PvPod& o) const {
            return a == o.a && b == o.b && c == o.c;
        }
    };

    PvPod pod(int v) {
        return PvPod{ static_cast<int16_t>(v & 0x7FFF),
                      static_cast<int16_t>((v + 1) & 0x7FFF),
                      static_cast<int16_t>((v + 2) & 0x7FFF) };
    }
}

TEST(paged_vector, push_and_index_across_page_boundary) {
    gl::GlobalMemoryManager m;
    m.init(kPvCfg);
    gl::LbArena a(&m);
    // The container reads its geometry from the arena's page size.
    ASSERT_EQ(a.pageBytes(), 1 << 13);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::PagedVector<PvPod> v(&a, &d);
    ASSERT_TRUE(v.empty());
    for (int i = 0; i < 1025; ++i) v.push_back(pod(i));   // crosses 1 page
    ASSERT_EQ(v.size(), 1025);
    ASSERT_FALSE(v.empty());
    ASSERT_EQ(v.pagesHeld(), 2);
    ASSERT_TRUE(v[0] == pod(0));
    ASSERT_TRUE(v[1023] == pod(1023));   // last of page 0
    ASSERT_TRUE(v[1024] == pod(1024));   // first of page 1
    ASSERT_NE(static_cast<int>(d), static_cast<int>(gl::DirtyState::Clean));
}

TEST(paged_vector, add_scalar_to_suffix_crosses_pages_byte_identically) {
    gl::GlobalMemoryManager m;
    m.init(kPvCfg);
    gl::LbArena a(&m);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::PagedVector<int32_t> v(&a, &d);
    std::vector<int32_t> oracle;
    for (int32_t i = 0; i < 5000; ++i) {
        v.push_back(i * 3);
        oracle.push_back(i * 3);
    }

    d = gl::DirtyState::Clean;
    const int32_t first = 1023;
    const int32_t delta = -17;
    v.addScalarToSuffix(first, delta);
    for (int32_t i = first; i < static_cast<int32_t>(oracle.size()); ++i)
        oracle[static_cast<std::size_t>(i)] += delta;

    ASSERT_EQ(static_cast<int>(d),
              static_cast<int>(gl::DirtyState::Restructured));
    for (int32_t i = 0; i < v.size(); ++i)
        ASSERT_EQ(v[i], oracle[static_cast<std::size_t>(i)]);

    d = gl::DirtyState::Clean;
    v.addScalarToSuffix(v.size(), 9);
    v.addScalarToSuffix(0, 0);
    ASSERT_EQ(static_cast<int>(d), static_cast<int>(gl::DirtyState::Clean));
    for (int32_t i = 0; i < v.size(); ++i)
        ASSERT_EQ(v[i], oracle[static_cast<std::size_t>(i)]);
}

TEST(paged_vector, single_page_uses_no_directory_page) {
    gl::GlobalMemoryManager m;
    m.init(kPvCfg);
    gl::LbArena a(&m);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::PagedVector<PvPod> v(&a, &d);
    for (int i = 0; i < 1024; ++i) v.push_back(pod(i));   // exactly one page
    ASSERT_EQ(v.pagesHeld(), 1);
    ASSERT_EQ(a.livePages(), 1);          // the single data page, no directory
    v.push_back(pod(1024));               // the 1025th promotes to a directory
    ASSERT_EQ(v.pagesHeld(), 2);
    ASSERT_EQ(a.livePages(), 3);          // 2 data pages + 1 directory page
}

TEST(paged_vector, references_stable_across_growth) {
    gl::GlobalMemoryManager m;
    m.init(kPvCfg);
    gl::LbArena a(&m);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::PagedVector<PvPod> v(&a, &d);
    v.push_back(PvPod{ 7, 8, 9 });
    const PvPod* p0 = &v[0];
    for (int i = 1; i < 3001; ++i) v.push_back(pod(i));   // pages never move
    ASSERT_EQ(p0, &v[0]);
    ASSERT_TRUE(*p0 == (PvPod{ 7, 8, 9 }));
}

TEST(paged_vector, erase_matches_std_vector_oracle) {
    gl::GlobalMemoryManager m;
    m.init(kPvCfg);
    gl::LbArena a(&m);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::PagedVector<PvPod> v(&a, &d);
    std::vector<PvPod> oracle;
    // Deterministic LCG so the op sequence is fixed across runs.
    uint32_t rng = 0x12345678u;
    auto next = [&]() { rng = rng * 1664525u + 1013904223u; return rng; };
    for (int step = 0; step < 4000; ++step) {
        if (oracle.empty() || (next() & 1u)) {
            const PvPod e = pod(static_cast<int>(next()));
            v.push_back(e);
            oracle.push_back(e);
        } else {
            const int32_t idx =
                static_cast<int32_t>(next() % static_cast<uint32_t>(oracle.size()));
            v.erase(idx);
            oracle.erase(oracle.begin() + idx);
        }
    }
    ASSERT_EQ(v.size(), static_cast<int32_t>(oracle.size()));
    for (int32_t i = 0; i < v.size(); ++i) ASSERT_TRUE(v[i] == oracle[i]);
}

TEST(paged_vector, replace_range_matches_std_vector_oracle) {
    // replaceRange's BULK page-span shift (blockMove) + fill (writeBytesAt) must
    // match std::vector across page boundaries: small grows / shrinks / same-size
    // at random positions over ~5 pages, so the moved tail repeatedly spans
    // several pages (the path the single-page cold-map tests never reach).
    gl::GlobalMemoryManager m;
    m.init(kPvCfg);
    gl::LbArena a(&m);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::PagedVector<PvPod> v(&a, &d);
    std::vector<PvPod> oracle;
    for (int i = 0; i < 5200; ++i) {                  // ~5 pages
        const PvPod e = pod(i);
        v.push_back(e);
        oracle.push_back(e);
    }
    uint32_t rng = 0x9e3779b9u;
    auto next = [&]() { rng = rng * 1664525u + 1013904223u; return rng; };
    for (int step = 0; step < 1500; ++step) {
        const int32_t sz = static_cast<int32_t>(oracle.size());
        const int32_t pos =
            static_cast<int32_t>(next() % static_cast<uint32_t>(sz + 1));
        const int32_t maxOld = sz - pos;
        int32_t oldLen = static_cast<int32_t>(next() % 7u);   // small replace
        if (oldLen > maxOld) oldLen = maxOld;
        const int32_t newLen = static_cast<int32_t>(next() % 9u);
        std::vector<PvPod> src;
        for (int32_t k = 0; k < newLen; ++k)
            src.push_back(pod(static_cast<int>(next())));
        v.replaceRange(pos, oldLen, src.empty() ? nullptr : src.data(), newLen);
        oracle.erase(oracle.begin() + pos, oracle.begin() + pos + oldLen);
        oracle.insert(oracle.begin() + pos, src.begin(), src.end());
    }
    ASSERT_EQ(v.size(), static_cast<int32_t>(oracle.size()));
    for (int32_t i = 0; i < v.size(); ++i) ASSERT_TRUE(v[i] == oracle[i]);
}

TEST(paged_vector, replace_range_generated_matches_segmented_oracle) {
    gl::GlobalMemoryManager m;
    m.init(kPvCfg);
    gl::LbArena a(&m);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::PagedVector<PvPod> v(&a, &d);
    std::vector<PvPod> oracle;
    for (int i = 0; i < 5200; ++i) {
        const PvPod e = pod(i);
        v.push_back(e);
        oracle.push_back(e);
    }
    std::vector<PvPod> first, second;
    for (int i = 0; i < 1700; ++i) first.push_back(pod(10000 + i));
    for (int i = 0; i < 1900; ++i) second.push_back(pod(20000 + i));
    constexpr int32_t pos = 900;
    constexpr int32_t oldLen = 2500;
    const int32_t newLen = static_cast<int32_t>(first.size() + second.size());
    v.replaceRangeGenerated(pos, oldLen, newLen, [&](const auto& sink) {
        sink(first.data(), static_cast<int32_t>(first.size()));
        sink(second.data(), static_cast<int32_t>(second.size()));
    });
    oracle.erase(oracle.begin() + pos, oracle.begin() + pos + oldLen);
    oracle.insert(oracle.begin() + pos, second.begin(), second.end());
    oracle.insert(oracle.begin() + pos, first.begin(), first.end());
    ASSERT_EQ(v.size(), static_cast<int32_t>(oracle.size()));
    for (int32_t i = 0; i < v.size(); ++i) ASSERT_TRUE(v[i] == oracle[i]);
    ASSERT_EQ(static_cast<int>(d),
              static_cast<int>(gl::DirtyState::Restructured));
}

TEST(paged_vector, back_to_front_filter_erase) {
    gl::GlobalMemoryManager m;
    m.init(kPvCfg);
    gl::LbArena a(&m);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::PagedVector<PvPod> v(&a, &d);
    for (int i = 0; i < 100; ++i) v.push_back(pod(i));
    // wipeSubtree-style back-to-front erase of every even index.
    for (int32_t j = v.size(); j-- > 0;)
        if ((v[j].a % 2) == 0) v.erase(j);
    ASSERT_EQ(v.size(), 50);
    for (int32_t i = 0; i < v.size(); ++i)
        ASSERT_EQ(v[i].a % 2, static_cast<int16_t>(1));
}

TEST(paged_vector, set_at_in_place_overwrite_marks_restructured) {
    gl::GlobalMemoryManager m;
    m.init(kPvCfg);
    gl::LbArena a(&m);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::PagedVector<PvPod> v(&a, &d);
    for (int i = 0; i < 1500; ++i) v.push_back(pod(i));   // crosses a page
    d = gl::DirtyState::Clean;                            // observe the next write
    // In-place overwrite across a page boundary: the element changes in place,
    // size is unchanged, neighbours untouched, and the dirty state escalates to
    // Restructured (an in-place write is not a tail-delta-eligible append).
    v.setAt(0, pod(9000));
    v.setAt(1024, pod(9001));                             // first slot of page 1
    v.setAt(1499, pod(9002));                             // last element
    ASSERT_EQ(v.size(), 1500);
    ASSERT_TRUE(v[0] == pod(9000));
    ASSERT_TRUE(v[1024] == pod(9001));
    ASSERT_TRUE(v[1499] == pod(9002));
    ASSERT_TRUE(v[1] == pod(1));                          // neighbour untouched
    ASSERT_TRUE(v[1023] == pod(1023));
    ASSERT_EQ(static_cast<int>(d),
              static_cast<int>(gl::DirtyState::Restructured));
}

TEST(paged_vector, set_at_relaxed_overwrites_without_dirtying) {
    gl::GlobalMemoryManager m;
    m.init(kPvCfg);
    gl::LbArena a(&m);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::PagedVector<PvPod> v(&a, &d);
    for (int i = 0; i < 1500; ++i) v.push_back(pod(i));   // crosses a page
    d = gl::DirtyState::Clean;                            // observe the next write
    // The parallel-safe twin of setAt: the element changes in place exactly like
    // setAt, but the shared dirty flag is left UNTOUCHED (stays Clean). That is
    // the property that makes a disjoint parallel write to a never-deloaded
    // container race-free — no thread ever writes the one *dirty_ byte.
    v.setAtRelaxed(0, pod(9000));
    v.setAtRelaxed(1024, pod(9001));                     // first slot of page 1
    v.setAtRelaxed(1499, pod(9002));                     // last element
    ASSERT_EQ(v.size(), 1500);
    ASSERT_TRUE(v[0] == pod(9000));
    ASSERT_TRUE(v[1024] == pod(9001));
    ASSERT_TRUE(v[1499] == pod(9002));
    ASSERT_TRUE(v[1] == pod(1));                          // neighbour untouched
    ASSERT_TRUE(v[1023] == pod(1023));
    ASSERT_EQ(static_cast<int>(d),
              static_cast<int>(gl::DirtyState::Clean));   // NOT escalated
}

TEST(paged_vector, truncate_drops_tail_and_frees_pages) {
    gl::GlobalMemoryManager m;
    m.init(kPvCfg);
    gl::LbArena a(&m);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::PagedVector<PvPod> v(&a, &d);
    for (int i = 0; i < 3000; ++i) v.push_back(pod(i));   // ~3 pages (1024/page)
    ASSERT_EQ(v.pagesHeld(), 3);
    d = gl::DirtyState::Clean;                            // observe the shrink

    // Truncate into the first page: the two trailing data pages free, the
    // directory demotes to a single inline vid, survivors intact, Restructured.
    v.truncate(500);
    ASSERT_EQ(v.size(), 500);
    ASSERT_EQ(v.pagesHeld(), 1);
    for (int i = 0; i < 500; ++i) ASSERT_TRUE(v[i] == pod(i));
    ASSERT_EQ(static_cast<int>(d),
              static_cast<int>(gl::DirtyState::Restructured));

    // A partially-used final page is retained.
    v.truncate(1);
    ASSERT_EQ(v.size(), 1);
    ASSERT_EQ(v.pagesHeld(), 1);
    ASSERT_TRUE(v[0] == pod(0));

    // truncate(0) frees everything; a same-size truncate is a no-op.
    v.truncate(0);
    ASSERT_EQ(v.size(), 0);
    ASSERT_EQ(v.pagesHeld(), 0);
    ASSERT_EQ(a.livePages(), 0);
    v.truncate(0);
    ASSERT_EQ(v.size(), 0);
}

TEST(paged_vector, clear_and_tail_shrink_free_pages) {
    gl::GlobalMemoryManager m;
    m.init(kPvCfg);
    gl::LbArena a(&m);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::PagedVector<PvPod> v(&a, &d);
    for (int i = 0; i < 1025; ++i) v.push_back(pod(i));
    ASSERT_EQ(v.pagesHeld(), 2);
    ASSERT_EQ(a.livePages(), 3);          // 2 data + 1 directory
    // Erasing the 1025th element empties the 2nd data page: it is freed AND
    // the directory page is dropped (demote back to a single inline vid).
    v.erase(1024);
    ASSERT_EQ(v.pagesHeld(), 1);
    ASSERT_EQ(a.livePages(), 1);
    v.clear();
    ASSERT_EQ(v.size(), 0);
    ASSERT_EQ(v.pagesHeld(), 0);
    ASSERT_EQ(a.livePages(), 0);          // every page returned to the arena
    // release() is a legal no-op after clear().
    v.release();
    ASSERT_EQ(v.size(), 0);
}

TEST(paged_vector, copy_assign_across_arenas) {
    gl::GlobalMemoryManager m;
    m.init(kPvCfg);
    gl::LbArena a1(&m), a2(&m);
    gl::DirtyState d1 = gl::DirtyState::Clean, d2 = gl::DirtyState::Clean;
    gl::PagedVector<PvPod> src(&a1, &d1), dst(&a2, &d2);
    for (int i = 0; i < 1500; ++i) src.push_back(pod(i));
    dst.push_back(PvPod{ 99, 99, 99 });   // pre-existing, must be discarded
    dst = src;
    ASSERT_EQ(dst.size(), src.size());
    for (int32_t i = 0; i < dst.size(); ++i) ASSERT_TRUE(dst[i] == src[i]);
    ASSERT_NE(&dst[0], &src[0]);          // distinct storage
}

TEST(paged_vector, assign_from_std_vector) {
    gl::GlobalMemoryManager m;
    m.init(kPvCfg);
    gl::LbArena a(&m);
    gl::DirtyState d = gl::DirtyState::Clean;
    std::vector<PvPod> src;
    for (int i = 0; i < 1200; ++i) src.push_back(pod(i));
    gl::PagedVector<PvPod> v(&a, &d);
    v.push_back(PvPod{ 5, 5, 5 });        // discarded by the assign
    v = src;
    ASSERT_EQ(v.size(), 1200);
    for (int32_t i = 0; i < v.size(); ++i) ASSERT_TRUE(v[i] == src[i]);
}

TEST(paged_vector, bulk_append_reconstructs_content) {
    gl::GlobalMemoryManager m;
    m.init(kPvCfg);
    gl::LbArena a1(&m), a2(&m);
    gl::DirtyState d1 = gl::DirtyState::Clean, d2 = gl::DirtyState::Clean;
    gl::PagedVector<PvPod> v(&a1, &d1);
    for (int i = 0; i < 2500; ++i) v.push_back(pod(i));
    std::vector<char> bytes;
    v.appendSpanBytes(bytes, 0);
    gl::PagedVector<PvPod> w(&a2, &d2);
    w.bulkAppendBytes(bytes.data(), 2500);
    ASSERT_EQ(w.size(), 2500);
    for (int32_t i = 0; i < w.size(); ++i) ASSERT_TRUE(w[i] == v[i]);
}

TEST(paged_vector, deload_bytes_match_arena_vector) {
    // The determinism gate for the commit-5 type swap: identical content must
    // serialise identically whether the backing is paged or scattered.
    gl::GlobalMemoryManager m;
    m.init(kPvCfg);
    gl::LbArena ap(&m), aa(&m);
    gl::DirtyState dp = gl::DirtyState::Clean, da = gl::DirtyState::Clean;
    gl::PagedVector<PvPod> pv(&ap, &dp);
    gl::ArenaVector<PvPod> av(&aa, &da);
    for (int i = 0; i < 3000; ++i) {
        const PvPod e = pod(i * 7);
        pv.push_back(e);
        av.push_back(e);
    }
    std::vector<char> bp, ba;
    pv.appendSpanBytes(bp, 0);
    av.appendSpanBytes(ba, 0);
    ASSERT_EQ(static_cast<int32_t>(bp.size()), static_cast<int32_t>(ba.size()));
    ASSERT_TRUE(bp == ba);
}

TEST(paged_vector, append_run_bulk_appends_with_append_only_dirty) {
    gl::GlobalMemoryManager m;
    m.init(kPvCfg);
    gl::LbArena a(&m);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::PagedVector<PvPod> v(&a, &d);
    std::vector<PvPod> src;
    for (int i = 0; i < 2500; ++i) src.push_back(pod(i));
    v.appendRun(src.data(), static_cast<int32_t>(src.size()));   // crosses pages
    ASSERT_EQ(v.size(), 2500);
    for (int32_t i = 0; i < v.size(); ++i) ASSERT_TRUE(v[i] == src[i]);
    // Append-only: a clean container escalates to AppendedOnly, NOT Restructured.
    ASSERT_EQ(static_cast<int>(d),
              static_cast<int>(gl::DirtyState::AppendedOnly));
    v.appendRun(src.data(), 10);                                 // continues
    ASSERT_EQ(v.size(), 2510);
    ASSERT_TRUE(v[2509] == src[9]);
}

TEST(paged_vector, contiguous_run_tiles_the_vector) {
    gl::GlobalMemoryManager m;
    m.init(kPvCfg);
    gl::LbArena a(&m);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::PagedVector<PvPod> v(&a, &d);
    for (int i = 0; i < 2600; ++i) v.push_back(pod(i));          // > 2 pages
    int32_t pos = 0, runs = 0;
    while (pos < v.size()) {
        int32_t run = 0;
        const PvPod* p = v.contiguousRun(pos, run);
        ASSERT_TRUE(run >= 1);
        for (int32_t k = 0; k < run; ++k) ASSERT_TRUE(p[k] == v[pos + k]);
        pos += run;
        ++runs;
    }
    ASSERT_EQ(pos, v.size());
    ASSERT_TRUE(runs >= 3);   // tiled across at least three pages
}

TEST(paged_vector, append_run_no_straddle_pads_to_a_fresh_page) {
    gl::GlobalMemoryManager m;
    m.init(kPvCfg);
    gl::LbArena a(&m);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::PagedVector<char> v(&a, &d);
    const int32_t epp = a.pageBytes();              // chars per page (8192)
    // A near-page run fills most of page 0; a second equal run cannot share the
    // remainder, so it is padded onto a fresh page — each run reads back as one
    // contiguous span (the cold-string no-straddle guarantee).
    const std::vector<char> runA(static_cast<size_t>(epp - 8), 'a');
    const std::vector<char> runB(static_cast<size_t>(epp - 8), 'b');
    const int32_t startA =
        v.appendRunNoStraddle(runA.data(), static_cast<int32_t>(runA.size()));
    const int32_t startB =
        v.appendRunNoStraddle(runB.data(), static_cast<int32_t>(runB.size()));
    ASSERT_EQ(startA, 0);
    ASSERT_EQ(startB, epp);                         // padded past page 0's tail
    // Each run is contiguous within its page (contiguousRun spans the whole run).
    int32_t run = 0;
    const char* pa = v.contiguousRun(startA, run);
    ASSERT_TRUE(run >= static_cast<int32_t>(runA.size()));
    ASSERT_EQ(pa[0], 'a');
    ASSERT_EQ(pa[runA.size() - 1], 'a');
    const char* pb = v.contiguousRun(startB, run);
    ASSERT_TRUE(run >= static_cast<int32_t>(runB.size()));
    ASSERT_EQ(pb[0], 'b');
    ASSERT_EQ(pb[runB.size() - 1], 'b');
    // Append-only dirty: padding + fill never escalate to Restructured.
    ASSERT_EQ(static_cast<int>(d),
              static_cast<int>(gl::DirtyState::AppendedOnly));
    // The exactly-one-page boundary run (n == elemsPerPage) still fits.
    const std::vector<char> full(static_cast<size_t>(epp), 'c');
    const int32_t startFull =
        v.appendRunNoStraddle(full.data(), static_cast<int32_t>(full.size()));
    const char* pf = v.contiguousRun(startFull, run);
    ASSERT_TRUE(run >= epp);
    ASSERT_EQ(pf[epp - 1], 'c');
}

// ---------------------------------------------------------------------------
// Two-level directory spill coverage.
//
// A 256-byte page holds 64 int32 elements/page AND dirCap = 256/4 = 64
// data-page vids per directory page, so the single directory page caps at
// 64 * 64 = 4096 elements; past that the directory promotes to two-level
// (root L2 page -> level-1 directory pages -> data pages). A 4 MiB pool leaves
// room for a spill container plus its reload twin.
// ---------------------------------------------------------------------------

namespace {
    const gl::StaticMemoryConfig kPvSpillCfg{ 1 << 22, 1 << 18, 256 };
    constexpr int32_t kEppI32  = 64;     // int32 elements per 256-byte page
    constexpr int32_t kDirCap  = 64;     // data-page vids per directory page
    constexpr int32_t kSingleCap = kEppI32 * kDirCap;   // 4096: single-dir element cap

    // Integer hash so a mis-indexed read across the two-level boundary shows up
    // as a wrong value, not merely an off-by-one slip.
    int32_t mix(int32_t i) {
        uint32_t x = static_cast<uint32_t>(i);
        x = (x ^ 61u) ^ (x >> 16);
        x *= 9u;
        x ^= x >> 4;
        x *= 0x27d4eb2du;
        x ^= x >> 15;
        return static_cast<int32_t>(x);
    }

    // Level-1 directory pages held by a resident TWO-LEVEL vector that is the
    // arena's only user: total live = data pages (pagesHeld) + L1 pages + L2 root.
    int32_t dirPagesOf(const gl::LbArena& a, const gl::PagedVector<int32_t>& v) {
        return a.livePages() - v.pagesHeld() - 1;
    }
}

TEST(paged_vector, two_level_directory_spans_several_pages) {
    gl::GlobalMemoryManager m;
    m.init(kPvSpillCfg);
    gl::LbArena a(&m);
    ASSERT_EQ(a.pageBytes(), 256);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::PagedVector<int32_t> v(&a, &d);
    const int32_t N = 20000;                       // ~313 data pages -> ~5 L1 pages
    for (int32_t i = 0; i < N; ++i) v.push_back(mix(i));
    ASSERT_EQ(v.size(), N);
    ASSERT_TRUE(v.pagesHeld() > kDirCap);          // past the single-directory cap
    ASSERT_TRUE(dirPagesOf(a, v) >= 3);            // the index spans several pages
    for (int32_t i = 0; i < N; ++i) ASSERT_EQ(v[i], mix(i));   // all readable
}

TEST(paged_vector, two_level_index_boundaries_exact) {
    gl::GlobalMemoryManager m;
    m.init(kPvSpillCfg);
    gl::LbArena a(&m);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::PagedVector<int32_t> v(&a, &d);
    for (int32_t i = 0; i < 12000; ++i) v.push_back(mix(i));
    // page boundaries (64), the single->two-level boundary (4096), and an
    // interior L1 boundary (8192 = 2*dirCap*epp).
    const int32_t idx[] = { 0, 1, 63, 64, 65, 4095, 4096, 4097,
                            8191, 8192, 8193, 11999 };
    for (int32_t i : idx) ASSERT_EQ(v[i], mix(i));
}

TEST(paged_vector, two_level_references_stable_across_promotion) {
    gl::GlobalMemoryManager m;
    m.init(kPvSpillCfg);
    gl::LbArena a(&m);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::PagedVector<int32_t> v(&a, &d);
    v.push_back(mix(0));
    const int32_t* p0 = &v[0];
    for (int32_t i = 1; i < 20000; ++i) v.push_back(mix(i));   // inline -> two-level
    ASSERT_EQ(p0, &v[0]);                          // pages never move
    ASSERT_EQ(*p0, mix(0));
}

TEST(paged_vector, two_level_erase_matches_oracle) {
    gl::GlobalMemoryManager m;
    m.init(kPvSpillCfg);
    gl::LbArena a(&m);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::PagedVector<int32_t> v(&a, &d);
    std::vector<int32_t> oracle;
    for (int32_t i = 0; i < 6000; ++i) { v.push_back(mix(i)); oracle.push_back(mix(i)); }
    uint32_t rng = 0x1234abcdu;
    auto nx = [&]() { rng = rng * 1664525u + 1013904223u; return rng; };
    for (int step = 0; step < 2500; ++step) {       // stays mostly two-level (> 4096)
        if (oracle.empty() || (nx() & 1u)) {
            const int32_t e = static_cast<int32_t>(nx());
            v.push_back(e); oracle.push_back(e);
        } else {
            const int32_t idx =
                static_cast<int32_t>(nx() % static_cast<uint32_t>(oracle.size()));
            v.erase(idx); oracle.erase(oracle.begin() + idx);
        }
    }
    ASSERT_EQ(v.size(), static_cast<int32_t>(oracle.size()));
    for (int32_t i = 0; i < v.size(); ++i) ASSERT_EQ(v[i], oracle[i]);
}

TEST(paged_vector, two_level_insert_at_matches_oracle) {
    gl::GlobalMemoryManager m;
    m.init(kPvSpillCfg);
    gl::LbArena a(&m);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::PagedVector<int32_t> v(&a, &d);
    std::vector<int32_t> oracle;
    for (int32_t i = 0; i < 6000; ++i) { v.push_back(mix(i)); oracle.push_back(mix(i)); }
    uint32_t rng = 0x55aa1234u;
    auto nx = [&]() { rng = rng * 1664525u + 1013904223u; return rng; };
    for (int step = 0; step < 1500; ++step) {
        const int32_t pos =
            static_cast<int32_t>(nx() % static_cast<uint32_t>(oracle.size() + 1));
        const int32_t e = static_cast<int32_t>(nx());
        v.insertAt(pos, e);
        oracle.insert(oracle.begin() + pos, e);
    }
    ASSERT_EQ(v.size(), static_cast<int32_t>(oracle.size()));
    for (int32_t i = 0; i < v.size(); ++i) ASSERT_EQ(v[i], oracle[i]);
}

TEST(paged_vector, two_level_replace_range_matches_oracle) {
    gl::GlobalMemoryManager m;
    m.init(kPvSpillCfg);
    gl::LbArena a(&m);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::PagedVector<int32_t> v(&a, &d);
    std::vector<int32_t> oracle;
    for (int32_t i = 0; i < 6000; ++i) { v.push_back(mix(i)); oracle.push_back(mix(i)); }
    uint32_t rng = 0x9e3779b9u;
    auto nx = [&]() { rng = rng * 1664525u + 1013904223u; return rng; };
    for (int step = 0; step < 1200; ++step) {
        const int32_t sz = static_cast<int32_t>(oracle.size());
        const int32_t pos = static_cast<int32_t>(nx() % static_cast<uint32_t>(sz + 1));
        const int32_t maxOld = sz - pos;
        int32_t oldLen = static_cast<int32_t>(nx() % 7u);
        if (oldLen > maxOld) oldLen = maxOld;
        const int32_t newLen = static_cast<int32_t>(nx() % 9u);
        std::vector<int32_t> src;
        for (int32_t k = 0; k < newLen; ++k) src.push_back(static_cast<int32_t>(nx()));
        v.replaceRange(pos, oldLen, src.empty() ? nullptr : src.data(), newLen);
        oracle.erase(oracle.begin() + pos, oracle.begin() + pos + oldLen);
        oracle.insert(oracle.begin() + pos, src.begin(), src.end());
    }
    ASSERT_EQ(v.size(), static_cast<int32_t>(oracle.size()));
    for (int32_t i = 0; i < v.size(); ++i) ASSERT_EQ(v[i], oracle[i]);
}

TEST(paged_vector, two_level_set_at_across_boundaries) {
    gl::GlobalMemoryManager m;
    m.init(kPvSpillCfg);
    gl::LbArena a(&m);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::PagedVector<int32_t> v(&a, &d);
    for (int32_t i = 0; i < 12000; ++i) v.push_back(mix(i));
    d = gl::DirtyState::Clean;
    v.setAt(0, mix(90000));
    v.setAt(4096, mix(90001));        // first index of the two-level region
    v.setAt(8192, mix(90002));        // an interior L1 boundary
    v.setAt(11999, mix(90003));       // last
    ASSERT_EQ(v[0], mix(90000));
    ASSERT_EQ(v[4096], mix(90001));
    ASSERT_EQ(v[8192], mix(90002));
    ASSERT_EQ(v[11999], mix(90003));
    ASSERT_EQ(v[1], mix(1));          // neighbours untouched
    ASSERT_EQ(v[4095], mix(4095));
    ASSERT_EQ(v[8191], mix(8191));
    ASSERT_EQ(static_cast<int>(d), static_cast<int>(gl::DirtyState::Restructured));
}

TEST(paged_vector, two_level_contiguous_run_tiles) {
    gl::GlobalMemoryManager m;
    m.init(kPvSpillCfg);
    gl::LbArena a(&m);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::PagedVector<int32_t> v(&a, &d);
    for (int32_t i = 0; i < 12000; ++i) v.push_back(mix(i));
    int32_t pos = 0, runs = 0;
    while (pos < v.size()) {
        int32_t run = 0;
        const int32_t* p = v.contiguousRun(pos, run);
        ASSERT_TRUE(run >= 1);
        for (int32_t k = 0; k < run; ++k) ASSERT_EQ(p[k], v[pos + k]);
        pos += run;
        ++runs;
    }
    ASSERT_EQ(pos, v.size());
    ASSERT_TRUE(runs >= 100);          // tiled across many pages
}

TEST(paged_vector, two_level_truncate_demotes) {
    gl::GlobalMemoryManager m;
    m.init(kPvSpillCfg);
    gl::LbArena a(&m);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::PagedVector<int32_t> v(&a, &d);
    for (int32_t i = 0; i < 20000; ++i) v.push_back(mix(i));
    ASSERT_TRUE(v.pagesHeld() > kDirCap);

    v.truncate(10000);                 // still two-level
    ASSERT_EQ(v.size(), 10000);
    ASSERT_TRUE(v.pagesHeld() > kDirCap);
    for (int32_t i = 0; i < 10000; ++i) ASSERT_EQ(v[i], mix(i));

    v.truncate(kSingleCap);            // exactly the single-directory cap (64 pages)
    ASSERT_EQ(v.size(), kSingleCap);
    ASSERT_EQ(v.pagesHeld(), kDirCap);
    ASSERT_EQ(a.livePages(), kDirCap + 1);   // 64 data + 1 directory page

    v.truncate(kEppI32);              // one page -> inline
    ASSERT_EQ(v.pagesHeld(), 1);
    ASSERT_EQ(a.livePages(), 1);

    v.truncate(0);
    ASSERT_EQ(v.size(), 0);
    ASSERT_EQ(v.pagesHeld(), 0);
    ASSERT_EQ(a.livePages(), 0);
}

TEST(paged_vector, two_level_tail_erase_demotes_to_single) {
    gl::GlobalMemoryManager m;
    m.init(kPvSpillCfg);
    gl::LbArena a(&m);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::PagedVector<int32_t> v(&a, &d);
    const int32_t N = (kDirCap + 2) * kEppI32;     // 66 pages -> 2 L1 pages
    for (int32_t i = 0; i < N; ++i) v.push_back(mix(i));
    ASSERT_EQ(v.pagesHeld(), kDirCap + 2);
    ASSERT_EQ(dirPagesOf(a, v), 2);
    while (v.size() > kSingleCap) v.erase(v.size() - 1);   // back across the boundary
    ASSERT_EQ(v.size(), kSingleCap);
    ASSERT_EQ(v.pagesHeld(), kDirCap);             // demoted to a single directory page
    ASSERT_EQ(a.livePages(), kDirCap + 1);
    for (int32_t i = 0; i < kSingleCap; ++i) ASSERT_EQ(v[i], mix(i));
}

TEST(paged_vector, two_level_clear_frees_all_pages) {
    gl::GlobalMemoryManager m;
    m.init(kPvSpillCfg);
    gl::LbArena a(&m);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::PagedVector<int32_t> v(&a, &d);
    for (int32_t i = 0; i < 20000; ++i) v.push_back(mix(i));
    ASSERT_TRUE(v.pagesHeld() > kDirCap);
    ASSERT_TRUE(a.livePages() > 0);
    v.clear();
    ASSERT_EQ(v.size(), 0);
    ASSERT_EQ(v.pagesHeld(), 0);
    ASSERT_EQ(a.livePages(), 0);                   // every page (data + L1 + root) returned
}

TEST(paged_vector, two_level_deload_bytes_match_arena_vector) {
    gl::GlobalMemoryManager m;
    m.init(kPvSpillCfg);
    gl::LbArena ap(&m), aa(&m);
    gl::DirtyState dp = gl::DirtyState::Clean, da = gl::DirtyState::Clean;
    gl::PagedVector<int32_t> pv(&ap, &dp);
    gl::ArenaVector<int32_t> av(&aa, &da);
    const int32_t N = 10000;
    for (int32_t i = 0; i < N; ++i) { pv.push_back(mix(i)); av.push_back(mix(i)); }
    ASSERT_TRUE(pv.pagesHeld() > kDirCap);         // pv is two-level

    // Definitive: the dense stream equals the raw element bytes, regardless of
    // the (two-level) page layout.
    std::vector<char> expect;
    for (int32_t i = 0; i < N; ++i) {
        const int32_t e = mix(i);
        const char* p = reinterpret_cast<const char*>(&e);
        expect.insert(expect.end(), p, p + sizeof(int32_t));
    }
    std::vector<char> bp; pv.appendSpanBytes(bp, 0);
    ASSERT_TRUE(bp == expect);

    // Cross-type: byte-identical to ArenaVector for the same content.
    std::vector<char> ba; av.appendSpanBytes(ba, 0);
    ASSERT_TRUE(bp == ba);
}

TEST(paged_vector, two_level_bulk_append_round_trip) {
    gl::GlobalMemoryManager m;
    m.init(kPvSpillCfg);
    gl::LbArena a1(&m), a2(&m);
    gl::DirtyState d1 = gl::DirtyState::Clean, d2 = gl::DirtyState::Clean;
    gl::PagedVector<int32_t> v(&a1, &d1);
    const int32_t N = 10000;
    for (int32_t i = 0; i < N; ++i) v.push_back(mix(i));
    std::vector<char> bytes; v.appendSpanBytes(bytes, 0);
    gl::PagedVector<int32_t> w(&a2, &d2);
    w.bulkAppendBytes(bytes.data(), N);
    ASSERT_EQ(w.size(), N);
    ASSERT_TRUE(w.pagesHeld() > kDirCap);          // reload rebuilt a two-level directory
    for (int32_t i = 0; i < N; ++i) ASSERT_EQ(w[i], v[i]);
}

TEST(paged_vector, two_level_copy_assign_across_arenas) {
    gl::GlobalMemoryManager m;
    m.init(kPvSpillCfg);
    gl::LbArena a1(&m), a2(&m);
    gl::DirtyState d1 = gl::DirtyState::Clean, d2 = gl::DirtyState::Clean;
    gl::PagedVector<int32_t> src(&a1, &d1), dst(&a2, &d2);
    const int32_t N = 10000;
    for (int32_t i = 0; i < N; ++i) src.push_back(mix(i));
    dst.push_back(123);                            // pre-existing, discarded
    dst = src;
    ASSERT_EQ(dst.size(), src.size());
    ASSERT_TRUE(dst.pagesHeld() > kDirCap);
    for (int32_t i = 0; i < N; ++i) ASSERT_EQ(dst[i], src[i]);
    ASSERT_NE(&dst[0], &src[0]);                   // distinct storage
}

TEST(paged_vector, two_level_append_run_crosses_in) {
    gl::GlobalMemoryManager m;
    m.init(kPvSpillCfg);
    gl::LbArena a(&m);
    gl::DirtyState d = gl::DirtyState::Clean;
    std::vector<int32_t> src;
    for (int32_t i = 0; i < 10000; ++i) src.push_back(mix(i));
    gl::PagedVector<int32_t> v(&a, &d);
    v.appendRun(src.data(), static_cast<int32_t>(src.size()));   // single -> two-level
    ASSERT_EQ(v.size(), 10000);
    ASSERT_TRUE(v.pagesHeld() > kDirCap);
    for (int32_t i = 0; i < v.size(); ++i) ASSERT_EQ(v[i], src[i]);
    ASSERT_EQ(static_cast<int>(d), static_cast<int>(gl::DirtyState::AppendedOnly));
}

TEST(paged_vector, two_level_append_run_no_straddle_char) {
    gl::GlobalMemoryManager m;
    m.init(kPvSpillCfg);
    gl::LbArena a(&m);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::PagedVector<char> v(&a, &d);   // char: dirCap = 256/4 = 64, epp = 256
    const int32_t runLen = 200;        // two 200-byte runs can't share a 256-byte page
    std::vector<int32_t> starts;
    for (int k = 0; k < 80; ++k) {     // 80 padded runs -> 80 pages > dirCap 64
        std::vector<char> run(static_cast<size_t>(runLen),
                              static_cast<char>('a' + (k % 26)));
        starts.push_back(v.appendRunNoStraddle(run.data(), runLen));
    }
    ASSERT_TRUE(v.pagesHeld() > kDirCap);          // the byte vector went two-level
    for (int k = 0; k < 80; ++k) {                 // each run reads back contiguous
        int32_t run = 0;
        const char* p = v.contiguousRun(starts[k], run);
        ASSERT_TRUE(run >= runLen);
        ASSERT_EQ(p[0], static_cast<char>('a' + (k % 26)));
        ASSERT_EQ(p[runLen - 1], static_cast<char>('a' + (k % 26)));
    }
}

TEST(paged_vector, two_level_promotion_boundary_page_accounting) {
    gl::GlobalMemoryManager m;
    m.init(kPvSpillCfg);
    gl::LbArena a(&m);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::PagedVector<int32_t> v(&a, &d);
    for (int32_t i = 0; i < kSingleCap; ++i) v.push_back(mix(i));   // 4096 -> 64 pages
    ASSERT_EQ(v.pagesHeld(), kDirCap);
    ASSERT_EQ(a.livePages(), kDirCap + 1);         // 64 data + 1 directory page

    v.push_back(mix(kSingleCap));                  // the 4097th -> promote to two-level
    ASSERT_EQ(v.pagesHeld(), kDirCap + 1);
    ASSERT_EQ(a.livePages(), kDirCap + 4);         // 65 data + 2 L1 + 1 root = 68

    v.erase(kSingleCap);                           // back to 4096 -> demote to single
    ASSERT_EQ(v.pagesHeld(), kDirCap);
    ASSERT_EQ(a.livePages(), kDirCap + 1);
    for (int32_t i = 0; i < kSingleCap; ++i) ASSERT_EQ(v[i], mix(i));
}

TEST(paged_vector, two_level_promotion_records_telemetry) {
    gl::GlobalMemoryManager m;
    m.init(kPvSpillCfg);
    gl::LbArena a(&m);
    gl::DirtyState d = gl::DirtyState::Clean;
    ASSERT_EQ(m.twoLevelPromotions(), 0);          // fresh manager, nothing promoted
    gl::PagedVector<int32_t> v(&a, &d);
    for (int32_t i = 0; i < kSingleCap; ++i) v.push_back(mix(i));   // single dir, no promotion
    ASSERT_EQ(m.twoLevelPromotions(), 0);
    v.push_back(mix(kSingleCap));                  // crosses dirCap -> records the promotion
    ASSERT_TRUE(m.twoLevelPromotions() >= 1);
    ASSERT_TRUE(m.peakPagesHeld() >= kDirCap + 1);
}
