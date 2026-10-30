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
/// @brief Unit tests for the per-LB bump arena (`LbArena`, cold-path
///        statification core).
///
/// @details
/// Covers the bump model: lazy block acquisition and consecutive
/// allocation, alignment of returned offsets, the no-straddle block-tail
/// padding, offset-to-physical resolution across a block boundary,
/// last-in-first-out `popTo` with poisoning and block retention, `releaseAll`
/// + residency lifecycle, destructor block return, and the load-bearing
/// determinism property — offsets are a pure function of allocation order,
/// independent of which physical blocks back them.
///
/// Also covers the page tier that backs the paged containers: vid allocation
/// with free-vid LIFO reuse (D-228 — the table is bounded by
/// the live-page high water), block-granular page carving, physical free-list
/// reuse, and page-block return on `releaseAll`.

#include "test_harness.hpp"

#include "../memory_infra/lb_arena.hpp"
#include "../parameters.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {
    // 1 MiB pool / 256 KiB block — 4 blocks, as in the static-memory and
    // hot-arena suites.
    const gl::StaticMemoryConfig kArenaTestCfg{ 1 << 20, 1 << 18 };
}

TEST(lb_arena, lazy_acquisition_and_consecutive_bump) {
    gl::GlobalMemoryManager m;
    m.init(kArenaTestCfg);
    gl::LbArena a(&m);
    // Idle arena holds nothing.
    ASSERT_EQ(a.blocksHeld(), static_cast<int64_t>(0));
    ASSERT_EQ(m.blocksInUse(), static_cast<int64_t>(0));
    ASSERT_EQ(a.cursor(), static_cast<gl::ArenaOffset>(0));
    const gl::ArenaOffset o0 = a.alloc(16);
    const gl::ArenaOffset o1 = a.alloc(7);
    const gl::ArenaOffset o2 = a.alloc(1);
    ASSERT_EQ(o0, static_cast<gl::ArenaOffset>(0));
    ASSERT_EQ(o1, static_cast<gl::ArenaOffset>(16));
    ASSERT_EQ(o2, static_cast<gl::ArenaOffset>(23));
    ASSERT_EQ(a.blocksHeld(), static_cast<int64_t>(1));
    ASSERT_EQ(m.blocksInUse(), static_cast<int64_t>(1));
    ASSERT_EQ(a.usedBytes(), static_cast<int64_t>(24));
    // Resolution gives consecutive, writable, non-overlapping storage.
    char* p0 = a.resolve(o0);
    char* p1 = a.resolve(o1);
    ASSERT_EQ(p1, p0 + 16);
    std::memset(p0, 0xA5, 16);
    ASSERT_EQ(p0[15], static_cast<char>(0xA5));
}

TEST(lb_arena, alloc_returns_aligned_offsets) {
    gl::GlobalMemoryManager m;
    m.init(kArenaTestCfg);
    gl::LbArena a(&m);
    const gl::ArenaOffset o0 = a.alloc(1);        // off 0, cursor 1
    const gl::ArenaOffset o8 = a.alloc(8, 8);     // align 8: cursor 1 -> 8
    ASSERT_EQ(o0, static_cast<gl::ArenaOffset>(0));
    ASSERT_EQ(o8, static_cast<gl::ArenaOffset>(8));
    ASSERT_EQ(o8 % 8u, static_cast<gl::ArenaOffset>(0));
    // The resolved physical address inherits the alignment.
    const char* p = a.resolve(o8);
    ASSERT_EQ(reinterpret_cast<uintptr_t>(p) & static_cast<uintptr_t>(7),
              static_cast<uintptr_t>(0));
    const gl::ArenaOffset o16 = a.alloc(4, 16);   // cursor 16 already aligned
    ASSERT_EQ(o16, static_cast<gl::ArenaOffset>(16));
}

// ---- Page tier ----------------------------------------------------------

TEST(lb_arena, page_alloc_monotonic_distinct_and_writable) {
    gl::GlobalMemoryManager m;
    m.init(kArenaTestCfg);
    gl::LbArena a(&m);
    ASSERT_EQ(a.livePages(), 0);
    ASSERT_EQ(a.pageHighWater(), 0);
    const int32_t v0 = a.allocPage();
    const int32_t v1 = a.allocPage();
    const int32_t v2 = a.allocPage();
    ASSERT_EQ(v0, 0);
    ASSERT_EQ(v1, 1);
    ASSERT_EQ(v2, 2);
    ASSERT_EQ(a.livePages(), 3);
    ASSERT_EQ(a.pageHighWater(), 3);
    // One 256 KiB block carved into 32 pages backs all three; the small page
    // table (3 vids) stays inline in the PtrDirectory, costing no pool block.
    ASSERT_EQ(m.blocksInUse(), static_cast<int64_t>(1));
    // Pages are distinct and each is a full page of writable storage.
    char* p0 = a.pageAt(v0);
    char* p1 = a.pageAt(v1);
    ASSERT_NE(p0, p1);
    std::memset(p0, 0xA5, static_cast<size_t>(m.pageBytes()));
    std::memset(p1, 0x5A, static_cast<size_t>(m.pageBytes()));
    ASSERT_EQ(p0[m.pageBytes() - 1], static_cast<char>(0xA5));
    ASSERT_EQ(p1[m.pageBytes() - 1], static_cast<char>(0x5A));
}

TEST(lb_arena, page_grows_by_one_block_per_block_of_pages) {
    gl::GlobalMemoryManager m;
    m.init(kArenaTestCfg);
    gl::LbArena a(&m);
    const int32_t ppb = m.blockBytes() / m.pageBytes();   // 32
    for (int32_t i = 0; i < ppb; ++i) a.allocPage();
    ASSERT_EQ(m.blocksInUse(), static_cast<int64_t>(1));   // one carved block; table inline
    a.allocPage();                                         // the (ppb+1)-th page
    ASSERT_EQ(m.blocksInUse(), static_cast<int64_t>(2));   // forces a 2nd carved block
    ASSERT_EQ(a.livePages(), ppb + 1);
}

TEST(lb_arena, page_tail_free_vid_recycles) {
    gl::GlobalMemoryManager m;
    m.init(kArenaTestCfg);
    gl::LbArena a(&m);
    a.allocPage();                       // v0
    a.allocPage();                       // v1
    const int32_t v2 = a.allocPage();    // v2 (tail)
    ASSERT_EQ(a.pageHighWater(), 3);
    a.freePage(v2);                      // tail free: vid parks on the free list
    ASSERT_EQ(a.pageHighWater(), 3);     // live high water — never shrinks
    ASSERT_EQ(a.livePages(), 2);
    // The next alloc reuses the freed vid.
    const int32_t v2b = a.allocPage();
    ASSERT_EQ(v2b, 2);
    ASSERT_EQ(a.pageHighWater(), 3);
    a.assertInvariants();
}

TEST(lb_arena, page_interior_free_vid_recycles) {
    gl::GlobalMemoryManager m;
    m.init(kArenaTestCfg);
    gl::LbArena a(&m);
    const int32_t v0 = a.allocPage();
    const int32_t v1 = a.allocPage();
    a.allocPage();                       // v2 stays live -> v1 is interior
    char* freed = a.pageAt(v1);
    const int32_t freeBefore = a.freePageCount();
    a.freePage(v1);                      // interior free: vid parks on the free list
    ASSERT_EQ(a.pageHighWater(), 3);     // table size unchanged
    ASSERT_EQ(a.livePages(), 2);
    ASSERT_EQ(a.freePageCount(), freeBefore + 1);
    // The freed vid AND its physical page both recycle (LIFO).
    const int32_t v1b = a.allocPage();
    ASSERT_EQ(v1b, v1);
    ASSERT_EQ(a.pageAt(v1b), freed);
    ASSERT_EQ(a.pageHighWater(), 3);     // no fresh tail vid minted
    a.assertInvariants();
    (void)v0;
}

// The vid-ratchet pin (D-228): a churn loop with a live pin
// above the churned vids must NOT grow the page table — freed vids recycle,
// so the table stays at the live-page high water instead of the cumulative
// allocation count (the rung-2 524K-vid PtrDirectory overflow shape).
TEST(lb_arena, page_churn_with_pin_does_not_ratchet_vid_space) {
    gl::GlobalMemoryManager m;
    m.init(kArenaTestCfg);
    gl::LbArena a(&m);
    a.allocPage();                                 // v0: long-lived pin
    const int32_t churn = a.allocPage();           // v1
    a.allocPage();                                 // v2: a second pin ABOVE
    a.freePage(churn);                             // interior hole below a pin
    const int32_t peak = a.pageHighWater();        // 3
    for (int32_t i = 0; i < 10000; ++i) {
        const int32_t v = a.allocPage();           // reuses the hole every time
        ASSERT_EQ(v, churn);
        a.freePage(v);
    }
    ASSERT_EQ(a.pageHighWater(), peak);            // NOT 3 + 10000
    ASSERT_EQ(a.livePages(), 2);
    a.assertInvariants();
}

// Deterministic LIFO reuse: two arenas driven through the same allocation/free
// sequence hand out identical vids at every step (I-107's allocation/free-
// sequence purity), and the LIFO pops the most recently freed vid first.
TEST(lb_arena, page_free_vid_reuse_is_lifo_and_deterministic) {
    gl::GlobalMemoryManager m;
    m.init(kArenaTestCfg);
    gl::LbArena a(&m);
    gl::LbArena b(&m);
    const auto drive = [](gl::LbArena& x) {
        std::vector<int32_t> got;
        for (int32_t i = 0; i < 6; ++i) got.push_back(x.allocPage()); // 0..5
        x.freePage(1);
        x.freePage(4);
        x.freePage(2);
        got.push_back(x.allocPage());   // LIFO head: 2
        got.push_back(x.allocPage());   // then 4
        got.push_back(x.allocPage());   // then 1
        got.push_back(x.allocPage());   // list dry: fresh tail vid 6
        return got;
    };
    const std::vector<int32_t> ga = drive(a);
    const std::vector<int32_t> gb = drive(b);
    ASSERT_TRUE(ga == gb);
    ASSERT_EQ(ga[6], 2);
    ASSERT_EQ(ga[7], 4);
    ASSERT_EQ(ga[8], 1);
    ASSERT_EQ(ga[9], 6);
    ASSERT_EQ(a.pageHighWater(), 7);
    a.assertInvariants();
    b.assertInvariants();
}

TEST(lb_arena, page_release_all_returns_blocks_and_resets) {
    gl::GlobalMemoryManager m;
    m.init(kArenaTestCfg);
    gl::LbArena a(&m);
    for (int i = 0; i < 40; ++i) a.allocPage();            // > 32 -> 2 blocks
    ASSERT_EQ(m.blocksInUse(), static_cast<int64_t>(2));   // 2 carved blocks; table inline
    a.releaseAll();
    ASSERT_EQ(a.livePages(), 0);
    ASSERT_EQ(a.pageHighWater(), 0);
    ASSERT_EQ(a.freePageCount(), 0);
    ASSERT_EQ(m.blocksInUse(), static_cast<int64_t>(0));   // returned to pool
}

TEST(lb_arena, page_compact_packs_live_and_frees_blocks) {
    gl::GlobalMemoryManager m;
    m.init(kArenaTestCfg);
    gl::LbArena a(&m);
    // Compaction scratch on an INDEPENDENT pool (production draws it from the
    // never-deloaded LB-body pool, not the deloadable pool being compacted).
    gl::GlobalMemoryManager mScr;
    mScr.init(kArenaTestCfg);
    gl::LbArena scratch(&mScr);
    const int32_t ppb = m.blockBytes() / m.pageBytes();   // 32 pages / block
    // Fill three blocks' worth of pages, tagging each with its vid so we can
    // verify content survives the relocation.
    std::vector<int32_t> vids;
    for (int32_t i = 0; i < ppb * 3; ++i) {
        const int32_t v = a.allocPage();
        vids.push_back(v);
        std::memcpy(a.pageAt(v), &v, sizeof(int32_t));
    }
    ASSERT_EQ(m.blocksInUse(), static_cast<int64_t>(3));   // 3 carved blocks; table inline
    // Free two of every three (scattered interior holes across all blocks),
    // leaving exactly one block's worth (ppb) of live pages.
    std::vector<int32_t> liveVids;
    for (std::size_t i = 0; i < vids.size(); ++i) {
        if (i % 3 == 0) liveVids.push_back(vids[i]);
        else a.freePage(vids[i]);
    }
    const int32_t liveCount = static_cast<int32_t>(liveVids.size());  // 32
    ASSERT_EQ(a.livePages(), liveCount);
    const int64_t before = m.blocksInUse();               // still 3 (retained)
    // Pack: the live pages collapse onto one block; the other two return.
    const int64_t reclaimed = a.compactPages(scratch);
    ASSERT_TRUE(reclaimed > 0);
    ASSERT_EQ(m.blocksInUse(), before - reclaimed);
    ASSERT_EQ(a.livePages(), liveCount);                  // unchanged
    // Content-invisible: every live vid still resolves to its tagged content.
    for (int32_t v : liveVids) {
        int32_t got = 0;
        std::memcpy(&got, a.pageAt(v), sizeof(int32_t));
        ASSERT_EQ(got, v);
    }
    // Already dense: a second pass moves nothing and reclaims nothing.
    ASSERT_EQ(a.compactPages(scratch), static_cast<int64_t>(0));
}

TEST(lb_arena, no_straddle_pads_to_next_block) {
    gl::GlobalMemoryManager m;
    m.init(kArenaTestCfg);
    gl::LbArena a(&m);
    const int32_t bb = m.blockBytes();
    const gl::ArenaOffset o0 = a.alloc(bb - 32);  // fills all but 32 bytes
    ASSERT_EQ(o0, static_cast<gl::ArenaOffset>(0));
    ASSERT_EQ(a.blocksHeld(), static_cast<int64_t>(1));
    // 64 will not fit the 32-byte remainder: the cursor pads to the block
    // boundary and the allocation lands at the start of block 1.
    const gl::ArenaOffset o1 = a.alloc(64);
    ASSERT_EQ(o1, static_cast<gl::ArenaOffset>(bb));
    ASSERT_EQ(a.blocksHeld(), static_cast<int64_t>(2));
    ASSERT_EQ(a.usedBytes(), static_cast<int64_t>(bb) + 64);
    // Both resolve, in different blocks.
    ASSERT_NE(a.resolve(o0), a.resolve(o1));
}

TEST(lb_arena, resolve_round_trips_across_blocks) {
    gl::GlobalMemoryManager m;
    m.init(kArenaTestCfg);
    gl::LbArena a(&m);
    const int32_t bb = m.blockBytes();
    const gl::ArenaOffset oA = a.alloc(bb);   // block 0, exactly full
    const gl::ArenaOffset oB = a.alloc(100);  // block 1
    std::memset(a.resolve(oA), 0x11, static_cast<size_t>(bb));
    std::memset(a.resolve(oB), 0x22, 100);
    // Re-resolution is stable while resident; content reads back intact.
    ASSERT_EQ(a.resolve(oA)[0], static_cast<char>(0x11));
    ASSERT_EQ(a.resolve(oA)[bb - 1], static_cast<char>(0x11));
    ASSERT_EQ(a.resolve(oB)[0], static_cast<char>(0x22));
    ASSERT_EQ(a.resolve(oB)[99], static_cast<char>(0x22));
    ASSERT_EQ(a.blocksHeld(), static_cast<int64_t>(2));
}

TEST(lb_arena, pop_to_rewinds_poisons_and_reuses) {
    gl::GlobalMemoryManager m;
    m.init(kArenaTestCfg);
    gl::LbArena a(&m);
    const gl::ArenaOffset keep = a.alloc(8);
    char* pk = a.resolve(keep);
    std::memset(pk, 0x33, 8);
    const gl::ArenaOffset mark = a.cursor();   // == 8
    const gl::ArenaOffset scratch = a.alloc(32);
    char* ps = a.resolve(scratch);
    std::memset(ps, 0x44, 32);
    ASSERT_EQ(a.usedBytes(), static_cast<int64_t>(40));
    a.popTo(mark);
    ASSERT_EQ(a.usedBytes(), static_cast<int64_t>(8));
    // The reclaimed span is poisoned; the kept prefix is untouched.
    ASSERT_EQ(ps[0], static_cast<char>(gl::kArenaPoisonByte));
    ASSERT_EQ(ps[31], static_cast<char>(gl::kArenaPoisonByte));
    ASSERT_EQ(pk[7], static_cast<char>(0x33));
    // The next allocation reuses the popped space at the same offset and
    // physical address — the block stayed retained.
    const gl::ArenaOffset again = a.alloc(32);
    ASSERT_EQ(again, scratch);
    ASSERT_EQ(a.resolve(again), ps);
    ASSERT_EQ(a.blocksHeld(), static_cast<int64_t>(1));
}

TEST(lb_arena, release_all_returns_blocks_and_stays_usable) {
    gl::GlobalMemoryManager m;
    m.init(kArenaTestCfg);
    gl::LbArena a(&m);
    a.alloc(32);
    ASSERT_EQ(a.blocksHeld(), static_cast<int64_t>(1));
    ASSERT_EQ(m.blocksInUse(), static_cast<int64_t>(1));
    a.releaseAll();
    ASSERT_EQ(a.blocksHeld(), static_cast<int64_t>(0));
    ASSERT_EQ(m.blocksInUse(), static_cast<int64_t>(0));
    ASSERT_EQ(a.cursor(), static_cast<gl::ArenaOffset>(0));
    // Reusable: the cursor restarts at 0 and a block re-acquires lazily.
    const gl::ArenaOffset o = a.alloc(16);
    ASSERT_EQ(o, static_cast<gl::ArenaOffset>(0));
    ASSERT_EQ(a.blocksHeld(), static_cast<int64_t>(1));
    a.releaseAll();
    ASSERT_EQ(m.blocksInUse(), static_cast<int64_t>(0));
}

TEST(lb_arena, residency_flags) {
    gl::GlobalMemoryManager m;
    m.init(kArenaTestCfg);
    gl::LbArena a(&m);
    ASSERT_TRUE(a.resident());
    a.alloc(16);
    a.releaseAll();
    a.markDeloaded();
    ASSERT_FALSE(a.resident());
    a.markResident();
    ASSERT_TRUE(a.resident());
}

TEST(lb_arena, destructor_returns_blocks) {
    gl::GlobalMemoryManager m;
    m.init(kArenaTestCfg);
    {
        gl::LbArena a(&m);
        a.alloc(16);
        ASSERT_EQ(m.blocksInUse(), static_cast<int64_t>(1));
    }
    ASSERT_EQ(m.blocksInUse(), static_cast<int64_t>(0));
}

TEST(lb_arena, offsets_are_position_independent) {
    gl::GlobalMemoryManager m;
    m.init(kArenaTestCfg);
    const int32_t bb = m.blockBytes();
    gl::ArenaOffset s0, s1, s2;
    {
        gl::LbArena a(&m);
        s0 = a.alloc(100);
        s1 = a.alloc(8, 8);
        s2 = a.alloc(bb - 50);   // forces a second block
    }                            // destructor returns both blocks
    // Perturb physical recycling so a re-run is backed by different blocks.
    char* x = m.acquireBlock();
    char* y = m.acquireBlock();
    m.releaseBlock(x);
    // The identical allocation sequence yields identical offsets — they are
    // a pure function of allocation order, not of physical block identity.
    gl::LbArena b(&m);
    ASSERT_EQ(b.alloc(100), s0);
    ASSERT_EQ(b.alloc(8, 8), s1);
    ASSERT_EQ(b.alloc(bb - 50), s2);
    m.releaseBlock(y);
}

// The page-tier scratch surface (allocBytes / mark / rewind / reset on a
// cold-bound arena) is covered by test_scratch_arena.cpp; HOT mode is gone.

// ---- Sub-page tier (D-464) ---------------------------

namespace {
    // 4 MiB pool / 256 KiB block / 2 KiB page: the production page size, five
    // sub-page classes (64 .. 1,024 bytes).
    const gl::StaticMemoryConfig kSubPageCfg{ 4 << 20, 1 << 18, 2048 };
}

TEST(lb_arena, sub_page_classes_follow_the_page_size) {
    const int32_t pages[] = { 64, 256, 2048, 8192 };
    const int32_t classes[] = { 0, 2, 5, 7 };
    for (int32_t i = 0; i < 4; ++i) {
        gl::GlobalMemoryManager m;
        m.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 14, pages[i] });
        gl::LbArena a(&m);
        ASSERT_FALSE(a.subPagesEnabled());
        a.enableSubPages();
        ASSERT_TRUE(a.subPagesEnabled());
        ASSERT_EQ(a.subPageClassCount(), classes[i]);
    }
    ASSERT_EQ(gl::LbArena::subPageBytes(0), 64);
    ASSERT_EQ(gl::LbArena::subPageBytes(4), 1024);
}

TEST(lb_arena, sub_page_slots_carve_reuse_poison_and_audit) {
    gl::GlobalMemoryManager m;
    m.init(kSubPageCfg);
    gl::LbArena a(&m);
    a.enableSubPages();
    // The first slot of a class carves one page; the next slots ascend in it.
    std::vector<gl::LbArena::SubPage> slots;
    for (int32_t k = 0; k < 32; ++k) slots.push_back(a.allocSubPage(0));
    ASSERT_EQ(a.subPageSlabs(0), 1);
    ASSERT_EQ(a.liveSubPages(0), 32);
    ASSERT_EQ(a.livePages(), 1);
    for (int32_t k = 0; k < 32; ++k) {
        ASSERT_EQ(slots[k].vid, slots[0].vid);
        ASSERT_EQ(slots[k].offset, k * 64);
        std::memset(a.pageAt(slots[k].vid) + slots[k].offset, k, 64);
    }
    // A full class carves the next page.
    const gl::LbArena::SubPage next = a.allocSubPage(0);
    ASSERT_NE(next.vid, slots[0].vid);
    ASSERT_EQ(next.offset, 0);
    ASSERT_EQ(a.subPageSlabs(0), 2);
    // Another class never shares a page with class 0.
    const gl::LbArena::SubPage large = a.allocSubPage(4);
    ASSERT_NE(large.vid, slots[0].vid);
    ASSERT_NE(large.vid, next.vid);
    ASSERT_EQ(a.subPageSlabs(4), 1);
    // A freed slot is poisoned up to its link and comes back last in, first
    // out; its neighbours keep their bytes.
    a.freeSubPage(slots[5], 0);
    a.freeSubPage(slots[9], 0);
    const char* freed = a.pageAt(slots[5].vid) + slots[5].offset;
    for (int32_t b = 0; b < 56; ++b)
        ASSERT_EQ(static_cast<unsigned char>(freed[b]), gl::kArenaPoisonByte);
    ASSERT_EQ(static_cast<int32_t>(a.pageAt(slots[6].vid)[slots[6].offset]), 6);
    a.assertInvariants();
    const gl::LbArena::SubPage again1 = a.allocSubPage(0);
    const gl::LbArena::SubPage again2 = a.allocSubPage(0);
    ASSERT_EQ(again1.offset, slots[9].offset);
    ASSERT_EQ(again2.offset, slots[5].offset);
    ASSERT_EQ(a.liveSubPages(0), 33);
    a.assertInvariants();
    // releaseAll returns the carved pages and empties every list.
    a.releaseAll();
    ASSERT_EQ(a.liveSubPages(0), 0);
    ASSERT_EQ(a.subPageSlabs(0), 0);
    ASSERT_EQ(a.subPageSlabs(4), 0);
    ASSERT_EQ(m.blocksInUse(), static_cast<int64_t>(0));
    const gl::LbArena::SubPage fresh = a.allocSubPage(0);
    ASSERT_EQ(fresh.offset, 0);
    ASSERT_EQ(a.subPageSlabs(0), 1);
}

TEST(lb_arena, sub_page_lists_survive_compaction) {
    gl::GlobalMemoryManager m;
    m.init(kSubPageCfg);
    gl::GlobalMemoryManager mScr;
    mScr.init(kSubPageCfg);
    gl::LbArena a(&m);
    gl::LbArena scratch(&mScr);
    a.enableSubPages();
    const int32_t ppb = m.blockBytes() / m.pageBytes();   // 128 pages per block
    std::vector<int32_t> pages;
    for (int32_t i = 0; i < ppb * 2; ++i) pages.push_back(a.allocPage());
    std::vector<gl::LbArena::SubPage> slots;
    for (int32_t k = 0; k < 20; ++k) {                     // 16 slots per page
        slots.push_back(a.allocSubPage(1));
        std::memset(a.pageAt(slots.back().vid) + slots.back().offset, 'a' + k, 128);
    }
    for (int32_t i = 0; i < ppb * 2; i += 2) a.freePage(pages[i]);
    a.freeSubPage(slots[3], 1);
    const int64_t reclaimed = a.compactPages(scratch);
    ASSERT_TRUE(reclaimed > 0);
    for (int32_t k = 0; k < 20; ++k) {
        if (k == 3) continue;
        const char* slot = a.pageAt(slots[k].vid) + slots[k].offset;
        ASSERT_EQ(slot[0], static_cast<char>('a' + k));
        ASSERT_EQ(slot[127], static_cast<char>('a' + k));
    }
    const gl::LbArena::SubPage reused = a.allocSubPage(1);
    ASSERT_EQ(reused.vid, slots[3].vid);
    ASSERT_EQ(reused.offset, slots[3].offset);
    a.assertInvariants();
}

TEST(lb_arena, sub_page_pages_without_live_slots_return_at_compaction) {
    gl::GlobalMemoryManager m;
    m.init(kSubPageCfg);
    gl::GlobalMemoryManager mScr;
    mScr.init(kSubPageCfg);
    gl::LbArena a(&m);
    gl::LbArena scratch(&mScr);
    a.enableSubPages();
    // Class 3 (512 bytes): four slots per page; three pages.
    std::vector<gl::LbArena::SubPage> slots;
    for (int32_t k = 0; k < 12; ++k) slots.push_back(a.allocSubPage(3));
    ASSERT_EQ(a.subPageSlabs(3), 3);
    // Empty the middle page completely and one slot of the last page.
    for (int32_t k = 4; k < 8; ++k) a.freeSubPage(slots[k], 3);
    a.freeSubPage(slots[9], 3);
    std::memset(a.pageAt(slots[0].vid) + slots[0].offset, 'q', 512);
    a.compactPages(scratch);
    ASSERT_EQ(a.subPageSlabs(3), 2);
    ASSERT_EQ(a.livePages(), 2);
    ASSERT_EQ(a.liveSubPages(3), 7);
    ASSERT_EQ(a.pageAt(slots[0].vid)[slots[0].offset + 511], 'q');
    a.assertInvariants();
    // The remaining free slot is the last page's; the next one carves a page.
    const gl::LbArena::SubPage reused = a.allocSubPage(3);
    ASSERT_EQ(reused.vid, slots[9].vid);
    ASSERT_EQ(reused.offset, slots[9].offset);
    const gl::LbArena::SubPage carved = a.allocSubPage(3);
    ASSERT_EQ(carved.offset, 0);
    ASSERT_EQ(a.subPageSlabs(3), 3);
    // Once every slot is free, compaction returns every block.
    a.freeSubPage(reused, 3);
    a.freeSubPage(carved, 3);
    for (int32_t k = 0; k < 12; ++k)
        if (k < 4 || k == 8 || k == 10 || k == 11) a.freeSubPage(slots[k], 3);
    ASSERT_EQ(a.liveSubPages(3), 0);
    a.compactPages(scratch);
    ASSERT_EQ(a.subPageSlabs(3), 0);
    ASSERT_EQ(a.livePages(), 0);
    ASSERT_EQ(a.blocksHeld(), static_cast<int64_t>(0));
    a.assertInvariants();
}
