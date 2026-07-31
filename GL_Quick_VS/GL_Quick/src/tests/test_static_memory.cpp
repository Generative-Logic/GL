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
/// @brief Unit tests for the statification static-memory machinery —
///        `GlobalMemoryManager`, the `LbMemory` aggregate, and the reshuffle
///        gate.
///
/// @details
/// Covers the config-sizing predicates `gl::isValidStaticMemoryConfig` (pool a
/// whole multiple of a power-of-two block) and `gl::isValidStaticPageConfig` (a
/// power-of-two page dividing the block) — positive and rejecting branches,
/// the predicates being pure precisely so the rejecting branches are testable
/// without aborting the harness); the global block pool (grant / release /
/// recycle, the grant ledger and its one-shot trigger); and the per-LB
/// aggregate's copying compaction (`LbMemory::reshuffle`) — content
/// preservation and hole reclamation. The arena core and arena vector have
/// their own suites (`test_lb_arena`, `test_arena_vector`).

#include "test_harness.hpp"

#include "../memory_infra/lb_memory.hpp"
#include "../memory_infra/ptr_directory.hpp"
#include "../memory_infra/steward.hpp"
#include "../parameters.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

TEST(static_memory, config_defaults_are_valid) {
    const gl::ProverParameters p;
    ASSERT_TRUE(gl::isValidStaticMemoryConfig(
        p.static_pool_bytes, p.static_block_bytes));
}

TEST(static_memory, config_accepts_pool_multiple_of_pow2_block) {
    ASSERT_TRUE(gl::isValidStaticMemoryConfig(1024, 256));
    // Degenerate but legal: one block per pool.
    ASSERT_TRUE(gl::isValidStaticMemoryConfig(256, 256));
}

TEST(static_memory, config_rejects_nonpositive_sizes) {
    ASSERT_FALSE(gl::isValidStaticMemoryConfig(0, 256));
    ASSERT_FALSE(gl::isValidStaticMemoryConfig(1024, 0));
    ASSERT_FALSE(gl::isValidStaticMemoryConfig(-1024, 256));
    ASSERT_FALSE(gl::isValidStaticMemoryConfig(1024, -256));
}

TEST(static_memory, config_rejects_pool_not_multiple_of_block) {
    ASSERT_FALSE(gl::isValidStaticMemoryConfig(1000, 256));
}

TEST(static_memory, config_rejects_non_power_of_two_block) {
    // 192 divides 768 but is not a power of two — the arena's offset
    // shift/mask requires a power-of-two block.
    ASSERT_FALSE(gl::isValidStaticMemoryConfig(768, 192));
}

TEST(static_memory, page_config_accepts_pow2_divisor_of_block) {
    // 256 KiB block / 8 KiB page — the production sizing.
    ASSERT_TRUE(gl::isValidStaticPageConfig(1 << 18, 1 << 13));
    // Degenerate but legal: one page per block.
    ASSERT_TRUE(gl::isValidStaticPageConfig(1 << 18, 1 << 18));
}

TEST(static_memory, page_config_rejects_nonpositive_sizes) {
    ASSERT_FALSE(gl::isValidStaticPageConfig(0, 1 << 13));
    ASSERT_FALSE(gl::isValidStaticPageConfig(1 << 18, 0));
    ASSERT_FALSE(gl::isValidStaticPageConfig(-(1 << 18), 1 << 13));
    ASSERT_FALSE(gl::isValidStaticPageConfig(1 << 18, -(1 << 13)));
}

TEST(static_memory, page_config_rejects_block_not_multiple_of_page) {
    // 8192 does not divide 262145 evenly (one byte over a whole block).
    ASSERT_FALSE(gl::isValidStaticPageConfig((1 << 18) + 1, 1 << 13));
}

TEST(static_memory, page_config_rejects_non_power_of_two_page) {
    // 192 divides 768 but is not a power of two — the within-page index is a
    // shift/mask, so a non-pow2 page is a hard misconfig.
    ASSERT_FALSE(gl::isValidStaticPageConfig(768, 192));
}

TEST(static_memory, page_config_defaults_are_valid) {
    const gl::ProverParameters p;
    ASSERT_TRUE(gl::isValidStaticPageConfig(
        p.static_block_bytes, p.static_page_bytes));
}

namespace {
    // 1 MiB pool / 256 KiB block — 4 blocks.
    const gl::StaticMemoryConfig kTestCfg{ 1 << 20, 1 << 18 };
}

TEST(static_memory, manager_init_and_shape) {
    gl::GlobalMemoryManager m;
    ASSERT_FALSE(m.initialized());
    m.init(kTestCfg);
    ASSERT_TRUE(m.initialized());
    ASSERT_EQ(m.totalBlocks(), static_cast<int64_t>(4));
    ASSERT_EQ(m.blockBytes(), 1 << 18);
    // kTestCfg gives pageBytes its struct default (8 KiB = 1 << 13).
    ASSERT_EQ(m.pageBytes(), 1 << 13);
    ASSERT_EQ(m.blocksInUse(), static_cast<int64_t>(0));
    ASSERT_EQ(m.peakBlocksInUse(), static_cast<int64_t>(0));
}

TEST(static_memory, manager_reports_configured_page_bytes) {
    // pageBytes() reflects the configured page size, not the default.
    gl::GlobalMemoryManager m;
    m.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18, 1 << 14 });
    ASSERT_EQ(m.pageBytes(), 1 << 14);
}

TEST(static_memory, persistent_flag_grants_and_recycles_like_main) {
    // The PoolKind changes only the exhaustion-assert knob name, not grant
    // behavior: a persistent-kind manager grants, releases, and recycles
    // exactly like the main pool. (The exhaustion assert itself is the
    // standard-assert abort path, untestable by this harness — same as the main
    // pool's.) 1 MiB pool / 32 KiB block (4 x 8 KiB pages) = 32 blocks.
    gl::GlobalMemoryManager m;
    m.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 15, 1 << 13,
                                   gl::PoolKind::Persistent });
    ASSERT_TRUE(m.initialized());
    ASSERT_EQ(m.blockBytes(), 1 << 15);
    ASSERT_EQ(m.pageBytes(), 1 << 13);
    ASSERT_EQ(m.totalBlocks(), static_cast<int64_t>(32));
    char* a = m.acquireBlock();
    char* b = m.acquireBlock();
    ASSERT_TRUE(a != b);
    ASSERT_EQ(m.blocksInUse(), static_cast<int64_t>(2));
    m.releaseBlock(a);
    char* c = m.acquireBlock();   // recycled block re-granted before a fresh carve
    ASSERT_EQ(c, a);
    m.releaseBlock(b);
    m.releaseBlock(c);
}

TEST(static_memory, persistent_singleton_is_distinct_from_main) {
    // The process-wide persistent pool (initialized by the harness) is a
    // SEPARATE instance from staticMemory(): distinct object, distinct ledger,
    // grants on one never touch the other (the "not depleting the other pool"
    // contract).
    ASSERT_TRUE(gl::persistentMemory().initialized());
    ASSERT_TRUE(&gl::persistentMemory() != &gl::staticMemory());
    const gl::ProverParameters defaults;
    ASSERT_EQ(gl::persistentMemory().blockBytes(),
              defaults.static_persistent_block_bytes);
    ASSERT_EQ(gl::persistentMemory().pageBytes(), defaults.static_page_bytes);

    const int64_t mainBefore = gl::staticMemory().blocksInUse();
    const int64_t persBefore = gl::persistentMemory().blocksInUse();
    char* p = gl::persistentMemory().acquireBlock();
    ASSERT_EQ(gl::staticMemory().blocksInUse(), mainBefore);   // main untouched
    ASSERT_EQ(gl::persistentMemory().blocksInUse(), persBefore + 1);
    gl::persistentMemory().releaseBlock(p);
}

TEST(static_memory, mail_kind_grants_and_recycles_like_main) {
    // PoolKind::Mail, like PoolKind::Persistent, changes only the
    // exhaustion-assert knob name, not grant behavior: a mail-kind manager
    // grants, releases, and recycles exactly like the main pool. 1 MiB pool /
    // 256 KiB block (32 x 8 KiB pages) = 4 blocks.
    gl::GlobalMemoryManager m;
    m.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18, 1 << 13,
                                   gl::PoolKind::Mail });
    ASSERT_TRUE(m.initialized());
    ASSERT_EQ(m.blockBytes(), 1 << 18);
    ASSERT_EQ(m.pageBytes(), 1 << 13);
    ASSERT_EQ(m.totalBlocks(), static_cast<int64_t>(4));
    char* a = m.acquireBlock();
    char* b = m.acquireBlock();
    ASSERT_TRUE(a != b);
    ASSERT_EQ(m.blocksInUse(), static_cast<int64_t>(2));
    m.releaseBlock(a);
    char* c = m.acquireBlock();   // recycled block re-granted before a fresh carve
    ASSERT_EQ(c, a);
    m.releaseBlock(b);
    m.releaseBlock(c);
}

TEST(static_memory, mail_singleton_is_distinct_from_other_pools) {
    // The process-wide mail pool (initialized by the harness) is a SEPARATE
    // instance from BOTH staticMemory() and persistentMemory(): distinct
    // object, distinct ledger, grants on one never touch the others (the
    // stand-alone-pool contract — mail never competes with the deloadable pool).
    ASSERT_TRUE(gl::mailMemory().initialized());
    ASSERT_TRUE(&gl::mailMemory() != &gl::staticMemory());
    ASSERT_TRUE(&gl::mailMemory() != &gl::persistentMemory());
    const gl::ProverParameters defaults;
    ASSERT_EQ(gl::mailMemory().blockBytes(), defaults.static_mail_block_bytes);
    ASSERT_EQ(gl::mailMemory().pageBytes(), defaults.static_page_bytes);

    const int64_t mainBefore = gl::staticMemory().blocksInUse();
    const int64_t persBefore = gl::persistentMemory().blocksInUse();
    const int64_t mailBefore = gl::mailMemory().blocksInUse();
    char* p = gl::mailMemory().acquireBlock();
    ASSERT_EQ(gl::staticMemory().blocksInUse(), mainBefore);        // main untouched
    ASSERT_EQ(gl::persistentMemory().blocksInUse(), persBefore);    // persistent untouched
    ASSERT_EQ(gl::mailMemory().blocksInUse(), mailBefore + 1);
    gl::mailMemory().releaseBlock(p);
}

TEST(static_memory, manager_reinit_same_config_is_noop) {
    gl::GlobalMemoryManager m;
    m.init(kTestCfg);
    char* a = m.acquireBlock();
    m.init(kTestCfg);
    ASSERT_EQ(m.totalBlocks(), static_cast<int64_t>(4));
    ASSERT_EQ(m.blocksInUse(), static_cast<int64_t>(1));
    m.releaseBlock(a);
}

TEST(static_memory, manager_blocks_are_distinct_block_strided_and_writable) {
    gl::GlobalMemoryManager m;
    m.init(kTestCfg);
    char* blocks[4];
    for (int i = 0; i < 4; ++i) blocks[i] = m.acquireBlock();
    for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 4; ++j) {
            const std::ptrdiff_t gap =
                blocks[i] < blocks[j] ? blocks[j] - blocks[i]
                                      : blocks[i] - blocks[j];
            ASSERT_GE(gap, static_cast<std::ptrdiff_t>(m.blockBytes()));
        }
    }
    // Every granted byte is usable.
    std::memset(blocks[0], 0xA5, static_cast<size_t>(m.blockBytes()));
    ASSERT_EQ(blocks[0][0], static_cast<char>(0xA5));
    ASSERT_EQ(blocks[0][m.blockBytes() - 1], static_cast<char>(0xA5));
    for (int i = 0; i < 4; ++i) m.releaseBlock(blocks[i]);
}

TEST(static_memory, lb_memory_reshuffle_preserves_content) {
    gl::GlobalMemoryManager g;
    g.init(kTestCfg);
    gl::LbMemory mem(&g);
    // Compaction scratch on an INDEPENDENT pool (mirrors production: the
    // never-deloaded LB-body pool, not the deloadable pool being compacted).
    gl::GlobalMemoryManager gScr;
    gScr.init(kTestCfg);
    gl::LbArena scratch(&gScr);
    gl::IntEncodedExpr row{};
    // Interleave growth across two containers, then hollow one out.
    for (int16_t i = 0; i < 200; ++i) {
        row.nameId = i;
        mem.intEncodedStatements.push_back(row);
        mem.intLocalEncodedStatements.push_back(row);
        mem.intLocalEncodedStatements.push_back(row);
    }
    mem.intLocalEncodedStatements.clear();
    const gl::DirtyState dirtyBefore = mem.dirty;
    mem.reshuffle(scratch);
    // Content-invisible: bytes identical, dirty untouched (a reshuffle must
    // never force a rewrite). Everything here fits one block, so no block is
    // reclaimed — the content-preservation contract is what this pins.
    ASSERT_EQ(static_cast<int>(mem.dirty),
              static_cast<int>(dirtyBefore));
    ASSERT_EQ(mem.intEncodedStatements.size(), 200);
    for (int32_t i = 0; i < 200; ++i)
        ASSERT_EQ(mem.intEncodedStatements[i].nameId,
                  static_cast<int16_t>(i));
}

TEST(static_memory, lb_memory_compaction_reclaims_holes) {
    // 16 MiB pool so the interleaved registries span several blocks and a
    // reclaimable amount of holes forms.
    const gl::StaticMemoryConfig kBigCfg{ 16 << 20, 1 << 18 };
    gl::GlobalMemoryManager g;
    g.init(kBigCfg);
    gl::LbMemory mem(&g);
    gl::GlobalMemoryManager gScr;
    gScr.init(kBigCfg);
    gl::LbArena scratch(&gScr);   // independent compaction-scratch pool
    gl::IntEncodedExpr row{};
    // Interleave a kept registry with a churned one across several blocks,
    // then hollow the churned one out — its pages return to the free-list.
    for (int16_t i = 0; i < 4000; ++i) {
        row.nameId = i;
        mem.intEncodedStatements.push_back(row);
        mem.intLocalEncodedStatements.push_back(row);
        mem.intLocalEncodedStatements.push_back(row);
    }
    mem.intLocalEncodedStatements.clear();
    const int64_t heldBefore = mem.manager.blocksHeld();
    const gl::DirtyState dirtyBefore = mem.dirty;
    const int64_t reclaimed = mem.reshuffle(scratch);
    // Reclaimed: emptied blocks returned to the pool, kept registry intact,
    // dirty untouched (the compaction is content-invisible).
    ASSERT_TRUE(reclaimed > 0);
    ASSERT_TRUE(mem.manager.blocksHeld() < heldBefore);
    ASSERT_EQ(static_cast<int>(mem.dirty), static_cast<int>(dirtyBefore));
    ASSERT_EQ(mem.intEncodedStatements.size(), 4000);
    ASSERT_EQ(mem.intLocalEncodedStatements.size(), 0);
    for (int32_t i = 0; i < 4000; ++i)
        ASSERT_EQ(mem.intEncodedStatements[i].nameId,
                  static_cast<int16_t>(i));
    // Already dense: a second pass moves nothing and reclaims nothing.
    ASSERT_EQ(mem.reshuffle(scratch), static_cast<int64_t>(0));
    ASSERT_EQ(mem.intEncodedStatements[2345].nameId,
              static_cast<int16_t>(2345));
}

TEST(static_memory, reshuffle_gate_dual_and_truth_table) {
    // The dual AND gate (bytes): reclaimable >= one block AND >= span/8.
    // Both halves hold.
    ASSERT_TRUE(gl::steward::needsReshuffle(40, 96, 32));
    // Absolute floor fails (reclaimable below one block) even at a huge
    // fraction.
    ASSERT_FALSE(gl::steward::needsReshuffle(4, 8, 32));
    // Fraction fails (reclaimable a trivial share of a big span) even above
    // the absolute floor.
    ASSERT_FALSE(gl::steward::needsReshuffle(33, 320, 32));
    // Exact boundaries qualify.
    ASSERT_TRUE(gl::steward::needsReshuffle(32, 256, 32));
}

TEST(static_memory, grant_ledger_counts_grants_only_and_resets) {
    gl::GlobalMemoryManager m;
    m.init(kTestCfg);
    ASSERT_EQ(m.grantsSinceBarrier(), static_cast<int64_t>(0));
    char* a = m.acquireBlock();
    char* b = m.acquireBlock();
    ASSERT_EQ(m.grantsSinceBarrier(), static_cast<int64_t>(2));
    // Releases do not move the ledger — it is monotone within a barrier
    // window (I-106).
    m.releaseBlock(a);
    ASSERT_EQ(m.grantsSinceBarrier(), static_cast<int64_t>(2));
    char* c = m.acquireBlock();
    ASSERT_EQ(m.grantsSinceBarrier(), static_cast<int64_t>(3));
    m.resetGrantLedger();
    ASSERT_EQ(m.grantsSinceBarrier(), static_cast<int64_t>(0));
    m.releaseBlock(b);
    m.releaseBlock(c);
}

TEST(static_memory, grant_trigger_fires_exactly_once_at_crossing) {
    gl::GlobalMemoryManager m;
    m.init(kTestCfg);
    int fired = 0;
    m.armGrantTrigger(3, [&fired]() { ++fired; });
    char* a = m.acquireBlock();
    char* b = m.acquireBlock();
    ASSERT_EQ(fired, 0);             // below threshold — silent
    char* c = m.acquireBlock();
    ASSERT_EQ(fired, 1);             // the crossing grant fires
    char* d = m.acquireBlock();
    ASSERT_EQ(fired, 1);             // self-disarmed — never twice
    m.releaseBlock(a); m.releaseBlock(b);
    m.releaseBlock(c); m.releaseBlock(d);
}

TEST(static_memory, grant_trigger_rearm_after_reset_and_disarm) {
    gl::GlobalMemoryManager m;
    m.init(kTestCfg);
    int fired = 0;
    // Disarmed manager: grants fire nothing.
    char* a = m.acquireBlock();
    ASSERT_EQ(fired, 0);
    // Armed, then disarmed at the barrier before crossing: silent.
    m.armGrantTrigger(2, [&fired]() { ++fired; });
    m.disarmGrantTrigger();
    char* b = m.acquireBlock();
    ASSERT_EQ(fired, 0);
    // Barrier reset + re-arm: the trigger lives again.
    m.resetGrantLedger();
    m.armGrantTrigger(1, [&fired]() { ++fired; });
    char* c = m.acquireBlock();
    ASSERT_EQ(fired, 1);
    // disarm after fire is a defined no-op (barrier cleanup path).
    m.disarmGrantTrigger();
    m.releaseBlock(a); m.releaseBlock(b); m.releaseBlock(c);
}

TEST(static_memory, manager_release_then_reacquire_recycles) {
    gl::GlobalMemoryManager m;
    m.init(kTestCfg);
    char* a = m.acquireBlock();
    char* b = m.acquireBlock();
    char* c = m.acquireBlock();
    char* d = m.acquireBlock();
    ASSERT_EQ(m.blocksInUse(), static_cast<int64_t>(4));
    ASSERT_EQ(m.peakBlocksInUse(), static_cast<int64_t>(4));
    m.releaseBlock(b);
    m.releaseBlock(d);
    ASSERT_EQ(m.blocksInUse(), static_cast<int64_t>(2));
    // The pool has only 4 blocks and 2 are still granted, so the next two
    // grants MUST be the recycled ones (in some order) — exhaustion would
    // assert otherwise.
    char* e = m.acquireBlock();
    char* f = m.acquireBlock();
    ASSERT_TRUE((e == b && f == d) || (e == d && f == b));
    ASSERT_EQ(m.blocksInUse(), static_cast<int64_t>(4));
    ASSERT_EQ(m.peakBlocksInUse(), static_cast<int64_t>(4));
    m.releaseBlock(a);
    m.releaseBlock(c);
    m.releaseBlock(e);
    m.releaseBlock(f);
    ASSERT_EQ(m.blocksInUse(), static_cast<int64_t>(0));
    ASSERT_EQ(m.peakBlocksInUse(), static_cast<int64_t>(4));
}

// ---- PtrDirectory -------------------------------------------------------
// The mostly-inline, pool-backed char* array that replaces LbArena's heap
// std::vector<char*> bookkeeping. A tiny inline buffer (PtrDirectory<4>) keeps
// the inline -> spill transition cheap to test: the first 4 entries hold no
// pool block, entries past that spill to 8 KiB-block-backed pages (1024 entries
// each). The grant-accounting cross-check (g.blocksInUse() == the directory's
// spilled blocks) is the legitimate-assert pairing surfaced as a test. The
// ceiling assert (a spill block past kArenaDirRootCap, 128) aborts and so is
// validated by the GL_ARENA_PARANOID run, not a death test.

namespace {
    // 1 MiB pool / 8 KiB block / 8 KiB page — 128 blocks; 1024 entries per
    // spilled block. A tiny inline buffer (4) makes the inline -> spill
    // transition cheap to test in both directions.
    const gl::StaticMemoryConfig kPtrDirCfg{ 1 << 20, 1 << 13, 1 << 13 };
    constexpr int32_t kPtrDirInline = 4;
    constexpr int32_t kSpilledPerBlock =
        (1 << 13) / static_cast<int32_t>(sizeof(char*));   // 1024
    using PtrDir = gl::PtrDirectory<kPtrDirInline>;

    // PtrDirectory stores opaque char* it never dereferences, so a synthetic
    // non-null value keyed by index is a fine stand-in for a real block/page.
    char* fakePtr(int32_t i) {
        return reinterpret_cast<char*>(static_cast<std::uintptr_t>(i) + 1u);
    }
}

TEST(ptr_directory, empty_is_consistent) {
    gl::GlobalMemoryManager g;
    g.init(kPtrDirCfg);
    PtrDir d(&g);
    ASSERT_EQ(d.size(), 0);
    ASSERT_TRUE(d.empty());
    ASSERT_EQ(d.blocksHeld(), 0);
    d.assertInvariants();
    ASSERT_EQ(g.blocksInUse(), static_cast<int64_t>(0));   // unused -> no blocks
}

TEST(ptr_directory, inline_region_holds_no_pool_block) {
    gl::GlobalMemoryManager g;
    g.init(kPtrDirCfg);
    PtrDir d(&g);
    for (int32_t i = 0; i < kPtrDirInline; ++i) {
        d.push_back(fakePtr(i));
        d.assertInvariants();
    }
    ASSERT_EQ(d.size(), kPtrDirInline);
    ASSERT_FALSE(d.empty());
    ASSERT_EQ(d.blocksHeld(), 0);                          // entirely inline
    ASSERT_EQ(g.blocksInUse(), static_cast<int64_t>(0));   // no pool block
    for (int32_t i = 0; i < kPtrDirInline; ++i) ASSERT_EQ(d[i], fakePtr(i));
    ASSERT_EQ(d.back(), fakePtr(kPtrDirInline - 1));
}

TEST(ptr_directory, spills_past_the_inline_buffer) {
    gl::GlobalMemoryManager g;
    g.init(kPtrDirCfg);
    PtrDir d(&g);
    const int32_t n = kPtrDirInline + 10;            // 10 entries past inline
    for (int32_t i = 0; i < n; ++i) d.push_back(fakePtr(i));
    d.assertInvariants();
    ASSERT_EQ(d.size(), n);
    ASSERT_EQ(d.blocksHeld(), 1);                    // one spilled block
    ASSERT_EQ(g.blocksInUse(), static_cast<int64_t>(1));
    // Entries either side of the inline/spill boundary resolve correctly.
    ASSERT_EQ(d[kPtrDirInline - 1], fakePtr(kPtrDirInline - 1));
    ASSERT_EQ(d[kPtrDirInline],     fakePtr(kPtrDirInline));
    ASSERT_EQ(d[n - 1],             fakePtr(n - 1));
}

TEST(ptr_directory, spill_crosses_block_boundary) {
    gl::GlobalMemoryManager g;
    g.init(kPtrDirCfg);
    PtrDir d(&g);
    const int32_t n = kPtrDirInline + kSpilledPerBlock + 5;  // 2nd spilled block
    for (int32_t i = 0; i < n; ++i) d.push_back(fakePtr(i));
    d.assertInvariants();
    ASSERT_EQ(d.size(), n);
    ASSERT_EQ(d.blocksHeld(), 2);
    ASSERT_EQ(g.blocksInUse(), static_cast<int64_t>(2));
    const int32_t boundary = kPtrDirInline + kSpilledPerBlock;
    ASSERT_EQ(d[boundary - 1], fakePtr(boundary - 1));
    ASSERT_EQ(d[boundary],     fakePtr(boundary));
    ASSERT_EQ(d[n - 1],        fakePtr(n - 1));
}

TEST(ptr_directory, set_overwrites_in_place_including_null) {
    gl::GlobalMemoryManager g;
    g.init(kPtrDirCfg);
    PtrDir d(&g);
    for (int32_t i = 0; i < kPtrDirInline + 5; ++i) d.push_back(fakePtr(i));
    d.set(0, fakePtr(999));                          // inline slot
    d.set(kPtrDirInline + 2, nullptr);               // spilled slot, null is legal
    ASSERT_EQ(d[0], fakePtr(999));
    ASSERT_EQ(d[kPtrDirInline + 2], nullptr);
    d.assertInvariants();
}

TEST(ptr_directory, pop_back_retains_block) {
    gl::GlobalMemoryManager g;
    g.init(kPtrDirCfg);
    PtrDir d(&g);
    for (int32_t i = 0; i < kPtrDirInline + 1; ++i)
        d.push_back(fakePtr(i));                     // one spilled block
    ASSERT_EQ(d.blocksHeld(), 1);
    ASSERT_EQ(g.blocksInUse(), static_cast<int64_t>(1));
    d.pop_back();                                    // back into the inline region
    ASSERT_EQ(d.size(), kPtrDirInline);
    ASSERT_EQ(d.blocksHeld(), 1);                    // spilled block retained (no churn)
    ASSERT_EQ(g.blocksInUse(), static_cast<int64_t>(1));
    d.assertInvariants();
    d.push_back(fakePtr(42));                        // refills the retained block
    ASSERT_EQ(d.blocksHeld(), 1);
    ASSERT_EQ(g.blocksInUse(), static_cast<int64_t>(1));   // no new grant
    ASSERT_EQ(d.back(), fakePtr(42));
}

TEST(ptr_directory, shrink_to_fit_releases_spilled_blocks) {
    gl::GlobalMemoryManager g;
    g.init(kPtrDirCfg);
    PtrDir d(&g);
    for (int32_t i = 0; i < kPtrDirInline + 2 * kSpilledPerBlock; ++i)
        d.push_back(fakePtr(i));                     // two spilled blocks
    ASSERT_EQ(d.blocksHeld(), 2);
    d.truncate(kPtrDirInline);                       // shrink to all-inline, RETAIN
    ASSERT_EQ(d.blocksHeld(), 2);                    // retained, no pool churn
    ASSERT_EQ(g.blocksInUse(), static_cast<int64_t>(2));
    d.assertInvariants();
    d.shrinkToFit();                                 // coarse-seam reclaim
    ASSERT_EQ(d.blocksHeld(), 0);                    // spilled blocks returned
    ASSERT_EQ(g.blocksInUse(), static_cast<int64_t>(0));
    d.assertInvariants();
}

TEST(ptr_directory, clear_returns_all_blocks_and_reuses) {
    gl::GlobalMemoryManager g;
    g.init(kPtrDirCfg);
    PtrDir d(&g);
    for (int32_t i = 0; i < kPtrDirInline + 2 * kSpilledPerBlock + 7; ++i)
        d.push_back(fakePtr(i));                     // three spilled blocks
    ASSERT_EQ(g.blocksInUse(), static_cast<int64_t>(3));
    d.clear();
    ASSERT_EQ(d.size(), 0);
    ASSERT_EQ(d.blocksHeld(), 0);
    ASSERT_EQ(g.blocksInUse(), static_cast<int64_t>(0));   // every block back
    d.assertInvariants();
    d.push_back(fakePtr(1));                         // usable after clear
    ASSERT_EQ(d.size(), 1);
    ASSERT_EQ(d[0], fakePtr(1));
}

TEST(ptr_directory, lazy_bind_consumes_nothing) {
    gl::GlobalMemoryManager g;
    g.init(kPtrDirCfg);
    PtrDir d;                                        // unbound
    d.bind(&g);
    ASSERT_EQ(d.blocksHeld(), 0);
    ASSERT_EQ(g.blocksInUse(), static_cast<int64_t>(0));
    d.push_back(fakePtr(5));                         // inline, no block
    ASSERT_EQ(d[0], fakePtr(5));
    ASSERT_EQ(g.blocksInUse(), static_cast<int64_t>(0));
}

TEST(ptr_directory, destructor_returns_blocks) {
    gl::GlobalMemoryManager g;
    g.init(kPtrDirCfg);
    {
        PtrDir d(&g);
        for (int32_t i = 0; i < kPtrDirInline + 1; ++i)
            d.push_back(fakePtr(i));
        ASSERT_EQ(g.blocksInUse(), static_cast<int64_t>(1));
    }                                                // ~PtrDirectory -> clear()
    ASSERT_EQ(g.blocksInUse(), static_cast<int64_t>(0));
}

// The raised root cap (kArenaDirRootCap 16 -> 128) must keep every entry
// addressable deep into the spilled region: a directory pushed past the OLD
// 16-spill-block ceiling (where the vid-ratcheted rung-2 pageTable_ formerly
// asserted) resolves entries exactly at and beyond that boundary, including
// interior null holes (the freed-vid shape that drives the ratchet).
TEST(ptr_directory, spilled_region_past_old_sixteen_block_cap) {
    gl::GlobalMemoryManager g;
    g.init(kPtrDirCfg);
    PtrDir d(&g);
    const int32_t kOldCapEntries = kPtrDirInline + 16 * kSpilledPerBlock;
    const int32_t n = kPtrDirInline + 20 * kSpilledPerBlock;   // 20 spill blocks
    for (int32_t i = 0; i < n; ++i) {
        // Interior null holes every 7th entry — the freed-vid pattern.
        d.push_back((i % 7 == 3) ? nullptr : fakePtr(i));
    }
    d.assertInvariants();
    ASSERT_EQ(d.size(), n);
    ASSERT_EQ(d.blocksHeld(), 20);
    ASSERT_EQ(g.blocksInUse(), static_cast<int64_t>(20));
    // Entries either side of the old 16-block ceiling and at the far tail.
    for (const int32_t i : { 0, kPtrDirInline - 1, kPtrDirInline,
                             kOldCapEntries - 1, kOldCapEntries,
                             kOldCapEntries + 1, n - 1 }) {
        if (i % 7 == 3) ASSERT_TRUE(d[i] == nullptr);
        else            ASSERT_EQ(d[i], fakePtr(i));
    }
    // Sampled sweep across every spill block.
    for (int32_t i = 0; i < n; i += 997) {
        if (i % 7 == 3) ASSERT_TRUE(d[i] == nullptr);
        else            ASSERT_EQ(d[i], fakePtr(i));
    }
    d.clear();
    ASSERT_EQ(g.blocksInUse(), static_cast<int64_t>(0));
}
