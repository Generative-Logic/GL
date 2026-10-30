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
/// @brief Unit tests for the memory-playbook instrumentation — pool
///        attribution by purpose and prover step, the arena / registry /
///        sealed-page / LB-store purpose wiring, the process census, the
///        physical view of the per-structure tracker, and the CUDA host-arena
///        byte measures.
///
/// @details
/// Every instrument is observability only; these tests pin the accounting
/// identities the memory report relies on: a purpose split sums to the level
/// it describes, a return under the grant's purpose leaves no residue, a
/// step's high-water keeps the split of its own instant, the census folds its
/// samples into the step that was current, and `LbMemory::liveBytes` is
/// exactly the sum of the named terms the tracker attributes.

#include <cstddef>
#include <utility>
#include "test_harness.hpp"

#include "../memory.hpp"
#include "../memory_infra/global_memory_manager.hpp"
#include "../memory_infra/lb_arena.hpp"
#include "../memory_infra/lb_store.hpp"
#include "../memory_infra/paged_vector.hpp"
#include "../memory_infra/scratch_arena.hpp"
#include "../memory_infra/sealed_pages.hpp"
#include "../memory_infra/steward.hpp"
#include "../infra/memory_census.hpp"
#include "../gpu/phase2_projection.hpp"

#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

namespace {

    // 2 MiB pool / 256 KiB block / 8 KiB page — eight blocks.
    const gl::StaticMemoryConfig kPlaybookCfg{ 2 << 20, 1 << 18, 1 << 13 };

    /// Sum a purpose row.
    int64_t sumRow(const int64_t* row) {
        int64_t s = 0;
        for (int k = 0; k < gl::kPoolPurposeCount; ++k) s += row[k];
        return s;
    }

    int idx(gl::PoolPurpose p) { return static_cast<int>(p); }
    int idx(gl::PoolPhase p) { return static_cast<int>(p); }

    /// Build a statement row with a distinguishable name id.
    gl::IntEncodedExpr makeRow(gl::NameId i) {
        gl::IntEncodedExpr e{};
        e.nameId = i;
        e.originalId = i;
        e.validityId = 1;
        return e;
    }

}  // namespace

TEST(memory_playbook, pool_purpose_split_sums_to_the_peak) {
    gl::GlobalMemoryManager m;
    m.init(kPlaybookCfg);
    char* a = m.acquireBlock(gl::PoolPurpose::LbState);
    char* b = m.acquireBlock(gl::PoolPurpose::LbState);
    char* c = m.acquireBlock(gl::PoolPurpose::ScratchGen);
    gl::PoolAttribution at = m.attribution();
    ASSERT_EQ(at.peakBlocks, int64_t(3));
    ASSERT_EQ(at.atPeak[idx(gl::PoolPurpose::LbState)], int64_t(2));
    ASSERT_EQ(at.atPeak[idx(gl::PoolPurpose::ScratchGen)], int64_t(1));
    ASSERT_EQ(sumRow(at.atPeak), at.peakBlocks);
    ASSERT_EQ(sumRow(at.inUse), m.blocksInUse());

    // Returning below the peak keeps the peak row; the own peaks stay.
    m.releaseBlock(c, gl::PoolPurpose::ScratchGen);
    char* d = m.acquireBlock(gl::PoolPurpose::Sealed);
    at = m.attribution();
    ASSERT_EQ(at.peakBlocks, int64_t(3));
    ASSERT_EQ(at.atPeak[idx(gl::PoolPurpose::ScratchGen)], int64_t(1));
    ASSERT_EQ(at.atPeak[idx(gl::PoolPurpose::Sealed)], int64_t(0));
    ASSERT_EQ(at.purposePeak[idx(gl::PoolPurpose::Sealed)], int64_t(1));
    ASSERT_EQ(at.inUse[idx(gl::PoolPurpose::ScratchGen)], int64_t(0));

    // A new high re-snapshots the whole split at that instant.
    char* e = m.acquireBlock(gl::PoolPurpose::GpuHost);
    at = m.attribution();
    ASSERT_EQ(at.peakBlocks, int64_t(4));
    ASSERT_EQ(at.atPeak[idx(gl::PoolPurpose::Sealed)], int64_t(1));
    ASSERT_EQ(at.atPeak[idx(gl::PoolPurpose::GpuHost)], int64_t(1));
    ASSERT_EQ(at.atPeak[idx(gl::PoolPurpose::ScratchGen)], int64_t(0));
    ASSERT_EQ(sumRow(at.atPeak), int64_t(4));

    // An unspecified purpose is the visible Unattributed residual.
    char* f = m.acquireBlock();
    ASSERT_EQ(m.attribution().inUse[idx(gl::PoolPurpose::Unattributed)], int64_t(1));
    m.releaseBlock(f);
    for (char* x : { a, b }) m.releaseBlock(x, gl::PoolPurpose::LbState);
    m.releaseBlock(d, gl::PoolPurpose::Sealed);
    m.releaseBlock(e, gl::PoolPurpose::GpuHost);
    at = m.attribution();
    ASSERT_EQ(sumRow(at.inUse), int64_t(0));
    ASSERT_EQ(m.blocksInUse(), int64_t(0));
}

TEST(memory_playbook, bulk_grant_and_return_carry_the_purpose) {
    gl::GlobalMemoryManager m;
    m.init(kPlaybookCfg);
    char* blocks[3] = {};
    m.acquireBlocks(3, blocks, gl::PoolPurpose::LbState);
    gl::PoolAttribution at = m.attribution();
    ASSERT_EQ(at.inUse[idx(gl::PoolPurpose::LbState)], int64_t(3));
    ASSERT_EQ(at.atPeak[idx(gl::PoolPurpose::LbState)], int64_t(3));
    m.releaseBlocks(blocks, 3, gl::PoolPurpose::LbState);
    ASSERT_EQ(m.attribution().inUse[idx(gl::PoolPurpose::LbState)], int64_t(0));
}

TEST(memory_playbook, step_peaks_keep_their_own_instant) {
    gl::GlobalMemoryManager m;
    m.init(kPlaybookCfg);
    m.setAttributionPhase(gl::PoolPhase::Phase1);
    char* a = m.acquireBlock(gl::PoolPurpose::LbState);
    char* b = m.acquireBlock(gl::PoolPurpose::ScratchString);
    m.releaseBlock(b, gl::PoolPurpose::ScratchString);
    m.setAttributionPhase(gl::PoolPhase::Phase2Device);
    char* c = m.acquireBlock(gl::PoolPurpose::GpuHost);
    char* d = m.acquireBlock(gl::PoolPurpose::GpuHost);
    char* e = m.acquireBlock(gl::PoolPurpose::Sealed);
    m.setAttributionPhase(gl::PoolPhase::Phase3);   // grants nothing
    const gl::PoolAttribution at = m.attribution();

    ASSERT_EQ(at.phasePeak[idx(gl::PoolPhase::Phase1)], int64_t(2));
    ASSERT_EQ(at.phaseAtPeak[idx(gl::PoolPhase::Phase1)][idx(gl::PoolPurpose::ScratchString)], int64_t(1));
    ASSERT_EQ(at.phasePeak[idx(gl::PoolPhase::Phase2Device)], int64_t(4));
    ASSERT_EQ(at.phaseAtPeak[idx(gl::PoolPhase::Phase2Device)][idx(gl::PoolPurpose::GpuHost)], int64_t(2));
    ASSERT_EQ(sumRow(at.phaseAtPeak[idx(gl::PoolPhase::Phase2Device)]), int64_t(4));
    // A step that grants nothing still reports the level it ran at.
    ASSERT_EQ(at.phasePeak[idx(gl::PoolPhase::Phase3)], int64_t(4));
    // A step never entered stays zero.
    ASSERT_EQ(at.phasePeak[idx(gl::PoolPhase::Phase2Finalize)], int64_t(0));

    m.releaseBlock(a, gl::PoolPurpose::LbState);
    for (char* x : { c, d }) m.releaseBlock(x, gl::PoolPurpose::GpuHost);
    m.releaseBlock(e, gl::PoolPurpose::Sealed);
}

TEST(memory_playbook, arena_purpose_covers_data_and_directory_blocks) {
    gl::GlobalMemoryManager m;
    m.init(gl::StaticMemoryConfig{ 8 << 20, 1 << 14, 1 << 10 });  // 16 KiB blocks, 1 KiB pages
    {
        gl::LbArena arena(&m);
        ASSERT_EQ(arena.purpose(), gl::PoolPurpose::Unattributed);
        arena.setPurpose(gl::PoolPurpose::LbState);
        ASSERT_EQ(arena.purpose(), gl::PoolPurpose::LbState);
        // Enough pages to spill the 128-entry inline page table and the
        // 16-entry inline block table into pool blocks of their own.
        std::vector<int32_t> vids;
        for (int i = 0; i < 400; ++i) vids.push_back(arena.allocPage());
        arena.alloc(64);   // one byte-bump block too
        const int64_t held = arena.blocksHeld();
        ASSERT_TRUE(held > 25);   // 400 pages / 16 per block + spills + bump
        const gl::PoolAttribution at = m.attribution();
        ASSERT_EQ(at.inUse[idx(gl::PoolPurpose::LbState)], held);
        ASSERT_EQ(at.inUse[idx(gl::PoolPurpose::Unattributed)], int64_t(0));
        arena.releaseAll();
        ASSERT_EQ(m.attribution().inUse[idx(gl::PoolPurpose::LbState)], int64_t(0));
    }
    ASSERT_EQ(m.blocksInUse(), int64_t(0));
}

TEST(memory_playbook, compaction_returns_blocks_under_the_arena_purpose) {
    gl::GlobalMemoryManager m;
    m.init(kPlaybookCfg);
    gl::GlobalMemoryManager scratchPool;
    scratchPool.init(kPlaybookCfg);
    gl::LbArena arena(&m);
    arena.setPurpose(gl::PoolPurpose::LbState);
    gl::LbArena scr(&scratchPool);
    std::vector<int32_t> vids;
    for (int i = 0; i < 70; ++i) vids.push_back(arena.allocPage());   // 3 blocks of 32 pages
    for (int i = 0; i < 60; ++i) arena.freePage(vids[static_cast<std::size_t>(i)]);
    const int64_t before = m.attribution().inUse[idx(gl::PoolPurpose::LbState)];
    const int64_t reclaimed = arena.compactPages(scr);
    ASSERT_TRUE(reclaimed >= 1);
    ASSERT_EQ(m.attribution().inUse[idx(gl::PoolPurpose::LbState)], before - reclaimed);
    ASSERT_EQ(m.attribution().inUse[idx(gl::PoolPurpose::LbState)], arena.blocksHeld());
    arena.releaseAll();
    scr.releaseAll();
}

TEST(memory_playbook, scratch_registry_and_sealed_pages_are_attributed) {
    gl::GlobalMemoryManager m;
    m.init(kPlaybookCfg);
    gl::ScratchArenaRegistry reg;
    reg.init(&m, 2, gl::PoolPurpose::ScratchGen);
    ASSERT_EQ(reg.forSlot(0).purpose(), gl::PoolPurpose::ScratchGen);
    reg.forSlot(1).allocBytes(32);
    ASSERT_EQ(m.attribution().inUse[idx(gl::PoolPurpose::ScratchGen)], int64_t(1));
    reg.forSlot(1).releaseAll();

    gl::SealedPageSet set;
    set.bind(&m);
    set.alloc(16);
    ASSERT_EQ(m.attribution().inUse[idx(gl::PoolPurpose::Sealed)], int64_t(1));
    set.seal();
    set.freePages();
    ASSERT_EQ(m.attribution().inUse[idx(gl::PoolPurpose::Sealed)], int64_t(0));
    ASSERT_EQ(m.blocksInUse(), int64_t(0));
}

TEST(memory_playbook, lb_store_slots_are_lb_shell_blocks) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 32768, 8192, 8192, gl::PoolKind::Lb });
    {
        gl::LbStore store(&g, 64, 16);
        struct Item { int v; };
        Item* item = store.create<Item>(Item{ 7 });
        ASSERT_EQ(g.attribution().inUse[idx(gl::PoolPurpose::LbShell)], int64_t(1));
        store.destroy(item);
    }
    ASSERT_EQ(g.attribution().inUse[idx(gl::PoolPurpose::LbShell)], int64_t(0));
}

TEST(memory_playbook, census_folds_samples_into_the_current_step) {
    gl::memory_census::resetForTest();
    const gl::memory_census::ProcessSample now = gl::memory_census::readProcess();
    ASSERT_TRUE(now.workingSetBytes > 0);
    ASSERT_TRUE(now.privateBytes > 0);
    const gl::memory_census::ProcessPeaks peaks = gl::memory_census::readProcessPeaks();
    ASSERT_GE(peaks.peakWorkingSetBytes, now.workingSetBytes);

    gl::memory_census::markPhase(gl::PoolPhase::Phase1);   // closes Outside
    // Touch 64 MiB inside Phase1; the step's maximum must cover a reading
    // taken right after the touch (the census keeps the maximum, not the
    // last sample).
    std::vector<char> ballast(64u << 20);
    std::memset(ballast.data(), 1, ballast.size());
    const gl::memory_census::ProcessSample touched = gl::memory_census::readProcess();
    gl::memory_census::sampleNow();
    ballast = std::vector<char>();
    gl::memory_census::markPhase(gl::PoolPhase::Outside);  // closes Phase1
    const gl::memory_census::PhaseMaxima pm = gl::memory_census::phaseMaxima();
    ASSERT_EQ(pm.samples[idx(gl::PoolPhase::Outside)], int64_t(1));
    ASSERT_EQ(pm.samples[idx(gl::PoolPhase::Phase1)], int64_t(2));
    ASSERT_GE(pm.workingSet[idx(gl::PoolPhase::Phase1)],
              touched.workingSetBytes - (int64_t(8) << 20));
    ASSERT_EQ(pm.samples[idx(gl::PoolPhase::Phase3)], int64_t(0));
    gl::memory_census::resetForTest();
    ASSERT_EQ(gl::memory_census::phaseMaxima().samples[idx(gl::PoolPhase::Phase1)], int64_t(0));
}

TEST(memory_playbook, census_keeps_gpu_maxima_and_device_totals) {
    gl::memory_census::resetForTest();
    gl::memory_census::noteGpuChunk({ 100, 200, 10, 5, 3, 50, 40 });
    gl::memory_census::noteGpuChunk({ 300, 50, 1, 1, 1, 1, 1 });
    gl::memory_census::noteGpuDevice(8000, 7000, 6000);
    gl::memory_census::noteGpuDevice(8000, 7000, 6000);   // a repeat must match
    const gl::memory_census::GpuMaxima g = gl::memory_census::gpuMaxima();
    ASSERT_EQ(g.chunks, int64_t(2));
    ASSERT_EQ(g.maxOfEach.projection, uint64_t(300));
    ASSERT_EQ(g.maxOfEach.shards, uint64_t(200));
    ASSERT_EQ(g.maxChunkTotal, uint64_t(408));   // first chunk sums to 408
    ASSERT_TRUE(g.deviceSeen);
    ASSERT_EQ(g.deviceBudgetBytes, uint64_t(6000));

    std::ostringstream diag, md;
    gl::memory_census::writeReport(diag, md);
    const std::string d = diag.str();
    ASSERT_TRUE(d.find("[process-memory]") != std::string::npos);
    ASSERT_TRUE(d.find("[gpu-memory]") != std::string::npos);
    ASSERT_TRUE(d.find("max_chunk_host_total_bytes=408") != std::string::npos);
    ASSERT_TRUE(md.str().find("### Process memory") != std::string::npos);
    gl::memory_census::resetForTest();
}

TEST(memory_playbook, live_bytes_is_the_sum_of_the_named_terms) {
    gl::Memory m;
    m.setExprKey("(memory_playbook_live_bytes)");
    for (gl::NameId i = 0; i < 40; ++i) ((m).intEncodedStatements).push_back(makeRow(i));
    int64_t sum = 0;
    int terms = 0;
    bool sawStatements = false;
    bool sawSharedRecords = false;
    ((m).lbMemory).visitLiveBytes([&](const char* name, int64_t bytes) {
        sum += bytes;
        ++terms;
        if (std::strcmp(name, "intEncodedStatements") == 0) {
            sawStatements = true;
            ASSERT_TRUE(bytes >= int64_t(40 * sizeof(int32_t)));
        }
        if (std::strcmp(name, "sharedExpressions") == 0) {
            sawSharedRecords = true;
            ASSERT_TRUE(bytes >= int64_t(40 * sizeof(gl::IntEncodedExpr)));
        }
    });
    ASSERT_TRUE(sawStatements);
    ASSERT_TRUE(sawSharedRecords);
    ASSERT_TRUE(terms > 40);
    ASSERT_EQ(sum, ((m).lbMemory).liveBytes());
}

TEST(memory_playbook, host_arena_bytes_follow_the_reserved_capacity) {
    std::vector<int32_t> a;
    a.reserve(10);
    std::vector<char> b;
    b.reserve(7);
    ASSERT_EQ(gl::gpu::vectorCapacityBytes(a, b),
              uint64_t(a.capacity() * sizeof(int32_t) + b.capacity()));
    ASSERT_EQ(gl::gpu::vectorCapacityBytes(), uint64_t(0));

    gl::gpu::Phase2TaskProjectionCapacity capacity{};
    capacity.tasks = 2;
    capacity.batches = 8;
    capacity.terms = 8;
    capacity.stumps = 4;
    gl::gpu::Phase2TaskProjectionArena arena(capacity);
    ASSERT_EQ(arena.hostBytes(), gl::gpu::Phase2TaskProjectionArena::hostLayout(capacity).roundedBytes(262144));
    ASSERT_TRUE(arena.hostBytes() > 0);
}

TEST(memory_playbook, compaction_gate_reads_parked_free_pages) {
    gl::GlobalMemoryManager m;
    m.init(kPlaybookCfg);   // 256 KiB blocks, 8 KiB pages: 32 pages per block
    gl::LbMemory lb(&m);
    std::vector<int32_t> vids;
    for (int i = 0; i < 10; ++i) vids.push_back(lb.manager.allocPage());
    // One block, 22 free pages parked: less than a block of room.
    ASSERT_FALSE(gl::steward::arenaNeedsCompaction(lb.manager));
    for (int i = 10; i < 70; ++i) vids.push_back(lb.manager.allocPage());
    ASSERT_EQ(lb.manager.blocksHeld(), int64_t(3));
    ASSERT_FALSE(gl::steward::arenaNeedsCompaction(lb.manager));   // 26 free of 96
    for (int i = 0; i < 60; ++i) lb.manager.freePage(vids[static_cast<std::size_t>(i)]);
    // 86 free pages: two whole blocks of room, far above an eighth of the span.
    ASSERT_TRUE(gl::steward::arenaNeedsCompaction(lb.manager));
    lb.manager.releaseAll();
}

TEST(memory_playbook, compact_if_fragmented_returns_whole_blocks_once) {
    gl::GlobalMemoryManager m;
    m.init(kPlaybookCfg);
    gl::GlobalMemoryManager neverDeloaded;
    neverDeloaded.init(kPlaybookCfg);
    gl::LbArena scratch(&neverDeloaded);
    gl::LbMemory lb(&m);
    std::vector<int32_t> vids;
    for (int i = 0; i < 70; ++i) vids.push_back(lb.manager.allocPage());
    for (int i = 0; i < 60; ++i) lb.manager.freePage(vids[static_cast<std::size_t>(i)]);
    // Mark the ten live pages so the pack can be checked content-preserving.
    for (int i = 60; i < 70; ++i)
        std::memset(lb.manager.pageAt(vids[static_cast<std::size_t>(i)]), i, 16);
    ASSERT_EQ(gl::steward::compactIfFragmented(lb, scratch), int64_t(2));
    ASSERT_EQ(lb.manager.blocksHeld(), int64_t(1));
    ASSERT_EQ(m.attribution().inUse[idx(gl::PoolPurpose::LbState)], int64_t(1));
    for (int i = 60; i < 70; ++i)
        ASSERT_EQ(lb.manager.pageAt(vids[static_cast<std::size_t>(i)])[15], static_cast<char>(i));
    // Below the gate now: a second call is the defined no-op.
    ASSERT_EQ(gl::steward::compactIfFragmented(lb, scratch), int64_t(0));
    lb.manager.releaseAll();
    scratch.releaseAll();
}

TEST(memory_playbook, compaction_scratch_registry_is_on_the_lb_pool) {
    ASSERT_TRUE(gl::compactionScratchArenas().initialized());
    const unsigned slots = gl::compactionScratchArenas().slotCount();
    ASSERT_TRUE(slots >= 2u);
    gl::LbArena& a = gl::compactionScratchArenas().forSlot(slots - 1);
    ASSERT_EQ(a.purpose(), gl::PoolPurpose::ScratchGen);
    const int64_t before = gl::lbMemory().blocksInUse();
    const int32_t vid = a.allocPage();
    ASSERT_EQ(gl::lbMemory().blocksInUse(), before + 1);   // drawn from the LB-body pool
    a.freePage(vid);
    a.releaseAll();
    ASSERT_EQ(gl::lbMemory().blocksInUse(), before);
}

// 16 KiB blocks, 1 KiB pages: 400 pages fill 25 page blocks, past both the
// 128-entry inline page table and the 16-entry inline carved-block list.
namespace {
    const gl::StaticMemoryConfig kDirectoryCfg{ 8 << 20, 1 << 14, 1 << 10 };
}

TEST(memory_playbook, attached_directory_storage_holds_no_spill_block) {
    gl::GlobalMemoryManager m;
    m.init(kDirectoryCfg);
    gl::GlobalMemoryManager scratchPool;
    scratchPool.init(kDirectoryCfg);
    {
        // Reference: an arena with inline directories spills both of them.
        gl::LbArena plain(&m);
        for (int i = 0; i < 400; ++i) plain.allocPage();
        ASSERT_EQ(plain.blocksHeld(), int64_t(27));   // 25 data + 2 directory
        plain.releaseAll();
    }
    char* pageTable[512];
    char* pageBlocks[32];
    gl::LbArena arena(&m);
    arena.attachDirectoryStorage(pageTable, 512, pageBlocks, 32);
    std::vector<int32_t> vids;
    for (int i = 0; i < 400; ++i) vids.push_back(arena.allocPage());
    ASSERT_EQ(arena.blocksHeld(), int64_t(25));       // data blocks only
    ASSERT_EQ(m.blocksInUse(), int64_t(25));
    for (int i = 0; i < 400; ++i) {
        const int32_t v = vids[static_cast<std::size_t>(i)];
        ASSERT_TRUE(pageTable[v] == arena.pageAt(v));   // the table is the caller's
        std::memset(arena.pageAt(v), i & 0x7f, 8);
    }
    // Compaction rebinds vids through the attached table and keeps content.
    for (int i = 0; i < 300; ++i) arena.freePage(vids[static_cast<std::size_t>(i)]);
    gl::LbArena scr(&scratchPool);
    ASSERT_TRUE(arena.compactPages(scr) >= 1);
    ASSERT_EQ(arena.blocksHeld(), int64_t(7));        // 100 live pages / 16 per block
    for (int i = 300; i < 400; ++i) {
        const int32_t v = vids[static_cast<std::size_t>(i)];
        ASSERT_TRUE(pageTable[v] == arena.pageAt(v));
        ASSERT_EQ(arena.pageAt(v)[7], static_cast<char>(i & 0x7f));
    }
    // A release keeps the attachment: the next fill again needs no spill.
    arena.releaseAll();
    ASSERT_EQ(m.blocksInUse(), int64_t(0));
    for (int i = 0; i < 400; ++i) arena.allocPage();
    ASSERT_EQ(arena.blocksHeld(), int64_t(25));
    arena.releaseAll();
    scr.releaseAll();
}

TEST(memory_playbook, lb_memory_directories_live_in_its_shell) {
    gl::GlobalMemoryManager m;
    m.init(kDirectoryCfg);
    gl::LbMemory lb(&m);
    for (int i = 0; i < 400; ++i) lb.manager.allocPage();
    ASSERT_EQ(lb.manager.blocksHeld(), int64_t(25));   // no directory spill
    ASSERT_EQ(m.attribution().inUse[idx(gl::PoolPurpose::LbState)], int64_t(25));
    lb.manager.releaseAll();
    // The shell still fits one LB-pool slot with room to spare.
    ASSERT_TRUE(sizeof(gl::Memory) <= static_cast<std::size_t>(gl::lbMemory().blockBytes()));
}

TEST(memory_playbook, gpu_host_report_separates_reservation_commit_and_registration) {
    gl::GpuHostArena storage;
    storage.reserve(262144);
    auto& pool = gl::gpuHostStorage();
    pool.setGpuHostPinned(storage.data(), storage.blockCount(), true);
    std::ostringstream diagnostics, markdown;
    gl::memory_census::writeReport(diagnostics, markdown);
    ASSERT_TRUE(diagnostics.str().find("[GPU-HOST-MEMORY] reserved_bytes=") != std::string::npos);
    ASSERT_TRUE(diagnostics.str().find(" pinned_bytes=262144") != std::string::npos);
    ASSERT_TRUE(diagnostics.str().find("pool=gpu_host") != std::string::npos);
    ASSERT_TRUE(markdown.str().find("Independent GPU-host reservation") != std::string::npos);
    ASSERT_TRUE(gl::memory_census::poolTouchedBytes() >= pool.gpuHostCommittedBytes());
    pool.setGpuHostPinned(storage.data(), storage.blockCount(), false);
    ASSERT_TRUE(pool.gpuHostPeakPinnedBytes() >= 262144);
}
