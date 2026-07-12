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
/// @brief Unit tests for the deload telemetry aggregate
///        (`memory_infra/deload_stats.*`).
///
/// @details
/// Pins the pure-telemetry contract: the record helpers accumulate, a real
/// LB deload/reload round-trip moves the v3 dump/load counters, a POD-key
/// cold-set reload times and counts its throw-away index rebuild, a resident
/// re-dump counts the skip-clean branch, the per-phase window snapshot math
/// is exact, and the reports render. Every test `reset()`s the process-wide
/// aggregate first — it is shared, but the suite runs sequentially. Deload
/// files land under `.debug/test_deload_stats_*` (gitignored, recreated).

#include "test_harness.hpp"

#include "../memory.hpp"
#include "../memory_infra/cold_hash_map.hpp"
#include "../memory_infra/deload_stats.hpp"
#include "../memory_infra/global_memory_manager.hpp"
#include "../memory_infra/lb_arena.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace {

    using PodSet = gl::ColdHashSet<gl::PodKeyStore<int32_t>>;

    const gl::StaticMemoryConfig kStatsTestCfg{ 1 << 20, 1 << 18 };

    /// Minimal patterned statement row (value-initialized so padding is
    /// deterministic zero).
    gl::IntEncodedExpr statsExpr(int16_t seed) {
        gl::IntEncodedExpr e{};
        e.nameId = seed;
        e.originalId = static_cast<int16_t>(seed + 1);
        e.validityId = 1;
        e.arity = 1;
        e.maxIteration = -1;
        return e;
    }

    /// Fresh per-test deload directory under .debug (name carries "deload"
    /// so the purge-safety assert would hold if ever applied).
    std::string freshStatsDir(const std::string& name) {
        const std::filesystem::path dir =
            std::filesystem::path(".debug") / ("deload_" + name);
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
        std::filesystem::create_directories(dir, ec);
        return dir.string();
    }

}

TEST(deload_stats, record_helpers_accumulate) {
    gl::DeloadStats& st = gl::deloadStats();
    st.reset();

    st.recordV3Dump(1000, 500);
    st.recordV3Dump(200, 100);
    ASSERT_EQ(st.v3DumpBytes.load(), 1200);
    ASSERT_EQ(st.v3DumpNs.load(), 600);
    ASSERT_EQ(st.v3DumpCount.load(), 2);

    st.recordV3Load(3000, 900);
    ASSERT_EQ(st.v3LoadBytes.load(), 3000);
    ASSERT_EQ(st.v3LoadCount.load(), 1);

    st.recordIndexRebuild(42, 7);
    ASSERT_EQ(st.indexRebuildKeys.load(), 42);
    ASSERT_EQ(st.indexRebuildNs.load(), 7);

    st.recordEviction(8, 111);
    ASSERT_EQ(st.evictCount.load(), 1);
    ASSERT_EQ(st.evictBlocks.load(), 8);
    ASSERT_EQ(st.evictDumpNs.load(), 111);

    st.recordWorkerSelfLoad(55);
    st.recordWorkerInlineEvict(66);
    ASSERT_EQ(st.workerSelfLoadCount.load(), 1);
    ASSERT_EQ(st.workerInlineEvictCount.load(), 1);

    st.recordWorkerClaimWait(2, 400);
    st.recordWorkerClaimWait(2, 100);
    ASSERT_EQ(st.workerClaimWaitNs[2].load(), 500);
    ASSERT_EQ(st.workerClaimWaitCount[2].load(), 2);

    // Single-counter members are bumped directly at call sites.
    st.prefetchHit.fetch_add(1);
    ASSERT_EQ(st.prefetchHit.load(), 1);

    st.reset();
    ASSERT_EQ(st.v3DumpBytes.load(), 0);
    ASSERT_EQ(st.workerClaimWaitNs[2].load(), 0);
    ASSERT_EQ(st.prefetchHit.load(), 0);
    ASSERT_TRUE(st.lbHistogramPending());
}

TEST(deload_stats, round_trip_moves_dump_and_load_counters) {
    gl::DeloadStats& st = gl::deloadStats();
    st.reset();
    const std::string dir = freshStatsDir("round_trip");

    gl::Memory m;
    m.setExprKey("(deload_stats_round_trip_lb)");
    for (int16_t i = 0; i < 16; ++i)
        m.intEncodedStatements.push_back(statsExpr(i));

    m.deloadStaticContainers(dir);       // dump + release
    ASSERT_FALSE(m.lbMemory.manager.resident());
    ASSERT_EQ(st.v3DumpCount.load(), 1);
    ASSERT_EQ(st.fullDumpCount.load(), 1);
    ASSERT_TRUE(st.v3DumpBytes.load() > 0);

    m.ensureLoaded(dir);                 // reload
    ASSERT_TRUE(m.lbMemory.manager.resident());
    ASSERT_EQ(st.v3LoadCount.load(), 1);
    ASSERT_TRUE(st.v3LoadBytes.load() > 0);
    // The reload restreams exactly what was dumped.
    ASSERT_EQ(st.v3LoadBytes.load(), st.v3DumpBytes.load());
}

TEST(deload_stats, index_rebuild_timed_and_counted_on_reload) {
    gl::DeloadStats& st = gl::deloadStats();
    st.reset();

    gl::GlobalMemoryManager g;
    g.init(kStatsTestCfg);
    gl::LbArena lb(&g);
    gl::DirtyState dirty = gl::DirtyState::Clean;

    // Enough keys that the reload rebuild is comfortably above the
    // steady-clock resolution (so the ns delta is reliably nonzero).
    const int32_t N = 4000;
    PodSet s(&lb, &dirty);
    for (int32_t i = 0; i < N; ++i) s.mint(5 * i + 1);
    PodSet::KeysView sKeys(&s);
    std::vector<char> sk;
    sKeys.appendSpanBytes(sk, 0);

    gl::LbArena lb2(&g);
    gl::DirtyState dirty2 = gl::DirtyState::Clean;
    PodSet s2(&lb2, &dirty2);
    // The instrumented reload path: bulkAppendBytes -> rebuildIndex.
    PodSet::KeysView(&s2).bulkAppendBytes(sk.data(), sKeys.size());
    ASSERT_EQ(s2.count(), N);

    ASSERT_EQ(st.indexRebuildKeys.load(), static_cast<int64_t>(N));
    ASSERT_TRUE(st.indexRebuildNs.load() > 0);
}

TEST(deload_stats, skip_clean_counted_on_resident_redump) {
    gl::DeloadStats& st = gl::deloadStats();
    st.reset();
    const std::string dir = freshStatsDir("skip_clean");

    gl::Memory m;
    m.setExprKey("(deload_stats_skip_clean_lb)");
    for (int16_t i = 0; i < 8; ++i)
        m.intEncodedStatements.push_back(statsExpr(i));

    // First dump: full canonical write (LB stays resident — dump does not
    // release). Second dump: RAM == disk (Clean) with a base on disk, so the
    // skip-clean branch fires without writing.
    m.dumpStaticContainers(dir);
    ASSERT_EQ(st.fullDumpCount.load(), 1);
    ASSERT_EQ(st.evictSkippedClean.load(), 0);
    ASSERT_TRUE(m.lbMemory.manager.resident());

    m.dumpStaticContainers(dir);
    ASSERT_EQ(st.fullDumpCount.load(), 1);          // no new full dump
    ASSERT_EQ(st.evictSkippedClean.load(), 1);      // counted the skip
}

TEST(deload_stats, phase_window_delta_math) {
    gl::DeloadStats& st = gl::deloadStats();
    st.reset();

    // Pre-existing traffic outside the window must NOT enter the delta.
    st.recordV3Dump(999, 1);
    st.recordWorkerClaimWait(2, 5);

    st.beginPhaseWindow(2);
    st.recordV3Dump(1000, 400);
    st.recordV3Load(2000, 600);
    st.recordWorkerSelfLoad(300);
    st.recordWorkerInlineEvict(50);
    st.recordWorkerClaimWait(2, 700);
    const gl::PhaseDelta d = st.endPhaseWindow();

    ASSERT_EQ(d.phase, 2);
    ASSERT_EQ(d.outBytes, 1000);
    ASSERT_EQ(d.outNs, 400);
    ASSERT_EQ(d.inBytes, 2000);
    ASSERT_EQ(d.inNs, 600);
    ASSERT_EQ(d.selfLoads, 1);
    ASSERT_EQ(d.inlineEvicts, 1);
    ASSERT_EQ(d.workerWaitNs, 700);      // window-local, excludes the pre-5
    ASSERT_TRUE(d.wallSeconds >= 0.0);
}

TEST(deload_stats, empty_phase_window_returns_zero_delta) {
    gl::DeloadStats& st = gl::deloadStats();
    st.reset();
    st.beginPhaseWindow(1);
    const gl::PhaseDelta d = st.endPhaseWindow();   // no traffic -> no print
    ASSERT_EQ(d.phase, 1);
    ASSERT_EQ(d.outBytes, 0);
    ASSERT_EQ(d.inBytes, 0);
    ASSERT_EQ(d.selfLoads, 0);
}

TEST(deload_stats, reports_render_and_histogram_guard) {
    gl::DeloadStats& st = gl::deloadStats();
    st.reset();
    st.recordV3Dump(4096, 800);
    st.recordV3Load(8192, 1200);
    st.recordEviction(3, 42);

    // Smoke: the summary block renders without touching any counter's value.
    st.reportSummary();
    ASSERT_EQ(st.v3DumpCount.load(), 1);

    // The histogram is once-per-batch: pending before, printed after.
    ASSERT_TRUE(st.lbHistogramPending());
    std::vector<int64_t> held{ 10, 1, 7, 3, 20, 2 };
    st.reportLbHistogram(6, held);
    ASSERT_FALSE(st.lbHistogramPending());
    // The input is sorted in place (max is the last element).
    ASSERT_EQ(held.back(), 20);
}
