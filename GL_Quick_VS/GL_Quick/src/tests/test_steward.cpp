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
/// @brief Unit tests for the `MemorySteward` background custodian
///        (`memory_infra/steward.*`).
///
/// @details
/// Pins the barrier protocol: install does not start work, wake releases
/// exactly the installed list, quiesce blocks until the drain completes,
/// `drainedSinceInstall` reports the handoff state, and the grant
/// trigger's crossing callback is a valid wake source. LBs here are
/// bare `Memory` objects bound to the harness-initialized process pool;
/// dump directories land under `.debug/test_steward_*` (gitignored,
/// recreated per run).

#include "test_harness.hpp"

#include "../memory.hpp"
#include "../memory_infra/deload_stats.hpp"
#include "../memory_infra/steward.hpp"

#include <atomic>
#include <deque>
#include <filesystem>
#include <string>
#include <vector>

namespace {

    const gl::StaticMemoryConfig kStewardTestCfg{ 1 << 20, 1 << 18 };

    /// Minimal patterned row; value-initialized so the slots beyond the
    /// written fields are deterministic zero.
    gl::IntEncodedExpr stewardExpr(int16_t seed) {
        gl::IntEncodedExpr e{};
        e.nameId = seed;
        e.originalId = static_cast<int16_t>(seed + 1);
        e.validityId = 1;
        e.arity = 1;
        e.maxIteration = -1;
        return e;
    }

    /// Fresh per-test dump directory under .debug (name carries
    /// "deload" so the purge-safety assert holds if ever applied).
    std::string freshStewardDir(const std::string& name) {
        const std::filesystem::path dir =
            std::filesystem::path(".debug") / name;
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
        std::filesystem::create_directories(dir, ec);
        return dir.string();
    }

}

TEST(steward, start_stop_lifecycle_and_idle_quiesce) {
    gl::MemorySteward s;
    s.start();
    s.quiesce();                       // idle — immediate return
    ASSERT_FALSE(s.drainedSinceInstall());
    s.stop();
    s.start();                         // restartable after a clean stop
    s.quiesce();
    s.stop();
    s.stop();                          // defined no-op when not running
}

TEST(steward, install_wake_drains_in_order_and_reports) {
    const std::string dir =
        freshStewardDir("test_steward_deload_drain");
    gl::Memory a;
    a.setExprKey("(test_steward_lb_a)");
    gl::Memory b;
    b.setExprKey("(test_steward_lb_b)");
    for (int16_t i = 0; i < 40; ++i)
        a.intEncodedStatements.push_back(stewardExpr(i));
    for (int16_t i = 0; i < 7; ++i)
        b.intLocalEncodedStatements.push_back(stewardExpr(i));

    gl::MemorySteward s;
    s.start();
    s.installDischargeWork({ &a, &b }, dir);
    ASSERT_FALSE(s.drainedSinceInstall());
    s.wake();
    s.quiesce();
    ASSERT_TRUE(s.drainedSinceInstall());
    ASSERT_FALSE(a.lbMemory.manager.resident());
    ASSERT_FALSE(b.lbMemory.manager.resident());
    ASSERT_FALSE(a.deloadFiles.empty());
    ASSERT_FALSE(b.deloadFiles.empty());
    s.stop();

    // The image the steward wrote is a faithful canonical dump — the
    // normal reload path restores the exact content.
    a.ensureLoaded(dir);
    ASSERT_EQ(a.intEncodedStatements.size(), 40);
    ASSERT_EQ(a.intEncodedStatements[39].nameId,
              static_cast<int16_t>(39));
}

TEST(steward, install_without_wake_keeps_lbs_resident) {
    const std::string dir =
        freshStewardDir("test_steward_deload_idle");
    gl::Memory a;
    a.setExprKey("(test_steward_lb_c)");
    a.intEncodedStatements.push_back(stewardExpr(3));

    gl::MemorySteward s;
    s.start();
    s.installDischargeWork({ &a }, dir);
    s.quiesce();                       // install alone starts nothing
    ASSERT_TRUE(a.lbMemory.manager.resident());
    ASSERT_FALSE(s.drainedSinceInstall());
    s.wake();                          // the below-watermark idle path
    s.quiesce();                       // ends when a later wake arrives
    ASSERT_FALSE(a.lbMemory.manager.resident());
    ASSERT_TRUE(s.drainedSinceInstall());
    s.stop();
}

TEST(steward, eviction_pass_claims_planned_and_skips_worker_owned) {
    const std::string dir =
        freshStewardDir("test_steward_deload_evict");
    gl::Memory planned;
    planned.setExprKey("(test_steward_lb_e)");
    planned.intEncodedStatements.push_back(stewardExpr(5));
    planned.stewardClaim.store(
        static_cast<uint8_t>(gl::Memory::StewardClaim::Planned));
    gl::Memory workerOwned;
    workerOwned.setExprKey("(test_steward_lb_f)");
    workerOwned.intEncodedStatements.push_back(stewardExpr(6));
    workerOwned.stewardClaim.store(
        static_cast<uint8_t>(gl::Memory::StewardClaim::WorkerOwned));

    gl::MemorySteward s;
    s.start();
    s.installEvictionWork({ &planned, &workerOwned }, dir);
    s.wake();
    s.quiesce();
    s.stop();

    // The Planned victim was claimed, dumped, released, and reports
    // Dumped; the WorkerOwned one was skipped untouched (the worker's
    // self-service owns its dump).
    ASSERT_EQ(planned.stewardClaim.load(),
              static_cast<uint8_t>(gl::Memory::StewardClaim::Dumped));
    ASSERT_FALSE(planned.lbMemory.manager.resident());
    ASSERT_FALSE(planned.deloadFiles.empty());
    ASSERT_EQ(workerOwned.stewardClaim.load(),
              static_cast<uint8_t>(
                  gl::Memory::StewardClaim::WorkerOwned));
    ASSERT_TRUE(workerOwned.lbMemory.manager.resident());
    ASSERT_TRUE(workerOwned.deloadFiles.empty());

    // The worker-side completion of the steward path: claim the Dumped
    // victim and reload it — the round-trip restores the content.
    uint8_t expected =
        static_cast<uint8_t>(gl::Memory::StewardClaim::Dumped);
    ASSERT_TRUE(planned.stewardClaim.compare_exchange_strong(
        expected,
        static_cast<uint8_t>(gl::Memory::StewardClaim::WorkerOwned)));
    planned.ensureLoaded(dir);
    ASSERT_EQ(planned.intEncodedStatements.size(), 1);
    ASSERT_EQ(planned.intEncodedStatements[0].nameId,
              static_cast<int16_t>(5));
}

TEST(steward, discharge_protocol_captures_empties_and_returns_blocks) {
    gl::Memory lb;
    // Compaction scratch on an INDEPENDENT pool (production: the never-deloaded
    // LB-body pool, not the deloadable pool the discharge compacts).
    gl::GlobalMemoryManager gScr;
    gScr.init(kStewardTestCfg);
    gl::LbArena scratch(&gScr);
    lb.setExprKey("(test_steward_lb_i)");
    for (int16_t i = 0; i < 100; ++i)
        lb.intEncodedStatements.push_back(stewardExpr(i));
    for (int16_t i = 0; i < 50; ++i)
        lb.intLocalEncodedStatements.push_back(stewardExpr(i));
    ASSERT_TRUE(lb.lbMemory.manager.blocksHeld() > 0);
    lb.isActive = false;
    lb.dischargeStatementContent(scratch);
    // Flagged, resident, empty, zero blocks — the protocol's contract: the
    // copying compaction in reshuffle reclaims the emptied statement storage,
    // and this LB has no cold-string survivors, so every block returns.
    ASSERT_TRUE(lb.dischargedForever);
    ASSERT_TRUE(lb.lbMemory.manager.resident());
    ASSERT_EQ(lb.lbMemory.manager.blocksHeld(), static_cast<int64_t>(0));
    ASSERT_EQ(lb.intEncodedStatements.size(), 0);
    ASSERT_EQ(lb.intLocalEncodedStatements.size(), 0);
    // The captured record answers exactly what the registry held:
    // stewardExpr(i) carries originalId = i + 1, validityId = 1.
    ASSERT_EQ(lb.dischargedRegistryKeys.size(),
              static_cast<std::size_t>(100));
    ASSERT_TRUE(lb.dischargedRegistryKeys.count(
                    gl::packStatementKey(static_cast<int16_t>(51),
                                         static_cast<int16_t>(1))) > 0);
    ASSERT_FALSE(lb.dischargedRegistryKeys.count(
                     gl::packStatementKey(static_cast<int16_t>(101),
                                          static_cast<int16_t>(1))) > 0);
}

TEST(steward, forward_caching_window_prefetches_dumped) {
    const std::string dir =
        freshStewardDir("test_steward_deload_window");
    // A steward-dumped eviction victim, cold, awaiting its slot.
    gl::Memory dumped;
    dumped.setExprKey("(test_steward_lb_g)");
    dumped.intEncodedStatements.push_back(stewardExpr(7));
    dumped.deloadStaticContainers(dir);
    dumped.stewardClaim.store(
        static_cast<uint8_t>(gl::Memory::StewardClaim::Dumped));

    std::vector<gl::Memory*> order = { &dumped };
    std::atomic<std::size_t> cursor{ 0 };
    gl::MemorySteward s;
    s.start();
    // workers = 1: the unified prefetch region starts AT the cursor and spans
    // `workers + kLookaheadWorkerMultiple * workers` indices, so index 0 is
    // covered. (workers = 0 collapses the window to empty — the degenerate
    // case, never used in a real phase.)
    s.beginPhaseWindow(/*phase=*/1, &cursor, &order, 1, dir);
    // Bounded wait on the atomic claim: the scan pass prefetch-reloads the
    // dumped victim ahead of its slot. (The bump arena reclaims interior
    // holes through the copying compaction, not a steward reshuffle, so the
    // idle-LB reshuffle path is gone — see LbMemory::reshuffle.)
    for (int spin = 0; spin < 20000; ++spin) {
        if (dumped.stewardClaim.load()
            == static_cast<uint8_t>(gl::Memory::StewardClaim::Idle))
            break;
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    s.endPhaseWindow();
    s.quiesce();
    s.stop();

    ASSERT_EQ(dumped.stewardClaim.load(),
              static_cast<uint8_t>(gl::Memory::StewardClaim::Idle));
    ASSERT_TRUE(dumped.lbMemory.manager.resident());
    ASSERT_EQ(dumped.intEncodedStatements.size(), 1);
    ASSERT_EQ(dumped.intEncodedStatements[0].nameId,
              static_cast<int16_t>(7));
}

TEST(steward, grant_trigger_crossing_is_a_wake_source) {
    const std::string dir =
        freshStewardDir("test_steward_deload_trigger");
    gl::Memory a;
    a.setExprKey("(test_steward_lb_d)");
    a.intEncodedStatements.push_back(stewardExpr(9));

    gl::MemorySteward s;
    s.start();
    s.installDischargeWork({ &a }, dir);

    gl::GlobalMemoryManager g;
    g.init(kStewardTestCfg);
    g.armGrantTrigger(2, [&s]() { s.wake(); });
    char* b1 = g.acquireBlock();       // below threshold — no wake
    ASSERT_FALSE(s.drainedSinceInstall());
    char* b2 = g.acquireBlock();       // crossing grant wakes the steward
    s.quiesce();
    ASSERT_TRUE(s.drainedSinceInstall());
    ASSERT_FALSE(a.lbMemory.manager.resident());
    g.releaseBlock(b1);
    g.releaseBlock(b2);
    s.stop();
}

TEST(steward, stuck_deadline_predicate) {
    // The stuck tripwire: a freshly-started wait is not stuck; a wait whose
    // start stamp is older than kStuckSeconds is. A worker on a Busy claim
    // stamps the start once and polls this each spin, asserting on a true
    // result instead of spinning forever (Rule 19 — no silent freeze).
    const auto now = std::chrono::steady_clock::now();
    ASSERT_FALSE(gl::steward::stuckDeadlineExceeded(now));
    const auto pastDeadline =
        now - std::chrono::seconds(gl::steward::kStuckSeconds + 1);
    ASSERT_TRUE(gl::steward::stuckDeadlineExceeded(pastDeadline));
    // Exactly at the boundary counts as stuck (the comparison is >=).
    const auto atDeadline =
        now - std::chrono::seconds(gl::steward::kStuckSeconds);
    ASSERT_TRUE(gl::steward::stuckDeadlineExceeded(atDeadline));
}

TEST(steward, pick_victim_two_tier_floor_preferred_then_floorless) {
    // The survival fix's two-tier victim selection: tier 1 is the
    // floor-preferred Belady scan; ONLY when no above-floor victim exists
    // does the floorless second scan run (both tiers are defined results —
    // survival trumps per-operation efficiency). Window [keepLo=3, keepHi=6),
    // cursor 4; out-of-window scan order (farthest first): 2, 1, 0, wrap 7, 6.
    gl::MemorySteward s;
    auto fill = [](gl::Memory& m, const std::string& key, int count) {
        m.setExprKey(key.c_str());
        m.isActive = true;
        for (int i = 0; i < count; ++i)
            m.intEncodedStatements.push_back(
                stewardExpr(static_cast<int16_t>(i)));
    };
    std::deque<gl::Memory> lbs;
    for (int i = 0; i < 8; ++i) {
        lbs.emplace_back();
        fill(lbs.back(), "(tt_" + std::to_string(i) + ")", 40);   // tiny
    }
    std::vector<gl::Memory*> order;
    for (int i = 0; i < 8; ++i) order.push_back(&lbs[i]);

    // A big LB at index 0 — NEARER to the cursor (smaller cyclic next-use
    // distance) than the farthest tiny LB at index 2.
    gl::Memory big;
    fill(big, "(tt_big)", 8000);
    order[0] = &big;
    const int64_t tinyHeld = lbs[2].lbMemory.manager.blocksHeld();
    const int64_t bigHeld = big.lbMemory.manager.blocksHeld();
    ASSERT_TRUE(bigHeld > tinyHeld);          // the floor is meaningful
    // Floor above the tiny population: tier 1 sees only the big LB.
    s.setMinEvictBlocksForTest(bigHeld);

    // (a) FLOOR-PREFERRED: with an above-floor victim available, tier 1 wins
    // even though tiny lbs[2] is FARTHER behind the cursor — the size-floor
    // hygiene beats pure Belady distance while it can.
    ASSERT_TRUE(s.pickVictimTwoTier(&order, 4, 3, 6) == &big);

    // (b) FLOORLESS FALLBACK: no above-floor victim left (big in flight) ->
    // the second scan runs with a one-block floor and returns the FARTHEST
    // tiny LB. This is the survival branch: the post-pager 4 GiB run died
    // because a floor-only drain found no victim in the median-2-block
    // early-batch population while grants continued.
    big.stewardClaim.store(
        static_cast<uint8_t>(gl::Memory::StewardClaim::WorkerOwned));
    ASSERT_TRUE(s.pickVictimTwoTier(&order, 4, 3, 6) == &lbs[2]);

    // (c) NOTHING AT ALL: every out-of-window LB in flight -> nullptr in both
    // tiers (the defined working-set-fits result, never a failure).
    for (int i : { 1, 2, 6, 7 })
        lbs[i].stewardClaim.store(
            static_cast<uint8_t>(gl::Memory::StewardClaim::WorkerOwned));
    ASSERT_TRUE(s.pickVictimTwoTier(&order, 4, 3, 6) == nullptr);
}

TEST(steward, claim_and_load_for_work_handshake) {
    // The unified per-worker handshake (all three phases). With no window
    // open and the process pool well under the watermark, evictOneForReload
    // is inert, so this exercises the claim + reload transitions: a cold
    // (Dumped) LB is reloaded and claimed; a resident Idle LB is claimed with
    // no reload; a WorkerOwned LB (a split sibling) returns untouched.
    const std::string dir = freshStewardDir("test_steward_claim_load");
    gl::MemorySteward s;

    gl::Memory cold;
    cold.setExprKey("(claim_cold)");
    cold.intEncodedStatements.push_back(stewardExpr(21));
    cold.deloadStaticContainers(dir);   // -> not resident
    cold.stewardClaim.store(
        static_cast<uint8_t>(gl::Memory::StewardClaim::Dumped));
    s.claimAndLoadForWork(cold, /*phase=*/1, dir);
    ASSERT_EQ(cold.stewardClaim.load(),
              static_cast<uint8_t>(gl::Memory::StewardClaim::WorkerOwned));
    ASSERT_TRUE(cold.lbMemory.manager.resident());
    ASSERT_EQ(cold.intEncodedStatements.size(), 1);
    ASSERT_EQ(cold.intEncodedStatements[0].nameId,
              static_cast<int16_t>(21));

    gl::Memory warm;
    warm.setExprKey("(claim_warm)");
    warm.intEncodedStatements.push_back(stewardExpr(22));
    ASSERT_TRUE(warm.lbMemory.manager.resident());
    s.claimAndLoadForWork(warm, /*phase=*/1, dir);
    ASSERT_EQ(warm.stewardClaim.load(),
              static_cast<uint8_t>(gl::Memory::StewardClaim::WorkerOwned));
    ASSERT_EQ(warm.intEncodedStatements.size(), 1);

    // Already WorkerOwned (a sibling part owns it) -> idempotent no-op.
    s.claimAndLoadForWork(warm, /*phase=*/1, dir);
    ASSERT_EQ(warm.stewardClaim.load(),
              static_cast<uint8_t>(gl::Memory::StewardClaim::WorkerOwned));
}

TEST(steward, unified_window_prefetched_lb_survives_same_pass) {
    // Window unification: the prefetch region and the keep window share an
    // end, so a freshly prefetched LB inside the window is NOT planned for
    // eviction by the same maintainWorkingSet pass. A small private pool held
    // empty of free blocks breaches the reserve and forces the drain plan;
    // planning + execution are driven synchronously (the planner thread and
    // the executors run the same two calls).
    const std::string dir = freshStewardDir("test_steward_unified_window");

    // Pressure source: a 4-block pool, all four held -> free 0.
    gl::GlobalMemoryManager pressure;
    pressure.init(kStewardTestCfg);            // 1 MiB / 256 KiB = 4 blocks
    std::vector<char*> held;
    for (int i = 0; i < 4; ++i) held.push_back(pressure.acquireBlock());
    ASSERT_TRUE(pressure.blocksInUse() * 4 > pressure.totalBlocks() * 3);

    const unsigned workers = 3;
    const std::size_t cur = 5;
    // Unified keep window [cur-workers, cur+workers+kLookaheadWorkerMultiple*
    // workers) = [2, keepHi); the prefetch region [cur, keepHi) shares its END
    // with the keep window, so a freshly prefetched LB inside the window is NOT
    // planned for eviction the same pass — the window asymmetry the unification
    // fixes (the old split code prefetched an in-window index then evicted it
    // when workers exceeded the fixed lookahead). The geometry is computed from
    // the constant so the test tracks the lookahead tuning; victims sit OUTSIDE
    // the window (index < cur-workers, or index >= keepHi).
    const std::size_t keepHi =
        cur + workers + gl::steward::kLookaheadWorkerMultiple * workers;
    const std::size_t dumpedIdx = 10;    // inside [cur, keepHi)
    const std::size_t n = keepHi + 4;    // indices [keepHi, n) are ahead victims

    std::deque<gl::Memory> lbs;
    for (std::size_t i = 0; i < n; ++i) {
        lbs.emplace_back();
        lbs.back().setExprKey(("(uw_" + std::to_string(i) + ")").c_str());
        lbs.back().intEncodedStatements.push_back(
            stewardExpr(static_cast<int16_t>(i)));
    }
    // The prefetch target: dumped + cold, at an in-window index.
    lbs[dumpedIdx].deloadStaticContainers(dir);
    lbs[dumpedIdx].stewardClaim.store(
        static_cast<uint8_t>(gl::Memory::StewardClaim::Dumped));

    std::vector<gl::Memory*> order;
    for (std::size_t i = 0; i < n; ++i) order.push_back(&lbs[i]);

    std::atomic<std::size_t> cursor{ cur };
    gl::MemorySteward s;
    s.setPressurePoolForTest(&pressure);
    // The small fixture LBs are below the production 4-block floor; lower it so
    // they qualify as victims (the floor itself is covered by the direct
    // pickVictimBehindCursor test). The tiny pool sits below the reserve
    // either way, so the drain plans evictions.
    s.setMinEvictBlocksForTest(1);
    // ONE synchronous planning pass + a synchronous ring drain (production
    // executes on the executor pool; the test seam applies the same tasks
    // deterministically). The window must be open: tasks bind to its
    // generation and re-validate at execution.
    s.beginPhaseWindow(/*phase=*/1, &cursor, &order, workers, dir);
    s.maintainWorkingSet(&cursor, &order, workers);
    s.drainIoRingsForTest();
    s.endPhaseWindow();
    s.setPressurePoolForTest(nullptr);

    // The prefetched LB (inside the unified window) was reloaded and NOT
    // drained: resident + Idle, content intact.
    ASSERT_TRUE(lbs[dumpedIdx].lbMemory.manager.resident());
    ASSERT_EQ(lbs[dumpedIdx].stewardClaim.load(),
              static_cast<uint8_t>(gl::Memory::StewardClaim::Idle));
    ASSERT_EQ(lbs[dumpedIdx].intEncodedStatements.size(), 1);

    // The drain really ran this pass: out-of-window victims were mass-evicted.
    ASSERT_FALSE(lbs[0].lbMemory.manager.resident());
    ASSERT_FALSE(lbs[n - 1].lbMemory.manager.resident());
    ASSERT_EQ(lbs[0].stewardClaim.load(),
              static_cast<uint8_t>(gl::Memory::StewardClaim::Dumped));

    // An in-window LB that was not the prefetch target stayed resident.
    ASSERT_TRUE(lbs[7].lbMemory.manager.resident());

    for (char* b : held) pressure.releaseBlock(b);
}

TEST(steward, pick_victim_behind_cursor_selects_and_skips) {
    // The reserve-target drain's victim selector — a pure function (no running
    // thread, no pool pressure). 8 small LBs; window [keepLo=3, keepHi=6),
    // cursor 4. Out-of-window indices in farthest-next-use order
    // (cyclic d = (i-4) mod 8): 2 (d=6), 1 (d=5), 0 (d=4), then WRAP past the
    // bottom to 7 (d=3), 6 (d=2).
    gl::MemorySteward s;
    auto fill = [](gl::Memory& m, const std::string& key, int count) {
        m.setExprKey(key.c_str());
        for (int i = 0; i < count; ++i)
            m.intEncodedStatements.push_back(
                stewardExpr(static_cast<int16_t>(i)));
    };
    auto claim = [](gl::Memory& m, gl::Memory::StewardClaim c) {
        m.stewardClaim.store(static_cast<uint8_t>(c));
    };

    std::deque<gl::Memory> lbs;
    for (int i = 0; i < 8; ++i) {
        lbs.emplace_back();
        fill(lbs.back(), "(pv_" + std::to_string(i) + ")", 40);
    }
    std::vector<gl::Memory*> order;
    for (int i = 0; i < 8; ++i) order.push_back(&lbs[i]);

    // (a) BACKWARD / farthest-behind: the victim is the farthest-next-use
    // eligible LB behind the cursor — index 2, not the nearer 0/1, not the
    // wrap entries 6/7.
    ASSERT_TRUE(s.pickVictimBehindCursor(&order, 4, 3, 6, 1) == &lbs[2]);

    // (b) WRAP: with 2,1,0 in flight (WorkerOwned), the backward scan wraps
    // past the bottom to the top and returns index 7 (the next farthest).
    claim(lbs[2], gl::Memory::StewardClaim::WorkerOwned);
    claim(lbs[1], gl::Memory::StewardClaim::WorkerOwned);
    claim(lbs[0], gl::Memory::StewardClaim::WorkerOwned);
    ASSERT_TRUE(s.pickVictimBehindCursor(&order, 4, 3, 6, 1) == &lbs[7]);

    // (c) CLAIMED skip: index 7 also in flight -> falls through to 6.
    claim(lbs[7], gl::Memory::StewardClaim::WorkerOwned);
    ASSERT_TRUE(s.pickVictimBehindCursor(&order, 4, 3, 6, 1) == &lbs[6]);

    // (d) WINDOW EXCLUSION: every out-of-window LB in flight -> nullptr; an
    // in-window LB (3,4,5) is never returned.
    claim(lbs[6], gl::Memory::StewardClaim::WorkerOwned);
    ASSERT_TRUE(s.pickVictimBehindCursor(&order, 4, 3, 6, 1) == nullptr);
    for (int i : { 0, 1, 2, 6, 7 })
        claim(lbs[i], gl::Memory::StewardClaim::Idle);

    // (e) INACTIVE / SPLIT skips: index 2 inactive, index 1 split -> the scan
    // falls through to index 0.
    lbs[2].isActive = false;
    lbs[1].numberOfParts = 4;
    ASSERT_TRUE(s.pickVictimBehindCursor(&order, 4, 3, 6, 1) == &lbs[0]);
    lbs[2].isActive = true;
    lbs[1].numberOfParts = 1;

    // (f) FLOOR skip: a below-floor LB at the farthest position is skipped in
    // favour of a larger one further along the scan. A big LB (several blocks)
    // sits at index 1; the floor is set above index 2's block count.
    gl::Memory big;
    fill(big, "(pv_big)", 8000);
    order[1] = &big;
    const int64_t smallHeld = lbs[2].lbMemory.manager.blocksHeld();
    const int64_t bigHeld = big.lbMemory.manager.blocksHeld();
    ASSERT_TRUE(bigHeld > smallHeld);          // the floor is meaningful
    // Floor = bigHeld: index 2 (smallHeld < bigHeld) is skipped by the floor;
    // big at index 1 (blocksHeld == bigHeld) qualifies and is the first hit.
    ASSERT_TRUE(
        s.pickVictimBehindCursor(&order, 4, 3, 6, bigHeld) == &big);
}

TEST(steward, thrash_regime_reserve_target_evicts_the_deficit_not_the_set) {
    // The campaign's core regression guard. A fixture of N uniform LBs on
    // held-blocks-constrained free space; the reserve is set to the current
    // free plus a fixed deficit of 3 LBs. Sweeping the cursor over the order
    // twice must (a) always pick the farthest-behind eligible victims, (b)
    // never dump an in-window LB, and (c) evict only the DEFICIT — never the
    // whole working set (the old mass-evict would dump every out-of-window LB
    // each pass; the reserve model evicts at most the deficit per pass).
    const std::string dir = freshStewardDir("test_steward_thrash");
    const std::size_t n = 20;
    const unsigned workers = 1;              // window [cur-1, cur+3), size 4
    const int deficitLbs = 3;

    std::deque<gl::Memory> lbs;
    for (std::size_t i = 0; i < n; ++i) {
        lbs.emplace_back();
        lbs.back().setExprKey(("(th_" + std::to_string(i) + ")").c_str());
        for (int k = 0; k < 200; ++k)
            lbs.back().intEncodedStatements.push_back(
                stewardExpr(static_cast<int16_t>(k)));
    }
    std::vector<gl::Memory*> order;
    for (std::size_t i = 0; i < n; ++i) order.push_back(&lbs[i]);

    // Uniform content -> uniform block count; the deficit is exactly
    // deficitLbs whole LBs.
    const int64_t b = lbs[0].lbMemory.manager.blocksHeld();
    ASSERT_TRUE(b >= 1);
    for (std::size_t i = 0; i < n; ++i)
        ASSERT_EQ(lbs[i].lbMemory.manager.blocksHeld(), b);

    gl::MemorySteward s;
    const int64_t total = gl::staticMemory().totalBlocks();
    const int64_t free0 = total - gl::staticMemory().blocksInUse();
    s.setReserveBlocksForTest(free0 + deficitLbs * b);
    // The 1-block fixture LBs are below the production 4-block floor; lower it
    // so the deficit is exactly deficitLbs whole LBs (the floor is covered by
    // the direct pickVictimBehindCursor test).
    s.setMinEvictBlocksForTest(1);

    auto dumpedCount = [&]() {
        int c = 0;
        for (std::size_t i = 0; i < n; ++i)
            if (!lbs[i].lbMemory.manager.resident()) ++c;
        return c;
    };

    // Plan + drain one pass: open the window (tasks bind to its generation),
    // run one planning pass, execute the plan synchronously, close.
    // A lambda local to a covered test driver (Rule 18 exempt).
    std::atomic<std::size_t> cursor{ 0 };
    auto onePass = [&](std::size_t c) {
        cursor.store(c);
        s.beginPhaseWindow(/*phase=*/1, &cursor, &order, workers, dir);
        s.maintainWorkingSet(&cursor, &order, workers);
        s.drainIoRingsForTest();
        s.endPhaseWindow();
    };

    // First pass at cursor 0: the drain must evict EXACTLY the 3 farthest-behind
    // LBs (the highest indices, cyclic distance largest from cursor 0) — the
    // deficit — and nothing else.
    const int64_t evict0 = gl::deloadStats().evictCount.load();
    onePass(0);
    ASSERT_EQ(gl::deloadStats().evictCount.load() - evict0,
              static_cast<int64_t>(deficitLbs));
    ASSERT_FALSE(lbs[n - 1].lbMemory.manager.resident());
    ASSERT_FALSE(lbs[n - 2].lbMemory.manager.resident());
    ASSERT_FALSE(lbs[n - 3].lbMemory.manager.resident());
    ASSERT_TRUE(lbs[n - 4].lbMemory.manager.resident());   // 4th-farthest kept
    ASSERT_TRUE(lbs[0].lbMemory.manager.resident());       // window kept
    ASSERT_TRUE(lbs[1].lbMemory.manager.resident());
    ASSERT_EQ(dumpedCount(), deficitLbs);

    // Sweep the cursor over the order twice more. Each pass: no LB inside that
    // pass's window is ever dumped (b), and no single pass evicts more than the
    // deficit (c) — the reserve is the steady-state target, so evictions track
    // the (bounded) deficit, never the ~16 out-of-window eligible LBs a
    // mass-evict pass would dump.
    int64_t maxPerPass = deficitLbs;         // the first pass covered the deficit
    for (int lap = 0; lap < 2; ++lap) {
        for (std::size_t c = 0; c < n; ++c) {
            const int64_t before = gl::deloadStats().evictCount.load();
            onePass(c);
            const int64_t perPass = gl::deloadStats().evictCount.load() - before;
            if (perPass > maxPerPass) maxPerPass = perPass;
            // (b) the steward keeps the prefetch region [cursor, keepHi)
            // resident (it reloads any Dumped LB there and never drains a window
            // LB). The in-flight margin [keepLo, cursor) is the workers' job to
            // reload — not simulated here — so it is not asserted.
            const std::size_t keepHi = c + workers
                + gl::steward::kLookaheadWorkerMultiple * workers;
            for (std::size_t i = c; i < keepHi && i < n; ++i)
                ASSERT_TRUE(lbs[i].lbMemory.manager.resident());
        }
    }
    // (c) NO pass ever evicted more than the deficit — the discriminator vs the
    // old mass-evict (which would dump ~16 LBs in the very first pass).
    ASSERT_TRUE(maxPerPass <= static_cast<int64_t>(deficitLbs));
    // The pager also held the resident set near the working set + reserve, so
    // it never collapsed to "everything evicted": most LBs stay resident.
    ASSERT_TRUE(dumpedCount() < static_cast<int>(n) / 2);
}

TEST(steward, worker_reload_healthy_reserve_does_no_eviction) {
    // With a healthy reserve (the process pool's free blocks far above the
    // emergency floor), a worker reloading a cold LB does NO eviction — it
    // reloads straight into the reserve's free blocks; neither valve trigger
    // (emergency floor, reload fit) fires. emergencyEvictCount stays flat.
    const std::string dir = freshStewardDir("test_steward_worker_healthy");
    gl::MemorySteward s;
    gl::Memory cold;
    cold.setExprKey("(worker_healthy)");
    cold.intEncodedStatements.push_back(stewardExpr(31));
    cold.deloadStaticContainersRaw(dir);        // raw -> lastRawImageBytes > 0
    cold.stewardClaim.store(
        static_cast<uint8_t>(gl::Memory::StewardClaim::Dumped));

    const int64_t emg0 = gl::deloadStats().emergencyEvictCount.load();
    s.claimAndLoadForWork(cold, /*phase=*/1, dir);
    ASSERT_EQ(cold.stewardClaim.load(),
              static_cast<uint8_t>(gl::Memory::StewardClaim::WorkerOwned));
    ASSERT_TRUE(cold.lbMemory.manager.resident());
    ASSERT_EQ(cold.intEncodedStatements.size(), 1);
    ASSERT_EQ(cold.intEncodedStatements[0].nameId, static_cast<int16_t>(31));
    // Zero worker-path evictions on the healthy path.
    ASSERT_EQ(gl::deloadStats().emergencyEvictCount.load() - emg0,
              static_cast<int64_t>(0));
}

TEST(steward, worker_reload_forced_exhaustion_fires_fallback_exactly_once) {
    // The widened valve under a genuine squeeze: free blocks (0) sit below
    // BOTH triggers — the emergency floor AND this reload's need — yet
    // exactly ONE counted eviction fires per claim (the per-claim guard) and
    // the reload completes. The victim is the two-tier Belady choice behind
    // the cursor, not a biggest-first pick.
    const std::string dir = freshStewardDir("test_steward_worker_exhaust");
    gl::GlobalMemoryManager pressure;
    pressure.init(kStewardTestCfg);             // 4 blocks
    std::vector<char*> held;
    for (int i = 0; i < 4; ++i) held.push_back(pressure.acquireBlock());
    ASSERT_EQ(pressure.totalBlocks() - pressure.blocksInUse(),
              static_cast<int64_t>(0));

    // A resident, Idle, active, unsplit victim behind the window.
    gl::Memory victim;
    victim.setExprKey("(worker_exhaust_victim)");
    victim.isActive = true;
    victim.intEncodedStatements.push_back(stewardExpr(41));
    ASSERT_TRUE(victim.lbMemory.manager.resident());

    // The cold LB the worker must reload (raw-dumped -> lastRawImageBytes > 0).
    gl::Memory cold;
    cold.setExprKey("(worker_exhaust_cold)");
    cold.isActive = true;
    cold.intEncodedStatements.push_back(stewardExpr(42));
    cold.deloadStaticContainersRaw(dir);
    cold.stewardClaim.store(
        static_cast<uint8_t>(gl::Memory::StewardClaim::Dumped));

    // Window [keepLo=1, keepHi=5) at cursor 2 over {victim, filler}: index 0
    // is outside the kept range, so the two-tier Belady scan (backward from
    // keepLo - 1 = 0) selects the victim; the in-window filler is protected.
    gl::Memory filler;
    filler.setExprKey("(worker_exhaust_filler)");
    filler.isActive = true;
    filler.intEncodedStatements.push_back(stewardExpr(43));
    std::vector<gl::Memory*> order = { &victim, &filler };
    std::atomic<std::size_t> cursor{ 2 };
    gl::MemorySteward s;
    s.setPressurePoolForTest(&pressure);
    s.beginPhaseWindow(/*phase=*/1, &cursor, &order, /*workers=*/1, dir);

    const int64_t emg0 = gl::deloadStats().emergencyEvictCount.load();
    s.claimAndLoadForWork(cold, /*phase=*/1, dir);

    // The valve fired exactly once; the reload completed.
    ASSERT_EQ(gl::deloadStats().emergencyEvictCount.load() - emg0,
              static_cast<int64_t>(1));
    ASSERT_EQ(cold.stewardClaim.load(),
              static_cast<uint8_t>(gl::Memory::StewardClaim::WorkerOwned));
    ASSERT_TRUE(cold.lbMemory.manager.resident());
    ASSERT_EQ(cold.intEncodedStatements.size(), 1);
    // The victim was the one evicted to make room; the in-window filler kept.
    ASSERT_FALSE(victim.lbMemory.manager.resident());
    ASSERT_EQ(victim.stewardClaim.load(),
              static_cast<uint8_t>(gl::Memory::StewardClaim::Dumped));
    ASSERT_TRUE(filler.lbMemory.manager.resident());

    s.endPhaseWindow();
    s.setPressurePoolForTest(nullptr);
    for (char* b : held) pressure.releaseBlock(b);
}

TEST(steward, worker_valve_fires_below_emergency_floor_on_pure_grant_path) {
    // The WIDENED WORKER PRESSURE VALVE: free blocks below the emergency
    // floor (reserve / kEmergencyFloorDivisor) fire ONE counted eviction on
    // EVERY claim — here a claim of an already-RESIDENT LB (the pure-grant
    // path: the worker's task will grant scratch / statement blocks but needs
    // no reload), which the old own-reload-does-not-fit condition never
    // covered. The post-pager 4 GiB failure came through exactly this path.
    const std::string dir = freshStewardDir("test_steward_valve_floor");
    gl::GlobalMemoryManager pressure;
    pressure.init(kStewardTestCfg);             // 4 blocks
    std::vector<char*> held;
    for (int i = 0; i < 4; ++i) held.push_back(pressure.acquireBlock());

    gl::Memory victim;
    victim.setExprKey("(valve_victim)");
    victim.isActive = true;
    victim.intEncodedStatements.push_back(stewardExpr(51));
    gl::Memory warm;
    warm.setExprKey("(valve_warm)");
    warm.isActive = true;
    warm.intEncodedStatements.push_back(stewardExpr(52));
    ASSERT_TRUE(warm.lbMemory.manager.resident());

    gl::MemorySteward s;
    s.setPressurePoolForTest(&pressure);

    // (a) NO WINDOW OPEN: the valve trips but has no victim source —
    // evictOneForReload returns false, nothing is counted, and the claim
    // completes normally (a defined result, not a failure).
    const int64_t emg0 = gl::deloadStats().emergencyEvictCount.load();
    s.claimAndLoadForWork(warm, /*phase=*/1, dir);
    ASSERT_EQ(gl::deloadStats().emergencyEvictCount.load() - emg0,
              static_cast<int64_t>(0));
    ASSERT_EQ(warm.stewardClaim.load(),
              static_cast<uint8_t>(gl::Memory::StewardClaim::WorkerOwned));
    warm.stewardClaim.store(
        static_cast<uint8_t>(gl::Memory::StewardClaim::Idle));

    // (b) WINDOW OPEN: the same pure-grant claim now evicts the
    // behind-cursor victim exactly once (counted) even though the warm LB
    // needs NO reload; the warm LB itself sits inside the kept window and is
    // never its own victim.
    std::vector<gl::Memory*> order = { &victim, &warm };
    std::atomic<std::size_t> cursor{ 2 };   // keepLo=1: victim at 0 evictable
    s.beginPhaseWindow(/*phase=*/1, &cursor, &order, /*workers=*/1, dir);
    s.claimAndLoadForWork(warm, /*phase=*/1, dir);
    ASSERT_EQ(gl::deloadStats().emergencyEvictCount.load() - emg0,
              static_cast<int64_t>(1));
    ASSERT_FALSE(victim.lbMemory.manager.resident());
    ASSERT_EQ(victim.stewardClaim.load(),
              static_cast<uint8_t>(gl::Memory::StewardClaim::Dumped));
    ASSERT_EQ(warm.stewardClaim.load(),
              static_cast<uint8_t>(gl::Memory::StewardClaim::WorkerOwned));
    ASSERT_TRUE(warm.lbMemory.manager.resident());
    s.endPhaseWindow();
    s.setPressurePoolForTest(nullptr);
    for (char* b : held) pressure.releaseBlock(b);
}

TEST(steward, io_thread_count_clamps_and_overrides) {
    // The executor-pool sizing policy: workers / kIoThreadsPerWorkers clamped
    // to [kIoThreadsMin, kIoThreadsMax]; a nonzero override forces the count
    // (the kIoThreadsOverride concurrency-isolation switch).
    ASSERT_EQ(gl::steward::ioThreadCountFor(0, 0), 2u);    // no hint -> floor
    ASSERT_EQ(gl::steward::ioThreadCountFor(8, 0), 2u);
    ASSERT_EQ(gl::steward::ioThreadCountFor(16, 0), 2u);
    ASSERT_EQ(gl::steward::ioThreadCountFor(32, 0), 4u);   // production
    ASSERT_EQ(gl::steward::ioThreadCountFor(64, 0), 8u);
    ASSERT_EQ(gl::steward::ioThreadCountFor(256, 0), 8u);  // ceiling
    ASSERT_EQ(gl::steward::ioThreadCountFor(32, 1), 1u);   // isolation switch
    ASSERT_EQ(gl::steward::ioThreadCountFor(0, 3), 3u);    // explicit force
}

TEST(steward, executor_lane_priority_flips_to_low_below_emergency_floor) {
    // Executors serve the HIGH (prefetch-load) lane first EXCEPT when free
    // blocks sit below the emergency floor — then the LOW (eviction) lane
    // goes first, so evictions never starve behind reloads with no blocks to
    // land in. Made observable through runOneIoTaskForTest, which pops with
    // the same rule the executors use.
    const std::string dir = freshStewardDir("test_steward_lane_flip");
    gl::GlobalMemoryManager pressure;
    pressure.init(kStewardTestCfg);             // 4 blocks
    std::vector<char*> held;
    for (int i = 0; i < 4; ++i) held.push_back(pressure.acquireBlock());

    // A Dumped LB INSIDE the prefetch region -> a HIGH load task; an Idle
    // victim BEHIND the window -> a LOW eviction task.
    gl::Memory victim;
    victim.setExprKey("(flip_victim)");
    victim.isActive = true;
    victim.intEncodedStatements.push_back(stewardExpr(61));
    gl::Memory filler;
    filler.setExprKey("(flip_filler)");
    filler.isActive = true;
    filler.intEncodedStatements.push_back(stewardExpr(62));
    gl::Memory ahead;
    ahead.setExprKey("(flip_ahead)");
    ahead.isActive = true;
    ahead.intEncodedStatements.push_back(stewardExpr(63));
    ahead.deloadStaticContainersRaw(dir);
    ahead.stewardClaim.store(
        static_cast<uint8_t>(gl::Memory::StewardClaim::Dumped));

    // Window [keepLo=1, keepHi=5) at cursor 2 over {victim, filler, ahead}:
    // index 0 behind (eviction candidate), index 2 in the prefetch region.
    std::vector<gl::Memory*> order = { &victim, &filler, &ahead };
    std::atomic<std::size_t> cursor{ 2 };
    gl::MemorySteward s;
    s.setPressurePoolForTest(&pressure);
    // Reserve 4 -> emergency floor 1 (floors scale together); free is 0, so
    // the plan wants evictions AND the pop rule sees an emergency.
    s.setReserveBlocksForTest(4);
    s.setMinEvictBlocksForTest(1);
    s.beginPhaseWindow(/*phase=*/1, &cursor, &order, /*workers=*/1, dir);
    s.maintainWorkingSet(&cursor, &order, /*workers=*/1);
    ASSERT_EQ(s.ioRingDepthForTest(true), static_cast<std::size_t>(1));
    ASSERT_EQ(s.ioRingDepthForTest(false), static_cast<std::size_t>(1));

    // free (0) < floor (1): the LOW lane goes first — the eviction runs.
    ASSERT_EQ(s.runOneIoTaskForTest(), 2);
    ASSERT_FALSE(victim.lbMemory.manager.resident());

    // Pressure relieved: free (4) >= floor (1) — HIGH first again.
    for (char* b : held) pressure.releaseBlock(b);
    held.clear();
    ASSERT_EQ(s.runOneIoTaskForTest(), 1);
    ASSERT_TRUE(ahead.lbMemory.manager.resident());
    ASSERT_EQ(ahead.stewardClaim.load(),
              static_cast<uint8_t>(gl::Memory::StewardClaim::Idle));
    ASSERT_EQ(s.runOneIoTaskForTest(), 0);   // both rings empty

    s.endPhaseWindow();
    s.setPressurePoolForTest(nullptr);
}

TEST(steward, planner_enqueue_dedups_drops_count_and_stale_tasks_drop) {
    // Three ring contracts in one fixture. (1) DEDUP: a second planning pass
    // over an unchanged window re-enqueues nothing (ring depth unchanged).
    // (2) DROP: a victim population beyond kIoRingCapacity fills the LOW
    // ring; the first overflow ends the plan and is counted (the designed
    // load-shedding — the planner regenerates next pass). (3) STALE DROP:
    // closing the window invalidates the queued generation, so a drain
    // executes NOTHING — no eviction runs, every LB stays resident.
    const std::string dir = freshStewardDir("test_steward_ring_contracts");
    gl::GlobalMemoryManager pressure;
    pressure.init(kStewardTestCfg);             // 4 blocks
    std::vector<char*> held;
    for (int i = 0; i < 4; ++i) held.push_back(pressure.acquireBlock());

    const std::size_t n = gl::steward::kIoRingCapacity + 44;   // 300
    std::deque<gl::Memory> lbs;
    for (std::size_t i = 0; i < n; ++i) {
        lbs.emplace_back();
        lbs.back().setExprKey(("(ring_" + std::to_string(i) + ")").c_str());
        lbs.back().isActive = true;
        lbs.back().intEncodedStatements.push_back(
            stewardExpr(static_cast<int16_t>(i % 1000)));
    }
    std::vector<gl::Memory*> order;
    for (std::size_t i = 0; i < n; ++i) order.push_back(&lbs[i]);

    // Cursor near the end: window [297, 302) over n=300; every index below
    // 297 is an eviction candidate (297 candidates > 256 slots).
    std::atomic<std::size_t> cursor{ 298 };
    gl::MemorySteward s;
    s.setPressurePoolForTest(&pressure);
    s.setReserveBlocksForTest(1 << 20);          // unreachable: wants them all
    s.setMinEvictBlocksForTest(1);
    s.beginPhaseWindow(/*phase=*/1, &cursor, &order, /*workers=*/1, dir);

    const int64_t drop0 = gl::deloadStats().evictDroppedFullRing.load();
    s.maintainWorkingSet(&cursor, &order, /*workers=*/1);
    ASSERT_EQ(s.ioRingDepthForTest(false), gl::steward::kIoRingCapacity);
    ASSERT_EQ(gl::deloadStats().evictDroppedFullRing.load() - drop0,
              static_cast<int64_t>(1));

    // (1) Re-plan without draining: every candidate is either queued
    // (Duplicate) or dropped once more at the full ring — depth unchanged.
    s.maintainWorkingSet(&cursor, &order, /*workers=*/1);
    ASSERT_EQ(s.ioRingDepthForTest(false), gl::steward::kIoRingCapacity);
    ASSERT_EQ(gl::deloadStats().evictDroppedFullRing.load() - drop0,
              static_cast<int64_t>(2));

    // (3) Close the window, then drain: the queued generation is stale, so
    // every task drops — zero evictions, everything still resident.
    s.endPhaseWindow();
    const int64_t evict0 = gl::deloadStats().evictCount.load();
    s.drainIoRingsForTest();
    ASSERT_EQ(s.ioRingDepthForTest(false), static_cast<std::size_t>(0));
    ASSERT_EQ(gl::deloadStats().evictCount.load() - evict0,
              static_cast<int64_t>(0));
    ASSERT_TRUE(lbs[0].lbMemory.manager.resident());
    ASSERT_TRUE(lbs[100].lbMemory.manager.resident());
    ASSERT_TRUE(lbs[296].lbMemory.manager.resident());

    s.setPressurePoolForTest(nullptr);
    for (char* b : held) pressure.releaseBlock(b);
}

TEST(steward, async_dump_publishes_dumped_only_after_blocks_returned) {
    // The async dump contract through the LIVE executor pool: an eviction
    // task holds the LB Busy and its blocks until the raw image write
    // completes, releases the blocks, and only THEN publishes Dumped — so
    // observing Dumped from another thread guarantees the image is complete
    // on disk and the blocks are back in the pool.
    const std::string dir = freshStewardDir("test_steward_async_dump");
    gl::GlobalMemoryManager pressure;
    pressure.init(kStewardTestCfg);             // 4 blocks
    std::vector<char*> held;
    for (int i = 0; i < 4; ++i) held.push_back(pressure.acquireBlock());

    gl::Memory victim;
    victim.setExprKey("(async_dump_victim)");
    victim.isActive = true;
    for (int16_t i = 0; i < 25; ++i)
        victim.intEncodedStatements.push_back(stewardExpr(i));
    gl::Memory filler;
    filler.setExprKey("(async_dump_filler)");
    filler.isActive = true;
    filler.intEncodedStatements.push_back(stewardExpr(71));

    std::vector<gl::Memory*> order = { &victim, &filler };
    std::atomic<std::size_t> cursor{ 2 };
    gl::MemorySteward s;
    s.setPressurePoolForTest(&pressure);
    s.setMinEvictBlocksForTest(1);
    s.start(/*workers=*/8);                     // live planner + 2 executors
    s.beginPhaseWindow(/*phase=*/1, &cursor, &order, /*workers=*/1, dir);

    // The live planner plans the eviction on its poll; an executor runs it.
    bool dumped = false;
    for (int spin = 0; spin < 20000; ++spin) {
        if (victim.stewardClaim.load(std::memory_order_acquire)
            == static_cast<uint8_t>(gl::Memory::StewardClaim::Dumped)) {
            dumped = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    ASSERT_TRUE(dumped);
    // Dumped observed => blocks already returned, image already recorded,
    // and this (sole) dump's in-flight accounting already unwound.
    ASSERT_FALSE(victim.lbMemory.manager.resident());
    ASSERT_EQ(static_cast<int>(victim.deloadFiles.size()), 1);
    ASSERT_TRUE(std::filesystem::exists(
        std::filesystem::path(dir) / victim.deloadFiles.front()));
    ASSERT_EQ(s.inFlightBlocks(), static_cast<int64_t>(0));
    ASSERT_EQ(s.inFlightBytes(), static_cast<int64_t>(0));

    s.endPhaseWindow();
    s.quiesce();
    ASSERT_EQ(s.ioRingDepthForTest(true), static_cast<std::size_t>(0));
    ASSERT_EQ(s.ioRingDepthForTest(false), static_cast<std::size_t>(0));
    s.stop();
    s.setPressurePoolForTest(nullptr);

    victim.ensureLoaded(dir);
    ASSERT_EQ(victim.intEncodedStatements.size(), 25);
    ASSERT_EQ(victim.intEncodedStatements[24].nameId,
              static_cast<int16_t>(24));
    for (char* b : held) pressure.releaseBlock(b);
}

TEST(steward, prefetch_head_loads_next_iteration_head) {
    // The barrier head prefetch: one-shot HIGH-lane loads for the Dumped,
    // still-active LBs among order[0, count) — no window open (the tasks
    // carry the any-window sentinel). Skips: inactive (discharged this
    // barrier), claimed, resident (idempotence), and anything at or beyond
    // `count`.
    const std::string dir = freshStewardDir("test_steward_prefetch_head");
    auto fill = [](gl::Memory& m, const std::string& key, int16_t seed) {
        m.setExprKey(key.c_str());
        m.isActive = true;
        m.intEncodedStatements.push_back(stewardExpr(seed));
    };
    gl::Memory coldHead;      // index 0: Dumped + active -> loaded
    fill(coldHead, "(ph_cold_head)", 81);
    coldHead.deloadStaticContainersRaw(dir);
    coldHead.stewardClaim.store(
        static_cast<uint8_t>(gl::Memory::StewardClaim::Dumped));
    gl::Memory warm;          // index 1: resident Idle -> untouched
    fill(warm, "(ph_warm)", 82);
    gl::Memory coldInactive;  // index 2: Dumped but INACTIVE -> skipped
    fill(coldInactive, "(ph_cold_inactive)", 83);
    coldInactive.deloadStaticContainersRaw(dir);
    coldInactive.stewardClaim.store(
        static_cast<uint8_t>(gl::Memory::StewardClaim::Dumped));
    coldInactive.isActive = false;
    gl::Memory claimed;       // index 3: WorkerOwned -> skipped
    fill(claimed, "(ph_claimed)", 84);
    claimed.stewardClaim.store(
        static_cast<uint8_t>(gl::Memory::StewardClaim::WorkerOwned));
    gl::Memory coldTail;      // index 4: Dumped + active but BEYOND count
    fill(coldTail, "(ph_cold_tail)", 85);
    coldTail.deloadStaticContainersRaw(dir);
    coldTail.stewardClaim.store(
        static_cast<uint8_t>(gl::Memory::StewardClaim::Dumped));

    std::vector<gl::Memory*> order = {
        &coldHead, &warm, &coldInactive, &claimed, &coldTail };
    gl::MemorySteward s;
    s.prefetchHead(&order, /*count=*/4, dir);
    // Exactly one load enqueued: the Dumped active head inside [0, 4).
    ASSERT_EQ(s.ioRingDepthForTest(true), static_cast<std::size_t>(1));
    s.drainIoRingsForTest();

    ASSERT_TRUE(coldHead.lbMemory.manager.resident());
    ASSERT_EQ(coldHead.stewardClaim.load(),
              static_cast<uint8_t>(gl::Memory::StewardClaim::Idle));
    ASSERT_EQ(coldHead.intEncodedStatements.size(), 1);
    ASSERT_EQ(coldHead.intEncodedStatements[0].nameId,
              static_cast<int16_t>(81));
    ASSERT_TRUE(warm.lbMemory.manager.resident());
    ASSERT_FALSE(coldInactive.lbMemory.manager.resident());
    ASSERT_EQ(claimed.stewardClaim.load(),
              static_cast<uint8_t>(gl::Memory::StewardClaim::WorkerOwned));
    ASSERT_FALSE(coldTail.lbMemory.manager.resident());

    // Idempotent when the head is warm: a second call enqueues nothing.
    s.prefetchHead(&order, /*count=*/4, dir);
    ASSERT_EQ(s.ioRingDepthForTest(true), static_cast<std::size_t>(0));
}

TEST(steward, prefetch_head_respects_budget) {
    // The PREFETCH BUDGET at the head-prefetch issue site: enqueueing stops
    // when the estimated blocks of the issued loads reach
    // reserveBlocks_ / kPrefetchBudgetDivisor. These loads execute with no
    // window open (no planner eviction can relieve them; the executor load
    // path has no valve), so pending loads must always fit inside the
    // reserve's free blocks — the unbudgeted burst was the 4 GiB teardown
    // wall. Six one-block dumped LBs against a budget of 4/2 = 2 blocks:
    // exactly two loads enqueue; the tail stays cold (a DEFINED stop — the
    // un-warmed tail self-loads through the phase-1 handshake).
    const std::string dir = freshStewardDir("test_steward_head_budget");
    std::deque<gl::Memory> lbs;
    for (int i = 0; i < 6; ++i) {
        lbs.emplace_back();
        lbs.back().setExprKey(("(hb_" + std::to_string(i) + ")").c_str());
        lbs.back().isActive = true;
        for (int k = 0; k < 200; ++k)
            lbs.back().intEncodedStatements.push_back(
                stewardExpr(static_cast<int16_t>(k)));
        lbs.back().deloadStaticContainersRaw(dir);
        lbs.back().stewardClaim.store(
            static_cast<uint8_t>(gl::Memory::StewardClaim::Dumped));
        // The budget estimate comes from the raw image: one block each.
        ASSERT_TRUE(lbs.back().lastRawImageBytes() > 0);
        ASSERT_TRUE(lbs.back().lastRawImageBytes()
                    <= gl::staticMemory().blockBytes());
    }
    std::vector<gl::Memory*> order;
    for (int i = 0; i < 6; ++i) order.push_back(&lbs[i]);

    gl::MemorySteward s;
    s.setReserveBlocksForTest(4);            // budget = 4 / 2 = 2 blocks
    s.prefetchHead(&order, /*count=*/6, dir);
    ASSERT_EQ(s.ioRingDepthForTest(true), static_cast<std::size_t>(2));
    s.drainIoRingsForTest();
    ASSERT_TRUE(lbs[0].lbMemory.manager.resident());
    ASSERT_TRUE(lbs[1].lbMemory.manager.resident());
    ASSERT_FALSE(lbs[2].lbMemory.manager.resident());
    ASSERT_FALSE(lbs[5].lbMemory.manager.resident());
}

TEST(steward, planner_prefetch_respects_budget) {
    // The PREFETCH BUDGET at the planner's per-pass HIGH-lane issue site:
    // pending queued load blocks plus this pass's issues never exceed the
    // budget — the load-side twin of the eviction projection. Six one-block
    // dumped LBs inside the prefetch region against a budget of 2: the pass
    // plans exactly two loads, and a re-plan adds nothing (the queued
    // blocks are counted under the mutex).
    const std::string dir = freshStewardDir("test_steward_planner_budget");
    std::deque<gl::Memory> lbs;
    for (int i = 0; i < 8; ++i) {
        lbs.emplace_back();
        lbs.back().setExprKey(("(pb_" + std::to_string(i) + ")").c_str());
        lbs.back().isActive = true;
        for (int k = 0; k < 200; ++k)
            lbs.back().intEncodedStatements.push_back(
                stewardExpr(static_cast<int16_t>(k)));
    }
    // The first six go cold; the prefetch region [0, 6) covers exactly them.
    for (int i = 0; i < 6; ++i) {
        lbs[i].deloadStaticContainersRaw(dir);
        lbs[i].stewardClaim.store(
            static_cast<uint8_t>(gl::Memory::StewardClaim::Dumped));
    }
    std::vector<gl::Memory*> order;
    for (int i = 0; i < 8; ++i) order.push_back(&lbs[i]);

    gl::MemorySteward s;
    s.setReserveBlocksForTest(4);            // budget = 2 blocks
    std::atomic<std::size_t> cursor{ 0 };    // region [0, 2 + 4) = [0, 6)
    s.beginPhaseWindow(/*phase=*/1, &cursor, &order, /*workers=*/2, dir);
    s.maintainWorkingSet(&cursor, &order, /*workers=*/2);
    ASSERT_EQ(s.ioRingDepthForTest(true), static_cast<std::size_t>(2));
    // A second planning pass adds nothing: the two queued loads already
    // fill the budget.
    s.maintainWorkingSet(&cursor, &order, /*workers=*/2);
    ASSERT_EQ(s.ioRingDepthForTest(true), static_cast<std::size_t>(2));
    s.drainIoRingsForTest();
    ASSERT_TRUE(lbs[0].lbMemory.manager.resident());
    ASSERT_TRUE(lbs[1].lbMemory.manager.resident());
    ASSERT_FALSE(lbs[2].lbMemory.manager.resident());
    s.endPhaseWindow();
}

TEST(steward, discard_queued_io_tasks_drops_everything_unstarted) {
    // prove()'s teardown load-shedding: discardQueuedIoTasks drops every
    // queued-but-unstarted task on both rings — after the last iteration
    // the head warm-up is pure waste, and with no window open nothing could
    // relieve its grants. A subsequent drain executes NOTHING.
    const std::string dir = freshStewardDir("test_steward_discard");
    std::deque<gl::Memory> lbs;
    for (int i = 0; i < 3; ++i) {
        lbs.emplace_back();
        lbs.back().setExprKey(("(dq_" + std::to_string(i) + ")").c_str());
        lbs.back().isActive = true;
        lbs.back().intEncodedStatements.push_back(
            stewardExpr(static_cast<int16_t>(60 + i)));
        lbs.back().deloadStaticContainersRaw(dir);
        lbs.back().stewardClaim.store(
            static_cast<uint8_t>(gl::Memory::StewardClaim::Dumped));
    }
    std::vector<gl::Memory*> order;
    for (int i = 0; i < 3; ++i) order.push_back(&lbs[i]);

    gl::MemorySteward s;                     // default budget: all three fit
    s.prefetchHead(&order, /*count=*/3, dir);
    ASSERT_EQ(s.ioRingDepthForTest(true), static_cast<std::size_t>(3));
    s.discardQueuedIoTasks();
    ASSERT_EQ(s.ioRingDepthForTest(true), static_cast<std::size_t>(0));
    ASSERT_EQ(s.ioRingDepthForTest(false), static_cast<std::size_t>(0));
    s.drainIoRingsForTest();                 // nothing left to execute
    for (int i = 0; i < 3; ++i) {
        ASSERT_FALSE(lbs[i].lbMemory.manager.resident());
        ASSERT_EQ(lbs[i].stewardClaim.load(),
                  static_cast<uint8_t>(gl::Memory::StewardClaim::Dumped));
    }
}

TEST(steward, seam_door_phase4_reloads_claim_correct) {
    // The barrier-seam DOOR: a seam reload goes through the uniform
    // handshake with the phase-4 (barrier) telemetry bucket, so the claim
    // word ends consistent — Dumped-but-resident (the 4 GiB forensic
    // census's 290-LB anomaly, produced by direct ensureLoaded calls that
    // bypassed the claim word) is impossible by construction, and the
    // handshake's Dumped-implies-not-resident assert would catch any
    // bypass loudly. The assert's positive path runs in every round-trip
    // here; its firing path aborts the process and is untestable, like
    // every assert in this suite.
    const std::string dir = freshStewardDir("test_steward_seam_door");
    gl::MemorySteward s;
    gl::Memory recipient;
    recipient.setExprKey("(seam_recipient)");
    recipient.isActive = true;
    recipient.intEncodedStatements.push_back(stewardExpr(95));
    recipient.deloadStaticContainersRaw(dir);       // -> cold
    recipient.stewardClaim.store(
        static_cast<uint8_t>(gl::Memory::StewardClaim::Dumped));
    ASSERT_FALSE(recipient.lbMemory.manager.resident());

    // The door: claim + reload under phase 4 (the barrier bucket).
    s.claimAndLoadForWork(recipient, /*phase=*/4, dir);
    ASSERT_EQ(recipient.stewardClaim.load(),
              static_cast<uint8_t>(gl::Memory::StewardClaim::WorkerOwned));
    ASSERT_TRUE(recipient.lbMemory.manager.resident());
    ASSERT_EQ(recipient.intEncodedStatements.size(), 1);
    ASSERT_EQ(recipient.intEncodedStatements[0].nameId,
              static_cast<int16_t>(95));
    // ... the seam writes its deposit here, under WorkerOwned ...
    recipient.stewardClaim.store(
        static_cast<uint8_t>(gl::Memory::StewardClaim::Idle),
        std::memory_order_release);
    // Claim-correct end state: Idle + resident (evictable again).
    ASSERT_EQ(recipient.stewardClaim.load(),
              static_cast<uint8_t>(gl::Memory::StewardClaim::Idle));
    ASSERT_TRUE(recipient.lbMemory.manager.resident());
}

TEST(steward, barrier_seam_window_pinned_cursor_keeps_head_evicts_tail) {
    // BARRIER SEAM WINDOW B semantics: between iterations the next use of
    // active[i] IS position i, so a window with the cursor PINNED AT 0 keeps
    // the next iteration's head resident and evicts everything else
    // farthest-first while the (random-access) drains churn. Recipient churn
    // is correct behavior: a door-reloaded recipient released to Idle is
    // immediately eligible again and the next planning pass re-evicts it.
    const std::string dir = freshStewardDir("test_steward_seam_window");
    const std::size_t n = 20;
    const unsigned workers = 1;          // pinned window [0, 3)
    const int deficitLbs = 3;

    std::deque<gl::Memory> lbs;
    for (std::size_t i = 0; i < n; ++i) {
        lbs.emplace_back();
        lbs.back().setExprKey(("(sw_" + std::to_string(i) + ")").c_str());
        lbs.back().isActive = true;
        for (int k = 0; k < 200; ++k)
            lbs.back().intEncodedStatements.push_back(
                stewardExpr(static_cast<int16_t>(k)));
    }
    std::vector<gl::Memory*> order;
    for (std::size_t i = 0; i < n; ++i) order.push_back(&lbs[i]);
    const int64_t b = lbs[0].lbMemory.manager.blocksHeld();
    ASSERT_TRUE(b >= 1);

    gl::MemorySteward s;
    const int64_t total = gl::staticMemory().totalBlocks();
    const int64_t free0 = total - gl::staticMemory().blocksInUse();
    s.setReserveBlocksForTest(free0 + deficitLbs * b);
    s.setMinEvictBlocksForTest(1);

    std::atomic<std::size_t> cursor{ 0 };   // PINNED — never advanced
    s.beginPhaseWindow(/*phase=*/4, &cursor, &order, workers, dir);
    s.maintainWorkingSet(&cursor, &order, workers);
    s.drainIoRingsForTest();

    // Farthest-first from the pinned origin: the TAIL evicts, exactly the
    // deficit; the head [0, 3) — the next iteration's first LBs — is kept.
    ASSERT_FALSE(lbs[n - 1].lbMemory.manager.resident());
    ASSERT_FALSE(lbs[n - 2].lbMemory.manager.resident());
    ASSERT_FALSE(lbs[n - 3].lbMemory.manager.resident());
    ASSERT_TRUE(lbs[n - 4].lbMemory.manager.resident());
    ASSERT_TRUE(lbs[0].lbMemory.manager.resident());
    ASSERT_TRUE(lbs[1].lbMemory.manager.resident());
    ASSERT_TRUE(lbs[2].lbMemory.manager.resident());

    // RECIPIENT CHURN: a random-access seam deposit door-reloads an evicted
    // tail LB, writes under WorkerOwned, releases Idle...
    s.claimAndLoadForWork(lbs[n - 1], /*phase=*/4, dir);
    ASSERT_EQ(lbs[n - 1].stewardClaim.load(),
              static_cast<uint8_t>(gl::Memory::StewardClaim::WorkerOwned));
    ASSERT_TRUE(lbs[n - 1].lbMemory.manager.resident());
    lbs[n - 1].stewardClaim.store(
        static_cast<uint8_t>(gl::Memory::StewardClaim::Idle),
        std::memory_order_release);

    // ...and the next planning pass takes it right back (its reload consumed
    // reserve blocks; it is the farthest eligible victim behind the pinned
    // head) — the churn the design accepts as correct.
    s.maintainWorkingSet(&cursor, &order, workers);
    s.drainIoRingsForTest();
    ASSERT_FALSE(lbs[n - 1].lbMemory.manager.resident());
    ASSERT_EQ(lbs[n - 1].stewardClaim.load(),
              static_cast<uint8_t>(gl::Memory::StewardClaim::Dumped));
    ASSERT_TRUE(lbs[0].lbMemory.manager.resident());   // head still kept

    s.endPhaseWindow();
}

TEST(steward, exhaustion_reporter_census_runs_under_pool_mutex) {
    // The forensic exhaustion trap. The DEATH PATH ITSELF IS UNTESTABLE in
    // this harness: the real invocation site (grantLocked at the wall) is
    // immediately followed by an assert that aborts the whole process, and
    // the harness has no death-test support. What CAN be pinned is the exact
    // calling context: the manager's test seam takes the pool mutex and
    // invokes the installed reporter under it — precisely grantLocked's
    // context minus the abort. A census that re-entered the manager (any
    // mutex-taking call) would DEADLOCK here and fail the suite loudly; a
    // clean return proves the no-re-entry contract, and the captured
    // arguments prove the counters arrive from the held-mutex read.
    const std::string dir = freshStewardDir("test_steward_exhaustion");
    gl::GlobalMemoryManager pool;
    pool.init(kStewardTestCfg);                 // 4 blocks
    char* b1 = pool.acquireBlock();
    char* b2 = pool.acquireBlock();             // blocksInUse = 2

    // A small grid covering the census buckets: Idle+resident (an eligible
    // victim), Dumped (cold), WorkerOwned (pinned).
    gl::Memory idleLb;
    idleLb.setExprKey("(exh_idle)");
    idleLb.isActive = true;
    idleLb.intEncodedStatements.push_back(stewardExpr(91));
    gl::Memory dumpedLb;
    dumpedLb.setExprKey("(exh_dumped)");
    dumpedLb.isActive = true;
    dumpedLb.intEncodedStatements.push_back(stewardExpr(92));
    dumpedLb.deloadStaticContainersRaw(dir);
    dumpedLb.stewardClaim.store(
        static_cast<uint8_t>(gl::Memory::StewardClaim::Dumped));
    gl::Memory ownedLb;
    ownedLb.setExprKey("(exh_owned)");
    ownedLb.isActive = true;
    ownedLb.intEncodedStatements.push_back(stewardExpr(93));
    ownedLb.stewardClaim.store(
        static_cast<uint8_t>(gl::Memory::StewardClaim::WorkerOwned));
    std::vector<gl::Memory*> grid = { &idleLb, &dumpedLb, &ownedLb };

    gl::MemorySteward s;                        // census reads its atomics
    bool reported = false;
    int64_t seenInUse = -1;
    int64_t seenTotal = -1;
    pool.setExhaustionReporter(
        [&](int64_t inUse, int64_t total) {
            reported = true;
            seenInUse = inUse;
            seenTotal = total;
            // The production census body, under the held pool mutex — the
            // no-re-entry audit made executable.
            gl::printExhaustionCensus(grid, &s, inUse, total);
        });
    pool.invokeExhaustionReporterForTest();
    // (The census's UNATTRIBUTED line goes negative in THIS fixture — the
    // grid LBs' blocks live on the process pool while the counters passed in
    // are the private pool's. In production both come from the same main
    // pool, so the residue is coherent; the test pins the call contract, not
    // the arithmetic.)

    ASSERT_TRUE(reported);
    ASSERT_EQ(seenInUse, static_cast<int64_t>(2));
    ASSERT_EQ(seenTotal, static_cast<int64_t>(4));

    pool.setExhaustionReporter(nullptr);        // the scope-guard lifecycle
    pool.releaseBlock(b1);
    pool.releaseBlock(b2);
}

TEST(steward, stress_concurrent_dump_load_disjoint_lbs_across_executors) {
    // The executor-pool stress: a live planner + executor pool continuously
    // evicts out-of-window LBs and reloads prefetch-region ones, while four
    // test worker threads concurrently claim, verify, and release DISJOINT
    // LB subsets through the production handshake (whose emergency valve
    // also fires on every claim — free blocks are pinned at 0). Repeated
    // concurrent dump/load traffic on the same claim words from executors,
    // the planner, and workers at once; the pass criterion is pure content
    // integrity plus a clean quiesce.
    const std::string dir = freshStewardDir("test_steward_stress");
    gl::GlobalMemoryManager pressure;
    pressure.init(kStewardTestCfg);             // 4 blocks
    std::vector<char*> held;
    for (int i = 0; i < 4; ++i) held.push_back(pressure.acquireBlock());

    const std::size_t n = 16;
    std::deque<gl::Memory> lbs;
    for (std::size_t i = 0; i < n; ++i) {
        lbs.emplace_back();
        lbs.back().setExprKey(("(st_" + std::to_string(i) + ")").c_str());
        lbs.back().isActive = true;
        // Distinct content signature per LB: 10 + i rows, seeded by i.
        for (int16_t k = 0; k < static_cast<int16_t>(10 + i); ++k)
            lbs.back().intEncodedStatements.push_back(
                stewardExpr(static_cast<int16_t>(i * 100 + k)));
    }
    std::vector<gl::Memory*> order;
    for (std::size_t i = 0; i < n; ++i) order.push_back(&lbs[i]);

    // Window [7, 11) at a fixed cursor 8: indices 0..6 and 11..15 are the
    // executors' eviction hunting ground; 8..10 the prefetch region.
    std::atomic<std::size_t> cursor{ 8 };
    gl::MemorySteward s;
    s.setPressurePoolForTest(&pressure);
    s.setMinEvictBlocksForTest(1);
    s.start(/*workers=*/32);                    // 4 executors
    s.beginPhaseWindow(/*phase=*/1, &cursor, &order, /*workers=*/1, dir);

    // Four workers, four DISJOINT subsets of four LBs each; every claim
    // round-trips the production handshake and verifies content.
    std::atomic<int> contentErrors{ 0 };
    auto workerFn = [&](unsigned w) {
        for (int round = 0; round < 20; ++round) {
            for (std::size_t k = 0; k < 4; ++k) {
                const std::size_t i = w * 4 + k;
                gl::Memory& lb = lbs[i];
                s.claimAndLoadForWork(lb, /*phase=*/(round % 3) + 1, dir);
                if (lb.intEncodedStatements.size()
                        != static_cast<int32_t>(10 + i)
                    || lb.intEncodedStatements[0].nameId
                           != static_cast<int16_t>(i * 100))
                    contentErrors.fetch_add(1);
                lb.stewardClaim.store(
                    static_cast<uint8_t>(gl::Memory::StewardClaim::Idle),
                    std::memory_order_release);
            }
        }
    };
    std::vector<std::thread> workers;
    for (unsigned w = 0; w < 4; ++w) workers.emplace_back(workerFn, w);
    for (std::thread& t : workers) t.join();

    s.endPhaseWindow();
    s.quiesce();
    ASSERT_EQ(s.ioRingDepthForTest(true), static_cast<std::size_t>(0));
    ASSERT_EQ(s.ioRingDepthForTest(false), static_cast<std::size_t>(0));
    ASSERT_EQ(s.inFlightBlocks(), static_cast<int64_t>(0));
    s.stop();
    s.setPressurePoolForTest(nullptr);

    ASSERT_EQ(contentErrors.load(), 0);
    // The stress really moved dump/load traffic.
    ASSERT_TRUE(gl::deloadStats().rawDumpCount.load() > 0);
    // Final integrity sweep: every LB reloads to its exact signature.
    for (std::size_t i = 0; i < n; ++i) {
        lbs[i].ensureLoaded(dir);
        ASSERT_EQ(lbs[i].intEncodedStatements.size(),
                  static_cast<int32_t>(10 + i));
        ASSERT_EQ(lbs[i].intEncodedStatements[0].nameId,
                  static_cast<int16_t>(i * 100));
    }
    for (char* b : held) pressure.releaseBlock(b);
}
