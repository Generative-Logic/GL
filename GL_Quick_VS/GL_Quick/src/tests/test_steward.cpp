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
#include "../memory_infra/steward.hpp"

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
    s.beginPhaseWindow(&cursor, &order, 0, dir);
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

TEST(steward, pick_biggest_deloadable_selects_and_skips) {
    // The working-set drain's victim selector. A pure selector — no running
    // thread needed. The skip-cases deliberately hold MORE blocks than the
    // eligible LBs, so a selector with a broken skip predicate would wrongly
    // return a bigger skipped LB rather than the eligible one.
    gl::MemorySteward s;
    auto fill = [](gl::Memory& m, const char* key, int count) {
        m.setExprKey(key);
        for (int i = 0; i < count; ++i)
            m.intEncodedStatements.push_back(
                stewardExpr(static_cast<int16_t>(i)));
    };
    gl::Memory bigOk;    fill(bigOk, "(pick_big_ok)", 5000);      // eligible, bigger
    gl::Memory smallOk;  fill(smallOk, "(pick_small_ok)", 200);   // eligible, smaller
    gl::Memory worker;   fill(worker, "(pick_worker)", 8000);     // skip: in flight
    gl::Memory split;    fill(split, "(pick_split)", 8000);       // skip: split LB
    gl::Memory inactive; fill(inactive, "(pick_inactive)", 8000); // skip: inactive
    worker.stewardClaim.store(
        static_cast<uint8_t>(gl::Memory::StewardClaim::WorkerOwned));
    split.numberOfParts = 4;
    inactive.isActive = false;

    // Among the eligible LBs the biggest wins; the bigger skipped ones are
    // ignored. Window kept empty so eligibility is purely the predicate.
    std::vector<gl::Memory*> order = {
        &worker, &split, &smallOk, &inactive, &bigOk };
    ASSERT_TRUE(s.pickBiggestDeloadable(&order, 0, 0) == &bigOk);

    // The smaller eligible LB wins once the bigger eligible one (index 4)
    // is inside the kept window — the working set is never a victim.
    ASSERT_TRUE(s.pickBiggestDeloadable(&order, 4, 5) == &smallOk);

    // Every candidate ineligible -> nullptr.
    std::vector<gl::Memory*> none = { &worker, &split, &inactive };
    ASSERT_TRUE(s.pickBiggestDeloadable(&none, 0, 0) == nullptr);

    // The whole order kept inside the window -> nullptr.
    ASSERT_TRUE(
        s.pickBiggestDeloadable(&order, 0, order.size()) == nullptr);
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
    s.claimAndLoadForWork(cold, dir);
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
    s.claimAndLoadForWork(warm, dir);
    ASSERT_EQ(warm.stewardClaim.load(),
              static_cast<uint8_t>(gl::Memory::StewardClaim::WorkerOwned));
    ASSERT_EQ(warm.intEncodedStatements.size(), 1);

    // Already WorkerOwned (a sibling part owns it) -> idempotent no-op.
    s.claimAndLoadForWork(warm, dir);
    ASSERT_EQ(warm.stewardClaim.load(),
              static_cast<uint8_t>(gl::Memory::StewardClaim::WorkerOwned));
}
