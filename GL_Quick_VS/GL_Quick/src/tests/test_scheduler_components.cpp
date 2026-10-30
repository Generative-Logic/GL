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
#include "../memory.hpp"
#include <cstddef>
#include <utility>
#include "test_harness.hpp"
#include "prover_fixture.hpp"
#include "../prover.hpp"

TEST(scheduler_components, constructs_explicit_operation_interfaces) {
    gl::tests::ProverFixture engine(std::string("Peano"));
    const auto initialTheoremCount = engine.globalTheoremList.size();
    [[maybe_unused]] gl::Phase2Executor phase2Executor{engine.parameters, engine.phase2Backend, engine.phase2CumulativeSeconds, engine.phase2MeasuredIterations, ((engine).phase2MeasurementEpoch), engine.ceFilteringActive, engine.requestGenerator, engine.ruleEvaluator, engine.anchorID_, engine.logicalCores, engine.steward, engine.admissionMemory, engine.anchorInfo, engine.definitions};
    [[maybe_unused]] gl::ProofBarrier proofBarrier{engine.parameters, engine.body, engine.updateGlobalTuples, engine.inductionMemoryBlocks, engine.ceFilteringActive, engine.definitions, engine.mailLog, engine.rollingMailHistoryEnabled, engine.peakDeloadableMailOutBytes, engine.simpleMapStore, engine.dischargeScratch, engine.pendingCompactionQueue, engine.pendingDischarge, engine.steward, engine.structuralCompiler, engine.orRuleCompiler, engine.deactivationSweepOwed, engine.admissionMemory, engine.proofHistory, engine.scopeLifecycle, engine.blockMailProcessor, engine.anchorScopeQueries, engine.theoremPublisher, engine.orTheoremBuilder, engine.orEliminationBuilder};
    ASSERT_EQ(engine.globalTheoremList.size(), initialTheoremCount);
}

TEST(scheduler_components, empty_phase_two_preserves_barrier_accounting) {
    gl::tests::ProverFixture engine("Peano");
    std::vector<gl::Memory*> active;
    gl::Phase2TraceScope trace("unit.empty.phase2", 0, false, false);
#if RT_MEASUREMENT
    std::vector<int64_t> requestTimes, finalizeTimes, work;
#endif
    const auto prior = engine.phase2MeasuredIterations;
    const bool mainPath = engine.phase2Executor.execute(active, engine.logicalCores, 0,
        std::chrono::steady_clock::now(), trace
#if RT_MEASUREMENT
        , requestTimes, finalizeTimes, work
#endif
    );
    ASSERT_EQ(mainPath, engine.parameters.lb_split && !engine.parameters.disable_lb_split && !engine.ceFilteringActive);
    ASSERT_EQ(engine.phase2MeasuredIterations, prior + 1);
    ASSERT_TRUE(engine.phase2CumulativeSeconds >= 0.0);
}

TEST(scheduler_components, empty_post_phase_drain_releases_staging) {
    gl::tests::ProverFixture engine("Peano");
    engine.steward = std::make_unique<gl::MemorySteward>();
    engine.deferredAncestorPages.emplace();
    engine.deferredAncestorPages->bind(&gl::staticMemory());
    engine.updateGlobalSingleChapterPages.emplace();
    engine.updateGlobalSingleChapterPages->bind(&gl::staticMemory());
    const std::vector<gl::Memory*> blocks;
    engine.proofBarrier.finish(blocks, blocks, engine.logicalCores, gl::genScratchArenas().slotCount() - 1);
    ASSERT_FALSE(engine.deferredAncestorPages.has_value());
    ASSERT_FALSE(engine.updateGlobalSingleChapterPages.has_value());
    ASSERT_TRUE(engine.pendingCompactionQueue.empty());
    ASSERT_TRUE(engine.updateGlobalTuples.empty());
    engine.steward->quiesce();
}

#if RT_MEASUREMENT
TEST(scheduler_components, target_counter_reset_and_reads) {
    gl::Phase2Executor::resetTargetCounters();
    ASSERT_EQ(gl::Phase2Executor::targetAccepted(), int64_t(0));
    ASSERT_EQ(gl::Phase2Executor::targetAcceptedUnlinked(), int64_t(0));
    ASSERT_EQ(gl::Phase2Executor::targetAttempts(), int64_t(0));
    ASSERT_EQ(gl::Phase2Executor::targetAttemptsUnlinked(), int64_t(0));
}
#endif
