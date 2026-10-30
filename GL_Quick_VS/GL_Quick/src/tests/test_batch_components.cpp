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
#include "test_harness.hpp"
#include "prover_fixture.hpp"
#include "../prover.hpp"

TEST(batch_components, constructs_explicit_operation_interfaces) {
    gl::tests::ProverFixture engine(std::string("Peano"));
    const auto initialTheoremCount = engine.globalTheoremList.size();
    [[maybe_unused]] gl::TheoremPublisher theoremPublisher{engine.anchorScopeQueries, engine.body, engine.definitionExpansion, engine.definitions, engine.dependenciesMutex, engine.fullTheoremList, engine.globalDependencies, engine.globalTheoremList, engine.globalTheoremProducers, engine.globalTheoremStrings, engine.orInRunScannedRows, engine.parameters, engine.pendingCompactionMutex, engine.pendingCompactionQueue, engine.proofHistory, engine.retractedVacuousTheorems, engine.scopeLifecycle, engine.simpleMapStore, engine.steward, engine.structuralCompiler, engine.theoremListMutex, engine.updateGlobalSingleChapterPages, engine.orCandidateProgress, engine.orInRunCompanions};
    [[maybe_unused]] gl::ProofGridBuilder proofGridBuilder{engine.admissionMemory, engine.anchorInfo, engine.anchorScopeQueries, engine.blockIdentity, engine.body, engine.definitions, engine.hashRuleInstaller, engine.inductionUsefulness, engine.integrationProcessor, engine.lbStore, engine.mailLog, engine.parameters, engine.permanentBodies, engine.scopeTypes, engine.simpleMapStore, engine.statementMemory, engine.theoremPublisher};
    [[maybe_unused]] gl::OrTheoremBuilder orTheoremBuilder{engine.definitions, engine.fullTheoremList, engine.globalTheoremList, engine.orBuiltByPair, engine.orCandidates, engine.orCandidatesMutex, engine.orInRunScannedRows, engine.orPairsFromHeadSwitch, engine.orRuleCompiler, engine.parameters, engine.proofHistory, engine.theoremListMutex, engine.theoremPublisher, engine.orInRunCompanions, engine.scopeLifecycle, engine.deactivationSweepOwed};
    [[maybe_unused]] gl::OrEliminationBuilder orEliminationBuilder{engine.definitions, engine.fullTheoremList, engine.globalTheoremList, engine.orElimBuiltByPair, engine.orElimCitedOrByMerged, engine.orElimGuardIndex, engine.orElimInRunScannedRows, engine.orElimPendingPairs, engine.orRuleCompiler, engine.parameters, engine.proofHistory, engine.theoremListMutex, engine.theoremPublisher};
    [[maybe_unused]] gl::CounterexampleFilter counterexampleFilter{engine.anchorInfo, engine.ceBody, engine.ceSimpleMapStore, engine.definitions, engine.filteredConjecturesDirectory, engine.hashRuleInstaller, engine.inductionMemoryBlocks, engine.lbStore, engine.parameters, engine.permanentBodiesCE, engine.statementMemory, engine.updateGlobalTuples};
    [[maybe_unused]] gl::EncodedExpressionNormalizer encodedExpressionNormalizer{};
    ASSERT_EQ(engine.globalTheoremList.size(), initialTheoremCount);
}
