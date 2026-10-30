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

TEST(runtime_components, constructs_explicit_operation_interfaces) {
    gl::tests::ProverFixture engine(std::string("Peano"));
    [[maybe_unused]] gl::AdmissionMemory admissionMemory{engine.admissionPolicy, engine.anchorInfo, engine.canonicalTemplates, engine.deferredAncestorAdmissionsMutex, engine.deferredAncestorPages, engine.definitions, engine.parameters, engine.scopeTypes, engine.steward, engine.integrationProcessor, engine.rejectedSourceReplay, engine.equivalenceQueries};
    [[maybe_unused]] gl::IntegrationProcessor integrationProcessor{engine.admissionPolicy, engine.canonicalTemplates, engine.canonicalizer, engine.definitions, engine.integrationBuilder, engine.parameters, engine.scopeTypes, engine.admissionMemory, engine.disintegrationProcessor, engine.hashRuleInstaller, engine.implicationSubproofs, engine.rejectedSourceReplay, engine.statementMemory, engine.unaryIntegrationProcessor, engine.compactRuleProcessor};
    [[maybe_unused]] gl::ImplicationSubproofs implicationSubproofs{engine.canonicalTemplates, engine.canonicalizer, engine.definitions, engine.integrationBuilder, engine.parameters, engine.equivalenceQueries, engine.statementMemory, engine.rejectedSourceReplay};
    [[maybe_unused]] gl::ContradictionDemand contradictionDemand{engine.definitions, engine.parameters, engine.integrationProcessor, engine.implicationSubproofs};
    [[maybe_unused]] gl::RejectedSourceReplay rejectedSourceReplay{engine.admissionPolicy, engine.anchorInfo, engine.canonicalTemplates, engine.canonicalizer, engine.definitions, engine.parameters, engine.rejectedSources, engine.admissionMemory, engine.caseLedger, engine.proofHistory, engine.watchProcessor};
    [[maybe_unused]] gl::HashRuleInstaller hashRuleInstaller{engine.admissionMarkers, engine.allPermutationsAna, engine.hashRuleClassifier, engine.implicationMultiplier, engine.parameters, engine.unaryMemory, engine.admissionMemory};
    [[maybe_unused]] gl::CompactRuleProcessor compactRuleProcessor{engine.canonicalizer, engine.parameters, engine.disintegrationProcessor, engine.hashRuleInstaller, engine.statementMemory, engine.integrationBuilder, engine.rejectedSourceReplay};
    [[maybe_unused]] gl::StatementMemory statementMemory{engine.admissionPolicy, engine.canonicalizer, engine.definitions, engine.parameters, engine.admissionMemory, engine.caseLedger, engine.caseProcessor, engine.compactRuleProcessor, engine.disintegrationProcessor, engine.equalityPropagation, engine.equivalenceProcessor, engine.equivalenceQueries, engine.integrationProcessor, engine.orImplicationProcessor, engine.proofHistory, engine.unaryIntegrationProcessor, engine.watchProcessor};
    [[maybe_unused]] gl::EquivalenceProcessor equivalenceProcessor{engine.canonicalTemplates, engine.canonicalizer, engine.parameters, engine.rejectedSources, engine.caseLedger, engine.caseProcessor, engine.compactRuleProcessor, engine.equalityPropagation, engine.equivalenceAdmissionRewriter, engine.rejectedSourceReplay, engine.watchProcessor};
    [[maybe_unused]] gl::EqualityPropagation equalityPropagation{engine.admissionPolicy, engine.allMappingsAna, engine.canonicalizer, engine.ceFilteringActive, engine.parameters, engine.caseProcessor, engine.proofHistory, engine.statementMemory, engine.watchProcessor};
    [[maybe_unused]] gl::EquivalenceAdmissionRewriter equivalenceAdmissionRewriter{engine.canonicalTemplates, engine.canonicalizer, engine.parameters, engine.rejectedSources, engine.caseLedger, engine.proofHistory, engine.rejectedSourceReplay, engine.watchProcessor};
    [[maybe_unused]] gl::WatchProcessor watchProcessor{engine.canonicalTemplates, engine.canonicalizer, engine.ceFilteringActive, engine.parameters, engine.caseLedger, engine.caseProcessor, engine.proofHistory};
    [[maybe_unused]] gl::CaseLedger caseLedger{engine.canonicalTemplates, engine.canonicalizer, engine.definitions};
    [[maybe_unused]] gl::CaseProcessor caseProcessor{engine.admissionPolicy, engine.anchorInfo, engine.canonicalTemplates, engine.canonicalizer, engine.definitions, engine.parameters, engine.caseLedger, engine.equivalenceQueries, engine.proofHistory};
    [[maybe_unused]] gl::DisintegrationProcessor disintegrationProcessor{engine.admissionPolicy, engine.canonicalTemplates, engine.canonicalizer, engine.ceFilteringActive, engine.definitions, engine.existenceRuleCompiler, engine.hashRuleClassifier, engine.integrationBuilder, engine.parameters, engine.rejectedSources, engine.admissionMemory, engine.caseLedger, engine.caseProcessor, engine.rejectedSourceReplay, engine.statementMemory};
    [[maybe_unused]] gl::OrImplicationProcessor orImplicationProcessor{engine.blockIdentity, engine.canonicalizer, engine.ceFilteringActive, engine.definitions, engine.hashRuleClassifier, engine.parameters, engine.caseLedger, engine.caseProcessor, engine.disintegrationProcessor, engine.proofHistory, engine.watchProcessor};
    [[maybe_unused]] gl::ProofHistory proofHistory{engine.parameters};
    [[maybe_unused]] gl::UnaryIntegrationProcessor unaryIntegrationProcessor{engine.canonicalTemplates, engine.canonicalizer, engine.parameters, engine.unaryMemory, engine.hashRuleInstaller, engine.rejectedSourceReplay, engine.integrationProcessor};
    [[maybe_unused]] gl::GoalProcessor goalProcessor{engine.canonicalizer, engine.contradictionTable, engine.definitions, engine.inductionMemoryBlocks, engine.inductionMemoryBlocksMutex, engine.parameters, engine.updateGlobalMutex, engine.updateGlobalSingleChapterMutex, engine.updateGlobalSingleChapterPages, engine.updateGlobalTuples, engine.blockKeyReader, engine.caseProcessor, engine.implicationSubproofs, engine.proofHistory, engine.scopeLifecycle, engine.statementMemory};
    [[maybe_unused]] gl::ScopeLifecycle scopeLifecycle{engine.body, engine.deactivationSweepOwed, engine.definitions, engine.globalDependencies, engine.pendingSubtreeDeactivations, engine.pendingSubtreeDeactivationsMutex, engine.pendingTwinDeactivations, engine.pendingTwinDeactivationsMutex, engine.simpleMapStore, engine.anchorScopeQueries};
    [[maybe_unused]] gl::LogicBlockOperations logicBlockOperations{engine.blockIdentity, engine.ceFilteringActive, engine.contradictionTable, engine.parameters, engine.rejectedSources, engine.blockMailProcessor, engine.caseProcessor, engine.equalityPropagation, engine.equivalenceProcessor, engine.goalProcessor, engine.implicationSubproofs, engine.inductionUsefulness, engine.orImplicationProcessor, engine.statementMemory, engine.unaryIntegrationProcessor, engine.watchProcessor, engine.contradictionDemand, engine.rejectedSourceReplay
#if PHASE13_DEEP_TIMING
, engine.phase13TimingRows
#endif

#if PHASE13_DEEP_TIMING
, engine.phase13TimingWorkers
#endif
};
    [[maybe_unused]] gl::BlockMailProcessor blockMailProcessor{engine.blockIdentity, engine.body, engine.definitions, engine.mailLog, engine.parameters, engine.permanentBodies, engine.simpleMapStore, engine.structuralCompiler, engine.proofHistory};
    [[maybe_unused]] gl::AnchorScopeQueries anchorScopeQueries{engine.definitionExpansion};
    [[maybe_unused]] gl::EquivalenceQueries equivalenceQueries{engine.canonicalTemplates, engine.canonicalizer};
    [[maybe_unused]] gl::BlockKeyReader blockKeyReader{};
    ASSERT_TRUE(blockKeyReader.getGlobalKey(engine.body).empty());
    gl::StrSpan keys[gl::ExecutionParameters::MAX_SCOPE_DEPTH];
    ASSERT_EQ(blockKeyReader.getGlobalKey(engine.body, keys,
        gl::ExecutionParameters::MAX_SCOPE_DEPTH), 0);
}
