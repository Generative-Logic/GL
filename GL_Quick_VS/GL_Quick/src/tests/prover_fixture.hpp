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
#pragma once
#include "../prover.hpp"
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
namespace gl::tests {

// The first compile of each configuration in this test process, keyed by the
// anchor name. Compiling the configured definitions dominates a Prover
// construction (about 1.2 s for FTA); a fixture of an anchor that already has a
// record installs it instead (`Prover`'s `precompiledDefinitions`), which yields
// exactly the state its own compile would produce. A fixture given a
// compiled-registry input always compiles and records nothing.
inline std::map<std::string, std::unique_ptr<const PrecompiledDefinitions>>& precompiledDefinitionRecords() {
    static std::map<std::string, std::unique_ptr<const PrecompiledDefinitions>> records;
    return records;
}

// The record a fixture of this anchor installs; null when it must compile.
inline const PrecompiledDefinitions* precompiledDefinitionsFor(
    const std::string& anchorID, const std::optional<std::filesystem::path>& glBinaryFile) {
    if (glBinaryFile.has_value()) return nullptr;
    const auto& records = precompiledDefinitionRecords();
    const auto it = records.find(anchorID);
    return it == records.end() ? nullptr : it->second.get();
}

struct ProverFixture : Prover {
    // Selects a fresh compile even when the anchor has a record.
    struct FreshCompile {};

    // Installs the anchor's recorded compile when there is one; otherwise
    // compiles and, without a compiled-registry input, records the result.
    explicit ProverFixture(std::string anchorID,
                           std::optional<Phase2Backend> selectedPhase2Backend = std::nullopt,
                           std::optional<std::filesystem::path> glBinaryFile = std::nullopt)
        : Prover(anchorID, selectedPhase2Backend, glBinaryFile,
                 precompiledDefinitionsFor(anchorID, glBinaryFile)) {
        auto& records = precompiledDefinitionRecords();
        if (!glBinaryFile.has_value() && records.find(anchorID) == records.end()) {
            records.emplace(anchorID, std::make_unique<const PrecompiledDefinitions>(
                PrecompiledDefinitions{ definitions, compilationReuse.state() }));
        }
    }

    // Compiles the configured definitions even when the anchor has a record;
    // records nothing.
    ProverFixture(FreshCompile, std::string anchorID,
                  std::optional<Phase2Backend> selectedPhase2Backend = std::nullopt)
        : Prover(anchorID, selectedPhase2Backend) {}

    // Installs the given recorded compile; records nothing.
    ProverFixture(const PrecompiledDefinitions& recorded, std::string anchorID)
        : Prover(anchorID, std::nullopt, std::nullopt, &recorded) {}

    using Prover::parameters;
    using Prover::phase2Backend;
    using Prover::filteredConjecturesDirectory;
    using Prover::phase2ShortcutProfile;
    using Prover::phase2CumulativeSeconds;
    using Prover::phase2MeasuredIterations;
    using Prover::phase2MeasurementEpoch;
#if PHASE13_DEEP_TIMING
    using Prover::phase13TimingRows;
#endif
#if PHASE13_DEEP_TIMING
    using Prover::phase13TimingWorkers;
#endif
    using Prover::body;
    using Prover::ceBody;
    using Prover::updateGlobalTuples;
    using Prover::globalTheoremList;
    using Prover::globalTheoremStrings;
    using Prover::globalTheoremProducers;
    using Prover::retractedVacuousTheorems;
    using Prover::fullTheoremList;
    using Prover::lastCompressionSurvivors;
    using Prover::inductionMemoryBlocks;
    using Prover::pendingTwinDeactivations;
    using Prover::pendingSubtreeDeactivations;
    using Prover::deferredAncestorPages;
    using Prover::updateGlobalSingleChapterPages;
    using Prover::allMappingsAna;
    using Prover::allPermutationsAna;
    using Prover::globalDependencies;
    using Prover::contradictionTable;
    using Prover::mirrorPartnerMap;
    using Prover::ceFilteringActive;
    using Prover::definitions;
    using Prover::integrationBuilder;
    using Prover::unaryMemory;
    using Prover::canonicalizer;
    using Prover::canonicalTemplates;
    using Prover::rejectedSources;
    using Prover::hashRuleClassifier;
    using Prover::implicationMultiplier;
    using Prover::admissionPolicy;
    using Prover::admissionMarkers;
    using Prover::anchorInfo;
    using Prover::blockIdentity;
    using Prover::requestGenerator;
    using Prover::ruleEvaluator;
    using Prover::anchorID_;
    using Prover::permanentBodies;
    using Prover::permanentBodiesCE;
    using Prover::logicalCores;
    using Prover::mailArena;
    using Prover::mailDirty;
    using Prover::mailLog;
    using Prover::dormantLogicBlocksAtGridBuild;
    using Prover::rollingMailHistoryEnabled;
    using Prover::routingMailInBlocksInFlight;
    using Prover::peakRoutingMailInBlocks;
    using Prover::peakDeloadableMailOutBytes;
    using Prover::warmUpPhase;
    using Prover::proofBurstNumber;
    using Prover::lastSweptCount;
    using Prover::lastSkippedCount;
    using Prover::lbStore;
    using Prover::simpleMapArena;
    using Prover::simpleMapDirty;
    using Prover::simpleMapStore;
    using Prover::ceSimpleMapArena;
    using Prover::ceSimpleMapDirty;
    using Prover::ceSimpleMapStore;
    using Prover::dischargeScratch;
    using Prover::pendingCompactionQueue;
    using Prover::pendingDischarge;
    using Prover::steward;
    using Prover::stewardEvictionPlan;
    using Prover::pendingCompactionMutex;
    using Prover::cachedProofStacks;
    using Prover::exportTheoremUsage;
    using Prover::exportCurrentChapterTheorem;
    using Prover::precompiledExternalsExport;
    using Prover::orCandidates;
    using Prover::orCandidatesMutex;
    using Prover::orPairsFromHeadSwitch;
    using Prover::orInRunScannedRows;
    using Prover::orBuiltByPair;
    using Prover::orInRunCompanions;
    using Prover::orElimInRunScannedRows;
    using Prover::orElimGuardIndex;
    using Prover::orElimPendingPairs;
    using Prover::orElimBuiltByPair;
    using Prover::orElimCitedOrByMerged;
    using Prover::dependenciesMutex;
    using Prover::theoremListMutex;
    using Prover::inductionMemoryBlocksMutex;
    using Prover::pendingTwinDeactivationsMutex;
    using Prover::pendingSubtreeDeactivationsMutex;
    using Prover::updateGlobalMutex;
    using Prover::updateGlobalSingleChapterMutex;
    using Prover::deferredAncestorAdmissionsMutex;
    using Prover::definitionExpansion;
    using Prover::compilationReuse;
    using Prover::structuralCompiler;
    using Prover::scopeTypes;
    using Prover::orRuleCompiler;
    using Prover::existenceRuleCompiler;
    using Prover::supportingRuleCompiler;
    using Prover::integrationRuleCompiler;
    using Prover::compilationPersistence;
    using Prover::deactivationSweepOwed;
    using Prover::admissionMemory;
    using Prover::integrationProcessor;
    using Prover::implicationSubproofs;
    using Prover::contradictionDemand;
    using Prover::rejectedSourceReplay;
    using Prover::hashRuleInstaller;
    using Prover::compactRuleProcessor;
    using Prover::statementMemory;
    using Prover::equivalenceProcessor;
    using Prover::equalityPropagation;
    using Prover::equivalenceAdmissionRewriter;
    using Prover::watchProcessor;
    using Prover::caseLedger;
    using Prover::caseProcessor;
    using Prover::disintegrationProcessor;
    using Prover::orImplicationProcessor;
    using Prover::proofHistory;
    using Prover::unaryIntegrationProcessor;
    using Prover::goalProcessor;
    using Prover::scopeLifecycle;
    using Prover::logicBlockOperations;
    using Prover::blockMailProcessor;
    using Prover::anchorScopeQueries;
    using Prover::equivalenceQueries;
    using Prover::blockKeyReader;
    using Prover::theoremPublisher;
    using Prover::inductionUsefulness;
    using Prover::proofGridBuilder;
    using Prover::orTheoremBuilder;
    using Prover::orCandidateProgress;
    using Prover::orEliminationBuilder;
    using Prover::counterexampleFilter;
    using Prover::encodedExpressionNormalizer;
    using Prover::proofReconstructor;
    using Prover::proofGraphWriter;
    using Prover::phase2Executor;
    using Prover::proofBarrier;
    using Prover::proverInitialization;
    using Prover::proofRunReporter;
    using Prover::performElemPhase1;
    using Prover::performElemPhase3;
    using Prover::proveKernel;
    using Prover::filterConjecturesWithCE;
    using Prover::startsWithStr;
};
}
