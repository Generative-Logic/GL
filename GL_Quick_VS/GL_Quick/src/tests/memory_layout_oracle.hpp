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
#include "../memory.hpp"
namespace gl::tests {
/// Original logical-block declarations retained only as a storage-layout oracle.
struct MemoryLayoutOracle {
using DeloadKind = Memory::DeloadKind;
using StewardClaim = Memory::StewardClaim;
int startInt;
int startIntRepl;
int startIntPi;
TypedColdSet<LbStatePairKey>& expandedImplications = lbMemory.expandedImplications;
TypedColdBlobMap<int64_t, CompactExpansionRec>& compactExpansions = lbMemory.compactExpansions;
ColdHashMap<PodKeyStore<int64_t>, int32_t>& expansionCarrierCount = lbMemory.expansionCarrierCount;
int32_t exprKeyId = 0;
int32_t varDefSetsId = 0;
NameId recursionHypothesisId = 0;
int32_t recursionAuxyImplId = 0;
bool premiseSplitFiled = false;
bool weakStepBlock = false;
bool strongStepBlock = false;
NameId inductionZeroId = 0;
NameId twinPartnerId = 0;
NameId twinZeroId = 0;
NameId twinSuccessorId = 0;
TwinExperiment twinExperiment = TwinExperiment::none;
int32_t twinCohortId = 0;
NameId twinCaseScopeId = 0;
bool twinValueDemandsFiled = false;
NameId twinFirstGoalId = 0;
NameId twinPredecessorId = 0;
int32_t twinFirstLevel = 0;
int32_t twinSecondLevel = 0;
int32_t recursionProductImplId = 0;
Memory* parentMemory;
int level;
HashMemory& overallHashMemory = lbMemory.overallHashMemory;
HashMemory& localHashMemory = lbMemory.localHashMemory;
HashMemory& localHashMemoryDelta = lbMemory.localHashMemoryDelta;
HashMemory& workingMemory = lbMemory.workingMemory;
TypedColdBlobMap<NameId, EquivalenceClass>& equivalenceClassesMap = lbMemory.equivalenceClassesMap;
TypedColdBlobMap<NameId, EquivalenceClass>& canonicalEquivalenceClassesMap = lbMemory.canonicalEquivalenceClassesMap;
ChangedClassesBuffer& changedClassesThisStep = lbMemory.changedClassesThisStep;
std::vector<AdmissionKeyAlgebraRecord> admissionKeysAlgebra;
std::vector<DeferredIntegrationPrep> deferredIntegrationPreps;
std::vector<Ordis2DemandRecord> admissionKeysOrdis2;
ColdHashMap<PodKeyStore<NameId>, int>& integrationStartIntMap = lbMemory.integrationStartIntMap;
RoutingColdMail mailIn;
DeloadableMailOut& mailOut = lbMemory.mailOut;
ColdMail& sameIterationInternalMail = lbMemory.sameInternalMail;
ColdMail& nextIterationInternalMail = lbMemory.nextInternalMail;
ColdMail* replayMailDuringAbsorb = nullptr;
TypedColdMap<EqClassKey, int>& eqClassSttmntIndexMapMap = lbMemory.eqClassSttmntIndexMapMap;
bool isActive;
const ColdHashSet<PodKeyStore<NameId>>* phase3IgnoredOrRoots = nullptr;
int32_t ruleStagingArmDepth = 0;
std::atomic<std::thread::id> claimHolder{};
bool hasWork = true;
bool mailOutPending = false;
int64_t mailOutLiveBytes = 0;
int64_t parkedSourcePeakBytes = 0;
bool mutatedThisBurst = false;
DeferredExpandedErase* deferredExpandedErase = nullptr;
int encodedCountAtBurstStart = 0;
bool shadowWouldSkip = false;
int numberOfParts = 1;
bool justActivated = false;
gpu::Phase2TaskProjectionUsage phase2TaskUsage{};
uint32_t phase2SelectedHashMemories = 0;
gpu::Phase2ProjectionUsage phase2ProjectionUsage{};
uint64_t phase2MeasurementEpoch = 0;
bool isPartOfRecursion;
int deltaNumberStatements;
TypedColdBlobMap<int64_t, IdOrigin>& exprOriginMap = lbMemory.exprOriginMap;
int recursionCounter;
int contradictionIndex;
ColdHashSet<PodKeyStore<int64_t>>& integrationPrepared = lbMemory.integrationPrepared;
ColdHashSet<PodKeyStore<int64_t>>& integrationPreparedMarker = lbMemory.integrationPreparedMarker;
ColdHashSet<PodKeyStore<NameId>>& pendingWipeScopes = lbMemory.pendingWipeScopes;
ColdHashSet<PodKeyStore<int64_t>>& intWeakVariables = lbMemory.intWeakVariables;
TemplateInterner templateInterner;
ValueInterner valueInterner;
ValueInterner originInterner;
ValueInterner mailOutInterner;
ValueInterner ruleInterner;
ValueInterner lbStateInterner;
EqClassNameCaches& eqClassNameCaches = lbMemory.eqClassNameCaches;
bool primedForContradiction = false;
bool goalPending = false;
int64_t inductionDoubleKey = 0;
int32_t inductionParameterDigits[ExecutionParameters::MAX_ARITY] = {};
int32_t inductionParameterDigitCount = 0;
int32_t inductionVariableDigits[ExecutionParameters::MAX_ARITY] = {};
int32_t inductionVariableDigitCount = 0;
NameId inductionDoubleChainId = 0;
int64_t inductionProductKey = 0;
NameId inductionProductChainId = 0;
Memory::UsefulnessTest usefulnessTest = Memory::UsefulnessTest::None;
int32_t usefulnessOpenSlot = -1;
NameId usefulnessScopeId = 0;
NameId usefulnessHeadId = 0;
NameId usefulnessKeyId = 0;
NameId contradictionTheoremId = 0;
bool mainContradiction = false;
TypedColdSetMap<LbStatePairKey, int32_t>& orBookkeeping = lbMemory.orBookkeeping;
ColdHashMap<PodKeyStore<int32_t>, int>& orDisjunctCount = lbMemory.orDisjunctCount;
TypedColdSetMap<int32_t, int32_t>& orPendingBranches = lbMemory.orPendingBranches;
TypedColdSetMap<int32_t, int32_t>& orPendingLevels = lbMemory.orPendingLevels;
TypedColdMap<int32_t, int32_t>& pendingRelayIter = lbMemory.pendingRelayIter;
TypedColdSetMap<int32_t, int32_t>& pendingRelayLevels = lbMemory.pendingRelayLevels;
NameMap nameMap;
LbMemory lbMemory{ &staticMemory() };
DirtyState persistentDirty = DirtyState::Clean;
LbArena persistentArena{ &persistentMemory() };
SharedExpressionList& intEncodedStatements = lbMemory.intEncodedStatements;
std::vector<std::string> deloadFiles;
std::string deloadChain;
int64_t deloadOrdinal = -1;
DeloadKind deloadKind = DeloadKind::Canonical;
int64_t lastRawImageBytes_ = 0;
int64_t rawExtentOffset_ = -1;
int32_t rawExtentClassBytes_ = 0;
int64_t rawExtentEpoch_ = -1;
int64_t lastExtentLiveBytes_ = 0;
int64_t deloadBaseRows = 0;
int64_t deloadTailRows = 0;
int32_t deloadTailCount = 0;
std::vector<int32_t> deloadedCounts;
bool dischargedForever = false;
std::unordered_set<int64_t> dischargedRegistryKeys;
std::atomic<uint8_t> stewardClaim{ static_cast<uint8_t>(StewardClaim::Idle) };
SharedExpressionList& intLocalEncodedStatements = lbMemory.intLocalEncodedStatements;
SharedExpressionList& intLocalEncodedStatementsDelta = lbMemory.intLocalEncodedStatementsDelta;
SharedExpressionList& intExternalStatements = lbMemory.intExternalStatements;
TypedColdMap<StatementKey, StatementFlags>& intKnownStatements = lbMemory.intKnownStatements;
OrImplications& orImplications = lbMemory.orImplications;
TypedColdMap<int64_t, int32_t>& existenceReplays = lbMemory.existenceReplays;
OrImplicationMatchIndex& orImplicationMatches = lbMemory.orImplicationMatches;
ColdHashSet<PodKeyStore<int64_t>>& intLocalEncodedStatementsSet = lbMemory.intLocalEncodedStatementsSet;
TypedColdSetMap<StatementKey, int>& intStatementLevelsMap = lbMemory.intStatementLevelsMap;
TypedColdSetMap<StatementKey, int> intToBeProved{ &persistentArena, &persistentDirty };
ColdHashSet<BytesKeyStore> pendingDisprovedGoals{ &persistentArena, &persistentDirty };
ColdHashSet<PodKeyStore<NameId>> pendingDeadOrBranches{ &persistentArena, &persistentDirty };
ColdHashSet<PodKeyStore<NameId>> orLiveBranches{ &persistentArena, &persistentDirty };
ColdHashSet<PodKeyStore<NameId>> frozenOrBranches{ &persistentArena, &persistentDirty };
ColdHashSet<BytesKeyStore> recursionHeadWitnesses{ &persistentArena, &persistentDirty };
ColdHashSet<BytesKeyStore> hiddenArgumentWitnesses;
ColdHashSet<BytesKeyStore> builtOnListedWitnesses;
TypedColdMap<int64_t, int64_t> subproofGoals{ &persistentArena, &persistentDirty };
TypedColdSetMap<int32_t, int64_t> orRetiredDisjuncts{ &persistentArena, &persistentDirty };
ColdHashSet<PodKeyStore<int32_t>> pendingOrReleases{ &persistentArena, &persistentDirty };
TypedColdMap<int32_t, int32_t> orStarterPick{ &persistentArena, &persistentDirty };
TypedColdSetMap<NameId, int64_t> processedOrLedger{ &persistentArena, &persistentDirty };
TypedColdSetMap<int64_t, int64_t> negationLedger{ &persistentArena, &persistentDirty };
TypedColdSetMap<int64_t, int64_t> goalLedger{ &persistentArena, &persistentDirty };
uint32_t goalLedgerEpoch = 0;
int32_t goalLedgerCount = -1;
bool goalLedgerStale = true;
PagedVector<int64_t> pendingWatchArrivals{ &persistentArena, &persistentDirty };
PagedVector<int64_t> pendingDeferredOrEvents{ &persistentArena, &persistentDirty };
TypedColdSetMap<int32_t, int64_t> deferredOrWaiting{ &persistentArena, &persistentDirty };
PagedVector<int64_t> pendingWatchRechecks{ &persistentArena, &persistentDirty };
PagedVector<int64_t> pendingSubproofArrivals{ &persistentArena, &persistentDirty };
PagedVector<NameId> pendingSubproofRechecks{ &persistentArena, &persistentDirty };
PagedVector<int64_t> pendingContradictionArrivals{ &persistentArena, &persistentDirty };
PagedVector<int64_t> pendingContradictionDemands{ &persistentArena, &persistentDirty };
ColdHashSet<PodKeyStore<NameId>>& intValidityNamesToFilter = lbMemory.intValidityNamesToFilter;
ColdHashSet<PodKeyStore<NameId>>& intAxedVariables = lbMemory.intAxedVariables;
};
}
