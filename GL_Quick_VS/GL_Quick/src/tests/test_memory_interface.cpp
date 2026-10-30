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
#include "test_harness.hpp"
#include "memory_layout_oracle.hpp"
#include <type_traits>
#include <utility>
namespace {
template<class T, class = void> struct Public_parentMemory : std::false_type {};
template<class T> struct Public_parentMemory<T, std::void_t<decltype(std::declval<T&>().parentMemory)>> : std::true_type {};
static_assert(Public_parentMemory<gl::Memory>::value);
template<class T, class = void> struct Public_nameMap : std::false_type {};
template<class T> struct Public_nameMap<T, std::void_t<decltype(std::declval<T&>().nameMap)>> : std::true_type {};
static_assert(Public_nameMap<gl::Memory>::value);
template<class T, class = void> struct Public_exprOriginMap : std::false_type {};
template<class T> struct Public_exprOriginMap<T, std::void_t<decltype(std::declval<T&>().exprOriginMap)>> : std::true_type {};
static_assert(Public_exprOriginMap<gl::Memory>::value);
template<class T, class = void> struct Public_isActive : std::false_type {};
template<class T> struct Public_isActive<T, std::void_t<decltype(std::declval<T&>().isActive)>> : std::true_type {};
static_assert(Public_isActive<gl::Memory>::value);
static_assert(sizeof(gl::Memory) == sizeof(gl::tests::MemoryLayoutOracle));
static_assert(alignof(gl::Memory) == alignof(gl::tests::MemoryLayoutOracle));
static_assert(offsetof(gl::Memory, startInt) == offsetof(gl::tests::MemoryLayoutOracle, startInt));
static_assert(offsetof(gl::Memory, startIntRepl) == offsetof(gl::tests::MemoryLayoutOracle, startIntRepl));
static_assert(offsetof(gl::Memory, startIntPi) == offsetof(gl::tests::MemoryLayoutOracle, startIntPi));
static_assert(offsetof(gl::Memory, expandedImplications) == offsetof(gl::tests::MemoryLayoutOracle, expandedImplications));
static_assert(offsetof(gl::Memory, compactExpansions) == offsetof(gl::tests::MemoryLayoutOracle, compactExpansions));
static_assert(offsetof(gl::Memory, expansionCarrierCount) == offsetof(gl::tests::MemoryLayoutOracle, expansionCarrierCount));
static_assert(offsetof(gl::Memory, exprKeyId) == offsetof(gl::tests::MemoryLayoutOracle, exprKeyId));
static_assert(offsetof(gl::Memory, varDefSetsId) == offsetof(gl::tests::MemoryLayoutOracle, varDefSetsId));
static_assert(offsetof(gl::Memory, recursionHypothesisId) == offsetof(gl::tests::MemoryLayoutOracle, recursionHypothesisId));
static_assert(offsetof(gl::Memory, recursionAuxyImplId) == offsetof(gl::tests::MemoryLayoutOracle, recursionAuxyImplId));
static_assert(offsetof(gl::Memory, premiseSplitFiled) == offsetof(gl::tests::MemoryLayoutOracle, premiseSplitFiled));
static_assert(offsetof(gl::Memory, weakStepBlock) == offsetof(gl::tests::MemoryLayoutOracle, weakStepBlock));
static_assert(offsetof(gl::Memory, strongStepBlock) == offsetof(gl::tests::MemoryLayoutOracle, strongStepBlock));
static_assert(offsetof(gl::Memory, inductionZeroId) == offsetof(gl::tests::MemoryLayoutOracle, inductionZeroId));
static_assert(offsetof(gl::Memory, twinPartnerId) == offsetof(gl::tests::MemoryLayoutOracle, twinPartnerId));
static_assert(offsetof(gl::Memory, twinZeroId) == offsetof(gl::tests::MemoryLayoutOracle, twinZeroId));
static_assert(offsetof(gl::Memory, twinSuccessorId) == offsetof(gl::tests::MemoryLayoutOracle, twinSuccessorId));
static_assert(offsetof(gl::Memory, twinExperiment) == offsetof(gl::tests::MemoryLayoutOracle, twinExperiment));
static_assert(offsetof(gl::Memory, twinCohortId) == offsetof(gl::tests::MemoryLayoutOracle, twinCohortId));
static_assert(offsetof(gl::Memory, twinCaseScopeId) == offsetof(gl::tests::MemoryLayoutOracle, twinCaseScopeId));
static_assert(offsetof(gl::Memory, twinValueDemandsFiled) == offsetof(gl::tests::MemoryLayoutOracle, twinValueDemandsFiled));
static_assert(offsetof(gl::Memory, twinFirstGoalId) == offsetof(gl::tests::MemoryLayoutOracle, twinFirstGoalId));
static_assert(offsetof(gl::Memory, twinPredecessorId) == offsetof(gl::tests::MemoryLayoutOracle, twinPredecessorId));
static_assert(offsetof(gl::Memory, twinFirstLevel) == offsetof(gl::tests::MemoryLayoutOracle, twinFirstLevel));
static_assert(offsetof(gl::Memory, twinSecondLevel) == offsetof(gl::tests::MemoryLayoutOracle, twinSecondLevel));
static_assert(offsetof(gl::Memory, recursionProductImplId) == offsetof(gl::tests::MemoryLayoutOracle, recursionProductImplId));
static_assert(offsetof(gl::Memory, parentMemory) == offsetof(gl::tests::MemoryLayoutOracle, parentMemory));
static_assert(offsetof(gl::Memory, level) == offsetof(gl::tests::MemoryLayoutOracle, level));
static_assert(offsetof(gl::Memory, overallHashMemory) == offsetof(gl::tests::MemoryLayoutOracle, overallHashMemory));
static_assert(offsetof(gl::Memory, localHashMemory) == offsetof(gl::tests::MemoryLayoutOracle, localHashMemory));
static_assert(offsetof(gl::Memory, localHashMemoryDelta) == offsetof(gl::tests::MemoryLayoutOracle, localHashMemoryDelta));
static_assert(offsetof(gl::Memory, workingMemory) == offsetof(gl::tests::MemoryLayoutOracle, workingMemory));
static_assert(offsetof(gl::Memory, equivalenceClassesMap) == offsetof(gl::tests::MemoryLayoutOracle, equivalenceClassesMap));
static_assert(offsetof(gl::Memory, canonicalEquivalenceClassesMap) == offsetof(gl::tests::MemoryLayoutOracle, canonicalEquivalenceClassesMap));
static_assert(offsetof(gl::Memory, changedClassesThisStep) == offsetof(gl::tests::MemoryLayoutOracle, changedClassesThisStep));
static_assert(offsetof(gl::Memory, admissionKeysAlgebra) == offsetof(gl::tests::MemoryLayoutOracle, admissionKeysAlgebra));
static_assert(offsetof(gl::Memory, deferredIntegrationPreps) == offsetof(gl::tests::MemoryLayoutOracle, deferredIntegrationPreps));
static_assert(offsetof(gl::Memory, admissionKeysOrdis2) == offsetof(gl::tests::MemoryLayoutOracle, admissionKeysOrdis2));
static_assert(offsetof(gl::Memory, integrationStartIntMap) == offsetof(gl::tests::MemoryLayoutOracle, integrationStartIntMap));
static_assert(offsetof(gl::Memory, mailIn) == offsetof(gl::tests::MemoryLayoutOracle, mailIn));
static_assert(offsetof(gl::Memory, mailOut) == offsetof(gl::tests::MemoryLayoutOracle, mailOut));
static_assert(offsetof(gl::Memory, sameIterationInternalMail) == offsetof(gl::tests::MemoryLayoutOracle, sameIterationInternalMail));
static_assert(offsetof(gl::Memory, nextIterationInternalMail) == offsetof(gl::tests::MemoryLayoutOracle, nextIterationInternalMail));
static_assert(offsetof(gl::Memory, replayMailDuringAbsorb) == offsetof(gl::tests::MemoryLayoutOracle, replayMailDuringAbsorb));
static_assert(offsetof(gl::Memory, eqClassSttmntIndexMapMap) == offsetof(gl::tests::MemoryLayoutOracle, eqClassSttmntIndexMapMap));
static_assert(offsetof(gl::Memory, isActive) == offsetof(gl::tests::MemoryLayoutOracle, isActive));
static_assert(offsetof(gl::Memory, phase3IgnoredOrRoots) == offsetof(gl::tests::MemoryLayoutOracle, phase3IgnoredOrRoots));
static_assert(offsetof(gl::Memory, ruleStagingArmDepth) == offsetof(gl::tests::MemoryLayoutOracle, ruleStagingArmDepth));
static_assert(offsetof(gl::Memory, claimHolder) == offsetof(gl::tests::MemoryLayoutOracle, claimHolder));
static_assert(offsetof(gl::Memory, hasWork) == offsetof(gl::tests::MemoryLayoutOracle, hasWork));
static_assert(offsetof(gl::Memory, mailOutPending) == offsetof(gl::tests::MemoryLayoutOracle, mailOutPending));
static_assert(offsetof(gl::Memory, mailOutLiveBytes) == offsetof(gl::tests::MemoryLayoutOracle, mailOutLiveBytes));
static_assert(offsetof(gl::Memory, parkedSourcePeakBytes) == offsetof(gl::tests::MemoryLayoutOracle, parkedSourcePeakBytes));
static_assert(offsetof(gl::Memory, mutatedThisBurst) == offsetof(gl::tests::MemoryLayoutOracle, mutatedThisBurst));
static_assert(offsetof(gl::Memory, deferredExpandedErase) == offsetof(gl::tests::MemoryLayoutOracle, deferredExpandedErase));
static_assert(offsetof(gl::Memory, encodedCountAtBurstStart) == offsetof(gl::tests::MemoryLayoutOracle, encodedCountAtBurstStart));
static_assert(offsetof(gl::Memory, shadowWouldSkip) == offsetof(gl::tests::MemoryLayoutOracle, shadowWouldSkip));
static_assert(offsetof(gl::Memory, numberOfParts) == offsetof(gl::tests::MemoryLayoutOracle, numberOfParts));
static_assert(offsetof(gl::Memory, justActivated) == offsetof(gl::tests::MemoryLayoutOracle, justActivated));
static_assert(offsetof(gl::Memory, phase2TaskUsage) == offsetof(gl::tests::MemoryLayoutOracle, phase2TaskUsage));
static_assert(offsetof(gl::Memory, phase2SelectedHashMemories) == offsetof(gl::tests::MemoryLayoutOracle, phase2SelectedHashMemories));
static_assert(offsetof(gl::Memory, phase2ProjectionUsage) == offsetof(gl::tests::MemoryLayoutOracle, phase2ProjectionUsage));
static_assert(offsetof(gl::Memory, phase2MeasurementEpoch) == offsetof(gl::tests::MemoryLayoutOracle, phase2MeasurementEpoch));
static_assert(offsetof(gl::Memory, isPartOfRecursion) == offsetof(gl::tests::MemoryLayoutOracle, isPartOfRecursion));
static_assert(offsetof(gl::Memory, deltaNumberStatements) == offsetof(gl::tests::MemoryLayoutOracle, deltaNumberStatements));
static_assert(offsetof(gl::Memory, exprOriginMap) == offsetof(gl::tests::MemoryLayoutOracle, exprOriginMap));
static_assert(offsetof(gl::Memory, recursionCounter) == offsetof(gl::tests::MemoryLayoutOracle, recursionCounter));
static_assert(offsetof(gl::Memory, contradictionIndex) == offsetof(gl::tests::MemoryLayoutOracle, contradictionIndex));
static_assert(offsetof(gl::Memory, integrationPrepared) == offsetof(gl::tests::MemoryLayoutOracle, integrationPrepared));
static_assert(offsetof(gl::Memory, integrationPreparedMarker) == offsetof(gl::tests::MemoryLayoutOracle, integrationPreparedMarker));
static_assert(offsetof(gl::Memory, pendingWipeScopes) == offsetof(gl::tests::MemoryLayoutOracle, pendingWipeScopes));
static_assert(offsetof(gl::Memory, intWeakVariables) == offsetof(gl::tests::MemoryLayoutOracle, intWeakVariables));
static_assert(offsetof(gl::Memory, templateInterner) == offsetof(gl::tests::MemoryLayoutOracle, templateInterner));
static_assert(offsetof(gl::Memory, valueInterner) == offsetof(gl::tests::MemoryLayoutOracle, valueInterner));
static_assert(offsetof(gl::Memory, originInterner) == offsetof(gl::tests::MemoryLayoutOracle, originInterner));
static_assert(offsetof(gl::Memory, mailOutInterner) == offsetof(gl::tests::MemoryLayoutOracle, mailOutInterner));
static_assert(offsetof(gl::Memory, ruleInterner) == offsetof(gl::tests::MemoryLayoutOracle, ruleInterner));
static_assert(offsetof(gl::Memory, lbStateInterner) == offsetof(gl::tests::MemoryLayoutOracle, lbStateInterner));
static_assert(offsetof(gl::Memory, eqClassNameCaches) == offsetof(gl::tests::MemoryLayoutOracle, eqClassNameCaches));
static_assert(offsetof(gl::Memory, primedForContradiction) == offsetof(gl::tests::MemoryLayoutOracle, primedForContradiction));
static_assert(offsetof(gl::Memory, goalPending) == offsetof(gl::tests::MemoryLayoutOracle, goalPending));
static_assert(offsetof(gl::Memory, inductionDoubleKey) == offsetof(gl::tests::MemoryLayoutOracle, inductionDoubleKey));
static_assert(offsetof(gl::Memory, inductionParameterDigits) == offsetof(gl::tests::MemoryLayoutOracle, inductionParameterDigits));
static_assert(offsetof(gl::Memory, inductionParameterDigitCount) == offsetof(gl::tests::MemoryLayoutOracle, inductionParameterDigitCount));
static_assert(offsetof(gl::Memory, inductionVariableDigits) == offsetof(gl::tests::MemoryLayoutOracle, inductionVariableDigits));
static_assert(offsetof(gl::Memory, inductionVariableDigitCount) == offsetof(gl::tests::MemoryLayoutOracle, inductionVariableDigitCount));
static_assert(offsetof(gl::Memory, inductionDoubleChainId) == offsetof(gl::tests::MemoryLayoutOracle, inductionDoubleChainId));
static_assert(offsetof(gl::Memory, inductionProductKey) == offsetof(gl::tests::MemoryLayoutOracle, inductionProductKey));
static_assert(offsetof(gl::Memory, inductionProductChainId) == offsetof(gl::tests::MemoryLayoutOracle, inductionProductChainId));
static_assert(offsetof(gl::Memory, usefulnessTest) == offsetof(gl::tests::MemoryLayoutOracle, usefulnessTest));
static_assert(offsetof(gl::Memory, usefulnessOpenSlot) == offsetof(gl::tests::MemoryLayoutOracle, usefulnessOpenSlot));
static_assert(offsetof(gl::Memory, usefulnessScopeId) == offsetof(gl::tests::MemoryLayoutOracle, usefulnessScopeId));
static_assert(offsetof(gl::Memory, usefulnessHeadId) == offsetof(gl::tests::MemoryLayoutOracle, usefulnessHeadId));
static_assert(offsetof(gl::Memory, usefulnessKeyId) == offsetof(gl::tests::MemoryLayoutOracle, usefulnessKeyId));
static_assert(offsetof(gl::Memory, contradictionTheoremId) == offsetof(gl::tests::MemoryLayoutOracle, contradictionTheoremId));
static_assert(offsetof(gl::Memory, mainContradiction) == offsetof(gl::tests::MemoryLayoutOracle, mainContradiction));
static_assert(offsetof(gl::Memory, orBookkeeping) == offsetof(gl::tests::MemoryLayoutOracle, orBookkeeping));
static_assert(offsetof(gl::Memory, orDisjunctCount) == offsetof(gl::tests::MemoryLayoutOracle, orDisjunctCount));
static_assert(offsetof(gl::Memory, orPendingBranches) == offsetof(gl::tests::MemoryLayoutOracle, orPendingBranches));
static_assert(offsetof(gl::Memory, orPendingLevels) == offsetof(gl::tests::MemoryLayoutOracle, orPendingLevels));
static_assert(offsetof(gl::Memory, pendingRelayIter) == offsetof(gl::tests::MemoryLayoutOracle, pendingRelayIter));
static_assert(offsetof(gl::Memory, pendingRelayLevels) == offsetof(gl::tests::MemoryLayoutOracle, pendingRelayLevels));
static_assert(offsetof(gl::Memory, nameMap) == offsetof(gl::tests::MemoryLayoutOracle, nameMap));
static_assert(offsetof(gl::Memory, lbMemory) == offsetof(gl::tests::MemoryLayoutOracle, lbMemory));
static_assert(offsetof(gl::Memory, persistentDirty) == offsetof(gl::tests::MemoryLayoutOracle, persistentDirty));
static_assert(offsetof(gl::Memory, persistentArena) == offsetof(gl::tests::MemoryLayoutOracle, persistentArena));
static_assert(offsetof(gl::Memory, intEncodedStatements) == offsetof(gl::tests::MemoryLayoutOracle, intEncodedStatements));
static_assert(offsetof(gl::Memory, deloadFiles) == offsetof(gl::tests::MemoryLayoutOracle, deloadFiles));
static_assert(offsetof(gl::Memory, deloadChain) == offsetof(gl::tests::MemoryLayoutOracle, deloadChain));
static_assert(offsetof(gl::Memory, deloadOrdinal) == offsetof(gl::tests::MemoryLayoutOracle, deloadOrdinal));
static_assert(offsetof(gl::Memory, deloadKind) == offsetof(gl::tests::MemoryLayoutOracle, deloadKind));
static_assert(offsetof(gl::Memory, lastRawImageBytes_) == offsetof(gl::tests::MemoryLayoutOracle, lastRawImageBytes_));
static_assert(offsetof(gl::Memory, rawExtentOffset_) == offsetof(gl::tests::MemoryLayoutOracle, rawExtentOffset_));
static_assert(offsetof(gl::Memory, rawExtentClassBytes_) == offsetof(gl::tests::MemoryLayoutOracle, rawExtentClassBytes_));
static_assert(offsetof(gl::Memory, rawExtentEpoch_) == offsetof(gl::tests::MemoryLayoutOracle, rawExtentEpoch_));
static_assert(offsetof(gl::Memory, lastExtentLiveBytes_) == offsetof(gl::tests::MemoryLayoutOracle, lastExtentLiveBytes_));
static_assert(offsetof(gl::Memory, deloadBaseRows) == offsetof(gl::tests::MemoryLayoutOracle, deloadBaseRows));
static_assert(offsetof(gl::Memory, deloadTailRows) == offsetof(gl::tests::MemoryLayoutOracle, deloadTailRows));
static_assert(offsetof(gl::Memory, deloadTailCount) == offsetof(gl::tests::MemoryLayoutOracle, deloadTailCount));
static_assert(offsetof(gl::Memory, deloadedCounts) == offsetof(gl::tests::MemoryLayoutOracle, deloadedCounts));
static_assert(offsetof(gl::Memory, dischargedForever) == offsetof(gl::tests::MemoryLayoutOracle, dischargedForever));
static_assert(offsetof(gl::Memory, dischargedRegistryKeys) == offsetof(gl::tests::MemoryLayoutOracle, dischargedRegistryKeys));
static_assert(offsetof(gl::Memory, stewardClaim) == offsetof(gl::tests::MemoryLayoutOracle, stewardClaim));
static_assert(offsetof(gl::Memory, intLocalEncodedStatements) == offsetof(gl::tests::MemoryLayoutOracle, intLocalEncodedStatements));
static_assert(offsetof(gl::Memory, intLocalEncodedStatementsDelta) == offsetof(gl::tests::MemoryLayoutOracle, intLocalEncodedStatementsDelta));
static_assert(offsetof(gl::Memory, intExternalStatements) == offsetof(gl::tests::MemoryLayoutOracle, intExternalStatements));
static_assert(offsetof(gl::Memory, intKnownStatements) == offsetof(gl::tests::MemoryLayoutOracle, intKnownStatements));
static_assert(offsetof(gl::Memory, orImplications) == offsetof(gl::tests::MemoryLayoutOracle, orImplications));
static_assert(offsetof(gl::Memory, existenceReplays) == offsetof(gl::tests::MemoryLayoutOracle, existenceReplays));
static_assert(offsetof(gl::Memory, orImplicationMatches) == offsetof(gl::tests::MemoryLayoutOracle, orImplicationMatches));
static_assert(offsetof(gl::Memory, intLocalEncodedStatementsSet) == offsetof(gl::tests::MemoryLayoutOracle, intLocalEncodedStatementsSet));
static_assert(offsetof(gl::Memory, intStatementLevelsMap) == offsetof(gl::tests::MemoryLayoutOracle, intStatementLevelsMap));
static_assert(offsetof(gl::Memory, intToBeProved) == offsetof(gl::tests::MemoryLayoutOracle, intToBeProved));
static_assert(offsetof(gl::Memory, pendingDisprovedGoals) == offsetof(gl::tests::MemoryLayoutOracle, pendingDisprovedGoals));
static_assert(offsetof(gl::Memory, pendingDeadOrBranches) == offsetof(gl::tests::MemoryLayoutOracle, pendingDeadOrBranches));
static_assert(offsetof(gl::Memory, orLiveBranches) == offsetof(gl::tests::MemoryLayoutOracle, orLiveBranches));
static_assert(offsetof(gl::Memory, frozenOrBranches) == offsetof(gl::tests::MemoryLayoutOracle, frozenOrBranches));
static_assert(offsetof(gl::Memory, recursionHeadWitnesses) == offsetof(gl::tests::MemoryLayoutOracle, recursionHeadWitnesses));
static_assert(offsetof(gl::Memory, hiddenArgumentWitnesses) == offsetof(gl::tests::MemoryLayoutOracle, hiddenArgumentWitnesses));
static_assert(offsetof(gl::Memory, builtOnListedWitnesses) == offsetof(gl::tests::MemoryLayoutOracle, builtOnListedWitnesses));
static_assert(offsetof(gl::Memory, subproofGoals) == offsetof(gl::tests::MemoryLayoutOracle, subproofGoals));
static_assert(offsetof(gl::Memory, orRetiredDisjuncts) == offsetof(gl::tests::MemoryLayoutOracle, orRetiredDisjuncts));
static_assert(offsetof(gl::Memory, pendingOrReleases) == offsetof(gl::tests::MemoryLayoutOracle, pendingOrReleases));
static_assert(offsetof(gl::Memory, orStarterPick) == offsetof(gl::tests::MemoryLayoutOracle, orStarterPick));
static_assert(offsetof(gl::Memory, processedOrLedger) == offsetof(gl::tests::MemoryLayoutOracle, processedOrLedger));
static_assert(offsetof(gl::Memory, negationLedger) == offsetof(gl::tests::MemoryLayoutOracle, negationLedger));
static_assert(offsetof(gl::Memory, goalLedger) == offsetof(gl::tests::MemoryLayoutOracle, goalLedger));
static_assert(offsetof(gl::Memory, goalLedgerEpoch) == offsetof(gl::tests::MemoryLayoutOracle, goalLedgerEpoch));
static_assert(offsetof(gl::Memory, goalLedgerCount) == offsetof(gl::tests::MemoryLayoutOracle, goalLedgerCount));
static_assert(offsetof(gl::Memory, goalLedgerStale) == offsetof(gl::tests::MemoryLayoutOracle, goalLedgerStale));
static_assert(offsetof(gl::Memory, pendingWatchArrivals) == offsetof(gl::tests::MemoryLayoutOracle, pendingWatchArrivals));
static_assert(offsetof(gl::Memory, pendingDeferredOrEvents) == offsetof(gl::tests::MemoryLayoutOracle, pendingDeferredOrEvents));
static_assert(offsetof(gl::Memory, deferredOrWaiting) == offsetof(gl::tests::MemoryLayoutOracle, deferredOrWaiting));
static_assert(offsetof(gl::Memory, pendingWatchRechecks) == offsetof(gl::tests::MemoryLayoutOracle, pendingWatchRechecks));
static_assert(offsetof(gl::Memory, pendingSubproofArrivals) == offsetof(gl::tests::MemoryLayoutOracle, pendingSubproofArrivals));
static_assert(offsetof(gl::Memory, pendingSubproofRechecks) == offsetof(gl::tests::MemoryLayoutOracle, pendingSubproofRechecks));
static_assert(offsetof(gl::Memory, pendingContradictionArrivals) == offsetof(gl::tests::MemoryLayoutOracle, pendingContradictionArrivals));
static_assert(offsetof(gl::Memory, pendingContradictionDemands) == offsetof(gl::tests::MemoryLayoutOracle, pendingContradictionDemands));
static_assert(offsetof(gl::Memory, intValidityNamesToFilter) == offsetof(gl::tests::MemoryLayoutOracle, intValidityNamesToFilter));
static_assert(offsetof(gl::Memory, intAxedVariables) == offsetof(gl::tests::MemoryLayoutOracle, intAxedVariables));
}
TEST(memory_interface, preserves_original_field_layout) {
ASSERT_EQ(sizeof(gl::Memory), sizeof(gl::tests::MemoryLayoutOracle));
ASSERT_EQ(alignof(gl::Memory), alignof(gl::tests::MemoryLayoutOracle));
}
