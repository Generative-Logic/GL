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
#include "hash_memory_layout_oracle.hpp"
#include <type_traits>
#include <utility>
namespace {
template<class T, class = void> struct Public_encodedMap : std::false_type {};
template<class T> struct Public_encodedMap<T, std::void_t<decltype(std::declval<T&>().encodedMap)>> : std::true_type {};
static_assert(Public_encodedMap<gl::HashMemory>::value);
template<class T, class = void> struct Public_admissionMap : std::false_type {};
template<class T> struct Public_admissionMap<T, std::void_t<decltype(std::declval<T&>().admissionMap)>> : std::true_type {};
static_assert(Public_admissionMap<gl::HashMemory>::value);
template<class T, class = void> struct Public_rejectedMap : std::false_type {};
template<class T> struct Public_rejectedMap<T, std::void_t<decltype(std::declval<T&>().rejectedMap)>> : std::true_type {};
static_assert(Public_rejectedMap<gl::HashMemory>::value);
static_assert(sizeof(gl::HashMemory) == sizeof(gl::tests::HashMemoryLayoutOracle));
static_assert(alignof(gl::HashMemory) == alignof(gl::tests::HashMemoryLayoutOracle));
static_assert(offsetof(gl::HashMemory, arena_) == offsetof(gl::tests::HashMemoryLayoutOracle, arena_));
static_assert(offsetof(gl::HashMemory, dirty_) == offsetof(gl::tests::HashMemoryLayoutOracle, dirty_));
static_assert(offsetof(gl::HashMemory, encodedMap) == offsetof(gl::tests::HashMemoryLayoutOracle, encodedMap));
static_assert(offsetof(gl::HashMemory, encodedMapLengths) == offsetof(gl::tests::HashMemoryLayoutOracle, encodedMapLengths));
static_assert(offsetof(gl::HashMemory, encodedMapBytes) == offsetof(gl::tests::HashMemoryLayoutOracle, encodedMapBytes));
static_assert(offsetof(gl::HashMemory, encodedMapRunStarts) == offsetof(gl::tests::HashMemoryLayoutOracle, encodedMapRunStarts));
static_assert(offsetof(gl::HashMemory, encodedMapBlobStarts) == offsetof(gl::tests::HashMemoryLayoutOracle, encodedMapBlobStarts));
static_assert(offsetof(gl::HashMemory, encodedMapBlobPool) == offsetof(gl::tests::HashMemoryLayoutOracle, encodedMapBlobPool));
static_assert(offsetof(gl::HashMemory, remainingArgsNormalizedEncodedMap) == offsetof(gl::tests::HashMemoryLayoutOracle, remainingArgsNormalizedEncodedMap));
static_assert(offsetof(gl::HashMemory, remainingArgsNormalizedEncodedMapLengths) == offsetof(gl::tests::HashMemoryLayoutOracle, remainingArgsNormalizedEncodedMapLengths));
static_assert(offsetof(gl::HashMemory, remainingArgsNormalizedEncodedMapBytes) == offsetof(gl::tests::HashMemoryLayoutOracle, remainingArgsNormalizedEncodedMapBytes));
static_assert(offsetof(gl::HashMemory, remainingArgsNormalizedEncodedMapRunStarts) == offsetof(gl::tests::HashMemoryLayoutOracle, remainingArgsNormalizedEncodedMapRunStarts));
static_assert(offsetof(gl::HashMemory, remainingArgsNormalizedEncodedMapBlobStarts) == offsetof(gl::tests::HashMemoryLayoutOracle, remainingArgsNormalizedEncodedMapBlobStarts));
static_assert(offsetof(gl::HashMemory, remainingArgsNormalizedEncodedMapBlobPool) == offsetof(gl::tests::HashMemoryLayoutOracle, remainingArgsNormalizedEncodedMapBlobPool));
static_assert(offsetof(gl::HashMemory, remainingArgsReverseIndex) == offsetof(gl::tests::HashMemoryLayoutOracle, remainingArgsReverseIndex));
static_assert(offsetof(gl::HashMemory, originalsHeadIndex) == offsetof(gl::tests::HashMemoryLayoutOracle, originalsHeadIndex));
static_assert(offsetof(gl::HashMemory, normalizedEncodedKeys) == offsetof(gl::tests::HashMemoryLayoutOracle, normalizedEncodedKeys));
static_assert(offsetof(gl::HashMemory, normalizedEncodedKeysLengths) == offsetof(gl::tests::HashMemoryLayoutOracle, normalizedEncodedKeysLengths));
static_assert(offsetof(gl::HashMemory, normalizedEncodedKeysBytes) == offsetof(gl::tests::HashMemoryLayoutOracle, normalizedEncodedKeysBytes));
static_assert(offsetof(gl::HashMemory, normalizedEncodedKeysRunStarts) == offsetof(gl::tests::HashMemoryLayoutOracle, normalizedEncodedKeysRunStarts));
static_assert(offsetof(gl::HashMemory, normalizedEncodedKeysBlobStarts) == offsetof(gl::tests::HashMemoryLayoutOracle, normalizedEncodedKeysBlobStarts));
static_assert(offsetof(gl::HashMemory, normalizedEncodedKeysBlobPool) == offsetof(gl::tests::HashMemoryLayoutOracle, normalizedEncodedKeysBlobPool));
static_assert(offsetof(gl::HashMemory, normalizedEncodedSubkeys) == offsetof(gl::tests::HashMemoryLayoutOracle, normalizedEncodedSubkeys));
static_assert(offsetof(gl::HashMemory, normalizedEncodedSubkeysLengths) == offsetof(gl::tests::HashMemoryLayoutOracle, normalizedEncodedSubkeysLengths));
static_assert(offsetof(gl::HashMemory, normalizedEncodedSubkeysBytes) == offsetof(gl::tests::HashMemoryLayoutOracle, normalizedEncodedSubkeysBytes));
static_assert(offsetof(gl::HashMemory, normalizedEncodedSubkeysRunStarts) == offsetof(gl::tests::HashMemoryLayoutOracle, normalizedEncodedSubkeysRunStarts));
static_assert(offsetof(gl::HashMemory, normalizedEncodedSubkeysBlobStarts) == offsetof(gl::tests::HashMemoryLayoutOracle, normalizedEncodedSubkeysBlobStarts));
static_assert(offsetof(gl::HashMemory, normalizedEncodedSubkeysBlobPool) == offsetof(gl::tests::HashMemoryLayoutOracle, normalizedEncodedSubkeysBlobPool));
static_assert(offsetof(gl::HashMemory, maxKeyLength) == offsetof(gl::tests::HashMemoryLayoutOracle, maxKeyLength));
static_assert(offsetof(gl::HashMemory, ownerlessPending) == offsetof(gl::tests::HashMemoryLayoutOracle, ownerlessPending));
static_assert(offsetof(gl::HashMemory, goalScanChainMark) == offsetof(gl::tests::HashMemoryLayoutOracle, goalScanChainMark));
static_assert(offsetof(gl::HashMemory, contradictionChainMark) == offsetof(gl::tests::HashMemoryLayoutOracle, contradictionChainMark));
static_assert(offsetof(gl::HashMemory, originals) == offsetof(gl::tests::HashMemoryLayoutOracle, originals));
static_assert(offsetof(gl::HashMemory, originalsLengths) == offsetof(gl::tests::HashMemoryLayoutOracle, originalsLengths));
static_assert(offsetof(gl::HashMemory, originalsBytes) == offsetof(gl::tests::HashMemoryLayoutOracle, originalsBytes));
static_assert(offsetof(gl::HashMemory, originalsRunStarts) == offsetof(gl::tests::HashMemoryLayoutOracle, originalsRunStarts));
static_assert(offsetof(gl::HashMemory, originalsBlobStarts) == offsetof(gl::tests::HashMemoryLayoutOracle, originalsBlobStarts));
static_assert(offsetof(gl::HashMemory, originalsBlobPool) == offsetof(gl::tests::HashMemoryLayoutOracle, originalsBlobPool));
static_assert(offsetof(gl::HashMemory, remainingArgsOwners) == offsetof(gl::tests::HashMemoryLayoutOracle, remainingArgsOwners));
static_assert(offsetof(gl::HashMemory, remainingArgsOwnersLengths) == offsetof(gl::tests::HashMemoryLayoutOracle, remainingArgsOwnersLengths));
static_assert(offsetof(gl::HashMemory, remainingArgsOwnersBytes) == offsetof(gl::tests::HashMemoryLayoutOracle, remainingArgsOwnersBytes));
static_assert(offsetof(gl::HashMemory, remainingArgsOwnersRunStarts) == offsetof(gl::tests::HashMemoryLayoutOracle, remainingArgsOwnersRunStarts));
static_assert(offsetof(gl::HashMemory, remainingArgsOwnersBlobStarts) == offsetof(gl::tests::HashMemoryLayoutOracle, remainingArgsOwnersBlobStarts));
static_assert(offsetof(gl::HashMemory, remainingArgsOwnersBlobPool) == offsetof(gl::tests::HashMemoryLayoutOracle, remainingArgsOwnersBlobPool));
static_assert(offsetof(gl::HashMemory, copyOwners) == offsetof(gl::tests::HashMemoryLayoutOracle, copyOwners));
static_assert(offsetof(gl::HashMemory, copyOwnersLengths) == offsetof(gl::tests::HashMemoryLayoutOracle, copyOwnersLengths));
static_assert(offsetof(gl::HashMemory, copyOwnersBytes) == offsetof(gl::tests::HashMemoryLayoutOracle, copyOwnersBytes));
static_assert(offsetof(gl::HashMemory, copyOwnersRunStarts) == offsetof(gl::tests::HashMemoryLayoutOracle, copyOwnersRunStarts));
static_assert(offsetof(gl::HashMemory, copyOwnersBlobStarts) == offsetof(gl::tests::HashMemoryLayoutOracle, copyOwnersBlobStarts));
static_assert(offsetof(gl::HashMemory, copyOwnersBlobPool) == offsetof(gl::tests::HashMemoryLayoutOracle, copyOwnersBlobPool));
static_assert(offsetof(gl::HashMemory, premisePatterns) == offsetof(gl::tests::HashMemoryLayoutOracle, premisePatterns));
static_assert(offsetof(gl::HashMemory, premisePatternsLengths) == offsetof(gl::tests::HashMemoryLayoutOracle, premisePatternsLengths));
static_assert(offsetof(gl::HashMemory, premisePatternsBytes) == offsetof(gl::tests::HashMemoryLayoutOracle, premisePatternsBytes));
static_assert(offsetof(gl::HashMemory, premisePatternsRunStarts) == offsetof(gl::tests::HashMemoryLayoutOracle, premisePatternsRunStarts));
static_assert(offsetof(gl::HashMemory, premisePatternsBlobStarts) == offsetof(gl::tests::HashMemoryLayoutOracle, premisePatternsBlobStarts));
static_assert(offsetof(gl::HashMemory, premisePatternsBlobPool) == offsetof(gl::tests::HashMemoryLayoutOracle, premisePatternsBlobPool));
static_assert(offsetof(gl::HashMemory, admissionMap) == offsetof(gl::tests::HashMemoryLayoutOracle, admissionMap));
static_assert(offsetof(gl::HashMemory, admissionMapKeys) == offsetof(gl::tests::HashMemoryLayoutOracle, admissionMapKeys));
static_assert(offsetof(gl::HashMemory, admissionMapRunStarts) == offsetof(gl::tests::HashMemoryLayoutOracle, admissionMapRunStarts));
static_assert(offsetof(gl::HashMemory, admissionMapBlobStarts) == offsetof(gl::tests::HashMemoryLayoutOracle, admissionMapBlobStarts));
static_assert(offsetof(gl::HashMemory, admissionMapBlobPool) == offsetof(gl::tests::HashMemoryLayoutOracle, admissionMapBlobPool));
static_assert(offsetof(gl::HashMemory, admissionMapIntegration) == offsetof(gl::tests::HashMemoryLayoutOracle, admissionMapIntegration));
static_assert(offsetof(gl::HashMemory, admissionMapIntegrationKeys) == offsetof(gl::tests::HashMemoryLayoutOracle, admissionMapIntegrationKeys));
static_assert(offsetof(gl::HashMemory, admissionMapIntegrationRunStarts) == offsetof(gl::tests::HashMemoryLayoutOracle, admissionMapIntegrationRunStarts));
static_assert(offsetof(gl::HashMemory, admissionMapIntegrationBlobStarts) == offsetof(gl::tests::HashMemoryLayoutOracle, admissionMapIntegrationBlobStarts));
static_assert(offsetof(gl::HashMemory, admissionMapIntegrationBlobPool) == offsetof(gl::tests::HashMemoryLayoutOracle, admissionMapIntegrationBlobPool));
static_assert(offsetof(gl::HashMemory, admissionSetIntegration) == offsetof(gl::tests::HashMemoryLayoutOracle, admissionSetIntegration));
static_assert(offsetof(gl::HashMemory, admissionSetIntegrationKeys) == offsetof(gl::tests::HashMemoryLayoutOracle, admissionSetIntegrationKeys));
static_assert(offsetof(gl::HashMemory, triggersForAdmissionSetIntegration) == offsetof(gl::tests::HashMemoryLayoutOracle, triggersForAdmissionSetIntegration));
static_assert(offsetof(gl::HashMemory, triggersForAdmissionSetIntegrationKeys) == offsetof(gl::tests::HashMemoryLayoutOracle, triggersForAdmissionSetIntegrationKeys));
static_assert(offsetof(gl::HashMemory, admissionMapIntegrationDouble) == offsetof(gl::tests::HashMemoryLayoutOracle, admissionMapIntegrationDouble));
static_assert(offsetof(gl::HashMemory, admissionMapIntegrationDoubleKeys) == offsetof(gl::tests::HashMemoryLayoutOracle, admissionMapIntegrationDoubleKeys));
static_assert(offsetof(gl::HashMemory, admissionMapIntegrationDoubleValues) == offsetof(gl::tests::HashMemoryLayoutOracle, admissionMapIntegrationDoubleValues));
static_assert(offsetof(gl::HashMemory, rejectedMap) == offsetof(gl::tests::HashMemoryLayoutOracle, rejectedMap));
static_assert(offsetof(gl::HashMemory, rejectedMapKeys) == offsetof(gl::tests::HashMemoryLayoutOracle, rejectedMapKeys));
static_assert(offsetof(gl::HashMemory, rejectedMapRunStarts) == offsetof(gl::tests::HashMemoryLayoutOracle, rejectedMapRunStarts));
static_assert(offsetof(gl::HashMemory, rejectedMapBlobStarts) == offsetof(gl::tests::HashMemoryLayoutOracle, rejectedMapBlobStarts));
static_assert(offsetof(gl::HashMemory, rejectedMapBlobPool) == offsetof(gl::tests::HashMemoryLayoutOracle, rejectedMapBlobPool));
static_assert(offsetof(gl::HashMemory, rejectedMapIntegration) == offsetof(gl::tests::HashMemoryLayoutOracle, rejectedMapIntegration));
static_assert(offsetof(gl::HashMemory, rejectedMapIntegrationKeys) == offsetof(gl::tests::HashMemoryLayoutOracle, rejectedMapIntegrationKeys));
static_assert(offsetof(gl::HashMemory, rejectedMapIntegrationRunStarts) == offsetof(gl::tests::HashMemoryLayoutOracle, rejectedMapIntegrationRunStarts));
static_assert(offsetof(gl::HashMemory, rejectedMapIntegrationBlobStarts) == offsetof(gl::tests::HashMemoryLayoutOracle, rejectedMapIntegrationBlobStarts));
static_assert(offsetof(gl::HashMemory, rejectedMapIntegrationBlobPool) == offsetof(gl::tests::HashMemoryLayoutOracle, rejectedMapIntegrationBlobPool));
static_assert(offsetof(gl::HashMemory, rejectedMapOrdis) == offsetof(gl::tests::HashMemoryLayoutOracle, rejectedMapOrdis));
static_assert(offsetof(gl::HashMemory, rejectedMapOrdisKeys) == offsetof(gl::tests::HashMemoryLayoutOracle, rejectedMapOrdisKeys));
static_assert(offsetof(gl::HashMemory, rejectedMapOrdisRunStarts) == offsetof(gl::tests::HashMemoryLayoutOracle, rejectedMapOrdisRunStarts));
static_assert(offsetof(gl::HashMemory, rejectedMapOrdisBlobStarts) == offsetof(gl::tests::HashMemoryLayoutOracle, rejectedMapOrdisBlobStarts));
static_assert(offsetof(gl::HashMemory, rejectedMapOrdisBlobPool) == offsetof(gl::tests::HashMemoryLayoutOracle, rejectedMapOrdisBlobPool));
static_assert(offsetof(gl::HashMemory, varsInRejectedMapIntegrationKeys) == offsetof(gl::tests::HashMemoryLayoutOracle, varsInRejectedMapIntegrationKeys));
static_assert(offsetof(gl::HashMemory, varsInRejectedMapIntegrationKeysKeys) == offsetof(gl::tests::HashMemoryLayoutOracle, varsInRejectedMapIntegrationKeysKeys));
static_assert(offsetof(gl::HashMemory, varsInAdmissionMapKeys) == offsetof(gl::tests::HashMemoryLayoutOracle, varsInAdmissionMapKeys));
static_assert(offsetof(gl::HashMemory, varsInAdmissionMapKeysKeys) == offsetof(gl::tests::HashMemoryLayoutOracle, varsInAdmissionMapKeysKeys));
static_assert(offsetof(gl::HashMemory, varsInAdmissionMapIntegrationKeys) == offsetof(gl::tests::HashMemoryLayoutOracle, varsInAdmissionMapIntegrationKeys));
static_assert(offsetof(gl::HashMemory, varsInAdmissionMapIntegrationKeysKeys) == offsetof(gl::tests::HashMemoryLayoutOracle, varsInAdmissionMapIntegrationKeysKeys));
static_assert(offsetof(gl::HashMemory, admissionStatusMap) == offsetof(gl::tests::HashMemoryLayoutOracle, admissionStatusMap));
static_assert(offsetof(gl::HashMemory, admissionStatusMapKeys) == offsetof(gl::tests::HashMemoryLayoutOracle, admissionStatusMapKeys));
static_assert(offsetof(gl::HashMemory, admissionStatusMapValues) == offsetof(gl::tests::HashMemoryLayoutOracle, admissionStatusMapValues));
static_assert(offsetof(gl::HashMemory, productsOfRecursionIds) == offsetof(gl::tests::HashMemoryLayoutOracle, productsOfRecursionIds));
static_assert(offsetof(gl::HashMemory, productsOfRecursionIdsKeys) == offsetof(gl::tests::HashMemoryLayoutOracle, productsOfRecursionIdsKeys));
static_assert(offsetof(gl::HashMemory, revisitInProgress) == offsetof(gl::tests::HashMemoryLayoutOracle, revisitInProgress));
static_assert(offsetof(gl::HashMemory, revisitInProgressKeys) == offsetof(gl::tests::HashMemoryLayoutOracle, revisitInProgressKeys));
static_assert(offsetof(gl::HashMemory, ordisRevisitInProgress) == offsetof(gl::tests::HashMemoryLayoutOracle, ordisRevisitInProgress));
static_assert(offsetof(gl::HashMemory, ordisRevisitInProgressKeys) == offsetof(gl::tests::HashMemoryLayoutOracle, ordisRevisitInProgressKeys));
static_assert(offsetof(gl::HashMemory, admissionMapOrdis2) == offsetof(gl::tests::HashMemoryLayoutOracle, admissionMapOrdis2));
static_assert(offsetof(gl::HashMemory, admissionMapOrdis2Keys) == offsetof(gl::tests::HashMemoryLayoutOracle, admissionMapOrdis2Keys));
static_assert(offsetof(gl::HashMemory, admissionMapOrdis2RunStarts) == offsetof(gl::tests::HashMemoryLayoutOracle, admissionMapOrdis2RunStarts));
static_assert(offsetof(gl::HashMemory, admissionMapOrdis2BlobStarts) == offsetof(gl::tests::HashMemoryLayoutOracle, admissionMapOrdis2BlobStarts));
static_assert(offsetof(gl::HashMemory, admissionMapOrdis2BlobPool) == offsetof(gl::tests::HashMemoryLayoutOracle, admissionMapOrdis2BlobPool));
static_assert(offsetof(gl::HashMemory, rejectedMapOrdis2) == offsetof(gl::tests::HashMemoryLayoutOracle, rejectedMapOrdis2));
static_assert(offsetof(gl::HashMemory, rejectedMapOrdis2Keys) == offsetof(gl::tests::HashMemoryLayoutOracle, rejectedMapOrdis2Keys));
static_assert(offsetof(gl::HashMemory, rejectedMapOrdis2RunStarts) == offsetof(gl::tests::HashMemoryLayoutOracle, rejectedMapOrdis2RunStarts));
static_assert(offsetof(gl::HashMemory, rejectedMapOrdis2BlobStarts) == offsetof(gl::tests::HashMemoryLayoutOracle, rejectedMapOrdis2BlobStarts));
static_assert(offsetof(gl::HashMemory, rejectedMapOrdis2BlobPool) == offsetof(gl::tests::HashMemoryLayoutOracle, rejectedMapOrdis2BlobPool));
static_assert(offsetof(gl::HashMemory, ordis2RevisitInProgress) == offsetof(gl::tests::HashMemoryLayoutOracle, ordis2RevisitInProgress));
static_assert(offsetof(gl::HashMemory, ordis2RevisitInProgressKeys) == offsetof(gl::tests::HashMemoryLayoutOracle, ordis2RevisitInProgressKeys));
static_assert(offsetof(gl::HashMemory, rejectedMapSubproof) == offsetof(gl::tests::HashMemoryLayoutOracle, rejectedMapSubproof));
static_assert(offsetof(gl::HashMemory, rejectedMapSubproofKeys) == offsetof(gl::tests::HashMemoryLayoutOracle, rejectedMapSubproofKeys));
static_assert(offsetof(gl::HashMemory, rejectedMapSubproofRunStarts) == offsetof(gl::tests::HashMemoryLayoutOracle, rejectedMapSubproofRunStarts));
static_assert(offsetof(gl::HashMemory, rejectedMapSubproofBlobStarts) == offsetof(gl::tests::HashMemoryLayoutOracle, rejectedMapSubproofBlobStarts));
static_assert(offsetof(gl::HashMemory, rejectedMapSubproofBlobPool) == offsetof(gl::tests::HashMemoryLayoutOracle, rejectedMapSubproofBlobPool));
static_assert(offsetof(gl::HashMemory, subproofRevisitInProgress) == offsetof(gl::tests::HashMemoryLayoutOracle, subproofRevisitInProgress));
static_assert(offsetof(gl::HashMemory, subproofRevisitInProgressKeys) == offsetof(gl::tests::HashMemoryLayoutOracle, subproofRevisitInProgressKeys));
}
TEST(hash_memory_interface, preserves_original_field_layout) {
ASSERT_EQ(sizeof(gl::HashMemory), sizeof(gl::tests::HashMemoryLayoutOracle));
ASSERT_EQ(alignof(gl::HashMemory), alignof(gl::tests::HashMemoryLayoutOracle));
}
