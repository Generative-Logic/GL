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
/// Public copy of the original field declarations used only to pin storage layout.
struct HashMemoryLayoutOracle {
LbArena* arena_;
DirtyState* dirty_;
TypedColdBlobMap<NormKey, LocalMemoryValue> encodedMap;
TypedColdBlobMap<NormKey, LocalMemoryValue>::LengthsView encodedMapLengths;
TypedColdBlobMap<NormKey, LocalMemoryValue>::BytesView encodedMapBytes;
TypedColdBlobMap<NormKey, LocalMemoryValue>::RunStartsView encodedMapRunStarts;
TypedColdBlobMap<NormKey, LocalMemoryValue>::BlobStartsView encodedMapBlobStarts;
TypedColdBlobMap<NormKey, LocalMemoryValue>::BlobPoolView encodedMapBlobPool;
TypedColdBlobMap<Int16SetKey, NormKey> remainingArgsNormalizedEncodedMap;
TypedColdBlobMap<Int16SetKey, NormKey>::LengthsView remainingArgsNormalizedEncodedMapLengths;
TypedColdBlobMap<Int16SetKey, NormKey>::BytesView remainingArgsNormalizedEncodedMapBytes;
TypedColdBlobMap<Int16SetKey, NormKey>::RunStartsView remainingArgsNormalizedEncodedMapRunStarts;
TypedColdBlobMap<Int16SetKey, NormKey>::BlobStartsView remainingArgsNormalizedEncodedMapBlobStarts;
TypedColdBlobMap<Int16SetKey, NormKey>::BlobPoolView remainingArgsNormalizedEncodedMapBlobPool;
ReverseArgsIndex remainingArgsReverseIndex;
OriginalsHeadIndex originalsHeadIndex;
TypedColdBlobMap<NormKey, RuleOwnerRec> normalizedEncodedKeys;
TypedColdBlobMap<NormKey, RuleOwnerRec>::LengthsView normalizedEncodedKeysLengths;
TypedColdBlobMap<NormKey, RuleOwnerRec>::BytesView normalizedEncodedKeysBytes;
TypedColdBlobMap<NormKey, RuleOwnerRec>::RunStartsView normalizedEncodedKeysRunStarts;
TypedColdBlobMap<NormKey, RuleOwnerRec>::BlobStartsView normalizedEncodedKeysBlobStarts;
TypedColdBlobMap<NormKey, RuleOwnerRec>::BlobPoolView normalizedEncodedKeysBlobPool;
TypedColdBlobMap<NormKey, OwnerSet> normalizedEncodedSubkeys;
TypedColdBlobMap<NormKey, OwnerSet>::LengthsView normalizedEncodedSubkeysLengths;
TypedColdBlobMap<NormKey, OwnerSet>::BytesView normalizedEncodedSubkeysBytes;
TypedColdBlobMap<NormKey, OwnerSet>::RunStartsView normalizedEncodedSubkeysRunStarts;
TypedColdBlobMap<NormKey, OwnerSet>::BlobStartsView normalizedEncodedSubkeysBlobStarts;
TypedColdBlobMap<NormKey, OwnerSet>::BlobPoolView normalizedEncodedSubkeysBlobPool;
NameId maxKeyLength = 0;
bool ownerlessPending = false;
int32_t goalScanChainMark = 0;
int32_t contradictionChainMark = 0;
TypedColdBlobMap<IdVecKey, RuleOwnerRec> originals;
TypedColdBlobMap<IdVecKey, RuleOwnerRec>::LengthsView originalsLengths;
TypedColdBlobMap<IdVecKey, RuleOwnerRec>::BytesView originalsBytes;
TypedColdBlobMap<IdVecKey, RuleOwnerRec>::RunStartsView originalsRunStarts;
TypedColdBlobMap<IdVecKey, RuleOwnerRec>::BlobStartsView originalsBlobStarts;
TypedColdBlobMap<IdVecKey, RuleOwnerRec>::BlobPoolView originalsBlobPool;
TypedColdBlobMap<IdVecKey, RuleOwnerRec> remainingArgsOwners;
TypedColdBlobMap<IdVecKey, RuleOwnerRec>::LengthsView remainingArgsOwnersLengths;
TypedColdBlobMap<IdVecKey, RuleOwnerRec>::BytesView remainingArgsOwnersBytes;
TypedColdBlobMap<IdVecKey, RuleOwnerRec>::RunStartsView remainingArgsOwnersRunStarts;
TypedColdBlobMap<IdVecKey, RuleOwnerRec>::BlobStartsView remainingArgsOwnersBlobStarts;
TypedColdBlobMap<IdVecKey, RuleOwnerRec>::BlobPoolView remainingArgsOwnersBlobPool;
TypedColdBlobMap<IdVecKey, RuleOwnerRec> copyOwners;
TypedColdBlobMap<IdVecKey, RuleOwnerRec>::LengthsView copyOwnersLengths;
TypedColdBlobMap<IdVecKey, RuleOwnerRec>::BytesView copyOwnersBytes;
TypedColdBlobMap<IdVecKey, RuleOwnerRec>::RunStartsView copyOwnersRunStarts;
TypedColdBlobMap<IdVecKey, RuleOwnerRec>::BlobStartsView copyOwnersBlobStarts;
TypedColdBlobMap<IdVecKey, RuleOwnerRec>::BlobPoolView copyOwnersBlobPool;
TypedColdBlobMap<IdVecKey, RuleOwnerRec> premisePatterns;
TypedColdBlobMap<IdVecKey, RuleOwnerRec>::LengthsView premisePatternsLengths;
TypedColdBlobMap<IdVecKey, RuleOwnerRec>::BytesView premisePatternsBytes;
TypedColdBlobMap<IdVecKey, RuleOwnerRec>::RunStartsView premisePatternsRunStarts;
TypedColdBlobMap<IdVecKey, RuleOwnerRec>::BlobStartsView premisePatternsBlobStarts;
TypedColdBlobMap<IdVecKey, RuleOwnerRec>::BlobPoolView premisePatternsBlobPool;
TypedColdBlobMap<int64_t, AdmissionMapValue> admissionMap;
TypedColdBlobMap<int64_t, AdmissionMapValue>::KeysView admissionMapKeys;
TypedColdBlobMap<int64_t, AdmissionMapValue>::RunStartsView admissionMapRunStarts;
TypedColdBlobMap<int64_t, AdmissionMapValue>::BlobStartsView admissionMapBlobStarts;
TypedColdBlobMap<int64_t, AdmissionMapValue>::BlobPoolView admissionMapBlobPool;
TypedColdBlobMap<int64_t, IntegrationEntry> admissionMapIntegration;
TypedColdBlobMap<int64_t, IntegrationEntry>::KeysView admissionMapIntegrationKeys;
TypedColdBlobMap<int64_t, IntegrationEntry>::RunStartsView admissionMapIntegrationRunStarts;
TypedColdBlobMap<int64_t, IntegrationEntry>::BlobStartsView admissionMapIntegrationBlobStarts;
TypedColdBlobMap<int64_t, IntegrationEntry>::BlobPoolView admissionMapIntegrationBlobPool;
TypedColdSet<int64_t> admissionSetIntegration;
TypedColdSet<int64_t>::KeysView admissionSetIntegrationKeys;
TypedColdSet<int64_t> triggersForAdmissionSetIntegration;
TypedColdSet<int64_t>::KeysView triggersForAdmissionSetIntegrationKeys;
TypedColdMap<int64_t, int64_t> admissionMapIntegrationDouble;
TypedColdMap<int64_t, int64_t>::KeysView admissionMapIntegrationDoubleKeys;
TypedColdMap<int64_t, int64_t>::ValuesView admissionMapIntegrationDoubleValues;
TypedColdBlobMap<int64_t, RejectedMapValue> rejectedMap;
TypedColdBlobMap<int64_t, RejectedMapValue>::KeysView rejectedMapKeys;
TypedColdBlobMap<int64_t, RejectedMapValue>::RunStartsView rejectedMapRunStarts;
TypedColdBlobMap<int64_t, RejectedMapValue>::BlobStartsView rejectedMapBlobStarts;
TypedColdBlobMap<int64_t, RejectedMapValue>::BlobPoolView rejectedMapBlobPool;
TypedColdBlobMap<int64_t, RejectedMapIntegrationValue> rejectedMapIntegration;
TypedColdBlobMap<int64_t, RejectedMapIntegrationValue>::KeysView rejectedMapIntegrationKeys;
TypedColdBlobMap<int64_t, RejectedMapIntegrationValue>::RunStartsView rejectedMapIntegrationRunStarts;
TypedColdBlobMap<int64_t, RejectedMapIntegrationValue>::BlobStartsView rejectedMapIntegrationBlobStarts;
TypedColdBlobMap<int64_t, RejectedMapIntegrationValue>::BlobPoolView rejectedMapIntegrationBlobPool;
TypedColdBlobMap<int64_t, RejectedMapOrdisValue> rejectedMapOrdis;
TypedColdBlobMap<int64_t, RejectedMapOrdisValue>::KeysView rejectedMapOrdisKeys;
TypedColdBlobMap<int64_t, RejectedMapOrdisValue>::RunStartsView rejectedMapOrdisRunStarts;
TypedColdBlobMap<int64_t, RejectedMapOrdisValue>::BlobStartsView rejectedMapOrdisBlobStarts;
TypedColdBlobMap<int64_t, RejectedMapOrdisValue>::BlobPoolView rejectedMapOrdisBlobPool;
TypedColdSet<NameId> varsInRejectedMapIntegrationKeys;
TypedColdSet<NameId>::KeysView varsInRejectedMapIntegrationKeysKeys;
TypedColdSet<NameId> varsInAdmissionMapKeys;
TypedColdSet<NameId>::KeysView varsInAdmissionMapKeysKeys;
TypedColdSet<NameId> varsInAdmissionMapIntegrationKeys;
TypedColdSet<NameId>::KeysView varsInAdmissionMapIntegrationKeysKeys;
TypedColdMap<int64_t, uint8_t> admissionStatusMap;
TypedColdMap<int64_t, uint8_t>::KeysView admissionStatusMapKeys;
TypedColdMap<int64_t, uint8_t>::ValuesView admissionStatusMapValues;
TypedColdSet<NameId> productsOfRecursionIds;
TypedColdSet<NameId>::KeysView productsOfRecursionIdsKeys;
TypedColdSet<int64_t> revisitInProgress;
TypedColdSet<int64_t>::KeysView revisitInProgressKeys;
TypedColdSet<int64_t> ordisRevisitInProgress;
TypedColdSet<int64_t>::KeysView ordisRevisitInProgressKeys;
TypedColdBlobMap<int64_t, AdmissionMapOrdis2Value> admissionMapOrdis2;
TypedColdBlobMap<int64_t, AdmissionMapOrdis2Value>::KeysView admissionMapOrdis2Keys;
TypedColdBlobMap<int64_t, AdmissionMapOrdis2Value>::RunStartsView admissionMapOrdis2RunStarts;
TypedColdBlobMap<int64_t, AdmissionMapOrdis2Value>::BlobStartsView admissionMapOrdis2BlobStarts;
TypedColdBlobMap<int64_t, AdmissionMapOrdis2Value>::BlobPoolView admissionMapOrdis2BlobPool;
TypedColdBlobMap<int64_t, RejectedMapOrdisValue> rejectedMapOrdis2;
TypedColdBlobMap<int64_t, RejectedMapOrdisValue>::KeysView rejectedMapOrdis2Keys;
TypedColdBlobMap<int64_t, RejectedMapOrdisValue>::RunStartsView rejectedMapOrdis2RunStarts;
TypedColdBlobMap<int64_t, RejectedMapOrdisValue>::BlobStartsView rejectedMapOrdis2BlobStarts;
TypedColdBlobMap<int64_t, RejectedMapOrdisValue>::BlobPoolView rejectedMapOrdis2BlobPool;
TypedColdSet<int64_t> ordis2RevisitInProgress;
TypedColdSet<int64_t>::KeysView ordis2RevisitInProgressKeys;
TypedColdBlobMap<int64_t, IntegrationEntry> rejectedMapSubproof;
TypedColdBlobMap<int64_t, IntegrationEntry>::KeysView rejectedMapSubproofKeys;
TypedColdBlobMap<int64_t, IntegrationEntry>::RunStartsView rejectedMapSubproofRunStarts;
TypedColdBlobMap<int64_t, IntegrationEntry>::BlobStartsView rejectedMapSubproofBlobStarts;
TypedColdBlobMap<int64_t, IntegrationEntry>::BlobPoolView rejectedMapSubproofBlobPool;
TypedColdSet<int64_t> subproofRevisitInProgress;
TypedColdSet<int64_t>::KeysView subproofRevisitInProgressKeys;
};
}
