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

// HashMemory -- the per-LB hash-inference engine (one struct, four instances
// per Memory). After Batch 4 every member is a cold (arena-backed,
// deload-persisted) container, so the struct lives in memory_infra/ alongside
// the rest of the static-memory substrate and can be held directly by LbMemory
// (D-147). The blob-map record codecs live beside their record
// types in memory.hpp.

#include "typed_cold_map.hpp"
#include "reverse_args_index.hpp"

#include <utility>

namespace gl {

    // Record value types -- fully defined in memory.hpp. Forward-declared here
    // because the TypedColdBlobMap facade names its Record type only inside
    // member-template methods (assignRun / recordsAt / appendRecord), which
    // instantiate at the call sites in memory.cpp / prover.hpp (both include
    // memory.hpp), never inside this struct's own definition.
    struct LocalMemoryValue;
    struct OwnerSet;
    struct AdmissionMapValue;
    struct IntegrationEntry;
    struct RejectedMapValue;
    struct RejectedMapIntegrationValue;

    /// @brief Deload tag-block bases for the four cold HashMemory instances —
    ///        their facets stream at base+0.., one 100-tag block each, past
    ///        LbMemory's own tags 0..50. `LbMemory::visitContainers` enumerates
    ///        each instance at its base (D-147); the 100-tag
    ///        blocks leave headroom as the per-instance container set grows.
    constexpr uint32_t kOverallHashMemoryDeloadBase = 51;
    constexpr uint32_t kLocalHashMemoryDeloadBase = 151;
    constexpr uint32_t kLocalHashMemoryDeltaDeloadBase = 251;
    constexpr uint32_t kWorkingMemoryDeloadBase = 351;

    struct HashMemory {
        // The owning LB's arena + shared dirty flag, forwarded into every cold
        // container at construction. All four instances (overall / local /
        // localHashMemoryDelta / workingMemory) bind the LB's single cold arena
        // (lbMemory.manager); the transient pair is emptied per burst by
        // resetToFresh, not by a separate arena
        // (D-176).
        LbArena* arena_;
        DirtyState* dirty_;
        // encodedMap: normalized premise key -> the stack of LocalMemoryValues
        // that can fire from it. Cold BLOB map (bytes key NormKey -> a run of
        // LocalMemoryValue records); 5 facets (key Lengths+Bytes, value
        // RunStarts+BlobStarts+BlobPool). Writes are read-modify-write of the
        // whole run (the blob store replaces a key's run wholesale).
        TypedColdBlobMap<NormKey, LocalMemoryValue> encodedMap;
        TypedColdBlobMap<NormKey, LocalMemoryValue>::LengthsView encodedMapLengths;
        TypedColdBlobMap<NormKey, LocalMemoryValue>::BytesView encodedMapBytes;
        TypedColdBlobMap<NormKey, LocalMemoryValue>::RunStartsView encodedMapRunStarts;
        TypedColdBlobMap<NormKey, LocalMemoryValue>::BlobStartsView encodedMapBlobStarts;
        TypedColdBlobMap<NormKey, LocalMemoryValue>::BlobPoolView encodedMapBlobPool;
        // Secondary index: a remaining-arg id set -> the run of normalized keys
        // (NormKey blobs) whose encodedMap entry carries exactly those args.
        // Statified COLD blob map (D-173): the run is
        // kept sorted by (numberExpressions, data) so the deload + dump stay
        // canonical; insertRemainingArgsNormKey is the RMW insert. Five facets:
        // key Lengths+Bytes, value RunStarts+BlobStarts+BlobPool.
        TypedColdBlobMap<Int16SetKey, NormKey> remainingArgsNormalizedEncodedMap;
        TypedColdBlobMap<Int16SetKey, NormKey>::LengthsView remainingArgsNormalizedEncodedMapLengths;
        TypedColdBlobMap<Int16SetKey, NormKey>::BytesView remainingArgsNormalizedEncodedMapBytes;
        TypedColdBlobMap<Int16SetKey, NormKey>::RunStartsView remainingArgsNormalizedEncodedMapRunStarts;
        TypedColdBlobMap<Int16SetKey, NormKey>::BlobStartsView remainingArgsNormalizedEncodedMapBlobStarts;
        TypedColdBlobMap<Int16SetKey, NormKey>::BlobPoolView remainingArgsNormalizedEncodedMapBlobPool;
        // Derived reverse membership side-index of
        // remainingArgsNormalizedEncodedMap: a NormKey's bytes -> the run of
        // forward-map key ids whose stored run contains that NormKey. Rides the
        // LB's deloadable arena but is NEVER enrolled in visitContainers, NEVER
        // deloaded, NEVER dumped (I-117 pattern); maintained by appendEdge at the
        // insertRemainingArgsNormKey installs, rebuilt on canonical reload +
        // after wipeRemainingArgsForClosed, cleared at the canonical release
        // seam, captured verbatim by the raw image (I-154).
        // It inverts checkLocalEncodedMemoryStatic's candidate enumeration from
        // an O(keys) forward scan to one hash probe.
        ReverseArgsIndex remainingArgsReverseIndex;
        // D-72 owner-set maps — the four "subkey" fast-rejection indices, now
        // COLD blob maps (bytes key NormKey -> ONE OwnerSet blob per key: the
        // whole codec'd value, run-length-1, whole-value replace on update). The
        // value (`OwnerSet`) records every implication+scope that "birthed" the
        // key as a packed composite id, plus the u_ literal signatures for the
        // D-120 prune. Reads go through the no-alloc byte peek
        // (ExpressionAnalyzer::ownerKeyAccepts -> peekRecordBytes + OwnerSetBlob),
        // never a full decode; writes are RMW (ExpressionAnalyzer::
        // mergeOwnerRecord). On impl-scope close the radical wipe drops owners
        // matching the closed scope; if the owner-set empties, the key is dropped
        // from the rebuild. Each map contributes 5 deload facets (key
        // Lengths+Bytes, value RunStarts+BlobStarts+BlobPool).
        TypedColdBlobMap<NormKey, OwnerSet> normalizedEncodedKeys;
        TypedColdBlobMap<NormKey, OwnerSet>::LengthsView normalizedEncodedKeysLengths;
        TypedColdBlobMap<NormKey, OwnerSet>::BytesView normalizedEncodedKeysBytes;
        TypedColdBlobMap<NormKey, OwnerSet>::RunStartsView normalizedEncodedKeysRunStarts;
        TypedColdBlobMap<NormKey, OwnerSet>::BlobStartsView normalizedEncodedKeysBlobStarts;
        TypedColdBlobMap<NormKey, OwnerSet>::BlobPoolView normalizedEncodedKeysBlobPool;
        TypedColdBlobMap<NormKey, OwnerSet> normalizedEncodedSubkeys;
        TypedColdBlobMap<NormKey, OwnerSet>::LengthsView normalizedEncodedSubkeysLengths;
        TypedColdBlobMap<NormKey, OwnerSet>::BytesView normalizedEncodedSubkeysBytes;
        TypedColdBlobMap<NormKey, OwnerSet>::RunStartsView normalizedEncodedSubkeysRunStarts;
        TypedColdBlobMap<NormKey, OwnerSet>::BlobStartsView normalizedEncodedSubkeysBlobStarts;
        TypedColdBlobMap<NormKey, OwnerSet>::BlobPoolView normalizedEncodedSubkeysBlobPool;
        TypedColdBlobMap<NormKey, OwnerSet> normalizedEncodedSubkeysMinusOne;
        TypedColdBlobMap<NormKey, OwnerSet>::LengthsView normalizedEncodedSubkeysMinusOneLengths;
        TypedColdBlobMap<NormKey, OwnerSet>::BytesView normalizedEncodedSubkeysMinusOneBytes;
        TypedColdBlobMap<NormKey, OwnerSet>::RunStartsView normalizedEncodedSubkeysMinusOneRunStarts;
        TypedColdBlobMap<NormKey, OwnerSet>::BlobStartsView normalizedEncodedSubkeysMinusOneBlobStarts;
        TypedColdBlobMap<NormKey, OwnerSet>::BlobPoolView normalizedEncodedSubkeysMinusOneBlobPool;
        TypedColdBlobMap<NormKey, OwnerSet> normalizedEncodedSubkeysMinusTwo;
        TypedColdBlobMap<NormKey, OwnerSet>::LengthsView normalizedEncodedSubkeysMinusTwoLengths;
        TypedColdBlobMap<NormKey, OwnerSet>::BytesView normalizedEncodedSubkeysMinusTwoBytes;
        TypedColdBlobMap<NormKey, OwnerSet>::RunStartsView normalizedEncodedSubkeysMinusTwoRunStarts;
        TypedColdBlobMap<NormKey, OwnerSet>::BlobStartsView normalizedEncodedSubkeysMinusTwoBlobStarts;
        TypedColdBlobMap<NormKey, OwnerSet>::BlobPoolView normalizedEncodedSubkeysMinusTwoBlobPool;
        NameId maxKeyLength = 0;
        // --- shared members (path-independent) ---
        // Implication chains (premises + head) as ruleInterner id vectors
        // (D-133). Written at install; iterated (decoded lex-sorted) by the
        // integration premise-matcher and the dump; the orphan check erases a
        // chain once no LMV references it.
        // Statified COLD set (D-173): byte-key set,
        // one IdVecKey blob per chain. Two facets: key Lengths + Bytes.
        TypedColdSet<IdVecKey> originals;
        TypedColdSet<IdVecKey>::LengthsView originalsLengths;
        TypedColdSet<IdVecKey>::BytesView originalsBytes;
        // Packed (templateId, validityId) keys — template half from
        // Memory::templateInterner, validity half from Memory::nameMap
        // (D-132). Order-sensitive walks iterate
        // decoded (template, validity) lex-sorted snapshots.
        // Statified COLD blob map (D-172): packed
        // key -> a run of AdmissionMapValue record blobs. The run is kept sorted
        // by DecodedAdmissionValueLess so it is canonical (byte-identical deload
        // + dump). Writes are read-modify-write of the whole run
        // (insertAdmissionValue); the hot read (isAdmitted) snapshots via
        // admissionRecordsAt. Four facets: key, run-starts, blob-starts,
        // blob-pool. Only ever used via overallHashMemory.
        TypedColdBlobMap<int64_t, AdmissionMapValue> admissionMap;
        TypedColdBlobMap<int64_t, AdmissionMapValue>::KeysView admissionMapKeys;
        TypedColdBlobMap<int64_t, AdmissionMapValue>::RunStartsView admissionMapRunStarts;
        TypedColdBlobMap<int64_t, AdmissionMapValue>::BlobStartsView admissionMapBlobStarts;
        TypedColdBlobMap<int64_t, AdmissionMapValue>::BlobPoolView admissionMapBlobPool;
        // Packed (templateId, validityId) keys — u_-form templates
        // (D-132); stored instructions and payloads
        // id-form (D-1322).
        // Statified COLD blob map (D-172): packed
        // key -> a run of IntegrationEntry record blobs, one per inner-map entry
        // (IntInstruction -> ValueIdSet). The run is held in the inner map's
        // decoded order (DecodedInstructionLess outer, DecodedIdLess inner) so it
        // is canonical (byte-identical deload + dump). Writes are read-modify-write
        // of the whole run (insertAdmissionIntegrationValue); reads snapshot the
        // nested map via admissionIntegrationRecordsAt. Entries persist across
        // revival/consumption (I-22). Four facets: key, run-starts, blob-starts,
        // blob-pool. Only ever used via overallHashMemory.
        TypedColdBlobMap<int64_t, IntegrationEntry> admissionMapIntegration;
        TypedColdBlobMap<int64_t, IntegrationEntry>::KeysView admissionMapIntegrationKeys;
        TypedColdBlobMap<int64_t, IntegrationEntry>::RunStartsView admissionMapIntegrationRunStarts;
        TypedColdBlobMap<int64_t, IntegrationEntry>::BlobStartsView admissionMapIntegrationBlobStarts;
        TypedColdBlobMap<int64_t, IntegrationEntry>::BlobPoolView admissionMapIntegrationBlobPool;
        // Packed (templateId, validityId) keys (D-1322).
        // Members are TEMPLATE-population strings (u_-stripped marker forms
        // and repl_-form trigger expressions), so the template space owns
        // them — they are never NameMap-interned. Order-sensitive walks
        // (the makeAdmissionKeys-per-trigger production loop) iterate
        // decoded lex-sorted snapshots.
        // Statified COLD set (D-173): packed key set,
        // per-scope radical-wipe removal via eraseIf. One facet each: key.
        TypedColdSet<int64_t> admissionSetIntegration;
        TypedColdSet<int64_t>::KeysView admissionSetIntegrationKeys;
        TypedColdSet<int64_t> triggersForAdmissionSetIntegration;
        TypedColdSet<int64_t>::KeysView triggersForAdmissionSetIntegrationKeys;
        // Packed (templateId, validityId) keys (D-132);
        // id-form values ordered by DecodedRejectedValueLess.
        // Statified COLD blob map (D-172): packed
        // key -> a run of RejectedMapValue record blobs, kept sorted by
        // DecodedRejectedValueLess. Writes via updateRejectedMap (RMW
        // insertRejectedValue); reads snapshot (rejectedRecordsAt). NEVER
        // written directly by the equi-class hook (I-37); the revival consumer
        // revisitRejected2 erases the consumed cohort. Four facets: key,
        // run-starts, blob-starts, blob-pool.
        TypedColdBlobMap<int64_t, RejectedMapValue> rejectedMap;
        TypedColdBlobMap<int64_t, RejectedMapValue>::KeysView rejectedMapKeys;
        TypedColdBlobMap<int64_t, RejectedMapValue>::RunStartsView rejectedMapRunStarts;
        TypedColdBlobMap<int64_t, RejectedMapValue>::BlobStartsView rejectedMapBlobStarts;
        TypedColdBlobMap<int64_t, RejectedMapValue>::BlobPoolView rejectedMapBlobPool;
        // Integration-side rejection buffer — keyed on a non-in[] constituent's
        // marker form. See RejectedMapIntegrationValue for shape + rationale.
        // Packed (templateId, validityId) keys (D-132);
        // id-form values ordered by DecodedRejectedIntegrationValueLess.
        // Statified COLD blob map (D-172): packed
        // key -> a run of RejectedMapIntegrationValue record blobs, kept sorted
        // by DecodedRejectedIntegrationValueLess. Writes via
        // updateRejectedMapIntegration (RMW insertRejectedIntegrationValue); reads
        // snapshot (rejectedIntegrationRecordsAt). NEVER written directly by the
        // equi-class hook (I-37); revisitRejectedIntegration2 erases the consumed
        // cohort. Four facets: key, run-starts, blob-starts, blob-pool.
        TypedColdBlobMap<int64_t, RejectedMapIntegrationValue> rejectedMapIntegration;
        TypedColdBlobMap<int64_t, RejectedMapIntegrationValue>::KeysView rejectedMapIntegrationKeys;
        TypedColdBlobMap<int64_t, RejectedMapIntegrationValue>::RunStartsView rejectedMapIntegrationRunStarts;
        TypedColdBlobMap<int64_t, RejectedMapIntegrationValue>::BlobStartsView rejectedMapIntegrationBlobStarts;
        TypedColdBlobMap<int64_t, RejectedMapIntegrationValue>::BlobPoolView rejectedMapIntegrationBlobPool;
        // Monotonically-growing cache of non-marker args that appear in any
        // rejectedMapIntegration key. Used by applyEquivalenceClassToRejectedMapIntegration
        // to short-circuit when an eq class has no overlap with any stored
        // key — saves O(|rmi|) walk per class call on batches where rmi is
        // large but most classes are unrelated (observed in Gauss: ~10^5 rmi
        // entries and ~10^6 class calls).
        // Template-space ids of the bare arg names (D-132).
        // Over-approximation caches — never wiped; a stale id only costs a
        // wasted walk. Probes: non-minting templateInterner.lookup of the
        // decoded class member.
        // Statified COLD set (D-172). Monotone
        // over-approximation cache; never wiped. One facet: key.
        TypedColdSet<NameId> varsInRejectedMapIntegrationKeys;
        TypedColdSet<NameId>::KeysView varsInRejectedMapIntegrationKeysKeys;
        // Symmetric cache for admissionMap, populated at every admissionMap
        // insert. Used by applyEquivalenceClassToAdmissionMap to short-circuit
        // when an eq class has no overlap with any stored admission key.
        // Statified COLD set (D-172). Monotone
        // over-approximation cache; never wiped. One facet: key.
        TypedColdSet<NameId> varsInAdmissionMapKeys;
        TypedColdSet<NameId>::KeysView varsInAdmissionMapKeysKeys;
        // Symmetric cache for admissionMapIntegration. Stores BARE-form
        // (u_-stripped) non-marker args because admissionMapIntegration keys
        // are u_-prefixed and class members are bare names; the overlap probe
        // is on bare names. Used by applyEquivalenceClassToAdmissionMapIntegration
        // to short-circuit when an eq class has no overlap with any stored
        // admission-integration key. Populated at every
        // admissionMapIntegration insert site.
        // Statified COLD set (D-172). Monotone
        // over-approximation cache; never wiped. One facet: key.
        TypedColdSet<NameId> varsInAdmissionMapIntegrationKeys;
        TypedColdSet<NameId>::KeysView varsInAdmissionMapIntegrationKeysKeys;
        // Statified COLD single-value map (D-172):
        // packed key -> partOfRecursion flag (uint8). Parallel to admissionMap.
        // Two facets: key, value.
        TypedColdMap<int64_t, uint8_t> admissionStatusMap;
        TypedColdMap<int64_t, uint8_t>::KeysView admissionStatusMapKeys;
        TypedColdMap<int64_t, uint8_t>::ValuesView admissionStatusMapValues;
        // Statified COLD set (D-173): NameMap-id
        // membership cache (never per-scope wiped). One facet: key.
        TypedColdSet<NameId> productsOfRecursionIds;
        TypedColdSet<NameId>::KeysView productsOfRecursionIdsKeys;
        // Statified COLD set (D-172). One facet: key.
        TypedColdSet<int64_t> consumedAdmissionKeys;
        TypedColdSet<int64_t>::KeysView consumedAdmissionKeysKeys;
        // Statified COLD set (D-172). Re-entrant
        // revival guard (empty at barriers). One facet: key.
        TypedColdSet<int64_t> revisitInProgress;
        TypedColdSet<int64_t>::KeysView revisitInProgressKeys;

        /// @brief Bind the rule store to the owning LB's arena + dirty flag.
        ///
        /// @details
        /// Replaces the former default constructor: an arena-bound `HashMemory`
        /// is constructed with the arena its (migrating) cold containers will
        /// allocate on and the dirty flag they will set. Stored but unused
        /// while every member is still heap (Part C 1b-i).
        ///
        /// @param arena The arena backing this instance's cold containers
        ///              (LbMemory's cold arena for the deloaded overall/local
        ///              instances; a HOT arena for the transient working/delta
        ///              instances; may be null in a heap-only unit test).
        /// @param dirty The shared content-change flag the cold containers set.
        HashMemory(LbArena* arena, DirtyState* dirty)
            : arena_(arena), dirty_(dirty),
              encodedMap(arena, dirty),
              encodedMapLengths(&encodedMap.inner()),
              encodedMapBytes(&encodedMap.inner()),
              encodedMapRunStarts(&encodedMap.inner()),
              encodedMapBlobStarts(&encodedMap.inner()),
              encodedMapBlobPool(&encodedMap.inner()),
              remainingArgsNormalizedEncodedMap(arena, dirty),
              remainingArgsNormalizedEncodedMapLengths(&remainingArgsNormalizedEncodedMap.inner()),
              remainingArgsNormalizedEncodedMapBytes(&remainingArgsNormalizedEncodedMap.inner()),
              remainingArgsNormalizedEncodedMapRunStarts(&remainingArgsNormalizedEncodedMap.inner()),
              remainingArgsNormalizedEncodedMapBlobStarts(&remainingArgsNormalizedEncodedMap.inner()),
              remainingArgsNormalizedEncodedMapBlobPool(&remainingArgsNormalizedEncodedMap.inner()),
              remainingArgsReverseIndex(arena),
              maxKeyLength(0),
              normalizedEncodedKeys(arena, dirty),
              normalizedEncodedKeysLengths(&normalizedEncodedKeys.inner()),
              normalizedEncodedKeysBytes(&normalizedEncodedKeys.inner()),
              normalizedEncodedKeysRunStarts(&normalizedEncodedKeys.inner()),
              normalizedEncodedKeysBlobStarts(&normalizedEncodedKeys.inner()),
              normalizedEncodedKeysBlobPool(&normalizedEncodedKeys.inner()),
              normalizedEncodedSubkeys(arena, dirty),
              normalizedEncodedSubkeysLengths(&normalizedEncodedSubkeys.inner()),
              normalizedEncodedSubkeysBytes(&normalizedEncodedSubkeys.inner()),
              normalizedEncodedSubkeysRunStarts(&normalizedEncodedSubkeys.inner()),
              normalizedEncodedSubkeysBlobStarts(&normalizedEncodedSubkeys.inner()),
              normalizedEncodedSubkeysBlobPool(&normalizedEncodedSubkeys.inner()),
              normalizedEncodedSubkeysMinusOne(arena, dirty),
              normalizedEncodedSubkeysMinusOneLengths(&normalizedEncodedSubkeysMinusOne.inner()),
              normalizedEncodedSubkeysMinusOneBytes(&normalizedEncodedSubkeysMinusOne.inner()),
              normalizedEncodedSubkeysMinusOneRunStarts(&normalizedEncodedSubkeysMinusOne.inner()),
              normalizedEncodedSubkeysMinusOneBlobStarts(&normalizedEncodedSubkeysMinusOne.inner()),
              normalizedEncodedSubkeysMinusOneBlobPool(&normalizedEncodedSubkeysMinusOne.inner()),
              normalizedEncodedSubkeysMinusTwo(arena, dirty),
              normalizedEncodedSubkeysMinusTwoLengths(&normalizedEncodedSubkeysMinusTwo.inner()),
              normalizedEncodedSubkeysMinusTwoBytes(&normalizedEncodedSubkeysMinusTwo.inner()),
              normalizedEncodedSubkeysMinusTwoRunStarts(&normalizedEncodedSubkeysMinusTwo.inner()),
              normalizedEncodedSubkeysMinusTwoBlobStarts(&normalizedEncodedSubkeysMinusTwo.inner()),
              normalizedEncodedSubkeysMinusTwoBlobPool(&normalizedEncodedSubkeysMinusTwo.inner()),
              originals(arena, dirty),
              originalsLengths(&originals.inner()),
              originalsBytes(&originals.inner()),
              admissionMap(arena, dirty),
              admissionMapKeys(&admissionMap.inner()),
              admissionMapRunStarts(&admissionMap.inner()),
              admissionMapBlobStarts(&admissionMap.inner()),
              admissionMapBlobPool(&admissionMap.inner()),
              admissionMapIntegration(arena, dirty),
              admissionMapIntegrationKeys(&admissionMapIntegration.inner()),
              admissionMapIntegrationRunStarts(&admissionMapIntegration.inner()),
              admissionMapIntegrationBlobStarts(&admissionMapIntegration.inner()),
              admissionMapIntegrationBlobPool(&admissionMapIntegration.inner()),
              admissionSetIntegration(arena, dirty),
              admissionSetIntegrationKeys(&admissionSetIntegration.inner()),
              triggersForAdmissionSetIntegration(arena, dirty),
              triggersForAdmissionSetIntegrationKeys(&triggersForAdmissionSetIntegration.inner()),
              rejectedMap(arena, dirty),
              rejectedMapKeys(&rejectedMap.inner()),
              rejectedMapRunStarts(&rejectedMap.inner()),
              rejectedMapBlobStarts(&rejectedMap.inner()),
              rejectedMapBlobPool(&rejectedMap.inner()),
              rejectedMapIntegration(arena, dirty),
              rejectedMapIntegrationKeys(&rejectedMapIntegration.inner()),
              rejectedMapIntegrationRunStarts(&rejectedMapIntegration.inner()),
              rejectedMapIntegrationBlobStarts(&rejectedMapIntegration.inner()),
              rejectedMapIntegrationBlobPool(&rejectedMapIntegration.inner()),
              varsInRejectedMapIntegrationKeys(arena, dirty),
              varsInRejectedMapIntegrationKeysKeys(&varsInRejectedMapIntegrationKeys.inner()),
              varsInAdmissionMapKeys(arena, dirty),
              varsInAdmissionMapKeysKeys(&varsInAdmissionMapKeys.inner()),
              varsInAdmissionMapIntegrationKeys(arena, dirty),
              varsInAdmissionMapIntegrationKeysKeys(&varsInAdmissionMapIntegrationKeys.inner()),
              admissionStatusMap(arena, dirty),
              admissionStatusMapKeys(&admissionStatusMap.inner()),
              admissionStatusMapValues(&admissionStatusMap.inner()),
              productsOfRecursionIds(arena, dirty),
              productsOfRecursionIdsKeys(&productsOfRecursionIds.inner()),
              consumedAdmissionKeys(arena, dirty),
              consumedAdmissionKeysKeys(&consumedAdmissionKeys.inner()),
              revisitInProgress(arena, dirty),
              revisitInProgressKeys(&revisitInProgress.inner())
        {}

        // Arena-bound: non-copyable (an arena pointer must not be blindly
        // duplicated). The former swap-with-empty / `= HashMemory()` resets
        // become `resetToFresh()`.
        HashMemory(const HashMemory&) = delete;
        HashMemory& operator=(const HashMemory&) = delete;

        /// @brief Wholesale reset to empty — the destroyGrid / per-burst path.
        ///
        /// @details
        /// Empties every member. As containers migrate onto the cold facade,
        /// each adds its `resetToFresh()` here; the heap members stay
        /// `clear()`ed. Replaces the former `= HashMemory()` and
        /// swap-with-empty resets, which cannot move an arena-bound instance
        /// between arenas.
        void resetToFresh() {
            clear();
        }

        /// @brief Live byte footprint of the cold (arena-backed) containers.
        ///
        /// @details
        /// 0 until a container migrates onto the facade (heap members carry no
        /// arena bytes); each migrated container adds its `liveBytes()` here.
        ///
        /// @return Live bytes across this instance's cold containers.
        int64_t liveBytes() const {
            return encodedMap.liveBytes()
                 + normalizedEncodedKeys.liveBytes()
                 + normalizedEncodedSubkeys.liveBytes()
                 + normalizedEncodedSubkeysMinusOne.liveBytes()
                 + normalizedEncodedSubkeysMinusTwo.liveBytes()
                 + admissionMap.liveBytes()
                 + admissionStatusMap.liveBytes()
                 + consumedAdmissionKeys.liveBytes()
                 + varsInAdmissionMapKeys.liveBytes()
                 + rejectedMap.liveBytes()
                 + revisitInProgress.liveBytes()
                 + admissionMapIntegration.liveBytes()
                 + rejectedMapIntegration.liveBytes()
                 + varsInAdmissionMapIntegrationKeys.liveBytes()
                 + varsInRejectedMapIntegrationKeys.liveBytes()
                 + admissionSetIntegration.liveBytes()
                 + triggersForAdmissionSetIntegration.liveBytes()
                 + productsOfRecursionIds.liveBytes()
                 + originals.liveBytes()
                 + remainingArgsNormalizedEncodedMap.liveBytes()
                 + remainingArgsReverseIndex.liveBytes();
        }

        /// @brief Release every cold (arena-backed) container's blocks + index —
        ///        the full-release path for teardown (`~Memory`, `destroyGrid`).
        ///
        /// @details
        /// Mirrors what `releaseStaticBlocks` does through the deload facet walk,
        /// but as the direct `TypedCold::release()` (columns AND throw-away index)
        /// each container teardown path uses. Residency is the caller's
        /// precondition. Add each migrated cold container here.
        void releaseAllCold() {
            encodedMap.release();
            normalizedEncodedKeys.release();
            normalizedEncodedSubkeys.release();
            normalizedEncodedSubkeysMinusOne.release();
            normalizedEncodedSubkeysMinusTwo.release();
            admissionMap.release();
            admissionStatusMap.release();
            consumedAdmissionKeys.release();
            varsInAdmissionMapKeys.release();
            rejectedMap.release();
            revisitInProgress.release();
            admissionMapIntegration.release();
            rejectedMapIntegration.release();
            varsInAdmissionMapIntegrationKeys.release();
            varsInRejectedMapIntegrationKeys.release();
            admissionSetIntegration.release();
            triggersForAdmissionSetIntegration.release();
            productsOfRecursionIds.release();
            originals.release();
            remainingArgsNormalizedEncodedMap.release();
            remainingArgsReverseIndex.clear();
        }

        void clear() {
            encodedMap.resetToFresh();
            remainingArgsNormalizedEncodedMap.resetToFresh();
            remainingArgsReverseIndex.clear();
            normalizedEncodedKeys.resetToFresh();
            normalizedEncodedSubkeys.resetToFresh();
            normalizedEncodedSubkeysMinusOne.resetToFresh();
            normalizedEncodedSubkeysMinusTwo.resetToFresh();
            maxKeyLength = 0;
            originals.resetToFresh();
            admissionMap.resetToFresh();
            admissionMapIntegration.resetToFresh();
            admissionSetIntegration.resetToFresh();
            triggersForAdmissionSetIntegration.resetToFresh();
            rejectedMap.resetToFresh();
            rejectedMapIntegration.resetToFresh();
            varsInRejectedMapIntegrationKeys.resetToFresh();
            varsInAdmissionMapKeys.resetToFresh();
            varsInAdmissionMapIntegrationKeys.resetToFresh();
            admissionStatusMap.resetToFresh();
            productsOfRecursionIds.resetToFresh();
            consumedAdmissionKeys.resetToFresh();
            revisitInProgress.resetToFresh();
        }

        /// @brief Enumerate this instance's COLD (deload-visited) facets at
        ///        `base + offset` tags — the seam the deload uses to reach the
        ///        cold `HashMemory` instances during the Part-C migration.
        ///
        /// @details
        /// `HashMemory` is not yet an `LbMemory` member (it still holds heap
        /// members that need complete `memory.hpp` types), so `Memory` wraps
        /// each facet this emits as a `lbdeload::DeloadColumn` and hands the
        /// list to the deload alongside `LbMemory`. Empty until `encodedMap`
        /// migrates onto the typed cold facade (Part C commit 1b); thereafter
        /// each migrated container contributes its facets at consecutive
        /// `base + offset` tags, ascending. When every member is migrated,
        /// `HashMemory` moves INTO `LbMemory` and this is walked by
        /// `LbMemory::visitContainers` instead.
        ///
        /// @tparam Visitor Callable `(uint32_t tag, facet&)`.
        /// @param base  The instance's reserved tag-block base
        ///              (`kOverallHashMemoryDeloadBase` / `..Local..`).
        /// @param visit The per-facet visitor.
        /// @see `kOverallHashMemoryDeloadBase`, `lbdeload::DeloadColumn`.
        // Enumerate this instance's COLD facets at base+offset tags. Shared
        // body for the const + non-const overloads -- the deload DUMP reads a
        // const LbMemory&, the LOAD / RELEASE walk a mutable one (now that
        // HashMemory is an LbMemory member, D-147).
        template <typename Self, typename Visitor>
        static void visitContainersImpl(Self& self, uint32_t base, Visitor&& visit) {
            // encodedMap: 5 facets at base+0..4 (key Lengths+Bytes, value
            // RunStarts+BlobStarts+BlobPool). Further migrated containers
            // append after (base+5..).
            visit(base + 0u, self.encodedMapLengths);
            visit(base + 1u, self.encodedMapBytes);
            visit(base + 2u, self.encodedMapRunStarts);
            visit(base + 3u, self.encodedMapBlobStarts);
            visit(base + 4u, self.encodedMapBlobPool);
            // The four owner-set maps: 5 facets each at base+5..24.
            visit(base + 5u, self.normalizedEncodedKeysLengths);
            visit(base + 6u, self.normalizedEncodedKeysBytes);
            visit(base + 7u, self.normalizedEncodedKeysRunStarts);
            visit(base + 8u, self.normalizedEncodedKeysBlobStarts);
            visit(base + 9u, self.normalizedEncodedKeysBlobPool);
            visit(base + 10u, self.normalizedEncodedSubkeysLengths);
            visit(base + 11u, self.normalizedEncodedSubkeysBytes);
            visit(base + 12u, self.normalizedEncodedSubkeysRunStarts);
            visit(base + 13u, self.normalizedEncodedSubkeysBlobStarts);
            visit(base + 14u, self.normalizedEncodedSubkeysBlobPool);
            visit(base + 15u, self.normalizedEncodedSubkeysMinusOneLengths);
            visit(base + 16u, self.normalizedEncodedSubkeysMinusOneBytes);
            visit(base + 17u, self.normalizedEncodedSubkeysMinusOneRunStarts);
            visit(base + 18u, self.normalizedEncodedSubkeysMinusOneBlobStarts);
            visit(base + 19u, self.normalizedEncodedSubkeysMinusOneBlobPool);
            visit(base + 20u, self.normalizedEncodedSubkeysMinusTwoLengths);
            visit(base + 21u, self.normalizedEncodedSubkeysMinusTwoBytes);
            visit(base + 22u, self.normalizedEncodedSubkeysMinusTwoRunStarts);
            visit(base + 23u, self.normalizedEncodedSubkeysMinusTwoBlobStarts);
            visit(base + 24u, self.normalizedEncodedSubkeysMinusTwoBlobPool);
            // Algebra admission subsystem (D-172):
            // admissionMap 4 facets at base+25..28, admissionStatusMap 2 at
            // base+29..30, consumedAdmissionKeys 1 at base+31,
            // varsInAdmissionMapKeys 1 at base+32. Rejection side: rejectedMap
            // 4 facets at base+33..36, revisitInProgress 1 at base+37.
            visit(base + 25u, self.admissionMapKeys);
            visit(base + 26u, self.admissionMapRunStarts);
            visit(base + 27u, self.admissionMapBlobStarts);
            visit(base + 28u, self.admissionMapBlobPool);
            visit(base + 29u, self.admissionStatusMapKeys);
            visit(base + 30u, self.admissionStatusMapValues);
            visit(base + 31u, self.consumedAdmissionKeysKeys);
            visit(base + 32u, self.varsInAdmissionMapKeysKeys);
            visit(base + 33u, self.rejectedMapKeys);
            visit(base + 34u, self.rejectedMapRunStarts);
            visit(base + 35u, self.rejectedMapBlobStarts);
            visit(base + 36u, self.rejectedMapBlobPool);
            visit(base + 37u, self.revisitInProgressKeys);
            // Integration admission/rejection subsystem
            // (D-172): admissionMapIntegration 4
            // facets at base+38..41, rejectedMapIntegration 4 at base+42..45,
            // varsInAdmissionMapIntegrationKeys 1 at base+46,
            // varsInRejectedMapIntegrationKeys 1 at base+47.
            visit(base + 38u, self.admissionMapIntegrationKeys);
            visit(base + 39u, self.admissionMapIntegrationRunStarts);
            visit(base + 40u, self.admissionMapIntegrationBlobStarts);
            visit(base + 41u, self.admissionMapIntegrationBlobPool);
            visit(base + 42u, self.rejectedMapIntegrationKeys);
            visit(base + 43u, self.rejectedMapIntegrationRunStarts);
            visit(base + 44u, self.rejectedMapIntegrationBlobStarts);
            visit(base + 45u, self.rejectedMapIntegrationBlobPool);
            visit(base + 46u, self.varsInAdmissionMapIntegrationKeysKeys);
            visit(base + 47u, self.varsInRejectedMapIntegrationKeysKeys);
            // Residual statified sets (D-173):
            // admissionSetIntegration base+48, triggersForAdmissionSetIntegration
            // base+49, productsOfRecursionIds base+50.
            visit(base + 48u, self.admissionSetIntegrationKeys);
            visit(base + 49u, self.triggersForAdmissionSetIntegrationKeys);
            visit(base + 50u, self.productsOfRecursionIdsKeys);
            // originals byte-key set (D-173): 2 facets.
            visit(base + 51u, self.originalsLengths);
            visit(base + 52u, self.originalsBytes);
            // remainingArgsNormalizedEncodedMap byte-key blob map
            // (D-173): 5 facets.
            visit(base + 53u, self.remainingArgsNormalizedEncodedMapLengths);
            visit(base + 54u, self.remainingArgsNormalizedEncodedMapBytes);
            visit(base + 55u, self.remainingArgsNormalizedEncodedMapRunStarts);
            visit(base + 56u, self.remainingArgsNormalizedEncodedMapBlobStarts);
            visit(base + 57u, self.remainingArgsNormalizedEncodedMapBlobPool);
        }

        template <typename Visitor>
        void visitContainers(uint32_t base, Visitor&& visit) {
            visitContainersImpl(*this, base, std::forward<Visitor>(visit));
        }

        template <typename Visitor>
        void visitContainers(uint32_t base, Visitor&& visit) const {
            visitContainersImpl(*this, base, std::forward<Visitor>(visit));
        }
    };

}
