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
#include "rule_index_staging.hpp"

#include <utility>

namespace gl {

    // Record value types -- fully defined in memory.hpp. Forward-declared here
    // because the TypedColdBlobMap facade names its Record type only inside
    // member-template methods (assignRun / recordsAt / appendRecord), which
    // instantiate at the call sites in memory.cpp / prover.hpp (both include
    // memory.hpp), never inside this struct's own definition.
    struct LocalMemoryValue;
    struct OwnerSet;
    struct RuleOwnerRec;
    struct AdmissionMapValue;
    struct IntegrationEntry;
    struct RejectedMapValue;
    struct RejectedMapIntegrationValue;
    struct RejectedMapOrdisValue;
    struct AdmissionMapOrdis2Value;

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
        // insertRemainingArgsNormKey installs, rebuilt on canonical reload,
        // cleared at the canonical release seam, captured verbatim by the raw
        // image (I-154). Never scope-wiped (the whole-key set is not either).
        // It inverts checkLocalEncodedMemoryStatic's candidate enumeration from
        // an O(keys) forward scan to one hash probe.
        ReverseArgsIndex remainingArgsReverseIndex;
        // The two normalized-key indexes (I-49). normalizedEncodedKeys maps
        // every whole key to the run of the rules that installed it
        // (RuleOwnerRec, sorted-unique, addOwnerToRun via addWholeKeyOwner);
        // the request generator's emission gate is a bare lookup
        // (ExpressionAnalyzer::wholeKeyPresent). A rule's removal deletes its
        // owner and an owner-less key is erased at the end of the apply
        // (eraseEmptyRuns); never scope-wiped. Five deload facets: the key
        // Lengths+Bytes at 5/6, the owner run's RunStarts+BlobStarts+BlobPool
        // appended at 72..74 (append-only tags). normalizedEncodedSubkeys is
        // the growable-subkey record map (bytes key NormKey -> ONE OwnerSet
        // blob per key: the loose byte + the owners' u_ signatures for the
        // D-120 growth prune + the owner list behind them, run-length-1,
        // whole-value replace; read through the no-alloc byte peek
        // subkeyUSatisfied -> peekRecordBytes + OwnerSetBlob, which never
        // reaches the owner section; written by mergeSubkeySignatures /
        // addShortSubkeyOwner; never scope-wiped; 5 deload facets: key
        // Lengths+Bytes, value RunStarts+BlobStarts+BlobPool).
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
        // Raised by every owner removal (a rule leaving hash memory), consumed
        // by ExpressionAnalyzer::eraseOwnerlessEntries at the end of the
        // apply: the one compacting pass per owner map runs only on an
        // instance that lost an owner since the last pass. Transient
        // bookkeeping, never deloaded (a removal and its erasure sit inside
        // one single-threaded seam).
        bool ownerlessPending = false;
        // --- shared members (path-independent) ---
        // Implication chains (premises + head) as ruleInterner id vectors
        // (D-133). Written at install; iterated (decoded lex-sorted) by the
        // integration premise-matcher and the dump; the orphan check erases a
        // chain once no LMV references it.
        // Statified COLD set (D-173): byte-key set,
        // one IdVecKey blob per chain. Two facets: key Lengths + Bytes.
        // Owner lists (RuleOwnerRec run per chain): the same chain text can be
        // installed by several rules (one per scope), so removal deletes one
        // owner and erases the chain only when no owner is left. Five
        // facets: key Lengths+Bytes at 51/52, the owner run's columns
        // appended at 75..77.
        TypedColdBlobMap<IdVecKey, RuleOwnerRec> originals;
        TypedColdBlobMap<IdVecKey, RuleOwnerRec>::LengthsView originalsLengths;
        TypedColdBlobMap<IdVecKey, RuleOwnerRec>::BytesView originalsBytes;
        TypedColdBlobMap<IdVecKey, RuleOwnerRec>::RunStartsView originalsRunStarts;
        TypedColdBlobMap<IdVecKey, RuleOwnerRec>::BlobStartsView originalsBlobStarts;
        TypedColdBlobMap<IdVecKey, RuleOwnerRec>::BlobPoolView originalsBlobPool;
        // Owner lists of the remaining-args index: one owner run per
        // (arg-set, NormKey) EDGE of remainingArgsNormalizedEncodedMap, keyed
        // by the edge's bytes (Int16SetKey bytes ++ NormKey bytes as one
        // IdVecKey-shaped id run). The forward run itself stays a pure NormKey
        // run (its readers and the derived reverse index are untouched); an
        // edge whose owners run empty leaves the forward run and the reverse
        // index. Five facets at 78..82.
        TypedColdBlobMap<IdVecKey, RuleOwnerRec> remainingArgsOwners;
        TypedColdBlobMap<IdVecKey, RuleOwnerRec>::LengthsView remainingArgsOwnersLengths;
        TypedColdBlobMap<IdVecKey, RuleOwnerRec>::BytesView remainingArgsOwnersBytes;
        TypedColdBlobMap<IdVecKey, RuleOwnerRec>::RunStartsView remainingArgsOwnersRunStarts;
        TypedColdBlobMap<IdVecKey, RuleOwnerRec>::BlobStartsView remainingArgsOwnersBlobStarts;
        TypedColdBlobMap<IdVecKey, RuleOwnerRec>::BlobPoolView remainingArgsOwnersBlobPool;
        // Owner lists of the multiplication copies: one run per copy owner
        // (the copy text's ruleInterner id + the install scope — the packed
        // RuleOwner as eight key bytes) naming the RULES (recorded text id +
        // scope, same packing) whose install produced that copy. Two rules
        // can multiply into one copy text; the copy's index entries and LMVs
        // leave only with its last rule (I-49). Five facets at 83..87.
        TypedColdBlobMap<IdVecKey, RuleOwnerRec> copyOwners;
        TypedColdBlobMap<IdVecKey, RuleOwnerRec>::LengthsView copyOwnersLengths;
        TypedColdBlobMap<IdVecKey, RuleOwnerRec>::BytesView copyOwnersBytes;
        TypedColdBlobMap<IdVecKey, RuleOwnerRec>::RunStartsView copyOwnersRunStarts;
        TypedColdBlobMap<IdVecKey, RuleOwnerRec>::BlobStartsView copyOwnersBlobStarts;
        TypedColdBlobMap<IdVecKey, RuleOwnerRec>::BlobPoolView copyOwnersBlobPool;
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
        // Parked or-cohorts (the admission-based ordis polarity): an or head
        // whose cohort-opening probe found no admission demand parks here,
        // one entry per operator-based disjunct product template at the
        // cohort parent's validity. Packed (templateId, validityId) keys —
        // the SAME admission key space, so revival at key-gain is a direct
        // probe. Values = RejectedMapOrdisValue (or statement + levels),
        // canonical run under DecodedRejectedOrdisValueLess. Mirror contract
        // with rejectedMap: same RMW / wipe / deload / dump discipline;
        // NEVER written directly by the equi-class hook (I-37); the revival
        // consumer revisitRejectedOrdis erases the consumed cohort. Four
        // facets: key, run-starts, blob-starts, blob-pool.
        TypedColdBlobMap<int64_t, RejectedMapOrdisValue> rejectedMapOrdis;
        TypedColdBlobMap<int64_t, RejectedMapOrdisValue>::KeysView rejectedMapOrdisKeys;
        TypedColdBlobMap<int64_t, RejectedMapOrdisValue>::RunStartsView rejectedMapOrdisRunStarts;
        TypedColdBlobMap<int64_t, RejectedMapOrdisValue>::BlobStartsView rejectedMapOrdisBlobStarts;
        TypedColdBlobMap<int64_t, RejectedMapOrdisValue>::BlobPoolView rejectedMapOrdisBlobPool;
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
        // Statified COLD set (D-172). Re-entrant
        // revival guard (empty at barriers). One facet: key.
        TypedColdSet<int64_t> revisitInProgress;
        TypedColdSet<int64_t>::KeysView revisitInProgressKeys;
        // Re-entrant guard for revisitRejectedOrdis — DEDICATED (never
        // shared with revisitInProgress: a concurrent general revisit of
        // the same packed key must not swallow an ordis wake). Empty at
        // barriers. One facet: key.
        TypedColdSet<int64_t> ordisRevisitInProgress;
        TypedColdSet<int64_t>::KeysView ordisRevisitInProgressKeys;

        // Ordis2 demand admission map (D-267):
        // keyed by packed (templateId, validityId) where the template half
        // is a GROUND compound premise text (the one missing premise of a
        // rule that matched everything else); a parked or head in
        // rejectedMapOrdis2 under the SAME key opens its cohort (route (c)),
        // consuming both sides. Written only at the post-fixpoint drain;
        // separate from the algebra and integration admission maps by
        // design (incompatible key languages).
        TypedColdBlobMap<int64_t, AdmissionMapOrdis2Value> admissionMapOrdis2;
        TypedColdBlobMap<int64_t, AdmissionMapOrdis2Value>::KeysView admissionMapOrdis2Keys;
        TypedColdBlobMap<int64_t, AdmissionMapOrdis2Value>::RunStartsView admissionMapOrdis2RunStarts;
        TypedColdBlobMap<int64_t, AdmissionMapOrdis2Value>::BlobStartsView admissionMapOrdis2BlobStarts;
        TypedColdBlobMap<int64_t, AdmissionMapOrdis2Value>::BlobPoolView admissionMapOrdis2BlobPool;

        // Ordis2 park index — the rejected half of the pair
        // (D-267): the SAME or heads that park in
        // rejectedMapOrdis additionally file here under each eligible
        // disjunct's clean GROUND text (polarity verbatim, I-175) at the
        // cohort parent's validity — the demand map's key language, so the
        // drain's wake is a plain key rendezvous. Value type REUSES
        // RejectedMapOrdisValue (same semantic content: or statement id +
        // seed levels), inheriting the whole codec family.
        TypedColdBlobMap<int64_t, RejectedMapOrdisValue> rejectedMapOrdis2;
        TypedColdBlobMap<int64_t, RejectedMapOrdisValue>::KeysView rejectedMapOrdis2Keys;
        TypedColdBlobMap<int64_t, RejectedMapOrdisValue>::RunStartsView rejectedMapOrdis2RunStarts;
        TypedColdBlobMap<int64_t, RejectedMapOrdisValue>::BlobStartsView rejectedMapOrdis2BlobStarts;
        TypedColdBlobMap<int64_t, RejectedMapOrdisValue>::BlobPoolView rejectedMapOrdis2BlobPool;
        // Re-entrant guard for revisitRejectedOrdis2 — DEDICATED (never
        // shared with ordisRevisitInProgress: a concurrent old-map wake of
        // the same packed key must not swallow an ordis2 wake). Empty at
        // barriers. One facet: key.
        TypedColdSet<int64_t> ordis2RevisitInProgress;
        TypedColdSet<int64_t>::KeysView ordis2RevisitInProgressKeys;

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
              originals(arena, dirty),
              originalsLengths(&originals.inner()),
              originalsBytes(&originals.inner()),
              originalsRunStarts(&originals.inner()),
              originalsBlobStarts(&originals.inner()),
              originalsBlobPool(&originals.inner()),
              remainingArgsOwners(arena, dirty),
              remainingArgsOwnersLengths(&remainingArgsOwners.inner()),
              remainingArgsOwnersBytes(&remainingArgsOwners.inner()),
              remainingArgsOwnersRunStarts(&remainingArgsOwners.inner()),
              remainingArgsOwnersBlobStarts(&remainingArgsOwners.inner()),
              remainingArgsOwnersBlobPool(&remainingArgsOwners.inner()),
              copyOwners(arena, dirty),
              copyOwnersLengths(&copyOwners.inner()),
              copyOwnersBytes(&copyOwners.inner()),
              copyOwnersRunStarts(&copyOwners.inner()),
              copyOwnersBlobStarts(&copyOwners.inner()),
              copyOwnersBlobPool(&copyOwners.inner()),
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
              rejectedMapOrdis(arena, dirty),
              rejectedMapOrdisKeys(&rejectedMapOrdis.inner()),
              rejectedMapOrdisRunStarts(&rejectedMapOrdis.inner()),
              rejectedMapOrdisBlobStarts(&rejectedMapOrdis.inner()),
              rejectedMapOrdisBlobPool(&rejectedMapOrdis.inner()),
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
              revisitInProgress(arena, dirty),
              revisitInProgressKeys(&revisitInProgress.inner()),
              ordisRevisitInProgress(arena, dirty),
              ordisRevisitInProgressKeys(&ordisRevisitInProgress.inner()),
              admissionMapOrdis2(arena, dirty),
              admissionMapOrdis2Keys(&admissionMapOrdis2.inner()),
              admissionMapOrdis2RunStarts(&admissionMapOrdis2.inner()),
              admissionMapOrdis2BlobStarts(&admissionMapOrdis2.inner()),
              admissionMapOrdis2BlobPool(&admissionMapOrdis2.inner()),
              rejectedMapOrdis2(arena, dirty),
              rejectedMapOrdis2Keys(&rejectedMapOrdis2.inner()),
              rejectedMapOrdis2RunStarts(&rejectedMapOrdis2.inner()),
              rejectedMapOrdis2BlobStarts(&rejectedMapOrdis2.inner()),
              rejectedMapOrdis2BlobPool(&rejectedMapOrdis2.inner()),
              ordis2RevisitInProgress(arena, dirty),
              ordis2RevisitInProgressKeys(&ordis2RevisitInProgress.inner())
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
                 + admissionMap.liveBytes()
                 + admissionStatusMap.liveBytes()
                 + varsInAdmissionMapKeys.liveBytes()
                 + rejectedMap.liveBytes()
                 + revisitInProgress.liveBytes()
                 + ordisRevisitInProgress.liveBytes()
                 + admissionMapOrdis2.liveBytes()
                 + rejectedMapOrdis2.liveBytes()
                 + ordis2RevisitInProgress.liveBytes()
                 + admissionMapIntegration.liveBytes()
                 + rejectedMapIntegration.liveBytes()
                 + rejectedMapOrdis.liveBytes()
                 + varsInAdmissionMapIntegrationKeys.liveBytes()
                 + varsInRejectedMapIntegrationKeys.liveBytes()
                 + admissionSetIntegration.liveBytes()
                 + triggersForAdmissionSetIntegration.liveBytes()
                 + productsOfRecursionIds.liveBytes()
                 + originals.liveBytes()
                 + remainingArgsNormalizedEncodedMap.liveBytes()
                 + remainingArgsOwners.liveBytes()
                 + copyOwners.liveBytes()
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
            admissionMap.release();
            admissionStatusMap.release();
            varsInAdmissionMapKeys.release();
            rejectedMap.release();
            revisitInProgress.release();
            ordisRevisitInProgress.release();
            admissionMapOrdis2.release();
            rejectedMapOrdis2.release();
            ordis2RevisitInProgress.release();
            admissionMapIntegration.release();
            rejectedMapIntegration.release();
            rejectedMapOrdis.release();
            varsInAdmissionMapIntegrationKeys.release();
            varsInRejectedMapIntegrationKeys.release();
            admissionSetIntegration.release();
            triggersForAdmissionSetIntegration.release();
            productsOfRecursionIds.release();
            originals.release();
            remainingArgsNormalizedEncodedMap.release();
            remainingArgsOwners.release();
            copyOwners.release();
            remainingArgsReverseIndex.clear();
        }

        void clear() {
            encodedMap.resetToFresh();
            remainingArgsNormalizedEncodedMap.resetToFresh();
            remainingArgsOwners.resetToFresh();
            copyOwners.resetToFresh();
            remainingArgsReverseIndex.clear();
            normalizedEncodedKeys.resetToFresh();
            normalizedEncodedSubkeys.resetToFresh();
            maxKeyLength = 0;
            ownerlessPending = false;
            originals.resetToFresh();
            admissionMap.resetToFresh();
            admissionMapIntegration.resetToFresh();
            admissionSetIntegration.resetToFresh();
            triggersForAdmissionSetIntegration.resetToFresh();
            rejectedMap.resetToFresh();
            rejectedMapIntegration.resetToFresh();
            rejectedMapOrdis.resetToFresh();
            varsInRejectedMapIntegrationKeys.resetToFresh();
            varsInAdmissionMapKeys.resetToFresh();
            varsInAdmissionMapIntegrationKeys.resetToFresh();
            admissionStatusMap.resetToFresh();
            productsOfRecursionIds.resetToFresh();
            revisitInProgress.resetToFresh();
            ordisRevisitInProgress.resetToFresh();
            admissionMapOrdis2.resetToFresh();
            rejectedMapOrdis2.resetToFresh();
            ordis2RevisitInProgress.resetToFresh();
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
            // The whole-key set: 2 facets at base+5..6 (base+7..9, its former
            // value facets, are a hole); the subkey map: 5 facets at base+10..14.
            // The minus-one / minus-two maps that held base+15..24 are deleted;
            // the band keeps a hole there rather than renumbering the survivors.
            visit(base + 5u, self.normalizedEncodedKeysLengths);
            visit(base + 6u, self.normalizedEncodedKeysBytes);
            visit(base + 10u, self.normalizedEncodedSubkeysLengths);
            visit(base + 11u, self.normalizedEncodedSubkeysBytes);
            visit(base + 12u, self.normalizedEncodedSubkeysRunStarts);
            visit(base + 13u, self.normalizedEncodedSubkeysBlobStarts);
            visit(base + 14u, self.normalizedEncodedSubkeysBlobPool);
            // Algebra admission subsystem (D-172):
            // admissionMap 4 facets at base+25..28, admissionStatusMap 2 at
            // base+29..30, varsInAdmissionMapKeys 1 at base+32 (base+31 is
            // permanently unused). Rejection side: rejectedMap
            // 4 facets at base+33..36, revisitInProgress 1 at base+37.
            visit(base + 25u, self.admissionMapKeys);
            visit(base + 26u, self.admissionMapRunStarts);
            visit(base + 27u, self.admissionMapBlobStarts);
            visit(base + 28u, self.admissionMapBlobPool);
            visit(base + 29u, self.admissionStatusMapKeys);
            visit(base + 30u, self.admissionStatusMapValues);
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
            // Parked or-cohorts (rejectedMapOrdis, the admission-based ordis
            // polarity): 4 facets at base+58..61, appended — the band is
            // append-only, key tag first so reload rebuilds the index before
            // the parallel columns.
            visit(base + 58u, self.rejectedMapOrdisKeys);
            visit(base + 59u, self.rejectedMapOrdisRunStarts);
            visit(base + 60u, self.rejectedMapOrdisBlobStarts);
            visit(base + 61u, self.rejectedMapOrdisBlobPool);
            // ordisRevisitInProgress re-entrancy guard: 1 facet at base+62.
            visit(base + 62u, self.ordisRevisitInProgressKeys);
            // Ordis2 demand admission map
            // (D-267): 4 facets at base+63..66,
            // appended per the band's append-only discipline (D-174).
            visit(base + 63u, self.admissionMapOrdis2Keys);
            visit(base + 64u, self.admissionMapOrdis2RunStarts);
            visit(base + 65u, self.admissionMapOrdis2BlobStarts);
            visit(base + 66u, self.admissionMapOrdis2BlobPool);
            // Ordis2 park index + its revisit guard
            // (D-267): 5 facets at base+67..71,
            // appended per the band's append-only discipline (D-174).
            visit(base + 67u, self.rejectedMapOrdis2Keys);
            visit(base + 68u, self.rejectedMapOrdis2RunStarts);
            visit(base + 69u, self.rejectedMapOrdis2BlobStarts);
            visit(base + 70u, self.rejectedMapOrdis2BlobPool);
            visit(base + 71u, self.ordis2RevisitInProgressKeys);
            // Owner-run columns (append-only tags, after every older facet so
            // the enumeration stays ascending): the whole-key owners at
            // 72..74 and the chain owners at 75..77 extend the key facets
            // visited at 5/6 and 51/52; the remaining-args edge owners are a
            // whole map at 78..82, the multiplication-copy owners at 83..87.
            visit(base + 72u, self.normalizedEncodedKeysRunStarts);
            visit(base + 73u, self.normalizedEncodedKeysBlobStarts);
            visit(base + 74u, self.normalizedEncodedKeysBlobPool);
            visit(base + 75u, self.originalsRunStarts);
            visit(base + 76u, self.originalsBlobStarts);
            visit(base + 77u, self.originalsBlobPool);
            visit(base + 78u, self.remainingArgsOwnersLengths);
            visit(base + 79u, self.remainingArgsOwnersBytes);
            visit(base + 80u, self.remainingArgsOwnersRunStarts);
            visit(base + 81u, self.remainingArgsOwnersBlobStarts);
            visit(base + 82u, self.remainingArgsOwnersBlobPool);
            visit(base + 83u, self.copyOwnersLengths);
            visit(base + 84u, self.copyOwnersBytes);
            visit(base + 85u, self.copyOwnersRunStarts);
            visit(base + 86u, self.copyOwnersBlobStarts);
            visit(base + 87u, self.copyOwnersBlobPool);
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
