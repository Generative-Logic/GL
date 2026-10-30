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

// HashMemory stores the rule, admission and rejected-source columns of one
// logical block. Its four instances share the block's cold arena. Algorithm
// owners access public columns; reset and tagged visitation preserve the
// established allocation and deload contracts.

#include "typed_cold_map.hpp"
#include "reverse_args_index.hpp"
#include "originals_head_index.hpp"
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
        OriginalsHeadIndex originalsHeadIndex;
        // The two normalized-key indexes (I-49). normalizedEncodedKeys maps
        // every whole key to the run of the rules that installed it
        // (RuleOwnerRec, sorted-unique, addOwnerToRun via addWholeKeyOwner);
        // the request generator's emission gate is a bare lookup
        // (HashRuleIndex::wholeKeyPresent). A rule's removal deletes its
        // owner and an owner-less key is erased at the end of the apply
        // (eraseEmptyRuns); never scope-wiped. Five deload facets: the key
        // Lengths+Bytes at 5/6, the owner run's RunStarts+BlobStarts+BlobPool
        // appended at 72..74 (append-only tags). normalizedEncodedSubkeys is
        // the growable-subkey record map (bytes key NormKey -> ONE OwnerSet
        // blob per key: empty below three premises; otherwise the loose byte,
        // u_ signatures for the D-120 growth prune and their owner list, run-length-1,
        // whole-value replace; read through the no-alloc byte peek
        // subkeyUSatisfied -> peekRecordBytes + OwnerSetBlob, which never
        // reaches the owner section; written by mergeSubkeySignatures /
        // ensureEmptySubkeyRecord; never scope-wiped; 5 deload facets: key
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
        // by HashRuleIndex::eraseOwnerlessEntries at the end of the
        // apply: the one compacting pass per owner map runs only on an
        // instance that lost an owner since the last pass. Transient
        // bookkeeping, never deloaded (a removal and its erasure sit inside
        // one single-threaded seam).
        bool ownerlessPending = false;
        // The goal-rescan floor of checkNecessityForEquality: every goal of
        // the LB has been visited against the `originals` chains
        // 1..goalScanChainMark, so the rescan after an implication install
        // visits only the chains above it (a repeated (goal, chain) visit
        // changes nothing). Raised at the end of each rescan; zeroed when a
        // goal enters the registry without the goal door's full scan
        // (sanitizeToBeProved's re-key), at every teardown, and remapped by
        // the owner-less sweep, whose compaction keeps the survivors' order.
        // Resident scalar, never deloaded: a reload keeps the chain ids.
        int32_t goalScanChainMark = 0;
        // The contradiction-scope demand's chain floor
        // (D-560): every contradiction scope the
        // demand drain has bootstrapped has been matched against the
        // `originals` chains 1..contradictionChainMark, so a drain matches
        // only the chains above it against the facts those scopes see.
        // Raised at the end of each drain, zeroed at every teardown and
        // remapped by the owner-less sweep exactly as the goal-rescan floor.
        // Resident scalar, never deloaded.
        int32_t contradictionChainMark = 0;
        // --- shared members (path-independent) ---
        // Implication chains (premises + head) as ruleInterner id vectors
        // (D-133). Written at install; iterated in ascending chain-id (install)
        // order by the integration premise-matcher, the necessity check and
        // the goal rescan, decoded and sorted by the dump; the orphan check
        // erases a chain once no LMV references it.
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
        // Premise patterns of the installed rules, read by the request
        // statement filter (premisePatternAdmits). Three key families in one
        // owner-run map: a premise SHAPE key (kind 1: signed predicate plus
        // first-appearance argument numbering) -> the run of distinct u_ slot
        // masks seen for that shape; a PATTERN key (kind 2: signed predicate
        // plus, per slot, the literal id of a u_ argument or the numbered
        // variable) -> the run of scope-major owners (install scope in the
        // high half, copy id in the low half); a CLOSED-SCOPE key (kind 3:
        // the scope) -> one record while a subtree wipe has removed the
        // scope's rules from encodedMap and no install has reopened it.
        // Installs stage per copy and premise; a removal deletes the owner and
        // the owner-less sweep erases an emptied key; the wipe keeps the
        // owners and records the closed scope. Mask runs only grow. Five
        // facets at 93..97.
        TypedColdBlobMap<IdVecKey, RuleOwnerRec> premisePatterns;
        TypedColdBlobMap<IdVecKey, RuleOwnerRec>::LengthsView premisePatternsLengths;
        TypedColdBlobMap<IdVecKey, RuleOwnerRec>::BytesView premisePatternsBytes;
        TypedColdBlobMap<IdVecKey, RuleOwnerRec>::RunStartsView premisePatternsRunStarts;
        TypedColdBlobMap<IdVecKey, RuleOwnerRec>::BlobStartsView premisePatternsBlobStarts;
        TypedColdBlobMap<IdVecKey, RuleOwnerRec>::BlobPoolView premisePatternsBlobPool;
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
        // Double-marker integration demands (D-492):
        // packed (templateId, validityId) key of a flattened root goal's OR
        // disjunct with its two open arguments as `marker1` / `marker2` (the
        // other arguments `u_`) -> 0. The value carries no information; the
        // map shape stays because the Rule-14 hashburst dump reads it as a
        // map. No instruction payload: a hit only admits an `int_` witness. Statified COLD single-value map, scope-wiped like the
        // admission set. Two facets: key, value. Only ever used via
        // overallHashMemory.
        TypedColdMap<int64_t, int64_t> admissionMapIntegrationDouble;
        TypedColdMap<int64_t, int64_t>::KeysView admissionMapIntegrationDoubleKeys;
        TypedColdMap<int64_t, int64_t>::ValuesView admissionMapIntegrationDoubleValues;
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

        // Parked implication subproofs (D-406): an
        // implication element of an and whose sibling non-implication element
        // is not yet known at the preparing scope is kept here instead of
        // opening its boundary subproof. Key: packed (templateId, validityId)
        // of the unknown element's clean ground text at that scope. Value:
        // the IntegrationEntry family (canonical run under
        // DecodedInstructionLess) — one stored instruction per parked
        // implication: entity 0 the implication, entities 1..k its parent ands
        // with their gating elements, markedGoal the embedded root goal.
        // Same RMW / wipe / deload / dump discipline as rejectedMapOrdis; the
        // revival revisitRejectedSubproof erases the consumed run. Four facets:
        // key, run-starts, blob-starts, blob-pool.
        TypedColdBlobMap<int64_t, IntegrationEntry> rejectedMapSubproof;
        TypedColdBlobMap<int64_t, IntegrationEntry>::KeysView rejectedMapSubproofKeys;
        TypedColdBlobMap<int64_t, IntegrationEntry>::RunStartsView rejectedMapSubproofRunStarts;
        TypedColdBlobMap<int64_t, IntegrationEntry>::BlobStartsView rejectedMapSubproofBlobStarts;
        TypedColdBlobMap<int64_t, IntegrationEntry>::BlobPoolView rejectedMapSubproofBlobPool;
        // Re-entrant guard for revisitRejectedSubproof — DEDICATED. Empty at
        // barriers. One facet: key.
        TypedColdSet<int64_t> subproofRevisitInProgress;
        TypedColdSet<int64_t>::KeysView subproofRevisitInProgressKeys;

    public:
        /// @brief Bind the rule store to its logical block's arena and dirty flag.
        /// @details Every cold container uses the supplied arena and shares the
        /// content-change flag. The four hash stores retain their original arena
        /// ownership, shared blob pools and per-burst reset boundaries.
        /// @param arena Arena backing the instance's cold containers.
        /// @param dirty Shared content-change flag written by those containers.
        /// @return Constructed bound rule store.
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
              originalsHeadIndex(arena),
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
              premisePatterns(arena, dirty),
              premisePatternsLengths(&premisePatterns.inner()),
              premisePatternsBytes(&premisePatterns.inner()),
              premisePatternsRunStarts(&premisePatterns.inner()),
              premisePatternsBlobStarts(&premisePatterns.inner()),
              premisePatternsBlobPool(&premisePatterns.inner()),
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
              admissionMapIntegrationDouble(arena, dirty),
              admissionMapIntegrationDoubleKeys(&admissionMapIntegrationDouble.inner()),
              admissionMapIntegrationDoubleValues(&admissionMapIntegrationDouble.inner()),
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
              ordis2RevisitInProgressKeys(&ordis2RevisitInProgress.inner()),
              rejectedMapSubproof(arena, dirty),
              rejectedMapSubproofKeys(&rejectedMapSubproof.inner()),
              rejectedMapSubproofRunStarts(&rejectedMapSubproof.inner()),
              rejectedMapSubproofBlobStarts(&rejectedMapSubproof.inner()),
              rejectedMapSubproofBlobPool(&rejectedMapSubproof.inner()),
              subproofRevisitInProgress(arena, dirty),
              subproofRevisitInProgressKeys(&subproofRevisitInProgress.inner())
        {}

        // Arena-bound: non-copyable (an arena pointer must not be blindly
        // duplicated). The former swap-with-empty / `= HashMemory()` resets
        // become `resetToFresh()`.
        /// @brief Bind native key payload to the owning block shared store.
        ///
        /// @details Each map keeps its own key identifiers, hash index and owner
        /// runs. Equal canonical bytes share resident payload across the family.
        /// Binding happens once while every participating container is empty.
        ///
        /// @param pool Same-arena store that outlives this hash-memory instance.
        /// @return Nothing.
        /// @invariant Shared storage never merges map membership or scope.
        void bindSharedBytes(SharedBytePool& pool) {
            encodedMap.inner().keyStore().bindSharedBytes(pool);
            normalizedEncodedKeys.inner().keyStore().bindSharedBytes(pool);
            normalizedEncodedSubkeys.inner().keyStore().bindSharedBytes(pool);
            remainingArgsNormalizedEncodedMap.inner().keyStore().bindSharedBytes(pool);
            remainingArgsNormalizedEncodedMap.inner().bindSharedBlobBytes(pool);
            remainingArgsOwners.inner().keyStore().bindSharedBytes(pool);
            copyOwners.inner().keyStore().bindSharedBytes(pool);
            premisePatterns.inner().keyStore().bindSharedBytes(pool);
            remainingArgsReverseIndex.bindSharedBytes(pool);
            encodedMap.inner().bindSharedBlobBytes(pool, true);
            normalizedEncodedSubkeys.inner().bindSharedBlobBytes(pool, true);
            normalizedEncodedKeys.inner().bindSharedBlobBytes(pool, true);
            remainingArgsOwners.inner().bindSharedBlobBytes(pool, true);
            copyOwners.inner().bindSharedBlobBytes(pool, true);
            premisePatterns.inner().bindSharedBlobBytes(pool, true);
        }

        HashMemory(const HashMemory&) = delete;
        HashMemory& operator=(const HashMemory&) = delete;

        /// @brief Reset every rule-store container without moving its arena binding.
        /// @details Calls the existing clear operation so the bound storage can
        /// be reused at the grid-teardown and per-burst boundaries.
        /// @return No value.
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
            return peak_snapshot::measure("encodedMap", encodedMap, [&] { return encodedMap.liveBytes(); })
                 + peak_snapshot::measure("normalizedEncodedKeys", normalizedEncodedKeys, [&] { return normalizedEncodedKeys.liveBytes(); })
                 + peak_snapshot::measure("normalizedEncodedSubkeys", normalizedEncodedSubkeys, [&] { return normalizedEncodedSubkeys.liveBytes(); })
                 + peak_snapshot::measure("admissionMap", admissionMap, [&] { return admissionMap.liveBytes(); })
                 + peak_snapshot::measure("admissionStatusMap", admissionStatusMap, [&] { return admissionStatusMap.liveBytes(); })
                 + peak_snapshot::measure("varsInAdmissionMapKeys", varsInAdmissionMapKeys, [&] { return varsInAdmissionMapKeys.liveBytes(); })
                 + peak_snapshot::measure("rejectedMap", rejectedMap, [&] { return rejectedMap.liveBytes(); })
                 + peak_snapshot::measure("revisitInProgress", revisitInProgress, [&] { return revisitInProgress.liveBytes(); })
                 + peak_snapshot::measure("ordisRevisitInProgress", ordisRevisitInProgress, [&] { return ordisRevisitInProgress.liveBytes(); })
                 + peak_snapshot::measure("admissionMapOrdis2", admissionMapOrdis2, [&] { return admissionMapOrdis2.liveBytes(); })
                 + peak_snapshot::measure("rejectedMapOrdis2", rejectedMapOrdis2, [&] { return rejectedMapOrdis2.liveBytes(); })
                 + peak_snapshot::measure("ordis2RevisitInProgress", ordis2RevisitInProgress, [&] { return ordis2RevisitInProgress.liveBytes(); })
                 + peak_snapshot::measure("rejectedMapSubproof", rejectedMapSubproof, [&] { return rejectedMapSubproof.liveBytes(); })
                 + peak_snapshot::measure("subproofRevisitInProgress", subproofRevisitInProgress, [&] { return subproofRevisitInProgress.liveBytes(); })
                 + peak_snapshot::measure("admissionMapIntegration", admissionMapIntegration, [&] { return admissionMapIntegration.liveBytes(); })
                 + peak_snapshot::measure("rejectedMapIntegration", rejectedMapIntegration, [&] { return rejectedMapIntegration.liveBytes(); })
                 + peak_snapshot::measure("rejectedMapOrdis", rejectedMapOrdis, [&] { return rejectedMapOrdis.liveBytes(); })
                 + peak_snapshot::measure("varsInAdmissionMapIntegrationKeys", varsInAdmissionMapIntegrationKeys, [&] { return varsInAdmissionMapIntegrationKeys.liveBytes(); })
                 + peak_snapshot::measure("varsInRejectedMapIntegrationKeys", varsInRejectedMapIntegrationKeys, [&] { return varsInRejectedMapIntegrationKeys.liveBytes(); })
                 + peak_snapshot::measure("admissionSetIntegration", admissionSetIntegration, [&] { return admissionSetIntegration.liveBytes(); })
                 + peak_snapshot::measure("triggersForAdmissionSetIntegration", triggersForAdmissionSetIntegration, [&] { return triggersForAdmissionSetIntegration.liveBytes(); })
                 + peak_snapshot::measure("admissionMapIntegrationDouble", admissionMapIntegrationDouble, [&] { return admissionMapIntegrationDouble.liveBytes(); })
                 + peak_snapshot::measure("productsOfRecursionIds", productsOfRecursionIds, [&] { return productsOfRecursionIds.liveBytes(); })
                 + peak_snapshot::measure("originals", originals, [&] { return originals.liveBytes(); })
                 + peak_snapshot::measure("remainingArgsNormalizedEncodedMap", remainingArgsNormalizedEncodedMap, [&] { return remainingArgsNormalizedEncodedMap.liveBytes(); })
                 + peak_snapshot::measure("remainingArgsOwners", remainingArgsOwners, [&] { return remainingArgsOwners.liveBytes(); })
                 + peak_snapshot::measure("copyOwners", copyOwners, [&] { return copyOwners.liveBytes(); })
                 + peak_snapshot::measure("premisePatterns", premisePatterns, [&] { return premisePatterns.liveBytes(); })
                 + peak_snapshot::measure("remainingArgsReverseIndex", remainingArgsReverseIndex, [&] { return remainingArgsReverseIndex.liveBytes(); });
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
            rejectedMapSubproof.release();
            subproofRevisitInProgress.release();
            admissionMapIntegration.release();
            rejectedMapIntegration.release();
            rejectedMapOrdis.release();
            varsInAdmissionMapIntegrationKeys.release();
            varsInRejectedMapIntegrationKeys.release();
            admissionSetIntegration.release();
            triggersForAdmissionSetIntegration.release();
            admissionMapIntegrationDouble.release();
            productsOfRecursionIds.release();
            originals.release();
            remainingArgsNormalizedEncodedMap.release();
            remainingArgsOwners.release();
            copyOwners.release();
            premisePatterns.release();
            remainingArgsReverseIndex.clear();
            originalsHeadIndex.clear();
            goalScanChainMark = 0;
            contradictionChainMark = 0;
        }

        void clear() {
            encodedMap.resetToFresh();
            remainingArgsNormalizedEncodedMap.resetToFresh();
            remainingArgsOwners.resetToFresh();
            copyOwners.resetToFresh();
            premisePatterns.resetToFresh();
            remainingArgsReverseIndex.clear();
            originalsHeadIndex.clear();
            normalizedEncodedKeys.resetToFresh();
            normalizedEncodedSubkeys.resetToFresh();
            maxKeyLength = 0;
            ownerlessPending = false;
            goalScanChainMark = 0;
            contradictionChainMark = 0;
            originals.resetToFresh();
            admissionMap.resetToFresh();
            admissionMapIntegration.resetToFresh();
            admissionSetIntegration.resetToFresh();
            triggersForAdmissionSetIntegration.resetToFresh();
            admissionMapIntegrationDouble.resetToFresh();
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
            rejectedMapSubproof.resetToFresh();
            subproofRevisitInProgress.resetToFresh();
        }

        /// @brief Visit the established cold-container facets in deload-tag order.
        /// @details Shared implementation of the const dump walk and mutable
        /// load/release walk. LbMemory supplies the instance's reserved tag base;
        /// all field tags and facet ordering remain fixed.
        /// @tparam Self Const or mutable HashMemory instance.
        /// @tparam Visitor Callable receiving each tag and container facet.
        /// @param self Instance whose facets are visited.
        /// @param base Reserved tag-block base for this instance.
        /// @param visit Operation applied to each tagged facet.
        /// @return No value.
    private:
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
            // Parked implication subproofs + their revisit guard
            // (D-406): 5 facets at base+88..92,
            // appended per the band's append-only discipline (D-174).
            visit(base + 88u, self.rejectedMapSubproofKeys);
            visit(base + 89u, self.rejectedMapSubproofRunStarts);
            visit(base + 90u, self.rejectedMapSubproofBlobStarts);
            visit(base + 91u, self.rejectedMapSubproofBlobPool);
            visit(base + 92u, self.subproofRevisitInProgressKeys);
            // Premise patterns of the installed rules: 5 facets at
            // base+93..97, appended per the band's append-only discipline
            // (D-174).
            visit(base + 93u, self.premisePatternsLengths);
            visit(base + 94u, self.premisePatternsBytes);
            visit(base + 95u, self.premisePatternsRunStarts);
            visit(base + 96u, self.premisePatternsBlobStarts);
            visit(base + 97u, self.premisePatternsBlobPool);
            // Double-marker integration demands
            // (D-492): 2 facets at base+98..99,
            // the band's last tags, appended per the append-only discipline
            // (D-174).
            visit(base + 98u, self.admissionMapIntegrationDoubleKeys);
            visit(base + 99u, self.admissionMapIntegrationDoubleValues);
        }

    public:
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
