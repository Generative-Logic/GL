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

#include "arena_vector.hpp"
#include "paged_vector.hpp"
#include "cold_string_table.hpp"
#include "cold_hash_map.hpp"
#include "typed_cold_map.hpp"
#include "hash_memory.hpp"
#include "int_encoded_expr.hpp"
#include "mail_types.hpp"
#include "cold_mail.hpp"
#include "changed_classes_buffer.hpp"
#include "eq_class_name_caches.hpp"
#include "deloadable_mail_out.hpp"

#include <cstdint>

namespace gl {

    /// @brief One validity scope's place in the per-LB scope forest: its
    ///        immediate parent id and its own payload sub-id.
    ///
    /// @details
    /// The flat per-id node of `NameMap`'s validity metadata — the
    /// parent-pointer forest that replaced the jagged stack/ancestor lists.
    /// `parentId == 0` marks a root (`"main"` and any flat `encode` root); a
    /// non-root's own `_boundary_` payload sub-id is `ownSubId`. Ancestor and
    /// payload-stack queries walk the `parentId` chain. The node is 8 bytes
    /// (two `NameId`s), trivially copyable, streamed verbatim on deload.
    struct ValidityNode {
        NameId parentId;
        NameId ownSubId;
    };

    // The blob-map equivalenceClassesMap stores EquivalenceClass records keyed by
    // a validity id; the record type is named only in TypedCold's member
    // templates (defined in memory.hpp), so a forward declaration suffices here.
    struct EquivalenceClass;
    struct CompactExpansionRec;

    /// @brief The growable per-LB aggregate of statified containers — the
    ///        unit the static memory hierarchy manages and the deload format
    ///        serializes.
    ///
    /// @details
    /// One `LbMemory` bundles the LB's `LbArena` (its bump arena) with every
    /// container that has migrated onto static memory. Its growth contract is
    /// the user-required core of the statification design
    /// (D-174): adding a future container is ONE new
    /// member + ONE appended `ContainerTag` + ONE line in each
    /// `visitContainers` overload — the deload file format (a per-tag
    /// directory + tag-ordered element streams) and all machinery built on
    /// the visitor never restructure.
    ///
    /// Declaration order is load-bearing: `manager` (the arena) is declared
    /// FIRST so the containers (declared after it) destruct before the arena
    /// whose blocks back their storage.
    ///
    /// @invariant `ContainerTag` is append-only: tags are never renumbered,
    ///            reordered, or reused — they are the on-disk identity of each
    ///            container in the deload directory.
    /// @invariant `visitContainers` enumerates members in ascending tag
    ///            order; the deload directory and payload follow that order.
    /// @see `lbdeload::dumpLbMemory` / `lbdeload::loadLbMemory`
    ///      (`memory_infra/lb_deload.hpp`), `ArenaVector`, `LbArena`.
    struct LbMemory {
        /// @brief Append-only on-disk identity of each statified container.
        ///
        /// @details
        /// A `ColdStringTable` contributes TWO tags — its lengths column and
        /// its content bytes (D-153); the table itself
        /// is not visited.
        enum class ContainerTag : uint32_t {
            IntEncodedStatements = 0,
            IntLocalEncodedStatements = 1,
            IntLocalEncodedStatementsDelta = 2,
            IntExternalStatements = 3,
            TemplateStringLengths = 4,
            TemplateStringBytes = 5,
            ValueStringLengths = 6,
            ValueStringBytes = 7,
            OriginStringLengths = 8,
            OriginStringBytes = 9,
            RuleStringLengths = 10,
            RuleStringBytes = 11,
            LbStateStringLengths = 12,
            LbStateStringBytes = 13,
            NameStringLengths = 14,
            NameStringBytes = 15,
            SubStringLengths = 16,
            SubStringBytes = 17,
            ValidityNodes = 18,
            IntValidityNamesToFilter = 19,
            IntAxedVariables = 20,
            CanBeSentIds = 21,
            CanBeSentMarkerIds = 22,
            PendingWipeScopes = 23,
            OrDisjunctCountKeys = 24,
            OrDisjunctCountValues = 25,
            IntegrationStartIntMapKeys = 26,
            IntegrationStartIntMapValues = 27,
            IntLocalEncodedStatementsSet = 28,
            IntWeakVariables = 29,
            IntegrationPrepared = 30,
            IntegrationPreparedMarker = 31,
            ExpandedImplications = 32,
            IntKnownStatementsKeys = 33,
            IntKnownStatementsValues = 34,
            // Batch 2: set-valued int maps on the cold-map SET form
            // (ColdSetMap = HashMap<.., SetValueStore<int>>). Each is THREE
            // tags (key, run-start, value), the key tag first so reload loads
            // keys + rebuilds the index before the parallel columns.
            //
            // Tags 35-37 are RETIRED: they named intToBeProved, which now lives
            // in the persistent (never-deloaded) pool as a direct Memory member
            // (Memory::intToBeProved) so the deactivation survey reads it while
            // the main arena is deloaded. Append-only tags are never reused or
            // renumbered, so 35-37 stay reserved; the surviving Batch-2 set maps
            // (intStatementLevelsMap 38-40, orBookkeeping 41-43) are
            // dischargeable side tables (outside survivesDischarge).
            IntToBeProvedKeys = 35,        // retired (see note above)
            IntToBeProvedRunStarts = 36,   // retired
            IntToBeProvedValues = 37,      // retired
            IntStatementLevelsMapKeys = 38,
            IntStatementLevelsMapRunStarts = 39,
            IntStatementLevelsMapValues = 40,
            // orBookkeeping (OR convergence tracking): packed
            // (expression id, parent-scoped cohort id) key64 -> a set of
            // disjunct ids kept in DECODED order (a per-call DecodedIdLess
            // comparator). Dischargeable, reset at destroyGrid like
            // orDisjunctCount (the lbStateInterner ids it holds persist, but
            // the map itself is rebuilt per grid).
            OrBookkeepingKeys = 41,
            OrBookkeepingRunStarts = 42,
            OrBookkeepingValues = 43,
            // eqClassSttmntIndexMapMap (the cross-pair equality2 gate cache),
            // the nested map flattened to a byte-key single-value map
            // (bytes(validityId ++ memberIds) -> waterline). A byte key
            // contributes a lengths + a bytes tag; the value store its value
            // tag. Dischargeable, reset at destroyGrid.
            EqClassSttmntIndexMapMapLengths = 44,
            EqClassSttmntIndexMapMapBytes = 45,
            EqClassSttmntIndexMapMapValues = 46,
            // Batch 3: equivalenceClassesMap (validity id -> list of
            // EquivalenceClass records) on the cold BLOB map. Each class is one
            // canonical byte blob (serializeEquivalenceClass); the value is the
            // two-level CSR's three columns. FOUR tags (POD key + run-starts +
            // blob-starts + blob-pool), the key tag first so reload rebuilds the
            // index before the parallel columns. Dischargeable, reset at
            // destroyGrid like its companion eqClassSttmntIndexMapMap.
            EquivalenceClassesKeys = 47,
            EquivalenceClassesRunStarts = 48,
            EquivalenceClassesBlobStarts = 49,
            EquivalenceClassesBlobPool = 50,
            // Tags 51..450 belong to the four HashMemory instances (overall
            // 51.., local 151.., localHashMemoryDelta 251.., workingMemory
            // 351.., 100-tag blocks). They are now LbMemory members
            // (D-147); visitContainers enumerates each at its
            // base via HashMemory::visitContainers (bases in hash_memory.hpp), so
            // the deload reaches them through the one enumeration -- no extra
            // columns. A new LbMemory container must NOT take a tag in that
            // range; the next free LbMemory tag (after Batch 5 below) is 455.
            // Batch 5: exprOriginMap (packed (expressionId, validityId) int64 ->
            // the key's run of IdOrigin history lines, each line one canonical
            // blob via Codec<IdOrigin>) on the cold BLOB map. FOUR tags (POD key +
            // run-starts + blob-starts + blob-pool), placed AFTER the HashMemory
            // band so visitContainers stays ascending. SURVIVES discharge (the
            // chapter export reads origin history on discharged LBs) and survives
            // wipeSubtree (I-44); reset only at CE teardown / destroyGrid in
            // lockstep with originInterner. I-121.
            ExprOriginKeys = 451,
            ExprOriginRunStarts = 452,
            ExprOriginBlobStarts = 453,
            ExprOriginBlobPool = 454,
            // Tags 455..554 are RESERVED for the two internal-mail ColdMail
            // instances (sameInternalMail base 455, nextInternalMail base 505;
            // 10 facets each, in a 50-tag block; bases in cold_mail.hpp). They
            // are spliced via ColdMail::visitContainers(base) like the HashMemory
            // bands, so no enumerators live in that range. I-102.
            // Tags 555..654 are an UNUSED gap (were reserved for routing mail,
            // which now rides the never-deloaded mail pool instead of deload —
            // RoutingColdMail owns its own arena; routing_cold_mail.hpp). Never
            // reuse (append-only).
            // Tags 655..704 are RESERVED for changedClassesThisStep (base 655, 3
            // paged facets; base in changed_classes_buffer.hpp), spliced the same
            // at-a-base way.
            // Tags 705..754 are RESERVED for eqClassNameCaches (base 705, 5 paged
            // facets; base in eq_class_name_caches.hpp), spliced the same
            // at-a-base way.
            // Tags 755..804 are RESERVED for the deloadable mailOut aggregate
            // (base 755, 10 facets: private string table + statements +
            // origins + statement flags; base in deloadable_mail_out.hpp).
            // Sequenced or-disintegration: the pending-branch queue. Per
            // cohort (the lbStateInterner-packed (parent, orSignature) id):
            // orPendingBranches holds the UNRELEASED disjunct payload-body
            // ids in decoded-lex order (per-call DecodedIdLess, the
            // orBookkeeping discipline — release order is re-derived by the
            // ranking at each release, never stored); orPendingLevels holds
            // the cohort's seed level run (ascending ints, natural order).
            // Both dischargeable side tables like orBookkeeping.
            OrPendingBranchesKeys = 805,
            OrPendingBranchesRunStarts = 806,
            OrPendingBranchesValues = 807,
            OrPendingLevelsKeys = 808,
            OrPendingLevelsRunStarts = 809,
            OrPendingLevelsValues = 810,
            // Flag-5 relay staging (D-284): per compact
            // (NameMap expression id, main-scope only) awaiting a flag-5 mail
            // deposit — pendingRelayIter holds the sender disintegration's
            // witness-generation stamp (min-merged on restage);
            // pendingRelayLevels holds the staging deposit's ascending level
            // run (first-seen kept). Entries erase when fillMailOut ships the
            // compact. Dischargeable side tables like orPending*. Next free
            // tag: 816.
            PendingRelayIterKeys = 811,
            PendingRelayIterValues = 812,
            PendingRelayLevelsKeys = 813,
            PendingRelayLevelsRunStarts = 814,
            PendingRelayLevelsValues = 815,
            // compactExpansions — the carrier index (tags 816..819): the
            // packed statement key of every carrier that was expanded into
            // hash-memory rules -> one CompactExpansionRec per expanded
            // implication (its lbStateInterner text + scope ids, the install
            // kind). Read by the compact canonicalization to remove exactly
            // the rules a non-canonical compact installed; swept by
            // wipeSubtree by carrier validity; dischargeable like
            // expandedImplications (the dump's source, which stays).
            CompactExpansionsKeys = 816,
            CompactExpansionsRunStarts = 817,
            CompactExpansionsBlobStarts = 818,
            CompactExpansionsBlobPool = 819,
            // expansionCarrierCount (tags 820..821): packed (implTextId, scopeId)
            // of an expanded implication -> the number of carriers in
            // compactExpansions that expanded into it. Two different compacts
            // can expand into one rule text; the rule leaves hash memory only
            // with its LAST carrier. Dischargeable like the carrier index.
            ExpansionCarrierCountKeys = 820,
            ExpansionCarrierCountValues = 821,
            // Append new tags here - never renumber, reorder, or reuse.
            // Tags 4..18 survive discharge (survivesDischarge): the cold STRING
            // tags (4..17) plus the NameMap validity-node forest (18). Tag 28
            // (IntLocalEncodedStatementsSet) ALSO survives — the post-prove
            // recursion-node gate probes it on discharged LBs. The other Batch-1
            // tags (19-27, 29-31) are dischargeable side tables (dead weight on a
            // discharged LB). survivesDischarge encodes this exactly.
        };

        /// @brief Bind the aggregate to a block source.
        ///
        /// @details
        /// Lazy like its arena: constructing an `LbMemory` consumes no pool
        /// blocks (transient `Memory` objects must stay free).
        ///
        /// @param global The block source (the process-wide instance in
        ///               production; a private instance in unit tests).
        explicit LbMemory(GlobalMemoryManager* global)
            : manager(global),
              intEncodedStatements(&manager, &dirty),
              intLocalEncodedStatements(&manager, &dirty),
              intLocalEncodedStatementsDelta(&manager, &dirty),
              intExternalStatements(&manager, &dirty),
              templateStrings(&manager, &dirty),
              templateStringsLengths(&templateStrings),
              templateStringsBytes(&templateStrings),
              valueStrings(&manager, &dirty),
              valueStringsLengths(&valueStrings),
              valueStringsBytes(&valueStrings),
              originStrings(&manager, &dirty),
              originStringsLengths(&originStrings),
              originStringsBytes(&originStrings),
              ruleStrings(&manager, &dirty),
              ruleStringsLengths(&ruleStrings),
              ruleStringsBytes(&ruleStrings),
              lbStateStrings(&manager, &dirty),
              lbStateStringsLengths(&lbStateStrings),
              lbStateStringsBytes(&lbStateStrings),
              nameStrings(&manager, &dirty),
              nameStringsLengths(&nameStrings),
              nameStringsBytes(&nameStrings),
              subStrings(&manager, &dirty),
              subStringsLengths(&subStrings),
              subStringsBytes(&subStrings),
              validityNodes(&manager, &dirty),
              intValidityNamesToFilter(&manager, &dirty),
              intValidityNamesToFilterKeys(&intValidityNamesToFilter),
              intAxedVariables(&manager, &dirty),
              intAxedVariablesKeys(&intAxedVariables),
              canBeSentIds(&manager, &dirty),
              canBeSentIdsKeys(&canBeSentIds),
              canBeSentMarkerIds(&manager, &dirty),
              canBeSentMarkerIdsKeys(&canBeSentMarkerIds),
              pendingWipeScopes(&manager, &dirty),
              pendingWipeScopesKeys(&pendingWipeScopes),
              orDisjunctCount(&manager, &dirty),
              orDisjunctCountKeys(&orDisjunctCount),
              orDisjunctCountValues(&orDisjunctCount),
              integrationStartIntMap(&manager, &dirty),
              integrationStartIntMapKeys(&integrationStartIntMap),
              integrationStartIntMapValues(&integrationStartIntMap),
              intLocalEncodedStatementsSet(&manager, &dirty),
              intLocalEncodedStatementsSetKeys(&intLocalEncodedStatementsSet),
              intWeakVariables(&manager, &dirty),
              intWeakVariablesKeys(&intWeakVariables),
              integrationPrepared(&manager, &dirty),
              integrationPreparedKeys(&integrationPrepared),
              integrationPreparedMarker(&manager, &dirty),
              integrationPreparedMarkerKeys(&integrationPreparedMarker),
              expandedImplications(&manager, &dirty),
              expandedImplicationsKeys(&expandedImplications.inner()),
              intKnownStatements(&manager, &dirty),
              intKnownStatementsKeys(&intKnownStatements.inner()),
              intKnownStatementsValues(&intKnownStatements.inner()),
              intStatementLevelsMap(&manager, &dirty),
              intStatementLevelsMapKeys(&intStatementLevelsMap.inner()),
              intStatementLevelsMapRunStarts(&intStatementLevelsMap.inner()),
              intStatementLevelsMapValues(&intStatementLevelsMap.inner()),
              orBookkeeping(&manager, &dirty),
              orBookkeepingKeys(&orBookkeeping.inner()),
              orBookkeepingRunStarts(&orBookkeeping.inner()),
              orBookkeepingValues(&orBookkeeping.inner()),
              eqClassSttmntIndexMapMap(&manager, &dirty),
              eqClassSttmntIndexMapMapLengths(&eqClassSttmntIndexMapMap.inner()),
              eqClassSttmntIndexMapMapBytes(&eqClassSttmntIndexMapMap.inner()),
              eqClassSttmntIndexMapMapValues(&eqClassSttmntIndexMapMap.inner()),
              equivalenceClassesMap(&manager, &dirty),
              equivalenceClassesMapKeys(&equivalenceClassesMap.inner()),
              equivalenceClassesMapRunStarts(&equivalenceClassesMap.inner()),
              equivalenceClassesMapBlobStarts(&equivalenceClassesMap.inner()),
              equivalenceClassesMapBlobPool(&equivalenceClassesMap.inner()),
              exprOriginMap(&manager, &dirty),
              exprOriginMapKeys(&exprOriginMap.inner()),
              exprOriginMapRunStarts(&exprOriginMap.inner()),
              exprOriginMapBlobStarts(&exprOriginMap.inner()),
              exprOriginMapBlobPool(&exprOriginMap.inner()),
              overallHashMemory(&manager, &dirty),
              localHashMemory(&manager, &dirty),
              localHashMemoryDelta(&manager, &dirty),
              workingMemory(&manager, &dirty),
              sameInternalMail(&manager, &dirty),
              nextInternalMail(&manager, &dirty),
              changedClassesThisStep(&manager, &dirty),
              eqClassNameCaches(&manager, &dirty),
              mailOut(&manager, &dirty),
              orPendingBranches(&manager, &dirty),
              orPendingBranchesKeys(&orPendingBranches.inner()),
              orPendingBranchesRunStarts(&orPendingBranches.inner()),
              orPendingBranchesValues(&orPendingBranches.inner()),
              orPendingLevels(&manager, &dirty),
              orPendingLevelsKeys(&orPendingLevels.inner()),
              orPendingLevelsRunStarts(&orPendingLevels.inner()),
              orPendingLevelsValues(&orPendingLevels.inner()),
              pendingRelayIter(&manager, &dirty),
              pendingRelayIterKeys(&pendingRelayIter.inner()),
              pendingRelayIterValues(&pendingRelayIter.inner()),
              pendingRelayLevels(&manager, &dirty),
              pendingRelayLevelsKeys(&pendingRelayLevels.inner()),
              pendingRelayLevelsRunStarts(&pendingRelayLevels.inner()),
              pendingRelayLevelsValues(&pendingRelayLevels.inner()),
              compactExpansions(&manager, &dirty),
              compactExpansionsKeys(&compactExpansions.inner()),
              compactExpansionsRunStarts(&compactExpansions.inner()),
              compactExpansionsBlobStarts(&compactExpansions.inner()),
              compactExpansionsBlobPool(&compactExpansions.inner()),
              expansionCarrierCount(&manager, &dirty),
              expansionCarrierCountKeys(&expansionCarrierCount),
              expansionCarrierCountValues(&expansionCarrierCount) {}

        LbMemory(const LbMemory&) = delete;
        LbMemory& operator=(const LbMemory&) = delete;

        LbArena manager;
        // Content-change state shared by every statified container (set by
        // all their mutators): Clean = the on-disk file set still equals the
        // in-memory content (deload releases blocks without writing);
        // AppendedOnly = only push_backs since the last dump (deload may write
        // a small tail file); Restructured = full canonical rewrite required.
        // Reset to Clean after a dump and after a reload (both points where
        // RAM == disk by construction).
        DirtyState dirty = DirtyState::Clean;
        PagedVector<IntEncodedExpr> intEncodedStatements;
        PagedVector<IntEncodedExpr> intLocalEncodedStatements;
        PagedVector<IntEncodedExpr> intLocalEncodedStatementsDelta;
        PagedVector<IntEncodedExpr> intExternalStatements;
        // The interner cold string tables (strings campaign); each is
        // serialized through its two facets, never directly. The lb-state
        // table additionally survives destroyGrid (its façade never calls
        // resetToFresh — expandedImplications outlives the grid).
        ColdStringTable templateStrings;
        ColdStringTable::LengthsView templateStringsLengths;
        ColdStringTable::BytesView templateStringsBytes;
        ColdStringTable valueStrings;
        ColdStringTable::LengthsView valueStringsLengths;
        ColdStringTable::BytesView valueStringsBytes;
        ColdStringTable originStrings;
        ColdStringTable::LengthsView originStringsLengths;
        ColdStringTable::BytesView originStringsBytes;
        ColdStringTable ruleStrings;
        ColdStringTable::LengthsView ruleStringsLengths;
        ColdStringTable::BytesView ruleStringsBytes;
        ColdStringTable lbStateStrings;
        ColdStringTable::LengthsView lbStateStringsLengths;
        ColdStringTable::BytesView lbStateStringsBytes;
        ColdStringTable nameStrings;
        ColdStringTable::LengthsView nameStringsLengths;
        ColdStringTable::BytesView nameStringsBytes;
        ColdStringTable subStrings;
        ColdStringTable::LengthsView subStringsLengths;
        ColdStringTable::BytesView subStringsBytes;
        // NameMap validity metadata: the per-LB scope forest as a flat
        // parent-pointer array (was the heap std::vector<std::vector<int16_t>>
        // pair, then a jagged CSR). One ValidityNode per validity id (id 0 =
        // unused sentinel, id 1 = "main"); ancestor / payload-stack queries walk
        // the parentId chain. Lazily seeded on the LB's first encode (zero pool
        // blocks for a transient Memory). Flat + trivially copyable, so it
        // deloads as ONE tag, visited directly like the statement vectors.
        PagedVector<ValidityNode> validityNodes;
        // Batch 1 flat NameId id sets (NameMap ids), migrated off the heap onto
        // the cold-map family (D-170). Each is one
        // ColdHashSet<PodKeyStore<NameId>> plus its KeysView deload facet (one
        // tag). All DISCHARGEABLE (tags 19+, outside survivesDischarge) — dead
        // weight on a discharged LB. Exposed on Memory via reference alias like
        // the statement vectors. pendingWipeScopes is drained (and reset) every
        // burst; the others reset at destroyGrid.
        ColdHashSet<PodKeyStore<NameId>> intValidityNamesToFilter;
        ColdHashSet<PodKeyStore<NameId>>::KeysView intValidityNamesToFilterKeys;
        ColdHashSet<PodKeyStore<NameId>> intAxedVariables;
        ColdHashSet<PodKeyStore<NameId>>::KeysView intAxedVariablesKeys;
        ColdHashSet<PodKeyStore<NameId>> canBeSentIds;
        ColdHashSet<PodKeyStore<NameId>>::KeysView canBeSentIdsKeys;
        ColdHashSet<PodKeyStore<NameId>> canBeSentMarkerIds;
        ColdHashSet<PodKeyStore<NameId>>::KeysView canBeSentMarkerIdsKeys;
        ColdHashSet<PodKeyStore<NameId>> pendingWipeScopes;
        ColdHashSet<PodKeyStore<NameId>>::KeysView pendingWipeScopesKeys;
        // Batch 1 write-once int->int maps (D-170):
        // ColdHashMap with a key facet + a value facet (two tags each). Both
        // dischargeable. orDisjunctCount: parent-scoped OR cohort id
        // (lbStateInterner) -> disjunct count; integrationStartIntMap: template
        // id -> startInt snapshot.
        ColdHashMap<PodKeyStore<int32_t>, int> orDisjunctCount;
        ColdHashMap<PodKeyStore<int32_t>, int>::KeysView orDisjunctCountKeys;
        ColdHashMap<PodKeyStore<int32_t>, int>::ValuesView orDisjunctCountValues;
        ColdHashMap<PodKeyStore<NameId>, int> integrationStartIntMap;
        ColdHashMap<PodKeyStore<NameId>, int>::KeysView integrationStartIntMapKeys;
        ColdHashMap<PodKeyStore<NameId>, int>::ValuesView integrationStartIntMapValues;
        // Batch 1 erase-needing packed int64 sets (D-170):
        // each a ColdHashSet + its KeysView facet (one tag). erased per-key in
        // wipeSubtree / eradicate. Keyed by packStatementKey / mintTemplateKey,
        // both of which pack two NameId halves into an int64. intLocalEncoded-
        // StatementsSet SURVIVES discharge (the recursion-node gate reads it
        // post-prove); the other three are dischargeable.
        ColdHashSet<PodKeyStore<int64_t>> intLocalEncodedStatementsSet;
        ColdHashSet<PodKeyStore<int64_t>>::KeysView intLocalEncodedStatementsSetKeys;
        ColdHashSet<PodKeyStore<int64_t>> intWeakVariables;
        ColdHashSet<PodKeyStore<int64_t>>::KeysView intWeakVariablesKeys;
        ColdHashSet<PodKeyStore<int64_t>> integrationPrepared;
        ColdHashSet<PodKeyStore<int64_t>>::KeysView integrationPreparedKeys;
        ColdHashSet<PodKeyStore<int64_t>> integrationPreparedMarker;
        ColdHashSet<PodKeyStore<int64_t>>::KeysView integrationPreparedMarkerKeys;
        // expandedImplications: packed (implTextId, implScopeId) lbStateInterner
        // pairs. DISCHARGEABLE (tag 32, outside survivesDischarge) but NOT reset
        // at destroyGrid — it outlives the grid like its lbStateInterner keys.
        TypedColdSet<LbStatePairKey> expandedImplications;
        TypedColdSet<LbStatePairKey>::KeysView expandedImplicationsKeys;
        // intKnownStatements: packed (originalId, validityId) -> StatementFlags
        // (row presence IS the known membership, I-85; the value carries the
        // local/fullyDisintegrated payload). A ColdHashMap: key facet
        // + value facet (two tags). Dischargeable — the equality-node gate
        // probes the captured dischargedRegistryKeys for discharged LBs instead.
        TypedColdMap<StatementKey, StatementFlags> intKnownStatements;
        TypedColdMap<StatementKey, StatementFlags>::KeysView intKnownStatementsKeys;
        TypedColdMap<StatementKey, StatementFlags>::ValuesView intKnownStatementsValues;
        // Batch 2: the set-valued int maps (packed (origId, validityId) -> a
        // SORTED-UNIQUE int set). ColdSetMap = HashMap<.., SetValueStore<int>>;
        // three facets each (key, run-start, value). Dischargeable.
        // (intToBeProved was here at tags 35-37; it moved to the persistent pool
        // as a direct Memory member — see the ContainerTag retirement note.)
        TypedColdSetMap<StatementKey, int> intStatementLevelsMap;
        TypedColdSetMap<StatementKey, int>::KeysView intStatementLevelsMapKeys;
        TypedColdSetMap<StatementKey, int>::RunStartsView intStatementLevelsMapRunStarts;
        TypedColdSetMap<StatementKey, int>::RunValuesView intStatementLevelsMapValues;
        // orBookkeeping: key64 (packed expression id + parent-scoped cohort id)
        // -> a set of branch-disjunct ids kept in DECODED order (the run is
        // built by insertSorted with a per-call DecodedIdLess; never
        // coldIntSetAt for reads, which would re-sort by raw int).
        TypedColdSetMap<LbStatePairKey, int32_t> orBookkeeping;
        TypedColdSetMap<LbStatePairKey, int32_t>::KeysView orBookkeepingKeys;
        TypedColdSetMap<LbStatePairKey, int32_t>::RunStartsView orBookkeepingRunStarts;
        TypedColdSetMap<LbStatePairKey, int32_t>::RunValuesView orBookkeepingValues;
        // eqClassSttmntIndexMapMap (flattened): byte key (validity ++ members)
        // -> the statement-registry waterline. A byte-key single-value map; the
        // byte store contributes Lengths + Bytes facets, the value store its
        // Values facet (3 tags). Dischargeable.
        TypedColdMap<EqClassKey, int> eqClassSttmntIndexMapMap;
        TypedColdMap<EqClassKey, int>::LengthsView eqClassSttmntIndexMapMapLengths;
        TypedColdMap<EqClassKey, int>::BytesView eqClassSttmntIndexMapMapBytes;
        TypedColdMap<EqClassKey, int>::ValuesView eqClassSttmntIndexMapMapValues;
        // Batch 3: equivalenceClassesMap on the cold BLOB map (the record value
        // store). Key = validity id; value = the validity's class list, each
        // class one canonical byte blob. Four facets: keys, run-starts,
        // blob-starts, blob-pool. Dischargeable; reset at destroyGrid.
        TypedColdBlobMap<NameId, EquivalenceClass> equivalenceClassesMap;
        TypedColdBlobMap<NameId, EquivalenceClass>::KeysView equivalenceClassesMapKeys;
        TypedColdBlobMap<NameId, EquivalenceClass>::RunStartsView equivalenceClassesMapRunStarts;
        TypedColdBlobMap<NameId, EquivalenceClass>::BlobStartsView equivalenceClassesMapBlobStarts;
        TypedColdBlobMap<NameId, EquivalenceClass>::BlobPoolView equivalenceClassesMapBlobPool;
        // Batch 5: exprOriginMap on the cold BLOB map (the record value store).
        // Key = packed (expressionId, validityId) int64; value = the key's run of
        // IdOrigin history lines, each line one canonical blob (Codec<IdOrigin>).
        // Four facets: keys, run-starts, blob-starts, blob-pool. Survives
        // discharge AND wipeSubtree (I-44); reset only at CE teardown / destroyGrid
        // in lockstep with originInterner. I-121.
        TypedColdBlobMap<int64_t, IdOrigin> exprOriginMap;
        TypedColdBlobMap<int64_t, IdOrigin>::KeysView exprOriginMapKeys;
        TypedColdBlobMap<int64_t, IdOrigin>::RunStartsView exprOriginMapRunStarts;
        TypedColdBlobMap<int64_t, IdOrigin>::BlobStartsView exprOriginMapBlobStarts;
        TypedColdBlobMap<int64_t, IdOrigin>::BlobPoolView exprOriginMapBlobPool;
        // The four HashMemory instances (D-147): the per-LB
        // hash-inference engine, folded in as members. Declared AFTER manager so
        // they destruct before the arena their cold containers ride — the natural
        // order that retired Memory's ~Memory releaseAllCold safeguard. Two
        // persistent (overall / local) + two transient (localHashMemoryDelta /
        // workingMemory, emptied per burst by resetToFresh). visitContainers
        // enumerates each at bases 51 / 151 / 251 / 351; all survive discharge.
        HashMemory overallHashMemory;
        HashMemory localHashMemory;
        HashMemory localHashMemoryDelta;
        HashMemory workingMemory;
        // The two internal-mail channels on the cold deloadable path
        // (I-102). Declared after manager so they
        // destruct before the arena their columns ride. Spliced into
        // visitContainers at bases 455 / 505; SURVIVE discharge (the heap never
        // cleared internal mail at discharge -- faithful mirror).
        ColdMail sameInternalMail;
        ColdMail nextInternalMail;
        // Cross-LB mailIn does not live here: it uses RoutingColdMail's dedicated
        // mail-pool arena and returns every block immediately after phase-1
        // absorb. mailOut must cross the phase-3-to-commit seam, so it rides this
        // deloadable arena and the serial barrier claim/reloads pending LBs.
        // The per-step changed-equivalence-class delta on the cold deloadable
        // path. Written only by standardProcessing (single-threaded phase-1/3),
        // emptied every step, so empty at the deload seam -- dischargeable (NOT in
        // survivesDischarge). Spliced into visitContainers at base 655.
        ChangedClassesBuffer changedClassesThisStep;
        // The classifyName / scanSpecialTokens memo on the cold deloadable path.
        // A persistent derived memo (never cleared per-step, single-threaded probe
        // per I-83); SURVIVES discharge (the post-prove readers + cross-grid reuse
        // keep it). Spliced into visitContainers at base 705.
        EqClassNameCaches eqClassNameCaches;
        // Cross-LB outgoing mail on the cold deloadable path, including its own
        // string id space. Spliced into visitContainers at base 755 and kept on
        // discharge until its final committed batch is cleared.
        DeloadableMailOut mailOut;
        // Sequenced or-disintegration (tags 805..810): cohortId -> the
        // unreleased disjunct payload-body ids (decoded-lex runs via a
        // per-call DecodedIdLess, read in run order like orBookkeeping), and
        // cohortId -> the cohort's seed level run (ascending ints, natural
        // order). Both dischargeable, rebuilt per grid like orBookkeeping /
        // orDisjunctCount.
        TypedColdSetMap<int32_t, int32_t> orPendingBranches;
        TypedColdSetMap<int32_t, int32_t>::KeysView orPendingBranchesKeys;
        TypedColdSetMap<int32_t, int32_t>::RunStartsView orPendingBranchesRunStarts;
        TypedColdSetMap<int32_t, int32_t>::RunValuesView orPendingBranchesValues;
        TypedColdSetMap<int32_t, int32_t> orPendingLevels;
        TypedColdSetMap<int32_t, int32_t>::KeysView orPendingLevelsKeys;
        TypedColdSetMap<int32_t, int32_t>::RunStartsView orPendingLevelsRunStarts;
        TypedColdSetMap<int32_t, int32_t>::RunValuesView orPendingLevelsValues;

        // Flag-5 relay staging (D-284): compact NameMap id →
        // witness-generation stamp, plus the staging deposit's level run.
        // Main-scope only, erased when fillMailOut ships the compact.
        TypedColdMap<int32_t, int32_t> pendingRelayIter;
        TypedColdMap<int32_t, int32_t>::KeysView pendingRelayIterKeys;
        TypedColdMap<int32_t, int32_t>::ValuesView pendingRelayIterValues;
        TypedColdSetMap<int32_t, int32_t> pendingRelayLevels;
        TypedColdSetMap<int32_t, int32_t>::KeysView pendingRelayLevelsKeys;
        TypedColdSetMap<int32_t, int32_t>::RunStartsView pendingRelayLevelsRunStarts;
        TypedColdSetMap<int32_t, int32_t>::RunValuesView pendingRelayLevelsValues;
        // compactExpansions (tags 816..819): carrier statement key (packed
        // (originalId, validityId), NameMap ids) -> the run of the expanded
        // implications it installed (CompactExpansionRec: lbStateInterner
        // text + scope ids, install kind). The carrier index the compact
        // canonicalization removes rules through (see ContainerTag).
        TypedColdBlobMap<int64_t, CompactExpansionRec> compactExpansions;
        TypedColdBlobMap<int64_t, CompactExpansionRec>::KeysView compactExpansionsKeys;
        TypedColdBlobMap<int64_t, CompactExpansionRec>::RunStartsView compactExpansionsRunStarts;
        TypedColdBlobMap<int64_t, CompactExpansionRec>::BlobStartsView compactExpansionsBlobStarts;
        TypedColdBlobMap<int64_t, CompactExpansionRec>::BlobPoolView compactExpansionsBlobPool;
        // expansionCarrierCount (tags 820..821): how many carriers expanded into
        // an implication (packed lbStateInterner text + scope ids); a rule leaves
        // only when the count reaches zero (see ContainerTag).
        ColdHashMap<PodKeyStore<int64_t>, int32_t> expansionCarrierCount;
        ColdHashMap<PodKeyStore<int64_t>, int32_t>::KeysView expansionCarrierCountKeys;
        ColdHashMap<PodKeyStore<int64_t>, int32_t>::ValuesView expansionCarrierCountValues;

        /// @brief Enumerate the statified containers in tag order (mutable).
        ///
        /// @details
        /// The single source of container enumeration for the deload
        /// serializer and any future whole-aggregate operation. The visitor
        /// is called as `visit(ContainerTag, container&)` once per member,
        /// ascending tag order.
        ///
        /// @param visit Callable accepting `(ContainerTag, container&)` for
        ///              every statified container (an `ArenaVector<T>` or a
        ///              cold-string facet).
        template <typename Visitor>
        void visitContainers(Visitor&& visit) {
            visit(ContainerTag::IntEncodedStatements, intEncodedStatements);
            visit(ContainerTag::IntLocalEncodedStatements,
                  intLocalEncodedStatements);
            visit(ContainerTag::IntLocalEncodedStatementsDelta,
                  intLocalEncodedStatementsDelta);
            visit(ContainerTag::IntExternalStatements,
                  intExternalStatements);
            visit(ContainerTag::TemplateStringLengths,
                  templateStringsLengths);
            visit(ContainerTag::TemplateStringBytes, templateStringsBytes);
            visit(ContainerTag::ValueStringLengths, valueStringsLengths);
            visit(ContainerTag::ValueStringBytes, valueStringsBytes);
            visit(ContainerTag::OriginStringLengths, originStringsLengths);
            visit(ContainerTag::OriginStringBytes, originStringsBytes);
            visit(ContainerTag::RuleStringLengths, ruleStringsLengths);
            visit(ContainerTag::RuleStringBytes, ruleStringsBytes);
            visit(ContainerTag::LbStateStringLengths, lbStateStringsLengths);
            visit(ContainerTag::LbStateStringBytes, lbStateStringsBytes);
            visit(ContainerTag::NameStringLengths, nameStringsLengths);
            visit(ContainerTag::NameStringBytes, nameStringsBytes);
            visit(ContainerTag::SubStringLengths, subStringsLengths);
            visit(ContainerTag::SubStringBytes, subStringsBytes);
            visit(ContainerTag::ValidityNodes, validityNodes);
            visit(ContainerTag::IntValidityNamesToFilter,
                  intValidityNamesToFilterKeys);
            visit(ContainerTag::IntAxedVariables, intAxedVariablesKeys);
            visit(ContainerTag::CanBeSentIds, canBeSentIdsKeys);
            visit(ContainerTag::CanBeSentMarkerIds, canBeSentMarkerIdsKeys);
            visit(ContainerTag::PendingWipeScopes, pendingWipeScopesKeys);
            visit(ContainerTag::OrDisjunctCountKeys, orDisjunctCountKeys);
            visit(ContainerTag::OrDisjunctCountValues, orDisjunctCountValues);
            visit(ContainerTag::IntegrationStartIntMapKeys,
                  integrationStartIntMapKeys);
            visit(ContainerTag::IntegrationStartIntMapValues,
                  integrationStartIntMapValues);
            visit(ContainerTag::IntLocalEncodedStatementsSet,
                  intLocalEncodedStatementsSetKeys);
            visit(ContainerTag::IntWeakVariables, intWeakVariablesKeys);
            visit(ContainerTag::IntegrationPrepared, integrationPreparedKeys);
            visit(ContainerTag::IntegrationPreparedMarker,
                  integrationPreparedMarkerKeys);
            visit(ContainerTag::ExpandedImplications, expandedImplicationsKeys);
            visit(ContainerTag::IntKnownStatementsKeys, intKnownStatementsKeys);
            visit(ContainerTag::IntKnownStatementsValues,
                  intKnownStatementsValues);
            visit(ContainerTag::IntStatementLevelsMapKeys,
                  intStatementLevelsMapKeys);
            visit(ContainerTag::IntStatementLevelsMapRunStarts,
                  intStatementLevelsMapRunStarts);
            visit(ContainerTag::IntStatementLevelsMapValues,
                  intStatementLevelsMapValues);
            visit(ContainerTag::OrBookkeepingKeys, orBookkeepingKeys);
            visit(ContainerTag::OrBookkeepingRunStarts, orBookkeepingRunStarts);
            visit(ContainerTag::OrBookkeepingValues, orBookkeepingValues);
            visit(ContainerTag::EqClassSttmntIndexMapMapLengths,
                  eqClassSttmntIndexMapMapLengths);
            visit(ContainerTag::EqClassSttmntIndexMapMapBytes,
                  eqClassSttmntIndexMapMapBytes);
            visit(ContainerTag::EqClassSttmntIndexMapMapValues,
                  eqClassSttmntIndexMapMapValues);
            visit(ContainerTag::EquivalenceClassesKeys,
                  equivalenceClassesMapKeys);
            visit(ContainerTag::EquivalenceClassesRunStarts,
                  equivalenceClassesMapRunStarts);
            visit(ContainerTag::EquivalenceClassesBlobStarts,
                  equivalenceClassesMapBlobStarts);
            visit(ContainerTag::EquivalenceClassesBlobPool,
                  equivalenceClassesMapBlobPool);
            // The four HashMemory instances (D-147): enumerate
            // each at its reserved 100-tag base (51 / 151 / 251 / 351), bridging
            // HashMemory's uint32 base+offset tag into a ContainerTag so the
            // deload directory records 51..450 exactly as the former extra-column
            // path did (byte-identical on disk).
            auto visitHashMemory = [&visit](uint32_t t, auto& facet) {
                visit(static_cast<ContainerTag>(t), facet);
            };
            overallHashMemory.visitContainers(kOverallHashMemoryDeloadBase,
                                              visitHashMemory);
            localHashMemory.visitContainers(kLocalHashMemoryDeloadBase,
                                            visitHashMemory);
            localHashMemoryDelta.visitContainers(kLocalHashMemoryDeltaDeloadBase,
                                                 visitHashMemory);
            workingMemory.visitContainers(kWorkingMemoryDeloadBase,
                                          visitHashMemory);
            // Batch 5: exprOriginMap (tags 451..454) — after the HashMemory band
            // (51..450) so the enumeration stays ascending.
            visit(ContainerTag::ExprOriginKeys, exprOriginMapKeys);
            visit(ContainerTag::ExprOriginRunStarts, exprOriginMapRunStarts);
            visit(ContainerTag::ExprOriginBlobStarts, exprOriginMapBlobStarts);
            visit(ContainerTag::ExprOriginBlobPool, exprOriginMapBlobPool);
            // Internal-mail channels (tags 455..464 / 505..514) — after the
            // exprOrigin band so the enumeration stays ascending. Spliced via
            // ColdMail::visitContainers(base), the HashMemory-at-a-base pattern.
            sameInternalMail.visitContainers(kSameInternalMailDeloadBase,
                                             visitHashMemory);
            nextInternalMail.visitContainers(kNextInternalMailDeloadBase,
                                             visitHashMemory);
            // Changed-classes delta (tags 655..657) — dischargeable.
            changedClassesThisStep.visitContainers(kChangedClassesDeloadBase,
                                                   visitHashMemory);
            // EqClassNameCaches memo (tags 705..709) — survives discharge.
            eqClassNameCaches.visitContainers(kEqClassNameCachesDeloadBase,
                                              visitHashMemory);
            // Outgoing routing mail (tags 755..762) — private string table plus
            // statements and origins, all deloaded as one unit.
            mailOut.visitContainers(kMailOutDeloadBase, visitHashMemory);
            // Sequenced or-disintegration pending queue (tags 805..810) —
            // dischargeable.
            visit(ContainerTag::OrPendingBranchesKeys, orPendingBranchesKeys);
            visit(ContainerTag::OrPendingBranchesRunStarts,
                  orPendingBranchesRunStarts);
            visit(ContainerTag::OrPendingBranchesValues,
                  orPendingBranchesValues);
            visit(ContainerTag::OrPendingLevelsKeys, orPendingLevelsKeys);
            visit(ContainerTag::OrPendingLevelsRunStarts,
                  orPendingLevelsRunStarts);
            visit(ContainerTag::OrPendingLevelsValues, orPendingLevelsValues);
            visit(ContainerTag::PendingRelayIterKeys, pendingRelayIterKeys);
            visit(ContainerTag::PendingRelayIterValues, pendingRelayIterValues);
            visit(ContainerTag::PendingRelayLevelsKeys, pendingRelayLevelsKeys);
            visit(ContainerTag::PendingRelayLevelsRunStarts,
                  pendingRelayLevelsRunStarts);
            visit(ContainerTag::PendingRelayLevelsValues,
                  pendingRelayLevelsValues);
            visit(ContainerTag::CompactExpansionsKeys, compactExpansionsKeys);
            visit(ContainerTag::CompactExpansionsRunStarts, compactExpansionsRunStarts);
            visit(ContainerTag::CompactExpansionsBlobStarts, compactExpansionsBlobStarts);
            visit(ContainerTag::CompactExpansionsBlobPool, compactExpansionsBlobPool);
            visit(ContainerTag::ExpansionCarrierCountKeys, expansionCarrierCountKeys);
            visit(ContainerTag::ExpansionCarrierCountValues, expansionCarrierCountValues);
        }

        /// @brief Enumerate the statified containers in tag order
        ///        (read-only).
        ///
        /// @param visit Callable accepting `(ContainerTag, const container&)`
        ///              for every statified container.
        template <typename Visitor>
        void visitContainers(Visitor&& visit) const {
            visit(ContainerTag::IntEncodedStatements, intEncodedStatements);
            visit(ContainerTag::IntLocalEncodedStatements,
                  intLocalEncodedStatements);
            visit(ContainerTag::IntLocalEncodedStatementsDelta,
                  intLocalEncodedStatementsDelta);
            visit(ContainerTag::IntExternalStatements,
                  intExternalStatements);
            visit(ContainerTag::TemplateStringLengths,
                  templateStringsLengths);
            visit(ContainerTag::TemplateStringBytes, templateStringsBytes);
            visit(ContainerTag::ValueStringLengths, valueStringsLengths);
            visit(ContainerTag::ValueStringBytes, valueStringsBytes);
            visit(ContainerTag::OriginStringLengths, originStringsLengths);
            visit(ContainerTag::OriginStringBytes, originStringsBytes);
            visit(ContainerTag::RuleStringLengths, ruleStringsLengths);
            visit(ContainerTag::RuleStringBytes, ruleStringsBytes);
            visit(ContainerTag::LbStateStringLengths, lbStateStringsLengths);
            visit(ContainerTag::LbStateStringBytes, lbStateStringsBytes);
            visit(ContainerTag::NameStringLengths, nameStringsLengths);
            visit(ContainerTag::NameStringBytes, nameStringsBytes);
            visit(ContainerTag::SubStringLengths, subStringsLengths);
            visit(ContainerTag::SubStringBytes, subStringsBytes);
            visit(ContainerTag::ValidityNodes, validityNodes);
            visit(ContainerTag::IntValidityNamesToFilter,
                  intValidityNamesToFilterKeys);
            visit(ContainerTag::IntAxedVariables, intAxedVariablesKeys);
            visit(ContainerTag::CanBeSentIds, canBeSentIdsKeys);
            visit(ContainerTag::CanBeSentMarkerIds, canBeSentMarkerIdsKeys);
            visit(ContainerTag::PendingWipeScopes, pendingWipeScopesKeys);
            visit(ContainerTag::OrDisjunctCountKeys, orDisjunctCountKeys);
            visit(ContainerTag::OrDisjunctCountValues, orDisjunctCountValues);
            visit(ContainerTag::IntegrationStartIntMapKeys,
                  integrationStartIntMapKeys);
            visit(ContainerTag::IntegrationStartIntMapValues,
                  integrationStartIntMapValues);
            visit(ContainerTag::IntLocalEncodedStatementsSet,
                  intLocalEncodedStatementsSetKeys);
            visit(ContainerTag::IntWeakVariables, intWeakVariablesKeys);
            visit(ContainerTag::IntegrationPrepared, integrationPreparedKeys);
            visit(ContainerTag::IntegrationPreparedMarker,
                  integrationPreparedMarkerKeys);
            visit(ContainerTag::ExpandedImplications, expandedImplicationsKeys);
            visit(ContainerTag::IntKnownStatementsKeys, intKnownStatementsKeys);
            visit(ContainerTag::IntKnownStatementsValues,
                  intKnownStatementsValues);
            visit(ContainerTag::IntStatementLevelsMapKeys,
                  intStatementLevelsMapKeys);
            visit(ContainerTag::IntStatementLevelsMapRunStarts,
                  intStatementLevelsMapRunStarts);
            visit(ContainerTag::IntStatementLevelsMapValues,
                  intStatementLevelsMapValues);
            visit(ContainerTag::OrBookkeepingKeys, orBookkeepingKeys);
            visit(ContainerTag::OrBookkeepingRunStarts, orBookkeepingRunStarts);
            visit(ContainerTag::OrBookkeepingValues, orBookkeepingValues);
            visit(ContainerTag::EqClassSttmntIndexMapMapLengths,
                  eqClassSttmntIndexMapMapLengths);
            visit(ContainerTag::EqClassSttmntIndexMapMapBytes,
                  eqClassSttmntIndexMapMapBytes);
            visit(ContainerTag::EqClassSttmntIndexMapMapValues,
                  eqClassSttmntIndexMapMapValues);
            visit(ContainerTag::EquivalenceClassesKeys,
                  equivalenceClassesMapKeys);
            visit(ContainerTag::EquivalenceClassesRunStarts,
                  equivalenceClassesMapRunStarts);
            visit(ContainerTag::EquivalenceClassesBlobStarts,
                  equivalenceClassesMapBlobStarts);
            visit(ContainerTag::EquivalenceClassesBlobPool,
                  equivalenceClassesMapBlobPool);
            // The four HashMemory instances (D-147): enumerate
            // each at its reserved 100-tag base (51 / 151 / 251 / 351), bridging
            // HashMemory's uint32 base+offset tag into a ContainerTag so the
            // deload directory records 51..450 exactly as the former extra-column
            // path did (byte-identical on disk).
            auto visitHashMemory = [&visit](uint32_t t, auto& facet) {
                visit(static_cast<ContainerTag>(t), facet);
            };
            overallHashMemory.visitContainers(kOverallHashMemoryDeloadBase,
                                              visitHashMemory);
            localHashMemory.visitContainers(kLocalHashMemoryDeloadBase,
                                            visitHashMemory);
            localHashMemoryDelta.visitContainers(kLocalHashMemoryDeltaDeloadBase,
                                                 visitHashMemory);
            workingMemory.visitContainers(kWorkingMemoryDeloadBase,
                                          visitHashMemory);
            // Batch 5: exprOriginMap (tags 451..454) — after the HashMemory band
            // (51..450) so the enumeration stays ascending.
            visit(ContainerTag::ExprOriginKeys, exprOriginMapKeys);
            visit(ContainerTag::ExprOriginRunStarts, exprOriginMapRunStarts);
            visit(ContainerTag::ExprOriginBlobStarts, exprOriginMapBlobStarts);
            visit(ContainerTag::ExprOriginBlobPool, exprOriginMapBlobPool);
            // Internal-mail channels (tags 455..464 / 505..514) — after the
            // exprOrigin band so the enumeration stays ascending. Spliced via
            // ColdMail::visitContainers(base), the HashMemory-at-a-base pattern.
            sameInternalMail.visitContainers(kSameInternalMailDeloadBase,
                                             visitHashMemory);
            nextInternalMail.visitContainers(kNextInternalMailDeloadBase,
                                             visitHashMemory);
            // Changed-classes delta (tags 655..657) — dischargeable.
            changedClassesThisStep.visitContainers(kChangedClassesDeloadBase,
                                                   visitHashMemory);
            // EqClassNameCaches memo (tags 705..709) — survives discharge.
            eqClassNameCaches.visitContainers(kEqClassNameCachesDeloadBase,
                                              visitHashMemory);
            // Outgoing routing mail (tags 755..762) — private string table plus
            // statements and origins, all deloaded as one unit.
            mailOut.visitContainers(kMailOutDeloadBase, visitHashMemory);
            // Sequenced or-disintegration pending queue (tags 805..810) —
            // dischargeable.
            visit(ContainerTag::OrPendingBranchesKeys, orPendingBranchesKeys);
            visit(ContainerTag::OrPendingBranchesRunStarts,
                  orPendingBranchesRunStarts);
            visit(ContainerTag::OrPendingBranchesValues,
                  orPendingBranchesValues);
            visit(ContainerTag::OrPendingLevelsKeys, orPendingLevelsKeys);
            visit(ContainerTag::OrPendingLevelsRunStarts,
                  orPendingLevelsRunStarts);
            visit(ContainerTag::OrPendingLevelsValues, orPendingLevelsValues);
            visit(ContainerTag::PendingRelayIterKeys, pendingRelayIterKeys);
            visit(ContainerTag::PendingRelayIterValues, pendingRelayIterValues);
            visit(ContainerTag::PendingRelayLevelsKeys, pendingRelayLevelsKeys);
            visit(ContainerTag::PendingRelayLevelsRunStarts,
                  pendingRelayLevelsRunStarts);
            visit(ContainerTag::PendingRelayLevelsValues,
                  pendingRelayLevelsValues);
            visit(ContainerTag::CompactExpansionsKeys, compactExpansionsKeys);
            visit(ContainerTag::CompactExpansionsRunStarts, compactExpansionsRunStarts);
            visit(ContainerTag::CompactExpansionsBlobStarts, compactExpansionsBlobStarts);
            visit(ContainerTag::CompactExpansionsBlobPool, compactExpansionsBlobPool);
            visit(ContainerTag::ExpansionCarrierCountKeys, expansionCarrierCountKeys);
            visit(ContainerTag::ExpansionCarrierCountValues, expansionCarrierCountValues);
        }

        /// @brief Whether a tag's content survives discharge (rides the
        ///        near-empty image instead of being emptied).
        ///
        /// @details
        /// The cold STRING tags and the NameMap validity-metadata tags opt out
        /// of discharging: RAM-side id containers (admission maps, registries,
        /// the packed-key statement maps that key on validity ids) still hold
        /// ids into them, and clearing the backings would leave those ids
        /// dangling against an assert instead of against content — and an
        /// emptied metadata container would read as "unseeded", asserting on a
        /// non-root id. The cost — a discharged LB keeps this content until the
        /// pending drain dumps its image — is bounded by the pool telemetry.
        /// (D-153.)
        ///
        /// @param tag The container tag.
        /// @return `true` for the cold string + validity-node tags (4..18).
        static constexpr bool survivesDischarge(ContainerTag tag) {
            return (tag >= ContainerTag::TemplateStringLengths
                    && tag <= ContainerTag::ValidityNodes)
                || tag == ContainerTag::IntLocalEncodedStatementsSet
                // The four HashMemory instances (tags 51..450) survive discharge:
                // a discharged LB stays resident with its hash engine intact so
                // the post-prove chapter export reads its origin history, then the
                // pending drain dumps the full image — matching the pre-fold
                // extra-column behaviour where discharge never emptied HashMemory
                // (D-147).
                || (static_cast<uint32_t>(tag) >= kOverallHashMemoryDeloadBase
                    && static_cast<uint32_t>(tag)
                           <= kWorkingMemoryDeloadBase + 99u)
                // Batch 5: exprOriginMap (tags 451..454) survives discharge — the
                // post-prove chapter export reads origin history on discharged LBs
                // (matching the pre-statification heap map, never wiped on teardown,
                // I-44 / I-121).
                || (tag >= ContainerTag::ExprOriginKeys
                    && tag <= ContainerTag::ExprOriginBlobPool)
                // Internal-mail channels (tags 455..554) survive discharge: the
                // heap never cleared internal mail at discharge, so the faithful
                // cold mirror keeps them (I-102).
                || (static_cast<uint32_t>(tag) >= kSameInternalMailDeloadBase
                    && static_cast<uint32_t>(tag)
                           <= kNextInternalMailDeloadBase + 49u)
                // eqClassNameCaches (tags 705..754) survives discharge: a
                // persistent derived memo the post-prove readers + cross-grid
                // reuse keep resident.
                || (static_cast<uint32_t>(tag) >= kEqClassNameCachesDeloadBase
                    && static_cast<uint32_t>(tag)
                           <= kEqClassNameCachesDeloadBase + 49u)
                // Pending outgoing mail survives discharge so the final serial
                // commit can reload and publish it.
                || (static_cast<uint32_t>(tag) >= kMailOutDeloadBase
                    && static_cast<uint32_t>(tag)
                           <= kMailOutDeloadBase + 49u);
        }

        /// @brief Empty every dischargeable container — the first act of the
        ///        discharge protocol.
        ///
        /// @details
        /// Tag semantics: ALL FOUR tier-1 tags (the statement vectors) are
        /// **dischargeable** — their content is dead once the LB leaves the
        /// active set forever (the post-prove readers probe exact RAM-side
        /// records instead, D-157). The cold string
        /// tags survive (`survivesDischarge`) and ride the eventual image.
        /// The emptied storage becomes arena holes the copying compaction
        /// reclaims.
        void clearDischargeableContainers() {
            visitContainers([](ContainerTag tag, auto& container) {
                if (survivesDischarge(tag)) return;
                container.clear();
            });
        }

        /// @brief Total live byte footprint across every container — the
        ///        steward's fragmentation gate compares it to
        ///        `manager.usedBytes()` to size the reclaimable holes.
        ///
        /// @details
        /// Sums each container's `liveBytes()` (elements/strings + their index
        /// structures). `manager.usedBytes() - liveBytes()` is the arena's
        /// dead space the copying compaction would reclaim. Deterministic
        /// (logical counts only).
        ///
        /// @return Live bytes summed over all statified containers.
        int64_t liveBytes() const {
            return intEncodedStatements.liveBytes()
                 + intLocalEncodedStatements.liveBytes()
                 + intLocalEncodedStatementsDelta.liveBytes()
                 + intExternalStatements.liveBytes()
                 + templateStrings.liveBytes()
                 + valueStrings.liveBytes()
                 + originStrings.liveBytes()
                 + ruleStrings.liveBytes()
                 + lbStateStrings.liveBytes()
                 + nameStrings.liveBytes()
                 + subStrings.liveBytes()
                 + validityNodes.liveBytes()
                 + intValidityNamesToFilter.liveBytes()
                 + intAxedVariables.liveBytes()
                 + canBeSentIds.liveBytes()
                 + canBeSentMarkerIds.liveBytes()
                 + pendingWipeScopes.liveBytes()
                 + orDisjunctCount.liveBytes()
                 + integrationStartIntMap.liveBytes()
                 + intLocalEncodedStatementsSet.liveBytes()
                 + intWeakVariables.liveBytes()
                 + integrationPrepared.liveBytes()
                 + integrationPreparedMarker.liveBytes()
                 + expandedImplications.liveBytes()
                 + compactExpansions.liveBytes()
                 + expansionCarrierCount.liveBytes()
                 + intKnownStatements.liveBytes()
                 + equivalenceClassesMap.liveBytes()
                 + exprOriginMap.liveBytes()
                 + overallHashMemory.liveBytes()
                 + localHashMemory.liveBytes()
                 + localHashMemoryDelta.liveBytes()
                 + workingMemory.liveBytes()
                 + sameInternalMail.liveBytes()
                 + nextInternalMail.liveBytes()
                 + changedClassesThisStep.liveBytes()
                 + eqClassNameCaches.liveBytes()
                 + mailOut.liveBytes()
                 + orPendingBranches.liveBytes()
                 + orPendingLevels.liveBytes()
                 + pendingRelayIter.liveBytes()
                 + pendingRelayLevels.liveBytes();
        }

        /// @brief Compaction: pack the LB's live pages onto the contiguous
        ///        prefix of its blocks and return the emptied blocks to the
        ///        pool — the steward's background reclaim.
        ///
        /// @details
        /// Every cold container is paged, so reclamation is the arena's in-place
        /// page pack (`LbArena::compactPages`): live pages move onto the block
        /// prefix in vid order (one `memcpy` per page, cycle-following with a
        /// single scratch page) and the emptied blocks return to the pool — no
        /// heap round trip.
        ///
        /// CONTENT-INVISIBLE: a container addresses its pages by stable virtual
        /// id and resolves through `pageAt` on every access, so only the
        /// vid→physical bindings move — no container is touched, no element
        /// reference is rebuilt, and the `dirty` state is left alone (a
        /// compaction must never force a deload rewrite). MUST run under
        /// exclusive LB access (steward claim or single-threaded barrier),
        /// never during a burst when phase-2 holds resolved element pointers
        /// (I-107).
        ///
        /// @param scr A scratch arena bound to a never-deloaded pool, exclusive
        ///            to the calling thread (forwarded to `LbArena::compactPages`
        ///            — the reclaim's scratch must not draw from the deloadable
        ///            pool it frees).
        /// @return Number of blocks returned to the global pool.
        int64_t reshuffle(LbArena& scr) {
            return manager.compactPages(scr);
        }
    };

}
