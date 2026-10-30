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
#include "shared_expression_list.hpp"
#include "mail_types.hpp"
#include "cold_mail.hpp"
#include "changed_classes_buffer.hpp"
#include "eq_class_name_caches.hpp"
#include "deloadable_mail_out.hpp"
#include "or_implications.hpp"
#include "or_implication_match_index.hpp"
#include "existence_lookup_indexes.hpp"
#include "canonical_use_index.hpp"
#include "unary_composition_store.hpp"
#include "contradiction_fact_index.hpp"

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

    /// @brief One integration trial: a gated integration admission made in
    ///        its own trial scope until its gate elements are known there.
    ///
    /// @details
    /// The record of `LbMemory::integrationTrials`
    /// (D-481), keyed by the packed
    /// `(canonical source, parent scope)` pair of the existence whose
    /// admission is on trial. `trialScope` is the minted
    /// `<parent>_boundary_integration_<names>` scope, `<names>` the witnesses
    /// the admission mints; `source` names the concrete source text the
    /// promotion admits at the parent; `generation`, `integrationDepth` and
    /// `flags` are the source's replay metadata, with the park-flag layout
    /// (2: integration construction, 4: OR disintegration allowed, 8:
    /// inherited relay source).
    ///
    /// The gate elements are NameMap ids of the fully fixed non-implication
    /// elements of the gated instructions, each filled argument replaced by
    /// the new name filling it. They are grouped per gated instruction: group
    /// `g` covers `gates[groupEnd[g - 1] .. groupEnd[g])` (from 0 for
    /// `g == 0`). The trial passes when every gate of one group is known at
    /// the trial scope. A passed record stays as the parent's receipt: the
    /// promotion admits at the parent the witnesses and expressions listed in
    /// `LbMemory::integrationTrialContents` under `trialScope` —
    /// `witnessCount` witnesses, then `expressionCount` expressions.
    ///
    /// Every field is a 32-bit integer and unused slots stay zero, so the
    /// record is padding-free and its deload bytes are a pure function of its
    /// content.
    ///
    /// @see LbMemory::integrationTrials, ScopeSyntax::atOrBelowIntegrationTrial.
    struct IntegrationTrialRec {
        /// @brief `status`: the gate is not yet known at the trial scope.
        static constexpr int32_t kOpen = 0;
        /// @brief `status`: the gate became known; the parent admits the source.
        static constexpr int32_t kPassed = 1;
        /// @brief Capacity of `groupEnd` — gated instructions per trial.
        static constexpr int32_t kMaxGroups = 8;
        /// @brief Capacity of `gates` — gate elements per trial.
        static constexpr int32_t kMaxGates = 16;

        NameId trialScope = 0;
        NameId source = 0;
        int32_t status = kOpen;
        int32_t generation = 0;
        int32_t integrationDepth = 0;
        int32_t flags = 0;
        int32_t groupCount = 0;
        int32_t groupEnd[kMaxGroups] = {};
        NameId gates[kMaxGates] = {};
        int32_t witnessCount = 0;
        int32_t expressionCount = 0;
    };

    /// @brief Kind of one listed expression of an integration trial: a
    ///        statement.
    static constexpr int32_t kIntegrationTrialStatement = 0;
    /// @brief Kind of one listed expression of an integration trial: an
    ///        implication extracted as a rule.
    static constexpr int32_t kIntegrationTrialRule = 1;

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
            // 21 and 22 are retired mail-permission container tags.
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
            // the cohort's OR-source levels until retirement. Convergence
            // adds them to the conditional cases' levels (ascending ints).
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
            UsedAsPremiseCorePairs = 822,
            // Tag 823 is retired: the unsigned retained-usage set (goals,
            // negations, seeds) has no stream — those rows are watched
            // through the persistent ledgers, never multiplied.
            AppliedPremiseCorePairs = 824,
            // Tag 825 is retired; suppressed statement identities have no stream.
            ClosedOrGoalLevelsKeys = 826,
            ClosedOrGoalLevelsRunStarts = 827,
            ClosedOrGoalLevelsValues = 828,
            OrImplicationsBase = 829,
            ExistenceReplaysKeys = 837,
            ExistenceReplaysValues = 838,
            CanonicalEquivalenceClassesKeys = 839,
            CanonicalEquivalenceClassesRunStarts = 840,
            CanonicalEquivalenceClassesBlobStarts = 841,
            CanonicalEquivalenceClassesBlobPool = 842,
            CanonicalMemberRepresentativesKeys = 843,
            CanonicalMemberRepresentativesValues = 844,
            CanonicalClassChangesKeys = 845,
            CanonicalClassChangesValues = 846,
            UnaryCompositionBase = 847,
            UnaryCompositionEnd = 868,
            // Tags 869 through 871 are retired and must not be reassigned.
            OrSupportFactsKeys = 872,
            OrSupportFactsValues = 873,
            OrSupportEqualitiesKeys = 874,
            OrSupportEqualitiesRunStarts = 875,
            OrSupportEqualitiesValues = 876,
            // Tags 877 through 893 are retired and must not be reassigned.
            // Goal enrollment of unary composition (D-469).
            UnaryCompositionGoalsKeys = 894,
            UnaryCompositionGoalsRunStarts = 895,
            UnaryCompositionGoalsValues = 896,
            UnaryCompositionObservedGoals = 897,
            // Integration trials (D-481):
            // dischargeable side table.
            IntegrationTrialsKeys = 898,
            IntegrationTrialsValues = 899,
            // What an integration trial admits on success
            // (D-481): dischargeable side table.
            IntegrationTrialContentsKeys = 900,
            IntegrationTrialContentsValues = 901,
            // The slot the induction hypothesis ate at each scope
            // (D-519): dischargeable side table.
            InductionEatsKeys = 902,
            InductionEatsValues = 903,
            // The burst's induction-hypothesis firings and the settled
            // induction trees (D-520): dischargeable.
            InductionFirings = 904,
            InductionSettledRoots = 905,
            // ORs reached by unfolding a conjecture premise, awaiting the
            // flag-6 unconditional-ordis mail
            // (D-528): dischargeable side table.
            PendingUnconditionalOrdis = 906,
            // Cohorts opened from a conjecture premise's OR, whose case
            // assumptions are disintegrated with force_ witnesses
            // (D-527): dischargeable side table.
            ForcedOrCohorts = 907,
            // Statements a forced disintegration produced, and those a parent
            // shipped with the from-premise mail flag
            // (D-522): dischargeable side table.
            ForcedProducts = 908,
            // Open negated-compound contradiction scopes the demand drain
            // serves, with their bootstrap state
            // (D-560): dischargeable side table.
            ContradictionScopesKeys = 909,
            ContradictionScopesValues = 910,
            // Cohorts opened by a demand of the induction hypothesis, released
            // below an unsettled induction scope
            // (D-575): dischargeable side table.
            HypothesisOrCohorts = 911,
            // Goal-driven marked-head implications installed per scope
            // (D-574): dischargeable side
            // table.
            GoalDemandInstalls = 912,
            // Scopes where the induction hypothesis shot
            // (D-572): dischargeable side
            // table.
            HypothesisShots = 913,
            // Int scopes of premise-bound integrations: each scope's goal
            // group and premise level, each group's premise levels and
            // promoted mark, and each premise's first finished scope
            // (D-578): dischargeable side tables.
            IntScopesKeys = 914,
            IntScopesValues = 915,
            IntScopeGroupsKeys = 916,
            IntScopeGroupsValues = 917,
            IntScopeFirstFinishKeys = 918,
            IntScopeFirstFinishValues = 919,
            // Size markers of a weak step block
            // (D-583): dischargeable side table.
            SizeMarkers = 920,
            // Witnesses a route-3 commit wrote into an integration instruction
            // (D-585): dischargeable side table.
            OneWayWitnesses = 921,
            // Case-experiment cohorts of a twin block, which release one case
            // only (D-588): dischargeable side
            // table.
            ExperimentOrCohorts = 922,
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
              sharedExpressions(&manager),
              intEncodedStatements(&manager, &dirty, &sharedExpressions),
              intLocalEncodedStatements(&manager, &dirty, &sharedExpressions),
              intLocalEncodedStatementsDelta(&manager, &dirty, &sharedExpressions),
              intExternalStatements(&manager, &dirty, &sharedExpressions),
              templateStrings(&manager, &dirty),
              templateStringsLengths(&templateStrings),
              templateStringsBytes(&templateStrings),
              valueStrings(&manager, &dirty),
              valueStringsLengths(&valueStrings),
              valueStringsBytes(&valueStrings),
              originPrefixes(&manager, &dirty),
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
              sharedNativeKeys(&manager, &dirty),
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
              expansionCarrierCountValues(&expansionCarrierCount) {
            // Pool attribution: every block this aggregate draws is logic-block
            // state (observability only; the arena holds no block yet).
            manager.setPurpose(PoolPurpose::LbState);
            // Small persistent containers take sub-page slots instead of whole
            // pages (D-464).
            manager.enableSubPages();
            originStrings.keyStore().bindPrefixes(originPrefixes);
            overallHashMemory.bindSharedBytes(sharedNativeKeys);
            localHashMemory.bindSharedBytes(sharedNativeKeys);
            localHashMemoryDelta.bindSharedBytes(sharedNativeKeys);
            workingMemory.bindSharedBytes(sharedNativeKeys);
            // The stores whose key bytes recur across logic blocks hold
            // references into the process-wide batch byte dictionary.
            forEachBatchSharedStore([](BytesKeyStore& store) { store.enableBatchSharing(); });
            // The arena's directories use this object's direct storage (see
            // kPageTableDirectEntries).
            manager.attachDirectoryStorage(pageTableDirect_, kPageTableDirectEntries,
                                           pageBlockDirect_, kPageBlockDirectEntries);
        }

        LbMemory(const LbMemory&) = delete;
        LbMemory& operator=(const LbMemory&) = delete;

        /// @brief Visit every byte-key store that takes part in the batch byte
        ///        dictionary.
        ///
        /// @details
        /// The seven per-block string tables and the payload store of the
        /// per-block native-key pool (`sharedNativeKeys`), whose clients keep
        /// their block-local pool identifiers. These are the stores whose bytes
        /// recur across logic blocks (the F12 structure census).
        ///
        /// @tparam F Callable `void(BytesKeyStore&)`.
        /// @param visit Called once per participating store, in a fixed order.
        /// @return Nothing.
        template <typename F>
        void forEachBatchSharedStore(F&& visit) {
            visit(templateStrings.keyStore());
            visit(valueStrings.keyStore());
            visit(originStrings.keyStore());
            visit(ruleStrings.keyStore());
            visit(lbStateStrings.keyStore());
            visit(nameStrings.keyStore());
            visit(subStrings.keyStore());
            visit(sharedNativeKeys.payloadStore());
        }

        /// @brief End-of-phase-1 batch-sharing seam: convert the rows nominated
        ///        before the last barrier whose bytes the dictionary now holds.
        ///
        /// @details
        /// Runs under the worker's exclusive claim of this logic block while the
        /// dictionary is read-only. Converted stores reclaim their local bytes
        /// at once. Content, identifiers, hashes and canonical images are
        /// unchanged; only where the bytes live changes.
        ///
        /// @param scratch The worker slot's page-tier scratch arena (prefix
        ///        repack only).
        /// @return Nothing.
        void shareBatchBytesAfterPhase1(LbArena& scratch) {
            RT_SCOPE_HERE("PH1_BATCH_BYTES_CONVERT_PENDING");
            forEachBatchSharedStore([&](BytesKeyStore& store) {
                if (store.sharePendingRows() == 0) return;
                RT_SCOPE_HERE("BATCH_BYTES_COMPACT");
                store.compactAfterBatchConversion(scratch);
            });
        }

        /// @brief End-of-phase-3 batch-sharing seam: convert the rows minted
        ///        since the last scan whose bytes the dictionary holds, and
        ///        nominate the rest.
        ///
        /// @details
        /// Runs under the worker's exclusive claim of this logic block while the
        /// dictionary is read-only. Each missing payload goes to the shared,
        /// hash-striped nomination table, which keeps one copy per distinct
        /// payload; the barrier promotes payloads nominated at least twice.
        /// Converted stores reclaim their local bytes at once.
        ///
        /// @param candidates The process-wide striped nomination table.
        /// @param scratch The worker slot's page-tier scratch arena (prefix
        ///        repack only).
        /// @return Nothing.
        void shareBatchBytesAfterPhase3(BatchByteCandidateRegistry& candidates, LbArena& scratch) {
            RT_SCOPE_HERE("PH3_BATCH_BYTES_SCAN_NEW");
            forEachBatchSharedStore([&](BytesKeyStore& store) {
                const int32_t rows = store.shareNewRows(
                    [&](StrSpan bytes, uint64_t hash) { candidates.nominate(bytes, hash); });
                if (rows == 0) return;
                RT_SCOPE_HERE("BATCH_BYTES_COMPACT");
                store.compactAfterBatchConversion(scratch);
            });
        }

        /// @brief Direct entries of the arena's page table (vid -> page), held
        ///        in this object instead of a spilled pool block.
        ///
        /// @details
        /// A logic block's `Memory` shell takes one `LbStore` slot of an
        /// LB-pool block with room to spare; `pageTableDirect_` and
        /// `pageBlockDirect_` use that room, so the directories of a logic
        /// block with up to this many pages need no pool block of their own.
        /// Sized so the shell still fits one 256 KiB LB-pool block with about
        /// 12 KiB spare; `LbStore` asserts if `Memory` outgrows the block, and
        /// these capacities are then the ones to shrink. The arrays are
        /// declared before `manager` so they outlive its directories, and are
        /// never read past the directory's size, so they are not initialized.
        static constexpr int32_t kPageTableDirectEntries = 12288;

        /// @brief Direct entries of the arena's carved-block list (see
        ///        `kPageTableDirectEntries`).
        static constexpr int32_t kPageBlockDirectEntries = 512;

        char* pageTableDirect_[kPageTableDirectEntries];
        char* pageBlockDirect_[kPageBlockDirectEntries];

        LbArena manager;
        // Content-change state shared by every statified container (set by
        // all their mutators): Clean = the on-disk file set still equals the
        // in-memory content (deload releases blocks without writing);
        // AppendedOnly = only push_backs since the last dump (deload may write
        // a small tail file); Restructured = full canonical rewrite required.
        // Reset to Clean after a dump and after a reload (both points where
        // RAM == disk by construction).
        DirtyState dirty = DirtyState::Clean;
        SharedExpressionStore sharedExpressions;
        SharedExpressionList intEncodedStatements;
        SharedExpressionList intLocalEncodedStatements;
        SharedExpressionList intLocalEncodedStatementsDelta;
        SharedExpressionList intExternalStatements;
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
        PrefixKeyMetadata originPrefixes;
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
        // Derived immutable payload owner. Canonical streams remain per map;
        // clients destruct first and raw eviction preserves all shared pages.
        SharedBytePool sharedNativeKeys;
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
        // cohortId -> OR-source levels retained through queue exhaustion and
        // added at convergence (ascending ints, natural order). Both dischargeable, rebuilt per grid like orBookkeeping /
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
            visit(ContainerTag::UsedAsPremiseCorePairs, usedAsPremiseCorePairsKeys);
            visit(ContainerTag::AppliedPremiseCorePairs, appliedPremiseCorePairsKeys);
            visit(ContainerTag::ClosedOrGoalLevelsKeys, closedOrGoalLevelsKeys);
            visit(ContainerTag::ClosedOrGoalLevelsRunStarts, closedOrGoalLevelsRunStarts);
            visit(ContainerTag::ClosedOrGoalLevelsValues, closedOrGoalLevelsValues);
            orImplications.visitContainers(
                static_cast<uint32_t>(ContainerTag::OrImplicationsBase),
                [&](uint32_t tag, auto& column) {
                    visit(static_cast<ContainerTag>(tag), column);
                });
            visit(ContainerTag::ExistenceReplaysKeys, existenceReplaysKeys);
            visit(ContainerTag::ExistenceReplaysValues, existenceReplaysValues);
            visit(ContainerTag::CanonicalEquivalenceClassesKeys, canonicalEquivalenceClassesKeys);
            visit(ContainerTag::CanonicalEquivalenceClassesRunStarts, canonicalEquivalenceClassesRunStarts);
            visit(ContainerTag::CanonicalEquivalenceClassesBlobStarts, canonicalEquivalenceClassesBlobStarts);
            visit(ContainerTag::CanonicalEquivalenceClassesBlobPool, canonicalEquivalenceClassesBlobPool);
            visit(ContainerTag::CanonicalMemberRepresentativesKeys, canonicalMemberRepresentativesKeys);
            visit(ContainerTag::CanonicalMemberRepresentativesValues, canonicalMemberRepresentativesValues);
            visit(ContainerTag::CanonicalClassChangesKeys, canonicalClassChangesKeys);
            visit(ContainerTag::CanonicalClassChangesValues, canonicalClassChangesValues);
            unaryComposition.visitContainers(static_cast<uint32_t>(ContainerTag::UnaryCompositionBase),
                [&](uint32_t tag, auto& column) { visit(static_cast<ContainerTag>(tag), column); });

            visit(ContainerTag::OrSupportFactsKeys, orSupportFactsKeys);
            visit(ContainerTag::OrSupportFactsValues, orSupportFactsValues);
            visit(ContainerTag::OrSupportEqualitiesKeys, orSupportEqualitiesKeys);
            visit(ContainerTag::OrSupportEqualitiesRunStarts, orSupportEqualitiesRunStarts);
            visit(ContainerTag::OrSupportEqualitiesValues, orSupportEqualitiesValues);
            unaryComposition.visitGoalContainers(static_cast<uint32_t>(ContainerTag::UnaryCompositionGoalsKeys),
                [&](uint32_t tag, auto& column) { visit(static_cast<ContainerTag>(tag), column); });
            visit(ContainerTag::IntegrationTrialsKeys, integrationTrialsKeys);
            visit(ContainerTag::IntegrationTrialsValues, integrationTrialsValues);
            visit(ContainerTag::IntegrationTrialContentsKeys, integrationTrialContentsKeys);
            visit(ContainerTag::IntegrationTrialContentsValues, integrationTrialContentsValues);
            visit(ContainerTag::InductionEatsKeys, inductionEatsKeys);
            visit(ContainerTag::InductionEatsValues, inductionEatsValues);
            visit(ContainerTag::InductionFirings, inductionFiringsKeys);
            visit(ContainerTag::InductionSettledRoots, inductionSettledRootsKeys);
            visit(ContainerTag::PendingUnconditionalOrdis, pendingUnconditionalOrdisKeys);
            visit(ContainerTag::ForcedOrCohorts, forcedOrCohortsKeys);
            visit(ContainerTag::ForcedProducts, forcedProductsKeys);
            visit(ContainerTag::ContradictionScopesKeys, contradictionScopesKeys);
            visit(ContainerTag::ContradictionScopesValues, contradictionScopesValues);
            visit(ContainerTag::HypothesisOrCohorts, hypothesisOrCohortsKeys);
            visit(ContainerTag::GoalDemandInstalls, goalDemandInstallsKeys);
            visit(ContainerTag::HypothesisShots, hypothesisShotsKeys);
            visit(ContainerTag::IntScopesKeys, intScopesKeys);
            visit(ContainerTag::IntScopesValues, intScopesValues);
            visit(ContainerTag::IntScopeGroupsKeys, intScopeGroupsKeys);
            visit(ContainerTag::IntScopeGroupsValues, intScopeGroupsValues);
            visit(ContainerTag::IntScopeFirstFinishKeys, intScopeFirstFinishKeys);
            visit(ContainerTag::IntScopeFirstFinishValues, intScopeFirstFinishValues);
            visit(ContainerTag::SizeMarkers, sizeMarkersKeys);
            visit(ContainerTag::OneWayWitnesses, oneWayWitnessesKeys);
            visit(ContainerTag::ExperimentOrCohorts, experimentOrCohortsKeys);
        }

        // Current signed premise eligibility (the only multiplication and
        // GPU-upload criterion) and completed class coverage.
        ColdHashSet<PodKeyStore<int64_t>> usedAsPremiseCorePairs{ &manager, &dirty };
        ColdHashSet<PodKeyStore<int64_t>>::KeysView usedAsPremiseCorePairsKeys{ &usedAsPremiseCorePairs };
        ColdHashSet<PodKeyStore<int64_t>> appliedPremiseCorePairs{ &manager, &dirty };
        ColdHashSet<PodKeyStore<int64_t>>::KeysView appliedPremiseCorePairsKeys{ &appliedPremiseCorePairs };

        // Derivation levels of current chain goals booked before a successful
        // OR branch wipe. The key uses NameMap expression and branch-scope ids.
        // This convergence-only state deloads normally and clears at discharge.
        TypedColdSetMap<StatementKey, int> closedOrGoalLevels{ &manager, &dirty };
        TypedColdSetMap<StatementKey, int>::KeysView closedOrGoalLevelsKeys{ &closedOrGoalLevels.inner() };
        TypedColdSetMap<StatementKey, int>::RunStartsView closedOrGoalLevelsRunStarts{ &closedOrGoalLevels.inner() };
        TypedColdSetMap<StatementKey, int>::RunValuesView closedOrGoalLevelsValues{ &closedOrGoalLevels.inner() };

        // One canonical source store per block; its eight facets occupy
        // tags 829 through 836 and are discharged with active proof work.
        OrImplications orImplications{ &manager, &dirty };
        OrImplicationMatchIndex orImplicationMatches{ &manager };
        ExistenceLookupIndexes existenceLookups{ &manager };
        CanonicalUseIndexes canonicalUses{ &manager };
        UnaryCompositionStore unaryComposition{ &manager, &dirty };

        // A target/branch contribution cites its actual fact and the positive
        // equalities used in that branch. These are bookkeeping, not statements.
        // The paired level run lives in closedOrGoalLevels and survives freezing.
        TypedColdMap<StatementKey, int64_t> orSupportFacts{ &manager, &dirty };
        TypedColdMap<StatementKey, int64_t>::KeysView orSupportFactsKeys{ &orSupportFacts.inner() };
        TypedColdMap<StatementKey, int64_t>::ValuesView orSupportFactsValues{ &orSupportFacts.inner() };
        TypedColdSetMap<StatementKey, int64_t> orSupportEqualities{ &manager, &dirty };
        TypedColdSetMap<StatementKey, int64_t>::KeysView orSupportEqualitiesKeys{ &orSupportEqualities.inner() };
        TypedColdSetMap<StatementKey, int64_t>::RunStartsView orSupportEqualitiesRunStarts{ &orSupportEqualities.inner() };
        TypedColdSetMap<StatementKey, int64_t>::RunValuesView orSupportEqualitiesValues{ &orSupportEqualities.inner() };

        // Complete scope-visible classes, used only for canonical identities.
        TypedColdBlobMap<NameId, EquivalenceClass> canonicalEquivalenceClassesMap{ &manager, &dirty };
        TypedColdBlobMap<NameId, EquivalenceClass>::KeysView canonicalEquivalenceClassesKeys{ &canonicalEquivalenceClassesMap.inner() };
        TypedColdBlobMap<NameId, EquivalenceClass>::RunStartsView canonicalEquivalenceClassesRunStarts{ &canonicalEquivalenceClassesMap.inner() };
        TypedColdBlobMap<NameId, EquivalenceClass>::BlobStartsView canonicalEquivalenceClassesBlobStarts{ &canonicalEquivalenceClassesMap.inner() };
        TypedColdBlobMap<NameId, EquivalenceClass>::BlobPoolView canonicalEquivalenceClassesBlobPool{ &canonicalEquivalenceClassesMap.inner() };

        // Previous target-scope normal/special representatives and pending
        // canonical changes, keyed by (member, scope), independently of application.
        TypedColdMap<int64_t, int64_t> canonicalMemberRepresentatives{ &manager, &dirty };
        TypedColdMap<int64_t, int64_t>::KeysView canonicalMemberRepresentativesKeys{ &canonicalMemberRepresentatives.inner() };
        TypedColdMap<int64_t, int64_t>::ValuesView canonicalMemberRepresentativesValues{ &canonicalMemberRepresentatives.inner() };
        TypedColdMap<int64_t, int32_t> canonicalClassChanges{ &manager, &dirty };
        TypedColdMap<int64_t, int32_t>::KeysView canonicalClassChangesKeys{ &canonicalClassChanges.inner() };
        TypedColdMap<int64_t, int32_t>::ValuesView canonicalClassChangesValues{ &canonicalClassChanges.inner() };

        // Source/scope receipts: completed construction bits 0..1 and queued bits 2..3.
        TypedColdMap<int64_t, int32_t> existenceReplays{ &manager, &dirty };
        TypedColdMap<int64_t, int32_t>::KeysView existenceReplaysKeys{ &existenceReplays.inner() };
        TypedColdMap<int64_t, int32_t>::ValuesView existenceReplaysValues{ &existenceReplays.inner() };

        // Integration trials (D-481),
        // keyed by the packed (canonical source, parent scope) pair. An open
        // record leaves with its trial scope, every record with its parent
        // scope (`wipeSubtrees`). Dischargeable.
        TypedColdMap<int64_t, IntegrationTrialRec> integrationTrials{ &manager, &dirty };
        TypedColdMap<int64_t, IntegrationTrialRec>::KeysView integrationTrialsKeys{ &integrationTrials.inner() };
        TypedColdMap<int64_t, IntegrationTrialRec>::ValuesView integrationTrialsValues{ &integrationTrials.inner() };
        // What each integration trial admits on success, keyed by the packed
        // (trial scope, index) pair: indices below the record's
        // `witnessCount` hold (slot ordinal, witness name) pairs in slot
        // order, the next `expressionCount` hold (kind, expression) pairs,
        // kind `kIntegrationTrialStatement` or `kIntegrationTrialRule`
        // (D-481). Leaves with its record.
        // Dischargeable.
        TypedColdMap<int64_t, int64_t> integrationTrialContents{ &manager, &dirty };
        TypedColdMap<int64_t, int64_t>::KeysView integrationTrialContentsKeys{ &integrationTrialContents.inner() };
        TypedColdMap<int64_t, int64_t>::ValuesView integrationTrialContentsValues{ &integrationTrialContents.inner() };
        // Induction eats (D-519),
        // keyed by the scope where the induction hypothesis admitted a witness:
        // the slot it ate there — the admitting template's id, or
        // `Memory::kStrongInductionSlot` for the strong demand. One order per
        // path: at that scope the hypothesis eats no other slot. Leaves with
        // its scope (`wipeSubtrees`). Dischargeable.
        TypedColdMap<int64_t, int32_t> inductionEats{ &manager, &dirty };
        TypedColdMap<int64_t, int32_t>::KeysView inductionEatsKeys{ &inductionEats.inner() };
        TypedColdMap<int64_t, int32_t>::ValuesView inductionEatsValues{ &inductionEats.inner() };
        // Induction tree settlement (D-520): the burst's
        // firings of the induction hypothesis at induction scopes, drained and
        // reset every burst by `Memory::settleInductionTrees`; and the roots of
        // settled trees, whose first firing wiped the rest of the tree - no
        // induction scope opens below a settled root. A settled root leaves
        // with its scope (`wipeSubtrees`). Both dischargeable.
        ColdHashSet<PodKeyStore<NameId>> inductionFirings{ &manager, &dirty };
        ColdHashSet<PodKeyStore<NameId>>::KeysView inductionFiringsKeys{ &inductionFirings };
        ColdHashSet<PodKeyStore<NameId>> inductionSettledRoots{ &manager, &dirty };
        ColdHashSet<PodKeyStore<NameId>>::KeysView inductionSettledRootsKeys{ &inductionSettledRoots };
        // ORs reached by unfolding a conjecture premise (NameMap expression ids
        // at main) awaiting the flag-6 unconditional-ordis mail; `fillMailOut`
        // erases each once it ships
        // (D-528). Dischargeable.
        ColdHashSet<PodKeyStore<NameId>> pendingUnconditionalOrdis{ &manager, &dirty };
        ColdHashSet<PodKeyStore<NameId>>::KeysView pendingUnconditionalOrdisKeys{ &pendingUnconditionalOrdis };
        // OR cohorts (lbStateInterner cohort ids) opened from a conjecture
        // premise's OR, in place or on a flag-6 arrival: each case
        // `drainPendingOrReleases` releases carries the forced signal, and the
        // absorb disintegrates it with force_ witnesses
        // (D-527). Dischargeable.
        ColdHashSet<PodKeyStore<int32_t>> forcedOrCohorts{ &manager, &dirty };
        ColdHashSet<PodKeyStore<int32_t>>::KeysView forcedOrCohortsKeys{ &forcedOrCohorts };
        // OR cohorts (lbStateInterner cohort ids) opened through a demand whose
        // source is an induction hypothesis rule (the hypothesis's missing
        // expression): `drainPendingOrReleases` releases their cases below an
        // unsettled induction scope, exempt from the I-328 block
        // (D-575). Dischargeable.
        ColdHashSet<PodKeyStore<int32_t>> hypothesisOrCohorts{ &manager, &dirty };
        ColdHashSet<PodKeyStore<int32_t>>::KeysView hypothesisOrCohortsKeys{ &hypothesisOrCohorts };
        // Goal-driven marked-head implications already installed: their rule
        // owners (`packRuleOwner` of the identity text's ruleInterner id and
        // the install scope's NameMap id). `IntegrationProcessor::serveGoalNearMiss`
        // installs each once, since an LMV run appends without dedup
        // (D-574). Dischargeable.
        ColdHashSet<PodKeyStore<int64_t>> goalDemandInstalls{ &manager, &dirty };
        ColdHashSet<PodKeyStore<int64_t>>::KeysView goalDemandInstallsKeys{ &goalDemandInstalls };
        // Scopes (NameMap ids) where the induction hypothesis shot: a
        // statement fired by an installed induction hypothesis rule reached
        // the statement door there (`recursionHead`). A commit served by an
        // ancestor's integration demand updates the ancestor's instruction
        // only after a shot on its path
        // (`AdmissionMemory::hypothesisShotOnPath`,
        // D-572). An entry leaves with its
        // scope. Dischargeable.
        ColdHashSet<PodKeyStore<NameId>> hypothesisShots{ &manager, &dirty };
        ColdHashSet<PodKeyStore<NameId>>::KeysView hypothesisShotsKeys{ &hypothesisShots };
        // Int scopes of premise-bound integrations
        // (D-578). `intScopes`: each open int scope
        // (NameMap id) -> `packInt32Pair(goal group, premise level)`, the goal
        // group being the NameMap id of the opening instruction's integration
        // goal with unlabelled markers; an entry leaves with its scope.
        // `intScopeGroups`: goal group -> the bit set of its premises' block
        // levels, bit 63 set once the group was promoted. `intScopeFirstFinish`:
        // `packInt32Pair(goal group, premise level)` -> the premise's first
        // finished scope. Both group tables outlive the wipe of the promoted
        // scopes: a group promotes once. A first finish of a group not yet
        // promoted leaves with its scope. All three dischargeable.
        TypedColdMap<int64_t, int64_t> intScopes{ &manager, &dirty };
        TypedColdMap<int64_t, int64_t>::KeysView intScopesKeys{ &intScopes.inner() };
        TypedColdMap<int64_t, int64_t>::ValuesView intScopesValues{ &intScopes.inner() };
        TypedColdMap<int64_t, int64_t> intScopeGroups{ &manager, &dirty };
        TypedColdMap<int64_t, int64_t>::KeysView intScopeGroupsKeys{ &intScopeGroups.inner() };
        TypedColdMap<int64_t, int64_t>::ValuesView intScopeGroupsValues{ &intScopeGroups.inner() };
        TypedColdMap<int64_t, int64_t> intScopeFirstFinish{ &manager, &dirty };
        TypedColdMap<int64_t, int64_t>::KeysView intScopeFirstFinishKeys{ &intScopeFirstFinish.inner() };
        TypedColdMap<int64_t, int64_t>::ValuesView intScopeFirstFinishValues{ &intScopeFirstFinish.inner() };
        // Size markers of a weak step block: the values (`valueInterner` ids)
        // filled into an integration instruction there, at the route-3
        // commit or the statement fill, each through a key connected to the
        // induction. A later key holding one is connected too
        // (D-583). A witness name is never
        // reused within a block, so an entry outlives every scope wipe.
        // Dischargeable.
        ColdHashSet<PodKeyStore<int32_t>> sizeMarkers{ &manager, &dirty };
        ColdHashSet<PodKeyStore<int32_t>>::KeysView sizeMarkersKeys{ &sizeMarkers };
        // One-way witnesses: the witnesses (NameMap ids) a route-3 commit
        // wrote into an integration instruction. The statement fill never
        // tries one of them, or a name of its class at the deposit scope, as
        // a marker again (D-585). A witness name is never
        // reused within a block, so an entry outlives every scope wipe.
        // Dischargeable.
        ColdHashSet<PodKeyStore<int32_t>> oneWayWitnesses{ &manager, &dirty };
        ColdHashSet<PodKeyStore<int32_t>>::KeysView oneWayWitnessesKeys{ &oneWayWitnesses };
        // OR cohorts (lbStateInterner cohort ids) a twin block opened as its
        // case experiment: `drainPendingOrReleases` releases their starter
        // case and never another one
        // (D-588). Dischargeable.
        ColdHashSet<PodKeyStore<int32_t>> experimentOrCohorts{ &manager, &dirty };
        ColdHashSet<PodKeyStore<int32_t>>::KeysView experimentOrCohortsKeys{ &experimentOrCohorts };
        // Statements produced by a forced disintegration, a status-0 seed or a
        // hypothesis shot (packed NameMap (expression id, scope id)), and the
        // statements a parent shipped with the from-premise mail flag. The
        // witness-history check ends its walk at a listed statement
        // (D-522). Entries leave with their scope.
        // Dischargeable.
        ColdHashSet<PodKeyStore<int64_t>> forcedProducts{ &manager, &dirty };
        ColdHashSet<PodKeyStore<int64_t>>::KeysView forcedProductsKeys{ &forcedProducts };
        // Contradiction-scope demand (D-560): every
        // open `contradiction_` scope a negated compound's reductio pushed
        // (`prepareNegatedCompoundContradiction`), keyed by its scope id, with
        // 0 until the demand drain has matched the facts visible there against
        // the rule heads that negate them, 1 after. Leaves with its scope
        // (`wipeSubtrees`). Dischargeable.
        TypedColdMap<int64_t, int32_t> contradictionScopes{ &manager, &dirty };
        TypedColdMap<int64_t, int32_t>::KeysView contradictionScopesKeys{ &contradictionScopes.inner() };
        TypedColdMap<int64_t, int32_t>::ValuesView contradictionScopesValues{ &contradictionScopes.inner() };
        // The demand's derived fact index: the facts those scopes see under
        // the masked keys of the rule heads that negate them. Never deloaded;
        // cleared at the release seam, the canonical reload, discharge and
        // every teardown; the raw image carries it verbatim.
        ContradictionFactIndex contradictionFacts{ &manager };

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
            visit(ContainerTag::UsedAsPremiseCorePairs, usedAsPremiseCorePairsKeys);
            visit(ContainerTag::AppliedPremiseCorePairs, appliedPremiseCorePairsKeys);
            visit(ContainerTag::ClosedOrGoalLevelsKeys, closedOrGoalLevelsKeys);
            visit(ContainerTag::ClosedOrGoalLevelsRunStarts, closedOrGoalLevelsRunStarts);
            visit(ContainerTag::ClosedOrGoalLevelsValues, closedOrGoalLevelsValues);
            orImplications.visitContainers(
                static_cast<uint32_t>(ContainerTag::OrImplicationsBase),
                [&](uint32_t tag, const auto& column) {
                    visit(static_cast<ContainerTag>(tag), column);
                });
            visit(ContainerTag::ExistenceReplaysKeys, existenceReplaysKeys);
            visit(ContainerTag::ExistenceReplaysValues, existenceReplaysValues);
            visit(ContainerTag::CanonicalEquivalenceClassesKeys, canonicalEquivalenceClassesKeys);
            visit(ContainerTag::CanonicalEquivalenceClassesRunStarts, canonicalEquivalenceClassesRunStarts);
            visit(ContainerTag::CanonicalEquivalenceClassesBlobStarts, canonicalEquivalenceClassesBlobStarts);
            visit(ContainerTag::CanonicalEquivalenceClassesBlobPool, canonicalEquivalenceClassesBlobPool);
            visit(ContainerTag::CanonicalMemberRepresentativesKeys, canonicalMemberRepresentativesKeys);
            visit(ContainerTag::CanonicalMemberRepresentativesValues, canonicalMemberRepresentativesValues);
            visit(ContainerTag::CanonicalClassChangesKeys, canonicalClassChangesKeys);
            visit(ContainerTag::CanonicalClassChangesValues, canonicalClassChangesValues);
            unaryComposition.visitContainers(static_cast<uint32_t>(ContainerTag::UnaryCompositionBase),
                [&](uint32_t tag, auto& column) { visit(static_cast<ContainerTag>(tag), column); });

            visit(ContainerTag::OrSupportFactsKeys, orSupportFactsKeys);
            visit(ContainerTag::OrSupportFactsValues, orSupportFactsValues);
            visit(ContainerTag::OrSupportEqualitiesKeys, orSupportEqualitiesKeys);
            visit(ContainerTag::OrSupportEqualitiesRunStarts, orSupportEqualitiesRunStarts);
            visit(ContainerTag::OrSupportEqualitiesValues, orSupportEqualitiesValues);
            unaryComposition.visitGoalContainers(static_cast<uint32_t>(ContainerTag::UnaryCompositionGoalsKeys),
                [&](uint32_t tag, auto& column) { visit(static_cast<ContainerTag>(tag), column); });
            visit(ContainerTag::IntegrationTrialsKeys, integrationTrialsKeys);
            visit(ContainerTag::IntegrationTrialsValues, integrationTrialsValues);
            visit(ContainerTag::IntegrationTrialContentsKeys, integrationTrialContentsKeys);
            visit(ContainerTag::IntegrationTrialContentsValues, integrationTrialContentsValues);
            visit(ContainerTag::InductionEatsKeys, inductionEatsKeys);
            visit(ContainerTag::InductionEatsValues, inductionEatsValues);
            visit(ContainerTag::InductionFirings, inductionFiringsKeys);
            visit(ContainerTag::InductionSettledRoots, inductionSettledRootsKeys);
            visit(ContainerTag::PendingUnconditionalOrdis, pendingUnconditionalOrdisKeys);
            visit(ContainerTag::ForcedOrCohorts, forcedOrCohortsKeys);
            visit(ContainerTag::ForcedProducts, forcedProductsKeys);
            visit(ContainerTag::ContradictionScopesKeys, contradictionScopesKeys);
            visit(ContainerTag::ContradictionScopesValues, contradictionScopesValues);
            visit(ContainerTag::HypothesisOrCohorts, hypothesisOrCohortsKeys);
            visit(ContainerTag::GoalDemandInstalls, goalDemandInstallsKeys);
            visit(ContainerTag::HypothesisShots, hypothesisShotsKeys);
            visit(ContainerTag::IntScopesKeys, intScopesKeys);
            visit(ContainerTag::IntScopesValues, intScopesValues);
            visit(ContainerTag::IntScopeGroupsKeys, intScopeGroupsKeys);
            visit(ContainerTag::IntScopeGroupsValues, intScopeGroupsValues);
            visit(ContainerTag::IntScopeFirstFinishKeys, intScopeFirstFinishKeys);
            visit(ContainerTag::IntScopeFirstFinishValues, intScopeFirstFinishValues);
            visit(ContainerTag::SizeMarkers, sizeMarkersKeys);
            visit(ContainerTag::OneWayWitnesses, oneWayWitnessesKeys);
            visit(ContainerTag::ExperimentOrCohorts, experimentOrCohortsKeys);
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
            orImplicationMatches.clear();
            existenceLookups.clear();
            canonicalUses.clear();
            contradictionFacts.clear();
            visitContainers([](ContainerTag tag, auto& container) {
                if (survivesDischarge(tag)) return;
                container.clear();
            });
        }

        /// @brief Visit every term of `liveBytes` with its container name —
        ///        the per-container physical footprint the memory measurement
        ///        reports.
        ///
        /// @details
        /// The single list of `liveBytes` terms: each call passes the owning
        /// member's name (a string literal) and the bytes that member pins —
        /// its held pages, index pages included, for a paged container, or the
        /// logical estimate `liveBytes` has always used for the few count-based
        /// terms. `liveBytes` sums exactly these calls, so the two can never
        /// drift. Read-only; legal whether or not the arena is resident (every
        /// term reads scalar bookkeeping only).
        ///
        /// @tparam Visitor Callable `(const char* name, int64_t bytes)`.
        /// @param visit The visitor, called once per term in a fixed order.
        template <typename Visitor>
        void visitLiveBytes(Visitor&& visit) const {
            visit("sharedExpressions", peak_snapshot::measure("sharedExpressions", sharedExpressions, [&] { return sharedExpressions.liveBytes(); }));
            visit("intEncodedStatements", static_cast<int64_t>(peak_snapshot::measure("intEncodedStatements", intEncodedStatements, [&] { return intEncodedStatements.liveBytes(); })));
            visit("intLocalEncodedStatements", static_cast<int64_t>(peak_snapshot::measure("intLocalEncodedStatements", intLocalEncodedStatements, [&] { return intLocalEncodedStatements.liveBytes(); })));
            visit("intLocalEncodedStatementsDelta", static_cast<int64_t>(peak_snapshot::measure("intLocalEncodedStatementsDelta", intLocalEncodedStatementsDelta, [&] { return intLocalEncodedStatementsDelta.liveBytes(); })));
            visit("intExternalStatements", static_cast<int64_t>(peak_snapshot::measure("intExternalStatements", intExternalStatements, [&] { return intExternalStatements.liveBytes(); })));
            visit("templateStrings", static_cast<int64_t>(peak_snapshot::measure("templateStrings", templateStrings, [&] { return templateStrings.liveBytes(); })));
            visit("valueStrings", static_cast<int64_t>(peak_snapshot::measure("valueStrings", valueStrings, [&] { return valueStrings.liveBytes(); })));
            visit("originStrings", static_cast<int64_t>(peak_snapshot::measure("originStrings", originStrings, [&] { return originStrings.liveBytes(); })));
            visit("ruleStrings", static_cast<int64_t>(peak_snapshot::measure("ruleStrings", ruleStrings, [&] { return ruleStrings.liveBytes(); })));
            visit("lbStateStrings", static_cast<int64_t>(peak_snapshot::measure("lbStateStrings", lbStateStrings, [&] { return lbStateStrings.liveBytes(); })));
            visit("nameStrings", static_cast<int64_t>(peak_snapshot::measure("nameStrings", nameStrings, [&] { return nameStrings.liveBytes(); })));
            visit("subStrings", static_cast<int64_t>(peak_snapshot::measure("subStrings", subStrings, [&] { return subStrings.liveBytes(); })));
            visit("validityNodes", static_cast<int64_t>(peak_snapshot::measure("validityNodes", validityNodes, [&] { return validityNodes.liveBytes(); })));
            visit("intValidityNamesToFilter", static_cast<int64_t>(peak_snapshot::measure("intValidityNamesToFilter", intValidityNamesToFilter, [&] { return intValidityNamesToFilter.liveBytes(); })));
            visit("intAxedVariables", static_cast<int64_t>(peak_snapshot::measure("intAxedVariables", intAxedVariables, [&] { return intAxedVariables.liveBytes(); })));
            visit("pendingWipeScopes", static_cast<int64_t>(peak_snapshot::measure("pendingWipeScopes", pendingWipeScopes, [&] { return pendingWipeScopes.liveBytes(); })));
            visit("orDisjunctCount", static_cast<int64_t>(peak_snapshot::measure("orDisjunctCount", orDisjunctCount, [&] { return orDisjunctCount.liveBytes(); })));
            visit("integrationStartIntMap", static_cast<int64_t>(peak_snapshot::measure("integrationStartIntMap", integrationStartIntMap, [&] { return integrationStartIntMap.liveBytes(); })));
            visit("intLocalEncodedStatementsSet", static_cast<int64_t>(peak_snapshot::measure("intLocalEncodedStatementsSet", intLocalEncodedStatementsSet, [&] { return intLocalEncodedStatementsSet.liveBytes(); })));
            visit("usedAsPremiseCorePairs", static_cast<int64_t>(peak_snapshot::measure("usedAsPremiseCorePairs", usedAsPremiseCorePairs, [&] { return usedAsPremiseCorePairs.liveBytes(); })));
            visit("appliedPremiseCorePairs", static_cast<int64_t>(peak_snapshot::measure("appliedPremiseCorePairs", appliedPremiseCorePairs, [&] { return appliedPremiseCorePairs.liveBytes(); })));
            visit("intWeakVariables", static_cast<int64_t>(peak_snapshot::measure("intWeakVariables", intWeakVariables, [&] { return intWeakVariables.liveBytes(); })));
            visit("integrationPrepared", static_cast<int64_t>(peak_snapshot::measure("integrationPrepared", integrationPrepared, [&] { return integrationPrepared.liveBytes(); })));
            visit("integrationPreparedMarker", static_cast<int64_t>(peak_snapshot::measure("integrationPreparedMarker", integrationPreparedMarker, [&] { return integrationPreparedMarker.liveBytes(); })));
            visit("expandedImplications", static_cast<int64_t>(peak_snapshot::measure("expandedImplications", expandedImplications, [&] { return expandedImplications.liveBytes(); })));
            visit("compactExpansions", static_cast<int64_t>(peak_snapshot::measure("compactExpansions", compactExpansions, [&] { return compactExpansions.liveBytes(); })));
            visit("expansionCarrierCount", static_cast<int64_t>(peak_snapshot::measure("expansionCarrierCount", expansionCarrierCount, [&] { return expansionCarrierCount.liveBytes(); })));
            visit("intKnownStatements", static_cast<int64_t>(peak_snapshot::measure("intKnownStatements", intKnownStatements, [&] { return intKnownStatements.liveBytes(); })));
            visit("equivalenceClassesMap", static_cast<int64_t>(peak_snapshot::measure("equivalenceClassesMap", equivalenceClassesMap, [&] { return equivalenceClassesMap.liveBytes(); })));
            visit("canonicalEquivalenceClassesMap", static_cast<int64_t>(peak_snapshot::measure("canonicalEquivalenceClassesMap", canonicalEquivalenceClassesMap, [&] { return canonicalEquivalenceClassesMap.liveBytes(); })));
            visit("canonicalMemberRepresentatives", static_cast<int64_t>(peak_snapshot::measure("canonicalMemberRepresentatives", canonicalMemberRepresentatives, [&] { if (peak_snapshot::active()) canonicalMemberRepresentatives.liveBytes(); return canonicalMemberRepresentatives.count(); }) * (sizeof(int64_t) + sizeof(int64_t))));
            visit("canonicalClassChanges", static_cast<int64_t>(peak_snapshot::measure("canonicalClassChanges", canonicalClassChanges, [&] { if (peak_snapshot::active()) canonicalClassChanges.liveBytes(); return canonicalClassChanges.count(); }) * (sizeof(int64_t) + sizeof(int32_t))));
            visit("exprOriginMap", static_cast<int64_t>(peak_snapshot::measure("exprOriginMap", exprOriginMap, [&] { return exprOriginMap.liveBytes(); })));
            visit("sharedNativeKeys", peak_snapshot::measure("sharedNativeKeys", sharedNativeKeys, [&] { return sharedNativeKeys.liveBytes(); }));
            visit("overallHashMemory", static_cast<int64_t>(peak_snapshot::measure("overallHashMemory", overallHashMemory, [&] { return overallHashMemory.liveBytes(); })));
            visit("localHashMemory", static_cast<int64_t>(peak_snapshot::measure("localHashMemory", localHashMemory, [&] { return localHashMemory.liveBytes(); })));
            visit("localHashMemoryDelta", static_cast<int64_t>(peak_snapshot::measure("localHashMemoryDelta", localHashMemoryDelta, [&] { return localHashMemoryDelta.liveBytes(); })));
            visit("workingMemory", static_cast<int64_t>(peak_snapshot::measure("workingMemory", workingMemory, [&] { return workingMemory.liveBytes(); })));
            visit("sameInternalMail", static_cast<int64_t>(peak_snapshot::measure("sameInternalMail", sameInternalMail, [&] { return sameInternalMail.liveBytes(); })));
            visit("nextInternalMail", static_cast<int64_t>(peak_snapshot::measure("nextInternalMail", nextInternalMail, [&] { return nextInternalMail.liveBytes(); })));
            visit("changedClassesThisStep", static_cast<int64_t>(peak_snapshot::measure("changedClassesThisStep", changedClassesThisStep, [&] { return changedClassesThisStep.liveBytes(); })));
            visit("eqClassNameCaches", static_cast<int64_t>(peak_snapshot::measure("eqClassNameCaches", eqClassNameCaches, [&] { return eqClassNameCaches.liveBytes(); })));
            visit("mailOut", static_cast<int64_t>(peak_snapshot::measure("mailOut", mailOut, [&] { return mailOut.liveBytes(); })));
            visit("orPendingBranches", static_cast<int64_t>(peak_snapshot::measure("orPendingBranches", orPendingBranches, [&] { return orPendingBranches.liveBytes(); })));
            visit("orPendingLevels", static_cast<int64_t>(peak_snapshot::measure("orPendingLevels", orPendingLevels, [&] { return orPendingLevels.liveBytes(); })));
            visit("pendingRelayIter", static_cast<int64_t>(peak_snapshot::measure("pendingRelayIter", pendingRelayIter, [&] { return pendingRelayIter.liveBytes(); })));
            visit("pendingRelayLevels", static_cast<int64_t>(peak_snapshot::measure("pendingRelayLevels", pendingRelayLevels, [&] { return pendingRelayLevels.liveBytes(); })));
            visit("orImplications", static_cast<int64_t>(peak_snapshot::measure("orImplications", orImplications, [&] { return orImplications.liveBytes(); })));
            visit("existenceReplays", static_cast<int64_t>(peak_snapshot::measure("existenceReplays", existenceReplays, [&] { if (peak_snapshot::active()) existenceReplays.liveBytes(); return existenceReplays.count(); }) * (sizeof(int64_t) + sizeof(int32_t))));
            visit("orImplicationMatches", static_cast<int64_t>(peak_snapshot::measure("orImplicationMatches", orImplicationMatches, [&] { return orImplicationMatches.liveBytes(); })));
            visit("existenceLookups", static_cast<int64_t>(peak_snapshot::measure("existenceLookups", existenceLookups, [&] { return existenceLookups.liveBytes(); })));
            visit("canonicalUses", static_cast<int64_t>(peak_snapshot::measure("canonicalUses", canonicalUses, [&] { return canonicalUses.liveBytes(); })));
            visit("unaryComposition", static_cast<int64_t>(peak_snapshot::measure("unaryComposition", unaryComposition, [&] { return unaryComposition.liveBytes(); })));
            visit("orSupportFacts", static_cast<int64_t>(peak_snapshot::measure("orSupportFacts", orSupportFacts, [&] { if (peak_snapshot::active()) orSupportFacts.liveBytes(); return orSupportFacts.count(); }) * (sizeof(int64_t) * 2)));
            visit("orSupportEqualities", static_cast<int64_t>(peak_snapshot::measure("orSupportEqualities", orSupportEqualities, [&] { return orSupportEqualities.liveBytes(); })));
            visit("integrationTrials", static_cast<int64_t>(peak_snapshot::measure("integrationTrials", integrationTrials, [&] { return integrationTrials.liveBytes(); })
                + peak_snapshot::measure("integrationTrialContents", integrationTrialContents, [&] { return integrationTrialContents.liveBytes(); })));
            visit("inductionEats", static_cast<int64_t>(peak_snapshot::measure("inductionEats", inductionEats, [&] { return inductionEats.liveBytes(); })));
            visit("inductionFirings", static_cast<int64_t>(peak_snapshot::measure("inductionFirings", inductionFirings, [&] { return inductionFirings.liveBytes(); })));
            visit("inductionSettledRoots", static_cast<int64_t>(peak_snapshot::measure("inductionSettledRoots", inductionSettledRoots, [&] { return inductionSettledRoots.liveBytes(); })));
            visit("pendingUnconditionalOrdis", static_cast<int64_t>(peak_snapshot::measure("pendingUnconditionalOrdis", pendingUnconditionalOrdis, [&] { return pendingUnconditionalOrdis.liveBytes(); })));
            visit("forcedOrCohorts", static_cast<int64_t>(peak_snapshot::measure("forcedOrCohorts", forcedOrCohorts, [&] { return forcedOrCohorts.liveBytes(); })));
            visit("hypothesisOrCohorts", static_cast<int64_t>(peak_snapshot::measure("hypothesisOrCohorts", hypothesisOrCohorts, [&] { return hypothesisOrCohorts.liveBytes(); })));
            visit("goalDemandInstalls", static_cast<int64_t>(peak_snapshot::measure("goalDemandInstalls", goalDemandInstalls, [&] { return goalDemandInstalls.liveBytes(); })));
            visit("hypothesisShots", static_cast<int64_t>(peak_snapshot::measure("hypothesisShots", hypothesisShots, [&] { return hypothesisShots.liveBytes(); })));
            visit("intScopes", static_cast<int64_t>(peak_snapshot::measure("intScopes", intScopes, [&] { return intScopes.liveBytes(); })
                + peak_snapshot::measure("intScopeGroups", intScopeGroups, [&] { return intScopeGroups.liveBytes(); })
                + peak_snapshot::measure("intScopeFirstFinish", intScopeFirstFinish, [&] { return intScopeFirstFinish.liveBytes(); })));
            visit("sizeMarkers", static_cast<int64_t>(peak_snapshot::measure("sizeMarkers", sizeMarkers, [&] { return sizeMarkers.liveBytes(); })));
            visit("oneWayWitnesses", static_cast<int64_t>(peak_snapshot::measure("oneWayWitnesses", oneWayWitnesses, [&] { return oneWayWitnesses.liveBytes(); })));
            visit("experimentOrCohorts", static_cast<int64_t>(peak_snapshot::measure("experimentOrCohorts", experimentOrCohorts, [&] { return experimentOrCohorts.liveBytes(); })));
            visit("forcedProducts", static_cast<int64_t>(peak_snapshot::measure("forcedProducts", forcedProducts, [&] { return forcedProducts.liveBytes(); })));
            visit("contradictionScopes", static_cast<int64_t>(peak_snapshot::measure("contradictionScopes", contradictionScopes, [&] { return contradictionScopes.liveBytes(); })));
            visit("contradictionFacts", static_cast<int64_t>(peak_snapshot::measure("contradictionFacts", contradictionFacts, [&] { return contradictionFacts.liveBytes(); })));
        }

        /// @brief Total live byte footprint across the named container census.
        ///
        /// @details
        /// Sums every term `visitLiveBytes` reports: elements, strings, their
        /// index structures and shared native-key storage counted once.
        /// The total includes reusable record slots and derived indexes; the steward's
        /// separate compaction gate reads page-tier fragmentation.
        ///
        /// @return Live bytes summed over all statified containers.
        int64_t liveBytes() const {
            int64_t total = 0;
            visitLiveBytes([&total](const char*, int64_t bytes) { total += bytes; });
            return total;
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
