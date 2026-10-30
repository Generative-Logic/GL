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
#include "../compiler/compact_expression_syntax.hpp"
#include "../project_paths.hpp"
#include "../infra/hashburst_dump.hpp"
#include "../memory.hpp"
#include "../filter.hpp"
#include "../compilation_state.hpp"
#include "../compiler/expression_parser.hpp"
#include "../compiler/compilation_normalizer.hpp"
#include "../compiler/compilation_persistence.hpp"
#include "../compiler/compiled_definition_registry.hpp"
#include "../inference/hash_key_builder.hpp"
#include "../inference/implication_multiplier.hpp"
#include "../inference/binding_partitions.hpp"
#include "../inference/hash_rule_index.hpp"
#include "../inference/hash_rule_classifier.hpp"
#include "../inference/hash_request_generator.hpp"
#include "../inference/hash_rule_evaluator.hpp"
#include "../inference/burst_sink.hpp"
#include "../logical_block/logic_block_identity.hpp"
#include "../logical_block/statement_knowledge.hpp"
#include "../logical_block/deferred_ancestor_admission.hpp"
#include "../logical_block/update_global_single_chapter_record.hpp"
#include "../logical_block/or_implication_refuter.hpp"
#include "../logical_block/or_implication_selection.hpp"
#include "../logical_block/watch_bridge.hpp"
#include "../logical_block/canonical_form_cache.hpp"
#include "../logical_block/compact_canon_pair.hpp"
#include "../logical_block/numeric_less.hpp"
#include "../logical_block/hypo_scope_parse.hpp"
#include "../logical_block/implication_subproof_outcome.hpp"
#include "../logical_block/collected_record.hpp"
#include "../logical_block/collected_arena.hpp"
#include "../logical_block/new_variable_store.hpp"
#include "../logical_block/witness_meta_store.hpp"
#include "../logical_block/rejection_record.hpp"
#include "../logical_block/rejection_store.hpp"
#include "../logical_block/scanned_token_runs.hpp"
#include "../logical_block/equivalence_span_pair.hpp"
#include "../logical_block/equivalence_rewrite.hpp"
#include "../logical_block/admission_policy.hpp"
#include "../logical_block/scope_type_resolver.hpp"
#include "../logical_block/canonicalizer.hpp"
#include "../logical_block/canonical_template_matcher.hpp"
#include "../logical_block/rejected_source_memory.hpp"
#include "../logical_block/integration_instruction_builder.hpp"
#include "../logical_block/unary_rule_memory.hpp"
#include "../logical_block/admission_marker_builder.hpp"
#include "../compiler/existence_rule_compiler.hpp"
#include "../compiler/implication_builder.hpp"
#include "../compiler/or_rule_compiler.hpp"
#include "../compiler/structural_compiler.hpp"
#include "../compiler/supporting_rule_compiler.hpp"
#include "../compiler/compilation_reuse.hpp"
#include "../compiler/definition_expansion.hpp"
#include "../proof/worker_context.hpp"
#include "../infra/rt_tracker.hpp"
#include "../infra/hashburst_dump.hpp"
#include "../memory_infra/steward.hpp"
#include "../memory_infra/lb_store.hpp"
#include "../mail_log.hpp"
#include "../simple_map_store.hpp"
#include "../memory_infra/or_implication_match_index.hpp"
#include "../memory_infra/temporary_witness_store.hpp"
#include "phase13_timing_slot.hpp"
#include "watch_ledger.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <unordered_set>
#include <utility>
#include <vector>
namespace gl {
class CaseLedger;
class CaseProcessor;
class CompactRuleProcessor;
class EqualityPropagation;
class EquivalenceAdmissionRewriter;
class RejectedSourceReplay;
class WatchProcessor;
namespace tests { struct EquivalenceProcessorFixture; }
/// @brief Equivalence Processor operations for the proof algorithm.
/// @details Uses explicit peer interfaces while the supplied logical blocks retain storage and proof state.
class EquivalenceProcessor {
friend struct tests::EquivalenceProcessorFixture;
public:
/// @brief Bind inputs for equivalence processor.
/// @details Construction stores explicit references; block storage and all invocation lifetimes remain with their owners.
/// @param canonicalTemplates Borrowed CanonicalTemplateMatcher interface or state; must outlive this component.
/// @param canonicalizer Borrowed Canonicalizer interface or state; must outlive this component.
/// @param parameters Borrowed ProverParameters interface or state; must outlive this component.
/// @param rejectedSources Borrowed RejectedSourceMemory interface or state; must outlive this component.
/// @param caseLedger Borrowed CaseLedger interface or state; must outlive this component.
/// @param caseProcessor Borrowed CaseProcessor interface or state; must outlive this component.
/// @param compactRuleProcessor Borrowed CompactRuleProcessor interface or state; must outlive this component.
/// @param equalityPropagation Borrowed EqualityPropagation interface or state; must outlive this component.
/// @param equivalenceAdmissionRewriter Borrowed EquivalenceAdmissionRewriter interface or state; must outlive this component.
/// @param rejectedSourceReplay Borrowed RejectedSourceReplay interface or state; must outlive this component.
/// @param watchProcessor Borrowed WatchProcessor interface or state; must outlive this component.
/// @return None; construction binds the supplied references.
EquivalenceProcessor(CanonicalTemplateMatcher& canonicalTemplates, Canonicalizer& canonicalizer, const ProverParameters& parameters, RejectedSourceMemory& rejectedSources, CaseLedger& caseLedger, CaseProcessor& caseProcessor, CompactRuleProcessor& compactRuleProcessor, EqualityPropagation& equalityPropagation, EquivalenceAdmissionRewriter& equivalenceAdmissionRewriter, RejectedSourceReplay& rejectedSourceReplay, WatchProcessor& watchProcessor);

private:
/// @brief Index the equality dependencies of one canonical identity record.
        /// @details Both its key and expression-bearing values contribute tokens.
        /// The integer kind selects the existing codec without allocating heap values.
        /// @tparam Map Authoritative cold map type.
        /// @param mb Owning block and interners.
        /// @param map Record store.
        /// @param uses Derived dependency store for this record kind.
        /// @param key Stable expression/scope key.
        /// @param kind 0 algebra demand, 1 integration demand, 2 parked existence,
        /// 3 parked OR, 4 OR demand, 5 statement, or 6 eligible compact carrier.
        /// @param arena Caller-owned scratch arena.
        /// @return Nothing.
template<class Map>
        inline void indexCanonicalRecord(Memory& mb, const Map& map, CanonicalUseIndex& uses,
                int64_t key, int kind, ScratchArena& arena);

private:
/// @brief Select only records affected by a complete component or new values.
        /// @details Unchanged member versions and posting revisions skip all record
        /// visits. The result is a logical-key set; consumers impose decoded order.
        /// @tparam Map Authoritative map type.
        /// @tparam ClassT Complete class read interface.
        /// @param mb Owning block with current true classes.
        /// @param map Authoritative record store.
        /// @param uses Dependency index for this consumer.
        /// @param clss Complete class at exactly scope.
        /// @param scope Effective canonical scope.
        /// @param kind Codec selector documented by indexCanonicalRecord.
        /// @param selected Empty result set receiving affected logical keys.
        /// @param arena Caller-owned scratch.
        /// @return Nothing.
template<class Map, class ClassT>
        inline void selectCanonicalRecords(Memory& mb, const Map& map, CanonicalUseIndex& uses,
                const ClassT& clss, NameId scope, int kind, TypedColdSet<int64_t>& selected,
                ScratchArena& arena);

private:
/// @brief Weak-variable reduction, id path — drop members marked
        ///        weak at @p validityName; decoded-lex order preserved.
        ///
        /// @details Probes the packed `Memory::intWeakVariables` twin with
        /// `packStatementKey(memberId, validityId)`. The validity lookup is
        /// non-minting; a validity that was never interned cannot have weak
        /// entries (the weak insert in `updateWeakVariables` interns it at
        /// write), so a lookup miss keeps every member — a defined result,
        /// not a fallback. Input order (decoded-lex, from `memberIds`
        /// storage) is preserved, so downstream first-of-tier selections
        /// stay lex-min.
        ///
        /// @param src          The equivalence class (member read interface).
        /// @param mb           Owning LB — weak-variable set + NameMap.
        /// @param validityName Scope at which weakness is judged.
        /// @param out          Caller buffer receiving the strong member ids in
        ///                     input (decoded-lex `memberIds` storage) order.
        /// @param cap          Capacity of @p out; overflow asserts (Rule 19).
        /// @return The count of strong members written to @p out; 0 when every
        ///         member is weak.
        /// @see reduceEqClass — the string-side sibling; chooseCanonical —
        ///      primary consumer.
template <class ClassT>
        inline int32_t reduceEqClassIds(
            const ClassT& src,
            const Memory& mb,
            StrSpan validityName,
            NameId* out,
            int32_t cap);

private:
/// @brief Fill the keep-last dedup verdicts for every current
        ///        `changedClassesThisStep` entry — one linear pre-pass per
        ///        `applyEquiClasses` Pass-1 round.
        ///
        /// @details
        /// `changedClassesThisStep` receives one snapshot per class CHANGE,
        /// so a class updated N times in one step appears N times — and
        /// every snapshot before the last triggers a redundant full
        /// registry sweep in Pass 1 (the A20 hot burst carried ~11
        /// duplicate snapshots of one collapsed class). Entry k is
        /// superseded when some later entry j > k holds the SAME
        /// (validity id, member-id run): member runs are stored
        /// decoded-lex sorted (I-88), so run equality IS set equality, and
        /// the LAST such snapshot — the surviving one — carries the most
        /// complete pair-level and origin state at apply time.
        ///
        /// The pass walks the delta DESCENDING and interns each entry's
        /// packed `(validity, members)` key (`encodeEqClassKeyFromViewScratch`
        /// — the Pass-2 `deltaIds` key builder) into @p seenKeys: a key
        /// already present means a later entry owns it, so the current
        /// entry's flag is set. One view + one key per entry — O(n) total.
        /// The first form of this dedup compared entry PAIRS (O(n²) views
        /// per round) and ground the incubator's chatty-class steps
        /// (hundreds of same-validity delta entries) ~30–100× slower; the
        /// verdicts are identical.
        ///
        /// Entries pushed mid-round (after this pass ran) carry no flag and
        /// are applied unfiltered — conservative keep-last: a late push is
        /// the newest snapshot, and the NEXT round's pre-pass re-runs over
        /// the full list, superseding anything it outdated.
        ///
        /// @param memoryBlock The LB whose delta buffer is examined.
        /// @param viewScratch Byte-bump arena for view straddle assembly
        ///                    and the packed keys.
        /// @param seenKeys    Fresh per-round key interner (page tier).
        /// @param superseded  Fresh per-round flag column; filled to the
        ///                    current delta size, 1 = skip the entry.
        /// @see D-261 — the decision this implements.
inline void fillDeltaKeepLastFlags(Memory& memoryBlock,
                                           ScratchArena& viewScratch,
                                           ColdHashSet<BytesKeyStore>& seenKeys,
                                           PagedVector<uint8_t>& superseded);

private:
/// @brief Build the statement batch's class table for one
        ///        `applyEquiClasses` call — every class of the LB, its strong
        ///        run, rank, due-from position and delta flag, with the token
        ///        postings.
        ///
        /// @details
        /// Walks the class store in the Pass-2 order (validity ids by
        /// decoded name, then class ordinal — I-84, never id order) and adds
        /// every class. A class whose exact `(validity, members)` key matches
        /// a delta entry is a delta class: it takes that entry's `startIdx` as
        /// `dueFrom` and the entry's index as `rank` (delta order comes
        /// first); a delta entry Pass 1 skipped as dead-scope (`deltaLive`
        /// 0, D-262) is left out of the table entirely, as it is applied
        /// nowhere in this apply. A non-delta class takes its persistent
        /// waterline (`eqClassSttmntIndexMapMap`, 0 when absent) as `dueFrom`
        /// and a rank after every delta entry. The strong run is
        /// `reduceEqClassIds` once per class (the pairwise loop reduced once
        /// per pair). Straddle copies of class blobs and the ordering index
        /// ride @p gArena's byte-bump tier without a rewind — they live for
        /// the apply, like the table itself; the delta-key interner and the
        /// validity list are page-tier locals.
        ///
        /// @param mb        The LB (class store, delta buffer, waterlines read).
        /// @param table     The table to fill (empty on entry).
        /// @param deltaLive Per delta entry: 1 iff Pass 1 applied it this apply.
        /// @param startIdx  Per delta entry: the first registry position it is
        ///                  due for.
        /// @param gArena    The apply's gen-scratch arena.
        /// @invariant `deltaLive.size() == startIdx.size() ==
        ///            changedClassesThisStep.size()` (asserted).
        /// @see EquiClassTable, collectEquiAffectedRows.
void buildEquiClassTable(Memory& mb, EquiClassTable& table,
                                 const PagedVector<uint8_t>& deltaLive,
                                 const PagedVector<int64_t>& startIdx,
                                 ScratchArena& gArena);

private:
/// @brief Collect the registry rows the statement batch must visit
        ///        this round — the rows some table class is due for and can
        ///        touch.
        ///
        /// @details
        /// One pass over `intEncodedStatements` reading only the int columns:
        /// a row is affected when one of its `argId`s is a member of a class
        /// `c` with `s >= c.dueFrom` whose scope is comparable with the row's
        /// (same, strict ancestor, or strict descendant — the pairwise loop's
        /// three-way test). Anchors (`isAnchor`) never rewrite and are
        /// skipped; positive equalities (`nameId` of `=` without negation)
        /// are class inputs, not rewrite inputs, and are skipped; a negated
        /// equality goes to @p negeqRows when a DELTA class is due for it
        /// (the D-226 expander path Pass 1 ran per delta class), every other
        /// affected row to @p rows. Both lists are ascending row positions;
        /// the negated-equality list also records each row's packed
        /// `(originalId, validityId)` key, because the expander's own
        /// `addStatement` path can erase registry rows (goal discharge, park
        /// resets) and shift the positions after this pass ran.
        ///
        /// @param mb        The LB (registry and NameMap read).
        /// @param table     The apply's class table.
        /// @param rows      Out: affected used non-equality rows, ascending.
        /// @param negeqRows Out: affected negated-equality rows, ascending.
        /// @param negeqKeys Out: `packStatementKey` of each @p negeqRows entry.
        /// @param newlyEligible Signed cores requiring all-row coverage after
        ///                      newly used cores; null supplies no overrides.
        /// @param reactivatedScopes Optional output of scopes whose older rows
        ///                          bypassed a class waterline and need cleanup.
        /// @return Nothing; output lists and optional activation scopes are updated.
        /// @invariant Unused rows do not enter class token lookup or rewrite closure.
        /// @see buildEquiClassTable, EquiClassTable::forEachClassContaining.
void collectEquiAffectedRows(const Memory& mb, const EquiClassTable& table,
                                     PagedVector<int32_t>& rows,
                                     PagedVector<int32_t>& negeqRows,
                                     PagedVector<int64_t>& negeqKeys,
                                     const ColdHashSet<PodKeyStore<int64_t>>* newlyEligible = nullptr,
                                     ColdHashSet<PodKeyStore<NameId>>* reactivatedScopes = nullptr) const;

public:
/// @brief One explicit equivalence-class application pass per
        ///        elementary step.
        ///
        /// @details
        /// Equivalence-class application is NOT fired in-place from inside
        /// `addExprToMemoryBlock` or `addStatement` on every membership
        /// change: that would trigger an "application burst" of rewrites,
        /// mailOut emissions, and toBeProved discharges with recursive,
        /// timing-unpredictable control flow — a real-time-budget problem
        /// and a maintenance burden.
        ///
        /// Instead, `applyEquiClasses` is called once per
        /// elementary step, between mail
        /// absorption and request generation. It iterates over the
        /// equivalence classes whose membership changed during the
        /// just-finished mail-absorb pass (delta classes), applies
        /// each against every current registry statement, commits
        /// each new rewrite into the block (intEncodedStatements + the
        /// local deltas + `intStatementLevelsMap` + `intKnownStatements`
        /// + history), pushes main-scope commits to `mailOut`, and
        /// invokes `dischargeToBeProved` per committed expression so
        /// goal closure happens at a single explicit site rather than
        /// recursively inside `addExprToMemoryBlock`.
        ///
        /// In tandem, the flat `addExprToMemoryBlock`
        /// does not trigger an apply burst from inside
        /// `updateEquivalenceClasses` — the class-merge step still
        /// runs there, but the per-statement application loop
        /// lives here. The hashburst rule-firing output (today
        /// `checkLocalEncodedMemoryStatic` → `addExprToMemoryBlock`)
        /// is redirected to `sameIterationInternalMail`, so the next elementary
        /// step's mail-absorb pass picks up the new statements and
        /// this driver re-applies classes against the enriched set.
        ///
        /// @param memoryBlock Owning LB. Mutated in place:
        ///        `intEncodedStatements`,
        ///        `intEncodedStatements`, `localEncoded*` deltas,
        ///        `intStatementLevelsMap`, `intKnownStatements`,
        ///        `exprOriginMap`, `mailOut.statements`,
        ///        `mailOut.exprOriginMap`, and `toBeProved` via
        ///        `dischargeToBeProved`.
        /// @see `applyEquivalenceClass` — per-class helper that this
        ///      driver calls.
        /// @see `dischargeToBeProved` — toBeProved closure called per
        ///      committed expression (introduced alongside this
        ///      function).
        /// @see `updateEquivalenceClasses` — class merger; runs first
        ///      during mail absorption and feeds the delta-class set
        ///      this driver will consume.
        /// @see [`docs/10_pipeline/04_prover.md`](../../docs/10_pipeline/04_prover.md).
        /// @see [`docs/20_core_concepts/05_equivalence_classes.md`](../../docs/20_core_concepts/05_equivalence_classes.md).
        /// @brief The equivalence-class apply under a rule-index staging
        ///        window: the compact hook's canonical expansions stage their
        ///        index writes and the window closes with one flush per
        ///        touched map (D-333); the hook's
        ///        removals flush on their own entry.
        ///
        /// @param memoryBlock The LB.
        /// @return Nothing.
        /// @see `applyEquiClassesCore` — the apply itself;
        ///      `flushRuleIndexStaging`, `Memory::ruleStagingArmDepth`.
inline void applyEquiClasses(Memory& memoryBlock);

private:

inline void applyEquiClassesCore(Memory& memoryBlock);

public:
/// @brief Refresh canonicalization classes after an application-class change.
        ///
        /// @details Rebuilds the changed scope and its initialized descendants
        /// in depth and decoded-name order, excluding refuted scopes through
        /// refutedOrBranchAtOrAbove. The rebuilt runs are held and
        /// written to the class store in one pass (`flushCanonicalRuns`).
        /// All class views settle before any
        /// dependent canonical identity is refreshed. No application-class event
        /// is emitted for inherited copies in this separate store.
        /// Source and watch identities use only changed normal representatives,
        /// including newly registered members; OR caches use those and changed
        /// rewrite localities (change bit 16). Other consumers retain the full change set
        /// for equality evidence, levels, locality and dependency maintenance.
        /// Pending statement arrivals retain the full ancestor-import list so
        /// newly registered negations become visible at the same update seam.
        /// @param mb Owning block.
        /// @param scope Scope whose application classes or metadata changed.
        /// @return Nothing.
        /// @see initializeCanonicalScope, updateEquivalenceClasses.
inline void refreshCanonicalClasses(Memory& mb, NameId scope);

public:

void
            updateEquivalenceClasses(Memory& mb,
                StrSpan eqlty,
                const int* levels, int32_t levelCount,
                const TransientOrigin& origin,
                PagedVector<IntEncodedExpr>& newStatements,
                StrSpan validityName);

public:
// Turn optimizations OFF for just this section
//#pragma optimize("", off)

/// @brief Install an equality `(=[a,b])` into an LB.
///
/// @details
/// - Asserts `args[0] != args[1]` per I-9. A firing assert here means an
///   upstream produced a trivial reflexive equality, which the
///   conjecturer / prover should never emit (per I-8 trivial equality
///   is forbidden in heads).
/// - Searches `memoryBlock.equivalenceClassesMap` for an existing class
///   containing either `args[0]` or `args[1]`; if found, merges the
///   other arg into the class via `mergeTwoEquivalenceClasses` per I-33.
/// - Otherwise creates a fresh
///   `EquivalenceClass{ {args[0], args[1]}, ... }` and seeds
///   `equalityLevelsMap` / `equalityOriginMap`.
/// - Emits the canonical mirror `(=[args[1], args[0]])` as a separate
///   statement so the verifier sees both directions; the per-pair
///   `args[0] != args[1]` assert keeps this from looping.
///
/// @param expr         Equality expression text.
/// @param memoryBlock  Owning LB.
/// @param local        True if the deposit is local-only (no propagation
///                     to the overall hash memory).
/// @param levels       Ascending-unique level run — LB levels at which the
///                     equality is admissible (caller-owned stack/arena
///                     memory, never a cold-column pointer).
/// @param levelCount   The level count (`>= 0`; 0 with null = empty).
/// @param ...          Routing parameters.
/// @invariant I-9 — mirror guard.
/// @invariant I-33 — class-merge cross-vN preconditions.
/// @see `addNegatedEquality` — negated counterpart.
void addEquality(StrSpan expr,
            Memory& memoryBlock,
            bool local,
            const int* levels, int32_t levelCount,
            const TransientOrigin& origin,
            StrSpan validityName);

public:
//#pragma optimize("", off)

/// @brief Install a negated equality `!(=[a,b])` into an LB.
///
/// @details
/// Per I-12, `!(=[a,b])` expands **one-sidedly** via equivalence classes
/// — never as a symmetric cross-product. The expansion picks the
/// deeper-scoped variable as the substitution target so the resulting
/// negated expression sits at the union scope of the original and the
/// class's scope.
///
/// Concrete: if `b` is in a class `C = {b, c}` at scope `S`, then
/// `!(=[a, b])` produces `!(=[a, c])` at the deeper of (current scope,
/// S). It does NOT also produce `!(=[c, a])` — that would be the
/// symmetric counterpart, which the one-sided rule forbids.
///
/// @param expr         Negated equality expression text.
/// @param memoryBlock  Owning LB.
/// @param local        True if the deposit is local-only (no propagation
///                     to the overall hash memory).
/// @param levels       Ascending-unique level run — LB levels at which the
///                     negated equality is admissible (caller-owned).
/// @param levelCount   The level count (`>= 0`; 0 with null = empty).
/// @param ...          Routing parameters.
/// @invariant I-12 — one-sided expansion via equivalence classes; no
///            symmetric cross-product.
/// @see `addEquality` — positive counterpart.
/// @see `applyEquivalenceClassToNegatedEquality` — class-driven sibling
///      rewrite.
void addNegatedEquality(StrSpan expr,
            Memory& memoryBlock,
            bool local,
            const int* levels, int32_t levelCount,
            const TransientOrigin& origin,
            StrSpan validityName);

private:

void updateWeakVariables(Memory& mb, StrSpan validityName, StrSpan eqlty);
private:

CanonicalTemplateMatcher& canonicalTemplates;
Canonicalizer& canonicalizer;
const ProverParameters& parameters;
RejectedSourceMemory& rejectedSources;
CaseLedger& caseLedger;
CaseProcessor& caseProcessor;
CompactRuleProcessor& compactRuleProcessor;
EqualityPropagation& equalityPropagation;
EquivalenceAdmissionRewriter& equivalenceAdmissionRewriter;
RejectedSourceReplay& rejectedSourceReplay;
WatchProcessor& watchProcessor;
};
}
#include "../logical_block/case_ledger.hpp"
#include "../logical_block/case_processor.hpp"
#include "../logical_block/compact_rule_processor.hpp"
#include "../logical_block/equality_propagation.hpp"
#include "../logical_block/equivalence_admission_rewriter.hpp"
#include "../logical_block/rejected_source_replay.hpp"
#include "../logical_block/watch_processor.hpp"
namespace gl {

/// @brief Index the equality dependencies of one canonical identity record.
        /// @details Both its key and expression-bearing values contribute tokens.
        /// The integer kind selects the existing codec without allocating heap values.
        /// @tparam Map Authoritative cold map type.
        /// @param mb Owning block and interners.
        /// @param map Record store.
        /// @param uses Derived dependency store for this record kind.
        /// @param key Stable expression/scope key.
        /// @param kind 0 algebra demand, 1 integration demand, 2 parked existence,
        /// 3 parked OR, 4 OR demand, 5 statement, or 6 eligible compact carrier.
        /// @param arena Caller-owned scratch arena.
        /// @return Nothing.
template<class Map>
        inline void EquivalenceProcessor::indexCanonicalRecord(Memory& mb, const Map& map, CanonicalUseIndex& uses,
                int64_t key, int kind, ScratchArena& arena) {
            const StatementKey owner = Codec<StatementKey>::decode(key);
            if (kind == 5 || kind == 6) {
                const StrSpan original=mb.nameMap.decodeView(owner.orig);
                if(kind==5 || gl::CompactExpressionSyntax::isCompactImplicationSpan(original))
                    canonicalTemplates.indexCanonicalText(uses, key, owner.validity, original);
                return;
            }
            canonicalTemplates.indexCanonicalText(uses, key, owner.validity, mb.templateInterner.decodeView(owner.orig));
            const auto value = [&](int32_t id) {
                canonicalTemplates.indexCanonicalText(uses, key, owner.validity, mb.valueInterner.decodeView(id));
            };
            if constexpr (std::is_same_v<Map, std::remove_reference_t<decltype(mb.intKnownStatements)>>) {
                assert(kind == 5);
            } else {
                if (kind == 1) {
                    ArenaIntegrationMap entries(&arena, &mb.valueInterner);
                    entries.buildFromCold(mb.overallHashMemory.admissionMapIntegration, key);
                    for (int32_t i = 0; i < entries.entries.size(); ++i) {
                        value(entries.entries[i].markedGoalId);
                        if (entries.entries[i].pathVarsId != 0) value(entries.entries[i].pathVarsId);
                    }
                    for (int32_t i = 0; i < entries.les.size(); ++i) value(entries.les[i].sigId);
                    for (int32_t i = 0; i < entries.elemCol.size(); ++i) value(entries.elemCol[i]);
                    for (int32_t i = 0; i < entries.valNodes.size(); ++i) value(entries.valNodes[i].valueId);
                    return;
                }
                const int32_t id = map.lookup(key);
                assert(id != 0);
                for (int32_t r = 0; r < map.runLen(id); ++r) {
                    ScratchScope frame(arena);
                    int32_t length = 0;
                    const char* bytes = map.peekRecordBytes(id, r, length, arena);
                    if (kind == 0) {
                        const AdmissionValueBlobView row(bytes, length);
                        for (int32_t j = 0; j < row.remCount(); ++j) value(row.remId(j));
                        for (int32_t j = 0; j < row.keyCount(); ++j)
                            forEachSubstitutionToken(mb.valueInterner.decodeView(row.keyId(j)), [&](StrSpan token) {
                                bool fixed = startsWithSpan(token, "u_", 2);
                                bool numbered = !token.empty();
                                for (int32_t i = 0; i < token.len; ++i)
                                    if (token[i] < '0' || token[i] > '9') numbered = false;
                                fixed = fixed || (!numbered && !equalSpans(token, StrSpan("marker", 6)));
                                for (int32_t i = 0; i < row.remCount() && !fixed; ++i)
                                    fixed = equalSpans(token, mb.valueInterner.decodeView(row.remId(i)));
                                if (fixed) canonicalTemplates.indexCanonicalText(uses, key, owner.validity, token);
                            });
                    } else if (kind == 2) {
                        const RejectedValueBlobView row(bytes, length);
                        value(row.sourceExpressionId());
                        for (int32_t j = 0; j < row.rewriteCount(); ++j) value(row.rewriteId(j));
                    } else if (kind == 3) {
                        value(RejectedOrdisValueBlobView(bytes, length).orStatementId());
                    } else {
                        assert(kind == 4);
                        canonicalTemplates.indexCanonicalText(uses, key, owner.validity, mb.ruleInterner.decodeView(
                            AdmissionOrdis2ValueBlobView(bytes, length).sourceImplId()));
                    }
                }
            }
        }

/// @brief Select only records affected by a complete component or new values.
        /// @details Unchanged member versions and posting revisions skip all record
        /// visits. The result is a logical-key set; consumers impose decoded order.
        /// @tparam Map Authoritative map type.
        /// @tparam ClassT Complete class read interface.
        /// @param mb Owning block with current true classes.
        /// @param map Authoritative record store.
        /// @param uses Dependency index for this consumer.
        /// @param clss Complete class at exactly scope.
        /// @param scope Effective canonical scope.
        /// @param kind Codec selector documented by indexCanonicalRecord.
        /// @param selected Empty result set receiving affected logical keys.
        /// @param arena Caller-owned scratch.
        /// @return Nothing.
template<class Map, class ClassT>
        inline void EquivalenceProcessor::selectCanonicalRecords(Memory& mb, const Map& map, CanonicalUseIndex& uses,
                const ClassT& clss, NameId scope, int kind, TypedColdSet<int64_t>& selected,
                ScratchArena& arena) {
            {

                uses.synchronize(map, arena, [&](int64_t key) {
                    indexCanonicalRecord(mb, map, uses, key, kind, arena);
                });
            }

            for (int32_t i = 0; i < clss.memberCount(); ++i) {
                const NameId member = clss.memberId(i);
                const int32_t* version = mb.lbMemory.canonicalUses.memberVersions.find(packInt32Pair(member, scope));
                uses.select(scope, mb.nameMap.decodeView(member), version == nullptr ? 0 : *version,
                    [&](int64_t key) { selected.mint(key); });
            }
        }

/// @brief Weak-variable reduction, id path — drop members marked
        ///        weak at @p validityName; decoded-lex order preserved.
        ///
        /// @details Probes the packed `Memory::intWeakVariables` twin with
        /// `packStatementKey(memberId, validityId)`. The validity lookup is
        /// non-minting; a validity that was never interned cannot have weak
        /// entries (the weak insert in `updateWeakVariables` interns it at
        /// write), so a lookup miss keeps every member — a defined result,
        /// not a fallback. Input order (decoded-lex, from `memberIds`
        /// storage) is preserved, so downstream first-of-tier selections
        /// stay lex-min.
        ///
        /// @param src          The equivalence class (member read interface).
        /// @param mb           Owning LB — weak-variable set + NameMap.
        /// @param validityName Scope at which weakness is judged.
        /// @param out          Caller buffer receiving the strong member ids in
        ///                     input (decoded-lex `memberIds` storage) order.
        /// @param cap          Capacity of @p out; overflow asserts (Rule 19).
        /// @return The count of strong members written to @p out; 0 when every
        ///         member is weak.
        /// @see reduceEqClass — the string-side sibling; chooseCanonical —
        ///      primary consumer.
template <class ClassT>
        inline int32_t EquivalenceProcessor::reduceEqClassIds(
            const ClassT& src,
            const Memory& mb,
            StrSpan validityName,
            NameId* out,
            int32_t cap) {
            const NameId valId = mb.nameMap.lookup(validityName);
            const int32_t n = src.memberCount();
            int32_t w = 0;
            for (int32_t i = 0; i < n; ++i) {
                const NameId id = src.memberId(i);
                if (valId != 0
                    && mb.intWeakVariables.contains(packStatementKey(id, valId))) {
                    continue;
                }
                assert(w < cap
                    && "reduceEqClassIds: strong-member run exceeds caller capacity");
                out[w++] = id;
            }
            return w;
        }

/// @brief Fill the keep-last dedup verdicts for every current
        ///        `changedClassesThisStep` entry — one linear pre-pass per
        ///        `applyEquiClasses` Pass-1 round.
        ///
        /// @details
        /// `changedClassesThisStep` receives one snapshot per class CHANGE,
        /// so a class updated N times in one step appears N times — and
        /// every snapshot before the last triggers a redundant full
        /// registry sweep in Pass 1 (the A20 hot burst carried ~11
        /// duplicate snapshots of one collapsed class). Entry k is
        /// superseded when some later entry j > k holds the SAME
        /// (validity id, member-id run): member runs are stored
        /// decoded-lex sorted (I-88), so run equality IS set equality, and
        /// the LAST such snapshot — the surviving one — carries the most
        /// complete pair-level and origin state at apply time.
        ///
        /// The pass walks the delta DESCENDING and interns each entry's
        /// packed `(validity, members)` key (`encodeEqClassKeyFromViewScratch`
        /// — the Pass-2 `deltaIds` key builder) into @p seenKeys: a key
        /// already present means a later entry owns it, so the current
        /// entry's flag is set. One view + one key per entry — O(n) total.
        /// The first form of this dedup compared entry PAIRS (O(n²) views
        /// per round) and ground the incubator's chatty-class steps
        /// (hundreds of same-validity delta entries) ~30–100× slower; the
        /// verdicts are identical.
        ///
        /// Entries pushed mid-round (after this pass ran) carry no flag and
        /// are applied unfiltered — conservative keep-last: a late push is
        /// the newest snapshot, and the NEXT round's pre-pass re-runs over
        /// the full list, superseding anything it outdated.
        ///
        /// @param memoryBlock The LB whose delta buffer is examined.
        /// @param viewScratch Byte-bump arena for view straddle assembly
        ///                    and the packed keys.
        /// @param seenKeys    Fresh per-round key interner (page tier).
        /// @param superseded  Fresh per-round flag column; filled to the
        ///                    current delta size, 1 = skip the entry.
        /// @see D-261 — the decision this implements.
inline void EquivalenceProcessor::fillDeltaKeepLastFlags(Memory& memoryBlock,
                                           ScratchArena& viewScratch,
                                           ColdHashSet<BytesKeyStore>& seenKeys,
                                           PagedVector<uint8_t>& superseded) {
            assert(superseded.size() == 0
                && "keep-last flag column must be freshly constructed");
            const int32_t n = memoryBlock.changedClassesThisStep.size();
            for (int32_t i = 0; i < n; ++i) superseded.push_back(0);
            for (int32_t j = n - 1; j >= 0; --j) {
                const EquivalenceClassView v =
                    memoryBlock.changedClassesThisStep.classViewAt(
                        j, viewScratch);
                const ScratchString key = encodeEqClassKeyFromViewScratch(
                    viewScratch,
                    memoryBlock.changedClassesThisStep.validityAt(j), v);
                if (seenKeys.lookup(StrSpan(key)) != 0) {
                    superseded.setAtRelaxed(j, 1);
                } else {
                    seenKeys.mint(StrSpan(key));
                }
            }
        }

/// @brief Build the statement batch's class table for one
        ///        `applyEquiClasses` call — every class of the LB, its strong
        ///        run, rank, due-from position and delta flag, with the token
        ///        postings.
        ///
        /// @details
        /// Walks the class store in the Pass-2 order (validity ids by
        /// decoded name, then class ordinal — I-84, never id order) and adds
        /// every class. A class whose exact `(validity, members)` key matches
        /// a delta entry is a delta class: it takes that entry's `startIdx` as
        /// `dueFrom` and the entry's index as `rank` (delta order comes
        /// first); a delta entry Pass 1 skipped as dead-scope (`deltaLive`
        /// 0, D-262) is left out of the table entirely, as it is applied
        /// nowhere in this apply. A non-delta class takes its persistent
        /// waterline (`eqClassSttmntIndexMapMap`, 0 when absent) as `dueFrom`
        /// and a rank after every delta entry. The strong run is
        /// `reduceEqClassIds` once per class (the pairwise loop reduced once
        /// per pair). Straddle copies of class blobs and the ordering index
        /// ride @p gArena's byte-bump tier without a rewind — they live for
        /// the apply, like the table itself; the delta-key interner and the
        /// validity list are page-tier locals.
        ///
        /// @param mb        The LB (class store, delta buffer, waterlines read).
        /// @param table     The table to fill (empty on entry).
        /// @param deltaLive Per delta entry: 1 iff Pass 1 applied it this apply.
        /// @param startIdx  Per delta entry: the first registry position it is
        ///                  due for.
        /// @param gArena    The apply's gen-scratch arena.
        /// @invariant `deltaLive.size() == startIdx.size() ==
        ///            changedClassesThisStep.size()` (asserted).
        /// @see EquiClassTable, collectEquiAffectedRows.
inline void EquivalenceProcessor::buildEquiClassTable(Memory& mb, EquiClassTable& table,
                                 const PagedVector<uint8_t>& deltaLive,
                                 const PagedVector<int64_t>& startIdx,
                                 ScratchArena& gArena) {
            const int32_t deltaN = mb.changedClassesThisStep.size();
            assert(deltaLive.size() == deltaN && startIdx.size() == deltaN
                && "buildEquiClassTable: delta columns must cover every delta entry");
            assert(table.classCount() == 0
                && "buildEquiClassTable: the table is built once per apply");
            DirtyState dirty = DirtyState::Clean;

            // Exact delta keys -> the LAST entry carrying that key (an earlier
            // same-key snapshot is superseded by keep-last, D-261).
            ColdHashSet<BytesKeyStore> deltaKeys(&gArena, &dirty);
            PagedVector<int32_t> deltaIdxOfKey(&gArena, &dirty);
            for (int32_t k = 0; k < deltaN; ++k) {
                ScratchScope keyScope(gArena);
                const EquivalenceClassView v =
                    mb.changedClassesThisStep.classViewAt(k, gArena);
                const int32_t id = deltaKeys.mint(StrSpan(encodeEqClassKeyFromViewScratch(
                    gArena, mb.changedClassesThisStep.validityAt(k), v)));
                while (deltaIdxOfKey.size() < id) deltaIdxOfKey.push_back(-1);
                deltaIdxOfKey.setAt(id - 1, k);
            }

            // Validity ids in decoded-name order — the Pass-2 walk order.
            const int32_t vCount = mb.equivalenceClassesMap.count();
            PagedVector<NameId> vIds(&gArena, &dirty);
            for (int32_t kid = 1; kid <= vCount; ++kid)
                vIds.push_back(mb.equivalenceClassesMap.keyAt(kid));
            int32_t* order = (vCount == 0) ? nullptr
                : reinterpret_cast<int32_t*>(gArena.resolve(gArena.alloc(
                      vCount * static_cast<int32_t>(sizeof(int32_t)),
                      static_cast<int32_t>(alignof(int32_t)))));
            for (int32_t i = 0; i < vCount; ++i) order[i] = i;
            std::sort(order, order + vCount, [&](int32_t a, int32_t b) {
                return compareSpans(mb.nameMap.decodeView(vIds[a]),
                                    mb.nameMap.decodeView(vIds[b])) < 0;
            });

            int32_t nonDeltaRank = deltaN;
            for (int32_t oi = 0; oi < vCount; ++oi) {
                const NameId vid = vIds[order[oi]];
                if (mb.phase3IgnoresScope(vid)) continue;
                const int32_t bucketId = mb.equivalenceClassesMap.lookup(vid);
                const int32_t cc = (bucketId == 0) ? 0
                    : mb.equivalenceClassesMap.runLen(bucketId);
                const StrSpan vName = mb.nameMap.decodeView(vid);
                for (int32_t ci = 0; ci < cc; ++ci) {
                    int32_t blen = 0;
                    const char* bp = mb.equivalenceClassesMap.inner().peekBlobContiguous(
                        bucketId, ci, blen, gArena);
                    const EquivalenceClassView view{ bp, blen };
                    int32_t deltaK = -1;
                    {
                        ScratchScope keyScope(gArena);
                        const int32_t id = deltaKeys.lookup(StrSpan(
                            encodeEqClassKeyFromViewScratch(gArena, vid, view)));
                        if (id != 0) deltaK = deltaIdxOfKey[id - 1];
                    }
                    if (deltaK >= 0 && deltaLive[deltaK] == 0) continue;
                    NameId strongBuf[ExecutionParameters::MAX_ARITY
                                     * ExecutionParameters::MAX_KEY_SLOTS];
                    const int32_t strongN = this->reduceEqClassIds(
                        view, mb, vName, strongBuf,
                        ExecutionParameters::MAX_ARITY * ExecutionParameters::MAX_KEY_SLOTS);
                    const int32_t rank = (deltaK >= 0) ? deltaK : nonDeltaRank++;
                    const int64_t dueFrom = (deltaK >= 0)
                        ? startIdx[deltaK]
                        : static_cast<int64_t>(lookupEqClassIndex(
                              mb.eqClassSttmntIndexMapMap, vid, view));
                    table.addClass(vid, view, strongBuf, strongN, rank, dueFrom,
                                   deltaK >= 0);
                }
            }
        }

/// @brief Collect the registry rows the statement batch must visit
        ///        this round — the rows some table class is due for and can
        ///        touch.
        ///
        /// @details
        /// One pass over `intEncodedStatements` reading only the int columns:
        /// a row is affected when one of its `argId`s is a member of a class
        /// `c` with `s >= c.dueFrom` whose scope is comparable with the row's
        /// (same, strict ancestor, or strict descendant — the pairwise loop's
        /// three-way test). Anchors (`isAnchor`) never rewrite and are
        /// skipped; positive equalities (`nameId` of `=` without negation)
        /// are class inputs, not rewrite inputs, and are skipped; a negated
        /// equality goes to @p negeqRows when a DELTA class is due for it
        /// (the D-226 expander path Pass 1 ran per delta class), every other
        /// affected row to @p rows. Both lists are ascending row positions;
        /// the negated-equality list also records each row's packed
        /// `(originalId, validityId)` key, because the expander's own
        /// `addStatement` path can erase registry rows (goal discharge, park
        /// resets) and shift the positions after this pass ran.
        ///
        /// @param mb        The LB (registry and NameMap read).
        /// @param table     The apply's class table.
        /// @param rows      Out: affected used non-equality rows, ascending.
        /// @param negeqRows Out: affected negated-equality rows, ascending.
        /// @param negeqKeys Out: `packStatementKey` of each @p negeqRows entry.
        /// @param newlyEligible Signed cores requiring all-row coverage after
        ///                      newly used cores; null supplies no overrides.
        /// @param reactivatedScopes Optional output of scopes whose older rows
        ///                          bypassed a class waterline and need cleanup.
        /// @return Nothing; output lists and optional activation scopes are updated.
        /// @invariant Unused rows do not enter class token lookup or rewrite closure.
        /// @see buildEquiClassTable, EquiClassTable::forEachClassContaining.
inline void EquivalenceProcessor::collectEquiAffectedRows(const Memory& mb, const EquiClassTable& table,
                                     PagedVector<int32_t>& rows,
                                     PagedVector<int32_t>& negeqRows,
                                     PagedVector<int64_t>& negeqKeys,
                                     const ColdHashSet<PodKeyStore<int64_t>>* newlyEligible,
                                     ColdHashSet<PodKeyStore<NameId>>* reactivatedScopes) const {
            rows.clear();
            negeqRows.clear();
            negeqKeys.clear();
            const NameId eqNameId = mb.nameMap.lookup(StrSpan("=", 1));
            const int32_t n = mb.intEncodedStatements.size();
            for (int32_t s = 0; s < n; ++s) {
                const IntEncodedExpr& row = mb.intEncodedStatements[s];
                if (mb.phase3IgnoresScope(row.validityId)) continue;
                if (row.isAnchor != 0) continue;
                if (!usedAsPremise(mb, row)) continue;
                const bool isEq = (eqNameId != 0 && row.nameId == eqNameId);
                if (isEq && row.negation == 0) continue;
                const bool newCore = newlyEligible != nullptr
                    && newlyEligible->lookup(packInt32Pair(row.nameId, row.negation)) != 0;
                const bool collectActivationScopes = newCore && reactivatedScopes != nullptr;
                bool due = false;
                for (int32_t i = 0; i < row.arity && (!due || collectActivationScopes); ++i) {
                    table.forEachClassContaining(row.argId[i], [&](int32_t c) {
                        if (due && !collectActivationScopes) return;
                        const EquiClassTable::ClassRow& cr = table.rowAt(c);
                        if (isEq && cr.isDelta == 0) return;
                        if (static_cast<int64_t>(s) < cr.dueFrom && !newCore) return;
                        if (cr.vid == row.validityId
                            || mb.nameMap.isStrictAncestor(cr.vid, row.validityId)
                            || mb.nameMap.isStrictAncestor(row.validityId, cr.vid)) {
                            due = true;
                            if (collectActivationScopes && static_cast<int64_t>(s) < cr.dueFrom) {
                                reactivatedScopes->mint(row.validityId);
                                if (mb.nameMap.isStrictAncestor(row.validityId, cr.vid))
                                    reactivatedScopes->mint(cr.vid);
                            }
                        }
                    });
                }
                if (!due) continue;
                if (isEq) {
                    negeqRows.push_back(s);
                    negeqKeys.push_back(packStatementKey(row.originalId, row.validityId));
                } else {
                    rows.push_back(s);
                }
            }
        }

/// @brief One explicit equivalence-class application pass per
        ///        elementary step.
        ///
        /// @details
        /// Equivalence-class application is NOT fired in-place from inside
        /// `addExprToMemoryBlock` or `addStatement` on every membership
        /// change: that would trigger an "application burst" of rewrites,
        /// mailOut emissions, and toBeProved discharges with recursive,
        /// timing-unpredictable control flow — a real-time-budget problem
        /// and a maintenance burden.
        ///
        /// Instead, `applyEquiClasses` is called once per
        /// elementary step, between mail
        /// absorption and request generation. It iterates over the
        /// equivalence classes whose membership changed during the
        /// just-finished mail-absorb pass (delta classes), applies
        /// each against every current registry statement, commits
        /// each new rewrite into the block (intEncodedStatements + the
        /// local deltas + `intStatementLevelsMap` + `intKnownStatements`
        /// + history), pushes main-scope commits to `mailOut`, and
        /// invokes `dischargeToBeProved` per committed expression so
        /// goal closure happens at a single explicit site rather than
        /// recursively inside `addExprToMemoryBlock`.
        ///
        /// In tandem, the flat `addExprToMemoryBlock`
        /// does not trigger an apply burst from inside
        /// `updateEquivalenceClasses` — the class-merge step still
        /// runs there, but the per-statement application loop
        /// lives here. The hashburst rule-firing output (today
        /// `checkLocalEncodedMemoryStatic` → `addExprToMemoryBlock`)
        /// is redirected to `sameIterationInternalMail`, so the next elementary
        /// step's mail-absorb pass picks up the new statements and
        /// this driver re-applies classes against the enriched set.
        ///
        /// @param memoryBlock Owning LB. Mutated in place:
        ///        `intEncodedStatements`,
        ///        `intEncodedStatements`, `localEncoded*` deltas,
        ///        `intStatementLevelsMap`, `intKnownStatements`,
        ///        `exprOriginMap`, `mailOut.statements`,
        ///        `mailOut.exprOriginMap`, and `toBeProved` via
        ///        `dischargeToBeProved`.
        /// @see `applyEquivalenceClass` — per-class helper that this
        ///      driver calls.
        /// @see `dischargeToBeProved` — toBeProved closure called per
        ///      committed expression (introduced alongside this
        ///      function).
        /// @see `updateEquivalenceClasses` — class merger; runs first
        ///      during mail absorption and feeds the delta-class set
        ///      this driver will consume.
        /// @see [`docs/10_pipeline/04_prover.md`](../../docs/10_pipeline/04_prover.md).
        /// @see [`docs/20_core_concepts/05_equivalence_classes.md`](../../docs/20_core_concepts/05_equivalence_classes.md).
        /// @brief The equivalence-class apply under a rule-index staging
        ///        window: the compact hook's canonical expansions stage their
        ///        index writes and the window closes with one flush per
        ///        touched map (D-333); the hook's
        ///        removals flush on their own entry.
        ///
        /// @param memoryBlock The LB.
        /// @return Nothing.
        /// @see `applyEquiClassesCore` — the apply itself;
        ///      `flushRuleIndexStaging`, `Memory::ruleStagingArmDepth`.
inline void EquivalenceProcessor::applyEquiClasses(Memory& memoryBlock) {
            gl::HashRuleIndex::closeRuleIndexStaging(memoryBlock);
            ++memoryBlock.ruleStagingArmDepth;
            applyEquiClassesCore(memoryBlock);
            --memoryBlock.ruleStagingArmDepth;
            if (memoryBlock.ruleStagingArmDepth == 0) {
                RT_SCOPE_HERE("AEC_FLUSH_RULE_STAGING");
                gl::HashRuleIndex::closeRuleIndexStaging(memoryBlock);
            }
        }


inline void EquivalenceProcessor::applyEquiClassesCore(Memory& memoryBlock) {
            if (parameters.skip_eq_classes) {
                return;
            }
            // All three namespace directions (same-NS, ancestor, descendant)
            // and the fixpoint over newly-generated statements live here
            // rather than in `addStatement`.
            //
            // Two passes inside one fixpoint:
            //
            //   PASS 1 — delta classes against ALL intEncodedStatements.
            //     For each class K in `changedClassesThisStep`:
            //       * Re-fire the 4 admission + rejected map helpers
            //         each iteration so late-appearing
            //         admissionMap / rejectedMap entries get covered
            //         too.
            //       * Apply K to every encodedStatement
            //         [startIdx[K], intEncodedStatements.size()) whose
            //         validity is comparable to K's validity.
            //         Initial startIdx[K] = 0 — a class that just
            //         changed gets back-applied to every existing
            //         statement. Equality / negated-equality
            //         statements are SKIPPED — see filter below.
            //       * Advance startIdx[K] to current
            //         intEncodedStatements size after the pass.
            //
            //   PASS 2 — non-delta classes against NEW intEncodedStatements.
            //     For each class C in `equivalenceClassesMap` that is
            //     NOT also in `changedClassesThisStep`:
            //       * Re-fire the same 4 admission + rejected map
            //         helpers — a non-delta class can still acquire
            //         late-arriving admission / rejected entries
            //         within the same fixpoint round.
            //       * Apply C to every encodedStatement
            //         [eqClassSttmntIndexMapMap[v][C.variables],
            //          intEncodedStatements.size()) with the comparability
            //         check. The per-class index in
            //         `eqClassSttmntIndexMapMap` is seeded by
            //         `updateEquivalenceClasses` at class-creation
            //         time to the then-current intEncodedStatements size,
            //         so this loop visits exactly the statements that
            //         joined the LB after the class was registered.
            //       * Update `eqClassSttmntIndexMapMap[v][C.variables]`
            //         to current intEncodedStatements size.
            //
            // Comparability (both passes):
            //   - same-NS:    stmt.validityName == classValidity
            //   - ancestor:   classValidity is a strict ancestor of
            //                  stmt.validityName (rewrite deposits at
            //                  stmt's scope via deeperOf inside
            //                  applyEquivalenceClass)
            //   - descendant: stmt.validityName is a strict ancestor
            //                  of classValidity (D-33; rewrite
            //                  deposits at classValidity via
            //                  deeperOf)
            //   In every direction the `validityName` argument passed
            //   to `applyEquivalenceClass` is the CLASS's validity.
            //
            // Filter: positive equalities are SKIPPED in both passes —
            // consumed by `updateEquivalenceClasses` (class merge).
            // Negated equalities are expanded by
            // `applyEquivalenceClassToNegatedEquality` at TWO moments:
            // inline from `addStatement` on incoming `!(=...)` arrivals
            // (negation-after-class ordering), and from PASS 1 below when
            // a delta class touching either arg formed or grew after the
            // negation's arrival (class-after-negation ordering,
            // D-226). Pass 2 skips them — a NEW
            // statement always had its arrival-time expansion.
            // `applyEquivalenceClass` is for everything else (`in`,
            // `in2`, `in3`, `fXY`, `fXYZ`, operator expressions, …).
            //
            // Pass-2 motivation. Without Pass 2 a class that is
            // REGISTERED (in `equivalenceClassesMap`) but not
            // RECENTLY MODIFIED (so absent from
            // `changedClassesThisStep`) is never back-applied to
            // statements that arrive later in the LB's lifetime.
            // Surfaced by the equi `__contradiction__(=[10,11])` LB
            // where `(=[10,11])` registers class `{10,11}` at LB
            // creation; `(in2[9,10,3])` lands in intEncodedStatements
            // two bursts later via the AnchorIncubator chain
            // disintegration; with delta-only coverage the `10 -> 11`
            // rewrite to `(in2[9,11,3])` never fires and the
            // contradiction chain dies. Pass 2 provides the symmetric
            // coverage without a recursive apply flow.
            //
            // Per-class startIndex tracking lets each pass re-apply
            // classes only to genuinely new intEncodedStatements each
            // iteration instead of re-scanning the entire vector.

            // Per-slot scratch arena for this apply's transients: the per-delta
            // waterlines (`startIdx`), the Pass-2 delta-skip set (`deltaIds`,
            // rebuilt per outer iteration from the changed-class VIEWS keyed by
            // `encodeEqClassKeyFromView`), the Pass-2 validity snapshot, AND the
            // page-straddle class-view buffers below (the `classViewAt` /
            // `peekBlobContiguous` `ScratchArena&` overloads): a straddling class
            // blob is assembled onto this arena's byte-bump tier by pure
            // accumulation — a FRESH `alloc` per straddle, NO per-peek rewind, so
            // a view stays valid across its consumer calls, reclaimed by the
            // per-task `releaseAll`. One slot for the whole call; page-tier
            // containers coexist with the byte-bump sort index / straddle
            // allocations (the I-143 arena rule: byte-bump `alloc` +
            // `PagedVector` coexist, only `allocBytes` must not).
            const unsigned aecSlot = (gl::WorkerContext::coreId() >= 0)
                ? static_cast<unsigned>(gl::WorkerContext::coreId())
                : genScratchArenas().slotCount() - 1;
            ScratchArena& deltaArena = genScratchArenas().forSlot(aecSlot);
            // String-tier arena for the per-class classValidity copy (both
            // passes); a DIFFERENT arena than the NameMap the apply* hooks mint.
            ScratchArena& cvStrArena = scratchArenas().forSlot(aecSlot);
            DirtyState deltaDirty = DirtyState::Clean;

            // Per-delta-class statement-registry waterline — was a heap
            // `std::vector<std::size_t>`, now a page-tier arena vector (an
            // unobservable per-class index, values identical). Held for the whole
            // apply; its pages persist while the per-iteration containers cycle.
            PagedVector<int64_t> startIdx(&deltaArena, &deltaDirty);
            for (int32_t k = 0; k < memoryBlock.changedClassesThisStep.size(); ++k)
                startIdx.push_back(0);

            // OR-convergence for substitution products (user-directed).
            // The per-statement `ordisMerge` in
            // `addExprToMemoryBlock`'s post-`addStatement` loop stays as
            // the input / disintegration-product path; it covers every
            // statement (and not-yet-applied equivalence class) at the
            // moment it is first added. This lambda additionally registers
            // each equi-class REWRITE PRODUCT for OR-convergence, so a
            // branch disjunct created by substitution (e.g. a branch-scope
            // `preorder` rewritten from the branch's equality assumption)
            // reaches `orBookkeeping`. Without it, rewrite-created
            // disjuncts never converge and an OR theorem whose branches
            // are closed by substitution is never proved. Consume products
            // in their existing emission order. Repeated bookkeeping inserts
            // are idempotent; capped history retains the first witness.
            // Id-form OR-convergence buffer for one statement's substitution
            // products (formerly a per-statement
            // `std::vector<ExpressionWithValidity>`). Hoisted once and cleared per
            // statement so the arena pages recycle; the caller fills it via
            // `applyEquivalenceClass`, `mergeProducts` drains it. Its page-tier
            // storage is separate from the merge's byte-bump scratch window.
            PagedVector<IntEncodedExpr> products(&deltaArena, &deltaDirty);
            auto mergeProducts = [&]() {
                    // Preserve the outer class view while reclaiming any
                    // byte-bump scratch allocated by the merge consumers.
                    ScratchScope prodMergeScope(deltaArena);
                    // Each row's (original, validity) is copied into a per-row
                    // string-scratch ScratchString — byte-identical to
                    // nameMap.decode, on the string tier (scratchArenas), disjoint
                    // from the NameMap / lbStateInterner cold tables ordisMerge mints
                    // into, so the StrSpan survives those mints where a raw decodeView
                    // span would dangle (I-3 / I-138).
                    // A per-row ScratchScope on the string arena reclaims the two
                    // copies independently of the gen-scratch merge window.
                    ScratchArena& prodStrArena = scratchArenas().forSlot(aecSlot);
                    for (int32_t r = 0; r < products.size(); ++r) {
                        const IntEncodedExpr& prod = products[r];
                        ScratchScope prodStrScope(prodStrArena);
                        const StrSpan prodOrigView =
                            memoryBlock.nameMap.decodeView(prod.originalId);
                        const ScratchString prodOriginal = ScratchString::copyFrom(
                            prodStrArena, prodOrigView.ptr, prodOrigView.len);
                        const StrSpan prodValidView =
                            memoryBlock.nameMap.decodeView(prod.validityId);
                        const ScratchString prodValidity = ScratchString::copyFrom(
                            prodStrArena, prodValidView.ptr, prodValidView.len);
                        const int32_t plvId = lookupStatementLevels(
                            memoryBlock.intStatementLevelsMap, memoryBlock.nameMap,
                            StrSpan(prodOriginal), StrSpan(prodValidity));
                        assert(plvId
                               && memoryBlock.intStatementLevelsMap.runLen(plvId) > 0
                               && "applyEquiClasses product missing non-empty intStatementLevelsMap entry");
                        int prodLvRun[256];
                        const int32_t prodLvN = coldIntRunAt(
                            memoryBlock.intStatementLevelsMap, plvId,
                            prodLvRun, 256);
                        caseProcessor.ordisMerge(StrSpan(prodOriginal), StrSpan(prodValidity),
                                         prodLvRun, prodLvN,
                                         memoryBlock);
                    }
                };

            // Rejected-map hook bookkeeping (D-307):
            // one validity-bucket index per map per apply, built on first hook
            // use, plus the round skip — a round in which a map's hook dropped
            // nothing left that map without a new hit for every class that ran
            // the hook (the maps only shrink inside an apply: hook drops and
            // admission-hook revivals erase, nothing inserts — I-37), so those
            // classes skip the hook in the next round; delta entries pushed
            // mid-round never ran and still run. The insert epoch is the
            // tripwire for an unenumerated insert path.
            RejectedValidityBuckets rmBuckets(&deltaArena, &deltaDirty);
            RejectedValidityBuckets rmiBuckets(&deltaArena, &deltaDirty);
            // The parked-or hooks read their own per-map posting indexes,
            // refreshed on the map's insert epoch (the park site may insert
            // inside an apply); no round skip for them.
            RejectedValidityBuckets rmoBuckets(&deltaArena, &deltaDirty);
            RejectedValidityBuckets rmo2Buckets(&deltaArena, &deltaDirty);
            int64_t rmDropsPrevRound = -1;
            int64_t rmiDropsPrevRound = -1;
            // The compact hook's per-apply state (D-312):
            // the snapshot index over the compact rows, the staged removals
            // (finished once after the fixpoint), the same round skip.
            RejectedValidityBuckets cbBuckets(&deltaArena, &deltaDirty);
            PagedVector<int64_t> cbKeys(&deltaArena, &deltaDirty);
            ColdHashSet<BytesKeyStore> cbVisited(&deltaArena, &deltaDirty);
            ColdHashSet<PodKeyStore<int64_t>> cbDropSet(&deltaArena, &deltaDirty);
            ColdHashSet<PodKeyStore<int64_t>> cbRemoved(&deltaArena, &deltaDirty);
            TypedColdSet<int64_t> pendingCompacts(&deltaArena, &deltaDirty);
            TypedColdMap<int64_t, uint8_t> pendingIntegrationCompacts(&deltaArena, &deltaDirty);
            cbDropSet.resetToFresh();
            cbRemoved.resetToFresh();
            // The deferred expanded-pair erase context (`DeferredExpandedErase`):
            // armed for the whole apply so every install door — the hooks'
            // canonical expansions and the batch's deposits alike — sees the
            // revive rule; the guard clears the pointer on every exit and
            // asserts that no deferred erase was left unfinished.
            ColdHashSet<PodKeyStore<int64_t>> cbPendingErase(&deltaArena, &deltaDirty);
            ColdHashSet<PodKeyStore<int64_t>> cbRevivedErase(&deltaArena, &deltaDirty);
            cbPendingErase.resetToFresh();
            cbRevivedErase.resetToFresh();
            DeferredExpandedErase deferredErase{ cbPendingErase, cbRevivedErase };
            assert(memoryBlock.deferredExpandedErase == nullptr
                && "applyEquiClasses: a deferred-erase context is already armed (nested apply)");
            memoryBlock.deferredExpandedErase = &deferredErase;
            struct DeferredEraseGuard {
                Memory& m;
                const ColdHashSet<PodKeyStore<int64_t>>& pending;
                ~DeferredEraseGuard() {
                    assert(pending.count() == 0
                        && "applyEquiClasses: deferred expanded-pair erases left unfinished");
                    m.deferredExpandedErase = nullptr;
                }
            } deferredEraseGuard{ memoryBlock, cbPendingErase };
            int64_t cbHitsPrevRound = -1;
            std::size_t deltaCountPrevRound = 0;

            // The statement batch's per-apply state: which delta entries the
            // Pass-1 loop applied (the table excludes dead-scope entries), the
            // class table built once after the first round's hooks, the
            // round's staged products, the closure scratch, the affected-row
            // lists and the committed rows of the drain.
            PagedVector<uint8_t> deltaLive(&deltaArena, &deltaDirty);
            EquiClassTable table(&deltaArena, &deltaDirty);
            bool tableBuilt = false;
            StagedEquiProducts staged(&deltaArena, &deltaDirty);
            PagedVector<int32_t> classStamp(&deltaArena, &deltaDirty);
            int32_t closureStamp = 0;
            PagedVector<int32_t> closureCand(&deltaArena, &deltaDirty);
            PagedVector<int32_t> affectedRows(&deltaArena, &deltaDirty);
            PagedVector<int32_t> negeqRows(&deltaArena, &deltaDirty);
            PagedVector<int64_t> negeqKeys(&deltaArena, &deltaDirty);
            PagedVector<IntEncodedExpr> committed(&deltaArena, &deltaDirty);
            ColdHashSet<PodKeyStore<int64_t>> newlyEligible(&deltaArena, &deltaDirty);
            // A core's first all-row coverage is complete after one round;
            // rows appended afterward follow the ordinary class waterlines.
            ColdHashSet<PodKeyStore<int64_t>> newlyCovered(&deltaArena, &deltaDirty);
            ColdHashSet<PodKeyStore<NameId>> reactivatedScopes(&deltaArena, &deltaDirty);

            while (true) {
                const std::size_t prevSize = memoryBlock.intEncodedStatements.size();
                const bool skipRm = rmDropsPrevRound == 0;
                const bool skipRmi = rmiDropsPrevRound == 0;
                const bool skipCb = cbHitsPrevRound == 0;
                int64_t rmDropsThisRound = 0;
                int64_t rmiDropsThisRound = 0;
                int64_t cbHitsThisRound = 0;
                int64_t canonicalRecordsVisitedThisRound = 0;

                // ============================================================
                // PASS 1 — delta classes × ALL statements.
                // ============================================================
                // Keep-last dedup pre-pass (D-261):
                // one linear descending walk fills a per-round flag column;
                // entries whose (validity, member set) reappears later are
                // skipped below. Entries pushed mid-round have no flag and
                // apply unfiltered (the next round's pre-pass covers them).
                PagedVector<uint8_t> deltaSuperseded(&deltaArena, &deltaDirty);
                {
                    ColdHashSet<BytesKeyStore> deltaKeysSeen(&deltaArena,
                                                             &deltaDirty);
                    this->fillDeltaKeepLastFlags(memoryBlock, deltaArena,
                                                 deltaKeysSeen,
                                                 deltaSuperseded);
                }
                while (deltaLive.size() < memoryBlock.changedClassesThisStep.size())
                    deltaLive.push_back(0);
                while (startIdx.size() < memoryBlock.changedClassesThisStep.size())
                    startIdx.push_back(0);
                for (std::size_t k = 0; k < static_cast<std::size_t>(
                         memoryBlock.changedClassesThisStep.size()); ++k) {
                    // `classValidity` is an owned copy (the apply loop below
                    // mints, I-3). `cls` reads the changed-class blob through a
                    // zero-copy view instead of the heap classAt decode: the view
                    // points into the ChangedClassesBuffer's paged blob pool (or
                    // `deltaArena`'s byte-bump tier on a page straddle), which
                    // stays valid across the consumer calls — PagedVector pages
                    // are address-stable under append and no compaction runs
                    // mid-kernel (phase-1/3 is single-threaded), so a mid-loop
                    // changedClassesThisStep.push (equality formation) does not
                    // move this entry's blob.
                    // classValidity feeds the apply* hooks + applyEquivalenceClass,
                    // which MINT NameMap; copy onto the string-scratch arena (a
                    // DIFFERENT arena) so the span survives those mints (I-3). Held
                    // for the whole per-class iteration (map hooks + statement loop);
                    // nested consumers open their own ScratchScope ABOVE this copy.
                    ScratchScope cvScope(cvStrArena);
                    const NameId classValidityId =
                        memoryBlock.changedClassesThisStep.validityAt(
                            static_cast<int32_t>(k));
                    if (memoryBlock.phase3IgnoresScope(classValidityId)) continue;

                    // Keep-last dedup: a class updated several times this
                    // step appears once per change; every snapshot before
                    // the last with the same (validity, member set) would
                    // re-sweep the registry for rewrites the surviving
                    // snapshot re-emits with at least the same level state
                    // (D-261). Flag column filled by
                    // the pre-pass above; unflagged tail = mid-round push.
                    if (k < static_cast<std::size_t>(deltaSuperseded.size())
                        && deltaSuperseded[static_cast<int32_t>(k)] != 0)
                        continue;

                    // Dead-scope skip: a delta class at or below a refuted
                    // `_ordis_` branch is ex-falso content in a subtree the
                    // end-of-burst drain wipes; back-applying it across the
                    // registry is pure waste
                    // (D-262). The helper stages
                    // the branch so this same burst's drainDeadOrBranches
                    // retires it (I-181).
                    if (caseProcessor.refutedOrBranchAtOrAbove(classValidityId,
                                                       memoryBlock) != 0)
                        continue;

                    const StrSpan cvView =
                        memoryBlock.nameMap.decodeView(classValidityId);
                    const ScratchString classValidity =
                        ScratchString::copyFrom(cvStrArena, cvView.ptr, cvView.len);
                    const EquivalenceClassView cls =
                        memoryBlock.changedClassesThisStep.classViewAt(
                            static_cast<int32_t>(k), deltaArena);

                    // This entry's statement work belongs to the batch below;
                    // the flag admits its class into the table.
                    deltaLive.setAt(static_cast<int32_t>(k), 1);
                }

                // Canonical consumers use each complete view at exactly its own scope.
                // Application deltas above continue to drive only multiplication.
                assert(!memoryBlock.lbMemory.canonicalUses.deferWakes);
                memoryBlock.lbMemory.canonicalUses.deferWakes=true;
                {
                PagedVector<NameId> vIds(&deltaArena, &deltaDirty);
                const int32_t eqKeyCount =
                    memoryBlock.canonicalEquivalenceClassesMap.count();
                for (int32_t kid = 1; kid <= eqKeyCount; ++kid)
                    vIds.push_back(memoryBlock.canonicalEquivalenceClassesMap.keyAt(kid));

                // Iterate the validity ids in decoded-name ascending order via a
                // byte-bump index sorted by `compareSpans(decodeView)` — byte-
                // identical to the former `std::sort` of `(decodedName, id)`
                // pairs (each id has a unique decoded name, so the name is the
                // sole ordering key; `compareSpans` == `std::string::compare`).
                // Never id order (I-84). Byte-bump `alloc` coexists with the
                // page-tier `vIds`.
                ScratchScope vkScope(deltaArena);
                // Block-chunked (ChunkSortedOrdinals — the rmi hook's shape):
                // unique decoded names ⇒ strict total order ⇒ byte-identical
                // sequence to the former single std::sort.
                const auto vkLess = [&](int32_t a, int32_t b) {
                    return compareSpans(
                        memoryBlock.nameMap.decodeView(vIds[static_cast<std::size_t>(a)]),
                        memoryBlock.nameMap.decodeView(vIds[static_cast<std::size_t>(b)])) < 0;
                };
                ChunkSortedOrdinals<decltype(vkLess)> vkOrder(
                    deltaArena, eqKeyCount, 0, vkLess);

                RT_SCOPE_HERE("AEC_PASS2");
                for (int32_t oi = 0; oi < eqKeyCount; ++oi) {
                    // classValidity survives the apply* / applyEquivalenceClass
                    // NameMap mints on the string-scratch arena (I-3); held for the
                    // whole per-validity iteration (all ci classes below); nested
                    // consumers open their own ScratchScope ABOVE this copy.
                    ScratchScope cvScope(cvStrArena);
                    const NameId classValidityId = [&] {

                        return vIds[static_cast<std::size_t>(vkOrder.next())];
                    }();
                    if (memoryBlock.phase3IgnoresScope(classValidityId)) continue;
                    const StrSpan cvView =
                        memoryBlock.nameMap.decodeView(classValidityId);
                    const ScratchString classValidity =
                        ScratchString::copyFrom(cvStrArena, cvView.ptr, cvView.len);
                    // Re-fetch the bucket id each iteration — same mutation
                    // risk as the validity keys above. assignRun keeps a key's
                    // id, so the id is stable across mid-loop bucket replaces.
                    const int32_t bucketId =
                        memoryBlock.canonicalEquivalenceClassesMap.lookup(classValidityId);
                    // Snapshot the class count at the start of this
                    // pass — newly-merged classes mid-iteration are
                    // picked up on the next outer-while round (when
                    // they'll be in `changedClassesThisStep` and Pass
                    // 1 handles them).
                    const std::size_t classCountAtStart = (bucketId == 0)
                        ? 0
                        : static_cast<std::size_t>(
                              memoryBlock.canonicalEquivalenceClassesMap.runLen(bucketId));

                    for (std::size_t ci = 0; ci < classCountAtStart; ++ci) {
                        // Re-decode class ci from the CURRENT cold bucket each
                        // iteration — `applyEquivalenceClass` -> rewrite ->
                        // (eventual) equality formation can replace this
                        // validity's bucket mid-iteration (the former code
                        // re-indexed `map[classValidity][ci]` for the same
                        // reason, copying to dodge invalidated references).
                        int32_t clsLen = 0;
                        const char* clsPtr = nullptr;
                        {

                            clsPtr = memoryBlock.canonicalEquivalenceClassesMap.inner().peekBlobContiguous(
                                bucketId, static_cast<int32_t>(ci), clsLen, deltaArena);
                        }
                        const EquivalenceClassView cls{ clsPtr, clsLen };

                        RT_SCOPE_HERE("CANONICAL_DEPENDENT_RECORDS");
                        auto& uses = memoryBlock.lbMemory.canonicalUses;
                        {

                            TypedColdSet<int64_t> selected(&deltaArena, &deltaDirty);
                            selectCanonicalRecords(memoryBlock, memoryBlock.overallHashMemory.rejectedMapIntegration, uses.integrationSources,
                                cls, classValidityId, 2, selected, deltaArena);
                            if (!selected.empty()) {
                                RT_NOTE_ITERATIONS_HERE(selected.count());
                                canonicalRecordsVisitedThisRound += selected.count();
                                rmiDropsThisRound += equivalenceAdmissionRewriter.applyEquivalenceClassToRejectedMapIntegration(cls, memoryBlock, StrSpan(classValidity), rmiBuckets, &selected);
                            }
                        }
                        {

                            TypedColdSet<int64_t> selected(&deltaArena, &deltaDirty);
                            selectCanonicalRecords(memoryBlock, memoryBlock.overallHashMemory.rejectedMap, uses.algebraSources,
                                cls, classValidityId, 2, selected, deltaArena);
                            if (!selected.empty()) {
                                RT_NOTE_ITERATIONS_HERE(selected.count());
                                canonicalRecordsVisitedThisRound += selected.count();
                                rmDropsThisRound += equivalenceAdmissionRewriter.applyEquivalenceClassToRejectedMap(cls, memoryBlock, StrSpan(classValidity), rmBuckets, &selected);
                            }
                        }
                        {

                            TypedColdSet<int64_t> selected(&deltaArena, &deltaDirty);
                            selectCanonicalRecords(memoryBlock, memoryBlock.overallHashMemory.rejectedMapOrdis, uses.ordisSources,
                                cls, classValidityId, 3, selected, deltaArena);
                            if (!selected.empty()) {
                                RT_NOTE_ITERATIONS_HERE(selected.count());
                                canonicalRecordsVisitedThisRound += selected.count();
                                equivalenceAdmissionRewriter.applyEquivalenceClassToRejectedMapOrdis(cls, memoryBlock, StrSpan(classValidity), rmoBuckets, &selected);
                            }
                        }
                        {

                            TypedColdSet<int64_t> selected(&deltaArena, &deltaDirty);
                            selectCanonicalRecords(memoryBlock, memoryBlock.overallHashMemory.rejectedMapOrdis2, uses.ordis2Sources,
                                cls, classValidityId, 3, selected, deltaArena);
                            if (!selected.empty()) {
                                RT_NOTE_ITERATIONS_HERE(selected.count());
                                canonicalRecordsVisitedThisRound += selected.count();
                                equivalenceAdmissionRewriter.applyEquivalenceClassToRejectedMapOrdis2(cls, memoryBlock, StrSpan(classValidity), rmo2Buckets, &selected);
                            }
                        }
                        {

                            TypedColdSet<int64_t> selected(&deltaArena, &deltaDirty);
                            selectCanonicalRecords(memoryBlock, memoryBlock.overallHashMemory.admissionMapIntegration, uses.integrationDemands,
                                cls, classValidityId, 1, selected, deltaArena);
                            if (!selected.empty()) {
                                RT_NOTE_ITERATIONS_HERE(selected.count());
                                canonicalRecordsVisitedThisRound += selected.count();
                                equivalenceAdmissionRewriter.applyEquivalenceClassToAdmissionMapIntegration(cls, memoryBlock, StrSpan(classValidity), &selected);
                            }
                        }
                        {

                            TypedColdSet<int64_t> selected(&deltaArena, &deltaDirty);
                            selectCanonicalRecords(memoryBlock, memoryBlock.overallHashMemory.admissionMapOrdis2, uses.ordis2Demands,
                                cls, classValidityId, 4, selected, deltaArena);
                            if (!selected.empty()) {
                                RT_NOTE_ITERATIONS_HERE(selected.count());
                                canonicalRecordsVisitedThisRound += selected.count();
                                equivalenceAdmissionRewriter.applyEquivalenceClassToAdmissionMapOrdis2(cls, memoryBlock, StrSpan(classValidity), &selected);
                            }
                        }
                        {

                            TypedColdSet<int64_t> selected(&deltaArena, &deltaDirty);
                            selectCanonicalRecords(memoryBlock, memoryBlock.overallHashMemory.admissionMap, uses.algebraDemands,
                                cls, classValidityId, 0, selected, deltaArena);
                            if (!selected.empty()) {
                                RT_NOTE_ITERATIONS_HERE(selected.count());
                                canonicalRecordsVisitedThisRound += selected.count();
                                equivalenceAdmissionRewriter.applyEquivalenceClassToAdmissionMap(cls, memoryBlock, StrSpan(classValidity), &selected);
                            }
                        }
                        {

                            TypedColdSet<int64_t> selected(&deltaArena, &deltaDirty);
                            selectCanonicalRecords(memoryBlock, memoryBlock.compactExpansions, uses.compacts,
                                cls, classValidityId, 6, selected, deltaArena);
                            if (!selected.empty()) {
                                RT_NOTE_ITERATIONS_HERE(selected.count());
                                canonicalRecordsVisitedThisRound += selected.count();
                                cbHitsThisRound += compactRuleProcessor.applyEquivalenceClassToCompactImplications(cls, memoryBlock, StrSpan(classValidity), cbBuckets, cbKeys, cbVisited, cbDropSet, cbRemoved, pendingCompacts, pendingIntegrationCompacts, &selected);
                            }
                        }

                    }
                }
                }   // complete-scope consumer pass

                if (cbRemoved.count() > 0 || cbDropSet.count() > 0) {
                    gl::CompactRuleProcessor::finishCompactRemovals(memoryBlock, cbDropSet, cbRemoved, deltaArena);
                    cbDropSet.resetToFresh();
                    cbRemoved.resetToFresh();
                }
                compactRuleProcessor.installCanonicalCompacts(memoryBlock,pendingCompacts,pendingIntegrationCompacts);
                rejectedSourceReplay.drainCanonicalWakes(memoryBlock);

                // ============================================================
                // THE STATEMENT BATCH — one statement-major pass replaces the
                // per-class registry sweeps of both passes.
                // ============================================================
                // The class table is built once per apply, after the first
                // round's hook loops: the class state cannot change inside an
                // apply (nothing here admits a positive equality), only the
                // registry grows, and growth is what the due-from positions
                // track.
                gl::HashRuleIndex::closeRuleIndexStaging(memoryBlock);
                newlyEligible.resetToFresh();
                const auto markNewCore = [&](int64_t core) {
                    if (memoryBlock.lbMemory.appliedPremiseCorePairs.lookup(core) == 0
                        && newlyCovered.lookup(core) == 0)
                        newlyEligible.mint(core);
                };
                for (int32_t id = 1; id <= memoryBlock.lbMemory.usedAsPremiseCorePairs.count(); ++id)
                    markNewCore(memoryBlock.lbMemory.usedAsPremiseCorePairs.keyAt(id));
                if (!tableBuilt) {
                    RT_SCOPE_HERE("AEC_BATCH_TABLE");
                    this->buildEquiClassTable(memoryBlock, table, deltaLive,
                                              startIdx, deltaArena);
                    for (int32_t c = 0; c < table.classCount(); ++c)
                        classStamp.push_back(-1);
                    tableBuilt = true;
                }
                {
                    RT_SCOPE_HERE("AEC_BATCH_SCAN");
                    this->collectEquiAffectedRows(memoryBlock, table,
                                                  affectedRows, negeqRows, negeqKeys, &newlyEligible,
                                                  &reactivatedScopes);
                }
                staged.clear();
                {
                    RT_SCOPE_HERE("AEC_BATCH_CLOSURE");
                    for (int32_t i = 0; i < affectedRows.size(); ++i) {
                        if (!usedAsPremise(memoryBlock,
                                memoryBlock.intEncodedStatements[affectedRows[i]])) continue;
                        equalityPropagation.stageEquiClosure(memoryBlock, table, affectedRows[i],
                                               staged, classStamp, closureStamp,
                                               closureCand);
                    }
                }
                {
                    RT_SCOPE_HERE("AEC_BATCH_DRAIN");
                    equalityPropagation.drainStagedEquiProducts(memoryBlock, staged, table,
                                                  committed, deltaArena);
                }
                for (int32_t id = 1; id <= newlyEligible.count(); ++id)
                    newlyCovered.mint(newlyEligible.keyAt(id));
                // Every class has now been applied to every row below the
                // registry size (the committed products are complete
                // closures); rows appended from here on are the next round's.
                const int64_t sizeAfterBatch =
                    static_cast<int64_t>(memoryBlock.intEncodedStatements.size());
                for (int32_t c = 0; c < table.classCount(); ++c)
                    table.setDueFrom(c, sizeAfterBatch);
                for (int32_t k = 0; k < startIdx.size(); ++k)
                    startIdx.setAtRelaxed(k, sizeAfterBatch);

                // Negated equalities a live delta class is due for (D-226 —
                // the rung-2 witness-binding hole): the expander once per row.
                // It walks the negation's own scope, its strict ancestors and
                // its strict descendants and self-dedups via
                // `lookupStatementLevels`, so once per row equals the
                // pairwise once per (class, row); its rows land above
                // `sizeAfterBatch` and drive the next round.
                for (int32_t i = 0; i < negeqRows.size(); ++i) {
                    RT_SCOPE_HERE("AEC_NEGEQ_EXPANSION");
                    // An earlier expander call in this loop can erase registry
                    // rows (its addStatement path discharges goals and resets
                    // parks), shifting the positions collected above: re-locate
                    // the row by its identity key, downward from the recorded
                    // position (compaction only moves rows down); an erased row
                    // is skipped, as the live pairwise scan no longer saw it.
                    int32_t s = negeqRows[i];
                    const int64_t key = negeqKeys[i];
                    const int32_t regN = memoryBlock.intEncodedStatements.size();
                    if (s >= regN) s = regN - 1;
                    while (s >= 0
                           && packStatementKey(memoryBlock.intEncodedStatements[s].originalId,
                                               memoryBlock.intEncodedStatements[s].validityId)
                                  != key) {
                        --s;
                    }
                    if (s < 0) continue;
                    const NameId stmtOriginalId = memoryBlock.intEncodedStatements[s].originalId;
                    const NameId stmtValidityId = memoryBlock.intEncodedStatements[s].validityId;
                    // Owned copies — the expander mints the NameMap, decodeView
                    // rows would dangle (I-3).
                    ScratchScope negRowScope(cvStrArena);
                    const StrSpan stmtOrigView = memoryBlock.nameMap.decodeView(stmtOriginalId);
                    const ScratchString negExpr =
                        ScratchString::copyFrom(cvStrArena, stmtOrigView.ptr, stmtOrigView.len);
                    const StrSpan stmtValidView =
                        memoryBlock.nameMap.decodeView(stmtValidityId);
                    const ScratchString negValidity =
                        ScratchString::copyFrom(cvStrArena, stmtValidView.ptr, stmtValidView.len);
                    const int32_t negLvId =
                        memoryBlock.intStatementLevelsMap.lookup(
                            packStatementKey(stmtOriginalId, stmtValidityId));
                    assert(negLvId != 0
                        && memoryBlock.intStatementLevelsMap.runLen(negLvId) > 0
                        && "statement registry / intStatementLevelsMap invariant violated");
                    int negLvRun[256];
                    const int32_t negLvN = coldIntRunAt(
                        memoryBlock.intStatementLevelsMap, negLvId, negLvRun, 256);
                    assert(canonicalizer.isNegatedEquality(StrSpan(negExpr))
                        && "applyEquiClasses: the negated-equality list holds a non-equality row");
                    products.clear();
                    equalityPropagation.applyEquivalenceClassToNegatedEquality(
                        StrSpan(negExpr), memoryBlock, /*local=*/true,
                        negLvRun, negLvN, products, StrSpan(negValidity));
                    mergeProducts();
                }

                rmDropsPrevRound = rmDropsThisRound;
                rmiDropsPrevRound = rmiDropsThisRound;
                cbHitsPrevRound = cbHitsThisRound;
                deltaCountPrevRound = static_cast<std::size_t>(
                    memoryBlock.changedClassesThisStep.size());

                if (memoryBlock.intEncodedStatements.size() == prevSize && canonicalRecordsVisitedThisRound == 0) {
                    break;
                }
            }

            // Every table class has been applied to every row below the final
            // registry size — the persistent per-class waterlines record
            // exactly that (the pairwise Pass 2 advanced them per visit).
            if (tableBuilt) {
                const int finalSize =
                    static_cast<int>(memoryBlock.intEncodedStatements.size());
                for (int32_t c = 0; c < table.classCount(); ++c) {
                    upsertEqClassIndex(memoryBlock.eqClassSttmntIndexMapMap,
                                       table.rowAt(c).vid, table.viewAt(c),
                                       finalSize);
                }
            }

            // The compact hook's removals finish once per apply: the removed
            // carriers' index entries erased in one pass, every instance's
            // LMV compaction over the staged drop set and its owner-less
            // erasure (the rules a non-canonical compact installed are gone
            // before the next hashburst).
            if (cbRemoved.count() > 0 || cbDropSet.count() > 0) {
                gl::CompactRuleProcessor::finishCompactRemovals(memoryBlock, cbDropSet, cbRemoved, deltaArena);
            }
            // Activation can revisit rows without a new class delta. Clean
            // those scopes only after every rewrite round has completed, in
            // one cleanup pass over them in their first-seen order.
            // Delta scopes keep standardProcessing's ordinary cleanup order.
            {
                DirtyState reactivatedCleanupDirty = DirtyState::Clean;
                ColdHashSet<PodKeyStore<NameId>> reactivatedCleanup(
                    &deltaArena, &reactivatedCleanupDirty);
                reactivatedCleanup.resetToFresh();
                for (int32_t id = 1; id <= reactivatedScopes.count(); ++id) {
                    const NameId validity = reactivatedScopes.keyAt(id);
                    bool deltaScope = false;
                    for (int32_t d = 0; d < memoryBlock.changedClassesThisStep.size(); ++d) {
                        if (memoryBlock.changedClassesThisStep.validityAt(d) == validity) {
                            deltaScope = true;
                            break;
                        }
                    }
                    if (deltaScope || memoryBlock.phase3IgnoresScope(validity)) continue;
                    reactivatedCleanup.mint(validity);
                }
                if (reactivatedCleanup.count() > 0) {
                    RT_SCOPE_HERE("AEC_REACTIVATED_CLEANUP");
                    equalityPropagation.cleanUpExpressions(memoryBlock, reactivatedCleanup);
                }
            }
            auto& applied = memoryBlock.lbMemory.appliedPremiseCorePairs;
            applied.resetToFresh();
            for (int32_t id = 1; id <= memoryBlock.lbMemory.usedAsPremiseCorePairs.count(); ++id)
                applied.mint(memoryBlock.lbMemory.usedAsPremiseCorePairs.keyAt(id));
        }

/// @brief Refresh canonicalization classes after an application-class change.
        ///
        /// @details Rebuilds the changed scope and its initialized descendants
        /// in depth and decoded-name order, excluding refuted scopes through
        /// refutedOrBranchAtOrAbove. The rebuilt runs are held and
        /// written to the class store in one pass (`flushCanonicalRuns`).
        /// All class views settle before any
        /// dependent canonical identity is refreshed. No application-class event
        /// is emitted for inherited copies in this separate store.
        /// Source and watch identities use only changed normal representatives,
        /// including newly registered members; OR caches use those and changed
        /// rewrite localities (change bit 16). Other consumers retain the full change set
        /// for equality evidence, levels, locality and dependency maintenance.
        /// Pending statement arrivals retain the full ancestor-import list so
        /// newly registered negations become visible at the same update seam.
        /// @param mb Owning block.
        /// @param scope Scope whose application classes or metadata changed.
        /// @return Nothing.
        /// @see initializeCanonicalScope, updateEquivalenceClasses.
inline void EquivalenceProcessor::refreshCanonicalClasses(Memory& mb, NameId scope) {
            RT_SCOPE_HERE("CANONICAL_REFRESH_CLASSES");
            if (caseProcessor.refutedOrBranchAtOrAbove(scope, mb) != 0) return;
            canonicalizer.initializeCanonicalScope(mb, scope);
            const unsigned slot = gl::WorkerContext::coreId() >= 0
                ? static_cast<unsigned>(gl::WorkerContext::coreId()) : genScratchArenas().slotCount() - 1;
            ScratchArena& arena = genScratchArenas().forSlot(slot);
            ScratchScope frame(arena);
            const int32_t count = mb.canonicalEquivalenceClassesMap.count();
            auto* scopes = reinterpret_cast<NameId*>(arena.resolve(arena.alloc(count * sizeof(NameId), alignof(NameId))));
            int32_t n = 0;
            for (int32_t i = 1; i <= count; ++i) {
                const NameId candidate = mb.canonicalEquivalenceClassesMap.keyAt(i);
                if ((candidate == scope || mb.nameMap.isStrictAncestor(scope, candidate))
                        && caseProcessor.refutedOrBranchAtOrAbove(candidate, mb) == 0)
                    scopes[n++] = candidate;
            }
            std::sort(scopes, scopes + n, [&](NameId a, NameId b) {
                const int32_t da = mb.nameMap.ancLen(a), db = mb.nameMap.ancLen(b);
                return da != db ? da < db : compareSpans(mb.nameMap.decodeView(a), mb.nameMap.decodeView(b)) < 0;
            });
            DirtyState dirty = DirtyState::Clean;
            PagedVector<CanonicalMemberChange> fresh(&arena, &dirty);
            // Every changed run of this refresh is held and written in one pass;
            // descendants read their parent's new run from the held runs.
            CanonicalRunOverlay held(arena, n);
            for (int32_t i = 0; i < n; ++i) canonicalizer.rebuildCanonicalScope(mb, scopes[i], &fresh, &held);
            canonicalizer.flushCanonicalRuns(mb, held, arena);
            RT_SCOPE_HERE("CANONICAL_DEPENDENT_IDENTITIES");
            RT_NOTE_ITERATIONS_HERE(fresh.size());
            // One ancestor negation is imported into many targets; its
            // canonical form is shared wherever the representatives agree.
            CanonicalFormCache canonicalForms(arena);
            for (int32_t i = 0; i < n; ++i) {
                const NameId target = scopes[i];
                PagedVector<NameId> members(&arena, &dirty);
                PagedVector<NameId> identityMembers(&arena, &dirty);
                PagedVector<NameId> orStaleMembers(&arena, &dirty);
                for (int32_t j = 0; j < fresh.size(); ++j) {
                    if (fresh[j].scope != target) continue;
                    if ((fresh[j].flags & 7) != 0) {
                        members.push_back(fresh[j].member);
                        if ((fresh[j].flags & 1) != 0)
                            identityMembers.push_back(fresh[j].member);
                    }
                    if ((fresh[j].flags & (1 | 16)) != 0)
                        orStaleMembers.push_back(fresh[j].member);
                }
                // An OR entry's canonical form replaces each token of its
                // spellings by the token's normal representative at the entry's
                // scope, and a firing reads the locality of a spelling's rewrite
                // equalities, so only a moved representative (bit 1) or a changed
                // rewrite locality (bit 16) of a token of the canonical form or
                // of a spelling's rewritten token dirties the entry (D-498,
                // D-547).
                const auto markOrCachesStale = [&]() {
                    RT_SCOPE_HERE("CANONICAL_OR_CACHE_INVALIDATION");
                    auto& matches = mb.orImplicationMatches;
                    if (!matches.built) return;
                    for (int32_t j = 0; j < orStaleMembers.size(); ++j)
                        matches.markTokenUsersDirty(orStaleMembers[j], target);
                };
                if (members.empty()) {
                    markOrCachesStale();
                    continue;
                }
                caseLedger.recanonicalizeProcessedOrLedger(mb, target, &members);
                rejectedSources.rewriteExistenceCompletions(mb, target, &members);
                rejectedSources.refreshCanonicalSourceUses(mb, target, identityMembers, arena);
                markOrCachesStale();
                watchProcessor.recanonicalizeWatchLedgers(mb, target, identityMembers, true);
                const auto& importMembers = mb.pendingWatchArrivals.empty() ? identityMembers : members;
                watchProcessor.importAncestorNegationsForClass(mb, target, importMembers, &canonicalForms);
            }
        }


inline void
            EquivalenceProcessor::updateEquivalenceClasses(Memory& mb,
                StrSpan eqlty,
                const int* levels, int32_t levelCount,
                const TransientOrigin& origin,
                PagedVector<IntEncodedExpr>& newStatements,
                StrSpan validityName) {
            const NameId phase3Scope = mb.nameMap.lookup(validityName);
            if (phase3Scope != 0 && mb.phase3IgnoresScope(phase3Scope))
                return;

            if (ftaBlowupAudit) ++ftaBlowupAudit->equalityCalls;
            auto auditClassMark = ftaBlowupAudit ? std::chrono::steady_clock::now()
                : std::chrono::steady_clock::time_point{};
            const auto auditClassStage = [&](int index) {
                if (!ftaBlowupAudit) return;
                const auto now = std::chrono::steady_clock::now();
                ftaBlowupAudit->classStageNs[index] +=
                    std::chrono::duration_cast<std::chrono::nanoseconds>(now - auditClassMark).count();
                auditClassMark = now;
            };

            // A non-derived equality ({-1} statement tier) contributes no
            // derivation levels to class state — the empty-run convention the
            // class machinery already tolerates (max-of-empty INT_MIN
            // fallback). The {-1} tier never enters per-pair class levels.
            if (levelCount == 1 && levels[0] == -1) {
                levels = nullptr;
                levelCount = 0;
            }
            assert((levelCount == 0 || levels[0] >= 0)
                && "class registration level run mixes the non-derived tier with real levels");
            // args_list — arg spans sliced off `eqlty` (no heap arg-string
            // vector); `getArgsSpans` is the proven byte-exact `ce::getArgs` twin.
            StrSpan argSpans[ExecutionParameters::MAX_ARITY];
            const int argN = getArgsSpans(eqlty, argSpans,
                                          ExecutionParameters::MAX_ARITY);
            // eqArgs — sorted-unique arg spans aliasing `eqlty` (no heap
            // std::set<std::string>); byte-lex `compareSpans` order equals the
            // former set's order. The spans point into the caller-stable `eqlty`,
            // never into NameMap bytes, so the encode mints below cannot dangle
            // them (I-3).
            StrSpan sortedArgs[ExecutionParameters::MAX_ARITY];
            int sortedN = 0;
            for (int i = 0; i < argN; ++i) sortedArgs[sortedN++] = argSpans[i];
            std::sort(sortedArgs, sortedArgs + sortedN,
                [](const StrSpan& x, const StrSpan& y) { return compareSpans(x, y) < 0; });
            int uniqN = 0;
            for (int i = 0; i < sortedN; ++i) {
                if (uniqN == 0
                    || compareSpans(sortedArgs[uniqN - 1], sortedArgs[i]) != 0)
                    sortedArgs[uniqN++] = sortedArgs[i];
            }
            // Write site: the class commit below interns the eqArgs anyway,
            // so encode up front — the ids serve the overlap probes, the
            // merge bridge search, and the merged-class member init.
            // eqArgIds is decoded-lex sorted (sortedArgs' order). A fresh
            // (just-minted) id is a member of no existing class, so the overlap
            // probes stay exact.
            NameId eqArgIds[ExecutionParameters::MAX_ARITY];
            int32_t eqArgN = 0;
            for (int i = 0; i < uniqN; ++i)
                eqArgIds[eqArgN++] = mb.nameMap.encode(sortedArgs[i]);

#if 0
            // mirrored "(=[b,a])"
            const std::string mirrored = std::string("(=[") + argsList[1] + "," + argsList[0] + "])";
			const ExpressionWithValidity mirroredEnc(mirrored, validityName);
#endif

            // merged_class init — eqArgIds is decoded-lex sorted (the set's
            // iteration order), exactly the memberIds storage order. The merged
            // class rides a per-slot genScratchArenas MergeClassAccum (no heap
            // EquivalenceClass); its serialize() is the cold-write boundary.
            const unsigned mcSlot = (gl::WorkerContext::coreId() >= 0)
                ? static_cast<unsigned>(gl::WorkerContext::coreId())
                : genScratchArenas().slotCount() - 1;
            ScratchArena& mcArena = genScratchArenas().forSlot(mcSlot);
            MergeClassAccum mergedClass(&mcArena);
            assert(eqArgN > 0 && eqArgN <= 2);
            for (int32_t i = 0; i < eqArgN; ++i) mergedClass.addMember(eqArgIds[i]);
            {
                // levels is already an ascending-unique run; copy it
                // into one int32 run on mcArena's byte-bump tier under a
                // ScratchScope (setLevel copies into the page-tier levelPool).
                const int32_t lvN = levelCount;
                ScratchScope lvScope(mcArena);
                int32_t* run = nullptr;
                if (lvN > 0) {
                    run = reinterpret_cast<int32_t*>(mcArena.resolve(
                        mcArena.alloc(lvN * static_cast<int32_t>(sizeof(int32_t)),
                                      static_cast<int32_t>(alignof(int32_t)))));
                    int32_t w = 0;
                    for (int32_t k = 0; k < levelCount; ++k) run[w++] = levels[k];
                    assert(w == lvN
                        && "updateEquivalenceClasses lv count drift");
                }
                mergedClass.setLevel(
                    packEqPairKey(eqArgIds[0], eqArgIds[eqArgN - 1]), run, lvN);
            }
            if (origin.present) {
                assert(origin.tag != OriginTag::COUNT);
                mergedClass.addOriginEnc(mb.originInterner, eqlty, StrSpan(validityName), origin.tag, origin.deps, origin.depN, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
            }
#if 0
            std::pair<std::string, std::vector<ExpressionWithValidity>> mirroredOrigin;
            mirroredOrigin.first = "symmetry of equality";
            mirroredOrigin.second.push_back(ExpressionWithValidity(eqlty, validityName));
            addOriginEncoded(mergedClass.equalityOriginMap, mb.originInterner, mirroredEnc, mirroredOrigin, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
#endif

            if (parameters.trackHistory && origin.present) {
                assert(origin.tag != OriginTag::COUNT);
                // addOrigin handles the vector push_back, duplicate checking,
                // and respects parameters.compressor_max_origins_per_expr.
                addOriginEncoded(mb.exprOriginMap, mb.originInterner, eqlty, StrSpan(validityName), origin.tag, origin.deps, origin.depN, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
                // No direct mailOut.exprOriginMap write here. Single-writer
                // policy: fillMailOut is the sole writer to mailOut. The
                // eqltyEnc reaches
                // intLocalEncodedStatementsDelta via the addExprToMemoryBlock
                // disintegration-products path; fillMailOut iterates the
                // delta and copies origins from mb.exprOriginMap (populated
                // immediately above) into mailOut.exprOriginMap under its
                // main + allowedForMail gate.
#if 0
                const ExpressionWithValidity mirroredEnc(mirrored, validityName);
                addOriginEncoded(mb.exprOriginMap, mb.originInterner, mirroredEnc, mirroredOrigin, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
#endif
            }

            // A reflexive equality joins only an existing class: when no
            // application class at its scope or a strict ancestor holds the
            // name, it founds no class, because a stored class has two or more
            // members (I-295). Its statement row and the
            // history line above stay.
            if (eqArgN == 1 && !mb.applicationClassHoldsName(validityName, eqArgIds[0])) return;

            auditClassStage(0);
            // Merge all eq-classes that overlap with eq_args
            // Write site — the class store is keyed by validity id; the
            // mint (encode) is allowed and idempotent here.
            const NameId classValidityId = mb.nameMap.encode(validityName);
            // Class watcher: the complete class views are refreshed only when
            // this equality changes a class's membership at its scope. Before
            // any merge, record the size of the one class at this scope that
            // already holds every side (-1 when there is none); after the
            // merges, an unchanged size means an unchanged member set — the
            // merged class always contains that class.
            int32_t priorMembers = -1;
            {
                const int32_t rid = mb.equivalenceClassesMap.lookup(classValidityId);
                const int32_t rl = rid == 0 ? 0 : mb.equivalenceClassesMap.runLen(rid);
                for (int32_t j = 0; j < rl && priorMembers < 0; ++j) {
                    ScratchScope peekScope(mcArena);
                    int32_t blen = 0;
                    const char* bp = mb.equivalenceClassesMap.peekRecordBytes(rid, j, blen, mcArena);
                    const EquivalenceClassView view{ bp, blen };
                    if (!classHasMember(view, eqArgIds[0])) continue;
                    bool holdsAll = true;
                    for (int32_t ei = 1; ei < eqArgN && holdsAll; ++ei)
                        holdsAll = classHasMember(view, eqArgIds[ei]);
                    if (holdsAll) priorMembers = view.memberCount();
                    break;
                }
            }
            // Same-scope classes: walk the cold blob run via `EquivalenceClassView`
            // instead of decoding the whole run to a heap
            // `std::vector<EquivalenceClass>`. Only a RARE overlapping class is
            // materialized (deserialized) for the `const&` merge — exactly as the
            // ancestor loop below does. The non-overlapping KEPT classes keep their
            // raw serialized blob and are spliced into the rewritten run verbatim
            // via the raw byte door (no decode). Pass 1 (here) does the merges +
            // index erases and counts the kept blobs; pass 2 (at the write-back
            // below, after the ancestor merges and the `mergedClass` serialize)
            // copies the kept blobs + the merged blob into one contiguous
            // scratch-arena buffer. The run is not mutated until that final
            // `assignRun`, so the peeked pointers stay valid and pass 2 re-walks
            // the same (still-unmutated) run, re-checking overlap deterministically.
            // A page straddle is assembled once onto mcArena's byte-bump tier
            // (the sanctioned fallback; byte-bump coexists with the page-tier
            // mergedClass on the same arena). Accumulates, reclaimed per task.
            int64_t totalKeptBytes = 0;
            int32_t keptCount = 0;
            {
                const int32_t rid = mb.equivalenceClassesMap.lookup(classValidityId);
                if (rid != 0) {
                    const int32_t rl = mb.equivalenceClassesMap.runLen(rid);
                    for (int32_t j = 0; j < rl; ++j) {
                        int32_t blen = 0;
                        const char* bp = mb.equivalenceClassesMap.peekRecordBytes(
                            rid, j, blen, mcArena);
                        const EquivalenceClassView view{ bp, blen };
                        if (ftaBlowupAudit) ftaBlowupAudit->equalityMembers += view.memberCount();
                        bool overlaps = false;
                        for (int32_t ei = 0; ei < eqArgN; ++ei) {
                            for (int32_t mi = 0; mi < view.memberCount(); ++mi) {
                                if (view.memberId(mi) == eqArgIds[ei]) { overlaps = true; break; }
                            }
                            if (overlaps) break;
                        }
                        if (overlaps) {
                            // classB read through the view (no deserialize); the
                            // run is not mutated until the pass-2 assignRun, so
                            // the peeked view stays valid across the merge.
                            canonicalizer.mergeTwoEquivalenceClasses(mergedClass, view, eqArgIds, eqArgN, levels, levelCount, mb, validityName, validityName);
                            eraseEqClassIndex(mb.eqClassSttmntIndexMapMap, classValidityId, view); // pop(frozenset, None)
                        }
                        else {
                            totalKeptBytes += blen;
                            ++keptCount;
                        }
                    }
                }
            }

            auditClassStage(1);
            // Ancestor-scope merge (cross-NS extension, D-44 + I-31).
            // For each strict ancestor V_a of validityName, absorb every
            // equivalence class at V_a that overlaps eqArgs into mergedClass.
            // The ancestor class itself stays UNCHANGED at V_a — the new
            // equality is invisible to V_a (same-or-deeper visibility rule),
            // so V_a's class cannot be modified. mergeTwoEquivalenceClasses
            // takes classB by const&, so the ancestor class is read-only;
            // no eqClassSttmntIndexMapMap[ancestorV] erase, no
            // equivalenceClassesMap[ancestorV] write. Mirrors the additive
            // principle established by D-33 (apply-side cross-scope) and
            // D-43 (rmi additive rewrite).
            {
                // Read ancestor classes through `EquivalenceClassView` over the
                // cold blob run instead of decoding the whole scope to a heap
                // `std::vector<EquivalenceClass>`. The overlap test scans the
                // view's member ids (byte-identical to the heap `memberIds`
                // scan); only the RARE overlapping ancestor is materialized to a
                // heap class for the merge (which reads classB by const&). The
                // common non-overlapping majority stay view-only — no heap
                // decode. Ancestor classes are read-only here (D-44 / I-31:
                // never re-serialized), so no scope's run is mutated inside this
                // loop and the peeked pointers stay valid across it.
                // Straddle fallback rides mcArena's byte-bump tier (accumulate,
                // no rewind); ancestor runs are read-only here so nothing moves.
                // strictAncestorSpans is the zero-heap twin of
                // strictAncestorNames (same compareSpans lex order, tie-free —
                // I-140). This loop is mint-free on
                // NameMap: mergeTwoEquivalenceClasses only nameMap.lookups and
                // mints originInterner (a DIFFERENT interner) while reading
                // equivalenceClassesMap (ancestor runs read-only, D-44/I-31), so
                // the ancestor NAME spans (aliasing NameMap) stay valid across
                // the whole loop — no copy.
                StrSpan ancSpans[ExecutionParameters::MAX_SCOPE_DEPTH];
                const int32_t ancN = mb.nameMap.strictAncestorSpans(
                    validityName, ancSpans, ExecutionParameters::MAX_SCOPE_DEPTH);
                for (int32_t ai = 0; ai < ancN; ++ai) {
                    const StrSpan ancestorV = ancSpans[ai];
                    const NameId ancVid = mb.nameMap.lookup(ancestorV);
                    if (ancVid == 0) continue;
                    const int32_t rid = mb.equivalenceClassesMap.lookup(ancVid);
                    if (rid == 0) continue;
                    const int32_t rl = mb.equivalenceClassesMap.runLen(rid);
                    for (int32_t j = 0; j < rl; ++j) {
                        int32_t blen = 0;
                        const char* bp = mb.equivalenceClassesMap.peekRecordBytes(
                            rid, j, blen, mcArena);
                        const EquivalenceClassView ancView{ bp, blen };
                        bool overlapsAnc = false;
                        for (int32_t ei = 0; ei < eqArgN; ++ei) {
                            for (int32_t mi = 0; mi < ancView.memberCount(); ++mi) {
                                if (ancView.memberId(mi) == eqArgIds[ei]) { overlapsAnc = true; break; }
                            }
                            if (overlapsAnc) break;
                        }
                        if (!overlapsAnc) continue;
                        // Overlapping ancestor: merge through the view (no
                        // deserialize; ancestor runs are read-only here — D-44 /
                        // I-31 — so nothing mutates them across the loop).
                        canonicalizer.mergeTwoEquivalenceClasses(mergedClass, ancView, eqArgIds, eqArgN, levels, levelCount, mb, validityName, ancestorV);
                    }
                }
            }

            auditClassStage(2);
            // Same-scope fixpoint (D-322). The
            // ancestor pass can grow mergedClass with members that a KEPT
            // same-scope class also holds — a class that absorbed the same
            // ancestor class before it grew. Classes at one scope must stay
            // pairwise disjoint (I-218), so every
            // kept class overlapping the grown mergedClass is merged as well
            // (bridge = the shared member, no equality levels) until no kept
            // class overlaps. Ancestor classes stay untouched (I-31). The
            // flags exclude the merged classes from the write-back below.
            DirtyState fpDirty = DirtyState::Clean;
            PagedVector<uint8_t> mergedFlags(&mcArena, &fpDirty);
            {
                const int32_t rid = mb.equivalenceClassesMap.lookup(classValidityId);
                const int32_t rl = rid ? mb.equivalenceClassesMap.runLen(rid) : 0;
                for (int32_t j = 0; j < rl; ++j) mergedFlags.push_back(0);
                bool grew = true;
                while (grew) {
                    grew = false;
                    for (int32_t j = 0; j < rl; ++j) {
                        if (mergedFlags[j] != 0) continue;
                        int32_t blen = 0;
                        const char* bp = mb.equivalenceClassesMap.peekRecordBytes(
                            rid, j, blen, mcArena);
                        const EquivalenceClassView view{ bp, blen };
                        bool overlapsEq = false;
                        for (int32_t ei = 0; ei < eqArgN && !overlapsEq; ++ei)
                            for (int32_t mi = 0; mi < view.memberCount(); ++mi)
                                if (view.memberId(mi) == eqArgIds[ei]) { overlapsEq = true; break; }
                        if (overlapsEq) continue;   // merged in pass 1
                        bool overlapsMerged = false;
                        for (int32_t mi = 0; mi < view.memberCount() && !overlapsMerged; ++mi)
                            for (int32_t ai = 0; ai < mergedClass.members.size(); ++ai)
                                if (mergedClass.members[ai] == view.memberId(mi)) { overlapsMerged = true; break; }
                        if (!overlapsMerged) continue;
                        canonicalizer.mergeTwoEquivalenceClasses(mergedClass, view, eqArgIds, eqArgN,
                                                         nullptr, 0, mb, validityName, validityName);
                        eraseEqClassIndex(mb.eqClassSttmntIndexMapMap, classValidityId, view);
                        mergedFlags.setAt(j, 1);
                        totalKeptBytes -= blen;
                        --keptCount;
                        grew = true;
                    }
                }
            }

            auditClassStage(3);
            const bool membershipChanged =
                priorMembers < 0 || mergedClass.members.size() != priorMembers;
            upsertEqClassIndex(mb.eqClassSttmntIndexMapMap, classValidityId, mergedClass, 0);
            // Write-back via the raw byte door (pass 2): the kept blobs verbatim
            // in original run order, then the serialized `mergedClass` last —
            // byte-identical to the former `assignClassesById([kept..., merged])`
            // (a kept blob equals re-serializing its decoded class; the codec is a
            // lossless pure function of content, I-103). Staging rides the per-slot
            // `genScratchArenas` byte-bump tier — no heap run buffer. The kept
            // blobs are copied out BEFORE the `assignRun` that replaces the run, so
            // the peeked pointers never dangle.
            {
                assert(mergedClass.members.size() >= 2
                    && "updateEquivalenceClasses: a stored equivalence class has two or more members");
                const int32_t mergedSize = mergedClass.serializedSize();
                const int32_t total = static_cast<int32_t>(
                    totalKeptBytes + static_cast<int64_t>(mergedSize));
                const int32_t blobCount = keptCount + 1;
                const unsigned slot = (gl::WorkerContext::coreId() >= 0)
                    ? static_cast<unsigned>(gl::WorkerContext::coreId())
                    : genScratchArenas().slotCount() - 1;
                ScratchArena& spliceArena = genScratchArenas().forSlot(slot);
                ScratchScope spliceScope(spliceArena);
                // Per-blob staging + the emitter run door (assignRunGenerated):
                // each kept blob rides its OWN byte-bump allocation (single-blob
                // ≤ one block, the pre-existing peek bound), so the run total
                // may exceed one pool block; the engine's contract makes the
                // CSR bytes identical to the former one-buffer concat +
                // assignRun. Copies are taken BEFORE the run-replacing write,
                // so the peeked pointers never dangle.
                StrSpan* runBlobs = reinterpret_cast<StrSpan*>(spliceArena.resolve(
                    spliceArena.alloc(
                        blobCount * static_cast<int32_t>(sizeof(StrSpan)),
                        static_cast<int32_t>(alignof(StrSpan)))));
                int64_t off = 0;
                int32_t k = 0;
                const int32_t rid = mb.equivalenceClassesMap.lookup(classValidityId);
                if (rid != 0) {
                    const int32_t rl = mb.equivalenceClassesMap.runLen(rid);
                    for (int32_t j = 0; j < rl; ++j) {
                        int32_t blen = 0;
                        const char* bp = mb.equivalenceClassesMap.peekRecordBytes(
                            rid, j, blen, mcArena);
                        const EquivalenceClassView view{ bp, blen };
                        bool overlaps = false;
                        for (int32_t ei = 0; ei < eqArgN; ++ei) {
                            for (int32_t mi = 0; mi < view.memberCount(); ++mi) {
                                if (view.memberId(mi) == eqArgIds[ei]) { overlaps = true; break; }
                            }
                            if (overlaps) break;
                        }
                        assert(j < mergedFlags.size());
                        if (overlaps || mergedFlags[j] != 0) continue;   // merged in pass 1 or the fixpoint pass — not kept
                        char* copy = spliceArena.resolve(
                            spliceArena.alloc(blen > 0 ? blen : 1, 1));
                        std::memcpy(copy, bp, static_cast<std::size_t>(blen));
                        runBlobs[k++] = StrSpan(copy, blen);
                        off += blen;
                    }
                }
                assert(k == keptCount);
                char* mergedBuf = spliceArena.resolve(
                    spliceArena.alloc(mergedSize, 1));
                const int32_t mw = mergedClass.serializeInto(mergedBuf);
                assert(mw == mergedSize);
                runBlobs[k++] = StrSpan(mergedBuf, mergedSize);
                off += mergedSize;
                assert(off == total && k == blobCount);
                mb.equivalenceClassesMap.inner().assignRunGenerated(
                    classValidityId, blobCount, total,
                    [&](auto&& sink) {
                        for (int32_t j = 0; j < blobCount; ++j)
                            sink(runBlobs[j].ptr, runBlobs[j].len);
                    });
            }

            auditClassStage(4);
            // Record the freshly-committed merged class in the per-step
            // delta tracker so `applyEquiClasses` re-applies it against
            // every current `encodedStatement` between mail-absorb-clear
            // and request generation. Stored as a value copy (validity
            // + class) rather than an index, because subsequent merges
            // at the same scope can shrink the underlying
            // `equivalenceClassesMap[validityName]` vector and
            // invalidate any cached index.
            {
                const int32_t bsize = mergedClass.serializedSize();
                const unsigned bslot = (gl::WorkerContext::coreId() >= 0)
                    ? static_cast<unsigned>(gl::WorkerContext::coreId())
                    : genScratchArenas().slotCount() - 1;
                ScratchArena& bArena = genScratchArenas().forSlot(bslot);
                ScratchScope bScope(bArena);
                char* const buf = bArena.resolve(bArena.alloc(bsize, 1));
                const int32_t bw = mergedClass.serializeInto(buf);
                assert(bw == bsize);
                mb.changedClassesThisStep.push(classValidityId, buf, bsize);
            }
            auditClassStage(5);
            if (mb.orImplicationMatches.built) {
                for (int32_t member = 0; member < mergedClass.members.size(); ++member)
                    mb.orImplications.pendingClasses.push_back(
                        packInt32Pair(mergedClass.members[member], classValidityId));
            }

            auditClassStage(6);
            // updateEquivalenceClasses has no post-merge apply loop of
            // its own — that would be a double-apply. The merged class
            // is recorded in `Memory::changedClassesThisStep` two lines
            // above, and `applyEquiClasses` re-applies the same class
            // against every comparable encodedStatement once mail-absorb
            // finishes. Omitting the in-line loop eliminates the
            // double work AND removes the latent assert risk that
            // would surface if a future wipe drained
            // equivalenceClassesMap between mail-absorb and
            // applyEquiClasses.
            upsertEqClassIndex(mb.eqClassSttmntIndexMapMap, classValidityId,
                mergedClass,
                static_cast<int>(mb.intEncodedStatements.size()));

            // The expression-side cleanup (cleanUpExpressions) must NOT run
            // HERE, inline in updateEquivalenceClasses, which executes
            // before `standardProcessing` advances to its applyEquiClasses
            // step. That order would drop non-canonical
            // intEncodedStatements rows BEFORE the rewriter could read them
            // and emit canonical replacements, so equi-class application
            // would miss every row that existed only in the non-canonical
            // form (the in3[10,rec,it_0_lev_3_8,4] → in3[10,rec,it_0_lev_3_32,4]
            // case for the right-cancellation theorem). The cleanup runs
            // once per (per-elementary-step) `applyEquiClasses`, after the
            // rewriter's fixpoint has completed — see the Step-4b block
            // right after `applyEquiClasses(memoryBlock)` in
            // `standardProcessing` below.
			updateWeakVariables(mb, validityName, eqlty);


            auditClassStage(7);
            // expr_origin_map = merged_class.equality_origin_map | mb.expr_origin_map
            // Per key: the body's existing lines first, verbatim (never dropped —
            // I-216), then the class lines deduped
            // and capped. mb.exprOriginMap is the cold blob
            // map, so this runs IN PLACE: each class key's run is rebuilt, body-only
            // keys keep their content untouched (they were already merged[key] =
            // bodyLines in the heap form). Class keys are processed in SORTED order
            // so a class-only key is minted into the cold map deterministically
            // (canonical, cross-host-stable deload — the heap twin relied on the
            // dump's key-sort and did not need this). Pure id-form — class and body
            // share the LB's origin space, so no re-encode happens here. The
            // per-key run rebuild is the 0% heap serialize+memcmp raw-door
            // exprOriginRunReplace (no heap newRun / recordsAt snapshot); its
            // ScratchScope nests ABOVE forEachOriginSorted's mcArena okeys/idx.
            {
                const int cap = parameters.compressor_mode
                    ? parameters.compressor_max_origins_per_expr
                    : parameters.max_origin_per_expr;
                mergedClass.forEachOriginSorted(
                    [&](int64_t key, auto classLines) {
                        exprOriginRunReplace(mb.exprOriginMap, key, classLines,
                                             cap, mcArena);
                    });
            }
            auditClassStage(8);
            // An equality between members already in one class leaves every
            // complete view's membership, representative and weak status as it
            // was; its pair level and origin stay in this scope's own class and
            // reach the complete views at their next rebuild.
            if (membershipChanged) {
                refreshCanonicalClasses(mb, classValidityId);
            }

        }
}
