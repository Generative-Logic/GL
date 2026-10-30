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
#include "../logical_block/statement_registry_operations.hpp"
#include "../logical_block/equivalence_queries.hpp"
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
class EquivalenceQueries;
class ProofHistory;
namespace tests { struct CaseProcessorFixture; }
/// @brief Case Processor operations for the proof algorithm.
/// @details Uses explicit peer interfaces while the supplied logical blocks retain storage and proof state.
class CaseProcessor {
friend struct tests::CaseProcessorFixture;
public:
/// @brief Bind inputs for case processor.
/// @details Construction stores explicit references; block storage and all invocation lifetimes remain with their owners.
/// @param admissionPolicy Borrowed AdmissionPolicy interface or state; must outlive this component.
/// @param anchorInfo Borrowed ce::AnchorInfo interface or state; must outlive this component.
/// @param canonicalTemplates Borrowed CanonicalTemplateMatcher interface or state; must outlive this component.
/// @param canonicalizer Borrowed Canonicalizer interface or state; must outlive this component.
/// @param definitions Borrowed CompiledDefinitionRegistry interface or state; must outlive this component.
/// @param parameters Borrowed ProverParameters interface or state; must outlive this component.
/// @param caseLedger Borrowed CaseLedger interface or state; must outlive this component.
/// @param equivalenceQueries Borrowed EquivalenceQueries interface or state; must outlive this component.
/// @param proofHistory Borrowed ProofHistory interface or state; must outlive this component.
/// @return None; construction binds the supplied references.
CaseProcessor(AdmissionPolicy& admissionPolicy, const ce::AnchorInfo& anchorInfo, CanonicalTemplateMatcher& canonicalTemplates, Canonicalizer& canonicalizer, const CompiledDefinitionRegistry& definitions, const ProverParameters& parameters, CaseLedger& caseLedger, EquivalenceQueries& equivalenceQueries, ProofHistory& proofHistory);

private:
/// @brief Whether one OR disjunct already holds at a scope, so that a
        ///        case split on its OR could only reproduce that scope
        ///        (D-438).
        ///
        /// @details
        /// True when the disjunct is known at @p scopeId or an ancestor —
        /// literally, as a reflexive equality, or up to the complete classes
        /// of @p scopeId (`elementKnownAtOrAbove`) — or when it is a positive
        /// conjunction (an `and`-category compact, named or numbered) whose
        /// flattened elements all hold by the same test, e.g. the literally
        /// true `(and27[m,m,p,p])` of two reflexive equalities. A negated
        /// disjunct holds only through its own known row. The branch of a
        /// disjunct that already holds equals its parent scope, so
        /// convergence through that branch can only conclude what the parent
        /// scope already proves; `consumeOrLeavesCohort` therefore neither
        /// opens nor parks an OR with such a disjunct.
        ///
        /// @param clean   Clean (u_-stripped) disjunct text on caller-owned
        ///                storage.
        /// @param scopeId Minted scope of the OR statement.
        /// @param mb      Owning block, its classes and known-statement store.
        /// @param depth   Conjunction nesting depth of this call (0 at the
        ///                top; bounded by `MAX_INSTRUCTION_ELEMENTS`).
        /// @return True iff the disjunct holds at @p scopeId or an ancestor.
        /// @invariant Creates no proof row and mints nothing in the NameMap;
        ///            only `elementKnownAtOrAbove`'s arena-backed derived
        ///            comparison indexes are synchronized.
        /// @see elementKnownAtOrAbove, consumeOrLeavesCohort, orDisjunctTier.
bool orDisjunctTrueAt(StrSpan clean, NameId scopeId, Memory& mb,
                              int32_t depth = 0);

public:
/// @brief Return the refuted `_ordis_` branch scope at or above a
        ///        validity scope, staging it for retirement — or 0 when the
        ///        chain holds no refuted branch.
        ///
        /// @details
        /// Walks the validity ancestor chain from @p validityId to the root.
        /// At each node two checks decide:
        ///
        /// 1. **Already staged** — a node present in
        ///    `Memory::pendingDeadOrBranches` is a branch already proven
        ///    dead this burst; it is returned immediately, no re-probe.
        /// 2. **Refutation probe** — at an `_ordis_` DISINTEGRATION scope
        ///    (`classifyOrScopeView`), the branch's asserted disjunct is
        ///    negated (a leading `!` sliced off, otherwise a `!`-prefixed
        ///    form built on the string-scratch arena) and probed `known` at
        ///    the branch scope or any ancestor — the exact staging predicate
        ///    `ordisMerge` uses (D-242): a scope's fact set is monotone and
        ///    the `known` record immortal (I-58), so a once-derived
        ///    refutation stays a refutation. On a hit the branch is minted
        ///    into `pendingDeadOrBranches` so the SAME burst's
        ///    `drainDeadOrBranches` retires it, and its id is returned. A
        ///    branch already retired (its scope is in
        ///    `intValidityNamesToFilter`) is returned the same way but not
        ///    staged again — the `ordisMerge` probe skips it alike — so each
        ///    branch retires once and its cohort count drops once (I-172).
        ///
        /// Every probe is non-minting (`NameMap::lookup`,
        /// `intKnownStatements` reads, `parentOf` metadata walks); the ONLY
        /// write is the `pendingDeadOrBranches` staging mint — the drain's
        /// own inbox, whose record step re-runs this predicate and asserts a
        /// hit, so a staged id always satisfies it.
        ///
        /// Consumed by `applyEquiClasses` Pass 1: a delta class whose
        /// validity sits at or below a refuted branch is ex-falso content in
        /// a subtree the drain wipes this same burst, so back-applying it
        /// across the registry is pure waste (the A20 runtime explosion —
        /// D-262,
        /// I-181).
        ///
        /// @param validityId  The scope whose ancestor chain is examined.
        /// @param memoryBlock The LB owning the scope forest and registries.
        /// @return The dead branch's validity id, or 0 when no `_ordis_`
        ///         branch at or above @p validityId is refuted.
        /// @invariant Single-threaded (phase-1/3) — writes only the LB-local
        ///            `pendingDeadOrBranches`.
        /// @see ordisMerge — the staging peer on the deposit path.
        /// @see drainDeadOrBranches — the consumer of the staged id.
inline NameId refutedOrBranchAtOrAbove(NameId validityId,
                                               Memory& memoryBlock);

public:
/// @brief Stage the sequenced-release goal probe (I-174) for a
        ///        deposit REFUSED by the ancestor-known dedup gate.
        ///
        /// @details
        /// The or-cohort sequenced release is deposit-driven: `ordisMerge`
        /// mints `pendingOrReleases` when a branch deposit's expression
        /// matches an open `toBeProved` goal of the innermost goal scope on the
        /// branch's parent chain (D-549).
        /// An ancestor-known refusal (D-273)
        /// suppresses the deposit — but the branch still DERIVED the
        /// expression, so the resolution signal must survive, or a live
        /// branch whose goal-matching derivation happens to be
        /// ancestor-known (the goal sitting shallower than the knowing
        /// scope) would never release its cohort's next branch — an I-174
        /// stall that loses theorems. This helper runs exactly
        /// `ordisMerge`'s probe on the refused (expression, scope) pair.
        ///
        /// Entirely non-minting: a scope that is not an `_ordis_` branch,
        /// or whose parent / signature / packed cohort key was never
        /// interned, means no cohort ever opened here — a defined miss
        /// (I-83), never a fallback. The only mutation is the
        /// `pendingOrReleases` mint itself (the intended signal; the
        /// pod-set mint dedups repeats).
        ///
        /// @param memoryBlock Owning LB.
        /// @param origId      Minted NameMap id of the refused expression.
        /// @param valId       Minted NameMap id of the refused deposit
        ///                    scope.
        /// @invariant [I-174](../../docs/agentic_swdd/30_invariants.md#i-174)
        ///            — sequenced release; the probe mirrors `ordisMerge`'s
        ///            byte-for-byte (pending-row gate, dedup gate, innermost
        ///            goal scope of the parent chain).
        /// @see ordisMerge — the deposit-driven twin of this probe;
        ///      drainPendingOrReleases — the end-of-burst consumer.
inline void stageOrReleaseForRefusedDeposit(Memory& memoryBlock,
                                                    NameId origId,
                                                    NameId valId);

private:
/// @brief Count the argument tokens of an MPL expression — the
        ///        structural-complexity measure of the sequenced
        ///        or-disintegration ranking.
        ///
        /// @details
        /// Fires `collectExprTokens`'s flat `[...]` bracket scan with a
        /// counting sink: every comma-separated token inside every bracket
        /// group contributes one, so a nested compound counts the tokens of
        /// all its levels. A bare equality `(=[a,b])` counts 2; an
        /// `(existence3[x,y,z])` counts 3; a De-Morgan compound counts every
        /// nested argument. Zero-allocation, pure function of the bytes.
        ///
        /// @param expr Span over one disjunct in clean (u_-stripped) form.
        /// @return The token count (>= 0).
        /// @see collectExprTokens, pickTopOrDisjunct.
static inline int32_t orDisjunctTokenCount(StrSpan expr);

private:
/// @brief Rank tier of one or-disjunct for the sequenced release
        ///        order.
        ///
        /// @details
        /// Tier 0 — everything that is not a positive equality (compounds,
        /// existence forms, negations): released first, most complex first.
        /// Tier 1 — a positive equality `(=[a,b])` with no anchor-slot
        /// argument. Tier 2 — a positive equality with at least one
        /// anchor-slot argument (e.g. `(=[m,2])` where `2` is the anchor's
        /// zero slot): dead last — an assumed equality against an anchor
        /// constant feeds the equivalence-class machinery hardest, so it
        /// opens only after every other branch resolved. A NEGATED equality
        /// is tier 0: assuming `!(=[a,b])` expands one-sidedly (I-12) and
        /// carries none of the class-merge pathology.
        ///
        /// A positive conjunction (an `and`-category compact, named or
        /// numbered) whose flattened elements are all positive equalities
        /// assumes equalities just as a bare equality does (e.g. `and27` =
        /// two equalities), so it takes the highest tier among its elements:
        /// tier 1, or tier 2 when an element carries an anchor-slot argument
        /// (D-437). One non-equality element keeps the
        /// whole conjunction in tier 0; a negated conjunction is tier 0.
        ///
        /// @param clean      Span over the disjunct in clean form.
        /// @param anchorArgs The LB's anchor argument spans
        ///                   (`collectAnchorArgs`); may be null when
        ///                   `anchorArgN` is 0.
        /// @param anchorArgN Number of anchor argument spans.
        /// @param depth      Conjunction nesting depth of this call (0 at the
        ///                   top; bounded by `MAX_INSTRUCTION_ELEMENTS`).
        /// @return 0, 1, or 2 (lower releases earlier).
        /// @invariant Mints nothing; the element instances ride a
        ///            `ScratchScope` on the string-scratch arena.
        /// @see pickTopOrDisjunct, collectAnchorArgs.
inline int32_t orDisjunctTier(StrSpan clean,
                                      const StrSpan* anchorArgs,
                                      int32_t anchorArgN,
                                      int32_t depth = 0);

private:
/// @brief Pick the disjunct the sequenced or-disintegration releases
        ///        next — the argmin under the deterministic rank order.
        ///
        /// @details
        /// Rank order (best first): ascending tier (`orDisjunctTier` — non-
        /// equalities before bare equalities before anchor-argument
        /// equalities; a positive conjunction of equalities ranks as an
        /// equality), then DESCENDING token count (most structurally
        /// complicated first), then ascending byte-lex (`compareSpans`) as
        /// the tie-breaker. A pure function of the disjunct bytes plus the
        /// anchor argument set — the same inputs at cohort mint time and at
        /// every later release, so the release sequence is deterministic
        /// (I-84) and never stored.
        ///
        /// @param disjuncts  `n` spans over the candidate disjuncts in clean
        ///                   form.
        /// @param n          Candidate count (>= 1, asserted).
        /// @param anchorArgs The LB's anchor argument spans.
        /// @param anchorArgN Number of anchor argument spans.
        /// @return The index (0-based) of the top-ranked disjunct.
        /// @see orDisjunctTier, orDisjunctTokenCount, collectAnchorArgs.
inline int32_t pickTopOrDisjunct(const StrSpan* disjuncts, int32_t n,
                                         const StrSpan* anchorArgs,
                                         int32_t anchorArgN);

private:
/// @brief Collect the LB's anchor arguments — the ranking's
        ///        anchor-slot set.
        ///
        /// @details
        /// Walks the `parentMemory` chain from the given LB to the root and
        /// returns the argument spans of the first exprKey of the form
        /// `(<anchor name>[...])` — the batch's anchor element (the
        /// `createAuxyImplication` untouchables idiom). The spans alias the
        /// process-wide never-deloaded `skeletonInterner` cold pages (frozen
        /// through the parallel phase, D-186), so they survive every NameMap
        /// / per-LB interner mint. An LB whose chain carries no anchor
        /// element (or a batch without an anchor) yields 0 — a defined
        /// empty set, which simply makes ranking tier 2 unreachable.
        ///
        /// @param memoryBlock The LB whose chain is walked.
        /// @param out         Caller stack array receiving the spans.
        /// @param cap         Capacity of `out` (anchor arity <= MAX_ARITY).
        /// @return The number of anchor argument spans written.
        /// @see orDisjunctTier, pickTopOrDisjunct.
inline int32_t collectAnchorArgs(const Memory& memoryBlock,
                                         StrSpan* out, int32_t cap);

public:
/// @brief Record an OR-branch contribution and promote a complete cohort.
        ///
        /// @details
        /// Tracks each expression in its exact parent/signature cohort, retains
        /// branch derivation history, and promotes a conclusion when every live
        /// disjunct contributes. During mail absorption, the caller supplies the
        /// opposite output channel so the input clear cannot discard the new
        /// parent statement or its convergence history. Other callers use the
        /// ordinary same-iteration destination. Refutation routing is unchanged.
        ///
        /// @param addExpression The derived branch expression.
        /// @param effectiveValidity Its exact branch scope.
        /// @param addExpressionLevels Caller-owned ascending derivation levels.
        /// @param addExpressionLevelCount Number of derivation levels.
        /// @param memoryBlock The owning logical block.
        /// @param convergenceMailOut Optional output for convergence and branch
        ///        refutation during absorption;
        ///        nullptr selects the same-iteration channel outside absorption.
        /// @param rowRemovals Optional batch that records the converged
        ///        expression's branch-row removals for the caller to apply
        ///        after its drain; nullptr removes the rows immediately.
        /// @return Nothing; records contributions and emits complete conclusions.
        /// @invariant Contributions never mix cohort parents or OR signatures.
        /// @see standardProcessing, addExprToMemoryBlock, drainDeadOrBranches,
        ///      StatementRegistryOperations::RowRemovalBatch.
inline void ordisMerge(StrSpan addExpression,
                               StrSpan effectiveValidity,
                               const int* addExpressionLevels,
                               int32_t addExpressionLevelCount,
                               Memory& memoryBlock,
                               ColdMail* convergenceMailOut = nullptr,
                               StatementRegistryOperations::RowRemovalBatch* rowRemovals = nullptr);

public:
/// @brief Count statements beneath one exact parent/signature OR cohort or
///        OR-integration family.
///
/// @details Counts every statement whose scope is a branch of the cohort or a
/// case of the family, `<parent>_boundary_ordis_<signature>_(...)` or
/// `<parent>_boundary_orint_<signature>_(...)` by @p kind, or a scope nested
/// below one. Equal-looking signatures under other parents and goal-qualified
/// main families (`<goal>_subproof_orint_...`) never match the prefix. The
/// scan is read-only and uses no interner mint.
/// @param body Owning resident block.
/// @param parent Parent validity text.
/// @param signature Original OR signature.
/// @param kind `Disintegration` for an `_ordis_` cohort, `Integration` for an
///        `_orint_` family; `NotOrScope` asserts.
/// @return Statement count of the branches or cases and their nested scopes.
/// @see retireDuplicateOrCohorts; retireDuplicateOrIntegrationFamilies;
///      retireDuplicateOrSources.
int32_t orCohortStatementCount(const Memory& body, StrSpan parent,
            StrSpan signature, OrScopeKind kind) const;

public:
/// @brief End-of-burst drain of the dead OR-branch inbox — retire every
///        `_ordis_` branch whose asserted disjunct is refuted, shrink
///        its cohort's bookkeeping, and re-check convergence at the
///        reduced disjunct count.
///
/// @details
/// `ordisMerge`'s dead-branch probe stages a branch scope's NameMap vid
/// on `Memory::pendingDeadOrBranches` when the branch's asserted
/// disjunct has its negation `known` at the branch scope or one of its
/// ancestors — a refuted assumption makes every fact the branch
/// derives ex falso, so the branch only burns runtime and memory and
/// blocks its cohort's convergence forever. This drain — running
/// under the owning worker claim at standard-processing completion, after
/// `drainDisprovedGoals` (so cohorts retired wholesale by a goal
/// disproof are gone first) and immediately BEFORE the
/// `pendingWipeScopes` drain so the branch wipes land in the same
/// call — retires each staged branch:
///
/// 1. the branch scope is queued on `pendingWipeScopes` and inserted
///    into `intValidityNamesToFilter`, so the radical subtree wipe
///    removes its statements and no future deposit revives it (a
///    filtered scope also never re-stages — the `ordisMerge` probe and
///    `refutedOrBranchAtOrAbove` both skip it — and the drain asserts that
///    a staged branch is not one its cohort already retired, I-172);
/// 2. the cohort's `orDisjunctCount` drops by the number of its dead
///    branches; ZERO survivors retires the cohort wholesale — the
///    count row leaves the container and every bookkeeping run of the
///    cohort empties in step 3 (a fully-refuted disjunction implies a
///    joint contradiction at the disjunction's scope, which the
///    parent-scope K mutual-exclusion rules and the contradiction
///    machinery surface: primed reductio discharge; the
///    vacuous-premise flag plus the global-theorem reversion for a
///    normal LB at main; and for a cohort under a deeper
///    hypothetical scope of a normal LB, nothing at the LB level —
///    the refuted disjunction only proves that scope's assumption
///    unsatisfiable, the enclosing clause is vacuously true, and the
///    LB and its theorems stay valid);
/// 3. every `orBookkeeping` row of the cohort loses the dead branches'
///    disjunct entries (a run holding only dead entries leaves the
///    container) — required for soundness: a stale dead entry would
///    let the reduced count fire a convergence no surviving branch
///    ever derived;
/// 4. convergence is re-checked: a surviving row whose run already
///    reaches the reduced count re-fires through `ordisMerge` (the
///    canonical promotion path — parent-scope emission via
///    `sameIterationInternalMail`, `or convergence` history citing the
///    surviving branches, per-branch collapse of the converged copy).
///    A row whose survivor statement is gone from its branch scope is
///    a defined skip — that expression converged earlier and already
///    lives at the parent.
///
/// A staged branch whose cohort has no `orDisjunctCount` row is a
/// defined skip: the cohort was retired wholesale by this same seam's
/// `drainDisprovedGoals` (its parent scope lies in a disproof-wiped
/// subtree, which covers the branch scope too).
///
/// Branch-subtree `exprOriginMap` history is KEPT (I-44 preserve rule):
/// earlier full-cohort convergences cite dead-branch derivations as
/// `or convergence` ingredients, and the chapter walker must still
/// resolve them.
///
/// @param memoryBlock The LB whose inbox is drained (its containers
///                    are resident — the drain runs inside the LB's
///                    own burst).
/// @invariant Deterministic: staged vids process in decoded-name
///            byte-lex order; every erase predicate and the rebuilt
///            run contents are pure functions of container content.
/// @see Memory::pendingDeadOrBranches, ordisMerge, mintOrCohortId,
///      Memory::orBookkeeping, Memory::orDisjunctCount.
void drainDeadOrBranches(Memory& memoryBlock);

public:
/// @brief End-of-burst retirement of duplicate or cohorts — when a
///        class merge makes two OPEN cohorts at one parent equi
///        variants of each other, keep the most advanced and retire
///        the rest (the or-uniqueness gate's merge-seam companion).
///
/// @details
/// The deposit-door gate stops NEW duplicates; cohorts opened
/// BEFORE their signatures became class-equal both keep running.
/// This drain walks the `processedOrLedger` (a snapshot — the
/// clears below restructure it): per scope with two or more
/// recorded ors, each spelling is canonicalized under the scope's
/// complete class bucket (`canonicalFormAtScope`, the gate's own test)
/// and grouped; a group holding two or more OPEN cohorts (an
/// `orDisjunctCount` row via the non-minting `lbStateInterner`
/// recovery) is a duplicate case split. The KEEPER is the cohort
/// with the most statements already registered under its
/// `_ordis_` branch scopes (one pass over `intEncodedStatements`,
/// prefix match on `<parent>_boundary_ordis_<sig>_(`), ties to the
/// byte-lex-smaller signature — deterministic. Every loser is
/// retired: its LIVE branches (still in `orLiveBranches`) are
/// queued on `pendingWipeScopes` and leave the live registry (the
/// freeze sweep runs after this drain and never sees them), its
/// scheduling rows leave (`orDisjunctCount`, `orPendingBranches`,
/// `orPendingLevels`, `orStarterPick` — a stale `pendingOrReleases`
/// staging is the release drain's defined skip on the missing
/// count row), and its ledger row is cleared so a re-deposit of
/// that spelling suppresses against the keeper. FROZEN branches,
/// `orBookkeeping` runs, `orRetiredDisjuncts` and every history
/// line are KEPT — a frozen branch's derivations are load-bearing
/// (I-206), and with the count row gone the loser can never
/// converge. Retired work re-derives under the keeper when needed
/// (the maintainer-accepted trade-off). Parked-only duplicates
/// need no handling here: the ordis park hooks rewrite and re-mail
/// them, and the door's gate makes the re-file passive.
///
/// Runs in `standardProcessing` between `drainDeadOrBranches` and
/// `freezeResolvedOrBranches`; the queued wipes land in this same
/// burst's `pendingWipeScopes` drain. Heap-free (Rule 28): the
/// ledger snapshot and canonical texts ride the gen-scratch and
/// string tiers, reclaimed at exit. Mints nothing into any
/// interner (canonicalization, recovery probes and the wipe-queue
/// staging are all non-minting), so held `decodeView` spans stay
/// valid throughout (I-3).
///
/// @param body The LB whose ledger is swept (containers resident —
///             the drain runs inside the LB's own burst).
/// @return (void)
/// @invariant Scopes and group members process in ledger snapshot
///            order (deterministic single-threaded insert order);
///            the keeper is a pure function of container content.
/// @invariant Never retires a cohort that is the only open one in
///            its equi group.
/// @see orEquiRepresentativeRecorded — the deposit-side gate;
///      drainPendingOrReleases — the defined skips the retirement
///      relies on; Memory::processedOrLedger.
void retireDuplicateOrCohorts(Memory& body);

public:
/// @brief End-of-burst retirement of duplicate OR-integration families —
///        of two open families under one parent whose ORs became
///        class-equal after both opened, the one with the most statements
///        survives (D-550).
///
/// @details
/// To integrate an OR at a parent scope, Case OR of
/// `prepareIntegrationCore2` opens one case scope per disjunct,
/// `<parent>_boundary_orint_<signature>_(<disjunct>)`, which assumes the
/// other disjuncts false and holds its own disjunct as goal. The cases of
/// one parent and one signature form a family; the first case that reaches
/// its goal emits the OR at the parent and `cleanUpOrIntegrationBranches`
/// wipes the family. The opening's prepared-template gate is class-aware,
/// so two families of class-equal ORs coexist only when the equality
/// arrived after both opened; both would then search to the end.
///
/// Candidates are the case scopes that hold a goal row in `intToBeProved`
/// (an open case always does: its goal leaves only with the family), whose
/// payload is bare (`orint_` first; a goal-qualified main family
/// `<goal>_subproof_orint_...` follows its goal's lifecycle, as D-414 keeps
/// goal-driven subproofs), and which are neither wiped nor queued, itself
/// or through an ancestor. Families of one parent whose signatures have the
/// same canonical form there (`canonicalFormAtScope`, text only) form one
/// group. The KEEPER has the most statements in its cases and every scope
/// nested below them (`orCohortStatementCount`, `Integration`); ties go to
/// the byte-lex smaller signature. Every case scope of every other member
/// is queued on `pendingWipeScopes`, so this burst's wipe drain removes it
/// with its nested scopes. `cleanUpOrIntegrationBranches` is not used: on
/// main it would also take goal-qualified families of the same signature.
///
/// The parent's prepared key stays, so a retired family never reopens, and
/// the wipe adds its scopes to `intValidityNamesToFilter`, so it is never
/// queued again. Proof history is untouched (I-44). Whatever the retired
/// family had derived re-derives under the keeper, whose OR every
/// class-equal consumer matches through the classes (I-254).
///
/// Runs in `standardProcessing` immediately after
/// `retireSettledImplicationSubproofs`, before the wipe drain. Heap-free
/// (Rule 28): the case and family runs ride the gen-scratch page tier and
/// the canonical texts the string tier, reclaimed at exit. Mints nothing
/// into any interner, so the held `decodeView` signature spans stay valid
/// (I-3).
///
/// @param body The LB whose open OR-integration families are swept
///             (containers resident — the sweep runs inside the LB's own
///             burst).
/// @return (void)
/// @invariant The retired set is a pure function of container content
///            (I-84): grouping and keeper choice sort by parent id,
///            canonical text and signature bytes.
/// @invariant Never retires the only open family of its group.
/// @see orCohortStatementCount; cleanUpOrIntegrationBranches;
///      retireDuplicateOrCohorts — the `_ordis_` twin;
///      ImplicationSubproofs::retireSettledImplicationSubproofs — the
///      implication-subproof twin.
void retireDuplicateOrIntegrationFamilies(Memory& body);

public:
/// @brief Retire every open OR cohort one of whose still-open
///        disjuncts has come to hold at the cohort's parent scope
///        (D-438, the retire-later half).
///
/// @details
/// The deposit-side gate in `consumeOrLeavesCohort` opens no case
/// split for an OR whose disjunct already holds; this sweep covers a
/// disjunct that comes to hold after its cohort opened. The
/// still-open disjuncts of a cohort are the disjunct of its live
/// branch (at most one per cohort, I-174) and its queued disjuncts
/// (`orPendingBranches`); each is tested with `orDisjunctTrueAt` at
/// the cohort's parent scope. The branch of a disjunct that holds is
/// the parent scope itself, so the cohort can no longer conclude
/// anything the parent does not prove by itself.
///
/// A satisfied cohort is retired exactly like a duplicate loser in
/// `retireDuplicateOrCohorts`: its live branch is queued on
/// `pendingWipeScopes` and leaves `orLiveBranches`, and its
/// scheduling rows leave (`orDisjunctCount`, `orPendingBranches`,
/// `orPendingLevels`, `orStarterPick`; a stale `pendingOrReleases`
/// staging is the release drain's defined skip on the missing count
/// row). Frozen branches, `orBookkeeping` runs, `orRetiredDisjuncts`
/// and every history line are kept. The processed-or ledger row
/// stays, so a re-deposit of the OR at that scope stays suppressed.
///
/// Runs in `standardProcessing` after `retireDuplicateOrCohorts` and
/// before `freezeResolvedOrBranches` and `drainPendingOrReleases`, so
/// a retired cohort is neither evaluated nor released and its wipes
/// land in this same call's `pendingWipeScopes` drain. Heap-free
/// (Rule 28): the satisfied-cohort set rides the gen-scratch page
/// tier, reclaimed at exit. Mints nothing into the NameMap or the
/// `lbStateInterner`, so held `decodeView` spans stay valid (I-3).
///
/// @param body The LB whose open cohorts are swept (containers
///             resident — the sweep runs inside the LB's own burst).
/// @return (void)
/// @invariant Only cohorts with a still-open disjunct that holds at
///            the parent scope are retired; the decision is a pure
///            function of container content (I-84).
/// @see orDisjunctTrueAt, consumeOrLeavesCohort,
///      retireDuplicateOrCohorts, drainPendingOrReleases.
void retireSatisfiedOrCohorts(Memory& body);

/// @brief Retire every implication subproof one of whose live OR cases
///        holds the negation of a goal of the subproof
///        (D-549).
///
/// @details
/// A live case serves the goals of its innermost goal scope
/// (`StatementKnowledge::innermostGoalScope`). When that scope is a
/// numbered implication subproof (its payload, bare or behind a
/// `<goal>_subproof_` prefix, is `(implication...)`) and the exact
/// negation of one of its goals is known at the case or above, the case
/// can reach that goal only by becoming contradictory: its cohort
/// cannot converge, so the subproof cannot close. The subproof scope is
/// queued on `pendingWipeScopes`, as `retireEqualInputImplicationSubproofs`
/// does, and its live cases leave `orLiveBranches`, as
/// `retireDuplicateOrCohorts` drops a loser's live branch, so the freeze
/// sweep never evaluates them. The prepared key survives, so the subproof
/// does not reopen. Nothing is proved or refuted: the wipe abandons search
/// only, and history is kept. Main's goals, integration subproofs and
/// OR-integration scopes are never retired here. A block primed for
/// contradiction is skipped: there the parent is meant to be inconsistent.
///
/// Runs in `standardProcessing` after `retireSatisfiedOrCohorts` and
/// before `freezeResolvedOrBranches`; the wipes land in this same call's
/// `pendingWipeScopes` drain. Heap-free (Rule 28): the retired-scope run
/// rides the gen-scratch page tier and the negated goal texts the string
/// tier, both reclaimed at exit. Mints nothing.
///
/// @param body The LB whose live cases are swept (containers resident —
///             the sweep runs inside the LB's own burst).
/// @return (void)
/// @invariant The decision is a pure function of container content
///            (I-84); a scope already closed or queued is skipped.
/// @see StatementKnowledge::innermostGoalScope,
///      ImplicationSubproofs::retireEqualInputImplicationSubproofs,
///      retireDuplicateOrCohorts, freezeResolvedOrBranches.
void retireRefutedGoalSubproofs(Memory& body);

public:
/// @brief Test whether this block has work that can require OR cases.
///
/// @details
/// Explicit goals retain their usual eligibility. An active primed
/// contradiction block also has proof work even without explicit goals;
/// its obligation is discharged by a registered main-scope contradiction.
/// A conjecture's goal block has proof work while its last premise is
/// seeded, before its goal is registered (`Memory::goalPending`,
/// D-495).
///
/// @param body The resident logical block whose proof state is inspected.
/// @return True for explicit goals, a pending conjecture goal, or an active
///         primed contradiction block.
/// @invariant Does not allocate or change proof, history or scheduling state.
/// @see consumeOrLeavesCohort, drainPendingOrReleases.
bool hasOrProofWork(const Memory& body) const;

private:
/// @brief Test whether an OR branch covers every goal of its innermost goal
///        scope.
///
/// @details
/// Relevant goals are those of the deepest goal-holding scope on the branch's
/// chain (`StatementKnowledge::innermostGoalScope`, D-549);
/// goals further up are shadowed, and goals in sibling and descendant scopes
/// do not participate. Each relevant goal must be known at the branch or
/// above, or supported up to equality at the branch. A chain without any goal
/// remains unresolved in an active primed contradiction block, whose main owns
/// the contradiction as its implicit goal; otherwise it qualifies.
///
/// @param body The resident logical block whose registries are read.
/// @param branchId A minted OR-disintegration branch root.
/// @return True when no relevant goal remains unknown.
/// @invariant The predicate neither mints names nor reads proof history.
/// @see ancestorKnown, innermostGoalScope, collectPhase3IgnoredOrRoots,
///      freezeResolvedOrBranches.
bool orBranchGoalsResolved(const Memory& body, NameId branchId);

public:
/// @brief Snapshot the OR roots eligible for Phase 3 inference exclusion.
///
/// @details
/// Walk existing scope identifiers, including roots whose seeds are still
/// queued. The result is fixed before any Phase 3 inference runs; descendants
/// are covered by the later ancestry test without extending this root set.
///
/// @param body The resident logical block, read without mutation.
/// @param roots An initially empty arena-backed output set.
/// @return Nothing.
/// @invariant No goal, branch marker, name or wipe queue is changed.
/// @see orBranchGoalsResolved, Memory::phase3IgnoresScope.
void collectPhase3IgnoredOrRoots(const Memory& body,
            ColdHashSet<PodKeyStore<NameId>>& roots);

public:
/// @brief Queue resolved OR-disintegration branches for subtree wipeout.
///
/// @details
/// Evaluate every live branch against the current goals on the branch's own
/// ancestor chain. A goal must be known at the branch or above; unrelated goals
/// are ignored and an empty chain-goal set qualifies. The resolved marker and
/// pending wipe queue are updated after the read-only predicate completes.
///
/// The ordered end-of-burst wipe removes runtime state throughout each queued
/// subtree, preserving convergence bookkeeping, cohort counts and history.
/// Only booked current-goal derivation levels are saved separately for later
/// reduced-count convergence. Parent-chain goals close through the ordinary
/// convergence deposit path.
///
/// @param body The resident logical block whose live branches are evaluated.
/// @return Nothing.
/// @invariant The predicate reads statement membership, never origin history.
/// @invariant No runtime registry is erased while the goal snapshot is scanned.
/// @see Memory::wipeSubtree, Memory::pendingWipeScopes, ancestorKnown.
void freezeResolvedOrBranches(Memory& memoryBlock);

public:
/// @brief Test whether an explicit induction ancestor still awaits its hypothesis.
///
/// @details A settled tree keeps its winning scopes, so scope syntax alone is
/// insufficient. Unopened OR queues wait until the existing settlement drain
/// records the firing; ordinary scopes and an induction block's main stay open.
/// @param body Resident owning block.
/// @param parent Minted parent of the proposed case.
/// @return True while any enclosing induction tree is unsettled.
/// @invariant Reads runtime settlement state, never proof history.
/// @see Memory::settleInductionTrees, drainPendingOrReleases.
bool orSplitBlockedByInduction(const Memory& body, NameId parent) const;

/// @brief Whether one of an OR leaf's matched demands is a missing expression
///        of the block's induction hypothesis.
///
/// @details
/// The route-(c) probe of `consumeOrLeavesCohort` matches a leaf against the
/// ordis2 demand keys of the OR's scope. A key holds one record per demanding
/// rule; a record whose source is an induction hypothesis rule of the block
/// (`Memory::isInductionHypothesisRule`) is the hypothesis's missing
/// expression (D-575). A cohort opened by such a demand is
/// released below an unsettled induction scope, the I-328 exemption that lets
/// the hypothesis fire in its case. The records are read in place, heap-free.
///
/// @param mb    The block owning the demand map.
/// @param keys  The matched demand keys, each present in `admissionMapOrdis2`
///              (asserted).
/// @param arena The per-slot page-tier scratch arena for record reads.
/// @return `true` iff some record of some key has a hypothesis rule as source.
/// @invariant Pure read; mints nothing.
/// @see consumeOrLeavesCohort, drainPendingOrReleases,
///      AdmissionMemory::fileHypothesisOrDemand.
static bool inductionHypothesisDemanded(const Memory& mb, const PagedVector<int64_t>& keys,
    ScratchArena& arena);

/// @brief Find an existing execution of a case at exactly the requested scope.
///
/// @details Compare assumptions using only the parent's visible equalities.
/// Live executions include queued seeds. Completed executions qualify only
/// when saved results cover the still-unknown goals on the common parent chain.
/// Ties use decoded scope order. Wiped parents and retiring live cases cannot
/// supply work; a completed case may survive its own runtime wipe.
/// @param body Resident owning block.
/// @param parent Minted common parent scope.
/// @param assumption Unwrapped case expression.
/// @param completed Select completed cases instead of live executions.
/// @return Matching case scope, or zero for a defined cache miss.
/// @invariant No sibling or child equality can identify two cases.
/// @see freezeResolvedOrBranches, reuseOrCase.
NameId commonOrCaseAt(Memory& body, NameId parent,
    StrSpan assumption, bool completed);

/// @brief Collect exactly the parent equalities that identify two case assumptions.
///
/// @details A differing argument contributes the directed equality from the
/// destination argument to the source argument, together with that pair's
/// recorded derivation levels. Equal argument positions contribute nothing.
/// Repeated pairs are deduplicated. All evidence comes from the common parent's
/// complete class view; no history is read and no interner is minted.
/// @param body Resident owning block.
/// @param source Source case scope.
/// @param target Destination case at the same parent, with an equal assumption.
/// @param arena String arena retaining output equality text for the caller.
/// @param equalities Caller output buffer for directed equality expressions.
/// @param capacity Number of entries available in the output buffer.
/// @param levels Caller level buffer, ascending and unique, with capacity 256.
/// @param levelCount Current level count, extended by only the used equalities.
/// @return Number of directed equalities; zero for identical assumptions.
/// @invariant Unchanged arguments never add premise credit.
/// @see commonOrCaseAt, copySharedOrCaseHistory, reuseOrCase.
int32_t collectSharedOrCaseEqualities(Memory& body, NameId source,
    NameId target, ScratchArena& arena, StrSpan* equalities, int32_t capacity,
    int* levels, int32_t& levelCount);

/// @brief Write a cohort's own proof history for a reused case derivation.
///
/// @details Copy the completed source's scoped history with one structural
/// prefix substitution. Parent evidence remains at its original scope. Replace
/// the source case assumption by the destination cohort's own OR source and,
/// for parent-equal assumptions, an explicit equality bridge. This function
/// writes documentation only; no inference or scheduling reads its results.
/// @param body Resident block containing both named cases.
/// @param source Completed case scope.
/// @param target Destination cohort's case scope, at the same parent.
/// @return Nothing.
/// @invariant Every exported case retains the existing verifier contract.
/// @see reuseOrCase, ProofHistory::findEqualityCiteScope.
void copySharedOrCaseHistory(Memory& body, NameId source, NameId target);

/// @brief Reuse a completed case without creating a second runtime subtree.
///
/// @details Save the destination's goal contributions and derivation levels,
/// including the parent equalities identifying the case. The destination
/// cohort adds its own OR-source levels only at convergence.
/// Equality-supported goals retain their actual fact and equality evidence.
/// Each contribution enters ordinary cohort convergence; the destination is
/// frozen before booking so it can never become a live inference branch.
/// @param body Resident owning block.
/// @param source Completed execution at the common parent.
/// @param target Newly named destination case at that same parent.
/// @return Nothing.
/// @invariant Reuse never imports a sibling assumption or reads history to prove a goal.
/// @see copySharedOrCaseHistory, commonOrCaseAt, ordisMerge.
void reuseOrCase(Memory& body, NameId source, NameId target);

public:
/// @brief End-of-burst release drain of the sequenced
        ///        or-disintegration cohorts — mint the next pending branch of
        ///        every cohort whose live branch resolved this burst.
        ///
        /// @details
        /// Consumes `Memory::pendingOrReleases` (staged by the cohort
        /// bootstrap in `disintegrateExprCore2` — the INITIAL release, since
        /// the bootstrap itself may not touch the mail channel being
        /// drained around it; by `ordisMerge`'s goal probe when a
        /// branch-scope deposit matches a `toBeProved` goal on the branch's
        /// parent chain; and by `drainDeadOrBranches` when a live branch
        /// retires refuted). Per staged cohort, in decoded cohort-identity
        /// order (parent bytes, then signature bytes — I-84):
        ///
        /// 1. **Defined skips.** No `orDisjunctCount` row (the cohort retired
        ///    wholesale — a stale staging), no `orPendingBranches` row
        ///    (nothing left to release), or the cohort's parent scope sits in
        ///    `intValidityNamesToFilter` (a retired region — the queue is
        ///    left untouched).
        /// 2. **Wait or reuse.** An unfired induction ancestor parks the
        ///    release, except for a cohort in `LbMemory::hypothesisOrCohorts`
        ///    (opened by the induction hypothesis's missing expression,
        ///    D-575), whose cases release beneath it.
        ///    Any live same-parent case contained in the remaining
        ///    cohort blocks the whole cohort, irrespective of rank. Completed
        ///    shared cases contribute their saved results and own scoped proof.
        /// 3. **Rank.** The pending disjuncts are re-ranked by
        ///    `pickTopOrDisjunct` over their clean forms plus the LB's anchor
        ///    arguments (`collectAnchorArgs`) — the release order is a pure
        ///    function of the remaining set, never stored.
        /// 4. **Release.** The top-ranked disjunct's branch scope name
        ///    `<parent>_boundary_ordis_<orSig>_(<disjunct>)` is built on the
        ///    string tier; the seed statement plus its `or disintegration`
        ///    origin ride `sameIterationInternalMail` (the `ordisMerge`
        ///    promotion channel) as a level-free hypothetical assumption;
        ///    `orPendingLevels` supplies this cohort's source levels at convergence, so the next absorb runs the seed through
        ///    the full kernel pipeline — a compound disjunct decomposes there
        ///    like any deposited statement.
        /// 5. **Queue update.** The released disjunct leaves
        ///    `orPendingBranches` (erase + reinsert through the canonical
        ///    doors); `orPendingLevels` stays until the cohort retires.
        ///    One release or reuse per staged cohort per burst.
        ///
        /// Runs in `standardProcessing` between `drainDeadOrBranches` (so a
        /// refuted live branch releases its successor in the same burst) and
        /// the `pendingWipeScopes` drain. Heap-free (Rule 28): the cohort
        /// snapshot rides the gen-scratch tier, the branch-validity strings
        /// the string tier, both reclaimed at exit.
        ///
        /// @param memoryBlock The LB whose release inbox is drained (its
        ///                    containers are resident — the drain runs inside
        ///                    the LB's own burst).
        /// @invariant A block without explicit goals or active contradiction
        ///            work releases nothing; its staging drops.
        /// @invariant At most one branch is released per cohort per burst;
        ///            the staged-cohort processing order and the released
        ///            disjunct are pure functions of container content.
        /// @see Memory::pendingOrReleases, Memory::orPendingBranches,
        ///      Memory::orPendingLevels, pickTopOrDisjunct,
        ///      collectAnchorArgs, ordisMerge, drainDeadOrBranches.
void drainPendingOrReleases(Memory& memoryBlock);

public:
// Turn optimizations OFF for just this section
//#pragma optimize("", off)


/// @brief Queue every sibling branch of one proved OR-integration cohort for
///        radical subtree cleanup.
///
/// @details
/// Selects only validity scopes that classify as OR integration, carry the
/// exact outer signature @p orExprSig, and have @p parentValidityId as their
/// direct NameMap parent. The victims are synchronously filtered for the rest
/// of the current burst and queued for the end-of-burst `Memory::wipeSubtree`
/// drain. Because parent validity participates in selection, equal OR
/// signatures at another stack position remain live.
///
/// The scan stores ids on the request-generation scratch page tier and mints
/// no NameMap entries, preserving the caller-owned signature span and every
/// classify-view slice throughout both loops.
///
/// @param orExprSig Outer compiled OR signature shared by the cohort.
/// @param parentValidityId Exact direct parent of the proving goal scope.
/// @param mb Owning logic-block memory.
/// @return Nothing.
/// @invariant Every queued victim is a direct child of
///            @p parentValidityId and belongs to @p orExprSig.
/// @see Memory::wipeSubtree, ScopeSyntax::classifyOrScopeView.
void cleanUpOrIntegrationBranches(StrSpan orExprSig,
            NameId parentValidityId,
            Memory& memoryBlock);

public:
/// @brief Open or park one case-split leaf cohort.
/// @details Ordered flattened leaves share one parent-scope identity.
/// Proof work, depth, demand, and disintegration permission control cases;
/// an OR with a disjunct that already holds at the scope (`orDisjunctTrueAt`)
/// neither opens a case split nor parks (D-438). The depth is the number of
/// `_ordis_` case scopes on the scope's ancestry; `_orint_` branches do not
/// count (D-458). A source is a positive category-or
/// entity, or a negated conjunction deposited as an integration premise,
/// whose leaves are its complemented conjuncts and which always carries
/// route-(b) permission (D-460). A cohort opened through a demand whose
/// source is an induction hypothesis rule (`inductionHypothesisDemanded`) is
/// recorded in `LbMemory::hypothesisOrCohorts`: its cases are exempt from the
/// I-328 release block below an unsettled induction scope
/// (D-575).
/// Exclusion is handled separately by finalized source registration and
/// the on-demand evaluator; this consumer installs no exclusion rules.
/// @param expr Deposited OR expression for history.
/// @param entSignature Instantiated source signature with u-prefixed
///        arguments: an OR compact, or the negated conjunction.
/// @param orLeaves Ordered disjunct spans at their true polarity.
/// @param orLeafN Number of flattened disjuncts.
/// @param currentStatement Collected OR key.
/// @param memoryBlock Owning block.
/// @param collected Disintegration product sink.
/// @param validityName Parent scope.
/// @param trackHistoryLocal History permission.
/// @param allowOrDisintegration Unconditional case-opening signal.
/// @param orSeedLevels Ordered seed levels.
/// @param orSeedLevelCount Number of seed levels.
/// @param allowOrProbe Real-deposit demand-probe permission.
/// @param negatedConjunction The source is a negated conjunction integration
///        premise; false for a category-or entity.
/// @param forcedAssumptions The OR came from a conjecture premise (in place or
///        on a flag-6 arrival): a freshly opened cohort is recorded in
///        `LbMemory::forcedOrCohorts`, so each released case is disintegrated
///        with force_ witnesses (D-527).
/// @param sArena Caller string scratch.
/// @param instructions Caller instruction context.
/// @param iteration Caller witness iteration.
/// @param newVarMap Caller witness map.
/// @param parentWitness Caller parent witness.
/// @param witnessMeta Caller witness metadata.
/// @param experimentLeaf The leaf a twin block's case experiment releases
///        first (`twinExperimentLeaf`), or -1. A freshly opened cohort then
///        starts with that leaf. A zero block's cohort is recorded in
///        `LbMemory::experimentOrCohorts`, so `drainPendingOrReleases`
///        releases that one case only; the weak step block's cohort is
///        recorded as `Memory::twinCohortId` and follows the ordinary order
///        (D-588). The caller passes
///        route-(b) permission with it.
/// @return Void.
/// @invariant Only category-or entities and route-(b) negated conjunction
///            premises enter this consumer; all writes are block-local.
/// @see registerOrImplicationSource, evaluateOrImplication, flattenOrLeaves.
void consumeOrLeavesCohort(StrSpan expr,
            StrSpan entSignature,
            const StrSpan* orLeaves, int32_t orLeafN,
            StrSpan currentStatement,
            Memory& memoryBlock,
            CollectedArena& collected,
            StrSpan validityName,
            bool trackHistoryLocal,
            bool allowOrDisintegration,
            const int* orSeedLevels, int32_t orSeedLevelCount,
            bool allowOrProbe,
            bool negatedConjunction,
            bool forcedAssumptions,
            ScratchArena& sArena,
            WorkInstruction& instructions,
            int iteration,
            NewVarStore& newVarMap,
            StrSpan parentWitness,
            WitnessMetaStore* witnessMeta,
            int32_t experimentLeaf = -1);

/// @brief The leaf a twin block's case experiment releases first from an
///        OR, or -1 (D-588).
///
/// @details
/// A block armed for a case experiment (`Memory::twinExperiment`) opens the
/// zero-or-successor OR of its twin partner l at `main`: an OR of exactly two
/// leaves, the positive equality of l and the anchor's zero
/// (`Memory::twinZeroId`, either order), and a positive existence whose
/// compiled body holds `in2[w, l, s]` for its witness w and the anchor's
/// successor s (`Memory::twinSuccessorId`); FTA's `or1`, `inSequence[N,l,s]`
/// or `l = 0`. Both armed kinds release the existence leaf first: the weak
/// step block follows it with the zero leaf in the ordinary order, a zero
/// block releases nothing else. Any other OR, scope or block gives -1.
///
/// @param mb           The block whose OR is deposited.
/// @param orLeaves     The OR's flattened leaves, true polarity, `u_` form.
/// @param orLeafN      Number of leaves.
/// @param validityName The OR's scope.
/// @param sArena       String scratch for the cleaned and instantiated texts.
/// @return The index of the leaf to release, or -1.
/// @invariant Mints nothing; reads the compiled configuration and the block's
///            NameMap only.
/// @see consumeOrLeavesCohort, drainPendingOrReleases,
///      AdmissionMemory::updateAdmissionMap (arms the step block).
int32_t twinExperimentLeaf(const Memory& mb, const StrSpan* orLeaves, int32_t orLeafN,
            StrSpan validityName, ScratchArena& sArena) const;
private:

AdmissionPolicy& admissionPolicy;
const ce::AnchorInfo& anchorInfo;
CanonicalTemplateMatcher& canonicalTemplates;
Canonicalizer& canonicalizer;
const CompiledDefinitionRegistry& definitions;
const ProverParameters& parameters;
CaseLedger& caseLedger;
EquivalenceQueries& equivalenceQueries;
ProofHistory& proofHistory;
};
}
#include "../logical_block/case_ledger.hpp"
#include "../logical_block/equivalence_queries.hpp"
#include "../logical_block/proof_history.hpp"
namespace gl {

/// @brief Whether one OR disjunct already holds at a scope, so that a
        ///        case split on its OR could only reproduce that scope
        ///        (D-438).
        ///
        /// @details
        /// True when the disjunct is known at @p scopeId or an ancestor —
        /// literally, as a reflexive equality, or up to the complete classes
        /// of @p scopeId (`elementKnownAtOrAbove`) — or when it is a positive
        /// conjunction (an `and`-category compact, named or numbered) whose
        /// flattened elements all hold by the same test, e.g. the literally
        /// true `(and27[m,m,p,p])` of two reflexive equalities. A negated
        /// disjunct holds only through its own known row. The branch of a
        /// disjunct that already holds equals its parent scope, so
        /// convergence through that branch can only conclude what the parent
        /// scope already proves; `consumeOrLeavesCohort` therefore neither
        /// opens nor parks an OR with such a disjunct.
        ///
        /// @param clean   Clean (u_-stripped) disjunct text on caller-owned
        ///                storage.
        /// @param scopeId Minted scope of the OR statement.
        /// @param mb      Owning block, its classes and known-statement store.
        /// @param depth   Conjunction nesting depth of this call (0 at the
        ///                top; bounded by `MAX_INSTRUCTION_ELEMENTS`).
        /// @return True iff the disjunct holds at @p scopeId or an ancestor.
        /// @invariant Creates no proof row and mints nothing in the NameMap;
        ///            only `elementKnownAtOrAbove`'s arena-backed derived
        ///            comparison indexes are synchronized.
        /// @see elementKnownAtOrAbove, consumeOrLeavesCohort, orDisjunctTier.
inline bool CaseProcessor::orDisjunctTrueAt(StrSpan clean, NameId scopeId, Memory& mb,
                              int32_t depth) {
            assert(depth < ExecutionParameters::MAX_INSTRUCTION_ELEMENTS
                && "orDisjunctTrueAt: cyclic conjunction definition");
            if (equivalenceQueries.elementKnownAtOrAbove(clean, scopeId, mb)) return true;
            if (clean.len == 0 || clean.ptr[0] != '(') return false;
            const LogicalEntity* definition =
                this->definitions.compiledEntity(extractExpressionUniversalSpan(clean));
            if (definition == nullptr || definition->category != "and") return false;
            StrSpan formal[ExecutionParameters::MAX_ARITY];
            StrSpan actual[ExecutionParameters::MAX_ARITY];
            const int32_t count = getArgsSpans(StrSpan(definition->signature),
                formal, ExecutionParameters::MAX_ARITY);
            assert(count == getArgsSpans(clean, actual, ExecutionParameters::MAX_ARITY)
                && "orDisjunctTrueAt: conjunction arity differs from its definition");
            StrReplacement substitution[ExecutionParameters::MAX_ARITY];
            for (int32_t i = 0; i < count; ++i) substitution[i] = { formal[i], actual[i] };
            assert(!definition->elements.empty()
                && "orDisjunctTrueAt: conjunction definition without elements");
            const unsigned slot = (gl::WorkerContext::coreId() >= 0)
                ? static_cast<unsigned>(gl::WorkerContext::coreId())
                : scratchArenas().slotCount() - 1;
            ScratchArena& arena = scratchArenas().forSlot(slot);
            ScratchScope scope(arena);
            for (const auto& element : definition->elements) {
                const ScratchString child = replaceKeysScratch(arena,
                    StrSpan(element), substitution, count);
                if (!orDisjunctTrueAt(StrSpan(child), scopeId, mb, depth + 1))
                    return false;
            }
            return true;
        }

/// @brief Return the refuted `_ordis_` branch scope at or above a
        ///        validity scope, staging it for retirement — or 0 when the
        ///        chain holds no refuted branch.
        ///
        /// @details
        /// Walks the validity ancestor chain from @p validityId to the root.
        /// At each node two checks decide:
        ///
        /// 1. **Already staged** — a node present in
        ///    `Memory::pendingDeadOrBranches` is a branch already proven
        ///    dead this burst; it is returned immediately, no re-probe.
        /// 2. **Refutation probe** — at an `_ordis_` DISINTEGRATION scope
        ///    (`classifyOrScopeView`), the branch's asserted disjunct is
        ///    negated (a leading `!` sliced off, otherwise a `!`-prefixed
        ///    form built on the string-scratch arena) and probed `known` at
        ///    the branch scope or any ancestor — the exact staging predicate
        ///    `ordisMerge` uses (D-242): a scope's fact set is monotone and
        ///    the `known` record immortal (I-58), so a once-derived
        ///    refutation stays a refutation. On a hit the branch is minted
        ///    into `pendingDeadOrBranches` so the SAME burst's
        ///    `drainDeadOrBranches` retires it, and its id is returned. A
        ///    branch already retired (its scope is in
        ///    `intValidityNamesToFilter`) is returned the same way but not
        ///    staged again — the `ordisMerge` probe skips it alike — so each
        ///    branch retires once and its cohort count drops once (I-172).
        ///
        /// Every probe is non-minting (`NameMap::lookup`,
        /// `intKnownStatements` reads, `parentOf` metadata walks); the ONLY
        /// write is the `pendingDeadOrBranches` staging mint — the drain's
        /// own inbox, whose record step re-runs this predicate and asserts a
        /// hit, so a staged id always satisfies it.
        ///
        /// Consumed by `applyEquiClasses` Pass 1: a delta class whose
        /// validity sits at or below a refuted branch is ex-falso content in
        /// a subtree the drain wipes this same burst, so back-applying it
        /// across the registry is pure waste (the A20 runtime explosion —
        /// D-262,
        /// I-181).
        ///
        /// @param validityId  The scope whose ancestor chain is examined.
        /// @param memoryBlock The LB owning the scope forest and registries.
        /// @return The dead branch's validity id, or 0 when no `_ordis_`
        ///         branch at or above @p validityId is refuted.
        /// @invariant Single-threaded (phase-1/3) — writes only the LB-local
        ///            `pendingDeadOrBranches`.
        /// @see ordisMerge — the staging peer on the deposit path.
        /// @see drainDeadOrBranches — the consumer of the staged id.
inline NameId CaseProcessor::refutedOrBranchAtOrAbove(NameId validityId,
                                               Memory& memoryBlock) {
            for (NameId scopeId = validityId; scopeId != 0;
                 scopeId = memoryBlock.nameMap.parentOf(scopeId)) {
                if (memoryBlock.pendingDeadOrBranches.lookup(scopeId) != 0)
                    return scopeId;
                StrSpan orSigView;
                StrSpan branchBodyView;
                if (gl::ScopeSyntax::classifyOrScopeView(memoryBlock.nameMap, scopeId,
                                        orSigView, branchBodyView)
                        == OrScopeKind::Disintegration) {
                    const unsigned slot = (gl::WorkerContext::coreId() >= 0)
                        ? static_cast<unsigned>(gl::WorkerContext::coreId())
                        : scratchArenas().slotCount() - 1;
                    ScratchArena& negArena = scratchArenas().forSlot(slot);
                    ScratchScope negScope(negArena);
                    const NameId negId = memoryBlock.nameMap.lookup(
                        negateOrCaseAssumption(branchBodyView, negArena));
                    if (negId != 0) {
                        for (NameId probeId = scopeId; probeId != 0;
                             probeId = memoryBlock.nameMap.parentOf(probeId)) {
                            const StatementFlags* row =
                                memoryBlock.intKnownStatements.find(
                                    StatementKey{ negId, probeId });
                            if (row != nullptr) {
                                // A branch retired earlier sits in the
                                // filter: still refuted, but drainDeadOrBranches
                                // already retired it, so it is not staged
                                // again (each branch retires once, I-172).
                                if (memoryBlock.intValidityNamesToFilter.lookup(scopeId) == 0)
                                    memoryBlock.pendingDeadOrBranches.mint(
                                        scopeId);
                                return scopeId;
                            }
                            if (probeId == NameMap::MAIN_ID) break;
                        }
                    }
                }
                if (scopeId == NameMap::MAIN_ID) break;
            }
            return 0;
        }

/// @brief Stage the sequenced-release goal probe (I-174) for a
        ///        deposit REFUSED by the ancestor-known dedup gate.
        ///
        /// @details
        /// The or-cohort sequenced release is deposit-driven: `ordisMerge`
        /// mints `pendingOrReleases` when a branch deposit's expression
        /// matches an open `toBeProved` goal of the innermost goal scope on the
        /// branch's parent chain (D-549).
        /// An ancestor-known refusal (D-273)
        /// suppresses the deposit — but the branch still DERIVED the
        /// expression, so the resolution signal must survive, or a live
        /// branch whose goal-matching derivation happens to be
        /// ancestor-known (the goal sitting shallower than the knowing
        /// scope) would never release its cohort's next branch — an I-174
        /// stall that loses theorems. This helper runs exactly
        /// `ordisMerge`'s probe on the refused (expression, scope) pair.
        ///
        /// Entirely non-minting: a scope that is not an `_ordis_` branch,
        /// or whose parent / signature / packed cohort key was never
        /// interned, means no cohort ever opened here — a defined miss
        /// (I-83), never a fallback. The only mutation is the
        /// `pendingOrReleases` mint itself (the intended signal; the
        /// pod-set mint dedups repeats).
        ///
        /// @param memoryBlock Owning LB.
        /// @param origId      Minted NameMap id of the refused expression.
        /// @param valId       Minted NameMap id of the refused deposit
        ///                    scope.
        /// @invariant [I-174](../../docs/agentic_swdd/30_invariants.md#i-174)
        ///            — sequenced release; the probe mirrors `ordisMerge`'s
        ///            byte-for-byte (pending-row gate, dedup gate, innermost
        ///            goal scope of the parent chain).
        /// @see ordisMerge — the deposit-driven twin of this probe;
        ///      drainPendingOrReleases — the end-of-burst consumer.
inline void CaseProcessor::stageOrReleaseForRefusedDeposit(Memory& memoryBlock,
                                                    NameId origId,
                                                    NameId valId) {
            StrSpan orSigV, branchBodyV;
            if (gl::ScopeSyntax::classifyOrScopeView(memoryBlock.nameMap, valId,
                                    orSigV, branchBodyV)
                    != OrScopeKind::Disintegration) {
                return;
            }
            const NameId parentVid = memoryBlock.nameMap.parentOf(valId);
            if (parentVid == 0) return;
            const StrSpan parentV = memoryBlock.nameMap.decodeView(parentVid);
            const int32_t parentId =
                memoryBlock.lbStateInterner.lookup(parentV);
            if (parentId == 0) return;
            const int32_t sigId = memoryBlock.lbStateInterner.lookup(orSigV);
            if (sigId == 0) return;
            const int32_t cohortId = lookupOrCohortId(
                memoryBlock.lbStateInterner, parentId, sigId);
            if (cohortId == 0) return;
            if (memoryBlock.orPendingBranches.lookup(cohortId) == 0
                || memoryBlock.pendingOrReleases.lookup(cohortId) != 0) {
                return;
            }
            const NameId goalScope =
                StatementKnowledge::innermostGoalScope(memoryBlock, parentVid);
            if (goalScope != 0 && memoryBlock.intToBeProved.lookup(
                    packStatementKey(origId, goalScope)) != 0)
                memoryBlock.pendingOrReleases.mint(cohortId);
        }

/// @brief Count the argument tokens of an MPL expression — the
        ///        structural-complexity measure of the sequenced
        ///        or-disintegration ranking.
        ///
        /// @details
        /// Fires `collectExprTokens`'s flat `[...]` bracket scan with a
        /// counting sink: every comma-separated token inside every bracket
        /// group contributes one, so a nested compound counts the tokens of
        /// all its levels. A bare equality `(=[a,b])` counts 2; an
        /// `(existence3[x,y,z])` counts 3; a De-Morgan compound counts every
        /// nested argument. Zero-allocation, pure function of the bytes.
        ///
        /// @param expr Span over one disjunct in clean (u_-stripped) form.
        /// @return The token count (>= 0).
        /// @see collectExprTokens, pickTopOrDisjunct.
inline int32_t CaseProcessor::orDisjunctTokenCount(StrSpan expr) {
            int32_t n = 0;
            collectExprTokens(expr, [&](StrSpan) { ++n; });
            return n;
        }

/// @brief Rank tier of one or-disjunct for the sequenced release
        ///        order.
        ///
        /// @details
        /// Tier 0 — everything that is not a positive equality (compounds,
        /// existence forms, negations): released first, most complex first.
        /// Tier 1 — a positive equality `(=[a,b])` with no anchor-slot
        /// argument. Tier 2 — a positive equality with at least one
        /// anchor-slot argument (e.g. `(=[m,2])` where `2` is the anchor's
        /// zero slot): dead last — an assumed equality against an anchor
        /// constant feeds the equivalence-class machinery hardest, so it
        /// opens only after every other branch resolved. A NEGATED equality
        /// is tier 0: assuming `!(=[a,b])` expands one-sidedly (I-12) and
        /// carries none of the class-merge pathology.
        ///
        /// A positive conjunction (an `and`-category compact, named or
        /// numbered) whose flattened elements are all positive equalities
        /// assumes equalities just as a bare equality does (e.g. `and27` =
        /// two equalities), so it takes the highest tier among its elements:
        /// tier 1, or tier 2 when an element carries an anchor-slot argument
        /// (D-437). One non-equality element keeps the
        /// whole conjunction in tier 0; a negated conjunction is tier 0.
        ///
        /// @param clean      Span over the disjunct in clean form.
        /// @param anchorArgs The LB's anchor argument spans
        ///                   (`collectAnchorArgs`); may be null when
        ///                   `anchorArgN` is 0.
        /// @param anchorArgN Number of anchor argument spans.
        /// @param depth      Conjunction nesting depth of this call (0 at the
        ///                   top; bounded by `MAX_INSTRUCTION_ELEMENTS`).
        /// @return 0, 1, or 2 (lower releases earlier).
        /// @invariant Mints nothing; the element instances ride a
        ///            `ScratchScope` on the string-scratch arena.
        /// @see pickTopOrDisjunct, collectAnchorArgs.
inline int32_t CaseProcessor::orDisjunctTier(StrSpan clean,
                                      const StrSpan* anchorArgs,
                                      int32_t anchorArgN,
                                      int32_t depth) {
            assert(depth < ExecutionParameters::MAX_INSTRUCTION_ELEMENTS
                && "orDisjunctTier: cyclic conjunction definition");
            if (canonicalizer.isEquality(clean)) {
                StrSpan args[ExecutionParameters::MAX_ARITY];
                const int32_t argN =
                    getArgsSpans(clean, args, ExecutionParameters::MAX_ARITY);
                for (int32_t a = 0; a < argN; ++a)
                    for (int32_t k = 0; k < anchorArgN; ++k)
                        if (equalSpans(args[a], anchorArgs[k])) return 2;
                return 1;
            }
            if (clean.len == 0 || clean.ptr[0] != '(') return 0;
            const LogicalEntity* definition =
                this->definitions.compiledEntity(extractExpressionUniversalSpan(clean));
            if (definition == nullptr || definition->category != "and") return 0;
            StrSpan formal[ExecutionParameters::MAX_ARITY];
            StrSpan actual[ExecutionParameters::MAX_ARITY];
            const int32_t count = getArgsSpans(StrSpan(definition->signature),
                formal, ExecutionParameters::MAX_ARITY);
            assert(count == getArgsSpans(clean, actual, ExecutionParameters::MAX_ARITY)
                && "orDisjunctTier: conjunction arity differs from its definition");
            StrReplacement substitution[ExecutionParameters::MAX_ARITY];
            for (int32_t i = 0; i < count; ++i) substitution[i] = { formal[i], actual[i] };
            assert(!definition->elements.empty()
                && "orDisjunctTier: conjunction definition without elements");
            const unsigned slot = (gl::WorkerContext::coreId() >= 0)
                ? static_cast<unsigned>(gl::WorkerContext::coreId())
                : scratchArenas().slotCount() - 1;
            ScratchArena& arena = scratchArenas().forSlot(slot);
            ScratchScope scope(arena);
            int32_t tier = 1;
            for (const auto& element : definition->elements) {
                const ScratchString child = replaceKeysScratch(arena,
                    StrSpan(element), substitution, count);
                const int32_t elementTier = orDisjunctTier(StrSpan(child),
                    anchorArgs, anchorArgN, depth + 1);
                if (elementTier == 0) return 0;
                if (elementTier > tier) tier = elementTier;
            }
            return tier;
        }

/// @brief Pick the disjunct the sequenced or-disintegration releases
        ///        next — the argmin under the deterministic rank order.
        ///
        /// @details
        /// Rank order (best first): ascending tier (`orDisjunctTier` — non-
        /// equalities before bare equalities before anchor-argument
        /// equalities; a positive conjunction of equalities ranks as an
        /// equality), then DESCENDING token count (most structurally
        /// complicated first), then ascending byte-lex (`compareSpans`) as
        /// the tie-breaker. A pure function of the disjunct bytes plus the
        /// anchor argument set — the same inputs at cohort mint time and at
        /// every later release, so the release sequence is deterministic
        /// (I-84) and never stored.
        ///
        /// @param disjuncts  `n` spans over the candidate disjuncts in clean
        ///                   form.
        /// @param n          Candidate count (>= 1, asserted).
        /// @param anchorArgs The LB's anchor argument spans.
        /// @param anchorArgN Number of anchor argument spans.
        /// @return The index (0-based) of the top-ranked disjunct.
        /// @see orDisjunctTier, orDisjunctTokenCount, collectAnchorArgs.
inline int32_t CaseProcessor::pickTopOrDisjunct(const StrSpan* disjuncts, int32_t n,
                                         const StrSpan* anchorArgs,
                                         int32_t anchorArgN) {
            assert(n >= 1 && "pickTopOrDisjunct needs at least one candidate");
            int32_t best = 0;
            int32_t bestTier = orDisjunctTier(disjuncts[0], anchorArgs, anchorArgN);
            int32_t bestTokens = orDisjunctTokenCount(disjuncts[0]);
            for (int32_t i = 1; i < n; ++i) {
                const int32_t tier =
                    orDisjunctTier(disjuncts[i], anchorArgs, anchorArgN);
                const int32_t tokens = orDisjunctTokenCount(disjuncts[i]);
                bool better;
                if (tier != bestTier) better = tier < bestTier;
                else if (tokens != bestTokens) better = tokens > bestTokens;
                else better = compareSpans(disjuncts[i], disjuncts[best]) < 0;
                if (better) {
                    best = i;
                    bestTier = tier;
                    bestTokens = tokens;
                }
            }
            return best;
        }

/// @brief Collect the LB's anchor arguments — the ranking's
        ///        anchor-slot set.
        ///
        /// @details
        /// Walks the `parentMemory` chain from the given LB to the root and
        /// returns the argument spans of the first exprKey of the form
        /// `(<anchor name>[...])` — the batch's anchor element (the
        /// `createAuxyImplication` untouchables idiom). The spans alias the
        /// process-wide never-deloaded `skeletonInterner` cold pages (frozen
        /// through the parallel phase, D-186), so they survive every NameMap
        /// / per-LB interner mint. An LB whose chain carries no anchor
        /// element (or a batch without an anchor) yields 0 — a defined
        /// empty set, which simply makes ranking tier 2 unreachable.
        ///
        /// @param memoryBlock The LB whose chain is walked.
        /// @param out         Caller stack array receiving the spans.
        /// @param cap         Capacity of `out` (anchor arity <= MAX_ARITY).
        /// @return The number of anchor argument spans written.
        /// @see orDisjunctTier, pickTopOrDisjunct.
inline int32_t CaseProcessor::collectAnchorArgs(const Memory& memoryBlock,
                                         StrSpan* out, int32_t cap) {
            return gl::ScopeSyntax::anchorArgumentSpans(memoryBlock, StrSpan(anchorInfo.name), out, cap);
        }

/// @brief Record an OR-branch contribution and promote a complete cohort.
        ///
        /// @details
        /// Tracks each expression in its exact parent/signature cohort, retains
        /// branch derivation history, and promotes a conclusion when every live
        /// disjunct contributes. During mail absorption, the caller supplies the
        /// opposite output channel so the input clear cannot discard the new
        /// parent statement or its convergence history. Other callers use the
        /// ordinary same-iteration destination. Refutation routing is unchanged.
        ///
        /// @param addExpression The derived branch expression.
        /// @param effectiveValidity Its exact branch scope.
        /// @param addExpressionLevels Caller-owned ascending derivation levels.
        /// @param addExpressionLevelCount Number of derivation levels.
        /// @param memoryBlock The owning logical block.
        /// @param convergenceMailOut Optional output for convergence and branch
        ///        refutation during absorption;
        ///        nullptr selects the same-iteration channel outside absorption.
        /// @param rowRemovals Optional batch that records the converged
        ///        expression's branch-row removals for the caller to apply
        ///        after its drain; nullptr removes the rows immediately.
        /// @return Nothing; records contributions and emits complete conclusions.
        /// @invariant Contributions never mix cohort parents or OR signatures.
        /// @see standardProcessing, addExprToMemoryBlock, drainDeadOrBranches,
        ///      StatementRegistryOperations::RowRemovalBatch.
inline void CaseProcessor::ordisMerge(StrSpan addExpression,
                               StrSpan effectiveValidity,
                               const int* addExpressionLevels,
                               int32_t addExpressionLevelCount,
                               Memory& memoryBlock,
                               ColdMail* convergenceMailOut,
                               StatementRegistryOperations::RowRemovalBatch* rowRemovals) {
            if (equalSpans(effectiveValidity, StrSpan("main", 4))) return;

            NameId mergeValId = memoryBlock.nameMap.encode(effectiveValidity);
            // (orSig, branchBody) via the span twin: both are slices of the
            // validity payload (nameMap sub-table cold bytes). They feed the
            // lbStateInterner mints below BEFORE any NameMap mint
            // (insertInternalStatement, step 1), so the spans stay valid to last
            // use (I-3); mergeOrSig's later string uses recover the bytes from its
            // interned id via lbStateInterner.decodeView, which survives that mint.
            StrSpan mergeOrSigView;
            StrSpan mergeBranchBodyView;
            OrScopeKind mergeKind = gl::ScopeSyntax::classifyOrScopeView(memoryBlock.nameMap,
                                                        mergeValId,
                                                        mergeOrSigView,
                                                        mergeBranchBodyView);
            if (mergeKind != OrScopeKind::Disintegration) return;

            // During input absorption the caller supplies the opposite output
            // channel. Other callers retain the same-iteration destination.
            ColdMail& resultMail = convergenceMailOut != nullptr
                ? *convergenceMailOut : memoryBlock.sameIterationInternalMail;


            // Parent scope = effectiveValidity stripped of trailing
            // _boundary_<payload>. Same shape the legacy trackOrBookkeeping
            // computed and D-30 uses for _orint_. Only the payload LENGTH is
            // needed, so read it as a span (decodeSubView), not an owned copy;
            // the parent name is a prefix slice of the stable input.
            assert(!memoryBlock.nameMap.stackEmpty(mergeValId));
            const int32_t mergePayloadLen = memoryBlock.nameMap.decodeSubView(
                memoryBlock.nameMap.stackBack(mergeValId)).len;
            const std::size_t mergeParentEnd =
                static_cast<std::size_t>(effectiveValidity.len)
                    - (NameMap::BOUNDARY_LEN + static_cast<std::size_t>(mergePayloadLen));
            const StrSpan mergeParentView(effectiveValidity.ptr,
                                          static_cast<int32_t>(mergeParentEnd));

            // Dead-branch probe: the branch's asserted disjunct refuted at the
            // branch scope or an ancestor makes every branch fact ex falso —
            // stage the branch vid for the end-of-burst drainDeadOrBranches
            // retirement. The probe is non-minting on every path (a negation
            // lookup miss means the refutation was never derived — a live
            // branch, the contracted negative). `known` is the right bit: the
            // record is immortal (I-58), and a scope's fact set is monotone,
            // so a once-derived refutation stays a refutation. The pod-set
            // mint dedups; an already-staged or already-filtered branch skips
            // the probe outright.
            const NameId contributionId = memoryBlock.nameMap.lookup(addExpression);
            const bool equalitySupport = contributionId != 0
                && memoryBlock.lbMemory.orSupportFacts.lookup(
                    StatementKey{ contributionId, mergeValId }) != 0;
            if (!equalitySupport
                && memoryBlock.pendingDeadOrBranches.lookup(mergeValId) == 0
                && memoryBlock.intValidityNamesToFilter.lookup(mergeValId) == 0) {
                assert(mergeBranchBodyView.len >= 2
                    && mergeBranchBodyView.ptr[0] == '('
                    && mergeBranchBodyView.ptr[mergeBranchBodyView.len - 1] == ')'
                    && "ordis branch payload must be a wrapped (disjunct)");
                const StrSpan probeAssumption(mergeBranchBodyView.ptr + 1,
                                              mergeBranchBodyView.len - 2);
                const unsigned probeSlot = (gl::WorkerContext::coreId() >= 0)
                    ? static_cast<unsigned>(gl::WorkerContext::coreId())
                    : scratchArenas().slotCount() - 1;
                ScratchArena& probeArena = scratchArenas().forSlot(probeSlot);
                ScratchScope probeScope(probeArena);
                // negate(assertedDisjunct): slice off a leading '!' or build
                // the '!'-prefixed form on the string tier (held for both the
                // staging probe and the refutation emission below).
                StrSpan disjNeg;
                if (probeAssumption.len > 0 && probeAssumption.ptr[0] == '!') {
                    disjNeg = StrSpan(probeAssumption.ptr + 1,
                                      probeAssumption.len - 1);
                } else {
                    char* probeBuf = probeArena.allocBytes(probeAssumption.len + 1);
                    probeBuf[0] = '!';
                    std::memcpy(probeBuf + 1, probeAssumption.ptr,
                                static_cast<std::size_t>(probeAssumption.len));
                    disjNeg = StrSpan(probeBuf, probeAssumption.len + 1);
                }
                const NameId probeNegId = memoryBlock.nameMap.lookup(disjNeg);
                bool disjunctRefuted = false;
                if (probeNegId != 0) {
                    for (NameId probeScopeId = mergeValId; probeScopeId != 0;
                         probeScopeId = memoryBlock.nameMap.parentOf(probeScopeId)) {
                        const StatementFlags* probeRow =
                            memoryBlock.intKnownStatements.find(
                                StatementKey{ probeNegId, probeScopeId });
                        if (probeRow != nullptr) {
                            memoryBlock.pendingDeadOrBranches.mint(mergeValId);
                            disjunctRefuted = true;
                            break;
                        }
                        if (probeScopeId == NameMap::MAIN_ID) break;
                    }
                }

                // Branch-contradiction refutation: the DEPOSIT's negation
                // `known` at the branch scope or an ancestor means the branch
                // holds a contradiction — parent facts plus the asserted
                // disjunct derive both X and !X, so the parent refutes the
                // disjunct. EMIT negate(assertedDisjunct) at the cohort
                // parent (internal mail, two-antecedent `contradiction`
                // origin — the dischargeContradictionScopes row shape); once
                // absorbed and `known` there, the staging probe above
                // retires the branch on its next deposit (D-242). Only when
                // the refutation is not already known (the emission is the
                // producer, not a duplicate); non-minting probes throughout.
                if (!disjunctRefuted) {
                    StrSpan depNeg;
                    if (addExpression.len > 0 && addExpression.ptr[0] == '!') {
                        depNeg = StrSpan(addExpression.ptr + 1,
                                         addExpression.len - 1);
                    } else {
                        char* depBuf = probeArena.allocBytes(addExpression.len + 1);
                        depBuf[0] = '!';
                        std::memcpy(depBuf + 1, addExpression.ptr,
                                    static_cast<std::size_t>(addExpression.len));
                        depNeg = StrSpan(depBuf, addExpression.len + 1);
                    }
                    const NameId depNegId = memoryBlock.nameMap.lookup(depNeg);
                    NameId negKnownScope = 0;
                    if (depNegId != 0) {
                        for (NameId s = mergeValId; s != 0;
                             s = memoryBlock.nameMap.parentOf(s)) {
                            const StatementFlags* row =
                                memoryBlock.intKnownStatements.find(
                                    StatementKey{ depNegId, s });
                            if (row != nullptr) { negKnownScope = s; break; }
                            if (s == NameMap::MAIN_ID) break;
                        }
                    }
                    if (negKnownScope != 0) {
                        // The refutation's levels: the branch derivation's
                        // levels UNIONED with the ambient negation's — both
                        // antecedents feed the reductio (the
                        // dischargeContradictionScopes level contract). The
                        // {-1} non-derived tier is transparent: it never
                        // enters the union (an all-non-derived union stays
                        // empty and re-enters as {-1} at the addStatement
                        // door).
                        int refutRun[256];
                        int32_t refutN = 0;
                        assert(addExpressionLevelCount <= 256);
                        for (int32_t j = 0; j < addExpressionLevelCount; ++j) {
                            if (addExpressionLevels[j] < 0) continue;
                            refutRun[refutN++] = addExpressionLevels[j];
                        }
                        {
                            const int32_t negLvlsId = lookupStatementLevels(
                                memoryBlock.intStatementLevelsMap,
                                memoryBlock.nameMap, depNeg,
                                memoryBlock.nameMap.decodeView(negKnownScope));
                            assert(negLvlsId != 0
                                && memoryBlock.intStatementLevelsMap.runLen(negLvlsId) > 0
                                && "ordis branch-contradiction: known "
                                   "ambient negation without a non-empty level row");
                            int negRun[256];
                            const int32_t negN = coldIntRunAt(
                                memoryBlock.intStatementLevelsMap, negLvlsId,
                                negRun, 256);
                            for (int32_t j = 0; j < negN; ++j)
                                refutN = insertLevelSorted(
                                    refutRun, refutN, negRun[j], 256);
                        }
                        insertInternalStatement(
                            resultMail,
                            memoryBlock.nameMap, disjNeg, mergeParentView,
                            refutRun, refutN);
                        if (parameters.trackHistory) {
                            // Antecedents: the branch derivation and the
                            // ambient negation at its shallowest-found scope.
                            // The scope decodeView runs AFTER the statement
                            // door's NameMap mints (I-3); the origin door
                            // mints originInterner only.
                            const OriginDep contraDeps[2] = {
                                { addExpression, effectiveValidity },
                                { depNeg, memoryBlock.nameMap.decodeView(
                                      negKnownScope) } };
                            addInternalMailOrigin(
                                resultMail,
                                memoryBlock.originInterner, disjNeg,
                                mergeParentView, OriginTag::contradiction,
                                contraDeps, 2,
                                (parameters.compressor_mode
                                    ? parameters.compressor_max_origins_per_expr
                                    : parameters.max_origin_per_expr));
                        }
                    }
                }
            }

            // Live-branch sighting (I-206): every deposit at
            // the branch scope reaches this site, the seed first, so the branch
            // registers as live here for the end-of-burst freeze sweep. The
            // guards keep a frozen branch frozen (equi-class products and the
            // retirement re-fire still route through ordisMerge) and a
            // dead-staged or filtered branch out; the pod-set mint dedups.
            if (memoryBlock.frozenOrBranches.lookup(mergeValId) == 0
                && memoryBlock.pendingDeadOrBranches.lookup(mergeValId) == 0
                && memoryBlock.intValidityNamesToFilter.lookup(mergeValId) == 0) {
                memoryBlock.orLiveBranches.mint(mergeValId);
            }

            // Register this branch's deposit. Single-threaded site — minting into
            // the LB-state space is allowed here. Parent and signature form one
            // canonical cohort id, so identical signatures at different stack
            // positions cannot contribute to each other's convergence count.
            // The (parent, orSig, branchBody) spans feed the interner mints
            // directly (span overloads), no heap string.
            const int32_t mergeExprId =
                memoryBlock.lbStateInterner.encode(addExpression);
            const int32_t mergeParentId =
                memoryBlock.lbStateInterner.encode(mergeParentView);
            const int32_t mergeSigId =
                memoryBlock.lbStateInterner.encode(mergeOrSigView);
            const int32_t mergeCohortId = mintOrCohortId(
                memoryBlock.lbStateInterner, mergeParentId, mergeSigId);
            const int32_t mergeObId = memoryBlock.orBookkeeping.insertSorted(
                LbStatePairKey{ mergeExprId, mergeCohortId },
                memoryBlock.lbStateInterner.encode(mergeBranchBodyView),
                DecodedIdLess{ &memoryBlock.lbStateInterner });

            // Sequenced-release goal probe: the live branch RESOLVES when it
            // derives an expression matching a `toBeProved` goal of the
            // innermost goal scope on its parent chain
            // (D-549) — stage the cohort so the
            // end-of-burst drainPendingOrReleases mints the next pending
            // branch. Non-minting on every path (an un-interned expression
            // cannot be a registered goal — a defined miss, I-83); gated on
            // an existing pending row so cohorts with nothing left to
            // release never stage; the pod-set mint dedups repeats.
            if (memoryBlock.orPendingBranches.lookup(mergeCohortId) != 0
                && memoryBlock.pendingOrReleases.lookup(mergeCohortId) == 0) {
                const NameId goalExprId =
                    memoryBlock.nameMap.lookup(addExpression);
                if (goalExprId != 0) {
                    const NameId goalParentId =
                        memoryBlock.nameMap.lookup(mergeParentView);
                    assert(goalParentId != 0
                        && "ordis branch parent scope must be interned");
                    const NameId goalScope =
                        StatementKnowledge::innermostGoalScope(memoryBlock, goalParentId);
                    if (goalScope != 0 && memoryBlock.intToBeProved.lookup(
                            packStatementKey(goalExprId, goalScope)) != 0)
                        memoryBlock.pendingOrReleases.mint(mergeCohortId);
                }
            }

            // Convergence test.
            const int* mergeCount = memoryBlock.orDisjunctCount.find(mergeCohortId);
            if (mergeCount == nullptr) return;
            const int32_t mergeRunLen = memoryBlock.orBookkeeping.runLen(mergeObId);
            if (static_cast<int>(mergeRunLen) < *mergeCount) return;

            // 1. Promote to the parent scope through the selected output. The
            //    expression + parent-scope key go straight to the StrSpan door
            //    as spans over the stable inputs (addExpression, and the
            //    mergeParentView slice of effectiveValidity) — no transient
            //    ExpressionWithValidity, no heap std::string. Both spans alias
            //    caller-owned ScratchString buffers, NOT the NameMap the door
            //    mints into, so they survive the encode (I-3).
            // Branch-scope prefix "<parent>_boundary_ordis_<orSig>_" on the
            // per-slot string-scratch arena; each branch validity is this prefix
            // plus the disjunct payload. mergeOrSig bytes are recovered from its
            // interned id (post-mint; no lbStateInterner mint occurs after
            // insertSorted, so the span is stable for the rest of the call). The
            // ScratchScope spans both loops below.
            const unsigned slot = (gl::WorkerContext::coreId() >= 0)
                ? static_cast<unsigned>(gl::WorkerContext::coreId())
                : scratchArenas().slotCount() - 1;
            ScratchArena& sArena = scratchArenas().forSlot(slot);
            ScratchScope sScope(sArena);
            const StrSpan mergeOrSigSpan =
                memoryBlock.lbStateInterner.decodeView(mergeSigId);
            const StrSpan BOUNDARY_ORDIS("_boundary_ordis_", 16);
            const int32_t prefixLen =
                mergeParentView.len + BOUNDARY_ORDIS.len + mergeOrSigSpan.len + 1;
            char* prefixBuf = sArena.allocBytes(prefixLen);
            {
                int32_t at = 0;
                std::memcpy(prefixBuf + at, mergeParentView.ptr,
                    static_cast<std::size_t>(mergeParentView.len));
                at += mergeParentView.len;
                std::memcpy(prefixBuf + at, BOUNDARY_ORDIS.ptr,
                    static_cast<std::size_t>(BOUNDARY_ORDIS.len));
                at += BOUNDARY_ORDIS.len;
                std::memcpy(prefixBuf + at, mergeOrSigSpan.ptr,
                    static_cast<std::size_t>(mergeOrSigSpan.len));
                at += mergeOrSigSpan.len;
                prefixBuf[at++] = '_';
                assert(at == prefixLen);
            }
            const StrSpan branchPrefix(prefixBuf, prefixLen);

            // The promoted conclusion depends on every case, so it carries the
            // union of the triggering derivation's levels, every contributing
            // case's current or saved derivation levels, and every refuted
            // case's refutation levels saved at its retirement
            // (D-470).
            int mergedLevels[256];
            int32_t mergedLevelCount = 0;
            const auto mergeLevelRun = [&](const auto& table, int32_t row) {
                for (int32_t k = 0; k < table.runLen(row); ++k) {
                    const int level = table.valueAt(row, k);
                    if (level >= 0) mergedLevelCount = insertLevelSorted(
                        mergedLevels, mergedLevelCount, level, 256);
                }
            };
            for (int32_t k = 0; k < addExpressionLevelCount; ++k) {
                if (addExpressionLevels[k] >= 0) mergedLevelCount = insertLevelSorted(
                    mergedLevels, mergedLevelCount, addExpressionLevels[k], 256);
            }
            // The cases are conditional proofs. Discharging this cohort
            // uses its own OR source, independently of who executed a case.
            const int32_t sourceLevels = memoryBlock.orPendingLevels.lookup(mergeCohortId);
            if (sourceLevels != 0) mergeLevelRun(memoryBlock.orPendingLevels, sourceLevels);
            for (int32_t j = 0; j < mergeRunLen; ++j) {
                const StrSpan disjunct = memoryBlock.lbStateInterner.decodeView(
                    memoryBlock.orBookkeeping.valueAt(mergeObId, j));
                char* buffer = sArena.allocBytes(prefixLen + disjunct.len);
                std::memcpy(buffer, prefixBuf, prefixLen);
                std::memcpy(buffer + prefixLen, disjunct.ptr, disjunct.len);
                const NameId branch = memoryBlock.nameMap.lookup(StrSpan(buffer, prefixLen + disjunct.len));
                const StatementKey key{ contributionId, branch };
                const int32_t current = memoryBlock.intStatementLevelsMap.lookup(key);
                const int32_t saved = memoryBlock.lbMemory.closedOrGoalLevels.lookup(key);
                if (current != 0) mergeLevelRun(memoryBlock.intStatementLevelsMap, current);
                else if (saved != 0) mergeLevelRun(memoryBlock.lbMemory.closedOrGoalLevels, saved);
            }
            const int32_t retiredRow = memoryBlock.orRetiredDisjuncts.lookup(mergeCohortId);
            if (retiredRow != 0) {
                for (int32_t j = 0; j < memoryBlock.orRetiredDisjuncts.runLen(retiredRow); ++j) {
                    const LbStatePairKey record = Codec<LbStatePairKey>::decode(
                        memoryBlock.orRetiredDisjuncts.valueAt(retiredRow, j));
                    const StrSpan disjunct = memoryBlock.lbStateInterner.decodeView(
                        static_cast<int32_t>(record.high));
                    char* buffer = sArena.allocBytes(prefixLen + disjunct.len);
                    std::memcpy(buffer, prefixBuf, prefixLen);
                    std::memcpy(buffer + prefixLen, disjunct.ptr, disjunct.len);
                    const NameId branch = memoryBlock.nameMap.lookup(StrSpan(buffer, prefixLen + disjunct.len));
                    assert(branch != 0 && "a retired case keeps its interned scope");
                    const NameId refutation = memoryBlock.nameMap.lookup(
                        negateOrCaseAssumption(disjunct, sArena));
                    assert(refutation != 0 && "a retired case keeps its interned refutation");
                    const int32_t saved = memoryBlock.lbMemory.closedOrGoalLevels.lookup(
                        StatementKey{ refutation, branch });
                    assert(saved != 0
                        && "a retired case's refutation levels are saved at its retirement");
                    mergeLevelRun(memoryBlock.lbMemory.closedOrGoalLevels, saved);
                }
            }
            if (mergedLevelCount == 0)
                mergedLevels[mergedLevelCount++] = -1;
            insertInternalStatement(resultMail, memoryBlock.nameMap,
                addExpression, mergeParentView, mergedLevels, mergedLevelCount);

            // The seen-disjunct payloads are read straight off the orBookkeeping
            // run in decoded-lex storage order (I-93) via valueAt — byte-identical
            // to the former std::set<std::string> (DecodedIdLess == std::string
            // lex, and lbStateInterner dedups so the ids, hence the strings, are
            // unique). No lbStateInterner mint occurs after insertSorted, and
            // removeExpressionFromMemoryBlock does only non-minting nameMap.lookup,
            // so the decodeView spans and the run stay stable across both loops.

            if (parameters.trackHistory) {
                // D-36 spec'd `or convergence` row layout:
                //   <C> <parent> or convergence <OR> <parent>
                //                                <C> <branch_D1>
                //                                <C> <branch_D2>
                //                                ...
                //                                <C> <branch_DK>
                // The verifier (check_or_convergence) requires each
                // (C, branch_Di) ingredient to have its own chapter row.
                // Branch derivations of C remain in exprOriginMap because
                // removeExpressionFromMemoryBlock(state=0) below touches
                // only intEncodedStatements; buildStack therefore renders
                // them naturally when it recurses on each new ingredient.
                // L3: "or convergence" antecedents into a bounded stack
                // OriginDep[]. Antecedent 0 spans mergeOrSigSpan / mergeParentView
                // directly; each branch antecedent concats branchPrefix + djView
                // onto the string-scratch arena (explicit length), all coexisting
                // under one ScratchScope live to the single door call below. The
                // door mints into originInterner, never these stores (I-3).
                OriginDep mergeDeps[64];
                int mergeDepN = 0;
                mergeDeps[mergeDepN++] = { mergeOrSigSpan, mergeParentView };
                const unsigned mcSlot = (gl::WorkerContext::coreId() >= 0)
                    ? static_cast<unsigned>(gl::WorkerContext::coreId())
                    : scratchArenas().slotCount() - 1;
                ScratchArena& mcArena = scratchArenas().forSlot(mcSlot);
                ScratchScope mcScope(mcArena);
                for (int32_t j = 0; j < mergeRunLen; ++j) {
                    const int32_t djId = memoryBlock.orBookkeeping.valueAt(mergeObId, j);
                    const StrSpan djView = memoryBlock.lbStateInterner.decodeView(djId);
                    const int32_t bvLen = prefixLen + djView.len;
                    char* bvBuf = mcArena.allocBytes(bvLen);
                    std::memcpy(bvBuf, branchPrefix.ptr,
                                static_cast<std::size_t>(prefixLen));
                    std::memcpy(bvBuf + prefixLen, djView.ptr,
                                static_cast<std::size_t>(djView.len));
                    assert(mergeDepN < 64 && "or convergence origin chain exceeds 64");
                    const NameId branch = memoryBlock.nameMap.lookup(StrSpan(bvBuf, bvLen));
                    assert(branch != 0);
                    const StatementKey target{ contributionId, branch };
                    const int64_t* support = memoryBlock.lbMemory.orSupportFacts.find(target);
                    if (support == nullptr) {
                        mergeDeps[mergeDepN++] = { addExpression, StrSpan(bvBuf, bvLen) };
                    } else {
                        const StatementKey fact = Codec<StatementKey>::decode(*support);
                        mergeDeps[mergeDepN++] = { memoryBlock.nameMap.decodeView(fact.orig),
                                                  memoryBlock.nameMap.decodeView(fact.validity) };
                        const int32_t row = memoryBlock.lbMemory.orSupportEqualities.lookup(target);
                        assert(row != 0);
                        for (int32_t k = 0; k < memoryBlock.lbMemory.orSupportEqualities.runLen(row); ++k) {
                            const StatementKey equality = Codec<StatementKey>::decode(
                                memoryBlock.lbMemory.orSupportEqualities.valueAt(row, k));
                            assert(mergeDepN < 64 && "or convergence origin chain exceeds 64");
                            mergeDeps[mergeDepN++] = { memoryBlock.nameMap.decodeView(equality.orig),
                                                      memoryBlock.nameMap.decodeView(equality.validity) };
                        }
                    }
                }
                // Retired-branch ingredients: a reduced cohort (dead-branch
                // retirement shrank the count) still accounts for every
                // flattened disjunct — each retired branch contributes the
                // reductio ingredient (its asserted disjunct's negation at
                // the recorded shallowest known scope) instead of a branch
                // derivation of C. Spans: the negation rides mcArena (live to
                // the door call), the scope is a NameMap decodeView (the door
                // mints into originInterner only, I-3), the disjunct bytes
                // are lbStateInterner cold pages (append-stable).
                {
                    const int32_t retRow =
                        memoryBlock.orRetiredDisjuncts.lookup(mergeCohortId);
                    if (retRow != 0) {
                        const int32_t retN =
                            memoryBlock.orRetiredDisjuncts.runLen(retRow);
                        for (int32_t j = 0; j < retN; ++j) {
                            const LbStatePairKey rec =
                                Codec<LbStatePairKey>::decode(
                                    memoryBlock.orRetiredDisjuncts.valueAt(
                                        retRow, j));
                            const StrSpan retDj =
                                memoryBlock.lbStateInterner.decodeView(
                                    static_cast<int32_t>(rec.high));
                            const StrSpan retNeg = negateOrCaseAssumption(retDj, mcArena);
                            assert(mergeDepN < 64
                                && "or convergence origin chain exceeds 64");
                            mergeDeps[mergeDepN++] = { retNeg,
                                memoryBlock.nameMap.decodeView(
                                    static_cast<NameId>(rec.low)) };
                        }
                    }
                }
                // Shared ancestor support can occur in several cases. Keep
                // each exact cited pair once; case reconstruction is scoped.
                int uniqueCount = 0;
                for (int j = 0; j < mergeDepN; ++j) {
                    bool seen = false;
                    for (int k = 0; k < uniqueCount; ++k)
                        seen = seen || (equalSpans(mergeDeps[j].original, mergeDeps[k].original)
                            && equalSpans(mergeDeps[j].validity, mergeDeps[k].validity));
                    if (!seen) mergeDeps[uniqueCount++] = mergeDeps[j];
                }
                mergeDepN = uniqueCount;
                // KEY as spans (addExpression + the mergeParentView slice).
                addInternalMailOrigin(resultMail,
                          memoryBlock.originInterner,
                          addExpression, mergeParentView,
                          OriginTag::orConvergence, mergeDeps, mergeDepN,
                          (parameters.compressor_mode
                              ? parameters.compressor_max_origins_per_expr
                              : parameters.max_origin_per_expr));
            }

            // 2. Remove `addExpression` from each of the N branch scopes via the
            //    standard removal procedure. The bookkeeping run holds the disjunct
            //    payload bodies (e.g. "((=[2,repl_lev_1_0]))"); each branch
            //    validity "<prefix><dj>" is built on the scratch arena and fed to
            //    the StrSpan door — no transient EncodedExpression, no per-branch
            //    heap std::string.
            const int mergeMaxOrigins = parameters.compressor_mode
                ? parameters.compressor_max_origins_per_expr
                : parameters.max_origin_per_expr;
            for (int32_t j = 0; j < mergeRunLen; ++j) {
                const int32_t djId = memoryBlock.orBookkeeping.valueAt(mergeObId, j);
                const StrSpan djView = memoryBlock.lbStateInterner.decodeView(djId);
                const int32_t bvLen = prefixLen + djView.len;
                char* bvBuf = sArena.allocBytes(bvLen);
                std::memcpy(bvBuf, branchPrefix.ptr,
                    static_cast<std::size_t>(prefixLen));
                std::memcpy(bvBuf + prefixLen, djView.ptr,
                    static_cast<std::size_t>(djView.len));
                // Ship the branch derivation's history BEFORE the removal
                // erases its delta row: fillMailOut walks only the per-step
                // delta, and the cohort's last-released branch converges in
                // the step it derives, so its `equality1` (or other) line
                // would otherwise never reach a descendant that later walks
                // this convergence row (D-286 contract; I-64 sanctioned
                // writer; D-311).
                const NameId branch = memoryBlock.nameMap.lookup(StrSpan(bvBuf, bvLen));
                const StatementKey target{ contributionId, branch };
                const int64_t* support = memoryBlock.lbMemory.orSupportFacts.find(target);
                if (support != nullptr) {
                    const StatementKey fact = Codec<StatementKey>::decode(*support);
                    proofHistory.copyOriginRowsToMailOut(memoryBlock, memoryBlock.nameMap.decodeView(fact.orig),
                        memoryBlock.nameMap.decodeView(fact.validity), mergeMaxOrigins);
                    const int32_t row = memoryBlock.lbMemory.orSupportEqualities.lookup(target);
                    assert(row != 0);
                    for (int32_t k = 0; k < memoryBlock.lbMemory.orSupportEqualities.runLen(row); ++k) {
                        const StatementKey equality = Codec<StatementKey>::decode(
                            memoryBlock.lbMemory.orSupportEqualities.valueAt(row, k));
                        proofHistory.copyOriginRowsToMailOut(memoryBlock, memoryBlock.nameMap.decodeView(equality.orig),
                            memoryBlock.nameMap.decodeView(equality.validity), mergeMaxOrigins);
                    }
                } else {
                    proofHistory.copyOriginRowsToMailOut(memoryBlock, addExpression,
                        StrSpan(bvBuf, bvLen), mergeMaxOrigins);
                }
                if (rowRemovals != nullptr) {
                    rowRemovals->stage(addExpression, StrSpan(bvBuf, bvLen), memoryBlock);
                } else {
                    gl::StatementRegistryOperations::removeExpressionFromMemoryBlock(
                        addExpression, StrSpan(bvBuf, bvLen),
                        memoryBlock, /*state=*/0);
                }
            }
        }
}
