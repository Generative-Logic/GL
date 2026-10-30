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
class BlockKeyReader;
class CaseProcessor;
class ImplicationSubproofs;
class ProofHistory;
class ScopeLifecycle;
class StatementMemory;
namespace tests { struct GoalProcessorFixture; }
/// @brief Goal Processor operations for the proof algorithm.
/// @details Uses explicit peer interfaces while the supplied logical blocks retain storage and proof state.
class GoalProcessor {
friend struct tests::GoalProcessorFixture;
public:
/// @brief Bind inputs for goal processor.
/// @details Construction stores explicit references; block storage and all invocation lifetimes remain with their owners.
/// @param canonicalizer Borrowed Canonicalizer interface or state; must outlive this component.
/// @param contradictionTable Borrowed std::vector interface or state; must outlive this component.
/// @param definitions Borrowed CompiledDefinitionRegistry interface or state; must outlive this component.
/// @param inductionMemoryBlocks Borrowed std::vector interface or state; must outlive this component.
/// @param inductionMemoryBlocksMutex Borrowed std::mutex interface or state; must outlive this component.
/// @param parameters Borrowed ProverParameters interface or state; must outlive this component.
/// @param updateGlobalMutex Borrowed std::mutex interface or state; must outlive this component.
/// @param updateGlobalSingleChapterMutex Borrowed std::mutex interface or state; must outlive this component.
/// @param updateGlobalSingleChapterPages Borrowed std::vector interface or state; must outlive this component.
/// @param updateGlobalTuples Borrowed std::vector interface or state; must outlive this component.
/// @param blockKeyReader Borrowed BlockKeyReader interface or state; must outlive this component.
/// @param caseProcessor Borrowed CaseProcessor interface or state; must outlive this component.
/// @param implicationSubproofs Borrowed ImplicationSubproofs interface or state; must outlive this component.
/// @param proofHistory Borrowed ProofHistory interface or state; must outlive this component.
/// @param scopeLifecycle Borrowed ScopeLifecycle interface or state; must outlive this component.
/// @param statementMemory Borrowed StatementMemory interface or state; must outlive this component.
/// @return None; construction binds the supplied references.
GoalProcessor(Canonicalizer& canonicalizer, std::vector<ContradictionItem>& contradictionTable, const CompiledDefinitionRegistry& definitions, std::vector<Memory*>& inductionMemoryBlocks, std::mutex& inductionMemoryBlocksMutex, const ProverParameters& parameters, std::mutex& updateGlobalMutex, std::mutex& updateGlobalSingleChapterMutex, std::optional<SealedPageSet>& updateGlobalSingleChapterPages, std::vector< std::tuple<int, uint64_t, int>>& updateGlobalTuples, BlockKeyReader& blockKeyReader, CaseProcessor& caseProcessor, ImplicationSubproofs& implicationSubproofs, ProofHistory& proofHistory, ScopeLifecycle& scopeLifecycle, StatementMemory& statementMemory);

private:
/// @brief Parse a `contradiction_`-marked scope payload into its
        ///        product — the negated compact the scope proves.
        ///
        /// @details
        /// A contradiction-integration scope's payload is
        /// `contradiction_!(X[args])` (see
        /// `prepareNegatedCompoundContradiction`). This parser returns the
        /// product `!(X[args])` as a zero-copy slice of @p payload, or the
        /// empty span when the payload is not contradiction-marked (wrong
        /// prefix, or the tail is not a negated form) — the empty span is a
        /// defined "not this scope kind" verdict, mirroring
        /// `splitSubproofPayload`'s false verdict, not a failure fallback.
        ///
        /// @param payload A scope node's payload (the `decodeSubView` of its
        ///                `stackBack`).
        /// @return The product slice, or the empty span when unmarked.
        /// @invariant Pure; allocates nothing; the result aliases @p payload.
        /// @see prepareNegatedCompoundContradiction,
        ///      dischargeContradictionScopes.
static StrSpan contradictionScopeProductView(StrSpan payload);

public:
/// @brief Single per-elementary-step sweep that detects an in-scope
        ///        contradiction across the LB's whole `intEncodedStatements`
        ///        set and fires the matching discharge reaction.
        ///
        /// @details
        /// Handles the incubator and vacuous-truth contradiction reactions
        /// (whose fired heads DO enter `intEncodedStatements`, so a post-burst
        /// sweep can see them). It consolidates two of the three reactions
        /// that formerly sat as the per-insertion "Site G" block inside
        /// `addExprToMemoryBlock` (`logical_block/goal_processor.hpp`). Site G fired only at
        /// insertion time on the incoming expression, so it missed a
        /// contradiction when the shallower (ancestor-scope) negation was
        /// added AFTER the deeper positive — neither insertion's ancestor scan
        /// saw the other. This sweep re-checks every statement each step, so
        /// detection is order-independent.
        ///
        /// For the first contradicting statement found — a positive `expr`
        /// at exact `main` whose negation is also proved at `main` —
        /// exactly one of four mutually-exclusive LB roles reacts, the LB
        /// is deactivated, and the sweep returns (fire-once):
        ///   1. `primedForContradiction` (incubator) — record a
        ///      `contradiction` origin for `!`+cleanOp @ "main" (cleanOp is
        ///      `exprKey` with the `__contradiction__` prefix stripped) and
        ///      broadcast the contradiction theorem via `updateGlobalSingleChapter`,
        ///      then clear the theorem. D-51: the record stays local to this
        ///      LB — no ancestor propagation, no `mailOut`.
        ///   2. `contradictionIndex >= 0` (CE filter) — the conjecture is
        ///      refuted: mark its `contradictionTable` row.
        ///   3. `isPartOfRecursion` at validity "main" (vacuous truth) —
        ///      record a `vacuous truth` origin for the first main goal of
        ///      `toBeProved` (deps: expr, negation, the recursion hypothesis)
        ///      and close that goal directly through `closeRecursionGoal`,
        ///      with every level of the block. The goal is not deposited as a
        ///      statement: the deposit door respells it under the block's
        ///      classes (a zero block's own premise `(=[n,0])` respells n),
        ///      where `dischargeToBeProved`'s exact-key match would miss it
        ///      (D-573).
        ///   4. No role (a NORMAL premise LB) — the pair was derived from
        ///      the premises alone, so the premise set is inconsistent and
        ///      everything at or below this LB is vacuous: set
        ///      `mainContradiction` (read by the post-join emission gates
        ///      via `hasContradictedAncestor`) and stage the subtree for
        ///      the post-join deactivation drain. At the batch root or an
        ///      anchor premise LB this state asserts — the axioms are
        ///      consistent, so it can only mean a broken definition set.
        ///
        /// Wired into `standardProcessing` directly before
        /// `dischargeToBeProved`. `standardProcessing` is driven in both the
        /// pre-burst (`performElemPhase1`) and post-burst
        /// (`performElemPhase3`) phases, and `proveKernel` does not
        /// re-filter the post-burst sweep by `isActive`; the leading
        /// `!isActive` early return makes the second pass over an LB this
        /// sweep already closed a no-op (no double theorem / double head /
        /// double origin row). That early return is contracted control flow
        /// — an inactive LB has no contradiction to discharge — not a banned
        /// defensive fallback (I-19).
        ///
        /// @param memoryBlock The LB under inspection. On a hit it is
        ///        deactivated and one of: `exprOriginMap` +
        ///        `contradictionTheoremId` (incubator); `exprOriginMap` + the
        ///        closed goal (vacuous); or the conjecture's
        ///        `contradictionTable` row (CE-filter) is mutated.
        /// @param coreId Worker core id, forwarded to `updateGlobalSingleChapter`
        ///        and carried by the vacuous closure's induction tuples.
        /// @invariant Fires at most once per call and is a no-op when
        ///            `!memoryBlock.isActive`. Runs in the serial
        ///            `standardProcessing` phase, so `nameMap.encode` minting
        ///            is safe. Handles four LB roles: incubator
        ///            (`primedForContradiction`), CE-filter
        ///            (`contradictionIndex >= 0`), vacuous-truth
        ///            (`isPartOfRecursion`), and the normal-LB
        ///            vacuous-premise branch (no role).
        /// @see `closeRecursionGoal` — the induction closure the vacuous-truth
        ///      branch shares with `dischargeToBeProved`.
        /// @see `burstDeactivates` — the read-only phase-2 mirror that stops a
        ///      doomed LB's hashburst early (incubator / vacuous / CE-filter).
inline void dischargeContradiction(Memory& memoryBlock, int coreId);

public:
/// @brief Discharge `contradiction_`-marked scopes: when a per-step
        ///        delta statement at/below such a scope has its negation known
        ///        at a comparable scope, emit the scope's product at the
        ///        scope's parent and wipe the scope.
        ///
        /// @details
        /// The discharge half of contradiction-based integration
        /// (`prepareNegatedCompoundContradiction` is the priming half). The
        /// walk mirrors `dischargeToBeProved`'s delta snapshot: for every new
        /// local statement whose scope chain carries a `contradiction_`
        /// payload node, the statement's negation is probed (known-bit) at
        /// every scope on the statement's chain up to and including `"main"`.
        /// A hit refutes the scope's assumption: the product
        /// (`contradictionScopeProductView` of the marked node's payload) is
        /// emitted as an internal-mail statement at the marked node's PARENT
        /// with a `contradiction` history row citing the two opposing
        /// expressions, and the scope subtree is queued for the end-of-burst
        /// wipe (`intValidityNamesToFilter` + `pendingWipeScopes` — the
        /// scoped-goal-closure cleanup pair). `pendingWipeScopes` membership
        /// doubles as the once-per-step guard.
        ///
        /// Deliberate scoped relaxation of the main/main contradiction
        /// contract ([I-78](../../docs/agentic_swdd/30_invariants.md#i-78)):
        /// it fires ONLY for scopes carrying the `contradiction_` marker,
        /// never for the LB-level discharge; precedent is the
        /// refuted-disjunct retirement (D-242).
        ///
        /// @param memoryBlock     The LB whose statement delta is scanned.
        /// @param internalMailOut The step's internal-mail channel — the
        ///                        emission rides the standard absorb, which
        ///                        sets the product's `registered` bit.
        /// @invariant Heap-free; probes non-minting; single-threaded (the
        ///            phase-3 seam, like `dischargeToBeProved`).
        /// @see prepareNegatedCompoundContradiction, dischargeToBeProved,
        ///      contradictionScopeProductView.
inline void dischargeContradictionScopes(Memory& memoryBlock,
                                                 ColdMail& internalMailOut);

public:
/// @brief Prove an ordinary implication goal from contradictory visible facts.
        ///
        /// @details
        /// Inspect unresolved positive implication scopes in decoded order. One
        /// opposing fact must belong to the exact goal scope; its partner may
        /// belong to that scope or an ancestor. Record the scoped head with both
        /// actual dependencies and their combined levels. Normal goal discharge
        /// then records the compact implication and schedules its scope cleanup.
        /// The known registry covers both fact arrival orders without consulting
        /// proof-history containers.
        ///
        /// @param memoryBlock Active logical block containing unresolved goals.
        /// @return Nothing; proven heads enter the ordinary local statement delta.
        /// @invariant No main, sibling, descendant-only, OR, or reductio closure.
        /// @invariant Heap-free; copied strings survive NameMap mints.
        /// @see dischargeToBeProved, dischargeContradiction, materializeWatchHits.
inline void dischargeVacuousImplicationScopes(Memory& memoryBlock);

public:

inline void dischargeToBeProved(Memory& memoryBlock, int coreId,
                                        ColdMail& internalMailOut);

public:
/// @brief Close one main goal of an induction block: report its induction
///        tuples, erase it and switch the block off
///        (D-573).
///
/// @details
/// The induction closure shared by `dischargeToBeProved` (a new statement at
/// the goal's exact key) and the vacuous-truth branch of
/// `dischargeContradiction` (a contradiction at main). Every auxy of the
/// goal's run, in its stored ascending order, is reported to the induction
/// promotion as `(auxy, levelsMask, coreId)` in `updateGlobalTuples`; the goal
/// is erased from `intToBeProved` and the block deactivated. A step block
/// (key `(in2[rec`) is queued in `inductionMemoryBlocks` for the step-win
/// seam (`ProofBarrier::settleStepWins`).
///
/// @param memoryBlock The induction block (`isPartOfRecursion`).
/// @param goalKey     The goal's packed key: an open goal at `main`.
/// @param levelsMask  The levels the closure used, one bit per level, none
///                    above the block's level.
/// @param coreId      Worker core id carried by the tuples.
/// @return Nothing.
/// @invariant Not in compressor mode; each push runs under its mutex.
/// @see dischargeToBeProved, dischargeContradiction,
///      ScopeLifecycle::settleStepClosure.
inline void closeRecursionGoal(Memory& memoryBlock, int64_t goalKey,
                                       uint64_t levelsMask, int coreId);

private:
/// @brief Does an integration-gate template denote @p goal — verbatim
        ///        or as one of its marker forms?
        ///
        /// @details
        /// `prepareIntegration` keys its `integrationPrepared` /
        /// `integrationPreparedMarker` / `integrationStartIntMap` rows by the
        /// prepared expression and by its MARKER form (each changeable
        /// argument replaced by the literal `marker`; with zero changeable
        /// arguments the marker form is the expression verbatim). The disproof
        /// cleanup cannot re-derive the exact unchangeable-argument set the
        /// original preparation used, but it does not need to: a template
        /// matches the goal iff the (optionally `!`-negated) core names agree,
        /// the arities agree, and every template argument equals the goal's
        /// argument at that slot or the literal `marker`. That covers the
        /// verbatim key and every possible marker variant, and nothing else —
        /// a template of a DIFFERENT expression differs in core, arity, or a
        /// non-`marker` argument.
        ///
        /// @param tmpl The gate template (decoded `templateInterner` text).
        /// @param goal The goal expression (verbatim).
        /// @return `true` iff @p tmpl is the goal or one of its marker forms.
        /// @see drainDisprovedGoals — the sole production consumer.
static inline bool templateMatchesGoalWithMarkers(StrSpan tmpl,
                                                          StrSpan goal);

private:
/// @brief Whether an expression is "proved" — carries no unresolved
        ///        iteration lexeme and no `c<digit>` marker.
        ///
        /// @details
        /// The regex-free twin of the former two `regex_search` existence
        /// tests (`it_\d+_lev_\d+_` and `c\d+`): an expression is proved iff
        /// NEITHER pattern occurs. Uses the `str_ops` existence scanners
        /// `containsItLevPrefixShape` / `containsCDigit`, byte-identical verdicts
        /// to the regex form; the regex machinery survives only as the in-test
        /// oracle (`test_memory.cpp::is_proved_matches_regex_oracle`).
        ///
        /// @param s Expression view to test.
        /// @return `true` iff neither the `it_\d+_lev_\d+_` nor the `c\d+`
        ///         pattern occurs.
        /// @see `str_ops::containsItLevPrefixShape`, `str_ops::containsCDigit`.
inline bool isProved(StrSpan s);

public:
/// @brief End-of-burst drain of the settled-goal inbox — close the
///        MAIN goal a contradiction twin settled: a disproved goal
///        loses its whole integration machinery, a proved goal closes
///        with the ordinary success-path semantics.
///
/// @details
/// The post-join theorem drain deposits the contradiction seed of every
/// emitting primed `__contradiction__` LB onto its parent's
/// `Memory::pendingDisprovedGoals` (the theorem head is `negate(seed)`
/// by I-165). This drain — running single-threaded at the end of the
/// parent's own burst, immediately BEFORE the `pendingWipeScopes`
/// drain so the wipes land in the same burst — probes each seed
/// against the MAIN goals in `intToBeProved`, in both directions.
///
/// PROOF direction (D-279): when
/// `negate(seed)` is a registered MAIN goal, the emitted theorem's
/// head IS that goal — the goal succeeded. The drain erases the goal
/// row and queues the goal-embedding MAIN scope roots (the same three
/// shapes as below) on `pendingWipeScopes` +
/// `intValidityNamesToFilter`, and does nothing else: no origin
/// erase (Rule 16 / I-44 — a proved goal's history stays), no
/// gate/cohort erase, no closed-subproof exception. Emission and twin
/// retirement already happened at the deposit seam. Closing the row
/// lets the I-48 sweeps retire the goal's LB chain even when the
/// level-gated local discharge could never fire (a
/// contradiction-scope proof may legitimately carry a narrow level
/// run).
///
/// DISPROOF direction: when the seed itself is a registered MAIN
/// goal, the verbatim twin fired — the goal is disproved and the
/// cleanup below triggers. A double miss is a defined no-op (a goal
/// already closed):
///
/// 1. the goal row leaves `intToBeProved`;
/// 2. every MAIN-level scope whose payload embeds the goal — the
///    `<goal>_subproof_<bare>` subproof/orint scopes
///    (I-170) and the `_var…_hypo_<goal>` working
///    scope — is queued on `pendingWipeScopes` (+ the synchronous
///    `intValidityNamesToFilter` insert, D-73), so the existing
///    radical subtree wipe removes the whole nested machinery.
///    EXCEPTION: a subproof root whose product (the bare compact /
///    OR signature) is a REGISTERED statement at main is skipped
///    whole — that subproof CLOSED before the disproof, its scope was
///    already wiped by the success path, its product is a legitimate
///    proven fact, and its preserved history is what the chapter
///    walker reads;
/// 3. the goal-template gates leave `integrationPrepared` /
///    `integrationPreparedMarker` (MAIN rows matching the goal, its
///    marker forms, or a goal-prefixed template) and
///    `integrationStartIntMap`;
/// 4. `exprOriginMap` rows keyed in the wiped subtrees, plus MAIN rows
///    whose key CONTAINS the goal text — excepting the disproof
///    products `negate(goal)` and the anchor-rooted theorem rows —
///    are erased (the maintainer-approved deviation from the
///    wipeSubtree I-44 preserve rule — the disproof's own record
///    lives in the `__contradiction__` LB per D-51);
/// 5. internal-mail origin runs at wiped scopes are erased
///    (`ColdMail::filterOrigins`; the statements are filtered by the
///    subtree wipe itself) so no orphaned history rows outlive their
///    statements;
/// 6. OR cohort bookkeeping (`orDisjunctCount` / `orBookkeeping`)
///    whose cohort parent validity lies in a wiped subtree is erased —
///    a dead parent scope means the cohort can never act again.
///
/// Known limitation (flagged, not silent): the seed is matched
/// verbatim in both directions; a MAIN goal rewritten by equivalence
/// classes between spawn and settlement would miss the
/// `intToBeProved` probe. Incubator batches run with classes skipped,
/// so the contradiction pipeline never hits this; see the chapter
/// Weaknesses entry.
///
/// @param memoryBlock The LB whose inbox is drained (its containers
///                    are resident — the drain runs inside the LB's
///                    own burst).
/// @invariant Deterministic: seeds process in byte-lex order; every
///            erase predicate is a pure function of container content.
/// @see Memory::pendingDisprovedGoals, Memory::wipeSubtree,
///      splitSubproofPayload, templateMatchesGoalWithMarkers.
void drainDisprovedGoals(Memory& memoryBlock);
private:

Canonicalizer& canonicalizer;
std::vector<ContradictionItem>& contradictionTable;
const CompiledDefinitionRegistry& definitions;
std::vector<Memory*>& inductionMemoryBlocks;
std::mutex& inductionMemoryBlocksMutex;
const ProverParameters& parameters;
std::mutex& updateGlobalMutex;
std::mutex& updateGlobalSingleChapterMutex;
std::optional<SealedPageSet>& updateGlobalSingleChapterPages;
std::vector< std::tuple<int, uint64_t, int>>& updateGlobalTuples;
BlockKeyReader& blockKeyReader;
CaseProcessor& caseProcessor;
ImplicationSubproofs& implicationSubproofs;
ProofHistory& proofHistory;
ScopeLifecycle& scopeLifecycle;
StatementMemory& statementMemory;
};
}
#include "../logical_block/block_key_reader.hpp"
#include "../logical_block/case_processor.hpp"
#include "../logical_block/implication_subproofs.hpp"
#include "../logical_block/proof_history.hpp"
#include "../logical_block/scope_lifecycle.hpp"
#include "../logical_block/statement_memory.hpp"
namespace gl {

/// @brief Parse a `contradiction_`-marked scope payload into its
        ///        product — the negated compact the scope proves.
        ///
        /// @details
        /// A contradiction-integration scope's payload is
        /// `contradiction_!(X[args])` (see
        /// `prepareNegatedCompoundContradiction`). This parser returns the
        /// product `!(X[args])` as a zero-copy slice of @p payload, or the
        /// empty span when the payload is not contradiction-marked (wrong
        /// prefix, or the tail is not a negated form) — the empty span is a
        /// defined "not this scope kind" verdict, mirroring
        /// `splitSubproofPayload`'s false verdict, not a failure fallback.
        ///
        /// @param payload A scope node's payload (the `decodeSubView` of its
        ///                `stackBack`).
        /// @return The product slice, or the empty span when unmarked.
        /// @invariant Pure; allocates nothing; the result aliases @p payload.
        /// @see prepareNegatedCompoundContradiction,
        ///      dischargeContradictionScopes.
inline StrSpan GoalProcessor::contradictionScopeProductView(StrSpan payload) {
            const int32_t pfxLen =
                static_cast<int32_t>(sizeof(gl::ScopeSyntax::kContradictionScopePrefix) - 1);
            if (payload.len <= pfxLen + 1) return StrSpan();
            if (std::memcmp(payload.ptr, gl::ScopeSyntax::kContradictionScopePrefix,
                            static_cast<size_t>(pfxLen)) != 0) {
                return StrSpan();
            }
            if (payload.ptr[pfxLen] != '!' || payload.ptr[pfxLen + 1] != '(') {
                return StrSpan();
            }
            return StrSpan(payload.ptr + pfxLen, payload.len - pfxLen);
        }

/// @brief Single per-elementary-step sweep that detects an in-scope
        ///        contradiction across the LB's whole `intEncodedStatements`
        ///        set and fires the matching discharge reaction.
        ///
        /// @details
        /// Handles the incubator and vacuous-truth contradiction reactions
        /// (whose fired heads DO enter `intEncodedStatements`, so a post-burst
        /// sweep can see them). It consolidates two of the three reactions
        /// that formerly sat as the per-insertion "Site G" block inside
        /// `addExprToMemoryBlock` (`logical_block/goal_processor.hpp`). Site G fired only at
        /// insertion time on the incoming expression, so it missed a
        /// contradiction when the shallower (ancestor-scope) negation was
        /// added AFTER the deeper positive — neither insertion's ancestor scan
        /// saw the other. This sweep re-checks every statement each step, so
        /// detection is order-independent.
        ///
        /// For the first contradicting statement found — a positive `expr`
        /// at exact `main` whose negation is also proved at `main` —
        /// exactly one of four mutually-exclusive LB roles reacts, the LB
        /// is deactivated, and the sweep returns (fire-once):
        ///   1. `primedForContradiction` (incubator) — record a
        ///      `contradiction` origin for `!`+cleanOp @ "main" (cleanOp is
        ///      `exprKey` with the `__contradiction__` prefix stripped) and
        ///      broadcast the contradiction theorem via `updateGlobalSingleChapter`,
        ///      then clear the theorem. D-51: the record stays local to this
        ///      LB — no ancestor propagation, no `mailOut`.
        ///   2. `contradictionIndex >= 0` (CE filter) — the conjecture is
        ///      refuted: mark its `contradictionTable` row.
        ///   3. `isPartOfRecursion` at validity "main" (vacuous truth) —
        ///      record a `vacuous truth` origin for the first main goal of
        ///      `toBeProved` (deps: expr, negation, the recursion hypothesis)
        ///      and close that goal directly through `closeRecursionGoal`,
        ///      with every level of the block. The goal is not deposited as a
        ///      statement: the deposit door respells it under the block's
        ///      classes (a zero block's own premise `(=[n,0])` respells n),
        ///      where `dischargeToBeProved`'s exact-key match would miss it
        ///      (D-573).
        ///   4. No role (a NORMAL premise LB) — the pair was derived from
        ///      the premises alone, so the premise set is inconsistent and
        ///      everything at or below this LB is vacuous: set
        ///      `mainContradiction` (read by the post-join emission gates
        ///      via `hasContradictedAncestor`) and stage the subtree for
        ///      the post-join deactivation drain. At the batch root or an
        ///      anchor premise LB this state asserts — the axioms are
        ///      consistent, so it can only mean a broken definition set.
        ///
        /// Wired into `standardProcessing` directly before
        /// `dischargeToBeProved`. `standardProcessing` is driven in both the
        /// pre-burst (`performElemPhase1`) and post-burst
        /// (`performElemPhase3`) phases, and `proveKernel` does not
        /// re-filter the post-burst sweep by `isActive`; the leading
        /// `!isActive` early return makes the second pass over an LB this
        /// sweep already closed a no-op (no double theorem / double head /
        /// double origin row). That early return is contracted control flow
        /// — an inactive LB has no contradiction to discharge — not a banned
        /// defensive fallback (I-19).
        ///
        /// @param memoryBlock The LB under inspection. On a hit it is
        ///        deactivated and one of: `exprOriginMap` +
        ///        `contradictionTheoremId` (incubator); `exprOriginMap` + the
        ///        closed goal (vacuous); or the conjecture's
        ///        `contradictionTable` row (CE-filter) is mutated.
        /// @param coreId Worker core id, forwarded to `updateGlobalSingleChapter`
        ///        and carried by the vacuous closure's induction tuples.
        /// @invariant Fires at most once per call and is a no-op when
        ///            `!memoryBlock.isActive`. Runs in the serial
        ///            `standardProcessing` phase, so `nameMap.encode` minting
        ///            is safe. Handles four LB roles: incubator
        ///            (`primedForContradiction`), CE-filter
        ///            (`contradictionIndex >= 0`), vacuous-truth
        ///            (`isPartOfRecursion`), and the normal-LB
        ///            vacuous-premise branch (no role).
        /// @see `closeRecursionGoal` — the induction closure the vacuous-truth
        ///      branch shares with `dischargeToBeProved`.
        /// @see `burstDeactivates` — the read-only phase-2 mirror that stops a
        ///      doomed LB's hashburst early (incubator / vacuous / CE-filter).
inline void GoalProcessor::dischargeContradiction(Memory& memoryBlock, int coreId) {
            // Idempotency across the phase-1 / phase-3 double-run: phase 3 is
            // not re-filtered by isActive, so an LB this sweep closed in
            // phase 1 is re-entered in phase 3. An inactive LB has nothing to
            // discharge.
            if (!memoryBlock.isActive) return;

            const int maxOrig = parameters.compressor_mode
                ? parameters.compressor_max_origins_per_expr
                : parameters.max_origin_per_expr;

            // Whole-set sweep, insertion-ordered (deterministic in the serial
            // phase). Fire on the first contradiction, react, deactivate,
            // return. The return is also required for iteration safety: the
            // vacuous-truth reaction's addStatement may push_back to the
            // statement registry, which this index loop must not walk into.
            for (int32_t si = 0; si < memoryBlock.intEncodedStatements.size(); ++si) {
                const IntEncodedExpr& ie = memoryBlock.intEncodedStatements[si];

                // Per-iteration string window on the string-scratch tier. Every
                // decoded / fresh name below rides sArena and dies at this
                // scope's close, so the sweep visits every statement without
                // unbounded arena growth (the heap version freed each iteration
                // via RAII; here both the no-contradiction `continue` and every
                // discharge `return` rewind iterScope). LOAD-BEARING.
                const unsigned slot = (coreId >= 0)
                    ? static_cast<unsigned>(coreId)
                    : scratchArenas().slotCount() - 1;
                ScratchArena& sArena = scratchArenas().forSlot(slot);
                ScratchScope iterScope(sArena);

                // Copies onto the string tier, not decodeView spans: encode(
                // negation) and addStatement below mint the NameMap and can
                // reallocate its id-to-name store (I-3); a raw span would
                // dangle. sArena is a DIFFERENT arena than the NameMap, so the
                // copies survive the mint.
                const StrSpan exprView = memoryBlock.nameMap.decodeView(ie.originalId);
                const ScratchString expr =
                    ScratchString::copyFrom(sArena, exprView.ptr, exprView.len);
                const StrSpan validityView = memoryBlock.nameMap.decodeView(ie.validityId);
                const ScratchString validityName =
                    ScratchString::copyFrom(sArena, validityView.ptr, validityView.len);

                // D-222: a full LB contradiction
                // requires BOTH sides at the shallowest scope, `main`. A
                // branch-local copy is conditional and cannot discharge the
                // LB; if the same expression also exists at `main`, its own
                // main row is encountered separately by this sweep.
                if (ie.validityId != NameMap::MAIN_ID) continue;

                // negation rides sArena (negateScratch's arena, NOT the
                // NameMap), so the bare span survives every NameMap mint below;
                // nothing rewinds sArena between here and its last use (the
                // vacuous branch's dcRowScope marks ABOVE it). LOAD-BEARING.
                const StrSpan negation = gl::DefinitionExpansion::negateScratch(sArena, StrSpan(expr));
                const NameId negOrigId = memoryBlock.nameMap.encode(negation);
                const StatementFlags* mainNegation =
                    memoryBlock.intKnownStatements.find(
                        StatementKey{ negOrigId, NameMap::MAIN_ID });
                if (mainNegation == nullptr) continue;

                // ---- Incubator contradiction: record provenance, emit theorem ----
                if (memoryBlock.primedForContradiction) {
                    const NameId theoremId = memoryBlock.contradictionTheoremId;
                    memoryBlock.contradictionTheoremId = 0;
                    memoryBlock.isActive = false;
                    // The conjecture is settled by this fire — retire the
                    // polarity sibling (staged, drained post-join; I-28).
                    scopeLifecycle.stageContradictionSiblingTwin(memoryBlock, sArena);
                    if (theoremId != 0) {
                        // decodeView (not copyFrom): between here and the
                        // SealedString::copyFrom below NO statement mints the
                        // NameMap — the cleanOp/mbKey work reads the frozen
                        // skeletonInterner (D-186) + slices spans, negCleanOp
                        // byte-bumps sArena, and addOriginEncoded mints
                        // originInterner only. So the span never dangles (I-3).
                        const StrSpan theorem =
                            memoryBlock.nameMap.decodeView(theoremId);
                        if (parameters.trackHistory) {
                            // Extract cleanOp from exprKey: "__contradiction__(op[...])"
                            // — a zero-copy sub-span of the frozen exprKeyView.
                            const StrSpan prefix("__contradiction__", 17);
                            const StrSpan mbKey = memoryBlock.exprKeyView();
                            StrSpan cleanOp(mbKey.ptr, 0);
                            if (mbKey.len >= prefix.len
                                && equalSpans(StrSpan(mbKey.ptr, prefix.len), prefix)) {
                                cleanOp = StrSpan(mbKey.ptr + prefix.len,
                                                  mbKey.len - prefix.len);
                            }
                            // negCleanOp = gl::DefinitionExpansion::negate(cleanOp): '!'-prefix for a
                            // positive seed (byte-identical to the historical
                            // "!" + cleanOp build) or the leading-'!' slice for
                            // a negated seed (a complement LB), so the record
                            // always cites the THEOREM HEAD, never a raw double
                            // negation (I-165).
                            // Lifetime: the prefix branch byte-bumps sArena; the
                            // slice branch aliases the frozen exprKeyView — both
                            // stable across the originInterner mint below (I-3).
                            const StrSpan negCleanOp = gl::DefinitionExpansion::negateScratch(sArena, cleanOp);
                            // L3: the contradiction RECORD antecedents are all
                            // in-hand spans (expr / negation / validityName on
                            // sArena; cleanOp on the frozen skeletonInterner;
                            // "main" a static literal); the door mints
                            // originInterner, never these arenas, so no span
                            // dangles across the mint (I-3).
                            const StrSpan cMain("main", 4);
                            const OriginDep cDeps[3] = {
                                { StrSpan(expr), StrSpan(validityName) },
                                { StrSpan(negation), StrSpan(validityName) },
                                { cleanOp, cMain } };
                            // D-51: contradiction record stays IN this
                            // __contradiction__ LB only — no upward
                            // propagation to ancestors, no broadcast via
                            // mailOut. The chapter walker (buildStack) finds
                            // it by switching INTO this LB via the ancestor's
                            // SimpleMapStore edge.
                            addOriginEncoded(memoryBlock.exprOriginMap,
                                memoryBlock.originInterner, negCleanOp,
                                cMain, OriginTag::contradiction, cDeps, 3,
                                maxOrig);
                        }
                        // Registration verdict
                        // (D-278): union the
                        // colliding pair's derivation levels — the levels the
                        // reductio genuinely consumed — EXCLUDING this twin's
                        // own level (the assumed seed's tier, deposited at
                        // {level} on creation). The theorem's premise chain
                        // occupies levels 0..level-1 in all three twin
                        // geometries, so the same count-based full-run test as
                        // dischargeToBeProved applies: {0..level-1}, or
                        // {1..level-1} for an anchor-free reductio. Computed
                        // here from the twin's own resident rows — a
                        // drain-time re-probe could miss (end-of-burst
                        // sanitize rewrites statement text; the LB may be
                        // cold post-join). The verdict gates ONLY the global
                        // registration at the drain; the settled-goal deposit
                        // and every lifecycle step run regardless.
                        bool twinAllLevels = false;
                        {
                            const int32_t posId =
                                memoryBlock.intStatementLevelsMap.lookup(
                                    packStatementKey(ie.originalId,
                                                     NameMap::MAIN_ID));
                            const int32_t negId =
                                memoryBlock.intStatementLevelsMap.lookup(
                                    packStatementKey(negOrigId,
                                                     NameMap::MAIN_ID));
                            assert(posId != 0
                                && memoryBlock.intStatementLevelsMap.runLen(posId) > 0
                                && "dischargeContradiction: colliding statement "
                                   "without a non-empty level row");
                            assert(negId != 0
                                && memoryBlock.intStatementLevelsMap.runLen(negId) > 0
                                && "dischargeContradiction: collision partner "
                                   "without a non-empty level row");
                            int lvRun[256];
                            int32_t lvN = 0;
                            int run[256];
                            const int32_t nPos = coldIntRunNonNegAt(
                                memoryBlock.intStatementLevelsMap, posId,
                                run, 256);
                            for (int32_t j = 0; j < nPos; ++j)
                                if (run[j] != memoryBlock.level)
                                    lvN = insertLevelSorted(lvRun, lvN,
                                                            run[j], 256);
                            const int32_t nNeg = coldIntRunNonNegAt(
                                memoryBlock.intStatementLevelsMap, negId,
                                run, 256);
                            for (int32_t j = 0; j < nNeg; ++j)
                                if (run[j] != memoryBlock.level)
                                    lvN = insertLevelSorted(lvRun, lvN,
                                                            run[j], 256);
                            // Nothing inside a twin may carry a level above
                            // its own: compact rules ship empty level sets
                            // (I-51) and broadcasts never target
                            // __contradiction__ routing keys.
                            assert(lvN <= memoryBlock.level
                                && "dischargeContradiction: collision level "
                                   "union wider than the twin's premise chain");
                            const bool has0 = lvN > 0 && lvRun[0] == 0;
                            twinAllLevels = (lvN == memoryBlock.level)
                                || (lvN == memoryBlock.level - 1 && !has0);
                        }
                        // I-28: updateGlobalSingleChapter performs cross-LB writes (root
                        // mailOut via mergeBatchInto, tree-wide deactivateUnnecessary)
                        // that are forbidden on a worker during the parallel phase.
                        // Defer it to the post-join single-threaded drain, as
                        // dischargeToBeProved does. The drain index-sorts
                        // updateGlobalSingleChapterPages (deterministic emission order) and
                        // runs before the D-76 compaction drain, so the compact-rule
                        // broadcast timing holds.
                        {
                            std::lock_guard<std::mutex> lock(this->updateGlobalSingleChapterMutex);
                            assert(this->updateGlobalSingleChapterPages
                                && "dischargeContradiction: updateGlobalSingleChapterPages not emplaced by proveKernel");
                            SealedPageSet& ps = *this->updateGlobalSingleChapterPages;
                            ps.appendRecord(UpdateGlobalSingleChapterRec{
                                SealedString::copyFrom(ps, theorem.ptr,
                                    theorem.len),
                                coreId, &memoryBlock, twinAllLevels });
                        }
                    }
                    return;
                }

                // ---- CE filter: conjecture refuted. A CE LB's fired heads now
                //      enter the statement registry like any derived statement,
                //      so this sweep sees the refuting head whose negation is a
                //      loaded fact. Mark the conjecture's contradictionTable row
                //      and deactivate. CE LBs are never primed or recursion, so
                //      the incubator / vacuous branches never fire for them. ----
                if (memoryBlock.contradictionIndex >= 0) {
                    this->contradictionTable[memoryBlock.contradictionIndex].successful = true;
                    memoryBlock.isActive = false;
                    return;
                }

                // ---- Vacuous truth in induction steps ----
                // Confined to contradicting pairs at validity "main": a
                // hypo-scope contradiction must not justify a main-scope
                // conclusion. There is deliberately no mb.level gate — for
                // theorems whose premise is itself impossible
                // (chapter101-style lemmas) the contradiction is necessarily
                // rooted in the theorem's outer premise, not the recursion
                // step's hypothesis, and an mb.level gate would reject those
                // legitimate vacuous-truth cases. Soundness rests on axiom
                // consistency.
                if (memoryBlock.isPartOfRecursion
                    && equalSpans(StrSpan(validityName), StrSpan("main", 4))) {
                    // The LB's concrete recursion hypothesis is the 3rd
                    // ingredient so the verifier's chapter-local trace can
                    // reach it (derivation chains use this form, not exprKey).
                    // copyFrom onto sArena: the history line below cites it in
                    // its deps; the copy covers BOTH source branches uniformly.
                    const StrSpan rpView = memoryBlock.recursionHypothesisId != 0
                        ? memoryBlock.nameMap.decodeView(memoryBlock.recursionHypothesisId)
                        : memoryBlock.exprKeyView();
                    const ScratchString recursionPremise =
                        ScratchString::copyFrom(sArena, rpView.ptr, rpView.len);

                    // Decoded-lex key INDEX over intToBeProved — the former
                    // decodeToBeProvedSorted heap snapshot. Reproduces its
                    // (original, validityName) lex order byte-for-byte
                    // (compareSpans == std::string::compare, decoded pairs unique
                    // -> tie-free), so the FIRST main-scope row here is the same
                    // goal the former walk picked. The key snapshot (tbpKeys)
                    // isolates the walk from consumer inserts/erases; the sort is
                    // mint-free; only the ONE consumed row's original is copied.
                    const unsigned dcSlot = (gl::WorkerContext::coreId() >= 0)
                        ? static_cast<unsigned>(gl::WorkerContext::coreId())
                        : genScratchArenas().slotCount() - 1;
                    ScratchArena& dcG = genScratchArenas().forSlot(dcSlot);
                    ScratchArena& dcS = scratchArenas().forSlot(dcSlot);
                    DirtyState dcKeyDirty = DirtyState::Clean;
                    PagedVector<int64_t> tbpKeys(&dcG, &dcKeyDirty);
                    const int32_t tbpN = memoryBlock.intToBeProved.count();
                    for (int32_t id = 1; id <= tbpN; ++id)
                        tbpKeys.push_back(memoryBlock.intToBeProved.keyAt(id));
                    ScratchScope dcIdxScope(dcG);
                    int32_t* tbpOrder = tbpN
                        ? reinterpret_cast<int32_t*>(dcG.resolve(dcG.alloc(
                              tbpN * static_cast<int32_t>(sizeof(int32_t)),
                              static_cast<int32_t>(alignof(int32_t)))))
                        : nullptr;
                    for (int32_t i = 0; i < tbpN; ++i) tbpOrder[i] = i;
                    std::sort(tbpOrder, tbpOrder + tbpN, [&](int32_t a, int32_t b) {
                        const NameId oa = Codec<StatementKey>::decode(tbpKeys[a]).orig;
                        const NameId ob = Codec<StatementKey>::decode(tbpKeys[b]).orig;
                        const int c = compareSpans(memoryBlock.nameMap.decodeView(oa),
                                                   memoryBlock.nameMap.decodeView(ob));
                        if (c != 0) return c < 0;
                        const NameId va = Codec<StatementKey>::decode(tbpKeys[a]).validity;
                        const NameId vb = Codec<StatementKey>::decode(tbpKeys[b]).validity;
                        return compareSpans(memoryBlock.nameMap.decodeView(va),
                                            memoryBlock.nameMap.decodeView(vb)) < 0;
                    });
                    for (int32_t oi = 0; oi < tbpN; ++oi) {
                        const int64_t rowKey = tbpKeys[tbpOrder[oi]];
                        if (Codec<StatementKey>::decode(rowKey).validity
                            == NameMap::MAIN_ID) {
                            // Copy the consumed row's original onto the string tier
                            // for its history line.
                            const NameId rowOrigId =
                                Codec<StatementKey>::decode(rowKey).orig;
                            ScratchScope dcRowScope(dcS);
                            const StrSpan rowOrigView =
                                memoryBlock.nameMap.decodeView(rowOrigId);
                            const ScratchString rowOrig = ScratchString::copyFrom(
                                dcS, rowOrigView.ptr, rowOrigView.len);
                            if (parameters.trackHistory) {
                                // The vacuous-truth RECORD antecedents are all
                                // in-hand spans (expr / negation / recursionPremise
                                // on sArena; rowOrig on dcS; "main" a literal); the
                                // door mints originInterner, never these arenas, so
                                // no span dangles across the mint (I-3).
                                const StrSpan vtMain("main", 4);
                                const OriginDep vtDeps[3] = {
                                    { StrSpan(expr), StrSpan(validityName) },
                                    { StrSpan(negation), StrSpan(validityName) },
                                    { StrSpan(recursionPremise), vtMain } };
                                addOriginEncoded(memoryBlock.exprOriginMap,
                                    memoryBlock.originInterner,
                                    StrSpan(rowOrig), vtMain,
                                    OriginTag::vacuousTruth, vtDeps, 3, maxOrig);
                            }
                            // The contradiction proves the goal at every level
                            // of the block. The branch closes it itself: a
                            // deposited head would be respelled under the
                            // block's classes, and the exact-key discharge
                            // would not find it (D-573).
                            assert(memoryBlock.level >= 0 && memoryBlock.level < 64
                                && "an induction block deeper than the level mask");
                            uint64_t levelsMask = 0;
                            for (int l = 0; l <= memoryBlock.level; ++l)
                                levelsMask |= uint64_t(1) << l;
                            if (!parameters.compressor_mode)
                                closeRecursionGoal(memoryBlock, rowKey, levelsMask, coreId);
                            break;
                        }
                    }
                    memoryBlock.isActive = false;
                    return;
                }
                // ---- Normal LB: the premise set itself is contradictory ----
                // No special role applies, so the E / !E pair at exact
                // `main` was derived from the LB's premises alone — the
                // premise set is inconsistent and every theorem provable at
                // or below this LB is vacuous (ex falso). Flag the LB for
                // the post-join emission gates (`hasContradictedAncestor`
                // at drainUpdateGlobalSingleChapter / the induction promotion),
                // deactivate it, and stage its subtree for the post-join
                // deactivation drain. A contradiction at the batch root or
                // an anchor premise LB would mean the definition set itself
                // is inconsistent — a configuration bug, never a defined
                // state.
                {
                    const StrSpan mbKey = memoryBlock.exprKeyView();
                    assert(mbKey.len != 0
                        && "definition-set inconsistency: contradiction at the batch root");
                    assert(!startsWithSpan(mbKey, "(Anchor", 7)
                        && "definition-set inconsistency: contradiction at an anchor premise LB");
                    memoryBlock.mainContradiction = true;
                    memoryBlock.isActive = false;
                    scopeLifecycle.stageSubtreeDeactivation(&memoryBlock);
                    return;
                }
            }
        }

/// @brief Discharge `contradiction_`-marked scopes: when a per-step
        ///        delta statement at/below such a scope has its negation known
        ///        at a comparable scope, emit the scope's product at the
        ///        scope's parent and wipe the scope.
        ///
        /// @details
        /// The discharge half of contradiction-based integration
        /// (`prepareNegatedCompoundContradiction` is the priming half). The
        /// walk mirrors `dischargeToBeProved`'s delta snapshot: for every new
        /// local statement whose scope chain carries a `contradiction_`
        /// payload node, the statement's negation is probed (known-bit) at
        /// every scope on the statement's chain up to and including `"main"`.
        /// A hit refutes the scope's assumption: the product
        /// (`contradictionScopeProductView` of the marked node's payload) is
        /// emitted as an internal-mail statement at the marked node's PARENT
        /// with a `contradiction` history row citing the two opposing
        /// expressions, and the scope subtree is queued for the end-of-burst
        /// wipe (`intValidityNamesToFilter` + `pendingWipeScopes` — the
        /// scoped-goal-closure cleanup pair). `pendingWipeScopes` membership
        /// doubles as the once-per-step guard.
        ///
        /// Deliberate scoped relaxation of the main/main contradiction
        /// contract ([I-78](../../docs/agentic_swdd/30_invariants.md#i-78)):
        /// it fires ONLY for scopes carrying the `contradiction_` marker,
        /// never for the LB-level discharge; precedent is the
        /// refuted-disjunct retirement (D-242).
        ///
        /// @param memoryBlock     The LB whose statement delta is scanned.
        /// @param internalMailOut The step's internal-mail channel — the
        ///                        emission rides the standard absorb, which
        ///                        sets the product's `registered` bit.
        /// @invariant Heap-free; probes non-minting; single-threaded (the
        ///            phase-3 seam, like `dischargeToBeProved`).
        /// @see prepareNegatedCompoundContradiction, dischargeToBeProved,
        ///      contradictionScopeProductView.
inline void GoalProcessor::dischargeContradictionScopes(Memory& memoryBlock,
                                                 ColdMail& internalMailOut) {
            const unsigned dcsGenSlot = (gl::WorkerContext::coreId() >= 0)
                ? static_cast<unsigned>(gl::WorkerContext::coreId())
                : genScratchArenas().slotCount() - 1;
            ScratchArena& gArena = genScratchArenas().forSlot(dcsGenSlot);
            DirtyState snapDirty = DirtyState::Clean;
            PagedVector<IntEncodedExpr> deltaSnapshot(&gArena, &snapDirty);
            for (int32_t di = 0;
                 di < memoryBlock.intLocalEncodedStatementsDelta.size();
                 ++di) {
                deltaSnapshot.push_back(
                    memoryBlock.intLocalEncodedStatementsDelta[di]);
            }

            const unsigned dcsStrSlot = (gl::WorkerContext::coreId() >= 0)
                ? static_cast<unsigned>(gl::WorkerContext::coreId())
                : scratchArenas().slotCount() - 1;
            ScratchArena& sArena = scratchArenas().forSlot(dcsStrSlot);

            for (int32_t di = 0; di < deltaSnapshot.size(); ++di) {
                const IntEncodedExpr ie = deltaSnapshot[di];
                if (memoryBlock.phase3IgnoresScope(ie.validityId)) continue;
                if (ie.validityId == NameMap::MAIN_ID) continue;

                // The statement must sit EXACTLY at a contradiction-marked
                // scope. A statement at a DEEPER node (an `_ordis_`/`orint_`
                // branch, a nested subproof) is conditional on that node's
                // own assumption — its contradiction refutes the branch
                // (D-242's machinery), never the reductio seed; firing on it
                // would be unsound.
                const StrSpan ownPl = memoryBlock.nameMap.decodeSubView(
                    memoryBlock.nameMap.stackBack(ie.validityId));
                const StrSpan productView =
                    contradictionScopeProductView(ownPl);
                if (productView.empty()) continue;
                const NameId contraId = ie.validityId;
                if (memoryBlock.pendingWipeScopes.contains(contraId)) continue;

                ScratchScope entryScope(sArena);
                // Owned copies (I-3): the probes below are non-minting, but
                // the fire path mints (internal-mail encode), so every span
                // read from the NameMap is copied before use.
                const StrSpan stmtView =
                    memoryBlock.nameMap.decodeView(ie.originalId);
                const ScratchString stmt = ScratchString::copyFrom(
                    sArena, stmtView.ptr, stmtView.len);
                const StrSpan stmtScopeView =
                    memoryBlock.nameMap.decodeView(ie.validityId);
                const ScratchString stmtScope = ScratchString::copyFrom(
                    sArena, stmtScopeView.ptr, stmtScopeView.len);
                const ScratchString productS = ScratchString::copyFrom(
                    sArena, productView.ptr, productView.len);
                const NameId parentId =
                    memoryBlock.nameMap.parentOf(contraId);
                const StrSpan parentView =
                    memoryBlock.nameMap.decodeView(parentId);
                const ScratchString parentScope = ScratchString::copyFrom(
                    sArena, parentView.ptr, parentView.len);

                // gl::DefinitionExpansion::negate(stmt): strip or prefix `!` (fresh bytes on sArena).
                StrSpan negStmt;
                if (StrSpan(stmt).len > 0 && StrSpan(stmt).ptr[0] == '!') {
                    negStmt = StrSpan(StrSpan(stmt).ptr + 1,
                                      StrSpan(stmt).len - 1);
                } else {
                    char* nb = sArena.allocBytes(1 + StrSpan(stmt).len);
                    nb[0] = '!';
                    std::memcpy(nb + 1, StrSpan(stmt).ptr,
                                static_cast<size_t>(StrSpan(stmt).len));
                    negStmt = StrSpan(nb, 1 + StrSpan(stmt).len);
                }

                // Probe the negation (known-bit) at every scope on the chain
                // from the statement's scope up to and including main.
                NameId hitScopeId = 0;
                for (NameId w = ie.validityId;;) {
                    const StrSpan wView = memoryBlock.nameMap.decodeView(w);
                    const StatementFlags* row = lookupStatementFlags(
                        memoryBlock.intKnownStatements, memoryBlock.nameMap,
                        negStmt, wView);
                    if (row != nullptr) {
                        hitScopeId = w;
                        break;
                    }
                    if (w == NameMap::MAIN_ID) break;
                    w = memoryBlock.nameMap.parentOf(w);
                    if (w == 0) break;
                }
                if (hitScopeId == 0) continue;

                const StrSpan hitScopeView =
                    memoryBlock.nameMap.decodeView(hitScopeId);
                const ScratchString hitScope = ScratchString::copyFrom(
                    sArena, hitScopeView.ptr, hitScopeView.len);

                // Levels: the delta statement's own level row (contract-backed
                // by the addStatement post-loop invariant), UNIONED with the
                // ambient collision partner's levels — the product's
                // derivation used BOTH antecedents, so its level set carries
                // both sides (dropping the partner's side starved the
                // discharge gate's level accounting of the premise levels the
                // reductio genuinely consumed). The {-1} non-derived tier is
                // transparent to the union (an all-non-derived product
                // re-enters as {-1} at the addStatement door).
                int lvRun[256];
                const int32_t lvlsId = lookupStatementLevels(
                    memoryBlock.intStatementLevelsMap, memoryBlock.nameMap,
                    StrSpan(stmt), StrSpan(stmtScope));
                assert(lvlsId != 0
                    && memoryBlock.intStatementLevelsMap.runLen(lvlsId) > 0
                    && "dischargeContradictionScopes: delta statement without "
                       "a non-empty level row");
                int32_t lvN = coldIntRunNonNegAt(
                    memoryBlock.intStatementLevelsMap, lvlsId, lvRun, 256);
                {
                    const int32_t negLvlsId = lookupStatementLevels(
                        memoryBlock.intStatementLevelsMap, memoryBlock.nameMap,
                        negStmt, StrSpan(hitScope));
                    assert(negLvlsId != 0
                        && memoryBlock.intStatementLevelsMap.runLen(negLvlsId) > 0
                        && "dischargeContradictionScopes: known collision "
                           "partner without a non-empty level row");
                    int negRun[256];
                    const int32_t negN = coldIntRunAt(
                        memoryBlock.intStatementLevelsMap, negLvlsId,
                        negRun, 256);
                    for (int32_t j = 0; j < negN; ++j)
                        lvN = insertLevelSorted(lvRun, lvN, negRun[j], 256);
                }

                RT_SCOPE_HERE("DISCHARGE_CONTRA_SCOPE_HIT");
                insertInternalStatement(internalMailOut, memoryBlock.nameMap,
                    StrSpan(productS), StrSpan(parentScope), lvRun, lvN);
                if (parameters.trackHistory) {
                    const int cap = parameters.compressor_mode
                        ? parameters.compressor_max_origins_per_expr
                        : parameters.max_origin_per_expr;
                    const OriginDep deps[2] = {
                        { StrSpan(stmt), StrSpan(stmtScope) },
                        { negStmt, StrSpan(hitScope) } };
                    addInternalMailOrigin(internalMailOut,
                        memoryBlock.originInterner, StrSpan(productS),
                        StrSpan(parentScope), OriginTag::contradiction,
                        deps, 2, cap);
                }

                memoryBlock.intValidityNamesToFilter.mint(contraId);
                memoryBlock.pendingWipeScopes.mint(contraId);
            }
        }

/// @brief Prove an ordinary implication goal from contradictory visible facts.
        ///
        /// @details
        /// Inspect unresolved positive implication scopes in decoded order. One
        /// opposing fact must belong to the exact goal scope; its partner may
        /// belong to that scope or an ancestor. Record the scoped head with both
        /// actual dependencies and their combined levels. Normal goal discharge
        /// then records the compact implication and schedules its scope cleanup.
        /// The known registry covers both fact arrival orders without consulting
        /// proof-history containers.
        ///
        /// @param memoryBlock Active logical block containing unresolved goals.
        /// @return Nothing; proven heads enter the ordinary local statement delta.
        /// @invariant No main, sibling, descendant-only, OR, or reductio closure.
        /// @invariant Heap-free; copied strings survive NameMap mints.
        /// @see dischargeToBeProved, dischargeContradiction, materializeWatchHits.
inline void GoalProcessor::dischargeVacuousImplicationScopes(Memory& memoryBlock) {
            if (!memoryBlock.isActive) return;
            const unsigned slot = (gl::WorkerContext::coreId() >= 0)
                ? static_cast<unsigned>(gl::WorkerContext::coreId())
                : genScratchArenas().slotCount() - 1;
            ScratchArena& gArena = genScratchArenas().forSlot(slot);
            ScratchArena& sArena = scratchArenas().forSlot(slot);
            ScratchScope goalScope(gArena);
            int32_t goalCount = 0;
            const int64_t* goalKeys = sortToBeProvedKeys(
                memoryBlock.intToBeProved, memoryBlock.nameMap, gArena, goalCount);
            DirtyState outputDirty = DirtyState::Clean;
            PagedVector<IntEncodedExpr> output(&gArena, &outputDirty);
            for (int32_t gi = 0; gi < goalCount; ++gi) {
                const StatementKey goal = Codec<StatementKey>::decode(goalKeys[gi]);
                if (goal.validity == NameMap::MAIN_ID
                    || memoryBlock.phase3IgnoresScope(goal.validity)) continue;
                const StrSpan compact = gl::ScopeSyntax::stripSubproofPrefixView(
                    memoryBlock.nameMap.decodeSubView(
                        memoryBlock.nameMap.stackBack(goal.validity)));
                if (compact.empty() || compact.ptr[0] != '(') continue;
                const LogicalEntity* entity = this->definitions.compiledEntity(
                    extractExpressionUniversalSpan(compact));
                if (entity == nullptr || entity->category != "implication") continue;
                const int32_t ancestorCount = memoryBlock.nameMap.ancLen(goal.validity);
                bool alreadyClosing = false;
                for (int32_t ai = 0; ai < ancestorCount; ++ai) {
                    const NameId ancestor = memoryBlock.nameMap.ancAt(goal.validity, ai);
                    if (memoryBlock.pendingWipeScopes.contains(ancestor)
                        || memoryBlock.intValidityNamesToFilter.contains(ancestor)
                        || memoryBlock.intKnownStatements.find(
                            StatementKey{goal.orig, ancestor}) != nullptr) {
                        alreadyClosing = true;
                        break;
                    }
                }
                if (alreadyClosing) continue;
                NameId selectedFact = 0;
                NameId selectedOpposite = 0;
                NameId selectedScope = 0;
                for (int32_t si = 0; si < memoryBlock.intEncodedStatements.size(); ++si) {
                    const IntEncodedExpr row = memoryBlock.intEncodedStatements[si];
                    if (row.validityId != goal.validity
                        || memoryBlock.intKnownStatements.find(
                            StatementKey{row.originalId, row.validityId}) == nullptr) continue;
                    const StrSpan fact = memoryBlock.nameMap.decodeView(row.originalId);
                    if (selectedFact != 0 && compareSpans(fact,
                            memoryBlock.nameMap.decodeView(selectedFact)) >= 0) continue;
                    ScratchScope probeScope(sArena);
                    const StrSpan opposite = gl::DefinitionExpansion::negateScratch(sArena, fact);
                    const NameId oppositeId = memoryBlock.nameMap.lookup(opposite);
                    if (oppositeId == 0) continue;
                    for (int32_t ai = 0; ai < ancestorCount; ++ai) {
                        const NameId ancestor = memoryBlock.nameMap.ancAt(goal.validity, ai);
                        if (memoryBlock.intKnownStatements.find(
                                StatementKey{oppositeId, ancestor}) != nullptr) {
                            selectedFact = row.originalId;
                            selectedOpposite = oppositeId;
                            selectedScope = ancestor;
                            break;
                        }
                    }
                }
                if (selectedFact == 0) continue;
                int levels[256];
                int32_t levelCount = 0;
                const StatementKey sources[2] = {
                    {selectedFact, goal.validity}, {selectedOpposite, selectedScope}};
                for (const StatementKey& source : sources) {
                    const int32_t levelRow = memoryBlock.intStatementLevelsMap.lookup(
                        packStatementKey(source.orig, source.validity));
                    assert(levelRow != 0
                        && memoryBlock.intStatementLevelsMap.runLen(levelRow) > 0
                        && "known contradiction fact requires a non-empty level row");
                    int sourceLevels[256];
                    const int32_t sourceCount = coldIntRunNonNegAt(
                        memoryBlock.intStatementLevelsMap, levelRow, sourceLevels, 256);
                    for (int32_t li = 0; li < sourceCount; ++li)
                        levelCount = insertLevelSorted(levels, levelCount, sourceLevels[li], 256);
                }
                ScratchScope emissionScope(sArena);
                const auto copyName = [&](NameId id) {
                    const StrSpan view = memoryBlock.nameMap.decodeView(id);
                    return ScratchString::copyFrom(sArena, view.ptr, view.len);
                };
                const ScratchString head = copyName(goal.orig);
                const ScratchString scope = copyName(goal.validity);
                const ScratchString fact = copyName(selectedFact);
                const ScratchString opposite = copyName(selectedOpposite);
                const ScratchString oppositeScope = copyName(selectedScope);
                const OriginDep dependencies[2] = {
                    {StrSpan(fact), StrSpan(scope)},
                    {StrSpan(opposite), StrSpan(oppositeScope)}};
                const TransientOrigin origin{true, OriginTag::vacuousTruth, dependencies, 2};
                statementMemory.addStatement(StrSpan(head), memoryBlock, true, levels, levelCount,
                    origin, StrSpan(scope), output);
            }
        }


/// @brief Close one main goal of an induction block: report its induction
///        tuples, erase it and switch the block off
///        (D-573).
///
/// @details
/// The induction closure shared by `dischargeToBeProved` (a new statement at
/// the goal's exact key) and the vacuous-truth branch of
/// `dischargeContradiction` (a contradiction at main). Every auxy of the
/// goal's run, in its stored ascending order, is reported to the induction
/// promotion as `(auxy, levelsMask, coreId)` in `updateGlobalTuples`; the goal
/// is erased from `intToBeProved` and the block deactivated. A step block
/// (key `(in2[rec`) is queued in `inductionMemoryBlocks` for the step-win
/// seam (`ProofBarrier::settleStepWins`).
///
/// @param memoryBlock The induction block (`isPartOfRecursion`).
/// @param goalKey     The goal's packed key: an open goal at `main`.
/// @param levelsMask  The levels the closure used, one bit per level, none
///                    above the block's level.
/// @param coreId      Worker core id carried by the tuples.
/// @return Nothing.
/// @invariant Not in compressor mode; each push runs under its mutex.
/// @see dischargeToBeProved, dischargeContradiction,
///      ScopeLifecycle::settleStepClosure.
inline void GoalProcessor::closeRecursionGoal(Memory& memoryBlock, int64_t goalKey,
                                              uint64_t levelsMask, int coreId) {
            assert(memoryBlock.isPartOfRecursion && "closeRecursionGoal: an induction block");
            assert(!parameters.compressor_mode
                && "closeRecursionGoal: no induction closure in compressor mode");
            assert(Codec<StatementKey>::decode(goalKey).validity == NameMap::MAIN_ID
                && "closeRecursionGoal: a main goal");
            assert(memoryBlock.level >= 0 && memoryBlock.level < 64
                && (memoryBlock.level == 63 || (levelsMask >> (memoryBlock.level + 1)) == 0)
                && "closeRecursionGoal: a level above the block's level");
            const int32_t goalId = memoryBlock.intToBeProved.lookup(goalKey);
            assert(goalId != 0 && "closeRecursionGoal: an open goal");
            // The goal registry's run is stored ascending-unique, so the
            // valueAt walk is the former set order (I-138); it completes
            // before the eraseSet below.
            {
                std::lock_guard<std::mutex> lock(this->updateGlobalMutex);
                const int32_t auxN = memoryBlock.intToBeProved.runLen(goalId);
                for (int32_t aj = 0; aj < auxN; ++aj) {
                    this->updateGlobalTuples.push_back(std::make_tuple(
                        memoryBlock.intToBeProved.valueAt(goalId, aj), levelsMask, coreId));
                }
            }
            memoryBlock.intToBeProved.eraseSet(goalKey);
            memoryBlock.isActive = false;
            if (startsWithSpan(memoryBlock.exprKeyView(), "(in2[rec", 8)) {
                std::lock_guard<std::mutex> lock(this->inductionMemoryBlocksMutex);
                this->inductionMemoryBlocks.push_back(&memoryBlock);
            }
        }

inline void GoalProcessor::dischargeToBeProved(Memory& memoryBlock, int coreId,
                                        ColdMail& internalMailOut) {
            // Iterate the local-statement delta (int registry) — the per-step
            // delta of new local statements. For each entry, run
            // the three toBeProved cases:
            //   1. Induction-recursion proof  — record proved tuples,
            //      mark LB inactive, queue induction-LB cleanup.
            //   2. Non-recursion proof — check `isProved`, reconstruct
            //      implication, emit via `updateGlobalSingleChapterPages`,
            //      stage scope wipe.
            //   3. OR-integration / NotOrScope — classify OR scope,
            //      emit wrapping OR expression at parent scope via
            //      recursive addExprToMemoryBlock.
            //
            // Snapshot the delta so the OR-integration / closure
            // paths (which may grow the statement registry or
            // sameIterationInternalMail) don't drive a runaway loop over the
            // freshly-added entries. Element-by-element copy off the paged
            // container; the transient snapshot rides the per-slot gen-scratch
            // PAGE tier (I-138), freed by
            // ~PagedVector at scope exit. Recursion-safe: a nested
            // dischargeToBeProved frame builds its own PagedVector on the same
            // slot with its own page vids, and page-tier pages survive every
            // nested byte-bump ScratchScope on that arena.
            const unsigned slot = (gl::WorkerContext::coreId() >= 0)
                ? static_cast<unsigned>(gl::WorkerContext::coreId())
                : genScratchArenas().slotCount() - 1;
            ScratchArena& gArena = genScratchArenas().forSlot(slot);
            DirtyState snapDirty = DirtyState::Clean;
            PagedVector<IntEncodedExpr> deltaSnapshot(&gArena, &snapDirty);
            for (int32_t di = 0;
                 di < memoryBlock.intLocalEncodedStatementsDelta.size();
                 ++di) {
                deltaSnapshot.push_back(
                    memoryBlock.intLocalEncodedStatementsDelta[di]);
            }

            const unsigned dtpSlot = (gl::WorkerContext::coreId() >= 0)
                ? static_cast<unsigned>(gl::WorkerContext::coreId())
                : scratchArenas().slotCount() - 1;
            ScratchArena& sArena = scratchArenas().forSlot(dtpSlot);

            // Close one non-main goal from a statement visible at that goal.
            // The proof source and exact goal scope are deliberately separate:
            // an ancestor supplies the fact and levels, while classification,
            // emission, and cleanup remain anchored to the goal that closes.
            const auto closeScopedGoal = [&](NameId expressionId,
                                             NameId sourceValidityId,
                                             NameId goalValidityId,
                                             const int* levels,
                                             int32_t levelCount) {
                assert(goalValidityId != NameMap::MAIN_ID
                       && "scoped-goal closure requires a non-main goal");
                assert(memoryBlock.intToBeProved.lookup(
                           packStatementKey(expressionId, goalValidityId)) != 0
                       && "scoped-goal closure requires an unresolved exact goal");

                ScratchScope closureScope(sArena);
                const StrSpan expressionView =
                    memoryBlock.nameMap.decodeView(expressionId);
                const ScratchString expression =
                    ScratchString::copyFrom(
                        sArena, expressionView.ptr, expressionView.len);
                const StrSpan sourceValidityView =
                    memoryBlock.nameMap.decodeView(sourceValidityId);
                const ScratchString sourceValidity =
                    ScratchString::copyFrom(
                        sArena, sourceValidityView.ptr, sourceValidityView.len);
                const StrSpan goalValidityView =
                    memoryBlock.nameMap.decodeView(goalValidityId);
                const ScratchString goalValidity =
                    ScratchString::copyFrom(
                        sArena, goalValidityView.ptr, goalValidityView.len);

                StrSpan orExprView;
                StrSpan orBranchBodyView;
                const OrScopeKind orKind = gl::ScopeSyntax::classifyOrScopeView(
                    memoryBlock.nameMap, goalValidityId,
                    orExprView, orBranchBodyView);

                if (orKind == OrScopeKind::Integration) {
                    const NameId orParentValidityId =
                        memoryBlock.nameMap.parentOf(goalValidityId);
                    assert(orParentValidityId != 0
                           && "OR integration scope requires a direct parent");

                    const ScratchString orExpr =
                        ScratchString::copyFrom(
                            sArena, orExprView.ptr, orExprView.len);
                    const StrSpan goalValiditySpan(goalValidity);
                    const int32_t lastBoundary = rfindSpanBefore(
                        goalValiditySpan,
                        StrSpan(NameMap::BOUNDARY_STR,
                                static_cast<int32_t>(NameMap::BOUNDARY_LEN)),
                        goalValiditySpan.len);
                    const StrSpan orEmitScope =
                        (lastBoundary < 0)
                            ? StrSpan("main", 4)
                            : StrSpan(goalValiditySpan.ptr, lastBoundary);
                    const OriginDep branchProvenDeps[1] = {
                        { StrSpan(expression), StrSpan(sourceValidity) } };

                    const int maxOrigins = parameters.compressor_mode
                        ? parameters.compressor_max_origins_per_expr
                        : parameters.max_origin_per_expr;
                    insertInternalStatement(
                        internalMailOut, memoryBlock.nameMap,
                        StrSpan(orExpr), orEmitScope, levels, levelCount);
                    if (parameters.trackHistory) {
                        addInternalMailOrigin(
                            internalMailOut, memoryBlock.originInterner,
                            StrSpan(orExpr), orEmitScope,
                            OriginTag::orBranchProven,
                            branchProvenDeps, 1, maxOrigins);
                    }

                    caseProcessor.cleanUpOrIntegrationBranches(
                        StrSpan(orExpr), orParentValidityId, memoryBlock);
                }
                else if (orKind == OrScopeKind::NotOrScope) {
                    assert(!memoryBlock.nameMap.stackEmpty(goalValidityId)
                           && "NotOrScope branch requires non-empty stack");
                    // The subproof deposits its BARE compact at the scope that
                    // opened it, the subproof scope's parent: its proof may use
                    // every fact of that scope, so the compact holds there and
                    // below, not above. A goal-carrying payload exists only on
                    // main, so it lands at main; its goal prefix is ownership
                    // metadata, never part of the proven statement
                    // (D-562).
                    const StrSpan bareSigView = gl::ScopeSyntax::stripSubproofPrefixView(
                        memoryBlock.nameMap.decodeSubView(
                            memoryBlock.nameMap.stackBack(goalValidityId)));
                    const ScratchString bareSig =
                        ScratchString::copyFrom(
                            sArena, bareSigView.ptr, bareSigView.len);
                    const NameId openingScopeId = memoryBlock.nameMap.parentOf(goalValidityId);
                    assert(openingScopeId != 0
                           && "subproof scope requires a direct parent");
                    const StrSpan openingScopeView = memoryBlock.nameMap.decodeView(openingScopeId);
                    const ScratchString openingScope =
                        ScratchString::copyFrom(
                            sArena, openingScopeView.ptr, openingScopeView.len);
                    // The closing row cites the subproof's own goal. A class
                    // rewrite of its open goal (sanitizeToBeProved) closes
                    // it in another spelling, from which one equality step
                    // leads back (D-557).
                    StrSpan citedGoal(expression);
                    StrSpan citedScope(sourceValidity);
                    const int64_t* ownGoal = memoryBlock.subproofGoals.find(
                        static_cast<int64_t>(goalValidityId));
                    if (ownGoal != nullptr
                        && static_cast<NameId>(*ownGoal) != expressionId) {
                        const StrSpan ownView = memoryBlock.nameMap.decodeView(
                            static_cast<NameId>(*ownGoal));
                        citedGoal = StrSpan(ScratchString::copyFrom(
                            sArena, ownView.ptr, ownView.len));
                        citedScope = StrSpan(goalValidity);
                        if (parameters.trackHistory) {
                            proofHistory.writeEqualityBridge(memoryBlock, citedGoal,
                                goalValidityId, StrSpan(expression),
                                StrSpan(sourceValidity));
                        }
                    }
                    const OriginDep validityNameDeps[1] = {
                        { citedGoal, citedScope } };

                    const int maxOrigins = parameters.compressor_mode
                        ? parameters.compressor_max_origins_per_expr
                        : parameters.max_origin_per_expr;
                    insertInternalStatement(
                        internalMailOut, memoryBlock.nameMap,
                        StrSpan(bareSig), StrSpan(openingScope),
                        levels, levelCount);
                    if (parameters.trackHistory) {
                        addInternalMailOrigin(
                            internalMailOut, memoryBlock.originInterner,
                            StrSpan(bareSig), StrSpan(openingScope),
                            OriginTag::validityName,
                            validityNameDeps, 1, maxOrigins);
                    }

                    memoryBlock.intValidityNamesToFilter.mint(goalValidityId);
                    memoryBlock.pendingWipeScopes.mint(goalValidityId);
                }
            };

            RT_NOTE_ITERATIONS_HERE(static_cast<int>(deltaSnapshot.size()));
            for (int32_t k = 0; k < deltaSnapshot.size(); ++k) {
                RT_SCOPE_HERE("DTP_DELTA_ROW");
                const IntEncodedExpr& ie = deltaSnapshot[k];
                if (memoryBlock.phase3IgnoresScope(ie.validityId)) continue;
                // The decoded original is COPIED onto the string arena
                // (per-entry entryScope): the main-goal reaction paths below mint
                // NameMap, which would dangle a raw decodeView span, but a copy on
                // the string tier survives every NameMap mint (I-3). Scoped-goal
                // closure makes its own source/goal copies inside closeScopedGoal.
                // No EncodedExpression / parseArgument reconstruction is needed
                // (that machinery only recomputed arity/negation/maxIteration for
                // asserts this caller never reads).
                ScratchScope entryScope(sArena);
                const StrSpan addExprView = memoryBlock.nameMap.decodeView(ie.originalId);
                const ScratchString addExpression =
                    ScratchString::copyFrom(sArena, addExprView.ptr, addExprView.len);

                // Packed probe via the delta row's ids; the value reference
                // stays valid under rehash (only iterators invalidate), so
                // it survives the inserts the reaction paths below commit.
                // The {-1} non-derived tier is transparent to the level
                // arithmetic: the filtered read makes a {-1} row gate exactly
                // like the former empty run.
                const int32_t sitId = memoryBlock.intStatementLevelsMap.lookup(
                    packStatementKey(ie.originalId, ie.validityId));
                assert(sitId != 0
                       && memoryBlock.intStatementLevelsMap.runLen(sitId) > 0
                       && "intLocalEncodedStatementsDelta / intStatementLevelsMap invariant violated");
                int lvRun[256];
                const int32_t lvN = coldIntRunNonNegAt(
                    memoryBlock.intStatementLevelsMap, sitId, lvRun, 256);

                bool has0 = false;
                for (int32_t j = 0; j < lvN; ++j)
                    if (lvRun[j] == 0) { has0 = true; break; }
                bool allLevelsInvolved = (lvN == memoryBlock.level + 1);
                if (lvN == memoryBlock.level && !has0)
                {
                    allLevelsInvolved = true;
                }

                if (memoryBlock.isPartOfRecursion) {
                    // An induction closure reports the levels its proof used,
                    // one bit per level. The induction promotion judges the
                    // union of the step and base closures and the typing
                    // fact against the conjecture's premise levels
                    // (ScopeLifecycle::inductionPremiseLevels,
                    // D-516).
                    assert(memoryBlock.level < 64
                        && "an induction block deeper than the level mask");
                    uint64_t levelsMask = 0;
                    for (int32_t j = 0; j < lvN; ++j) {
                        assert(lvRun[j] <= memoryBlock.level
                            && "a statement level outside its block's level universe");
                        levelsMask |= uint64_t(1) << lvRun[j];
                    }
                    // Packed probe — the delta row's ids ARE the goal key;
                    // "main" is the pre-registered NameMap::MAIN_ID.
                    const int64_t pkProof =
                        packStatementKey(ie.originalId, ie.validityId);
                    if (memoryBlock.intToBeProved.lookup(pkProof) != 0
                        && ie.validityId == NameMap::MAIN_ID
                        && !parameters.compressor_mode)
                        closeRecursionGoal(memoryBlock, pkProof, levelsMask, coreId);
                }
                else {
                    // There is no `status != 0` predicate here — delta
                    // entries do not carry per-entry status, and there
                    // is no per-expression `statementPublishable`
                    // tracking. The `toBeProved.find` gate below
                    // preserves the effective behaviour for current
                    // proof sets; rare status-0 false-publish cases
                    // are an accepted residual semantic widening.
                    // Closure is level-free: a MAIN goal whose head is
                    // proved at the exact goal key closes here (goal
                    // erase, scope wipes, twin retirement) regardless of
                    // level coverage — a level-poor head must not hold
                    // the LB chain active. The level verdict rides the
                    // sealed record instead and gates ONLY the global
                    // registration at drainUpdateGlobalSingleChapter
                    // (D-278).
                    if (this->isProved(StrSpan(addExpression))
                        && ie.validityId == NameMap::MAIN_ID)
                    {
                        // The proven theorem's bytes as a span: the contradiction
                        // form aliases NameMap (no NameMap mint before the sealed
                        // copy below), the reconstruct form rides sArena (held in
                        // theoremHold). Part B seals it onto the shared page set.
                        StrSpan theoremSpan;
                        ScratchString theoremHold;
                        if (memoryBlock.primedForContradiction
                            && memoryBlock.contradictionTheoremId != 0) {
                            theoremSpan = memoryBlock.nameMap.decodeView(
                                memoryBlock.contradictionTheoremId);
                        } else {
                            // reconstructImplication is a D-75 forwarder to
                            // reconstructImplicationFullBind; call the byte-identical
                            // scratch twin directly. The global-key spans alias the
                            // never-deloaded skeletonInterner (frozen in the parallel
                            // phase, no mint here) so they stay valid to their last
                            // use.
                            StrSpan gkSpans[64];
                            const int32_t gkN = blockKeyReader.getGlobalKey(memoryBlock, gkSpans, 64);
                            theoremHold = gl::ImplicationBuilder::reconstructImplicationFullBindScratch(
                                sArena, gkSpans, gkN, StrSpan(addExpression));
                            theoremSpan = StrSpan(theoremHold);
                        }

                        const int64_t pkTBP2 =
                            packStatementKey(ie.originalId, ie.validityId);
                        const int32_t itTBP2Id = memoryBlock.intToBeProved.lookup(pkTBP2);
                        if (itTBP2Id != 0) {
                            if (!parameters.compressor_mode) {
                                {
                                    // Seal the theorem bytes onto the shared page
                                    // set (crosses the parallel -> join boundary;
                                    // the LB deloads after its burst). Not
                                    // internally thread-safe -> under the mutex.
                                    std::lock_guard<std::mutex> lock(this->updateGlobalSingleChapterMutex);
                                    assert(this->updateGlobalSingleChapterPages
                                        && "dischargeToBeProved: updateGlobalSingleChapterPages not emplaced by proveKernel");
                                    SealedPageSet& ps = *this->updateGlobalSingleChapterPages;
                                    ps.appendRecord(UpdateGlobalSingleChapterRec{
                                        SealedString::copyFrom(ps, theoremSpan.ptr,
                                                               theoremSpan.len),
                                        coreId, &memoryBlock, allLevelsInvolved });
                                }
                                memoryBlock.intToBeProved.eraseSet(pkTBP2);
                                NameId addExprScopeId =
                                    memoryBlock.nameMap.encodePush(NameMap::MAIN_ID, StrSpan(addExpression));
                                canonicalizer.initializeCanonicalScope(memoryBlock, addExprScopeId);
                                const StrSpan addExprRootedScope =
                                    memoryBlock.nameMap.decodeView(addExprScopeId);
                                {
                                    // Non-minting: a closing scope was
                                    // opened via encodePush — pre-interned.
                                    const NameId wipeVid =
                                        memoryBlock.nameMap.lookup(addExprRootedScope);
                                    assert(wipeVid != 0);
                                    memoryBlock.pendingWipeScopes.mint(wipeVid);
                                }
                                // The conjecture is settled — retire its
                                // contradiction twins (staged, drained
                                // post-join; I-28). A primed closer retires
                                // its polarity sibling instead.
                                if (memoryBlock.primedForContradiction) {
                                    scopeLifecycle.stageContradictionSiblingTwin(
                                        memoryBlock, sArena);
                                } else {
                                    scopeLifecycle.stageContradictionTwinsOfHead(
                                        memoryBlock, StrSpan(addExpression),
                                        sArena);
                                }
                            }
                        }
                    }
                }

                // Packed read-only probe; this path never erases the goal —
                // it emits at the parent scope and schedules scope wipes.
                const bool tbpGoalPresent = memoryBlock.intToBeProved.lookup(
                    packStatementKey(ie.originalId, ie.validityId)) != 0;

                if (ie.validityId != NameMap::MAIN_ID && tbpGoalPresent) {
                    closeScopedGoal(
                        ie.originalId, ie.validityId, ie.validityId,
                        lvRun, lvN);
                }
            }

            // A fact stored at an ancestor is visible to every descendant
            // scope. Snapshot unresolved goals in decoded lexical order, then
            // scan each ancestry chain root-to-goal so the shallowest visible
            // fact supplies the canonical proof source and level row.
            ScratchScope inheritedGoalScope(gArena);
            RT_SCOPE_HERE("DTP_ANCESTOR_GOAL_SCAN");
            int32_t goalCount = 0;
            int64_t* goalKeys = sortToBeProvedKeys(
                memoryBlock.intToBeProved,
                memoryBlock.nameMap, gArena, goalCount);
            for (int32_t goalIndex = 0;
                 goalIndex < goalCount; ++goalIndex) {
                const StatementKey goalKey =
                    Codec<StatementKey>::decode(goalKeys[goalIndex]);
                if (memoryBlock.phase3IgnoresScope(goalKey.validity)) continue;
                if (goalKey.validity == NameMap::MAIN_ID) continue;

                const int64_t packedGoal =
                    packStatementKey(goalKey.orig, goalKey.validity);
                if (memoryBlock.intToBeProved.lookup(packedGoal) == 0)
                    continue;

                bool removalScheduled = false;
                const int32_t ancestorCount =
                    memoryBlock.nameMap.ancLen(goalKey.validity);
                for (int32_t ancestorIndex = 0;
                     ancestorIndex < ancestorCount; ++ancestorIndex) {
                    const NameId ancestorValidityId =
                        memoryBlock.nameMap.ancAt(
                            goalKey.validity, ancestorIndex);
                    if (memoryBlock.intValidityNamesToFilter.contains(
                            ancestorValidityId)) {
                        removalScheduled = true;
                        break;
                    }
                }
                if (removalScheduled) continue;

                NameId sourceValidityId = 0;
                for (int32_t ancestorIndex = 0;
                     ancestorIndex < ancestorCount; ++ancestorIndex) {
                    const NameId ancestorValidityId =
                        memoryBlock.nameMap.ancAt(
                            goalKey.validity, ancestorIndex);
                    const StatementFlags* sourceFlags =
                        memoryBlock.intKnownStatements.find(
                            StatementKey{
                                goalKey.orig, ancestorValidityId });
                    if (sourceFlags != nullptr) {
                        sourceValidityId = ancestorValidityId;
                        break;
                    }
                }
                if (sourceValidityId == 0) continue;

                const int32_t levelRowId =
                    memoryBlock.intStatementLevelsMap.lookup(
                        packStatementKey(
                            goalKey.orig, sourceValidityId));
                assert(levelRowId != 0
                       && memoryBlock.intStatementLevelsMap.runLen(levelRowId) > 0
                       && "known ancestor statement requires a non-empty level row");
                int sourceLevels[256];
                const int32_t sourceLevelCount = coldIntRunAt(
                    memoryBlock.intStatementLevelsMap,
                    levelRowId, sourceLevels, 256);
                closeScopedGoal(
                    goalKey.orig, sourceValidityId, goalKey.validity,
                    sourceLevels, sourceLevelCount);
            }
        }

/// @brief Does an integration-gate template denote @p goal — verbatim
        ///        or as one of its marker forms?
        ///
        /// @details
        /// `prepareIntegration` keys its `integrationPrepared` /
        /// `integrationPreparedMarker` / `integrationStartIntMap` rows by the
        /// prepared expression and by its MARKER form (each changeable
        /// argument replaced by the literal `marker`; with zero changeable
        /// arguments the marker form is the expression verbatim). The disproof
        /// cleanup cannot re-derive the exact unchangeable-argument set the
        /// original preparation used, but it does not need to: a template
        /// matches the goal iff the (optionally `!`-negated) core names agree,
        /// the arities agree, and every template argument equals the goal's
        /// argument at that slot or the literal `marker`. That covers the
        /// verbatim key and every possible marker variant, and nothing else —
        /// a template of a DIFFERENT expression differs in core, arity, or a
        /// non-`marker` argument.
        ///
        /// @param tmpl The gate template (decoded `templateInterner` text).
        /// @param goal The goal expression (verbatim).
        /// @return `true` iff @p tmpl is the goal or one of its marker forms.
        /// @see drainDisprovedGoals — the sole production consumer.
inline bool GoalProcessor::templateMatchesGoalWithMarkers(StrSpan tmpl,
                                                          StrSpan goal) {
            if (equalSpans(tmpl, goal)) return true;
            const bool tNeg = tmpl.len > 0 && tmpl.ptr[0] == '!';
            const bool gNeg = goal.len > 0 && goal.ptr[0] == '!';
            if (tNeg != gNeg) return false;
            const StrSpan t = tNeg ? StrSpan(tmpl.ptr + 1, tmpl.len - 1) : tmpl;
            const StrSpan g = gNeg ? StrSpan(goal.ptr + 1, goal.len - 1) : goal;
            if (!equalSpans(extractExpressionSpan(t), extractExpressionSpan(g)))
                return false;
            StrSpan tArgs[ExecutionParameters::MAX_ARITY];
            const int32_t tN = getArgsSpans(t, tArgs,
                                            ExecutionParameters::MAX_ARITY);
            StrSpan gArgs[ExecutionParameters::MAX_ARITY];
            const int32_t gN = getArgsSpans(g, gArgs,
                                            ExecutionParameters::MAX_ARITY);
            if (tN != gN || tN == 0) return false;
            for (int32_t i = 0; i < tN; ++i) {
                if (!equalSpans(tArgs[i], gArgs[i])
                    && !equalSpans(tArgs[i], StrSpan("marker", 6)))
                    return false;
            }
            return true;
        }

/// @brief Whether an expression is "proved" — carries no unresolved
        ///        iteration lexeme and no `c<digit>` marker.
        ///
        /// @details
        /// The regex-free twin of the former two `regex_search` existence
        /// tests (`it_\d+_lev_\d+_` and `c\d+`): an expression is proved iff
        /// NEITHER pattern occurs. Uses the `str_ops` existence scanners
        /// `containsItLevPrefixShape` / `containsCDigit`, byte-identical verdicts
        /// to the regex form; the regex machinery survives only as the in-test
        /// oracle (`test_memory.cpp::is_proved_matches_regex_oracle`).
        ///
        /// @param s Expression view to test.
        /// @return `true` iff neither the `it_\d+_lev_\d+_` nor the `c\d+`
        ///         pattern occurs.
        /// @see `str_ops::containsItLevPrefixShape`, `str_ops::containsCDigit`.
inline bool GoalProcessor::isProved(StrSpan s) {
            return !containsItLevPrefixShape(s) && !containsCDigit(s);
        }
}
