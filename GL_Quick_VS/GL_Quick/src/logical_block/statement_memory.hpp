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
#include "../logical_block/scope_syntax.hpp"
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
class AdmissionMemory;
class CaseLedger;
class CaseProcessor;
class CompactRuleProcessor;
class DisintegrationProcessor;
class EqualityPropagation;
class EquivalenceProcessor;
class EquivalenceQueries;
class IntegrationProcessor;
class OrImplicationProcessor;
class ProofHistory;
class UnaryIntegrationProcessor;
class WatchProcessor;
namespace tests { struct StatementMemoryFixture; }
/// @brief Statement Memory operations for the proof algorithm.
/// @details Uses explicit peer interfaces while the supplied logical blocks retain storage and proof state.
class StatementMemory {
friend struct tests::StatementMemoryFixture;
public:
/// @brief Bind inputs for statement memory.
/// @details Construction stores explicit references; block storage and all invocation lifetimes remain with their owners.
/// @param admissionPolicy Borrowed AdmissionPolicy interface or state; must outlive this component.
/// @param canonicalizer Borrowed Canonicalizer interface or state; must outlive this component.
/// @param definitions Borrowed CompiledDefinitionRegistry interface or state; must outlive this component.
/// @param parameters Borrowed ProverParameters interface or state; must outlive this component.
/// @param admissionMemory Borrowed AdmissionMemory interface or state; must outlive this component.
/// @param caseLedger Borrowed CaseLedger interface or state; must outlive this component.
/// @param caseProcessor Borrowed CaseProcessor interface or state; must outlive this component.
/// @param compactRuleProcessor Borrowed CompactRuleProcessor interface or state; must outlive this component.
/// @param disintegrationProcessor Borrowed DisintegrationProcessor interface or state; must outlive this component.
/// @param equalityPropagation Borrowed EqualityPropagation interface or state; must outlive this component.
/// @param equivalenceProcessor Borrowed EquivalenceProcessor interface or state; must outlive this component.
/// @param equivalenceQueries Borrowed EquivalenceQueries interface or state; must outlive this component.
/// @param integrationProcessor Dis value used by this operation.
/// @param orImplicationProcessor Borrowed OrImplicationProcessor interface or state; must outlive this component.
/// @param proofHistory Borrowed ProofHistory interface or state; must outlive this component.
/// @param unaryIntegrationProcessor Borrowed UnaryIntegrationProcessor interface or state; must outlive this component.
/// @param watchProcessor Borrowed WatchProcessor interface or state; must outlive this component.
/// @return None; construction binds the supplied references.
StatementMemory(AdmissionPolicy& admissionPolicy, Canonicalizer& canonicalizer, const CompiledDefinitionRegistry& definitions, const ProverParameters& parameters, AdmissionMemory& admissionMemory, CaseLedger& caseLedger, CaseProcessor& caseProcessor, CompactRuleProcessor& compactRuleProcessor, DisintegrationProcessor& disintegrationProcessor, EqualityPropagation& equalityPropagation, EquivalenceProcessor& equivalenceProcessor, EquivalenceQueries& equivalenceQueries, IntegrationProcessor& integrationProcessor, OrImplicationProcessor& orImplicationProcessor, ProofHistory& proofHistory, UnaryIntegrationProcessor& unaryIntegrationProcessor, WatchProcessor& watchProcessor);

public:


public:


public:
//#pragma optimize("", off)

        /// @brief Admit a finalized statement and its paired known/level state.
        /// @details Applies the existing scope, variable, canonicalization and
        /// shape gates. OR source intent activates only after successful
        /// registration, including an already-known nonlocal statement.
        /// @param expr Expression bytes to admit.
        /// @param memoryBlock Owning logic block.
        /// @param local Whether this block owns the derivation.
        /// @param levels Sorted derivation-level run.
        /// @param levelCount Number of levels; zero means non-derived input.
        /// @param origin Optional history row for the admitted form.
        /// @param validityName Source validity scope.
        /// @param newStatements Output rows for subsequent local processing.
        /// @param activateOrSource Whether final disintegration admitted an OR source.
        /// @param publicationMail Current phase output, or same-iteration mail for ordinary callers.
        /// @param negatedExpansionReady Optional inequality continuation output: true only when
        /// registration reaches expansion. The caller completes that depth-first
        /// expansion before its post-return publication; ordinary callers leave it null.
        /// @return Void.
inline void
            addStatement(StrSpan expr,
                Memory& memoryBlock,
                bool local,
                const int* levels, int32_t levelCount,
                const TransientOrigin& origin,
                StrSpan validityName,
                PagedVector<IntEncodedExpr>& newStatements,
                bool activateOrSource = false, ColdMail* publicationMail = nullptr,
                bool* negatedExpansionReady = nullptr);

public:
// Turn optimizations OFF for just this section
//#pragma optimize("", off)
/// @brief Public entry to install an expression into an LB — the main
/// kernel of the prover.
///
/// @details
/// Every expression that lands in any LB goes through this function:
/// fresh statements from the prover loop, mail-in deposits from the
/// hash burst, head emissions from `checkLocalEncodedMemoryStatic`,
/// and disintegration products from `disintegrateExpr2`. The function
/// dispatches by expression shape:
///
/// 1. **Atomic predicate** (e.g. `(p[a])`) — installs as a statement
///    via the kernel; updates `intEncodedStatements`,
///    `intEncodedStatements`, `intLocalEncodedStatementsSet`, and
///    `intKnownStatements`.
/// 2. **Equality** (e.g. `(=[a,b])`) — routes through `addEquality`
///    so the equivalence-class machinery can absorb it
///    (with mirror per [I-9](../../docs/agentic_swdd/30_invariants.md#i-9)).
/// 3. **Negated equality** (e.g. `!(=[a,b])`) — routes through
///    `addNegatedEquality` for the one-sided expansion per
///    [I-12](../../docs/agentic_swdd/30_invariants.md#i-12).
/// 4. **Implication** (e.g. `(>[v](p[v])(q[v]))`) — installed as a
///    hash rule via `addToHashMemory` (after disintegration when
///    appropriate).
///
/// Anchor slot typing is NOT dispatched here: the x-prefixed anchor
/// variants are installed grid-wide by the `prehandleAnchor` pre-pass
/// before the iteration loop runs.
///
/// The `status` mode selector controls the deposit path; the full
/// table lives in [`docs/agentic_swdd/10_pipeline/04_prover.md`](../../docs/agentic_swdd/10_pipeline/04_prover.md)
/// (`addExprToMemoryBlock — status parameter reference`). Statuses
/// 0/1 disintegrate via `disintegrateExpr2`. **status 3** (external-mail
/// absorb, ASIC 0.1 reshuffle) is admitted to `disintegrateExpr2` ONLY
/// when the expression is a compact implication `(implication<N>...)`
/// (gate at the call site, OUTSIDE disintegrateExpr2); a plain mail
/// statement is absorbed as a statement, never disintegrated. Recovered
/// implications install through the usual persistent 3-way
/// `addToHashMemory` path.
///
/// **Per [I-27](../../docs/agentic_swdd/30_invariants.md#i-27)**, ancestor-scan
/// dedupe runs at this entry (Site F / Site H): if the expression
/// already exists in any ancestor scope, the deposit is skipped.
/// This is what keeps cross-LB writes during the parallel phase
/// from causing duplicate-deposit storms
/// ([I-28](../../docs/agentic_swdd/30_invariants.md#i-28)).
///
/// @param expr           Canonical MPL expression text.
/// @param memoryBlock    Target LB.
/// @param iteration      Witness generation carried by the arrival.
/// @param status         Deposit route selecting the processing contract.
/// @param involvedLevels Ascending-unique level run — LB levels at which
///                       the expression is admissible (caller-owned per
///                       the levels-run chain contract).
/// @param involvedLevelCount The level count (`>= 0`).
/// @param origin         Recorded derivation of the deposit.
/// @param coreId         Worker slot; -1 selects the setup slot.
/// @param auxyIndex      Auxiliary goal registration index.
/// @param validityName   Scope in which the expression is offered.
/// @param doNotDisintegrate Suppress disintegration for this offer.
/// @param allowOrDisintegration Permit rule-product OR cohort opening.
/// @param registeredOut  Optional encoded registration result.
/// @param recursionHead  Force-disintegrate the induction implication's head.
/// @param convergenceMailOut Explicit convergence output during input absorption;
///                       nullptr selects same-iteration mail for ordinary callers.
/// @param replayKinds    Queued construction replay bits (bit 8 selects the
///                       flag-5 relay route).
/// @param replayIntegrationDepth Original assigned depth of a replayed source,
///                       or -1.
/// @param integrationPremise The offer is a premise of an implication being
///                       integrated: an OR compact opens its case split, a
///                       negated conjunction splits into its complemented
///                       conjuncts (D-460).
/// @param conjecturePremise The offer is a conjecture's non-anchor premise
///                       seeded into its block: every OR it unfolds into,
///                       through existence unfolding, opens its case split
///                       with the rule-product permission where this block has
///                       proof work (D-495), and an OR published at main is
///                       staged for the flag-6 unconditional-ordis mail to
///                       every descendant (D-528).
/// @return Nothing; writes the expression through the ordinary deposit doors.
/// @invariant [I-9](../../docs/agentic_swdd/30_invariants.md#i-9) — addEquality
///            mirror guard.
/// @invariant [I-12](../../docs/agentic_swdd/30_invariants.md#i-12) — !(=[a,b])
///            one-sided expansion.
/// @invariant [I-27](../../docs/agentic_swdd/30_invariants.md#i-27) — Site F / H
///            ancestor-scan dedupe at entry.
/// @invariant [I-28](../../docs/agentic_swdd/30_invariants.md#i-28) — cross-LB
///            writes deferred to post-`pool.join()` collectors.
/// @see `disintegrateExpr2` — structural disintegration.
/// @see `addToHashMemory` (memory.cpp) — implication branch.
/// @see `addEquality`, `addNegatedEquality` — shape-specific branches.
void addExprToMemoryBlock(StrSpan expr,
            Memory& memoryBlock,
            int iteration,
            int status,
            const int* involvedLevels, int32_t involvedLevelCount,
            const TransientOrigin& origin,
            int coreId,
            int auxyIndex,
            StrSpan validityName,
            bool doNotDisintegrate,
            bool allowOrDisintegration = false,
            IntEncodedExpr* registeredOut = nullptr,
            bool recursionHead = false,
            ColdMail* convergenceMailOut = nullptr,
            int32_t replayKinds = 0, int32_t replayIntegrationDepth = -1,
            bool integrationPremise = false,
            bool conjecturePremise = false);

public:


public:



public:
/// @brief End-of-burst ancestor-known sweep: drop statement-LIST rows at
///        non-main scopes whose text a strict ancestor knows
///        (I-187).
///
/// @details
/// Detection first, mutation second: one read-only pass over
/// `intEncodedStatements` collects the packed `(originalId, validityId)`
/// keys of every non-main row whose text `ancestorKnown` reports at a
/// strict ancestor; the hits then drop in decoded-lex
/// `(original, validity)` order (I-84) through
/// `gl::StatementRegistryOperations::removeExpressionFromMemoryBlock(state=0)` — the statement lists only,
/// registry rows surviving as tombstones. The decodeView spans handed to
/// the removal door stay valid because the door only does non-minting
/// lookups and `PagedVector` erases — no NameMap mint (I-3). Scratch
/// (hit list on the page tier, sort index on the byte-bump tier) rides
/// the per-slot gen-scratch arena, reclaimed before return. Full
/// contract at the declaration.
///
/// @param body Owning LB; statement lists and waterlines mutated in place
///             through the removal door.
/// @invariant Registry rows are never erased here (I-58 / I-85).
/// @see sweepAncestorKnownRows (decl) — placement and tombstone rationale.
void sweepAncestorKnownRows(Memory& body);

public:

public:
/// @brief Whether a span is a positive instance of a configured
        ///        definition whose compiled category is `implication`.
        ///
        /// @details
        /// A configured definition whose whole body is one implication
        /// (`primeseq`) is itself the compact form of its rule: the public
        /// name replaces any numbered equivalent, so the block that expands
        /// it has no separate `implication<N>` compact to mail. An inherited
        /// statement of this shape is therefore a rule carrier like
        /// `(implication<N>[…])`, and the deposit door expands it
        /// (D-467). "Configured" is
        /// `expressionsFromConfig` membership, filled before generated
        /// compacts extend `coreExpressionMap`; numbered compacts are matched
        /// by `isCompactImplicationSpan`. Zero allocation: the name is a span
        /// of the input, both probes are transparent lookups.
        ///
        /// @param s The deposited expression.
        /// @return `true` exactly for `(name[…])` with `name` configured and
        ///         compiled as category `implication`; an atomic configured
        ///         operator has no compiled entity and is no carrier.
        /// @see `isCompactImplicationSpan`, `addExprToMemoryBlock`.
        bool isConfiguredImplicationCarrier(StrSpan s) const;

/// @brief Record an int scope's finish when a statement registered there is
        ///        an instance of its integration goal, and promote the group's
        ///        first finished scopes once every premise has one.
        ///
        /// @details
        /// An int scope (`LbMemory::intScopes`) finishes when a statement
        /// registered at the scope itself is an instance of its goal group's
        /// goal (`ScopeSyntax::intScopeGoalMatches`, the goal in its canonical
        /// form at the scope; for a scope of the second twin premise the goal
        /// with `rec2`, the name of the twin partner's predecessor, for `rec`,
        /// D-589). Only the first finished scope of each premise of
        /// the group counts (`LbMemory::intScopeFirstFinish`); later finishes
        /// of that premise change nothing. When every premise level of the
        /// group (`LbMemory::intScopeGroups`) has its first finished scope, the
        /// group is marked promoted and those scopes are promoted together in
        /// decoded-name order (@ref promoteIntScope); then every int scope of
        /// the group, promoted or not, is queued on `pendingWipeScopes`, and
        /// the closed group opens no further scope
        /// (D-578, D-582).
        ///
        /// @param memoryBlock     Owning logic block (single-threaded seam).
        /// @param originalId      The registered statement's id.
        /// @param scopeId         Its scope.
        /// @param newStatements   The registration's output rows; the promoted
        ///                        rows are appended.
        /// @param publicationMail The registration's output channel.
        /// @return Void.
        /// @invariant Mints into the NameMap only through the promoted
        ///            registrations.
        /// @see promoteIntScope, AdmissionMemory::openIntScope.
inline void finishIntScope(Memory& memoryBlock, NameId originalId, NameId scopeId,
            PagedVector<IntEncodedExpr>& newStatements, ColdMail* publicationMail);

/// @brief Register every statement of a finished int scope at the scope's
        ///        parent.
        ///
        /// @details
        /// Every statement registered at @p scopeId itself (nested scopes
        /// excluded) and not known at its parent is registered at the parent
        /// through @ref addStatement, in registration order, with its own
        /// levels and locality and the raw-only `int scope` origin citing the
        /// statement at the scope, as the induction goal exit does. The rows
        /// are snapshot first: the registrations append to the registry
        /// (D-578).
        ///
        /// @param memoryBlock     Owning logic block (single-threaded seam).
        /// @param scopeId         An open int scope (asserted).
        /// @param newStatements   The registration's output rows.
        /// @param publicationMail The registration's output channel.
        /// @return Void.
        /// @invariant Heap-free: the row snapshot rides the gen-scratch page
        ///            tier, the texts the string-scratch tier.
        /// @see finishIntScope, OriginTag::intScope.
inline void promoteIntScope(Memory& memoryBlock, NameId scopeId,
            PagedVector<IntEncodedExpr>& newStatements, ColdMail* publicationMail);
private:

AdmissionPolicy& admissionPolicy;
Canonicalizer& canonicalizer;
const CompiledDefinitionRegistry& definitions;
const ProverParameters& parameters;
AdmissionMemory& admissionMemory;
CaseLedger& caseLedger;
CaseProcessor& caseProcessor;
CompactRuleProcessor& compactRuleProcessor;
DisintegrationProcessor& disintegrationProcessor;
EqualityPropagation& equalityPropagation;
EquivalenceProcessor& equivalenceProcessor;
EquivalenceQueries& equivalenceQueries;
IntegrationProcessor& integrationProcessor;
OrImplicationProcessor& orImplicationProcessor;
ProofHistory& proofHistory;
UnaryIntegrationProcessor& unaryIntegrationProcessor;
WatchProcessor& watchProcessor;
};
}
#include "../logical_block/admission_memory.hpp"
#include "../logical_block/case_ledger.hpp"
#include "../logical_block/case_processor.hpp"
#include "../logical_block/compact_rule_processor.hpp"
#include "../logical_block/disintegration_processor.hpp"
#include "../logical_block/equality_propagation.hpp"
#include "../logical_block/equivalence_processor.hpp"
#include "../logical_block/equivalence_queries.hpp"
#include "../logical_block/integration_processor.hpp"
#include "../logical_block/or_implication_processor.hpp"
#include "../logical_block/proof_history.hpp"
#include "../logical_block/unary_integration_processor.hpp"
#include "../logical_block/watch_processor.hpp"
namespace gl {





//#pragma optimize("", off)

        /// @brief Admit a finalized statement and its paired known/level state.
        /// @details Applies the existing scope, variable, canonicalization and
        /// shape gates. OR source intent activates only after successful
        /// registration, including an already-known nonlocal statement.
        /// @param expr Expression bytes to admit.
        /// @param memoryBlock Owning logic block.
        /// @param local Whether this block owns the derivation.
        /// @param levels Sorted derivation-level run.
        /// @param levelCount Number of levels; zero means non-derived input.
        /// @param origin Optional history row for the admitted form.
        /// @param validityName Source validity scope.
        /// @param newStatements Output rows for subsequent local processing.
        /// @param activateOrSource Whether final disintegration admitted an OR source.
        /// @param publicationMail Current phase output, or same-iteration mail for ordinary callers.
        /// @param negatedExpansionReady Optional inequality continuation output: true only when
        /// registration reaches expansion. The caller completes that depth-first
        /// expansion before its post-return publication; ordinary callers leave it null.
        /// @return Void.
inline void
            StatementMemory::addStatement(StrSpan expr,
                Memory& memoryBlock,
                bool local,
                const int* levels, int32_t levelCount,
                const TransientOrigin& origin,
                StrSpan validityName,
                PagedVector<IntEncodedExpr>& newStatements,
                bool activateOrSource, ColdMail* publicationMail,
                bool* negatedExpansionReady) {
            if (negatedExpansionReady != nullptr) {
                assert(canonicalizer.isNegatedEquality(expr) && !activateOrSource);
                *negatedExpansionReady = false;
            }
            // `newStatements` is the caller-owned id-form out-param — PagedVector is
            // non-movable, so addStatement appends in place instead of returning.
            // Each row's decoded (originalId, validityId) IS the former
            // ExpressionWithValidity pair (expression + scope); the kernel's
            // post-`addStatement` loop consumes emission order and decodes each
            // row's `validityId` for its intStatementLevelsMap lookup,
            // so cross-scope deposits (descendant or ancestor direction) ride out
            // through this single channel without tripping the kernel's same-scope
            // assert.

            // Non-derived tier: a statement arriving with NO level provenance
            // (loaded fact, broadcast theorem compact, assumed premise element)
            // carries the singleton {-1} — a levels row is never empty. The
            // run invariant this door pins: a level run is either exactly
            // {-1} or all-values-≥0, never mixed; -1 is transparent to level
            // accounting (`insertLevelSorted` / `coldIntRunNonNegAt`), so a
            // union of non-derived inputs comes back empty and re-enters
            // here as {-1} again.
            static const int kNonDerivedRun[1] = { -1 };
            if (levelCount == 0) {
                levels = kNonDerivedRun;
                levelCount = 1;
            }
            assert(((levelCount == 1 && levels[0] == -1) || levels[0] >= 0)
                && "statement level run mixes the non-derived tier with real levels");

            // Span twin: parse + encode in one pass off the stable expr/
            // validityName buffers, no intermediate EncodedExpression heap.
            IntEncodedExpr ieStmt =
                encodeExpression(expr, StrSpan(validityName), memoryBlock.nameMap);
            if (memoryBlock.phase3IgnoresScope(ieStmt.validityId)) return;
            canonicalizer.initializeCanonicalScope(memoryBlock, ieStmt.validityId);
            // Axed-variable containment at THIS door too (x only in
            // anchors): disintegration products, equalities and every other
            // registration that did not pass the addExprToMemoryBlock
            // prologue are refused here on an armed x-argument — same
            // predicate and positive-anchor exemption as the door (D-234).
            if (!(parameters.axed_anchor_exception && ieStmt.isAnchor
                  && !ieStmt.negation)) {
                for (NameId a = 0; a < ieStmt.arity; ++a) {
                    if (memoryBlock.intAxedVariables.contains(
                            ieStmt.argFullId[a])) {
                        return;
                    }
                }
            }

            // Ancestor-known refusal — the Site F contract enforced at this
            // door (D-273): an expression already
            // known at a STRICT ancestor of the deposit scope is fully
            // visible there through comparable-scope inheritance, so the
            // descendant copy is refused outright — no registration, no
            // equality-pair dispatch, no arrival-time expansion, no
            // `newStatements` row (the kernel's per-statement follow-up
            // loop consequently skips it, matching an entry-level Site F
            // refusal, whose or-convergence outcome I-27 documents as
            // correct). Exact-scope handling stays with the pre-existing
            // gates below. The or-branch resolution signal survives the
            // refusal (I-174).
            if (!parameters.compressor_mode
                && gl::StatementKnowledge::ancestorKnown(memoryBlock, ieStmt.originalId,
                                 ieStmt.validityId, /*includeSelf=*/false)) {
                caseProcessor.stageOrReleaseForRefusedDeposit(memoryBlock,
                    ieStmt.originalId, ieStmt.validityId);
                return;
            }
            const unsigned canonicalSlot = gl::WorkerContext::coreId() >= 0
                ? static_cast<unsigned>(gl::WorkerContext::coreId()) : scratchArenas().slotCount() - 1;
            ScratchArena& canonicalArena = scratchArenas().forSlot(canonicalSlot);
            ScratchScope canonicalScope(canonicalArena);
            int canonicalLevels[256];
            assert(levelCount <= 256);
            std::copy(levels, levels + levelCount, canonicalLevels);
            int32_t canonicalLevelCount = levelCount;
            TransientOrigin registeredOrigin = origin;
            const CanonicalForm unusedForm = equalityPropagation.canonicalizeUnusedDeposit(ieStmt,
                expr, validityName, memoryBlock, local, canonicalLevels,
                canonicalLevelCount, origin, canonicalArena);
            if (unusedForm.changed) {
                expr = unusedForm.text;
                levels = canonicalLevels;
                levelCount = canonicalLevelCount;
                registeredOrigin = TransientOrigin{};
                ieStmt = encodeExpression(expr, validityName, memoryBlock.nameMap);
                for (NameId argument = 0; argument < ieStmt.arity; ++argument) {
                    if (memoryBlock.intAxedVariables.contains(ieStmt.argFullId[argument])) return;
                }
                if (!parameters.compressor_mode && gl::StatementKnowledge::ancestorKnown(memoryBlock,
                        ieStmt.originalId, ieStmt.validityId, false)) {
                    caseProcessor.stageOrReleaseForRefusedDeposit(memoryBlock,
                        ieStmt.originalId, ieStmt.validityId);
                    return;
                }
            }


            // Gate the equality-mirror push on `local` (= status 0 or 1 in the
            // kernel). Local additions go through addEquality(allowSymmetry=true)
            // which registers the mirror in intStatementLevelsMap; the post-
            // addStatement loop's lookup then succeeds. Non-local paths (mail-in
            // absorb status=3) call addEquality(allowSymmetry=false) — no mirror
            // entry exists, so pushing the mirror onto newStatements would trip
            // the intStatementLevelsMap assert at prover.cpp::addExprToMemoryBlockKernel.
            // Restores the toBeProved-discharge side-effect that 46b4ecce's
            // unconditional `&& false` killed (cost: 366 incubator-Peano theorems).
            if (canonicalizer.isEquality(expr) && local) {
                StrSpan args[ExecutionParameters::MAX_ARITY];
                const int argN = getArgsSpans(expr, args, ExecutionParameters::MAX_ARITY);
                assert(argN == 2);
                // "(=[a1,a0])" on the per-slot string-scratch arena — no heap
                // std::string. args alias the caller-stable expr and
                // encodeExpression mints into nameMap (not this arena), so the
                // span stays live to the push_back (I-3); the ScratchScope
                // reclaims it before the shape dispatch below.
                const unsigned mSlot = (gl::WorkerContext::coreId() >= 0)
                    ? static_cast<unsigned>(gl::WorkerContext::coreId())
                    : scratchArenas().slotCount() - 1;
                ScratchArena& mArena = scratchArenas().forSlot(mSlot);
                ScratchScope mScope(mArena);
                const int32_t mirrorLen = 3 + args[1].len + 1 + args[0].len + 2;
                char* mBuf = mArena.allocBytes(mirrorLen);
                int32_t mAt = 0;
                mBuf[mAt++] = '('; mBuf[mAt++] = '='; mBuf[mAt++] = '[';
                std::memcpy(mBuf + mAt, args[1].ptr,
                            static_cast<std::size_t>(args[1].len));
                mAt += args[1].len;
                mBuf[mAt++] = ',';
                std::memcpy(mBuf + mAt, args[0].ptr,
                            static_cast<std::size_t>(args[0].len));
                mAt += args[0].len;
                mBuf[mAt++] = ']'; mBuf[mAt++] = ')';
                assert(mAt == mirrorLen);
                // Id form: encodeExpression FINDS expr's already-minted name/args
                // and FIRST-mints only the mirror's originalId — the same id
                // addEquality's mirror registration would mint next (nothing mints
                // between here and there — verified), so the NameMap id order is
                // byte-identical. Sorted before use, so push order is immaterial.
                newStatements.push_back(encodeExpression(
                    StrSpan(mBuf, mirrorLen), StrSpan(validityName), memoryBlock.nameMap));
            }

            // Shape dispatch + statement registration. Moved
            // here from addExprToMemoryBlock's disintegration-product
            // stmts loop, where the equivalent block was wrapped in
            // `if (status != 2)`. `addStatement` has exactly one call
            // site (prover.cpp::addExprToMemoryBlock at the post-
            // disintegration `stmts` loop) and only statuses 0, 1, 3
            // reach that loop (status 2 and status 4 return from the
            // kernel before it), so no gate is needed here. The two
            // helpers stay as private members; this is the single
            // dispatch entry.
            if (canonicalizer.isEquality(expr)) {
                RT_SCOPE_HERE("ADDSTMT_ADD_EQUALITY");
                equivalenceProcessor.addEquality(expr, memoryBlock, local, levels, levelCount, origin, validityName);
            }
            else if (canonicalizer.isNegatedEquality(expr)) {
                RT_SCOPE_HERE("ADDSTMT_ADD_NEGATED_EQUALITY");
                equivalenceProcessor.addNegatedEquality(expr, memoryBlock, local, levels, levelCount, origin, validityName);
            }
            // No unconditional registration: a row in intKnownStatements
            // means ADMITTED. A statement the gates below refuse (iteration
            // cap, secondary-variable cap, equivalence filter) leaves no
            // row and no levels — it simply is not known here.
            const int maxIteration = admissionPolicy.extractMaxIterationNumber(expr);
            if (maxIteration != -1 && maxIteration > parameters.maxIterationNumberVariable) {
                return;
            }

            {
                RT_SCOPE_HERE("ADDSTMT_COUNT_PATTERNS");
                if (admissionPolicy.countPatternOccurrences(expr, memoryBlock.overallHashMemory, memoryBlock.nameMap) >
                    parameters.maxNumberSecondaryVariables) {
                    return;
                }
            }

            if (!canonicalizer.isEquality(expr) || parameters.skip_eq_classes) {
                // not in memory already? — packed probe via the entry ids.
                if (memoryBlock.intStatementLevelsMap.lookup(
                        packStatementKey(ieStmt.originalId, ieStmt.validityId)) == 0) {

                    // Check if any equivalence class filters the expr out.
                    // Zero-copy view walk over the validity's class run — the
                    // former decodeClassesById heap snapshot. filterIterations
                    // mutates only eqClassNameCaches (the token memo via
                    // tokensViewOf), NEVER equivalenceClassesMap, so a peeked view
                    // stays valid across the call; filterIterations' own genArena
                    // mark/pop nests ABOVE this class-view peek (LIFO-safe).
                    // decodeClassesById returned {} at bucketId==0 -> the loop
                    // never ran -> anyFilteredOut false: the bucketId!=0 guard
                    // reproduces that. anyFilteredOut is an OR over all classes, so
                    // the early break is byte-identical.
                    bool anyFilteredOut = false;
                    const int32_t ieBucketId =
                        memoryBlock.canonicalEquivalenceClassesMap.lookup(ieStmt.validityId);
                    if (ieBucketId != 0) {
                        const unsigned slot = (gl::WorkerContext::coreId() >= 0)
                            ? static_cast<unsigned>(gl::WorkerContext::coreId())
                            : genScratchArenas().slotCount() - 1;
                        ScratchArena& gArena = genScratchArenas().forSlot(slot);
                        const int32_t cc =
                            memoryBlock.canonicalEquivalenceClassesMap.runLen(ieBucketId);
                        for (int32_t ci = 0; ci < cc; ++ci) {
                            ScratchScope viewScope(gArena);
                            int32_t blen = 0;
                            const char* bp =
                                memoryBlock.canonicalEquivalenceClassesMap.peekRecordBytes(
                                    ieBucketId, ci, blen, gArena);
                            const EquivalenceClassView view{ bp, blen };
                            if (!equalityPropagation.filterIterations(ieStmt.originalId, view, memoryBlock)) {
                                anyFilteredOut = true; break;
                            }
                        }
                    }

                    if (!anyFilteredOut) {
                        memoryBlock.intStatementLevelsMap.assignSetRange(
                            packStatementKey(ieStmt.originalId, ieStmt.validityId),
                            levels, levels + levelCount);
                        upsertStatementKey(memoryBlock.intKnownStatements,
                            packStatementKey(ieStmt.originalId, ieStmt.validityId),
                            local);
                        watchProcessor.recordNegationWatchFor(memoryBlock, ieStmt);
                        watchProcessor.stageWatchArrival(memoryBlock, ieStmt);

                        if (parameters.trackHistory && registeredOrigin.present) {
                            assert(registeredOrigin.tag != OriginTag::COUNT);
                            // addOrigin handles existence checks, vector push_back,
                            // and respects the compressor_max_origins_per_expr limit.
                            addOriginEncoded(memoryBlock.exprOriginMap, memoryBlock.originInterner, expr, StrSpan(validityName), registeredOrigin.tag, registeredOrigin.deps, registeredOrigin.depN, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
                            // mailOut.exprOriginMap addOrigin is driven by
                            // `fillMailOut`.
                        }

                        { const IntEncodedExpr& ie = ieStmt;
                          memoryBlock.intEncodedStatements.push_back(ie);

                        if (local) {
                            memoryBlock.intLocalEncodedStatementsSet.mint(
                                packStatementKey(ie.originalId, ie.validityId));
                            memoryBlock.intLocalEncodedStatements.push_back(ie);
                            memoryBlock.intLocalEncodedStatementsDelta.push_back(ie);
                            newStatements.push_back(ie);   // id form (== ieStmt, minted at entry)
                            // mailOut.statements + mailOut.exprOriginMap
                            // are driven by `fillMailOut`.
                        }
                        } // close IntEncodedExpr ie scope
                    }
                }

                // Apply each equivalence class to generate more statements from expr.
                // Same-NS classes first (existing behaviour), then — additively —
                // classes at every strict-ancestor validity of `validityName`.
                // The ancestor-class branch passes the DESCENDANT's validityName
                // as the `validityName` argument so deposit bookkeeping (levels,
                // origins, eqClassSttmntIndexMapMap keying, reduceEqClass weak
                // filter) lives at the expression's scope.
                if (!parameters.skip_eq_classes) {
                    // All three namespace directions (same-NS, ancestor,
                    // descendant) are applied in `applyEquiClasses`. The
                    // delta-class tracker `Memory::changedClassesThisStep`
                    // records each merged class in
                    // `updateEquivalenceClasses`; `applyEquiClasses`
                    // (called once per elementary step
                    // between mail-absorb-clear and request generation)
                    // re-applies the delta classes against every
                    // comparable `encodedStatement` (same-NS / ancestor /
                    // descendant) and updates the 4 admission / rejected
                    // map helpers per class. Negated-equality handling
                    // stays below.
                    if (canonicalizer.isNegatedEquality(expr)) {
                        RT_SCOPE_HERE("ADDSTMT_APPLY_TO_NEGEQ");
                        if (negatedExpansionReady != nullptr) {
                            *negatedExpansionReady = true;
                        } else {
                            equalityPropagation.applyEquivalenceClassToNegatedEquality(expr,
                                memoryBlock, local, levels, levelCount,
                                newStatements, validityName);
                        }
                    }
                }
            }
            else {
                RT_SCOPE_HERE("ADDSTMT_UPDATE_EQUI_CLASSES");
                equivalenceProcessor.updateEquivalenceClasses(memoryBlock, expr, levels, levelCount, origin, newStatements, validityName);
                newStatements.push_back(ieStmt); // also add the equality itself (id form, minted at entry)
            }

            if (activateOrSource && memoryBlock.intKnownStatements.find(
                    StatementKey{ieStmt.originalId, ieStmt.validityId}) != nullptr) {
                orImplicationProcessor.registerOrImplicationSource(memoryBlock, ieStmt, publicationMail);
            }

            // A goal hit inside an induction scope leaves the tree in the
            // same absorb: the statement is registered at the tree's root as
            // well, with the raw-only `induction scope` row citing the leaf,
            // so this absorb's discharge pass closes the goal in the burst
            // that derived it (D-519).
            NameId exitRoot = 0;
            if (memoryBlock.intKnownStatements.find(
                    StatementKey{ieStmt.originalId, ieStmt.validityId}) != nullptr
                && gl::ScopeSyntax::inductionGoalExit(memoryBlock, ieStmt.originalId,
                    ieStmt.validityId, exitRoot)
                && memoryBlock.intKnownStatements.find(
                    StatementKey{ieStmt.originalId, exitRoot}) == nullptr) {
                const unsigned exitSlot = gl::WorkerContext::coreId() >= 0
                    ? static_cast<unsigned>(gl::WorkerContext::coreId()) : scratchArenas().slotCount() - 1;
                ScratchArena& exitArena = scratchArenas().forSlot(exitSlot);
                ScratchScope exitScope(exitArena);
                // Owned copy: the root registration below mints the NameMap
                // (I-3); `expr` and `validityName` are stable inputs.
                const StrSpan rootView = memoryBlock.nameMap.decodeView(exitRoot);
                const ScratchString rootValidity =
                    ScratchString::copyFrom(exitArena, rootView.ptr, rootView.len);
                const OriginDep leafDeps[1] = { { expr, validityName } };
                const TransientOrigin exitOrigin{ true, OriginTag::inductionScope, leafDeps, 1 };
                addStatement(expr, memoryBlock, local, levels, levelCount, exitOrigin,
                    StrSpan(rootValidity), newStatements, false, publicationMail);
            }

            // A statement registered at an int scope may finish it; the
            // group's first finished scopes are promoted to their parents
            // together (D-578).
            if (memoryBlock.lbMemory.intScopes.count() != 0
                && memoryBlock.intKnownStatements.find(
                    StatementKey{ieStmt.originalId, ieStmt.validityId}) != nullptr)
                finishIntScope(memoryBlock, ieStmt.originalId, ieStmt.validityId, newStatements, publicationMail);

            // The twin partner's predecessor q registered at `main` or in the
            // twin successor case of a twin weak step block,
            // `(in2[q,<partner>,<successor>])`, files the second twin
            // premise's integration goal there, with the copy `(=[q,rec2])`
            // (D-589, D-581).
            if (memoryBlock.twinFirstGoalId != 0 && memoryBlock.twinPredecessorId == 0
                && (ieStmt.validityId == NameMap::MAIN_ID
                    || (memoryBlock.twinCaseScopeId != 0 && ieStmt.validityId == memoryBlock.twinCaseScopeId))
                && ieStmt.negation == 0 && ieStmt.arity == 3
                && ieStmt.argFullId[1] == memoryBlock.twinPartnerId
                && ieStmt.argFullId[2] == memoryBlock.twinSuccessorId
                && ieStmt.argFullId[0] != memoryBlock.twinPartnerId
                && equalSpans(memoryBlock.nameMap.decodeView(ieStmt.nameId), StrSpan("in2", 3))
                && memoryBlock.intKnownStatements.find(
                    StatementKey{ieStmt.originalId, ieStmt.validityId}) != nullptr)
                admissionMemory.fileTwinSecondGoal(memoryBlock, ieStmt.argFullId[0], ieStmt.validityId);

            // In a base-case block, the partner's predecessor registered in
            // the released successor case files there the demands for the
            // partner's values of the second twin premise's functions, Q's
            // last entry for the full I5
            // (D-588).
            if (memoryBlock.twinExperiment == TwinExperiment::successorCase && !memoryBlock.twinValueDemandsFiled
                && memoryBlock.twinCaseScopeId != 0 && ieStmt.validityId == memoryBlock.twinCaseScopeId
                && ieStmt.negation == 0 && ieStmt.arity == 3
                && ieStmt.argFullId[1] == memoryBlock.twinPartnerId
                && ieStmt.argFullId[2] == memoryBlock.twinSuccessorId
                && ieStmt.argFullId[0] != memoryBlock.twinPartnerId
                && equalSpans(memoryBlock.nameMap.decodeView(ieStmt.nameId), StrSpan("in2", 3))
                && memoryBlock.intKnownStatements.find(
                    StatementKey{ieStmt.originalId, ieStmt.validityId}) != nullptr)
                admissionMemory.fileTwinValueDemands(memoryBlock, ieStmt.validityId);


            if (!parameters.skip_eq_classes) {
                // The fixpoint over newly-generated statements runs in
                // `applyEquiClasses`. Its driver uses a
                // per-class startIndex tracker and iterates the
                // delta-class set until no new encodedStatement is
                // added in a pass — covering same-NS, ancestor, and
                // descendant directions via a single comparability
                // check (one comparability test for all three
                // directional cases).
                //
                // cleanUpExpressions on isEquality(expr) runs in the
                // post-applyEquiClasses deferred cleanup in
                // `standardProcessing`, not here. Same reason as the
                // partner block inside updateEquivalenceClasses above —
                // inline cleanup would run before the rewriter could read
                // non-canonical rows.
            }
        }

/// @brief Record an int scope's finish when a statement registered there is
        ///        an instance of its integration goal, and promote the group's
        ///        first finished scopes once every premise has one.
        ///
        /// @details
        /// An int scope (`LbMemory::intScopes`) finishes when a statement
        /// registered at the scope itself is an instance of its goal group's
        /// goal (`ScopeSyntax::intScopeGoalMatches`, the goal in its canonical
        /// form at the scope; for a scope of the second twin premise the goal
        /// with `rec2`, the name of the twin partner's predecessor, for `rec`,
        /// D-589). Only the first finished scope of each premise of
        /// the group counts (`LbMemory::intScopeFirstFinish`); later finishes
        /// of that premise change nothing. When every premise level of the
        /// group (`LbMemory::intScopeGroups`) has its first finished scope, the
        /// group is marked promoted and those scopes are promoted together in
        /// decoded-name order (@ref promoteIntScope); then every int scope of
        /// the group, promoted or not, is queued on `pendingWipeScopes`, and
        /// the closed group opens no further scope
        /// (D-578, D-582).
        ///
        /// @param memoryBlock     Owning logic block (single-threaded seam).
        /// @param originalId      The registered statement's id.
        /// @param scopeId         Its scope.
        /// @param newStatements   The registration's output rows; the promoted
        ///                        rows are appended.
        /// @param publicationMail The registration's output channel.
        /// @return Void.
        /// @invariant Mints into the NameMap only through the promoted
        ///            registrations.
        /// @see promoteIntScope, AdmissionMemory::openIntScope.
inline void StatementMemory::finishIntScope(Memory& memoryBlock, NameId originalId, NameId scopeId,
            PagedVector<IntEncodedExpr>& newStatements, ColdMail* publicationMail) {
            const int64_t* record = memoryBlock.lbMemory.intScopes.find(static_cast<int64_t>(scopeId));
            if (record == nullptr) return;
            const NameId groupId = static_cast<NameId>(static_cast<int32_t>(static_cast<uint64_t>(*record) >> 32));
            const int32_t premiseLevel = static_cast<int32_t>(static_cast<uint32_t>(*record));
            const int64_t finishKey = packInt32Pair(static_cast<int32_t>(groupId), premiseLevel);
            if (memoryBlock.lbMemory.intScopeFirstFinish.find(finishKey) != nullptr) return;
            {
                const unsigned slot = gl::WorkerContext::coreId() >= 0
                    ? static_cast<unsigned>(gl::WorkerContext::coreId()) : scratchArenas().slotCount() - 1;
                ScratchArena& strings = scratchArenas().forSlot(slot);
                ScratchScope frame(strings);
                // Nothing below mints the NameMap, so the views stay valid
                // (I-3).
                int32_t noLevels = 0;
                // A scope of the second twin premise finishes on the group's
                // goal with `rec2` for `rec` (D-589).
                StrSpan groupGoal = memoryBlock.nameMap.decodeView(groupId);
                ScratchString secondGoal;
                if (premiseLevel == memoryBlock.twinSecondLevel) {
                    assert(memoryBlock.twinPredecessorId != 0
                        && "finishIntScope: a second-twin scope exists once its predecessor is recorded");
                    const StrReplacement recToPredecessor{ StrSpan("rec", 3),
                        memoryBlock.nameMap.decodeView(memoryBlock.twinPredecessorId) };
                    secondGoal = replaceKeysScratch(strings, groupGoal, &recToPredecessor, 1);
                    groupGoal = StrSpan(secondGoal);
                }
                const CanonicalForm goal = canonicalizer.canonicalFormAtScope(
                    groupGoal, memoryBlock.nameMap.decodeView(scopeId),
                    memoryBlock, strings, nullptr, noLevels, 0, false, false, false);
                if (!gl::ScopeSyntax::intScopeGoalMatches(goal.text, memoryBlock.nameMap.decodeView(originalId)))
                    return;
            }
            memoryBlock.lbMemory.intScopeFirstFinish.insert(finishKey, static_cast<int64_t>(scopeId));
            const int64_t* groupRecord = memoryBlock.lbMemory.intScopeGroups.find(static_cast<int64_t>(groupId));
            assert(groupRecord != nullptr && (*groupRecord & kIntScopeGroupPromoted) == 0
                && "finishIntScope: an open group records its premise levels");
            const int64_t groupLevels = *groupRecord;
            NameId finished[64];
            int32_t finishedN = 0;
            for (int32_t level = 0; level < 62; ++level) {
                if ((groupLevels & (int64_t{1} << level)) == 0) continue;
                const int64_t* first = memoryBlock.lbMemory.intScopeFirstFinish.find(
                    packInt32Pair(static_cast<int32_t>(groupId), level));
                if (first == nullptr) return;
                finished[finishedN++] = static_cast<NameId>(*first);
            }
            assert(finishedN >= 2 && "finishIntScope: a group has two premises or more");
            memoryBlock.lbMemory.intScopeGroups.upsert(static_cast<int64_t>(groupId),
                groupLevels | kIntScopeGroupPromoted);
            // Promote in decoded-name order: id order is mint order.
            for (int32_t i = 1; i < finishedN; ++i)
                for (int32_t j = i; j > 0 && compareSpans(memoryBlock.nameMap.decodeView(finished[j]),
                        memoryBlock.nameMap.decodeView(finished[j - 1])) < 0; --j) {
                    const NameId swap = finished[j];
                    finished[j] = finished[j - 1];
                    finished[j - 1] = swap;
                }
            for (int32_t i = 0; i < finishedN; ++i)
                promoteIntScope(memoryBlock, finished[i], newStatements, publicationMail);
            // The promotion closes the group: every int scope of the group, the
            // promoted ones and those that never finished first, is queued for
            // the wipe, since nothing in the others can reach the parent any
            // more (D-582). The drain orders by name.
            const int32_t scopeCount = memoryBlock.lbMemory.intScopes.count();
            for (int32_t id = 1; id <= scopeCount; ++id) {
                const int64_t scopeRecord = memoryBlock.lbMemory.intScopes.valueAt(id);
                const NameId scopeGroup = static_cast<NameId>(static_cast<int32_t>(static_cast<uint64_t>(scopeRecord) >> 32));
                if (scopeGroup == groupId)
                    memoryBlock.pendingWipeScopes.mint(static_cast<NameId>(memoryBlock.lbMemory.intScopes.decodeKey(id)));
            }
        }

/// @brief Register every statement of a finished int scope at the scope's
        ///        parent.
        ///
        /// @details
        /// Every statement registered at @p scopeId itself (nested scopes
        /// excluded) and not known at its parent is registered at the parent
        /// through @ref addStatement, in registration order, with its own
        /// levels and locality and the raw-only `int scope` origin citing the
        /// statement at the scope, as the induction goal exit does. The rows
        /// are snapshot first: the registrations append to the registry
        /// (D-578).
        ///
        /// @param memoryBlock     Owning logic block (single-threaded seam).
        /// @param scopeId         An open int scope (asserted).
        /// @param newStatements   The registration's output rows.
        /// @param publicationMail The registration's output channel.
        /// @return Void.
        /// @invariant Heap-free: the row snapshot rides the gen-scratch page
        ///            tier, the texts the string-scratch tier.
        /// @see finishIntScope, OriginTag::intScope.
inline void StatementMemory::promoteIntScope(Memory& memoryBlock, NameId scopeId,
            PagedVector<IntEncodedExpr>& newStatements, ColdMail* publicationMail) {
            assert(gl::ScopeSyntax::isIntScope(memoryBlock.nameMap, scopeId)
                && memoryBlock.lbMemory.intValidityNamesToFilter.lookup(scopeId) == 0
                && "promoteIntScope: an open int scope");
            const NameId parentId = memoryBlock.nameMap.parentOf(scopeId);
            assert(parentId != 0 && "promoteIntScope: an int scope has a parent");
            const unsigned genSlot = gl::WorkerContext::coreId() >= 0
                ? static_cast<unsigned>(gl::WorkerContext::coreId()) : genScratchArenas().slotCount() - 1;
            const unsigned strSlot = gl::WorkerContext::coreId() >= 0
                ? static_cast<unsigned>(gl::WorkerContext::coreId()) : scratchArenas().slotCount() - 1;
            ScratchArena& rowsArena = genScratchArenas().forSlot(genSlot);
            ScratchArena& strings = scratchArenas().forSlot(strSlot);
            ScratchScope rowsFrame(rowsArena);
            ScratchScope stringsFrame(strings);
            DirtyState dirty = DirtyState::Clean;
            PagedVector<NameId> rows(&rowsArena, &dirty);
            for (int32_t r = 0; r < memoryBlock.intEncodedStatements.size(); ++r) {
                const IntEncodedExpr& row = memoryBlock.intEncodedStatements[r];
                if (row.validityId != scopeId) continue;
                if (memoryBlock.intKnownStatements.find(StatementKey{row.originalId, scopeId}) == nullptr) continue;
                rows.push_back(row.originalId);
            }
            // Owned copies of both names: the registrations mint the NameMap
            // (I-3).
            const StrSpan parentView = memoryBlock.nameMap.decodeView(parentId);
            const ScratchString parentName = ScratchString::copyFrom(strings, parentView.ptr, parentView.len);
            const StrSpan scopeView = memoryBlock.nameMap.decodeView(scopeId);
            const ScratchString scopeName = ScratchString::copyFrom(strings, scopeView.ptr, scopeView.len);
            for (int32_t i = 0; i < rows.size(); ++i) {
                const NameId originalId = rows[i];
                if (memoryBlock.intKnownStatements.find(StatementKey{originalId, parentId}) != nullptr) continue;
                ScratchScope rowFrame(strings);
                const StrSpan textView = memoryBlock.nameMap.decodeView(originalId);
                const ScratchString text = ScratchString::copyFrom(strings, textView.ptr, textView.len);
                const StatementFlags* flags = memoryBlock.intKnownStatements.find(StatementKey{originalId, scopeId});
                assert(flags != nullptr && "promoteIntScope: a snapshot row stays known");
                const bool local = flags->local;
                const int32_t levelsId = memoryBlock.intStatementLevelsMap.lookup(packStatementKey(originalId, scopeId));
                assert(levelsId != 0 && "promoteIntScope: a known statement has its levels row");
                int levels[256];
                const int32_t levelCount = coldIntRunAt(memoryBlock.intStatementLevelsMap, levelsId, levels, 256);
                const OriginDep leafDeps[1] = { { StrSpan(text), StrSpan(scopeName) } };
                const TransientOrigin promotedOrigin{ true, OriginTag::intScope, leafDeps, 1 };
                addStatement(StrSpan(text), memoryBlock, local, levels, levelCount, promotedOrigin,
                    StrSpan(parentName), newStatements, false, publicationMail);
            }
        }

}
