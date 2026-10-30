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
#include "../logical_block/statement_arrivals.hpp"
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
class BlockMailProcessor;
class CaseProcessor;
class EqualityPropagation;
class EquivalenceProcessor;
class GoalProcessor;
class ImplicationSubproofs;
class InductionUsefulness;
class OrImplicationProcessor;
class StatementMemory;
class UnaryIntegrationProcessor;
class WatchProcessor;
class ContradictionDemand;
class RejectedSourceReplay;
namespace tests { struct LogicBlockOperationsFixture; }
/// @brief Logic Block Operations operations for the proof algorithm.
/// @details Uses explicit peer interfaces while the supplied logical blocks retain storage and proof state.
class LogicBlockOperations {
friend struct tests::LogicBlockOperationsFixture;
public:
/// @brief Bind inputs for logic block operations.
/// @details Construction stores explicit references; block storage and all invocation lifetimes remain with their owners.
/// @param blockIdentity Borrowed LogicBlockIdentity interface or state; must outlive this component.
/// @param ceFilteringActive Borrowed bool interface or state; must outlive this component.
/// @param contradictionTable Borrowed std::vector interface or state; must outlive this component.
/// @param parameters Borrowed ProverParameters interface or state; must outlive this component.
/// @param rejectedSources Borrowed RejectedSourceMemory interface or state; must outlive this component.
/// @param blockMailProcessor Borrowed BlockMailProcessor interface or state; must outlive this component.
/// @param caseProcessor Borrowed CaseProcessor interface or state; must outlive this component.
/// @param equalityPropagation Borrowed EqualityPropagation interface or state; must outlive this component.
/// @param equivalenceProcessor Borrowed EquivalenceProcessor interface or state; must outlive this component.
/// @param goalProcessor Borrowed GoalProcessor interface or state; must outlive this component.
/// @param implicationSubproofs Borrowed ImplicationSubproofs interface or state; must outlive this component.
/// @param inductionUsefulness Borrowed InductionUsefulness interface or state; must outlive this component.
/// @param orImplicationProcessor Borrowed OrImplicationProcessor interface or state; must outlive this component.
/// @param statementMemory Borrowed StatementMemory interface or state; must outlive this component.
/// @param unaryIntegrationProcessor Borrowed UnaryIntegrationProcessor interface or state; must outlive this component.
/// @param watchProcessor Borrowed WatchProcessor interface or state; must outlive this component.
/// @param contradictionDemand Borrowed ContradictionDemand (the contradiction-scope demand drain of step 4b); must outlive this component.
/// @param rejectedSourceReplay Borrowed RejectedSourceReplay (the demand-built OR event drain of step 4b); must outlive this component.
/// @param phase13TimingRows Borrowed per-worker timing storage, available when detailed phase timing is enabled.
/// @param phase13TimingWorkers Borrowed unsigned interface or state; must outlive this component.
/// @return None; construction binds the supplied references.
LogicBlockOperations(LogicBlockIdentity& blockIdentity, const bool& ceFilteringActive, std::vector<ContradictionItem>& contradictionTable, const ProverParameters& parameters, RejectedSourceMemory& rejectedSources, BlockMailProcessor& blockMailProcessor, CaseProcessor& caseProcessor, EqualityPropagation& equalityPropagation, EquivalenceProcessor& equivalenceProcessor, GoalProcessor& goalProcessor, ImplicationSubproofs& implicationSubproofs, InductionUsefulness& inductionUsefulness, OrImplicationProcessor& orImplicationProcessor, StatementMemory& statementMemory, UnaryIntegrationProcessor& unaryIntegrationProcessor, WatchProcessor& watchProcessor, ContradictionDemand& contradictionDemand, RejectedSourceReplay& rejectedSourceReplay
#if PHASE13_DEEP_TIMING
, int64_t*& phase13TimingRows
#endif

#if PHASE13_DEEP_TIMING
, unsigned& phase13TimingWorkers
#endif
);

private:
/// @brief Parse a hypo validity name into its parent scope and its
        ///        `_var<i>_` payload variables — the byte-exact extraction
        ///        twin of `reactToHypo`'s retired inline `std::string` block.
        ///
        /// @details
        /// The canonical grammar (producer
        /// `logical_block/logic_block_operations.hpp::disintegrateExprHypothetically`) is
        /// `parent + "_boundary_" + "_var0_<x>[_var1_<y>...]" + "_hypo_" + expr`.
        /// The four gates run in the historical order, each a bare
        /// reject:
        /// 1. the LAST `"_hypo_"` occurrence bounds the payload (`rfind`
        ///    no-pos form); absent -> reject;
        /// 2. any `"__or__"` occurrence rejects (OR-tagged scopes never
        ///    react — case-analysis interference);
        /// 3. the last `"_boundary__var0_"` at-or-before the hypo marker
        ///    anchors the payload (`rfind` bounded at `posHypo`); absent ->
        ///    reject; the parent slice is `[0, posBoundary)` and the marker
        ///    walk starts at `posBoundary + 10` (skipping `"_boundary_"`);
        /// 4. an empty extracted-variable set rejects (folded into the
        ///    return value — the historical four `continue`s had no side
        ///    effects between them, so one boolean reproduces them exactly).
        ///
        /// The marker walk is the verbatim twin of the historical loop,
        /// including its two subtleties: the `_var<i+1>_` search runs over
        /// the WHOLE name from the content start (the historical `find` was
        /// NOT bounded by the hypo position — the `nextPos >= suffixPos`
        /// check comes after and turns a post-suffix hit into "last
        /// variable"), and the marker-existence check is the same two-step
        /// gate (bounds first, then byte equality). Markers are built in a
        /// stack `char[24]` via `writeDecimalDigits` — no heap, no
        /// `std::to_string`.
        ///
        /// The producer bound (`MAX_ARITY` variables) is asserted as a
        /// Rule-19 tripwire on both the marker index and the dedup insert; a
        /// firing assert means a NEW producer of hypo names appeared and
        /// must be reported, never widened silently. The producer always
        /// appends the `"_hypo_"` marker AFTER the payload, so the last
        /// variable's end never precedes its start. On the
        /// producer-unreachable overlap shape where the LAST `"_hypo_"`
        /// begins INSIDE the matched `"_boundary__var0_"` fragment (payload
        /// content opening with `"hypo_"` and no later `"_hypo_"`), the
        /// end-before-start slice fires the `StrSpan` length assert BY
        /// DESIGN — a deliberate tripwire replacing the retired `substr`'s
        /// silent `size_t` wrap-around, which returned a garbage whole-tail
        /// "variable" (Rule 19: a loud assert over silent nonsense; the
        /// retired garbage tail matched no class member anyway).
        ///
        /// @param validityName The validity name to parse; the out-spans
        ///                     alias its bytes (caller keeps them stable).
        /// @param out          Receives the parse; written only on the path
        ///                     that reaches each field (parent + vars are
        ///                     valid only when the return is `true`).
        /// @return `true` when all four gates pass (at least one variable
        ///         extracted); `false` otherwise.
        /// @invariant Byte-exact gate verdicts, parent slice, and variable
        ///            SET versus the retired `rfind`/`find`/`substr`/
        ///            `std::set<std::string>` block (unit-test-pinned
        ///            against that block as the oracle).
        /// @see HypoScopeParse, reactToHypo, findSpanFrom, rfindSpanBefore,
        ///      writeDecimalDigits.
inline bool parseHypoScopeVars(const StrSpan& validityName,
                                       HypoScopeParse& out);

public:
/// @brief Per-call mail-absorb pipeline: drain one external and
        /// one internal `Mail` container, apply equivalence classes,
        /// discharge `toBeProved` with the caller-supplied output
        /// internal-mail target, populate `mailOut`. Called twice per
        /// elementary step (pre-burst in `performElemPhase1`, post-burst in
        /// `performElemPhase3`):
        ///   - **pre-burst** with
        ///     `externalMailIn = &body.mailIn`,
        ///     `internalMailIn = body.nextIterationInternalMail`,
        ///     `internalMailOut = body.sameIterationInternalMail` —
        ///     any discharge emission during this absorb writes to
        ///     `sameIter` and is processed by the **same** step's
        ///     hashburst-following call 2 (zero-delay).
        ///   - **post-burst** with
        ///     `externalMailIn = nullptr` (no external arrivals after
        ///     the burst), `internalMailIn = body.sameIterationInternalMail`,
        ///     `internalMailOut = body.nextIterationInternalMail` —
        ///     any discharge emission during this absorb writes to
        ///     `nextIter` and is processed by the **next** step's
        ///     call 1 (one-step delay).
        ///
        /// @details
        /// In counterexample mode, checks existing registered contradictions and
        /// completed internal-mail heads against the loaded main-scope facts,
        /// then returns. Mail, equivalence classes, goals and all other pipeline
        /// stages are untouched in that mode; each clone ends after one burst.
        ///
        /// In ordinary mode, consolidates the absorb pipeline into a
        /// single reusable entry point. The two-mail input shape lets
        /// the caller choose between same-iteration (zero-delay) and
        /// cross-iteration (one-step-delay) routing for discharge
        /// emissions, without any new status code or parameter on
        /// `addExprToMemoryBlock`.
        ///
        /// Steps in order:
        ///
        /// 1. **Absorb external mail** (only if `externalMailIn` is
        ///    non-null): bulk merge `externalMailIn->exprOriginMap` into
        ///    `body.exprOriginMap` (capped, existing rows win); sync
        ///    equality origins into each containing class's
        ///    `equalityOriginMap` (trackHistory only); drain
        ///    `externalMailIn->statements` via flat `addExprToMemoryBlock`
        ///    with `status = 3` (each entry additionally staged into
        ///    `externalStatements` and `intExternalStatements` so the
        ///    burst's request generation sees the freshly-arrived rules
        ///    and facts as workingMemory-eligible — `vName == "main"`
        ///    asserted per element, [I-26](../../docs/30_invariants.md#i-26));
        ///    merge `externalMailIn->expandedImplications` into
        ///    `body.expandedImplications`.
        /// 2. **Absorb internal mail**: same four-step recipe but with
        ///    `status = 1` (no `intExternalStatements` staging — the
        ///    internal channel is integration-revival traffic, not a
        ///    workingMemory feed).
        /// 3. **Clear the internal channel** (`internalMailIn`). Clear
        ///    placement is **after drain, before** apply/discharge/fillMailOut
        ///    so any subsequent write to `internalMailOut` survives — this is
        ///    the load-bearing detail that makes the same-iter zero-delay
        ///    routing work for the pre-burst call (the same channel is both
        ///    `internalMailIn` of one direction and `internalMailOut` of the
        ///    other across the two calls; clearing in the middle is the only
        ///    timing that keeps both correct). The external routing inbox
        ///    (`body.mailIn`) is cleared by the caller after the call returns.
        /// 4. `reactToHypo(body)`, then `applyEquiClasses(body)` — hypothesis
        ///    copy equalities enter the same call's class processing and history
        ///    publication. One explicit equivalence-class
        ///    application pass over the freshly-populated
        ///    `intLocalEncodedStatementsDelta`.
        /// 5. `dischargeToBeProved(body, coreId, internalMailOut)` —
        ///    `toBeProved` closure pass. Discharge's two parent-scope
        ///    emission sites (OR-integration + NotOrScope) write to
        ///    `internalMailOut.statements` + `internalMailOut.exprOriginMap`.
        /// 6. `fillMailOut(body)` — centralised cross-LB `mailOut`
        ///    population from the same delta with `main` +
        ///    `allowedForMail` gates.
        /// 7. Clear `body.changedClassesThisStep` — the SOLE clear of the
        ///    delta tracker anywhere in the pipeline. Every entry present
        ///    here was consumed by this call's `applyEquiClasses` / deferred
        ///    cleanup sweep / recursion-products closure, except a class
        ///    minted by `dischargeContradiction`'s head deposit — whose LB
        ///    is deactivated in the same act and so has no future consumer.
        ///    Deltas minted OUTSIDE any `standardProcessing` call (LB
        ///    seeding, post-join barrier deposits) are cleared nowhere else
        ///    and survive to the next call's `applyEquiClasses` Pass 1.
        ///
        /// 8. Sanitize goals, unbind the phase-3 exclusion, drain disproved
        ///    goals and dead branches, retire duplicate cohorts, freeze resolved
        ///    branches, release pending cases, wipe the sorted union of closed
        ///    scopes once, then sweep ancestor-known rows. All inference loops
        ///    have finished before destructive cleanup. History and surviving
        ///    convergence evidence remain available.
        ///
        /// @param memoryBlock Owning LB. Read and mutated across most
        ///        steps: `exprOriginMap`, `equivalenceClassesMap`,
        ///        `externalStatements` and `intExternalStatements`
        ///        (only when `externalMailIn != nullptr`),
        ///        `expandedImplications`, `intStatementLevelsMap`,
        ///        `intEncodedStatements` (via `addExprToMemoryBlock` and
        ///        downstream calls), `mailOut`,
        ///        `changedClassesThisStep`.
        /// @param externalMailIn External-mail routing inbox (status=3), the
        ///        `RoutingColdMail` `&memoryBlock.mailIn`. May be `nullptr` (post-burst
        ///        call). Absorbed directly from its sorted snapshots (no transient
        ///        heap `Mail`); cleared by the CALLER after this call returns
        ///        (nothing between the drain and return reads it).
        /// @param internalMailIn Internal-mail input (status=1).
        ///        Pre-burst: `body.nextIterationInternalMail`. Post-burst:
        ///        `body.sameIterationInternalMail`. Cleared after absorption;
        ///        subsequent default-channel emissions survive the call.
        /// @param internalMailOut Output target for the call's
        ///        in-flight internal-mail writes (parent-scope
        ///        discharge emissions). Pre-burst:
        ///        `body.sameIterationInternalMail`. Post-burst:
        ///        `body.nextIterationInternalMail`. Not cleared by this
        ///        function — discharge writes here must survive past
        ///        this call to be processed by the appropriate future
        ///        absorb.
        /// @param coreId Core identifier forwarded to
        ///        `addExprToMemoryBlock` and `dischargeToBeProved`
        ///        for thread-local routing of global emissions.
        /// @see `applyEquiClasses`, `dischargeToBeProved`,
        ///      `fillMailOut` — the three pipeline stages this driver
        ///      orchestrates.
        /// @see [`docs/10_pipeline/04_prover.md`](../../docs/10_pipeline/04_prover.md).
        /// @see [`docs/20_core_concepts/03_mail_system.md`](../../docs/20_core_concepts/03_mail_system.md).
inline void standardProcessing(Memory& memoryBlock,
                                       RoutingColdMail* externalMailIn,
                                       ColdMail& internalMailIn,
                                       ColdMail& internalMailOut,
                                       int coreId);

private:
/// @brief Close `productsOfRecursionIds` under main-scope
        ///        equivalence-class membership for every class changed this
        ///        step.
        ///
        /// @details
        /// A product of recursion is a name built by eating through the
        /// induction implication's premise cursors (the two mint sites:
        /// `updateAdmissionMapRecursion`'s output substitution and
        /// `isAdmitted`'s cursor-witness admission). The marker-capture purity
        /// gate in `checkLocalEncodedMemoryStatic` stages admission demand only
        /// on requests whose iteration-bearing premise arguments are all
        /// products of recursion. A name PROVEN EQUAL to a product of
        /// recursion denotes the same value, so demand built on it is anchored
        /// by the induction progression exactly as demand built on the product
        /// itself. Without this closure a mailed foreign witness-form (e.g. a
        /// flag-5 relay product minted at an ancestor LB) can become the sole
        /// surviving instance of induction-built content and starve the purity
        /// gate of every pure instance — the Gauss fold loss
        /// (D-285).
        ///
        /// Walks `changedClassesThisStep` (the per-step delta tracker) at the
        /// single-threaded `standardProcessing` seam, immediately after the
        /// `applyEquiClasses` fixpoint: for each MAIN-scope class holding at
        /// least one member already in `productsOfRecursionIds`, every other
        /// member is minted into the set. Non-main scopes are excluded — a
        /// conditional (hypothesis / boundary) equality must not enrich the
        /// LB-global set. A class whose member joins `productsOfRecursionIds`
        /// only AFTER the class's last change is not re-visited here; both
        /// mint sites run before the first classes form in practice
        /// (I-194 documents the limitation).
        ///
        /// @param mb The LB whose changed classes are walked and whose
        ///        `overallHashMemory.productsOfRecursionIds` is enriched.
        /// @return (void)
        /// @invariant Walk order is the buffer's append order and every mint is
        ///            guarded by a `contains` probe — the cold set's insertion
        ///            order is a pure function of the step's class state
        ///            (deterministic deload bytes).
        /// @see `checkLocalEncodedMemoryStatic` — the purity-gate consumer.
void enrichProductsOfRecursionFromChangedClasses(Memory& mb);

private:
// Turn optimizations OFF for just this section
//#pragma optimize("", off)

// The hash-engine bodies live in memory.cpp: addToHashMemory(),
// makeNormalizedKeysForAdmission(), lessByName(), lessByOriginal(),
// filterIntEncodedStatements(), generateEncodedRequestsStatic(), and
// checkLocalEncodedMemoryStatic().

// ========================================================================
// Static pipeline: IntEncodedExpr-based, zero-alloc request generation
// ========================================================================

//#pragma optimize("", off)
/// @brief Install the dead-end variable-copy equalities for every
/// hypo-scoped validity before standard class application.
///
/// @details
/// `disintegrateExprHypothetically` opens validity scopes named
/// `parent + "_boundary_" + "_var0_<x>[_var1_<y>...]" + "_hypo_" + expr`,
/// one per hypothetically disintegrated integration goal. Both ordinary
/// standard-processing calls react after input clearing and before class
/// application, with one worker owning the LB. This function walks every equivalence-class scope of the LB and, on each
/// hypo-scoped row, checks every class member that is one of the payload
/// variables: when ALL of its class-mates are forced or ordinary integration witness names and at
/// least one class-mate exists, the dead-end equality `(=[var,var_copy])`
/// is installed at the PARENT scope via `addExprToMemoryBlock` with the
/// `variableCopy` origin (the "variable copy" proof tag).
///
/// The interior is statified
/// (I-142; 09b/09c cookbooks):
/// - Row order: a validity-id array on the per-slot gen-scratch byte-bump
///   tier, sorted by `compareSpans(decodeView(a), decodeView(b))` — the
///   decoded-name lex order of the retired string-pair snapshot, never id
///   order (I-84). Tie-free (distinct vids decode to distinct names) and
///   mint-free for the whole sort window (I-3).
/// - Row names: the cheap `_hypo_` gate runs on the raw `decodeView`
///   span; a passing row is copied to a per-row `ScratchString` on the
///   string-scratch tier so the `parseHypoScopeVars` spans survive the
///   class loop's NameMap mints.
/// - Per-row class read: an arena MEMBER-RUN SNAPSHOT (`memberPool` /
///   `classStarts`, gen-scratch page tier, cleared per row) taken BEFORE
///   the class loop. The loop's own `addExprToMemoryBlock` can splice
///   this same map's shared blob pool through a DIFFERENT key (the
///   parent-scope equality), so live blob views would both see mid-row
///   mutations the retired heap decode never saw and dangle; the snapshot
///   reproduces the heap `decodeClassesById` isolation exactly, and is
///   content-complete because the loop reads only member ids. Bucket
///   content is read at row start — the decode-at-the-point-of-processing
///   semantics: an earlier row's mutations to a later row's bucket ARE
///   seen, exactly as before.
/// - Firing product: exact-length `ScratchString` build of
///   `(=[var,var_copy])` plus the byte-exact `removeUPrefixScratch` twin;
///   the add door receives byte-identical spans, levels, origin, and
///   flags in the identical call order.
///
/// The door levels ride a hoisted one-element stack run (`{ mb.level }`):
/// `addExprToMemoryBlock` takes an ascending-unique `(const int*, int32_t)`
/// run, caller-owned per the levels-run chain contract. `mb.level` is
/// written only at LB creation, so one run per call equals the retired
/// one set per firing.
///
/// @param mb The LB whose equivalence-class scopes are surveyed; the add
///           door mints into `mb.nameMap` under the owning worker claim.
/// @see parseHypoScopeVars, HypoScopeParse — the name parse;
///      `disintegrateExprHypothetically` — the hypo-scope producer;
///      `EqClassNameCaches::kindOf` — the class-mate tier probe.
void reactToHypo(Memory& mb);
private:

LogicBlockIdentity& blockIdentity;
const bool& ceFilteringActive;
std::vector<ContradictionItem>& contradictionTable;
const ProverParameters& parameters;
RejectedSourceMemory& rejectedSources;
BlockMailProcessor& blockMailProcessor;
CaseProcessor& caseProcessor;
EqualityPropagation& equalityPropagation;
EquivalenceProcessor& equivalenceProcessor;
GoalProcessor& goalProcessor;
ImplicationSubproofs& implicationSubproofs;
InductionUsefulness& inductionUsefulness;
OrImplicationProcessor& orImplicationProcessor;
StatementMemory& statementMemory;
UnaryIntegrationProcessor& unaryIntegrationProcessor;
WatchProcessor& watchProcessor;
ContradictionDemand& contradictionDemand;
RejectedSourceReplay& rejectedSourceReplay;
#if PHASE13_DEEP_TIMING
int64_t*& phase13TimingRows;
#endif
#if PHASE13_DEEP_TIMING
unsigned& phase13TimingWorkers;
#endif
};
}
#include "../logical_block/block_mail_processor.hpp"
#include "../logical_block/case_processor.hpp"
#include "../logical_block/equality_propagation.hpp"
#include "../logical_block/equivalence_processor.hpp"
#include "../logical_block/goal_processor.hpp"
#include "../logical_block/implication_subproofs.hpp"
#include "../logical_block/contradiction_demand.hpp"
#include "../logical_block/rejected_source_replay.hpp"
#include "../proof/induction_usefulness.hpp"
#include "../logical_block/or_implication_processor.hpp"
#include "../logical_block/statement_memory.hpp"
#include "../logical_block/unary_integration_processor.hpp"
#include "../logical_block/watch_processor.hpp"
namespace gl {

/// @brief Parse a hypo validity name into its parent scope and its
        ///        `_var<i>_` payload variables — the byte-exact extraction
        ///        twin of `reactToHypo`'s retired inline `std::string` block.
        ///
        /// @details
        /// The canonical grammar (producer
        /// `logical_block/logic_block_operations.hpp::disintegrateExprHypothetically`) is
        /// `parent + "_boundary_" + "_var0_<x>[_var1_<y>...]" + "_hypo_" + expr`.
        /// The four gates run in the historical order, each a bare
        /// reject:
        /// 1. the LAST `"_hypo_"` occurrence bounds the payload (`rfind`
        ///    no-pos form); absent -> reject;
        /// 2. any `"__or__"` occurrence rejects (OR-tagged scopes never
        ///    react — case-analysis interference);
        /// 3. the last `"_boundary__var0_"` at-or-before the hypo marker
        ///    anchors the payload (`rfind` bounded at `posHypo`); absent ->
        ///    reject; the parent slice is `[0, posBoundary)` and the marker
        ///    walk starts at `posBoundary + 10` (skipping `"_boundary_"`);
        /// 4. an empty extracted-variable set rejects (folded into the
        ///    return value — the historical four `continue`s had no side
        ///    effects between them, so one boolean reproduces them exactly).
        ///
        /// The marker walk is the verbatim twin of the historical loop,
        /// including its two subtleties: the `_var<i+1>_` search runs over
        /// the WHOLE name from the content start (the historical `find` was
        /// NOT bounded by the hypo position — the `nextPos >= suffixPos`
        /// check comes after and turns a post-suffix hit into "last
        /// variable"), and the marker-existence check is the same two-step
        /// gate (bounds first, then byte equality). Markers are built in a
        /// stack `char[24]` via `writeDecimalDigits` — no heap, no
        /// `std::to_string`.
        ///
        /// The producer bound (`MAX_ARITY` variables) is asserted as a
        /// Rule-19 tripwire on both the marker index and the dedup insert; a
        /// firing assert means a NEW producer of hypo names appeared and
        /// must be reported, never widened silently. The producer always
        /// appends the `"_hypo_"` marker AFTER the payload, so the last
        /// variable's end never precedes its start. On the
        /// producer-unreachable overlap shape where the LAST `"_hypo_"`
        /// begins INSIDE the matched `"_boundary__var0_"` fragment (payload
        /// content opening with `"hypo_"` and no later `"_hypo_"`), the
        /// end-before-start slice fires the `StrSpan` length assert BY
        /// DESIGN — a deliberate tripwire replacing the retired `substr`'s
        /// silent `size_t` wrap-around, which returned a garbage whole-tail
        /// "variable" (Rule 19: a loud assert over silent nonsense; the
        /// retired garbage tail matched no class member anyway).
        ///
        /// @param validityName The validity name to parse; the out-spans
        ///                     alias its bytes (caller keeps them stable).
        /// @param out          Receives the parse; written only on the path
        ///                     that reaches each field (parent + vars are
        ///                     valid only when the return is `true`).
        /// @return `true` when all four gates pass (at least one variable
        ///         extracted); `false` otherwise.
        /// @invariant Byte-exact gate verdicts, parent slice, and variable
        ///            SET versus the retired `rfind`/`find`/`substr`/
        ///            `std::set<std::string>` block (unit-test-pinned
        ///            against that block as the oracle).
        /// @see HypoScopeParse, reactToHypo, findSpanFrom, rfindSpanBefore,
        ///      writeDecimalDigits.
inline bool LogicBlockOperations::parseHypoScopeVars(const StrSpan& validityName,
                                       HypoScopeParse& out) {
            const StrSpan kHypo("_hypo_", 6);
            const StrSpan kOr("__or__", 6);
            const StrSpan kBoundaryVar0("_boundary__var0_", 16);

            // Gate 1: the LAST "_hypo_" bounds the variable payload.
            const int32_t posHypo =
                rfindSpanBefore(validityName, kHypo, validityName.len);
            if (posHypo == -1) return false;

            // Gate 2: OR-tagged validity names never react (case-analysis
            // interference).
            if (findSpanFrom(validityName, kOr, 0) != -1) return false;

            // Gate 3: the last "_boundary__var0_" at-or-before the hypo
            // marker separates the parent from the payload.
            const int32_t posBoundary =
                rfindSpanBefore(validityName, kBoundaryVar0, posHypo);
            if (posBoundary == -1) return false;

            out.parentValidity = StrSpan(validityName.ptr, posBoundary);

            // posVar points at the "_var0_" sub-token — skip the
            // "_boundary_" (10 bytes).
            const int32_t posVar = posBoundary + 10;

            int32_t idx = 0;
            int32_t currentPos = posVar;
            const int32_t suffixPos = posHypo;  // variables end at "_hypo_"

            while (true) {
                assert(idx < ExecutionParameters::MAX_ARITY
                    && "parseHypoScopeVars: marker index exceeds the producer "
                       "arity bound — a new hypo-name producer appeared");

                // currentMarker = "_var" + digits(idx) + "_" in a stack
                // buffer ("_var" 4 + <=10 digits + "_" 1 <= 15 bytes).
                char curBuf[24];
                std::memcpy(curBuf, "_var", 4);
                int32_t curLen = 4 + writeDecimalDigits(curBuf + 4, idx);
                curBuf[curLen++] = '_';
                const StrSpan currentMarker(curBuf, curLen);

                // Marker-existence check — the same two-step gate as the
                // historical compare (bounds first, then bytes).
                if (currentPos + currentMarker.len > validityName.len
                    || !equalSpans(StrSpan(validityName.ptr + currentPos,
                                           currentMarker.len),
                                   currentMarker)) {
                    break;  // sequence broken or malformed
                }

                const int32_t startContent = currentPos + currentMarker.len;

                char nextBuf[24];
                std::memcpy(nextBuf, "_var", 4);
                int32_t nextLen = 4 + writeDecimalDigits(nextBuf + 4, idx + 1);
                nextBuf[nextLen++] = '_';
                const StrSpan nextMarker(nextBuf, nextLen);

                // Whole-span search — deliberately NOT bounded by suffixPos;
                // the >= suffixPos test below reproduces the historical
                // semantics exactly.
                const int32_t nextPos =
                    findSpanFrom(validityName, nextMarker, startContent);

                int32_t endContent;
                bool isLast = false;
                if (nextPos == -1 || nextPos >= suffixPos) {
                    endContent = suffixPos;
                    isLast = true;
                } else {
                    endContent = nextPos;
                }

                const StrSpan extractedVar(validityName.ptr + startContent,
                                           endContent - startContent);

                // Dedup insert — content equality, insertion order kept.
                bool duplicate = false;
                for (int32_t i = 0; i < out.varCount; ++i) {
                    if (equalSpans(out.vars[i], extractedVar)) {
                        duplicate = true;
                        break;
                    }
                }
                if (!duplicate) {
                    assert(out.varCount < ExecutionParameters::MAX_ARITY
                        && "parseHypoScopeVars: variable count exceeds the "
                           "producer arity bound");
                    out.vars[out.varCount++] = extractedVar;
                }

                if (isLast) break;
                currentPos = nextPos;
                ++idx;
            }

            // Gate 4: no variables extracted -> reject.
            return out.varCount > 0;
        }

/// @brief Per-call mail-absorb pipeline: drain one external and
        /// one internal `Mail` container, apply equivalence classes,
        /// discharge `toBeProved` with the caller-supplied output
        /// internal-mail target, populate `mailOut`. Called twice per
        /// elementary step (pre-burst in `performElemPhase1`, post-burst in
        /// `performElemPhase3`):
        ///   - **pre-burst** with
        ///     `externalMailIn = &body.mailIn`,
        ///     `internalMailIn = body.nextIterationInternalMail`,
        ///     `internalMailOut = body.sameIterationInternalMail` —
        ///     any discharge emission during this absorb writes to
        ///     `sameIter` and is processed by the **same** step's
        ///     hashburst-following call 2 (zero-delay).
        ///   - **post-burst** with
        ///     `externalMailIn = nullptr` (no external arrivals after
        ///     the burst), `internalMailIn = body.sameIterationInternalMail`,
        ///     `internalMailOut = body.nextIterationInternalMail` —
        ///     any discharge emission during this absorb writes to
        ///     `nextIter` and is processed by the **next** step's
        ///     call 1 (one-step delay).
        ///
        /// @details
        /// In counterexample mode, checks existing registered contradictions and
        /// completed internal-mail heads against the loaded main-scope facts,
        /// then returns. Mail, equivalence classes, goals and all other pipeline
        /// stages are untouched in that mode; each clone ends after one burst.
        ///
        /// In ordinary mode, consolidates the absorb pipeline into a
        /// single reusable entry point. The two-mail input shape lets
        /// the caller choose between same-iteration (zero-delay) and
        /// cross-iteration (one-step-delay) routing for discharge
        /// emissions, without any new status code or parameter on
        /// `addExprToMemoryBlock`.
        ///
        /// Steps in order:
        ///
        /// 1. **Absorb external mail** (only if `externalMailIn` is
        ///    non-null): bulk merge `externalMailIn->exprOriginMap` into
        ///    `body.exprOriginMap` (capped, existing rows win); sync
        ///    equality origins into each containing class's
        ///    `equalityOriginMap` (trackHistory only); drain
        ///    `externalMailIn->statements` via flat `addExprToMemoryBlock`
        ///    with `status = 3` (each entry additionally staged into
        ///    `externalStatements` and `intExternalStatements` so the
        ///    burst's request generation sees the freshly-arrived rules
        ///    and facts as workingMemory-eligible — `vName == "main"`
        ///    asserted per element, [I-26](../../docs/30_invariants.md#i-26));
        ///    merge `externalMailIn->expandedImplications` into
        ///    `body.expandedImplications`.
        /// 2. **Absorb internal mail**: same four-step recipe but with
        ///    `status = 1` (no `intExternalStatements` staging — the
        ///    internal channel is integration-revival traffic, not a
        ///    workingMemory feed).
        /// 3. **Clear the internal channel** (`internalMailIn`). Clear
        ///    placement is **after drain, before** apply/discharge/fillMailOut
        ///    so any subsequent write to `internalMailOut` survives — this is
        ///    the load-bearing detail that makes the same-iter zero-delay
        ///    routing work for the pre-burst call (the same channel is both
        ///    `internalMailIn` of one direction and `internalMailOut` of the
        ///    other across the two calls; clearing in the middle is the only
        ///    timing that keeps both correct). The external routing inbox
        ///    (`body.mailIn`) is cleared by the caller after the call returns.
        /// 4. `reactToHypo(body)`, then `applyEquiClasses(body)` — hypothesis
        ///    copy equalities enter the same call's class processing and history
        ///    publication. One explicit equivalence-class
        ///    application pass over the freshly-populated
        ///    `intLocalEncodedStatementsDelta`.
        /// 5. `dischargeToBeProved(body, coreId, internalMailOut)` —
        ///    `toBeProved` closure pass. Discharge's two parent-scope
        ///    emission sites (OR-integration + NotOrScope) write to
        ///    `internalMailOut.statements` + `internalMailOut.exprOriginMap`.
        /// 6. `fillMailOut(body)` — centralised cross-LB `mailOut`
        ///    population from the same delta with `main` +
        ///    `allowedForMail` gates.
        /// 7. Clear `body.changedClassesThisStep` — the SOLE clear of the
        ///    delta tracker anywhere in the pipeline. Every entry present
        ///    here was consumed by this call's `applyEquiClasses` / deferred
        ///    cleanup sweep / recursion-products closure, except a class
        ///    minted by `dischargeContradiction`'s head deposit — whose LB
        ///    is deactivated in the same act and so has no future consumer.
        ///    Deltas minted OUTSIDE any `standardProcessing` call (LB
        ///    seeding, post-join barrier deposits) are cleared nowhere else
        ///    and survive to the next call's `applyEquiClasses` Pass 1.
        ///
        /// 8. Sanitize goals, unbind the phase-3 exclusion, drain disproved
        ///    goals and dead branches, retire duplicate cohorts, freeze resolved
        ///    branches, release pending cases, wipe the sorted union of closed
        ///    scopes once, then sweep ancestor-known rows. All inference loops
        ///    have finished before destructive cleanup. History and surviving
        ///    convergence evidence remain available.
        ///
        /// @param memoryBlock Owning LB. Read and mutated across most
        ///        steps: `exprOriginMap`, `equivalenceClassesMap`,
        ///        `externalStatements` and `intExternalStatements`
        ///        (only when `externalMailIn != nullptr`),
        ///        `expandedImplications`, `intStatementLevelsMap`,
        ///        `intEncodedStatements` (via `addExprToMemoryBlock` and
        ///        downstream calls), `mailOut`,
        ///        `changedClassesThisStep`.
        /// @param externalMailIn External-mail routing inbox (status=3), the
        ///        `RoutingColdMail` `&memoryBlock.mailIn`. May be `nullptr` (post-burst
        ///        call). Absorbed directly from its sorted snapshots (no transient
        ///        heap `Mail`); cleared by the CALLER after this call returns
        ///        (nothing between the drain and return reads it).
        /// @param internalMailIn Internal-mail input (status=1).
        ///        Pre-burst: `body.nextIterationInternalMail`. Post-burst:
        ///        `body.sameIterationInternalMail`. Cleared after absorption;
        ///        subsequent default-channel emissions survive the call.
        /// @param internalMailOut Output target for the call's
        ///        in-flight internal-mail writes (parent-scope
        ///        discharge emissions). Pre-burst:
        ///        `body.sameIterationInternalMail`. Post-burst:
        ///        `body.nextIterationInternalMail`. Not cleared by this
        ///        function — discharge writes here must survive past
        ///        this call to be processed by the appropriate future
        ///        absorb.
        /// @param coreId Core identifier forwarded to
        ///        `addExprToMemoryBlock` and `dischargeToBeProved`
        ///        for thread-local routing of global emissions.
        /// @see `applyEquiClasses`, `dischargeToBeProved`,
        ///      `fillMailOut` — the three pipeline stages this driver
        ///      orchestrates.
        /// @see [`docs/10_pipeline/04_prover.md`](../../docs/10_pipeline/04_prover.md).
        /// @see [`docs/20_core_concepts/03_mail_system.md`](../../docs/20_core_concepts/03_mail_system.md).
inline void LogicBlockOperations::standardProcessing(Memory& memoryBlock,
                                       RoutingColdMail* externalMailIn,
                                       ColdMail& internalMailIn,
                                       ColdMail& internalMailOut,
                                       int coreId) {
            if (ceFilteringActive) {
                assert(memoryBlock.contradictionIndex >= 0);
                assert(static_cast<std::size_t>(memoryBlock.contradictionIndex)
                    < contradictionTable.size());
                assert(!memoryBlock.primedForContradiction && !memoryBlock.isPartOfRecursion);
                watchProcessor.materializeWatchHits(memoryBlock, coreId);
                if (memoryBlock.isActive) orImplicationProcessor.drainOrImplications(memoryBlock, internalMailIn);
                goalProcessor.dischargeContradiction(memoryBlock, coreId);
                // Completed heads are in the internal channel, before any absorb.
                // CE only checks a main-scope head against the loaded main facts.
                const unsigned slot = coreId >= 0 ? static_cast<unsigned>(coreId)
                    : scratchArenas().slotCount() - 1;
                ScratchArena& arena = scratchArenas().forSlot(slot);
                for (int32_t row = 1; memoryBlock.isActive
                    && row <= internalMailIn.statements_.count(); ++row) {
                    const StrSpan key = internalMailIn.statements_.keyAt(row);
                    assert(key.len >= 12 && "IntMailStatementKey blob truncated");
                    int32_t originalId, validityId;
                    std::memcpy(&originalId, key.ptr, sizeof(originalId));
                    std::memcpy(&validityId, key.ptr + 4, sizeof(validityId));
                    if (validityId != NameMap::MAIN_ID) continue;
                    ScratchScope rowScope(arena);
                    const StrSpan head = memoryBlock.nameMap.decodeView(originalId);
                    const NameId opposite = memoryBlock.nameMap.lookup(gl::DefinitionExpansion::negateScratch(arena, head));
                    if (memoryBlock.intKnownStatements.find(
                        StatementKey{opposite, NameMap::MAIN_ID}) == nullptr) continue;
                    contradictionTable[memoryBlock.contradictionIndex].successful = true;
                    memoryBlock.isActive = false;
                }
                return;
            }
#if PHASE13_DEEP_TIMING
            const auto recordPhase13Detail =
                [this, coreId](Phase13TimingSlot slot,
                               std::chrono::steady_clock::time_point started) {
                    if (phase13TimingRows == nullptr) return;
                    assert(coreId >= 0
                        && static_cast<unsigned>(coreId) < phase13TimingWorkers
                        && "Phase 1/3 timing core id outside the worker rows");
                    const int64_t elapsed = std::chrono::duration_cast<
                        std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - started).count();
                    phase13TimingRows[
                        static_cast<std::size_t>(coreId)
                            * kPhase13TimingSlotCount
                        + static_cast<std::size_t>(slot)] += elapsed;
                };
#endif
            // Single per-mail absorb recipe — used for the optional
            // external drain (status=3) and the always-on internal
            // drain (status=1).
            auto absorb = [&](auto& mail, int status) {
                using MailT = std::decay_t<decltype(mail)>;
                constexpr bool isRoutingColdMail = std::is_same<MailT, RoutingColdMail>::value;

                // Origin processing (bulk exprOriginMap merge + equality-class
                // equalityOriginMap sync), iterating the cold origin ids directly —
                // no heap decodeMailInOrigins / decodeInternalMailOrigins snapshot.
                // The former snapshot was EWV-key sorted with each run std::sorted;
                // that is reproduced by two arena-backed decoded-lex indices on the
                // per-slot genScratchArenas: a CROSS-ROW index (decodedOriginKeyLess)
                // for the EWV-key row order, and a PER-RUN index (originLineLessId)
                // for the sorted-run order. Both orders are observable — the cross-
                // row order drives the exprOriginMap cold key mint order (and, for
                // the routing channel, the originInterner re-mint order → the deload
                // stream), the run order drives the cap gate — so
                // both are byte-reproduced (I-84, plan risk 3).
                // I-138 (L8 S2).
#if PHASE13_DEEP_TIMING
                const auto originAbsorbStarted =
                    std::chrono::steady_clock::now();
#endif
                {
                    RT_SCOPE_HERE(status == 3 ? "SP_ABSORB_ORIGINS_EXT"
                                              : "SP_ABSORB_ORIGINS_INT");
                    int32_t originN = 0;
                    if constexpr (isRoutingColdMail) {
                        if (mail.arenaInited_) originN = mail.origins_.count();
                    } else {
                        originN = mail.origins_.count();
                    }
                    if (originN > 0) {
                        RT_NOTE_ITERATIONS_HERE(originN);
                        const unsigned ogSlot = (gl::WorkerContext::coreId() >= 0)
                            ? static_cast<unsigned>(gl::WorkerContext::coreId())
                            : genScratchArenas().slotCount() - 1;
                        ScratchArena& ogArena = genScratchArenas().forSlot(ogSlot);
                        ScratchScope ogScope(ogArena);
                        // Cross-row index: origins_ ids in decoded EWV-key
                        // order. Block-chunked (ChunkSortedOrdinals): a mail
                        // deposit's row count is unbounded at rung-2+ scale,
                        // so the contiguous one-block index it replaces
                        // outgrew LbArena::alloc. decodedOriginKeyLess is a
                        // strict TOTAL order over distinct cold keys, so the
                        // merged sequence is byte-identical to the former
                        // single std::sort (I-84).
                        const auto rowLess = [&](int32_t a, int32_t b) {
                            if constexpr (isRoutingColdMail) {
                                return decodedOriginKeyLess(mail.origins_.decodeKey(a),
                                    mail.origins_.decodeKey(b), mailInterner());
                            } else {
                                return decodedOriginKeyLess(mail.origins_.decodeKey(a),
                                    mail.origins_.decodeKey(b), memoryBlock.originInterner);
                            }
                        };
                        ChunkSortedOrdinals<decltype(rowLess)> rowOrder(
                            ogArena, originN, 1, rowLess);

                        // Bulk exprOriginMap merge, in cross-row × sorted-run order.
                        // Nothing reads exprOriginMap inside this loop except
                        // each addition's own duplicate and cap check, so the
                        // accepted lines are staged and written in one
                        // rebuild after the loop (OriginAbsorbBatch, same
                        // policy, key ids and bytes as addOriginId per line).
                        {
                            const int maxOriginsForBulk = parameters.compressor_mode
                                ? parameters.compressor_max_origins_per_expr
                                : parameters.max_origin_per_expr;
                            // Incubator anchor LB: relay every routing-mail
                            // origin row into this LB's mailOut so the
                            // descendants (which do not read the root's log)
                            // still receive the history the root shipped —
                            // rows only, never the statements
                            // (D-332).
                            [[maybe_unused]] const bool relayHistory =
                                status == 3 && parameters.trackHistory
                                && blockIdentity.relaysAbsorbedHistory(memoryBlock);
                            OriginAbsorbBatch originBatch(memoryBlock.exprOriginMap, ogArena);
                            for (int32_t ri = 0; ri < originN; ++ri) {
                                const int32_t rid = rowOrder.next();
                                const int64_t coldKey = mail.origins_.decodeKey(rid);
                                const int32_t rr = mail.origins_.runLen(rid);
                                ScratchScope runScope(ogArena);
                                int32_t* runIdx = reinterpret_cast<int32_t*>(
                                    ogArena.resolve(ogArena.alloc(
                                        rr * static_cast<int32_t>(sizeof(int32_t)),
                                        static_cast<int32_t>(alignof(int32_t)))));
                                for (int32_t j = 0; j < rr; ++j) runIdx[j] = j;
                                if constexpr (isRoutingColdMail) {
                                    std::sort(runIdx, runIdx + rr, [&](int32_t a, int32_t b) {
                                        int32_t la, lb;
                                        const char* pa = mail.origins_.peekRecordBytes(rid, a, la, ogArena);
                                        const char* pb = mail.origins_.peekRecordBytes(rid, b, lb, ogArena);
                                        return originLineLessId(viewMailOriginBlob(pa, la),
                                            viewMailOriginBlob(pb, lb), mailInterner());
                                    });
                                } else {
                                    std::sort(runIdx, runIdx + rr, [&](int32_t a, int32_t b) {
                                        int32_t la, lb;
                                        const char* pa = mail.origins_.peekRecordBytes(rid, a, la, ogArena);
                                        const char* pb = mail.origins_.peekRecordBytes(rid, b, lb, ogArena);
                                        return originLineLessId(viewMailOriginBlob(pa, la),
                                            viewMailOriginBlob(pb, lb), memoryBlock.originInterner);
                                    });
                                }
                                if constexpr (isRoutingColdMail) {
                                    // Routing: key + deps are global mailInterner ids;
                                    // the span door re-mints them into originInterner
                                    // (byte-identically to the former decode → EWV →
                                    // addOriginEncoded), preserving the mint order.
                                    const int32_t keyExprId = static_cast<int32_t>(
                                        static_cast<uint64_t>(coldKey) >> 32);
                                    const int32_t keyValId = static_cast<int32_t>(
                                        coldKey & 0xFFFFFFFFLL);
                                    const StrSpan keyExpr = mailInterner().view(keyExprId);
                                    const StrSpan keyVal = mailInterner().view(keyValId);
                                    for (int32_t jj = 0; jj < rr; ++jj) {
                                        int32_t rl;
                                        const char* rp = mail.origins_.peekRecordBytes(
                                            rid, runIdx[jj], rl, ogArena);
                                        const MailOriginBlobView v = viewMailOriginBlob(rp, rl);
                                        OriginDep deps[64];
                                        assert(v.depN <= 64 && "origin chain exceeds 64");
                                        for (int32_t d = 0; d < v.depN; ++d) {
                                            const int64_t dk = mailOriginDepAt(v, d);
                                            const int32_t de = static_cast<int32_t>(
                                                static_cast<uint64_t>(dk) >> 32);
                                            const int32_t dv = static_cast<int32_t>(
                                                dk & 0xFFFFFFFFLL);
                                            deps[d] = { mailInterner().view(de),
                                                        mailInterner().view(dv) };
                                        }
                                        originBatch.addEncoded(
                                            memoryBlock.originInterner, keyExpr, keyVal,
                                            static_cast<OriginTag>(v.tag), deps, v.depN,
                                            maxOriginsForBulk);
                                        if (relayHistory) {
                                            memoryBlock.addMailOutOrigin(keyExpr, keyVal,
                                                static_cast<OriginTag>(v.tag), deps,
                                                v.depN, maxOriginsForBulk);
                                        }
                                    }
                                } else {
                                    // Internal: the cold record's ids ARE originInterner
                                    // ids and coldKey IS the exprOriginMap key — a pure
                                    // id copy, no re-mint (byte-identical to the former
                                    // decode → re-encode, which re-interned no-ops).
                                    // Mirror the routing branch's peek-view read: read
                                    // the blob zero-copy and feed the POD cold addOriginId
                                    // overload, so no heap IntMailOrigin / IdOrigin is
                                    // materialized.
                                    for (int32_t jj = 0; jj < rr; ++jj) {
                                        int32_t rl;
                                        const char* rp = mail.origins_.peekRecordBytes(
                                            rid, runIdx[jj], rl, ogArena);
                                        const MailOriginBlobView v = viewMailOriginBlob(rp, rl);
                                        int64_t deps[64];
                                        assert(v.depN <= 64 && "internal origin chain exceeds 64");
                                        for (int32_t d = 0; d < v.depN; ++d) {
                                            deps[d] = mailOriginDepAt(v, d);
                                        }
                                        originBatch.add(coldKey,
                                            v.tag, deps, v.depN, maxOriginsForBulk);
                                    }
                                }
                            }
                            originBatch.apply();
                        }

                        // Equality-class equalityOriginMap sync (trackHistory), same
                        // cross-row order (rowOrder replayed from the start). The
                        // matched class is read + rewritten via a zero-copy
                        // EquivalenceClassView cold RMW (no heap
                        // std::vector<EquivalenceClass> snapshot).
                        rowOrder.reset();
                        if (parameters.trackHistory) {
                            RT_SCOPE_HERE("SP_ABSORB_EQORIGIN_SYNC");
                            DirtyState metadataDirty = DirtyState::Clean;
                            TypedColdSet<NameId> metadataScopes(&ogArena,&metadataDirty);
                            const int maxOrig = parameters.compressor_mode
                                ? parameters.compressor_max_origins_per_expr
                                : parameters.max_origin_per_expr;
                            for (int32_t ri = 0; ri < originN; ++ri) {
                                const int32_t rid = rowOrder.next();
                                const int64_t coldKey = mail.origins_.decodeKey(rid);
                                const int32_t keyExprId = static_cast<int32_t>(
                                    static_cast<uint64_t>(coldKey) >> 32);
                                const int32_t keyValId = static_cast<int32_t>(
                                    coldKey & 0xFFFFFFFFLL);
                                const unsigned decodeSlot = gl::WorkerContext::coreId() >= 0 ? static_cast<unsigned>(gl::WorkerContext::coreId()) : scratchArenas().slotCount() - 1;
                                ScratchArena& decodeArena = scratchArenas().forSlot(decodeSlot);
                                ScratchScope decodeScope(decodeArena);
                                StrSpan exprSpan;
                                StrSpan vNameSpan;
                                if constexpr (isRoutingColdMail) {
                                    exprSpan = mailInterner().view(keyExprId);
                                    vNameSpan = mailInterner().view(keyValId);
                                } else {
                                    exprSpan = memoryBlock.originInterner.copyView(keyExprId, decodeArena);
                                    vNameSpan = memoryBlock.originInterner.copyView(keyValId, decodeArena);
                                }
                                if (exprSpan.len < 3
                                    || std::memcmp(exprSpan.ptr, "(=[", 3) != 0) continue;
                                StrSpan args[ExecutionParameters::MAX_ARITY];
                                const int32_t argN = getArgsSpans(exprSpan, args,
                                    ExecutionParameters::MAX_ARITY);
                                if (argN != 2) continue;
                                const NameId vId = memoryBlock.nameMap.lookup(vNameSpan);
                                if (vId == 0) continue;
                                const NameId arg0Id = memoryBlock.nameMap.lookup(args[0]);
                                const NameId arg1Id = memoryBlock.nameMap.lookup(args[1]);
                                if (arg0Id == 0 || arg1Id == 0) continue;
                                // Locate the matched class via a zero-copy
                                // EquivalenceClassView walk over the cold blob run —
                                // no heap std::vector<EquivalenceClass> snapshot. The
                                // classes at a scope are DISJOINT, so at most one
                                // contains BOTH args; find it (and the run's total
                                // bytes) in one pass. This is the updateEquivalenceClasses
                                // RMW twin. Per-row scratch (class peeks + the sorted
                                // origin index + the splice buffer) rides `ogArena`
                                // under `rowScope`.
                                ScratchScope rowScope(ogArena);
                                const int32_t crid =
                                    memoryBlock.equivalenceClassesMap.lookup(vId);
                                if (crid == 0) continue;
                                const int32_t crl =
                                    memoryBlock.equivalenceClassesMap.runLen(crid);
                                int32_t matchJ = -1;
                                int64_t sumAllBlens = 0;
                                int32_t matchedBlen = 0;
                                for (int32_t cj = 0; cj < crl; ++cj) {
                                    int32_t cblen = 0;
                                    const char* cbp = memoryBlock.equivalenceClassesMap
                                        .peekRecordBytes(crid, cj, cblen, ogArena);
                                    sumAllBlens += cblen;
                                    if (matchJ >= 0) continue;
                                    const EquivalenceClassView cv{ cbp, cblen };
                                    bool has0 = false;
                                    bool has1 = false;
                                    for (int32_t mi = 0; mi < cv.memberCount(); ++mi) {
                                        const NameId m = cv.memberId(mi);
                                        if (m == arg0Id) has0 = true;
                                        if (m == arg1Id) has1 = true;
                                    }
                                    if (has0 && has1) { matchJ = cj; matchedBlen = cblen; }
                                }
                                if (matchJ < 0) continue;   // no matching class

                                // Sorted per-run origin index (the former per-run
                                // std::sort), then decode ONLY the matched class,
                                // append the run's origins to its equalityOriginMap,
                                // and re-serialize.
                                const int32_t rr = mail.origins_.runLen(rid);
                                int32_t* runIdx = reinterpret_cast<int32_t*>(
                                    ogArena.resolve(ogArena.alloc(
                                        rr * static_cast<int32_t>(sizeof(int32_t)),
                                        static_cast<int32_t>(alignof(int32_t)))));
                                for (int32_t j = 0; j < rr; ++j) runIdx[j] = j;
                                if constexpr (isRoutingColdMail) {
                                    std::sort(runIdx, runIdx + rr, [&](int32_t a, int32_t b) {
                                        int32_t la, lb;
                                        const char* pa = mail.origins_.peekRecordBytes(rid, a, la, ogArena);
                                        const char* pb = mail.origins_.peekRecordBytes(rid, b, lb, ogArena);
                                        return originLineLessId(viewMailOriginBlob(pa, la),
                                            viewMailOriginBlob(pb, lb), mailInterner());
                                    });
                                } else {
                                    std::sort(runIdx, runIdx + rr, [&](int32_t a, int32_t b) {
                                        int32_t la, lb;
                                        const char* pa = mail.origins_.peekRecordBytes(rid, a, la, ogArena);
                                        const char* pb = mail.origins_.peekRecordBytes(rid, b, lb, ogArena);
                                        return originLineLessId(viewMailOriginBlob(pa, la),
                                            viewMailOriginBlob(pb, lb), memoryBlock.originInterner);
                                    });
                                }
                                // Seed a MergeClassAccum from the matched class's
                                // zero-copy view instead of decoding a heap
                                // EquivalenceClass: on an EMPTY accum
                                // unionMembersFromView / mergeLevelsBBase /
                                // mergeOriginsBBase are verbatim copies of the
                                // class's members / levels / origins (view order),
                                // so the accum holds the exact former mcls state.
                                // acc is declared AFTER rowScope so it destructs
                                // first — frees its page-tier containers before the
                                // scope rewinds the byte-bump tier (I-130/I-131).
                                int32_t mblen = 0;
                                const char* mbp = memoryBlock.equivalenceClassesMap
                                    .peekRecordBytes(crid, matchJ, mblen, ogArena);
                                const EquivalenceClassView mview{ mbp, mblen };
                                MergeClassAccum acc(&ogArena);
                                acc.unionMembersFromView(mview, memoryBlock.nameMap);
                                acc.mergeLevelsBBase(mview);
                                acc.mergeOriginsBBase(mview, maxOrig);
                                for (int32_t jj = 0; jj < rr; ++jj) {
                                    if constexpr (isRoutingColdMail) {
                                        int32_t rl;
                                        const char* rp = mail.origins_.peekRecordBytes(
                                            rid, runIdx[jj], rl, ogArena);
                                        const MailOriginBlobView v = viewMailOriginBlob(rp, rl);
                                        OriginDep deps[64];
                                        assert(v.depN <= 64 && "origin chain exceeds 64");
                                        for (int32_t d = 0; d < v.depN; ++d) {
                                            const int64_t dk = mailOriginDepAt(v, d);
                                            const int32_t de = static_cast<int32_t>(
                                                static_cast<uint64_t>(dk) >> 32);
                                            const int32_t dv = static_cast<int32_t>(
                                                dk & 0xFFFFFFFFLL);
                                            deps[d] = { mailInterner().view(de),
                                                        mailInterner().view(dv) };
                                        }
                                        // Byte-twin of addOriginEncoded (key-first
                                        // mint into originInterner, same runIdx
                                        // sorted-run order — mint sequence preserved).
                                        acc.addOriginEnc(memoryBlock.originInterner,
                                            exprSpan, vNameSpan,
                                            static_cast<OriginTag>(v.tag), deps, v.depN,
                                            maxOrig);
                                    } else {
                                        int32_t rl;
                                        const char* rp = mail.origins_.peekRecordBytes(
                                            rid, runIdx[jj], rl, ogArena);
                                        const MailOriginBlobView v = viewMailOriginBlob(rp, rl);
                                        int64_t deps[64];
                                        assert(v.depN <= 64 && "origin chain exceeds 64");
                                        for (int32_t d = 0; d < v.depN; ++d) {
                                            deps[d] = mailOriginDepAt(v, d);
                                        }
                                        // Byte-twin of the POD addOriginId (mints
                                        // nothing — internal deps are originInterner
                                        // ids already).
                                        acc.addOriginId(coldKey, v.tag, deps, v.depN,
                                            maxOrig);
                                    }
                                }
                                // serialize -> mBlob byte-identical to the former
                                // serializeEquivalenceClassInto(ogArena, mcls): the
                                // two serializers share layout field-for-field
                                // (members, ascending level keys, ascending origin
                                // keys with per-key lines in insertion order). The
                                // splice below is unchanged.
                                const int32_t mBlobLen = acc.serializedSize();
                                char* mBlobBuf = reinterpret_cast<char*>(
                                    ogArena.resolve(ogArena.alloc(mBlobLen, 1)));
                                acc.serializeInto(mBlobBuf);
                                const StrSpan mBlob(mBlobBuf, mBlobLen);
                                if (mBlobLen != mblen || std::memcmp(mBlobBuf, mbp, mBlobLen) != 0)
                                    metadataScopes.mint(vId);


                                // Write-back: splice the run in order, the
                                // matched class replaced by mBlob, the rest raw blobs
                                // verbatim (a raw blob == re-serializing its decoded
                                // class — the codec is a lossless pure function of
                                // content, I-103), so byte-identical to the former
                                // decodeClassesById + append + assignClassesById. The
                                // kept blobs are copied out BEFORE the assignRun that
                                // replaces the run, so the peeked pointers never dangle.
                                // mBlob rides ogArena's byte-bump tier below the outBytes
                                // alloc; byte-bump blocks are stable (I-107) so it stays
                                // valid across the splice allocs and peeks.
                                {
                                    const int64_t total = sumAllBlens
                                        - static_cast<int64_t>(matchedBlen)
                                        + static_cast<int64_t>(mBlob.len);
                                    // Per-blob staging + the emitter run door:
                                    // each kept blob is copied into its OWN
                                    // byte-bump allocation (single-blob ≤ one
                                    // block, the pre-existing peek bound) and
                                    // the run is written via assignRunGenerated
                                    // — no whole-run contiguous buffer, so the
                                    // run total may exceed one pool block. The
                                    // engine's contract makes the CSR bytes
                                    // identical to the former one-buffer concat
                                    // + assignRun. Copies are taken BEFORE the
                                    // run-replacing write, so the peeked
                                    // pointers never dangle.
                                    StrSpan* runBlobs = reinterpret_cast<StrSpan*>(
                                        ogArena.resolve(ogArena.alloc(
                                            crl * static_cast<int32_t>(sizeof(StrSpan)),
                                            static_cast<int32_t>(alignof(StrSpan)))));
                                    int64_t off = 0;
                                    for (int32_t cj = 0; cj < crl; ++cj) {
                                        if (cj == matchJ) {
                                            runBlobs[cj] = mBlob;
                                        } else {
                                            int32_t cblen = 0;
                                            const char* cbp = memoryBlock.equivalenceClassesMap
                                                .peekRecordBytes(crid, cj, cblen, ogArena);
                                            char* copy = reinterpret_cast<char*>(
                                                ogArena.resolve(ogArena.alloc(
                                                    cblen > 0 ? cblen : 1, 1)));
                                            std::memcpy(copy, cbp,
                                                        static_cast<std::size_t>(cblen));
                                            runBlobs[cj] = StrSpan(copy, cblen);
                                        }
                                        off += runBlobs[cj].len;
                                    }
                                    assert(off == total);
                                    memoryBlock.equivalenceClassesMap.inner()
                                        .assignRunGenerated(vId, crl,
                                            static_cast<int32_t>(total),
                                            [&](auto&& sink) {
                                                for (int32_t cj = 0; cj < crl; ++cj)
                                                    sink(runBlobs[cj].ptr,
                                                         runBlobs[cj].len);
                                            });
                                }
                            }
                            for (int32_t i=1;i<=metadataScopes.count();++i) {
                                const NameId scope=metadataScopes.keyAt(i);
                                bool inherited=false;
                                for (NameId parent=memoryBlock.nameMap.parentOf(scope);parent!=0;parent=memoryBlock.nameMap.parentOf(parent))
                                    if (metadataScopes.contains(parent)) { inherited=true; break; }
                                if (!inherited) equivalenceProcessor.refreshCanonicalClasses(memoryBlock,scope);
                            }
                        }
                    }
                }
#if PHASE13_DEEP_TIMING
                recordPhase13Detail(
                    status == 3
                        ? Phase13TimingSlot::externalOriginAbsorb
                        : Phase13TimingSlot::internalOriginAbsorb,
                    originAbsorbStarted);
                const auto statementAbsorbStarted =
                    std::chrono::steady_clock::now();
#endif

                // Drain mail.statements via flat addExprToMemoryBlock, iterating
                // the cold statement ids directly in decoded-lex order. The former
                // `decodeMailInStatements` / `decodeInternalMailStatements` heap
                // snapshot's `std::pair<EWV, std::set<int>>::operator<` `std::sort`
                // is reproduced by `decodedStatementLess` over a byte-bump index on
                // the per-slot gen-scratch arena (the drain order is observable —
                // `intExternalStatements` push order, rule-firing order, delta
                // order downstream — so the sort must be byte-reproduced).
                // I-138. The two owning-`std::string`
                // boundaries the off-limits `addExprToMemoryBlock` door takes are
                // ONE reused per-drain buffer each, materialized on demand from the
                // id decode spans (the honest residual; a span door for that
                // hash-memory door is a separate out-of-scope campaign).
                {
                    RT_SCOPE_HERE(status == 3 ? "SP_ABSORB_STMTS_EXT"
                                              : "SP_ABSORB_STMTS_INT");
                    int32_t stmtN = 0;
                    if constexpr (isRoutingColdMail) {
                        if (mail.arenaInited_) stmtN = mail.statements_.count();
                    } else {
                        stmtN = mail.statements_.count();
                    }
                    if (stmtN > 0) {
                        RT_NOTE_ITERATIONS_HERE(stmtN);
                        const unsigned stSlot = (gl::WorkerContext::coreId() >= 0)
                            ? static_cast<unsigned>(gl::WorkerContext::coreId())
                            : genScratchArenas().slotCount() - 1;
                        ScratchArena& stArena = genScratchArenas().forSlot(stSlot);
                        ScratchScope stScope(stArena);
                        // Statement rows in decoded (original, validityName)
                        // order. Block-chunked (ChunkSortedOrdinals) like the
                        // origin cross-row index above: the deposit's row
                        // count is unbounded, decodedStatementLess is a
                        // strict TOTAL order over distinct cold keys, so the
                        // merged sequence is byte-identical to the former
                        // single std::sort (I-84 / I-102 sorted deposit
                        // boundary).
                        const auto stmtLess = [&](int32_t a, int32_t b) {
                            if constexpr (isRoutingColdMail) {
                                return decodedStatementLess(mail.statements_.keyAt(a),
                                    mail.statements_.keyAt(b), mailInterner());
                            } else {
                                return decodedStatementLess(mail.statements_.keyAt(a),
                                    mail.statements_.keyAt(b), memoryBlock.nameMap);
                            }
                        };
                        ChunkSortedOrdinals<decltype(stmtLess)> stmtOrder(
                            stArena, stmtN, 1, stmtLess);

                        // addExprToMemoryBlock now takes StrSpan expr + StrSpan
                        // validityName; the two former reused heap std::string
                        // buffers are gone. Its mints into the NameMap /
                        // mailInterner would dangle a raw decodeView span (I-3), so
                        // the drain copies each decoded key onto the string-scratch
                        // arena (disjoint from those cold tables) under a per-row
                        // ScratchScope — the S4 consumer-edge recipe.
                        const unsigned drStrSlot = (gl::WorkerContext::coreId() >= 0)
                            ? static_cast<unsigned>(gl::WorkerContext::coreId())
                            : scratchArenas().slotCount() - 1;
                        ScratchArena& drStrArena = scratchArenas().forSlot(drStrSlot);
                        for (int32_t ii = 0; ii < stmtN; ++ii) {
                            // Read the fixed-layout key blob zero-copy (int32
                            // originalId, int32 validityId, int32 count, then count
                            // int32 levels — Codec<IntMailStatementKey>) via memcpy
                            // field reads, so no heap IntMailStatementKey / its
                            // std::vector<int32_t> levels is built. Mirrors how
                            // decodedStatementLess reads the same blob.
                            const StrSpan kb = mail.statements_.keyAt(stmtOrder.next());
                            assert(kb.len >= 12 && "IntMailStatementKey blob truncated");
                            int32_t kOrig, kVal, levelCount;
                            std::memcpy(&kOrig, kb.ptr + 0, 4);
                            std::memcpy(&kVal, kb.ptr + 4, 4);
                            if constexpr (!isRoutingColdMail) {
                                if (memoryBlock.phase3IgnoresScope(static_cast<NameId>(kVal))) {
                                    // The excluded subtree drops this delivery unabsorbed;
                                    // a source replay it carried retires its pending
                                    // receipt, as an absorbed one would (I-232).
                                    bool droppedDnd = false;
                                    bool droppedAod = false;
                                    int32_t droppedGeneration = -1;
                                    int32_t droppedDepth = -1;
                                    mail.getDisintegrationSignal(packOriginKey(kOrig, kVal),
                                        droppedDnd, droppedAod, droppedGeneration, nullptr, nullptr, &droppedDepth);
                                    rejectedSources.retireDroppedReplay(memoryBlock,
                                        packOriginKey(kOrig, kVal), droppedDepth);
                                    continue;
                                }
                            }
                            std::memcpy(&levelCount, kb.ptr + 8, 4);
                            assert(levelCount >= 0 && levelCount <= 256
                                && "mail level run exceeds caller cap");
                            int lvRun[256];
                            std::memcpy(lvRun, kb.ptr + 12,
                                static_cast<size_t>(levelCount) * sizeof(int32_t));
                            StrSpan so;
                            StrSpan sv;
                            if constexpr (isRoutingColdMail) {
                                so = mailInterner().view(kOrig);
                                sv = mailInterner().view(kVal);
                            } else {
                                so = memoryBlock.nameMap.decodeView(
                                    static_cast<NameId>(kOrig));
                                sv = memoryBlock.nameMap.decodeView(
                                    static_cast<NameId>(kVal));
                            }
                            ScratchScope drRowScope(drStrArena);
                            const ScratchString statement =
                                ScratchString::copyFrom(drStrArena, so.ptr, so.len);
                            const ScratchString vName =
                                ScratchString::copyFrom(drStrArena, sv.ptr, sv.len);
                            // The arrival is offered RAW: the door
                            // (`addExprToMemoryBlock`) owns the class
                            // canonicalization for every deposit and hands
                            // back the encoded row of the form it proceeded
                            // with, which the status-3 staging below consumes
                            // (from an equivalence class at most one member
                            // reaches a hashburst — the externalStatements
                            // purge — and both consumers see the SAME form).
                            IntEncodedExpr registered{};
                            if (status == 3) {
                                assert(equalSpans(StrSpan(vName), StrSpan("main", 4)));
                            }

                            // addExprToMemoryBlock owns all dedup (its Site F
                            // intKnownStatements ancestor scan is a superset of this
                            // per-statement intStatementLevelsMap gate), so every
                            // mailed statement is drained unconditionally. Bare scope
                            // keeps the origin/signal locals from leaking.
                            {
                                ScratchScope frontScope(stArena);
                                OriginDep drainDeps[64];
                                TransientOrigin origin{};
                                RT_SCOPE_HERE("SP_ABSORB_STMT_ROW");
                                if (parameters.trackHistory) {
                                    RT_SCOPE_HERE("SP_ABSORB_ROW_ORIGIN_MIN");
                                    // D-45: every mailed statement has a paired
                                    // origin. A direct origin-key lookup + a decoded
                                    // min-scan replaces the lower_bound on the former
                                    // sorted origins snapshot — min-of-run under
                                    // originLineLessId == the former sorted run's
                                    // .front() (no non-identical ties). Routing shares
                                    // mailInterner with the statement key; internal
                                    // crosses NameMap → originInterner via the
                                    // non-minting span probe.
                                    int64_t okey = 0;
                                    if constexpr (isRoutingColdMail) {
                                        okey = packOriginKey(kOrig, kVal);
                                    } else {
                                        const bool hit = lookupOriginKey(
                                            memoryBlock.originInterner, StrSpan(statement),
                                            StrSpan(vName), okey);
                                        assert(hit && "D-45: mailed statement missing "
                                            "its internal origin key");
                                    }
                                    const int32_t oid = mail.origins_.lookup(okey);
                                    assert(oid != 0 && "D-45: mailed statement missing "
                                        "its origin run");
                                    {
                                    const int32_t rr = mail.origins_.runLen(oid);
                                    assert(rr > 0);
                                    int32_t bestLen = 0;
                                    const char* bestPtr = mail.origins_.peekRecordBytes(
                                        oid, 0, bestLen, stArena);
                                    MailOriginBlobView best =
                                        viewMailOriginBlob(bestPtr, bestLen);
                                    for (int32_t j = 1; j < rr; ++j) {
                                        int32_t lj;
                                        const char* pj = mail.origins_.peekRecordBytes(
                                            oid, j, lj, stArena);
                                        const MailOriginBlobView vj =
                                            viewMailOriginBlob(pj, lj);
                                        bool less;
                                        if constexpr (isRoutingColdMail)
                                            less = originLineLessId(vj, best, mailInterner());
                                        else
                                            less = originLineLessId(vj, best,
                                                memoryBlock.originInterner);
                                        if (less) best = vj;
                                    }
                                    // Materialize the winning record's deps as spans
                                    // over COPIES on the scratch arena:
                                    // addExprToMemoryBlock mints into originInterner,
                                    // so a live decodeView span would dangle (I-3) —
                                    // copy uniformly for both channels.
                                    int n = 0;
                                    for (int32_t d = 0; d < best.depN; ++d) {
                                        assert(n < 64 && "drain origin chain exceeds 64");
                                        const int64_t dk = mailOriginDepAt(best, d);
                                        const int32_t de = static_cast<int32_t>(
                                            static_cast<uint64_t>(dk) >> 32);
                                        const int32_t dv = static_cast<int32_t>(
                                            dk & 0xFFFFFFFFLL);
                                        StrSpan eSp;
                                        StrSpan vSp;
                                        if constexpr (isRoutingColdMail) {
                                            eSp = mailInterner().view(de);
                                            vSp = mailInterner().view(dv);
                                        } else {
                                            eSp = memoryBlock.originInterner.copyView(de, stArena);
                                            vSp = memoryBlock.originInterner.copyView(dv, stArena);
                                        }
                                        if constexpr (isRoutingColdMail) {
                                        char* eC = reinterpret_cast<char*>(stArena.resolve(
                                            stArena.alloc(eSp.len > 0 ? eSp.len : 1, 1)));
                                        char* vC = reinterpret_cast<char*>(stArena.resolve(
                                            stArena.alloc(vSp.len > 0 ? vSp.len : 1, 1)));
                                        if (eSp.len > 0)
                                            std::memcpy(eC, eSp.ptr,
                                                        static_cast<size_t>(eSp.len));
                                        if (vSp.len > 0)
                                            std::memcpy(vC, vSp.ptr,
                                                        static_cast<size_t>(vSp.len));
                                        drainDeps[n++] = { StrSpan(eC, eSp.len),
                                                           StrSpan(vC, vSp.len) };
                                        } else {
                                            drainDeps[n++] = { eSp, vSp };
                                        }
                                    }
                                    origin = TransientOrigin{
                                        true, static_cast<OriginTag>(best.tag),
                                        drainDeps, n };
                                    }
                                }
                                // Recover the firing-time disintegration decision
                                // and the witness-generation stamp carried
                                // alongside this mailed statement (absent =>
                                // both false, iteration -1). The bools restore
                                // the OR-disintegration of or0 facts that the
                                // flat mail deposit had dropped; the iteration
                                // reaches the it_-witness mint so a fired
                                // head's witnesses carry max premise iteration
                                // + 1 (D-233) —
                                // the -1 default maps to generation 0 inside
                                // disintegrateExprCore2, keeping every
                                // non-firing deposit at generation 0. Only the
                                // internal (ColdMail) channel carries signals;
                                // RoutingColdMail has no such column.
                                bool dndSig = false;
                                bool aodSig = false;
                                bool rhSig = false;
                                bool orSourceSig = false;
                                bool forcedSig = false;
                                bool premiseOrdisSig = false;
                                bool fromPremiseSig = false;
                                int32_t itSig = -1;
                                int32_t integrationDepthSig = -1;
                                int stmtStatus = status;
                                if constexpr (!isRoutingColdMail) {
                                    // Id-form signal lookup: the statement's NameMap
                                    // pair is the decoded key's ids in hand (no
                                    // re-lookup). The out-param door writes the
                                    // fields directly, so the drain holds no
                                    // by-value std::pair local.
                                    mail.getDisintegrationSignal(
                                        packOriginKey(kOrig, kVal),
                                        dndSig, aodSig, itSig, &rhSig, &orSourceSig, &integrationDepthSig,
                                        &forcedSig);
                                    // The case of a cohort opened from a
                                    // conjecture premise's OR is disintegrated
                                    // like a premise, with force_ witnesses
                                    // (D-527).
                                    if (forcedSig) {
                                        assert(status == 1
                                            && "standardProcessing: a forced case assumption rides internal mail");
                                        stmtStatus = 0;
                                    }
                                } else {
                                    // Flag-5 relay (D-284): a
                                    // routing statement carrying the "external,
                                    // may be disintegrated" flag enters the door
                                    // at STATUS 5 with the sender's carried
                                    // witness-generation stamp — the receiver
                                    // disintegrates it with witness minting,
                                    // registers the carrier non-local, and its
                                    // admitted products local. Absent flag keeps
                                    // the plain status-3 bare-row absorb.
                                    if (status == 3) {
                                        int32_t flagCode = kMailStatementFlagNone;
                                        int32_t flagIter = -1;
                                        mail.getStatementFlag(
                                            packOriginKey(kOrig, kVal),
                                            flagCode, flagIter, &fromPremiseSig);
                                        if (flagCode ==
                                            kMailStatementFlagExternalDisintegrate) {
                                            stmtStatus = 5;
                                            itSig = flagIter;
                                        } else if (flagCode ==
                                            kMailStatementFlagUnconditionalOrdis) {
                                            // An OR reached by unfolding a
                                            // conjecture premise: it stays a
                                            // status-3 arrival with the
                                            // conjecture-premise permission, so a
                                            // receiver with proof work opens its
                                            // case split unconditionally and
                                            // forces its cases
                                            // (D-528,
                                            // D-527).
                                            premiseOrdisSig = true;
                                        }
                                    }
                                }
                                // The registry offer of the RAW arrival; the
                                // door canonicalizes, unions the applied pairs'
                                // levels, and reports the encoded row it
                                // proceeded with in `registered`.
                                int32_t replayKinds = 0;
                                if constexpr (!isRoutingColdMail) {
                                    const int64_t sourceKey = packOriginKey(kOrig, kVal);
                                    const int32_t* state = memoryBlock.existenceReplays.find(sourceKey);
                                    if (state != nullptr && (*state & 12) != 0) {
                                        // A regular arrival can share this source key with
                                        // a replay waiting in the other internal channel.
                                        // Only the delivery carrying its assignment consumes
                                        // the pending receipt.
                                        const int32_t pendingDepth = memoryBlock.pendingIntegrationDepth(sourceKey);
                                        assert(pendingDepth >= 0);
                                        if (integrationDepthSig >= 0) replayKinds = (*state >> 2) & 3;
                                        if (replayKinds != 0) {
                                            const int32_t savedState = *state;
                                            assert(integrationDepthSig >= 0);
                                            const NameId canonical = rejectedSources.canonicalExistenceId(memoryBlock,
                                                StrSpan(statement), static_cast<NameId>(kVal));
                                            rejectedSources.ensureCanonicalReceiptUses(memoryBlock, stArena);
                                            memoryBlock.lbMemory.canonicalUses.pendingCarriers.forEach(
                                                packOriginKey(canonical, kVal), [&](int64_t carrier) {
                                                    const int32_t* other = memoryBlock.existenceReplays.find(carrier);
                                                    assert(other != nullptr);
                                                    if (((*other >> 2) & replayKinds) == 0) return;
                                                    integrationDepthSig = std::min(integrationDepthSig,
                                                        memoryBlock.pendingIntegrationDepth(carrier));
                                                });
                                            if ((savedState & 16) != 0) replayKinds |= 8;
                                            rejectedSources.writeExistenceState(memoryBlock, sourceKey, savedState & 3);
                                        }
                                    }
                                }
                                statementMemory.addExprToMemoryBlock(StrSpan(statement), memoryBlock,
                                    itSig, stmtStatus,
                                    lvRun, levelCount,
                                    origin, coreId, -1, vName, dndSig, aodSig,
                                    &registered, rhSig, &internalMailOut, replayKinds,
                                    replayKinds != 0 ? integrationDepthSig : -1,
                                    /*integrationPremise=*/false, /*conjecturePremise=*/premiseOrdisSig);
                                // A product of an ancestor's forced
                                // disintegration joins this block's list as it
                                // arrived, whether or not the block knew it
                                // already, and in the form the door registered
                                // (D-522).
                                if (fromPremiseSig) {
                                    const NameId arrivalId = memoryBlock.nameMap.lookup(StrSpan(statement));
                                    const NameId arrivalScope = memoryBlock.nameMap.lookup(vName);
                                    assert(arrivalId != 0 && arrivalScope != 0
                                        && "standardProcessing: the door encodes every arrival");
                                    memoryBlock.lbMemory.forcedProducts.mint(
                                        packStatementKey(arrivalId, arrivalScope));
                                    if (registered.originalId != 0)
                                        memoryBlock.lbMemory.forcedProducts.mint(
                                            packStatementKey(registered.originalId, registered.validityId));
                                }
                                if (orSourceSig) {
                                    const NameId scope = memoryBlock.nameMap.lookup(vName);
                                    assert(scope != 0);
                                    orImplicationProcessor.activateOrImplicationContinuation(memoryBlock,
                                        StrSpan(statement), scope, registered.originalId, &internalMailOut);
                                }

                                // External-mail staging — enables request
                                // generation to see freshly-arrived rules and
                                // facts as workingMemory-eligible. I-26
                                // (routing-channel main-only) asserted per item.
                                // Staged in the door's CANONICAL form; two raw
                                // arrivals collapsing to one canonical row
                                // stage once (id-pair dedup — the drain's
                                // sorted raw-row visit keeps the push order
                                // deterministic).
                                //
                                // Gated on the offer above having REGISTERED the
                                // form it proceeded with (`registered` stays
                                // zeroed when the door returned before encoding:
                                // Site F / H refusal, the canonical skip): the
                                // door refuses some shapes outright (an axed
                                // anchor-completion argument, D-234), and a
                                // statement with no registry row can never be a
                                // request premise — the hashburst sink's
                                // dependency skip discards every request that
                                // contains one. Staging such a row would hand
                                // request generation a mandatory ingredient that
                                // cannot fire. An arrival the door rewrote
                                // under a LOCAL equality registered as this
                                // LB's own class product and rides the local
                                // delta — it is not external any more — so
                                // only a row that stayed non-local stages.
                                const StatementFlags* stagedFlags =
                                    registered.originalId == 0 ? nullptr
                                    : memoryBlock.intKnownStatements.find(
                                          StatementKey{ registered.originalId,
                                                        registered.validityId });
                                if (status == 3
                                    && stagedFlags != nullptr
                                    && !stagedFlags->local) {
                                    RT_SCOPE_HERE("SP_ABSORB_STAGE_EXTERNAL");
                                    bool alreadyStaged = false;
                                    for (std::size_t si = 0;
                                         si < memoryBlock.intExternalStatements.size();
                                         ++si) {
                                        const IntEncodedExpr& row =
                                            memoryBlock.intExternalStatements[si];
                                        if (row.originalId == registered.originalId
                                            && row.validityId == registered.validityId) {
                                            alreadyStaged = true;
                                            break;
                                        }
                                    }
                                    if (!alreadyStaged)
                                        memoryBlock.intExternalStatements.push_back(
                                            registered);
                                }
                            }
                        }
                    }
                }
#if PHASE13_DEEP_TIMING
                recordPhase13Detail(
                    status == 3
                        ? Phase13TimingSlot::externalStatementAbsorb
                        : Phase13TimingSlot::internalStatementAbsorb,
                    statementAbsorbStarted);
#endif

                // (The mail expandedImplications column was removed: it was never
                // serialized / delivered cross-LB — Codec<Mail> carries only
                // statements + exprOriginMap — so this merge was always a no-op.
                // Each LB's own expandedImplications index is populated locally at
                // its sanitizeHashMemory site, not via mail.)
            };

            // Rule-index staging window (D-333):
            // every rule install of both absorbs stages its index writes;
            // the window closes with one flush per touched map, before the
            // first reader (the apply's compact hook, phase 2).
            ++memoryBlock.ruleStagingArmDepth;
            assert(memoryBlock.replayMailDuringAbsorb == nullptr);
            assert(&internalMailIn != &internalMailOut);
            memoryBlock.replayMailDuringAbsorb = &internalMailOut;
            // Step 1: external absorb (optional, status=3). externalMailIn is the
            // routing inbox (RoutingColdMail, &body.mailIn); the absorb reads its
            // canonical sorted snapshots directly — no transient heap Mail.
            if (externalMailIn != nullptr) {
                absorb(*externalMailIn, 3);
            }
            // Step 2: internal absorb (always, status=1). internalMailIn is a
            // ColdMail, absorbed directly from its sorted snapshots + the
            // disintegration-signal read door — no transient heap Mail.
            absorb(internalMailIn, 1);
            memoryBlock.replayMailDuringAbsorb = nullptr;
            --memoryBlock.ruleStagingArmDepth;
            if (memoryBlock.ruleStagingArmDepth == 0) {
                RT_SCOPE_HERE("SP_FLUSH_RULE_STAGING");
                gl::HashRuleIndex::closeRuleIndexStaging(memoryBlock);
            }

            // Step 3: clear the internal channel AFTER drain, BEFORE apply /
            // discharge / fillMailOut. This ordering is load-bearing — discharge
            // writes to internalMailOut during step 5; sameIter is internalMailOut
            // pre-burst and internalMailIn post-burst, so the only safe clear
            // position is here, in the middle of the per-call body. The external
            // routing inbox (body.mailIn) is cleared by the caller after this call
            // returns — nothing between the drain and return reads it.
#if PHASE13_DEEP_TIMING
            const auto internalMailClearStarted =
                std::chrono::steady_clock::now();
#endif
            {
                RT_SCOPE_HERE("SP_CLEAR_INTERNAL_MAIL");
                internalMailIn.clear();   // ColdMail clear (resetToFresh x3 columns)
            }
#if PHASE13_DEEP_TIMING
            recordPhase13Detail(Phase13TimingSlot::internalMailClear,
                                internalMailClearStarted);
#endif

#if PHASE13_DEEP_TIMING
            const auto reactToHypothesisStarted = std::chrono::steady_clock::now();
#endif
            { RT_SCOPE_HERE("REACT_TO_HYPO");
                this->reactToHypo(memoryBlock);
            } // RT_SCOPE REACT_TO_HYPO
#if PHASE13_DEEP_TIMING
            recordPhase13Detail(Phase13TimingSlot::reactToHypothesis,
                               reactToHypothesisStarted);
#endif

#if PHASE13_DEEP_TIMING
            const auto applyEquivalenceStarted =
                std::chrono::steady_clock::now();
#endif

            // Step 4: apply equivalence classes.
            {
                RT_SCOPE_HERE("SP_APPLY_EQUI_CLASSES");
                equivalenceProcessor.applyEquiClasses(memoryBlock);
                unaryIntegrationProcessor.drainUnaryComposition(memoryBlock);
                unaryIntegrationProcessor.transferUnaryIntegrationDemands(memoryBlock);
            }
#if PHASE13_DEEP_TIMING
            recordPhase13Detail(Phase13TimingSlot::applyEquivalenceClasses,
                                applyEquivalenceStarted);
            const auto enrichProductsStarted =
                std::chrono::steady_clock::now();
#endif

            // Step 4a2: close productsOfRecursionIds under main-scope class
            // membership for the classes this step changed — a name proven
            // equal to a product of recursion is one
            // (D-285).
            {
                RT_SCOPE_HERE("SP_ENRICH_RECURSION_PRODUCTS");
                this->enrichProductsOfRecursionFromChangedClasses(memoryBlock);
            }
#if PHASE13_DEEP_TIMING
            recordPhase13Detail(Phase13TimingSlot::enrichProductsOfRecursion,
                                enrichProductsStarted);
            const auto cleanupExpressionsStarted =
                std::chrono::steady_clock::now();
#endif

            // Step 4b: deferred expression-side cleanup. cleanUpExpressions
            // must NOT run inline in `updateEquivalenceClasses` or in
            // `addStatement`'s isEquality-suffix block — both of which run
            // BEFORE this step. That ordering would delete non-canonical
            // intEncodedStatements rows before `applyEquiClasses` could read
            // them and emit canonical rewrites, so any row that existed
            // only in the non-canonical form would be lost without
            // replacement. The cleanup runs once HERE, after the
            // rewriter's fixpoint. One call sweeps every unique validity that
            // gained a delta class or a complete-class change this step.
            //
            // The admission maps need no sweep at this step:
            // applyEquivalenceClassToAdmissionMap and
            // applyEquivalenceClassToAdmissionMapIntegration both canonicalise
            // inline — each hook drops a key the moment a class rewrite
            // changes it and inserts the canonical form in its place
            // (D-106), so no non-canonical admission key survives to this
            // point. The expression-side sweep is the only deferred one.
            {
                RT_SCOPE_HERE("SP_CLEANUP_EXPRESSIONS");
                // The scheduled scopes in first-occurrence order — the
                // complete-class changes, then this step's application
                // deltas, minus the scopes excluded from Phase 3 — on a
                // page-tier ColdHashSet of the per-slot gen-scratch arena
                // (changedClassesThisStep is not statically bounded, so
                // ColdHashSet not a stack array; I-138). Its ids are the sweep
                // order and its lookup maps a row's scope to that order.
                // DirtyState is declared before the set so the set destructs
                // first.
                const unsigned dcSlot = (gl::WorkerContext::coreId() >= 0)
                    ? static_cast<unsigned>(gl::WorkerContext::coreId())
                    : genScratchArenas().slotCount() - 1;
                ScratchArena& dcArena = genScratchArenas().forSlot(dcSlot);
                DirtyState dcDirty = DirtyState::Clean;
                ColdHashSet<PodKeyStore<NameId>> cleanupScopes(
                    &dcArena, &dcDirty);
                cleanupScopes.resetToFresh();
                const auto scheduleScope = [&](const NameId scope) {
                    if (memoryBlock.phase3IgnoresScope(scope)) return;
                    if (cleanupScopes.lookup(scope) == 0) cleanupScopes.mint(scope);
                };
                for (int32_t i = 1; i <= memoryBlock.lbMemory.canonicalClassChanges.count(); ++i)
                    scheduleScope(Codec<StatementKey>::decode(memoryBlock.lbMemory.canonicalClassChanges.keyAt(i)).validity);
                for (int32_t i = 0; i < memoryBlock.changedClassesThisStep.size(); ++i)
                    scheduleScope(memoryBlock.changedClassesThisStep.validityAt(i));
                equalityPropagation.cleanUpExpressions(memoryBlock, cleanupScopes);
                memoryBlock.lbMemory.canonicalClassChanges.resetToFresh();
            }
#if PHASE13_DEEP_TIMING
            recordPhase13Detail(Phase13TimingSlot::cleanupExpressions,
                                cleanupExpressionsStarted);
            const auto dischargeContradictionStarted =
                std::chrono::steady_clock::now();
#endif

            // Step 4c: contradiction closure. One sweep over
            // intEncodedStatements, fire-once then deactivate; its
            // vacuous-truth branch closes an induction goal itself
            // (GoalProcessor::closeRecursionGoal, D-573).
            // Step 4b: the implication-gate drain first — every parked
            // implication subproof whose gate element became known reopens
            // (D-406), and every integration trial whose gate became known is
            // promoted to its parent (D-481) —, then the contradiction-scope
            // demand drain: every open negated-compound contradiction scope
            // integrates the one missing compound of a rule whose head
            // negates a fact it sees (D-560) —
            // then the watch drain: every
            // canonical contradiction / goal pair becomes the exact pair the
            // two sweeps below fire on.
            if (memoryBlock.isActive) {
                RT_SCOPE_HERE("SP_SUBPROOF_DRAIN");
                implicationSubproofs.drainSubproofArrivals(memoryBlock);
                implicationSubproofs.drainIntegrationTrials(memoryBlock);
                contradictionDemand.drainContradictionDemands(memoryBlock);
            }
            {
                RT_SCOPE_HERE("SP_WATCH_DRAIN");
                watchProcessor.materializeWatchHits(memoryBlock, coreId);
                // The OR heads built on demand enter before the exclusion
                // drain (D-597).
                if (memoryBlock.isActive) rejectedSourceReplay.drainDeferredOrEvents(memoryBlock, internalMailOut);
                if (memoryBlock.isActive) orImplicationProcessor.drainOrImplications(memoryBlock, internalMailOut);
            }
            {
                RT_SCOPE_HERE("SP_DISCHARGE_CONTRADICTION");
                goalProcessor.dischargeContradiction(memoryBlock, coreId);
                goalProcessor.dischargeVacuousImplicationScopes(memoryBlock);
            }
#if PHASE13_DEEP_TIMING
            recordPhase13Detail(Phase13TimingSlot::dischargeContradiction,
                                dischargeContradictionStarted);
            const auto dischargeToBeProvedStarted =
                std::chrono::steady_clock::now();
#endif

            // Step 5: toBeProved closure. Discharge's OR-integration /
            // NotOrScope parent-scope emissions land in `internalMailOut`.
            {
                RT_SCOPE_HERE("SP_DISCHARGE_TOBEPROVED");
                goalProcessor.dischargeToBeProved(memoryBlock, coreId, internalMailOut);
            }
#if PHASE13_DEEP_TIMING
            recordPhase13Detail(Phase13TimingSlot::dischargeToBeProved,
                                dischargeToBeProvedStarted);
            const auto dischargeContradictionScopesStarted =
                std::chrono::steady_clock::now();
#endif
            {
                RT_SCOPE_HERE("SP_DISCHARGE_CONTRA_SCOPES");
                goalProcessor.dischargeContradictionScopes(memoryBlock, internalMailOut);
            }
#if PHASE13_DEEP_TIMING
            recordPhase13Detail(
                Phase13TimingSlot::dischargeContradictionScopes,
                dischargeContradictionScopesStarted);
            const auto fillMailOutStarted =
                std::chrono::steady_clock::now();
#endif

            // Step 6: centralised mailOut population.
            {
                RT_SCOPE_HERE("SP_FILL_MAILOUT");
                blockMailProcessor.fillMailOut(memoryBlock);
            }
#if PHASE13_DEEP_TIMING
            recordPhase13Detail(Phase13TimingSlot::fillMailOut,
                                fillMailOutStarted);
            const auto changedClassesClearStarted =
                std::chrono::steady_clock::now();
#endif

            // Step 7: the SOLE clear of the per-step delta-class tracker —
            // eligible entries were consumed by this call's applyEquiClasses /
            // cleanup sweep / recursion-products closure. Entry-excluded
            // OR-scope deltas belong to subtrees wiped at this phase's end. Deltas
            // minted outside standardProcessing (LB seeding, post-join
            // barrier deposits) are cleared nowhere else and survive to the
            // next call's applyEquiClasses Pass 1.
            {
                RT_SCOPE_HERE("SP_CLEAR_CHANGED_CLASSES");
                memoryBlock.changedClassesThisStep.clear();
            }
#if PHASE13_DEEP_TIMING
            recordPhase13Detail(Phase13TimingSlot::changedClassesClear,
                                changedClassesClearStarted);
#endif
            // Step 7b: a weak step block's pending usefulness test, read once
            // this call's classes are applied; a caught or expired test queues
            // its scope for the wipe drain below
            // (D-521).
            {
                RT_SCOPE_HERE("SP_USEFULNESS_TEST");
                inductionUsefulness.settle(memoryBlock);
            }
            { RT_SCOPE_HERE("SP_SCOPE_CLEANUP");
                // Sanitize goals after class application and discharge have finished.
                // All inference loops are complete before the ordered lifecycle drains.
#if PHASE13_DEEP_TIMING
                const auto sanitizeToBeProvedStarted = std::chrono::steady_clock::now();
#endif
                {
                    RT_SCOPE_HERE("SANITIZE_TOBEPROVED");
                    equalityPropagation.sanitizeToBeProved(memoryBlock);
                }
                memoryBlock.phase3IgnoredOrRoots = nullptr;
#if PHASE13_DEEP_TIMING
                recordPhase13Detail(Phase13TimingSlot::sanitizeToBeProved,
                                   sanitizeToBeProvedStarted);
                const auto drainDisprovedGoalsStarted = std::chrono::steady_clock::now();
#endif

                // Disproved-goal cleanup: probe the inbox against this LB's MAIN goals
                // and, on a hit, erase the goal plus its integration machinery — the
                // matched scope roots land on pendingWipeScopes so the radical wipe
                // below removes the nested state in this same call.
                {
                    RT_SCOPE_HERE("DRAIN_DISPROVED_GOALS");
                    goalProcessor.drainDisprovedGoals(memoryBlock);
                }
#if PHASE13_DEEP_TIMING
                recordPhase13Detail(Phase13TimingSlot::drainDisprovedGoals,
                                   drainDisprovedGoalsStarted);
                const auto drainDeadOrBranchesStarted = std::chrono::steady_clock::now();
#endif

                // Dead _ordis_ branch retirement: a branch whose asserted disjunct is
                // refuted (staged by ordisMerge's probe) is wiped, its cohort's
                // bookkeeping shrinks, and convergence re-checks at the reduced count.
                // Runs after the disproof drain (wholesale-retired cohorts are gone
                // first) and before the wipe drain below so the branch wipes land in
                // this same call.
                {
                    RT_SCOPE_HERE("DRAIN_DEAD_OR_BRANCHES");
                    caseProcessor.drainDeadOrBranches(memoryBlock);
                }
#if PHASE13_DEEP_TIMING
                recordPhase13Detail(Phase13TimingSlot::drainDeadOrBranches,
                                   drainDeadOrBranchesStarted);
                const auto freezeResolvedOrBranchesStarted =
                    std::chrono::steady_clock::now();
#endif

                // Duplicate-or-cohort retirement: when a class merge made two OPEN
                // cohorts at one parent equi variants of each other, keep the most
                // advanced (branch-statement census, lex tie-break) and retire the
                // rest — live branches to pendingWipeScopes, scheduling rows erased,
                // frozen branches and history kept. Runs before the freeze sweep so
                // retired branches are never evaluated, and before the wipe drain so
                // the subtree wipes land in this same call.
                {
                    RT_SCOPE_HERE("RETIRE_DUPLICATE_OR_COHORTS");
                    caseProcessor.retireDuplicateOrCohorts(memoryBlock);
                }

                // Satisfied-cohort retirement: an open cohort whose live or
                // queued disjunct has come to hold at its parent scope can no
                // longer conclude anything the parent does not prove by
                // itself; it is retired with the duplicate-loser steps before
                // the freeze sweep and the release drain see it, and its
                // wipes land in the wipe drain below.
                {
                    RT_SCOPE_HERE("RETIRE_SATISFIED_OR_COHORTS");
                    caseProcessor.retireSatisfiedOrCohorts(memoryBlock);
                }

                // Refuted-goal subproofs: a live case holding the negation of
                // a goal of its enclosing implication subproof (its innermost
                // goal scope) makes that subproof unclosable; the subproof is
                // queued for the wipe drain below and its live cases leave
                // before the freeze sweep (D-549).
                {
                    RT_SCOPE_HERE("RETIRE_REFUTED_GOAL_SUBPROOFS");
                    caseProcessor.retireRefutedGoalSubproofs(memoryBlock);
                }

                // Resolved OR branches join the deferred subtree wipe when every goal of
                // their innermost goal scope is known at the branch or above. The resolved marker prevents
                // re-registration; the ordered wipe below removes runtime state after the
                // release drain. Cohort counts, booked contributions and history survive.
                {
                    RT_SCOPE_HERE("FREEZE_RESOLVED_OR_BRANCHES");
                    caseProcessor.freezeResolvedOrBranches(memoryBlock);
                }
#if PHASE13_DEEP_TIMING
                recordPhase13Detail(Phase13TimingSlot::freezeResolvedOrBranches,
                                   freezeResolvedOrBranchesStarted);
                const auto drainPendingOrReleasesStarted =
                    std::chrono::steady_clock::now();
#endif

                // Sequenced or-disintegration: release the next pending branch of every
                // cohort whose live branch resolved this burst (a toBeProved goal
                // reached under the branch, or the branch retired refuted by the drain
                // above — the retirement staging must land first so a refuted live
                // branch releases its successor in this same call).
                {
                    RT_SCOPE_HERE("DRAIN_PENDING_OR_RELEASES");
                    caseProcessor.drainPendingOrReleases(memoryBlock);
                }
#if PHASE13_DEEP_TIMING
                recordPhase13Detail(Phase13TimingSlot::drainPendingOrReleases,
                                   drainPendingOrReleasesStarted);
                const auto wipeSubtreesStarted = std::chrono::steady_clock::now();
#endif

                // In induction-step blocks, equal-input refusal covers numbered
                // implication subproofs, including goal-qualified scopes. Retire these first
                // so the ordinary settled-subproof sweep sees them as closed.
                {
                    RT_SCOPE_HERE("RETIRE_EQUAL_INPUT_IMPLICATION_SUBPROOFS");
                    implicationSubproofs.retireEqualInputImplicationSubproofs(memoryBlock);
                }

                // Settled implication subproofs: an open statement-driven or
                // nested subproof whose implication is already known at its
                // parent is wiped, and of subproofs whose implications became
                // class-equal the one with the most statements survives. Runs
                // after the OR freeze and release drains, immediately before the
                // wipe drain, so no other sweep sees a queued subproof.
                {
                    RT_SCOPE_HERE("RETIRE_SETTLED_IMPLICATION_SUBPROOFS");
                    implicationSubproofs.retireSettledImplicationSubproofs(memoryBlock);
                }

                // Duplicate OR-integration families: of two open families under
                // one parent whose ORs became class-equal after both opened, the
                // one with the most statements survives; the others' cases join
                // the wipe drain below (D-550).
                {
                    RT_SCOPE_HERE("RETIRE_DUPLICATE_OR_INTEGRATION_FAMILIES");
                    caseProcessor.retireDuplicateOrIntegrationFamilies(memoryBlock);
                }

                // Induction trees whose hypothesis fired in a leaf this burst
                // settle: the rest of each tree joins the wipe drain below,
                // and no further induction scope opens in it
                // (D-520).
                {
                    RT_SCOPE_HERE("SETTLE_INDUCTION_TREES");
                    memoryBlock.settleInductionTrees();
                }

                // An open induction scope of an unsettled tree whose admitted
                // names all equal names its root holds adds nothing the root
                // does not hold: it joins the wipe drain below
                // (D-545).
                {
                    RT_SCOPE_HERE("RETIRE_ALIAS_INDUCTION_SCOPES");
                    implicationSubproofs.retireAliasInductionScopes(memoryBlock);
                }

                // A closed or queued implication subproof takes its reductio
                // scope along; runs last so it sees every scope queued this
                // burst (D-532).
                {
                    RT_SCOPE_HERE("RETIRE_SUBPROOF_REDUCTIOS");
                    implicationSubproofs.retireSubproofReductios(memoryBlock);
                }

                // Wipe the union of closed scopes only after every absorption and class
                // loop has finished. No live levels-map or class-map view crosses this seam.
                if (!memoryBlock.pendingWipeScopes.empty()) {
                    RT_SCOPE_HERE("WIPE_SUBTREES");
                    // Quiescence (D-194): a subtree wipe eradicates state
                    // (and can leave the net statement count unchanged if it also erased what
                    // this burst added), so flag it as a mutation directly rather than relying
                    // on the count diff.
                    memoryBlock.mutatedThisBurst = true;
                    // Snapshot the cold set's ids then reset it, so wipeSubtree's own
                    // inserts into other fields run against a fresh container.
                    // Sort the IDS by their DECODED names (compareSpans == std::string
                    // byte-lex): the ColdHashSet dedups vids and the interner is
                    // injective, so the id sort is tie-free and reproduces the former
                    // decoded-string std::sort drain order exactly (I-84). The id run
                    // rides the per-slot gen-scratch byte-bump tier (I-124), reclaimed
                    // by the popTo below; the comparator's decodeView spans are
                    // consumed inside each pure-read comparison and nothing mints into
                    // the NameMap before the wipes consume ids, not spans (I-3).
                    const unsigned slot = (gl::WorkerContext::coreId() >= 0)
                        ? static_cast<unsigned>(gl::WorkerContext::coreId())
                        : genScratchArenas().slotCount() - 1;
                    ScratchArena& gArena = genScratchArenas().forSlot(slot);
                    const ArenaOffset drainMark = gArena.cursor();
                    const int32_t n = memoryBlock.pendingWipeScopes.count();
                    NameId* ids = reinterpret_cast<NameId*>(gArena.resolve(
                        gArena.alloc(n * static_cast<int32_t>(sizeof(NameId)),
                                     alignof(NameId))));
                    for (int32_t i = 1; i <= n; ++i)
                        ids[i - 1] = memoryBlock.pendingWipeScopes.decode(i);
                    memoryBlock.pendingWipeScopes.resetToFresh();
                    std::sort(ids, ids + n, [&](NameId a, NameId b) {
                        return compareSpans(memoryBlock.nameMap.decodeView(a),
                                            memoryBlock.nameMap.decodeView(b)) < 0;
                    });
                    memoryBlock.wipeSubtrees(ids, n);
                    gArena.popTo(drainMark);
                }
#if PHASE13_DEEP_TIMING
                recordPhase13Detail(Phase13TimingSlot::wipeSubtrees,
                                   wipeSubtreesStarted);
                const auto sweepAncestorKnownRowsStarted =
                    std::chrono::steady_clock::now();
#endif

                // Ancestor-known sweep (I-187): drop
                // statement-LIST rows at non-main scopes whose text a strict ancestor
                // knows — the branch-first residue the deposit-time gates cannot see
                // (NameMap has no child index). Runs LAST: after goal sanitization
                // (rewrites stabilized) and after the dead-branch and wipe drains
                // (retired scopes are gone, so the sweep never touches a scope a wipe
                // just erased). Registry rows survive as the dedup tombstones the
                // gates and re-check paths read.
                {
                    RT_SCOPE_HERE("SWEEP_ANCESTOR_KNOWN_ROWS");
                    statementMemory.sweepAncestorKnownRows(memoryBlock);
                }
#if PHASE13_DEEP_TIMING
                recordPhase13Detail(Phase13TimingSlot::sweepAncestorKnownRows,
                                   sweepAncestorKnownRowsStarted);
#endif
            } // RT_SCOPE SP_SCOPE_CLEANUP

        }

/// @brief Close `productsOfRecursionIds` under main-scope
        ///        equivalence-class membership for every class changed this
        ///        step.
        ///
        /// @details
        /// A product of recursion is a name built by eating through the
        /// induction implication's premise cursors (the two mint sites:
        /// `updateAdmissionMapRecursion`'s output substitution and
        /// `isAdmitted`'s cursor-witness admission). The marker-capture purity
        /// gate in `checkLocalEncodedMemoryStatic` stages admission demand only
        /// on requests whose iteration-bearing premise arguments are all
        /// products of recursion. A name PROVEN EQUAL to a product of
        /// recursion denotes the same value, so demand built on it is anchored
        /// by the induction progression exactly as demand built on the product
        /// itself. Without this closure a mailed foreign witness-form (e.g. a
        /// flag-5 relay product minted at an ancestor LB) can become the sole
        /// surviving instance of induction-built content and starve the purity
        /// gate of every pure instance — the Gauss fold loss
        /// (D-285).
        ///
        /// Walks `changedClassesThisStep` (the per-step delta tracker) at the
        /// single-threaded `standardProcessing` seam, immediately after the
        /// `applyEquiClasses` fixpoint: for each MAIN-scope class holding at
        /// least one member already in `productsOfRecursionIds`, every other
        /// member is minted into the set. Non-main scopes are excluded — a
        /// conditional (hypothesis / boundary) equality must not enrich the
        /// LB-global set. A class whose member joins `productsOfRecursionIds`
        /// only AFTER the class's last change is not re-visited here; both
        /// mint sites run before the first classes form in practice
        /// (I-194 documents the limitation).
        ///
        /// @param mb The LB whose changed classes are walked and whose
        ///        `overallHashMemory.productsOfRecursionIds` is enriched.
        /// @return (void)
        /// @invariant Walk order is the buffer's append order and every mint is
        ///            guarded by a `contains` probe — the cold set's insertion
        ///            order is a pure function of the step's class state
        ///            (deterministic deload bytes).
        /// @see `checkLocalEncodedMemoryStatic` — the purity-gate consumer.
inline void LogicBlockOperations::enrichProductsOfRecursionFromChangedClasses(Memory& mb) {
            if (mb.changedClassesThisStep.size() == 0) return;
            auto& por = mb.overallHashMemory.productsOfRecursionIds;
            const unsigned slot = (gl::WorkerContext::coreId() >= 0)
                ? static_cast<unsigned>(gl::WorkerContext::coreId())
                : genScratchArenas().slotCount() - 1;
            ScratchArena& arena = genScratchArenas().forSlot(slot);
            for (int32_t i = 0; i < mb.changedClassesThisStep.size(); ++i) {
                if (mb.changedClassesThisStep.validityAt(i) != NameMap::MAIN_ID)
                    continue;
                const EquivalenceClassView v =
                    mb.changedClassesThisStep.classViewAt(i, arena);
                const int32_t n = v.memberCount();
                bool hasProduct = false;
                for (int32_t m = 0; m < n; ++m) {
                    if (por.contains(v.memberId(m))) { hasProduct = true; break; }
                }
                if (!hasProduct) continue;
                for (int32_t m = 0; m < n; ++m) {
                    const NameId id = v.memberId(m);
                    if (por.contains(id)) continue;
                    por.mint(id);
                }
            }
        }
}
