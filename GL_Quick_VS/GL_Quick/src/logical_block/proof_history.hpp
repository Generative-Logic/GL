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
namespace tests { struct ProofHistoryFixture; }
/// @brief Proof History operations for the proof algorithm.
/// @details Uses explicit peer interfaces while the supplied logical blocks retain storage and proof state.
class ProofHistory {
friend struct tests::ProofHistoryFixture;
public:
/// @brief Bind inputs for proof history.
/// @details Construction stores explicit references; block storage and all invocation lifetimes remain with their owners.
/// @param parameters Borrowed ProverParameters interface or state; must outlive this component.
/// @return None; construction binds the supplied references.
ProofHistory(const ProverParameters& parameters);

public:
/// @brief Append an origin record into a per-(expression, validity)
        ///        vector; at the cap the existing records win.
        ///
        /// @details
        /// The string-form origin sink (the id-form twins are `addOriginId`).
        /// Below the `maxOrigins` cap the origin is appended if absent (set
        /// semantics within the vector, insertion order kept). At the cap
        /// nothing is written: a run is well-founded only in insertion order
        /// (a record's antecedents were recorded before it), and no tag says
        /// anything about a record's foundational value — the retired
        /// tag-based displacement removed the one non-circular record of a
        /// class-pair equality (I-216).
        ///
        /// @param map        The per-key origin vectors.
        /// @param ev         The (expression, validity) key.
        /// @param origin     The `(tag, antecedents)` record.
        /// @param maxOrigins Per-key cap (`max_origin_per_expr` or the compressor twin).
        /// @invariant Existing records are never displaced.
        /// @see addOriginId — the id-form twins with the same contract.
inline void addOrigin(std::map<ExpressionWithValidity, std::vector<std::pair<std::string, std::vector<ExpressionWithValidity>>>>& map,
            const ExpressionWithValidity& ev,
            const std::pair<std::string, std::vector<ExpressionWithValidity>>& origin,
            int maxOrigins) const;

public:
/// @brief Whether `(expr, validity)` has at least one history line
        ///        in the LB's `exprOriginMap` — the shared non-minting
        ///        probe behind every I-34-style "already documented" gate.
        ///
        /// @details Two defined misses: the pair was never interned in the
        /// origin interner (no line was ever written under it), or the key
        /// exists with an empty run. Neither mints anything (Rule 16: a
        /// write-side probe — the callers gate only WHETHER a second
        /// history line is written; no proof decision reads it).
        ///
        /// @param mb       Owning LB — origin interner + origin map.
        /// @param expr     The expression bytes.
        /// @param validity The scope name.
        /// @return `true` iff a non-empty origin run exists for the pair.
        /// @see findEqualityCiteScope, applyEquivalenceClass,
        ///      addExprToMemoryBlock — the three consumers.
inline bool originRowExists(Memory& mb, StrSpan expr, StrSpan validity);

public:
/// @brief The scope at which a justifying equality's history line
        ///        actually lives — the class scope first, then its strict
        ///        ancestors deepest-first (the D-227 walk).
        ///
        /// @details A class can inherit equality content from an ancestor
        /// class (I-33), so the `(=[member,canonical])` justifier of a
        /// rewrite under a class at scope `S` may have its body row at an
        /// ancestor of `S`; citing it at `S` would strand the chapter
        /// walker at an or boundary. Every emitter of an `equality1` line
        /// (the equi-class apply, the canonical door) resolves each
        /// justifier through this one walk. A probe that misses everywhere
        /// asserts (Rule 19) — the equality's body rows are written at
        /// arrival by `updateEquivalenceClasses` / the mirror machinery, so
        /// a miss is a producer gap, never a state to route around.
        /// Non-minting; the returned span aliases NameMap cold bytes (the
        /// class-scope argument itself, or a `strictAncestorSpans` decode)
        /// and is valid until the caller's next NameMap mint (I-3).
        ///
        /// @param mb         Owning LB — NameMap + origin map.
        /// @param equality   The justifier text `(=[member,canonical])`.
        /// @param classScope The scope of the class the pair came from.
        /// @return The cite scope: @p classScope when its row lives there,
        ///         else the deepest strict ancestor holding the row.
        /// @invariant Asserts a row exists at @p classScope or one of its
        ///            strict ancestors.
        /// @see applyEquivalenceClass, addExprToMemoryBlock — the emitters;
        ///      originRowExists — the probe.
inline StrSpan findEqualityCiteScope(Memory& mb, StrSpan equality,
                                             StrSpan classScope);

public:
/// @brief Write the `equality1` history line that derives a statement from
        ///        its spelling under other members of the same equality
        ///        classes.
        ///
        /// @details
        /// The bridge behind a subproof whose open goal a class rewrite
        /// respelled (`EqualityPropagation::sanitizeToBeProved`): the goal
        /// closes in the new spelling, and this line leads from that
        /// statement back to the subproof's own goal, which the closing row
        /// then cites (D-557). Both
        /// spellings share operator, polarity and arity. Per differing
        /// argument position the justifier `(=[source argument,result
        /// argument])` is cited, both arguments members of one class of the
        /// complete class view at @p resultScopeId (I-254), each pair once,
        /// at the scope `findEqualityCiteScope` resolves; the source is cited
        /// at @p sourceScope. The line is the row `check_equality1` accepts:
        /// source first, then the equalities. Documentation only: nothing
        /// is inserted into a statement registry (Rule 16).
        ///
        /// @param mb            Owning LB — NameMap, class views, origin map.
        /// @param result        The statement the line derives, in the
        ///                      spelling the caller cites.
        /// @param resultScopeId Scope of @p result; the class view is read
        ///                      there.
        /// @param source        The statement it is derived from.
        /// @param sourceScope   Scope of @p source: @p resultScopeId or one
        ///                      of its ancestors.
        /// @return Nothing.
        /// @invariant Asserts equal operator, polarity and arity, at least one
        ///            differing argument, every differing pair interned and
        ///            class-equal at @p resultScopeId, and the justifier
        ///            rows `findEqualityCiteScope` requires.
        /// @see GoalProcessor::dischargeToBeProved — the caller;
        ///      findEqualityCiteScope; CaseProcessor::copySharedOrCaseHistory —
        ///      the case-reuse twin.
inline void writeEqualityBridge(Memory& mb, StrSpan result, NameId resultScopeId,
                                        StrSpan source, StrSpan sourceScope);

public:
/// @brief Discharge any `toBeProved` goal matched by a freshly-
        ///        committed expression.
        ///
        /// @details
        /// The toBeProved discharge is NOT run inline inside
        /// `addExprToMemoryBlock` — doing so could recursively call
        /// `addExprToMemoryBlock` to emit an OR-integration wrapper or
        /// implication closure at parent scope, producing a tangled
        /// control flow.
        ///
        /// Instead this helper is invoked from inside
        /// `applyEquiClasses` once per expression committed to
        /// `intEncodedStatements`. Three discharge paths are handled:
        ///
        /// 1. **Induction-recursion proven** — mark proved tuples,
        ///    queue induction-LB cleanup.
        /// 2. **Non-recursion proven** — check `isProved`, reconstruct
        ///    the implication, emit via `updateGlobalSingleChapterPages`,
        ///    mark `toBeProved` erased, stage scope wipe.
        /// 3. **OR-integration / implication closure** — classify the
        ///    OR scope, then enqueue the wrapping OR or implication
        ///    expression into the same block's `sameIterationInternalMail` with
        ///    the parent scope's validity name. The next elementary
        ///    step's mail-absorb pass picks it up — a one-cycle delay
        ///    rather than a recursive `addExprToMemoryBlock` call.
        ///
        /// @param memoryBlock Owning LB. Mutated in place:
        ///        `toBeProved` (entries erased on
        ///        discharge), `sameIterationInternalMail.statements` and
        ///        `sameIterationInternalMail.exprOriginMap` (parent-scope OR /
        ///        implication closure emissions), induction-LB
        ///        cleanup queues, deferred scope wipe queue.
        /// @param expr Compiled expression string of the freshly-
        ///        committed statement that may match a `toBeProved`
        ///        goal.
        /// @param validity Validity name carried by the committed
        ///        statement. Determines which scope's `toBeProved`
        ///        entries are eligible for matching and which
        ///        ancestor scope receives any OR-closure emission.
        /// @see `applyEquiClasses` — the per-elementary-step driver
        ///      that calls this helper on every commit.
        /// @see [`docs/10_pipeline/04_prover.md`](../../docs/10_pipeline/04_prover.md).
        /// @see [`docs/20_core_concepts/03_mail_system.md`](../../docs/20_core_concepts/03_mail_system.md).
        /// @brief (Retired — kept for ABI/test compatibility only.)
        ///        Originally walked the transitive history chain of one
        ///        expression into `mailOut.exprOriginMap`. Superseded by
        ///        paired `mailOut.exprOriginMap`
        ///        writes at doc-only addOrigin sites (e.g.
        ///        `trackExpansionHistory`'s expansion conjunction).
        ///        The walk caused per-step
        ///        repeat-shipment of the same chain (mailOut clears
        ///        end-of-step; walk re-discovered each step) and bloated
        ///        chapter export.
        ///
        /// @param memoryBlock Owning LB. Unused.
        /// @param dep Expression. Unused.
        /// @param maxOrigins Cap. Unused.
inline void shipExprHistoryTransitively(Memory& memoryBlock,
                                                const ExpressionWithValidity& dep,
                                                int maxOrigins);

private:
// (No-op stub above. Original implementation retained below in
        // an `#if 0` block for reference; safe to delete after the
        // doc-only paired-write approach is verified.)
#if 0
        inline void shipExprHistoryTransitivelyRetired(Memory& memoryBlock,
                                                       const ExpressionWithValidity& dep,
                                                       int maxOrigins);

public:

#endif

        /// @brief Centralised mailOut population from
        ///        `intLocalEncodedStatementsDelta`. Called once per
        ///        standard-processing call, after
        ///        `applyEquiClasses` (and, when wired,
        ///        `dischargeToBeProved`).
        ///
        /// @details
        /// Every mailOut write driven by a local commit to
        /// `intEncodedStatements` runs from this single site. The
        /// function iterates `intLocalEncodedStatementsDelta` — the
        /// per-step delta of new local statements — and for each
        /// entry that is main-scope and `allowedForMail`, copies the
        /// statement's levels (from `intStatementLevelsMap`) into
        /// `mailOut.statements` and copies every origin line (from
        /// `exprOriginMap`) into `mailOut.exprOriginMap`. All
        /// scattered per-commit mailOut writes (flat
        /// `addExprToMemoryBlock`'s disintegration-product loop,
        /// `applyEquivalenceClass`, `addEquality` original + mirror,
        /// `addNegatedEquality` original + mirror, anchor handling)
        /// do not write mailOut directly; every site they covered
        /// pushes to `intLocalEncodedStatementsDelta`, so fillMailOut
        /// sees them via the delta.
        ///
        /// **Transitive history walk.** After shipping each delta
        /// entry's own origin lines, the function recursively walks
        /// every sub-dep referenced by those origins via
        /// `shipExprHistoryTransitively`. This is required because
        /// documentation-only expressions (notably the big
        /// conjunctions produced by `expansion` of compact forms like
        /// `NaturalNumbers[…]`) never enter `intLocalEncodedStatementsDelta`
        /// — the prover disintegrates them immediately — so they would
        /// be missed by the delta loop alone. Their history lines DO
        /// exist in the parent LB's `exprOriginMap` and are cited as
        /// `disintegration`-deps by their children. Without the
        /// transitive walk, the children ship but the source
        /// conjunction does not, and chapter export's
        /// `buildStack` asserts on the missing origin (e.g. on
        /// `__contradiction__(in2[10,10,3])` and
        /// `__contradiction__(=[2,6])`).
        ///
        /// @param memoryBlock Owning LB. Reads
        ///        `intLocalEncodedStatementsDelta`, `intStatementLevelsMap`,
        ///        `exprOriginMap`; writes `mailOut.statements` and
        ///        `mailOut.exprOriginMap`.
        /// @see `applyEquiClasses` — the upstream pass that may have
        ///      committed rewrites into `intLocalEncodedStatementsDelta`.
        /// @see `dischargeToBeProved` — the toBeProved discharge that
        ///      also iterates `intLocalEncodedStatementsDelta`; both
        ///      drive off the same delta channel.
        /// @see `shipExprHistoryTransitively` — the recursive helper
        ///      that walks documentation-only history into mailOut.
        /// @see [`docs/20_core_concepts/03_mail_system.md`](../../docs/20_core_concepts/03_mail_system.md).
        /// @brief Copy one expression's origin rows from `body.exprOriginMap`
        ///        into `mailOut.exprOriginMap` — the per-key half of the
        ///        fillMailOut origin mirror.
        ///
        /// @details Zero-copy blob peek over the cold origin map (the
        /// Codec<IdOrigin> / mail-origin blob layouts coincide), translating
        /// each record's key and dependencies from the LB's `originInterner`
        /// spans into the mailbox's private interner via `addMailOutOrigin`.
        /// A no-op when history tracking is off or the key has no rows —
        /// both defined outcomes (an unregistered relay compact may carry no
        /// rows only in a non-tracking run). Extracted verbatim from the
        /// fillMailOut delta loop so the flag-5 direct-deposit path ships the
        /// same origin mirror (D-45 paired-origin requirement at receivers).
        ///
        /// @param memoryBlock The producing LB (resident, claimed).
        /// @param origView The expression span (NameMap decodeView).
        /// @param valView The validity span.
        /// @param maxOrigins Per-key retained-line cap.
        /// @return Nothing.
        /// @see fillMailOut — both call sites; addMailOutOrigin.
        public:
        inline void copyOriginRowsToMailOut(Memory& memoryBlock,
            StrSpan origView, StrSpan valView, int maxOrigins);
private:

const ProverParameters& parameters;
};
}
namespace gl {

/// @brief Append an origin record into a per-(expression, validity)
        ///        vector; at the cap the existing records win.
        ///
        /// @details
        /// The string-form origin sink (the id-form twins are `addOriginId`).
        /// Below the `maxOrigins` cap the origin is appended if absent (set
        /// semantics within the vector, insertion order kept). At the cap
        /// nothing is written: a run is well-founded only in insertion order
        /// (a record's antecedents were recorded before it), and no tag says
        /// anything about a record's foundational value — the retired
        /// tag-based displacement removed the one non-circular record of a
        /// class-pair equality (I-216).
        ///
        /// @param map        The per-key origin vectors.
        /// @param ev         The (expression, validity) key.
        /// @param origin     The `(tag, antecedents)` record.
        /// @param maxOrigins Per-key cap (`max_origin_per_expr` or the compressor twin).
        /// @invariant Existing records are never displaced.
        /// @see addOriginId — the id-form twins with the same contract.
inline void ProofHistory::addOrigin(std::map<ExpressionWithValidity, std::vector<std::pair<std::string, std::vector<ExpressionWithValidity>>>>& map,
            const ExpressionWithValidity& ev,
            const std::pair<std::string, std::vector<ExpressionWithValidity>>& origin,
            int maxOrigins) const {
            auto& vec = map[ev];
            if (vec.size() < static_cast<size_t>(maxOrigins)) {
                if (std::find(vec.begin(), vec.end(), origin) == vec.end()) {
                    vec.push_back(origin);
                }
                return;
            }

            // At the cap the existing rows win: the newcomer is dropped, nothing
            // is displaced (I-216).
        }

/// @brief Whether `(expr, validity)` has at least one history line
        ///        in the LB's `exprOriginMap` — the shared non-minting
        ///        probe behind every I-34-style "already documented" gate.
        ///
        /// @details Two defined misses: the pair was never interned in the
        /// origin interner (no line was ever written under it), or the key
        /// exists with an empty run. Neither mints anything (Rule 16: a
        /// write-side probe — the callers gate only WHETHER a second
        /// history line is written; no proof decision reads it).
        ///
        /// @param mb       Owning LB — origin interner + origin map.
        /// @param expr     The expression bytes.
        /// @param validity The scope name.
        /// @return `true` iff a non-empty origin run exists for the pair.
        /// @see findEqualityCiteScope, applyEquivalenceClass,
        ///      addExprToMemoryBlock — the three consumers.
inline bool ProofHistory::originRowExists(Memory& mb, StrSpan expr, StrSpan validity) {
            int64_t pk = 0;
            if (!lookupOriginKey(mb.originInterner, expr, validity, pk)) return false;
            const int32_t oid = mb.exprOriginMap.lookup(pk);
            return oid != 0 && mb.exprOriginMap.runLen(oid) > 0;
        }

/// @brief The scope at which a justifying equality's history line
        ///        actually lives — the class scope first, then its strict
        ///        ancestors deepest-first (the D-227 walk).
        ///
        /// @details A class can inherit equality content from an ancestor
        /// class (I-33), so the `(=[member,canonical])` justifier of a
        /// rewrite under a class at scope `S` may have its body row at an
        /// ancestor of `S`; citing it at `S` would strand the chapter
        /// walker at an or boundary. Every emitter of an `equality1` line
        /// (the equi-class apply, the canonical door) resolves each
        /// justifier through this one walk. A probe that misses everywhere
        /// asserts (Rule 19) — the equality's body rows are written at
        /// arrival by `updateEquivalenceClasses` / the mirror machinery, so
        /// a miss is a producer gap, never a state to route around.
        /// Non-minting; the returned span aliases NameMap cold bytes (the
        /// class-scope argument itself, or a `strictAncestorSpans` decode)
        /// and is valid until the caller's next NameMap mint (I-3).
        ///
        /// @param mb         Owning LB — NameMap + origin map.
        /// @param equality   The justifier text `(=[member,canonical])`.
        /// @param classScope The scope of the class the pair came from.
        /// @return The cite scope: @p classScope when its row lives there,
        ///         else the deepest strict ancestor holding the row.
        /// @invariant Asserts a row exists at @p classScope or one of its
        ///            strict ancestors.
        /// @see applyEquivalenceClass, addExprToMemoryBlock — the emitters;
        ///      originRowExists — the probe.
inline StrSpan ProofHistory::findEqualityCiteScope(Memory& mb, StrSpan equality,
                                             StrSpan classScope) {
            if (originRowExists(mb, equality, classScope)) return classScope;
            StrSpan ancestors[ExecutionParameters::MAX_SCOPE_DEPTH];
            const int32_t ancN = mb.nameMap.strictAncestorSpans(
                classScope, ancestors, ExecutionParameters::MAX_SCOPE_DEPTH);
            for (int32_t ai = ancN - 1; ai >= 0; --ai) {
                if (originRowExists(mb, equality, ancestors[ai])) return ancestors[ai];
            }
            assert(false
                && "findEqualityCiteScope: justifying equality has no origin row at the class scope or any of its ancestors");
            return classScope;
        }

/// @brief Write the `equality1` history line that derives a statement from
        ///        its spelling under other members of the same equality
        ///        classes.
        ///
        /// @details
        /// The bridge behind a subproof whose open goal a class rewrite
        /// respelled (`EqualityPropagation::sanitizeToBeProved`): the goal
        /// closes in the new spelling, and this line leads from that
        /// statement back to the subproof's own goal, which the closing row
        /// then cites (D-557). Both
        /// spellings share operator, polarity and arity. Per differing
        /// argument position the justifier `(=[source argument,result
        /// argument])` is cited, both arguments members of one class of the
        /// complete class view at @p resultScopeId (I-254), each pair once,
        /// at the scope `findEqualityCiteScope` resolves; the source is cited
        /// at @p sourceScope. The line is the row `check_equality1` accepts:
        /// source first, then the equalities. Documentation only: nothing
        /// is inserted into a statement registry (Rule 16).
        ///
        /// @param mb            Owning LB — NameMap, class views, origin map.
        /// @param result        The statement the line derives, in the
        ///                      spelling the caller cites.
        /// @param resultScopeId Scope of @p result; the class view is read
        ///                      there.
        /// @param source        The statement it is derived from.
        /// @param sourceScope   Scope of @p source: @p resultScopeId or one
        ///                      of its ancestors.
        /// @return Nothing.
        /// @invariant Asserts equal operator, polarity and arity, at least one
        ///            differing argument, every differing pair interned and
        ///            class-equal at @p resultScopeId, and the justifier
        ///            rows `findEqualityCiteScope` requires.
        /// @see GoalProcessor::dischargeToBeProved — the caller;
        ///      findEqualityCiteScope; CaseProcessor::copySharedOrCaseHistory —
        ///      the case-reuse twin.
inline void ProofHistory::writeEqualityBridge(Memory& mb, StrSpan result, NameId resultScopeId,
                                               StrSpan source, StrSpan sourceScope) {
            StrSpan resultArgs[ExecutionParameters::MAX_ARITY];
            StrSpan sourceArgs[ExecutionParameters::MAX_ARITY];
            const int32_t count = getArgsSpans(result, resultArgs, ExecutionParameters::MAX_ARITY);
            assert(count == getArgsSpans(source, sourceArgs, ExecutionParameters::MAX_ARITY)
                && "writeEqualityBridge: the two spellings differ in arity");
            int32_t head = 0;
            while (head < result.len && result.ptr[head] != '[') ++head;
            assert(head < result.len && head < source.len
                && std::memcmp(result.ptr, source.ptr, static_cast<std::size_t>(head + 1)) == 0
                && "writeEqualityBridge: the two spellings differ in operator or polarity");

            const unsigned slot = gl::WorkerContext::coreId() >= 0
                ? static_cast<unsigned>(gl::WorkerContext::coreId()) : scratchArenas().slotCount() - 1;
            ScratchArena& arena = scratchArenas().forSlot(slot);
            ScratchScope frame(arena);
            const unsigned generalSlot = gl::WorkerContext::coreId() >= 0
                ? static_cast<unsigned>(gl::WorkerContext::coreId()) : genScratchArenas().slotCount() - 1;
            ScratchArena& general = genScratchArenas().forSlot(generalSlot);
            ScratchScope generalFrame(general);

            // Copied: the cite-scope walk decodes ancestors (I-3).
            const StrSpan scopeView = mb.nameMap.decodeView(resultScopeId);
            const StrSpan scope(ScratchString::copyFrom(arena, scopeView.ptr, scopeView.len));
            const int32_t bucket = mb.canonicalEquivalenceClassesMap.lookup(resultScopeId);
            assert(bucket != 0 && "writeEqualityBridge: no class view at the result scope");

            OriginDep dependencies[ExecutionParameters::kMaxOriginDeps];
            dependencies[0] = { source, sourceScope };
            int32_t dependencyCount = 1;
            NameId fromIds[ExecutionParameters::MAX_ARITY];
            NameId toIds[ExecutionParameters::MAX_ARITY];
            int32_t used = 0;
            for (int32_t j = 0; j < count; ++j) {
                if (equalSpans(sourceArgs[j], resultArgs[j])) continue;
                const NameId from = mb.nameMap.lookup(sourceArgs[j]);
                const NameId to = mb.nameMap.lookup(resultArgs[j]);
                assert(from != 0 && to != 0 && "writeEqualityBridge: a differing argument is not interned");
                bool repeated = false;
                for (int32_t k = 0; k < used; ++k)
                    repeated = repeated || (fromIds[k] == from && toIds[k] == to);
                if (repeated) continue;
                bool classEqual = false;
                const int32_t classCount = mb.canonicalEquivalenceClassesMap.runLen(bucket);
                for (int32_t c = 0; c < classCount && !classEqual; ++c) {
                    ScratchScope classFrame(general);
                    int32_t length = 0;
                    const char* bytes = mb.canonicalEquivalenceClassesMap.peekRecordBytes(bucket, c, length, general);
                    const EquivalenceClassView view{bytes, length};
                    classEqual = classHasMember(view, from) && classHasMember(view, to);
                }
                assert(classEqual && "writeEqualityBridge: a differing argument pair is not class-equal at the result scope");
                assert(dependencyCount < ExecutionParameters::kMaxOriginDeps
                    && "writeEqualityBridge: more justifiers than an origin row holds");
                const int32_t length = sourceArgs[j].len + resultArgs[j].len + 6;
                char* bytes = arena.allocBytes(length);
                int32_t at = 0;
                std::memcpy(bytes + at, "(=[", 3); at += 3;
                std::memcpy(bytes + at, sourceArgs[j].ptr, static_cast<std::size_t>(sourceArgs[j].len));
                at += sourceArgs[j].len;
                bytes[at++] = ',';
                std::memcpy(bytes + at, resultArgs[j].ptr, static_cast<std::size_t>(resultArgs[j].len));
                at += resultArgs[j].len;
                std::memcpy(bytes + at, "])", 2); at += 2;
                assert(at == length);
                const StrSpan equality(bytes, length);
                const StrSpan cite = findEqualityCiteScope(mb, equality, scope);
                dependencies[dependencyCount++] =
                    { equality, StrSpan(ScratchString::copyFrom(arena, cite.ptr, cite.len)) };
                fromIds[used] = from;
                toIds[used] = to;
                ++used;
            }
            assert(used > 0 && "writeEqualityBridge: the two spellings are equal");
            const int cap = parameters.compressor_mode
                ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr;
            addOriginEncoded(mb.exprOriginMap, mb.originInterner, result, scope,
                OriginTag::equality1, dependencies, dependencyCount, cap);
        }

/// @brief Discharge any `toBeProved` goal matched by a freshly-
        ///        committed expression.
        ///
        /// @details
        /// The toBeProved discharge is NOT run inline inside
        /// `addExprToMemoryBlock` — doing so could recursively call
        /// `addExprToMemoryBlock` to emit an OR-integration wrapper or
        /// implication closure at parent scope, producing a tangled
        /// control flow.
        ///
        /// Instead this helper is invoked from inside
        /// `applyEquiClasses` once per expression committed to
        /// `intEncodedStatements`. Three discharge paths are handled:
        ///
        /// 1. **Induction-recursion proven** — mark proved tuples,
        ///    queue induction-LB cleanup.
        /// 2. **Non-recursion proven** — check `isProved`, reconstruct
        ///    the implication, emit via `updateGlobalSingleChapterPages`,
        ///    mark `toBeProved` erased, stage scope wipe.
        /// 3. **OR-integration / implication closure** — classify the
        ///    OR scope, then enqueue the wrapping OR or implication
        ///    expression into the same block's `sameIterationInternalMail` with
        ///    the parent scope's validity name. The next elementary
        ///    step's mail-absorb pass picks it up — a one-cycle delay
        ///    rather than a recursive `addExprToMemoryBlock` call.
        ///
        /// @param memoryBlock Owning LB. Mutated in place:
        ///        `toBeProved` (entries erased on
        ///        discharge), `sameIterationInternalMail.statements` and
        ///        `sameIterationInternalMail.exprOriginMap` (parent-scope OR /
        ///        implication closure emissions), induction-LB
        ///        cleanup queues, deferred scope wipe queue.
        /// @param expr Compiled expression string of the freshly-
        ///        committed statement that may match a `toBeProved`
        ///        goal.
        /// @param validity Validity name carried by the committed
        ///        statement. Determines which scope's `toBeProved`
        ///        entries are eligible for matching and which
        ///        ancestor scope receives any OR-closure emission.
        /// @see `applyEquiClasses` — the per-elementary-step driver
        ///      that calls this helper on every commit.
        /// @see [`docs/10_pipeline/04_prover.md`](../../docs/10_pipeline/04_prover.md).
        /// @see [`docs/20_core_concepts/03_mail_system.md`](../../docs/20_core_concepts/03_mail_system.md).
        /// @brief (Retired — kept for ABI/test compatibility only.)
        ///        Originally walked the transitive history chain of one
        ///        expression into `mailOut.exprOriginMap`. Superseded by
        ///        paired `mailOut.exprOriginMap`
        ///        writes at doc-only addOrigin sites (e.g.
        ///        `trackExpansionHistory`'s expansion conjunction).
        ///        The walk caused per-step
        ///        repeat-shipment of the same chain (mailOut clears
        ///        end-of-step; walk re-discovered each step) and bloated
        ///        chapter export.
        ///
        /// @param memoryBlock Owning LB. Unused.
        /// @param dep Expression. Unused.
        /// @param maxOrigins Cap. Unused.
inline void ProofHistory::shipExprHistoryTransitively(Memory& memoryBlock,
                                                const ExpressionWithValidity& dep,
                                                int maxOrigins) {
            (void)memoryBlock;
            (void)dep;
            (void)maxOrigins;
            return;
        }

// (No-op stub above. Original implementation retained below in
        // an `#if 0` block for reference; safe to delete after the
        // doc-only paired-write approach is verified.)
#if 0
        inline void ProofHistory::shipExprHistoryTransitivelyRetired(Memory& memoryBlock,
                                                       const ExpressionWithValidity& dep,
                                                       int maxOrigins) {
            // Early return: history already shipped (this call or earlier
            // in the same fillMailOut sweep). Also breaks cycles.
            if (memoryBlock.mailOut.exprOriginMap.find(dep)
                != memoryBlock.mailOut.exprOriginMap.end()) {
                return;
            }
            // Statement-membership gate:
            // skip the walk when `dep` is itself a statement in this LB
            // (present in `intEncodedStatements` and/or the per-step
            // `intLocalEncodedStatementsDelta`). Statements' history reaches
            // descendants via fillMailOut's main delta loop — both when
            // they ARE in the current step's delta (shipped directly) and
            // when they were added in a prior step (already shipped then,
            // already in the receivers' exprOriginMap, addOrigin would
            // dedupe a re-ship anyway). Letting the walk recurse into
            // statement-shaped deps causes every step's `fillMailOut` to
            // re-traverse and re-ship every statement's history chain
            // (mailOut clears at end-of-step, so the dedup at line above
            // cannot help across steps). The `registered` membership bit
            // is the statement-registration record: it marks every entry
            // that passed an add-path registration door. Restricting the
            // walk to non-registered entries (documentation-only deps —
            // expansion conjunctions etc., which never pass a registration
            // door) keeps the per-step shipment bounded to what only the
            // walk can carry.
            {
                const StatementFlags* depRow = lookupStatementFlags(
                    memoryBlock.intKnownStatements, memoryBlock.nameMap,
                    dep.original, dep.validityName);
                if (depRow && depRow->registered) {
                    return;
                }
            }
            // Contract: a dep cited by a shipped origin must have its own
            // history entry in this LB's exprOriginMap. Per Rule 19, no
            // silent skip; if the entry is missing the parent's shipment
            // path is broken — that's the bug to fix, not the assert.
            const auto orIt = memoryBlock.exprOriginMap.find(dep);
            assert(orIt != memoryBlock.exprOriginMap.end()
                   && "fillMailOut transitive walk: dep has no origin in LB exprOriginMap");

            // Ship every origin line for this dep BEFORE recursing — the
            // presence check above then short-circuits any cycle back to
            // this dep.
            for (const auto& origin : orIt->second) {
                addOrigin(memoryBlock.mailOut.exprOriginMap, dep, origin, maxOrigins);
            }
            // Recurse on every sub-dep of every shipped origin.
            for (const auto& origin : orIt->second) {
                for (const auto& sub : origin.second) {
                    shipExprHistoryTransitivelyRetired(memoryBlock, sub, maxOrigins);
                }
            }
        }


#endif

        /// @brief Centralised mailOut population from
        ///        `intLocalEncodedStatementsDelta`. Called once per
        ///        standard-processing call, after
        ///        `applyEquiClasses` (and, when wired,
        ///        `dischargeToBeProved`).
        ///
        /// @details
        /// Every mailOut write driven by a local commit to
        /// `intEncodedStatements` runs from this single site. The
        /// function iterates `intLocalEncodedStatementsDelta` — the
        /// per-step delta of new local statements — and for each
        /// entry that is main-scope and `allowedForMail`, copies the
        /// statement's levels (from `intStatementLevelsMap`) into
        /// `mailOut.statements` and copies every origin line (from
        /// `exprOriginMap`) into `mailOut.exprOriginMap`. All
        /// scattered per-commit mailOut writes (flat
        /// `addExprToMemoryBlock`'s disintegration-product loop,
        /// `applyEquivalenceClass`, `addEquality` original + mirror,
        /// `addNegatedEquality` original + mirror, anchor handling)
        /// do not write mailOut directly; every site they covered
        /// pushes to `intLocalEncodedStatementsDelta`, so fillMailOut
        /// sees them via the delta.
        ///
        /// **Transitive history walk.** After shipping each delta
        /// entry's own origin lines, the function recursively walks
        /// every sub-dep referenced by those origins via
        /// `shipExprHistoryTransitively`. This is required because
        /// documentation-only expressions (notably the big
        /// conjunctions produced by `expansion` of compact forms like
        /// `NaturalNumbers[…]`) never enter `intLocalEncodedStatementsDelta`
        /// — the prover disintegrates them immediately — so they would
        /// be missed by the delta loop alone. Their history lines DO
        /// exist in the parent LB's `exprOriginMap` and are cited as
        /// `disintegration`-deps by their children. Without the
        /// transitive walk, the children ship but the source
        /// conjunction does not, and chapter export's
        /// `buildStack` asserts on the missing origin (e.g. on
        /// `__contradiction__(in2[10,10,3])` and
        /// `__contradiction__(=[2,6])`).
        ///
        /// @param memoryBlock Owning LB. Reads
        ///        `intLocalEncodedStatementsDelta`, `intStatementLevelsMap`,
        ///        `exprOriginMap`; writes `mailOut.statements` and
        ///        `mailOut.exprOriginMap`.
        /// @see `applyEquiClasses` — the upstream pass that may have
        ///      committed rewrites into `intLocalEncodedStatementsDelta`.
        /// @see `dischargeToBeProved` — the toBeProved discharge that
        ///      also iterates `intLocalEncodedStatementsDelta`; both
        ///      drive off the same delta channel.
        /// @see `shipExprHistoryTransitively` — the recursive helper
        ///      that walks documentation-only history into mailOut.
        /// @see [`docs/20_core_concepts/03_mail_system.md`](../../docs/20_core_concepts/03_mail_system.md).
        /// @brief Copy one expression's origin rows from `body.exprOriginMap`
        ///        into `mailOut.exprOriginMap` — the per-key half of the
        ///        fillMailOut origin mirror.
        ///
        /// @details Zero-copy blob peek over the cold origin map (the
        /// Codec<IdOrigin> / mail-origin blob layouts coincide), translating
        /// each record's key and dependencies from the LB's `originInterner`
        /// spans into the mailbox's private interner via `addMailOutOrigin`.
        /// A no-op when history tracking is off or the key has no rows —
        /// both defined outcomes (an unregistered relay compact may carry no
        /// rows only in a non-tracking run). Extracted verbatim from the
        /// fillMailOut delta loop so the flag-5 direct-deposit path ships the
        /// same origin mirror (D-45 paired-origin requirement at receivers).
        ///
        /// @param memoryBlock The producing LB (resident, claimed).
        /// @param origView The expression span (NameMap decodeView).
        /// @param valView The validity span.
        /// @param maxOrigins Per-key retained-line cap.
        /// @return Nothing.
        /// @see fillMailOut — both call sites; addMailOutOrigin.
        inline void ProofHistory::copyOriginRowsToMailOut(Memory& memoryBlock,
            StrSpan origView, StrSpan valView, int maxOrigins) {
            if (!parameters.trackHistory) return;
            int64_t pkEv = 0;
            if (!lookupOriginKey(memoryBlock.originInterner,
                    origView, valView, pkEv)) {
                return;
            }
            const int32_t oid = memoryBlock.exprOriginMap.lookup(pkEv);
            if (oid == 0) return;
            // Translate each source-origin record into mailOut's
            // private interner. The source and destination id
            // spaces are deliberately independent.
            // Zero-copy blob peek in place of the owned
            // std::vector<IdOrigin> decode (recordAt returns a heap
            // std::vector<int64_t>): Codec<IdOrigin> and the
            // mail-origin blob share the uint8 tag / int32 depN /
            // depN x int64 layout, so a peeked exprOriginMap record
            // IS a byte-valid mail-origin blob. Records in [0,n)
            // order == recordsAt order, same key/tag/deps, so the
            // mailOut copies are byte-identical (I-138). Deps are
            // unaligned in the blob -> read via mailOriginDepAt
            // (memcpy), never a raw int64_t* cast.
            const int32_t n = memoryBlock.exprOriginMap.runLen(oid);
            char blobBuf[ExecutionParameters::kMaxOriginBlobBytes];
            const unsigned decodeSlot = gl::WorkerContext::coreId() >= 0 ? static_cast<unsigned>(gl::WorkerContext::coreId()) : scratchArenas().slotCount() - 1;
            ScratchArena& decodeArena = scratchArenas().forSlot(decodeSlot);
            for (int32_t k = 0; k < n; ++k) {
                ScratchScope decodeScope(decodeArena);
                int32_t bl = 0;
                const char* bp =
                    memoryBlock.exprOriginMap.peekRecordBytes(
                        oid, k, bl, blobBuf,
                        ExecutionParameters::kMaxOriginBlobBytes);
                const MailOriginBlobView v = viewMailOriginBlob(bp, bl);
                assert(v.depN <= ExecutionParameters::kMaxOriginDeps);
                OriginDep deps[ExecutionParameters::kMaxOriginDeps];
                for (int32_t d = 0; d < v.depN; ++d) {
                    StrSpan depExpr, depValidity;
                    decodeOriginKeyCopy(mailOriginDepAt(v, d),
                        memoryBlock.originInterner, depExpr, depValidity, decodeArena);
                    deps[d] = { depExpr, depValidity };
                }
                memoryBlock.addMailOutOrigin(origView, valView,
                    static_cast<OriginTag>(v.tag), deps, v.depN, maxOrigins);
            }
        }
}

namespace gl {
/// @brief Bind inputs for proof history.
/// @details Construction stores explicit references; block storage and all invocation lifetimes remain with their owners.
/// @param parameters Borrowed ProverParameters interface or state; must outlive this component.
/// @return None; construction binds the supplied references.
inline ProofHistory::ProofHistory(const ProverParameters& parameters) : parameters(parameters) {}
} // namespace gl
