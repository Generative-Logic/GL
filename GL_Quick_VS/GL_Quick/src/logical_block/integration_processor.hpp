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
#include "../logical_block/or_introduction_rules.hpp"
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
class CompactRuleProcessor;
class DisintegrationProcessor;
class HashRuleInstaller;
class ImplicationSubproofs;
class RejectedSourceReplay;
class StatementMemory;
class UnaryIntegrationProcessor;
namespace tests { struct IntegrationProcessorFixture; }
/// @brief Integration Processor operations for the proof algorithm.
/// @details Uses explicit peer interfaces while the supplied logical blocks retain storage and proof state.
class IntegrationProcessor {
friend struct tests::IntegrationProcessorFixture;
public:
/// @brief Bind inputs for integration processor.
/// @details Construction stores explicit references; block storage and all invocation lifetimes remain with their owners.
/// @param admissionPolicy Borrowed AdmissionPolicy interface or state; must outlive this component.
/// @param canonicalTemplates Borrowed CanonicalTemplateMatcher interface or state; must outlive this component.
/// @param canonicalizer Borrowed Canonicalizer interface or state; must outlive this component.
/// @param definitions Borrowed CompiledDefinitionRegistry interface or state; must outlive this component.
/// @param integrationBuilder Borrowed IntegrationInstructionBuilder interface or state; must outlive this component.
/// @param parameters Borrowed ProverParameters interface or state; must outlive this component.
/// @param scopeTypes Borrowed ScopeTypeResolver interface or state; must outlive this component.
/// @param admissionMemory Borrowed AdmissionMemory interface or state; must outlive this component.
/// @param disintegrationProcessor Borrowed DisintegrationProcessor interface or state; must outlive this component.
/// @param hashRuleInstaller Borrowed HashRuleInstaller interface or state; must outlive this component.
/// @param implicationSubproofs Borrowed ImplicationSubproofs interface or state; must outlive this component.
/// @param rejectedSourceReplay Borrowed RejectedSourceReplay interface or state; must outlive this component.
/// @param statementMemory Borrowed StatementMemory interface or state; must outlive this component.
/// @param unaryIntegrationProcessor Borrowed UnaryIntegrationProcessor interface or state; must outlive this component.
/// @param compactRuleProcessor Borrowed CompactRuleProcessor interface or state; must outlive this component.
/// @return None; construction binds the supplied references.
IntegrationProcessor(AdmissionPolicy& admissionPolicy, CanonicalTemplateMatcher& canonicalTemplates, Canonicalizer& canonicalizer, const CompiledDefinitionRegistry& definitions, IntegrationInstructionBuilder& integrationBuilder, const ProverParameters& parameters, ScopeTypeResolver& scopeTypes, AdmissionMemory& admissionMemory, DisintegrationProcessor& disintegrationProcessor, HashRuleInstaller& hashRuleInstaller, ImplicationSubproofs& implicationSubproofs, RejectedSourceReplay& rejectedSourceReplay, StatementMemory& statementMemory, UnaryIntegrationProcessor& unaryIntegrationProcessor, CompactRuleProcessor& compactRuleProcessor);

public:
/// @brief Run CNFE for every open goal over the rule chains no earlier
        ///        rescan visited.
        ///
        /// @details
        /// Every goal has been visited against chains 1..goalScanChainMark,
        /// so each goal is visited from the first chain above the mark; the
        /// mark is raised to the chain count seen at entry when the walk ends.
        /// Without new chains no goal is visited, but the unary-demand
        /// transfer runs once when goals exist. Goals are walked in the sorted
        /// order of their decoded statement and scope (I-87). Called by the
        /// door after it installs implications and by the unary composition
        /// drain after it installs composed rules.
        ///
        /// @param memoryBlock The owning LB.
        /// @return Nothing.
        /// @invariant Zero heap; the chain map never shrinks during the walk
        ///            (I-60).
        /// @see checkNecessityForEquality, HashMemory::goalScanChainMark,
        ///      drainUnaryComposition.
void rescanGoalsAgainstNewChains(Memory& memoryBlock);
private:
/// @brief Fill @p out with the `originals` rule-registry ids in ID ORDER
        ///        (`1..count` — the registry's deterministic insertion order).
        ///
        /// @details
        /// The former decoded-chain lexicographic `std::sort` here is RETIRED
        /// (maintainer directive 2026-09-01): the decoded-lex order existed only
        /// to reproduce the pre-statification heap snapshot's byte order, and it
        /// cost 35% of all phase-1/3 runtime on the FTA shortcut (measured
        /// 107 worker-seconds, `CNFE_SORT_CHAINS`). Ids are minted in the
        /// machine's own deterministic sequence, so id order is byte-stable
        /// across runs and hosts — deterministic, merely a DIFFERENT observable
        /// order than the retired one. Downstream mint order (the consumers'
        /// `prepareIntegration` walks) changed with it, gated by the verifier
        /// and the theorems.txt comparison.
        ///
        /// @param mb  The memory block (reads `overallHashMemory.originals`).
        /// @param out Caller-owned buffer (sized to `originals.count()` on the
        ///            byte-bump tier at the production consumers) receiving the ids.
        /// @param cap Capacity of @p out (== `originals.count()` at the consumers);
        ///            a `count > cap` overrun is a loud Rule-19 assert.
        /// @return The chain count written into `out[0..count)`.
        /// @invariant `out[k] == k + 1` — the deterministic id order.
        /// @see prepareIntegrationCore2 (Case B), the one consumer;
        ///      checkNecessityForEquality walks the derived
        ///      `OriginalsHeadIndex` instead.
int32_t sortOriginalChainIndex(Memory& mb, int32_t* out, int32_t cap);

private:
/// @brief The compiled entity whose integration set a Case B table entity
        ///        takes (D-503).
        ///
        /// @details
        /// A registry entity by its core name; a set- or relation-typed `=`
        /// (category `and`) is the extensionality entity of its type label
        /// (I-267), resolved as `prepareIntegration` resolves the goal's
        /// (`ScopeTypeResolver::resolveEqualityGoalLabel`). Every such entity
        /// carries its integration set from its creation — asserted
        /// non-empty, the one-level rule first.
        ///
        /// @param signature      The entity's `u_`-prefixed signature.
        /// @param category       Its category (`and` / `existence`).
        /// @param cleanSignature The signature `u_`-stripped.
        /// @param mb             The owning LB.
        /// @param validityName   The integration scope.
        /// @return The entity (registry storage, stable through the call).
        /// @see prepareIntegrationCore2 (Case B), IntegrationRuleCompiler.
const LogicalEntity& integrationEntityOf(StrSpan signature, StrSpan category, StrSpan cleanSignature,
            Memory& mb, StrSpan validityName);

private:
/// @brief Whether a Case B existence carries its witness fixed, and which.
        ///
        /// @details
        /// The table's elements are the entity's compiled elements instantiated
        /// in order; the compiled elements name the witness by a placeholder
        /// (the one argument that is not a `u_` token). The table element at
        /// the placeholder's first position holds a `pi_` name while the
        /// witness is bound and a `u_`-prefixed name once a marker replay
        /// fixed it; nothing else stands there (asserted).
        ///
        /// @param entity   The compiled existence.
        /// @param elements The table entity's elements.
        /// @param elemN    Their number (asserted equal to the entity's).
        /// @param witness  Out: the fixed witness (only when `true`).
        /// @return `true` when the witness is fixed (the occupied form).
        /// @see prepareIntegrationCore2 (Case B).
static bool occupiedWitness(const LogicalEntity& entity, const StrSpan* elements, int32_t elemN,
            StrSpan& witness);

private:
/// @brief Whether an existence part of an instruction has its fixed witness
        ///        among its own arguments
        ///        (D-590).
        ///
        /// @details
        /// The canonical rewrite at the integration scope gives every name its
        /// class representative, so a witness that a marker fill fixed and an
        /// argument of the same existence can become one name: the
        /// partial-product witness of a fold and the sequence it folds, for
        /// example (`and0[i0,x,x,…]`). That existence could only be
        /// introduced with its witness doubling as its argument; the
        /// integration went the wrong way, and `prepareIntegrationCore2`
        /// stops the continuation instead of building it. Only existence
        /// parts whose signature arguments are all `u_` are read, the parts
        /// Case B would introduce; their witness slot is found by position
        /// (`occupiedWitness`), and a bound (`pi_`) witness never stands among
        /// the arguments.
        ///
        /// @param instruction The rewritten instruction, canonical at the scope.
        /// @return True iff some such existence part has a fixed witness equal
        ///         to one of its signature arguments.
        /// @invariant Pure: reads the definition registry only.
        /// @see prepareIntegrationCore2, occupiedWitness.
bool witnessDoublesAsArgument(const WorkInstruction& instruction) const;

private:
/// @brief Install the root family of a root integration goal from its
        ///        compiled integration set: the flat variants, the
        ///        introductions of their OR premises and the flat rules of
        ///        those premises' existence disjuncts
        ///        (D-503, D-352, D-492).
        ///
        /// @details
        /// The addon to Case B for the entity that IS the expression handed to
        /// `prepareIntegration` (its `u_`-stripped signature equals the
        /// instruction's marked goal). Case B installs the set's one-level
        /// rule; this walks the remaining records in their compiled order —
        /// the order the rules were formerly built in — and installs each
        /// through `CompactRuleProcessor::installIntegrationCompact`, the
        /// owner tokens filled with the root's arguments:
        /// - `Flat`: tag `reformulation for integration flattened`, citing
        ///   `<root>_integration_goal`, install kind `IntegrationFlat` (its
        ///   near-miss demand variants report its one missing compound
        ///   premise to the goal-driven rule search,
        ///   D-574);
        /// - `OrIntroduction`: tag `expansion`, citing the filled compact of
        ///   the rule it serves (`cites`); its disjunct (found among the OR
        ///   premise's `forEachOrIntroductionRule` disjuncts by the expanded
        ///   premises) is then offered to
        ///   `AdmissionMemory::updateAdmissionMapIntegrationDouble`, installed
        ///   or not;
        /// - `DisjunctFlat`: tag `reformulation for integration flattened`,
        ///   citing `<disjunct>_integration_goal`, the disjunct being the
        ///   expanded rule's head.
        /// A compact already expanded at the scope notes the reach of its
        /// rule's owner instead (once per scope).
        ///
        /// @param entity         The root's compiled entity.
        /// @param owners         The root's arguments (`u_`-prefixed), the owner
        ///                       tokens' values.
        /// @param ownerN         Their number.
        /// @param oneLevel       The root's filled one-level compact, `u_`-stripped.
        /// @param cleanSignature The root signature `u_`-stripped.
        /// @param mb             The owning LB (single-threaded seam).
        /// @param validityName   The install scope.
        /// @param coreArena      The caller's string-tier scratch arena.
        /// @return Nothing.
        /// @invariant 0% heap; every filled compact rides @p coreArena (the
        ///            caller's scope holds them through the installs).
        /// @see prepareIntegrationCore2 (Case B), installIntegrationCompact,
        ///      forEachOrIntroductionRule.
void installRootIntegrationFamily(const LogicalEntity& entity, const StrSpan* owners, int32_t ownerN,
            StrSpan oneLevel, StrSpan cleanSignature, Memory& mb, StrSpan validityName,
            ScratchArena& coreArena);

public:
/// @brief Prepare an integration instruction's entities at a scope: open
        ///        implication subproofs (Case A), split ORs (Case OR), install
        ///        the compacts and admission keys of existences and ands
        ///        (Case B) and file the marker keys (Case C).
        ///
        /// @details
        /// Every entity is first rewritten with @p replacementMap. An entity
        /// already prepared counts as done: Cases A, OR and B skip it when its
        /// gate template is in `integrationPrepared`, and mint the template
        /// otherwise. A fresh preparation (from `prepareIntegration`) reads the
        /// gate at @p validityName only. A @p continuation, the replay of a
        /// stored instruction after its marker was filled (the commit of
        /// `AdmissionMemory::isAdmittedIntegration`,
        /// `AdmissionMemory::updateAdmissionMapIntegration`), reads it at
        /// @p validityName and at every strict ancestor
        /// (`CanonicalTemplateMatcher::canonicalTemplatePresentAtOrAbove`):
        /// an instruction owned at an ancestor continues below it as at its
        /// owner, whose preparation, the root goal's included, holds there
        /// (D-572).
        ///
        /// @param instructions The instruction (entities, marked goal, path
        ///                     list).
        /// @param replacementMap The rewrite pairs applied to every entity.
        /// @param replacementN Their number.
        /// @param mb           The owning LB.
        /// @param validityName The preparation scope.
        /// @param rootGoal     The goal driving a fresh preparation (embedded
        ///                     into scope payloads minted at `main`), or empty.
        /// @param continuation Whether this replays a stored instruction after
        ///                     a marker fill.
        /// @param splitSets    The forbidden sets of the copies of a fresh
        ///                     preparation split by premise, one per identical
        ///                     goal (null when @p splitN is 0): Case C files
        ///                     every instruction once per set
        ///                     (D-579).
        ///                     Without them each filed copy keeps the sets of
        ///                     @p instructions.
        /// @param splitN       Number of @p splitSets (0; at least 2, or 1 for a twin premise's goal, D-589).
        /// @return Nothing.
        /// @invariant Nothing is prepared at or below a restricted scope
        ///            (an integration trial or the usefulness scope).
        /// @see prepareIntegration, AdmissionMemory::isAdmittedIntegration,
        ///      AdmissionMemory::updateAdmissionMapIntegration.
//#pragma optimize("", off)
void prepareIntegrationCore2(WorkInstruction& instructions,
            const StrReplacement* replacementMap, int32_t replacementN,
            Memory& mb,
            StrSpan validityName,
            StrSpan rootGoal,
            bool continuation = false,
            const IntegrationForbiddenSets* splitSets = nullptr,
            int32_t splitN = 0);

private:
//#pragma optimize("", off)

        /// @brief Sorted-run membership probe — `binary_search` under
        ///        `compareSpans` over a sorted-unique `StrSpan` run.
        ///
        /// @details
        /// The span twin of `std::set<std::string>::find != end()`:
        /// `std::set` iterates ascending byte-lex, which is exactly
        /// `compareSpans` order, so membership in the sorted run is
        /// byte-identical to membership in the former set. Used by the
        /// `prepareIntegration` span core for its three unchangeable-args
        /// membership tests.
        ///
        /// @pre `[sorted, sorted + n)` sorted ascending strictly-unique under
        ///      `compareSpans`.
        /// @param sorted The sorted-unique run.
        /// @param n      The run length (`>= 0`).
        /// @param s      The probe bytes.
        /// @return `true` iff `s` is in the run.
        /// @see prepareIntegration(StrSpan, const StrSpan*, int32_t, Memory&, StrSpan)
        ///      — the consumer; compareSpans.
static bool spanSetContains(const StrSpan* sorted, int32_t n,
            const StrSpan& s);

public:
/// @brief Whether a premise of the block already settles a marked
///        integration goal whose marked argument is uniquely definable
///        (D-567).
///
/// @details
/// The goal is @p expression with its one changeable argument, an argument
/// outside @p unchangeableArgsSorted, as the marker. It is settled when the
/// marked position is listed in the core's `uniquely_definable` and an
/// `exprKey` on the block's `parentMemory` chain, the block's own included,
/// is a version of the goal: the same core, positive like the goal, and at
/// every other position an argument equal to the goal's by equivalence class
/// at the goal's scope. Integrating such a goal could only rebuild the value
/// that premise already names.
///
/// Not settled: a negated goal, a goal with no or with more than one
/// changeable position, a core without a configuration (a compiled-only
/// definition such as an `existence<N>`), an untagged marked position. A
/// premise key that is not one positive expression (the root's empty key, a
/// negated or suffixed block key) is no version. An unregistered scope
/// compares at `main`, whose classes hold in every scope.
///
/// @param expression             The expression `prepareIntegration`
///                               received (span over a stable buffer).
/// @param unchangeableArgsSorted Its unchangeable arguments as a sorted-unique
///                               span run.
/// @param unchangeableArgN       The run length.
/// @param mb                     The owning block.
/// @param validityName           The goal's scope.
/// @return True iff a premise version settles the goal.
/// @invariant Allocates nothing; mints nothing beyond the canonical view
///            `Canonicalizer::initializeCanonicalScope` builds for a scope
///            seen for the first time.
/// @see prepareIntegration, CanonicalTemplateMatcher::canonicalArgumentAtScope.
bool premiseVersionSettlesMarkedGoal(StrSpan expression,
            const StrSpan* unchangeableArgsSorted, int32_t unchangeableArgN,
            Memory& mb, StrSpan validityName);

public:
/// @brief Whether a statement known at the firing scope or an ancestor
///        already settles a missing premise whose open argument is uniquely
///        definable (D-568).
///
/// @details
/// The premise is @p expression, the missing premise a marker rule's firing
/// queued for integration (`AdmissionMemory::drainDeferredIntegrationPreps`),
/// with its one open argument, the argument outside @p unchangeableArgsSorted,
/// as the marker. It is settled when the open position is listed in the
/// core's `uniquely_definable` and a statement known at the scope or at an
/// ancestor of it is a version of the premise: the same core, positive like
/// the premise, and at every other position an argument equal to the
/// premise's by equivalence class at the scope. That statement already names
/// the one object the open position denotes, so integrating the premise could
/// only rebuild it.
///
/// Not settled: a negated premise, a premise with no or with more than one
/// open position, a core without a configuration, an untagged open position,
/// a version known only at a descendant or in another branch of the scope
/// tree.
///
/// @param expression             The queued premise (span over a stable
///                               buffer).
/// @param unchangeableArgsSorted Its fixed arguments as a sorted-unique span
///                               run.
/// @param unchangeableArgN       The run length.
/// @param mb                     The owning block.
/// @param validityName           The firing scope, registered in the block's
///                               name map.
/// @return True iff a known version settles the premise.
/// @invariant Mints nothing into the name map; synchronizes only the derived
///            statement index (`canonicalUses.statementMatches`) and the
///            canonical view `Canonicalizer::initializeCanonicalScope` builds
///            for a scope seen for the first time.
/// @see premiseVersionSettlesMarkedGoal — the premise-chain twin;
///      AdmissionMemory::drainDeferredIntegrationPreps — the caller.
bool knownVersionSettlesMarkedPremise(StrSpan expression,
            const StrSpan* unchangeableArgsSorted, int32_t unchangeableArgN,
            Memory& mb, StrSpan validityName);

public:
/// @brief Integration-template preparation from spans — the span-front
        ///        core of `prepareIntegration`.
        ///
        /// @details
        /// The full gated stage, in order: (1) marker-arg scan; (2)
        /// `integrationPrepared` double gate (non-minting probes at the scope
        /// AND at `"main"`) — every early return is copy-free; (2b) the
        /// premise-version gate (`premiseVersionSettlesMarkedGoal`,
        /// D-567); (3) the
        /// compiled-definition category lookup (vestigial empty-bodied `if`);
        /// (4) `disintegrateExprHypothetically`; (5) `integrationStartIntMap`
        /// set-or-insert (span `templateInterner.encode`); (6) the
        /// changeable→`"marker"` rename + `integrationPreparedMarker` double
        /// gate + marker-key mint; (7) the `tempKept <= 1` assert; (8) the
        /// unchangeable/changeable split; (9) `prepareIntegrationCore`; (10)
        /// `prepareIntegrationCore2`.
        ///
        /// 0% heap: every consumer is span-native, so NO expression / scope
        /// `std::string` is materialized — `extractExpressionUniversalSpan`,
        /// `disintegrateExprHypothetically`, `prepareIntegrationCore`'s goal, and
        /// `prepareIntegrationCore2`'s validityName / original all take spans.
        /// The two working strings (`replacedExpr`, `expressionWithUnchangeables`)
        /// are FRESH `ScratchString`s on the string-scratch tier (the byte-exact
        /// `replaceKeysScratch` twin of `replaceKeysToString`), held under their
        /// scopes across the probe/mint and the Core / Core2 calls.
        /// Unchangeable-args membership is `spanSetContains` over the sorted
        /// run — byte-identical to the former `std::set` lookups.
        ///
        /// @pre `[unchangeableArgsSorted, unchangeableArgsSorted +
        ///      unchangeableArgN)` sorted ascending strictly-unique under
        ///      `compareSpans` (== the former `std::set<std::string>`
        ///      iteration order), spans over buffers stable across the call.
        /// @param expression            The expression to prepare (span over a
        ///                              stable buffer).
        /// @param unchangeableArgsSorted The unchangeable args as a
        ///                              sorted-unique span run.
        /// @param unchangeableArgN      The run length.
        /// @param mb                    The owning LB.
        /// @param validityName          The scope (span over a stable buffer).
        /// @param rootGoal              The `toBeProved` goal driving this
        ///                              preparation (verbatim; span over a
        ///                              stable buffer), or empty for
        ///                              statement-driven preparations
        ///                              (admission / marker replays, the
        ///                              load-time prefill pass). Forwarded to
        ///                              `prepareIntegrationCore2`, which embeds
        ///                              it in every scope payload minted on
        ///                              MAIN (`<goal>_subproof_<bare>`) for
        ///                              disproof cleanup.
        /// @param typedEqualityGoal     True only at the goal door, where
        ///                              @p expression is the `toBeProved` goal
        ///                              itself: a positive `=` goal is then typed
        ///                              on the spot (`resolveEqualityGoalLabel`)
        ///                              and a set- or relation-typed one
        ///                              integrates by extensionality
        ///                              (I-267). Premise
        ///                              preparations (CNFE, admission, replays,
        ///                              prefill) carry rule-local names and stay
        ///                              atomic.
        /// @param twoChangeables        True only for the weak induction
        ///                              hypothesis's two-marker start: the
        ///                              one-changeable gate admits a template
        ///                              with two changeable arguments for this
        ///                              preparation
        ///                              (D-586).
        /// @param splitSets             The forbidden sets of a goal that one
        ///                              weak hypothesis install prepares
        ///                              @p splitN times, one per premise
        ///                              (null when @p splitN is 0); forwarded to
        ///                              `prepareIntegrationCore2`, which files
        ///                              every instruction once per set
        ///                              (D-579).
        /// @param splitN                Number of @p splitSets (0; at least 2, or 1 for a twin premise's goal, D-589).
        /// @invariant Mint-sequence-preserving vs the retired string body: per
        ///            call the interner touches are [gates: none] →
        ///            `disintegrateExprHypothetically`'s interior mints →
        ///            `templateInterner.encode(expression)` → [marker probes:
        ///            none] → `mintTemplateKey(replacedExpr, validityName)`
        ///            (`templateInterner` then `nameMap`) →
        ///            `prepareIntegrationCore` / Core2 interior mints — every
        ///            statement in the retired body's order, only argument
        ///            TYPES at existing calls changed.
        /// @see prepareIntegrationCore2; drainDeferredIntegrationPreps — the
        ///      sealed-view caller.
void prepareIntegration(StrSpan expression,
            const StrSpan* unchangeableArgsSorted, int32_t unchangeableArgN,
            Memory& mb,
            StrSpan validityName,
            StrSpan rootGoal,
            bool typedEqualityGoal = false,
            bool twoChangeables = false,
            const IntegrationForbiddenSets* splitSets = nullptr,
            int32_t splitN = 0);

/// @brief Whether `prepareIntegration` proves @p expression by a reductio
///        (the negated-compound intercept).
///
/// @details
/// True for a negated compact `!(X[args])` whose core compiles to an
/// entity other than `atomic` and `or`: its integration assumes the
/// positive compact in a `contradiction_` scope
/// (`ImplicationSubproofs::prepareNegatedCompoundContradiction`). A
/// positive form, an atomic core, an or-bodied core (all-branch refutation
/// is not this route) and an uncompiled core answer false.
///
/// @param expression Expression text (span over a stable buffer).
/// @return True when the intercept takes @p expression.
/// @see prepareIntegration, prepareHypothesisPremise.
bool negatedCompoundTakesReductio(StrSpan expression) const;

/// @brief Prepare one fully bound compound premise of an induction
///        hypothesis; the reductio of a negated premise waits for the
///        hypothesis's positive premises.
///
/// @details
/// The two routes that prepare a fully bound premise of a recursion-owned
/// rule (the install of `AdmissionMemory::updateAdmissionMap` and the
/// re-keyed chain of `AdmissionMemory::updateAdmissionMapRecursion`, I-294)
/// call this instead of `prepareIntegration`. A premise the intercept proves
/// by a reductio (`negatedCompoundTakesReductio`) at a scope that is no
/// integration trial is gated: the gates are the chain's positive elements
/// (no leading `!`) whose every argument is in @p remainingArgs, the
/// premise itself excluded; while one of them is not known at the scope or
/// an ancestor, the reductio parks under it
/// (`ImplicationSubproofs::deferPremiseReductio`) and opens when the
/// arrival drain finds every gate known. A hypothesis fires only when all
/// its premises hold, so a reductio opened before its positive premises
/// hold derives for a rule that cannot fire yet
/// (I-326). Every other premise,
/// and a reductio whose gates are all known, takes `prepareIntegration` as
/// before. Positive premises with an unbound argument are no gates: no
/// single instance of them can be known.
///
/// @param element       The fully bound premise (span over a stable buffer).
/// @param chain         The rule's premise chain, @p element among them.
/// @param chainN        Its length.
/// @param remainingArgs The chain's bound arguments, sorted strictly
///                      unique under `compareSpans`.
/// @param remN          Their number.
/// @param mb            The owning LB.
/// @param validityName  The preparing scope.
/// @return Nothing.
/// @see prepareIntegration, ImplicationSubproofs::deferPremiseReductio.
void prepareHypothesisPremise(StrSpan element, const StrSpan* chain, int32_t chainN,
            const StrSpan* remainingArgs, int32_t remN, Memory& mb, StrSpan validityName);

public:
//#pragma optimize("", off)

/// @brief Prepare the integration of every premise of every rule chain whose
///        head can establish a goal up to one or two `(1)`-typed constants
///        (CNFE), and add the demanded variable copies.
///
/// @details
/// Candidates are the chains of the masked-key family of `OriginalsHeadIndex`
/// whose key equals the goal's key masked at one slot, or at two slots when
/// at least one other slot remains; the key carries the sign, so a rule
/// concluding the goal's negation is never a candidate. Each constant is
/// replaced in every premise by the goal's argument at its slot, and every
/// premise whose arguments are then all fixed is prepared for integration. A
/// premise that still holds a bound argument the head does not fix is neither
/// prepared nor counted for a copy verdict: removing the `u_` prefixes would
/// read that variable as the block's same-spelled name
/// (D-534). A premise that reads `!(=[Y,Y])` after
/// the replacement is skipped the same way: it can never hold, so it is
/// neither prepared nor counted for a copy verdict
/// (D-535). A constant demands a variable copy
/// `(=[Y,Y_copy])` of its goal argument Y when the replacement makes Y occur
/// at least twice in a premise where it occurred once (the premise already
/// held Y as a fixed argument), unless Y is already in an equivalence class at
/// the goal's scope; with two constants each gets its own verdict and copy
/// (D-483). The class test reads Y by its block name and the copy is written
/// at the goal's scope (D-536); an anchor argument whose
/// anchor slot is not `(1)` is never a copy target (asserted,
/// I-327). Only chains with an id of at least
/// @p firstChainId are visited: the goal door passes `1`, the goal rescan
/// passes the first chain its previous rescans left unvisited, since a
/// repeated visit of a (goal, chain) pair changes nothing. The candidate list
/// is snapshotted at entry; chains minted by the preparations are left to the
/// next call.
/// A registered open goal can request the sole remaining bound argument of
/// a positive atomic single premise, when an installed owner is visible at
/// the goal scope. The argument is marked before removing u_ prefixes; every
/// other argument stays fixed. Ordinary admission records the demand and
/// wakes parked sources; atomic self-integration remains
/// excluded (D-488).
///
/// @param inputExprStr The goal (span over a caller-stable buffer).
/// @param mb           The owning LB.
/// @param validityName The goal's scope (span over a caller-stable buffer).
/// @param firstChainId The smallest chain id to visit; `>= 1`.
/// @return Nothing.
/// @invariant Zero heap; visits candidates in ascending chain-id order.
/// @see OriginalsHeadIndex, maskedHeadKey, prepareIntegration.
void checkNecessityForEquality(StrSpan inputExprStr, Memory& mb, StrSpan validityName,
                                       int32_t firstChainId);

/// @brief Visit the goal-driven marked-head implications of the rules
///        whose head can conclude a missing expression
///        (D-574).
///
/// @details
/// The rule search of the goal-driven operator demand. Candidates are the
/// rule chains whose head carries @p expression's operator
/// (`OriginalsHeadIndex::chainsWithHead`, ascending chain id, snapshotted at
/// entry) with the expression's sign and an owner visible at @p validity.
/// The head is bound to the expression: a fixed `u_` argument must name the
/// expression's argument at its slot, a rule variable binds to it
/// (consistently where it repeats), and each bound variable is replaced in
/// the premises by the `u_` form of its value.
///
/// Then each positive operator premise (a configured operator, never the
/// anchor) whose output argument the head leaves open — a rule variable
/// occurring once in it — qualifies when the premises that do not name that
/// output hold at least `ExecutionParameters::kGoalDemandMinPresentPremises`
/// positive non-anchor premises and fix every other argument of the
/// operator premise: a `u_` name, or an argument of a present premise, the
/// anchor included (an anchor-slot argument is ground, D-287). For each,
/// @p sink receives the present premises followed by the operator premise,
/// the output argument, and the identity text: the fully bound implication
/// of the present premises and the operator premise with `marker` in its
/// output slot.
///
/// @tparam Sink Callable `void(const StrSpan* key, int32_t keyN,
///              StrSpan markedArg, StrSpan identity)`; it may mint into the
///              block's interners (every chain text is copied first).
/// @param mb         The owning LB; its head index is built (asserted).
/// @param expression The ground missing expression (never `marker`).
/// @param validity   The scope it is missing at, an existing scope name.
/// @param strArena   String-tier scratch arena; the spans handed to @p sink
///                   live until it returns.
/// @param genArena   Gen-tier scratch arena holding the candidate snapshot,
///                   popped before the return.
/// @return Nothing.
/// @invariant Reads the rule chains only; 0% heap.
/// @see serveGoalNearMiss, HashRuleInstaller::installGoalDemandVariant.
template <typename Sink>
void forEachGoalDemandVariant(Memory& mb, StrSpan expression, StrSpan validity,
            ScratchArena& strArena, ScratchArena& genArena, Sink&& sink) const;

/// @brief Serve one near-miss of a goal's flat integration rule: install
///        the goal-driven marked-head implications for the missing
///        expression (D-574).
///
/// @details
/// Called by `AdmissionMemory::drainAdmissionKeysOrdis2` for a demand record
/// whose variant was a goal near-miss (`Ordis2DemandRecord::goalNearMiss`),
/// in canonical firing order at the post-fixpoint seam. The rule search
/// (`forEachGoalDemandVariant`) yields the marked-head implications of the
/// rules that can conclude @p expression; each is installed at @p validity
/// (`HashRuleInstaller::installGoalDemandVariant`) once:
/// `LbMemory::goalDemandInstalls` holds the rule owner (identity text,
/// scope) of every implication installed, since a marker LMV run appends
/// without dedup. Under `ban_disintegration` (the compressor) nothing is
/// served, as in CNFE: the witness a demand wakes needs the disintegration
/// the compressor bans.
///
/// @param mb         The owning LB (single-threaded seam, I-83).
/// @param expression The ground missing expression; span stable across
///                   the call.
/// @param validity   The scope the near-miss fired at, an existing scope
///                   name; span stable across the call.
/// @return Nothing.
/// @invariant Each (implication, scope) pair is installed at most once per
///            block; 0% heap.
/// @see forEachGoalDemandVariant, AdmissionMemory::drainAdmissionKeysOrdis2.
void serveGoalNearMiss(Memory& mb, StrSpan expression, StrSpan validity);

/// @brief Whether any rule head can match an input under CNFE's head match,
///        judged by the input's operator alone.
///
/// @details
/// CNFE accepts a chain only when its head names the input's operator, that
/// operator has a config, and each of the head's one or two constants sits at
/// a slot whose definition set is `(1)`. Both config facts depend on the
/// operator alone, so an input whose operator has no config or no `(1)` slot
/// cannot match any chain and its probe is skipped before any chain is
/// touched. Shared by CNFE and the contradiction-scope demand.
///
/// @param inputName The input's operator name (`extractExpressionSpan`).
/// @param inputArgN The input's argument count.
/// @return True iff the operator has a config with at least one `(1)` slot
///         among the first @p inputArgN slots.
/// @invariant Pure; allocates nothing.
/// @see checkNecessityForEquality, collectHeadMatchChains.
bool headMatchPossible(StrSpan inputName, int32_t inputArgN) const;

/// @brief The masked head key under which CNFE's head match indexes one
///        rule chain, or an empty span when no input can ever match it.
///
/// @details
/// A chain is acceptable when its head has one constant (a non-`u_`
/// argument), or two constants beside at least one other argument, the
/// head's operator has a config, and each constant's slot has the definition
/// set `(1)`. Its key is `maskedHeadKey` of the head with the constant slots
/// masked; the key carries the head's sign. These are the per-chain checks
/// that depend on the chain alone; the input side is probed with the same key
/// shape (`collectHeadMatchChains`). The contradiction-scope demand reads the
/// key of a newly installed chain to find the facts it can match.
///
/// @param mb      The owning LB (its `originals` and `ruleInterner`).
/// @param chainId The chain id in `overallHashMemory.originals`; `>= 1`.
/// @param keyBuf  Caller buffer receiving the key bytes.
/// @param cap     Capacity of @p keyBuf; an overflow asserts.
/// @return The key as a span over @p keyBuf, or an empty span.
/// @invariant Reads only; mints nothing.
/// @see ensureOriginalsHeadIndexBuilt, maskedHeadKey, OriginalsHeadIndex.
StrSpan maskedKeyOfChain(Memory& mb, int32_t chainId, char* keyBuf, int32_t cap) const;

/// @brief Bring the block's derived rule-head index up to date with its
///        rule-chain map.
///
/// @details
/// Extends `overallHashMemory.originalsHeadIndex` over the chains installed
/// since its last build: each chain under its head operator and, when
/// `maskedKeyOfChain` accepts it, under its masked head key. Every consumer of
/// the masked family builds it through this one function, so the family holds
/// exactly the keys CNFE's verdict accepts.
///
/// @param mb The owning LB.
/// @return Nothing.
/// @invariant Mints only into the index's own derived interners.
/// @see OriginalsHeadIndex::ensureBuilt, maskedKeyOfChain.
void ensureOriginalsHeadIndexBuilt(Memory& mb) const;

/// @brief Collect the rule chains whose head matches an input under CNFE's
///        head match, in ascending chain-id order.
///
/// @details
/// One probe of the masked family per argument slot of the input, and one
/// per pair of slots when a third slot remains; each probe carries the
/// input's sign. The answers are disjoint (a key fixes its set of masked
/// slots); their union restricted to `[firstChainId, lastChainId]` is sorted
/// ascending and appended to @p chainIdx. The index must be built
/// (`ensureOriginalsHeadIndexBuilt`).
///
/// @param genericInput The input with every argument `u_`-prefixed (span over
///                     a caller-stable buffer).
/// @param mb           The owning LB.
/// @param firstChainId The smallest chain id to collect; `>= 1`.
/// @param lastChainId  The largest chain id to collect.
/// @param ocArena      The gen-scratch arena holding the candidate run; the
///                     byte-bump run is popped before the return.
/// @param chainIdx     Receives the chain ids (appended).
/// @return The number of chain ids appended.
/// @invariant Zero heap; reads the index only.
/// @see checkNecessityForEquality, maskedHeadKey.
int32_t collectHeadMatchChains(StrSpan genericInput, Memory& mb, int32_t firstChainId,
                                int32_t lastChainId, ScratchArena& ocArena,
                                PagedVector<int32_t>& chainIdx) const;

/// @brief Locate the constant slots of a head that the masked family
///        matched against an input, asserting the rest of the match.
///
/// @details
/// The index guarantees the verdict: the head names the input's operator with
/// the input's sign and arity, carries one constant (a non-`u_` argument), or
/// two beside at least one other argument, each at a `(1)` definition slot,
/// and equals the input at every other slot. This function finds the
/// constant slots and asserts each of those facts (Rule 19).
///
/// @param headView      The chain's head text.
/// @param inputName     The input's operator name.
/// @param negatedInput  Whether the input is negated.
/// @param inputArgs     The input's `u_`-prefixed argument spans.
/// @param inputArgN     Their count.
/// @param constantSlots Receives the 0-based constant slots (unused entries
///                      `-1`).
/// @return The number of constant slots, 1 or 2.
/// @invariant Pure; allocates nothing.
/// @see collectHeadMatchChains, checkNecessityForEquality.
int32_t headMatchConstantSlots(StrSpan headView, StrSpan inputName, bool negatedInput,
                               const StrSpan* inputArgs, int32_t inputArgN,
                               int32_t constantSlots[2]) const;
private:

AdmissionPolicy& admissionPolicy;
CanonicalTemplateMatcher& canonicalTemplates;
Canonicalizer& canonicalizer;
const CompiledDefinitionRegistry& definitions;
IntegrationInstructionBuilder& integrationBuilder;
const ProverParameters& parameters;
ScopeTypeResolver& scopeTypes;
AdmissionMemory& admissionMemory;
DisintegrationProcessor& disintegrationProcessor;
HashRuleInstaller& hashRuleInstaller;
ImplicationSubproofs& implicationSubproofs;
RejectedSourceReplay& rejectedSourceReplay;
StatementMemory& statementMemory;
UnaryIntegrationProcessor& unaryIntegrationProcessor;
CompactRuleProcessor& compactRuleProcessor;
};
}
#include "../logical_block/admission_memory.hpp"
#include "../logical_block/compact_rule_processor.hpp"
#include "../logical_block/disintegration_processor.hpp"
#include "../inference/hash_rule_installer.hpp"
#include "../logical_block/implication_subproofs.hpp"
#include "../logical_block/rejected_source_replay.hpp"
#include "../logical_block/statement_memory.hpp"
#include "../logical_block/unary_integration_processor.hpp"
namespace gl {

/// @brief Fill @p out with the `originals` rule-registry ids in ID ORDER
        ///        (`1..count` — the registry's deterministic insertion order).
        ///
        /// @details
        /// The former decoded-chain lexicographic `std::sort` here is RETIRED
        /// (maintainer directive 2026-09-01): the decoded-lex order existed only
        /// to reproduce the pre-statification heap snapshot's byte order, and it
        /// cost 35% of all phase-1/3 runtime on the FTA shortcut (measured
        /// 107 worker-seconds, `CNFE_SORT_CHAINS`). Ids are minted in the
        /// machine's own deterministic sequence, so id order is byte-stable
        /// across runs and hosts — deterministic, merely a DIFFERENT observable
        /// order than the retired one. Downstream mint order (the consumers'
        /// `prepareIntegration` walks) changed with it, gated by the verifier
        /// and the theorems.txt comparison.
        ///
        /// @param mb  The memory block (reads `overallHashMemory.originals`).
        /// @param out Caller-owned buffer (sized to `originals.count()` on the
        ///            byte-bump tier at the production consumers) receiving the ids.
        /// @param cap Capacity of @p out (== `originals.count()` at the consumers);
        ///            a `count > cap` overrun is a loud Rule-19 assert.
        /// @return The chain count written into `out[0..count)`.
        /// @invariant `out[k] == k + 1` — the deterministic id order.
        /// @see prepareIntegrationCore2 (Case B), the one consumer;
        ///      checkNecessityForEquality walks the derived
        ///      `OriginalsHeadIndex` instead.
inline int32_t IntegrationProcessor::sortOriginalChainIndex(Memory& mb, int32_t* out, int32_t cap) {
            const int32_t count = mb.overallHashMemory.originals.count();
            assert(count <= cap
                && "sortOriginalChainIndex: chain count exceeds caller capacity");
            for (int32_t k = 0; k < count; ++k) out[k] = k + 1;
            return count;
        }

/// @brief The compiled entity whose integration set a Case B table entity
        ///        takes (D-503).
        ///
        /// @details
        /// A registry entity by its core name; a set- or relation-typed `=`
        /// (category `and`) is the extensionality entity of its type label
        /// (I-267), resolved as `prepareIntegration` resolves the goal's
        /// (`ScopeTypeResolver::resolveEqualityGoalLabel`). Every such entity
        /// carries its integration set from its creation — asserted
        /// non-empty, the one-level rule first.
        ///
        /// @param signature      The entity's `u_`-prefixed signature.
        /// @param category       Its category (`and` / `existence`).
        /// @param cleanSignature The signature `u_`-stripped.
        /// @param mb             The owning LB.
        /// @param validityName   The integration scope.
        /// @return The entity (registry storage, stable through the call).
        /// @see prepareIntegrationCore2 (Case B), IntegrationRuleCompiler.
inline const LogicalEntity& IntegrationProcessor::integrationEntityOf(StrSpan signature, StrSpan category,
            StrSpan cleanSignature, Memory& mb, StrSpan validityName) {
            const StrSpan core = extractExpressionSpan(signature);
            const LogicalEntity* entity = nullptr;
            if (equalSpans(core, StrSpan("=", 1))) {
                assert(equalSpans(category, StrSpan("and", 3))
                    && "integrationEntityOf: an `=` integrates only as its typed extensionality and");
                entity = this->definitions.typedEqualityEntity(
                    scopeTypes.resolveEqualityGoalLabel(cleanSignature, mb, validityName));
            } else {
                entity = this->definitions.compiledEntity(core);
            }
            assert(entity != nullptr && equalSpans(StrSpan(entity->category), category)
                && "integrationEntityOf: a Case B entity without its compiled entity");
            assert(!entity->integrationImplications.empty()
                && entity->integrationImplications.front().kind == IntegrationRuleKind::OneLevel
                && "integrationEntityOf: an entity without its integration set");
            return *entity;
        }

/// @brief Whether a Case B existence carries its witness fixed, and which.
        ///
        /// @details
        /// The table's elements are the entity's compiled elements instantiated
        /// in order; the compiled elements name the witness by a placeholder
        /// (the one argument that is not a `u_` token). The table element at
        /// the placeholder's first position holds a `pi_` name while the
        /// witness is bound and a `u_`-prefixed name once a marker replay
        /// fixed it; nothing else stands there (asserted).
        ///
        /// @param entity   The compiled existence.
        /// @param elements The table entity's elements.
        /// @param elemN    Their number (asserted equal to the entity's).
        /// @param witness  Out: the fixed witness (only when `true`).
        /// @return `true` when the witness is fixed (the occupied form).
        /// @see prepareIntegrationCore2 (Case B).
inline bool IntegrationProcessor::occupiedWitness(const LogicalEntity& entity, const StrSpan* elements,
            int32_t elemN, StrSpan& witness) {
            assert(entity.category == "existence" && static_cast<int32_t>(entity.elements.size()) == elemN
                && "occupiedWitness: the table entity does not instantiate its compiled existence");
            for (int32_t j = 0; j < elemN; ++j) {
                StrSpan compiledArgs[ExecutionParameters::MAX_ARITY];
                const int32_t compiledN = getArgsSpans(StrSpan(entity.elements[static_cast<std::size_t>(j)]),
                    compiledArgs, ExecutionParameters::MAX_ARITY);
                StrSpan tableArgs[ExecutionParameters::MAX_ARITY];
                const int32_t tableN = getArgsSpans(elements[j], tableArgs, ExecutionParameters::MAX_ARITY);
                assert(compiledN == tableN && "occupiedWitness: a table element of another arity");
                for (int32_t a = 0; a < compiledN; ++a) {
                    if (startsWithSpan(compiledArgs[a], "u_", 2)) continue;
                    const bool fixed = startsWithSpan(tableArgs[a], "u_", 2);
                    assert((fixed || startsWithSpan(tableArgs[a], "pi_", 3))
                        && "occupiedWitness: a witness that is neither bound nor fixed");
                    if (fixed) witness = tableArgs[a];
                    return fixed;
                }
            }
            assert(false && "occupiedWitness: an existence without a witness placeholder");
            return false;
        }

/// @brief Whether an existence part of an instruction has its fixed witness
        ///        among its own arguments
        ///        (D-590).
        ///
        /// @details
        /// The canonical rewrite at the integration scope gives every name its
        /// class representative, so a witness that a marker fill fixed and an
        /// argument of the same existence can become one name: the
        /// partial-product witness of a fold and the sequence it folds, for
        /// example (`and0[i0,x,x,…]`). That existence could only be
        /// introduced with its witness doubling as its argument; the
        /// integration went the wrong way, and `prepareIntegrationCore2`
        /// stops the continuation instead of building it. Only existence
        /// parts whose signature arguments are all `u_` are read, the parts
        /// Case B would introduce; their witness slot is found by position
        /// (`occupiedWitness`), and a bound (`pi_`) witness never stands among
        /// the arguments.
        ///
        /// @param instruction The rewritten instruction, canonical at the scope.
        /// @return True iff some such existence part has a fixed witness equal
        ///         to one of its signature arguments.
        /// @invariant Pure: reads the definition registry only.
        /// @see prepareIntegrationCore2, occupiedWitness.
inline bool IntegrationProcessor::witnessDoublesAsArgument(const WorkInstruction& instruction) const {
            for (int32_t li = 0; li < instruction.entityCount(); ++li) {
                if (!equalSpans(instruction.category(li), StrSpan("existence", 9))) continue;
                const StrSpan signature = instruction.signature(li);
                StrSpan args[ExecutionParameters::MAX_ARITY];
                const int32_t argN = getArgsSpans(signature, args, ExecutionParameters::MAX_ARITY);
                bool allU = true;
                for (int32_t a = 0; a < argN && allU; ++a) allU = startsWithSpan(args[a], "u_", 2);
                if (!allU) continue;
                const LogicalEntity* entity = this->definitions.compiledEntity(extractExpressionUniversalSpan(signature));
                assert(entity != nullptr && entity->category == "existence"
                    && "witnessDoublesAsArgument: an existence part without its compiled existence");
                const int32_t elemN = instruction.elemCount(li);
                assert(elemN <= ExecutionParameters::MAX_INSTRUCTION_ELEMENTS
                    && "witnessDoublesAsArgument: element count exceeds MAX_INSTRUCTION_ELEMENTS");
                StrSpan elements[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
                for (int32_t j = 0; j < elemN; ++j) elements[j] = instruction.elemAt(li, j);
                StrSpan witness;
                if (!occupiedWitness(*entity, elements, elemN, witness)) continue;
                for (int32_t a = 0; a < argN; ++a)
                    if (equalSpans(args[a], witness)) return true;
            }
            return false;
        }

/// @brief Install the root family of a root integration goal from its
        ///        compiled integration set: the flat variants, the
        ///        introductions of their OR premises and the flat rules of
        ///        those premises' existence disjuncts
        ///        (D-503, D-352, D-492).
        ///
        /// @details
        /// The addon to Case B for the entity that IS the expression handed to
        /// `prepareIntegration` (its `u_`-stripped signature equals the
        /// instruction's marked goal). Case B installs the set's one-level
        /// rule; this walks the remaining records in their compiled order —
        /// the order the rules were formerly built in — and installs each
        /// through `CompactRuleProcessor::installIntegrationCompact`, the
        /// owner tokens filled with the root's arguments:
        /// - `Flat`: tag `reformulation for integration flattened`, citing
        ///   `<root>_integration_goal`, install kind `IntegrationFlat` (its
        ///   near-miss demand variants report its one missing compound
        ///   premise to the goal-driven rule search,
        ///   D-574);
        /// - `OrIntroduction`: tag `expansion`, citing the filled compact of
        ///   the rule it serves (`cites`); its disjunct (found among the OR
        ///   premise's `forEachOrIntroductionRule` disjuncts by the expanded
        ///   premises) is then offered to
        ///   `AdmissionMemory::updateAdmissionMapIntegrationDouble`, installed
        ///   or not;
        /// - `DisjunctFlat`: tag `reformulation for integration flattened`,
        ///   citing `<disjunct>_integration_goal`, the disjunct being the
        ///   expanded rule's head.
        /// A compact already expanded at the scope notes the reach of its
        /// rule's owner instead (once per scope).
        ///
        /// @param entity         The root's compiled entity.
        /// @param owners         The root's arguments (`u_`-prefixed), the owner
        ///                       tokens' values.
        /// @param ownerN         Their number.
        /// @param oneLevel       The root's filled one-level compact, `u_`-stripped.
        /// @param cleanSignature The root signature `u_`-stripped.
        /// @param mb             The owning LB (single-threaded seam).
        /// @param validityName   The install scope.
        /// @param coreArena      The caller's string-tier scratch arena.
        /// @return Nothing.
        /// @invariant 0% heap; every filled compact rides @p coreArena (the
        ///            caller's scope holds them through the installs).
        /// @see prepareIntegrationCore2 (Case B), installIntegrationCompact,
        ///      forEachOrIntroductionRule.
inline void IntegrationProcessor::installRootIntegrationFamily(const LogicalEntity& entity,
            const StrSpan* owners, int32_t ownerN, StrSpan oneLevel, StrSpan cleanSignature,
            Memory& mb, StrSpan validityName, ScratchArena& coreArena) {
            const int32_t recordN = static_cast<int32_t>(entity.integrationImplications.size());
            // An existence's occupied form (record 1) is never part of the
            // root family: the root's witness is bound.
            const int32_t first = entity.category == "existence" ? 2 : 1;
            if (recordN <= first) return;
            StrSpan* filled = reinterpret_cast<StrSpan*>(coreArena.resolve(coreArena.alloc(
                recordN * static_cast<int32_t>(sizeof(StrSpan)), static_cast<int32_t>(alignof(StrSpan)))));
            filled[0] = oneLevel;
            const StrSpan IG_GOAL("_integration_goal",
                static_cast<int32_t>(sizeof("_integration_goal") - 1));
            const auto goalOf = [&](StrSpan base) -> StrSpan {
                const int32_t len = base.len + IG_GOAL.len;
                char* buf = coreArena.allocBytes(len);
                std::memcpy(buf, base.ptr, static_cast<std::size_t>(base.len));
                std::memcpy(buf + base.len, IG_GOAL.ptr, static_cast<std::size_t>(IG_GOAL.len));
                return StrSpan(buf, len);
            };
            const StrSpan rootGoal = goalOf(cleanSignature);
            for (int32_t i = first; i < recordN; ++i) {
                const IntegrationImplication& record = entity.integrationImplications[static_cast<std::size_t>(i)];
                const ScratchString compactU = integrationBuilder.fillIntegrationCompactScratch(coreArena,
                    StrSpan(record.compact), owners, ownerN);
                filled[i] = StrSpan(gl::DefinitionExpansion::removeUPrefixScratch(coreArena, StrSpan(compactU)));
                StrSpan premises[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
                int32_t premiseN = 0;
                StrSpan head;
                if (record.kind == IntegrationRuleKind::Flat) {
                    // A root goal's flat variant reports its one missing
                    // compound premise (D-574).
                    const OriginDep deps[1] = { { rootGoal, validityName } };
                    compactRuleProcessor.installIntegrationCompact(mb, StrSpan(compactU), validityName,
                        OriginTag::reformulationForIntegrationFlattened, deps, 1, coreArena,
                        premises, ExecutionParameters::MAX_INSTRUCTION_ELEMENTS, premiseN, head,
                        RuleInstallKind::IntegrationFlat);
                } else if (record.kind == IntegrationRuleKind::OrIntroduction) {
                    assert(record.cites >= 0 && record.cites < i && (record.cites == 0 || record.cites >= first)
                        && "installRootIntegrationFamily: an introduction cites a rule outside the root family");
                    const OriginDep deps[1] = { { filled[record.cites], validityName } };
                    compactRuleProcessor.installIntegrationCompact(mb, StrSpan(compactU), validityName,
                        OriginTag::expansion, deps, 1, coreArena,
                        premises, ExecutionParameters::MAX_INSTRUCTION_ELEMENTS, premiseN, head);
                    // The disjunct this introduction concludes from files its
                    // double-marker demand, whether or not the rule was new
                    // (D-492).
                    ScratchScope introScope(coreArena);
                    int32_t matches = 0;
                    forEachOrIntroductionRule(this->definitions, head, coreArena,
                        [&](StrSpan disjunct, const StrSpan* introPremises, int32_t introPremiseN, StrSpan) {
                            if (introPremiseN != premiseN) return;
                            for (int32_t p = 0; p < premiseN; ++p)
                                if (!equalSpans(introPremises[p], premises[p])) return;
                            ++matches;
                            admissionMemory.updateAdmissionMapIntegrationDouble(mb, disjunct, validityName);
                        });
                    assert(matches == 1 && "installRootIntegrationFamily: an introduction without its disjunct");
                } else {
                    assert(record.kind == IntegrationRuleKind::DisjunctFlat
                        && "installRootIntegrationFamily: a one-level record inside the root family");
                    premiseN = integrationBuilder.expandIntegrationCompactScratch(coreArena, StrSpan(compactU), mb,
                        premises, ExecutionParameters::MAX_INSTRUCTION_ELEMENTS, head);
                    const OriginDep deps[1] = {
                        { goalOf(StrSpan(gl::DefinitionExpansion::removeUPrefixScratch(coreArena, head))), validityName } };
                    compactRuleProcessor.installExpandedIntegrationCompact(mb, StrSpan(compactU), validityName,
                        OriginTag::reformulationForIntegrationFlattened, deps, 1, coreArena,
                        premises, premiseN, head);
                }
            }
        }

/// @brief Visit the goal-driven marked-head implications of the rules
        ///        whose head can conclude a missing expression
        ///        (D-574).
        ///
        /// @details
        /// The rule search of the goal-driven operator demand. Candidates are
        /// the rule chains whose head carries @p expression's operator
        /// (`OriginalsHeadIndex::chainsWithHead`, ascending chain id,
        /// snapshotted at entry) with the expression's sign and an owner
        /// visible at @p validity. The head is bound to the expression: a fixed
        /// `u_` argument must name the expression's argument at its slot, a
        /// rule variable binds to it (consistently where it repeats), and each
        /// bound variable is replaced in the premises by the `u_` form of its
        /// value.
        ///
        /// Then each positive operator premise (a configured operator, never
        /// the anchor) whose output argument the head leaves open — a rule
        /// variable occurring once in it — qualifies when the premises that do
        /// not name that output hold at least
        /// `ExecutionParameters::kGoalDemandMinPresentPremises` positive
        /// non-anchor premises and fix every other argument of the operator
        /// premise: a `u_` name, or an argument of a present premise, the
        /// anchor included (an anchor-slot argument is ground, D-287). For
        /// each, @p sink receives the present premises followed by the
        /// operator premise, the output argument, and the identity text: the
        /// fully bound implication of the present premises and the operator
        /// premise with `marker` in its output slot.
        ///
        /// @tparam Sink Callable `void(const StrSpan* key, int32_t keyN,
        ///              StrSpan markedArg, StrSpan identity)`; it may mint into
        ///              the block's interners (every chain text is copied
        ///              first).
        /// @param mb         The owning LB; its head index is built (asserted).
        /// @param expression The ground missing expression (never `marker`).
        /// @param validity   The scope it is missing at, an existing scope name.
        /// @param strArena   String-tier scratch arena; the spans handed to
        ///                   @p sink live until it returns.
        /// @param genArena   Gen-tier scratch arena holding the candidate
        ///                   snapshot, popped before the return.
        /// @return Nothing.
        /// @invariant Reads the rule chains only; 0% heap.
        /// @see serveGoalNearMiss, HashRuleInstaller::installGoalDemandVariant.
template <typename Sink>
inline void IntegrationProcessor::forEachGoalDemandVariant(Memory& mb, StrSpan expression,
            StrSpan validity, ScratchArena& strArena, ScratchArena& genArena, Sink&& sink) const {
            const OriginalsHeadIndex& headIndex = mb.overallHashMemory.originalsHeadIndex;
            assert(headIndex.built() && "forEachGoalDemandVariant: probe on an unbuilt head index");
            assert(!containsSpan(expression, StrSpan("marker", 6))
                && "forEachGoalDemandVariant: a missing expression is ground");
            const NameId scope = mb.nameMap.lookup(validity);
            assert(scope != 0 && "forEachGoalDemandVariant: a scope the block does not know");
            const bool negated = expression.len > 0 && expression.ptr[0] == '!';
            StrSpan exprArgs[ExecutionParameters::MAX_ARITY];
            const int32_t exprArgN = getArgsSpans(expression, exprArgs, ExecutionParameters::MAX_ARITY);
            const auto isAnchor = [](StrSpan e) -> bool {
                return e.len >= 7 && equalSpans(StrSpan(e.ptr, 7), StrSpan("(Anchor", 7));
            };
            const auto names = [](StrSpan premise, StrSpan arg) -> bool {
                StrSpan args[ExecutionParameters::MAX_ARITY];
                const int32_t n = getArgsSpans(premise, args, ExecutionParameters::MAX_ARITY);
                for (int32_t a = 0; a < n; ++a)
                    if (equalSpans(args[a], arg)) return true;
                return false;
            };

            // The candidates, snapshotted: the sink mints into the block's
            // interners.
            const ArenaOffset genMark = genArena.cursor();
            DirtyState dirty = DirtyState::Clean;
            PagedVector<int32_t> chains(&genArena, &dirty);
            headIndex.chainsWithHead(extractExpressionSpan(expression),
                [&](int32_t id) { chains.push_back(id); });

            for (int32_t c = 0; c < chains.size(); ++c) {
                const int32_t chainId = chains[c];
                // A rule applies only where one of its owners is visible.
                bool visible = false;
                const int32_t ownerN = mb.overallHashMemory.originals.runLen(chainId);
                for (int32_t o = 0; o < ownerN && !visible; ++o) {
                    const RuleOwner owner = mb.overallHashMemory.originals.recordAt(chainId, o).owner;
                    const NameId ownerScope = static_cast<NameId>(static_cast<uint64_t>(owner) & 0xffffffffu);
                    visible = !mb.phase3IgnoresScope(ownerScope)
                        && (ownerScope == scope || mb.nameMap.isStrictAncestor(ownerScope, scope));
                }
                if (!visible) continue;

                // The chain's texts are copied before any mint (I-3).
                ScratchScope chainFrame(strArena);
                StrSpan elems[64];
                int32_t count = 0;
                {
                    const StrSpan kb = mb.overallHashMemory.originals.inner().keyAt(chainId);
                    const IdVecKeyView kv = viewIdVecKey(kb);
                    count = kv.count();
                    assert(count >= 1 && count <= 64
                        && "forEachGoalDemandVariant: a headed chain of at most 64 elements");
                    for (int32_t i = 0; i < count; ++i) {
                        const StrSpan dv = mb.ruleInterner.decodeView(kv.idAt(i));
                        elems[i] = StrSpan(ScratchString::copyFrom(strArena, dv.ptr, dv.len));
                    }
                }
                const StrSpan head = elems[count - 1];
                const int32_t premiseN = count - 1;
                if ((head.ptr[0] == '!') != negated) continue;
                StrSpan headArgs[ExecutionParameters::MAX_ARITY];
                const int32_t headArgN = getArgsSpans(head, headArgs, ExecutionParameters::MAX_ARITY);
                assert(headArgN == exprArgN
                    && "forEachGoalDemandVariant: a head of the expression's operator with another arity");

                // Bind the head to the expression.
                StrReplacement bind[ExecutionParameters::MAX_ARITY];
                int32_t bindN = 0;
                bool matches = true;
                for (int32_t i = 0; i < headArgN && matches; ++i) {
                    const StrSpan h = headArgs[i];
                    if (startsWithSpan(h, "u_", 2)) {
                        matches = equalSpans(StrSpan(h.ptr + 2, h.len - 2), exprArgs[i]);
                        continue;
                    }
                    int32_t prior = -1;
                    for (int32_t b = 0; b < bindN && prior < 0; ++b)
                        if (equalSpans(bind[b].key, h)) prior = b;
                    if (prior >= 0) {
                        const StrSpan value = bind[prior].value;
                        matches = equalSpans(StrSpan(value.ptr + 2, value.len - 2), exprArgs[i]);
                        continue;
                    }
                    char* buf = strArena.allocBytes(exprArgs[i].len + 2);
                    buf[0] = 'u';
                    buf[1] = '_';
                    std::memcpy(buf + 2, exprArgs[i].ptr, static_cast<std::size_t>(exprArgs[i].len));
                    bind[bindN++] = StrReplacement{ h, StrSpan(buf, exprArgs[i].len + 2) };
                }
                if (!matches) continue;
                StrSpan bound[64];
                for (int32_t k = 0; k < premiseN; ++k)
                    bound[k] = StrSpan(replaceKeysScratch(strArena, elems[k], bind, bindN));

                for (int32_t o = 0; o < premiseN; ++o) {
                    const StrSpan premise = bound[o];
                    if (isAnchor(premise) || premise.ptr[0] != '(') continue;
                    const StrSpan opName = extractExpressionSpan(premise);
                    if (this->definitions.operatorNames().find(std::string_view(opName.ptr,
                            static_cast<std::size_t>(opName.len)))
                        == this->definitions.operatorNames().end()) continue;
                    const ce::CoreExpressionConfig* cfg = this->definitions.coreConfig(opName);
                    assert(cfg != nullptr && cfg->outputIndices.size() == 1
                        && "forEachGoalDemandVariant: an operator without its one output slot");
                    StrSpan opArgs[ExecutionParameters::MAX_ARITY];
                    const int32_t opArgN = getArgsSpans(premise, opArgs, ExecutionParameters::MAX_ARITY);
                    const int32_t outSlot = cfg->outputIndices[0];
                    assert(outSlot >= 0 && outSlot < opArgN
                        && "forEachGoalDemandVariant: an output slot outside the operator's arity");
                    // The head leaves the output open: a rule variable, once in
                    // the operator premise.
                    const StrSpan out = opArgs[outSlot];
                    if (startsWithSpan(out, "u_", 2)) continue;
                    int32_t outCount = 0;
                    for (int32_t a = 0; a < opArgN; ++a)
                        if (equalSpans(opArgs[a], out)) ++outCount;
                    if (outCount != 1) continue;

                    // The present premises: every other premise without the
                    // output, the anchor included.
                    StrSpan key[64];
                    int32_t keyN = 0;
                    int32_t positive = 0;
                    for (int32_t k = 0; k < premiseN; ++k) {
                        if (k == o || names(bound[k], out)) continue;
                        key[keyN++] = bound[k];
                        if (!isAnchor(bound[k]) && bound[k].ptr[0] == '(') ++positive;
                    }
                    if (positive < ExecutionParameters::kGoalDemandMinPresentPremises) continue;
                    bool fixed = true;
                    for (int32_t a = 0; a < opArgN && fixed; ++a) {
                        if (a == outSlot || startsWithSpan(opArgs[a], "u_", 2)) continue;
                        bool named = false;
                        for (int32_t k = 0; k < keyN && !named; ++k) named = names(key[k], opArgs[a]);
                        fixed = named;
                    }
                    if (!fixed) continue;

                    ScratchScope variantFrame(strArena);
                    const StrReplacement markOutput[1] = { { out, StrSpan("marker", 6) } };
                    const ScratchString markedHead = replaceKeysScratch(strArena, premise, markOutput, 1);
                    const ScratchString identity = gl::ImplicationBuilder::reconstructImplicationFullBindScratch(
                        strArena, key, keyN, StrSpan(markedHead));
                    key[keyN++] = premise;
                    sink(static_cast<const StrSpan*>(key), keyN, out, StrSpan(identity));
                }
            }
            chains.clear();
            genArena.popTo(genMark);
        }

/// @brief Prepare an integration instruction's entities at a scope: open
        ///        implication subproofs (Case A), split ORs (Case OR), install
        ///        the compacts and admission keys of existences and ands
        ///        (Case B) and file the marker keys (Case C).
        ///
        /// @details
        /// Every entity is first rewritten with @p replacementMap. An entity
        /// already prepared counts as done: Cases A, OR and B skip it when its
        /// gate template is in `integrationPrepared`, and mint the template
        /// otherwise. A fresh preparation (from `prepareIntegration`) reads the
        /// gate at @p validityName only. A @p continuation, the replay of a
        /// stored instruction after its marker was filled (the commit of
        /// `AdmissionMemory::isAdmittedIntegration`,
        /// `AdmissionMemory::updateAdmissionMapIntegration`), reads it at
        /// @p validityName and at every strict ancestor
        /// (`CanonicalTemplateMatcher::canonicalTemplatePresentAtOrAbove`):
        /// an instruction owned at an ancestor continues below it as at its
        /// owner, whose preparation, the root goal's included, holds there
        /// (D-572).
        ///
        /// @param instructions The instruction (entities, marked goal, path
        ///                     list).
        /// @param replacementMap The rewrite pairs applied to every entity.
        /// @param replacementN Their number.
        /// @param mb           The owning LB.
        /// @param validityName The preparation scope.
        /// @param rootGoal     The goal driving a fresh preparation (embedded
        ///                     into scope payloads minted at `main`), or empty.
        /// @param continuation Whether this replays a stored instruction after
        ///                     a marker fill.
        /// @param splitSets    The forbidden sets of the copies of a fresh
        ///                     preparation split by premise, one per identical
        ///                     goal (null when @p splitN is 0): Case C files
        ///                     every instruction once per set
        ///                     (D-579).
        ///                     Without them each filed copy keeps the sets of
        ///                     @p instructions.
        /// @param splitN       Number of @p splitSets (0; at least 2, or 1 for a twin premise's goal, D-589).
        /// @return Nothing.
        /// @invariant Nothing is prepared at or below a restricted scope
        ///            (an integration trial or the usefulness scope).
        /// @see prepareIntegration, AdmissionMemory::isAdmittedIntegration,
        ///      AdmissionMemory::updateAdmissionMapIntegration.
//#pragma optimize("", off)
inline void IntegrationProcessor::prepareIntegrationCore2(WorkInstruction& instructions,
            const StrReplacement* replacementMap, int32_t replacementN,
            Memory& mb,
            StrSpan validityName,
            StrSpan rootGoal,
            bool continuation,
            const IntegrationForbiddenSets* splitSets,
            int32_t splitN) {
            assert((splitN == 0 || ((splitN >= 2 || mb.twinFirstGoalId != 0) && splitSets != nullptr && !continuation))
                && "prepareIntegrationCore2: a premise split needs at least two sets, or one twin premise's, on a fresh preparation");
            // An integration trial scope, the usefulness scope and the scopes
            // below them prepare no integration (D-481,
            // D-521).
            {
                const NameId scope = mb.nameMap.lookup(validityName);
                if (scope != 0 && gl::ScopeSyntax::atOrBelowRestrictedScope(mb.nameMap, scope)) return;
            }
            // 0% heap: makeAdmissionKeys and addToHashMemory now take StrSpan
            // (their span-run overloads landed in c1), so no validityName
            // std::string is materialized -- every use reads the span param.
            //
            // rootGoal: the toBeProved goal whose integration drives this call
            // (verbatim), or empty for statement-driven mints (admission /
            // marker replays, the load-time prefill pass). Case A / Case OR
            // embed it in every scope payload minted ON MAIN
            // (`<goal>_subproof_<bare>`) so a later disproof of the goal can
            // enumerate and wipe the scopes its integration spawned; deeper
            // mints need no embedding — they nest inside a main-level scope in
            // the validityNodes forest and ride its subtree wipe.
            const bool embedGoal =
                equalSpans(validityName, StrSpan("main", 4)) && rootGoal.len > 0;
            // One function-level string-scratch scope holds every span/ScratchString
            // this body builds (rewrite temporaries, per-entity leElements run, the
            // per-case clean/renamed/expanded strings); the existing per-record origin
            // scopes (gScope / peScope / igScope / ogScope / oaScope) NEST inside it
            // and rewind only their OWN allocations, so the case-level strings survive
            // every nested rewind (09b pitfall 6, byte-bump LIFO). Re-entrant Core2
            // calls open their own nested scope above the caller's replacementMap
            // pairs, which stay live below the mark (I-124).
            const unsigned coreSlot = (gl::WorkerContext::coreId() >= 0)
                ? static_cast<unsigned>(gl::WorkerContext::coreId())
                : scratchArenas().slotCount() - 1;
            ScratchArena& coreArena = scratchArenas().forSlot(coreSlot);
            ScratchScope coreScope(coreArena);

            // Concatenate spans into a fresh ScratchString on coreArena (explicit
            // length, single allocation) — the Case OR / branchPayload builders.
            const auto concatCore = [&](std::initializer_list<StrSpan> parts) -> ScratchString {
                int32_t total = 0;
                for (const StrSpan& p : parts) total += p.len;
                if (total == 0) return ScratchString();
                char* b = coreArena.allocBytes(total);
                int32_t at = 0;
                for (const StrSpan& p : parts) {
                    if (p.len > 0) { std::memcpy(b + at, p.ptr, static_cast<size_t>(p.len)); at += p.len; }
                }
                return ScratchString::wrap(coreArena, b, total);
            };

            auto makeMarkedExprLambda = [&](StrSpan expression) -> ScratchString {
                StrSpan args[ExecutionParameters::MAX_ARITY];
                const int32_t argN = getArgsSpans(expression, args,
                                                  ExecutionParameters::MAX_ARITY);
                StrReplacement pairs[ExecutionParameters::MAX_ARITY];
                int32_t pairN = 0;

                int counter = 0;
                for (int32_t ai = 0; ai < argN; ++ai) {
                    const StrSpan& arg = args[ai];
                    // Check for "u_repl_" prefix (length 7)
                    if (arg.len >= 7 && std::memcmp(arg.ptr, "u_repl_", 7) == 0) {
                        pairs[pairN].key = arg;
                        pairs[pairN].value = StrSpan(arg.ptr + 2, arg.len - 2); // "u_repl_..." -> "repl_..."
                        ++pairN;
                    }
                    // Check for "pi_" prefix (length 3)
                    else if (arg.len >= 3 && std::memcmp(arg.ptr, "pi_", 3) == 0) {
                        pairs[pairN].key = arg;
                        pairs[pairN].value = StrSpan("marker", 6);
                        ++pairN;
                        counter++;
                    }
                }

                assert(counter <= 1);

                return replaceKeysScratch(coreArena, expression, pairs, pairN);
                };

            // [EXTENSION] Pre-calculate allowed repetitions based on instructions
            // signatures. allowedRepeatedVars is a sorted-unique StrSpan run whose
            // repeated-arg bytes are COPIED onto coreArena (they must outlive the
            // per-li signature window). expressionsFromConfig is now transparent
            // (std::less<>), so the core-expression probe is a zero-allocation
            // std::string_view lookup (0% heap; the set's contents are unchanged).
            StrSpan allowedRepeatedVars[ExecutionParameters::MAX_ADMISSION_REM_ARGS];
            int32_t allowedN = 0;
            const auto insSortedSpan = [](StrSpan* arr, int32_t& cnt, const StrSpan& v) {
                int32_t lo = 0, hi = cnt;
                while (lo < hi) {
                    const int32_t mid = (lo + hi) / 2;
                    const int c = compareSpans(arr[mid], v);
                    if (c == 0) return;
                    if (c < 0) lo = mid + 1; else hi = mid;
                }
                assert(cnt < ExecutionParameters::MAX_ADMISSION_REM_ARGS
                    && "prepareIntegrationCore2: allowedRepeatedVars exceeds cap");
                for (int32_t k = cnt; k > lo; --k) arr[k] = arr[k - 1];
                arr[lo] = v; ++cnt;
            };
            for (int32_t li = 0; li < instructions.entityCount(); ++li) {
                const StrSpan sigSpan = instructions.signature(li);
                const StrSpan coreExpr = extractExpressionSpan(sigSpan);

                // If signature's core expression is part of coreExpressionMap keys
                if (this->definitions.configuredNames().find(
                        std::string_view(coreExpr.ptr, static_cast<std::size_t>(coreExpr.len)))
                    != this->definitions.configuredNames().end()) {
                    StrSpan args[ExecutionParameters::MAX_ARITY];
                    const int32_t argN = getArgsSpans(sigSpan, args,
                                                      ExecutionParameters::MAX_ARITY);
                    // Take those args which happen more than once (first-occurrence
                    // insert into the sorted-unique run; copyFrom so the span outlives
                    // this li's signature window).
                    for (int32_t a = 0; a < argN; ++a) {
                        int32_t occ = 0; bool first = true;
                        for (int32_t b = 0; b < argN; ++b) {
                            if (equalSpans(args[a], args[b])) { ++occ; if (b < a) first = false; }
                        }
                        if (occ > 1 && first) {
                            const ScratchString kept =
                                ScratchString::copyFrom(coreArena, args[a].ptr, args[a].len);
                            insSortedSpan(allowedRepeatedVars, allowedN, StrSpan(kept));
                        }
                    }
                }
            }

            // 1. Apply replacement map on all signatures and elements of every
            // instruction, rebuilding into a fresh working instruction (records are
            // read-only after commit, so the former in-place rewrite is a rebuild).
            const NameId canonicalScope = mb.nameMap.encode(validityName);
            canonicalizer.initializeCanonicalScope(mb, canonicalScope);
            const int32_t canonicalBucket = mb.canonicalEquivalenceClassesMap.lookup(canonicalScope);
            const bool canonicalFields = canonicalBucket != 0 && mb.canonicalEquivalenceClassesMap.runLen(canonicalBucket) != 0;
            WorkInstruction rewritten(instructions.arena);
            rewritten.setMarkedGoal(instructions.markedGoal());
            // The path list keeps the canonical form of every other field
            // (D-529).
            rewritten.setPathVariables(canonicalFields && !instructions.pathVariables().empty()
                ? StrSpan(admissionMemory.canonicalPathListScratch(mb, instructions.pathVariables(),
                    canonicalScope, coreArena))
                : instructions.pathVariables());
            // The forbidden sets name original variables and block levels;
            // classes do not rewrite them
            // (D-579).
            rewritten.forbidden = instructions.forbidden;
            for (int32_t li = 0; li < instructions.entityCount(); ++li) {
                const StrSpan sigSpan = instructions.signature(li);
                const ScratchString newSig =
                    replaceKeysScratch(coreArena, sigSpan, replacementMap, replacementN);
                const int32_t em = rewritten.elemMark();
                const int32_t ec = instructions.elemCount(li);
                for (int32_t j = 0; j < ec; ++j) {
                    const StrSpan rawSpan = instructions.elemAt(li, j);
                    const ScratchString elem =
                        replaceKeysScratch(coreArena, rawSpan, replacementMap, replacementN);
                    const StrSpan elemSpan(elem);

                    const bool isControl =
                        (elemSpan.len >= 12 && std::memcmp(elemSpan.ptr, "(implication", 12) == 0) ||
                        (elemSpan.len >= 10 && std::memcmp(elemSpan.ptr, "(existence", 10) == 0) ||
                        (elemSpan.len >= 4  && std::memcmp(elemSpan.ptr, "(and", 4) == 0);

                    if (isControl) {
                        StrSpan args[ExecutionParameters::MAX_ARITY];
                        const int32_t argN = getArgsSpans(elemSpan, args,
                                                          ExecutionParameters::MAX_ARITY);
                        // [MODIFIED CHECK] any repeated arg?
                        bool hasRepeat = false;
                        for (int32_t a = 0; a < argN && !hasRepeat; ++a)
                            for (int32_t b = a + 1; b < argN; ++b)
                                if (equalSpans(args[a], args[b])) { hasRepeat = true; break; }

                        if (hasRepeat) {
                            // A variable repeated HERE but not in the allowed set fails.
                            bool unauthorizedRepetition = false;
                            for (int32_t a = 0; a < argN && !unauthorizedRepetition; ++a) {
                                int32_t occ = 0; bool first = true;
                                for (int32_t b = 0; b < argN; ++b) {
                                    if (equalSpans(args[a], args[b])) { ++occ; if (b < a) first = false; }
                                }
                                if (occ > 1 && first
                                    && !spanSetContains(allowedRepeatedVars, allowedN, args[a])) {
                                    unauthorizedRepetition = true;
                                }
                            }

                            if (unauthorizedRepetition) {
                                return;
                            }
                        }
                    }
                    rewritten.addElement(canonicalFields
                        ? StrSpan(canonicalTemplates.canonicalAdmissionText(mb, elemSpan, canonicalScope, coreArena, true)) : elemSpan);
                }
                const StrSpan admittedSignature = canonicalFields
                    ? StrSpan(canonicalTemplates.canonicalAdmissionText(mb, StrSpan(newSig), canonicalScope, coreArena, true)) : StrSpan(newSig);
                rewritten.commitEntity(instructions.category(li), admittedSignature,
                    instructions.definedSet(li), instructions.arity(li), em);
            }

            // An existence part whose fixed witness is, after the canonical
            // rewrite, also one of its own arguments: the integration went the
            // wrong way and stops here, before anything is filed
            // (D-590).
            if (witnessDoublesAsArgument(rewritten)) return;

            // A nested OR entity is prepared only through its maximal
            // contiguous OR root. Validate every integration-eligible OR
            // through the shared flattener first so a cyclic graph cannot hide
            // by marking every member as somebody else's child.
            assert(rewritten.entityCount()
                       <= ExecutionParameters::MAX_INSTRUCTION_ELEMENTS
                && "prepareIntegrationCore2: entity count exceeds fixed OR-root table");
            bool integrationOrEntity[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS] = {};
            bool nestedIntegrationOrEntity[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS] = {};
            for (int32_t li = 0; li < rewritten.entityCount(); ++li) {
                if (!equalSpans(rewritten.category(li), StrSpan("or", 2))) continue;

                StrSpan args[ExecutionParameters::MAX_ARITY];
                const int32_t argN = getArgsSpans(
                    rewritten.signature(li), args, ExecutionParameters::MAX_ARITY);
                bool allArgsAreU = true;
                for (int32_t a = 0; a < argN; ++a) {
                    if (!(args[a].len >= 2
                          && args[a].ptr[0] == 'u'
                          && args[a].ptr[1] == '_')) {
                        allArgsAreU = false;
                        break;
                    }
                }
                if (!allArgsAreU) continue;

                integrationOrEntity[li] = true;
                StrSpan validationLeaves[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
                (void)disintegrationProcessor.flattenOrLeaves(
                    rewritten, li, validationLeaves,
                    ExecutionParameters::MAX_INSTRUCTION_ELEMENTS);
            }

            for (int32_t parent = 0; parent < rewritten.entityCount(); ++parent) {
                if (!integrationOrEntity[parent]) continue;
                for (int32_t ei = 0; ei < rewritten.elemCount(parent); ++ei) {
                    const StrSpan element = rewritten.elemAt(parent, ei);
                    for (int32_t child = 0; child < rewritten.entityCount(); ++child) {
                        if (equalSpans(rewritten.signature(child), element)
                            && equalSpans(rewritten.category(child), StrSpan("or", 2))) {
                            nestedIntegrationOrEntity[child] = true;
                        }
                    }
                }
            }

            // The two-marker start (D-586): a
            // fresh preparation of a `marker1` / `marker2` goal, whose two
            // changeables are open under their `pi_lev` names. Each open
            // argument is a path; a path's keys put `marker` on its argument,
            // and its instruction copies carry the path label, the marked goal
            // with `marker1` on that argument.
            StrSpan twoMarkerOpen[2];
            if (!continuation) {
                for (int32_t r = 0; r < replacementN; ++r) {
                    if (equalSpans(replacementMap[r].key, StrSpan("marker1", 7))) twoMarkerOpen[0] = replacementMap[r].value;
                    if (equalSpans(replacementMap[r].key, StrSpan("marker2", 7))) twoMarkerOpen[1] = replacementMap[r].value;
                }
            }
            const bool twoMarkerStart = twoMarkerOpen[0].len > 0 && twoMarkerOpen[1].len > 0;
            const StrReplacement labelSwap[2] = {
                { StrSpan("marker1", 7), StrSpan("marker2", 7) },
                { StrSpan("marker2", 7), StrSpan("marker1", 7) } };
            const ScratchString secondPathLabel = twoMarkerStart
                ? replaceKeysScratch(coreArena, rewritten.markedGoal(), labelSwap, 2) : ScratchString();
            const auto pathLabel = [&](StrSpan marked) -> StrSpan {
                return twoMarkerStart && equalSpans(marked, twoMarkerOpen[1])
                    ? StrSpan(secondPathLabel) : rewritten.markedGoal();
            };

            // File one Case C key: the instruction copy with @p copyPair applied
            // throughout, labelled @p label, under @p keyString at this scope,
            // then wake the parks it can admit. A wildcard key (`pi_` at the
            // other open argument) names no park text, so its own scope and
            // its ancestors are woken by predicate. A preparation split by
            // premise files the copy once per forbidden-set pair
            // (D-579).
            const auto fileIntegrationKey = [&](StrSpan keyString, const StrReplacement& copyPair,
                                                StrSpan label, bool wildcard) {
                // Where an induction hypothesis has an instance, every key is
                // connected to the induction: at every scope of a weak step
                // block, at an induction scope and below in a strong one
                // (D-583,
                // D-584). A preparation split
                // by premise belongs to a weak install, so no int scope starts
                // from a singleton (D-578).
                assert((splitN == 0 || mb.weakStepBlock)
                    && "prepareIntegrationCore2: a premise split belongs to a weak step block");
                // A copy bound to the second twin premise is connected through
                // the twin partner's predecessor and the partner
                // (D-589).
                assert((splitN <= 1 || mb.twinSecondLevel == 0)
                    && "prepareIntegrationCore2: a twin block files no identical-goal group");
                const int32_t copyLevel = splitN > 0 ? splitSets[0].premiseLevel : rewritten.forbidden.premiseLevel;
                if (admissionMemory.stepKeyRuleBinds(mb, StrSpan(validityName))
                        && !admissionMemory.stepKeyConnected(mb, keyString, StrSpan(validityName), copyLevel)) return;
                // Outside the step blocks every key is connected to its goal:
                // it names a head variable of the goal or a connected value
                // (D-593).
                if (!mb.weakStepBlock && !mb.strongStepBlock
                        && !admissionMemory.goalKeyConnected(mb, keyString, rewritten.markedGoal(), canonicalScope)) return;
                // Build the marker-rewritten copy for the admission map.
                WorkInstruction markedCopy(rewritten.arena);
                markedCopy.setMarkedGoal(label);
                markedCopy.setPathVariables(rewritten.pathVariables());
                markedCopy.forbidden = rewritten.forbidden;
                for (int32_t mi = 0; mi < rewritten.entityCount(); ++mi) {
                    const StrSpan _mSigSpan = rewritten.signature(mi);
                    const ScratchString mSig =
                        replaceKeysScratch(coreArena, _mSigSpan, &copyPair, 1);
                    const int32_t mem = markedCopy.elemMark();
                    const int32_t _mec = rewritten.elemCount(mi);
                    for (int32_t _mj = 0; _mj < _mec; ++_mj) {
                        const StrSpan _me = rewritten.elemAt(mi, _mj);
                        const ScratchString e =
                            replaceKeysScratch(coreArena, _me, &copyPair, 1);
                        markedCopy.addElement(StrSpan(e));
                    }
                    markedCopy.commitEntity(rewritten.category(mi), StrSpan(mSig),
                        rewritten.definedSet(mi), rewritten.arity(mi), mem);
                }

                // Ensure the instruction entry exists (empty payload if
                // new) at this admission key.
                {
                    const int64_t admPk = mintTemplateKey(mb.templateInterner,
                        mb.nameMap, keyString, StrSpan(validityName));
                    const unsigned aimSlot = (gl::WorkerContext::coreId() >= 0)
                        ? static_cast<unsigned>(gl::WorkerContext::coreId())
                        : genScratchArenas().slotCount() - 1;
                    ArenaIntegrationMap admEntries(
                        &genScratchArenas().forSlot(aimSlot), &mb.valueInterner);
                    admEntries.buildFromCold(
                        mb.overallHashMemory.admissionMapIntegration, admPk);
                    // Flatten markedCopy into a byte-bump run (minting all
                    // fields unconditionally in encodeWorkInstruction order,
                    // markedGoal LAST) then find-or-emplace the run — no heap
                    // IntInstruction.
                    const int32_t previousInstructionCount = admEntries.entries.size();
                    if (splitN == 0) {
                        admEntries.findOrEmplaceWork(markedCopy, mb.valueInterner);
                    } else {
                        for (int32_t s = 0; s < splitN; ++s) {
                            markedCopy.forbidden = splitSets[s];
                            admEntries.findOrEmplaceWork(markedCopy, mb.valueInterner);
                        }
                    }
                    if (admEntries.entries.size() != previousInstructionCount) {
                        admEntries.writeToCold(mb.overallHashMemory.admissionMapIntegration, admPk);
                        mb.lbMemory.canonicalUses.integrationDemands.changed(admPk);
                    }
                }

                // Populate varsInAdmissionMapIntegrationKeys cache —
                // bare-form (u_-stripped) non-marker args. Used by
                // applyEquivalenceClassToAdmissionMapIntegration to
                // short-circuit when an equi-class has no overlap
                // with any stored admission-integration key, and by
                // integrationWildcardFiled (the `pi_` token).
                {
                    const ScratchString bareKeyForCacheS =
                        gl::DefinitionExpansion::removeUPrefixScratch(coreArena, keyString);
                    StrSpan bkArgs[ExecutionParameters::MAX_ARITY];
                    const int32_t bkArgN = getArgsSpans(StrSpan(bareKeyForCacheS),
                                                        bkArgs, ExecutionParameters::MAX_ARITY);
                    for (int32_t a = 0; a < bkArgN; ++a) {
                        if (!(bkArgs[a].len == 6 && std::memcmp(bkArgs[a].ptr, "marker", 6) == 0)) {
                            mb.overallHashMemory.varsInAdmissionMapIntegrationKeys.mint(mb.templateInterner.encode(bkArgs[a]));
                        }
                    }
                }

                // New admission key — any prior integration-side
                // rejection keyed at this marker form can now be
                // revived. rejectedMapIntegration keys are bare
                // concrete (no u_ prefix), while keyString here is
                // u_-prefixed (built from le.signature which is in
                // u_ form). Strip u_ before lookup.
                {
                    const ScratchString bareKeyS =
                        gl::DefinitionExpansion::removeUPrefixScratch(coreArena, keyString);
                    if (wildcard)
                        rejectedSourceReplay.wakeWildcardIntegrationParks(mb, StrSpan(bareKeyS), validityName);
                    else
                        rejectedSourceReplay.revisitRejectedIntegration2(StrSpan(bareKeyS), mb, validityName);
                    // The key is visible below its scope too
                    // (D-572): wake
                    // the parks there.
                    rejectedSourceReplay.wakeDescendantIntegrationParks(mb, StrSpan(bareKeyS), validityName);
                }
            };

            // 2. Analyze Instructions for Admission/Memory
            for (int32_t li = 0; li < rewritten.entityCount(); ++li)
            {
                const StrSpan leSignatureSpan = rewritten.signature(li);
                const StrSpan leCategorySpan = rewritten.category(li);
                // leElements: span run into the (stable, read-only) `rewritten`
                // WorkInstruction; the addToHashMemory / makeAdmissionKeys edges
                // materialize std::string / std::vector from these spans (J1
                // ruling — heap only AT those boundaries; findAllUArgs reads the
                // spans directly).
                StrSpan leElements[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
                int32_t leElemN = 0;
                {
                    const int32_t _ec = rewritten.elemCount(li);
                    assert(_ec <= ExecutionParameters::MAX_INSTRUCTION_ELEMENTS
                        && "prepareIntegrationCore2: element count exceeds MAX_INSTRUCTION_ELEMENTS");
                    for (int32_t _j = 0; _j < _ec; ++_j)
                        leElements[leElemN++] = rewritten.elemAt(li, _j);
                }

                StrSpan sigArgs[ExecutionParameters::MAX_ARITY];
                const int32_t sigArgN = getArgsSpans(leSignatureSpan, sigArgs,
                                                     ExecutionParameters::MAX_ARITY);
                bool allSigArgsAreU = true;
                for (int32_t a = 0; a < sigArgN; ++a) {
                    if (!(sigArgs[a].len >= 2 && sigArgs[a].ptr[0] == 'u' && sigArgs[a].ptr[1] == '_')) {
                        allSigArgsAreU = false;
                        break;
                    }
                }

                // Case A: Implication with all u_ args — opened (or found
                // known / already prepared) by openImplicationSubproof. An
                // implication entity has nothing left for Cases OR, B and C
                // unless it was opened, exactly as before the extraction.
                if (equalSpans(leCategorySpan, StrSpan("implication", 11)) && allSigArgsAreU) {
                    if (implicationSubproofs.openImplicationSubproof(rewritten, li, mb, validityName, rootGoal,
                            continuation) != ImplicationSubproofOutcome::Opened)
                        continue;
                }

                // Case OR: disjunction with all u_ args — prove by case split.
                // Each disjunct a_k becomes a branch scope where ¬a_j (j != k) are
                // status=0 premises and a_k is the status=2 proof goal. Branch
                // validity carries the self-describing "orint_" marker so any
                // downstream consumer (emission, sibling cleanup) can classify
                // the scope from the NameMap stack alone.
                if (equalSpans(leCategorySpan, StrSpan("or", 2)) && allSigArgsAreU) {
                    if (nestedIntegrationOrEntity[li]) continue;

                    const ScratchString cleanSignatureS =
                        gl::DefinitionExpansion::removeUPrefixScratch(coreArena, leSignatureSpan);
                    // Goal-qualified prep gate on MAIN (Case A's rule; the
                    // branch payloads below carry the same prefix).
                    const ScratchString orGateS = embedGoal
                        ? concatCore({ rootGoal, StrSpan("_subproof_", 10),
                                       StrSpan(cleanSignatureS) })
                        : ScratchString();
                    const StrSpan orGate = embedGoal
                        ? StrSpan(orGateS) : StrSpan(cleanSignatureS);

                    {
                        const StatementFlags* orSigRow = lookupStatementFlags(
                            mb.intKnownStatements, mb.nameMap, StrSpan(cleanSignatureS), StrSpan(validityName));
                        if (orSigRow != nullptr) continue;
                    }
                    if (continuation
                            ? canonicalTemplates.canonicalTemplatePresentAtOrAbove(mb, mb.integrationPrepared,
                                mb.lbMemory.canonicalUses.preparedMatches, orGate, StrSpan(validityName))
                            : canonicalTemplates.canonicalTemplatePresent(mb, mb.integrationPrepared,
                                mb.lbMemory.canonicalUses.preparedMatches, orGate, StrSpan(validityName),
                                mb.nameMap.lookup(validityName))) {
                        continue;
                    }
                    const int64_t orGateKey = mintTemplateKey(
                        mb.templateInterner, mb.nameMap, orGate, StrSpan(validityName));
                    mb.integrationPrepared.mint(orGateKey);

                    StrSpan flattenedElements[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
                    const int32_t flattenedElemN = disintegrationProcessor.flattenOrLeaves(
                        rewritten, li, flattenedElements,
                        ExecutionParameters::MAX_INSTRUCTION_ELEMENTS);
                    StrSpan renamedElements[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
                    const int32_t renamedElemN = integrationBuilder.renamingChain2Scratch(
                        coreArena, flattenedElements, flattenedElemN, mb,
                        renamedElements, ExecutionParameters::MAX_INSTRUCTION_ELEMENTS);

                    NameId parentValId = mb.nameMap.encode(validityName);
                    for (int32_t k = 0; k < renamedElemN; ++k) {
                        const StrSpan head = renamedElements[k];
                        const ScratchString branchPayloadS = embedGoal
                            ? concatCore(
                                { rootGoal, StrSpan("_subproof_", 10),
                                  StrSpan("orint_", 6), StrSpan(cleanSignatureS),
                                  StrSpan("_(", 2), head, StrSpan(")", 1) })
                            : concatCore(
                                { StrSpan("orint_", 6), StrSpan(cleanSignatureS),
                                  StrSpan("_(", 2), head, StrSpan(")", 1) });
                        NameId branchValId = mb.nameMap.encodePush(parentValId, StrSpan(branchPayloadS));
                        canonicalizer.initializeCanonicalScope(mb, branchValId);
                        // copyFrom onto coreArena (a DIFFERENT arena than NameMap):
                        // nested addExprToMemoryBlock calls below mint NameMap, so a
                        // raw decodeView span would dangle (I-3); the copy survives
                        // every mint and rides below every nested scope's mark.
                        const ScratchString branchValidity = ScratchString::copyFrom(
                            coreArena, mb.nameMap.decodeView(branchValId).ptr,
                            mb.nameMap.decodeView(branchValId).len);

                        // Goal-expansion origin (D-52).
                        // The OR-integration goal `(or<N>)_integration_goal` at the
                        // OR's parent scope is rewritten as K sub-implications
                        // `(!D_others -> D_k)` (one per chosen branch), each at the
                        // SAME parent scope. The sub-implication's premise is the
                        // (left-nested) AND of negated other-disjuncts; its head is
                        // the chosen disjunct. For K=2 (or2): premise = single
                        // negation `!D_other`, no AND wrapper. For K>=3: AND of
                        // negations, left-nested in the same shape that
                        // `expandSignatureForIntegration::and` produces.
                        //
                        // Pre-D-52 the producer emitted a malformed history line:
                        // KEY = bare disjunct `head` at branch validity, ORIGIN =
                        // `expansion for integration ← (or)_integration_goal` at
                        // parent validity. That row was rejected by the verifier
                        // on TWO grounds — (a) namespace mismatch (line.namespace
                        // != right_ns), and (b) bare disjunct is not the structural
                        // expansion of `or<N>` per `_try_expand`. The new emission
                        // fixes both: KEY and ORIGIN both at parent validity, and
                        // LEFT side is a sub-implication shape that the verifier's
                        // extended `_try_expand` accepts as one of K valid OR
                        // expansions. See `docs/agentic_swdd/20_core_concepts/07_or_branching.md`
                        // §3a `_orint_` and `docs/agentic_swdd/40_decisions.md` D-52.
                        if (parameters.trackHistory) {
                            // Build the AND-of-negations premise for branch k.
                            StrSpan negOthers[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
                            int32_t negOthersN = 0;
                            for (int32_t j = 0; j < renamedElemN; ++j) {
                                if (j == k) continue;
                                negOthers[negOthersN++] = gl::DefinitionExpansion::negateScratch(coreArena, renamedElements[j]);
                            }
                            StrSpan premise;
                            ScratchString premiseHold;
                            if (negOthersN == 1) {
                                premise = negOthers[0];
                            } else {
                                premiseHold = ScratchString::copyFrom(coreArena,
                                    negOthers[0].ptr, negOthers[0].len);
                                for (int32_t i = 1; i < negOthersN; ++i) {
                                    premiseHold = concatCore({ StrSpan("(&", 2), StrSpan(premiseHold),
                                                               negOthers[i], StrSpan(")", 1) });
                                }
                                premise = StrSpan(premiseHold);
                            }
                            // Sub-implication: `(>[]premise(head))` — empty
                            // bound-var list since the orint branch sub-implication
                            // doesn't introduce new quantifiers.
                            const ScratchString subImplS = concatCore(
                                { StrSpan("(>[]", 4), premise, head, StrSpan(")", 1) });
                            const StrSpan subImpl(subImplS);

                            // L3 span-record doors. KEY (subImpl) + antecedent
                            // (cleanSignature) both take the "_integration_goal"
                            // suffix, built on the per-slot string-scratch arena
                            // (explicit length). validityName is a stable local;
                            // the doors mint into originInterner, not this arena
                            // (I-3).
                            const unsigned ogSlot = (gl::WorkerContext::coreId() >= 0)
                                ? static_cast<unsigned>(gl::WorkerContext::coreId())
                                : scratchArenas().slotCount() - 1;
                            ScratchArena& ogArena = scratchArenas().forSlot(ogSlot);
                            ScratchScope ogScope(ogArena);
                            const StrSpan OG_GOAL("_integration_goal",
                                static_cast<int32_t>(sizeof("_integration_goal") - 1));
                            const auto ogGoalOf = [&](const StrSpan base) -> StrSpan {
                                const int32_t len = base.len + OG_GOAL.len;
                                char* buf = ogArena.allocBytes(len);
                                std::memcpy(buf, base.ptr,
                                            static_cast<std::size_t>(base.len));
                                std::memcpy(buf + base.len, OG_GOAL.ptr,
                                            static_cast<std::size_t>(OG_GOAL.len));
                                return StrSpan(buf, len);
                            };
                            const StrSpan ogKey = ogGoalOf(subImpl);
                            const OriginDep ogDeps[1] = {
                                { ogGoalOf(StrSpan(cleanSignatureS)), StrSpan(validityName) } };
                            const int ogCap = (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr);
                            addOriginEncoded(mb.exprOriginMap, mb.originInterner, ogKey, StrSpan(validityName), OriginTag::expansionForIntegration, ogDeps, 1, ogCap);
                            mb.addMailOutOrigin(ogKey, StrSpan(validityName), OriginTag::expansionForIntegration, ogDeps, 1, ogCap);
                        }

                        // Assumptions: ¬a_j for every j != k at branch scope, status=0.
                        for (int32_t j = 0; j < renamedElemN; ++j) {
                            if (j == k) continue;

                            const unsigned oaSlot = (gl::WorkerContext::coreId() >= 0)
                                ? static_cast<unsigned>(gl::WorkerContext::coreId())
                                : scratchArenas().slotCount() - 1;
                            ScratchArena& oaArena = scratchArenas().forSlot(oaSlot);
                            ScratchScope oaScope(oaArena);

                            const StrSpan negAssumption = gl::DefinitionExpansion::negateScratch(oaArena, renamedElements[j]);

                            OriginDep oaDeps[1];
                            TransientOrigin originAssumption{};
                            if (parameters.trackHistory) {
                                // antecedent = cleanSignature + "_integration_goal",
                                // built on the per-slot scratch arena (explicit length).
                                // The scope spans the addExprToMemoryBlock pass-through
                                // below, whose span door mints the antecedent into
                                // originInterner, never this arena (I-3).
                                const StrSpan OA_GOAL("_integration_goal",
                                    static_cast<int32_t>(sizeof("_integration_goal") - 1));
                                const int32_t antLen = cleanSignatureS.size() + OA_GOAL.len;
                                char* antBuf = oaArena.allocBytes(antLen);
                                std::memcpy(antBuf, cleanSignatureS.data(),
                                            static_cast<std::size_t>(cleanSignatureS.size()));
                                std::memcpy(antBuf + cleanSignatureS.size(),
                                            OA_GOAL.ptr, static_cast<std::size_t>(OA_GOAL.len));
                                oaDeps[0] = { StrSpan(antBuf, antLen), StrSpan(validityName) };
                                originAssumption =
                                    TransientOrigin{ true, OriginTag::orBranchAssumption, oaDeps, 1 };

                                const int oaCap = (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr);
                                addOriginEncoded(mb.exprOriginMap, mb.originInterner, negAssumption, StrSpan(branchValidity), OriginTag::orBranchAssumption, oaDeps, 1, oaCap);
                                mb.addMailOutOrigin(negAssumption, StrSpan(branchValidity), OriginTag::orBranchAssumption, oaDeps, 1, oaCap);
                            }

                            // Stamp with mb.level so the branch seed contributes
                            // its LB's level to downstream combinedLevels; an
                            // empty set here left branch-rooted inferences with
                            // level {0} (from anchor premises), mismatching the
                            // LB level at the equality-substitution gate.
                            const int seedLevels[1] = { mb.level };
                            // The assumption is the premise of the branch's
                            // implication `!D_j -> D_k`: an integration premise
                            // (D-460).
                            statementMemory.addExprToMemoryBlock(negAssumption, mb, -1, 0, seedLevels, 1,
                                originAssumption, -1, -1, StrSpan(branchValidity), false,
                                /*allowOrDisintegration=*/false, /*registeredOut=*/nullptr,
                                /*recursionHead=*/false, /*convergenceMailOut=*/nullptr,
                                /*replayKinds=*/0, /*replayIntegrationDepth=*/-1,
                                /*integrationPremise=*/true);
                        }

                        // Head as toBeProved at branch scope (status=2).
                        const int headLevels[1] = { mb.level };
                        statementMemory.addExprToMemoryBlock(head, mb, -1, 2, headLevels, 1,
                            TransientOrigin{},
                            -1, -1, StrSpan(branchValidity), false);
                    }
                }

                // Case B: Existence or And with all u_ args
                if ((equalSpans(leCategorySpan, StrSpan("existence", 9))
                     || equalSpans(leCategorySpan, StrSpan("and", 3))) && allSigArgsAreU) {

                    //test
                    if (true) {
                        StrSpan bArgs[ExecutionParameters::MAX_ARITY];
                        const int32_t bArgN = getArgsSpans(leSignatureSpan, bArgs,
                                                           ExecutionParameters::MAX_ARITY);
                        bool bRepeat = false;
                        for (int32_t a = 0; a < bArgN && !bRepeat; ++a)
                            for (int32_t b = a + 1; b < bArgN; ++b)
                                if (equalSpans(bArgs[a], bArgs[b])) { bRepeat = true; break; }
                        if (bRepeat) {
                            //continue;
                        }
                    }

                    const ScratchString cleanSignatureS =
                        gl::DefinitionExpansion::removeUPrefixScratch(coreArena, leSignatureSpan);
                    if (continuation
                            ? canonicalTemplates.canonicalTemplatePresentAtOrAbove(mb, mb.integrationPrepared,
                                mb.lbMemory.canonicalUses.preparedMatches, StrSpan(cleanSignatureS), StrSpan(validityName))
                            : canonicalTemplates.canonicalTemplatePresent(mb, mb.integrationPrepared,
                                mb.lbMemory.canonicalUses.preparedMatches, StrSpan(cleanSignatureS), StrSpan(validityName),
                                mb.nameMap.lookup(validityName)))
                    {
                        continue;
                    }

                    const int64_t entityGateKey = mintTemplateKey(
                        mb.templateInterner, mb.nameMap, StrSpan(cleanSignatureS), StrSpan(validityName));
                    mb.integrationPrepared.mint(entityGateKey);

                    if (equalSpans(leCategorySpan, StrSpan("existence", 9)))
                    {
                        // Decoded-lex INDEX over the originals rule registry (the
                        // shared sortOriginalChainIndex, byte-bump int32[] on
                        // genScratchArenas) reproduces the former originalChains
                        // snapshot + std::sort order; makeAdmissionKeys writes
                        // admission state, so the order is observable (I-84).
                        // makeAdmissionKeys never writes `originals`, so the index is
                        // stable across the whole elements loop AND the chain key
                        // bytes stay resident -- each chain reads its rule ids
                        // zero-copy (IdVecKeyView) and decodeViews them into a stack
                        // StrSpan run (0% heap); markedElem is a coreArena
                        // ScratchString, passed as a span.
                        const int32_t chainCount = mb.overallHashMemory.originals.count();
                        const unsigned ocSlot = (gl::WorkerContext::coreId() >= 0)
                            ? static_cast<unsigned>(gl::WorkerContext::coreId())
                            : genScratchArenas().slotCount() - 1;
                        ScratchArena& ocArena = genScratchArenas().forSlot(ocSlot);
                        ScratchScope ocScope(ocArena);
                        int32_t* chainIdx = (chainCount == 0) ? nullptr
                            : reinterpret_cast<int32_t*>(ocArena.resolve(ocArena.alloc(
                                  chainCount * static_cast<int32_t>(sizeof(int32_t)),
                                  static_cast<int32_t>(alignof(int32_t)))));
                        const int32_t chainN = sortOriginalChainIndex(mb, chainIdx, chainCount);
                        for (int32_t ei = 0; ei < leElemN; ++ei) {
                            const ScratchString markedElemS = makeMarkedExprLambda(leElements[ei]);

                            for (int32_t ci = 0; ci < chainN; ++ci)
                            {
                                // Read the chain key's rule ids zero-copy (IdVecKeyView
                                // over inner().keyAt) and decodeView each into a stack
                                // StrSpan run. makeAdmissionKeys mints templateInterner /
                                // nameMap, never ruleInterner or `originals`, so both the
                                // keyAt bytes AND the ruleInterner.decodeView spans stay
                                // valid across the call (I-3, different interner) -- the
                                // former owned .toStdString() copies were conservative.
                                const StrSpan kb = mb.overallHashMemory.originals
                                    .inner().keyAt(chainIdx[ci]);
                                const IdVecKeyView kv = viewIdVecKey(kb);
                                const int32_t implN = kv.count();
                                assert(implN <= ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS + 1
                                    && "prepareIntegrationCore2: implication chain exceeds cap");
                                StrSpan implRun[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS + 1];
                                for (int32_t t = 0; t < implN; ++t)
                                    implRun[t] = mb.ruleInterner.decodeView(kv.idAt(t));
                                admissionMemory.makeAdmissionKeys(implRun, implN, StrSpan(markedElemS),
                                    mb.overallHashMemory, validityName, mb);
                            }
                        }
                    }


                    // The entity's compiled integration set
                    // (D-503): its one-level rule
                    // — the witness bound, or the occupied form once a marker
                    // replay fixed it — with the owner tokens filled from the
                    // signature positionally (the witness last).
                    const LogicalEntity& integrationEntity = integrationEntityOf(leSignatureSpan,
                        leCategorySpan, StrSpan(cleanSignatureS), mb, validityName);
                    StrSpan owners[ExecutionParameters::MAX_ARITY];
                    int32_t ownerN = 0;
                    for (int32_t a = 0; a < sigArgN; ++a) owners[ownerN++] = sigArgs[a];
                    bool occupied = false;
                    if (equalSpans(leCategorySpan, StrSpan("existence", 9))) {
                        StrSpan witness;
                        occupied = occupiedWitness(integrationEntity, leElements, leElemN, witness);
                        if (occupied) {
                            assert(ownerN < ExecutionParameters::MAX_ARITY
                                && "prepareIntegrationCore2: an occupied witness exceeds the owner cap");
                            owners[ownerN++] = witness;
                        }
                    }
                    const IntegrationImplication& oneLevelRecord =
                        integrationEntity.integrationImplications[occupied ? 1u : 0u];
                    assert(oneLevelRecord.kind
                            == (occupied ? IntegrationRuleKind::Occupied : IntegrationRuleKind::OneLevel)
                        && "prepareIntegrationCore2: the integration set does not start with the one-level rules");
                    const ScratchString oneLevelU = integrationBuilder.fillIntegrationCompactScratch(coreArena,
                        StrSpan(oneLevelRecord.compact), owners, ownerN);
                    const OriginTag instrTag = equalSpans(leCategorySpan, StrSpan("and", 3))
                        ? OriginTag::reformulationForIntegrationAnd
                        : (occupied ? OriginTag::reformulationForIntegrationBracket
                                    : OriginTag::reformulationForIntegrationBound);

                    // History: the expansion-for-integration row (KEY
                    // expandedSignature + goal, antecedent cleanSignature +
                    // goal), which the compact cites. Both spans ride
                    // coreArena; the doors mint into originInterner (I-3).
                    OriginDep compactDeps[1];
                    int32_t compactDepN = 0;
                    if (parameters.trackHistory)
                    {
                        ScratchString expandedSignatureS;
                        bool hasPiBoundVars = false;
                        if (equalSpans(leCategorySpan, StrSpan("existence", 9)))
                        {
                            expandedSignatureS = integrationBuilder.expandSignatureForIntegrationScratch(coreArena, StrSpan("existence", 9), leElements, leElemN, StrSpan(cleanSignatureS), &hasPiBoundVars);
                            assert(hasPiBoundVars == !occupied
                                && "prepareIntegrationCore2: the witness is bound exactly when it is not fixed");
                        }
                        if (equalSpans(leCategorySpan, StrSpan("and", 3)))
                        {
                            expandedSignatureS = integrationBuilder.expandSignatureForIntegrationScratch(coreArena, StrSpan("and", 3), leElements, leElemN, StrSpan(cleanSignatureS));
                        }
                        const StrSpan IG_GOAL("_integration_goal",
                            static_cast<int32_t>(sizeof("_integration_goal") - 1));
                        const auto igGoalOf = [&](const StrSpan base) -> StrSpan {
                            const int32_t len = base.len + IG_GOAL.len;
                            char* buf = coreArena.allocBytes(len);
                            std::memcpy(buf, base.ptr,
                                        static_cast<std::size_t>(base.len));
                            std::memcpy(buf + base.len, IG_GOAL.ptr,
                                        static_cast<std::size_t>(IG_GOAL.len));
                            return StrSpan(buf, len);
                        };
                        const int igCap = (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr);
                        const StrSpan vSpan(validityName);
                        const StrSpan expGoalSpan = igGoalOf(StrSpan(expandedSignatureS));
                        const OriginDep exDeps[1] = {
                            { igGoalOf(StrSpan(cleanSignatureS)), vSpan } };
                        addOriginEncoded(mb.exprOriginMap, mb.originInterner, expGoalSpan, vSpan, OriginTag::expansionForIntegration, exDeps, 1, igCap);
                        mb.addMailOutOrigin(expGoalSpan, vSpan, OriginTag::expansionForIntegration, exDeps, 1, igCap);
                        compactDeps[0] = OriginDep{ expGoalSpan, vSpan };
                        compactDepN = 1;
                    }
                    {
                        StrSpan oneLevelPremises[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
                        int32_t oneLevelPremiseN = 0;
                        StrSpan oneLevelHead;
                        compactRuleProcessor.installIntegrationCompact(mb, StrSpan(oneLevelU), validityName,
                            instrTag, compactDeps, compactDepN, coreArena, oneLevelPremises,
                            ExecutionParameters::MAX_INSTRUCTION_ELEMENTS, oneLevelPremiseN, oneLevelHead);
                    }

                    // The root family — only for the ROOT integration goal (the
                    // expression handed to prepareIntegration): its u_-stripped
                    // signature is the instruction's marked goal. Nested
                    // entities and the marker replays never match
                    // (D-352).
                    if (equalSpans(StrSpan(cleanSignatureS), rewritten.markedGoal())) {
                        assert(!occupied && "prepareIntegrationCore2: a root integration goal with its witness fixed");
                        installRootIntegrationFamily(integrationEntity, owners, ownerN,
                            StrSpan(gl::DefinitionExpansion::removeUPrefixScratch(coreArena, StrSpan(oneLevelU))),
                            StrSpan(cleanSignatureS), mb, validityName, coreArena);
                    }
                }

                // Case C: Marker Logic (Admission)

                // C1. Check Signature (Only for "atomic")
                //if (le.category == "atomic")
                {
                    {
                        const StrSpan mgSpan = rewritten.markedGoal();
                        if (equalSpans(extractExpressionUniversalSpan(leSignatureSpan),
                                       extractExpressionUniversalSpan(mgSpan)))
                        {
                            continue;
                        }
                    }

                    StrSpan sArgs[ExecutionParameters::MAX_ARITY];
                    const int32_t sArgN = getArgsSpans(leSignatureSpan, sArgs,
                                                       ExecutionParameters::MAX_ARITY);
                    StrSpan nonUArg;
                    int countNonU = 0;

                    for (int32_t a = 0; a < sArgN; ++a) {
                        if (!(sArgs[a].len >= 2 && sArgs[a].ptr[0] == 'u' && sArgs[a].ptr[1] == '_')) {
                            nonUArg = sArgs[a];
                            countNonU++;
                        }
                    }

                    const StrSpan coreU = extractExpressionUniversalSpan(leSignatureSpan);
                    // Two-input predicates without outputs remain constituents
                    // but cannot key replay, except positive equality.
                    const bool keyEligible = admissionPolicy.isIntegrationInstructionKeyEligible(leSignatureSpan)
                        && this->definitions.configuredNames().count(
                            std::string_view(coreU.ptr, static_cast<std::size_t>(coreU.len))) > 0;
                    if (keyEligible && countNonU == 1)
                    {
                        const StrReplacement markerPair[1] = { { nonUArg, StrSpan("marker", 6) } };
                        const ScratchString keyStringS =
                            replaceKeysScratch(coreArena, leSignatureSpan, markerPair, 1);
                        fileIntegrationKey(StrSpan(keyStringS), markerPair[0], pathLabel(nonUArg), false);
                    }
                    // A part holding both open arguments of the two-marker
                    // start keys both paths: the path's argument becomes
                    // `marker`, the other one the wildcard `pi_`. The copy
                    // keeps the other argument open, to be filled after the
                    // marker (D-586).
                    else if (keyEligible && twoMarkerStart && countNonU == 2) {
                        bool both[2] = { false, false };
                        for (int32_t a = 0; a < sArgN; ++a)
                            for (int32_t p = 0; p < 2; ++p)
                                if (equalSpans(sArgs[a], twoMarkerOpen[p])) both[p] = true;
                        if (both[0] && both[1]) {
                            for (int32_t p = 0; p < 2; ++p) {
                                const StrReplacement keyPairs[2] = {
                                    { twoMarkerOpen[p], StrSpan("marker", 6) },
                                    { twoMarkerOpen[1 - p], StrSpan("pi_", 3) } };
                                const ScratchString keyStringS =
                                    replaceKeysScratch(coreArena, leSignatureSpan, keyPairs, 2);
                                fileIntegrationKey(StrSpan(keyStringS), keyPairs[0], pathLabel(twoMarkerOpen[p]), true);
                            }
                        }
                    }

                }

                // C2. Check Elements (Only for "existence" or "and")
                /*
                 if (le.category == "existence" || le.category == "and") {
                    for (const std::string& element : le.elements) {

                        // Exclusion: Do not process if the element itself is an implication expression
                        if (element.size() >= 12 && element.substr(0, 12) == "(implication") {
                            continue;
                        }

                        std::vector<std::string> eArgs = ce::getArgs(element);
                        std::string nonUArg;
                        int countNonU = 0;

                        for (const std::string& a : eArgs) {
                            if (!(a.size() >= 2 && a[0] == 'u' && a[1] == '_')) {
                                nonUArg = a;
                                countNonU++;
                            }
                        }

                        if (countNonU == 1) {
                            std::map<std::string, std::string> markerMap;
                            markerMap[nonUArg] = "marker";
                            std::string keyString = ce::replaceKeysInString(element, markerMap);

                            // Create copy for admission map
                            std::vector<LogicalEntity> instructionCopy = instructions;
                            for (LogicalEntity& inst : instructionCopy) {
                                inst.signature = ce::replaceKeysInString(inst.signature, markerMap);
                                for (std::string& elem : inst.elements) {
                                    elem = ce::replaceKeysInString(elem, markerMap);
                                }
                            }
                            mb.overallHashMemory.admissionMapIntegration[keyString].insert(instructionCopy);
                        }
                    }
                }
                */
            }
        }

//#pragma optimize("", off)

        /// @brief Sorted-run membership probe — `binary_search` under
        ///        `compareSpans` over a sorted-unique `StrSpan` run.
        ///
        /// @details
        /// The span twin of `std::set<std::string>::find != end()`:
        /// `std::set` iterates ascending byte-lex, which is exactly
        /// `compareSpans` order, so membership in the sorted run is
        /// byte-identical to membership in the former set. Used by the
        /// `prepareIntegration` span core for its three unchangeable-args
        /// membership tests.
        ///
        /// @pre `[sorted, sorted + n)` sorted ascending strictly-unique under
        ///      `compareSpans`.
        /// @param sorted The sorted-unique run.
        /// @param n      The run length (`>= 0`).
        /// @param s      The probe bytes.
        /// @return `true` iff `s` is in the run.
        /// @see prepareIntegration(StrSpan, const StrSpan*, int32_t, Memory&, StrSpan)
        ///      — the consumer; compareSpans.
inline bool IntegrationProcessor::spanSetContains(const StrSpan* sorted, int32_t n,
            const StrSpan& s) {
            return std::binary_search(sorted, sorted + n, s,
                [](const StrSpan& a, const StrSpan& b) {
                    return compareSpans(a, b) < 0;
                });
        }

/// @brief Whether a premise of the block already settles a marked
///        integration goal whose marked argument is uniquely definable
///        (D-567).
///
/// @details
/// The goal is @p expression with its one changeable argument, an argument
/// outside @p unchangeableArgsSorted, as the marker. It is settled when the
/// marked position is listed in the core's `uniquely_definable` and an
/// `exprKey` on the block's `parentMemory` chain, the block's own included,
/// is a version of the goal: the same core, positive like the goal, and at
/// every other position an argument equal to the goal's by equivalence class
/// at the goal's scope. Integrating such a goal could only rebuild the value
/// that premise already names.
///
/// Not settled: a negated goal, a goal with no or with more than one
/// changeable position, a core without a configuration (a compiled-only
/// definition such as an `existence<N>`), an untagged marked position. A
/// premise key that is not one positive expression (the root's empty key, a
/// negated or suffixed block key) is no version. An unregistered scope
/// compares at `main`, whose classes hold in every scope.
///
/// @param expression             The expression `prepareIntegration`
///                               received (span over a stable buffer).
/// @param unchangeableArgsSorted Its unchangeable arguments as a sorted-unique
///                               span run.
/// @param unchangeableArgN       The run length.
/// @param mb                     The owning block.
/// @param validityName           The goal's scope.
/// @return True iff a premise version settles the goal.
/// @invariant Allocates nothing; mints nothing beyond the canonical view
///            `Canonicalizer::initializeCanonicalScope` builds for a scope
///            seen for the first time.
/// @see prepareIntegration, CanonicalTemplateMatcher::canonicalArgumentAtScope.
inline bool IntegrationProcessor::premiseVersionSettlesMarkedGoal(StrSpan expression,
            const StrSpan* unchangeableArgsSorted, int32_t unchangeableArgN,
            Memory& mb, StrSpan validityName) {
            if (expression.len == 0 || expression.ptr[0] != '(') return false;
            StrSpan goalArgs[ExecutionParameters::MAX_ARITY];
            const int32_t argN = getArgsSpans(expression, goalArgs, ExecutionParameters::MAX_ARITY);
            int32_t marked = -1;
            for (int32_t i = 0; i < argN; ++i) {
                if (spanSetContains(unchangeableArgsSorted, unchangeableArgN, goalArgs[i])) continue;
                if (marked >= 0) return false;
                marked = i;
            }
            if (marked < 0) return false;

            const StrSpan core = extractExpressionUniversalSpan(expression);
            const ce::CoreExpressionConfig* cfg = this->definitions.coreConfig(core);
            if (cfg == nullptr) return false;
            if (std::find(cfg->uniquelyDefinableIndices.begin(), cfg->uniquelyDefinableIndices.end(), marked)
                == cfg->uniquelyDefinableIndices.end()) return false;

            const NameId registered = mb.nameMap.lookup(validityName);
            const NameId scope = registered != 0 ? registered : NameMap::MAIN_ID;
            canonicalizer.initializeCanonicalScope(mb, scope);
            for (const Memory* block = &mb; block != nullptr; block = block->parentMemory) {
                const StrSpan premise = block->exprKeyView();
                if (premise.len == 0 || premise.ptr[0] != '(' || premise.ptr[premise.len - 1] != ')') continue;
                if (!equalSpans(extractExpressionUniversalSpan(premise), core)) continue;
                StrSpan premiseArgs[ExecutionParameters::MAX_ARITY];
                const int32_t premiseN = getArgsSpans(premise, premiseArgs, ExecutionParameters::MAX_ARITY);
                assert(premiseN == argN
                    && "premiseVersionSettlesMarkedGoal: a premise with the goal's core has another arity");
                bool version = true;
                for (int32_t i = 0; i < argN && version; ++i) {
                    if (i == marked) continue;
                    version = equalSpans(canonicalTemplates.canonicalArgumentAtScope(mb, goalArgs[i], scope),
                        canonicalTemplates.canonicalArgumentAtScope(mb, premiseArgs[i], scope));
                }
                if (version) return true;
            }
            return false;
        }

/// @brief Whether a statement known at the firing scope or an ancestor
///        already settles a missing premise whose open argument is uniquely
///        definable (D-568).
///
/// @details
/// The premise is @p expression, the missing premise a marker rule's firing
/// queued for integration (`AdmissionMemory::drainDeferredIntegrationPreps`),
/// with its one open argument, the argument outside @p unchangeableArgsSorted,
/// as the marker. It is settled when the open position is listed in the
/// core's `uniquely_definable` and a statement known at the scope or at an
/// ancestor of it is a version of the premise: the same core, positive like
/// the premise, and at every other position an argument equal to the
/// premise's by equivalence class at the scope. That statement already names
/// the one object the open position denotes, so integrating the premise could
/// only rebuild it.
///
/// The candidates are read from the statement index's shape postings (the
/// known statements of one core owned at one scope), so only statements of
/// the premise's core are compared; a posting whose statement left
/// `intKnownStatements` is skipped, as every reader of that index does.
///
/// Not settled: a negated premise, a premise with no or with more than one
/// open position, a core without a configuration, an untagged open position,
/// a version known only at a descendant or in another branch of the scope
/// tree.
///
/// @param expression             The queued premise (span over a stable
///                               buffer).
/// @param unchangeableArgsSorted Its fixed arguments as a sorted-unique span
///                               run.
/// @param unchangeableArgN       The run length.
/// @param mb                     The owning block.
/// @param validityName           The firing scope, registered in the block's
///                               name map.
/// @return True iff a known version settles the premise.
/// @invariant Mints nothing into the name map; synchronizes only the derived
///            statement index (`canonicalUses.statementMatches`) and the
///            canonical view `Canonicalizer::initializeCanonicalScope` builds
///            for a scope seen for the first time.
/// @see premiseVersionSettlesMarkedGoal — the premise-chain twin;
///      AdmissionMemory::drainDeferredIntegrationPreps — the caller.
inline bool IntegrationProcessor::knownVersionSettlesMarkedPremise(StrSpan expression,
            const StrSpan* unchangeableArgsSorted, int32_t unchangeableArgN,
            Memory& mb, StrSpan validityName) {
            if (expression.len == 0 || expression.ptr[0] != '(') return false;
            StrSpan premiseArgs[ExecutionParameters::MAX_ARITY];
            const int32_t argN = getArgsSpans(expression, premiseArgs, ExecutionParameters::MAX_ARITY);
            int32_t open = -1;
            for (int32_t i = 0; i < argN; ++i) {
                if (spanSetContains(unchangeableArgsSorted, unchangeableArgN, premiseArgs[i])) continue;
                if (open >= 0) return false;
                open = i;
            }
            if (open < 0) return false;

            const StrSpan core = extractExpressionUniversalSpan(expression);
            const ce::CoreExpressionConfig* cfg = this->definitions.coreConfig(core);
            if (cfg == nullptr) return false;
            if (std::find(cfg->uniquelyDefinableIndices.begin(), cfg->uniquelyDefinableIndices.end(), open)
                == cfg->uniquelyDefinableIndices.end()) return false;

            const NameId scope = mb.nameMap.lookup(validityName);
            assert(scope != 0 && "knownVersionSettlesMarkedPremise: the firing scope is registered");
            canonicalizer.initializeCanonicalScope(mb, scope);
            const unsigned slot = gl::WorkerContext::coreId() >= 0
                ? static_cast<unsigned>(gl::WorkerContext::coreId()) : genScratchArenas().slotCount() - 1;
            ScratchArena& arena = genScratchArenas().forSlot(slot);
            ScratchScope frame(arena);
            CanonicalTemplateIndex& index = mb.lbMemory.canonicalUses.statementMatches;
            canonicalTemplates.ensureCanonicalTemplateIndex(mb, mb.intKnownStatements, index, arena, true);
            int32_t shapeLength = 0;
            while (shapeLength < expression.len && expression.ptr[shapeLength] != '[') ++shapeLength;
            const int32_t label = index.labels.lookup(StrSpan(expression.ptr, shapeLength));
            if (label == 0) return false;

            bool settled = false;
            const int32_t ancestorN = mb.nameMap.ancLen(scope);
            for (int32_t k = 0; k < ancestorN && !settled; ++k) {
                index.keysByShape.forEachWhile(packInt32Pair(label, mb.nameMap.ancAt(scope, k)), [&](int64_t key) {
                    if (mb.intKnownStatements.lookup(key) == 0) return true;
                    const StrSpan known = mb.nameMap.decodeView(Codec<StatementKey>::decode(key).orig);
                    StrSpan knownArgs[ExecutionParameters::MAX_ARITY];
                    const int32_t knownN = getArgsSpans(known, knownArgs, ExecutionParameters::MAX_ARITY);
                    assert(knownN == argN
                        && "knownVersionSettlesMarkedPremise: a known statement with the premise's core has another arity");
                    bool version = true;
                    for (int32_t i = 0; i < argN && version; ++i) {
                        if (i == open) continue;
                        version = equalSpans(canonicalTemplates.canonicalArgumentAtScope(mb, premiseArgs[i], scope),
                            canonicalTemplates.canonicalArgumentAtScope(mb, knownArgs[i], scope));
                    }
                    settled = version;
                    return !settled;
                });
            }
            return settled;
        }

/// @brief Whether `prepareIntegration` proves @p expression by a reductio
///        (the negated-compound intercept).
///
/// @details
/// True for a negated compact `!(X[args])` whose core compiles to an
/// entity other than `atomic` and `or`: its integration assumes the
/// positive compact in a `contradiction_` scope
/// (`ImplicationSubproofs::prepareNegatedCompoundContradiction`). A
/// positive form, an atomic core, an or-bodied core (all-branch refutation
/// is not this route) and an uncompiled core answer false.
///
/// @param expression Expression text (span over a stable buffer).
/// @return True when the intercept takes @p expression.
/// @see prepareIntegration, prepareHypothesisPremise.
inline bool IntegrationProcessor::negatedCompoundTakesReductio(StrSpan expression) const {
            if (expression.len < 2 || expression.ptr[0] != '!' || expression.ptr[1] != '(')
                return false;
            const LogicalEntity* negLe = this->definitions.compiledEntity(
                extractExpressionUniversalSpan(expression));
            return negLe != nullptr && negLe->category != "atomic" && negLe->category != "or";
        }

/// @brief Prepare one fully bound compound premise of an induction
///        hypothesis; the reductio of a negated premise waits for the
///        hypothesis's positive premises.
///
/// @details
/// The two routes that prepare a fully bound premise of a recursion-owned
/// rule (the install of `AdmissionMemory::updateAdmissionMap` and the
/// re-keyed chain of `AdmissionMemory::updateAdmissionMapRecursion`, I-294)
/// call this instead of `prepareIntegration`. A premise the intercept proves
/// by a reductio (`negatedCompoundTakesReductio`) at a scope that is no
/// integration trial is gated: the gates are the chain's positive elements
/// (no leading `!`) whose every argument is in @p remainingArgs, the
/// premise itself excluded; while one of them is not known at the scope or
/// an ancestor, the reductio parks under it
/// (`ImplicationSubproofs::deferPremiseReductio`) and opens when the
/// arrival drain finds every gate known. A hypothesis fires only when all
/// its premises hold, so a reductio opened before its positive premises
/// hold derives for a rule that cannot fire yet
/// (I-326). Every other premise,
/// and a reductio whose gates are all known, takes `prepareIntegration` as
/// before. Positive premises with an unbound argument are no gates: no
/// single instance of them can be known.
///
/// @param element       The fully bound premise (span over a stable buffer).
/// @param chain         The rule's premise chain, @p element among them.
/// @param chainN        Its length.
/// @param remainingArgs The chain's bound arguments, sorted strictly
///                      unique under `compareSpans`.
/// @param remN          Their number.
/// @param mb            The owning LB.
/// @param validityName  The preparing scope.
/// @return Nothing.
/// @see prepareIntegration, ImplicationSubproofs::deferPremiseReductio.
inline void IntegrationProcessor::prepareHypothesisPremise(StrSpan element, const StrSpan* chain,
            int32_t chainN, const StrSpan* remainingArgs, int32_t remN, Memory& mb,
            StrSpan validityName) {
            const NameId scope = mb.nameMap.lookup(validityName);
            const bool trial = scope != 0
                && gl::ScopeSyntax::atOrBelowRestrictedScope(mb.nameMap, scope);
            if (!trial && negatedCompoundTakesReductio(element)) {
                StrSpan gates[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
                int32_t gateN = 0;
                for (int32_t i = 0; i < chainN; ++i) {
                    const StrSpan candidate = chain[i];
                    if (equalSpans(candidate, element)) continue;
                    if (candidate.len > 0 && candidate.ptr[0] == '!') continue;
                    StrSpan args[ExecutionParameters::MAX_ARITY];
                    const int32_t argN = getArgsSpans(candidate, args, ExecutionParameters::MAX_ARITY);
                    bool bound = true;
                    for (int32_t a = 0; a < argN && bound; ++a)
                        bound = spanSetContains(remainingArgs, remN, args[a]);
                    if (!bound) continue;
                    assert(gateN < ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS
                        && "prepareHypothesisPremise: more gates than chain elements");
                    gates[gateN++] = candidate;
                }
                if (implicationSubproofs.deferPremiseReductio(element, gates, gateN, mb, validityName))
                    return;
            }
            prepareIntegration(element, remainingArgs, remN, mb, validityName, StrSpan());
        }

/// @brief Integration-template preparation from spans — the span-front
        ///        core of `prepareIntegration`.
        ///
        /// @details
        /// The full gated stage, in order: (1) marker-arg scan; (2)
        /// `integrationPrepared` double gate (non-minting probes at the scope
        /// AND at `"main"`) — every early return is copy-free; (2b) the
        /// premise-version gate (`premiseVersionSettlesMarkedGoal`,
        /// D-567); (3) the
        /// compiled-definition category lookup (vestigial empty-bodied `if`);
        /// (4) `disintegrateExprHypothetically`; (5) `integrationStartIntMap`
        /// set-or-insert (span `templateInterner.encode`); (6) the
        /// changeable→`"marker"` rename + `integrationPreparedMarker` double
        /// gate + marker-key mint; (7) the `tempKept <= 1` assert; (8) the
        /// unchangeable/changeable split; (9) `prepareIntegrationCore`; (10)
        /// `prepareIntegrationCore2`.
        ///
        /// 0% heap: every consumer is span-native, so NO expression / scope
        /// `std::string` is materialized — `extractExpressionUniversalSpan`,
        /// `disintegrateExprHypothetically`, `prepareIntegrationCore`'s goal, and
        /// `prepareIntegrationCore2`'s validityName / original all take spans.
        /// The two working strings (`replacedExpr`, `expressionWithUnchangeables`)
        /// are FRESH `ScratchString`s on the string-scratch tier (the byte-exact
        /// `replaceKeysScratch` twin of `replaceKeysToString`), held under their
        /// scopes across the probe/mint and the Core / Core2 calls.
        /// Unchangeable-args membership is `spanSetContains` over the sorted
        /// run — byte-identical to the former `std::set` lookups.
        ///
        /// @pre `[unchangeableArgsSorted, unchangeableArgsSorted +
        ///      unchangeableArgN)` sorted ascending strictly-unique under
        ///      `compareSpans` (== the former `std::set<std::string>`
        ///      iteration order), spans over buffers stable across the call.
        /// @param expression            The expression to prepare (span over a
        ///                              stable buffer).
        /// @param unchangeableArgsSorted The unchangeable args as a
        ///                              sorted-unique span run.
        /// @param unchangeableArgN      The run length.
        /// @param mb                    The owning LB.
        /// @param validityName          The scope (span over a stable buffer).
        /// @param rootGoal              The `toBeProved` goal driving this
        ///                              preparation (verbatim; span over a
        ///                              stable buffer), or empty for
        ///                              statement-driven preparations
        ///                              (admission / marker replays, the
        ///                              load-time prefill pass). Forwarded to
        ///                              `prepareIntegrationCore2`, which embeds
        ///                              it in every scope payload minted on
        ///                              MAIN (`<goal>_subproof_<bare>`) for
        ///                              disproof cleanup.
        /// @param typedEqualityGoal     True only at the goal door, where
        ///                              @p expression is the `toBeProved` goal
        ///                              itself: a positive `=` goal is then typed
        ///                              on the spot (`resolveEqualityGoalLabel`)
        ///                              and a set- or relation-typed one
        ///                              integrates by extensionality
        ///                              (I-267). Premise
        ///                              preparations (CNFE, admission, replays,
        ///                              prefill) carry rule-local names and stay
        ///                              atomic.
        /// @param twoChangeables        True only for the weak induction
        ///                              hypothesis's two-marker start: the
        ///                              one-changeable gate admits a template
        ///                              with two changeable arguments for this
        ///                              preparation
        ///                              (D-586).
        /// @param splitSets             The forbidden sets of a goal that one
        ///                              weak hypothesis install prepares
        ///                              @p splitN times, one per premise
        ///                              (null when @p splitN is 0); forwarded to
        ///                              `prepareIntegrationCore2`, which files
        ///                              every instruction once per set
        ///                              (D-579).
        /// @param splitN                Number of @p splitSets (0; at least 2, or 1 for a twin premise's goal, D-589).
        /// @invariant Mint-sequence-preserving vs the retired string body: per
        ///            call the interner touches are [gates: none] →
        ///            `disintegrateExprHypothetically`'s interior mints →
        ///            `templateInterner.encode(expression)` → [marker probes:
        ///            none] → `mintTemplateKey(replacedExpr, validityName)`
        ///            (`templateInterner` then `nameMap`) →
        ///            `prepareIntegrationCore` / Core2 interior mints — every
        ///            statement in the retired body's order, only argument
        ///            TYPES at existing calls changed.
        /// @see prepareIntegrationCore2; drainDeferredIntegrationPreps — the
        ///      sealed-view caller.
inline void IntegrationProcessor::prepareIntegration(StrSpan expression,
            const StrSpan* unchangeableArgsSorted, int32_t unchangeableArgN,
            Memory& mb,
            StrSpan validityName,
            StrSpan rootGoal,
            bool typedEqualityGoal,
            bool twoChangeables,
            const IntegrationForbiddenSets* splitSets,
            int32_t splitN) {
            // An integration trial scope, the usefulness scope and the scopes
            // below them prepare no integration: no subproof, OR branch,
            // contradiction scope or admission key opens there (D-481,
            // D-521).
            {
                const NameId scope = mb.nameMap.lookup(validityName);
                if (scope != 0 && gl::ScopeSyntax::atOrBelowRestrictedScope(mb.nameMap, scope)) return;
            }
            if (negatedCompoundTakesReductio(expression)) {
                implicationSubproofs.prepareNegatedCompoundContradiction(
                    expression, mb, validityName);
                return;
            }

            StrSpan argsCheck[ExecutionParameters::MAX_ARITY];
            const int32_t argsCheckN = getArgsSpans(
                expression, argsCheck, ExecutionParameters::MAX_ARITY);
            // `marker1` / `marker2` mark the two-marker start's goal
            // (D-586).
            bool hasMarker = false;
            for (int32_t i = 0; i < argsCheckN; ++i) {
                if (equalSpans(argsCheck[i], StrSpan("marker", 6))
                    || equalSpans(argsCheck[i], StrSpan("marker1", 7))
                    || equalSpans(argsCheck[i], StrSpan("marker2", 7))) {
                    hasMarker = true;
                    break;
                }
            }

            if (!hasMarker) {
                // Non-minting probes — a never-interned template was never
                // prepared (at this scope or at "main").
                if (canonicalTemplates.canonicalTemplatePresent(mb, mb.integrationPrepared, mb.lbMemory.canonicalUses.preparedMatches,
                    expression, validityName, mb.nameMap.lookup(validityName))) {
                    return;
                }
                if (canonicalTemplates.canonicalTemplatePresent(mb, mb.integrationPrepared, mb.lbMemory.canonicalUses.preparedMatches,
                    expression, StrSpan("main", 4), mb.nameMap.lookup(validityName))) {
                    return;
                }
            }

            // A goal whose one changeable argument is uniquely definable and
            // already named by a premise version is not integrated: the
            // integration could only rebuild that value
            // (D-567).
            if (premiseVersionSettlesMarkedGoal(expression, unchangeableArgsSorted,
                    unchangeableArgN, mb, validityName)) return;

            // Every consumer below is span-native (extractExpressionUniversalSpan,
            // disintegrateExprHypothetically, prepareIntegrationCore's goal, and
            // Core2's validityName / original all take StrSpan), so no expression
            // / scope std::string is materialized here — the former compiled-
            // definition boundary pair is retired (09c Section 2).

            // The core expression must be compiled here — make the
            // must-be-present contract loud (Rule 19) through the compiledEntity
            // reader fence, replacing the former unchecked
            // compiledExpressions.find(core)->second deref (latent UB on an
            // impossible path). The dead `if (category == "atomic") { //return; }`
            // branch is removed (user standing dead-code authorization).
            assert(this->definitions.compiledEntity(extractExpressionUniversalSpan(expression)) != nullptr
                && "prepareIntegration: core expression must be compiled");

            disintegrationProcessor.disintegrateExprHypothetically(expression, mb, validityName);


            {
                // Set-or-insert (the operator[] = overwrite the cold set-once
                // insert forbids): a re-prepared template takes the latest
                // startInt snapshot.
                const NameId istiTid = mb.templateInterner.encode(expression);
                const int32_t istiRow = mb.integrationStartIntMap.lookup(istiTid);
                if (istiRow != 0)
                    mb.integrationStartIntMap.setValueAt(istiRow, mb.startInt);
                else
                    mb.integrationStartIntMap.insert(istiTid, mb.startInt);
            }

            // --- New Logic Start ---
            StrSpan currentArgs[ExecutionParameters::MAX_ARITY];
            const int32_t currentArgsN = getArgsSpans(
                expression, currentArgs, ExecutionParameters::MAX_ARITY);
            // marker rename set as a stack StrReplacement run (key = arg span into
            // the stable expression, value = the static literal "marker"), so no
            // arena is needed — replaceKeysToString returns the std::string result
            // (I-138). Duplicate changeable args add
            // same-key/same-value pairs, which replaceKeysToString tolerates
            // byte-identically to the former std::map.
            StrReplacement markerPairs[ExecutionParameters::MAX_ARITY];
            int32_t markerN = 0;
            for (int32_t i = 0; i < currentArgsN; ++i) {
                // If arg is NOT in unchangeableArgs, it is changeable. The
                // two-marker start's `marker1` / `marker2` keep their names:
                // the two markers are distinct
                // (D-586).
                if (equalSpans(currentArgs[i], StrSpan("marker1", 7))
                    || equalSpans(currentArgs[i], StrSpan("marker2", 7))) continue;
                if (!spanSetContains(unchangeableArgsSorted, unchangeableArgN,
                        currentArgs[i])) {
                    markerPairs[markerN].key = currentArgs[i];
                    markerPairs[markerN].value = StrSpan("marker", 6);
                    ++markerN;
                }
            }

            // replacedExpr is a FRESH string on the string-scratch tier (the
            // byte-exact replaceKeysScratch twin of replaceKeysToString), held
            // through the two probes + the mint below. Its rScope frees it at
            // function exit; the later uScope (same slot arena) nests ABOVE it, so
            // the string survives (I-124, 09c pitfall 4).
            const unsigned rSlot = (gl::WorkerContext::coreId() >= 0)
                ? static_cast<unsigned>(gl::WorkerContext::coreId())
                : scratchArenas().slotCount() - 1;
            ScratchArena& rArena = scratchArenas().forSlot(rSlot);
            ScratchScope rScope(rArena);
            const ScratchString replacedExpr = replaceKeysScratch(
                rArena, expression, markerPairs, markerN);

            // Non-minting probes — at this scope or at "main".
            if (canonicalTemplates.canonicalTemplatePresent(mb, mb.integrationPreparedMarker, mb.lbMemory.canonicalUses.preparedMarkerMatches,
                    StrSpan(replacedExpr), validityName, mb.nameMap.lookup(validityName))) {
                return;
            }
            if (canonicalTemplates.canonicalTemplatePresent(mb, mb.integrationPreparedMarker, mb.lbMemory.canonicalUses.preparedMarkerMatches,
                    StrSpan(replacedExpr), StrSpan("main", 4), mb.nameMap.lookup(validityName))) {
                return;
            }

            const int64_t preparedKey = mintTemplateKey(
                mb.templateInterner, mb.nameMap, StrSpan(replacedExpr), validityName);
            mb.integrationPreparedMarker.mint(preparedKey);

            // --- New Logic End ---

            StrSpan tempArgs[ExecutionParameters::MAX_ARITY];
            const int32_t tempArgsN = getArgsSpans(
                expression, tempArgs, ExecutionParameters::MAX_ARITY);
            int32_t tempKept = 0;
            for (int32_t i = 0; i < tempArgsN; ++i) {
                if (!spanSetContains(unchangeableArgsSorted, unchangeableArgN,
                        tempArgs[i])) {
                    ++tempKept;
                }
            }
            // One-changeable-arg gate (maintainer decision 2026-08-10): a
            // template with more than one changeable argument is not
            // integrable — skipping it is a defined result of the
            // integration contract, not a failure. The prepared-marker
            // mint above already recorded the template, so re-entry
            // probes skip it without re-running the gate. Incubator mode
            // keeps its historical multi-changeable processing. The weak
            // induction hypothesis's two-marker start admits two
            // (D-586).
            if (!parameters.incubator_mode && tempKept > (twoChangeables ? 2 : 1)) {
                return;
            }


            const unsigned integSlot = (gl::WorkerContext::coreId() >= 0)
                ? static_cast<unsigned>(gl::WorkerContext::coreId())
                : genScratchArenas().slotCount() - 1;
            WorkInstruction instructions(&genScratchArenas().forSlot(integSlot));

            // 1. Separate Unchangeables and Changeables. Both rename sets ride stack
            // StrReplacement runs whose fresh values ("u_"+arg for unchangeables,
            // "pi_lev_<level>_<startIntPi>" for changeables) are bytes on the
            // per-slot string-scratch arena, held under a ScratchScope spanning the
            // replaceKeysToString AND the prepareIntegrationCore / Core2 calls below
            // (the Core2 map door consumes changeablePairs' spans). The cross-set
            // dedup guard is a linear span-scan over each run; every key is a span
            // over the stable `expression` — no arg string is materialized. startIntPi
            // increment order FROZEN (one per first-seen changeable arg).
            StrReplacement changeablePairs[ExecutionParameters::MAX_ARITY];
            int32_t changeableN = 0;
            {
                const unsigned uSlot = (gl::WorkerContext::coreId() >= 0)
                    ? static_cast<unsigned>(gl::WorkerContext::coreId())
                    : scratchArenas().slotCount() - 1;
                ScratchArena& uArena = scratchArenas().forSlot(uSlot);
                ScratchScope uScope(uArena);

                StrSpan args[ExecutionParameters::MAX_ARITY];
                const int32_t argN = getArgsSpans(
                    expression, args, ExecutionParameters::MAX_ARITY);
                StrReplacement unchangeablePairs[ExecutionParameters::MAX_ARITY];
                int32_t unchangeableN = 0;

                for (int32_t ai = 0; ai < argN; ++ai) {
                    const StrSpan arg = args[ai];
                    // Check if already processed (avoid duplicates from getArgs)
                    bool inUnchangeable = false;
                    for (int32_t j = 0; j < unchangeableN; ++j) {
                        if (equalSpans(unchangeablePairs[j].key, arg)) {
                            inUnchangeable = true;
                            break;
                        }
                    }
                    if (inUnchangeable) {
                        continue;
                    }

                    if (spanSetContains(unchangeableArgsSorted,
                            unchangeableArgN, arg)) {
                        // Unchangeable Mapping — span membership, no arg
                        // string materialized on this branch.
                        const bool alreadyU = (arg.len >= 2
                            && arg.ptr[0] == 'u' && arg.ptr[1] == '_');

                        if (!alreadyU)
                        {
                            // value = "u_" + arg : NEW bytes on the scratch arena.
                            char* v = uArena.allocBytes(2 + arg.len);
                            v[0] = 'u';
                            v[1] = '_';
                            if (arg.len > 0) {
                                std::memcpy(v + 2, arg.ptr,
                                    static_cast<size_t>(arg.len));
                            }
                            unchangeablePairs[unchangeableN].key = arg;
                            unchangeablePairs[unchangeableN].value =
                                StrSpan(v, 2 + arg.len);
                            ++unchangeableN;
                        }
                    }
                    else {
                        // Changeable Mapping — dedup by linear span-scan over the
                        // changeable pairs built so far (the former changeableMap
                        // .count(argStr)).
                        bool inChangeable = false;
                        for (int32_t c = 0; c < changeableN; ++c)
                            if (equalSpans(changeablePairs[c].key, arg)) { inChangeable = true; break; }
                        if (inChangeable) {
                            continue;
                        }
                        // freshVar = "pi_lev_<level>_<startIntPi>" : NEW bytes on uArena.
                        char pbuf[64]; int32_t pat = 0;
                        const char ppref[] = "pi_lev_";
                        for (int32_t k = 0; k < 7; ++k) pbuf[pat++] = ppref[k];
                        {
                            int v = mb.level; char tmp[16]; int32_t t = 0;
                            if (v == 0) tmp[t++] = '0';
                            else { int vv = v; while (vv > 0) { tmp[t++] = static_cast<char>('0' + vv % 10); vv /= 10; } }
                            for (int32_t k = t - 1; k >= 0; --k) pbuf[pat++] = tmp[k];
                        }
                        pbuf[pat++] = '_';
                        {
                            int v = mb.startIntPi; char tmp[16]; int32_t t = 0;
                            if (v == 0) tmp[t++] = '0';
                            else { int vv = v; while (vv > 0) { tmp[t++] = static_cast<char>('0' + vv % 10); vv /= 10; } }
                            for (int32_t k = t - 1; k >= 0; --k) pbuf[pat++] = tmp[k];
                        }
                        const ScratchString fresh = ScratchString::copyFrom(uArena, pbuf, pat);
                        assert(changeableN < ExecutionParameters::MAX_ARITY
                            && "prepareIntegration: changeable pair count exceeds cap");
                        changeablePairs[changeableN].key = arg;   // span over stable `expression`
                        changeablePairs[changeableN].value = StrSpan(fresh);
                        ++changeableN;
                        mb.startIntPi++;
                    }
                }

                // 2. Replace Unchangeables in Expression — a FRESH string on
                // uArena (the byte-exact replaceKeysScratch twin), held through
                // both Core / Core2 calls below (uScope spans them).
                const ScratchString expressionWithUnchangeables = replaceKeysScratch(
                    uArena, expression, unchangeablePairs, unchangeableN);

                // 3. Generate initial instructions. A set- or relation-typed
                // `=` goal integrates by extensionality (I-267):
                // its root is the typed `and` of the two subset implications
                // instead of the atomic registry entry. Typed on the spot.
                const LogicalEntity* rootEntity = nullptr;
                if (typedEqualityGoal && startsWithSpan(expression, "(=[", 3)) {
                    const StrSpan label = scopeTypes.resolveEqualityGoalLabel(expression, mb, validityName);
                    if (!equalSpans(label, StrSpan("(1)", 3))) {
                        rootEntity = this->definitions.typedEqualityEntity(label);
                        assert(rootEntity != nullptr
                            && "prepareIntegration: no extensionality entity for a typed `=` goal");
                    }
                }
                integrationBuilder.prepareIntegrationCore(StrSpan(expressionWithUnchangeables), instructions, mb,
                    expression, false, rootEntity);

                // In an induction step block a new integration tree starts its
                // path list with the goal's non-anchor fixed arguments
                // (D-529).
                if (mb.recursionAuxyImplId != 0) {
                    StrSpan fixed[ExecutionParameters::MAX_ARITY];
                    int32_t fixedN = 0;
                    for (int32_t ai = 0; ai < argN; ++ai)
                        if (spanSetContains(unchangeableArgsSorted, unchangeableArgN, args[ai]))
                            fixed[fixedN++] = args[ai];
                    instructions.setPathVariables(StrSpan(
                        admissionMemory.rootPathListScratch(mb, fixed, fixedN, uArena)));
                }

                // 4. Call Core2 to handle changeables and admission logic. INSIDE the
                // widened uScope so changeablePairs' pi_lev values (on uArena) stay
                // live through the map-door call.
                prepareIntegrationCore2(instructions, changeablePairs, changeableN, mb, validityName, rootGoal,
                    false, splitSets, splitN);
            }
        }
}
