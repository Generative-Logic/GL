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
class ProofHistory;
class RejectedSourceReplay;
class WatchProcessor;
namespace tests { struct EquivalenceAdmissionRewriterFixture; }
/// @brief Equivalence Admission Rewriter operations for the proof algorithm.
/// @details Uses explicit peer interfaces while the supplied logical blocks retain storage and proof state.
class EquivalenceAdmissionRewriter {
friend struct tests::EquivalenceAdmissionRewriterFixture;
public:
/// @brief Bind inputs for equivalence admission rewriter.
/// @details Construction stores explicit references; block storage and all invocation lifetimes remain with their owners.
/// @param canonicalTemplates Borrowed CanonicalTemplateMatcher interface or state; must outlive this component.
/// @param canonicalizer Borrowed Canonicalizer interface or state; must outlive this component.
/// @param parameters Borrowed ProverParameters interface or state; must outlive this component.
/// @param rejectedSources Borrowed RejectedSourceMemory interface or state; must outlive this component.
/// @param caseLedger Borrowed CaseLedger interface or state; must outlive this component.
/// @param proofHistory Borrowed ProofHistory interface or state; must outlive this component.
/// @param rejectedSourceReplay Borrowed RejectedSourceReplay interface or state; must outlive this component.
/// @param watchProcessor Borrowed WatchProcessor interface or state; must outlive this component.
/// @return None; construction binds the supplied references.
EquivalenceAdmissionRewriter(CanonicalTemplateMatcher& canonicalTemplates, Canonicalizer& canonicalizer, const ProverParameters& parameters, RejectedSourceMemory& rejectedSources, CaseLedger& caseLedger, ProofHistory& proofHistory, RejectedSourceReplay& rejectedSourceReplay, WatchProcessor& watchProcessor);

public:
//#pragma optimize("", off)


        // `applyEquivalenceClass` — the pairwise (class × statement) rewrite
        // driver — left the production tree with the statement batch; its
        // byte-twin survives as the test oracle `applyEquivalenceClassOracle`
        // (`tests/test_equi_reshuffle.cpp`), the D-193 survivorship rule.

        /// @brief Total per-LB dedup reset for a resent (equi-class-rewritten)
        ///        expression — the "delete" of delete -> send -> reabsorb.
        ///
        /// @details
        /// When an equivalence-class hook (`applyEquivalenceClassToRejectedMap` /
        /// `applyEquivalenceClassToRejectedMapIntegration`) canonicalises a
        /// compound's variables and resends the rewritten form on
        /// `sameIterationInternalMail`, that rewritten form may already sit in
        /// this LB's dedup registries (it arrived earlier as external mail and
        /// was never disintegrated). The plain absorb gate (`intStatementLevelsMap`)
        /// and Site F (`intKnownStatements`) would then dedup it away, so it is
        /// never disintegrated and its admission/rejection is never renewed. This
        /// helper erases the resent expression from every per-LB registry that
        /// would otherwise dedup the re-entry, so the plain absorb treats it as
        /// fresh and re-disintegrates it.
        ///
        /// History is preserved: `exprOriginMap` is NOT touched
        /// ([I-44](../../docs/agentic_swdd/30_invariants.md#i-44), Rule 16). The
        /// `localEncoded*` containers are NOT touched: the caller invokes this
        /// only for a NON-local expression (absent from
        /// `intLocalEncodedStatementsSet`), so it is already absent from them. The
        /// implication-specific `encodedMap` LMV walk + `expandedImplications`
        /// erase done by `eradicateImplicationFromLB` are intentionally NOT
        /// mirrored — a resent compound is not an implication.
        ///
        /// @param mb           Owning LB.
        /// @param original     Expression text of the resent compound to reset.
        /// @param validityName Validity-scope name of the resent compound.
        /// @invariant Caller guarantees `(original, validityName)` is absent from
        ///            `mb.intLocalEncodedStatementsSet`.
        /// @see `applyEquivalenceClassToRejectedMap`,
        ///      `applyEquivalenceClassToRejectedMapIntegration` — callers.
        /// @see `eradicateImplicationFromLB` — implication-side per-expression erase.
inline void resetResentExpressionRegistries(Memory& mb,
                                                     StrSpan original,
                                                     StrSpan validityName);

/// @brief The registry part of the resent-expression reset, without the
        ///        statement-list sweep; returns the expression's packed key.
        ///
        /// @details
        /// Mints the expression and its scope (deliberately, in that order),
        /// erases the levels row (probing first: a key without a row leaves
        /// the map untouched, and the probe spares `eraseSet`'s full
        /// compaction pass), erases the known row with its OR staging, and
        /// clears the negation watch of a row that was known. The rows of
        /// `intEncodedStatements` and `intExternalStatements` with the key
        /// stay: the caller removes them with `eraseResentRows`, at once
        /// (`resetResentExpressionRegistries`) or for all the resets of one
        /// hook call together (the ordis hook's mail run, which neither reads
        /// nor writes those lists).
        ///
        /// @param mb           Owning LB.
        /// @param original     Expression text of the resent compound.
        /// @param validityName Validity-scope name of the resent compound.
        /// @return `packStatementKey(original id, scope id)`.
        /// @invariant Caller guarantees `(original, validityName)` is absent
        ///            from `mb.intLocalEncodedStatementsSet`.
        /// @see resetResentExpressionRegistries, eraseResentRows.
inline int64_t resetResentRegistryKeys(Memory& mb, StrSpan original, StrSpan validityName);

/// @brief Remove the statement-list and external-staging rows of a set of
        ///        reset expressions, one stable sweep per list.
        ///
        /// @details
        /// A row leaves `intEncodedStatements` or `intExternalStatements`
        /// when its packed `(originalId, validityId)` is one of @p keys
        /// (sorted and deduplicated on the gen-scratch tier, then a binary
        /// search per row). `SharedExpressionList::eraseIf` compacts each
        /// list in one pass and leaves the survivors, their order and the
        /// dirty mark exactly as erasing the same rows one by one from back
        /// to front would. As in the per-row form, the class waterlines are
        /// not repaired (an open weakness, equivalence-classes chapter).
        ///
        /// @param mb       Owning LB.
        /// @param keys     Packed keys from `resetResentRegistryKeys`.
        /// @param keyCount Number of @p keys (>= 0).
        /// @return Void.
        /// @invariant Allocates only on the caller slot's gen-scratch arena,
        ///            rewound on return.
        /// @see resetResentRegistryKeys, resetResentExpressionRegistries.
inline void eraseResentRows(Memory& mb, const int64_t* keys, int32_t keyCount);

private:
/// @brief Re-key parked sources and defer their equality provenance.
        /// @details The substitution-token scan filters keys and source arguments
        /// before sorting affected keys or constructing rewritten text. Source-only
        /// arguments remain covered. Each source applies only replacements for
        /// tokens it contains, in the original class-member order, so deferred
        /// provenance stays identical. Each old run is cleared before its ordered
        /// replacement/wakeup; empty old keys are compacted in one final sweep.
        /// Rewritten parks retain generation and flags. Admission is revalidated
        /// by the normal replay. No empty source key survives the call.
        /// @tparam Integration Select the integration demand index.
        /// @tparam ClassT Read-only equivalence-class view.
        /// @param clss Class supplying substitutions and derivation levels.
        /// @param mb Owning memory block.
        /// @param validityName Class scope.
        /// @param buckets Scope index, refreshed after new keys appear.
        /// @return Number of source-index keys rewritten.
template <bool Integration, class ClassT>
        int32_t rewriteParkedExistences(const ClassT& clss, Memory& mb,
            StrSpan validityName, RejectedValidityBuckets& buckets,
            const TypedColdSet<int64_t>* selected = nullptr);

public:
/// @brief Apply class substitutions to parked existence sources.
        /// @details Retains deferred provenance and wakes the ordinary replay path.
        /// @tparam ClassT Read-only equivalence-class view.
        /// @param clss Class supplying substitutions.
        /// @param memoryBlock Owning memory block.
        /// @param validityName Class scope.
        /// @param buckets Source-key scope index.
        /// @return Number of rewritten keys.
template <class ClassT>
        inline int32_t applyEquivalenceClassToRejectedMapIntegration(
            const ClassT& clss, Memory& memoryBlock, StrSpan validityName,
            RejectedValidityBuckets& buckets,
            const TypedColdSet<int64_t>* selected = nullptr);

public:
// Algebra equi-class hook on admissionMap. For each entry, rewrites
        // its marker KEY and AdmissionMapValue contents (key + remainingArgs)
        // toward the single canonical representative of the class
        // (chooseCanonical), DROPS the changed key and inserts the canonical
        // K' (re-key), moves admissionStatusMap onto K', and fires
        // revisitRejected2(K', mb, depositValidity) to revive any matching
        // rejected cohort.
        //
        // Design principles (algebra-only):
        //   1. This function rewrites demand only. rejectedMap holds parked
        //      source existences and deferred provenance; its own source hook
        //      performs their equality transformations.
        //   2. admissionMap is metadata — keys and values carry no proof-graph
        //      history. Rewriting them is provenance-free.
        //   3. One canonical representative per class (chooseCanonical), so a
        //      key changes in at most one way; the changed key is dropped and
        //      the canonical K' inserted. Slot collapses are allowed (the
        //      companion rejected hook collapses identically). See
        //      D-106.
        //
        // Applies only at the same validityName (no ancestor/descendant
        // directions); the deposit scope is that shared validity. Anchor
        // predicates are never rewritten (D-82).
        //
        // u_-prefix note. admissionMap keys are bare-marker form (no u_).
        // AdmissionMapValue.key elements may contain u_-prefixed args; these
        // are naturally excluded from substitution because the equi-class
        // substMap keys are bare variable names (u_-prefixed names are never
        // class members). `ce::replaceKeysInString` only substitutes keys
        // present in the substMap.
        //
        // Inserts (and admissionStatusMap updates + cache populates +
        // revisitRejected2 calls) are queued and applied AFTER the outer
        // admissionMap walk to avoid iterator invalidation.
template <class ClassT>
        inline void applyEquivalenceClassToAdmissionMap(
            const ClassT& clss,
            Memory& memoryBlock,
            StrSpan validityName,
            const TypedColdSet<int64_t>* selected = nullptr);

public:
// Integration equi-class hook on admissionMapIntegration. Mirror of
        // algebra applyEquivalenceClassToAdmissionMap (D-57), with three
        // adjustments for integration-side data types:
        //
        //   1. admissionMapIntegration keys are u_-prefixed (built by
        //      disintegrateExprCore2's C1 block from le.signature with one
        //      nonUArg replaced by "marker"). The substMap from a class has
        //      bare-name keys — they would never match u_-prefixed args
        //      directly. We strip u_ from the key to obtain bareKey, rewrite
        //      it once with the single canonical substMap (chooseCanonical),
        //      then re-apply u_ to obtain the u_-form K'.
        //   2. admissionMapIntegration values are
        //      map<Instruction, set<string>> rather than
        //      set<AdmissionMapValue>. Each (Instruction, appliedVars) pair
        //      is rewritten independently. Instruction.data[*].signature /
        //      Instruction.data[*].elements[*] / Instruction.markedGoal are
        //      u_-form strings; appliedVars members are bare names per the
        //      construction sites in updateAdmissionMapIntegration and
        //      isAdmittedIntegration. We build an augmented substMap that
        //      includes both bare → canon and u_<bare> → u_<canon> pairs so
        //      both bare and u_-form occurrences get substituted
        //      consistently.
        //   3. revisitRejectedIntegration2 is called after each K' insert
        //      (mirror of revisitRejected2 in the algebra hook). bareK' is
        //      used because rejectedMapIntegration keys are bare-marker
        //      form, not u_-form.
        //
        // Design principles (mirror of the algebra admission hook, with the
        // I-22 asymmetry preserved):
        //   1. rejectedMapIntegration is sacred — never written by this
        //      function (it is rewritten via
        //      applyEquivalenceClassToRejectedMapIntegration which now
        //      drops+mails per the revised I-37).
        //   2. admissionMapIntegration keys and Instruction values are
        //      metadata — equivalence-class rewrites apply freely.
        //   3. One canonical representative per class (chooseCanonical): the
        //      changed key is DROPPED and the canonical K' inserted (re-key),
        //      not additive. Applies only at the same validityName. See
        //      D-106.
        //   4. Slot collapses are allowed (no arg-equalization filter): both
        //      sides canonicalise identically, so the positional-collision
        //      preservation is no longer load-bearing.
        //
        // I-22: integration admission entries persist across revival firings —
        // a single template can admit multiple distinct int_ witnesses over its
        // lifetime. No path erases an admission key.
template <class ClassT>
        inline void applyEquivalenceClassToAdmissionMapIntegration(
            const ClassT& clss,
            Memory& memoryBlock,
            StrSpan validityName,
            const TypedColdSet<int64_t>* selected = nullptr);

public:
/// @brief Apply class substitutions to parked existence sources.
        /// @details Retains deferred provenance and wakes the ordinary replay path.
        /// @tparam ClassT Read-only equivalence-class view.
        /// @param clss Class supplying substitutions.
        /// @param memoryBlock Owning memory block.
        /// @param validityName Class scope.
        /// @param buckets Source-key scope index.
        /// @return Number of rewritten keys.
template <class ClassT>
        inline int32_t applyEquivalenceClassToRejectedMap(
            const ClassT& clss, Memory& memoryBlock, StrSpan validityName,
            RejectedValidityBuckets& buckets,
            const TypedColdSet<int64_t>* selected = nullptr);

private:
/// @brief Shared core of the two parked-or equi-class hooks — I-37
        ///        drop + mail for the map handed in, read through the
        ///        per-apply substitution-token posting index instead of a
        ///        whole-map walk.
        ///
        /// @details
        /// A hit is a key of @p rmo at the class's own validity whose
        /// template changes under the class's canonical substitution. The
        /// candidates come from @p buckets, the per-apply
        /// `RejectedValidityBuckets` whose postings map `(validity, token id)`
        /// to the keys whose template carries that token
        /// (`forEachSubstitutionToken` — the boundary rule of
        /// `replaceKeysScratch`, so a template changes iff one of its tokens
        /// IS a non-canonical member): the postings of every non-canonical
        /// member at the class validity, sorted and deduplicated, each
        /// re-checked with the substitution test — the hit set is exactly
        /// the whole-map walk's. The index is refreshed through
        /// `RejectedValidityBuckets::ensureCurrent`: the park site may insert
        /// into the ordis maps inside an apply, and a moved insert epoch
        /// rebuilds the index, so a key parked earlier in the same apply is
        /// visited exactly as the whole-map walk visited it; a key dropped
        /// since the build is a `lookup` miss. No round skip: every class
        /// call probes the index.
        ///
        /// Each hit DROPS the entry (batched `eraseBlobIf`) and MAILS the
        /// rewritten parked OR STATEMENT onto `sameIterationInternalMail`
        /// with an `equality1` history line — never a direct re-insert. The
        /// absorb re-runs the full or consumption on the rewritten statement
        /// (fresh probe, standing tie rule, K rules deduped), which re-parks
        /// or admits it under the canonical keys through the normal ordis
        /// path. A key's non-marker tokens are drawn from the or statement's
        /// own arguments, so a rewrite that fires on the key fires on the
        /// statement too; the unchanged-statement skip mirrors the
        /// rejectedMap hook's defined skip. Hits are processed in template
        /// lex order — the former (template, validity) order restricted to
        /// the one validity — so the mail sequence is unchanged. Canonical
        /// selection, staging discipline (arena-interned records, deferred
        /// NameMap-minting drain per I-3) and the fired-substitutions-only
        /// `equality1` tail mirror `applyEquivalenceClassToRejectedMap`.
        ///
        /// @tparam ClassT The equivalence-class read interface
        ///                (`memberCount` / `memberId` / `forEachLevel`).
        /// @tparam MapT   The parked-or map type (`rejectedMapOrdis` and
        ///                `rejectedMapOrdis2` share it: a `TypedColdBlobMap`
        ///                keyed by the packed `(templateId, validityId)`).
        /// @param clss         The (delta) class driving the rewrite.
        /// @param memoryBlock  The owning LB.
        /// @param validityName The class's scope; only same-scope entries
        ///                     rewrite.
        /// @param rmo          The parked-or map to rewrite.
        /// @param buckets      The per-apply posting index over @p rmo (one
        ///                     per map per `applyEquiClasses` call).
        /// @return (void)
        /// @invariant Never writes @p rmo except the batched drop (I-37);
        ///            insertions happen only through the re-consumption of
        ///            the mailed statement. A class member name carries
        ///            neither `]` nor `,` (asserted) — the premise under
        ///            which the posting index is exact for the substitution.
        /// @see applyEquivalenceClassToRejectedMapOrdis,
        ///      applyEquivalenceClassToRejectedMapOrdis2 — the two doors;
        ///      applyEquivalenceClassToRejectedMap — the discipline template;
        ///      RejectedValidityBuckets::ensureCurrent — the freshness rule.
template <class ClassT, class MapT>
        inline void applyEquivalenceClassToRejectedOrdisCore(
            const ClassT& clss,
            Memory& memoryBlock,
            StrSpan validityName,
            MapT& rmo,
            RejectedValidityBuckets& buckets,
            const TypedColdSet<int64_t>* selected = nullptr);

public:
/// @brief Ordis equi-class hook on `rejectedMapOrdis` — I-37
        ///        verbatim for the parked-cohort map.
        ///
        /// @details
        /// The `rejectedMapOrdis` door of
        /// @ref applyEquivalenceClassToRejectedOrdisCore: every same-scope
        /// park whose key template changes under the class is dropped and
        /// its rewritten or statement mailed onto
        /// `sameIterationInternalMail` with an `equality1` history line, the
        /// candidates read from the per-apply posting index.
        ///
        /// @tparam ClassT The equivalence-class read interface
        ///                (`memberCount` / `memberId` / `forEachLevel`).
        /// @param clss         The (delta) class driving the rewrite.
        /// @param memoryBlock  The owning LB.
        /// @param validityName The class's scope; only same-scope entries
        ///                     rewrite.
        /// @param buckets      The per-apply posting index over
        ///                     `rejectedMapOrdis`.
        /// @return (void)
        /// @invariant Never writes `rejectedMapOrdis` directly (I-37);
        ///            insertions happen only through the re-consumption of
        ///            the mailed statement.
        /// @see applyEquivalenceClassToRejectedOrdisCore — the body;
        ///      applyEquiClasses — both registration blocks.
template <class ClassT>
        inline void applyEquivalenceClassToRejectedMapOrdis(
            const ClassT& clss,
            Memory& memoryBlock,
            StrSpan validityName,
            RejectedValidityBuckets& buckets,
            const TypedColdSet<int64_t>* selected = nullptr);

public:
/// @brief Ordis2 equi-class hook on `rejectedMapOrdis2` — I-37
        ///        verbatim for the second park index
        ///        (D-267).
        ///
        /// @details
        /// The `rejectedMapOrdis2` door of
        /// @ref applyEquivalenceClassToRejectedOrdisCore, pointed at the
        /// ground-text park index: every same-scope park whose key text
        /// changes under the class is dropped and its rewritten or statement
        /// mailed onto `sameIterationInternalMail` with an `equality1`
        /// history line. An eligible-disjunct key is a sub-text of the or
        /// statement itself, so a rewrite that fires on the key fires on the
        /// statement too. Both parked-or hooks and the demand hook run at the
        /// same applyEquiClasses seams, so the pair's ground-to-ground
        /// matching never desynchronizes.
        ///
        /// @tparam ClassT The equivalence-class read interface
        ///                (`memberCount` / `memberId` / `forEachLevel`).
        /// @param clss         The (delta) class driving the rewrite.
        /// @param memoryBlock  The owning LB.
        /// @param validityName The class's scope; only same-scope entries
        ///                     rewrite.
        /// @param buckets      The per-apply posting index over
        ///                     `rejectedMapOrdis2`.
        /// @return (void)
        /// @invariant Never writes `rejectedMapOrdis2` directly (I-37);
        ///            insertions happen only through the re-consumption of
        ///            the mailed statement.
        /// @see applyEquivalenceClassToRejectedOrdisCore — the body;
        ///      applyEquiClasses — both registration blocks.
template <class ClassT>
        inline void applyEquivalenceClassToRejectedMapOrdis2(
            const ClassT& clss,
            Memory& memoryBlock,
            StrSpan validityName,
            RejectedValidityBuckets& buckets,
            const TypedColdSet<int64_t>* selected = nullptr);

public:
/// @brief Equi-class hook for `admissionMapOrdis2` — drop-and-rekey
        ///        each demand key whose ground template the class rewrites
        ///        (D-267).
        ///
        /// @details
        /// Sibling of `applyEquivalenceClassToRejectedMapOrdis`, with the
        /// admission-path drop-and-rekey discipline (the D-106
        /// form) instead of drop-and-mail: a demand entry
        /// is not a statement — there is nothing to re-deposit — so the
        /// stale key's value blobs are copied verbatim, the key erased,
        /// and the blobs re-inserted under the rewritten (canonicalized)
        /// template at the same validity. Both sides of the
        /// ground-to-ground route-(c) match rewrite at this same seam (the
        /// parked-or hooks run beside it), so matching never
        /// desynchronizes. The walk is a decoded-lex-sorted key snapshot
        /// (I-84); mutation runs strictly after the walk. Blob-level: no
        /// record decode — `{sourceImplId, levels}` payloads transfer
        /// byte-verbatim.
        ///
        /// @tparam ClassT Equivalence-class view type (member-id surface).
        /// @param clss         The delta class driving the rewrite.
        /// @param memoryBlock  Owning LB.
        /// @param validityName The class scope; only demand keys at exactly
        ///        this validity rewrite (the ordis hook's same filter).
        /// @return (void)
        /// @invariant Single-threaded seam only (I-83); the map's canonical
        ///            run order is preserved by the RMW re-insert (I-99).
        /// @see `applyEquivalenceClassToRejectedMapOrdis` — the walk
        ///      template; `consumeOrLeavesCohort` route (c) — the reader.
template <typename ClassT>
        inline void applyEquivalenceClassToAdmissionMapOrdis2(
            const ClassT& clss,
            Memory& memoryBlock,
            StrSpan validityName,
            const TypedColdSet<int64_t>* selected = nullptr);
private:

CanonicalTemplateMatcher& canonicalTemplates;
Canonicalizer& canonicalizer;
const ProverParameters& parameters;
RejectedSourceMemory& rejectedSources;
CaseLedger& caseLedger;
ProofHistory& proofHistory;
RejectedSourceReplay& rejectedSourceReplay;
WatchProcessor& watchProcessor;
};
}
#include "../logical_block/case_ledger.hpp"
#include "../logical_block/proof_history.hpp"
#include "../logical_block/rejected_source_replay.hpp"
#include "../logical_block/watch_processor.hpp"
namespace gl {

//#pragma optimize("", off)


        // `applyEquivalenceClass` — the pairwise (class × statement) rewrite
        // driver — left the production tree with the statement batch; its
        // byte-twin survives as the test oracle `applyEquivalenceClassOracle`
        // (`tests/test_equi_reshuffle.cpp`), the D-193 survivorship rule.

        /// @brief Total per-LB dedup reset for a resent (equi-class-rewritten)
        ///        expression — the "delete" of delete -> send -> reabsorb.
        ///
        /// @details
        /// When an equivalence-class hook (`applyEquivalenceClassToRejectedMap` /
        /// `applyEquivalenceClassToRejectedMapIntegration`) canonicalises a
        /// compound's variables and resends the rewritten form on
        /// `sameIterationInternalMail`, that rewritten form may already sit in
        /// this LB's dedup registries (it arrived earlier as external mail and
        /// was never disintegrated). The plain absorb gate (`intStatementLevelsMap`)
        /// and Site F (`intKnownStatements`) would then dedup it away, so it is
        /// never disintegrated and its admission/rejection is never renewed. This
        /// helper erases the resent expression from every per-LB registry that
        /// would otherwise dedup the re-entry, so the plain absorb treats it as
        /// fresh and re-disintegrates it.
        ///
        /// History is preserved: `exprOriginMap` is NOT touched
        /// ([I-44](../../docs/agentic_swdd/30_invariants.md#i-44), Rule 16). The
        /// `localEncoded*` containers are NOT touched: the caller invokes this
        /// only for a NON-local expression (absent from
        /// `intLocalEncodedStatementsSet`), so it is already absent from them. The
        /// implication-specific `encodedMap` LMV walk + `expandedImplications`
        /// erase done by `eradicateImplicationFromLB` are intentionally NOT
        /// mirrored — a resent compound is not an implication.
        ///
        /// @param mb           Owning LB.
        /// @param original     Expression text of the resent compound to reset.
        /// @param validityName Validity-scope name of the resent compound.
        /// @invariant Caller guarantees `(original, validityName)` is absent from
        ///            `mb.intLocalEncodedStatementsSet`.
        /// @see `applyEquivalenceClassToRejectedMap`,
        ///      `applyEquivalenceClassToRejectedMapIntegration` — callers.
        /// @see `eradicateImplicationFromLB` — implication-side per-expression erase.
inline void EquivalenceAdmissionRewriter::resetResentExpressionRegistries(Memory& mb,
                                                     StrSpan original,
                                                     StrSpan validityName) {
            const int64_t key = resetResentRegistryKeys(mb, original, validityName);
            eraseResentRows(mb, &key, 1);
        }

/// @brief The registry part of the resent-expression reset, without the
        ///        statement-list sweep; returns the expression's packed key.
        ///
        /// @details
        /// Mints the expression and its scope, erases the levels row when
        /// there is one, erases the known row with its OR staging and clears
        /// the negation watch of a known row. The statement-list and staging
        /// rows are left to `eraseResentRows`. Full contract at the
        /// declaration.
        ///
        /// @param mb           Owning LB.
        /// @param original     Expression text of the resent compound.
        /// @param validityName Validity-scope name of the resent compound.
        /// @return `packStatementKey(original id, scope id)`.
        /// @see eraseResentRows.
inline int64_t EquivalenceAdmissionRewriter::resetResentRegistryKeys(Memory& mb,
                                                     StrSpan original,
                                                     StrSpan validityName) {
            RT_SCOPE_HERE("RESET_RESENT_REGISTRIES");
            // encode (not lookup) — deliberate: this call minted on a
            // never-seen expression before the migration too, and swapping it
            // for a non-minting lookup would shift the LB's id-assignment
            // order on that path. Same two mints (original, then validityName),
            // same order, same bytes as the former ExpressionWithValidity form.
            const NameId origId = mb.nameMap.encode(original);
            const NameId valId = mb.nameMap.encode(validityName);
            const int64_t key = packStatementKey(origId, valId);
            {
                RT_SCOPE_HERE("RESET_LEVELS_ERASE");
                // A key without a row leaves the map and its dirty state
                // untouched; the probe spares eraseSet's compaction pass.
                if (mb.intStatementLevelsMap.lookup(key) != 0)
                    mb.intStatementLevelsMap.eraseSet(key);
            }
            const bool wasKnown = mb.intKnownStatements.find(
                StatementKey{ origId, valId }) != nullptr;
            {
                RT_SCOPE_HERE("RESET_KNOWN_ERASE");
                gl::StatementRegistryOperations::eraseKnownStatementKey(mb, origId, valId);
            }
            watchProcessor.clearNegationWatchFor(mb, origId, valId, wasKnown);
            return key;
        }

/// @brief Remove the statement-list and external-staging rows of a set of
        ///        reset expressions, one stable sweep per list.
        ///
        /// @details
        /// The keys are sorted and deduplicated on the gen-scratch tier; a
        /// row leaves when its packed id pair is found by binary search.
        /// Stored rows are canonical-pipeline encodings, so the id pair is
        /// the full identity. The external staging is a view of the registry
        /// for this burst's request generation: a row whose registry entry is
        /// gone must leave it too, or request generation keeps a mandatory
        /// ingredient that can no longer be a premise. Full contract at the
        /// declaration.
        ///
        /// @param mb       Owning LB.
        /// @param keys     Packed keys from `resetResentRegistryKeys`.
        /// @param keyCount Number of @p keys (>= 0).
        /// @return Void.
        /// @see resetResentRegistryKeys.
inline void EquivalenceAdmissionRewriter::eraseResentRows(Memory& mb, const int64_t* keys,
                                                     int32_t keyCount) {
            RT_SCOPE_HERE("RESET_VECTOR_SCANS");
            assert(keyCount >= 0 && (keyCount == 0 || keys != nullptr)
                && "eraseResentRows: a key run");
            if (keyCount == 0) return;
            const unsigned slot = (gl::WorkerContext::coreId() >= 0)
                ? static_cast<unsigned>(gl::WorkerContext::coreId())
                : genScratchArenas().slotCount() - 1;
            ScratchArena& arena = genScratchArenas().forSlot(slot);
            ScratchScope frame(arena);
            int64_t* sorted = reinterpret_cast<int64_t*>(arena.resolve(arena.alloc(
                keyCount * static_cast<int32_t>(sizeof(int64_t)),
                static_cast<int32_t>(alignof(int64_t)))));
            std::copy(keys, keys + keyCount, sorted);
            std::sort(sorted, sorted + keyCount);
            const int64_t* sortedEnd = std::unique(sorted, sorted + keyCount);
            const auto resent = [&](const IntEncodedExpr& row) {
                return std::binary_search(static_cast<const int64_t*>(sorted), sortedEnd,
                    packStatementKey(row.originalId, row.validityId));
            };
            mb.intEncodedStatements.eraseIf(resent, [](int32_t) {});
            mb.intExternalStatements.eraseIf(resent, [](int32_t) {});
        }

/// @brief Re-key parked sources and defer their equality provenance.
        /// @details The substitution-token scan filters keys and source arguments
        /// before sorting affected keys or constructing rewritten text. Source-only
        /// arguments remain covered. Each source applies only replacements for
        /// tokens it contains, in the original class-member order, so deferred
        /// provenance stays identical. Each old run is cleared before its ordered
        /// replacement/wakeup; empty old keys are compacted in one final sweep.
        /// Rewritten parks retain generation and flags. Admission is revalidated
        /// by the normal replay. No empty source key survives the call.
        /// @tparam Integration Select the integration demand index.
        /// @tparam ClassT Read-only equivalence-class view.
        /// @param clss Class supplying substitutions and derivation levels.
        /// @param mb Owning memory block.
        /// @param validityName Class scope.
        /// @param buckets Scope index, refreshed after new keys appear.
        /// @return Number of source-index keys rewritten.
template <bool Integration, class ClassT>
inline int32_t EquivalenceAdmissionRewriter::rewriteParkedExistences(const ClassT& clss, Memory& mb,
            StrSpan validityName, RejectedValidityBuckets& buckets,
            const TypedColdSet<int64_t>* selected) {
            const NameId scopeId = mb.nameMap.lookup(validityName);
            assert(scopeId != 0);
            if (mb.phase3IgnoresScope(scopeId)) return 0;
            auto& map = [&]() -> auto& {
                if constexpr (Integration) return mb.overallHashMemory.rejectedMapIntegration;
                else return mb.overallHashMemory.rejectedMap;
            }();
            if (map.empty() && (Integration || mb.existenceReplays.empty())) return 0;
            const NameId canonicalId = canonicalizer.chooseCanonicalId(clss, mb, validityName);
            if (canonicalId == 0) return 0;
            const unsigned slot = gl::WorkerContext::coreId() >= 0
                ? static_cast<unsigned>(gl::WorkerContext::coreId()) : genScratchArenas().slotCount() - 1;
            ScratchArena& arena = genScratchArenas().forSlot(slot);
            ScratchArena& strings = scratchArenas().forSlot(slot);
            ScratchScope frame(arena);
            ScratchScope stringFrame(strings);
            const StrSpan canonicalView = mb.nameMap.decodeView(canonicalId);
            const ScratchString canonical = ScratchString::copyFrom(strings,
                canonicalView.ptr, canonicalView.len);
            const int32_t memberCount = clss.memberCount();
            StrReplacement* pairs = reinterpret_cast<StrReplacement*>(arena.resolve(
                arena.alloc(memberCount * sizeof(StrReplacement), alignof(StrReplacement))));
            int32_t pairCount = 0;
            for (int32_t i = 0; i < memberCount; ++i) {
                if (clss.memberId(i) == canonicalId) continue;
                const StrSpan view = mb.nameMap.decodeView(clss.memberId(i));
                pairs[pairCount++] = {StrSpan(ScratchString::copyFrom(strings, view.ptr, view.len)), StrSpan(canonical)};
            }
            if (pairCount == 0) return 0;
            DirtyState dirty = DirtyState::Clean;
            ColdHashSet<BytesKeyStore> replacedTokens(&arena, &dirty);
            for (int32_t i = 0; i < pairCount; ++i) replacedTokens.mint(pairs[i].key);
            assert(replacedTokens.count() == pairCount);
            const auto touches = [&](StrSpan text) {
                bool changed = false;
                forEachSubstitutionToken(text, [&](StrSpan token) {
                    if (replacedTokens.contains(token)) changed = true;
                });
                return changed;
            };
            PagedVector<int64_t> keys(&arena, &dirty);
            if (selected != nullptr) {
                for (int32_t i = 1; i <= selected->count(); ++i) {
                    const int64_t key = selected->keyAt(i);
                    if (map.lookup(key) != 0) keys.push_back(key);
                }
            } else {
                buckets.ensureCurrent(map, mb.templateInterner, mb.nameMap);
                for (int32_t i = buckets.runBegin(scopeId); i < buckets.runEnd(scopeId); ++i) {
                    const int64_t key = buckets.pkAt(i);
                    if (map.lookup(key) != 0) keys.push_back(key);
                }
            }
            RT_SCOPE_HERE("SOURCE_REWRITE_ORDER_AND_APPLY");
            const auto less = [&](int32_t a, int32_t b) {
                return compareSpans(mb.templateInterner.decodeView(Codec<StatementKey>::decode(keys[a]).orig),
                    mb.templateInterner.decodeView(Codec<StatementKey>::decode(keys[b]).orig)) < 0;
            };
            ChunkSortedOrdinals<decltype(less)> order(arena, keys.size(), 0, less);
            ColdHashSet<BytesKeyStore> changedKeys(&arena, &dirty);
            ColdHashSet<BytesKeyStore> emptyKeys(&arena, &dirty);
            ColdHashSet<PodKeyStore<int64_t>> emptiedKeys(&arena, &dirty);
            int32_t changedCount = 0;
            for (int32_t i = 0; i < keys.size(); ++i) {
                ScratchScope rowFrame(arena);
                ScratchScope rowStrings(strings);
                const int64_t pk = keys[order.next()];
                const StrSpan keyView = mb.templateInterner.decodeView(Codec<StatementKey>::decode(pk).orig);
                const ScratchString keyBefore = ScratchString::copyFrom(strings, keyView.ptr, keyView.len);
                const ScratchString keyAfter = replaceKeysScratch(strings, StrSpan(keyBefore), pairs, pairCount);
                const auto snapshot = [&]() {
                    if constexpr (Integration) return snapshotRejectedIntegrationRun(map, pk, arena);
                    else return snapshotRejectedRun(map, pk, arena);
                }();
                bool changed = !equalSpans(StrSpan(keyBefore), StrSpan(keyAfter));
                for (int32_t r = 0; r < snapshot.count && !changed; ++r) {
                    ScratchScope probeFrame(strings);
                    const StrSpan source = mb.valueInterner.decodeView(snapshot.views[r].sourceExpressionId());
                    changed = !equalSpans(source, StrSpan(replaceKeysScratch(strings, source, pairs, pairCount)));
                }
                if (!changed) continue;
                assert(snapshot.count > 0);
                // Clear this source run before its replacement and wakeup, as
                // required by the sequential rewrite contract. Empty old keys
                // have no source records and are compacted together below.
                auto& sourceLookup = Integration ? mb.lbMemory.existenceLookups.integrationSources
                    : mb.lbMemory.existenceLookups.algebraSources;
                if (sourceLookup.built) {
                    const NameId sourceScope = Codec<StatementKey>::decode(pk).validity;
                    for (int32_t r = 0; r < snapshot.count; ++r)
                        sourceLookup.remove(packInt32Pair(rejectedSources.canonicalExistenceId(mb,
                            mb.valueInterner.decodeView(snapshot.views[r].sourceExpressionId()), sourceScope), sourceScope), pk);
                }
                map.inner().assignRun(pk, nullptr, nullptr, 0);
                if (!equalSpans(StrSpan(keyBefore), StrSpan(keyAfter))) emptiedKeys.mint(pk);
                ++changedCount;
                changedKeys.mint(StrSpan(keyAfter));
                for (int32_t r = 0; r < snapshot.count; ++r) {
                    ScratchScope recordFrame(arena);
                    ScratchScope recordStrings(strings);
                    const auto& source = snapshot.views[r];
                    PagedVector<int32_t> rewrites(&arena, &dirty);
                    for (int32_t j = 0; j < source.rewriteCount(); ++j)
                        rewrites.push_back(source.rewriteId(j));
                    int levels[256];
                    assert(source.levelCount() <= 256);
                    source.copyLevels(levels);
                    int32_t levelCount = source.levelCount();
                    const StrSpan originalView = mb.valueInterner.decodeView(source.sourceExpressionId());
                    ScratchString current = ScratchString::copyFrom(strings, originalView.ptr, originalView.len);
                    ColdHashSet<PodKeyStore<int32_t>> presentPairs(&arena, &dirty);
                    forEachSubstitutionToken(StrSpan(current), [&](StrSpan token) {
                        const int32_t id = replacedTokens.lookup(token);
                        if (id != 0) presentPairs.mint(id - 1);
                    });
                    const auto pairLess = [&](int32_t a, int32_t b) {
                        return presentPairs.keyAt(a + 1) < presentPairs.keyAt(b + 1);
                    };
                    ChunkSortedOrdinals<decltype(pairLess)> pairOrder(
                        arena, presentPairs.count(), 0, pairLess);
                    for (int32_t p = 0; p < presentPairs.count(); ++p) {
                        const int32_t j = presentPairs.keyAt(pairOrder.next() + 1);
                        const ScratchString after = replaceKeysScratch(strings, StrSpan(current), &pairs[j], 1);
                        if (equalSpans(StrSpan(current), StrSpan(after))) continue;
                        char equality[512];
                        const int32_t length = std::snprintf(equality, sizeof(equality), "(=[%.*s,%.*s])",
                            pairs[j].key.len, pairs[j].key.ptr, pairs[j].value.len, pairs[j].value.ptr);
                        assert(length > 0 && length < sizeof(equality));
                        rewrites.push_back(mb.valueInterner.encode(StrSpan(current)));
                        rewrites.push_back(mb.valueInterner.encode(StrSpan(after)));
                        rewrites.push_back(mb.valueInterner.encode(StrSpan(equality, length)));
                        const NameId from = mb.nameMap.lookup(pairs[j].key);
                        assert(from != 0);
                        clss.forEachLevel(packEqPairKey(from, canonicalId), [&](int level) {
                            levelCount = insertLevelSorted(levels, levelCount, level, 256);
                        });
                        current = after;
                    }
                    const int32_t n = rewrites.size();
                    int32_t* ids = n == 0 ? nullptr : reinterpret_cast<int32_t*>(arena.resolve(arena.alloc(n * 4, 4)));
                    for (int32_t j = 0; j < n; ++j) ids[j] = rewrites[j];
                    {
                        RT_SCOPE_HERE("SOURCE_REWRITE_STORE");
                        if constexpr (Integration)
                            rejectedSources.updateRejectedMapIntegration(StrSpan(keyAfter), StrSpan(current), source.iteration(),
                                source.flags(), levels, levelCount, mb.overallHashMemory, mb, validityName, ids, n, source.integrationDepth());
                        else
                            rejectedSources.updateRejectedMap(StrSpan(keyAfter), StrSpan(current), source.iteration(),
                                source.flags(), levels, levelCount, mb.overallHashMemory, mb, validityName, ids, n, source.integrationDepth());
                    }
                }
                if constexpr (Integration) rejectedSourceReplay.revisitRejectedIntegration2(StrSpan(keyAfter), mb, validityName);
                else rejectedSourceReplay.revisitRejected2(StrSpan(keyAfter), mb, validityName);
            }
            if (!emptiedKeys.empty()) {
                RT_SCOPE_HERE("SOURCE_REWRITE_COMPACT_EMPTY");
                for (int32_t i = 1; i <= emptiedKeys.count(); ++i) {
                    const int32_t id = map.lookup(emptiedKeys.keyAt(i));
                    assert(id != 0 && map.runLen(id) == 0);
                }
                const int32_t removed = map.eraseBlobIf(
                    [&](int64_t key) { return emptiedKeys.contains(key); });
                assert(removed == emptiedKeys.count());
            }
            if constexpr (Integration)
                rejectedSourceReplay.wakeParkedExistencesAtExistingDemands(mb, emptyKeys, changedKeys);
            else rejectedSourceReplay.wakeParkedExistencesAtExistingDemands(mb, changedKeys, emptyKeys);
            auto& sourceLookup = Integration ? mb.lbMemory.existenceLookups.integrationSources
                : mb.lbMemory.existenceLookups.algebraSources;
            if (sourceLookup.built) sourceLookup.postings.maintain();
            if (changedCount != 0) mb.mutatedThisBurst = true;
            return changedCount;
        }

/// @brief Apply class substitutions to parked existence sources.
        /// @details Retains deferred provenance and wakes the ordinary replay path.
        /// @tparam ClassT Read-only equivalence-class view.
        /// @param clss Class supplying substitutions.
        /// @param memoryBlock Owning memory block.
        /// @param validityName Class scope.
        /// @param buckets Source-key scope index.
        /// @return Number of rewritten keys.
template <class ClassT>
        inline int32_t EquivalenceAdmissionRewriter::applyEquivalenceClassToRejectedMapIntegration(
            const ClassT& clss, Memory& memoryBlock, StrSpan validityName,
            RejectedValidityBuckets& buckets,
            const TypedColdSet<int64_t>* selected) {
            return rewriteParkedExistences<true>(clss, memoryBlock, validityName, buckets, selected);
        }

// Algebra equi-class hook on admissionMap. For each entry, rewrites
        // its marker KEY and AdmissionMapValue contents (key + remainingArgs)
        // toward the single canonical representative of the class
        // (chooseCanonical), DROPS the changed key and inserts the canonical
        // K' (re-key), moves admissionStatusMap onto K', and fires
        // revisitRejected2(K', mb, depositValidity) to revive any matching
        // rejected cohort.
        //
        // Design principles (algebra-only):
        //   1. This function rewrites demand only. rejectedMap holds parked
        //      source existences and deferred provenance; its own source hook
        //      performs their equality transformations.
        //   2. admissionMap is metadata — keys and values carry no proof-graph
        //      history. Rewriting them is provenance-free.
        //   3. One canonical representative per class (chooseCanonical), so a
        //      key changes in at most one way; the changed key is dropped and
        //      the canonical K' inserted. Slot collapses are allowed (the
        //      companion rejected hook collapses identically). See
        //      D-106.
        //
        // Applies only at the same validityName (no ancestor/descendant
        // directions); the deposit scope is that shared validity. Anchor
        // predicates are never rewritten (D-82).
        //
        // u_-prefix note. admissionMap keys are bare-marker form (no u_).
        // AdmissionMapValue.key elements may contain u_-prefixed args; these
        // are naturally excluded from substitution because the equi-class
        // substMap keys are bare variable names (u_-prefixed names are never
        // class members). `ce::replaceKeysInString` only substitutes keys
        // present in the substMap.
        //
        // Inserts (and admissionStatusMap updates + cache populates +
        // revisitRejected2 calls) are queued and applied AFTER the outer
        // admissionMap walk to avoid iterator invalidation.
template <class ClassT>
        inline void EquivalenceAdmissionRewriter::applyEquivalenceClassToAdmissionMap(
            const ClassT& clss,
            Memory& memoryBlock,
            StrSpan validityName,
            const TypedColdSet<int64_t>* selected) {

            const NameId phase3Scope = memoryBlock.nameMap.lookup(validityName);
            if (phase3Scope != 0 && memoryBlock.phase3IgnoresScope(phase3Scope))
                return;

            auto& am = memoryBlock.overallHashMemory.admissionMap;
            if (am.empty()) return;

            // One canonical representative for the whole class (normal > int_
            // > it_, lex-smallest within a tier, strong members only). The
            // single replacement run (built below, on the scratch arena) sends
            // every other member to it.
            const NameId canonId = canonicalizer.chooseCanonicalId(clss, memoryBlock, validityName);
            if (canonId == 0) return;

            // Queued post-loop application. A changed key K is dropped and the
            // canonical key K' inserted with its value rewritten the same way;
            // K's status is moved onto K'. Queue the work so the am walk is
            // not mutated mid-iteration.
            // Insert staging on the arena (drained post-walk; the drain mints in
            // NameMap / templateInterner, so it stays deferred). The rewritten K'
            // template + deposit-validity are interned into `amStrings` (NOT
            // NameMap) — the record carries their amStrings ids. The rewritten
            // value is written STRAIGHT into the shared amKeyPool / amRemPool int
            // columns as half-open [start,count) runs (no owning AdmissionMapValue,
            // depth / sec / flag carried as record fields). Records append in walk
            // order and drain in the same order — NO re-sort. The value ids are
            // already minted in valueInterner during the walk (a different interner
            // from NameMap, so the decodeView subPairs stay valid).
            struct AmInsertRec {
                int32_t keyTmplId, depVldId, sourceStatus;
                int32_t depth, sec, flag;
                // The source value's ordisOnly tag, CARRIED through the
                // re-key — an ordis demand stays an ordis demand under the
                // canonical name.
                int32_t ordis;
                int32_t keyStart, keyCount, remStart, remCount;
            };

            // Decoded lex-sorted snapshot off the malloc heap (mirrors the
            // integration rmi/ami hooks): the (template, validity) rows intern
            // into a per-slot scratch-arena table and a byte-bump index is
            // sorted over their decoded spans. compareSpans reproduces the
            // former `a.first < b.first` pair<string,string> lex EXACTLY
            // (template then validity); that walk order drives the
            // revisitRejected2 emission order (never id order, I-84).
            struct AmSortRow { int32_t tmplId; int32_t vldId; int64_t pk; };
            const unsigned amSlot = (gl::WorkerContext::coreId() >= 0)
                ? static_cast<unsigned>(gl::WorkerContext::coreId())
                : genScratchArenas().slotCount() - 1;
            ScratchArena& amArena = genScratchArenas().forSlot(amSlot);
            ScratchArena& amStrArena = scratchArenas().forSlot(amSlot);
            DirtyState amDirty = DirtyState::Clean;
            // Keys to drop from admissionMap — scratch int set on the same
            // arena; membership erase is order-independent (byte-identical to
            // the former heap vector + unordered_set).
            TypedColdSet<int64_t> eraseSet(&amArena, &amDirty);
            // Canonical replacement run on the arena (member count is not
            // arity-bounded -> byte-bump run, not a stack array): every
            // non-canonical member -> the chooseCanonical representative. Member
            // NAMES are stable NameMap spans (decodeView; no NameMap mint before
            // the post-loop drain). Each `ce::replaceKeysInString(x, substMap)`
            // below becomes `replaceKeysToString(StrSpan(x), subPairs, subCount)`
            // — the proven byte-exact, source-scanned, order-independent twin.
            // subPairs iterates memberIds (decoded-lex) = the former std::map key
            // order.
            const StrSpan amCanonSpan = memoryBlock.nameMap.decodeView(canonId);
            const int32_t amMemberN = clss.memberCount();
            StrReplacement* subPairs = (amMemberN == 0) ? nullptr
                : reinterpret_cast<StrReplacement*>(amArena.resolve(amArena.alloc(
                      amMemberN * static_cast<int32_t>(sizeof(StrReplacement)),
                      static_cast<int32_t>(alignof(StrReplacement)))));
            int32_t subCount = 0;
            for (int32_t mIdx_ = 0; mIdx_ < clss.memberCount(); ++mIdx_) { const NameId mid = clss.memberId(mIdx_);
                const StrSpan mv = memoryBlock.nameMap.decodeView(mid);
                if (!equalSpans(mv, amCanonSpan)) {
                    subPairs[subCount].key = mv;
                    subPairs[subCount].value = amCanonSpan;
                    ++subCount;
                }
            }
            if (subCount == 0) return;
            ColdHashSet<BytesKeyStore> amStrings(&amArena, &amDirty);
            PagedVector<AmSortRow> amRows(&amArena, &amDirty);
            // Insert-staging pools (page-tier). Drained after the walk.
            PagedVector<int32_t> amKeyPool(&amArena, &amDirty);
            PagedVector<int32_t> amRemPool(&amArena, &amDirty);
            PagedVector<AmInsertRec> amInserts(&amArena, &amDirty);
            const int32_t amStoredCount = selected == nullptr ? am.count() : selected->count();
            for (int32_t id = 1; id <= amStoredCount; ++id) {
                const int64_t pkRow = selected == nullptr ? am.keyAt(id) : selected->keyAt(id);
                if (am.lookup(pkRow) == 0) continue;
                if (memoryBlock.phase3IgnoresScope(Codec<StatementKey>::decode(pkRow).validity)) continue;
                StrSpan tKey, tVal;
                decodeTemplateKeyView(pkRow, memoryBlock.templateInterner,
                    memoryBlock.nameMap, tKey, tVal);
                amRows.push_back(AmSortRow{ amStrings.mint(tKey),
                    amStrings.mint(tVal), pkRow });
            }
            const int32_t amCount = amRows.size();
            ScratchScope amSortScope(amArena);
            // Block-chunked sorted index — the rmi hook's shape (see there);
            // distinct pks ⇒ strict total order ⇒ byte-identical sequence.
            const auto amLess = [&](int32_t a, int32_t b) {
                const AmSortRow& ra = amRows[a];
                const AmSortRow& rb = amRows[b];
                const int ct = compareSpans(amStrings.keyAt(ra.tmplId),
                                            amStrings.keyAt(rb.tmplId));
                if (ct != 0) return ct < 0;
                return compareSpans(amStrings.keyAt(ra.vldId),
                                    amStrings.keyAt(rb.vldId)) < 0;
            };
            ChunkSortedOrdinals<decltype(amLess)> amOrder(
                amArena, amCount, 0, amLess);

            for (int32_t amII = 0; amII < amCount; ++amII) {
                // Per-entry string window (see the rmi hook): free the minted-
                // then-dead rewrite ScratchStrings each entry.
                ScratchScope amStrScope(amStrArena);
                const AmSortRow& amCur = amRows[amOrder.next()];
                const int64_t pk = amCur.pk;
                const StrSpan amTmplSpan = amStrings.keyAt(amCur.tmplId);
                const StrSpan amVldSpan = amStrings.keyAt(amCur.vldId);
                // tv.first dropped: markedKey's only reads (the '(' guard, the
                // Anchor guard, the rewrite) all precede this entry's
                // amStrings.mint (the K' deposit), so the interned template span
                // is used directly — no amStrings mint intervenes (batch-7 rmi
                // recipe). tv.second is copied onto the string tier (amStrArena):
                // deeperOf's depositValidity aliases it and is held across that
                // mint, so it must ride a DIFFERENT arena than amStrings (I-3
                // copyFrom; amVldSpan aliases amStrings).
                const StrSpan markedKey = amTmplSpan;
                const ScratchString amVldCopy =
                    ScratchString::copyFrom(amStrArena, amVldSpan.ptr, amVldSpan.len);

                // Apply only when the class and the key share the same scope
                // (identical validityName). Cross-scope equalities are folded
                // into the maintained complete view for the target scope,
                // so key application is same-validityName only.
                if (!equalSpans(StrSpan(amVldCopy), validityName)) continue;

                if (markedKey.empty() || markedKey[0] != '(') continue;

                // Anchor predicates are positional scope identities, never
                // rewritten by an equivalence class (D-82). Admission keys are
                // operator-marker keys; this is a guard for parity.
                const StrSpan baseExpr = extractExpressionSpan(markedKey);
                if (baseExpr.len >= 6 && std::memcmp(baseExpr.ptr, "Anchor", 6) == 0) continue;

                const auto normalizeValue = [&](const AdmissionValueBlobView& value,
                        StrSpan* chain, StrSpan* remaining, bool& excluded) {
                    assert(value.keyCount() <= ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS
                        && value.remCount() <= ExecutionParameters::MAX_ADMISSION_REM_ARGS);
                    StrSpan originalChain[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
                    StrSpan originalRemaining[ExecutionParameters::MAX_ADMISSION_REM_ARGS];
                    for (int32_t i = 0; i < value.keyCount(); ++i)
                        originalChain[i] = memoryBlock.valueInterner.decodeView(value.keyId(i));
                    for (int32_t i = 0; i < value.remCount(); ++i)
                        originalRemaining[i] = memoryBlock.valueInterner.decodeView(value.remId(i));
                    return canonicalTemplates.canonicalAdmissionChain(memoryBlock, originalChain, value.keyCount(),
                        originalRemaining, value.remCount(), memoryBlock.nameMap.lookup(validityName),
                        amStrArena, chain, remaining, excluded, subPairs, subCount);
                };
                const ScratchString rewrittenMarkedKey =
                    replaceKeysScratch(amStrArena, markedKey, subPairs, subCount);
                const bool keyChanged = !equalSpans(StrSpan(rewrittenMarkedKey), markedKey);
                bool valueChanged = false;
                if (!keyChanged) {
                    ScratchScope probe(amArena);
                    const AdmissionRunSnapshot values = snapshotAdmissionRun(am, pk, amArena);
                    for (int32_t r = 0; r < values.count && !valueChanged; ++r) {
                        const auto& row = values.views[r];
                        StrSpan chain[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
                        StrSpan remaining[ExecutionParameters::MAX_ADMISSION_REM_ARGS];
                        bool refused = false;
                        const int32_t remainingN = normalizeValue(row, chain, remaining, refused);
                        if (refused) continue;
                        bool rowChanged = remainingN != row.remCount();
                        for (int32_t j = 0; j < row.keyCount() && !rowChanged; ++j)
                            rowChanged = !equalSpans(chain[j], memoryBlock.valueInterner.decodeView(row.keyId(j)));
                        for (int32_t j = 0; j < remainingN && !rowChanged; ++j)
                            rowChanged = !equalSpans(remaining[j], memoryBlock.valueInterner.decodeView(row.remId(j)));
                        valueChanged = rowChanged;
                    }
                    if (!valueChanged) continue;
                }

                const StrSpan depositValidity =
                    memoryBlock.nameMap.deeperOf(StrSpan(validityName), StrSpan(amVldCopy));

                // Capture K's status before the post-loop erase removes it.
                const uint8_t* itSrcStatus =
                    memoryBlock.overallHashMemory.admissionStatusMap.find(pk);
                const bool sourceStatus =
                    (itSrcStatus != nullptr) ? (*itSrcStatus != 0) : false;

                eraseSet.mint(pk);  // the changed key K is dropped

                // Apply substMap to each AdmissionMapValue. u_-prefixed args
                // inside value.key / value.remainingArgs are naturally
                // untouched (substMap keys are bare class members). am is NOT
                // mutated in this walk (the erase/insert drain is deferred
                // post-loop), so a blob-view snapshot on amArena is safe;
                // snapshotAdmissionRun copies the run's blobs verbatim (uniform
                // with the recursion caller). admissionRecordsAt stays the
                // retained oracle; the run is canonical so the view order
                // reproduces the former set order.
                const ArenaOffset amSnapMark = amArena.cursor();
                const AdmissionRunSnapshot amSnap =
                    snapshotAdmissionRun(am, pk, amArena);
                for (int32_t amSnapIx = 0; amSnapIx < amSnap.count; ++amSnapIx) {
                    const AdmissionValueBlobView& v = amSnap.views[amSnapIx];
                    // Rewrite the value's key elements STRAIGHT into amKeyPool —
                    // no owning AdmissionMapValue::key vector. valueInterner.encode
                    // mints (a DIFFERENT interner than the amStrings the amVldCopy
                    // protects), and each elem span is read before its own encode
                    // (I-3-safe). If the substitution would CHANGE an Anchor
                    // element of this value, REFUSE it (D-82: anchor slots are
                    // fixed positional identities) — the partial key run appended
                    // so far is rolled back by truncate.
                    const int32_t keyStart = static_cast<int32_t>(amKeyPool.size());
                    StrSpan normalizedKey[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
                    StrSpan normalizedRemaining[ExecutionParameters::MAX_ADMISSION_REM_ARGS];
                    bool anchorChanged = false;
                    const int32_t normalizedRemainingN = normalizeValue(v, normalizedKey, normalizedRemaining, anchorChanged);
                    if (!anchorChanged) for (int32_t ke = 0; ke < v.keyCount(); ++ke)
                        amKeyPool.push_back(memoryBlock.valueInterner.encode(normalizedKey[ke]));
                    if (anchorChanged) {
                        amKeyPool.truncate(keyStart);   // refuse this value
                        if (!keyChanged) {
                            // Another value changed under this same key. The
                            // excluded row retains its original positional anchor.
                            for(int32_t j=0;j<v.keyCount();++j) amKeyPool.push_back(v.keyId(j));
                            const int32_t remStart=amRemPool.size();
                            for(int32_t j=0;j<v.remCount();++j) amRemPool.push_back(v.remId(j));
                            amInserts.push_back(AmInsertRec{amStrings.mint(StrSpan(rewrittenMarkedKey)),
                                amStrings.mint(depositValidity),sourceStatus ? 1 : 0,
                                v.depth(),v.sec(),v.flagByte(),v.ordisByte()!=0 ? 1 : 0,
                                keyStart,v.keyCount(),remStart,v.remCount()});
                        }
                        continue;
                    }
                    const int32_t keyCount = static_cast<int32_t>(amKeyPool.size()) - keyStart;

                    const int32_t remStart = static_cast<int32_t>(amRemPool.size());
                    for (int32_t j = 0; j < normalizedRemainingN; ++j)
                        amRemPool.push_back(memoryBlock.valueInterner.encode(normalizedRemaining[j]));
                    const int32_t remCount = static_cast<int32_t>(amRemPool.size()) - remStart;

                    amInserts.push_back(AmInsertRec{
                        amStrings.mint(StrSpan(rewrittenMarkedKey)),
                        amStrings.mint(StrSpan(depositValidity)),
                        sourceStatus ? 1 : 0,
                        v.depth(), v.sec(), v.flagByte(),
                        (v.ordisByte() != 0) ? 1 : 0,
                        keyStart, keyCount,
                        remStart, remCount });
                }
                amArena.popTo(amSnapMark);
            }

            // Post-loop: drop each changed key K from admissionMap and the
            // parallel admissionStatusMap, then insert the canonical K' with
            // K's status moved onto it, cache populate, and revisitRejected2.
            // K' is canonical, so it is never itself one of the dropped keys.
            {
                for (int32_t esId = 1; esId <= eraseSet.count(); ++esId) {
                    memoryBlock.overallHashMemory.admissionStatusMap.erase(
                        eraseSet.decodeKey(esId));
                }
                memoryBlock.overallHashMemory.admissionMap.eraseBlobIf(
                    [&eraseSet](int64_t k) { return eraseSet.contains(k); });
            }
            for (int32_t ii = 0; ii < static_cast<int32_t>(amInserts.size()); ++ii) {
                const AmInsertRec& pi = amInserts[static_cast<std::size_t>(ii)];
                // keySpan / depSpan alias amStrings — stable through the drain
                // (no amStrings mint here; the mints are NameMap/templateInterner
                // via mintTemplateKey and valueInterner via insertAdmissionIdsBlob,
                // all DIFFERENT interners), so they feed the span doors directly.
                const StrSpan keySpan = amStrings.keyAt(pi.keyTmplId);
                const StrSpan depSpan = amStrings.keyAt(pi.depVldId);
                // Copy the two id runs out of the paged pools into contiguous
                // arena buffers (PagedVector has no cross-page contiguity
                // guarantee); the ids were minted in valueInterner DURING the
                // walk, so no mint fires here — the deposit order is settled and
                // the id-run blob door needs no value object.
                const ArenaOffset dMark = amArena.cursor();
                int32_t* keyIds = (pi.keyCount > 0)
                    ? reinterpret_cast<int32_t*>(amArena.resolve(
                          amArena.alloc(pi.keyCount * sizeof(int32_t), alignof(int32_t))))
                    : nullptr;
                for (int32_t j = 0; j < pi.keyCount; ++j)
                    keyIds[j] = amKeyPool[static_cast<std::size_t>(pi.keyStart + j)];
                int32_t* remIds = (pi.remCount > 0)
                    ? reinterpret_cast<int32_t*>(amArena.resolve(
                          amArena.alloc(pi.remCount * sizeof(int32_t), alignof(int32_t))))
                    : nullptr;
                for (int32_t j = 0; j < pi.remCount; ++j)
                    remIds[j] = amRemPool[static_cast<std::size_t>(pi.remStart + j)];

                // Write site — mint K'.
                const int64_t newPk = mintTemplateKey(memoryBlock.templateInterner,
                    memoryBlock.nameMap, keySpan, depSpan);
                insertAdmissionIdsBlob(memoryBlock.overallHashMemory.admissionMap,
                                       newPk, pi.depth, pi.sec, static_cast<uint8_t>(pi.flag),
                                       keyIds, pi.keyCount, remIds, pi.remCount,
                                       memoryBlock.valueInterner, amArena,
                                       pi.ordis != 0);
                memoryBlock.lbMemory.canonicalUses.algebraDemands.changed(newPk);
                amArena.popTo(dMark);

                // Move the status onto K', only if K' has no status entry yet
                // (preserve an existing one).
                if (memoryBlock.overallHashMemory.admissionStatusMap.find(newPk)
                    == nullptr) {
                    memoryBlock.overallHashMemory.admissionStatusMap.upsert(
                        newPk, static_cast<uint8_t>(pi.sourceStatus ? 1 : 0));
                }

                // Populate vars-in-keys cache for K'.
                StrSpan ktArgs[ExecutionParameters::MAX_ARITY];
                const int32_t ktArgsN = getArgsSpans(keySpan, ktArgs,
                                                     ExecutionParameters::MAX_ARITY);
                for (int32_t ai = 0; ai < ktArgsN; ++ai) {
                    if (!equalSpans(ktArgs[ai], StrSpan("marker", 6))) {
                        memoryBlock.overallHashMemory.varsInAdmissionMapKeys.mint(memoryBlock.templateInterner.encode(ktArgs[ai]));
                    }
                }

                // Fire revival: walk rejectedMap[K'] and queue matching
                // source existences via revisitRejected2. Admission keys are
                // already bare-marker form, so no u_-strip is needed. An
                // ordis-tagged re-insert must not probe the GENERAL
                // rejectedMap (its revival seam is revisitRejectedOrdis,
                // which wakes on EVERY key gain — tagged or untagged).
                if (pi.ordis == 0) {
                    rejectedSourceReplay.revisitRejected2(keySpan, memoryBlock, depSpan);
                }
                rejectedSourceReplay.revisitRejectedOrdis(keySpan, memoryBlock, depSpan);
            }
        }

// Integration equi-class hook on admissionMapIntegration. Mirror of
        // algebra applyEquivalenceClassToAdmissionMap (D-57), with three
        // adjustments for integration-side data types:
        //
        //   1. admissionMapIntegration keys are u_-prefixed (built by
        //      disintegrateExprCore2's C1 block from le.signature with one
        //      nonUArg replaced by "marker"). The substMap from a class has
        //      bare-name keys — they would never match u_-prefixed args
        //      directly. We strip u_ from the key to obtain bareKey, rewrite
        //      it once with the single canonical substMap (chooseCanonical),
        //      then re-apply u_ to obtain the u_-form K'.
        //   2. admissionMapIntegration values are
        //      map<Instruction, set<string>> rather than
        //      set<AdmissionMapValue>. Each (Instruction, appliedVars) pair
        //      is rewritten independently. Instruction.data[*].signature /
        //      Instruction.data[*].elements[*] / Instruction.markedGoal are
        //      u_-form strings; appliedVars members are bare names per the
        //      construction sites in updateAdmissionMapIntegration and
        //      isAdmittedIntegration. We build an augmented substMap that
        //      includes both bare → canon and u_<bare> → u_<canon> pairs so
        //      both bare and u_-form occurrences get substituted
        //      consistently.
        //   3. revisitRejectedIntegration2 is called after each K' insert
        //      (mirror of revisitRejected2 in the algebra hook). bareK' is
        //      used because rejectedMapIntegration keys are bare-marker
        //      form, not u_-form.
        //
        // Design principles (mirror of the algebra admission hook, with the
        // I-22 asymmetry preserved):
        //   1. rejectedMapIntegration is sacred — never written by this
        //      function (it is rewritten via
        //      applyEquivalenceClassToRejectedMapIntegration which now
        //      drops+mails per the revised I-37).
        //   2. admissionMapIntegration keys and Instruction values are
        //      metadata — equivalence-class rewrites apply freely.
        //   3. One canonical representative per class (chooseCanonical): the
        //      changed key is DROPPED and the canonical K' inserted (re-key),
        //      not additive. Applies only at the same validityName. See
        //      D-106.
        //   4. Slot collapses are allowed (no arg-equalization filter): both
        //      sides canonicalise identically, so the positional-collision
        //      preservation is no longer load-bearing.
        //
        // I-22: integration admission entries persist across revival firings —
        // a single template can admit multiple distinct int_ witnesses over its
        // lifetime. No path erases an admission key.
template <class ClassT>
        inline void EquivalenceAdmissionRewriter::applyEquivalenceClassToAdmissionMapIntegration(
            const ClassT& clss,
            Memory& memoryBlock,
            StrSpan validityName,
            const TypedColdSet<int64_t>* selected) {

            const NameId phase3Scope = memoryBlock.nameMap.lookup(validityName);
            if (phase3Scope != 0 && memoryBlock.phase3IgnoresScope(phase3Scope))
                return;

            auto& ami = memoryBlock.overallHashMemory.admissionMapIntegration;
            if (ami.count() == 0) return;

            // One canonical representative for the whole class (normal > int_
            // > it_, lex-smallest, strong members): the same chooseCanonical
            // selection the algebra hooks use. The bare substMap sends every
            // other member to it.
            const NameId canonId = canonicalizer.chooseCanonicalId(clss, memoryBlock, validityName);
            if (canonId == 0) return;

            // Insert staging on the arena (drained post-walk; the drain mints in
            // NameMap / templateInterner / valueInterner, so it stays deferred —
            // the arena interner is NOT the NameMap). Strings intern into
            // `amiStrings`; the working instruction is flattened into an int
            // run by flattenWorkInstructionInto (layout |data|, then per entity:
            // category, signature, arity, definedSet, |elements|, elements...;
            // markedGoal LAST) and consumed at the findOrEmplaceFlattened
            // boundary; newAppliedVars ride a sorted interned-id run
            // (set-iteration order preserved). Records append + drain in walk
            // order — NO re-sort.
            struct AmiInsertRec {
                int32_t uFormId, bareId, depVldId;
                int32_t instrStart, instrLen;
                int32_t appStart, appCount;
            };

            // Decoded (template,validity) lex-sorted snapshot, off the heap onto
            // the per-slot gen arena (the disint forEachVarSorted idiom B7 uses
            // for the rejected hook). The insert order below drives the
            // revisitRejectedIntegration2 emission order, which the former
            // EWV-keyed std::map iteration fixed at (template,validity) lex order;
            // compareSpans reproduces that std::string lex exactly, and the
            // pk <-> (template,validity) bijection means no sort ties.
            struct AmiSortRow { int32_t tmplId; int32_t vldId; int64_t pk; };
            const unsigned amiSlot = (gl::WorkerContext::coreId() >= 0)
                ? static_cast<unsigned>(gl::WorkerContext::coreId())
                : genScratchArenas().slotCount() - 1;
            ScratchArena& amiArena = genScratchArenas().forSlot(amiSlot);
            ScratchArena& amiStrArena = scratchArenas().forSlot(amiSlot);
            DirtyState amiDirty = DirtyState::Clean;
            // Keys to drop from admissionMapIntegration — scratch int set;
            // membership erase is order-independent (byte-identical to the
            // former heap vector + unordered_set).
            TypedColdSet<int64_t> eraseSet(&amiArena, &amiDirty);
            // Canonical replacement runs on the arena (member count is not
            // arity-bounded -> byte-bump runs). `subPairs` = bare member ->
            // canon (bareKey gate + applied vars). `augPairs` = the same bare
            // pairs FOLLOWED by u_<member> -> u_<canon> (the Instruction
            // signature / elements / markedGoal are u_-form): the first subCount
            // entries ARE the bare pairs, so `subPairs == augPairs` and a bare-
            // only use passes count `subCount`, an augmented use passes
            // `augCount == 2*subCount`. Bare names are stable NameMap spans
            // (decodeView; no NameMap mint before the post-loop drain); the
            // u_-prefixed names are materialized on the arena. Every
            // ce::replaceKeysInString(x, map) -> replaceKeysToString(StrSpan(x),
            // run, n), the proven byte-exact order-independent twin.
            const StrSpan amiCanonSpan = memoryBlock.nameMap.decodeView(canonId);
            const int32_t amiMemberN = clss.memberCount();
            const auto arenaUPrefix = [&](const StrSpan& s) -> StrSpan {
                const int32_t ul = 2 + s.len;
                char* ub = reinterpret_cast<char*>(amiArena.resolve(amiArena.alloc(ul, 1)));
                ub[0] = 'u'; ub[1] = '_';
                if (s.len > 0) std::memcpy(ub + 2, s.ptr, static_cast<std::size_t>(s.len));
                return StrSpan(ub, ul);
            };
            StrReplacement* augPairs = (amiMemberN == 0) ? nullptr
                : reinterpret_cast<StrReplacement*>(amiArena.resolve(amiArena.alloc(
                      2 * amiMemberN * static_cast<int32_t>(sizeof(StrReplacement)),
                      static_cast<int32_t>(alignof(StrReplacement)))));
            StrReplacement* subPairs = augPairs;
            int32_t subCount = 0;
            for (int32_t mIdx_ = 0; mIdx_ < clss.memberCount(); ++mIdx_) { const NameId mid = clss.memberId(mIdx_);
                const StrSpan mv = memoryBlock.nameMap.decodeView(mid);
                if (!equalSpans(mv, amiCanonSpan)) {
                    subPairs[subCount].key = mv;
                    subPairs[subCount].value = amiCanonSpan;
                    ++subCount;
                }
            }
            if (subCount == 0) return;
            const StrSpan amiUCanonSpan = arenaUPrefix(amiCanonSpan);
            for (int32_t si = 0; si < subCount; ++si) {
                augPairs[subCount + si].key = arenaUPrefix(subPairs[si].key);
                augPairs[subCount + si].value = amiUCanonSpan;
            }
            const int32_t augCount = 2 * subCount;
            ColdHashSet<BytesKeyStore> amiStrings(&amiArena, &amiDirty);
            PagedVector<AmiSortRow> amiRows(&amiArena, &amiDirty);
            // Insert-staging pools (page-tier). Drained after the walk.
            PagedVector<int32_t> amiInstrPool(&amiArena, &amiDirty);  // flattened IntInstruction runs
            PagedVector<int32_t> amiAppPool(&amiArena, &amiDirty);    // interned newAppliedVars ids
            PagedVector<AmiInsertRec> amiInserts(&amiArena, &amiDirty);
            const int32_t amiStoredCount = selected == nullptr ? ami.count() : selected->count();
            for (int32_t id = 1; id <= amiStoredCount; ++id) {
                const int64_t pk = selected == nullptr ? ami.keyAt(id) : selected->keyAt(id);
                if (ami.lookup(pk) == 0) continue;
                if (memoryBlock.phase3IgnoresScope(Codec<StatementKey>::decode(pk).validity)) continue;
                StrSpan tKey, tVal;
                decodeTemplateKeyView(pk, memoryBlock.templateInterner,
                    memoryBlock.nameMap, tKey, tVal);
                amiRows.push_back(AmiSortRow{ amiStrings.mint(tKey),
                    amiStrings.mint(tVal), pk });
            }
            const int32_t amiCount = amiRows.size();
            ScratchScope amiSortScope(amiArena);
            // Block-chunked sorted index — the rmi hook's shape (see there);
            // distinct pks ⇒ strict total order ⇒ byte-identical sequence.
            const auto amiLess = [&](int32_t a, int32_t b) {
                const AmiSortRow& ra = amiRows[a];
                const AmiSortRow& rb = amiRows[b];
                const int ct = compareSpans(amiStrings.keyAt(ra.tmplId),
                                            amiStrings.keyAt(rb.tmplId));
                if (ct != 0) return ct < 0;
                return compareSpans(amiStrings.keyAt(ra.vldId),
                                    amiStrings.keyAt(rb.vldId)) < 0;
            };
            ChunkSortedOrdinals<decltype(amiLess)> amiOrder(
                amiArena, amiCount, 0, amiLess);

            for (int32_t amiII = 0; amiII < amiCount; ++amiII) {
                // Per-entry string window (see the rmi hook): free the minted-
                // then-dead rewrite ScratchStrings each entry.
                ScratchScope amiStrScope(amiStrArena);
                const AmiSortRow& amiCur = amiRows[amiOrder.next()];
                const int64_t pk = amiCur.pk;
                const StrSpan amiTmplSpan = amiStrings.keyAt(amiCur.tmplId);
                const StrSpan amiVldSpan = amiStrings.keyAt(amiCur.vldId);
                ArenaIntegrationMap innerMap(
                    &genScratchArenas().forSlot(amiSlot), &memoryBlock.valueInterner);
                innerMap.buildFromCold(ami, pk);

                // Apply only when the class and the entry share the same scope
                // (identical validityName) -- same rule as the algebra hooks.
                if (!equalSpans(amiVldSpan, validityName)) continue;

                const StrSpan markedKey = amiTmplSpan;
                if (markedKey.len == 0 || markedKey.ptr[0] != '(') continue;

                // Anchor predicates are positional scope identities, never
                // rewritten by an equivalence class (D-82).
                const StrSpan baseExpr = extractExpressionSpan(markedKey);
                if (baseExpr.len >= 6 && std::memcmp(baseExpr.ptr, "Anchor", 6) == 0) continue;

                // Strip u_ to a bare-form key so the bare substMap can fire,
                // rewrite once with the single canonical map, skip if unchanged.
                const ScratchString bareKey = gl::DefinitionExpansion::removeUPrefixScratch(amiStrArena, markedKey);
                const ScratchString rewrittenBareKey = replaceKeysScratch(amiStrArena, StrSpan(bareKey), subPairs, subCount);
                if (equalSpans(StrSpan(rewrittenBareKey), StrSpan(bareKey))) {
                    bool changed = false;
                    const auto probe = [&](int32_t id) {
                        const StrSpan text = memoryBlock.valueInterner.decodeView(id);
                        if (!equalSpans(text, StrSpan(replaceKeysScratch(amiStrArena, text, augPairs, augCount)))) changed = true;
                    };
                    for (int32_t i = 0; i < innerMap.entries.size(); ++i) {
                        probe(innerMap.entries[i].markedGoalId);
                        if (innerMap.entries[i].pathVarsId != 0) probe(innerMap.entries[i].pathVarsId);
                    }
                    for (int32_t i = 0; i < innerMap.les.size(); ++i) probe(innerMap.les[i].sigId);
                    for (int32_t i = 0; i < innerMap.elemCol.size(); ++i) probe(innerMap.elemCol[i]);
                    for (int32_t i = 0; i < innerMap.valNodes.size(); ++i) probe(innerMap.valNodes[i].valueId);
                    if (!changed) continue;
                }

                const StrSpan depositValidity(validityName);  // same scope

                // Restore u_ prefix on each non-marker bare arg in the rewrite.
                // rArgs is arity-bounded, so stack spans + a per-row scratch
                // scope hold the u_<arg> values (rewrittenUFormKey is copied out
                // before the scope rewinds). The augmented replacement run
                // `augPairs` (bare -> canon AND u_<bare> -> u_<canon>) was built
                // once above the walk: bare pairs substitute the bare
                // appliedVars, u_ pairs the u_-form signature / elements /
                // markedGoal.
                ScratchString rewrittenUFormKey;
                {
                    ScratchScope uScope(amiArena);
                    StrSpan rArgs[ExecutionParameters::MAX_ARITY];
                    const int32_t rArgN = getArgsSpans(StrSpan(rewrittenBareKey),
                        rArgs, ExecutionParameters::MAX_ARITY);
                    StrReplacement uPairs[ExecutionParameters::MAX_ARITY];
                    int32_t uCount = 0;
                    for (int32_t ai = 0; ai < rArgN; ++ai) {
                        const StrSpan a = rArgs[ai];
                        const bool isMarker = (a.len == 6
                            && std::memcmp(a.ptr, "marker", 6) == 0);
                        const bool isU = (a.len >= 2 && a.ptr[0] == 'u' && a.ptr[1] == '_');
                        if (isMarker || isU) continue;
                        uPairs[uCount].key = a;
                        uPairs[uCount].value = arenaUPrefix(a);
                        ++uCount;
                    }
                    rewrittenUFormKey = replaceKeysScratch(amiStrArena,
                        StrSpan(rewrittenBareKey), uPairs, uCount);
                }

                eraseSet.mint(pk);  // the changed key K is dropped

                int32_t innerOrder[ExecutionParameters::MAX_INTEGRATION_ENTRIES];
                const int32_t innerOrderN = innerMap.sortedIndices(
                    innerOrder, ExecutionParameters::MAX_INTEGRATION_ENTRIES);
                for (int32_t io = 0; io < innerOrderN; ++io) {
                    const int32_t innerIdx = innerOrder[io];
                    int32_t instrSecond[ExecutionParameters::MAX_INTEGRATION_ENTRY_VALUES];
                    const int32_t instrSecondN = innerMap.valuesAt(innerIdx,
                        instrSecond, ExecutionParameters::MAX_INTEGRATION_ENTRY_VALUES);
                    // Load the stored instruction straight into a WorkInstruction
                    // (no heap IntInstruction), then rewrite each field by augSubst
                    // into a fresh WorkInstruction and stage its flattened id run
                    // (the cold-insert boundary).
                    WorkInstruction wiSrc(&genScratchArenas().forSlot(amiSlot));
                    innerMap.loadInstructionInto(innerIdx, wiSrc, memoryBlock.valueInterner);

                    WorkInstruction newInstrWork(&genScratchArenas().forSlot(amiSlot));
                    for (int32_t ei = 0; ei < wiSrc.entityCount(); ++ei) {
                        const StrSpan sigSpan = wiSrc.signature(ei);
                        const ScratchString newSig = replaceKeysScratch(amiStrArena, sigSpan, augPairs, augCount);
                        const int32_t em = newInstrWork.elemMark();
                        const int32_t ec = wiSrc.elemCount(ei);
                        for (int32_t ej = 0; ej < ec; ++ej) {
                            const StrSpan elemSpan = wiSrc.elemAt(ei, ej);
                            const ScratchString newElem = replaceKeysScratch(amiStrArena, elemSpan, augPairs, augCount);
                            newInstrWork.addElement(StrSpan(newElem));
                        }
                        newInstrWork.commitEntity(wiSrc.category(ei), StrSpan(newSig),
                            wiSrc.definedSet(ei), wiSrc.arity(ei), em);
                    }
                    {
                        const StrSpan mgSpan = wiSrc.markedGoal();
                        newInstrWork.setMarkedGoal(StrSpan(replaceKeysScratch(
                            amiStrArena, mgSpan, augPairs, augCount)));
                    }
                    // The path list's names take their canonical form like the
                    // payload values, re-sorted and deduplicated
                    // (D-529).
                    if (!wiSrc.pathVariables().empty()) {
                        const ScratchString replaced = replaceKeysScratch(
                            amiStrArena, wiSrc.pathVariables(), subPairs, subCount);
                        StrSpan names[ExecutionParameters::MAX_PATH_VARIABLES];
                        const int32_t nameN = getArgsSpans(StrSpan(replaced), names,
                            ExecutionParameters::MAX_PATH_VARIABLES);
                        newInstrWork.setPathVariables(StrSpan(buildPathListScratch(amiStrArena, names, nameN)));
                    }
                    // The forbidden sets name original variables and block
                    // levels; the class re-key copies them unchanged
                    // (D-579).
                    newInstrWork.forbidden = wiSrc.forbidden;

                    // Mint newInstrWork's fields into valueInterner (frozen
                    // encodeWorkInstruction order: per entity category, elements,
                    // signature, definedSet; markedGoal LAST) and append the
                    // flattened id run to amiInstrPool — no heap IntInstruction.
                    // The mint happens at this exact loop position, so the
                    // walk-order of the mints is unchanged.
                    const ArenaIntegrationMap::FlatRange fr =
                        innerMap.flattenWorkInstructionInto(
                            newInstrWork, memoryBlock.valueInterner, amiInstrPool);
                    const int32_t instrStart = fr.start;
                    const int32_t instrLen = fr.len;
                    // newAppliedVars, statified: each value-interner member is
                    // rewritten onto the per-entry string arena, deduped +
                    // decoded-lex sorted, then interned into amiStrings in that
                    // order -- the byte-identical twin of the std::set<std::string>
                    // distinct-sorted staging (compareSpans == std::string::compare,
                    // I-84; adjacent-dup skip == std::set uniqueness). amiStrings
                    // mint order, hence amiAppPool, hence the post-loop
                    // valueInterner encode order, is unchanged.
                    const int32_t appStart = static_cast<int32_t>(amiAppPool.size());
                    {
                        ScratchScope avScope(amiStrArena);
                        const int32_t avN = instrSecondN;
                        StrSpan* avMembers = (avN == 0) ? nullptr
                            : reinterpret_cast<StrSpan*>(amiStrArena.resolve(amiStrArena.alloc(
                                  avN * static_cast<int32_t>(sizeof(StrSpan)),
                                  static_cast<int32_t>(alignof(StrSpan)))));
                        for (int32_t i = 0; i < avN; ++i) {
                            const StrSpan av = memoryBlock.valueInterner.decodeView(instrSecond[i]);
                            avMembers[i] =
                                StrSpan(replaceKeysScratch(amiStrArena, av, subPairs, subCount));
                        }
                        int32_t* avOrder = (avN == 0) ? nullptr
                            : reinterpret_cast<int32_t*>(amiStrArena.resolve(amiStrArena.alloc(
                                  avN * static_cast<int32_t>(sizeof(int32_t)),
                                  static_cast<int32_t>(alignof(int32_t)))));
                        for (int32_t i = 0; i < avN; ++i) avOrder[i] = i;
                        std::sort(avOrder, avOrder + avN, [&avMembers](int32_t a, int32_t b) {
                            return compareSpans(avMembers[a], avMembers[b]) < 0;
                        });
                        for (int32_t i = 0; i < avN; ++i) {
                            if (i > 0 && equalSpans(avMembers[avOrder[i]],
                                                    avMembers[avOrder[i - 1]]))
                                continue;   // std::set uniqueness (skip adjacent dup)
                            amiAppPool.push_back(amiStrings.mint(avMembers[avOrder[i]]));
                        }
                    }
                    const int32_t appCount = static_cast<int32_t>(amiAppPool.size()) - appStart;
                    amiInserts.push_back(AmiInsertRec{
                        amiStrings.mint(StrSpan(rewrittenUFormKey)),
                        amiStrings.mint(StrSpan(rewrittenBareKey)),
                        amiStrings.mint(StrSpan(depositValidity)),
                        instrStart, instrLen, appStart, appCount });
                }
            }

            // Post-loop: drop each changed key K, then insert the canonical K'
            // (re-key). No admissionStatusMap on the integration side -- nothing
            // to move. I-22 preserved: the canonical K' persists for future
            // revivals; revisitRejectedIntegration2 does not clean it. K' is
            // canonical, so it is never itself one of the dropped keys.
            memoryBlock.overallHashMemory.admissionMapIntegration.eraseBlobIf(
                [&eraseSet](int64_t k) { return eraseSet.contains(k); });
            for (int32_t ii = 0; ii < static_cast<int32_t>(amiInserts.size()); ++ii) {
                const AmiInsertRec& pi = amiInserts[static_cast<std::size_t>(ii)];
                const StrSpan uSpan = amiStrings.keyAt(pi.uFormId);
                const StrSpan bareSpan = amiStrings.keyAt(pi.bareId);
                const StrSpan depSpan = amiStrings.keyAt(pi.depVldId);
                // The rewritten instruction stays FLAT in amiInstrPool (the
                // flattenWorkInstructionInto / Door-2 layout) — its run is copied
                // verbatim into a byte-bump run at the findOrEmplaceFlattened
                // boundary below; no heap IntInstruction is rebuilt.

                // Cold round trip — decode K''s run, add the rewritten
                // instruction + applied vars, write back. Items targeting the
                // same K' accumulate: each re-decodes the prior write-back.
                // All three drained keys are span-fed from amiStrings (uSpan /
                // bareSpan / depSpan); no newKeyUForm / newKeyBare /
                // depositValidity std::string is materialized.
                const int64_t newPk = mintTemplateKey(memoryBlock.templateInterner,
                    memoryBlock.nameMap, uSpan, depSpan);
                const unsigned innerSlot = (gl::WorkerContext::coreId() >= 0)
                    ? static_cast<unsigned>(gl::WorkerContext::coreId())
                    : genScratchArenas().slotCount() - 1;
                ArenaIntegrationMap innerMap(
                    &genScratchArenas().forSlot(innerSlot), &memoryBlock.valueInterner);
                innerMap.buildFromCold(
                    memoryBlock.overallHashMemory.admissionMapIntegration, newPk);
                // Stage the flattened run (amiInstrPool is a PagedVector — a page
                // straddle is possible) into a contiguous byte-bump run on
                // innerMap's own arena, then find-or-emplace it (mints NOTHING —
                // the ids were minted at flatten time in C2). appendEntryFlattened
                // copies the ids into innerMap's page-tier records, so the scope
                // rewinds with nothing dangling (byte-bump/page-tier coexistence).
                int32_t targetEntry = 0;
                {
                    ScratchScope c3Scope(*innerMap.arena);
                    int32_t* run = reinterpret_cast<int32_t*>(innerMap.arena->resolve(
                        innerMap.arena->alloc(
                            pi.instrLen * static_cast<int32_t>(sizeof(int32_t)),
                            static_cast<int32_t>(alignof(int32_t)))));
                    for (int32_t k = 0; k < pi.instrLen; ++k)
                        run[k] = amiInstrPool[static_cast<std::size_t>(pi.instrStart + k)];
                    targetEntry = innerMap.findOrEmplaceFlattened(run, pi.instrLen);
                }
                for (int32_t j = 0; j < pi.appCount; ++j) {
                    const StrSpan avSpan = amiStrings.keyAt(
                        amiAppPool[static_cast<std::size_t>(pi.appStart + j)]);
                    innerMap.insertValue(targetEntry,
                        memoryBlock.valueInterner.encode(avSpan));
                }
                innerMap.writeToCold(
                    memoryBlock.overallHashMemory.admissionMapIntegration, newPk);
                memoryBlock.lbMemory.canonicalUses.integrationDemands.changed(newPk);

                // Populate the bare-form cache for K'.
                StrSpan kbArgs[ExecutionParameters::MAX_ARITY];
                const int32_t kbArgsN = getArgsSpans(bareSpan, kbArgs,
                                                     ExecutionParameters::MAX_ARITY);
                bool wildcardKey = false;
                for (int32_t ai = 0; ai < kbArgsN; ++ai) {
                    if (equalSpans(kbArgs[ai], StrSpan("pi_", 3))) wildcardKey = true;
                    if (!equalSpans(kbArgs[ai], StrSpan("marker", 6))) {
                        memoryBlock.overallHashMemory.varsInAdmissionMapIntegrationKeys.mint(memoryBlock.templateInterner.encode(kbArgs[ai]));
                    }
                }

                // Fire revival on the bare K' (rejectedMapIntegration keys are
                // bare-marker form). Per I-22 the integration admission entry
                // persists for future revivals. A wildcard K' names no park
                // text, so its parks are woken by predicate
                // (D-586).
                rejectedSourceReplay.revisitRejectedIntegration2(bareSpan, memoryBlock, depSpan);
                if (wildcardKey)
                    rejectedSourceReplay.wakeWildcardIntegrationParks(memoryBlock, bareSpan, depSpan);
                // K' is visible below its scope too
                // (D-572): wake the parks there.
                rejectedSourceReplay.wakeDescendantIntegrationParks(memoryBlock, bareSpan, depSpan);
            }
        }

/// @brief Apply class substitutions to parked existence sources.
        /// @details Retains deferred provenance and wakes the ordinary replay path.
        /// @tparam ClassT Read-only equivalence-class view.
        /// @param clss Class supplying substitutions.
        /// @param memoryBlock Owning memory block.
        /// @param validityName Class scope.
        /// @param buckets Source-key scope index.
        /// @return Number of rewritten keys.
template <class ClassT>
        inline int32_t EquivalenceAdmissionRewriter::applyEquivalenceClassToRejectedMap(
            const ClassT& clss, Memory& memoryBlock, StrSpan validityName,
            RejectedValidityBuckets& buckets,
            const TypedColdSet<int64_t>* selected) {
            return rewriteParkedExistences<false>(clss, memoryBlock, validityName, buckets, selected);
        }

/// @brief Shared core of the two parked-or equi-class hooks — I-37
        ///        drop + mail for the map handed in, read through the
        ///        per-apply substitution-token posting index instead of a
        ///        whole-map walk.
        ///
        /// @details
        /// A hit is a key of @p rmo at the class's own validity whose
        /// template changes under the class's canonical substitution. The
        /// candidates come from @p buckets, the per-apply
        /// `RejectedValidityBuckets` whose postings map `(validity, token id)`
        /// to the keys whose template carries that token
        /// (`forEachSubstitutionToken` — the boundary rule of
        /// `replaceKeysScratch`, so a template changes iff one of its tokens
        /// IS a non-canonical member): the postings of every non-canonical
        /// member at the class validity, sorted and deduplicated, each
        /// re-checked with the substitution test — the hit set is exactly
        /// the whole-map walk's. The index is refreshed through
        /// `RejectedValidityBuckets::ensureCurrent`: the park site may insert
        /// into the ordis maps inside an apply, and a moved insert epoch
        /// rebuilds the index, so a key parked earlier in the same apply is
        /// visited exactly as the whole-map walk visited it; a key dropped
        /// since the build is a `lookup` miss. No round skip: every class
        /// call probes the index.
        ///
        /// Each hit DROPS the entry (batched `eraseBlobIf`) and MAILS the
        /// rewritten parked OR STATEMENT onto `sameIterationInternalMail`
        /// with an `equality1` history line — never a direct re-insert. The
        /// absorb re-runs the full or consumption on the rewritten statement
        /// (fresh probe, standing tie rule, K rules deduped), which re-parks
        /// or admits it under the canonical keys through the normal ordis
        /// path. A key's non-marker tokens are drawn from the or statement's
        /// own arguments, so a rewrite that fires on the key fires on the
        /// statement too; the unchanged-statement skip mirrors the
        /// rejectedMap hook's defined skip. Hits are processed in template
        /// lex order — the former (template, validity) order restricted to
        /// the one validity — so the mail sequence is unchanged. Canonical
        /// selection, staging discipline (arena-interned records, deferred
        /// NameMap-minting drain per I-3) and the fired-substitutions-only
        /// `equality1` tail mirror `applyEquivalenceClassToRejectedMap`.
        ///
        /// @tparam ClassT The equivalence-class read interface
        ///                (`memberCount` / `memberId` / `forEachLevel`).
        /// @tparam MapT   The parked-or map type (`rejectedMapOrdis` and
        ///                `rejectedMapOrdis2` share it: a `TypedColdBlobMap`
        ///                keyed by the packed `(templateId, validityId)`).
        /// @param clss         The (delta) class driving the rewrite.
        /// @param memoryBlock  The owning LB.
        /// @param validityName The class's scope; only same-scope entries
        ///                     rewrite.
        /// @param rmo          The parked-or map to rewrite.
        /// @param buckets      The per-apply posting index over @p rmo (one
        ///                     per map per `applyEquiClasses` call).
        /// @return (void)
        /// @invariant Never writes @p rmo except the batched drop (I-37);
        ///            insertions happen only through the re-consumption of
        ///            the mailed statement. A class member name carries
        ///            neither `]` nor `,` (asserted) — the premise under
        ///            which the posting index is exact for the substitution.
        /// @see applyEquivalenceClassToRejectedMapOrdis,
        ///      applyEquivalenceClassToRejectedMapOrdis2 — the two doors;
        ///      applyEquivalenceClassToRejectedMap — the discipline template;
        ///      RejectedValidityBuckets::ensureCurrent — the freshness rule.
template <class ClassT, class MapT>
        inline void EquivalenceAdmissionRewriter::applyEquivalenceClassToRejectedOrdisCore(
            const ClassT& clss,
            Memory& memoryBlock,
            StrSpan validityName,
            MapT& rmo,
            RejectedValidityBuckets& buckets,
            const TypedColdSet<int64_t>* selected) {

            const NameId classVid = memoryBlock.nameMap.lookup(validityName);
            assert(classVid != 0
                && "ordis hook: a class validity is always a minted name");
            if (memoryBlock.phase3IgnoresScope(classVid)) return;

            if (rmo.empty()) return;

            const NameId canonId = canonicalizer.chooseCanonicalId(clss, memoryBlock, validityName);
            if (canonId == 0) return;

            struct MailRec {
                int32_t postId, preId;
                int32_t lvlStart, lvlCount, eqStart, eqCount;
            };

            struct RmoSortRow { int32_t tmplId; int64_t pk; };
            const unsigned rmoSlot = (gl::WorkerContext::coreId() >= 0)
                ? static_cast<unsigned>(gl::WorkerContext::coreId())
                : genScratchArenas().slotCount() - 1;
            ScratchArena& rmoArena = genScratchArenas().forSlot(rmoSlot);
            ScratchArena& rmoStrArena = scratchArenas().forSlot(rmoSlot);
            DirtyState rmoDirty = DirtyState::Clean;
            TypedColdSet<int64_t> eraseSet(&rmoArena, &rmoDirty);
            const StrSpan rmoCanonSpan = memoryBlock.nameMap.decodeView(canonId);
            const int32_t rmoMemberN = clss.memberCount();
            StrReplacement* subPairs = (rmoMemberN == 0) ? nullptr
                : reinterpret_cast<StrReplacement*>(rmoArena.resolve(rmoArena.alloc(
                      rmoMemberN * static_cast<int32_t>(sizeof(StrReplacement)),
                      static_cast<int32_t>(alignof(StrReplacement)))));
            int32_t subCount = 0;
            for (int32_t mIdx_ = 0; mIdx_ < clss.memberCount(); ++mIdx_) {
                const NameId mid = clss.memberId(mIdx_);
                const StrSpan mv = memoryBlock.nameMap.decodeView(mid);
                if (!equalSpans(mv, rmoCanonSpan)) {
                    subPairs[subCount].key = mv;
                    subPairs[subCount].value = rmoCanonSpan;
                    ++subCount;
                }
            }
            if (subCount == 0) return;
            // The posting index answers "which keys carry this token"; the
            // substitution replaces a key only as a whole token — the same
            // question exactly when no member name carries a token delimiter.
            for (int32_t si = 0; si < subCount; ++si) {
                const StrSpan k = subPairs[si].key;
                for (int32_t b = 0; b < k.len; ++b) {
                    assert(k.ptr[b] != ']' && k.ptr[b] != ','
                        && "ordis hook: a class member name carries a token "
                           "delimiter");
                }
            }
            ColdHashSet<BytesKeyStore> rmoStrings(&rmoArena, &rmoDirty);
            PagedVector<RmoSortRow> rmoRows(&rmoArena, &rmoDirty);
            PagedVector<int32_t> mailLvlPool(&rmoArena, &rmoDirty);
            PagedVector<int32_t> mailEqPool(&rmoArena, &rmoDirty);
            PagedVector<MailRec> mailRecs(&rmoArena, &rmoDirty);
            struct RetainedPark { int32_t templateId,start,length; };
            PagedVector<RetainedPark> retained(&rmoArena,&rmoDirty);
            PagedVector<char> retainedBytes(&rmoArena,&rmoDirty);


            // Candidates: the postings of every non-canonical member at the
            // class validity (index refreshed to the map's current key set),
            // sorted and deduplicated so a key naming two members is visited
            // once; the substitution test below re-checks every candidate.
            TypedColdSet<int64_t> candidateKeys(&rmoArena, &rmoDirty);
            if (selected != nullptr) {
                for (int32_t i = 1; i <= selected->count(); ++i) candidateKeys.mint(selected->keyAt(i));
            } else {
                buckets.ensureCurrent(rmo, memoryBlock.templateInterner, memoryBlock.nameMap);
                for (int32_t i = 0; i < clss.memberCount(); ++i) {
                    const NameId member = clss.memberId(i);
                    for (int32_t j = buckets.postBegin(classVid, member), end = buckets.postEnd(classVid, member); j < end; ++j)
                        candidateKeys.mint(buckets.pkAt(buckets.postAt(j)));
                }
            }
            for (int32_t i = 1; i <= candidateKeys.count(); ++i) {
                const int64_t key = candidateKeys.keyAt(i);
                const int32_t id = rmo.lookup(key);
                if (id == 0) continue;
                const StrSpan text = memoryBlock.templateInterner.decodeView(Codec<StatementKey>::decode(key).orig);
                ScratchScope probe(rmoStrArena);
                bool changed = !equalSpans(text, StrSpan(replaceKeysScratch(rmoStrArena, text, subPairs, subCount)));
                for (int32_t r = 0; r < rmo.runLen(id) && !changed; ++r) {
                    int32_t length = 0;
                    const char* bytes = rmo.peekRecordBytes(id, r, length, rmoArena);
                    const StrSpan source = memoryBlock.valueInterner.decodeView(RejectedOrdisValueBlobView(bytes, length).orStatementId());
                    changed = !equalSpans(source, StrSpan(replaceKeysScratch(rmoStrArena, source, subPairs, subCount)));
                }
                if (changed) rmoRows.push_back(RmoSortRow{rmoStrings.mint(text), key});
            }
            const int32_t rmoCount = rmoRows.size();
            ScratchScope rmoSortScope(rmoArena);
            const auto rmoLess = [&](int32_t a, int32_t b) {
                return compareSpans(rmoStrings.keyAt(rmoRows[a].tmplId),
                                    rmoStrings.keyAt(rmoRows[b].tmplId)) < 0;
            };
            ChunkSortedOrdinals<decltype(rmoLess)> rmoOrder(
                rmoArena, rmoCount, 0, rmoLess);

            for (int32_t rmoII = 0; rmoII < rmoCount; ++rmoII) {
                ScratchScope rmoStrScope(rmoStrArena);
                const RmoSortRow& rmoCur = rmoRows[rmoOrder.next()];
                const int64_t pk = rmoCur.pk;

                eraseSet.mint(pk);

                const ArenaOffset valMark = rmoArena.cursor();
                const RejectedOrdisRunSnapshot valSnap =
                    snapshotRejectedOrdisRun(rmo, pk, rmoArena);
                for (int32_t vi = 0; vi < valSnap.count; ++vi) {
                    const RejectedOrdisValueBlobView& v = valSnap.views[vi];
                    const StrSpan vStatement =
                        memoryBlock.valueInterner.decodeView(v.orStatementId());
                    const ScratchString stmtPost = replaceKeysScratch(
                        rmoStrArena, vStatement, subPairs, subCount);
                    if (equalSpans(StrSpan(stmtPost), vStatement)) {
                        const StrSpan oldKey=rmoStrings.keyAt(rmoCur.tmplId);
                        const ScratchString currentKey=replaceKeysScratch(rmoStrArena,oldKey,subPairs,subCount);
                        const int32_t start=retainedBytes.size();
                        for(int32_t j=0;j<v.len;++j) retainedBytes.push_back(v.p[j]);
                        retained.push_back(RetainedPark{rmoStrings.mint(StrSpan(currentKey)),start,v.len});
                        continue;
                    }

                    // Level run: stored run UNIONED with the fired pairs'
                    // levels — a fact derived through an equality at level L
                    // is itself at level L
                    // (D-280; the {-1}
                    // non-derived tier is transparent, insertLevelSorted
                    // skips negatives).
                    int lvBuf[256];
                    int32_t lvBufN = 0;
                    for (int32_t j = 0; j < v.levelCount(); ++j)
                        lvBufN = insertLevelSorted(lvBuf, lvBufN,
                                                   v.levelAt(j), 256);

                    const int32_t eqStart = static_cast<int32_t>(mailEqPool.size());
                    for (int32_t si = 0; si < subCount; ++si) {
                        ScratchScope probeScope(rmoStrArena);
                        if (!equalSpans(StrSpan(replaceKeysScratch(rmoStrArena, vStatement,
                                &subPairs[si], 1)), vStatement)) {
                            const StrSpan ek = subPairs[si].key;
                            const StrSpan ev = subPairs[si].value;
                            char eqbuf[256];
                            assert(ek.len + ev.len + 7 <= 256
                                && "equality1 justifier exceeds the eqbuf bound");
                            int32_t en = 0;
                            eqbuf[en++] = '('; eqbuf[en++] = '='; eqbuf[en++] = '[';
                            std::memcpy(eqbuf + en, ek.ptr, static_cast<std::size_t>(ek.len));
                            en += ek.len;
                            eqbuf[en++] = ',';
                            std::memcpy(eqbuf + en, ev.ptr, static_cast<std::size_t>(ev.len));
                            en += ev.len;
                            eqbuf[en++] = ']'; eqbuf[en++] = ')';
                            mailEqPool.push_back(rmoStrings.mint(StrSpan(eqbuf, en)));
                            // The fired pair's stored admission levels join
                            // the deposit run (non-minting probes — class
                            // members are interned).
                            const NameId subFromId =
                                memoryBlock.nameMap.lookup(ek);
                            const NameId subToId =
                                memoryBlock.nameMap.lookup(ev);
                            assert(subFromId != 0 && subToId != 0
                                && "ordis-hook substitution member "
                                   "not interned");
                            clss.forEachLevel(
                                packEqPairKey(subFromId, subToId),
                                [&](int lv) {
                                    lvBufN = insertLevelSorted(
                                        lvBuf, lvBufN, lv, 256);
                                });
                        }
                    }
                    const int32_t eqCount = static_cast<int32_t>(mailEqPool.size()) - eqStart;

                    const int32_t lvlStart = static_cast<int32_t>(mailLvlPool.size());
                    for (int32_t j = 0; j < lvBufN; ++j)
                        mailLvlPool.push_back(lvBuf[j]);
                    const int32_t lvlCount = lvBufN;

                    mailRecs.push_back(MailRec{
                        rmoStrings.mint(StrSpan(stmtPost)),
                        rmoStrings.mint(vStatement),
                        lvlStart, lvlCount, eqStart, eqCount });
                }
                rmoArena.popTo(valMark);
            }

            rmo.eraseBlobIf([&eraseSet](int64_t k) { return eraseSet.contains(k); });
            for(int32_t i=0;i<retained.size();++i) {
                ScratchScope frame(rmoArena);
                const auto& row=retained[i];
                char* bytes=rmoArena.resolve(rmoArena.alloc(row.length,1));
                for(int32_t j=0;j<row.length;++j) bytes[j]=retainedBytes[row.start+j];
                const int64_t key=mintTemplateKey(memoryBlock.templateInterner,memoryBlock.nameMap,
                    rmoStrings.keyAt(row.templateId),validityName);
                insertRejectedOrdisBlobSorted(rmo,key,bytes,row.length,memoryBlock.valueInterner,rmoArena);
                auto& uses=&rmo==&memoryBlock.overallHashMemory.rejectedMapOrdis
                    ? memoryBlock.lbMemory.canonicalUses.ordisSources : memoryBlock.lbMemory.canonicalUses.ordis2Sources;
                uses.changed(key);
            }


            // The deposit scope of every mailed statement is the class
            // validity itself (entry validity == class validity for a hit).
            // Each reset clears its registries in record order; the
            // statement-list and staging rows of all resets leave in one
            // sweep after the run, which neither reads nor writes those lists.
            const int32_t mailRecN = static_cast<int32_t>(mailRecs.size());
            int64_t* resetKeys = mailRecN == 0 ? nullptr
                : reinterpret_cast<int64_t*>(rmoArena.resolve(rmoArena.alloc(
                    mailRecN * static_cast<int32_t>(sizeof(int64_t)),
                    static_cast<int32_t>(alignof(int64_t)))));
            int32_t resetN = 0;
            for (int32_t mi = 0; mi < mailRecN; ++mi) {
                const MailRec& m = mailRecs[static_cast<std::size_t>(mi)];
                const StrSpan postSpan = rmoStrings.keyAt(m.postId);
                int lvRun[256];
                assert(m.lvlCount <= 256
                    && "ordis-hook level run exceeds lvRun");
                for (int32_t j = 0; j < m.lvlCount; ++j)
                    lvRun[j] = mailLvlPool[static_cast<std::size_t>(m.lvlStart + j)];

                if (!isLocalEncodedStatement(memoryBlock.intLocalEncodedStatementsSet,
                        memoryBlock.nameMap, postSpan, validityName)) {
                    resetKeys[resetN++] = resetResentRegistryKeys(memoryBlock, postSpan, validityName);
                }
                // The mailed rewritten statement re-runs the full or
                // consumption at the deposit scope: drop the parked (pre)
                // statement's processed-or row first, or the door's
                // or-uniqueness gate would suppress the canonical re-file.
                // Ledger rows hold NameMap ids of DEPOSITED ors — a
                // never-interned statement has no row by construction, the
                // defined nothing-recorded state (the lookup is non-minting).
                {
                    const NameId preOrigId = memoryBlock.nameMap.lookup(
                        rmoStrings.keyAt(m.preId));
                    if (preOrigId != 0) {
                        caseLedger.clearProcessedOr(memoryBlock, preOrigId, classVid);
                    }
                }
                insertInternalStatement(memoryBlock.sameIterationInternalMail,
                    memoryBlock.nameMap,
                    postSpan, validityName,
                    lvRun, m.lvlCount);

                if (parameters.trackHistory) {
                    OriginDep eqDeps[64];
                    int eqDepN = 0;
                    eqDeps[eqDepN++] = { rmoStrings.keyAt(m.preId), validityName };
                    for (int32_t j = 0; j < m.eqCount; ++j) {
                        assert(eqDepN < 64 && "equality1 origin chain exceeds 64");
                        const StrSpan equality=rmoStrings.keyAt(mailEqPool[static_cast<std::size_t>(m.eqStart+j)]);
                        const StrSpan originalScope=proofHistory.findEqualityCiteScope(memoryBlock,equality,validityName);
                        const ScratchString citeScope=ScratchString::copyFrom(rmoStrArena,originalScope.ptr,originalScope.len);
                        eqDeps[eqDepN++] = {equality,StrSpan(citeScope)};
                    }
                    addInternalMailOrigin(memoryBlock.sameIterationInternalMail,
                              memoryBlock.originInterner,
                              postSpan, validityName,
                              OriginTag::equality1, eqDeps, eqDepN,
                              (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
                }
            }
            eraseResentRows(memoryBlock, resetKeys, resetN);
        }

/// @brief Ordis equi-class hook on `rejectedMapOrdis` — I-37
        ///        verbatim for the parked-cohort map.
        ///
        /// @details
        /// The `rejectedMapOrdis` door of
        /// @ref applyEquivalenceClassToRejectedOrdisCore: every same-scope
        /// park whose key template changes under the class is dropped and
        /// its rewritten or statement mailed onto
        /// `sameIterationInternalMail` with an `equality1` history line, the
        /// candidates read from the per-apply posting index.
        ///
        /// @tparam ClassT The equivalence-class read interface
        ///                (`memberCount` / `memberId` / `forEachLevel`).
        /// @param clss         The (delta) class driving the rewrite.
        /// @param memoryBlock  The owning LB.
        /// @param validityName The class's scope; only same-scope entries
        ///                     rewrite.
        /// @param buckets      The per-apply posting index over
        ///                     `rejectedMapOrdis`.
        /// @return (void)
        /// @invariant Never writes `rejectedMapOrdis` directly (I-37);
        ///            insertions happen only through the re-consumption of
        ///            the mailed statement.
        /// @see applyEquivalenceClassToRejectedOrdisCore — the body;
        ///      applyEquiClasses — both registration blocks.
template <class ClassT>
        inline void EquivalenceAdmissionRewriter::applyEquivalenceClassToRejectedMapOrdis(
            const ClassT& clss,
            Memory& memoryBlock,
            StrSpan validityName,
            RejectedValidityBuckets& buckets,
            const TypedColdSet<int64_t>* selected) {
            this->applyEquivalenceClassToRejectedOrdisCore(
                clss, memoryBlock, validityName,
                memoryBlock.overallHashMemory.rejectedMapOrdis, buckets, selected);
        }

/// @brief Ordis2 equi-class hook on `rejectedMapOrdis2` — I-37
        ///        verbatim for the second park index
        ///        (D-267).
        ///
        /// @details
        /// The `rejectedMapOrdis2` door of
        /// @ref applyEquivalenceClassToRejectedOrdisCore, pointed at the
        /// ground-text park index: every same-scope park whose key text
        /// changes under the class is dropped and its rewritten or statement
        /// mailed onto `sameIterationInternalMail` with an `equality1`
        /// history line. An eligible-disjunct key is a sub-text of the or
        /// statement itself, so a rewrite that fires on the key fires on the
        /// statement too. Both parked-or hooks and the demand hook run at the
        /// same applyEquiClasses seams, so the pair's ground-to-ground
        /// matching never desynchronizes.
        ///
        /// @tparam ClassT The equivalence-class read interface
        ///                (`memberCount` / `memberId` / `forEachLevel`).
        /// @param clss         The (delta) class driving the rewrite.
        /// @param memoryBlock  The owning LB.
        /// @param validityName The class's scope; only same-scope entries
        ///                     rewrite.
        /// @param buckets      The per-apply posting index over
        ///                     `rejectedMapOrdis2`.
        /// @return (void)
        /// @invariant Never writes `rejectedMapOrdis2` directly (I-37);
        ///            insertions happen only through the re-consumption of
        ///            the mailed statement.
        /// @see applyEquivalenceClassToRejectedOrdisCore — the body;
        ///      applyEquiClasses — both registration blocks.
template <class ClassT>
        inline void EquivalenceAdmissionRewriter::applyEquivalenceClassToRejectedMapOrdis2(
            const ClassT& clss,
            Memory& memoryBlock,
            StrSpan validityName,
            RejectedValidityBuckets& buckets,
            const TypedColdSet<int64_t>* selected) {
            this->applyEquivalenceClassToRejectedOrdisCore(
                clss, memoryBlock, validityName,
                memoryBlock.overallHashMemory.rejectedMapOrdis2, buckets, selected);
        }

/// @brief Equi-class hook for `admissionMapOrdis2` — drop-and-rekey
        ///        each demand key whose ground template the class rewrites
        ///        (D-267).
        ///
        /// @details
        /// Sibling of `applyEquivalenceClassToRejectedMapOrdis`, with the
        /// admission-path drop-and-rekey discipline (the D-106
        /// form) instead of drop-and-mail: a demand entry
        /// is not a statement — there is nothing to re-deposit — so the
        /// stale key's value blobs are copied verbatim, the key erased,
        /// and the blobs re-inserted under the rewritten (canonicalized)
        /// template at the same validity. Both sides of the
        /// ground-to-ground route-(c) match rewrite at this same seam (the
        /// parked-or hooks run beside it), so matching never
        /// desynchronizes. The walk is a decoded-lex-sorted key snapshot
        /// (I-84); mutation runs strictly after the walk. Blob-level: no
        /// record decode — `{sourceImplId, levels}` payloads transfer
        /// byte-verbatim.
        ///
        /// @tparam ClassT Equivalence-class view type (member-id surface).
        /// @param clss         The delta class driving the rewrite.
        /// @param memoryBlock  Owning LB.
        /// @param validityName The class scope; only demand keys at exactly
        ///        this validity rewrite (the ordis hook's same filter).
        /// @return (void)
        /// @invariant Single-threaded seam only (I-83); the map's canonical
        ///            run order is preserved by the RMW re-insert (I-99).
        /// @see `applyEquivalenceClassToRejectedMapOrdis` — the walk
        ///      template; `consumeOrLeavesCohort` route (c) — the reader.
template <typename ClassT>
        inline void EquivalenceAdmissionRewriter::applyEquivalenceClassToAdmissionMapOrdis2(
            const ClassT& clss,
            Memory& memoryBlock,
            StrSpan validityName,
            const TypedColdSet<int64_t>* selected) {

            const NameId phase3Scope = memoryBlock.nameMap.lookup(validityName);
            if (phase3Scope != 0 && memoryBlock.phase3IgnoresScope(phase3Scope))
                return;

            auto& cdm = memoryBlock.overallHashMemory.admissionMapOrdis2;
            if (cdm.empty()) return;

            const NameId canonId = canonicalizer.chooseCanonicalId(clss, memoryBlock, validityName);
            if (canonId == 0) return;

            const unsigned cdSlot = (gl::WorkerContext::coreId() >= 0)
                ? static_cast<unsigned>(gl::WorkerContext::coreId())
                : genScratchArenas().slotCount() - 1;
            ScratchArena& cdArena = genScratchArenas().forSlot(cdSlot);
            const ArenaOffset cdMark = cdArena.cursor();
            DirtyState cdDirty = DirtyState::Clean;

            const StrSpan cdCanonSpan = memoryBlock.nameMap.decodeView(canonId);
            const int32_t cdMemberN = clss.memberCount();
            StrReplacement* subPairs = (cdMemberN == 0) ? nullptr
                : reinterpret_cast<StrReplacement*>(cdArena.resolve(cdArena.alloc(
                      cdMemberN * static_cast<int32_t>(sizeof(StrReplacement)),
                      static_cast<int32_t>(alignof(StrReplacement)))));
            int32_t subCount = 0;
            for (int32_t mIdx = 0; mIdx < clss.memberCount(); ++mIdx) {
                const NameId mid = clss.memberId(mIdx);
                const StrSpan mv = memoryBlock.nameMap.decodeView(mid);
                if (!equalSpans(mv, cdCanonSpan)) {
                    subPairs[subCount].key = mv;
                    subPairs[subCount].value = cdCanonSpan;
                    ++subCount;
                }
            }
            if (subCount == 0) { cdArena.popTo(cdMark); return; }

            // Decoded-lex key snapshot (interned into a throwaway arena
            // interner so the spans survive later mints), walked sorted;
            // matches staged, mutation after the walk.
            struct CdRow { int32_t tmplId; int32_t vldId; int64_t pk; };
            struct CdReKey { int64_t oldPk; int32_t newTmplId;
                             int32_t blobStart; int32_t blobCount; };
            struct CdBlob { ArenaOffset off; int32_t len; };
            ColdHashSet<BytesKeyStore> cdStrings(&cdArena, &cdDirty);
            PagedVector<CdRow> cdRows(&cdArena, &cdDirty);
            PagedVector<CdReKey> cdReKeys(&cdArena, &cdDirty);
            PagedVector<CdBlob> cdBlobs(&cdArena, &cdDirty);
            const int32_t cdStoredCount = selected == nullptr ? cdm.count() : selected->count();
            for (int32_t id = 1; id <= cdStoredCount; ++id) {
                const int64_t pkRow = selected == nullptr ? cdm.keyAt(id) : selected->keyAt(id);
                if (cdm.lookup(pkRow) == 0) continue;
                if (memoryBlock.phase3IgnoresScope(Codec<StatementKey>::decode(pkRow).validity)) continue;
                StrSpan tKey, tVal;
                decodeTemplateKeyView(pkRow, memoryBlock.templateInterner,
                    memoryBlock.nameMap, tKey, tVal);
                cdRows.push_back(CdRow{ cdStrings.mint(tKey),
                    cdStrings.mint(tVal), pkRow });
            }
            const int32_t cdCount = cdRows.size();
            {
                // NO ScratchScope here: the staged blob copies below ride
                // the byte-bump tier and must survive into the mutation
                // phase — everything is reclaimed by the function-end
                // popTo(cdMark).
                const auto cdLess = [&](int32_t a, int32_t b) {
                    const CdRow& ra = cdRows[a];
                    const CdRow& rb = cdRows[b];
                    const int ct = compareSpans(cdStrings.keyAt(ra.tmplId),
                                                cdStrings.keyAt(rb.tmplId));
                    if (ct != 0) return ct < 0;
                    return compareSpans(cdStrings.keyAt(ra.vldId),
                                        cdStrings.keyAt(rb.vldId)) < 0;
                };
                ChunkSortedOrdinals<decltype(cdLess)> cdOrder(
                    cdArena, cdCount, 0, cdLess);

                const unsigned cdStrSlot = (gl::WorkerContext::coreId() >= 0)
                    ? static_cast<unsigned>(gl::WorkerContext::coreId())
                    : scratchArenas().slotCount() - 1;
                ScratchArena& cdStrArena = scratchArenas().forSlot(cdStrSlot);
                for (int32_t ii = 0; ii < cdCount; ++ii) {
                    ScratchScope cdStrScope(cdStrArena);
                    const CdRow& cur = cdRows[cdOrder.next()];
                    if (!equalSpans(cdStrings.keyAt(cur.vldId), validityName))
                        continue;
                    const StrSpan tmplSpan = cdStrings.keyAt(cur.tmplId);
                    const ScratchString rewritten = replaceKeysScratch(
                        cdStrArena, tmplSpan, subPairs, subCount);
                    if (equalSpans(StrSpan(rewritten), tmplSpan)) continue;

                    const int32_t blobStart = cdBlobs.size();
                    const int32_t rid = cdm.lookup(cur.pk);
                    assert(rid != 0
                        && "ordis2-demand rekey: snapshot key vanished mid-walk");
                    const int32_t runN = cdm.runLen(rid);
                    for (int32_t j = 0; j < runN; ++j) {
                        int32_t bl = 0;
                        const char* bp =
                            cdm.peekRecordBytes(rid, j, bl, cdArena);
                        const ArenaOffset coff = cdArena.alloc(bl, 1);
                        std::memcpy(cdArena.resolve(coff), bp,
                                    static_cast<std::size_t>(bl));
                        cdBlobs.push_back(CdBlob{ coff, bl });
                    }
                    cdReKeys.push_back(CdReKey{ cur.pk,
                        cdStrings.mint(StrSpan(rewritten)),
                        blobStart, runN });
                }
            }

            // Mutation strictly after the walk, in the staged (decoded-lex)
            // order: erase the stale key, re-insert its blobs verbatim
            // under the canonical template (the mint order is a pure
            // function of the sorted key content — I-84).
            const int32_t rkN = cdReKeys.size();
            for (int32_t r = 0; r < rkN; ++r) {
                const CdReKey& rk = cdReKeys[r];
                cdm.eraseBlobIf([&rk](int64_t k) { return k == rk.oldPk; });
                const int64_t newPk = mintTemplateKey(
                    memoryBlock.templateInterner, memoryBlock.nameMap,
                    cdStrings.keyAt(rk.newTmplId), validityName);
                for (int32_t j = 0; j < rk.blobCount; ++j) {
                    const CdBlob& b = cdBlobs[rk.blobStart + j];
                    insertAdmissionOrdis2BlobSorted(cdm, newPk,
                        reinterpret_cast<const char*>(cdArena.resolve(b.off)),
                        b.len, memoryBlock.ruleInterner, cdArena);
                }
            }
            cdArena.popTo(cdMark);
        }
}

namespace gl {
/// @brief Bind inputs for equivalence admission rewriter.
/// @details Construction stores explicit references; block storage and all invocation lifetimes remain with their owners.
/// @param canonicalTemplates Borrowed CanonicalTemplateMatcher interface or state; must outlive this component.
/// @param canonicalizer Borrowed Canonicalizer interface or state; must outlive this component.
/// @param parameters Borrowed ProverParameters interface or state; must outlive this component.
/// @param rejectedSources Borrowed RejectedSourceMemory interface or state; must outlive this component.
/// @param caseLedger Borrowed CaseLedger interface or state; must outlive this component.
/// @param proofHistory Borrowed ProofHistory interface or state; must outlive this component.
/// @param rejectedSourceReplay Borrowed RejectedSourceReplay interface or state; must outlive this component.
/// @param watchProcessor Borrowed WatchProcessor interface or state; must outlive this component.
/// @return None; construction binds the supplied references.
inline EquivalenceAdmissionRewriter::EquivalenceAdmissionRewriter(CanonicalTemplateMatcher& canonicalTemplates, Canonicalizer& canonicalizer, const ProverParameters& parameters, RejectedSourceMemory& rejectedSources, CaseLedger& caseLedger, ProofHistory& proofHistory, RejectedSourceReplay& rejectedSourceReplay, WatchProcessor& watchProcessor) : canonicalTemplates(canonicalTemplates), canonicalizer(canonicalizer), parameters(parameters), rejectedSources(rejectedSources), caseLedger(caseLedger), proofHistory(proofHistory), rejectedSourceReplay(rejectedSourceReplay), watchProcessor(watchProcessor) {}
} // namespace gl
