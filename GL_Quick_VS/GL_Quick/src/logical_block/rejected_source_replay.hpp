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
#include "../logical_block/demand_built_rules.hpp"
#include "../logical_block/scope_syntax.hpp"
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
class ProofHistory;
class WatchProcessor;
namespace tests { struct RejectedSourceReplayFixture; }
/// @brief Rejected Source Replay operations for the proof algorithm.
/// @details Uses explicit peer interfaces while the supplied logical blocks retain storage and proof state.
class RejectedSourceReplay {
friend struct tests::RejectedSourceReplayFixture;
public:
/// @brief Bind inputs for rejected source replay.
/// @details Construction stores explicit references; block storage and all invocation lifetimes remain with their owners.
/// @param admissionPolicy Borrowed AdmissionPolicy interface or state; must outlive this component.
/// @param anchorInfo Borrowed anchor description of the batch; must outlive this component.
/// @param canonicalTemplates Borrowed CanonicalTemplateMatcher interface or state; must outlive this component.
/// @param canonicalizer Borrowed Canonicalizer interface or state; must outlive this component.
/// @param definitions Borrowed CompiledDefinitionRegistry interface or state; must outlive this component.
/// @param parameters Borrowed ProverParameters interface or state; must outlive this component.
/// @param rejectedSources Borrowed RejectedSourceMemory interface or state; must outlive this component.
/// @param admissionMemory Borrowed AdmissionMemory interface or state; must outlive this component.
/// @param caseLedger Borrowed CaseLedger interface or state; must outlive this component.
/// @param proofHistory Borrowed ProofHistory interface or state; must outlive this component.
/// @param watchProcessor Borrowed WatchProcessor interface or state; must outlive this component.
/// @return None; construction binds the supplied references.
RejectedSourceReplay(AdmissionPolicy& admissionPolicy, const ce::AnchorInfo& anchorInfo, CanonicalTemplateMatcher& canonicalTemplates, Canonicalizer& canonicalizer, const CompiledDefinitionRegistry& definitions, const ProverParameters& parameters, RejectedSourceMemory& rejectedSources, AdmissionMemory& admissionMemory, CaseLedger& caseLedger, ProofHistory& proofHistory, WatchProcessor& watchProcessor);

public:
/// @brief Wake newly parked or rewritten sources against existing demand.
        /// @details A derived index maps prefix-stripped demand text to packed keys.
        /// Only matching keys enter the original decoded expression/scope sort;
        /// every wake retains the ordinary admission probe. An integration park
        /// is also woken at each double-marker key scope whose route-5 probe
        /// admits it (D-492), in decoded
        /// (template, scope) order.
        /// @param memoryBlock Owning memory block.
        /// @param algebraParks Concrete algebra keys to revisit.
        /// @param integrationParks Concrete integration keys to revisit.
        /// @param defer Whether an active canonical maintenance pass queues the parks.
        /// @return Void.
inline void wakeParkedExistencesAtExistingDemands(Memory& memoryBlock,
            const ColdHashSet<BytesKeyStore>& algebraParks,
            const ColdHashSet<BytesKeyStore>& integrationParks, bool defer = true);

private:
/// @brief Coalesce a wake while canonical source and demand identities change.
        /// @details Outside maintenance the caller continues its ordinary immediate
        /// wake. During maintenance only logical template and scope keys are stored.
        /// @param mb Owning block.
        /// @param text Bare marked or ground wake template.
        /// @param validity Existing target scope.
        /// @param kind Algebra, integration, OR or OR2 bit (1, 2, 4 or 8).
        /// @return True when queued, false when immediate wake processing applies.
inline bool queueCanonicalWake(Memory& mb, StrSpan text, StrSpan validity, int32_t kind);

public:
/// @brief Release coalesced wakes after the complete canonical maintenance pass.
        /// @details Source-last matches are collected against settled demand keys.
        /// Wakes run in decoded template and scope order with original kind priority;
        /// the queue is detached first so ordinary arrivals can wake immediately.
        /// @param mb Owning block with an active maintenance pass.
        /// @return Nothing.
        /// @invariant Obsolete rule removal and all source/demand re-keying precede this call.
inline void drainCanonicalWakes(Memory& mb);

private:
/// @brief Reject parked-source searches with no possible admission route.
        ///
        /// @details This is a conservative negative filter. Algebra reads local
        /// and ancestor demands at the target scope and preserves standalone
        /// single-input operator admission. Integration reads the prefixed
        /// instruction keys visible at the scope — owned there or at a strict
        /// ancestor (`AdmissionMemory::integrationDemandVisible`,
        /// D-572) — and the exact-scope ordinary
        /// admission set, then the double-marker keys at the scope and its
        /// ancestors (route 5). A positive answer leaves every source-specific
        /// budget and replay check intact. Deferred canonical wakes are queued
        /// before this filter is called.
        /// @param mb Owning block and current demand indexes.
        /// @param key Caller-owned concrete marked key, stable across cache updates.
        /// @param validity Caller-owned interned target scope spelling.
        /// @param integration Select the integration rather than algebra route.
        /// @return False only when the existing replay predicate cannot admit a source.
        /// @invariant Ancestor demand is visible through the ordinary algebra
        ///            route and the double-marker route; the integration
        ///            instruction route reads, beyond its own scope, the
        ///            serving owner of an OR case
        ///            (`AdmissionMemory::ancestorDemandOwner`); the integration
        ///            admission set stays exact-scope.
        /// @see queueParkedExistence, revisitRejected2, revisitRejectedIntegration2.
inline bool hasPotentialParkedDemand(Memory& mb, StrSpan key,
                StrSpan validity, bool integration);

private:
/// @brief Queue one demanded existence for ordinary internal-mail absorption.
        /// @details Admission is probed without effects. Completed source/scope
        /// constructions and already queued constructions are deduplicated. The
        /// ancestor record stays available to other demanding scopes.
        /// @param mb Owning memory block.
        /// @param key Concrete marked demand key.
        /// @param source Parked source record.
        /// @param parkValidity Source scope.
        /// @param targetValidity Demanding scope.
        /// @param integration Whether this is an integration demand index.
        /// @return Whether new replay work was queued.
inline bool queueParkedExistence(Memory& mb, StrSpan key,
            const RejectedValueBlobView& source, StrSpan parkValidity,
            StrSpan targetValidity, bool integration);

public:
/// @brief Queue one existence source for ordinary internal-mail replay at
        ///        a scope — the shared tail of every source replay.
        ///
        /// @details
        /// Marks the construction kinds @p available as pending in the
        /// source's receipt at @p targetValidity (bits `available << 2`, the
        /// relay bit 16 following @p flags bit 8), deposits the source on the
        /// replay mail (`Memory::replayMailDuringAbsorb` while an absorb
        /// runs, `sameIterationInternalMail` otherwise) with its level run and
        /// a disintegration signal carrying the OR permission (@p flags bit 4),
        /// @p iteration and @p replayDepth, and marks the burst mutated. With
        /// history on, the deposit cites the source at @p parkValidity: a
        /// `zoom in` origin when the scopes differ, otherwise the source's own
        /// origin records copied onto the mail.
        ///
        /// @param mb             Owning block.
        /// @param existence      Source existence text (stable across mints
        ///                       into the NameMap, origin interner and mail).
        /// @param parkValidity   Scope where the source is known.
        /// @param targetValidity Scope that replays the source (the park scope
        ///                       or one below it).
        /// @param available      Construction kinds to replay (bit 1 algebra,
        ///                       bit 2 integration), nonzero.
        /// @param flags          Park-layout flags (4 OR disintegration
        ///                       allowed, 8 inherited relay source).
        /// @param iteration      Witness generation of the replayed source.
        /// @param levels         Ascending level run of the source.
        /// @param levelCount     Length of @p levels.
        /// @param replayDepth    Integration depth the replay assigns.
        /// @return Void.
        /// @invariant Mints the source and target into the NameMap only when new.
        /// @see queueParkedExistence, replayTrialSource.
inline void queueSourceReplay(Memory& mb, StrSpan existence, StrSpan parkValidity,
            StrSpan targetValidity, int32_t available, int32_t flags, int32_t iteration,
            const int* levels, int32_t levelCount, int32_t replayDepth);

public:
/// @brief Replay a promoted integration trial's source at its parent scope
        ///        (D-481).
        ///
        /// @details
        /// The parent is the trial scope's NameMap parent. The source's level
        /// run is read at the nearest scope at or above the parent that
        /// stores it — the scope where the source is known, which the replay
        /// cites. When the source's integration construction at the parent is
        /// neither completed nor already pending (`existenceStateAt`), the
        /// construction is queued through `queueSourceReplay` with the trial's
        /// flags, generation and integration depth; the replayed source then
        /// meets the passed trial record, which admits the recorded witnesses
        /// under their recorded names, and the ordinary routes commit. A
        /// completed or pending construction needs no new replay.
        ///
        /// @param mb    Owning block.
        /// @param trial The passed trial record.
        /// @return True iff a replay was queued.
        /// @invariant Asserts that the source is known with levels at or above
        ///            the parent.
        /// @see ImplicationSubproofs::drainIntegrationTrials, queueSourceReplay.
inline bool replayTrialSource(Memory& mb, const IntegrationTrialRec& trial);

private:
// Push a set of constituents (primary + siblings) onto this LB's
        // sameIterationInternalMail for revival at the next hashburst. Used by both the
        // equi-class-driven rewrite path and the new-admission-key revisit
        // path. The compound is NOT re-emitted — it is already in LB memory
        // via normal applyEquivalenceClass deposit (or was already present at
        // the rejection time). Each emitted statement carries its original
        // validity (may be non-main).
        //
        // Origin uses the `equality1` tag (eq-class substitution), so the
        // origin.second[0] slot must be the PRE-rewrite form of each emitted
        // statement — not the compound — to match the verifier's equality1
        // checker. For the direct-revisit path (no rewrite), pre == post.
inline void emitIntegrationRevivalToInternalMailIn(
            const std::string& primaryConstituentPre,
            const std::string& primaryConstituentPost,
            const std::vector<std::string>& siblingsPre,
            const std::vector<std::string>& siblingsPost,
            const std::set<int>& levels,
            const std::string& validityName,
            const std::set<std::string>& equalitiesForOrigin,
            Memory& mb);

private:
/// @brief Span/run door of @ref emitIntegrationRevivalToInternalMailIn —
        ///        revive a rejected cohort onto `sameIterationInternalMail` from
        ///        spans plus an ascending level run, no transient
        ///        `std::string` / `std::set` materialized.
        ///
        /// @details
        /// Exact twin of the string overload: the same `emitOne(pre, post)`
        /// core — the levels-RUN `insertInternalStatement` door followed
        /// (under `trackHistory`) by the span-antecedent
        /// `addInternalMailOrigin` with a bounded stack `OriginDep[64]` filled
        /// `pre` first then the equalities in array order, the same
        /// `maxOrigins` expression — applied to the primary and then the
        /// paired siblings `i in [0, min(siblingsPreN, siblingsPostN))`
        /// (mirroring the string form's `std::min` pairing).
        ///
        /// Per emitted statement this door performs the SAME interner touches
        /// in the SAME order as the string form: `nm.encode(post)`,
        /// `nm.encode(validityName)` (statement door), then under
        /// `trackHistory` `mintOriginKey(oi, post, validityName)` followed by
        /// the positional dep mints inside `encodeOriginSpans`. The string
        /// form already routes through these span doors internally; only its
        /// parameters were heap, so the deposits are byte-identical.
        ///
        /// @pre `[levels, levels + levelCount)` is sorted ascending strictly
        ///      unique — `ColdMail::insertStatement`'s run-door key-byte
        ///      contract; `levelCount == 0` equals the empty set.
        /// @pre `[equalitiesForOrigin, equalitiesForOrigin + equalityN)` is
        ///      sorted ascending unique under `compareSpans` — the string form
        ///      iterates its `std::set<std::string>` ascending.
        /// @param primaryPre          Pre-rewrite form of the primary
        ///                            constituent (origin antecedent slot 0).
        /// @param primaryPost         Post form of the primary constituent
        ///                            (the deposited statement).
        /// @param siblingsPre         Pre-rewrite sibling spans, positional.
        /// @param siblingsPreN        Count of `siblingsPre`.
        /// @param siblingsPost        Post sibling spans, positional.
        /// @param siblingsPostN       Count of `siblingsPost`.
        /// @param levels              Pointer to `levelCount` ascending-unique
        ///                            levels; may be null only when
        ///                            `levelCount` is 0.
        /// @param levelCount          The level count (`>= 0`).
        /// @param validityName        The emitted statements' validity scope.
        /// @param equalitiesForOrigin Extra `equality1` origin antecedents;
        ///                            may be null only when `equalityN` is 0.
        /// @param equalityN           Count of `equalitiesForOrigin`.
        /// @param mb                  The owning LB.
        /// @invariant Span lifetime: the door mints `nameMap` +
        ///            `originInterner`, so caller spans must alias OTHER
        ///            buffers — `valueInterner` `decodeView`s and gen-arena
        ///            blob copies qualify (I-3 different-interner rule).
        /// @see emitIntegrationRevivalToInternalMailIn(const std::string&, const std::string&, const std::vector<std::string>&, const std::vector<std::string>&, const std::set<int>&, const std::string&, const std::set<std::string>&, Memory&)
        ///      — the byte-oracle string overload; insertInternalStatement —
        ///      the levels-run statement door; addInternalMailOrigin — the
        ///      span-antecedent origin door.
inline void emitIntegrationRevivalToInternalMailIn(
            StrSpan primaryPre, StrSpan primaryPost,
            const StrSpan* siblingsPre, int32_t siblingsPreN,
            const StrSpan* siblingsPost, int32_t siblingsPostN,
            const int* levels, int32_t levelCount,
            StrSpan validityName,
            const StrSpan* equalitiesForOrigin, int32_t equalityN,
            Memory& mb);

public:
/// @brief Wake the `rejectedMap` cohorts an admission key gain can
        ///        satisfy — the exact-scope cohort (erased, revived at the
        ///        key's scope) and every cohort parked at a strict ANCESTOR
        ///        scope (kept, released into the key's scope).
        ///
        /// @details
        /// Two paths on one key gain, both gated by the same two revival
        /// budgets (secondary-variable count and iteration depth of
        /// `markedExpr`):
        /// 1. **Exact scope.** The cohort parked under
        ///    `(markedExpr, validityName)` is snapshotted, erased, and every
        ///    record mailed on `sameIterationInternalMail` at `validityName`
        ///    through the `equality1` self-source revival door — unchanged.
        /// 2. **Ancestor scopes.** @ref releaseAncestorRejected2 — a park at
        ///    a strict ancestor of `validityName` holds a product that is
        ///    valid at `validityName` too; the demand lives only here, so the
        ///    product is released INTO `validityName` and the ancestor's park
        ///    is left in place for the next scope that asks
        ///    (`D-341`).
        ///
        /// @param markedExpr   The admission template that gained a key
        ///                     (bare marker form).
        /// @param memoryBlock  The owning LB.
        /// @param validityName The key's validity scope.
        /// @return (void)
        /// @invariant Probes never mint (I-89); the `revisitInProgress` guard
        ///            covers the exact path only — the ancestor path is
        ///            mail-only and cannot re-enter.
        /// @see releaseAncestorRejected2, revisitRejectedIntegration2 — the
        ///      integration twin; drainAdmissionKeysAlgebra,
        ///      updateAdmissionMapRecursion, applyEquivalenceClassToAdmissionMap
        ///      and the D-243 park-side probe — the seams.
void revisitRejected2(StrSpan markedExpr,
            Memory& memoryBlock,
            StrSpan validityName);

private:
/// @brief Record an unchanged expression inherited into a descendant scope.
///
/// @details The zoom in row has exactly one dependency: the same expression
/// at its source scope. Parent history is neither read nor copied. The normal
/// history writer retains its duplicate and capacity policy; this operation
/// returns no decision to the reasoning algorithm.
///
/// @param mb Owning logic block and internal-mail history sink.
/// @param expression Released expression, unchanged by the transfer.
/// @param sourceValidity Exact source scope of the ancestor park.
/// @param targetValidity Exact descendant scope receiving the expression.
/// @return (void)
/// @invariant Only history is written; source derivations and release state are untouched.
/// @see releaseParkedProductIntoScope, addInternalMailOrigin.
void recordZoomInOrigin(Memory& mb, StrSpan expression,
            StrSpan sourceValidity, StrSpan targetValidity);

private:
/// @brief Mail one parked product (with its sibling cohort) into a
        ///        descendant scope, carrying its park-scope provenance.
        ///
        /// @details
        /// The per-record door of the ancestor-park release, shared by the
        /// algebra and integration twins. Known-skip first: a product already
        /// known at `targetValidity` or at one of its ancestors
        /// (`ancestorKnown`, self included) is not released — the park stays
        /// for a later scope, and re-mailing would only feed the Site F
        /// refusal. Otherwise the product and every sibling are deposited on
        /// `sameIterationInternalMail` at `targetValidity` with the caller's
        /// level run. Optional history maintenance records one zoom in row
        /// per expression at the target scope, citing that same expression at
        /// parkValidity. Parent history is neither read nor copied, and never
        /// decides whether the product or siblings are released.
        ///
        /// @param mb             The owning LB.
        /// @param product        The parked concrete constituent.
        /// @param siblings       Its sibling cohort (positional).
        /// @param siblingCount   Sibling count.
        /// @param levels         Ascending-unique level run for every deposit;
        ///                       may be null only when `levelCount` is 0.
        /// @param levelCount     Level count.
        /// @param parkValidity   The scope the cohort is parked at (the
        ///                       provenance source).
        /// @param targetValidity The demand's scope (the deposit scope); must
        ///                       be interned.
        /// @param gArena         Caller scratch arena; unchanged by history recording.
        /// @return True iff the cohort was mailed; false on the known-skip.
        /// @invariant Parent parks are retained; release has no history prerequisite.
        /// @see releaseAncestorRejected2, releaseAncestorRejectedIntegration2
        ///      — the callers; recordZoomInOrigin;
        ///      ancestorKnown.
bool releaseParkedProductIntoScope(Memory& mb,
            StrSpan product,
            const StrSpan* siblings, int32_t siblingCount,
            const int* levels, int32_t levelCount,
            StrSpan parkValidity,
            StrSpan targetValidity,
            ScratchArena& gArena);

private:
/// @brief Release into `validityName` every `rejectedMap` cohort
        ///        parked under `markedExpr` at a strict ancestor scope of
        ///        `validityName`; the ancestor parks stay.
        ///
        /// @details
        /// Walks the owned ancestor copies from @ref copyStrictAncestorSpans
        /// (root-first, deterministic), probes `rejectedMap` at each
        /// `(markedExpr, ancestor)` key without minting, snapshots a hit's
        /// run onto gen-scratch and hands every record to
        /// @ref releaseParkedProductIntoScope after the exact-scope revival's
        /// per-record witness-depth gate (`maxIterationNumberVariable` on the
        /// marker-slot witness). Nothing is erased: the park at the ancestor
        /// remains the supply for any other descendant scope — or the
        /// ancestor itself — that later demands the same template. The
        /// caller (@ref revisitRejected2) has already applied the key-level
        /// budget gates.
        ///
        /// @param markedExpr   The admission template that gained a key
        ///                     (bare marker form).
        /// @param memoryBlock  The owning LB.
        /// @param validityName The key's scope — the release target.
        /// @return (void)
        /// @invariant Never writes `rejectedMap` (I-37's writer set is
        ///            unchanged); probes never mint (I-89).
        /// @see revisitRejected2 — the caller; releaseParkedProductIntoScope —
        ///      the per-record door; copyStrictAncestorSpans.
void releaseAncestorRejected2(StrSpan markedExpr,
            Memory& memoryBlock,
            StrSpan validityName);

private:
/// @brief Full per-LB registry reset for a PARKED or statement about
        ///        to be re-deposited — the ordis revival's "un-know".
        ///
        /// @details
        /// The local-statement sibling of
        /// @ref resetResentExpressionRegistries: the parked or statement was
        /// consumed flat at its first arrival and is a KNOWN (and local)
        /// statement of the LB, so a plain re-deposit would be dropped by
        /// the Site F ancestor dedup before the or consumption re-runs.
        /// This helper erases the statement's rows from every dedup
        /// registry INCLUDING the local statement vectors (which
        /// `resetResentExpressionRegistries`'s non-local contract excludes):
        /// `intStatementLevelsMap`, `intKnownStatements` (the known bit's
        /// row — the documented, maintainer-approved exception to the
        /// OR-only flag discipline of I-85, precedented by the CE
        /// teardown's full paired reset per I-58),
        /// `intEncodedStatements`, `intLocalEncodedStatements`, and
        /// `intLocalEncodedStatementsDelta`. The
        /// `intLocalEncodedStatementsSet` membership is left in place: the
        /// re-absorbed statement re-registers as local, and the set is a
        /// monotone locality index whose transient over-approximation is
        /// harmless (the row-bearing registries are the gates). History
        /// (`exprOriginMap`) is NOT touched (I-44, Rule 16).
        ///
        /// @param mb           Owning LB.
        /// @param original     The parked or statement's clean text.
        /// @param validityName The cohort parent scope.
        /// @return (void)
        /// @invariant Called only by @ref revisitRejectedOrdis, immediately
        ///            before the mail re-deposit of the same statement.
        /// @see resetResentExpressionRegistries — the non-local sibling.
void resetParkedOrStatementRegistries(Memory& mb,
            StrSpan original,
            StrSpan validityName);

public:
/// @brief Revive parked or-cohorts whose product template just
        ///        gained an admission key — the `rejectedMapOrdis` mirror of
        ///        @ref revisitRejected2.
        ///
        /// @details
        /// Hooked at every seam where the algebra `admissionMap` gains keys
        /// (the drain step after `revisitRejected2`, the Writer-B inline
        /// insert, the recursion propagation insert, and the equi-class
        /// admission re-key), firing for TAGGED and untagged gains alike —
        /// the cohort probe reads both. Mechanics mirror `revisitRejected2`:
        /// non-minting template probe (a never-interned template was never
        /// parked — a defined miss), a DEDICATED re-entrancy guard
        /// (`ordisRevisitInProgress`), copy-before-erase snapshot of the
        /// parked run onto gen-scratch, whole-cohort `eraseBlobIf`, then a
        /// per-record MAIL re-deposit of the parked or statement on
        /// `sameIterationInternalMail` — the absorb re-runs the FULL or
        /// consumption (fresh probe over all standing demands, standing
        /// starter tie rule, K rules deduped, fresh-cohort guard against
        /// double-open; a revival may legitimately re-park, the map entry
        /// already erased). Because the parked statement is a KNOWN local
        /// statement, each re-deposit is preceded by
        /// @ref resetParkedOrStatementRegistries — the maintainer-approved
        /// un-know that lets the deposit pass the Site F ancestor dedup.
        ///
        /// @param markedExpr   The admission template that gained a key
        ///                     (bare marker form).
        /// @param memoryBlock  The owning LB.
        /// @param validityName The key's validity scope (exact-validity
        ///                     matching; widening deferred by design).
        /// @return (void)
        /// @invariant Never inserts into `rejectedMapOrdis`; re-park happens
        ///            only through the re-consumption's own park path.
        /// @see revisitRejected2 — the discipline template;
        ///      drainAdmissionKeysAlgebra, updateAdmissionMap,
        ///      updateAdmissionMapRecursion,
        ///      applyEquivalenceClassToAdmissionMap — the four seams.
void revisitRejectedOrdis(StrSpan markedExpr,
            Memory& memoryBlock,
            StrSpan validityName, int64_t selectedKey = 0);

public:
/// @brief Wake the or heads filed in `rejectedMapOrdis2` under one
        ///        ground-text key — the ordis2 pair's rendezvous, fired by
        ///        the demand drain (D-267).
        ///
        /// @details
        /// Twin of @ref revisitRejectedOrdis pointed at the SECOND park
        /// index, whose keys are eligible disjuncts' clean GROUND texts —
        /// the demand map's own key language, so the wake is a plain key
        /// rendezvous with no product-template translation. It is called by
        /// the two demand writers, `drainAdmissionKeysOrdis2` and
        /// `AdmissionMemory::fileHypothesisOrDemand` (the induction
        /// hypothesis's missing expression, D-575),
        /// and by `drainCanonicalWakes` for a wake deferred during canonical
        /// maintenance. The algebra key-gain seams keep waking only the old
        /// map — different key language; a cross-probe would double-wake
        /// operator-application disjuncts whose old-map product template IS
        /// the disjunct text. Mechanics mirror the twin: non-minting template
        /// probe (a never-interned text was never filed — a defined miss),
        /// the DEDICATED `ordis2RevisitInProgress` guard, copy-before-erase
        /// snapshot onto gen-scratch, whole-key `eraseBlobIf`, per-record
        /// @ref resetParkedOrStatementRegistries un-know + mail re-deposit on
        /// `sameIterationInternalMail`, or on the running absorb's output
        /// channel (`Memory::replayMailDuringAbsorb`) when the wake comes from
        /// inside an absorb, whose own input clear would drop it (the
        /// `queueParkedExistence` routing); the absorb re-runs the FULL or
        /// consumption, whose route-(c) probe reads the freshly-drained
        /// demand and opens while preserving it for other sources. Entries
        /// under other keys and in `rejectedMapOrdis` stay; the cohort
        /// bootstrap guard makes stale wakes idempotent. A route-(c) probe
        /// removes a satisfied demand, and scope wipe cleans at discharge.
        ///
        /// @param groundText   The demand's ground compound premise text
        ///                     (polarity verbatim, I-175).
        /// @param memoryBlock  The owning LB.
        /// @param validityName The demand key's validity scope
        ///                     (exact-validity matching, like the twin).
        /// @return (void)
        /// @invariant Never inserts into `rejectedMapOrdis2`; re-park
        ///            happens only through the re-consumption's own park
        ///            path.
        /// @see revisitRejectedOrdis — the discipline template;
        ///      drainAdmissionKeysOrdis2, AdmissionMemory::fileHypothesisOrDemand,
        ///      drainCanonicalWakes — the calling seams.
void revisitRejectedOrdis2(StrSpan groundText,
            Memory& memoryBlock,
            StrSpan validityName, int64_t selectedKey = 0);

public:
// Integration-side analog of revisitRejected2. Fired whenever a NEW
        // admissionMapIntegration key (or admissionSetIntegration entry) is
        // inserted that could now admit a previously-rejected constituent.
        // Emits constituents to sameIterationInternalMail — does NOT call
        // addExprToMemoryBlock (user-specified, to avoid cyclic re-entry).
        // Does NOT erase the admission key. A cohort parked at a strict
        // ANCESTOR of validityName is released into validityName and KEPT
        // (releaseAncestorRejectedIntegration2), mirroring revisitRejected2.
void revisitRejectedIntegration2(StrSpan markedKey,
            Memory& memoryBlock,
            StrSpan validityName);

/// @brief Decide at a rule's installation whether it is demand-built, and
///        register it when it is.
///
/// @details
/// The block's anchor arguments bound the sets and operators the criterion
/// admits; `analyzeDemandBuiltRule` applies the criterion to the rule text and
/// the compiled definitions. A demand-built rule is parsed into the
/// process-wide registry, once per rule text; the caller then records that the
/// block holds it instead of installing it
/// (D-596).
///
/// @param memoryBlock The installing block.
/// @param rule        The expanded rule text.
/// @param carrier     The statement whose expansion produced it.
/// @return True when the rule is demand-built and must not be installed.
/// @invariant Mints nothing in the block; allocates nothing.
/// @see analyzeDemandBuiltRule, CompactRuleProcessor::installExpandedImplication.
bool deferDemandBuiltRule(Memory& memoryBlock, StrSpan rule, StrSpan carrier);

/// @brief Build the instances of the demand-built rules a demand key names.
///
/// @details
/// Every registered rule whose witness template matches @p markedKey is built
/// for the matched values, in rule-text order, by `buildDemandedInstance`.
/// Called at every key gain that passes a revival seam with a visible demand,
/// and by `buildWaitingInstances`.
///
/// @param memoryBlock  The owning block.
/// @param markedKey    The demand key in bare marker form.
/// @param validityName The demand's scope.
/// @return Nothing; writes the internal mail.
/// @invariant Heap-free; the matched order is deterministic.
/// @see buildDemandedInstance, buildWaitingInstances.
void buildDemandedInstances(Memory& memoryBlock, StrSpan markedKey, StrSpan validityName);

/// @brief Offer every stored demand of a demand-built rule's shape to the
///        build again, once per burst.
///
/// @details
/// A demand filed before the block held the rule, or before an input was
/// typed, built nothing when it was filed; the eager rule would fire in the
/// burst the late rule or typing arrived. At the end of every burst's algebra
/// admission drain, each stored algebra and integration demand key that
/// matches a registered witness template and has a visible demand is offered
/// to `buildDemandedInstances`, and each stored case demand
/// (`admissionMapOrdis2`) shaped like a registered OR head's disjunct to
/// `buildDemandedOrCaseInstances`; a known instance builds nothing.
///
/// @param memoryBlock The owning block.
/// @return Nothing; writes the internal mail.
/// @invariant Heap-free; keys are visited in map order.
/// @see AdmissionMemory::drainAdmissionKeysAlgebra.
void buildWaitingInstances(Memory& memoryBlock);

/// @brief Build the instances of the OR heads the staged events name
///        (D-597).
///
/// @details
/// Drains `Memory::pendingDeferredOrEvents` once per `standardProcessing`
/// call, before the OR exclusion drain. A staged row that is neither known
/// nor a goal any more is skipped. A row a waiting event names as its missing
/// premise re-stages those events (`Memory::deferredOrWaiting`) into this
/// drain. A known row that matches a registered OR head's refutation
/// template, or a goal that matches its goal template
/// (`matchDemandBuiltTemplate`), builds that instance through
/// `buildDemandedInstance` into @p output, in rule-text then template order;
/// an instance whose premise is not known on the row's chain waits under that
/// premise's text. Every other outcome (built, known, no rule held, diagonal,
/// beyond the generation cap) ends the event: a later rule arrival re-scans
/// the block (`stageExistingDeferredOrEvents`).
///
/// @param memoryBlock The owning block.
/// @param output      The running call's internal-mail output channel.
/// @return Nothing; writes @p output and the waiting index.
/// @invariant Heap-free (the slot's scratch arenas); the pending list is
///            empty on return.
/// @see StatementArrivals::stageDeferredOrEvent, OrImplicationProcessor::drainOrImplications.
void drainDeferredOrEvents(Memory& memoryBlock, ColdMail& output);

/// @brief Stage every known row that is an event of a newly held OR-headed
///        demand-built rule (D-597).
///
/// @details
/// A fact that refuted a disjunct, or a goal equal to an instance, before the
/// block held the rule built nothing when it arrived. When the block first
/// holds the rule at a scope, every known row with the shape of one of the
/// rule's refutation templates and every goal with the shape of its goal
/// template is staged for the next `drainDeferredOrEvents`. An
/// operator-totality rule stages nothing (its demands are served by
/// `buildWaitingInstances`).
///
/// @param memoryBlock The owning block.
/// @param rule        The rule text just recorded as held.
/// @return Nothing; appends to `Memory::pendingDeferredOrEvents`.
/// @invariant The rule is registered (asserted); mints nothing.
/// @see CompactRuleProcessor::installExpandedImplication.
void stageExistingDeferredOrEvents(Memory& memoryBlock, StrSpan rule);

/// @brief Build the instances of the OR heads a case demand names
///        (D-597).
///
/// @details
/// A case-split demand for a disjunct opens an OR only when that OR is
/// deposited. Every registered OR head whose case-demand template matches
/// @p groundText is built through `buildDemandedInstance`, in rule-text then
/// template order, into the running absorb's output channel inside an
/// absorb, else into the same-iteration channel. A refused build ends here:
/// the stored demand is offered again once per burst by
/// `buildWaitingInstances`.
///
/// @param memoryBlock  The owning block.
/// @param groundText   The demanded disjunct, ground text.
/// @param validityName The demand's scope.
/// @return Nothing; writes the internal mail.
/// @invariant Heap-free; copies its inputs before the build mints (I-3).
/// @see revisitRejectedOrdis2, buildWaitingInstances.
void buildDemandedOrCaseInstances(Memory& memoryBlock, StrSpan groundText, StrSpan validityName);

private:
/// @brief Build one demand-built rule's instance for matched values, exactly
///        as the rule's firing would.
///
/// @details
/// Nothing is built unless, on the demand scope's chain, the block holds the
/// rule (its installed-rule index) and knows every instantiated premise; nor
/// for a repeated premise instance (the hash engine never uses one statement
/// for two premises), a premise beyond the generation cap, or a head already
/// known. Otherwise the head instance enters @p output as the firing record
/// would deposit it: at the deepest of the rule's scope and the premises'
/// scopes (each the shallowest scope that knows it), with the union of their
/// levels and the carrier's, witness generation one above the premises', the
/// `implication` history row (the rule, then the premises in expression
/// order) and the incubator disintegration gate (D-29).
///
/// @param memoryBlock    The owning block.
/// @param rule           The demand-built rule.
/// @param values         Each bound variable's value.
/// @param validityName   The demand's scope.
/// @param output         The internal-mail channel the instance enters.
/// @param missingPremise Receives the position of the first premise not known
///                       on the chain when the result is `Untyped`.
/// @return The outcome (`DemandBuildResult`).
/// @invariant Heap-free (stack buffers and the slot's scratch arena).
/// @see HashRuleEvaluator::checkLocalEncodedMemoryStatic, the firing record.
DemandBuildResult buildDemandedInstance(Memory& memoryBlock, const DemandBuiltRule& rule,
            const StrSpan* values, StrSpan validityName, ColdMail& output, int32_t& missingPremise);

public:

public:
/// @brief Wake the parked integration sources a newly filed double-marker
        ///        key admits (D-492).
        ///
        /// @details
        /// The double-marker twin of the revival after a new
        /// `admissionMapIntegration` key. Candidates are the
        /// `rejectedMapIntegration` parks with the key's predicate whose scope
        /// is comparable to @p validityName. A park at that scope or below is
        /// revisited at its own scope; a park at a strict ancestor is revisited
        /// at the key's scope, which releases it into the key's scope
        /// (`releaseAncestorRejectedIntegration2`). A candidate is revisited only
        /// when the route-5 probe (`AdmissionMemory::isAdmittedIntegrationDouble`)
        /// admits its marked text at the revisit scope. Revisits run in decoded
        /// (template, scope) order, each distinct pair once; the park texts are
        /// copied before any probe, since the class-aware match mints templates.
        ///
        /// @param mb           Owning block (single-threaded seam).
        /// @param keyTemplate  The new key's template (`u_` form with
        ///                     `marker1` / `marker2`); only its predicate is read.
        /// @param validityName The key's scope.
        /// @return (void)
        /// @invariant Writes no admission or park record itself; every wake goes
        ///            through `revisitRejectedIntegration2`.
        /// @see AdmissionMemory::updateAdmissionMapIntegrationDouble,
        ///      revisitRejectedIntegration2.
void revisitRejectedIntegrationDouble(Memory& mb, StrSpan keyTemplate, StrSpan validityName);

public:
/// @brief Wake the integration parks in the OR cases a scope that gained a
        ///        route-3 demand key serves (D-572).
        ///
        /// @details
        /// A route-3 demand is visible at its scope and at the OR cases it
        /// serves (`AdmissionMemory::ancestorDemandOwner`), so a source parked
        /// in such a case before the demand arrived can now be admitted there.
        /// Candidates are the `rejectedMapIntegration` parks at a scope whose
        /// serving owner is @p validityName and whose template has the key's
        /// predicate; each distinct (template, park scope) pair whose route-3
        /// probe at the park scope (`AdmissionMemory::isAdmittedIntegration`,
        /// no commit) succeeds is revisited at the park scope through
        /// `revisitRejectedIntegration2`, in decoded (template, scope) order.
        /// The key's own scope and its ancestors are the existing
        /// `revisitRejectedIntegration2` call's at the filing site. Park texts
        /// and scopes are copied before any probe, since the class-aware match
        /// mints templates.
        ///
        /// @param mb           Owning block (single-threaded seam).
        /// @param bareKey      The new key's bare marker template (no `u_`
        ///                     prefixes); only its predicate is read.
        /// @param validityName The key's scope, registered.
        /// @return (void)
        /// @invariant Writes no admission or park record itself; every wake goes
        ///            through `revisitRejectedIntegration2`.
        /// @see IntegrationProcessor::prepareIntegrationCore2,
        ///      EquivalenceAdmissionRewriter::applyEquivalenceClassToAdmissionMapIntegration
        ///      — the filing sites; revisitRejectedIntegrationDouble — the
        ///      route-5 twin.
void wakeDescendantIntegrationParks(Memory& mb, StrSpan bareKey, StrSpan validityName);

public:
/// @brief Wake the parked integration sources a newly filed wildcard key
        ///        admits (D-586).
        ///
        /// @details
        /// The wildcard twin of the revival after a new
        /// `admissionMapIntegration` key. A wildcard key spells the other open
        /// argument of its part `pi_`, so its text names no park and the exact
        /// revival finds none. Candidates are the `rejectedMapIntegration`
        /// parks with the key's predicate owned at the key's scope or at a
        /// strict ancestor of it; each distinct park template whose route-3
        /// probe at the key's scope (`AdmissionMemory::isAdmittedIntegration`,
        /// no commit, the wildcard variants included) succeeds is revisited at
        /// the key's scope through `revisitRejectedIntegration2`, which also
        /// releases an ancestor's park into the scope. During a canonical
        /// maintenance pass the revisit only queues, as in
        /// `wakeDescendantIntegrationParks`. Revisits run in decoded template
        /// order; the park texts are copied before any probe, since the
        /// class-aware match mints templates. The parks in the OR cases the
        /// scope serves remain `wakeDescendantIntegrationParks`'s.
        ///
        /// @param mb           Owning block (single-threaded seam).
        /// @param bareKey      The new key's bare template (no `u_` prefixes);
        ///                     only its predicate is read.
        /// @param validityName The key's scope, registered.
        /// @return (void)
        /// @invariant Writes no admission or park record itself; every wake goes
        ///            through `revisitRejectedIntegration2`.
        /// @see IntegrationProcessor::prepareIntegrationCore2 — the filing site;
        ///      revisitRejectedIntegrationDouble — the route-5 twin.
void wakeWildcardIntegrationParks(Memory& mb, StrSpan bareKey, StrSpan validityName);

private:
/// @brief Release into `validityName` every `rejectedMapIntegration`
        ///        cohort parked under `markedKey` at a strict ancestor scope
        ///        of `validityName`; the ancestor parks stay.
        ///
        /// @details
        /// The integration twin of @ref releaseAncestorRejected2: owned
        /// ancestor copies (@ref copyStrictAncestorSpans), a non-minting
        /// `rejectedMapIntegration` probe per `(markedKey, ancestor)`, a
        /// gen-scratch snapshot of each hit, and one
        /// @ref releaseParkedProductIntoScope call per record — constituent
        /// plus siblings, with the compound's level run looked up at the PARK
        /// scope (where the compound is known), exactly as the exact-scope
        /// revival looks it up at its own scope. No budget gates (the
        /// integration revival has none) and nothing erased.
        ///
        /// @param markedKey    The bare marker-form integration template that
        ///                     gained a key.
        /// @param memoryBlock  The owning LB.
        /// @param validityName The key's scope — the release target.
        /// @return (void)
        /// @invariant Never writes `rejectedMapIntegration` (I-37); probes
        ///            never mint (I-89); I-22 untouched (no admission entry
        ///            is consumed).
        /// @see revisitRejectedIntegration2 — the caller;
        ///      releaseParkedProductIntoScope — the per-record door.
void releaseAncestorRejectedIntegration2(StrSpan markedKey,
            Memory& memoryBlock,
            StrSpan validityName);
private:

AdmissionPolicy& admissionPolicy;
const ce::AnchorInfo& anchorInfo;
CanonicalTemplateMatcher& canonicalTemplates;
Canonicalizer& canonicalizer;
const CompiledDefinitionRegistry& definitions;
const ProverParameters& parameters;
RejectedSourceMemory& rejectedSources;
AdmissionMemory& admissionMemory;
CaseLedger& caseLedger;
ProofHistory& proofHistory;
WatchProcessor& watchProcessor;
};
}
#include "../logical_block/admission_memory.hpp"
#include "../logical_block/case_ledger.hpp"
#include "../logical_block/proof_history.hpp"
#include "../logical_block/watch_processor.hpp"
namespace gl {

/// @brief Wake newly parked or rewritten sources against existing demand.
        /// @details A derived index maps prefix-stripped demand text to packed keys.
        /// Only matching keys enter the original decoded expression/scope sort;
        /// every wake retains the ordinary admission probe. An integration park
        /// is also woken at each double-marker key scope whose route-5 probe
        /// admits it (D-492), in decoded
        /// (template, scope) order.
        /// @param memoryBlock Owning memory block.
        /// @param algebraParks Concrete algebra keys to revisit.
        /// @param integrationParks Concrete integration keys to revisit.
        /// @param defer Whether an active canonical maintenance pass queues the parks.
        /// @return Void.
inline void RejectedSourceReplay::wakeParkedExistencesAtExistingDemands(Memory& memoryBlock,
            const ColdHashSet<BytesKeyStore>& algebraParks,
            const ColdHashSet<BytesKeyStore>& integrationParks, bool defer) {
            auto& pending = memoryBlock.lbMemory.canonicalUses;
            if (defer && pending.deferWakes) {
                for (int32_t i=1;i<=algebraParks.count();++i) pending.algebraWakeParks.mint(algebraParks.keyAt(i));
                for (int32_t i=1;i<=integrationParks.count();++i) pending.integrationWakeParks.mint(integrationParks.keyAt(i));
                return;
            }
            const unsigned slot = gl::WorkerContext::coreId() >= 0
                ? static_cast<unsigned>(gl::WorkerContext::coreId()) : genScratchArenas().slotCount() - 1;
            ScratchArena& arena = genScratchArenas().forSlot(slot);
            ScratchArena& strings = scratchArenas().forSlot(slot);
            ScratchScope frame(arena);
            ScratchScope stringFrame(strings);
            const auto wakeExisting = [&](const auto& demands, CanonicalTemplateIndex& lookup, bool integration, bool prefixed) {
                DirtyState dirty = DirtyState::Clean;
                PagedVector<int64_t> matches(&arena, &dirty);
                {
                    RT_SCOPE_HERE("SOURCE_DEMAND_FILTER");
                    const auto& parks = integration ? integrationParks : algebraParks;
                    if(demands.empty() || parks.empty()) return;
                    canonicalTemplates.ensureCanonicalTemplateIndex(memoryBlock,demands,lookup,arena);
                    TypedColdSet<int64_t> selected(&arena,&dirty);
                    for(int32_t i=1;i<=parks.count();++i) {
                        ScratchScope parkFrame(strings);
                        const StrSpan park=parks.keyAt(i);
                        const ScratchString query = prefixed
                            ? gl::DefinitionExpansion::prefixNonMarkerArgumentsWithUScratch(strings, park)
                            : ScratchString::copyFrom(strings, park.ptr, park.len);


                        const StrSpan text(query);
                        int32_t length=0;
                        while(length<text.len && text[length]!='[') ++length;
                        const int32_t label=lookup.labels.lookup(StrSpan(text.ptr,length));
                        if(label==0) continue;
                        PagedVector<NameId> scopes(&arena,&dirty);
                        lookup.scopesByShape.forEach(label,[&](int64_t scope) { scopes.push_back(static_cast<NameId>(scope)); });
                        for(int32_t j=0;j<scopes.size();++j) {
                            PagedVector<int64_t> found(&arena,&dirty);
                            canonicalTemplates.matchingCanonicalTemplates(memoryBlock,demands,lookup,text,scopes[j],scopes[j],found,arena);
                            for(int32_t k=0;k<found.size();++k) selected.mint(found[k]);
                        }
                    }
                    for(int32_t i=1;i<=selected.count();++i) matches.push_back(selected.keyAt(i));
                }
                RT_SCOPE_HERE("SOURCE_DEMAND_ORDER_AND_WAKE");
                const int32_t n = matches.size();
                const auto less = [&](int32_t a, int32_t b) {
                    const auto ka = Codec<StatementKey>::decode(matches[a]);
                    const auto kb = Codec<StatementKey>::decode(matches[b]);
                    const int c = compareSpans(memoryBlock.templateInterner.decodeView(ka.orig),
                        memoryBlock.templateInterner.decodeView(kb.orig));
                    return c != 0 ? c < 0 : compareSpans(memoryBlock.nameMap.decodeView(ka.validity),
                        memoryBlock.nameMap.decodeView(kb.validity)) < 0;
                };
                ChunkSortedOrdinals<decltype(less)> order(arena, n, 0, less);
                for (int32_t i = 0; i < n; ++i) {
                    ScratchScope demandFrame(strings);
                    const auto key = Codec<StatementKey>::decode(matches[order.next()]);
                    const ScratchString clean = gl::DefinitionExpansion::removeUPrefixScratch(strings, memoryBlock.templateInterner.decodeView(key.orig));
                    const StrSpan scopeView = memoryBlock.nameMap.decodeView(key.validity);
                    const ScratchString scope = ScratchString::copyFrom(strings, scopeView.ptr, scopeView.len);
                    if (integration) revisitRejectedIntegration2(StrSpan(clean), memoryBlock, StrSpan(scope));
                    else revisitRejected2(StrSpan(clean), memoryBlock, StrSpan(scope));
                }
            };
            auto& indexes = memoryBlock.lbMemory.canonicalUses;
            if (!algebraParks.empty())
                wakeExisting(memoryBlock.overallHashMemory.admissionMap, indexes.algebraMatches, false, false);
            if (!integrationParks.empty()) {
                wakeExisting(memoryBlock.overallHashMemory.admissionMapIntegration, indexes.integrationMatches, true, true);
                wakeExisting(memoryBlock.overallHashMemory.admissionSetIntegration, indexes.integrationSetMatches, true, false);
            }
            // Double-marker keys (D-492): a park
            // wakes at every key scope whose route-5 probe admits it; the
            // revisit there releases the park's copies at that scope and its
            // ancestors. Park texts and key scopes are copied before any probe,
            // since the class-aware match mints templates.
            const TypedColdMap<int64_t, int64_t>& doubles = memoryBlock.overallHashMemory.admissionMapIntegrationDouble;
            if (integrationParks.empty() || doubles.empty()) return;
            RT_SCOPE_HERE("SOURCE_DOUBLE_DEMAND_WAKE");
            DirtyState doubleDirty = DirtyState::Clean;
            ColdHashSet<BytesKeyStore> texts(&arena, &doubleDirty);
            ColdHashSet<PodKeyStore<int64_t>> pairs(&arena, &doubleDirty);
            struct DoubleWake { int32_t text; NameId scope; };
            PagedVector<DoubleWake> wakes(&arena, &doubleDirty);
            for (int32_t i = 1; i <= integrationParks.count(); ++i) {
                ScratchScope parkFrame(strings);
                const ScratchString park = ScratchString::copyFrom(strings,
                    integrationParks.keyAt(i).ptr, integrationParks.keyAt(i).len);
                int32_t predicate = 0;
                while (predicate < StrSpan(park).len && StrSpan(park).ptr[predicate] != '[') ++predicate;
                for (int32_t id = 1; id <= doubles.count(); ++id) {
                    const StatementKey key = Codec<StatementKey>::decode(doubles.keyAt(id));
                    {
                        const StrSpan keyText = memoryBlock.templateInterner.decodeView(key.orig);
                        if (keyText.len <= predicate || keyText.ptr[predicate] != '['
                            || std::memcmp(keyText.ptr, StrSpan(park).ptr, static_cast<std::size_t>(predicate)) != 0)
                            continue;
                    }
                    ScratchScope scopeFrame(strings);
                    const StrSpan scopeView = memoryBlock.nameMap.decodeView(key.validity);
                    const ScratchString scope = ScratchString::copyFrom(strings, scopeView.ptr, scopeView.len);
                    if (!admissionMemory.isAdmittedIntegrationDouble(memoryBlock, StrSpan(park), StrSpan(scope)))
                        continue;
                    const int32_t text = texts.mint(StrSpan(park));
                    const int32_t before = pairs.count();
                    pairs.mint(packInt32Pair(text, key.validity));
                    if (pairs.count() != before) wakes.push_back(DoubleWake{ text, key.validity });
                }
            }
            const auto less = [&](int32_t a, int32_t b) {
                const int c = compareSpans(texts.keyAt(wakes[a].text), texts.keyAt(wakes[b].text));
                return c != 0 ? c < 0 : compareSpans(memoryBlock.nameMap.decodeView(wakes[a].scope),
                    memoryBlock.nameMap.decodeView(wakes[b].scope)) < 0;
            };
            ChunkSortedOrdinals<decltype(less)> order(arena, wakes.size(), 0, less);
            for (int32_t i = 0; i < wakes.size(); ++i) {
                ScratchScope wakeFrame(strings);
                const DoubleWake wake = wakes[order.next()];
                const StrSpan textView = texts.keyAt(wake.text);
                const ScratchString text = ScratchString::copyFrom(strings, textView.ptr, textView.len);
                const StrSpan scopeView = memoryBlock.nameMap.decodeView(wake.scope);
                const ScratchString scope = ScratchString::copyFrom(strings, scopeView.ptr, scopeView.len);
                revisitRejectedIntegration2(StrSpan(text), memoryBlock, StrSpan(scope));
            }
        }

/// @brief Coalesce a wake while canonical source and demand identities change.
        /// @details Outside maintenance the caller continues its ordinary immediate
        /// wake. During maintenance only logical template and scope keys are stored.
        /// @param mb Owning block.
        /// @param text Bare marked or ground wake template.
        /// @param validity Existing target scope.
        /// @param kind Algebra, integration, OR or OR2 bit (1, 2, 4 or 8).
        /// @return True when queued, false when immediate wake processing applies.
inline bool RejectedSourceReplay::queueCanonicalWake(Memory& mb, StrSpan text, StrSpan validity, int32_t kind) {
            assert(kind==1 || kind==2 || kind==4 || kind==8);
            auto& pending=mb.lbMemory.canonicalUses;
            if (!pending.deferWakes) return false;
            const NameId scope=mb.nameMap.lookup(validity);
            assert(scope!=0);
            const int64_t key=packStatementKey(mb.templateInterner.encode(text),scope);
            const int32_t* previous=pending.wakeKinds.find(key);
            pending.wakeKinds.upsert(key,kind | (previous==nullptr ? 0 : *previous));
            return true;
        }

/// @brief Release coalesced wakes after the complete canonical maintenance pass.
        /// @details Source-last matches are collected against settled demand keys.
        /// Wakes run in decoded template and scope order with original kind priority;
        /// the queue is detached first so ordinary arrivals can wake immediately.
        /// @param mb Owning block with an active maintenance pass.
        /// @return Nothing.
        /// @invariant Obsolete rule removal and all source/demand re-keying precede this call.
inline void RejectedSourceReplay::drainCanonicalWakes(Memory& mb) {
            auto& pending=mb.lbMemory.canonicalUses;
            assert(pending.deferWakes);
            wakeParkedExistencesAtExistingDemands(mb,pending.algebraWakeParks,pending.integrationWakeParks,false);
            pending.algebraWakeParks.resetToFresh();
            pending.integrationWakeParks.resetToFresh();
            const unsigned slot=gl::WorkerContext::coreId()>=0 ? static_cast<unsigned>(gl::WorkerContext::coreId()) : genScratchArenas().slotCount()-1;
            ScratchArena& arena=genScratchArenas().forSlot(slot);
            ScratchArena& strings=scratchArenas().forSlot(slot);
            ScratchScope frame(arena), textFrame(strings);
            DirtyState dirty=DirtyState::Clean;
            struct Wake { int64_t key; int32_t kinds; };
            PagedVector<Wake> wakes(&arena,&dirty);
            for(int32_t i=1;i<=pending.wakeKinds.count();++i) {
                const int64_t key=pending.wakeKinds.keyAt(i);
                wakes.push_back(Wake{key,*pending.wakeKinds.find(key)});
            }
            pending.wakeKinds.resetToFresh();
            pending.deferWakes=false;
            const auto less=[&](int32_t a,int32_t b) {
                const auto x=Codec<StatementKey>::decode(wakes[a].key);
                const auto y=Codec<StatementKey>::decode(wakes[b].key);
                const int c=compareSpans(mb.templateInterner.decodeView(x.orig),mb.templateInterner.decodeView(y.orig));
                return c!=0 ? c<0 : compareSpans(mb.nameMap.decodeView(x.validity),mb.nameMap.decodeView(y.validity))<0;
            };
            ChunkSortedOrdinals<decltype(less)> order(arena,wakes.size(),0,less);
            for(int32_t i=0;i<wakes.size();++i) {
                ScratchScope rowFrame(strings);
                const Wake wake=wakes[order.next()];
                const auto key=Codec<StatementKey>::decode(wake.key);
                if(mb.phase3IgnoresScope(key.validity)) continue;
                const StrSpan textView=mb.templateInterner.decodeView(key.orig);
                const ScratchString text=ScratchString::copyFrom(strings,textView.ptr,textView.len);
                const StrSpan scopeView=mb.nameMap.decodeView(key.validity);
                const ScratchString scope=ScratchString::copyFrom(strings,scopeView.ptr,scopeView.len);
                if(wake.kinds&1) revisitRejected2(StrSpan(text),mb,StrSpan(scope));
                if(wake.kinds&2) revisitRejectedIntegration2(StrSpan(text),mb,StrSpan(scope));
                if(wake.kinds&4) revisitRejectedOrdis(StrSpan(text),mb,StrSpan(scope));
                if(wake.kinds&8) revisitRejectedOrdis2(StrSpan(text),mb,StrSpan(scope));
            }
        }

/// @brief Reject parked-source searches with no possible admission route.
        ///
        /// @details This is a conservative negative filter. Algebra reads local
        /// and ancestor demands at the target scope and preserves standalone
        /// single-input operator admission. Integration reads the prefixed
        /// instruction keys visible at the scope — owned there or at a strict
        /// ancestor (`AdmissionMemory::integrationDemandVisible`,
        /// D-572) — and the exact-scope ordinary
        /// admission set, then the double-marker keys at the scope and its
        /// ancestors (route 5). A positive answer leaves every source-specific
        /// budget and replay check intact. Deferred canonical wakes are queued
        /// before this filter is called.
        /// @param mb Owning block and current demand indexes.
        /// @param key Caller-owned concrete marked key, stable across cache updates.
        /// @param validity Caller-owned interned target scope spelling.
        /// @param integration Select the integration rather than algebra route.
        /// @return False only when the existing replay predicate cannot admit a source.
        /// @invariant Ancestor demand is visible through the ordinary algebra
        ///            route and the double-marker route; the integration
        ///            instruction route reads, beyond its own scope, the
        ///            serving owner of an OR case
        ///            (`AdmissionMemory::ancestorDemandOwner`); the integration
        ///            admission set stays exact-scope.
        /// @see queueParkedExistence, revisitRejected2, revisitRejectedIntegration2.
inline bool RejectedSourceReplay::hasPotentialParkedDemand(Memory& mb, StrSpan key,
                StrSpan validity, bool integration) {
            RT_SCOPE_HERE("PARK_DEMAND_PREFILTER");
            const NameId target = mb.nameMap.lookup(validity);
            assert(target != 0);
            const unsigned slot = gl::WorkerContext::coreId() >= 0
                ? static_cast<unsigned>(gl::WorkerContext::coreId()) : scratchArenas().slotCount() - 1;
            ScratchArena& strings = scratchArenas().forSlot(slot);
            ScratchScope frame(strings);
            if (integration) {
                if (!mb.overallHashMemory.admissionMapIntegration.empty()) {
                    const ScratchString prefixed = gl::DefinitionExpansion::prefixNonMarkerArgumentsWithUScratch(strings, key);
                    if (admissionMemory.integrationDemandVisible(mb, StrSpan(prefixed), validity, target))
                        return true;
                }
                if (!mb.overallHashMemory.admissionSetIntegration.empty()
                    && canonicalTemplates.canonicalTemplatePresent(mb, mb.overallHashMemory.admissionSetIntegration,
                        mb.lbMemory.canonicalUses.integrationSetMatches, key, validity, target))
                    return true;
                // Route 5: a double-marker key at the scope or an ancestor
                // (D-492).
                return admissionMemory.isAdmittedIntegrationDouble(mb, key, validity);
            }
            if (admissionPolicy.extractMaxIterationNumber(key) > parameters.maxIterationNumberVariable) return false;
            const StrSpan core = extractExpressionSpan(key);
            if (this->definitions.operatorNames().find(std::string_view(core.ptr, core.len)) != this->definitions.operatorNames().end()) {
                const ce::CoreExpressionConfig* config = this->definitions.coreConfig(core);
                if (config != nullptr && config->inputIndices.size() == 1 && !config->outputIndices.empty()) {
                    StrSpan arguments[ExecutionParameters::MAX_ARITY];
                    const int32_t count = getArgsSpans(key, arguments, ExecutionParameters::MAX_ARITY);
                    const int input = config->inputIndices[0];
                    if (input >= 0 && input < count && equalSpans(arguments[input], StrSpan("marker", 6)))
                        return true;
                }
            }
            if (mb.overallHashMemory.admissionMap.empty()) return false;
            const ScratchString query = canonicalTemplates.canonicalAdmissionText(mb, key, target, strings);
            if (canonicalTemplates.canonicalTemplatePresent(mb, mb.overallHashMemory.admissionMap,
                    mb.lbMemory.canonicalUses.algebraMatches, StrSpan(query), validity, target)) return true;
            StrSpan ancestors[ExecutionParameters::MAX_SCOPE_DEPTH];
            const int32_t count = gl::ScopeSyntax::copyStrictAncestorSpans(mb, validity, ancestors, strings);
            for (int32_t i = 0; i < count; ++i)
                if (canonicalTemplates.canonicalTemplatePresent(mb, mb.overallHashMemory.admissionMap,
                        mb.lbMemory.canonicalUses.algebraMatches, StrSpan(query), ancestors[i], target)) return true;
            return false;
        }

/// @brief Queue one demanded existence for ordinary internal-mail absorption.
        /// @details Admission is probed without effects. Completed source/scope
        /// constructions and already queued constructions are deduplicated. The
        /// ancestor record stays available to other demanding scopes.
        /// @param mb Owning memory block.
        /// @param key Concrete marked demand key.
        /// @param source Parked source record.
        /// @param parkValidity Source scope.
        /// @param targetValidity Demanding scope.
        /// @param integration Whether this is an integration demand index.
        /// @return Whether new replay work was queued.
inline bool RejectedSourceReplay::queueParkedExistence(Memory& mb, StrSpan key,
            const RejectedValueBlobView& source, StrSpan parkValidity,
            StrSpan targetValidity, bool integration) {
            RT_SCOPE_HERE("SOURCE_REPLAY_QUEUE");
            // An integration trial scope, the usefulness scope and the scopes
            // below them mint no witness, so a parked existence never replays
            // into one (D-481, D-521); the park stays
            // available to the other demanding scopes.
            {
                const NameId target = mb.nameMap.lookup(targetValidity);
                if (target != 0 && gl::ScopeSyntax::atOrBelowRestrictedScope(mb.nameMap, target))
                    return false;
            }
            const StrSpan marker("marker", 6);
            int32_t markerOccurrences = 0;
            StrSpan arguments[ExecutionParameters::MAX_ARITY];
            const int32_t argumentCount = getArgsSpans(key, arguments, ExecutionParameters::MAX_ARITY);
            for (int32_t i = 0; i < argumentCount; ++i)
                if (equalSpans(arguments[i], marker)) ++markerOccurrences;
            assert(markerOccurrences != 0);
            const WitnessAdmissionMetrics metrics{source.iteration(),
                admissionPolicy.countPatternOccurrences(key, mb.overallHashMemory, mb.nameMap) + markerOccurrences};
            if (!integration && (source.iteration() > parameters.maxIterationNumberVariable
                || admissionPolicy.extractMaxIterationNumber(key) > parameters.maxIterationNumberVariable)) return false;
            // The parked source becomes the original of its replay: a
            // recursion-owned demand wakes it only when its arguments are
            // all numbers (I-285), the strong step's induction demand only
            // when they are the lemma's input values, sets and operators
            // (D-502) and its inputs are the lemma's induction variables
            // (D-601); neither when an argument
            // repeats (D-530).
            const StrSpan sourceText = mb.valueInterner.decodeView(source.sourceExpressionId());
            const bool sourceRepeats = hasRepeatedArgument(sourceText);
            const bool sourceAdmitsInduction = hasOnlyNumberArguments(sourceText) && !sourceRepeats;
            assert((mb.inductionDoubleKey == 0 || mb.inductionVariableDigitCount > 0)
                && "queueParkedExistence: a strong step block records the lemma's induction variables");
            const bool sourceInductionDigitsAllowed = hasOnlyInductionDigits(sourceText,
                mb.inductionParameterDigits, mb.inductionParameterDigitCount) && !sourceRepeats
                && (mb.inductionDoubleKey == 0
                    || admissionPolicy.inductionInputsAllowed(sourceText, mb.inductionVariableDigits,
                        mb.inductionVariableDigitCount));
            bool demanded;
            if (integration) {
                // An instruction that already refused this source through the
                // witness-link gate is no demand for it
                // (D-529).
                demanded = admissionMemory.integrationDemandOpen(mb, key, targetValidity, sourceText)
                    || canonicalTemplates.canonicalTemplatePresent(mb, mb.overallHashMemory.admissionSetIntegration,
                        mb.lbMemory.canonicalUses.integrationSetMatches, key, targetValidity)
                    || admissionMemory.isAdmittedIntegrationDouble(mb, key, targetValidity,
                        sourceInductionDigitsAllowed);
            } else {
                demanded = admissionMemory.isAdmittedIncludingAncestors(mb, key, marker, key,
                    targetValidity, &metrics, false, sourceAdmitsInduction)
                    || admissionPolicy.isAllowedAsOperatorInput(mb, key, marker, &metrics);
            }
            if (!demanded) return false;
            const StrSpan existence = mb.valueInterner.decodeView(source.sourceExpressionId());
            // The replay's receipt identity, (source, target), is minted
            // before the canonical lookups; queueSourceReplay reads it.
            mb.nameMap.encode(existence);
            const NameId scopeId = mb.nameMap.encode(targetValidity);
            canonicalizer.initializeCanonicalScope(mb, scopeId);
            const NameId canonical = rejectedSources.canonicalExistenceId(mb, existence, scopeId);
            const int32_t aggregate = rejectedSources.existenceStateAt(mb, canonical, scopeId);
            const int32_t available = (source.flags() & 3) & ~(aggregate | (aggregate >> 2));
            if (available == 0 && ((aggregate >> 2) & source.flags() & 3) == 0) return false;
            const unsigned replaySlot = gl::WorkerContext::coreId() >= 0 ? static_cast<unsigned>(gl::WorkerContext::coreId())
                : genScratchArenas().slotCount() - 1;
            ScratchArena& replayArena = genScratchArenas().forSlot(replaySlot);
            ScratchScope replayFrame(replayArena);
            const NameId parkScope = mb.nameMap.lookup(parkValidity);
            assert(parkScope != 0);
            const NameId parkedCanonical = rejectedSources.canonicalExistenceId(mb, existence, parkScope);
            int32_t replayDepth = source.integrationDepth();
            const auto minimizeParkedDepth = [&](const auto& map, ExistenceSourceLookup& lookup) {
                rejectedSources.ensureExistenceSourceLookup(map, lookup, replayArena, mb);
                lookup.postings.forEach(packInt32Pair(parkedCanonical, parkScope), [&](int64_t parkedKey) {
                    const int32_t run = map.lookup(parkedKey);
                    assert(run != 0);
                    for (int32_t r = 0; r < map.runLen(run); ++r) {
                        ScratchScope recordFrame(replayArena);
                        int32_t length = 0;
                        const char* bytes = map.peekRecordBytes(run, r, length, replayArena);
                        const RejectedValueBlobView candidate(bytes, length);
                        if ((candidate.flags() & source.flags() & 3) == 0) continue;
                        const NameId identity = rejectedSources.canonicalExistenceId(mb,
                            mb.valueInterner.decodeView(candidate.sourceExpressionId()), parkScope);
                        if (identity == parkedCanonical)
                            replayDepth = std::min(replayDepth, candidate.integrationDepth());
                    }
                });
            };
            if (integration) minimizeParkedDepth(mb.overallHashMemory.rejectedMapIntegration,
                mb.lbMemory.existenceLookups.integrationSources);
            else minimizeParkedDepth(mb.overallHashMemory.rejectedMap,
                mb.lbMemory.existenceLookups.algebraSources);
            if (available == 0) {
                const int32_t pending = (aggregate >> 2) & source.flags() & 3;
                if (pending != 0) {
                    auto& carriers = mb.lbMemory.canonicalUses.pendingCarriers;
                    const int64_t group = packOriginKey(canonical, scopeId);
                    carriers.forEach(group, [&](int64_t carrier) {
                        const int32_t* receipt = mb.existenceReplays.find(carrier);
                        assert(receipt != nullptr);
                        if (((*receipt >> 2) & pending) == 0) return;
                        replayDepth = std::min(replayDepth, mb.pendingIntegrationDepth(carrier));
                    });
                    carriers.forEach(group, [&](int64_t carrier) {
                        const int32_t* receipt = mb.existenceReplays.find(carrier);
                        assert(receipt != nullptr);
                        if (((*receipt >> 2) & pending) == 0) return;
                        mb.lowerPendingIntegrationDepth(carrier, replayDepth);
                    });
                }
                return false;
            }
            int levels[256];
            assert(source.levelCount() <= 256);
            source.copyLevels(levels);
            if (parameters.trackHistory) {
                const int cap = parameters.compressor_mode
                    ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr;
                for (int32_t i = 0; i < source.rewriteCount(); i += 3) {
                    const StrSpan pre = mb.valueInterner.decodeView(source.rewriteId(i));
                    const StrSpan post = mb.valueInterner.decodeView(source.rewriteId(i + 1));
                    const StrSpan equality = mb.valueInterner.decodeView(source.rewriteId(i + 2));
                    const StrSpan equalityScope=proofHistory.findEqualityCiteScope(mb,equality,parkValidity);
                    const OriginDep dependencies[2] = {{pre, parkValidity}, {equality, equalityScope}};
                    addOriginEncoded(mb.exprOriginMap, mb.originInterner, post,
                        parkValidity, OriginTag::equality1, dependencies, 2, cap);
                    mb.addMailOutOrigin(post, parkValidity, OriginTag::equality1,
                        dependencies, 2, cap);
                }
            }
            queueSourceReplay(mb, existence, parkValidity, targetValidity, available,
                source.flags(), source.iteration(), levels, source.levelCount(), replayDepth);
            return true;
        }

/// @brief Queue one existence source for ordinary internal-mail replay at
        ///        a scope — the shared tail of every source replay.
        ///
        /// @details
        /// Marks the construction kinds @p available as pending in the
        /// source's receipt at @p targetValidity (bits `available << 2`, the
        /// relay bit 16 following @p flags bit 8), deposits the source on the
        /// replay mail (`Memory::replayMailDuringAbsorb` while an absorb
        /// runs, `sameIterationInternalMail` otherwise) with its level run and
        /// a disintegration signal carrying the OR permission (@p flags bit 4),
        /// @p iteration and @p replayDepth, and marks the burst mutated. With
        /// history on, the deposit cites the source at @p parkValidity: a
        /// `zoom in` origin when the scopes differ, otherwise the source's own
        /// origin records copied onto the mail.
        ///
        /// @param mb             Owning block.
        /// @param existence      Source existence text (stable across mints
        ///                       into the NameMap, origin interner and mail).
        /// @param parkValidity   Scope where the source is known.
        /// @param targetValidity Scope that replays the source (the park scope
        ///                       or one below it).
        /// @param available      Construction kinds to replay (bit 1 algebra,
        ///                       bit 2 integration), nonzero.
        /// @param flags          Park-layout flags (4 OR disintegration
        ///                       allowed, 8 inherited relay source).
        /// @param iteration      Witness generation of the replayed source.
        /// @param levels         Ascending level run of the source.
        /// @param levelCount     Length of @p levels.
        /// @param replayDepth    Integration depth the replay assigns.
        /// @return Void.
        /// @invariant Mints the source and target into the NameMap only when new.
        /// @see queueParkedExistence, replayTrialSource.
inline void RejectedSourceReplay::queueSourceReplay(Memory& mb, StrSpan existence, StrSpan parkValidity,
            StrSpan targetValidity, int32_t available, int32_t flags, int32_t iteration,
            const int* levels, int32_t levelCount, int32_t replayDepth) {
            assert(available != 0 && (available & ~3) == 0
                && "queueSourceReplay: replays a nonempty set of construction kinds");
            const NameId expressionId = mb.nameMap.encode(existence);
            const NameId scopeId = mb.nameMap.encode(targetValidity);
            const int64_t receiptKey = packOriginKey(expressionId, scopeId);
            const int32_t* saved = mb.existenceReplays.find(receiptKey);
            const int32_t state = saved ? *saved : 0;
            ColdMail& replayMail = mb.replayMailDuringAbsorb != nullptr
                ? *mb.replayMailDuringAbsorb : mb.sameIterationInternalMail;
            int32_t queued = state | (available << 2);
            if ((flags & 8) == 0) queued &= ~16;
            else if ((state & 12) == 0) queued |= 16;
            rejectedSources.writeExistenceState(mb, receiptKey, queued);
            insertInternalStatement(replayMail, mb.nameMap,
                existence, targetValidity, levels, levelCount);
            replayMail.setDisintegrationSignal(
                packOriginKey(expressionId, scopeId), false,
                (flags & 4) != 0, iteration, false, false, replayDepth);
            mb.mutatedThisBurst = true;
            if (parameters.trackHistory) {
                const int cap = parameters.compressor_mode
                    ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr;
                if (!equalSpans(parkValidity, targetValidity)) {
                    const OriginDep dependency{existence, parkValidity};
                    addInternalMailOrigin(replayMail, mb.originInterner, existence,
                        targetValidity, OriginTag::zoomIn, &dependency, 1, cap);
                    // The replayed source rides the internal mail, never the
                    // delta `fillMailOut` walks, while what is derived from it
                    // reaches the descendants; they must resolve its citation
                    // here (I-64 exception).
                    mb.addMailOutOrigin(existence, targetValidity, OriginTag::zoomIn, &dependency, 1, cap);
                } else {
                    // Ordinary absorption requires the source's paired history.
                    // Internal mail and local origins share the origin interner.
                    int64_t originKey = 0;
                    const bool hasKey = lookupOriginKey(mb.originInterner,
                        existence, parkValidity, originKey);
                    assert(hasKey);
                    const int32_t originId = mb.exprOriginMap.lookup(originKey);
                    assert(originId != 0 && mb.exprOriginMap.runLen(originId) > 0);
                    const unsigned slot = gl::WorkerContext::coreId() >= 0
                        ? static_cast<unsigned>(gl::WorkerContext::coreId()) : genScratchArenas().slotCount() - 1;
                    ScratchArena& arena = genScratchArenas().forSlot(slot);
                    ScratchScope historyFrame(arena);
                    for (int32_t r = 0; r < mb.exprOriginMap.runLen(originId); ++r) {
                        int32_t length = 0;
                        const char* bytes = mb.exprOriginMap.peekRecordBytes(originId, r, length, arena);
                        const IdOriginBlobView view = viewIdOriginBlob(bytes, length);
                        int64_t dependencies[ExecutionParameters::kMaxOriginDeps];
                        assert(view.depN <= ExecutionParameters::kMaxOriginDeps);
                        for (int32_t i = 0; i < view.depN; ++i)
                            dependencies[i] = idOriginDepAt(view, i);
                        addMailOriginRecord(replayMail.origins_, originKey,
                            view.tag, dependencies, view.depN, cap);
                    }
                }
            }
        }

/// @brief Replay a promoted integration trial's source at its parent scope
        ///        (D-481).
        ///
        /// @details
        /// The parent is the trial scope's NameMap parent. The source's level
        /// run is read at the nearest scope at or above the parent that
        /// stores it — the scope where the source is known, which the replay
        /// cites. When the source's integration construction at the parent is
        /// neither completed nor already pending (`existenceStateAt`), the
        /// construction is queued through `queueSourceReplay` with the trial's
        /// flags, generation and integration depth; the replayed source then
        /// meets the passed trial record, which admits the recorded witnesses
        /// under their recorded names, and the ordinary routes commit. A
        /// completed or pending construction needs no new replay.
        ///
        /// @param mb    Owning block.
        /// @param trial The passed trial record.
        /// @return True iff a replay was queued.
        /// @invariant Asserts that the source is known with levels at or above
        ///            the parent.
        /// @see ImplicationSubproofs::drainIntegrationTrials, queueSourceReplay.
inline bool RejectedSourceReplay::replayTrialSource(Memory& mb, const IntegrationTrialRec& trial) {
            assert(trial.status == IntegrationTrialRec::kPassed
                && "replayTrialSource: only a passed trial replays its source");
            const NameId parent = mb.nameMap.parentOf(trial.trialScope);
            assert(parent != 0 && "replayTrialSource: a trial scope has a parent");
            const unsigned slot = gl::WorkerContext::coreId() >= 0
                ? static_cast<unsigned>(gl::WorkerContext::coreId()) : scratchArenas().slotCount() - 1;
            ScratchArena& strings = scratchArenas().forSlot(slot);
            ScratchScope frame(strings);
            const StrSpan sourceView = mb.nameMap.decodeView(trial.source);
            const ScratchString source = ScratchString::copyFrom(strings, sourceView.ptr, sourceView.len);
            const StrSpan parentView = mb.nameMap.decodeView(parent);
            const ScratchString parentValidity = ScratchString::copyFrom(strings, parentView.ptr, parentView.len);
            const NameId canonical = rejectedSources.canonicalExistenceId(mb, StrSpan(source), parent);
            const int32_t aggregate = rejectedSources.existenceStateAt(mb, canonical, parent);
            const int32_t available = 2 & ~(aggregate | (aggregate >> 2));
            if (available == 0) return false;
            // The source is known at the parent, or at an ancestor when it
            // reached the parent as a replay of an ancestor's park.
            NameId known = parent;
            int32_t run = 0;
            while (known != 0) {
                run = lookupStatementLevels(mb.intStatementLevelsMap, mb.nameMap, StrSpan(source),
                    mb.nameMap.decodeView(known));
                if (run != 0) break;
                known = mb.nameMap.parentOf(known);
            }
            assert(run != 0 && "replayTrialSource: the source is known with levels at or above the parent");
            const int32_t levelCount = mb.intStatementLevelsMap.runLen(run);
            assert(levelCount <= 256);
            int levels[256];
            for (int32_t i = 0; i < levelCount; ++i) levels[i] = mb.intStatementLevelsMap.valueAt(run, i);
            const StrSpan knownView = mb.nameMap.decodeView(known);
            const ScratchString knownValidity = ScratchString::copyFrom(strings, knownView.ptr, knownView.len);
            queueSourceReplay(mb, StrSpan(source), StrSpan(knownValidity), StrSpan(parentValidity),
                available, trial.flags, trial.generation, levels, levelCount, trial.integrationDepth);
            return true;
        }

// Push a set of constituents (primary + siblings) onto this LB's
        // sameIterationInternalMail for revival at the next hashburst. Used by both the
        // equi-class-driven rewrite path and the new-admission-key revisit
        // path. The compound is NOT re-emitted — it is already in LB memory
        // via normal applyEquivalenceClass deposit (or was already present at
        // the rejection time). Each emitted statement carries its original
        // validity (may be non-main).
        //
        // Origin uses the `equality1` tag (eq-class substitution), so the
        // origin.second[0] slot must be the PRE-rewrite form of each emitted
        // statement — not the compound — to match the verifier's equality1
        // checker. For the direct-revisit path (no rewrite), pre == post.
inline void RejectedSourceReplay::emitIntegrationRevivalToInternalMailIn(
            const std::string& primaryConstituentPre,
            const std::string& primaryConstituentPost,
            const std::vector<std::string>& siblingsPre,
            const std::vector<std::string>& siblingsPost,
            const std::set<int>& levels,
            const std::string& validityName,
            const std::set<std::string>& equalitiesForOrigin,
            Memory& mb) {
            auto emitOne = [&](const std::string& pre, const std::string& post) {
                insertInternalStatement(mb.sameIterationInternalMail, mb.nameMap,
                    StrSpan(post), StrSpan(validityName), levels);

                if (parameters.trackHistory) {
                    // L3: "equality1" antecedents (pre + the equalitiesForOrigin
                    // set, all in-hand std::string) fill a bounded stack
                    // OriginDep[]; validityName is a stable local.
                    OriginDep eqDeps[64];
                    int eqDepN = 0;
                    eqDeps[eqDepN++] = { StrSpan(pre), StrSpan(validityName) };
                    for (const std::string& eq : equalitiesForOrigin) {
                        assert(eqDepN < 64 && "equality1 origin chain exceeds 64");
                        eqDeps[eqDepN++] = { StrSpan(eq), StrSpan(validityName) };
                    }
                    addInternalMailOrigin(mb.sameIterationInternalMail, mb.originInterner,
                              StrSpan(post), StrSpan(validityName),
                              OriginTag::equality1, eqDeps, eqDepN,
                              (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
                }
            };
            emitOne(primaryConstituentPre, primaryConstituentPost);
            const std::size_t n = std::min(siblingsPre.size(), siblingsPost.size());
            for (std::size_t i = 0; i < n; ++i) emitOne(siblingsPre[i], siblingsPost[i]);
        }

/// @brief Span/run door of @ref emitIntegrationRevivalToInternalMailIn —
        ///        revive a rejected cohort onto `sameIterationInternalMail` from
        ///        spans plus an ascending level run, no transient
        ///        `std::string` / `std::set` materialized.
        ///
        /// @details
        /// Exact twin of the string overload: the same `emitOne(pre, post)`
        /// core — the levels-RUN `insertInternalStatement` door followed
        /// (under `trackHistory`) by the span-antecedent
        /// `addInternalMailOrigin` with a bounded stack `OriginDep[64]` filled
        /// `pre` first then the equalities in array order, the same
        /// `maxOrigins` expression — applied to the primary and then the
        /// paired siblings `i in [0, min(siblingsPreN, siblingsPostN))`
        /// (mirroring the string form's `std::min` pairing).
        ///
        /// Per emitted statement this door performs the SAME interner touches
        /// in the SAME order as the string form: `nm.encode(post)`,
        /// `nm.encode(validityName)` (statement door), then under
        /// `trackHistory` `mintOriginKey(oi, post, validityName)` followed by
        /// the positional dep mints inside `encodeOriginSpans`. The string
        /// form already routes through these span doors internally; only its
        /// parameters were heap, so the deposits are byte-identical.
        ///
        /// @pre `[levels, levels + levelCount)` is sorted ascending strictly
        ///      unique — `ColdMail::insertStatement`'s run-door key-byte
        ///      contract; `levelCount == 0` equals the empty set.
        /// @pre `[equalitiesForOrigin, equalitiesForOrigin + equalityN)` is
        ///      sorted ascending unique under `compareSpans` — the string form
        ///      iterates its `std::set<std::string>` ascending.
        /// @param primaryPre          Pre-rewrite form of the primary
        ///                            constituent (origin antecedent slot 0).
        /// @param primaryPost         Post form of the primary constituent
        ///                            (the deposited statement).
        /// @param siblingsPre         Pre-rewrite sibling spans, positional.
        /// @param siblingsPreN        Count of `siblingsPre`.
        /// @param siblingsPost        Post sibling spans, positional.
        /// @param siblingsPostN       Count of `siblingsPost`.
        /// @param levels              Pointer to `levelCount` ascending-unique
        ///                            levels; may be null only when
        ///                            `levelCount` is 0.
        /// @param levelCount          The level count (`>= 0`).
        /// @param validityName        The emitted statements' validity scope.
        /// @param equalitiesForOrigin Extra `equality1` origin antecedents;
        ///                            may be null only when `equalityN` is 0.
        /// @param equalityN           Count of `equalitiesForOrigin`.
        /// @param mb                  The owning LB.
        /// @invariant Span lifetime: the door mints `nameMap` +
        ///            `originInterner`, so caller spans must alias OTHER
        ///            buffers — `valueInterner` `decodeView`s and gen-arena
        ///            blob copies qualify (I-3 different-interner rule).
        /// @see emitIntegrationRevivalToInternalMailIn(const std::string&, const std::string&, const std::vector<std::string>&, const std::vector<std::string>&, const std::set<int>&, const std::string&, const std::set<std::string>&, Memory&)
        ///      — the byte-oracle string overload; insertInternalStatement —
        ///      the levels-run statement door; addInternalMailOrigin — the
        ///      span-antecedent origin door.
inline void RejectedSourceReplay::emitIntegrationRevivalToInternalMailIn(
            StrSpan primaryPre, StrSpan primaryPost,
            const StrSpan* siblingsPre, int32_t siblingsPreN,
            const StrSpan* siblingsPost, int32_t siblingsPostN,
            const int* levels, int32_t levelCount,
            StrSpan validityName,
            const StrSpan* equalitiesForOrigin, int32_t equalityN,
            Memory& mb) {
            auto emitOne = [&](const StrSpan& pre, const StrSpan& post) {
                insertInternalStatement(mb.sameIterationInternalMail, mb.nameMap,
                    post, validityName, levels, levelCount);

                if (parameters.trackHistory) {
                    OriginDep eqDeps[64];
                    int eqDepN = 0;
                    eqDeps[eqDepN++] = { pre, validityName };
                    for (int32_t i = 0; i < equalityN; ++i) {
                        assert(eqDepN < 64 && "equality1 origin chain exceeds 64");
                        eqDeps[eqDepN++] = { equalitiesForOrigin[i], validityName };
                    }
                    addInternalMailOrigin(mb.sameIterationInternalMail, mb.originInterner,
                              post, validityName,
                              OriginTag::equality1, eqDeps, eqDepN,
                              (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
                }
            };
            emitOne(primaryPre, primaryPost);
            const int32_t n = std::min(siblingsPreN, siblingsPostN);
            for (int32_t i = 0; i < n; ++i) emitOne(siblingsPre[i], siblingsPost[i]);
        }
}
