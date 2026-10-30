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
class CaseLedger;
class CaseProcessor;
class ProofHistory;
namespace tests { struct WatchProcessorFixture; }
/// @brief Watch Processor operations for the proof algorithm.
/// @details Uses explicit peer interfaces while the supplied logical blocks retain storage and proof state.
class WatchProcessor {
friend struct tests::WatchProcessorFixture;
public:
/// @brief Bind inputs for watch processor.
/// @details Construction stores explicit references; block storage and all invocation lifetimes remain with their owners.
/// @param canonicalTemplates Borrowed CanonicalTemplateMatcher interface or state; must outlive this component.
/// @param canonicalizer Borrowed Canonicalizer interface or state; must outlive this component.
/// @param ceFilteringActive Borrowed bool interface or state; must outlive this component.
/// @param parameters Borrowed ProverParameters interface or state; must outlive this component.
/// @param caseLedger Borrowed CaseLedger interface or state; must outlive this component.
/// @param caseProcessor Borrowed CaseProcessor interface or state; must outlive this component.
/// @param proofHistory Borrowed ProofHistory interface or state; must outlive this component.
/// @return None; construction binds the supplied references.
WatchProcessor(CanonicalTemplateMatcher& canonicalTemplates, Canonicalizer& canonicalizer, const bool& ceFilteringActive, const ProverParameters& parameters, CaseLedger& caseLedger, CaseProcessor& caseProcessor, ProofHistory& proofHistory);

private:
/// @brief Pack a watch-ledger key.
        ///
        /// @details
        /// Both ledgers key on the scope and the UNSIGNED core name, so a
        /// probe for "any negated `p` at this scope" or "any `p`-shaped goal
        /// at this scope" is one run lookup.
        ///
        /// @param scopeId The validity id.
        /// @param coreId  The core name id (`IntEncodedExpr::nameId`).
        /// @return `packInt32Pair(scopeId, coreId)`.
        /// @see recordWatch; watchRunHasCanonical.
static int64_t watchKey(NameId scopeId, NameId coreId);

private:
/// @brief Whether a row belongs in a watch ledger by shape.
        ///
        /// @details
        /// Equality shapes (their pair + one-sided expansion machinery is
        /// untouched; the `=` core is always multiplied) and anchors (never
        /// rewritten, I-53) stay out of both ledgers.
        ///
        /// @param mb The owning LB (for the interned `=` id).
        /// @param ie The encoded row.
        /// @return `true` for a non-equality, non-anchor row.
        /// @see recordNegationWatchFor; ensureGoalLedgerCurrent.
static bool watchShaped(const Memory& mb, const IntEncodedExpr& ie);

private:
/// @brief Initialize exact-scope watch dependencies from their own ledger.
        /// @details Original and canonical spellings are indexed independently of
        /// expression-registry retention. Watch writers maintain later additions.
        /// @param mb Owning block with resident watch ledger.
        /// @return Nothing.
inline void ensureCanonicalWatchUses(Memory& mb);

private:
/// @brief Extend dependency postings for known statement identities.
        /// @details Known identities survive ordinary cleanup; key synchronization
        /// decodes new rows only and removes disappeared keys after lifecycle edits.
        /// @param mb Owning resident block.
        /// @param arena Caller-owned page-tier scratch.
        /// @return Nothing.
inline void ensureCanonicalStatementUses(Memory& mb, ScratchArena& arena);

public:
/// @brief Compose a canonical form and its proof bridge at a visible scope.
///
/// @details
/// One `canonicalFormAtScope` call reads the maintained complete target-scope
/// bucket and writes the rewritten text on caller-owned string scratch.
/// Applied equalities contribute levels and justifiers at their original
/// citation scopes; the read does not build or merge class state.
/// OR matching opts into equality views; ordinary watch behavior is unchanged.
///
/// @param mb      Owning LB.
/// @param origId  The statement's original id.
/// @param scopeId The scope whose visible classes apply.
/// @param outArena String scratch owning the returned spans.
/// @param out Canonical text and the applied equality bridge.
/// @param lvRun Ascending level buffer, extended by applied equalities.
/// @param lvN Current level count, updated in place.
/// @param lvCap Level-buffer capacity.
/// @param rewriteEqualities Include standalone equality and inequality views.
/// @param collectProofEvidence Include supporting levels and citation scopes.
///        False preserves equality dependency text and locality for lookup only;
///        lvRun must be null, lvN and lvCap zero, and eqScope entries are empty.
/// @return Void.
void canonicalWatchFormAt(Memory& mb, NameId origId, NameId scopeId,
            ScratchArena& outArena, WatchBridge& out, int* lvRun,
            int32_t& lvN, int32_t lvCap, bool rewriteEqualities = false,
            bool collectProofEvidence = true);

private:
/// @brief Return the canonical identifier without constructing proof metadata.
///
/// @details Uses the same scope-visible chooser and substitution walk as proof
/// canonicalization. Only the resulting expression can mint a NameMap entry;
/// supporting levels, equality justifiers and citation scopes are not requested.
/// @param mb Owning block with initialized visible canonical classes.
/// @param origId Minted original expression identifier.
/// @param scopeId Minted effective scope identifier.
/// @return Original identifier or the identifier of its canonical text.
NameId canonicalWatchIdAt(Memory& mb, NameId origId, NameId scopeId);

private:
/// @brief The token a class member is rewritten to by canonical forms at a
///        scope.
/// @details Reads `canonicalMemberRepresentatives`, which the complete-view
/// rebuild writes for every member of every class of the scope's view. No
/// entry (the token is in no class there) and a zero normal representative
/// (every member weak there) both leave the token unchanged, as
/// `canonicalFormAtScope` does.
/// @param mb Owning LB.
/// @param member Minted token id.
/// @param scopeId Minted scope id.
/// @return The replacing token, or @p member when nothing replaces it.
static NameId canonicalRepresentativeAt(const Memory& mb, NameId member,
            NameId scopeId);

private:
/// @brief Whether an expression has one canonical spelling at two scopes.
/// @details Every bracket token (the tokens `canonicalFormAtScope` may
/// replace) must have the same representative at both scopes; a token never
/// interned is in no class. Reads only.
/// @param mb Owning LB.
/// @param origId Minted expression id.
/// @param fromScope First scope.
/// @param toScope Second scope.
/// @return True when the representatives agree for every bracket token.
static bool sameCanonicalSpelling(const Memory& mb, NameId origId,
            NameId fromScope, NameId toScope);

private:
/// @brief Canonical form of an expression at a scope through a pass cache.
/// @details A new `canonicalUses.version` empties the cache. A miss whose
/// representatives agree with @p anchorScope takes the anchor's form;
/// otherwise one text-only `canonicalFormAtScope` call computes it and a
/// changed text is interned in the cache's own table (never the NameMap).
/// @param mb Owning LB.
/// @param cache The pass's cache.
/// @param origId Minted expression id.
/// @param anchorScope The scope whose form is likely shared.
/// @param scopeId The scope whose classes apply.
/// @return 0 when already canonical, else the text id in `cache.texts`.
int32_t canonicalFormCached(Memory& mb, CanonicalFormCache& cache,
            NameId origId, NameId anchorScope, NameId scopeId);

private:
/// @brief Canonical text of an expression at a scope through a pass cache.
/// @details The text (the expression itself when already canonical) is
/// copied onto @p outArena; the span aliases neither the NameMap nor the cache.
/// @param mb Owning LB.
/// @param cache The pass's cache.
/// @param origId Minted expression id.
/// @param anchorScope The scope whose form is likely shared.
/// @param scopeId The scope whose classes apply.
/// @param outArena String-tier arena receiving the copy.
/// @return The canonical text on @p outArena.
StrSpan canonicalTextCached(Memory& mb, CanonicalFormCache& cache,
            NameId origId, NameId anchorScope, NameId scopeId, ScratchArena& outArena);

private:
/// @brief Canonical id of an expression at a scope through a pass cache.
/// @details A changed form is encoded in the NameMap from the cache's own
/// table, so the name and the moment of its first mint match
/// `canonicalWatchIdAt` for the same call sequence.
/// @param mb Owning LB.
/// @param cache The pass's cache.
/// @param origId Minted expression id.
/// @param anchorScope The scope whose form is likely shared.
/// @param scopeId The scope whose classes apply.
/// @return @p origId when nothing applies, else the canonical id.
NameId canonicalIdCached(Memory& mb, CanonicalFormCache& cache,
            NameId origId, NameId anchorScope, NameId scopeId);

public:
/// @brief Registration-door hook: stage a freshly admitted row for
        ///        the next watch drain.
        ///
        /// @details
        /// Called next to `recordNegationWatchFor` at the three registration
        /// seams; both polarities are staged (a positive arrival probes the
        /// negation ledger, a negated one probes the positive rows). Rows
        /// at phase-3-ignored scopes and non-watch shapes are defined
        /// no-ops. Nothing is canonicalized here — the drain does that once,
        /// single-threaded.
        ///
        /// @param mb The owning LB.
        /// @param ie The row just registered (ids minted).
        /// @return (void)
        /// @see materializeWatchHits.
void stageWatchArrival(Memory& mb, const IntEncodedExpr& ie);

public:
/// @brief Class-merge hook: import the ancestor-scope negated rows a
        ///        new class at @p scopeId can rewrite into @p scopeId's
        ///        negation run.
        ///
        /// @details
        /// A class formed at a scope is visible there and below but not at
        /// its ancestors, so an ancestor's negated row that names a class
        /// member has, at @p scopeId, a canonical spelling its own run does
        /// not carry. Every such row (member postings over known statements:
        /// negated, watch-shaped, at a strict ancestor, an argument among
        /// the members) is recorded in the run `(scopeId, core)` with its
        /// canonical id under the classes visible at @p scopeId, and staged
        /// for a re-check. The mirror of `applyEquiClasses`' "descendant
        /// classes rewrite ancestor facts" pass, as ledger entries instead
        /// of statement rows.
        /// Positive statements and negated equalities are excluded before
        /// candidate collection and sorting; eligible sources retain the
        /// same decoded expression/scope order.
        ///
        /// Sources already recorded at @p scopeId are dropped before the
        /// sort. A source whose tokens have the same representatives at its
        /// own scope and at @p scopeId takes its own scope's canonical id
        /// from @p cache, computed once for every target of a refresh.
        ///
        /// @param mb      The owning LB.
        /// @param scopeId The scope whose class bucket just gained a class.
        /// @param members The merged class's member ids.
        /// @param cache   The refresh's canonical forms; `nullptr` uses a
        ///                cache local to this call.
        /// @return (void)
        /// @see importNegationIntoDescendantClassScopes; recanonicalizeWatchLedgers.
void importAncestorNegationsForClass(Memory& mb, NameId scopeId,
            const PagedVector<NameId>& members, CanonicalFormCache* cache = nullptr);

private:
/// @brief Drain-time hook: import a newly registered negated row
        ///        into every descendant scope whose classes can rewrite it.
        ///
        /// @details
        /// The registration-time twin of `importAncestorNegationsForClass`:
        /// the class buckets at strict descendants of the row's scope are
        /// scanned for a member among the row's arguments; a hit records
        /// the row in that scope's run and stages a re-check there. The
        /// canonical id at each such scope comes from @p cache, shared with
        /// the row's own scope when the representatives agree.
        ///
        /// @param mb    The owning LB.
        /// @param ie    The negated row (ids minted, watch-shaped).
        /// @param cache The drain's canonical forms; `nullptr` uses a cache
        ///              local to this call.
        /// @return (void)
        /// @see materializeWatchHits.
void importNegationIntoDescendantClassScopes(Memory& mb,
            const IntEncodedExpr& ie, CanonicalFormCache* cache = nullptr);

public:
/// @brief The registry commit shared by the equivalence-product drain
        ///        and the watch twins: one derived local row, no door.
        ///
        /// @details
        /// Levels row (asserted non-empty and well-shaped), known entry
        /// (local), local-set membership, the three encoded lists and the
        /// negation ledger — byte for byte the block `drainStagedEquiProducts`
        /// committed inline before. Writes no history: the caller records
        /// the row's own origin. A row already carrying a levels row asserts
        /// (the callers probe `known` first).
        ///
        /// @param mb      The owning LB.
        /// @param text    The row's text.
        /// @param scopeId Its scope id (minted).
        /// @param lv      Ascending level run.
        /// @param lvN     Its length (> 0).
        /// @return The committed row (id form).
        /// @see drainStagedEquiProducts; writeWatchTwinCanonical.
IntEncodedExpr commitDerivedLocalRow(Memory& mb, StrSpan text,
            NameId scopeId, const int* lv, int32_t lvN);

public:
/// @brief Materialize a selected canonical fact with its equality proof bridge.
/// @details Known twins and unchanged forms require no new row. A newly
/// materialized twin is local, carries all applied equality levels, and enters
/// the ordinary negation and OR fact notifications. OR callers can include
/// equality and inequality refuters while other callers preserve their spelling.
/// @param mb Owning block and known registries.
/// @param srcOrigId Known source expression.
/// @param srcScopeId Scope where the source is known.
/// @param atScopeId Comparable scope where the twin is required.
/// @param rewriteEqualities Include standalone equality and inequality views.
/// @return True when a new canonical row is materialized.
bool writeWatchTwinCanonical(Memory& mb, NameId srcOrigId,
            NameId srcScopeId, NameId atScopeId, bool rewriteEqualities = false);

private:
/// @brief Materialize a statement's LITERAL spelling from its
        ///        canonical twin — a goal keeps the conjecture's text, a
        ///        branch assumption's negation keeps the scope name's text —
        ///        so the exact probes fire without re-spelling anything.
        ///
        /// @details
        /// Precondition: the literal's canonical form (under the classes
        /// visible at @p scope) is known at @p scope or an ancestor — a
        /// watch hit guarantees it (the peer row or its canonical twin).
        /// The bridge cites that canonical row and, per applied pair, the
        /// MIRROR equality `(=[canonical,member])` (co-registered with every
        /// equality, I-180). Defined no-ops: the literal is already
        /// canonical; the literal text is already known at the scope or an
        /// ancestor.
        ///
        /// @param mb            The owning LB.
        /// @param literalOrigId The literal text's id (minted).
        /// @param scope         The scope the literal row is deposited at.
        /// @return `true` when the literal row was written.
        /// @see writeWatchTwinCanonical; materializeWatchHits.
bool writeLiteralTwin(Memory& mb, NameId literalOrigId, NameId scope);

private:
/// @brief Book a branch conclusion justified by its visible equalities.
///
/// @details
/// Retains the actual source and both canonicalization bridges without adding
/// a target statement. The source and target must have the same canonical form
/// at the live branch. Saved support and combined levels survive branch freeze;
/// ordinary convergence publishes the parent conclusion and exact dependencies.
///
/// @param mb The resident owning logical block.
/// @param targetId The fixed parent conclusion's expression identifier.
/// @param branchId The live OR case scope.
/// @param sourceId The actual supporting expression identifier.
/// @param sourceScope Its known scope, at the case or an ancestor.
/// @return Nothing; duplicate support preserves the first complete record.
/// @invariant No target statement or sibling equality context is created.
/// @see ordisMerge, materializeWatchHits, orBranchGoalsResolved.
void bookOrEqualitySupport(Memory& mb, NameId targetId,
            NameId branchId, NameId sourceId, NameId sourceScope);

public:
/// @brief The per-step watch drain: turn every canonical
        ///        contradiction / goal pair into the exact pair the sweeps
        ///        fire on.
        ///
        /// @details
        /// Runs once per `standardProcessing` after `applyEquiClasses` and
        /// its cleanup and before `dischargeContradiction` /
        /// `dischargeToBeProved`, single-threaded (I-66, I-28). Four
        /// sources of hits: (a) a POSITIVE arrival at S probes the negation
        /// runs at S and every ancestor for `gl::DefinitionExpansion::negate(canonical_S(arrival))`;
        /// (b) a NEGATED arrival (or a re-checked entry) at S first imports
        /// itself into the descendant class scopes, then — when its
        /// positive core is used as a premise, so the positive rows are
        /// multiplied — probes `intKnownStatements` for its canonical
        /// positive at S and every ancestor, else files a batch request;
        /// (c) every goal (few) does the same with its own sign; (d) one
        /// sweep of `intEncodedStatements` serves every batch request (rows
        /// filtered by core, sign and scope before any canonicalization).
        /// Ordinary hits materialize the canonical twin of each side at the
        /// probing scope (`writeWatchTwinCanonical`; goals also
        /// `writeLiteralTwin`). (f) Every live `_ordis_` branch whose
        /// assumption's negation is not known exactly but is canonically
        /// equal (under the classes visible at the branch) to a negation
        /// entry at the branch or above gets that entry's canonical twin
        /// and the literal negation deposited at the branch, so the
        /// existing exact refutation probes (`refutedOrBranchAtOrAbove`,
        /// `ordisMerge`) retire it. Both pending lists are cleared at the
        /// end. Existing ancestor goals are also compared at every live OR
        /// case. These hits call `bookOrEqualitySupport` instead of writing
        /// either twin: actual facts and local equalities enter bookkeeping.
        /// With `skip_eq_classes` there are no classes and the exact
        /// sweeps already see every pair: the lists are just cleared.
        ///
        /// @param mb     The owning LB.
        /// @param coreId The worker slot (scratch arenas).
        /// @return (void)
        /// @invariant Never discharges, never writes another LB; every row
        ///            it writes is a derived local row with a bridge.
        /// @see stageWatchArrival; recanonicalizeWatchLedgers.
void materializeWatchHits(Memory& mb, int coreId);

private:
/// @brief Record a statement in a watch ledger.
        ///
        /// @details
        /// Inserts `packInt32Pair(canonicalWatchIdAt(origId, scopeId),
        /// origId)` into the run of `watchKey(scopeId, coreId)`; a
        /// duplicate insert is a defined no-op (sorted-unique run), so the
        /// registration doors may record without a presence probe.
        ///
        /// @param ledger  `mb.negationLedger` or `mb.goalLedger`.
        /// @param mb      Owning LB.
        /// @param origId  The statement's original id (minted).
        /// @param scopeId Its scope id (minted).
        /// @param coreId  Its unsigned core name id.
        /// @return (void)
        /// @see clearWatch; watchRunHasCanonical.
void recordWatch(WatchLedger& ledger, Memory& mb, NameId origId,
            NameId scopeId, NameId coreId);

private:
/// @brief Record a statement whose canonical id is already known in a watch
///        ledger.
/// @details Inserts `packInt32Pair(canonId, origId)` into the sorted-unique
/// run of `watchKey(scopeId, coreId)` (a duplicate is a no-op) and, for the
/// negation ledger, indexes the original and canonical texts.
/// @param ledger `mb.negationLedger` or `mb.goalLedger`.
/// @param mb Owning LB.
/// @param origId The statement's original id.
/// @param scopeId Its scope id.
/// @param coreId Its unsigned core name id.
/// @param canonId Its canonical id at @p scopeId.
/// @return (void)
void recordWatchCanonical(WatchLedger& ledger, Memory& mb, NameId origId,
            NameId scopeId, NameId coreId, NameId canonId);

private:
/// @brief Whether a watch run holds an exact packed entry.
/// @details Binary search over the ascending, sorted-unique run.
/// @param ledger `mb.negationLedger` or `mb.goalLedger`.
/// @param key `watchKey(scope, core)`.
/// @param value `packInt32Pair(canonicalId, originalId)`.
/// @return True when the run of @p key contains @p value.
static bool watchRunContainsValue(const WatchLedger& ledger, int64_t key,
            int64_t value);

private:
/// @brief Remove a statement's entry from a watch ledger.
        ///
        /// @details
        /// Drops every run value whose original half is @p origId (one at
        /// most: the run is unique per original). The rebuilt run rides the
        /// gen-scratch arena and is written back with `assignSet`.
        ///
        /// @param ledger  `mb.negationLedger` or `mb.goalLedger`.
        /// @param mb      Owning LB.
        /// @param origId  The statement's original id.
        /// @param scopeId Its scope id.
        /// @param coreId  Its unsigned core name id.
        /// @return `true` when an entry was removed; `false` when the run
        ///         held none (a caller that knows the row was recorded
        ///         asserts on the result).
        /// @see recordWatch.
bool clearWatch(WatchLedger& ledger, Memory& mb, NameId origId,
            NameId scopeId, NameId coreId);

private:
/// @brief Probe a watch ledger for an entry with a given canonical
        ///        id.
        ///
        /// @details
        /// Binary search over the run of `watchKey(scopeId, coreId)` on the
        /// packed value (canonical id in the high half), so a hit costs
        /// log(run). Non-minting; a scope or core never interned has no
        /// run.
        ///
        /// @param ledger  `mb.negationLedger` or `mb.goalLedger`.
        /// @param scopeId The scope probed.
        /// @param coreId  The unsigned core name id probed.
        /// @param canonId The canonical id looked for.
        /// @param origOut Receives the matching entry's original id on a
        ///                hit (the first of the contiguous block).
        /// @return `true` on a hit.
        /// @see recordWatch; canonicalWatchIdAt.
static bool watchRunHasCanonical(const WatchLedger& ledger,
            NameId scopeId, NameId coreId, NameId canonId, NameId* origOut);

public:
/// @brief Registration-door hook: record a freshly known negated
        ///        row in `negationLedger`.
        ///
        /// @details
        /// Called right after every `intKnownStatements` upsert of a
        /// statement (`addStatement`, `drainStagedEquiProducts`, the
        /// status-4 fact load); a positive row, an equality shape or an
        /// anchor is a defined no-op, so the callers need no shape probe.
        ///
        /// @param mb The owning LB.
        /// @param ie The row just registered (ids minted).
        /// @return (void)
        /// @invariant After the call `negationLedger` mirrors
        ///            `intKnownStatements` for the watch-shaped negated rows.
        /// @see clearNegationWatchFor.
void recordNegationWatchFor(Memory& mb, const IntEncodedExpr& ie);

public:
/// @brief Registry-erase hook: drop a statement's `negationLedger`
        ///        entry when its known row is erased.
        ///
        /// @details
        /// Called by the two un-know seams (`resetResentExpressionRegistries`,
        /// `resetParkedOrStatementRegistries`) after their
        /// `intKnownStatements` erase with the pre-erase membership. A
        /// statement that was never known had no entry (defined no-op); a
        /// known watch-shaped negated statement MUST have one — a miss
        /// asserts (the ledger would be out of sync with the registry).
        /// Whole-scope teardowns (`wipeSubtree`, discharge) erase the
        /// ledgers by scope themselves.
        ///
        /// @param mb       The owning LB.
        /// @param origId   The statement's original id.
        /// @param valId    Its scope id.
        /// @param wasKnown Whether `intKnownStatements` held the row before
        ///                 the erase.
        /// @return (void)
        /// @see recordNegationWatchFor.
void clearNegationWatchFor(Memory& mb, NameId origId, NameId valId,
            bool wasKnown);

private:
/// @brief Bring the DERIVED `goalLedger` up to date with
        ///        `intToBeProved`.
        ///
        /// @details
        /// The goal ledger is not mirrored at the goal doors: it is rebuilt
        /// from the goal registry whenever the registry's key-set epoch or
        /// its row count moved (every insert bumps the epoch, every erase
        /// changes the count) or a class merge / scope wipe / discharge
        /// marked it stale (`Memory::goalLedgerStale`). Goals are few per
        /// block, so the rebuild is a handful of `canonicalWatchIdAt`
        /// calls, and a test that seeds the registry directly needs no
        /// door. Equality-shaped and anchor goals stay out (`watchShaped`).
        ///
        /// @param mb The owning LB (registry resident).
        /// @return (void)
        /// @invariant After the call every watch-shaped goal row has exactly
        ///            one `goalLedger` entry whose canonical half is
        ///            `canonicalWatchIdAt(original, scope)`.
        /// @see recanonicalizeWatchLedgers; watchRunHasCanonical.
void ensureGoalLedgerCurrent(Memory& mb);

public:
/// @brief Re-derive the canonical halves of the negation ledger's
        ///        entries a class merge at @p scopeId can change, and mark the
        ///        derived goal ledger stale.
        ///
        /// @details
        /// Exact-scope dependency postings select original/canonical token
        /// users once across all cores. Live indexed records map directly to
        /// their ledger run; no full-ledger scan or per-row membership scan is
        /// needed for an exact-scope refresh. Run order and decoded lexical
        /// expression order preserve canonical identifier minting. Each touched
        /// run is rebuilt once and moved entries are staged for a re-check.
        /// Goal dependencies synchronize even when no representative changed.
        ///
        /// @param mb Owning block.
        /// @param scopeId Scope whose class bucket changed.
        /// @param members Members with changed normal representatives, including new members.
        /// @param exactScope True limits work to this scope; false includes descendants.
        /// @return The number of entries whose canonical id changed.
        /// @invariant After the call every entry's canonical half equals
        ///            `canonicalWatchIdAt(original, run scope)`.
        /// @see recanonicalizeProcessedOrLedger; importAncestorNegationsForClass;
        ///      updateEquivalenceClasses.
int32_t recanonicalizeWatchLedgers(Memory& mb, NameId scopeId,
            const PagedVector<NameId>& members, bool exactScope = false);
private:

CanonicalTemplateMatcher& canonicalTemplates;
Canonicalizer& canonicalizer;
const bool& ceFilteringActive;
const ProverParameters& parameters;
CaseLedger& caseLedger;
CaseProcessor& caseProcessor;
ProofHistory& proofHistory;
};
}
#include "../logical_block/case_ledger.hpp"
#include "../logical_block/case_processor.hpp"
#include "../logical_block/proof_history.hpp"
namespace gl {

/// @brief Pack a watch-ledger key.
        ///
        /// @details
        /// Both ledgers key on the scope and the UNSIGNED core name, so a
        /// probe for "any negated `p` at this scope" or "any `p`-shaped goal
        /// at this scope" is one run lookup.
        ///
        /// @param scopeId The validity id.
        /// @param coreId  The core name id (`IntEncodedExpr::nameId`).
        /// @return `packInt32Pair(scopeId, coreId)`.
        /// @see recordWatch; watchRunHasCanonical.
inline int64_t WatchProcessor::watchKey(NameId scopeId, NameId coreId) {
            return packInt32Pair(scopeId, coreId);
        }

/// @brief Whether a row belongs in a watch ledger by shape.
        ///
        /// @details
        /// Equality shapes (their pair + one-sided expansion machinery is
        /// untouched; the `=` core is always multiplied) and anchors (never
        /// rewritten, I-53) stay out of both ledgers.
        ///
        /// @param mb The owning LB (for the interned `=` id).
        /// @param ie The encoded row.
        /// @return `true` for a non-equality, non-anchor row.
        /// @see recordNegationWatchFor; ensureGoalLedgerCurrent.
inline bool WatchProcessor::watchShaped(const Memory& mb, const IntEncodedExpr& ie) {
            return !ie.isAnchor
                && ie.nameId != mb.nameMap.lookup(StrSpan("=", 1));
        }

/// @brief Initialize exact-scope watch dependencies from their own ledger.
        /// @details Original and canonical spellings are indexed independently of
        /// expression-registry retention. Watch writers maintain later additions.
        /// @param mb Owning block with resident watch ledger.
        /// @return Nothing.
inline void WatchProcessor::ensureCanonicalWatchUses(Memory& mb) {
            auto& uses = mb.lbMemory.canonicalUses;
            if (uses.watchesBuilt) return;
            for (int32_t i = 1; i <= mb.negationLedger.count(); ++i) {
                const NameId scope = static_cast<NameId>(uint64_t(mb.negationLedger.decodeKey(i)) >> 32);
                for (int32_t j = 0; j < mb.negationLedger.runLen(i); ++j) {
                    const int64_t entry = mb.negationLedger.valueAt(i,j);
                    const NameId original = static_cast<NameId>(uint64_t(entry));
                    const NameId canonical = static_cast<NameId>(uint64_t(entry) >> 32);
                    const int64_t key = packStatementKey(original,scope);
                    canonicalTemplates.indexCanonicalText(uses.watches,key,scope,mb.nameMap.decodeView(original));
                    canonicalTemplates.indexCanonicalText(uses.watches,key,scope,mb.nameMap.decodeView(canonical));
                }
            }
            uses.watchesBuilt = true;
        }

/// @brief Extend dependency postings for known statement identities.
        /// @details Known identities survive ordinary cleanup; key synchronization
        /// decodes new rows only and removes disappeared keys after lifecycle edits.
        /// @param mb Owning resident block.
        /// @param arena Caller-owned page-tier scratch.
        /// @return Nothing.
inline void WatchProcessor::ensureCanonicalStatementUses(Memory& mb, ScratchArena& arena) {
            auto& index = mb.lbMemory.canonicalUses.statements;
            index.synchronize(mb.intKnownStatements, arena, [&](int64_t key) {
                const StatementKey row = Codec<StatementKey>::decode(key);
                canonicalTemplates.indexCanonicalText(index, key, row.validity, mb.nameMap.decodeView(row.orig));
            });
        }
}
