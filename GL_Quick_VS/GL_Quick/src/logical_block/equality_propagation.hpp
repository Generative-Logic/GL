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
class CaseProcessor;
class ProofHistory;
class StatementMemory;
class WatchProcessor;
namespace tests { struct EqualityPropagationFixture; }
/// @brief Equality Propagation operations for the proof algorithm.
/// @details Uses explicit peer interfaces while the supplied logical blocks retain storage and proof state.
class EqualityPropagation {
friend struct tests::EqualityPropagationFixture;
public:
/// @brief Bind inputs for equality propagation.
/// @details Construction stores explicit references; block storage and all invocation lifetimes remain with their owners.
/// @param admissionPolicy Borrowed AdmissionPolicy interface or state; must outlive this component.
/// @param allMappingsAna Borrowed std::map interface or state; must outlive this component.
/// @param canonicalizer Borrowed Canonicalizer interface or state; must outlive this component.
/// @param ceFilteringActive Borrowed bool interface or state; must outlive this component.
/// @param parameters Borrowed ProverParameters interface or state; must outlive this component.
/// @param caseProcessor Borrowed CaseProcessor interface or state; must outlive this component.
/// @param proofHistory Borrowed ProofHistory interface or state; must outlive this component.
/// @param statementMemory Borrowed StatementMemory interface or state; must outlive this component.
/// @param watchProcessor Borrowed WatchProcessor interface or state; must outlive this component.
/// @return None; construction binds the supplied references.
EqualityPropagation(AdmissionPolicy& admissionPolicy, const std::map<int64_t, std::vector<std::vector<int> > >& allMappingsAna, Canonicalizer& canonicalizer, const bool& ceFilteringActive, const ProverParameters& parameters, CaseProcessor& caseProcessor, ProofHistory& proofHistory, StatementMemory& statementMemory, WatchProcessor& watchProcessor);

private:
/// @brief The flag-5 relay decision over one disintegration's witness
        ///        groups: mark every witness whose existence group is
        ///        UNCOVERED (no admitted witness) while its enclosing group is
        ///        covered — or that belongs to the top compound.
        ///
        /// @details
        /// Implements the "highest not-disintegrated compound" rule of
        /// [D-284]: an uncovered group nested inside another
        /// uncovered group is NOT selected (relaying the enclosing compound
        /// already delivers it), and a covered group is never selected. The
        /// caller dedups selected witnesses by their spawning-compact instance
        /// (the `it_` and `int_` witness of one existence share the group and
        /// the instance).
        ///
        /// @tparam CoveredFn Callable `bool(int32_t sigId)` — whether the
        ///         group with that signature id has an admitted witness.
        /// @param witnessSigId Per-witness group signature id, `n` entries.
        /// @param parentIdx Per-witness index of the enclosing witness in the
        ///        same arrays; `-1` for the top compound.
        /// @param n Witness count.
        /// @param covered The group-coverage predicate.
        /// @param outRelay Out: per-witness selection verdict, `n` entries.
        /// @return Number of selected witnesses.
        /// @see WitnessMetaStore, disintegrateExpr2.
template <typename CoveredFn>
        static int32_t selectRelayWitnesses(const int32_t* witnessSigId,
            const int32_t* parentIdx, int32_t n, CoveredFn covered,
            bool* outRelay);

private:
/// @brief Shared core of the two `filterIterations` overloads —
        ///        judge a statement's special tokens against one class's
        ///        canonical member.
        ///
        /// @details
        /// Selects the same special representative as weak-variable handling.
        /// Every scanned special token that belongs to the class must equal
        /// that representative; unrelated tokens impose no restriction.
        ///
        /// Token membership probes use `NameMap::lookup` — non-minting. A
        /// token that was never interned cannot be a class member (members
        /// are interned by `rebuildIntMirror`), so a lookup miss means
        /// not-a-member: a defined result, not a fallback. Tier-bucket
        /// membership and whole-class membership coincide for scanned
        /// tokens (a token's text fully matches its pattern, so its kind
        /// equals the scan tier), which the assert pins.
        ///
        /// @param scan    The statement's special tokens (cached or direct).
        /// @param eqClass The class to judge against.
        /// @param mb      Owning LB — NameMap + name-kind cache.
        /// @return `false` iff the statement carries a non-canonical member
        ///         token of this class.
        /// @invariant Canonical selection uses class and numeric depth, then
        ///            decoded lexical ties — never interner identifier order
        ///            ([I-84](../../docs/agentic_swdd/30_invariants.md#i-84)).
template <class ClassT>
        inline bool filterIterationsCore(const SpecialTokenScan& scan,
            const ClassT& eqClass,
            Memory& mb);

private:
/// @brief One special-tier token of the per-class member walk.
        ///
        /// @details The per-token rule of `forEachNonCanonicalMemberTokenCore`
        /// for the two scanned sections: a never interned token or a non-member
        /// continues; a member's kind must be the section's tier (assert); a
        /// member other than @p canonicalId goes to @p sink.
        /// @param eqClass     The class judged.
        /// @param mb          Owning LB (name-kind cache).
        /// @param canonicalId The class's canonical member.
        /// @param tier        The scan section's tier.
        /// @param token       Token bytes.
        /// @param tokenId     Its NameMap id, 0 when never interned.
        /// @param sink        Walk sink; a true return stops the walk.
        /// @return True iff the sink asked to stop.
template <class ClassT, class Sink>
        inline bool visitSpecialMemberToken(const ClassT& eqClass, Memory& mb,
            NameId canonicalId, NameKind tier, StrSpan token, NameId tokenId, Sink& sink);

private:
/// @brief One normal-source token of the per-class member walk.
        ///
        /// @details The per-token rule of `forEachNonCanonicalMemberTokenCore`
        /// for bracket tokens: a never interned token or a non-member
        /// continues, a special tier is skipped (its section covered it), and a
        /// normal member other than @p canonicalId goes to @p sink.
        /// @param eqClass     The class judged.
        /// @param mb          Owning LB (name-kind cache).
        /// @param canonicalId The class's canonical member.
        /// @param token       Token bytes.
        /// @param tokenId     Its NameMap id, 0 when never interned.
        /// @param sink        Walk sink; a true return stops the walk.
        /// @return True iff the sink asked to stop.
template <class ClassT, class Sink>
        inline bool visitNormalMemberToken(const ClassT& eqClass, Memory& mb,
            NameId canonicalId, StrSpan token, NameId tokenId, Sink& sink);

private:
/// @brief The ONE per-class member-token walk core — behind the
        ///        registry filter (`filterIterationsCore`, view overload)
        ///        and the canonical door (`canonicalFormAtScope`),
        ///        parameterized by the token source, the canonical member,
        ///        and what happens on a non-canonical member token.
        ///
        /// @details
        /// Walks the statement's tokens against one class in a fixed
        /// order: the `force_lev_*` / `int_<depth>_lev_*` run, the `it_*_lev_*` run, then the
        /// normal-name tokens, each token with its NameMap id as @p runs
        /// supplies it (0 = never interned -> not a member). Per token — the
        /// `classHasMember` continue, the tier assert on the two special runs
        /// (a normal-run token whose kind is a special tier is skipped there:
        /// the special runs already covered it). A token that is a MEMBER but
        /// NOT @p canonicalId invokes @p sink; a truthy sink return stops the
        /// walk immediately (the filter's first-forbidden-token early exit).
        /// @p canonicalId == 0 walks nothing — the caller's "no canonical in
        /// this class" state.
        ///
        /// Two token sources feed it: `ScannedTokenRuns` looks each token up
        /// as it is reached (the registry filter, through the scan overload
        /// below), `ResolvedStatementTokens` carries ids resolved once per
        /// statement (`resolveStatementTokens`). Either way the tokens,
        /// their order and the per-token rule — including the assert's
        /// `kindOf` lazy fills into the deload-enrolled `kindById_` facet — are
        /// the same, so the probe sequence is byte-identical to the
        /// pre-refactor `filterIterationsCore` body (the [I-134] probe-order
        /// contract). The canonical door applies this per-token rule once per
        /// token with the recorded representatives instead of walking every
        /// class (`canonicalFormAtScope`, I-287).
        ///
        /// @param runs        Token source: `forEachInt` / `forEachIt` /
        ///                    `forEachNormal`, each taking a `(StrSpan token,
        ///                    NameId id) -> bool` visitor that stops on true.
        /// @param eqClass     The class to judge against.
        /// @param mb          Owning LB — name-kind cache.
        /// @param canonicalId The class member every other member token
        ///                    rewrites to; 0 = nothing to walk.
        /// @param sink        Callable `(StrSpan token, NameId tokenId,
        ///                    NameId canonicalId) -> bool`; `true` stops.
        /// @return `true` iff the sink stopped the walk (at least one
        ///         non-canonical member token was found AND the sink asked
        ///         to stop there).
        /// @invariant Canonical selection uses class and numeric depth, then
        ///            decoded lexical ties — never interner identifier order
        ///            ([I-84](../../docs/agentic_swdd/30_invariants.md#i-84)).
        /// @see forEachNonCanonicalMemberToken — the registry filter's
        ///      forwarding wrapper; canonicalFormAtScope — the door's
        ///      token-wise twin of this rule; visitSpecialMemberToken /
        ///      visitNormalMemberToken — the per-token rule.
template <class Runs, class ClassT, class Sink>
        inline bool forEachNonCanonicalMemberTokenCore(
            const Runs& runs,
            const ClassT& eqClass,
            Memory& mb,
            const NameId canonicalId,
            Sink&& sink);

private:
/// @brief The member-token walk over a token scan and a normal-name
        ///        source — `forEachNonCanonicalMemberTokenCore` with the
        ///        scan-and-lookup token source.
        ///
        /// @details Builds `ScannedTokenRuns` over @p scan and @p normalSource
        /// and runs the core walk: the int section, the it section, then the
        /// tokens @p normalSource yields (a callable that invokes its argument
        /// once per bracket token of the statement; an empty source walks no
        /// normal tokens), each looked up (non-minting) just before it is
        /// judged.
        ///
        /// @param scan         View over the statement's canonical token blob.
        /// @param eqClass      The class to judge against.
        /// @param mb           Owning LB — NameMap + name-kind cache.
        /// @param canonicalId  The class member every other member token
        ///                     rewrites to; 0 = nothing to walk.
        /// @param normalSource Callable `(Fn) -> void` invoking `Fn(StrSpan
        ///                     token)` per normal-name token of the statement.
        /// @param sink         Callable `(StrSpan token, NameId tokenId,
        ///                     NameId canonicalId) -> bool`; `true` stops.
        /// @return See the core overload.
        /// @see forEachNonCanonicalMemberTokenCore (token-source overload).
template <class ClassT, class NormalSource, class Sink>
        inline bool forEachNonCanonicalMemberTokenCore(
            const SpecialTokenScanView& scan,
            const ClassT& eqClass,
            Memory& mb,
            const NameId canonicalId,
            NormalSource&& normalSource,
            Sink&& sink);

private:
/// @brief The registry filter's walk — the special-tier-only,
        ///        `firstSpecialMemberId`-canonical instance of
        ///        `forEachNonCanonicalMemberTokenCore`.
        ///
        /// @details
        /// Forwards with the class's `firstSpecialMemberId` as the
        /// canonical and an empty normal-name source, so the walk is the
        /// two scanned special sections only — byte-identical probe
        /// sequence to the pre-core `filterIterationsCore` body (the
        /// [I-134] contract): `firstSpecialMemberId` first (a class with
        /// neither special tier walks nothing), then the int section, then
        /// the it section.
        ///
        /// @param scan    View over the statement's canonical token blob.
        /// @param eqClass The class to judge against.
        /// @param mb      Owning LB — NameMap + name-kind cache.
        /// @param sink    Callable `(StrSpan token, NameId tokenId,
        ///                NameId canonicalId) -> bool`; `true` stops the walk.
        /// @return See `forEachNonCanonicalMemberTokenCore`.
        /// @invariant Canonical selection uses class and numeric depth, then
        ///            decoded lexical ties — never interner identifier order
        ///            ([I-84](../../docs/agentic_swdd/30_invariants.md#i-84)).
        /// @see filterIterationsCore — the stop-at-first-hit wrapper.
template <class ClassT, class Sink>
        inline bool forEachNonCanonicalMemberToken(
            const SpecialTokenScanView& scan,
            const ClassT& eqClass,
            Memory& mb,
            Sink&& sink);

private:
/// @brief View overload of `filterIterationsCore` — same judgement
        ///        over a zero-copy `SpecialTokenScanView` cursor walk.
        ///
        /// @details
        /// A thin stop-at-first-hit wrapper over the shared
        /// `forEachNonCanonicalMemberToken` core — the judgement, the probe
        /// sequence, and the assert's `kindOf` lazy fills into the
        /// deload-enrolled `kindById_` facet are byte-identical to the heap
        /// form's (the [I-134] probe-order contract): the core walks the int
        /// section first, then the it section, and the always-stop sink
        /// reproduces the early exit on the first forbidden token.
        /// The heap overload stays untouched as the retained-oracle path
        /// (string `filterIterations` wrapper + unit tests).
        ///
        /// @param scan    View over the statement's canonical token blob.
        /// @param eqClass The class to judge against.
        /// @param mb      Owning LB — NameMap + name-kind cache.
        /// @return `false` iff the statement carries a non-canonical member
        ///         token of this class.
        /// @invariant Canonical selection uses class and numeric depth, then
        ///            decoded lexical ties — never interner identifier order
        ///            ([I-84](../../docs/agentic_swdd/30_invariants.md#i-84)).
        /// @see forEachNonCanonicalMemberToken — the shared core.
template <class ClassT>
        inline bool filterIterationsCore(const SpecialTokenScanView& scan,
            const ClassT& eqClass,
            Memory& mb);

public:
/// @brief Class-canonical statement filter, id path — the statement
        ///        is identified by its interned original id and its token
        ///        scan is served from the per-id cache.
        ///
        /// @details One token scan per distinct expression id for the whole
        /// run (lazy, in `EqClassNameCaches::tokensViewOf`) instead of one
        /// per (statement × class) probe — and zero-heap on hit AND miss:
        /// the view core reads the memo blob in place; the miss build rides
        /// the per-slot gen-scratch byte-bump tier, reclaimed by the
        /// `popTo` below. LIFO safety of that pop: callers
        /// (`cleanUpExpressions`) hold their own peeked class views on the
        /// SAME slot arena BELOW this mark — untouched; the core allocates
        /// nothing on the arena, and the view (straddle copy or miss build)
        /// dies before the pop. Callers in the registry sweeps and in
        /// `addStatement` hold the row ids already.
        ///
        /// @param exprOriginalId NameMap id of the statement's original text.
        /// @param eqClass        The class to judge against.
        /// @param mb             Owning LB — NameMap + caches.
        /// @return See filterIterationsCore.
template <class ClassT>
        inline bool filterIterations(const NameId exprOriginalId,
            const ClassT& eqClass,
            Memory& mb);

public:
/// @brief Class-canonical statement filter, string path — for keys
        ///        that are not interned statements (admission / rejected map
        ///        keys, u_-stripped bare forms).
        ///
        /// @details Scans the text directly (uncached) via
        /// `scanSpecialTokens`; identical verdict to the id path on any
        /// input. RETAINED as the production-caller-free unit-test ORACLE
        /// wrapper (the `classifyName(std::string)` precedent): production
        /// rides the id path above; this overload keeps the regex scan +
        /// heap core pinned for the twin tests.
        ///
        /// @param expr2   Full expression text to judge.
        /// @param eqClass The class to judge against.
        /// @param mb      Owning LB — NameMap + caches.
        /// @return See filterIterationsCore.
template <class ClassT>
        inline bool filterIterations(const std::string& expr2,
            const ClassT& eqClass,
            Memory& mb);

private:
// Variable-equalization core shared by the statement batch's closure
        // (`stageEquiClosure`) and the rejected-integration hook. Encapsulates
        // ONLY the inner mapping/substitution loop — everything else
        // (early-exit guards, scope-direction admission, dedup, downstream
        // emission) stays at the call site.
        //
        // `argsExpr` holds the expression's argument spans (arity <= MAX_ARITY);
        // `eqListIds` is the reduced member-id list in memberIds' storage order
        // (decoded-lex, via reduceEqClassIds). Member NAMES are decoded on
        // demand from `nm` (decodeView) — no pre-decoded string vector. Passed
        // in rather than recomputed because a caller may iterate many entries
        // with the SAME class/validity; recomputing per entry would be a 10^5x
        // regression at Gauss scale. Determinism note: the list order is
        // memberIds' storage order — mapping output and thus the proof graph
        // depend on it; never reorder by id (I-84).
template <class ClassT, class Sink>
        inline void enumerateEqClassRewrites(
            const ClassT& clss,
            const StrSpan* argsExpr,
            int32_t argCount,
            const NameId* eqListIds,
            int32_t eqListCount,
            const StrSpan& baseExpr,
            const StrSpan& wrapLeft,
            const StrSpan& wrapRight,
            const NameMap& nm,
            Sink&& sink);

public:
/// @brief Once-per-burst sanitization of `toBeProved` under
        ///        the active equi-classes (called from
        ///        `standardProcessing` after discharge).
        ///
        /// @details Walks `mb.toBeProved`. For each entry, inspects its
        /// special witness arguments against the complete visible class.
        /// A strictly preferred special peer replaces the argument: forced,
        /// integration by numeric depth, then algebraic, with lexical ties.
        /// Ordinary names retain the existing special-only exclusion.
        ///
        /// The entry's `validityName` is preserved verbatim — the
        /// goal's namespace never changes. Only the body args at
        /// `it_/int_` positions get substituted. After the substitution
        /// the old entry is erased and the new key is inserted with
        /// the same value (the auxy set). If the new key already
        /// exists in `toBeProved`, only the erase happens — the two
        /// instances collapse to one.
        ///
        /// The goal-side surface retains its special-name replacement policy.
        /// One call per ordinary `standardProcessing` completion — at the
        /// final EXIT block after all equi-class machinery has
        /// stabilized for the burst.
        ///
        /// Interior (statified): the registry is snapshotted as packed keys
        /// on the per-slot gen-scratch arena and index-sorted by decoded
        /// spans — byte-identical processing order to the retired
        /// `decodeToBeProvedSorted` heap snapshot; the scan phase is
        /// nameMap-mint-free (every `decodeView` span stays valid); each
        /// pending rewrite carries OWNED payloads (string-arena
        /// `ScratchString` original, gen-arena auxy-run copy, packed key by
        /// value) because the apply loop mints. See
        /// `I-141`.
        ///
        /// No origin emission. Goals are exempt from the equality-line
        /// emission rule per CLAUDE.md Rule 16.
        ///
        /// @param mb   Owning LB.
inline void sanitizeToBeProved(Memory& mb);

public:
// For a negated equality !(=[a, b]), expand through the per-arg
        // equivalence classes at the given validity:
        //   * for each c in class(a) \ {a} -> emit !(=[c, b])
        //   * for each d in class(b) \ {b} -> emit !(=[a, d])
        // Each newly-generated expression is committed via the standard local
        // deposit block (intLocalEncodedStatements{,Delta},
        // intLocalEncodedStatements{,Delta}, newStatements), matching the
        // pattern used by addStatement's main branch.
        /// @brief Apply an equivalence class to negated-equality
        /// expressions `!(=[a,b])`, expanding via the one-sided rule
        /// of [I-12](../../docs/agentic_swdd/30_invariants.md#i-12).
        ///
        /// @details
        /// Per I-12, `addStatement` expands `!(=[a,b])` one-sidedly via
        /// equivalence classes; no symmetric cross-product. This
        /// function's analogue applies that rule when an equivalence
        /// class is applied to an existing negated-equality
        /// statement: it picks the appropriate side of the substitution
        /// (the deeper-scoped variable wins) and emits the rewritten
        /// negation, never the symmetric counterpart.
        ///
        /// **Scope symmetry.** The class walk covers all three scope
        /// directions, mirroring `applyEquivalenceClass`'s comparability
        /// rule: classes at the negation's own scope and at strict
        /// ancestors emit at the negation's scope; classes at strict
        /// DESCENDANTS emit at the class's scope
        /// (`deeperOf(negation scope, class scope)`), so an
        /// ancestor-scope negation fully substitutes for a
        /// descendant-scope copy — the precondition of the
        /// ancestor-known dedup contract.
        ///
        /// Each child expansion lives in an arena-backed depth-first frame. The
        /// statement door reports its expansion seam without re-entering this
        /// function; the child finishes before the parent publishes and resumes.
        /// Refuted target scopes are excluded before scanning and publication.
        /// Scratch class views and owned source spans outlive suspended children.
        ///
        /// @param expr Negated equality to expand.
        /// @param memoryBlock Owning block.
        /// @param local Whether resulting statements are local.
        /// @param levels Source supporting levels.
        /// @param levelCount Number of source levels.
        /// @param newStatements Receives emitted local statements.
        /// @param validityName Source validity scope.
        /// @return Nothing.
        /// @invariant [I-12](../../docs/agentic_swdd/30_invariants.md#i-12) —
        ///            one-sided expansion.
inline void applyEquivalenceClassToNegatedEquality(
            StrSpan expr,
            Memory& memoryBlock,
            bool local,
            const int* levels, int32_t levelCount,
            PagedVector<IntEncodedExpr>& newStatements,
            StrSpan validityName);

private:

static bool isSubsetOf(const std::set<std::string>& a, const std::set<std::string>& b);

private:
// tmp := left; overwrite with right's entries (Python dict union: left | right -> right wins)
static void overwriteLevels(std::map<uint32_t, std::set<int> >& left,
            const std::map<uint32_t, std::set<int> >& right);

public:
/// @brief Drop, for every scheduled validity scope in one pass, the
        ///        statement-registry rows at that scope that mention a
        ///        non-canonical member of any equivalence class at that
        ///        scope — the deferred expression-side canonicalization that
        ///        follows `applyEquiClasses` (Step 4b of `standardProcessing`).
        ///
        /// @details
        /// A row is judged only against the classes of its OWN scope, so the
        /// scheduled scopes are independent of each other. Each of the three
        /// registries — `intLocalEncodedStatements`, its delta, and
        /// `intEncodedStatements` — is walked once to group the positions of
        /// the rows at a scheduled scope by that scope's ordinal in
        /// @p scopes, ascending within a group; rows at any other scope pass
        /// through untouched. The scopes are then judged in ordinal order:
        /// first the scope's local rows, then its delta rows, then its
        /// registry rows, each in registry order. A row is kept while unused,
        /// or when `filterIterations` keeps it for every class of its scope
        /// (classes probed in class order until the first drop). This is the
        /// exact `filterIterations` call sequence — and so the token memo's
        /// first-touch order — of one full-registry sweep per scope in
        /// ordinal order, and the kept rows and their order are the same. A
        /// dropped delta row ships its origin history to mailOut before it
        /// goes (D-286), scope by scope and class-major within a scope, the
        /// mailbox mint order of the per-scope sweeps. After the judging the
        /// three lists compact their surviving shared references without
        /// re-interning records. The local packed-key set is re-minted from
        /// the kept rows unconditionally whenever a scheduled scope holds
        /// classes, because that rebuild is also what purges the set's
        /// stale keys after a single-row list erase — and the per-class
        /// waterlines are repaired once with the pre-erase positions of every
        /// dropped registry row. That single repair equals the per-scope
        /// repairs: a waterline moves down by the number of erased positions
        /// below it, and a position taken relative to an intermediate
        /// compaction is offset by exactly the count already erased below it.
        ///
        /// @param mb     Owning LB — registries, classes, NameMap, caches.
        /// @param scopes The distinct scopes to sweep. The ids `1..count()`
        ///               are the sweep order and `lookup` maps a row's scope
        ///               to its ordinal. A scope with no class bucket is a
        ///               defined no-op; a scope excluded from Phase 3 must
        ///               not be scheduled.
        /// @return Nothing.
        /// @invariant `intKnownStatements` and `intStatementLevelsMap` are
        ///            never touched (I-58 / D-93); kept rows keep their order;
        ///            unused rows keep their stored identity without class
        ///            token walks; used rows retain class first-touch order.
        /// @see filterIterations, repairEqClassWaterlines, copyOriginRowsToMailOut,
        ///      standardProcessing (the Step-4b caller).
inline void
            cleanUpExpressions(Memory& mb,
                const ColdHashSet<PodKeyStore<NameId>>& scopes);

public:
/// @brief Mail one positive equality at `main` that the class-canonical
        ///        sweep drops from the outgoing delta — the merge it records
        ///        travels as a statement, not only as history.
        ///
        /// @details
        /// The sweep (`cleanUpExpressions`) drops every outgoing delta row that
        /// names a non-canonical member of a class at its scope, and only the
        /// row's history reaches `mailOut` (D-286). An
        /// equality that merges a member into a class always names that member,
        /// so without this call no merge would ever reach a descendant: the
        /// descendant keeps the aliases apart and never fires an inherited rule
        /// on their mailed rows (I-57). A positive equality at `main` that
        /// passes the statement channel's witness gate (`allowedForMail`: no
        /// ordinary integration witness) is inserted into `mailOut.statements`
        /// with its levels row, and with the from-premise bit when it is a
        /// forced product, as `fillMailOut` would ship it. Any other row is a
        /// defined no-op: its history alone travels, the canonicalization
        /// policy unchanged (D-544).
        ///
        /// @param mb  Owning LB; resident and claimed (phase 3).
        /// @param row The dropped delta row; its levels row exists (I-58).
        /// @return True iff the row was inserted into `mailOut.statements`.
        /// @invariant An equality is never a pending relay or an unconditional
        ///            OR (asserted), so no `fillMailOut` flag is lost.
        /// @see cleanUpExpressions, BlockMailProcessor::fillMailOut,
        ///      BlockMailProcessor::allowedForMail.
inline bool
            shipDroppedMerge(Memory& mb, const IntEncodedExpr& row);

public:
/// @brief Stage the class-rewrite closure of one registry row — every
        ///        product the pairwise fixpoint would register from it, under
        ///        every comparable class, products of products included.
        ///
        /// @details
        /// The row is staged as a `Seed` record; the records appended from it
        /// form a queue the loop walks by index. For each queue item the
        /// candidate classes are the table classes containing one of its
        /// argument ids whose scope is comparable with the item's (same,
        /// strict ancestor, strict descendant — the pairwise three-way
        /// test), once each (the stamp column), in `rank` order. Each
        /// candidate runs `enumerateEqClassRewrites` exactly as the pairwise
        /// loop did (mapping table, strong run, anchor skip); the deposit
        /// scope is the deeper of the class and item scopes. Per rewrite the
        /// pairwise sink's filters run in order: a product already staged at
        /// that scope; known at that scope (dropped, or in `compressor_mode`
        /// kept as `OriginOnly` — beside the row's own `Seed` when the row is
        /// visited too); known at a strict ancestor — in the registry or
        /// among the records staged so far (a product committed earlier in
        /// the pairwise order) — refused with the I-174 signal when the text
        /// is interned; the iteration cap; the `countPatternOccurrences`
        /// cap; an axed variable. A surviving `Full` record joins the queue,
        /// so later classes reach it — the fixpoint confined to this row.
        /// Levels are the item's run ∪ the rewrite's pair levels
        /// (`mergeLevelRuns`); the justifiers are the rewrite's sorted
        /// `(from, to)` pairs as ids. Nothing here mints the NameMap
        /// (asserted), so every `decodeView` span stays valid for the call.
        ///
        /// @param mb         The LB (registry, classes, NameMap — read only,
        ///                   except the refusal signal's pending set).
        /// @param table      The apply's class table.
        /// @param rowIdx     The registry row to close over.
        /// @param staged     The round's product list (appended to).
        /// @param classStamp Per-table-class stamp column (size = class count).
        /// @param stamp      The running stamp; bumped once per queue item.
        /// @param cand       Scratch list of candidate class indices.
        /// @invariant On return every record in `[seed, count)` is a `Seed`,
        ///            `Full` or `OriginOnly` record of this row's closure and
        ///            `(scope, text)` is unique across @p staged.
        /// @see drainStagedEquiProducts, buildEquiClassTable.
void stageEquiClosure(Memory& mb, const EquiClassTable& table, int32_t rowIdx,
                              StagedEquiProducts& staged,
                              PagedVector<int32_t>& classStamp, int32_t& stamp,
                              PagedVector<int32_t>& cand);

public:
/// @brief Convert the round's staged products into the cold containers
        ///        in one step — the commit block of the pairwise
        ///        `applyEquivalenceClass` per `Full` record, its `equality1`
        ///        row per `Full` / `OriginOnly` record, then one `ordisMerge`
        ///        pass over the committed rows.
        ///
        /// @details
        /// Records drain in staging order. A `Full` record mints its text,
        /// asserts it is still unknown at its scope (nothing registers between
        /// staging and drain), writes the levels row, the known + local bits,
        /// the local set and the three registry pushes. Under `trackHistory`
        /// a record without an origin row gets the `equality1` row (I-34):
        /// source = the record it was rewritten from (a registry row or an
        /// earlier product) at that record's scope, then each justifier
        /// `(=[from,to])` cited at `findEqualityCiteScope` from the producing
        /// class's scope. The target scope name is copied to the string tier
        /// before the mints (`encodeExpression` may mint a stripped argument
        /// name, I-3). After every commit the committed rows run `ordisMerge`
        /// in commit order over the whole batch. The first matching deposit
        /// selects any capped OR contradiction witness.
        ///
        /// @param mb        The LB (registry, levels, origins, or bookkeeping written).
        /// @param staged    The round's product list.
        /// @param table     The apply's class table (class scopes for the citations).
        /// @param committed Out: the committed rows in commit order (cleared first).
        /// @param gArena    The apply's gen-scratch arena for the merge window.
        /// @invariant Every `Full` record is unknown at its scope when it commits.
        /// @see stageEquiClosure, `applyEquivalenceClass` (the pairwise oracle).
void drainStagedEquiProducts(Memory& mb, const StagedEquiProducts& staged,
                                     const EquiClassTable& table,
                                     PagedVector<IntEncodedExpr>& committed,
                                     ScratchArena& gArena);

public:
/// @brief Canonicalize an unused deposit and record its equality bridge.
        ///
        /// @details
        /// All substitutions use the completed classes at the row's own scope
        /// and the original argument text. Eligible cores and excluded shapes
        /// return unchanged. The caller owns the returned text and level run.
        ///
        /// @param row Original encoded row used for core eligibility.
        /// @param text Stable original expression text.
        /// @param validity Stable scope text.
        /// @param mb Owning block and current eligibility snapshots.
        /// @param local Input locality, enriched by locally justified rewrites.
        /// @param levels Caller-owned ascending level buffer, capacity 256.
        /// @param levelCount Input/output length of the level buffer.
        /// @param source Incoming producer origin, or absent for an existing row.
        /// @param arena String scratch owning rewritten text and dependencies.
        /// @return Canonical text and the complete simultaneous substitution evidence.
        /// @invariant History is written but never consulted for a proof decision.
        /// @see canonicalFormAtScope, usedAsPremise.
CanonicalForm canonicalizeUnusedDeposit(const IntEncodedExpr& row,
            StrSpan text, StrSpan validity, Memory& mb, bool& local,
            int* levels, int32_t& levelCount, const TransientOrigin& source,
            ScratchArena& arena);
private:
static constexpr int kEqRewriteLevelCap = 256;
AdmissionPolicy& admissionPolicy;
const std::map<int64_t, std::vector<std::vector<int> > >& allMappingsAna;
Canonicalizer& canonicalizer;
const bool& ceFilteringActive;
const ProverParameters& parameters;
CaseProcessor& caseProcessor;
ProofHistory& proofHistory;
StatementMemory& statementMemory;
WatchProcessor& watchProcessor;
};
}
#include "../logical_block/case_processor.hpp"
#include "../logical_block/proof_history.hpp"
#include "../logical_block/statement_memory.hpp"
#include "../logical_block/watch_processor.hpp"
namespace gl {

/// @brief The flag-5 relay decision over one disintegration's witness
        ///        groups: mark every witness whose existence group is
        ///        UNCOVERED (no admitted witness) while its enclosing group is
        ///        covered — or that belongs to the top compound.
        ///
        /// @details
        /// Implements the "highest not-disintegrated compound" rule of
        /// [D-284]: an uncovered group nested inside another
        /// uncovered group is NOT selected (relaying the enclosing compound
        /// already delivers it), and a covered group is never selected. The
        /// caller dedups selected witnesses by their spawning-compact instance
        /// (the `it_` and `int_` witness of one existence share the group and
        /// the instance).
        ///
        /// @tparam CoveredFn Callable `bool(int32_t sigId)` — whether the
        ///         group with that signature id has an admitted witness.
        /// @param witnessSigId Per-witness group signature id, `n` entries.
        /// @param parentIdx Per-witness index of the enclosing witness in the
        ///        same arrays; `-1` for the top compound.
        /// @param n Witness count.
        /// @param covered The group-coverage predicate.
        /// @param outRelay Out: per-witness selection verdict, `n` entries.
        /// @return Number of selected witnesses.
        /// @see WitnessMetaStore, disintegrateExpr2.
template <typename CoveredFn>
inline int32_t EqualityPropagation::selectRelayWitnesses(const int32_t* witnessSigId,
            const int32_t* parentIdx, int32_t n, CoveredFn covered,
            bool* outRelay) {
            int32_t selected = 0;
            for (int32_t i = 0; i < n; ++i) {
                const bool own = covered(witnessSigId[i]);
                const bool parentCovered =
                    parentIdx[i] < 0 || covered(witnessSigId[parentIdx[i]]);
                outRelay[i] = !own && parentCovered;
                if (outRelay[i]) ++selected;
            }
            return selected;
        }

/// @brief Shared core of the two `filterIterations` overloads —
        ///        judge a statement's special tokens against one class's
        ///        canonical member.
        ///
        /// @details
        /// Selects the same special representative as weak-variable handling.
        /// Every scanned special token that belongs to the class must equal
        /// that representative; unrelated tokens impose no restriction.
        ///
        /// Token membership probes use `NameMap::lookup` — non-minting. A
        /// token that was never interned cannot be a class member (members
        /// are interned by `rebuildIntMirror`), so a lookup miss means
        /// not-a-member: a defined result, not a fallback. Tier-bucket
        /// membership and whole-class membership coincide for scanned
        /// tokens (a token's text fully matches its pattern, so its kind
        /// equals the scan tier), which the assert pins.
        ///
        /// @param scan    The statement's special tokens (cached or direct).
        /// @param eqClass The class to judge against.
        /// @param mb      Owning LB — NameMap + name-kind cache.
        /// @return `false` iff the statement carries a non-canonical member
        ///         token of this class.
        /// @invariant Canonical selection uses class and numeric depth, then
        ///            decoded lexical ties — never interner identifier order
        ///            ([I-84](../../docs/agentic_swdd/30_invariants.md#i-84)).
template <class ClassT>
        inline bool EqualityPropagation::filterIterationsCore(const SpecialTokenScan& scan,
            const ClassT& eqClass,
            Memory& mb) {
            const NameId canonicalId = canonicalizer.firstSpecialMemberId(eqClass, mb);
            if (canonicalId == 0) {
                // No special variables in this class -> no filtering needed.
                return true;
            }

            const auto isMember = [&eqClass](const NameId id) {
                return classHasMember(eqClass, id);
            };
            const auto hasForbiddenToken = [&](const std::vector<std::string>& tokens,
                                               const NameKind tier) {
                for (const std::string& token : tokens) {
                    const NameId tokenId = mb.nameMap.lookup(token);
                    if (tokenId == 0) continue;        // never interned -> not a member
                    if (!isMember(tokenId)) continue;
                    assert(mb.eqClassNameCaches.kindOf(tokenId, mb.nameMap) == tier
                        || (tier == NameKind::IntLev && mb.eqClassNameCaches.kindOf(tokenId, mb.nameMap) == NameKind::Forced));
                    if (tokenId != canonicalId) return true;
                }
                return false;
            };
            if (hasForbiddenToken(scan.intTokens, NameKind::IntLev)) return false;
            if (hasForbiddenToken(scan.itTokens, NameKind::ItLev)) return false;
            return true;
        }

/// @brief One special-tier token of the per-class member walk.
        ///
        /// @details The per-token rule of `forEachNonCanonicalMemberTokenCore`
        /// for the two scanned sections: a never interned token or a non-member
        /// continues; a member's kind must be the section's tier (assert); a
        /// member other than @p canonicalId goes to @p sink.
        /// @param eqClass     The class judged.
        /// @param mb          Owning LB (name-kind cache).
        /// @param canonicalId The class's canonical member.
        /// @param tier        The scan section's tier.
        /// @param token       Token bytes.
        /// @param tokenId     Its NameMap id, 0 when never interned.
        /// @param sink        Walk sink; a true return stops the walk.
        /// @return True iff the sink asked to stop.
template <class ClassT, class Sink>
        inline bool EqualityPropagation::visitSpecialMemberToken(const ClassT& eqClass, Memory& mb,
            NameId canonicalId, NameKind tier, StrSpan token, NameId tokenId, Sink& sink) {
            if (tokenId == 0) return false;            // never interned -> not a member
            if (!classHasMember(eqClass, tokenId)) return false;
            assert(mb.eqClassNameCaches.kindOf(tokenId, mb.nameMap) == tier
                        || (tier == NameKind::IntLev && mb.eqClassNameCaches.kindOf(tokenId, mb.nameMap) == NameKind::Forced));
            (void)tier;
            return tokenId != canonicalId && sink(token, tokenId, canonicalId);
        }

/// @brief One normal-source token of the per-class member walk.
        ///
        /// @details The per-token rule of `forEachNonCanonicalMemberTokenCore`
        /// for bracket tokens: a never interned token or a non-member
        /// continues, a special tier is skipped (its section covered it), and a
        /// normal member other than @p canonicalId goes to @p sink.
        /// @param eqClass     The class judged.
        /// @param mb          Owning LB (name-kind cache).
        /// @param canonicalId The class's canonical member.
        /// @param token       Token bytes.
        /// @param tokenId     Its NameMap id, 0 when never interned.
        /// @param sink        Walk sink; a true return stops the walk.
        /// @return True iff the sink asked to stop.
template <class ClassT, class Sink>
        inline bool EqualityPropagation::visitNormalMemberToken(const ClassT& eqClass, Memory& mb,
            NameId canonicalId, StrSpan token, NameId tokenId, Sink& sink) {
            if (tokenId == 0) return false;            // never interned -> not a member
            if (!classHasMember(eqClass, tokenId)) return false;
            if (mb.eqClassNameCaches.kindOf(tokenId, mb.nameMap)
                    != NameKind::Normal) return false; // special tiers walked above
            return tokenId != canonicalId && sink(token, tokenId, canonicalId);
        }

/// @brief The ONE per-class member-token walk core — behind the
        ///        registry filter (`filterIterationsCore`, view overload)
        ///        and the canonical door (`canonicalFormAtScope`),
        ///        parameterized by the token source, the canonical member,
        ///        and what happens on a non-canonical member token.
        ///
        /// @details
        /// Walks the statement's tokens against one class in a fixed
        /// order: the `force_lev_*` / `int_<depth>_lev_*` run, the `it_*_lev_*` run, then the
        /// normal-name tokens, each token with its NameMap id as @p runs
        /// supplies it (0 = never interned -> not a member). Per token — the
        /// `classHasMember` continue, the tier assert on the two special runs
        /// (a normal-run token whose kind is a special tier is skipped there:
        /// the special runs already covered it). A token that is a MEMBER but
        /// NOT @p canonicalId invokes @p sink; a truthy sink return stops the
        /// walk immediately (the filter's first-forbidden-token early exit).
        /// @p canonicalId == 0 walks nothing — the caller's "no canonical in
        /// this class" state.
        ///
        /// Two token sources feed it: `ScannedTokenRuns` looks each token up
        /// as it is reached (the registry filter, through the scan overload
        /// below), `ResolvedStatementTokens` carries ids resolved once per
        /// statement (`resolveStatementTokens`). Either way the tokens,
        /// their order and the per-token rule — including the assert's
        /// `kindOf` lazy fills into the deload-enrolled `kindById_` facet — are
        /// the same, so the probe sequence is byte-identical to the
        /// pre-refactor `filterIterationsCore` body (the [I-134] probe-order
        /// contract). The canonical door applies this per-token rule once per
        /// token with the recorded representatives instead of walking every
        /// class (`canonicalFormAtScope`, I-287).
        ///
        /// @param runs        Token source: `forEachInt` / `forEachIt` /
        ///                    `forEachNormal`, each taking a `(StrSpan token,
        ///                    NameId id) -> bool` visitor that stops on true.
        /// @param eqClass     The class to judge against.
        /// @param mb          Owning LB — name-kind cache.
        /// @param canonicalId The class member every other member token
        ///                    rewrites to; 0 = nothing to walk.
        /// @param sink        Callable `(StrSpan token, NameId tokenId,
        ///                    NameId canonicalId) -> bool`; `true` stops.
        /// @return `true` iff the sink stopped the walk (at least one
        ///         non-canonical member token was found AND the sink asked
        ///         to stop there).
        /// @invariant Canonical selection uses class and numeric depth, then
        ///            decoded lexical ties — never interner identifier order
        ///            ([I-84](../../docs/agentic_swdd/30_invariants.md#i-84)).
        /// @see forEachNonCanonicalMemberToken — the registry filter's
        ///      forwarding wrapper; canonicalFormAtScope — the door's
        ///      token-wise twin of this rule; visitSpecialMemberToken /
        ///      visitNormalMemberToken — the per-token rule.
template <class Runs, class ClassT, class Sink>
        inline bool EqualityPropagation::forEachNonCanonicalMemberTokenCore(
            const Runs& runs,
            const ClassT& eqClass,
            Memory& mb,
            const NameId canonicalId,
            Sink&& sink) {
            if (canonicalId == 0) {
                // No canonical member in this class -> nothing to walk.
                return false;
            }
            if (runs.forEachInt([&](StrSpan token, NameId tokenId) {
                    return visitSpecialMemberToken(eqClass, mb, canonicalId,
                        classifyName(token), token, tokenId, sink);
                })) return true;
            if (runs.forEachIt([&](StrSpan token, NameId tokenId) {
                    return visitSpecialMemberToken(eqClass, mb, canonicalId,
                        NameKind::ItLev, token, tokenId, sink);
                })) return true;
            return runs.forEachNormal([&](StrSpan token, NameId tokenId) {
                return visitNormalMemberToken(eqClass, mb, canonicalId,
                    token, tokenId, sink);
            });
        }

/// @brief The member-token walk over a token scan and a normal-name
        ///        source — `forEachNonCanonicalMemberTokenCore` with the
        ///        scan-and-lookup token source.
        ///
        /// @details Builds `ScannedTokenRuns` over @p scan and @p normalSource
        /// and runs the core walk: the int section, the it section, then the
        /// tokens @p normalSource yields (a callable that invokes its argument
        /// once per bracket token of the statement; an empty source walks no
        /// normal tokens), each looked up (non-minting) just before it is
        /// judged.
        ///
        /// @param scan         View over the statement's canonical token blob.
        /// @param eqClass      The class to judge against.
        /// @param mb           Owning LB — NameMap + name-kind cache.
        /// @param canonicalId  The class member every other member token
        ///                     rewrites to; 0 = nothing to walk.
        /// @param normalSource Callable `(Fn) -> void` invoking `Fn(StrSpan
        ///                     token)` per normal-name token of the statement.
        /// @param sink         Callable `(StrSpan token, NameId tokenId,
        ///                     NameId canonicalId) -> bool`; `true` stops.
        /// @return See the core overload.
        /// @see forEachNonCanonicalMemberTokenCore (token-source overload).
template <class ClassT, class NormalSource, class Sink>
        inline bool EqualityPropagation::forEachNonCanonicalMemberTokenCore(
            const SpecialTokenScanView& scan,
            const ClassT& eqClass,
            Memory& mb,
            const NameId canonicalId,
            NormalSource&& normalSource,
            Sink&& sink) {
            const ScannedTokenRuns<std::remove_reference_t<NormalSource>> runs{
                scan, normalSource, mb };
            return forEachNonCanonicalMemberTokenCore(runs, eqClass, mb, canonicalId,
                static_cast<Sink&&>(sink));
        }

/// @brief The registry filter's walk — the special-tier-only,
        ///        `firstSpecialMemberId`-canonical instance of
        ///        `forEachNonCanonicalMemberTokenCore`.
        ///
        /// @details
        /// Forwards with the class's `firstSpecialMemberId` as the
        /// canonical and an empty normal-name source, so the walk is the
        /// two scanned special sections only — byte-identical probe
        /// sequence to the pre-core `filterIterationsCore` body (the
        /// [I-134] contract): `firstSpecialMemberId` first (a class with
        /// neither special tier walks nothing), then the int section, then
        /// the it section.
        ///
        /// @param scan    View over the statement's canonical token blob.
        /// @param eqClass The class to judge against.
        /// @param mb      Owning LB — NameMap + name-kind cache.
        /// @param sink    Callable `(StrSpan token, NameId tokenId,
        ///                NameId canonicalId) -> bool`; `true` stops the walk.
        /// @return See `forEachNonCanonicalMemberTokenCore`.
        /// @invariant Canonical selection uses class and numeric depth, then
        ///            decoded lexical ties — never interner identifier order
        ///            ([I-84](../../docs/agentic_swdd/30_invariants.md#i-84)).
        /// @see filterIterationsCore — the stop-at-first-hit wrapper.
template <class ClassT, class Sink>
        inline bool EqualityPropagation::forEachNonCanonicalMemberToken(
            const SpecialTokenScanView& scan,
            const ClassT& eqClass,
            Memory& mb,
            Sink&& sink) {
            return forEachNonCanonicalMemberTokenCore(
                scan, eqClass, mb, canonicalizer.firstSpecialMemberId(eqClass, mb),
                [](auto&&) {}, static_cast<Sink&&>(sink));
        }

/// @brief View overload of `filterIterationsCore` — same judgement
        ///        over a zero-copy `SpecialTokenScanView` cursor walk.
        ///
        /// @details
        /// A thin stop-at-first-hit wrapper over the shared
        /// `forEachNonCanonicalMemberToken` core — the judgement, the probe
        /// sequence, and the assert's `kindOf` lazy fills into the
        /// deload-enrolled `kindById_` facet are byte-identical to the heap
        /// form's (the [I-134] probe-order contract): the core walks the int
        /// section first, then the it section, and the always-stop sink
        /// reproduces the early exit on the first forbidden token.
        /// The heap overload stays untouched as the retained-oracle path
        /// (string `filterIterations` wrapper + unit tests).
        ///
        /// @param scan    View over the statement's canonical token blob.
        /// @param eqClass The class to judge against.
        /// @param mb      Owning LB — NameMap + name-kind cache.
        /// @return `false` iff the statement carries a non-canonical member
        ///         token of this class.
        /// @invariant Canonical selection uses class and numeric depth, then
        ///            decoded lexical ties — never interner identifier order
        ///            ([I-84](../../docs/agentic_swdd/30_invariants.md#i-84)).
        /// @see forEachNonCanonicalMemberToken — the shared core.
template <class ClassT>
        inline bool EqualityPropagation::filterIterationsCore(const SpecialTokenScanView& scan,
            const ClassT& eqClass,
            Memory& mb) {
            return !forEachNonCanonicalMemberToken(scan, eqClass, mb,
                [](StrSpan, NameId, NameId) { return true; });
        }

/// @brief Class-canonical statement filter, id path — the statement
        ///        is identified by its interned original id and its token
        ///        scan is served from the per-id cache.
        ///
        /// @details One token scan per distinct expression id for the whole
        /// run (lazy, in `EqClassNameCaches::tokensViewOf`) instead of one
        /// per (statement × class) probe — and zero-heap on hit AND miss:
        /// the view core reads the memo blob in place; the miss build rides
        /// the per-slot gen-scratch byte-bump tier, reclaimed by the
        /// `popTo` below. LIFO safety of that pop: callers
        /// (`cleanUpExpressions`) hold their own peeked class views on the
        /// SAME slot arena BELOW this mark — untouched; the core allocates
        /// nothing on the arena, and the view (straddle copy or miss build)
        /// dies before the pop. Callers in the registry sweeps and in
        /// `addStatement` hold the row ids already.
        ///
        /// @param exprOriginalId NameMap id of the statement's original text.
        /// @param eqClass        The class to judge against.
        /// @param mb             Owning LB — NameMap + caches.
        /// @return See filterIterationsCore.
template <class ClassT>
        inline bool EqualityPropagation::filterIterations(const NameId exprOriginalId,
            const ClassT& eqClass,
            Memory& mb) {
            const unsigned slot = (gl::WorkerContext::coreId() >= 0)
                ? static_cast<unsigned>(gl::WorkerContext::coreId())
                : genScratchArenas().slotCount() - 1;
            ScratchArena& gArena = genScratchArenas().forSlot(slot);
            const ArenaOffset mark = gArena.cursor();
            const bool verdict = filterIterationsCore(
                mb.eqClassNameCaches.tokensViewOf(exprOriginalId, mb.nameMap,
                                                  gArena),
                eqClass, mb);
            gArena.popTo(mark);
            return verdict;
        }

/// @brief Class-canonical statement filter, string path — for keys
        ///        that are not interned statements (admission / rejected map
        ///        keys, u_-stripped bare forms).
        ///
        /// @details Scans the text directly (uncached) via
        /// `scanSpecialTokens`; identical verdict to the id path on any
        /// input. RETAINED as the production-caller-free unit-test ORACLE
        /// wrapper (the `classifyName(std::string)` precedent): production
        /// rides the id path above; this overload keeps the regex scan +
        /// heap core pinned for the twin tests.
        ///
        /// @param expr2   Full expression text to judge.
        /// @param eqClass The class to judge against.
        /// @param mb      Owning LB — NameMap + caches.
        /// @return See filterIterationsCore.
template <class ClassT>
        inline bool EqualityPropagation::filterIterations(const std::string& expr2,
            const ClassT& eqClass,
            Memory& mb) {
            const SpecialTokenScan scan = scanSpecialTokens(expr2);
            return filterIterationsCore(scan, eqClass, mb);
        }

// Variable-equalization core shared by the statement batch's closure
        // (`stageEquiClosure`) and the rejected-integration hook. Encapsulates
        // ONLY the inner mapping/substitution loop — everything else
        // (early-exit guards, scope-direction admission, dedup, downstream
        // emission) stays at the call site.
        //
        // `argsExpr` holds the expression's argument spans (arity <= MAX_ARITY);
        // `eqListIds` is the reduced member-id list in memberIds' storage order
        // (decoded-lex, via reduceEqClassIds). Member NAMES are decoded on
        // demand from `nm` (decodeView) — no pre-decoded string vector. Passed
        // in rather than recomputed because a caller may iterate many entries
        // with the SAME class/validity; recomputing per entry would be a 10^5x
        // regression at Gauss scale. Determinism note: the list order is
        // memberIds' storage order — mapping output and thus the proof graph
        // depend on it; never reorder by id (I-84).
template <class ClassT, class Sink>
        inline void EqualityPropagation::enumerateEqClassRewrites(
            const ClassT& clss,
            const StrSpan* argsExpr,
            int32_t argCount,
            const NameId* eqListIds,
            int32_t eqListCount,
            const StrSpan& baseExpr,
            const StrSpan& wrapLeft,
            const StrSpan& wrapRight,
            const NameMap& nm,
            Sink&& sink) {
            // User-directed: anchor predicates are scope identities — their
            // positional argument slots must never be rewritten by
            // equivalence-class substitution. Skip any rewrite whose base
            // operator is an Anchor.
            if (baseExpr.len >= 6 && std::memcmp(baseExpr.ptr, "Anchor", 6) == 0) return;

            // Member-arg positions + per-arg ids on the stack — arity is
            // bounded by MAX_ARITY, so no heap vectors. A never-interned arg
            // cannot be a member (members are interned by rebuildIntMirror), so
            // a lookup miss is a definitive no.
            assert(argCount <= static_cast<int32_t>(ExecutionParameters::MAX_ARITY));
            int indices[ExecutionParameters::MAX_ARITY];
            NameId argIds[ExecutionParameters::MAX_ARITY];
            int indexCount = 0;
            for (int32_t i = 0; i < argCount; ++i) {
                const NameId argId = nm.lookup(argsExpr[i]);
                argIds[i] = argId;
                if (argId != 0 && classHasMember(clss, argId)) {
                    indices[indexCount++] = i;
                }
            }
            if (indexCount == 0) return;
            if (eqListCount == 0) return;

            // Look up mappings table by the packed (|indices|, |eqList|) key
            // — (int64_t(i) << 32) | uint32_t(j), matching generateAllMappings'
            // insertion packing. Both legacy callers bail on miss; preserve
            // that. Order-preserving only for non-negative halves (Rule-19
            // contract documentation; both counts are positive by the guards
            // above).
            constexpr auto packMappingKey = [](int i, int j) -> int64_t {
                assert(i >= 0 && j >= 0
                    && "packMappingKey: packing requires non-negative halves");
                return packInt32Pair(i, j);
            };
            const int64_t mk = packMappingKey(indexCount,
                                              static_cast<int>(eqListCount));
            std::map<int64_t, std::vector<std::vector<int> > >::const_iterator mit =
                this->allMappingsAna.find(mk);
            if (mit == this->allMappingsAna.end()) return;
            const std::vector<std::vector<int> >& mappings = mit->second;

            // Per-slot string-tier scratch for the rewritten expression. One
            // ScratchScope per mapping iteration rewinds between mappings, so
            // the footprint tracks a single fresh string; the span is read
            // only at the synchronous sink(r) below, never after the rewind.
            const unsigned eqrSlot = (gl::WorkerContext::coreId() >= 0)
                ? static_cast<unsigned>(gl::WorkerContext::coreId())
                : scratchArenas().slotCount() - 1;
            ScratchArena& eqrArena = scratchArenas().forSlot(eqrSlot);

            for (std::size_t m = 0; m < mappings.size(); ++m) {
                const std::vector<int>& mapping = mappings[m];
                ScratchScope eqrScope(eqrArena);

                EqClassRewrite r;
                // Per-position final arg spans on the stack — each points into
                // the caller's stable argsExpr spans or the NameMap's stable
                // member-name storage (decodeView). `fromId` reads argIds
                // directly: each member position is touched at most once per
                // mapping, so its id is still the original before substitution.
                StrSpan tempList[ExecutionParameters::MAX_ARITY];
                for (int32_t i = 0; i < argCount; ++i) {
                    tempList[i] = argsExpr[i];
                    r.argIds[i] = argIds[i];
                }
                r.argCount = argCount;

                // Residual accumulators on the stack (no heap set): extra levels
                // raw-collected then sorted-unique; justifying equalities as
                // (from,to) span pairs then sorted-unique. Both live only for
                // this iteration and are consumed by sink(r).
                int levelBuf[kEqRewriteLevelCap];
                int levelRaw = 0;
                EqClassSpanPair eqBuf[ExecutionParameters::MAX_ARITY];
                int eqRaw = 0;

                for (int i = 0; i < indexCount && i < static_cast<int>(mapping.size()); ++i) {
                    const int idxInArgs = indices[i];
                    const int mappedIdx = mapping[i];
                    assert(mappedIdx >= 0 && mappedIdx < eqListCount);

                    const StrSpan fromVar = tempList[static_cast<std::size_t>(idxInArgs)];
                    const StrSpan toVar = nm.decodeView(eqListIds[static_cast<std::size_t>(mappedIdx)]);
                    const NameId fromId = argIds[static_cast<std::size_t>(idxInArgs)];
                    const NameId toId = eqListIds[static_cast<std::size_t>(mappedIdx)];

                    if (!equalSpans(fromVar, toVar)) {
                        // Real substitution. A substituted slot always holds
                        // a member (the index gate above), so its id is
                        // never 0.
                        assert(fromId != 0);
                        clss.forEachLevel(packEqPairKey(fromId, toId),
                            [&](int lv) {
                                assert(levelRaw < kEqRewriteLevelCap);
                                levelBuf[levelRaw++] = lv;
                            });
                        if (parameters.trackHistory) {
                            assert(eqRaw < ExecutionParameters::MAX_ARITY);
                            eqBuf[eqRaw].from = fromVar;
                            eqBuf[eqRaw].to = toVar;
                            ++eqRaw;
                        }
                    }

                    tempList[static_cast<std::size_t>(idxInArgs)] = toVar;
                    r.argIds[idxInArgs] = toId;
                }

                // Sort-unique the residuals; the sink relies on ascending order
                // (extraLevels linear-merged with the caller's level set; the
                // equalities interned as the equality1 justifying tail —
                // verifier check_equality1 is order-sensitive). The (from,to)
                // sort matches the former std::set<std::string> order because
                // member names carry no comma.
                std::sort(levelBuf, levelBuf + levelRaw);
                r.extraLevels = levelBuf;
                r.extraLevelCount = static_cast<int32_t>(
                    std::unique(levelBuf, levelBuf + levelRaw) - levelBuf);
                std::sort(eqBuf, eqBuf + eqRaw,
                    [](const EqClassSpanPair& a, const EqClassSpanPair& b) {
                        const int c = compareSpans(a.from, b.from);
                        return c != 0 ? c < 0 : compareSpans(a.to, b.to) < 0;
                    });
                r.setEqualities = eqBuf;
                r.setEqualityCount = static_cast<int32_t>(
                    std::unique(eqBuf, eqBuf + eqRaw,
                        [](const EqClassSpanPair& a, const EqClassSpanPair& b) {
                            return equalSpans(a.from, b.from)
                                && equalSpans(a.to, b.to);
                        }) - eqBuf);

                // Build rewritten expression on the string-tier scratch arena,
                // no intermediate joined heap string: wrapLeft + baseExpr + "["
                // + ",".join(tempList) + "]" + wrapRight. Exact-length doctrine
                // (09c §4): compute n in a first pass, allocBytes(n), fill
                // byte-by-byte, wrap, assert(at == n).
                int32_t eqrN = wrapLeft.len + baseExpr.len + 1 /*[*/ + 1 /*]*/
                    + wrapRight.len;
                for (int32_t i = 0; i < argCount; ++i) {
                    if (i > 0) eqrN += 1; // comma
                    eqrN += tempList[i].len;
                }
                char* eqrBuf = eqrArena.allocBytes(eqrN);
                int32_t eqrAt = 0;
                if (wrapLeft.len > 0) {
                    std::memcpy(eqrBuf + eqrAt, wrapLeft.ptr,
                        static_cast<size_t>(wrapLeft.len));
                    eqrAt += wrapLeft.len;
                }
                if (baseExpr.len > 0) {
                    std::memcpy(eqrBuf + eqrAt, baseExpr.ptr,
                        static_cast<size_t>(baseExpr.len));
                    eqrAt += baseExpr.len;
                }
                eqrBuf[eqrAt++] = '[';
                for (int32_t i = 0; i < argCount; ++i) {
                    if (i > 0) eqrBuf[eqrAt++] = ',';
                    if (tempList[i].len > 0) {
                        std::memcpy(eqrBuf + eqrAt, tempList[i].ptr,
                            static_cast<size_t>(tempList[i].len));
                        eqrAt += tempList[i].len;
                    }
                }
                eqrBuf[eqrAt++] = ']';
                if (wrapRight.len > 0) {
                    std::memcpy(eqrBuf + eqrAt, wrapRight.ptr,
                        static_cast<size_t>(wrapRight.len));
                    eqrAt += wrapRight.len;
                }
                assert(eqrAt == eqrN);
                const ScratchString eqrStr = ScratchString::wrap(eqrArena, eqrBuf, eqrN);
                r.rewrittenExpr = StrSpan(eqrStr);

                sink(r);
            }
        }

/// @brief Once-per-burst sanitization of `toBeProved` under
        ///        the active equi-classes (called from
        ///        `standardProcessing` after discharge).
        ///
        /// @details Walks `mb.toBeProved`. For each entry, inspects its
        /// special witness arguments against the complete visible class.
        /// A strictly preferred special peer replaces the argument: forced,
        /// integration by numeric depth, then algebraic, with lexical ties.
        /// Ordinary names retain the existing special-only exclusion.
        ///
        /// The entry's `validityName` is preserved verbatim — the
        /// goal's namespace never changes. Only the body args at
        /// `it_/int_` positions get substituted. After the substitution
        /// the old entry is erased and the new key is inserted with
        /// the same value (the auxy set). If the new key already
        /// exists in `toBeProved`, only the erase happens — the two
        /// instances collapse to one.
        ///
        /// The goal-side surface retains its special-name replacement policy.
        /// One call per ordinary `standardProcessing` completion — at the
        /// final EXIT block after all equi-class machinery has
        /// stabilized for the burst.
        ///
        /// Interior (statified): the registry is snapshotted as packed keys
        /// on the per-slot gen-scratch arena and index-sorted by decoded
        /// spans — byte-identical processing order to the retired
        /// `decodeToBeProvedSorted` heap snapshot; the scan phase is
        /// nameMap-mint-free (every `decodeView` span stays valid); each
        /// pending rewrite carries OWNED payloads (string-arena
        /// `ScratchString` original, gen-arena auxy-run copy, packed key by
        /// value) because the apply loop mints. See
        /// `I-141`.
        ///
        /// No origin emission. Goals are exempt from the equality-line
        /// emission rule per CLAUDE.md Rule 16.
        ///
        /// @param mb   Owning LB.
inline void EqualityPropagation::sanitizeToBeProved(Memory& mb) {
            if (parameters.skip_eq_classes) return;
            if (mb.intToBeProved.empty()) return;

            // Slot + arenas (house idiom; phase-3 finalize worker slot, or
            // the reserved last slot in single-threaded setup).
            const unsigned slot = (gl::WorkerContext::coreId() >= 0)
                ? static_cast<unsigned>(gl::WorkerContext::coreId())
                : genScratchArenas().slotCount() - 1;
            ScratchArena& gArena = genScratchArenas().forSlot(slot);
            ScratchArena& strArena = scratchArenas().forSlot(
                (gl::WorkerContext::coreId() >= 0) ? static_cast<unsigned>(gl::WorkerContext::coreId())
                                       : scratchArenas().slotCount() - 1);
            const ArenaOffset gMark = gArena.cursor();
            ScratchScope strScope(strArena);  // owns every ScratchString below
                                              // until function exit

            /// @brief One staged goal rewrite — OWNED payloads only: the
            ///        rewritten original as string-arena bytes, the packed old
            ///        key by value, the auxy run as a gen-arena byte-bump
            ///        copy. Nothing aliases NameMap cold bytes (the apply
            ///        loop mints).
            struct PendingTbpRec {
                ScratchString newOriginal;
                int64_t oldKey;
                ArenaOffset lvlOff;
                int32_t lvlCount;
            };

            // ---- snapshot (packed keys; the idx sort — compareSpans on the
            //      decoded high half, tie on the decoded low half — is
            //      byte-identical to decodeToBeProvedSorted's (original,
            //      validityName) sort; the sortStatementRows idiom) ----
            DirtyState keysDirty = DirtyState::Clean;
            PagedVector<int64_t> tbpKeys(&gArena, &keysDirty);
            const int32_t n = mb.intToBeProved.count();
            for (int32_t id = 1; id <= n; ++id)
                tbpKeys.push_back(mb.intToBeProved.keyAt(id));
            int32_t* idx = reinterpret_cast<int32_t*>(gArena.resolve(gArena.alloc(
                n * static_cast<int32_t>(sizeof(int32_t)),
                static_cast<int32_t>(alignof(int32_t)))));
            for (int32_t k = 0; k < n; ++k) idx[k] = k;
            std::sort(idx, idx + n, [&](int32_t a, int32_t b) {
                const int64_t ka = tbpKeys[a], kb = tbpKeys[b];
                const int c = compareSpans(
                    mb.nameMap.decodeView(Codec<StatementKey>::decode(ka).orig),
                    mb.nameMap.decodeView(Codec<StatementKey>::decode(kb).orig));
                if (c != 0) return c < 0;
                return compareSpans(
                    mb.nameMap.decodeView(Codec<StatementKey>::decode(ka).validity),
                    mb.nameMap.decodeView(Codec<StatementKey>::decode(kb).validity)) < 0;
            });

            // ---- scan (NON-MINTING w.r.t. nameMap: decodeView reads,
            //      non-minting lookups, strictAncestorSpans (lookup +
            //      decodeView), bestSanitizePeer (non-minting; its kindOf memo
            //      write hits a different container — I-3), getArgsSpans /
            //      classifyName / compareSpans / replaceKeysScratch pure or
            //      scratch-only, intToBeProved reads only. No decodeView span
            //      can dangle during the scan; the pending payloads are the
            //      only state that must survive the minting apply loop, and
            //      they are OWNED copies.) ----
            DirtyState pendDirty = DirtyState::Clean;
            PagedVector<PendingTbpRec> pending(&gArena, &pendDirty);
            for (int32_t k = 0; k < n; ++k) {
                const int64_t key = tbpKeys[idx[k]];
                const NameId origId = Codec<StatementKey>::decode(key).orig;
                const NameId valId = Codec<StatementKey>::decode(key).validity;
                if (mb.phase3IgnoresScope(valId)) continue;
                const StrSpan origSpan = mb.nameMap.decodeView(origId);
                if (origSpan.len == 0) continue;  // twin of row.original.empty()
                const StrSpan valSpan = mb.nameMap.decodeView(valId);

                StrSpan argSpans[ExecutionParameters::MAX_ARITY];
                const int32_t argN = getArgsSpans(
                    origSpan, argSpans, ExecutionParameters::MAX_ARITY);

                StrSpan scopes[ExecutionParameters::MAX_SCOPE_DEPTH];
                scopes[0] = valSpan;
                const int32_t ancN = mb.nameMap.strictAncestorSpans(
                    valSpan, scopes + 1,
                    ExecutionParameters::MAX_SCOPE_DEPTH - 1);
                const int32_t scopeCount = 1 + ancN;

                StrReplacement pairs[ExecutionParameters::MAX_ARITY];
                int32_t pairN = 0;
                for (int32_t ai = 0; ai < argN; ++ai) {
                    const NameKind argKind = classifyName(argSpans[ai]);
                    if (argKind == NameKind::Normal) continue;
                    // Non-minting: a never-interned arg is in no class.
                    const NameId argId = mb.nameMap.lookup(argSpans[ai]);

                    StrSpan bestPeer;
                    bool bestPeerIsInt = false;
                    bestSanitizePeer(mb, argId, scopes, scopeCount, gArena,
                                     bestPeer, bestPeerIsInt);
                    if (bestPeer.len == 0) continue;

                    const bool argRanksAbove = compareWitnessNames(argSpans[ai], bestPeer) <= 0;
                    if (argRanksAbove) continue;

                    addSanitizeSubstPair(pairs, pairN, argSpans[ai], bestPeer,
                                         ExecutionParameters::MAX_ARITY);
                }
                if (pairN == 0) continue;
                sortSanitizeSubstPairs(pairs, pairN);

                const ScratchString newScratch =
                    replaceKeysScratch(strArena, origSpan, pairs, pairN);
                if (equalSpans(StrSpan(newScratch), origSpan)) continue;

                const int32_t rowId = mb.intToBeProved.lookup(key);
                assert(rowId != 0
                       && "sanitizeToBeProved: snapshot row vanished mid-staging");
                PendingTbpRec rec;
                rec.newOriginal = newScratch;
                rec.oldKey = key;
                rec.lvlCount = mb.intToBeProved.runLen(rowId);
                rec.lvlOff = 0;
                if (rec.lvlCount > 0) {
                    rec.lvlOff = gArena.alloc(
                        rec.lvlCount * static_cast<int32_t>(sizeof(int)),
                        static_cast<int32_t>(alignof(int)));
                    int* run = reinterpret_cast<int*>(gArena.resolve(rec.lvlOff));
                    for (int32_t j = 0; j < rec.lvlCount; ++j)
                        run[j] = mb.intToBeProved.valueAt(rowId, j);
                }
                pending.push_back(rec);
            }

            // ---- apply (MINTS: nameMap.encode + registry mutation; every
            //      pending payload is an owned copy, so nothing dangles) ----
            const int32_t pendN = pending.size();
            for (int32_t pi = 0; pi < pendN; ++pi) {
                const PendingTbpRec& p = pending[pi];
                // Re-key the packed registry. The rewritten original may be
                // fresh (write site — encode is allowed); the validity id is
                // reused verbatim from the old key: the goal's namespace
                // never changes (I-45).
                const NameId tbpValId =
                    Codec<StatementKey>::decode(p.oldKey).validity;
                assert(mb.intToBeProved.lookup(p.oldKey) != 0
                       && "sanitizeToBeProved: pending old key missing");
                mb.intToBeProved.eraseSet(p.oldKey);
                const int64_t newPk = packStatementKey(
                    mb.nameMap.encode(StrSpan(p.newOriginal)), tbpValId);
                if (mb.intToBeProved.lookup(newPk) == 0) {
                    const int* rp = (p.lvlCount > 0)
                        ? reinterpret_cast<const int*>(gArena.resolve(p.lvlOff))
                        : nullptr;
                    mb.intToBeProved.assignSetRange(newPk, rp, rp + p.lvlCount);
                    // The re-keyed goal skipped the goal door's full CNFE
                    // scan: the next goal rescan visits every chain again.
                    mb.overallHashMemory.goalScanChainMark = 0;
                }
            }

            gArena.popTo(gMark);  // byte tier: idx + auxy runs + peek straddles;
                                  // the page containers self-free on destruction
        }

// For a negated equality !(=[a, b]), expand through the per-arg
        // equivalence classes at the given validity:
        //   * for each c in class(a) \ {a} -> emit !(=[c, b])
        //   * for each d in class(b) \ {b} -> emit !(=[a, d])
        // Each newly-generated expression is committed via the standard local
        // deposit block (intLocalEncodedStatements{,Delta},
        // intLocalEncodedStatements{,Delta}, newStatements), matching the
        // pattern used by addStatement's main branch.
        /// @brief Apply an equivalence class to negated-equality
        /// expressions `!(=[a,b])`, expanding via the one-sided rule
        /// of [I-12](../../docs/agentic_swdd/30_invariants.md#i-12).
        ///
        /// @details
        /// Per I-12, `addStatement` expands `!(=[a,b])` one-sidedly via
        /// equivalence classes; no symmetric cross-product. This
        /// function's analogue applies that rule when an equivalence
        /// class is applied to an existing negated-equality
        /// statement: it picks the appropriate side of the substitution
        /// (the deeper-scoped variable wins) and emits the rewritten
        /// negation, never the symmetric counterpart.
        ///
        /// **Scope symmetry.** The class walk covers all three scope
        /// directions, mirroring `applyEquivalenceClass`'s comparability
        /// rule: classes at the negation's own scope and at strict
        /// ancestors emit at the negation's scope; classes at strict
        /// DESCENDANTS emit at the class's scope
        /// (`deeperOf(negation scope, class scope)`), so an
        /// ancestor-scope negation fully substitutes for a
        /// descendant-scope copy — the precondition of the
        /// ancestor-known dedup contract.
        ///
        /// Each child expansion lives in an arena-backed depth-first frame. The
        /// statement door reports its expansion seam without re-entering this
        /// function; the child finishes before the parent publishes and resumes.
        /// Refuted target scopes are excluded before scanning and publication.
        /// Scratch class views and owned source spans outlive suspended children.
        ///
        /// @param expr Negated equality to expand.
        /// @param memoryBlock Owning block.
        /// @param local Whether resulting statements are local.
        /// @param levels Source supporting levels.
        /// @param levelCount Number of source levels.
        /// @param newStatements Receives emitted local statements.
        /// @param validityName Source validity scope.
        /// @return Nothing.
        /// @invariant [I-12](../../docs/agentic_swdd/30_invariants.md#i-12) —
        ///            one-sided expansion.
inline void EqualityPropagation::applyEquivalenceClassToNegatedEquality(
            StrSpan expr,
            Memory& memoryBlock,
            bool local,
            const int* levels, int32_t levelCount,
            PagedVector<IntEncodedExpr>& newStatements,
            StrSpan validityName) {
            assert(canonicalizer.isNegatedEquality(expr));
            assert(levelCount >= 0 && levelCount <= 256);
            const NameId sourceScope = memoryBlock.nameMap.lookup(validityName);
            assert(sourceScope != 0);
            if (caseProcessor.refutedOrBranchAtOrAbove(sourceScope, memoryBlock) != 0) return;

            const unsigned stringSlot = gl::WorkerContext::coreId() >= 0
                ? static_cast<unsigned>(gl::WorkerContext::coreId()) : scratchArenas().slotCount() - 1;
            const unsigned frameSlot = gl::WorkerContext::coreId() >= 0
                ? static_cast<unsigned>(gl::WorkerContext::coreId()) : genScratchArenas().slotCount() - 1;
            ScratchArena& strings = scratchArenas().forSlot(stringSlot);
            ScratchArena& arena = genScratchArenas().forSlot(frameSlot);
            ScratchScope stringWindow(strings);
            ScratchScope frameWindow(arena);

            // Page-backed frames coexist with class blobs on the byte-bump
            // tier. Each suspended class view survives all of its children.
            struct Frame {
                StrSpan expression, validity, left, right;
                ScratchArena::Mark stringMark{}, classMark{}, emissionMark{};
                int levels[256]{};
                int32_t levelCount = 0;
                NameId validityId = 0, classScope = 0, depositScope = 0;
                NameId leftId = 0, rightId = 0;
                int32_t scopeBase = 0, scopeCount = 0, ancestorCount = 0;
                int32_t scopeIndex = 0, bucket = 0, classCount = 0, classIndex = 0;
                int32_t side = 0, memberIndex = 0;
                bool descendantsPrepared = false, scopeActive = false;
                bool classLoaded = false, hasLeft = false, hasRight = false;
                bool awaitingChild = false;
                EquivalenceClassView view{};
                StrSpan pendingExpression, pendingScope;
            };
            static_assert(std::is_trivially_copyable<Frame>::value,
                "negated-equality continuations must be plain arena records");
            DirtyState dirty = DirtyState::Clean;
            PagedVector<Frame> frames(&arena, &dirty);
            PagedVector<NameId> scopeStack(&arena, &dirty);
            PagedVector<IntEncodedExpr> discardedOutputs(&arena, &dirty);

            const auto copyText = [&](StrSpan text) {
                const ScratchString copy = ScratchString::copyFrom(strings, text.ptr, text.len);
                return StrSpan(copy);
            };
            const auto pushFrame = [&](StrSpan expression, StrSpan validity,
                                       const int* sourceLevels, int32_t sourceLevelCount) {
                assert(sourceLevelCount >= 0 && sourceLevelCount <= 256);
                Frame frame;
                frame.stringMark = strings.mark();
                frame.expression = copyText(expression);
                frame.validity = copyText(validity);
                StrSpan arguments[ExecutionParameters::MAX_ARITY];
                const int32_t count = getArgsSpans(frame.expression, arguments,
                    ExecutionParameters::MAX_ARITY);
                assert(count == 2);
                frame.left = arguments[0];
                frame.right = arguments[1];
                frame.levelCount = sourceLevelCount;
                for (int32_t i = 0; i < sourceLevelCount; ++i) frame.levels[i] = sourceLevels[i];
                frame.validityId = memoryBlock.nameMap.lookup(frame.validity);
                assert(frame.validityId != 0);
                frame.scopeBase = scopeStack.size();
                scopeStack.push_back(frame.validityId);
                StrSpan ancestors[ExecutionParameters::MAX_SCOPE_DEPTH];
                frame.ancestorCount = memoryBlock.nameMap.strictAncestorSpans(
                    frame.validity, ancestors, ExecutionParameters::MAX_SCOPE_DEPTH);
                for (int32_t i = 0; i < frame.ancestorCount; ++i) {
                    const NameId id = memoryBlock.nameMap.lookup(ancestors[i]);
                    assert(id != 0);
                    scopeStack.push_back(id);
                }
                frame.scopeCount = 1 + frame.ancestorCount;
                frames.push_back(frame);
            };
            const auto buildPair = [&](StrSpan left, StrSpan right, bool negated) {
                const int32_t length = (negated ? 4 : 3) + left.len + 1 + right.len + 2;
                char* bytes = strings.allocBytes(length);
                int32_t at = 0;
                if (negated) bytes[at++] = '!';
                bytes[at++] = '('; bytes[at++] = '='; bytes[at++] = '[';
                std::memcpy(bytes + at, left.ptr, static_cast<std::size_t>(left.len)); at += left.len;
                bytes[at++] = ',';
                std::memcpy(bytes + at, right.ptr, static_cast<std::size_t>(right.len)); at += right.len;
                bytes[at++] = ']'; bytes[at++] = ')';
                assert(at == length);
                const ScratchString pair = ScratchString::wrap(strings, bytes, length);
                return StrSpan(pair);
            };
            const auto publishAfterChild = [&](const Frame& frame, int32_t depth) {
                if (local) {
                    const IntEncodedExpr emitted = encodeExpression(
                        frame.pendingExpression, frame.pendingScope, memoryBlock.nameMap);
                    if (depth == 0) newStatements.push_back(emitted);
                    else discardedOutputs.push_back(emitted);
                }
            };
            pushFrame(expr, validityName, levels, levelCount);
            while (!frames.empty()) {
                const int32_t depth = frames.size() - 1;
                Frame frame = frames[depth];
                if (frame.awaitingChild) {
                    // The child's registration and full closure precede this
                    // publication, exactly as a returned depth-first call does.
                    publishAfterChild(frame, depth);
                    strings.rewind(frame.emissionMark);
                    frame.awaitingChild = false;
                    frames.setAt(depth, frame);
                    continue;
                }
                if (caseProcessor.refutedOrBranchAtOrAbove(frame.validityId, memoryBlock) != 0) {
                    if (frame.classLoaded) arena.rewind(frame.classMark);
                    scopeStack.truncate(frame.scopeBase);
                    frames.truncate(depth);
                    strings.rewind(frame.stringMark);
                    continue;
                }
                if (frame.scopeIndex == frame.scopeCount && !frame.descendantsPrepared) {
                    // Descendants are selected only after self and ancestors
                    // finish; child publications may have changed scope state.
                    ScratchScope orderWindow(arena);
                    DirtyState orderDirty = DirtyState::Clean;
                    PagedVector<NameId> descendants(&arena, &orderDirty);
                    for (int32_t i = 1; i <= memoryBlock.equivalenceClassesMap.count(); ++i) {
                        const NameId id = memoryBlock.equivalenceClassesMap.keyAt(i);
                        if (memoryBlock.nameMap.isStrictAncestor(frame.validityId, id))
                            descendants.push_back(id);
                    }
                    const int32_t count = descendants.size();
                    if (count > 0) {
                        auto* order = reinterpret_cast<int32_t*>(arena.resolve(
                            arena.alloc(count * static_cast<int32_t>(sizeof(int32_t)), alignof(int32_t))));
                        for (int32_t i = 0; i < count; ++i) order[i] = i;
                        std::sort(order, order + count, [&](int32_t a, int32_t b) {
                            return compareSpans(memoryBlock.nameMap.decodeView(descendants[a]),
                                                memoryBlock.nameMap.decodeView(descendants[b])) < 0;
                        });
                        for (int32_t i = 0; i < count; ++i) scopeStack.push_back(descendants[order[i]]);
                    }
                    frame.scopeCount += count;
                    frame.descendantsPrepared = true;
                    frames.setAt(depth, frame);
                    continue;
                }
                if (frame.scopeIndex == frame.scopeCount) {
                    assert(!frame.classLoaded);
                    scopeStack.truncate(frame.scopeBase);
                    frames.truncate(depth);
                    strings.rewind(frame.stringMark);
                    continue;
                }
                const NameId selectedScope = scopeStack[frame.scopeBase + frame.scopeIndex];
                const NameId depositScope = frame.scopeIndex <= frame.ancestorCount
                    ? frame.validityId : selectedScope;
                if (caseProcessor.refutedOrBranchAtOrAbove(depositScope, memoryBlock) != 0) {
                    if (frame.classLoaded) arena.rewind(frame.classMark);
                    frame.classLoaded = false;
                    frame.scopeActive = false;
                    ++frame.scopeIndex;
                    frames.setAt(depth, frame);
                    continue;
                }
                if (!frame.scopeActive) {
                    frame.classScope = selectedScope;
                    frame.depositScope = depositScope;
                    frame.leftId = memoryBlock.nameMap.lookup(frame.left);
                    frame.rightId = memoryBlock.nameMap.lookup(frame.right);
                    frame.bucket = memoryBlock.equivalenceClassesMap.lookup(selectedScope);
                    frame.classCount = frame.bucket == 0 ? 0
                        : memoryBlock.equivalenceClassesMap.runLen(frame.bucket);
                    frame.classIndex = 0;
                    frame.scopeActive = true;
                }
                if (frame.classIndex == frame.classCount) {
                    frame.scopeActive = false;
                    ++frame.scopeIndex;
                    frames.setAt(depth, frame);
                    continue;
                }
                if (!frame.classLoaded) {
                    frame.classMark = arena.mark();
                    int32_t length = 0;
                    const char* bytes = memoryBlock.equivalenceClassesMap.inner().peekBlobContiguous(
                        frame.bucket, frame.classIndex, length, arena);
                    frame.view = EquivalenceClassView{bytes, length};
                    frame.hasLeft = frame.leftId != 0 && classHasMember(frame.view, frame.leftId);
                    frame.hasRight = frame.rightId != 0 && classHasMember(frame.view, frame.rightId);
                    frame.side = 0; frame.memberIndex = 0;
                    frame.classLoaded = true;
                }
                if (frame.side == 0 && (!frame.hasLeft || frame.memberIndex == frame.view.memberCount())) {
                    frame.side = 1; frame.memberIndex = 0;
                }
                if (frame.side == 1 && (!frame.hasRight || frame.memberIndex == frame.view.memberCount())) {
                    arena.rewind(frame.classMark);
                    frame.classLoaded = false;
                    ++frame.classIndex;
                    frames.setAt(depth, frame);
                    continue;
                }
                const NameId oldMember = frame.side == 0 ? frame.leftId : frame.rightId;
                const NameId memberId = frame.view.memberId(frame.memberIndex++);
                frames.setAt(depth, frame);
                if (memberId == oldMember) continue;
                frame.emissionMark = strings.mark();
                const StrSpan member = copyText(memoryBlock.nameMap.decodeView(memberId));
                const StrSpan classScope = copyText(memoryBlock.nameMap.decodeView(frame.classScope));
                const StrSpan targetScope = copyText(memoryBlock.nameMap.decodeView(frame.depositScope));
                const StrSpan product = frame.side == 0
                    ? buildPair(member, frame.right, true) : buildPair(frame.left, member, true);
                const StrSpan equality = buildPair(frame.side == 0 ? frame.left : frame.right, member, false);
                int pairLevels[64]; int32_t pairCount = 0;
                frame.view.forEachLevel(packEqPairKey(oldMember, memberId), [&](int level) {
                    assert(pairCount < 64 && "negated-equality pair run exceeds 64");
                    pairLevels[pairCount++] = level;
                });
                const auto preparePublication = [&]() {
                    if (lookupStatementLevels(memoryBlock.intStatementLevelsMap,
                            memoryBlock.nameMap, product, targetScope) != 0) return false;
                    if (!parameters.compressor_mode) {
                        const NameId original = memoryBlock.nameMap.lookup(product);
                        if (original != 0 && gl::StatementKnowledge::ancestorKnown(memoryBlock, original,
                                frame.depositScope, false)) {
                            caseProcessor.stageOrReleaseForRefusedDeposit(memoryBlock, original, frame.depositScope);
                            return false;
                        }
                    }
                    const int iteration = admissionPolicy.extractMaxIterationNumber(product);
                    if (iteration != -1 && iteration > parameters.maxIterationNumberVariable) return false;
                    if (admissionPolicy.countPatternOccurrences(product, memoryBlock.overallHashMemory, memoryBlock.nameMap)
                            > parameters.maxNumberSecondaryVariables) return false;
                    return caseProcessor.refutedOrBranchAtOrAbove(frame.depositScope, memoryBlock) == 0;
                };
                if (!preparePublication()) {
                    strings.rewind(frame.emissionMark);
                    continue;
                }
                StrSpan citationScope = classScope;
                if (parameters.trackHistory) {
                    const auto originAt = [&](StrSpan scope) {
                        int64_t key = 0;
                        if (!lookupOriginKey(memoryBlock.originInterner, equality, scope, key)) return false;
                        const int32_t row = memoryBlock.exprOriginMap.lookup(key);
                        return row != 0 && memoryBlock.exprOriginMap.runLen(row) > 0;
                    };
                    bool found = originAt(classScope);
                    if (!found && frame.validity.len < classScope.len && originAt(frame.validity)) {
                        found = true; citationScope = frame.validity;
                    }
                    for (int32_t i = frame.ancestorCount - 1; i >= 0 && !found; --i) {
                        const StrSpan ancestor = memoryBlock.nameMap.decodeView(
                            scopeStack[frame.scopeBase + 1 + i]);
                        if (ancestor.len >= classScope.len) continue;
                        if (originAt(ancestor)) { found = true; citationScope = copyText(ancestor); }
                    }
                    assert(found && "negated-equality expansion: justifying equality has no origin row at the class scope or any of its ancestors");
                }
                const OriginDep dependencies[2] = {
                    {frame.expression, frame.validity}, {equality, citationScope}};
                const TransientOrigin origin{true, OriginTag::equality1, dependencies, 2};
                int mergedLevels[256]; int32_t mergedCount = 0;
                for (int32_t i = 0; i < frame.levelCount; ++i)
                    mergedCount = insertLevelSorted(mergedLevels, mergedCount, frame.levels[i], 256);
                for (int32_t i = 0; i < pairCount; ++i)
                    mergedCount = insertLevelSorted(mergedLevels, mergedCount, pairLevels[i], 256);
                discardedOutputs.clear();
                bool expansionReady = false;
                statementMemory.addStatement(product, memoryBlock, local, mergedLevels, mergedCount,
                    origin, targetScope, discardedOutputs, false, nullptr, &expansionReady);
                frame.pendingExpression = product;
                frame.pendingScope = targetScope;
                if (expansionReady) {
                    frame.awaitingChild = true;
                    frames.setAt(depth, frame);
                    pushFrame(product, targetScope, mergedLevels, mergedCount);
                } else {
                    publishAfterChild(frame, depth);
                    strings.rewind(frame.emissionMark);
                }
            }
        }


inline bool EqualityPropagation::isSubsetOf(const std::set<std::string>& a, const std::set<std::string>& b) {
            for (std::set<std::string>::const_iterator it = a.begin(); it != a.end(); ++it) {
                if (b.find(*it) == b.end()) return false;
            }
            return true;
        }

// tmp := left; overwrite with right's entries (Python dict union: left | right -> right wins)
inline void EqualityPropagation::overwriteLevels(std::map<uint32_t, std::set<int> >& left,
            const std::map<uint32_t, std::set<int> >& right) {
            for (std::map<uint32_t, std::set<int> >::const_iterator it = right.begin();
                it != right.end(); ++it) {
                left[it->first] = it->second;
            }
        }

/// @brief Drop, for every scheduled validity scope in one pass, the
        ///        statement-registry rows at that scope that mention a
        ///        non-canonical member of any equivalence class at that
        ///        scope — the deferred expression-side canonicalization that
        ///        follows `applyEquiClasses` (Step 4b of `standardProcessing`).
        ///
        /// @details
        /// A row is judged only against the classes of its OWN scope, so the
        /// scheduled scopes are independent of each other. Each of the three
        /// registries — `intLocalEncodedStatements`, its delta, and
        /// `intEncodedStatements` — is walked once to group the positions of
        /// the rows at a scheduled scope by that scope's ordinal in
        /// @p scopes, ascending within a group; rows at any other scope pass
        /// through untouched. The scopes are then judged in ordinal order:
        /// first the scope's local rows, then its delta rows, then its
        /// registry rows, each in registry order. A row is kept while unused,
        /// or when `filterIterations` keeps it for every class of its scope
        /// (classes probed in class order until the first drop). This is the
        /// exact `filterIterations` call sequence — and so the token memo's
        /// first-touch order — of one full-registry sweep per scope in
        /// ordinal order, and the kept rows and their order are the same. A
        /// dropped delta row ships its origin history to mailOut before it
        /// goes (D-286), scope by scope and class-major within a scope, the
        /// mailbox mint order of the per-scope sweeps; a dropped positive
        /// equality at `main` ships as a statement too
        /// (`shipDroppedMerge`, D-544). After the
        /// judging the
        /// three lists compact their surviving shared references without
        /// re-interning records. The local packed-key set is re-minted from
        /// the kept rows unconditionally whenever a scheduled scope holds
        /// classes, because that rebuild is also what purges the set's
        /// stale keys after a single-row list erase — and the per-class
        /// waterlines are repaired once with the pre-erase positions of every
        /// dropped registry row. That single repair equals the per-scope
        /// repairs: a waterline moves down by the number of erased positions
        /// below it, and a position taken relative to an intermediate
        /// compaction is offset by exactly the count already erased below it.
        ///
        /// @param mb     Owning LB — registries, classes, NameMap, caches.
        /// @param scopes The distinct scopes to sweep. The ids `1..count()`
        ///               are the sweep order and `lookup` maps a row's scope
        ///               to its ordinal. A scope with no class bucket is a
        ///               defined no-op; a scope excluded from Phase 3 must
        ///               not be scheduled.
        /// @return Nothing.
        /// @invariant `intKnownStatements` and `intStatementLevelsMap` are
        ///            never touched (I-58 / D-93); kept rows keep their order;
        ///            unused rows keep their stored identity without class
        ///            token walks; used rows retain class first-touch order.
        /// @see filterIterations, repairEqClassWaterlines, copyOriginRowsToMailOut,
        ///      shipDroppedMerge, standardProcessing (the Step-4b caller).
inline void
            EqualityPropagation::cleanUpExpressions(Memory& mb,
                const ColdHashSet<PodKeyStore<NameId>>& scopes) {
            // The former `newStatements`-pair filter (a cross-subsystem sink whose
            // only caller passed an empty throwaway and discarded the return, per
            // D-93 / I-58) is retired with the L5 newStatements id-form flip — this
            // routine's live work is the intEncodedStatements / level-registry
            // canonicalization below.
            const int32_t scopeCount = scopes.count();

            // Per-call page-tier scratch — the class views, the row groups and
            // the filtered-row rebuilds.
            const unsigned cueSlot = (gl::WorkerContext::coreId() >= 0)
                ? static_cast<unsigned>(gl::WorkerContext::coreId())
                : genScratchArenas().slotCount() - 1;
            ScratchArena& cueArena = genScratchArenas().forSlot(cueSlot);
            DirtyState cueDirty = DirtyState::Clean;

            // Classes read via EquivalenceClassView over the cold blob run — no
            // heap std::vector<EquivalenceClass> decode. The views of every
            // scheduled scope are peeked up front (zero-copy, or a page straddle
            // assembled once onto cueArena's byte-bump tier) and stay valid for
            // the whole call: cleanUpExpressions never mutates
            // equivalenceClassesMap, and filterIterations pops the slot arena
            // only back to its own mark, taken above them. filterIterations is
            // templated on the class type (equi-7b), so it takes a view
            // unchanged. Ordinal k's classes are
            // classViews[classFirst[k] .. classFirst[k] + classCountOf[k]);
            // ordinal 0 stands for a row that passes through.
            PagedVector<EquivalenceClassView> classViews(&cueArena, &cueDirty);
            PagedVector<int32_t> classFirst(&cueArena, &cueDirty);
            PagedVector<int32_t> classCountOf(&cueArena, &cueDirty);
            classFirst.push_back(0);
            classCountOf.push_back(0);
            bool anyClasses = false;
            {
                RT_SCOPE_HERE("CUE_CLASS_VIEWS");
                for (int32_t k = 1; k <= scopeCount; ++k) {
                    const NameId scope = scopes.keyAt(k);
                    assert(!mb.phase3IgnoresScope(scope)
                        && "cleanUpExpressions: a scope excluded from Phase 3 "
                           "was scheduled");
                    const int32_t bucketId =
                        mb.canonicalEquivalenceClassesMap.lookup(scope);
                    const int32_t classCount = (bucketId == 0)
                        ? 0
                        : static_cast<int32_t>(
                              mb.canonicalEquivalenceClassesMap.runLen(bucketId));
                    classFirst.push_back(classViews.size());
                    classCountOf.push_back(classCount);
                    for (int32_t ci = 0; ci < classCount; ++ci) {
                        int32_t clsLen = 0;
                        const char* clsPtr =
                            mb.canonicalEquivalenceClassesMap.inner().peekBlobContiguous(
                                bucketId, ci, clsLen, cueArena);
                        classViews.push_back(EquivalenceClassView{ clsPtr, clsLen });
                    }
                    if (classCount > 0) anyClasses = true;
                }
                RT_NOTE_ITERATIONS_HERE(static_cast<int>(classViews.size()));
            }
            if (!anyClasses) return;

            // Row groups of one registry: `ordinal[i]` is the ordinal of row
            // i's scope when that scope is scheduled and holds classes, else 0
            // (the row passes through); it becomes -1 once the row is dropped.
            // Ordinal k's rows are rows[start[k] .. start[k + 1]), ascending.
            struct RowGroups {
                PagedVector<int32_t> ordinal;
                PagedVector<int32_t> start;
                PagedVector<int32_t> rows;
                RowGroups(ScratchArena* arena, DirtyState* dirty)
                    : ordinal(arena, dirty), start(arena, dirty), rows(arena, dirty) {}
            };
            const auto groupRows = [&](const auto& list,
                                       RowGroups& g) {
                const int32_t n = list.size();
                for (int32_t k = 0; k <= scopeCount + 1; ++k) g.start.push_back(0);
                // Rows of one scope tend to be adjacent: re-probe the scope set
                // only when the scope changes. Every stored validity is
                // interned, so id 0 never matches a row.
                NameId lastScope = 0;
                int32_t lastOrdinal = 0;
                for (int32_t i = 0; i < n; ++i) {
                    const NameId scope = list[i].validityId;
                    if (scope != lastScope) {
                        lastScope = scope;
                        const int32_t k = scopes.lookup(scope);
                        lastOrdinal = (k != 0 && classCountOf[k] > 0) ? k : 0;
                    }
                    g.ordinal.push_back(lastOrdinal);
                    if (lastOrdinal != 0)
                        g.start.setAt(lastOrdinal + 1, g.start[lastOrdinal + 1] + 1);
                }
                for (int32_t k = 2; k <= scopeCount + 1; ++k)
                    g.start.setAt(k, g.start[k] + g.start[k - 1]);
                const int32_t grouped = g.start[scopeCount + 1];
                for (int32_t j = 0; j < grouped; ++j) g.rows.push_back(0);
                PagedVector<int32_t> cursor(&cueArena, &cueDirty);
                for (int32_t k = 0; k <= scopeCount; ++k) cursor.push_back(g.start[k]);
                for (int32_t i = 0; i < n; ++i) {
                    const int32_t k = g.ordinal[i];
                    if (k == 0) continue;
                    const int32_t at = cursor[k];
                    g.rows.setAt(at, i);
                    cursor.setAt(k, at + 1);
                }
            };
            RowGroups localRows(&cueArena, &cueDirty);
            RowGroups deltaRows(&cueArena, &cueDirty);
            RowGroups stmtRows(&cueArena, &cueDirty);
            {
                RT_SCOPE_HERE("CUE_GROUP_ROWS");
                RT_NOTE_ITERATIONS_HERE(static_cast<int>(
                    mb.intLocalEncodedStatements.size()
                    + mb.intLocalEncodedStatementsDelta.size()
                    + mb.intEncodedStatements.size()));
                groupRows(mb.intLocalEncodedStatements, localRows);
                groupRows(mb.intLocalEncodedStatementsDelta, deltaRows);
                groupRows(mb.intEncodedStatements, stmtRows);
            }

            const int originCap = parameters.compressor_mode
                ? parameters.compressor_max_origins_per_expr
                : parameters.max_origin_per_expr;
            // The current scope's dropped delta rows paired with the index of
            // the class that dropped them — their origin history ships
            // class-major (below).
            PagedVector<int32_t> droppedDeltaPos(&cueArena, &cueDirty);
            PagedVector<int32_t> droppedDeltaClass(&cueArena, &cueDirty);

            // RT: one CUE_VALIDITY row per judged scope; `iter` = the classes
            // at that scope, the filters' `iter` = the scope's own rows.
            for (int32_t k = 1; k <= scopeCount; ++k) {
                const int32_t classCount = classCountOf[k];
                if (classCount == 0) continue;
                RT_SCOPE_HERE("CUE_VALIDITY");
                RT_NOTE_ITERATIONS_HERE(classCount);
                const int32_t firstClass = classFirst[k];
                const NameId scope = scopes.keyAt(k);
                // Index of the first class (class order) that drops the
                // statement, or -1 when every class keeps it. A class drops a
                // statement carrying one of its special members other than its
                // first special member (filterIterations). The statement's
                // scanned special tokens are walked once; each token's class
                // and that class's first special member come from the recorded
                // representatives (low half of canonicalMemberRepresentatives,
                // written with the bucket the class views were peeked from), so
                // a class is located only for a forbidden token. Without
                // needIndex the walk stops at the first forbidden token and
                // returns 0. The token scan and the per-token rule (lookup,
                // tier assert) are filterIterations' own.
                const auto droppingClass = [&](const NameId originalId,
                                               const bool needIndex) -> int32_t {
                    const ArenaOffset mark = cueArena.cursor();
                    const SpecialTokenScanView scan = mb.eqClassNameCaches.tokensViewOf(
                        originalId, mb.nameMap, cueArena);
                    int32_t best = -1;
                    const auto judge = [&](SpecialTokenScanView::TokenCursor cursor,
                                           const bool intRun) -> bool {
                        StrSpan token;
                        while (cursor.next(token)) {
                            const NameId tokenId = mb.nameMap.lookup(token);
                            if (tokenId == 0) continue;     // never interned -> not a member
                            const int64_t* recorded =
                                mb.lbMemory.canonicalMemberRepresentatives.find(
                                    packStatementKey(tokenId, scope));
                            if (recorded == nullptr) continue;   // in no class of the view
                            const NameId special = canonicalRecordSpecial(*recorded);
                            if (special == 0) continue;     // its class has no special member
                            const NameKind tier = intRun ? classifyName(token) : NameKind::ItLev;
                            assert(mb.eqClassNameCaches.kindOf(tokenId, mb.nameMap) == tier
                                || (tier == NameKind::IntLev
                                    && mb.eqClassNameCaches.kindOf(tokenId, mb.nameMap) == NameKind::Forced));
                            (void)tier;
                            if (tokenId == special) continue;
                            if (!needIndex) { best = 0; return true; }
                            int32_t ci = 0;
                            while (ci < classCount
                                && !classHasMember(classViews[firstClass + ci], tokenId)) ++ci;
                            assert(ci < classCount
                                && "cleanUpExpressions: a recorded member has no class in the scope's view");
                            if (best < 0 || ci < best) best = ci;
                        }
                        return false;
                    };
                    if (!judge(scan.intTokens(), true)) judge(scan.itTokens(), false);
                    cueArena.popTo(mark);
                    return best;
                };

                // ---- local statement registry ----
                {
                    RT_SCOPE_HERE("CUE_LOCAL_FILTER");
                    RT_NOTE_ITERATIONS_HERE(localRows.start[k + 1] - localRows.start[k]);
                    for (int32_t j = localRows.start[k]; j < localRows.start[k + 1]; ++j) {
                        const int32_t i = localRows.rows[j];
                        const IntEncodedExpr& ie = mb.intLocalEncodedStatements[i];
                        if (usedAsPremise(mb, ie) && droppingClass(ie.originalId, false) >= 0)
                            localRows.ordinal.setAt(i, -1);
                    }
                }

                {
                    RT_SCOPE_HERE("CUE_LOCAL_DELTA_FILTER");
                    RT_NOTE_ITERATIONS_HERE(deltaRows.start[k + 1] - deltaRows.start[k]);
                    droppedDeltaPos.clear();
                    droppedDeltaClass.clear();
                    for (int32_t j = deltaRows.start[k]; j < deltaRows.start[k + 1]; ++j) {
                        const int32_t i = deltaRows.rows[j];
                        const IntEncodedExpr& ie = mb.intLocalEncodedStatementsDelta[i];
                        const int32_t dc = usedAsPremise(mb, ie)
                            ? droppingClass(ie.originalId, true) : -1;
                        if (dc < 0) continue;
                        deltaRows.ordinal.setAt(i, -1);
                        droppedDeltaPos.push_back(i);
                        droppedDeltaClass.push_back(dc);
                    }
                    // A dropped delta row has not been walked by this burst's
                    // fillMailOut yet (this sweep runs earlier in the same phase-3
                    // call), so nothing of it would ever reach mailOut — while
                    // equality1 rewrite rows citing it DO mail. Ship the row's origin
                    // history before the drop (the statement itself stays unmailed —
                    // the canonicalization policy is unchanged) so every receiver can
                    // close the citation chain. The copy is process documentation,
                    // never a proof input (Rule 16 / I-44). See D-286 / I-26.
                    // Shipped class-major — every row the first class drops in row
                    // order, then the second class's, ... — per scope in ordinal
                    // order. decodeView spans are I-3-safe here: the mailOut origin
                    // door mints only the mailbox's private interner, never this
                    // NameMap (the fillMailOut delta loop makes the same argument).
                    // A dropped merge at main ships as a statement as well, or
                    // no descendant would ever learn it
                    // (D-544).
                    for (int32_t ci = 0; ci < classCount; ++ci) {
                        for (int32_t d = 0; d < droppedDeltaPos.size(); ++d) {
                            if (droppedDeltaClass[d] != ci) continue;
                            const IntEncodedExpr& dropped =
                                mb.intLocalEncodedStatementsDelta[droppedDeltaPos[d]];
                            proofHistory.copyOriginRowsToMailOut(mb,
                                mb.nameMap.decodeView(dropped.originalId),
                                mb.nameMap.decodeView(dropped.validityId),
                                originCap);
                            shipDroppedMerge(mb, dropped);
                        }
                    }
                }

                // ---- statement registry ----
                {
                    RT_SCOPE_HERE("CUE_STMTS_FILTER");
                    RT_NOTE_ITERATIONS_HERE(stmtRows.start[k + 1] - stmtRows.start[k]);
                    for (int32_t j = stmtRows.start[k]; j < stmtRows.start[k + 1]; ++j) {
                        const int32_t i = stmtRows.rows[j];
                        const IntEncodedExpr& ie = mb.intEncodedStatements[i];
                        if (usedAsPremise(mb, ie) && droppingClass(ie.originalId, false) >= 0)
                            stmtRows.ordinal.setAt(i, -1);
                    }
                }
            }

            // Dropped locals leave the runtime registries only. Their
            // intStatementLevelsMap rows are RETAINED, exactly like
            // intKnownStatements ([I-58] / [D-93]): the registered bit is
            // immortal and every registration path skips row re-creation
            // while it is set, so an erased row would make the statement
            // re-emittable but never re-registrable — the negated-equality
            // expansion's emit gate reads this row as its permanent
            // already-emitted memory and would regenerate sibling
            // variants without bound.

            // Compact survivor references without copying or re-interning
            // records, then rebuild the parallel set from their packed keys.
            // The set rebuild is unconditional even when
            // nothing was dropped: the single-row erasers of the local list
            // (resetParkedOrStatementRegistries) leave the set untouched, and
            // this rebuild is what purges their stale keys.
            {
                RT_SCOPE_HERE("CUE_LOCAL_REBUILD");
                int32_t localPosition = 0;
                mb.intLocalEncodedStatements.eraseIf([&](const IntEncodedExpr&) {
                    return localRows.ordinal[localPosition++] < 0;
                });
                int32_t deltaPosition = 0;
                mb.intLocalEncodedStatementsDelta.eraseIf([&](const IntEncodedExpr&) {
                    return deltaRows.ordinal[deltaPosition++] < 0;
                });
                mb.intLocalEncodedStatementsSet.resetToFresh();
                for (int32_t j = 0; j < mb.intLocalEncodedStatements.size(); ++j) {
                    const IntEncodedExpr& kept = mb.intLocalEncodedStatements[j];
                    mb.intLocalEncodedStatementsSet.mint(packStatementKey(
                        kept.originalId, kept.validityId));
                }
            }

            // Dropped rows leave intEncodedStatements only. Neither
            // intKnownStatements nor intStatementLevelsMap is touched
            // ([I-58] / [D-93]). The registry record means "already
            // added, do not re-enter the kernel"; the levels row is the
            // emit-side twin of that memory — the negated-equality
            // expansion's emit gate and addEquality's registered-bit
            // skip both assume a registered statement still owns its
            // row. Erasing it leaves a registered-but-rowless statement:
            // re-emittable, never re-registrable — the class expansion
            // then regenerates sibling variants without bound on the
            // emit path, and an equality re-arrival trips the post-loop
            // intStatementLevelsMap assert.
            // Pre-erase positions of the dropped registry rows, ascending —
            // the waterline repair's input.
            PagedVector<int32_t> droppedPos(&cueArena, &cueDirty);
            {
                RT_SCOPE_HERE("CUE_STMTS_REBUILD");
                int32_t statementPosition = 0;
                mb.intEncodedStatements.eraseIf([&](const IntEncodedExpr&) {
                    return stmtRows.ordinal[statementPosition++] < 0;
                }, [&](int32_t erasedPosition) {
                    droppedPos.push_back(erasedPosition);
                });
            }

            // Every dropped row shifts the rows above it down by one, and
            // the persistent per-class waterlines
            // (eqClassSttmntIndexMapMap) point into the compacted list —
            // without the repair a waterline points past the rows that
            // slid under it and the class application silently skips
            // them (the same repair removeExpressionFromMemoryBlock runs
            // for its single-row erase). One repair with the pre-erase
            // positions of every scope's dropped rows equals the former
            // per-scope repairs (see @details).
            if (droppedPos.size() > 0) {
                RT_SCOPE_HERE("CUE_WATERLINE_REPAIR");
                gl::StatementRegistryOperations::repairEqClassWaterlines(mb, droppedPos,
                                        static_cast<int32_t>(droppedPos.size()));
            }
        }

/// @brief Stage the class-rewrite closure of one registry row — every
        ///        product the pairwise fixpoint would register from it, under
        ///        every comparable class, products of products included.
        ///
        /// @details
        /// The row is staged as a `Seed` record; the records appended from it
        /// form a queue the loop walks by index. For each queue item the
        /// candidate classes are the table classes containing one of its
        /// argument ids whose scope is comparable with the item's (same,
        /// strict ancestor, strict descendant — the pairwise three-way
        /// test), once each (the stamp column), in `rank` order. Each
        /// candidate runs `enumerateEqClassRewrites` exactly as the pairwise
        /// loop did (mapping table, strong run, anchor skip); the deposit
        /// scope is the deeper of the class and item scopes. Per rewrite the
        /// pairwise sink's filters run in order: a product already staged at
        /// that scope; known at that scope (dropped, or in `compressor_mode`
        /// kept as `OriginOnly` — beside the row's own `Seed` when the row is
        /// visited too); known at a strict ancestor — in the registry or
        /// among the records staged so far (a product committed earlier in
        /// the pairwise order) — refused with the I-174 signal when the text
        /// is interned; the iteration cap; the `countPatternOccurrences`
        /// cap; an axed variable. A surviving `Full` record joins the queue,
        /// so later classes reach it — the fixpoint confined to this row.
        /// Levels are the item's run ∪ the rewrite's pair levels
        /// (`mergeLevelRuns`); the justifiers are the rewrite's sorted
        /// `(from, to)` pairs as ids. Nothing here mints the NameMap
        /// (asserted), so every `decodeView` span stays valid for the call.
        ///
        /// @param mb         The LB (registry, classes, NameMap — read only,
        ///                   except the refusal signal's pending set).
        /// @param table      The apply's class table.
        /// @param rowIdx     The registry row to close over.
        /// @param staged     The round's product list (appended to).
        /// @param classStamp Per-table-class stamp column (size = class count).
        /// @param stamp      The running stamp; bumped once per queue item.
        /// @param cand       Scratch list of candidate class indices.
        /// @invariant On return every record in `[seed, count)` is a `Seed`,
        ///            `Full` or `OriginOnly` record of this row's closure and
        ///            `(scope, text)` is unique across @p staged.
        /// @see drainStagedEquiProducts, buildEquiClassTable.
inline void EqualityPropagation::stageEquiClosure(Memory& mb, const EquiClassTable& table, int32_t rowIdx,
                              StagedEquiProducts& staged,
                              PagedVector<int32_t>& classStamp, int32_t& stamp,
                              PagedVector<int32_t>& cand) {
            assert(classStamp.size() == table.classCount()
                && "stageEquiClosure: stamp column must cover the table");
            const int32_t nameCountAtEntry = mb.nameMap.nameCount();

            // Seed record: the row's text, scope, level run and argument ids.
            const IntEncodedExpr& row = mb.intEncodedStatements[rowIdx];
            if (mb.phase3IgnoresScope(row.validityId)) return;
            const StrSpan rowText = mb.nameMap.decodeView(row.originalId);
            const int32_t rowLvId = mb.intStatementLevelsMap.lookup(
                packStatementKey(row.originalId, row.validityId));
            assert(rowLvId != 0 && mb.intStatementLevelsMap.runLen(rowLvId) > 0
                && "statement registry / intStatementLevelsMap invariant violated");
            int seedLv[256];
            const int32_t seedLvN = coldIntRunAt(mb.intStatementLevelsMap, rowLvId,
                                                 seedLv, 256);
            NameId seedArgs[ExecutionParameters::MAX_ARITY];
            for (int32_t i = 0; i < row.arity; ++i) seedArgs[i] = row.argId[i];
            assert(!staged.hasSeed(row.validityId, rowText)
                && "stageEquiClosure: a registry row's (scope, text) is seeded twice");
            const int32_t seedRec = staged.stage(row.validityId, rowText,
                StagedEquiProducts::Seed, -1, -1, seedLv, seedLvN,
                nullptr, nullptr, 0, seedArgs, row.arity);

            for (int32_t q = seedRec; q < staged.count(); ++q) {
                if (q != seedRec
                    && staged.recordAt(q).kind != StagedEquiProducts::Full) continue;
                const NameId qVid = staged.recordAt(q).targetVid;
                const StrSpan qText = staged.textAt(q);
                const int32_t qArgN = staged.recordAt(q).argCount;

                // Candidate classes: posted for one of the item's argument
                // ids, comparable with its scope, once each, by rank.
                ++stamp;
                cand.clear();
                for (int32_t i = 0; i < qArgN; ++i) {
                    table.forEachClassContaining(staged.argAt(q, i), [&](int32_t c) {
                        if (classStamp[c] == stamp) return;
                        classStamp.setAt(c, stamp);
                        const EquiClassTable::ClassRow& cr = table.rowAt(c);
                        if (cr.vid == qVid
                            || mb.nameMap.isStrictAncestor(cr.vid, qVid)
                            || mb.nameMap.isStrictAncestor(qVid, cr.vid)) {
                            cand.push_back(c);
                        }
                    });
                }
                for (int32_t i = 1; i < cand.size(); ++i) {
                    for (int32_t j = i; j > 0
                         && table.rowAt(cand[j - 1]).rank > table.rowAt(cand[j]).rank; --j) {
                        const int32_t t = cand[j - 1];
                        cand.setAt(j - 1, cand[j]);
                        cand.setAt(j, t);
                    }
                }
                if (cand.size() == 0) continue;

                int qLv[256];
                const int32_t qLvN = staged.copyLevels(q, qLv, 256);
                StrSpan argSpans[ExecutionParameters::MAX_ARITY];
                const int32_t argCount = getArgsSpans(qText, argSpans,
                    ExecutionParameters::MAX_ARITY);
                assert(argCount == qArgN
                    && "stageEquiClosure: staged argument ids disagree with the text");
                StrSpan baseExpr;
                StrSpan wrapLeft;
                StrSpan wrapRight;
                if (!qText.empty() && qText[0] == '(') {
                    baseExpr = extractExpressionSpan(qText);
                    wrapLeft = StrSpan("(", 1);
                    wrapRight = StrSpan(")", 1);
                } else {
                    assert(qText.len >= 2 && qText[0] == '!' && qText[1] == '('
                        && "stageEquiClosure: statement is neither (…) nor !(…)");
                    baseExpr = extractExpressionFromNegationSpan(qText);
                    wrapLeft = StrSpan("!(", 2);
                    wrapRight = StrSpan(")", 1);
                }

                for (int32_t ci = 0; ci < cand.size(); ++ci) {
                    const int32_t c = cand[ci];
                    const EquiClassTable::ClassRow& cr = table.rowAt(c);
                    const NameId targetVid =
                        (cr.vid == qVid || mb.nameMap.isStrictAncestor(cr.vid, qVid))
                            ? qVid : cr.vid;
                    this->enumerateEqClassRewrites(table.viewAt(c),
                        argSpans, argCount, cr.strong, cr.strongCount,
                        baseExpr, wrapLeft, wrapRight, mb.nameMap,
                        [&](const EqClassRewrite& r) {
                            if (staged.hasProduct(targetVid, r.rewrittenExpr)) return;
                            const NameId prodId = mb.nameMap.lookup(r.rewrittenExpr);
                            const bool known = prodId != 0
                                && mb.intStatementLevelsMap.lookup(
                                       packStatementKey(prodId, targetVid)) != 0;
                            if (!parameters.compressor_mode) {
                                if (known) return;
                                // Ancestor-known refusal (D-273): the registry,
                                // then the products staged so far at a strict
                                // ancestor — committed before this one in the
                                // pairwise order.
                                bool aboveKnown = prodId != 0
                                    && gl::StatementKnowledge::ancestorKnown(mb, prodId, targetVid,
                                                     /*includeSelf=*/false);
                                if (!aboveKnown) {
                                    const int32_t ancN = mb.nameMap.ancLen(targetVid);
                                    for (int32_t k = 0; k < ancN - 1 && !aboveKnown; ++k) {
                                        aboveKnown = staged.contains(
                                            mb.nameMap.ancAt(targetVid, k), r.rewrittenExpr);
                                    }
                                }
                                if (aboveKnown) {
                                    if (prodId != 0) {
                                        caseProcessor.stageOrReleaseForRefusedDeposit(
                                            mb, prodId, targetVid);
                                    }
                                    return;
                                }
                            }
                            const int mn = admissionPolicy.extractMaxIterationNumber(r.rewrittenExpr);
                            if (mn != -1 && mn > parameters.maxIterationNumberVariable) return;
                            if (admissionPolicy.countPatternOccurrences(r.rewrittenExpr,
                                    mb.overallHashMemory, mb.nameMap)
                                > parameters.maxNumberSecondaryVariables) return;
                            for (int32_t k = 0; k < r.argCount; ++k) {
                                if (mb.intAxedVariables.contains(r.argIds[k])) return;
                            }
                            int lv[256];
                            const int32_t lvN = mergeLevelRuns(qLv, qLvN,
                                r.extraLevels, r.extraLevelCount, lv, 256);
                            NameId from[ExecutionParameters::MAX_ARITY];
                            NameId to[ExecutionParameters::MAX_ARITY];
                            for (int32_t k = 0; k < r.setEqualityCount; ++k) {
                                from[k] = mb.nameMap.lookup(r.setEqualities[k].from);
                                to[k] = mb.nameMap.lookup(r.setEqualities[k].to);
                                assert(from[k] != 0 && to[k] != 0
                                    && "stageEquiClosure: justifier names must be interned");
                            }
                            staged.stage(targetVid, r.rewrittenExpr,
                                known ? StagedEquiProducts::OriginOnly
                                      : StagedEquiProducts::Full,
                                q, c, lv, lvN, from, to, r.setEqualityCount,
                                r.argIds, r.argCount);
                        });
                }
            }
            assert(mb.nameMap.nameCount() == nameCountAtEntry
                && "stageEquiClosure: the closure must not mint the NameMap");
        }

/// @brief Convert the round's staged products into the cold containers
        ///        in one step — the commit block of the pairwise
        ///        `applyEquivalenceClass` per `Full` record, its `equality1`
        ///        row per `Full` / `OriginOnly` record, then one `ordisMerge`
        ///        pass over the committed rows.
        ///
        /// @details
        /// Records drain in staging order. A `Full` record mints its text,
        /// asserts it is still unknown at its scope (nothing registers between
        /// staging and drain), writes the levels row, the known + local bits,
        /// the local set and the three registry pushes. Under `trackHistory`
        /// a record without an origin row gets the `equality1` row (I-34):
        /// source = the record it was rewritten from (a registry row or an
        /// earlier product) at that record's scope, then each justifier
        /// `(=[from,to])` cited at `findEqualityCiteScope` from the producing
        /// class's scope. The target scope name is copied to the string tier
        /// before the mints (`encodeExpression` may mint a stripped argument
        /// name, I-3). After every commit the committed rows run `ordisMerge`
        /// in commit order over the whole batch. The first matching deposit
        /// selects any capped OR contradiction witness.
        ///
        /// @param mb        The LB (registry, levels, origins, or bookkeeping written).
        /// @param staged    The round's product list.
        /// @param table     The apply's class table (class scopes for the citations).
        /// @param committed Out: the committed rows in commit order (cleared first).
        /// @param gArena    The apply's gen-scratch arena for the merge window.
        /// @invariant Every `Full` record is unknown at its scope when it commits.
        /// @see stageEquiClosure, `applyEquivalenceClass` (the pairwise oracle).
inline void EqualityPropagation::drainStagedEquiProducts(Memory& mb, const StagedEquiProducts& staged,
                                     const EquiClassTable& table,
                                     PagedVector<IntEncodedExpr>& committed,
                                     ScratchArena& gArena) {
            committed.clear();
            const unsigned strSlot = (gl::WorkerContext::coreId() >= 0)
                ? static_cast<unsigned>(gl::WorkerContext::coreId())
                : scratchArenas().slotCount() - 1;
            ScratchArena& strArena = scratchArenas().forSlot(strSlot);
            const int maxOrigins = parameters.compressor_mode
                ? parameters.compressor_max_origins_per_expr
                : parameters.max_origin_per_expr;

            for (int32_t i = 0; i < staged.count(); ++i) {
                const StagedEquiProducts::Record& rec = staged.recordAt(i);
                if (rec.kind == StagedEquiProducts::Seed) continue;
                if (mb.phase3IgnoresScope(rec.targetVid)) continue;
                ScratchScope recScope(strArena);
                const StrSpan text = staged.textAt(i);
                const StrSpan tView = mb.nameMap.decodeView(rec.targetVid);
                const ScratchString tName =
                    ScratchString::copyFrom(strArena, tView.ptr, tView.len);

                if (rec.kind == StagedEquiProducts::Full) {
                    int lv[512];
                    const int32_t lvN = staged.copyLevels(i, lv, 512);
                    const IntEncodedExpr ie =
                        watchProcessor.commitDerivedLocalRow(mb, text, rec.targetVid, lv, lvN);
                    committed.push_back(ie);
                }

                if (parameters.trackHistory
                    && !proofHistory.originRowExists(mb, text, StrSpan(tName))) {
                    assert(rec.sourceRec >= 0 && rec.classIdx >= 0
                        && "drainStagedEquiProducts: a product must carry its source and class");
                    const StrSpan srcText = staged.textAt(rec.sourceRec);
                    const StrSpan srcScopeV = mb.nameMap.decodeView(
                        staged.recordAt(rec.sourceRec).targetVid);
                    const ScratchString srcScope =
                        ScratchString::copyFrom(strArena, srcScopeV.ptr, srcScopeV.len);
                    const StrSpan classScopeV =
                        mb.nameMap.decodeView(table.rowAt(rec.classIdx).vid);
                    const ScratchString classScope =
                        ScratchString::copyFrom(strArena, classScopeV.ptr, classScopeV.len);
                    const int32_t depN = 1 + rec.pairCount;
                    OriginDep deps[1 + ExecutionParameters::MAX_ARITY];
                    assert(depN <= 1 + ExecutionParameters::MAX_ARITY
                        && "drainStagedEquiProducts: equality1 dep run exceeds cap");
                    deps[0] = OriginDep{ srcText, StrSpan(srcScope) };
                    for (int32_t k = 0; k < rec.pairCount; ++k) {
                        const StrSpan f = mb.nameMap.decodeView(staged.pairFromAt(i, k));
                        const StrSpan t = mb.nameMap.decodeView(staged.pairToAt(i, k));
                        const int32_t n = 3 + f.len + 1 + t.len + 2;
                        char* buf = strArena.allocBytes(n);
                        int32_t at = 0;
                        buf[at++] = '('; buf[at++] = '='; buf[at++] = '[';
                        std::memcpy(buf + at, f.ptr, static_cast<std::size_t>(f.len));
                        at += f.len;
                        buf[at++] = ',';
                        std::memcpy(buf + at, t.ptr, static_cast<std::size_t>(t.len));
                        at += t.len;
                        buf[at++] = ']'; buf[at++] = ')';
                        assert(at == n);
                        const StrSpan equality(buf, n);
                        deps[1 + k] = OriginDep{ equality,
                            proofHistory.findEqualityCiteScope(mb, equality, StrSpan(classScope)) };
                    }
                    addOriginEncoded(mb.exprOriginMap, mb.originInterner,
                        text, StrSpan(tName), OriginTag::equality1, deps, depN,
                        maxOrigins);
                }
            }

            // OR convergence follows the existing product commit order. The
            // branch-row removals of every convergence are applied together
            // after the loop: ordisMerge reads neither the statement lists
            // nor the class waterlines, so the lists end as if each removal
            // ran at once (StatementRegistryOperations::RowRemovalBatch).
            if (committed.size() > 0) {
                RT_SCOPE_HERE("AEC_BATCH_ORDIS");
                ScratchScope mergeScope(gArena);
                gl::StatementRegistryOperations::RowRemovalBatch rowRemovals(gArena);
                for (int32_t r = 0; r < committed.size(); ++r) {
                    const IntEncodedExpr& prod = committed[r];
                    ScratchScope prodScope(strArena);
                    const StrSpan origV = mb.nameMap.decodeView(prod.originalId);
                    const ScratchString original =
                        ScratchString::copyFrom(strArena, origV.ptr, origV.len);
                    const StrSpan validV = mb.nameMap.decodeView(prod.validityId);
                    const ScratchString validity =
                        ScratchString::copyFrom(strArena, validV.ptr, validV.len);
                    const int32_t plvId = mb.intStatementLevelsMap.lookup(
                        packStatementKey(prod.originalId, prod.validityId));
                    assert(plvId != 0 && mb.intStatementLevelsMap.runLen(plvId) > 0
                        && "drainStagedEquiProducts: committed product lost its levels row");
                    int prodLv[256];
                    const int32_t prodLvN = coldIntRunAt(mb.intStatementLevelsMap,
                                                         plvId, prodLv, 256);
                    caseProcessor.ordisMerge(StrSpan(original), StrSpan(validity),
                                     prodLv, prodLvN, mb, nullptr, &rowRemovals);
                }
                rowRemovals.apply(mb);
            }
        }

/// @brief Canonicalize an unused deposit and record its equality bridge.
        ///
        /// @details
        /// All substitutions use the completed classes at the row's own scope
        /// and the original argument text. Eligible cores and excluded shapes
        /// return unchanged. The caller owns the returned text and level run.
        ///
        /// @param row Original encoded row used for core eligibility.
        /// @param text Stable original expression text.
        /// @param validity Stable scope text.
        /// @param mb Owning block and current eligibility snapshots.
        /// @param local Input locality, enriched by locally justified rewrites.
        /// @param levels Caller-owned ascending level buffer, capacity 256.
        /// @param levelCount Input/output length of the level buffer.
        /// @param source Incoming producer origin, or absent for an existing row.
        /// @param arena String scratch owning rewritten text and dependencies.
        /// @return Canonical text and the complete simultaneous substitution evidence.
        /// @invariant History is written but never consulted for a proof decision.
        /// @see canonicalFormAtScope, usedAsPremise.
inline CanonicalForm EqualityPropagation::canonicalizeUnusedDeposit(const IntEncodedExpr& row,
            StrSpan text, StrSpan validity, Memory& mb, bool& local,
            int* levels, int32_t& levelCount, const TransientOrigin& source,
            ScratchArena& arena) {
            CanonicalForm unchanged{};
            unchanged.text = text;
            if (parameters.skip_eq_classes || usedAsPremise(mb, row))
                return unchanged;
            CanonicalForm result = canonicalizer.canonicalFormAtScope(text, validity, mb,
                arena, levels, levelCount, 256);
            if (!result.changed) return result;
            for (int32_t i = 0; i < result.eqN; ++i) local = local || result.eqLocal[i];
            // Counterexample fact templates intentionally carry no history;
            // their disposable clones never export proof certificates.
            if (parameters.trackHistory && !ceFilteringActive) {
                const int cap = parameters.compressor_mode
                    ? parameters.compressor_max_origins_per_expr
                    : parameters.max_origin_per_expr;
                if (source.present) {
                    addOriginEncoded(mb.exprOriginMap, mb.originInterner,
                        text, validity, source.tag, source.deps, source.depN, cap);
                } else {
                    assert(proofHistory.originRowExists(mb, text, validity)
                        && "canonical-only rewrite has no source history");
                }
                if (local) proofHistory.copyOriginRowsToMailOut(mb, text, validity, cap);
                if (!proofHistory.originRowExists(mb, result.text, validity)) {
                    OriginDep dependencies[1 + ExecutionParameters::MAX_ARITY];
                    dependencies[0] = { text, validity };
                    for (int32_t i = 0; i < result.eqN; ++i) {
                        const StrSpan cite = proofHistory.findEqualityCiteScope(mb, result.eqJust[i], validity);
                        const ScratchString owned = ScratchString::copyFrom(arena, cite.ptr, cite.len);
                        dependencies[1 + i] = { result.eqJust[i], StrSpan(owned) };
                    }
                    addOriginEncoded(mb.exprOriginMap, mb.originInterner,
                        result.text, validity, OriginTag::equality1,
                        dependencies, 1 + result.eqN, cap);
                }
            }
            return result;
        }
}

namespace gl {
/// @brief Bind inputs for equality propagation.
/// @details Construction stores explicit references; block storage and all invocation lifetimes remain with their owners.
/// @param admissionPolicy Borrowed AdmissionPolicy interface or state; must outlive this component.
/// @param allMappingsAna Borrowed std::map interface or state; must outlive this component.
/// @param canonicalizer Borrowed Canonicalizer interface or state; must outlive this component.
/// @param ceFilteringActive Borrowed bool interface or state; must outlive this component.
/// @param parameters Borrowed ProverParameters interface or state; must outlive this component.
/// @param caseProcessor Borrowed CaseProcessor interface or state; must outlive this component.
/// @param proofHistory Borrowed ProofHistory interface or state; must outlive this component.
/// @param statementMemory Borrowed StatementMemory interface or state; must outlive this component.
/// @param watchProcessor Borrowed WatchProcessor interface or state; must outlive this component.
/// @return None; construction binds the supplied references.
inline EqualityPropagation::EqualityPropagation(AdmissionPolicy& admissionPolicy, const std::map<int64_t, std::vector<std::vector<int> > >& allMappingsAna, Canonicalizer& canonicalizer, const bool& ceFilteringActive, const ProverParameters& parameters, CaseProcessor& caseProcessor, ProofHistory& proofHistory, StatementMemory& statementMemory, WatchProcessor& watchProcessor) : admissionPolicy(admissionPolicy), allMappingsAna(allMappingsAna), canonicalizer(canonicalizer), ceFilteringActive(ceFilteringActive), parameters(parameters), caseProcessor(caseProcessor), proofHistory(proofHistory), statementMemory(statementMemory), watchProcessor(watchProcessor) {}
/// @brief Mail one positive equality at `main` that the class-canonical
        ///        sweep drops from the outgoing delta — the merge it records
        ///        travels as a statement, not only as history.
        ///
        /// @details
        /// The sweep (`cleanUpExpressions`) drops every outgoing delta row that
        /// names a non-canonical member of a class at its scope, and only the
        /// row's history reaches `mailOut` (D-286). An
        /// equality that merges a member into a class always names that member,
        /// so without this call no merge would ever reach a descendant: the
        /// descendant keeps the aliases apart and never fires an inherited rule
        /// on their mailed rows (I-57). A positive equality at `main` that
        /// passes the statement channel's witness gate (`allowedForMail`: no
        /// ordinary integration witness) is inserted into `mailOut.statements`
        /// with its levels row, and with the from-premise bit when it is a
        /// forced product, as `fillMailOut` would ship it. Any other row is a
        /// defined no-op: its history alone travels, the canonicalization
        /// policy unchanged (D-544).
        ///
        /// @param mb  Owning LB; resident and claimed (phase 3).
        /// @param row The dropped delta row; its levels row exists (I-58).
        /// @return True iff the row was inserted into `mailOut.statements`.
        /// @invariant An equality is never a pending relay or an unconditional
        ///            OR (asserted), so no `fillMailOut` flag is lost.
        /// @see cleanUpExpressions, BlockMailProcessor::fillMailOut,
        ///      BlockMailProcessor::allowedForMail.
inline bool
            EqualityPropagation::shipDroppedMerge(Memory& mb, const IntEncodedExpr& row) {
            if (row.validityId != NameMap::MAIN_ID) return false;
            const StrSpan original = mb.nameMap.decodeView(row.originalId);
            const bool positiveEquality = original.len > 3 && original.ptr[0] == '('
                && original.ptr[1] == '=' && original.ptr[2] == '[';
            if (!positiveEquality || containsIntegrationWitness(original)) return false;
            assert(mb.pendingRelayIter.lookup(static_cast<int32_t>(row.originalId)) == 0
                && mb.lbMemory.pendingUnconditionalOrdis.lookup(row.originalId) == 0
                && "shipDroppedMerge: an equality is never a relay or an unconditional OR");
            const int64_t key = packStatementKey(row.originalId, row.validityId);
            const int32_t levelsId = mb.intStatementLevelsMap.lookup(key);
            assert(levelsId != 0 && mb.intStatementLevelsMap.runLen(levelsId) > 0
                && "shipDroppedMerge: a delta row has its levels row (I-58)");
            int levels[256];
            const int32_t levelN = coldIntRunAt(mb.intStatementLevelsMap, levelsId, levels, 256);
            // mailOut mints into its private interner only, so the NameMap
            // views stay valid (the fillMailOut argument).
            const StrSpan validity = mb.nameMap.decodeView(row.validityId);
            mb.insertMailOutStatement(original, validity, levels, levelN);
            if (mb.lbMemory.forcedProducts.lookup(key) != 0)
                mb.setMailOutStatementFlag(original, validity, kMailStatementFlagNone, -1, true);
            return true;
        }
} // namespace gl
