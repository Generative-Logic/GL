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
#include "../memory.hpp"
#include "hash_rule_index.hpp"
#include "hash_key_builder.hpp"
#include "request_gate_state.hpp"
#include "../infra/rt_tracker.hpp"
namespace gl {
namespace tests { struct HashRequestGeneratorFixture; }
/// @brief Generate deterministic mandatory-aware hash requests and split stumps.
/// @details Owns this algorithm boundary; logical blocks retain their physical stores.
class HashRequestGenerator {
friend struct tests::HashRequestGeneratorFixture;
public:
/// @brief Bind HashRequestGenerator inputs.
/// @details Binds the existing inputs without performing proof work or moving storage.
/// @param parameters Existing parameters interface or settings.
/// @return None; construction binds the supplied references.
HashRequestGenerator(const ProverParameters& parameters);
        static constexpr int kMaxSplitParts = 16384;

/// @brief Read this worker part's growth count.
/// @details Reads or resets the existing request counters at their established boundaries.
/// @return The requested counter value.
static int64_t growthMatchCount() { return g_growthMatchCount; }

/// @brief Reset this worker part's growth count.
/// @details Reads or resets the existing request counters at their established boundaries.
/// @return Nothing.
static void resetGrowthMatchCount() { g_growthMatchCount = 0; }

/// @brief Reset one depth of the worker growth census.
/// @details Reads or resets the existing request counters at their established boundaries.
/// @param depth Depth row to reset.
/// @return Nothing.
static void resetDepthCounters(int32_t depth) { assert(depth >= 0 && depth <= ExecutionParameters::MAX_EXPRESSIONS); g_gpuGrowAttemptsByDepth[depth] = 0; g_gpuGrowFrontierByDepth[depth] = 0; g_gpuGrowSubkeysByDepth[depth] = 0; g_gpuGrowRequestsByDepth[depth] = 0; g_gpuProducerAttemptsByDepth[depth] = 0; g_gpuProducerSurvivorsByDepth[depth] = 0;  }

/// @brief Read one worker growth census counter.
/// @details Reads or resets the existing request counters at their established boundaries.
/// @param depth Depth row to inspect.
/// @return The requested counter value.
static uint64_t gpuGrowAttemptsByDepth(int32_t depth) { assert(depth >= 0 && depth <= ExecutionParameters::MAX_EXPRESSIONS); return g_gpuGrowAttemptsByDepth[depth]; }

/// @brief Read one worker growth census counter.
/// @details Reads or resets the existing request counters at their established boundaries.
/// @param depth Depth row to inspect.
/// @return The requested counter value.
static uint64_t gpuGrowFrontierByDepth(int32_t depth) { assert(depth >= 0 && depth <= ExecutionParameters::MAX_EXPRESSIONS); return g_gpuGrowFrontierByDepth[depth]; }

/// @brief Read one worker growth census counter.
/// @details Reads or resets the existing request counters at their established boundaries.
/// @param depth Depth row to inspect.
/// @return The requested counter value.
static uint64_t gpuGrowSubkeysByDepth(int32_t depth) { assert(depth >= 0 && depth <= ExecutionParameters::MAX_EXPRESSIONS); return g_gpuGrowSubkeysByDepth[depth]; }

/// @brief Read one worker growth census counter.
/// @details Reads or resets the existing request counters at their established boundaries.
/// @param depth Depth row to inspect.
/// @return The requested counter value.
static uint64_t gpuGrowRequestsByDepth(int32_t depth) { assert(depth >= 0 && depth <= ExecutionParameters::MAX_EXPRESSIONS); return g_gpuGrowRequestsByDepth[depth]; }

/// @brief Read one worker growth census counter.
/// @details Reads or resets the existing request counters at their established boundaries.
/// @param depth Depth row to inspect.
/// @return The requested counter value.
static uint64_t gpuProducerAttemptsByDepth(int32_t depth) { assert(depth >= 0 && depth <= ExecutionParameters::MAX_EXPRESSIONS); return g_gpuProducerAttemptsByDepth[depth]; }

/// @brief Read one worker growth census counter.
/// @details Reads or resets the existing request counters at their established boundaries.
/// @param depth Depth row to inspect.
/// @return The requested counter value.
static uint64_t gpuProducerSurvivorsByDepth(int32_t depth) { assert(depth >= 0 && depth <= ExecutionParameters::MAX_EXPRESSIONS); return g_gpuProducerSurvivorsByDepth[depth]; }

/// @brief Reset the batch request-filter counters.
/// @details Reads or resets the existing request counters at their established boundaries.
/// @return Nothing.
void resetFilterCounters() { gpuRequestFilterCalls.store(0, std::memory_order_relaxed); gpuProducerFilterCalls.store(0, std::memory_order_relaxed); gpuFilterInputStatements.store(0, std::memory_order_relaxed); gpuFilterOutputStatements.store(0, std::memory_order_relaxed); gpuFilterMaximumInputStatements.store(0, std::memory_order_relaxed); gpuFilterMaximumOutputStatements.store(0, std::memory_order_relaxed);  }

/// @brief Read one batch request-filter counter.
/// @details Reads or resets the existing request counters at their established boundaries.
/// @return The requested counter value.
uint64_t gpuRequestFilterCallsValue() const { return gpuRequestFilterCalls.load(std::memory_order_relaxed); }

/// @brief Read one batch request-filter counter.
/// @details Reads or resets the existing request counters at their established boundaries.
/// @return The requested counter value.
uint64_t gpuProducerFilterCallsValue() const { return gpuProducerFilterCalls.load(std::memory_order_relaxed); }

/// @brief Read one batch request-filter counter.
/// @details Reads or resets the existing request counters at their established boundaries.
/// @return The requested counter value.
uint64_t gpuFilterInputStatementsValue() const { return gpuFilterInputStatements.load(std::memory_order_relaxed); }

/// @brief Read one batch request-filter counter.
/// @details Reads or resets the existing request counters at their established boundaries.
/// @return The requested counter value.
uint64_t gpuFilterOutputStatementsValue() const { return gpuFilterOutputStatements.load(std::memory_order_relaxed); }

/// @brief Read one batch request-filter counter.
/// @details Reads or resets the existing request counters at their established boundaries.
/// @return The requested counter value.
uint64_t gpuFilterMaximumInputStatementsValue() const { return gpuFilterMaximumInputStatements.load(std::memory_order_relaxed); }

/// @brief Read one batch request-filter counter.
/// @details Reads or resets the existing request counters at their established boundaries.
/// @return The requested counter value.
uint64_t gpuFilterMaximumOutputStatementsValue() const { return gpuFilterMaximumOutputStatements.load(std::memory_order_relaxed); }

private:
/// @brief Probe the subkey owner-set map with key presence plus, from
        ///        `kSubkeyUCheckMinElements` premises on, the D-120 u_-signature
        ///        check — the growth probe of the request generator.
        ///
        /// @details
        /// `normKeyLookup` first (a miss returns `false` before any value read).
        /// A subkey shorter than `kSubkeyUCheckMinElements` premises is accepted
        /// on presence alone. Otherwise a zero-copy peek of the key's single
        /// `OwnerSet` blob (`peekRecordBytes` + `OwnerSetBlob`) is consumed by
        /// `ownerSetUSatisfied` alone: the request continues only when some
        /// owner's cached u_ signature is satisfiable by its flattened
        /// `argFullId` (or an owner is loose). Nothing else is read: the
        /// whole-key record probe (`wholeKeyPresent`) is presence only, and the
        /// firing site re-checks every gate. Sound by
        /// [I-79](../30_invariants.md#i-79). Read-only, mints nothing, burst-safe
        /// on the shared LB ([I-83](../30_invariants.md#i-83)); a `thread_local`
        /// scratch backs the rare page-straddle blob copy.
        ///
        /// @param map       The subkey owner-set map (`normalizedEncodedSubkeys`).
        /// @param keyBuf    The normalized-key `NameId` payload.
        /// @param keyLen    The payload length (`<= MAX_KEY_SLOTS`).
        /// @param exprs     The request's premise pointers, in name-sorted order.
        /// @param exprCount Premise count == the key's `numberExpressions`.
        /// @return `true` to continue the search, `false` on a miss or a key of
        ///         at least `kSubkeyUCheckMinElements` premises that no owner's
        ///         u_ signature admits.
        /// @see `normKeyLookup`, `ownerSetUSatisfied`, `wholeKeyPresent`,
        ///      `kSubkeyUCheckMinElements`, `mergeSubkeySignatures`.
static bool subkeyUSatisfied(
            const TypedColdBlobMap<NormKey, OwnerSet>& map,
            const NameId* keyBuf, NameId keyLen,
            const IntEncodedExpr* const* exprs, NameId exprCount) {
            const int32_t id = gl::HashRuleIndex::normKeyLookup(map, keyBuf, keyLen, exprCount);
            if (id == 0) {
                return false;
            }
            if (exprCount < gl::HashRuleIndex::kSubkeyUCheckMinElements) {
                return true;
            }
            thread_local std::vector<char> scratch;
            int32_t blen = 0;
            const char* bp = map.peekRecordBytes(id, 0, blen, scratch);
            const OwnerSetBlob ob{ bp, blen };
            return ownerSetUSatisfied(ob, exprs, exprCount);
        }

private:
/// @brief Clear a gate summary to the empty-premise state.
        ///
        /// @param state  Summary to clear; the `seenSecondary` array is left
        ///               untouched because `secondaryCount` gates every read.
        /// @post Folding zero premises leaves exactly this state.
        /// @see `foldExprIntoRequestGates` — what runs next.
static inline void resetRequestGateState(RequestGateState& state) {
            state.hypoFound = 0;
            state.hypoValidityId = -1;
            state.nonExemptScopes = 0;
            state.nonExemptValidityId = -1;
            state.secondaryCount = 0;
        }

private:
/// @brief Fold one premise into a gate summary.
        ///
        /// @details
        /// The single per-premise body of the request gates: `requestGatesPass`
        /// runs it over a whole candidate and the grow search runs it over a
        /// prefix once plus one appended premise per position, so the two paths
        /// cannot drift apart.
        ///
        /// The `productsOfRecursionIds` membership probe — the most expensive
        /// thing the gates do — happens here, once per argument of THIS premise,
        /// which is why folding beats re-scanning the whole candidate.
        ///
        /// @param e               Premise to fold in.
        /// @param state           Summary to advance, in place.
        /// @param prodRecIds      The LB's recursion-product id set; an argument
        ///                        in it is not a secondary variable.
        /// @param mainValidityId  `NameMap::MAIN_ID`; an anchor at this scope is
        ///                        exempt from the hypothesis scope agreement.
        /// @return `false` when @p e is a hypothesis premise at a scope that
        ///         disagrees with one already folded in — the candidate can never
        ///         fire. `true` otherwise; the verdict still needs
        ///         `requestGateStateAccepts`.
        /// @invariant Allocates nothing.
        /// @see `requestGatesPass` — the whole-candidate entry.
static inline bool foldExprIntoRequestGates(
                const IntEncodedExpr& e, RequestGateState& state,
                const TypedColdSet<NameId>& prodRecIds,
                NameId mainValidityId) {
            if (e.isHypo) {
                if (state.hypoFound && e.validityId != state.hypoValidityId)
                    return false;
                state.hypoValidityId = e.validityId;
                state.hypoFound = 1;
            }

            // A main-scope anchor rides along with any hypothesis scope, so it
            // never constrains the agreement and stays out of the count.
            if (!(e.validityId == mainValidityId && e.isAnchor)) {
                if (state.nonExemptScopes == 0) {
                    state.nonExemptScopes = 1;
                    state.nonExemptValidityId = e.validityId;
                }
                else if (state.nonExemptScopes == 1
                         && state.nonExemptValidityId != e.validityId) {
                    state.nonExemptScopes = 2;
                }
            }

            // Secondary variables are counted DISTINCT across the whole request:
            // the same it_/int_ variable in several premises is one secondary.
            for (NameId a = 0; a < e.arity; ++a) {
                if (e.argIteration[a] > -1
                 && !prodRecIds.contains(e.argFullId[a])) {
                    const NameId id = e.argFullId[a];
                    bool found = false;
                    for (int s = 0; s < state.secondaryCount; ++s) {
                        if (state.seenSecondary[s] == id) { found = true; break; }
                    }
                    if (!found) {
                        assert(state.secondaryCount
                               < RequestGateState::kMaxSeenSecondary);
                        state.seenSecondary[state.secondaryCount++] = id;
                    }
                }
            }
            return true;
        }

private:
/// @brief Fold a premise prefix that has already passed the gates.
        ///
        /// @details
        /// The grow search's per-pop entry. Every node the DFS pushes passed the
        /// gates before it was pushed, and the gates are closed downward under
        /// subsets, so no premise of a grown prefix can conflict here. That is
        /// asserted rather than branched on: a firing assert would mean the
        /// search pushed a node it had rejected.
        ///
        /// @param exprs           Prefix premises, in key order.
        /// @param count           Prefix length; `0` is legal.
        /// @param prodRecIds      The LB's recursion-product id set.
        /// @param mainValidityId  `NameMap::MAIN_ID`.
        /// @param state           Out-parameter; cleared then filled.
        /// @invariant Allocates nothing.
        /// @see `foldExprIntoRequestGates` — the per-premise step.
static inline void foldPrefixIntoRequestGates(
                const IntEncodedExpr* const* exprs, NameId count,
                const TypedColdSet<NameId>& prodRecIds,
                NameId mainValidityId, RequestGateState& state) {
            resetRequestGateState(state);
            for (NameId i = 0; i < count; ++i) {
                const bool prefixAccepted = foldExprIntoRequestGates(
                    *exprs[i], state, prodRecIds, mainValidityId);
                assert(prefixAccepted
                    && "a grown prefix has already passed the request gates");
                static_cast<void>(prefixAccepted);
            }
        }

private:
/// @brief The gate verdict over a finished summary.
        ///
        /// @details
        /// Everything the gates decide once the premises are all folded in: the
        /// hypothesis-key length cap and scope agreement, the distinct-secondary
        /// cap with its `_orint_` widening, and the key-length cap. Only the rare
        /// over-cap widening path re-reads the premise array, and only to check
        /// that they share one scope and to decode that scope's name.
        ///
        /// @param state           Summary of all @p count premises.
        /// @param exprs           The premises, needed only by the widening path.
        /// @param count           Premise count (`>= 1`).
        /// @param body            Owning LB — read for `maxKeyLength` and, on the
        ///                        widening path, the scope name.
        /// @param mainValidityId  `NameMap::MAIN_ID`, unused here but kept in the
        ///                        signature so the two gate entries read alike.
        /// @return `true` when the request may be probed against a normalized-key index.
        /// @invariant Allocates nothing.
        /// @see `requestGatesPass` — the whole-candidate entry.
inline bool requestGateStateAccepts(
                const RequestGateState& state,
                const IntEncodedExpr* const* exprs, NameId count,
                const Memory& body, NameId mainValidityId) const {
            static_cast<void>(mainValidityId);

            if (state.hypoFound) {
                if (count > parameters.maxLenHypoKey) return false;
                // Every non-exempt premise must sit at the hypothesis scope, so
                // the non-exempt scopes must be a subset of { hypoValidityId }.
                if (state.nonExemptScopes > 1) return false;
                if (state.nonExemptScopes == 1
                    && state.nonExemptValidityId != state.hypoValidityId)
                    return false;
            }

            if (state.secondaryCount > parameters.maxNumberSecondaryVariables) {
                // Scoped widening: the deeper cap applies only when EVERY
                // premise sits at ONE shared validity scope AND that scope is
                // an _orint_ branch. The decode runs only on this rare
                // over-standard-cap path, never on the common accept path.
                if (state.secondaryCount
                    > parameters.maxNumberSecondaryVariablesOrint) {
                    return false;
                }
                const NameId v0 = exprs[0]->validityId;
                for (NameId i = 1; i < count; ++i) {
                    if (exprs[i]->validityId != v0) {
                        return false;
                    }
                }
                if (!containsSpan(body.nameMap.decodeView(v0),
                                  StrSpan("_orint_", 7))) {
                    return false;
                }
            }

            if (count > body.overallHashMemory.maxKeyLength) {
                return false;
            }

            return true;
        }

private:
/// @brief The three map-independent request-shape gates: hypothesis-scope
        ///        consensus, distinct-secondary-variable cap, key length.
        ///
        /// @details
        /// A candidate request is a name-sorted array of premise pointers. Before
        /// any normalized-key index is probed it must clear three cheap checks, each of
        /// which is a necessary condition for the request to fire:
        ///
        /// 1. **Hypothesis-scope consensus.** All hypothetical premises must share
        ///    one `validityId` — a request straddling two hypothesis scopes can
        ///    never fire. When a hypothesis is present the request may not exceed
        ///    `parameters.maxLenHypoKey`, and every remaining premise must either
        ///    sit at that hypothesis scope or be a `main`-scope anchor.
        /// 2. **Distinct secondary variables.** The number of DISTINCT secondary
        ///    variables across the whole request (`argIteration > -1`, products of
        ///    recursion excluded) may not exceed
        ///    `parameters.maxNumberSecondaryVariables`. Dedup is by `argFullId`, so
        ///    one variable occurring in several premises counts once.
        ///    **Scoped widening:** when every premise sits at ONE shared validity
        ///    scope and that scope is an `_orint_` branch, the cap is
        ///    `parameters.maxNumberSecondaryVariablesOrint` instead (default equal
        ///    to the standard cap = no widening). An `_orint_` branch is where
        ///    predecessor-descent chains mint one extra `it_` witness per level,
        ///    so the widened budget is confined to exactly those scopes; requests
        ///    mixing scopes or outside `_orint_` keep the standard cap.
        /// 3. **Length.** The request may not be longer than the LB's longest
        ///    installed key (`overallHashMemory.maxKeyLength`).
        ///
        /// The request generator's grow loop runs the gates once, builds the
        /// normalized key once, and then probes the TWO normalized-key indexes with that one
        /// key — the subkey map that decides whether the candidate may grow
        /// (`subkeyUSatisfied`), and the whole-key map that decides whether it is
        /// recorded (`wholeKeyPresent`) — instead of paying for two full
        /// pre-evaluations (which would also double-count the submatch tally).
        ///
        /// The distinct-secondary dedup runs on a stack array sized by
        /// `(MAX_EXPRESSIONS + 2) * MAX_ARITY`: the distinct count is tiny in
        /// practice (2-3), so a linear insert-if-absent beats a hash set's per-node
        /// allocation on the hottest burst path. Overflow is a loud assert, never a
        /// truncation.
        ///
        /// @param exprs           Premise pointers, name-sorted; pairwise scope
        ///                        comparability is the caller's precondition.
        /// @param count           Premise count (`>= 1`).
        /// @param body            Owning LB — read for `overallHashMemory`'s
        ///                        `maxKeyLength` and `productsOfRecursionIds`.
        /// @param mainValidityId  `NameMap::MAIN_ID`; anchors at this scope are
        ///                        admitted alongside a hypothesis scope.
        /// @return `true` when the request may be probed against a normalized-key index;
        ///         `false` when it can never fire and must be dropped.
        /// @invariant Reads only `const` LB state and touches no arena, so it is
        ///            safe on the shared read-only LB that the phase-2 split parts
        ///            run over in parallel
        ///            ([I-83](../../docs/agentic_swdd/30_invariants.md#i-83)).
        /// @invariant Allocates nothing — bounded stack scratch only
        ///            ([I-130](../../docs/agentic_swdd/30_invariants.md#i-130)).
        /// @see `generateEncodedRequestsStatic` — the probing caller.
        /// @see `subkeyUSatisfied` / `wholeKeyPresent` — the map probes the gates
        ///      precede.
        /// @see `foldExprIntoRequestGates` + `requestGateStateAccepts` — the core
        ///      this delegates to, which the grow search runs one premise at a
        ///      time over a fixed prefix instead of re-scanning each candidate.
inline bool requestGatesPass(const IntEncodedExpr* const* exprs,
                                     NameId count, const Memory& body,
                                     NameId mainValidityId) const {

            RequestGateState state;
            resetRequestGateState(state);
            const TypedColdSet<NameId>& prodRecIds =
                body.overallHashMemory.productsOfRecursionIds;
            for (NameId i = 0; i < count; ++i) {
                if (!foldExprIntoRequestGates(*exprs[i], state, prodRecIds,
                                              mainValidityId)) {
                    return false;
                }
            }
            return requestGateStateAccepts(state, exprs, count, body,
                                           mainValidityId);
        }

private:
/// @brief Statement pre-filter for request generation: keep the statements
        ///        that can appear in a request at all.
        ///
        /// @details
        /// Builds each statement's single-element normalized key on the stack and
        /// keeps the statement when that key is a growable subkey
        /// (`subkeyUSatisfied`) or, with @p alsoAcceptFullKeys, a present whole
        /// key (`wholeKeyPresent`), when its `maxIteration` is within
        /// `parameters.maxIterationNumberVariable`, and when some live rule
        /// premise of the LB accepts it (`premisePatternAdmits`: any of the
        /// four hash memories holds a premise with the statement's signed
        /// predicate and repetition shape, carrying the statement's arguments
        /// at its `u_` slots, at a scope comparable with the statement's that
        /// no subtree wipe closed). Each part is necessary for the statement
        /// to take part in a firing request; the GPU upload prefilter applies
        /// the same union, so both routes keep one list.
        ///
        /// Before the key build, the frozen or-branch exclusion
        /// (I-206): a statement whose scope is a frozen
        /// `_ordis_` branch (`Memory::frozenOrBranches`) or ANY descendant of
        /// one — one upward `NameMap::parentOf` walk probing the frozen set —
        /// is dropped. This is the single consumer of the frozen set, shared by
        /// the generator and the stump producer so the split partition (I-156)
        /// stays exact. An empty frozen set short-circuits the walk.
        ///
        /// Which key map decides acceptance depends on the obligatory stump the
        /// caller will attach:
        ///
        /// - **Stump production** (`alsoAcceptFullKeys == false`) — a surviving
        ///   statement must be *growable*, so only `subkeyUSatisfied` on
        ///   `normalizedEncodedSubkeys` accepts it.
        /// - **Firing-request generation** (`alsoAcceptFullKeys == true`) — a
        ///   statement may itself be a complete request, so
        ///   `normalizedEncodedKeys` accepts it too: the gate is the UNION of
        ///   the subkey and full-key maps. This includes unsplit searches,
        ///   counter-example searches, and split-stump buckets.
        ///
        /// Both probes run before the verdict (no short-circuit visible to the
        /// caller); `wholeKeyPresent` is a pure lookup, so this is
        /// only a cost question, never a behaviour one.
        ///
        /// @param stmts              The statement universe to filter.
        /// @param mem                Reference hash memory supplying the two owner-set
        ///                           maps.
        /// @param body               Owning LB: its `frozenOrBranches` set and its
        ///                           `nameMap` parent forest drive the frozen
        ///                           subtree exclusion; its four hash memories'
        ///                           premise patterns drive the premise-pattern
        ///                           condition (`premisePatternAdmits`).
        /// @param alsoAcceptFullKeys Widen the gate with `normalizedEncodedKeys`.
        ///                           True for request generation and false for
        ///                           stump production.
        /// @param outIndices         Caller buffer receiving the surviving indices
        ///                           into @p stmts, in ascending index order.
        /// @param maxOut             Capacity of @p outIndices.
        /// @return Number of surviving indices written to @p outIndices.
        /// @invariant The key prune and the premise-pattern condition are sound
        ///            over-approximations of the firing gate — neither drops a
        ///            statement that could take part in a firing request
        ///            ([I-79](../../docs/agentic_swdd/30_invariants.md#i-79),
        ///            I-300); the
        ///            frozen-subtree exclusion is the one deliberate withholding
        ///            ([I-206](../../docs/agentic_swdd/30_invariants.md#i-206)).
        /// @invariant Allocates nothing — stack scratch only; reads shared LB
        ///            state `const` (phase 2, I-83)
        ///            ([I-130](../../docs/agentic_swdd/30_invariants.md#i-130)).
        /// @see `generateEncodedRequestsStatic`, `produceExpressionStumps` — the
        ///      two consumers.
        /// @see `subkeyUSatisfied`, `wholeKeyPresent`, `Memory::frozenOrBranches`.
        ///
        /// Body lives in hash_request_generator.cpp.
NameId filterIntEncodedStatements(
            IntStmtView stmts,
            const HashMemory& mem, const Memory& body,
            bool alsoAcceptFullKeys,
            NameId* outIndices, NameId maxOut);

public:
/// @brief THE request generator: grow every candidate to a whole key and
        ///        emit the ones that carry a mandatory ingredient.
        ///
        /// @details
        /// One function serves every request-generation mode, and every caller
        /// drives it the same way: with @p terms, the mandatory-containment
        /// control. A candidate is grown all the way to a whole key of
        /// `maxKeyLength` elements against `normalizedEncodedKeys`, and emitted
        /// only when it holds a statement from every view of at least ONE term.
        ///
        /// An EMPTY term list is a first-class input, not the absence of one: it
        /// says no ingredient is mandatory, so every whole key is a request. Two
        /// callers pass it - the counter-example filter, and the batch whose rule
        /// registry is itself the new thing (`localHashMemoryDelta`), which has to
        /// meet every visible statement rather than only this burst's. Nothing in
        /// the body may read an empty term list as "this is the counter-example
        /// filter": the second caller IS stump-split.
        ///
        /// **Mandatory containment.** The terms are a disjunction of conjunctions
        /// - a request is emitted only when it satisfies at least ONE term, and a
        /// term is satisfied when the request contains at least one statement from
        /// EVERY view that term names. Membership becomes one bit per filtered
        /// statement, computed once per call, so the check at a finished candidate
        /// is a mask test. The search prunes a node as soon as no term can still
        /// be completed - neither within the depth still available, nor from the
        /// statements still ahead of it in the sorted list.
        ///
        /// This is the same emission set the retired pairing merge produced, at a
        /// fraction of the cost: the merge enumerated (base candidate x obligatory
        /// stump) pairs and collapsed the duplicates afterwards, while the
        /// containment search reaches every request exactly once. Equivalence rests
        /// on the subkey map being closed downward - every subset of a whole key is
        /// a growable subkey, so the merge's per-stump viability pre-filter and its
        /// recordable-base probe are both implied by the whole-key probe at the
        /// finished candidate.
        ///
        /// Two phases:
        ///
        /// 1. **Filter.** The statement universe is filtered
        ///    (`filterIntEncodedStatements`, always widened to full keys since a
        ///    candidate is the whole request) and name-sorted. This is the fixed
        ///    per-call cost, and under a stump split the whole bucket shares it.
        /// 2. **Grow.** A depth-first search extends the empty candidate one
        ///    statement at a time, in ascending name order, up to `maxKeyLength`.
        ///    Per node it builds the normalized key once and probes two owner-set
        ///    maps with it: `normalizedEncodedSubkeys` decides whether the
        ///    candidate may grow further — by key presence plus the D-120
        ///    u_-signature check alone (`subkeyUSatisfied`; scope comparability
        ///    and partition are not read at the growth probe) — and is the
        ///    sole contributor to the `g_growthMatchCount` submatch tally (D-109);
        ///    `normalizedEncodedKeys` decides whether it is a finished request,
        ///    by key presence alone (`wholeKeyPresent`; the owner record is not
        ///    read — the firing site re-checks every gate). A recorded candidate is
        ///    emitted on the spot, which is what preserves the counter-example
        ///    filter's contradiction early-exit.
        ///
        ///    Under a **stump split** (a non-empty @p splitStump bucket) each stump
        ///    NAMES A NODE of this same search. The run seeds its stack with that
        ///    node - the stump's elements ARE the growing candidate - and grows it
        ///    onward, children starting past the stump's last element exactly as
        ///    the unsplit search grows them. A run therefore owns that node's
        ///    SUBTREE and nothing else. Since the search extends in ascending list
        ///    order, every candidate has exactly one path, so distinct stumps'
        ///    subtrees are DISJOINT and together cover the whole unsplit
        ///    enumeration: the split is an exact partition and generates the
        ///    identical total work. The stump alone is probed once before the
        ///    search: it is the one candidate no descendant reproduces, since
        ///    unsplit it was recorded inside the loop of the candidate one level
        ///    up and this sub-part never runs that loop. A terminal-only stump
        ///    stops after this probe: it preserves a recordable shallow producer
        ///    node, while the node's surviving children cover every larger
        ///    candidate. A stump holding a statement this call's rule registry
        ///    cannot use is skipped before the seed frame - see
        ///    [I-156](../../docs/agentic_swdd/30_invariants.md#i-156).
        ///
        /// @tparam Consumer  Streaming request sink; must expose `canAccept()` and
        ///                   `consume(const StaticRequest&)`. `BurstSink` is the
        ///                   only production instantiation.
        /// @param body       Owning LB; supplies `nameMap` and `intEncodedStatements`.
        /// @param intMemory  Reference hash memory: `maxKeyLength` and the two
        ///                   `normalizedEncoded*` normalized-key indexes.
        /// @param terms      Mandatory-containment terms, OR-ed; `nullptr` iff
        ///                   @p termCount is 0.
        /// @param termCount  Number of terms, at most `kMaxMandatoryTerms`; 0 means
        ///                   no requirement at all.
        /// @param splitStump The LB split's expression dimension: this sub-part's
        ///                   stump bucket plus its place among its siblings.
        ///                   Default-constructed when the LB is not stump-split,
        ///                   and the generator is then line-for-line the unsplit
        ///                   one.
        /// @param coreId     Per-slot scratch-arena / mailbox id.
        /// @param consumer   Sink; a `false` from it ends generation immediately.
        /// @invariant Every emitted request's `IntNormalizedKey::data` and premise
        ///            copies live on the per-slot gen scratch arena and stay valid
        ///            until the per-task `releaseAll`.
        /// @invariant Heap-free: stack arrays plus the gen scratch arena's byte-bump
        ///            and page tiers
        ///            ([I-130](../../docs/agentic_swdd/30_invariants.md#i-130), Rule 28).
        /// @invariant Writes no shared cross-LB state during the parallel phase
        ///            ([I-28](../../docs/agentic_swdd/30_invariants.md#i-28),
        ///            [I-83](../../docs/agentic_swdd/30_invariants.md#i-83)).
        /// @invariant The owner-set prunes never drop a request that could fire
        ///            ([I-70](../../docs/agentic_swdd/30_invariants.md#i-70),
        ///            [I-79](../../docs/agentic_swdd/30_invariants.md#i-79)).
        /// @see `filterIntEncodedStatements`, `requestGatesPass`, `wholeKeyPresent`,
        ///      `StaticRequestEmitter`, `BurstSink`.
template <typename Consumer>
        void generateEncodedRequestsStatic(
                const Memory& body,
                const HashMemory& intMemory,
                const MandatoryTerm* terms, NameId termCount,
                const SplitStumpRef& splitStump,
                unsigned coreId,
                Consumer& consumer);

public:
/// @brief Build a straggler's whole-LB stump list — the base candidates
        ///        `proveKernel` deals into `logicalCores` expression buckets.
        ///
        /// @details
        /// This is the request generator run in a different mode: no obligatory
        /// stump, all expressions, and the grow search stopped as soon as it
        /// holds enough candidates. The round-1 PRODUCER task calls it;
        /// `filterIntEncodedStatements` (via `subkeyUSatisfied` — presence +
        /// u_ signature) prunes the statement universe to the WHOLE LB's rules —
        /// the stumps span the LB's whole expression search.
        /// It fires nothing; the buckets that run its stumps do the firing work.
        ///
        /// A stump is a candidate that **survives the filter**: its normalized
        /// key passes `subkeyUSatisfied` on `normalizedEncodedSubkeys`, exactly
        /// as a growing candidate is inside the search. The list is one uniform level:
        ///
        /// 1. Every expression surviving `filterIntEncodedStatements` is a
        ///    1-stump. If that reaches @p target, stop.
        /// 2. Otherwise **all** surviving 2-stumps replace them. A recordable
        ///    1-stump is retained as terminal-only work; every other 1-stump is
        ///    dropped because a surviving 2-stump contains it.
        /// 3. Otherwise all surviving 3-stumps replace those, and so on, up to
        ///    `min(MAX_EXPRESSIONS, maxKeyLength)` or until no candidate grows.
        ///
        /// A level overshoots @p target freely — the stump count is the LB's
        /// sub-part count, and @p target is a floor, not a ceiling. Growth is
        /// only ever reached when the filter yields fewer than @p target
        /// expressions.
        ///
        /// **Terminal pre-stumps.** A stump of length `L` can only cover base
        /// candidates of size `>= L`, since a sub-part's candidates are `C u S`.
        /// Before replacing a level, the producer therefore probes the overall
        /// hash memory's minus-one and minus-two maps. Every recordable node is
        /// appended as a terminal-only stump, assigned to exactly one bucket, and
        /// revalidated by each request batch against that batch's actual hash
        /// memory. It may materialise its own `BaseCandidate` but never grows;
        /// the replacement level covers all larger candidates. The producer only
        /// detects work and fires nothing.
        ///
        /// @param body    Owning LB; supplies `nameMap` and `intEncodedStatements`.
        /// @param coreId  Per-slot scratch-arena id.
        /// @param target  Floor on the stump count; the growth trigger.
        /// @param out     Task-owned sealed page set, `Filling`; receives one
        ///                `ExpressionStump` record per stump, in candidate order.
        /// @return Number of regular and terminal-only stumps appended — the
        ///         bucket-deal work-item count.
        /// @invariant Every expression surviving the filter lies inside at least
        ///            one stump, so no base candidate of size `>= L` is orphaned.
        /// @invariant Leaves `g_growthMatchCount` untouched: its growth probe
        ///            (`subkeyUSatisfied`) is a filter and its terminal probe is the
        ///            non-counting `wholeKeyPresent`, so the split policy's work
        ///            statistic still measures the request generator alone (D-109).
        /// @invariant Heap-free: stack arrays plus the gen scratch arena's page
        ///            tier ([I-130](../../docs/agentic_swdd/30_invariants.md#i-130),
        ///            Rule 28).
        /// @invariant Read-only on @p body, like the burst it replaces
        ///            ([I-83](../../docs/agentic_swdd/30_invariants.md#i-83)).
        /// @see `generateEncodedRequestsStatic` — the consumer of these stumps.
        /// @see `ExpressionStump`, `filterIntEncodedStatements`, `wholeKeyPresent`.
int32_t produceExpressionStumps(const Memory& body, unsigned coreId,
                                        int32_t target, SealedPageSet& out);
private:
const ProverParameters& parameters;
        std::atomic<uint64_t> gpuRequestFilterCalls{ 0 };
        std::atomic<uint64_t> gpuProducerFilterCalls{ 0 };
        std::atomic<uint64_t> gpuFilterInputStatements{ 0 };
        std::atomic<uint64_t> gpuFilterOutputStatements{ 0 };
        std::atomic<uint64_t> gpuFilterMaximumInputStatements{ 0 };
        std::atomic<uint64_t> gpuFilterMaximumOutputStatements{ 0 };

        // Per-split-part submatch counter — THE hashburst work metric (replaces
        // the former emitted-request count). Counts each match: a growing request
        // allowed to add an expression. ONE site bumps it — the grow-DFS subkey
        // probe inside generateEncodedRequestsStatic. It drives BOTH the cap (BurstSink::canAccept stops the burst, and the
        // grow-DFS bails on it, once a part reaches maxNumberHashRequests) AND the
        // split policy (the busiest part's count is the fill-ratio numerator).
        // performElem2 resets it to 0 at entry, so it is per-split-part (one
        // performElem2 call == one part). The subkey probe reads no rule partition
        // (D-302), so a part counts every submatch its own enumeration visits; the
        // split stump partitions the enumeration itself (I-156). An empty stump (the CE
        // filter) runs uncapped and unsplit, so nothing reads its tally.
        // thread_local: no shared write / race. See D-109.
        static thread_local int64_t g_growthMatchCount;

        // Observation-only GPU growth-capacity census. Each worker task resets
        // these fixed depth arrays, the request generator and stump producer add
        // their local node counts without contention, and proveKernel publishes
        // them after the task returns. No proof branch reads them.
        static thread_local uint64_t g_gpuGrowAttemptsByDepth[
            ExecutionParameters::MAX_EXPRESSIONS + 1];
        static thread_local uint64_t g_gpuGrowFrontierByDepth[
            ExecutionParameters::MAX_EXPRESSIONS + 1];
        static thread_local uint64_t g_gpuGrowSubkeysByDepth[
            ExecutionParameters::MAX_EXPRESSIONS + 1];
        static thread_local uint64_t g_gpuGrowRequestsByDepth[
            ExecutionParameters::MAX_EXPRESSIONS + 1];
        static thread_local uint64_t g_gpuProducerAttemptsByDepth[
            ExecutionParameters::MAX_EXPRESSIONS + 1];
        static thread_local uint64_t g_gpuProducerSurvivorsByDepth[
            ExecutionParameters::MAX_EXPRESSIONS + 1];


};
}
