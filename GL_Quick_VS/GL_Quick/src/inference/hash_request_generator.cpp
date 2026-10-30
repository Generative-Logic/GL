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
#include "hash_request_generator.hpp"
#include "burst_sink.hpp"
#include "../proof/worker_context.hpp"
#include "../memory_infra/paged_sorted_ordinals.hpp"
#include "../memory_infra/arena_stack.hpp"
namespace gl {
thread_local int64_t HashRequestGenerator::g_growthMatchCount = 0;
thread_local uint64_t HashRequestGenerator::g_gpuGrowAttemptsByDepth[
    ExecutionParameters::MAX_EXPRESSIONS + 1]{};
thread_local uint64_t HashRequestGenerator::g_gpuGrowFrontierByDepth[
    ExecutionParameters::MAX_EXPRESSIONS + 1]{};
thread_local uint64_t HashRequestGenerator::g_gpuGrowSubkeysByDepth[
    ExecutionParameters::MAX_EXPRESSIONS + 1]{};
thread_local uint64_t HashRequestGenerator::g_gpuGrowRequestsByDepth[
    ExecutionParameters::MAX_EXPRESSIONS + 1]{};
thread_local uint64_t HashRequestGenerator::g_gpuProducerAttemptsByDepth[
    ExecutionParameters::MAX_EXPRESSIONS + 1]{};
thread_local uint64_t HashRequestGenerator::g_gpuProducerSurvivorsByDepth[
    ExecutionParameters::MAX_EXPRESSIONS + 1]{};

/// @brief Bind HashRequestGenerator inputs.
/// @details Binds the existing inputs without performing proof work or moving storage.
/// @param parameters Existing parameters interface or settings.
/// @return None; construction binds the supplied references.
HashRequestGenerator::HashRequestGenerator(const ProverParameters& parameters) : parameters(parameters) {}

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
NameId HashRequestGenerator::filterIntEncodedStatements(
        IntStmtView stmts,
        const HashMemory& mem, const Memory& body,
        bool alsoAcceptFullKeys,
        NameId* outIndices, NameId maxOut) {

        const NameId count = static_cast<NameId>(stmts.size());
        NameId buf[ExecutionParameters::MAX_KEY_SLOTS];
        NameId nOut = 0;
        // Frozen or-branch subtree exclusion (I-206): a
        // statement whose scope is a frozen `_ordis_` branch or any descendant
        // of one is not used to build requests. Pure reads of the persistent
        // frozen set and the NameMap parent forest (I-83); the empty fast path
        // keeps the common no-frozen case at one branch per call.
        const bool anyFrozen = !body.frozenOrBranches.empty();

        for (NameId i = 0; i < count && nOut < maxOut; ++i) {
            const IntEncodedExpr& s = stmts[i];
            if (anyFrozen) {
                bool frozen = false;
                for (NameId sc = s.validityId; sc != 0;
                     sc = body.nameMap.parentOf(sc)) {
                    if (body.frozenOrBranches.lookup(sc) != 0) {
                        frozen = true;
                        break;
                    }
                }
                if (frozen) continue;
            }

            // Build single-expr IntNormalizedKey on stack
            NameId pos = 0;
            buf[pos++] = s.nameId;
            buf[pos++] = s.negation;
            for (NameId j = 0; j < s.arity; ++j) {
                buf[pos++] = s.argId[j];
                buf[pos++] = 0; // changeable (ignoreU=false)
            }
            // Normalize: sequential IDs by first appearance
            {
                NameId varMap[ExecutionParameters::MAX_KEY_SLOTS];
                NameId nV = 0;
                NameId nextN = 1;
                // Start after nameId+negation (pos 2), step by 2 (varId, changeable)
                for (NameId p = 2; p < pos; p += 2) {
                    NameId raw = buf[p];
                    NameId norm = 0;
                    for (NameId v = 0; v < nV; ++v) {
                        if (varMap[v * 2] == raw) { norm = varMap[v * 2 + 1]; break; }
                    }
                    if (norm == 0) {
                        norm = nextN++;
                        varMap[nV * 2] = raw;
                        varMap[nV * 2 + 1] = norm;
                        ++nV;
                    }
                    buf[p] = norm;
                }
            }

            // Growth half: keep the statement when its single-element subkey is
            // present and some owner's u_ signature admits it (D-120); scope
            // comparability and partition are not read here.
            const IntEncodedExpr* sp = &s;
            const bool subOk =
                subkeyUSatisfied(mem.normalizedEncodedSubkeys, buf, pos, &sp, 1);
            // Empty stump: a statement may be a whole request on its own, so a
            // present whole key also accepts it (presence only; the owner
            // record is not read).
            const bool keyOk =
                alsoAcceptFullKeys
                && gl::HashRuleIndex::wholeKeyPresent(mem.normalizedEncodedKeys, buf, pos, 1);
            if (!subOk && !keyOk)
                continue;
            if (s.maxIteration > parameters.maxIterationNumberVariable)
                continue;
            // Premise-pattern condition: some live rule premise of the LB
            // accepts the row on its u_ literals, repetitions and scope — the
            // same union the GPU upload prefilter applies, so both routes keep
            // one row list.
            if (!premisePatternAdmits(body, s))
                continue;

            outIndices[nOut++] = i;
        }
        return nOut;
    }

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
    void HashRequestGenerator::generateEncodedRequestsStatic(
        const Memory& body,
        const HashMemory& intMemory,
        const MandatoryTerm* terms, NameId termCount,
        const SplitStumpRef& splitStump,
        unsigned coreId,
        Consumer& consumer) {
        RT_SCOPE_HERE("GENERATE_ENCODED_REQUESTS_STATIC");
        // This sub-part's bucket of stumps: the search below runs once per stump,
        // over one shared filtered statement list.
        const ExpressionStump* const bucket = splitStump.stumps;
        const NameId bucketCount = splitStump.count;
        assert((bucketCount == 0) == (bucket == nullptr)
            && "generateEncodedRequestsStatic: stump bucket present iff non-empty");
        assert((bucketCount == 0
                || (splitStump.total >= 1
                    && splitStump.ordinal >= 0
                    && splitStump.ordinal < splitStump.total))
            && "generateEncodedRequestsStatic: a stump sub-part has a place among "
               "its siblings");
        // The mandatory-containment control: a request must contain a statement
        // from every view of at least ONE term.
        //
        // An EMPTY term list is a first-class input, not the absence of one: it
        // says no ingredient is mandatory, so every whole key is a request. Two
        // callers use it — the counter-example filter, and the batch whose rule
        // registry is itself the new thing (`localHashMemoryDelta`), which has to
        // meet every visible statement rather than only this burst's. The second
        // one IS stump-split, so nothing here may infer "termless" ⇒ "unsplit";
        // the counter-example call site asserts its own unsplit-ness, where that
        // is still true.
        const bool containmentMode = (termCount > 0);
        assert(termCount >= 0 && termCount <= kMaxMandatoryTerms
            && "generateEncodedRequestsStatic: at most kMaxMandatoryTerms terms");
        assert(((termCount == 0) == (terms == nullptr))
            && "generateEncodedRequestsStatic: term array present iff termCount > 0");

        const NameMap& nm = body.nameMap;
        const NameId mainValidityId = NameMap::MAIN_ID;
        const int targetLen = intMemory.maxKeyLength;

        // Which normalized-key index decides that a candidate is RECORDED. The candidate
        // is the whole request, so it has to be a whole key.
        const TypedColdBlobMap<NormKey, RuleOwnerRec>& targetKeys =
            intMemory.normalizedEncodedKeys;

        // The request keys + the IntEncodedExpr copies ride this slot's gen scratch
        // arena byte-bump tier (no per-thread heap arena); persistent per task,
        // freed by the per-task releaseAll.
        ScratchArena& genArena = genScratchArenas().forSlot(coreId);
        StaticRequestEmitter<Consumer> emitter(genArena, consumer);

        NameId buf[ExecutionParameters::MAX_KEY_SLOTS];

        // Copy an accepted key off the stack onto the arena, where a StaticRequest
        // can point at it for the whole task.
        const auto arenaKey = [&genArena](const NameId* src, NameId len) -> NameId* {
            NameId* dst = reinterpret_cast<NameId*>(genArena.resolve(
                genArena.alloc(len * static_cast<int32_t>(sizeof(NameId)),
                               static_cast<int32_t>(alignof(NameId)))));
            std::memcpy(dst, src, static_cast<std::size_t>(len) * sizeof(NameId));
            return dst;
        };

        if (targetLen <= 0) return;

        // ---------------------------------------------------------------
        // The grow search. Filter the statement universe, name-sort it, then
        // depth-first extend the empty candidate up to targetLen elements,
        // emitting every candidate that lands on a whole key and carries a
        // mandatory ingredient.
        //
        // Under a stump split the search runs ONCE PER STUMP in this sub-part's
        // bucket, over this one filtered list -- which is why a bucket exists: the
        // filter is the fixed cost, and it is paid once for the whole bucket.
        //
        // A stump NAMES A NODE of this same search: the run seeds its stack with
        // that node and grows it onward, children starting past the stump's last
        // element exactly as the unsplit search grows them. A run therefore owns
        // that node's SUBTREE and nothing else. Because the search extends in
        // ascending list order every candidate has one path, so the stumps'
        // subtrees are disjoint and their union is the whole unsplit enumeration:
        // the split is an exact partition, not a filter. Split and unsplit totals
        // are identical.
        // ---------------------------------------------------------------
        const IntStmtView allIntStmts(body.intEncodedStatements);
        assert(parameters.maxNumberFilteredStatements > 0
            && parameters.maxNumberFilteredStatements <= ExecutionParameters::MAX_FILTERED_POSITION_CODE);
        NameId* filteredIdx = reinterpret_cast<NameId*>(genArena.resolve(genArena.alloc(
            parameters.maxNumberFilteredStatements * static_cast<int32_t>(sizeof(NameId)),
            static_cast<int32_t>(alignof(NameId)))));
        NameId nFiltered = 0;
        {
        // RT part 1 of 2: statement-universe filter + name sort — the fixed
        // per-call cost the whole bucket shares.
        RT_SCOPE_HERE("STATIC_REQGEN_FILTER_SORT");
        // A candidate is the whole request, so a statement that is a whole key on
        // its own is admitted alongside the growable subkeys.
        nFiltered = filterIntEncodedStatements(allIntStmts,
            intMemory, body, /*alsoAcceptFullKeys=*/true, filteredIdx, parameters.maxNumberFilteredStatements);
        // Name-only stable_sort. Emergence-order tie resolution is deterministic by
        // the stable_sort contract across MSVC STL and libstdc++ — cross-host
        // byte-identical at this site. The Gauss / fold theorem proves under this
        // tie order once the integration-side admission machinery closed the
        // asymmetry that had previously made the fold proof's search path
        // tie-order-sensitive.
        std::stable_sort(filteredIdx, filteredIdx + nFiltered, [&](NameId a, NameId b) {
            return compareSpans(nm.decodeView(allIntStmts[a].nameId),
                                nm.decodeView(allIntStmts[b].nameId)) < 0;
        });
        gpuRequestFilterCalls.fetch_add(1, std::memory_order_relaxed);
        gpuFilterInputStatements.fetch_add(
            static_cast<uint64_t>(allIntStmts.size()),
            std::memory_order_relaxed);
        gpuFilterOutputStatements.fetch_add(
            static_cast<uint64_t>(nFiltered), std::memory_order_relaxed);
        const auto publishMaximum = [](std::atomic<uint64_t>& maximum,
                                       uint64_t value) {
            uint64_t observed = maximum.load(std::memory_order_relaxed);
            while (observed < value
                && !maximum.compare_exchange_weak(
                    observed, value,
                    std::memory_order_relaxed,
                    std::memory_order_relaxed)) {
            }
        };
        publishMaximum(gpuFilterMaximumInputStatements,
            static_cast<uint64_t>(allIntStmts.size()));
        publishMaximum(gpuFilterMaximumOutputStatements,
            static_cast<uint64_t>(nFiltered));
        }

        // ---------------------------------------------------------------
        // Mandatory containment: turn view membership into one bit per filtered
        // statement, and precompute what each suffix of the sorted list can still
        // contribute. Bit (t, s) marks membership in term t's view s, so a term is
        // satisfied by a mask that carries every one of its bits.
        //
        // All three arrays ride this slot's gen scratch arena byte-bump tier and
        // are allocated BEFORE the DFS stack's first push, so they sit below every
        // stack frame and no `pop` can reclaim them.
        // ---------------------------------------------------------------
        uint8_t termMask[kMaxMandatoryTerms] = { 0, 0 };
        const uint8_t* viewMask = nullptr;    // per filtered position
        const uint8_t* suffixMask = nullptr;  // views live at or after a position
        if (containmentMode) {
            for (NameId t = 0; t < termCount; ++t) {
                assert(terms[t].viewCount >= 1 && terms[t].viewCount <= kMaxTermViews
                    && "generateEncodedRequestsStatic: a mandatory term names one "
                       "or two statement views");
                for (NameId s = 0; s < terms[t].viewCount; ++s)
                    termMask[t] |= static_cast<uint8_t>(
                        1u << static_cast<unsigned>(t * kMaxTermViews + s));
            }

            // Intern the filtered universe by (originalId, validityId) so a view
            // element finds its filtered position in O(1). Two filtered rows can
            // share a key, so the bits are collected per interned id and read back
            // per position — no assumption that ids and positions coincide.
            DirtyState maskDirty = DirtyState::Clean;
            ColdHashSet<PodKeyStore<int64_t>> filteredKeys(&genArena, &maskDirty);
            int32_t* keyId = reinterpret_cast<int32_t*>(genArena.resolve(
                genArena.alloc(
                    static_cast<int32_t>(nFiltered + 1) * static_cast<int32_t>(sizeof(int32_t)),
                    static_cast<int32_t>(alignof(int32_t)))));
            for (NameId k = 0; k < nFiltered; ++k) {
                const IntEncodedExpr& fe = allIntStmts[filteredIdx[k]];
                keyId[k] = filteredKeys.mint(
                    packStatementKey(fe.originalId, fe.validityId));
            }

            uint8_t* bitsById = reinterpret_cast<uint8_t*>(genArena.resolve(
                genArena.alloc(static_cast<int32_t>(nFiltered + 1), 1)));
            std::memset(bitsById, 0, static_cast<std::size_t>(nFiltered) + 1);
            for (NameId t = 0; t < termCount; ++t) {
                for (NameId s = 0; s < terms[t].viewCount; ++s) {
                    const uint8_t bit = static_cast<uint8_t>(
                        1u << static_cast<unsigned>(t * kMaxTermViews + s));
                    const IntStmtView& view = terms[t].views[s];
                    for (int32_t j = 0; j < view.size(); ++j) {
                        const int32_t id = filteredKeys.lookup(
                            packStatementKey(view[j].originalId, view[j].validityId));
                        // A miss means the view's statement is not a viable request
                        // ingredient at all (the filter dropped it) — the retired
                        // stump builders filtered their source views the same way.
                        if (id != 0) bitsById[id] |= bit;
                    }
                }
            }

            uint8_t* vm = reinterpret_cast<uint8_t*>(genArena.resolve(
                genArena.alloc(static_cast<int32_t>(nFiltered + 1), 1)));
            uint8_t* sm = reinterpret_cast<uint8_t*>(genArena.resolve(
                genArena.alloc(static_cast<int32_t>(nFiltered + 1), 1)));
            sm[nFiltered] = 0;
            for (int k = static_cast<int>(nFiltered) - 1; k >= 0; --k) {
                vm[k] = bitsById[keyId[k]];
                sm[k] = static_cast<uint8_t>(sm[k + 1] | vm[k]);
            }
            viewMask = vm;
            suffixMask = sm;
        }

        // Does this mask already satisfy some term — is the candidate emittable?
        const auto termsSatisfied = [&](uint8_t m) -> bool {
            for (NameId t = 0; t < termCount; ++t)
                if ((m & termMask[t]) == termMask[t]) return true;
            return false;
        };

        // Can some term still be completed from here: either it is satisfied
        // already, or there is depth left AND every view it still misses is live
        // somewhere at or after `from` in the sorted list. Conservative on purpose
        // — one statement may carry two of a term's views at once, so the count of
        // missing views is not a depth bound.
        const auto termsReachable =
            [&](uint8_t m, NameId count, int from) -> bool {
            for (NameId t = 0; t < termCount; ++t) {
                const uint8_t missing = static_cast<uint8_t>(termMask[t] & ~m);
                if (missing == 0) return true;
                if (count >= targetLen) continue;
                if ((missing & suffixMask[from]) == missing) return true;
            }
            return false;
        };

        struct StackItem {
            int start;
            NameId allIdx[ExecutionParameters::MAX_EXPRESSIONS];
            NameId count;
            NameId validityId;
            uint8_t mask;   // union of the term-view bits this candidate carries
        };

        // The stump of the run in progress. Its elements ascend by (decoded name,
        // statement index), the same total order the filtered list is sorted into,
        // so the union with a candidate is one linear merge.
        const NameId* curIdx = nullptr;
        NameId curCount = 0;
        NameId curVid = mainValidityId;
        // The union of the term-view bits the run's stump already carries.
        uint8_t curMask = 0;

        // Position of a statement in the name-sorted filtered list, or -1 when the
        // statement is not in it. A stump that survives the subkey gate names a node
        // of THIS search's own enumeration, so each of its elements is a filtered
        // statement: the subkey map is closed downward, so a stump the map accepts
        // has every element accepted, and the filter keeps exactly those (same
        // maxIteration gate the producer applied). A stump the gate rejects — which
        // is what a reference memory narrower than the producer's yields — is
        // skipped before the seed frame, so -1 never reaches it.
        const auto posInFiltered = [&](NameId stmtIdx) -> int {
            for (NameId i = 0; i < nFiltered; ++i)
                if (filteredIdx[i] == stmtIdx) return static_cast<int>(i);
            return -1;
        };

        const IntEncodedExpr* candPtrs[ExecutionParameters::MAX_EXPRESSIONS];

        // The resumable key build behind `buf` and the resumable request gates.
        // Function-scope so the DFS pays one stack reservation for the whole call
        // rather than one per pop.
        gl::HashKeyBuilder keyState;
        RequestGateState gateState;
        const TypedColdSet<NameId>& prodRecIds =
            body.overallHashMemory.productsOfRecursionIds;

        // The DFS frontier rides the byte-bump tier: grow on push, reclaim on
        // backtrack via popTo, so the footprint tracks the live frontier, not the
        // total nodes explored. Reused across the bucket's runs — each run drains it.
        ArenaStack<StackItem> stack(genArena);

        // One search per stump; one search with no stump when the LB is not
        // stump-split.
        const NameId runCount = (bucketCount > 0) ? bucketCount : 1;
        {
        // RT part 2 of 2: the grow search — DFS candidate enumeration with its
        // per-node gates and owner-set probes. A recorded candidate is emitted
        // right here, inside the scope, which is what preserves the CE filter's
        // contradiction early-exit.
        // iter = the filtered statement universe the search grows over.
        RT_SCOPE_HERE("STATIC_REQGEN_GROW_SEARCH");
        RT_NOTE_ITERATIONS_HERE(static_cast<int>(nFiltered));
        for (NameId si = 0; si < runCount; ++si) {
            const bool terminalOnly = bucketCount > 0
                && bucket[si].terminalOnly != 0;
            if (bucketCount > 0) {
                curIdx = bucket[si].allIdx;
                curCount = bucket[si].count;
                assert(bucket[si].terminalOnly <= 1
                    && "an expression stump's terminal-only marker is bivalent");
                assert(curCount > 0
                    && curCount <= ExecutionParameters::MAX_EXPRESSIONS
                    && "a stump is between one expression and a whole key long");
                // The stump is part of every base candidate, so a stump that alone
                // fills the grow depth leaves no room for one. Nothing to generate
                // for this stump; the bucket's others may still have room.
                if (curCount > targetLen) continue;
                // Its scope is the fold of its elements' scopes; the producer built
                // it under that comparability, so the fold cannot fail.
                curVid = mainValidityId;
                for (NameId k = 0; k < curCount; ++k) {
                    const IntEncodedExpr& se = allIntStmts[curIdx[k]];
                    assert((k == 0 || nm.comparable(curVid, se.validityId))
                        && "split stump elements must be pairwise scope-comparable");
                    curVid = (k == 0) ? se.validityId
                                      : nm.deeperOf(curVid, se.validityId);
                }
                // The containment control needs the stump's view bits BEFORE the
                // stump-alone probe below, which may emit it — so the positions
                // are resolved here rather than in the seed frame.
                //
                // An element may be legitimately absent from this call's filtered
                // universe. The producer grew its stumps over the LB's OVERALL rule
                // registry, and a batch reading a narrower one (working memory, the
                // local registries) may hold no rule that could ever use that
                // statement. Such a stump yields nothing here and is skipped: the
                // subkey map is closed downward, so a set containing a non-subkey is
                // itself a non-subkey, which makes the stump fail both the growth
                // and the record probe — and so does every candidate in its subtree,
                // since each one still contains that element. The stumped modes
                // reach the same verdict one probe later, at the subkey gate below.
                curMask = 0;
                bool stumpIsUsableHere = true;
                if (containmentMode) {
                    for (NameId k = 0; k < curCount; ++k) {
                        const int pos = posInFiltered(curIdx[k]);
                        if (pos < 0) { stumpIsUsableHere = false; break; }
                        curMask = static_cast<uint8_t>(curMask | viewMask[pos]);
                    }
                }
                if (!stumpIsUsableHere) continue;
            }

            // The stump alone is the one base candidate no growing candidate can
            // reproduce: unsplit, it was recorded inside the loop of the candidate
            // one level up, and this sub-part never runs that loop. Probe it exactly
            // as the search probes a node, tally bump included.
            //
            // Both bail-outs skip to the next stump. A stump the gates reject can
            // never sit inside a passing request (the gates are closed downward
            // under subsets), and a stump the subkey map rejects cannot grow (every
            // superset of a non-subkey is a non-subkey).
            if (curCount > 0) {
                for (NameId k = 0; k < curCount; ++k)
                    candPtrs[k] = &allIntStmts[curIdx[k]];
                if (!requestGatesPass(candPtrs, curCount, body, mainValidityId))
                    continue;
                const NameId keyLen = gl::HashKeyBuilder::makeIntNormalizedKeyFromEncoded(candPtrs,
                    curCount, buf, ExecutionParameters::MAX_KEY_SLOTS);
                const bool subOk = subkeyUSatisfied(intMemory.normalizedEncodedSubkeys,
                                                    buf, keyLen, candPtrs, curCount);
                if (subOk) {
                    ++g_growthMatchCount;
                    ++g_gpuGrowSubkeysByDepth[curCount];
                }
                // Record gate, mirroring the search's. The candidate is already the
                // finished request and nothing more will be added to it, so it need
                // not be growable — only a present whole key (the owner record is
                // not read), and only carrying a term's ingredients when there are
                // terms.
                if (gl::HashRuleIndex::wholeKeyPresent(targetKeys, buf, keyLen, curCount)
                    && (!containmentMode || termsSatisfied(curMask))) {
                    ++g_gpuGrowRequestsByDepth[curCount];
                    IntNormalizedKey nk(curCount, arenaKey(buf, keyLen), keyLen);
                    if (!emitter.emit(candPtrs, curCount, nk)) return;
                }
                if (!subOk) continue;
            }

            // A terminal pre-stump is a recordable shallow producer node whose
            // level was replaced by its children. Its own probe above is the only
            // work it owns; growing it would overlap those children.
            if (terminalOnly) continue;

            {
                StackItem init;
                init.start = 0;
                init.count = 0;
                init.validityId = curVid;
                init.mask = 0;
                if (curCount > 0) {
                    // The stump IS the growing candidate. This run resumes the
                    // unsplit search at the node the stump names and grows it
                    // onward; children start past the stump's last element, so the
                    // run owns exactly that node's subtree and no sibling bucket
                    // can reach into it. Ascending positions are the producer's own
                    // enumeration order — it grows a child at pos + 1.
                    int lastPos = -1;
                    for (NameId k = 0; k < curCount; ++k) {
                        const int pos = posInFiltered(curIdx[k]);
                        assert(pos > lastPos
                            && "a split stump's elements must be filtered statements "
                               "in ascending list order");
                        lastPos = pos;
                        init.allIdx[k] = curIdx[k];
                    }
                    init.count = curCount;
                    init.start = lastPos + 1;
                    init.mask = curMask;
                }
                stack.push(init);
                ++g_gpuGrowFrontierByDepth[curCount];
            }

            while (!stack.empty()) {
                // Per-new-seed checkpoint: each stack pop starts exploring one
                // growing seed. Refresh the on-disk snapshot so a mid-burst reader
                // sees which seed level the grow loop is on.
                RT_REFRESH_HERE();

                StackItem top = stack.back();
                stack.pop();

                // The prefix is the same for every position this pop visits: only
                // the appended slot moves. Its statement indices and its premise
                // pointers are therefore written once here, and the loop below
                // writes slot `top.count` alone.
                NameId cand[ExecutionParameters::MAX_EXPRESSIONS];
                std::memcpy(cand, top.allIdx, top.count * sizeof(NameId));
                for (NameId k = 0; k < top.count; ++k)
                    candPtrs[k] = &allIntStmts[top.allIdx[k]];
                // Those pointers resolve through the LB's statement vector, so they
                // stay valid for the whole pop only because phase 2 never writes
                // shared LB state (I-83).
                assert(body.intEncodedStatements.size() == allIntStmts.size()
                    && "the statement universe must not change during a grow pop "
                       "— the hoisted premise pointers resolve through it");

                // The prefix's normalized key, built once for the same reason.
                // Its bytes stay in the written prefix of `buf` for the whole pop —
                // nothing in the loop writes that buffer — so each position only
                // rewinds the three counters and folds in its own premise.
                keyState.buildPrefix(candPtrs, top.count, buf, ExecutionParameters::MAX_KEY_SLOTS);
                const auto prefixCheckpoint = keyState.checkpoint();

                // The prefix's gate summary, same story: the hypothesis scope,
                // the non-exempt scope count and the distinct secondaries are all
                // order-free over the premises, so the prefix folds once and each
                // position folds in its own premise alone. That is what keeps the
                // per-argument productsOfRecursionIds probes off the prefix.
                foldPrefixIntoRequestGates(candPtrs, top.count, prodRecIds,
                                           mainValidityId, gateState);
                const int32_t prefixHypoFound = gateState.hypoFound;
                const NameId prefixHypoValidityId = gateState.hypoValidityId;
                const int32_t prefixNonExemptScopes = gateState.nonExemptScopes;
                const NameId prefixNonExemptValidityId =
                    gateState.nonExemptValidityId;
                const int32_t prefixSecondaryCount = gateState.secondaryCount;

                for (int i = top.start; i < nFiltered; ++i) {
                    // The stump sits inside top.count, so this is the plain
                    // unsplit depth test.
                    if (top.count + 1 > targetLen) break;
                    // Submatch cap (D-109). canAccept bypasses the cap for an empty
                    // stump (I-73) and honors this LB's early-exit stop when it runs
                    // unsplit (I-76); under split it never observes a sibling's stop.
                    if (!consumer.canAccept()) return;

                    const NameId allIdx = filteredIdx[i];
                    const IntEncodedExpr& ie = allIntStmts[allIdx];
                    const NameId newCount = static_cast<NameId>(top.count + 1);
                    ++g_gpuGrowAttemptsByDepth[newCount];

                    // Mandatory containment, cheapest gate first: skip a position
                    // that neither finishes a term here nor leaves one completable
                    // deeper. Nothing else happens at such a position — it can be
                    // neither emitted nor grown — so skipping it costs no request
                    // and no submatch.
                    const uint8_t newMask = containmentMode
                        ? static_cast<uint8_t>(top.mask | viewMask[i])
                        : static_cast<uint8_t>(0);
                    if (containmentMode
                        && !termsReachable(newMask, newCount, i + 1)) continue;

                    if (!nm.comparable(top.validityId, ie.validityId)) continue;
                    const NameId newValidityId = nm.deeperOf(top.validityId, ie.validityId);

                    // The growing candidate. The stump's elements are already its
                    // prefix, written above; this position contributes the last
                    // element only.
                    cand[top.count] = allIdx;
                    candPtrs[top.count] = &ie;

                    // The map-independent request-shape gates, then the key. Both
                    // maps below are probed with this one key. Rewind to the
                    // prefix summary, fold in this position's premise, then take
                    // the verdict over the whole candidate.
                    gateState.hypoFound = prefixHypoFound;
                    gateState.hypoValidityId = prefixHypoValidityId;
                    gateState.nonExemptScopes = prefixNonExemptScopes;
                    gateState.nonExemptValidityId = prefixNonExemptValidityId;
                    gateState.secondaryCount = prefixSecondaryCount;
                    if (!foldExprIntoRequestGates(ie, gateState, prodRecIds,
                                                  mainValidityId)) {
                        continue;
                    }
                    if (!requestGateStateAccepts(gateState, candPtrs, newCount,
                                                 body, mainValidityId)) {
                        continue;
                    }
                    // Rewind to the prefix, then fold in this position's premise.
                    // Byte-identical to rebuilding the whole key: the renumbering
                    // is first-appearance order, so the prefix's slot assignments
                    // are exactly what a full rebuild would have reached here.
                    keyState.restore(prefixCheckpoint);
                    keyState.append(ie, buf, ExecutionParameters::MAX_KEY_SLOTS);
                    const NameId keyLen = keyState.length();

                    // Growth probe: may this candidate be extended? Key presence
                    // plus the u_ signature (D-120) only — scope and partition are
                    // not read. These are the submatches the doom line and the
                    // split policy count.
                    const bool subOk = subkeyUSatisfied(intMemory.normalizedEncodedSubkeys,
                                                        buf, keyLen, candPtrs, newCount);
                    if (subOk) {
                        ++g_growthMatchCount;
                        ++g_gpuGrowSubkeysByDepth[newCount];
                    }
                    // Record probe: is this candidate a present whole key (presence
                    // only, the owner record is not read)? It is the finished
                    // request, so it need not also be growable. Emit here, inside
                    // the search, so a refuting head can halt the burst — and, under
                    // the containment control, only once the candidate actually
                    // carries a term's ingredients.
                    if (gl::HashRuleIndex::wholeKeyPresent(targetKeys, buf, keyLen, newCount)
                        && (!containmentMode || termsSatisfied(newMask))) {
                        ++g_gpuGrowRequestsByDepth[newCount];
                        IntNormalizedKey nk(newCount, arenaKey(buf, keyLen), keyLen);
                        if (!emitter.emit(candPtrs, newCount, nk)) return;
                    }

                    if (subOk && newCount < targetLen) {
                        StackItem next;
                        next.start = i + 1;
                        std::memcpy(next.allIdx, cand, newCount * sizeof(NameId));
                        next.count = newCount;
                        next.validityId = newValidityId;
                        next.mask = newMask;
                        stack.push(next);
                        ++g_gpuGrowFrontierByDepth[newCount];
                    }
                }
            }
        }
        }

    }

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
int32_t HashRequestGenerator::produceExpressionStumps(
        const Memory& body, unsigned coreId, int32_t target, SealedPageSet& out) {
        assert(target > 0
            && "produceExpressionStumps: the stump-count floor must be positive");

        const NameMap& nm = body.nameMap;
        const HashMemory& mem = body.overallHashMemory;
        const IntStmtView allIntStmts(body.intEncodedStatements);
        ScratchArena& genArena = genScratchArenas().forSlot(coreId);

        // The statement universe, filtered by the subkey growth probe, then
        // name-sorted exactly as the search sorts it (ties keep ascending
        // statement index).
        assert(parameters.maxNumberFilteredStatements > 0
            && parameters.maxNumberFilteredStatements <= ExecutionParameters::MAX_FILTERED_POSITION_CODE);
        NameId* filteredIdx = reinterpret_cast<NameId*>(genArena.resolve(genArena.alloc(
            parameters.maxNumberFilteredStatements * static_cast<int32_t>(sizeof(NameId)),
            static_cast<int32_t>(alignof(NameId)))));
        const NameId nFiltered = filterIntEncodedStatements(allIntStmts,
            mem, body, /*alsoAcceptFullKeys=*/false, filteredIdx, parameters.maxNumberFilteredStatements);
        gpuProducerFilterCalls.fetch_add(1, std::memory_order_relaxed);
        gpuFilterInputStatements.fetch_add(
            static_cast<uint64_t>(allIntStmts.size()),
            std::memory_order_relaxed);
        gpuFilterOutputStatements.fetch_add(
            static_cast<uint64_t>(nFiltered), std::memory_order_relaxed);
        const auto publishMaximum = [](std::atomic<uint64_t>& maximum,
                                       uint64_t value) {
            uint64_t observed = maximum.load(std::memory_order_relaxed);
            while (observed < value
                && !maximum.compare_exchange_weak(
                    observed, value,
                    std::memory_order_relaxed,
                    std::memory_order_relaxed)) {
            }
        };
        publishMaximum(gpuFilterMaximumInputStatements,
            static_cast<uint64_t>(allIntStmts.size()));
        publishMaximum(gpuFilterMaximumOutputStatements,
            static_cast<uint64_t>(nFiltered));
        if (nFiltered == 0) return 0;
        std::stable_sort(filteredIdx, filteredIdx + nFiltered, [&](NameId a, NameId b) {
            return compareSpans(nm.decodeView(allIntStmts[a].nameId),
                                nm.decodeView(allIntStmts[b].nameId)) < 0;
        });

        // One append-only column holds every level; [levelBegin, levelEnd) is the
        // current one. PAGE tier of this slot's gen scratch arena, released with
        // the task.
        struct Node {
            NameId pos[ExecutionParameters::MAX_EXPRESSIONS];
            NameId count;
            NameId validityId;
        };
        DirtyState nodesDirty = DirtyState::Clean;
        PagedVector<Node> nodes(&genArena, &nodesDirty);
        for (NameId i = 0; i < nFiltered; ++i) {
            Node n;
            n.pos[0] = i;
            n.count = 1;
            n.validityId = allIntStmts[filteredIdx[i]].validityId;
            nodes.push_back(n);
        }
        g_gpuProducerSurvivorsByDepth[1] +=
            static_cast<uint64_t>(nFiltered);
        int32_t levelBegin = 0;
        int32_t levelEnd = nodes.size();

        const NameId maxStumpLen = static_cast<NameId>(std::min<int>(
            ExecutionParameters::MAX_EXPRESSIONS, mem.maxKeyLength));

        NameId buf[ExecutionParameters::MAX_KEY_SLOTS];
        const IntEncodedExpr* ptrs[ExecutionParameters::MAX_EXPRESSIONS];

        // Build the key of a level node into buf/ptrs; returns the key length.
        const auto keyOf = [&](const Node& n) -> NameId {
            for (NameId k = 0; k < n.count; ++k)
                ptrs[k] = &allIntStmts[filteredIdx[n.pos[k]]];
            return gl::HashKeyBuilder::makeIntNormalizedKeyFromEncoded(ptrs, n.count, buf,
                ExecutionParameters::MAX_KEY_SLOTS);
        };

        const auto appendStump = [&](const Node& n, uint8_t terminalOnly) {
            ExpressionStump s{};
            s.count = n.count;
            s.terminalOnly = terminalOnly;
            for (NameId k = 0; k < n.count; ++k)
                s.allIdx[k] = filteredIdx[n.pos[k]];
            out.appendRecord(s);
        };

        int32_t terminalCount = 0;

        while ((levelEnd - levelBegin) < target) {
            if (nodes[levelBegin].count + 1 > maxStumpLen) break;
            const int32_t nextBegin = nodes.size();
            for (int32_t ni = levelBegin; ni < levelEnd; ++ni) {
                const Node p = nodes[ni];
                for (NameId j = static_cast<NameId>(p.pos[p.count - 1] + 1);
                     j < nFiltered; ++j) {
                    const NameId newCount = static_cast<NameId>(p.count + 1);
                    ++g_gpuProducerAttemptsByDepth[newCount];
                    const IntEncodedExpr& ie = allIntStmts[filteredIdx[j]];
                    if (!nm.comparable(p.validityId, ie.validityId)) continue;
                    for (NameId k = 0; k < p.count; ++k)
                        ptrs[k] = &allIntStmts[filteredIdx[p.pos[k]]];
                    ptrs[p.count] = &ie;
                    if (!requestGatesPass(ptrs, newCount, body, NameMap::MAIN_ID))
                        continue;
                    const NameId keyLen = gl::HashKeyBuilder::makeIntNormalizedKeyFromEncoded(ptrs,
                        newCount, buf, ExecutionParameters::MAX_KEY_SLOTS);
                    // A stump survives the filter (subkey presence + u_ signature)
                    // — nothing more is asked of it.
                    if (!subkeyUSatisfied(mem.normalizedEncodedSubkeys, buf, keyLen,
                                          ptrs, newCount))
                        continue;
                    ++g_gpuProducerSurvivorsByDepth[newCount];
                    Node c = p;
                    c.pos[p.count] = j;
                    c.count = newCount;
                    c.validityId = nm.deeperOf(p.validityId, ie.validityId);
                    nodes.push_back(c);
                }
            }
            const int32_t nextEnd = nodes.size();
            if (nextEnd == nextBegin) break;  // nothing grew: keep this level

            // A level-(L+1) stump covers only base candidates of size >= L+1.
            // Preserve every recordable node of the level being replaced as a
            // terminal pre-stump: its bucket checks the node itself against the
            // batch's actual hash memory but does not grow it. The child level
            // remains the sole owner of all larger base candidates.
            for (int32_t ni = levelBegin; ni < levelEnd; ++ni) {
                const Node& p = nodes[ni];
                const NameId keyLen = keyOf(p);
                // Every batch records against the whole-key map, so one presence
                // probe covers all of them. Missing it would silently drop the
                // node's own requests from every bucket.
                const bool recordable =
                    gl::HashRuleIndex::wholeKeyPresent(mem.normalizedEncodedKeys, buf, keyLen, p.count);
                if (recordable) {
                    appendStump(p, 1);
                    ++terminalCount;
                }
            }
            levelBegin = nextBegin;
            levelEnd = nextEnd;
        }

        const int32_t frontierCount = levelEnd - levelBegin;
        const int32_t count = terminalCount + frontierCount;
        assert(count <= kMaxSplitParts
            && "stump work-item count exceeds kMaxSplitParts — raise the named constant "
               "deliberately, never truncate the stump list");
        for (int32_t ni = levelBegin; ni < levelEnd; ++ni) {
            const Node& n = nodes[ni];
            appendStump(n, 0);
        }
        return count;
    }
template void HashRequestGenerator::generateEncodedRequestsStatic<BurstSink>(
    const Memory&, const HashMemory&, const MandatoryTerm*, NameId,
    const SplitStumpRef&, unsigned, BurstSink&);
}
