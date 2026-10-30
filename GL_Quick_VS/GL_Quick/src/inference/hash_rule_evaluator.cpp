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
#include "hash_rule_evaluator.hpp"
#include "../logical_block/strong_shot_history.hpp"
#include "../proof/worker_context.hpp"
#include "../memory_infra/paged_sorted_ordinals.hpp"
#include "../memory_infra/arena_stack.hpp"
namespace gl {
thread_local GpuEvaluationUsage
    HashRuleEvaluator::g_gpuEvaluationUsage{};

/// @brief Bind HashRuleEvaluator inputs.
/// @details Binds the existing inputs without performing proof work or moving storage.
/// @param parameters Existing parameters interface or settings.
/// @param definitions Existing definitions interface or settings.
/// @param admissionPolicy Existing admissionPolicy interface or settings.
/// @param blockIdentity Existing blockIdentity interface or settings.
/// @return None; construction binds the supplied references.
HashRuleEvaluator::HashRuleEvaluator(const ProverParameters& parameters, const CompiledDefinitionRegistry& definitions, AdmissionPolicy& admissionPolicy, const LogicBlockIdentity& blockIdentity) : parameters(parameters), definitions(definitions), admissionPolicy(admissionPolicy), blockIdentity(blockIdentity) {}

// ------------------------------------------------------------------
    // Hash-engine lookup-and-fire. The largest single body in the cluster
    // (~300 LOC). Consumes one StaticRequest produced by the request generator
    // above and either fires the matched rule (path 1, head LMV) or installs
    // the marker admission (path 2, marker LMV).
    // ------------------------------------------------------------------

    /// @brief Static-pipeline rule lookup — given a `StaticRequest` keyed on
    /// an `IntNormalizedKey`, find matching LMVs in `localHashMemory` and
    /// dispatch the head-firing path.
    ///
    /// @details
    /// This is the consumer of the LMVs that `addToHashMemory` installed.
    /// The function executes int-only fast pre-hit checks before any string
    /// work, then on a hit decodes the relevant pieces and walks the
    /// admission / OR / equivalence-class machinery.
    ///
    /// Phase outline:
    /// 1. **Validity-id consensus.** Iterate `req.intExprs[i]->validityId`
    ///    for each request element; require pairwise `nm.comparable` so a
    ///    consensus deepest scope can be picked. A divergent set returns
    ///    early (the request cannot fire under any single scope).
    /// 2. **encodedMap lookup.** Probe `memoryBlock.localHashMemory.encodedMap[tple]`.
    ///    Each candidate `LocalMemoryValue` is then admitted against:
    ///    - the request's max-iteration (`req.maxIteration + 1` cap),
    ///    - level-set admissibility (`lmv.levels`),
    ///    - the `productOfDisintegration` D-32 marker (gates OR firing),
    ///    - and per-LMV equivalence-class state. The equivalence-class
    ///      gate routes through `applyEquivalenceClass` and emits
    ///      `equality1`/`equality2` per
    ///      [I-32](../../docs/agentic_swdd/30_invariants.md#i-32) /
    ///      [I-34](../../docs/agentic_swdd/30_invariants.md#i-34).
    /// 3. **Head emit.** Each surviving LMV produces a head-firing event:
    ///    `addExprToMemoryBlock` for the head, `mailOut` for any
    ///    cross-scope deposits per [I-26](../../docs/agentic_swdd/30_invariants.md#i-26),
    ///    origin record under the LMV's `justification` tag.
    /// 4. **Admission record-keeping.** Each negative outcome (rejected
    ///    head, exhausted depth) is recorded in `rejectedMap` so the same
    ///    rejection isn't re-emitted next iteration.
    ///
    /// `coreId` is the per-core mailbox identifier; combined with the
    /// receiver's `Memory*` it picks the per-core slot of `mailOut` so
    /// `proveKernel`'s parallel phase does not need cross-LB locks
    /// ([I-28](../../docs/agentic_swdd/30_invariants.md#i-28)).
    ///
    /// @param req          The static request to look up.
    /// @param memoryBlock  Owning LB.
    /// @param coreId       Logical core id of the calling thread.
    ///
    /// @pre  `req.normalizedKey.data` references a live gen scratch arena allocation on the
    ///       per-slot gen scratch arena.
    /// @post `memoryBlock.localHashMemory` may have new origin entries;
    ///       `memoryBlock.mailOut[coreId]` may have new pending deposits;
    ///       `memoryBlock.overallHashMemory.admissionStatusMap` /
    ///       `rejectedMap` may have new rejection records.
    /// @invariant [I-12](../../docs/agentic_swdd/30_invariants.md#i-12) — `addStatement`
    ///            applies equivalence classes to `!(=[a,b])` one-sidedly;
    ///            this function honours that on the head-emit path.
    /// @invariant [I-26](../../docs/agentic_swdd/30_invariants.md#i-26) — mail-out
    ///            implications/statements MAIN-ONLY; exprOriginMap
    ///            ALL-SCOPES.
    /// @invariant [I-28](../../docs/agentic_swdd/30_invariants.md#i-28) — cross-LB
    ///            writes during the parallel phase forbidden; the head
    ///            emit defers to post-`pool.join()` collectors via
    ///            `mailOut`.
    /// @see `inference/hash_rule_evaluator.hpp::addToHashMemory` — installer of the LMVs read here.
    /// @see `inference/hash_rule_evaluator.hpp::addExprToMemoryBlock` — head-emit consumer.
    /// @see `inference/hash_rule_evaluator.hpp::applyEquivalenceClass` — class-gated rewrite.
void HashRuleEvaluator::checkLocalEncodedMemoryStatic(
        const StaticRequest& req,
        const Memory& memoryBlock,
        unsigned coreId,
        SealedPageSet& sealedPages) {
        const NameMap& nm = memoryBlock.nameMap;
        // Record-string writer: every string a FiringRecord carries is
        // copied onto this task's sealed pages and crosses the pool join
        // there (D-164).
        const auto seal = [&sealedPages](const StrSpan& s) {
            return SealedString::copyFrom(sealedPages, s.ptr, s.len);
        };
        // This slot's scratch string arena (one executor per slot at a time;
        // released at performElem2 exit, rewound per hit / per value below).
        // The burst never mints (I-83 / D-116), so NameMap / interner decode
        // references are stable for the whole call and may back spans.
        ScratchArena& scratchArena = scratchArenas().forSlot(coreId);
        // This slot's request-gen scratch arena (T1's second per-slot arena,
        // distinct from the string scratch above). The typed firing-record
        // runs (the level ints, the admv key / remaining-arg SealedString
        // arrays) are assembled here on the byte-bump tier — aligned, so the
        // views are real typed pointers — then sealed onto the page set. All
        // of it is reclaimed in one popTo at the single function exit; no
        // early return follows the mark.
        ScratchArena& genArena = genScratchArenas().forSlot(coreId);
        const NameId reqCount = req.count;
        const int iteration = req.maxIteration + 1;
        const IntNormalizedKey& tple = req.normalizedKey;

        // 1. Validity check using int fields
        const NameId mainValidityId = NameMap::MAIN_ID;
        NameId consensusValidityId = mainValidityId;
        for (NameId i = 0; i < reqCount; ++i) {
            const NameId vid = req.intExprs[i]->validityId;
            if (!nm.comparable(consensusValidityId, vid)) {
                return;
            }
            consensusValidityId = nm.deeperOf(consensusValidityId, vid);
        }

        // Zero-copy view over the cold name bytes — stable for the whole
        // call (the read-only burst never mints, and the LB is resident).
        // The few string-keyed consumers materialize where needed.
        const StrSpan validityView = nm.decodeView(consensusValidityId);

        if (containsSpan(validityView, StrSpan("_hypo_", 6))) {
            for (NameId i = 0; i < reqCount; ++i) {
                if (req.intExprs[i]->validityId != consensusValidityId) {
                    if (!req.intExprs[i]->isAnchor) {
                        return;
                    }
                }
            }
        }

        // Filter scan via int ancestor-walk, mirroring Site H at
        // prover.cpp::addExprToMemoryBlockKernel. An exact-match
        // lookup would catch only the closing scope itself —
        // rule-fires whose consensus scope is a descendant
        // (e.g. _boundary_orint_* / _boundary_ordis_*) must bounce
        // too. Walking ancestorsOf[consensusValidityId] catches the
        // whole implication subtree from the moment the closure
        // inserts the int id into intValidityNamesToFilter.
        for (int32_t ancK = 0, ancN = nm.ancLen(consensusValidityId);
             ancK < ancN; ++ancK) {
            const NameId anc = nm.ancAt(consensusValidityId, ancK);
            if (memoryBlock.intValidityNamesToFilter.contains(anc)) return;
        }

        // 2. productsOfRecursion check using int fields
        int algebraicCounter = 0;
        const auto& prodRecIds = memoryBlock.overallHashMemory.productsOfRecursionIds;
        for (NameId i = 0; i < reqCount; ++i) {
            const IntEncodedExpr& ex = *req.intExprs[i];
            for (NameId a = 0; a < ex.arity; ++a) {
                if (ex.argIteration[a] > -1) {
                    if (!prodRecIds.contains(ex.argFullId[a])) {
                        algebraicCounter++;
                    }
                }
            }
        }
        const bool pure = (algebraicCounter == 0);

        // 3. combinedLevels is built below, after the early returns, onto the
        // gen-scratch arena as a sorted-unique int run (its only consumer is
        // the per-firing level set in the value loop).

        // 4. Build intAllArgs from int fields. Stack array (no heap): a request
        // carries at most (MAX_EXPRESSIONS + 2) premises of arity MAX_ARITY, so
        // the arg ids fit a fixed buffer; sorted once for the binary-search
        // membership in the candidate loop below.
        constexpr int kMaxIntAllArgs =
            (ExecutionParameters::MAX_EXPRESSIONS + 2) * ExecutionParameters::MAX_ARITY;
        NameId intAllArgs[kMaxIntAllArgs];
        int intAllArgsCount = 0;
        for (NameId i = 0; i < reqCount; ++i) {
            const IntEncodedExpr& ex = *req.intExprs[i];
            for (NameId a = 0; a < ex.arity; ++a) {
                assert(intAllArgsCount < kMaxIntAllArgs);
                intAllArgs[intAllArgsCount++] = ex.argFullId[a];
            }
        }
        std::sort(intAllArgs, intAllArgs + intAllArgsCount);

        if (memoryBlock.overallHashMemory.encodedMap.empty()) {
            return;
        }

        // combinedLevels (per request): collect each premise's level run via
        // the packed-key index, sort-unique into one gen-scratch int buffer
        // kept for the whole value loop. The per-firing set folds in the
        // rule's own levels on top of this. Reclaimed by the function-exit
        // popTo below (no early return follows this mark).
        const auto genReqMark = genArena.cursor();
        int32_t levelLvIds[ExecutionParameters::MAX_EXPRESSIONS + 2];
        int combinedRawN = 0;
        for (NameId i = 0; i < reqCount; ++i) {
            levelLvIds[i] = memoryBlock.intStatementLevelsMap.lookup(
                packStatementKey(req.intExprs[i]->originalId,
                                 req.intExprs[i]->validityId));
            if (levelLvIds[i] != 0)
                combinedRawN += memoryBlock.intStatementLevelsMap.runLen(levelLvIds[i]);
        }
        int* combinedBuf = nullptr;
        int combinedCount = 0;
        if (combinedRawN > 0) {
            combinedBuf = reinterpret_cast<int*>(genArena.resolve(
                genArena.alloc(combinedRawN * static_cast<int32_t>(sizeof(int)),
                               static_cast<int32_t>(alignof(int)))));
            int w = 0;
            for (NameId i = 0; i < reqCount; ++i) {
                if (levelLvIds[i] == 0) continue;
                const int32_t rl = memoryBlock.intStatementLevelsMap.runLen(levelLvIds[i]);
                for (int32_t j = 0; j < rl; ++j) {
                    const int lv = memoryBlock.intStatementLevelsMap.valueAt(levelLvIds[i], j);
                    // The {-1} non-derived tier is transparent to level
                    // accounting: it never enters a derivation union.
                    if (lv < 0) continue;
                    combinedBuf[w++] = lv;
                }
            }
            std::sort(combinedBuf, combinedBuf + w);
            combinedCount = static_cast<int>(
                std::unique(combinedBuf, combinedBuf + w) - combinedBuf);
        }

        // 5. Candidate loop: the forward keys whose remaining-arg set is a
        // SUBSET of intAllArgs AND whose stored run CONTAINS the request's
        // normalized key (tpleNorm). The reverse index answers the run-contains
        // half in ONE hash probe (I-154), replacing the
        // former O(keys) forward scan + per-candidate O(run) byte-peek recheck.
        // Surviving ids are enumerated in std::set<NameId> lex order (R1) via a
        // decoded-lex INDEX: int16SetKeyLexCompare is the byte-for-byte twin of
        // std::set<NameId>::operator< (signed element lex, shorter-is-prefix --
        // NOT the count-prefixed Int16SetKey byte order, which compares the
        // count field first). raMap keys are pairwise distinct AND the reverse
        // index yields each owner once -> a tie-free total order -> the identical
        // enumeration sequence to the former scan.
        const auto& raMap = memoryBlock.overallHashMemory.remainingArgsNormalizedEncodedMap;

        // tpleNorm probe bytes (Codec<NormKey>::encode layout: int16
        // numberExpressions, int16 length, length x int16 data) built directly
        // from the request's normalized key -- no heap NormKey. These bytes are
        // BOTH the reverse-index probe key AND the stored blob bytes the reverse
        // index was built from (Codec<NormKey>::serialize == encode), so the
        // reverse answer is EXACTLY {key id : that key's run contains tpleNorm}.
        NameId tpleProbe[2 + ExecutionParameters::MAX_KEY_SLOTS];
        assert(2 + tple.length <= 2 + ExecutionParameters::MAX_KEY_SLOTS);
        tpleProbe[0] = tple.numberExpressions;
        tpleProbe[1] = tple.length;
        std::memcpy(tpleProbe + 2, tple.data,
            static_cast<std::size_t>(tple.length) * sizeof(NameId));
        const int32_t tpleProbeLen =
            (2 + tple.length) * static_cast<int32_t>(sizeof(NameId));

        // Passing candidate ids on the PAGE tier -- the run-contains set is
        // unbounded at Gauss scale, never a stack array. Freed by candIds' own
        // destructor (page tier, disjoint from the byte-bump popTo below).
        DirtyState candDirty = DirtyState::Clean;
        PagedVector<int32_t> candIds(&genArena, &candDirty);
        uint64_t reverseOwnersThisRequest = 0;
        uint64_t candidateOwnersThisRequest = 0;
        memoryBlock.overallHashMemory.remainingArgsReverseIndex.reverseIndexRunOf(
            StrSpan(reinterpret_cast<const char*>(tpleProbe), tpleProbeLen),
            [&](int32_t kid) {
                ++reverseOwnersThisRequest;
                // Apply the EXISTING subset test to the run-contains owners only.
                // Both branches are defined outcomes (keep / drop), not a failure
                // fallback.
                const StrSpan k = raMap.keyAt(kid);
                const NameId cn = int16SetKeyCount(k);
                for (NameId a = 0; a < cn; ++a) {
                    if (!std::binary_search(intAllArgs, intAllArgs + intAllArgsCount,
                                            int16SetKeyIdAt(k, a))) {
                        return;   // not a subset -- drop this owner
                    }
                }
                ++candidateOwnersThisRequest;
                candIds.push_back(kid);
            });
        g_gpuEvaluationUsage.reverseOwners += reverseOwnersThisRequest;
        g_gpuEvaluationUsage.candidateOwners += candidateOwnersThisRequest;
        g_gpuEvaluationUsage.maximumReverseOwnersPerRequest = std::max(
            g_gpuEvaluationUsage.maximumReverseOwnersPerRequest,
            reverseOwnersThisRequest);
        g_gpuEvaluationUsage.maximumCandidateOwnersPerRequest = std::max(
            g_gpuEvaluationUsage.maximumCandidateOwnersPerRequest,
            candidateOwnersThisRequest);
        // Decoded-lex index (std::set<NameId> order == R1) on the gen-scratch
        // byte-bump tier -- coexists with the page-tier candIds (independent
        // substrates). Reclaimed by the function-exit popTo(genReqMark).
        const int32_t candN = candIds.size();
        int32_t* candOrder = candN
            ? reinterpret_cast<int32_t*>(genArena.resolve(genArena.alloc(
                  candN * static_cast<int32_t>(sizeof(int32_t)),
                  static_cast<int32_t>(alignof(int32_t)))))
            : nullptr;
        for (int32_t i = 0; i < candN; ++i) candOrder[i] = i;
        std::sort(candOrder, candOrder + candN, [&](int32_t a, int32_t b) {
            return int16SetKeyLexCompare(raMap.keyAt(candIds[a]),
                                         raMap.keyAt(candIds[b])) < 0;
        });

        for (int32_t oi = 0; oi < candN; ++oi) {
            // Each candidate's run contains tpleNorm by construction (the reverse
            // index answer), so the former per-candidate byte-peek recheck is
            // dropped. The raMap id IS kid (the interner is injective, id in
            // [1,count]).
            const int32_t kid = candIds[candOrder[oi]];

            // HIT -- rare (only rows whose run carries the request's tpleNorm).
            // Reconstruct the unchangeable-arg set for the unchanged
            // makeIntNormalizedKeyFromEncodedWithMap; an edge materialization per
            // HIT, not the per-candidate heap the scan above retired.
            const Int16SetKey candKey = raMap.decodeKey(kid);
            const std::set<NameId> intSt(candKey.ids.begin(), candKey.ids.end());

            NameId buf[ExecutionParameters::MAX_KEY_SLOTS];
            NameId reverseMap[ExecutionParameters::MAX_KEY_SLOTS];
            std::memset(reverseMap, 0, sizeof(reverseMap));
            NameId numNormVars = 0;

            // Build pointer array from request
            const IntEncodedExpr* reqExprs[ExecutionParameters::MAX_EXPRESSIONS];
            for (NameId i = 0; i < reqCount; ++i) reqExprs[i] = req.intExprs[i];

            const NameId len = gl::HashKeyBuilder::makeIntNormalizedKeyFromEncodedWithMap(
                reqExprs, reqCount, intSt,
                buf, ExecutionParameters::MAX_KEY_SLOTS,
                reverseMap, numNormVars);

            // Serialize the probe to NormKey bytes (Codec<NormKey>::encode
            // layout: int16 numberExpressions, int16 length, length x int16
            // data) and raw-StrSpan lookup -- no per-probe heap key on the
            // engine's hottest path.
            NameId probeBuf[2 + ExecutionParameters::MAX_KEY_SLOTS];
            probeBuf[0] = reqCount;
            probeBuf[1] = len;
            std::memcpy(probeBuf + 2, buf,
                static_cast<std::size_t>(len) * sizeof(NameId));
            const int32_t encId =
                memoryBlock.overallHashMemory.encodedMap.lookup(
                    StrSpan(reinterpret_cast<const char*>(probeBuf),
                            static_cast<int32_t>((2 + len) * sizeof(NameId))));
            if (encId == 0) {
                continue;
            }
            ++g_gpuEvaluationUsage.encodedHits;

            // HIT — per-hit hot window: the back-replacement pairs and every
            // transient built in the value loop below die when this hit is
            // fully processed (I-116).
            ScratchScope hitScope(scratchArena);

            // Back-replacement pairs from the int reverseMap: digit keys in
            // hot storage, values as spans over the stable decode refs.
            // Replaces the former per-hit std::map<std::string, std::string>;
            // replaceKeysScratch's greedy-longest dispatch is KeyTrie-identical,
            // so the substitution outcome is byte-equal.
            StrReplacement brPairs[ExecutionParameters::MAX_KEY_SLOTS];
            for (NameId v = 1; v <= numNormVars; ++v) {
                char digits[8];
                const int digitLen = std::snprintf(
                    digits, sizeof(digits), "%d", static_cast<int>(v));
                assert(digitLen > 0);
                const ScratchString key =
                    ScratchString::copyFrom(scratchArena, digits, digitLen);
                brPairs[v - 1].key = StrSpan(key);
                brPairs[v - 1].value = nm.decodeView(reverseMap[v]);
            }

            // Value run: peek every LocalMemoryValue blob up front (zero-decode)
            // and sort an INDEX by decoded head order (R3) -- the former
            // recordsAt heap vector + std::sort(ordered) is gone. The burst is
            // read-only on encodedMap (I-83 / D-116), so a peekRecordBytes span
            // stays valid for the whole value loop; on a straddle each peek does
            // a FRESH byte-bump alloc (no rewind) so the runN peeks coexist,
            // reclaimed by the function-exit popTo(genReqMark).
            const int32_t runN =
                memoryBlock.overallHashMemory.encodedMap.runLen(encId);
            g_gpuEvaluationUsage.localValues += static_cast<uint64_t>(runN);
            g_gpuEvaluationUsage.maximumLocalValuesPerHit = std::max(
                g_gpuEvaluationUsage.maximumLocalValuesPerHit,
                static_cast<uint64_t>(runN));
            struct LmvPeek { const char* p; int32_t len; };
            LmvPeek* peeks = runN
                ? reinterpret_cast<LmvPeek*>(genArena.resolve(genArena.alloc(
                      runN * static_cast<int32_t>(sizeof(LmvPeek)),
                      static_cast<int32_t>(alignof(LmvPeek)))))
                : nullptr;
            int32_t* vids = runN
                ? reinterpret_cast<int32_t*>(genArena.resolve(genArena.alloc(
                      runN * static_cast<int32_t>(sizeof(int32_t)),
                      static_cast<int32_t>(alignof(int32_t)))))
                : nullptr;
            int32_t* order = runN
                ? reinterpret_cast<int32_t*>(genArena.resolve(genArena.alloc(
                      runN * static_cast<int32_t>(sizeof(int32_t)),
                      static_cast<int32_t>(alignof(int32_t)))))
                : nullptr;
            for (int32_t j = 0; j < runN; ++j) {
                int32_t plen = 0;
                const char* pp =
                    memoryBlock.overallHashMemory.encodedMap.peekRecordBytes(
                        encId, j, plen, genArena);
                peeks[j] = LmvPeek{ pp, plen };
                vids[j] = LmvBlobView(pp, plen).valueId();
                order[j] = j;
            }
            // Decoded-head order (R3) -- byte-identical to the former
            // std::sort(ordered) permutation: same comparator (valueId tie ->
            // false, else compareSpans(decodeView)) lifted through the index, so
            // sorting order[] makes identical decisions to sorting the objects.
            const ValueInterner& ruleIn = memoryBlock.ruleInterner;
            std::sort(order, order + runN, [&](int32_t a, int32_t b) {
                if (vids[a] == vids[b]) return false;
                return compareSpans(ruleIn.decodeView(vids[a]),
                                    ruleIn.decodeView(vids[b])) < 0;
            });

            // Origin premise tail -- a decoded-lex INDEX over the request's
            // premises (R4), no heap decode. compareSpans == std::string::compare
            // and the (original, validity) lex order is tie-free, so the index
            // reproduces the former std::sort(orderedPremises) permutation; each
            // premise's spans are sealed synchronously at the lazy seal site
            // below (seal copies onto the sealed pages, mint-free burst, no
            // dangle).
            int32_t premOrder[ExecutionParameters::MAX_EXPRESSIONS + 2];
            assert(reqCount <= ExecutionParameters::MAX_EXPRESSIONS + 2
                && "checkLocalEncodedMemoryStatic: premise count exceeds cap");
            for (NameId i = 0; i < reqCount; ++i) premOrder[i] = i;
            std::sort(premOrder, premOrder + reqCount, [&](int32_t a, int32_t b) {
                const int c = compareSpans(nm.decodeView(req.intExprs[a]->originalId),
                                           nm.decodeView(req.intExprs[b]->originalId));
                if (c != 0) return c < 0;
                return compareSpans(nm.decodeView(req.intExprs[a]->validityId),
                                    nm.decodeView(req.intExprs[b]->validityId)) < 0;
            });

            // Sealed twin of the premise tail, built lazily on the hit's
            // first head record — most hits fire zero head records, and
            // sealed bytes never rewind, so eager sealing would leak dead
            // bytes until task end.
            // Premise count is bounded by the request's expression count, so
            // the sealed twin rides a stack buffer (no heap spine).
            SealedExpressionWithValidity sealedPremises[
                ExecutionParameters::MAX_EXPRESSIONS + 2];
            int sealedPremisesCount = 0;
            bool premisesSealed = false;

            for (int32_t vi = 0; vi < runN; ++vi) {
                const LmvBlobView lmv(peeks[order[vi]].p, peeks[order[vi]].len);

                // Per-value hot window: the replace/strip intermediate dies
                // with this iteration. rplExpr2 materializes once — it
                // escapes into the FiringRecord and feeds the string-keyed
                // probes (record fields move to sealed pages in the handoff
                // step).
                ScratchScope valueScope(scratchArena);
                const ScratchString rplScratch1 = replaceKeysScratch(
                    scratchArena, ruleIn.decodeView(lmv.valueId()),
                    brPairs, numNormVars);
                const ScratchString rplScratch2 =
                    replaceUSubstringsScratch(scratchArena, StrSpan(rplScratch1));

                // Cross-scope validity check (D-55): comparability, not
                // strict equality — a rule registered at an ancestor scope
                // of the consensus (e.g. K mutual-exclusion implications at
                // the OR's parent scope, fact at an OR-introduction subproof
                // descendant) is a legitimate firing. Mirrors the
                // request-generation-side rule documented in
                // `02_hash_engine.md` §Locality semantics.
                // The rule's scope id was minted at install and rides on
                // the LMV — no per-firing lookup (D-116's read-only burst
                // is preserved; this is not even a read of the NameMap).
                const NameId lmvVid = lmv.validityId();
                if (!nm.comparable(lmvVid, consensusValidityId)) {
                    continue;
                }

                // I-38: the deposit scope is deeperOf over ALL
                // constituents — the premises AND the rule. A rule deeper
                // than the premise consensus (e.g. a K rule of an
                // or-statement living at a branch scope, fired on main
                // facts) deposits at the RULE's scope: its knowledge is
                // conditioned on that branch context and must never leak
                // into the shallower consensus. The comparability check
                // above guarantees the pair has a deeperOf. Shared by the
                // head-record and marker-staging branches below.
                const NameId hitValidityId =
                    nm.deeperOf(consensusValidityId, lmvVid);
                const StrSpan hitValidityView =
                    (hitValidityId == consensusValidityId)
                        ? validityView : nm.decodeView(hitValidityId);
                if (hitValidityId != consensusValidityId) {
                    // The closed-scope wipe filter ran on the consensus
                    // ancestors only (per-request); a deeper deposit
                    // target must bounce on the same filter.
                    bool hitFiltered = false;
                    for (int32_t ancK = 0,
                         ancN = nm.ancLen(hitValidityId);
                         ancK < ancN; ++ancK) {
                        if (memoryBlock.intValidityNamesToFilter.contains(
                                nm.ancAt(hitValidityId, ancK))) {
                            hitFiltered = true;
                            break;
                        }
                    }
                    if (hitFiltered) continue;
                }

                if (lmv.ordis2Demand()) {
                    // Demand branch (D-267): the
                    // rule matched everything except its qualifying slot,
                    // and rplScratch2 IS the ground missing premise (every
                    // value token was subkey-bound at install; unchangeable
                    // u_ args grounded by the u_-strip above). Stage a
                    // demand record — an OWN kind, neither head nor marker
                    // (maintainer decision 2026-08-10: no marker anywhere
                    // in the ordis2 machinery): checked FIRST so neither
                    // sibling branch ever processes a demand LMV. Nothing
                    // of the marker / admission machinery is touched (no
                    // admv, no sawMarker path). Ordinary demands require
                    // pure premises in every scope, including OR branches.
                    // Integration demands are exempt from the purity gate
                    // (maintainer decision 2026-08-24,
                    // D-357): a goal's
                    // integration rule is matched by the goal's own
                    // hypothesis-chain witnesses (int_/it_ products the
                    // recursion head unpacks without enrolling them as
                    // products of recursion), so the impurity that guards
                    // general rules against algebra-iteration churn would
                    // silence exactly the near-miss the demand exists for.
                    const bool integrationDemand =
                        lmv.justification() == RuleJustification::integration;
                    // A near-miss of a goal's flat integration rule is a
                    // goal's demand too: exempt the same way, and it goes
                    // to the goal-driven rule search, not to the OR-demand
                    // map (D-574).
                    const bool goalNearMiss =
                        lmv.justification() == RuleJustification::goalDemand;
                    if (!pure && !integrationDemand && !goalNearMiss) {
                        continue;
                    }

                    // "Really missing" gate (maintainer decision
                    // 2026-08-24, D-358):
                    // a demand names a premise the rule is waiting for; a
                    // ground text already known at the deposit scope or an
                    // ancestor is not missing — the full rule fires on the
                    // same facts — so no demand is staged. The shared
                    // ancestorKnown predicate (Site F semantics), non-minting
                    // span lookup; intKnownStatements is burst-fixed, so the
                    // verdict is the same here as at apply time.
                    {
                        const NameId knownOrigId =
                            memoryBlock.nameMap.lookup(StrSpan(rplScratch2));
                        if (knownOrigId != 0
                            && gl::StatementKnowledge::ancestorKnown(memoryBlock, knownOrigId, hitValidityId,
                                             /*includeSelf=*/true)) {
                            continue;
                        }
                    }

                    // I-171 mirror (maintainer-approved 2026-08-10): a
                    // demand whose ground text carries witnesses beyond the
                    // per-batch generation cap is not staged — the
                    // admission side gates depth the same way, and without
                    // this every deeper it_ generation would mint a fresh
                    // distinct demand key (the RMW dedup collapses only
                    // exact repeats).
                    if (admissionPolicy.extractMaxIterationNumber(StrSpan(rplScratch2))
                        > parameters.maxIterationNumberVariable) {
                        continue;
                    }

                    // Combined level run (request premises + rule levels),
                    // sorted-unique — the head branch's computation.
                    SealedSpan<int> levelsSpan;
                    {
                        const int Ln = lmv.levelCount();
                        const int total = combinedCount + Ln;
                        if (total > 0) {
                            int* tmp = reinterpret_cast<int*>(genArena.resolve(
                                genArena.alloc(
                                    total * static_cast<int32_t>(sizeof(int)),
                                    static_cast<int32_t>(alignof(int)))));
                            int w = 0;
                            for (int c = 0; c < combinedCount; ++c) tmp[w++] = combinedBuf[c];
                            for (int32_t li = 0; li < lmv.levelCount(); ++li)
                                tmp[w++] = lmv.levelAt(li);
                            std::sort(tmp, tmp + w);
                            const int uniq = static_cast<int>(
                                std::unique(tmp, tmp + w) - tmp);
                            levelsSpan = SealedSpan<int>::copyFrom(sealedPages, tmp, uniq);
                        }
                    }

                    FiringRecord rec;
                    rec.isOrdis2Demand = true;
                    rec.isGoalNearMiss = goalNearMiss;
                    rec.rplExpr2 = seal(StrSpan(rplScratch2));
                    rec.validityName = seal(hitValidityView);
                    rec.levels = levelsSpan;
                    rec.demandSourceImplId = lmv.originalImplicationId();
                    sealedPages.appendRecord(rec);
                    ++g_gpuEvaluationUsage.demandRecords;
                    g_gpuEvaluationUsage.generatedBytes +=
                        static_cast<uint64_t>(StrSpan(rplScratch2).len);
                    g_gpuEvaluationUsage.levelValues +=
                        static_cast<uint64_t>(levelsSpan.size());
                }
                else if (!lmv.isMarker()) {
                    // Per-firing level set: the request's combinedLevels plus
                    // this rule's own levels, sorted-unique on the gen-scratch
                    // byte-bump tier, then sealed onto the page set as the
                    // record's level span (the former std::set<int>).
                    SealedSpan<int> levelsSpan;
                    {
                        const int Ln = lmv.levelCount();
                        const int total = combinedCount + Ln;
                        if (total > 0) {
                            int* tmp = reinterpret_cast<int*>(genArena.resolve(
                                genArena.alloc(
                                    total * static_cast<int32_t>(sizeof(int)),
                                    static_cast<int32_t>(alignof(int)))));
                            int w = 0;
                            for (int c = 0; c < combinedCount; ++c) tmp[w++] = combinedBuf[c];
                            for (int32_t li = 0; li < lmv.levelCount(); ++li) {
                                // Rule level sets are empty-or-real by
                                // construction (I-51 mail contract; local
                                // installs carry derivation levels) — the
                                // {-1} statement tier never reaches them.
                                assert(lmv.levelAt(li) >= 0
                                    && "rule level set carries the non-derived tier");
                                tmp[w++] = lmv.levelAt(li);
                            }
                            std::sort(tmp, tmp + w);
                            const int uniq = static_cast<int>(
                                std::unique(tmp, tmp + w) - tmp);
                            levelsSpan = SealedSpan<int>::copyFrom(sealedPages, tmp, uniq);
                        }
                    }

                    // Origin tail goes straight onto the sealed pages at the
                    // record fill below: the implication source, then the
                    // hit's sorted premises (sealed once per hit, lazily).
                    if (!premisesSealed) {
                        for (NameId i = 0; i < reqCount; ++i) {
                            const IntEncodedExpr* pe = req.intExprs[premOrder[i]];
                            sealedPremises[sealedPremisesCount++] =
                                SealedExpressionWithValidity{
                                    seal(nm.decodeView(pe->originalId)),
                                    seal(nm.decodeView(pe->validityId)) };
                        }
                        premisesSealed = true;
                    }

                    // Rule-intrinsic half of the disintegration decision:
                    // derived once at install (integration justification /
                    // or-intro shape => not allowed,
                    // D-241). The D-29
                    // clauses below are the firing-context half and can only
                    // further restrict.
                    bool doNotDisintegrate = !lmv.disintegrationAllowed();
                    // D-29 two-part disintegration gate (only active when
                    // incubator_mode && !ban_disintegration — the SE2 /
                    // FTA-rung-1 combination):
                    //   (1) anchor LB: ALWAYS block disintegration (anchor never
                    //       disintegrates in this mode — its job is broadcast +
                    //       handle external rules without local fan-out).
                    //   (2) non-anchor LB: block disintegration unless at least
                    //       one matched premise is in this LB's
                    //       intLocalEncodedStatementsSet (i.e. the rule was
                    //       triggered at least partly by THIS LB's own derivation
                    //       work, not solely by anchor / external-rule inputs;
                    //       incubator-derived theorem rules no longer reach a
                    //       non-anchor LB at all -- the root's log has one
                    //       reader, D-332 --
                    //       so here the external inputs are the pool lemmas).
                    // Without (1) the prover crashed at burst 2 / ~4570 exprs.
                    // Without (2) the prover crashed at burst 3 / ~7686 exprs.
                    // Both clauses are load-bearing.
                    // Pre-D-29 the prover crashed with NameMap exhaustion (cap
                    // 16384) at prover.hpp:807 — fold/sequence/etc. external
                    // rules from the prior IncubatorGauss batch fanning out
                    // across local (in[X, N]) rows.
                    if (parameters.incubator_mode && !parameters.ban_disintegration && !doNotDisintegrate) {
                        // The shared zero-allocation anchor-LB predicate
                        // (exprKey starts with "(" + anchor name).
                        if (blockIdentity.isAnchorLb(memoryBlock)) {
                            doNotDisintegrate = true;
                        } else {
                            // O(1) packed probes on the request's int rows —
                            // an exists-check, so premise order is irrelevant
                            // (the premOrder index sorts the premises for the
                            // origin tail only).
                            bool hasLocalPremise = false;
                            for (NameId i = 0; i < reqCount; ++i) {
                                if (memoryBlock.intLocalEncodedStatementsSet.contains(
                                        packStatementKey(req.intExprs[i]->originalId,
                                                         req.intExprs[i]->validityId))) {
                                    hasLocalPremise = true;
                                    break;
                                }
                            }
                            if (!hasLocalPremise) doNotDisintegrate = true;
                        }
                    }


                    // Site F mirror — gate the deposit on the same already-known
                    // ancestor-scan addExprToMemoryBlock runs at entry: a head
                    // already known at any ancestor scope records no fresh
                    // statement/origin (without it, addOrigin's identical-tuple-
                    // only dedup lets common atoms hit the origin cap and explode
                    // buildStack). intKnownStatements is burst-fixed, so the
                    // result is the same whether computed here or at apply time;
                    // it is carried in the record. Compressor mode keeps multiples.
                    bool alreadyKnown = false;
                    if (!parameters.compressor_mode) {
                        // Non-minting lookups (parallel read-only LB,
                        // D-116). The consensus validity is already an id.
                        // rplExpr2 (the fired head) may
                        // not be interned yet -> 0, which is never a key in
                        // intKnownStatements, so the scan correctly reports
                        // not-already-known without minting the head here (it is
                        // interned later at deposit time in applyFiringRecords).
                        // Span probe over the value-scratch bytes — the
                        // NameMap span lookup overload is non-minting, so no
                        // per-firing heap std::string on this hot path. The
                        // scan is the shared `ancestorKnown` predicate (the
                        // single definition of the Site F contract).
                        const NameId origId =
                            memoryBlock.nameMap.lookup(StrSpan(rplScratch2));
                        const NameId valId  = hitValidityId;
                        alreadyKnown = origId != 0
                            && gl::StatementKnowledge::ancestorKnown(memoryBlock, origId, valId,
                                             /*includeSelf=*/true);
                    }

                    // Capture the head deposit. applyFiringRecords sorts the
                    // burst's records into canonical order and applies them:
                    // the statement and its history when not already known
                    // sameIterationInternalMail.statements + exprOriginMap +
                    // disintegrationSignals. Capturing instead of depositing
                    // inline makes the cap-bounded addOrigin selection and the
                    // disintegrationSignals last-write partition-independent —
                    // the property the LB split relies on.
                    // See D-117.
                    {
                        FiringRecord rec;
                        rec.isMarker = false;
                        rec.rplExpr2 = seal(StrSpan(rplScratch2));
                        rec.validityName =
                            seal(hitValidityView);
                        rec.levels = levelsSpan;
                        rec.originTag = seal(StrSpan("implication", 11));
                        // Origin run: the implication source, then the hit's
                        // sorted premises — assembled in a stack buffer
                        // (bounded by the request's premise count) and sealed
                        // onto the page set as one span.
                        SealedExpressionWithValidity originBuf[
                            1 + ExecutionParameters::MAX_EXPRESSIONS + 2];
                        originBuf[0] = SealedExpressionWithValidity{
                            seal(ruleIn.decodeView(lmv.originalImplicationId())),
                            seal(nm.decodeView(lmv.validityId())) };
                        for (int pi = 0; pi < sealedPremisesCount; ++pi)
                            originBuf[1 + pi] = sealedPremises[pi];
                        rec.originDeps =
                            SealedSpan<SealedExpressionWithValidity>::copyFrom(
                                sealedPages, originBuf, 1 + sealedPremisesCount);
                        rec.doNotDisintegrate = doNotDisintegrate;
                        // The induction implication's head is the induction
                        // hypothesis: its disintegration products are admitted
                        // demand-free (D-356).
                        // Identity = the rule's producer implication id against
                        // the block's installed induction implications (none
                        // outside recursion step blocks).
                        rec.recursionHead = memoryBlock.isInductionHypothesisRule(
                            lmv.originalImplicationId());
                        // Park-first (I-184):
                        // a single-exclusion rule IS a disintegration product,
                        // but its fired reduced-or head must not open a cohort
                        // via route (b) — allowOrProbe stays untouched, so the
                        // head parks in rejectedMapOrdis and opens only from
                        // the compound-demand map.
                        rec.allowOrDisintegration =
                            lmv.productOfDisintegration()
                            && !lmv.subsetExclusion();
                        rec.alreadyKnown = alreadyKnown;
                        rec.iteration = iteration;
                        sealedPages.appendRecord(rec);
                        ++g_gpuEvaluationUsage.headRecords;
                        g_gpuEvaluationUsage.generatedBytes +=
                            static_cast<uint64_t>(StrSpan(rplScratch2).len);
                        g_gpuEvaluationUsage.levelValues +=
                            static_cast<uint64_t>(levelsSpan.size());
                        g_gpuEvaluationUsage.originDependencies +=
                            static_cast<uint64_t>(1 + reqCount);
                    }
                }
                else {
                    // Algebraic premises cannot demand another construction.
                    // The same purity guard applies inside OR branches. A
                    // goal-driven marked-head implication is exempt: a goal
                    // installed it for the missing output of one operator
                    // premise (D-574).
                    if (!pure
                        && lmv.justification() != RuleJustification::goalDemand) {
                        continue;
                    }

                    StagedAdmissionValue admv;
                    uint64_t markerGeneratedBytes =
                        static_cast<uint64_t>(StrSpan(rplScratch2).len);
                    int32_t markerKeyCount = 0;
                    int32_t markerRemainingCount = 0;
                    {
                        // Replaced key elements in rule order: each stripped
                        // result is sealed onto the page set, the view array
                        // assembled in a gen-scratch buffer, then sealed as the
                        // staged value's key span.
                        const int kn = lmv.keyIdCount();
                        markerKeyCount = kn;
                        if (kn > 0) {
                            SealedString* keyBuf = reinterpret_cast<SealedString*>(
                                genArena.resolve(genArena.alloc(
                                    kn * static_cast<int32_t>(sizeof(SealedString)),
                                    static_cast<int32_t>(alignof(SealedString)))));
                            for (int k = 0; k < kn; ++k) {
                                ScratchScope keyScope(scratchArena);
                                const ScratchString tmp = replaceKeysScratch(
                                    scratchArena, ruleIn.decodeView(lmv.keyIdAt(k)),
                                    brPairs, numNormVars);
                                keyBuf[k] = seal(StrSpan(replaceUSubstringsScratch(
                                    scratchArena, StrSpan(tmp))));
                                markerGeneratedBytes += static_cast<uint64_t>(
                                    keyBuf[k].size());
                            }
                            admv.key = SealedSpan<SealedString>::copyFrom(
                                sealedPages, keyBuf, kn);
                        }
                    }

                    // Remaining args: zero-copy views over the cold bytes
                    // (stable while resident), sort + dedupe by content,
                    // seal in lex order — the former
                    // decodeValueVectorToSet set, flattened.
                    {
                        // Remaining args: decode the views into a gen-scratch
                        // span buffer, sort + dedupe by content, seal the
                        // unique run onto the page set in lex order (the former
                        // decodeValueVectorToSet set, flattened).
                        const int rn = lmv.remainingCount();
                        if (rn > 0) {
                            StrSpan* remSpans = reinterpret_cast<StrSpan*>(
                                genArena.resolve(genArena.alloc(
                                    rn * static_cast<int32_t>(sizeof(StrSpan)),
                                    static_cast<int32_t>(alignof(StrSpan)))));
                            for (int ri = 0; ri < rn; ++ri)
                                remSpans[ri] = ruleIn.decodeView(lmv.remainingArgIdAt(ri));
                            std::sort(remSpans, remSpans + rn,
                                [](const StrSpan& a, const StrSpan& b) {
                                    return compareSpans(a, b) < 0;
                                });
                            SealedString* remBuf = reinterpret_cast<SealedString*>(
                                genArena.resolve(genArena.alloc(
                                    rn * static_cast<int32_t>(sizeof(SealedString)),
                                    static_cast<int32_t>(alignof(SealedString)))));
                            int uniq = 0;
                            for (int ri = 0; ri < rn; ++ri) {
                                if (ri == 0
                                    || !equalSpans(remSpans[ri], remSpans[ri - 1]))
                                    remBuf[uniq++] = seal(remSpans[ri]);
                            }
                            markerRemainingCount = uniq;
                            admv.remainingArgsSorted =
                                SealedSpan<SealedString>::copyFrom(
                                    sealedPages, remBuf, uniq);
                        }
                    }
                    admv.standardMaxAdmissionDepth = parameters.maxIterationNumberVariable;
                    admv.standardMaxSecondaryNumber = parameters.standardMaxSecondaryNumber;
                    admv.flag = false;
                    // The ordis-route tag travels rule -> staged value -> the
                    // drained admission blob; a plain uint8 copy, no interner
                    // touch (the frozen mint sequence is untouched).
                    admv.ordisOnly = lmv.ordisOnly();

                    // The templateInterner probe and the compiledExpressions
                    // read below run span-native — the probe on the existing
                    // (StrSpan, StrSpan) lookupTemplateKey overload and the
                    // category read through the compiledEntity reader fence
                    // (I-137). No per-row std::string is
                    // materialized on this hot capture edge.

                    // Zero-copy arg slicing, then sort + dedupe the
                    // non-marker args in place — the former
                    // std::set<std::string> as a flat sorted-unique sequence
                    // (same lex order, same dedup, "marker" never lands).
                    StrSpan argSpans[ExecutionParameters::MAX_ARITY];
                    const int32_t argCount = getArgsSpans(
                        StrSpan(rplScratch2), argSpans,
                        ExecutionParameters::MAX_ARITY);
                    const StrSpan markerSpan("marker", 6);
                    StrSpan bareArgs[ExecutionParameters::MAX_ARITY];
                    int32_t bareCount = 0;
                    bool sawMarker = false;
                    for (int32_t ai = 0; ai < argCount; ++ai) {
                        if (equalSpans(argSpans[ai], markerSpan)) {
                            sawMarker = true;
                            continue;
                        }
                        bareArgs[bareCount++] = argSpans[ai];
                    }
                    assert(sawMarker && "marker rule head lost its marker");
                    (void)sawMarker;
                    std::sort(bareArgs, bareArgs + bareCount,
                        [](const StrSpan& a, const StrSpan& b) {
                            return compareSpans(a, b) < 0;
                        });
                    int32_t uniqueArgCount = 0;
                    for (int32_t ai = 0; ai < bareCount; ++ai) {
                        if (ai == 0
                            || !equalSpans(bareArgs[ai], bareArgs[ai - 1])) {
                            bareArgs[uniqueArgCount++] = bareArgs[ai];
                        }
                    }

                    const LogicalEntity* markerEntity =
                        definitions.compiledEntity(extractExpressionSpan(StrSpan(rplScratch2)));
                    assert(markerEntity && "marker rule head core expression must be compiled");

                    // Capture the marker deposit. applyFiringRecords replays it
                    // in canonical order: deferredIntegrationPreps (the
                    // integration-template registration) and admissionKeysAlgebra (the algebra-
                    // admissionMap staging drained by drainAdmissionKeysAlgebra in
                    // firing order). Capturing keeps the staged-vector drain order
                    // partition-independent. See D-117.
                    {
                        FiringRecord rec;
                        rec.isMarker = true;
                        rec.rplExpr2 = seal(StrSpan(rplScratch2));
                        rec.validityName =
                            seal(hitValidityView);
                        // Bare args are bounded by MAX_ARITY, so seal them
                        // through a stack buffer onto the page set as one span.
                        SealedString markerArgsBuf[ExecutionParameters::MAX_ARITY];
                        for (int32_t ai = 0; ai < uniqueArgCount; ++ai)
                            markerArgsBuf[ai] = seal(bareArgs[ai]);
                        rec.markerArgsSorted = SealedSpan<SealedString>::copyFrom(
                            sealedPages, markerArgsBuf, uniqueArgCount);
                        rec.admv = admv;
                        sealedPages.appendRecord(rec);
                        ++g_gpuEvaluationUsage.markerRecords;
                        g_gpuEvaluationUsage.generatedBytes +=
                            markerGeneratedBytes;
                        g_gpuEvaluationUsage.markerKeys +=
                            static_cast<uint64_t>(markerKeyCount);
                        g_gpuEvaluationUsage.markerRemainingArgs +=
                            static_cast<uint64_t>(markerRemainingCount);
                        g_gpuEvaluationUsage.markerArgs +=
                            static_cast<uint64_t>(uniqueArgCount);
                    }
                }
            }
        }
        // Free every gen-scratch run this request assembled (combinedLevels +
        // the per-firing level / admv buffers) in one tail-pop.
        genArena.popTo(genReqMark);
    }

/// @brief Canonically order one hashburst's firing records when needed,
        ///        then apply their deposits to the LB.
        ///
        /// @details
        /// Consumer of the `FiringRecord`s produced by
        /// `checkLocalEncodedMemoryStatic` across one static request-evaluation
        /// pass — handed in as the LB's sealed part sets in part order, each
        /// carrying its records on its own record chain
        /// (`SealedPageSet::forEachRecord`, append order). Sorting before
        /// applying makes the cap-bounded `addOrigin`
        /// selection, the per-head `disintegrationSignals` last-write, and the
        /// `admissionKeysAlgebra` / `deferredIntegrationPreps` drain order a
        /// function of the firing SET rather than the request-generation ORDER —
        /// the determinism the LB split depends on (a partition of the rules
        /// reproduces the same merged deposits). The sort is a pointer INDEX
        /// permutation over the chains' stable payload addresses under the same
        /// strict total content order (I-77): a strict-total-order sort has a
        /// unique output sequence and content-identical duplicates are
        /// byte-indistinguishable in the deposit stream, so the index sort's
        /// deposit sequence is byte-identical to the former in-place record
        /// sort regardless of input permutation and sort algorithm. Head
        /// records deposit, when not already known,
        /// `sameIterationInternalMail.statements` + `.exprOriginMap` +
        /// `.disintegrationSignals`. Marker records feed `deferredIntegrationPreps`,
        /// `admissionKeysAlgebra`.
        ///
        /// @param memoryBlock LB whose containers receive the deposits.
        /// @param parts     The LB's sealed part sets in part order; each set's
        ///                  record chain is read in append order (never
        ///                  mutated).
        /// @param partCount Number of part sets (`>= 0`; 0 deposits nothing).
        /// @param firingRecordsCanonical Whether the concatenated record chains
        ///                  already carry the exact canonical content order. The
        ///                  processor producer passes `false`; the GPU sealing
        ///                  adapter passes `true` only after device ordering and
        ///                  doom-prefix compaction.
        /// @see `memory.cpp::checkLocalEncodedMemoryStatic` — the producer.
        /// @see `FiringRecord` — the record shape.
        /// @see `D-117`.
void HashRuleEvaluator::applyFiringRecords(Memory& memoryBlock,
        SealedPageSet* const* parts, int32_t partCount,
        bool firingRecordsCanonical) {
        // This finalize worker's slot + gen-scratch arena frame (the house
        // slot idiom): the record-pointer list and the sort index below are
        // per-call scratch, reclaimed by the single-exit popTo. The deposit
        // doors write LB-arena cold containers, interners, and heap staging
        // vectors — never this slot's gen arena — so the frame is LIFO-safe.
        const unsigned slot = (gl::WorkerContext::coreId() >= 0)
            ? static_cast<unsigned>(gl::WorkerContext::coreId())
            : genScratchArenas().slotCount() - 1;
        ScratchArena& gArena = genScratchArenas().forSlot(slot);
        const ArenaOffset mark = gArena.cursor();

        // Collect the records' stable payload addresses part by part (part
        // order), each chain in append order — exactly the concatenation
        // sequence the former per-task-vector merge produced.
        DirtyState refsDirty = DirtyState::Clean;
        PagedVector<const FiringRecord*> refs(&gArena, &refsDirty);
        for (int32_t p = 0; p < partCount; ++p) {
            parts[p]->forEachRecord<FiringRecord>([&](const FiringRecord& r) {
                refs.push_back(&r);
            });
        }

        const int32_t total = refs.size();
        const auto recordLess =
                [&memoryBlock](const FiringRecord& a, const FiringRecord& b) -> bool {
                // Sealed views compare by content (compareSpans ==
                // std::string byte order), so the canonical order is
                // byte-identical to the former string-member sort.
                int c = compareSpans(StrSpan(a.rplExpr2), StrSpan(b.rplExpr2));
                if (c != 0) return c < 0;
                c = compareSpans(StrSpan(a.validityName), StrSpan(b.validityName));
                if (c != 0) return c < 0;
                if (a.isMarker != b.isMarker) return a.isMarker < b.isMarker; // head before marker
                // Demand records are an OWN kind (isMarker false — no
                // marker anywhere in the ordis2 machinery); the
                // discriminator separates them from heads at equal
                // (rplExpr2, validity), then they order on (levels,
                // decoded source rule) — content compares only (I-84),
                // completing the strict total order for the kind (I-77).
                if (a.isOrdis2Demand != b.isOrdis2Demand)
                    return a.isOrdis2Demand < b.isOrdis2Demand;
                if (a.isOrdis2Demand) {
                    const int32_t na = a.levels.size();
                    const int32_t nb = b.levels.size();
                    const int32_t mn = na < nb ? na : nb;
                    for (int32_t i = 0; i < mn; ++i) {
                        if (a.levels[i] != b.levels[i])
                            return a.levels[i] < b.levels[i];
                    }
                    if (na != nb) return na < nb;
                    if (a.demandSourceImplId != b.demandSourceImplId) {
                        const int cs = compareSpans(
                            memoryBlock.ruleInterner.decodeView(
                                a.demandSourceImplId),
                            memoryBlock.ruleInterner.decodeView(
                                b.demandSourceImplId));
                        assert(cs != 0
                            && "distinct rule ids decode to equal bytes — "
                               "interner injectivity broken");
                        return cs < 0;
                    }
                    // An ordinary and a near-miss demand of one rule and
                    // slot differ only in their kind (D-574).
                    return a.isGoalNearMiss < b.isGoalNearMiss;
                }
                if (!a.isMarker) {
                    c = compareSpans(StrSpan(a.originTag), StrSpan(b.originTag));
                    if (c != 0) return c < 0;
                    if (a.originDeps.size() != b.originDeps.size())
                        return a.originDeps.size() < b.originDeps.size();
                    for (int32_t i = 0; i < a.originDeps.size(); ++i) {
                        c = compareSpans(StrSpan(a.originDeps[i].original),
                                         StrSpan(b.originDeps[i].original));
                        if (c != 0) return c < 0;
                        c = compareSpans(StrSpan(a.originDeps[i].validityName),
                                         StrSpan(b.originDeps[i].validityName));
                        if (c != 0) return c < 0;
                    }
                    {
                        // Sorted-unique int spans compare lexicographically —
                        // byte-identical to the former std::set<int> order.
                        const int32_t na = a.levels.size();
                        const int32_t nb = b.levels.size();
                        const int32_t mn = na < nb ? na : nb;
                        for (int32_t i = 0; i < mn; ++i) {
                            if (a.levels[i] != b.levels[i])
                                return a.levels[i] < b.levels[i];
                        }
                        if (na != nb) return na < nb;
                    }
                    if (a.doNotDisintegrate != b.doNotDisintegrate)
                        return a.doNotDisintegrate < b.doNotDisintegrate;
                    if (a.allowOrDisintegration != b.allowOrDisintegration)
                        return a.allowOrDisintegration < b.allowOrDisintegration;
                    if (a.alreadyKnown != b.alreadyKnown) return a.alreadyKnown < b.alreadyKnown;
                    return false;
                }
                // Element-wise lex over the sealed vectors — exactly the
                // former std::vector<std::string> / std::set<std::string>
                // lexicographical orderings on the same content.
                const auto lexLess3 = [](const SealedSpan<SealedString>& x,
                                         const SealedSpan<SealedString>& y) -> int {
                    const int32_t n = x.size() < y.size() ? x.size() : y.size();
                    for (int32_t i = 0; i < n; ++i) {
                        const int e = compareSpans(StrSpan(x[i]), StrSpan(y[i]));
                        if (e != 0) return e;
                    }
                    if (x.size() != y.size())
                        return x.size() < y.size() ? -1 : 1;
                    return 0;
                };
                c = lexLess3(a.admv.key, b.admv.key);
                if (c != 0) return c < 0;
                c = lexLess3(a.markerArgsSorted, b.markerArgsSorted);
                if (c != 0) return c < 0;
                c = lexLess3(a.admv.remainingArgsSorted,
                             b.admv.remainingArgsSorted);
                if (c != 0) return c < 0;
                return false;
            };
        PagedSortedOrdinals sorted(gArena, firingRecordsCanonical ? 0 : total,
            [&refs, &recordLess](int32_t left, int32_t right) {
                return recordLess(*refs[left], *refs[right]);
            }, ExecutionParameters::kFiringRecordSortChunk);
        int32_t canonicalAt = 0;
        const auto nextIdx = [&]() -> int32_t {
            if (firingRecordsCanonical) {
                assert(canonicalAt < total);
                return canonicalAt++;
            }
            return sorted.next();
        };

        const int maxOrigins = parameters.compressor_mode
            ? parameters.compressor_max_origins_per_expr
            : parameters.max_origin_per_expr;

        for (int32_t ri = 0; ri < total; ++ri) {
            const FiringRecord& rec = *refs[nextIdx()];
            if (rec.isOrdis2Demand) {
                // Demand deposits — an OWN kind, checked FIRST (neither
                // head nor marker; maintainer decision 2026-08-10). They
                // ride their own staging vector (sealed views;
                // performElemPhase2 clears post-drain like the sibling
                // vectors). No admission / marker container is touched
                // here.
                memoryBlock.admissionKeysOrdis2.push_back(
                    Ordis2DemandRecord{ rec.rplExpr2, rec.validityName,
                                        rec.demandSourceImplId,
                                        rec.levels, rec.isGoalNearMiss });
            } else if (!rec.isMarker) {
                // Sealed views feed the span doors directly — no head-record
                // materialization at this deposit boundary; the doors intern
                // the bytes into the LB's NameMap (statements / signals) and
                // originInterner (origins). Single-threaded drain, so
                // minting is legal. The spans alias the sealed page set,
                // stable until the post-drain sweep frees it (I-3).
                const StrSpan expr(rec.rplExpr2);
                const StrSpan validity(rec.validityName);
                // A strong shot whose premise is a strong shot's conclusion,
                // or derives from one, is dropped before it touches the
                // block: the strong hypothesis never fires on its own
                // products (I-348).
                if (!rec.alreadyKnown && rec.recursionHead && memoryBlock.strongStepBlock
                    && StrongShotHistory::firesOnOwnProduct(memoryBlock, rec))
                    continue;
                if (!rec.alreadyKnown) {
                    insertInternalStatement(memoryBlock.sameIterationInternalMail,
                                            memoryBlock.nameMap, expr, validity,
                                            rec.levels.begin(), rec.levels.size());
                    if (parameters.trackHistory) {
                        const OriginTag tag =
                            originTagFromString(StrSpan(rec.originTag));
                        // The capacity mirrors the PRODUCER's own staging
                        // bound (checkLocalEncodedMemoryStatic assembles
                        // originBuf[1 + MAX_EXPRESSIONS + 2] before sealing);
                        // the assert pins that the consumer's bound tracks
                        // the producer's (Rule 19).
                        OriginDep depsBuf[1 + ExecutionParameters::MAX_EXPRESSIONS + 2];
                        const int32_t depN = rec.originDeps.size();
                        assert(depN <= 1 + ExecutionParameters::MAX_EXPRESSIONS + 2
                            && "originDeps exceed the producer's own staging bound");
                        for (int32_t i = 0; i < depN; ++i) {
                            depsBuf[i] = OriginDep{
                                StrSpan(rec.originDeps[i].original),
                                StrSpan(rec.originDeps[i].validityName) };
                        }
                        addInternalMailOrigin(memoryBlock.sameIterationInternalMail,
                                      memoryBlock.originInterner,
                                      expr, validity, tag, depsBuf, depN,
                                      maxOrigins);
                    }
                    setInternalDisintegrationSignal(
                        memoryBlock.sameIterationInternalMail,
                        memoryBlock.nameMap, expr, validity,
                        rec.doNotDisintegrate, rec.allowOrDisintegration,
                        rec.iteration, rec.recursionHead);
                }
            } else {
                // Marker deposits stay sealed end-to-end: the staging
                // vectors carry the views, the drains materialize at their
                // mint points, and performElemPhase2 clears the vectors
                // post-drain — before the post-join sweep frees the pages.
                memoryBlock.deferredIntegrationPreps.push_back(
                    DeferredIntegrationPrep{ rec.rplExpr2, rec.markerArgsSorted,
                                             rec.validityName });
                memoryBlock.admissionKeysAlgebra.push_back(
                    AdmissionKeyAlgebraRecord{
                        SealedExpressionWithValidity{ rec.rplExpr2,
                                                      rec.validityName },
                        rec.admv });
            }
        }
        // Reclaim the index + the refs list's byte-bump footprint; refs's
        // page-tier pages are freed by its destructor (independent tier).
        gArena.popTo(mark);
    }
}
