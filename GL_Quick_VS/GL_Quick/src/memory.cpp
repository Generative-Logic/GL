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

// Hash engine — translation unit.
//
// Hosts the bodies of the hash-engine member functions of ExpressionAnalyzer
// so they live in the same TU as the data structures they operate on
// (HashMemory, LocalMemoryValue, EncodedExpression, IntNormalizedKey,
// NameMap — all already in memory.hpp). The class declaration itself remains
// in prover.hpp because C++ requires a single class definition; only the
// function bodies live here.

#include "memory.hpp"
#include "prover.hpp"
#include "parameters.hpp"
#include "memory_infra/lb_deload.hpp"
#include <cstdio>
#include "memory_infra/str_ops.hpp"
#include "memory_infra/arena_stack.hpp"
#include "memory_infra/scratch_arena.hpp"
#include <fstream>  // [ORDIS-TRAP] diagnostic trap output (sandbox/equi_reshuffle_rung1)
#include "infra/rt_tracker.hpp"

#include <algorithm>
#include <cstring>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

namespace gl {

    // D-119: per-executor LB-split context. Default
    // (0, 1) is the unsplit identity (partitionAccepts is then a no-op). Set per
    // executor at the top of its hashburst call; thread-local so parallel
    // executors of one LB never race on them.
    thread_local int g_splitProcessID = 0;
    thread_local int g_splitCount = 1;

    // Out-of-line IntEncodedExpr recordUSignature (declared in memory.hpp).
    // Heap-free: builds the (slot, argFullId) pairs on the per-slot gen-scratch
    // byte-bump tier, then range-constructs the signature vector INSIDE the set
    // node via emplace(first, last) — no function-local std::vector. Out-of-line
    // because it reaches genScratchArenas() + ExpressionAnalyzer::g_currentCoreId,
    // which an inline header body cannot see (the encodePush(StrSpan) precedent).
    void recordUSignature(OwnerSet& os,
                          const IntEncodedExpr* encList, int16_t count) {
        using Pair = std::pair<int16_t, int16_t>;
        // Single-threaded install (mergeOwnerRecord, I-83): workers publish
        // g_currentCoreId; -1 (setup contexts) maps to the reserved last slot.
        const unsigned uSlot = (ExpressionAnalyzer::g_currentCoreId >= 0)
            ? static_cast<unsigned>(ExpressionAnalyzer::g_currentCoreId)
            : genScratchArenas().slotCount() - 1;
        ScratchArena& a = genScratchArenas().forSlot(uSlot);
        const ArenaOffset mark = a.cursor();

        // Worst-case run length = total arg slots = Sum arity <= count*MAX_ARITY;
        // count is not tightly bounded, so an arena run (cap-free) not a stack
        // array. The assert below is the Rule-19 tripwire on the derivation.
        const int32_t cap = static_cast<int32_t>(count)
            * static_cast<int32_t>(ExecutionParameters::MAX_ARITY);
        Pair* sig = (cap > 0)
            ? reinterpret_cast<Pair*>(a.resolve(a.alloc(
                  cap * static_cast<int32_t>(sizeof(Pair)),
                  static_cast<int32_t>(alignof(Pair)))))
            : nullptr;

        int32_t n = 0;
        int16_t slot = 0;
        bool hasUArg = false;
        for (int16_t e = 0; e < count; ++e) {
            const IntEncodedExpr& expr = encList[e];
            const int16_t arity = std::min(expr.arity,
                static_cast<int16_t>(ExecutionParameters::MAX_ARITY));
            for (int16_t j = 0; j < arity; ++j) {
                if (expr.argUnchangeable[j] != 0) {
                    hasUArg = true;
                    assert(n < cap);
                    sig[n++] = Pair(slot, expr.argFullId[j]);
                }
                ++slot;
            }
        }
        if (!hasUArg) os.hasLooseOwner = true;
        else os.uSignatures.emplace(sig, sig + n);
        a.popTo(mark);
    }


    // ------------------------------------------------------------------
    // Hash-engine rule installation (moved from prover.cpp in commit B).
    // Path 1 — head-implication LMV inserts via addToHashMemory.
    // Path 2 — marker LMV inserts via makeNormalizedKeysForAdmission.
    // ------------------------------------------------------------------

    /// @brief Install one or more `LocalMemoryValue` records into a `HashMemory`,
    /// keyed on the canonical normalization of `key`.
    ///
    /// @details
    /// Path 1 of the hash-engine rule installation: head-implication LMV inserts.
    /// Each call corresponds to one head-implication that the prover wants to
    /// register with the hash engine for later firing.
    ///
    /// The body unfolds in three logical phases:
    ///
    /// 1. **multiplyImplication.** The producer implication is expanded via
    ///    `multiplyImplication(originalImplication)` into one or more copies that
    ///    differ in how their bound variables and remaining-args are partitioned.
    ///    Copy 0 reuses the caller-supplied `key`/`value`; copies 1..N synthesize
    ///    their own from `multiplyImplication`'s output. Provenance for the
    ///    multiplied copies is recorded in `mb.exprOriginMap` (and the mail-out
    ///    mirror) under the `"multiplied from"` tag, capped per `max_origin_per_expr`
    ///    (`compressor_max_origins_per_expr` in compressor mode).
    /// 2. **Per-copy normalization + dedupe.** For each copy, build the
    ///    `IntNormalizedKey` from the int16_t-encoded key bytes; consult
    ///    `normalizedEncodedKeys` and `normalizedEncodedSubkeys*` for fast
    ///    rejection; if novel, allocate on the gen scratch arena byte-bump tier so the key
    ///    pointer stays stable for the rest of the LB's lifetime.
    /// 3. **LMV emit + admission.** Construct a `LocalMemoryValue` carrying
    ///    the value, levels, justification, originalImplication, key,
    ///    remainingArgs, validityName. Set `productOfDisintegration` per
    ///    [D-32](../../docs/agentic_swdd/40_decisions.md#d-32) when the producing
    ///    implication has at least one `u_`-prefixed premise arg (this gates
    ///    later OR-disintegration on the head-firing path). Push into
    ///    `targetIntMemory.encodedMap[normKey]`. When
    ///    `performAdmissionMapUpdate == true`, also call into the admission
    ///    side via `updateAdmissionMap3` so the `(expression, scope)` pair
    ///    is registered in `targetIntMemory.admissionMap`.
    ///
    /// @param key                     Pre-built normalization key (string form;
    ///                                int16_t mirror is computed inside).
    /// @param value                   Head expression text (what the rule produces).
    /// @param remainingArgs           Argument names not consumed by the key
    ///                                that must travel with the head.
    /// @param mb                      Owning `Memory` (used for `exprOriginMap`,
    ///                                `mailOut`, `ruleInterner`).
    /// @param targetIntMemory         The `HashMemory` to install into. Usually
    ///                                `mb.localHashMemory` for fresh installs;
    ///                                deltas go to `mb.localHashMemoryDelta`.
    /// @param levels                  Ascending-unique level run — LB levels at
    ///                                which the rule is admissible.
    /// @param levelCount              The level count (`>= 0`).
    /// @param originalImplication     Full producer implication text, kept for
    ///                                provenance and for `multiplyImplication`.
    /// @param maxAdmissionDepth       Standard depth budget passed to the
    ///                                admission map.
    /// @param maxSecondaryNumber      Standard secondary-iteration budget.
    /// @param partOfRecursion         True when the producer is part of a
    ///                                recursion block; routes admission-map
    ///                                updates differently.
    /// @param minNumOperatorsKey      Minimum-operator threshold used by some
    ///                                downstream filters; carried through.
    /// @param justification           Proof tag string emitted on hit
    ///                                (e.g. `"hash"`, `"head"`, `"equality1"`).
    /// @param performAdmissionMapUpdate If true, call `updateAdmissionMap3`
    ///                                in lockstep with the LMV install.
    /// @param originalImplicationClean De-renamed form used as the antecedent
    ///                                in `"multiplied from"` origin records.
    /// @param validityName            Scope at which the rule fires. `"main"`
    ///                                or a `_boundary_`-bearing form minted via
    ///                                `NameMap::encodePush` per
    ///                                [I-2](../../docs/agentic_swdd/30_invariants.md#i-2).
    ///
    /// @pre  `mb.lbMemory.manager` is resident — the cold `encodedMap` and the
    ///       four `normalizedEncoded*` owner maps install onto it. Keys are owning
    ///       (`NormKey`); the former keyArena-backed `IntNormalizedKey`
    ///       persistence is retired.
    /// @post `targetIntMemory.encodedMap` and the related `normalizedEncoded*`
    ///       sets are extended with the new LMV(s). Origin entries are added
    ///       under the `"multiplied from"` tag for each non-zero copy.
    /// @invariant [I-2](../../docs/agentic_swdd/30_invariants.md#i-2) — non-`"main"`
    ///            validityName must be minted via `NameMap::encodePush`.
    /// @invariant [I-3](../../docs/agentic_swdd/30_invariants.md#i-3) — when this function
    ///            calls `nm.decode()` / reads `idToSub[]`, the result is
    ///            copied before any nested mint (re-entrance through
    ///            `multiplyImplication` may grow the underlying vectors).
    /// @invariant [I-25](../../docs/agentic_swdd/30_invariants.md#i-25) — cross-scope
    ///            deposits keep `validityName` end-to-end via
    ///            `ExpressionWithValidity` pairs.
    /// @see `prover.hpp::checkLocalEncodedMemoryStatic` — consumer of the
    ///      LMVs installed here.
    /// @see `prover.hpp::updateAdmissionMap3` — admission-side counterpart.
    /// @see `multiplyImplication` (`prover.cpp`) — generator of the per-copy
    ///      key/value pairs.
    void ExpressionAnalyzer::addToHashMemory(const StrSpan* keyRun, int32_t keyN,
        StrSpan valueSpan,
        const StrSpan* remRun, int32_t remN,
        Memory& mb,
        HashMemory& targetIntMemory,
        const int* levels, int32_t levelCount,
        StrSpan originalImplicationSpan,
        int maxAdmissionDepth,
        int maxSecondaryNumber,
        bool partOfRecursion,
        int minNumOperatorsKey,
        StrSpan justificationSpan,
        bool performAdmissionMapUpdate,
        StrSpan originalImplicationCleanSpan,
        StrSpan validityNameSpan) {

        // 0% heap: the caller feeds spans over its own stable bytes and this
        // install body reads them directly (key = keyRun, value = valueSpan,
        // remainingArgs = the sorted-unique remRun, original / justification /
        // clean / validity = their spans). Byte-identical: the ordered chain +
        // the sorted-unique remaining-arg run are preserved.

        // Own multiplyImplication loop — fully independent from addToHashMemory.
        // The span form emits each copy onto mulStrArena (string tier) as a
        // CopyRef index on mulGenArena (page tier, I-124); each copy is read as a
        // zero-copy copySpan over mulStrArena (0% heap). mulStrScope holds the
        // copies live across the whole loop.
        // Per-registry slot derivation: the two registries deliberately differ
        // by one slot (string: logicalCores+1 with a DEDICATED reserved
        // single-threaded slot; gen: logicalCores, single-threaded fallback
        // sharing the last worker slot), so each registry's fallback comes from
        // its OWN slotCount(). Workers (g_currentCoreId >= 0) use their own
        // coreId on both — disjoint arenas, in range for both registries.
        const unsigned mulStrSlot = (g_currentCoreId >= 0)
            ? static_cast<unsigned>(g_currentCoreId)
            : scratchArenas().slotCount() - 1;
        const unsigned mulGenSlot = (g_currentCoreId >= 0)
            ? static_cast<unsigned>(g_currentCoreId)
            : genScratchArenas().slotCount() - 1;
        ScratchArena& mulStrArena = scratchArenas().forSlot(mulStrSlot);
        ScratchScope mulStrScope(mulStrArena);
        ScratchArena& mulGenArena = genScratchArenas().forSlot(mulGenSlot);
        DirtyState mulDirty = DirtyState::Clean;
        PagedVector<CopyRef> copies(&mulGenArena, &mulDirty);
        multiplyImplication(originalImplicationSpan, mulStrArena, copies);

        for (int32_t c = 0; c < copies.size(); ++c) {
            const CopyRef cr = copies[c];
            const StrSpan copySpan(
                reinterpret_cast<const char*>(mulStrArena.resolve(cr.off)), cr.len);
            // curKey is a StrSpan run; curOrigImpl == copySpan (mulStrArena,
            // stable for this copy iteration).
            StrSpan curKeyRun[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
            int32_t curKeyN = 0;
            StrSpan curValueSpan;

            if (c == 0) {
                for (int32_t i = 0; i < keyN; ++i) {
                    assert(curKeyN < ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS
                        && "addToHashMemory: curKey element count exceeds cap");
                    curKeyRun[curKeyN++] = keyRun[i];
                }
                curValueSpan = valueSpan;
            } else {
                // Track history for multiplied copies (L3 span-record door; drops
                // the EWV / OriginLine build).
                if (parameters.trackHistory) {
                    const OriginDep mulDeps[1] = {
                        { originalImplicationCleanSpan, StrSpan("main", 4) } };
                    const int mulCap = (parameters.compressor_mode
                        ? parameters.compressor_max_origins_per_expr
                        : parameters.max_origin_per_expr);
                    addOriginEncoded(mb.exprOriginMap, mb.originInterner, copySpan,
                        StrSpan("main", 4), OriginTag::multipliedFrom, mulDeps, 1, mulCap);
                    // Direct mailOut.exprOriginMap write — DOCUMENTED EXCEPTION
                    // to the fillMailOut-sole-writer policy. The multiplied
                    // implication installs as a *rule* in overallHashMemory
                    // (and never lands in intEncodedStatements /
                    // intLocalEncodedStatementsDelta), so `fillMailOut` cannot
                    // propagate the "multiplied from" history line via the
                    // delta-driven copy. See I-64 Caveats.
                    addRoutingMailOrigin(mb.mailOut, mb.originInterner, copySpan,
                        StrSpan("main", 4), OriginTag::multipliedFrom, mulDeps, 1, mulCap);
                }

                // Row 238: span twin — curKey from each triple's key (get<0>),
                // curValue = head. Spans slice copySpan (mulStrArena, stable).
                ce::disintegrateImplicationSpans(copySpan, curValueSpan,
                    [&curKeyRun, &curKeyN](StrSpan keySpan, const StrSpan*, int32_t) {
                        assert(curKeyN < ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS
                            && "addToHashMemory: curKey element count exceeds cap");
                        curKeyRun[curKeyN++] = keySpan;
                    });
            }

            // --- Shared members (path-independent, needed for integration) ---
            // implication = curKey + curValue as a StrSpan run.
            StrSpan implRun[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS + 1];
            int32_t implRunN = 0;
            for (int32_t i = 0; i < curKeyN; ++i) implRun[implRunN++] = curKeyRun[i];
            implRun[implRunN++] = curValueSpan;
            // Positional value-id mint (byte-identical to encodeValueVector, I-84);
            // originals.mint via the raw encodeIdVecKeyInto door (no owning IdVecKey).
            {
                int32_t implIds[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS + 1];
                for (int32_t i = 0; i < implRunN; ++i)
                    implIds[i] = mb.ruleInterner.encode(implRun[i]);
                char ivKeyBuf[4 * (ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS + 2)];
                const int32_t ivLen = encodeIdVecKeyInto(implIds, implRunN,
                    ivKeyBuf, static_cast<int32_t>(sizeof(ivKeyBuf)));
                targetIntMemory.originals.inner().mint(StrSpan(ivKeyBuf, ivLen));
            }
            {
                // Decoded-lex INDEX over the triggers, sorted by decodeTemplateKeyView
                // (template then validity, compareSpans) — reproduces the former
                // std::sort(triggerRows) order (I-84). Per row copyFrom template +
                // validity to the string tier (MANDATORY: makeAdmissionKeys mints
                // templateInterner, the SAME interner the trigger decodes from, I-3).
                auto& trig = targetIntMemory.triggersForAdmissionSetIntegration;
                const int32_t trigN = trig.count();
                if (trigN > 0) {
                    const unsigned tgGenSlot = (g_currentCoreId >= 0)
                        ? static_cast<unsigned>(g_currentCoreId)
                        : genScratchArenas().slotCount() - 1;
                    ScratchArena& tgArena = genScratchArenas().forSlot(tgGenSlot);
                    ScratchScope tgScope(tgArena);
                    int32_t* tidx = reinterpret_cast<int32_t*>(
                        tgArena.resolve(tgArena.alloc(trigN * 4, 4)));
                    for (int32_t i = 0; i < trigN; ++i) tidx[i] = i + 1;
                    std::sort(tidx, tidx + trigN, [&](int32_t a, int32_t b) {
                        StrSpan ka, va, kb, vb;
                        decodeTemplateKeyView(trig.keyAt(a), mb.templateInterner, mb.nameMap, ka, va);
                        decodeTemplateKeyView(trig.keyAt(b), mb.templateInterner, mb.nameMap, kb, vb);
                        const int ck = compareSpans(ka, kb);
                        if (ck != 0) return ck < 0;
                        return compareSpans(va, vb) < 0;
                    });
                    const unsigned tgStrSlot = (g_currentCoreId >= 0)
                        ? static_cast<unsigned>(g_currentCoreId)
                        : scratchArenas().slotCount() - 1;
                    ScratchArena& tgStrArena = scratchArenas().forSlot(tgStrSlot);
                    for (int32_t i = 0; i < trigN; ++i) {
                        ScratchScope trScope(tgStrArena);
                        StrSpan tKey, tVal;
                        decodeTemplateKeyView(trig.keyAt(tidx[i]), mb.templateInterner, mb.nameMap, tKey, tVal);
                        const ScratchString tKeyC = ScratchString::copyFrom(tgStrArena, tKey.ptr, tKey.len);
                        const ScratchString tValC = ScratchString::copyFrom(tgStrArena, tVal.ptr, tVal.len);
                        makeAdmissionKeys(implRun, implRunN, StrSpan(tKeyC),
                            targetIntMemory, StrSpan(tValC), mb);
                    }
                }
            }

            this->makeNormalizedKeysForAdmission(curKeyRun, curKeyN, targetIntMemory,
                mb.nameMap, mb.ruleInterner, curValueSpan, minNumOperatorsKey,
                copySpan, validityNameSpan, &mb);

            // Remaining args: c==0 = the caller's sorted-unique remRun; c!=0 =
            // getRemainingArgs over curKeyRun (u_ vars may have been equalized),
            // also sorted-unique. 0% heap.
            const StrSpan* curRemRun;
            int32_t curRemN;
            StrSpan remScratch[ExecutionParameters::MAX_ADMISSION_REM_ARGS];
            if (c == 0) {
                curRemRun = remRun; curRemN = remN;
            } else {
                curRemN = getRemainingArgs(curKeyRun, curKeyN, remScratch,
                    ExecutionParameters::MAX_ADMISSION_REM_ARGS);
                curRemRun = remScratch;
            }

            if (performAdmissionMapUpdate)
            {
                // keyPlusValue == implRun (curKey + curValue, already built).
                updateAdmissionMap(mb,
                    implRun, implRunN,
                    curRemRun, curRemN,
                    maxAdmissionDepth,
                    maxSecondaryNumber,
                    partOfRecursion,
                    StrSpan("main", 4));
            }

            // --- Inner: add this single key/value to int hash memory ---
            const int32_t n = curKeyN;
            NameMap& nm = mb.nameMap;
            // D-105: encode the rule's owner scope id
            // once per install; the owner-set inserts below store it with each
            // owner so the validity prune never re-encodes.
            const int16_t ownerVid = nm.encode(validityNameSpan);

            // D-119: composite id of this rule's
            // (expanded-original, validity) pair. nm.encode(copySpan) is the
            // int16 id of the full expanded implication (idempotent — minted at
            // first install, returned thereafter). Stored in every (sub)key's
            // partitionIds below so a split executor claims the keys of the
            // rules assigned to it via id % splitCount.
            const int32_t partitionId = makePartitionId(nm.encode(copySpan), ownerVid);

            // intRemainingArgs: mint NameMap in curRemRun (sorted-lex) order ==
            // the former set-lex mint order (I-84), then a sorted-ASCENDING copy
            // for the Int16SetKey == the former std::set<int16_t> iteration.
            int16_t intRemArgs[ExecutionParameters::MAX_ADMISSION_REM_ARGS];
            for (int32_t i = 0; i < curRemN; ++i) intRemArgs[i] = nm.encode(curRemRun[i]);
            int16_t intRemArgsSorted[ExecutionParameters::MAX_ADMISSION_REM_ARGS];
            if (curRemN > 0)
                std::memcpy(intRemArgsSorted, intRemArgs,
                    static_cast<std::size_t>(curRemN) * sizeof(int16_t));
            std::sort(intRemArgsSorted, intRemArgsSorted + curRemN);

            std::map<int, std::vector<std::vector<int>>>::const_iterator pit =
                this->allPermutationsAna.find(static_cast<int>(n));
            if (pit == this->allPermutationsAna.end()) {
                continue;
            }
            const std::vector<std::vector<int>>& permuts = pit->second;

            // idsRun[i] = the core expression span of curKeyRun[i] (the permutation
            // sort gate; compareSpans == the former ids[i] std::string compare).
            StrSpan idsRun[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
            for (int32_t i = 0; i < n; ++i)
                idsRun[i] = extractExpressionSpan(curKeyRun[i]);

            int16_t bufIgnored[ExecutionParameters::MAX_KEY_SLOTS];
            int16_t bufNotIgnored[ExecutionParameters::MAX_KEY_SLOTS];

            const unsigned genSlot = (g_currentCoreId >= 0)
                ? static_cast<unsigned>(g_currentCoreId)
                : genScratchArenas().slotCount() - 1;
            ScratchArena& genArena = genScratchArenas().forSlot(genSlot);
            const unsigned strSlot = (g_currentCoreId >= 0)
                ? static_cast<unsigned>(g_currentCoreId)
                : scratchArenas().slotCount() - 1;
            ScratchArena& strArena = scratchArenas().forSlot(strSlot);

            for (std::size_t p = 0; p < permuts.size(); ++p) {
                const std::vector<int>& permutation = permuts[p];

                bool toContinue = false;
                if (n >= 2) {
                    for (int32_t k = 0; k + 1 < n; ++k) {
                        if (compareSpans(idsRun[permutation[k]], idsRun[permutation[k + 1]]) > 0) {
                            toContinue = true;
                            break;
                        }
                    }
                }
                if (toContinue) continue;

                ScratchScope permScope(strArena);

                // Span-native int encode (no heap EncodedExpression -> no
                // parseArgument): each key element becomes an IntEncodedExpr via
                // the encodeExpression span twin, then makeIntNormalizedKeyFromEncoded*
                // (byte-identical outBuf + reverseMap to the EncodedExpression form).
                IntEncodedExpr intEncoded[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
                const int16_t intEncCount = static_cast<int16_t>(permutation.size());
                assert(intEncCount <= ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS
                    && "addToHashMemory: key element count exceeds cap");
                for (std::size_t k = 0; k < permutation.size(); ++k) {
                    intEncoded[k] = encodeExpression(
                        curKeyRun[permutation[k]], StrSpan("main", 4), nm);
                }

                int16_t reverseMap[ExecutionParameters::MAX_KEY_SLOTS];
                std::memset(reverseMap, 0, sizeof(reverseMap));
                int16_t numNormVars = 0;
                int16_t lenIgnored = makeIntNormalizedKeyFromEncodedWithMap(
                    intEncoded, intEncCount, true, bufIgnored,
                    ExecutionParameters::MAX_KEY_SLOTS, reverseMap, numNormVars);

                int16_t lenNotIgnored = makeIntNormalizedKeyFromEncoded(
                    intEncoded, intEncCount, false, bufNotIgnored,
                    ExecutionParameters::MAX_KEY_SLOTS);

                // Rename run { normalized-var name -> decimal id } from reverseMap.
                // decodeView keys: no NameMap mint falls between here and the last
                // replaceKeysScratch use (the mints below are ruleInterner), so
                // decodeView is I-3-safe. Byte-identical to the former lex-ordered
                // std::map (replaceKeysScratch is greedy-longest, order-independent).
                StrReplacement mp2Pairs[ExecutionParameters::MAX_KEY_SLOTS];
                int32_t mp2PairsN = 0;
                for (int16_t id = 1; id <= numNormVars; ++id) {
                    assert(mp2PairsN < ExecutionParameters::MAX_KEY_SLOTS
                        && "addToHashMemory: rename pair count exceeds cap");
                    char decTmp[12];
                    const int32_t decN = writeDecimalDigits(decTmp, id);
                    char* decBuf = strArena.allocBytes(decN);
                    std::memcpy(decBuf, decTmp, static_cast<std::size_t>(decN));
                    mp2Pairs[mp2PairsN].key = nm.decodeView(reverseMap[id]);
                    mp2Pairs[mp2PairsN].value = StrSpan(decBuf, decN);
                    ++mp2PairsN;
                }
                const ScratchString valueVariantS =
                    replaceKeysScratch(strArena, curValueSpan, mp2Pairs, mp2PairsN);

                // LMV fields (id-run door). ruleInterner mint ORDER: valueVariant,
                // originalImplication, then each remaining arg (== the former
                // per-field order). keyIds empty (the head install carries none).
                const int32_t valueId = mb.ruleInterner.encode(StrSpan(valueVariantS));
                const bool isMarker =
                    containsSpan(StrSpan(valueVariantS), StrSpan("marker", 6));
                const int32_t originalImplId = mb.ruleInterner.encode(copySpan);
                int32_t remIds[ExecutionParameters::MAX_ADMISSION_REM_ARGS];
                for (int32_t i = 0; i < curRemN; ++i)
                    remIds[i] = mb.ruleInterner.encode(curRemRun[i]);
                const RuleJustification just = ruleJustificationFromString(justificationSpan);

                // D-32: product-of-disintegration — true iff at least one premise
                // (chain element) has an arg starting with "u_" (bound-variable
                // placeholder). Consumed by checkLocalEncodedMemoryStatic to gate
                // OR-disintegration on the head-firing path.
                bool productOf = false;
                for (int32_t i = 0; i < curKeyN && !productOf; ++i) {
                    StrSpan pArgs[ExecutionParameters::MAX_ARITY];
                    const int32_t pArgsN = getArgsSpans(curKeyRun[i], pArgs,
                                                        ExecutionParameters::MAX_ARITY);
                    for (int32_t a = 0; a < pArgsN; ++a) {
                        if (pArgs[a].len >= 2 && pArgs[a].ptr[0] == 'u' && pArgs[a].ptr[1] == '_') {
                            productOf = true; break;
                        }
                    }
                }

                // encodedMap HEAD record + owner record + remaining-args index,
                // all via the raw / id-run doors (no owning NormKey / LMV). The
                // levels run is caller-owned ascending-unique (I-136) == the former
                // std::set<int> serialize order.
                appendLmvIdsRecord(targetIntMemory.encodedMap,
                    static_cast<int16_t>(n), bufIgnored, lenIgnored,
                    valueId, isMarker, nullptr, 0, remIds, curRemN,
                    originalImplId, ownerVid, genArena,
                    levels, levelCount, just, productOf);
                mergeOwnerRecord(targetIntMemory.normalizedEncodedKeys,
                    static_cast<int16_t>(n), bufNotIgnored, lenNotIgnored,
                    partitionId, intEncoded, intEncCount, nm);
                insertRemainingArgsNormKey(targetIntMemory.remainingArgsNormalizedEncodedMap,
                    intRemArgsSorted, curRemN,
                    static_cast<int16_t>(n), bufNotIgnored, lenNotIgnored, genArena);
            }

            // --- makeNormalizedSubkeys equivalent ---
            for (std::size_t p = 0; p < permuts.size(); ++p) {
                const std::vector<int>& permut = permuts[p];

                StrSpan tempRun[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
                int32_t tempN = 0;
                for (std::size_t k = 0; k < permut.size(); ++k)
                    tempRun[tempN++] = curKeyRun[permut[k]];

                for (int32_t index = 0; index < n; ++index) {
                    bool toBreak = false;
                    for (int32_t index2 = 0; index2 < index; ++index2) {
                        if (compareSpans(idsRun[permut[index2]], idsRun[permut[index2 + 1]]) > 0) {
                            toBreak = true;
                            break;
                        }
                    }
                    if (toBreak) break;

                    IntEncodedExpr subEncoded[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
                    const int16_t subEncCount = static_cast<int16_t>(index + 1);
                    assert(subEncCount <= ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS
                        && "addToHashMemory subkey: element count exceeds cap");
                    for (int32_t t = 0; t <= index; ++t) {
                        subEncoded[t] = encodeExpression(tempRun[t], StrSpan("main", 4), nm);
                    }

                    int16_t subBuf[ExecutionParameters::MAX_KEY_SLOTS];
                    int16_t subLen = makeIntNormalizedKeyFromEncoded(
                        subEncoded, subEncCount, false, subBuf,
                        ExecutionParameters::MAX_KEY_SLOTS);

                    // D-72 owner record = the packed composite id (D-119: the same
                    // id is the LB-split partition cover), via the raw-key door
                    // (no owning NormKey).
                    mergeOwnerRecord(targetIntMemory.normalizedEncodedSubkeys,
                        static_cast<int16_t>(index + 1), subBuf, subLen,
                        partitionId, subEncoded, subEncCount, nm);
                    if (index + 1 == n - 1)
                        mergeOwnerRecord(targetIntMemory.normalizedEncodedSubkeysMinusOne,
                            static_cast<int16_t>(index + 1), subBuf, subLen,
                            partitionId, subEncoded, subEncCount, nm);
                    if (index + 1 == n - 2)
                        mergeOwnerRecord(targetIntMemory.normalizedEncodedSubkeysMinusTwo,
                            static_cast<int16_t>(index + 1), subBuf, subLen,
                            partitionId, subEncoded, subEncCount, nm);
                }
            }

            targetIntMemory.maxKeyLength = std::max(static_cast<int16_t>(n), targetIntMemory.maxKeyLength);
        }
    }

    /// @brief Path 2 of hash-engine rule installation — emit *marker* LMVs into
    /// `intHashMemory` for the per-element subkeys of an admitted implication.
    ///
    /// @details
    /// Where `addToHashMemory` installs the head-firing LMV, this function
    /// installs *marker* LMVs whose role is purely to gate the firing path —
    /// they don't carry a usable head, but their presence in the
    /// `normalizedEncodedSubkeys*` sets advertises that the rule is admissible
    /// for fast-rejection lookups.
    ///
    /// For each element index of the input `key`, the function:
    /// 1. Extracts the core expression and its declared output-index list via
    ///    `ce::getArgs` + the per-anchor `coreExpressionMap`.
    /// 2. Asserts there is at most one output argument
    ///    ([I-19](../../docs/agentic_swdd/30_invariants.md#i-19) — assert is first-class;
    ///    a firing assert here means the core map declared a multi-output
    ///    operator, which the rest of the engine does not handle).
    /// 3. Builds a binary mask + subkey by selecting other elements that do
    ///    NOT contain the output arg, then qualifies the result via two gates:
    ///    - the *classic* gate, baseline-only behavior preserved by the
    ///      static-pipeline migration; fires when the subkey size is exactly
    ///      `n - 1` and `baselineClassicQualifies` accepts.
    ///    - the *local-u* gate, used for `(C)` qualification (see
    ///      `implicationIsQualified` / `baselineClassicQualifies` in
    ///      prover.hpp); admits a wider class of subkeys but only when the
    ///      key has a `u_`-prefixed arg meeting the local-u criterion.
    /// 4. Emits the marker LMV via the same gen scratch arena byte-bump storage
    ///    contract as `addToHashMemory` — the int16_t key bytes live in the
    ///    arena so the resulting `IntNormalizedKey::data` pointer stays
    ///    stable for the rest of the LB's lifetime.
    ///
    /// `mbTrap` is reserved for instrumented builds; the runtime path
    /// drops it on the floor (`(void)mbTrap;` at entry).
    ///
    /// @param key                 Producer's key (string form).
    /// @param intHashMemory       Target hash memory (typically the same as
    ///                            `targetIntMemory` in the matching
    ///                            `addToHashMemory` call).
    /// @param nameMap             `NameMap` for int16_t encoding.
    /// @param arena               Gen scratch arena for stable storage of the
    ///                            normalized key bytes.
    /// @param value               Head expression text — used by the
    ///                            qualification gate but NOT stored in
    ///                            the resulting marker LMV.
    /// @param minNumOperatorsKey  Threshold for `implicationIsQualified`.
    /// @param mbTrap              Reserved (debug instrumentation hook).
    ///
    /// @pre  `coreExpressionMap` is populated for every core expression
    ///       referenced in `key` — populated at compiler-side init.
    /// @post `intHashMemory.normalizedEncodedSubkeys`,
    ///       `…SubkeysMinusOne`, and `…SubkeysMinusTwo` are extended with
    ///       fast-rejection entries for the qualifying subkeys.
    /// @invariant [I-19](../../docs/agentic_swdd/30_invariants.md#i-19) — the
    ///            single-output-arg assert here is intentional and must
    ///            not be weakened.
    /// @see [`HashMemory::normalizedEncodedSubkeys`](memory.hpp#hashmemory).
    /// @see `prover.hpp::implicationIsQualified`,
    ///      `prover.hpp::baselineClassicQualifies` — gate predicates.
    void ExpressionAnalyzer::makeNormalizedKeysForAdmission(
        const std::vector<std::string>& key,
        HashMemory& intHashMemory,
        NameMap& nameMap,
        ValueInterner& ruleInterner,
        const std::string& value,
        int minNumOperatorsKey,
        const std::string& originalImpl,
        const std::string& validityName,
        const Memory* mbTrap) {
        // Vector forwarder onto the span-run body (the additive twin) — build a
        // stack StrSpan run over the caller-stable vector strings, byte-identical.
        StrSpan keyRun[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS + 1];
        const int32_t keyFwdN = static_cast<int32_t>(key.size());
        assert(keyFwdN <= ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS + 1
            && "makeNormalizedKeysForAdmission forwarder: key size exceeds cap");
        for (int32_t i = 0; i < keyFwdN; ++i) keyRun[i] = StrSpan(key[i]);
        makeNormalizedKeysForAdmission(keyRun, keyFwdN, intHashMemory, nameMap,
            ruleInterner, StrSpan(value), minNumOperatorsKey,
            StrSpan(originalImpl), StrSpan(validityName), mbTrap);
    }

    void ExpressionAnalyzer::makeNormalizedKeysForAdmission(
        const StrSpan* key, int32_t keyN,
        HashMemory& intHashMemory,
        NameMap& nameMap,
        ValueInterner& ruleInterner,
        StrSpan value,
        int minNumOperatorsKey,
        StrSpan originalImpl,
        StrSpan validityName,
        const Memory* mbTrap) {

        (void)mbTrap;
        const int16_t ownerVid = nameMap.encode(validityName);

        // D-119: composite id of this marker rule's
        // (expanded-original, validity) pair — same packing as addToHashMemory
        // so a rule's head and marker entries carry the same partition id.
        const int32_t partitionId = makePartitionId(nameMap.encode(originalImpl), ownerVid);

        if (!implicationIsQualified(key, keyN, value, minNumOperatorsKey)) {
            return;
        }

        const std::size_t n = static_cast<std::size_t>(keyN);

        for (std::size_t index = 0; index < n; ++index) {
            const ce::CoreExpressionConfig* cfg =
                coreConfig(extractExpressionSpan(key[index]));
            assert(cfg && "Core expression not found");
            StrSpan args[ExecutionParameters::MAX_ARITY];
            const int32_t argsN = getArgsSpans(key[index], args,
                                               ExecutionParameters::MAX_ARITY);

            // Single output arg (0 or 1) — a StrSpan slice of key[index].
            constexpr int32_t kKeyCap =
                ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS + 1;
            assert(n <= static_cast<std::size_t>(kKeyCap)
                && "makeNormalizedKeysForAdmission: key size exceeds cap");
            StrSpan outputArg;
            bool haveOutput = false;
            for (int idx : cfg->outputIndices) {
                if (idx >= 0 && idx < argsN) {
                    assert(!haveOutput
                        && "Expression must have 0 or 1 output argument");
                    outputArg = args[idx];
                    haveOutput = true;
                }
            }
            if (!haveOutput) {
                continue;
            }

            // Binary mask + subkey (spans into key[], caller-stable).
            int binary[kKeyCap];
            for (int32_t i = 0; i < static_cast<int32_t>(n); ++i) binary[i] = 0;
            StrSpan subkey[kKeyCap];
            int32_t subCount = 0;
            int validCount = 0;
            for (std::size_t i = 0; i < n; ++i) {
                if (i == index) continue;
                StrSpan iArgs[ExecutionParameters::MAX_ARITY];
                const int32_t iArgsN = getArgsSpans(key[i], iArgs,
                                                    ExecutionParameters::MAX_ARITY);
                bool containsOutput = false;
                for (int32_t a = 0; a < iArgsN; ++a) {
                    if (equalSpans(iArgs[a], outputArg)) { containsOutput = true; break; }
                }
                if (!containsOutput) {
                    binary[i] = 1;
                    subkey[subCount++] = key[i];
                    validCount++;
                }
            }

            // Classic path is baseline behavior untouched — fires ONLY on
            // keys that would have qualified via (A) or (B) in baseline.
            // (C) qualification (hasUPrefix + localUCriterion) admits keys
            // into this loop but does NOT enable the classic per-element
            // gates. Those keys must go through the local-u path below.
            const bool baselineQualified = baselineClassicQualifies(key, keyN, value, minNumOperatorsKey);
            const bool acceptedByClassic = baselineQualified && (
                (validCount == static_cast<int>(n) - 1) ||
                (validCount == parameters.minLenLongKey - 1));
            const bool acceptedByLocalU = hasUPrefixInKey(key, keyN) &&
                this->elementMatchesLocalUCriterion(index, key, keyN, value);
            bool accept = acceptedByClassic || acceptedByLocalU;

            // Narrowing guard applies ONLY to the local u_-impl path — the
            // NEW case. Classic path behavior stays exactly as baseline.
            if (accept && !acceptedByClassic &&
                this->outputMatchesHeadOutputSlot(outputArg, value)) {
                accept = false;
            }

            if (!accept) continue;

            // Replace output arg with "marker" -> a ScratchString on the string
            // tier (replaceKeysScratch, byte-exact to ce::replaceKeysInString),
            // held across both blocks by mnkStrScope for this index.
            // Per-registry slot derivation (the two registries differ by one
            // slot; each fallback comes from its OWN slotCount() — see the
            // addToHashMemory site note).
            const unsigned mnkStrSlot = (g_currentCoreId >= 0)
                ? static_cast<unsigned>(g_currentCoreId)
                : scratchArenas().slotCount() - 1;
            const unsigned mnkGenSlot = (g_currentCoreId >= 0)
                ? static_cast<unsigned>(g_currentCoreId)
                : genScratchArenas().slotCount() - 1;
            ScratchArena& mnkStrArena = scratchArenas().forSlot(mnkStrSlot);
            ScratchScope mnkStrScope(mnkStrArena);
            StrReplacement markerPair[1];
            markerPair[0].key = outputArg;
            markerPair[0].value = StrSpan("marker", 6);
            const ScratchString replaced =
                replaceKeysScratch(mnkStrArena, key[index], markerPair, 1);

            // --- Int subkeys (mirrors makeNormalizedSubkeys for HashMemory) ---
            {
                const int32_t sn = subCount;
                auto pit = this->allPermutationsAna.find(sn);
                if (pit != this->allPermutationsAna.end()) {
                    const auto& permuts = pit->second;
                    StrSpan sids[kKeyCap];
                    for (int32_t i = 0; i < sn; ++i)
                        sids[i] = extractExpressionSpan(subkey[i]);

                    for (std::size_t p = 0; p < permuts.size(); ++p) {
                        const auto& permut = permuts[p];
                        StrSpan tempList[kKeyCap];
                        for (std::size_t k = 0; k < permut.size(); ++k)
                            tempList[k] = subkey[permut[k]];

                        for (int32_t si = 0; si < sn; ++si) {
                            bool toBreak = false;
                            for (int32_t si2 = 0; si2 < si; ++si2) {
                                if (compareSpans(sids[permut[si2]],
                                                 sids[permut[si2 + 1]]) > 0) {
                                    toBreak = true; break;
                                }
                            }
                            if (toBreak) break;

                            IntEncodedExpr subEncoded[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
                            const int16_t subEncCount = static_cast<int16_t>(si + 1);
                            assert(subEncCount <= ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS
                                && "makeNormalizedKeysForAdmission subkey: element count exceeds cap");
                            for (int32_t t = 0; t <= si; ++t)
                                subEncoded[t] = encodeExpression(
                                    tempList[t], StrSpan("main", 4), nameMap);

                            int16_t subBuf[ExecutionParameters::MAX_KEY_SLOTS];
                            int16_t subLen = makeIntNormalizedKeyFromEncoded(
                                subEncoded, subEncCount, false, subBuf, ExecutionParameters::MAX_KEY_SLOTS);
                            // Owner maps via the raw-key door (no owning NormKey).
                            mergeOwnerRecord(intHashMemory.normalizedEncodedSubkeys,
                                static_cast<int16_t>(si + 1), subBuf, subLen,
                                partitionId, subEncoded, subEncCount, nameMap);
                            if (si + 1 == sn - 1)
                                mergeOwnerRecord(intHashMemory.normalizedEncodedSubkeysMinusOne,
                                    static_cast<int16_t>(si + 1), subBuf, subLen,
                                    partitionId, subEncoded, subEncCount, nameMap);
                            if (si + 1 == sn - 2)
                                mergeOwnerRecord(intHashMemory.normalizedEncodedSubkeysMinusTwo,
                                    static_cast<int16_t>(si + 1), subBuf, subLen,
                                    partitionId, subEncoded, subEncCount, nameMap);
                        }
                    }
                }
            }

            // --- Main variants: create int keys for encodedMap marker entries ---
            {
                const int32_t sn = subCount;
                auto pit = this->allPermutationsAna.find(sn);
                if (pit == this->allPermutationsAna.end()) continue;
                const auto& permuts = pit->second;

                StrSpan sids[kKeyCap];
                for (int32_t i = 0; i < sn; ++i)
                    sids[i] = extractExpressionSpan(subkey[i]);

                ScratchArena& mnkGenArena = genScratchArenas().forSlot(mnkGenSlot);

                // getRemainingArgs yields a compareSpans sorted-unique run == the
                // former std::set<std::string> order. MINT the NameMap ids in that
                // run order (I-84: mint order reaches the deload stream), then a
                // SORTED-ASCENDING copy is the Int16SetKey (the former
                // std::set<int16_t> iteration order). The lmv.remainingArgIds mint
                // (ruleInterner) walks the run order = former set-lex order.
                StrSpan remScratch[ExecutionParameters::MAX_ADMISSION_REM_ARGS];
                const int32_t remScratchN = getRemainingArgs(
                    subkey, subCount, remScratch,
                    ExecutionParameters::MAX_ADMISSION_REM_ARGS);
                int16_t intRemArgs[ExecutionParameters::MAX_ADMISSION_REM_ARGS];
                for (int32_t i = 0; i < remScratchN; ++i)
                    intRemArgs[i] = nameMap.encode(remScratch[i]);   // mint in run order
                int16_t intRemArgsSorted[ExecutionParameters::MAX_ADMISSION_REM_ARGS];
                if (remScratchN > 0)
                    std::memcpy(intRemArgsSorted, intRemArgs,
                        static_cast<std::size_t>(remScratchN) * sizeof(int16_t));
                std::sort(intRemArgsSorted, intRemArgsSorted + remScratchN);

                int16_t bufIgnored[ExecutionParameters::MAX_KEY_SLOTS];
                int16_t bufNotIgnored[ExecutionParameters::MAX_KEY_SLOTS];

                for (std::size_t p = 0; p < permuts.size(); ++p) {
                    const auto& permutation = permuts[p];

                    bool toContinue = false;
                    if (sn >= 2) {
                        for (int32_t k = 0; k + 1 < sn; ++k) {
                            if (compareSpans(sids[permutation[k]],
                                             sids[permutation[k + 1]]) > 0) {
                                toContinue = true; break;
                            }
                        }
                    }
                    if (toContinue) continue;

                    // Per-permutation string window (valueVariant / replKey / decimals).
                    ScratchScope permScope(mnkStrArena);

                    // Span-native int encode (no heap EncodedExpression -> no
                    // parseArgument). The ignored buffer uses the WithMap variant
                    // (byte-identical outBuf) so its reverseMap can rebuild the
                    // value-variant mapping without makeNormalizedEncodedKey.
                    IntEncodedExpr intEncoded[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
                    const int16_t intEncCount = static_cast<int16_t>(permutation.size());
                    assert(intEncCount <= ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS
                        && "makeNormalizedKeysForAdmission: subkey element count exceeds cap");
                    for (std::size_t k = 0; k < permutation.size(); ++k)
                        intEncoded[k] = encodeExpression(
                            subkey[permutation[k]], StrSpan("main", 4), nameMap);

                    int16_t reverseMap[ExecutionParameters::MAX_KEY_SLOTS];
                    std::memset(reverseMap, 0, sizeof(reverseMap));
                    int16_t numNormVars = 0;
                    int16_t lenIgnored = makeIntNormalizedKeyFromEncodedWithMap(
                        intEncoded, intEncCount, true, bufIgnored,
                        ExecutionParameters::MAX_KEY_SLOTS, reverseMap, numNormVars);
                    int16_t lenNotIgnored = makeIntNormalizedKeyFromEncoded(
                        intEncoded, intEncCount, false, bufNotIgnored, ExecutionParameters::MAX_KEY_SLOTS);

                    // Rename run { normalized-var name -> decimal id } from
                    // reverseMap. mp2 order is NOT observable (replaceKeysScratch is
                    // greedy-longest, order-independent), so id order suffices —
                    // byte-identical to the former lex-ordered std::map. The key
                    // spans are NameMap decodeView: no NameMap mint falls between
                    // here and the last replaceKeysScratch use (the mints below are
                    // ruleInterner), so decodeView is I-3-safe (proved per-site).
                    StrReplacement mp2Pairs[ExecutionParameters::MAX_KEY_SLOTS];
                    int32_t mp2PairsN = 0;
                    for (int16_t id = 1; id <= numNormVars; ++id) {
                        assert(mp2PairsN < ExecutionParameters::MAX_KEY_SLOTS
                            && "makeNormalizedKeysForAdmission valueVariant: rename pair count exceeds cap");
                        char decTmp[12];
                        const int32_t decN = writeDecimalDigits(decTmp, id);
                        char* decBuf = mnkStrArena.allocBytes(decN);
                        std::memcpy(decBuf, decTmp, static_cast<std::size_t>(decN));
                        mp2Pairs[mp2PairsN].key = nameMap.decodeView(reverseMap[id]);
                        mp2Pairs[mp2PairsN].value = StrSpan(decBuf, decN);
                        ++mp2PairsN;
                    }

                    const ScratchString valueVariant =
                        replaceKeysScratch(mnkStrArena, StrSpan(replaced), mp2Pairs, mp2PairsN);

                    // Marker value id + isMarker flag (id-run door, no owning LMV).
                    const int32_t valueId = ruleInterner.encode(StrSpan(valueVariant));
                    const bool isMarker =
                        containsSpan(StrSpan(valueVariant), StrSpan("marker", 6));

                    // replKey ids: each valid subkey element (binary[i], subkey only
                    // — never key[index]) with the rename applied, minted in binary
                    // order == the former replKey vector order.
                    int32_t keyIds[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
                    int32_t keyIdsN = 0;
                    for (std::size_t i = 0; i < n; ++i) {
                        if (binary[i]) {
                            assert(keyIdsN < ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS
                                && "makeNormalizedKeysForAdmission replKey: element count exceeds cap");
                            const ScratchString e = replaceKeysScratch(
                                mnkStrArena, key[i], mp2Pairs, mp2PairsN);
                            keyIds[keyIdsN++] = ruleInterner.encode(StrSpan(e));
                        }
                    }

                    // remainingArgIds: walk remScratch (run order == former set-lex).
                    int32_t remIds[ExecutionParameters::MAX_ADMISSION_REM_ARGS];
                    for (int32_t i = 0; i < remScratchN; ++i)
                        remIds[i] = ruleInterner.encode(remScratch[i]);

                    const int32_t originalImplId = ruleInterner.encode(originalImpl);

                    // encodedMap marker record + owner record + remaining-args index,
                    // all via the raw / id-run doors (no owning NormKey / LMV). D-72:
                    // the LMV carries the admitting implication + scope.
                    appendLmvIdsRecord(intHashMemory.encodedMap,
                        static_cast<int16_t>(sn), bufIgnored, lenIgnored,
                        valueId, isMarker, keyIds, keyIdsN, remIds, remScratchN,
                        originalImplId, ownerVid, mnkGenArena);
                    mergeOwnerRecord(intHashMemory.normalizedEncodedKeys,
                        static_cast<int16_t>(sn), bufNotIgnored, lenNotIgnored,
                        partitionId, intEncoded, intEncCount, nameMap);
                    insertRemainingArgsNormKey(intHashMemory.remainingArgsNormalizedEncodedMap,
                        intRemArgsSorted, remScratchN,
                        static_cast<int16_t>(sn), bufNotIgnored, lenNotIgnored,
                        mnkGenArena);
                }
            }
        }
    }

    /// @brief Comparator on `EncodedExpression::name` — used when sorting
    /// statements by their core expression name.
    /// @details Stable lexicographic order over `.name`. Used when the static
    /// pipeline wants name-grouped iteration; pairs with `lessByOriginal` for
    /// the alternative ordering.
    bool ExpressionAnalyzer::lessByName(const EncodedExpression& a, const EncodedExpression& b) {
        return a.name < b.name;
    }

    /// @brief Comparator on `EncodedExpression::original` — used to sort by
    /// raw source text.
    /// @details Stable lexicographic order over `.original`, the un-rewritten
    /// expression string. Used by paths that need ordering by the
    /// user-visible form rather than by the internal name; pairs with
    /// `lessByName`.
    bool ExpressionAnalyzer::lessByOriginal(const EncodedExpression& a, const EncodedExpression& b) {
        return a.original < b.original;
    }

    // ------------------------------------------------------------------
    // Hash-engine mandatory-element filters (moved from prover.cpp in
    // commit C). Singles + Pairs versions used by the request-generator
    // pipeline.
    // ------------------------------------------------------------------

    /// @brief Singles variant of the mandatory-element filter — narrows a
    /// list of pre-encoded statements down to those that pass the
    /// `filterIntEncodedStatements` admission gate, then sorts by
    /// `originalId` for stable output.
    ///
    /// @details
    /// The static request pipeline operates on pre-encoded `IntEncodedExpr`
    /// arrays. Before generating requests it must filter the candidates so
    /// only statements whose normalized form is admissible against the
    /// rule's hash key remain. This singles variant handles the
    /// 1-statement case; `makeMandatoryEncodedStatementLists2Static` (in
    /// the same file) handles the pair case used during the
    /// `makeMandatory2` merge.
    ///
    /// @param mem        Reference `HashMemory` whose `normalizedEncoded*`
    ///                   sets the admission check consults.
    /// @param stmts      Pre-encoded statement view.
    /// @param outIndices Destination array for the surviving indices.
    /// @param maxOut     Capacity of `outIndices`.
    /// @return Number of indices written into `outIndices`. Capped at
    ///         `maxOut`; the local filter buffer is sized at 4096 which
    ///         comfortably exceeds any per-LB statement count seen so far.
    /// @pre  The container behind `stmts` lives at least until this
    ///       function returns.
    /// @post `outIndices[0..return-1]` is the surviving subset, sorted
    ///       ascending by `originalId`.
    int16_t ExpressionAnalyzer::makeMandatoryEncodedStatementLists1Static(
        const HashMemory& mem, const NameMap& nm,
        IntStmtView stmts,
        int16_t* outIndices, int16_t maxOut)
    {
        RT_SCOPE_HERE("MAKE_MANDATORY_LISTS_1_STATIC");
        // Filter, then sort by originalId (proxy for stable sort by original string)
        int16_t filtBuf[4096];
        int16_t nFilt = filterIntEncodedStatements(stmts, mem, nm, filtBuf, 4096);

        // Insertion sort filtered indices by originalId (counts are small)
        for (int16_t i = 1; i < nFilt; ++i) {
            int16_t key = filtBuf[i];
            int16_t j = i - 1;
            while (j >= 0 && stmts[filtBuf[j]].originalId > stmts[key].originalId) {
                filtBuf[j + 1] = filtBuf[j];
                --j;
            }
            filtBuf[j + 1] = key;
        }

        int16_t nOut = std::min(nFilt, maxOut);
        std::memcpy(outIndices, filtBuf, nOut * sizeof(int16_t));
        return nOut;
    }

    // ------------------------------------------------------------------
    // Hash-engine request generators (moved from prover.cpp in commit D).
    // filterIntEncodedStatements was inline in prover.hpp's class body;
    // its declaration stays there, the body now lives here alongside its
    // single caller cluster (the singles/pairs generators).
    // ------------------------------------------------------------------

    /// @brief Filter pre-encoded statements down to those that pass the
    /// fast-rejection subkey-set check and the max-iteration cap.
    ///
    /// @details
    /// For each `IntEncodedExpr`, builds a single-expression
    /// `IntNormalizedKey` on a stack buffer (no heap allocation), normalizes
    /// the var ids sequentially by first appearance (`changeable=0`,
    /// `ignoreU=false`), and tests:
    /// 1. Membership in `mem.normalizedEncodedSubkeys` — the bloom-filter-
    ///    style fast-rejection set populated by `addToHashMemory` /
    ///    `makeNormalizedKeysForAdmission`. Misses are dropped.
    /// 2. `maxIteration <= parameters.maxIterationNumberVariable` — caps
    ///    blow-up on iteration counters.
    ///
    /// Output is the surviving indices into `stmts`, in original order.
    /// No heap allocation; the in-function `buf` and `varMap` are bounded
    /// by `ExecutionParameters::MAX_KEY_SLOTS`.
    ///
    /// @param stmts       Source view of pre-encoded statements.
    /// @param mem         Reference `HashMemory` for the subkey-set check.
    /// @param outIndices  Destination array for surviving indices.
    /// @param maxOut      Capacity of `outIndices`.
    /// @return Count of surviving indices written into `outIndices`.
    /// @pre  The container behind `stmts` and `outIndices` live for the
    ///       duration of the call.
    int16_t ExpressionAnalyzer::filterIntEncodedStatements(
        IntStmtView stmts,
        const HashMemory& mem, const NameMap& nm,
        int16_t* outIndices, int16_t maxOut) {

        const int16_t count = static_cast<int16_t>(stmts.size());
        int16_t buf[ExecutionParameters::MAX_KEY_SLOTS];
        int16_t nOut = 0;

        for (int16_t i = 0; i < count && nOut < maxOut; ++i) {
            const IntEncodedExpr& s = stmts[i];

            // Build single-expr IntNormalizedKey on stack
            int16_t pos = 0;
            buf[pos++] = s.nameId;
            buf[pos++] = s.negation;
            for (int16_t j = 0; j < s.arity; ++j) {
                buf[pos++] = s.argId[j];
                buf[pos++] = 0; // changeable (ignoreU=false)
            }
            // Normalize: sequential IDs by first appearance
            {
                int16_t varMap[ExecutionParameters::MAX_KEY_SLOTS];
                int16_t nV = 0;
                int16_t nextN = 1;
                // Start after nameId+negation (pos 2), step by 2 (varId, changeable)
                for (int16_t p = 2; p < pos; p += 2) {
                    int16_t raw = buf[p];
                    int16_t norm = 0;
                    for (int16_t v = 0; v < nV; ++v) {
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

            // D-105/D-120: keep the statement only if an owner of its
            // single-element subkey is at a comparable scope and its u_ literals
            // are satisfiable by this statement's argFullId.
            const IntEncodedExpr* sp = &s;
            if (!ownerKeyAccepts(mem.normalizedEncodedSubkeys, buf, pos, nm, &sp, 1))
                continue;
            if (s.maxIteration > parameters.maxIterationNumberVariable)
                continue;

            outIndices[nOut++] = i;
        }
        return nOut;
    }

    /// @brief Shared grow loop for the *Singles* and *Pairs* variants of the
    /// mandatory-element filter — enumerates subsets of filtered statements
    /// up to `targetLen` collecting candidates whose subkey membership is
    /// in `targetSubkeys`.
    ///
    /// @details
    /// Implements the depth-first "grow" phase of `makeMandatory`. Starts with
    /// the empty set and grows by appending one filtered index at a time;
    /// at each depth checks whether the partial subkey is still a prefix of
    /// some target subkey (via `normalizedEncodedSubkeys` lookup) — pruning
    /// branches that cannot possibly converge. Each surviving candidate at
    /// `targetLen` is recorded as a `BaseCandidate` (its `allIdx[count]`
    /// statement indices plus the candidate's scope id).
    ///
    /// The function is hot — it runs inside `proveKernel`'s parallel hash
    /// burst — so it operates entirely on stack buffers and `int16_t` IDs.
    /// Per [I-28](../../docs/agentic_swdd/30_invariants.md#i-28), it does NOT write any
    /// shared cross-LB state during the parallel phase; results land only
    /// in the caller-supplied `baseCandidates` vector.
    ///
    /// @param body            Owning LB; supplies `nameMap` for any id work.
    /// @param intMemory       Reference hash memory for prefix-subkey lookup.
    /// @param allIntStmts     Pre-encoded statement view.
    /// @param filteredIdx     Indices into `allIntStmts` that passed the
    ///                        upstream filter pass.
    /// @param nFiltered       Length of `filteredIdx`.
    /// @param mainValidityId  Validity id used for candidates that match
    ///                        without forcing a deeper scope.
    /// @param targetLen       Subset size to grow to (typically 1 for
    ///                        Singles, 2 for Pairs).
    /// @param targetSubkeys   Set of subkeys the grow phase is steering
    ///                        toward.
    /// @param baseCandidates  Output vector. Each surviving subset is
    ///                        appended as a `BaseCandidate`.
    ///
    /// @pre  `allIntStmts` and `filteredIdx` live for the duration of the call.
    /// @post `baseCandidates` is appended (never cleared) with the
    ///       enumerated subsets.
    /// @invariant [I-28](../../docs/agentic_swdd/30_invariants.md#i-28) — no cross-LB
    ///            writes during the parallel phase.
    /// @see [`BaseCandidate`](memory.hpp#basecandidate).
    void ExpressionAnalyzer::growBaseCandidates(
        const Memory& body,
        const HashMemory& intMemory,
        IntStmtView allIntStmts,
        const int16_t* filteredIdx, int16_t nFiltered,
        int16_t mainValidityId,
        int targetLen,
        const TypedColdBlobMap<NormKey, OwnerSet>& targetSubkeys,
        unsigned coreId,
        PagedVector<BaseCandidate>& baseCandidates)
    {
        RT_SCOPE_HERE("GROW_BASE_CANDIDATES");
        const NameMap& nm = body.nameMap;
        struct StackItem {
            int start;
            int16_t allIdx[ExecutionParameters::MAX_EXPRESSIONS];
            int16_t count;
            int16_t validityId;
        };

        // The DFS frontier rides this slot's request-generation scratch arena on
        // its byte-bump tier: grow on push, reclaim on backtrack via popTo, so
        // the footprint tracks the live frontier, not the total nodes explored
        // (the per-batch containers ride the same arena's page tier, freed at
        // task end). The stack owns the byte-bump tail exclusively here.
        ScratchArena& dfsArena = genScratchArenas().forSlot(coreId);
        ArenaStack<StackItem> stack(dfsArena);
        {
            StackItem init;
            init.start = 0;
            init.count = 0;
            init.validityId = mainValidityId;
            stack.push(init);
        }

        while (!stack.empty()) {
            // Per-new-seed checkpoint: each stack pop is the start of
            // exploring one growing seed. Refresh the on-disk snapshot
            // here so a mid-burst reader sees which seed level the
            // grow loop is currently on.
            RT_REFRESH_HERE();

            StackItem top = stack.back();
            stack.pop();

            for (int i = top.start; i < nFiltered; ++i) {
                if (top.count + 1 > targetLen) break;
                // Submatch cap: stop growing once this part has reached the cap.
                // The submatch count is the burst-stop metric (replaces the former
                // emitted-request cap); the grow-DFS is the main submatch source,
                // so it must honor the cap too. g_growthMatchCount is this part's
                // tally. See D-109.
                if (g_growthMatchCount >= parameters.maxNumberHashRequests) return;

                const int16_t allIdx = filteredIdx[i];
                const IntEncodedExpr& ie = allIntStmts[allIdx];

                if (!nm.comparable(top.validityId, ie.validityId)) continue;
                int16_t newValidityId = nm.deeperOf(top.validityId, ie.validityId);

                const IntEncodedExpr* ptrs[ExecutionParameters::MAX_EXPRESSIONS];
                for (int16_t k = 0; k < top.count; ++k)
                    ptrs[k] = &allIntStmts[top.allIdx[k]];
                ptrs[top.count] = &ie;
                const int16_t newCount = static_cast<int16_t>(top.count + 1);

                std::pair<bool, IntNormalizedKey> pr =
                    preEvaluateFromEncoded(ptrs, newCount, body, mainValidityId,
                        intMemory.normalizedEncodedSubkeys, dfsArena);

                if (pr.first) {
                    // D-105/D-120: only record the base candidate if an owner of
                    // the matched target subkey is comparable + partition/u_ ok.
                    if (ownerKeyAccepts(targetSubkeys, pr.second.data,
                                        pr.second.length, nm, ptrs, newCount)) {
                        BaseCandidate bc;
                        std::memcpy(bc.allIdx, top.allIdx, top.count * sizeof(int16_t));
                        bc.allIdx[top.count] = allIdx;
                        bc.count = newCount;
                        bc.validityId = newValidityId;
                        baseCandidates.push_back(bc);
                    }
                    if (newCount < targetLen) {
                        StackItem next;
                        next.start = i + 1;
                        std::memcpy(next.allIdx, top.allIdx, top.count * sizeof(int16_t));
                        next.allIdx[top.count] = allIdx;
                        next.count = newCount;
                        next.validityId = newValidityId;
                        stack.push(next);
                    }
                }
            }
        }
    }

    /// @brief Singles variant of the static request generator — seed +
    /// grow + merge for one-mandatory-element kernel matches.
    ///
    /// @details
    /// Three-phase pipeline:
    /// - **Seed.** Each mandatory single is tested as a complete key
    ///   against `intMemory.normalizedEncodedKeys`. Hits emit a request via
    ///   `StaticRequestEmitter::emit`.
    /// - **Grow.** Statements are filtered through `filterIntEncodedStatements`
    ///   and sorted by core-expression name. `growBaseCandidates` enumerates
    ///   subsets of length `maxKeyLen - 1` whose subkey is in
    ///   `normalizedEncodedSubkeysMinusOne`.
    /// - **Merge.** Each `BaseCandidate` is paired with each mandatory
    ///   single; if `comparable(base.validityId, mandatory.validityId)`
    ///   passes and the merged set has no `originalId` duplicate, the
    ///   merged pointer array (in name-sorted order) is emitted as a
    ///   request.
    ///
    /// @param body                Owning LB; supplies `nameMap` and
    ///                            `intEncodedStatements`.
    /// @param intMemory           Reference hash memory (provides
    ///                            `maxKeyLength`, the `normalizedEncoded*`
    ///                            sets, and `comparable` for validity-id
    ///                            checks).
    /// @param mandatoryIndices    Indices into `mandatorySrcInt` of the
    ///                            mandatory singles.
    /// @param mandatoryCount      Length of `mandatoryIndices`.
    /// @param mandatorySrcInt     Source array for the mandatory entries.
    /// @param outBuf              Destination buffer for emitted requests.
    /// @param maxOut              Capacity of `outBuf`.
    /// @return Number of requests emitted (`StaticRequestEmitter::outCount`).
    /// @pre  `mandatoryCount > 0` to do useful work; the early-return at
    ///       `mandatoryCount <= 0` short-circuits.
    /// @post Each emitted request's `IntNormalizedKey::data` references
    ///       memory owned by the per-slot gen scratch arena; the pointer is stable
    ///       until that arena is released.
    /// @invariant [I-28](../../docs/agentic_swdd/30_invariants.md#i-28) — only
    ///            per-thread (arena, body) state is mutated.
    /// @see [`StaticRequestEmitter`](memory.hpp#staticrequestemitter).
    template <typename Consumer>
    void ExpressionAnalyzer::generateEncodedRequestsStatic(
        const Memory& body,
        const HashMemory& intMemory,
        const int16_t* mandatoryIndices, int16_t mandatoryCount,
        IntStmtView mandatorySrcInt,
        unsigned coreId,
        Consumer& consumer)
    {
        RT_SCOPE_HERE("GENERATE_ENCODED_REQUESTS_STATIC");
        if (mandatoryCount <= 0) return;

        const NameMap& nm = body.nameMap;
        int16_t buf[ExecutionParameters::MAX_KEY_SLOTS];
        const int16_t mainValidityId = NameMap::MAIN_ID;
        const int maxKeyLen = intMemory.maxKeyLength;
        const int targetLen = std::max(0, maxKeyLen - 1);

        // The request keys + the IntEncodedExpr copies ride this slot's gen
        // scratch arena byte-bump tier (no per-thread heap arena); persistent per task,
        // freed by the per-task releaseAll.
        ScratchArena& genArena = genScratchArenas().forSlot(coreId);
        StaticRequestEmitter<Consumer> emitter(genArena, consumer);

        // --- Seed: each mandatory single as complete key ---
        for (int16_t i = 0; i < mandatoryCount; ++i) {
            const IntEncodedExpr& ie = mandatorySrcInt[mandatoryIndices[i]];
            const IntEncodedExpr* ptr = &ie;
            int16_t len = makeIntNormalizedKeyFromEncoded(&ptr, 1, buf,
                ExecutionParameters::MAX_KEY_SLOTS);
            // D-105/D-120: keep the seed only if an owner of the matched key is
            // at a comparable scope and its u_ literals are satisfiable.
            if (ownerKeyAccepts(intMemory.normalizedEncodedKeys, buf, len, nm, &ptr, 1)) {
                int16_t* keyDst = reinterpret_cast<int16_t*>(genArena.resolve(
                    genArena.alloc(len * static_cast<int32_t>(sizeof(int16_t)),
                                   static_cast<int32_t>(alignof(int16_t)))));
                std::memcpy(keyDst, buf,
                            static_cast<std::size_t>(len) * sizeof(int16_t));
                IntNormalizedKey nk(1, keyDst, len);
                if (!emitter.emit(&ptr, 1, nk)) return;
            }
        }

        if (targetLen <= 0) return;

        // --- Filter + sort ---
        const IntStmtView allIntStmts(body.intEncodedStatements);
        int16_t filteredIdx[8192];
        int16_t nFiltered = filterIntEncodedStatements(allIntStmts,
            intMemory, nm, filteredIdx, 8192);
        // Name-only stable_sort. Emergence-order tie resolution is
        // deterministic by the stable_sort contract across MSVC STL
        // and libstdc++ — cross-host byte-identical at this site.
        // The Gauss / fold theorem proves under this tie order once
        // the integration-side admission machinery (the algebra-mirror
        // hooks landed in the prior release) closed the asymmetry
        // that had previously made the fold proof's search path
        // tie-order-sensitive.
        std::stable_sort(filteredIdx, filteredIdx + nFiltered, [&](int16_t a, int16_t b) {
            return compareSpans(nm.decodeView(allIntStmts[a].nameId),
                                nm.decodeView(allIntStmts[b].nameId)) < 0;
        });

        // --- Grow ---
        // baseCandidates accumulates the grow-phase survivors then is read by
        // index in the merge: rides this slot's gen scratch arena PAGE tier
        // (PagedVector), independent of the DFS stack's byte-bump tier on the
        // same arena. The throwaway dirty flag is unused (scratch never deloads).
        DirtyState dfsDirty = DirtyState::Clean;
        PagedVector<BaseCandidate> baseCandidates(
            &genScratchArenas().forSlot(coreId), &dfsDirty);
        growBaseCandidates(body, intMemory, allIntStmts, filteredIdx, nFiltered,
            mainValidityId, targetLen,
            intMemory.normalizedEncodedSubkeysMinusOne, coreId, baseCandidates);

        // --- Merge: baseCandidates × mandatory singles ---
        for (int32_t bi = 0; bi < baseCandidates.size(); ++bi) {
            const BaseCandidate& base = baseCandidates[bi];

            for (int16_t j = 0; j < mandatoryCount; ++j) {
                const int16_t mIdx = mandatoryIndices[j];
                const IntEncodedExpr& mIe = mandatorySrcInt[mIdx];
                const int16_t mValidityId = mIe.validityId;

                if (!nm.comparable(base.validityId, mValidityId)) continue;

                bool dup = false;
                for (int16_t k = 0; k < base.count && !dup; ++k) {
                    const IntEncodedExpr& bIe = allIntStmts[base.allIdx[k]];
                    if (bIe.originalId == mIe.originalId && nm.comparable(bIe.validityId, mIe.validityId)) {
                        dup = true;
                    }
                }
                if (dup) continue;

                // Build merged pointer array in name-sorted order
                const IntEncodedExpr* merged[ExecutionParameters::MAX_EXPRESSIONS + 1];
                int16_t mc = 0;
                const StrSpan mName = nm.decodeView(mIe.nameId);
                bool mInserted = false;
                for (int16_t k = 0; k < base.count; ++k) {
                    const StrSpan bName = nm.decodeView(allIntStmts[base.allIdx[k]].nameId);
                    if (!mInserted && !(compareSpans(bName, mName) < 0)) {
                        merged[mc++] = &mIe;
                        mInserted = true;
                    }
                    merged[mc++] = &allIntStmts[base.allIdx[k]];
                }
                if (!mInserted) merged[mc++] = &mIe;

                std::pair<bool, IntNormalizedKey> pr2 =
                    preEvaluateFromEncoded(merged, mc, body, mainValidityId,
                        intMemory.normalizedEncodedKeys, genArena);

                if (pr2.first) {
                    if (!emitter.emit(merged, mc, pr2.second)) return;
                }
            }
        }

        return;
    }

    /// @brief Pairs variant of the static request generator — seed +
    /// grow + merge for two-mandatory-element kernel matches.
    ///
    /// @details
    /// Mirror of `generateEncodedRequestsStatic` but consumes
    /// `MandatoryPair`s (each carrying an index into `firstSrcInt` and an
    /// index into `secondSrcInt`). Grow targets length `maxKeyLen - 2`
    /// using `normalizedEncodedSubkeysMinusTwo`. Merge produces
    /// `(base + first + second)` ordered by name; emits via
    /// `StaticRequestEmitter`.
    ///
    /// @param body          Owning LB.
    /// @param intMemory     Reference hash memory.
    /// @param pairs         Mandatory-pair array.
    /// @param pairCount     Length of `pairs`.
    /// @param firstSrcInt   Source array for the *first* element of each pair.
    /// @param secondSrcInt  Source array for the *second* element of each pair.
    /// @param outBuf        Destination buffer for emitted requests.
    /// @param maxOut        Capacity of `outBuf`.
    /// @return Number of requests emitted.
    /// @see `generateEncodedRequestsStatic` — singles counterpart.
    template <typename Consumer>
    void ExpressionAnalyzer::generateEncodedRequestsStaticPairs(
        const Memory& body,
        const HashMemory& intMemory,
        const MandatoryPair* pairs, int16_t pairCount,
        IntStmtView firstSrcInt,
        IntStmtView secondSrcInt,
        unsigned coreId,
        Consumer& consumer)
    {
        RT_SCOPE_HERE("GENERATE_ENCODED_REQUESTS_STATIC_PAIRS");
        if (pairCount <= 0) return;

        const NameMap& nm = body.nameMap;
        int16_t buf[ExecutionParameters::MAX_KEY_SLOTS];
        const int16_t mainValidityId = NameMap::MAIN_ID;
        const int maxKeyLen = intMemory.maxKeyLength;
        const int targetLen = std::max(0, maxKeyLen - 2);

        // The request keys + the IntEncodedExpr copies ride this slot's gen
        // scratch arena byte-bump tier (no per-thread heap arena); persistent per task,
        // freed by the per-task releaseAll.
        ScratchArena& genArena = genScratchArenas().forSlot(coreId);
        StaticRequestEmitter<Consumer> emitter(genArena, consumer);

        // Pre-sort each pair by name for consistent key generation. The pair
        // array rides this slot's gen scratch arena PAGE tier; the former
        // separate pairValid bitset is folded in as a per-pair flag. Built by
        // push_back, read by index. Throwaway dirty flag (scratch never deloads).
        struct SortedPair {
            IntEncodedExpr sorted[2];
            uint8_t valid;
        };
        DirtyState pairsDirty = DirtyState::Clean;
        PagedVector<SortedPair> sortedPairs(&genArena, &pairsDirty);

        for (int16_t i = 0; i < pairCount; ++i) {
            const IntEncodedExpr& e1 = firstSrcInt[pairs[i].idx1];
            const IntEncodedExpr& e2 = secondSrcInt[pairs[i].idx2];
            const StrSpan n1 = nm.decodeView(e1.nameId);
            const StrSpan n2 = nm.decodeView(e2.nameId);
            SortedPair sp;
            if (compareSpans(n1, n2) <= 0) { sp.sorted[0] = e1; sp.sorted[1] = e2; }
            else                          { sp.sorted[0] = e2; sp.sorted[1] = e1; }
            sp.valid = nm.comparable(e1.validityId, e2.validityId) ? 1 : 0;
            sortedPairs.push_back(sp);
        }

        // --- Seed: each pair as complete 2-element key ---
        for (int16_t i = 0; i < pairCount; ++i) {
            if (!sortedPairs[i].valid) continue;
            const IntEncodedExpr* ptrs[2] = { &sortedPairs[i].sorted[0], &sortedPairs[i].sorted[1] };
            int16_t len = makeIntNormalizedKeyFromEncoded(ptrs, 2, buf,
                ExecutionParameters::MAX_KEY_SLOTS);
            // D-105/D-120: keep the pair seed only if an owner of the matched key
            // is comparable to the pair's deeper scope and u_ literals satisfiable.
            if (ownerKeyAccepts(intMemory.normalizedEncodedKeys, buf, len, nm, ptrs, 2)) {
                int16_t* keyDst = reinterpret_cast<int16_t*>(genArena.resolve(
                    genArena.alloc(len * static_cast<int32_t>(sizeof(int16_t)),
                                   static_cast<int32_t>(alignof(int16_t)))));
                std::memcpy(keyDst, buf,
                            static_cast<std::size_t>(len) * sizeof(int16_t));
                IntNormalizedKey nk(2, keyDst, len);
                if (!emitter.emit(ptrs, 2, nk)) return;
            }
        }

        if (targetLen <= 0) return;

        // --- Filter + sort ---
        const IntStmtView allIntStmts(body.intEncodedStatements);
        int16_t filteredIdx[8192];
        int16_t nFiltered = filterIntEncodedStatements(allIntStmts,
            intMemory, nm, filteredIdx, 8192);
        // Name-only stable_sort — same rationale as the matching sort
        // in generateEncodedRequestsStatic above.
        std::stable_sort(filteredIdx, filteredIdx + nFiltered, [&](int16_t a, int16_t b) {
            return compareSpans(nm.decodeView(allIntStmts[a].nameId),
                                nm.decodeView(allIntStmts[b].nameId)) < 0;
        });

        // --- Grow ---
        // baseCandidates accumulates the grow-phase survivors then is read by
        // index in the merge: rides this slot's gen scratch arena PAGE tier
        // (PagedVector), independent of the DFS stack's byte-bump tier on the
        // same arena. The throwaway dirty flag is unused (scratch never deloads).
        DirtyState dfsDirty = DirtyState::Clean;
        PagedVector<BaseCandidate> baseCandidates(
            &genScratchArenas().forSlot(coreId), &dfsDirty);
        growBaseCandidates(body, intMemory, allIntStmts, filteredIdx, nFiltered,
            mainValidityId, targetLen,
            intMemory.normalizedEncodedSubkeysMinusTwo, coreId, baseCandidates);

        // --- Merge: baseCandidates × mandatory pairs ---
        for (int32_t bi = 0; bi < baseCandidates.size(); ++bi) {
            // Per-new-seed checkpoint: each baseCandidate is a fresh
            // growing seed about to be merged against every mandatory
            // pair. Fire RT_REFRESH_HERE so a mid-burst snapshot shows
            // we're inside the merge loop and which seed index we're on.
            RT_REFRESH_HERE();
            const BaseCandidate& base = baseCandidates[bi];

            for (int16_t pi = 0; pi < pairCount; ++pi) {
                if (!sortedPairs[pi].valid) continue;
                const IntEncodedExpr& m0 = sortedPairs[pi].sorted[0];
                const IntEncodedExpr& m1 = sortedPairs[pi].sorted[1];

                // m0 and m1 are already known comparable (pair filter above).
                int16_t pairVid = nm.deeperOf(m0.validityId, m1.validityId);
                if (!nm.comparable(base.validityId, pairVid)) continue;

                bool dup = false;
                for (int16_t k = 0; k < base.count && !dup; ++k) {
                    const IntEncodedExpr& bIe = allIntStmts[base.allIdx[k]];
                    if ((bIe.originalId == m0.originalId && nm.comparable(bIe.validityId, m0.validityId)) ||
                        (bIe.originalId == m1.originalId && nm.comparable(bIe.validityId, m1.validityId))) {
                        dup = true;
                    }
                }
                if (dup) continue;

                const IntEncodedExpr* merged[ExecutionParameters::MAX_EXPRESSIONS + 2];
                int16_t mc = 0;
                for (int16_t k = 0; k < base.count; ++k)
                    merged[mc++] = &allIntStmts[base.allIdx[k]];
                merged[mc++] = &m0;
                merged[mc++] = &m1;
                // Name-only stable_sort — same rationale as the
                // matching sorts above.
                std::stable_sort(merged, merged + mc, [&](const IntEncodedExpr* a, const IntEncodedExpr* b) {
                    return compareSpans(nm.decodeView(a->nameId),
                                        nm.decodeView(b->nameId)) < 0;
                });

                std::pair<bool, IntNormalizedKey> pr2 =
                    preEvaluateFromEncoded(merged, mc, body, mainValidityId,
                        intMemory.normalizedEncodedKeys, genArena);

                if (pr2.first) {
                    if (!emitter.emit(merged, mc, pr2.second)) return;
                }
            }
        }

        return;
    }

    // ------------------------------------------------------------------
    // Hash-engine lookup-and-fire (moved from prover.cpp in commit E).
    // The largest single body in the cluster (~300 LOC). Consumes one
    // StaticRequest produced by the singles/pairs/CE generators above and
    // either fires the matched rule (path 1, head LMV) or installs the
    // marker admission (path 2, marker LMV).
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
    /// @see `prover.hpp::addToHashMemory` — installer of the LMVs read here.
    /// @see `prover.hpp::addExprToMemoryBlock` — head-emit consumer.
    /// @see `prover.hpp::applyEquivalenceClass` — class-gated rewrite.
    void ExpressionAnalyzer::checkLocalEncodedMemoryStatic(
        const StaticRequest& req,
        const Memory& memoryBlock,
        unsigned coreId,
        SealedPageSet& sealedPages)
    {
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
        const int16_t reqCount = req.count;
        const int iteration = req.maxIteration + 1;
        const IntNormalizedKey& tple = req.normalizedKey;

        // 1. Validity check using int fields
        const int16_t mainValidityId = NameMap::MAIN_ID;
        int16_t consensusValidityId = mainValidityId;
        for (int16_t i = 0; i < reqCount; ++i) {
            const int16_t vid = req.intExprs[i]->validityId;
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
            for (int16_t i = 0; i < reqCount; ++i) {
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
            const int16_t anc = nm.ancAt(consensusValidityId, ancK);
            if (memoryBlock.intValidityNamesToFilter.contains(anc)) return;
        }

        // 2. productsOfRecursion check using int fields
        int algebraicCounter = 0;
        const auto& prodRecIds = memoryBlock.overallHashMemory.productsOfRecursionIds;
        for (int16_t i = 0; i < reqCount; ++i) {
            const IntEncodedExpr& ex = *req.intExprs[i];
            for (int16_t a = 0; a < ex.arity; ++a) {
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
        int16_t intAllArgs[kMaxIntAllArgs];
        int intAllArgsCount = 0;
        for (int16_t i = 0; i < reqCount; ++i) {
            const IntEncodedExpr& ex = *req.intExprs[i];
            for (int16_t a = 0; a < ex.arity; ++a) {
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
        for (int16_t i = 0; i < reqCount; ++i) {
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
            for (int16_t i = 0; i < reqCount; ++i) {
                if (levelLvIds[i] == 0) continue;
                const int32_t rl = memoryBlock.intStatementLevelsMap.runLen(levelLvIds[i]);
                for (int32_t j = 0; j < rl; ++j)
                    combinedBuf[w++] = memoryBlock.intStatementLevelsMap.valueAt(levelLvIds[i], j);
            }
            std::sort(combinedBuf, combinedBuf + combinedRawN);
            combinedCount = static_cast<int>(
                std::unique(combinedBuf, combinedBuf + combinedRawN) - combinedBuf);
        }

        // 5. Candidate loop: find remaining-arg sets that are subsets of
        // intAllArgs, enumerated in std::set<int16_t> lex order (R1). The former
        // std::vector<std::set<int16_t>> accumulator + std::sort is replaced by
        // a decoded-lex INDEX over the PASSING raMap ids: the subset test reads
        // count+ids straight off raMap.keyAt(id) (no per-candidate heap
        // Int16SetKey), and int16SetKeyLexCompare is the byte-for-byte twin of
        // std::set<int16_t>::operator< (signed element lex, shorter-is-prefix --
        // NOT the count-prefixed Int16SetKey byte order, which compares the
        // count field first). raMap keys are pairwise distinct -> the index is a
        // tie-free total order -> the identical enumeration sequence.
        const auto& raMap = memoryBlock.overallHashMemory.remainingArgsNormalizedEncodedMap;
        // Passing candidate ids on the PAGE tier -- raN is unbounded at Gauss
        // scale, never a stack array. Freed by candIds' own destructor (page
        // tier, disjoint from the byte-bump popTo below).
        DirtyState candDirty = DirtyState::Clean;
        PagedVector<int32_t> candIds(&genArena, &candDirty);
        {
            const int32_t raN = raMap.count();
            for (int32_t id = 1; id <= raN; ++id) {
                const StrSpan k = raMap.keyAt(id);
                const int16_t cn = int16SetKeyCount(k);
                bool subsetOk = true;
                for (int16_t a = 0; a < cn; ++a) {
                    if (!std::binary_search(intAllArgs, intAllArgs + intAllArgsCount,
                                            int16SetKeyIdAt(k, a))) {
                        subsetOk = false; break;
                    }
                }
                if (subsetOk) candIds.push_back(id);
            }
        }
        // Decoded-lex index (std::set<int16_t> order == R1) on the gen-scratch
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

        // tpleNorm probe bytes (Codec<NormKey>::encode layout: int16
        // numberExpressions, int16 length, length x int16 data) built directly
        // from the request's normalized key -- no heap NormKey. The nkRun
        // membership below is a zero-decode byte peek (Codec<NormKey>::serialize
        // == encode, injective), order-free (R2), any-match short-circuit.
        int16_t tpleProbe[2 + ExecutionParameters::MAX_KEY_SLOTS];
        assert(2 + tple.length <= 2 + ExecutionParameters::MAX_KEY_SLOTS);
        tpleProbe[0] = tple.numberExpressions;
        tpleProbe[1] = tple.length;
        std::memcpy(tpleProbe + 2, tple.data,
            static_cast<std::size_t>(tple.length) * sizeof(int16_t));
        const int32_t tpleProbeLen =
            (2 + tple.length) * static_cast<int32_t>(sizeof(int16_t));

        for (int32_t oi = 0; oi < candN; ++oi) {
            // The raMap id IS kid: lookup(decodeKey(id)) == id (the interner is
            // injective, id in [1,count]), so the former per-candidate
            // raMap.lookup(Int16SetKey{...}) + the kid==0 guard were dead.
            // Carrying the id directly is byte-identical and drops the heap key.
            const int32_t kid = candIds[candOrder[oi]];

            bool nkPresent = false;
            const int32_t rl = raMap.runLen(kid);
            for (int32_t j = 0; j < rl; ++j) {
                int32_t blen = 0;
                const char* bp = raMap.peekRecordBytes(kid, j, blen, genArena);
                if (blen == tpleProbeLen
                    && std::memcmp(bp, tpleProbe,
                                   static_cast<std::size_t>(tpleProbeLen)) == 0) {
                    nkPresent = true; break;
                }
            }
            if (!nkPresent) {
                continue;
            }

            // HIT -- rare (only rows whose run carries the request's tpleNorm).
            // Reconstruct the unchangeable-arg set for the unchanged
            // makeIntNormalizedKeyFromEncodedWithMap; an edge materialization per
            // HIT, not the per-candidate heap the scan above retired.
            const Int16SetKey candKey = raMap.decodeKey(kid);
            const std::set<int16_t> intSt(candKey.ids.begin(), candKey.ids.end());

            int16_t buf[ExecutionParameters::MAX_KEY_SLOTS];
            int16_t reverseMap[ExecutionParameters::MAX_KEY_SLOTS];
            std::memset(reverseMap, 0, sizeof(reverseMap));
            int16_t numNormVars = 0;

            // Build pointer array from request
            const IntEncodedExpr* reqExprs[ExecutionParameters::MAX_EXPRESSIONS];
            for (int16_t i = 0; i < reqCount; ++i) reqExprs[i] = req.intExprs[i];

            const int16_t len = makeIntNormalizedKeyFromEncodedWithMap(
                reqExprs, reqCount, intSt,
                buf, ExecutionParameters::MAX_KEY_SLOTS,
                reverseMap, numNormVars);

            // Serialize the probe to NormKey bytes (Codec<NormKey>::encode
            // layout: int16 numberExpressions, int16 length, length x int16
            // data) and raw-StrSpan lookup -- no per-probe heap key on the
            // engine's hottest path.
            int16_t probeBuf[2 + ExecutionParameters::MAX_KEY_SLOTS];
            probeBuf[0] = reqCount;
            probeBuf[1] = len;
            std::memcpy(probeBuf + 2, buf,
                static_cast<std::size_t>(len) * sizeof(int16_t));
            const int32_t encId =
                memoryBlock.overallHashMemory.encodedMap.lookup(
                    StrSpan(reinterpret_cast<const char*>(probeBuf),
                            static_cast<int32_t>((2 + len) * sizeof(int16_t))));
            if (encId == 0) {
                continue;
            }

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
            for (int16_t v = 1; v <= numNormVars; ++v) {
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
            for (int16_t i = 0; i < reqCount; ++i) premOrder[i] = i;
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

                if (!lmv.isMarker()) {
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
                    const int16_t lmvVid = lmv.validityId();
                    if (!nm.comparable(lmvVid, consensusValidityId)) {
                        continue;
                    }

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
                            for (int32_t li = 0; li < lmv.levelCount(); ++li)
                                tmp[w++] = lmv.levelAt(li);
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
                        for (int16_t i = 0; i < reqCount; ++i) {
                            const IntEncodedExpr* pe = req.intExprs[premOrder[i]];
                            sealedPremises[sealedPremisesCount++] =
                                SealedExpressionWithValidity{
                                    seal(nm.decodeView(pe->originalId)),
                                    seal(nm.decodeView(pe->validityId)) };
                        }
                        premisesSealed = true;
                    }

                    bool doNotDisintegrate = (lmv.justification() == RuleJustification::integration);
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
                    //       work, not solely by anchor / broadcast inputs).
                    // Without (1) the prover crashed at burst 2 / ~4570 exprs.
                    // Without (2) the prover crashed at burst 3 / ~7686 exprs.
                    // Both clauses are load-bearing.
                    // Pre-D-29 the prover crashed with NameMap exhaustion (cap
                    // 16384) at prover.hpp:807 — fold/sequence/etc. external
                    // rules from the prior IncubatorGauss batch fanning out
                    // across local (in[X, N]) rows.
                    if (parameters.incubator_mode && !parameters.ban_disintegration && !doNotDisintegrate) {
                        // Zero-allocation prefix test: exprKey starts with
                        // "(" + anchor name (the former per-record
                        // anchorPrefix concatenation, byte-identical
                        // outcome).
                        const std::string& anchorName = this->anchorInfo.name;
                        const StrSpan mbKey = memoryBlock.exprKeyView();
                        const bool isAnchorLB =
                            mbKey.len > static_cast<int32_t>(anchorName.size())
                            && mbKey.len > 0 && mbKey.ptr[0] == '('
                            && std::memcmp(mbKey.ptr + 1,
                                           anchorName.data(),
                                           anchorName.size()) == 0;
                        if (isAnchorLB) {
                            doNotDisintegrate = true;
                        } else {
                            // O(1) packed probes on the request's int rows —
                            // an exists-check, so premise order is irrelevant
                            // (the premOrder index sorts the premises for the
                            // origin tail only).
                            bool hasLocalPremise = false;
                            for (int16_t i = 0; i < reqCount; ++i) {
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


                    bool allGood = (consensusValidityId == NameMap::MAIN_ID);
                    for (int16_t i = 0; i < reqCount; ++i) {
                        if (!allowedForMail(nm.decodeView(req.intExprs[i]->originalId), memoryBlock)) {
                            allGood = false;
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
                        // per-firing heap std::string on this hot path.
                        const int16_t origId =
                            memoryBlock.nameMap.lookup(StrSpan(rplScratch2));
                        const int16_t valId  = consensusValidityId;
                        for (int32_t ancK = 0, ancN = memoryBlock.nameMap.ancLen(valId);
                             ancK < ancN; ++ancK) {
                            const int16_t anc = memoryBlock.nameMap.ancAt(valId, ancK);
                            const StatementFlags* kf = memoryBlock.intKnownStatements.find(StatementKey{ origId, anc });
                            if (kf != nullptr && kf->known) {
                                alreadyKnown = true;
                                break;
                            }
                        }
                    }

                    // Capture the head deposit. applyFiringRecords sorts the
                    // burst's records into canonical order and applies them:
                    // canBeSentIds on allGood int_lev carriers, then (when not alreadyKnown)
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
                            seal(validityView);
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
                        rec.allowOrDisintegration = lmv.productOfDisintegration();
                        rec.allGood = allGood;
                        rec.alreadyKnown = alreadyKnown;
                        sealedPages.appendRecord(rec);
                    }
                }
                else {
                    // Purity gate is relaxed ONLY inside OR-branch scopes
                    // (`_boundary_orint_` in the validity name). Case-split
                    // branches legitimately carry fresh it_/int_ witnesses
                    // that are iteration-bearing but not yet products of
                    // recursion — denying those admissions leaves §4.2-style
                    // ladder steps unreachable.
                    // Anywhere else (main / hypo / non-or boundary) the
                    // original purity gate stands, to prevent runaway fan-out.
                    const bool inOrBranch =
                        containsSpan(validityView,
                                     StrSpan("_boundary_orint_", 16));
                    if (!pure && !inOrBranch) {
                        continue;
                    }

                    StagedAdmissionValue admv;
                    {
                        // Replaced key elements in rule order: each stripped
                        // result is sealed onto the page set, the view array
                        // assembled in a gen-scratch buffer, then sealed as the
                        // staged value's key span.
                        const int kn = lmv.keyIdCount();
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
                            admv.remainingArgsSorted =
                                SealedSpan<SealedString>::copyFrom(
                                    sealedPages, remBuf, uniq);
                        }
                    }
                    admv.standardMaxAdmissionDepth = parameters.standardMaxAdmissionDepth;
                    admv.standardMaxSecondaryNumber = parameters.standardMaxSecondaryNumber;
                    admv.flag = false;

                    // The templateInterner probe and the compiledExpressions
                    // read below run span-native — the probe on the existing
                    // (StrSpan, StrSpan) lookupTemplateKey overload and the
                    // category read through the compiledEntity reader fence
                    // (I-137). No per-row std::string is
                    // materialized on this hot capture edge.

                    // Non-minting probe — this runs in the phase-2 parallel
                    // staging path (I-83); a never-interned template was
                    // never consumed. validityView is the decodeView of
                    // consensusValidityId (mint-free path, cannot dangle).
                    {
                        int32_t consumedPk = 0;
                        if (lookupTemplateKey(memoryBlock.templateInterner, memoryBlock.nameMap,
                                              StrSpan(rplScratch2), validityView, consumedPk)
                            && memoryBlock.overallHashMemory.consumedAdmissionKeys.contains(consumedPk)) {
                            continue;
                        }
                    }

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
                        compiledEntity(extractExpressionSpan(StrSpan(rplScratch2)));
                    assert(markerEntity && "marker rule head core expression must be compiled");
                    const bool markerNotAtomic =
                        !equalSpans(StrSpan(markerEntity->category), StrSpan("atomic", 6));

                    // Capture the marker deposit. applyFiringRecords replays it
                    // in canonical order: deferredIntegrationPreps (the
                    // integration-template registration), canBeSentMarkerIds
                    // (non-atomic), and admissionKeysAlgebra (the algebra-
                    // admissionMap staging drained by drainAdmissionKeysAlgebra in
                    // firing order). Capturing keeps the staged-vector drain order
                    // partition-independent. See D-117.
                    {
                        FiringRecord rec;
                        rec.isMarker = true;
                        rec.rplExpr2 = seal(StrSpan(rplScratch2));
                        rec.validityName =
                            seal(validityView);
                        // Bare args are bounded by MAX_ARITY, so seal them
                        // through a stack buffer onto the page set as one span.
                        SealedString markerArgsBuf[ExecutionParameters::MAX_ARITY];
                        for (int32_t ai = 0; ai < uniqueArgCount; ++ai)
                            markerArgsBuf[ai] = seal(bareArgs[ai]);
                        rec.markerArgsSorted = SealedSpan<SealedString>::copyFrom(
                            sealedPages, markerArgsBuf, uniqueArgCount);
                        rec.admv = admv;
                        rec.markerNotAtomic = markerNotAtomic;
                        sealedPages.appendRecord(rec);
                    }
                }
            }
        }
        // Free every gen-scratch run this request assembled (combinedLevels +
        // the per-firing level / admv buffers) in one tail-pop.
        genArena.popTo(genReqMark);
    }

    /// @brief Sort one hashburst's captured firing records into a canonical
    ///        content order, then apply their deposits to the LB.
    ///
    /// @details
    /// Definition of the consumer declared in `prover.hpp`. Replays the deposit
    /// side of `checkLocalEncodedMemoryStatic` for one static request-evaluation
    /// pass, reading the records off the LB's sealed part sets (part order,
    /// each chain in append order). A pointer INDEX over the chains' stable
    /// payload addresses is sorted by a total content key (expression,
    /// validity,
    /// kind, then per-kind fields) so the resulting container state is a
    /// function of the firing SET, not the request-generation ORDER — the
    /// determinism the LB split relies on. Head records feed `canBeSentIds`
    /// (gated on `allGood`) and, when not already known,
    /// `sameIterationInternalMail.statements` + `.exprOriginMap` (capped
    /// `addOrigin`) + `.disintegrationSignals`. Marker records feed
    /// `deferredIntegrationPreps`, `canBeSentMarkerIds` (non-atomic), and
    /// `admissionKeysAlgebra`. Deactivation already fired inline during the
    /// burst, so it is not handled here. The sealed views feed the span
    /// doors directly — no per-record string materialization at this
    /// boundary; the doors intern the sealed bytes into the LB's NameMap /
    /// originInterner with the identical touch order the string doors
    /// performed.
    ///
    /// @param memoryBlock LB whose containers receive the deposits.
    /// @param parts     The LB's sealed part sets in part order (chains read,
    ///                  never mutated).
    /// @param partCount Number of part sets (`>= 0`).
    /// @see `checkLocalEncodedMemoryStatic` — the producer.
    /// @see `FiringRecord` — the record shape.
    /// @see `D-117`.
    void ExpressionAnalyzer::applyFiringRecords(Memory& memoryBlock,
        SealedPageSet* const* parts, int32_t partCount)
    {
        // This finalize worker's slot + gen-scratch arena frame (the house
        // slot idiom): the record-pointer list and the sort index below are
        // per-call scratch, reclaimed by the single-exit popTo. The deposit
        // doors write LB-arena cold containers, interners, and heap staging
        // vectors — never this slot's gen arena — so the frame is LIFO-safe.
        const unsigned slot = (g_currentCoreId >= 0)
            ? static_cast<unsigned>(g_currentCoreId)
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

        // Pointer-INDEX std::sort, not stable_sort: the comparator below is a
        // strict total order over every deposit-affecting field (I-77), so no
        // two distinct firings tie and the stable variant's insertion-order
        // tie-break is moot. Sorting a pointer index under the same strict
        // total order yields the identical content sequence as the former
        // in-place record sort: a strict-total-order sort has a unique output
        // sequence, and content-identical duplicates are byte-indistinguishable
        // in the deposit stream — so the deposit bytes cannot depend on input
        // permutation or sort algorithm.
        const int32_t total = static_cast<int32_t>(refs.size());
        // Named capacity tripwire (Rule 19): the contiguous index must fit
        // one arena block; a scale overrun stops HERE, reading as the
        // designed ceiling it is — see the constant's Doxygen for the
        // page-tier indirect-sort widening path.
        assert(total <= ExecutionParameters::kMaxFiringRecordsPerLbBurst
            && "firing-record sort index exceeds one arena block — widen via "
               "a page-tier indirect sort, see kMaxFiringRecordsPerLbBurst");
        int32_t* idx = nullptr;
        if (total > 0) {
            idx = reinterpret_cast<int32_t*>(gArena.resolve(gArena.alloc(
                total * static_cast<int32_t>(sizeof(int32_t)),
                static_cast<int32_t>(alignof(int32_t)))));
            for (int32_t k = 0; k < total; ++k) idx[k] = k;
            const auto recordLess =
                [](const FiringRecord& a, const FiringRecord& b) -> bool {
                // Sealed views compare by content (compareSpans ==
                // std::string byte order), so the canonical order is
                // byte-identical to the former string-member sort.
                int c = compareSpans(StrSpan(a.rplExpr2), StrSpan(b.rplExpr2));
                if (c != 0) return c < 0;
                c = compareSpans(StrSpan(a.validityName), StrSpan(b.validityName));
                if (c != 0) return c < 0;
                if (a.isMarker != b.isMarker) return a.isMarker < b.isMarker; // head before marker
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
                    if (a.allGood != b.allGood) return a.allGood < b.allGood;
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
                if (a.markerNotAtomic != b.markerNotAtomic)
                    return a.markerNotAtomic < b.markerNotAtomic;
                return false;
            };
            std::sort(idx, idx + total, [&](int32_t x, int32_t y) {
                return recordLess(*refs[x], *refs[y]);
            });
        }

        const int maxOrigins = parameters.compressor_mode
            ? parameters.compressor_max_origins_per_expr
            : parameters.max_origin_per_expr;

        for (int32_t ri = 0; ri < total; ++ri) {
            const FiringRecord& rec = *refs[idx[ri]];
            if (!rec.isMarker) {
                // Sealed views feed the span doors directly — no head-record
                // materialization at this deposit boundary; the doors intern
                // the bytes into the LB's NameMap (statements / signals) and
                // originInterner (origins). Single-threaded drain, so
                // minting is legal. The spans alias the sealed page set,
                // stable until the post-drain sweep frees it (I-3).
                const StrSpan expr(rec.rplExpr2);
                const StrSpan validity(rec.validityName);
                if (rec.allGood && containsSpan(expr, StrSpan("int_lev_", 8))) {
                    // `allowedForMail` probes the memo only for
                    // `int_lev_*` carriers — any other entry could never
                    // be read.
                    memoryBlock.canBeSentIds.mint(
                        memoryBlock.nameMap.encode(expr));
                }
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
                        rec.doNotDisintegrate, rec.allowOrDisintegration);
                }
            } else {
                // Marker deposits stay sealed end-to-end: the staging
                // vectors carry the views, the drains materialize at their
                // mint points, and performElemPhase2 clears the vectors
                // post-drain — before the post-join sweep frees the pages.
                memoryBlock.deferredIntegrationPreps.push_back(
                    DeferredIntegrationPrep{ rec.rplExpr2,
                                             rec.markerArgsSorted,
                                             rec.validityName });
                if (rec.markerNotAtomic) {
                    memoryBlock.canBeSentMarkerIds.mint(
                        memoryBlock.nameMap.encode(StrSpan(rec.rplExpr2)));
                }
                memoryBlock.admissionKeysAlgebra.push_back(
                    AdmissionKeyAlgebraRecord{
                        SealedExpressionWithValidity{ rec.rplExpr2,
                                                      rec.validityName },
                        rec.admv});
            }
        }
        // Reclaim the index + the refs list's byte-bump footprint; refs's
        // page-tier pages are freed by its destructor (independent tier).
        gArena.popTo(mark);
    }

    /// @brief Replays the per-burst staged algebra-`admissionMap` writes
    ///        recorded on `Memory::admissionKeysAlgebra`, once, after the
    ///        hashburst fixpoint loop.
    ///
    /// @details
    /// Companion drain to the record append in
    /// `checkLocalEncodedMemoryStatic`'s marker branch. For each
    /// `AdmissionKeyAlgebraRecord` on `memoryBlock.admissionKeysAlgebra`, in
    /// firing order, it replays the four writes the marker branch used to
    /// perform inline mid-burst:
    ///   1. re-check the consumed-key gate — skip the record if an earlier
    ///      record's `revisitRejected2` (transitively `cleanAdmissionMap`) has
    ///      since consumed this key, matching the inline loop's within-burst
    ///      consume→skip ordering;
    ///   2. insert `record.value` into `overallHashMemory.admissionMap` under
    ///      `record.key` (fresh per-record lookup; create the set if absent);
    ///   3. set `admissionStatusMap[record.key] = false`;
    ///   4. add every non-`marker` argument of `record.key.original` to
    ///      `varsInAdmissionMapKeys`;
    ///   5. fire `revisitRejected2`, depositing any revival cohort on
    ///      `sameIterationInternalMail`.
    /// Called once from `performElemPhase2`, immediately before phase 3's
    /// post-burst `standardProcessing`, so the revival cohorts are present for
    /// that absorb and `admissionMap` is current before its `isAdmitted` reads
    /// it.
    ///
    /// @param memoryBlock The LB whose `admissionKeysAlgebra` buffer is
    ///        replayed and whose `overallHashMemory` admission containers
    ///        receive the writes. The buffer is left intact; the burst-start
    ///        clear in `performElem2` empties it.
    /// @return (void)
    /// @invariant Drains in append (firing) order — the canonical-closure scan
    ///            in `cleanAdmissionMap` reached via `revisitRejected2` is
    ///            order-sensitive.
    /// @invariant FROZEN interner-touch sequence per record: (1)
    ///            `templateInterner.encode(key.original)` inside
    ///            `mintTemplateKey`, (2) `nameMap.encode(key.validityName)`
    ///            inside `mintTemplateKey`, (3) `valueInterner.encode` per
    ///            staged key element in RULE order inside `stagedToArenaBlob`,
    ///            (4) `valueInterner.encode` per staged remainingArg in
    ///            presorted LEX order inside `stagedToArenaBlob` (the same
    ///            positions the retained `stagedToIdValue` oracle touches),
    ///            (5)
    ///            `templateInterner.encode` per non-`"marker"` argument of
    ///            `key.original` in `getArgs` order, (6) `revisitRejected2`'s
    ///            interior mints. Same bytes at the same positions →
    ///            identical ids → identical deload streams.
    /// @see `memory.cpp::checkLocalEncodedMemoryStatic` — the producer.
    /// @see `AdmissionKeyAlgebraRecord` — the staged record type.
    void ExpressionAnalyzer::drainAdmissionKeysAlgebra(Memory& memoryBlock)
    {
        // This drain worker's slot + gen-scratch arena (the house idiom):
        // hosts each record's serialized admission blob + the RMW splice's
        // verbatim run copies, reclaimed per record by the popTo below.
        const unsigned drainSlot = (g_currentCoreId >= 0)
            ? static_cast<unsigned>(g_currentCoreId)
            : genScratchArenas().slotCount() - 1;
        ScratchArena& drainArena = genScratchArenas().forSlot(drainSlot);
        for (const AdmissionKeyAlgebraRecord& rec : memoryBlock.admissionKeysAlgebra) {
            // The staged record carries the template as SEALED VIEWS (the
            // parallel producer must not mint — I-68/I-83); this drain is
            // the single-threaded write site that interns the sealed views
            // through the span doors — no materialization. The spans alias
            // the record's page set, sealed (stable) until the post-drain
            // sweep, so they stay valid across the whole record replay.
            const StrSpan keyOriginal(rec.key.original);
            const StrSpan keyValidity(rec.key.validityName);
            const int32_t recPk = mintTemplateKey(memoryBlock.templateInterner,
                memoryBlock.nameMap, keyOriginal, keyValidity);

            // (1) Consumed-key gate, re-applied per record: an earlier record's
            // revisitRejected2 (transitively cleanAdmissionMap) may have
            // consumed this key since it was staged — matches the inline
            // loop's within-burst consume→skip ordering.
            if (memoryBlock.overallHashMemory.consumedAdmissionKeys.contains(recPk)) {
                continue;
            }

            // (2) admissionMap insert — this drain is the single-threaded
            // intern point for the staged value (I-68/I-83). Fast path: the
            // staged strings encode STRAIGHT into the serialized blob
            // (stagedToArenaBlob — the stagedToIdValue interner-touch order
            // preserved byte-for-byte) and the blob splices into the
            // canonical run with no heap value and no heap set.
            {
                const ArenaOffset recMark = drainArena.cursor();
                const StrSpan blob = stagedToArenaBlob(
                    rec.value, memoryBlock.valueInterner, drainArena);
                insertAdmissionBlobSorted(
                    memoryBlock.overallHashMemory.admissionMap, recPk,
                    blob.ptr, blob.len, memoryBlock.valueInterner, drainArena);
                drainArena.popTo(recMark);
            }

            // (3) admissionStatusMap.
            memoryBlock.overallHashMemory.admissionStatusMap.upsert(recPk,
                static_cast<uint8_t>(0));

            // (4) vars-in-keys cache for the algebra equi-class hook. The
            // args are zero-copy span slices, encoded per-arg in
            // ce::getArgs left-to-right order (getArgsSpans is the
            // documented content+order twin).
            StrSpan argSpans[ExecutionParameters::MAX_ARITY];
            const int argN = getArgsSpans(keyOriginal, argSpans,
                ExecutionParameters::MAX_ARITY);
            for (int i = 0; i < argN; ++i) {
                const StrSpan a = argSpans[i];
                if (!equalSpans(a, StrSpan("marker", 6))) {
                    memoryBlock.overallHashMemory.varsInAdmissionMapKeys.mint(
                        memoryBlock.templateInterner.encode(a));
                }
            }

            // (5) Revival — deposits the rejection cohort on
            // sameIterationInternalMail (spans over the sealed record).
            this->revisitRejected2(keyOriginal, memoryBlock, keyValidity);
        }
    }

    /// @brief Replays the per-burst staged integration-template registrations
    ///        recorded on `Memory::deferredIntegrationPreps`, once, after the
    ///        hashburst fixpoint loop.
    ///
    /// @details
    /// Integration-side companion of `drainAdmissionKeysAlgebra`. For each
    /// `DeferredIntegrationPrep` on `memoryBlock.deferredIntegrationPreps`, in
    /// firing order, it replays the `prepareIntegration` call that
    /// `checkLocalEncodedMemoryStatic`'s marker branch deferred — registering the
    /// integration admission template (`admissionMapIntegration` plus the
    /// `varsInAdmissionMapIntegrationKeys` cache, via `prepareIntegrationCore2`).
    /// The sealed views feed the `prepareIntegration` span core directly —
    /// no set rebuild and no string materialization at this boundary; the
    /// core's two compiled-definition edge strings are built inside it, past
    /// its early-return gates. Called once from `performElemPhase2`,
    /// immediately before phase 3's post-burst `standardProcessing`, so the
    /// templates are registered before that absorb's
    /// `updateAdmissionMapIntegration` cascade consumes them.
    ///
    /// @param memoryBlock The LB whose `deferredIntegrationPreps` buffer is
    ///        replayed; its `overallHashMemory` integration-admission containers
    ///        receive the writes. The buffer is left intact; the burst-start
    ///        clear in `performElem2` empties it.
    /// @return (void)
    /// @invariant Drains in append (firing) order.
    /// @see `memory.cpp::checkLocalEncodedMemoryStatic` — the producer.
    /// @see `DeferredIntegrationPrep` — the staged record type.
    void ExpressionAnalyzer::drainDeferredIntegrationPreps(Memory& memoryBlock)
    {
        for (const DeferredIntegrationPrep& rec : memoryBlock.deferredIntegrationPreps) {
            // Sealed views pass straight through the prepareIntegration span
            // core — no set rebuild, no materialization at this boundary
            // (the core's two compiled-definition edge strings live inside
            // it, past the gates). The ascending-strict assert pins the
            // producer contract the pass-through relies on: the record's
            // args are the former set as a sorted-unique sealed span
            // (FiringRecord.markerArgsSorted, forwarded verbatim by
            // applyFiringRecords). A firing assert is a gift (Rule 19).
            const int32_t n = rec.unchangeableArgsSorted.size();
            assert(n <= ExecutionParameters::MAX_ARITY);
            StrSpan args[ExecutionParameters::MAX_ARITY];
            for (int32_t i = 0; i < n; ++i) {
                args[i] = StrSpan(rec.unchangeableArgsSorted[i]);
                assert((i == 0 || compareSpans(args[i - 1], args[i]) < 0)
                    && "deferred integration args not sorted-unique — producer contract broken");
            }
            prepareIntegration(StrSpan(rec.expression), args, n,
                               memoryBlock, StrSpan(rec.validityName));
        }
    }

    /// @brief Pairs variant of the mandatory-element filter — emits index
    /// pairs whose merged subkey is mutually mandatory in `mem`.
    ///
    /// @details
    /// Two-layer counterpart to `makeMandatoryEncodedStatementLists1Static`.
    /// Both `first` and `second` are first filtered through
    /// `filterIntEncodedStatements`; the surviving indices are sorted by
    /// `originalId`; then each candidate pair `(i ∈ first, j ∈ second)` is
    /// tested for mutual mandatoryness against `mem`'s
    /// `normalizedEncodedSubkeysMinusOne` set. Surviving pairs are written
    /// as `MandatoryPair{ idx1: i, idx2: j }` into `outPairs`, capped at
    /// `maxOut`.
    ///
    /// O(n²) over the filtered counts; in practice both filters cut the
    /// statement count by an order of magnitude or two, so the inner loop
    /// stays bounded.
    ///
    /// @param body         Owning LB; supplies `nameMap`.
    /// @param mem          Reference hash memory.
    /// @param first        First-layer source array.
    /// @param firstCount   Length of `first`.
    /// @param second       Second-layer source array.
    /// @param secondCount  Length of `second`.
    /// @param outPairs     Destination buffer for emitted pairs.
    /// @param maxOut       Capacity of `outPairs`.
    /// @return Count of pairs emitted into `outPairs`.
    /// @pre  `first` and `second` live for the duration of the call.
    /// @post `outPairs[0..return-1]` contains the mutually-mandatory pairs.
    /// @see `generateEncodedRequestsStaticPairs` — primary consumer.
    int16_t ExpressionAnalyzer::makeMandatoryEncodedStatementLists2Static(
        const Memory& body, const HashMemory& mem,
        IntStmtView first,
        IntStmtView second,
        MandatoryPair* outPairs, int16_t maxOut)
    {
        RT_SCOPE_HERE("MAKE_MANDATORY_LISTS_2_STATIC");
        if (first.empty() || second.empty()) return 0;

        // Filter both layers
        int16_t filt1Buf[4096], filt2Buf[4096];
        int16_t nF1 = filterIntEncodedStatements(first, mem, body.nameMap, filt1Buf, 4096);
        int16_t nF2 = filterIntEncodedStatements(second, mem, body.nameMap, filt2Buf, 4096);

        // Sort by originalId
        auto sortByOriginal = [](int16_t* arr, int16_t n, IntStmtView stmts) {
            for (int16_t i = 1; i < n; ++i) {
                int16_t key = arr[i];
                int16_t j = i - 1;
                while (j >= 0 && stmts[arr[j]].originalId > stmts[key].originalId) {
                    arr[j + 1] = arr[j];
                    --j;
                }
                arr[j + 1] = key;
            }
        };
        sortByOriginal(filt1Buf, nF1, first);
        sortByOriginal(filt2Buf, nF2, second);

        // "main" ID is guaranteed to be NameMap::MAIN_ID (== 1).
        int16_t mainId = NameMap::MAIN_ID;

        int16_t nOut = 0;
        int16_t buf[ExecutionParameters::MAX_KEY_SLOTS];

        for (int16_t i = 0; i < nF1 && nOut < maxOut; ++i) {
            const IntEncodedExpr& e1 = first[filt1Buf[i]];
            for (int16_t j = 0; j < nF2 && nOut < maxOut; ++j) {
                const IntEncodedExpr& e2 = second[filt2Buf[j]];

                // Validity check
                if (e1.validityId != e2.validityId) {
                    if (e1.validityId != mainId && e2.validityId != mainId) continue;
                }

                // Skip identical expressions
                if (e1.originalId == e2.originalId && e1.validityId == e2.validityId)
                    continue;

                // Build sorted pair and preEvaluate — sort by name string (lexicographic),
                // NOT by nameId (insertion order), to match storage sort in addToHashMemory
                IntEncodedExpr sorted[2];
                const StrSpan name1 = body.nameMap.decodeView(e1.nameId);
                const StrSpan name2 = body.nameMap.decodeView(e2.nameId);
                if (compareSpans(name1, name2) <= 0) { sorted[0] = e1; sorted[1] = e2; }
                else { sorted[0] = e2; sorted[1] = e1; }

                int16_t len = makeIntNormalizedKeyFromEncoded(sorted, 2, false, buf,
                    ExecutionParameters::MAX_KEY_SLOTS);

                // D-105/D-120: keep the mandatory pair only if an owner of the
                // matched subkey is comparable to the pair's deeper scope and u_
                // literals satisfiable (e1/e2 comparable by the validity check above).
                const IntEncodedExpr* sortedPtrs[2] = { &sorted[0], &sorted[1] };
                if (ownerKeyAccepts(mem.normalizedEncodedSubkeys, buf, len,
                                    body.nameMap, sortedPtrs, 2)) {
                    outPairs[nOut].idx1 = filt1Buf[i];
                    outPairs[nOut].idx2 = filt2Buf[j];
                    ++nOut;
                }
            }
        }
        return nOut;
    }

    // ------------------------------------------------------------------
    // Memory::wipeSubtree — radical subtree wipe on implication closure.
    // Takes the closed scope's NameMap validity id (`closedVid`, minted
    // and non-main, both assert-enforced).
    //
    // Replaces the pre-existing selective cleanup
    // (`cleanUpIntegrationPreparation*` and the equivalent per-victim
    // work inside `cleanUpOrIntegrationBranches`). See the Doxygen block
    // above the declaration in memory.hpp for the full structure list,
    // preserved-on-purpose rationale, and references to invariants.
    //
    // Implementation note. Membership is the `validityNodes` forest walk:
    // an id is closed iff `closedVid` lies on its parent chain, self
    // included (`NameMap::collectClosedSubtreeIds`). This is provably
    // equivalent to the retired text predicate `validityName ==
    // closedScope OR validityName starts with closedScope + "_boundary_"`
    // because per [I-2](../../docs/agentic_swdd/30_invariants.md#i-2) every
    // non-`"main"` scope name was minted via
    // `NameMap::encodePush(parent, payload)` whose payload assert forbids
    // embedded delimiters — each name's `"_boundary_"` decomposition is
    // unique and equals its recorded parent chain
    // (I-139). The one non-NameMap id space —
    // `expandedImplications`' lbStateInterner scope half — keeps the text
    // gate as a zero-copy span twin.
    // ------------------------------------------------------------------
    void Memory::wipeSubtree(int16_t closedVid) {
        assert(closedVid > 0 && closedVid <= nameMap.nameCount()
            && "wipeSubtree: closedVid is not a minted NameMap id");
        assert(closedVid != NameMap::MAIN_ID
            && "wipeSubtree: 'main' is never closed");

        // Closed-subtree membership via ONE validityNodes forest walk
        // (`collectClosedSubtreeIds`): bit `id` is set iff `closedVid` is
        // on id's parent chain (self included) — provably equivalent to
        // the retired text predicate `v == closedScope || v starts with
        // closedScope + "_boundary_"` (I-139: the
        // encodePush payload assert makes every name's delimiter
        // decomposition unique and equal to its recorded parent chain).
        // The 4 KB bitmap is stack; the ascending id vector rides the
        // per-slot gen-scratch PAGE tier (I-124) and doubles as the
        // step-11 mint order. Every int-keyed sweep below consults the
        // bitmap; one O(nameCount x depth) walk amortizes the membership
        // cost across all of them.
        const unsigned slot = (ExpressionAnalyzer::g_currentCoreId >= 0)
            ? static_cast<unsigned>(ExpressionAnalyzer::g_currentCoreId)
            : genScratchArenas().slotCount() - 1;
        ScratchArena& gArena = genScratchArenas().forSlot(slot);
        uint64_t closedBits[512];                       // 32768 bits, stack
        DirtyState ascDirty = DirtyState::Clean;
        PagedVector<int16_t> closedAsc(&gArena, &ascDirty);
        const int32_t nameHighWater = nameMap.nameCount();
        nameMap.collectClosedSubtreeIds(closedVid, closedBits, 512, closedAsc);
        const auto inClosedBit = [&](int32_t v) -> bool {
            // EXACT twin of the former closedIds.count(x): x can arrive
            // from int16_t casts whose value is NEGATIVE (e.g. the low
            // half of a GLOBAL mailIn id at the routing-mail filter) —
            // count(negative) was false; the bitmap guards the index the
            // same way, never indexing out of range.
            return v >= 1 && v <= nameHighWater
                && ((closedBits[static_cast<uint16_t>(v) >> 6]
                     >> (static_cast<uint16_t>(v) & 63)) & 1ull) != 0;
        };

        // ---- 1. statement registry ----
        // Keyed on the bitmap — a stored row's validity is interned, so
        // bit membership of its validityId is exactly the forest predicate.
        // Erase back to front to keep indices valid.
        for (int32_t i = intEncodedStatements.size(); i-- > 0; ) {
            if (inClosedBit(intEncodedStatements[i].validityId)) {
                intEncodedStatements.erase(i);
            }
        }

        // ---- 2. local statement registry + Set + Delta ----
        // The set mirrors the local-statement vector (I-86); drop its closed
        // keys in ONE eraseIf (O(n)) rather than a cold-index rebuild per row.
        intLocalEncodedStatementsSet.eraseIf([&](int32_t k) {
            return inClosedBit(static_cast<int16_t>(k & 0xFFFF));
        });
        for (int32_t i = intLocalEncodedStatements.size(); i-- > 0; ) {
            if (inClosedBit(intLocalEncodedStatements[i].validityId)) {
                intLocalEncodedStatements.erase(i);
            }
        }
        for (int32_t i = intLocalEncodedStatementsDelta.size(); i-- > 0; ) {
            if (inClosedBit(intLocalEncodedStatementsDelta[i].validityId)) {
                intLocalEncodedStatementsDelta.erase(i);
            }
        }

        // ---- 3. toBeProved ----
        // A stored goal's validity is interned, so bitmap membership of
        // the low 16 key bits is exactly the forest predicate (same
        // argument as the intStatementLevelsMap sweep below).
        intToBeProved.eraseSetIf([&](const StatementKey& k) {
            return inClosedBit(k.validity);
        });

        // (Section 4 retired: the former wholeExpressions sweep — the
        // registered membership now lives in intKnownStatements, erased
        // in section 6 by the same bitmap predicate.)

        // ---- 5. intStatementLevelsMap (packed (origId, validityId)) ----
        // Safe to wipe because wipeSubtree now runs at end-of-burst
        // (drained by `performElemPhase3` immediately
        // after `sanitizeToBeProved`); the kernel's `sortedNew` loop
        // assertions all finished earlier in the burst. Low 16 key bits
        // carry the validityId, and a stored row's validity is interned, so
        // bitmap membership is exactly the forest predicate
        // (same argument as the intKnownStatements sweep below).
        intStatementLevelsMap.eraseSetIf([&](const StatementKey& k) {
            return inClosedBit(k.validity);
        });

        // ---- 6. intKnownStatements (packed (origId, validityId) — low 16 bits) ----
        intKnownStatements.eraseIf([&](const StatementKey& k) {
            return inClosedBit(k.validity);
        });

        // ---- 7. equivalenceClassesMap (keyed by validity id) ----
        // Safe to wipe at end-of-burst (see step 5 comment). Run-aware
        // compacting erase on the cold blob map (Batch 3).
        equivalenceClassesMap.eraseBlobIf([&](int16_t vId) {
            return inClosedBit(vId);
        });

        // ---- 8. Packed-keyed sets ----
        // expandedImplications: k.low is an lbStateInterner id — a DIFFERENT
        // id space than the NameMap bitmap — so this one sweep keeps a
        // string-shaped predicate, as a zero-copy span twin of the retired
        // `v == closedScope || prefix-with-delimiter` gate (strict `>`:
        // a name equal to closedScope + "_boundary_" with empty tail cannot
        // exist, but the twin keeps the exact gate). Lifetime (I-3): eraseIf
        // mutates only this TypedColdSet — nothing mints into nameMap or
        // lbStateInterner during the sweep, so both decodeView spans stay
        // valid across it (different-container rule).
        {
            const StrSpan closedName = nameMap.decodeView(closedVid);
            const StrSpan boundaryTok(NameMap::BOUNDARY_STR,
                                      static_cast<int32_t>(NameMap::BOUNDARY_LEN));
            expandedImplications.eraseIf([&](const LbStatePairKey& k) {
                const StrSpan v = lbStateInterner.decodeView(k.low);
                if (equalSpans(v, closedName)) return true;
                return v.len > closedName.len + boundaryTok.len
                    && equalSpans(StrSpan(v.ptr, closedName.len), closedName)
                    && equalSpans(StrSpan(v.ptr + closedName.len,
                                          boundaryTok.len),
                                  boundaryTok);
            });
        }
        // Packed template-space twins — same low-16-bits validity
        // predicate as the other packed sweeps.
        auto filterPackedSet = [&](ColdHashSet<PodKeyStore<int32_t>>& s) {
            s.eraseIf([&](int32_t k) {
                return inClosedBit(static_cast<int16_t>(k & 0xFFFF));
            });
        };
        filterPackedSet(integrationPrepared);
        filterPackedSet(integrationPreparedMarker);

        // Packed twin of weakVariables — same low-16-bits validity
        // predicate as the intKnownStatements sweep in step 6.
        intWeakVariables.eraseIf([&](int32_t k) {
            return inClosedBit(static_cast<int16_t>(k & 0xFFFF));
        });

        // ---- 9. Mail statements (in / out / internal) ----
        // For the routing channels (mailIn / mailOut) per I-26 every item
        // is "main" and the wipe is a no-op except for the internal
        // revival inbox. Walk all three for symmetry.
        // Routing channels are COLD deloadable: the same scope-targeted erase
        // via RoutingColdMail::filterStatements. A guaranteed no-op in practice
        // (every routing item is "main", never closed — I-26), kept explicit
        // rather than skipped (Rule 19).
        auto filterRoutingMailStatements = [&](RoutingColdMail& m) {
            // Routing mail is main-only (I-26), so this never actually removes
            // anything; kept explicit (Rule 19). mailOut holds SENDER NameMap ids
            // (bitmap-comparable); mailIn holds GLOBAL ids and is a no-op (main
            // is never closed; the guarded cast keeps count()'s
            // negative-value semantics). The id-form key carries the
            // validityId either way.
            m.filterStatements(
                [&](const IntMailStatementKey& k) {
                    return inClosedBit(static_cast<int16_t>(k.validityId));
                });
        };
        filterRoutingMailStatements(mailIn);
        filterRoutingMailStatements(mailOut);
        // sameIterationInternalMail is COLD (I-102): the
        // same scope-targeted erase via ColdMail::filterStatements. Only
        // statements is swept (origins / disintegrationSignals untouched),
        // matching the former heap filter. The id-form key carries the
        // statement's NameMap validityId, so the closed-scope test is the same
        // bitmap membership the intStatementLevelsMap / intKnownStatements
        // sweeps use (no decode).
        sameIterationInternalMail.filterStatements(
            [&](const IntMailStatementKey& k) {
                return inClosedBit(static_cast<int16_t>(k.validityId));
            });

        // ---- 10. HashMemory wipe (overall / local / delta) ----
        auto wipeHashMem = [&](HashMemory& hm) {
            // 10a. encodedMap LMVs: filter each key's run by the LMV's
            //      scope (a fixed-offset byte peek, no record decode);
            //      drop the key when its run empties. Verbatim two-pass
            //      blob splice through the raw doors — survivor key order,
            //      per-key blob order, and blob bytes identical to the
            //      former decode-erase-reencode rebuild.
            wipeEncodedMapForClosed(hm.encodedMap, closedBits, nameHighWater,
                                    gArena);

            // 10b. Four owner-set maps (COLD blob maps): zero-decode
            //      blob-byte filter — drop owners whose scope (the LOW half
            //      of the packed composite id, D-105) is closed, drop keys
            //      whose owner-set empties, splice survivors verbatim.
            //      normalizedEncodedKeys' dropped keys are collected (as
            //      encoded key bytes — Codec<NormKey>::serialize is
            //      byte-identical to the key encode) so 10c can prune the
            //      secondary index too. Fresh per HashMemory instance,
            //      matching the former per-hm unordered_set scoping.
            DirtyState droppedDirty = DirtyState::Clean;
            ColdHashSet<BytesKeyStore> droppedKeys(&gArena, &droppedDirty);
            wipeOwnerSetMapForClosed(hm.normalizedEncodedKeys, closedBits,
                                     nameHighWater, gArena, &droppedKeys);
            wipeOwnerSetMapForClosed(hm.normalizedEncodedSubkeys, closedBits,
                                     nameHighWater, gArena, nullptr);
            wipeOwnerSetMapForClosed(hm.normalizedEncodedSubkeysMinusOne,
                                     closedBits, nameHighWater, gArena,
                                     nullptr);
            wipeOwnerSetMapForClosed(hm.normalizedEncodedSubkeysMinusTwo,
                                     closedBits, nameHighWater, gArena,
                                     nullptr);

            // 10c. remainingArgsNormalizedEncodedMap secondary index: a
            //      record IS an encoded NormKey, so membership against the
            //      dropped-key byte set is a raw byte-peek — no decode.
            wipeRemainingArgsForClosed(hm.remainingArgsNormalizedEncodedMap,
                                       droppedKeys, gArena);

            // Cold admission/rejection containers
            // (D-172, D-173)
            // use the compacting / run-aware cold erase with the same low-16-bits
            // closed-scope predicate.
            const auto coldScopeWipe = [&](int32_t k) {
                return inClosedBit(static_cast<int16_t>(
                    static_cast<uint32_t>(k) & 0xFFFF));
            };
            hm.admissionMap.eraseBlobIf(coldScopeWipe);
            hm.admissionStatusMap.eraseIf(coldScopeWipe);
            hm.rejectedMap.eraseBlobIf(coldScopeWipe);
            hm.admissionMapIntegration.eraseBlobIf(coldScopeWipe);
            hm.rejectedMapIntegration.eraseBlobIf(coldScopeWipe);
            hm.consumedAdmissionKeys.eraseIf(coldScopeWipe);
            hm.revisitInProgress.eraseIf(coldScopeWipe);
            hm.admissionSetIntegration.eraseIf(coldScopeWipe);
            hm.triggersForAdmissionSetIntegration.eraseIf(coldScopeWipe);
        };
        wipeHashMem(overallHashMemory);
        wipeHashMem(localHashMemory);
        wipeHashMem(localHashMemoryDelta);

        // ---- 11. Belt-and-suspenders filter inserts. Site H ancestor-scan
        //          on subsequent addExprToMemoryBlock entry will block any
        //          in-flight mail referencing the closed subtree.
        //          Ascending id order (closedAsc is built ascending):
        //          intValidityNamesToFilter is deload-enrolled (its KeysView
        //          facet streams keys in id == mint order), so the mint
        //          order here is deload-byte-observable and must be a pure
        //          function of content, never hash-container iteration
        //          history (D-190).
        for (int32_t i = 0; i < closedAsc.size(); ++i)
            intValidityNamesToFilter.mint(closedAsc[i]);
    }

    // ------------------------------------------------------------------
    // Memory::cloneFactsTemplate — a fresh single-use CE LB from a facts
    // template. See the Doxygen block above the declaration in memory.hpp
    // for the value-copy / reset rationale and the empty-template contract.
    // ------------------------------------------------------------------
    Memory* Memory::cloneFactsTemplate(LbStore& store) const {
        assert(overallHashMemory.encodedMap.empty()
            && "cloneFactsTemplate: template must carry no hash rules "
               "(facts load via the status-4 statement path)");

        Memory* out = store.create<Memory>();

        // Fact-statement containers — the outputs of the status-4 load. All
        // pure value types (EncodedExpression / IntEncodedExpr / packed keys),
        // so copy-assignment is a safe deep copy.
        out->intEncodedStatements           = intEncodedStatements;
        out->intLocalEncodedStatementsSet.copyFrom(intLocalEncodedStatementsSet);
        out->intLocalEncodedStatements      = intLocalEncodedStatements;
        out->intLocalEncodedStatementsDelta = intLocalEncodedStatementsDelta;
        out->intStatementLevelsMap.copyFrom(intStatementLevelsMap);
        out->intKnownStatements.copyFrom(intKnownStatements);

        // Name dictionary — independent copy so the clone mints the conjecture's
        // names without touching the shared template. The façade copies via
        // copyFrom: the cold tables re-intern in id order onto the CLONE's
        // manager (every id matches the template's), the id-form metadata
        // copies as plain containers.
        out->nameMap.copyFrom(nameMap);

        // Scalars the template carries from its load (everything else stays
        // Memory()-default).
        out->level        = level;
        out->startInt     = startInt;
        out->startIntRepl = startIntRepl;
        out->startIntPi   = startIntPi;

        // Everything else (overallHashMemory, mail, flags,
        // equivalence / origin / intToBeProved) stays at its Memory() default:
        // empty containers, isActive=true,
        // contradictionIndex=-1, primedForContradiction=false. intToBeProved is
        // now a persistent-pool member (its own fresh persistentArena), so the
        // clone starts it empty automatically — no copyFrom, by design.
        return out;
    }

    /// @brief The process-wide, never-deloaded skeleton string interner. See
    ///        the declaration in `memory.hpp` for the full contract.
    ColdStringTable& skeletonInterner() {
        static LbArena arena{ &lbMemory() };
        static DirtyState dirty = DirtyState::Clean;
        static ColdStringTable table{ &arena, &dirty };
        return table;
    }

    /// @brief The process-wide, never-deloaded global mail string interner. See
    ///        the declaration in `memory.hpp` for the full contract.
    ColdStringTable& mailInterner() {
        static LbArena arena{ &mailMemory() };
        static DirtyState dirty = DirtyState::Clean;
        static ColdStringTable table{ &arena, &dirty };
        return table;
    }

    /// @brief The LB's expression key as a string. See the declaration.
    std::string Memory::exprKey() const {
        return exprKeyId == 0 ? std::string()
                              : skeletonInterner().decodeString(exprKeyId);
    }

    /// @brief The LB's expression key as a zero-copy span. See the declaration.
    StrSpan Memory::exprKeyView() const {
        return exprKeyId == 0 ? StrSpan(nullptr, 0)
                              : skeletonInterner().view(exprKeyId);
    }

    /// @brief Set the LB's expression key, interning it. See the declaration.
    void Memory::setExprKey(const std::string& s) {
        exprKeyId = s.empty() ? 0 : skeletonInterner().intern(s);
    }

    /// @brief Build an LB's full chain string — every `exprKey` from the
    ///        LB itself up to the root sentinel, joined by `__`.
    ///
    /// @details
    /// The statification deload identity (file naming, header chain
    /// field): same walk as the hashburst dump's LB-chain section,
    /// flattened to one string. The root sentinel's empty `exprKey` is
    /// included as the final (empty) element, so a child of the root
    /// ends in a trailing `__` and the root alone yields the empty
    /// string. The full parent chain — never a bare `exprKey` — is the
    /// LB identity (Rule 12: many LBs share an `exprKey` at different
    /// tree positions).
    ///
    /// @param body The LB to identify.
    /// @return The chain string, innermost first.
    /// @see `lbdeload::deloadFileName`, `Memory::deloadStaticContainers`.
    std::string buildLbChainString(const Memory& body) {
        std::string chain;
        const Memory* cur = &body;
        bool first = true;
        while (cur != nullptr) {
            if (!first) chain += "__";
            chain += cur->exprKey();
            first = false;
            cur = cur->parentMemory;
        }
        return chain;
    }

    /// @brief Assign this LB's deload ordinal on first sight and record its
    ///        `ordinal -> chain` mapping in the global registry.
    ///
    /// @details
    /// See the declaration for the full contract. The `deloadOrdinal < 0`
    /// guard is the single assignment that makes the file-name identity
    /// lifetime-stable; the chain rebuild is a short parent-chain walk (the
    /// LB's stable identity, Rule 12). Re-registers every call so the
    /// per-batch-cleared registry always carries the current LB.
    void Memory::ensureDeloadOrdinal() {
        if (deloadOrdinal < 0)
            deloadOrdinal = staticMemory().assignDeloadOrdinal();
        staticMemory().registerDeloadOrdinal(deloadOrdinal,
                                              buildLbChainString(*this));
    }

    /// @brief Stream the statified containers to their `.deload/` file
    ///        set, return the blocks to the pool, and flag the LB
    ///        deloaded.
    ///
    /// @details
    /// Composes the standalone primitives in the contract order:
    /// `lbdeload::dumpLbMemory` (canonical element-order stream —
    /// I-103), then `release()` on every
    /// statified container (pages back to the LB queue), then
    /// `manager.releaseAll()` (blocks back to the global pool), then
    /// `markDeloaded()`. The file list + chain land in `deloadFiles` /
    /// `deloadChain` for `ensureLoaded`. Asserts the LB is resident — a
    /// double deload is a lifecycle bug, not a no-op.
    ///
    /// @param directory The deload directory (production: `.deload`).
    void Memory::dumpStaticContainers(const std::string& directory) {
        assert(lbMemory.manager.resident()
            && "dumpStaticContainers on a deloaded LB");
        deloadChain = buildLbChainString(*this);
        // Fallback single-assignment for direct (test / teardown) callers;
        // production pre-stamps at the barrier, making this idempotent.
        ensureDeloadOrdinal();
        // Three-way deload decision (skip / tail-delta / full rewrite —
        // the WAL/threshold pattern, all branches defined results):
        // - Clean: the on-disk file set already equals the in-memory
        //   content — release blocks without writing.
        // - AppendedOnly within thresholds: only push_backs happened
        //   since the last dump — write a small tail file set with the
        //   new rows; base files stay untouched.
        // - Restructured, first dump, or thresholds exceeded: full
        //   canonical rewrite (which is also where the straightening
        //   re-canonicalizes the whole image).
        const bool haveBase = !deloadFiles.empty();
        if (lbMemory.dirty == DirtyState::Clean && haveBase) {
            // Skip: nothing changed since RAM == disk was established.
        }
        else {
            // Current totals (tag order) and the appended-row volume relative to
            // the counts recorded at the LAST deload. The four HashMemory
            // instances ride lbMemory.visitContainers now
            // (D-147), so their tags 51..450 enter the counts
            // AND the dump through the one enumeration.
            std::vector<int32_t> currentCounts;
            lbMemory.visitContainers(
                [&currentCounts](LbMemory::ContainerTag,
                                 const auto& container) {
                    currentCounts.push_back(container.size());
                });
            int64_t newTailRows = 0;
            if (haveBase && deloadedCounts.size() == currentCounts.size()) {
                for (std::size_t i = 0; i < currentCounts.size(); ++i)
                    newTailRows += currentCounts[i] - deloadedCounts[i];
            }
            const bool tailEligible =
                lbMemory.dirty == DirtyState::AppendedOnly
                && haveBase
                && deloadedCounts.size() == currentCounts.size()
                && deloadTailCount < lbdeload::kMaxTailSets
                && (deloadTailRows + newTailRows)
                       * lbdeload::kTailCompactionDenominator
                   < deloadBaseRows;
            if (tailEligible) {
                const std::vector<std::string> tailFiles =
                    lbdeload::dumpLbMemoryTail(
                        lbMemory, deloadChain, deloadOrdinal, deloadedCounts,
                        static_cast<uint32_t>(deloadTailCount + 1),
                        directory, staticMemory().blockBytes());
                deloadFiles.insert(deloadFiles.end(), tailFiles.begin(),
                                   tailFiles.end());
                ++deloadTailCount;
                deloadTailRows += newTailRows;
            }
            else {
                deloadFiles = lbdeload::dumpLbMemory(
                    lbMemory, deloadChain, deloadOrdinal, directory,
                    staticMemory().blockBytes());
                deloadTailCount = 0;
                deloadTailRows = 0;
                deloadBaseRows = 0;
                for (const int32_t c : currentCounts)
                    deloadBaseRows += c;
            }
            // Record per-container totals (tag order) — the only legal
            // metadata about the containers while the LB is cold, and
            // the next tail's start positions.
            deloadedCounts = currentCounts;
        }
        // RAM == disk holds from here until the next mutation.
        lbMemory.dirty = DirtyState::Clean;
    }

    /// @brief Return the LB's blocks to the pool and flag it deloaded —
    ///        legal only when the on-disk image is fresh.
    ///
    /// @details
    /// The pressure path: asserts residency, a recorded file set, and
    /// `DirtyState::Clean` (i.e., `dumpStaticContainers` ran after the
    /// last mutation — releasing stale-imaged blocks would lose
    /// content). Releases every statified container's pages, returns the
    /// blocks, `markDeloaded()`.
    void Memory::releaseStaticBlocks() {
        assert(lbMemory.manager.resident()
            && "releaseStaticBlocks on an already deloaded LB");
        assert(!deloadFiles.empty()
            && "releaseStaticBlocks without an on-disk image");
        assert(lbMemory.dirty == DirtyState::Clean
            && "releaseStaticBlocks with a stale on-disk image - dump "
               "first");
        lbMemory.visitContainers(
            [](LbMemory::ContainerTag, auto& container) {
                container.release();
            });
        // The four HashMemory instances now ride lbMemory.visitContainers
        // (D-147), so the walk above releases their cold
        // containers too -- the former manual per-instance release (needed while
        // they lived outside LbMemory) is retired.
        // The release() calls escalate the dirty state (they are
        // mutators); the next reload re-establishes RAM == disk.
        lbMemory.dirty = DirtyState::Clean;
        lbMemory.manager.releaseAll();
        lbMemory.manager.markDeloaded();
    }

    /// @brief Dump + release in one call — the unconditional-deload
    ///        composition (tests, teardown paths).
    ///
    /// @param directory The deload directory (production: `.deload`).
    void Memory::deloadStaticContainers(const std::string& directory) {
        dumpStaticContainers(directory);
        releaseStaticBlocks();
    }

    /// @brief The discharge protocol's memory step: capture the exact
    ///        registry record, empty every dischargeable container, and
    ///        reshuffle — the copying compaction returns every block except
    ///        the cold-string survivors' to the pool with zero SSD I/O.
    ///
    /// @details
    /// See the declaration for the full contract. Order is load-bearing:
    /// the capture reads `intEncodedStatements` BEFORE the clear; the
    /// reshuffle after the clear reclaims the freed storage through the
    /// copying compaction (see `LbMemory::reshuffle`).
    /// `intLocalEncodedStatementsSet` — the local-origin gate's exact
    /// mirror — survives discharge as a cold tag (`LbMemory::survivesDischarge`
    /// keeps it), so the recursion-node gate still reads it post-prove.
    /// The persistent goal registry `intToBeProved` is reclaimed here too
    /// (resetToFresh then `persistentArena.releaseAll()`) — its block was
    /// attached only until discharge ([I-108]).
    void Memory::dischargeStatementContent(LbArena& scr) {
        assert(lbMemory.manager.resident()
            && "discharge on a deloaded LB - the protocol runs at the "
               "barrier of the LB's last active iteration");
        assert(!dischargedForever
            && "dischargeStatementContent ran twice on one LB");
        dischargedForever = true;
        for (int32_t r = 0; r < intEncodedStatements.size(); ++r) {
            dischargedRegistryKeys.insert(packStatementKey(
                intEncodedStatements[r].originalId,
                intEncodedStatements[r].validityId));
        }
        // Reclaim the persistent goal registry: its block is attached to the LB
        // only until discharge (I-108). resetToFresh frees
        // its pages into persistentArena FIRST, then releaseAll returns the
        // arena's blocks to persistentMemory() (the reverse order would free
        // pages into a released arena and assert). A discharged LB never reads
        // intToBeProved again (it is !isActive; ensureLoaded asserts the flag).
        intToBeProved.resetToFresh();
        persistentArena.releaseAll();
        lbMemory.clearDischargeableContainers();
        lbMemory.reshuffle(scr);
        // The cold string tags survive discharge (their ids stay referenced
        // by RAM-side containers); the emptied statement storage becomes
        // arena holes the copying compaction reclaims. The byte-exact
        // survivor residual check lands with that compaction.
    }

    /// @brief Bring a deloaded LB's statified containers back from the
    ///        file set recorded at deload; no-op when resident.
    ///
    /// @details
    /// The resident no-op is a defined contract (the touch points call
    /// unconditionally), not a fallback. On the reload path: asserts the
    /// chain still matches the LB's tree position, `markResident()`,
    /// then `lbdeload::loadLbMemory` rebuilds every container
    /// element-by-element onto a fresh consecutive virtual index (the
    /// approved "straightening").
    ///
    /// @param directory The deload directory the files live in.
    void Memory::ensureLoaded(const std::string& directory) {
        if (lbMemory.manager.resident()) return;
        assert(!dischargedForever
            && "ensureLoaded on a discharged LB - deactivation is "
               "permanent and the kernel-side readers probe RAM records "
               "(I-112); the post-prove export "
               "uses ensureLoadedForRead");
        reloadFromImage(directory);
    }

    /// @brief Sanctioned post-prove READ reload — like `ensureLoaded` but
    ///        legal on a discharged LB (D-158).
    ///
    /// @details
    /// The chapter export (`buildStack`) reads origin history in full; the
    /// strings campaign moved that history's strings into the deloadable
    /// cold image, so a discharged-and-drained LB must come back for its
    /// chapter walk. This does NOT reactivate anything: `isActive` and
    /// `dischargedForever` are untouched, no kernel runs after the export,
    /// and the reload is content-identical to the dumped image
    /// (I-112 keeps its meaning — an LB never
    /// returns to the ACTIVE set; readability is restored deliberately).
    /// Defined no-op when resident, like `ensureLoaded`.
    ///
    /// @param directory The deload directory the image lives in.
    void Memory::ensureLoadedForRead(const std::string& directory) {
        if (lbMemory.manager.resident()) return;
        reloadFromImage(directory);
    }

    // Export-phase reload sink (declared in memory.hpp). Null except during
    // the chapter export, which points it at a local vector so it can release
    // per-chapter the LBs the read-only walk reloads
    // (G-53).
    std::vector<Memory*>* g_exportReloadSink = nullptr;

    /// @brief Shared reload core of `ensureLoaded` / `ensureLoadedForRead`:
    ///        chain check, mark resident, element-by-element rebuild, dirty
    ///        reset. Caller established non-residency and its own policy
    ///        asserts.
    ///
    /// @param directory The deload directory the image lives in.
    void Memory::reloadFromImage(const std::string& directory) {
        assert(!deloadFiles.empty()
            && "reload on a deloaded LB with no recorded file set");
        assert(buildLbChainString(*this) == deloadChain
            && "LB moved in the tree between deload and reload");
        lbMemory.manager.markResident();
        // The four HashMemory instances reload through lbMemory.visitContainers
        // (D-147), so loadLbMemory rebuilds their tags 51..450
        // alongside LbMemory's own -- no extra-column list.
        lbdeload::loadLbMemory(lbMemory, deloadChain, deloadFiles, directory);
        // The rebuild's clear()+push_back escalate the dirty state, but
        // reload by definition re-establishes RAM == disk: reset so an
        // unchanged LB skips its next dump.
        lbMemory.dirty = DirtyState::Clean;
        // Export-phase reload tracking (G-53): the
        // read-only chapter export reloads LBs on demand; record each so the
        // driver releases it per chapter, else the reloads accumulate to
        // static-pool exhaustion. Null (untracked) during the kernel's own
        // pressure reloads in proveKernel.
        if (g_exportReloadSink != nullptr) g_exportReloadSink->push_back(this);
    }

    // Explicit instantiation of the request-generator member templates for the
    // streaming consumer (the only consumer; see BurstSink in prover.hpp).
    template void ExpressionAnalyzer::generateEncodedRequestsStatic<BurstSink>(
        const Memory&, const HashMemory&, const int16_t*, int16_t, IntStmtView,
        unsigned, BurstSink&);
    template void ExpressionAnalyzer::generateEncodedRequestsStaticPairs<BurstSink>(
        const Memory&, const HashMemory&, const MandatoryPair*, int16_t, IntStmtView,
        IntStmtView, unsigned, BurstSink&);

    // ---- NameMap::encodePush (span overload) --------------------------------
    // Out-of-line: the heap-free canonical build reaches the string scratch
    // registry (scratchArenas()) and the worker-slot id
    // (ExpressionAnalyzer::g_currentCoreId), neither visible to an inline header
    // body. Byte-identical to the former inline heap-std::string body — same
    // canonical bytes (decodeView == decode), same dedup, same subs-then-names
    // mint order.
    int16_t NameMap::encodePush(int16_t parentId, const StrSpan& payload) {
        // Delimiter self-overlap contract (I-139): the payload may not contain
        // "_boundary_", nor end "_boundary" / start "boundary_".
        const std::string_view pv(payload.ptr,
                                  static_cast<std::size_t>(payload.len));
        assert(pv.find(BOUNDARY_STR, 0, BOUNDARY_LEN) == std::string_view::npos
               && "payload must not contain '_boundary_'");
        assert(!(payload.len >= 9
                 && equalSpans(StrSpan(payload.ptr + payload.len - 9, 9),
                               StrSpan("_boundary", 9)))
               && "payload must not end with '_boundary'");
        assert(!(payload.len >= 9
                 && equalSpans(StrSpan(payload.ptr, 9),
                               StrSpan("boundary_", 9)))
               && "payload must not start with 'boundary_'");
        seedIfEmpty();

        // canonical = decodeView(parentId) + "_boundary_" + payload, assembled
        // on the per-slot string-scratch arena. decodeView aliases the names
        // cold pages; it is copied into the scratch buffer here, before any mint
        // below, so the later mintName cannot dangle it (I-3).
        const StrSpan parentSpan = decodeView(parentId);
        const int32_t total = parentSpan.len
                            + static_cast<int32_t>(BOUNDARY_LEN)
                            + payload.len;
        const unsigned slot = (ExpressionAnalyzer::g_currentCoreId >= 0)
            ? static_cast<unsigned>(ExpressionAnalyzer::g_currentCoreId)
            : scratchArenas().slotCount() - 1;
        ScratchArena& sa = scratchArenas().forSlot(slot);
        ScratchScope scope(sa);
        char* const buf = sa.allocBytes(total);
        std::memcpy(buf, parentSpan.ptr,
                    static_cast<std::size_t>(parentSpan.len));
        std::memcpy(buf + parentSpan.len, BOUNDARY_STR, BOUNDARY_LEN);
        std::memcpy(buf + parentSpan.len + BOUNDARY_LEN, payload.ptr,
                    static_cast<std::size_t>(payload.len));
        const StrSpan canonical(buf, total);

        const int16_t existing = lookup(canonical);
        if (existing != 0) return existing;

        // Mint order (subs payload, then names canonical) preserved.
        const int16_t subId = encodeSub(payload);
        const int16_t newId = mintName(canonical);
        nodes->push_back(ValidityNode{parentId, subId});
        return newId;
    }

} // namespace gl
