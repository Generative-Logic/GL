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
#include "memory_infra/deload_stats.hpp"
#include <cstdio>
#include <iostream>
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

    // Out-of-line IntEncodedExpr recordUSignature (declared in memory.hpp).
    // Heap-free: builds the (slot, argFullId) pairs on the per-slot gen-scratch
    // byte-bump tier, then range-constructs the signature vector INSIDE the set
    // node via emplace(first, last) — no function-local std::vector. Out-of-line
    // because it reaches genScratchArenas() + ExpressionAnalyzer::g_currentCoreId,
    // which an inline header body cannot see (the encodePush(StrSpan) precedent).
    void recordUSignature(OwnerSet& os,
                          const IntEncodedExpr* encList, NameId count) {
        using Pair = std::pair<int32_t, NameId>;
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
        int32_t slot = 0;
        bool hasUArg = false;
        for (NameId e = 0; e < count; ++e) {
            const IntEncodedExpr& expr = encList[e];
            const int32_t arity = std::min<int32_t>(expr.arity,
                static_cast<int32_t>(ExecutionParameters::MAX_ARITY));
            for (int32_t j = 0; j < arity; ++j) {
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

    /// @brief Whether a rule install has the or-INTRO shape — a single
    ///        premise that is one flattened leaf of its or-category head.
    ///
    /// @details
    /// The install-time detector behind the `LocalMemoryValue::
    /// disintegrationAllowed` derivation
    /// (D-241). An or-intro rule —
    /// `(>[bound](D_k)(or…))`, emitted per flattened leaf by
    /// `disintegrateExprCore2`'s implication branch
    /// (D-237) — states a disjunction as a FACT from one
    /// true disjunct; its fired head is consumed flat, so it must never
    /// re-disintegrate into `_ordis_` branch scopes. Detection is by shape,
    /// not emission provenance, so a status-3 mail-recovered intro rule
    /// re-marks at its receiver-side reinstall exactly like a local install.
    ///
    /// The leaf flattening is the compiled-map recursion the emission site
    /// uses: substitute each level's signature args by the instance args,
    /// recurse into or-category elements only (I-166), compare each non-or
    /// leaf byte-for-byte against the premise. A compiled-map miss on the
    /// head is a defined negative (atomic or uncompiled head — not an
    /// or-intro), mirroring the emission site's probe contract.
    ///
    /// @param keyRun   The ordered implication-chain premises.
    /// @param keyN     Number of premises; only `keyN == 1` can match.
    /// @param headSpan The head (implication consequent).
    /// @return Whether the install is or-intro-shaped.
    /// @invariant Heap-free: compiled-map read-out probes plus scratch-arena
    ///            string builds under a per-call `ScratchScope` (Rule 28).
    /// @see `disintegrateExprCore2` — the intro emission;
    ///      `addToHashMemory` — the sole production caller;
    ///      `LocalMemoryValue::disintegrationAllowed` — the derived flag.
    bool ExpressionAnalyzer::isOrIntroInstall(const StrSpan* keyRun,
                                              int32_t keyN,
                                              StrSpan headSpan) {
        if (keyN != 1) return false;
        if (headSpan.len < 1 || headSpan.ptr[0] != '(') return false;
        const LogicalEntity* orLe =
            compiledEntity(extractExpressionUniversalSpan(headSpan));
        if (orLe == nullptr || orLe->category != "or") return false;

        const unsigned slot = (g_currentCoreId >= 0)
            ? static_cast<unsigned>(g_currentCoreId)
            : scratchArenas().slotCount() - 1;
        ScratchArena& sArena = scratchArenas().forSlot(slot);
        ScratchScope scope(sArena);

        const StrSpan premise = keyRun[0];
        bool matched = false;
        const auto walk = [&](const LogicalEntity& node, StrSpan instance,
                              const auto& self) -> void {
            StrSpan sigArgs[ExecutionParameters::MAX_ARITY];
            const int32_t sigN = getArgsSpans(StrSpan(node.signature), sigArgs,
                                              ExecutionParameters::MAX_ARITY);
            StrSpan instArgs[ExecutionParameters::MAX_ARITY];
            const int32_t instN = getArgsSpans(instance, instArgs,
                                               ExecutionParameters::MAX_ARITY);
            assert(sigN == instN
                && "isOrIntroInstall: instance arity differs from compiled or");
            StrReplacement pairs[ExecutionParameters::MAX_ARITY];
            int32_t pairN = 0;
            for (int32_t a = 0; a < sigN; ++a) {
                pairs[pairN].key = sigArgs[a];
                pairs[pairN].value = instArgs[a];
                ++pairN;
            }
            for (const std::string& rawElem : node.elements) {
                if (matched) return;
                const ScratchString subst =
                    replaceKeysScratch(sArena, StrSpan(rawElem), pairs, pairN);
                const StrSpan substSpan(subst);
                const LogicalEntity* childLe =
                    (substSpan.len >= 1 && substSpan.ptr[0] == '(')
                    ? compiledEntity(extractExpressionUniversalSpan(substSpan))
                    : nullptr;
                if (childLe != nullptr && childLe->category == "or") {
                    self(*childLe, substSpan, self);
                } else if (equalSpans(substSpan, premise)) {
                    matched = true;
                }
            }
        };
        walk(*orLe, headSpan, walk);
        return matched;
    }

    // Doxygen at the declaration (prover.hpp).
    int32_t ExpressionAnalyzer::flattenRegistryOrLeavesScratch(
        const LogicalEntity& orLe, StrSpan instance,
        StrSpan* out, int32_t cap, ScratchArena& sArena) {

        assert(orLe.category == "or"
            && "flattenRegistryOrLeavesScratch: entity must be or-category");
        int32_t outN = 0;
        const auto walk = [&](const LogicalEntity& node, StrSpan inst,
                              const auto& self) -> void {
            StrSpan sigArgs[ExecutionParameters::MAX_ARITY];
            const int32_t sigN = getArgsSpans(StrSpan(node.signature), sigArgs,
                                              ExecutionParameters::MAX_ARITY);
            StrSpan instArgs[ExecutionParameters::MAX_ARITY];
            const int32_t instN = getArgsSpans(inst, instArgs,
                                               ExecutionParameters::MAX_ARITY);
            assert(sigN == instN
                && "flattenRegistryOrLeavesScratch: instance arity differs from compiled or");
            StrReplacement pairs[ExecutionParameters::MAX_ARITY];
            int32_t pairN = 0;
            for (int32_t a = 0; a < sigN; ++a) {
                pairs[pairN].key = sigArgs[a];
                pairs[pairN].value = instArgs[a];
                ++pairN;
            }
            for (const std::string& rawElem : node.elements) {
                const ScratchString subst =
                    replaceKeysScratch(sArena, StrSpan(rawElem), pairs, pairN);
                const StrSpan substSpan(subst);
                const LogicalEntity* childLe =
                    (substSpan.len >= 1 && substSpan.ptr[0] == '(')
                    ? compiledEntity(extractExpressionUniversalSpan(substSpan))
                    : nullptr;
                if (childLe != nullptr && childLe->category == "or") {
                    self(*childLe, substSpan, self);
                } else {
                    assert(outN < cap
                        && "flattenRegistryOrLeavesScratch: leaf count exceeds cap");
                    out[outN++] = substSpan;
                }
            }
        };
        walk(orLe, instance, walk);
        assert(outN >= 2
            && "flattenRegistryOrLeavesScratch: or entity flattens to fewer than two leaves");
        return outN;
    }

    // Doxygen at the declaration (prover.hpp).
    bool ExpressionAnalyzer::isSubsetExclusionInstall(const StrSpan* keyRun,
                                                      int32_t keyN,
                                                      StrSpan headSpan) {
        if (keyN < 1
            || keyN > ExecutionParameters::kMaxReducedOrLeaves - 2) {
            return false;
        }
        if (headSpan.len < 1 || headSpan.ptr[0] != '(') return false;
        const LogicalEntity* redLe =
            compiledEntity(extractExpressionUniversalSpan(headSpan));
        if (redLe == nullptr || redLe->category != "or") return false;

        const unsigned slot = (g_currentCoreId >= 0)
            ? static_cast<unsigned>(g_currentCoreId)
            : scratchArenas().slotCount() - 1;
        ScratchArena& sArena = scratchArenas().forSlot(slot);
        ScratchScope scope(sArena);

        // The excluded-disjunct candidates: negate each premise with
        // double-negation cancellation — the emission builds premises as
        // negateScratch(leaf), so negating recovers the TRUE-polarity leaf
        // whatever its sign (I-175). Premise ORDER is deliberately ignored:
        // the mail-compact round-trip stores only the first-seen body and
        // dedups whole name-sorted permutation families (I-52), so a
        // reinstalled rule's premises may arrive in any order.
        StrSpan dSpans[ExecutionParameters::kMaxReducedOrLeaves];
        for (int32_t p = 0; p < keyN; ++p)
            dSpans[p] = negateScratch(sArena, keyRun[p]);

        // The head's leaves in INSTANCE terms. Bytes sit below the
        // per-entry scopes and survive the whole registry scan.
        StrSpan leaves[ExecutionParameters::kMaxReducedOrLeaves];
        const int32_t leafN = flattenRegistryOrLeavesScratch(
            *redLe, headSpan, leaves,
            ExecutionParameters::kMaxReducedOrLeaves, sArena);
        const int32_t parentN = leafN + keyN;
        if (parentN > ExecutionParameters::kMaxReducedOrLeaves) return false;

        // Token->arg binding for one registry entry: registry-term leaves
        // carry the entity's canonical u_<digits> signature tokens; the
        // binding is INJECTIVE (two distinct tokens never bind one arg),
        // which reproduces the first-appearance-canonicalization verdict —
        // in particular a DEGENERATE repeated-arg instance stays untagged
        // (D-270 keeps its head safe to
        // flat-consume).
        StrSpan boundArg[ExecutionParameters::MAX_ARITY + 1];
        bool boundSet[ExecutionParameters::MAX_ARITY + 1] = {};
        int32_t dfsSteps = 0;

        // Match one registry-term element against one instance-term
        // candidate under the growing binding. Newly bound token indices
        // are appended to undo[]; the CALLER unbinds them when its branch
        // fails (a failed match may leave partial binds behind — the undo
        // log covers them).
        const auto matchElem = [&](StrSpan regElem, StrSpan cand,
                                   int32_t* undo, int32_t& undoN) -> bool {
            const bool regNeg = (regElem.len > 0 && regElem.ptr[0] == '!');
            const bool candNeg = (cand.len > 0 && cand.ptr[0] == '!');
            if (regNeg != candNeg) return false;
            const StrSpan r = regNeg
                ? StrSpan(regElem.ptr + 1, regElem.len - 1) : regElem;
            const StrSpan c = candNeg
                ? StrSpan(cand.ptr + 1, cand.len - 1) : cand;
            int32_t rb = -1, cb = -1;
            for (int32_t i = 0; i < r.len && rb < 0; ++i)
                if (r.ptr[i] == '[') rb = i;
            for (int32_t i = 0; i < c.len && cb < 0; ++i)
                if (c.ptr[i] == '[') cb = i;
            if ((rb < 0) != (cb < 0)) return false;
            if (rb < 0) return equalSpans(r, c);
            if (rb != cb
                || std::memcmp(r.ptr, c.ptr, static_cast<std::size_t>(rb)) != 0) {
                return false;
            }
            StrSpan rArgs[ExecutionParameters::MAX_ARITY];
            StrSpan cArgs[ExecutionParameters::MAX_ARITY];
            const int32_t rN = getArgsSpans(r, rArgs,
                                            ExecutionParameters::MAX_ARITY);
            const int32_t cN = getArgsSpans(c, cArgs,
                                            ExecutionParameters::MAX_ARITY);
            if (rN != cN) return false;
            for (int32_t a = 0; a < rN; ++a) {
                const StrSpan t = rArgs[a];
                int32_t idx = 0;
                bool isTok = (t.len >= 3 && t.ptr[0] == 'u' && t.ptr[1] == '_');
                for (int32_t d = 2; d < t.len && isTok; ++d) {
                    isTok = (t.ptr[d] >= '0' && t.ptr[d] <= '9');
                    if (isTok) idx = idx * 10 + (t.ptr[d] - '0');
                }
                if (!isTok || idx < 1
                    || idx > ExecutionParameters::MAX_ARITY) {
                    // A non-signature token (shared literal) matches by
                    // bytes only.
                    if (!equalSpans(t, cArgs[a])) return false;
                    continue;
                }
                if (boundSet[idx]) {
                    if (!equalSpans(boundArg[idx], cArgs[a])) return false;
                    continue;
                }
                bool argTaken = false;
                for (int32_t s = 1;
                     s <= ExecutionParameters::MAX_ARITY && !argTaken; ++s) {
                    argTaken = boundSet[s] && equalSpans(boundArg[s], cArgs[a]);
                }
                if (argTaken) return false;
                boundSet[idx] = true;
                boundArg[idx] = cArgs[a];
                undo[undoN++] = idx;
            }
            return true;
        };

        // Embed walk over one registry entry's flattened leaf list pl[]:
        // element ei matches either the next head leaf (order preserved) or
        // any unused premise negation (order free). At ei == plN the counts
        // force hi == leafN and a full usedMask, so reaching the end IS
        // acceptance.
        const auto embed = [&](int32_t ei, int32_t hi, uint32_t usedMask,
                               const StrSpan* pl, int32_t plN,
                               const auto& self) -> bool {
            ++dfsSteps;
            assert(dfsSteps <= ExecutionParameters::kSubsetExclusionEmbedCap
                && "isSubsetExclusionInstall: embed search exceeds the Rule-19 cap");
            if (ei == plN) return true;
            int32_t undo[ExecutionParameters::MAX_ARITY];
            if (hi < leafN) {
                int32_t undoN = 0;
                const bool m = matchElem(pl[ei], leaves[hi], undo, undoN);
                if (m && self(ei + 1, hi + 1, usedMask, pl, plN, self))
                    return true;
                for (int32_t u = 0; u < undoN; ++u) boundSet[undo[u]] = false;
            }
            for (int32_t p = 0; p < keyN; ++p) {
                if (usedMask & (1u << p)) continue;
                int32_t undoN = 0;
                const bool m = matchElem(pl[ei], dSpans[p], undo, undoN);
                if (m && self(ei + 1, hi, usedMask | (1u << p), pl, plN, self))
                    return true;
                for (int32_t u = 0; u < undoN; ++u) boundSet[undo[u]] = false;
            }
            return false;
        };

        // ONE order-free registry scan: a parent candidate is any or entry
        // whose FLATTENED leaf count equals leafN + keyN (flatten-based on
        // both sides, so a nested parent's flat form matches — the same
        // flattened-cohort discipline the emission uses, I-166). The
        // element-count prefilter is sound because flattening only grows
        // the count.
        bool found = false;
        forEachCompiledOr([&](const std::string& name,
                              const LogicalEntity& le) -> bool {
            (void)name;
            if (static_cast<int32_t>(le.elements.size()) > parentN)
                return false;
            ScratchScope entScope(sArena);
            StrSpan pl[ExecutionParameters::kMaxReducedOrLeaves];
            const int32_t plN = flattenRegistryOrLeavesScratch(
                le, StrSpan(le.signature), pl,
                ExecutionParameters::kMaxReducedOrLeaves, sArena);
            if (plN != parentN) return false;
            for (int32_t s = 0; s <= ExecutionParameters::MAX_ARITY; ++s)
                boundSet[s] = false;
            dfsSteps = 0;
            if (embed(0, 0, 0u, pl, plN, embed)) {
                found = true;
                return true;
            }
            return false;
        });
        return found;
    }

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
    ///    `IntNormalizedKey` from the NameId-encoded key bytes; consult
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
    ///                                NameId mirror is computed inside).
    /// @param value                   Head expression text (what the rule produces).
    /// @param remainingArgs           Argument names not consumed by the key
    ///                                that must travel with the head.
    /// @param mb                      Owning `Memory` (used for `exprOriginMap`,
    ///                                `mailOut`, `ruleInterner`).
    /// @param secondTarget            Optional second instance receiving every
    ///                                staged record of the same build (the
    ///                                incubator's `workingMemory` beside
    ///                                `overallHashMemory`); install-only,
    ///                                never the first target.
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
    ///       two `normalizedEncoded*` owner maps install onto it. Keys are owning
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
        StrSpan validityNameSpan,
        const RuleIndexOp& op,
        HashMemory* secondTarget) {

        // 0% heap: the caller feeds spans over its own stable bytes and this
        // install body reads them directly (key = keyRun, value = valueSpan,
        // remainingArgs = the sorted-unique remRun, original / justification /
        // clean / validity = their spans). Byte-identical: the ordered chain +
        // the sorted-unique remaining-arg run are preserved.

        // A rule installed from non-derived material (the {-1} statement
        // level tier, e.g. a mail-carried non-derived implication) has no
        // level constraints — that is the empty-set convention (I-51). Rule
        // level sets therefore stay empty-or-real; the {-1} tier never
        // enters hash memory.
        if (levelCount == 1 && levels[0] == -1) {
            levels = nullptr;
            levelCount = 0;
        }
        assert((levelCount == 0 || levels[0] >= 0)
            && "rule install level run mixes the non-derived tier with real levels");
        // A removal leaves owner-less entries behind for the end-of-apply
        // compaction (eraseOwnerlessEntries) — flag the instance.
        assert((!op.isRemove() || (op.visited != nullptr && op.dropSet != nullptr))
            && "addToHashMemory: a removal without its visited set or drop set");
        if (op.isRemove()) targetIntMemory.ownerlessPending = true;

        // Own multiplyImplication loop — fully independent from addToHashMemory.
        // The span form emits each copy onto mulStrArena (string tier) as a
        // CopyRef index on mulGenArena (page tier, I-124); each copy is read as a
        // zero-copy copySpan over mulStrArena (0% heap). mulStrScope holds the
        // copies live across the whole loop.
        // Per-registry slot derivation: both registries carry logicalCores
        // worker slots plus one DEDICATED reserved single-threaded slot (index
        // logicalCores), and each fallback still reads its OWN slotCount().
        // Workers (g_currentCoreId >= 0) use their own coreId on both —
        // disjoint arenas, in range for both registries.
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
        // The rule-index staging serving this instance on this slot (an
        // install only; a removal uses the direct doors): the install stages
        // every index write into it; the window's closer flushes it. The
        // install is a staging window of its own: a nested deposit into the
        // same LB (a copy equality, an integration seed) shares the bindings
        // and must not close them from under this install — the outermost
        // window closes all four instances.
        RuleIndexStaging* stagingPtr = op.isRemove()
            ? nullptr : &ruleStagings().acquire(mulGenSlot, &targetIntMemory);
        RuleIndexStaging& staging = stagingPtr != nullptr
            ? *stagingPtr : ruleStagings().at(mulGenSlot, 0);   // never read on the remove path
        // A second target (the incubator's workingMemory beside
        // overallHashMemory) receives every staged record of the same build:
        // the copies, permutations and keys are computed once and written to
        // both stagings (D-333).
        assert((secondTarget == nullptr || (!op.isRemove() && secondTarget != &targetIntMemory))
            && "addToHashMemory: a second target is an install-only twin of a different instance");
        RuleIndexStaging* const staging2Ptr = (secondTarget != nullptr)
            ? &ruleStagings().acquire(mulGenSlot, secondTarget) : nullptr;
        ++mb.ruleStagingArmDepth;
        {
            RT_SCOPE_HERE("HM_MULTIPLY");
            multiplyImplication(originalImplicationSpan, mulStrArena, copies);
            RT_NOTE_ITERATIONS_HERE(copies.size());   // copies produced
        }

        for (int32_t c = 0; c < copies.size(); ++c) {
            RT_SCOPE_HERE("HM_COPY");
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
                    RT_SCOPE_HERE("HM_COPY_HISTORY");
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
                    mb.addMailOutOrigin(copySpan, StrSpan("main", 4),
                        OriginTag::multipliedFrom, mulDeps, 1, mulCap);
                }

                // Row 238: span twin — curKey from each triple's key (get<0>),
                // curValue = head. Spans slice copySpan (mulStrArena, stable).
                RT_SCOPE_HERE("HM_COPY_DISINT");
                ce::disintegrateImplicationSpans(copySpan, curValueSpan,
                    [&curKeyRun, &curKeyN](StrSpan keySpan, const StrSpan*, int32_t) {
                        assert(curKeyN < ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS
                            && "addToHashMemory: curKey element count exceeds cap");
                        curKeyRun[curKeyN++] = keySpan;
                    });
            }

            // One encode per premise per copy: every whole-key permutation,
            // every subkey prefix and every admission derivative below reads
            // this memo. The lazy fill keeps the NameMap first-touch order
            // (a premise is first read where it used to be first encoded).
            KeyEncodeMemo encMemo;
            encMemo.bind(curKeyRun, curKeyN);

            // The rule's owner identity in this HashMemory instance: the copy
            // text's ruleInterner id + the install scope's NameMap id, recorded
            // on every index entry the install below mints (whole keys, subkeys,
            // remaining-args edges, the originals chain) so a later removal of
            // this rule deletes exactly its owner from each. Minted here, before
            // the chain ids, so the owner exists for the originals insert.
            int32_t copyImplId = 0;
            NameId ownerVid = 0;
            RuleOwner owner = 0;
            RuleOwner ruleOwner = 0;
            {
                RT_SCOPE_HERE("HM_COPY_OWNER_MINT");
                copyImplId = mb.ruleInterner.encode(copySpan);
                ownerVid = mb.nameMap.encode(validityNameSpan);
                owner = packRuleOwner(copyImplId, ownerVid);
                ruleOwner = packRuleOwner(
                    mb.ruleInterner.encode(originalImplicationSpan), ownerVid);
            }

            // The copy's owning rules (copyOwners, keyed by the packed copy
            // owner): two rules can multiply into one copy text, so the
            // install lists this rule on the copy and a removal delists it —
            // the copy's index entries and LMVs are touched only when no
            // other rule owns the copy any more. A removal then stages the
            // copy's LMVs for the batched compaction: the pair every LMV
            // record of this copy carries (originalImplicationId = the copy's
            // id). The drop set follows the multiplication exactly — several
            // copies, several pairs; no copy (the trivial-head skip), nothing
            // staged, nothing installed.
            char copyKey[sizeof(RuleOwner)];
            std::memcpy(copyKey, &owner, sizeof(RuleOwner));
            {
            RT_SCOPE_HERE("HM_COPY_OWNERS");
            if (!op.isRemove()) {
                stageOwnerToRun(staging.copyOwners,
                                StrSpan(copyKey, static_cast<int32_t>(sizeof(RuleOwner))),
                                ruleOwner);
                if (staging2Ptr != nullptr) {
                    stageOwnerToRun(staging2Ptr->copyOwners,
                                    StrSpan(copyKey, static_cast<int32_t>(sizeof(RuleOwner))),
                                    ruleOwner);
                }
            } else {
                if (removeOwnerFromRun(targetIntMemory.copyOwners,
                                       StrSpan(copyKey, static_cast<int32_t>(sizeof(RuleOwner))),
                                       ruleOwner, mulGenArena) > 0) {
                    continue;   // another rule still owns this copy
                }
                op.dropSet->mint(owner);
            }
            } // RT_SCOPE HM_COPY_OWNERS

            // --- Shared members (path-independent, needed for integration) ---
            // implication = curKey + curValue as a StrSpan run.
            StrSpan implRun[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS + 1];
            int32_t implRunN = 0;
            for (int32_t i = 0; i < curKeyN; ++i) implRun[implRunN++] = curKeyRun[i];
            implRun[implRunN++] = curValueSpan;
            // Positional value-id mint (byte-identical to encodeValueVector, I-84);
            // the originals chain gains this rule as an owner via the raw
            // encodeIdVecKeyInto door (no owning IdVecKey).
            {
                RT_SCOPE_HERE("HM_ORIGINALS_CHAIN");
                int32_t implIds[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS + 1];
                for (int32_t i = 0; i < implRunN; ++i)
                    implIds[i] = mb.ruleInterner.encode(implRun[i]);
                char ivKeyBuf[4 * (ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS + 2)];
                const int32_t ivLen = encodeIdVecKeyInto(implIds, implRunN,
                    ivKeyBuf, static_cast<int32_t>(sizeof(ivKeyBuf)));
                const StrSpan ivKey(ivKeyBuf, ivLen);
                if (!op.isRemove()) {
                    stageOwnerToRun(staging.originals, ivKey, owner);
                    if (staging2Ptr != nullptr) stageOwnerToRun(staging2Ptr->originals, ivKey, owner);
                } else if (op.firstVisit(RuleIndexOp::Chain, ivKey, owner, mulGenArena)) {
                    removeOwnerFromRun(targetIntMemory.originals, ivKey, owner,
                                       mulGenArena);
                }
            }
            for (int32_t tgtIx = 0; tgtIx < 2; ++tgtIx) {
                // Each target's own triggers, the second target after the
                // first (a second target is an install-only twin).
                HashMemory* const tgtPtr = (tgtIx == 0) ? &targetIntMemory : secondTarget;
                if (tgtPtr == nullptr) continue;
                HashMemory& trigTarget = *tgtPtr;
                // Decoded-lex INDEX over the triggers, sorted by decodeTemplateKeyView
                // (template then validity, compareSpans) — reproduces the former
                // std::sort(triggerRows) order (I-84). Per row copyFrom template +
                // validity to the string tier (MANDATORY: makeAdmissionKeys mints
                // templateInterner, the SAME interner the trigger decodes from, I-3).
                RT_SCOPE_HERE("HM_TRIGGERS");
                auto& trig = trigTarget.triggersForAdmissionSetIntegration;
                const int32_t trigN = trig.count();
                RT_NOTE_ITERATIONS_HERE(trigN);
                if (trigN > 0 && !op.isRemove()) {   // admission keys are install-only
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
                            trigTarget, StrSpan(tValC), mb);
                    }
                }
            }

            {
            RT_SCOPE_HERE("HM_ADMISSION_NORMKEYS");
            this->makeNormalizedKeysForAdmission(curKeyRun, curKeyN, targetIntMemory,
                mb.nameMap, mb.ruleInterner, curValueSpan, minNumOperatorsKey,
                copySpan, validityNameSpan, &mb, op, &encMemo, stagingPtr);
            if (secondTarget != nullptr) {
                this->makeNormalizedKeysForAdmission(curKeyRun, curKeyN, *secondTarget,
                    mb.nameMap, mb.ruleInterner, curValueSpan, minNumOperatorsKey,
                    copySpan, validityNameSpan, &mb, op, &encMemo, staging2Ptr);
            }
            }

            // Remaining args: c==0 = the caller's sorted-unique remRun; c!=0 =
            // getRemainingArgs over curKeyRun (u_ vars may have been equalized),
            // also sorted-unique. 0% heap.
            const StrSpan* curRemRun;
            int32_t curRemN;
            StrSpan remScratch[ExecutionParameters::MAX_ADMISSION_REM_ARGS];
            if (c == 0) {
                curRemRun = remRun; curRemN = remN;
            } else {
                RT_SCOPE_HERE("HM_REMARGS_MINT");
                curRemN = getRemainingArgs(curKeyRun, curKeyN, remScratch,
                    ExecutionParameters::MAX_ADMISSION_REM_ARGS);
                curRemRun = remScratch;
            }

            if (performAdmissionMapUpdate && !op.isRemove())
            {
                RT_SCOPE_HERE("HM_UPDATE_ADMISSION_MAP");
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

            // intRemainingArgs: mint NameMap in curRemRun (sorted-lex) order ==
            // the former set-lex mint order (I-84), then a sorted-ASCENDING copy
            // for the Int16SetKey == the former std::set<NameId> iteration.
            NameId intRemArgs[ExecutionParameters::MAX_ADMISSION_REM_ARGS];
            NameId intRemArgsSorted[ExecutionParameters::MAX_ADMISSION_REM_ARGS];
            {
            RT_SCOPE_HERE("HM_REMARGS_MINT");
            for (int32_t i = 0; i < curRemN; ++i) intRemArgs[i] = nm.encode(curRemRun[i]);
            if (curRemN > 0)
                std::memcpy(intRemArgsSorted, intRemArgs,
                    static_cast<std::size_t>(curRemN) * sizeof(NameId));
            std::sort(intRemArgsSorted, intRemArgsSorted + curRemN);
            }

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

            NameId bufIgnored[ExecutionParameters::MAX_KEY_SLOTS];
            NameId bufNotIgnored[ExecutionParameters::MAX_KEY_SLOTS];

            const unsigned genSlot = (g_currentCoreId >= 0)
                ? static_cast<unsigned>(g_currentCoreId)
                : genScratchArenas().slotCount() - 1;
            ScratchArena& genArena = genScratchArenas().forSlot(genSlot);
            const unsigned strSlot = (g_currentCoreId >= 0)
                ? static_cast<unsigned>(g_currentCoreId)
                : scratchArenas().slotCount() - 1;
            ScratchArena& strArena = scratchArenas().forSlot(strSlot);

            // Batch the remaining-args insert across the permutation loop: the
            // key (intRemArgsSorted/curRemN) is loop-invariant, so accumulate each
            // permutation's NormKey on the gen-scratch byte-bump tier and do ONE
            // RMW after the loop instead of one-per-permutation
            // (D-199). The blobs survive the loop's self-framing helpers
            // (appendLmvIdsRecord popTo's only its own frame).
            const ArenaOffset remArgsBatchMark = genArena.cursor();
            DirtyState remArgsBatchDirty = DirtyState::Clean;
            PagedVector<RemArgsBatchBlob> remArgsBatch(&genArena, &remArgsBatchDirty);

            // Both shape detectors read only curKeyRun / curValueSpan, which
            // do not vary across the installed permutations — one detection
            // per multiplied copy, not one per permutation (at keyN >= 2 the
            // subset-exclusion registry scan would otherwise repeat up to
            // n! times).
            bool copyOrIntro = false;
            bool copySubsetExclusion = false;
            {
                RT_SCOPE_HERE("HM_SHAPE_DETECT");
                copyOrIntro = isOrIntroInstall(curKeyRun, curKeyN, curValueSpan);
                copySubsetExclusion =
                    isSubsetExclusionInstall(curKeyRun, curKeyN, curValueSpan);
            }

            {
            RT_SCOPE_HERE("HM_WHOLEKEY_RECORDS");
            RT_NOTE_ITERATIONS_HERE(static_cast<int64_t>(permuts.size()));   // permutations enumerated
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
                const NameId intEncCount = static_cast<NameId>(permutation.size());
                assert(intEncCount <= ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS
                    && "addToHashMemory: key element count exceeds cap");
                {
                RT_SCOPE_HERE("HM_WK_ENCODE");
                for (std::size_t k = 0; k < permutation.size(); ++k) {
                    intEncoded[k] = encMemo.at(permutation[k], nm);
                }
                }

                NameId reverseMap[ExecutionParameters::MAX_KEY_SLOTS];
                std::memset(reverseMap, 0, sizeof(reverseMap));
                NameId numNormVars = 0;
                NameId lenIgnored = 0;
                NameId lenNotIgnored = 0;
                {
                RT_SCOPE_HERE("HM_WK_NORMKEY");
                lenIgnored = makeIntNormalizedKeyFromEncodedWithMap(
                    intEncoded, intEncCount, true, bufIgnored,
                    ExecutionParameters::MAX_KEY_SLOTS, reverseMap, numNormVars);

                lenNotIgnored = makeIntNormalizedKeyFromEncoded(
                    intEncoded, intEncCount, false, bufNotIgnored,
                    ExecutionParameters::MAX_KEY_SLOTS);
                }

                // Remove policy: this permutation's whole key and remaining-args
                // edge lose the owner (first visit only); no LMV, no batch.
                if (op.isRemove()) {
                    removeRuleKeyEntries(targetIntMemory, op,
                        static_cast<NameId>(n), bufNotIgnored, lenNotIgnored,
                        intRemArgsSorted, curRemN, owner, genArena);
                    continue;
                }

                // Rename run { normalized-var name -> decimal id } from reverseMap.
                // decodeView keys: no NameMap mint falls between here and the last
                // replaceKeysScratch use (the mints below are ruleInterner), so
                // decodeView is I-3-safe. Byte-identical to the former lex-ordered
                // std::map (replaceKeysScratch is greedy-longest, order-independent).
                int32_t valueId = 0;
                bool isMarker = false;
                {
                RT_SCOPE_HERE("HM_WK_VALUE_VARIANT");
                StrReplacement mp2Pairs[ExecutionParameters::MAX_KEY_SLOTS];
                int32_t mp2PairsN = 0;
                for (NameId id = 1; id <= numNormVars; ++id) {
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
                // then each remaining arg (the copy text itself was minted with
                // the owner above). keyIds empty (the head install carries none).
                valueId = mb.ruleInterner.encode(StrSpan(valueVariantS));
                isMarker =
                    containsSpan(StrSpan(valueVariantS), StrSpan("marker", 6));
                } // RT_SCOPE HM_WK_VALUE_VARIANT
                const int32_t originalImplId = copyImplId;
                int32_t remIds[ExecutionParameters::MAX_ADMISSION_REM_ARGS];
                {
                RT_SCOPE_HERE("HM_WK_LMV_FIELDS");
                for (int32_t i = 0; i < curRemN; ++i)
                    remIds[i] = mb.ruleInterner.encode(curRemRun[i]);
                }
                const RuleJustification just = ruleJustificationFromString(justificationSpan);

                // D-32: product-of-disintegration — true iff at least one premise
                // (chain element) has an arg starting with "u_" (bound-variable
                // placeholder). Consumed by checkLocalEncodedMemoryStatic to gate
                // OR-disintegration on the head-firing path.
                bool productOf = false;
                {
                RT_SCOPE_HERE("HM_WK_LMV_FIELDS");
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
                }

                // D-241: may heads fired
                // by this rule be disintegrated. False for integration
                // instructions and for or-intro-shaped rules (their heads are
                // integration instructions consumed flat); the firing site
                // reads this as the rule-intrinsic half of
                // FiringRecord::doNotDisintegrate.
                const bool disintegrationAllowed =
                    just != RuleJustification::integration
                    && !copyOrIntro;

                // I-184: shape-detected
                // subset-exclusion rules fire PARK-FIRST — the firing site
                // strips the route-(b) open signal from their heads, so the
                // fired reduced or parks in rejectedMapOrdis until the
                // compound-demand map opens it.
                const bool subsetExclusion = copySubsetExclusion;

                // encodedMap HEAD record + owner record + remaining-args index,
                // all via the raw / id-run doors (no owning NormKey / LMV). The
                // levels run is caller-owned ascending-unique (I-136) == the former
                // std::set<int> serialize order.
                {
                RT_SCOPE_HERE("HM_LMV_APPEND");
                stageLmvIdsRecord(staging.lmv,
                    static_cast<NameId>(n), bufIgnored, lenIgnored,
                    valueId, isMarker, nullptr, 0, remIds, curRemN,
                    originalImplId, ownerVid, genArena,
                    levels, levelCount, just, productOf, disintegrationAllowed,
                    /*ordisOnly=*/false, subsetExclusion);
                if (staging2Ptr != nullptr) {
                    stageLmvIdsRecord(staging2Ptr->lmv,
                        static_cast<NameId>(n), bufIgnored, lenIgnored,
                        valueId, isMarker, nullptr, 0, remIds, curRemN,
                        originalImplId, ownerVid, genArena,
                        levels, levelCount, just, productOf, disintegrationAllowed,
                        /*ordisOnly=*/false, subsetExclusion);
                }
                }
                {
                RT_SCOPE_HERE("HM_WHOLEKEY_OWNER");
                stageWholeKeyOwner(staging.wholeKeys,
                    static_cast<NameId>(n), bufNotIgnored, lenNotIgnored, owner);
                if (staging2Ptr != nullptr) {
                    stageWholeKeyOwner(staging2Ptr->wholeKeys,
                        static_cast<NameId>(n), bufNotIgnored, lenNotIgnored, owner);
                }
                }
                // Accumulate this permutation's NormKey (Codec<NormKey> bytes:
                // int16 numberExpressions ++ int16 length ++ data) for the one
                // batched insert after the loop.
                {
                    const int32_t nkBlobLen = (lenNotIgnored + 2)
                        * static_cast<int32_t>(sizeof(NameId));
                    const ArenaOffset nkBlobOff = genArena.alloc(
                        nkBlobLen, static_cast<int32_t>(alignof(NameId)));
                    NameId* pnk =
                        reinterpret_cast<NameId*>(genArena.resolve(nkBlobOff));
                    pnk[0] = static_cast<NameId>(n);
                    pnk[1] = lenNotIgnored;
                    std::memcpy(pnk + 2, bufNotIgnored,
                        static_cast<std::size_t>(lenNotIgnored) * sizeof(NameId));
                    remArgsBatch.push_back(RemArgsBatchBlob{ nkBlobOff, nkBlobLen });
                }
            }
            } // RT_SCOPE HM_WHOLEKEY_RECORDS
            {
            RT_SCOPE_HERE("HM_REMARGS_BATCH");
            // ONE RMW for the whole permutation batch into the invariant key.
            stageRemainingArgsNormKeyBatch(
                staging.remArgs,
                staging.remArgsOwners,
                intRemArgsSorted, curRemN, remArgsBatch, owner, genArena);
            if (staging2Ptr != nullptr) {
                stageRemainingArgsNormKeyBatch(
                    staging2Ptr->remArgs,
                    staging2Ptr->remArgsOwners,
                    intRemArgsSorted, curRemN, remArgsBatch, owner, genArena);
            }
            genArena.popTo(remArgsBatchMark);
            } // RT_SCOPE HM_REMARGS_BATCH

            // --- makeNormalizedSubkeys equivalent ---
            {
            RT_SCOPE_HERE("HM_SUBKEYS");
            RT_NOTE_ITERATIONS_HERE(static_cast<int64_t>(permuts.size()));   // permutations enumerated
            for (std::size_t p = 0; p < permuts.size(); ++p) {
                const std::vector<int>& permut = permuts[p];

                // The permutation's premises come from the per-copy memo and
                // its normalized key is built ONCE, one premise per prefix
                // step (`appendExprToIntNormalizedKey`): after step L the
                // first `st.pos` slots ARE the subkey of prefix length L (the
                // resumable builder's contract), so no prefix is re-encoded
                // or re-normalized. subEncoded[0..index] is the prefix in key
                // order — what the signature record reads.
                IntEncodedExpr subEncoded[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
                NameId subBuf[ExecutionParameters::MAX_KEY_SLOTS];
                NormKeyBuildState st;
                st.nVars = 0;
                st.nextNormId = 1;
                st.pos = 0;

                for (int32_t index = 0; index < n; ++index) {
                    bool toBreak = false;
                    for (int32_t index2 = 0; index2 < index; ++index2) {
                        if (compareSpans(idsRun[permut[index2]], idsRun[permut[index2 + 1]]) > 0) {
                            toBreak = true;
                            break;
                        }
                    }
                    if (toBreak) break;

                    const NameId subEncCount = static_cast<NameId>(index + 1);
                    assert(subEncCount <= ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS
                        && "addToHashMemory subkey: element count exceeds cap");
                    {
                    RT_SCOPE_HERE("HM_SK_ENCODE");
                    subEncoded[index] = encMemo.at(permut[index], nm);
                    }

                    NameId subLen = 0;
                    {
                    RT_SCOPE_HERE("HM_SK_NORMKEY");
                    appendExprToIntNormalizedKey(subEncoded[index], st, subBuf,
                        ExecutionParameters::MAX_KEY_SLOTS);
                    subLen = st.pos;
                    }

                    // A subkey below kSubkeyUCheckMinElements records its owner
                    // on the signature-free record; from there on its
                    // u_-signature record gains the owner with the signature's
                    // index — via the raw-key door (no owning NormKey). Under
                    // the remove policy the owner's pairs leave the record.
                    if (op.isRemove()) {
                        removeRuleSubkeyEntry(targetIntMemory, op,
                            static_cast<NameId>(index + 1), subBuf, subLen,
                            owner, genArena);
                    } else if (subEncCount < kSubkeyUCheckMinElements) {
                        RT_SCOPE_HERE("HM_SK_WRITE_SHORT");
                        stageShortSubkeyOwner(staging.subkeys,
                            static_cast<NameId>(index + 1), subBuf, subLen, owner);
                        if (staging2Ptr != nullptr) {
                            stageShortSubkeyOwner(staging2Ptr->subkeys,
                                static_cast<NameId>(index + 1), subBuf, subLen, owner);
                        }
                    } else {
                        RT_SCOPE_HERE("HM_SK_WRITE_MERGE");
                        stageSubkeySignatures(staging.subkeys,
                            static_cast<NameId>(index + 1), subBuf, subLen,
                            subEncoded, subEncCount, owner);
                        if (staging2Ptr != nullptr) {
                            stageSubkeySignatures(staging2Ptr->subkeys,
                                static_cast<NameId>(index + 1), subBuf, subLen,
                                subEncoded, subEncCount, owner);
                        }
                    }
                }
            }

            } // RT_SCOPE HM_SUBKEYS
            if (!op.isRemove()) {
                targetIntMemory.maxKeyLength =
                    std::max(static_cast<NameId>(n), targetIntMemory.maxKeyLength);
                if (secondTarget != nullptr) {
                    secondTarget->maxKeyLength =
                        std::max(static_cast<NameId>(n), secondTarget->maxKeyLength);
                }
            }
        }

        // The outermost window of this LB closes here (LB seeding, the CE
        // facts, the compressor, a standalone door call): every instance the
        // window's nested rounds staged into is flushed and released. Inside an
        // outer window (the absorbs, the apply, an enclosing install) that
        // window's closer flushes every install of the window at once.
        --mb.ruleStagingArmDepth;
        if (mb.ruleStagingArmDepth == 0) {
            closeRuleIndexStaging(mb);
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
    /// For each non-anchor element index of the input `key`, the function:
    /// 1. Extracts the core expression and its declared output-index list via
    ///    `ce::getArgs` + the per-anchor `coreExpressionMap`.
    /// 2. Asserts there is at most one output argument
    ///    ([I-19](../../docs/agentic_swdd/30_invariants.md#i-19) — assert is first-class;
    ///    a firing assert here means the core map declared a multi-output
    ///    operator, which the rest of the engine does not handle).
    /// 3. Builds a binary mask + subkey by selecting other non-anchor elements
    ///    that do NOT contain the output arg and always retaining an anchor
    ///    premise. The anchor is normalized together with the derivative key,
    ///    preserving its equality classes without changing normalization.
    ///    The result then qualifies via two gates:
    ///    - the *classic* gate, baseline-only behavior preserved by the
    ///      static-pipeline migration; fires when the subkey size is exactly
    ///      `n - 1` and `baselineClassicQualifies` accepts.
    ///    - the *local-u* gate, used for `(C)` qualification (see
    ///      `implicationIsQualified` / `baselineClassicQualifies` in
    ///      prover.hpp); admits a wider class of subkeys but only when the
    ///      key has a `u_`-prefixed arg meeting the local-u criterion.
    /// 4. Emits the marker LMV via the same gen scratch arena byte-bump storage
    ///    contract as `addToHashMemory` — the NameId key bytes live in the
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
    /// @param nameMap             `NameMap` for NameId encoding.
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
    ///       is extended with
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

    bool ExpressionAnalyzer::ordisRouteQualifies(const StrSpan* key,
        int32_t keyN, StrSpan value) const
    {
        const auto isOperatorApplication = [this](StrSpan elem) -> bool {
            if (elem.len > 0 && elem.ptr[0] == '!') return false;
            const StrSpan core = extractExpressionSpan(elem);
            return this->operators.find(std::string_view(core.ptr,
                static_cast<std::size_t>(core.len))) != this->operators.end();
        };
        int32_t opCount = 0;
        for (int32_t i = 0; i < keyN; ++i) {
            const StrSpan elem = key[i];
            if (elem.len >= 7
                && equalSpans(StrSpan(elem.ptr, 7), StrSpan("(Anchor", 7))) {
                continue;   // anchors: excluded from requirement and count
            }
            if (!isOperatorApplication(elem)) return false;
            ++opCount;
        }
        if (!isOperatorApplication(value)) return false;
        ++opCount;
        return opCount >= ExecutionParameters::kOrdisMinOperatorExpressions;
    }

    // Doxygen at the declaration (prover.hpp).
    bool ExpressionAnalyzer::ordis2KeyEligible(StrSpan elem) const
    {
        // An anchor can never be selected as the missing compound premise.
        if (elem.len >= 7
            && equalSpans(StrSpan(elem.ptr, 7), StrSpan("(Anchor", 7))) {
            return false;
        }

        // COMPOUND: a compact with a compiled non-atomic entity;
        // polarity-transparent (a negated form qualifies on its core,
        // I-175 — the demand and the filing both carry the '!' verbatim,
        // only the eligibility test looks through it). The compiledEntity
        // fence alone IS the operator test: compiledExpressions holds only
        // compiled operator DEFINITIONS, so any hit is an operator by
        // construction — and crucially it admits PREDICATES (preorder /
        // strictOrder, output_args empty), which `operators` (the
        // input-AND-output core-config set) excludes. Gating on
        // `operators` here was the 3X install-route defect that silently
        // disqualified the whole order family: no demand ever installed,
        // no disjunct ever filed (root-caused 2026-08-10, trap4 run).
        const StrSpan bare = (elem.len > 0 && elem.ptr[0] == '!')
            ? StrSpan(elem.ptr + 1, elem.len - 1) : elem;
        if (bare.len < 1 || bare.ptr[0] != '(') return false;
        const StrSpan core = extractExpressionSpan(bare);
        const LogicalEntity* le = compiledEntity(core);
        if (le == nullptr || le->category == "atomic") return false;

        // ARITY.
        StrSpan args[ExecutionParameters::MAX_ARITY];
        const int32_t argsN = getArgsSpans(elem, args,
                                           ExecutionParameters::MAX_ARITY);
        return argsN >= ExecutionParameters::kOrdis2DemandMinArity;
    }

    // Doxygen at the declaration (prover.hpp).
    bool ExpressionAnalyzer::ordis2DemandSlotQualifies(const StrSpan* key,
        int32_t keyN, int32_t index) const
    {
        const StrSpan elem = key[index];

        // Key-language filters (compound, arity, anchor refuse) — the
        // pair's ONE shared predicate, so a minted demand key is always a
        // text the park filing could have filed.
        if (!ordis2KeyEligible(elem)) return false;

        // SUBSET over the other NON-ANCHOR premises (the sharpened form,
        // maintainer-kept 2026-08-10): every slot argument must be bound by
        // a substantive premise. Anchor-only-bound short rules (the
        // A14 shape) mint no demand — the strict route through the Part-1
        // reduced or closes without them, and the parked totality cohorts
        // stay parked (D-267, filter decision).
        const auto isAnchor = [](StrSpan e) -> bool {
            return e.len >= 7
                && equalSpans(StrSpan(e.ptr, 7), StrSpan("(Anchor", 7));
        };
        StrSpan args[ExecutionParameters::MAX_ARITY];
        const int32_t argsN = getArgsSpans(elem, args,
                                           ExecutionParameters::MAX_ARITY);
        bool anySubstantive = false;
        for (int32_t a = 0; a < argsN; ++a) {
            bool bound = false;
            for (int32_t i = 0; i < keyN && !bound; ++i) {
                if (i == index) continue;
                if (isAnchor(key[i])) continue;
                StrSpan iArgs[ExecutionParameters::MAX_ARITY];
                const int32_t iN = getArgsSpans(key[i], iArgs,
                    ExecutionParameters::MAX_ARITY);
                for (int32_t x = 0; x < iN; ++x) {
                    if (equalSpans(iArgs[x], args[a])) {
                        bound = true;
                        break;
                    }
                }
            }
            if (bound) {
                anySubstantive = true;
                continue;
            }

            // An anchor-slot argument is GROUND by construction (D-287): it
            // names a fixed value of the anchor context -- the numeral 1 in
            // `1 <= d`, the carrier set, an operator -- so no substantive
            // premise can ever bind it. Requiring one rejects every consumer
            // whose slot mentions a constant.
            bool anchorSlot = false;
            for (int32_t i = 0; i < keyN && !anchorSlot; ++i) {
                if (!isAnchor(key[i])) continue;
                StrSpan iArgs[ExecutionParameters::MAX_ARITY];
                const int32_t iN = getArgsSpans(key[i], iArgs,
                    ExecutionParameters::MAX_ARITY);
                for (int32_t x = 0; x < iN; ++x) {
                    if (equalSpans(iArgs[x], args[a])) {
                        anchorSlot = true;
                        break;
                    }
                }
            }
            if (!anchorSlot) return false;
        }
        // A slot whose arguments are anchor slots alone still mints nothing
        // (the A14 shape): at least one argument must come from a
        // substantive premise.
        if (!anySubstantive) return false;

        // MINIMUM PREMISE COUNT (maintainer-set 2026-08-10), checked LAST:
        // only premise-rich rules mint demand — at least
        // kOrdis2DemandMinPremises NON-anchor premises, the qualifying
        // slot included (the B5 shape: slot + guard + two products).
        // Without this gate every short rule with a qualifying compound
        // slot installs demand-marker families across all LBs (the
        // post-eligibility-fix RT explosion).
        {
            int32_t nonAnchorN = 0;
            for (int32_t i = 0; i < keyN; ++i) {
                if (isAnchor(key[i])) continue;
                ++nonAnchorN;
            }
            if (nonAnchorN < ExecutionParameters::kOrdis2DemandMinPremises) {
                return false;
            }
        }
        return true;
    }

    // Doxygen at the declaration (prover.hpp).
    bool ExpressionAnalyzer::inputSlotDemandSlotQualifies(const StrSpan* key,
        int32_t keyN, int32_t index, int32_t inputPos, StrSpan value) const
    {
        const auto isAnchor = [](StrSpan e) -> bool {
            return e.len >= 7
                && equalSpans(StrSpan(e.ptr, 7), StrSpan("(Anchor", 7));
        };

        // The route's domain is positive operator-application premises: an
        // anchor, a negated form, and a core without a config (equality)
        // are defined misses, not errors — regular-qualified rules carry
        // such elements legitimately.
        const StrSpan elem = key[index];
        if (isAnchor(elem)) return false;
        if (elem.len > 0 && elem.ptr[0] == '!') return false;
        const ce::CoreExpressionConfig* cfg =
            coreConfig(extractExpressionSpan(elem));
        if (cfg == nullptr) return false;
        if (inputPos < 0
            || inputPos >= static_cast<int32_t>(cfg->inputIndices.size())) {
            return false;
        }
        StrSpan args[ExecutionParameters::MAX_ARITY];
        const int32_t argsN = getArgsSpans(elem, args,
                                           ExecutionParameters::MAX_ARITY);
        const int idx = cfg->inputIndices[inputPos];
        if (idx < 0 || idx >= argsN) return false;
        const StrSpan cand = args[idx];

        // (1) Candidate at a config input slot of the HEAD — the demanded
        // witness must feed the head, or releasing it buys nothing.
        if (value.len > 0 && value.ptr[0] == '!') return false;
        const ce::CoreExpressionConfig* headCfg =
            coreConfig(extractExpressionSpan(value));
        if (headCfg == nullptr) return false;
        StrSpan headArgs[ExecutionParameters::MAX_ARITY];
        const int32_t headArgsN = getArgsSpans(value, headArgs,
                                               ExecutionParameters::MAX_ARITY);
        bool atHeadInput = false;
        for (const int hIdx : headCfg->inputIndices) {
            if (hIdx >= 0 && hIdx < headArgsN
                && equalSpans(headArgs[hIdx], cand)) {
                atHeadInput = true;
                break;
            }
        }
        if (!atHeadInput) return false;

        // (2) Confinement: the candidate appears in no other key element
        // (the head is allowed). An anchor-slot candidate fails here by
        // construction — it appears in the anchor element.
        for (int32_t i = 0; i < keyN; ++i) {
            if (i == index) continue;
            StrSpan iArgs[ExecutionParameters::MAX_ARITY];
            const int32_t iN = getArgsSpans(key[i], iArgs,
                                            ExecutionParameters::MAX_ARITY);
            for (int32_t a = 0; a < iN; ++a) {
                if (equalSpans(iArgs[a], cand)) return false;
            }
        }

        // (3) Concreteness: every other argument of the marked premise
        // appears in some other key element (anchor included), so a fired
        // key is fully instantiated; a repeated candidate fails — it
        // cannot be bound.
        for (int32_t a = 0; a < argsN; ++a) {
            if (a == idx) continue;
            const StrSpan other = args[a];
            if (equalSpans(other, cand)) return false;
            bool bound = false;
            for (int32_t i = 0; i < keyN && !bound; ++i) {
                if (i == index) continue;
                StrSpan iArgs[ExecutionParameters::MAX_ARITY];
                const int32_t iN = getArgsSpans(key[i], iArgs,
                                                ExecutionParameters::MAX_ARITY);
                for (int32_t x = 0; x < iN; ++x) {
                    if (equalSpans(iArgs[x], other)) {
                        bound = true;
                        break;
                    }
                }
            }
            if (!bound) return false;
        }

        // MINIMUM PREMISE COUNT, checked LAST — the fan-out gate mirroring
        // kOrdis2DemandMinPremises (the D-267 RT lesson): only premise-rich
        // rules mint input-slot demand.
        {
            int32_t nonAnchorN = 0;
            for (int32_t i = 0; i < keyN; ++i) {
                if (isAnchor(key[i])) continue;
                ++nonAnchorN;
            }
            if (nonAnchorN < ExecutionParameters::kInputSlotDemandMinPremises) {
                return false;
            }
        }
        return true;
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
        const Memory* mbTrap,
        const RuleIndexOp& op,
        KeyEncodeMemo* memo,
        RuleIndexStaging* staging) {

        (void)mbTrap;
        // The premise encode memo: the install's per-copy memo when the
        // caller shares one (bound to this very key run — asserted), else a
        // local one for a standalone call. Both are the same contract; every
        // derivative key below reads its premises through it.
        const bool standalone = (memo == nullptr);
        KeyEncodeMemo localMemo;
        if (standalone) {
            localMemo.bind(key, keyN);
            memo = &localMemo;
        }
        assert(memo->key == key && memo->keyN == keyN
            && "makeNormalizedKeysForAdmission: the memo is bound to another key run");
        KeyEncodeMemo& encMemo = *memo;
        // A standalone call (the twin-oracle use: no shared memo) owns its
        // staging window — it acquires the slot's staging for the instance
        // and its staged index writes flush at exit on every return path; a
        // call sharing the install's memo is inside addToHashMemory, whose
        // exit (or the armed window's closer) flushes.
        const unsigned stagingSlot = (g_currentCoreId >= 0)
            ? static_cast<unsigned>(g_currentCoreId)
            : genScratchArenas().slotCount() - 1;
        if (staging == nullptr && !op.isRemove()) {
            staging = &ruleStagings().acquire(stagingSlot, &intHashMemory);
        }
        assert((op.isRemove() || staging->boundTo == &intHashMemory)
            && "makeNormalizedKeysForAdmission: the staging serves another instance");
        RuleIndexStaging& stg = (staging != nullptr) ? *staging
                                                      : ruleStagings().at(stagingSlot, 0);   // never read on the remove path
        struct StandaloneFlush {
            HashMemory* hm;
            RuleIndexStaging* st;
            ~StandaloneFlush() {
                if (hm == nullptr) return;
                ExpressionAnalyzer::flushRuleIndexStaging(*hm, *st);
                st->boundTo = nullptr;   // the standalone call is its own window
            }
        } standaloneFlush{ (standalone && !op.isRemove()) ? &intHashMemory : nullptr, &stg };
        const NameId ownerVid = nameMap.encode(validityName);
        // The admitting rule's owner identity for every derivative index entry
        // (same pair as the head install's: the implication text's ruleInterner
        // id + the install scope id).
        const RuleOwner owner =
            packRuleOwner(ruleInterner.encode(originalImpl), ownerVid);
        const auto isAnchor = [](StrSpan elem) -> bool {
            return elem.len >= 7
                && equalSpans(StrSpan(elem.ptr, 7), StrSpan("(Anchor", 7));
        };

        // THIRD, independent qualification pass
        // (D-267) — runs REGARDLESS of the two
        // mutually exclusive routes below (it neither enables nor
        // suppresses them): each qualifying premise slot installs a
        // marker-style variant whose key is every other premise, including
        // the anchor context, and whose VALUE is the slot verbatim (no marker
        // token, polarity kept), flagged isMarker + ordis2Demand. Retaining
        // the anchor preserves the normalized equality classes that bind an
        // actual rule's anchor slots to its non-anchor premises. When the key
        // matches, the firing site's demand branch back-substitutes the match
        // into the value — the GROUND missing premise — and stages a demand
        // record for the post-fixpoint drain.
        {
            RT_SCOPE_HERE("ADM_PASS_ORDIS2");
            constexpr int32_t kO2KeyCap =
                ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS + 1;
            assert(keyN <= kO2KeyCap
                && "ordis2-demand pass: key size exceeds cap");
            for (int32_t index = 0; index < keyN; ++index) {
                if (!ordis2DemandSlotQualifies(key, keyN, index)) continue;
                int binary[kO2KeyCap];
                for (int32_t i = 0; i < keyN; ++i) binary[i] = 0;
                StrSpan subkey[kO2KeyCap];
                int32_t subCount = 0;
                for (int32_t i = 0; i < keyN; ++i) {
                    if (i == index) continue;
                    binary[i] = 1;
                    subkey[subCount++] = key[i];
                }
                // The subset condition binds >= 4 args through non-anchor
                // others, so a qualifying slot always has a subkey.
                assert(subCount >= 1
                    && "ordis2-demand pass: qualifying slot with empty subkey");
                installAdmissionMarkerVariants(key,
                    static_cast<std::size_t>(keyN),
                    static_cast<std::size_t>(index), StrSpan(),
                    binary, subkey, subCount, intHashMemory, nameMap,
                    ruleInterner, originalImpl, ownerVid, owner,
                    /*ordisOnly=*/false, /*ordis2Demand=*/true, op, encMemo, stg);
            }
        }

        // FOURTH, independent qualification pass
        // (D-288) — like the ordis2 pass above,
        // it neither enables nor suppresses the two mutually exclusive
        // routes below: each qualifying (premise, input slot) pair installs
        // an UNTAGGED marker variant whose marked argument is the premise's
        // input-slot candidate and whose subkey is every other premise,
        // anchor included. When the subkey matches live facts, the marker
        // branch stages the INSTANTIATED key — the exact marker form under
        // which a witness-minting disintegration probes (and parks in
        // rejectedMap): the demanded witness registers instead of parking,
        // and revisitRejected2 wakes an already-parked product. Untagged,
        // so isAdmitted reads it (unlike the ordisOnly route's or-cohort
        // evidence); consumers probe through the ancestor-inclusive twin
        // because a fired key lands at deeperOf(subkey constituents) while
        // the disintegration may deposit in a descendant (ordis-branch)
        // scope.
        {
            RT_SCOPE_HERE("ADM_PASS_INPUT_SLOT");
            constexpr int32_t kISKeyCap =
                ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS + 1;
            assert(keyN <= kISKeyCap
                && "input-slot demand pass: key size exceeds cap");
            for (int32_t index = 0; index < keyN; ++index) {
                if (isAnchor(key[index])) continue;
                if (key[index].len > 0 && key[index].ptr[0] == '!') continue;
                const ce::CoreExpressionConfig* elemCfg =
                    coreConfig(extractExpressionSpan(key[index]));
                if (elemCfg == nullptr) continue;
                StrSpan elemArgs[ExecutionParameters::MAX_ARITY];
                const int32_t elemArgsN = getArgsSpans(key[index], elemArgs,
                    ExecutionParameters::MAX_ARITY);
                const int32_t inputN =
                    static_cast<int32_t>(elemCfg->inputIndices.size());
                for (int32_t ip = 0; ip < inputN; ++ip) {
                    if (!inputSlotDemandSlotQualifies(key, keyN, index, ip,
                            value)) {
                        continue;
                    }
                    const int idx = elemCfg->inputIndices[ip];
                    assert(idx >= 0 && idx < elemArgsN
                        && "input-slot demand pass: qualified slot out of range");
                    const StrSpan cand = elemArgs[idx];
                    int binary[kISKeyCap];
                    for (int32_t i = 0; i < keyN; ++i) binary[i] = 0;
                    StrSpan subkey[kISKeyCap];
                    int32_t subCount = 0;
                    for (int32_t i = 0; i < keyN; ++i) {
                        if (i == index) continue;
                        binary[i] = 1;
                        subkey[subCount++] = key[i];
                    }
                    assert(subCount >= 1
                        && "input-slot demand pass: qualifying slot with empty subkey");
                    installAdmissionMarkerVariants(key,
                        static_cast<std::size_t>(keyN),
                        static_cast<std::size_t>(index), cand,
                        binary, subkey, subCount, intHashMemory, nameMap,
                        ruleInterner, originalImpl, ownerVid, owner,
                        /*ordisOnly=*/false, /*ordis2Demand=*/false, op, encMemo, stg);
                }
            }
        }

        // Two mutually exclusive qualification routes: the regular
        // (A)/(B)/(C) gates retain their existing eligibility criteria; a
        // rule failing all three may still install ordisOnly-tagged marker
        // keys through the ordis candidate loop below — or-cohort opening
        // demand evidence, invisible to general Pass-B admission.
        const bool regularQualified =
            implicationIsQualified(key, keyN, value, minNumOperatorsKey);
        if (!regularQualified && !ordisRouteQualifies(key, keyN, value)) {
            return;
        }

        const std::size_t n = static_cast<std::size_t>(keyN);

        if (!regularQualified) {
            RT_SCOPE_HERE("ADM_PASS_ORDIS");
            // Ordis-only route: candidates are variables at config INPUT
            // slots of a premise, gated by (1) presence at a config input
            // slot of the HEAD (the maintainer-adopted extra filter), (2)
            // confinement — the candidate appears in no other key element
            // (the head is allowed), (3) concreteness — every other argument
            // of the marked premise appears in some subkey element, so a
            // fired key is fully instantiated. The subkey is every other
            // premise, including the anchor context (confinement already
            // guarantees none of them mentions the candidate). Deterministic
            // order: premise index ascending, config input-slot order.
            constexpr int32_t kKeyCap =
                ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS + 1;
            assert(n <= static_cast<std::size_t>(kKeyCap)
                && "makeNormalizedKeysForAdmission ordis: key size exceeds cap");

            const ce::CoreExpressionConfig* headCfg =
                coreConfig(extractExpressionSpan(value));
            assert(headCfg
                && "ordis route: head operator core must be compiled");
            StrSpan headArgs[ExecutionParameters::MAX_ARITY];
            const int32_t headArgsN = getArgsSpans(value, headArgs,
                                                   ExecutionParameters::MAX_ARITY);

            for (std::size_t index = 0; index < n; ++index) {
                const StrSpan elem = key[index];
                if (isAnchor(elem)) continue;
                const ce::CoreExpressionConfig* cfg =
                    coreConfig(extractExpressionSpan(elem));
                assert(cfg && "ordis route: premise operator core must be compiled");
                StrSpan args[ExecutionParameters::MAX_ARITY];
                const int32_t argsN = getArgsSpans(elem, args,
                                                   ExecutionParameters::MAX_ARITY);

                for (const int inIdx : cfg->inputIndices) {
                    if (inIdx < 0 || inIdx >= argsN) continue;
                    const StrSpan cand = args[inIdx];

                    // (1) Candidate at a config input slot of the head.
                    bool atHeadInput = false;
                    for (const int hIdx : headCfg->inputIndices) {
                        if (hIdx >= 0 && hIdx < headArgsN
                            && equalSpans(headArgs[hIdx], cand)) {
                            atHeadInput = true;
                            break;
                        }
                    }
                    if (!atHeadInput) continue;

                    // (2) Confinement: cand in no other key element.
                    bool confined = true;
                    for (std::size_t i = 0; i < n && confined; ++i) {
                        if (i == index) continue;
                        StrSpan iArgs[ExecutionParameters::MAX_ARITY];
                        const int32_t iArgsN = getArgsSpans(key[i], iArgs,
                            ExecutionParameters::MAX_ARITY);
                        for (int32_t a = 0; a < iArgsN; ++a) {
                            if (equalSpans(iArgs[a], cand)) {
                                confined = false;
                                break;
                            }
                        }
                    }
                    if (!confined) continue;

                    // Subkey = every other premise, including the anchor.
                    int binary[kKeyCap];
                    for (int32_t i = 0; i < static_cast<int32_t>(n); ++i)
                        binary[i] = 0;
                    StrSpan subkey[kKeyCap];
                    int32_t subCount = 0;
                    for (std::size_t i = 0; i < n; ++i) {
                        if (i == index) continue;
                        binary[i] = 1;
                        subkey[subCount++] = key[i];
                    }

                    // (3) Concreteness: every other argument of the marked
                    // premise appears among some subkey element's args (a
                    // repeated candidate arg fails — it cannot be bound).
                    bool concrete = true;
                    for (int32_t a = 0; a < argsN && concrete; ++a) {
                        if (a == inIdx) continue;
                        const StrSpan other = args[a];
                        if (equalSpans(other, cand)) {
                            concrete = false;
                            break;
                        }
                        bool bound = false;
                        for (int32_t s = 0; s < subCount && !bound; ++s) {
                            StrSpan sArgs[ExecutionParameters::MAX_ARITY];
                            const int32_t sArgsN = getArgsSpans(subkey[s],
                                sArgs, ExecutionParameters::MAX_ARITY);
                            for (int32_t sa = 0; sa < sArgsN; ++sa) {
                                if (equalSpans(sArgs[sa], other)) {
                                    bound = true;
                                    break;
                                }
                            }
                        }
                        if (!bound) concrete = false;
                    }
                    if (!concrete) continue;

                    installAdmissionMarkerVariants(key, n, index, cand,
                        binary, subkey, subCount, intHashMemory, nameMap,
                        ruleInterner, originalImpl, ownerVid, owner,
                        /*ordisOnly=*/true, /*ordis2Demand=*/false, op, encMemo, stg);
                }
            }
            return;
        }

        RT_SCOPE_HERE("ADM_PASS_REGULAR");
        for (std::size_t index = 0; index < n; ++index) {
            // The anchor is permanent context for every derivative rule; it
            // is never itself the missing premise.
            if (isAnchor(key[index])) continue;
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
                if (isAnchor(key[i])) {
                    // The anchor always remains in the derivative key, while
                    // the existing non-output qualification count is unchanged.
                    binary[i] = 1;
                    subkey[subCount++] = key[i];
                    if (!containsOutput) validCount++;
                    continue;
                }
                if (!containsOutput) {
                    binary[i] = 1;
                    subkey[subCount++] = key[i];
                    validCount++;
                }
            }

            // Classic eligibility remains based on the existing (A)/(B)
            // qualification and non-output premise count. Anchor retention
            // changes only the installed derivative key.
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

            installAdmissionMarkerVariants(key, n, index, outputArg,
                binary, subkey, subCount, intHashMemory, nameMap,
                ruleInterner, originalImpl, ownerVid, owner,
                /*ordisOnly=*/false, /*ordis2Demand=*/false, op, encMemo, stg);
        }
    }

    void ExpressionAnalyzer::installAdmissionMarkerVariants(
        const StrSpan* key, std::size_t n,
        std::size_t index, StrSpan markedArg,
        const int* binary, const StrSpan* subkey, int32_t subCount,
        HashMemory& intHashMemory,
        NameMap& nameMap,
        ValueInterner& ruleInterner,
        StrSpan originalImpl,
        NameId ownerVid,
        RuleOwner owner,
        bool ordisOnly,
        bool ordis2Demand,
        const RuleIndexOp& op,
        KeyEncodeMemo& memo,
        RuleIndexStaging& staging) {
        {
            constexpr int32_t kKeyCap =
                ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS + 1;
            assert(n <= static_cast<std::size_t>(kKeyCap)
                && "installAdmissionMarkerVariants: key size exceeds cap");
            assert(memo.key == key && memo.keyN == static_cast<int32_t>(n)
                && "installAdmissionMarkerVariants: the memo is bound to another key run");

            // subkey[j] is key[subIdx[j]] — the j-th premise the caller's
            // mask selected, in key order (the span identity is asserted) —
            // so the memo is read through the mask.
            int32_t subIdx[kKeyCap];
            {
                int32_t j = 0;
                for (std::size_t i = 0; i < n; ++i) {
                    if (!binary[i]) continue;
                    assert(j < subCount
                        && "installAdmissionMarkerVariants: more masked premises than subkey elements");
                    assert(subkey[j].ptr == key[i].ptr && subkey[j].len == key[i].len
                        && "installAdmissionMarkerVariants: a subkey element is not the masked premise");
                    subIdx[j++] = static_cast<int32_t>(i);
                }
                assert(j == subCount
                    && "installAdmissionMarkerVariants: mask and subkey disagree");
            }

            // Replace the marked arg with "marker" -> a ScratchString on the
            // string tier (replaceKeysScratch, byte-exact to
            // ce::replaceKeysInString), held across both blocks by mnkStrScope.
            // The ordis2-demand route keeps the value VERBATIM instead — no
            // marker token; changeable args ride the decimal rename below
            // and back-substitute at fire, unchangeable u_ args match
            // literally and the firing path's u_-strip grounds them — the
            // fired demand is ground either way.
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
            markerPair[0].key = ordis2Demand ? StrSpan() : markedArg;
            markerPair[0].value = StrSpan("marker", 6);
            const ScratchString replaced = ordis2Demand
                ? ScratchString::copyFrom(mnkStrArena, key[index].ptr,
                                          key[index].len)
                : replaceKeysScratch(mnkStrArena, key[index], markerPair, 1);

            // --- Int subkeys (mirrors makeNormalizedSubkeys for HashMemory) ---
            {
                const int32_t sn = subCount;
                auto pit = this->allPermutationsAna.find(sn);
                if (pit != this->allPermutationsAna.end()) {
                    const auto& permuts = pit->second;
                    RT_SCOPE_HERE("ADM_SUBKEYS");
                    RT_NOTE_ITERATIONS_HERE(static_cast<int64_t>(permuts.size()));
                    StrSpan sids[kKeyCap];
                    for (int32_t i = 0; i < sn; ++i)
                        sids[i] = extractExpressionSpan(subkey[i]);

                    for (std::size_t p = 0; p < permuts.size(); ++p) {
                        const auto& permut = permuts[p];

                        // Premises from the per-copy memo (through the mask);
                        // the normalized key built ONCE, one premise per
                        // prefix step — after step L the first `st.pos` slots
                        // are the subkey of prefix length L (the resumable
                        // builder's contract). subEncoded[0..si] is the prefix
                        // in key order for the signature record.
                        IntEncodedExpr subEncoded[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
                        NameId subBuf[ExecutionParameters::MAX_KEY_SLOTS];
                        NormKeyBuildState st;
                        st.nVars = 0;
                        st.nextNormId = 1;
                        st.pos = 0;

                        for (int32_t si = 0; si < sn; ++si) {
                            bool toBreak = false;
                            for (int32_t si2 = 0; si2 < si; ++si2) {
                                if (compareSpans(sids[permut[si2]],
                                                 sids[permut[si2 + 1]]) > 0) {
                                    toBreak = true; break;
                                }
                            }
                            if (toBreak) break;

                            const NameId subEncCount = static_cast<NameId>(si + 1);
                            assert(subEncCount <= ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS
                                && "makeNormalizedKeysForAdmission subkey: element count exceeds cap");
                            {
                            RT_SCOPE_HERE("ADM_SK_ENCODE");
                            subEncoded[si] = memo.at(subIdx[permut[si]], nameMap);
                            }

                            NameId subLen = 0;
                            {
                            RT_SCOPE_HERE("ADM_SK_NORMKEY");
                            appendExprToIntNormalizedKey(subEncoded[si], st, subBuf,
                                ExecutionParameters::MAX_KEY_SLOTS);
                            subLen = st.pos;
                            }
                            // Subkey map via the raw-key door (no owning NormKey):
                            // the owner on the signature-free record below
                            // kSubkeyUCheckMinElements, on the u_-signature
                            // record from there on; under the remove policy the
                            // owner's pairs leave the record.
                            if (op.isRemove()) {
                                removeRuleSubkeyEntry(intHashMemory, op,
                                    static_cast<NameId>(si + 1), subBuf, subLen,
                                    owner, genScratchArenas().forSlot(mnkGenSlot));
                            } else if (subEncCount < kSubkeyUCheckMinElements) {
                                RT_SCOPE_HERE("ADM_SK_WRITE_SHORT");
                                stageShortSubkeyOwner(staging.subkeys,
                                    static_cast<NameId>(si + 1), subBuf, subLen, owner);
                            } else {
                                RT_SCOPE_HERE("ADM_SK_WRITE_MERGE");
                                stageSubkeySignatures(staging.subkeys,
                                    static_cast<NameId>(si + 1), subBuf, subLen,
                                    subEncoded, subEncCount, owner);
                            }
                        }
                    }
                }
            }

            // --- Main variants: create int keys for encodedMap marker entries ---
            {
                const int32_t sn = subCount;
                auto pit = this->allPermutationsAna.find(sn);
                if (pit == this->allPermutationsAna.end()) return;
                const auto& permuts = pit->second;
                RT_SCOPE_HERE("ADM_WHOLEKEY");
                RT_NOTE_ITERATIONS_HERE(static_cast<int64_t>(permuts.size()));

                StrSpan sids[kKeyCap];
                for (int32_t i = 0; i < sn; ++i)
                    sids[i] = extractExpressionSpan(subkey[i]);

                ScratchArena& mnkGenArena = genScratchArenas().forSlot(mnkGenSlot);

                // getRemainingArgs yields a compareSpans sorted-unique run == the
                // former std::set<std::string> order. MINT the NameMap ids in that
                // run order (I-84: mint order reaches the deload stream), then a
                // SORTED-ASCENDING copy is the Int16SetKey (the former
                // std::set<NameId> iteration order). The lmv.remainingArgIds mint
                // (ruleInterner) walks the run order = former set-lex order.
                StrSpan remScratch[ExecutionParameters::MAX_ADMISSION_REM_ARGS];
                const int32_t remScratchN = getRemainingArgs(
                    subkey, subCount, remScratch,
                    ExecutionParameters::MAX_ADMISSION_REM_ARGS);
                NameId intRemArgs[ExecutionParameters::MAX_ADMISSION_REM_ARGS];
                for (int32_t i = 0; i < remScratchN; ++i)
                    intRemArgs[i] = nameMap.encode(remScratch[i]);   // mint in run order
                NameId intRemArgsSorted[ExecutionParameters::MAX_ADMISSION_REM_ARGS];
                if (remScratchN > 0)
                    std::memcpy(intRemArgsSorted, intRemArgs,
                        static_cast<std::size_t>(remScratchN) * sizeof(NameId));
                std::sort(intRemArgsSorted, intRemArgsSorted + remScratchN);

                NameId bufIgnored[ExecutionParameters::MAX_KEY_SLOTS];
                NameId bufNotIgnored[ExecutionParameters::MAX_KEY_SLOTS];

                // Batch the remaining-args insert across the permutation loop:
                // the key (intRemArgsSorted/remScratchN) is loop-invariant, so
                // accumulate each permutation's NormKey and do ONE RMW after the
                // loop (D-199). The byte-bump blobs survive
                // the loop's self-framing appendLmvIdsRecord.
                const ArenaOffset remArgsBatchMark = mnkGenArena.cursor();
                DirtyState remArgsBatchDirty = DirtyState::Clean;
                PagedVector<RemArgsBatchBlob> remArgsBatch(
                    &mnkGenArena, &remArgsBatchDirty);

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
                    const NameId intEncCount = static_cast<NameId>(permutation.size());
                    assert(intEncCount <= ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS
                        && "makeNormalizedKeysForAdmission: subkey element count exceeds cap");
                    {
                    RT_SCOPE_HERE("ADM_WK_ENCODE");
                    for (std::size_t k = 0; k < permutation.size(); ++k)
                        intEncoded[k] = memo.at(subIdx[permutation[k]], nameMap);
                    }

                    NameId reverseMap[ExecutionParameters::MAX_KEY_SLOTS];
                    std::memset(reverseMap, 0, sizeof(reverseMap));
                    NameId numNormVars = 0;
                    NameId lenIgnored = 0;
                    NameId lenNotIgnored = 0;
                    {
                    RT_SCOPE_HERE("ADM_WK_NORMKEY");
                    lenIgnored = makeIntNormalizedKeyFromEncodedWithMap(
                        intEncoded, intEncCount, true, bufIgnored,
                        ExecutionParameters::MAX_KEY_SLOTS, reverseMap, numNormVars);
                    lenNotIgnored = makeIntNormalizedKeyFromEncoded(
                        intEncoded, intEncCount, false, bufNotIgnored, ExecutionParameters::MAX_KEY_SLOTS);
                    }

                    // Remove policy: this permutation's whole key and
                    // remaining-args edge lose the owner (first visit only);
                    // no marker LMV, no batch.
                    if (op.isRemove()) {
                        removeRuleKeyEntries(intHashMemory, op,
                            static_cast<NameId>(sn), bufNotIgnored, lenNotIgnored,
                            intRemArgsSorted, remScratchN, owner, mnkGenArena);
                        continue;
                    }

                    // Rename run { normalized-var name -> decimal id } from
                    // reverseMap. mp2 order is NOT observable (replaceKeysScratch is
                    // greedy-longest, order-independent), so id order suffices —
                    // byte-identical to the former lex-ordered std::map. The key
                    // spans are NameMap decodeView: no NameMap mint falls between
                    // here and the last replaceKeysScratch use (the mints below are
                    // ruleInterner), so decodeView is I-3-safe (proved per-site).
                    StrReplacement mp2Pairs[ExecutionParameters::MAX_KEY_SLOTS];
                    int32_t mp2PairsN = 0;
                    int32_t valueId = 0;
                    bool isMarker = false;
                    {
                    RT_SCOPE_HERE("ADM_WK_VALUE_VARIANT");
                    for (NameId id = 1; id <= numNormVars; ++id) {
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

                    // Marker value id + isMarker flag (id-run door, no owning
                    // LMV). An ordis2-demand value carries no marker token
                    // and is NOT flagged isMarker — the demand kind is its
                    // own (maintainer decision 2026-08-10); the firing
                    // dispatch checks ordis2Demand FIRST, so the head
                    // branch never fires it.
                    valueId = ruleInterner.encode(StrSpan(valueVariant));
                    isMarker =
                        containsSpan(StrSpan(valueVariant), StrSpan("marker", 6));
                    } // RT_SCOPE ADM_WK_VALUE_VARIANT

                    // replKey ids: each valid subkey element (binary[i], subkey only
                    // — never key[index]) with the rename applied, minted in binary
                    // order == the former replKey vector order.
                    int32_t keyIds[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
                    int32_t keyIdsN = 0;
                    {
                    RT_SCOPE_HERE("ADM_WK_KEYIDS");
                    for (std::size_t i = 0; i < n; ++i) {
                        if (binary[i]) {
                            assert(keyIdsN < ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS
                                && "makeNormalizedKeysForAdmission replKey: element count exceeds cap");
                            const ScratchString e = replaceKeysScratch(
                                mnkStrArena, key[i], mp2Pairs, mp2PairsN);
                            keyIds[keyIdsN++] = ruleInterner.encode(StrSpan(e));
                        }
                    }
                    } // RT_SCOPE ADM_WK_KEYIDS

                    // remainingArgIds: walk remScratch (run order == former set-lex).
                    int32_t remIds[ExecutionParameters::MAX_ADMISSION_REM_ARGS];
                    int32_t originalImplId = 0;
                    {
                    RT_SCOPE_HERE("ADM_WK_LMV_FIELDS");
                    for (int32_t i = 0; i < remScratchN; ++i)
                        remIds[i] = ruleInterner.encode(remScratch[i]);
                    originalImplId = ruleInterner.encode(originalImpl);
                    }

                    // encodedMap marker record + owner record + remaining-args index,
                    // all via the raw / id-run doors (no owning NormKey / LMV). D-72:
                    // the LMV carries the admitting implication + scope. The
                    // trailing defaults are spelled out to reach the ordisOnly
                    // slot; regular installs pass false — byte-identical.
                    {
                    RT_SCOPE_HERE("ADM_LMV_APPEND");
                    stageLmvIdsRecord(staging.lmv,
                        static_cast<NameId>(sn), bufIgnored, lenIgnored,
                        valueId, isMarker, keyIds, keyIdsN, remIds, remScratchN,
                        originalImplId, ownerVid, mnkGenArena,
                        nullptr, 0, RuleJustification::none, false, true,
                        ordisOnly, /*subsetExclusion=*/false, ordis2Demand);
                    }
                    {
                    RT_SCOPE_HERE("ADM_WHOLEKEY_OWNER");
                    stageWholeKeyOwner(staging.wholeKeys,
                        static_cast<NameId>(sn), bufNotIgnored, lenNotIgnored, owner);
                    }
                    // Accumulate this permutation's NormKey (Codec<NormKey> bytes)
                    // for the one batched insert after the loop.
                    {
                        const int32_t nkBlobLen = (lenNotIgnored + 2)
                            * static_cast<int32_t>(sizeof(NameId));
                        const ArenaOffset nkBlobOff = mnkGenArena.alloc(
                            nkBlobLen, static_cast<int32_t>(alignof(NameId)));
                        NameId* pnk = reinterpret_cast<NameId*>(
                            mnkGenArena.resolve(nkBlobOff));
                        pnk[0] = static_cast<NameId>(sn);
                        pnk[1] = lenNotIgnored;
                        std::memcpy(pnk + 2, bufNotIgnored,
                            static_cast<std::size_t>(lenNotIgnored) * sizeof(NameId));
                        remArgsBatch.push_back(
                            RemArgsBatchBlob{ nkBlobOff, nkBlobLen });
                    }
                }
                // ONE RMW for the whole permutation batch into the invariant key.
                {
                RT_SCOPE_HERE("ADM_REMARGS_BATCH");
                stageRemainingArgsNormKeyBatch(
                    staging.remArgs,
                    staging.remArgsOwners,
                    intRemArgsSorted, remScratchN, remArgsBatch, owner, mnkGenArena);
                }
                mnkGenArena.popTo(remArgsBatchMark);
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
    // Hash-engine request generation.
    // filterIntEncodedStatements was inline in prover.hpp's class body;
    // its declaration stays there, the body lives here alongside its
    // single caller, the request generator.
    // ------------------------------------------------------------------

    /// @brief Filter pre-encoded statements down to those that pass the
    /// fast-rejection owner-set check and the max-iteration cap.
    ///
    /// @details
    /// For each `IntEncodedExpr`, builds a single-expression
    /// `IntNormalizedKey` on a stack buffer (no heap allocation), normalizes
    /// the var ids sequentially by first appearance (`changeable=0`,
    /// `ignoreU=false`), and tests:
    /// 1. `subkeyUSatisfied` against `mem.normalizedEncodedSubkeys` — the
    ///    fast-rejection subkey record map populated by `addToHashMemory` /
    ///    `makeNormalizedKeysForAdmission`, probed for key presence plus the
    ///    D-120 `u_` literal prune only (scope comparability and partition are
    ///    not read). When @p alsoAcceptFullKeys is set, a hit
    ///    in `mem.normalizedEncodedKeys` (`wholeKeyPresent` — key presence only,
    ///    the owner record is not read) accepts
    ///    the statement as well, so the gate becomes the UNION of the two maps.
    ///    Misses are dropped.
    /// 2. `maxIteration <= parameters.maxIterationNumberVariable` — caps
    ///    blow-up on iteration counters.
    ///
    /// The union form is what an EMPTY obligatory stump needs: with no mandatory
    /// element the statement may itself be a complete request, so a full-key hit
    /// must keep it even though it can never be grown further. With a non-empty
    /// stump every survivor still has to be extended, so only a subkey hit counts.
    /// That is the whole difference between the counter-example filter's statement
    /// universe and the main prover's.
    ///
    /// Output is the surviving indices into `stmts`, in original order.
    /// No heap allocation; the in-function `buf` and `varMap` are bounded
    /// by `ExecutionParameters::MAX_KEY_SLOTS`.
    ///
    /// @param stmts              Source view of pre-encoded statements.
    /// @param mem                Reference `HashMemory` for the owner-set checks.
    /// @param nm                 Scope name map, for the comparability predicate.
    /// @param alsoAcceptFullKeys Widen the gate with `normalizedEncodedKeys`;
    ///                           true for every firing-request generator call,
    ///                           including split-stump buckets, and false only
    ///                           for stump production.
    /// @param outIndices         Destination array for surviving indices.
    /// @param maxOut             Capacity of `outIndices`.
    /// @return Count of surviving indices written into `outIndices`.
    /// @pre  The container behind `stmts` and `outIndices` live for the
    ///       duration of the call.
    /// @invariant The prune never drops a statement that could take part in a
    ///            firing request ([I-70](../../docs/agentic_swdd/30_invariants.md#i-70),
    ///            [I-79](../../docs/agentic_swdd/30_invariants.md#i-79)).
    NameId ExpressionAnalyzer::filterIntEncodedStatements(
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
                && wholeKeyPresent(mem.normalizedEncodedKeys, buf, pos, 1);
            if (!subOk && !keyOk)
                continue;
            if (s.maxIteration > parameters.maxIterationNumberVariable)
                continue;

            outIndices[nOut++] = i;
        }
        return nOut;
    }

    /// @brief THE request generator: grow every candidate to a whole key and emit
    /// the ones that carry a mandatory ingredient.
    ///
    /// @details
    /// See the declaration in `prover.hpp` for the phase-by-phase contract. The body
    /// below is a direct transcription of it.
    ///
    /// Two details are load-bearing and easy to break:
    ///
    /// - **The submatch tally.** `g_growthMatchCount` is bumped once per node whose
    ///   key `subkeyUSatisfied` accepts on `normalizedEncodedSubkeys` (presence +
    ///   the D-120 u_ signature; scope and partition are not read). The tally is
    ///   the doom-line coordinate and the LB-split policy's work statistic
    ///   (D-109), so a second probe per node would change the split decisions and
    ///   therefore the proof. That is why the node runs `requestGatesPass` +
    ///   `makeIntNormalizedKeyFromEncoded` + one bare `subkeyUSatisfied` growth
    ///   probe + one bare `wholeKeyPresent` record probe.
    /// - **Emission happens inside the search.** A recorded candidate IS the
    ///   finished request. Buffering the candidates and emitting after the search
    ///   would defer every firing past the whole enumeration and defeat the
    ///   counter-example filter's contradiction early-exit, which is the only thing
    ///   that halts a CE burst (I-73).
    ///
    /// @see The declaration in `prover.hpp` for the full parameter documentation.
    template <typename Consumer>
    void ExpressionAnalyzer::generateEncodedRequestsStatic(
        const Memory& body,
        const HashMemory& intMemory,
        const MandatoryTerm* terms, NameId termCount,
        const SplitStumpRef& splitStump,
        unsigned coreId,
        Consumer& consumer)
    {
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
        NameId filteredIdx[8192];
        NameId nFiltered = 0;
        {
        // RT part 1 of 2: statement-universe filter + name sort — the fixed
        // per-call cost the whole bucket shares.
        RT_SCOPE_HERE("STATIC_REQGEN_FILTER_SORT");
        // A candidate is the whole request, so a statement that is a whole key on
        // its own is admitted alongside the growable subkeys.
        nFiltered = filterIntEncodedStatements(allIntStmts,
            intMemory, body, /*alsoAcceptFullKeys=*/true, filteredIdx, 8192);
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
        NormKeyBuildState keyState;
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
                const NameId keyLen = makeIntNormalizedKeyFromEncoded(candPtrs,
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
                if (wholeKeyPresent(targetKeys, buf, keyLen, curCount)
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
                // Its bytes stay in `buf[0, prefixKeyLen)` for the whole pop —
                // nothing in the loop writes that buffer — so each position only
                // rewinds the three counters and folds in its own premise.
                buildIntNormalizedKeyPrefix(candPtrs, top.count, buf,
                                            ExecutionParameters::MAX_KEY_SLOTS,
                                            keyState);
                const NameId prefixVars = keyState.nVars;
                const NameId prefixNextNormId = keyState.nextNormId;
                const NameId prefixKeyLen = keyState.pos;

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
                    keyState.nVars = prefixVars;
                    keyState.nextNormId = prefixNextNormId;
                    keyState.pos = prefixKeyLen;
                    appendExprToIntNormalizedKey(ie, keyState, buf,
                                                 ExecutionParameters::MAX_KEY_SLOTS);
                    const NameId keyLen = keyState.pos;

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
                    if (wholeKeyPresent(targetKeys, buf, keyLen, newCount)
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

    /// @brief The stump list a rule-part returns when it reaches the second
    /// submatch cap: the grow search with no obligatory stump, stopped early.
    ///
    /// @details
    /// See the declaration in `prover.hpp` for the contract. The body is the
    /// grow phase of `generateEncodedRequestsStatic` with three differences:
    /// it probes only `normalizedEncodedSubkeys` (a stump is a candidate that
    /// survives the filter, nothing more), it never bumps `g_growthMatchCount`,
    /// and it walks level by level instead of depth-first so it can stop the
    /// moment one whole level reaches @p target.
    ///
    /// Candidates are stored as ascending positions into the name-sorted
    /// `filteredIdx`, which makes the child enumeration (`j > last position`)
    /// visit every set exactly once — the same ascending walk the search's
    /// `start = i + 1` performs. Positions become statement indices only at
    /// emit, because the consumer rebuilds a different `filteredIdx` for each
    /// request batch.
    ///
    /// @see The declaration in `prover.hpp` for the full parameter documentation.
    int32_t ExpressionAnalyzer::produceExpressionStumps(
        const Memory& body, unsigned coreId, int32_t target, SealedPageSet& out)
    {
        assert(target > 0
            && "produceExpressionStumps: the stump-count floor must be positive");

        const NameMap& nm = body.nameMap;
        const HashMemory& mem = body.overallHashMemory;
        const IntStmtView allIntStmts(body.intEncodedStatements);
        ScratchArena& genArena = genScratchArenas().forSlot(coreId);

        // The statement universe, filtered by the subkey growth probe, then
        // name-sorted exactly as the search sorts it (ties keep ascending
        // statement index).
        NameId filteredIdx[8192];
        const NameId nFiltered = filterIntEncodedStatements(allIntStmts,
            mem, body, /*alsoAcceptFullKeys=*/false, filteredIdx, 8192);
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
            return makeIntNormalizedKeyFromEncoded(ptrs, n.count, buf,
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
                    const NameId keyLen = makeIntNormalizedKeyFromEncoded(ptrs,
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
                    wholeKeyPresent(mem.normalizedEncodedKeys, buf, keyLen, p.count);
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

            const NameId len = makeIntNormalizedKeyFromEncodedWithMap(
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
                    // admv, no sawMarker path). Purity discipline as the
                    // marker branch's: an
                    // impure match outside an or branch registers no
                    // demand.
                    const bool inOrBranch =
                        containsSpan(hitValidityView,
                                     StrSpan("_boundary_orint_", 16));
                    if (!pure && !inOrBranch) {
                        continue;
                    }

                    // I-171 mirror (maintainer-approved 2026-08-10): a
                    // demand whose ground text carries witnesses beyond the
                    // per-batch generation cap is not staged — the
                    // admission side gates depth the same way, and without
                    // this every deeper it_ generation would mint a fresh
                    // distinct demand key (the RMW dedup collapses only
                    // exact repeats).
                    if (extractMaxIterationNumber(StrSpan(rplScratch2))
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
                        if (this->isAnchorLb(memoryBlock)) {
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


                    bool allGood = (hitValidityId == NameMap::MAIN_ID);
                    for (NameId i = 0; i < reqCount; ++i) {
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
                        // per-firing heap std::string on this hot path. The
                        // scan is the shared `ancestorKnown` predicate (the
                        // single definition of the Site F contract).
                        const NameId origId =
                            memoryBlock.nameMap.lookup(StrSpan(rplScratch2));
                        const NameId valId  = hitValidityId;
                        alreadyKnown = origId != 0
                            && ancestorKnown(memoryBlock, origId, valId,
                                             /*includeSelf=*/true);
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
                        // Park-first (I-184):
                        // a single-exclusion rule IS a disintegration product,
                        // but its fired reduced-or head must not open a cohort
                        // via route (b) — allowOrProbe stays untouched, so the
                        // head parks in rejectedMapOrdis and opens only from
                        // the compound-demand map.
                        rec.allowOrDisintegration =
                            lmv.productOfDisintegration()
                            && !lmv.subsetExclusion();
                        rec.allGood = allGood;
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
                    // Purity gate is relaxed ONLY inside OR-branch scopes
                    // (`_boundary_orint_` in the validity name). Case-split
                    // branches legitimately carry fresh it_/int_ witnesses
                    // that are iteration-bearing but not yet products of
                    // recursion — denying those admissions leaves §4.2-style
                    // ladder steps unreachable.
                    // Anywhere else (main / hypo / non-or boundary) the
                    // original purity gate stands, to prevent runaway fan-out.
                    const bool inOrBranch =
                        containsSpan(hitValidityView,
                                     StrSpan("_boundary_orint_", 16));
                    if (!pure && !inOrBranch) {
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
                            seal(hitValidityView);
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

    /// @brief Canonically order one hashburst's captured firing records when
    ///        needed, then apply their deposits to the LB.
    ///
    /// @details
    /// Definition of the consumer declared in `prover.hpp`. Replays the deposit
    /// side of `checkLocalEncodedMemoryStatic` for one static request-evaluation
    /// pass, reading the records off the LB's sealed part sets (part order,
    /// each chain in append order). Processor chains build a pointer INDEX over
    /// the chains' stable payload addresses and sort it by a total content key
    /// (expression, validity, kind, then per-kind fields). A GPU chain already
    /// carrying that exact order walks its index directly. The resulting
    /// container state is a function of the firing SET, not the request-
    /// generation ORDER — the determinism the LB split relies on. Head records
    /// feed `canBeSentIds`
    /// (gated on `allGood`) and, when not already known,
    /// `sameIterationInternalMail.statements` + `.exprOriginMap` (capped
    /// `addOrigin`) + `.disintegrationSignals`. Marker records feed
    /// `deferredIntegrationPreps`, `canBeSentMarkerIds` (non-atomic), and
    /// `admissionKeysAlgebra`. Deactivation and discharge are deferred to Phase
    /// 3's post-burst absorb, so they are not handled here. The sealed views feed the span
    /// doors directly — no per-record string materialization at this
    /// boundary; the doors intern the sealed bytes into the LB's NameMap /
    /// originInterner with the identical touch order the string doors
    /// performed.
    ///
    /// @param memoryBlock LB whose containers receive the deposits.
    /// @param parts     The LB's sealed part sets in part order (chains read,
    ///                  never mutated).
    /// @param partCount Number of part sets (`>= 0`).
    /// @param firingRecordsCanonical Whether the concatenated chains already
    ///        carry the exact canonical content order and may bypass sorting.
    /// @see `checkLocalEncodedMemoryStatic` — the producer.
    /// @see `FiringRecord` — the record shape.
    /// @see `D-117`.
    void ExpressionAnalyzer::applyFiringRecords(Memory& memoryBlock,
        SealedPageSet* const* parts, int32_t partCount,
        bool firingRecordsCanonical)
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
        // The index is contiguous, and one arena allocation holds at most one pool
        // block -- kFiringRecordSortChunk slots. A burst producing more firings than
        // that is sorted in CHUNKS of one block, each on its own, and the chunks are
        // consumed merged below. At or below one chunk this is exactly what it was:
        // a single std::sort over one contiguous index, at the same cost.
        //
        // The chunking exists because the stump split pushed a Peano induction LB
        // past the old hard ceiling: its rule-parts stopped truncating at the
        // submatch cap, so the burst ran to completion and produced 80,147 records.
        // The ceiling was a property of the allocation, never of the merge.
        constexpr int32_t kChunk = ExecutionParameters::kFiringRecordSortChunk;
        const int32_t chunkCount = firingRecordsCanonical
            ? 0 : (total + kChunk - 1) / kChunk;
        assert(chunkCount <= ExecutionParameters::kMaxFiringRecordSortChunks
            && "firing-record sort needs more chunks than the cursor arrays hold - "
               "raise kMaxFiringRecordSortChunks deliberately, never truncate");
        int32_t* chunk[ExecutionParameters::kMaxFiringRecordSortChunks] = { nullptr };
        int32_t chunkLen[ExecutionParameters::kMaxFiringRecordSortChunks] = { 0 };
        int32_t chunkAt[ExecutionParameters::kMaxFiringRecordSortChunks] = { 0 };
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
                    return false;
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
        for (int32_t c = 0; c < chunkCount; ++c) {
            const int32_t lo = c * kChunk;
            const int32_t len = (total - lo < kChunk) ? (total - lo) : kChunk;
            chunk[c] = reinterpret_cast<int32_t*>(gArena.resolve(gArena.alloc(
                len * static_cast<int32_t>(sizeof(int32_t)),
                static_cast<int32_t>(alignof(int32_t)))));
            for (int32_t k = 0; k < len; ++k) chunk[c][k] = lo + k;
            std::sort(chunk[c], chunk[c] + len, [&](int32_t x, int32_t y) {
                return recordLess(*refs[x], *refs[y]);
            });
            chunkLen[c] = len;
        }
        // Pull the next record index in sorted order across the chunks. The
        // comparator is a strict TOTAL order (no two distinct firings tie, I-77), so
        // the merged sequence is unique -- identical to the single std::sort it
        // generalises, and the deposit bytes still cannot depend on input
        // permutation, part count, chunking or sort algorithm. With one chunk this
        // is a walk down chunk[0] and costs no comparison at all.
        int32_t canonicalAt = 0;
        const auto nextIdx = [&]() -> int32_t {
            if (firingRecordsCanonical) {
                assert(canonicalAt < total);
                return canonicalAt++;
            }
            int32_t bestC = -1;
            for (int32_t c = 0; c < chunkCount; ++c) {
                if (chunkAt[c] >= chunkLen[c]) continue;
                if (bestC < 0) { bestC = c; continue; }
                if (recordLess(*refs[chunk[c][chunkAt[c]]],
                               *refs[chunk[bestC][chunkAt[bestC]]]))
                    bestC = c;
            }
            assert(bestC >= 0 && "firing-record merge ran dry before its total");
            return chunk[bestC][chunkAt[bestC]++];
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
                                        rec.levels });
            } else if (!rec.isMarker) {
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
                        rec.doNotDisintegrate, rec.allowOrDisintegration,
                        rec.iteration);
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
    ///   1. insert `record.value` into `overallHashMemory.admissionMap` under
    ///      `record.key` (fresh per-record lookup; create the set if absent);
    ///   2. set `admissionStatusMap[record.key] = false`;
    ///   3. add every non-`marker` argument of `record.key.original` to
    ///      `varsInAdmissionMapKeys`;
    ///   4. fire `revisitRejected2`, depositing any revival cohort on
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
    /// @invariant Drains in append (firing) order.
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
            const int64_t recPk = mintTemplateKey(memoryBlock.templateInterner,
                memoryBlock.nameMap, keyOriginal, keyValidity);

            // (1) admissionMap insert — this drain is the single-threaded
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
            // Ordis-only key gains must not probe the GENERAL rejectedMap
            // (they are cohort-opening demand evidence; their revival seam
            // is revisitRejectedOrdis).
            if (!rec.value.ordisOnly) {
                this->revisitRejected2(keyOriginal, memoryBlock, keyValidity);
            }
            // (6) Ordis revival — parked cohorts wake on EVERY key gain,
            // tagged or untagged (the cohort probe reads both).
            this->revisitRejectedOrdis(keyOriginal, memoryBlock, keyValidity);
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
            // Marker replays are statement-driven — no owning goal, so no
            // goal-carrying payload (the replay's mints nest under scopes the
            // admission machinery already owns, or stay bare on main).
            prepareIntegration(StrSpan(rec.expression), args, n,
                               memoryBlock, StrSpan(rec.validityName), StrSpan());
        }
    }

    // Doxygen at the declaration (prover.hpp).
    void ExpressionAnalyzer::drainAdmissionKeysOrdis2(Memory& memoryBlock)
    {
        const unsigned slot = (g_currentCoreId >= 0)
            ? static_cast<unsigned>(g_currentCoreId)
            : genScratchArenas().slotCount() - 1;
        ScratchArena& gArena = genScratchArenas().forSlot(slot);

        for (const Ordis2DemandRecord& rec : memoryBlock.admissionKeysOrdis2) {
            // FROZEN interner-touch sequence per record (the
            // drainAdmissionKeysAlgebra discipline): (1) mintTemplateKey
            // (templateInterner + NameMap), (2) the RMW blob insert
            // (valueInterner untouched — the value carries rule-interner
            // ids minted at install), (3) the ordis2 wake.
            const ArenaOffset mark = gArena.cursor();
            const int64_t pk = mintTemplateKey(memoryBlock.templateInterner,
                memoryBlock.nameMap, StrSpan(rec.expression),
                StrSpan(rec.validityName));
            insertAdmissionOrdis2IdsBlob(
                memoryBlock.overallHashMemory.admissionMapOrdis2, pk,
                rec.sourceImplId, rec.levels.begin(), rec.levels.size(),
                memoryBlock.ruleInterner, gArena);
            gArena.popTo(mark);

            // The pair's rendezvous (D-267):
            // rejectedMapOrdis2 is keyed by disjunct clean GROUND texts —
            // the demand's own key language — so this keyed probe IS the
            // wake. A parked or filed under a text byte-equal to the
            // demand is un-known and re-deposited; the absorb re-runs the
            // opener, whose route (c) reads the fresh entry above and
            // opens (consuming it). A probe miss (no filed head yet, or a
            // never-parked text) is a defined no-op — the demand entry
            // stands and a later deposit's route-(c) probe covers the
            // demand-first order. The OLD rejectedMapOrdis is deliberately
            // NOT probed here (different key language; a cross-probe would
            // double-wake operator-application disjuncts).
            this->revisitRejectedOrdis2(StrSpan(rec.expression), memoryBlock,
                                        StrSpan(rec.validityName));
        }
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
    void Memory::wipeSubtree(NameId closedVid) {
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
        // The bitmap is sized from the live nameCount at this call seam
        // (one bit per minted id; words * 64 > nameCount strictly, re-checked
        // by the collectClosedSubtreeIds capacity assert) and rides the
        // per-slot gen-scratch BYTE-BUMP tier — a fixed stack array cannot
        // follow MAX_NAME_IDS (1M ids = 125 KB of stack). The ascending id
        // vector rides the same arena's PAGE tier (I-124) and doubles as the
        // step-11 mint order; both share the per-task releaseAll lifetime.
        // Every int-keyed sweep below consults the bitmap; one
        // O(nameCount x depth) walk amortizes the membership cost across
        // all of them.
        const unsigned slot = (ExpressionAnalyzer::g_currentCoreId >= 0)
            ? static_cast<unsigned>(ExpressionAnalyzer::g_currentCoreId)
            : genScratchArenas().slotCount() - 1;
        ScratchArena& gArena = genScratchArenas().forSlot(slot);
        DirtyState ascDirty = DirtyState::Clean;
        PagedVector<NameId> closedAsc(&gArena, &ascDirty);
        const int32_t nameHighWater = nameMap.nameCount();
        const int32_t closedWords = nameHighWater / 64 + 1;
        uint64_t* closedBits = reinterpret_cast<uint64_t*>(gArena.resolve(
            gArena.alloc(closedWords * static_cast<int32_t>(sizeof(uint64_t)),
                         static_cast<int32_t>(alignof(uint64_t)))));
        nameMap.collectClosedSubtreeIds(closedVid, closedBits, closedWords,
                                        closedAsc);
        const auto inClosedBit = [&](int32_t v) -> bool {
            // EXACT twin of the former closedIds.count(x): x can arrive
            // NEGATIVE (e.g. the low half of a GLOBAL mailIn id at the
            // routing-mail filter) — count(negative) was false; the
            // `v >= 1` guard short-circuits before the bitmap index, so
            // the index is always a non-negative NameId (no uint16 wrap —
            // a wrap would alias ids >= 65536 into the wrong bit once the
            // ceiling rises).
            return v >= 1 && v <= nameHighWater
                && ((closedBits[v >> 6] >> (v & 63)) & 1ull) != 0;
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
        intLocalEncodedStatementsSet.eraseIf([&](int64_t k) {
            return inClosedBit(Codec<StatementKey>::decode(k).validity);
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
        equivalenceClassesMap.eraseBlobIf([&](NameId vId) {
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
        // The carrier index: a carrier registered at a closed scope leaves
        // with its scope (its rules leave through wipeHashMem below; the
        // owner maps keep their owners — I-49).
        compactExpansions.eraseBlobIf([&](int64_t k) {
            return inClosedBit(Codec<StatementKey>::decode(k).validity);
        });
        // Packed template-space twins — same low-16-bits validity
        // predicate as the other packed sweeps.
        auto filterPackedSet = [&](ColdHashSet<PodKeyStore<int64_t>>& s) {
            s.eraseIf([&](int64_t k) {
                return inClosedBit(Codec<StatementKey>::decode(k).validity);
            });
        };
        filterPackedSet(integrationPrepared);
        filterPackedSet(integrationPreparedMarker);

        // Packed twin of weakVariables — same low-32-bits validity
        // predicate as the intKnownStatements sweep in step 6.
        intWeakVariables.eraseIf([&](int64_t k) {
            return inClosedBit(Codec<StatementKey>::decode(k).validity);
        });

        // ---- 9. Mail statements (in / out / internal) ----
        // Routing mail is main-only (I-26), so a subtree wipe removes nothing.
        // Walk each channel explicitly and assert the id-space-specific scope
        // contract rather than comparing unrelated interner ids.
        auto filterRoutingMailStatements = [&](RoutingColdMail& m) {
            m.filterStatements(
                [&](const IntMailStatementKey& k) {
                    assert(mailInterner().decodeString(k.validityId) == "main"
                        && "routing mailIn must be main-scope");
                    return false;
                });
        };
        filterRoutingMailStatements(mailIn);
        mailOut.filterStatements([&](const IntMailStatementKey& k) {
            assert(mailOutInterner.decode(k.validityId) == "main"
                && "routing mailOut must be main-scope");
            return false;
        });
        mailOutPending = !mailOut.empty();
        mailOutLiveBytes = mailOut.liveBytes();
        // sameIterationInternalMail is COLD (I-102): the
        // same scope-targeted erase via ColdMail::filterStatements. Only
        // statements is swept (origins / disintegrationSignals untouched),
        // matching the former heap filter. The id-form key carries the
        // statement's NameMap validityId, so the closed-scope test is the same
        // bitmap membership the intStatementLevelsMap / intKnownStatements
        // sweeps use (no decode).
        sameIterationInternalMail.filterStatements(
            [&](const IntMailStatementKey& k) {
                return inClosedBit(static_cast<NameId>(k.validityId));
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

            // The two normalized-key indexes and the remaining-args index are
            // NOT scope-wiped (D-303, I-49): they carry no owner record, so
            // there is nothing to filter, and a stale key is a sound
            // over-approximation — its request reaches the firing site, whose
            // encodedMap lookup (wiped in 10a) misses and skips it.

            // Cold admission/rejection containers
            // (D-172, D-173)
            // use the compacting / run-aware cold erase with the same low-32-bits
            // closed-scope predicate (the packed template key is int64).
            const auto coldScopeWipe = [&](int64_t k) {
                return inClosedBit(Codec<StatementKey>::decode(k).validity);
            };
            hm.admissionMap.eraseBlobIf(coldScopeWipe);
            hm.admissionStatusMap.eraseIf(coldScopeWipe);
            hm.rejectedMap.eraseBlobIf(coldScopeWipe);
            hm.admissionMapIntegration.eraseBlobIf(coldScopeWipe);
            hm.rejectedMapIntegration.eraseBlobIf(coldScopeWipe);
            // Parked or-cohorts of wiped subproof scopes must not leak: the
            // key's validity half carries the cohort parent, so the same
            // closed-bit predicate covers the ordis park.
            hm.rejectedMapOrdis.eraseBlobIf(coldScopeWipe);
            // Demand entries of wiped subproof scopes must not survive to
            // open cohorts in a dead scope — same closed-bit predicate on
            // the key's validity half (D-267).
            hm.admissionMapOrdis2.eraseBlobIf(coldScopeWipe);
            // The ordis2 park index files under the same cohort-parent
            // validity as rejectedMapOrdis — same closed-bit predicate
            // (D-267).
            hm.rejectedMapOrdis2.eraseBlobIf(coldScopeWipe);
            hm.revisitInProgress.eraseIf(coldScopeWipe);
            hm.ordisRevisitInProgress.eraseIf(coldScopeWipe);
            hm.ordis2RevisitInProgress.eraseIf(coldScopeWipe);
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

    /// @brief Return the process-wide, never-deloaded global mail string
    ///        interner.
    ///
    /// @details
    /// The table provides one shared id space for routing mail retained in the
    /// current delivery-history window. Its arena stays attached to
    /// `mailMemory`; rolling grids may clear the table at a quiescent
    /// post-phase-3 seam through `resetMailInterner`, while dormant grids retain
    /// it for the complete execution batch.
    ///
    /// @return The singleton global mail interner.
    /// @invariant Mutation is single-threaded; parallel proof phases only read
    ///            the table.
    /// @see resetMailInterner, MailLog::retireDeliveredBatches.
    ColdStringTable& mailInterner() {
        static LbArena arena{ &mailMemory() };
        static DirtyState dirty = DirtyState::Clean;
        static ColdStringTable table{ &arena, &dirty };
        return table;
    }

    /// @brief Clear the global mail interner after a delivered history window
    ///        has been retired.
    ///
    /// @details
    /// Invalidates every global mail id and returns the table's pages to its
    /// mail-pool arena for reuse. The arena keeps its physical blocks, bounding
    /// the interner by the largest single delivery window instead of cumulative
    /// windows. The caller must first prove that no retained `MailLog` blob,
    /// routing `mailIn`, or other live object can decode an old id.
    ///
    /// @return Nothing.
    /// @invariant Called single-threaded after phase 3 joins and after
    ///            `MailLog::retireDeliveredBatches`, before the next commit.
    /// @see mailInterner, ExpressionAnalyzer::proveKernel.
    void resetMailInterner() {
        mailInterner().resetToFresh();
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
            assert(deloadKind == DeloadKind::Canonical
                && "v3 dumpStaticContainers skip-clean over a raw base — the "
                   "v3 path only handles discharge, which is never Clean");
            deloadStats().evictSkippedClean.fetch_add(
                1, std::memory_order_relaxed);
        }
        else {
            // Every write below is the v3 canonical stream.
            deloadKind = DeloadKind::Canonical;
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
                deloadStats().tailDumpCount.fetch_add(
                    1, std::memory_order_relaxed);
            }
            else {
                deloadFiles = lbdeload::dumpLbMemory(
                    lbMemory, deloadChain, deloadOrdinal, directory,
                    staticMemory().blockBytes());
                deloadStats().fullDumpCount.fetch_add(
                    1, std::memory_order_relaxed);
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
        // The derived reverse membership indexes are NOT enrolled in
        // visitContainers, so the walk above did not free their throw-away pages.
        // Free them now, while the arena is still resident, BEFORE releaseAll
        // returns the blocks -- else their stale vids would double-free at the
        // next canonical reload's rebuild (I-154). The raw
        // release path (releaseStaticBlocksRaw) deliberately skips this: the raw
        // image carries these pages verbatim.
        lbMemory.overallHashMemory.remainingArgsReverseIndex.clear();
        lbMemory.localHashMemory.remainingArgsReverseIndex.clear();
        lbMemory.localHashMemoryDelta.remainingArgsReverseIndex.clear();
        lbMemory.workingMemory.remainingArgsReverseIndex.clear();
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

    /// @brief Write-through v4 RAW dump — the near-memcpy eviction dump (see
    ///        the declaration for the full contract).
    ///
    /// @details
    /// Captures `deloadedCounts` (the only legal cold metadata) BEFORE the dump
    /// — the raw image never releases the containers, so the counts are read
    /// straight off the live `size()`s. Skip-clean is a defined cache-hit
    /// (Rule 19), not a fallback. The raw path writes ONE file and no tails, so
    /// it clears the v3 tail bookkeeping.
    ///
    /// @param directory The deload directory (production: `.deload`).
    void Memory::dumpStaticContainersRaw(const std::string& directory) {
        assert(lbMemory.manager.resident()
            && "dumpStaticContainersRaw on a deloaded LB");
        deloadChain = buildLbChainString(*this);
        ensureDeloadOrdinal();
        const bool useExtent = staticMemory().useExtent();
        // A live slab exists (skip-clean / in-place eligible) when this LB has
        // a raw base. On the extent path the base is a slab in the current
        // epoch; on the named path it is a file in deloadFiles.
        const bool haveExtentSlab = useExtent
            && rawExtentOffset_ >= 0
            && rawExtentEpoch_ == staticMemory().extentEpoch()
            && deloadKind == DeloadKind::Raw;
        const bool haveRawBase = useExtent
            ? haveExtentSlab
            : (!deloadFiles.empty() && deloadKind == DeloadKind::Raw);
        if (lbMemory.dirty == DirtyState::Clean && haveRawBase) {
            // Skip: the on-disk raw image already equals the in-memory content.
            deloadStats().evictSkippedClean.fetch_add(
                1, std::memory_order_relaxed);
        }
        else {
            // Cold metadata: per-container totals in tag order (the only legal
            // read while the LB is deloaded, I-111). Captured off the live
            // sizes — the raw dump releases nothing. Written DIRECTLY into
            // deloadedCounts (clear + push_back reuses its capacity after the
            // first dump): this is the eviction hot loop, where per-operation
            // fixed costs dominate on small LBs, so a per-dump heap temporary
            // is not acceptable. The v3 path keeps its temporary (discharge
            // only — off the hot loop, and its tail decision reads BOTH the
            // old and the new counts at once).
            deloadedCounts.clear();
            lbMemory.visitContainers(
                [this](LbMemory::ContainerTag, const auto& container) {
                    deloadedCounts.push_back(container.size());
                });
            if (useExtent) {
                // Extent path: place the image at the LB's slab. A stable /
                // shrinking size overwrites IN PLACE (no allocator touch — the
                // near-memcpy hot path); a first dump OR a growth past the slab
                // class allocates a fresh slab (freeing the old one first).
                const int64_t need =
                    lbdeload::rawImageBytesFor(lbMemory, deloadChain);
                const bool haveSlab = rawExtentOffset_ >= 0
                    && rawExtentEpoch_ == staticMemory().extentEpoch();
                if (!haveSlab || need > rawExtentClassBytes_) {
                    if (haveSlab)
                        staticMemory().freeExtentSlab(deloadOrdinal,
                                                      rawExtentOffset_,
                                                      rawExtentClassBytes_);
                    const SlabAllocation slab =
                        staticMemory().allocateExtentSlab(deloadOrdinal, need);
                    rawExtentOffset_ = slab.offset;
                    rawExtentClassBytes_ =
                        static_cast<int32_t>(slab.classBytes);
                    rawExtentEpoch_ = staticMemory().extentEpoch();
                }
                lastRawImageBytes_ = lbdeload::dumpLbMemoryRawAt(
                    lbMemory, deloadChain, deloadOrdinal,
                    staticMemory().extentFile(), rawExtentOffset_,
                    rawExtentClassBytes_);
                // extentLiveBytes delta: this LB's occupancy replaces its prior.
                const int64_t image =
                    lbdeload::rawImageBytesFor(lbMemory, deloadChain);
                staticMemory().addExtentLive(image - lastExtentLiveBytes_);
                lastExtentLiveBytes_ = image;
                // The extent path does not use deloadFiles (reload seeks to
                // rawExtentOffset_); keep it empty so a stale name is never read.
                deloadFiles.clear();
            } else {
                lastRawImageBytes_ = lbdeload::dumpLbMemoryRaw(
                    lbMemory, deloadChain, deloadOrdinal, directory);
                deloadFiles.clear();
                deloadFiles.push_back(lbdeload::rawFileName(deloadOrdinal));
            }
            deloadKind = DeloadKind::Raw;
            // The raw path writes no tails — reset the v3 tail bookkeeping so a
            // later v3 discharge dump of this LB starts a clean base.
            deloadTailCount = 0;
            deloadTailRows = 0;
            deloadBaseRows = 0;
            deloadStats().fullDumpCount.fetch_add(
                1, std::memory_order_relaxed);
        }
        // RAM == disk holds from here until the next mutation.
        lbMemory.dirty = DirtyState::Clean;
    }

    /// @brief Return the LB's blocks WITHOUT the container-release walk — the
    ///        raw eviction's release step (see the declaration for the
    ///        contract).
    ///
    /// @details
    /// NO `container.release()` walk (the near-memcpy win): the raw image
    /// preserves the vids, so the container bookkeeping must SURVIVE for the
    /// raw reload to rebind it. Just return the blocks and flag deloaded; the
    /// dirty state is already `Clean` from the dump and stays `Clean` (no
    /// mutator ran). The teardown residency branch in `~PagedVector` /
    /// `~PagedHashIndex` makes a destruction on the now-deloaded arena a
    /// no-op; a LIVE `clear()` there still asserts.
    void Memory::releaseStaticBlocksRaw() {
        assert(lbMemory.manager.resident()
            && "releaseStaticBlocksRaw on an already deloaded LB");
        // The fresh-raw-image evidence is the slab on the extent path, the
        // file set on the named path.
        assert((staticMemory().useExtent()
                    ? rawExtentOffset_ >= 0
                    : !deloadFiles.empty())
                && deloadKind == DeloadKind::Raw
            && "releaseStaticBlocksRaw without a fresh raw image");
        assert(lbMemory.dirty == DirtyState::Clean
            && "releaseStaticBlocksRaw with a stale image - dump first");
        lbMemory.manager.releaseAll();
        lbMemory.manager.markDeloaded();
    }

    /// @brief Release through the path matching the recorded on-disk format —
    ///        the release-side mirror of `reloadFromImage`'s dispatch (see the
    ///        declaration for the full contract).
    void Memory::releaseStaticBlocksDispatch() {
        if (deloadKind == DeloadKind::Raw)
            releaseStaticBlocksRaw();
        else
            releaseStaticBlocks();
    }

    /// @brief Raw dump + raw release in one call (see the declaration).
    ///
    /// @param directory The deload directory (production: `.deload`).
    void Memory::deloadStaticContainersRaw(const std::string& directory) {
        dumpStaticContainersRaw(directory);
        releaseStaticBlocksRaw();
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
        // A raw-evicted LB being discharged flips Raw->Canonical (the next dump
        // is v3) and never raw-reloads again, so its extent slab is returned to
        // the allocator free-list — the one new line the extent path adds to
        // discharge (D-195 §4). Guarded on a live slab
        // in the current epoch (a purged prior-batch slab was already recycled).
        if (staticMemory().useExtent() && rawExtentOffset_ >= 0
            && rawExtentEpoch_ == staticMemory().extentEpoch()) {
            staticMemory().freeExtentSlab(deloadOrdinal, rawExtentOffset_,
                                          rawExtentClassBytes_);
            staticMemory().addExtentLive(-lastExtentLiveBytes_);
            lastExtentLiveBytes_ = 0;
            rawExtentOffset_ = -1;
            rawExtentClassBytes_ = 0;
            rawExtentEpoch_ = -1;
        }
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
        pendingDisprovedGoals.resetToFresh();
        pendingDeadOrBranches.resetToFresh();
        orRetiredDisjuncts.resetToFresh();
        pendingOrReleases.resetToFresh();
        orStarterPick.resetToFresh();
        processedOrLedger.resetToFresh();
        orLiveBranches.resetToFresh();
        frozenOrBranches.resetToFresh();
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
        // No request generation follows a read-only reload — every consumer of
        // this door reads origin history only (the chapter export's walk and
        // `repairTierCitationOrigins`). The reverse membership indexes exist
        // solely for the firing-check candidate probe, so rebuilding them here
        // is pure cost; on the contradiction-twin probe it dominated the whole
        // chapter export. They are marked unbuilt instead, which turns any
        // future reader on this path into an assert rather than a silent
        // empty-candidate answer.
        reloadFromImage(directory, /*rebuildDerivedIndexes=*/false);
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
    /// @param rebuildDerivedIndexes When true (the default, and every
    ///        work-path caller), the canonical branch re-derives the four
    ///        `ReverseArgsIndex` membership indexes, which the canonical
    ///        stream deliberately omits (`I-117` / `I-154`). When false — the
    ///        read-only door — the rebuild is skipped and each index is marked
    ///        unbuilt, because no request generation follows. The RAW branch
    ///        ignores this: its whole-arena image restores the indexes
    ///        verbatim, so there is nothing to rebuild or to skip.
    void Memory::reloadFromImage(const std::string& directory,
                                 bool rebuildDerivedIndexes) {
        // The image reference is the extent slab on the raw-extent path, the
        // recorded file set otherwise.
        const bool extentRaw = deloadKind == DeloadKind::Raw
            && staticMemory().useExtent();
        assert((extentRaw ? rawExtentOffset_ >= 0 : !deloadFiles.empty())
            && "reload on a deloaded LB with no recorded image reference");
        assert(buildLbChainString(*this) == deloadChain
            && "LB moved in the tree between deload and reload");
        lbMemory.manager.markResident();
        // Dispatch on the recorded format of the LAST dump. Raw = the v4
        // near-memcpy arena image (one file, no rebuild); Canonical = the v3
        // element stream (discharge / export). Both loaders assert their own
        // header version, so a field/file mismatch is caught (Rule 19).
        if (deloadKind == DeloadKind::Raw) {
            if (extentRaw) {
                // Extent path: seek to the LB's slab in the one open file.
                lbdeload::loadLbMemoryRawAt(
                    lbMemory, deloadChain, deloadOrdinal,
                    staticMemory().extentFile(), rawExtentOffset_);
            } else {
                assert(deloadFiles.size() == 1
                    && "raw image is a single file");
                lbdeload::loadLbMemoryRaw(lbMemory, deloadChain, deloadOrdinal,
                                          deloadFiles.front(), directory);
            }
        }
        else {
            // The four HashMemory instances reload through
            // lbMemory.visitContainers (D-147), so loadLbMemory rebuilds their
            // tags 51..450 alongside LbMemory's own -- no extra-column list.
            lbdeload::loadLbMemory(lbMemory, deloadChain, deloadFiles,
                                   directory);
            // The derived reverse membership indexes are NOT in the canonical
            // deload stream (I-117 pattern), so a canonical reload restores an
            // empty index -- rebuild each from its now-resident forward map, the
            // twin of the cold-map family's KeysView -> rebuildIndex reload hook.
            // The RAW branch above needs no rebuild (the whole-arena image
            // restored these pages verbatim, I-154).
            //
            // A read-only reload skips this: it is proportional to hash-memory
            // size and only request generation ever reads the result.
            if (!rebuildDerivedIndexes) {
                lbMemory.overallHashMemory.remainingArgsReverseIndex.markUnbuilt();
                lbMemory.localHashMemory.remainingArgsReverseIndex.markUnbuilt();
                lbMemory.localHashMemoryDelta.remainingArgsReverseIndex.markUnbuilt();
                lbMemory.workingMemory.remainingArgsReverseIndex.markUnbuilt();
            } else {
            const unsigned revSlot =
                (ExpressionAnalyzer::g_currentCoreId >= 0)
                    ? static_cast<unsigned>(ExpressionAnalyzer::g_currentCoreId)
                    : genScratchArenas().slotCount() - 1;
            ScratchArena& revArena = genScratchArenas().forSlot(revSlot);
            const ArenaOffset revMark = revArena.cursor();
            lbMemory.overallHashMemory.remainingArgsReverseIndex.rebuildReverseIndex(
                lbMemory.overallHashMemory.remainingArgsNormalizedEncodedMap,
                revArena);
            lbMemory.localHashMemory.remainingArgsReverseIndex.rebuildReverseIndex(
                lbMemory.localHashMemory.remainingArgsNormalizedEncodedMap,
                revArena);
            lbMemory.localHashMemoryDelta.remainingArgsReverseIndex.rebuildReverseIndex(
                lbMemory.localHashMemoryDelta.remainingArgsNormalizedEncodedMap,
                revArena);
            lbMemory.workingMemory.remainingArgsReverseIndex.rebuildReverseIndex(
                lbMemory.workingMemory.remainingArgsNormalizedEncodedMap,
                revArena);
            revArena.popTo(revMark);
            }
        }
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

    // Explicit instantiation of the request-generator member template for the
    // streaming consumer (the only consumer; see BurstSink in prover.hpp).
    template void ExpressionAnalyzer::generateEncodedRequestsStatic<BurstSink>(
        const Memory&, const HashMemory&, const MandatoryTerm*, NameId,
        const SplitStumpRef&, unsigned, BurstSink&);

    // ---- NameMap::encodePush (span overload) --------------------------------
    // Out-of-line: the heap-free canonical build reaches the string scratch
    // registry (scratchArenas()) and the worker-slot id
    // (ExpressionAnalyzer::g_currentCoreId), neither visible to an inline header
    // body. Byte-identical to the former inline heap-std::string body — same
    // canonical bytes (decodeView == decode), same dedup, same subs-then-names
    // mint order.
    NameId NameMap::encodePush(NameId parentId, const StrSpan& payload) {
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

        const NameId existing = lookup(canonical);
        if (existing != 0) return existing;

        // Mint order (subs payload, then names canonical) preserved.
        const NameId subId = encodeSub(payload);
        const NameId newId = mintName(canonical);
        nodes->push_back(ValidityNode{parentId, subId});
        return newId;
    }

} // namespace gl
