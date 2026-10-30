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
#include "hash_rule_classifier.hpp"
#include "../compiler/definition_expansion.hpp"
#include "../proof/worker_context.hpp"
namespace gl {
/// @brief Bind compiled definitions used to classify rule shapes.
/// @details The registry outlives this reader; construction performs no analysis.
/// @param definitions Read-only compiled definitions and configurations.
/// @return Constructed rule classifier.
HashRuleClassifier::HashRuleClassifier(const CompiledDefinitionRegistry& definitions) : definitions(definitions) {}

/// @brief Whether a rule install has the or-INTRO shape — a single
        ///        premise that is one flattened leaf of its or-category head.
        ///
        /// @details
        /// The install-time detector behind the `LocalMemoryValue::
        /// disintegrationAllowed` derivation
        /// (D-241). An or-intro rule —
        /// `(>[bound](D_k)(or…))`, emitted per flattened leaf by
        /// `disintegrateExprCore2`'s implication branch
        /// (D-237) — states a disjunction as a FACT from
        /// one true disjunct; its fired head is consumed flat (the
        /// description's membership-intro clause, the K rules), so it must
        /// never re-disintegrate into `_ordis_` branch scopes. Detection is
        /// by shape, not emission provenance, so a status-3 mail-recovered
        /// intro rule re-marks at its receiver-side reinstall exactly like a
        /// local install: exactly one premise; the head a positive compact
        /// whose compiled entity has category `or`; the premise byte-equal to
        /// one leaf of the head instance, where the leaves come from the same
        /// signature-to-instance substitution recursion the emission uses
        /// (recursing into or-category elements only, I-166 flattening).
        ///
        /// A compiled-map miss on the head is a defined negative (atomic or
        /// uncompiled head — not an or-intro), mirroring the emission site's
        /// probe contract.
        ///
        /// @param keyRun   The ordered implication-chain premises.
        /// @param keyN     Number of premises; only `keyN == 1` can match.
        /// @param headSpan The head (implication consequent).
        /// @return Whether the install is or-intro-shaped.
        /// @invariant Heap-free: compiled-map read-out probes plus scratch-
        ///            arena string builds under a per-call `ScratchScope`
        ///            (Rule 28).
        /// @see `disintegrateExprCore2` — the intro emission;
        ///      `addToHashMemory` — the sole production caller;
        ///      `LocalMemoryValue::disintegrationAllowed` — the derived flag.
bool HashRuleClassifier::isOrIntroInstall(const StrSpan* keyRun,
                                              int32_t keyN,
                                              StrSpan headSpan) {
        if (keyN != 1) return false;
        if (headSpan.len < 1 || headSpan.ptr[0] != '(') return false;
        const LogicalEntity* orLe =
            definitions.compiledEntity(extractExpressionUniversalSpan(headSpan));
        if (orLe == nullptr || orLe->category != "or") return false;

        const unsigned slot = (gl::WorkerContext::coreId() >= 0)
            ? static_cast<unsigned>(gl::WorkerContext::coreId())
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
                    ? definitions.compiledEntity(extractExpressionUniversalSpan(substSpan))
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

/// @brief Scratch-arena twin of `flattenRegistryOrLeaves` — flatten
        /// a compiled or-category entity under an arbitrary instance, leaves
        /// as spans on the caller's arena.
        ///
        /// @details
        /// The heap-free form for the statified paths (Rule 28): the same
        /// signature-to-instance substitution walk (recursing into
        /// or-category elements only, I-166), each substituted element built
        /// with `replaceKeysScratch` on @p sArena. Passing the entity's own
        /// signature as @p instance yields the registry-term leaves (the
        /// heap twin's output); passing an instantiated cohort signature
        /// yields instance-term leaves. Output spans alias @p sArena (and,
        /// for substitution-free elements, the entity's element strings) —
        /// valid until the caller's enclosing scope rewinds.
        ///
        /// @param orLe     The or-category compiled entity to flatten.
        /// @param instance The root instance (`(name[args])` form) whose
        ///        args substitute the entity's signature args.
        /// @param out      Caller span array receiving the ordered leaves.
        /// @param cap      Capacity of @p out; overflow asserts before a
        ///        write (Rule 19).
        /// @param sArena   The caller's per-slot string-scratch arena.
        /// @return The number of leaves written (at least two, asserted).
        /// @invariant Byte-identical leaf sequence to the heap twin under
        ///            the identity instance (twin-tested).
        /// @see `flattenRegistryOrLeaves` — the heap twin (barrier seams);
        ///      `isSubsetExclusionInstall`, `consumeOrLeavesCohort` — the
        ///      statified consumers.
int32_t HashRuleClassifier::flattenRegistryOrLeavesScratch(
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
                    ? definitions.compiledEntity(extractExpressionUniversalSpan(substSpan))
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

/// @brief Whether a rule install has the SUBSET-EXCLUSION shape —
        ///        j negated disjuncts implying the or of the rest of a
        ///        registered parent or (j = 1..k−2).
        ///
        /// @details
        /// The install-time detector behind `LocalMemoryValue::
        /// subsetExclusion` (I-184). A
        /// subset-exclusion rule — `!D_i1 & .. & !D_ij -> or(D_rest)` with
        /// a genuine reduced (k−j)-ary or-compact head, emitted at flat or
        /// consumption — must fire PARK-FIRST: its head never carries the
        /// route-(b) open signal. Detection is by shape, not emission
        /// provenance (the D-241 discipline `isOrIntroInstall` set), so a
        /// status-3 mail-recovered rule re-marks at its receiver-side
        /// reinstall.
        ///
        /// Shape: 1..k−2 premises; the head a compact whose compiled
        /// entity has category `or`; and — the registry-parent requirement
        /// that keeps proved `!X -> or(..)`-shaped theorems untagged — some
        /// registered or's FLATTENED leaf list (same flattened-cohort
        /// discipline as the emission, I-166) embeds the head's
        /// instance-substituted leaves as an order-preserving subsequence
        /// while its remaining elements match the premises' negations
        /// (double-negation cancelling, I-175) as a multiset, under a
        /// consistent INJECTIVE signature-token-to-instance-arg binding.
        ///
        /// Premise ORDER is deliberately ignored: the mail-compact
        /// round-trip stores only the first-seen body and dedups whole
        /// name-sorted permutation families (I-52), so a reinstalled
        /// rule's premises may arrive permuted relative to emission order —
        /// an order-sensitive probe would silently untag exactly the
        /// mail-recovered rules the D-241 re-detection exists for.
        ///
        /// Known accepted edge: a DEGENERATE instance (repeated concrete
        /// args) violates binding injectivity and misses its parent — the
        /// rule stays untagged; its fired degenerate head is safe to
        /// flat-consume (D-270).
        ///
        /// @param keyRun   The implication-chain premises (any order).
        /// @param keyN     Number of premises; 1..kMaxReducedOrLeaves−2.
        /// @param headSpan The head (implication consequent).
        /// @return Whether the install is subset-exclusion-shaped.
        /// @invariant Heap-free: compiled-map read-out scans plus
        ///            scratch-arena builds under per-call `ScratchScope`s
        ///            (Rule 28); the embed walk is capped by
        ///            `kSubsetExclusionEmbedCap` (Rule 19 tripwire).
        /// @see `consumeOrLeavesCohort` — the emission;
        ///      `addToHashMemory` — the sole production caller;
        ///      `forEachCompiledOr` — the registry scan fence;
        ///      `LocalMemoryValue::subsetExclusion` — the derived flag.
bool HashRuleClassifier::isSubsetExclusionInstall(const StrSpan* keyRun,
                                                      int32_t keyN,
                                                      StrSpan headSpan) {
        if (keyN < 1
            || keyN > ExecutionParameters::kMaxReducedOrLeaves - 2) {
            return false;
        }
        if (headSpan.len < 1 || headSpan.ptr[0] != '(') return false;
        const LogicalEntity* redLe =
            definitions.compiledEntity(extractExpressionUniversalSpan(headSpan));
        if (redLe == nullptr || redLe->category != "or") return false;

        const unsigned slot = (gl::WorkerContext::coreId() >= 0)
            ? static_cast<unsigned>(gl::WorkerContext::coreId())
            : scratchArenas().slotCount() - 1;
        ScratchArena& sArena = scratchArenas().forSlot(slot);
        ScratchScope scope(sArena);

        // The excluded-disjunct candidates: negate each premise with
        // double-negation cancellation — the emission builds premises as
        // DefinitionExpansion::negateScratch(leaf), so negating recovers the TRUE-polarity leaf
        // whatever its sign (I-175). Premise ORDER is deliberately ignored:
        // the mail-compact round-trip stores only the first-seen body and
        // dedups whole name-sorted permutation families (I-52), so a
        // reinstalled rule's premises may arrive in any order.
        StrSpan dSpans[ExecutionParameters::kMaxReducedOrLeaves];
        for (int32_t p = 0; p < keyN; ++p)
            dSpans[p] = DefinitionExpansion::negateScratch(sArena, keyRun[p]);

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
        definitions.forEachCompiledOr([&](const std::string& name,
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
}
