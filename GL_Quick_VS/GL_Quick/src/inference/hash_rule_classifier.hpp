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
#include "../compiler/compiled_definition_registry.hpp"
namespace gl {
/// @brief Classify OR rule shapes and enumerate their instantiated leaves.
/// @details All operations read the compiled registry and use caller-scoped scratch.
class HashRuleClassifier {
public:
/// @brief Bind compiled definitions used to classify rule shapes.
/// @details The registry outlives this reader; construction performs no analysis.
/// @param definitions Read-only compiled definitions and configurations.
/// @return Constructed rule classifier.
HashRuleClassifier(const CompiledDefinitionRegistry& definitions);

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
bool isOrIntroInstall(const StrSpan* keyRun, int32_t keyN,
                              StrSpan headSpan);

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
int32_t flattenRegistryOrLeavesScratch(const LogicalEntity& orLe,
                                               StrSpan instance,
                                               StrSpan* out, int32_t cap,
                                               ScratchArena& sArena);

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
bool isSubsetExclusionInstall(const StrSpan* keyRun, int32_t keyN,
                                      StrSpan headSpan);
private:
const CompiledDefinitionRegistry& definitions;
};
}
