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
#include "copy_ref.hpp"
#include "../parameters.hpp"
namespace gl {
/// @brief Expand an implication into its permitted variable-binding copies.
/// @details Preserves the caller-owned scratch storage and ordered output contract.
class ImplicationMultiplier {
public:
/// @brief Bind definition readers and the existing multiplication settings.
/// @details References outlive the algorithm; construction performs no expansion.
/// @param definitions Configuration and compiled-definition readers.
/// @param parameters Existing multiplication flags and partition limit.
/// @param ceFilteringActive Whether the counterexample phase is currently active.
/// @return Constructed implication multiplier.
ImplicationMultiplier(const CompiledDefinitionRegistry& definitions, const ProverParameters& parameters, const bool& ceFilteringActive);

/// @brief Expand an implication into all variable-renamed copies needed
/// to cover every meaningful binding partition.
///
/// @details
/// Used by `addToHashMemory` (Path 1) to produce the per-copy
/// `(key, value)` pairs that get installed in the hash engine. For an
/// implication `(>[v0,v1](p[v0])(q[v0,v1]))`, multiplyImplication
/// generates the family of permutations / partition variants whose
/// renormalized keys differ. The first copy is the input as-is; later
/// copies carry origin records under the `"multiplied from"` tag so the
/// verifier can chain the multiplications back to the original.
///
/// **Per [I-24](../../docs/agentic_swdd/30_invariants.md#i-24)**, multiplyImplication
/// must NOT equate two distinct free `u_*` anchor parameters — they are
/// the pinned anchor slots and cannot be unified by partition renaming.
/// The function asserts on attempts to do so; a firing assert here means
/// the input implication had a malformed anchor binding.
///
/// @param implication Source implication text.
/// @return Vector of all multiplied copies. Always non-empty (at least
///         the input itself). Order is deterministic.
/// @invariant [I-24](../../docs/agentic_swdd/30_invariants.md#i-24) — no
///            free-anchor-param equation.
/// @see `addToHashMemory` — primary consumer (Path 1 head LMV inserts).
std::vector<std::string> multiplyImplication(const std::string& implication);

/// @brief Span form of @ref multiplyImplication — emit each surviving
        ///        multiplied copy as a `CopyRef` into a caller-owned index, 0%
        ///        heap.
        ///
        /// @details
        /// The production path (the `std::vector<std::string>` overload is the
        /// retained heap oracle). Byte-for-byte identical emitted copy SEQUENCE
        /// (bytes + order) to the oracle: same `disintegrateImplicationSpans`
        /// var/expr collection, the `numericLessSpan` sort (the `numericLess`
        /// twin), the `generateSetPartitions` StrSpan-run overload, the iterated
        /// `replaceArgSpanScratch` word-boundary substitution (the
        /// `replaceArgInString` twin), `deduplicateBoundVarsScratch`, the
        /// trivial-equality-head skip, and the content-dedup. Vars/exprs are
        /// span/offset references into the caller-stable @p implication; the
        /// dedup rides a gen-scratch `ColdHashSet`; each surviving copy is built
        /// on @p outStrArena's string tier and emitted as a `CopyRef` (the
        /// dropped partitions' scratch is reclaimed per partition). The two
        /// early-outs (`!allow_multiplication` / `<= 1` one-typed var) emit a
        /// single copy of @p implication.
        ///
        /// @param implication Source implication text (caller-stable for the call).
        /// @param outStrArena Caller string-tier arena receiving the surviving
        ///                    copies' bytes; the caller keeps it live (a
        ///                    `ScratchScope`) across its read of @p out.
        /// @param out         Caller-owned page-tier index; receives one `CopyRef`
        ///                    per emitted copy in emission order.
        /// @invariant [I-24](../../docs/agentic_swdd/30_invariants.md#i-24) — no
        ///            free-anchor-param equation (the `hasDoubleU` guard).
        /// @invariant Emitted copy sequence byte-identical to the heap oracle
        ///            (twin test `multiply_implication_span_matches_heap`).
        /// @see multiplyImplication(const std::string&) — the retained heap oracle;
        ///      replaceArgSpanScratch, numericLessSpan, generateSetPartitions.
/// @return Nothing.
void multiplyImplication(StrSpan implication, ScratchArena& outStrArena,
            PagedVector<CopyRef>& out);
private:
/// @brief Compare argument labels in numeric binding order.
/// @details Numeric labels use their values and other labels retain string order.
/// @param a First argument label.
/// @param b Second argument label.
/// @return Whether a precedes b in the binding order.
static bool numericLess(const std::string& a, const std::string& b);
const CompiledDefinitionRegistry& definitions;
const ProverParameters& parameters;
const bool& ceFilteringActive;
};
}
