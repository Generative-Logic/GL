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
#include "compiled_definition_registry.hpp"

namespace gl {
class CompiledDefinitionRegistry;
class CompilationReuse;
class StructuralCompiler;
class SupportingRuleCompiler;

/// @brief Compile OR closure, implications and symmetry metadata.
/// @details Composes through definition interfaces without retaining a prover.
class OrRuleCompiler {
public:
/// @brief Bind the cooperating compilation components.
/// @details Construction stores references only; all bound objects outlive this component.
/// @param definitions Borrowed definitions component.
/// @param compilationReuse Borrowed compilationReuse component.
/// @param structuralCompiler Borrowed structuralCompiler component.
/// @param supportingRuleCompiler Borrowed supportingRuleCompiler component.
/// @return Constructed compilation component.
OrRuleCompiler(CompiledDefinitionRegistry& definitions, CompilationReuse& compilationReuse, StructuralCompiler& structuralCompiler, SupportingRuleCompiler& supportingRuleCompiler);
/// @brief Find the registered `or`-category operator whose body is the
        /// canonical body of @p elems, or mint a fresh `or<orCounter>` entry
        /// in BOTH compile registries.
        ///
        /// @details
        /// The registry-dedup-then-mint block of every or-operator mint
        /// (D-268), one or per canonical body (I-323,
        /// D-556). The canonical key of
        /// @p elems (`CompilationReuse::canonicalReuseKey`: disjunct order
        /// and `u_` numbering do not matter) finds a public or first, else
        /// the numbered or registered under that body at its mint, its
        /// compilation or its binary load. The retained or may store its
        /// leaves in another order, so a reuse returns its signature's
        /// argument permutation and records the ordered list for
        /// `CompilationReuse::compiledOrByElements`. A miss mints
        /// `or<orCounter>` with the canonical `(or<N>[u_1,..,u_arity])`
        /// signature, registers `compiledExpressions` (entity resolution),
        /// `coreExpressionMap` (expansion / disintegration) and its canonical
        /// body in the repetition map, and bumps `orCounter`. Without the
        /// body key a construction in another disjunct order minted a second
        /// name for the same disjunction, and a later batch's compile of a
        /// reloaded theorem then named the theorem differently from its
        /// registration.
        ///
        /// @param elems     Ordered u_-canonical element list, each
        ///        disjunct in TRUE polarity (I-175).
        /// @param arity     Number of distinct `u_` arguments across
        ///        @p elems (the operator's signature arity).
        /// @param mintedOut Optional out-flag — set true iff a fresh entry
        ///        was minted (false on registry reuse).
        /// @param argumentPermutationOut Optional one-based input slots in
        ///        retained-signature order; required whenever the retained
        ///        or orders its signature differently (asserted).
        /// @return The operator name (existing or freshly minted).
        /// @invariant Single-threaded seam only (I-137) — no worker reads
        ///            the compile registries while a writer seam holds.
        ///            Every or is registered under its canonical body, so a
        ///            mint finds no or with the same elements (asserted).
        /// @see `constructOrTheorem`, `preMintReducedOrs` — the callers.
std::string findOrMintOrOperator(const std::vector<std::string>& elems,
                                         int arity,
                                         bool* mintedOut = nullptr,
                                         std::vector<int>* argumentPermutationOut = nullptr);
/// @brief Flatten a registered or-category entity into its ordered
        /// non-or leaves, expressed in the entity's OWN `u_` terms.
        ///
        /// @details
        /// The registry twin of `flattenOrLeaves` (which walks an
        /// instantiated `WorkInstruction`): walks the COMPILED entity
        /// graph instead, starting from the entity's own signature as the
        /// root instance (identity substitution), substituting each nested
        /// or-child's signature args by its instance args (the
        /// `isOrIntroInstall` substitution walk, heap form) and recursing
        /// into or-category children only (I-166). Non-or elements append
        /// in element order, so left- and right-nested shapes flatten to
        /// the same ordered leaf list the instance-side walk produces.
        ///
        /// Barrier-seam helper — heap use is sanctioned here (never on the
        /// burst path).
        ///
        /// @param orLe      The or-category compiled entity to flatten.
        /// @param outLeaves Receives the ordered non-or leaves (appended).
        /// @return (void)
        /// @invariant Recursion depth capped by MAX_INSTRUCTION_ELEMENTS
        ///            (assert); at least two leaves result; every leaf arg
        ///            is a `u_` token (registry canonical form).
        /// @see `flattenOrLeaves` — the instantiated-instruction twin.
        /// @see `preMintReducedOrs` — the pre-mint consumer.
void flattenRegistryOrLeaves(const LogicalEntity& orLe,
                                     std::vector<std::string>& outLeaves) const;
/// @brief Renumber the `u_` tokens of a reduced leaf list to the
        /// canonical `u_1..u_m` first-appearance scheme.
        ///
        /// @details
        /// A reduced leaf list (a parent or's leaves minus one) inherits
        /// the PARENT's `u_` numbering with a hole; the or-operator
        /// registry identity (I-23) is the u_-canonical form, so the
        /// surviving tokens renumber to `u_1..u_m` in first-appearance
        /// order across the list — the exact numbering
        /// `constructOrTheorem`'s `argToU` assigns. Substitution is
        /// simultaneous (`ce::replaceKeysInString`, token-boundary
        /// multi-key), so overlapping old/new token names
        /// (`u_2 -> u_1` beside `u_1 -> u_1`) rewrite correctly.
        ///
        /// @param leaves   The leaf list, rewritten in place.
        /// @param orderOut Optional: receives the ORIGINAL tokens in the
        ///        first-appearance order that became `u_1..u_m` — the
        ///        subset-exclusion head's argument order
        ///        (`compileOrSubsetExclusions`).
        /// @return The number of distinct `u_` tokens across the list —
        ///         the reduced or-operator's arity.
        /// @invariant Every leaf argument is a `u_` token (asserted).
        /// @see `preMintReducedOrs`, `compileOrSubsetExclusions` — the
        ///      production callers.
static int renumberULeaves(std::vector<std::string>& leaves,
                                   std::vector<std::string>* orderOut = nullptr) {
    std::vector<std::string> order;
    std::set<std::string> seen;
    for (const std::string& d : leaves) {
        for (const std::string& a : ce::getArgs(d)) {
            assert(a.rfind("u_", 0) == 0
                && "renumberULeaves: registry leaf argument is not a u_ token");
            if (seen.insert(a).second) order.push_back(a);
        }
    }
    if (orderOut != nullptr) *orderOut = order;
    std::map<std::string, std::string> ren;
    for (std::size_t i = 0; i < order.size(); ++i) {
        ren[order[i]] = "u_" + std::to_string(i + 1);
    }
    for (std::string& d : leaves) {
        d = ce::replaceKeysInString(d, ren);
    }
    return static_cast<int>(order.size());
}
/// @brief Pre-mint the reduced (k-1)-ary or-operators for every
        /// registered or, to fixpoint (the single-elimination closure).
        ///
        /// @details
        /// Part 1 of the B8 three-extension design
        /// (D-268): single-exclusion rules
        /// emitted at flat or consumption carry a genuine (k-1)-ary
        /// or-compact head, and registry writers are single-threaded seams
        /// (I-137) — so every reduced operator the emission could need is
        /// minted HERE, at the barrier / pre-burst seams, never mid-burst.
        /// The emission-side registry lookup then treats a miss as a hard
        /// assert.
        ///
        /// Walks a snapshot of the or-category entries of
        /// `compiledExpressions` in name order (deterministic), flattens
        /// each to its ordered non-or leaves (`flattenRegistryOrLeaves`),
        /// and for every entry with k >= 3 leaves mints the k
        /// leaves-minus-one lists (u_-renumbered, I-23 dedup) through
        /// `findOrMintOrOperator`. A fresh mint joins the worklist so ITS
        /// reductions exist too — the closure reaches single-elimination
        /// chains, bounded per entry by `kMaxReducedOrLeaves`.
        ///
        /// Idempotent: a re-run dedups to zero mints. Runs in compressor
        /// mode too — the mints are a pure function of the or-registry
        /// content, so the audited run and the compressor re-derivation
        /// stay registry-aligned.
        ///
        /// @return (void)
        /// @invariant Single-threaded seam only (I-137);
        ///            k <= kMaxReducedOrLeaves asserted per flattened
        ///            entry (Rule-19 tripwire naming the constant).
        /// @see `constructOrTheoremsInRun` — tail caller (in-run ors);
        ///      `proveKernel` — entry + post-compaction-drain callers.
void preMintReducedOrs();
/// @brief Compile an or's K-rule compacts — its "or implications"
        /// `!d_0 ∧ … ∧ !d_{i-1} ∧ !d_{i+1} ∧ … → d_i`, one per leaf, as
        /// `(implication<N>[u_…])` instances over the or's OWN `u_` tokens —
        /// into `LogicalEntity::implications` (D-309).
        ///
        /// @details
        /// Flattens the or to its registry-term leaves
        /// (`flattenRegistryOrLeaves`, the disintegrator's leaf order); for
        /// leaf `i` the premises are the other leaves negated with
        /// double-negation cancellation (`negate`, I-175) and the head is the
        /// leaf itself; the chain is assembled with
        /// `reconstructImplicationFullBind` (every argument is a `u_` token,
        /// so the rule is binder-free) and compiled through
        /// `compileImplicationToCompact` — the same dedup-or-mint the
        /// mail-broadcast compaction uses (`excludeRepetitions`, category
        /// `implication`), which strips the `u_` prefixes and returns the
        /// compact over the bare token names; `prefixArgumentsWithU`
        /// restores the or's token names, so the stored string's argument
        /// order IS the projection from the implication's signature positions
        /// onto the or's tokens. A dedup hit against an earlier-registered
        /// implication keeps THAT entity's first-seen premise order (I-52),
        /// exactly as the mail round-trip does.
        ///
        /// Idempotent: an or that already carries a list returns unchanged
        /// (a registry reuse or a reload re-registering the same or). Barrier
        /// seam only — heap use is sanctioned here, never on the burst path.
        ///
        /// @param orName Registry name of an or-category entity (asserted).
        /// @return (void)
        /// @invariant Single-threaded seam only (I-137). Afterwards the list
        ///            holds exactly `k` compacts, each with the or's full
        ///            arity (every leaf occurs in every K-rule, asserted).
        /// @see `compileOrSubsetExclusions` — appends the second family;
        ///      `findOrMintOrOperator`, `compileExpression` —
        ///      the mint-site callers.
void compileOrKRules(const std::string& orName);
/// @brief Mark an OR whose two digit parameters are interchangeable.
        ///
        /// @details Flattens positive OR nesting and compares signed leaf
        /// multisets after equality orientation, keeping free slots fixed.
        /// Exactly two definition-typed digit slots qualify. Other predicates
        /// retain ordered arguments; metadata is derived before proof work.
        /// @param orName Registered OR operator whose metadata is refreshed.
        /// @return Void.
        /// @invariant Does not change definitions, names or proof expressions.
        /// @see preMintReducedOrs; canonicalOrProcessingForm.
void compileOrSymmetry(const std::string& orName);
/// @brief The length of a COMPLETE or-implication list for an or with
        /// @p k leaves: `k` K-rules, plus `2^k − k − 2` subset-exclusion
        /// rules when k ≥ 3.
        ///
        /// @details
        /// The disintegrator's completeness tripwire and the closure seam's
        /// post-condition share this one formula: k < 3 → k; k ≥ 3 →
        /// 2^k − 2.
        ///
        /// @param k The or's flattened leaf count (2 ≤ k ≤
        ///        kMaxReducedOrLeaves, asserted).
        /// @return The expected list length.
        /// @see `compileOrKRules`, `compileOrSubsetExclusions`.
static int32_t expectedOrImplicationCount(int32_t k) {
    assert(k >= 2 && k <= ExecutionParameters::kMaxReducedOrLeaves
        && "expectedOrImplicationCount: leaf count outside [2, kMaxReducedOrLeaves]");
    if (k < 3) return k;
    return (int32_t{ 1 } << k) - 2;
}

private:
/// @brief Append an or's subset-exclusion compacts — the rules
        /// `!D_i1 ∧ … ∧ !D_ij → (reducedOr[survivors])` for every excluded
        /// index subset with 1 ≤ j ≤ k−2 (I-184, D-269) — to
        /// `LogicalEntity::implications`, after the K-rules.
        ///
        /// @details
        /// Enumerates the excluded subsets exactly as `consumeOrLeavesCohort`
        /// section 1b does (j ascending, size-j index subsets in lexicographic
        /// order), so list position `k + n` is the disintegrator's n-th
        /// emission. Per subset: the survivors in parent order are
        /// `u_`-renumbered (`renumberULeaves`, which also reports the parent
        /// tokens in first-appearance order) and resolved through
        /// `compiledOrByElements` — a miss asserts (I-185: the closure is
        /// registered before this runs); the head is
        /// `(reducedName[parent tokens in that order])`, the premises the
        /// excluded leaves negated in parent order; the chain is compiled like
        /// a K-rule (`reconstructImplicationFullBind` →
        /// `compileImplicationToCompact` → `prefixArgumentsWithU`).
        ///
        /// Called only from `preMintReducedOrs` after its closure walk, for an
        /// or whose list holds exactly its K-rules (precondition asserted).
        ///
        /// @param orName Registry name of an or-category entity with
        ///        3 ≤ k ≤ kMaxReducedOrLeaves leaves (asserted).
        /// @return (void)
        /// @invariant Single-threaded seam only (I-137). Afterwards the list
        ///            holds `expectedOrImplicationCount(k)` compacts.
        /// @see `compileOrKRules`, `preMintReducedOrs`,
        ///      `consumeOrLeavesCohort` (the consumer, same order).
void compileOrSubsetExclusions(const std::string& orName);
CompiledDefinitionRegistry& definitions;
CompilationReuse& compilationReuse;
StructuralCompiler& structuralCompiler;
SupportingRuleCompiler& supportingRuleCompiler;
};
} // namespace gl
