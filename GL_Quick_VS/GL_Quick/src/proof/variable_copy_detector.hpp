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
namespace gl {
/// @brief Detect the existing theorem shapes requiring a variable-copy witness.
/// @details Pure shape queries retain premise order, anchor exclusions and the original copy-variable selection.
class VariableCopyDetector {
public:
/// @brief Detect the antisymmetry conjecture shape and name the
    ///        variable to duplicate — the fourth variable-copy trigger.
    ///
    /// @details
    /// An antisymmetry-shaped conjecture carries two positive premises of
    /// the same operator whose argument lists are equal except exactly two
    /// positions holding the same two arguments swapped
    /// (`P[..,x,..,y,..]` / `P[..,y,..,x,..]`), with an equality head over
    /// exactly that argument pair. Proving such a head forces downstream
    /// rule instances to bind two template-distinct slots to one name —
    /// the collision pattern the exact normalized-key request match cannot
    /// assemble. The cure is a dead-end variable-copy axiom
    /// `(=[x,x_copy])` at the innermost premise LB: the equivalence class
    /// then generates all-distinct statement variants and the standard
    /// machinery (demand markers, witness revival, rule firings) closes
    /// the proof.
    ///
    /// This function is the pure detection half: it scans every ordered
    /// pair of chain elements for the shape and, on the first match,
    /// reports the swapped argument that appears first left-to-right in
    /// the earlier premise — the variable the caller duplicates. Both
    /// verdicts are defined results of the detection contract (most
    /// conjectures simply do not have the shape).
    ///
    /// @param chain      The disintegrated premise chain of the conjecture
    ///                   (raw element strings, e.g.
    ///                   `"(preorder[1,4,7,8])"`).
    /// @param head       The conjecture head (e.g. `"(=[7,8])"`).
    /// @param copyVarOut On detection, the variable to duplicate (the
    ///                   swapped argument at the earlier mismatch position
    ///                   of the earlier element); untouched otherwise.
    /// @return True iff the antisymmetry shape was detected.
    /// @see ProofGridBuilder::addTheoremToMemory — the deposit site;
    ///      IntegrationProcessor::checkNecessityForEquality and
    ///      LogicBlockOperations::reactToHypo — sibling variable-copy
    ///      emission sites sharing the `variableCopy` origin contract.
static bool detectAntisymmetryCopyVar(const std::vector<std::string>& chain,
                                   const std::string& head,
                                   std::string& copyVarOut);
/// @brief Detect the output-collision conjecture shape and name the
    ///        variable to duplicate — the fifth variable-copy trigger.
    ///
    /// @details
    /// Two positive premises of the same operator carrying the same name
    /// in the operator's single output slot while their input slots
    /// differ (every other slot byte-equal) — the MPL encoding of an
    /// equality between two compound terms. The caller deposits the
    /// dead-end axiom `(=[r,r_copy])` at the SECOND colliding premise LB
    /// so the equivalence class can generate the all-distinct statement
    /// variants request matching needs while foreign conjectures sharing
    /// only the first colliding premise as a chain prefix never receive
    /// the copy. The gate is exactly-two premises by design; a third
    /// premise matching the same signature suppresses it, and an
    /// anchor-argument shared name never fires (theory constant — the
    /// I-24 analogue). Full contract at the definition.
    ///
    /// @param chain         Disintegrated premise chain of the conjecture.
    /// @param coreMap       Compiled operator configurations (slot
    ///                      classification source).
    /// @param copyVarOut    On detection, the shared output name;
    ///                      untouched otherwise.
    /// @param firstIndexOut On detection, the chain index of the second
    ///                      colliding premise (the deposit LB); untouched
    ///                      otherwise.
    /// @return True iff the output-collision shape was detected.
    /// @see gl::detectAntisymmetryCopyVar — sibling trigger sharing the
    ///      `variableCopy` deposit contract;
    ///      ProofGridBuilder::addTheoremToMemory — the deposit site.
static bool detectOutputCollisionCopyVar(const std::vector<std::string>& chain,
                                      const ce::CoreExpressionMap& coreMap,
                                      std::string& copyVarOut,
                                      std::size_t& firstIndexOut);
/// @brief Detect the defined-set conjecture shape and name the number to
    ///        duplicate — the sixth variable-copy trigger
    ///        (D-599).
    ///
    /// @details
    /// The head is a property of one set D: D is the head's only argument
    /// besides the anchor's, at an input slot of the head's predicate. A
    /// positive premise defines D from one number n: D sits at a uniquely
    /// definable slot of that premise typed as a set of numbers (`P(1)`),
    /// and n is the premise's only other argument outside the anchor at an
    /// input slot typed as a number (`(1)`) — I1's `divisorSet[N,+,*,2,n,D]`
    /// with head `hasMinimum[N,+,D]`. A proof about D's members then
    /// applies rules at n with two of their variables bound to n (n ≤ n,
    /// n ∣ n for n ∈ D), which the exact normalized-key match cannot
    /// assemble while rule multiplication is off. The caller deposits the
    /// dead-end axiom `(=[n,n_copy])` at the innermost premise LB, and the
    /// equivalence class supplies the second name.
    ///
    /// Both verdicts are defined results of the detection contract; most
    /// conjectures do not have the shape.
    ///
    /// @param chain      Disintegrated premise chain of the conjecture, the
    ///                   anchor premise included.
    /// @param head       The conjecture head, possibly negated.
    /// @param coreMap    Compiled operator configurations (slot
    ///                   classification and definition sets).
    /// @param copyVarOut On detection, the number n to duplicate; untouched
    ///                   otherwise.
    /// @return True iff the defined-set shape was detected.
    /// @see gl::detectAntisymmetryCopyVar, gl::detectOutputCollisionCopyVar —
    ///      sibling triggers sharing the `variableCopy` deposit contract;
    ///      ProofGridBuilder::addTheoremToMemory — the deposit site.
static bool detectDefinedSetCopyVar(const std::vector<std::string>& chain,
                                    const std::string& head,
                                    const ce::CoreExpressionMap& coreMap,
                                    std::string& copyVarOut);
};
}
