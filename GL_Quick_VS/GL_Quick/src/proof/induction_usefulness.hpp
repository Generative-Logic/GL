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
#include "../logical_block/canonicalizer.hpp"
#include <string>
#include <vector>

namespace gl {

/// @brief The usefulness test of a weak induction step block: can its
///        hypothesis head, in the best case, only repeat a premise of the
///        conjecture?
///
/// @details
/// A weak step block holds the conjecture at `rec` as its induction
/// hypothesis. When the hypothesis head is an operator expression whose output
/// the configuration marks `uniquely_definable`, and the conjecture has exactly
/// one premise with the head's operator, that premise already fixes the
/// operator's output once the inputs agree: the head can at best restate it.
/// The grid build plants, in the scope `main_boundary_is_useful` on the step
/// block's `main`, the hypothesis premises with their fixed-name prefixes
/// removed (the hypothesis variables keep the conjecture's names: the best
/// case) and the head with its output replaced by a test variable
/// `test_level_<level>_<serial>` (@ref qualify, @ref arm). The scope sees
/// `main`'s statements and rules, so `main`'s rules fire on both and their
/// results land in the scope (I-38); equality classes form and apply there as
/// anywhere, while nothing disintegrates, is admitted or becomes a goal there.
/// For row 81 the Peano uniqueness of the successor joins 10 and 9 from the
/// planted s(rec) = 10 and the seed s(rec) = 9. After each standard
/// processing the key, the conjecture premise with the open slot written
/// `marker`, is compared with the head through the scope's classes
/// (@ref caught). A catch flags the block for the barrier's switch-off; the
/// scope is queued for wiping then, or after burst @ref kTestBursts of the
/// batch's proof without a catch (@ref settle)
/// (D-521).
///
/// The test only prunes: a wrong hint or verdict can lose an induction route,
/// never add a fact.
///
/// @invariant Nothing planted or derived in the scope leaves it: the scope
///            exports no mail, goal or chapter row.
/// @see ProofGridBuilder::addTheoremToMemory, LogicBlockOperations::standardProcessing,
///      ScopeLifecycle::switchOffUselessStep, ScopeSyntax::atOrBelowUsefulnessScope.
class InductionUsefulness {
public:
    /// The proof bursts a test stays open: the two warm-up bursts and one
    /// more, the first in which the anchors' uniqueness rules can join row
    /// 81's s(rec) values (maintainer, 2026-10-16).
    static constexpr int kTestBursts = 3;

    /// @brief The planted content and key of one step block's usefulness test.
    ///
    /// @details Built by @ref qualify from the conjecture and the hypothesis;
    /// planted by the grid build and recorded by @ref arm. Every expression is
    /// in plain form (no `u_` prefixes).
    struct Test {
        /// The hypothesis premises, fixed-name prefixes removed, in order.
        std::vector<std::string> premises;
        /// The hypothesis head with its output replaced by @ref testVariable.
        std::string head;
        /// The conjecture premise with the head's operator, its output `marker`.
        std::string key;
        /// The open slot: the head operator's output argument index.
        int32_t openSlot = -1;
        /// The test variable `test_level_<level>_<serial>`.
        std::string testVariable;
    };

    /// @brief Bind the test to the prover's canonicalizer, definitions and
    ///        proof burst counter.
    ///
    /// @details The component owns no state; it borrows the three
    /// collaborators for the prover's lifetime. The burst counter is the
    /// prover's own (`Prover::proofBurstNumber`, 1-based, warm-up included):
    /// a test still pending after burst @ref kTestBursts has failed.
    ///
    /// @param canonicalizer    Class-canonical forms at a scope.
    /// @param definitions      Compiled definitions and their configuration.
    /// @param proofBurstNumber The number of the batch's current proof burst.
    /// @return None; construction binds the references.
    InductionUsefulness(Canonicalizer& canonicalizer,
                        const CompiledDefinitionRegistry& definitions,
                        const int& proofBurstNumber);

    /// @brief Decide whether a weak step block's hypothesis qualifies for the
    ///        usefulness test, and build the test when it does.
    ///
    /// @details
    /// The hypothesis qualifies when its head is a positive operator
    /// expression with exactly one output argument, that output is a free
    /// variable (no `u_` prefix) and is listed in the operator's
    /// `uniquely_definable`, every other head argument is fixed or named in
    /// a hypothesis premise, and the conjecture has exactly one positive
    /// premise with the head's operator. On success @p out receives the
    /// plain-form premises and head (output replaced by the test variable
    /// `test_level_<blockLevel>_0`), the key (that premise with its output
    /// written as `marker`) and the open slot. A hypothesis that does not
    /// qualify is a defined result: most do not.
    ///
    /// @param conjectureChain The conjecture's premises, anchor included, in
    ///                        order (plain form).
    /// @param anchorName      The batch anchor's name (`AnchorFTA`, ...).
    /// @param hypothesisChain The hypothesis premises (`u_` form), as built by
    ///                        `ProofGridBuilder::createAuxyImplication`.
    /// @param hypothesisHead  The hypothesis head (`u_` form).
    /// @param blockLevel      The step block's level (the test variable's level).
    /// @param out             Receives the test on success.
    /// @return `true` iff the hypothesis qualifies.
    /// @see arm.
    bool qualify(const std::vector<std::string>& conjectureChain,
                 const std::string& anchorName,
                 const std::vector<std::string>& hypothesisChain,
                 const std::string& hypothesisHead,
                 int blockLevel,
                 Test& out) const;

    /// @brief Record a planted test on its step block and mark it pending.
    ///
    /// @details Called by the grid build right after the test's premises and
    /// head entered the step block at `main_boundary_is_useful` through the
    /// statement door. Stores the scope, head and key ids and the open slot
    /// on the block (`Memory::usefulnessScopeId`, `usefulnessHeadId`,
    /// `usefulnessKeyId`, `usefulnessOpenSlot`) and sets
    /// `Memory::usefulnessTest` to `Pending`.
    ///
    /// @param stepBlock The weak step block (resident, at grid build).
    /// @param test      The planted test.
    /// @return None.
    /// @invariant The block carries no test yet, and the head registered at
    ///            the scope (both asserted).
    /// @see qualify, settle.
    static void arm(Memory& stepBlock, const Test& test);

    /// @brief Whether the key catches the planted head under the usefulness
    ///        scope's classes.
    ///
    /// @details Reads the head and the key through the classes at
    /// `main_boundary_is_useful` (text-only `canonicalFormAtScope`, which
    /// mints nothing) and compares them: the same operator and every argument
    /// equal except the open slot. Heap-free; the canonical texts ride the
    /// per-slot string scratch and are released on return.
    ///
    /// @param stepBlock A step block with a pending test (asserted).
    /// @return `true` iff the key catches the head: the step is useless.
    /// @see settle.
    bool caught(Memory& stepBlock) const;

    /// @brief Settle a pending usefulness test at the end of a standard
    ///        processing call.
    ///
    /// @details Does nothing for a block without a pending test. A caught
    /// test becomes `Caught` (the barrier switches the block off,
    /// `ScopeLifecycle::switchOffUselessStep`); a test still uncaught after
    /// burst @ref kTestBursts of the batch's proof becomes `Done`. In both
    /// cases the scope is queued on `Memory::pendingWipeScopes`, so the same
    /// call's scope cleanup wipes it. Heap-free.
    ///
    /// @param stepBlock The block whose standard processing call ends.
    /// @return None.
    /// @see caught, arm.
    void settle(Memory& stepBlock) const;

    /// @brief Return a flat expression with one argument replaced.
    ///
    /// @details Rebuilds `(<core>[a0,...,an])` with argument @p slot set to
    /// @p value; the expression must be positive and flat.
    ///
    /// @param expression A positive flat expression.
    /// @param slot       The argument index to replace (asserted in range).
    /// @param value      The replacement argument.
    /// @return The rebuilt expression.
    static std::string withArgumentAt(const std::string& expression, int32_t slot,
                                      const std::string& value);

private:
    Canonicalizer& canonicalizer;
    const CompiledDefinitionRegistry& definitions;
    const int& proofBurstNumber;
};

} // namespace gl
