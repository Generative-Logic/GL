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

/// @file
/// @brief Tests for the output-collision variable-copy trigger:
///        `gl::detectOutputCollisionCopyVar` (shape detection) and the
///        `(=[r,r_copy])` deposit at the second colliding premise LB in
///        `ExpressionAnalyzer::addTheoremToMemory`.

#include "test_harness.hpp"

#include "../prover.hpp"

namespace {
    // Two same-operator products fusing their result binders into one
    // shared output name 9 — the multiplicative-cancellation premise
    // shape (a·c = r ∧ b·c = r).
    const std::vector<std::string> collisionChain = {
        "(AnchorPeano[1,2,3,4,5,6])",
        "(in3[8,7,9,5])",
        "(in3[10,7,9,5])",
    };
}

TEST(output_collision_copy, detect_positive) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    std::string copyVar;
    std::size_t firstIdx = 999;
    ASSERT_TRUE(gl::detectOutputCollisionCopyVar(
        collisionChain, ea.coreExpressionMap, copyVar, firstIdx));
    ASSERT_TRUE(copyVar == std::string("9"));
    ASSERT_TRUE(firstIdx == 2u);
}

TEST(output_collision_copy, detect_deposit_index_is_second_colliding_premise) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    // The shared name 9 appears in an earlier premise and in the first
    // colliding premise; the deposit index is the SECOND colliding
    // premise — earlier carriers are shared chain prefixes and a deposit
    // there would mail the copy into foreign conjectures' subtrees.
    const std::vector<std::string> chain = {
        "(AnchorPeano[1,2,3,4,5,6])",
        "(preorder[1,4,9,7])",
        "(in3[8,7,9,5])",
        "(in3[10,7,9,5])",
    };
    std::string copyVar;
    std::size_t firstIdx = 999;
    ASSERT_TRUE(gl::detectOutputCollisionCopyVar(
        chain, ea.coreExpressionMap, copyVar, firstIdx));
    ASSERT_TRUE(copyVar == std::string("9"));
    ASSERT_TRUE(firstIdx == 3u);
}

TEST(output_collision_copy, detect_negative_different_outputs) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    const std::vector<std::string> chain = {
        "(in3[8,7,9,5])",
        "(in3[10,7,11,5])",
    };
    std::string copyVar;
    std::size_t firstIdx = 0;
    ASSERT_FALSE(gl::detectOutputCollisionCopyVar(
        chain, ea.coreExpressionMap, copyVar, firstIdx));
}

TEST(output_collision_copy, detect_negative_non_io_arg_differs) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    // Same operator token, same output name, but the non-input/output
    // slot (the operator anchor: · vs +) differs.
    const std::vector<std::string> chain = {
        "(in3[8,7,9,5])",
        "(in3[10,7,9,4])",
    };
    std::string copyVar;
    std::size_t firstIdx = 0;
    ASSERT_FALSE(gl::detectOutputCollisionCopyVar(
        chain, ea.coreExpressionMap, copyVar, firstIdx));
}

TEST(output_collision_copy, detect_negative_third_premise_shares_output) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    // Exactly-two gate: a third premise matching the signature suppresses.
    const std::vector<std::string> chain = {
        "(in3[8,7,9,5])",
        "(in3[10,7,9,5])",
        "(in3[11,7,9,5])",
    };
    std::string copyVar;
    std::size_t firstIdx = 0;
    ASSERT_FALSE(gl::detectOutputCollisionCopyVar(
        chain, ea.coreExpressionMap, copyVar, firstIdx));
}

TEST(output_collision_copy, detect_negative_negated_premise) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    const std::vector<std::string> chain = {
        "(in3[8,7,9,5])",
        "!(in3[10,7,9,5])",
    };
    std::string copyVar;
    std::size_t firstIdx = 0;
    ASSERT_FALSE(gl::detectOutputCollisionCopyVar(
        chain, ea.coreExpressionMap, copyVar, firstIdx));
}

TEST(output_collision_copy, detect_negative_operator_without_output) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    // preorder has no output slot; sharing a name across its argument
    // lists is ordinary joint structure, never a collision.
    const std::vector<std::string> chain = {
        "(preorder[1,4,7,8])",
        "(preorder[1,4,9,8])",
    };
    std::string copyVar;
    std::size_t firstIdx = 0;
    ASSERT_FALSE(gl::detectOutputCollisionCopyVar(
        chain, ea.coreExpressionMap, copyVar, firstIdx));
}

TEST(output_collision_copy, detect_negative_anchor_argument_output) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    // Shared output 2 is an anchor argument (the zero slot) — a theory
    // constant, never copied (the I-24 analogue guard). The shape is the
    // "no successor equals zero" family: s(a)=0 ∧ s(b)=0.
    const std::vector<std::string> chain = {
        "(AnchorPeano[1,2,3,4,5,6])",
        "(in2[7,2,3])",
        "(in2[8,2,3])",
    };
    std::string copyVar;
    std::size_t firstIdx = 0;
    ASSERT_FALSE(gl::detectOutputCollisionCopyVar(
        chain, ea.coreExpressionMap, copyVar, firstIdx));
}

TEST(output_collision_copy, detect_negative_identical_premises) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    const std::vector<std::string> chain = {
        "(in3[8,7,9,5])",
        "(in3[8,7,9,5])",
    };
    std::string copyVar;
    std::size_t firstIdx = 0;
    ASSERT_FALSE(gl::detectOutputCollisionCopyVar(
        chain, ea.coreExpressionMap, copyVar, firstIdx));
}

TEST(output_collision_copy, deposit_at_second_colliding_premise_lb) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    const std::string conj =
        "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8,9](in3[8,7,9,5])(>[10](in3[10,7,9,5])(=[8,10]))))";
    ea.addTheoremToMemory(conj, ea.body, 0, false, ea.globalDependencies);

    gl::Memory* anchorLB =
        ea.simpleMapStore.findChild(&ea.body, "(AnchorPeano[1,2,3,4,5,6])");
    ASSERT_TRUE(anchorLB != nullptr);
    gl::Memory* firstPremiseLB =
        ea.simpleMapStore.findChild(anchorLB, "(in3[8,7,9,5])");
    ASSERT_TRUE(firstPremiseLB != nullptr);
    gl::Memory* secondPremiseLB =
        ea.simpleMapStore.findChild(firstPremiseLB, "(in3[10,7,9,5])");
    ASSERT_TRUE(secondPremiseLB != nullptr);

    // The copy equality is a known main-scope statement of the second
    // colliding premise LB.
    const gl::NameId copyId = secondPremiseLB->nameMap.lookup("(=[9,9_copy])");
    ASSERT_NE(copyId, 0);
    const gl::StatementFlags* fl = secondPremiseLB->intKnownStatements.find(
        gl::StatementKey{ copyId, gl::NameMap::MAIN_ID });
    ASSERT_TRUE(fl != nullptr);

    // Neither ancestor carries the axiom at load — the first colliding
    // premise is a chain prefix shared with foreign conjectures, and a
    // deposit there would mail the copy into their subtrees. Descendants
    // of the deposit LB receive it via mail.
    ASSERT_TRUE(anchorLB->nameMap.lookup("(=[9,9_copy])") == 0);
    ASSERT_TRUE(firstPremiseLB->nameMap.lookup("(=[9,9_copy])") == 0);
}

TEST(output_collision_copy, no_deposit_without_collision) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    const std::string conj =
        "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8,9](in3[8,7,9,5])(>[10,11](in3[10,7,11,5])(=[9,11]))))";
    ea.addTheoremToMemory(conj, ea.body, 0, false, ea.globalDependencies);

    gl::Memory* anchorLB =
        ea.simpleMapStore.findChild(&ea.body, "(AnchorPeano[1,2,3,4,5,6])");
    ASSERT_TRUE(anchorLB != nullptr);
    gl::Memory* productLB =
        ea.simpleMapStore.findChild(anchorLB, "(in3[8,7,9,5])");
    ASSERT_TRUE(productLB != nullptr);
    ASSERT_TRUE(productLB->nameMap.lookup("(=[9,9_copy])") == 0);
}
