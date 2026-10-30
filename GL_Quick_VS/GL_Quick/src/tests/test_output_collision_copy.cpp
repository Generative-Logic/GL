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
/// @brief Tests for the output-collision and defined-set variable-copy
///        triggers: `gl::detectOutputCollisionCopyVar` with the
///        `(=[r,r_copy])` deposit at the second colliding premise LB, and
///        `gl::detectDefinedSetCopyVar` with the `(=[n,n_copy])` deposit at
///        the innermost premise LB, both in
///        `gl::tests::ProverFixture::addTheoremToMemory`.

#include "../memory.hpp"
#include <cstddef>
#include <utility>
#include "test_harness.hpp"
#include "prover_fixture.hpp"
#include "compiled_definition_fixture.hpp"

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
    gl::tests::ProverFixture ea(std::string("Peano"));
    std::string copyVar;
    std::size_t firstIdx = 999;
    ASSERT_TRUE(gl::VariableCopyDetector::detectOutputCollisionCopyVar(
        collisionChain, gl::CompiledDefinitionFixture::coreExpressionMap(ea.definitions), copyVar, firstIdx));
    ASSERT_TRUE(copyVar == std::string("9"));
    ASSERT_TRUE(firstIdx == 2u);
}

TEST(output_collision_copy, detect_deposit_index_is_second_colliding_premise) {
    gl::tests::ProverFixture ea(std::string("Peano"));
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
    ASSERT_TRUE(gl::VariableCopyDetector::detectOutputCollisionCopyVar(
        chain, gl::CompiledDefinitionFixture::coreExpressionMap(ea.definitions), copyVar, firstIdx));
    ASSERT_TRUE(copyVar == std::string("9"));
    ASSERT_TRUE(firstIdx == 3u);
}

TEST(output_collision_copy, detect_negative_different_outputs) {
    gl::tests::ProverFixture ea(std::string("Peano"));
    const std::vector<std::string> chain = {
        "(in3[8,7,9,5])",
        "(in3[10,7,11,5])",
    };
    std::string copyVar;
    std::size_t firstIdx = 0;
    ASSERT_FALSE(gl::VariableCopyDetector::detectOutputCollisionCopyVar(
        chain, gl::CompiledDefinitionFixture::coreExpressionMap(ea.definitions), copyVar, firstIdx));
}

TEST(output_collision_copy, detect_negative_non_io_arg_differs) {
    gl::tests::ProverFixture ea(std::string("Peano"));
    // Same operator token, same output name, but the non-input/output
    // slot (the operator anchor: · vs +) differs.
    const std::vector<std::string> chain = {
        "(in3[8,7,9,5])",
        "(in3[10,7,9,4])",
    };
    std::string copyVar;
    std::size_t firstIdx = 0;
    ASSERT_FALSE(gl::VariableCopyDetector::detectOutputCollisionCopyVar(
        chain, gl::CompiledDefinitionFixture::coreExpressionMap(ea.definitions), copyVar, firstIdx));
}

TEST(output_collision_copy, detect_negative_third_premise_shares_output) {
    gl::tests::ProverFixture ea(std::string("Peano"));
    // Exactly-two gate: a third premise matching the signature suppresses.
    const std::vector<std::string> chain = {
        "(in3[8,7,9,5])",
        "(in3[10,7,9,5])",
        "(in3[11,7,9,5])",
    };
    std::string copyVar;
    std::size_t firstIdx = 0;
    ASSERT_FALSE(gl::VariableCopyDetector::detectOutputCollisionCopyVar(
        chain, gl::CompiledDefinitionFixture::coreExpressionMap(ea.definitions), copyVar, firstIdx));
}

TEST(output_collision_copy, detect_negative_negated_premise) {
    gl::tests::ProverFixture ea(std::string("Peano"));
    const std::vector<std::string> chain = {
        "(in3[8,7,9,5])",
        "!(in3[10,7,9,5])",
    };
    std::string copyVar;
    std::size_t firstIdx = 0;
    ASSERT_FALSE(gl::VariableCopyDetector::detectOutputCollisionCopyVar(
        chain, gl::CompiledDefinitionFixture::coreExpressionMap(ea.definitions), copyVar, firstIdx));
}

TEST(output_collision_copy, detect_negative_operator_without_output) {
    gl::tests::ProverFixture ea(std::string("Peano"));
    // preorder has no output slot; sharing a name across its argument
    // lists is ordinary joint structure, never a collision.
    const std::vector<std::string> chain = {
        "(preorder[1,4,7,8])",
        "(preorder[1,4,9,8])",
    };
    std::string copyVar;
    std::size_t firstIdx = 0;
    ASSERT_FALSE(gl::VariableCopyDetector::detectOutputCollisionCopyVar(
        chain, gl::CompiledDefinitionFixture::coreExpressionMap(ea.definitions), copyVar, firstIdx));
}

TEST(output_collision_copy, detect_negative_anchor_argument_output) {
    gl::tests::ProverFixture ea(std::string("Peano"));
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
    ASSERT_FALSE(gl::VariableCopyDetector::detectOutputCollisionCopyVar(
        chain, gl::CompiledDefinitionFixture::coreExpressionMap(ea.definitions), copyVar, firstIdx));
}

TEST(output_collision_copy, detect_negative_identical_premises) {
    gl::tests::ProverFixture ea(std::string("Peano"));
    const std::vector<std::string> chain = {
        "(in3[8,7,9,5])",
        "(in3[8,7,9,5])",
    };
    std::string copyVar;
    std::size_t firstIdx = 0;
    ASSERT_FALSE(gl::VariableCopyDetector::detectOutputCollisionCopyVar(
        chain, gl::CompiledDefinitionFixture::coreExpressionMap(ea.definitions), copyVar, firstIdx));
}

TEST(output_collision_copy, deposit_at_second_colliding_premise_lb) {
    gl::tests::ProverFixture ea(std::string("Peano"));
    const std::string conj =
        "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8,9](in3[8,7,9,5])(>[10](in3[10,7,9,5])(=[8,10]))))";
    ea.proofGridBuilder.addTheoremToMemory(conj, ea.body, 0, false, ea.globalDependencies);

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
    const gl::NameId copyId = ((*(secondPremiseLB)).nameMap).lookup("(=[9,9_copy])");
    ASSERT_NE(copyId, 0);
    const gl::StatementFlags* fl = ((*(secondPremiseLB)).intKnownStatements).find(
        gl::StatementKey{ copyId, gl::NameMap::MAIN_ID });
    ASSERT_TRUE(fl != nullptr);

    // Neither ancestor carries the axiom at load — the first colliding
    // premise is a chain prefix shared with foreign conjectures, and a
    // deposit there would mail the copy into their subtrees. Descendants
    // of the deposit LB receive it via mail.
    ASSERT_TRUE(((*(anchorLB)).nameMap).lookup("(=[9,9_copy])") == 0);
    ASSERT_TRUE(((*(firstPremiseLB)).nameMap).lookup("(=[9,9_copy])") == 0);
}

TEST(output_collision_copy, no_deposit_without_collision) {
    gl::tests::ProverFixture ea(std::string("Peano"));
    const std::string conj =
        "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8,9](in3[8,7,9,5])(>[10,11](in3[10,7,11,5])(=[9,11]))))";
    ea.proofGridBuilder.addTheoremToMemory(conj, ea.body, 0, false, ea.globalDependencies);

    gl::Memory* anchorLB =
        ea.simpleMapStore.findChild(&ea.body, "(AnchorPeano[1,2,3,4,5,6])");
    ASSERT_TRUE(anchorLB != nullptr);
    gl::Memory* productLB =
        ea.simpleMapStore.findChild(anchorLB, "(in3[8,7,9,5])");
    ASSERT_TRUE(productLB != nullptr);
    ASSERT_TRUE(((*(productLB)).nameMap).lookup("(=[9,9_copy])") == 0);
}

namespace {
    // I1: 2 <= n and D the divisor set of n above 2 imply that D has a
    // minimum; D is the head's only argument besides the anchor's, an input
    // of hasMinimum, and divisorSet defines it from the input n.
    const std::vector<std::string> definedSetChain = {
        "(AnchorFTA[1,2,3,4,5,6,7,8])",
        "(preorder[1,4,7,9])",
        "(divisorSet[1,4,5,7,9,10])",
    };
    const std::string kI1Conjecture =
        "(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9](preorder[1,4,7,9])"
        "(>[10](divisorSet[1,4,5,7,9,10])(hasMinimum[1,4,10]))))";
}

TEST(defined_set_copy, detect_positive) {
    gl::tests::ProverFixture ea(std::string("FTA"));
    std::string copyVar;
    ASSERT_TRUE(gl::VariableCopyDetector::detectDefinedSetCopyVar(definedSetChain, "(hasMinimum[1,4,10])",
        gl::CompiledDefinitionFixture::coreExpressionMap(ea.definitions), copyVar));
    ASSERT_TRUE(copyVar == std::string("9"));
}

TEST(defined_set_copy, detect_negative_set_not_an_input_of_the_head) {
    // subset declares no input: its set argument is no input slot.
    gl::tests::ProverFixture ea(std::string("FTA"));
    std::string copyVar;
    ASSERT_FALSE(gl::VariableCopyDetector::detectDefinedSetCopyVar(definedSetChain, "(subset[10,1])",
        gl::CompiledDefinitionFixture::coreExpressionMap(ea.definitions), copyVar));
    ASSERT_TRUE(copyVar.empty());
}

TEST(defined_set_copy, detect_negative_number_output) {
    // The product c of a*b = c is uniquely definable but a number, not a set.
    gl::tests::ProverFixture ea(std::string("FTA"));
    const std::vector<std::string> chain = {
        "(AnchorFTA[1,2,3,4,5,6,7,8])",
        "(in3[9,10,11,5])",
    };
    std::string copyVar;
    ASSERT_FALSE(gl::VariableCopyDetector::detectDefinedSetCopyVar(chain, "(preorder[1,4,6,11])",
        gl::CompiledDefinitionFixture::coreExpressionMap(ea.definitions), copyVar));
}

TEST(defined_set_copy, detect_negative_head_over_two_sets) {
    gl::tests::ProverFixture ea(std::string("FTA"));
    std::string copyVar;
    ASSERT_FALSE(gl::VariableCopyDetector::detectDefinedSetCopyVar(definedSetChain, "(subset[10,11])",
        gl::CompiledDefinitionFixture::coreExpressionMap(ea.definitions), copyVar));
}

TEST(defined_set_copy, detect_negative_negated_defining_premise) {
    gl::tests::ProverFixture ea(std::string("FTA"));
    const std::vector<std::string> chain = {
        "(AnchorFTA[1,2,3,4,5,6,7,8])",
        "(preorder[1,4,7,9])",
        "!(divisorSet[1,4,5,7,9,10])",
    };
    std::string copyVar;
    ASSERT_FALSE(gl::VariableCopyDetector::detectDefinedSetCopyVar(chain, "(hasMinimum[1,4,10])",
        gl::CompiledDefinitionFixture::coreExpressionMap(ea.definitions), copyVar));
}

TEST(defined_set_copy, deposit_at_innermost_lb) {
    gl::tests::ProverFixture ea(std::string("FTA"));
    ea.proofGridBuilder.addTheoremToMemory(kI1Conjecture, ea.body, 0, false, ea.globalDependencies);

    gl::Memory* anchorLB = ea.simpleMapStore.findChild(&ea.body, "(AnchorFTA[1,2,3,4,5,6,7,8])");
    ASSERT_TRUE(anchorLB != nullptr);
    gl::Memory* boundLB = ea.simpleMapStore.findChild(anchorLB, "(preorder[1,4,7,9])");
    ASSERT_TRUE(boundLB != nullptr);
    gl::Memory* setLB = ea.simpleMapStore.findChild(boundLB, "(divisorSet[1,4,5,7,9,10])");
    ASSERT_TRUE(setLB != nullptr);

    // The copy equality is a known main-scope statement of the innermost LB.
    const gl::NameId copyId = ((*(setLB)).nameMap).lookup("(=[9,9_copy])");
    ASSERT_NE(copyId, 0);
    const gl::StatementFlags* fl = ((*(setLB)).intKnownStatements).find(
        gl::StatementKey{ copyId, gl::NameMap::MAIN_ID });
    ASSERT_TRUE(fl != nullptr);
    // The premise LB above carries no copy axiom.
    ASSERT_TRUE(((*(boundLB)).nameMap).lookup("(=[9,9_copy])") == 0);
}

TEST(defined_set_copy, no_deposit_when_the_head_takes_no_input) {
    // "divisorSet is a subset of N": same defining premise, head without input.
    gl::tests::ProverFixture ea(std::string("FTA"));
    const std::string conj =
        "(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9](in[9,1])"
        "(>[10](divisorSet[1,4,5,7,9,10])(subset[10,1]))))";
    ea.proofGridBuilder.addTheoremToMemory(conj, ea.body, 0, false, ea.globalDependencies);

    gl::Memory* anchorLB = ea.simpleMapStore.findChild(&ea.body, "(AnchorFTA[1,2,3,4,5,6,7,8])");
    ASSERT_TRUE(anchorLB != nullptr);
    gl::Memory* typingLB = ea.simpleMapStore.findChild(anchorLB, "(in[9,1])");
    ASSERT_TRUE(typingLB != nullptr);
    gl::Memory* setLB = ea.simpleMapStore.findChild(typingLB, "(divisorSet[1,4,5,7,9,10])");
    ASSERT_TRUE(setLB != nullptr);
    ASSERT_TRUE(((*(setLB)).nameMap).lookup("(=[9,9_copy])") == 0);
}
