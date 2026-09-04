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
/// @brief Tests for the antisymmetry-shaped variable-copy trigger:
///        `gl::detectAntisymmetryCopyVar` (shape detection) and the
///        `(=[X,X_copy])` deposit in
///        `ExpressionAnalyzer::addTheoremToMemory`.

#include "test_harness.hpp"

#include "../prover.hpp"

namespace {
    const std::vector<std::string> antisymChain = {
        "(AnchorPeano[1,2,3,4,5,6])",
        "(preorder[1,4,7,8])",
        "(preorder[1,4,8,7])",
    };
}

TEST(antisym_variable_copy, detect_positive) {
    std::string copyVar;
    ASSERT_TRUE(gl::detectAntisymmetryCopyVar(antisymChain, "(=[7,8])", copyVar));
    ASSERT_TRUE(copyVar == std::string("7"));
}

TEST(antisym_variable_copy, detect_positive_mirrored_head) {
    std::string copyVar;
    ASSERT_TRUE(gl::detectAntisymmetryCopyVar(antisymChain, "(=[8,7])", copyVar));
    // The copy variable is the swapped argument that appears first
    // left-to-right in the earlier premise, independent of head order.
    ASSERT_TRUE(copyVar == std::string("7"));
}

TEST(antisym_variable_copy, detect_negative_non_equality_head) {
    std::string copyVar;
    ASSERT_FALSE(gl::detectAntisymmetryCopyVar(
        antisymChain, "(preorder[1,4,7,8])", copyVar));
}

TEST(antisym_variable_copy, detect_negative_negated_head) {
    std::string copyVar;
    ASSERT_FALSE(gl::detectAntisymmetryCopyVar(antisymChain, "!(=[7,8])", copyVar));
}

TEST(antisym_variable_copy, detect_negative_head_over_other_vars) {
    std::string copyVar;
    ASSERT_FALSE(gl::detectAntisymmetryCopyVar(antisymChain, "(=[7,9])", copyVar));
}

TEST(antisym_variable_copy, detect_negative_negated_premise) {
    const std::vector<std::string> chain = {
        "(preorder[1,4,7,8])",
        "!(preorder[1,4,8,7])",
    };
    std::string copyVar;
    ASSERT_FALSE(gl::detectAntisymmetryCopyVar(chain, "(=[7,8])", copyVar));
}

TEST(antisym_variable_copy, detect_negative_identical_premises) {
    const std::vector<std::string> chain = {
        "(preorder[1,4,7,8])",
        "(preorder[1,4,7,8])",
    };
    std::string copyVar;
    ASSERT_FALSE(gl::detectAntisymmetryCopyVar(chain, "(=[7,8])", copyVar));
}

TEST(antisym_variable_copy, detect_negative_transitivity_shape) {
    const std::vector<std::string> chain = {
        "(preorder[1,4,7,8])",
        "(preorder[1,4,8,9])",
    };
    std::string copyVar;
    ASSERT_FALSE(gl::detectAntisymmetryCopyVar(chain, "(=[7,9])", copyVar));
}

TEST(antisym_variable_copy, detect_negative_different_operators) {
    const std::vector<std::string> chain = {
        "(preorder[1,4,7,8])",
        "(in3[1,4,8,7])",
    };
    std::string copyVar;
    ASSERT_FALSE(gl::detectAntisymmetryCopyVar(chain, "(=[7,8])", copyVar));
}

TEST(antisym_variable_copy, deposit_at_innermost_lb) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    const std::string conj =
        "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8](preorder[1,4,7,8])(>[](preorder[1,4,8,7])(=[7,8]))))";
    ea.addTheoremToMemory(conj, ea.body, 0, false, ea.globalDependencies);

    gl::Memory* anchorLB =
        ea.simpleMapStore.findChild(&ea.body, "(AnchorPeano[1,2,3,4,5,6])");
    ASSERT_TRUE(anchorLB != nullptr);
    gl::Memory* outerLB =
        ea.simpleMapStore.findChild(anchorLB, "(preorder[1,4,7,8])");
    ASSERT_TRUE(outerLB != nullptr);
    gl::Memory* innerLB =
        ea.simpleMapStore.findChild(outerLB, "(preorder[1,4,8,7])");
    ASSERT_TRUE(innerLB != nullptr);

    // The copy equality is a known main-scope statement of the innermost LB.
    const gl::NameId copyId = innerLB->nameMap.lookup("(=[7,7_copy])");
    ASSERT_NE(copyId, 0);
    const gl::StatementFlags* fl = innerLB->intKnownStatements.find(
        gl::StatementKey{ copyId, gl::NameMap::MAIN_ID });
    ASSERT_TRUE(fl != nullptr);
    ASSERT_TRUE(fl != nullptr);

    // The ancestors carry no copy axiom (the deposit is innermost-only).
    ASSERT_TRUE(outerLB->nameMap.lookup("(=[7,7_copy])") == 0);
}

TEST(antisym_variable_copy, no_deposit_for_other_shapes) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    const std::string conj =
        "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8](in2[7,8,3])(=[7,8])))";
    ea.addTheoremToMemory(conj, ea.body, 0, false, ea.globalDependencies);

    gl::Memory* anchorLB =
        ea.simpleMapStore.findChild(&ea.body, "(AnchorPeano[1,2,3,4,5,6])");
    ASSERT_TRUE(anchorLB != nullptr);
    gl::Memory* innerLB = ea.simpleMapStore.findChild(anchorLB, "(in2[7,8,3])");
    ASSERT_TRUE(innerLB != nullptr);
    ASSERT_TRUE(innerLB->nameMap.lookup("(=[7,7_copy])") == 0);
}
