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
/// @brief Tests for the binder gate of `gl::TheoremPublisher::reformulateTheorem`
///        — a set-definition reformulation is formed only when the definition
///        premise binds the defined set alone, so its result never leaves a
///        variable without a binder (I-4).

#include "test_harness.hpp"
#include "theorem_publisher_fixture.hpp"
#include "prover_fixture.hpp"

#include "../prover.hpp"

#include <string>
#include <utility>
#include <vector>

namespace {

    // Every variable of `theorem` has a binder.
    bool isClosed(const std::string& theorem) {
        ce::TreeNode1* root = ce::parseExpr(theorem);
        const bool closed = root != nullptr && root->arguments.empty();
        ce::deleteTree(root);
        return closed;
    }

    // appendToSequence[n,a,f,g] records g as its defined set.
    void assertAppendIsSetDefinition(gl::tests::ProverFixture& ea) {
        const std::string name = "appendToSequence";
        const gl::LogicalEntity* le = ea.definitions.compiledEntity(gl::StrSpan(name));
        ASSERT_TRUE(le != nullptr);
        ASSERT_FALSE(le->definedSet.empty());
    }

} // namespace

// The appended value 12 is bound only by the definition premise
// appendToSequence[10,12,11,13], together with the defined sequence 13, and
// occurs in no other premise: the reformulated theorem would leave it
// without a binder, so none is formed.
TEST(reformulation_binding, definition_binding_another_variable_is_not_reformulated) {
    gl::tests::ProverFixture ea(std::string("FTA"));
    assertAppendIsSetDefinition(ea);
    const std::string theorem =
        "(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](in2[9,10,3])"
        "(>[11](sequence[1,4,2,9,11])(>[12,13](appendToSequence[10,12,11,13])"
        "(>[14,15](in2[14,15,11])(in2[14,15,13]))))))";
    ASSERT_TRUE(isClosed(theorem));
    std::vector<std::pair<std::string, std::string>> out;
    ASSERT_FALSE(gl::tests::TheoremPublisherFixture::reformulateTheorem(ea.theoremPublisher, theorem, out));
    ASSERT_TRUE(out.empty());
}

// The same statement with the appended value bound by its own typing
// premise: the definition premise binds the defined sequence alone, the
// reformulation is formed, and both of its forms are closed.
TEST(reformulation_binding, definition_binding_only_the_defined_set_is_reformulated) {
    gl::tests::ProverFixture ea(std::string("FTA"));
    assertAppendIsSetDefinition(ea);
    const std::string theorem =
        "(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](in2[9,10,3])"
        "(>[11](sequence[1,4,2,9,11])(>[12](in[12,1])(>[13](appendToSequence[10,12,11,13])"
        "(>[14,15](in2[14,15,11])(in2[14,15,13])))))))";
    ASSERT_TRUE(isClosed(theorem));
    std::vector<std::pair<std::string, std::string>> out;
    ASSERT_TRUE(gl::tests::TheoremPublisherFixture::reformulateTheorem(ea.theoremPublisher, theorem, out));
    ASSERT_EQ(out.size(), static_cast<std::size_t>(1));
    ASSERT_TRUE(isClosed(out[0].first));
    ASSERT_TRUE(isClosed(out[0].second));
}
