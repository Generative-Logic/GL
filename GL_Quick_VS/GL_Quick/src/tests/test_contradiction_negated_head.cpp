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
/// @brief Tests for the `try_contradiction_negated_head` complementary
///        contradiction LB (reductio: assume the negation of the head,
///        emit the conjecture on contradiction).
///
/// @details
/// Covers the registration-time creation block in
/// `ExpressionAnalyzer::addTheoremToMemory`: complement key / seed /
/// theorem-id wiring, independence from the verbatim `try_contradiction`
/// branch, the skip-on-exists dedup when opposite-head conjectures share
/// a chain, and the config parse (default false; IncubatorPeano2 true).

#include "test_harness.hpp"

#include "../prover.hpp"

TEST(contradiction_negated_head, creation_complement_lb) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    ea.parameters.try_contradiction = false;
    ea.parameters.try_contradiction_negated_head = true;

    const std::string conj =
        "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8](in2[7,8,3])(=[7,8])))";
    ea.addTheoremToMemory(conj, ea.body, 0, false, ea.globalDependencies);

    gl::Memory* anchorLB =
        ea.simpleMapStore.findChild(&ea.body, "(AnchorPeano[1,2,3,4,5,6])");
    ASSERT_TRUE(anchorLB != nullptr);
    gl::Memory* innerLB = ea.simpleMapStore.findChild(anchorLB, "(in2[7,8,3])");
    ASSERT_TRUE(innerLB != nullptr);

    // The complement LB exists; the verbatim LB does not (independence from
    // try_contradiction, which is off).
    gl::Memory* contra =
        ea.simpleMapStore.findChild(innerLB, "__contradiction__!(=[7,8])");
    ASSERT_TRUE(contra != nullptr);
    ASSERT_TRUE(ea.simpleMapStore.findChild(innerLB, "__contradiction__(=[7,8])")
                == nullptr);

    ASSERT_TRUE(contra->primedForContradiction);
    ASSERT_TRUE(contra->exprKey() == std::string("__contradiction__!(=[7,8])"));
    ASSERT_TRUE(contra->parentMemory == innerLB);

    // The stored theorem is the conjecture itself.
    ASSERT_NE(contra->contradictionTheoremId, 0);
    ASSERT_TRUE(contra->nameMap.decode(contra->contradictionTheoremId) == conj);

    // The negated head is seeded as a known main-scope statement.
    const gl::NameId seedId = contra->nameMap.lookup("!(=[7,8])");
    ASSERT_NE(seedId, 0);
    const gl::StatementFlags* fl = contra->intKnownStatements.find(
        gl::StatementKey{ seedId, gl::NameMap::MAIN_ID });
    ASSERT_TRUE(fl != nullptr);
    ASSERT_TRUE(fl != nullptr);
}

TEST(contradiction_negated_head, dedup_collision_opposite_heads) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    ea.parameters.try_contradiction = true;
    ea.parameters.try_contradiction_negated_head = true;

    const std::string conjPos =
        "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8](in2[7,8,3])(=[7,8])))";
    const std::string conjNeg =
        "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8](in2[7,8,3])!(=[7,8])))";

    ea.addTheoremToMemory(conjPos, ea.body, 0, false, ea.globalDependencies);

    gl::Memory* anchorLB =
        ea.simpleMapStore.findChild(&ea.body, "(AnchorPeano[1,2,3,4,5,6])");
    ASSERT_TRUE(anchorLB != nullptr);
    gl::Memory* innerLB = ea.simpleMapStore.findChild(anchorLB, "(in2[7,8,3])");
    ASSERT_TRUE(innerLB != nullptr);

    // conjPos created both polarities: verbatim (=[7,8]) + complement !(=[7,8]).
    gl::Memory* verbatimLB =
        ea.simpleMapStore.findChild(innerLB, "__contradiction__(=[7,8])");
    gl::Memory* complementLB =
        ea.simpleMapStore.findChild(innerLB, "__contradiction__!(=[7,8])");
    ASSERT_TRUE(verbatimLB != nullptr);
    ASSERT_TRUE(complementLB != nullptr);

    // conjNeg's verbatim and complement keys both collide with conjPos's LBs;
    // the byte-identical seed/theorem make skip-on-exists lossless, so both
    // children stay pointer-stable.
    ea.addTheoremToMemory(conjNeg, ea.body, 0, false, ea.globalDependencies);
    ASSERT_TRUE(ea.simpleMapStore.findChild(innerLB, "__contradiction__(=[7,8])")
                == verbatimLB);
    ASSERT_TRUE(ea.simpleMapStore.findChild(innerLB, "__contradiction__!(=[7,8])")
                == complementLB);
}

TEST(contradiction_negated_head, config_parse_defaults_and_incubator_peano2) {
    {
        // Default-false pin on the struct itself — config-free, because any
        // batch config's contradiction flags are a legitimate experiment
        // variable.
        gl::ProverParameters defaults;
        ASSERT_FALSE(defaults.try_contradiction);
        ASSERT_FALSE(defaults.try_contradiction_negated_head);
    }
    {
        gl::ExpressionAnalyzer ea(std::string("IncubatorPeano2"));
        ASSERT_TRUE(ea.parameters.try_contradiction_negated_head);
        ASSERT_TRUE(ea.parameters.try_contradiction);
    }
}
