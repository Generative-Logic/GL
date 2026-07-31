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
/// @brief Tests for the contradiction-twin deactivation machinery
///        (stage / drain, sibling and per-head staging) and the
///        `SimpleMapStore::findChild` span overload.

#include "test_harness.hpp"

#include "../prover.hpp"

TEST(twin_deactivation, stage_and_drain_deactivates_active_twin) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    gl::Memory mb;
    mb.contradictionTheoremId = mb.nameMap.encode("(x)");
    ASSERT_TRUE(mb.isActive);
    ea.stageTwinDeactivation(&mb);
    ea.drainPendingTwinDeactivations();
    ASSERT_FALSE(mb.isActive);
    ASSERT_EQ(mb.contradictionTheoremId, 0);
}

TEST(twin_deactivation, null_stage_and_inactive_twin_are_noops) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    // Contracted no-twin case: staging nullptr stages nothing.
    ea.stageTwinDeactivation(nullptr);
    ea.drainPendingTwinDeactivations();
    // Already-inactive twin (it fired on its own first): untouched.
    gl::Memory mb;
    mb.isActive = false;
    const gl::NameId keep = mb.nameMap.encode("(x)");
    mb.contradictionTheoremId = keep;
    ea.stageTwinDeactivation(&mb);
    ea.drainPendingTwinDeactivations();
    ASSERT_EQ(mb.contradictionTheoremId, keep);
}

TEST(twin_deactivation, sibling_staged_from_fired_complement) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    ea.parameters.try_contradiction = true;
    ea.parameters.try_contradiction_negated_head = true;
    const std::string conj =
        "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8](in2[7,8,3])(=[7,8])))";
    ea.addTheoremToMemory(conj, ea.body, 0, false, ea.globalDependencies);

    gl::Memory* anchorLB =
        ea.simpleMapStore.findChild(&ea.body, "(AnchorPeano[1,2,3,4,5,6])");
    gl::Memory* innerLB = ea.simpleMapStore.findChild(anchorLB, "(in2[7,8,3])");
    gl::Memory* verbatimLB =
        ea.simpleMapStore.findChild(innerLB, "__contradiction__(=[7,8])");
    gl::Memory* complementLB =
        ea.simpleMapStore.findChild(innerLB, "__contradiction__!(=[7,8])");
    ASSERT_TRUE(verbatimLB != nullptr);
    ASSERT_TRUE(complementLB != nullptr);

    gl::ScratchArena arena;
    arena.bind(&gl::staticMemory());
    // The complement "fired": its sibling (the verbatim LB) retires.
    ea.stageContradictionSiblingTwin(*complementLB, arena);
    ea.drainPendingTwinDeactivations();
    ASSERT_FALSE(verbatimLB->isActive);
    ASSERT_TRUE(complementLB->isActive);
}

TEST(twin_deactivation, both_twins_staged_from_chain_lb_head) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    ea.parameters.try_contradiction = true;
    ea.parameters.try_contradiction_negated_head = true;
    const std::string conj =
        "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8](in2[7,8,3])(=[7,8])))";
    ea.addTheoremToMemory(conj, ea.body, 0, false, ea.globalDependencies);

    gl::Memory* anchorLB =
        ea.simpleMapStore.findChild(&ea.body, "(AnchorPeano[1,2,3,4,5,6])");
    gl::Memory* innerLB = ea.simpleMapStore.findChild(anchorLB, "(in2[7,8,3])");
    gl::Memory* verbatimLB =
        ea.simpleMapStore.findChild(innerLB, "__contradiction__(=[7,8])");
    gl::Memory* complementLB =
        ea.simpleMapStore.findChild(innerLB, "__contradiction__!(=[7,8])");

    gl::ScratchArena arena;
    arena.bind(&gl::staticMemory());
    const std::string head = "(=[7,8])";
    ea.stageContradictionTwinsOfHead(*innerLB, gl::StrSpan(head), arena);
    ea.drainPendingTwinDeactivations();
    ASSERT_FALSE(verbatimLB->isActive);
    ASSERT_FALSE(complementLB->isActive);
    ASSERT_TRUE(innerLB->isActive);
}

TEST(twin_deactivation, find_child_span_overload_matches_string) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    ea.parameters.try_contradiction = true;
    const std::string conj =
        "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8](in2[7,8,3])(=[7,8])))";
    ea.addTheoremToMemory(conj, ea.body, 0, false, ea.globalDependencies);

    const std::string key = "(AnchorPeano[1,2,3,4,5,6])";
    gl::Memory* viaString = ea.simpleMapStore.findChild(&ea.body, key);
    gl::Memory* viaSpan = ea.simpleMapStore.findChild(&ea.body, gl::StrSpan(key));
    ASSERT_TRUE(viaString != nullptr);
    ASSERT_TRUE(viaString == viaSpan);
    // Unminted key: absent by construction.
    const std::string missing = "(neverInterned[1])";
    ASSERT_TRUE(ea.simpleMapStore.findChild(&ea.body, gl::StrSpan(missing))
                == nullptr);
}
