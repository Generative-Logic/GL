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
/// @brief Tests for the vacuous-premise suppression: the normal-LB branch
///        of `dischargeContradiction` (flag + stage), the
///        `drainSubtreeDeactivations` walk, and the
///        `hasContradictedAncestor` emission-gate helper.

#include "test_harness.hpp"

#include "../prover.hpp"

namespace {
    const char* kConj =
        "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8](in2[7,8,3])(=[7,8])))";

    // Registers kConj with both contradiction polarities enabled. The
    // premise-LB chain: root -> (AnchorPeano[...]) -> (in2[7,8,3]).
    void registerChain(gl::ExpressionAnalyzer& ea) {
        ea.parameters.try_contradiction = true;
        ea.parameters.try_contradiction_negated_head = true;
        ea.addTheoremToMemory(std::string(kConj), ea.body, 0, false,
                              ea.globalDependencies);
    }

    gl::Memory* innerOf(gl::ExpressionAnalyzer& ea) {
        gl::Memory* anchorLB = ea.simpleMapStore.findChild(
            &ea.body, "(AnchorPeano[1,2,3,4,5,6])");
        ASSERT_TRUE(anchorLB != nullptr);
        gl::Memory* innerLB =
            ea.simpleMapStore.findChild(anchorLB, "(in2[7,8,3])");
        ASSERT_TRUE(innerLB != nullptr);
        return innerLB;
    }

    // Deposits a contradicting statement pair at the LB's exact main.
    void depositMainPair(gl::ExpressionAnalyzer& ea, gl::Memory& lb) {
        const int lvRun[1] = { lb.level };
        const gl::TransientOrigin origin{
            true, gl::OriginTag::taskFormulation, nullptr, 0 };
        ea.addExprToMemoryBlock(gl::StrSpan("(in[7,1])", 9), lb, 0, 0,
            lvRun, 1, origin, -1, -1, gl::StrSpan("main", 4), false);
        ea.addExprToMemoryBlock(gl::StrSpan("!(in[7,1])", 10), lb, 0, 0,
            lvRun, 1, origin, -1, -1, gl::StrSpan("main", 4), false);
    }
}

TEST(vacuous_premise_suppression, normal_lb_pair_sets_flag_and_deactivates) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    registerChain(ea);
    gl::Memory* innerLB = innerOf(ea);

    depositMainPair(ea, *innerLB);
    ASSERT_TRUE(innerLB->isActive);
    ASSERT_FALSE(innerLB->mainContradiction);

    ea.dischargeContradiction(*innerLB, -1);

    ASSERT_TRUE(innerLB->mainContradiction);
    ASSERT_FALSE(innerLB->isActive);
}

TEST(vacuous_premise_suppression, no_pair_no_flag) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    registerChain(ea);
    gl::Memory* innerLB = innerOf(ea);

    ea.dischargeContradiction(*innerLB, -1);

    ASSERT_FALSE(innerLB->mainContradiction);
    ASSERT_TRUE(innerLB->isActive);
}

TEST(vacuous_premise_suppression, subtree_drain_deactivates_descendants) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    registerChain(ea);
    gl::Memory* innerLB = innerOf(ea);

    gl::Memory* contraPos =
        ea.simpleMapStore.findChild(innerLB, "__contradiction__(=[7,8])");
    gl::Memory* contraNeg =
        ea.simpleMapStore.findChild(innerLB, "__contradiction__!(=[7,8])");
    ASSERT_TRUE(contraPos != nullptr);
    ASSERT_TRUE(contraNeg != nullptr);
    ASSERT_TRUE(contraPos->isActive);
    ASSERT_TRUE(contraNeg->isActive);
    ASSERT_NE(contraPos->contradictionTheoremId, 0);
    ASSERT_NE(contraNeg->contradictionTheoremId, 0);

    depositMainPair(ea, *innerLB);
    ea.dischargeContradiction(*innerLB, -1);
    ea.drainSubtreeDeactivations();

    ASSERT_FALSE(contraPos->isActive);
    ASSERT_FALSE(contraNeg->isActive);
    ASSERT_EQ(contraPos->contradictionTheoremId, 0);
    ASSERT_EQ(contraNeg->contradictionTheoremId, 0);
    // The staging vector is cleared by the drain.
    ASSERT_EQ(static_cast<int>(ea.pendingSubtreeDeactivations.size()), 0);
}

TEST(vacuous_premise_suppression, has_contradicted_ancestor_walk) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    registerChain(ea);
    gl::Memory* innerLB = innerOf(ea);
    gl::Memory* contraNeg =
        ea.simpleMapStore.findChild(innerLB, "__contradiction__!(=[7,8])");
    ASSERT_TRUE(contraNeg != nullptr);

    ASSERT_FALSE(gl::hasContradictedAncestor(innerLB));
    ASSERT_FALSE(gl::hasContradictedAncestor(contraNeg));
    ASSERT_FALSE(gl::hasContradictedAncestor(&ea.body));

    innerLB->mainContradiction = true;

    ASSERT_TRUE(gl::hasContradictedAncestor(innerLB));      // self
    ASSERT_TRUE(gl::hasContradictedAncestor(contraNeg));    // parent
    ASSERT_FALSE(gl::hasContradictedAncestor(innerLB->parentMemory));
    ASSERT_FALSE(gl::hasContradictedAncestor(&ea.body));    // root untouched
}
