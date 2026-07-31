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
/// @brief Tests for the axed-variable deposit filter's flag-gated
///        positive-anchor exception in `addExprToMemoryBlock`
///        (D-234, `axed_anchor_exception`,
///        default off): with the flag ON, a positive anchor-category
///        statement carrying an axed x-copy name deposits; with the flag
///        at its default OFF, and for every other x-carrying deposit —
///        non-anchor or negated anchor — the filter drops it.

#include "test_harness.hpp"

#include "../prover.hpp"

namespace {
    const char* kConj =
        "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8](in2[7,8,3])(=[7,8])))";

    // Registers kConj; the premise-LB chain:
    // root -> (AnchorPeano[...]) -> (in2[7,8,3]).
    gl::Memory* innerOf(gl::ExpressionAnalyzer& ea) {
        ea.addTheoremToMemory(std::string(kConj), ea.body, 0, false,
                              ea.globalDependencies);
        gl::Memory* anchorLB = ea.simpleMapStore.findChild(
            &ea.body, "(AnchorPeano[1,2,3,4,5,6])");
        ASSERT_TRUE(anchorLB != nullptr);
        gl::Memory* innerLB =
            ea.simpleMapStore.findChild(anchorLB, "(in2[7,8,3])");
        ASSERT_TRUE(innerLB != nullptr);
        return innerLB;
    }

    // Mimics prehandleAnchor's axed-variable registration for slot 2.
    void axeX2(gl::Memory& lb) {
        lb.intAxedVariables.mint(lb.nameMap.encode("x2"));
    }

    void deposit(gl::ExpressionAnalyzer& ea, gl::Memory& lb,
                 const char* expr) {
        const int lvRun[1] = { lb.level };
        const gl::TransientOrigin origin{
            true, gl::OriginTag::taskFormulation, nullptr, 0 };
        ea.addExprToMemoryBlock(
            gl::StrSpan(expr, static_cast<int32_t>(std::strlen(expr))), lb, 0,
            0, lvRun, 1, origin, -1, -1, gl::StrSpan("main", 4), true);
    }

    bool knownAtMain(gl::Memory& lb, const char* expr) {
        const gl::StatementFlags* row = gl::lookupStatementFlags(
            lb.intKnownStatements, lb.nameMap, std::string(expr),
            std::string("main"));
        return row != nullptr && row->known;
    }
}

TEST(axed_anchor_exception, non_anchor_with_axed_arg_stays_filtered) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    gl::Memory* lb = innerOf(ea);
    axeX2(*lb);

    deposit(ea, *lb, "(in[x2,1])");

    ASSERT_FALSE(knownAtMain(*lb, "(in[x2,1])"));
}

TEST(axed_anchor_exception, positive_anchor_with_axed_arg_deposits_when_enabled) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    ea.parameters.axed_anchor_exception = true;
    gl::Memory* lb = innerOf(ea);
    axeX2(*lb);

    deposit(ea, *lb, "(AnchorPeano[1,x2,3,4,5,6])");

    ASSERT_TRUE(knownAtMain(*lb, "(AnchorPeano[1,x2,3,4,5,6])"));
}

TEST(axed_anchor_exception, positive_anchor_with_axed_arg_filtered_by_default) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    gl::Memory* lb = innerOf(ea);
    axeX2(*lb);

    deposit(ea, *lb, "(AnchorPeano[1,x2,3,4,5,6])");

    ASSERT_FALSE(knownAtMain(*lb, "(AnchorPeano[1,x2,3,4,5,6])"));
}

TEST(axed_anchor_exception, negated_anchor_with_axed_arg_stays_filtered) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    ea.parameters.axed_anchor_exception = true;
    gl::Memory* lb = innerOf(ea);
    axeX2(*lb);

    deposit(ea, *lb, "!(AnchorPeano[1,x2,3,4,5,6])");

    ASSERT_FALSE(knownAtMain(*lb, "!(AnchorPeano[1,x2,3,4,5,6])"));
}

TEST(axed_anchor_exception, axed_free_deposits_unaffected) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    gl::Memory* lb = innerOf(ea);
    axeX2(*lb);

    deposit(ea, *lb, "(in[7,1])");

    ASSERT_TRUE(knownAtMain(*lb, "(in[7,1])"));
}
