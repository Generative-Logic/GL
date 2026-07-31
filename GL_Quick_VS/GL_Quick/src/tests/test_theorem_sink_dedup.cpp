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
/// @brief Tests for `ExpressionAnalyzer::appendGlobalTheorem` — the
///        `globalTheoremList` sink dedup (insert-if-absent on the theorem
///        string, first emission wins).

#include "test_harness.hpp"

#include "../prover.hpp"

TEST(theorem_sink_dedup, duplicate_append_is_dropped) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    const std::string thm =
        "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8](in2[7,8,3])(=[7,8])))";

    ASSERT_TRUE(ea.appendGlobalTheorem(thm, "direct", "-1", "-1"));
    // Same theorem string through a different emission path: dropped.
    ASSERT_FALSE(ea.appendGlobalTheorem(thm, "induction", "v1", "0"));
    ASSERT_EQ(static_cast<int>(ea.globalTheoremList.size()), 1);
    // First emission wins: the retained tuple keeps the first method label.
    ASSERT_TRUE(std::get<1>(ea.globalTheoremList[0]) == std::string("direct"));
}

TEST(theorem_sink_dedup, distinct_theorems_both_land) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    ASSERT_TRUE(ea.appendGlobalTheorem("(a)", "direct", "-1", "-1"));
    ASSERT_TRUE(ea.appendGlobalTheorem("(b)", "direct", "-1", "-1"));
    ASSERT_EQ(static_cast<int>(ea.globalTheoremList.size()), 2);
}

// Vacuous reversion: rows whose recorded producer (or a producer ancestor)
// carries mainContradiction leave globalTheoremList, globalTheoremStrings
// (a legitimate re-prove may re-append), and fullTheoremList; rows of clean
// or producer-less paths stay.
TEST(theorem_sink_dedup, vacuous_producer_rows_are_reverted) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    gl::Memory clean;
    gl::Memory bad;
    gl::Memory badChild;
    badChild.parentMemory = &bad;

    ASSERT_TRUE(ea.appendGlobalTheorem("(a)", "direct", "-1", "-1", &clean));
    ASSERT_TRUE(ea.appendGlobalTheorem("(b)", "direct", "-1", "-1", &bad));
    ASSERT_TRUE(ea.appendGlobalTheorem("(c)", "reformulated statement", "(b)",
                                       "-1", &badChild));
    ASSERT_TRUE(ea.appendGlobalTheorem("(d)", "or theorem", "-1", "-1"));
    ea.fullTheoremList.emplace_back("(b)", "direct", "-1", "-1");
    ea.fullTheoremList.emplace_back("(a)", "direct", "-1", "-1");

    // Nothing flagged: sweep is a no-op.
    ea.revertVacuousGlobalTheorems();
    ASSERT_EQ(static_cast<int>(ea.globalTheoremList.size()), 4);

    // Flag the producer: its own row AND its descendant's row leave.
    bad.mainContradiction = true;
    ea.revertVacuousGlobalTheorems();
    ASSERT_EQ(static_cast<int>(ea.globalTheoremList.size()), 2);
    ASSERT_TRUE(std::get<0>(ea.globalTheoremList[0]) == std::string("(a)"));
    ASSERT_TRUE(std::get<0>(ea.globalTheoremList[1]) == std::string("(d)"));
    ASSERT_EQ(static_cast<int>(ea.globalTheoremProducers.size()), 2);
    ASSERT_EQ(static_cast<int>(ea.fullTheoremList.size()), 1);
    ASSERT_TRUE(std::get<0>(ea.fullTheoremList[0]) == std::string("(a)"));

    // The dedup string is free again: a legitimate re-prove re-appends.
    ASSERT_TRUE(ea.appendGlobalTheorem("(b)", "direct", "-1", "-1", &clean));
}

// Premise-contradiction vacuity certificate: contradictory heads on the same
// premise chain retire every theorem of that chain (including a third,
// non-contradictory head); an unrelated chain stays; the dedup string frees;
// the aligned producer vector and fullTheoremList follow.
TEST(theorem_sink_dedup, vacuous_premise_pairs_classified_out) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    ASSERT_TRUE(ea.appendGlobalTheorem("(>[1](a[1])(b[1]))", "direct", "-1", "-1"));
    ASSERT_TRUE(ea.appendGlobalTheorem("(>[1](a[1])!(b[1]))", "direct", "-1", "-1"));
    ASSERT_TRUE(ea.appendGlobalTheorem("(>[1](a[1])(c[1]))", "direct", "-1", "-1"));
    ASSERT_TRUE(ea.appendGlobalTheorem("(>[1](d[1])(e[1]))", "direct", "-1", "-1"));
    ea.fullTheoremList.emplace_back("(>[1](a[1])(b[1]))", "direct", "-1", "-1");
    ea.fullTheoremList.emplace_back("(>[1](d[1])(e[1]))", "direct", "-1", "-1");

    const std::vector<std::string> vac = ea.classifyVacuousPremisePairs();

    ASSERT_EQ(static_cast<int>(vac.size()), 3);
    ASSERT_TRUE(vac[0] == std::string("(>[1](a[1])(b[1]))"));
    ASSERT_TRUE(vac[1] == std::string("(>[1](a[1])!(b[1]))"));
    ASSERT_TRUE(vac[2] == std::string("(>[1](a[1])(c[1]))"));
    ASSERT_EQ(static_cast<int>(ea.globalTheoremList.size()), 1);
    ASSERT_TRUE(std::get<0>(ea.globalTheoremList[0])
                == std::string("(>[1](d[1])(e[1]))"));
    ASSERT_EQ(static_cast<int>(ea.globalTheoremProducers.size()), 1);
    ASSERT_EQ(static_cast<int>(ea.fullTheoremList.size()), 1);
    ASSERT_TRUE(std::get<0>(ea.fullTheoremList[0])
                == std::string("(>[1](d[1])(e[1]))"));
    // Dedup freed for the classified strings.
    ASSERT_TRUE(ea.appendGlobalTheorem("(>[1](a[1])(b[1]))", "direct", "-1", "-1"));

    // No contradiction, no classification: a defined no-op.
    gl::ExpressionAnalyzer ea2(std::string("Peano"));
    ASSERT_TRUE(ea2.appendGlobalTheorem("(>[1](a[1])(b[1]))", "direct", "-1", "-1"));
    ASSERT_TRUE(ea2.appendGlobalTheorem("(>[1](a[1])(c[1]))", "direct", "-1", "-1"));
    ASSERT_TRUE(ea2.classifyVacuousPremisePairs().empty());
    ASSERT_EQ(static_cast<int>(ea2.globalTheoremList.size()), 2);
}
