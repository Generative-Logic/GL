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
/// @brief Tests for the negated-AND De-Morgan door
///        (`disintegrateExprCore2`) and the shared
///        `consumeOrLeavesCohort`: a deposited `!(op[args])` whose compiled
///        definition body is an AND emits the K mutual-exclusion rules over
///        its true-polarity disjuncts (a negated conjunct's disjunct is its
///        bare positive core), keeps the flat negated statement registered,
///        and is fully suppressed by `doNotDisintegrate`.

#include "test_harness.hpp"

#include "../prover.hpp"

namespace {

    /// @brief Deposit @p expr at the root LB's main scope, status 0.
    ///
    /// @param ea                 The analyzer.
    /// @param expr               Statement text to deposit.
    /// @param doNotDisintegrate  The kernel door's suppression flag.
    void depositAtMain(gl::ExpressionAnalyzer& ea, const std::string& expr,
                       bool doNotDisintegrate) {
        const int lv[1] = { 0 };
        const gl::TransientOrigin origin{
            true, gl::OriginTag::taskFormulation, nullptr, 0 };
        ea.addExprToMemoryBlock(gl::StrSpan(expr), ea.body, 0, 0, lv, 1,
            origin, -1, -1, gl::StrSpan("main", 4), doNotDisintegrate);
    }

    /// @brief Scan the root LB's installed rules for the two De-Morgan K
    ///        shapes of `!(strictOrder[..,a,b])`.
    ///
    /// @details
    /// `strictOrder`'s body is AND[witness-form <=, !(=[a,b])], so the
    /// true-polarity disjuncts are the NEGATED <=-form and the POSITIVE
    /// equality. The K rules over two leaves are single-premise: one fires
    /// on the equality's negation with the negated <=-form as head, the
    /// sibling fires on the positive <=-form with the equality as head.
    /// The compiled body-element names are registry-relative, so the scan
    /// matches shapes, not exact operator names.
    ///
    /// @param ea   The analyzer holding the installed rules.
    /// @param k1   Out: found premise=!(=[..]) with a negated head.
    /// @param k2   Out: found head=(=[..]) with a positive premise.
    void findKShapes(gl::ExpressionAnalyzer& ea, bool& k1, bool& k2) {
        k1 = false;
        k2 = false;
        const int32_t n = ea.body.overallHashMemory.originals.count();
        for (int32_t oi = 1; oi <= n; ++oi) {
            const auto c = gl::decodeValueVector(
                ea.body.overallHashMemory.originals.decodeKey(oi).ids,
                ea.body.ruleInterner);
            if (c.size() != 2) continue;
            const std::string& prem = c[0];
            const std::string& head = c[1];
            if (prem.rfind("!(=[", 0) == 0 && head.rfind("!(", 0) == 0) {
                k1 = true;
            }
            if (head.rfind("(=[", 0) == 0 && !prem.empty()
                && prem[0] == '(') {
                k2 = true;
            }
        }
    }

} // namespace

TEST(negated_and_demorgan, k_rules_and_flat_statement) {
    gl::ExpressionAnalyzer ea(std::string("FTA"));
    const std::string neg = "!(strictOrder[1,4,9,10])";
    depositAtMain(ea, neg, /*doNotDisintegrate=*/false);

    bool k1 = false, k2 = false;
    findKShapes(ea, k1, k2);
    ASSERT_TRUE(k1);
    ASSERT_TRUE(k2);

    // The flat negated statement registers exactly as before the door.
    const gl::StatementFlags* row = gl::lookupStatementFlags(
        ea.body.intKnownStatements, ea.body.nameMap,
        gl::StrSpan(neg), gl::StrSpan("main", 4));
    ASSERT_TRUE(row != nullptr);
}

TEST(negated_and_demorgan, suppressed_by_do_not_disintegrate) {
    gl::ExpressionAnalyzer ea(std::string("FTA"));
    const int32_t before = ea.body.overallHashMemory.originals.count();
    const std::string neg = "!(strictOrder[1,4,9,10])";
    depositAtMain(ea, neg, /*doNotDisintegrate=*/true);

    // No disintegration, no K rules — the rule registry is untouched and
    // the statement still registers flat.
    ASSERT_TRUE(ea.body.overallHashMemory.originals.count() == before);
    const gl::StatementFlags* row = gl::lookupStatementFlags(
        ea.body.intKnownStatements, ea.body.nameMap,
        gl::StrSpan(neg), gl::StrSpan("main", 4));
    ASSERT_TRUE(row != nullptr);
}

TEST(negated_and_demorgan, non_and_negations_stay_flat) {
    gl::ExpressionAnalyzer ea(std::string("FTA"));
    // preorder's body is an existence, not an AND — the door must not fire.
    const std::string neg = "!(preorder[1,4,9,10])";
    depositAtMain(ea, neg, /*doNotDisintegrate=*/false);

    bool k1 = false, k2 = false;
    findKShapes(ea, k1, k2);
    ASSERT_TRUE(!k1 && !k2);
}
