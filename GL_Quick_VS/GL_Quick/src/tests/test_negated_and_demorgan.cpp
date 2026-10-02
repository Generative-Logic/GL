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

// Named negated conjunctions supply exclusion implications. Numbered
// conjunctions stay flat; neither kind opens or parks case scopes.

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

TEST(negated_and_demorgan, named_exclusion_rules_and_flat_statement) {
    gl::ExpressionAnalyzer ea(std::string("FTA"));
    const std::string neg = "!(strictOrder[1,4,9,10])";
    depositAtMain(ea, neg, /*doNotDisintegrate=*/false);

    bool k1 = false, k2 = false;
    findKShapes(ea, k1, k2);
    ASSERT_TRUE(k1);
    ASSERT_TRUE(k2);
    ASSERT_TRUE(ea.body.orImplications.sources.empty());

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

TEST(negated_and_demorgan, numbered_and_stays_excluded) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    ea.compiledExpressions["and987"] = gl::LogicalEntity("and",
        {"(in[u_1,u_2])", "!(in[u_2,u_1])"}, "(and987[u_1,u_2])", 2);
    const int32_t rules = ea.body.overallHashMemory.originals.count();
    depositAtMain(ea, "!(and987[a,N])", false);
    ASSERT_EQ(ea.body.overallHashMemory.originals.count(), rules);
    ASSERT_TRUE(ea.body.orImplications.sources.empty());
    ASSERT_TRUE(ea.body.orDisjunctCount.empty());
    ASSERT_TRUE(ea.body.overallHashMemory.rejectedMapOrdis.empty());
    ASSERT_TRUE(ea.body.overallHashMemory.rejectedMapOrdis2.empty());
    ASSERT_TRUE(gl::lookupStatementFlags(ea.body.intKnownStatements, ea.body.nameMap,
        gl::StrSpan("!(and987[a,N])", 14), gl::StrSpan("main", 4)) != nullptr);
}

TEST(negated_and_demorgan, digit_prefix_with_named_suffix_is_not_numbered) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    ea.compiledExpressions["and987extra"] = gl::LogicalEntity("and",
        {"(in[u_1,u_2])", "!(in[u_2,u_1])"}, "(and987extra[u_1,u_2])", 2);
    const int32_t rules = ea.body.overallHashMemory.originals.count();
    depositAtMain(ea, "!(and987extra[a,N])", false);
    ASSERT_EQ(ea.body.overallHashMemory.originals.count(), rules + 2);
    ASSERT_TRUE(ea.body.orImplications.sources.empty());
    ASSERT_TRUE(ea.body.orDisjunctCount.empty());
    ASSERT_TRUE(ea.body.overallHashMemory.rejectedMapOrdis.empty());
    ASSERT_TRUE(ea.body.overallHashMemory.rejectedMapOrdis2.empty());
}

TEST(negated_and_demorgan, named_exclusion_history_stays_at_the_source_scope) {
    gl::ExpressionAnalyzer ea(std::string("FTA"));
    ea.body.nameMap.encode("main");
    const gl::NameId scopeId = ea.body.nameMap.encodePush(gl::NameMap::MAIN_ID, "named_exclusion");
    const std::string scope = ea.body.nameMap.decode(scopeId);
    const std::string source = "!(strictOrder[1,4,9,10])";
    const int levels[]{0};
    const gl::TransientOrigin origin{true, gl::OriginTag::taskFormulation, nullptr, 0};
    ea.addExprToMemoryBlock(gl::StrSpan(source), ea.body, 0, 0, levels, 1,
        origin, -1, -1, gl::StrSpan(scope), false);

    const std::string expanded = "!(&(preorder[1,4,9,10])!(=[9,10]))";
    // Raw implication history keeps the unification parameters; the proof
    // processor later gives them the chapter's ordinary variable spelling.
    const std::string rules[]{"(>[]!(=[u_9,u_10])!(preorder[u_1,u_4,u_9,u_10]))",
        "(>[](preorder[u_1,u_4,u_9,u_10])(=[u_9,u_10]))"};
    bool foundExpansion = false;
    bool foundRules[]{false, false};
    const auto history = gl::decodeOriginMapSorted(ea.body.exprOriginMap, ea.body.originInterner);
    for (const auto& entry : history) {
        if (entry.first.first == expanded) {
            ASSERT_EQ(entry.first.second, scope);
            for (const auto& line : entry.second) {
                if (line.first != "expansion") continue;
                ASSERT_EQ(line.second.size(), std::size_t{1});
                ASSERT_EQ(line.second[0].original, source);
                ASSERT_EQ(line.second[0].validityName, scope);
                foundExpansion = true;
            }
        }
        for (int i = 0; i < 2; ++i) {
            if (entry.first.first != rules[i]) continue;
            ASSERT_EQ(entry.first.second, scope);
            for (const auto& line : entry.second) {
                if (line.first != "disintegration") continue;
                ASSERT_EQ(line.second.size(), std::size_t{1});
                ASSERT_EQ(line.second[0].original, expanded);
                ASSERT_EQ(line.second[0].validityName, scope);
                foundRules[i] = true;
            }
        }
    }
    ASSERT_TRUE(foundExpansion);
    ASSERT_TRUE(foundRules[0]);
    ASSERT_TRUE(foundRules[1]);
    ASSERT_TRUE(ea.body.orImplications.sources.empty());
    ASSERT_TRUE(ea.body.orDisjunctCount.empty());
}

TEST(negated_and_demorgan, non_and_negations_stay_flat) {
    gl::ExpressionAnalyzer ea(std::string("FTA"));
    ea.preMintReducedOrs();   // the seam completes every registry list first
    // preorder's body is an existence, not an AND — the door must not fire.
    const std::string neg = "!(preorder[1,4,9,10])";
    depositAtMain(ea, neg, /*doNotDisintegrate=*/false);

    bool k1 = false, k2 = false;
    findKShapes(ea, k1, k2);
    ASSERT_TRUE(!k1 && !k2);
}

TEST(negated_and_demorgan, never_opens_or_parks_a_cohort) {
    // Case analysis is reserved for or-category signatures: a negated
    // named conjunction creates exclusion implications but no cases, even where a compiled
    // or would open unconditionally — an active primed contradiction block
    // (or proof work without a goal) receiving a route-(b) deposit (a
    // product of disintegration).
    gl::ExpressionAnalyzer ea(std::string("FTA"));
    ea.body.primedForContradiction = true;
    ea.body.isActive = true;
    const std::string neg = "!(strictOrder[1,4,9,10])";
    const int lv[1] = { 0 };
    const gl::TransientOrigin origin{
        true, gl::OriginTag::taskFormulation, nullptr, 0 };
    ea.addExprToMemoryBlock(gl::StrSpan(neg), ea.body, 0, 0, lv, 1,
        origin, -1, -1, gl::StrSpan("main", 4),
        /*doNotDisintegrate=*/false, /*allowOrDisintegration=*/true);

    bool k1 = false, k2 = false;
    findKShapes(ea, k1, k2);
    ASSERT_TRUE(k1);
    ASSERT_TRUE(k2);
    ASSERT_TRUE(ea.body.orImplications.sources.empty());
    // No cohort registered, none queued for release, none parked.
    ASSERT_TRUE(ea.body.orDisjunctCount.empty());
    ASSERT_TRUE(ea.body.pendingOrReleases.empty());
    ASSERT_TRUE(ea.body.overallHashMemory.rejectedMapOrdis.empty());
    ASSERT_TRUE(ea.body.overallHashMemory.rejectedMapOrdis2.empty());
}
