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
// conjunctions stay flat; neither kind opens or parks case scopes, except as
// an integration premise, which splits into its complemented conjuncts.

#include "../memory.hpp"
#include <cstddef>
#include <utility>
#include "test_harness.hpp"
#include "prover_fixture.hpp"
#include "compiled_definition_fixture.hpp"

#include <algorithm>
#include <string>
#include <vector>

#include "../prover.hpp"

namespace {

    /// @brief Deposit @p expr at the root LB's main scope, status 0.
    ///
    /// @param ea                 The analyzer.
    /// @param expr               Statement text to deposit.
    /// @param doNotDisintegrate  The kernel door's suppression flag.
    void depositAtMain(gl::tests::ProverFixture& ea, const std::string& expr,
                       bool doNotDisintegrate) {
        const int lv[1] = { 0 };
        const gl::TransientOrigin origin{
            true, gl::OriginTag::taskFormulation, nullptr, 0 };
        ea.statementMemory.addExprToMemoryBlock(gl::StrSpan(expr), ea.body, 0, 0, lv, 1,
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
    void findKShapes(gl::tests::ProverFixture& ea, bool& k1, bool& k2) {
        k1 = false;
        k2 = false;
        const int32_t n = ((((ea.body).overallHashMemory)).originals).count();
        for (int32_t oi = 1; oi <= n; ++oi) {
            const auto c = gl::decodeValueVector(
                ((((ea.body).overallHashMemory)).originals).decodeKey(oi).ids,
                ((ea.body).ruleInterner));
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

    /// @brief Deposit @p expr at the root LB's main scope as an integration
    ///        premise (D-460), status 0.
    ///
    /// @param ea   The analyzer.
    /// @param expr Statement text to deposit.
    void depositPremiseAtMain(gl::tests::ProverFixture& ea, const std::string& expr) {
        const int lv[1] = { 0 };
        const gl::TransientOrigin origin{
            true, gl::OriginTag::premiseElement, nullptr, 0 };
        ea.statementMemory.addExprToMemoryBlock(gl::StrSpan(expr), ea.body, 0, 0, lv, 1,
            origin, -1, -1, gl::StrSpan("main", 4),
            /*doNotDisintegrate=*/false, /*allowOrDisintegration=*/false,
            /*registeredOut=*/nullptr, /*recursionHead=*/false,
            /*convergenceMailOut=*/nullptr, /*replayKinds=*/0,
            /*replayIntegrationDepth=*/-1, /*integrationPremise=*/true);
    }

    /// @brief Deposit @p expr at the root LB's main scope as a conjecture's
    ///        last premise seeded into its goal block
    ///        (D-495), status 0.
    ///
    /// @param ea                The analyzer.
    /// @param expr              Statement text to deposit.
    /// @param conjecturePremise The door's conjecture-premise permission.
    void depositGoalBlockPremiseAtMain(gl::tests::ProverFixture& ea, const std::string& expr,
                                       bool conjecturePremise) {
        // A goal block sits below the anchor block; the fixture root has no
        // level, and witness names carry the block level.
        ((ea.body).level) = 3;
        const int lv[1] = { 3 };
        const gl::TransientOrigin origin{
            true, gl::OriginTag::taskFormulation, nullptr, 0 };
        ea.statementMemory.addExprToMemoryBlock(gl::StrSpan(expr), ea.body, 0, 0, lv, 1,
            origin, -1, -1, gl::StrSpan("main", 4),
            /*doNotDisintegrate=*/false, /*allowOrDisintegration=*/false,
            /*registeredOut=*/nullptr, /*recursionHead=*/false,
            /*convergenceMailOut=*/nullptr, /*replayKinds=*/0,
            /*replayIntegrationDepth=*/-1, /*integrationPremise=*/false,
            conjecturePremise);
    }

    /// @brief The decoded queued case disjuncts of the single cohort of
    ///        @p block (asserted single, first row).
    ///
    /// @param block The logical block holding exactly one cohort.
    /// @return The wrapped `(disjunct)` texts of its pending queue.
    std::vector<std::string> singleCohortCases(gl::Memory& block) {
        std::vector<std::string> cases;
        assert(((block).orPendingBranches).count() == 1);
        for (int32_t j = 0; j < ((block).orPendingBranches).runLen(1); ++j) {
            const gl::StrSpan w = ((block).lbStateInterner).decodeView(
                ((block).orPendingBranches).valueAt(1, j));
            cases.emplace_back(w.ptr, static_cast<std::size_t>(w.len));
        }
        return cases;
    }

    /// @brief The queued case disjuncts of the cohort rooted at @p source
    ///        under `main`, in stored order; empty when no cohort exists.
    ///
    /// @param ea     The analyzer.
    /// @param source The cohort's source signature (clean form).
    /// @return The wrapped `(disjunct)` texts of the pending queue.
    std::vector<std::string> pendingCasesAtMain(gl::tests::ProverFixture& ea,
                                                const std::string& source) {
        std::vector<std::string> cases;
        const int32_t cohort = gl::lookupOrCohortId(((ea.body).lbStateInterner),
            ((ea.body).lbStateInterner).lookup(gl::StrSpan("main", 4)),
            ((ea.body).lbStateInterner).lookup(gl::StrSpan(source)));
        if (cohort == 0) return cases;
        const int32_t row = ((ea.body).orPendingBranches).lookup(cohort);
        if (row == 0) return cases;
        for (int32_t j = 0; j < ((ea.body).orPendingBranches).runLen(row); ++j) {
            const gl::StrSpan w = ((ea.body).lbStateInterner).decodeView(
                ((ea.body).orPendingBranches).valueAt(row, j));
            cases.emplace_back(w.ptr, static_cast<std::size_t>(w.len));
        }
        return cases;
    }

} // namespace

TEST(negated_and_demorgan, named_exclusion_rules_and_flat_statement) {
    gl::tests::ProverFixture ea(std::string("FTA"));
    const std::string neg = "!(strictOrder[1,4,9,10])";
    depositAtMain(ea, neg, /*doNotDisintegrate=*/false);

    bool k1 = false, k2 = false;
    findKShapes(ea, k1, k2);
    ASSERT_TRUE(k1);
    ASSERT_TRUE(k2);
    ASSERT_TRUE(((ea.body).orImplications).sources.empty());

    // The flat negated statement registers exactly as before the door.
    const gl::StatementFlags* row = gl::lookupStatementFlags(
        ((ea.body).intKnownStatements), ((ea.body).nameMap),
        gl::StrSpan(neg), gl::StrSpan("main", 4));
    ASSERT_TRUE(row != nullptr);
}

TEST(negated_and_demorgan, suppressed_by_do_not_disintegrate) {
    gl::tests::ProverFixture ea(std::string("FTA"));
    const int32_t before = ((((ea.body).overallHashMemory)).originals).count();
    const std::string neg = "!(strictOrder[1,4,9,10])";
    depositAtMain(ea, neg, /*doNotDisintegrate=*/true);

    // No disintegration, no K rules — the rule registry is untouched and
    // the statement still registers flat.
    ASSERT_TRUE(((((ea.body).overallHashMemory)).originals).count() == before);
    const gl::StatementFlags* row = gl::lookupStatementFlags(
        ((ea.body).intKnownStatements), ((ea.body).nameMap),
        gl::StrSpan(neg), gl::StrSpan("main", 4));
    ASSERT_TRUE(row != nullptr);
}

TEST(negated_and_demorgan, numbered_and_stays_excluded) {
    gl::tests::ProverFixture ea(std::string("Peano"));
    gl::CompiledDefinitionFixture::compiledExpressions(ea.definitions)["and987"] = gl::LogicalEntity("and",
        {"(in[u_1,u_2])", "!(in[u_2,u_1])"}, "(and987[u_1,u_2])", 2);
    const int32_t rules = ((((ea.body).overallHashMemory)).originals).count();
    depositAtMain(ea, "!(and987[a,N])", false);
    ASSERT_EQ(((((ea.body).overallHashMemory)).originals).count(), rules);
    ASSERT_TRUE(((ea.body).orImplications).sources.empty());
    ASSERT_TRUE(((ea.body).orDisjunctCount).empty());
    ASSERT_TRUE(((((ea.body).overallHashMemory)).rejectedMapOrdis).empty());
    ASSERT_TRUE(((((ea.body).overallHashMemory)).rejectedMapOrdis2).empty());
    ASSERT_TRUE(gl::lookupStatementFlags(((ea.body).intKnownStatements), ((ea.body).nameMap),
        gl::StrSpan("!(and987[a,N])", 14), gl::StrSpan("main", 4)) != nullptr);
}

TEST(negated_and_demorgan, digit_prefix_with_named_suffix_is_not_numbered) {
    gl::tests::ProverFixture ea(std::string("Peano"));
    gl::CompiledDefinitionFixture::compiledExpressions(ea.definitions)["and987extra"] = gl::LogicalEntity("and",
        {"(in[u_1,u_2])", "!(in[u_2,u_1])"}, "(and987extra[u_1,u_2])", 2);
    const int32_t rules = ((((ea.body).overallHashMemory)).originals).count();
    depositAtMain(ea, "!(and987extra[a,N])", false);
    ASSERT_EQ(((((ea.body).overallHashMemory)).originals).count(), rules + 2);
    ASSERT_TRUE(((ea.body).orImplications).sources.empty());
    ASSERT_TRUE(((ea.body).orDisjunctCount).empty());
    ASSERT_TRUE(((((ea.body).overallHashMemory)).rejectedMapOrdis).empty());
    ASSERT_TRUE(((((ea.body).overallHashMemory)).rejectedMapOrdis2).empty());
}

TEST(negated_and_demorgan, named_exclusion_history_stays_at_the_source_scope) {
    gl::tests::ProverFixture ea(std::string("FTA"));
    ((ea.body).nameMap).encode("main");
    const gl::NameId scopeId = ((ea.body).nameMap).encodePush(gl::NameMap::MAIN_ID, "named_exclusion");
    const std::string scope = ((ea.body).nameMap).decode(scopeId);
    const std::string source = "!(strictOrder[1,4,9,10])";
    const int levels[]{0};
    const gl::TransientOrigin origin{true, gl::OriginTag::taskFormulation, nullptr, 0};
    ea.statementMemory.addExprToMemoryBlock(gl::StrSpan(source), ea.body, 0, 0, levels, 1,
        origin, -1, -1, gl::StrSpan(scope), false);

    const std::string expanded = "!(&(preorder[1,4,9,10])!(=[9,10]))";
    // Raw implication history keeps the unification parameters; the proof
    // processor later gives them the chapter's ordinary variable spelling.
    const std::string rules[]{"(>[]!(=[u_9,u_10])!(preorder[u_1,u_4,u_9,u_10]))",
        "(>[](preorder[u_1,u_4,u_9,u_10])(=[u_9,u_10]))"};
    bool foundExpansion = false;
    bool foundRules[]{false, false};
    const auto history = gl::decodeOriginMapSorted(((ea.body).exprOriginMap), ((ea.body).originInterner));
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
    ASSERT_TRUE(((ea.body).orImplications).sources.empty());
    ASSERT_TRUE(((ea.body).orDisjunctCount).empty());
}

TEST(negated_and_demorgan, non_and_negations_stay_flat) {
    gl::tests::ProverFixture ea(std::string("FTA"));
    ea.orRuleCompiler.preMintReducedOrs();   // the seam completes every registry list first
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
    gl::tests::ProverFixture ea(std::string("FTA"));
    ((ea.body).primedForContradiction) = true;
    ((ea.body).isActive) = true;
    const std::string neg = "!(strictOrder[1,4,9,10])";
    const int lv[1] = { 0 };
    const gl::TransientOrigin origin{
        true, gl::OriginTag::taskFormulation, nullptr, 0 };
    ea.statementMemory.addExprToMemoryBlock(gl::StrSpan(neg), ea.body, 0, 0, lv, 1,
        origin, -1, -1, gl::StrSpan("main", 4),
        /*doNotDisintegrate=*/false, /*allowOrDisintegration=*/true);

    bool k1 = false, k2 = false;
    findKShapes(ea, k1, k2);
    ASSERT_TRUE(k1);
    ASSERT_TRUE(k2);
    ASSERT_TRUE(((ea.body).orImplications).sources.empty());
    // No cohort registered, none queued for release, none parked.
    ASSERT_TRUE(((ea.body).orDisjunctCount).empty());
    ASSERT_TRUE(((ea.body).pendingOrReleases).empty());
    ASSERT_TRUE(((((ea.body).overallHashMemory)).rejectedMapOrdis).empty());
    ASSERT_TRUE(((((ea.body).overallHashMemory)).rejectedMapOrdis2).empty());
}

TEST(negated_and_demorgan, integration_premise_numbered_and_opens_cases) {
    // A negated numbered conjunction that is an integration premise splits
    // into its complemented conjuncts as one cohort rooted at the negated
    // conjunction itself; no exclusion source or park is created
    // (D-460).
    gl::tests::ProverFixture ea(std::string("Peano"));
    gl::CompiledDefinitionFixture::compiledExpressions(ea.definitions)["and987"] = gl::LogicalEntity("and",
        {"(in[u_1,u_2])", "!(in[u_2,u_1])"}, "(and987[u_1,u_2])", 2);
    ((ea.body).primedForContradiction) = true;
    ((ea.body).isActive) = true;
    depositPremiseAtMain(ea, "!(and987[a,N])");

    const std::vector<std::string> cases = pendingCasesAtMain(ea, "!(and987[a,N])");
    ASSERT_EQ(cases.size(), std::size_t{2});
    ASSERT_TRUE(std::find(cases.begin(), cases.end(), "(!(in[a,N]))") != cases.end());
    ASSERT_TRUE(std::find(cases.begin(), cases.end(), "((in[N,a]))") != cases.end());
    ASSERT_EQ(((ea.body).pendingOrReleases).count(), 1);
    ASSERT_TRUE(((ea.body).orImplications).sources.empty());
    ASSERT_TRUE(((((ea.body).overallHashMemory)).rejectedMapOrdis).empty());
    ASSERT_TRUE(((((ea.body).overallHashMemory)).rejectedMapOrdis2).empty());
    ASSERT_TRUE(gl::lookupStatementFlags(((ea.body).intKnownStatements), ((ea.body).nameMap),
        gl::StrSpan("!(and987[a,N])", 14), gl::StrSpan("main", 4)) != nullptr);
}

TEST(negated_and_demorgan, integration_premise_named_and_keeps_rules_and_opens_cases) {
    // A negated named conjunction as an integration premise keeps its flat
    // exclusion implications and additionally opens its De Morgan cases.
    gl::tests::ProverFixture ea(std::string("FTA"));
    ((ea.body).primedForContradiction) = true;
    ((ea.body).isActive) = true;
    depositPremiseAtMain(ea, "!(strictOrder[1,4,9,10])");

    bool k1 = false, k2 = false;
    findKShapes(ea, k1, k2);
    ASSERT_TRUE(k1);
    ASSERT_TRUE(k2);
    const std::vector<std::string> cases =
        pendingCasesAtMain(ea, "!(strictOrder[1,4,9,10])");
    ASSERT_EQ(cases.size(), std::size_t{2});
    ASSERT_TRUE(std::find(cases.begin(), cases.end(), "((=[9,10]))") != cases.end());
    ASSERT_TRUE(((ea.body).orImplications).sources.empty());
}

TEST(negated_and_demorgan, conjecture_premise_opens_the_or_inside_its_existences) {
    // bez1 = exists x,y in N: bezEq(a,x,b,y) or bezEq(b,y,a,x). Its OR sits
    // two existence levels deep; the conjecture-premise permission reaches it
    // and the pending goal counts as proof work.
    gl::tests::ProverFixture ea(std::string("FTA"));
    ((ea.body).goalPending) = true;
    depositGoalBlockPremiseAtMain(ea, "(bez1[1,4,5,6,9,11])", /*conjecturePremise=*/true);
    ASSERT_EQ(((ea.body).orDisjunctCount).count(), 1);
    ASSERT_EQ(((ea.body).orDisjunctCount).valueAt(1), 2);
    const std::vector<std::string> cases = singleCohortCases(ea.body);
    ASSERT_EQ(cases.size(), std::size_t{2});
    for (const std::string& c : cases)
        ASSERT_TRUE(c.find("bezEq[1,4,5,6,") != std::string::npos);
    ASSERT_EQ(((ea.body).pendingOrReleases).count(), 1);
    // The OR also travels to the descendants with the unconditional-ordis
    // flag (D-528), and its cases are forced
    // (D-527).
    ASSERT_EQ(((ea.body).lbMemory.pendingUnconditionalOrdis).count(), 1);
    ASSERT_EQ(((ea.body).lbMemory.forcedOrCohorts).count(), 1);
}

TEST(negated_and_demorgan, pending_goal_without_permission_parks_the_premise_or) {
    gl::tests::ProverFixture ea(std::string("FTA"));
    ((ea.body).goalPending) = true;
    depositGoalBlockPremiseAtMain(ea, "(bez1[1,4,5,6,9,11])", /*conjecturePremise=*/false);
    ASSERT_TRUE(((ea.body).orDisjunctCount).empty());
    ASSERT_TRUE(((ea.body).orPendingBranches).empty());
    ASSERT_FALSE(((((ea.body).overallHashMemory)).rejectedMapOrdis2).empty());
    ASSERT_TRUE(((ea.body).lbMemory.pendingUnconditionalOrdis).empty());
    ASSERT_TRUE(((ea.body).lbMemory.forcedOrCohorts).empty());
}

TEST(negated_and_demorgan, conjecture_premise_without_proof_work_stages_its_or_for_the_descendants) {
    // An earlier premise's block has no goal: the premise OR neither opens nor
    // parks there and is staged for the flag-6 unconditional-ordis mail
    // (D-528).
    gl::tests::ProverFixture ea(std::string("FTA"));
    depositGoalBlockPremiseAtMain(ea, "(bez1[1,4,5,6,9,11])", /*conjecturePremise=*/true);
    ASSERT_TRUE(((ea.body).orDisjunctCount).empty());
    ASSERT_TRUE(((((ea.body).overallHashMemory)).rejectedMapOrdis).empty());
    ASSERT_TRUE(((((ea.body).overallHashMemory)).rejectedMapOrdis2).empty());
    ASSERT_EQ(((ea.body).lbMemory.pendingUnconditionalOrdis).count(), 1);
    const std::string staged = ((ea.body).nameMap).decode(
        ((ea.body).lbMemory.pendingUnconditionalOrdis).decode(1));
    ASSERT_TRUE(gl::CompactExpressionSyntax::isOrCompactSpan(gl::StrSpan(staged)));
}

TEST(negated_and_demorgan, premise_or_without_proof_work_is_neither_opened_nor_parked) {
    gl::tests::ProverFixture ea(std::string("FTA"));
    depositGoalBlockPremiseAtMain(ea, "(bez1[1,4,5,6,9,11])", /*conjecturePremise=*/false);
    ASSERT_TRUE(((ea.body).orDisjunctCount).empty());
    ASSERT_TRUE(((((ea.body).overallHashMemory)).rejectedMapOrdis).empty());
    ASSERT_TRUE(((((ea.body).overallHashMemory)).rejectedMapOrdis2).empty());
}

TEST(negated_and_demorgan, goal_block_seeding_opens_the_premise_or_and_clears_the_mark) {
    // Row 121 of the FTA input: b divides t, a=t+r, bez1(b,r) => bez1(a,b).
    gl::tests::ProverFixture ea(std::string("FTA"));
    const std::string conj =
        "(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](preorder[1,5,9,10])"
        "(>[11,12](in3[10,11,12,4])(>[](bez1[1,4,5,6,9,11])(bez1[1,4,5,6,12,9])))))";
    ea.proofGridBuilder.addTheoremToMemory(conj, ea.body, 0, false, ea.globalDependencies);
    gl::Memory* anchorLB = ea.simpleMapStore.findChild(&ea.body, "(AnchorFTA[1,2,3,4,5,6,7,8])");
    ASSERT_TRUE(anchorLB != nullptr);
    gl::Memory* divLB = ea.simpleMapStore.findChild(anchorLB, "(preorder[1,5,9,10])");
    ASSERT_TRUE(divLB != nullptr);
    gl::Memory* sumLB = ea.simpleMapStore.findChild(divLB, "(in3[10,11,12,4])");
    ASSERT_TRUE(sumLB != nullptr);
    gl::Memory* goalLB = ea.simpleMapStore.findChild(sumLB, "(bez1[1,4,5,6,9,11])");
    ASSERT_TRUE(goalLB != nullptr);

    ASSERT_FALSE((*goalLB).goalPending);
    ASSERT_FALSE((*anchorLB).goalPending);
    ASSERT_FALSE(((*goalLB).intToBeProved).empty());
    ASSERT_EQ(((*goalLB).orDisjunctCount).count(), 1);
    ASSERT_EQ(((*goalLB).orDisjunctCount).valueAt(1), 2);
    const std::vector<std::string> cases = singleCohortCases(*goalLB);
    ASSERT_EQ(cases.size(), std::size_t{2});
    for (const std::string& c : cases)
        ASSERT_TRUE(c.find("bezEq[1,4,5,6,") != std::string::npos);
    // Intermediate premise blocks hold no goal and open nothing.
    ASSERT_TRUE(((*divLB).orDisjunctCount).empty());
    ASSERT_TRUE(((*sumLB).orDisjunctCount).empty());
}

TEST(negated_and_demorgan, earlier_premise_or_ships_with_the_unconditional_ordis_flag) {
    // FTA row 125: bez1(p,a), a*b=c, p|c => p|b. bez1(p,a) is the first premise;
    // its block has no goal, so its OR p*x=a*y+1 or a*y=p*x+1 opens nowhere
    // there. It is staged in that block only and fillMailOut ships it with
    // flag 6, so every descendant with proof work opens it
    // (D-528).
    gl::tests::ProverFixture ea(std::string("FTA"));
    const std::string conj =
        "(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](bez1[1,4,5,6,9,10])"
        "(>[11,12](in3[10,11,12,5])(>[](preorder[1,5,9,12])(preorder[1,5,9,11])))))";
    ea.proofGridBuilder.addTheoremToMemory(conj, ea.body, 0, false, ea.globalDependencies);
    gl::Memory* anchorLB = ea.simpleMapStore.findChild(&ea.body, "(AnchorFTA[1,2,3,4,5,6,7,8])");
    ASSERT_TRUE(anchorLB != nullptr);
    gl::Memory* bezLB = ea.simpleMapStore.findChild(anchorLB, "(bez1[1,4,5,6,9,10])");
    ASSERT_TRUE(bezLB != nullptr);
    gl::Memory* prodLB = ea.simpleMapStore.findChild(bezLB, "(in3[10,11,12,5])");
    ASSERT_TRUE(prodLB != nullptr);
    gl::Memory* goalLB = ea.simpleMapStore.findChild(prodLB, "(preorder[1,5,9,12])");
    ASSERT_TRUE(goalLB != nullptr);

    ASSERT_TRUE(((*bezLB).orDisjunctCount).empty());
    ASSERT_TRUE(((((*bezLB).overallHashMemory)).rejectedMapOrdis2).empty());
    ASSERT_EQ(((*bezLB).lbMemory.pendingUnconditionalOrdis).count(), 1);
    ASSERT_TRUE(((*anchorLB).lbMemory.pendingUnconditionalOrdis).empty());
    ASSERT_TRUE(((*prodLB).lbMemory.pendingUnconditionalOrdis).empty());
    ASSERT_TRUE(((*goalLB).lbMemory.pendingUnconditionalOrdis).empty());
    const std::string staged = ((*bezLB).nameMap).decode(
        ((*bezLB).lbMemory.pendingUnconditionalOrdis).decode(1));
    ASSERT_TRUE(gl::CompactExpressionSyntax::isOrCompactSpan(gl::StrSpan(staged)));

    ea.blockMailProcessor.fillMailOut(*bezLB);
    ASSERT_TRUE(((*bezLB).lbMemory.pendingUnconditionalOrdis).empty());
    const std::vector<char> blob =
        gl::Codec<gl::Mail>::serialize(((*bezLB).mailOut), ((*bezLB).mailOutInterner));
    gl::RoutingColdMail inbox;
    gl::Codec<gl::Mail>::deserializeInto(
        blob.data(), static_cast<int32_t>(blob.size()), inbox);
    int32_t code = 0, iter = 0;
    inbox.getStatementFlag(gl::packOriginKey(
        gl::mailInterner().intern(gl::StrSpan(staged)),
        gl::mailInterner().intern(gl::StrSpan("main", 4))), code, iter);
    ASSERT_EQ(code, gl::kMailStatementFlagUnconditionalOrdis);
}

TEST(negated_and_demorgan, forced_case_assumption_opens_with_force_witnesses) {
    // bez1(p,a) as a conjecture premise in its goal block: the OR
    // bezEq(p,x,a,y) or bezEq(a,y,p,x) opens, its first case is released with
    // the forced signal, and the absorb opens the case itself (bezEq is an
    // existence) with force_ witnesses: product facts appear at the branch
    // (D-527).
    gl::tests::ProverFixture ea(std::string("FTA"));
    ((ea.body).goalPending) = true;
    depositGoalBlockPremiseAtMain(ea, "(bez1[1,4,5,6,9,11])", /*conjecturePremise=*/true);
    ASSERT_EQ(((ea.body).lbMemory.forcedOrCohorts).count(), 1);
    // A current goal keeps the branch open: with no goal on its chain the
    // branch counts as resolved and is wiped in the same call.
    const auto goal = gl::encodeExpression(gl::StrSpan("(preorder[1,5,9,12])", 20),
        gl::StrSpan("main", 4), ((ea.body).nameMap));
    ((ea.body).intToBeProved).assignSet(gl::StatementKey{goal.originalId, goal.validityId}, nullptr, 0);
    ea.caseProcessor.drainPendingOrReleases(ea.body);
    const auto seeds = gl::decodeInternalMailStatements(
        ((ea.body).sameIterationInternalMail), ((ea.body).nameMap));
    ASSERT_EQ(seeds.size(), std::size_t{1});
    const std::string branch = seeds[0].first.validityName;
    ASSERT_TRUE(branch.find("_boundary_ordis_") != std::string::npos);
    ea.logicBlockOperations.standardProcessing(ea.body, nullptr,
        ((ea.body).sameIterationInternalMail), ((ea.body).nextIterationInternalMail), -1);
    const gl::NameId branchId = ((ea.body).nameMap).lookup(gl::StrSpan(branch));
    ASSERT_TRUE(branchId != 0);
    bool forcedProduct = false;
    for (int32_t i = 0; i < ((ea.body).intEncodedStatements).size(); ++i) {
        const gl::IntEncodedExpr& row = ((ea.body).intEncodedStatements)[i];
        if (row.validityId != branchId) continue;
        const std::string text = ((ea.body).nameMap).decode(row.originalId);
        if (text.rfind("(in3[", 0) == 0 && text.find("force_lev_") != std::string::npos)
            forcedProduct = true;
    }
    ASSERT_TRUE(forcedProduct);
}
