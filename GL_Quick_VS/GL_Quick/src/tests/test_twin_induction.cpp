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
/// @brief Unit tests for the induction of a conjecture with twin premises
///        (D-587).
///
/// @details
/// Two non-anchor premises are twins when they are the only two premises of
/// one predicate and the conjecture is symmetric in them (exchanging the two
/// gives the same lemma after normalization; an equality head reads
/// unordered). A digit variable k of the first twin then has a partner l, the
/// variable at the same slot of the second twin. The grid builds triads on k
/// only, the weak hypothesis leaves l universal, and the weak step and zero
/// blocks record l. Covers `ProofGridBuilder::twinInductionPartners`,
/// `ProofGridBuilder::findImmutableArgs` with free digits and the grid built
/// for FTA2's full I5.

#include "../memory.hpp"
#include "test_harness.hpp"
#include "prover_fixture.hpp"
#include "../prover.hpp"

#include <algorithm>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <cassert>

namespace {
    /// @brief Span over a string literal (tests only).
    gl::StrSpan sp(const char* s) {
        return gl::StrSpan(s, static_cast<int32_t>(std::strlen(s)));
    }

    /// @brief FTA2's full I5: two prime factorizations of one n are equal;
    ///        the lengths 9 and 12 may differ.
    const char* const kFullI5 =
        "(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10,11](primeFactorSeq[1,3,4,5,2,6,7,9,10,11])"
        "(>[12,13](primeFactorSeq[1,3,4,5,2,6,7,12,13,11])(=[10,13]))))";

    /// @brief The twin partners of @p conjecture over its own digit arguments.
    std::map<std::string, std::string> partnersOf(gl::tests::ProverFixture& ea, const std::string& conjecture) {
        gl::StrSpan digits[gl::ExecutionParameters::MAX_ADMISSION_REM_ARGS];
        const int32_t digitN = ea.scopeTypes.findDigitArgs(conjecture, ea.anchorInfo, digits,
            gl::ExecutionParameters::MAX_ADMISSION_REM_ARGS);
        return ea.proofGridBuilder.twinInductionPartners(conjecture, digits, digitN);
    }
} // namespace

// Exchanging the full I5's premises gives the same lemma after
// normalization, the equality head read unordered: 9 has the partner 12.
// The same holds for the injectivity of s, whose head is an equality of the
// two partners themselves.
TEST(twin_induction, partners_of_symmetric_twin_premises) {
    gl::tests::ProverFixture ea(std::string("FTA"));
    const std::map<std::string, std::string> fullI5 = partnersOf(ea, kFullI5);
    ASSERT_EQ(fullI5.size(), std::size_t{1});
    ASSERT_EQ(fullI5.at("9"), std::string("12"));

    const std::string injective =
        "(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](in2[9,10,3])(>[11](in2[11,10,3])(=[9,11]))))";
    const std::map<std::string, std::string> successor = partnersOf(ea, injective);
    ASSERT_EQ(successor.size(), std::size_t{1});
    ASSERT_EQ(successor.at("9"), std::string("11"));
}

// No partner when the exchange changes the lemma, when the second twin
// holds the induction variable itself, or when a predicate repeats more than
// twice or two predicates repeat.
TEST(twin_induction, no_partner_without_twins) {
    gl::tests::ProverFixture ea(std::string("FTA"));
    const std::string asymmetric =
        "(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10,11](primeFactorSeq[1,3,4,5,2,6,7,9,10,11])"
        "(>[12,13](primeFactorSeq[1,3,4,5,2,6,7,12,13,11])(preorder[1,4,9,12]))))";
    ASSERT_TRUE(partnersOf(ea, asymmetric).empty());

    const std::string equalLength =
        "(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10,11](primeFactorSeq[1,3,4,5,2,6,7,9,10,11])"
        "(>[12](primeFactorSeq[1,3,4,5,2,6,7,9,12,11])(=[10,12]))))";
    ASSERT_TRUE(partnersOf(ea, equalLength).empty());

    const std::string threeTimes =
        "(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](in2[9,10,3])(>[11](in2[11,10,3])"
        "(>[12](in2[12,10,3])(=[9,11])))))";
    ASSERT_TRUE(partnersOf(ea, threeTimes).empty());

    const std::string twoRepeated =
        "(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](in2[9,10,3])(>[11](in2[11,10,3])"
        "(>[12](in[12,1])(>[13](in[13,1])(=[9,11]))))))";
    ASSERT_TRUE(partnersOf(ea, twoRepeated).empty());
}

// The weak hypothesis on 9 fixes 12 by default; with 12 free it fixes
// nothing (n stays universal either way: neither premise has all of its
// inputs fixed).
TEST(twin_induction, free_digits_leave_the_immutables) {
    gl::tests::ProverFixture ea(std::string("FTA"));
    const std::set<std::string> fixed = ea.proofGridBuilder.findImmutableArgs(kFullI5, "9");
    ASSERT_TRUE(fixed == std::set<std::string>{ "12" });
    const std::set<std::string> free = ea.proofGridBuilder.findImmutableArgs(kFullI5, "9",
        std::set<std::string>{ "12" });
    ASSERT_TRUE(free.empty());
}

// The full I5's grid: one weak triad on 9 (rec0), none on the partner 12 and
// no strong triad (its length sits in an operator with a numeric output,
// D-594); the weak hypothesis binds 12 at its second
// premise; the weak step block and its zero block record the partner.
TEST(twin_induction, grid_builds_one_induction_with_the_partner_free) {
    gl::tests::ProverFixture ea(std::string("FTA"));
    ea.proofGridBuilder.addTheoremToMemory(kFullI5, ea.body, 0, false, ea.globalDependencies);
    gl::Memory* anchorLB = ea.simpleMapStore.findChild(&ea.body, "(AnchorFTA[1,2,3,4,5,6,7,8])");
    ASSERT_TRUE(anchorLB != nullptr);
    gl::Memory* firstLB = ea.simpleMapStore.findChild(anchorLB, "(primeFactorSeq[1,3,4,5,2,6,7,9,10,11])");
    ASSERT_TRUE(firstLB != nullptr);
    gl::Memory* goalLB = ea.simpleMapStore.findChild(firstLB, "(primeFactorSeq[1,3,4,5,2,6,7,12,13,11])");
    ASSERT_TRUE(goalLB != nullptr);

    gl::Memory* weak = ea.simpleMapStore.findChild(goalLB, "(in2[rec0,9,3])");
    ASSERT_TRUE(weak != nullptr);
    for (const char* key : { "(in2[rec1,9,3])", "(in2[rec0,12,3])", "(in2[rec1,12,3])", "(in2[rec2,12,3])",
                             "(in2[rec3,12,3])", "(in2[rec2,9,3])", "(=[s(rec1),2])" })
        ASSERT_TRUE(ea.simpleMapStore.findChild(goalLB, key) == nullptr);
    ASSERT_EQ(goalLB->recursionCounter, 1);

    const std::string hypothesis =
        "(>[10,11](primeFactorSeq[u_1,u_3,u_4,u_5,u_2,u_6,u_7,u_rec,10,11])"
        "(>[12,13](primeFactorSeq[u_1,u_3,u_4,u_5,u_2,u_6,u_7,12,13,11])(=[10,13])))";
    ASSERT_NE(weak->recursionAuxyImplId, 0);
    ASSERT_EQ(weak->ruleInterner.lookup(gl::StrSpan(hypothesis)), weak->recursionAuxyImplId);

    ASSERT_NE(weak->twinPartnerId, gl::NameId(0));
    ASSERT_EQ(weak->nameMap.decode(weak->twinPartnerId), std::string("12"));
    gl::Memory* zero = ea.simpleMapStore.findChild(goalLB, "(=[s(rec0),2])");
    ASSERT_TRUE(zero != nullptr);
    ASSERT_NE(zero->twinPartnerId, gl::NameId(0));
    ASSERT_EQ(zero->nameMap.decode(zero->twinPartnerId), std::string("12"));
    ASSERT_EQ(std::get<0>(ea.globalDependencies.originalInductionVariableMap[0]), "9");
    ASSERT_EQ(ea.globalDependencies.originalInductionVariableMap.count(1), 0u);
}

// A conjecture without twins keeps today's weak triad: the equal-length I5
// builds its weak step on 9 and records no partner.
TEST(twin_induction, grid_without_twins_records_no_partner) {
    gl::tests::ProverFixture ea(std::string("FTA"));
    const std::string equalLength =
        "(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10,11](primeFactorSeq[1,3,4,5,2,6,7,9,10,11])"
        "(>[12](primeFactorSeq[1,3,4,5,2,6,7,9,12,11])(=[10,12]))))";
    ea.proofGridBuilder.addTheoremToMemory(equalLength, ea.body, 0, false, ea.globalDependencies);
    gl::Memory* anchorLB = ea.simpleMapStore.findChild(&ea.body, "(AnchorFTA[1,2,3,4,5,6,7,8])");
    gl::Memory* firstLB = ea.simpleMapStore.findChild(anchorLB, "(primeFactorSeq[1,3,4,5,2,6,7,9,10,11])");
    gl::Memory* goalLB = ea.simpleMapStore.findChild(firstLB, "(primeFactorSeq[1,3,4,5,2,6,7,9,12,11])");
    ASSERT_TRUE(goalLB != nullptr);
    gl::Memory* weak = ea.simpleMapStore.findChild(goalLB, "(in2[rec0,9,3])");
    ASSERT_TRUE(weak != nullptr);
    ASSERT_EQ(weak->twinPartnerId, gl::NameId(0));
    gl::Memory* zero = ea.simpleMapStore.findChild(goalLB, "(=[s(rec0),2])");
    ASSERT_TRUE(zero != nullptr);
    ASSERT_EQ(zero->twinPartnerId, gl::NameId(0));
}

namespace {
    /// @brief The fixture's root block as a twin block armed for @p experiment
    ///        with partner 12, zero 2 and successor 3, holding proof work.
    void armTwinBlock(gl::tests::ProverFixture& ea, gl::TwinExperiment experiment) {
        ea.body.level = 3;
        ea.body.goalPending = true;
        ea.body.twinPartnerId = ea.body.nameMap.encode(sp("12"));
        ea.body.twinZeroId = ea.body.nameMap.encode(sp("2"));
        ea.body.twinSuccessorId = ea.body.nameMap.encode(sp("3"));
        ea.body.twinExperiment = experiment;
    }

    /// @brief The compiled zero-or-successor OR on 12, "inSequence[1,12,3] or
    ///        12 = 2", as the compact the compiler mints for it.
    std::string zeroOrSuccessorCompact(gl::tests::ProverFixture& ea) {
        std::string text = "(>[12](in[12,1])!(&(>[13](in[13,1])!(in2[13,12,3]))!(=[12,2])))";
        ea.structuralCompiler.precompileStructuralOperators(text);
        std::vector<std::tuple<std::string, std::vector<std::string>, std::set<std::string>>> chain;
        ce::CoreExpressionMap unused;
        const std::string head = ce::disintegrateImplication(text, chain, unused);
        assert(head.rfind("(or", 0) == 0 && "zeroOrSuccessorCompact: the head compiles to an OR compact");
        return head;
    }

    /// @brief Deposit @p expr at the root block's main scope, status 0.
    void depositAtMain(gl::tests::ProverFixture& ea, const std::string& expr) {
        const int lv[1] = { 3 };
        const gl::TransientOrigin origin{ true, gl::OriginTag::taskFormulation, nullptr, 0 };
        ea.statementMemory.addExprToMemoryBlock(gl::StrSpan(expr), ea.body, 0, 0, lv, 1,
            origin, -1, -1, gl::StrSpan("main", 4), false);
    }

    /// @brief The decoded pending cases of @p cohort in @p block.
    std::vector<std::string> pendingCases(gl::Memory& block, int32_t cohort) {
        std::vector<std::string> cases;
        const int32_t row = block.orPendingBranches.lookup(cohort);
        for (int32_t j = 0; row != 0 && j < block.orPendingBranches.runLen(row); ++j) {
            const gl::StrSpan w = block.lbStateInterner.decodeView(block.orPendingBranches.valueAt(row, j));
            cases.emplace_back(w.ptr, static_cast<std::size_t>(w.len));
        }
        return cases;
    }
} // namespace

// The full I5's weak install arms "successor case, then zero case"
// (two-marker start, the second premise free where the first holds rec);
// its zero block is armed for the successor case at creation; no strong
// triad is built (D-594) and the blocks of a
// conjecture without twins are not armed.
TEST(twin_experiment, grid_arms_the_step_and_zero_blocks) {
    gl::tests::ProverFixture ea(std::string("FTA"));
    ea.proofGridBuilder.addTheoremToMemory(kFullI5, ea.body, 0, false, ea.globalDependencies);
    gl::Memory* anchorLB = ea.simpleMapStore.findChild(&ea.body, "(AnchorFTA[1,2,3,4,5,6,7,8])");
    gl::Memory* firstLB = ea.simpleMapStore.findChild(anchorLB, "(primeFactorSeq[1,3,4,5,2,6,7,9,10,11])");
    gl::Memory* goalLB = ea.simpleMapStore.findChild(firstLB, "(primeFactorSeq[1,3,4,5,2,6,7,12,13,11])");
    ASSERT_TRUE(goalLB != nullptr);
    gl::Memory* weak = ea.simpleMapStore.findChild(goalLB, "(in2[rec0,9,3])");
    ASSERT_TRUE(weak != nullptr);
    ASSERT_TRUE(ea.simpleMapStore.findChild(goalLB, "(in2[rec1,9,3])") == nullptr);
    ASSERT_TRUE(weak->twinExperiment == gl::TwinExperiment::successorThenZero);
    ASSERT_EQ(weak->nameMap.decode(weak->twinZeroId), std::string("2"));
    ASSERT_EQ(weak->nameMap.decode(weak->twinSuccessorId), std::string("3"));
    gl::Memory* zero = ea.simpleMapStore.findChild(goalLB, "(=[s(rec0),2])");
    ASSERT_TRUE(zero != nullptr);
    ASSERT_TRUE(zero->twinExperiment == gl::TwinExperiment::successorCase);

    gl::tests::ProverFixture other(std::string("FTA"));
    const std::string equalLength =
        "(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10,11](primeFactorSeq[1,3,4,5,2,6,7,9,10,11])"
        "(>[12](primeFactorSeq[1,3,4,5,2,6,7,9,12,11])(=[10,12]))))";
    other.proofGridBuilder.addTheoremToMemory(equalLength, other.body, 0, false, other.globalDependencies);
    gl::Memory* a2 = other.simpleMapStore.findChild(&other.body, "(AnchorFTA[1,2,3,4,5,6,7,8])");
    gl::Memory* f2 = other.simpleMapStore.findChild(a2, "(primeFactorSeq[1,3,4,5,2,6,7,9,10,11])");
    gl::Memory* g2 = other.simpleMapStore.findChild(f2, "(primeFactorSeq[1,3,4,5,2,6,7,9,12,11])");
    gl::Memory* w2 = other.simpleMapStore.findChild(g2, "(in2[rec0,9,3])");
    ASSERT_TRUE(w2 != nullptr);
    ASSERT_TRUE(w2->twinExperiment == gl::TwinExperiment::none);
}

// The experiment recognizes the partner's zero-or-successor OR at main only
// and starts both armed kinds with the existence leaf; an unarmed block,
// another scope, another variable or a third leaf gives -1.
TEST(twin_experiment, leaf_selection) {
    gl::tests::ProverFixture ea(std::string("FTA"));
    gl::ScratchArena& strings = gl::scratchArenas().forSlot(gl::scratchArenas().slotCount() - 1);
    const gl::StrSpan leaves[2] = { sp("(inSequence[u_1,u_12,u_3])"), sp("(=[u_12,u_2])") };
    const gl::StrSpan wrongPartner[2] = { sp("(inSequence[u_1,u_13,u_3])"), sp("(=[u_13,u_2])") };
    const gl::StrSpan wrongSuccessor[2] = { sp("(inSequence[u_1,u_12,u_4])"), sp("(=[u_12,u_2])") };
    const gl::StrSpan three[3] = { sp("(inSequence[u_1,u_12,u_3])"), sp("(=[u_12,u_2])"), sp("(=[u_12,u_6])") };

    ASSERT_EQ(ea.caseProcessor.twinExperimentLeaf(ea.body, leaves, 2, sp("main"), strings), -1);
    armTwinBlock(ea, gl::TwinExperiment::successorThenZero);
    ASSERT_EQ(ea.caseProcessor.twinExperimentLeaf(ea.body, leaves, 2, sp("main"), strings), 0);
    ASSERT_EQ(ea.caseProcessor.twinExperimentLeaf(ea.body, leaves, 2, sp("main_boundary_x"), strings), -1);
    ASSERT_EQ(ea.caseProcessor.twinExperimentLeaf(ea.body, wrongPartner, 2, sp("main"), strings), -1);
    ASSERT_EQ(ea.caseProcessor.twinExperimentLeaf(ea.body, wrongSuccessor, 2, sp("main"), strings), -1);
    ASSERT_EQ(ea.caseProcessor.twinExperimentLeaf(ea.body, three, 3, sp("main"), strings), -1);
    ea.body.twinExperiment = gl::TwinExperiment::successorCase;
    ASSERT_EQ(ea.caseProcessor.twinExperimentLeaf(ea.body, leaves, 2, sp("main"), strings), 0);
}

// In the armed step block the zero-or-successor OR opens one ordinary cohort
// at main that starts with the successor case: the drain releases that case
// and records its scope; once the case is resolved and the cohort staged
// again, the drain releases the zero case and the recorded scope stays.
TEST(twin_experiment, step_block_releases_the_successor_case_then_the_zero_case) {
    gl::tests::ProverFixture ea(std::string("FTA"));
    ea.parameters.trackHistory = false;
    armTwinBlock(ea, gl::TwinExperiment::successorThenZero);
    depositAtMain(ea, zeroOrSuccessorCompact(ea));
    ASSERT_EQ(ea.body.orDisjunctCount.count(), 1);
    const int32_t cohort = ea.body.orDisjunctCount.keyAt(1);
    ASSERT_TRUE(ea.body.lbMemory.experimentOrCohorts.lookup(cohort) == 0);
    ASSERT_EQ(ea.body.twinCohortId, cohort);
    const int32_t* starter = ea.body.orStarterPick.find(cohort);
    ASSERT_TRUE(starter != nullptr);
    ASSERT_TRUE(ea.body.lbStateInterner.decode(*starter).find("inSequence[1,12,3]") != std::string::npos);

    ea.caseProcessor.drainPendingOrReleases(ea.body);
    const auto seeds = gl::decodeInternalMailStatements(ea.body.sameIterationInternalMail, ea.body.nameMap);
    ASSERT_EQ(seeds.size(), std::size_t{1});
    ASSERT_EQ(seeds[0].first.original, std::string("(inSequence[1,12,3])"));
    ASSERT_TRUE(seeds[0].first.validityName.find("_boundary_ordis_") != std::string::npos);
    ASSERT_TRUE(ea.body.twinCaseScopeId != gl::NameId(0));
    ASSERT_EQ(ea.body.nameMap.decode(ea.body.twinCaseScopeId), seeds[0].first.validityName);
    ASSERT_TRUE(ea.body.orStarterPick.find(cohort) == nullptr);

    ea.body.sameIterationInternalMail.clear();
    ea.body.pendingOrReleases.mint(cohort);
    ea.caseProcessor.drainPendingOrReleases(ea.body);
    const auto second = gl::decodeInternalMailStatements(ea.body.sameIterationInternalMail, ea.body.nameMap);
    ASSERT_EQ(second.size(), std::size_t{1});
    ASSERT_EQ(second[0].first.original, std::string("(=[12,2])"));
    ASSERT_EQ(ea.body.nameMap.decode(ea.body.twinCaseScopeId), seeds[0].first.validityName);
    ASSERT_TRUE(pendingCases(ea.body, cohort).empty());
}

// A zero block's experiment starts with the successor case and keeps its
// cohort an experiment cohort, which releases no other case; the drain
// releases that case and records its scope.
TEST(twin_experiment, zero_block_releases_the_successor_case) {
    gl::tests::ProverFixture ea(std::string("FTA"));
    ea.parameters.trackHistory = false;
    armTwinBlock(ea, gl::TwinExperiment::successorCase);
    depositAtMain(ea, zeroOrSuccessorCompact(ea));
    ASSERT_EQ(ea.body.orDisjunctCount.count(), 1);
    const int32_t cohort = ea.body.orDisjunctCount.keyAt(1);
    ASSERT_TRUE(ea.body.lbMemory.experimentOrCohorts.lookup(cohort) != 0);
    ASSERT_EQ(ea.body.twinCohortId, 0);
    const int32_t* starter = ea.body.orStarterPick.find(cohort);
    ASSERT_TRUE(starter != nullptr);
    ASSERT_TRUE(ea.body.lbStateInterner.decode(*starter).find("inSequence[1,12,3]") != std::string::npos);

    ea.caseProcessor.drainPendingOrReleases(ea.body);
    const auto seeds = gl::decodeInternalMailStatements(ea.body.sameIterationInternalMail, ea.body.nameMap);
    ASSERT_EQ(seeds.size(), std::size_t{1});
    ASSERT_EQ(seeds[0].first.original, std::string("(inSequence[1,12,3])"));
    ASSERT_TRUE(seeds[0].first.validityName.find("_boundary_ordis_") != std::string::npos);
    ASSERT_TRUE(ea.body.twinCaseScopeId != gl::NameId(0));
    ASSERT_EQ(ea.body.nameMap.decode(ea.body.twinCaseScopeId), seeds[0].first.validityName);
}

// Without an armed experiment the same OR opens no case split.
TEST(twin_experiment, unarmed_block_opens_nothing) {
    gl::tests::ProverFixture ea(std::string("FTA"));
    ea.parameters.trackHistory = false;
    armTwinBlock(ea, gl::TwinExperiment::none);
    depositAtMain(ea, zeroOrSuccessorCompact(ea));
    ASSERT_TRUE(ea.body.orDisjunctCount.empty());
    ASSERT_TRUE(ea.body.lbMemory.experimentOrCohorts.empty());
}

namespace {
    /// @brief The bindings of every instruction labelled @p label in @p m,
    ///        with whether each key holding one is connected lexically through
    ///        @p rec or @p variable (tests only).
    std::vector<std::pair<gl::IntegrationForbiddenSets, bool>> copiesLabelled(gl::Memory& m,
            const std::string& label, const char* rec, const char* variable) {
        std::vector<std::pair<gl::IntegrationForbiddenSets, bool>> out;
        const auto& keys = m.overallHashMemory.admissionMapIntegration;
        gl::ScratchArena& arena = gl::genScratchArenas().forSlot(gl::genScratchArenas().slotCount() - 1);
        for (int32_t id = 1; id <= keys.count(); ++id) {
            if (keys.lookup(keys.keyAt(id)) == 0) continue;
            gl::ScratchScope frame(arena);
            gl::ArenaIntegrationMap entries(&arena, &m.valueInterner);
            entries.buildFromCold(keys, keys.keyAt(id));
            const gl::StrSpan key = m.templateInterner.decodeView(
                static_cast<int32_t>(static_cast<uint64_t>(keys.keyAt(id)) >> 32));
            const bool connected = gl::ScopeSyntax::keyHoldsInductionRole(key, sp(rec), sp(variable));
            for (int32_t i = 0; i < entries.entries.size(); ++i) {
                gl::WorkInstruction wi(&arena);
                entries.loadInstructionInto(i, wi, m.valueInterner);
                if (wi.markedGoal().toStdString() == label) out.emplace_back(wi.forbidden, connected);
            }
        }
        return out;
    }

    /// @brief Deposit @p expr at @p scope of @p block (by default `main`),
    ///        status 0, level 3.
    void depositIn(gl::tests::ProverFixture& ea, gl::Memory& block, const char* expr,
                   const std::string& scope = "main") {
        const int lv[1] = { 3 };
        const gl::TransientOrigin origin{ true, gl::OriginTag::taskFormulation, nullptr, 0 };
        ea.statementMemory.addExprToMemoryBlock(sp(expr), block, 0, 0, lv, 1, origin, -1, -1,
            gl::StrSpan(scope), false);
    }

    /// @brief The history tags @p mail of @p block stages for @p expr at
    ///        @p scope; empty when the mail holds no such row (tests only).
    std::vector<std::string> stagedTagsIn(gl::Memory& block, const gl::ColdMail& mail, const std::string& expr,
                                          const std::string& scope) {
        bool staged = false;
        for (const auto& row : gl::decodeInternalMailStatements(mail, block.nameMap))
            staged |= row.first.original == expr && row.first.validityName == scope;
        std::vector<std::string> tags;
        if (!staged) return tags;
        tags.emplace_back("staged");
        for (const auto& row : gl::decodeInternalMailOrigins(mail, block.originInterner)) {
            if (row.first.original != expr || row.first.validityName != scope) continue;
            for (const auto& line : row.second) tags.push_back(line.first);
        }
        return tags;
    }

    /// @brief The history tags @p block's same-iteration internal mail stages
    ///        for @p expr at @p scope (tests only).
    std::vector<std::string> stagedTags(gl::Memory& block, const std::string& expr, const std::string& scope) {
        return stagedTagsIn(block, block.sameIterationInternalMail, expr, scope);
    }

    /// @brief The scopes of the keys holding an instruction labelled
    ///        @p label in @p m, one per key (tests only).
    std::vector<gl::NameId> keyScopesLabelled(gl::Memory& m, const std::string& label) {
        std::vector<gl::NameId> out;
        const auto& keys = m.overallHashMemory.admissionMapIntegration;
        gl::ScratchArena& arena = gl::genScratchArenas().forSlot(gl::genScratchArenas().slotCount() - 1);
        for (int32_t id = 1; id <= keys.count(); ++id) {
            if (keys.lookup(keys.keyAt(id)) == 0) continue;
            gl::ScratchScope frame(arena);
            gl::ArenaIntegrationMap entries(&arena, &m.valueInterner);
            entries.buildFromCold(keys, keys.keyAt(id));
            for (int32_t i = 0; i < entries.entries.size(); ++i) {
                gl::WorkInstruction wi(&arena);
                entries.loadInstructionInto(i, wi, m.valueInterner);
                if (wi.markedGoal().toStdString() != label) continue;
                out.push_back(static_cast<gl::NameId>(static_cast<uint32_t>(keys.keyAt(id))));
                break;
            }
        }
        return out;
    }

    /// @brief The weak step block `(in2[rec0,9,3])` of @p conjecture's goal
    ///        block below the anchor and the first premise (tests only).
    gl::Memory* weakBlockOf(gl::tests::ProverFixture& ea, const char* firstPremise, const char* goalPremise) {
        gl::Memory* anchorLB = ea.simpleMapStore.findChild(&ea.body, "(AnchorFTA[1,2,3,4,5,6,7,8])");
        gl::Memory* firstLB = anchorLB ? ea.simpleMapStore.findChild(anchorLB, firstPremise) : nullptr;
        gl::Memory* goalLB = firstLB ? ea.simpleMapStore.findChild(firstLB, goalPremise) : nullptr;
        return goalLB ? ea.simpleMapStore.findChild(goalLB, "(in2[rec0,9,3])") : nullptr;
    }
} // namespace

// The full I5's weak install files the first twin premise's two-marker goal
// as the copy bound to its premise (level 1, forbidding 12, 13 and level 2);
// the partner's predecessor q reaching `main` stages the copy `(=[q,rec2])`
// there and files the second goal, the first with `rec2` for `rec`, bound to
// the second premise (level 2, forbidding 9, 10 and level 1) under keys
// connected through `rec2` or 12; a later predecessor files nothing.
TEST(twin_second_goal, install_and_predecessor_file_the_two_goals) {
    gl::tests::ProverFixture ea(std::string("FTA"));
    ea.proofGridBuilder.addTheoremToMemory(kFullI5, ea.body, 0, false, ea.globalDependencies);
    gl::Memory* weak = weakBlockOf(ea, "(primeFactorSeq[1,3,4,5,2,6,7,9,10,11])",
        "(primeFactorSeq[1,3,4,5,2,6,7,12,13,11])");
    ASSERT_TRUE(weak != nullptr);
    ASSERT_TRUE(weak->premiseSplitFiled);
    ASSERT_EQ(weak->nameMap.decode(weak->twinFirstGoalId),
        std::string("(primeFactorSeq[1,3,4,5,2,6,7,rec,marker1,marker2])"));
    ASSERT_EQ(weak->twinFirstLevel, 1);
    ASSERT_EQ(weak->twinSecondLevel, 2);
    ASSERT_EQ(weak->twinPredecessorId, gl::NameId(0));
    const uint64_t pVariables = (uint64_t{1} << 12) | (uint64_t{1} << 13);
    const uint64_t qVariables = (uint64_t{1} << 9) | (uint64_t{1} << 10);

    const auto first = copiesLabelled(*weak, "(primeFactorSeq[1,3,4,5,2,6,7,rec,marker1,marker2])", "rec", "9");
    ASSERT_FALSE(first.empty());
    for (const auto& copy : first) {
        ASSERT_EQ(copy.first.premiseLevel, 1);
        ASSERT_EQ(copy.first.variables, pVariables);
        ASSERT_EQ(copy.first.levels, uint64_t{1} << 2);
    }

    depositIn(ea, *weak, "(in2[it_0_lev_3_5,12,3])");
    ASSERT_EQ(weak->nameMap.decode(weak->twinPredecessorId), std::string("rec2"));
    ASSERT_TRUE(stagedTags(*weak, "(=[it_0_lev_3_5,rec2])", "main")
        == (std::vector<std::string>{ "staged", "variable copy" }));
    const char* const secondLabel = "(primeFactorSeq[1,3,4,5,2,6,7,rec2,marker1,marker2])";
    const auto second = copiesLabelled(*weak, secondLabel, "rec2", "12");
    ASSERT_FALSE(second.empty());
    for (const auto& copy : second) {
        ASSERT_EQ(copy.first.premiseLevel, 2);
        ASSERT_EQ(copy.first.variables, qVariables);
        ASSERT_EQ(copy.first.levels, uint64_t{1} << 1);
        ASSERT_TRUE(copy.second);
    }
    for (const gl::NameId scope : keyScopesLabelled(*weak, secondLabel)) ASSERT_EQ(scope, gl::NameMap::MAIN_ID);

    depositIn(ea, *weak, "(in2[it_0_lev_3_6,12,3])");
    ASSERT_EQ(weak->nameMap.decode(weak->twinPredecessorId), std::string("rec2"));
    ASSERT_TRUE(stagedTags(*weak, "(=[it_0_lev_3_6,rec2])", "main").empty());
    ASSERT_EQ(copiesLabelled(*weak, secondLabel, "rec2", "12").size(), second.size());
}

// In the twin successor case the partner's predecessor q stages the copy
// `(=[q,rec2])` in the case and files the second goal there; at another case
// scope it files nothing.
TEST(twin_second_goal, case_predecessor_files_the_goal_in_the_case) {
    gl::tests::ProverFixture ea(std::string("FTA"));
    ea.proofGridBuilder.addTheoremToMemory(kFullI5, ea.body, 0, false, ea.globalDependencies);
    gl::Memory* weak = weakBlockOf(ea, "(primeFactorSeq[1,3,4,5,2,6,7,9,10,11])",
        "(primeFactorSeq[1,3,4,5,2,6,7,12,13,11])");
    ASSERT_TRUE(weak != nullptr);
    const std::string caseScope = "main_boundary_ordis_(or1[1,12,3,2])_((inSequence[1,12,3]))";
    const std::string zeroCase = "main_boundary_ordis_(or1[1,12,3,2])_((=[12,2]))";
    weak->twinCaseScopeId = weak->nameMap.encode(gl::StrSpan(caseScope));
    ea.canonicalizer.initializeCanonicalScope(*weak, weak->twinCaseScopeId);
    ea.canonicalizer.initializeCanonicalScope(*weak, weak->nameMap.encode(gl::StrSpan(zeroCase)));

    depositIn(ea, *weak, "(in2[it_0_lev_3_5,12,3])", zeroCase);
    ASSERT_EQ(weak->twinPredecessorId, gl::NameId(0));
    ASSERT_TRUE(stagedTags(*weak, "(=[it_0_lev_3_5,rec2])", zeroCase).empty());

    depositIn(ea, *weak, "(in2[it_0_lev_3_5,12,3])", caseScope);
    ASSERT_EQ(weak->nameMap.decode(weak->twinPredecessorId), std::string("rec2"));
    ASSERT_TRUE(stagedTags(*weak, "(=[it_0_lev_3_5,rec2])", caseScope)
        == (std::vector<std::string>{ "staged", "variable copy" }));
    ASSERT_TRUE(stagedTags(*weak, "(=[it_0_lev_3_5,rec2])", "main").empty());
    const char* const secondLabel = "(primeFactorSeq[1,3,4,5,2,6,7,rec2,marker1,marker2])";
    ASSERT_FALSE(copiesLabelled(*weak, secondLabel, "rec2", "12").empty());
    const std::vector<gl::NameId> scopes = keyScopesLabelled(*weak, secondLabel);
    ASSERT_FALSE(scopes.empty());
    for (const gl::NameId scope : scopes) ASSERT_EQ(scope, weak->twinCaseScopeId);
    // The partner's value demands belong to base-case blocks only.
    ASSERT_FALSE(weak->twinValueDemandsFiled);
}

// Inside an absorb the copy rides the absorb's output channel
// (`Memory::replayMailDuringAbsorb`), never the drained input that the absorb
// clears afterwards.
TEST(twin_second_goal, copy_inside_an_absorb_rides_its_output_channel) {
    gl::tests::ProverFixture ea(std::string("FTA"));
    ea.proofGridBuilder.addTheoremToMemory(kFullI5, ea.body, 0, false, ea.globalDependencies);
    gl::Memory* weak = weakBlockOf(ea, "(primeFactorSeq[1,3,4,5,2,6,7,9,10,11])",
        "(primeFactorSeq[1,3,4,5,2,6,7,12,13,11])");
    ASSERT_TRUE(weak != nullptr);
    weak->replayMailDuringAbsorb = &weak->nextIterationInternalMail;
    depositIn(ea, *weak, "(in2[it_0_lev_3_5,12,3])");
    weak->replayMailDuringAbsorb = nullptr;
    ASSERT_EQ(weak->nameMap.decode(weak->twinPredecessorId), std::string("rec2"));
    ASSERT_TRUE(stagedTagsIn(*weak, weak->nextIterationInternalMail, "(=[it_0_lev_3_5,rec2])", "main")
        == (std::vector<std::string>{ "staged", "variable copy" }));
    ASSERT_TRUE(stagedTags(*weak, "(=[it_0_lev_3_5,rec2])", "main").empty());
}

namespace {
    /// @brief The base-case block `(=[s(rec0),2])` of the full I5's goal
    ///        block (tests only).
    gl::Memory* zeroBlockOfFullI5(gl::tests::ProverFixture& ea) {
        gl::Memory* anchorLB = ea.simpleMapStore.findChild(&ea.body, "(AnchorFTA[1,2,3,4,5,6,7,8])");
        gl::Memory* firstLB = anchorLB
            ? ea.simpleMapStore.findChild(anchorLB, "(primeFactorSeq[1,3,4,5,2,6,7,9,10,11])") : nullptr;
        gl::Memory* goalLB = firstLB
            ? ea.simpleMapStore.findChild(firstLB, "(primeFactorSeq[1,3,4,5,2,6,7,12,13,11])") : nullptr;
        return goalLB ? ea.simpleMapStore.findChild(goalLB, "(=[s(rec0),2])") : nullptr;
    }

    /// @brief The partner 12's successor case in @p block, minted below
    ///        `main` with its canonical scope and recorded as the released
    ///        case (tests only).
    std::string releaseSuccessorCase(gl::tests::ProverFixture& ea, gl::Memory& block) {
        const gl::NameId id = block.nameMap.encodePush(gl::NameMap::MAIN_ID,
            std::string("ordis_(or1[1,12,3,2])_((inSequence[1,12,3]))"));
        ea.canonicalizer.initializeCanonicalScope(block, id);
        block.twinCaseScopeId = id;
        return block.nameMap.decode(id);
    }

    /// @brief Whether @p block's admission map holds a value under the marker
    ///        template @p templ at @p scope (tests only).
    bool admissionDemandAt(gl::Memory& block, const std::string& templ, const std::string& scope) {
        int64_t pk = 0;
        return gl::lookupTemplateKey(block.templateInterner, block.nameMap, templ, scope, pk)
            && block.overallHashMemory.admissionMap.lookup(pk) != 0;
    }
} // namespace

// In the full I5's base-case block the released successor case demands the
// partner's entry of the second twin premise's sequence, `(in2[12,marker,13])`,
// there and not at `main`; n (typed (1)) and the anchor's arguments get no
// demand (D-588).
TEST(twin_value_demand, successor_case_demands_the_partners_entry_of_the_sequence) {
    gl::tests::ProverFixture ea(std::string("FTA"));
    ea.proofGridBuilder.addTheoremToMemory(kFullI5, ea.body, 0, false, ea.globalDependencies);
    gl::Memory* zero = zeroBlockOfFullI5(ea);
    ASSERT_TRUE(zero != nullptr);
    ASSERT_TRUE(zero->twinExperiment == gl::TwinExperiment::successorCase);
    ASSERT_FALSE(zero->twinValueDemandsFiled);
    const std::string caseScope = releaseSuccessorCase(ea, *zero);
    ea.admissionMemory.fileTwinValueDemands(*zero, zero->twinCaseScopeId);
    ASSERT_TRUE(zero->twinValueDemandsFiled);
    ASSERT_TRUE(admissionDemandAt(*zero, "(in2[12,marker,13])", caseScope));
    ASSERT_FALSE(admissionDemandAt(*zero, "(in2[12,marker,13])", "main"));
    for (const char* other : { "(in2[12,marker,11])", "(in2[12,marker,3])", "(in2[12,marker,1])" })
        ASSERT_FALSE(admissionDemandAt(*zero, other, caseScope));
}

// The partner's predecessor registered in the released successor case files
// the demand there, and the entry's existence parked at `main` is replayed
// into the case; the same fact at `main` files nothing, and a later
// predecessor in the case files nothing again.
TEST(twin_value_demand, case_predecessor_replays_the_parked_entry_into_the_case) {
    gl::tests::ProverFixture ea(std::string("FTA"));
    ea.proofGridBuilder.addTheoremToMemory(kFullI5, ea.body, 0, false, ea.globalDependencies);
    gl::Memory* zero = zeroBlockOfFullI5(ea);
    ASSERT_TRUE(zero != nullptr);
    const char* const source = "(existence0[1,12,13])";
    const int levels[2] = { 0, 2 };
    ea.rejectedSources.updateRejectedMap(sp("(in2[12,marker,13])"), sp(source), 0, 1, levels, 2,
        zero->overallHashMemory, *zero, sp("main"));
    const std::string caseScope = releaseSuccessorCase(ea, *zero);

    depositIn(ea, *zero, "(in2[it_0_lev_3_4,12,3])");
    ASSERT_FALSE(zero->twinValueDemandsFiled);
    ASSERT_TRUE(stagedTags(*zero, source, caseScope).empty());

    depositIn(ea, *zero, "(in2[it_0_lev_3_3,12,3])", caseScope);
    ASSERT_TRUE(zero->twinValueDemandsFiled);
    ASSERT_TRUE(admissionDemandAt(*zero, "(in2[12,marker,13])", caseScope));
    const std::vector<std::string> tags = stagedTags(*zero, source, caseScope);
    ASSERT_FALSE(tags.empty());
    ASSERT_TRUE(std::find(tags.begin(), tags.end(), std::string("zoom in")) != tags.end());
    ASSERT_TRUE(stagedTags(*zero, source, "main").empty());

    // The block files once (fileTwinValueDemands asserts it).
    depositIn(ea, *zero, "(in2[it_0_lev_3_5,12,3])", caseScope);
    ASSERT_TRUE(zero->twinValueDemandsFiled);
}

// Without twins the install files no twin goal, and a predecessor fact files
// nothing.
TEST(twin_second_goal, no_second_goal_without_twins) {
    gl::tests::ProverFixture ea(std::string("FTA"));
    const char* const equalLength =
        "(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10,11](primeFactorSeq[1,3,4,5,2,6,7,9,10,11])"
        "(>[12](primeFactorSeq[1,3,4,5,2,6,7,9,12,11])(=[10,12]))))";
    ea.proofGridBuilder.addTheoremToMemory(equalLength, ea.body, 0, false, ea.globalDependencies);
    gl::Memory* weak = weakBlockOf(ea, "(primeFactorSeq[1,3,4,5,2,6,7,9,10,11])",
        "(primeFactorSeq[1,3,4,5,2,6,7,9,12,11])");
    ASSERT_TRUE(weak != nullptr);
    ASSERT_EQ(weak->twinFirstGoalId, gl::NameId(0));
    ASSERT_EQ(weak->twinSecondLevel, 0);
    depositIn(ea, *weak, "(in2[it_0_lev_3_5,12,3])");
    ASSERT_EQ(weak->twinPredecessorId, gl::NameId(0));
    ASSERT_TRUE(stagedTags(*weak, "(=[it_0_lev_3_5,rec2])", "main").empty());
}
