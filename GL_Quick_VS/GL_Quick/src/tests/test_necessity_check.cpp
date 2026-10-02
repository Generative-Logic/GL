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
// Unit tests for checkNecessityForEquality (CNFE) and its goal-rescan floor:
// the sign of a rule head must equal the goal's sign, a call visits only the
// chains from its first chain id on, the door's goal rescan raises the floor
// over the chains it visited, and the owner-less sweep remaps the floor.
#include "test_harness.hpp"

#include "../prover.hpp"

#include <cstring>
#include <string>

namespace {

    /// @brief Span over a string literal (tests only).
    gl::StrSpan sp(const char* s) {
        return gl::StrSpan(s, static_cast<int32_t>(std::strlen(s)));
    }

    /// @brief Install one implication into the LB's overall hash memory
    ///        exactly as the door does (chain, head and remaining arguments
    ///        from the one decomposition).
    void install(gl::ExpressionAnalyzer& ea, const std::string& impl) {
        gl::ScratchArena& kvArena = gl::scratchArenas().forSlot(
            gl::scratchArenas().slotCount() - 1);
        gl::ScratchScope kvScope(kvArena);
        gl::StrSpan chainRun[gl::ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
        int32_t chainRunN = 0;
        gl::StrSpan headSpan;
        gl::StrSpan remKeyRun[gl::ExecutionParameters::MAX_ADMISSION_REM_ARGS];
        int32_t remKeyRunN = 0;
        ea.decomposeImplicationForInstall(gl::StrSpan(impl), kvArena,
            chainRun, chainRunN, headSpan, remKeyRun, remKeyRunN);
        const int lv[1] = { 0 };
        ea.addToHashMemory(chainRun, chainRunN, headSpan, remKeyRun, remKeyRunN,
            ea.body, ea.body.overallHashMemory, lv, 1, gl::StrSpan(impl),
            ea.parameters.maxIterationNumberVariable,
            ea.parameters.standardMaxSecondaryNumber, false,
            ea.parameters.minNumOperatorsKey, gl::StrSpan("implication", 11),
            false, gl::StrSpan(impl));
    }

    /// @brief Whether CNFE prepared @p premise at `main` (its prepared-marker
    ///        template is recorded; the premise carries no changeable
    ///        argument, so the marker form is the premise itself).
    bool prepared(gl::ExpressionAnalyzer& ea, const char* premise) {
        return ea.body.integrationPreparedMarker.contains(gl::mintTemplateKey(
            ea.body.templateInterner, ea.body.nameMap, premise, "main"));
    }

    /// @brief Deposit @p text at `main` through the door with @p status.
    void deposit(gl::ExpressionAnalyzer& ea, const char* text, int status) {
        const int lv[1] = { 0 };
        const gl::TransientOrigin origin{ true, gl::OriginTag::taskFormulation, nullptr, 0 };
        ea.addExprToMemoryBlock(sp(text), ea.body, 0, status, lv, 1, origin,
            -1, -1, sp("main"), false);
    }

} // namespace

// A positive goal prepares the premises of the chains whose head is positive
// only; a negated goal those of the negated heads only. A rule concluding the
// goal's negation cannot help establish the goal.
TEST(necessity_check, goal_prepares_only_heads_of_its_own_sign) {
    gl::ExpressionAnalyzer ea("Peano");
    install(ea, "(>[1](in3[1,u_7,u_8,u_4])(in[1,u_5]))");
    install(ea, "(>[1](in2[1,u_7,u_6])!(in[1,u_5]))");
    ASSERT_EQ(ea.body.overallHashMemory.originals.count(), 2);

    ea.checkNecessityForEquality(sp("(in[x,5])"), ea.body, sp("main"), 1);
    ASSERT_TRUE(prepared(ea, "(in3[x,7,8,4])"));
    ASSERT_FALSE(prepared(ea, "(in2[x,7,6])"));

    ea.checkNecessityForEquality(sp("!(in[y,5])"), ea.body, sp("main"), 1);
    ASSERT_TRUE(prepared(ea, "(in2[y,7,6])"));
    ASSERT_FALSE(prepared(ea, "(in3[y,7,8,4])"));
}

// A call visits only the chains from its first chain id on: the goal rescan's
// floor skips the chains an earlier call already visited.
TEST(necessity_check, first_chain_id_skips_the_earlier_chains) {
    gl::ExpressionAnalyzer ea("Peano");
    install(ea, "(>[1](in3[1,u_7,u_8,u_4])(in[1,u_5]))");
    install(ea, "(>[1](in2[1,u_7,u_6])(in[1,u_5]))");
    ASSERT_EQ(ea.body.overallHashMemory.originals.count(), 2);

    ea.checkNecessityForEquality(sp("(in[x,5])"), ea.body, sp("main"), 2);
    ASSERT_TRUE(prepared(ea, "(in2[x,7,6])"));
    ASSERT_FALSE(prepared(ea, "(in3[x,7,8,4])"));

    ea.checkNecessityForEquality(sp("(in[x,5])"), ea.body, sp("main"), 1);
    ASSERT_TRUE(prepared(ea, "(in3[x,7,8,4])"));
}

// The door: a goal registered before a matching rule gets the rule's premise
// prepared by the goal rescan that follows the rule's install, and every
// rescan raises the floor to the chain count it started with.
TEST(necessity_check, door_goal_rescan_prepares_new_chains_and_raises_the_floor) {
    gl::ExpressionAnalyzer ea("Peano");
    gl::HashMemory& hm = ea.body.overallHashMemory;
    ASSERT_EQ(hm.goalScanChainMark, 0);

    deposit(ea, "(in[x,5])", 2);   // the goal
    ASSERT_EQ(ea.body.intToBeProved.count(), 1);
    ASSERT_EQ(hm.goalScanChainMark, 0);

    // The door installs rules from compact implications. The rescan raises
    // the floor to the chain count it started with; the preparations it runs
    // may install further chains above the floor.
    const std::string rule1 =
        ea.compileImplicationToCompact("(>[1](in3[1,u_7,u_8,u_4])(in[1,u_5]))");
    const std::string rule2 =
        ea.compileImplicationToCompact("(>[1](in2[1,u_7,u_6])(in[1,u_5]))");
    ea.preMintReducedOrs();
    deposit(ea, rule1.c_str(), 0);
    ASSERT_TRUE(hm.originals.count() >= 1);
    ASSERT_TRUE(prepared(ea, "(in3[x,7,8,4])"));
    const int32_t m1 = hm.goalScanChainMark;
    ASSERT_TRUE(m1 >= 1 && m1 <= hm.originals.count());

    const int32_t before = hm.originals.count();
    deposit(ea, rule2.c_str(), 0);
    ASSERT_TRUE(hm.originals.count() > before);
    ASSERT_TRUE(prepared(ea, "(in2[x,7,6])"));
    ASSERT_TRUE(hm.goalScanChainMark > before
                && hm.goalScanChainMark <= hm.originals.count());
}

// The owner-less sweep renumbers the surviving chains in order; the floor
// becomes the number of survivors among the old chains below it.
TEST(necessity_check, owner_less_sweep_remaps_the_floor) {
    gl::ExpressionAnalyzer ea("Peano");
    gl::HashMemory& hm = ea.body.overallHashMemory;
    const std::string r1 = "(>[1](in3[1,u_7,u_8,u_4])(in[1,u_5]))";
    const std::string r2 = "(>[1](in2[1,u_7,u_6])(in[1,u_5]))";
    const std::string r3 = "(>[1](in2[1,u_6,u_7])(in[1,u_9]))";
    install(ea, r1);
    install(ea, r2);
    install(ea, r3);
    ASSERT_EQ(hm.originals.count(), 3);

    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 4 << 20, 1 << 18 });
    gl::LbArena sets(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::ColdHashSet<gl::BytesKeyStore> visited(&sets, &d);
    gl::ColdHashSet<gl::PodKeyStore<int64_t>> dropSet(&sets, &d);
    gl::ScratchArena& gArena = gl::genScratchArenas().forSlot(
        gl::genScratchArenas().slotCount() - 1);
    const auto removeRule = [&](const std::string& impl) {
        visited.resetToFresh();
        dropSet.resetToFresh();
        ASSERT_TRUE(ea.removeRuleFromHashMemory(ea.body, hm, gl::StrSpan(impl),
            sp("main"), visited, dropSet));
        gl::ExpressionAnalyzer::finishRuleRemovals(hm, dropSet, gArena);
    };

    // Chains 1 and 2 were visited by every goal; chain 1 leaves: the old
    // chain 2 becomes chain 1, the only survivor below the floor.
    hm.goalScanChainMark = 2;
    removeRule(r1);
    ASSERT_EQ(hm.originals.count(), 2);
    ASSERT_EQ(hm.goalScanChainMark, 1);

    // A chain above the floor leaves: the floor keeps its survivor.
    removeRule(r3);
    ASSERT_EQ(hm.originals.count(), 1);
    ASSERT_EQ(hm.goalScanChainMark, 1);

    // Teardown clears the floor with the chains.
    hm.clear();
    ASSERT_EQ(hm.goalScanChainMark, 0);
}
