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
#include "../memory.hpp"
#include <cstddef>
#include <utility>
#include "test_harness.hpp"
#include "prover_fixture.hpp"

#include "../prover.hpp"

#include <cstring>
#include <map>
#include <vector>
#include <string>

namespace {

    /// @brief Span over a string literal (tests only).
    gl::StrSpan sp(const char* s) {
        return gl::StrSpan(s, static_cast<int32_t>(std::strlen(s)));
    }

    /// @brief Install one implication into the LB's overall hash memory
    ///        exactly as the door does (chain, head and remaining arguments
    ///        from the one decomposition).
    void install(gl::tests::ProverFixture& ea, const std::string& impl, const char* scope = "main") {
        gl::ScratchArena& kvArena = gl::scratchArenas().forSlot(
            gl::scratchArenas().slotCount() - 1);
        gl::ScratchScope kvScope(kvArena);
        gl::StrSpan chainRun[gl::ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
        int32_t chainRunN = 0;
        gl::StrSpan headSpan;
        gl::StrSpan remKeyRun[gl::ExecutionParameters::MAX_ADMISSION_REM_ARGS];
        int32_t remKeyRunN = 0;
        ea.hashRuleInstaller.decomposeImplicationForInstall(gl::StrSpan(impl), kvArena,
            chainRun, chainRunN, headSpan, remKeyRun, remKeyRunN);
        const int lv[1] = { 0 };
        ea.hashRuleInstaller.addToHashMemory(chainRun, chainRunN, headSpan, remKeyRun, remKeyRunN,
            ea.body, ((ea.body).overallHashMemory), lv, 1, gl::StrSpan(impl),
            ea.parameters.maxIterationNumberVariable,
            ea.parameters.standardMaxSecondaryNumber, false,
            ea.parameters.minNumOperatorsKey, gl::StrSpan("implication", 11),
            false, gl::StrSpan(impl), sp(scope));
    }

    /// @brief Whether CNFE prepared @p premise at `main` (its prepared-marker
    ///        template is recorded; the premise carries no changeable
    ///        argument, so the marker form is the premise itself).
    bool prepared(gl::tests::ProverFixture& ea, const char* premise) {
        return ((ea.body).integrationPreparedMarker).contains(gl::mintTemplateKey(
            ((ea.body).templateInterner), ((ea.body).nameMap), premise, "main"));
    }

    /// @brief Deposit @p text at `main` through the door with @p status.
    void deposit(gl::tests::ProverFixture& ea, const char* text, int status) {
        const int lv[1] = { 0 };
        const gl::TransientOrigin origin{ true, gl::OriginTag::taskFormulation, nullptr, 0 };
        ea.statementMemory.addExprToMemoryBlock(sp(text), ea.body, 0, status, lv, 1, origin,
            -1, -1, sp("main"), false);
    }

    /// @brief Whether @p statement is a known statement at `main`.
    bool known(gl::tests::ProverFixture& ea, const char* statement) {
        return gl::lookupStatementFlags(((ea.body).intKnownStatements), ((ea.body).nameMap),
            std::string(statement), std::string("main")) != nullptr;
    }

    /// @brief Whether @p statement is a known statement at exactly @p scope.
    bool knownAt(gl::tests::ProverFixture& ea, const char* statement, const std::string& scope) {
        return gl::lookupStatementFlags(((ea.body).intKnownStatements), ((ea.body).nameMap),
            std::string(statement), scope) != nullptr;
    }

    /// @brief The two-constant rule of the tests below: its head `in2[1,2,u_6]`
    ///        carries the constants 1 and 2 beside the fixed argument `u_6`;
    ///        the equality premises hold the goal's arguments as fixed
    ///        arguments, so replacing the constants repeats them.
    const char* const kTwoConstantRule =
        "(>[1,2](in3[1,2,u_8,u_4])(>[](=[1,u_9])(>[](=[2,u_10])(in2[1,2,u_6]))))";

} // namespace

// A positive goal prepares the premises of the chains whose head is positive
// only; a negated goal those of the negated heads only. A rule concluding the
// goal's negation cannot help establish the goal.
TEST(necessity_check, goal_prepares_only_heads_of_its_own_sign) {
    gl::tests::ProverFixture ea("Peano");
    install(ea, "(>[1](in3[1,u_7,u_8,u_4])(in[1,u_5]))");
    install(ea, "(>[1](in2[1,u_7,u_6])!(in[1,u_5]))");
    ASSERT_EQ(((((ea.body).overallHashMemory)).originals).count(), 2);

    ea.integrationProcessor.checkNecessityForEquality(sp("(in[x,5])"), ea.body, sp("main"), 1);
    ASSERT_TRUE(prepared(ea, "(in3[x,7,8,4])"));
    ASSERT_FALSE(prepared(ea, "(in2[x,7,6])"));

    ea.integrationProcessor.checkNecessityForEquality(sp("!(in[y,5])"), ea.body, sp("main"), 1);
    ASSERT_TRUE(prepared(ea, "(in2[y,7,6])"));
    ASSERT_FALSE(prepared(ea, "(in3[y,7,8,4])"));
}

// A head with two constants beside a fixed argument matches the goal at both
// constant slots: every premise is prepared with both goal arguments, and each
// constant whose goal argument then repeats in a premise gets its own copy.
TEST(necessity_check, goal_matches_a_head_with_two_constants_and_copies_both) {
    gl::tests::ProverFixture ea("Peano");
    install(ea, kTwoConstantRule);
    ASSERT_EQ(((((ea.body).overallHashMemory)).originals).count(), 1);

    ea.integrationProcessor.checkNecessityForEquality(sp("(in2[9,10,6])"), ea.body, sp("main"), 1);
    ASSERT_TRUE(prepared(ea, "(in3[9,10,8,4])"));
    ASSERT_TRUE(known(ea, "(=[9,9_copy])"));
    ASSERT_TRUE(known(ea, "(=[10,10_copy])"));
}

// Only a constant whose goal argument repeats in a premise demands a copy: with
// the equality premise for the first constant alone, the second gets none.
TEST(necessity_check, only_the_constant_whose_argument_repeats_gets_a_copy) {
    gl::tests::ProverFixture ea("Peano");
    install(ea, "(>[1,2](in3[1,2,u_8,u_4])(>[](=[1,u_9])(in2[1,2,u_6])))");

    ea.integrationProcessor.checkNecessityForEquality(sp("(in2[9,10,6])"), ea.body, sp("main"), 1);
    ASSERT_TRUE(prepared(ea, "(in3[9,10,8,4])"));
    ASSERT_TRUE(known(ea, "(=[9,9_copy])"));
    ASSERT_FALSE(known(ea, "(=[10,10_copy])"));
}

// The fixed argument beside the two constants must equal the goal's argument
// at its slot; otherwise the chain is no candidate.
TEST(necessity_check, two_constant_head_needs_its_fixed_argument_to_match) {
    gl::tests::ProverFixture ea("Peano");
    install(ea, kTwoConstantRule);

    ea.integrationProcessor.checkNecessityForEquality(sp("(in2[9,10,7])"), ea.body, sp("main"), 1);
    ASSERT_FALSE(prepared(ea, "(in3[9,10,7,4])"));
    ASSERT_FALSE(prepared(ea, "(in3[9,10,8,4])"));
    ASSERT_FALSE(known(ea, "(=[9,9_copy])"));
    ASSERT_FALSE(known(ea, "(=[10,10_copy])"));
}

// A head made of two constants and nothing else leaves no argument to match
// and is no candidate.
TEST(necessity_check, head_of_only_two_constants_is_no_candidate) {
    gl::tests::ProverFixture ea("Peano");
    install(ea, "(>[1,2](in3[1,2,u_8,u_4])(=[1,2]))");

    ea.integrationProcessor.checkNecessityForEquality(sp("(=[9,10])"), ea.body, sp("main"), 1);
    ASSERT_FALSE(prepared(ea, "(in3[9,10,8,4])"));
}

// A call visits only the chains from its first chain id on: the goal rescan's
// floor skips the chains an earlier call already visited.
TEST(necessity_check, first_chain_id_skips_the_earlier_chains) {
    gl::tests::ProverFixture ea("Peano");
    install(ea, "(>[1](in3[1,u_7,u_8,u_4])(in[1,u_5]))");
    install(ea, "(>[1](in2[1,u_7,u_6])(in[1,u_5]))");
    ASSERT_EQ(((((ea.body).overallHashMemory)).originals).count(), 2);

    ea.integrationProcessor.checkNecessityForEquality(sp("(in[x,5])"), ea.body, sp("main"), 2);
    ASSERT_TRUE(prepared(ea, "(in2[x,7,6])"));
    ASSERT_FALSE(prepared(ea, "(in3[x,7,8,4])"));

    ea.integrationProcessor.checkNecessityForEquality(sp("(in[x,5])"), ea.body, sp("main"), 1);
    ASSERT_TRUE(prepared(ea, "(in3[x,7,8,4])"));
}

// A goal prepared again at the same scope (the second call revisits the chain
// the first one prepared) dry-runs under a new numbered sentinel scope, and
// every sentinel is queued for the step's wipe and filtered
// (D-496).
TEST(necessity_check, repeated_preparation_opens_a_new_sentinel_scope) {
    gl::tests::ProverFixture ea("Peano");
    install(ea, "(>[1](in3[1,u_7,u_8,u_4])(in[1,u_5]))");
    install(ea, "(>[1](in2[1,u_7,u_6])(in[1,u_5]))");
    ea.integrationProcessor.checkNecessityForEquality(sp("(in[x,5])"), ea.body, sp("main"), 2);
    ea.integrationProcessor.checkNecessityForEquality(sp("(in[x,5])"), ea.body, sp("main"), 1);

    gl::Memory& m = ea.body;
    const int32_t prefixLen = static_cast<int32_t>(sizeof(gl::ScopeSyntax::kHypotheticalSentinelPrefix) - 1);
    std::map<std::string, int> dryRunsPerGoal;
    int32_t sentinels = 0;
    for (gl::NameId id = 2; id <= m.nameMap.nameCount(); ++id) {
        if (m.nameMap.stackEmpty(id) || m.nameMap.parentOf(id) != gl::NameMap::MAIN_ID) continue;
        const gl::StrSpan payload = m.nameMap.decodeSubView(m.nameMap.stackBack(id));
        if (payload.len <= prefixLen || std::memcmp(payload.ptr, gl::ScopeSyntax::kHypotheticalSentinelPrefix,
                static_cast<std::size_t>(prefixLen)) != 0) continue;
        ++sentinels;
        ASSERT_TRUE(m.pendingWipeScopes.contains(id));
        ASSERT_TRUE(m.intValidityNamesToFilter.lookup(id) != 0);
        // The payload is the prefix, a decimal number, '_' and the goal.
        const std::string rest(payload.ptr + prefixLen, static_cast<std::size_t>(payload.len - prefixLen));
        const std::size_t cut = rest.find('_');
        ASSERT_TRUE(cut != std::string::npos && cut > 0);
        for (std::size_t i = 0; i < cut; ++i) ASSERT_TRUE(rest[i] >= '0' && rest[i] <= '9');
        ++dryRunsPerGoal[rest.substr(cut + 1)];
    }
    ASSERT_TRUE(sentinels >= 2);
    bool repeated = false;
    for (const auto& entry : dryRunsPerGoal) repeated = repeated || entry.second >= 2;
    ASSERT_TRUE(repeated);
}

// The door: a goal registered before a matching rule gets the rule's premise
// prepared by the goal rescan that follows the rule's install, and every
// rescan raises the floor to the chain count it started with.
TEST(necessity_check, door_goal_rescan_prepares_new_chains_and_raises_the_floor) {
    gl::tests::ProverFixture ea("Peano");
    gl::HashMemory& hm = ((ea.body).overallHashMemory);
    ASSERT_EQ(((hm).goalScanChainMark), 0);

    deposit(ea, "(in[x,5])", 2);   // the goal
    ASSERT_EQ(((ea.body).intToBeProved).count(), 1);
    ASSERT_EQ(((hm).goalScanChainMark), 0);

    // The door installs rules from compact implications. The rescan raises
    // the floor to the chain count it started with; the preparations it runs
    // may install further chains above the floor.
    const std::string rule1 =
        ea.structuralCompiler.compileImplicationToCompact("(>[1](in3[1,u_7,u_8,u_4])(in[1,u_5]))");
    const std::string rule2 =
        ea.structuralCompiler.compileImplicationToCompact("(>[1](in2[1,u_7,u_6])(in[1,u_5]))");
    ea.orRuleCompiler.preMintReducedOrs();
    deposit(ea, rule1.c_str(), 0);
    ASSERT_TRUE(((hm).originals).count() >= 1);
    ASSERT_TRUE(prepared(ea, "(in3[x,7,8,4])"));
    const int32_t m1 = ((hm).goalScanChainMark);
    ASSERT_TRUE(m1 >= 1 && m1 <= ((hm).originals).count());

    const int32_t before = ((hm).originals).count();
    deposit(ea, rule2.c_str(), 0);
    ASSERT_TRUE(((hm).originals).count() > before);
    ASSERT_TRUE(prepared(ea, "(in2[x,7,6])"));
    ASSERT_TRUE(((hm).goalScanChainMark) > before
                && ((hm).goalScanChainMark) <= ((hm).originals).count());
}

// The goal rescan visits every goal over the chains above the floor only and
// raises the floor to the chain count it started with.
TEST(necessity_check, goal_rescan_visits_only_chains_above_the_floor) {
    gl::tests::ProverFixture ea("Peano");
    gl::HashMemory& hm = ((ea.body).overallHashMemory);
    deposit(ea, "(in[x,5])", 2);   // the goal; its door visits no chain
    install(ea, "(>[1](in3[1,u_7,u_8,u_4])(in[1,u_5]))");
    ((hm).goalScanChainMark) = ((hm).originals).count();
    ea.integrationProcessor.rescanGoalsAgainstNewChains(ea.body);
    ASSERT_FALSE(prepared(ea, "(in3[x,7,8,4])"));
    ASSERT_EQ(((hm).goalScanChainMark), ((hm).originals).count());

    const int32_t before = ((hm).originals).count();
    install(ea, "(>[1](in2[1,u_7,u_6])(in[1,u_5]))");
    ea.integrationProcessor.rescanGoalsAgainstNewChains(ea.body);
    ASSERT_TRUE(prepared(ea, "(in2[x,7,6])"));
    ASSERT_FALSE(prepared(ea, "(in3[x,7,8,4])"));
    ASSERT_TRUE(((hm).goalScanChainMark) > before
                && ((hm).goalScanChainMark) <= ((hm).originals).count());
}

// The owner-less sweep renumbers the surviving chains in order; the floor
// becomes the number of survivors among the old chains below it.
TEST(necessity_check, owner_less_sweep_remaps_the_floor) {
    gl::tests::ProverFixture ea("Peano");
    gl::HashMemory& hm = ((ea.body).overallHashMemory);
    const std::string r1 = "(>[1](in3[1,u_7,u_8,u_4])(in[1,u_5]))";
    const std::string r2 = "(>[1](in2[1,u_7,u_6])(in[1,u_5]))";
    const std::string r3 = "(>[1](in2[1,u_6,u_7])(in[1,u_9]))";
    install(ea, r1);
    install(ea, r2);
    install(ea, r3);
    ASSERT_EQ(((hm).originals).count(), 3);

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
        ASSERT_TRUE(ea.hashRuleInstaller.removeRuleFromHashMemory(ea.body, hm, gl::StrSpan(impl),
            sp("main"), visited, dropSet));
        gl::HashRuleInstaller::finishRuleRemovals(hm, dropSet, gArena);
    };

    // Chains 1 and 2 were visited by every goal; chain 1 leaves: the old
    // chain 2 becomes chain 1, the only survivor below the floor.
    ((hm).goalScanChainMark) = 2;
    removeRule(r1);
    ASSERT_EQ(((hm).originals).count(), 2);
    ASSERT_EQ(((hm).goalScanChainMark), 1);

    // A chain above the floor leaves: the floor keeps its survivor.
    removeRule(r3);
    ASSERT_EQ(((hm).originals).count(), 1);
    ASSERT_EQ(((hm).goalScanChainMark), 1);

    // Teardown clears the floor with the chains.
    hm.clear();
    ASSERT_EQ(((hm).goalScanChainMark), 0);
}


TEST(necessity_check, open_goal_prepares_single_projection_witness) {
    gl::tests::ProverFixture ea("Peano");
    deposit(ea, "(in[x,11])", 2);
    install(ea, "(>[1,2](in2[1,2,u_9])(in[1,u_11]))");
    ea.integrationProcessor.rescanGoalsAgainstNewChains(ea.body);
    ASSERT_TRUE(prepared(ea, "(in2[x,marker,9])"));
    ASSERT_FALSE(prepared(ea, "(in2[x,2,9])"));
    const auto key = gl::mintTemplateKey(ea.body.templateInterner, ea.body.nameMap,
        "(in2[x,marker,9])", "main");
    ASSERT_TRUE(ea.body.overallHashMemory.admissionMap.lookup(key) != 0);
    const auto integrationKey = gl::mintTemplateKey(ea.body.templateInterner, ea.body.nameMap,
        "(in2[u_x,marker,u_9])", "main");
    ASSERT_EQ(ea.body.overallHashMemory.admissionMapIntegration.lookup(integrationKey), 0);
    ASSERT_FALSE(known(ea, "(in[x,11])"));
}

TEST(necessity_check, projection_witness_preserves_equal_spelled_fixed_argument) {
    gl::tests::ProverFixture ea("Peano");
    install(ea, "(>[1,2](in3[1,2,u_2,u_4])(in[1,u_5]))");
    deposit(ea, "(in[x,5])", 2);
    ASSERT_TRUE(prepared(ea, "(in3[x,marker,2,4])"));
    ASSERT_FALSE(prepared(ea, "(in3[x,marker,marker,4])"));
    const auto key = gl::mintTemplateKey(ea.body.templateInterner, ea.body.nameMap,
        "(in3[x,marker,2,4])", "main");
    ASSERT_TRUE(ea.body.overallHashMemory.admissionMap.lookup(key) != 0);
}

// Without a registered goal the projection does not apply, and the premise's
// unfixed bound `2` keeps it from being prepared as the block's `2`.
TEST(necessity_check, projection_witness_requires_a_registered_goal) {
    gl::tests::ProverFixture ea("Peano");
    install(ea, "(>[1,2](in2[1,2,u_9])(in[1,u_11]))");
    ea.integrationProcessor.checkNecessityForEquality(sp("(in[x,11])"), ea.body, sp("main"), 1);
    ASSERT_FALSE(prepared(ea, "(in2[x,marker,9])"));
    ASSERT_FALSE(prepared(ea, "(in2[x,2,9])"));
}

// Two missing arguments are no projection, and the premise holding them is
// not prepared with the block's `2` and `3` in their places.
TEST(necessity_check, projection_witness_does_not_expand_multiple_missing_arguments) {
    gl::tests::ProverFixture ea("Peano");
    install(ea, "(>[1,2,3](in3[1,2,3,u_4])(in[1,u_5]))");
    deposit(ea, "(in[x,5])", 2);
    ASSERT_FALSE(prepared(ea, "(in3[x,marker,marker,4])"));
    ASSERT_FALSE(prepared(ea, "(in3[x,2,3,4])"));
}

// A premise that still holds a bound variable after the replacement is
// neither prepared nor counted for a copy, while the chain's determined
// premise is prepared: the bound `2` would otherwise be read as the block's
// `2`, and its premise would demand the copy of 9.
TEST(necessity_check, premise_with_an_unfixed_bound_variable_is_neither_prepared_nor_copied) {
    gl::tests::ProverFixture ea("Peano");
    install(ea, "(>[1,2](in3[1,u_7,u_8,u_4])(>[](in2[2,1,u_9])(in[1,u_5])))");
    ea.integrationProcessor.checkNecessityForEquality(sp("(in[9,5])"), ea.body, sp("main"), 1);
    ASSERT_TRUE(prepared(ea, "(in3[9,7,8,4])"));
    ASSERT_FALSE(prepared(ea, "(in2[2,9,9])"));
    ASSERT_FALSE(known(ea, "(=[9,9_copy])"));
}

// The same premise with its variable fixed by the head is prepared and demands
// the copy: only the unfixed bound variable withholds it.
TEST(necessity_check, premise_determined_by_the_head_is_prepared_and_copied) {
    gl::tests::ProverFixture ea("Peano");
    install(ea, "(>[1](in3[1,u_7,u_8,u_4])(>[](in2[u_2,1,u_9])(in[1,u_5])))");
    ea.integrationProcessor.checkNecessityForEquality(sp("(in[9,5])"), ea.body, sp("main"), 1);
    ASSERT_TRUE(prepared(ea, "(in3[9,7,8,4])"));
    ASSERT_TRUE(prepared(ea, "(in2[2,9,9])"));
    ASSERT_TRUE(known(ea, "(=[9,9_copy])"));
}

// A premise that reads `!(=[Y,Y])` after the replacement can never hold: it
// demands no copy, while the chain's other premise is still prepared
// (D-535).
TEST(necessity_check, negated_reflexive_equality_premise_demands_no_copy) {
    gl::tests::ProverFixture ea("Peano");
    install(ea, "(>[1](in3[1,u_7,u_8,u_4])(>[]!(=[1,u_9])(in[1,u_5])))");
    ea.integrationProcessor.checkNecessityForEquality(sp("(in[9,5])"), ea.body, sp("main"), 1);
    ASSERT_TRUE(prepared(ea, "(in3[9,7,8,4])"));
    ASSERT_FALSE(known(ea, "(=[9,9_copy])"));
    ASSERT_FALSE(known(ea, "(=[9_copy,9])"));
}

// The positive form `(=[Y,Y])` still demands its copy: the copy is the only
// way the premise can fire, since GL stores no `(=[Y,Y])`.
TEST(necessity_check, positive_reflexive_equality_premise_still_demands_its_copy) {
    gl::tests::ProverFixture ea("Peano");
    install(ea, "(>[1](in3[1,u_7,u_8,u_4])(>[](=[1,u_9])(in[1,u_5])))");
    ea.integrationProcessor.checkNecessityForEquality(sp("(in[9,5])"), ea.body, sp("main"), 1);
    ASSERT_TRUE(prepared(ea, "(in3[9,7,8,4])"));
    ASSERT_TRUE(known(ea, "(=[9,9_copy])"));
}

// A goal below main gets its copy in its own scope, not at main.
TEST(necessity_check, copy_is_written_at_the_goal_scope) {
    gl::tests::ProverFixture ea("Peano");
    const auto scope = ea.body.nameMap.encodePush(gl::NameMap::MAIN_ID, "left");
    const std::string scopeName = ea.body.nameMap.decode(scope);
    install(ea, "(>[1](in3[1,u_7,u_8,u_4])(>[](in2[u_2,1,u_9])(in[1,u_5])))");
    ea.integrationProcessor.checkNecessityForEquality(sp("(in[9,5])"), ea.body,
        gl::StrSpan(scopeName), 1);
    ASSERT_TRUE(knownAt(ea, "(=[9,9_copy])", scopeName));
    ASSERT_FALSE(known(ea, "(=[9,9_copy])"));
}

// A goal argument that already has an alias in an equivalence class at the
// goal's scope needs no copy (without the alias the same call copies it,
// `premise_determined_by_the_head_is_prepared_and_copied`).
TEST(necessity_check, goal_argument_with_a_visible_alias_gets_no_copy) {
    gl::tests::ProverFixture ea("Peano");
    deposit(ea, "(=[9,15])", 0);
    install(ea, "(>[1](in3[1,u_7,u_8,u_4])(>[](in2[u_2,1,u_9])(in[1,u_5])))");
    ea.integrationProcessor.checkNecessityForEquality(sp("(in[9,5])"), ea.body, sp("main"), 1);
    ASSERT_FALSE(known(ea, "(=[9,9_copy])"));
    ASSERT_FALSE(known(ea, "(=[9_copy,9])"));
}

TEST(necessity_check, projection_witness_does_not_expand_multiple_premises) {
    gl::tests::ProverFixture ea("Peano");
    install(ea, "(>[1,2](in2[1,2,u_9])(>[](in[2,u_1])(in[1,u_11])))");
    deposit(ea, "(in[x,11])", 2);
    ASSERT_FALSE(prepared(ea, "(in2[x,marker,9])"));
}

TEST(necessity_check, projection_witness_requires_an_owner_visible_at_the_goal_scope) {
    for (bool sibling : {false, true}) {
        gl::tests::ProverFixture ea("Peano");
        const auto main = ea.body.nameMap.encode("main");
        const auto goalScope = ea.body.nameMap.encodePush(main, "hypo_(goal[x])");
        const auto siblingScope = ea.body.nameMap.encodePush(main, "hypo_(other[x])");
        const std::string goalName = ea.body.nameMap.decode(goalScope);
        const std::string ownerName = sibling ? ea.body.nameMap.decode(siblingScope) : "main";
        install(ea, "(>[1,2](in2[1,2,u_9])(in[1,u_11]))", ownerName.c_str());
        const auto goal = gl::encodeExpression(sp("(in[x,11])"), gl::StrSpan(goalName), ea.body.nameMap);
        ea.body.intToBeProved.assignSet(gl::packStatementKey(goal.originalId, goalScope), nullptr, 0);
        ea.integrationProcessor.checkNecessityForEquality(sp("(in[x,11])"), ea.body, gl::StrSpan(goalName), 1);
        const auto key = gl::mintTemplateKey(ea.body.templateInterner, ea.body.nameMap,
            "(in2[x,marker,9])", goalName);
        ASSERT_EQ(ea.body.integrationPreparedMarker.contains(key), !sibling);
        ASSERT_EQ(ea.body.overallHashMemory.admissionMap.lookup(key) != 0, !sibling);
    }
}

TEST(necessity_check, projection_witness_queues_ordinary_replay_for_parked_sources) {
    gl::tests::ProverFixture ea("Peano");
    install(ea, "(>[1,2](in2[1,2,u_9])(in[1,u_11]))");
    ea.body.lbMemory.canonicalUses.deferWakes = true;
    deposit(ea, "(in[x,11])", 2);
    const auto key = gl::mintTemplateKey(ea.body.templateInterner, ea.body.nameMap,
        "(in2[x,marker,9])", "main");
    const auto* wake = ea.body.lbMemory.canonicalUses.wakeKinds.find(key);
    ASSERT_TRUE(wake != nullptr);
    ASSERT_TRUE((*wake & 1) != 0);
    ea.body.lbMemory.canonicalUses.deferWakes = false;
}

// headMatchPossible: only an operator with a config and a (1) slot among its
// first argN slots can match a rule head under CNFE's head match.
TEST(necessity_check, head_match_possible_needs_a_config_with_a_numeral_slot) {
    gl::tests::ProverFixture ea("Peano");
    ASSERT_TRUE(ea.integrationProcessor.headMatchPossible(sp("in"), 2));
    ASSERT_FALSE(ea.integrationProcessor.headMatchPossible(sp("in"), 0));
    ASSERT_FALSE(ea.integrationProcessor.headMatchPossible(sp("noSuchOperatorAnywhere"), 2));
}

// maskedKeyOfChain: a head with one constant at a (1) slot is keyed with that
// slot masked and its sign; a constant outside a (1) slot leaves the chain out
// of the masked family.
TEST(necessity_check, masked_key_of_chain_masks_numeral_constants_only) {
    gl::tests::ProverFixture ea("Peano");
    install(ea, "(>[1](in3[1,u_7,u_8,u_4])(in[1,u_5]))");
    install(ea, "(>[1](in2[1,u_7,u_6])!(in[1,u_5]))");
    install(ea, "(>[1](in3[u_2,u_7,1,u_4])(in[u_2,1]))");
    ASSERT_EQ(ea.body.overallHashMemory.originals.count(), 3);
    char buf[gl::kMaxMaskedHeadKeyBytes];
    const gl::StrSpan k1 = ea.integrationProcessor.maskedKeyOfChain(ea.body, 1, buf,
        gl::kMaxMaskedHeadKeyBytes);
    ASSERT_EQ(std::string(k1.ptr, k1.len), std::string("in,\x01,u_5"));
    const gl::StrSpan k2 = ea.integrationProcessor.maskedKeyOfChain(ea.body, 2, buf,
        gl::kMaxMaskedHeadKeyBytes);
    ASSERT_EQ(std::string(k2.ptr, k2.len), std::string("!in,\x01,u_5"));
    const gl::StrSpan k3 = ea.integrationProcessor.maskedKeyOfChain(ea.body, 3, buf,
        gl::kMaxMaskedHeadKeyBytes);
    ASSERT_EQ(k3.len, 0);
}

// collectHeadMatchChains: the chains whose head matches the input with the
// input's sign, ascending, restricted to [firstChainId, lastChainId].
TEST(necessity_check, collect_head_match_chains_respects_sign_and_bounds) {
    gl::tests::ProverFixture ea("Peano");
    install(ea, "(>[1](in3[1,u_7,u_8,u_4])(in[1,u_5]))");
    install(ea, "(>[1](in2[1,u_7,u_6])!(in[1,u_5]))");
    install(ea, "(>[1](in2[u_7,1,u_6])(in[1,u_5]))");
    ea.integrationProcessor.ensureOriginalsHeadIndexBuilt(ea.body);
    gl::ScratchArena& arena = gl::genScratchArenas().forSlot(
        gl::genScratchArenas().slotCount() - 1);
    const auto collect = [&](const char* input, int32_t first, int32_t last) {
        const gl::ArenaOffset mark = arena.cursor();
        gl::DirtyState dirty = gl::DirtyState::Clean;
        gl::PagedVector<int32_t> ids(&arena, &dirty);
        const int32_t n = ea.integrationProcessor.collectHeadMatchChains(sp(input), ea.body,
            first, last, arena, ids);
        std::vector<int32_t> out;
        for (int32_t i = 0; i < static_cast<int32_t>(ids.size()); ++i) out.push_back(ids[i]);
        ids.clear();
        arena.popTo(mark);
        ASSERT_EQ(static_cast<int32_t>(out.size()), n);
        return out;
    };
    ASSERT_EQ(collect("(in[u_x,u_5])", 1, 3), (std::vector<int32_t>{1, 3}));
    ASSERT_EQ(collect("(in[u_x,u_5])", 1, 2), (std::vector<int32_t>{1}));
    ASSERT_EQ(collect("(in[u_x,u_5])", 2, 3), (std::vector<int32_t>{3}));
    ASSERT_EQ(collect("!(in[u_x,u_5])", 1, 3), (std::vector<int32_t>{2}));
    ASSERT_TRUE(collect("(in[u_x,u_6])", 1, 3).empty());
}

// headMatchConstantSlots: the constants of a matched head, one or two.
TEST(necessity_check, head_match_constant_slots_locates_the_constants) {
    gl::tests::ProverFixture ea("Peano");
    int32_t slots[2] = { 7, 7 };
    const gl::StrSpan oneArgs[] = { sp("u_x"), sp("u_5") };
    ASSERT_EQ(ea.integrationProcessor.headMatchConstantSlots(sp("(in[1,u_5])"), sp("in"),
        false, oneArgs, 2, slots), 1);
    ASSERT_EQ(slots[0], 0);
    ASSERT_EQ(slots[1], -1);
    const gl::StrSpan twoArgs[] = { sp("u_a"), sp("u_b"), sp("u_6") };
    ASSERT_EQ(ea.integrationProcessor.headMatchConstantSlots(sp("(in2[1,2,u_6])"), sp("in2"),
        false, twoArgs, 3, slots), 2);
    ASSERT_EQ(slots[0], 0);
    ASSERT_EQ(slots[1], 1);
}
