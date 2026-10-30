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
/// @brief Tests for the induction usefulness test: the `uniquely_definable`
///        configuration field, the test variable kind, the usefulness scope
///        `main_boundary_is_useful`, `gl::InductionUsefulness`, its planting
///        in `ProofGridBuilder::addTheoremToMemory` and the barrier's
///        switch-off.
#include "../memory.hpp"
#include "test_harness.hpp"
#include "prover_fixture.hpp"
#include "proof_grid_builder_fixture.hpp"
#include "../prover.hpp"
#include "../json.hpp"
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

// FTA row 81, the fold step lemma: s(n) = m, P a sequence of length m,
// P(m) = t, fold(P, n) = a, a * t = b  ->  fold(P, m) = b.
const char* const kRow81 =
    "(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](in2[9,10,3])(>[11](sequence[1,4,2,10,11])"
    "(>[12](in2[10,12,11])(>[13](fold[1,3,4,5,11,2,9,13])(>[14](in3[13,12,14,5])(fold[1,3,4,5,11,2,10,14])))))))";

// Peano: a + b = c, s(b) = d, a + d = e  ->  s(c) = e. Both steps are tested
// (the head s(c) = ? and the premise s(b) = d share the successor), but no
// planted fact joins c and b.
const char* const kPeanoSuccessorOfSum =
    "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,4])(>[10](in2[8,10,3])"
    "(>[11](in3[7,10,11,4])(in2[9,11,3])))))";

/// @brief Walk a key chain below the root of a fixture's grid.
gl::Memory* blockAt(gl::tests::ProverFixture& ea, const std::vector<std::string>& keys) {
    gl::Memory* block = &ea.body;
    for (const std::string& key : keys) {
        block = ea.simpleMapStore.findChild(block, key);
        ASSERT_TRUE(block != nullptr);
    }
    return block;
}

/// @brief The number of statement rows a block holds at a scope.
int rowsAt(const gl::Memory& block, gl::NameId scope) {
    int rows = 0;
    for (int32_t i = 0; i < block.intEncodedStatements.size(); ++i)
        if (block.intEncodedStatements[i].validityId == scope) ++rows;
    return rows;
}

/// @brief Whether a block knows a statement at a scope.
bool knownAt(gl::Memory& block, const std::string& statement, gl::NameId scope) {
    const gl::NameId id = block.nameMap.lookup(gl::StrSpan(statement));
    return id != 0 && block.intKnownStatements.find(gl::StatementKey{ id, scope }) != nullptr;
}

/// @brief The usefulness scope's full name.
gl::StrSpan usefulnessScopeName() {
    return gl::StrSpan(gl::ScopeSyntax::kUsefulnessScopeName,
        static_cast<int32_t>(sizeof(gl::ScopeSyntax::kUsefulnessScopeName) - 1));
}

/// @brief Deposit an equality at a step block's usefulness scope through the
///        statement door, as a firing of `main`'s uniqueness rule does.
void depositAtUsefulnessScope(gl::tests::ProverFixture& ea, gl::Memory& step, const char* equality) {
    const int levels[] = { step.level };
    const gl::TransientOrigin origin{ true, gl::OriginTag::taskFormulation, nullptr, 0 };
    ea.statementMemory.addExprToMemoryBlock(gl::StrSpan(std::string(equality)), step, 0, 1,
        levels, 1, origin, -1, -1, usefulnessScopeName(), false);
}

const std::vector<std::string> kRow81Chain = {
    "(AnchorFTA[1,2,3,4,5,6,7,8])", "(in2[9,10,3])", "(sequence[1,4,2,10,11])",
    "(in2[10,12,11])", "(fold[1,3,4,5,11,2,9,13])", "(in3[13,12,14,5])" };

/// @brief Row 81's grid: the goal block, its step block on 9 and its zero
///        block.
struct Row81Grid {
    gl::tests::ProverFixture ea;
    gl::Memory* goal = nullptr;
    gl::Memory* step = nullptr;
    gl::Memory* zero = nullptr;
    Row81Grid() : ea("FTA") {
        ea.proofGridBuilder.addTheoremToMemory(kRow81, ea.body, 0, false, ea.globalDependencies);
        goal = blockAt(ea, kRow81Chain);
        step = ea.simpleMapStore.findChild(goal, "(in2[rec0,9,3])");
        zero = ea.simpleMapStore.findChild(goal, "(=[s(rec0),2])");
        ASSERT_TRUE(step != nullptr && zero != nullptr);
    }
};

} // anonymous namespace

TEST(induction_usefulness, config_loader_reads_uniquely_definable) {
    gl::tests::ProverFixture ea("FTA");
    const auto indices = [&](const char* core) {
        const ce::CoreExpressionConfig* cfg = ea.definitions.coreConfig(gl::StrSpan(std::string(core)));
        ASSERT_TRUE(cfg != nullptr);
        return cfg->uniquelyDefinableIndices;
    };
    ASSERT_TRUE(indices("fold") == std::vector<int>({ 7 }));
    ASSERT_TRUE(indices("in2") == std::vector<int>({ 1 }));
    ASSERT_TRUE(indices("in3") == std::vector<int>({ 2 }));
    ASSERT_TRUE(indices("appendToSequence") == std::vector<int>({ 3 }));
    ASSERT_TRUE(indices("sequence").empty());
    ASSERT_TRUE(indices("nontrivialFactor").empty());
}

// The prover loads every Config<Anchor>.json; each definition entry states
// its list (the loader asserts it), so a new entry without the field fails
// here before it fails a run.
TEST(induction_usefulness, every_config_definition_lists_uniquely_definable) {
    const std::filesystem::path dir = gl::projectRoot() / "files" / "config";
    int configs = 0;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        const std::string name = entry.path().filename().string();
        if (name.rfind("Config", 0) != 0 || entry.path().extension() != ".json") continue;
        std::ifstream in(entry.path());
        ASSERT_TRUE(in.is_open());
        nlohmann::json j;
        in >> j;
        for (auto it = j.begin(); it != j.end(); ++it) {
            if (!it.value().is_object() || !it.value().contains("arity")) continue;
            ASSERT_TRUE(it.value().contains("uniquely_definable"));
            ASSERT_TRUE(it.value()["uniquely_definable"].is_array());
        }
        ++configs;
    }
    ASSERT_TRUE(configs >= 9);
}

// The relations state whether exchanging their two inputs keeps their
// meaning: coprime, bez1 and = do, preorder and strictOrder do not; an
// operator is no relation and keeps the default
// (D-595).
TEST(induction_usefulness, config_loader_reads_symmetric) {
    gl::tests::ProverFixture ea("FTA");
    const auto symmetric = [&](const char* core) {
        const ce::CoreExpressionConfig* cfg = ea.definitions.coreConfig(gl::StrSpan(std::string(core)));
        ASSERT_TRUE(cfg != nullptr);
        return cfg->symmetric;
    };
    ASSERT_TRUE(symmetric("coprime"));
    ASSERT_TRUE(symmetric("bez1"));
    ASSERT_TRUE(symmetric("="));
    ASSERT_FALSE(symmetric("preorder"));
    ASSERT_FALSE(symmetric("strictOrder"));
    ASSERT_FALSE(symmetric("in3"));
}

// The prover loads every Config<Anchor>.json; exactly the relations (two
// inputs, no output) state symmetric (the loader asserts it), so a new
// relation without the flag fails here before it fails a run.
TEST(induction_usefulness, every_config_relation_states_symmetric) {
    const std::filesystem::path dir = gl::projectRoot() / "files" / "config";
    int relations = 0;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        const std::string name = entry.path().filename().string();
        if (name.rfind("Config", 0) != 0 || entry.path().extension() != ".json") continue;
        std::ifstream in(entry.path());
        ASSERT_TRUE(in.is_open());
        nlohmann::json j;
        in >> j;
        for (auto it = j.begin(); it != j.end(); ++it) {
            if (!it.value().is_object() || !it.value().contains("arity")) continue;
            const nlohmann::json& spec = it.value();
            const bool relation = spec.contains("input_args") && spec["input_args"].size() == 2
                && (!spec.contains("output_args") || spec["output_args"].empty());
            ASSERT_TRUE(spec.contains("symmetric") == relation);
            if (!relation) continue;
            ASSERT_TRUE(spec["symmetric"].is_boolean());
            ++relations;
        }
    }
    ASSERT_TRUE(relations >= 9);
}

TEST(induction_usefulness, test_variable_is_its_own_last_ranked_kind) {
    const gl::WitnessName test = gl::classifyWitnessName(gl::StrSpan(std::string("test_level_6_0")));
    ASSERT_TRUE(test.kind == gl::WitnessNameKind::Test);
    ASSERT_EQ(test.level, 6);
    ASSERT_EQ(test.serial, 0);
    ASSERT_TRUE(gl::classifyWitnessName(gl::StrSpan(std::string("test_level_6"))).kind
        == gl::WitnessNameKind::Ordinary);
    ASSERT_TRUE(gl::classifyWitnessName(gl::StrSpan(std::string("test_6_0"))).kind
        == gl::WitnessNameKind::Ordinary);
    ASSERT_TRUE(gl::compareWitnessNames(gl::StrSpan(std::string("9")),
        gl::StrSpan(std::string("test_level_6_0"))) < 0);
    ASSERT_TRUE(gl::compareWitnessNames(gl::StrSpan(std::string("it_0_lev_6_1")),
        gl::StrSpan(std::string("test_level_6_0"))) < 0);
    ASSERT_TRUE(gl::compareWitnessNames(gl::StrSpan(std::string("test_level_6_0")),
        gl::StrSpan(std::string("force_lev_6_1"))) > 0);
}

// The usefulness scope is a child of main; it and its descendants are
// restricted like an integration trial, it is no induction scope although its
// payload starts with `is_`, and it holds no goal.
TEST(induction_usefulness, usefulness_scope_is_a_restricted_child_of_main) {
    using gl::ScopeSyntax;
    gl::Memory m;
    gl::NameMap& nm = m.nameMap;
    const gl::NameId main = nm.encode(std::string("main"));
    ASSERT_EQ(main, gl::NameMap::MAIN_ID);
    const gl::NameId useful = nm.encode(usefulnessScopeName());
    const gl::NameId below = nm.encodePush(useful, std::string("(implication7[a])"));
    const gl::NameId induction = nm.encodePush(main, std::string("is_(in2[marker,7,3])_8"));
    const gl::NameId trial = nm.encodePush(main, std::string("integration_int_0_lev_6_119"));
    const gl::NameId sibling = nm.encodePush(main, std::string("(implication8[a])"));
    ASSERT_EQ(nm.parentOf(useful), main);
    ASSERT_EQ(nm.decode(useful), std::string("main_boundary_is_useful"));
    ASSERT_TRUE(ScopeSyntax::isUsefulnessPayload(gl::StrSpan(std::string("is_useful"))));
    ASSERT_FALSE(ScopeSyntax::isUsefulnessPayload(gl::StrSpan(std::string("is_usefulx"))));
    ASSERT_FALSE(ScopeSyntax::isUsefulnessPayload(gl::StrSpan(std::string("is_(in2[marker,7,3])_8"))));
    ASSERT_TRUE(ScopeSyntax::atOrBelowUsefulnessScope(nm, useful));
    ASSERT_TRUE(ScopeSyntax::atOrBelowUsefulnessScope(nm, below));
    ASSERT_FALSE(ScopeSyntax::atOrBelowUsefulnessScope(nm, main));
    ASSERT_FALSE(ScopeSyntax::atOrBelowUsefulnessScope(nm, induction));
    ASSERT_FALSE(ScopeSyntax::atOrBelowUsefulnessScope(nm, trial));
    ASSERT_TRUE(ScopeSyntax::atOrBelowRestrictedScope(nm, useful));
    ASSERT_TRUE(ScopeSyntax::atOrBelowRestrictedScope(nm, below));
    ASSERT_TRUE(ScopeSyntax::atOrBelowRestrictedScope(nm, trial));
    ASSERT_FALSE(ScopeSyntax::atOrBelowRestrictedScope(nm, main));
    ASSERT_FALSE(ScopeSyntax::atOrBelowRestrictedScope(nm, sibling));
    ASSERT_FALSE(ScopeSyntax::atOrBelowRestrictedScope(nm, induction));
    ASSERT_FALSE(ScopeSyntax::isInductionScope(nm, useful));
    ASSERT_TRUE(ScopeSyntax::isInductionScope(nm, induction));
    ASSERT_FALSE(ScopeSyntax::canHoldGoal(nm, useful));
}

TEST(induction_usefulness, with_argument_at_replaces_one_slot) {
    ASSERT_EQ(gl::InductionUsefulness::withArgumentAt("(fold[1,3,4,5,11,2,9,13])", 7, "marker"),
        std::string("(fold[1,3,4,5,11,2,9,marker])"));
    ASSERT_EQ(gl::InductionUsefulness::withArgumentAt("(in3[9,9,10,4])", 0, "x"),
        std::string("(in3[x,9,10,4])"));
}

TEST(induction_usefulness, qualify_builds_row81_test) {
    gl::tests::ProverFixture ea("FTA");
    const auto tpl = gl::tests::ProofGridBuilderFixture::createAuxyImplication(ea.proofGridBuilder,
        std::string(kRow81), std::string("9"), std::string("rec"),
        ea.proofGridBuilder.findImmutableArgs(kRow81, "9"), ea.anchorInfo.name);
    gl::InductionUsefulness::Test test;
    ASSERT_TRUE(ea.inductionUsefulness.qualify(kRow81Chain, ea.anchorInfo.name, std::get<1>(tpl),
        std::get<2>(tpl), 6, test));
    ASSERT_EQ(test.key, std::string("(fold[1,3,4,5,11,2,9,marker])"));
    ASSERT_EQ(test.head, std::string("(fold[1,3,4,5,11,2,10,test_level_6_0])"));
    ASSERT_EQ(test.testVariable, std::string("test_level_6_0"));
    ASSERT_EQ(test.openSlot, 7);
    const std::vector<std::string> premises = {
        "(in2[rec,10,3])", "(sequence[1,4,2,10,11])", "(in2[10,12,11])",
        "(fold[1,3,4,5,11,2,rec,13])", "(in3[13,12,14,5])" };
    ASSERT_TRUE(test.premises == premises);
}

TEST(induction_usefulness, qualify_refuses_other_shapes) {
    gl::tests::ProverFixture ea("FTA");
    gl::InductionUsefulness::Test test;
    const std::vector<std::string> premises = { "(in2[u_rec,10,u_3])", "(fold[u_1,u_3,u_4,u_5,11,u_2,u_rec,13])" };
    // An equality head is no operator.
    ASSERT_FALSE(ea.inductionUsefulness.qualify(kRow81Chain, ea.anchorInfo.name, premises,
        "(=[10,13])", 6, test));
    // A negated head.
    ASSERT_FALSE(ea.inductionUsefulness.qualify(kRow81Chain, ea.anchorInfo.name, premises,
        "!(fold[u_1,u_3,u_4,u_5,11,u_2,10,14])", 6, test));
    // A fixed output.
    ASSERT_FALSE(ea.inductionUsefulness.qualify(kRow81Chain, ea.anchorInfo.name, premises,
        "(fold[u_1,u_3,u_4,u_5,11,u_2,10,u_14])", 6, test));
    // A head input no premise names.
    ASSERT_FALSE(ea.inductionUsefulness.qualify(kRow81Chain, ea.anchorInfo.name, premises,
        "(fold[u_1,u_3,u_4,u_5,15,u_2,10,14])", 6, test));
    // An operator whose output is not uniquely definable.
    ASSERT_FALSE(ea.inductionUsefulness.qualify(kRow81Chain, ea.anchorInfo.name, premises,
        "(nontrivialFactor[u_1,u_4,u_5,u_7,10,13])", 6, test));
    // No conjecture premise with the head's operator.
    ASSERT_FALSE(ea.inductionUsefulness.qualify(kRow81Chain, ea.anchorInfo.name, premises,
        "(quotient[u_1,u_4,u_5,10,13,14])", 6, test));
    // Two conjecture premises with the head's operator.
    std::vector<std::string> twoFolds = kRow81Chain;
    twoFolds.push_back("(fold[1,3,4,5,11,2,10,15])");
    ASSERT_FALSE(ea.inductionUsefulness.qualify(twoFolds, ea.anchorInfo.name, premises,
        "(fold[u_1,u_3,u_4,u_5,11,u_2,10,14])", 6, test));
    // The row 81 head itself qualifies with the same premises.
    ASSERT_TRUE(ea.inductionUsefulness.qualify(kRow81Chain, ea.anchorInfo.name, premises,
        "(fold[u_1,u_3,u_4,u_5,11,u_2,10,14])", 6, test));
}

// Row 81's step: the grid build plants the hypothesis premises and the head
// at the usefulness scope, disintegrates none of them, records the test and
// leaves the block active; nothing catches the head yet.
TEST(induction_usefulness, row81_step_is_armed_at_grid_build) {
    Row81Grid g;
    ASSERT_TRUE(g.step->isActive);
    ASSERT_FALSE(g.zero->isActive);
    ASSERT_TRUE(g.goal->isActive);
    ASSERT_TRUE(g.step->usefulnessTest == gl::Memory::UsefulnessTest::Pending);
    ASSERT_TRUE(g.goal->usefulnessTest == gl::Memory::UsefulnessTest::None);
    const gl::NameId scope = g.step->nameMap.lookup(usefulnessScopeName());
    ASSERT_TRUE(scope != 0);
    ASSERT_EQ(g.step->usefulnessScopeId, scope);
    ASSERT_EQ(g.step->nameMap.parentOf(scope), gl::NameMap::MAIN_ID);
    ASSERT_EQ(g.step->nameMap.decode(g.step->usefulnessHeadId),
        std::string("(fold[1,3,4,5,11,2,10,test_level_6_0])"));
    ASSERT_EQ(g.step->nameMap.decode(g.step->usefulnessKeyId),
        std::string("(fold[1,3,4,5,11,2,9,marker])"));
    ASSERT_EQ(g.step->usefulnessOpenSlot, 7);
    for (const char* row : { "(in2[rec,10,3])", "(sequence[1,4,2,10,11])", "(in2[10,12,11])",
                             "(fold[1,3,4,5,11,2,rec,13])", "(in3[13,12,14,5])",
                             "(fold[1,3,4,5,11,2,10,test_level_6_0])" })
        ASSERT_TRUE(knownAt(*g.step, row, scope));
    // Only the planted rows: no disintegration product reached the scope.
    ASSERT_EQ(rowsAt(*g.step, scope), 6);
    // The seed and the hypothesis rule stay at main.
    ASSERT_TRUE(g.step->recursionAuxyImplId != 0);
    ASSERT_TRUE(knownAt(*g.step, "(in2[rec,9,3])", gl::NameMap::MAIN_ID));
    ASSERT_FALSE(g.ea.inductionUsefulness.caught(*g.step));
}

// The equality the successor's uniqueness derives at the scope, 9 = 10, lets
// the key fold(P, 9) = marker catch the head fold(P, 10) = ?; an equality of
// other names does not.
TEST(induction_usefulness, uniqueness_join_lets_the_key_catch_the_head) {
    Row81Grid caughtGrid;
    depositAtUsefulnessScope(caughtGrid.ea, *caughtGrid.step, "(=[9,10])");
    ASSERT_TRUE(caughtGrid.ea.inductionUsefulness.caught(*caughtGrid.step));
    // The class lives at the scope only.
    ASSERT_FALSE(knownAt(*caughtGrid.step, "(=[9,10])", gl::NameMap::MAIN_ID));

    Row81Grid other;
    depositAtUsefulnessScope(other.ea, *other.step, "(=[10,12])");
    ASSERT_FALSE(other.ea.inductionUsefulness.caught(*other.step));
}

// Through burst 3 an uncaught test waits; a catch flags the block and
// queues the scope for the wipe; a settled test is not settled again.
TEST(induction_usefulness, settle_waits_through_burst_three_and_flags_a_catch) {
    Row81Grid g;
    g.ea.proofBurstNumber = gl::InductionUsefulness::kTestBursts;
    const gl::NameId scope = g.step->usefulnessScopeId;
    g.ea.inductionUsefulness.settle(*g.step);
    ASSERT_TRUE(g.step->usefulnessTest == gl::Memory::UsefulnessTest::Pending);
    ASSERT_EQ(g.step->pendingWipeScopes.lookup(scope), 0);
    depositAtUsefulnessScope(g.ea, *g.step, "(=[9,10])");
    g.ea.inductionUsefulness.settle(*g.step);
    ASSERT_TRUE(g.step->usefulnessTest == gl::Memory::UsefulnessTest::Caught);
    ASSERT_TRUE(g.step->pendingWipeScopes.lookup(scope) != 0);
    ASSERT_TRUE(g.step->isActive);
    g.ea.inductionUsefulness.settle(*g.step);
    ASSERT_TRUE(g.step->usefulnessTest == gl::Memory::UsefulnessTest::Caught);
    // A block without a test is left alone.
    g.ea.inductionUsefulness.settle(*g.goal);
    ASSERT_TRUE(g.goal->usefulnessTest == gl::Memory::UsefulnessTest::None);
}

// After burst 3 an uncaught test ends: the scope is queued for the wipe and
// the block keeps running. The test stays open in bursts 1 to 3.
TEST(induction_usefulness, settle_ends_an_uncaught_test_after_burst_three) {
    Row81Grid g;
    ASSERT_EQ(gl::InductionUsefulness::kTestBursts, 3);
    for (int burst = 1; burst <= gl::InductionUsefulness::kTestBursts; ++burst) {
        g.ea.proofBurstNumber = burst;
        g.ea.inductionUsefulness.settle(*g.step);
        ASSERT_TRUE(g.step->usefulnessTest == gl::Memory::UsefulnessTest::Pending);
        ASSERT_EQ(g.step->pendingWipeScopes.lookup(g.step->usefulnessScopeId), 0);
    }
    g.ea.proofBurstNumber = gl::InductionUsefulness::kTestBursts + 1;
    g.ea.inductionUsefulness.settle(*g.step);
    ASSERT_TRUE(g.step->usefulnessTest == gl::Memory::UsefulnessTest::Done);
    ASSERT_TRUE(g.step->pendingWipeScopes.lookup(g.step->usefulnessScopeId) != 0);
    ASSERT_TRUE(g.step->isActive);
}

// The barrier switches a caught step off; its zero block stays parked and
// the goal block stays active.
TEST(induction_usefulness, barrier_switches_a_caught_step_off) {
    Row81Grid g;
    g.ea.proofBurstNumber = 1;
    depositAtUsefulnessScope(g.ea, *g.step, "(=[9,10])");
    g.ea.inductionUsefulness.settle(*g.step);
    g.ea.deactivationSweepOwed = false;
    const std::vector<gl::Memory*> bodies = { g.goal, g.step, g.zero };
    g.ea.proofBarrier.settleUselessSteps(bodies);
    ASSERT_FALSE(g.step->isActive);
    ASSERT_TRUE(g.step->usefulnessTest == gl::Memory::UsefulnessTest::Done);
    ASSERT_TRUE(g.ea.deactivationSweepOwed);
    ASSERT_FALSE(g.zero->isActive);
    ASSERT_TRUE(g.goal->isActive);
    // A second barrier finds nothing caught.
    g.ea.deactivationSweepOwed = false;
    g.ea.proofBarrier.settleUselessSteps(bodies);
    ASSERT_FALSE(g.ea.deactivationSweepOwed);
}

// The switch-off names the triad; a block that already left the active set
// is reported without a second switch-off.
TEST(induction_usefulness, switch_off_reports_the_triad) {
    Row81Grid g;
    g.ea.proofBurstNumber = 1;
    depositAtUsefulnessScope(g.ea, *g.step, "(=[9,10])");
    g.ea.inductionUsefulness.settle(*g.step);
    const int original = g.ea.globalDependencies.stepBlockOriginal.at(g.step);
    g.ea.deactivationSweepOwed = false;
    const auto event = g.ea.scopeLifecycle.switchOffUselessStep(*g.step);
    ASSERT_TRUE(event.switchedOff);
    ASSERT_EQ(event.inductionVariable, std::string("9"));
    ASSERT_EQ(event.conjecture, g.ea.globalDependencies.originalAuxyMap.at(original).expr);
    ASSERT_TRUE(g.ea.deactivationSweepOwed);

    Row81Grid closed;
    closed.ea.proofBurstNumber = 1;
    depositAtUsefulnessScope(closed.ea, *closed.step, "(=[9,10])");
    closed.ea.inductionUsefulness.settle(*closed.step);
    closed.step->isActive = false;
    closed.ea.deactivationSweepOwed = false;
    const auto late = closed.ea.scopeLifecycle.switchOffUselessStep(*closed.step);
    ASSERT_FALSE(late.switchedOff);
    ASSERT_FALSE(closed.ea.deactivationSweepOwed);
    ASSERT_TRUE(closed.step->usefulnessTest == gl::Memory::UsefulnessTest::Done);
}

// A hypothesis whose head speaks about an output no planted fact ties to the
// key's input stays tested but uncaught: both weak steps are armed and active.
TEST(induction_usefulness, unrelated_head_is_armed_but_not_caught) {
    gl::tests::ProverFixture ea("Peano");
    ea.proofGridBuilder.addTheoremToMemory(kPeanoSuccessorOfSum, ea.body, 0, false, ea.globalDependencies);
    gl::Memory* goal = blockAt(ea, { "(AnchorPeano[1,2,3,4,5,6])", "(in3[7,8,9,4])",
        "(in2[8,10,3])", "(in3[7,10,11,4])" });
    int steps = 0;
    for (const char* key : { "(in2[rec0,7,3])", "(in2[rec1,8,3])" }) {
        gl::Memory* step = ea.simpleMapStore.findChild(goal, key);
        ASSERT_TRUE(step != nullptr);
        ASSERT_TRUE(step->isActive);
        ASSERT_TRUE(step->usefulnessTest == gl::Memory::UsefulnessTest::Pending);
        ASSERT_EQ(step->nameMap.decode(step->usefulnessKeyId), std::string("(in2[8,marker,3])"));
        ASSERT_FALSE(ea.inductionUsefulness.caught(*step));
        ++steps;
    }
    ASSERT_EQ(steps, 2);
}
