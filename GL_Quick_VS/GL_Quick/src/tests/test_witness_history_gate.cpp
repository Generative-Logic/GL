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
/// @brief Unit tests for the witness-history extension of the witness-link
///        admission gate (D-522).
///
/// @details
/// In an induction step block an integration instruction that would accept a
/// route also checks the derivation tree of the existence being
/// disintegrated: one derivation per statement, as buildStack takes it,
/// premises only. An ancestor whose leaf names a member of the instruction's
/// path list with the definition set of an admitted witness refuses the
/// route; operator outputs carry no definition set to check. Covers the leaf
/// scanner, the typing and operator-output helpers, the scope lift, the
/// history location, the walk, the gate predicate, the route probe and the
/// refusal marks. The disintegration-level cases live beside the integration
/// trial tests (`test_integration_trial.cpp`), whose fixture they share. The
/// strong-shot history gate (D-600) reads the
/// same walk; its cases (rule texts, shot conclusions, derivation from a shot,
/// the finalize drop) close the file.

#include "../memory.hpp"
#include "test_harness.hpp"
#include "prover_fixture.hpp"
#include "canonical_test_helpers.hpp"
#include "../logical_block/derivation_tree.hpp"
#include "../logical_block/strong_shot_history.hpp"
#include "../prover.hpp"

#include <algorithm>
#include <cstring>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {
    /// @brief Span over a string literal (tests only).
    gl::StrSpan sp(const char* s) {
        return gl::StrSpan(s, static_cast<int32_t>(std::strlen(s)));
    }

    /// @brief Owned copy of a span (tests only).
    std::string str(gl::StrSpan s) {
        return std::string(s.ptr, static_cast<std::size_t>(s.len));
    }

    /// @brief The tests' page-tier scratch arena.
    gl::ScratchArena& pages() {
        return gl::genScratchArenas().forSlot(gl::genScratchArenas().slotCount() - 1);
    }

    /// @brief The rule text every implication row of these tests cites first.
    const char* const kRule = "(>[1,2,3](in2[1,2,3])(in[1,u_N]))";

    /// @brief An induction step block below the anchor block
    ///        `(AnchorTest91[N,0,s])`, with `main` registered, its canonical
    ///        scope initialized and history tracked.
    struct HistoryFixture {
        gl::tests::ProverFixture analyzer{"Peano"};
        gl::Memory root;
        gl::Memory anchor;
        gl::Memory memory;

        HistoryFixture() {
            analyzer.anchorInfo.name = "AnchorTest91";
            analyzer.parameters.trackHistory = true;
            anchor.parentMemory = &root;
            memory.parentMemory = &anchor;
            anchor.setExprKey("(AnchorTest91[N,0,s])");
            memory.level = 2;
            memory.nameMap.encode(sp("main"));
            memory.recursionAuxyImplId = 1;
            analyzer.canonicalizer.initializeCanonicalScope(memory, gl::NameMap::MAIN_ID);
        }

        /// @brief Record one history row of @p expr at @p scope.
        void row(const char* expr, const char* scope, gl::OriginTag tag,
                 const std::vector<std::pair<const char*, const char*>>& deps) {
            std::vector<gl::OriginDep> d;
            for (const auto& dep : deps) d.push_back(gl::OriginDep{ sp(dep.first), sp(dep.second) });
            gl::addOriginEncoded(memory.exprOriginMap, memory.originInterner, sp(expr), sp(scope), tag,
                d.empty() ? nullptr : d.data(), static_cast<int>(d.size()), 30);
        }

        /// @brief The ancestors the walk visits from @p expr at @p scope, as
        ///        `expression@scope`, in visit order.
        std::vector<std::string> ancestors(const char* expr, const char* scope,
                                           const std::set<std::string>& endPoints = {}) {
            std::vector<std::string> out;
            gl::DerivationTree::walk(memory, sp(expr), sp(scope), [&](gl::StrSpan e, gl::StrSpan s) {
                out.push_back(str(e) + "@" + str(s));
                return false;
            }, [&](gl::StrSpan e, gl::StrSpan) { return endPoints.count(str(e)) != 0; });
            return out;
        }

        /// @brief List @p expr at @p scope as a forced product of the block.
        void force(const char* expr, gl::NameId scope) {
            memory.lbMemory.forcedProducts.mint(gl::packStatementKey(memory.nameMap.encode(sp(expr)), scope));
        }

        /// @brief The function-valued history of these tests: E's existence
        ///        derives from R, R from P (each named at a function slot), and
        ///        x appears only at a number slot.
        void functionHistory() {
            row("(in2[a,b,E])", "main", gl::OriginTag::implication, { { kRule, "main" }, { "(in2[c,d,R])", "main" } });
            row("(in2[c,d,R])", "main", gl::OriginTag::implication, { { kRule, "main" }, { "(in2[R,x,P])", "main" } });
            row("(in2[R,x,P])", "main", gl::OriginTag::taskFormulation, {});
        }

        /// @brief A probe rooted at `(in2[a,b,E])` checking a function witness.
        gl::WitnessHistoryProbe functionProbe() {
            gl::WitnessHistoryProbe probe;
            probe.start = sp("(in2[a,b,E])");
            probe.startScope = sp("main");
            probe.addLabel(analyzer.scopeTypes.definitionSetOf(sp("(in2[1,2,E])"), sp("E")));
            return probe;
        }
    };

    /// @brief File under @p key at `main` a one-entity instruction with the
    ///        path list @p list.
    void fileListed(gl::Memory& m, const std::string& key, const std::string& list) {
        gl::ScratchArena& arena = pages();
        gl::ScratchScope frame(arena);
        gl::WorkInstruction wi(&arena);
        const int32_t mark = wi.elemMark();
        const int32_t arity = static_cast<int32_t>(std::count(key.begin(), key.end(), ',')) + 1;
        wi.commitEntity(sp("atomic"), gl::StrSpan(key), sp(""), arity, mark);
        wi.setMarkedGoal(sp("(goal72[1,2,3])"));
        wi.setPathVariables(gl::StrSpan(list));
        const int64_t pk = gl::mintTemplateKey(m.templateInterner, m.nameMap, key, std::string("main"));
        gl::ArenaIntegrationMap entries(&arena, &m.valueInterner);
        entries.buildFromCold(m.overallHashMemory.admissionMapIntegration, pk);
        entries.findOrEmplaceWork(wi, m.valueInterner);
        entries.writeToCold(m.overallHashMemory.admissionMapIntegration, pk);
    }

    /// @brief The payload values of every instruction under @p key at `main`.
    std::vector<std::vector<std::string>> payloads(gl::Memory& m, const std::string& key) {
        gl::ScratchArena& arena = pages();
        gl::ScratchScope frame(arena);
        const int64_t pk = gl::mintTemplateKey(m.templateInterner, m.nameMap, key, std::string("main"));
        gl::ArenaIntegrationMap entries(&arena, &m.valueInterner);
        entries.buildFromCold(m.overallHashMemory.admissionMapIntegration, pk);
        std::vector<std::vector<std::string>> out;
        for (int32_t i = 0; i < entries.entries.size(); ++i) {
            int32_t ids[gl::ExecutionParameters::MAX_INTEGRATION_ENTRY_VALUES];
            const int32_t n = entries.valuesAt(i, ids, gl::ExecutionParameters::MAX_INTEGRATION_ENTRY_VALUES);
            std::vector<std::string> values;
            for (int32_t v = 0; v < n; ++v) values.push_back(m.valueInterner.decode(ids[v]));
            out.push_back(values);
        }
        return out;
    }

    /// @brief Put @p names into one class at `main` and seed the canonical
    ///        views.
    void classAtMain(gl::Memory& m, const std::set<std::string>& names) {
        gl::EquivalenceClass cls;
        cls.setMembersFromNames(names, m.nameMap);
        for (auto a : cls.memberIds) for (auto b : cls.memberIds)
            if (a < b) cls.intEqualityLevelsMap[gl::packEqPairKey(a, b)] = { 2 };
        m.assignClassesById(gl::NameMap::MAIN_ID, { cls });
        seedFixtureCanonicalClasses(m);
    }
} // namespace

// The scan visits every flat leaf once, left to right, skips binder lists and
// ignores a goal key's suffix; a stopping sink ends it.
TEST(witness_history_gate, leaf_scan_visits_every_flat_leaf) {
    std::vector<std::string> leaves;
    const auto collect = [&](gl::StrSpan leaf) { leaves.push_back(str(leaf)); return false; };
    ASSERT_FALSE(gl::forEachLeafSpan(sp("!(>[R](limitSequence[1,4,9,11,R])!(sequence[1,4,2,9,R]))"), collect));
    ASSERT_EQ(leaves.size(), std::size_t{2});
    ASSERT_EQ(leaves[0], std::string("(limitSequence[1,4,9,11,R])"));
    ASSERT_EQ(leaves[1], std::string("(sequence[1,4,2,9,R])"));
    leaves.clear();
    ASSERT_FALSE(gl::forEachLeafSpan(sp("(fold[1,3,4,5,11,2,10,14])_integration_goal"), collect));
    ASSERT_EQ(leaves.size(), std::size_t{1});
    ASSERT_EQ(leaves[0], std::string("(fold[1,3,4,5,11,2,10,14])"));
    leaves.clear();
    ASSERT_FALSE(gl::forEachLeafSpan(sp("(>[]!(=[u_12,x])!(=[u_10,y]))"), collect));
    ASSERT_EQ(leaves.size(), std::size_t{2});
    ASSERT_EQ(leaves[0], std::string("(=[u_12,x])"));
    ASSERT_EQ(leaves[1], std::string("(=[u_10,y])"));
    int visited = 0;
    ASSERT_TRUE(gl::forEachLeafSpan(sp("(in[a,N])(in[b,N])"), [&](gl::StrSpan) { ++visited; return true; }));
    ASSERT_EQ(visited, 1);
}

// The definition set of an argument follows the positional typing; negation
// changes no type and an absent argument has none.
TEST(witness_history_gate, definition_set_of_follows_the_typing) {
    gl::tests::ProverFixture analyzer("Peano");
    const gl::StrSpan number = analyzer.scopeTypes.definitionSetOf(sp("(in3[w,b,c,+])"), sp("w"));
    ASSERT_TRUE(number.len > 0);
    ASSERT_EQ(str(analyzer.scopeTypes.definitionSetOf(sp("(in3[w,b,c,+])"), sp("b"))), str(number));
    ASSERT_EQ(str(analyzer.scopeTypes.definitionSetOf(sp("!(in3[w,b,c,+])"), sp("c"))), str(number));
    const gl::StrSpan function = analyzer.scopeTypes.definitionSetOf(sp("(in2[b,w,s])"), sp("s"));
    ASSERT_TRUE(function.len > 0);
    ASSERT_TRUE(str(function) != str(number));
    ASSERT_EQ(str(analyzer.scopeTypes.definitionSetOf(sp("(in2[b,w,s])"), sp("w"))), str(number));
    ASSERT_TRUE(analyzer.scopeTypes.definitionSetOf(sp("(in[w,N])"), sp("x")).empty());
}

// A witness is an operator output exactly in an output slot of a positive
// operator application.
TEST(witness_history_gate, operator_output_is_an_output_slot_of_an_operator) {
    gl::tests::ProverFixture analyzer("Peano");
    const gl::AdmissionPolicy& policy = analyzer.admissionPolicy;
    ASSERT_TRUE(policy.isOperatorOutput(sp("(in2[i,v,f])"), sp("v")));
    ASSERT_FALSE(policy.isOperatorOutput(sp("(in2[i,v,f])"), sp("i")));
    ASSERT_FALSE(policy.isOperatorOutput(sp("(in2[i,v,f])"), sp("f")));
    ASSERT_TRUE(policy.isOperatorOutput(sp("(in3[a,b,c,+])"), sp("c")));
    ASSERT_FALSE(policy.isOperatorOutput(sp("(in3[a,b,c,+])"), sp("a")));
    ASSERT_FALSE(policy.isOperatorOutput(sp("!(in2[i,v,f])"), sp("v")));
    ASSERT_FALSE(policy.isOperatorOutput(sp("(in[v,N])"), sp("v")));
}

// The lift candidates are main, the boundary prefixes and the scope itself;
// an OR branch is the shallowest scope a dependency inside it may be read at.
TEST(witness_history_gate, lift_scopes_follow_boundaries_and_or_barriers) {
    gl::StrSpan out[8];
    int32_t minLift = -1;
    ASSERT_EQ(gl::DerivationTree::liftScopes(sp("main"), out, 8, minLift), 1);
    ASSERT_EQ(str(out[0]), std::string("main"));
    ASSERT_EQ(minLift, 0);
    ASSERT_EQ(gl::DerivationTree::liftScopes(sp("main_boundary_A_boundary_B"), out, 8, minLift), 3);
    ASSERT_EQ(str(out[1]), std::string("main_boundary_A"));
    ASSERT_EQ(str(out[2]), std::string("main_boundary_A_boundary_B"));
    ASSERT_EQ(minLift, 0);
    ASSERT_EQ(gl::DerivationTree::liftScopes(sp("main_boundary_A_boundary_orint_X_boundary_B"), out, 8, minLift), 4);
    ASSERT_EQ(str(out[2]), std::string("main_boundary_A_boundary_orint_X"));
    ASSERT_EQ(minLift, 2);
    ASSERT_EQ(gl::DerivationTree::liftScopes(sp("other"), out, 8, minLift), 1);
    ASSERT_TRUE(gl::DerivationTree::isIntegrationGoalKey(sp("(fold[1,2])_integration_goal")));
    ASSERT_FALSE(gl::DerivationTree::isIntegrationGoalKey(sp("(fold[1,2])")));
    ASSERT_TRUE(gl::DerivationTree::isRuleText(sp(kRule)));
    ASSERT_FALSE(gl::DerivationTree::isRuleText(sp("(in[a,N])")));
}

// A dependency is read at the shallowest scope holding its rows, a `zoom in`
// pair at its own scope, an equality without rows in its class; a pair
// without history has no location.
TEST(witness_history_gate, locate_lifts_and_reads_class_history) {
    HistoryFixture f;
    f.row("(in[d,N])", "main", gl::OriginTag::taskFormulation, {});
    f.row("(in[e,N])", "main", gl::OriginTag::taskFormulation, {});
    f.row("(in[e,N])", "main_boundary_K", gl::OriginTag::zoomIn, { { "(in[e,N])", "main" } });
    gl::ScratchArena& arena = pages();
    gl::ScratchScope frame(arena);
    gl::DerivationTree::Node node;
    ASSERT_TRUE(gl::DerivationTree::locate(f.memory, sp("(in[d,N])"), sp("main_boundary_K"), false, arena, node));
    ASSERT_EQ(node.key, gl::mintOriginKey(f.memory.originInterner, sp("(in[d,N])"), sp("main")));
    ASSERT_TRUE(node.bodyRun != 0);
    ASSERT_TRUE(gl::DerivationTree::locate(f.memory, sp("(in[e,N])"), sp("main_boundary_K"), false, arena, node));
    ASSERT_EQ(node.key, gl::mintOriginKey(f.memory.originInterner, sp("(in[e,N])"), sp("main_boundary_K")));
    ASSERT_FALSE(gl::DerivationTree::locate(f.memory, sp("(in[z,N])"), sp("main"), false, arena, node));

    gl::EquivalenceClass cls;
    cls.setMembersFromNames(std::set<std::string>{ "g", "k" }, f.memory.nameMap);
    for (auto a : cls.memberIds) for (auto b : cls.memberIds)
        if (a < b) cls.intEqualityLevelsMap[gl::packEqPairKey(a, b)] = { 2 };
    const int64_t equality = gl::mintOriginKey(f.memory.originInterner, sp("(=[g,k])"), sp("main"));
    cls.equalityOriginMap[equality] = { gl::IdOrigin{ gl::OriginTag::implication, {
        gl::mintOriginKey(f.memory.originInterner, sp(kRule), sp("main")),
        gl::mintOriginKey(f.memory.originInterner, sp("(in[g,N])"), sp("main")) } } };
    f.memory.assignClassesById(gl::NameMap::MAIN_ID, { cls });
    ASSERT_TRUE(gl::DerivationTree::locate(f.memory, sp("(=[g,k])"), sp("main"), false, arena, node));
    ASSERT_EQ(node.key, equality);
    ASSERT_EQ(node.bodyRun, 0);
    ASSERT_TRUE(node.classBucket != 0);
}

// The walk follows premises only, one row per statement: the applied rule
// and a second row are not followed, and the root is not visited.
TEST(witness_history_gate, walk_follows_one_derivation_of_premises) {
    HistoryFixture f;
    f.row("(in3[A,B,C,+])", "main", gl::OriginTag::implication,
        { { kRule, "main" }, { "(in2[R,A,s])", "main" }, { "(in[B,N])", "main" } });
    f.row("(in2[R,A,s])", "main", gl::OriginTag::implication, { { kRule, "main" }, { "(in2[P,R,s])", "main" } });
    f.row("(in2[R,A,s])", "main", gl::OriginTag::implication, { { kRule, "main" }, { "(in2[Q,R,s])", "main" } });
    f.row("(in2[P,R,s])", "main", gl::OriginTag::taskFormulation, {});
    f.row("(in2[Q,R,s])", "main", gl::OriginTag::taskFormulation, {});
    f.row("(in[B,N])", "main", gl::OriginTag::taskFormulation, {});
    const std::vector<std::string> seen = f.ancestors("(in3[A,B,C,+])", "main");
    ASSERT_EQ(seen.size(), std::size_t{3});
    ASSERT_EQ(seen[0], std::string("(in2[R,A,s])@main"));
    ASSERT_EQ(seen[1], std::string("(in2[P,R,s])@main"));
    ASSERT_EQ(seen[2], std::string("(in[B,N])@main"));
}

// Equality rows come after the other rows; a row closing a cycle on the walk
// path gives way to the next row; goals end the walk.
TEST(witness_history_gate, walk_orders_rows_and_ends_cycles_and_goals) {
    HistoryFixture f;
    f.row("(in2[X,Y,s])", "main", gl::OriginTag::equality1,
        { { "(=[Y,Z])", "main" }, { "(in2[X,Z,s])", "main" } });
    f.row("(in2[X,Y,s])", "main", gl::OriginTag::implication, { { kRule, "main" }, { "(in[X,N])", "main" } });
    f.row("(in[X,N])", "main", gl::OriginTag::taskFormulation, {});
    std::vector<std::string> seen = f.ancestors("(in2[X,Y,s])", "main");
    ASSERT_EQ(seen.size(), std::size_t{1});
    ASSERT_EQ(seen[0], std::string("(in[X,N])@main"));

    f.row("(in[a,N])", "main", gl::OriginTag::implication, { { kRule, "main" }, { "(in[b,N])", "main" } });
    f.row("(in[b,N])", "main", gl::OriginTag::implication, { { kRule, "main" }, { "(in[a,N])", "main" } });
    f.row("(in[b,N])", "main", gl::OriginTag::taskFormulation, {});
    seen = f.ancestors("(in[a,N])", "main");
    ASSERT_EQ(seen.size(), std::size_t{1});
    ASSERT_EQ(seen[0], std::string("(in[b,N])@main"));

    f.row("(in[m,N])", "main", gl::OriginTag::expansionForIntegration,
        { { "(fold[1,2,3,4,m,5,6,7])_integration_goal", "main" } });
    ASSERT_TRUE(f.ancestors("(in[m,N])", "main").empty());

    f.row("(in[c,N])", "main_boundary_K", gl::OriginTag::implication,
        { { kRule, "main" }, { "(in[d,N])", "main_boundary_K" } });
    f.row("(in[d,N])", "main", gl::OriginTag::taskFormulation, {});
    seen = f.ancestors("(in[c,N])", "main_boundary_K");
    ASSERT_EQ(seen.size(), std::size_t{1});
    ASSERT_EQ(seen[0], std::string("(in[d,N])@main"));
}

// An equality whose history its class holds is walked through the class's
// lines; a visitor returning true stops the walk.
TEST(witness_history_gate, walk_reads_class_history_and_stops_on_request) {
    HistoryFixture f;
    f.row("(in2[g,h,s])", "main", gl::OriginTag::equality1,
        { { "(=[g,k])", "main" }, { "(in2[k,h,s])", "main" } });
    f.row("(in2[k,h,s])", "main", gl::OriginTag::taskFormulation, {});
    f.row("(in[g,N])", "main", gl::OriginTag::taskFormulation, {});
    gl::EquivalenceClass cls;
    cls.setMembersFromNames(std::set<std::string>{ "g", "k" }, f.memory.nameMap);
    for (auto a : cls.memberIds) for (auto b : cls.memberIds)
        if (a < b) cls.intEqualityLevelsMap[gl::packEqPairKey(a, b)] = { 2 };
    cls.equalityOriginMap[gl::mintOriginKey(f.memory.originInterner, sp("(=[g,k])"), sp("main"))] = {
        gl::IdOrigin{ gl::OriginTag::implication, {
            gl::mintOriginKey(f.memory.originInterner, sp(kRule), sp("main")),
            gl::mintOriginKey(f.memory.originInterner, sp("(in[g,N])"), sp("main")) } } };
    f.memory.assignClassesById(gl::NameMap::MAIN_ID, { cls });
    const std::vector<std::string> seen = f.ancestors("(in2[g,h,s])", "main");
    ASSERT_EQ(seen.size(), std::size_t{3});
    ASSERT_EQ(seen[0], std::string("(=[g,k])@main"));
    ASSERT_EQ(seen[1], std::string("(in[g,N])@main"));
    ASSERT_EQ(seen[2], std::string("(in2[k,h,s])@main"));

    int32_t visited = 0;
    int calls = 0;
    ASSERT_TRUE(gl::DerivationTree::walk(f.memory, sp("(in2[g,h,s])"), sp("main"),
        [&](gl::StrSpan, gl::StrSpan) { ++calls; return true; },
        [](gl::StrSpan, gl::StrSpan) { return false; }, &visited));
    ASSERT_EQ(calls, 1);
    ASSERT_EQ(visited, 1);
}

// The tree refuses when an ancestor names a list member at a slot of the
// checked witness's definition set; a member named only at another kind of
// slot, an inactive probe, a non-step block or an empty list refuse nothing.
TEST(witness_history_gate, history_refuses_a_path_member_of_the_witness_kind) {
    HistoryFixture f;
    f.functionHistory();
    auto& gate = f.analyzer.admissionMemory;
    const gl::WitnessHistoryProbe probe = f.functionProbe();
    ASSERT_TRUE(probe.active());
    ASSERT_TRUE(gate.witnessLinkRefuses(f.memory, sp("[P]"), nullptr, 0, gl::NameMap::MAIN_ID, &probe));
    ASSERT_TRUE(gate.witnessLinkRefuses(f.memory, sp("[R]"), nullptr, 0, gl::NameMap::MAIN_ID, &probe));
    ASSERT_FALSE(gate.witnessLinkRefuses(f.memory, sp("[x]"), nullptr, 0, gl::NameMap::MAIN_ID, &probe));
    ASSERT_FALSE(gate.witnessLinkRefuses(f.memory, sp("[Q]"), nullptr, 0, gl::NameMap::MAIN_ID, &probe));
    ASSERT_FALSE(gate.witnessLinkRefuses(f.memory, gl::StrSpan(), nullptr, 0, gl::NameMap::MAIN_ID, &probe));
    const gl::WitnessHistoryProbe inactive;
    ASSERT_FALSE(gate.witnessLinkRefuses(f.memory, sp("[P]"), nullptr, 0, gl::NameMap::MAIN_ID, &inactive));
    const gl::StrSpan linkedA[1] = { sp("A") };
    ASSERT_TRUE(gate.witnessLinkRefuses(f.memory, sp("[P]"), linkedA, 1, gl::NameMap::MAIN_ID, &probe));
    f.memory.recursionAuxyImplId = 0;
    ASSERT_FALSE(gate.witnessLinkRefuses(f.memory, sp("[P]"), nullptr, 0, gl::NameMap::MAIN_ID, &probe));
}

// The tree compares through the classes of the admission scope: a class
// member of an ancestor's argument on the list refuses.
TEST(witness_history_gate, history_compares_through_the_classes) {
    HistoryFixture f;
    f.functionHistory();
    classAtMain(f.memory, std::set<std::string>{ "P", "T" });
    const gl::WitnessHistoryProbe probe = f.functionProbe();
    auto& gate = f.analyzer.admissionMemory;
    ASSERT_TRUE(gate.witnessLinkRefuses(f.memory, sp("[T]"), nullptr, 0, gl::NameMap::MAIN_ID, &probe));
    ASSERT_FALSE(gate.witnessLinkRefuses(f.memory, sp("[Q]"), nullptr, 0, gl::NameMap::MAIN_ID, &probe));
}

// The route probe counts only instructions the tree leaves open; a refused
// commit leaves the key untouched.
TEST(witness_history_gate, route_probe_counts_only_instructions_the_tree_leaves_open) {
    HistoryFixture f;
    f.functionHistory();
    const gl::WitnessHistoryProbe probe = f.functionProbe();
    auto& gate = f.analyzer.admissionMemory;
    const std::string key = "(in2[u_2,marker,u_3])";
    fileListed(f.memory, key, "[P]");
    const gl::StrSpan expression = sp("(in2[2,w,3])");
    const gl::StrSpan witness = sp("w");
    const gl::StrSpan marked = sp("(in2[2,marker,3])");
    ASSERT_FALSE(gate.isAdmittedIntegration(f.memory, expression, witness, marked, sp("main"), false, 0,
        nullptr, 0, &probe));
    ASSERT_TRUE(gate.isAdmittedIntegration(f.memory, expression, witness, marked, sp("main"), false));
    ASSERT_FALSE(gate.isAdmittedIntegration(f.memory, expression, witness, marked, sp("main"), true, 0,
        nullptr, 0, &probe));
    ASSERT_TRUE(payloads(f.memory, key)[0].empty());
    fileListed(f.memory, key, "[Q]");
    ASSERT_TRUE(gate.isAdmittedIntegration(f.memory, expression, witness, marked, sp("main"), false, 0,
        nullptr, 0, &probe));
}

// A park records the tree's refusal in the refusing instruction, and the
// demand probe then no longer counts it for that source.
TEST(witness_history_gate, tree_refusal_marks_close_the_demand) {
    HistoryFixture f;
    f.functionHistory();
    const gl::WitnessHistoryProbe probe = f.functionProbe();
    auto& gate = f.analyzer.admissionMemory;
    const std::string key = "(in3[u_2,u_3,marker,u_4])";
    const gl::StrSpan park = sp("(in3[2,3,marker,4])");
    const gl::StrSpan source = sp("(existence93[1,2,3,4,5])");
    fileListed(f.memory, key, "[P]");
    ASSERT_TRUE(gate.integrationDemandOpen(f.memory, park, sp("main"), source));
    gate.markWitnessLinkRefusals(f.memory, park, sp("main"), nullptr, 0, source, &probe);
    const std::vector<std::vector<std::string>> values = payloads(f.memory, key);
    ASSERT_EQ(values.size(), std::size_t{1});
    ASSERT_EQ(values[0].size(), std::size_t{1});
    ASSERT_EQ(values[0][0], std::string("refused:(existence93[1,2,3,4,5])"));
    ASSERT_FALSE(gate.integrationDemandOpen(f.memory, park, sp("main"), source));
    fileListed(f.memory, key, "[Q]");
    ASSERT_TRUE(gate.integrationDemandOpen(f.memory, park, sp("main"), source));
}

// An end point is neither visited nor followed: ending the walk at R's fact
// hides P's fact behind it, ending it at P's fact leaves R's visited.
TEST(witness_history_gate, walk_neither_visits_nor_follows_an_end_point) {
    HistoryFixture f;
    f.functionHistory();
    ASSERT_TRUE(f.ancestors("(in2[a,b,E])", "main", { "(in2[c,d,R])" }).empty());
    const std::vector<std::string> seen = f.ancestors("(in2[a,b,E])", "main", { "(in2[R,x,P])" });
    ASSERT_EQ(seen.size(), std::size_t{1});
    ASSERT_EQ(seen[0], std::string("(in2[c,d,R])@main"));
    int32_t visited = -1;
    ASSERT_FALSE(gl::DerivationTree::walk(f.memory, sp("(in2[a,b,E])"), sp("main"),
        [](gl::StrSpan, gl::StrSpan) { return false; },
        [](gl::StrSpan, gl::StrSpan) { return true; }, &visited));
    ASSERT_EQ(visited, 0);
}

// The matcher finds a forced product listed at the visited scope or at an
// ancestor scope, and a class rewrite of one; a product listed at a child
// scope is not visible at main, and an unlisted statement never matches.
TEST(witness_history_gate, forced_product_match_is_exact_inherited_and_canonical) {
    HistoryFixture f;
    gl::ForcedProductMatch empty(f.memory, f.analyzer.canonicalTemplates, pages(),
        gl::scratchArenas().forSlot(gl::scratchArenas().slotCount() - 1));
    ASSERT_FALSE(empty.matches(sp("(fXY[A,M,N])"), sp("main")));
    const gl::NameId child = f.memory.nameMap.encodePush(gl::NameMap::MAIN_ID, sp("(sub91[1])"));
    f.analyzer.canonicalizer.initializeCanonicalScope(f.memory, child);
    const std::string childName = f.memory.nameMap.decode(child);
    f.force("(fXY[A,M,N])", gl::NameMap::MAIN_ID);
    f.force("(in[q,N])", child);
    classAtMain(f.memory, std::set<std::string>{ "A", "X" });
    gl::ForcedProductMatch match(f.memory, f.analyzer.canonicalTemplates, pages(),
        gl::scratchArenas().forSlot(gl::scratchArenas().slotCount() - 1));
    ASSERT_TRUE(match.matches(sp("(fXY[A,M,N])"), sp("main")));
    ASSERT_TRUE(match.matches(sp("(fXY[A,M,N])"), gl::StrSpan(childName)));
    ASSERT_TRUE(match.matches(sp("(fXY[X,M,N])"), sp("main")));
    ASSERT_TRUE(match.matches(sp("(in[q,N])"), gl::StrSpan(childName)));
    ASSERT_FALSE(match.matches(sp("(in[q,N])"), sp("main")));
    ASSERT_FALSE(match.matches(sp("(fXY[B,M,N])"), sp("main")));
    // A scope the block never registered (a mailed row cites the sender's
    // scope) reads as its deepest registered prefix; with none, nothing
    // matches.
    const std::string belowMain = "main_boundary_(unknown91[1])";
    const std::string belowChild = childName + "_boundary_(unknown92[2])";
    ASSERT_EQ(f.memory.nameMap.lookup(gl::StrSpan(belowMain)), gl::NameId(0));
    ASSERT_EQ(f.memory.nameMap.lookup(gl::StrSpan(belowChild)), gl::NameId(0));
    ASSERT_TRUE(match.matches(sp("(fXY[A,M,N])"), gl::StrSpan(belowMain)));
    ASSERT_TRUE(match.matches(sp("(fXY[X,M,N])"), gl::StrSpan(belowMain)));
    ASSERT_FALSE(match.matches(sp("(in[q,N])"), gl::StrSpan(belowMain)));
    ASSERT_TRUE(match.matches(sp("(in[q,N])"), gl::StrSpan(belowChild)));
    ASSERT_FALSE(match.matches(sp("(fXY[A,M,N])"), sp("elsewhere91")));
}

// A connection that comes from a forced disintegration does not count: with
// R's fact a forced product the tree no longer reaches P, as with a class
// rewrite of that fact listed instead; unlisted, the same tree refuses.
TEST(witness_history_gate, history_ignores_connections_through_forced_products) {
    for (const int listing : { 0, 1, 2 }) {
        HistoryFixture f;
        f.functionHistory();
        if (listing == 1) f.force("(in2[c,d,R])", gl::NameMap::MAIN_ID);
        if (listing == 2) {
            f.force("(in2[c,d,Y])", gl::NameMap::MAIN_ID);
            classAtMain(f.memory, std::set<std::string>{ "R", "Y" });
        }
        const gl::WitnessHistoryProbe probe = f.functionProbe();
        auto& gate = f.analyzer.admissionMemory;
        ASSERT_EQ(gate.witnessLinkRefuses(f.memory, sp("[P]"), nullptr, 0, gl::NameMap::MAIN_ID, &probe),
            listing == 0);
    }
}

// A located statement's rows come in insertion order with their dependency
// counts; the callback stops the visit; an equality whose history its class
// holds yields that class's lines under its key.
TEST(witness_history_gate, for_each_row_reads_body_rows_and_class_lines) {
    HistoryFixture f;
    f.row("(in[x,N])", "main", gl::OriginTag::implication, { { kRule, "main" }, { "(in[y,N])", "main" } });
    f.row("(in[x,N])", "main", gl::OriginTag::taskFormulation, {});
    gl::DerivationTree::Node node;
    ASSERT_TRUE(gl::DerivationTree::locate(f.memory, sp("(in[x,N])"), sp("main"), true, pages(), node));
    std::vector<int> tags;
    std::vector<int> depCounts;
    gl::DerivationTree::forEachRow(f.memory, node, pages(), [&](uint8_t tag, const auto&, int32_t depN) {
        tags.push_back(tag);
        depCounts.push_back(depN);
        return false;
    });
    ASSERT_EQ(tags.size(), std::size_t{2});
    ASSERT_EQ(tags[0], static_cast<int>(gl::OriginTag::implication));
    ASSERT_EQ(depCounts[0], 2);
    ASSERT_EQ(tags[1], static_cast<int>(gl::OriginTag::taskFormulation));
    ASSERT_EQ(depCounts[1], 0);
    int calls = 0;
    gl::DerivationTree::forEachRow(f.memory, node, pages(), [&](uint8_t, const auto&, int32_t) {
        ++calls;
        return true;
    });
    ASSERT_EQ(calls, 1);

    gl::EquivalenceClass cls;
    cls.setMembersFromNames(std::set<std::string>{ "g", "k" }, f.memory.nameMap);
    for (auto a : cls.memberIds) for (auto b : cls.memberIds)
        if (a < b) cls.intEqualityLevelsMap[gl::packEqPairKey(a, b)] = { 2 };
    const int64_t ruleKey = gl::mintOriginKey(f.memory.originInterner, sp(kRule), sp("main"));
    cls.equalityOriginMap[gl::mintOriginKey(f.memory.originInterner, sp("(=[g,k])"), sp("main"))] = {
        gl::IdOrigin{ gl::OriginTag::implication, {
            ruleKey, gl::mintOriginKey(f.memory.originInterner, sp("(in[g,N])"), sp("main")) } } };
    f.memory.assignClassesById(gl::NameMap::MAIN_ID, { cls });
    gl::DerivationTree::Node classNode;
    ASSERT_TRUE(gl::DerivationTree::locate(f.memory, sp("(=[g,k])"), sp("main"), true, pages(), classNode));
    ASSERT_EQ(classNode.bodyRun, 0);
    std::vector<int64_t> firstDeps;
    gl::DerivationTree::forEachRow(f.memory, classNode, pages(), [&](uint8_t tag, const auto& depAt, int32_t depN) {
        ASSERT_EQ(static_cast<int>(tag), static_cast<int>(gl::OriginTag::implication));
        ASSERT_EQ(depN, 2);
        firstDeps.push_back(depAt(0));
        return false;
    });
    ASSERT_EQ(firstDeps.size(), std::size_t{1});
    ASSERT_EQ(firstDeps[0], ruleKey);
}

// ---- Strong-shot history gate (D-600) ---------

namespace {
    /// @brief The strong hypothesis text the strong-shot tests install.
    const char* const kHypothesis = "(>[11,12](in3[11,12,u_n,u_+])(>[]!(=[11,u_n])(in[11,u_N])))";

    /// @brief The multiplication catcher text the strong-shot tests install.
    const char* const kCatcher = "(>[11,12](in3[12,11,u_n,u_*])(>[]!(=[11,u_n])(in[11,u_N])))";

    /// @brief A strong step block with the hypothesis installed: the seeds
    ///        `(in[a,N])` and `(in[b,N])`, the shot conclusion `(rem[a,b])`,
    ///        its forced product `(in[r,N])` and `(in[g,N])`, built two
    ///        derivations below that product.
    struct ShotFixture : HistoryFixture {
        ShotFixture() {
            memory.strongStepBlock = true;
            memory.recursionAuxyImplId = memory.ruleInterner.encode(sp(kHypothesis));
            row("(in[a,N])", "main", gl::OriginTag::taskFormulation, {});
            row("(in[b,N])", "main", gl::OriginTag::taskFormulation, {});
            row("(rem[a,b])", "main", gl::OriginTag::implication,
                { { kHypothesis, "main" }, { "(in[a,N])", "main" }, { "(in[b,N])", "main" } });
            row("(in[r,N])", "main", gl::OriginTag::disintegration, { { "(rem[a,b])", "main" } });
            row("(in3[r,1,g,+])", "main", gl::OriginTag::implication,
                { { kRule, "main" }, { "(in[r,N])", "main" } });
            row("(in[g,N])", "main", gl::OriginTag::implication,
                { { kRule, "main" }, { "(in3[r,1,g,+])", "main" } });
        }

        /// @brief Whether @p expr at `main` derives from a strong shot.
        bool derives(const char* expr) {
            return gl::StrongShotHistory::derivesFromShot(memory, sp(expr), sp("main"),
                gl::StrongShotHistory::inductionRuleTexts(memory));
        }

        /// @brief Whether @p expr at `main` is a strong shot's conclusion.
        bool conclusion(const char* expr) {
            gl::DerivationTree::Node node;
            const bool located = gl::DerivationTree::locate(memory, sp(expr), sp("main"), true, pages(), node);
            ASSERT_TRUE(located);
            return gl::StrongShotHistory::isShotConclusion(memory, node,
                gl::StrongShotHistory::inductionRuleTexts(memory), pages());
        }
    };

    /// @brief Sealed-page rig for the finalize tests' firing records; seals
    ///        and frees its pages in its destructor.
    struct ShotRecordRig {
        gl::SealedPageSet pages;
        ShotRecordRig() { pages.bind(&gl::staticMemory()); }
        ~ShotRecordRig() {
            if (pages.filling()) pages.seal();
            pages.freePages();
        }

        /// @brief Seal @p text onto the rig's pages.
        gl::SealedString s(const char* text) {
            return gl::SealedString::copyFrom(pages, text, static_cast<int32_t>(std::strlen(text)));
        }

        /// @brief Append a head firing of the hypothesis that concludes
        ///        @p head at `main` from @p premises.
        void shot(const char* head, const std::vector<const char*>& premises) {
            gl::FiringRecord r;
            r.isMarker = false;
            r.rplExpr2 = s(head);
            r.validityName = s("main");
            const int levels[1] = { 0 };
            r.levels = gl::SealedSpan<int>::copyFrom(pages, levels, 1);
            r.originTag = s("implication");
            std::vector<gl::SealedExpressionWithValidity> deps;
            deps.push_back(gl::SealedExpressionWithValidity{ s(kHypothesis), s("main") });
            for (const char* p : premises) deps.push_back(gl::SealedExpressionWithValidity{ s(p), s("main") });
            r.originDeps = gl::SealedSpan<gl::SealedExpressionWithValidity>::copyFrom(
                pages, deps.data(), static_cast<int32_t>(deps.size()));
            r.recursionHead = true;
            r.alreadyKnown = false;
            pages.appendRecord(r);
        }
    };

    /// @brief The statements a block's same-iteration internal mail holds.
    std::set<std::string> internalStatements(gl::Memory& m) {
        std::set<std::string> out;
        const gl::Mail mail = gl::makeHeapMail(m.sameIterationInternalMail, m.nameMap, m.originInterner);
        for (const auto& st : mail.statements) out.insert(st.first.original);
        return out;
    }
} // namespace

// The rule texts are found once a row cites them: the hypothesis first, the
// multiplication catcher second, each 0 while no row cites it.
TEST(strong_shot_history, rule_texts_follow_the_cited_implications) {
    HistoryFixture f;
    f.memory.strongStepBlock = true;
    f.memory.recursionAuxyImplId = f.memory.ruleInterner.encode(sp(kHypothesis));
    gl::StrongShotHistory::RuleTexts texts = gl::StrongShotHistory::inductionRuleTexts(f.memory);
    ASSERT_EQ(texts.ids[0], 0);
    ASSERT_EQ(texts.ids[1], 0);
    f.row("(rem[a,b])", "main", gl::OriginTag::implication, { { kHypothesis, "main" } });
    texts = gl::StrongShotHistory::inductionRuleTexts(f.memory);
    ASSERT_TRUE(texts.ids[0] != 0);
    ASSERT_EQ(texts.ids[1], 0);
    f.memory.recursionProductImplId = f.memory.ruleInterner.encode(sp(kCatcher));
    f.row("(rem[c,b])", "main", gl::OriginTag::implication, { { kCatcher, "main" } });
    texts = gl::StrongShotHistory::inductionRuleTexts(f.memory);
    ASSERT_TRUE(texts.ids[1] != 0);
    ASSERT_TRUE(texts.ids[0] != texts.ids[1]);
}

// Only a row that applied an induction implication makes a conclusion: a row
// of another rule, a seed and a disintegration product do not; the
// multiplication catcher's rows count like the hypothesis's.
TEST(strong_shot_history, shot_conclusion_needs_an_induction_implication_row) {
    ShotFixture f;
    ASSERT_TRUE(f.conclusion("(rem[a,b])"));
    ASSERT_FALSE(f.conclusion("(in[g,N])"));
    ASSERT_FALSE(f.conclusion("(in[a,N])"));
    ASSERT_FALSE(f.conclusion("(in[r,N])"));
    f.memory.recursionProductImplId = f.memory.ruleInterner.encode(sp(kCatcher));
    f.row("(rem[b,a])", "main", gl::OriginTag::implication, { { kCatcher, "main" }, { "(in[b,N])", "main" } });
    ASSERT_TRUE(f.conclusion("(rem[b,a])"));
}

// A conclusion, its forced product, a statement two derivations below it and
// the product of a class rewrite of it all derive from the shot; statements
// derived from seeds alone do not.
TEST(strong_shot_history, premise_derives_from_shot_directly_and_indirectly) {
    ShotFixture f;
    ASSERT_TRUE(f.derives("(rem[a,b])"));
    ASSERT_TRUE(f.derives("(in[r,N])"));
    ASSERT_TRUE(f.derives("(in[g,N])"));
    ASSERT_FALSE(f.derives("(in[a,N])"));
    f.row("(=[a,c])", "main", gl::OriginTag::taskFormulation, {});
    f.row("(rem[c,b])", "main", gl::OriginTag::equality1, { { "(rem[a,b])", "main" }, { "(=[a,c])", "main" } });
    f.row("(in[q,N])", "main", gl::OriginTag::disintegration, { { "(rem[c,b])", "main" } });
    ASSERT_TRUE(f.derives("(in[q,N])"));
    f.row("(in3[a,b,d,+])", "main", gl::OriginTag::implication,
        { { kRule, "main" }, { "(in[a,N])", "main" }, { "(in[b,N])", "main" } });
    ASSERT_FALSE(f.derives("(in3[a,b,d,+])"));
}

// The finalize drops a strong shot whose premise derives from a shot and keeps
// a shot on seeds; in a weak step block both shots stay.
TEST(strong_shot_history, finalize_drops_a_shot_on_its_own_products) {
    for (int strong = 0; strong < 2; ++strong) {
        ShotFixture f;
        f.memory.strongStepBlock = strong == 1;
        ShotRecordRig rig;
        rig.shot("(rem[a,g])", { "(in[a,N])", "(in[g,N])" });
        rig.shot("(rem[b,a])", { "(in[b,N])", "(in[a,N])" });
        gl::SealedPageSet* parts[1] = { &rig.pages };
        f.analyzer.ruleEvaluator.applyFiringRecords(f.memory, parts, 1, false);
        const std::set<std::string> deposited = internalStatements(f.memory);
        ASSERT_EQ(deposited.count("(rem[b,a])"), std::size_t{1});
        ASSERT_EQ(deposited.count("(rem[a,g])"), std::size_t{strong == 1 ? 0u : 1u});
    }
}
