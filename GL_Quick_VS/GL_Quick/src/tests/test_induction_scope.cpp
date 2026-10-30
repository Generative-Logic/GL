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
/// @brief Tests for the induction scopes (D-519).
///
/// @details
/// Covers the scope syntax (payload, chain match, strong chain, the
/// validity-forest walks), the scope sink and the strong step demand's
/// once-per-path rule with its hit report. The admission, publication, goal
/// and chapter paths are covered in `test_memory.cpp` beside the other
/// induction admission tests.

#include "../memory.hpp"
#include <cstring>
#include <string>
#include "test_harness.hpp"
#include "prover_fixture.hpp"
#include "canonical_test_helpers.hpp"

#include "../prover.hpp"

#include <map>
#include <set>
#include <tuple>

namespace {

    /// @brief Span over a NUL-terminated literal.
    ///
    /// @param text The literal.
    /// @return The span (no terminator).
    gl::StrSpan span(const char* text) {
        return gl::StrSpan(text, static_cast<int32_t>(std::strlen(text)));
    }

    /// @brief Copy of a scratch string's bytes.
    ///
    /// @param s The scratch string.
    /// @return Its text.
    std::string text(const gl::ScratchString& s) {
        const gl::StrSpan view(s);
        return std::string(view.ptr, static_cast<std::size_t>(view.len));
    }

    /// @brief The test's string-tier scratch arena (the main-thread slot).
    ///
    /// @return The arena.
    gl::ScratchArena& stringArena() {
        return gl::scratchArenas().forSlot(gl::scratchArenas().slotCount() - 1);
    }

} // namespace

// The payload is `is_`, the chain, then each admitted name; it names exactly
// its own chain (a strict prefix does not match), and no other scope kind is
// an induction scope — `ordis_` ends in `is_`.
TEST(induction_scope, payload_names_its_chain_and_admitted_names) {
    gl::ScratchArena& arena = stringArena();
    gl::ScratchScope frame(arena);
    const gl::StrSpan chain[2] = { span("(in3[2,3,marker,4])"), span("(in3[marker,5,1,4])") };
    const gl::StrSpan names[1] = { span("it_0_lev_2_7") };
    const gl::ScratchString payload = gl::ScopeSyntax::inductionScopePayloadScratch(arena, chain, 2, names, 1);
    ASSERT_TRUE(text(payload) == "is_(in3[2,3,marker,4])(in3[marker,5,1,4])_it_0_lev_2_7");
    ASSERT_TRUE(gl::ScopeSyntax::isInductionScopePayload(gl::StrSpan(payload)));
    ASSERT_TRUE(gl::ScopeSyntax::inductionPayloadNamesChain(gl::StrSpan(payload), chain, 2));
    ASSERT_FALSE(gl::ScopeSyntax::inductionPayloadNamesChain(gl::StrSpan(payload), chain, 1));

    const gl::StrSpan strong[1] = { span("(in3[marker1,marker2,u_10,u_4])!(=[marker1,u_10])") };
    const gl::StrSpan pair[2] = { span("int_0_lev_5_3"), span("int_0_lev_5_4") };
    ASSERT_TRUE(text(gl::ScopeSyntax::inductionScopePayloadScratch(arena, strong, 1, pair, 2))
        == "is_(in3[marker1,marker2,u_10,u_4])!(=[marker1,u_10])_int_0_lev_5_3_int_0_lev_5_4");

    ASSERT_FALSE(gl::ScopeSyntax::isInductionScopePayload(span("ordis_(or9[a,6,9])_((=[a,9]))")));
    ASSERT_FALSE(gl::ScopeSyntax::isInductionScopePayload(span("integration_int_0_lev_6_119")));
    ASSERT_FALSE(gl::ScopeSyntax::isInductionScopePayload(span("is_")));
}

// The names after the chain: the rest of the payload for one name, the one
// boundary with a name on both sides for two (a witness, or letters and
// digits with `_copy` suffixes).
TEST(induction_scope, scope_names_follow_the_chain) {
    gl::StrSpan names[2];
    gl::ScopeSyntax::inductionScopeNames(span("is_(in3[2,3,marker,4])_it_0_lev_2_7"), 1, names);
    ASSERT_TRUE(gl::equalSpans(names[0], span("it_0_lev_2_7")));
    gl::ScopeSyntax::inductionScopeNames(span("is_(in3[2,3,marker,4])(in3[marker,5,1,4])_9"), 1, names);
    ASSERT_TRUE(gl::equalSpans(names[0], span("9")));
    const char* strong = "is_(in3[marker1,marker2,u_10,u_4])!(=[marker1,u_10])";
    for (const auto& c : { std::pair<const char*, const char*>{"int_0_lev_5_3", "int_0_lev_5_4"},
            {"10", "int_0_lev_5_2"}, {"10_copy", "int_0_lev_5_2"}, {"int_0_lev_5_28", "2"},
            {"it_0_lev_1_3", "force_lev_1_0"} }) {
        const std::string payload = std::string(strong) + "_" + c.first + "_" + c.second;
        gl::ScopeSyntax::inductionScopeNames(gl::StrSpan(payload), 2, names);
        ASSERT_TRUE(gl::equalSpans(names[0], span(c.first)));
        ASSERT_TRUE(gl::equalSpans(names[1], span(c.second)));
    }
}

// The strong chain writes the bound premise's two open arguments as marker1
// and marker2 in every premise and keeps the fixed arguments' spelling.
TEST(induction_scope, strong_chain_marks_the_bound_value_and_its_distance) {
    gl::ScratchArena& arena = stringArena();
    gl::ScratchScope frame(arena);
    const gl::StrSpan chain[3] = { span("(in3[11,12,u_10,u_4])"), span("!(=[11,u_10])"),
        span("(coprime[u_1,u_2,u_3,u_5,9,11])") };
    ASSERT_TRUE(text(gl::ScopeSyntax::inductionDoubleChainScratch(arena, chain, 3))
        == "(in3[marker1,marker2,u_10,u_4])!(=[marker1,u_10])(coprime[u_1,u_2,u_3,u_5,9,marker1])");
}

// The root drops every trailing induction segment and stops at the first other
// scope; the path check sees the scope itself and its ancestors, never a
// sibling or a descendant.
TEST(induction_scope, roots_and_paths_follow_the_validity_forest) {
    gl::Memory m;
    gl::NameMap& nm = m.nameMap;
    const gl::NameId root = nm.encode(std::string("main"));
    const gl::NameId branch = nm.encodePush(root, std::string("ordis_(or9[a,6,9])_((=[a,9]))"));
    const gl::NameId first = nm.encodePush(branch, std::string("is_(in3[2,3,marker,4])_it_0_lev_2_7"));
    const gl::NameId second = nm.encodePush(first,
        std::string("is_(in3[it_0_lev_2_7,5,marker,4])_it_0_lev_2_8"));
    const gl::NameId sibling = nm.encodePush(branch, std::string("is_(in3[2,5,marker,4])_it_0_lev_2_9"));

    ASSERT_FALSE(gl::ScopeSyntax::isInductionScope(nm, root));
    ASSERT_FALSE(gl::ScopeSyntax::isInductionScope(nm, branch));
    ASSERT_TRUE(gl::ScopeSyntax::isInductionScope(nm, first));
    ASSERT_TRUE(gl::ScopeSyntax::isInductionScope(nm, second));
    ASSERT_EQ(gl::ScopeSyntax::inductionRootOf(nm, second), branch);
    ASSERT_EQ(gl::ScopeSyntax::inductionRootOf(nm, first), branch);
    ASSERT_EQ(gl::ScopeSyntax::inductionRootOf(nm, branch), branch);
    ASSERT_TRUE(gl::ScopeSyntax::atOrBelowInductionScope(nm, second));
    ASSERT_FALSE(gl::ScopeSyntax::atOrBelowInductionScope(nm, branch));
    ASSERT_FALSE(gl::ScopeSyntax::atOrBelowInductionScope(nm, root));

    const gl::StrSpan used[1] = { span("(in3[2,3,marker,4])") };
    const gl::StrSpan next[1] = { span("(in3[it_0_lev_2_7,5,marker,4])") };
    const gl::StrSpan parallel[1] = { span("(in3[2,5,marker,4])") };
    ASSERT_TRUE(gl::ScopeSyntax::inductionChainOnPath(nm, second, used, 1));
    ASSERT_TRUE(gl::ScopeSyntax::inductionChainOnPath(nm, second, next, 1));
    ASSERT_FALSE(gl::ScopeSyntax::inductionChainOnPath(nm, first, next, 1));
    ASSERT_FALSE(gl::ScopeSyntax::inductionChainOnPath(nm, second, parallel, 1));
    ASSERT_TRUE(gl::ScopeSyntax::inductionChainOnPath(nm, sibling, parallel, 1));
    ASSERT_FALSE(gl::ScopeSyntax::inductionChainOnPath(nm, sibling, used, 1));
    ASSERT_FALSE(gl::ScopeSyntax::inductionChainOnPath(nm, branch, used, 1));
}

// The sink lists each opened scope once, in the order of its first opening.
TEST(induction_scope, sink_lists_each_scope_once) {
    gl::InductionScopeSink sink;
    sink.add(gl::NameId(5));
    sink.add(gl::NameId(7));
    sink.add(gl::NameId(5));
    ASSERT_EQ(sink.count, 2);
    ASSERT_EQ(sink.scopes[0], gl::NameId(5));
    ASSERT_EQ(sink.scopes[1], gl::NameId(7));
}

// Route 5 reports the key and the marker positions it admitted through. The
// strong step's demand admits once per path: inside an induction scope its
// chain opened it refuses a second smaller value, while a goal's double key
// still admits there and an induction scope of another chain refuses nothing.
TEST(induction_scope, strong_demand_admits_once_per_path) {
    gl::tests::ProverFixture ana(std::string("FTA"));
    gl::Memory& mb = ana.body;
    const gl::StrSpan mainScope("main", 4);
    ASSERT_TRUE(ana.admissionMemory.updateAdmissionMapIntegrationDouble(mb,
        span("(in3[11,12,u_10,u_4])"), mainScope, gl::InductionDemand::sum));
    ASSERT_TRUE(ana.admissionMemory.updateAdmissionMapIntegrationDouble(mb,
        span("(bezEq[u_1,u_4,u_5,u_6,u_12,pi_lev_3_12,u_9,pi_lev_3_13])"), mainScope));
    gl::ScratchArena& arena = stringArena();
    gl::ScratchScope frame(arena);
    const gl::StrSpan chain[2] = { span("(in3[11,12,u_10,u_4])"), span("!(=[11,u_10])") };
    const gl::ScratchString marked = gl::ScopeSyntax::inductionDoubleChainScratch(arena, chain, 2);
    mb.inductionDoubleChainId = mb.nameMap.encode(gl::StrSpan(marked));

    const gl::StrSpan step = span("(in3[r,marker,10,4])");
    gl::DoubleMarkerHit hit;
    ASSERT_TRUE(ana.admissionMemory.isAdmittedIntegrationDouble(mb, step, mainScope, true, &hit));
    ASSERT_EQ(hit.key, mb.inductionDoubleKey);
    ASSERT_EQ(hit.markerAt, 1);
    ASSERT_EQ(hit.otherAt, 0);

    const gl::StrSpan markedChain(marked);
    const gl::StrSpan names[2] = { span("r"), span("int_0_lev_2_5") };
    const gl::ScratchString payload = gl::ScopeSyntax::inductionScopePayloadScratch(
        arena, &markedChain, 1, names, 2);
    const gl::NameId opened = mb.nameMap.encodePush(gl::NameMap::MAIN_ID, gl::StrSpan(payload));
    const std::string openedName = mb.nameMap.decode(opened);
    ASSERT_FALSE(ana.admissionMemory.isAdmittedIntegrationDouble(mb, step, gl::StrSpan(openedName)));
    const gl::StrSpan caseHead = span("(bezEq[1,4,5,6,12,force_lev_3_1,9,marker])");
    gl::DoubleMarkerHit goalHit;
    ASSERT_TRUE(ana.admissionMemory.isAdmittedIntegrationDouble(mb, caseHead,
        gl::StrSpan(openedName), true, &goalHit));
    ASSERT_TRUE(goalHit.key != mb.inductionDoubleKey);

    const gl::NameId other = mb.nameMap.encodePush(gl::NameMap::MAIN_ID,
        std::string("is_(in3[2,3,marker,4])_it_0_lev_2_7"));
    const std::string otherName = mb.nameMap.decode(other);
    ASSERT_TRUE(ana.admissionMemory.isAdmittedIntegrationDouble(mb, step, gl::StrSpan(otherName)));
}

// The eating stands at the deepest induction scope on a scope's path, else at
// `main`; an OR branch below an induction scope eats where that scope eats,
// a branch above every induction scope where `main` eats.
TEST(induction_scope, eating_position_is_the_deepest_induction_scope_on_the_path) {
    gl::Memory m;
    gl::NameMap& nm = m.nameMap;
    const gl::NameId root = nm.encode(std::string("main"));
    const gl::NameId branch = nm.encodePush(root, std::string("ordis_(or9[a,6,9])_((=[a,9]))"));
    const gl::NameId first = nm.encodePush(root, std::string("is_(in3[2,3,marker,4])_it_0_lev_2_7"));
    const gl::NameId inner = nm.encodePush(first, std::string("ordis_(or9[b,6,9])_((=[b,9]))"));
    const gl::NameId second = nm.encodePush(inner,
        std::string("is_(in3[it_0_lev_2_7,5,marker,4])_it_0_lev_2_8"));
    ASSERT_EQ(gl::ScopeSyntax::inductionEatingPosition(nm, root), gl::NameMap::MAIN_ID);
    ASSERT_EQ(gl::ScopeSyntax::inductionEatingPosition(nm, branch), gl::NameMap::MAIN_ID);
    ASSERT_EQ(gl::ScopeSyntax::inductionEatingPosition(nm, first), first);
    ASSERT_EQ(gl::ScopeSyntax::inductionEatingPosition(nm, inner), first);
    ASSERT_EQ(gl::ScopeSyntax::inductionEatingPosition(nm, second), second);
}

// A statement in an induction scope leaves the tree when it is an open goal
// at the tree's root or above, to the root; a statement that is no goal, or
// a goal outside every induction scope, does not leave.
TEST(induction_scope, goal_exit_names_the_root_of_a_goal_hit) {
    gl::Memory m;
    gl::NameMap& nm = m.nameMap;
    const gl::NameId root = nm.encode(std::string("main"));
    const gl::NameId branch = nm.encodePush(root, std::string("ordis_(or9[a,6,9])_((=[a,9]))"));
    const gl::NameId first = nm.encodePush(branch, std::string("is_(in3[2,3,marker,4])_it_0_lev_2_7"));
    const gl::NameId leaf = nm.encodePush(first,
        std::string("is_(in3[it_0_lev_2_7,5,marker,4])_it_0_lev_2_8"));
    const gl::NameId goal = nm.encode(std::string("(=[a,X])"));
    const gl::NameId other = nm.encode(std::string("(=[b,X])"));
    m.intToBeProved.assignSet(gl::packStatementKey(goal, root), nullptr, 0);

    gl::NameId exit = 0;
    ASSERT_TRUE(gl::ScopeSyntax::inductionGoalExit(m, goal, leaf, exit));
    ASSERT_EQ(exit, branch);
    gl::NameId untouched = 0;
    ASSERT_FALSE(gl::ScopeSyntax::inductionGoalExit(m, other, leaf, untouched));
    ASSERT_FALSE(gl::ScopeSyntax::inductionGoalExit(m, goal, branch, untouched));
    ASSERT_EQ(untouched, gl::NameId(0));
}

// One order per path: the first recorded eat at a scope fixes its slot; the
// same slot stays open there, every other slot (the strong demand included)
// closes, and other scopes are unaffected.
TEST(induction_scope, eats_fix_one_slot_per_scope) {
    gl::Memory m;
    const gl::NameId root = m.nameMap.encode(std::string("main"));
    const gl::NameId child = m.nameMap.encodePush(root, std::string("child"));
    ASSERT_TRUE(m.inductionEatenSlot(child) == nullptr);
    ASSERT_TRUE(m.inductionSlotOpen(child, 5));
    ASSERT_TRUE(m.inductionSlotOpen(child, gl::Memory::kStrongInductionSlot));
    m.recordInductionEat(child, 5);
    ASSERT_EQ(*m.inductionEatenSlot(child), 5);
    ASSERT_TRUE(m.inductionSlotOpen(child, 5));
    ASSERT_FALSE(m.inductionSlotOpen(child, 6));
    ASSERT_FALSE(m.inductionSlotOpen(child, gl::Memory::kStrongInductionSlot));
    m.recordInductionEat(child, 5);
    ASSERT_EQ(m.lbMemory.inductionEats.count(), 1);
    ASSERT_TRUE(m.inductionSlotOpen(root, 6));
    m.recordInductionEat(root, gl::Memory::kStrongInductionSlot);
    ASSERT_FALSE(m.inductionSlotOpen(root, 6));
    ASSERT_EQ(m.lbMemory.inductionEats.count(), 2);
}

// The first firing in a leaf settles its tree (D-520):
// every induction scope whose parent lies on the leaf's path - the siblings
// along the path and the leaf's own children - joins the wipe drain, the
// leaf's path and other trees stay, and a later firing in the settled tree
// changes nothing.
TEST(induction_scope, first_firing_settles_its_tree_and_wipes_the_rest) {
    gl::Memory m;
    gl::NameMap& nm = m.nameMap;
    const gl::NameId root = nm.encode(std::string("main"));
    const gl::NameId first = nm.encodePush(root, std::string("is_(in3[2,3,marker,4])_it_0_lev_2_7"));
    const gl::NameId firstSibling = nm.encodePush(root, std::string("is_(in3[2,3,marker,4])_it_0_lev_2_9"));
    const gl::NameId leaf = nm.encodePush(first, std::string("is_(in3[it_0_lev_2_7,5,marker,4])_it_0_lev_2_8"));
    const gl::NameId leafSibling = nm.encodePush(first, std::string("is_(in3[it_0_lev_2_7,5,marker,4])_it_0_lev_2_10"));
    const gl::NameId leafChild = nm.encodePush(leaf, std::string("is_(in3[it_0_lev_2_8,6,marker,4])_it_0_lev_2_11"));
    const gl::NameId belowSibling = nm.encodePush(firstSibling, std::string("is_(in3[it_0_lev_2_9,5,marker,4])_it_0_lev_2_12"));
    const gl::NameId branch = nm.encodePush(root, std::string("ordis_(or9[a,6,9])_((=[a,9]))"));
    const gl::NameId otherTree = nm.encodePush(branch, std::string("is_(in3[2,3,marker,4])_it_0_lev_2_13"));

    m.recordInductionFiring(leaf);
    ASSERT_FALSE(m.inductionTreeSettled(leaf));
    m.settleInductionTrees();
    ASSERT_TRUE(m.lbMemory.inductionFirings.empty());
    ASSERT_TRUE(m.inductionTreeSettled(root));
    ASSERT_TRUE(m.inductionTreeSettled(leaf));
    ASSERT_TRUE(m.inductionTreeSettled(firstSibling));
    ASSERT_FALSE(m.inductionTreeSettled(branch));
    ASSERT_FALSE(m.inductionTreeSettled(otherTree));
    ASSERT_EQ(m.pendingWipeScopes.count(), 3);
    ASSERT_TRUE(m.pendingWipeScopes.lookup(firstSibling) != 0);
    ASSERT_TRUE(m.pendingWipeScopes.lookup(leafSibling) != 0);
    ASSERT_TRUE(m.pendingWipeScopes.lookup(leafChild) != 0);
    ASSERT_TRUE(m.pendingWipeScopes.lookup(belowSibling) == 0);
    ASSERT_TRUE(m.pendingWipeScopes.lookup(first) == 0);
    ASSERT_TRUE(m.pendingWipeScopes.lookup(leaf) == 0);

    m.recordInductionFiring(first);
    m.settleInductionTrees();
    ASSERT_EQ(m.pendingWipeScopes.count(), 3);
    ASSERT_EQ(m.lbMemory.inductionSettledRoots.count(), 1);
}

// Two firings in one tree in the same burst: the first by decoded scope name
// settles it and the other leaf is wiped; a firing in another tree settles
// that tree on its own (D-520).
TEST(induction_scope, same_burst_firings_settle_by_the_first_decoded_name) {
    gl::Memory m;
    gl::NameMap& nm = m.nameMap;
    const gl::NameId root = nm.encode(std::string("main"));
    const gl::NameId earlier = nm.encodePush(root, std::string("is_(in3[2,3,marker,4])_it_0_lev_2_7"));
    const gl::NameId later = nm.encodePush(root, std::string("is_(in3[2,3,marker,4])_it_0_lev_2_9"));
    const gl::NameId branch = nm.encodePush(root, std::string("ordis_(or9[a,6,9])_((=[a,9]))"));
    const gl::NameId branchLeaf = nm.encodePush(branch, std::string("is_(in3[2,3,marker,4])_it_0_lev_2_13"));
    const gl::NameId branchSibling = nm.encodePush(branch, std::string("is_(in3[2,3,marker,4])_it_0_lev_2_14"));

    m.recordInductionFiring(later);
    m.recordInductionFiring(branchLeaf);
    m.recordInductionFiring(earlier);
    m.settleInductionTrees();
    ASSERT_TRUE(m.inductionTreeSettled(root));
    ASSERT_TRUE(m.inductionTreeSettled(branch));
    ASSERT_EQ(m.pendingWipeScopes.count(), 2);
    ASSERT_TRUE(m.pendingWipeScopes.lookup(later) != 0);
    ASSERT_TRUE(m.pendingWipeScopes.lookup(branchSibling) != 0);
    ASSERT_TRUE(m.pendingWipeScopes.lookup(earlier) == 0);
    ASSERT_TRUE(m.pendingWipeScopes.lookup(branchLeaf) == 0);
}

// A firing whose path is already being wiped this burst settles nothing: the
// leaf goes with its path (D-520).
TEST(induction_scope, firing_on_a_path_being_wiped_settles_nothing) {
    gl::Memory m;
    gl::NameMap& nm = m.nameMap;
    const gl::NameId root = nm.encode(std::string("main"));
    const gl::NameId branch = nm.encodePush(root, std::string("ordis_(or9[a,6,9])_((=[a,9]))"));
    const gl::NameId leaf = nm.encodePush(branch, std::string("is_(in3[2,3,marker,4])_it_0_lev_2_7"));
    const gl::NameId sibling = nm.encodePush(branch, std::string("is_(in3[2,3,marker,4])_it_0_lev_2_9"));
    m.pendingWipeScopes.mint(branch);
    m.recordInductionFiring(leaf);
    m.settleInductionTrees();
    ASSERT_TRUE(m.lbMemory.inductionFirings.empty());
    ASSERT_FALSE(m.inductionTreeSettled(leaf));
    ASSERT_EQ(m.pendingWipeScopes.count(), 1);
    ASSERT_TRUE(m.pendingWipeScopes.lookup(sibling) == 0);
}

// A settled tree refuses the strong step's demand, while a goal's double key
// still admits there (D-520).
TEST(induction_scope, settled_tree_refuses_the_strong_demand) {
    gl::tests::ProverFixture ana(std::string("FTA"));
    gl::Memory& mb = ana.body;
    const gl::StrSpan mainScope("main", 4);
    ASSERT_TRUE(ana.admissionMemory.updateAdmissionMapIntegrationDouble(mb,
        span("(in3[11,12,u_10,u_4])"), mainScope, gl::InductionDemand::sum));
    ASSERT_TRUE(ana.admissionMemory.updateAdmissionMapIntegrationDouble(mb,
        span("(bezEq[u_1,u_4,u_5,u_6,u_12,pi_lev_3_12,u_9,pi_lev_3_13])"), mainScope));
    gl::ScratchArena& arena = stringArena();
    gl::ScratchScope frame(arena);
    const gl::StrSpan chain[2] = { span("(in3[11,12,u_10,u_4])"), span("!(=[11,u_10])") };
    const gl::ScratchString marked = gl::ScopeSyntax::inductionDoubleChainScratch(arena, chain, 2);
    mb.inductionDoubleChainId = mb.nameMap.encode(gl::StrSpan(marked));

    const gl::StrSpan step = span("(in3[r,marker,10,4])");
    ASSERT_TRUE(ana.admissionMemory.isAdmittedIntegrationDouble(mb, step, mainScope));
    mb.lbMemory.inductionSettledRoots.mint(gl::NameMap::MAIN_ID);
    ASSERT_FALSE(ana.admissionMemory.isAdmittedIntegrationDouble(mb, step, mainScope));
    ASSERT_TRUE(ana.admissionMemory.isAdmittedIntegrationDouble(mb,
        span("(bezEq[1,4,5,6,12,force_lev_3_1,9,marker])"), mainScope));
}

// A settled root and a pending firing leave with their scope
// (D-520); other settled roots stay.
TEST(induction_scope, settlement_state_leaves_with_its_scope) {
    gl::tests::ProverFixture ana(std::string("Peano"));
    gl::Memory& mb = ana.body;
    const gl::NameId root = mb.nameMap.encode(std::string("main"));
    const gl::NameId branch = mb.nameMap.encodePush(root, std::string("ordis_(or9[a,6,9])_((=[a,9]))"));
    const gl::NameId leaf = mb.nameMap.encodePush(branch, std::string("is_(in3[2,3,marker,4])_it_0_lev_2_7"));
    mb.lbMemory.inductionSettledRoots.mint(branch);
    mb.lbMemory.inductionSettledRoots.mint(root);
    mb.recordInductionFiring(leaf);
    mb.wipeSubtree(branch);
    ASSERT_TRUE(mb.lbMemory.inductionFirings.empty());
    ASSERT_EQ(mb.lbMemory.inductionSettledRoots.count(), 1);
    ASSERT_TRUE(mb.lbMemory.inductionSettledRoots.lookup(root) != 0);
}

namespace {

    /// @brief One class at @p scope, with its canonical view seeded.
    ///
    /// @param memory The block.
    /// @param scope  The scope the class lives at.
    /// @param members The class members' names.
    void aliasClass(gl::Memory& memory, gl::NameId scope, std::initializer_list<std::string> members) {
        gl::EquivalenceClass equality;
        equality.setMembersFromNames(members, memory.nameMap);
        memory.assignClassesById(scope, {equality});
        seedFixtureCanonicalClasses(memory);
    }

} // namespace

// An open induction scope whose admitted name shares a class with a number or
// an ancestor block's witness joins the wipe drain; a name equal only to this
// block's own witnesses keeps its scope, and so does an in-place scope, a
// one-name scope named by a root name
// (D-545, D-546).
TEST(induction_scope, alias_scope_joins_the_wipe_drain) {
    gl::tests::ProverFixture analyzer("Peano");
    analyzer.parameters.trackHistory = false;
    gl::Memory m;
    m.level = 2;
    gl::NameMap& nm = m.nameMap;
    const gl::NameId root = nm.encode(std::string("main"));
    const gl::NameId number = nm.encodePush(root, std::string("is_(in3[2,3,marker,4])_it_0_lev_2_7"));
    const gl::NameId ancestor = nm.encodePush(root, std::string("is_(in3[2,3,marker,4])_it_0_lev_2_9"));
    const gl::NameId fresh = nm.encodePush(root, std::string("is_(in3[2,3,marker,4])_it_0_lev_2_11"));
    const gl::NameId plain = nm.encodePush(root, std::string("is_(in3[2,3,marker,4])_9"));
    m.recordInductionEat(root, 1);
    aliasClass(m, number, {"it_0_lev_2_7", "9"});
    aliasClass(m, ancestor, {"it_0_lev_2_9", "it_0_lev_1_3"});
    aliasClass(m, fresh, {"it_0_lev_2_11", "it_0_lev_2_12"});
    analyzer.implicationSubproofs.retireAliasInductionScopes(m);
    ASSERT_TRUE(m.pendingWipeScopes.lookup(number) != 0);
    ASSERT_TRUE(m.pendingWipeScopes.lookup(ancestor) != 0);
    ASSERT_TRUE(m.pendingWipeScopes.lookup(plain) == 0);
    ASSERT_TRUE(m.pendingWipeScopes.lookup(fresh) == 0);
    ASSERT_EQ(m.pendingWipeScopes.count(), 2);
}

// A scope of the strong demand carries two names; it is retired only when
// both equal root names, also when both are root names themselves: it is a
// witness scope, no in-place one. A settled tree and a block whose
// hypothesis never ate keep their scopes
// (D-545, D-546).
TEST(induction_scope, strong_alias_scope_needs_both_names_and_an_open_tree) {
    for (int variant = 0; variant < 3; ++variant) {
        gl::tests::ProverFixture analyzer("Peano");
        analyzer.parameters.trackHistory = false;
        gl::Memory m;
        m.level = 2;
        gl::NameMap& nm = m.nameMap;
        const gl::NameId root = nm.encode(std::string("main"));
        const std::string chain = "(in3[marker1,marker2,u_10,u_4])!(=[marker1,u_10])";
        m.inductionDoubleChainId = nm.encode(gl::StrSpan(chain));
        const gl::NameId both = nm.encodePush(root, "is_" + chain + "_10_int_0_lev_2_2");
        const gl::NameId half = nm.encodePush(root, "is_" + chain + "_int_0_lev_2_3_int_0_lev_2_4");
        const gl::NameId roots = nm.encodePush(root, "is_" + chain + "_10_2");
        if (variant != 2) m.recordInductionEat(root, gl::Memory::kStrongInductionSlot);
        if (variant == 1) m.lbMemory.inductionSettledRoots.mint(root);
        aliasClass(m, both, {"int_0_lev_2_2", "2"});
        aliasClass(m, half, {"int_0_lev_2_3", "2"});
        analyzer.implicationSubproofs.retireAliasInductionScopes(m);
        ASSERT_EQ(m.pendingWipeScopes.lookup(both) != 0, variant == 0);
        ASSERT_TRUE(m.pendingWipeScopes.lookup(half) == 0);
        ASSERT_EQ(m.pendingWipeScopes.lookup(roots) != 0, variant == 0);
    }
}

// The strong step's second demand, the multiplication catcher's bound premise
// x*m = n, admits m as the second factor and reports its own key. One strong
// catch per path: inside a scope that either demand's chain opened, both
// demands are refused, and the lemma-input gate closes both
// (D-566,
// D-577).
TEST(induction_scope, strong_demands_admit_one_catch_per_path) {
    gl::tests::ProverFixture ana(std::string("FTA"));
    gl::Memory& mb = ana.body;
    const gl::StrSpan mainScope("main", 4);
    ASSERT_TRUE(ana.admissionMemory.updateAdmissionMapIntegrationDouble(mb,
        span("(in3[11,12,u_10,u_4])"), mainScope, gl::InductionDemand::sum));
    ASSERT_TRUE(ana.admissionMemory.updateAdmissionMapIntegrationDouble(mb,
        span("(in3[12,11,u_10,u_5])"), mainScope, gl::InductionDemand::product));
    ASSERT_TRUE(mb.inductionDoubleKey != 0 && mb.inductionProductKey != 0
        && mb.inductionDoubleKey != mb.inductionProductKey);
    gl::ScratchArena& arena = stringArena();
    gl::ScratchScope frame(arena);
    const gl::StrSpan sumChain[2] = { span("(in3[11,12,u_10,u_4])"), span("!(=[11,u_10])") };
    const gl::StrSpan productChain[2] = { span("(in3[12,11,u_10,u_5])"), span("!(=[11,u_10])") };
    const gl::ScratchString sumMarked = gl::ScopeSyntax::inductionDoubleChainScratch(arena, sumChain, 2);
    const gl::ScratchString productMarked = gl::ScopeSyntax::inductionDoubleChainScratch(arena, productChain, 2);
    ASSERT_TRUE(text(productMarked) == "(in3[marker1,marker2,u_10,u_5])!(=[marker2,u_10])");
    mb.inductionDoubleChainId = mb.nameMap.encode(gl::StrSpan(sumMarked));
    mb.inductionProductChainId = mb.nameMap.encode(gl::StrSpan(productMarked));

    const gl::StrSpan factor = span("(in3[p,marker,10,5])");
    gl::DoubleMarkerHit hit;
    ASSERT_TRUE(ana.admissionMemory.isAdmittedIntegrationDouble(mb, factor, mainScope, true, &hit));
    ASSERT_EQ(hit.key, mb.inductionProductKey);
    ASSERT_EQ(hit.markerAt, 1);
    ASSERT_EQ(hit.otherAt, 0);
    ASSERT_FALSE(ana.admissionMemory.isAdmittedIntegrationDouble(mb, factor, mainScope, false));

    const gl::StrSpan names[2] = { span("p"), span("int_0_lev_2_5") };
    const gl::StrSpan productSpan(productMarked);
    const gl::NameId productScope = mb.nameMap.encodePush(gl::NameMap::MAIN_ID,
        gl::StrSpan(gl::ScopeSyntax::inductionScopePayloadScratch(arena, &productSpan, 1, names, 2)));
    const std::string productName = mb.nameMap.decode(productScope);
    const gl::StrSpan sumSpan(sumMarked);
    const gl::NameId sumScope = mb.nameMap.encodePush(gl::NameMap::MAIN_ID,
        gl::StrSpan(gl::ScopeSyntax::inductionScopePayloadScratch(arena, &sumSpan, 1, names, 2)));
    const std::string sumName = mb.nameMap.decode(sumScope);
    ASSERT_FALSE(ana.admissionMemory.isAdmittedIntegrationDouble(mb, factor, gl::StrSpan(productName)));
    ASSERT_FALSE(ana.admissionMemory.isAdmittedIntegrationDouble(mb, factor, gl::StrSpan(sumName)));
    const gl::StrSpan summand = span("(in3[r,marker,10,4])");
    ASSERT_FALSE(ana.admissionMemory.isAdmittedIntegrationDouble(mb, summand, gl::StrSpan(productName)));
    ASSERT_FALSE(ana.admissionMemory.isAdmittedIntegrationDouble(mb, summand, gl::StrSpan(sumName)));
    // Below main outside any induction scope both still admit.
    gl::DoubleMarkerHit atMain;
    ASSERT_TRUE(ana.admissionMemory.isAdmittedIntegrationDouble(mb, summand, mainScope, true, &atMain));
    ASSERT_EQ(atMain.key, mb.inductionDoubleKey);
}

// A block's induction implications are its first hypothesis and, in a strong
// step block, the multiplication catcher; a firing of either marks its head
// as the hypothesis head. No id names a rule in other blocks
// (D-566).
TEST(induction_scope, both_strong_hypotheses_are_induction_implications) {
    gl::Memory m;
    ASSERT_FALSE(m.isInductionHypothesisRule(0));
    ASSERT_FALSE(m.isInductionHypothesisRule(7));
    m.recursionAuxyImplId = 7;
    ASSERT_TRUE(m.isInductionHypothesisRule(7));
    ASSERT_FALSE(m.isInductionHypothesisRule(9));
    m.recursionProductImplId = 9;
    ASSERT_TRUE(m.isInductionHypothesisRule(9));
    ASSERT_TRUE(m.isInductionHypothesisRule(7));
    ASSERT_FALSE(m.isInductionHypothesisRule(8));
    ASSERT_FALSE(m.isInductionHypothesisRule(0));
}

// One unfolding of I1's existence (the divisors D of n with a least element
// p) offers both strong demands a witness: x with p + x = n through the head
// p <= n, d with p * d = n through the head p | n. Each catch opens its own
// scope, named by its demand's chain and its two names; each scope gets its
// own copy of the path D, p and its own catch, never the other one, and main
// gets none of them (D-566).
TEST(induction_scope, two_catches_of_one_unfolding_get_separate_scopes) {
    gl::tests::ProverFixture ea("FTA");
    const std::string compiled =
        ea.structuralCompiler.compileExpression("!(>[D](divisorSet[N,a,m,t,n,D])!(hasMinimum[N,a,D]))");
    ea.supportingRuleCompiler.compileDisintegrationPackages();
    const std::map<std::string, std::string> values{
        {"N", "1"}, {"a", "4"}, {"m", "5"}, {"t", "7"}, {"n", "9"}};
    for (const std::string& argument : ce::getArgs(compiled)) ASSERT_EQ(values.count(argument), std::size_t(1));
    const std::string source = ce::replaceKeysInString(compiled, values);
    ea.parameters.trackHistory = true;
    gl::Memory memory;
    memory.level = 2;
    memory.nameMap.encode(gl::StrSpan("main", 4));
    const gl::StrSpan mainScope("main", 4);
    ASSERT_TRUE(ea.admissionMemory.updateAdmissionMapIntegrationDouble(memory,
        span("(in3[11,12,u_9,u_4])"), mainScope, gl::InductionDemand::sum));
    ASSERT_TRUE(ea.admissionMemory.updateAdmissionMapIntegrationDouble(memory,
        span("(in3[12,11,u_9,u_5])"), mainScope, gl::InductionDemand::product));
    // The lemma's induction variable n, recorded with the demands as the grid
    // builder records it (D-601).
    memory.inductionVariableDigits[0] = 9;
    memory.inductionVariableDigitCount = 1;
    {
        gl::ScratchArena& strings = stringArena();
        gl::ScratchScope frame(strings);
        const gl::StrSpan sumChain[2] = { span("(in3[11,12,u_9,u_4])"), span("!(=[11,u_9])") };
        const gl::StrSpan productChain[2] = { span("(in3[12,11,u_9,u_5])"), span("!(=[11,u_9])") };
        memory.inductionDoubleChainId = memory.nameMap.encode(
            gl::StrSpan(gl::ScopeSyntax::inductionDoubleChainScratch(strings, sumChain, 2)));
        memory.inductionProductChainId = memory.nameMap.encode(
            gl::StrSpan(gl::ScopeSyntax::inductionDoubleChainScratch(strings, productChain, 2)));
    }
    gl::ScratchArena& arena = gl::genScratchArenas().forSlot(gl::genScratchArenas().slotCount() - 1);
    int levels[2] = {0, 2};
    gl::DisintProducts output(&arena);
    ea.disintegrationProcessor.disintegrateExpr2(gl::StrSpan(source), memory, 0, false,
        mainScope, output, true, false, levels, 2, true);
    std::map<std::string, std::set<std::string>> byScope;
    output.statements.forEachSorted([&](gl::StrSpan text, gl::StrSpan validity) {
        byScope[validity.toStdString()].insert(text.toStdString());
    });

    const std::string prefix = "main_boundary_is_";
    const std::string sumMarked = memory.nameMap.decode(memory.inductionDoubleChainId);
    const std::string productMarked = memory.nameMap.decode(memory.inductionProductChainId);
    std::string sumScope, productScope;
    for (const auto& entry : byScope) {
        if (entry.first.rfind(prefix + sumMarked + "_", 0) == 0) {
            ASSERT_TRUE(sumScope.empty());
            sumScope = entry.first;
        }
        if (entry.first.rfind(prefix + productMarked + "_", 0) == 0) {
            ASSERT_TRUE(productScope.empty());
            productScope = entry.first;
        }
    }
    ASSERT_FALSE(sumScope.empty());
    ASSERT_FALSE(productScope.empty());
    const std::string sumPayload = sumScope.substr(std::string("main_boundary_").size());
    const std::string productPayload = productScope.substr(std::string("main_boundary_").size());
    gl::StrSpan sumNames[2], productNames[2];
    gl::ScopeSyntax::inductionScopeNames(gl::StrSpan(sumPayload), 2, sumNames);
    gl::ScopeSyntax::inductionScopeNames(gl::StrSpan(productPayload), 2, productNames);
    const std::string p = productNames[0].toStdString();
    const std::string d = productNames[1].toStdString();
    const std::string x = sumNames[1].toStdString();
    ASSERT_EQ(sumNames[0].toStdString(), p);

    std::string minimum;
    for (const std::string& statement : byScope[productScope])
        if (statement.rfind("(minimum[1,4,", 0) == 0) minimum = statement;
    ASSERT_FALSE(minimum.empty());
    const std::vector<std::string> minimumArgs = ce::getArgs(minimum);
    const std::string D = minimumArgs[2];
    ASSERT_EQ(minimumArgs[3], p);
    for (const std::string& scope : { sumScope, productScope }) {
        ASSERT_EQ(byScope[scope].count("(divisorSet[1,4,5,7,9," + D + "])"), std::size_t(1));
        ASSERT_EQ(byScope[scope].count("(minimum[1,4," + D + "," + p + "])"), std::size_t(1));
        ASSERT_EQ(byScope[scope].count("(preorder[1,5," + p + ",9])"), std::size_t(1));
    }
    ASSERT_EQ(byScope[sumScope].count("(in3[" + p + "," + x + ",9,4])"), std::size_t(1));
    ASSERT_EQ(byScope[productScope].count("(in3[" + p + "," + d + ",9,5])"), std::size_t(1));
    ASSERT_EQ(byScope[sumScope].count("(in3[" + p + "," + d + ",9,5])"), std::size_t(0));
    ASSERT_EQ(byScope[productScope].count("(in3[" + p + "," + x + ",9,4])"), std::size_t(0));
    ASSERT_EQ(byScope["main"].count("(divisorSet[1,4,5,7,9," + D + "])"), std::size_t(0));
    ASSERT_EQ(byScope["main"].count("(minimum[1,4," + D + "," + p + "])"), std::size_t(0));
}

// In a strong step block both catches record their two names as size
// markers, the strong step's sizes; outside a strong step block a catch
// records none (D-584).
TEST(induction_scope, strong_catch_records_its_names_as_size_markers) {
    for (const bool strongStep : { true, false }) {
        gl::tests::ProverFixture ea("FTA");
        const std::string compiled =
            ea.structuralCompiler.compileExpression("!(>[D](divisorSet[N,a,m,t,n,D])!(hasMinimum[N,a,D]))");
        ea.supportingRuleCompiler.compileDisintegrationPackages();
        const std::map<std::string, std::string> values{
            {"N", "1"}, {"a", "4"}, {"m", "5"}, {"t", "7"}, {"n", "9"}};
        const std::string source = ce::replaceKeysInString(compiled, values);
        ea.parameters.trackHistory = true;
        gl::Memory memory;
        memory.level = 2;
        memory.nameMap.encode(gl::StrSpan("main", 4));
        memory.recursionHypothesisId = memory.nameMap.encode(gl::StrSpan("(in2[rec,9,3])"));
        memory.strongStepBlock = strongStep;
        if (strongStep) memory.inductionZeroId = memory.nameMap.encode(gl::StrSpan("2", 1));
        const gl::StrSpan mainScope("main", 4);
        ASSERT_TRUE(ea.admissionMemory.updateAdmissionMapIntegrationDouble(memory,
            span("(in3[11,12,u_9,u_4])"), mainScope, gl::InductionDemand::sum));
        ASSERT_TRUE(ea.admissionMemory.updateAdmissionMapIntegrationDouble(memory,
            span("(in3[12,11,u_9,u_5])"), mainScope, gl::InductionDemand::product));
        // The lemma's induction variable n, recorded with the demands as the
        // grid builder records it (D-601).
        memory.inductionVariableDigits[0] = 9;
        memory.inductionVariableDigitCount = 1;
        {
            gl::ScratchArena& strings = stringArena();
            gl::ScratchScope frame(strings);
            const gl::StrSpan sumChain[2] = { span("(in3[11,12,u_9,u_4])"), span("!(=[11,u_9])") };
            const gl::StrSpan productChain[2] = { span("(in3[12,11,u_9,u_5])"), span("!(=[11,u_9])") };
            memory.inductionDoubleChainId = memory.nameMap.encode(
                gl::StrSpan(gl::ScopeSyntax::inductionDoubleChainScratch(strings, sumChain, 2)));
            memory.inductionProductChainId = memory.nameMap.encode(
                gl::StrSpan(gl::ScopeSyntax::inductionDoubleChainScratch(strings, productChain, 2)));
        }
        gl::ScratchArena& arena = gl::genScratchArenas().forSlot(gl::genScratchArenas().slotCount() - 1);
        int levels[2] = {0, 2};
        gl::DisintProducts output(&arena);
        ea.disintegrationProcessor.disintegrateExpr2(gl::StrSpan(source), memory, 0, false,
            mainScope, output, true, false, levels, 2, true);
        std::set<std::string> caught;
        output.statements.forEachSorted([&](gl::StrSpan, gl::StrSpan validity) {
            const std::string scope = validity.toStdString();
            const std::string prefix = "main_boundary_is_";
            if (scope.rfind(prefix, 0) != 0) return;
            gl::StrSpan names[2];
            const std::string payload = scope.substr(std::string("main_boundary_").size());
            gl::ScopeSyntax::inductionScopeNames(gl::StrSpan(payload), 2, names);
            caught.insert(names[0].toStdString());
            caught.insert(names[1].toStdString());
        });
        ASSERT_EQ(caught.size(), std::size_t(3));
        for (const std::string& name : caught) {
            const int32_t value = memory.valueInterner.lookup(gl::StrSpan(name));
            ASSERT_EQ(value != 0 && memory.lbMemory.sizeMarkers.lookup(value) != 0, strongStep);
        }
        ASSERT_EQ(memory.lbMemory.sizeMarkers.count() >= 3, strongStep);
    }
}

// A chain splits at the parenthesis that closes each premise, a negation
// included and nested parentheses inside an argument kept whole
// (D-575).
TEST(induction_scope, chain_splits_into_its_premises) {
    gl::StrSpan out[4];
    const char* chain =
        "(in3[marker1,marker2,u_9,u_5])!(=[marker2,u_9])(=[s(u_rec),u_2])(preorder[u_1,u_4,u_7,marker2])";
    ASSERT_EQ(gl::ScopeSyntax::splitChainElements(span(chain), out, 4), 4);
    ASSERT_TRUE(gl::equalSpans(out[0], span("(in3[marker1,marker2,u_9,u_5])")));
    ASSERT_TRUE(gl::equalSpans(out[1], span("!(=[marker2,u_9])")));
    ASSERT_TRUE(gl::equalSpans(out[2], span("(=[s(u_rec),u_2])")));
    ASSERT_TRUE(gl::equalSpans(out[3], span("(preorder[u_1,u_4,u_7,marker2])")));
    ASSERT_EQ(gl::ScopeSyntax::splitChainElements(span(""), out, 4), 0);
}

// The fully specified premises of a hypothesis as ground statements: a strong
// catch fills marker1 and marker2 with its two names and every fixed argument
// loses its u_ prefix, a repeated one included. A premise with a variable is
// no missing expression; without a catch a marker is a variable
// (D-575).
TEST(induction_scope, ground_premises_fill_the_markers_and_drop_the_fixed_prefix) {
    gl::ScratchArena& arena = stringArena();
    gl::ScratchScope frame(arena);
    const gl::StrSpan premises[5] = { span("(in3[marker1,marker2,u_9,u_5])"), span("!(=[marker2,u_9])"),
        span("(preorder[u_1,u_4,u_7,marker2])"), span("(in3[11,marker2,u_9,u_5])"),
        span("(in3[u_7,u_7,u_9,u_4])") };
    const gl::StrSpan names[2] = { span("int_0_lev_2_5"), span("int_0_lev_2_6") };
    gl::ScratchString ground[5];
    ASSERT_EQ(gl::ScopeSyntax::groundHypothesisPremisesScratch(arena, premises, 5, names, 2, ground, 5), 4);
    ASSERT_TRUE(text(ground[0]) == "(in3[int_0_lev_2_5,int_0_lev_2_6,9,5])");
    ASSERT_TRUE(text(ground[1]) == "!(=[int_0_lev_2_6,9])");
    ASSERT_TRUE(text(ground[2]) == "(preorder[1,4,7,int_0_lev_2_6])");
    ASSERT_TRUE(text(ground[3]) == "(in3[7,7,9,4])");
    ASSERT_EQ(gl::ScopeSyntax::groundHypothesisPremisesScratch(arena, premises, 5, nullptr, 0, ground, 5), 1);
    ASSERT_TRUE(text(ground[0]) == "(in3[7,7,9,4])");
}

namespace {

    /// @brief Prepare a fresh FTA2 block for strong catches: the sum and
    ///        product demands filed at main, their marked chains, and the
    ///        multiplication catcher as the block's installed hypothesis rule.
    ///
    /// @param ea           The FTA2 fixture.
    /// @param memory       The fresh block.
    /// @param productChain The catcher's premises, instruction form; the first
    ///                     is the bound premise.
    /// @param productN     The premise count.
    /// @param catcher      The catcher rule's text.
    void prepareStrongCatchBlock(gl::tests::ProverFixture& ea, gl::Memory& memory,
            const gl::StrSpan* productChain, int32_t productN, const char* catcher) {
        memory.level = 2;
        memory.nameMap.encode(gl::StrSpan("main", 4));
        const gl::StrSpan mainScope("main", 4);
        ASSERT_TRUE(ea.admissionMemory.updateAdmissionMapIntegrationDouble(memory,
            span("(in3[11,12,u_9,u_4])"), mainScope, gl::InductionDemand::sum));
        ASSERT_TRUE(ea.admissionMemory.updateAdmissionMapIntegrationDouble(memory,
            productChain[0], mainScope, gl::InductionDemand::product));
        // The lemma's induction variable n, recorded with the demands as the
        // grid builder records it (D-601).
        memory.inductionVariableDigits[0] = 9;
        memory.inductionVariableDigitCount = 1;
        gl::ScratchArena& strings = stringArena();
        gl::ScratchScope frame(strings);
        const gl::StrSpan sumChain[2] = { span("(in3[11,12,u_9,u_4])"), span("!(=[11,u_9])") };
        memory.inductionDoubleChainId = memory.nameMap.encode(
            gl::StrSpan(gl::ScopeSyntax::inductionDoubleChainScratch(strings, sumChain, 2)));
        memory.inductionProductChainId = memory.nameMap.encode(
            gl::StrSpan(gl::ScopeSyntax::inductionDoubleChainScratch(strings, productChain, productN)));
        memory.recursionProductImplId = memory.ruleInterner.encode(catcher);
    }

    /// @brief The one product-catch scope of a block: the scope named by the
    ///        product chain directly below main.
    ///
    /// @param memory The block after the catch.
    /// @return The scope name; a test failure when there is none or several.
    std::string productCatchScope(const gl::Memory& memory) {
        const std::string prefix = "main_boundary_is_" + memory.nameMap.decode(memory.inductionProductChainId) + "_";
        std::string found;
        for (gl::NameId id = 1; id <= memory.nameMap.nameCount(); ++id) {
            const std::string name = memory.nameMap.decode(id);
            if (name.rfind(prefix, 0) != 0 || name.find("_boundary_", prefix.size()) != std::string::npos) continue;
            ASSERT_TRUE(found.empty());
            found = name;
        }
        ASSERT_FALSE(found.empty());
        return found;
    }

} // namespace

// A strong catch fills both markers of its demand's chain. Each premise left
// with only fixed arguments and the two names that can key an OR park is a
// candidate for the hypothesis's missing expression in the new scope,
// collected with the catching rule as source; the bound premise and the
// distance cannot key a park. The statement door files a candidate only when
// the catch's products leave it missing: d >= 2 is filed, p | n, which the
// unfolding publishes in the scope, is not (D-575).
TEST(induction_scope, strong_catch_files_the_premise_its_scope_misses) {
    gl::tests::ProverFixture ea("FTA");
    const std::string compiled =
        ea.structuralCompiler.compileExpression("!(>[D](divisorSet[N,a,m,t,n,D])!(hasMinimum[N,a,D]))");
    ea.supportingRuleCompiler.compileDisintegrationPackages();
    const std::map<std::string, std::string> values{
        {"N", "1"}, {"a", "4"}, {"m", "5"}, {"t", "7"}, {"n", "9"}};
    const std::string source = ce::replaceKeysInString(compiled, values);
    ea.parameters.trackHistory = true;
    const gl::StrSpan productChain[4] = { span("(in3[12,11,u_9,u_5])"), span("!(=[11,u_9])"),
        span("(preorder[u_1,u_5,12,u_9])"), span("(preorder[u_1,u_4,u_7,11])") };
    const char* catcher = "(>[11,12](in3[12,11,u_9,u_5])(>[]!(=[11,u_9])(>[](preorder[u_1,u_5,12,u_9])"
        "(>[](preorder[u_1,u_4,u_7,11])(hasPrimeFactorization[u_1,u_3,u_4,u_5,u_2,u_6,u_7,11])))))";
    const gl::StrSpan mainScope("main", 4);
    int levels[2] = {0, 2};

    gl::Memory collected;
    prepareStrongCatchBlock(ea, collected, productChain, 4, catcher);
    gl::ScratchArena& arena = gl::genScratchArenas().forSlot(gl::genScratchArenas().slotCount() - 1);
    gl::DisintProducts output(&arena);
    ea.disintegrationProcessor.disintegrateExpr2(gl::StrSpan(source), collected, 0, false,
        mainScope, output, true, false, levels, 2, true);
    const std::string productScope = productCatchScope(collected);
    gl::StrSpan names[2];
    const std::string productPayload = productScope.substr(std::string("main_boundary_").size());
    gl::ScopeSyntax::inductionScopeNames(gl::StrSpan(productPayload), 2, names);
    const std::string p = names[0].toStdString();
    const std::string d = names[1].toStdString();
    std::set<std::tuple<std::string, std::string, std::string>> demands;
    output.hypothesisDemands.forEachSortedWithSource([&](gl::StrSpan premise, gl::StrSpan scope, gl::StrSpan from) {
        demands.insert(std::make_tuple(premise.toStdString(), scope.toStdString(), from.toStdString()));
    });
    const std::set<std::tuple<std::string, std::string, std::string>> expected{
        std::make_tuple("(preorder[1,4,7," + d + "])", productScope, std::string(catcher)),
        std::make_tuple("(preorder[1,5," + p + ",9])", productScope, std::string(catcher)) };
    ASSERT_TRUE(demands == expected);
    ASSERT_EQ(collected.overallHashMemory.admissionMapOrdis2.count(), 0);

    gl::Memory filed;
    prepareStrongCatchBlock(ea, filed, productChain, 4, catcher);
    const gl::TransientOrigin origin{ true, gl::OriginTag::taskFormulation, nullptr, 0 };
    ea.statementMemory.addExprToMemoryBlock(gl::StrSpan(source), filed, 0, 1, levels, 2, origin,
        -1, -1, mainScope, false);
    const std::string doorScope = productCatchScope(filed);
    const std::string doorPayload = doorScope.substr(std::string("main_boundary_").size());
    gl::ScopeSyntax::inductionScopeNames(gl::StrSpan(doorPayload), 2, names);
    const std::string doorP = names[0].toStdString();
    const std::string doorD = names[1].toStdString();
    const auto demandsAt = [&](const std::string& premise) {
        int64_t pk = 0;
        if (!gl::lookupTemplateKey(filed.templateInterner, filed.nameMap, premise, doorScope, pk))
            return gl::AdmissionOrdis2ValueSet(gl::DecodedAdmissionOrdis2ValueLess{ &filed.ruleInterner });
        return gl::admissionOrdis2RecordsAt(filed.overallHashMemory.admissionMapOrdis2, pk, filed.ruleInterner);
    };
    const gl::AdmissionOrdis2ValueSet missing = demandsAt("(preorder[1,4,7," + doorD + "])");
    ASSERT_EQ(missing.size(), std::size_t(1));
    ASSERT_EQ(missing.begin()->sourceImplId, filed.recursionProductImplId);
    ASSERT_TRUE(missing.begin()->levels.empty());
    const std::string published = "(preorder[1,5," + doorP + ",9])";
    ASSERT_TRUE(gl::lookupStatementFlags(filed.intKnownStatements, filed.nameMap,
        gl::StrSpan(published), gl::StrSpan(doorScope)) != nullptr);
    ASSERT_TRUE(demandsAt(published).empty());
    ASSERT_EQ(filed.overallHashMemory.admissionMapOrdis2.count(), 1);
}

// A forced disintegration in a strong step block records the forced witnesses
// typed `(1)` with their minting scope for the successor demand: the end index
// k of a shot `hasPrimeFactorization` (typed by `in[k,N]`), not the sequence P.
// A block without the strong induction demand records nothing
// (D-571).
TEST(induction_scope, forced_witness_typed_one_records_its_successor_demand) {
    gl::tests::ProverFixture ea("FTA");
    const gl::StrSpan mainScope("main", 4);
    const std::string source = "(hasPrimeFactorization[1,3,4,5,2,6,7,9])";
    int levels[2] = {0, 2};
    gl::ScratchArena& arena = gl::genScratchArenas().forSlot(gl::genScratchArenas().slotCount() - 1);

    gl::Memory strong;
    strong.level = 2;
    strong.nameMap.encode(mainScope);
    ASSERT_TRUE(ea.admissionMemory.updateAdmissionMapIntegrationDouble(strong,
        span("(in3[11,12,u_9,u_4])"), mainScope, gl::InductionDemand::sum));
    ASSERT_TRUE(strong.inductionDoubleKey != 0);
    // The lemma's induction variable n, recorded with the demand as the grid
    // builder records it (D-601).
    strong.inductionVariableDigits[0] = 9;
    strong.inductionVariableDigitCount = 1;
    gl::DisintProducts output(&arena);
    ea.disintegrationProcessor.disintegrateExpr2(gl::StrSpan(source), strong, 0, true,
        mainScope, output, true, false, levels, 2, true);
    std::set<std::pair<std::string, std::string>> demands;
    output.successorDemands.forEachSorted([&](gl::StrSpan witness, gl::StrSpan scope) {
        demands.insert(std::make_pair(witness.toStdString(), scope.toStdString()));
    });
    ASSERT_EQ(demands.size(), std::size_t(1));
    const std::string k = demands.begin()->first;
    ASSERT_EQ(demands.begin()->second, std::string("main"));
    ASSERT_EQ(k.rfind("force_lev_2_", 0), std::size_t(0));
    bool typed = false;
    output.statements.forEachSorted([&](gl::StrSpan text, gl::StrSpan) {
        if (text.toStdString() == "(in[" + k + ",1])") typed = true;
    });
    ASSERT_TRUE(typed);

    gl::Memory plain;
    plain.level = 2;
    plain.nameMap.encode(mainScope);
    gl::DisintProducts plainOutput(&arena);
    ea.disintegrationProcessor.disintegrateExpr2(gl::StrSpan(source), plain, 0, true,
        mainScope, plainOutput, true, false, levels, 2, true);
    ASSERT_TRUE(plainOutput.statements.count() > 0);
    ASSERT_EQ(plainOutput.successorDemands.count(), 0);
}

// The successor demand of a forced witness is `(in2[w,marker,s])`, s the
// successor argument of the block's recursion hypothesis, in the algebra
// admission map at the scope that minted the witness and nowhere else
// (D-571).
TEST(induction_scope, forced_successor_demand_is_filed_at_the_minting_scope) {
    gl::tests::ProverFixture ea("FTA");
    const gl::StrSpan mainScope("main", 4);
    gl::Memory block;
    block.level = 2;
    block.nameMap.encode(mainScope);
    ASSERT_TRUE(ea.admissionMemory.updateAdmissionMapIntegrationDouble(block,
        span("(in3[11,12,u_9,u_4])"), mainScope, gl::InductionDemand::sum));
    block.recursionHypothesisId = block.nameMap.encode(span("(in2[rec,9,3])"));
    const gl::NameId minted = block.nameMap.encodePush(gl::NameMap::MAIN_ID,
        std::string("is_(in3[2,3,marker,4])_it_0_lev_2_7"));
    const std::string mintedName = block.nameMap.decode(minted);

    const int32_t before = block.overallHashMemory.admissionMap.count();
    ea.admissionMemory.fileForcedSuccessorDemand(block, span("force_lev_2_5"), gl::StrSpan(mintedName));
    ASSERT_EQ(block.overallHashMemory.admissionMap.count(), before + 1);
    int64_t pk = 0;
    ASSERT_TRUE(gl::lookupTemplateKey(block.templateInterner, block.nameMap,
        "(in2[force_lev_2_5,marker,3])", mintedName, pk));
    ASSERT_TRUE(block.overallHashMemory.admissionMap.lookup(pk) != 0);

    // An ordinary value under the recursion status: the successor it admits
    // becomes a product of recursion, and the value keeps the ordinary gate.
    const int32_t id = block.overallHashMemory.admissionMap.lookup(pk);
    ASSERT_EQ(block.overallHashMemory.admissionMap.runLen(id), 1);
    gl::ScratchArena& arena = gl::genScratchArenas().forSlot(gl::genScratchArenas().slotCount() - 1);
    int32_t length = 0;
    const char* bytes = block.overallHashMemory.admissionMap.peekRecordBytes(id, 0, length, arena);
    ASSERT_EQ(gl::AdmissionValueBlobView(bytes, length).flagByte(), static_cast<uint8_t>(0));
    const uint8_t* status = block.overallHashMemory.admissionStatusMap.find(pk);
    ASSERT_TRUE(status != nullptr && *status == 1);
    int64_t mainPk = 0;
    ASSERT_FALSE(gl::lookupTemplateKey(block.templateInterner, block.nameMap,
        "(in2[force_lev_2_5,marker,3])", "main", mainPk)
        && block.overallHashMemory.admissionMap.lookup(mainPk) != 0);
}

// At a minting scope with an equivalence class the admission chain is
// canonicalized: its fixed arguments become u_ names, and the renamer accepts
// only those and numbered arguments. The successor demand still lands as
// (in2[w,marker,s]) at that scope, its chain carrying a free number where the
// demand has its marker (D-571).
TEST(induction_scope, forced_successor_demand_is_filed_at_a_scope_with_equalities) {
    gl::tests::ProverFixture ea("FTA");
    gl::Memory& block = ea.body;
    const gl::StrSpan mainScope("main", 4);
    ASSERT_TRUE(ea.admissionMemory.updateAdmissionMapIntegrationDouble(block,
        span("(in3[11,12,u_9,u_4])"), mainScope, gl::InductionDemand::sum));
    block.recursionHypothesisId = block.nameMap.encode(span("(in2[rec,9,3])"));
    const gl::NameId minted = block.nameMap.encodePush(gl::NameMap::MAIN_ID,
        std::string("is_(in3[2,3,marker,4])_it_0_lev_2_7"));
    const std::string mintedName = block.nameMap.decode(minted);

    int level = 2;
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::PagedVector<gl::IntEncodedExpr> rows(&gl::genScratchArenas().forSlot(0), &dirty);
    ea.statementMemory.addStatement(gl::StrSpan("(=[a,b])", 8), block, true, &level, 1,
        gl::TransientOrigin{true, gl::OriginTag::broadcast, nullptr, 0}, gl::StrSpan(mintedName), rows);
    ea.canonicalizer.initializeCanonicalScope(block, minted);
    const int32_t bucket = block.canonicalEquivalenceClassesMap.lookup(minted);
    ASSERT_TRUE(bucket != 0 && block.canonicalEquivalenceClassesMap.runLen(bucket) != 0);

    ea.admissionMemory.fileForcedSuccessorDemand(block, span("force_lev_2_5"), gl::StrSpan(mintedName));
    int64_t pk = 0;
    ASSERT_TRUE(gl::lookupTemplateKey(block.templateInterner, block.nameMap,
        "(in2[force_lev_2_5,marker,3])", mintedName, pk));
    ASSERT_TRUE(block.overallHashMemory.admissionMap.lookup(pk) != 0);
}
