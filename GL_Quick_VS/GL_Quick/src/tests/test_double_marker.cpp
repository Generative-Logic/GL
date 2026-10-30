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
/// @brief Tests for the double-marker integration demands
///        (D-492).
///
/// @details
/// Covers the `AdmissionMemory` side: the key template of a disjunct with
/// two open arguments, the probe template of a single-marker constituent,
/// predicate visibility, the route-5 probe at a scope and its strict
/// ancestors, and the filing door with its eligibility. The disjuncts are
/// the two `bezEq` leaves of FTA row 121's flattened `bez1` goal.

#include "../memory.hpp"
#include <cstring>
#include <set>
#include <string>
#include "test_harness.hpp"
#include "prover_fixture.hpp"

#include "../prover.hpp"

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

    /// @brief Every double-marker key of @p m as `template@scope`.
    ///
    /// @param m The memory block.
    /// @return The keys.
    std::set<std::string> doubleKeys(gl::Memory& m) {
        std::set<std::string> out;
        const auto& keys = m.overallHashMemory.admissionMapIntegrationDouble;
        for (int32_t id = 1; id <= keys.count(); ++id) {
            const gl::StatementKey key = gl::Codec<gl::StatementKey>::decode(keys.keyAt(id));
            out.insert(m.templateInterner.decode(key.orig) + "@" + m.nameMap.decode(key.validity));
        }
        return out;
    }

    // Row 121's two or9 disjuncts: a = u_12, b = u_9, x = pi_lev_3_12,
    // y = pi_lev_3_13.
    const char* const kDisjunctA = "(bezEq[u_1,u_4,u_5,u_6,u_12,pi_lev_3_12,u_9,pi_lev_3_13])";
    const char* const kDisjunctB = "(bezEq[u_1,u_4,u_5,u_6,u_9,pi_lev_3_13,u_12,pi_lev_3_12])";

} // namespace

// The key keeps the u_ arguments and marks the two open positions in order;
// any other count of open positions, or one name at both, is no double key.
TEST(double_marker, key_template_needs_two_distinct_open_arguments) {
    gl::ScratchArena& arena = stringArena();
    gl::ScratchScope frame(arena);
    gl::ScratchString out;
    ASSERT_TRUE(gl::AdmissionMemory::makeDoubleMarkerKeyScratch(span(kDisjunctA), arena, out));
    ASSERT_TRUE(text(out) == "(bezEq[u_1,u_4,u_5,u_6,u_12,marker1,u_9,marker2])");
    ASSERT_TRUE(gl::AdmissionMemory::makeDoubleMarkerKeyScratch(span(kDisjunctB), arena, out));
    ASSERT_TRUE(text(out) == "(bezEq[u_1,u_4,u_5,u_6,u_9,marker1,u_12,marker2])");
    ASSERT_TRUE(gl::AdmissionMemory::makeDoubleMarkerKeyScratch(span("!(=[pi_lev_0_1,pi_lev_0_2])"), arena, out));
    ASSERT_TRUE(text(out) == "!(=[marker1,marker2])");

    ASSERT_FALSE(gl::AdmissionMemory::makeDoubleMarkerKeyScratch(
        span("(bezEq[u_1,u_4,u_5,u_6,u_12,pi_lev_3_12,u_9,u_7])"), arena, out));
    ASSERT_FALSE(gl::AdmissionMemory::makeDoubleMarkerKeyScratch(
        span("(bezEq[u_1,u_4,u_5,u_6,x,pi_lev_3_12,u_9,y])"), arena, out));
    ASSERT_FALSE(gl::AdmissionMemory::makeDoubleMarkerKeyScratch(
        span("(bezEq[u_1,u_4,u_5,u_6,u_12,x,u_9,x])"), arena, out));
    ASSERT_FALSE(gl::AdmissionMemory::makeDoubleMarkerKeyScratch(
        span("(in3[u_1,u_2,u_3,u_4])"), arena, out));
}

// The probe marks the witness and the chosen other position (earlier one
// first) and gives every remaining argument the stored keys' u_ spelling.
TEST(double_marker, probe_template_marks_the_witness_and_one_other_position) {
    gl::ScratchArena& arena = stringArena();
    gl::ScratchScope frame(arena);
    const gl::StrSpan caseHead = span("(bezEq[1,4,5,6,12,force_lev_3_1,9,marker])");
    ASSERT_TRUE(text(gl::AdmissionMemory::makeDoubleMarkerQueryScratch(caseHead, 7, 5, arena))
        == "(bezEq[u_1,u_4,u_5,u_6,u_12,marker1,u_9,marker2])");
    ASSERT_TRUE(text(gl::AdmissionMemory::makeDoubleMarkerQueryScratch(caseHead, 7, 0, arena))
        == "(bezEq[marker1,u_4,u_5,u_6,u_12,u_force_lev_3_1,u_9,marker2])");
    ASSERT_TRUE(text(gl::AdmissionMemory::makeDoubleMarkerQueryScratch(
        span("(bezEq[1,4,5,6,12,marker,9,z])"), 5, 7, arena))
        == "(bezEq[u_1,u_4,u_5,u_6,u_12,marker1,u_9,marker2])");
    ASSERT_TRUE(text(gl::AdmissionMemory::makeDoubleMarkerQueryScratch(span("!(=[marker,7])"), 0, 1, arena))
        == "!(=[marker1,marker2])");
}

// Filing, visibility and the route-5 probe: a key at main admits a case
// branch's one-open constituent; a key at a branch admits only there; a
// differing fixed argument or a second marker never matches.
TEST(double_marker, filed_keys_admit_one_open_constituents_at_scope_and_below) {
    gl::tests::ProverFixture ana(std::string("FTA"));
    gl::Memory& mb = ana.body;
    const gl::NameId root = gl::NameMap::MAIN_ID;
    const gl::NameId child = mb.nameMap.encodePush(root, "case_child");
    const std::string childName = mb.nameMap.decode(child);
    const gl::NameId sibling = mb.nameMap.encodePush(root, "case_sibling");
    const std::string siblingName = mb.nameMap.decode(sibling);
    const gl::StrSpan mainScope("main", 4);

    // Filing: a new key, a repeat that files nothing, a key at the branch.
    ASSERT_TRUE(ana.admissionMemory.updateAdmissionMapIntegrationDouble(mb, span(kDisjunctA), mainScope));
    ASSERT_FALSE(ana.admissionMemory.updateAdmissionMapIntegrationDouble(mb, span(kDisjunctA), mainScope));
    ASSERT_TRUE(ana.admissionMemory.updateAdmissionMapIntegrationDouble(mb, span(kDisjunctB),
        gl::StrSpan(childName)));
    // One open argument, or a predicate that keys no integration
    // instruction, files nothing.
    ASSERT_FALSE(ana.admissionMemory.updateAdmissionMapIntegrationDouble(mb,
        span("(bezEq[u_1,u_4,u_5,u_6,u_12,pi_lev_3_12,u_9,u_7])"), mainScope));
    ASSERT_FALSE(ana.admissionMemory.updateAdmissionMapIntegrationDouble(mb,
        span("(in[pi_lev_3_12,pi_lev_3_13])"), mainScope));
    const std::set<std::string> expected = {
        "(bezEq[u_1,u_4,u_5,u_6,u_12,marker1,u_9,marker2])@main",
        "(bezEq[u_1,u_4,u_5,u_6,u_9,marker1,u_12,marker2])@" + childName };
    ASSERT_TRUE(doubleKeys(mb) == expected);
    for (int32_t id = 1; id <= mb.overallHashMemory.admissionMapIntegrationDouble.count(); ++id)
        ASSERT_EQ(mb.overallHashMemory.admissionMapIntegrationDouble.valueAt(id), int64_t(0));

    // Visibility: at the key's scope and below; a predicate without keys
    // is never visible.
    ASSERT_TRUE(gl::AdmissionMemory::doubleMarkerPredicateVisible(mb,
        span("(bezEq[1,4,5,6,12,y,9,marker])"), root));
    ASSERT_TRUE(gl::AdmissionMemory::doubleMarkerPredicateVisible(mb,
        span("(bezEq[1,4,5,6,12,y,9,marker])"), child));
    ASSERT_FALSE(gl::AdmissionMemory::doubleMarkerPredicateVisible(mb,
        span("(in3[1,4,marker,5])"), child));

    // Route 5: the main key admits at main and in the branch below it.
    const gl::StrSpan caseHead = span("(bezEq[1,4,5,6,12,force_lev_3_1,9,marker])");
    ASSERT_TRUE(ana.admissionMemory.isAdmittedIntegrationDouble(mb, caseHead, gl::StrSpan(childName)));
    ASSERT_TRUE(ana.admissionMemory.isAdmittedIntegrationDouble(mb, caseHead, mainScope));
    ASSERT_FALSE(ana.admissionMemory.isAdmittedIntegrationDouble(mb,
        span("(bezEq[1,4,5,6,13,force_lev_3_1,9,marker])"), gl::StrSpan(childName)));
    ASSERT_FALSE(ana.admissionMemory.isAdmittedIntegrationDouble(mb,
        span("(bezEq[1,4,5,6,12,marker,9,marker])"), gl::StrSpan(childName)));
    // The branch key admits in its branch only.
    const gl::StrSpan reverse = span("(bezEq[1,4,5,6,9,y,12,marker])");
    ASSERT_TRUE(ana.admissionMemory.isAdmittedIntegrationDouble(mb, reverse, gl::StrSpan(childName)));
    ASSERT_FALSE(ana.admissionMemory.isAdmittedIntegrationDouble(mb, reverse, mainScope));
    ASSERT_FALSE(ana.admissionMemory.isAdmittedIntegrationDouble(mb, reverse, gl::StrSpan(siblingName)));
    // The probe records nothing.
    ASSERT_TRUE(doubleKeys(mb) == expected);
}

// The strong step's bound premise files the block's induction demand and
// records its key; a goal filing records none. Route 5 through the induction
// key admits only when the original is allowed (lemma input values, sets and
// operators; D-502); a goal key admits regardless
// of the original.
TEST(double_marker, induction_demand_records_its_key_and_gates_route_5) {
    gl::tests::ProverFixture ana(std::string("FTA"));
    gl::Memory& mb = ana.body;
    const gl::NameId child = mb.nameMap.encodePush(gl::NameMap::MAIN_ID, "case_child");
    const std::string childName = mb.nameMap.decode(child);
    const gl::StrSpan mainScope("main", 4);

    ASSERT_TRUE(ana.admissionMemory.updateAdmissionMapIntegrationDouble(mb, span(kDisjunctA), mainScope));
    ASSERT_EQ(mb.inductionDoubleKey, int64_t(0));
    // m = 11 and x = 12 open, n = 10 and + = 4 fixed.
    ASSERT_TRUE(ana.admissionMemory.updateAdmissionMapIntegrationDouble(mb,
        span("(in3[11,12,u_10,u_4])"), mainScope, gl::InductionDemand::sum));
    int64_t key = 0;
    ASSERT_TRUE(gl::lookupTemplateKey(mb.templateInterner, mb.nameMap,
        std::string("(in3[marker1,marker2,u_10,u_4])"), std::string("main"), key));
    ASSERT_EQ(mb.inductionDoubleKey, key);
    ASSERT_EQ(mb.overallHashMemory.admissionMapIntegrationDouble.count(), 2);

    // r + marker = 10 below main: the induction key admits only when allowed.
    const gl::StrSpan remainderStep = span("(in3[r,marker,10,4])");
    ASSERT_TRUE(ana.admissionMemory.isAdmittedIntegrationDouble(mb, remainderStep, gl::StrSpan(childName)));
    ASSERT_TRUE(ana.admissionMemory.isAdmittedIntegrationDouble(mb, remainderStep, gl::StrSpan(childName), true));
    ASSERT_FALSE(ana.admissionMemory.isAdmittedIntegrationDouble(mb, remainderStep, gl::StrSpan(childName), false));
    ASSERT_FALSE(ana.admissionMemory.isAdmittedIntegrationDouble(mb, remainderStep, mainScope, false));
    // The goal key is not gated.
    const gl::StrSpan caseHead = span("(bezEq[1,4,5,6,12,force_lev_3_1,9,marker])");
    ASSERT_TRUE(ana.admissionMemory.isAdmittedIntegrationDouble(mb, caseHead, gl::StrSpan(childName), false));
    ASSERT_TRUE(ana.admissionMemory.isAdmittedIntegrationDouble(mb, caseHead, mainScope, false));
}

// One order per path (D-519): where the induction
// hypothesis already ate another slot, the strong demand is refused; at a
// scope where it ate nothing, or ate the strong slot itself, it admits.
TEST(double_marker, induction_demand_keeps_one_order_per_path) {
    gl::tests::ProverFixture ana(std::string("FTA"));
    gl::Memory& mb = ana.body;
    const gl::NameId child = mb.nameMap.encodePush(gl::NameMap::MAIN_ID, "case_child");
    const gl::NameId other = mb.nameMap.encodePush(gl::NameMap::MAIN_ID, "case_other");
    const std::string childName = mb.nameMap.decode(child);
    const std::string otherName = mb.nameMap.decode(other);
    const gl::StrSpan mainScope("main", 4);
    ASSERT_TRUE(ana.admissionMemory.updateAdmissionMapIntegrationDouble(mb,
        span("(in3[11,12,u_10,u_4])"), mainScope, gl::InductionDemand::sum));
    ASSERT_TRUE(mb.inductionDoubleKey != 0);

    const gl::StrSpan remainderStep = span("(in3[r,marker,10,4])");
    ASSERT_TRUE(ana.admissionMemory.isAdmittedIntegrationDouble(mb, remainderStep, gl::StrSpan(childName)));
    mb.recordInductionEat(child, 7);
    ASSERT_FALSE(ana.admissionMemory.isAdmittedIntegrationDouble(mb, remainderStep, gl::StrSpan(childName)));
    ASSERT_TRUE(ana.admissionMemory.isAdmittedIntegrationDouble(mb, remainderStep, gl::StrSpan(otherName)));
    mb.recordInductionEat(other, gl::Memory::kStrongInductionSlot);
    ASSERT_TRUE(ana.admissionMemory.isAdmittedIntegrationDouble(mb, remainderStep, gl::StrSpan(otherName)));
}

// Linking through the path facts (D-501):
// D1's chain reaches every enclosing witness (p's `in3[r,p,b,+]` reaches r,
// t's `and30[q,b,t,*,r,a,+,N]` reaches t and q), also transitively through
// separate facts; a witness no fact connects is not reached, and a shared
// fixed argument links nothing.
TEST(double_marker, enclosing_witnesses_link_through_the_path_facts) {
    const gl::StrSpan d1[] = {
        span("(in[p,1])"), span("(in3[r,p,10,4])"),
        span("(in[t,1])"), span("(and30[q,10,t,5,r,9,4,1])"),
        span("(in[q,1])"), span("(existence20[1,q,10,5,r,9,4])"),
        span("(in[r,1])"), span("(remainder[1,4,5,9,10,r])") };
    const gl::StrSpan enclosing[] = { span("t"), span("q"), span("r") };
    ASSERT_TRUE(gl::AdmissionMemory::doubleMarkerEnclosingLinked(span("p"), enclosing, 3, d1, 8));

    // q reaches p only through t and r, each through its own fact.
    const gl::StrSpan chain[] = { span("(in3[q,10,t,5])"), span("(in3[t,r,9,4])"), span("(in3[r,p,10,4])") };
    ASSERT_TRUE(gl::AdmissionMemory::doubleMarkerEnclosingLinked(span("p"), enclosing, 3, chain, 3));

    // q is named only with fixed arguments: not reached.
    const gl::StrSpan loose[] = { span("(in3[r,p,10,4])"), span("(in3[t,r,9,4])"),
        span("(in[q,1])"), span("(in3[q,10,10,5])") };
    ASSERT_FALSE(gl::AdmissionMemory::doubleMarkerEnclosingLinked(span("p"), enclosing, 3, loose, 4));

    // The fixed 10 shared with the hit's fact does not link t.
    const gl::StrSpan onlyT[] = { span("t") };
    const gl::StrSpan fixedOnly[] = { span("(in3[r,p,10,4])"), span("(in3[t,10,9,4])") };
    ASSERT_FALSE(gl::AdmissionMemory::doubleMarkerEnclosingLinked(span("p"), onlyT, 1, fixedOnly, 2));

    // A negated fact links like any other.
    const gl::StrSpan negated[] = { span("!(=[t,p])") };
    ASSERT_TRUE(gl::AdmissionMemory::doubleMarkerEnclosingLinked(span("p"), onlyT, 1, negated, 1));
}
