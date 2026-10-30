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
// Unit tests for the goal-driven operator demand
// (D-574): the two integration install
// kinds, the near-miss slot gate and its admission pass, the rule search on
// the shape of FTA2 row 8, the marked-head install, and the drain's routing
// of a near-miss record to the rule search.
#include "../memory.hpp"
#include "test_harness.hpp"
#include "prover_fixture.hpp"
#include "compiled_definition_fixture.hpp"
#include "compact_rule_processor_fixture.hpp"

#include "../prover.hpp"

#include <cstring>
#include <string>
#include <vector>

namespace {

    /// @brief Span over a string literal (tests only).
    gl::StrSpan sp(const char* s) {
        return gl::StrSpan(s, static_cast<int32_t>(std::strlen(s)));
    }

    /// @brief Install one implication into the block's overall hash memory
    ///        exactly as the door does.
    void install(gl::tests::ProverFixture& ea, const std::string& impl) {
        gl::ScratchArena& kvArena = gl::scratchArenas().forSlot(gl::scratchArenas().slotCount() - 1);
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
            ea.body, ea.body.overallHashMemory, lv, 1, gl::StrSpan(impl),
            ea.parameters.maxIterationNumberVariable,
            ea.parameters.standardMaxSecondaryNumber, false,
            ea.parameters.minNumOperatorsKey, gl::StrSpan("implication", 11),
            false, gl::StrSpan(impl), sp("main"));
    }

    /// @brief One marked-head implication the rule search reported.
    struct Variant {
        std::vector<std::string> key;
        std::string marked;
        std::string identity;
    };

    /// @brief Run the rule search for @p expression at @p validity and
    ///        collect what it reports.
    std::vector<Variant> search(gl::tests::ProverFixture& ea, const char* expression,
                                const char* validity) {
        ea.integrationProcessor.ensureOriginalsHeadIndexBuilt(ea.body);
        gl::ScratchArena& strings = gl::scratchArenas().forSlot(gl::scratchArenas().slotCount() - 1);
        gl::ScratchArena& gen = gl::genScratchArenas().forSlot(gl::genScratchArenas().slotCount() - 1);
        gl::ScratchScope frame(strings);
        std::vector<Variant> out;
        ea.integrationProcessor.forEachGoalDemandVariant(ea.body, sp(expression), sp(validity),
            strings, gen,
            [&](const gl::StrSpan* key, int32_t keyN, gl::StrSpan marked, gl::StrSpan identity) {
                Variant v;
                for (int32_t i = 0; i < keyN; ++i) v.key.push_back(key[i].toStdString());
                v.marked = marked.toStdString();
                v.identity = identity.toStdString();
                out.push_back(v);
            });
        return out;
    }

    /// @brief The goal-driven marker LMVs (`isMarker`, justification
    ///        `goalDemand`) of one instance, by decoded value.
    std::vector<std::string> goalMarkers(gl::Memory& m, gl::HashMemory& hm) {
        std::vector<std::string> values;
        const auto& em = hm.encodedMap;
        for (int32_t id = 1; id <= em.count(); ++id) {
            for (const gl::LocalMemoryValue& lmv : em.recordsAt(id)) {
                if (!lmv.isMarker || lmv.justification != gl::RuleJustification::goalDemand) continue;
                values.push_back(m.ruleInterner.decode(lmv.valueId));
            }
        }
        return values;
    }

    /// @brief FTA2 row 8: an ordered sequence extended by an element no
    ///        smaller than its last entry stays ordered.
    const char* const kRowEight =
        "(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](in2[9,10,3])"
        "(>[11](sequence[1,4,2,9,11])(>[12](in2[9,12,11])(>[](ordered[1,4,11])"
        "(>[13](preorder[1,4,13,12])(>[14](appendToSequence[10,13,11,14])(ordered[1,4,14]))))))))";

    /// @brief The Peano fixture's synthetic compound `zzcomp` (a non-atomic
    ///        entity the near-miss gate accepts).
    void addCompound(gl::tests::ProverFixture& ea) {
        gl::CompiledDefinitionFixture::compiledExpressions(ea.definitions)["zzcomp"] =
            gl::LogicalEntity("existence",
                { "(in[w1,u_1])", "(in2[u_2,w1,u_3])" },
                "(zzcomp[u_1,u_2,u_3,u_4])", 4);
        ea.supportingRuleCompiler.compileDisintegrationPackages();
    }

} // namespace

// Integration and IntegrationFlat share the integration route; the local and
// external kinds do not.
TEST(goal_demand, install_kinds_split_into_two_integration_kinds) {
    ASSERT_TRUE(gl::isIntegrationInstallKind(gl::RuleInstallKind::Integration));
    ASSERT_TRUE(gl::isIntegrationInstallKind(gl::RuleInstallKind::IntegrationFlat));
    ASSERT_FALSE(gl::isIntegrationInstallKind(gl::RuleInstallKind::Local));
    ASSERT_FALSE(gl::isIntegrationInstallKind(gl::RuleInstallKind::External));
}

// The shared subset test: an argument is fixed by another non-anchor premise
// or by an anchor slot (D-287); at least one argument must come from a
// substantive premise.
TEST(goal_demand, slot_arguments_bound_by_others_matrix) {
    gl::tests::ProverFixture ea("Peano");
    const std::string anchor = "(AnchorPeano[u_1,u_2,u_3,u_4,u_5,u_6])";
    const std::string b1 = "(in3[u_1,u_4,u_10,u_5])";
    const std::string b2 = "(in3[u_12,u_9,u_11,u_5])";
    const std::string b3 = "(in2[u_10,u_12,u_5])";
    const auto bound = [&](const std::string& slot) {
        const gl::StrSpan key[5] = { gl::StrSpan(anchor), gl::StrSpan(b1), gl::StrSpan(b2),
                                     gl::StrSpan(b3), gl::StrSpan(slot) };
        return ea.admissionPolicy.slotArgumentsBoundByOthers(key, 5, 4);
    };
    ASSERT_TRUE(bound("(zzcomp[u_1,u_4,u_10,u_12])"));
    // u_6 is an anchor slot: ground by construction.
    ASSERT_TRUE(bound("(zzcomp[u_1,u_4,u_10,u_6])"));
    // Anchor slots alone: no substantive binder.
    ASSERT_FALSE(bound("(zzcomp[u_2,u_3,u_6,u_2])"));
    // u_77 is named nowhere else.
    ASSERT_FALSE(bound("(zzcomp[u_1,u_4,u_10,u_77])"));
}

// The near-miss gate: a positive compound premise, never the anchor or an
// atomic premise, every argument fixed, and at least four other non-anchor
// premises beside it.
TEST(goal_demand, near_miss_slot_gate_matrix) {
    gl::tests::ProverFixture ea("Peano");
    addCompound(ea);
    const std::string anchor = "(AnchorPeano[u_1,u_2,u_3,u_4,u_5,u_6])";
    const std::string b1 = "(in3[u_1,u_4,u_10,u_5])";
    const std::string b2 = "(in3[u_12,u_9,u_11,u_5])";
    const std::string b3 = "(in2[u_10,u_12,u_5])";
    const std::string b4 = "(in2[u_9,u_13,u_5])";
    const std::string slot = "(zzcomp[u_1,u_4,u_10,u_12])";
    {   // Four others: qualifies; the anchor and an atomic premise never do.
        const gl::StrSpan key[6] = { gl::StrSpan(anchor), gl::StrSpan(b1), gl::StrSpan(b2),
                                     gl::StrSpan(b3), gl::StrSpan(b4), gl::StrSpan(slot) };
        ASSERT_TRUE(ea.admissionPolicy.goalNearMissSlotQualifies(key, 6, 5));
        ASSERT_FALSE(ea.admissionPolicy.goalNearMissSlotQualifies(key, 6, 0));
        ASSERT_FALSE(ea.admissionPolicy.goalNearMissSlotQualifies(key, 6, 1));
    }
    {   // Three others: below the floor.
        const gl::StrSpan key[5] = { gl::StrSpan(anchor), gl::StrSpan(b1), gl::StrSpan(b2),
                                     gl::StrSpan(b3), gl::StrSpan(slot) };
        ASSERT_FALSE(ea.admissionPolicy.goalNearMissSlotQualifies(key, 5, 4));
    }
    {   // A negated compound is no missing premise to demand.
        const std::string negated = "!(zzcomp[u_1,u_4,u_10,u_12])";
        const gl::StrSpan key[6] = { gl::StrSpan(anchor), gl::StrSpan(b1), gl::StrSpan(b2),
                                     gl::StrSpan(b3), gl::StrSpan(b4), gl::StrSpan(negated) };
        ASSERT_FALSE(ea.admissionPolicy.goalNearMissSlotQualifies(key, 6, 5));
    }
    {   // An argument no other premise fixes.
        const std::string open = "(zzcomp[u_1,u_4,u_10,u_77])";
        const gl::StrSpan key[6] = { gl::StrSpan(anchor), gl::StrSpan(b1), gl::StrSpan(b2),
                                     gl::StrSpan(b3), gl::StrSpan(b4), gl::StrSpan(open) };
        ASSERT_FALSE(ea.admissionPolicy.goalNearMissSlotQualifies(key, 6, 5));
    }
}

// The near-miss pass on the shape of I4's flat rule: with the flag, the
// qualifying slots `sequence` and `ordered` install demand variants stamped
// `goalDemand`; without it, none. `primeseq` and `fold` hold arguments only
// the head names, so they stay out.
TEST(goal_demand, near_miss_pass_installs_goal_demand_variants) {
    const std::vector<std::string> keyText = {
        "(in[pi_1,u_1])",
        "(sequence[u_1,u_4,u_2,pi_1,pi_2])",
        "(primeseq[u_1,u_4,u_5,u_6,u_7,pi_2])",
        "(ordered[u_1,u_4,pi_2])",
        "(fold[u_1,u_3,u_4,u_5,pi_2,u_2,pi_1,u_9])" };
    const auto demandHeads = [&](bool goalNearMiss) {
        gl::tests::ProverFixture ea("FTA");
        ea.parameters.minLenLongKey = 100;
        gl::Memory m;
        std::vector<std::string> heads;
        if (m.nameMap.encode("main") != gl::NameMap::MAIN_ID) return heads;
        gl::StrSpan key[5];
        for (int32_t i = 0; i < 5; ++i) key[i] = gl::StrSpan(keyText[static_cast<std::size_t>(i)]);
        ea.admissionMarkers.makeNormalizedKeysForAdmission(key, 5, m.overallHashMemory, m.nameMap,
            m.ruleInterner, sp("(hasPrimeFactorization[u_1,u_3,u_4,u_5,u_2,u_6,u_7,u_9])"),
            /*minNumOperatorsKey=*/100, sp("(>[]flatRule(=[9,9]))"), sp("main"), nullptr,
            gl::RuleIndexOp{}, nullptr, nullptr, /*integrationJustified=*/true, goalNearMiss);
        const auto& em = m.overallHashMemory.encodedMap;
        for (int32_t id = 1; id <= em.count(); ++id) {
            for (const gl::LocalMemoryValue& lmv : em.recordsAt(id)) {
                if (lmv.justification != gl::RuleJustification::goalDemand) continue;
                heads.push_back(gl::extractExpressionSpan(
                    gl::StrSpan(m.ruleInterner.decode(lmv.valueId))).toStdString()
                    + (lmv.ordis2Demand ? "" : "!kind"));
            }
        }
        return heads;
    };
    const std::vector<std::string> on = demandHeads(true);
    ASSERT_FALSE(on.empty());
    bool sawSequence = false;
    bool sawOrdered = false;
    for (const std::string& h : on) {
        ASSERT_TRUE(h == "sequence" || h == "ordered");
        sawSequence = sawSequence || h == "sequence";
        sawOrdered = sawOrdered || h == "ordered";
    }
    ASSERT_TRUE(sawSequence);
    ASSERT_TRUE(sawOrdered);
    ASSERT_TRUE(demandHeads(false).empty());
}

// The rule search on FTA2 row 8 for the missing `(ordered[1,4,G])`: the head
// binds 1, 4 and G; each `in2` premise's output is open and its other
// arguments are fixed by four positive premises without that output, so both
// are reported; `appendToSequence` is not, since the head fixes its output.
TEST(goal_demand, rule_search_binds_row_eight_and_reports_its_open_operators) {
    gl::tests::ProverFixture ea("FTA");
    ea.parameters.allow_multiplication = false;
    install(ea, kRowEight);
    const std::vector<Variant> found = search(ea, "(ordered[1,4,G])", "main");
    ASSERT_EQ(found.size(), static_cast<std::size_t>(2));

    const std::string anchor = "(AnchorFTA[u_1,2,3,u_4,5,6,7,8])";
    ASSERT_EQ(found[0].marked, std::string("10"));
    ASSERT_TRUE((found[0].key == std::vector<std::string>{ anchor,
        "(sequence[u_1,u_4,2,9,11])", "(in2[9,12,11])", "(ordered[u_1,u_4,11])",
        "(preorder[u_1,u_4,13,12])", "(in2[9,10,3])" }));
    ASSERT_EQ(found[0].identity, std::string(
        "(>[2,3,5,6,7,8](AnchorFTA[u_1,2,3,u_4,5,6,7,8])(>[9,11](sequence[u_1,u_4,2,9,11])"
        "(>[12](in2[9,12,11])(>[](ordered[u_1,u_4,11])(>[13](preorder[u_1,u_4,13,12])"
        "(in2[9,marker,3]))))))"));

    ASSERT_EQ(found[1].marked, std::string("12"));
    ASSERT_TRUE((found[1].key == std::vector<std::string>{ anchor,
        "(in2[9,10,3])", "(sequence[u_1,u_4,2,9,11])", "(ordered[u_1,u_4,11])",
        "(appendToSequence[10,13,11,u_G])", "(in2[9,12,11])" }));
    ASSERT_EQ(found[1].identity, std::string(
        "(>[2,3,5,6,7,8](AnchorFTA[u_1,2,3,u_4,5,6,7,8])(>[9,10](in2[9,10,3])"
        "(>[11](sequence[u_1,u_4,2,9,11])(>[](ordered[u_1,u_4,11])"
        "(>[13](appendToSequence[10,13,11,u_G])(in2[9,marker,11]))))))"));

    // A head of the other sign, or another operator, is no candidate.
    ASSERT_TRUE(search(ea, "!(ordered[1,4,G])", "main").empty());
    ASSERT_TRUE(search(ea, "(primeseq[1,4,5,6,7,G])", "main").empty());
}

// The rule search reports nothing for rules that miss its conditions: three
// positive premises beside the operator premise, an operator output the head
// fixes, and no operator premise at all.
TEST(goal_demand, rule_search_skips_rules_outside_its_conditions) {
    gl::tests::ProverFixture ea("FTA");
    ea.parameters.allow_multiplication = false;
    // Three present premises beside each in2 premise.
    install(ea, "(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](in2[9,10,3])"
        "(>[11](sequence[1,4,2,9,11])(>[12](in2[9,12,11])"
        "(>[13,14](appendToSequence[10,13,11,14])(ordered[1,4,14]))))))");
    // The one operator premise outputs the head's argument.
    install(ea, "(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,11](sequence[1,4,2,9,11])"
        "(>[](ordered[1,4,11])(>[](in[9,1])(>[13](preorder[1,4,13,9])"
        "(>[14](appendToSequence[9,13,11,14])(ordered[1,4,14])))))))");
    // No operator premise.
    install(ea, "(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,11](sequence[1,4,2,9,11])"
        "(>[](ordered[1,4,11])(>[](in[9,1])(>[10,12](sequence[1,4,2,10,12])"
        "(>[](in[10,1])(ordered[1,4,12])))))))");
    ASSERT_TRUE(search(ea, "(ordered[1,4,G])", "main").empty());
}

// The marked-head install writes one goal-driven marker family into each of
// the three integration instances, owned by the identity at its scope.
TEST(goal_demand, install_writes_the_marker_into_the_three_instances) {
    gl::tests::ProverFixture ea("Peano");
    gl::Memory& m = ea.body;
    const std::string p1 = "(in3[1,u_4,2,u_5])";
    const std::string p2 = "(in3[2,u_4,3,u_5])";
    const std::string p3 = "(in[1,u_1])";
    const std::string p4 = "(in[2,u_1])";
    const std::string op = "(in2[3,4,u_6])";
    const gl::StrSpan key[5] = { gl::StrSpan(p1), gl::StrSpan(p2), gl::StrSpan(p3),
                                 gl::StrSpan(p4), gl::StrSpan(op) };
    const char* identity = "(>[1,2,3](in3[1,u_4,2,u_5])(in2[3,marker,u_6]))";
    ea.hashRuleInstaller.installGoalDemandVariant(m, key, 5, sp("4"), sp(identity), sp("main"));
    gl::HashMemory* const instances[3] = {
        &m.overallHashMemory, &m.localHashMemory, &m.localHashMemoryDelta };
    for (gl::HashMemory* hm : instances) {
        const std::vector<std::string> markers = goalMarkers(m, *hm);
        ASSERT_FALSE(markers.empty());
        for (const std::string& v : markers) {
            ASSERT_TRUE(v.rfind("(in2[", 0) == 0);
            ASSERT_TRUE(v.find("marker") != std::string::npos);
        }
        const auto& em = hm->encodedMap;
        for (int32_t id = 1; id <= em.count(); ++id) {
            for (const gl::LocalMemoryValue& lmv : em.recordsAt(id)) {
                if (lmv.justification != gl::RuleJustification::goalDemand) continue;
                ASSERT_EQ(lmv.originalImplicationId, m.ruleInterner.encode(identity));
                ASSERT_EQ(lmv.validityId, gl::NameMap::MAIN_ID);
                ASSERT_FALSE(lmv.ordis2Demand);
            }
        }
    }
}

// The marked-head install raises each instance's key-length bound to the
// premise count of the whole key it writes: request growth stops at that
// bound, so an instance whose other rules are all shorter (a burst's delta)
// must still grow to the new key. A larger bound stays.
TEST(goal_demand, install_raises_each_instance_key_length_bound) {
    gl::tests::ProverFixture ea("Peano");
    gl::Memory& m = ea.body;
    const std::string p1 = "(in3[1,u_4,2,u_5])";
    const std::string p2 = "(in3[2,u_4,3,u_5])";
    const std::string p3 = "(in[1,u_1])";
    const std::string p4 = "(in[2,u_1])";
    const std::string op = "(in2[3,4,u_6])";
    const gl::StrSpan key[5] = { gl::StrSpan(p1), gl::StrSpan(p2), gl::StrSpan(p3),
                                 gl::StrSpan(p4), gl::StrSpan(op) };
    const char* identity = "(>[1,2,3](in3[1,u_4,2,u_5])(in2[3,marker,u_6]))";
    m.overallHashMemory.maxKeyLength = 6;
    m.localHashMemory.maxKeyLength = 2;
    m.localHashMemoryDelta.maxKeyLength = 0;
    ea.hashRuleInstaller.installGoalDemandVariant(m, key, 5, sp("4"), sp(identity), sp("main"));
    ASSERT_EQ(m.overallHashMemory.maxKeyLength, 6);
    ASSERT_EQ(m.localHashMemory.maxKeyLength, 4);
    ASSERT_EQ(m.localHashMemoryDelta.maxKeyLength, 4);
}

// The drain hands a near-miss record to the rule search instead of the
// OR-demand map; the search installs row 8's two marked-head implications
// once, and a repeated near-miss installs nothing more.
TEST(goal_demand, drain_routes_a_near_miss_and_installs_each_implication_once) {
    gl::tests::ProverFixture ea("FTA");
    ea.parameters.allow_multiplication = false;
    gl::Memory& m = ea.body;
    install(ea, kRowEight);

    gl::SealedPageSet pages;
    pages.bind(&gl::staticMemory());
    const auto sealed = [&pages](const std::string& v) {
        return gl::SealedString::copyFrom(pages, v.data(), static_cast<int32_t>(v.size()));
    };
    gl::Ordis2DemandRecord rec;
    rec.expression = sealed("(ordered[1,4,G])");
    rec.validityName = sealed("main");
    rec.sourceImplId = m.ruleInterner.encode("(>[]flatRule(=[9,9]))");
    rec.goalNearMiss = true;

    const auto markerCounts = [&]() {
        return std::vector<std::size_t>{ goalMarkers(m, m.overallHashMemory).size(),
            goalMarkers(m, m.localHashMemory).size(), goalMarkers(m, m.localHashMemoryDelta).size() };
    };
    ASSERT_TRUE((markerCounts() == std::vector<std::size_t>{ 0, 0, 0 }));

    m.admissionKeysOrdis2.push_back(rec);
    ea.admissionMemory.drainAdmissionKeysOrdis2(m);
    m.admissionKeysOrdis2.clear();
    ASSERT_EQ(m.lbMemory.goalDemandInstalls.count(), 2);
    const std::vector<std::size_t> afterFirst = markerCounts();
    for (std::size_t c : afterFirst) ASSERT_TRUE(c > 0);
    // Both marked heads are in2 with the marker in the output slot.
    for (const std::string& v : goalMarkers(m, m.overallHashMemory)) {
        ASSERT_TRUE(v.rfind("(in2[", 0) == 0);
        ASSERT_TRUE(v.find(",marker,") != std::string::npos);
    }
    // No OR demand: the expression has no entry in the OR-demand map.
    int64_t pk = 0;
    ASSERT_FALSE(gl::lookupTemplateKey(m.templateInterner, m.nameMap,
        std::string("(ordered[1,4,G])"), std::string("main"), pk)
        && m.overallHashMemory.admissionMapOrdis2.lookup(pk) != 0);

    m.admissionKeysOrdis2.push_back(rec);
    ea.admissionMemory.drainAdmissionKeysOrdis2(m);
    m.admissionKeysOrdis2.clear();
    ASSERT_EQ(m.lbMemory.goalDemandInstalls.count(), 2);
    ASSERT_TRUE(markerCounts() == afterFirst);

    pages.seal();
    pages.freePages();
}

// The near-miss server installs each marked-head implication once per scope;
// under ban_disintegration (the compressor) it serves nothing.
TEST(goal_demand, serve_installs_each_implication_once) {
    gl::tests::ProverFixture ea("FTA");
    ea.parameters.allow_multiplication = false;
    gl::Memory& m = ea.body;
    install(ea, kRowEight);
    ea.parameters.ban_disintegration = true;
    ea.integrationProcessor.serveGoalNearMiss(m, sp("(ordered[1,4,G])"), sp("main"));
    ASSERT_EQ(m.lbMemory.goalDemandInstalls.count(), 0);
    ASSERT_TRUE(goalMarkers(m, m.overallHashMemory).empty());

    ea.parameters.ban_disintegration = false;
    ea.integrationProcessor.serveGoalNearMiss(m, sp("(ordered[1,4,G])"), sp("main"));
    ASSERT_EQ(m.lbMemory.goalDemandInstalls.count(), 2);
    const std::size_t overall = goalMarkers(m, m.overallHashMemory).size();
    ASSERT_TRUE(overall > 0);
    ea.integrationProcessor.serveGoalNearMiss(m, sp("(ordered[1,4,G])"), sp("main"));
    ASSERT_EQ(m.lbMemory.goalDemandInstalls.count(), 2);
    ASSERT_EQ(goalMarkers(m, m.overallHashMemory).size(), overall);
    // Another missing expression: the implication naming its argument is
    // new; the one for the successor's output names no argument of it and
    // is already installed.
    ea.integrationProcessor.serveGoalNearMiss(m, sp("(ordered[1,4,H])"), sp("main"));
    ASSERT_EQ(m.lbMemory.goalDemandInstalls.count(), 3);
}

// A root goal's flat variant rides its carrier as IntegrationFlat, which the
// carrier reports and its removal reads back; a plain integration install
// keeps the Integration kind.
TEST(goal_demand, flat_install_kind_rides_the_carrier) {
    for (const gl::RuleInstallKind kind :
         { gl::RuleInstallKind::Integration, gl::RuleInstallKind::IntegrationFlat }) {
        gl::tests::ProverFixture analyzer("FTA");
        const gl::LogicalEntity* bez1 = analyzer.definitions.compiledEntity(gl::StrSpan("bez1", 4));
        ASSERT_TRUE(bez1 != nullptr);
        const std::vector<std::string> args = ce::getArgs(bez1->signature);
        std::vector<gl::StrSpan> owners;
        for (const std::string& token : args) owners.emplace_back(token);
        gl::Memory memory;
        const gl::StrSpan mainScope("main", 4);
        gl::ScratchArena& arena = gl::scratchArenas().forSlot(gl::scratchArenas().slotCount() - 1);
        gl::ScratchScope frame(arena);
        const gl::ScratchString compact = analyzer.integrationBuilder.fillIntegrationCompactScratch(arena,
            gl::StrSpan(bez1->integrationImplications[0].compact), owners.data(),
            static_cast<int32_t>(owners.size()));
        const std::string clean =
            gl::DefinitionExpansion::removeUPrefixFromArguments(gl::StrSpan(compact).toStdString());
        const std::string cited = "(cited)_integration_goal";
        const gl::OriginDep deps[1] = { { gl::StrSpan(cited), mainScope } };
        gl::StrSpan premises[gl::ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
        int32_t premiseN = 0;
        gl::StrSpan head;
        ASSERT_TRUE(analyzer.compactRuleProcessor.installIntegrationCompact(memory, gl::StrSpan(compact),
            mainScope, gl::OriginTag::reformulationForIntegrationFlattened, deps, 1, arena, premises,
            gl::ExecutionParameters::MAX_INSTRUCTION_ELEMENTS, premiseN, head, kind));
        const gl::CompactRuleProcessor::CarrierKinds kinds =
            gl::CompactRuleProcessor::carrierKinds(memory, gl::StrSpan(clean), mainScope);
        ASSERT_TRUE(kinds.integration && !kinds.statement);
        ASSERT_EQ(kinds.integrationFlat, kind == gl::RuleInstallKind::IntegrationFlat);
        const int64_t carrierKey = gl::packStatementKey(memory.nameMap.lookup(gl::StrSpan(clean)),
            memory.nameMap.lookup(mainScope));
        const int32_t carrierId = memory.compactExpansions.lookup(carrierKey);
        ASSERT_TRUE(carrierId != 0);
        ASSERT_TRUE(memory.compactExpansions.recordAt(carrierId, 0).kind == kind);
        {
            gl::ScratchArena& sets = gl::genScratchArenas().forSlot(0);
            gl::DirtyState dirty = gl::DirtyState::Clean;
            gl::ColdHashSet<gl::BytesKeyStore> visited(&sets, &dirty);
            gl::ColdHashSet<gl::PodKeyStore<int64_t>> drops(&sets, &dirty), removed(&sets, &dirty);
            gl::tests::CompactRuleProcessorFixture::removeCompactExpansion(analyzer.compactRuleProcessor,
                memory, carrierKey, visited, drops, removed);
            gl::CompactRuleProcessor::finishCompactRemovals(memory, drops, removed, sets);
        }
        ASSERT_EQ(memory.compactExpansions.lookup(carrierKey), 0);
    }
}
