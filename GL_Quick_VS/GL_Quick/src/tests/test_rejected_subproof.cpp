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
/// @brief Unit tests for `rejectedMapSubproof` — parked implication subproofs
///        of an and whose non-implication element is not yet known.
///
/// @details
/// Storage discipline of the map (the reused `IntegrationEntry` family:
/// canonical read-modify-write, dedup, decoded order), its scope wipe, and
/// (from the gate commit on) the park / admission behaviour.

#include "test_harness.hpp"

#include "../prover.hpp"
#include "canonical_test_helpers.hpp"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

namespace {
    /// @brief Span over a string literal (tests only).
    gl::StrSpan sp(const char* s) {
        return gl::StrSpan(s, static_cast<int32_t>(std::strlen(s)));
    }

    /// @brief Append one parked-implication record to @p wi: entity 0 the
    ///        implication (premise, head), entity 1 its parent and with one
    ///        gating element.
    void buildParkRecord(gl::WorkInstruction& wi, const char* implication,
                         const char* premise, const char* head,
                         const char* andSignature, const char* gate) {
        int32_t start = wi.elemMark();
        wi.addElement(sp(premise));
        wi.addElement(sp(head));
        wi.commitEntity(sp("implication"), sp(implication), sp(""), 2, start);
        start = wi.elemMark();
        wi.addElement(sp(gate));
        wi.commitEntity(sp("and"), sp(andSignature), sp(""), 3, start);
    }

    /// @brief Park @p wi under @p pk through the production read-modify-write.
    void park(gl::Memory& m, int64_t pk, const gl::WorkInstruction& wi,
              gl::ScratchArena& arena) {
        gl::ArenaIntegrationMap map(&arena, &m.valueInterner);
        map.buildFromCold(m.overallHashMemory.rejectedMapSubproof, pk);
        map.findOrEmplaceWork(wi, m.valueInterner);
        map.writeToCold(m.overallHashMemory.rejectedMapSubproof, pk);
    }
} // namespace

// The reused IntegrationEntry family on the new instance: a repeated park of
// the same implication is deduplicated, distinct implications are kept in
// decoded order, and the map is independent of admissionMapIntegration.
TEST(rejected_subproof, integration_entry_family_rmw_dedups_and_orders) {
    gl::Memory m;
    gl::ScratchArena& arena = gl::genScratchArenas().forSlot(
        gl::genScratchArenas().slotCount() - 1);
    gl::ScratchScope scope(arena);
    const int64_t pk = gl::mintTemplateKey(m.templateInterner, m.nameMap,
        "(in2[10,14,g])", "main");

    gl::WorkInstruction step(&arena);
    buildParkRecord(step, "(implication33[u_I,u_3,u_g,u_11,u_5])",
        "(in[i,u_I])", "(in2[j,q,u_g])",
        "(and0[u_2,u_11,u_g,u_10,u_14,u_I,u_3,u_5])", "(in2[u_10,u_14,u_g])");
    gl::WorkInstruction start(&arena);
    buildParkRecord(start, "(implication32[u_2,u_11,u_g])",
        "(in2[u_2,q,u_11])", "(in2[u_2,q,u_g])",
        "(and0[u_2,u_11,u_g,u_10,u_14,u_I,u_3,u_5])", "(in2[u_10,u_14,u_g])");

    park(m, pk, step, arena);
    park(m, pk, start, arena);
    park(m, pk, step, arena);   // repeated park — deduplicated by the RMW

    const gl::IntegrationEntryMap records = gl::admissionIntegrationRecordsAt(
        m.overallHashMemory.rejectedMapSubproof, pk, m.valueInterner);
    ASSERT_EQ(records.size(), static_cast<std::size_t>(2));
    // Decoded order: implication32 sorts before implication33.
    const gl::IntInstruction& first = records.begin()->first;
    ASSERT_EQ(first.data.size(), static_cast<std::size_t>(2));
    ASSERT_EQ(m.valueInterner.decode(first.data[0].signature),
              std::string("(implication32[u_2,u_11,u_g])"));
    ASSERT_EQ(m.valueInterner.decode(first.data[1].category), std::string("and"));
    ASSERT_TRUE(records.begin()->second.empty());
    // Independence: the integration admission map never sees the park.
    ASSERT_TRUE(m.overallHashMemory.admissionMapIntegration.lookup(pk) == 0);
}

// Scope wipe through Memory::wipeSubtrees: the park filed at the closed child
// scope is erased with its guard entry; the park at main survives.
TEST(rejected_subproof, scope_wipe_erases_closed_validity_only) {
    gl::Memory m;
    gl::ScratchArena& arena = gl::genScratchArenas().forSlot(
        gl::genScratchArenas().slotCount() - 1);
    gl::ScratchScope scope(arena);
    const gl::NameId mainId = m.nameMap.encode(sp("main"));
    const gl::NameId childId = m.nameMap.encodePush(mainId,
        sp("(implication33[I,3,f,11,5])"));
    const std::string childName = m.nameMap.decode(childId);

    const int64_t pkMain = gl::mintTemplateKey(m.templateInterner, m.nameMap,
        "(in2[10,14,g])", "main");
    const int64_t pkChild = gl::mintTemplateKey(m.templateInterner, m.nameMap,
        "(in2[10,14,g])", childName);
    ASSERT_EQ(gl::Codec<gl::StatementKey>::decode(pkChild).validity, childId);

    gl::WorkInstruction record(&arena);
    buildParkRecord(record, "(implication33[u_I,u_3,u_g,u_11,u_5])",
        "(in[i,u_I])", "(in2[j,q,u_g])",
        "(and0[u_2,u_11,u_g,u_10,u_14,u_I,u_3,u_5])", "(in2[u_10,u_14,u_g])");
    park(m, pkMain, record, arena);
    park(m, pkChild, record, arena);
    m.overallHashMemory.subproofRevisitInProgress.mint(pkChild);
    m.overallHashMemory.subproofRevisitInProgress.mint(pkMain);

    m.wipeSubtrees(&childId, 1);

    ASSERT_TRUE(m.overallHashMemory.rejectedMapSubproof.lookup(pkChild) == 0);
    ASSERT_TRUE(m.overallHashMemory.rejectedMapSubproof.lookup(pkMain) != 0);
    ASSERT_FALSE(m.overallHashMemory.subproofRevisitInProgress.contains(pkChild));
    ASSERT_TRUE(m.overallHashMemory.subproofRevisitInProgress.contains(pkMain));
}

namespace {
    /// @brief One implication entity `(implication90[u_1,u_2])` with premise
    ///        `(=[u_1,X])` and head `(=[u_2,X])`.
    void buildImplication90(gl::WorkInstruction& wi) {
        const int32_t mark = wi.elemMark();
        wi.addElement(sp("(=[u_1,X])"));
        wi.addElement(sp("(=[u_2,X])"));
        wi.commitEntity(sp("implication"), sp("(implication90[u_1,u_2])"), sp(""), 2, mark);
    }

    /// @brief Count the main-level subproof scopes of @p goal whose bare
    ///        payload is @p bare.
    int32_t countSubproofScopes(gl::ExpressionAnalyzer& analyzer, gl::Memory& memory,
                                const std::string& goal, const std::string& bare) {
        int32_t n = 0;
        for (gl::NameId id = 2; id <= memory.nameMap.nameCount(); ++id) {
            if (static_cast<std::size_t>(id)
                    >= static_cast<std::size_t>(memory.nameMap.stackSize())
                || memory.nameMap.stackEmpty(id))
                continue;
            gl::StrSpan goalOut, bareOut;
            if (!analyzer.splitSubproofPayload(
                    memory.nameMap.decodeSubView(memory.nameMap.stackBack(id)),
                    goalOut, bareOut))
                continue;
            if (goalOut.toStdString() == goal && bareOut.toStdString() == bare) ++n;
        }
        return n;
    }
} // namespace

// openImplicationSubproof (Case A of prepareIntegrationCore2): the first call
// opens the goal-carrying boundary scope and mints its prepared key; a repeat
// hits the prepared gate and opens nothing new.
TEST(rejected_subproof, open_implication_subproof_opens_then_prepared) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    gl::ExpressionAnalyzer analyzer("Peano");
    gl::Memory memory;
    memory.setVarDefSets(gl::StrSpan("1:(1);2:(1);3:(1)", 17));
    const std::string goal = "(interval[1,4,2,7,10])";
    const std::string bare = "(implication90[1,2])";

    gl::WorkInstruction wi(&lb);
    buildImplication90(wi);
    ASSERT_TRUE(analyzer.openImplicationSubproof(wi, 0, memory, sp("main"),
                    gl::StrSpan(goal))
                == gl::ExpressionAnalyzer::ImplicationSubproofOutcome::Opened);
    ASSERT_EQ(countSubproofScopes(analyzer, memory, goal, bare), 1);
    ASSERT_TRUE(memory.integrationPrepared.contains(gl::mintTemplateKey(
        memory.templateInterner, memory.nameMap, goal + "_subproof_" + bare,
        std::string("main"))));

    ASSERT_TRUE(analyzer.openImplicationSubproof(wi, 0, memory, sp("main"),
                    gl::StrSpan(goal))
                == gl::ExpressionAnalyzer::ImplicationSubproofOutcome::Prepared);
    ASSERT_EQ(countSubproofScopes(analyzer, memory, goal, bare), 1);
}

// openImplicationSubproof: a clean compact already known at the scope needs no
// subproof — no scope, no prepared key.
TEST(rejected_subproof, open_implication_subproof_known_compact_opens_nothing) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    gl::ExpressionAnalyzer analyzer("Peano");
    gl::Memory memory;
    memory.setVarDefSets(gl::StrSpan("1:(1);2:(1);3:(1)", 17));
    const std::string goal = "(interval[1,4,2,7,10])";
    const std::string bare = "(implication90[1,2])";
    gl::upsertStatementKey(memory.intKnownStatements,
        gl::packStatementKey(memory.nameMap.encode(bare), gl::NameMap::MAIN_ID),
        /*local=*/true, /*fullyDisintegrated=*/false);

    gl::WorkInstruction wi(&lb);
    buildImplication90(wi);
    ASSERT_TRUE(analyzer.openImplicationSubproof(wi, 0, memory, sp("main"),
                    gl::StrSpan(goal))
                == gl::ExpressionAnalyzer::ImplicationSubproofOutcome::Known);
    ASSERT_EQ(countSubproofScopes(analyzer, memory, goal, bare), 0);
    ASSERT_EQ(memory.integrationPrepared.count(), 0);
}

namespace {
    /// @brief implication90 plus its parent and `(and90[u_1,u_2,u_3])`, whose
    ///        gate elements are `(in2[u_1,u_2,u_3])` and optionally @p gate2.
    void buildGatedImplication90(gl::WorkInstruction& wi, const char* gate2 = nullptr) {
        buildImplication90(wi);
        const int32_t mark = wi.elemMark();
        wi.addElement(sp("(implication90[u_1,u_2])"));
        wi.addElement(sp("(in2[u_1,u_2,u_3])"));
        if (gate2 != nullptr) wi.addElement(sp(gate2));
        wi.commitEntity(sp("and"), sp("(and90[u_1,u_2,u_3])"), sp(""), 3, mark);
    }

    /// @brief Make @p text a known row at @p scope (the registry write only).
    int64_t makeKnown(gl::Memory& memory, const char* text, gl::NameId scope) {
        const int64_t key = gl::packStatementKey(memory.nameMap.encode(sp(text)), scope);
        gl::upsertStatementKey(memory.intKnownStatements, key, /*local=*/true);
        return key;
    }

    /// @brief The packed park key of clean gate @p gate at main.
    int64_t mainParkKey(gl::Memory& memory, const char* gate) {
        return gl::mintTemplateKey(memory.templateInterner, memory.nameMap,
                                   std::string(gate), std::string("main"));
    }

    /// @brief Register @p goal as an open goal at main.
    void addMainGoal(gl::Memory& memory, const std::string& goal) {
        memory.intToBeProved.assignSet(gl::StatementKey{
            memory.nameMap.encode(gl::StrSpan(goal)), gl::NameMap::MAIN_ID }, nullptr, 0);
    }

    /// @brief Fresh Peano analyzer fixture state for the gate tests.
    struct GateFixture {
        gl::GlobalMemoryManager g;
        gl::LbArena* lb = nullptr;
        gl::ExpressionAnalyzer analyzer{ "Peano" };
        gl::Memory memory;
        const std::string goal = "(interval[1,4,2,7,10])";
        const std::string bare = "(implication90[1,2])";
        GateFixture() {
            g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
            lb = new gl::LbArena(&g);
            memory.setVarDefSets(gl::StrSpan("1:(1);2:(1);3:(1)", 17));
        }
        ~GateFixture() { delete lb; }
    };
} // namespace

// isSubproofGateElement / isParentAndOf: only a non-implication element with
// every argument fixed gates; the parent relation is element == signature.
TEST(rejected_subproof, gate_element_classification_and_parent_relation) {
    using EA = gl::ExpressionAnalyzer;
    ASSERT_TRUE(EA::isSubproofGateElement(sp("(in2[u_10,u_14,u_g])")));
    ASSERT_TRUE(EA::isSubproofGateElement(sp("!(in2[u_10,u_14,u_g])")));
    ASSERT_TRUE(EA::isSubproofGateElement(sp("(existence5[u_1,u_2])")));
    ASSERT_FALSE(EA::isSubproofGateElement(sp("(implication32[u_2,u_11,u_g])")));
    ASSERT_FALSE(EA::isSubproofGateElement(sp("(in2[i,u_14,u_g])")));
    ASSERT_FALSE(EA::isSubproofGateElement(sp("(in2[marker,u_14,u_g])")));
    ASSERT_FALSE(EA::isSubproofGateElement(sp("(in2[pi_1,u_14,u_g])")));

    GateFixture f;
    gl::WorkInstruction wi(f.lb);
    buildGatedImplication90(wi);
    ASSERT_FALSE(EA::isParentAndOf(wi, 0, sp("(implication90[u_1,u_2])")));
    ASSERT_TRUE(EA::isParentAndOf(wi, 1, sp("(implication90[u_1,u_2])")));
    ASSERT_FALSE(EA::isParentAndOf(wi, 1, sp("(implication91[u_1,u_2])")));
}

// elementKnownAtOrAbove: reflexive equality, a literal row at an ancestor, a
// class variant at the scope, and misses (descendant-only row, unrelated row).
TEST(rejected_subproof, element_known_at_or_above_probe) {
    gl::ExpressionAnalyzer analyzer("Peano");
    gl::Memory m;
    const gl::NameId child = m.nameMap.encodePush(gl::NameMap::MAIN_ID, sp("(implication7[a])"));
    ASSERT_TRUE(analyzer.elementKnownAtOrAbove(sp("(=[q,q])"), child, m));
    ASSERT_FALSE(analyzer.elementKnownAtOrAbove(sp("(=[q,r])"), child, m));

    makeKnown(m, "(in2[c,7,3])", gl::NameMap::MAIN_ID);
    ASSERT_TRUE(analyzer.elementKnownAtOrAbove(sp("(in2[c,7,3])"), child, m));
    ASSERT_TRUE(analyzer.elementKnownAtOrAbove(sp("(in2[c,7,3])"), gl::NameMap::MAIN_ID, m));
    makeKnown(m, "(in2[d,7,3])", child);
    ASSERT_FALSE(analyzer.elementKnownAtOrAbove(sp("(in2[d,7,3])"), gl::NameMap::MAIN_ID, m));

    gl::EquivalenceClass cls;
    cls.setMembersFromNames({ "a", "b" }, m.nameMap);
    m.assignClassesById(gl::NameMap::MAIN_ID, { cls });
    seedFixtureCanonicalClasses(m);
    makeKnown(m, "(in2[b,7,3])", gl::NameMap::MAIN_ID);
    ASSERT_TRUE(analyzer.elementKnownAtOrAbove(sp("(in2[a,7,3])"), gl::NameMap::MAIN_ID, m));
    ASSERT_FALSE(analyzer.elementKnownAtOrAbove(sp("(in2[e,7,3])"), gl::NameMap::MAIN_ID, m));
}

// orDisjunctTrueAt (D-438): a reflexive equality or a
// known row at the scope or an ancestor holds; a positive conjunction holds
// when every flattened element holds, nested conjunctions included; one open
// element, a negated conjunction without its own row, or a row known only in
// a descendant scope does not hold.
TEST(rejected_subproof, or_disjunct_true_at_probe) {
    gl::ExpressionAnalyzer analyzer("Peano");
    analyzer.compiledExpressions["and9001"] = gl::LogicalEntity("and",
        { "(=[u_1,u_2])", "(=[u_3,u_4])" }, "(and9001[u_1,u_2,u_3,u_4])", 4);
    analyzer.compiledExpressions["and9002"] = gl::LogicalEntity("and",
        { "(in2[u_1,u_2,u_3])", "(and9001[u_2,u_2,u_1,u_1])" }, "(and9002[u_1,u_2,u_3])", 3);
    gl::Memory m;
    const gl::NameId child = m.nameMap.encodePush(gl::NameMap::MAIN_ID, sp("(implication7[a])"));

    ASSERT_TRUE(analyzer.orDisjunctTrueAt(sp("(=[q,q])"), child, m));
    ASSERT_FALSE(analyzer.orDisjunctTrueAt(sp("(=[q,r])"), child, m));
    ASSERT_TRUE(analyzer.orDisjunctTrueAt(sp("(and9001[q,q,r,r])"), child, m));
    ASSERT_FALSE(analyzer.orDisjunctTrueAt(sp("(and9001[q,q,r,s])"), child, m));
    ASSERT_FALSE(analyzer.orDisjunctTrueAt(sp("!(and9001[q,q,r,r])"), child, m));

    makeKnown(m, "(in2[c,7,3])", gl::NameMap::MAIN_ID);
    ASSERT_TRUE(analyzer.orDisjunctTrueAt(sp("(in2[c,7,3])"), child, m));
    ASSERT_TRUE(analyzer.orDisjunctTrueAt(sp("(and9002[c,7,3])"), child, m));
    ASSERT_FALSE(analyzer.orDisjunctTrueAt(sp("(and9002[d,7,3])"), child, m));

    makeKnown(m, "(=[r,s])", gl::NameMap::MAIN_ID);
    ASSERT_TRUE(analyzer.orDisjunctTrueAt(sp("(and9001[q,q,r,s])"), child, m));

    makeKnown(m, "(in2[e,7,3])", child);
    ASSERT_FALSE(analyzer.orDisjunctTrueAt(sp("(in2[e,7,3])"), gl::NameMap::MAIN_ID, m));
    ASSERT_TRUE(analyzer.orDisjunctTrueAt(sp("(in2[e,7,3])"), child, m));

    makeKnown(m, "!(in2[d,7,3])", gl::NameMap::MAIN_ID);
    ASSERT_TRUE(analyzer.orDisjunctTrueAt(sp("!(in2[d,7,3])"), child, m));
}

// openImplicationSubproof gate: an unknown gate element parks the implication
// (record = implication + parent and + root goal) and opens nothing; a known
// gate element lets it open.
TEST(rejected_subproof, open_parks_under_unknown_gate_and_opens_when_known) {
    GateFixture f;
    gl::WorkInstruction wi(f.lb);
    buildGatedImplication90(wi);
    ASSERT_TRUE(f.analyzer.openImplicationSubproof(wi, 0, f.memory, sp("main"), gl::StrSpan(f.goal))
                == gl::ExpressionAnalyzer::ImplicationSubproofOutcome::Parked);
    ASSERT_EQ(countSubproofScopes(f.analyzer, f.memory, f.goal, f.bare), 0);
    ASSERT_EQ(f.memory.integrationPrepared.count(), 0);
    const int64_t pk = mainParkKey(f.memory, "(in2[1,2,3])");
    const gl::IntegrationEntryMap records = gl::admissionIntegrationRecordsAt(
        f.memory.overallHashMemory.rejectedMapSubproof, pk, f.memory.valueInterner);
    ASSERT_EQ(records.size(), static_cast<std::size_t>(1));
    const gl::IntInstruction& record = records.begin()->first;
    ASSERT_EQ(record.data.size(), static_cast<std::size_t>(2));
    ASSERT_EQ(f.memory.valueInterner.decode(record.data[0].signature),
              std::string("(implication90[u_1,u_2])"));
    ASSERT_EQ(f.memory.valueInterner.decode(record.data[1].signature),
              std::string("(and90[u_1,u_2,u_3])"));
    ASSERT_EQ(f.memory.valueInterner.decode(record.markedGoal), f.goal);

    makeKnown(f.memory, "(in2[1,2,3])", gl::NameMap::MAIN_ID);
    ASSERT_TRUE(f.analyzer.openImplicationSubproof(wi, 0, f.memory, sp("main"), gl::StrSpan(f.goal))
                == gl::ExpressionAnalyzer::ImplicationSubproofOutcome::Opened);
    ASSERT_EQ(countSubproofScopes(f.analyzer, f.memory, f.goal, f.bare), 1);
}

// parkImplicationSubproof: the key is canonical at the park scope, so
// class-equal gates share one key, while each record keeps the instruction's
// own text (the text the and's integration rule was installed with).
TEST(rejected_subproof, park_key_is_canonical_record_keeps_instruction_text) {
    GateFixture f;
    gl::EquivalenceClass cls;
    cls.setMembersFromNames({ "a", "b" }, f.memory.nameMap);
    f.memory.assignClassesById(gl::NameMap::MAIN_ID, { cls });
    seedFixtureCanonicalClasses(f.memory);
    const auto buildWith = [&](gl::WorkInstruction& wi, const char* gate) {
        buildImplication90(wi);
        const int32_t mark = wi.elemMark();
        wi.addElement(sp("(implication90[u_1,u_2])"));
        wi.addElement(sp(gate));
        wi.commitEntity(sp("and"), sp("(and90[u_1,u_2,u_3])"), sp(""), 3, mark);
    };
    gl::WorkInstruction viaA(f.lb), viaB(f.lb);
    buildWith(viaA, "(in2[u_a,u_2,u_3])");
    buildWith(viaB, "(in2[u_b,u_2,u_3])");
    f.analyzer.parkImplicationSubproof(viaA, 0, sp("(in2[a,2,3])"), f.memory, sp("main"), gl::StrSpan(f.goal));
    f.analyzer.parkImplicationSubproof(viaB, 0, sp("(in2[b,2,3])"), f.memory, sp("main"), gl::StrSpan(f.goal));

    ASSERT_EQ(f.memory.overallHashMemory.rejectedMapSubproof.count(), 1);
    gl::ScratchArena& strings = gl::scratchArenas().forSlot(gl::scratchArenas().slotCount() - 1);
    gl::ScratchScope frame(strings);
    const gl::ScratchString canonicalA = f.analyzer.canonicalAdmissionText(
        f.memory, sp("(in2[a,2,3])"), gl::NameMap::MAIN_ID, strings, true);
    const gl::ScratchString canonicalB = f.analyzer.canonicalAdmissionText(
        f.memory, sp("(in2[b,2,3])"), gl::NameMap::MAIN_ID, strings, true);
    ASSERT_TRUE(gl::equalSpans(gl::StrSpan(canonicalA), gl::StrSpan(canonicalB)));
    const int64_t pk = gl::mintTemplateKey(f.memory.templateInterner, f.memory.nameMap,
        gl::StrSpan(canonicalA), sp("main"));
    const gl::IntegrationEntryMap records = gl::admissionIntegrationRecordsAt(
        f.memory.overallHashMemory.rejectedMapSubproof, pk, f.memory.valueInterner);
    ASSERT_EQ(records.size(), static_cast<std::size_t>(2));
    std::vector<std::string> gates;
    for (const auto& entry : records)
        gates.push_back(f.memory.valueInterner.decode(entry.first.data[1].elements[1]));
    std::sort(gates.begin(), gates.end());
    ASSERT_EQ(gates[0], std::string("(in2[u_a,u_2,u_3])"));
    ASSERT_EQ(gates[1], std::string("(in2[u_b,u_2,u_3])"));
}

// stageSubproofArrival: nothing is staged while no implication is parked.
TEST(rejected_subproof, stage_arrival_only_while_parks_exist) {
    GateFixture f;
    const int64_t key = makeKnown(f.memory, "(in[1,3])", gl::NameMap::MAIN_ID);
    f.analyzer.stageSubproofArrival(f.memory, key);
    ASSERT_EQ(f.memory.pendingSubproofArrivals.size(), 0);
    gl::WorkInstruction wi(f.lb);
    buildGatedImplication90(wi);
    (void)f.analyzer.openImplicationSubproof(wi, 0, f.memory, sp("main"), gl::StrSpan(f.goal));
    f.analyzer.stageSubproofArrival(f.memory, key);
    ASSERT_EQ(f.memory.pendingSubproofArrivals.size(), 1);
}

// drainSubproofArrivals: an arrival only at a descendant scope leaves the park
// asleep; the gate arriving at main reopens it (key erased, subproof opened,
// queue consumed, guard released).
TEST(rejected_subproof, drain_reopens_on_gate_arrival_at_or_above_the_park) {
    GateFixture f;
    addMainGoal(f.memory, f.goal);
    gl::WorkInstruction wi(f.lb);
    buildGatedImplication90(wi);
    (void)f.analyzer.openImplicationSubproof(wi, 0, f.memory, sp("main"), gl::StrSpan(f.goal));
    const int64_t pk = mainParkKey(f.memory, "(in2[1,2,3])");
    ASSERT_TRUE(f.memory.overallHashMemory.rejectedMapSubproof.lookup(pk) != 0);

    const gl::NameId child = f.memory.nameMap.encodePush(gl::NameMap::MAIN_ID, sp("(implication7[a])"));
    f.analyzer.stageSubproofArrival(f.memory, makeKnown(f.memory, "(in2[1,2,3])", child));
    f.analyzer.drainSubproofArrivals(f.memory);
    ASSERT_EQ(f.memory.pendingSubproofArrivals.size(), 0);
    ASSERT_TRUE(f.memory.overallHashMemory.rejectedMapSubproof.lookup(pk) != 0);
    ASSERT_EQ(countSubproofScopes(f.analyzer, f.memory, f.goal, f.bare), 0);

    f.analyzer.stageSubproofArrival(f.memory, makeKnown(f.memory, "(in2[1,2,3])", gl::NameMap::MAIN_ID));
    f.analyzer.drainSubproofArrivals(f.memory);
    ASSERT_EQ(f.memory.pendingSubproofArrivals.size(), 0);
    ASSERT_TRUE(f.memory.overallHashMemory.rejectedMapSubproof.lookup(pk) == 0);
    ASSERT_EQ(countSubproofScopes(f.analyzer, f.memory, f.goal, f.bare), 1);
    ASSERT_FALSE(f.memory.overallHashMemory.subproofRevisitInProgress.contains(pk));
}

// revisitRejectedSubproof drops: a record whose root goal is no longer open,
// and a record whose parent and is already known, open nothing.
TEST(rejected_subproof, revisit_drops_gone_goal_and_known_parent_and) {
    {
        GateFixture f;
        gl::WorkInstruction wi(f.lb);
        buildGatedImplication90(wi);
        (void)f.analyzer.openImplicationSubproof(wi, 0, f.memory, sp("main"), gl::StrSpan(f.goal));
        const int64_t pk = mainParkKey(f.memory, "(in2[1,2,3])");
        makeKnown(f.memory, "(in2[1,2,3])", gl::NameMap::MAIN_ID);
        f.analyzer.revisitRejectedSubproof(pk, f.memory);
        ASSERT_TRUE(f.memory.overallHashMemory.rejectedMapSubproof.lookup(pk) == 0);
        ASSERT_EQ(countSubproofScopes(f.analyzer, f.memory, f.goal, f.bare), 0);
    }
    {
        GateFixture f;
        addMainGoal(f.memory, f.goal);
        gl::WorkInstruction wi(f.lb);
        buildGatedImplication90(wi);
        (void)f.analyzer.openImplicationSubproof(wi, 0, f.memory, sp("main"), gl::StrSpan(f.goal));
        const int64_t pk = mainParkKey(f.memory, "(in2[1,2,3])");
        makeKnown(f.memory, "(in2[1,2,3])", gl::NameMap::MAIN_ID);
        makeKnown(f.memory, "(and90[1,2,3])", gl::NameMap::MAIN_ID);
        f.analyzer.revisitRejectedSubproof(pk, f.memory);
        ASSERT_TRUE(f.memory.overallHashMemory.rejectedMapSubproof.lookup(pk) == 0);
        ASSERT_EQ(countSubproofScopes(f.analyzer, f.memory, f.goal, f.bare), 0);
    }
}

// revisitRejectedSubproof: a gate that is still unknown keeps the park; once
// known, the record re-parks under the and's next unknown gate element.
TEST(rejected_subproof, revisit_keeps_unknown_gate_and_reparks_under_next_gate) {
    GateFixture f;
    addMainGoal(f.memory, f.goal);
    gl::WorkInstruction wi(f.lb);
    buildGatedImplication90(wi, "(in[u_1,u_3])");
    (void)f.analyzer.openImplicationSubproof(wi, 0, f.memory, sp("main"), gl::StrSpan(f.goal));
    const int64_t first = mainParkKey(f.memory, "(in2[1,2,3])");
    f.analyzer.revisitRejectedSubproof(first, f.memory);
    ASSERT_TRUE(f.memory.overallHashMemory.rejectedMapSubproof.lookup(first) != 0);

    makeKnown(f.memory, "(in2[1,2,3])", gl::NameMap::MAIN_ID);
    f.analyzer.revisitRejectedSubproof(first, f.memory);
    const int64_t second = mainParkKey(f.memory, "(in[1,3])");
    ASSERT_TRUE(f.memory.overallHashMemory.rejectedMapSubproof.lookup(first) == 0);
    ASSERT_TRUE(f.memory.overallHashMemory.rejectedMapSubproof.lookup(second) != 0);
    ASSERT_EQ(countSubproofScopes(f.analyzer, f.memory, f.goal, f.bare), 0);
}

namespace {
    /// @brief implication90 plus its parent and whose only gate element is
    ///        @p gate.
    void buildImplication90WithGate(gl::WorkInstruction& wi, const char* gate) {
        buildImplication90(wi);
        const int32_t mark = wi.elemMark();
        wi.addElement(sp("(implication90[u_1,u_2])"));
        wi.addElement(sp(gate));
        wi.commitEntity(sp("and"), sp("(and90[u_1,u_2,u_3])"), sp(""), 3, mark);
    }

    /// @brief Give main the class {a, b} through the production class writer.
    void mergeAB(gl::Memory& memory) {
        gl::EquivalenceClass cls;
        cls.setMembersFromNames({ "a", "b" }, memory.nameMap);
        memory.assignClassesById(gl::NameMap::MAIN_ID, { cls });
        seedFixtureCanonicalClasses(memory);
    }
} // namespace

// stageSubproofRecheck: nothing is staged while no implication is parked.
TEST(rejected_subproof, stage_recheck_only_while_parks_exist) {
    GateFixture f;
    f.analyzer.stageSubproofRecheck(f.memory, gl::NameMap::MAIN_ID);
    ASSERT_EQ(f.memory.pendingSubproofRechecks.size(), 0);
    gl::WorkInstruction wi(f.lb);
    buildGatedImplication90(wi);
    (void)f.analyzer.openImplicationSubproof(wi, 0, f.memory, sp("main"), gl::StrSpan(f.goal));
    f.analyzer.stageSubproofRecheck(f.memory, gl::NameMap::MAIN_ID);
    ASSERT_EQ(f.memory.pendingSubproofRechecks.size(), 1);
}

// rekeySubproofPark: after a class merge the two spellings' parks end under
// one canonical key with both records verbatim; the class writer staged the
// recheck.
TEST(rejected_subproof, rekey_moves_records_verbatim_to_canonical_key) {
    GateFixture f;
    gl::WorkInstruction viaA(f.lb), viaB(f.lb);
    buildImplication90WithGate(viaA, "(in2[u_a,u_2,u_3])");
    buildImplication90WithGate(viaB, "(in2[u_b,u_2,u_3])");
    (void)f.analyzer.openImplicationSubproof(viaA, 0, f.memory, sp("main"), gl::StrSpan(f.goal));
    (void)f.analyzer.openImplicationSubproof(viaB, 0, f.memory, sp("main"), gl::StrSpan(f.goal));
    const int64_t keyA = mainParkKey(f.memory, "(in2[a,2,3])");
    const int64_t keyB = mainParkKey(f.memory, "(in2[b,2,3])");
    ASSERT_TRUE(f.memory.overallHashMemory.rejectedMapSubproof.lookup(keyA) != 0);
    ASSERT_TRUE(f.memory.overallHashMemory.rejectedMapSubproof.lookup(keyB) != 0);

    mergeAB(f.memory);
    ASSERT_TRUE(f.memory.pendingSubproofRechecks.size() > 0);
    const int64_t canonicalA = f.analyzer.rekeySubproofPark(keyA, f.memory);
    const int64_t canonicalB = f.analyzer.rekeySubproofPark(keyB, f.memory);
    ASSERT_EQ(canonicalA, canonicalB);
    ASSERT_TRUE((canonicalA == keyA) != (canonicalA == keyB));
    const int64_t moved = canonicalA == keyA ? keyB : keyA;
    ASSERT_TRUE(f.memory.overallHashMemory.rejectedMapSubproof.lookup(moved) == 0);
    const gl::IntegrationEntryMap records = gl::admissionIntegrationRecordsAt(
        f.memory.overallHashMemory.rejectedMapSubproof, canonicalA, f.memory.valueInterner);
    ASSERT_EQ(records.size(), static_cast<std::size_t>(2));
    std::vector<std::string> gates;
    for (const auto& entry : records)
        gates.push_back(f.memory.valueInterner.decode(entry.first.data[1].elements[1]));
    std::sort(gates.begin(), gates.end());
    ASSERT_EQ(gates[0], std::string("(in2[u_a,u_2,u_3])"));
    ASSERT_EQ(gates[1], std::string("(in2[u_b,u_2,u_3])"));
}

// drainSubproofArrivals, class side: a known row over another class member
// makes the gate known only once the classes merge; the class writer's staged
// recheck reopens the park without any new row.
TEST(rejected_subproof, drain_reopens_when_a_class_merge_makes_the_gate_known) {
    GateFixture f;
    addMainGoal(f.memory, f.goal);
    gl::WorkInstruction wi(f.lb);
    buildImplication90WithGate(wi, "(in2[u_a,u_2,u_3])");
    ASSERT_TRUE(f.analyzer.openImplicationSubproof(wi, 0, f.memory, sp("main"), gl::StrSpan(f.goal))
                == gl::ExpressionAnalyzer::ImplicationSubproofOutcome::Parked);
    makeKnown(f.memory, "(in2[b,2,3])", gl::NameMap::MAIN_ID);
    f.analyzer.drainSubproofArrivals(f.memory);
    ASSERT_EQ(countSubproofScopes(f.analyzer, f.memory, f.goal, f.bare), 0);

    mergeAB(f.memory);
    f.analyzer.drainSubproofArrivals(f.memory);
    ASSERT_EQ(f.memory.pendingSubproofRechecks.size(), 0);
    ASSERT_EQ(countSubproofScopes(f.analyzer, f.memory, f.goal, f.bare), 1);
    ASSERT_TRUE(f.memory.overallHashMemory.rejectedMapSubproof.lookup(
        mainParkKey(f.memory, "(in2[a,2,3])")) == 0);
    ASSERT_TRUE(f.memory.overallHashMemory.rejectedMapSubproof.lookup(
        mainParkKey(f.memory, "(in2[b,2,3])")) == 0);
}

// eraseGoalSubproofParks: a disproved goal loses exactly its own records; a
// key left empty is erased and another goal's record stays.
TEST(rejected_subproof, disproved_goal_erases_only_its_parks) {
    GateFixture f;
    const std::string other = "(interval[1,4,2,6,10])";
    gl::WorkInstruction shared(f.lb), own(f.lb);
    buildGatedImplication90(shared);
    buildImplication90WithGate(own, "(in[u_1,u_3])");
    (void)f.analyzer.openImplicationSubproof(shared, 0, f.memory, sp("main"), gl::StrSpan(f.goal));
    (void)f.analyzer.openImplicationSubproof(shared, 0, f.memory, sp("main"), gl::StrSpan(other));
    (void)f.analyzer.openImplicationSubproof(own, 0, f.memory, sp("main"), gl::StrSpan(f.goal));
    const int64_t sharedKey = mainParkKey(f.memory, "(in2[1,2,3])");
    const int64_t ownKey = mainParkKey(f.memory, "(in[1,3])");
    ASSERT_EQ(gl::admissionIntegrationRecordsAt(f.memory.overallHashMemory.rejectedMapSubproof,
        sharedKey, f.memory.valueInterner).size(), static_cast<std::size_t>(2));

    f.analyzer.eraseGoalSubproofParks(f.memory, gl::StrSpan(f.goal));
    ASSERT_TRUE(f.memory.overallHashMemory.rejectedMapSubproof.lookup(ownKey) == 0);
    const gl::IntegrationEntryMap left = gl::admissionIntegrationRecordsAt(
        f.memory.overallHashMemory.rejectedMapSubproof, sharedKey, f.memory.valueInterner);
    ASSERT_EQ(left.size(), static_cast<std::size_t>(1));
    ASSERT_EQ(f.memory.valueInterner.decode(left.begin()->first.markedGoal), other);
}
