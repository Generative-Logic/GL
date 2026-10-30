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

#include "../logical_block/statement_arrivals.hpp"
#include "../logical_block/equivalence_queries.hpp"
#include "../memory.hpp"
#include <cstddef>
#include <utility>
#include "test_harness.hpp"
#include "prover_fixture.hpp"
#include "case_processor_fixture.hpp"
#include "implication_subproofs_fixture.hpp"

#include "compiled_definition_fixture.hpp"

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
        gl::ArenaIntegrationMap map(&arena, &((m).valueInterner));
        map.buildFromCold(((((m).overallHashMemory)).rejectedMapSubproof), pk);
        map.findOrEmplaceWork(wi, ((m).valueInterner));
        map.writeToCold(((((m).overallHashMemory)).rejectedMapSubproof), pk);
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
    const int64_t pk = gl::mintTemplateKey(((m).templateInterner), ((m).nameMap),
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
        ((((m).overallHashMemory)).rejectedMapSubproof), pk, ((m).valueInterner));
    ASSERT_EQ(records.size(), static_cast<std::size_t>(2));
    // Decoded order: implication32 sorts before implication33.
    const gl::IntInstruction& first = records.begin()->first;
    ASSERT_EQ(first.data.size(), static_cast<std::size_t>(2));
    ASSERT_EQ(((m).valueInterner).decode(first.data[0].signature),
              std::string("(implication32[u_2,u_11,u_g])"));
    ASSERT_EQ(((m).valueInterner).decode(first.data[1].category), std::string("and"));
    ASSERT_TRUE(records.begin()->second.empty());
    // Independence: the integration admission map never sees the park.
    ASSERT_TRUE(((((m).overallHashMemory)).admissionMapIntegration).lookup(pk) == 0);
}

// Scope wipe through Memory::wipeSubtrees: the park filed at the closed child
// scope is erased with its guard entry; the park at main survives.
TEST(rejected_subproof, scope_wipe_erases_closed_validity_only) {
    gl::Memory m;
    gl::ScratchArena& arena = gl::genScratchArenas().forSlot(
        gl::genScratchArenas().slotCount() - 1);
    gl::ScratchScope scope(arena);
    const gl::NameId mainId = ((m).nameMap).encode(sp("main"));
    const gl::NameId childId = ((m).nameMap).encodePush(mainId,
        sp("(implication33[I,3,f,11,5])"));
    const std::string childName = ((m).nameMap).decode(childId);

    const int64_t pkMain = gl::mintTemplateKey(((m).templateInterner), ((m).nameMap),
        "(in2[10,14,g])", "main");
    const int64_t pkChild = gl::mintTemplateKey(((m).templateInterner), ((m).nameMap),
        "(in2[10,14,g])", childName);
    ASSERT_EQ(gl::Codec<gl::StatementKey>::decode(pkChild).validity, childId);

    gl::WorkInstruction record(&arena);
    buildParkRecord(record, "(implication33[u_I,u_3,u_g,u_11,u_5])",
        "(in[i,u_I])", "(in2[j,q,u_g])",
        "(and0[u_2,u_11,u_g,u_10,u_14,u_I,u_3,u_5])", "(in2[u_10,u_14,u_g])");
    park(m, pkMain, record, arena);
    park(m, pkChild, record, arena);
    ((((m).overallHashMemory)).subproofRevisitInProgress).mint(pkChild);
    ((((m).overallHashMemory)).subproofRevisitInProgress).mint(pkMain);

    m.wipeSubtrees(&childId, 1);

    ASSERT_TRUE(((((m).overallHashMemory)).rejectedMapSubproof).lookup(pkChild) == 0);
    ASSERT_TRUE(((((m).overallHashMemory)).rejectedMapSubproof).lookup(pkMain) != 0);
    ASSERT_FALSE(((((m).overallHashMemory)).subproofRevisitInProgress).contains(pkChild));
    ASSERT_TRUE(((((m).overallHashMemory)).subproofRevisitInProgress).contains(pkMain));
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
    int32_t countSubproofScopes(gl::tests::ProverFixture& analyzer, gl::Memory& memory,
                                const std::string& goal, const std::string& bare) {
        int32_t n = 0;
        for (gl::NameId id = 2; id <= ((memory).nameMap).nameCount(); ++id) {
            if (static_cast<std::size_t>(id)
                    >= static_cast<std::size_t>(((memory).nameMap).stackSize())
                || ((memory).nameMap).stackEmpty(id))
                continue;
            gl::StrSpan goalOut, bareOut;
            if (!gl::ScopeSyntax::splitSubproofPayload(
                    ((memory).nameMap).decodeSubView(((memory).nameMap).stackBack(id)),
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
    gl::tests::ProverFixture analyzer("Peano");
    gl::Memory memory;
    memory.setVarDefSets(gl::StrSpan("1:(1);2:(1);3:(1)", 17));
    const std::string goal = "(interval[1,4,2,7,10])";
    const std::string bare = "(implication90[1,2])";

    gl::WorkInstruction wi(&lb);
    buildImplication90(wi);
    ASSERT_TRUE(analyzer.implicationSubproofs.openImplicationSubproof(wi, 0, memory, sp("main"),
                    gl::StrSpan(goal))
                == gl::ImplicationSubproofOutcome::Opened);
    ASSERT_EQ(countSubproofScopes(analyzer, memory, goal, bare), 1);
    ASSERT_TRUE(((memory).integrationPrepared).contains(gl::mintTemplateKey(
        ((memory).templateInterner), ((memory).nameMap), goal + "_subproof_" + bare,
        std::string("main"))));

    ASSERT_TRUE(analyzer.implicationSubproofs.openImplicationSubproof(wi, 0, memory, sp("main"),
                    gl::StrSpan(goal))
                == gl::ImplicationSubproofOutcome::Prepared);
    ASSERT_EQ(countSubproofScopes(analyzer, memory, goal, bare), 1);
}

// openImplicationSubproof: a clean compact already known at the scope needs no
// subproof — no scope, no prepared key.
TEST(rejected_subproof, open_implication_subproof_known_compact_opens_nothing) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    gl::tests::ProverFixture analyzer("Peano");
    gl::Memory memory;
    memory.setVarDefSets(gl::StrSpan("1:(1);2:(1);3:(1)", 17));
    const std::string goal = "(interval[1,4,2,7,10])";
    const std::string bare = "(implication90[1,2])";
    gl::upsertStatementKey(((memory).intKnownStatements),
        gl::packStatementKey(((memory).nameMap).encode(bare), gl::NameMap::MAIN_ID),
        /*local=*/true, /*fullyDisintegrated=*/false);

    gl::WorkInstruction wi(&lb);
    buildImplication90(wi);
    ASSERT_TRUE(analyzer.implicationSubproofs.openImplicationSubproof(wi, 0, memory, sp("main"),
                    gl::StrSpan(goal))
                == gl::ImplicationSubproofOutcome::Known);
    ASSERT_EQ(countSubproofScopes(analyzer, memory, goal, bare), 0);
    ASSERT_EQ(((memory).integrationPrepared).count(), 0);
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
        const int64_t key = gl::packStatementKey(((memory).nameMap).encode(sp(text)), scope);
        gl::upsertStatementKey(((memory).intKnownStatements), key, /*local=*/true);
        return key;
    }

    /// @brief The packed park key of clean gate @p gate at main.
    int64_t mainParkKey(gl::Memory& memory, const char* gate) {
        return gl::mintTemplateKey(((memory).templateInterner), ((memory).nameMap),
                                   std::string(gate), std::string("main"));
    }

    /// @brief Register @p goal as an open goal at main.
    void addMainGoal(gl::Memory& memory, const std::string& goal) {
        ((memory).intToBeProved).assignSet(gl::StatementKey{
            ((memory).nameMap).encode(gl::StrSpan(goal)), gl::NameMap::MAIN_ID }, nullptr, 0);
    }

    /// @brief Fresh Peano analyzer fixture state for the gate tests.
    struct GateFixture {
        gl::GlobalMemoryManager g;
        gl::LbArena* lb = nullptr;
        gl::tests::ProverFixture analyzer{ "Peano" };
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
    using EA = gl::tests::ProverFixture;
    EA ea(std::string("FTA"));
    ASSERT_TRUE(gl::tests::ImplicationSubproofsFixture::isSubproofGateElement(sp("(in2[u_10,u_14,u_g])"), ea.definitions));
    ASSERT_TRUE(gl::tests::ImplicationSubproofsFixture::isSubproofGateElement(sp("!(in2[u_10,u_14,u_g])"), ea.definitions));
    ASSERT_TRUE(gl::tests::ImplicationSubproofsFixture::isSubproofGateElement(sp("(existence5[u_1,u_2])"), ea.definitions));
    ASSERT_FALSE(gl::tests::ImplicationSubproofsFixture::isSubproofGateElement(sp("(implication32[u_2,u_11,u_g])"), ea.definitions));
    ASSERT_FALSE(gl::tests::ImplicationSubproofsFixture::isSubproofGateElement(sp("(in2[i,u_14,u_g])"), ea.definitions));
    ASSERT_FALSE(gl::tests::ImplicationSubproofsFixture::isSubproofGateElement(sp("(in2[marker,u_14,u_g])"), ea.definitions));
    ASSERT_FALSE(gl::tests::ImplicationSubproofsFixture::isSubproofGateElement(sp("(in2[pi_1,u_14,u_g])"), ea.definitions));
    // A configured definition whose whole body is one implication is an
    // implication element as well; its negation gates like any negated element.
    ASSERT_FALSE(gl::tests::ImplicationSubproofsFixture::isSubproofGateElement(sp("(subset[u_1,u_2])"), ea.definitions));
    ASSERT_FALSE(gl::tests::ImplicationSubproofsFixture::isSubproofGateElement(sp("(primeseq[u_1,u_4,u_5,u_6,u_7,u_10])"), ea.definitions));
    ASSERT_TRUE(gl::tests::ImplicationSubproofsFixture::isSubproofGateElement(sp("!(subset[u_1,u_2])"), ea.definitions));

    GateFixture f;
    gl::WorkInstruction wi(f.lb);
    buildGatedImplication90(wi);
    ASSERT_FALSE(gl::tests::ImplicationSubproofsFixture::isParentAndOf(wi, 0, sp("(implication90[u_1,u_2])")));
    ASSERT_TRUE(gl::tests::ImplicationSubproofsFixture::isParentAndOf(wi, 1, sp("(implication90[u_1,u_2])")));
    ASSERT_FALSE(gl::tests::ImplicationSubproofsFixture::isParentAndOf(wi, 1, sp("(implication91[u_1,u_2])")));
}

// elementKnownAtOrAbove: reflexive equality, a literal row at an ancestor, a
// class variant at the scope, and misses (descendant-only row, unrelated row).
TEST(rejected_subproof, element_known_at_or_above_probe) {
    gl::tests::ProverFixture analyzer("Peano");
    gl::Memory m;
    const gl::NameId child = ((m).nameMap).encodePush(gl::NameMap::MAIN_ID, sp("(implication7[a])"));
    ASSERT_TRUE(analyzer.equivalenceQueries.elementKnownAtOrAbove(sp("(=[q,q])"), child, m));
    ASSERT_FALSE(analyzer.equivalenceQueries.elementKnownAtOrAbove(sp("(=[q,r])"), child, m));

    makeKnown(m, "(in2[c,7,3])", gl::NameMap::MAIN_ID);
    ASSERT_TRUE(analyzer.equivalenceQueries.elementKnownAtOrAbove(sp("(in2[c,7,3])"), child, m));
    ASSERT_TRUE(analyzer.equivalenceQueries.elementKnownAtOrAbove(sp("(in2[c,7,3])"), gl::NameMap::MAIN_ID, m));
    makeKnown(m, "(in2[d,7,3])", child);
    ASSERT_FALSE(analyzer.equivalenceQueries.elementKnownAtOrAbove(sp("(in2[d,7,3])"), gl::NameMap::MAIN_ID, m));

    gl::EquivalenceClass cls;
    cls.setMembersFromNames({ "a", "b" }, ((m).nameMap));
    m.assignClassesById(gl::NameMap::MAIN_ID, { cls });
    seedFixtureCanonicalClasses(m);
    makeKnown(m, "(in2[b,7,3])", gl::NameMap::MAIN_ID);
    ASSERT_TRUE(analyzer.equivalenceQueries.elementKnownAtOrAbove(sp("(in2[a,7,3])"), gl::NameMap::MAIN_ID, m));
    ASSERT_FALSE(analyzer.equivalenceQueries.elementKnownAtOrAbove(sp("(in2[e,7,3])"), gl::NameMap::MAIN_ID, m));
}

// orDisjunctTrueAt (D-438): a reflexive equality or a
// known row at the scope or an ancestor holds; a positive conjunction holds
// when every flattened element holds, nested conjunctions included; one open
// element, a negated conjunction without its own row, or a row known only in
// a descendant scope does not hold.
TEST(rejected_subproof, or_disjunct_true_at_probe) {
    gl::tests::ProverFixture analyzer("Peano");
    gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions)["and9001"] = gl::LogicalEntity("and",
        { "(=[u_1,u_2])", "(=[u_3,u_4])" }, "(and9001[u_1,u_2,u_3,u_4])", 4);
    gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions)["and9002"] = gl::LogicalEntity("and",
        { "(in2[u_1,u_2,u_3])", "(and9001[u_2,u_2,u_1,u_1])" }, "(and9002[u_1,u_2,u_3])", 3);
    gl::Memory m;
    const gl::NameId child = ((m).nameMap).encodePush(gl::NameMap::MAIN_ID, sp("(implication7[a])"));

    ASSERT_TRUE(gl::tests::CaseProcessorFixture::orDisjunctTrueAt(analyzer.caseProcessor, sp("(=[q,q])"), child, m));
    ASSERT_FALSE(gl::tests::CaseProcessorFixture::orDisjunctTrueAt(analyzer.caseProcessor, sp("(=[q,r])"), child, m));
    ASSERT_TRUE(gl::tests::CaseProcessorFixture::orDisjunctTrueAt(analyzer.caseProcessor, sp("(and9001[q,q,r,r])"), child, m));
    ASSERT_FALSE(gl::tests::CaseProcessorFixture::orDisjunctTrueAt(analyzer.caseProcessor, sp("(and9001[q,q,r,s])"), child, m));
    ASSERT_FALSE(gl::tests::CaseProcessorFixture::orDisjunctTrueAt(analyzer.caseProcessor, sp("!(and9001[q,q,r,r])"), child, m));

    makeKnown(m, "(in2[c,7,3])", gl::NameMap::MAIN_ID);
    ASSERT_TRUE(gl::tests::CaseProcessorFixture::orDisjunctTrueAt(analyzer.caseProcessor, sp("(in2[c,7,3])"), child, m));
    ASSERT_TRUE(gl::tests::CaseProcessorFixture::orDisjunctTrueAt(analyzer.caseProcessor, sp("(and9002[c,7,3])"), child, m));
    ASSERT_FALSE(gl::tests::CaseProcessorFixture::orDisjunctTrueAt(analyzer.caseProcessor, sp("(and9002[d,7,3])"), child, m));

    makeKnown(m, "(=[r,s])", gl::NameMap::MAIN_ID);
    ASSERT_TRUE(gl::tests::CaseProcessorFixture::orDisjunctTrueAt(analyzer.caseProcessor, sp("(and9001[q,q,r,s])"), child, m));

    makeKnown(m, "(in2[e,7,3])", child);
    ASSERT_FALSE(gl::tests::CaseProcessorFixture::orDisjunctTrueAt(analyzer.caseProcessor, sp("(in2[e,7,3])"), gl::NameMap::MAIN_ID, m));
    ASSERT_TRUE(gl::tests::CaseProcessorFixture::orDisjunctTrueAt(analyzer.caseProcessor, sp("(in2[e,7,3])"), child, m));

    makeKnown(m, "!(in2[d,7,3])", gl::NameMap::MAIN_ID);
    ASSERT_TRUE(gl::tests::CaseProcessorFixture::orDisjunctTrueAt(analyzer.caseProcessor, sp("!(in2[d,7,3])"), child, m));
}

// openImplicationSubproof gate: an unknown gate element parks the implication
// (record = implication + parent and + root goal) and opens nothing; a known
// gate element lets it open.
TEST(rejected_subproof, open_parks_under_unknown_gate_and_opens_when_known) {
    GateFixture f;
    gl::WorkInstruction wi(f.lb);
    buildGatedImplication90(wi);
    ASSERT_TRUE(f.analyzer.implicationSubproofs.openImplicationSubproof(wi, 0, f.memory, sp("main"), gl::StrSpan(f.goal))
                == gl::ImplicationSubproofOutcome::Parked);
    ASSERT_EQ(countSubproofScopes(f.analyzer, f.memory, f.goal, f.bare), 0);
    ASSERT_EQ(((f.memory).integrationPrepared).count(), 0);
    const int64_t pk = mainParkKey(f.memory, "(in2[1,2,3])");
    const gl::IntegrationEntryMap records = gl::admissionIntegrationRecordsAt(
        ((((f.memory).overallHashMemory)).rejectedMapSubproof), pk, ((f.memory).valueInterner));
    ASSERT_EQ(records.size(), static_cast<std::size_t>(1));
    const gl::IntInstruction& record = records.begin()->first;
    ASSERT_EQ(record.data.size(), static_cast<std::size_t>(2));
    ASSERT_EQ(((f.memory).valueInterner).decode(record.data[0].signature),
              std::string("(implication90[u_1,u_2])"));
    ASSERT_EQ(((f.memory).valueInterner).decode(record.data[1].signature),
              std::string("(and90[u_1,u_2,u_3])"));
    ASSERT_EQ(((f.memory).valueInterner).decode(record.markedGoal), f.goal);

    makeKnown(f.memory, "(in2[1,2,3])", gl::NameMap::MAIN_ID);
    ASSERT_TRUE(f.analyzer.implicationSubproofs.openImplicationSubproof(wi, 0, f.memory, sp("main"), gl::StrSpan(f.goal))
                == gl::ImplicationSubproofOutcome::Opened);
    ASSERT_EQ(countSubproofScopes(f.analyzer, f.memory, f.goal, f.bare), 1);
}

namespace {
    /// @brief implication90 over the construction @p name — premise
    ///        `(=[u_<name>,X])`, head `(=[u_2,X])` — plus its parent and
    ///        `(and90[u_<name>,u_2,u_3])` gated by `(in2[u_<name>,u_2,u_3])`.
    void buildGatedImplicationOver(gl::WorkInstruction& wi, const std::string& name) {
        const std::string u = "u_" + name;
        const std::string premise = "(=[" + u + ",X])";
        const std::string signature = "(implication90[" + u + ",u_2])";
        const std::string gate = "(in2[" + u + ",u_2,u_3])";
        const std::string andSignature = "(and90[" + u + ",u_2,u_3])";
        int32_t mark = wi.elemMark();
        wi.addElement(gl::StrSpan(premise));
        wi.addElement(sp("(=[u_2,X])"));
        wi.commitEntity(sp("implication"), gl::StrSpan(signature), sp(""), 2, mark);
        mark = wi.elemMark();
        wi.addElement(gl::StrSpan(signature));
        wi.addElement(gl::StrSpan(gate));
        wi.commitEntity(sp("and"), gl::StrSpan(andSignature), sp(""), 3, mark);
    }
} // namespace

// openImplicationSubproof built-on-listed refusal
// (I-293): an implication naming a construction in
// Memory::builtOnListedWitnesses is refused before the gate — no park, no
// prepared key, no scope — even once its gate element is known; an
// implication over an unlisted name still parks and opens as before.
TEST(rejected_subproof, open_refuses_an_implication_over_a_built_on_listed_witness) {
    const std::string listed = "int_0_lev_6_123";
    {
        GateFixture f;
        ((f.memory).builtOnListedWitnesses).mint(gl::StrSpan(listed));
        gl::WorkInstruction wi(f.lb);
        buildGatedImplicationOver(wi, listed);
        ASSERT_TRUE(f.analyzer.implicationSubproofs.openImplicationSubproof(wi, 0, f.memory, sp("main"), gl::StrSpan())
                    == gl::ImplicationSubproofOutcome::Refused);
        const std::string gate = "(in2[" + listed + ",2,3])";
        ASSERT_TRUE(((((f.memory).overallHashMemory)).rejectedMapSubproof).lookup(
                        mainParkKey(f.memory, gate.c_str())) == 0);
        makeKnown(f.memory, gate.c_str(), gl::NameMap::MAIN_ID);
        ASSERT_TRUE(f.analyzer.implicationSubproofs.openImplicationSubproof(wi, 0, f.memory, sp("main"), gl::StrSpan(f.goal))
                    == gl::ImplicationSubproofOutcome::Refused);
        ASSERT_EQ(((f.memory).integrationPrepared).count(), 0);
        ASSERT_EQ(countSubproofScopes(f.analyzer, f.memory, f.goal,
                                      "(implication90[" + listed + ",2])"), 0);
    }
    {
        GateFixture f;
        ((f.memory).builtOnListedWitnesses).mint(gl::StrSpan(listed));
        gl::WorkInstruction wi(f.lb);
        buildGatedImplication90(wi);
        ASSERT_TRUE(f.analyzer.implicationSubproofs.openImplicationSubproof(wi, 0, f.memory, sp("main"), gl::StrSpan(f.goal))
                    == gl::ImplicationSubproofOutcome::Parked);
        makeKnown(f.memory, "(in2[1,2,3])", gl::NameMap::MAIN_ID);
        ASSERT_TRUE(f.analyzer.implicationSubproofs.openImplicationSubproof(wi, 0, f.memory, sp("main"), gl::StrSpan(f.goal))
                    == gl::ImplicationSubproofOutcome::Opened);
        ASSERT_EQ(countSubproofScopes(f.analyzer, f.memory, f.goal, f.bare), 1);
    }
}

namespace {
    /// @brief Peano's `implication0[u_1,u_2]` ("every index of relation u_1
    ///        lies in u_2") over @p relation and @p set: premise
    ///        `(in2[1,2,u_<relation>])`, head `(in[1,u_<set>])`.
    void buildIndexImplication(gl::WorkInstruction& wi, const std::string& relation,
                               const std::string& set) {
        const std::string premise = "(in2[1,2,u_" + relation + "])";
        const std::string head = "(in[1,u_" + set + "])";
        const std::string signature = "(implication0[u_" + relation + ",u_" + set + "])";
        const int32_t mark = wi.elemMark();
        wi.addElement(gl::StrSpan(premise));
        wi.addElement(gl::StrSpan(head));
        wi.commitEntity(sp("implication"), gl::StrSpan(signature), sp(""), 2, mark);
    }

    /// @brief Deposit @p text at @p scope through the statement door, level 0,
    ///        with a task-formulation origin.
    void depositAt(gl::tests::ProverFixture& analyzer, gl::Memory& memory,
                   const std::string& text, const std::string& scope) {
        const int level = 0;
        const gl::TransientOrigin origin{ true, gl::OriginTag::taskFormulation, nullptr, 0 };
        analyzer.statementMemory.addExprToMemoryBlock(gl::StrSpan(text), memory, 0, 0, &level, 1,
            origin, -1, -1, gl::StrSpan(scope), false);
    }
} // namespace

// openImplicationSubproof, step 5 (D-532): a bare
// numbered implication subproof opens with its reductio beside it — the
// contradiction scope at the same parent, seeded with the compact. A
// goal-qualified opening and a primed contradiction block get no reductio.
TEST(rejected_subproof, open_bare_implication_subproof_primes_its_reductio) {
    const std::string compact = "(implication0[7,8])";
    const std::string reductioPayload = "contradiction_!" + compact;
    const std::string reductioScope = "main_boundary_" + reductioPayload;
    {
        GateFixture f;
        gl::WorkInstruction wi(f.lb);
        buildIndexImplication(wi, "7", "8");
        ASSERT_TRUE(f.analyzer.implicationSubproofs.openImplicationSubproof(wi, 0, f.memory, sp("main"),
                        gl::StrSpan()) == gl::ImplicationSubproofOutcome::Opened);
        ASSERT_TRUE(((f.memory).nameMap).lookup(gl::StrSpan("main_boundary_" + compact)) != 0);
        ASSERT_TRUE(((f.memory).integrationPrepared).contains(gl::mintTemplateKey(
            ((f.memory).templateInterner), ((f.memory).nameMap), reductioPayload, std::string("main"))));
        ASSERT_TRUE(gl::lookupStatementFlags(((f.memory).intKnownStatements), ((f.memory).nameMap),
                        gl::StrSpan(compact), gl::StrSpan(reductioScope)) != nullptr);
    }
    {
        GateFixture f;
        gl::WorkInstruction wi(f.lb);
        buildIndexImplication(wi, "7", "8");
        ASSERT_TRUE(f.analyzer.implicationSubproofs.openImplicationSubproof(wi, 0, f.memory, sp("main"),
                        gl::StrSpan(f.goal)) == gl::ImplicationSubproofOutcome::Opened);
        ASSERT_EQ(countSubproofScopes(f.analyzer, f.memory, f.goal, compact), 1);
        ASSERT_EQ(((f.memory).nameMap).lookup(gl::StrSpan(reductioScope)), 0);
    }
    {
        GateFixture f;
        f.memory.primedForContradiction = true;
        gl::WorkInstruction wi(f.lb);
        buildIndexImplication(wi, "7", "8");
        ASSERT_TRUE(f.analyzer.implicationSubproofs.openImplicationSubproof(wi, 0, f.memory, sp("main"),
                        gl::StrSpan()) == gl::ImplicationSubproofOutcome::Opened);
        ASSERT_TRUE(((f.memory).nameMap).lookup(gl::StrSpan("main_boundary_" + compact)) != 0);
        ASSERT_EQ(((f.memory).nameMap).lookup(gl::StrSpan(reductioScope)), 0);
    }
}

// The reductio route end to end (D-532): the
// reductio's consequence (in[5,8]) against main's !(in[5,8]) — a counterexample
// to "every index of 7 lies in 8" — discharges !(implication0[7,8]) at main
// and queues the reductio; once the negated compact is known at main, the
// subproof is retired and the discharged reductio is not queued again.
TEST(rejected_subproof, reductio_counterexample_retires_the_subproof) {
    GateFixture f;
    const std::string compact = "(implication0[7,8])";
    const std::string reductioScope = "main_boundary_contradiction_!" + compact;
    gl::WorkInstruction wi(f.lb);
    buildIndexImplication(wi, "7", "8");
    ASSERT_TRUE(f.analyzer.implicationSubproofs.openImplicationSubproof(wi, 0, f.memory, sp("main"),
                    gl::StrSpan()) == gl::ImplicationSubproofOutcome::Opened);
    const gl::NameId subproof = ((f.memory).nameMap).lookup(gl::StrSpan("main_boundary_" + compact));
    const gl::NameId reductio = ((f.memory).nameMap).lookup(gl::StrSpan(reductioScope));
    ASSERT_TRUE(subproof != 0 && reductio != 0);

    depositAt(f.analyzer, f.memory, "!(in[5,8])", "main");
    depositAt(f.analyzer, f.memory, "(in[5,8])", reductioScope);
    f.analyzer.goalProcessor.dischargeContradictionScopes(f.memory,
        ((f.memory).nextIterationInternalMail));
    {
        const gl::Mail mail = gl::makeHeapMail(((f.memory).nextIterationInternalMail),
            ((f.memory).nameMap), ((f.memory).originInterner));
        bool sawProduct = false;
        for (const auto& st : mail.statements)
            sawProduct = sawProduct || (st.first.original == "!" + compact
                                        && st.first.validityName == "main");
        ASSERT_TRUE(sawProduct);
    }
    ASSERT_TRUE(((f.memory).pendingWipeScopes).contains(reductio));
    ASSERT_FALSE(((f.memory).pendingWipeScopes).contains(subproof));

    // The absorb makes the product known at main.
    gl::upsertStatementKey(((f.memory).intKnownStatements),
        gl::packStatementKey(((f.memory).nameMap).encode(gl::StrSpan("!" + compact)), gl::NameMap::MAIN_ID),
        /*local=*/true, /*fullyDisintegrated=*/false);
    f.analyzer.implicationSubproofs.retireSettledImplicationSubproofs(f.memory);
    ASSERT_TRUE(((f.memory).pendingWipeScopes).contains(subproof));
    const int32_t queued = ((f.memory).pendingWipeScopes).count();
    f.analyzer.implicationSubproofs.retireSubproofReductios(f.memory);
    ASSERT_EQ(((f.memory).pendingWipeScopes).count(), queued);
}

// parkImplicationSubproof: the key is canonical at the park scope, so
// class-equal gates share one key, while each record keeps the instruction's
// own text (the text the and's integration rule was installed with).
TEST(rejected_subproof, park_key_is_canonical_record_keeps_instruction_text) {
    GateFixture f;
    gl::EquivalenceClass cls;
    cls.setMembersFromNames({ "a", "b" }, ((f.memory).nameMap));
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
    gl::tests::ImplicationSubproofsFixture::parkImplicationSubproof(f.analyzer.implicationSubproofs, viaA, 0, sp("(in2[a,2,3])"), f.memory, sp("main"), gl::StrSpan(f.goal));
    gl::tests::ImplicationSubproofsFixture::parkImplicationSubproof(f.analyzer.implicationSubproofs, viaB, 0, sp("(in2[b,2,3])"), f.memory, sp("main"), gl::StrSpan(f.goal));

    ASSERT_EQ(((((f.memory).overallHashMemory)).rejectedMapSubproof).count(), 1);
    gl::ScratchArena& strings = gl::scratchArenas().forSlot(gl::scratchArenas().slotCount() - 1);
    gl::ScratchScope frame(strings);
    const gl::ScratchString canonicalA = f.analyzer.canonicalTemplates.canonicalAdmissionText(
        f.memory, sp("(in2[a,2,3])"), gl::NameMap::MAIN_ID, strings, true);
    const gl::ScratchString canonicalB = f.analyzer.canonicalTemplates.canonicalAdmissionText(
        f.memory, sp("(in2[b,2,3])"), gl::NameMap::MAIN_ID, strings, true);
    ASSERT_TRUE(gl::equalSpans(gl::StrSpan(canonicalA), gl::StrSpan(canonicalB)));
    const int64_t pk = gl::mintTemplateKey(((f.memory).templateInterner), ((f.memory).nameMap),
        gl::StrSpan(canonicalA), sp("main"));
    const gl::IntegrationEntryMap records = gl::admissionIntegrationRecordsAt(
        ((((f.memory).overallHashMemory)).rejectedMapSubproof), pk, ((f.memory).valueInterner));
    ASSERT_EQ(records.size(), static_cast<std::size_t>(2));
    std::vector<std::string> gates;
    for (const auto& entry : records)
        gates.push_back(((f.memory).valueInterner).decode(entry.first.data[1].elements[1]));
    std::sort(gates.begin(), gates.end());
    ASSERT_EQ(gates[0], std::string("(in2[u_a,u_2,u_3])"));
    ASSERT_EQ(gates[1], std::string("(in2[u_b,u_2,u_3])"));
}

// stageSubproofArrival: nothing is staged while no implication is parked.
TEST(rejected_subproof, stage_arrival_only_while_parks_exist) {
    GateFixture f;
    const int64_t key = makeKnown(f.memory, "(in[1,3])", gl::NameMap::MAIN_ID);
    gl::StatementArrivals::stageSubproofArrival(f.memory, key);
    ASSERT_EQ(((f.memory).pendingSubproofArrivals).size(), 0);
    gl::WorkInstruction wi(f.lb);
    buildGatedImplication90(wi);
    (void)f.analyzer.implicationSubproofs.openImplicationSubproof(wi, 0, f.memory, sp("main"), gl::StrSpan(f.goal));
    gl::StatementArrivals::stageSubproofArrival(f.memory, key);
    ASSERT_EQ(((f.memory).pendingSubproofArrivals).size(), 1);
}

// drainSubproofArrivals: an arrival only at a descendant scope leaves the park
// asleep; the gate arriving at main reopens it (key erased, subproof opened,
// queue consumed, guard released).
TEST(rejected_subproof, drain_reopens_on_gate_arrival_at_or_above_the_park) {
    GateFixture f;
    addMainGoal(f.memory, f.goal);
    gl::WorkInstruction wi(f.lb);
    buildGatedImplication90(wi);
    (void)f.analyzer.implicationSubproofs.openImplicationSubproof(wi, 0, f.memory, sp("main"), gl::StrSpan(f.goal));
    const int64_t pk = mainParkKey(f.memory, "(in2[1,2,3])");
    ASSERT_TRUE(((((f.memory).overallHashMemory)).rejectedMapSubproof).lookup(pk) != 0);

    const gl::NameId child = ((f.memory).nameMap).encodePush(gl::NameMap::MAIN_ID, sp("(implication7[a])"));
    gl::StatementArrivals::stageSubproofArrival(f.memory, makeKnown(f.memory, "(in2[1,2,3])", child));
    f.analyzer.implicationSubproofs.drainSubproofArrivals(f.memory);
    ASSERT_EQ(((f.memory).pendingSubproofArrivals).size(), 0);
    ASSERT_TRUE(((((f.memory).overallHashMemory)).rejectedMapSubproof).lookup(pk) != 0);
    ASSERT_EQ(countSubproofScopes(f.analyzer, f.memory, f.goal, f.bare), 0);

    gl::StatementArrivals::stageSubproofArrival(f.memory, makeKnown(f.memory, "(in2[1,2,3])", gl::NameMap::MAIN_ID));
    f.analyzer.implicationSubproofs.drainSubproofArrivals(f.memory);
    ASSERT_EQ(((f.memory).pendingSubproofArrivals).size(), 0);
    ASSERT_TRUE(((((f.memory).overallHashMemory)).rejectedMapSubproof).lookup(pk) == 0);
    ASSERT_EQ(countSubproofScopes(f.analyzer, f.memory, f.goal, f.bare), 1);
    ASSERT_FALSE(((((f.memory).overallHashMemory)).subproofRevisitInProgress).contains(pk));
}

// revisitRejectedSubproof drops: a record whose root goal is no longer open,
// and a record whose parent and is already known, open nothing.
TEST(rejected_subproof, revisit_drops_gone_goal_and_known_parent_and) {
    {
        GateFixture f;
        gl::WorkInstruction wi(f.lb);
        buildGatedImplication90(wi);
        (void)f.analyzer.implicationSubproofs.openImplicationSubproof(wi, 0, f.memory, sp("main"), gl::StrSpan(f.goal));
        const int64_t pk = mainParkKey(f.memory, "(in2[1,2,3])");
        makeKnown(f.memory, "(in2[1,2,3])", gl::NameMap::MAIN_ID);
        gl::tests::ImplicationSubproofsFixture::revisitRejectedSubproof(f.analyzer.implicationSubproofs, pk, f.memory);
        ASSERT_TRUE(((((f.memory).overallHashMemory)).rejectedMapSubproof).lookup(pk) == 0);
        ASSERT_EQ(countSubproofScopes(f.analyzer, f.memory, f.goal, f.bare), 0);
    }
    {
        GateFixture f;
        addMainGoal(f.memory, f.goal);
        gl::WorkInstruction wi(f.lb);
        buildGatedImplication90(wi);
        (void)f.analyzer.implicationSubproofs.openImplicationSubproof(wi, 0, f.memory, sp("main"), gl::StrSpan(f.goal));
        const int64_t pk = mainParkKey(f.memory, "(in2[1,2,3])");
        makeKnown(f.memory, "(in2[1,2,3])", gl::NameMap::MAIN_ID);
        makeKnown(f.memory, "(and90[1,2,3])", gl::NameMap::MAIN_ID);
        gl::tests::ImplicationSubproofsFixture::revisitRejectedSubproof(f.analyzer.implicationSubproofs, pk, f.memory);
        ASSERT_TRUE(((((f.memory).overallHashMemory)).rejectedMapSubproof).lookup(pk) == 0);
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
    (void)f.analyzer.implicationSubproofs.openImplicationSubproof(wi, 0, f.memory, sp("main"), gl::StrSpan(f.goal));
    const int64_t first = mainParkKey(f.memory, "(in2[1,2,3])");
    gl::tests::ImplicationSubproofsFixture::revisitRejectedSubproof(f.analyzer.implicationSubproofs, first, f.memory);
    ASSERT_TRUE(((((f.memory).overallHashMemory)).rejectedMapSubproof).lookup(first) != 0);

    makeKnown(f.memory, "(in2[1,2,3])", gl::NameMap::MAIN_ID);
    gl::tests::ImplicationSubproofsFixture::revisitRejectedSubproof(f.analyzer.implicationSubproofs, first, f.memory);
    const int64_t second = mainParkKey(f.memory, "(in[1,3])");
    ASSERT_TRUE(((((f.memory).overallHashMemory)).rejectedMapSubproof).lookup(first) == 0);
    ASSERT_TRUE(((((f.memory).overallHashMemory)).rejectedMapSubproof).lookup(second) != 0);
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
        cls.setMembersFromNames({ "a", "b" }, ((memory).nameMap));
        memory.assignClassesById(gl::NameMap::MAIN_ID, { cls });
        seedFixtureCanonicalClasses(memory);
    }
} // namespace

// stageSubproofRecheck: nothing is staged while no implication is parked.
TEST(rejected_subproof, stage_recheck_only_while_parks_exist) {
    GateFixture f;
    f.analyzer.canonicalizer.stageSubproofRecheck(f.memory, gl::NameMap::MAIN_ID);
    ASSERT_EQ(((f.memory).pendingSubproofRechecks).size(), 0);
    gl::WorkInstruction wi(f.lb);
    buildGatedImplication90(wi);
    (void)f.analyzer.implicationSubproofs.openImplicationSubproof(wi, 0, f.memory, sp("main"), gl::StrSpan(f.goal));
    f.analyzer.canonicalizer.stageSubproofRecheck(f.memory, gl::NameMap::MAIN_ID);
    ASSERT_EQ(((f.memory).pendingSubproofRechecks).size(), 1);
}

// rekeySubproofPark: after a class merge the two spellings' parks end under
// one canonical key with both records verbatim; the class writer staged the
// recheck.
TEST(rejected_subproof, rekey_moves_records_verbatim_to_canonical_key) {
    GateFixture f;
    gl::WorkInstruction viaA(f.lb), viaB(f.lb);
    buildImplication90WithGate(viaA, "(in2[u_a,u_2,u_3])");
    buildImplication90WithGate(viaB, "(in2[u_b,u_2,u_3])");
    (void)f.analyzer.implicationSubproofs.openImplicationSubproof(viaA, 0, f.memory, sp("main"), gl::StrSpan(f.goal));
    (void)f.analyzer.implicationSubproofs.openImplicationSubproof(viaB, 0, f.memory, sp("main"), gl::StrSpan(f.goal));
    const int64_t keyA = mainParkKey(f.memory, "(in2[a,2,3])");
    const int64_t keyB = mainParkKey(f.memory, "(in2[b,2,3])");
    ASSERT_TRUE(((((f.memory).overallHashMemory)).rejectedMapSubproof).lookup(keyA) != 0);
    ASSERT_TRUE(((((f.memory).overallHashMemory)).rejectedMapSubproof).lookup(keyB) != 0);

    mergeAB(f.memory);
    ASSERT_TRUE(((f.memory).pendingSubproofRechecks).size() > 0);
    const int64_t canonicalA = gl::tests::ImplicationSubproofsFixture::rekeySubproofPark(f.analyzer.implicationSubproofs, keyA, f.memory);
    const int64_t canonicalB = gl::tests::ImplicationSubproofsFixture::rekeySubproofPark(f.analyzer.implicationSubproofs, keyB, f.memory);
    ASSERT_EQ(canonicalA, canonicalB);
    ASSERT_TRUE((canonicalA == keyA) != (canonicalA == keyB));
    const int64_t moved = canonicalA == keyA ? keyB : keyA;
    ASSERT_TRUE(((((f.memory).overallHashMemory)).rejectedMapSubproof).lookup(moved) == 0);
    const gl::IntegrationEntryMap records = gl::admissionIntegrationRecordsAt(
        ((((f.memory).overallHashMemory)).rejectedMapSubproof), canonicalA, ((f.memory).valueInterner));
    ASSERT_EQ(records.size(), static_cast<std::size_t>(2));
    std::vector<std::string> gates;
    for (const auto& entry : records)
        gates.push_back(((f.memory).valueInterner).decode(entry.first.data[1].elements[1]));
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
    ASSERT_TRUE(f.analyzer.implicationSubproofs.openImplicationSubproof(wi, 0, f.memory, sp("main"), gl::StrSpan(f.goal))
                == gl::ImplicationSubproofOutcome::Parked);
    makeKnown(f.memory, "(in2[b,2,3])", gl::NameMap::MAIN_ID);
    f.analyzer.implicationSubproofs.drainSubproofArrivals(f.memory);
    ASSERT_EQ(countSubproofScopes(f.analyzer, f.memory, f.goal, f.bare), 0);

    mergeAB(f.memory);
    f.analyzer.implicationSubproofs.drainSubproofArrivals(f.memory);
    ASSERT_EQ(((f.memory).pendingSubproofRechecks).size(), 0);
    ASSERT_EQ(countSubproofScopes(f.analyzer, f.memory, f.goal, f.bare), 1);
    ASSERT_TRUE(((((f.memory).overallHashMemory)).rejectedMapSubproof).lookup(
        mainParkKey(f.memory, "(in2[a,2,3])")) == 0);
    ASSERT_TRUE(((((f.memory).overallHashMemory)).rejectedMapSubproof).lookup(
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
    (void)f.analyzer.implicationSubproofs.openImplicationSubproof(shared, 0, f.memory, sp("main"), gl::StrSpan(f.goal));
    (void)f.analyzer.implicationSubproofs.openImplicationSubproof(shared, 0, f.memory, sp("main"), gl::StrSpan(other));
    (void)f.analyzer.implicationSubproofs.openImplicationSubproof(own, 0, f.memory, sp("main"), gl::StrSpan(f.goal));
    const int64_t sharedKey = mainParkKey(f.memory, "(in2[1,2,3])");
    const int64_t ownKey = mainParkKey(f.memory, "(in[1,3])");
    ASSERT_EQ(gl::admissionIntegrationRecordsAt(((((f.memory).overallHashMemory)).rejectedMapSubproof),
        sharedKey, ((f.memory).valueInterner)).size(), static_cast<std::size_t>(2));

    f.analyzer.implicationSubproofs.eraseGoalSubproofParks(f.memory, gl::StrSpan(f.goal));
    ASSERT_TRUE(((((f.memory).overallHashMemory)).rejectedMapSubproof).lookup(ownKey) == 0);
    const gl::IntegrationEntryMap left = gl::admissionIntegrationRecordsAt(
        ((((f.memory).overallHashMemory)).rejectedMapSubproof), sharedKey, ((f.memory).valueInterner));
    ASSERT_EQ(left.size(), static_cast<std::size_t>(1));
    ASSERT_EQ(((f.memory).valueInterner).decode(left.begin()->first.markedGoal), other);
}


namespace {
    void equalInputDefinition(gl::tests::ProverFixture& analyzer, const std::vector<std::string>& elements,
                              int arity = 2) {
        gl::LogicalEntity definition;
        definition.category = "implication";
        definition.arity = arity;
        definition.signature = arity == 2 ? "(implication9100[u_1,u_2])" : "(implication9100[u_1,u_2,u_3])";
        definition.elements = elements;
        gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions)["implication9100"] = definition;
    }

    gl::NameId equalInputScope(gl::Memory& memory, gl::NameId parent, const std::string& payload) {
        memory.integrationPrepared.mint(gl::mintTemplateKey(memory.templateInterner, memory.nameMap,
            payload, memory.nameMap.decode(parent)));
        return memory.nameMap.encodePush(parent, payload);
    }

    void equalInputClass(gl::Memory& memory, gl::NameId scope) {
        gl::EquivalenceClass equality;
        equality.setMembersFromNames({"a", "b"}, memory.nameMap);
        memory.assignClassesById(scope, {equality});
        seedFixtureCanonicalClasses(memory);
    }
}

TEST(rejected_subproof, equal_inputs_require_the_exact_direct_positive_premise_pair) {
    gl::tests::ProverFixture analyzer("Peano");
    analyzer.parameters.trackHistory = false;
    gl::Memory memory;
    const gl::NameId main = memory.nameMap.encode(sp("main"));
    equalInputDefinition(analyzer, {"(in[u_1,1])", "(in[u_2,1])"});
    ASSERT_FALSE(analyzer.implicationSubproofs.hasForbiddenEqualInputs(sp("(implication9100[a,b])"), main, memory));
    ASSERT_TRUE(analyzer.implicationSubproofs.hasForbiddenEqualInputs(sp("(implication9100[a,a])"), main, memory));
    equalInputClass(memory, main);
    ASSERT_TRUE(analyzer.implicationSubproofs.hasForbiddenEqualInputs(sp("(implication9100[a,b])"), main, memory));
    for (const char* premise : {"(=[u_1,u_2])", "(=[u_2,u_1])"}) {
        equalInputDefinition(analyzer, {premise, "(in[u_1,1])"});
        ASSERT_FALSE(analyzer.implicationSubproofs.hasForbiddenEqualInputs(sp("(implication9100[a,b])"), main, memory));
    }
    equalInputDefinition(analyzer, {"!(=[u_1,u_2])", "(in[u_1,1])"});
    ASSERT_TRUE(analyzer.implicationSubproofs.hasForbiddenEqualInputs(sp("(implication9100[a,b])"), main, memory));
    equalInputDefinition(analyzer, {"(in[u_1,1])", "(=[u_1,u_2])"});
    ASSERT_TRUE(analyzer.implicationSubproofs.hasForbiddenEqualInputs(sp("(implication9100[a,b])"), main, memory));
    equalInputDefinition(analyzer, {"(=[u_1,1])", "(=[1,u_2])", "(in[u_1,2])"});
    ASSERT_TRUE(analyzer.implicationSubproofs.hasForbiddenEqualInputs(sp("(implication9100[a,b])"), main, memory));
    equalInputDefinition(analyzer, {"(and9100[u_1,u_2])", "(in[u_1,1])"});
    ASSERT_TRUE(analyzer.implicationSubproofs.hasForbiddenEqualInputs(sp("(implication9100[a,b])"), main, memory));
    ASSERT_EQ(memory.exprOriginMap.count(), 0);
}

TEST(rejected_subproof, equal_input_permissions_do_not_close_transitively) {
    gl::tests::ProverFixture analyzer("Peano");
    gl::Memory memory;
    const gl::NameId main = memory.nameMap.encode(sp("main"));
    equalInputDefinition(analyzer, {"(=[u_1,u_2])", "(=[u_2,u_3])", "(in[u_1,1])"}, 3);
    ASSERT_TRUE(analyzer.implicationSubproofs.hasForbiddenEqualInputs(sp("(implication9100[a,a,a])"), main, memory));
    equalInputDefinition(analyzer, {"(=[u_1,u_2])", "(=[u_2,u_3])", "(=[u_1,u_3])", "(in[u_1,1])"}, 3);
    ASSERT_FALSE(analyzer.implicationSubproofs.hasForbiddenEqualInputs(sp("(implication9100[a,a,a])"), main, memory));
    ASSERT_FALSE(analyzer.implicationSubproofs.hasForbiddenEqualInputs(sp("(implication9101[a])"), main, memory));
}

TEST(rejected_subproof, equal_input_check_uses_scope_membership_even_when_every_member_is_weak) {
    gl::tests::ProverFixture analyzer("Peano");
    gl::Memory memory;
    const gl::NameId main = memory.nameMap.encode(sp("main"));
    const gl::NameId left = memory.nameMap.encodePush(main, sp("left"));
    const gl::NameId right = memory.nameMap.encodePush(main, sp("right"));
    const gl::NameId nested = memory.nameMap.encodePush(left, sp("nested"));
    equalInputDefinition(analyzer, {"(in[u_1,1])", "(in[u_2,1])"});
    equalInputClass(memory, left);
    memory.intWeakVariables.mint(gl::packStatementKey(memory.nameMap.lookup(sp("a")), left));
    memory.intWeakVariables.mint(gl::packStatementKey(memory.nameMap.lookup(sp("b")), left));
    ASSERT_FALSE(analyzer.implicationSubproofs.hasForbiddenEqualInputs(sp("(implication9100[a,b])"), main, memory));
    ASSERT_FALSE(analyzer.implicationSubproofs.hasForbiddenEqualInputs(sp("(implication9100[a,b])"), right, memory));
    ASSERT_TRUE(analyzer.implicationSubproofs.hasForbiddenEqualInputs(sp("(implication9100[a,b])"), left, memory));
    ASSERT_TRUE(analyzer.implicationSubproofs.hasForbiddenEqualInputs(sp("(implication9100[a,b])"), nested, memory));
}

TEST(rejected_subproof, equal_input_opening_refuses_without_minting_a_scope_or_park) {
    gl::tests::ProverFixture analyzer("Peano");
    gl::Memory memory;
    memory.isPartOfRecursion = true;
    memory.recursionAuxyImplId = 1;
    const gl::NameId main = memory.nameMap.encode(sp("main"));
    equalInputDefinition(analyzer, {"(=[u_1,1])", "(=[u_2,1])"});
    equalInputClass(memory, main);
    gl::ScratchArena& arena = gl::genScratchArenas().forSlot(gl::genScratchArenas().slotCount() - 1);
    gl::ScratchScope frame(arena);
    gl::WorkInstruction instruction(&arena);
    const int32_t mark = instruction.elemMark();
    instruction.addElement(sp("(=[u_a,X])"));
    instruction.addElement(sp("(=[u_b,X])"));
    instruction.commitEntity(sp("implication"), sp("(implication9100[u_a,u_b])"), sp(""), 2, mark);
    ASSERT_TRUE(analyzer.implicationSubproofs.openImplicationSubproof(instruction, 0, memory, sp("main"), sp(""))
        == gl::ImplicationSubproofOutcome::Refused);
    ASSERT_EQ(memory.integrationPrepared.count(), 0);
    ASSERT_EQ(memory.overallHashMemory.rejectedMapSubproof.count(), 0);
    ASSERT_EQ(memory.nameMap.lookup(sp("main_boundary_(implication9100[a,b])")), 0);
}

TEST(rejected_subproof, equal_input_retirement_wipes_bare_and_goal_subtrees_without_parent_proof) {
    gl::tests::ProverFixture analyzer("Peano");
    gl::Memory memory;
    memory.isPartOfRecursion = true;
    memory.recursionAuxyImplId = 1;
    const gl::NameId main = memory.nameMap.encode(sp("main"));
    equalInputDefinition(analyzer, {"(in[u_1,1])", "(in[u_2,1])"});
    const std::string payload = "(implication9100[a,b])";
    const gl::NameId bare = equalInputScope(memory, main, payload);
    const gl::NameId child = equalInputScope(memory, bare, "(implication9100[a,b])");
    const gl::NameId goal = equalInputScope(memory, main, "(in[a,N])_subproof_" + payload);
    const gl::NameId other = equalInputScope(memory, main, "(implication9100[c,d])");
    const gl::NameId nonImplication = equalInputScope(memory, main, "(and9100[a,b])");
    for (const auto scope : {main, bare, child, goal, other, nonImplication}) {
        const gl::EncodedExpression row("(in[a,N])", memory.nameMap.decode(scope));
        memory.intEncodedStatements.push_back(gl::encodeExpression(row, memory.nameMap));
    }
    analyzer.implicationSubproofs.retireEqualInputImplicationSubproofs(memory);
    ASSERT_EQ(memory.pendingWipeScopes.count(), 0);
    equalInputClass(memory, main);
    analyzer.implicationSubproofs.retireEqualInputImplicationSubproofs(memory);
    ASSERT_EQ(memory.pendingWipeScopes.count(), 2);
    ASSERT_TRUE(memory.pendingWipeScopes.contains(bare));
    ASSERT_TRUE(memory.pendingWipeScopes.contains(goal));
    ASSERT_FALSE(memory.pendingWipeScopes.contains(child));
    ASSERT_FALSE(memory.pendingWipeScopes.contains(other));
    analyzer.implicationSubproofs.retireEqualInputImplicationSubproofs(memory);
    ASSERT_EQ(memory.pendingWipeScopes.count(), 2);
    const gl::NameId roots[] = {bare, goal};
    memory.wipeSubtrees(roots, 2);
    ASSERT_EQ(memory.intEncodedStatements.size(), 3);
    ASSERT_TRUE(memory.integrationPrepared.contains(gl::mintTemplateKey(memory.templateInterner,
        memory.nameMap, payload, std::string("main"))));
    ASSERT_EQ(memory.intKnownStatements.count(), 0);
    ASSERT_EQ(memory.exprOriginMap.count(), 0);
}

TEST(rejected_subproof, equal_input_retirement_preserves_the_directly_permitted_pair) {
    gl::tests::ProverFixture analyzer("Peano");
    gl::Memory memory;
    memory.isPartOfRecursion = true;
    memory.recursionAuxyImplId = 1;
    const gl::NameId main = memory.nameMap.encode(sp("main"));
    equalInputDefinition(analyzer, {"(=[u_2,u_1])", "(in[u_1,1])"});
    equalInputScope(memory, main, "(implication9100[a,b])");
    equalInputClass(memory, main);
    analyzer.implicationSubproofs.retireEqualInputImplicationSubproofs(memory);
    ASSERT_EQ(memory.pendingWipeScopes.count(), 0);
}

TEST(rejected_subproof, equal_input_opening_preserves_ordinary_and_induction_zero_blocks) {
    gl::tests::ProverFixture analyzer("Peano");
    // Isolate scope opening from the independent discovery-scope cleanup.
    // The companion test exercises that cleanup with disintegration enabled.
    analyzer.parameters.ban_disintegration = true;
    equalInputDefinition(analyzer, {"(=[u_1,1])", "(=[u_2,2])"});
    for (const bool zeroBlock : {false, true}) {
        gl::Memory memory;
        memory.isPartOfRecursion = zeroBlock;
        memory.setVarDefSets(sp("a:(1);b:(1)"));
        const gl::NameId main = memory.nameMap.encode(sp("main"));
        const int levels[] = {0};
        const gl::TransientOrigin equalityOrigin{true, gl::OriginTag::premise, nullptr, 0};
        analyzer.statementMemory.addExprToMemoryBlock(sp("(=[a,b])"), memory, 0, 0,
            levels, 1, equalityOrigin, -1, -1, sp("main"), false);
        ASSERT_TRUE(analyzer.implicationSubproofs.hasForbiddenEqualInputs(sp("(implication9100[a,b])"), main, memory));
        gl::ScratchArena& arena = gl::genScratchArenas().forSlot(gl::genScratchArenas().slotCount() - 1);
        gl::ScratchScope frame(arena);
        gl::WorkInstruction instruction(&arena);
        const int32_t mark = instruction.elemMark();
        instruction.addElement(sp("(=[u_a,X])"));
        instruction.addElement(sp("(=[u_b,Y])"));
        instruction.commitEntity(sp("implication"), sp("(implication9100[u_a,u_b])"), sp(""), 2, mark);
        ASSERT_EQ(memory.recursionAuxyImplId, 0);
        ASSERT_TRUE(analyzer.implicationSubproofs.openImplicationSubproof(instruction, 0, memory, sp("main"), sp(""))
            == gl::ImplicationSubproofOutcome::Opened);
        // The subproof's prepared key and its reductio's (D-532).
        ASSERT_EQ(memory.integrationPrepared.count(), 2);
        ASSERT_TRUE(memory.nameMap.lookup(sp("main_boundary_contradiction_!(implication9100[a,b])")) != 0);
        ASSERT_TRUE(memory.nameMap.lookup(sp("main_boundary_(implication9100[a,b])")) != 0);
        ASSERT_EQ(memory.pendingWipeScopes.count(), 0);
    }
}

TEST(rejected_subproof, equal_input_retirement_preserves_ordinary_and_induction_zero_blocks) {
    gl::tests::ProverFixture analyzer("Peano");
    equalInputDefinition(analyzer, {"(in[u_1,1])", "(in[u_2,1])"});
    for (const bool zeroBlock : {false, true}) {
        gl::Memory memory;
        memory.isPartOfRecursion = zeroBlock;
        const gl::NameId main = memory.nameMap.encode(sp("main"));
        const gl::NameId bare = equalInputScope(memory, main, "(implication9100[a,b])");
        equalInputScope(memory, bare, "(in[a,N])_subproof_(implication9100[a,b])");
        equalInputClass(memory, main);
        ASSERT_EQ(memory.recursionAuxyImplId, 0);
        ASSERT_TRUE(analyzer.implicationSubproofs.hasForbiddenEqualInputs(sp("(implication9100[a,b])"), bare, memory));
        analyzer.implicationSubproofs.retireEqualInputImplicationSubproofs(memory);
        ASSERT_EQ(memory.pendingWipeScopes.count(), 0);
        ASSERT_EQ(memory.integrationPrepared.count(), 2);
    }
}

TEST(rejected_subproof, equal_input_exempt_opening_preserves_its_scope_during_discovery_cleanup) {
    gl::tests::ProverFixture analyzer("Peano");
    equalInputDefinition(analyzer, {"(=[u_1,1])", "(=[u_2,2])"});
    for (const bool zeroBlock : {false, true}) {
        gl::Memory memory;
        memory.isPartOfRecursion = zeroBlock;
        memory.setVarDefSets(sp("a:(1);b:(1)"));
        const gl::NameId main = memory.nameMap.encode(sp("main"));
        const int levels[] = {0};
        const gl::TransientOrigin equalityOrigin{true, gl::OriginTag::premise, nullptr, 0};
        analyzer.statementMemory.addExprToMemoryBlock(sp("(=[a,b])"), memory, 0, 0,
            levels, 1, equalityOrigin, -1, -1, sp("main"), false);
        ASSERT_TRUE(analyzer.implicationSubproofs.hasForbiddenEqualInputs(sp("(implication9100[a,b])"), main, memory));
        gl::ScratchArena& arena = gl::genScratchArenas().forSlot(gl::genScratchArenas().slotCount() - 1);
        gl::ScratchScope frame(arena);
        gl::WorkInstruction instruction(&arena);
        const int32_t mark = instruction.elemMark();
        instruction.addElement(sp("(=[u_a,X])"));
        instruction.addElement(sp("(=[u_b,Y])"));
        instruction.commitEntity(sp("implication"), sp("(implication9100[u_a,u_b])"), sp(""), 2, mark);
        ASSERT_EQ(memory.recursionAuxyImplId, 0);
        ASSERT_TRUE(analyzer.implicationSubproofs.openImplicationSubproof(instruction, 0, memory, sp("main"), sp(""))
            == gl::ImplicationSubproofOutcome::Opened);
        // The subproof's prepared key and its reductio's (D-532).
        ASSERT_EQ(memory.integrationPrepared.count(), 2);
        ASSERT_TRUE(memory.nameMap.lookup(sp("main_boundary_contradiction_!(implication9100[a,b])")) != 0);
        const gl::NameId subproof = memory.nameMap.lookup(sp("main_boundary_(implication9100[a,b])"));
        ASSERT_TRUE(subproof != 0);
        const int32_t discoveryCount = memory.pendingWipeScopes.count();
        ASSERT_TRUE(discoveryCount > 0);
        for (int32_t i = 1; i <= discoveryCount; ++i) {
            const gl::NameId discovery = memory.pendingWipeScopes.decode(i);
            ASSERT_EQ(memory.nameMap.parentOf(discovery), subproof);
            ASSERT_TRUE(memory.intValidityNamesToFilter.contains(discovery));
        }
        analyzer.implicationSubproofs.retireEqualInputImplicationSubproofs(memory);
        ASSERT_EQ(memory.pendingWipeScopes.count(), discoveryCount);
        ASSERT_FALSE(memory.pendingWipeScopes.contains(subproof));
        ASSERT_FALSE(memory.intValidityNamesToFilter.contains(subproof));
    }
}

namespace {
    /// @brief Row 76's hypothesis shape under the FTA anchor: the positive
    ///        premise 2 <= 9 and the negated premise "9 is not prime".
    const char* kGatePremise = "(preorder[1,4,7,9])";
    const char* kNegatedPremise = "!(prime[1,4,5,6,7,9])";
    const char* kReductioScope = "main_boundary_contradiction_!(prime[1,4,5,6,7,9])";

    /// @brief The hypothesis chain and its sorted bound arguments.
    void hypothesisChain(gl::StrSpan* chain, gl::StrSpan* remaining) {
        chain[0] = sp(kGatePremise);
        chain[1] = sp(kNegatedPremise);
        static const char* names[6] = { "1", "4", "5", "6", "7", "9" };
        for (int32_t i = 0; i < 6; ++i) remaining[i] = sp(names[i]);
    }
} // namespace

// negatedCompoundTakesReductio: a negated compact whose core compiles to a
// non-atomic, non-or entity takes the reductio; its positive form, an
// uncompiled core and an or-bodied core do not.
TEST(rejected_subproof, negated_compound_takes_reductio_classification) {
    gl::tests::ProverFixture analyzer("FTA");
    gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions)["or9001"] = gl::LogicalEntity("or",
        { "(=[u_1,u_2])", "(=[u_1,u_3])" }, "(or9001[u_1,u_2,u_3])", 3);
    ASSERT_TRUE(analyzer.integrationProcessor.negatedCompoundTakesReductio(sp(kNegatedPremise)));
    ASSERT_FALSE(analyzer.integrationProcessor.negatedCompoundTakesReductio(sp("(prime[1,4,5,6,7,9])")));
    ASSERT_FALSE(analyzer.integrationProcessor.negatedCompoundTakesReductio(sp("!(unknown9001[1,2])")));
    ASSERT_FALSE(analyzer.integrationProcessor.negatedCompoundTakesReductio(sp("!(or9001[1,2,3])")));
}

// deferPremiseReductio (I-326): the
// first unknown gate keys the park, whose one record is the reductio
// (category, negated signature, every gate); with every gate known, or with
// no gate, nothing parks.
TEST(rejected_subproof, defer_premise_reductio_parks_under_first_unknown_gate) {
    gl::tests::ProverFixture analyzer("FTA");
    gl::Memory memory;
    ((memory).level) = 0;
    ((memory).nameMap).encode(sp("main"));
    const gl::StrSpan gates[2] = { sp("(in[9,1])"), sp(kGatePremise) };
    makeKnown(memory, "(in[9,1])", gl::NameMap::MAIN_ID);

    ASSERT_TRUE(analyzer.implicationSubproofs.deferPremiseReductio(
        sp(kNegatedPremise), gates, 2, memory, sp("main")));
    const int64_t pk = mainParkKey(memory, kGatePremise);
    const gl::IntegrationEntryMap records = gl::admissionIntegrationRecordsAt(
        ((((memory).overallHashMemory)).rejectedMapSubproof), pk, ((memory).valueInterner));
    ASSERT_EQ(records.size(), static_cast<std::size_t>(1));
    const gl::IntInstruction& record = records.begin()->first;
    ASSERT_EQ(record.data.size(), static_cast<std::size_t>(1));
    ASSERT_EQ(((memory).valueInterner).decode(record.data[0].category), std::string("reductio"));
    ASSERT_EQ(((memory).valueInterner).decode(record.data[0].signature), std::string(kNegatedPremise));
    ASSERT_EQ(record.data[0].elements.size(), static_cast<std::size_t>(2));
    ASSERT_EQ(((memory).nameMap).lookup(sp(kReductioScope)), 0);

    makeKnown(memory, kGatePremise, gl::NameMap::MAIN_ID);
    ASSERT_FALSE(analyzer.implicationSubproofs.deferPremiseReductio(
        sp(kNegatedPremise), gates, 2, memory, sp("main")));
    ASSERT_FALSE(analyzer.implicationSubproofs.deferPremiseReductio(
        sp(kNegatedPremise), gates, 0, memory, sp("main")));
}

// prepareHypothesisPremise (I-326):
// with the positive premise unknown the negated premise's reductio parks and
// opens no scope; the positive premise's arrival drain opens it. With the
// positive premise already known it opens at once and nothing parks.
TEST(rejected_subproof, hypothesis_reductio_waits_for_positive_premise) {
    {
        gl::tests::ProverFixture analyzer("FTA");
        gl::Memory memory;
        ((memory).level) = 0;
        ((memory).nameMap).encode(sp("main"));
        gl::StrSpan chain[2];
        gl::StrSpan remaining[6];
        hypothesisChain(chain, remaining);

        analyzer.integrationProcessor.prepareHypothesisPremise(sp(kNegatedPremise), chain, 2,
            remaining, 6, memory, sp("main"));
        const int64_t pk = mainParkKey(memory, kGatePremise);
        ASSERT_TRUE(((((memory).overallHashMemory)).rejectedMapSubproof).lookup(pk) != 0);
        ASSERT_EQ(((memory).nameMap).lookup(sp(kReductioScope)), 0);

        gl::StatementArrivals::stageSubproofArrival(memory,
            makeKnown(memory, kGatePremise, gl::NameMap::MAIN_ID));
        analyzer.implicationSubproofs.drainSubproofArrivals(memory);
        ASSERT_TRUE(((((memory).overallHashMemory)).rejectedMapSubproof).lookup(pk) == 0);
        ASSERT_TRUE(((memory).nameMap).lookup(sp(kReductioScope)) != 0);
    }
    {
        gl::tests::ProverFixture analyzer("FTA");
        gl::Memory memory;
        ((memory).level) = 0;
        ((memory).nameMap).encode(sp("main"));
        makeKnown(memory, kGatePremise, gl::NameMap::MAIN_ID);
        gl::StrSpan chain[2];
        gl::StrSpan remaining[6];
        hypothesisChain(chain, remaining);

        analyzer.integrationProcessor.prepareHypothesisPremise(sp(kNegatedPremise), chain, 2,
            remaining, 6, memory, sp("main"));
        ASSERT_TRUE(((((memory).overallHashMemory)).rejectedMapSubproof).empty());
        ASSERT_TRUE(((memory).nameMap).lookup(sp(kReductioScope)) != 0);
    }
}

// hasBuiltOnListedArgument (I-293): a listed argument refuses only when its
// whole complete class at the scope is listed; an argument in no class
// stands for itself.
TEST(rejected_subproof, built_on_listed_argument_is_judged_by_its_complete_class) {
    GateFixture f;
    const gl::NameId main = f.memory.nameMap.encode(sp("main"));
    ASSERT_FALSE(f.analyzer.implicationSubproofs.hasBuiltOnListedArgument(sp("(implication9100[a,c])"), main, f.memory));
    f.memory.builtOnListedWitnesses.mint(sp("a"));
    ASSERT_TRUE(f.analyzer.implicationSubproofs.hasBuiltOnListedArgument(sp("(implication9100[a,c])"), main, f.memory));
    ASSERT_TRUE(f.analyzer.implicationSubproofs.hasBuiltOnListedArgument(sp("(implication9100[c,a])"), main, f.memory));
    ASSERT_FALSE(f.analyzer.implicationSubproofs.hasBuiltOnListedArgument(sp("(implication9100[c,d])"), main, f.memory));
    equalInputClass(f.memory, main);
    ASSERT_FALSE(f.analyzer.implicationSubproofs.hasBuiltOnListedArgument(sp("(implication9100[a,c])"), main, f.memory));
    f.memory.builtOnListedWitnesses.mint(sp("b"));
    ASSERT_TRUE(f.analyzer.implicationSubproofs.hasBuiltOnListedArgument(sp("(implication9100[a,c])"), main, f.memory));
    ASSERT_TRUE(f.analyzer.implicationSubproofs.hasBuiltOnListedArgument(sp("(implication9100[c,b])"), main, f.memory));
    ASSERT_EQ(f.memory.exprOriginMap.count(), 0);
}

// openImplicationSubproof (I-293): an implication over a listed witness that
// is class-equal to an unlisted one is not refused; it parks on its gate and
// opens once the gate is known. With every class member listed it is refused.
TEST(rejected_subproof, open_judges_a_built_on_listed_argument_by_its_class) {
    const std::string listed = "int_0_lev_6_123";
    const std::string other = "int_0_lev_6_124";
    const std::string gate = "(in2[" + listed + ",2,3])";
    for (const bool otherListed : {false, true}) {
        GateFixture f;
        const gl::NameId main = f.memory.nameMap.encode(sp("main"));
        f.memory.builtOnListedWitnesses.mint(gl::StrSpan(listed));
        if (otherListed) f.memory.builtOnListedWitnesses.mint(gl::StrSpan(other));
        gl::EquivalenceClass equality;
        equality.setMembersFromNames({listed, other}, f.memory.nameMap);
        seedFixtureClassEvidence(f.memory, equality, sp("main"));
        f.memory.assignClassesById(main, {equality});
        seedFixtureCanonicalClasses(f.memory);
        gl::WorkInstruction wi(f.lb);
        buildGatedImplicationOver(wi, listed);
        if (otherListed) {
            makeKnown(f.memory, gate.c_str(), gl::NameMap::MAIN_ID);
            ASSERT_TRUE(f.analyzer.implicationSubproofs.openImplicationSubproof(wi, 0, f.memory, sp("main"), gl::StrSpan(f.goal))
                        == gl::ImplicationSubproofOutcome::Refused);
            ASSERT_EQ(f.memory.integrationPrepared.count(), 0);
            continue;
        }
        ASSERT_TRUE(f.analyzer.implicationSubproofs.openImplicationSubproof(wi, 0, f.memory, sp("main"), gl::StrSpan(f.goal))
                    == gl::ImplicationSubproofOutcome::Parked);
        makeKnown(f.memory, gate.c_str(), gl::NameMap::MAIN_ID);
        ASSERT_TRUE(f.analyzer.implicationSubproofs.openImplicationSubproof(wi, 0, f.memory, sp("main"), gl::StrSpan(f.goal))
                    == gl::ImplicationSubproofOutcome::Opened);
        // The opening records the subproof's own goal: the one goal it
        // places in toBeProved at its scope.
        ASSERT_EQ(f.memory.subproofGoals.count(), 1);
        const gl::NameId scope = static_cast<gl::NameId>(f.memory.subproofGoals.decodeKey(1));
        const int64_t* goal = f.memory.subproofGoals.find(static_cast<int64_t>(scope));
        ASSERT_TRUE(goal != nullptr);
        ASSERT_TRUE(f.memory.intToBeProved.lookup(
            gl::packStatementKey(static_cast<gl::NameId>(*goal), scope)) != 0);
    }
}

namespace {
    /// @brief The history lines of one (expression, scope) key, decoded.
    std::vector<std::pair<std::string, std::vector<gl::ExpressionWithValidity>>>
    historyOf(const gl::Memory& m, const std::string& expression, const std::string& scope) {
        for (const auto& row : gl::decodeOriginMapSorted(m.exprOriginMap, m.originInterner))
            if (row.first.first == expression && row.first.second == scope) return row.second;
        return {};
    }

    /// @brief A block with class {int_0_lev_5_11, int_0_lev_5_9} at main and
    ///        the subproof scope of `(implication121[2,R,int_0_lev_5_9])`
    ///        below it, its class view built.
    struct RespelledSubproof {
        gl::tests::ProverFixture analyzer{ "Peano" };
        gl::Memory memory;
        gl::NameId scope = 0;
        std::string scopeName;
        RespelledSubproof() {
            analyzer.parameters.trackHistory = true;
            const gl::NameId main = memory.nameMap.encode(sp("main"));
            scope = memory.nameMap.encodePush(main, sp("(implication121[2,R,int_0_lev_5_9])"));
            scopeName = memory.nameMap.decode(scope);
            gl::EquivalenceClass equality;
            equality.setMembersFromNames({ "int_0_lev_5_11", "int_0_lev_5_9" }, memory.nameMap);
            seedFixtureClassEvidence(memory, equality, sp("main"));
            memory.assignClassesById(main, { equality });
            seedFixtureCanonicalClasses(memory);
        }
    };
}

// writeEqualityBridge: one equality1 line from the source spelling to the
// result spelling at the result scope, the source first, then one justifier
// (=[source argument,result argument]) per differing pair, each pair once,
// cited where its row lives (main, above the scope).
TEST(rejected_subproof, equality_bridge_cites_source_then_each_pair_once) {
    RespelledSubproof f;
    f.analyzer.proofHistory.writeEqualityBridge(f.memory, sp("(in3[int_0_lev_5_9,int_0_lev_5_9,c,+])"),
        f.scope, sp("(in3[int_0_lev_5_11,int_0_lev_5_11,c,+])"), gl::StrSpan(f.scopeName));
    const auto lines = historyOf(f.memory, "(in3[int_0_lev_5_9,int_0_lev_5_9,c,+])", f.scopeName);
    ASSERT_EQ(lines.size(), static_cast<std::size_t>(1));
    ASSERT_EQ(lines[0].first, std::string("equality1"));
    ASSERT_EQ(lines[0].second.size(), static_cast<std::size_t>(2));
    ASSERT_EQ(lines[0].second[0].original, std::string("(in3[int_0_lev_5_11,int_0_lev_5_11,c,+])"));
    ASSERT_EQ(lines[0].second[0].validityName, f.scopeName);
    ASSERT_EQ(lines[0].second[1].original, std::string("(=[int_0_lev_5_11,int_0_lev_5_9])"));
    ASSERT_EQ(lines[0].second[1].validityName, std::string("main"));
}

// dischargeToBeProved, subproof closure: a goal a class rewrite respelled
// closes in the new spelling; the closing row cites the subproof's own goal,
// which an equality1 line derives from the closing statement. Without a
// respelling the closing row cites the closing statement, and no line is
// written.
TEST(rejected_subproof, closure_of_a_respelled_goal_cites_the_subproof_goal) {
    for (const bool respelled : { true, false }) {
        RespelledSubproof f;
        const std::string own = "(in2[2,r,int_0_lev_5_9])";
        const std::string closing = respelled ? "(in2[2,r,int_0_lev_5_11])" : own;
        const gl::NameId ownId = f.memory.nameMap.encode(gl::StrSpan(own));
        f.memory.subproofGoals.upsert(static_cast<int64_t>(f.scope), static_cast<int64_t>(ownId));
        const gl::NameId closingId = f.memory.nameMap.encode(gl::StrSpan(closing));
        f.memory.intToBeProved.assignSet(gl::packStatementKey(closingId, f.scope), nullptr, 0);
        const gl::IntEncodedExpr source = gl::encodeExpression(
            gl::EncodedExpression(closing, f.scopeName), f.memory.nameMap);
        f.memory.intKnownStatements.insert(gl::StatementKey{ source.originalId, source.validityId },
            gl::StatementFlags{ true, false });
        f.memory.intStatementLevelsMap.insertSorted(
            gl::packStatementKey(source.originalId, source.validityId), 3);
        f.memory.intLocalEncodedStatementsDelta.push_back(source);

        f.analyzer.goalProcessor.dischargeToBeProved(f.memory, 0, f.memory.sameIterationInternalMail);

        const gl::Mail mail = gl::makeHeapMail(f.memory.sameIterationInternalMail,
            f.memory.nameMap, f.memory.originInterner);
        const auto originIt = mail.exprOriginMap.find(
            gl::ExpressionWithValidity("(implication121[2,R,int_0_lev_5_9])", "main"));
        ASSERT_TRUE(originIt != mail.exprOriginMap.end());
        ASSERT_EQ(originIt->second.size(), static_cast<std::size_t>(1));
        ASSERT_EQ(originIt->second[0].first, std::string("validity name"));
        ASSERT_EQ(originIt->second[0].second.size(), static_cast<std::size_t>(1));
        ASSERT_EQ(originIt->second[0].second[0].original, own);
        ASSERT_EQ(originIt->second[0].second[0].validityName, f.scopeName);
        const auto bridge = historyOf(f.memory, own, f.scopeName);
        ASSERT_EQ(bridge.size(), static_cast<std::size_t>(respelled ? 1 : 0));
        if (respelled) {
            ASSERT_EQ(bridge[0].first, std::string("equality1"));
            ASSERT_EQ(bridge[0].second[0].original, closing);
            ASSERT_EQ(bridge[0].second[1].original, std::string("(=[int_0_lev_5_11,int_0_lev_5_9])"));
        }
        ASSERT_TRUE(f.memory.pendingWipeScopes.contains(f.scope));
    }
}

// dischargeToBeProved, subproof closure: a subproof opened below another scope
// deposits its compact at that scope, the subproof scope's parent, never at
// main, since its proof may use that scope's facts.
TEST(rejected_subproof, closure_of_a_nested_subproof_lands_at_its_parent) {
    gl::tests::ProverFixture analyzer("Peano");
    analyzer.parameters.trackHistory = true;
    gl::Memory memory;
    const gl::NameId main = memory.nameMap.encode(sp("main"));
    const gl::NameId opening = memory.nameMap.encodePush(main, sp("contradiction_!(in2[a,b,c])"));
    const gl::NameId scope = memory.nameMap.encodePush(opening, sp("(implication121[2,R,q])"));
    const std::string openingName = memory.nameMap.decode(opening);
    const std::string scopeName = memory.nameMap.decode(scope);
    seedFixtureCanonicalClasses(memory);
    const std::string goal = "(in2[2,r,q])";
    const gl::NameId goalId = memory.nameMap.encode(gl::StrSpan(goal));
    memory.subproofGoals.upsert(static_cast<int64_t>(scope), static_cast<int64_t>(goalId));
    memory.intToBeProved.assignSet(gl::packStatementKey(goalId, scope), nullptr, 0);
    const gl::IntEncodedExpr source = gl::encodeExpression(
        gl::EncodedExpression(goal, scopeName), memory.nameMap);
    memory.intKnownStatements.insert(gl::StatementKey{ source.originalId, source.validityId },
        gl::StatementFlags{ true, false });
    memory.intStatementLevelsMap.insertSorted(
        gl::packStatementKey(source.originalId, source.validityId), 3);
    memory.intLocalEncodedStatementsDelta.push_back(source);

    analyzer.goalProcessor.dischargeToBeProved(memory, 0, memory.sameIterationInternalMail);

    const gl::Mail mail = gl::makeHeapMail(memory.sameIterationInternalMail,
        memory.nameMap, memory.originInterner);
    ASSERT_TRUE(mail.exprOriginMap.find(
        gl::ExpressionWithValidity("(implication121[2,R,q])", "main")) == mail.exprOriginMap.end());
    const auto originIt = mail.exprOriginMap.find(
        gl::ExpressionWithValidity("(implication121[2,R,q])", openingName));
    ASSERT_TRUE(originIt != mail.exprOriginMap.end());
    ASSERT_EQ(originIt->second.size(), static_cast<std::size_t>(1));
    ASSERT_EQ(originIt->second[0].first, std::string("validity name"));
    ASSERT_EQ(originIt->second[0].second[0].original, goal);
    ASSERT_EQ(originIt->second[0].second[0].validityName, scopeName);
    ASSERT_TRUE(memory.pendingWipeScopes.contains(scope));
}
