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
// Unit tests for the contradiction-scope demand
// (D-560): the derived fact index, the fact keys,
// the candidate verdict with its park and revival, the arrival staging, the
// scope registration and wipe, and the drain end to end on the shape of FTA2
// row 10 (`minimum[N,+,M,n]` against `!in[n,N]`).
#include "../memory.hpp"
#include "../logical_block/statement_arrivals.hpp"
#include "../logical_block/statement_registry_operations.hpp"
#include "../logical_block/contradiction_demand.hpp"
#include "test_harness.hpp"
#include "prover_fixture.hpp"
#include "compiled_definition_fixture.hpp"

#include "../prover.hpp"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

namespace gl::tests {
/// @brief Test access to the private steps of the contradiction-scope demand.
struct ContradictionDemandFixture {
    static bool participates(const ContradictionDemand& d, const Memory& mb, NameId scope) {
        return d.participates(mb, scope);
    }
    static StrSpan negatedInput(StrSpan fact, ScratchArena& strings) {
        return ContradictionDemand::negatedInput(fact, strings);
    }
    static void fileFact(ContradictionDemand& d, Memory& mb, int64_t fact) { d.fileFact(mb, fact); }
    static void matchFact(ContradictionDemand& d, Memory& mb, NameId scope, int64_t fact,
                          int32_t first, int32_t last) {
        d.matchFact(mb, scope, fact, first, last);
    }
    static void decideCandidate(ContradictionDemand& d, Memory& mb, NameId scope, int32_t chain,
                                StrSpan input) {
        d.decideCandidate(mb, scope, chain, input);
    }
};
}

namespace {

    using Verdict = gl::ImplicationSubproofs::ContradictionDemandVerdict;
    using Fixture = gl::tests::ContradictionDemandFixture;

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

    /// @brief Make @p text a known row at @p scope without staging anything.
    int64_t makeKnown(gl::Memory& m, const char* text, gl::NameId scope) {
        const int64_t key = gl::packStatementKey(m.nameMap.encode(sp(text)), scope);
        gl::upsertStatementKey(m.intKnownStatements, key, /*local=*/true);
        return key;
    }

    /// @brief Make @p text a known row at @p scope through the publish door,
    ///        which stages its arrivals.
    int64_t publishKnown(gl::Memory& m, const char* text, gl::NameId scope) {
        const int64_t key = gl::packStatementKey(m.nameMap.encode(sp(text)), scope);
        gl::StatementRegistryOperations::publishKnownStatementKey(m, key, /*local=*/true);
        return key;
    }

    /// @brief Open a registered contradiction scope under main, as
    ///        `prepareNegatedCompoundContradiction` does, without its seed.
    gl::NameId openScope(gl::tests::ProverFixture& ea, const char* payload) {
        const gl::NameId scope = ea.body.nameMap.encodePush(gl::NameMap::MAIN_ID, sp(payload));
        ea.canonicalizer.initializeCanonicalScope(ea.body, scope);
        ea.body.lbMemory.contradictionScopes.insert(static_cast<int64_t>(scope), 0);
        return scope;
    }

    /// @brief The text of a scope.
    std::string scopeText(gl::Memory& m, gl::NameId scope) { return m.nameMap.decode(scope); }

    /// @brief The records parked under clean @p awaited at @p scope.
    gl::IntegrationEntryMap parksUnder(gl::Memory& m, const char* awaited, gl::NameId scope) {
        const int64_t pk = gl::mintTemplateKey(m.templateInterner, m.nameMap,
            std::string(awaited), scopeText(m, scope));
        return gl::admissionIntegrationRecordsAt(m.overallHashMemory.rejectedMapSubproof, pk,
            m.valueInterner);
    }

    /// @brief The staged demands as decoded (compound, scope) texts.
    std::vector<std::pair<std::string, std::string>> stagedDemands(gl::Memory& m) {
        std::vector<std::pair<std::string, std::string>> out;
        for (int32_t i = 0; i < m.pendingContradictionDemands.size(); ++i) {
            const gl::StatementKey k = gl::Codec<gl::StatementKey>::decode(m.pendingContradictionDemands[i]);
            out.emplace_back(m.templateInterner.decodeView(k.orig).toStdString(),
                             m.nameMap.decode(k.validity));
        }
        return out;
    }

    /// @brief A test and entity `(and9101[u_1,u_2,u_3])`: gated by
    ///        `(in2[u_1,u_2,u_3])`, with an implication element and an element
    ///        holding a bound argument, neither of which gates.
    void registerTestAnd(gl::tests::ProverFixture& ea) {
        gl::CompiledDefinitionFixture::compiledExpressions(ea.definitions)["and9101"] = gl::LogicalEntity("and",
            { "(in2[u_1,u_2,u_3])", "(implication90[u_1,u_2])", "(in[1,u_3])" },
            "(and9101[u_1,u_2,u_3])", 3);
    }

    /// @brief The facts the index files under @p key, sorted.
    std::vector<int64_t> filedUnder(const gl::ContradictionFactIndex& idx, const std::string& key) {
        std::vector<int64_t> out;
        idx.factsWithKey(gl::StrSpan(key), [&](int64_t f) { out.push_back(f); });
        std::sort(out.begin(), out.end());
        return out;
    }

} // namespace

// The fact index files a fact once under each key it is given, answers every
// key with exactly its facts, and forgets everything at clear.
TEST(contradiction_demand, fact_index_files_and_answers_by_key) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    gl::ContradictionFactIndex idx(&lb);
    ASSERT_FALSE(idx.built());
    ASSERT_TRUE(idx.addFact(5));
    ASSERT_FALSE(idx.addFact(5));
    idx.fileUnder(sp("k1"), 5);
    idx.fileUnder(sp("k2"), 5);
    ASSERT_TRUE(idx.addFact(7));
    idx.fileUnder(sp("k1"), 7);
    ASSERT_TRUE(filedUnder(idx, "k1") == (std::vector<int64_t>{ 5, 7 }));
    ASSERT_TRUE(filedUnder(idx, "k2") == (std::vector<int64_t>{ 5 }));
    ASSERT_TRUE(filedUnder(idx, "k3").empty());
    ASSERT_TRUE(idx.containsFact(5));
    ASSERT_FALSE(idx.containsFact(6));
    ASSERT_EQ(idx.factCount(), 2);
    idx.markBuilt();
    ASSERT_TRUE(idx.built());
    ASSERT_TRUE(idx.liveBytes() > 0);
    idx.clear();
    ASSERT_FALSE(idx.built());
    ASSERT_EQ(idx.factCount(), 0);
    ASSERT_TRUE(filedUnder(idx, "k1").empty());
}

// A fact is filed under the sign-flipped masked key at each (1) slot, and at
// each pair of (1) slots beside a third argument: exactly the key of a chain
// whose head negates it.
TEST(contradiction_demand, fact_keys_are_the_keys_of_the_negating_heads) {
    gl::tests::ProverFixture ea("Peano");
    gl::ScratchArena& strings = gl::scratchArenas().forSlot(gl::scratchArenas().slotCount() - 1);
    gl::ScratchScope frame(strings);
    const auto keysOf = [&](const char* fact) {
        std::vector<std::string> out;
        ea.contradictionDemand.forEachFactKey(sp(fact), strings,
            [&](gl::StrSpan k) { out.push_back(k.toStdString()); });
        return out;
    };
    ASSERT_TRUE(keysOf("(in[9,1])") == (std::vector<std::string>{ std::string("!in,\x01,u_1") }));
    ASSERT_TRUE(keysOf("!(in2[a,b,c])") == (std::vector<std::string>{
        std::string("in2,\x01,u_b,u_c"), std::string("in2,u_a,\x01,u_c"), std::string("in2,\x01,\x01,u_c") }));
    ASSERT_TRUE(keysOf("(noSuchOperatorAnywhere[x])").empty());

    install(ea, "(>[1](in3[1,u_7,u_8,u_4])!(in[1,u_1]))");
    char buf[gl::kMaxMaskedHeadKeyBytes];
    const gl::StrSpan chainKey = ea.integrationProcessor.maskedKeyOfChain(ea.body, 1, buf,
        gl::kMaxMaskedHeadKeyBytes);
    ASSERT_EQ(chainKey.toStdString(), keysOf("(in[9,1])")[0]);
}

// Only a positive instance of an and entity is demandable; its gate elements
// are its non-implication elements whose every argument it fixes.
TEST(contradiction_demand, demandable_compounds_and_their_gate_elements) {
    gl::tests::ProverFixture ea("Peano");
    registerTestAnd(ea);
    ASSERT_TRUE(ea.implicationSubproofs.isDemandableCompound(sp("(and9101[x,y,z])")));
    ASSERT_TRUE(ea.implicationSubproofs.isDemandableCompound(sp("(and9101[u_x,u_y,u_z])")));
    ASSERT_FALSE(ea.implicationSubproofs.isDemandableCompound(sp("!(and9101[x,y,z])")));
    ASSERT_FALSE(ea.implicationSubproofs.isDemandableCompound(sp("(in[x,y])")));
    ASSERT_FALSE(ea.implicationSubproofs.isDemandableCompound(sp("(noSuchOperatorAnywhere[x])")));

    gl::ScratchArena& strings = gl::scratchArenas().forSlot(gl::scratchArenas().slotCount() - 1);
    gl::ScratchScope frame(strings);
    gl::StrSpan gates[4];
    ASSERT_EQ(ea.implicationSubproofs.compoundGateElements(sp("(and9101[10,1,11])"), strings, gates, 4), 1);
    ASSERT_EQ(gates[0].toStdString(), std::string("(in2[10,1,11])"));
}

// The verdict: every premise known drops; a lone unknown compound waits for
// its first unknown gate element, is dropped when its gate holds only above
// the scope, and is ready when a gate element holds at the scope itself.
TEST(contradiction_demand, verdict_on_a_lone_unknown_compound) {
    gl::tests::ProverFixture ea("Peano");
    registerTestAnd(ea);
    gl::Memory& m = ea.body;
    const gl::NameId s = openScope(ea, "contradiction_!(in[q,q])");
    int32_t ready = 7;

    makeKnown(m, "(in[a,b])", gl::NameMap::MAIN_ID);
    const gl::StrSpan known[] = { sp("(in[a,b])") };
    ASSERT_TRUE(ea.implicationSubproofs.deferContradictionDemand(known, 1, m, s, ready) == Verdict::Dropped);
    ASSERT_EQ(ready, -1);

    const gl::StrSpan lone[] = { sp("(and9101[x,y,z])") };
    ASSERT_TRUE(ea.implicationSubproofs.deferContradictionDemand(lone, 1, m, s, ready) == Verdict::Parked);
    const gl::IntegrationEntryMap parks = parksUnder(m, "(in2[x,y,z])", s);
    ASSERT_EQ(parks.size(), static_cast<std::size_t>(1));
    const gl::IntInstruction& record = parks.begin()->first;
    ASSERT_EQ(record.data.size(), static_cast<std::size_t>(1));
    ASSERT_EQ(m.valueInterner.decode(record.data[0].category), std::string("contradiction_demand"));
    ASSERT_EQ(record.data[0].elements.size(), static_cast<std::size_t>(1));
    ASSERT_EQ(m.valueInterner.decode(record.data[0].elements[0]), std::string("(and9101[x,y,z])"));

    makeKnown(m, "(in2[x,y,z])", gl::NameMap::MAIN_ID);
    ASSERT_TRUE(ea.implicationSubproofs.deferContradictionDemand(lone, 1, m, s, ready) == Verdict::Dropped);

    makeKnown(m, "(in2[p,q,r])", s);
    const gl::StrSpan gated[] = { sp("(and9101[p,q,r])") };
    ASSERT_TRUE(ea.implicationSubproofs.deferContradictionDemand(gated, 1, m, s, ready) == Verdict::Ready);
    ASSERT_EQ(ready, 0);
}

// Two unknown premises: the candidate waits for the one ordinary derivation
// must supply, not for the compound; its arrival revives it, it then waits for
// the compound's gate, and the gate's arrival at the scope stages the demand.
TEST(contradiction_demand, park_then_revival_stages_the_demand) {
    gl::tests::ProverFixture ea("Peano");
    registerTestAnd(ea);
    gl::Memory& m = ea.body;
    const gl::NameId s = openScope(ea, "contradiction_!(in[q,q])");
    int32_t ready = -1;
    const gl::StrSpan premises[] = { sp("(and9101[x,y,z])"), sp("(in[c,d])") };
    ASSERT_TRUE(ea.implicationSubproofs.deferContradictionDemand(premises, 2, m, s, ready) == Verdict::Parked);
    ASSERT_EQ(parksUnder(m, "(in[c,d])", s).size(), static_cast<std::size_t>(1));
    ASSERT_TRUE(parksUnder(m, "(and9101[x,y,z])", s).empty());

    publishKnown(m, "(in[c,d])", gl::NameMap::MAIN_ID);
    ea.implicationSubproofs.drainSubproofArrivals(m);
    ASSERT_TRUE(parksUnder(m, "(in[c,d])", s).empty());
    ASSERT_EQ(parksUnder(m, "(in2[x,y,z])", s).size(), static_cast<std::size_t>(1));
    ASSERT_EQ(m.pendingContradictionDemands.size(), 0);

    publishKnown(m, "(in2[x,y,z])", s);
    ea.implicationSubproofs.drainSubproofArrivals(m);
    ASSERT_TRUE(parksUnder(m, "(in2[x,y,z])", s).empty());
    const auto staged = stagedDemands(m);
    ASSERT_EQ(staged.size(), static_cast<std::size_t>(1));
    ASSERT_EQ(staged[0].first, std::string("(and9101[x,y,z])"));
    ASSERT_EQ(staged[0].second, scopeText(m, s));
}

// A staged demand names the compound by template id and the scope by name id.
TEST(contradiction_demand, stage_contradiction_demand_queues_compound_and_scope) {
    gl::tests::ProverFixture ea("Peano");
    gl::Memory& m = ea.body;
    const gl::NameId s = m.nameMap.encodePush(gl::NameMap::MAIN_ID, sp("contradiction_!(in[q,q])"));
    gl::ImplicationSubproofs::stageContradictionDemand(m, sp("(and0[1,2,3,4,5,6,7,8])"), s);
    const auto staged = stagedDemands(m);
    ASSERT_EQ(staged.size(), static_cast<std::size_t>(1));
    ASSERT_EQ(staged[0].first, std::string("(and0[1,2,3,4,5,6,7,8])"));
    ASSERT_EQ(staged[0].second, scopeText(m, s));
}

// No arrival is staged while no contradiction scope is registered.
TEST(contradiction_demand, arrivals_are_staged_only_with_a_registered_scope) {
    gl::tests::ProverFixture ea("Peano");
    gl::Memory& m = ea.body;
    const int64_t before = publishKnown(m, "(in[a,b])", gl::NameMap::MAIN_ID);
    (void)before;
    ASSERT_EQ(m.pendingContradictionArrivals.size(), 0);
    openScope(ea, "contradiction_!(in[q,q])");
    const int64_t after = publishKnown(m, "(in[c,d])", gl::NameMap::MAIN_ID);
    ASSERT_EQ(m.pendingContradictionArrivals.size(), 1);
    ASSERT_EQ(m.pendingContradictionArrivals[0], after);
}

// A negated compound's reductio registers its contradiction scope, once.
TEST(contradiction_demand, negated_compound_contradiction_registers_its_scope) {
    gl::tests::ProverFixture ea("Peano");
    gl::Memory& m = ea.body;
    ea.implicationSubproofs.prepareNegatedCompoundContradiction(sp("!(and0[1,2,3,4,5,6,7,8])"), m, sp("main"));
    ASSERT_EQ(m.lbMemory.contradictionScopes.count(), 1);
    const gl::NameId s = static_cast<gl::NameId>(m.lbMemory.contradictionScopes.decodeKey(1));
    ASSERT_EQ(m.nameMap.parentOf(s), gl::NameMap::MAIN_ID);
    ASSERT_EQ(*m.lbMemory.contradictionScopes.find(static_cast<int64_t>(s)), 0);
    ea.implicationSubproofs.prepareNegatedCompoundContradiction(sp("!(and0[1,2,3,4,5,6,7,8])"), m, sp("main"));
    ASSERT_EQ(m.lbMemory.contradictionScopes.count(), 1);
}

// A scope takes part while it and its ancestors are not filtered for wiping;
// the wipe takes it out of the registry, and its parks with it.
TEST(contradiction_demand, scope_close_drops_the_scope_and_its_parks) {
    gl::tests::ProverFixture ea("Peano");
    registerTestAnd(ea);
    gl::Memory& m = ea.body;
    const gl::NameId s = openScope(ea, "contradiction_!(in[q,q])");
    const gl::NameId inner = m.nameMap.encodePush(s, sp("inner"));
    ASSERT_TRUE(Fixture::participates(ea.contradictionDemand, m, s));
    ASSERT_TRUE(Fixture::participates(ea.contradictionDemand, m, inner));
    m.intValidityNamesToFilter.mint(s);
    ASSERT_FALSE(Fixture::participates(ea.contradictionDemand, m, s));
    ASSERT_FALSE(Fixture::participates(ea.contradictionDemand, m, inner));

    int32_t ready = -1;
    const gl::StrSpan lone[] = { sp("(and9101[x,y,z])") };
    ASSERT_TRUE(ea.implicationSubproofs.deferContradictionDemand(lone, 1, m, s, ready) == Verdict::Parked);
    ASSERT_EQ(parksUnder(m, "(in2[x,y,z])", s).size(), static_cast<std::size_t>(1));
    m.wipeSubtrees(&s, 1);
    ASSERT_TRUE(m.lbMemory.contradictionScopes.empty());
    ASSERT_TRUE(parksUnder(m, "(in2[x,y,z])", s).empty());
}

// The head-match input of a fact is its negation, every argument u_-prefixed.
TEST(contradiction_demand, negated_input_flips_the_sign_and_prefixes) {
    gl::ScratchArena& strings = gl::scratchArenas().forSlot(gl::scratchArenas().slotCount() - 1);
    gl::ScratchScope frame(strings);
    ASSERT_EQ(Fixture::negatedInput(sp("(in[9,1])"), strings).toStdString(), std::string("!(in[u_9,u_1])"));
    ASSERT_EQ(Fixture::negatedInput(sp("!(in[9,1])"), strings).toStdString(), std::string("(in[u_9,u_1])"));
}

// fileFact files a known fact under its keys once; matchFact and
// decideCandidate leave a chain without a demandable premise and a chain with
// an unbound premise alone, and park a determined candidate.
TEST(contradiction_demand, file_match_and_decide) {
    gl::tests::ProverFixture ea("Peano");
    registerTestAnd(ea);
    gl::Memory& m = ea.body;
    const gl::NameId s = openScope(ea, "contradiction_!(in[q,q])");
    const int64_t fact = makeKnown(m, "(in[9,1])", gl::NameMap::MAIN_ID);
    Fixture::fileFact(ea.contradictionDemand, m, fact);
    Fixture::fileFact(ea.contradictionDemand, m, fact);
    ASSERT_EQ(m.lbMemory.contradictionFacts.factCount(), 1);
    ASSERT_TRUE(filedUnder(m.lbMemory.contradictionFacts, std::string("!in,\x01,u_1"))
        == (std::vector<int64_t>{ fact }));

    install(ea, "(>[1](in3[1,u_7,u_8,u_4])!(in[1,u_1]))");          // no demandable premise
    install(ea, "(>[1,2](and9101[1,2,u_3])!(in[1,u_1]))");         // premise left unbound
    install(ea, "(>[1](and9101[1,u_5,u_6])!(in[1,u_1]))");         // determined
    ea.integrationProcessor.ensureOriginalsHeadIndexBuilt(m);
    Fixture::matchFact(ea.contradictionDemand, m, s, fact, 1, 2);
    ASSERT_TRUE(m.overallHashMemory.rejectedMapSubproof.empty());
    Fixture::matchFact(ea.contradictionDemand, m, s, fact, 1, 3);
    ASSERT_EQ(parksUnder(m, "(in2[9,5,6])", s).size(), static_cast<std::size_t>(1));

    gl::ScratchArena& strings = gl::scratchArenas().forSlot(gl::scratchArenas().slotCount() - 1);
    gl::ScratchScope frame(strings);
    makeKnown(m, "(in2[9,5,6])", s);
    Fixture::decideCandidate(ea.contradictionDemand, m, s, 3, Fixture::negatedInput(sp("(in[9,1])"), strings));
    const auto staged = stagedDemands(m);
    ASSERT_EQ(staged.size(), static_cast<std::size_t>(1));
    ASSERT_EQ(staged[0].first, std::string("(and9101[9,5,6])"));
}

// The drain end to end on the stalled shape of FTA2 row 10 (`minimum` is an
// FTA2 definition): inside the
// contradiction scope, n is in M; `minimum[N,+,M,n]` would give `!(in[n,N])`
// against the known `(in[n,N])`, so the scope demands the minimum.
TEST(contradiction_demand, drain_demands_the_minimum_of_the_stalled_step) {
    gl::tests::ProverFixture ea("FTA");
    gl::Memory& m = ea.body;
    // A block level and witness counters, as every proving block has: the
    // integration of the demanded compound names its witnesses with them.
    m.level = 1;
    m.startInt = 0;
    m.startIntPi = 0;
    install(ea, "(>[1](minimum[u_1,u_4,u_10,1])!(in[1,u_1]))");
    makeKnown(m, "(in[9,1])", gl::NameMap::MAIN_ID);
    const gl::NameId s = openScope(ea, "contradiction_!(preorder[1,4,11,9])");
    makeKnown(m, "(in[9,10])", s);

    const int32_t chainsAtEntry = m.overallHashMemory.originals.count();
    ea.contradictionDemand.drainContradictionDemands(m);
    ASSERT_EQ(*m.lbMemory.contradictionScopes.find(static_cast<int64_t>(s)), 1);
    // The floor covers the chains present at entry; the rules the
    // demanded integration installs wait for the next drain.
    ASSERT_EQ(m.overallHashMemory.contradictionChainMark, chainsAtEntry);
    ASSERT_TRUE(m.overallHashMemory.originals.count() > chainsAtEntry);
    ASSERT_TRUE(m.lbMemory.contradictionFacts.built());
    ASSERT_EQ(m.pendingContradictionDemands.size(), 0);
    ASSERT_TRUE(m.integrationPreparedMarker.contains(gl::mintTemplateKey(m.templateInterner, m.nameMap,
        std::string("(minimum[1,4,10,9])"), scopeText(m, s))));
}

// A chain installed after the scope was served meets the filed facts through
// its masked key; with no scope left the drain frees the fact index.
TEST(contradiction_demand, drain_matches_a_new_chain_and_frees_the_index_when_idle) {
    gl::tests::ProverFixture ea("FTA");
    gl::Memory& m = ea.body;
    // A block level and witness counters, as every proving block has: the
    // integration of the demanded compound names its witnesses with them.
    m.level = 1;
    m.startInt = 0;
    m.startIntPi = 0;
    makeKnown(m, "(in[9,1])", gl::NameMap::MAIN_ID);
    const gl::NameId s = openScope(ea, "contradiction_!(preorder[1,4,11,9])");
    makeKnown(m, "(in[9,10])", s);
    ea.contradictionDemand.drainContradictionDemands(m);
    ASSERT_FALSE(m.integrationPreparedMarker.contains(gl::mintTemplateKey(m.templateInterner, m.nameMap,
        std::string("(minimum[1,4,10,9])"), scopeText(m, s))));

    install(ea, "(>[1](minimum[u_1,u_4,u_10,1])!(in[1,u_1]))");
    ea.contradictionDemand.drainContradictionDemands(m);
    ASSERT_TRUE(m.integrationPreparedMarker.contains(gl::mintTemplateKey(m.templateInterner, m.nameMap,
        std::string("(minimum[1,4,10,9])"), scopeText(m, s))));

    m.wipeSubtrees(&s, 1);
    ea.contradictionDemand.drainContradictionDemands(m);
    ASSERT_FALSE(m.lbMemory.contradictionFacts.built());
    ASSERT_EQ(m.lbMemory.contradictionFacts.factCount(), 0);
}

// The owner-less sweep renumbers the surviving chains in order; the demand's
// chain floor becomes the number of survivors below it, and teardown clears it.
TEST(contradiction_demand, owner_less_sweep_remaps_the_chain_floor) {
    gl::tests::ProverFixture ea("Peano");
    gl::HashMemory& hm = ea.body.overallHashMemory;
    const std::string r1 = "(>[1](in3[1,u_7,u_8,u_4])(in[1,u_5]))";
    const std::string r2 = "(>[1](in2[1,u_7,u_6])(in[1,u_5]))";
    install(ea, r1);
    install(ea, r2);
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 4 << 20, 1 << 18 });
    gl::LbArena sets(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::ColdHashSet<gl::BytesKeyStore> visited(&sets, &d);
    gl::ColdHashSet<gl::PodKeyStore<int64_t>> dropSet(&sets, &d);
    gl::ScratchArena& gArena = gl::genScratchArenas().forSlot(gl::genScratchArenas().slotCount() - 1);
    hm.contradictionChainMark = 2;
    ASSERT_TRUE(ea.hashRuleInstaller.removeRuleFromHashMemory(ea.body, hm, gl::StrSpan(r1),
        sp("main"), visited, dropSet));
    gl::HashRuleInstaller::finishRuleRemovals(hm, dropSet, gArena);
    ASSERT_EQ(hm.originals.count(), 1);
    ASSERT_EQ(hm.contradictionChainMark, 1);
    hm.clear();
    ASSERT_EQ(hm.contradictionChainMark, 0);
}
