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
#include "test_harness.hpp"
#include "prover_fixture.hpp"
#include "compiled_definition_fixture.hpp"
#include "compact_rule_processor_fixture.hpp"
#include "../compiler/integration_rule_compiler.hpp"
#include "../compiler/implication_builder.hpp"
#include "../logical_block/flat_integration_walk.hpp"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <vector>

namespace {

std::string coreName(const std::string& expression) {
    const gl::StrSpan core = gl::extractExpressionUniversalSpan(gl::StrSpan(expression));
    return std::string(core.ptr, static_cast<std::size_t>(core.len));
}

int countOccurrences(const std::string& text, const std::string& needle) {
    int n = 0;
    for (std::size_t at = text.find(needle); at != std::string::npos; at = text.find(needle, at + 1)) ++n;
    return n;
}

std::vector<gl::IntegrationRuleKind> kindsOf(const std::vector<gl::IntegrationImplication>& family) {
    std::vector<gl::IntegrationRuleKind> kinds;
    for (const gl::IntegrationImplication& record : family) kinds.push_back(record.kind);
    return kinds;
}

// A standalone entity table on its own pool block (the FlatVariantFixture
// shape of the run-time flat-variant tests).
struct TableFixture {
    gl::GlobalMemoryManager g;
    gl::LbArena lb;
    gl::WorkInstruction wi;
    TableFixture()
        : g(), lb((g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 }), &g)), wi(&lb) {}
    void addEntity(const char* category, const char* signature,
                   const std::initializer_list<const char*>& elements) {
        const int32_t mark = wi.elemMark();
        for (const char* element : elements)
            wi.addElement(gl::StrSpan(element, static_cast<int32_t>(std::strlen(element))));
        wi.commitEntity(gl::StrSpan(category, static_cast<int32_t>(std::strlen(category))),
            gl::StrSpan(signature, static_cast<int32_t>(std::strlen(signature))),
            gl::StrSpan(), static_cast<int32_t>(elements.size()), mark);
    }
};

} // namespace

TEST(integration_implications, rule_kind_names_round_trip) {
    for (const gl::IntegrationRuleKind kind : { gl::IntegrationRuleKind::OneLevel, gl::IntegrationRuleKind::Occupied,
             gl::IntegrationRuleKind::Flat, gl::IntegrationRuleKind::OrIntroduction,
             gl::IntegrationRuleKind::DisjunctFlat }) {
        ASSERT_TRUE(gl::integrationRuleKindFromName(gl::integrationRuleKindName(kind)) == kind);
    }
    ASSERT_EQ(std::string(gl::integrationRuleKindName(gl::IntegrationRuleKind::OneLevel)), std::string("one_level"));
    ASSERT_EQ(std::string(gl::integrationRuleKindName(gl::IntegrationRuleKind::DisjunctFlat)), std::string("disjunct_flat"));
}

// The block-free core builds the same table as the block door and advances
// the counter it is given exactly as the door advances the block's.
TEST(integration_implications, block_free_core_matches_block_door) {
    gl::tests::ProverFixture analyzer("FTA");
    const std::string signature = "(preorder[u_1,u_2,u_3,u_4])";
    gl::Memory memory;
    memory.level = 3;
    memory.startIntPi = 5;
    gl::ScratchArena& arena = gl::genScratchArenas().forSlot(gl::genScratchArenas().slotCount() - 1);
    gl::WorkInstruction viaBlock(&arena);
    gl::WorkInstruction viaCore(&arena);
    analyzer.integrationBuilder.prepareIntegrationCore(gl::StrSpan(signature), viaBlock, memory,
        gl::StrSpan("goal", 4));
    int counter = 5;
    analyzer.integrationBuilder.prepareIntegrationCore(gl::StrSpan(signature), viaCore, 3, counter,
        gl::StrSpan("goal", 4));
    ASSERT_EQ(memory.startIntPi, counter);
    ASSERT_TRUE(counter > 5);
    ASSERT_EQ(viaBlock.entityCount(), viaCore.entityCount());
    for (int32_t e = 0; e < viaBlock.entityCount(); ++e) {
        ASSERT_TRUE(gl::equalSpans(viaBlock.signature(e), viaCore.signature(e)));
        ASSERT_TRUE(gl::equalSpans(viaBlock.category(e), viaCore.category(e)));
        ASSERT_EQ(viaBlock.elemCount(e), viaCore.elemCount(e));
        for (int32_t j = 0; j < viaBlock.elemCount(e); ++j)
            ASSERT_TRUE(gl::equalSpans(viaBlock.elemAt(e, j), viaCore.elemAt(e, j)));
    }
    ASSERT_EQ(viaCore.elemAt(0, 0).toStdString(), std::string("(in[pi_lev_3_5,u_1])"));
}

// The shared walk visits the three deeper premise lists of the D1 goal once
// each (strictOrder intact, preorder intact, fully flat), never the one-level
// list.
TEST(integration_implications, flat_walk_visits_each_deeper_list_once) {
    gl::tests::ProverFixture analyzer("FTA");
    TableFixture f;
    f.addEntity("existence", "(existence4[u_1,u_4,u_5,u_9,u_10])",
        { "(in[pi_lev_0_1,u_1])", "(remainder[u_1,u_4,u_5,u_9,u_10,pi_lev_0_1])" });
    f.addEntity("atomic", "(in[pi_lev_0_1,u_1])", {});
    f.addEntity("existence", "(remainder[u_1,u_4,u_5,u_9,u_10,pi_lev_0_1])",
        { "(in[pi_lev_0_2,u_1])", "(existence3[u_1,pi_lev_0_2,u_10,u_5,pi_lev_0_1,u_9,u_4])" });
    f.addEntity("existence", "(existence3[u_1,pi_lev_0_2,u_10,u_5,pi_lev_0_1,u_9,u_4])",
        { "(in[pi_lev_0_3,u_1])", "(and1[pi_lev_0_2,u_10,pi_lev_0_3,u_5,pi_lev_0_1,u_9,u_4,u_1])" });
    f.addEntity("and", "(and1[pi_lev_0_2,u_10,pi_lev_0_3,u_5,pi_lev_0_1,u_9,u_4,u_1])",
        { "(in3[pi_lev_0_2,u_10,pi_lev_0_3,u_5])", "(in3[pi_lev_0_3,pi_lev_0_1,u_9,u_4])",
          "(strictOrder[u_1,u_4,pi_lev_0_1,u_10])" });
    f.addEntity("and", "(strictOrder[u_1,u_4,pi_lev_0_1,u_10])",
        { "(preorder[u_1,u_4,pi_lev_0_1,u_10])", "!(=[pi_lev_0_1,u_10])" });
    f.addEntity("existence", "(preorder[u_1,u_4,pi_lev_0_1,u_10])",
        { "(in[pi_lev_0_4,u_1])", "(in3[pi_lev_0_1,pi_lev_0_4,u_10,u_4])" });

    std::vector<std::vector<std::string>> visited;
    gl::forEachFlatIntegrationVariant(f.wi, 0, analyzer.definitions.configuredNames(),
        [&](const gl::StrSpan* premises, int32_t n) {
            std::vector<std::string> list;
            for (int32_t i = 0; i < n; ++i) list.push_back(premises[i].toStdString());
            visited.push_back(list);
        });
    const std::string inR = "(in[pi_lev_0_1,u_1])", inQ = "(in[pi_lev_0_2,u_1])", inT = "(in[pi_lev_0_3,u_1])";
    const std::string inP = "(in[pi_lev_0_4,u_1])";
    const std::string prod = "(in3[pi_lev_0_2,u_10,pi_lev_0_3,u_5])", sum = "(in3[pi_lev_0_3,pi_lev_0_1,u_9,u_4])";
    const std::string lt = "(strictOrder[u_1,u_4,pi_lev_0_1,u_10])", le = "(preorder[u_1,u_4,pi_lev_0_1,u_10])";
    const std::string neq = "!(=[pi_lev_0_1,u_10])", sumP = "(in3[pi_lev_0_1,pi_lev_0_4,u_10,u_4])";
    const std::set<std::vector<std::string>> expected = {
        { inR, inQ, inT, prod, sum, lt },
        { inR, inQ, inT, prod, sum, le, neq },
        { inR, inQ, inT, prod, sum, inP, sumP, neq } };
    ASSERT_EQ(visited.size(), static_cast<std::size_t>(3));
    ASSERT_TRUE(std::set<std::vector<std::string>>(visited.begin(), visited.end()) == expected);
}

// An expression reached through two branches of the definition tree is a
// premise once: the inner existence's binder premise is also the first
// conjunct of the configured bundle it binds (the hasPrimeFactorization /
// primeFactorSeq shape, here with strictOrder as the bundle). Extending the
// bundle keeps one copy; the intact list is unchanged.
TEST(integration_implications, flat_walk_names_a_shared_premise_once) {
    gl::tests::ProverFixture analyzer("FTA");
    TableFixture f;
    f.addEntity("existence", "(existence900[u_1,u_4,u_10])",
        { "(in[pi_lev_0_1,u_1])", "(existence901[u_1,u_4,pi_lev_0_1,u_10])" });
    f.addEntity("existence", "(existence901[u_1,u_4,pi_lev_0_1,u_10])",
        { "(preorder[u_1,u_4,pi_lev_0_1,u_10])", "(strictOrder[u_1,u_4,pi_lev_0_1,u_10])" });
    f.addEntity("and", "(strictOrder[u_1,u_4,pi_lev_0_1,u_10])",
        { "(preorder[u_1,u_4,pi_lev_0_1,u_10])", "!(=[pi_lev_0_1,u_10])" });

    std::vector<std::vector<std::string>> visited;
    gl::forEachFlatIntegrationVariant(f.wi, 0, analyzer.definitions.configuredNames(),
        [&](const gl::StrSpan* premises, int32_t n) {
            std::vector<std::string> list;
            for (int32_t i = 0; i < n; ++i) list.push_back(premises[i].toStdString());
            visited.push_back(list);
        });
    const std::string in = "(in[pi_lev_0_1,u_1])", le = "(preorder[u_1,u_4,pi_lev_0_1,u_10])";
    const std::string lt = "(strictOrder[u_1,u_4,pi_lev_0_1,u_10])", neq = "!(=[pi_lev_0_1,u_10])";
    const std::vector<std::vector<std::string>> expected = { { in, le, lt }, { in, le, neq } };
    ASSERT_TRUE(visited == expected);
    for (const std::vector<std::string>& list : visited)
        ASSERT_EQ(std::set<std::string>(list.begin(), list.end()).size(), list.size());
}

TEST(integration_implications, same_up_to_bound_names) {
    using C = gl::IntegrationRuleCompiler;
    ASSERT_TRUE(C::sameUpToBoundNames("(>[x](in[x,u_1])(>[y](in3[x,y,u_2,u_3])(p[u_1,u_2])))",
                                      "(>[1](in[1,u_1])(>[2](in3[1,2,u_2,u_3])(p[u_1,u_2])))"));
    ASSERT_TRUE(C::sameUpToBoundNames("(>[](a[u_1])(b[u_1]))", "(>[](a[u_1])(b[u_1]))"));
    // Premise order, binder placement and free arguments must agree.
    ASSERT_FALSE(C::sameUpToBoundNames("(>[x](in[x,u_1])(>[](q[x])(p[u_1])))",
                                       "(>[x](q[x])(>[](in[x,u_1])(p[u_1])))"));
    ASSERT_FALSE(C::sameUpToBoundNames("(>[x](in[x,u_1])(p[u_1]))", "(>[x](in[x,u_2])(p[u_1]))"));
    ASSERT_FALSE(C::sameUpToBoundNames("(>[x,y](r[x,y])(p[u_1]))", "(>[x,y](r[y,x])(p[u_1]))"));
}

// An existence's set starts with its one-level rule (witness bound) and its
// occupied form (witness as the extra token); a reuse leaves it unchanged.
TEST(integration_implications, existence_family_one_level_and_occupied) {
    gl::tests::ProverFixture analyzer("Peano");
    std::string text = "!(>[x](in[x,a])!(in2[b,x,c]))";
    analyzer.structuralCompiler.precompileStructuralOperators(text);
    const std::string name = coreName(text);
    const gl::LogicalEntity* entity = analyzer.definitions.compiledEntity(gl::StrSpan(name));
    ASSERT_TRUE(entity != nullptr);
    const std::vector<gl::IntegrationImplication> family = entity->integrationImplications;
    ASSERT_TRUE(family.size() >= 2);
    ASSERT_TRUE(family[0].kind == gl::IntegrationRuleKind::OneLevel);
    ASSERT_TRUE(family[1].kind == gl::IntegrationRuleKind::Occupied);
    ASSERT_EQ(family[0].cites, -1);

    const std::string oneLevel = gl::ImplicationBuilder::reconstructImplicationFullBind(
        entity->elements, entity->signature);
    ASSERT_TRUE(gl::IntegrationRuleCompiler::sameUpToBoundNames(
        analyzer.integrationRuleCompiler.expandIntegrationCompact(family[0].compact), oneLevel));

    std::vector<std::string> occupiedElements;
    const std::string token = "u_" + std::to_string(entity->arity + 1);
    for (const std::string& element : entity->elements)
        occupiedElements.push_back(ce::replaceKeysInString(element, { { "1", token } }));
    const std::string occupied = gl::ImplicationBuilder::reconstructImplicationFullBind(
        occupiedElements, entity->signature);
    ASSERT_TRUE(analyzer.integrationRuleCompiler.expandIntegrationCompact(family[1].compact) == occupied);
    ASSERT_TRUE(family[1].compact.find(token) != std::string::npos);

    // Idempotent: a reuse or a second call leaves the set as it is.
    std::string again = "!(>[y](in[y,a])!(in2[b,y,c]))";
    analyzer.structuralCompiler.precompileStructuralOperators(again);
    ASSERT_EQ(coreName(again), name);
    analyzer.integrationRuleCompiler.compileIntegrationImplications(name);
    ASSERT_TRUE(analyzer.definitions.compiledEntity(gl::StrSpan(name))->integrationImplications == family);
}

// bez1 (FTA row 121's goal): the one-level rule and its occupied form, the
// one flat variant in[x], in[y], or(a,x,b,y), the OR premise's two
// introductions citing that variant, and each bezEq disjunct's flat rule
// in[u], in[v], a*x = u, b*y = v, v + 1 = u.
TEST(integration_implications, bez1_family_holds_or_introductions_and_disjunct_rules) {
    gl::tests::ProverFixture analyzer("FTA");
    const gl::LogicalEntity* bez1 = analyzer.definitions.compiledEntity(gl::StrSpan("bez1", 4));
    ASSERT_TRUE(bez1 != nullptr);
    const std::vector<gl::IntegrationImplication>& family = bez1->integrationImplications;
    const std::vector<gl::IntegrationRuleKind> expectedKinds = {
        gl::IntegrationRuleKind::OneLevel, gl::IntegrationRuleKind::Occupied, gl::IntegrationRuleKind::Flat,
        gl::IntegrationRuleKind::OrIntroduction, gl::IntegrationRuleKind::OrIntroduction,
        gl::IntegrationRuleKind::DisjunctFlat, gl::IntegrationRuleKind::DisjunctFlat };
    ASSERT_TRUE(kindsOf(family) == expectedKinds);
    ASSERT_EQ(family[3].cites, 2);
    ASSERT_EQ(family[4].cites, 2);

    const std::string flat = analyzer.integrationRuleCompiler.expandIntegrationCompact(family[2].compact);
    ASSERT_EQ(countOccurrences(flat, "(in["), 2);
    ASSERT_EQ(countOccurrences(flat, "(bez1["), 1);
    for (int i = 5; i <= 6; ++i) {
        const std::string rule = analyzer.integrationRuleCompiler.expandIntegrationCompact(family[i].compact);
        ASSERT_EQ(countOccurrences(rule, "(in["), 2);
        ASSERT_EQ(countOccurrences(rule, "(in3["), 3);
        ASSERT_EQ(countOccurrences(rule, "(bezEq["), 1);
    }
    // Every compact names only bez1's own tokens (the witnesses are bound).
    for (const gl::IntegrationImplication& record : family) {
        for (const std::string& argument : ce::getArgs(record.compact))
            ASSERT_TRUE(argument.rfind("u_", 0) == 0);
    }
}

// The integration set is written for every and and existence and read back
// verbatim.
TEST(integration_implications, binary_round_trip_keeps_the_integration_set) {
    namespace fs = std::filesystem;
    gl::tests::ProverFixture analyzer("FTA");
    const std::vector<gl::IntegrationImplication> expected =
        analyzer.definitions.compiledEntity(gl::StrSpan("bez1", 4))->integrationImplications;
    ASSERT_TRUE(!expected.empty());
    const fs::path dir = fs::temp_directory_path() / "gl_integration_implications_test";
    fs::create_directories(dir);
    analyzer.compilationPersistence.exportCompiledExpressionsJSON(dir);
    const fs::path binPath = dir / "GL_binary_FTA.json";
    ASSERT_TRUE(fs::exists(binPath));

    gl::tests::ProverFixture restored("FTA");
    restored.compilationPersistence.loadGlBinary(binPath);
    const gl::LogicalEntity* bez1 = restored.definitions.compiledEntity(gl::StrSpan("bez1", 4));
    ASSERT_TRUE(bez1 != nullptr);
    ASSERT_TRUE(bez1->integrationImplications == expected);
    fs::remove(binPath);
    fs::remove(dir / "GL_compilation_state_FTA.json");
}

// A batch seeded from the shared binary, which holds the compound definitions
// but no atomic one, mints a compound whose definition tree reaches an atomic
// definition through a loaded compound: the atomic definitions register before
// any compound, so its integration set compiles.
TEST(integration_implications, seeded_registry_registers_atomic_definitions_first) {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "gl_integration_seeded_registry_test";
    fs::create_directories(dir);
    {
        gl::tests::ProverFixture first("IncubatorPeano1");
        first.compilationPersistence.exportCompiledExpressionsJSON(dir);
    }
    nlohmann::json shared;
    {
        std::ifstream input(dir / "GL_binary_IncubatorPeano1.json");
        ASSERT_TRUE(input.is_open());
        input >> shared;
    }
    ASSERT_TRUE(shared.contains("in") && shared.contains("NaturalNumbers"));
    for (auto it = shared.begin(); it != shared.end();) {
        if (it.value().value("category", std::string()) == "atomic") it = shared.erase(it);
        else ++it;
    }
    ASSERT_FALSE(shared.contains("in"));
    const fs::path seeded = dir / "GL_binary_IncubatorPeano2.json";
    {
        std::ofstream output(seeded);
        output << shared.dump();
    }
    fs::copy_file(dir / "GL_compilation_state_IncubatorPeano1.json",
        dir / "GL_compilation_state_IncubatorPeano2.json", fs::copy_options::overwrite_existing);

    gl::tests::ProverFixture second("IncubatorPeano2", std::nullopt, seeded);
    const gl::LogicalEntity* in = second.definitions.compiledEntity(gl::StrSpan("in", 2));
    ASSERT_TRUE(in != nullptr && in->category == "atomic");
    const gl::LogicalEntity* anchor = second.definitions.compiledEntity(
        gl::StrSpan("AnchorIncubator3", 16));
    ASSERT_TRUE(anchor != nullptr && !anchor->integrationImplications.empty());
    fs::remove_all(dir);
}

// Compiling a rule with more premises than the permutation table holds leaves
// the table as it is: an install looks its premise count up in the same table,
// and an empty entry there would leave every premise of the copy unencoded.
TEST(integration_implications, compiling_a_long_rule_leaves_the_permutation_table) {
    gl::tests::ProverFixture analyzer("Peano");
    const std::size_t sizes = analyzer.allPermutationsAna.size();
    ASSERT_TRUE(analyzer.allPermutationsAna.find(9) == analyzer.allPermutationsAna.end());
    analyzer.structuralCompiler.compileImplicationToCompact(
        "(>[](in[a,b])(>[](in[b,c])(>[](in[c,d])(>[](in[d,e])(>[](in[e,f])(>[](in[f,g])"
        "(>[](in[g,h])(>[](in[h,k])(>[](in[k,m])(in[a,m]))))))))))");
    ASSERT_EQ(analyzer.allPermutationsAna.size(), sizes);
    ASSERT_TRUE(analyzer.allPermutationsAna.find(9) == analyzer.allPermutationsAna.end());
}

// The extensionality entity of a set type carries its one-level rule:
// both subset implications imply the equality.
TEST(integration_implications, typed_equality_entity_carries_its_extensionality_rule) {
    gl::tests::ProverFixture analyzer("FTA");
    analyzer.structuralCompiler.ensureTypedEqualityEntity(gl::StrSpan(std::string("P(1)")));
    const gl::LogicalEntity* typed = analyzer.definitions.typedEqualityEntity(gl::StrSpan(std::string("P(1)")));
    ASSERT_TRUE(typed != nullptr);
    ASSERT_EQ(typed->integrationImplications.size(), static_cast<std::size_t>(1));
    ASSERT_TRUE(typed->integrationImplications[0].kind == gl::IntegrationRuleKind::OneLevel);
    const std::string rule = gl::ImplicationBuilder::reconstructImplicationFullBind(typed->elements, typed->signature);
    ASSERT_TRUE(analyzer.integrationRuleCompiler.expandIntegrationCompact(typed->integrationImplications[0].compact)
        == rule);
}

// ---------------------------------------------------------------------------
// Run time: the compiled compacts are filled, expanded and installed with
// their carriers (D-503).
// ---------------------------------------------------------------------------
namespace {

// The owner tokens of an entity's signature as spans (the identity filling:
// run-time names `u_1..u_k`).
std::vector<gl::StrSpan> signatureOwners(const gl::LogicalEntity& entity, std::vector<std::string>& storage) {
    storage = ce::getArgs(entity.signature);
    std::vector<gl::StrSpan> owners;
    for (const std::string& token : storage) owners.emplace_back(token);
    return owners;
}

// Every rule chain (premises then head) of the overall `originals` registry.
std::vector<std::vector<std::string>> originalChains(gl::Memory& m) {
    std::vector<std::vector<std::string>> out;
    auto& originals = m.overallHashMemory.originals;
    for (int32_t id = 1; id <= originals.count(); ++id) {
        const gl::IdVecKeyView kv = gl::viewIdVecKey(originals.inner().keyAt(id));
        std::vector<std::string> chain;
        for (int32_t k = 0; k < kv.count(); ++k) chain.push_back(m.ruleInterner.decode(kv.idAt(k)));
        out.push_back(std::move(chain));
    }
    return out;
}

// Whether some history row of @p m carries @p tag citing @p cited at main.
bool rowCites(gl::Memory& m, const std::string& tag, const std::string& cited) {
    for (int32_t id = 1; id <= m.exprOriginMap.count(); ++id)
        for (const gl::IdOrigin& o : m.exprOriginMap.recordsAt(id)) {
            const auto line = gl::decodeOrigin(o, m.originInterner);
            if (line.first != tag) continue;
            for (const auto& dep : line.second)
                if (dep.original == cited && dep.validityName == "main") return true;
        }
    return false;
}

// The carrier index as `text@scope -> install kinds`.
std::vector<std::pair<std::string, std::vector<gl::RuleInstallKind>>> carriers(gl::Memory& m) {
    std::vector<std::pair<std::string, std::vector<gl::RuleInstallKind>>> out;
    auto& index = m.compactExpansions;
    for (int32_t id = 1; id <= index.count(); ++id) {
        const int64_t key = index.keyAt(id);
        const int32_t live = index.lookup(key);
        if (live == 0) continue;
        const gl::StatementKey sk = gl::Codec<gl::StatementKey>::decode(key);
        std::vector<gl::RuleInstallKind> kinds;
        for (int32_t j = 0; j < index.runLen(live); ++j) kinds.push_back(index.recordAt(live, j).kind);
        out.emplace_back(m.nameMap.decode(sk.orig) + "@" + m.nameMap.decode(sk.validity), kinds);
    }
    return out;
}

} // namespace

// One token-boundary pass: an owner value that itself reads `u_<j>` is never
// replaced again.
TEST(integration_implications, fill_replaces_each_owner_token_once) {
    gl::tests::ProverFixture analyzer("FTA");
    gl::ScratchArena& arena = gl::scratchArenas().forSlot(gl::scratchArenas().slotCount() - 1);
    gl::ScratchScope frame(arena);
    const gl::StrSpan owners[3] = { gl::StrSpan("u_12", 4), gl::StrSpan("u_1", 3), gl::StrSpan("u_marker", 8) };
    const gl::ScratchString filled = analyzer.integrationBuilder.fillIntegrationCompactScratch(arena,
        gl::StrSpan("(implication5[u_2,u_1,u_3,u_2])", 31), owners, 3);
    ASSERT_EQ(gl::StrSpan(filled).toStdString(), std::string("(implication5[u_1,u_12,u_marker,u_1])"));
}

// Expansion fills the formals positionally and names every bound placeholder
// with a fresh `pi_lev_<level>_<n>` from the block counter.
TEST(integration_implications, expand_names_bound_placeholders_fresh) {
    gl::tests::ProverFixture analyzer("FTA");
    const gl::LogicalEntity* bez1 = analyzer.definitions.compiledEntity(gl::StrSpan("bez1", 4));
    ASSERT_TRUE(bez1 != nullptr && bez1->integrationImplications.size() == 7u);
    std::vector<std::string> storage;
    const std::vector<gl::StrSpan> owners = signatureOwners(*bez1, storage);
    gl::Memory memory;
    gl::ScratchArena& arena = gl::scratchArenas().forSlot(gl::scratchArenas().slotCount() - 1);
    gl::ScratchScope frame(arena);
    const gl::ScratchString compact = analyzer.integrationBuilder.fillIntegrationCompactScratch(arena,
        gl::StrSpan(bez1->integrationImplications[5].compact), owners.data(), static_cast<int32_t>(owners.size()));
    const int counterBefore = memory.startIntPi;
    gl::StrSpan premises[gl::ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
    gl::StrSpan head;
    const int32_t premiseN = analyzer.integrationBuilder.expandIntegrationCompactScratch(arena, gl::StrSpan(compact),
        memory, premises, gl::ExecutionParameters::MAX_INSTRUCTION_ELEMENTS, head);
    ASSERT_EQ(premiseN, 5);
    ASSERT_TRUE(head.toStdString().rfind("(bezEq[", 0) == 0);
    // The disjunct's open variables and the two products are bound: four
    // placeholders, four fresh names.
    ASSERT_EQ(memory.startIntPi - counterBefore, 4);
    std::set<std::string> fresh;
    for (int32_t p = 0; p < premiseN; ++p)
        for (const std::string& argument : ce::getArgs(premises[p].toStdString())) {
            ASSERT_TRUE(argument.rfind("u_", 0) == 0 || argument.rfind("pi_lev_", 0) == 0);
            if (argument.rfind("pi_lev_", 0) == 0) fresh.insert(argument);
        }
    ASSERT_EQ(fresh.size(), 4u);
}

// The unchangeables of a rule: the distinct u_-stripped arguments of head and
// premises, byte-lex sorted.
TEST(integration_implications, collect_unchangeables_sorted_unique) {
    gl::tests::ProverFixture analyzer("FTA");
    const gl::StrSpan head("(bez1[u_1,u_4,u_5,u_6,u_12,u_9])", 32);
    const gl::StrSpan premises[2] = {
        gl::StrSpan("(in[pi_lev_3_12,u_1])", 21),
        gl::StrSpan("(in3[u_9,pi_lev_3_12,u_12,u_4])", 31) };
    gl::StrSpan out[gl::ExecutionParameters::MAX_ADMISSION_REM_ARGS];
    const int32_t n = analyzer.integrationBuilder.collectUnchangeables(head, premises, 2, out,
        gl::ExecutionParameters::MAX_ADMISSION_REM_ARGS);
    std::vector<std::string> got;
    for (int32_t i = 0; i < n; ++i) got.emplace_back(out[i].ptr, static_cast<std::size_t>(out[i].len));
    ASSERT_TRUE(got == std::vector<std::string>({ "1", "12", "4", "5", "6", "9" }));
    const gl::StrSpan bare("(in[pi_lev_3_12,pi_lev_3_13])", 29);
    ASSERT_EQ(analyzer.integrationBuilder.collectUnchangeables(bare, nullptr, 0, out,
        gl::ExecutionParameters::MAX_ADMISSION_REM_ARGS), 0);
}

// One install: the compact cites its antecedent, the rule cites the compact,
// the carrier records an Integration rule; a repeat at the scope installs
// nothing; the removal takes the integration route and drops the rule and the
// carrier.
TEST(integration_implications, install_once_per_scope_and_remove_through_the_integration_route) {
    gl::tests::ProverFixture analyzer("FTA");
    analyzer.parameters.trackHistory = true;
    const gl::LogicalEntity* bez1 = analyzer.definitions.compiledEntity(gl::StrSpan("bez1", 4));
    ASSERT_TRUE(bez1 != nullptr);
    std::vector<std::string> storage;
    const std::vector<gl::StrSpan> owners = signatureOwners(*bez1, storage);
    gl::Memory memory;
    const gl::StrSpan mainScope("main", 4);
    gl::ScratchArena& arena = gl::scratchArenas().forSlot(gl::scratchArenas().slotCount() - 1);
    gl::ScratchScope frame(arena);
    const gl::ScratchString compact = analyzer.integrationBuilder.fillIntegrationCompactScratch(arena,
        gl::StrSpan(bez1->integrationImplications[0].compact), owners.data(), static_cast<int32_t>(owners.size()));
    const std::string clean = gl::DefinitionExpansion::removeUPrefixFromArguments(gl::StrSpan(compact).toStdString());
    const std::string cited = "(cited)_integration_goal";
    const gl::OriginDep deps[1] = { { gl::StrSpan(cited), mainScope } };
    gl::StrSpan premises[gl::ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
    int32_t premiseN = 0;
    gl::StrSpan head;
    ASSERT_TRUE(analyzer.compactRuleProcessor.installIntegrationCompact(memory, gl::StrSpan(compact), mainScope,
        gl::OriginTag::reformulationForIntegrationBound, deps, 1, arena, premises,
        gl::ExecutionParameters::MAX_INSTRUCTION_ELEMENTS, premiseN, head));
    ASSERT_EQ(premiseN, 2);
    ASSERT_EQ(head.toStdString(), bez1->signature);
    const gl::CompactRuleProcessor::CarrierKinds kinds =
        gl::CompactRuleProcessor::carrierKinds(memory, gl::StrSpan(clean), mainScope);
    ASSERT_TRUE(kinds.integration && !kinds.statement);
    ASSERT_TRUE(rowCites(memory, "reformulation for integration >[bound]", cited));
    ASSERT_TRUE(rowCites(memory, "expansion", clean));

    const int32_t originals = memory.overallHashMemory.originals.count();
    ASSERT_TRUE(originals > 0);
    ASSERT_FALSE(analyzer.compactRuleProcessor.installIntegrationCompact(memory, gl::StrSpan(compact), mainScope,
        gl::OriginTag::reformulationForIntegrationBound, deps, 1, arena, premises,
        gl::ExecutionParameters::MAX_INSTRUCTION_ELEMENTS, premiseN, head));
    ASSERT_EQ(memory.overallHashMemory.originals.count(), originals);

    const int64_t carrierKey = gl::packStatementKey(memory.nameMap.lookup(gl::StrSpan(clean)),
        memory.nameMap.lookup(mainScope));
    const int32_t carrierId = memory.compactExpansions.lookup(carrierKey);
    ASSERT_TRUE(carrierId != 0);
    const auto record = memory.compactExpansions.recordAt(carrierId, 0);
    ASSERT_TRUE(record.kind == gl::RuleInstallKind::Integration);
    const int32_t ruleId = memory.ruleInterner.lookup(memory.lbStateInterner.decodeView(record.implTextId));
    ASSERT_TRUE(ruleId != 0);
    {
        gl::ScratchArena& sets = gl::genScratchArenas().forSlot(0);
        gl::DirtyState dirty = gl::DirtyState::Clean;
        gl::ColdHashSet<gl::BytesKeyStore> visited(&sets, &dirty);
        gl::ColdHashSet<gl::PodKeyStore<int64_t>> drops(&sets, &dirty), removed(&sets, &dirty);
        gl::tests::CompactRuleProcessorFixture::removeCompactExpansion(analyzer.compactRuleProcessor, memory,
            carrierKey, visited, drops, removed);
        gl::CompactRuleProcessor::finishCompactRemovals(memory, drops, removed, sets);
    }
    ASSERT_EQ(memory.compactExpansions.lookup(carrierKey), 0);
    for (const gl::HashMemory* hm : { &memory.overallHashMemory, &memory.localHashMemory, &memory.localHashMemoryDelta })
        for (int32_t i = 1; i <= hm->encodedMap.count(); ++i)
            for (int32_t j = 0; j < hm->encodedMap.runLen(i); ++j)
                ASSERT_TRUE(hm->encodedMap.recordAt(i, j).originalImplicationId != ruleId);
}

// Case B over the bez1 table: the one-level compact and the root family —
// the flat variant, the OR premise's two introductions with their double
// keys, each disjunct's flat rule citing the disjunct's integration goal —
// each installed with its carrier; the root's flat variant is the one carrier
// of the flat kind (D-574). A marked goal
// that is not the root takes the one-level compact only.
TEST(integration_implications, case_b_installs_the_compiled_family_with_carriers) {
    for (const bool root : { true, false }) {
        gl::tests::ProverFixture analyzer("FTA");
        analyzer.parameters.trackHistory = true;
        const gl::LogicalEntity* bez1 = analyzer.definitions.compiledEntity(gl::StrSpan("bez1", 4));
        ASSERT_TRUE(bez1 != nullptr);
        const std::string goal = gl::DefinitionExpansion::removeUPrefixFromArguments(bez1->signature);
        gl::Memory memory;
        TableFixture f;
        analyzer.integrationBuilder.prepareIntegrationCore(gl::StrSpan(bez1->signature), f.wi, memory, gl::StrSpan(goal));
        if (!root) {
            std::string other = goal;
            other.replace(other.rfind(','), std::string::npos, ",marker])");
            f.wi.setMarkedGoal(gl::StrSpan(other));
        }
        analyzer.integrationProcessor.prepareIntegrationCore2(f.wi, nullptr, 0, memory,
            gl::StrSpan("main", 4), gl::StrSpan());

        const auto index = carriers(memory);
        ASSERT_EQ(index.size(), root ? 6u : 1u);
        int flatCarriers = 0;
        for (const auto& carrier : index) {
            ASSERT_TRUE(carrier.first.rfind("(implication", 0) == 0);
            ASSERT_EQ(carrier.second.size(), static_cast<std::size_t>(1));
            ASSERT_TRUE(gl::isIntegrationInstallKind(carrier.second[0]));
            if (carrier.second[0] == gl::RuleInstallKind::IntegrationFlat) ++flatCarriers;
        }
        ASSERT_EQ(flatCarriers, root ? 1 : 0);
        const auto& keys = memory.overallHashMemory.admissionMapIntegrationDouble;
        ASSERT_EQ(keys.count(), root ? 2 : 0);
        if (!root) continue;
        ASSERT_TRUE(rowCites(memory, "reformulation for integration flattened", goal + "_integration_goal"));
        // Each disjunct's flat rule: in[u], in[v], three sums and products, the disjunct as head.
        int disjunctRules = 0;
        for (const auto& chain : originalChains(memory)) {
            if (chain.back().rfind("(bezEq[", 0) != 0 || chain.size() != 6) continue;
            int ins = 0, in3s = 0;
            for (std::size_t p = 0; p + 1 < chain.size(); ++p) {
                if (chain[p].rfind("(in[", 0) == 0) ++ins;
                if (chain[p].rfind("(in3[", 0) == 0) ++in3s;
            }
            if (ins == 2 && in3s == 3) ++disjunctRules;
        }
        ASSERT_TRUE(disjunctRules >= 2);
        int disjunctGoals = 0;
        for (int32_t id = 1; id <= memory.exprOriginMap.count(); ++id)
            for (const gl::IdOrigin& o : memory.exprOriginMap.recordsAt(id)) {
                const auto line = gl::decodeOrigin(o, memory.originInterner);
                if (line.first != "reformulation for integration flattened") continue;
                for (const auto& dep : line.second)
                    if (dep.original.rfind("(bezEq[", 0) == 0) ++disjunctGoals;
            }
        ASSERT_EQ(disjunctGoals, 2);
    }
}
