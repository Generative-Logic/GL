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
#include "case_ledger_fixture.hpp"

#include "compiled_definition_fixture.hpp"
#include "../prover.hpp"
#include "../compiler/supporting_rule_compiler.hpp"
#include <fstream>
#include <set>

namespace {
std::string symmetricTestOr(gl::tests::ProverFixture& analyzer, bool equality = true) {
    std::vector<std::string> leaves{"(in2[u_1,u_2,u_3])", "(in2[u_2,u_1,u_3])"};
    if (equality) leaves.insert(leaves.begin() + 1, "(=[u_1,u_2])");
    return analyzer.orRuleCompiler.findOrMintOrOperator(leaves, 3);
}

void clearCompiler(gl::tests::ProverFixture& analyzer) {
    gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions).clear();
    analyzer.compilationReuse.clear();
    gl::CompiledDefinitionFixture::implCounter(analyzer.definitions) = gl::CompiledDefinitionFixture::existenceCounter(analyzer.definitions) = gl::CompiledDefinitionFixture::andCounter(analyzer.definitions) = gl::CompiledDefinitionFixture::orCounter(analyzer.definitions) = 0;
    gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions)["in"] = gl::LogicalEntity("atomic", {""}, "(in[u_1,u_2])", 2);
    gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions)["in3"] = gl::LogicalEntity("atomic", {""}, "(in3[u_1,u_2,u_3,u_4])", 4);
    gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions)["="] = gl::LogicalEntity("atomic", {""}, "(=[u_1,u_2])", 2);
}

gl::LogicalEntity preorder(const std::string& name, bool numbered) {
    return gl::LogicalEntity("existence", numbered
        ? std::vector<std::string>{"(in[1,u_1])", "(in3[u_2,1,u_3,u_4])"}
        : std::vector<std::string>{"(in[1,u_1])", "(in3[u_3,1,u_4,u_2])"},
        "(" + name + "[u_1,u_2,u_3,u_4])", 4);
}

std::string compile(gl::tests::ProverFixture& analyzer, const std::string& expression) {
    return analyzer.structuralCompiler.compileExpression(expression);
}
} // namespace

TEST(compilation_state, symmetric_or_marks_typed_totality_and_trichotomy) {
    gl::tests::ProverFixture analyzer("Peano");
    for (bool equality : {false, true}) {
        const auto name = symmetricTestOr(analyzer, equality);
        const auto& entity = gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions).at(name);
        ASSERT_EQ(entity.symmetricDigitSlots[0], 1);
        ASSERT_EQ(entity.symmetricDigitSlots[1], 2);
        auto withoutMetadata = entity;
        withoutMetadata.symmetricDigitSlots = {0, 0};
        ASSERT_TRUE(entity == withoutMetadata);
    }
    gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions)["or999"] = gl::LogicalEntity("or",
        {"(in2[u_1,u_2,u_3])", "(in2[u_2,u_1,u_3])"}, "(or999[u_1,u_2,u_3])", 3);
    gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions)["or998"] = gl::LogicalEntity("or",
        {"(or999[u_1,u_2,u_3])", "!(=[u_1,u_2])"}, "(or998[u_1,u_2,u_3])", 3);
    analyzer.orRuleCompiler.compileOrSymmetry("or998");
    ASSERT_EQ(gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions).at("or998").symmetricDigitSlots[1], 2);
}

TEST(compilation_state, symmetric_or_rejects_order_polarity_multiplicity_and_three_digits) {
    gl::tests::ProverFixture analyzer("Peano");
    const std::vector<std::vector<std::string>> examples{
        {"(in2[u_1,u_2,u_3])", "(=[u_1,u_2])"},
        {"(in2[u_1,u_2,u_3])", "!(in2[u_2,u_1,u_3])"},
        {"(in2[u_1,u_2,u_3])", "(in2[u_1,u_2,u_3])", "(in2[u_2,u_1,u_3])"},
        {"(in3[u_1,u_2,u_3,u_4])", "(in3[u_2,u_1,u_3,u_4])"}};
    for (std::size_t i = 0; i < examples.size(); ++i) {
        const int arity = i == 3 ? 4 : 3;
        const std::string signature = arity == 4 ? "(or999[u_1,u_2,u_3,u_4])" : "(or999[u_1,u_2,u_3])";
        gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions)["or999"] = gl::LogicalEntity("or", examples[i], signature, arity);
        analyzer.orRuleCompiler.compileOrSymmetry("or999");
        ASSERT_EQ(gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions).at("or999").symmetricDigitSlots[0], 0);
    }
}

TEST(compilation_state, symmetric_or_metadata_rebuilt_from_unchanged_binary_shape) {
    gl::tests::ProverFixture analyzer("Peano");
    const auto name = symmetricTestOr(analyzer);
    analyzer.orRuleCompiler.preMintReducedOrs();
    const auto original = gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions).at(name);
    const auto path = std::filesystem::temp_directory_path() / "GL_binary_symmetric_or_fixture.json";
    nlohmann::json entry{{"category", original.category}, {"arity", original.arity},
        {"signature", original.signature}, {"elements", original.elements}, {"implications", original.implications}};
    { std::ofstream output(path); output << nlohmann::json({{name, entry}}).dump(); }
    analyzer.compilationPersistence.loadGlBinary(path);
    ASSERT_EQ(gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions).at(name).symmetricDigitSlots[0], 0);
    analyzer.orRuleCompiler.preMintReducedOrs();
    ASSERT_TRUE(gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions).at(name).symmetricDigitSlots == original.symmetricDigitSlots);
    ASSERT_TRUE(gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions).at(name) == original);
    std::filesystem::remove(path);
}

TEST(compilation_state, symmetric_or_processing_key_swaps_positions_not_values) {
    gl::tests::ProverFixture analyzer("Peano");
    const auto name = symmetricTestOr(analyzer);
    auto& strings = gl::scratchArenas().forSlot(gl::scratchArenas().slotCount() - 1);
    gl::ScratchScope scope(strings);
    const std::string text = "(" + name + "[z,a,z])";
    ASSERT_EQ(gl::tests::CaseLedgerFixture::canonicalOrProcessingForm(analyzer.caseLedger, gl::StrSpan(text), strings).toStdString(),
        "(" + name + "[a,z,z])");
    const std::string equal = "(" + name + "[a,a,f])";
    ASSERT_EQ(gl::tests::CaseLedgerFixture::canonicalOrProcessingForm(analyzer.caseLedger, gl::StrSpan(equal), strings).toStdString(), equal);
}

TEST(compilation_state, generated_name_classification_and_expression_rewrite) {
    ASSERT_TRUE(gl::isNumberedCompilationName("existence16", "existence"));
    ASSERT_FALSE(gl::isNumberedCompilationName("preorder", "existence"));
    ASSERT_FALSE(gl::isNumberedCompilationName("existence16extra", "existence"));
    ASSERT_FALSE(gl::isNumberedCompilationName("existence16", "implication"));
    gl::CompilationReplacements replacements{{"existence16", {"preorder", {1, 4, 2, 3}}}};
    ASSERT_EQ(gl::rewriteCompilationReferences("!(existence16[1,6,2,4])", replacements),
              std::string("!(preorder[1,4,6,2])"));
    ASSERT_EQ(gl::rewriteCompilationReferences("main_boundary_(existence16[a,a,b,c])\t(>[existence16](p[existence16])(q[]))", replacements),
              std::string("main_boundary_(preorder[a,c,a,b])\t(>[existence16](p[existence16])(q[]))"));
    replacements["implication3"] = {"implication4", {2, 1}};
    replacements["implication4"] = {"namedRule", {2, 1}};
    ASSERT_EQ(gl::rewriteCompilationReferences("(implication3[a,b])", replacements), std::string("(namedRule[a,b])"));
    replacements["or9"] = {"truth", {}};
    ASSERT_EQ(gl::rewriteCompilationReferences("(or9[])", replacements), std::string("(truth[])"));
    ASSERT_EQ(gl::rewriteCompilationReferences("(unrelated[a,b])", replacements), std::string("(unrelated[a,b])"));
}

TEST(compilation_state, generated_name_span_classification_matches_string) {
    const std::string categories[]{"implication", "existence", "and", "or", "atomic"};
    const std::string names[]{"implication3", "existence16", "and0", "and987",
        "or12", "strictOrder", "and", "and987extra", "and-1", "", "atomic1"};
    for (const auto& category : categories) {
        for (const auto& name : names) {
            ASSERT_EQ(gl::isNumberedCompilationName(gl::StrSpan(name), gl::StrSpan(category)),
                gl::isNumberedCompilationName(name, category));
        }
    }
    ASSERT_TRUE(gl::isNumberedCompilationName(gl::StrSpan("and987suffix", 6), gl::StrSpan("and", 3)));
    ASSERT_FALSE(gl::isNumberedCompilationName(gl::StrSpan("and987suffix", 12), gl::StrSpan("and", 3)));
    ASSERT_FALSE(gl::isNumberedCompilationName(gl::StrSpan("strictOrder", 11), gl::StrSpan("and", 3)));
    ASSERT_FALSE(gl::isNumberedCompilationName(gl::StrSpan("atomic1", 7), gl::StrSpan("atomic", 6)));
}

TEST(compilation_state, numbered_first_is_removed_and_all_references_are_permuted) {
    gl::tests::ProverFixture analyzer("Peano");
    clearCompiler(analyzer);
    gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions)["existence16"] = preorder("existence16", true);
    analyzer.compilationReuse.registerStoredCompilation("existence16", false);
    gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions)["strictOrder"] = gl::LogicalEntity("and",
        {"(existence16[u_1,u_3,u_4,u_2])", "!(=[u_3,u_4])"},
        "(strictOrder[u_1,u_2,u_3,u_4])", 4);
    gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions)["preorder"] = preorder("preorder", false);
    analyzer.compilationReuse.registerLoadedNamedCompilation("preorder");
    analyzer.compilationReuse.finalizeCompilationReplacements();
    ASSERT_EQ(gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions).count("existence16"), std::size_t(0));
    ASSERT_EQ(gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions).at("strictOrder").elements[0],
              std::string("(preorder[u_1,u_2,u_3,u_4])"));
    ASSERT_TRUE(analyzer.compilationReuse.replacements().at("existence16").arguments == std::vector<int>({1, 4, 2, 3}));
    for (const auto& entry : analyzer.compilationReuse.entries())
        ASSERT_TRUE(std::get<1>(entry.second) != "existence16");
    const auto size = analyzer.compilationReuse.entries().size();
    analyzer.compilationReuse.finalizeCompilationReplacements();
    ASSERT_EQ(analyzer.compilationReuse.entries().size(), size);
}

TEST(compilation_state, first_public_name_wins_and_later_numbered_definition_is_removed) {
    gl::tests::ProverFixture analyzer("Peano");
    clearCompiler(analyzer);
    gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions)["zFirst"] = preorder("zFirst", false);
    analyzer.compilationReuse.registerLoadedNamedCompilation("zFirst");
    gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions)["aSecond"] = preorder("aSecond", false);
    analyzer.compilationReuse.registerLoadedNamedCompilation("aSecond");
    gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions)["existence16"] = preorder("existence16", true);
    analyzer.compilationReuse.finalizeCompilationReplacements();
    ASSERT_EQ(gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions).count("zFirst"), std::size_t(1));
    ASSERT_EQ(gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions).count("aSecond"), std::size_t(1));
    ASSERT_EQ(gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions).count("existence16"), std::size_t(0));
    ASSERT_EQ(analyzer.compilationReuse.replacements().at("existence16").target, std::string("zFirst"));
    for (const auto& entry : analyzer.compilationReuse.publicEntries())
        ASSERT_EQ(std::get<1>(entry.second), std::string("zFirst"));
}

TEST(compilation_state, categories_and_implication_heads_remain_distinct) {
    gl::tests::ProverFixture analyzer("Peano");
    clearCompiler(analyzer);
    gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions)["preorder"] = preorder("preorder", false);
    analyzer.compilationReuse.registerLoadedNamedCompilation("preorder");
    auto other = preorder("implication16", true);
    other.category = "implication";
    gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions)["implication16"] = other;
    analyzer.compilationReuse.finalizeCompilationReplacements();
    ASSERT_EQ(gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions).count("implication16"), std::size_t(1));
    analyzer.compilationReuse.registerStoredCompilation("implication16", false);
    bool foundImplication = false;
    for (const auto& entry : analyzer.compilationReuse.entries()) {
        if (std::get<1>(entry.second) != "implication16") continue;
        foundImplication = true;
        ASSERT_EQ(entry.first.second, std::string("implication"));
        ASSERT_EQ(entry.first.first.back(), std::string("(in3[u_2,1,u_3,u_4])"));
    }
    ASSERT_TRUE(foundImplication);
}

TEST(compilation_state, loaded_preorder_is_reused_by_fresh_strict_order_compilation) {
    gl::tests::ProverFixture analyzer("Peano");
    clearCompiler(analyzer);
    gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions)["preorder"] = preorder("preorder", false);
    analyzer.compilationReuse.registerLoadedNamedCompilation("preorder");
    const std::string body = "!(>[p](in[p,N])!(in3[n,p,m,plus]))";
    const std::string result = compile(analyzer, body);
    ASSERT_EQ(result, std::string("(preorder[N,plus,n,m])"));
    ASSERT_EQ(gl::CompiledDefinitionFixture::existenceCounter(analyzer.definitions), 0);
    ce::CoreExpressionConfig config;
    config.signature = "(strictOrder[N,plus,n,m])";
    config.arity = 4;
    config.definition = "(&" + body + "!(=[n,m]))";
    gl::CompiledDefinitionFixture::coreExpressionMap(analyzer.definitions)["strictOrder"] = config;
    ASSERT_EQ(compile(analyzer, config.signature), config.signature);
    ASSERT_EQ(gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions).at("strictOrder").elements[0],
              std::string("(preorder[u_1,u_2,u_3,u_4])"));
}

TEST(compilation_state, load_order_round_trip_and_counter_high_water_survive_retirement) {
    namespace fs = std::filesystem;
    gl::tests::ProverFixture analyzer("Peano");
    clearCompiler(analyzer);
    gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions)["existence16"] = preorder("existence16", true);
    gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions)["zLast"] = preorder("zLast", false);
    gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions)["aFirst"] = preorder("aFirst", false);
    gl::CompiledDefinitionFixture::existenceCounter(analyzer.definitions) = 30;
    const auto directory = fs::temp_directory_path() / "gl_compilation_state_test";
    fs::create_directories(directory);
    analyzer.compilationPersistence.exportCompiledExpressionsJSON(directory);
    clearCompiler(analyzer);
    analyzer.compilationPersistence.loadGlBinary(directory / "GL_binary_Peano.json");
    ASSERT_EQ(analyzer.compilationReuse.replacements().at("existence16").target, std::string("aFirst"));
    ASSERT_EQ(gl::CompiledDefinitionFixture::existenceCounter(analyzer.definitions), 30);
    ASSERT_EQ(gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions).count("existence16"), std::size_t(0));
    analyzer.compilationPersistence.exportCompiledExpressionsJSON(directory);
    analyzer.compilationPersistence.loadGlBinary(directory / "GL_binary_Peano.json");
    ASSERT_EQ(gl::CompiledDefinitionFixture::existenceCounter(analyzer.definitions), 30);
    ASSERT_EQ(gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions).count("existence16"), std::size_t(0));
    fs::remove(directory / "GL_binary_Peano.json");
    fs::remove(directory / "GL_compilation_state_Peano.json");
}

TEST(compilation_state, direct_or_reuse_returns_public_argument_order) {
    gl::tests::ProverFixture analyzer("Peano");
    clearCompiler(analyzer);
    gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions)["choice"] = gl::LogicalEntity("or",
        {"(in[u_2,u_1])", "!(=[u_1,u_2])"}, "(choice[u_1,u_2])", 2);
    analyzer.compilationReuse.registerLoadedNamedCompilation("choice");
    const std::vector<std::string> elements = {"(in[u_1,u_2])", "!(=[u_2,u_1])"};
    bool minted = true;
    std::vector<int> permutation;
    ASSERT_EQ(analyzer.orRuleCompiler.findOrMintOrOperator(elements, 2, &minted, &permutation), std::string("choice"));
    ASSERT_FALSE(minted);
    ASSERT_TRUE(permutation == std::vector<int>({2, 1}));
    gl::StrSpan spans[] = {gl::StrSpan(elements[0]), gl::StrSpan(elements[1])};
    int positions[2] = {};
    const auto* name = analyzer.compilationReuse.compiledOrByElements(spans, 2, positions, 2);
    ASSERT_TRUE(name != nullptr);
    ASSERT_EQ(*name, std::string("choice"));
    ASSERT_EQ(positions[0], 2);
    ASSERT_EQ(positions[1], 1);
    ASSERT_EQ(gl::CompiledDefinitionFixture::orCounter(analyzer.definitions), 0);
}

// One or per canonical body (D-556): the or
// minted for one disjunct order is reused, with its argument permutation, for
// the same disjunction in the other order, by the next mint and by a
// structural compile; nothing else is minted.
TEST(compilation_state, or_in_another_disjunct_order_reuses_the_minted_or) {
    gl::tests::ProverFixture analyzer("Peano");
    clearCompiler(analyzer);
    const auto orCount = [&analyzer] {
        std::size_t count = 0;
        for (const auto& entry : gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions))
            count += entry.second.category == "or" ? 1 : 0;
        return count;
    };
    bool minted = false;
    const std::string name = analyzer.orRuleCompiler.findOrMintOrOperator(
        {"(in[u_1,u_2])", "(=[u_2,u_3])"}, 3, &minted);
    ASSERT_TRUE(minted);
    ASSERT_EQ(name, std::string("or0"));
    // The disjuncts swapped and renumbered: x = u_3, y = u_1, z = u_2.
    const std::vector<std::string> swapped = {"(=[u_1,u_2])", "(in[u_3,u_1])"};
    bool mintedAgain = true;
    std::vector<int> permutation;
    ASSERT_EQ(analyzer.orRuleCompiler.findOrMintOrOperator(swapped, 3, &mintedAgain, &permutation), name);
    ASSERT_FALSE(mintedAgain);
    ASSERT_TRUE(permutation == std::vector<int>({3, 1, 2}));
    ASSERT_EQ(gl::CompiledDefinitionFixture::orCounter(analyzer.definitions), 1);
    ASSERT_EQ(orCount(), std::size_t(1));
    gl::StrSpan spans[] = {gl::StrSpan(swapped[0]), gl::StrSpan(swapped[1])};
    int positions[3] = {};
    const auto* retained = analyzer.compilationReuse.compiledOrByElements(spans, 2, positions, 3);
    ASSERT_TRUE(retained != nullptr);
    ASSERT_EQ(*retained, name);
    ASSERT_EQ(positions[0], 3);
    ASSERT_EQ(positions[1], 1);
    ASSERT_EQ(positions[2], 2);
    // The mint registered its body, so a compile of the swapped disjunction
    // names the same or with the arguments in its signature order.
    std::vector<gl::EncodedExpression> encoded;
    encoded.emplace_back("(=[u_y,u_z])", "main");
    encoded.emplace_back("(in[u_x,u_y])", "main");
    ASSERT_EQ(analyzer.compilationReuse.excludeRepetitions(encoded, {}, "or7", "or", false, ""),
        std::string("(or0[x,y,z])"));
    ASSERT_EQ(gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions).count("or7"), std::size_t(0));
    ASSERT_EQ(orCount(), std::size_t(1));
}

TEST(compilation_state, shared_registry_without_atomic_definitions_loads_without_minting_them) {
    namespace fs = std::filesystem;
    gl::tests::ProverFixture analyzer("Peano");
    clearCompiler(analyzer);
    const auto path = fs::temp_directory_path() / "GL_binary_compilation_shared_fixture.json";
    nlohmann::json entry = {{"category", "existence"}, {"arity", 4},
        {"signature", "(preorder[u_1,u_2,u_3,u_4])"},
        {"elements", {"(in[1,u_1])", "(in3[u_3,1,u_4,u_2])"}},
        {"implications", nlohmann::json::array()},
        {"integration_implications", nlohmann::json::array()}};
    { std::ofstream output(path); output << nlohmann::json({{"preorder", entry}}).dump(); }
    gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions).clear();
    analyzer.compilationPersistence.loadGlBinary(path);
    ASSERT_EQ(gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions).size(), std::size_t(1));
    ASSERT_EQ(gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions).count("in"), std::size_t(0));
    ASSERT_EQ(gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions).count("in3"), std::size_t(0));
    ASSERT_FALSE(analyzer.compilationReuse.publicEntries().empty());
    std::vector<gl::EncodedExpression> expressions;
    expressions.emplace_back("(unknown[])", "main");
    expressions.emplace_back("!(unknown2[u_a,u_a])", "main");
    const auto normalized = gl::CompilationNormalizer::makeNormalizedEncodedKey(expressions, 2);
    const auto split = analyzer.compilationReuse.splitCompilationKey(expressions, normalized.first.data);
    ASSERT_TRUE(split == std::vector<std::string>({"(unknown[])", "!(unknown2[u_1,u_1])"}));
    fs::remove(path);
}

TEST(compilation_state, replacement_chain_is_composed_before_intermediate_definitions_are_erased) {
    gl::tests::ProverFixture analyzer("Peano");
    clearCompiler(analyzer);
    gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions)["existence15"] = preorder("existence15", true);
    gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions)["existence16"] = preorder("existence16", true);
    gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions)["preorder"] = preorder("preorder", false);
    analyzer.compilationReuse.registerLoadedNamedCompilation("preorder");
    analyzer.compilationReuse.restoreReplacement("existence15", {"existence16", {1, 2, 3, 4}});
    analyzer.compilationReuse.restoreReplacement("existence16", {"preorder", {1, 4, 2, 3}});
    analyzer.compilationReuse.finalizeCompilationReplacements();
    ASSERT_EQ(gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions).count("existence15"), std::size_t(0));
    ASSERT_EQ(gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions).count("existence16"), std::size_t(0));
    ASSERT_EQ(analyzer.compilationReuse.replacements().at("existence15").target, std::string("preorder"));
    ASSERT_TRUE(analyzer.compilationReuse.replacements().at("existence15").arguments == std::vector<int>({1, 4, 2, 3}));
}

TEST(compilation_reuse, owns_metadata_and_exposes_ordered_read_views) {
    gl::CompiledDefinitionRegistry definitions;
    gl::CompiledDefinitionFixture::compiledExpressions(definitions)["publicOr"] = gl::LogicalEntity("or", {"(p[u_1])", "(q[u_1])"}, "(publicOr[u_1])", 1);
    std::map<int, std::vector<std::vector<int>>> permutations;
    gl::CompilationReuse reuse(definitions, permutations);
    const gl::CompilationReuseKey key{{"(p[u_1])", "(q[u_1])"}, "or"};
    const gl::CompilationReuseValue value{{{"u_1","u_1"}}, "publicOr", key.first};
    reuse.registerCompilationReuse(key, value, true);
    ASSERT_EQ(reuse.entries().size(), std::size_t(1));
    ASSERT_EQ(reuse.publicEntries().size(), std::size_t(1));
    reuse.rememberOrReuse(key.first, "publicOr", {1});
    const gl::StrSpan elements[]{gl::StrSpan(key.first[0]), gl::StrSpan(key.first[1])};
    int positions[1] = {};
    const auto* retained = reuse.compiledOrByElements(elements, 2, positions, 1);
    ASSERT_TRUE(retained != nullptr);
    ASSERT_EQ(*retained, std::string("publicOr"));
    ASSERT_EQ(positions[0], 1);
    ASSERT_TRUE(reuse.compiledOrByElements(elements, 1) == nullptr);
    reuse.restoreReplacement("or12", {"publicOr", {1}});
    reuse.restoreReplacement("or12", {"publicOr", {1}});
    ASSERT_EQ(reuse.replacements().size(), std::size_t(1));
    ASSERT_FALSE(reuse.publicNamesFrozen());
    reuse.freezePublicNames();
    reuse.freezePublicNames();
    ASSERT_TRUE(reuse.publicNamesFrozen());
    reuse.clear();
    ASSERT_TRUE(reuse.entries().empty());
    ASSERT_TRUE(reuse.publicEntries().empty());
    ASSERT_TRUE(reuse.replacements().empty());
    ASSERT_EQ(definitions.entries().size(), std::size_t(1));
    ASSERT_TRUE(reuse.publicNamesFrozen());
}

TEST(compilation_reuse, state_round_trips_through_restore) {
    gl::CompiledDefinitionRegistry definitions;
    gl::CompiledDefinitionFixture::compiledExpressions(definitions)["publicOr"] = gl::LogicalEntity("or", {"(p[u_1])", "(q[u_1])"}, "(publicOr[u_1])", 1);
    std::map<int, std::vector<std::vector<int>>> permutations;
    gl::CompilationReuse source(definitions, permutations);
    const gl::CompilationReuseKey key{{"(p[u_1])", "(q[u_1])"}, "or"};
    const gl::CompilationReuseValue value{{{"u_1","u_1"}}, "publicOr", key.first};
    source.registerCompilationReuse(key, value, true);
    source.rememberOrReuse(key.first, "publicOr", {1});
    source.restoreReplacement("or12", {"publicOr", {1}});
    source.freezePublicNames();
    const gl::CompilationReuseState recorded = source.state();

    // A fresh component bound to the same registry takes over every index,
    // the replacement metadata and the frozen phase.
    gl::CompilationReuse target(definitions, permutations);
    ASSERT_TRUE(target.entries().empty());
    ASSERT_FALSE(target.publicNamesFrozen());
    target.restoreState(recorded);
    ASSERT_TRUE(target.entries() == source.entries());
    ASSERT_TRUE(target.publicEntries() == source.publicEntries());
    ASSERT_TRUE(target.state().compiledOrReuse == source.state().compiledOrReuse);
    ASSERT_EQ(target.replacements().size(), std::size_t(1));
    ASSERT_EQ(target.replacements().at("or12").target, std::string("publicOr"));
    ASSERT_TRUE(target.publicNamesFrozen());
    const gl::StrSpan elements[]{gl::StrSpan(key.first[0]), gl::StrSpan(key.first[1])};
    int positions[1] = {};
    const auto* retained = target.compiledOrByElements(elements, 2, positions, 1);
    ASSERT_TRUE(retained != nullptr);
    ASSERT_EQ(*retained, std::string("publicOr"));
    ASSERT_EQ(positions[0], 1);

    // Restoring replaces, never merges: an empty state empties every index
    // and lifts the frozen phase. The registry is not touched, so the OR
    // still resolves through compiledOrByElements' registry scan.
    target.restoreState(gl::CompilationReuseState{});
    ASSERT_TRUE(target.entries().empty());
    ASSERT_TRUE(target.publicEntries().empty());
    ASSERT_TRUE(target.replacements().empty());
    ASSERT_TRUE(target.state().compiledOrReuse.empty());
    ASSERT_TRUE(target.state().namedCompilationOrder.empty());
    ASSERT_FALSE(target.publicNamesFrozen());
    const auto* fromRegistry = target.compiledOrByElements(elements, 2);
    ASSERT_TRUE(fromRegistry != nullptr);
    ASSERT_EQ(*fromRegistry, std::string("publicOr"));
    ASSERT_EQ(definitions.entries().size(), std::size_t(1));
    ASSERT_EQ(source.entries().size(), std::size_t(1));
}

TEST(compilation_reuse, a_rule_written_in_another_premise_order_reuses_its_name) {
    gl::tests::ProverFixture analyzer("Peano");
    clearCompiler(analyzer);
    const auto encode = [](std::initializer_list<const char*> texts) {
        std::vector<gl::EncodedExpression> out;
        for (const char* text : texts) out.emplace_back(text, "main");
        return out;
    };
    const auto first = analyzer.compilationReuse.excludeRepetitions(
        encode({"(in[u_a,u_N])", "(in[u_b,u_N])", "(in3[u_a,u_b,u_c,u_plus])"}), encode({"(in[u_c,u_N])"}),
        "implication1", "implication", false, "");
    ASSERT_EQ(first, std::string("(implication1[a,N,b,c,plus])"));
    ASSERT_EQ(analyzer.compilationReuse.entries().size(), std::size_t(1));
    // The same rule, its premises in another order, its variables renamed.
    const auto second = analyzer.compilationReuse.excludeRepetitions(
        encode({"(in3[u_y,u_x,u_z,u_plus])", "(in[u_x,u_N])", "(in[u_y,u_N])"}), encode({"(in[u_z,u_N])"}),
        "implication2", "implication", false, "");
    ASSERT_EQ(second, std::string("(implication1[y,N,x,z,plus])"));
    ASSERT_EQ(gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions).count("implication2"), std::size_t(0));
    ASSERT_EQ(analyzer.compilationReuse.entries().size(), std::size_t(1));
}

TEST(compilation_reuse, loaded_numbered_rule_is_found_whatever_its_stored_order) {
    namespace fs = std::filesystem;
    gl::tests::ProverFixture analyzer("Peano");
    clearCompiler(analyzer);
    // Stored with the in3 premise first: not the canonical order.
    const auto path = fs::temp_directory_path() / "GL_binary_canonical_reuse_fixture.json";
    nlohmann::json entry = {{"category", "implication"}, {"arity", 5},
        {"signature", "(implication7[u_1,u_2,u_3,u_4,u_5])"},
        {"elements", {"(in3[u_1,u_2,u_3,u_4])", "(in[u_1,u_5])", "(in[u_2,u_5])", "(in[u_3,u_5])"}}};
    { std::ofstream output(path); output << nlohmann::json({{"implication7", entry}}).dump(); }
    analyzer.compilationPersistence.loadGlBinary(path);
    ASSERT_EQ(analyzer.compilationReuse.entries().size(), std::size_t(1));
    std::vector<gl::EncodedExpression> tail{gl::EncodedExpression("(in[u_a,u_N])", "main"),
        gl::EncodedExpression("(in[u_b,u_N])", "main"), gl::EncodedExpression("(in3[u_a,u_b,u_c,u_plus])", "main")};
    std::vector<gl::EncodedExpression> head{gl::EncodedExpression("(in[u_c,u_N])", "main")};
    ASSERT_EQ(analyzer.compilationReuse.excludeRepetitions(tail, head, "implication8", "implication", false, ""),
              std::string("(implication7[a,b,c,plus,N])"));
    ASSERT_EQ(gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions).count("implication8"), std::size_t(0));
    fs::remove(path);
}

// The preparation passes' added-entry walk against a set of the names present
// at the recording: entries inserted around the recorded ones are added, an
// existing name assigned anew keeps its node and is not.
TEST(supporting_rule_compiler, entries_added_since_recording_match_the_name_set) {
    gl::CompiledExpressionMap map;
    const auto entity = [](const std::string& name) {
        return gl::LogicalEntity("atomic", {""}, "(" + name + "[u_1])", 1);
    };
    for (const char* name : { "b", "d", "f", "h" }) map[name] = entity(name);
    std::vector<const gl::LogicalEntity*> present;
    std::set<std::string> names;
    for (const auto& [name, recorded] : map) {
        present.push_back(&recorded);
        names.insert(name);
    }
    for (const char* name : { "a", "c", "e", "g", "i" }) map[name] = entity(name);
    map["d"] = entity("d2");

    std::vector<std::pair<std::string, bool>> visited;
    gl::SupportingRuleCompiler::forEachEntryWithAdded(map, present,
        [&](const std::string& name, gl::LogicalEntity&, bool added) { visited.emplace_back(name, added); });
    ASSERT_EQ(visited.size(), map.size());
    std::size_t index = 0;
    for (const auto& [name, unused] : map) {
        (void)unused;
        ASSERT_EQ(visited[index].first, name);
        ASSERT_EQ(visited[index].second, names.count(name) == 0);
        ++index;
    }

    // The visitor may change the entry it is given.
    gl::SupportingRuleCompiler::forEachEntryWithAdded(map, present,
        [](const std::string&, gl::LogicalEntity& visitedEntity, bool added) {
            if (added) visitedEntity.contrapositionPrepared = true;
        });
    ASSERT_TRUE(map.at("a").contrapositionPrepared);
    ASSERT_FALSE(map.at("b").contrapositionPrepared);

    // Recorded now: nothing is added. Recorded nothing: everything is.
    present.clear();
    for (const auto& [name, recorded] : map) { (void)name; present.push_back(&recorded); }
    int added = 0;
    gl::SupportingRuleCompiler::forEachEntryWithAdded(map, present,
        [&](const std::string&, gl::LogicalEntity&, bool isAdded) { added += isAdded ? 1 : 0; });
    ASSERT_EQ(added, 0);
    gl::SupportingRuleCompiler::forEachEntryWithAdded(map, {},
        [&](const std::string&, gl::LogicalEntity&, bool isAdded) { added += isAdded ? 1 : 0; });
    ASSERT_EQ(added, static_cast<int>(map.size()));
}

TEST(compiled_definition_registry, configure_preserves_definitions_and_exposes_const_views) {
    gl::CompiledDefinitionRegistry registry;
    gl::CompiledDefinitionFixture::compiledExpressions(registry)["known"] =
        gl::LogicalEntity("atomic", {""}, "(known[u_1])", 1);
    ce::CoreExpressionMap configurations;
    ce::CoreExpressionConfig operation(2, std::string(), "(operation[1,2])");
    operation.inputArgs = {"1"};
    operation.outputArgs = {"2"};
    configurations["operation"] = operation;
    configurations["predicate"] = ce::CoreExpressionConfig(1, std::string(), "(predicate[1])");
    registry.configure(configurations);
    const auto& view = registry;
    ASSERT_EQ(view.configurations().size(), std::size_t(2));
    ASSERT_EQ(view.operatorNames().size(), std::size_t(1));
    ASSERT_TRUE(view.operatorNames().count("operation") == 1);
    ASSERT_TRUE(view.coreConfig(gl::StrSpan("operation", 9)) == &view.configurations().at("operation"));
    ASSERT_TRUE(view.coreConfig(gl::StrSpan("missing", 7)) == nullptr);
    ASSERT_TRUE(view.compiledEntity(gl::StrSpan("known", 5)) == &view.entries().at("known"));
    ASSERT_TRUE(view.compiledEntity(gl::StrSpan("missing", 7)) == nullptr);
    ASSERT_TRUE(view.typedEqualityEntity(gl::StrSpan("P(1)", 4)) == nullptr);
    ASSERT_TRUE(view.configuredNames().empty());
    ASSERT_TRUE(view.baseNames().empty());
    ASSERT_EQ(view.nextImplication(), 0);
    ASSERT_EQ(view.nextExistence(), 0);
    ASSERT_EQ(view.nextConjunction(), 0);
    ASSERT_EQ(view.nextOr(), 0);
    ASSERT_EQ(view.nextVariable(), 0);
    int visited = 0;
    view.forEachCompiledOr([&](const std::string&, const gl::LogicalEntity&) { ++visited; return false; });
    ASSERT_EQ(visited, 0);
    registry.configure({});
    ASSERT_TRUE(view.configurations().empty());
    ASSERT_TRUE(view.operatorNames().empty());
    ASSERT_EQ(view.entries().size(), std::size_t(1));
}
