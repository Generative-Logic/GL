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
#include "../prover.hpp"
#include <fstream>

namespace {
std::string symmetricTestOr(gl::ExpressionAnalyzer& analyzer, bool equality = true) {
    std::vector<std::string> leaves{"(in2[u_1,u_2,u_3])", "(in2[u_2,u_1,u_3])"};
    if (equality) leaves.insert(leaves.begin() + 1, "(=[u_1,u_2])");
    return analyzer.findOrMintOrOperator(leaves, 3);
}

void clearCompiler(gl::ExpressionAnalyzer& analyzer) {
    analyzer.compiledExpressions.clear();
    analyzer.repetitionExclusionMap.clear();
    analyzer.namedRepetitionExclusionMap.clear();
    analyzer.compiledOrReuse.clear();
    analyzer.namedCompilationOrder.clear();
    analyzer.namedCompilationNames.clear();
    analyzer.compilationReplacements.clear();
    analyzer.orsAwaitingImplications.clear();
    analyzer.existencesAwaitingImplications.clear();
    analyzer.implCounter = analyzer.existenceCounter = analyzer.andCounter = analyzer.orCounter = 0;
    analyzer.compiledExpressions["in"] = gl::LogicalEntity("atomic", {""}, "(in[u_1,u_2])", 2);
    analyzer.compiledExpressions["in3"] = gl::LogicalEntity("atomic", {""}, "(in3[u_1,u_2,u_3,u_4])", 4);
    analyzer.compiledExpressions["="] = gl::LogicalEntity("atomic", {""}, "(=[u_1,u_2])", 2);
}

gl::LogicalEntity preorder(const std::string& name, bool numbered) {
    return gl::LogicalEntity("existence", numbered
        ? std::vector<std::string>{"(in[1,u_1])", "(in3[u_2,1,u_3,u_4])"}
        : std::vector<std::string>{"(in[1,u_1])", "(in3[u_3,1,u_4,u_2])"},
        "(" + name + "[u_1,u_2,u_3,u_4])", 4);
}

std::string compile(gl::ExpressionAnalyzer& analyzer, const std::string& expression) {
    return analyzer.compileCoreExpressionMapCore(expression, analyzer.implCounter,
        analyzer.existenceCounter, analyzer.andCounter, analyzer.orCounter, analyzer.variableCounter);
}
} // namespace

TEST(compilation_state, symmetric_or_marks_typed_totality_and_trichotomy) {
    gl::ExpressionAnalyzer analyzer("Peano");
    for (bool equality : {false, true}) {
        const auto name = symmetricTestOr(analyzer, equality);
        const auto& entity = analyzer.compiledExpressions.at(name);
        ASSERT_EQ(entity.symmetricDigitSlots[0], 1);
        ASSERT_EQ(entity.symmetricDigitSlots[1], 2);
        auto withoutMetadata = entity;
        withoutMetadata.symmetricDigitSlots = {0, 0};
        ASSERT_TRUE(entity == withoutMetadata);
    }
    analyzer.compiledExpressions["or999"] = gl::LogicalEntity("or",
        {"(in2[u_1,u_2,u_3])", "(in2[u_2,u_1,u_3])"}, "(or999[u_1,u_2,u_3])", 3);
    analyzer.compiledExpressions["or998"] = gl::LogicalEntity("or",
        {"(or999[u_1,u_2,u_3])", "!(=[u_1,u_2])"}, "(or998[u_1,u_2,u_3])", 3);
    analyzer.compileOrSymmetry("or998");
    ASSERT_EQ(analyzer.compiledExpressions.at("or998").symmetricDigitSlots[1], 2);
}

TEST(compilation_state, symmetric_or_rejects_order_polarity_multiplicity_and_three_digits) {
    gl::ExpressionAnalyzer analyzer("Peano");
    const std::vector<std::vector<std::string>> examples{
        {"(in2[u_1,u_2,u_3])", "(=[u_1,u_2])"},
        {"(in2[u_1,u_2,u_3])", "!(in2[u_2,u_1,u_3])"},
        {"(in2[u_1,u_2,u_3])", "(in2[u_1,u_2,u_3])", "(in2[u_2,u_1,u_3])"},
        {"(in3[u_1,u_2,u_3,u_4])", "(in3[u_2,u_1,u_3,u_4])"}};
    for (std::size_t i = 0; i < examples.size(); ++i) {
        const int arity = i == 3 ? 4 : 3;
        const std::string signature = arity == 4 ? "(or999[u_1,u_2,u_3,u_4])" : "(or999[u_1,u_2,u_3])";
        analyzer.compiledExpressions["or999"] = gl::LogicalEntity("or", examples[i], signature, arity);
        analyzer.compileOrSymmetry("or999");
        ASSERT_EQ(analyzer.compiledExpressions.at("or999").symmetricDigitSlots[0], 0);
    }
}

TEST(compilation_state, symmetric_or_metadata_rebuilt_from_unchanged_binary_shape) {
    gl::ExpressionAnalyzer analyzer("Peano");
    const auto name = symmetricTestOr(analyzer);
    analyzer.preMintReducedOrs();
    const auto original = analyzer.compiledExpressions.at(name);
    const auto path = std::filesystem::temp_directory_path() / "GL_binary_symmetric_or_fixture.json";
    nlohmann::json entry{{"category", original.category}, {"arity", original.arity},
        {"signature", original.signature}, {"elements", original.elements}, {"implications", original.implications}};
    { std::ofstream output(path); output << nlohmann::json({{name, entry}}).dump(); }
    analyzer.loadGlBinary(path);
    ASSERT_EQ(analyzer.compiledExpressions.at(name).symmetricDigitSlots[0], 0);
    analyzer.preMintReducedOrs();
    ASSERT_TRUE(analyzer.compiledExpressions.at(name).symmetricDigitSlots == original.symmetricDigitSlots);
    ASSERT_TRUE(analyzer.compiledExpressions.at(name) == original);
    std::filesystem::remove(path);
}

TEST(compilation_state, symmetric_or_processing_key_swaps_positions_not_values) {
    gl::ExpressionAnalyzer analyzer("Peano");
    const auto name = symmetricTestOr(analyzer);
    auto& strings = gl::scratchArenas().forSlot(gl::scratchArenas().slotCount() - 1);
    gl::ScratchScope scope(strings);
    const std::string text = "(" + name + "[z,a,z])";
    ASSERT_EQ(analyzer.canonicalOrProcessingForm(gl::StrSpan(text), strings).toStdString(),
        "(" + name + "[a,z,z])");
    const std::string equal = "(" + name + "[a,a,f])";
    ASSERT_EQ(analyzer.canonicalOrProcessingForm(gl::StrSpan(equal), strings).toStdString(), equal);
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
    gl::ExpressionAnalyzer analyzer("Peano");
    clearCompiler(analyzer);
    analyzer.compiledExpressions["existence16"] = preorder("existence16", true);
    for (const auto& entry : analyzer.compilationReuseEntries("existence16"))
        analyzer.registerCompilationReuse(entry.first, entry.second, false);
    analyzer.compiledExpressions["strictOrder"] = gl::LogicalEntity("and",
        {"(existence16[u_1,u_3,u_4,u_2])", "!(=[u_3,u_4])"},
        "(strictOrder[u_1,u_2,u_3,u_4])", 4);
    analyzer.compiledExpressions["preorder"] = preorder("preorder", false);
    analyzer.registerLoadedNamedCompilation("preorder");
    analyzer.finalizeCompilationReplacements();
    ASSERT_EQ(analyzer.compiledExpressions.count("existence16"), std::size_t(0));
    ASSERT_EQ(analyzer.compiledExpressions.at("strictOrder").elements[0],
              std::string("(preorder[u_1,u_2,u_3,u_4])"));
    ASSERT_TRUE(analyzer.compilationReplacements.at("existence16").arguments == std::vector<int>({1, 4, 2, 3}));
    for (const auto& entry : analyzer.repetitionExclusionMap)
        ASSERT_TRUE(std::get<1>(entry.second) != "existence16");
    const auto size = analyzer.repetitionExclusionMap.size();
    analyzer.finalizeCompilationReplacements();
    ASSERT_EQ(analyzer.repetitionExclusionMap.size(), size);
}

TEST(compilation_state, first_public_name_wins_and_later_numbered_definition_is_removed) {
    gl::ExpressionAnalyzer analyzer("Peano");
    clearCompiler(analyzer);
    analyzer.compiledExpressions["zFirst"] = preorder("zFirst", false);
    analyzer.registerLoadedNamedCompilation("zFirst");
    analyzer.compiledExpressions["aSecond"] = preorder("aSecond", false);
    analyzer.registerLoadedNamedCompilation("aSecond");
    analyzer.compiledExpressions["existence16"] = preorder("existence16", true);
    analyzer.finalizeCompilationReplacements();
    ASSERT_EQ(analyzer.compiledExpressions.count("zFirst"), std::size_t(1));
    ASSERT_EQ(analyzer.compiledExpressions.count("aSecond"), std::size_t(1));
    ASSERT_EQ(analyzer.compiledExpressions.count("existence16"), std::size_t(0));
    ASSERT_EQ(analyzer.compilationReplacements.at("existence16").target, std::string("zFirst"));
    for (const auto& entry : analyzer.namedRepetitionExclusionMap)
        ASSERT_EQ(std::get<1>(entry.second), std::string("zFirst"));
}

TEST(compilation_state, categories_and_implication_heads_remain_distinct) {
    gl::ExpressionAnalyzer analyzer("Peano");
    clearCompiler(analyzer);
    analyzer.compiledExpressions["preorder"] = preorder("preorder", false);
    analyzer.registerLoadedNamedCompilation("preorder");
    auto other = preorder("implication16", true);
    other.category = "implication";
    analyzer.compiledExpressions["implication16"] = other;
    analyzer.finalizeCompilationReplacements();
    ASSERT_EQ(analyzer.compiledExpressions.count("implication16"), std::size_t(1));
    for (const auto& entry : analyzer.compilationReuseEntries("implication16"))
        ASSERT_EQ(entry.first.first.back(), std::string("(in3[u_2,1,u_3,u_4])"));
}

TEST(compilation_state, loaded_preorder_is_reused_by_fresh_strict_order_compilation) {
    gl::ExpressionAnalyzer analyzer("Peano");
    clearCompiler(analyzer);
    analyzer.compiledExpressions["preorder"] = preorder("preorder", false);
    analyzer.registerLoadedNamedCompilation("preorder");
    const std::string body = "!(>[p](in[p,N])!(in3[n,p,m,plus]))";
    const std::string result = compile(analyzer, body);
    ASSERT_EQ(result, std::string("(preorder[N,plus,n,m])"));
    ASSERT_EQ(analyzer.existenceCounter, 0);
    ce::CoreExpressionConfig config;
    config.signature = "(strictOrder[N,plus,n,m])";
    config.arity = 4;
    config.definition = "(&" + body + "!(=[n,m]))";
    analyzer.coreExpressionMap["strictOrder"] = config;
    ASSERT_EQ(compile(analyzer, config.signature), config.signature);
    ASSERT_EQ(analyzer.compiledExpressions.at("strictOrder").elements[0],
              std::string("(preorder[u_1,u_2,u_3,u_4])"));
}

TEST(compilation_state, load_order_round_trip_and_counter_high_water_survive_retirement) {
    namespace fs = std::filesystem;
    gl::ExpressionAnalyzer analyzer("Peano");
    clearCompiler(analyzer);
    analyzer.compiledExpressions["existence16"] = preorder("existence16", true);
    analyzer.compiledExpressions["zLast"] = preorder("zLast", false);
    analyzer.compiledExpressions["aFirst"] = preorder("aFirst", false);
    analyzer.existenceCounter = 30;
    const auto directory = fs::temp_directory_path() / "gl_compilation_state_test";
    fs::create_directories(directory);
    analyzer.exportCompiledExpressionsJSON(directory);
    clearCompiler(analyzer);
    analyzer.loadGlBinary(directory / "GL_binary_Peano.json");
    ASSERT_EQ(analyzer.compilationReplacements.at("existence16").target, std::string("aFirst"));
    ASSERT_EQ(analyzer.existenceCounter, 30);
    ASSERT_EQ(analyzer.compiledExpressions.count("existence16"), std::size_t(0));
    analyzer.exportCompiledExpressionsJSON(directory);
    analyzer.loadGlBinary(directory / "GL_binary_Peano.json");
    ASSERT_EQ(analyzer.existenceCounter, 30);
    ASSERT_EQ(analyzer.compiledExpressions.count("existence16"), std::size_t(0));
    fs::remove(directory / "GL_binary_Peano.json");
    fs::remove(directory / "GL_compilation_state_Peano.json");
}

TEST(compilation_state, direct_or_reuse_returns_public_argument_order) {
    gl::ExpressionAnalyzer analyzer("Peano");
    clearCompiler(analyzer);
    analyzer.compiledExpressions["choice"] = gl::LogicalEntity("or",
        {"(in[u_2,u_1])", "!(=[u_1,u_2])"}, "(choice[u_1,u_2])", 2);
    analyzer.registerLoadedNamedCompilation("choice");
    const std::vector<std::string> elements = {"(in[u_1,u_2])", "!(=[u_2,u_1])"};
    bool minted = true;
    std::vector<int> permutation;
    ASSERT_EQ(analyzer.findOrMintOrOperator(elements, 2, &minted, &permutation), std::string("choice"));
    ASSERT_FALSE(minted);
    ASSERT_TRUE(permutation == std::vector<int>({2, 1}));
    gl::StrSpan spans[] = {gl::StrSpan(elements[0]), gl::StrSpan(elements[1])};
    int positions[2] = {};
    const auto* name = analyzer.compiledOrByElements(spans, 2, positions, 2);
    ASSERT_TRUE(name != nullptr);
    ASSERT_EQ(*name, std::string("choice"));
    ASSERT_EQ(positions[0], 2);
    ASSERT_EQ(positions[1], 1);
    ASSERT_EQ(analyzer.orCounter, 0);
}

TEST(compilation_state, shared_registry_without_atomic_definitions_loads_without_minting_them) {
    namespace fs = std::filesystem;
    gl::ExpressionAnalyzer analyzer("Peano");
    clearCompiler(analyzer);
    const auto path = fs::temp_directory_path() / "GL_binary_compilation_shared_fixture.json";
    nlohmann::json entry = {{"category", "existence"}, {"arity", 4},
        {"signature", "(preorder[u_1,u_2,u_3,u_4])"},
        {"elements", {"(in[1,u_1])", "(in3[u_3,1,u_4,u_2])"}}};
    { std::ofstream output(path); output << nlohmann::json({{"preorder", entry}}).dump(); }
    analyzer.compiledExpressions.clear();
    analyzer.loadGlBinary(path);
    ASSERT_EQ(analyzer.compiledExpressions.size(), std::size_t(1));
    ASSERT_EQ(analyzer.compiledExpressions.count("in"), std::size_t(0));
    ASSERT_EQ(analyzer.compiledExpressions.count("in3"), std::size_t(0));
    ASSERT_FALSE(analyzer.namedRepetitionExclusionMap.empty());
    std::vector<gl::EncodedExpression> expressions;
    expressions.emplace_back("(unknown[])", "main");
    expressions.emplace_back("!(unknown2[u_a,u_a])", "main");
    const auto normalized = analyzer.makeNormalizedEncodedKey(expressions, 2);
    const auto split = analyzer.splitCompilationKey(expressions, normalized.first.data);
    ASSERT_TRUE(split == std::vector<std::string>({"(unknown[])", "!(unknown2[u_1,u_1])"}));
    fs::remove(path);
}

TEST(compilation_state, replacement_chain_is_composed_before_intermediate_definitions_are_erased) {
    gl::ExpressionAnalyzer analyzer("Peano");
    clearCompiler(analyzer);
    analyzer.compiledExpressions["existence15"] = preorder("existence15", true);
    analyzer.compiledExpressions["existence16"] = preorder("existence16", true);
    analyzer.compiledExpressions["preorder"] = preorder("preorder", false);
    analyzer.registerLoadedNamedCompilation("preorder");
    analyzer.compilationReplacements["existence15"] = {"existence16", {1, 2, 3, 4}};
    analyzer.compilationReplacements["existence16"] = {"preorder", {1, 4, 2, 3}};
    analyzer.finalizeCompilationReplacements();
    ASSERT_EQ(analyzer.compiledExpressions.count("existence15"), std::size_t(0));
    ASSERT_EQ(analyzer.compiledExpressions.count("existence16"), std::size_t(0));
    ASSERT_EQ(analyzer.compilationReplacements.at("existence15").target, std::string("preorder"));
    ASSERT_TRUE(analyzer.compilationReplacements.at("existence15").arguments == std::vector<int>({1, 4, 2, 3}));
}
