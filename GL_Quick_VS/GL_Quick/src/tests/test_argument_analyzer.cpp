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
#include "../compiler/argument_analyzer.hpp"
#include "../compiler/argument_type_parser.hpp"
#include <memory>

namespace {
ce::CoreExpressionMap argumentTypeConfigurations() {
    ce::CoreExpressionMap configs;
    ce::CoreExpressionConfig membership(2, std::string(), "(in[1,2])");
    membership.definitionSets = {{"1", {"(1)", false}}, {"2", {"P(1)", false}}};
    configs["in"] = membership;
    return configs;
}
}

TEST(argument_type_parser, preserves_binding_and_signed_tree_shapes) {
    const auto configs = argumentTypeConfigurations();
    for (const std::string text : {"(>[x](in[x,S])(in[x,T]))", "!(>[x](in[x,S])(in[x,T]))"}) {
        gl::ArgumentTypeParser parser(text, configs);
        std::unique_ptr<gl::AnalysisNode> root(parser.parse());
        ASSERT_TRUE(root->left != nullptr && root->right != nullptr);
        ASSERT_EQ(root->remainingArgsDefs.size(), std::size_t(2));
        ASSERT_EQ(root->remainingArgsDefs.at("S"), std::string("P(1)"));
        ASSERT_EQ(root->remainingArgsDefs.at("T"), std::string("P(1)"));
        ASSERT_TRUE(root->remainingArgsDefs.count("x") == 0);
        ASSERT_EQ(root->left->remainingArgsDefs.at("x"), std::string("(1)"));
        ASSERT_EQ(root->value, text[0] == '!' ? std::string("!>[x]") : std::string(">[x]"));
    }
    for (const std::string text : {"(&(in[x,S])!(in[y,S]))", "!(&(in[x,S])(in[y,S]))"}) {
        gl::ArgumentTypeParser parser(text, configs);
        std::unique_ptr<gl::AnalysisNode> root(parser.parse());
        ASSERT_EQ(root->remainingArgsDefs.size(), std::size_t(3));
        ASSERT_EQ(root->remainingArgsDefs.at("x"), std::string("(1)"));
        ASSERT_EQ(root->remainingArgsDefs.at("y"), std::string("(1)"));
        ASSERT_EQ(root->remainingArgsDefs.at("S"), std::string("P(1)"));
        ASSERT_EQ(root->value, text[0] == '!' ? std::string("!&") : std::string("&"));
    }
    const std::string unknown = "(unconfigured[x])";
    gl::ArgumentTypeParser parser(unknown, configs);
    std::unique_ptr<gl::AnalysisNode> root(parser.parse());
    ASSERT_TRUE(root->remainingArgsDefs.empty());
}

TEST(argument_analyzer, strips_whitespace_and_checks_configured_arguments) {
    const auto configs = argumentTypeConfigurations();
    gl::ArgumentAnalyzer analyzer(configs);
    ASSERT_TRUE(analyzer.analyze(" \t\r\n") == nullptr);
    std::unique_ptr<gl::AnalysisNode> root(analyzer.analyze(" ( in[ x, S ] ) "));
    ASSERT_EQ(root->remainingArgsDefs.size(), std::size_t(2));
    analyzer.checkDefinitionConsistency(root.get(), configs.at("in"), "(in[x,S])");
    analyzer.checkDefinitionConsistency(nullptr, configs.at("in"), "(in[x,S])");
    std::string set, type;
    ASSERT_FALSE(analyzer.isSetDefinition(nullptr, set, type));
    ASSERT_FALSE(analyzer.isSetDefinition(root.get(), set, type));
}

TEST(argument_analyzer, recognizes_mirrored_set_definition_without_self_reference) {
    const auto configs = argumentTypeConfigurations();
    gl::ArgumentAnalyzer analyzer(configs);
    std::string set, type;
    std::unique_ptr<gl::AnalysisNode> root(analyzer.analyze(
        "(&(>[x](in[x,S])(in[x,T]))(>[x](in[x,T])(in[x,S])))"));
    ASSERT_TRUE(analyzer.isSetDefinition(root.get(), set, type));
    ASSERT_EQ(set, std::string("S"));
    ASSERT_EQ(type, std::string("P(1)"));
    std::unique_ptr<gl::AnalysisNode> selfReference(analyzer.analyze(
        "(&(>[x](in[x,S])(in[x,S]))(>[x](in[x,S])(in[x,S])))"));
    ASSERT_FALSE(analyzer.isSetDefinition(selfReference.get(), set, type));
    std::unique_ptr<gl::AnalysisNode> unmatched(analyzer.analyze(
        "(&(>[x](in[x,S])(in[x,T]))(>[x](in[x,T])(in[x,U])))"));
    ASSERT_FALSE(analyzer.isSetDefinition(unmatched.get(), set, type));
}
