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

/// @brief Unit tests for the or-implication compacts
///        (D-309): an or's K-rule and subset-exclusion
///        rules stored as `(implication<N>[u_…])` instances on its registry
///        entry, compiled at the or's mint / the reduced-or closure seam,
///        round-tripped through the GL binary, and completed at the seam
///        for an or loaded without the field.

#include "test_harness.hpp"

#include "../prover.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace {

    /// @brief Core name of a compact `(name[args])` / `!(name[args])`.
    std::string coreOf(const std::string& compact) {
        const std::size_t open = compact.find('(');
        const std::size_t bracket = compact.find('[');
        return compact.substr(open + 1, bracket - open - 1);
    }

    /// @brief Expand a compact `(implication<N>[args])` through its registry
    ///        entry exactly as the disintegrator's implication branch does:
    ///        signature tokens to the instance's arguments positionally, then
    ///        the full-bind reconstruction over the substituted elements.
    ///        Returns an empty string when the entry is missing or is not an
    ///        implication (the test asserts on it).
    std::string expandCompact(gl::ExpressionAnalyzer& ea,
                              const std::string& compact) {
        const gl::LogicalEntity* le =
            ea.compiledEntity(gl::StrSpan(coreOf(compact)));
        if (le == nullptr || le->category != "implication") return "";
        const std::vector<std::string> sigArgs = ce::getArgs(le->signature);
        const std::vector<std::string> instArgs = ce::getArgs(compact);
        if (sigArgs.size() != instArgs.size()) return "";
        std::map<std::string, std::string> subst;
        for (std::size_t i = 0; i < sigArgs.size(); ++i) {
            subst[sigArgs[i]] = instArgs[i];
        }
        std::vector<std::string> elems;
        for (const std::string& e : le->elements) {
            elems.push_back(ce::replaceKeysInString(e, subst));
        }
        if (elems.empty()) return "";
        const std::vector<std::string> premises(elems.begin(), elems.end() - 1);
        return ea.reconstructImplicationFullBind(premises, elems.back());
    }

    /// @brief The on-the-spot K-rule text for leaf @p i of @p leaves (the
    ///        disintegrator's former construction).
    std::string kRuleText(gl::ExpressionAnalyzer& ea,
                          const std::vector<std::string>& leaves,
                          std::size_t i) {
        std::vector<std::string> premises;
        for (std::size_t j = 0; j < leaves.size(); ++j) {
            if (j != i) premises.push_back(ea.negate(leaves[j]));
        }
        return ea.reconstructImplicationFullBind(premises, leaves[i]);
    }

    /// @brief A three-leaf or over two tokens, one leaf negated (I-175):
    ///        `(in[u_1,u_2]) OR (=[u_1,u_2]) OR !(in[u_2,u_1])`.
    const std::vector<std::string>& threeLeaves() {
        static const std::vector<std::string> leaves = {
            "(in[u_1,u_2])", "(=[u_1,u_2])", "!(in[u_2,u_1])" };
        return leaves;
    }

} // namespace

TEST(or_implications, expected_count_formula) {
    ASSERT_EQ(gl::ExpressionAnalyzer::expectedOrImplicationCount(2), 2);
    ASSERT_EQ(gl::ExpressionAnalyzer::expectedOrImplicationCount(3), 6);
    ASSERT_EQ(gl::ExpressionAnalyzer::expectedOrImplicationCount(4), 14);
    ASSERT_EQ(gl::ExpressionAnalyzer::expectedOrImplicationCount(8), 254);
}

TEST(or_implications, k_rules_compiled_at_mint_expand_to_todays_rules) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    const std::vector<std::string>& leaves = threeLeaves();
    const std::string name = ea.findOrMintOrOperator(leaves, 2);
    const gl::LogicalEntity* le = ea.compiledEntity(gl::StrSpan(name));
    ASSERT_TRUE(le != nullptr);
    // At the mint: exactly the K-rules, in leaf order.
    ASSERT_EQ(static_cast<int>(le->implications.size()), 3);
    for (std::size_t i = 0; i < leaves.size(); ++i) {
        const std::string& compact = le->implications[i];
        ASSERT_TRUE(compact.rfind("(implication", 0) == 0);
        // Every argument is one of the or's tokens (u_-prefixed projection).
        for (const std::string& a : ce::getArgs(compact)) {
            ASSERT_TRUE(a == "u_1" || a == "u_2");
        }
        ASSERT_EQ(static_cast<int>(ce::getArgs(compact).size()), 2);
        ASSERT_TRUE(expandCompact(ea, compact) == kRuleText(ea, leaves, i));
    }
    // A two-leaf or: two K-rules, single-premise each.
    const std::vector<std::string> two = { "(=[u_1,u_2])", "(in[u_2,u_1])" };
    const std::string name2 = ea.findOrMintOrOperator(two, 2);
    const gl::LogicalEntity* le2 = ea.compiledEntity(gl::StrSpan(name2));
    ASSERT_TRUE(le2 != nullptr);
    ASSERT_EQ(static_cast<int>(le2->implications.size()), 2);
    ASSERT_TRUE(expandCompact(ea, le2->implications[0]) == kRuleText(ea, two, 0));
    ASSERT_TRUE(expandCompact(ea, le2->implications[1]) == kRuleText(ea, two, 1));
    // Idempotent on a registry reuse: the same list, no growth.
    const std::string again = ea.findOrMintOrOperator(leaves, 2);
    ASSERT_TRUE(again == name);
    ASSERT_EQ(static_cast<int>(ea.compiledEntity(gl::StrSpan(name))->implications.size()), 3);
}

TEST(or_implications, subset_exclusions_appended_at_closure_seam) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    const std::vector<std::string>& leaves = threeLeaves();
    const std::string name = ea.findOrMintOrOperator(leaves, 2);
    ea.preMintReducedOrs();
    const gl::LogicalEntity* le = ea.compiledEntity(gl::StrSpan(name));
    ASSERT_TRUE(le != nullptr);
    // k = 3: 3 K-rules + 3 single-exclusion rules (j = 1 only).
    ASSERT_EQ(static_cast<int>(le->implications.size()), 6);
    for (std::size_t i = 0; i < 3; ++i) {
        ASSERT_TRUE(expandCompact(ea, le->implications[i]) == kRuleText(ea, leaves, i));
    }
    // Subset rule for excluded index s: !leaf_s -> the reduced or over the
    // two survivors in parent order, the head instantiated over the parent's
    // tokens in first-appearance order across the survivors.
    for (std::size_t s = 0; s < 3; ++s) {
        std::vector<std::string> reduced;
        for (std::size_t j = 0; j < 3; ++j) {
            if (j != s) reduced.push_back(leaves[j]);
        }
        std::vector<std::string> tokOrder;
        const int arity = gl::ExpressionAnalyzer::renumberULeaves(reduced, &tokOrder);
        const std::string redName = ea.findOrMintOrOperator(reduced, arity);
        std::string head = "(" + redName + "[";
        for (std::size_t t = 0; t < tokOrder.size(); ++t) {
            if (t > 0) head += ',';
            head += tokOrder[t];
        }
        head += "])";
        const std::vector<std::string> premises = { ea.negate(leaves[s]) };
        const std::string expected = ea.reconstructImplicationFullBind(premises, head);
        ASSERT_TRUE(expandCompact(ea, le->implications[3 + s]) == expected);
    }
    // The seam is idempotent.
    ea.preMintReducedOrs();
    ASSERT_EQ(static_cast<int>(ea.compiledEntity(gl::StrSpan(name))->implications.size()), 6);
    // The reduced ors carry their own (two-leaf) lists.
    ea.forEachCompiledOr([&](const std::string& n, const gl::LogicalEntity& e) {
        (void)n;
        ASSERT_TRUE(!e.implications.empty());
        return false;
    });
}

TEST(or_implications, binary_round_trip_and_seam_completion) {
    namespace fs = std::filesystem;
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    const std::vector<std::string>& leaves = threeLeaves();
    const std::string name = ea.findOrMintOrOperator(leaves, 2);
    ea.preMintReducedOrs();
    const std::vector<std::string> expectedList =
        ea.compiledEntity(gl::StrSpan(name))->implications;
    ASSERT_EQ(static_cast<int>(expectedList.size()), 6);

    const fs::path dir = fs::temp_directory_path() / "gl_or_implications_test";
    fs::create_directories(dir);
    ea.exportCompiledExpressionsJSON(dir);
    const fs::path binPath = dir / "GL_binary_Peano.json";
    ASSERT_TRUE(fs::exists(binPath));

    // The key is written only for ors: read the file back and check.
    nlohmann::json root;
    {
        std::ifstream f(binPath.string());
        f >> root;
    }
    ASSERT_TRUE(root[name].contains("implications"));
    ASSERT_EQ(static_cast<int>(root[name]["implications"].size()), 6);
    for (auto it = root.begin(); it != root.end(); ++it) {
        const std::string cat = it.value().value("category", std::string());
        if (cat != "or" && cat != "existence") {
            ASSERT_FALSE(it.value().contains("implications"));
        }
    }

    // Round trip: a fresh analyzer loads the same list.
    gl::ExpressionAnalyzer ea2(std::string("Peano"));
    ea2.loadGlBinary(binPath);
    const gl::LogicalEntity* le2 = ea2.compiledEntity(gl::StrSpan(name));
    ASSERT_TRUE(le2 != nullptr);
    ASSERT_TRUE(le2->implications == expectedList);
    ASSERT_TRUE(ea2.orsAwaitingImplications.empty());

    // A binary written before the field existed: strip the key from every
    // or, load, and let the closure seam complete the lists deterministically.
    nlohmann::json legacy = root;
    for (auto it = legacy.begin(); it != legacy.end(); ++it) {
        if (it.value().contains("implications")) it.value().erase("implications");
    }
    const fs::path legacyPath = dir / "GL_binary_legacy.json";
    {
        std::ofstream f(legacyPath.string());
        f << legacy.dump(2);
    }
    gl::ExpressionAnalyzer ea3(std::string("Peano"));
    ea3.loadGlBinary(legacyPath);
    ASSERT_TRUE(ea3.orsAwaitingImplications.count(name) == 1);
    ASSERT_TRUE(ea3.compiledEntity(gl::StrSpan(name))->implications.empty());
    ea3.preMintReducedOrs();
    ASSERT_TRUE(ea3.orsAwaitingImplications.empty());
    ASSERT_TRUE(ea3.compiledEntity(gl::StrSpan(name))->implications == expectedList);

    std::remove(binPath.string().c_str());
    std::remove(legacyPath.string().c_str());
}

namespace {

    /// @brief Drop every `u_` prefix — the disintegrator's rules carry the
    ///        deposit's arguments u_-prefixed; the comparison is on the bare
    ///        names.
    std::string stripU(const std::string& s) {
        std::string out;
        for (std::size_t i = 0; i < s.size(); ++i) {
            if (s[i] == 'u' && i + 1 < s.size() && s[i + 1] == '_') { ++i; continue; }
            out += s[i];
        }
        return out;
    }

} // namespace

TEST(or_implications, cohort_consumes_compacts_with_repeated_arguments) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    // Three tokens, two leaves; the instance below repeats one argument.
    const std::vector<std::string> leaves = { "(in[u_1,u_2])", "(in2[u_1,u_3,u_2])" };
    const std::string name = ea.findOrMintOrOperator(leaves, 3);
    ea.preMintReducedOrs();
    const gl::LogicalEntity* le = ea.compiledEntity(gl::StrSpan(name));
    ASSERT_TRUE(le != nullptr);
    ASSERT_EQ(static_cast<int>(le->implications.size()), 2);

    // Deposit the or fact at the root LB's main scope, status 0:
    // u_1 -> 9, u_2 -> 10, u_3 -> 9 (repeated argument).
    const std::string inst = "(" + name + "[9,10,9])";
    const int lv[1] = { 0 };
    const gl::TransientOrigin origin{ true, gl::OriginTag::taskFormulation, nullptr, 0 };
    ea.addExprToMemoryBlock(gl::StrSpan(inst), ea.body, 0, 0, lv, 1, origin,
        -1, -1, gl::StrSpan("main", 4), false);

    // Both compacts are registered statements at main (u_-stripped), with
    // the repeated argument in the positions the template projects.
    const std::map<std::string, std::string> subst = {
        { "u_1", "9" }, { "u_2", "10" }, { "u_3", "9" } };
    for (const std::string& tmpl : le->implications) {
        const std::string compact = ce::replaceKeysInString(tmpl, subst);
        ASSERT_TRUE(compact.rfind("(implication", 0) == 0);
        const gl::StatementFlags* row = gl::lookupStatementFlags(
            ea.body.intKnownStatements, ea.body.nameMap,
            gl::StrSpan(compact), gl::StrSpan("main", 4));
        ASSERT_TRUE(row != nullptr);
    }

    // Hash memory holds exactly the rules the on-the-spot construction
    // produced: !in2(9,9,10) -> in(9,10) and !in(9,10) -> in2(9,9,10).
    bool k1 = false, k2 = false;
    const int32_t n = ea.body.overallHashMemory.originals.count();
    for (int32_t oi = 1; oi <= n; ++oi) {
        const auto c = gl::decodeValueVector(
            ea.body.overallHashMemory.originals.decodeKey(oi).ids,
            ea.body.ruleInterner);
        if (c.size() != 2) continue;
        const std::string prem = stripU(c[0]);
        const std::string head = stripU(c[1]);
        if (prem == "!(in2[9,9,10])" && head == "(in[9,10])") k1 = true;
        if (prem == "!(in[9,10])" && head == "(in2[9,9,10])") k2 = true;
    }
    ASSERT_TRUE(k1);
    ASSERT_TRUE(k2);
}
