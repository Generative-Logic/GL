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

/// @brief Unit tests for the existence-implication compacts
///        (D-310): the two rules a negated
///        existence disintegrates into, `left -> !right` and
///        `right -> !left`, stored as `(implication<N>[u_…])` instances on
///        the existence's registry entry, compiled at the existence's mint,
///        round-tripped through the GL binary, and completed at the seam
///        for an existence loaded without the field.

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

    /// @brief The on-the-spot rule text of an existence entity's rule @p i
    ///        (the disintegrator's former construction over the registry
    ///        elements): `[0]` = left -> !right, `[1]` = right -> !left.
    std::string existenceRuleText(gl::ExpressionAnalyzer& ea,
                                  const gl::LogicalEntity& le,
                                  std::size_t i) {
        const std::vector<std::string> premises = { le.elements[i] };
        return ea.reconstructImplicationFullBind(premises,
                                                 ea.negate(le.elements[1 - i]));
    }

    /// @brief Mint an existence through the compiler's `!(>` branch (the
    ///        theorem-load precompile door) and return its registry name.
    std::string mintExistence(gl::ExpressionAnalyzer& ea, std::string text) {
        ea.precompileStructuralOperators(text);
        return coreOf(text);
    }

} // namespace

TEST(existence_implications, expected_count_formula) {
    ASSERT_EQ(gl::ExpressionAnalyzer::expectedExistenceImplicationCount(2), 2);
    ASSERT_EQ(gl::ExpressionAnalyzer::expectedExistenceImplicationCount(1), 0);
    ASSERT_EQ(gl::ExpressionAnalyzer::expectedExistenceImplicationCount(0), 0);
    ASSERT_EQ(gl::ExpressionAnalyzer::expectedExistenceImplicationCount(3), 0);
}

TEST(existence_implications, compacts_compiled_at_mint_expand_to_todays_rules) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    const std::string name =
        mintExistence(ea, "!(>[x](in[x,a])!(in2[b,x,c]))");
    const gl::LogicalEntity* le = ea.compiledEntity(gl::StrSpan(name));
    ASSERT_TRUE(le != nullptr);
    ASSERT_TRUE(le->category == "existence");
    ASSERT_EQ(static_cast<int>(le->elements.size()), 2);
    const std::vector<std::string> sigArgs = ce::getArgs(le->signature);
    ASSERT_EQ(static_cast<int>(sigArgs.size()), 3);

    // At the mint: exactly the two rules, left -> !right then right -> !left.
    ASSERT_EQ(static_cast<int>(le->implications.size()), 2);
    for (std::size_t i = 0; i < 2; ++i) {
        const std::string& compact = le->implications[i];
        ASSERT_TRUE(compact.rfind("(implication", 0) == 0);
        // Every argument is one of the existence's tokens (u_-prefixed
        // projection), and the compact spans the full arity.
        const std::vector<std::string> args = ce::getArgs(compact);
        ASSERT_EQ(static_cast<int>(args.size()), 3);
        for (const std::string& a : args) {
            bool isToken = false;
            for (const std::string& s : sigArgs) isToken = isToken || (a == s);
            ASSERT_TRUE(isToken);
        }
        ASSERT_TRUE(expandCompact(ea, compact) == existenceRuleText(ea, *le, i));
    }
    // The two rules differ (premise and head swap).
    ASSERT_TRUE(expandCompact(ea, le->implications[0])
                != expandCompact(ea, le->implications[1]));

    // Idempotent on a registry reuse: the same entity, the same list.
    const std::string again =
        mintExistence(ea, "!(>[y](in[y,p])!(in2[q,y,r]))");
    ASSERT_TRUE(again == name);
    ASSERT_EQ(static_cast<int>(ea.compiledEntity(gl::StrSpan(name))->implications.size()), 2);
    ea.compileExistenceImplications(name);
    ASSERT_TRUE(ea.compiledEntity(gl::StrSpan(name))->implications == le->implications);

    // The seam is idempotent and leaves a complete list complete.
    ea.preMintReducedOrs();
    ASSERT_TRUE(ea.existencesAwaitingImplications.empty());
    ASSERT_EQ(static_cast<int>(ea.compiledEntity(gl::StrSpan(name))->implications.size()), 2);
}

TEST(existence_implications, binary_round_trip_and_seam_completion) {
    namespace fs = std::filesystem;
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    const std::string name =
        mintExistence(ea, "!(>[x](in[x,a])!(in2[b,x,c]))");
    const std::vector<std::string> expectedList =
        ea.compiledEntity(gl::StrSpan(name))->implications;
    ASSERT_EQ(static_cast<int>(expectedList.size()), 2);

    const fs::path dir = fs::temp_directory_path() / "gl_existence_implications_test";
    fs::create_directories(dir);
    ea.exportCompiledExpressionsJSON(dir);
    const fs::path binPath = dir / "GL_binary_Peano.json";
    ASSERT_TRUE(fs::exists(binPath));

    // The key is written for the existence; never for an implication /
    // atomic entry.
    nlohmann::json root;
    {
        std::ifstream f(binPath.string());
        f >> root;
    }
    ASSERT_TRUE(root[name].contains("implications"));
    ASSERT_EQ(static_cast<int>(root[name]["implications"].size()), 2);
    for (auto it = root.begin(); it != root.end(); ++it) {
        const std::string cat = it.value().value("category", std::string());
        if (cat != "or" && cat != "existence") {
            ASSERT_FALSE(it.value().contains("implications"));
        }
    }

    // Round trip: a fresh analyzer loads the same list, nothing awaits.
    gl::ExpressionAnalyzer ea2(std::string("Peano"));
    ea2.loadGlBinary(binPath);
    const gl::LogicalEntity* le2 = ea2.compiledEntity(gl::StrSpan(name));
    ASSERT_TRUE(le2 != nullptr);
    ASSERT_TRUE(le2->implications == expectedList);
    ASSERT_TRUE(ea2.existencesAwaitingImplications.count(name) == 0);

    // A binary written before the field existed: strip the key from every
    // entry, load, and let the seam complete the list deterministically.
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
    ASSERT_TRUE(ea3.existencesAwaitingImplications.count(name) == 1);
    ASSERT_TRUE(ea3.compiledEntity(gl::StrSpan(name))->implications.empty());
    ea3.preMintReducedOrs();
    ASSERT_TRUE(ea3.existencesAwaitingImplications.empty());
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

    /// @brief Rewrite every argument of a simple `(op[args])` / `!(op[args])`
    ///        that is not one of the deposit's arguments to `B` — the bound
    ///        variable's name is the registry's business, not the test's.
    std::string canonBound(const std::string& s,
                           const std::vector<std::string>& deposit) {
        const std::size_t open = s.find('[');
        const std::size_t close = s.find(']', open);
        std::string out = s.substr(0, open + 1);
        std::size_t pos = open + 1;
        while (pos <= close) {
            std::size_t comma = s.find(',', pos);
            if (comma == std::string::npos || comma > close) comma = close;
            const std::string arg = s.substr(pos, comma - pos);
            bool isDeposit = false;
            for (const std::string& d : deposit) isDeposit = isDeposit || (d == arg);
            out += isDeposit ? arg : std::string("B");
            if (comma == close) break;
            out += ',';
            pos = comma + 1;
        }
        out += s.substr(close);
        return out;
    }

} // namespace

TEST(existence_implications, negated_existence_disintegrates_into_compacts) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    const std::string name =
        mintExistence(ea, "!(>[x](in[x,a])!(in2[b,x,c]))");
    const gl::LogicalEntity* le = ea.compiledEntity(gl::StrSpan(name));
    ASSERT_TRUE(le != nullptr);
    ASSERT_EQ(static_cast<int>(le->implications.size()), 2);
    const std::vector<std::string> sigArgs = ce::getArgs(le->signature);
    ASSERT_EQ(static_cast<int>(sigArgs.size()), 3);

    // Deposit the negated existence at the root LB's main scope, status 0:
    // the existence's tokens -> 9, 7, 3 positionally.
    const std::vector<std::string> deposit = { "9", "7", "3" };
    const std::string inst = "!(" + name + "[9,7,3])";
    const int lv[1] = { 0 };
    const gl::TransientOrigin origin{ true, gl::OriginTag::taskFormulation, nullptr, 0 };
    ea.addExprToMemoryBlock(gl::StrSpan(inst), ea.body, 0, 0, lv, 1, origin,
        -1, -1, gl::StrSpan("main", 4), false);

    // Both compacts are registered statements at main (u_-stripped).
    std::map<std::string, std::string> subst;
    for (std::size_t i = 0; i < 3; ++i) subst[sigArgs[i]] = deposit[i];
    for (const std::string& tmpl : le->implications) {
        const std::string compact = ce::replaceKeysInString(tmpl, subst);
        ASSERT_TRUE(compact.rfind("(implication", 0) == 0);
        const gl::StatementFlags* row = gl::lookupStatementFlags(
            ea.body.intKnownStatements, ea.body.nameMap,
            gl::StrSpan(compact), gl::StrSpan("main", 4));
        ASSERT_TRUE(row != nullptr);
    }

    // Hash memory holds exactly the two rules the on-the-spot construction
    // produced: left -> !right and right -> !left over the deposit's
    // arguments, the bound variable in the witness slot.
    const std::string left =
        canonBound(ce::replaceKeysInString(le->elements[0], subst), deposit);
    const std::string right =
        canonBound(ce::replaceKeysInString(le->elements[1], subst), deposit);
    bool r1 = false, r2 = false;
    int rules = 0;
    const int32_t n = ea.body.overallHashMemory.originals.count();
    for (int32_t oi = 1; oi <= n; ++oi) {
        const auto c = gl::decodeValueVector(
            ea.body.overallHashMemory.originals.decodeKey(oi).ids,
            ea.body.ruleInterner);
        if (c.size() != 2) continue;
        const std::string prem = canonBound(stripU(c[0]), deposit);
        const std::string head = canonBound(stripU(c[1]), deposit);
        if (prem == left && head == "!" + right) r1 = true;
        if (prem == right && head == "!" + left) r2 = true;
        ++rules;
    }
    ASSERT_TRUE(r1);
    ASSERT_TRUE(r2);
    ASSERT_EQ(rules, 2);
}
