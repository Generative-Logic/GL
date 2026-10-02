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

TEST(contraposition, all_covering_combinations_preserve_defining_elements) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    std::string expression = "(&(>[x,y](in2[x,y,f])(in[x,A]))(&(>[x,y](in2[x,y,f])(in[y,B]))(>[x,y](in2[x,y,f])(in2[x,y,g]))))";
    expression = ea.compileCoreExpressionMapCore(expression, ea.implCounter,
        ea.existenceCounter, ea.andCounter, ea.orCounter, ea.variableCounter);
    const std::string name = gl::extractExpressionUniversalSpan(gl::StrSpan(expression)).toStdString();
    const auto originalElements = ea.compiledExpressions.at(name).elements;
    ea.compileContrapositionFamilies();
    const auto& owner = ea.compiledExpressions.at(name);
    ASSERT_TRUE(owner.elements == originalElements);
    ASSERT_EQ(owner.contrapositives.size(), std::size_t(5));
    std::size_t singles = 0, pairs = 0, triples = 0;
    for (const auto& row : owner.contrapositives) {
        singles += row.sources.size() == 1;
        pairs += row.sources.size() == 2;
        triples += row.sources.size() == 3;
    }
    ASSERT_EQ(singles, std::size_t(1));
    ASSERT_EQ(pairs, std::size_t(3));
    ASSERT_EQ(triples, std::size_t(1));
    const auto count = ea.compiledExpressions.size();
    ea.compileContrapositionFamilies();
    ASSERT_EQ(ea.compiledExpressions.size(), count);
}

TEST(contraposition, excludes_negative_typing_body_rule_only) {
    const gl::LogicalEntity excluded("implication",
        {"(in2[u_1,1,u_2])", "!(in[1,u_3])"}, "(implication90[u_1,u_2,u_3])", 3);
    ASSERT_TRUE(gl::ExpressionAnalyzer::isContrapositionTypingExclusion(excluded));
    auto candidate = excluded;
    candidate.elements[1] = "(in[1,u_3])";
    ASSERT_FALSE(gl::ExpressionAnalyzer::isContrapositionTypingExclusion(candidate));
    candidate.elements[1] = "!(in2[1,u_3,u_2])";
    ASSERT_FALSE(gl::ExpressionAnalyzer::isContrapositionTypingExclusion(candidate));
    candidate.elements[1] = "!(in[u_3,u_2])";
    ASSERT_FALSE(gl::ExpressionAnalyzer::isContrapositionTypingExclusion(candidate));
    candidate.elements = {"(in2[1,2,u_1])", "!(in[1,u_2])"};
    ASSERT_FALSE(gl::ExpressionAnalyzer::isContrapositionTypingExclusion(candidate));
    candidate.elements = {"(in2[u_1,1,u_2])", "(in[1,u_3])", "!(in[1,u_3])"};
    ASSERT_FALSE(gl::ExpressionAnalyzer::isContrapositionTypingExclusion(candidate));
}

TEST(contraposition, fxy_domain_group_and_hypothetical_publication_boundary) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    ea.compileContrapositionFamilies();
    const auto& owner = ea.compiledExpressions.at("fXY");
    bool groupedDomain = false;
    for (const auto& row : owner.contrapositives)
        groupedDomain = groupedDomain || row.sources.size() == 2;
    ASSERT_TRUE(groupedDomain);
    const auto slot = gl::genScratchArenas().slotCount() - 1;
    auto& arena = gl::genScratchArenas().forSlot(slot);
    gl::ExpressionAnalyzer::CollectedArena collected(&arena);
    const auto before = ea.body.exprOriginMap.count();
    ea.collectContrapositionProducts(gl::StrSpan("(fXY[u_f,u_I,u_N])", 18),
        gl::StrSpan("(fXY[f,I,N])", 12), ea.body, collected, gl::StrSpan("main", 4), false);
    ASSERT_EQ(ea.body.exprOriginMap.count(), before);
    int rules = 0;
    int groupedCarriers = 0;
    collected.forEachKey([&](gl::StrSpan key) {
        if (gl::startsWithSpan(key, "(implication", 12)) ++groupedCarriers;
        collected.forImpls(key, [&](gl::StrSpan, gl::StrSpan) { ++rules; });
    });
    ASSERT_TRUE(groupedCarriers > 0);
    ASSERT_TRUE(rules >= 2);
}

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

TEST(existence_implications, count_formula_named_elements_only) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    const gl::LogicalEntity named("existence",
        {"(in[w1,u_1])", "(in2[w1,u_2,u_3])"}, "(existence90[u_1,u_2,u_3])", 3);
    ASSERT_EQ(ea.existenceImplicationCount(named), 2);
    // A numbered element — a generated compound, registered as a real run
    // registers it — leaves the existence without implications.
    ea.compiledExpressions["and2"] = gl::LogicalEntity("and",
        {"(in[u_1,u_2])", "(in[u_1,u_3])"}, "(and2[u_1,u_2,u_3])", 3);
    ea.compiledExpressions["existence17"] = gl::LogicalEntity("existence",
        {"(in[w1,u_1])", "(in2[w1,u_2,u_3])"}, "(existence17[u_1,u_2,u_3])", 3);
    const gl::LogicalEntity numbered("existence",
        {"(in[w1,u_1])", "(and2[w1,u_2,u_3])"}, "(existence91[u_1,u_2,u_3])", 3);
    ASSERT_EQ(ea.existenceImplicationCount(numbered), 0);
    const gl::LogicalEntity nested("existence",
        {"(in[w1,u_1])", "(existence17[w1,u_2,u_3])"}, "(existence92[u_1,u_2,u_3])", 3);
    ASSERT_EQ(ea.existenceImplicationCount(nested), 0);
    const gl::LogicalEntity one("existence", {"(in[w1,u_1])"}, "(existence93[u_1])", 1);
    ASSERT_EQ(ea.existenceImplicationCount(one), 0);
    const gl::LogicalEntity three("existence",
        {"(in[w1,u_1])", "(in[w1,u_2])", "(in[w1,u_3])"}, "(existence94[u_1,u_2,u_3])", 3);
    ASSERT_EQ(ea.existenceImplicationCount(three), 0);
}

TEST(existence_implications, negated_typed_existence_expands_body_direction_only) {
    // The door instantiates only the body direction of a typed existence:
    // of the two compiled compacts, the typing direction `in[x,a] ->
    // !in2[b,x,c]` is skipped and the body direction `in2[b,x,c] -> !in[x,a]`
    // becomes the one product statement and the one rule.
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    const std::string name = mintExistence(ea, "!(>[x](in[x,a])!(in2[b,x,c]))");
    const gl::LogicalEntity* le = ea.compiledEntity(gl::StrSpan(name));
    ASSERT_TRUE(le != nullptr);
    ASSERT_EQ(static_cast<int>(le->implications.size()), 2);
    ASSERT_TRUE(ea.isTypingDirectionCompact(gl::StrSpan(le->implications[0])));
    ASSERT_FALSE(ea.isTypingDirectionCompact(gl::StrSpan(le->implications[1])));

    const std::vector<std::string> sigArgs = ce::getArgs(le->signature);
    std::map<std::string, std::string> subst;
    std::string inst = "!(" + name + "[";
    for (std::size_t i = 0; i < sigArgs.size(); ++i) {
        subst[sigArgs[i]] = std::to_string(9 - i);
        inst += (i ? "," : "") + std::to_string(9 - i);
    }
    inst += "])";
    const std::string typingInst = ce::replaceKeysInString(le->implications[0], subst);
    const std::string bodyInst = ce::replaceKeysInString(le->implications[1], subst);

    ea.preMintReducedOrs();
    const int lv[1] = { 0 };
    const gl::TransientOrigin origin{ true, gl::OriginTag::taskFormulation, nullptr, 0 };
    const int32_t rulesBefore = ea.body.overallHashMemory.originals.count();
    ea.addExprToMemoryBlock(gl::StrSpan(inst), ea.body, 0, 0, lv, 1, origin,
        -1, -1, gl::StrSpan("main", 4), false);
    ASSERT_EQ(ea.body.overallHashMemory.originals.count(), rulesBefore + 1);
    ASSERT_TRUE(gl::lookupStatementFlags(ea.body.intKnownStatements, ea.body.nameMap,
        gl::StrSpan(bodyInst), gl::StrSpan("main", 4)) != nullptr);
    ASSERT_TRUE(gl::lookupStatementFlags(ea.body.intKnownStatements, ea.body.nameMap,
        gl::StrSpan(typingInst), gl::StrSpan("main", 4)) == nullptr);
}

TEST(existence_implications, nested_negated_existence_is_a_stored_statement_only) {
    // The D1 shape: an outer existence whose right element is a numbered
    // (nested) existence. The inner, over named predicates, keeps its two
    // compacts; the outer gets none, and its negation is registered as a
    // plain statement — no compact product, no rule.
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    const std::string inner = mintExistence(ea, "!(>[y](in[y,a])!(in2[b,y,c]))");
    const gl::LogicalEntity* innerLe = ea.compiledEntity(gl::StrSpan(inner));
    ASSERT_TRUE(innerLe != nullptr);
    ASSERT_EQ(static_cast<int>(innerLe->implications.size()), 2);
    const std::vector<std::string> innerArgs = ce::getArgs(innerLe->signature);
    ASSERT_EQ(static_cast<int>(innerArgs.size()), 3);
    // The outer binds a fresh variable z and names the inner instance over
    // it (the inner's tokens a, b, c in signature order, b := z), the D1
    // shape: the gate reads the element's core name only.
    const std::string innerInst = "(" + inner + "[a,z,c])";
    const std::string outer = mintExistence(ea, "!(>[z](in[z,a])!" + innerInst + ")");
    const gl::LogicalEntity* outerLe = ea.compiledEntity(gl::StrSpan(outer));
    ASSERT_TRUE(outerLe != nullptr);
    ASSERT_EQ(static_cast<int>(outerLe->elements.size()), 2);
    ASSERT_EQ(ea.existenceImplicationCount(*outerLe), 0);
    ASSERT_TRUE(outerLe->implications.empty());
    ea.preMintReducedOrs();                       // the seam leaves it empty too
    ASSERT_TRUE(ea.compiledEntity(gl::StrSpan(outer))->implications.empty());

    const std::vector<std::string> outerArgs = ce::getArgs(outerLe->signature);
    std::string inst = "!(" + outer + "[";
    for (std::size_t i = 0; i < outerArgs.size(); ++i) inst += (i ? "," : "") + std::to_string(9 - i);
    inst += "])";
    const int lv[1] = { 0 };
    const gl::TransientOrigin origin{ true, gl::OriginTag::taskFormulation, nullptr, 0 };
    const int32_t rulesBefore = ea.body.overallHashMemory.originals.count();
    const int32_t rowsBefore = ea.body.intEncodedStatements.size();
    ea.addExprToMemoryBlock(gl::StrSpan(inst), ea.body, 0, 0, lv, 1, origin,
        -1, -1, gl::StrSpan("main", 4), false);
    ASSERT_TRUE(gl::lookupStatementFlags(ea.body.intKnownStatements, ea.body.nameMap,
        gl::StrSpan(inst), gl::StrSpan("main", 4)) != nullptr);
    ASSERT_EQ(ea.body.intEncodedStatements.size(), rowsBefore + 1);
    ASSERT_EQ(ea.body.overallHashMemory.originals.count(), rulesBefore);
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

TEST(existence_implications, constructor_loads_only_the_binary_it_is_given) {
    namespace fs = std::filesystem;
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    // A nested existence (outer over an instance of a minted inner one):
    // a compound no config compile produces, so its presence in a registry
    // proves the binary was read.
    const std::string inner = mintExistence(ea, "!(>[y](in[y,a])!(in2[b,y,c]))");
    const std::string outer =
        mintExistence(ea, "!(>[z](in[z,a])!(" + inner + "[a,z,c]))");
    const std::vector<std::string> expectedList =
        ea.compiledEntity(gl::StrSpan(inner))->implications;
    const fs::path dir = fs::temp_directory_path() / "gl_constructor_binary_test";
    fs::create_directories(dir);
    ea.exportCompiledExpressionsJSON(dir);
    const fs::path binPath = dir / "GL_binary_Peano.json";
    ASSERT_TRUE(fs::exists(binPath));

    // Given the file: both existences are part of the registry before the
    // core map is compiled, the inner with its implication list.
    gl::ExpressionAnalyzer seeded(std::string("Peano"), std::nullopt, binPath);
    ASSERT_EQ(seeded.compiledExpressions.count(inner), std::size_t(1));
    ASSERT_EQ(seeded.compiledExpressions.count(outer), std::size_t(1));
    ASSERT_TRUE(seeded.compiledEntity(gl::StrSpan(inner))->implications == expectedList);

    // Without a file: nothing is read from disk, so the nested existence
    // does not exist here — the registry is the config's alone.
    gl::ExpressionAnalyzer fresh(std::string("Peano"));
    ASSERT_EQ(fresh.compiledExpressions.count(outer), std::size_t(0));
    fs::remove(binPath);
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

TEST(contraposition, grouped_source_and_reverse_are_independent_compact_carriers) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    const auto& family = ea.compiledExpressions.at("fXY");
    const gl::ContrapositionTemplate* grouped = nullptr;
    for (const auto& record : family.contrapositives)
        if (record.sources.size() == 2) grouped = &record;
    ASSERT_TRUE(grouped != nullptr);
    const std::map<std::string, std::string> substitutions = {
        {"u_1", "9"}, {"u_2", "10"}, {"u_3", "11"}};
    const std::string source = ce::replaceKeysInString(grouped->sourceRule, substitutions);
    const std::string reverse = ce::replaceKeysInString(grouped->reverseRule, substitutions);
    ASSERT_TRUE(source != reverse);
    const auto sourceArguments = ce::getArgs(source);
    const auto reverseArguments = ce::getArgs(reverse);
    ASSERT_EQ(sourceArguments.size(), 3);
    ASSERT_TRUE(std::set<std::string>(sourceArguments.begin(), sourceArguments.end())
        == std::set<std::string>(reverseArguments.begin(), reverseArguments.end()));
    const int levels[1] = {0};
    const gl::TransientOrigin origin{true, gl::OriginTag::taskFormulation, nullptr, 0};
    for (const int status : {0, 3}) {
        gl::Memory memory;
        ea.addExprToMemoryBlock(gl::StrSpan(source), memory, 0, status, levels, 1,
            origin, -1, -1, gl::StrSpan("main", 4), false);
        for (const std::string& compact : {source, reverse}) {
            ASSERT_TRUE(gl::lookupStatementFlags(memory.intKnownStatements, memory.nameMap,
                gl::StrSpan(compact), gl::StrSpan("main", 4)) != nullptr);
            const auto key = gl::packStatementKey(memory.nameMap.lookup(gl::StrSpan(compact)),
                memory.nameMap.lookup(gl::StrSpan("main", 4)));
            const int32_t id = memory.compactExpansions.lookup(key);
            ASSERT_TRUE(id != 0);
            ASSERT_EQ(memory.compactExpansions.runLen(id), 1);
            const auto installed = memory.compactExpansions.recordAt(id, 0);
            const std::string expected = expandCompact(ea, ea.prefixArgumentsWithU(compact));
            ASSERT_EQ(ce::getArgs(expected).size(), 2);
            ASSERT_EQ(memory.lbStateInterner.decodeView(installed.implTextId).toStdString(), expected);
            ASSERT_TRUE(installed.kind == (status == 3
                ? gl::RuleInstallKind::External : gl::RuleInstallKind::Local));
        }
    }
}

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

    // The body-direction compact is a registered statement at main
    // (u_-stripped); the typing-direction compact is skipped at the door.
    std::map<std::string, std::string> subst;
    for (std::size_t i = 0; i < 3; ++i) subst[sigArgs[i]] = deposit[i];
    for (const std::string& tmpl : le->implications) {
        const std::string compact = ce::replaceKeysInString(tmpl, subst);
        ASSERT_TRUE(compact.rfind("(implication", 0) == 0);
        const gl::StatementFlags* row = gl::lookupStatementFlags(
            ea.body.intKnownStatements, ea.body.nameMap,
            gl::StrSpan(compact), gl::StrSpan("main", 4));
        if (ea.isTypingDirectionCompact(gl::StrSpan(tmpl))) ASSERT_TRUE(row == nullptr);
        else ASSERT_TRUE(row != nullptr);
    }

    // Hash memory holds exactly the body-direction rule right -> !left over
    // the deposit's arguments, the bound variable in the witness slot; the
    // typing direction left -> !right is not installed.
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
    ASSERT_FALSE(r1);
    ASSERT_TRUE(r2);
    ASSERT_EQ(rules, 1);
}

TEST(contraposition, binary_round_trip_keeps_source_correspondence_and_definitions) {
    gl::ExpressionAnalyzer original(std::string("Peano"));
    const auto expected = original.compiledExpressions.at("fXY");
    ASSERT_TRUE(expected.contrapositionPrepared);
    ASSERT_FALSE(expected.contrapositives.empty());
    const auto directory = std::filesystem::temp_directory_path() / "gl_contraposition_binary_test";
    std::filesystem::create_directories(directory);
    original.exportCompiledExpressionsJSON(directory);
    gl::ExpressionAnalyzer restored(std::string("Peano"));
    restored.loadGlBinary(directory / "GL_binary_Peano.json");
    const auto& actual = restored.compiledExpressions.at("fXY");
    ASSERT_TRUE(actual.elements == expected.elements);
    ASSERT_EQ(actual.contrapositives.size(), expected.contrapositives.size());
    for (std::size_t i = 0; i < expected.contrapositives.size(); ++i) {
        ASSERT_TRUE(actual.contrapositives[i].sources == expected.contrapositives[i].sources);
        ASSERT_EQ(actual.contrapositives[i].sourceRule, expected.contrapositives[i].sourceRule);
        ASSERT_EQ(actual.contrapositives[i].reverseRule, expected.contrapositives[i].reverseRule);
        const auto* reverse = restored.compiledEntity(gl::StrSpan(coreOf(actual.contrapositives[i].reverseRule)));
        ASSERT_TRUE(reverse != nullptr && reverse->category == "implication");
    }
    const auto count = restored.compiledExpressions.size();
    restored.compileContrapositionFamilies();
    ASSERT_EQ(restored.compiledExpressions.size(), count);
}

TEST(conjunction_refutation, fixed_parameters_nested_members_and_signed_literals) {
    gl::ExpressionAnalyzer analyzer(std::string("Peano"));
    const auto compact = analyzer.compileCoreExpressionMapCore(
        "(&(in[x,N])(&(in2[x,y,f])!(=[y,a])))", analyzer.implCounter,
        analyzer.existenceCounter, analyzer.andCounter, analyzer.orCounter, analyzer.variableCounter);
    const auto name = coreOf(compact);
    const auto elements = analyzer.compiledExpressions.at(name).elements;
    analyzer.compileConjunctionRefutations();
    const auto& owner = analyzer.compiledExpressions.at(name);
    ASSERT_TRUE(owner.elements == elements);
    ASSERT_TRUE(owner.conjunctionRefutationsPrepared);
    ASSERT_TRUE(owner.conjunctionRefutations.size() >= std::size_t(3));
    bool positivePremise = false;
    for (const auto& row : owner.conjunctionRefutations) {
        const auto& rule = analyzer.compiledExpressions.at(coreOf(row.second));
        ASSERT_EQ(rule.category, std::string("implication"));
        ASSERT_EQ(rule.elements.size(), std::size_t(2));
        const auto formal = ce::getArgs(rule.signature);
        const auto actual = ce::getArgs(row.second);
        std::map<std::string, std::string> substitution;
        for (std::size_t i = 0; i < formal.size(); ++i) substitution[formal[i]] = actual[i];
        const auto premise = ce::replaceKeysInString(rule.elements[0], substitution);
        const auto head = ce::replaceKeysInString(rule.elements[1], substitution);
        ASSERT_EQ(premise, analyzer.negate(row.first));
        ASSERT_EQ(head, analyzer.negate(owner.signature));
        for (const auto& argument : ce::getArgs(premise)) ASSERT_TRUE(argument.rfind("u_", 0) == 0);
        for (const auto& argument : ce::getArgs(head)) ASSERT_TRUE(argument.rfind("u_", 0) == 0);
        if (row.first.front() == '!') {
            positivePremise = true;
            ASSERT_TRUE(premise.front() != '!');
        }
    }
    ASSERT_TRUE(positivePremise);
    const auto count = analyzer.compiledExpressions.size();
    analyzer.compileConjunctionRefutations();
    ASSERT_EQ(analyzer.compiledExpressions.size(), count);
    ASSERT_TRUE(analyzer.compiledExpressions.at("in").conjunctionRefutations.empty());
}

TEST(conjunction_refutation, binary_round_trip_preserves_direct_and_guarded_templates) {
    gl::ExpressionAnalyzer original(std::string("Peano"));
    const auto directory = std::filesystem::temp_directory_path() / "gl_conjunction_refutation_binary_test";
    std::filesystem::create_directories(directory);
    original.exportCompiledExpressionsJSON(directory);
    gl::ExpressionAnalyzer restored(std::string("Peano"));
    restored.loadGlBinary(directory / "GL_binary_Peano.json");
    bool reverseFound = false;
    for (const auto& [name, expected] : original.compiledExpressions) {
        const auto& actual = restored.compiledExpressions.at(name);
        ASSERT_TRUE(actual.conjunctionRefutations == expected.conjunctionRefutations);
        ASSERT_EQ(actual.conjunctionRefutationsPrepared, expected.conjunctionRefutationsPrepared);
        ASSERT_TRUE(actual.supportingImplications == expected.supportingImplications);
        ASSERT_TRUE(actual.composedSupportingImplications == expected.composedSupportingImplications);
        ASSERT_EQ(actual.supportingImplicationsPrepared, expected.supportingImplicationsPrepared);
        ASSERT_EQ(actual.supportingImplicationAuxiliary, expected.supportingImplicationAuxiliary);
        ASSERT_TRUE(actual.elements == expected.elements);
        reverseFound = reverseFound || !actual.supportingImplications.empty();
    }
    ASSERT_TRUE(reverseFound);
    const auto count = restored.compiledExpressions.size();
    restored.compileContrapositionFamilies();
    ASSERT_EQ(restored.compiledExpressions.size(), count);
}

TEST(conjunction_refutation, named_single_member_alias_retains_arity_and_exact_refutation) {
    gl::ExpressionAnalyzer analyzer(std::string("Peano"));
    ce::CoreExpressionConfig config;
    config.signature = "(typedAlias[x,N])";
    config.arity = 2;
    config.definition = std::string("(in[x,N])");
    analyzer.coreExpressionMap["typedAlias"] = config;
    analyzer.compileCoreExpressionMapCore(config.signature, analyzer.implCounter,
        analyzer.existenceCounter, analyzer.andCounter, analyzer.orCounter, analyzer.variableCounter);
    const auto& alias = analyzer.compiledExpressions.at("typedAlias");
    ASSERT_EQ(alias.arity, 2);
    ASSERT_EQ(alias.signature, std::string("(typedAlias[u_1,u_2])"));
    analyzer.compileConjunctionRefutations();
    ASSERT_EQ(alias.conjunctionRefutations.size(), static_cast<std::size_t>(1));
    const auto& rule = analyzer.compiledExpressions.at(coreOf(alias.conjunctionRefutations.front().second));
    ASSERT_EQ(rule.arity, 2);
    ASSERT_TRUE(rule.elements == std::vector<std::string>({"!(in[u_1,u_2])", "!(typedAlias[u_1,u_2])"}));
    analyzer.checkCompiledCoreExpressionMap();
}

namespace {
struct RefutationFixture { std::string source, conjunction, reverse; };
RefutationFixture refutationFixture(gl::ExpressionAnalyzer& analyzer) {
    const auto conjunction = analyzer.compileCoreExpressionMapCore(
        "(&(=[i,n])(=[p,a]))", analyzer.implCounter, analyzer.existenceCounter,
        analyzer.andCounter, analyzer.orCounter, analyzer.variableCounter);
    const auto mapped = ce::replaceKeysInString(conjunction,
        {{"i", "u_1"}, {"p", "u_2"}, {"n", "u_4"}, {"a", "u_5"}});
    const auto name = analyzer.findOrMintOrOperator({"(in2[u_1,u_2,u_3])", mapped}, 5);
    analyzer.preMintReducedOrs();
    analyzer.compileContrapositionFamilies();
    const auto& owner = analyzer.compiledExpressions.at("fXY");
    const auto formal = ce::getArgs(owner.signature);
    const std::vector<std::string> actual{"f", "I", "N"};
    std::map<std::string, std::string> mapping;
    for (std::size_t i = 0; i < actual.size(); ++i) mapping[formal[i]] = actual[i];
    std::string reverse;
    for (const auto& record : owner.contrapositives) {
        const auto& rule = analyzer.compiledExpressions.at(coreOf(record.reverseRule));
        if (record.sources.size() == 2 && !rule.supportingImplications.empty())
            reverse = ce::replaceKeysInString(record.reverseRule, mapping);
    }
    ASSERT_FALSE(reverse.empty());
    return {"(" + name + "[i,p,f,n,a])", conjunction, reverse};
}

void registerRefutationSource(gl::ExpressionAnalyzer& analyzer, const std::string& text,
    const std::string& scope = "main") {
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::PagedVector<gl::IntEncodedExpr> statements(&gl::genScratchArenas().forSlot(0), &dirty);
    const int levels[] = {0};
    const auto* entity = analyzer.compiledEntity(gl::extractExpressionUniversalSpan(gl::StrSpan(text)));
    analyzer.addStatement(gl::StrSpan(text), analyzer.body, true, levels, 1,
        gl::TransientOrigin{}, gl::StrSpan(scope), statements, entity != nullptr && entity->category == "or");
}
}

TEST(conjunction_refutation, ordinary_signed_firing_accepts_both_fact_arrival_orders) {
    for (bool factFirst : {false, true}) {
        gl::ExpressionAnalyzer analyzer(std::string("Peano"));
        analyzer.parameters.trackHistory = true;
        const auto fixture = refutationFixture(analyzer);
        auto& mb = analyzer.body;
        const int levels[] = {0};
        const gl::TransientOrigin origin{true, gl::OriginTag::taskFormulation, nullptr, 0};
        const std::string fact = "!(=[i,n])";
        const auto deposit = [&](const std::string& text) {
            analyzer.addExprToMemoryBlock(gl::StrSpan(text), mb, 0, 0, levels, 1,
                origin, -1, -1, gl::StrSpan("main", 4), false);
        };
        if (factFirst) deposit(fact);
        registerRefutationSource(analyzer, fixture.source);
        std::vector<std::string> compacts;
        for (int32_t row = 1; row <= mb.sameIterationInternalMail.statements_.count(); ++row) {
            const auto key = mb.sameIterationInternalMail.statements_.keyAt(row);
            int32_t expression;
            std::memcpy(&expression, key.ptr, sizeof(expression));
            compacts.push_back(mb.nameMap.decode(expression));
        }
        ASSERT_EQ(compacts.size(), static_cast<std::size_t>(2));
        analyzer.standardProcessing(mb, nullptr, mb.sameIterationInternalMail,
            mb.nextIterationInternalMail, -1);
        for (const auto& compact : compacts)
            ASSERT_TRUE(mb.intKnownStatements.find({mb.nameMap.lookup(compact), gl::NameMap::MAIN_ID}) != nullptr);
        if (!factFirst) deposit(fact);
        ASSERT_EQ(mb.overallHashMemory.originals.count(), 2);
        gl::SealedPageSet pages;
        pages.bind(&gl::staticMemory());
        std::atomic<int64_t> doomLine{gl::kNoDoomLine};
        gl::BurstSink sink{&analyzer, &mb, 0u, &pages, &doomLine, 0,
            gl::SealedRecordCursor<gl::FiringRecord>(pages)};
        analyzer.generateEncodedRequestsStatic(mb, mb.overallHashMemory, nullptr, 0,
            gl::SplitStumpRef{}, 0u, sink);
        pages.seal();
        bool fired = false, correctScope = false, correctOrigin = false;
        pages.forEachRecord<gl::FiringRecord>([&](const gl::FiringRecord& record) {
            if (!record.isMarker && !record.isOrdis2Demand
                && record.rplExpr2.toStdString() == "!" + fixture.conjunction) {
                correctScope = record.validityName.toStdString() == "main";
                correctOrigin = record.originTag.toStdString() == "implication";
                fired = true;
            }
        });
        pages.freePages();
        ASSERT_TRUE(fired);
        ASSERT_TRUE(correctScope && correctOrigin);
    }
}

TEST(supporting_implications, immediate_parts_complete_binding_and_no_recursive_families) {
    gl::ExpressionAnalyzer analyzer(std::string("Peano"));
    const auto conjunction = analyzer.compileCoreExpressionMapCore(
        "(&(in[x,u_N])(&(in[y,u_N])(in[z,u_N])))", analyzer.implCounter,
        analyzer.existenceCounter, analyzer.andCounter, analyzer.orCounter, analyzer.variableCounter);
    const auto source = analyzer.compileImplicationToCompact(
        "(>[x,y,z]!" + conjunction + "!(in3[x,y,z,u_f]))");
    analyzer.compileContrapositionFamilies();
    const auto& owner = analyzer.compiledExpressions.at(coreOf(source));
    ASSERT_EQ(owner.supportingImplications.size(), std::size_t(3));
    ASSERT_EQ(owner.composedSupportingImplications.size(), std::size_t(3));
    for (const auto& compact : owner.composedSupportingImplications) {
        const auto& direct = analyzer.compiledExpressions.at(coreOf(compact));
        ASSERT_EQ(direct.elements.size(), std::size_t(4));
        ASSERT_EQ(coreOf(direct.elements.back()), std::string("in3"));
        ASSERT_TRUE(direct.supportingImplicationAuxiliary);
        ASSERT_TRUE(direct.supportingImplications.empty());
        ASSERT_TRUE(direct.composedSupportingImplications.empty());
        ASSERT_TRUE(direct.contrapositives.empty());
    }
    for (const auto& compact : owner.supportingImplications) {
        const auto& helper = analyzer.compiledExpressions.at(coreOf(compact));
        ASSERT_EQ(helper.elements.size(), std::size_t(4));
        ASSERT_TRUE(helper.supportingImplicationAuxiliary);
        ASSERT_TRUE(helper.supportingImplications.empty());
        ASSERT_TRUE(helper.contrapositives.empty());
        int negatives = 0;
        std::set<std::string> covered;
        for (std::size_t i = 0; i + 1 < helper.elements.size(); ++i) {
            negatives += helper.elements[i].front() == '!';
            for (const auto& arg : ce::getArgs(helper.elements[i])) covered.insert(arg);
        }
        ASSERT_EQ(negatives, 1);
        for (const auto& arg : ce::getArgs(helper.elements.back()))
            ASSERT_TRUE(arg.rfind("u_", 0) == 0 || covered.count(arg) != 0);
    }
    const auto count = analyzer.compiledExpressions.size();
    analyzer.compileContrapositionFamilies();
    ASSERT_EQ(analyzer.compiledExpressions.size(), count);
}

TEST(supporting_implications, composed_fxy_rules_require_guards_and_retain_independent_consumer) {
    for (bool factFirst : {false, true}) for (bool includePositive : {false, true})
        for (bool independentConjunction : {false, true}) for (bool negateDomain : {false, true}) {
        gl::ExpressionAnalyzer analyzer(std::string("Peano"));
        analyzer.parameters.trackHistory = true;
        const auto fixture = refutationFixture(analyzer);
        const auto& reverse = analyzer.compiledExpressions.at(coreOf(fixture.reverse));
        std::map<std::string, std::string> substitution;
        const auto formal = ce::getArgs(reverse.signature);
        const auto actual = ce::getArgs(fixture.reverse);
        for (std::size_t k = 0; k < formal.size(); ++k) substitution[formal[k]] = actual[k];
        const auto head = ce::getArgs(reverse.elements.back());
        substitution[head[0]] = "i";
        substitution[head[1]] = "p";
        const auto intermediate = ce::replaceKeysInString(reverse.elements.front(), substitution);
        const auto expected = ce::replaceKeysInString(reverse.elements.back(), substitution);
        ASSERT_EQ(reverse.composedSupportingImplications.size(), std::size_t(2));
        const auto conjunctionCore = coreOf(reverse.elements.front());
        auto& mb = analyzer.body;
        const int levels[] = {0};
        const gl::TransientOrigin origin{true, gl::OriginTag::taskFormulation, nullptr, 0};
        const auto deposit = [&](const std::string& text) {
            analyzer.addExprToMemoryBlock(gl::StrSpan(text), mb, 0, 0, levels, 1,
                origin, -1, -1, gl::StrSpan("main", 4), false);
        };
        const auto facts = [&] {
            deposit(negateDomain ? "!(in[i,I])" : "!(in[p,N])");
            if (includePositive) deposit(negateDomain ? "(in[p,N])" : "(in[i,I])");
            if (independentConjunction) deposit(intermediate);
        };
        if (factFirst) facts();
        deposit(fixture.reverse);
        if (!factFirst) facts();
        ASSERT_TRUE(mb.orImplications.sources.count() == 0);
        std::map<std::string, std::string> fixed;
        for (std::size_t k = 0; k < formal.size(); ++k) fixed[formal[k]] = actual[k];
        for (const auto& helper : reverse.supportingImplications) {
            const auto carrier = ce::replaceKeysInString(helper, fixed);
            ASSERT_TRUE(mb.intKnownStatements.find({mb.nameMap.lookup(carrier), gl::NameMap::MAIN_ID}) == nullptr);
        }
        ASSERT_TRUE(mb.intKnownStatements.find({mb.nameMap.lookup(fixture.reverse), gl::NameMap::MAIN_ID}) != nullptr);
        gl::SealedPageSet pages; pages.bind(&gl::staticMemory());
        std::atomic<int64_t> doomLine{gl::kNoDoomLine};
        gl::BurstSink sink{&analyzer, &mb, 0u, &pages, &doomLine, 0,
            gl::SealedRecordCursor<gl::FiringRecord>(pages)};
        analyzer.generateEncodedRequestsStatic(mb, mb.overallHashMemory, nullptr, 0,
            gl::SplitStumpRef{}, 0u, sink);
        pages.seal();
        bool fired = false, producedIntermediate = false;
        pages.forEachRecord<gl::FiringRecord>([&](const gl::FiringRecord& record) {
            if (!record.isMarker && !record.isOrdis2Demand
                && record.rplExpr2.toStdString() == expected) fired = true;
            if (!record.isMarker && !record.isOrdis2Demand
                && record.rplExpr2.toStdString() == intermediate) producedIntermediate = true;
        });
        pages.freePages();
        ASSERT_EQ(fired, includePositive || independentConjunction);
        ASSERT_FALSE(producedIntermediate);
        bool helperProof = false, compositionProof = false;
        const auto history = gl::decodeOriginMapSorted(mb.exprOriginMap, mb.originInterner);
        for (const auto& [expression, origins] : history) for (const auto& row : origins)
            if (row.first == "expansion" && row.second.size() == 1
                && row.second.front().original.rfind("(>[", 0) == 0
                && expression.first.find("!(" + conjunctionCore + "[") != std::string::npos) helperProof = true;
        for (const auto& [expression, origins] : history) for (const auto& row : origins)
            if (row.first == "implication composition" && row.second.size() == 2
                && expression.first.find("!(in2[") != std::string::npos) compositionProof = true;
        ASSERT_TRUE(helperProof && compositionProof);
    }
}

TEST(conjunction_refutation, direct_admission_uses_requested_output_and_obeys_disintegration_gate) {
    for (bool blocked : {false, true}) {
        gl::ExpressionAnalyzer analyzer(std::string("Peano"));
        const auto fixture = refutationFixture(analyzer);
        analyzer.parameters.ban_disintegration = blocked;
        auto& mb = analyzer.body;
        gl::DirtyState dirty = gl::DirtyState::Clean;
        gl::PagedVector<gl::IntEncodedExpr> rows(&gl::genScratchArenas().forSlot(0), &dirty);
        const int levels[] = {0};
        analyzer.addStatement(gl::StrSpan(fixture.source), mb, true, levels, 1,
            gl::TransientOrigin{}, gl::StrSpan("main", 4), rows, true, &mb.nextIterationInternalMail);
        ASSERT_TRUE(mb.sameIterationInternalMail.statementsEmpty());
        ASSERT_EQ(mb.nextIterationInternalMail.statements_.count(), blocked ? 0 : 2);
    }
}

TEST(supporting_implications, nonunary_owner_preserves_its_guarded_helpers) {
    gl::ExpressionAnalyzer analyzer(std::string("Peano"));
    const auto conjunction = analyzer.compileCoreExpressionMapCore(
        "(&(in[x,u_N])(in[y,u_N]))", analyzer.implCounter,
        analyzer.existenceCounter, analyzer.andCounter, analyzer.orCounter, analyzer.variableCounter);
    const auto source = analyzer.compileImplicationToCompact(
        "(>[x,y](in[x,u_A])(>[]!" + conjunction + "!(in2[x,y,u_f])))");
    analyzer.compileContrapositionFamilies();
    const auto& owner = analyzer.compiledExpressions.at(coreOf(source));
    ASSERT_EQ(owner.elements.size(), std::size_t(3));
    ASSERT_EQ(owner.supportingImplications.size(), std::size_t(2));
    ASSERT_TRUE(owner.composedSupportingImplications.empty());
}
