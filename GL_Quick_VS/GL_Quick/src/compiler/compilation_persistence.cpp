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
#include "compilation_persistence.hpp"
#include "compilation_reuse.hpp"
#include "../compiler.hpp"
#include <numeric>
#include <fstream>

namespace gl {
/// @brief Bind the cooperating compilation components.
/// @details Construction stores references only; all bound objects outlive this component.
/// @param definitions Borrowed definitions component.
/// @param compilationReuse Borrowed compilationReuse component.
/// @param anchorID_ Borrowed anchorID_ component.
/// @return Constructed compilation component.
CompilationPersistence::CompilationPersistence(CompiledDefinitionRegistry& definitions, CompilationReuse& compilationReuse, const std::string& anchorID_)
    : definitions(definitions), compilationReuse(compilationReuse), anchorID_(anchorID_) {}
/// @brief Perform loadGlBinary using the registered definitions.
///
/// @details Preserves expression order, binding and compiler name allocation.
/// @param jsonPath Input to this compilation operation.
/// @return None.
void CompilationPersistence::loadGlBinary(const std::filesystem::path& jsonPath) {
    namespace fs = std::filesystem;
    if (!fs::exists(jsonPath)) {
        return;  // first batch on a clean run; nothing to load.
    }

    nlohmann::json root;
    {
        std::ifstream f(jsonPath.string());
        if (!f.is_open()) {
            return;
        }
        try {
            f >> root;
        } catch (const std::exception& e) {
            std::cerr << "[loadGlBinary] failed to parse " << jsonPath
                      << ": " << e.what() << std::endl;
            return;
        }
    }
    if (!root.is_object()) {
        return;
    }

    int loadedSpontaneous = 0;
    int loadedTotal = 0;
    int maxImpl = -1, maxExist = -1, maxAnd = -1, maxOr = -1;
    std::vector<std::string> loadedNamed;

    for (auto it = root.begin(); it != root.end(); ++it) {
        const std::string& name = it.key();
        const auto& entry = it.value();
        if (!entry.is_object()) continue;

        std::string category  = entry.value("category", std::string("atomic"));
        std::string signature = entry.value("signature", std::string());
        int arity             = entry.value("arity", 0);
        std::string definedSet = entry.value("definedSet", std::string());

        std::vector<std::string> elements;
        if (entry.contains("elements") && entry["elements"].is_array()) {
            for (const auto& e : entry["elements"]) {
                if (e.is_string()) elements.push_back(e.get<std::string>());
            }
        }

        // "implications" — the disintegration set of an or (D-309) or an
        // existence (D-310). The writer always writes it for those two
        // categories (an empty list included) and never for another, so a
        // missing or misplaced key is a stale or corrupt binary.
        const bool carriesDisintegrationSet = category == "or" || category == "existence";
        assert(entry.contains("implications") == carriesDisintegrationSet
            && "loadGlBinary: \"implications\" present exactly for or and existence entries");
        std::vector<std::string> implications;
        if (carriesDisintegrationSet) {
            assert(entry["implications"].is_array()
                && "loadGlBinary: \"implications\" must be a JSON array");
            for (const auto& e : entry["implications"]) {
                assert(e.is_string()
                    && "loadGlBinary: \"implications\" entries must be strings");
                implications.push_back(e.get<std::string>());
            }
        }

        // "integration_implications" — the integration set of an and or an
        // existence, always written for those two categories and never for
        // another.
        const bool carriesIntegrationSet = category == "and" || category == "existence";
        assert(entry.contains("integration_implications") == carriesIntegrationSet
            && "loadGlBinary: \"integration_implications\" present exactly for and and existence entries");
        std::vector<IntegrationImplication> integrationImplications;
        if (carriesIntegrationSet) {
            assert(entry["integration_implications"].is_array()
                && "loadGlBinary: \"integration_implications\" must be a JSON array");
            for (const auto& e : entry["integration_implications"]) {
                IntegrationImplication record;
                record.compact = e.at("compact").get<std::string>();
                record.kind = integrationRuleKindFromName(e.at("kind").get<std::string>());
                record.cites = e.at("cites").get<int32_t>();
                integrationImplications.push_back(std::move(record));
            }
        }

        // Always populate compiledExpressions so any read-side path
        // (e.g. encoded-expression resolution) sees the entry.
        {
            LogicalEntity le(category, elements, signature, arity, definedSet);
            le.implications = implications;
            le.integrationImplications = std::move(integrationImplications);
            if (entry.contains("conjunction_refutations")) {
                const auto& auxiliary = entry.at("conjunction_refutations");
                assert(auxiliary.at("version") == 1);
                le.conjunctionRefutationsPrepared = auxiliary.at("prepared").get<bool>();
                le.conjunctionRefutations = auxiliary.at("rules").get<
                    std::vector<std::pair<std::string, std::string>>>();
            }
            if (entry.contains("supporting_implications")) {
                const auto& auxiliary = entry.at("supporting_implications");
                const int version = auxiliary.at("version").get<int>();
                assert(version == 1 || version == 2);
                le.supportingImplicationsPrepared = version == 2 && auxiliary.at("prepared").get<bool>();
                le.supportingImplicationAuxiliary = auxiliary.at("auxiliary").get<bool>();
                le.supportingImplications = auxiliary.at("rules").get<std::vector<std::string>>();
                if (version == 2)
                    le.composedSupportingImplications = auxiliary.at("composed_rules").get<std::vector<std::string>>();
                assert(le.composedSupportingImplications.empty()
                    || le.composedSupportingImplications.size() == le.supportingImplications.size());
            }
            if (entry.contains("contraposition")) {
                const auto& auxiliary = entry.at("contraposition");
                assert(auxiliary.at("version") == 1);
                le.contrapositionPrepared = auxiliary.at("prepared").get<bool>();
                le.contrapositionAuxiliary = auxiliary.at("auxiliary").get<bool>();
                assert(auxiliary.at("rules").is_array());
                for (const auto& row : auxiliary.at("rules")) {
                    ContrapositionTemplate rule;
                    rule.sources = row.at("sources").get<std::vector<std::string>>();
                    rule.sourceRule = row.at("source").get<std::string>();
                    rule.reverseRule = row.at("reverse").get<std::string>();
                    assert(!rule.sources.empty());
                    le.contrapositives.push_back(std::move(rule));
                }
            }
            definitions.compiledExpressions[name] = le;
        }
        ++loadedTotal;

        // Logical categories participate in reuse; only generated spellings
        // seed numbered counters. Atomic predicates have no compound body.
        const bool isSpontaneous =
            (category == "implication") || (category == "existence")
            || (category == "or") || (category == "and");
        if (!isSpontaneous) continue;

        // Public logical predicates need the same normalized reuse identities
        // as fresh forced-name compilation, independently of counter seeding.
        if (!isNumberedCompilationName(name, category)) {
            loadedNamed.push_back(name);
            continue;
        }

        // Identify the prefix that matches the category and parse the
        // trailing integer. Names that don't fit (e.g. an anchor
        // mis-categorised to "and") are skipped for counter-seeding but
        // still appear in compiledExpressions.
        const std::string& prefix = category;  // implication / existence / or / and
        if (name.size() <= prefix.size() ||
            name.compare(0, prefix.size(), prefix) != 0) {
            continue;
        }
        int n = -1;
        try {
            n = std::stoi(name.substr(prefix.size()));
        } catch (...) {
            continue;
        }
        if (n < 0) continue;

        if      (category == "implication" && n > maxImpl)  maxImpl  = n;
        else if (category == "existence"   && n > maxExist) maxExist = n;
        else if (category == "and"         && n > maxAnd)   maxAnd   = n;
        else if (category == "or"          && n > maxOr)    maxOr    = n;

        // A numbered entry is keyed like a fresh registration: by the
        // canonical form of its stored elements, under its category (so an
        // implication and an existence with the same elements keep distinct
        // names), the canonical slots mapped to its own u_<i> signature slots.
        compilationReuse.registerStoredCompilation(name, false);
        ++loadedSpontaneous;
    }

    if (maxImpl  >= 0) definitions.implCounter      = std::max(definitions.implCounter,      maxImpl  + 1);
    if (maxExist >= 0) definitions.existenceCounter = std::max(definitions.existenceCounter, maxExist + 1);
    if (maxAnd   >= 0) definitions.andCounter       = std::max(definitions.andCounter,       maxAnd   + 1);
    if (maxOr    >= 0) definitions.orCounter        = std::max(definitions.orCounter,        maxOr    + 1);

    loadCompilationState(jsonPath);
    // Install all definitions before deriving public reuse and retirement.
    // Shared files omit atomic entries; splitCompilationKey uses source arities.
    for (const auto& name : loadedNamed) compilationReuse.registerLoadedNamedCompilation(name);
    compilationReuse.finalizeCompilationReplacements();

#if 0
    std::cout << "[loadGlBinary] loaded " << loadedTotal << " entries ("
              << loadedSpontaneous << " spontaneous) from " << jsonPath
              << " — counters seeded: implCounter=" << definitions.implCounter
              << " existenceCounter=" << definitions.existenceCounter
              << " andCounter=" << definitions.andCounter
              << " orCounter=" << definitions.orCounter << std::endl;
#endif
}

/// @brief Serialize compiled definitions and their compilation-state sidecar.
///
/// @details Preserves expression order, binding and compiler name allocation.
/// @param outDir Output directory for the batch registry.
/// @return None.
void CompilationPersistence::exportCompiledExpressionsJSON(const std::filesystem::path& outDir) {
    namespace fs = std::filesystem;
    fs::create_directories(outDir);

    fs::path outPath = outDir / ("GL_binary_" + anchorID_ + ".json");
    nlohmann::json root = nlohmann::json::object();

    for (const auto& kv : definitions.compiledExpressions) {
        const std::string& coreName = kv.first;
        const auto& compExpr = kv.second;

        nlohmann::json entry;
        entry["category"] = compExpr.category;
        entry["signature"] = compExpr.signature;
        entry["arity"] = compExpr.arity;
        entry["definedSet"] = compExpr.definedSet;

        nlohmann::json elems = nlohmann::json::array();
        for (const auto& e : compExpr.elements) {
            elems.push_back(e);
        }
        entry["elements"] = elems;

        // The disintegration set, always for an or or an existence (an empty
        // list included), never for another category.
        if (compExpr.category == "or" || compExpr.category == "existence") {
            nlohmann::json impls = nlohmann::json::array();
            for (const auto& s : compExpr.implications) {
                impls.push_back(s);
            }
            entry["implications"] = impls;
        } else {
            assert(compExpr.implications.empty()
                && "exportCompiledExpressionsJSON: a disintegration set outside or and existence");
        }
        // The integration set, always for an and or an existence, never for
        // another category.
        if (compExpr.category == "and" || compExpr.category == "existence") {
            nlohmann::json integration = nlohmann::json::array();
            for (const IntegrationImplication& record : compExpr.integrationImplications) {
                integration.push_back({{"compact", record.compact},
                    {"kind", integrationRuleKindName(record.kind)}, {"cites", record.cites}});
            }
            entry["integration_implications"] = integration;
        } else {
            assert(compExpr.integrationImplications.empty()
                && "exportCompiledExpressionsJSON: an integration set outside and and existence");
        }

        if (compExpr.contrapositionPrepared || compExpr.contrapositionAuxiliary
            || !compExpr.contrapositives.empty()) {
            nlohmann::json rules = nlohmann::json::array();
            for (const auto& row : compExpr.contrapositives)
                rules.push_back({{"sources", row.sources}, {"source", row.sourceRule},
                    {"reverse", row.reverseRule}});
            entry["contraposition"] = {{"version", 1},
                {"prepared", compExpr.contrapositionPrepared},
                {"auxiliary", compExpr.contrapositionAuxiliary}, {"rules", rules}};
        }
        if (compExpr.conjunctionRefutationsPrepared) {
            entry["conjunction_refutations"] = {{"version", 1},
                {"prepared", compExpr.conjunctionRefutationsPrepared},
                {"rules", compExpr.conjunctionRefutations}};
        }
        if (compExpr.supportingImplicationsPrepared || compExpr.supportingImplicationAuxiliary) {
            entry["supporting_implications"] = {{"version", 2},
                {"prepared", compExpr.supportingImplicationsPrepared},
                {"auxiliary", compExpr.supportingImplicationAuxiliary},
                {"rules", compExpr.supportingImplications},
                {"composed_rules", compExpr.composedSupportingImplications}};
        }
        root[coreName] = entry;
    }

    std::ofstream f(outPath.string().c_str());
    if (f.is_open()) {
        f << root.dump(2);
        f.close();
    }
    exportCompilationState(outDir);
}

/// @brief Restore monotone compiler counters and replacement metadata.
        ///
        /// @details A missing sidecar denotes the original registry format;
        /// registry-derived counters remain valid for that format. Present
        /// sidecars are strictly validated before their state is applied.
        /// @param registryPath Compiled registry whose sibling state is loaded.
        /// @return None.
void CompilationPersistence::loadCompilationState(const std::filesystem::path& registryPath) {
    const std::string filename = registryPath.filename().string();
    assert(filename.rfind("GL_binary_", 0) == 0);
    const auto path = registryPath.parent_path() / ("GL_compilation_state_" + filename.substr(10));
    if (!std::filesystem::exists(path)) return;
    std::ifstream input(path);
    assert(input.is_open());
    nlohmann::json state;
    input >> state;
    assert(state.is_object() && state.size() == 3);
    assert(state.at("schema").is_number_integer() && state.at("schema") == 1);
    assert(state.at("next_counters").is_object() && state.at("next_counters").size() == 4);
    assert(state.at("replacements").is_object());
    int* counters[] = {&definitions.implCounter, &definitions.existenceCounter, &definitions.andCounter, &definitions.orCounter};
    const char* categories[] = {"implication", "existence", "and", "or"};
    for (int index = 0; index < 4; ++index) {
        const auto& number = state.at("next_counters").at(categories[index]);
        assert(number.is_number_integer() && number.get<int>() >= 0);
        *counters[index] = std::max(*counters[index], number.get<int>());
    }
    for (const auto& item : state.at("replacements").items()) {
        const auto& record = item.value();
        assert(record.is_object() && record.size() == 2);
        assert(record.at("target").is_string() && !record.at("target").get<std::string>().empty());
        assert(record.at("arguments").is_array());
        assert(std::any_of(std::begin(categories), std::end(categories),
            [&](const char* category) { return isNumberedCompilationName(item.key(), category); }));
        for (const auto& position : record.at("arguments"))
            assert(position.is_number_integer() && position.get<int>() > 0);
        CompilationReplacement replacement{record.at("target").get<std::string>(),
                                           record.at("arguments").get<std::vector<int>>()};
        compilationReuse.restoreReplacement(item.key(), replacement);
    }
}

/// @brief Save the compiled registry's migration state and next counters.
        ///
        /// @details Emits schema 1 beside the existing unchanged registry JSON.
        /// Counters remain above all issued names, including removed names.
        /// @param outDir Registry output directory.
        /// @return None.
void CompilationPersistence::exportCompilationState(const std::filesystem::path& outDir) {
    nlohmann::json state = {{"schema", 1},
        {"next_counters", {{"implication", definitions.implCounter}, {"existence", definitions.existenceCounter},
                           {"and", definitions.andCounter}, {"or", definitions.orCounter}}},
        {"replacements", nlohmann::json::object()}};
    for (const auto& [name, replacement] : compilationReuse.replacements()) {
        state["replacements"][name] = {{"target", replacement.target}, {"arguments", replacement.arguments}};
    }
    std::ofstream output(outDir / ("GL_compilation_state_" + anchorID_ + ".json"));
    assert(output.is_open());
    output << state.dump(2) << '\n';
    output.flush();
    assert(output.good());
}

} // namespace gl
