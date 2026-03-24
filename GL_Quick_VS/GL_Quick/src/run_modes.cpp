/* Generative Logic : A deterministic reasoning and knowledge generation engine.
 Copyright(C) 2025 Generative Logic UG(haftungsbeschr�nkt)

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
#include "run_modes.hpp"
#include "analyze_expressions.hpp"
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <unordered_set>
#include <algorithm>
#include <iostream>
#include "compressor.hpp"
#include <json.hpp>

namespace run_modes {

    inline const std::filesystem::path PROJECT_ROOT =
        std::filesystem::path(__FILE__).parent_path()
        .parent_path()
        .parent_path()
        .parent_path();

    inline std::filesystem::path RAW_PROOF_DIR =
        PROJECT_ROOT / "files" / "raw_proof_graph";

    // Mutable: may be overridden by config (e.g. incubator mode)
    inline std::filesystem::path THEOREMS_FOLDER = PROJECT_ROOT / "files" / "theorems";
    inline std::filesystem::path THEOREMS_FILE = THEOREMS_FOLDER / "theorems.txt";
    inline std::filesystem::path PROVED_THEOREMS_FILE = THEOREMS_FOLDER / "proved_theorems.txt";
    inline std::filesystem::path COMPRESSED_EXTERNAL_THEOREMS_FILE =
        THEOREMS_FOLDER / "compressed_external_theorems.txt";
    inline std::filesystem::path BACKGROUND_PROVED_THEOREMS_FILE = PROVED_THEOREMS_FILE;

    static inline std::string trim_copy(const std::string& s) {
        const auto b = s.find_first_not_of(" \t\r\n");
        if (b == std::string::npos) return {};
        const auto e = s.find_last_not_of(" \t\r\n");
        return s.substr(b, e - b + 1);
    }

    // Helper to read a file into a unique set of non-empty lines
    std::unordered_set<std::string> loadLinesFromFile(const std::filesystem::path& path) {
        std::unordered_set<std::string> lines;
        if (!std::filesystem::exists(path)) return lines;

        std::ifstream in(path);
        std::string line;
        while (std::getline(in, line)) {
            line = trim_copy(line);
            if (line.empty() || line[0] == '#') continue;
            lines.emplace(std::move(line));
        }
        return lines;
    }


    void fullRun(const std::string& anchor_id) {
        using namespace std;
        namespace fs = std::filesystem;

        if (!anchor_id.empty()) {
            std::cout << "\n[fullRun] Processing Tag/Anchor: " << anchor_id << "\n";
        }

        // --- Read config JSON for path overrides ---
        bool skipCompression = false;
        bool skipProofGraph = false;
        {
            auto configPath = PROJECT_ROOT / "files" / "config" / ("Config" + anchor_id + ".json");
            if (fs::exists(configPath)) {
                try {
                    std::ifstream f(configPath);
                    nlohmann::json j;
                    f >> j;
                    if (j.contains("theorems_folder")) {
                        std::string folder = j["theorems_folder"];
                        THEOREMS_FOLDER = PROJECT_ROOT / folder;
                        THEOREMS_FILE = THEOREMS_FOLDER / "theorems.txt";
                        PROVED_THEOREMS_FILE = THEOREMS_FOLDER / "proved_theorems.txt";
                        COMPRESSED_EXTERNAL_THEOREMS_FILE = THEOREMS_FOLDER / "compressed_external_theorems.txt";
                    }
                    if (j.contains("raw_proof_graph_folder")) {
                        std::string folder = j["raw_proof_graph_folder"];
                        RAW_PROOF_DIR = PROJECT_ROOT / folder;
                    }
                    if (j.contains("background_theorems_folder")) {
                        std::string bgFolder = j["background_theorems_folder"];
                        BACKGROUND_PROVED_THEOREMS_FILE = PROJECT_ROOT / bgFolder / "proved_theorems.txt";
                    } else {
                        BACKGROUND_PROVED_THEOREMS_FILE = PROVED_THEOREMS_FILE;
                    }
                    if (j.contains("prover_parameters")) {
                        auto& pp = j["prover_parameters"];
                        if (pp.contains("ban_disintegration") && pp["ban_disintegration"].get<bool>()) {
                            skipCompression = true;
                        }
                    }
                } catch (...) {}
            }
        }

        // ====== PHASE 1: FULL PROVE ======
        gl::ExpressionAnalyzer expressionAnalyzer(anchor_id);

        std::cout << "Loading proved theorems..." << std::endl;
        std::unordered_set<std::string> proved_set = loadLinesFromFile(PROVED_THEOREMS_FILE);
        // Also load background theorems if different
        if (BACKGROUND_PROVED_THEOREMS_FILE != PROVED_THEOREMS_FILE) {
            auto bg_set = loadLinesFromFile(BACKGROUND_PROVED_THEOREMS_FILE);
            proved_set.insert(bg_set.begin(), bg_set.end());
        }
        std::vector<std::string> proved_lst(proved_set.begin(), proved_set.end());
        std::sort(proved_lst.begin(), proved_lst.end());
        std::cout << "Loaded " << proved_lst.size() << " proved theorems." << std::endl;

        std::cout << "Loading external theorems (incl. mirrors)..." << std::endl;
        std::unordered_set<std::string> external_set = loadLinesFromFile(COMPRESSED_EXTERNAL_THEOREMS_FILE);
        std::vector<std::string> external_lst(external_set.begin(), external_set.end());
        std::sort(external_lst.begin(), external_lst.end());
        std::cout << "Loaded " << external_lst.size() << " external theorems." << std::endl;

        if (!fs::exists(THEOREMS_FILE)) {
            std::cerr << "[full_run] Missing theorems file: " << THEOREMS_FILE << "\n";
            return;
        }

        std::unordered_set<std::string> theorem_set = loadLinesFromFile(THEOREMS_FILE);
        std::vector<std::string> tmp_lst(theorem_set.begin(), theorem_set.end());
        std::sort(tmp_lst.begin(), tmp_lst.end());

        expressionAnalyzer.analyzeExpressions(tmp_lst, proved_lst, external_lst);

        if (skipCompression) {
            // ====== INCUBATOR MODE: Save directly, no compression, no proof graph ======
            std::cout << "\nSkipping compression (incubator mode)." << std::endl;

            // Save proved theorems directly
            {
                std::ofstream ofs(PROVED_THEOREMS_FILE, std::ios::app);
                for (const auto& tpl : expressionAnalyzer.globalTheoremList) {
                    ofs << std::get<0>(tpl) << "\n";
                }
            }
            std::cout << "Saved " << expressionAnalyzer.globalTheoremList.size()
                      << " theorems to " << PROVED_THEOREMS_FILE << std::endl;

            // Generate proof graph in incubator mode too
            expressionAnalyzer.generateRawProofGraph(expressionAnalyzer.globalTheoremList, RAW_PROOF_DIR);
        } else {
            // ====== PHASE 2: COMPRESS ======
            std::cout << "\nInitiating Post-Proof Compression Phase..." << std::endl;

            // Start from previously proved theorems (already loaded into proved_lst)
            std::unordered_set<std::string> seenTheorems(proved_lst.begin(), proved_lst.end());
            std::vector<std::string> theoremsToCompress(proved_lst.begin(), proved_lst.end());

            // Add newly proved theorems not already present
            for (const auto& tpl : expressionAnalyzer.globalTheoremList) {
                const std::string& thm = std::get<0>(tpl);
                if (seenTheorems.insert(thm).second) {
                    theoremsToCompress.push_back(thm);
                }
            }

            // Add external theorems (incl. mirrors) to compression pool
            for (const auto& ext : external_lst) {
                if (seenTheorems.insert(ext).second) {
                    theoremsToCompress.push_back(ext);
                }
            }

            gl::Compressor compressor(expressionAnalyzer, theoremsToCompress);
            std::vector<std::string> survivingTheorems = compressor.run();

            // Determine which external theorems (incl. mirrors) survived compression
            std::unordered_set<std::string> survivingSet(survivingTheorems.begin(), survivingTheorems.end());
            std::vector<std::string> survivingExternals;
            std::vector<std::string> eliminatedExternals;
            for (const auto& ext : external_lst) {
                if (survivingSet.count(ext)) {
                    survivingExternals.push_back(ext);
                } else {
                    eliminatedExternals.push_back(ext);
                }
            }

            // Rewrite compressed_external_theorems.txt with survivors only
            {
                std::ofstream ofs(COMPRESSED_EXTERNAL_THEOREMS_FILE, std::ios::trunc);
                for (const auto& ext : survivingExternals) {
                    ofs << ext << "\n";
                }
            }

            // Append eliminated externals to compressed_out_theorems.txt
            if (!eliminatedExternals.empty()) {
                const auto compOutPath = THEOREMS_FOLDER / "compressed_out_theorems.txt";
                std::ofstream ofs(compOutPath, std::ios::app);
                for (const auto& ext : eliminatedExternals) {
                    ofs << ext << "\n";
                }
            }

            // ====== PHASE 3: SAVE + GRAPH ======
            // Save only essential theorems for cross-batch propagation
            // Exclude surviving external theorems (they are not GL-proved)
            std::unordered_set<std::string> survivingExternalSet(survivingExternals.begin(), survivingExternals.end());
            expressionAnalyzer.saveProvedTheoremsFiltered(survivingTheorems, survivingExternalSet);

            // Generate proof graph from the full globalTheoremList (unfiltered)
            if (!skipProofGraph) {
                if (expressionAnalyzer.parameters.debug) {
                    std::vector<std::string> expr_lst{ "(AnchorPeano[1,2,3,4,5,6])", "(in3[6,7,8,4])", "(in2[rec0,7,3])" };
                    expressionAnalyzer.findEnds(expr_lst, RAW_PROOF_DIR);
                }
                else {
                    expressionAnalyzer.generateRawProofGraph(expressionAnalyzer.globalTheoremList, RAW_PROOF_DIR);
                }
            }
        }
    }

} // namespace run_modes