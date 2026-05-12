/* Generative Logic : A deterministic reasoning and knowledge generation engine.
 Copyright(C) 2025-2026 Generative Logic UG(haftungsbeschr�nkt)

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
#include "prover.hpp"
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <unordered_set>
#include <set>
#include <utility>
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
                        if (pp.contains("incubator_mode") && pp["incubator_mode"].get<bool>()) {
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

        // DEBUG FILTER disabled for hash-burst investigation: with the culprit
        // theorem back in the broadcast pool, Gauss reverts to the failure
        // mode and the trap in prover.hpp can capture its hash-burst output
        // at the target LB.
        // if (anchor_id == "Gauss") {
        //     const std::string target =
        //         "(>[1,3,6](AnchorPeano[1,2,3,4,5,6])!(>[7](in[7,1])!(in2[7,6,3])))";
        //     proved_set.erase(target);
        // }

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

        // Load OR pairs if present
        {
            auto orPairsPath = THEOREMS_FOLDER / "or_pairs.txt";
            if (fs::exists(orPairsPath)) {
                std::ifstream orIn(orPairsPath);
                std::string line;
                while (std::getline(orIn, line)) {
                    line = trim_copy(line);
                    if (line.empty()) continue;
                    auto tabPos = line.find('\t');
                    if (tabPos != std::string::npos) {
                        gl::ExpressionAnalyzer::OrCandidate oc;
                        oc.existenceTheorem = line.substr(0, tabPos);
                        oc.companionTheorem = line.substr(tabPos + 1);
                        expressionAnalyzer.orCandidates.push_back(oc);
                    }
                }
                std::cout << "Loaded " << expressionAnalyzer.orCandidates.size() << " OR pairs." << std::endl;
            }
        }

        expressionAnalyzer.analyzeExpressions(tmp_lst, proved_lst, external_lst);

        // Sort globalTheoremList for deterministic downstream processing
        // (OR construction deferred to after compression)
        std::sort(expressionAnalyzer.globalTheoremList.begin(),
                  expressionAnalyzer.globalTheoremList.end());

        // ====== COMPRESS (D-25: moved out of analyzeExpressions) ======
        // Pool must include: (a) old proved theorems from prior batches, (b)
        // new theorems proved in this batch (globalTheoremList), (c) external
        // theorems read from compressed_external_theorems.txt. Otherwise the
        // compressor cannot evaluate whether old/external theorems remain
        // essential, and cross-batch pruning is broken.
        // fullTheoremList is populated from the pre-compression batch even when
        // skipCompression is true, so the proof-graph generator below can use
        // it (run_modes.cpp:303-305 falls back to globalTheoremList only when
        // fullTheoremList is empty).
        if (!skipCompression) {
            std::vector<std::string> theoremsForCompressor;
            std::unordered_set<std::string> seen;
            std::set<std::string> alreadyInFull;
            // (a) Old proved theorems from prior batches.
            for (const auto& thm : proved_lst) {
                std::string compiled = thm;
                expressionAnalyzer.precompileStructuralOperators(compiled);
                if (seen.insert(compiled).second) {
                    theoremsForCompressor.push_back(compiled);
                }
            }
            // (b) New theorems from this batch (already compiled — produced
            // in-flight through the compiled pipeline).
            for (const auto& tpl : expressionAnalyzer.globalTheoremList) {
                const std::string& thm = std::get<0>(tpl);
                if (seen.insert(thm).second) {
                    theoremsForCompressor.push_back(thm);
                }
                if (alreadyInFull.insert(thm).second) {
                    expressionAnalyzer.fullTheoremList.push_back(tpl);
                }
            }
            // (c) External theorems (mirrors included).
            for (const auto& ext : external_lst) {
                std::string compiled = ext;
                expressionAnalyzer.precompileStructuralOperators(compiled);
                if (seen.insert(compiled).second) {
                    theoremsForCompressor.push_back(compiled);
                }
            }
            if (theoremsForCompressor.size() > 1) {
                std::cout << "\nCompressing (" << theoremsForCompressor.size() << " theorems)..." << std::endl;
                gl::Compressor compressor(expressionAnalyzer, theoremsForCompressor);
                std::vector<std::string> survivors = compressor.run();
                std::unordered_set<std::string> survivorSet(survivors.begin(), survivors.end());

                expressionAnalyzer.lastCompressionSurvivors = survivors;

                auto& gtl = expressionAnalyzer.globalTheoremList;
                gtl.erase(std::remove_if(gtl.begin(), gtl.end(),
                    [&](const std::tuple<std::string, std::string, std::string, std::string>& t) {
                        return survivorSet.find(std::get<0>(t)) == survivorSet.end();
                    }), gtl.end());
                std::cout << "After compression: " << survivors.size() << " essential theorems." << std::endl;
            } else {
                expressionAnalyzer.lastCompressionSurvivors = theoremsForCompressor;
            }
        }

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
            // ====== PHASE 2: SAVE ======
            // globalTheoremList was compressed by the run_modes.cpp invocation above.
            // Handle external theorem pruning and save proved_theorems.txt.

            // Full survivor set from compression: includes old proved theorems
            // (from prior batches), new theorems from this batch, and external
            // theorems that remain essential. Sourced from analyzeExpressions via
            // lastCompressionSurvivors because globalTheoremList only contains
            // this-batch entries after pruning.
            const std::vector<std::string>& survivorsVec =
                expressionAnalyzer.lastCompressionSurvivors;
            std::unordered_set<std::string> survivorsSet(
                survivorsVec.begin(), survivorsVec.end());

            // Determine which external theorems survived (are still needed)
            std::vector<std::string> survivingExternals;
            std::vector<std::string> eliminatedExternals;
            for (const auto& ext : external_lst) {
                if (survivorsSet.count(ext)) {
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

            // Build survivingTheorems list (essential survivors from compression).
            // saveProvedTheoremsFiltered will be called AFTER OR construction so
            // we can drop OR-consumed parents in the same write.
            std::vector<std::string> survivingTheorems = survivorsVec;
            std::sort(survivingTheorems.begin(), survivingTheorems.end());
            survivingTheorems.erase(
                std::unique(survivingTheorems.begin(), survivingTheorems.end()),
                survivingTheorems.end());
            std::unordered_set<std::string> survivingExternalSet(survivingExternals.begin(), survivingExternals.end());

            // OR construction (after compression, before proof graph).
            //
            // Two cleanups beyond the prior straight-loop:
            //   1. De-dup A∨B vs B∨A.  orPairsFromHeadSwitch contains BOTH
            //      (mirror1, mirror2) AND (mirror2, mirror1) because head-switch
            //      walks every theorem with a negated premise.  Constructing
            //      both produces or<N> and or<N+1> for the same logical OR with
            //      disjuncts in opposite order — pre-fix this is exactly the
            //      or0/or1 duplication on Peano.  Canonicalize each pair as
            //      (min, max) and skip already-seen disjunct-sets.
            //   2. Remove parent theorems on OR creation.  Once or<N> is
            //      registered, the K mutual-exclusion implications recovered
            //      via OR-disintegration produce both mirrors as derived
            //      results.  The standalone parent theorems then shadow the
            //      OR — same procedure as the compressor's redundancy removal,
            //      done manually here at OR-construction time so it lands in
            //      the same emit pass rather than racing the compressor.
            //
            // Per D-55.
            std::vector<std::string> orTheorems;
            std::set<std::string> consumedParents;
            {
                std::unordered_set<std::string> provedSet;
                for (const auto& t : expressionAnalyzer.globalTheoremList)
                    provedSet.insert(std::get<0>(t));

                std::set<std::pair<std::string, std::string>> seenDisjunctSets;

                for (const auto& [exist, comp] : expressionAnalyzer.orPairsFromHeadSwitch) {
                    if (!provedSet.count(exist) || !provedSet.count(comp)) continue;

                    // (exist, comp) and (comp, exist) describe the same OR
                    // (mirror reformulations of one another).  Canonicalize
                    // by lexicographically sorting the pair and skip duplicates.
                    auto a = std::min(exist, comp);
                    auto b = std::max(exist, comp);
                    if (!seenDisjunctSets.insert({a, b}).second) {
                        std::cout << "OR variant skipped (duplicate disjunct-set already constructed)"
                                  << std::endl;
                        continue;
                    }

                    std::string orThm = expressionAnalyzer.constructOrTheorem(exist, comp);
                    if (orThm.empty()) continue;

                    expressionAnalyzer.globalTheoremList.emplace_back(orThm, "or theorem", exist, comp);
                    expressionAnalyzer.fullTheoremList.emplace_back(orThm, "or theorem", exist, comp);
                    orTheorems.push_back(orThm);
                    consumedParents.insert(exist);
                    consumedParents.insert(comp);
                    std::cout << "OR theorem constructed: " << orThm << std::endl;
                    std::cout << "  parent removed (subsumed by OR): " << exist << std::endl;
                    std::cout << "  parent removed (subsumed by OR): " << comp << std::endl;
                }
            }

            // Drop OR-consumed parents from the survivors list before saving.
            if (!consumedParents.empty()) {
                survivingTheorems.erase(
                    std::remove_if(survivingTheorems.begin(), survivingTheorems.end(),
                        [&](const std::string& t) { return consumedParents.count(t) > 0; }),
                    survivingTheorems.end());
            }

            // Save proved theorems (essential survivors minus OR-consumed parents,
            // excluding externals).  saveProvedTheoremsFiltered truncates and
            // rewrites both proved_theorems.txt and compiled_proved_theorems.txt.
            expressionAnalyzer.saveProvedTheoremsFiltered(survivingTheorems, survivingExternalSet);

            // Append OR theorems to both proved_theorems files.
            // proved_theorems.txt: expanded form for inter-batch communication.
            // compiled_proved_theorems.txt: compiled form for proof graph pruning.
            if (!orTheorems.empty()) {
                std::ofstream ofs(PROVED_THEOREMS_FILE, std::ios::app);
                std::ofstream ofsCompiled(THEOREMS_FOLDER / "compiled_proved_theorems.txt", std::ios::app);
                for (const auto& ot : orTheorems) {
                    if (ofsCompiled.is_open()) ofsCompiled << ot << "\n";
                    std::string expanded = expressionAnalyzer.expandToBaseForm(ot);
                    ofs << expanded << "\n";
                }
            }

            // Append OR-consumed parents to compressed_out_theorems.txt — they
            // are now subsumed by the OR theorem (recoverable via OR-disintegration's
            // K mutual-exclusion implications) and treated as redundant going
            // forward.  Same procedure the compressor applies to redundant
            // theorems, done manually here at OR-creation time.
            if (!consumedParents.empty()) {
                std::ofstream ofs(THEOREMS_FOLDER / "compressed_out_theorems.txt", std::ios::app);
                for (const auto& p : consumedParents) {
                    ofs << p << "\n";
                }
            }

            // Generate proof graph using fullTheoremList (all proved theorems incl. non-essential)
            // so proof stacks can reference non-essential theorems. Python pruning trims the rest.
            if (!skipProofGraph) {
                // Use fullTheoremList if populated (big iteration mode); otherwise globalTheoremList
                auto& listForGraph = expressionAnalyzer.fullTheoremList.empty()
                    ? expressionAnalyzer.globalTheoremList
                    : expressionAnalyzer.fullTheoremList;
                std::sort(listForGraph.begin(), listForGraph.end());
                if (expressionAnalyzer.parameters.debug) {
                    std::vector<std::string> expr_lst{ "(AnchorPeano[1,2,3,4,5,6])", "(in3[6,7,8,4])", "(in2[rec0,7,3])" };
                    expressionAnalyzer.findEnds(expr_lst, RAW_PROOF_DIR);
                }
                else {
                    expressionAnalyzer.generateRawProofGraph(listForGraph, RAW_PROOF_DIR);
                }
            }
        }
    }

} // namespace run_modes