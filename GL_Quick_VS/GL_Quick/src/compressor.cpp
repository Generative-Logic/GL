/* Generative Logic : A deterministic reasoning and knowledge generation engine.
 Copyright(C) 2025 Generative Logic UG(haftungsbeschraenkt)

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
#include "compressor.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <filesystem>
#include <unordered_map>

namespace gl {

    // =================================================================
    // Constructor
    // =================================================================

    Compressor::Compressor(ExpressionAnalyzer& analyzer,
        const std::vector<std::string>& all_theorems)
        : analyzer(analyzer),
        all_theorems(all_theorems)
    {
        for (const std::string& theorem : this->all_theorems) {
            compactToExpanded[theorem] = theorem;
        }
    }

    // =================================================================
    // run() — top-level entry point
    // =================================================================

    std::vector<std::string> Compressor::run() {
        std::cout << "\n======================================" << std::endl;
        std::cout << "Starting Compressor Phase 1 (Hash Bursts)" << std::endl;
        std::cout << "======================================" << std::endl;
        runPhase1();

        std::cout << "\n======================================" << std::endl;
        std::cout << "Starting Compressor Phase 2 (Greedy Elimination)" << std::endl;
        std::cout << "======================================" << std::endl;
        return runPhase2();
    }

    // =================================================================
    // Phase 1 — Build per-LB proof graphs
    // =================================================================

#pragma optimize("", off)

    void Compressor::runPhase1() {
        // --- Turn on compressor-specific flags ---
        analyzer.parameters.compressor_mode            = true;
        analyzer.parameters.ban_disintegration         = true;

        const size_t N = all_theorems.size();
        std::cout << "Building independent Logic Blocks for "
                  << N << " theorems..." << std::endl;

        std::vector<Memory*> compressorBodies;
        compressorBodies.reserve(N);
        extracted_graphs.reserve(N);

        // ---- 1. Prepare N independent Logic Blocks ----

        for (size_t i = 0; i < N; ++i) {
            const std::string& theorem = all_theorems[i];

            Memory* lb = new Memory();
            lb->level    = 0;
            lb->isActive = true;
            lb->exprKey  = "CompressorNode_" + std::to_string(i);

            // Load ALL proven theorems into hash memory as implication rules
            for (const std::string& rule : all_theorems) {
                std::vector<std::tuple<std::string,
                                       std::vector<std::string>,
                                       std::set<std::string>>> tempChain;
                std::string head = ce::disintegrateImplication(
                    rule, tempChain, analyzer.coreExpressionMap);
                std::vector<std::string> chain;
                for (auto& t : tempChain) chain.push_back(std::get<0>(t));

                analyzer.addToHashMemory(
                    chain, head, std::set<std::string>(), *lb, lb->overallHashMemory, { 0 }, rule,
                    analyzer.parameters.standardMaxAdmissionDepth,
                    analyzer.parameters.standardMaxSecondaryNumber,
                    false, analyzer.parameters.minNumOperatorsKey,
                    "implication", false, rule);

                lb->wholeExpressions.insert(EncodedExpression(rule, "main"));
            }

            // Split the target theorem into premises + head
            std::vector<std::tuple<std::string,
                                   std::vector<std::string>,
                                   std::set<std::string>>> targetChain;
            std::string targetHead = ce::disintegrateImplication(
                theorem, targetChain, analyzer.coreExpressionMap);

            CompressorNode pNode;
            pNode.originalTheorem = theorem;
            pNode.head = ExpressionWithValidity(targetHead, "main");

            // Insert premises as local fuel
            for (auto& t : targetChain) {
                std::string premise = std::get<0>(t);
                pNode.premises.insert(ExpressionWithValidity(premise, "main"));

                std::pair<std::string, std::vector<ExpressionWithValidity>> origin;
                origin.first = "premise";
                analyzer.addExprToMemoryBlock(
                    premise, *lb, -1, 1, { 0 }, origin, -1, -1, "main", false);
            }

            // Set head as proof goal
            analyzer.addExprToMemoryBlock(
                targetHead, *lb, -1, 2, { 0 },
                std::make_pair("goal", std::vector<ExpressionWithValidity>()),
                -1, -1, "main", true);

            extracted_graphs.push_back(pNode);
            compressorBodies.push_back(lb);
        }

        std::cout << "Running full Prover kernel on all Compressor nodes "
                     "simultaneously..." << std::endl;

        // ---- 2. Build networking & run hash bursts ----

        ParentChildrenMap compressorIndex =
            analyzer.buildParentChildrenMap(compressorBodies);
        PerCoreMailboxes  compressorBoxes =
            analyzer.buildPerCoreMailboxes(compressorIndex);

        analyzer.prove(analyzer.parameters.compressor_hash_bursts,
                       compressorBodies, compressorIndex, compressorBoxes);

        std::cout << "Extracting proof graphs and cleaning up..." << std::endl;

        // ---- 3. Extract lightweight graphs ----

        size_t total_origins = 0;
        for (size_t i = 0; i < compressorBodies.size(); ++i) {
            Memory* lb = compressorBodies[i];
            CompressorNode& pNode = extracted_graphs[i];

            for (const auto& kv : lb->exprOriginMap) {
                for (const auto& orig : kv.second) {
                    // Store the dep list (orig.second).
                    // Empty dep lists mean the expression is unconditionally
                    // derivable (e.g., premises, tautologies).
                    pNode.graph[kv.first].push_back(orig.second);
                    total_origins++;
                }
            }
            delete lb;
        }

        // ---- 4. Restore global parameters ----

        analyzer.parameters.compressor_mode            = false;
        analyzer.parameters.ban_disintegration         = false;

        std::cout << "Extracted " << extracted_graphs.size()
                  << " proof graphs  (" << total_origins
                  << " total origin entries)." << std::endl;
    }

    // =================================================================
    // isDerivable — forward reachability from premises + surviving thms
    //
    // Algorithm:
    //   1. Seed the "alive" set with:
    //      a. All premises of the node
    //      b. All theorems NOT in dead_theorems (as EWV with "main")
    //      c. All expressions that have at least one empty dep list
    //         (unconditionally derivable)
    //   2. Iterate: for each expression in the graph that is not yet
    //      alive, check if any of its dep lists has all deps alive.
    //      If so, mark it alive.
    //   3. Repeat until no new expressions become alive.
    //   4. Return whether the head is alive.
    // =================================================================

    bool Compressor::isDerivable(const CompressorNode& node,
                             const std::set<std::string>& dead_theorems) const
    {
        std::set<ExpressionWithValidity> alive;

        // (a) Seed premises
        for (const auto& p : node.premises) {
            alive.insert(p);
        }

        // (b) Seed surviving theorems
        for (const std::string& t : all_theorems) {
            if (dead_theorems.count(t) == 0) {
                alive.insert(ExpressionWithValidity(t, "main"));
            }
        }

        // (c) Seed unconditionally derivable expressions (empty dep list)
        for (const auto& kv : node.graph) {
            for (const auto& deps : kv.second) {
                if (deps.empty()) {
                    alive.insert(kv.first);
                    break;  // one empty dep list is enough
                }
            }
        }

        // Forward propagation
        bool changed = true;
        while (changed) {
            changed = false;
            for (const auto& kv : node.graph) {
                if (alive.count(kv.first)) continue;  // already alive
                for (const auto& deps : kv.second) {
                    bool all_alive = true;
                    for (const auto& dep : deps) {
                        if (alive.count(dep) == 0) {
                            all_alive = false;
                            break;
                        }
                    }
                    if (all_alive) {
                        alive.insert(kv.first);
                        changed = true;
                        break;  // found one live path, enough
                    }
                }
            }
        }

        return alive.count(node.head) > 0;
    }

    // =================================================================
    // Phase 2 — Greedy Elimination with multi-pass
    // =================================================================

    std::vector<std::string> Compressor::runPhase2() {

        // ---- 1. Compute per-theorem usage counts ----
        std::unordered_map<std::string, int> usage_count;
        for (const auto& thm : all_theorems) usage_count[thm] = 0;

        for (const auto& node : extracted_graphs) {
            for (const auto& kv : node.graph) {
                for (const auto& deps : kv.second) {
                    for (const auto& dep : deps) {
                        auto it = usage_count.find(dep.original);
                        if (it != usage_count.end()) {
                            it->second++;
                        }
                    }
                }
            }
        }

        // ---- 2. Sort by usage count ascending ----
        std::vector<std::string> sorted_theorems = all_theorems;
        std::stable_sort(sorted_theorems.begin(), sorted_theorems.end(),
            [&](const std::string& a, const std::string& b) {
                int ca = usage_count[a], cb = usage_count[b];
                return ca < cb || (ca == cb && a < b);
            });

        // ---- 3. Multi-pass greedy elimination ----
        std::set<std::string> essential;
        std::set<std::string> current_dead;

        bool pass_removed_any = true;
        int  pass_number = 0;

        while (pass_removed_any) {
            ++pass_number;
            pass_removed_any = false;
            int pass_removed_count = 0;
            int pass_essential_count = 0;

            std::cout << "\n--- Elimination pass " << pass_number
                << " ---" << std::endl;

            for (const std::string& candidate : sorted_theorems) {
                if (essential.count(candidate))    continue;
                if (current_dead.count(candidate)) continue;

                current_dead.insert(candidate);

                bool all_survive = true;
                for (const auto& node : extracted_graphs) {
                    if (!isDerivable(node, current_dead)) {
                        all_survive = false;
                        break;
                    }
                }

                if (!all_survive) {
                    essential.insert(candidate);
                    current_dead.erase(candidate);
                    pass_essential_count++;
                }
                else {
                    pass_removed_any = true;
                    pass_removed_count++;
                }
            }

            std::cout << "Pass " << pass_number << " complete: "
                << pass_removed_count << " removed this pass, "
                << current_dead.size() << " total removed, "
                << essential.size() << " essential so far."
                << std::endl;
        }

        // ---- 4. Collect results in expanded form ----
        std::vector<std::string> essential_list;
        essential_list.reserve(essential.size());
        for (const auto& thm : essential) {
            auto it = compactToExpanded.find(thm);
            essential_list.push_back(it != compactToExpanded.end() ? it->second : thm);
        }
        std::sort(essential_list.begin(), essential_list.end());

        // Anything not essential is compressed out
        std::vector<std::string> compressed_out_list;
        for (const auto& thm : all_theorems) {
            if (!essential.count(thm))
                compressed_out_list.push_back(thm);
        }

        // ---- 5. Write output files ----
        namespace fs = std::filesystem;
        std::error_code ec;
        const auto theoremsDir =
            fs::path(__FILE__).parent_path()
            .parent_path().parent_path().parent_path()
            / "files" / "theorems";

        fs::create_directories(theoremsDir, ec);

        {
            const auto path = theoremsDir / "compressed_out_theorems.txt";
            std::ofstream out(path, std::ios::app);
            if (out.is_open()) {
                for (const auto& th : compressed_out_list)
                    out << th << "\n";
            }
            else {
                std::cerr << "Compressor Error: Could not open "
                    << path << std::endl;
            }
        }

        // ---- 6. Summary ----
        std::cout << "\n*** Compression Complete ***" << std::endl;
        std::cout << "Essential core: " << essential_list.size()
            << " / " << all_theorems.size() << " theorems."
            << std::endl;
        std::cout << "Compressed out: " << compressed_out_list.size()
            << " redundant theorems saved to "
            "'files/theorems/compressed_out_theorems.txt'."
            << std::endl;
        std::cout << "Convergence reached after " << pass_number
            << " pass(es)." << std::endl;

        return essential_list;
    }

} // namespace gl
