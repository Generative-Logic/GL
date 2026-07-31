/* Generative Logic : A deterministic reasoning and knowledge generation engine.
 Copyright(C) 2025-2026 Generative Logic UG(haftungsbeschraenkt)

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

    /// @brief Constructor — store references to the prover state and the
    /// full theorem list, seed the compact↔expanded mapping.
    ///
    /// @details
    /// `compactToExpanded` is populated up-front with the identity mapping
    /// (compact == expanded). The compressor's Phase 1 may overwrite some
    /// entries when it discovers that a theorem appears in the prover's
    /// `globalTheoremList` under an expanded form different from its
    /// compact one (the `or0`/`existence2` rewrite at `prover.cpp:7407`
    /// is the canonical example). The map ensures `runPhase2`'s output
    /// is in the same form the rest of the pipeline expects.
    Compressor::Compressor(ExpressionAnalyzer& analyzer,
        const std::vector<std::string>& all_theorems)
        : analyzer(analyzer),
        all_theorems(all_theorems)
    {
        for (const std::string& theorem : this->all_theorems) {
            compactToExpanded[theorem] = theorem;
        }
    }

    /// @brief Top-level entry — run Phase 1 (graph extraction) then
    /// Phase 2 (greedy redundancy elimination).
    ///
    /// @details
    /// Prints progress banners between phases for the run-mode log; the
    /// banners are part of `run_modes`'s narrative and downstream log
    /// scrapers grep for them — do not silently change the text.
    /// Returns the surviving theorem strings in stable order.
    ///
    /// @return Surviving theorem texts.
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

    /// @brief Phase 1 — extract one `CompressorNode` per theorem by re-running
    /// the prover under compressor flags.
    ///
    /// @details
    /// Toggles `analyzer.parameters.compressor_mode = true` and
    /// `analyzer.parameters.ban_disintegration = true` for the duration of
    /// Phase 1. `compressor_mode` does several things:
    /// 1. Disables Pass B (gated by `!compressor_mode` in prover.cpp's
    ///    Pass B entry) so the compressor's per-theorem run does not
    ///    spawn disintegration products that would explode the graph.
    /// 2. Caps origin storage at `compressor_max_origins_per_expr`
    ///    (vs the larger `max_origin_per_expr` used in the main run) so
    ///    each `CompressorNode::graph` stays bounded.
    /// 3. Selects compressor-specific filter / mail behaviours where
    ///    relevant.
    ///
    /// `ban_disintegration` is the umbrella flag from D-28 (since
    /// 2026-04-29) that gates Pass B + back-reformulation +
    /// hypo-disintegration together; setting both flags is the
    /// belt-and-suspenders approach.
    ///
    /// `#pragma optimize("", off)` disables MSVC's whole-function
    /// optimizer for this body (some MSVC versions have eaten the
    /// per-theorem loop's invariants under aggressive inlining; the
    /// pragma is a known-good workaround). Do NOT remove without
    /// re-verifying on the affected compiler version.
    ///
    /// @post `extracted_graphs.size() == all_theorems.size()`. Each
    ///       node carries the LB's graph, premises, head, and original
    ///       theorem text.
    /// @invariant [I-7](../../docs/agentic_swdd/30_invariants.md#i-7) — Pass B is
    ///            gated by `!ban_disintegration`; setting that flag here
    ///            disables disintegration for compressor mode.
    void Compressor::runPhase1() {
        // --- Turn on compressor-specific flags ---
        analyzer.parameters.compressor_mode            = true;
        analyzer.parameters.ban_disintegration         = true;
        // Pass B is gated by !compressor_mode at prover.cpp:7216, so
        // setting compressor_mode=true is sufficient to disable it during
        // Phase 1 even when the config has allow_disintegration=true.

        const size_t N = all_theorems.size();
        std::cout << "Building independent Logic Blocks for "
                  << N << " theorems..." << std::endl;

        std::vector<Memory*> compressorBodies;
        compressorBodies.reserve(N);
        extracted_graphs.reserve(N);

        // ---- 1. Prepare N independent Logic Blocks ----

        // Compressor deposits are all level {0} — one shared stack run.
        const int lvl0Run[1] = { 0 };

        for (size_t i = 0; i < N; ++i) {
            const std::string& theorem = all_theorems[i];

            Memory* lb = analyzer.lbStore.create<Memory>();
            lb->level    = 0;
            lb->isActive = true;
            lb->setExprKey("CompressorNode_" + std::to_string(i));

            // Load ALL proven theorems into hash memory as implication rules
            for (const std::string& rule : all_theorems) {
                std::vector<std::tuple<std::string,
                                       std::vector<std::string>,
                                       std::set<std::string>>> tempChain;
                std::string head = ce::disintegrateImplication(
                    rule, tempChain, analyzer.coreExpressionMap);
                std::vector<std::string> chain;
                for (auto& t : tempChain) chain.push_back(std::get<0>(t));

                StrSpan chainRun[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
                int32_t chainRunN = 0;
                for (const std::string& s : chain) {
                    assert(chainRunN < ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS
                        && "addToHashMemory chain run exceeds cap");
                    chainRun[chainRunN++] = StrSpan(s);
                }
                analyzer.addToHashMemory(
                    chainRun, chainRunN, StrSpan(head), nullptr, 0, *lb, lb->overallHashMemory, lvl0Run, 1, StrSpan(rule),
                    analyzer.parameters.maxIterationNumberVariable,
                    analyzer.parameters.standardMaxSecondaryNumber,
                    false, analyzer.parameters.minNumOperatorsKey,
                    StrSpan("implication", 11), false, StrSpan(rule));

                // `registered` only: the rule load marks the theorem as a
                // registered statement of the scratch LB without admitting it
                // to the level registry (the Site F dedup record stays blind
                // to it, as the proof engine expects).
                upsertStatementKey(lb->intKnownStatements,
                    packStatementKey(lb->nameMap.encode(rule),
                                     lb->nameMap.encode("main")),
                    /*local=*/true, /*registered=*/true, /*known=*/false);
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

                const TransientOrigin origin{ true, OriginTag::premise, nullptr, 0 };
                analyzer.addExprToMemoryBlock(
                    premise, *lb, -1, 1, lvl0Run, 1, origin, -1, -1, StrSpan("main", 4), false);
            }

            // Set head as proof goal
            analyzer.addExprToMemoryBlock(
                targetHead, *lb, -1, 2, lvl0Run, 1,
                TransientOrigin{ true, OriginTag::goal, nullptr, 0 },
                -1, -1, StrSpan("main", 4), true);

            extracted_graphs.push_back(pNode);
            compressorBodies.push_back(lb);
        }

        std::cout << "Running full Prover kernel on all Compressor nodes "
                     "simultaneously..." << std::endl;

        // ---- 2. Run hash bursts ----

        analyzer.prove(analyzer.parameters.compressor_hash_bursts,
                       compressorBodies);

        std::cout << "Extracting proof graphs and cleaning up..." << std::endl;

        // ---- 3. Extract lightweight graphs ----

        size_t total_origins = 0;
        for (size_t i = 0; i < compressorBodies.size(); ++i) {
            Memory* lb = compressorBodies[i];
            CompressorNode& pNode = extracted_graphs[i];

            {
                // Decoded lex-sorted snapshot — strings are the cross-LB
                // common space of the graph (each compressor LB has its
                // own interner); per-row line order is insertion order.
                const auto originRows =
                    decodeOriginMapSorted(lb->exprOriginMap, lb->originInterner);
                for (const auto& row : originRows) {
                    const ExpressionWithValidity rowKey(row.first.first,
                                                        row.first.second);
                    for (const auto& orig : row.second) {
                        // Store the dep list (orig.second).
                        // Empty dep lists mean the expression is unconditionally
                        // derivable (e.g., premises, tautologies).
                        pNode.graph[rowKey].push_back(orig.second);
                        total_origins++;
                    }
                }
            }
            analyzer.lbStore.destroy(lb);
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

    /// @brief Forward-reachability oracle — true iff `node.head` is reachable
    /// from `node.premises` plus the surviving theorems given a candidate
    /// kill set.
    ///
    /// @details
    /// Walks `node.graph` BFS-style: a head is reachable if any of its
    /// alternative dependency-lists has every dep reachable. Reachability
    /// is determined by:
    /// - presence in `node.premises` (always alive),
    /// - presence in the surviving theorem set (i.e. the set of all
    ///   theorems minus `dead_theorems`),
    /// - or transitive reachability through other graph edges.
    ///
    /// Used inside Phase 2's per-theorem kill-test loop; called once per
    /// `(theorem, candidate_kill_set)` pair, so its cost dominates Phase
    /// 2 wall-clock. Implementation is iterative (no recursion) to
    /// avoid stack growth on the largest Gauss graphs.
    ///
    /// @param node           The CompressorNode whose head we are testing.
    /// @param dead_theorems  Candidate kill set; theorems in this set are
    ///                       treated as unavailable.
    /// @return True iff `node.head` is reachable.
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

    /// @brief Phase 2 — greedy multi-pass redundancy elimination.
    ///
    /// @details
    /// Iterates the theorem list in `std::stable_sort` order (per
    /// `OPEN-13` in `docs/agentic_swdd/SwDD.md` — output determinism depends on
    /// stable ordering). Each pass walks every theorem and tests
    /// `isDerivable(node, dead ∪ {theorem})` on every other LB's
    /// CompressorNode; if every head stays derivable, the theorem is
    /// killed. Repeats until no theorem is killed in a full pass —
    /// fixed-point convergence.
    ///
    /// Output is the surviving theorem texts in the same stable order
    /// as the input. Compressed-out theorems are written to
    /// `files/theorems/compressed_out_theorems.txt` (and
    /// `compressed_external_theorems.txt` for the externals path) by
    /// `run_modes::fullRun`'s post-compressor write step; this function
    /// returns only the survivors.
    ///
    /// @return Surviving theorem texts.
    /// @see OPEN-13 in `docs/agentic_swdd/SwDD.md` — stable-sort determinism.
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
