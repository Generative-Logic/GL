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
#include "project_paths.hpp"
#include "compressor.hpp"
#include "memory_infra/lb_deload.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <filesystem>
#include <unordered_map>
#include <chrono>

namespace gl {

    /// @brief Constructor — store references to the prover state and the
    /// full theorem list, seed the compact↔expanded mapping.
    ///
    /// @details
    /// `compactToExpanded` is populated up-front with the identity mapping
    /// (compact == expanded). The compressor's Phase 1 may overwrite some
    /// entries when it discovers that a theorem appears in the prover's
    /// `globalTheoremList` under an expanded form different from its
    /// compact one. The map ensures `runPhase2`'s output
    /// is in the same form the rest of the pipeline expects.
    Compressor::Compressor(Prover& analyzer,
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
    /// Returns surviving expanded theorem strings in lexicographic order.
    ///
    /// @return Surviving theorem texts.
    std::vector<std::string> Compressor::run() {
        std::cout << "\n======================================" << std::endl;
        std::cout << "Starting Compressor Phase 1 (Hash Bursts)" << std::endl;
        std::cout << "======================================" << std::endl;
        const auto phase1Started = std::chrono::steady_clock::now();
        runPhase1();
        std::cout << "Compressor Phase 1 seconds: "
                  << std::chrono::duration<double>(
                      std::chrono::steady_clock::now() - phase1Started).count()
                  << std::endl;

        std::cout << "\n======================================" << std::endl;
        std::cout << "Starting Compressor Phase 2 (Greedy Elimination)" << std::endl;
        std::cout << "======================================" << std::endl;
        const auto phase2Started = std::chrono::steady_clock::now();
        auto result = runPhase2();
        std::cout << "Compressor Phase 2 seconds: "
                  << std::chrono::duration<double>(
                      std::chrono::steady_clock::now() - phase2Started).count()
                  << std::endl;
        return result;
    }

    /// @brief Phase 1 — extract one `CompressorNode` per theorem by re-running
    /// the prover under compressor flags.
    ///
    /// @details
    /// Toggles `analyzer.configuration().compressor_mode = true` and
    /// `analyzer.configuration().ban_disintegration = true` for the duration of
    /// Phase 1. `compressor_mode` does several things:
    /// 1. Disables Pass B (gated by `!compressor_mode` in compressor.cpp's
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
    /// @post `extracted_graphs.size() == all_theorems.size()`. Each
    ///       node carries the LB's graph, premises, head, and original
    ///       theorem text.
    /// @invariant [I-7](../../docs/agentic_swdd/30_invariants.md#i-7) — Pass B is
    ///            gated by `!ban_disintegration`; setting that flag here
    ///            disables disintegration for compressor mode.
    void Compressor::runPhase1() {
        using PhaseClock = std::chrono::steady_clock;
        const auto secondsSince = [](PhaseClock::time_point started) {
            return std::chrono::duration<double>(PhaseClock::now() - started).count();
        };
        double creationSeconds = 0.0;
        double ruleInstallSeconds = 0.0;
        double ruleRegistrationSeconds = 0.0;
        double seedingSeconds = 0.0;
        double extractionSeconds = 0.0;
        double originDecodeSeconds = 0.0;
        double graphAssemblySeconds = 0.0;
        double destructionSeconds = 0.0;
        // --- Turn on compressor-specific flags ---
        analyzer.setCompressionMode(true);

        const size_t N = all_theorems.size();
        std::cout << "Building independent Logic Blocks for "
                  << N << " theorems..." << std::endl;

        std::vector<Memory*> compressorBodies;
        compressorBodies.reserve(N);
        extracted_graphs.reserve(N);

        // ---- 1. Prepare N independent Logic Blocks ----

        // Rule installs and goals keep level {0}; loaded premise STATEMENTS
        // are non-derived and carry the {-1} tier (transparent to level
        // accounting — the statement-levels contract).
        const int lvl0Run[1] = { 0 };
        const int lvlNonDerived[1] = { -1 };

        // The parser reads only theorem text. Keep its result once per theorem;
        // every independent Logic Block still installs the same rules in order.
        const auto parsingStarted = PhaseClock::now();
        std::vector<std::vector<std::string>> premiseChains(N);
        std::vector<std::string> theoremHeads(N);
        for (size_t i = 0; i < N; ++i) {
            std::vector<std::tuple<std::string, std::vector<std::string>,
                                   std::set<std::string>>> chain;
            theoremHeads[i] = ce::disintegrateImplication(
                all_theorems[i], chain, analyzer.definitionRegistry().configurations());
            for (const auto& entry : chain)
                premiseChains[i].push_back(std::get<0>(entry));
        }

        const double parsingSeconds = secondsSince(parsingStarted);

        for (size_t i = 0; i < N; ++i) {
            const auto creationStarted = PhaseClock::now();
            const std::string& theorem = all_theorems[i];

            Memory* lb = analyzer.createCompressionBlock();
            lb->level    = 0;
            lb->isActive = true;
            lb->setExprKey("CompressorNode_" + std::to_string(i));
            // The node key holds no arguments: record the definition sets of
            // the target theorem's variables before its premises and head are
            // seeded at the node's main (I-267).
            {
                StrSpan leaves[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS + 1];
                int32_t leafN = 0;
                for (const std::string& premise : premiseChains[i]) {
                    assert(leafN < ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS
                        && "compressor node: premise chain exceeds MAX_ADMISSION_KEY_ELEMENTS");
                    leaves[leafN++] = StrSpan(premise);
                }
                leaves[leafN++] = StrSpan(theoremHeads[i]);
                analyzer.types().assignTheoremVarDefinitionSets(*lb, leaves, leafN);
                // A set- or relation-typed `=` head integrates by
                // extensionality: its entity must exist before the head is
                // seeded as the node's goal.
                analyzer.types().ensureTypedEqualityEntityForTheorem(leaves, leafN);
            }
            creationSeconds += secondsSince(creationStarted);

            // No inference consumes the rule indexes during loading. Keep one
            // window across the input sequence and close it before seeding.
            assert(lb->ruleStagingArmDepth == 0);
            ++lb->ruleStagingArmDepth;
            // Load ALL proven theorems into hash memory as implication rules
            for (size_t ruleIndex = 0; ruleIndex < N; ++ruleIndex) {
                const auto ruleInstallStarted = PhaseClock::now();
                const std::string& rule = all_theorems[ruleIndex];
                const std::string& head = theoremHeads[ruleIndex];
                const auto& chain = premiseChains[ruleIndex];

                StrSpan chainRun[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
                int32_t chainRunN = 0;
                for (const std::string& s : chain) {
                    assert(chainRunN < ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS
                        && "addToHashMemory chain run exceeds cap");
                    chainRun[chainRunN++] = StrSpan(s);
                }
                analyzer.hashRules().addToHashMemory(
                    chainRun, chainRunN, StrSpan(head), nullptr, 0, *lb, lb->overallHashMemory, lvl0Run, 1, StrSpan(rule),
                    analyzer.configuration().maxIterationNumberVariable,
                    analyzer.configuration().standardMaxSecondaryNumber,
                    false, analyzer.configuration().minNumOperatorsKey,
                    StrSpan("implication", 11), false, StrSpan(rule));
                ruleInstallSeconds += secondsSince(ruleInstallStarted);

                // The rule load's statement row rides the one presence
                // convention: a row with its paired {-1} (non-derived)
                // levels. Site F stays blind to it in compressor mode (the
                // scan is compiled out there), so the proof engine's view is
                // unchanged.
                const auto registrationStarted = PhaseClock::now();
                {
                    const int64_t pkRule = packStatementKey(
                        lb->nameMap.encode(rule), lb->nameMap.encode("main"));
                    lb->intStatementLevelsMap.assignSetRange(
                        pkRule, lvlNonDerived, lvlNonDerived + 1);
                    upsertStatementKey(lb->intKnownStatements, pkRule,
                        /*local=*/true);
                }
                ruleRegistrationSeconds += secondsSince(registrationStarted);
            }
            const auto ruleFlushStarted = PhaseClock::now();
            assert(lb->ruleStagingArmDepth == 1);
            --lb->ruleStagingArmDepth;
            gl::HashRuleIndex::closeRuleIndexStaging(*lb);
            ruleInstallSeconds += secondsSince(ruleFlushStarted);

            // Split the target theorem into premises + head
            const auto seedingStarted = PhaseClock::now();
            assert(lb->ruleStagingArmDepth == 0);
            ++lb->ruleStagingArmDepth;
            const std::string& targetHead = theoremHeads[i];

            CompressorNode pNode;
            pNode.originalTheorem = theorem;
            pNode.head = ExpressionWithValidity(targetHead, "main");

            // Insert premises as local fuel
            for (const std::string& premise : premiseChains[i]) {
                pNode.premises.insert(ExpressionWithValidity(premise, "main"));

                const TransientOrigin origin{ true, OriginTag::premise, nullptr, 0 };
                analyzer.statements().addExprToMemoryBlock(
                    premise, *lb, -1, 1, lvlNonDerived, 1, origin, -1, -1, StrSpan("main", 4), false);
            }

            // Set head as proof goal
            analyzer.statements().addExprToMemoryBlock(
                targetHead, *lb, -1, 2, lvl0Run, 1,
                TransientOrigin{ true, OriginTag::goal, nullptr, 0 },
                -1, -1, StrSpan("main", 4), true);

            extracted_graphs.push_back(std::move(pNode));
            compressorBodies.push_back(lb);
            assert(lb->ruleStagingArmDepth == 1);
            --lb->ruleStagingArmDepth;
            gl::HashRuleIndex::closeRuleIndexStaging(*lb);
            seedingSeconds += secondsSince(seedingStarted);
        }

        std::cout << "Running full Prover kernel on all Compressor nodes "
                     "simultaneously..." << std::endl;

        // ---- 2. Run hash bursts ----

        const auto provingStarted = PhaseClock::now();
        analyzer.prove(analyzer.configuration().compressor_hash_bursts,
                       compressorBodies);
        const double provingSeconds = secondsSince(provingStarted);

        std::cout << "Extracting proof graphs and cleaning up..." << std::endl;

        // ---- 3. Extract lightweight graphs ----

        size_t total_origins = 0;
        for (size_t i = 0; i < compressorBodies.size(); ++i) {
            Memory* lb = compressorBodies[i];
            CompressorNode& pNode = extracted_graphs[i];

            const auto extractionStarted = PhaseClock::now();
            // A proved compressor LB was discharged, and its barrier drained
            // it to its canonical image: reload it read-only (no-op when
            // resident, sanctioned on discharged LBs, D-158) before reading
            // its origin history. The block is destroyed right after.
            lb->ensureLoadedForRead(lbdeload::kDeloadDirectory);
            {
                // Decoded lex-sorted snapshot — strings are the cross-LB
                // common space of the graph (each compressor LB has its
                // own interner); per-row line order is insertion order.
                const auto decodingStarted = PhaseClock::now();
                const auto originRows =
                    decodeOriginMapSorted(lb->exprOriginMap, lb->originInterner);
                originDecodeSeconds += secondsSince(decodingStarted);
                const auto assemblyStarted = PhaseClock::now();
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
                graphAssemblySeconds += secondsSince(assemblyStarted);
            }
            extractionSeconds += secondsSince(extractionStarted);
            const auto destructionStarted = PhaseClock::now();
            analyzer.destroyCompressionBlock(lb);
            destructionSeconds += secondsSince(destructionStarted);
        }

        // ---- 4. Restore global parameters ----

        analyzer.setCompressionMode(false);

        std::cout << "Extracted " << extracted_graphs.size()
                  << " proof graphs  (" << total_origins
                  << " total origin entries)." << std::endl;
        // Serial compressor orchestration timings; no timers enter the
        // statified inference functions. Extraction's remainder includes
        // destroying the decoded snapshot after its rows have been copied.
        const double extractionOtherSeconds = extractionSeconds
            - originDecodeSeconds - graphAssemblySeconds;
        assert(extractionOtherSeconds >= 0.0);
        std::cout << "Compressor Phase 1 breakdown seconds:"
                  << " parse=" << parsingSeconds
                  << " create=" << creationSeconds
                  << " rule_install=" << ruleInstallSeconds
                  << " rule_registration=" << ruleRegistrationSeconds
                  << " seeding=" << seedingSeconds
                  << " prove=" << provingSeconds
                  << " origin_decode=" << originDecodeSeconds
                  << " graph_assembly=" << graphAssemblySeconds
                  << " extraction_other=" << extractionOtherSeconds
                  << " destroy=" << destructionSeconds << std::endl;
    }

    /// @brief Build a reusable integer index of a compressor graph.
    ///
    /// @details Local indices identify full expression and scope pairs. Each
    /// dependency occurrence contributes one reverse link and one counter unit,
    /// including duplicates. Empty alternatives become unconditional seeds.
    /// @param node Extracted dependency graph and its premise/head contract.
    /// @param theorems The complete main-scope theorem seed list.
    /// @return An immutable dependency index for repeated candidate tests.
    /// @invariant The graph's logical edges and scope identities are preserved.
    /// @see CompressorReachability::isDerivable
    CompressorReachability::CompressorReachability(
        const CompressorNode& node, const std::vector<std::string>& theorems) {
        std::map<ExpressionWithValidity, std::size_t> indices;
        const auto indexOf = [&](const ExpressionWithValidity& expression) {
            const auto inserted = indices.emplace(expression, indices.size());
            if (inserted.second) consumers_.emplace_back();
            return inserted.first->second;
        };
        head_ = indexOf(node.head);
        for (const auto& premise : node.premises)
            initialAlive_.push_back(indexOf(premise));
        for (const auto& row : node.graph) {
            const std::size_t head = indexOf(row.first);
            for (const auto& dependencies : row.second) {
                const std::size_t derivation = derivationHeads_.size();
                derivationHeads_.push_back(head);
                dependencyCounts_.push_back(dependencies.size());
                if (dependencies.empty()) initialAlive_.push_back(head);
                for (const auto& dependency : dependencies) {
                    const std::size_t index = indexOf(dependency);
                    consumers_[index].push_back(derivation);
                }
            }
        }
        for (const auto& theorem : theorems) {
            const auto found = indices.find(ExpressionWithValidity(theorem, "main"));
            if (found != indices.end())
                theoremSeeds_.emplace_back(theorem, found->second);
        }
    }

    /// @brief Evaluate the graph's reachable expressions from available seeds.
    ///
    /// @details Each expression is enqueued once. Each dependency occurrence
    /// decrements its alternative's remaining count once; a zero count reaches
    /// that alternative's head. Early success is exact because reachability is
    /// monotone and this function produces only a Boolean result.
    /// @param deadTheorems Theorem expressions excluded from the initial seeds.
    /// @return Whether the target belongs to the least reachable fixed point.
    /// @invariant No unseeded cycle can create its own first reachable member.
    /// @see CompressorReachability::CompressorReachability
    bool CompressorReachability::isDerivable(
        const std::set<std::string>& deadTheorems) const {
        std::vector<unsigned char> alive(consumers_.size(), 0);
        std::vector<std::size_t> remaining = dependencyCounts_;
        std::vector<std::size_t> queue;
        queue.reserve(consumers_.size());
        const auto reach = [&](std::size_t expression) {
            assert(expression < alive.size());
            if (alive[expression] == 0) {
                alive[expression] = 1;
                queue.push_back(expression);
            }
        };
        for (const std::size_t seed : initialAlive_) reach(seed);
        for (const auto& seed : theoremSeeds_) {
            if (deadTheorems.count(seed.first) == 0) reach(seed.second);
        }
        if (alive[head_] != 0) return true;
        for (std::size_t cursor = 0; cursor < queue.size(); ++cursor) {
            for (const std::size_t derivation : consumers_[queue[cursor]]) {
                assert(remaining[derivation] > 0);
                if (--remaining[derivation] == 0) {
                    reach(derivationHeads_[derivation]);
                    if (alive[head_] != 0) return true;
                }
            }
        }
        return false;
    }


    // =================================================================
    // Phase 2 — Greedy Elimination with multi-pass
    // =================================================================

    /// @brief Phase 2 — greedy multi-pass redundancy elimination.
    ///
    /// @details
    /// Orders candidates by ascending usage count, then theorem text. Each
    /// candidate is tentatively unavailable as an initial theorem seed; the
    /// indexed derivability oracle then checks every extracted graph. A
    /// candidate is removed only if every original head remains derivable.
    /// Passes repeat until no further candidate is removed.
    ///
    /// Output is the surviving expanded theorem texts in lexicographic order.
    /// Compressed-out theorems are written to
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

        // Compile scope-preserving graph indices once for all candidate sets.
        std::vector<CompressorReachability> reachability;
        reachability.reserve(extracted_graphs.size());
        for (const auto& node : extracted_graphs)
            reachability.emplace_back(node, all_theorems);

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
                for (const auto& graph : reachability) {
                    if (!graph.isDerivable(current_dead)) {
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
            gl::projectRoot()
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
