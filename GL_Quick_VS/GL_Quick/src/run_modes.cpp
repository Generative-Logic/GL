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
#include <cassert>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <chrono>
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
    inline std::filesystem::path THEOREMS_FILE = THEOREMS_FOLDER / "conjectures.txt";
    inline std::filesystem::path PROVED_THEOREMS_FILE = THEOREMS_FOLDER / "theorems.txt";
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
        using FrameClock = std::chrono::steady_clock;

        const auto fullRunStarted = FrameClock::now();
        const auto frameSecondsSince = [](FrameClock::time_point started) {
            return std::chrono::duration<double>(FrameClock::now() - started).count();
        };
        const auto recordFrameTiming = [&](const char* stage,
                                           double seconds,
                                           bool excluded,
                                           int64_t count = 1) {
            assert(stage != nullptr && stage[0] != '\0');
            assert(seconds >= 0.0);
            assert(count > 0);
            std::string timingPath;
#ifdef _WIN32
            char* timingPathRaw = nullptr;
            size_t timingPathLength = 0;
            const errno_t timingEnvironmentResult = _dupenv_s(
                &timingPathRaw, &timingPathLength, "GL_FRAME_TIMING_PATH");
            assert(timingEnvironmentResult == 0);
            if (timingPathRaw == nullptr) return;
            timingPath.assign(timingPathRaw);
            std::free(timingPathRaw);
#else
            const char* timingPathRaw = std::getenv("GL_FRAME_TIMING_PATH");
            if (timingPathRaw == nullptr) return;
            timingPath.assign(timingPathRaw);
#endif
            std::ofstream timing(timingPath, std::ios::app);
            assert(timing.is_open());
            timing << "{\"batch\":\"" << anchor_id
                   << "\",\"count\":" << count
                   << ",\"excluded\":" << (excluded ? "true" : "false")
                   << ",\"parent\":\"native." << anchor_id
                   << "\",\"seconds\":" << std::setprecision(12) << seconds
                   << ",\"stage\":\"" << stage << "\"}\n";
            timing.flush();
            assert(timing.good());
        };

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
                        THEOREMS_FILE = THEOREMS_FOLDER / "conjectures.txt";
                        PROVED_THEOREMS_FILE = THEOREMS_FOLDER / "theorems.txt";
                        COMPRESSED_EXTERNAL_THEOREMS_FILE = THEOREMS_FOLDER / "compressed_external_theorems.txt";
                    }
                    if (j.contains("raw_proof_graph_folder")) {
                        std::string folder = j["raw_proof_graph_folder"];
                        RAW_PROOF_DIR = PROJECT_ROOT / folder;
                    }
                    if (j.contains("background_theorems_folder")) {
                        std::string bgFolder = j["background_theorems_folder"];
                        BACKGROUND_PROVED_THEOREMS_FILE = PROJECT_ROOT / bgFolder / "theorems.txt";
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

        recordFrameTiming(
            "native.setup", frameSecondsSince(fullRunStarted), false);

        const auto proverStarted = FrameClock::now();
        expressionAnalyzer.analyzeExpressions(tmp_lst, proved_lst, external_lst);
        recordFrameTiming(
            "native.prover", frameSecondsSince(proverStarted), true);

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
        const auto compressorStarted = FrameClock::now();
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
        recordFrameTiming(
            "native.compressor", frameSecondsSince(compressorStarted), true);

        auto saveStarted = FrameClock::now();

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
            recordFrameTiming(
                "native.save_orchestration", frameSecondsSince(saveStarted), false);
            const auto rawProofStarted = FrameClock::now();
            expressionAnalyzer.generateRawProofGraph(expressionAnalyzer.globalTheoremList, RAW_PROOF_DIR);
            // Grid's last reader done — wipe it so the root's arena-backed
            // encodedMaps don't outlive their arena at process teardown.
            expressionAnalyzer.destroyGrid();
            recordFrameTiming(
                "native.raw_proof", frameSecondsSince(rawProofStarted), false);
        } else {
            // ====== PHASE 2: SAVE ======
            // globalTheoremList was compressed by the run_modes.cpp invocation above.
            // Handle external theorem pruning and save theorems.txt.

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
            // rewrites both theorems.txt and compiled_theorems.txt.
            expressionAnalyzer.saveProvedTheoremsFiltered(survivingTheorems, survivingExternalSet);

            // Append OR theorems to both proved_theorems files.
            // theorems.txt: expanded form for inter-batch communication.
            // compiled_theorems.txt: compiled form for proof graph pruning.
            if (!orTheorems.empty()) {
                std::ofstream ofs(PROVED_THEOREMS_FILE, std::ios::app);
                std::ofstream ofsCompiled(THEOREMS_FOLDER / "compiled_theorems.txt", std::ios::app);
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
            recordFrameTiming(
                "native.save_orchestration", frameSecondsSince(saveStarted), false);
            if (!skipProofGraph) {
                const auto rawProofStarted = FrameClock::now();
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
                // Grid's last reader done — wipe it so the root's arena-backed
                // encodedMaps don't outlive their arena at process teardown.
                expressionAnalyzer.destroyGrid();
                recordFrameTiming(
                    "native.raw_proof", frameSecondsSince(rawProofStarted), false);
            }
        }

        const auto telemetryStarted = FrameClock::now();

        // `mailArena` is owned exclusively by MailLog. Its blocks are retained
        // until analyzer teardown, so the end-of-batch held count is also the
        // log's physical high-water, including spilled arena directories.
        const int64_t mailLogPeakBlocks =
            expressionAnalyzer.mailArena.blocksHeld();
        const int64_t mailPeakBlocks = gl::mailMemory().peakBlocksInUse();
        const int64_t mailBlockBytes =
            expressionAnalyzer.parameters.static_mail_block_bytes;
        const int64_t mailInternerPeakBlocks =
            gl::mailInterner().arenaBlocksHeld();
        const int64_t routingMailInPeakBlocks =
            expressionAnalyzer.peakRoutingMailInBlocks.load(
                std::memory_order_relaxed);
        const int64_t deloadableMailOutPeakBytes =
            expressionAnalyzer.peakDeloadableMailOutBytes;
        assert(expressionAnalyzer.routingMailInBlocksInFlight.load(
                   std::memory_order_relaxed) == 0
            && "batch ended with routing mailIn attribution still in flight");
        const int64_t mailEndBlocks = gl::mailMemory().blocksInUse();
        assert(mailEndBlocks >= mailLogPeakBlocks + mailInternerPeakBlocks);
        const int64_t mailEndUnattributedBlocks =
            mailEndBlocks - mailLogPeakBlocks - mailInternerPeakBlocks;
        std::cout << "[mail-memory] dormant_lbs="
                  << expressionAnalyzer.dormantLogicBlocksAtGridBuild
                  << " mode="
                  << (expressionAnalyzer.rollingMailHistoryEnabled
                          ? "rolling" : "full")
                  << " total_committed_history_bytes="
                  << expressionAnalyzer.mailLog.totalCommittedHistoryBytes
                  << " peak_retained_history_bytes="
                  << expressionAnalyzer.mailLog.peakRetainedHistoryBytes
                  << " mail_log_peak_blocks=" << mailLogPeakBlocks
                  << " mail_log_peak_bytes="
                  << mailLogPeakBlocks * mailBlockBytes
                  << " mail_interner_peak_blocks="
                  << mailInternerPeakBlocks
                  << " mail_interner_peak_bytes="
                  << mailInternerPeakBlocks * mailBlockBytes
                  << " routing_mail_in_peak_blocks="
                  << routingMailInPeakBlocks
                  << " routing_mail_in_peak_bytes="
                  << routingMailInPeakBlocks * mailBlockBytes
                  << " deloadable_mail_out_peak_bytes="
                  << deloadableMailOutPeakBytes
                  << " peak_pool_blocks=" << mailPeakBlocks
                  << " block_bytes=" << mailBlockBytes
                  << " peak_pool_bytes=" << mailPeakBlocks * mailBlockBytes
                  << " end_pool_blocks=" << mailEndBlocks
                  << " end_unattributed_blocks="
                  << mailEndUnattributedBlocks
                  << std::endl;

        // One executable invocation processes one batch, so each manager's
        // lifetime high-water is this batch's actual physical pool peak. Keep
        // every pool separate: their peaks need not occur simultaneously.
        const int64_t mainPeakBlocks = gl::staticMemory().peakBlocksInUse();
        const int64_t mainBlockBytes = gl::staticMemory().blockBytes();
        const int64_t persistentPeakBlocks =
            gl::persistentMemory().peakBlocksInUse();
        const int64_t persistentBlockBytes =
            gl::persistentMemory().blockBytes();
        const int64_t lbPeakBlocks = gl::lbMemory().peakBlocksInUse();
        const int64_t lbBlockBytes = gl::lbMemory().blockBytes();
        std::cout << "[pool-memory]"
                  << " main_peak_blocks=" << mainPeakBlocks
                  << " main_block_bytes=" << mainBlockBytes
                  << " main_peak_bytes=" << mainPeakBlocks * mainBlockBytes
                  << " main_capacity_bytes="
                  << gl::staticMemory().totalBlocks() * mainBlockBytes
                  << " persistent_peak_blocks=" << persistentPeakBlocks
                  << " persistent_block_bytes=" << persistentBlockBytes
                  << " persistent_peak_bytes="
                  << persistentPeakBlocks * persistentBlockBytes
                  << " persistent_capacity_bytes="
                  << gl::persistentMemory().totalBlocks()
                        * persistentBlockBytes
                  << " mail_peak_blocks=" << mailPeakBlocks
                  << " mail_block_bytes=" << mailBlockBytes
                  << " mail_peak_bytes=" << mailPeakBlocks * mailBlockBytes
                  << " mail_capacity_bytes="
                  << gl::mailMemory().totalBlocks() * mailBlockBytes
                  << " lb_peak_blocks=" << lbPeakBlocks
                  << " lb_block_bytes=" << lbBlockBytes
                  << " lb_peak_bytes=" << lbPeakBlocks * lbBlockBytes
                  << " lb_capacity_bytes="
                  << gl::lbMemory().totalBlocks() * lbBlockBytes
                  << std::endl;

        std::string memoryLogPath;
#ifdef _WIN32
        char* memoryLogPathRaw = nullptr;
        size_t memoryLogPathLength = 0;
        const errno_t environmentResult = _dupenv_s(
            &memoryLogPathRaw, &memoryLogPathLength, "GL_MEMORY_LOG_PATH");
        assert(environmentResult == 0);
        assert(memoryLogPathRaw != nullptr && memoryLogPathLength > 1);
        memoryLogPath.assign(memoryLogPathRaw);
        std::free(memoryLogPathRaw);
#else
        const char* memoryLogPathRaw = std::getenv("GL_MEMORY_LOG_PATH");
        assert(memoryLogPathRaw != nullptr && memoryLogPathRaw[0] != '\0');
        memoryLogPath.assign(memoryLogPathRaw);
#endif
        std::ofstream memoryLog(memoryLogPath, std::ios::app);
        assert(memoryLog.is_open());
        memoryLog << "\n## " << anchor_id << "\n\n"
                  << "| Pool | Block KiB | Peak blocks | Peak MiB | "
                     "Reservation MiB | Headroom MiB | Used |\n"
                  << "|---|---:|---:|---:|---:|---:|---:|\n";
        const auto writePoolRow = [&](const char* poolName,
                                      int64_t peakBlocks,
                                      int64_t blockBytes,
                                      int64_t capacityBytes) {
            const int64_t peakBytes = peakBlocks * blockBytes;
            assert(peakBytes <= capacityBytes);
            constexpr double bytesPerMiB = 1024.0 * 1024.0;
            memoryLog << "| " << poolName
                      << " | " << std::fixed << std::setprecision(3)
                      << blockBytes / 1024.0
                      << " | " << peakBlocks
                      << " | " << peakBytes / bytesPerMiB
                      << " | " << capacityBytes / bytesPerMiB
                      << " | " << (capacityBytes - peakBytes) / bytesPerMiB
                      << " | " << 100.0 * peakBytes / capacityBytes
                      << "% |\n";
        };
        writePoolRow("Main", mainPeakBlocks, mainBlockBytes,
                     gl::staticMemory().totalBlocks() * mainBlockBytes);
        writePoolRow("Persistent", persistentPeakBlocks,
                     persistentBlockBytes,
                     gl::persistentMemory().totalBlocks()
                         * persistentBlockBytes);
        writePoolRow("Mail", mailPeakBlocks, mailBlockBytes,
                     gl::mailMemory().totalBlocks() * mailBlockBytes);
        writePoolRow("LB-body", lbPeakBlocks, lbBlockBytes,
                     gl::lbMemory().totalBlocks() * lbBlockBytes);
        memoryLog << "\n### Mail-pool attribution\n\n"
                  << "Component peaks are independent and must not be summed. "
                     "The end-state rows are simultaneous.\n\n"
                  << "| Component | Blocks | MiB | Meaning |\n"
                  << "|---|---:|---:|---|\n";
        const auto writeMailAttributionRow = [&](const char* component,
                                                 int64_t blocks,
                                                 const char* meaning) {
            memoryLog << "| " << component
                      << " | " << blocks
                      << " | " << std::fixed << std::setprecision(3)
                      << blocks * mailBlockBytes / (1024.0 * 1024.0)
                      << " | " << meaning << " |\n";
        };
        writeMailAttributionRow("MailLog exclusive arena peak",
                                mailLogPeakBlocks,
                                "Retained blobs, references, routing, cursors");
        writeMailAttributionRow("Global mail interner arena peak",
                                mailInternerPeakBlocks,
                                "Largest retained global-id dictionary arena");
        writeMailAttributionRow("Routing mailIn simultaneous peak",
                                routingMailInPeakBlocks,
                                "All phase-1 inboxes concurrently in flight");
        memoryLog << "| Deloadable mailOut live-byte peak"
                  << " | main-pool shared"
                  << " | " << std::fixed << std::setprecision(3)
                  << deloadableMailOutPeakBytes / (1024.0 * 1024.0)
                  << " | Exact logical bytes across all per-LB output mailboxes and private interners; physical blocks are in Main static peak |\n";
        writeMailAttributionRow("Whole mail pool physical peak",
                                mailPeakBlocks,
                                "GlobalMemoryManager lifetime high-water");
        writeMailAttributionRow("End-of-batch pool in use",
                                mailEndBlocks,
                                "Simultaneous retained state after grid teardown");
        writeMailAttributionRow("End-of-batch unattributed",
                                mailEndUnattributedBlocks,
                                "End pool minus MailLog and global interner arenas");
        memoryLog.flush();
        assert(memoryLog.good());
        recordFrameTiming(
            "native.telemetry", frameSecondsSince(telemetryStarted), false);
    }

    void ceOnlyRun(const std::string& anchor_id) {
        using namespace std;
        namespace fs = std::filesystem;

        if (!anchor_id.empty()) {
            std::cout << "\n[ceOnlyRun] Processing Tag/Anchor: " << anchor_id << "\n";
        }

        // Same config sniff as `fullRun` to honour `theorems_folder` overrides
        // (incubator vs main tags). Only paths matter for the CE-filter step;
        // skipCompression / skipProofGraph are irrelevant here, but the
        // theorems folder override IS — without it an `IncubatorPeano` tag
        // would read main's `files/theorems/conjectures.txt` and write its
        // filtered_conjectures.txt to main's folder.
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
                        THEOREMS_FILE = THEOREMS_FOLDER / "conjectures.txt";
                        PROVED_THEOREMS_FILE = THEOREMS_FOLDER / "theorems.txt";
                        COMPRESSED_EXTERNAL_THEOREMS_FILE = THEOREMS_FOLDER / "compressed_external_theorems.txt";
                    }
                } catch (...) {}
            }
        }

        gl::ExpressionAnalyzer expressionAnalyzer(anchor_id);

        if (!fs::exists(THEOREMS_FILE)) {
            std::cerr << "[ceOnlyRun] Missing theorems file: " << THEOREMS_FILE << "\n";
            std::cerr << "[ceOnlyRun] Run `--conjecture " << anchor_id
                      << "` first to generate it." << std::endl;
            return;
        }

        std::unordered_set<std::string> theorem_set = loadLinesFromFile(THEOREMS_FILE);
        std::vector<std::string> tmp_lst(theorem_set.begin(), theorem_set.end());
        std::sort(tmp_lst.begin(), tmp_lst.end());
        std::cout << "[ceOnlyRun] Loaded " << tmp_lst.size()
                  << " conjectures from " << THEOREMS_FILE << "\n";

        std::vector<std::string> survivors = expressionAnalyzer.runCeFilterOnly(tmp_lst);
        std::cout << "[ceOnlyRun] Final survivors: " << survivors.size() << "\n";
    }

} // namespace run_modes
