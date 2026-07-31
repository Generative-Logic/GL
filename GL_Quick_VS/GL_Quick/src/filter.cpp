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

// Counterexample (CE) filter — translation unit.
//
// Hosts the bodies of the CE-specific member functions of ExpressionAnalyzer
// that were extracted from prover.cpp. The class declaration itself remains in
// prover.hpp because C++ requires a single class definition; only the function
// bodies live here.

#include "filter.hpp"
#include "prover.hpp"
#include "parameters.hpp"
#include "infra/rt_tracker.hpp"
#include "memory_infra/arena_stack.hpp"
#include "memory_infra/scratch_arena.hpp"
#include "memory_infra/str_ops.hpp"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

namespace gl {

    // ------------------------------------------------------------------
    // CE-filter file IO.
    // ------------------------------------------------------------------

    /// @brief Load the per-anchor *simple facts* — small ground-fact tables
    /// the CE filter probes each conjecture against — from the `files/simple_facts/`
    /// directory.
    ///
    /// @details
    /// Resolves the anchor's short name (strips the `"Anchor"` prefix and
    /// lowercases — `"AnchorPeano"` → `"peano"`, `"AnchorGauss"` → `"gauss"`),
    /// then walks `files/simple_facts/` for files matching
    /// `simple_facts_<name>(_<n>)?\.txt` (case-insensitive). Hits are sorted
    /// by the optional integer `n` ascending so j-copies appear in
    /// `simple_facts_<name>_5.txt`, `simple_facts_<name>_6.txt`, … order.
    /// Each surviving file is loaded line-by-line; trailing `\r` from Windows
    /// line endings is stripped. The result is a vector-of-vectors: one inner
    /// vector per file, in sorted order.
    ///
    /// j-copy ordering matters because the CE filter generates encoded
    /// requests against each fact-list separately and the prover's
    /// determinism depends on stable iteration order across runs.
    ///
    /// @return Vector of fact-lists, sorted by their numeric suffix.
    /// @pre  `this->anchorInfo.name` has already been initialized (it is set
    ///       in the `ExpressionAnalyzer` constructor by `initAnchor`).
    /// @post Files are read once; no caching — each call re-walks the
    ///       directory. The CE filter calls this once per batch.
    /// @see `docs/agentic_swdd/10_pipeline/03_ce_filter.md` — fact-base shape rationale.
    std::vector<std::vector<std::string>> ExpressionAnalyzer::readSimpleFacts() const {
        // 1) Derive actual name from anchor: strip leading "Anchor", lowercase everything.
        std::string anchorName = this->anchorInfo.name;  // e.g., "AnchorPeano", "AnchorGauss"
        const std::string prefix = "Anchor";
        if (anchorName.rfind(prefix, 0) == 0) { // starts with "Anchor"
            anchorName.erase(0, prefix.size());
        }
        else {
            // Best effort: remove first occurrence if it's not a strict prefix.
            std::size_t pos = anchorName.find(prefix);
            if (pos != std::string::npos) anchorName.erase(pos, prefix.size());
        }
        std::string actualName;
        actualName.reserve(anchorName.size());
        for (unsigned char ch : anchorName) actualName.push_back(static_cast<char>(std::tolower(ch)));

        // 2) Resolve directory: <repo>/files/simple_facts
        const auto simpleFactsDir =
            std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path()
            / "files" / "simple_facts";

        std::vector<std::pair<int, std::filesystem::path>> hits;   // (n, path)

        // 3) Filename pattern: simple_facts_<actual_name>.txt  or  simple_facts_<actual_name>_<n>.txt
        //    We match case-insensitively for the <actual_name> part and parse the optional trailing integer.
        const std::regex rx(R"(^(?:simple_facts)_([A-Za-z]+?)(?:_(\d+))?\.txt$)",
            std::regex::ECMAScript | std::regex::icase);

        if (std::filesystem::exists(simpleFactsDir) && std::filesystem::is_directory(simpleFactsDir)) {
            for (const auto& dirent : std::filesystem::directory_iterator(simpleFactsDir)) {
                if (!dirent.is_regular_file()) continue;

                const std::string fname = dirent.path().filename().string();
                std::smatch m;
                if (!std::regex_match(fname, m, rx)) continue;

                // m[1] -> actual_name candidate, m[2] -> n (optional)
                std::string candidateName = m[1].str();
                for (auto& c : candidateName) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                if (candidateName != actualName) continue;

                int n = m[2].matched ? std::stoi(m[2].str()) : 0;
                hits.emplace_back(n, dirent.path());
            }
        }

        // 4) Sort by n ascending (5 before 6, etc.)
        std::sort(hits.begin(), hits.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });

        // 5) Read files: one file -> one list<string> (one line == one element).
        //    Preserve line order; strip trailing '\r' if present (Windows line endings).
        std::vector<std::vector<std::string>> out;
        out.reserve(hits.size());

        for (const auto& [n, path] : hits) {
            std::ifstream in(path);
            std::vector<std::string> lines;
            if (in) {
                std::string line;
                while (std::getline(in, line)) {
                    if (!line.empty() && line.back() == '\r') line.pop_back();
                    lines.push_back(std::move(line));
                }
            }
            out.push_back(std::move(lines));
        }

        return out;
    }

    /// @brief Write the post-CE-filter conjecture list to
    /// `files/theorems/filtered_conjectures.txt`.
    ///
    /// @details
    /// Called once per batch after the CE filter has finished probing every
    /// conjecture. Survivors (those that did NOT produce a confirmed
    /// contradiction in the CE pass, or that the CE filter is configured to
    /// pass through verbatim) are persisted to disk so the main prover stage
    /// can read them as its theorem list. The file is overwritten on each
    /// call.
    ///
    /// @param filteredConjectures The full list of conjectures to persist,
    ///                            one per output line.
    /// @post `files/theorems/filtered_conjectures.txt` exists and contains
    ///       `filteredConjectures.size()` lines of MPL text.
    /// @see `prover.hpp::CEFilter` — primary caller.
    /// @see `docs/agentic_swdd/10_pipeline/03_ce_filter.md` — pipeline stage chapter.
    void ExpressionAnalyzer::saveFilteredConjectures(const std::vector<std::string>& filteredConjectures) {
        namespace fs = std::filesystem;

        const auto dir =
            fs::path(__FILE__).parent_path().parent_path().parent_path().parent_path()
            / "files" / "theorems";
        const auto out = dir / "filtered_conjectures.txt";

        std::error_code ec;
        fs::create_directories(dir, ec);
        if (ec) throw std::runtime_error("Failed to create directory: " + dir.string());

        std::ofstream ofs(out);
        if (!ofs) throw std::runtime_error("Failed to open output file: " + out.string());

        for (const auto& line : filteredConjectures) ofs << line << '\n';
        ofs.flush();
        if (!ofs) throw std::runtime_error("Failed while writing: " + out.string());
    }

    // ------------------------------------------------------------------
    // CE-filter batch lifecycle.
    // ------------------------------------------------------------------

    /// @brief Seed the CE-filter LB tree with the simple-fact base — every
    /// fact is installed as either a statement or a hash-engine rule,
    /// depending on its shape.
    ///
    /// @details
    /// One CE-filter LB is created per conjecture batch slot
    /// (`0..batchSize-1`) under `ceBody` in `ceSimpleMapStore`. For each slot, every
    /// entry of `simpleFacts` is dispatched through the generic
    /// `addExprToMemoryBlock` so atomic facts become statements in
    /// `Memory::intEncodedStatements` while implication-shaped facts become
    /// hash rules in `Memory::overallHashMemory`. The polymorphism is what
    /// makes the CE filter setup tractable across batches with very
    /// different fact-base shapes (Peano = mostly atomic; Gauss = mix of
    /// atomic + implication).
    ///
    /// Per I-1, this function is called AFTER `precompileStructuralOperators`
    /// for the conjecture list, so any structural operator (`or0`,
    /// `existence2`, etc.) has already been disintegrated to base form
    /// before the simple facts hit the admission map.
    ///
    /// @param simpleFacts  Flat list of fact texts. The CE-filter loop
    ///                     reuses the same fact set across every batch
    ///                     slot.
    /// @param batchSize    Number of CE-filter slots to seed. One LB is
    ///                     installed under `ceBody` at routing key
    ///                     `std::to_string(i)` in `ceSimpleMapStore`, for each
    ///                     `i` in `[0, batchSize)`.
    /// @pre  `precompileStructuralOperators` has run for the batch's
    ///       theorem list (I-1).
    /// @post Each slot LB's `intEncodedStatements`
    ///       and `…->overallHashMemory` carry the simple-fact base, ready
    ///       for the per-slot conjecture probe.
    /// @see `addExprToMemoryBlock` — installer.
    void ExpressionAnalyzer::loadFactsForCEFiltering(
        std::vector<std::string> simpleFacts,
        int batchSize) {

        for (std::size_t start = 0; start < batchSize; start++) {
            Memory* lb0 = nullptr;
            {
                Memory* existingSlot = ceSimpleMapStore.findChild(&ceBody, std::to_string(start));
                if (existingSlot) {
                    lb0 = existingSlot;
                }
                else {
                    lb0 = lbStore.create<Memory>();
                    ceSimpleMapStore.linkChild(&ceBody, std::to_string(start), lb0);
                    lb0->parentMemory = &ceBody;
                    lb0->level = 0;
                    lb0->setExprKey(std::to_string(start));
                    lb0->contradictionIndex = -1;
                    permanentBodiesCE.push_back(lb0);
                }
            }

            // Origins and levels
            TransientOrigin origin{};
            if (parameters.trackHistory) {
                origin = TransientOrigin{ true, OriginTag::ceBuildingBlock, nullptr, 0 };
            }
            const int lvl0[1] = { 0 };

            // LB0: +, *, s
            for (const auto& s : simpleFacts) this->addExprToMemoryBlock(s, *lb0, 0, 4, lvl0, 1, origin, -1, -1, StrSpan("main", 4), false);
        }
    }

    /// @brief Install one conjecture into the CE-filter LB so the prover can
    /// probe whether its negation contradicts the simple-fact base.
    ///
    /// @details
    /// Wraps the conjecture in its negated form (the contradiction probe
    /// looks for `!conjecture` becoming derivable from the simple facts;
    /// success means the conjecture is true on this fact base) and submits
    /// it via `addExprToMemoryBlock`. The result feeds into the per-row
    /// `ContradictionItem` of the CE-filter contradiction table.
    ///
    /// Called once per conjecture per CE-filter pass; the per-batch budget
    /// is `parameters.numberIterationsConjectureFiltering` outer iterations.
    ///
    /// @param conjecture        Raw MPL conjecture text.
    /// @param body              CE-filter LB (already seeded with simple
    ///                          facts via `loadFactsForCEFiltering`).
    /// @param coreId            Logical-core id (CE filter uses 0 in single
    ///                          core mode).
    /// @param conjectureIndex   Row index into `contradictionTable`.
    /// @post `body` carries the negated conjecture as a pending obligation;
    ///       on hash-burst, the prover may set
    ///       `contradictionTable[conjectureIndex].successful = true`.
    void ExpressionAnalyzer::addConjectureForCEFiltering(const std::string& conjecture,
        Memory *mb,
        int cIndex) {

        mb->contradictionIndex = cIndex;

        // Store conjecture in the LB0 hash memory
        const std::map<std::string, std::string> replacementMap{
        {"1","N"},
        {"2","i0"},
        {"3","s"},
        {"4","+"},
        {"5","*"},
        {"6","i1"},
        {"7","i2"},
        {"8","id"}
        };
        std::string replacedConjecture = ce::replaceKeysInString(conjecture, replacementMap);


        // 1) Disintegrate implication -> chain (left nodes) + head (rightmost)
        std::vector< std::tuple<
            std::string,                  // left expression
            std::vector<std::string>,     // node args
            std::set<std::string>         // left-node arguments
        > > tempChain;
        const std::string head = ce::disintegrateImplication(replacedConjecture, tempChain, this->coreExpressionMap);


        std::vector<std::string> chain;
        chain.reserve(tempChain.size());
        for (std::size_t i = 0; i < tempChain.size(); ++i) chain.push_back(std::get<0>(tempChain[i]));

        const int lvl0[1] = { 0 };

        StrSpan chainRun[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
        int32_t chainRunN = 0;
        for (const std::string& s : chain) {
            assert(chainRunN < ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS
                && "addToHashMemory chain run exceeds cap");
            chainRun[chainRunN++] = StrSpan(s);
        }
        this->addToHashMemory(chainRun, chainRunN, StrSpan(head), nullptr, 0,
            *mb, mb->overallHashMemory, lvl0, 1,
            StrSpan(replacedConjecture),
            parameters.maxIterationNumberVariable,
            parameters.standardMaxSecondaryNumber,
            false,
            parameters.minNumOperatorsKeyCE,
            StrSpan("implication", 11), true, StrSpan(replacedConjecture), StrSpan("main", 4));
    }

    /// @brief Tear down the CE-filter batch state — clear the simple-fact
    /// LB, free per-batch arenas, reset the contradiction table.
    ///
    /// @details
    /// Called once per batch after the CE filter has finished probing every
    /// conjecture and `saveFilteredConjectures` has persisted the survivors.
    /// Drops the simple-fact LB tree
    /// (so the main prover stage starts from a clean root), and clears the
    /// per-row `ContradictionItem` records so re-running on a fresh batch
    /// is well-defined.
    ///
    /// @post All CE-filter-only state is released; subsequent calls to the
    ///       main prover stage operate on a fresh LB tree.
    void ExpressionAnalyzer::releaseCEBatchMemory() {
        // ---- 0) Defensive: clear tiny per-iteration queues (cheap, may carry strings) ----
        inductionMemoryBlocks.clear();
        std::vector<Memory*>().swap(inductionMemoryBlocks);

        // (pendingAncestorOrigins reset retired with D-51 — see prover.hpp comment.)

        updateGlobalTuples.clear();
        std::vector<std::tuple<int, bool, int>>().swap(updateGlobalTuples);

        // updateGlobalDirectPages (std::optional<SealedPageSet>) is emplaced by
        // proveKernel and reset by drainUpdateGlobalDirect — nullopt between
        // proveKernel calls, so nothing to release here (the deferredAncestorPages
        // lifecycle).

        // ---- 1) Collect ALL potential CE roots from every holder ----
        std::vector<Memory*> roots;
        roots.reserve(permanentBodiesCE.size());

        // from ceBody root map
        ceSimpleMapStore.forEachChild(&ceBody, [&](const gl::StrSpan&, Memory* child) {
            if (child) roots.push_back(child);
        });
        // from permanentBodiesCE
        for (std::size_t i = 0; i < permanentBodiesCE.size(); ++i)
            if (permanentBodiesCE[i]) roots.push_back(permanentBodiesCE[i]);

        // ---- 2) Iterative DFS to delete each node exactly once ----
        std::unordered_set<Memory*> seen;
        std::vector<Memory*> stack;
        stack.reserve(roots.size());
        for (Memory* r : roots) if (r) stack.push_back(r);

        while (!stack.empty()) {
            Memory* node = stack.back(); stack.pop_back();
            if (!node || !seen.insert(node).second) continue;

            ceSimpleMapStore.forEachChild(node, [&](const gl::StrSpan&, Memory* child) {
                if (child) stack.push_back(child);
            });

            lbStore.destroy(node); // d-tor frees STL members, slot returns to the store
        }

        // ---- 3) Drop/compact all CE containers & maps (release capacity) ----
        // ceBody’s root map
        ceSimpleMapStore.clear();

        // explicit reset of ceBody's heavy members via swap-with-empty
        {
            Memory empty;
            using std::swap;
            swap(ceBody.startInt, empty.startInt);
            // Statified registries: reset to fresh (pages back to the LB
            // queue + table capacity dropped — the swap-with-empty idiom's
            // successor for the cold set maps and the statement vector).
            ceBody.intToBeProved.resetToFresh();
            ceBody.intEncodedStatements.release();
            ceBody.intStatementLevelsMap.resetToFresh();
            ceBody.setExprKey("");
            ceBody.parentMemory = nullptr;
            swap(ceBody.level, empty.level);
            ceBody.overallHashMemory.resetToFresh();
            ceBody.equivalenceClassesMap.resetToFresh();
            ceBody.changedClassesThisStep.release();
            // Statified: pages back to the LB queue (resetToFresh
            // successor, same as the registry above).
            ceBody.intLocalEncodedStatements.release();
            ceBody.intLocalEncodedStatementsDelta.release();
            // Mail containers are non-copyable. mailIn returns its arena blocks
            // to the mail pool; clearMailOut resets the deloadable columns,
            // private interner, and always-resident pending bit together.
            ceBody.mailIn.clear();
            ceBody.clearMailOut();
            // Registered-membership reset of the packed statement registry:
            // the `registered` membership empties, `known` rows stay
            // untouched (this teardown never reset the level-registry
            // record).
            {
                // Keep only `known` rows, with `registered` cleared; drop the
                // rest. The cold map has no value-aware erase, so snapshot the
                // survivors, reset, and re-insert in id order.
                std::vector<gl::StatementKey> keepKeys;
                std::vector<gl::StatementFlags> keepVals;
                for (int32_t i = 1; i <= ceBody.intKnownStatements.count(); ++i) {
                    gl::StatementFlags f = ceBody.intKnownStatements.valueAt(i);
                    if (!f.known) continue;
                    f.registered = false;
                    keepKeys.push_back(ceBody.intKnownStatements.decodeKey(i));
                    keepVals.push_back(f);
                }
                ceBody.intKnownStatements.resetToFresh();
                for (std::size_t i = 0; i < keepKeys.size(); ++i)
                    ceBody.intKnownStatements.insert(keepKeys[i], keepVals[i]);
            }
            ceBody.eqClassSttmntIndexMapMap.resetToFresh();
            swap(ceBody.isActive, empty.isActive);
            swap(ceBody.isPartOfRecursion, empty.isPartOfRecursion);
            swap(ceBody.deltaNumberStatements, empty.deltaNumberStatements);
            ceBody.exprOriginMap.resetToFresh();   // cold blob map (I-121)
            swap(ceBody.recursionCounter, empty.recursionCounter);
            swap(ceBody.contradictionIndex, empty.contradictionIndex);
        }

        // CE containers
        std::vector<Memory*>().swap(permanentBodiesCE);

    }

    // ------------------------------------------------------------------
    // CE-filter top-level orchestrator (moved from prover.cpp in commit E).
    // ------------------------------------------------------------------

    std::vector<std::string> ExpressionAnalyzer::filterConjecturesWithCE(
        const std::vector<std::string>& conjectures,
        const std::vector<std::string>& simpleFacts)
    {
        ceFilteringActive = true;

        // One contradictionTable slot per conjecture. Each worker writes only its
        // own (disjoint) index, so the post-pool survivor reads are race-free.
        contradictionTable.clear();
        contradictionTable.reserve(conjectures.size());
        for (int i = 0; i < static_cast<int>(conjectures.size()); ++i)
            contradictionTable.push_back(ContradictionItem(conjectures[i], false));

        // Load the fact base ONCE into a single template LB (status-4 statements;
        // empty hashmap — see Memory::cloneFactsTemplate). Each conjecture runs on
        // its own throwaway clone of this template.
        loadFactsForCEFiltering(simpleFacts, /*batchSize=*/1);
        const Memory* templateLB = ceSimpleMapStore.findChild(&ceBody, std::to_string(0));

        // Pre-intern every CE LB's exprKey ("0".."N-1") into the process-wide
        // skeletonInterner() SINGLE-THREADED, before the worker pool spawns. The
        // skeletonInterner is "single-threaded write side only" (I-83): the
        // workers below each call lb->setExprKey(std::to_string(i)), and a
        // concurrent mint() from the parallel pool races on its cold key store +
        // hash index, tripping the "minted key not findable at its id" desync
        // assert (I-82). After this loop every key already exists, so the
        // workers' setExprKey calls take the lookup-only (pure read) path, which
        // is safe to run concurrently on the now-unchanging table.
        for (std::size_t i = 0; i < conjectures.size(); ++i)
            (void)skeletonInterner().intern(std::to_string(i));

        // Dedicated CE thread pool over a global conjecture work queue. A worker
        // grabs the next conjecture, clones the facts template, installs the
        // conjecture's rule, runs that conjecture's CE check to completion on its
        // own single-owner LB (no batch barrier), records the result, throws the
        // clone away, and grabs the next. Each CE LB is unshared, so its burst may
        // write to it freely and stop the instant its contradiction fires.
        const unsigned workers = std::max(1u, logicalCores);
        // The CE filter runs exactly one hashburst per conjecture.
        assert(parameters.numberIterationsConjectureFiltering == 1
            && "CE filter does one hashburst per conjecture");
        std::atomic<std::size_t> next{ 0 };
        auto worker = [this, &conjectures, templateLB, &next](unsigned coreId) {
            // Claim this worker's scratch slot for the WHOLE task. performElemPhase1
            // publishes g_currentCoreId for the burst, but addConjectureForCEFiltering
            // below already reaches the per-slot scratch arenas (via addToHashMemory
            // -> makeNormalizedKeysForAdmission -> insertRemainingArgsNormKey, and
            // disintegrateExpr2). Without this, the first conjecture on each worker
            // runs with g_currentCoreId == -1, so every worker falls back to the
            // shared slot slotCount()-1 and they race on one scratch arena's byte
            // cursor (I-83 per-slot isolation; G-56, the same class the phase-2
            // executor worker hit).
            g_currentCoreId = static_cast<int>(coreId);
            // The request-expr copies + keys ride the per-slot gen scratch arena
            // inside performElem2 (released per task at its exit) — no per-thread
            // TypedArena here.
            for (;;) {
                const std::size_t i = next.fetch_add(1, std::memory_order_relaxed);
                if (i >= conjectures.size()) break;

                Memory* lb = templateLB->cloneFactsTemplate(lbStore);
                lb->parentMemory = &ceBody;
                lb->setExprKey(std::to_string(i));
                this->addConjectureForCEFiltering(conjectures[i], lb,
                                                  static_cast<int>(i));

                // One hashburst per conjecture: the burst checks whether the
                // conjecture's negation immediately contradicts the fact base.
                // burstDeactivates stops the burst the instant a refuting head
                // fires; phase 3's dischargeContradiction records the refutation
                // into contradictionTable and deactivates the LB.
                this->performElemPhase1(*lb, coreId);
                // Per-conjecture sealed pages (records + strings on one set):
                // this worker is both producer
                // and consumer (sequential phases), so the handoff degrades
                // to seal-after-burst, free-after-apply.
                SealedPageSet sealedPages;
                sealedPages.bind(&staticMemory());
                std::atomic<bool> stop{ false };
                this->performElem2(
                    *lb, coreId, /*processID=*/0, /*splitCount=*/1,
                    /*partCount=*/1, SplitStumpRef{}, sealedPages, stop);
                sealedPages.seal();
                // The CE LB is single-use (deleted below) and runs UNSPLIT
                // (splitCount=1), so no adaptive split decision applies (that lives
                // in proveKernel's finalize, which the CE filter never enters).
                // performElemPhase2 just applies this burst's firing records —
                // one part set, consumed while Sealed; the staging vectors are
                // cleared inside phase 2, before the free below.
                SealedPageSet* onePart[1] = { &sealedPages };
                this->performElemPhase2(*lb, onePart, 1);
                this->performElemPhase3(*lb, coreId);
                sealedPages.freePages();

                lbStore.destroy(lb);              // single-use LB, thrown away
            }
        };

        const auto ceStart = std::chrono::high_resolution_clock::now();
        std::vector<std::thread> pool;
        pool.reserve(workers);
        for (unsigned t = 0; t < workers; ++t) pool.emplace_back(worker, t);
        for (auto& th : pool) th.join();
        const auto ceElapsed =
            std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - ceStart).count();
        std::cout << "CE filter: " << conjectures.size() << " conjectures, "
                  << workers << " workers, " << ceElapsed << "s" << std::endl;
        std::cout.flush();

        // Extra explicit cleanup “as if between batches” (no-op if already clean)
        this->releaseCEBatchMemory();

        // Mirror refutation: a CE-refuted operator-only conjecture dooms its
        // pool mirror too (D-229). Runs
        // single-threaded between the pool join and survivor collection so
        // the flip is a pure function of this pass's CE verdicts.
        if (parameters.mirror_refutation) {
            const int flipped = applyMirrorRefutations(
                contradictionTable, conjectures, mirrorPartnerMap);
            std::cout << "CE filter: mirror refutation flipped " << flipped
                      << " conjectures." << std::endl;
        }

        // Keep non-contradictory conjectures
        std::vector<std::string> filtered;
        filtered.reserve(conjectures.size());
        for (int i = 0; i < static_cast<int>(conjectures.size()); ++i)
            if (!contradictionTable[i].successful) filtered.push_back(conjectures[i]);

        ceFilteringActive = false;
        return filtered;
    }

    /// @brief Load `mirror_pairs.txt` into a symmetric partner map.
    ///
    /// @details
    /// Reads the conjecturer-written pairs artifact (one
    /// `source<TAB>mirror` row per operator-only conjecture whose mirror
    /// entered the pool; both columns byte-identical to `conjectures.txt`
    /// lines) and inserts BOTH directions — `source → mirror` and
    /// `mirror → source` — with per-key duplicate suppression, so the CE
    /// filter's flip pass can follow the relation from whichever member a
    /// counterexample refutes. Rows are read in file order (source-sorted by
    /// the writer), keeping each key's partner vector deterministic.
    /// Trailing `\r` is stripped; empty lines are skipped (an empty file is
    /// a defined result: no pairs, no flips).
    ///
    /// The file's existence is part of the pipeline contract: the
    /// conjecturer writes it unconditionally whenever `conjectures.txt` is
    /// written, so absence means the pipeline was invoked out of order —
    /// asserted, not tolerated.
    ///
    /// @param path Full path to `files/theorems/mirror_pairs.txt`.
    /// @return Symmetric partner map; empty when the file has no rows.
    /// @see [D-229](../../docs/agentic_swdd/40_decisions.md#d-229)
    MirrorPartnerMap loadMirrorPairs(const std::filesystem::path& path) {
        assert(std::filesystem::exists(path)
            && "loadMirrorPairs: mirror_pairs.txt missing - the conjecturer "
               "writes it whenever conjectures.txt is written; run "
               "--conjecture <Tag> first");

        MirrorPartnerMap partners;
        auto addDirected = [&partners](const std::string& from, const std::string& to) {
            auto& vec = partners[from];
            if (std::find(vec.begin(), vec.end(), to) == vec.end()) {
                vec.push_back(to);
            }
        };

        std::ifstream in(path);
        assert(in && "loadMirrorPairs: mirror_pairs.txt exists but cannot be opened");
        std::string line;
        while (std::getline(in, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;
            const std::size_t tabPos = line.find('\t');
            assert(tabPos != std::string::npos
                && "loadMirrorPairs: malformed row - every non-empty line is "
                   "source<TAB>mirror by the writer contract");
            std::string source = line.substr(0, tabPos);
            std::string mirror = line.substr(tabPos + 1);
            assert(!source.empty() && !mirror.empty()
                && "loadMirrorPairs: empty column - violates the writer contract");
            addDirected(source, mirror);
            addDirected(mirror, source);
        }
        return partners;
    }

    /// @brief Flip the mirror partners of CE-refuted conjectures to refuted.
    ///
    /// @details
    /// The mirror-refutation heuristic's flip pass. Snapshots which slots the
    /// CE bursts refuted (`table[i].successful` seeds), then sweeps them in
    /// ascending index order: for every seeded conjecture, each partner from
    /// `partners` that is present in this batch has its slot's `successful`
    /// set `true`. Only CE-confirmed refutations seed — a flipped slot never
    /// seeds further flips (no cascading) — so the outcome is a pure function
    /// of the CE verdicts plus the pairs file, independent of sweep order. A
    /// partner absent from `conjectures` is a defined case (refuted or
    /// flipped in an earlier fact-file pass, or deduplicated at generation),
    /// not a failure.
    ///
    /// Refutation-side only: this marks conjectures as dropped pre-prover; no
    /// mirror ever re-enters as a proof step
    /// ([I-81](../../docs/agentic_swdd/30_invariants.md#i-81) / D-112).
    ///
    /// @param table       The CE contradiction table, one slot per
    ///                    conjecture; flipped in place.
    /// @param conjectures The batch's conjecture strings, index-aligned with
    ///                    `table`.
    /// @param partners    Symmetric partner map from `loadMirrorPairs`.
    /// @return Number of slots flipped by this pass.
    /// @see loadMirrorPairs — builds `partners`.
    int applyMirrorRefutations(std::vector<ContradictionItem>& table,
        const std::vector<std::string>& conjectures,
        const MirrorPartnerMap& partners) {
        assert(table.size() == conjectures.size()
            && "applyMirrorRefutations: table and conjecture list must be "
               "index-aligned");
        if (partners.empty()) return 0;

        std::unordered_map<std::string, int> slotOf;
        slotOf.reserve(conjectures.size());
        for (int i = 0; i < static_cast<int>(conjectures.size()); ++i) {
            slotOf.emplace(conjectures[i], i);
        }

        std::vector<char> seed(table.size(), 0);
        for (std::size_t i = 0; i < table.size(); ++i) {
            seed[i] = table[i].successful ? 1 : 0;
        }

        int flipped = 0;
        for (int i = 0; i < static_cast<int>(table.size()); ++i) {
            if (!seed[i]) continue;
            const auto it = partners.find(conjectures[i]);
            if (it == partners.end()) continue;
            for (const std::string& partner : it->second) {
                const auto slotIt = slotOf.find(partner);
                if (slotIt == slotOf.end()) continue;
                ContradictionItem& row = table[slotIt->second];
                if (!row.successful) {
                    row.successful = true;
                    ++flipped;
                }
            }
        }
        return flipped;
    }

} // namespace gl
