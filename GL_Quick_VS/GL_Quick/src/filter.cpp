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
//
// Bodies are added across a chain of incremental commits (B–E in
// docs/.../sandbox/filter_separation plan). This commit (A) is scaffolding:
// the file compiles and links empty so the build manifest change is verified
// in isolation.

#include "filter.hpp"
#include "prover.hpp"
#include "parameters.hpp"
#include "msvc_sort.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace gl {

    // ------------------------------------------------------------------
    // CE-filter file IO (moved from prover.cpp in commit B).
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
    /// @see `docs/10_pipeline/03_ce_filter.md` — fact-base shape rationale.
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
    /// @see `docs/10_pipeline/03_ce_filter.md` — pipeline stage chapter.
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
    // CE-filter request generation (moved from prover.cpp / prover.hpp in
    // commit C).
    // ------------------------------------------------------------------

    /// CE filter: accepts statements that appear in subkeys OR full keys.
    int16_t ExpressionAnalyzer::filterIntEncodedStatementsCE(
        const IntEncodedExpr* stmts, int16_t count,
        const HashMemory& mem,
        int16_t* outIndices, int16_t maxOut) {

        int16_t buf[ExecutionParameters::MAX_KEY_SLOTS];
        int16_t nOut = 0;

        for (int16_t i = 0; i < count && nOut < maxOut; ++i) {
            const IntEncodedExpr& s = stmts[i];

            // Build single-expr IntNormalizedKey on stack
            int16_t pos = 0;
            buf[pos++] = s.nameId;
            buf[pos++] = s.negation;
            for (int16_t j = 0; j < s.arity; ++j) {
                buf[pos++] = s.argId[j];
                buf[pos++] = 0; // changeable
            }
            // Normalize: sequential IDs by first appearance
            {
                int16_t varMap[ExecutionParameters::MAX_KEY_SLOTS];
                int16_t nV = 0;
                int16_t nextN = 1;
                for (int16_t p = 2; p < pos; p += 2) {
                    int16_t raw = buf[p];
                    int16_t norm = 0;
                    for (int16_t v = 0; v < nV; ++v) {
                        if (varMap[v * 2] == raw) { norm = varMap[v * 2 + 1]; break; }
                    }
                    if (norm == 0) {
                        norm = nextN++;
                        varMap[nV * 2] = raw;
                        varMap[nV * 2 + 1] = norm;
                        ++nV;
                    }
                    buf[p] = norm;
                }
            }

            IntNormalizedKey ik(1, buf, pos);
            if (mem.normalizedEncodedSubkeys.find(ik) == mem.normalizedEncodedSubkeys.end()
                && mem.normalizedEncodedKeys.find(ik) == mem.normalizedEncodedKeys.end())
                continue;
            if (s.maxIteration > parameters.maxIterationNumberVariable)
                continue;

            outIndices[nOut++] = i;
        }
        return nOut;
    }

    // ------------------------------------------------------------------
    // CE-filter batch lifecycle (moved from prover.cpp in commit D).
    // ------------------------------------------------------------------

    /// @brief Seed the CE-filter LB tree with the simple-fact base — every
    /// fact is installed as either a statement or a hash-engine rule,
    /// depending on its shape.
    ///
    /// @details
    /// One CE-filter LB is created per conjecture batch slot
    /// (`0..batchSize-1`) under `ceBody.simpleMap`. For each slot, every
    /// entry of `simpleFacts` is dispatched through the generic
    /// `addExprToMemoryBlock` so atomic facts become statements in
    /// `Memory::encodedStatements` while implication-shaped facts become
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
    ///                     installed under `ceBody.simpleMap[std::to_string(i)]`
    ///                     for each `i` in `[0, batchSize)`.
    /// @pre  `precompileStructuralOperators` has run for the batch's
    ///       theorem list (I-1).
    /// @post Each `ceBody.simpleMap[std::to_string(i)]->encodedStatements`
    ///       and `…->overallHashMemory` carry the simple-fact base, ready
    ///       for the per-slot conjecture probe.
    /// @see `addExprToMemoryBlock` — installer.
    void ExpressionAnalyzer::loadFactsForCEFiltering(
        std::vector<std::string> simpleFacts,
        int batchSize) {

        for (std::size_t start = 0; start < batchSize; start++) {
            Memory* lb0 = nullptr;
            {
                std::map<std::string, Memory*>::iterator it = ceBody.simpleMap.find(std::to_string(start));
                if (it != ceBody.simpleMap.end() && it->second) {
                    lb0 = it->second;
                }
                else {
                    lb0 = new Memory();
                    ceBody.simpleMap[std::to_string(start)] = lb0;
                    lb0->parentMemory = &ceBody;
                    lb0->level = 0;
                    lb0->exprKey = std::to_string(start);
                    lb0->contradictionIndex = -1;
                    permanentBodiesCE.push_back(lb0);
                }
            }

            // Origins and levels
            std::pair<std::string, std::vector<ExpressionWithValidity>> origin = std::make_pair("", std::vector<ExpressionWithValidity>());
            if (parameters.trackHistory) {
                origin.first = "CE_building_block";
            }
            const std::set<int> lvl0{ 0 };

            // LB0: +, *, s
            for (const auto& s : simpleFacts) this->addExprToMemoryBlock(s, *lb0, 0, 4, lvl0, origin, -1, -1, "main", false);
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

        const std::set<int> lvl0{ 0 };

        this->addToHashMemory(chain, head, std::set<std::string>{},
            *mb, mb->overallHashMemory, lvl0,
            replacedConjecture,
            parameters.standardMaxAdmissionDepth,
            parameters.standardMaxSecondaryNumber,
            false,
            parameters.minNumOperatorsKeyCE,
            "implication", true, replacedConjecture, "main");
    }

    /// @brief Tear down the CE-filter batch state — clear the simple-fact
    /// LB, free per-batch arenas, reset the contradiction table.
    ///
    /// @details
    /// Called once per batch after the CE filter has finished probing every
    /// conjecture and `saveFilteredConjectures` has persisted the survivors.
    /// Releases the CE-filter LB's `keyArena`, drops the simple-fact LB tree
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

        updateGlobalDirectTuples.clear();
        std::vector<std::tuple<std::string, int>>().swap(updateGlobalDirectTuples);

        // ---- 1) Collect ALL potential CE roots from every holder ----
        std::vector<Memory*> roots;
        roots.reserve(ceBody.simpleMap.size()
            + permanentBodiesCE.size()
            + indexCE.size() * 2);

        // from ceBody root map
        for (std::map<std::string, Memory*>::iterator it = ceBody.simpleMap.begin();
            it != ceBody.simpleMap.end(); ++it) {
            if (it->second) roots.push_back(it->second);
        }
        // from permanentBodiesCE
        for (std::size_t i = 0; i < permanentBodiesCE.size(); ++i)
            if (permanentBodiesCE[i]) roots.push_back(permanentBodiesCE[i]);
        // from indexCE keys + values (sorted by exprKey for determinism)
        {
            std::vector<Memory*> indexCEKeys;
            for (auto it = indexCE.begin(); it != indexCE.end(); ++it)
                if (it->first) indexCEKeys.push_back(it->first);
            std::sort(indexCEKeys.begin(), indexCEKeys.end(),
                [](const Memory* a, const Memory* b) { return a->exprKey < b->exprKey; });
            for (Memory* ik : indexCEKeys) {
                roots.push_back(ik);
                const std::vector<Memory*>& vs = indexCE[ik];
                for (std::size_t j = 0; j < vs.size(); ++j)
                    if (vs[j]) roots.push_back(vs[j]);
            }
        }

        // ---- 2) Iterative DFS to delete each node exactly once ----
        std::unordered_set<Memory*> seen;
        std::vector<Memory*> stack;
        stack.reserve(roots.size());
        for (Memory* r : roots) if (r) stack.push_back(r);

        while (!stack.empty()) {
            Memory* node = stack.back(); stack.pop_back();
            if (!node || !seen.insert(node).second) continue;

            for (std::map<std::string, Memory*>::iterator it = node->simpleMap.begin();
                it != node->simpleMap.end(); ++it)
                if (it->second) stack.push_back(it->second);

            delete node; // d-tor frees STL members
        }

        // ---- 3) Drop/compact all CE containers & maps (release capacity) ----
        // ceBody’s root map
        std::map<std::string, Memory*>().swap(ceBody.simpleMap);

        // explicit reset of ceBody's heavy members via swap-with-empty
        {
            Memory empty;
            using std::swap;
            swap(ceBody.startInt, empty.startInt);
            swap(ceBody.toBeProved, empty.toBeProved);
            swap(ceBody.encodedStatements, empty.encodedStatements);
            swap(ceBody.statementLevelsMap, empty.statementLevelsMap);
            swap(ceBody.exprKey, empty.exprKey);
            ceBody.parentMemory = nullptr;
            swap(ceBody.level, empty.level);
            swap(ceBody.overallHashMemory, empty.overallHashMemory);
            swap(ceBody.equivalenceClassesMap, empty.equivalenceClassesMap);
            swap(ceBody.localEncodedStatements, empty.localEncodedStatements);
            swap(ceBody.localEncodedStatementsDelta, empty.localEncodedStatementsDelta);
            swap(ceBody.mailIn, empty.mailIn);
            swap(ceBody.mailOut, empty.mailOut);
            swap(ceBody.wholeExpressions, empty.wholeExpressions);
            swap(ceBody.eqClassSttmntIndexMapMap, empty.eqClassSttmntIndexMapMap);
            swap(ceBody.isActive, empty.isActive);
            swap(ceBody.isPartOfRecursion, empty.isPartOfRecursion);
            swap(ceBody.deltaNumberStatements, empty.deltaNumberStatements);
            swap(ceBody.exprOriginMap, empty.exprOriginMap);
            swap(ceBody.recursionCounter, empty.recursionCounter);
            swap(ceBody.contradictionIndex, empty.contradictionIndex);
        }

        // CE containers
        destroyParentChildrenMap(indexCE);
        destroyMailboxes(boxesCE);
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

        // Keep contradiction indices stable across batches
        contradictionTable.clear();
        contradictionTable.reserve(contradictionTable.size() + conjectures.size());
        for (int i = 0; i < static_cast<int>(conjectures.size()); ++i)
            contradictionTable.push_back(ContradictionItem(conjectures[i], false));

        const unsigned batchSize = std::max(1u, logicalCores);

        loadFactsForCEFiltering(simpleFacts, batchSize);

        for (std::size_t start = 0; start < conjectures.size(); start += batchSize) {
            for (int index = 0; index < static_cast<int>(batchSize); ++index) {
                std::map<std::string, Memory*>::iterator it = ceBody.simpleMap.find(std::to_string(index));
                if (it != ceBody.simpleMap.end() && it->second) {
                    // Clear local memory from previous batch
                    it->second->overallHashMemory.clear();
                    it->second->contradictionIndex = -1;
                    it->second->isActive = true;
                }
            }

            const std::size_t end = std::min(start + batchSize, conjectures.size());

            // Build CE memory for THIS batch only
            for (std::size_t i = start; i < end; ++i) {
                const int cIndex = static_cast<int>(i);
                this->addConjectureForCEFiltering(conjectures[i], ceBody.simpleMap[std::to_string(i % batchSize)], cIndex);
            }

            // Parent->children index and mailboxes for THIS batch
            indexCE = buildParentChildrenMap(permanentBodiesCE);
            boxesCE = buildPerCoreMailboxes(indexCE);

            // Run the prover on this batch
            this->prove(parameters.numberIterationsConjectureFiltering,
                permanentBodiesCE, indexCE, boxesCE);
        }

        // Extra explicit cleanup “as if between batches” (no-op if already clean)
        this->releaseCEBatchMemory();

        // Keep non-contradictory conjectures
        std::vector<std::string> filtered;
        filtered.reserve(conjectures.size());
        for (int i = 0; i < static_cast<int>(conjectures.size()); ++i)
            if (!contradictionTable[i].successful) filtered.push_back(conjectures[i]);

        ceFilteringActive = false;
        return filtered;
    }

    // ========================================================================
    // generateEncodedRequestsStaticCE — CE mode: no mandatory elements.
    // Enumerates statement combinations that directly match full hash keys.
    // Own grow loop (dual-check: fullKeys for emit, subkeys for growth).
    // ========================================================================
    int16_t ExpressionAnalyzer::generateEncodedRequestsStaticCE(
        Memory& body,
        const HashMemory& intMemory,
        TypedArena<IntEncodedExpr>& exprArena,
        StaticRequest* outBuf, int16_t maxOut)
    {
        const int maxKeyLen = intMemory.maxKeyLength;
        if (maxKeyLen <= 0) return 0;

        NameMap& nm = body.nameMap;
        const int16_t mainValidityId = NameMap::MAIN_ID;

        StaticRequestEmitter emitter(exprArena, outBuf, maxOut);

        // Filter allStatements (CE filter: subkeys OR full keys)
        const IntEncodedExpr* allIntStmts = body.intEncodedStatements.data();
        const int16_t allIntCount = static_cast<int16_t>(body.intEncodedStatements.size());
        int16_t filteredIdx[8192];
        int16_t nFiltered = filterIntEncodedStatementsCE(allIntStmts, allIntCount,
            intMemory, filteredIdx, 8192);
        // gl::msvc_sort — CE-filter version of the cross-host
        // deterministic sort. See memory.cpp:generateEncodedRequestsStatic
        // for the rationale and msvc_sort.hpp for the algorithm.
        gl::msvc_sort(filteredIdx, filteredIdx + nFiltered, [&](int16_t a, int16_t b) {
            return nm.decode(allIntStmts[a].nameId) < nm.decode(allIntStmts[b].nameId);
        });

        // Combinatorial enumeration — directly target full keys
        struct StackItem {
            int start;
            int16_t allIdx[ExecutionParameters::MAX_EXPRESSIONS];
            int16_t count;
            int16_t validityId;
        };

        std::vector<StackItem> stack;
        {
            StackItem init;
            init.start = 0;
            init.count = 0;
            init.validityId = mainValidityId;
            stack.push_back(init);
        }

        while (!stack.empty()) {
            StackItem top = stack.back();
            stack.pop_back();

            for (int i = top.start; i < nFiltered; ++i) {
                if (top.count + 1 > maxKeyLen) break;

                const int16_t allIdx = filteredIdx[i];
                const IntEncodedExpr& ie = allIntStmts[allIdx];

                if (!nm.comparable(top.validityId, ie.validityId)) continue;
                int16_t newValidityId = nm.deeperOf(top.validityId, ie.validityId);

                const IntEncodedExpr* ptrs[ExecutionParameters::MAX_EXPRESSIONS];
                for (int16_t k = 0; k < top.count; ++k)
                    ptrs[k] = &allIntStmts[top.allIdx[k]];
                ptrs[top.count] = &ie;
                const int16_t newCount = static_cast<int16_t>(top.count + 1);

                // Check full key match → emit
                std::pair<bool, IntNormalizedKey> prFull =
                    preEvaluateFromEncoded(ptrs, newCount, body, mainValidityId,
                        intMemory.normalizedEncodedKeys);
                if (prFull.first) {
                    emitter.emit(ptrs, newCount, prFull.second);
                }

                // Check subkey for further growth
                if (newCount < maxKeyLen) {
                    std::pair<bool, IntNormalizedKey> prSub =
                        preEvaluateFromEncoded(ptrs, newCount, body, mainValidityId,
                            intMemory.normalizedEncodedSubkeys);
                    if (prSub.first) {
                        StackItem next;
                        next.start = i + 1;
                        std::memcpy(next.allIdx, top.allIdx, top.count * sizeof(int16_t));
                        next.allIdx[top.count] = allIdx;
                        next.count = newCount;
                        next.validityId = newValidityId;
                        stack.push_back(next);
                    }
                }
            }
        }

        return emitter.outCount;
    }

} // namespace gl
