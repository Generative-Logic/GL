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
#include "proof_graph_writer.hpp"
#include "../project_paths.hpp"
#include "../memory_infra/lb_deload.hpp"
#include <cstdlib>
#include <iomanip>
namespace gl {
/// @brief Bind the proof graph writer operation inputs.
/// @details Construction stores explicit references; block storage and all invocation lifetimes remain with their owners.
/// @param body Borrowed Memory interface or state; must outlive this component.
/// @param globalTheoremList Borrowed std::vector interface or state; must outlive this component.
/// @param globalTheoremStrings Borrowed std::vector interface or state; must outlive this component.
/// @param globalTheoremProducers Borrowed std::vector interface or state; must outlive this component.
/// @param definitions Borrowed CompiledDefinitionRegistry interface or state; must outlive this component.
/// @param simpleMapStore Borrowed SimpleMapStore interface or state; must outlive this component.
/// @param cachedProofStacks Borrowed std::vector interface or state; must outlive this component.
/// @param exportTheoremUsage Borrowed TheoremUsageGraph interface or state; must outlive this component.
/// @param exportCurrentChapterTheorem Borrowed std::string interface or state; must outlive this component.
/// @param orElimCitedOrByMerged Borrowed std::vector interface or state; must outlive this component.
/// @param compilationPersistence Borrowed CompilationPersistence interface or state; must outlive this component.
/// @param theoremPublisher Borrowed TheoremPublisher interface or state; must outlive this component.
/// @param proofReconstructor Borrowed ProofReconstructor interface or state; must outlive this component.
/// @return None; construction binds the supplied references.
ProofGraphWriter::ProofGraphWriter(Memory& body, std::vector< std::tuple<std::string, std::string, std::string, std::string >>& globalTheoremList, std::unordered_set<std::string>& globalTheoremStrings, std::vector<const Memory*>& globalTheoremProducers, const CompiledDefinitionRegistry& definitions, SimpleMapStore& simpleMapStore, std::map<std::string, CachedProofStack>& cachedProofStacks, TheoremUsageGraph& exportTheoremUsage, std::string& exportCurrentChapterTheorem, std::map<std::string, std::string>& orElimCitedOrByMerged, CompilationPersistence& compilationPersistence, TheoremPublisher& theoremPublisher, ProofReconstructor& proofReconstructor) : body(body), globalTheoremList(globalTheoremList), globalTheoremStrings(globalTheoremStrings), globalTheoremProducers(globalTheoremProducers), definitions(definitions), simpleMapStore(simpleMapStore), cachedProofStacks(cachedProofStacks), exportTheoremUsage(exportTheoremUsage), exportCurrentChapterTheorem(exportCurrentChapterTheorem), orElimCitedOrByMerged(orElimCitedOrByMerged), compilationPersistence(compilationPersistence), theoremPublisher(theoremPublisher), proofReconstructor(proofReconstructor) {}

/// @brief Whether a block's history records a `vacuous truth` line for a goal
///        at main (D-573).
/// @details The vacuous-truth branch of `GoalProcessor::dischargeContradiction`
/// closes an induction goal without depositing it as a statement; the closure
/// is recorded only as the goal's history line at main. The induction-chapter
/// gates of `generateRawProofGraph` accept such a block as the goal's closing
/// block, besides a block holding the goal as a statement; the chapter is then
/// built from that history line like any other. Reads the origin interner and
/// `exprOriginMap`, which a deload leaves in memory.
/// @param block The induction block whose history is read.
/// @param goal The goal expression in its literal spelling.
/// @return `true` iff the history holds a `vacuous truth` record for
///         (@p goal, main).
bool ProofGraphWriter::hasVacuousTruthAtMain(const Memory& block, const std::string& goal) {
    int64_t key = 0;
    if (!lookupOriginKey(block.originInterner, StrSpan(goal), StrSpan("main", 4), key)) return false;
    const int32_t id = block.exprOriginMap.lookup(key);
    if (id == 0) return false;
    for (const IdOrigin& origin : block.exprOriginMap.recordsAt(id)) {
        if (origin.first == OriginTag::vacuousTruth) return true;
    }
    return false;
}

/// @brief Write debug proof chapters for the requested logical block.
/// @details Resolves the original child path, reconstructs every recorded end and publishes the indexed chapters and mapping in the existing order.
/// @param path Logical-block routing keys from the root.
/// @param outDirParam Destination for debug chapters and their mapping.
/// @return No value.
void gl::ProofGraphWriter::findEnds(const std::vector<std::string>& path, const std::filesystem::path& outDirParam) {
    namespace fs = std::filesystem;

    // ---- prepare output dir: PROJECT_ROOT/files/raw_proof_graph ----
    std::filesystem::path outDir = outDirParam.empty() ? (fs::path("files") / "raw_proof_graph") : outDirParam;

    // MODIFIED: Do NOT remove existing files. Only ensure directory exists.
    std::error_code ec2;
    fs::create_directories(outDir, ec2);

    // ---- navigate to requested memory block ----
    Memory* memoryBlock = &this->body;
    for (std::size_t i = 0; i < path.size(); ++i) {
        const std::string& elt = path[i];
        Memory* child = simpleMapStore.findChild(memoryBlock, elt);
        if (child == NULL) {
            return; // path invalid
        }
        memoryBlock = child;
    }

    // ---- collect candidate ends (keys of exprOriginMap) ----
    std::set<ExpressionWithValidity> allExprs;
    for (int32_t oid = 1; oid <= memoryBlock->exprOriginMap.count(); ++oid) {
        auto [endExpr, endValidity] = decodeOriginKey(
            memoryBlock->exprOriginMap.decodeKey(oid),
            memoryBlock->originInterner);
        allExprs.insert(ExpressionWithValidity(std::move(endExpr),
                                               std::move(endValidity)));
    }

    // ---- compute stack sizes for sorting ----
    std::vector<ExpressionWithValidity> endsVec(allExprs.begin(), allExprs.end());
    std::vector<int> stackSizes;
    stackSizes.reserve(endsVec.size());
    for (std::size_t i = 0; i < endsVec.size(); ++i) {
        std::vector<std::vector<std::string>> stack;
        std::set<ExpressionWithValidity> covered;
        proofReconstructor.buildStack(*memoryBlock, endsVec[i], stack, covered);
        stackSizes.push_back(static_cast<int>(stack.size()));
    }

    // ---- order ends by size (desc) ----
    std::vector<ExpressionWithValidity> endsOrdered = proofReconstructor.sortByValuesDesc(endsVec, stackSizes);

    // ---- rebuild globalTheoremList (companion dedup set + producer
    // attribution cleared with it; the re-appends below record no producer) ----
    this->globalTheoremList.clear();
    this->globalTheoremStrings.clear();
    this->globalTheoremProducers.clear();
    std::string joinedPath;
    for (std::size_t i = 0; i < path.size(); ++i) {
        if (i > 0) joinedPath.push_back(';');
        joinedPath += path[i];
    }
    for (std::size_t i = 0; i < endsOrdered.size(); ++i) {
        std::string theoremStr;
        // Format: PATH + "+" + VALIDITY + "+" + ORIGINAL
        if (!joinedPath.empty()) {
            theoremStr = joinedPath + "+" + endsOrdered[i].validityName + "+" + endsOrdered[i].original;
        }
        else {
            theoremStr = "+" + endsOrdered[i].validityName + "+" + endsOrdered[i].original;
        }
        theoremPublisher.appendGlobalTheorem(theoremStr, "debug", "-1", "-1");
    }

    // MODIFIED: Scan for highest existing index to append
    int startIdx = 0;
    if (fs::exists(outDir)) {
        int max_found = -1;
        const std::regex reIdx(R"(^(\d+)_.*)");
        for (const auto& entry : fs::directory_iterator(outDir)) {
            if (entry.is_regular_file()) {
                std::string fname = entry.path().filename().string();
                std::smatch m;
                if (std::regex_search(fname, m, reIdx)) {
                    try {
                        int val = std::stoi(m[1].str());
                        if (val > max_found) max_found = val;
                    }
                    catch (...) {}
                }
            }
        }
        if (max_found >= 0) {
            startIdx = max_found + 1;
        }
    }

    // ---- write stacks as <index>_debug.txt and mapping file ----
    // MODIFIED: Open in Append Mode
    std::ofstream mapFile((outDir / "global_theorem_list.txt").string().c_str(), std::ios::out | std::ios::app);
    if (!mapFile.is_open()) {
        return;
    }

    for (std::size_t idx = 0; idx < this->globalTheoremList.size(); ++idx) {
        const std::tuple<std::string, std::string, std::string, std::string>& entry = this->globalTheoremList[idx];
        const std::string& theoremStr = std::get<0>(entry);
        const std::string& method = std::get<1>(entry);
        const std::string& var = std::get<2>(entry);

        // Parse theoremStr... (omitted logic remains same as original, just need to parse)
        std::string pathPart;
        std::string validityPart;
        std::string originalPart;
        std::size_t firstPlus = theoremStr.find('+');
        if (firstPlus == std::string::npos) {
            originalPart = theoremStr;
        }
        else {
            pathPart = theoremStr.substr(0, firstPlus);
            std::size_t secondPlus = theoremStr.find('+', firstPlus + 1);
            if (secondPlus == std::string::npos) {
                originalPart = theoremStr.substr(firstPlus + 1);
            }
            else {
                validityPart = theoremStr.substr(firstPlus + 1, secondPlus - (firstPlus + 1));
                originalPart = theoremStr.substr(secondPlus + 1);
            }
        }

        // navigate to target block
        Memory* mb = &this->body;
        if (!pathPart.empty()) {
            std::size_t start = 0;
            while (true) {
                std::size_t pos = pathPart.find(';', start);
                std::string node = (pos == std::string::npos)
                    ? pathPart.substr(start)
                    : pathPart.substr(start, pos - start);
                if (!node.empty()) {
                    Memory* child = simpleMapStore.findChild(mb, node);
                    if (child == NULL) {
                        mb = NULL; break;
                    }
                    mb = child;
                }
                if (pos == std::string::npos) break;
                start = pos + 1;
            }
        }
        if (mb == NULL) continue;

        // build stack
        ExpressionWithValidity targetKey(originalPart, validityPart);
        std::vector< std::vector<std::string> > stack;
        std::set<ExpressionWithValidity> covered;
        proofReconstructor.buildStack(*mb, targetKey, stack, covered);

        // MODIFIED: Use the calculated accumulated index
        int currentIdx = startIdx + static_cast<int>(idx);
        fs::path filePath = outDir / (std::to_string(static_cast<long long>(currentIdx)) + "_debug.txt");

        std::ofstream ofs(filePath.string().c_str(), std::ios::out | std::ios::trunc);
        if (ofs.is_open()) {
            for (std::size_t r = 0; r < stack.size(); ++r) {
                const std::vector<std::string>& row = stack[r];
                for (std::size_t c = 0; c < row.size(); ++c) {
                    if (c > 0) ofs << '\t';
                    ofs << row[c];
                }
                ofs << '\n';
            }
            ofs.close();
        }

        // record mapping line
        mapFile << theoremStr << '\t' << method << '\t' << var << '\n';
    }
    mapFile.close();
}


/// @brief Export the recorded theorem proofs into raw chapter files.
/// @details Preserves chapter selection, reconstruction, cross-chapter usage checks, serialization and deload release order. Synthetic citations are seeded before traversal; chosen edges are committed after writing each chapter.
/// @param theoremList Ordered theorem records to export.
/// @param outDir Destination for chapter files and their mapping.
/// @return No value.
void ProofGraphWriter::generateRawProofGraph(
    const std::vector<std::tuple<std::string, std::string, std::string, std::string>>& theoremList,
    const std::filesystem::path& outDirParam) {
    namespace fs = std::filesystem;
    using FrameClock = std::chrono::steady_clock;

    const auto rawGenerationStarted = FrameClock::now();
    std::string frameBatch;
#ifdef _WIN32
    char* batchRaw = nullptr;
    size_t batchLength = 0;
    const errno_t batchEnvironmentResult =
        _dupenv_s(&batchRaw, &batchLength, "GL_FRAME_BATCH");
    assert(batchEnvironmentResult == 0);
    if (batchRaw != nullptr) {
        frameBatch.assign(batchRaw);
        std::free(batchRaw);
    }
#else
    const char* batchRaw = std::getenv("GL_FRAME_BATCH");
    if (batchRaw != nullptr) frameBatch.assign(batchRaw);
#endif
    const auto frameSecondsSince = [](FrameClock::time_point started) {
        return std::chrono::duration<double>(FrameClock::now() - started).count();
    };
    const auto recordFrameTiming = [&](const char* stage,
                                       double seconds,
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
        timing << "{\"batch\":\"" << frameBatch
               << "\",\"count\":" << count
               << ",\"excluded\":false"
               << ",\"parent\":\"native.raw_proof\""
               << ",\"seconds\":" << std::setprecision(12) << seconds
               << ",\"stage\":\"" << stage << "\"}\n";
        timing.flush();
        assert(timing.good());
    };
    double buildStackSeconds = 0.0;
    double serializationSeconds = 0.0;
    double releaseSeconds = 0.0;
    int64_t buildStackCalls = 0;
    int64_t serializedChapters = 0;
    int64_t releasedTheorems = 0;

    // D-51 (option 1): reset thread_local path stack at run start. Each
    // chapter emission pushes/pops independently — the clear here is a
    // belt-and-suspenders against any leak across runs.
    proofReconstructor.clearBuildStackPath();

    // Fresh cross-chapter usage graph per export run; the current-chapter
    // theorem is set per loop iteration below and empty outside the loop
    // (empty = buildStack's cross-chapter admissibility check disabled).
    exportTheoremUsage.clear();
    exportCurrentChapterTheorem.clear();

    auto toLower = [](std::string s) {
        for (std::size_t i = 0; i < s.size(); ++i) {
            s[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(s[i])));
        }
        return s;
        };

    // Synthetic chapters have fixed citations, independent of origin choice.
    // Make those obligations visible before any earlier chapter can cite the
    // synthetic result; otherwise its later emission could close a cycle.
    for (const auto& theorem : theoremList) {
        const std::string& name = std::get<0>(theorem);
        const std::string method = toLower(std::get<1>(theorem));
        std::vector<std::string> row{name, "main"};
        if (method == "reformulated statement") row.push_back("reformulated from");
        else if (method == "incubator back reformulation") row.push_back(method);
        else if (method == "or theorem" || method == "or elimination") row.push_back(method);
        else continue; // Ordinary proof chapters choose their origins during the walk.
        row.push_back(std::get<2>(theorem));
        row.push_back("main");
        if (method == "or theorem" || method == "or elimination") {
            row.push_back(std::get<3>(theorem));
            row.push_back("main");
        }
        if (method == "or elimination") {
            const auto license = orElimCitedOrByMerged.find(name);
            assert(license != orElimCitedOrByMerged.end()
                && "generateRawProofGraph: fixed or-elimination edge has no license");
            row.push_back(license->second);
            row.push_back("main");
        }
        exportTheoremUsage.addEdgesFromStack(name, {row});
    }

    // Per-chapter reload release (G-53): the export
    // reloads LBs from their SSD images on demand; without releasing them the
    // resident block set grows monotonically across the walk to static-pool
    // exhaustion. Every reload appends to this sink (Memory::reloadFromImage);
    // each theorem's reloaded LBs are released after its chapters are written
    // (loop below). The on-disk image stays, so a later theorem that revisits
    // an LB reloads it again on demand.
    std::vector<Memory*> exportReloaded;
    g_exportReloadSink = &exportReloaded;

    // ---------- out dir: "files/raw_proof_graph" by default ----------
    fs::path outDir = outDirParam.empty() ? (fs::path("files") / "raw_proof_graph") : outDirParam;
    std::error_code ec;

    // No remove_all: earlier batches' chapters must survive — the export
    // appends, with the start index scanned from existing files below.
    fs::create_directories(outDir, ec);

    // Single canonical GL-binary folder regardless of per-batch outDir
    // (mirrors prover.cpp::compileCoreExpressionMap reader path). Without
    // this, configs that override raw_proof_graph_folder (incubator batches
    // set it to files/incubator/raw_proof_graph) would land the export in
    // files/incubator/GL_binaries/ — a directory the loader and the Python
    // _merge_into_shared step never read from. See D-54.
    fs::path glBinDir = gl::projectRoot()
                      / "files" / "GL_binaries";
    const auto binaryExportStarted = FrameClock::now();
    compilationPersistence.exportCompiledExpressionsJSON(glBinDir);
    const double binaryExportSeconds = frameSecondsSince(binaryExportStarted);

    // Scan for the start index from the highest existing numeric prefix.
    const auto indexStarted = FrameClock::now();
    int idx = 0;
    if (fs::exists(outDir)) {
        int max_found = -1;
        const std::regex reIdx(R"(^(\d+)_.*)");
        for (const auto& entry : fs::directory_iterator(outDir)) {
            if (entry.is_regular_file()) {
                std::string fname = entry.path().filename().string();
                std::smatch m;
                if (std::regex_search(fname, m, reIdx)) {
                    try {
                        int val = std::stoi(m[1].str());
                        if (val > max_found) max_found = val;
                    }
                    catch (...) {}
                }
            }
        }
        if (max_found >= 0) {
            idx = max_found + 1;
        }
    }
    const double indexSeconds = frameSecondsSince(indexStarted);

    const auto timedBuildStack = [&](Memory& memoryBlock,
                                     const ExpressionWithValidity& proved,
                                     std::vector<std::vector<std::string>>& stack,
                                     std::set<ExpressionWithValidity>& covered) {
        const auto started = FrameClock::now();
        // Each top-level walk starts with an empty `covered`, and
        // `buildChapterStack` starts its journal empty too. Without that the
        // outermost frame's surviving insertions — which nothing ever rolls
        // back — would accumulate across the export.
        const bool result = proofReconstructor.buildChapterStack(memoryBlock, proved, stack, covered);
        buildStackSeconds += frameSecondsSince(started);
        ++buildStackCalls;
        return result;
    };

    auto startsWith = [](const std::string& s, const std::string& pref) -> bool {
        return s.size() >= pref.size() && std::equal(pref.begin(), pref.end(), s.begin());
        };
    auto endsWith = [](const std::string& s, const std::string& suf) -> bool {
        return s.size() >= suf.size() && std::equal(s.end() - suf.size(), s.end(), suf.begin());
        };

    // Generic over the statement-container type: the statified registry and
    // its sibling vectors (ArenaVector) all pass through; each exposes
    // size() + operator[].
    auto containsEncoded = [](const auto& vec,
                              const NameMap& nm, const std::string& expr) -> bool {
        // Stored rows are canonical-pipeline encodings, so full struct
        // equality collapses to the (originalId, validityId) pair. lookup is
        // non-minting: a never-interned needle cannot be a stored statement.
        const NameId origId = nm.lookup(expr);
        if (origId == 0) return false;
        for (int32_t i = 0; i < static_cast<int32_t>(vec.size()); ++i) {
            if (vec[i].originalId == origId
                && vec[i].validityId == NameMap::MAIN_ID) return true;
        }
        return false;
        };

    // The single choke point every chapter kind passes through (direct,
    // induction triad, debug, reformulated, incubator back-reformulation,
    // or-theorem): writes the chapter file, then commits the chapter's
    // theorem-citation edges into the export usage graph so later chapters'
    // buildStack admissibility probes see them
    // (D-276). Committing on emitted rows —
    // not on how they were built — also covers cache-served stacks.
    auto writeStackIndexed = [&](int idx, const std::string& part,
        const std::string& theoremName,
        const std::vector<std::vector<std::string>>& stackRows) {

            const auto started = FrameClock::now();

            fs::path fp = outDir / (std::to_string(idx) + "_" + part + ".txt");
            std::ofstream f(fp.c_str());

            for (std::size_t r = 0; r < stackRows.size(); ++r) {
                const std::vector<std::string>& row = stackRows[r];
                for (std::size_t c = 0; c < row.size(); ++c) {
                    if (c > 0) f << '\t';
                    f << row[c];
                }
                f << '\n';
            }
            f.flush();
            assert(f.good());
            exportTheoremUsage.addEdgesFromStack(theoremName, stackRows);
            serializationSeconds += frameSecondsSince(started);
            ++serializedChapters;
        };

    auto singleChapterStack = [&](const std::string& theorem) -> std::vector<std::vector<std::string> > {
        std::vector< std::tuple<
            std::string,                    // leftExpr
            std::vector<std::string>,       // args of the current implication node
            std::set<std::string>           // node->left.arguments
        > > tempChain;

        std::string head = ce::disintegrateImplication(theorem, tempChain, this->definitions.configurations());

        std::vector<std::string> chain;
        chain.reserve(tempChain.size());
        for (std::size_t i = 0; i < tempChain.size(); ++i) {
            chain.push_back(std::get<0>(tempChain[i]));
        }

        Memory* mb = &body;
        for (std::size_t i = 0; i < chain.size(); ++i) {
            Memory* child = simpleMapStore.findChild(mb, chain[i]);
            if (child == NULL) return std::vector<std::vector<std::string> >();
            mb = child;
        }

        std::vector<std::vector<std::string> > stack;
        std::set<ExpressionWithValidity> covered;
        // D-51: insert the chapter goal (the wrapped theorem expression) into
        // the buildStack path. The existing path-cycle filter then rejects any
        // origin whose deps include the chapter goal — this is exactly the
        // self-reference shape (origin tag "implication" with the wrapped
        // theorem as a dep, which would emit a chapter row that fails the
        // verifier's self-reference check). Backtracking then falls through
        // to the contradiction-LB fallback (where the actual contradiction
        // recipe lives).
        ExpressionWithValidity chapterGoalEv(theorem, "main");
        proofReconstructor.enterChapterGoal(chapterGoalEv);
        timedBuildStack(*mb, ExpressionWithValidity(head, "main"), stack, covered);
        proofReconstructor.leaveChapterGoal(chapterGoalEv);
        return stack;
        };

    auto checkZeroStack = [&](const std::string& theorem,
        const std::string& inductionVar,
        const std::string& recCounter) -> std::vector<std::vector<std::string> > {
            std::vector< std::tuple<std::string, std::vector<std::string>, std::set<std::string> > > tempChain;
            std::string head = ce::disintegrateImplication(theorem, tempChain, this->definitions.configurations());

            std::vector<std::string> chain;
            chain.reserve(tempChain.size());
            for (std::size_t i = 0; i < tempChain.size(); ++i) {
                chain.push_back(std::get<0>(tempChain[i]));
            }

            Memory* mb = &body;
            for (std::size_t i = 0; i < chain.size(); ++i) {
                Memory* child = simpleMapStore.findChild(mb, chain[i]);
                if (child == NULL) {
                    return std::vector<std::vector<std::string> >();
                }
                mb = child;
            }

            std::vector<std::string> args0 = ce::getArgs(chain[0]);
            if (args0.size() < 2) {
                return std::vector<std::vector<std::string> >();
            }
            const std::string zeroName = args0[1];

            std::vector<std::pair<std::string, Memory*>> eqChildren;
            simpleMapStore.forEachChild(mb, [&](const gl::StrSpan& k, Memory* c) {
                eqChildren.emplace_back(std::string(k.ptr, static_cast<std::size_t>(k.len)), c);
            });
            for (const std::pair<std::string, Memory*>& kv : eqChildren) {
                const std::string& key = kv.first;
                if (!startsWith(key, std::string("(=[s(rec") + recCounter)
                    || !endsWith(key, std::string(",") + zeroName + "])")) {
                    continue;
                }

                Memory* eqNode = kv.second;
                if (eqNode == NULL) {
                    continue;
                }

                // Relaxed: head may live in the LB's full statement registry
                // (intEncodedStatements) without being in the local-origin
                // registry when it arrived via mailIn (status=3,
                // isLocal=false). A local-only gate misses this case after
                // the symmetry-disabling sequence.
                //
                // A DISCHARGED node emptied its registry at discharge —
                // the probe target is the exact captured pair set
                // (D-157). A live node (parked-
                // never-woken, or an active the final barrier's pressure
                // path dumped) reloads and walks the registry as before.
                bool inRegistry;
                if (eqNode->dischargedForever) {
                    // dischargedRegistryKeys (heap) is the probe target, but
                    // the nameMap lookup that forms its key needs the cold
                    // string tables resident; a discharged node may have been
                    // drained to SSD under pressure, so reload-for-read first
                    // (no-op if resident; sanctioned on discharged LBs,
                    // D-158). The registry stays empty.
                    eqNode->ensureLoadedForRead(lbdeload::kDeloadDirectory);
                    const NameId origId = eqNode->nameMap.lookup(head);
                    inRegistry = origId != 0
                        && eqNode->dischargedRegistryKeys.count(
                               packStatementKey(origId,
                                                NameMap::MAIN_ID)) > 0;
                }
                else {
                    eqNode->ensureLoaded(lbdeload::kDeloadDirectory);
                    inRegistry =
                        containsEncoded(eqNode->intEncodedStatements,
                                        eqNode->nameMap, head);
                }
                // A vacuous closure leaves no goal statement, only the goal's
                // history line (D-573).
                if (!inRegistry && !hasVacuousTruthAtMain(*eqNode, head)) continue;

                std::vector<std::string> ev = ce::getArgs(eqNode->exprKey());
                if (ev.empty() || ev[0] != inductionVar) {
                    continue;
                }

                std::vector<std::string> keyArgs = ce::getArgs(key);
                if (keyArgs.size() < 2) {
                    continue;
                }
                const std::string recName = keyArgs[0];

                const std::string tempExpr = std::string("(=[") + recName + "," + zeroName + "])";
                Memory* mbTarget = simpleMapStore.findChild(mb, tempExpr);
                if (mbTarget == NULL) {
                    continue;
                }
                std::vector<std::vector<std::string> > stack;
                std::set<ExpressionWithValidity> covered;
                timedBuildStack(
                    *mbTarget, ExpressionWithValidity(head, "main"), stack, covered);
                return stack;
            }
            return std::vector<std::vector<std::string> >();
        };

    auto checkInductionConditionStack = [&](const std::string& theorem,
        const std::string& inductionVar,
        const std::string& recCounter) -> std::vector<std::vector<std::string> > {
            std::vector< std::tuple<std::string, std::vector<std::string>, std::set<std::string> > > tempChain;
            std::string head = ce::disintegrateImplication(theorem, tempChain, this->definitions.configurations());

            std::vector<std::string> chain;
            chain.reserve(tempChain.size());
            for (std::size_t i = 0; i < tempChain.size(); ++i) chain.push_back(std::get<0>(tempChain[i]));

            Memory* mb = &body;
            for (std::size_t i = 0; i < chain.size(); ++i) {
                Memory* child = simpleMapStore.findChild(mb, chain[i]);
                if (child == NULL) return std::vector<std::vector<std::string> >();
                mb = child;
            }

            std::vector<std::string> args0 = ce::getArgs(chain[0]);
            if (args0.size() < 4) return std::vector<std::vector<std::string> >();
            const std::string sName = args0[2];

            std::vector<std::pair<std::string, Memory*>> in2Children;
            simpleMapStore.forEachChild(mb, [&](const gl::StrSpan& k, Memory* c) {
                in2Children.emplace_back(std::string(k.ptr, static_cast<std::size_t>(k.len)), c);
            });
            for (const std::pair<std::string, Memory*>& kv : in2Children) {
                const std::string& key = kv.first;
                if (!startsWith(key, std::string("(in2[rec") + recCounter)) continue;
                if (!endsWith(key, std::string("") + inductionVar + "," + sName + "])")) continue;

                Memory* node = kv.second;
                if (node == NULL) continue;
                // The intLocalEncodedStatementsSet probe is a heap mirror
                // (maintained at every mutation site — I-86) valid on
                // deloaded/discharged nodes, but the nameMap lookup that
                // forms its key needs the cold string tables resident —
                // reload-for-read first (no-op if resident; sanctioned on
                // discharged LBs, D-158). buildStack below
                // reads only RAM state (exprOriginMap, Rule 16).
                node->ensureLoadedForRead(lbdeload::kDeloadDirectory);
                const NameId locOrigId = node->nameMap.lookup(head);
                const bool isLocalStatement = locOrigId != 0
                    && node->intLocalEncodedStatementsSet.contains(
                        packStatementKey(locOrigId, NameMap::MAIN_ID));
                // A vacuous closure leaves no goal statement, only the goal's
                // history line (D-573).
                if (!isLocalStatement && !hasVacuousTruthAtMain(*node, head)) continue;

                std::vector<std::vector<std::string> > stack;
                std::set<ExpressionWithValidity> covered;
                timedBuildStack(
                    *node, ExpressionWithValidity(head, "main"), stack, covered);
                return stack;
            }
            return std::vector<std::vector<std::string> >();
        };

    // Induction typing chapter: walk back from (in[inductionVar, N]) in the
    // PARENT memoryBlock's exprOriginMap. fXY / fXYZ forward-chaining on a
    // positive (in2[inductionVar, …])/(in3[…, inductionVar, …, f]) premise
    // fires at the parent memoryBlock, so the derivation's origin chain lives
    // there (mail propagates statements to sub-blocks but the origin record
    // stays with the block where the rule fired). The chapter corresponds to
    // the typing sub-proof that gates induction promotion in updateGlobal.
    // See docs/agentic_swdd/induction_typing_plan.md.
    auto inductionTypingStack = [&](const std::string& theorem,
        const std::string& inductionVar,
        const std::string& /*recCounter*/) -> std::vector<std::vector<std::string> > {
            std::vector< std::tuple<std::string, std::vector<std::string>, std::set<std::string> > > tempChain;
            (void)ce::disintegrateImplication(theorem, tempChain, this->definitions.configurations());

            std::vector<std::string> chain;
            chain.reserve(tempChain.size());
            for (std::size_t i = 0; i < tempChain.size(); ++i) chain.push_back(std::get<0>(tempChain[i]));

            Memory* mb = &body;
            for (std::size_t i = 0; i < chain.size(); ++i) {
                Memory* child = simpleMapStore.findChild(mb, chain[i]);
                if (child == NULL) return std::vector<std::vector<std::string> >();
                mb = child;
            }

            // Anchor N slot (args[0] of the anchor expression in chain[0])
            std::vector<std::string> args0 = ce::getArgs(chain[0]);
            if (args0.empty()) return std::vector<std::vector<std::string> >();
            const std::string nName = args0[0];

            const std::string typingGoal =
                std::string("(in[") + inductionVar + "," + nName + "])";

            std::vector<std::vector<std::string> > stack;
            std::set<ExpressionWithValidity> covered;
            timedBuildStack(
                *mb, ExpressionWithValidity(typingGoal, "main"), stack, covered);
            return stack;
        };

    auto debugStack = [&](const std::string& pathPlusEnd) -> std::vector<std::vector<std::string>> {
        std::string::size_type firstPlus = pathPlusEnd.find('+');
        if (firstPlus == std::string::npos) return std::vector<std::vector<std::string>>();

        std::string pathPart = pathPlusEnd.substr(0, firstPlus);
        std::string validityName;
        std::string endExpr;

        std::string::size_type secondPlus = pathPlusEnd.find('+', firstPlus + 1);
        if (secondPlus != std::string::npos) {
            validityName = pathPlusEnd.substr(firstPlus + 1, secondPlus - (firstPlus + 1));
            endExpr = pathPlusEnd.substr(secondPlus + 1);
        }
        else {
            endExpr = pathPlusEnd.substr(firstPlus + 1);
        }

        Memory* mb = &body;
        if (!pathPart.empty()) {
            std::string token;
            for (std::size_t i = 0; i <= pathPart.size(); ++i) {
                if (i == pathPart.size() || pathPart[i] == ';') {
                    if (!token.empty()) {
                        Memory* child = simpleMapStore.findChild(mb, token);
                        if (child == NULL)
                            return std::vector<std::vector<std::string>>();
                        mb = child;
                        token.clear();
                    }
                }
                else {
                    token.push_back(pathPart[i]);
                }
            }
        }

        ExpressionWithValidity target(endExpr, validityName);
        std::vector<std::vector<std::string>> stack;
        std::set<ExpressionWithValidity> covered;

        timedBuildStack(*mb, target, stack, covered);

        return stack;
        };


    // ---------- emit stacks + mapping file ----------
    // Append mode: later batches extend the mapping file; 'idx' already
    // holds the scanned start offset.
    std::ofstream mapping((outDir / "global_theorem_list.txt").c_str(), std::ios::out | std::ios::app);

    int lastSingleChapterIdx = -1;
    for (std::size_t i = 0; i < theoremList.size(); ++i) {
        const std::string& name = std::get<0>(theoremList[i]);
        const std::string& methodOrig = std::get<1>(theoremList[i]);
        const std::string& var = std::get<2>(theoremList[i]);
        const std::string& recCtr = std::get<3>(theoremList[i]);

        const std::string method = toLower(methodOrig);

        // Arm buildStack's cross-chapter admissibility check for every
        // chapter of this theorem (induction triads included); inert for
        // the synthetic branches, which never call buildStack.
        exportCurrentChapterTheorem = name;

        // Weak and strong induction share the triad's three chapters; only
        // the step chapter's hypothesis row differs (D-500).
        if (method == "induction" || method == "strong induction") {
            auto cacheIt = cachedProofStacks.find(name);

            // Chapter order: typing first, then check_zero, then
            // check_induction_condition. Typing is the gate — it establishes
            // `(in[var,N])` which both subsequent chapters rely on.
            // `mapping << name …` is written once against the TYPING chapter
            // (chapter index `idx`) so the verifier pairs the induction row
            // with index `idx` and its two siblings at `idx+1`, `idx+2`.
            std::vector<std::vector<std::string> > stT =
                (cacheIt != cachedProofStacks.end()) ? cacheIt->second.stack2 : inductionTypingStack(name, var, recCtr);
            writeStackIndexed(idx, "induction_typing", name, stT);
            mapping << name << '\t' << methodOrig << '\t' << var << '\n';
            ++idx;

            std::vector<std::vector<std::string> > st0 =
                (cacheIt != cachedProofStacks.end()) ? cacheIt->second.stack0 : checkZeroStack(name, var, recCtr);
            writeStackIndexed(idx, "check_zero", name, st0);
            ++idx;

            std::vector<std::vector<std::string> > st1 =
                (cacheIt != cachedProofStacks.end()) ? cacheIt->second.stack1 : checkInductionConditionStack(name, var, recCtr);
            writeStackIndexed(idx, "check_induction_condition", name, st1);
            ++idx;
        }
        else if (method == "single chapter proof") {
            auto cacheIt = cachedProofStacks.find(name);
            std::vector<std::vector<std::string> > st =
                (cacheIt != cachedProofStacks.end()) ? cacheIt->second.stack0 : singleChapterStack(name);
            writeStackIndexed(idx, "single_chapter_proof", name, st);
            mapping << name << '\t' << methodOrig << '\t' << var << '\n';
            lastSingleChapterIdx = idx;
            ++idx;
        }
        else if (method == "proved not broadcast") {
            // Level-refused closure recorded by the proved-not-broadcast
            // tier: a real derivation exists in the producing LB, so the
            // chapter is the ordinary single-chapter walk under its own
            // chapter kind.
            std::vector<std::vector<std::string> > st = singleChapterStack(name);
            writeStackIndexed(idx, "proved_not_broadcast", name, st);
            mapping << name << '\t' << methodOrig << '\t' << var << '\n';
            ++idx;
        }
        else if (method == "debug") {
            std::vector<std::vector<std::string> > st = debugStack(name);
            writeStackIndexed(idx, "debug", name, st);
            mapping << name << '\t' << methodOrig << '\t' << var << '\n';
            ++idx;
        }
        else if (method == "reformulated statement") {
            std::vector<std::vector<std::string> > st;
            st.push_back(std::vector<std::string>());
            st.back().push_back(name);
            st.back().push_back("main");
            st.back().push_back("reformulated from");
            st.back().push_back(var);
            st.back().push_back("main");
            writeStackIndexed(idx, "reformulated_statement", name, st);
            mapping << name << '\t' << methodOrig << '\t' << var << '\n';
            ++idx;
        }
        else if (method == "incubator back reformulation") {
            std::vector<std::vector<std::string>> st;
            st.push_back(std::vector<std::string>());
            st.back().push_back(name);
            st.back().push_back("main");
            st.back().push_back("incubator back reformulation");
            st.back().push_back(var);
            st.back().push_back("main");
            writeStackIndexed(idx, "back_reformulated_statement", name, st);
            mapping << name << '\t' << methodOrig << '\t' << var << '\n';
            ++idx;
        }
        else if (method == "or theorem") {
            std::vector<std::vector<std::string>> st;
            st.push_back(std::vector<std::string>());
            st.back().push_back(name);
            st.back().push_back("main");
            st.back().push_back("or theorem");
            st.back().push_back(var);       // existence theorem
            st.back().push_back("main");
            st.back().push_back(recCtr);    // companion theorem
            st.back().push_back("main");
            writeStackIndexed(idx, "or_theorem", name, st);
            mapping << name << '\t' << methodOrig << '\t' << var << '\n';
            ++idx;
        }
        else if (method == "or elimination") {
            // Pre-split merge chapter: one fabricated row citing the two
            // guard variants (the row's aux slots) plus the licensing or
            // theorem (the seam's citation ledger — a registered merge
            // without its license is a seam bug, never a skippable state).
            const auto cit = orElimCitedOrByMerged.find(name);
            assert(cit != orElimCitedOrByMerged.end()
                && "generateRawProofGraph: or-elimination row without a recorded or-theorem license");
            std::vector<std::vector<std::string>> st;
            st.push_back(std::vector<std::string>());
            st.back().push_back(name);
            st.back().push_back("main");
            st.back().push_back("or elimination");
            st.back().push_back(var);          // guard variant A
            st.back().push_back("main");
            st.back().push_back(recCtr);       // guard variant B
            st.back().push_back("main");
            st.back().push_back(cit->second);  // licensing or theorem
            st.back().push_back("main");
            writeStackIndexed(idx, "or_elimination", name, st);
            mapping << name << '\t' << methodOrig << '\t' << var << '\n';
            ++idx;
        }
        else {
            std::vector<std::vector<std::string> > empty;
            writeStackIndexed(idx, "unknown", name, empty);
            mapping << name << '\t' << methodOrig << '\t' << var << '\n';
            ++idx;
        }

        exportCurrentChapterTheorem.clear();

        // Release the LBs this theorem's chapters reloaded from SSD images.
        // The read-only export brought them back on demand; their on-disk
        // images stay, so a later theorem revisiting an LB reloads it again.
        // Without this, the reloads accumulate across theorems to static-pool
        // exhaustion (G-53). Release DISPATCHES on the recorded format exactly
        // as the reload did: a raw-format LB (a live equality node the export
        // reloaded from its eviction image) must keep its container bookkeeping
        // for the next raw rebind — and on the extent path it has no named file
        // set at all — so the v3 release walk would be wrong on both counts.
        const auto releaseStarted = FrameClock::now();
        for (Memory* lb : exportReloaded) lb->releaseStaticBlocksDispatch();
        exportReloaded.clear();
        releaseSeconds += frameSecondsSince(releaseStarted);
        ++releasedTheorems;
    }
    g_exportReloadSink = nullptr;
    mapping.close();

    const double totalSeconds = frameSecondsSince(rawGenerationStarted);
    const double knownSeconds = binaryExportSeconds + indexSeconds
        + buildStackSeconds + serializationSeconds + releaseSeconds;
    assert(totalSeconds >= knownSeconds);
    recordFrameTiming("native.raw.binary_export", binaryExportSeconds);
    recordFrameTiming("native.raw.index", indexSeconds);
    recordFrameTiming(
        "native.raw.build_stack", buildStackSeconds,
        buildStackCalls == 0 ? 1 : buildStackCalls);
    recordFrameTiming(
        "native.raw.serialization", serializationSeconds,
        serializedChapters == 0 ? 1 : serializedChapters);
    recordFrameTiming(
        "native.raw.release", releaseSeconds,
        releasedTheorems == 0 ? 1 : releasedTheorems);
    recordFrameTiming("native.raw.other", totalSeconds - knownSeconds);
}
}
