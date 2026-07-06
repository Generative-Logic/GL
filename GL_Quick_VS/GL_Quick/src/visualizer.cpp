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

#include "prover.hpp"
#include "memory_infra/lb_deload.hpp"
#include <numeric>

namespace gl {

// =====================================================================
// D-51 (option 1 + backtracking) — path-stack cycle filter for buildStack
// origin choice.
//
// With max_origin_per_expr raised to 30 (matching compressor mode), each
// expression typically carries multiple candidate origins. buildStack:
//   1. Sorts origins per D-49-style preference: non-equality tags first
//      (preserving insertion order within group), then equality1/equality2
//      tags. Within each group, insertion order survives — matches the
//      legacy cap=1 + D-49 surviving-origin choice for the common case.
//   2. Walks candidates in that order; skips any whose deps include an
//      expression already on the current proof-tree path (would form a
//      cycle with the recursion stack — Knuth (1977) AND-OR graph
//      acyclic-derivation extraction).
//   3. Backtracks: if the recursive walk fails for a candidate's deps,
//      rolls back stack/covered/path snapshots and tries the next
//      candidate. Guarantees a valid acyclic tree if one exists.
//
// Returns true on success. False signals to the recursive caller that no
// acyclic origin exists for this node given the current path; caller
// retries with its own next candidate. Top-level callers (directStack,
// checkZeroStack, ...) ignore the return.
// =====================================================================
static thread_local std::set<ExpressionWithValidity> g_buildStackPath;

void clearBuildStackPath() { g_buildStackPath.clear(); }

// Order origins by (D-49-style preference, insertion index): non-equality
// tags first preserving insertion order, then equality1/equality2 tags
// preserving insertion order. This matches the cap=1 + D-49 surviving-
// origin choice for chapters whose proof structure is acyclic.
static std::vector<size_t> sortedOriginIndicesD49(
    const std::vector<std::pair<std::string, std::vector<ExpressionWithValidity>>>& origins)
{
    std::vector<size_t> result;
    result.reserve(origins.size());
    auto isEq = [&](size_t i) {
        return origins[i].first == "equality1" || origins[i].first == "equality2";
    };
    for (size_t i = 0; i < origins.size(); ++i) if (!isEq(i)) result.push_back(i);
    for (size_t i = 0; i < origins.size(); ++i) if (isEq(i)) result.push_back(i);
    return result;
}

// Lift (expr, validity) to the closest-to-"main" ancestor of validity whose
// (expr, ancestor) key has an origin in this LB's exprOriginMap. Per I-2 and
// NameMap::encodePush, every non-root validity is parent + "_boundary_" +
// payload, so the ancestor chain is recoverable by splitting on "_boundary_".
// Returns v unchanged if no ancestor (including v itself) has an origin —
// caller handles the miss (assertion / dump).
//
// OR-branch barrier: lifting may NOT cross "_boundary_orint_" or
// "_boundary_ordis_" delimiters. Those scopes are conditional on a disjunct
// hypothesis (or<N>-integration / -disintegration); chapter rows must preserve
// branch-distinct namespaces for the OR-family verifier checkers (or
// convergence, or branch proven, or branch assumption, or disintegration) to
// recognise the convergence/branch-proof pattern. Without the barrier, both
// branches' deps would be lifted to the parent boundary where the post-
// convergence origin lives, collapsing them to identical pairs and breaking
// `check_or_convergence`'s branch-distinctness expectation. The deepest
// orint_/ordis_ ancestor sets the shallowest allowed lift target.
static ExpressionWithValidity liftToShallowestOriginAncestor(
        const Memory& mb,
        const ExpressionWithValidity& v) {
    const std::string& s = v.validityName;
    static const std::string sep = "_boundary_";
    std::vector<std::string> ancestors;
    ancestors.reserve(8);
    ancestors.push_back("main");
    if (s.size() > 4
        && s.compare(0, 4, "main") == 0
        && s.size() >= 4 + sep.size()
        && s.compare(4, sep.size(), sep) == 0) {
        std::size_t pos = 4 + sep.size();
        while (pos <= s.size()) {
            std::size_t next = s.find(sep, pos);
            if (next == std::string::npos) {
                ancestors.push_back(s);
                break;
            }
            ancestors.push_back(s.substr(0, next));
            pos = next + sep.size();
        }
    }
    std::size_t minLift = 0;
    static const std::string kOrint = "orint_";
    static const std::string kOrdis = "ordis_";
    for (std::size_t i = 1; i < ancestors.size(); ++i) {
        const std::size_t payloadStart = ancestors[i - 1].size() + sep.size();
        if (payloadStart >= ancestors[i].size()) continue;
        const std::string& child = ancestors[i];
        const std::size_t payloadLen = child.size() - payloadStart;
        if ((payloadLen >= kOrint.size()
             && child.compare(payloadStart, kOrint.size(), kOrint) == 0)
            || (payloadLen >= kOrdis.size()
                && child.compare(payloadStart, kOrdis.size(), kOrdis) == 0)) {
            minLift = i;
        }
    }
    for (std::size_t i = minLift; i < ancestors.size(); ++i) {
        int64_t pk = 0;
        if (!lookupOriginKey(mb.originInterner, v.original, ancestors[i], pk)) {
            continue;
        }
        const int32_t oid = mb.exprOriginMap.lookup(pk);
        if (oid != 0 && mb.exprOriginMap.runLen(oid) > 0) {
            return ExpressionWithValidity(v.original, ancestors[i]);
        }
    }
    return v;
}

bool ExpressionAnalyzer::buildStack(Memory& memoryBlock,
    const ExpressionWithValidity& provedIn,
    std::vector<std::vector<std::string>>& stack,
    std::set<ExpressionWithValidity>& covered) {
    // Abortion trap — buildStack call counter. The chapter-export
    // hang on this branch is exponential candidate exploration (not
    // a real recursion cycle — verified 2026-05-25 with a depth +
    // revisit tripwire that never fired). Cap calls at 5M so the
    // process exits with a clear signal instead of hanging
    // indefinitely.
    static std::atomic<std::size_t> s_buildStackCalls{0};
    const std::size_t myCall = ++s_buildStackCalls;
    if ((myCall % 100000) == 0) {
        std::cerr << "[buildStack] call #" << myCall
                  << " LB=" << memoryBlock.exprKey()
                  << " proved=" << provedIn.original << std::endl;
    }

    if (myCall > 5000000) {
        std::cerr << "[buildStack] call cap 5M reached — aborting" << std::endl;
        std::abort();
    }

    // Post-prove READ reload (D-158): the chapter walk
    // reads this LB's origin history through its cold string tables; a
    // pressure-drained LB — discharged ones included — comes back here.
    // Defined no-op when resident. This is the sanctioned export-side
    // entry of the reload touch-point list.
    memoryBlock.ensureLoadedForRead(lbdeload::kDeloadDirectory);

    // TRIPWIRE: sentinel validity used by disintegrateExprHypothetically.
    // Hypothetical disintegration products must never reach buildStack —
    // they are throw-away structural probes and the lambda no-track path
    // already suppresses their origin writes. Check the INCOMING expression
    // before any lifting; the sentinel is structural, not a validity-stack
    // ancestor of anything legitimate.
    static const std::string kHypoDisintMarker = "_boundary_hypothetical_disintegration";
    if (provedIn.validityName.find(kHypoDisintMarker) != std::string::npos) {
        std::cerr << "[buildStack] TRIPWIRE: hypothetical-disintegration sentinel "
                  << "reached buildStack: " << provedIn.original
                  << " (v=" << provedIn.validityName << ")\n";
        assert(false && "buildStack: hypothetical disintegration sentinel leaked into proof graph");
    }

    // Lift to the closest-to-"main" ancestor of provedIn.validityName whose
    // (expr, ancestor) key has an origin in this LB's exprOriginMap. From
    // here on, `proved` is the lifted form — chapter rows are emitted at
    // the lifted scope (truthful "this is where the derivation lives"), the
    // path-cycle filter uses the lifted form, and recursion lifts each dep
    // so duplicates collapse to a single (expr, lifted_v) row per chapter.
    const ExpressionWithValidity proved = liftToShallowestOriginAncestor(memoryBlock, provedIn);

    int64_t pkProved = 0;
    bool hasOrigins = false;
    if (lookupOriginKey(memoryBlock.originInterner, proved.original,
                        proved.validityName, pkProved)) {
        const int32_t oid = memoryBlock.exprOriginMap.lookup(pkProved);
        hasOrigins = oid != 0 && memoryBlock.exprOriginMap.runLen(oid) > 0;
    }
    if (!hasOrigins) {
        // _integration_goal expressions are synthetic markers — no origin expected.
        if (proved.original.find("_integration_goal") != std::string::npos) {
            return true;
        }
        // D-51: contradiction-LB fallback. When `proved` is a negation !(X)
        // and has no direct origin in this LB (or any of its ancestors with
        // an origin entry), search the LB chain for "__contradiction__(X)"
        // and resolve locally there.
        if (proved.original.size() > 1 && proved.original[0] == '!') {
            std::string positive = proved.original.substr(1);
            std::string contraKey = "__contradiction__" + positive;
            Memory* contraLB = nullptr;
            for (Memory* anc = &memoryBlock; anc != nullptr; anc = anc->parentMemory) {
                Memory* sc = simpleMapStore.findChild(anc, contraKey);
                if (sc != nullptr) {
                    contraLB = sc;
                    break;
                }
            }
            // Self-guard (mirrors the post-candidate fallback below): when
            // the contradiction LB ITSELF lacks the record, re-entering it
            // is an infinite recursion, not a resolution — fall through to
            // the loud no-origin assert instead.
            if (contraLB != nullptr && contraLB != &memoryBlock) {
                return buildStack(*contraLB, proved, stack, covered);
            }
        }
        std::cerr << "[buildStack] no origin for: " << proved.original
                  << " | validity=" << proved.validityName
                  << " | exprKey=" << memoryBlock.exprKey() << "\n";
        assert(false && "buildStack: no origin found");
    }

    // Path-stack invariant fix: g_buildStackPath has no refcount.
    // If an OUTER scope already inserted `proved` (because this is a
    // recursive re-entry on a node further up the proof-tree path),
    // the inner scope MUST NOT erase on exit — that would wipe the
    // outer scope's entry too and break the cycle filter for the
    // outer's subsequent candidates. Track whether THIS invocation
    // is the inserter and only erase if so.
    const bool insertedHere = g_buildStackPath.insert(proved).second;

    // Decoded owned copy — every downstream consumer (D-49 candidate
    // sort, cycle filter, emitRow, the last-resort fallback) keeps its
    // historical string form.
    std::vector<std::pair<std::string, std::vector<ExpressionWithValidity>>> origins;
    {
        const std::vector<IdOrigin> idOrigins =
            memoryBlock.exprOriginMap.recordsAt(
                memoryBlock.exprOriginMap.lookup(pkProved));
        origins.reserve(idOrigins.size());
        for (const IdOrigin& o : idOrigins) {
            origins.push_back(decodeOrigin(o, memoryBlock.originInterner));
        }
    }
    auto candidateOrder = sortedOriginIndicesD49(origins);

    auto emitRow = [&](const std::pair<std::string, std::vector<ExpressionWithValidity>>& origin,
                       const std::vector<ExpressionWithValidity>& liftedDeps) {
        std::vector<std::string> row;
        row.reserve(3 + liftedDeps.size() * 2);
        row.push_back(proved.original);
        row.push_back(proved.validityName);
        row.push_back(origin.first);
        for (const auto& d : liftedDeps) {
            row.push_back(d.original);
            row.push_back(d.validityName);
        }
        stack.push_back(row);
    };

    // Candidate loop. Each candidate's deps are lifted before cycle-filter,
    // emission, and recursion so chapter cells, the path-cycle filter, and
    // the `covered` dedup all key on the same lifted form.
    for (size_t idx : candidateOrder) {
        const auto& origin = origins[idx];

        std::vector<ExpressionWithValidity> liftedDeps;
        liftedDeps.reserve(origin.second.size());
        for (const auto& d : origin.second) {
            liftedDeps.push_back(liftToShallowestOriginAncestor(memoryBlock, d));
        }

        bool cyclic = false;
        for (const auto& d : liftedDeps) {
            if (g_buildStackPath.count(d)) { cyclic = true; break; }
        }
        if (cyclic) continue;

        // Early-return tags: theorem proved in previous batch / external — no row, success.
        if (origin.first == "broadcast" || origin.first == "externally provided theorem") {
            if (insertedHere) g_buildStackPath.erase(proved);
            return true;
        }

        const size_t stackSnap = stack.size();
        const std::set<ExpressionWithValidity> coveredSnap = covered;

        emitRow(origin, liftedDeps);

        bool subtreeOk = true;
        for (const auto& ingredient : liftedDeps) {
            if (covered.insert(ingredient).second) {
                if (!buildStack(memoryBlock, ingredient, stack, covered)) {
                    subtreeOk = false; break;
                }
            }
        }

        if (subtreeOk) {
            if (insertedHere) g_buildStackPath.erase(proved);
            return true;
        }

        stack.resize(stackSnap);
        covered = coveredSnap;
    }

    // D-51: no acyclic direct origin worked. For a negated head, try the
    // contradiction-LB fallback before falling back to the degraded
    // front()-emit. Walk the LB chain for "__contradiction__(positive)";
    // if found, switch in and resolve there.
    if (proved.original.size() > 1 && proved.original[0] == '!') {
        std::string positive = proved.original.substr(1);
        std::string contraKey = "__contradiction__" + positive;
        Memory* contraLB = nullptr;
        for (Memory* anc = &memoryBlock; anc != nullptr; anc = anc->parentMemory) {
            Memory* sc = simpleMapStore.findChild(anc, contraKey);
            if (sc != nullptr) {
                contraLB = sc;
                break;
            }
        }
        if (contraLB != nullptr && contraLB != &memoryBlock) {
            if (insertedHere) g_buildStackPath.erase(proved);
            return buildStack(*contraLB, proved, stack, covered);
        }
    }

    // Last-resort fallback: emit front() and recurse on its deps fully
    // (matches pre-D-51 chapter shape — chapter has rows even if cyclic,
    // verifier flags). Deps are lifted like the per-candidate loop.
    if (!origins.empty()) {
        const auto& fb = origins.front();
        if (fb.first == "broadcast" || fb.first == "externally provided theorem") {
            if (insertedHere) g_buildStackPath.erase(proved);
            return false;
        }
        std::vector<ExpressionWithValidity> liftedDeps;
        liftedDeps.reserve(fb.second.size());
        for (const auto& d : fb.second) {
            liftedDeps.push_back(liftToShallowestOriginAncestor(memoryBlock, d));
        }
        emitRow(fb, liftedDeps);
        for (const auto& d : liftedDeps) {
            if (covered.insert(d).second) {
                (void) buildStack(memoryBlock, d, stack, covered);
            }
        }
    }
    if (insertedHere) g_buildStackPath.erase(proved);
    return false;
}




std::vector<ExpressionWithValidity> ExpressionAnalyzer::sortByValuesDesc(const std::vector<ExpressionWithValidity>& expressions,
    const std::vector<int>& values) {

    if (expressions.size() != values.size()) {
        throw std::invalid_argument("expressions and values must have the same size");
    }

    // Create pairs of (Value, OriginalIndex)
    std::vector<std::pair<int, std::size_t>> pairs;
    pairs.reserve(expressions.size());
    for (std::size_t i = 0; i < expressions.size(); ++i) {
        pairs.push_back(std::make_pair(values[i], i));
    }

    // Sort descending by value
    std::sort(pairs.begin(), pairs.end(),
        [](const std::pair<int, std::size_t>& a, const std::pair<int, std::size_t>& b) {
            return a.first > b.first;
        });

    // Reconstruct the sorted vector
    std::vector<ExpressionWithValidity> result;
    result.reserve(expressions.size());
    for (std::size_t i = 0; i < pairs.size(); ++i) {
        result.push_back(expressions[pairs[i].second]);
    }

    return result;
}


// In GL_Quick_VS/GL_Quick/src/analyze_expressions.cpp

void gl::ExpressionAnalyzer::findEnds(const std::vector<std::string>& path, const std::filesystem::path& outDirParam) {
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
        this->buildStack(*memoryBlock, endsVec[i], stack, covered);
        stackSizes.push_back(static_cast<int>(stack.size()));
    }

    // ---- order ends by size (desc) ----
    std::vector<ExpressionWithValidity> endsOrdered = this->sortByValuesDesc(endsVec, stackSizes);

    // ---- rebuild globalTheoremList ----
    this->globalTheoremList.clear();
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
        this->globalTheoremList.push_back(std::make_tuple(theoremStr, std::string("debug"), std::string("-1"), std::string("-1")));
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
        this->buildStack(*mb, targetKey, stack, covered);

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

void ExpressionAnalyzer::loadGlBinary(const std::filesystem::path& jsonPath) {
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

        // Always populate compiledExpressions so any read-side path
        // (e.g. encoded-expression resolution) sees the entry.
        compiledExpressions[name] = LogicalEntity(category, elements, signature, arity, definedSet);
        ++loadedTotal;

        // Spontaneous categories also need a repetitionExclusionMap entry
        // and contribute to counter seeding. Anchor / atomic entries are
        // batch-local and stay out of these structures.
        const bool isSpontaneous =
            (category == "implication") || (category == "existence")
            || (category == "or") || (category == "and");
        if (!isSpontaneous) continue;

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

        // Use the loaded elements vector directly as splitNK. excludeRepetitions
        // stores the splitNK in LogicalEntity::elements when allocating, so the
        // JSON's "elements" field for a spontaneous entry IS the splitNK that
        // was originally registered. We rebuild an identity reverseUnchMap over
        // the u_<i> slots; permutation variants are not pre-registered (only
        // the canonical chain order is), which is the conservative choice — the
        // worst case is that a future call computes a permuted splitNK and
        // misses the cache, falling through to the allocation path with a
        // counter that's already past the existing N, so identifiers stay
        // unique. The chapter-85 / Gauss-renames-Peano case (the bug this
        // change fixes) hits the canonical key so it is covered.
        std::map<std::string, std::string> reverseUnchMap;
        for (int i = 1; i <= arity; ++i) {
            const std::string uN = "u_" + std::to_string(i);
            reverseUnchMap[uN] = uN;
        }
        // Category included in the key so a cross-batch load can carry, e.g.,
        // an incubator-allocated implication with elements E and a main-batch
        // existence with the same elements E without the existence reusing the
        // implication's name. See repetitionExclusionMap declaration in
        // prover.hpp for the full rationale.
        repetitionExclusionMap[std::make_pair(elements, category)] = std::make_tuple(reverseUnchMap, name, elements);
        ++loadedSpontaneous;
    }

    if (maxImpl  >= 0) implCounter      = std::max(implCounter,      maxImpl  + 1);
    if (maxExist >= 0) existenceCounter = std::max(existenceCounter, maxExist + 1);
    if (maxAnd   >= 0) andCounter       = std::max(andCounter,       maxAnd   + 1);
    if (maxOr    >= 0) orCounter        = std::max(orCounter,        maxOr    + 1);

#if 0
    std::cout << "[loadGlBinary] loaded " << loadedTotal << " entries ("
              << loadedSpontaneous << " spontaneous) from " << jsonPath
              << " — counters seeded: implCounter=" << implCounter
              << " existenceCounter=" << existenceCounter
              << " andCounter=" << andCounter
              << " orCounter=" << orCounter << std::endl;
#endif
}


void ExpressionAnalyzer::exportCompiledExpressionsJSON(const std::filesystem::path& outDir) {
    namespace fs = std::filesystem;
    fs::create_directories(outDir);

    fs::path outPath = outDir / ("GL_binary_" + anchorID_ + ".json");
    nlohmann::json root = nlohmann::json::object();

    for (const auto& kv : this->compiledExpressions) {
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

        root[coreName] = entry;
    }

    std::ofstream f(outPath.string().c_str());
    if (f.is_open()) {
        f << root.dump(2);
        f.close();
    }
}

void ExpressionAnalyzer::generateRawProofGraph(
    const std::vector<std::tuple<std::string, std::string, std::string, std::string>>& theoremList,
    const std::filesystem::path& outDirParam)
{
    namespace fs = std::filesystem;

    // D-51 (option 1): reset thread_local path stack at run start. Each
    // chapter emission pushes/pops independently — the clear here is a
    // belt-and-suspenders against any leak across runs.
    clearBuildStackPath();

    // Per-chapter reload release (G-53): the export
    // reloads LBs from their SSD images on demand; without releasing them the
    // resident block set grows monotonically across the walk to static-pool
    // exhaustion. Every reload appends to this sink (Memory::reloadFromImage);
    // each theorem's reloaded LBs are released after its chapters are written
    // (loop below). The on-disk image stays, so a later theorem that revisits
    // an LB reloads it again on demand.
    std::vector<Memory*> exportReloaded;
    g_exportReloadSink = &exportReloaded;

    std::cout << "Number proven theorems: "
        << theoremList.size() / 2 << "\n";

    // ---------- out dir: "files/raw_proof_graph" by default ----------
    fs::path outDir = outDirParam.empty() ? (fs::path("files") / "raw_proof_graph") : outDirParam;
    std::error_code ec;

    // MODIFIED: Removed fs::remove_all to preserve previous runs
    fs::create_directories(outDir, ec);

    // Single canonical GL-binary folder regardless of per-batch outDir
    // (mirrors prover.cpp::compileCoreExpressionMap reader path). Without
    // this, configs that override raw_proof_graph_folder (incubator batches
    // set it to files/incubator/raw_proof_graph) would land the export in
    // files/incubator/GL_binaries/ — a directory the loader and the Python
    // _merge_into_shared step never read from. See D-54.
    fs::path glBinDir = fs::path(__FILE__).parent_path()
                        .parent_path().parent_path().parent_path()
                      / "files" / "GL_binaries";
    this->exportCompiledExpressionsJSON(glBinDir);

    // MODIFIED: Scan for start index based on existing files
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

    auto toLower = [](std::string s) {
        for (std::size_t i = 0; i < s.size(); ++i) {
            s[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(s[i])));
        }
        return s;
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
        const int16_t origId = nm.lookup(expr);
        if (origId == 0) return false;
        for (int32_t i = 0; i < static_cast<int32_t>(vec.size()); ++i) {
            if (vec[i].originalId == origId
                && vec[i].validityId == NameMap::MAIN_ID) return true;
        }
        return false;
        };

    auto writeStackIndexed = [&](int idx, const std::string& part,
        const std::vector<std::vector<std::string>>& stackRows) {

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
        };

    // ... (directStack, checkZeroStack, checkInductionConditionStack, debugStack definitions omitted for brevity - they are unchanged) ...
    // Note: You must include the unchanged lambdas here for the code to compile. 
    // I am omitting them here only to focus on the logic changes requested.
    // Copy them from your original file.

    // REDEFINING LAMBDAS FOR COMPLETENESS OF THE SNIPPET:
    // Updated: Returns vector<vector<string>>
    auto directStack = [&](const std::string& theorem) -> std::vector<std::vector<std::string> > {
        std::vector< std::tuple<
            std::string,                    // leftExpr
            std::vector<std::string>,       // args of the current implication node
            std::set<std::string>           // node->left.arguments
        > > tempChain;

        std::string head = ce::disintegrateImplication(theorem, tempChain, coreExpressionMap);

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
        g_buildStackPath.insert(chapterGoalEv);
        this->buildStack(*mb, ExpressionWithValidity(head, "main"), stack, covered);
        g_buildStackPath.erase(chapterGoalEv);
        return stack;
        };

    auto checkZeroStack = [&](const std::string& theorem,
        const std::string& inductionVar,
        const std::string& recCounter) -> std::vector<std::vector<std::string> > {
            std::vector< std::tuple<std::string, std::vector<std::string>, std::set<std::string> > > tempChain;
            std::string head = ce::disintegrateImplication(theorem, tempChain, coreExpressionMap);

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
                    const int16_t origId = eqNode->nameMap.lookup(head);
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
                if (!inRegistry) continue;

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
                this->buildStack(*mbTarget, ExpressionWithValidity(head, "main"), stack, covered);
                return stack;
            }
            return std::vector<std::vector<std::string> >();
        };

    auto checkInductionConditionStack = [&](const std::string& theorem,
        const std::string& inductionVar,
        const std::string& recCounter) -> std::vector<std::vector<std::string> > {
            std::vector< std::tuple<std::string, std::vector<std::string>, std::set<std::string> > > tempChain;
            std::string head = ce::disintegrateImplication(theorem, tempChain, coreExpressionMap);

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
                const int16_t locOrigId = node->nameMap.lookup(head);
                if (locOrigId == 0) continue;
                if (!node->intLocalEncodedStatementsSet.contains(
                        packStatementKey(locOrigId, NameMap::MAIN_ID))) continue;

                std::vector<std::vector<std::string> > stack;
                std::set<ExpressionWithValidity> covered;
                this->buildStack(*node, ExpressionWithValidity(head, "main"), stack, covered);
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
            (void)ce::disintegrateImplication(theorem, tempChain, coreExpressionMap);

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
            this->buildStack(*mb, ExpressionWithValidity(typingGoal, "main"), stack, covered);
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

        this->buildStack(*mb, target, stack, covered);

        return stack;
        };


    // ---------- emit stacks + mapping file ----------
    // MODIFIED: Open in Append Mode
    std::ofstream mapping((outDir / "global_theorem_list.txt").c_str(), std::ios::out | std::ios::app);

    // Note: 'idx' is already initialized to the correct start offset above

    int lastDirectIdx = -1;
    for (std::size_t i = 0; i < theoremList.size(); ++i) {
        const std::string& name = std::get<0>(theoremList[i]);
        const std::string& methodOrig = std::get<1>(theoremList[i]);
        const std::string& var = std::get<2>(theoremList[i]);
        const std::string& recCtr = std::get<3>(theoremList[i]);

        const std::string method = toLower(methodOrig);

        if (method == "induction") {
            auto cacheIt = cachedProofStacks.find(name);

            // Chapter order: typing first, then check_zero, then
            // check_induction_condition. Typing is the gate — it establishes
            // `(in[var,N])` which both subsequent chapters rely on.
            // `mapping << name …` is written once against the TYPING chapter
            // (chapter index `idx`) so the verifier pairs the induction row
            // with index `idx` and its two siblings at `idx+1`, `idx+2`.
            std::vector<std::vector<std::string> > stT =
                (cacheIt != cachedProofStacks.end()) ? cacheIt->second.stack2 : inductionTypingStack(name, var, recCtr);
            writeStackIndexed(idx, "induction_typing", stT);
            mapping << name << '\t' << methodOrig << '\t' << var << '\n';
            ++idx;

            std::vector<std::vector<std::string> > st0 =
                (cacheIt != cachedProofStacks.end()) ? cacheIt->second.stack0 : checkZeroStack(name, var, recCtr);
            writeStackIndexed(idx, "check_zero", st0);
            ++idx;

            std::vector<std::vector<std::string> > st1 =
                (cacheIt != cachedProofStacks.end()) ? cacheIt->second.stack1 : checkInductionConditionStack(name, var, recCtr);
            writeStackIndexed(idx, "check_induction_condition", st1);
            ++idx;
        }
        else if (method == "direct") {
            auto cacheIt = cachedProofStacks.find(name);
            std::vector<std::vector<std::string> > st =
                (cacheIt != cachedProofStacks.end()) ? cacheIt->second.stack0 : directStack(name);
            writeStackIndexed(idx, "direct_proof", st);
            mapping << name << '\t' << methodOrig << '\t' << var << '\n';
            lastDirectIdx = idx;
            ++idx;
        }
        else if (method == "debug") {
            std::vector<std::vector<std::string> > st = debugStack(name);
            writeStackIndexed(idx, "debug", st);
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
            writeStackIndexed(idx, "reformulated_statement", st);
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
            writeStackIndexed(idx, "back_reformulated_statement", st);
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
            writeStackIndexed(idx, "or_theorem", st);
            mapping << name << '\t' << methodOrig << '\t' << var << '\n';
            ++idx;
        }
        else {
            std::vector<std::vector<std::string> > empty;
            writeStackIndexed(idx, "unknown", empty);
            mapping << name << '\t' << methodOrig << '\t' << var << '\n';
            ++idx;
        }

        // Release the LBs this theorem's chapters reloaded from SSD images.
        // The read-only export brought them back on demand; their on-disk
        // images stay, so a later theorem revisiting an LB reloads it again.
        // Without this, the reloads accumulate across theorems to static-pool
        // exhaustion (G-53).
        for (Memory* lb : exportReloaded) lb->releaseStaticBlocks();
        exportReloaded.clear();
    }
    g_exportReloadSink = nullptr;
    mapping.close();
}

} // namespace gl
