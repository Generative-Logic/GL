/* Generative Logic : A deterministic reasoning and knowledge generation engine.
 Copyright(C) 2025 Generative Logic UG(haftungsbeschränkt)

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

namespace gl {

void ExpressionAnalyzer::buildStack(Memory& memoryBlock,
    const ExpressionWithValidity& proved,
    std::vector<std::vector<std::string>>& stack,
    std::set<ExpressionWithValidity>& covered) {
    // Lookup origin list for `proved`: try exact ns first, then fallback to "main"
    auto it = memoryBlock.exprOriginMap.find(proved);
    if (it == memoryBlock.exprOriginMap.end() || it->second.empty()) {
        if (proved.validityName != "main") {
            ExpressionWithValidity mainFallback(proved.original, "main");
            it = memoryBlock.exprOriginMap.find(mainFallback);
        }
        if (it == memoryBlock.exprOriginMap.end() || it->second.empty()) {
            // _integration_goal expressions are synthetic markers — no origin expected
            if (proved.original.find("_integration_goal") != std::string::npos) {
                return;
            }
            assert(false && "buildStack: no origin found");
        }
    }

    // Type adjustment: grab the first path from the vector to maintain exact same logic
    const std::pair<std::string, std::vector<ExpressionWithValidity>>& origin = it->second.front();

    if (origin.first == "broadcast" || origin.first == "externally provided theorem") {
        return;  // theorem proved in previous batch or externally provided, no local proof stack
    }

    // Push one row: [proved] + origin
    std::vector<std::string> row;
    row.reserve(1 + origin.second.size());
    row.push_back(proved.original);
    row.push_back(proved.validityName);
    row.push_back(origin.first);

    // CHANGED: Start from 0 to include the first element (rule/source)
    for (std::size_t i = 0; i < origin.second.size(); ++i) {
        row.push_back(origin.second[i].original);
        row.push_back(origin.second[i].validityName);
    }
    stack.push_back(row);

    // Recurse for each ingredient in origin
    // CHANGED: Start from 0 to recurse on the first element as well
    if (origin.first == "contradiction" && origin.second.size() >= 3) {
        // Navigate into the contradiction LB child to trace deps
        const std::string& cleanOp = origin.second[2].original;
        std::string contraKey = "__contradiction__" + cleanOp;
        auto childIt = memoryBlock.simpleMap.find(contraKey);
        if (childIt != memoryBlock.simpleMap.end() && childIt->second != nullptr) {
            Memory& contraLB = *childIt->second;
            for (std::size_t i = 0; i < origin.second.size(); ++i) {
                const ExpressionWithValidity& ingredient = origin.second[i];
                if (covered.insert(ingredient).second) {
                    buildStack(contraLB, ingredient, stack, covered);
                }
            }
        }
    } else {
        for (std::size_t i = 0; i < origin.second.size(); ++i) {
            const ExpressionWithValidity& ingredient = origin.second[i];
            // only visit once
            if (covered.insert(ingredient).second) {
                buildStack(memoryBlock, ingredient, stack, covered);
            }
        }
    }
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
        std::map<std::string, Memory*>::iterator it = memoryBlock->simpleMap.find(elt);
        if (it == memoryBlock->simpleMap.end() || it->second == NULL) {
            return; // path invalid
        }
        memoryBlock = it->second;
    }

    // ---- collect candidate ends (keys of exprOriginMap) ----
    std::set<ExpressionWithValidity> allExprs;
    auto itE = memoryBlock->exprOriginMap.begin();
    for (; itE != memoryBlock->exprOriginMap.end(); ++itE) {
        allExprs.insert(itE->first);
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
                    std::map<std::string, Memory*>::iterator itChild = mb->simpleMap.find(node);
                    if (itChild == mb->simpleMap.end() || itChild->second == NULL) {
                        mb = NULL; break;
                    }
                    mb = itChild->second;
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

    std::cout << "Number proven theorems: "
        << theoremList.size() / 2 << "\n";

    // ---------- out dir: "files/raw_proof_graph" by default ----------
    fs::path outDir = outDirParam.empty() ? (fs::path("files") / "raw_proof_graph") : outDirParam;
    std::error_code ec;

    // MODIFIED: Removed fs::remove_all to preserve previous runs
    fs::create_directories(outDir, ec);

    fs::path glBinDir = outDir.parent_path() / "GL_binaries";
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

    auto containsEncoded = [](const std::vector<EncodedExpression>& vec, const std::string& expr) -> bool {
        EncodedExpression needle(expr, "main");
        for (std::size_t i = 0; i < vec.size(); ++i) {
            if (vec[i] == needle) return true;
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
            std::map<std::string, Memory*>::iterator it = mb->simpleMap.find(chain[i]);
            if (it == mb->simpleMap.end() || it->second == NULL) return std::vector<std::vector<std::string> >();
            mb = it->second;
        }

        std::vector<std::vector<std::string> > stack;
        std::set<ExpressionWithValidity> covered;
        this->buildStack(*mb, ExpressionWithValidity(head, "main"), stack, covered);
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
                std::map<std::string, Memory*>::iterator it = mb->simpleMap.find(chain[i]);
                if (it == mb->simpleMap.end() || it->second == NULL) return std::vector<std::vector<std::string> >();
                mb = it->second;
            }

            std::vector<std::string> args0 = ce::getArgs(chain[0]);
            if (args0.size() < 2) return std::vector<std::vector<std::string> >();
            const std::string zeroName = args0[1];

            for (std::map<std::string, Memory*>::iterator it = mb->simpleMap.begin();
                it != mb->simpleMap.end(); ++it) {
                const std::string& key = it->first;
                if (!startsWith(key, std::string("(=[s(rec") + recCounter)
                    || !endsWith(key, std::string(",") + zeroName + "])")) {
                    continue;
                }

                Memory* eqNode = it->second;
                if (eqNode == NULL) continue;

                if (!containsEncoded(eqNode->localEncodedStatements, head)) continue;

                std::vector<std::string> ev = ce::getArgs(eqNode->exprKey);
                if (ev.empty() || ev[0] != inductionVar) continue;

                std::vector<std::string> keyArgs = ce::getArgs(key);
                if (keyArgs.size() < 2) continue;
                const std::string recName = keyArgs[0];

                const std::string tempExpr = std::string("(=[") + recName + "," + zeroName + "])";
                std::map<std::string, Memory*>::iterator it2 = mb->simpleMap.find(tempExpr);
                if (it2 == mb->simpleMap.end() || it2->second == NULL) continue;

                Memory* mbTarget = it2->second;
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
                std::map<std::string, Memory*>::iterator it = mb->simpleMap.find(chain[i]);
                if (it == mb->simpleMap.end() || it->second == NULL) return std::vector<std::vector<std::string> >();
                mb = it->second;
            }

            std::vector<std::string> args0 = ce::getArgs(chain[0]);
            if (args0.size() < 4) return std::vector<std::vector<std::string> >();
            const std::string sName = args0[2];

            for (std::map<std::string, Memory*>::iterator it = mb->simpleMap.begin();
                it != mb->simpleMap.end(); ++it) {
                const std::string& key = it->first;
                if (!startsWith(key, std::string("(in2[rec") + recCounter)) continue;
                if (!endsWith(key, std::string("") + inductionVar + "," + sName + "])")) continue;

                Memory* node = it->second;
                if (node == NULL) continue;
                if (!containsEncoded(node->localEncodedStatements, head)) continue;

                std::vector<std::vector<std::string> > stack;
                std::set<ExpressionWithValidity> covered;
                this->buildStack(*node, ExpressionWithValidity(head, "main"), stack, covered);
                return stack;
            }
            return std::vector<std::vector<std::string> >();
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
                        auto it = mb->simpleMap.find(token);
                        if (it == mb->simpleMap.end() || it->second == NULL)
                            return std::vector<std::vector<std::string>>();
                        mb = it->second;
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
            std::vector<std::vector<std::string> > st0 = checkZeroStack(name, var, recCtr);
            writeStackIndexed(idx, "check_zero", st0);
            mapping << name << '\t' << methodOrig << '\t' << var << '\n';
            ++idx;

            std::vector<std::vector<std::string> > st1 = checkInductionConditionStack(name, var, recCtr);
            writeStackIndexed(idx, "check_induction_condition", st1);
            ++idx;
        }
        else if (method == "direct") {
            std::vector<std::vector<std::string> > st = directStack(name);
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
        else if (method == "mirrored statement") {
            std::vector<std::vector<std::string> > st;
            st.push_back(std::vector<std::string>());
            st.back().push_back(name);
            st.back().push_back("main");
            st.back().push_back("mirrored from");
            st.back().push_back(var);
            st.back().push_back("main");
            writeStackIndexed(idx, "mirrored_statement", st);
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
        else {
            std::vector<std::vector<std::string> > empty;
            writeStackIndexed(idx, "unknown", empty);
            mapping << name << '\t' << methodOrig << '\t' << var << '\n';
            ++idx;
        }
    }
    mapping.close();
}

} // namespace gl
