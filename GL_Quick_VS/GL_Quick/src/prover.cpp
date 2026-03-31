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
#include "parameters.hpp"
#include <iostream>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <utility>
#include <filesystem>
#include <stdexcept>
#include <sstream>
#include <unordered_set>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include "compressor.hpp"
#include <cmath>

namespace gl {

    class SimpleBarrier {
    public:
        explicit SimpleBarrier(unsigned n) : n_(n) {}

        void arrive_and_wait(std::function<void()> last_fn) {
            std::unique_lock<std::mutex> lk(m_);
            const unsigned g = gen_;
            if (++arrived_ == n_) {
                arrived_ = 0;
                ++gen_;
                lk.unlock();              // run outside the lock
                if (last_fn) last_fn();
                cv_.notify_all();
            }
            else {
                cv_.wait(lk, [&] { return gen_ != g; });
            }
        }

    private:
        std::mutex m_;
        std::condition_variable cv_;
        unsigned n_;
        unsigned arrived_ = 0;
        unsigned gen_ = 0;
    };

    //Declarations
    static std::map<std::pair<int, int>, std::vector<std::vector<int>>> generateAllMappings(int n, int m);

    // Assumes n, m > 0.
    // Returns a map from (i, j) with 1 <= i < n and 1 <= j < m
    // to the list of all functions [0..i-1] -> [0..j-1], each function
    // represented as a list<int> of length i whose entries are in [0, j).
    std::map<std::pair<int, int>, std::vector<std::vector<int>>> generateAllMappings(int n, int m) {
        std::map<std::pair<int, int>, std::vector<std::vector<int>>> functionMap;

        for (int i = 1; i < n; ++i) {
            for (int j = 1; j < m; ++j) {
                // total number of functions is j^i
                long long total = 1;
                for (int t = 0; t < i; ++t) {
                    total *= static_cast<long long>(j);
                }

                std::vector<std::vector<int>> functions;
                functions.reserve(static_cast<std::size_t>(total));

                // Mixed-radix counting in base j over i positions.
                // Order matches Python itertools.product(range(j), repeat=i):
                // leftmost position changes slowest, rightmost fastest.
                for (long long index = 0; index < total; ++index) {
                    std::vector<int> f(static_cast<std::size_t>(i), 0);
                    long long value = index;
                    for (int pos = i - 1; pos >= 0; --pos) {
                        f[static_cast<std::size_t>(pos)] = static_cast<int>(value % j);
                        value /= j;
                    }
                    functions.push_back(f);
                }

                functionMap[std::make_pair(i, j)] = functions;
            }
        }

        return functionMap;
    }

// Global state analogous to the Python module-level variables.
static Memory* gBody = nullptr;
static std::ofstream gQuickFile;

// Small helper to print to both cout and quick_mode.txt
static void logLine(const std::string& s) {
    std::cout << s << std::endl;
    if (gQuickFile.is_open()) gQuickFile << s << '\n';
}







// Default constructor definition
ExpressionAnalyzer::ExpressionAnalyzer(std::string anchorID)
    :parameters(),
    body(),
    allBinariesAna(),
    allMappingsAna(),
    maxNumLeafsPerKey(0),
    allPermutationsAna(),
    globalDependencies(),
    inductionMemoryBlocks(),
    contradictionTable(),
    index(),
    boxes(),
    indexCE(),
    boxesCE(),
	expressionsFromConfig()
{
    std::string configFileName = "Config" + anchorID + ".json";

    // Adjust paths as needed for your specific execution environment
    std::vector<std::string> searchPaths = {
        "files/config/" + configFileName,
        "../../../files/config/" + configFileName,
        "../files/config/" + configFileName,
        "GL/files/config/" + configFileName
    };

    // ---------------------------------------------------------
    // LOAD JSON PARAMETERS (Using existing path logic)
    // ---------------------------------------------------------
    namespace fs = std::filesystem;

    // Path logic matches readSimpleFacts / saveProvedTheorems
    fs::path configPath = fs::path(__FILE__).parent_path().parent_path().parent_path().parent_path()
        / "files" / "config" / ("Config" + anchorID + ".json");

    if (fs::exists(configPath)) {
        try {
            std::ifstream f(configPath);
            nlohmann::json j;
            f >> j;

            if (j.contains("prover_parameters")) {
                auto& pp = j["prover_parameters"];
                if (pp.contains("sizeAllBinariesAna")) parameters.sizeAllBinariesAna = pp["sizeAllBinariesAna"];
                if (pp.contains("maxIterationNumberProof")) parameters.maxIterationNumberProof = pp["maxIterationNumberProof"];
                if (pp.contains("numberIterationsConjectureFiltering")) parameters.numberIterationsConjectureFiltering = pp["numberIterationsConjectureFiltering"];
                if (pp.contains("maxSizeDefSetMapping")) parameters.maxSizeDefSetMapping = pp["maxSizeDefSetMapping"];
                if (pp.contains("maxSizeTargetSetMapping")) parameters.maxSizeTargetSetMapping = pp["maxSizeTargetSetMapping"];
                if (pp.contains("maxNumberSecondaryVariables")) parameters.maxNumberSecondaryVariables = pp["maxNumberSecondaryVariables"];
                if (pp.contains("sizeAllPermutationsAna")) parameters.sizeAllPermutationsAna = pp["sizeAllPermutationsAna"];
                if (pp.contains("minNumOperatorsKey")) parameters.minNumOperatorsKey = pp["minNumOperatorsKey"];
                if (pp.contains("minNumOperatorsKeyCE")) parameters.minNumOperatorsKeyCE = pp["minNumOperatorsKeyCE"];
                if (pp.contains("maxIterationNumberVariable")) parameters.maxIterationNumberVariable = pp["maxIterationNumberVariable"];
                if (pp.contains("standardMaxSecondaryNumber")) parameters.standardMaxSecondaryNumber = pp["standardMaxSecondaryNumber"];
                if (pp.contains("trackHistory")) parameters.trackHistory = pp["trackHistory"];
                if (pp.contains("standardMaxAdmissionDepth")) parameters.standardMaxAdmissionDepth = pp["standardMaxAdmissionDepth"];
                if (pp.contains("inductionMaxAdmissionDepth")) parameters.inductionMaxAdmissionDepth = pp["inductionMaxAdmissionDepth"];
                if (pp.contains("inductionMaxSecondaryNumber")) parameters.inductionMaxSecondaryNumber = pp["inductionMaxSecondaryNumber"];
                if (pp.contains("counterExampleBoundary")) parameters.counterExampleBoundary = pp["counterExampleBoundary"];
                if (pp.contains("minLenLongKey")) parameters.minLenLongKey = pp["minLenLongKey"];
                if (pp.contains("maxLenHypoKey")) parameters.maxLenHypoKey = pp["maxLenHypoKey"];
                if (pp.contains("debug")) parameters.debug = pp["debug"];
                if (pp.contains("compressor_mode")) parameters.compressor_mode = pp["compressor_mode"];
                if (pp.contains("ban_disintegration")) parameters.ban_disintegration = pp["ban_disintegration"];
                if (pp.contains("max_origin_per_expr")) parameters.max_origin_per_expr = pp["max_origin_per_expr"];
                if (pp.contains("compressor_max_origins_per_expr")) parameters.compressor_max_origins_per_expr = pp["compressor_max_origins_per_expr"];
                if (pp.contains("compressor_hash_bursts")) parameters.compressor_hash_bursts = pp["compressor_hash_bursts"];
                if (pp.contains("try_contradiction")) parameters.try_contradiction = pp["try_contradiction"];
                if (pp.contains("skip_ce_filter")) parameters.skip_ce_filter = pp["skip_ce_filter"];
                if (pp.contains("skip_eq_classes")) parameters.skip_eq_classes = pp["skip_eq_classes"];
                if (pp.contains("incubator_mode")) parameters.incubator_mode = pp["incubator_mode"];
                if (pp.contains("max_partition_size")) parameters.max_partition_size = pp["max_partition_size"];
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Warning: Failed to load config from " << configPath << ": " << e.what() << std::endl;
        }
    }

    std::cout << "[INIT] generating binary sequences..." << std::endl;
    std::map<int, std::vector<std::vector<int> > > binarySeqsMap;
    for (int num = 0; num <= parameters.sizeAllBinariesAna; ++num) {
        binarySeqsMap[num] = ce::generateBinarySequencesAsLists(num);
    }
    allBinariesAna = binarySeqsMap;

    std::cout << "[INIT] generating all mappings (defSet=" << parameters.maxSizeDefSetMapping
              << ", targetSet=" << parameters.maxSizeTargetSetMapping << ")..." << std::endl;
    allMappingsAna = generateAllMappings(
        parameters.maxSizeDefSetMapping,
        parameters.maxSizeTargetSetMapping
    );

    maxNumLeafsPerKey = 0;

    std::cout << "[INIT] generating permutations (size=" << parameters.sizeAllPermutationsAna << ")..." << std::endl;
    allPermutationsAna = ce::generateAllPermutations(
        parameters.sizeAllPermutationsAna
    );

    implCounter = 0;
    existenceCounter = 0;
    statementCounter = 0;
    variableCounter = 0;

    Dependencies dependencyTable;
    globalDependencies = dependencyTable;

    std::cout << "[INIT] compiling core expression map..." << std::endl;
    coreExpressionMap = ce::modifyCoreExpressionMap(anchorID);
    std::cout << "[INIT] core expression map done, " << coreExpressionMap.size() << " entries." << std::endl;

    operators.clear();
    for (const auto& kv : coreExpressionMap) {
        // Python equivalent: if input_args and output_args
        if (!kv.second.inputArgs.empty() && !kv.second.outputArgs.empty()) {
            operators.insert(kv.first);
        }
    }



    this->anchorID_ = anchorID;

    // Check for explicit anchor_name override in config JSON
    std::string effectiveAnchorID = anchorID;
    if (fs::exists(configPath)) {
        try {
            std::ifstream f2(configPath);
            nlohmann::json j2;
            f2 >> j2;
            if (j2.contains("anchor_name")) {
                std::string anchorName = j2["anchor_name"].get<std::string>();
                // anchor_name is the full key (e.g. "AnchorIncubator"),
                // but initAnchor prepends "Anchor", so strip it
                if (anchorName.substr(0, 6) == "Anchor") {
                    effectiveAnchorID = anchorName.substr(6);
                }
            }
        } catch (...) {}
    }
    std::cout << "[INIT] initAnchor (effectiveID=" << effectiveAnchorID << ")..." << std::endl;
    anchorInfo = initAnchor(coreExpressionMap, effectiveAnchorID);
    std::cout << "[INIT] initAnchor done." << std::endl;

    permanentBodies.clear();
    permanentBodies.push_back(&body);

    logicalCores = std::max(1u, std::thread::hardware_concurrency());
    //logicalCores = 1;
    assert(logicalCores >= 1);

    compileCoreExpressionMap();

}

//#pragma optimize("", off)
std::string ExpressionAnalyzer::smoothenExpr(const std::string& expr) {
    struct Helper {
        static void deleteTree(ce::TreeNode1* n) {
            if (n == NULL) { return; }
            deleteTree(n->left);
            deleteTree(n->right);
            delete n;
        }

        static void nodeToStr(const ce::TreeNode1* node, std::string& out, bool& foundAnd) {
            if (node == NULL) {
                return;
            }
            const std::string& v = node->value;

            if (!v.empty() && v[0] == '>') {
                if (node->left != NULL && node->left->value == "&") {
                    foundAnd = true;

                    std::string leftLeftExpr = ce::treeToExpr(node->left->left);
                    std::string leftRightExpr = ce::treeToExpr(node->left->right);
                    std::string rightExpr = ce::treeToExpr(node->right);

                    std::string tempExpr = "(";
                    tempExpr += ">[]";
                    tempExpr += leftRightExpr;
                    tempExpr += rightExpr;
                    tempExpr += ")";

                    out += "(";
                    out += v;
                    out += leftLeftExpr;
                    out += tempExpr;
                    // fall through to the common ')' append below; no recursion on children in this case
                }
                else {
                    out += "(";
                    out += v;
                    nodeToStr(node->left, out, foundAnd);
                    nodeToStr(node->right, out, foundAnd);
                }
            }
            else if (v == "&") {
                out += "(&";
                nodeToStr(node->left, out, foundAnd);
                nodeToStr(node->right, out, foundAnd);
            }
            else if (v.size() >= 2 && v[0] == '!' && v[1] == '>') {
                out += "!(";
                out += v.substr(1); // strip leading '!' from value like "!>..."
                nodeToStr(node->left, out, foundAnd);
                nodeToStr(node->right, out, foundAnd);
            }
            else if (v == "!&") {
                out += "!(&";
                nodeToStr(node->left, out, foundAnd);
                nodeToStr(node->right, out, foundAnd);
            }
            else if (!v.empty() && v[0] == '!') {
                // value like "!( ... )" — take inside between "!(" and trailing ")"
                out += "!(";
                if (v.size() >= 3) {
                    out += v.substr(2, v.size() - 3);
                }
            }
            else {
                out += "(";
                out += v;
            }

            out.push_back(')');
        }
    };

    bool found = true;
    std::string smoothenedExpr = expr;

    while (found) {
        // smoothen_one_and(smoothenedExpr)
        ce::TreeNode1* root = ce::parseExpr(smoothenedExpr);

        std::string localExpr;
        bool foundAnd = false;
        Helper::nodeToStr(root, localExpr, foundAnd);

        Helper::deleteTree(root);

        smoothenedExpr = localExpr;
        found = foundAnd;
    }

    return smoothenedExpr;
}

std::vector<std::string> ExpressionAnalyzer::groomSubexpr(const std::string& s, int side, std::size_t& index) {
    std::vector<std::string> subexprList;

    if (s[index] == '(') {
        index += 1;
        if (index < s.size() && s[index] == '>') {
            index += 1;
            std::vector<std::string> argsToRemove = ce::getArgs(s.substr(index));
            std::size_t close = s.find(']', index);
            if (close == std::string::npos) {
                throw std::runtime_error("No closing ']'.");
            }
            index = close + 1;

            std::vector<std::string> left = groomSubexpr(s, 0, index);
            std::vector<std::string> right = groomSubexpr(s, side, index);

            std::string argsJoined = ce::joinWithComma(argsToRemove);
            for (std::size_t i = 0; i < left.size(); ++i) {
                for (std::size_t j = 0; j < right.size(); ++j) {
                    std::string t;
                    t.reserve(4 + argsJoined.size() + left[i].size() + right[j].size());
                    t += "(>[";
                    t += argsJoined;
                    t += "]";
                    t += left[i];
                    t += right[j];
                    t += ")";
                    subexprList.push_back(t);
                }
            }
        }
        else if (index < s.size() && s[index] == '&') {
            index += 1;

            std::vector<std::string> left = groomSubexpr(s, side, index);
            std::vector<std::string> right = groomSubexpr(s, side, index);

            if (side) {
                subexprList.reserve(left.size() + right.size());
                for (std::size_t i = 0; i < left.size(); ++i) subexprList.push_back(left[i]);
                for (std::size_t j = 0; j < right.size(); ++j) subexprList.push_back(right[j]);
            }
            else {
                std::string t;
                t.reserve(3 + left[0].size() + right[0].size());
                t += "(&";
                t += left[0];
                t += right[0];
                t += ")";
                subexprList.push_back(t);
            }
        }
        else {
            std::size_t endIndex = s.find(')', index);
            if (endIndex == std::string::npos) {
                throw std::runtime_error("No closing ')'.");
            }
            std::string nodeLabel = s.substr(index, endIndex - index);
            index = endIndex;
            std::string t;
            t.reserve(2 + nodeLabel.size());
            t.push_back('(');
            t += nodeLabel;
            t.push_back(')');
            subexprList.push_back(t);
        }
    }
    else if (index + 1 < s.size() && s[index] == '!' && s[index + 1] == '(') {
        index += 2;
        if (index < s.size() && s[index] == '>') {
            index += 1;
            std::vector<std::string> argsToRemove = ce::getArgs(s.substr(index));
            std::size_t close = s.find(']', index);
            if (close == std::string::npos) {
                throw std::runtime_error("No closing ']'.");
            }
            index = close + 1;

            std::vector<std::string> left = groomSubexpr(s, 0, index);
            std::vector<std::string> right = groomSubexpr(s, 0, index);

            std::string argsJoined = ce::joinWithComma(argsToRemove);
            std::string t;
            t.reserve(6 + argsJoined.size() + left[0].size() + right[0].size());
            t += "!(>[";
            t += argsJoined;
            t += "]";
            t += left[0];
            t += right[0];
            t += ")";
            subexprList.push_back(t);
        }
        else if (index < s.size() && s[index] == '&') {
            index += 1;

            std::vector<std::string> left = groomSubexpr(s, 0, index);
            std::vector<std::string> right = groomSubexpr(s, 0, index);

            if (side) {
                subexprList.reserve(left.size() + right.size());
                for (std::size_t i = 0; i < left.size(); ++i) subexprList.push_back(left[i]);
                for (std::size_t j = 0; j < right.size(); ++j) subexprList.push_back(right[j]);
            }
            else {
                std::string t;
                t.reserve(5 + left[0].size() + right[0].size());
                t += "!(&";
                t += left[0];
                t += right[0];
                t += ")";
                subexprList.push_back(t);
            }
        }
        else {
            std::size_t endIndex = s.find(')', index);
            if (endIndex == std::string::npos) {
                throw std::runtime_error("No closing ')'.");
            }
            std::string nodeLabel = s.substr(index, endIndex - index);
            index = endIndex;
            std::string t;
            t.reserve(4 + nodeLabel.size());
            t += "!(";
            t += nodeLabel;
            t += ")";
            subexprList.push_back(t);
        }
    }

    index += 1;
    return subexprList;
}

std::vector<std::string> ExpressionAnalyzer::groomExpr(const std::string& treeStr) {
    std::size_t index = 0;
    return groomSubexpr(treeStr, 1, index);
}

std::vector<std::string> ExpressionAnalyzer::listLastRemovedArgs(const std::string& expr) {
    std::vector<std::string> out;
    std::size_t start = expr.find(">[");
    if (start == std::string::npos) {
        return out;
    }
    std::size_t end = expr.find(']', start + 2);
    if (end == std::string::npos) {
        return out;
    }
    std::string inner = expr.substr(start + 2, end - (start + 2));
    std::size_t pos = 0;
    while (pos <= inner.size()) {
        std::size_t comma = inner.find(',', pos);
        if (comma == std::string::npos) {
            out.push_back(inner.substr(pos));
            break;
        }
        out.push_back(inner.substr(pos, comma - pos));
        pos = comma + 1;
    }
    return out;
}

std::tuple<std::string, int, std::string>
ExpressionAnalyzer::renameLastRemoved(const std::string& expr, int startInt) {
    const bool startsWithImp = expr.size() >= 2 && expr[0] == '(' && expr[1] == '>';
    const bool startsWithNotImp = expr.size() >= 3 && expr[0] == '!' && expr[1] == '(' && expr[2] == '>';
    assert(startsWithImp || startsWithNotImp);

    std::vector<std::string> args = listLastRemovedArgs(expr);
    std::set<std::string> argSet(args.begin(), args.end());
    std::vector<std::string> orderedArgs = ce::orderByPattern(expr, argSet);

    assert(orderedArgs.size() == 1);
    const std::string arg = orderedArgs[0];

    const std::string newVar =
        std::string("comp_") + std::to_string(startInt);

    std::map<std::string, std::string> replacementMap;
    replacementMap.insert(std::make_pair(arg, newVar));

    const std::string newExpr = ce::replaceKeysInString(expr, replacementMap);

    const int newStartInt = startInt + 1;
    return std::make_tuple(newExpr, newStartInt, newVar);
}


std::string ExpressionAnalyzer::expandExpr(const std::string& expr) {
    // 1. Check if expression is simple (e.g., "Macro[arg1, arg2]")
    if (!ce::expressionIsSimple(expr)) {
        return expr;
    }

    // 2. Extract the Head (Macro Name)
    std::string head = ce::extractExpressionUniversal(expr);
    auto it = this->coreExpressionMap.find(head);
    if (it == this->coreExpressionMap.end()) {
        // Not a known macro, return original
        return expr;
    }
    const ce::CoreExpressionConfig& cfg = it->second;

    // 3. Retrieve the Definition Template
    // Assuming definition is stored as a string
    if (!std::holds_alternative<std::string>(cfg.definition)) {
        return expr;
    }
    std::string defString = std::get<std::string>(cfg.definition);

    // 4. Extract Arguments
    // replacingArgs: The values passed in the current expression
    std::vector<std::string> replacingArgs = ce::getArgs(expr);
    // argsToBeReplaced: The formal parameters defined in the macro signature
    std::vector<std::string> argsToBeReplaced = ce::getArgs(cfg.signature);

    // Basic arity check
    if (replacingArgs.size() != argsToBeReplaced.size()) {
        return expr;
    }

    // 5. Collision Avoidance (Disentanglement)
    // Goal: If an input argument (e.g., "x") is NOT a formal parameter but appears 
    // inside the definition (as a bound variable), we must rename the bound variable 
    // in the definition to avoid capturing the input "x".

    std::map<std::string, std::string> collisionRenamingMap;
    int counter = 1;

    // Check every input argument for potential collision
    for (const std::string& inputArg : replacingArgs) {

        // Check if this input arg is actually one of the formal parameters.
        // If it is, it will be replaced by the expansion map later, so it's safe.
        bool isFormalParam = false;
        for (const std::string& param : argsToBeReplaced) {
            if (param == inputArg) {
                isFormalParam = true;
                break;
            }
        }

        if (isFormalParam) {
            continue;
        }

        // If we are here, 'inputArg' is being passed in, but it is NOT a parameter slot.
        // We must ensure 'defString' does not contain 'inputArg' as a local variable.
        // We speculatively add it to the renaming map. 
        // ce::replaceKeysInString will only perform the replacement if the key exists.

        // Generate a unique name: inputArg + Counter
        std::string newName;
        while (true) {
            newName = inputArg + std::to_string(counter);

            // Ensure the new name doesn't exist in the definition OR the input args
            // (Simple string find is used here for safety; ideally token-based)
            if (defString.find(newName) == std::string::npos) {
                // Also ensure we haven't generated a name that conflicts with another input
                bool nameClash = false;
                for (const auto& existingArg : replacingArgs) {
                    if (existingArg == newName) nameClash = true;
                }
                if (!nameClash) break;
            }
            counter++;
        }

        collisionRenamingMap[inputArg] = newName;
    }

    // Apply collision renaming to the definition template BEFORE standard expansion
    if (!collisionRenamingMap.empty()) {
        defString = ce::replaceKeysInString(defString, collisionRenamingMap);
    }

    // 6. Standard Expansion
    // Map formal parameters to input arguments
    std::map<std::string, std::string> expansionMap;
    for (std::size_t i = 0; i < argsToBeReplaced.size(); ++i) {
        expansionMap[argsToBeReplaced[i]] = replacingArgs[i];
    }

    // 7. Final Substitution
    std::string expandedExpr = ce::replaceKeysInString(defString, expansionMap);

    return expandedExpr;
}




std::vector<std::string> ExpressionAnalyzer::getGlobalKey(const Memory& memoryBlock) {
    std::vector<std::string> globalKey;
    const Memory* current = &memoryBlock;
    while (true) {
        if (!current->exprKey.empty()) {
            globalKey.push_back(current->exprKey);
        }
        if (current->parentMemory != NULL) {
            current = current->parentMemory;
        }
        else {
            break;
        }
    }
    std::reverse(globalKey.begin(), globalKey.end());
    return globalKey;
}

bool ExpressionAnalyzer::checkNoUArguments(const std::vector<std::string>& key, const std::string& value) {
    // Helper lambda to check a single expression string for "u_" args
    auto expressionHasU = [](const std::string& expr) -> bool {
        std::vector<std::string> args = ce::getArgs(expr);
        for (const std::string& arg : args) {
            if (startsWith(arg, "u_", 2)) {
                return true;
            }
        }
        return false;
        };

    // 1. Check the value
    if (expressionHasU(value)) {
        return false;
    }

    // 2. Check all elements in the key
    for (const std::string& k : key) {
        if (expressionHasU(k)) {
            return false;
        }
    }

    return true;
}


// ---------------------------------------------------------------------------
// Incubator: implication multiplication helpers
// ---------------------------------------------------------------------------

// Replace oldArg with newArg at argument boundaries only (won't match v1 inside v10,
// and won't match "7" inside "u_7" — underscore counts as a word character).
static std::string replaceArgInString(const std::string& str,
                                      const std::string& oldArg,
                                      const std::string& newArg) {
    auto isWordChar = [](char c) -> bool {
        return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
    };
    std::string result;
    result.reserve(str.size());
    std::size_t i = 0;
    while (i < str.size()) {
        bool atBoundary = (i == 0) || !isWordChar(str[i - 1]);
        if (atBoundary && str.compare(i, oldArg.size(), oldArg) == 0) {
            std::size_t after = i + oldArg.size();
            bool endBoundary = (after >= str.size()) || !isWordChar(str[after]);
            if (endBoundary) {
                result += newArg;
                i = after;
                continue;
            }
        }
        result += str[i];
        ++i;
    }
    return result;
}

// After variable replacement, >[...] lists may contain duplicates. Deduplicate them.
static std::string deduplicateBoundVars(const std::string& impl) {
    std::string result;
    result.reserve(impl.size());
    std::size_t i = 0;
    while (i < impl.size()) {
        // Look for "(>["
        if (i + 2 < impl.size() && impl[i] == '(' && impl[i + 1] == '>' && impl[i + 2] == '[') {
            result += "(>[";
            i += 3;
            std::size_t closeBracket = impl.find(']', i);
            if (closeBracket != std::string::npos) {
                std::string varList = impl.substr(i, closeBracket - i);
                // Parse comma-separated vars, deduplicate preserving order
                std::vector<std::string> vars;
                std::set<std::string> seen;
                std::size_t pos = 0;
                while (pos <= varList.size()) {
                    std::size_t comma = varList.find(',', pos);
                    std::string var;
                    if (comma == std::string::npos) {
                        var = varList.substr(pos);
                        pos = varList.size() + 1;
                    } else {
                        var = varList.substr(pos, comma - pos);
                        pos = comma + 1;
                    }
                    if (!var.empty() && seen.find(var) == seen.end()) {
                        // Strip u_-prefixed vars: they are fixed anchor args,
                        // must not appear as bound (wildcard) variables
                        if (var.size() >= 2 && var[0] == 'u' && var[1] == '_') continue;
                        seen.insert(var);
                        vars.push_back(var);
                    }
                }
                for (std::size_t j = 0; j < vars.size(); ++j) {
                    if (j > 0) result += ',';
                    result += vars[j];
                }
                i = closeBracket; // the ']' will be copied on next iteration
            }
        } else {
            result += impl[i];
            ++i;
        }
    }
    return result;
}

// Generate all set partitions (Bell number) of elements.
// Cap: if > 5 elements, return only original + all pairwise equalities.
using Partition = std::vector<std::vector<std::string>>;

static std::vector<Partition> generateSetPartitions(const std::vector<std::string>& elements, int cap = 5) {
    std::vector<Partition> result;
    if (elements.empty()) {
        result.push_back(Partition());
        return result;
    }

    // Beyond cap: pairwise + triples only.
    if (static_cast<int>(elements.size()) > cap) {
        // Original: all singletons
        Partition orig;
        for (auto& e : elements) orig.push_back({e});
        result.push_back(orig);
        // All pairwise equalities
        for (std::size_t i = 0; i < elements.size(); ++i) {
            for (std::size_t j = i + 1; j < elements.size(); ++j) {
                Partition p;
                for (std::size_t k = 0; k < elements.size(); ++k) {
                    if (k == j) continue;
                    if (k == i) {
                        std::vector<std::string> merged = {elements[i], elements[j]};
                        std::sort(merged.begin(), merged.end());
                        p.push_back(merged);
                    } else {
                        p.push_back({elements[k]});
                    }
                }
                result.push_back(p);
            }
        }
        // All triple equalities
        for (std::size_t i = 0; i < elements.size(); ++i) {
            for (std::size_t j = i + 1; j < elements.size(); ++j) {
                for (std::size_t l = j + 1; l < elements.size(); ++l) {
                    Partition p;
                    for (std::size_t k = 0; k < elements.size(); ++k) {
                        if (k == j || k == l) continue;
                        if (k == i) {
                            std::vector<std::string> merged = {elements[i], elements[j], elements[l]};
                            std::sort(merged.begin(), merged.end());
                            p.push_back(merged);
                        } else {
                            p.push_back({elements[k]});
                        }
                    }
                    result.push_back(p);
                }
            }
        }
        return result;
    }

    // Recursive Bell partition
    std::string first = elements[0];
    std::vector<std::string> rest(elements.begin() + 1, elements.end());
    auto subPartitions = generateSetPartitions(rest, cap);

    for (auto& sp : subPartitions) {
        // Option 1: first as new singleton class
        {
            Partition p = sp;
            p.push_back({first});
            result.push_back(p);
        }
        // Option 2: add first to each existing class
        for (std::size_t i = 0; i < sp.size(); ++i) {
            Partition p = sp;
            p[i].push_back(first);
            std::sort(p[i].begin(), p[i].end());
            result.push_back(p);
        }
    }
    return result;
}

// Numeric-aware comparison for variable names (handles "2" < "10" correctly).
static bool numericLess(const std::string& a, const std::string& b) {
    bool aDigit = !a.empty() && std::all_of(a.begin(), a.end(),
                      [](char c) { return std::isdigit(static_cast<unsigned char>(c)); });
    bool bDigit = !b.empty() && std::all_of(b.begin(), b.end(),
                      [](char c) { return std::isdigit(static_cast<unsigned char>(c)); });
    if (aDigit && bDigit) return std::stoi(a) < std::stoi(b);
    return a < b;
}

std::vector<std::string> ExpressionAnalyzer::multiplyImplication(
    const std::string& implication)
{
    if (!parameters.incubator_mode && !ceFilteringActive) return {implication};

    // 1. Parse implication to get chain + head
    std::vector< std::tuple<
        std::string,
        std::vector<std::string>,
        std::set<std::string>
    > > tempChain;
    std::string head = ce::disintegrateImplication(implication, tempChain, coreExpressionMap);

    // 2. Collect all bound vars from >[...] lists
    std::set<std::string> allBoundVars;
    for (auto& tpl : tempChain) {
        for (auto& arg : std::get<1>(tpl)) {
            if (!arg.empty()) allBoundVars.insert(arg);
        }
    }
    // 3. Collect all expressions (premises + head) for type lookup
    //    Skip Anchor expressions — their (1)-typed args inflate partitions needlessly.
    std::vector<std::string> allExprs;
    for (auto& tpl : tempChain) {
        const std::string& expr = std::get<0>(tpl);
        if (expr.find("(Anchor") == std::string::npos)
            allExprs.push_back(expr);
    }
    if (head.find("(Anchor") == std::string::npos)
        allExprs.push_back(head);

    // 4. Collect ALL unique variables (bound + u_) appearing in expressions
    std::set<std::string> allVars;
    allVars.insert(allBoundVars.begin(), allBoundVars.end());
    for (const std::string& expr : allExprs) {
        std::vector<std::string> args = ce::getArgs(expr);
        for (const std::string& arg : args) {
            allVars.insert(arg);
        }
    }

    // 5. Determine which vars are (1)-typed (both bound and u_)
    std::vector<std::string> oneTypedVars;
    for (const std::string& var : allVars) {
        bool isOneTyped = false;
        for (const std::string& expr : allExprs) {
            std::string coreName = ce::extractExpression(expr);
            if (coreName.empty()) continue;
            auto cfgIt = coreExpressionMap.find(coreName);
            if (cfgIt == coreExpressionMap.end()) continue;

            std::vector<std::string> args = ce::getArgs(expr);
            for (std::size_t i = 0; i < args.size(); ++i) {
                if (args[i] == var) {
                    std::string posKey = std::to_string(i + 1);
                    auto dsIt = cfgIt->second.definitionSets.find(posKey);
                    if (dsIt != cfgIt->second.definitionSets.end() &&
                        dsIt->second.first == "(1)") {
                        isOneTyped = true;
                        break;
                    }
                }
            }
            if (isOneTyped) break;
        }
        if (isOneTyped) oneTypedVars.push_back(var);
    }

    std::sort(oneTypedVars.begin(), oneTypedVars.end(), numericLess);

    if (oneTypedVars.size() <= 1) return {implication};

    // 6. Generate set partitions (equivalence relations on oneTypedVars).
    //    Each partition can compose multiple equalization groups simultaneously.
    auto partitions = generateSetPartitions(oneTypedVars, parameters.max_partition_size);

    // 7. For each partition, substitute and deduplicate.
    //    u_ variables always win as representative — non-u_ replaced by u_.
    std::set<std::string> seen;
    std::vector<std::string> result;

    for (auto& partition : partitions) {
        // Skip partitions where any equivalence class contains two different u_ vars
        bool hasDoubleU = false;
        for (auto& eqClass : partition) {
            int uCount = 0;
            for (const std::string& v : eqClass) {
                if (v.size() >= 2 && v[0] == 'u' && v[1] == '_') uCount++;
            }
            if (uCount >= 2) { hasDoubleU = true; break; }
        }
        // u_ equalization allowed in general (no double-u skip)

        std::string copy = implication;

        for (auto& eqClass : partition) {
            if (eqClass.size() <= 1) continue;

            // Pick representative: prefer u_ variable, else first element
            std::string rep = eqClass[0];
            for (const std::string& v : eqClass) {
                if (v.size() >= 2 && v[0] == 'u' && v[1] == '_') {
                    rep = v;
                    break;
                }
            }

            // Replace all non-representative vars with representative
            for (const std::string& v : eqClass) {
                if (v == rep) continue;
                copy = replaceArgInString(copy, v, rep);
            }
        }

        copy = deduplicateBoundVars(copy);

        // Skip copies whose head is a trivial equality (=[x,x]).
        {
            std::vector< std::tuple<std::string, std::vector<std::string>, std::set<std::string>> > copyChain;
            std::string copyHead = ce::disintegrateImplication(copy, copyChain, coreExpressionMap);
            if (ce::extractExpression(copyHead) == "=") {
                auto eqArgs = ce::getArgs(copyHead);
                if (eqArgs.size() == 2 && eqArgs[0] == eqArgs[1]) continue;
            }
        }

        if (seen.find(copy) == seen.end()) {
            seen.insert(copy);
            result.push_back(copy);
        }
    }

    return result;
}


// Turn optimizations OFF for just this section
//#pragma optimize("", off)

void ExpressionAnalyzer::addToHashMemory(const std::vector<std::string>& key,
    const std::string& value,
    const std::set<std::string>& remainingArgs,
    Memory& mb,
    HashMemory& targetIntMemory,
    const std::set<int>& levels,
    const std::string& originalImplication,
    int maxAdmissionDepth,
    int maxSecondaryNumber,
    bool partOfRecursion,
    int minNumOperatorsKey,
    const std::string& justification,
    bool performAdmissionMapUpdate,
    const std::string& originalImplicationClean,
    const std::string& validityName) {

    // Own multiplyImplication loop — fully independent from addToHashMemory.
    std::vector<std::string> copies = multiplyImplication(originalImplication);

    for (std::size_t c = 0; c < copies.size(); ++c) {
        std::vector<std::string> curKey;
        std::string curValue;
        std::string curOrigImpl = copies[c];

        if (c == 0) {
            curKey = key;
            curValue = value;
        } else {
            // Track history for multiplied copies
            if (parameters.trackHistory) {
                ExpressionWithValidity copyEv(copies[c], "main");
                std::pair<std::string, std::vector<ExpressionWithValidity>> mulOrigin;
                mulOrigin.first = "multiplied from";
                mulOrigin.second.push_back(ExpressionWithValidity(originalImplicationClean, "main"));
                addOrigin(mb.exprOriginMap, copyEv, mulOrigin,
                    (parameters.compressor_mode
                        ? parameters.compressor_max_origins_per_expr
                        : parameters.max_origin_per_expr));
                addOrigin(mb.mailOut.exprOriginMap, copyEv, mulOrigin,
                    (parameters.compressor_mode
                        ? parameters.compressor_max_origins_per_expr
                        : parameters.max_origin_per_expr));
            }

            std::vector< std::tuple<
                std::string,
                std::vector<std::string>,
                std::set<std::string>
            > > tc;
            curValue = ce::disintegrateImplication(copies[c], tc, coreExpressionMap);
            for (std::size_t ci = 0; ci < tc.size(); ++ci)
                curKey.push_back(std::get<0>(tc[ci]));
        }

        // --- Shared members (path-independent, needed for integration) ---
        std::vector<std::string> implication = curKey;
        implication.push_back(curValue);
        targetIntMemory.originals.insert(implication);
        for (const auto& trigger : targetIntMemory.triggersForAdmissionSetIntegration)
        {
            makeAdmissionKeys(implication, trigger.original, targetIntMemory, trigger.validityName);
        }

        this->makeNormalizedKeysForAdmission(curKey, targetIntMemory,
            mb.nameMap, mb.keyArena, curValue, minNumOperatorsKey);

        // Recompute remainingArgs for multiplied copies (u_ vars may have been equalized)
        const std::set<std::string>& curRemainingArgs = (c == 0) ? remainingArgs : getRemainingArgs(curKey);

        if (performAdmissionMapUpdate)
        {
            std::vector<std::string> keyPlusValue = curKey;
            keyPlusValue.push_back(curValue);
            updateAdmissionMap(mb,
                keyPlusValue,
                curRemainingArgs,
                maxAdmissionDepth,
                maxSecondaryNumber,
                partOfRecursion,
                "main");
        }

        // --- Inner: add this single key/value to int hash memory ---
        const std::size_t n = curKey.size();
        NameMap& nm = mb.nameMap;
        KeyArena& arena = mb.keyArena;

        std::set<int16_t> intRemainingArgs;
        for (const auto& ra : curRemainingArgs) {
            intRemainingArgs.insert(nm.encode(ra));
        }

        std::map<int, std::vector<std::vector<int>>>::const_iterator pit =
            this->allPermutationsAna.find(static_cast<int>(n));
        if (pit == this->allPermutationsAna.end()) {
            continue;
        }
        const std::vector<std::vector<int>>& permuts = pit->second;

        std::vector<std::string> ids;
        ids.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            ids.push_back(ce::extractExpression(curKey[i]));
        }

        int16_t bufIgnored[ExecutionParameters::MAX_KEY_SLOTS];
        int16_t bufNotIgnored[ExecutionParameters::MAX_KEY_SLOTS];

        for (std::size_t p = 0; p < permuts.size(); ++p) {
            const std::vector<int>& permutation = permuts[p];

            bool toContinue = false;
            if (n >= 2) {
                for (std::size_t k = 0; k + 1 < n; ++k) {
                    if (ids[permutation[k]] > ids[permutation[k + 1]]) {
                        toContinue = true;
                        break;
                    }
                }
            }
            if (toContinue) continue;

            std::vector<EncodedExpression> tempEncoded;
            tempEncoded.reserve(n);
            for (std::size_t k = 0; k < permutation.size(); ++k) {
                tempEncoded.push_back(EncodedExpression(curKey[permutation[k]], "main"));
            }

            int16_t reverseMap[ExecutionParameters::MAX_KEY_SLOTS];
            std::memset(reverseMap, 0, sizeof(reverseMap));
            int16_t numNormVars = 0;
            int16_t lenIgnored = makeIntNormalizedKeyWithMap(
                tempEncoded, nm, true, bufIgnored,
                ExecutionParameters::MAX_KEY_SLOTS, reverseMap, numNormVars);

            int16_t lenNotIgnored = makeIntNormalizedKey(
                tempEncoded, nm, false, bufNotIgnored,
                ExecutionParameters::MAX_KEY_SLOTS);

            const int16_t* pIgnored = arena.store(bufIgnored, lenIgnored);
            const int16_t* pNotIgnored = arena.store(bufNotIgnored, lenNotIgnored);

            IntNormalizedKey intIgnoredKey(static_cast<int16_t>(n), pIgnored, lenIgnored);
            IntNormalizedKey intNotIgnoredKey(static_cast<int16_t>(n), pNotIgnored, lenNotIgnored);

            std::string valueVariant = curValue;
            {
                std::pair<NormalizedKey, std::map<std::string, std::string>> pairIgnored =
                    makeNormalizedEncodedKey(tempEncoded, true);
                const std::map<std::string, std::string>& mp2 = pairIgnored.second;
                valueVariant = ce::replaceKeysInString(curValue, mp2);
            }

            LocalMemoryValue lmv;
            lmv.value = valueVariant;
            lmv.levels = levels;
            lmv.originalImplication = curOrigImpl;
            lmv.remainingArgs = curRemainingArgs;
            lmv.justification = justification;
            lmv.validityName = validityName;

            targetIntMemory.encodedMap[intIgnoredKey].push_back(lmv);
            targetIntMemory.normalizedEncodedKeys.insert(intNotIgnoredKey);
            targetIntMemory.remainingArgsNormalizedEncodedMap[intRemainingArgs].insert(intNotIgnoredKey);
        }

        // --- makeNormalizedSubkeys equivalent ---
        for (std::size_t p = 0; p < permuts.size(); ++p) {
            const std::vector<int>& permut = permuts[p];

            std::vector<std::string> tempList;
            tempList.reserve(n);
            for (std::size_t k = 0; k < permut.size(); ++k) {
                tempList.push_back(curKey[permut[k]]);
            }

            for (std::size_t index = 0; index < n; ++index) {
                bool toBreak = false;
                for (std::size_t index2 = 0; index2 < index; ++index2) {
                    if (ids[permut[index2]] > ids[permut[index2 + 1]]) {
                        toBreak = true;
                        break;
                    }
                }
                if (toBreak) break;

                std::vector<EncodedExpression> subkeyEncoded;
                subkeyEncoded.reserve(index + 1);
                for (std::size_t t = 0; t <= index; ++t) {
                    subkeyEncoded.push_back(EncodedExpression(tempList[t], "main"));
                }

                int16_t subBuf[ExecutionParameters::MAX_KEY_SLOTS];
                int16_t subLen = makeIntNormalizedKey(
                    subkeyEncoded, nm, false, subBuf,
                    ExecutionParameters::MAX_KEY_SLOTS);

                const int16_t* pSub = arena.store(subBuf, subLen);
                IntNormalizedKey intSubKey(static_cast<int16_t>(index + 1), pSub, subLen);

                targetIntMemory.normalizedEncodedSubkeys.insert(intSubKey);

                if (index + 1 == n - 1) {
                    targetIntMemory.normalizedEncodedSubkeysMinusOne.insert(intSubKey);
                }
                if (index + 1 == n - 2) {
                    targetIntMemory.normalizedEncodedSubkeysMinusTwo.insert(intSubKey);
                }
            }
        }

        targetIntMemory.maxKeyLength = std::max(static_cast<int16_t>(n), targetIntMemory.maxKeyLength);
    }
}

void ExpressionAnalyzer::makeNormalizedKeysForAdmission(
    const std::vector<std::string>& key,
    HashMemory& intHashMemory,
    NameMap& nameMap,
    KeyArena& arena,
    const std::string& value,
    int minNumOperatorsKey) {

    if (!implicationIsQualified(key, value, minNumOperatorsKey)) {
        return;
    }

    const std::size_t n = key.size();

    for (std::size_t index = 0; index < n; ++index) {
        const std::string coreExpr = ce::extractExpression(key[index]);
        const std::vector<std::string> args = ce::getArgs(key[index]);
        auto cit = this->coreExpressionMap.find(coreExpr);
        assert(cit != this->coreExpressionMap.end() && "Core expression not found");

        std::vector<std::string> outputArgsList;
        for (int idx : cit->second.outputIndices) {
            if (idx >= 0 && idx < static_cast<int>(args.size()))
                outputArgsList.push_back(args[idx]);
        }
        assert(outputArgsList.size() <= 1 && "Expression must have 0 or 1 output argument");
        if (outputArgsList.empty()) continue;
        const std::string outputArg = outputArgsList[0];

        // Build binary mask and subkey (same logic as string path)
        std::vector<int> binary(n, 0);
        std::vector<std::string> subkey;
        subkey.reserve(n);
        int validCount = 0;
        for (std::size_t i = 0; i < n; ++i) {
            if (i == index) continue;
            const std::vector<std::string> iArgs = ce::getArgs(key[i]);
            bool containsOutput = false;
            for (const auto& arg : iArgs) {
                if (arg == outputArg) { containsOutput = true; break; }
            }
            if (!containsOutput) {
                binary[i] = 1;
                subkey.push_back(key[i]);
                validCount++;
            }
        }

        bool accept = (validCount == static_cast<int>(n) - 1) ||
            (validCount == parameters.minLenLongKey - 1);
        if (!accept) continue;

        // Replace output arg with "marker"
        std::map<std::string, std::string> replacementMap;
        replacementMap.insert(std::make_pair(outputArg, "marker"));
        const std::string replaced = ce::replaceKeysInString(key[index], replacementMap);

        // --- Int subkeys (mirrors makeNormalizedSubkeys for HashMemory) ---
        {
            const std::size_t sn = subkey.size();
            auto pit = this->allPermutationsAna.find(static_cast<int>(sn));
            if (pit != this->allPermutationsAna.end()) {
                const auto& permuts = pit->second;
                std::vector<std::string> sids;
                sids.reserve(sn);
                for (std::size_t i = 0; i < sn; ++i)
                    sids.push_back(ce::extractExpression(subkey[i]));

                for (std::size_t p = 0; p < permuts.size(); ++p) {
                    const auto& permut = permuts[p];
                    std::vector<std::string> tempList;
                    tempList.reserve(sn);
                    for (std::size_t k = 0; k < permut.size(); ++k)
                        tempList.push_back(subkey[permut[k]]);

                    for (std::size_t si = 0; si < sn; ++si) {
                        bool toBreak = false;
                        for (std::size_t si2 = 0; si2 < si; ++si2) {
                            if (sids[permut[si2]] > sids[permut[si2 + 1]]) {
                                toBreak = true; break;
                            }
                        }
                        if (toBreak) break;

                        std::vector<EncodedExpression> subEnc;
                        subEnc.reserve(si + 1);
                        for (std::size_t t = 0; t <= si; ++t)
                            subEnc.push_back(EncodedExpression(tempList[t], "main"));

                        int16_t subBuf[ExecutionParameters::MAX_KEY_SLOTS];
                        int16_t subLen = makeIntNormalizedKey(
                            subEnc, nameMap, false, subBuf, ExecutionParameters::MAX_KEY_SLOTS);
                        const int16_t* pSub = arena.store(subBuf, subLen);
                        IntNormalizedKey intSubKey(static_cast<int16_t>(si + 1), pSub, subLen);

                        intHashMemory.normalizedEncodedSubkeys.insert(intSubKey);
                        if (si + 1 == sn - 1)
                            intHashMemory.normalizedEncodedSubkeysMinusOne.insert(intSubKey);
                        if (si + 1 == sn - 2)
                            intHashMemory.normalizedEncodedSubkeysMinusTwo.insert(intSubKey);
                    }
                }
            }
        }

        // --- Main variants: create int keys for encodedMap marker entries ---
        {
            const std::size_t sn = subkey.size();
            auto pit = this->allPermutationsAna.find(static_cast<int>(sn));
            if (pit == this->allPermutationsAna.end()) continue;
            const auto& permuts = pit->second;

            std::vector<std::string> sids;
            sids.reserve(sn);
            for (std::size_t i = 0; i < sn; ++i)
                sids.push_back(ce::extractExpression(subkey[i]));

            std::set<std::string> remainingArgs = getRemainingArgs(subkey);
            std::set<int16_t> intRemainingArgs;
            for (const auto& ra : remainingArgs)
                intRemainingArgs.insert(nameMap.encode(ra));

            int16_t bufIgnored[ExecutionParameters::MAX_KEY_SLOTS];
            int16_t bufNotIgnored[ExecutionParameters::MAX_KEY_SLOTS];

            for (std::size_t p = 0; p < permuts.size(); ++p) {
                const auto& permutation = permuts[p];

                bool toContinue = false;
                if (sn >= 2) {
                    for (std::size_t k = 0; k + 1 < sn; ++k) {
                        if (sids[permutation[k]] > sids[permutation[k + 1]]) {
                            toContinue = true; break;
                        }
                    }
                }
                if (toContinue) continue;

                std::vector<EncodedExpression> tempEncoded;
                tempEncoded.reserve(sn);
                for (std::size_t k = 0; k < permutation.size(); ++k)
                    tempEncoded.push_back(EncodedExpression(subkey[permutation[k]], "main"));

                // Int keys from subkey
                int16_t lenIgnored = makeIntNormalizedKey(
                    tempEncoded, nameMap, true, bufIgnored, ExecutionParameters::MAX_KEY_SLOTS);
                int16_t lenNotIgnored = makeIntNormalizedKey(
                    tempEncoded, nameMap, false, bufNotIgnored, ExecutionParameters::MAX_KEY_SLOTS);

                const int16_t* pIgnored = arena.store(bufIgnored, lenIgnored);
                const int16_t* pNotIgnored = arena.store(bufNotIgnored, lenNotIgnored);

                IntNormalizedKey intIgnoredKey(static_cast<int16_t>(sn), pIgnored, lenIgnored);
                IntNormalizedKey intNotIgnoredKey(static_cast<int16_t>(sn), pNotIgnored, lenNotIgnored);

                // String normalization mapping for value variant and replKey
                std::pair<NormalizedKey, std::map<std::string, std::string>> pairIgnored =
                    makeNormalizedEncodedKey(tempEncoded, true);
                const std::map<std::string, std::string>& mp2 = pairIgnored.second;

                std::string valueVariant = ce::replaceKeysInString(replaced, mp2);

                // Build replKey (valid subkey elements + head expression, with mapping applied)
                std::map<std::string, std::string> mpMap(mp2.begin(), mp2.end());
                std::vector<std::string> replKey;
                replKey.reserve(subkey.size() + 1);
                for (std::size_t i = 0; i < key.size(); ++i) {
                    if (binary[i] || i == index) {
                        replKey.push_back(ce::replaceKeysInString(key[i], mpMap));
                    }
                }

                LocalMemoryValue lmv;
                lmv.value = valueVariant;
                lmv.key = replKey;
                lmv.remainingArgs = remainingArgs;

                intHashMemory.encodedMap[intIgnoredKey].push_back(lmv);
                intHashMemory.normalizedEncodedKeys.insert(intNotIgnoredKey);
                intHashMemory.remainingArgsNormalizedEncodedMap[intRemainingArgs].insert(intNotIgnoredKey);
            }
        }
    }
}

bool ExpressionAnalyzer::lessByName(const EncodedExpression& a, const EncodedExpression& b) {
    return a.name < b.name;
}
bool ExpressionAnalyzer::lessByOriginal(const EncodedExpression& a, const EncodedExpression& b) {
    return a.original < b.original;
}

std::vector<EncodedExpression>
ExpressionAnalyzer::sortEncodedExpressionsByName(const std::vector<EncodedExpression>& in) {
    std::vector<EncodedExpression> out = in;
    std::stable_sort(out.begin(), out.end(), &ExpressionAnalyzer::lessByName);
    return out;
}


// .cpp
std::vector<EncodedExpression>
ExpressionAnalyzer::mergeInsertSortedEncoded(const std::vector<EncodedExpression>& listA,
    const std::vector<std::string>& valuesA,
    const std::vector<EncodedExpression>& listB,
    const std::vector<std::string>& valuesB)
{
    // Precondition: valuesA are sorted by name (they come from filteredAll sorted by name). :contentReference[oaicite:1]{index=1}
    // valuesB MAY BE UNSORTED. We sort B by its keys first, stably.
    const std::size_t nA = listA.size();
    const std::size_t nB = std::min(listB.size(), valuesB.size());

    std::vector<std::size_t> idxB(nB);
    for (std::size_t i = 0; i < nB; ++i) idxB[i] = i;

    // Stable sort ensures B’s original relative order for equal keys (matches repeated inserts).
    std::stable_sort(idxB.begin(), idxB.end(),
        [&](std::size_t i, std::size_t j) { return valuesB[i] < valuesB[j]; });

    std::vector<EncodedExpression> out;
    out.reserve(nA + nB);

    std::size_t iA = 0, iB = 0;
    while (iA < nA && iB < nB) {
        const std::string& aKey = valuesA[iA];
        const std::string& bKey = valuesB[idxB[iB]];

        // B-first on ties to match the original lower_bound+insert behavior. :contentReference[oaicite:2]{index=2}
        if (aKey < bKey) {
            out.push_back(listA[iA++]);
        }
        else {
            out.push_back(listB[idxB[iB++]]);
        }
    }
    while (iA < nA) out.push_back(listA[iA++]);
    while (iB < nB) out.push_back(listB[idxB[iB++]]);
    return out;
}





std::vector<std::vector<EncodedExpression>>
ExpressionAnalyzer::makeMandatoryEncodedStatementLists1(const HashMemory& intMemory,
    const std::vector<EncodedExpression>& localStatements, NameMap& nm) {
    std::vector<EncodedExpression> tmp = filterEncodedStatements(localStatements, intMemory, nm);
    std::stable_sort(tmp.begin(), tmp.end(), &ExpressionAnalyzer::lessByOriginal);

    std::vector<std::vector<EncodedExpression>> out;
    out.reserve(tmp.size());
    for (std::size_t i = 0; i < tmp.size(); ++i) {
        std::vector<EncodedExpression> one;
        one.push_back(tmp[i]);
        out.push_back(one);
    }
    return out;
}

std::vector<std::vector<EncodedExpression>>
ExpressionAnalyzer::makeMandatoryEncodedStatementLists2(Memory& body,
    const HashMemory& intMemory,
    const std::vector<EncodedExpression>& firstLayer,
    const std::vector<EncodedExpression>& secondLayer) {
    std::vector<std::vector<EncodedExpression>> mandatory;
    NameMap& nm = body.nameMap;

    if (!firstLayer.empty() && !secondLayer.empty()) {
        std::vector<EncodedExpression> f1 = filterEncodedStatements(firstLayer, intMemory, nm);
        std::vector<EncodedExpression> f2 = filterEncodedStatements(secondLayer, intMemory, nm);

        std::stable_sort(f1.begin(), f1.end(), &ExpressionAnalyzer::lessByOriginal);
        std::stable_sort(f2.begin(), f2.end(), &ExpressionAnalyzer::lessByOriginal);

        for (std::size_t i = 0; i < f1.size(); ++i) {
            for (std::size_t j = 0; j < f2.size(); ++j) {
                if (f1[i].validityName != f2[j].validityName) {
                    if ((f1[i].validityName != "main") && (f2[j].validityName != "main")) {
                        continue;
                    }
                }
                if (!(f1[i] == f2[j])) {
                    std::vector<EncodedExpression> list2;
                    list2.push_back(f1[i]);
                    list2.push_back(f2[j]);
                    std::sort(list2.begin(), list2.end(), &ExpressionAnalyzer::lessByName);

                    std::pair<bool, IntNormalizedKey> pe = preEvaluateEncodedKey(list2, body, body.overallHashMemory.normalizedEncodedSubkeys);
                    if (pe.first) {
                        mandatory.push_back(list2);
                    }
                }
            }
        }
    }
    return mandatory;
}

// ========================================================================
// ========================================================================
// growBaseCandidates — shared grow loop for Singles/Pairs.
// Enumerates subsets of filtered allStatements up to targetLen, collecting
// those that match targetSubkeys. Pruning via normalizedEncodedSubkeys.
// ========================================================================
void ExpressionAnalyzer::growBaseCandidates(
    Memory& body,
    const HashMemory& intMemory,
    const IntEncodedExpr* allIntStmts,
    const int16_t* filteredIdx, int16_t nFiltered,
    int16_t mainValidityId,
    int targetLen,
    const std::unordered_set<IntNormalizedKey, IntNormalizedKeyHash>& targetSubkeys,
    std::vector<BaseCandidate>& baseCandidates)
{
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
            if (top.count + 1 > targetLen) break;

            const int16_t allIdx = filteredIdx[i];
            const IntEncodedExpr& ie = allIntStmts[allIdx];

            int16_t newValidityId = top.validityId;
            if (ie.validityId != mainValidityId) {
                if (top.validityId == mainValidityId) {
                    newValidityId = ie.validityId;
                } else if (top.validityId != ie.validityId) {
                    continue;
                }
            }

            const IntEncodedExpr* ptrs[ExecutionParameters::MAX_EXPRESSIONS];
            for (int16_t k = 0; k < top.count; ++k)
                ptrs[k] = &allIntStmts[top.allIdx[k]];
            ptrs[top.count] = &ie;
            const int16_t newCount = static_cast<int16_t>(top.count + 1);

            std::pair<bool, IntNormalizedKey> pr =
                preEvaluateFromEncoded(ptrs, newCount, body, mainValidityId,
                    intMemory.normalizedEncodedSubkeys);

            if (pr.first) {
                if (targetSubkeys.find(pr.second) != targetSubkeys.end()) {
                    BaseCandidate bc;
                    std::memcpy(bc.allIdx, top.allIdx, top.count * sizeof(int16_t));
                    bc.allIdx[top.count] = allIdx;
                    bc.count = newCount;
                    bc.validityId = newValidityId;
                    baseCandidates.push_back(bc);
                }
                if (newCount < targetLen) {
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
}

// ========================================================================
// generateEncodedRequestsStatic — singles (sizeOne=1).
// Seed: mandatory singles as complete keys.
// Grow: subsets up to maxKeyLen-1 via growBaseCandidates.
// Merge: baseCandidates × mandatory singles → full keys.
// ========================================================================
int16_t ExpressionAnalyzer::generateEncodedRequestsStatic(
    Memory& body,
    const HashMemory& intMemory,
    const int16_t* mandatoryIndices, int16_t mandatoryCount,
    const IntEncodedExpr* mandatorySrcInt,
    TypedArena<IntEncodedExpr>& exprArena,
    StaticRequest* outBuf, int16_t maxOut)
{
    if (mandatoryCount <= 0) return 0;

    NameMap& nm = body.nameMap;
    int16_t buf[ExecutionParameters::MAX_KEY_SLOTS];
    const int16_t mainValidityId = nm.encode("main");
    const int maxKeyLen = intMemory.maxKeyLength;
    const int targetLen = std::max(0, maxKeyLen - 1);

    StaticRequestEmitter emitter(exprArena, outBuf, maxOut);

    // --- Seed: each mandatory single as complete key ---
    for (int16_t i = 0; i < mandatoryCount; ++i) {
        const IntEncodedExpr& ie = mandatorySrcInt[mandatoryIndices[i]];
        const IntEncodedExpr* ptr = &ie;
        int16_t len = makeIntNormalizedKeyFromEncoded(&ptr, 1, buf,
            ExecutionParameters::MAX_KEY_SLOTS);
        IntNormalizedKey probe(1, buf, len);
        if (intMemory.normalizedEncodedKeys.find(probe) != intMemory.normalizedEncodedKeys.end()) {
            const int16_t* p = body.keyArena.store(buf, len);
            IntNormalizedKey nk(1, p, len);
            emitter.emit(&ptr, 1, nk);
        }
    }

    if (targetLen <= 0) return emitter.outCount;

    // --- Filter + sort ---
    const IntEncodedExpr* allIntStmts = body.intEncodedStatements.data();
    const int16_t allIntCount = static_cast<int16_t>(body.intEncodedStatements.size());
    int16_t filteredIdx[8192];
    int16_t nFiltered = filterIntEncodedStatements(allIntStmts, allIntCount,
        intMemory, filteredIdx, 8192);
    std::sort(filteredIdx, filteredIdx + nFiltered, [&](int16_t a, int16_t b) {
        return nm.decode(allIntStmts[a].nameId) < nm.decode(allIntStmts[b].nameId);
    });

    // --- Grow ---
    std::vector<BaseCandidate> baseCandidates;
    growBaseCandidates(body, intMemory, allIntStmts, filteredIdx, nFiltered,
        mainValidityId, targetLen,
        intMemory.normalizedEncodedSubkeysMinusOne, baseCandidates);

    // --- Merge: baseCandidates × mandatory singles ---
    for (std::size_t bi = 0; bi < baseCandidates.size(); ++bi) {
        const BaseCandidate& base = baseCandidates[bi];

        for (int16_t j = 0; j < mandatoryCount; ++j) {
            const int16_t mIdx = mandatoryIndices[j];
            const IntEncodedExpr& mIe = mandatorySrcInt[mIdx];
            const int16_t mValidityId = mIe.validityId;

            if (base.validityId != mainValidityId && mValidityId != mainValidityId
                && base.validityId != mValidityId) continue;

            bool dup = false;
            for (int16_t k = 0; k < base.count && !dup; ++k) {
                const IntEncodedExpr& bIe = allIntStmts[base.allIdx[k]];
                if (bIe.originalId == mIe.originalId && bIe.validityId == mIe.validityId) {
                    dup = true;
                }
            }
            if (dup) continue;

            // Build merged pointer array in name-sorted order
            const IntEncodedExpr* merged[ExecutionParameters::MAX_EXPRESSIONS + 1];
            int16_t mc = 0;
            const std::string& mName = nm.decode(mIe.nameId);
            bool mInserted = false;
            for (int16_t k = 0; k < base.count; ++k) {
                const std::string& bName = nm.decode(allIntStmts[base.allIdx[k]].nameId);
                if (!mInserted && !(bName < mName)) {
                    merged[mc++] = &mIe;
                    mInserted = true;
                }
                merged[mc++] = &allIntStmts[base.allIdx[k]];
            }
            if (!mInserted) merged[mc++] = &mIe;

            std::pair<bool, IntNormalizedKey> pr2 =
                preEvaluateFromEncoded(merged, mc, body, mainValidityId,
                    intMemory.normalizedEncodedKeys);
            if (pr2.first) {
                emitter.emit(merged, mc, pr2.second);
            }
        }
    }

    return emitter.outCount;
}

// ========================================================================
// generateEncodedRequestsStaticPairs — pairs (sizeOne=2).
// Seed: mandatory pairs as complete 2-element keys.
// Grow: subsets up to maxKeyLen-2 via growBaseCandidates.
// Merge: baseCandidates × mandatory pairs → full keys.
// ========================================================================
int16_t ExpressionAnalyzer::generateEncodedRequestsStaticPairs(
    Memory& body,
    const HashMemory& intMemory,
    const MandatoryPair* pairs, int16_t pairCount,
    const IntEncodedExpr* firstSrcInt,
    const IntEncodedExpr* secondSrcInt,
    TypedArena<IntEncodedExpr>& exprArena,
    StaticRequest* outBuf, int16_t maxOut)
{
    if (pairCount <= 0) return 0;

    NameMap& nm = body.nameMap;
    int16_t buf[ExecutionParameters::MAX_KEY_SLOTS];
    const int16_t mainValidityId = nm.encode("main");
    const int maxKeyLen = intMemory.maxKeyLength;
    const int targetLen = std::max(0, maxKeyLen - 2);

    StaticRequestEmitter emitter(exprArena, outBuf, maxOut);

    // Pre-sort each pair by name for consistent key generation
    struct SortedPair {
        IntEncodedExpr sorted[2];
    };
    std::vector<SortedPair> sortedPairs(pairCount);
    std::vector<bool> pairValid(pairCount, true);

    for (int16_t i = 0; i < pairCount; ++i) {
        const IntEncodedExpr& e1 = firstSrcInt[pairs[i].idx1];
        const IntEncodedExpr& e2 = secondSrcInt[pairs[i].idx2];
        const std::string& n1 = nm.decode(e1.nameId);
        const std::string& n2 = nm.decode(e2.nameId);
        if (n1 <= n2) { sortedPairs[i].sorted[0] = e1; sortedPairs[i].sorted[1] = e2; }
        else          { sortedPairs[i].sorted[0] = e2; sortedPairs[i].sorted[1] = e1; }

        int16_t v1 = e1.validityId, v2 = e2.validityId;
        if (v1 != v2) {
            if (v1 != mainValidityId && v2 != mainValidityId) { pairValid[i] = false; }
        }
    }

    // --- Seed: each pair as complete 2-element key ---
    for (int16_t i = 0; i < pairCount; ++i) {
        if (!pairValid[i]) continue;
        const IntEncodedExpr* ptrs[2] = { &sortedPairs[i].sorted[0], &sortedPairs[i].sorted[1] };
        int16_t len = makeIntNormalizedKeyFromEncoded(ptrs, 2, buf,
            ExecutionParameters::MAX_KEY_SLOTS);
        IntNormalizedKey probe(2, buf, len);
        if (intMemory.normalizedEncodedKeys.find(probe) != intMemory.normalizedEncodedKeys.end()) {
            const int16_t* p = body.keyArena.store(buf, len);
            IntNormalizedKey nk(2, p, len);
            emitter.emit(ptrs, 2, nk);
        }
    }

    if (targetLen <= 0) return emitter.outCount;

    // --- Filter + sort ---
    const IntEncodedExpr* allIntStmts = body.intEncodedStatements.data();
    const int16_t allIntCount = static_cast<int16_t>(body.intEncodedStatements.size());
    int16_t filteredIdx[8192];
    int16_t nFiltered = filterIntEncodedStatements(allIntStmts, allIntCount,
        intMemory, filteredIdx, 8192);
    std::sort(filteredIdx, filteredIdx + nFiltered, [&](int16_t a, int16_t b) {
        return nm.decode(allIntStmts[a].nameId) < nm.decode(allIntStmts[b].nameId);
    });

    // --- Grow ---
    std::vector<BaseCandidate> baseCandidates;
    growBaseCandidates(body, intMemory, allIntStmts, filteredIdx, nFiltered,
        mainValidityId, targetLen,
        intMemory.normalizedEncodedSubkeysMinusTwo, baseCandidates);

    // --- Merge: baseCandidates × mandatory pairs ---
    for (std::size_t bi = 0; bi < baseCandidates.size(); ++bi) {
        const BaseCandidate& base = baseCandidates[bi];

        for (int16_t pi = 0; pi < pairCount; ++pi) {
            if (!pairValid[pi]) continue;
            const IntEncodedExpr& m0 = sortedPairs[pi].sorted[0];
            const IntEncodedExpr& m1 = sortedPairs[pi].sorted[1];

            int16_t pairVid = (m0.validityId != mainValidityId) ? m0.validityId : m1.validityId;
            if (pairVid == mainValidityId) pairVid = mainValidityId;
            if (base.validityId != mainValidityId && pairVid != mainValidityId
                && base.validityId != pairVid) continue;

            bool dup = false;
            for (int16_t k = 0; k < base.count && !dup; ++k) {
                const IntEncodedExpr& bIe = allIntStmts[base.allIdx[k]];
                if ((bIe.originalId == m0.originalId && bIe.validityId == m0.validityId) ||
                    (bIe.originalId == m1.originalId && bIe.validityId == m1.validityId)) {
                    dup = true;
                }
            }
            if (dup) continue;

            const IntEncodedExpr* merged[ExecutionParameters::MAX_EXPRESSIONS + 2];
            int16_t mc = 0;
            for (int16_t k = 0; k < base.count; ++k)
                merged[mc++] = &allIntStmts[base.allIdx[k]];
            merged[mc++] = &m0;
            merged[mc++] = &m1;
            std::sort(merged, merged + mc, [&](const IntEncodedExpr* a, const IntEncodedExpr* b) {
                return nm.decode(a->nameId) < nm.decode(b->nameId);
            });

            std::pair<bool, IntNormalizedKey> pr2 =
                preEvaluateFromEncoded(merged, mc, body, mainValidityId,
                    intMemory.normalizedEncodedKeys);
            if (pr2.first) {
                emitter.emit(merged, mc, pr2.second);
            }
        }
    }

    return emitter.outCount;
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
    const int16_t mainValidityId = nm.encode("main");

    StaticRequestEmitter emitter(exprArena, outBuf, maxOut);

    // Filter allStatements (CE filter: subkeys OR full keys)
    const IntEncodedExpr* allIntStmts = body.intEncodedStatements.data();
    const int16_t allIntCount = static_cast<int16_t>(body.intEncodedStatements.size());
    int16_t filteredIdx[8192];
    int16_t nFiltered = filterIntEncodedStatementsCE(allIntStmts, allIntCount,
        intMemory, filteredIdx, 8192);
    std::sort(filteredIdx, filteredIdx + nFiltered, [&](int16_t a, int16_t b) {
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

            int16_t newValidityId = top.validityId;
            if (ie.validityId != mainValidityId) {
                if (top.validityId == mainValidityId) {
                    newValidityId = ie.validityId;
                } else if (top.validityId != ie.validityId) {
                    continue;
                }
            }

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

void ExpressionAnalyzer::checkLocalEncodedMemory(const std::vector<EncodedExpression>& expressionList,
    Memory& memoryBlock,
    int iteration,
    const IntNormalizedKey& tple,
    unsigned coreId)
{
    std::string expressionListValidityName;
    bool resultValidityName = getValidityName(expressionList, expressionListValidityName);
    if (!resultValidityName) return;

    if (memoryBlock.validityNamesToFilter.count(expressionListValidityName)) return;

    // pure check: all args with iteration >=0 must be productsOfRecursion
    int algebraicCounter = 0;
    for (std::size_t i = 0; i < expressionList.size(); ++i) {
        const EncodedExpression& ex = expressionList[i];
        for (std::size_t a = 0; a < ex.arguments.size(); ++a) {
            const std::vector<std::string>& arg = ex.arguments[a];
            if (arg.size() >= 3) {
                const int itNum = std::atoi(arg[2].c_str());
                if (itNum > -1) {
                    const std::string& varName = arg[1];
                    if (memoryBlock.overallHashMemory.productsOfRecursion.find(varName) ==
                        memoryBlock.overallHashMemory.productsOfRecursion.end()) {
                        algebraicCounter++;
                    }
                }
            }
        }
    }
    bool pure = algebraicCounter == 0;

    const std::set<std::string> orUnchangeables = this->extractUnchangeables(expressionList);

    std::set<int> combinedLevels;
    for (std::size_t i = 0; i < expressionList.size(); ++i) {
        auto itL = memoryBlock.statementLevelsMap.find(expressionList[i]);
        if (itL != memoryBlock.statementLevelsMap.end()) {
            for (const auto& lv : itL->second) combinedLevels.insert(lv);
        }
    }

    const std::set<std::string> allArgs = this->getAllEncodedArgs(expressionList);

    // Convert allArgs to int16_t for subset check
    NameMap& nm = memoryBlock.nameMap;
    std::set<int16_t> intAllArgs;
    for (const auto& a : allArgs) intAllArgs.insert(nm.encode(a));

    if (!memoryBlock.overallHashMemory.encodedMap.empty()) {
        // Collect candidate remaining-arg sets (sorted for determinism)
        std::vector<std::set<int16_t>> candidates;
        for (const auto& kv : memoryBlock.overallHashMemory.remainingArgsNormalizedEncodedMap) {
            const std::set<int16_t>& st = kv.first;
            bool subsetOk = true;
            for (const auto& arg : st) {
                if (intAllArgs.find(arg) == intAllArgs.end()) {
                    subsetOk = false; break;
                }
            }
            if (subsetOk) candidates.push_back(st);
        }
        std::sort(candidates.begin(), candidates.end());

        for (const auto& intSt : candidates) {
            auto itRA = memoryBlock.overallHashMemory.remainingArgsNormalizedEncodedMap.find(intSt);
            if (itRA == memoryBlock.overallHashMemory.remainingArgsNormalizedEncodedMap.end()) continue;

            const std::unordered_set<IntNormalizedKey, IntNormalizedKeyHash>& nkSet = itRA->second;
            if (nkSet.find(tple) == nkSet.end()) continue;

            // Convert int16_t set back to string set for setUnchangeables
            std::set<std::string> st;
            for (const auto& id : intSt) st.insert(nm.decode(id));

            std::vector<EncodedExpression> workList = expressionList;
            this->setUnchangeables(workList, st);

            // Build int normalized key on stack for probe (no arena allocation)
            int16_t buf[ExecutionParameters::MAX_KEY_SLOTS];
            int16_t reverseMap[ExecutionParameters::MAX_KEY_SLOTS];
            std::memset(reverseMap, 0, sizeof(reverseMap));
            int16_t numNormVars = 0;
            int16_t len = makeIntNormalizedKeyWithMap(workList, nm, true, buf,
                ExecutionParameters::MAX_KEY_SLOTS, reverseMap, numNormVars);

            // Probe with stack buffer — skip expensive string work on miss
            IntNormalizedKey probe(static_cast<int16_t>(workList.size()), buf, len);
            auto itEnc = memoryBlock.overallHashMemory.encodedMap.find(probe);
            if (itEnc == memoryBlock.overallHashMemory.encodedMap.end()) {
                this->resetUnchangeables(workList, orUnchangeables);
                continue;
            }

            // Hit — get string normalization mapping for value back-substitution
            std::pair<NormalizedKey, std::map<std::string, std::string>> normPair =
                this->makeNormalizedEncodedKey(workList, true);
            const std::map<std::string, std::string>& normMap = normPair.second;

            this->resetUnchangeables(workList, orUnchangeables);

            // Build back replacement map (same as string path)
            std::map<std::string, std::string> backReplacementMap;
            for (auto it = normMap.begin(); it != normMap.end(); ++it) {
                backReplacementMap[it->second] = it->first;
            }

            const std::vector<LocalMemoryValue>& valueVec = itEnc->second;

            // Sort by value (same as string path)
            std::vector<LocalMemoryValue> ordered(valueVec.begin(), valueVec.end());
            struct ByValue {
                bool operator()(const LocalMemoryValue& a, const LocalMemoryValue& b) const {
                    return a.value < b.value;
                }
            };
            std::sort(ordered.begin(), ordered.end(), ByValue());

            // Build origin tail
            std::vector<std::string> originalsTail;
            originalsTail.reserve(expressionList.size());
            for (std::size_t i = 0; i < expressionList.size(); ++i) {
                originalsTail.push_back(expressionList[i].original);
            }
            std::sort(originalsTail.begin(), originalsTail.end());

            for (std::size_t vi = 0; vi < ordered.size(); ++vi) {
                const LocalMemoryValue& lmv = ordered[vi];

                const std::string rplExpr1 = ce::replaceKeysInString(lmv.value, backReplacementMap);
                const std::string rplExpr2 = ExpressionAnalyzer::replaceUSubstrings(rplExpr1);

                if (lmv.value.find("marker") == std::string::npos) {
                    if (lmv.validityName != "main" && expressionListValidityName != "main"
                        && lmv.validityName != expressionListValidityName) {
                        continue;
                    }

                    std::set<int> tempLevels = combinedLevels;
                    for (const auto& lv : lmv.levels) tempLevels.insert(lv);

                    std::pair<std::string, std::vector<ExpressionWithValidity>> origin =
                        std::make_pair("implication", std::vector<ExpressionWithValidity>());
                    origin.second.push_back(ExpressionWithValidity(lmv.originalImplication, lmv.validityName));
                    for (std::size_t oi = 0; oi < originalsTail.size(); ++oi) {
                        origin.second.push_back(ExpressionWithValidity(originalsTail[oi], expressionListValidityName));
                    }

                    bool doNotDisintegrate = (lmv.justification == "integration");

                    bool allGood = expressionListValidityName == "main";
                    for (const auto& expr : expressionList) {
                        if (!(allowedForMail(expr.original, memoryBlock))) {
                            allGood = false;
                        }
                    }
                    if (allGood) {
                        memoryBlock.canBeSentSet.insert(rplExpr2);
                    }

                    this->addExprToMemoryBlock(rplExpr2, memoryBlock, iteration, 1,
                        tempLevels, origin, coreId, -1, expressionListValidityName, doNotDisintegrate);
                }
                else {
                    if (!pure) continue;

                    std::vector<std::string> replacedKey;
                    replacedKey.reserve(lmv.key.size());
                    for (std::size_t k = 0; k < lmv.key.size(); ++k) {
                        const std::string tmp = ce::replaceKeysInString(lmv.key[k], backReplacementMap);
                        replacedKey.push_back(ExpressionAnalyzer::replaceUSubstrings(tmp));
                    }

                    std::set<std::string> remainingArgs = lmv.remainingArgs;

                    auto itAdm = memoryBlock.overallHashMemory.admissionMap.find(
                        ExpressionWithValidity(rplExpr2, expressionListValidityName));
                    AdmissionMapValue admv;
                    admv.key = replacedKey;
                    admv.remainingArgs = remainingArgs;
                    admv.standardMaxAdmissionDepth = parameters.standardMaxAdmissionDepth;
                    admv.standardMaxSecondaryNumber = parameters.standardMaxSecondaryNumber;
                    admv.flag = false;

                    if (memoryBlock.overallHashMemory.consumedAdmissionKeys.find(
                        ExpressionWithValidity(rplExpr2, expressionListValidityName))
                        != memoryBlock.overallHashMemory.consumedAdmissionKeys.end()) {
                        continue;
                    }

                    std::vector<std::string> tempArgs = ce::getArgs(rplExpr2);
                    std::set<std::string> argSet(tempArgs.begin(), tempArgs.end());
                    assert(argSet.find("marker") != argSet.end());
                    argSet.erase("marker");

                    prepareIntegration(rplExpr2, argSet, memoryBlock, expressionListValidityName);

                    std::string core = ce::extractExpression(rplExpr2);
                    auto itComp = compiledExpressions.find(core);
                    assert(itComp != compiledExpressions.end());
                    if (itComp->second.category != "atomic") {
                        memoryBlock.canBeSentMarkerSet.insert(rplExpr2);
                    }

                    if (itAdm != memoryBlock.overallHashMemory.admissionMap.end()) {
                        itAdm->second.insert(admv);
                    } else {
                        std::set<AdmissionMapValue> newSet;
                        newSet.insert(admv);
                        memoryBlock.overallHashMemory.admissionMap.insert(
                            std::make_pair(ExpressionWithValidity(rplExpr2, expressionListValidityName), newSet));
                    }

                    memoryBlock.overallHashMemory.admissionStatusMap[
                        ExpressionWithValidity(rplExpr2, expressionListValidityName)] = false;

                    this->revisitRejected2(rplExpr2, memoryBlock, expressionListValidityName);
                }
            }
        }
    }
}

// ========================================================================
// Static checkLocalEncodedMemory: int pre-hit checks, string post-hit
// Takes StaticRequest — no string data in the request itself.
// ========================================================================

void ExpressionAnalyzer::checkLocalEncodedMemoryStatic(
    const StaticRequest& req,
    Memory& memoryBlock,
    unsigned coreId)
{
    NameMap& nm = memoryBlock.nameMap;
    const int16_t reqCount = req.count;
    const int iteration = req.maxIteration + 1;
    const IntNormalizedKey& tple = req.normalizedKey;

    // 1. Validity check using int fields
    const int16_t mainValidityId = nm.encode("main");
    int16_t consensusValidityId = mainValidityId;
    bool foundNonMain = false;
    for (int16_t i = 0; i < reqCount; ++i) {
        const int16_t vid = req.intExprs[i]->validityId;
        if (vid != mainValidityId) {
            if (!foundNonMain) {
                consensusValidityId = vid;
                foundNonMain = true;
            } else if (vid != consensusValidityId) {
                return;
            }
        }
    }

    const std::string expressionListValidityName = nm.decode(consensusValidityId);

    if (expressionListValidityName.find("_hypo_") != std::string::npos) {
        for (int16_t i = 0; i < reqCount; ++i) {
            if (req.intExprs[i]->validityId != consensusValidityId) {
                if (!req.intExprs[i]->isAnchor) {
                    return;
                }
            }
        }
    }

    if (memoryBlock.validityNamesToFilter.count(expressionListValidityName)) return;

    // 2. productsOfRecursion check using int fields
    int algebraicCounter = 0;
    const auto& prodRecIds = memoryBlock.overallHashMemory.productsOfRecursionIds;
    for (int16_t i = 0; i < reqCount; ++i) {
        const IntEncodedExpr& ex = *req.intExprs[i];
        for (int16_t a = 0; a < ex.arity; ++a) {
            if (ex.argIteration[a] > -1) {
                if (prodRecIds.find(ex.argFullId[a]) == prodRecIds.end()) {
                    algebraicCounter++;
                }
            }
        }
    }
    const bool pure = (algebraicCounter == 0);

    // 3. combinedLevels via statementLevelsMap (decoded lookup, per-request cost)
    std::set<int> combinedLevels;
    for (int16_t i = 0; i < reqCount; ++i) {
        EncodedExpression lookup;
        lookup.original = nm.decode(req.intExprs[i]->originalId);
        lookup.validityName = nm.decode(req.intExprs[i]->validityId);
        auto itL = memoryBlock.statementLevelsMap.find(lookup);
        if (itL != memoryBlock.statementLevelsMap.end()) {
            for (const auto& lv : itL->second) combinedLevels.insert(lv);
        }
    }

    // 4. Build intAllArgs from int fields
    std::set<int16_t> intAllArgs;
    for (int16_t i = 0; i < reqCount; ++i) {
        const IntEncodedExpr& ex = *req.intExprs[i];
        for (int16_t a = 0; a < ex.arity; ++a) {
            intAllArgs.insert(ex.argFullId[a]);
        }
    }

    if (memoryBlock.overallHashMemory.encodedMap.empty()) return;

    // 5. Candidate loop: find remaining-arg sets that are subsets of intAllArgs (sorted for determinism)
    std::vector<std::set<int16_t>> candidates;
    for (const auto& kv : memoryBlock.overallHashMemory.remainingArgsNormalizedEncodedMap) {
        const std::set<int16_t>& st = kv.first;
        bool subsetOk = true;
        for (const auto& arg : st) {
            if (intAllArgs.find(arg) == intAllArgs.end()) {
                subsetOk = false; break;
            }
        }
        if (subsetOk) candidates.push_back(st);
    }
    std::sort(candidates.begin(), candidates.end());

    for (const auto& intSt : candidates) {
        auto itRA = memoryBlock.overallHashMemory.remainingArgsNormalizedEncodedMap.find(intSt);
        if (itRA == memoryBlock.overallHashMemory.remainingArgsNormalizedEncodedMap.end()) continue;

        const std::unordered_set<IntNormalizedKey, IntNormalizedKeyHash>& nkSet = itRA->second;
        if (nkSet.find(tple) == nkSet.end()) continue;

        int16_t buf[ExecutionParameters::MAX_KEY_SLOTS];
        int16_t reverseMap[ExecutionParameters::MAX_KEY_SLOTS];
        std::memset(reverseMap, 0, sizeof(reverseMap));
        int16_t numNormVars = 0;

        // Build pointer array from request
        const IntEncodedExpr* reqExprs[ExecutionParameters::MAX_EXPRESSIONS];
        for (int16_t i = 0; i < reqCount; ++i) reqExprs[i] = req.intExprs[i];

        const int16_t len = makeIntNormalizedKeyFromEncodedWithMap(
            reqExprs, reqCount, intSt,
            buf, ExecutionParameters::MAX_KEY_SLOTS,
            reverseMap, numNormVars);

        IntNormalizedKey probe(reqCount, buf, len);
        auto itEnc = memoryBlock.overallHashMemory.encodedMap.find(probe);
        if (itEnc == memoryBlock.overallHashMemory.encodedMap.end()) {
            continue;
        }

        // HIT — build string back-replacement map from int reverseMap
        std::map<std::string, std::string> backReplacementMap;
        for (int16_t v = 1; v <= numNormVars; ++v) {
            backReplacementMap[std::to_string(static_cast<int>(v))] = nm.decode(reverseMap[v]);
        }

        const std::vector<LocalMemoryValue>& valueVec = itEnc->second;

        std::vector<LocalMemoryValue> ordered(valueVec.begin(), valueVec.end());
        struct ByValue {
            bool operator()(const LocalMemoryValue& a, const LocalMemoryValue& b) const {
                return a.value < b.value;
            }
        };
        std::sort(ordered.begin(), ordered.end(), ByValue());

        // Build origin tail from decoded originals (hit path only)
        std::vector<std::string> originalsTail;
        originalsTail.reserve(reqCount);
        for (int16_t i = 0; i < reqCount; ++i) {
            originalsTail.push_back(nm.decode(req.intExprs[i]->originalId));
        }
        std::sort(originalsTail.begin(), originalsTail.end());

        for (std::size_t vi = 0; vi < ordered.size(); ++vi) {
            const LocalMemoryValue& lmv = ordered[vi];

            const std::string rplExpr1 = ce::replaceKeysInString(lmv.value, backReplacementMap);
            const std::string rplExpr2 = ExpressionAnalyzer::replaceUSubstrings(rplExpr1);

            if (lmv.value.find("marker") == std::string::npos) {
                if (lmv.validityName != "main" && expressionListValidityName != "main"
                    && lmv.validityName != expressionListValidityName) {
                    continue;
                }

                std::set<int> tempLevels = combinedLevels;
                for (const auto& lv : lmv.levels) tempLevels.insert(lv);

                std::pair<std::string, std::vector<ExpressionWithValidity>> origin =
                    std::make_pair("implication", std::vector<ExpressionWithValidity>());
                origin.second.push_back(ExpressionWithValidity(lmv.originalImplication, lmv.validityName));
                for (std::size_t oi = 0; oi < originalsTail.size(); ++oi) {
                    origin.second.push_back(ExpressionWithValidity(originalsTail[oi], expressionListValidityName));
                }

                const bool doNotDisintegrate = (lmv.justification == "integration");

                bool allGood = expressionListValidityName == "main";
                for (int16_t i = 0; i < reqCount; ++i) {
                    if (!allowedForMail(nm.decode(req.intExprs[i]->originalId), memoryBlock)) {
                        allGood = false;
                    }
                }
                if (allGood) {
                    memoryBlock.canBeSentSet.insert(rplExpr2);
                }

                this->addExprToMemoryBlock(rplExpr2, memoryBlock, iteration, 1,
                    tempLevels, origin, coreId, -1, expressionListValidityName, doNotDisintegrate);
            }
            else {
                if (!pure) continue;

                std::vector<std::string> replacedKey;
                replacedKey.reserve(lmv.key.size());
                for (std::size_t k = 0; k < lmv.key.size(); ++k) {
                    const std::string tmp = ce::replaceKeysInString(lmv.key[k], backReplacementMap);
                    replacedKey.push_back(ExpressionAnalyzer::replaceUSubstrings(tmp));
                }

                std::set<std::string> remainingArgs = lmv.remainingArgs;

                auto itAdm = memoryBlock.overallHashMemory.admissionMap.find(
                    ExpressionWithValidity(rplExpr2, expressionListValidityName));
                AdmissionMapValue admv;
                admv.key = replacedKey;
                admv.remainingArgs = remainingArgs;
                admv.standardMaxAdmissionDepth = parameters.standardMaxAdmissionDepth;
                admv.standardMaxSecondaryNumber = parameters.standardMaxSecondaryNumber;
                admv.flag = false;

                if (memoryBlock.overallHashMemory.consumedAdmissionKeys.find(
                    ExpressionWithValidity(rplExpr2, expressionListValidityName))
                    != memoryBlock.overallHashMemory.consumedAdmissionKeys.end()) {
                    continue;
                }

                std::vector<std::string> tempArgs = ce::getArgs(rplExpr2);
                std::set<std::string> argSet(tempArgs.begin(), tempArgs.end());
                assert(argSet.find("marker") != argSet.end());
                argSet.erase("marker");

                prepareIntegration(rplExpr2, argSet, memoryBlock, expressionListValidityName);

                std::string core = ce::extractExpression(rplExpr2);
                auto itComp = compiledExpressions.find(core);
                assert(itComp != compiledExpressions.end());
                if (itComp->second.category != "atomic") {
                    memoryBlock.canBeSentMarkerSet.insert(rplExpr2);
                }

                if (itAdm != memoryBlock.overallHashMemory.admissionMap.end()) {
                    itAdm->second.insert(admv);
                } else {
                    std::set<AdmissionMapValue> newSet;
                    newSet.insert(admv);
                    memoryBlock.overallHashMemory.admissionMap.insert(
                        std::make_pair(ExpressionWithValidity(rplExpr2, expressionListValidityName), newSet));
                }

                memoryBlock.overallHashMemory.admissionStatusMap[
                    ExpressionWithValidity(rplExpr2, expressionListValidityName)] = false;

                this->revisitRejected2(rplExpr2, memoryBlock, expressionListValidityName);
            }
        }
    }
}

// ========================================================================
// Static pipeline: IntEncodedExpr-based, zero-alloc request generation
// ========================================================================

int16_t ExpressionAnalyzer::makeMandatoryEncodedStatementLists1Static(
    const HashMemory& mem,
    const IntEncodedExpr* stmts, int16_t stmtCount,
    int16_t* outIndices, int16_t maxOut)
{
    // Filter, then sort by originalId (proxy for stable sort by original string)
    int16_t filtBuf[4096];
    int16_t nFilt = filterIntEncodedStatements(stmts, stmtCount, mem, filtBuf, 4096);

    // Insertion sort filtered indices by originalId (counts are small)
    for (int16_t i = 1; i < nFilt; ++i) {
        int16_t key = filtBuf[i];
        int16_t j = i - 1;
        while (j >= 0 && stmts[filtBuf[j]].originalId > stmts[key].originalId) {
            filtBuf[j + 1] = filtBuf[j];
            --j;
        }
        filtBuf[j + 1] = key;
    }

    int16_t nOut = std::min(nFilt, maxOut);
    std::memcpy(outIndices, filtBuf, nOut * sizeof(int16_t));
    return nOut;
}

int16_t ExpressionAnalyzer::makeMandatoryEncodedStatementLists2Static(
    Memory& body, const HashMemory& mem,
    const IntEncodedExpr* first, int16_t firstCount,
    const IntEncodedExpr* second, int16_t secondCount,
    MandatoryPair* outPairs, int16_t maxOut)
{
    if (firstCount == 0 || secondCount == 0) return 0;

    // Filter both layers
    int16_t filt1Buf[4096], filt2Buf[4096];
    int16_t nF1 = filterIntEncodedStatements(first, firstCount, mem, filt1Buf, 4096);
    int16_t nF2 = filterIntEncodedStatements(second, secondCount, mem, filt2Buf, 4096);

    // Sort by originalId
    auto sortByOriginal = [](int16_t* arr, int16_t n, const IntEncodedExpr* stmts) {
        for (int16_t i = 1; i < n; ++i) {
            int16_t key = arr[i];
            int16_t j = i - 1;
            while (j >= 0 && stmts[arr[j]].originalId > stmts[key].originalId) {
                arr[j + 1] = arr[j];
                --j;
            }
            arr[j + 1] = key;
        }
    };
    sortByOriginal(filt1Buf, nF1, first);
    sortByOriginal(filt2Buf, nF2, second);

    // Cache "main" ID for fast validity check
    int16_t mainId = body.nameMap.encode("main");

    int16_t nOut = 0;
    int16_t buf[ExecutionParameters::MAX_KEY_SLOTS];

    for (int16_t i = 0; i < nF1 && nOut < maxOut; ++i) {
        const IntEncodedExpr& e1 = first[filt1Buf[i]];
        for (int16_t j = 0; j < nF2 && nOut < maxOut; ++j) {
            const IntEncodedExpr& e2 = second[filt2Buf[j]];

            // Validity check
            if (e1.validityId != e2.validityId) {
                if (e1.validityId != mainId && e2.validityId != mainId) continue;
            }

            // Skip identical expressions
            if (e1.originalId == e2.originalId && e1.validityId == e2.validityId)
                continue;

            // Build sorted pair and preEvaluate — sort by name string (lexicographic),
            // NOT by nameId (insertion order), to match storage sort in addToHashMemory
            IntEncodedExpr sorted[2];
            const std::string& name1 = body.nameMap.decode(e1.nameId);
            const std::string& name2 = body.nameMap.decode(e2.nameId);
            if (name1 <= name2) { sorted[0] = e1; sorted[1] = e2; }
            else { sorted[0] = e2; sorted[1] = e1; }

            int16_t len = makeIntNormalizedKeyFromEncoded(sorted, 2, false, buf,
                ExecutionParameters::MAX_KEY_SLOTS);
            IntNormalizedKey probe(2, buf, len);

            if (mem.normalizedEncodedSubkeys.find(probe) != mem.normalizedEncodedSubkeys.end()) {
                outPairs[nOut].idx1 = filt1Buf[i];
                outPairs[nOut].idx2 = filt2Buf[j];
                ++nOut;
            }
        }
    }
    return nOut;
}

bool ExpressionAnalyzer::getValidityName(const std::vector<EncodedExpression>& req, std::string& validityName)
{
    std::string nonMain = "";
    bool foundNonMain = false;

    // 1. Determine the consensus validity name (same as original logic)
    for (const auto& expr : req) {
        if (expr.validityName != "main") {
            if (!foundNonMain) {
                nonMain = expr.validityName;
                foundNonMain = true;
            }
            else {
                if (nonMain != expr.validityName) {
                    return false; // Error: multiple different non-main validity names
                }
            }
        }
    }

    if (foundNonMain) {
        validityName = nonMain;
    }
    else {
        validityName = "main";
    }

    // 2. Extended check: If validityName contains "_hypo_"
    if (validityName.find("_hypo_") != std::string::npos) {
        for (const auto& expr : req) {
            // Requirement: Must match validityName OR be an Anchor
            if (expr.validityName != validityName) {
                std::string core = ce::extractExpression(expr.original);
                // Check if core starts with "Anchor"
                if (core.rfind("Anchor", 0) != 0) {
                    return false;
                }
            }
        }
    }

    return true;
}



//#pragma optimize("", off)
void ExpressionAnalyzer::reactToHypo(Memory& mb) {
    static const std::regex reInt(R"(^int_lev_\d+_\d+$)");
    const std::string hypoMarker = "_hypo_";

    // Iterate through all equivalence classes maps
    for (const auto& [validityName, classes] : mb.equivalenceClassesMap) {

        // 1. Find the LAST occurrence of "_hypo_"
        // The validity name format is now: ..._var0_x_var1_y_hypo_expr
        size_t posHypo = validityName.rfind(hypoMarker);

        if (posHypo == std::string::npos) {
            continue;
        }

        // 2. Parse variables from validityName
        // We look for the LAST occurrence of "_var0_" BEFORE posHypo
        std::string startMarker = "_var0_";
        size_t posVar = validityName.rfind(startMarker, posHypo);

        if (posVar == std::string::npos) continue;

        // EXTRACT: The original validity name (before the current hypothesis vars)
        std::string parentValidity = validityName.substr(0, posVar);

        std::set<std::string> targetVars;
        int idx = 0;
        size_t currentPos = posVar;
        size_t suffixPos = posHypo; // Variables end where "_hypo_" begins

        while (true) {
            std::string currentMarker = "_var" + std::to_string(idx) + "_";

            // Verify the marker exists at the current position
            if (currentPos + currentMarker.size() > validityName.size() ||
                validityName.compare(currentPos, currentMarker.size(), currentMarker) != 0) {
                // Sequence broken or malformed
                break;
            }

            size_t startContent = currentPos + currentMarker.size();

            // Prepare to find the next marker
            std::string nextMarker = "_var" + std::to_string(idx + 1) + "_";
            // Search for next marker, bounded by suffixPos
            size_t nextPos = validityName.find(nextMarker, startContent);

            size_t endContent;
            bool isLast = false;

            if (nextPos == std::string::npos || nextPos >= suffixPos) {
                // If next marker not found or appears after/at the hypo marker, this is the last variable
                endContent = suffixPos;
                isLast = true;
            }
            else {
                endContent = nextPos;
            }

            // Extract variable name
            std::string extractedVar = validityName.substr(startContent, endContent - startContent);
            targetVars.insert(extractedVar);

            if (isLast) break;

            // Move to next iteration
            currentPos = nextPos;
            idx++;
        }

        if (targetVars.empty()) continue;

        // 3. Check Equivalence Classes
        for (const auto& eqClass : classes) {
            for (const std::string& var : eqClass.variables) {

                // Check if this variable is one of the target variables
                if (targetVars.find(var) == targetVars.end()) continue;

                // Check "other" variables constraint
                bool conditionMet = true;
                bool hasOthers = false;

                for (const std::string& other : eqClass.variables) {
                    if (other == var) continue;

                    hasOthers = true;
                    // Check if other variable matches "int_lev_d+_d+"
                    if (!std::regex_match(other, reInt)) {
                        conditionMet = false;
                        break;
                    }
                }

                // If yes, add (=[var, var_copy])
                if (conditionMet && hasOthers) {
                    std::string Y = var;
                    std::string Y_copy = Y + "_copy";
                    std::string equalityExpr = "(=[" + Y + "," + Y_copy + "])";

                    std::string finalExpr = removeUPrefixFromArguments(equalityExpr);

                    std::set<int> levels;
                    levels.insert(mb.level);

                    std::pair<std::string, std::vector<ExpressionWithValidity>> origin;
                    origin.first = "reaction to hypo";
                    origin.second.push_back(ExpressionWithValidity(equalityExpr, parentValidity));

                    addExprToMemoryBlock(finalExpr,
                        mb,
                        -1, // iteration
                        0,  // status (Local statement)
                        levels,
                        origin,
                        -1, // coreId
                        -1, // auxyIndex
                        parentValidity, // UPDATED: Use the extracted original validity name
                        false);
                }
            }
        }
    }
}


// =================== Main driver ===================

// Turn optimizations OFF for just this section
//#pragma optimize("", off)

Memory& ExpressionAnalyzer::performElementaryLogicalStep(Memory& body,
    unsigned coreId,
    const ParentChildrenMap& index,
    PerCoreMailboxes& boxes) {

    if (!body.isActive) {
        return body;
    }

    // ingest implications into local memory
    for (std::set< std::tuple< std::vector<std::string>,
        std::string,
        std::set<std::string>,
        std::set<int>,
        std::string> >::const_iterator it = body.mailIn.implications.begin();
        it != body.mailIn.implications.end(); ++it) {
        const std::vector<std::string>& chain = std::get<0>(*it);
        const std::string& head = std::get<1>(*it);
        const std::set<std::string>& remainingArgsKey = std::get<2>(*it);
        const std::set<int>& levels = std::get<3>(*it);
        const std::string& orImpl = std::get<4>(*it);

        this->addToHashMemory(chain, head, remainingArgsKey, body, body.overallHashMemory, levels,
            orImpl, parameters.standardMaxAdmissionDepth, parameters.standardMaxSecondaryNumber, false,
            parameters.minNumOperatorsKey, "implication", true, orImpl, "main");
    }

    // expr_origin_map = mail_in | current   (right wins on conflicts)
    {
        // Sort mail origins for deterministic encoding order
        for (auto& kv : body.mailIn.exprOriginMap) {
            std::sort(kv.second.begin(), kv.second.end());
        }

        std::map<ExpressionWithValidity, std::vector<std::pair<std::string, std::vector<ExpressionWithValidity>>>> merged;
        for (const auto& kv : body.mailIn.exprOriginMap) merged[kv.first] = kv.second;
        for (auto it = body.exprOriginMap.begin(); it != body.exprOriginMap.end(); ++it) merged[it->first] = it->second;
        body.exprOriginMap.swap(merged);
    }

    // Arena for IntEncodedExpr copies (stable pointers through eval)
    TypedArena<IntEncodedExpr> exprArena(65536);

    // Request buffer
    std::vector<StaticRequest> reqBuf;
    reqBuf.resize(8192);
    int16_t totalReqs = 0;

    // Sort mail statements for deterministic absorption
    std::vector< std::pair<std::string, std::set<int> > > sortedStatements(
        body.mailIn.statements.begin(), body.mailIn.statements.end());

    if (ceFilteringActive) {
        // --- CE mode: absorb mail, then enumerate all statement combos ---
        for (std::vector< std::pair<std::string, std::set<int> > >::const_iterator it = sortedStatements.begin();
            it != sortedStatements.end(); ++it) {
            const std::string& statement = it->first;
            const std::set<int>& levels = it->second;
            EncodedExpression encStmt(statement, "main");
            ExpressionWithValidity stVal(statement, "main");
            if (body.statementLevelsMap.find(encStmt) == body.statementLevelsMap.end()) {
                std::pair<std::string, std::vector<ExpressionWithValidity>> origin;
                if (parameters.trackHistory) {
                    auto oit = body.mailIn.exprOriginMap.find(stVal);
                    if (oit != body.mailIn.exprOriginMap.end() && !oit->second.empty()) {
                        origin = oit->second.front();
                    }
                }
                addExprToMemoryBlock(statement, body, -1, 3, levels, origin, coreId, -1, "main", false);
            }
        }

        totalReqs = this->generateEncodedRequestsStaticCE(body, body.overallHashMemory,
            exprArena, reqBuf.data(), 8192);

    } else {
        // --- Normal mode: 5-batch mandatory pipeline ---

        // working memory (only from mail_in implications)
        Memory working;
        for (std::set< std::tuple< std::vector<std::string>,
            std::string,
            std::set<std::string>,
            std::set<int>,
            std::string> >::const_iterator it = body.mailIn.implications.begin();
            it != body.mailIn.implications.end(); ++it) {
            const std::vector<std::string>& chain = std::get<0>(*it);
            const std::string& head = std::get<1>(*it);
            const std::set<std::string>& remainingArgsKey = std::get<2>(*it);
            const std::set<int>& levels = std::get<3>(*it);
            const std::string& orImpl = std::get<4>(*it);
            this->addToHashMemory(chain, head, remainingArgsKey, body, working.overallHashMemory, levels,
                orImpl, parameters.standardMaxAdmissionDepth, parameters.standardMaxSecondaryNumber, false,
                parameters.minNumOperatorsKey, "implication", true, orImpl, "main");
        }

        // Encode mail statements for static filter
        std::vector<IntEncodedExpr> mailIntEncoded;
        mailIntEncoded.reserve(sortedStatements.size());
        for (const auto& st : sortedStatements) {
            EncodedExpression tmp(st.first, "main");
            mailIntEncoded.push_back(encodeExpression(tmp, body.nameMap));
        }

        int16_t mslBuf[4096];

        // --- Batch 1: working memory from mail_in implications ---
        if (!working.overallHashMemory.encodedMap.empty()) {
            int16_t nMsl1 = this->makeMandatoryEncodedStatementLists1Static(
                working.overallHashMemory,
                body.intLocalEncodedStatements.data(),
                static_cast<int16_t>(body.intLocalEncodedStatements.size()),
                mslBuf, 4096);
            if (nMsl1 > 0) {
                totalReqs += this->generateEncodedRequestsStatic(body, working.overallHashMemory,
                    mslBuf, nMsl1,
                    body.intLocalEncodedStatements.data(),
                    exprArena,
                    reqBuf.data() + totalReqs, static_cast<int16_t>(8192 - totalReqs));
            }
        }

        // absorb incoming statements
        for (std::vector< std::pair<std::string, std::set<int> > >::const_iterator it = sortedStatements.begin();
            it != sortedStatements.end(); ++it) {
            const std::string& statement = it->first;
            const std::set<int>& levels = it->second;
            EncodedExpression encStmt(statement, "main");
            ExpressionWithValidity stVal(statement, "main");
            if (body.statementLevelsMap.find(encStmt) == body.statementLevelsMap.end()) {
                std::pair<std::string, std::vector<ExpressionWithValidity>> origin;
                if (parameters.trackHistory) {
                    auto oit = body.mailIn.exprOriginMap.find(stVal);
                    if (oit != body.mailIn.exprOriginMap.end() && !oit->second.empty()) {
                        origin = oit->second.front();
                    }
                }
                addExprToMemoryBlock(statement, body, -1, 3, levels, origin, coreId, -1, "main", false);
            }
        }

        // --- Batch 2: new local delta ---
        {
            int16_t nMsl2 = this->makeMandatoryEncodedStatementLists1Static(
                body.overallHashMemory,
                body.intLocalEncodedStatementsDelta.data(),
                static_cast<int16_t>(body.intLocalEncodedStatementsDelta.size()),
                mslBuf, 4096);
            if (nMsl2 > 0) {
                totalReqs += this->generateEncodedRequestsStatic(body, body.overallHashMemory,
                    mslBuf, nMsl2,
                    body.intLocalEncodedStatementsDelta.data(),
                    exprArena,
                    reqBuf.data() + totalReqs, static_cast<int16_t>(8192 - totalReqs));
            }
        }

        // --- Batch 3: local × mail pairs (fully static) ---
        {
            MandatoryPair pairsBuf[8192];
            int16_t nPairs = this->makeMandatoryEncodedStatementLists2Static(
                body, body.overallHashMemory,
                body.intLocalEncodedStatements.data(),
                static_cast<int16_t>(body.intLocalEncodedStatements.size()),
                mailIntEncoded.data(),
                static_cast<int16_t>(mailIntEncoded.size()),
                pairsBuf, 8192);
            if (nPairs > 0) {
                totalReqs += this->generateEncodedRequestsStaticPairs(body, body.overallHashMemory,
                    pairsBuf, nPairs,
                    body.intLocalEncodedStatements.data(),
                    mailIntEncoded.data(),
                    exprArena,
                    reqBuf.data() + totalReqs, static_cast<int16_t>(8192 - totalReqs));
            }
        }

        // --- Batch 4: localHashMemory — mail mandatory singles ---
        if (!body.localHashMemory.encodedMap.empty() && !mailIntEncoded.empty()) {
            int16_t nMsl4 = this->makeMandatoryEncodedStatementLists1Static(
                body.localHashMemory,
                mailIntEncoded.data(),
                static_cast<int16_t>(mailIntEncoded.size()),
                mslBuf, 4096);
            if (nMsl4 > 0) {
                totalReqs += this->generateEncodedRequestsStatic(body, body.localHashMemory,
                    mslBuf, nMsl4,
                    mailIntEncoded.data(),
                    exprArena,
                    reqBuf.data() + totalReqs, static_cast<int16_t>(8192 - totalReqs));
            }
        }

        // --- Batch 5: localHashMemoryDelta ---
        if (!body.localHashMemoryDelta.encodedMap.empty()) {
            int16_t nMsl5 = this->makeMandatoryEncodedStatementLists1Static(
                body.localHashMemoryDelta,
                body.intEncodedStatements.data(),
                static_cast<int16_t>(body.intEncodedStatements.size()),
                mslBuf, 4096);
            if (nMsl5 > 0) {
                totalReqs += this->generateEncodedRequestsStatic(body, body.localHashMemoryDelta,
                    mslBuf, nMsl5,
                    body.intEncodedStatements.data(),
                    exprArena,
                    reqBuf.data() + totalReqs, static_cast<int16_t>(8192 - totalReqs));
            }
        }
    } // end normal mode

    // clear mail_in and local delta
    body.mailIn.statements.clear();
    body.mailIn.implications.clear();
    body.mailIn.exprOriginMap.clear();

    if (body.contradictionIndex == -1) {
        body.localEncodedStatementsDelta.clear();
        body.intLocalEncodedStatementsDelta.clear();
    }

    body.localHashMemoryDelta = HashMemory();

    // Fixpoint eval loop — ONE loop, fully static requests
    {
        bool changed = true;
        while (changed) {
            changed = false;
            if (!body.isActive) return body;

            for (int16_t r = 0; r < totalReqs; ++r) {
                const StaticRequest& req = reqBuf[r];

                // Dependency check via intKnownStatements (O(1) per element)
                bool skip = false;
                for (int16_t s = 0; s < req.count; ++s) {
                    int32_t key = packStatementKey(req.intExprs[s]->originalId,
                                                   req.intExprs[s]->validityId);
                    if (body.intKnownStatements.find(key) == body.intKnownStatements.end()) {
                        skip = true; break;
                    }
                }
                if (skip) continue;

                const std::size_t prevSize = body.intKnownStatements.size();
                this->checkLocalEncodedMemoryStatic(req, body, coreId);
                if (body.intKnownStatements.size() > prevSize) {
                    changed = true;
                }
            }
        }
    }

    // Send to per-core mailboxes for children of 'body'
    this->sendMail(body, body.mailOut, coreId, index, boxes);
    body.mailOut.statements.clear();
    body.mailOut.implications.clear();
    body.mailOut.exprOriginMap.clear();

	reactToHypo(body);

    return body;
}

std::set<std::string>
ExpressionAnalyzer::findImmutableArgs(const std::string& theorem,
    const std::string& digit) {
    std::set<std::string> immutables;

    // Disintegrate implication to a chain of left subexpressions.
    std::vector< std::tuple<
        std::string,                  // leftExpr
        std::vector<std::string>,     // args of current '>'
        std::set<std::string>         // left node arguments (unused here)
    > > tempChain;

    ce::disintegrateImplication(theorem, tempChain, this->coreExpressionMap);

    std::vector<std::string> chain;
    chain.reserve(tempChain.size());
    for (std::size_t i = 0; i < tempChain.size(); ++i) {
        chain.push_back(std::get<0>(tempChain[i]));
    }

    // Start with all digit args except the given 'digit'
    std::set<std::string> digits = findDigitArgs(theorem, anchorInfo, coreExpressionMap);
    immutables.insert(digits.begin(), digits.end());
    immutables.erase(digit);

    // Propagate immutability forward along the chain
    bool found = true;
    while (found) {
        found = false;

        for (std::size_t i = 0; i < chain.size(); ++i) {
            const std::string& expression = chain[i];
            const std::string coreExpr = ce::extractExpression(expression);

            // Use the pre-populated 'operators' attribute for efficiency
            if (this->operators.find(coreExpr) != this->operators.end()) {

                auto it = this->coreExpressionMap.find(coreExpr);
                assert(it != this->coreExpressionMap.end());

                const ce::CoreExpressionConfig& cfg = it->second;
                const std::vector<std::string> args = ce::getArgs(expression);

                // 1. Check if all configured Inputs are already Immutable
                bool inputsAreImmutable = true;
                for (int idx : cfg.inputIndices) {
                    // BLUNT ASSERT: Fail if config index is out of bounds for this expression
                    assert(idx >= 0 && idx < static_cast<int>(args.size()));

                    if (immutables.find(args[idx]) == immutables.end()) {
                        inputsAreImmutable = false;
                        break;
                    }
                }

                // 2. If inputs are immutable, add configured Outputs to immutables
                if (inputsAreImmutable) {
                    for (int idx : cfg.outputIndices) {
                        // BLUNT ASSERT: Fail if config index is out of bounds
                        assert(idx >= 0 && idx < static_cast<int>(args.size()));

                        const std::string& outputArg = args[idx];
                        if (immutables.find(outputArg) == immutables.end()) {
                            immutables.insert(outputArg);
                            found = true;
                        }
                    }
                }
            }
        }
    }

    return immutables;
}

Memory* ExpressionAnalyzer::accessMemory(const std::vector<std::string>& theoremKey,
    Memory& bodyOfProves1) {
    Memory* memoryBlock = &bodyOfProves1;
    for (std::size_t i = 0; i < theoremKey.size(); ++i) {
        const std::string& key = theoremKey[i];
        std::map<std::string, Memory*>::iterator it = memoryBlock->simpleMap.find(key);
        if (it == memoryBlock->simpleMap.end() || it->second == NULL) {
            return NULL;
        }
        memoryBlock = it->second; // follow the pointer
    }
    return memoryBlock;
}

bool ExpressionAnalyzer::tryBackReformulateOperatorHead(
    const std::string& theorem,
    std::string& backReformulated)
{
    if (!parameters.incubator_mode) return false;

    std::vector<std::tuple<std::string, std::vector<std::string>, std::set<std::string>>> tempChain;
    std::string head = ce::disintegrateImplication(theorem, tempChain, coreExpressionMap);

    // Must have exactly 2 chain elements (anchor + operator)
    if (tempChain.size() != 2) return false;

    // First chain element must be an Anchor
    {
        std::string firstName = ce::extractExpression(std::get<0>(tempChain[0]));
        if (firstName.find("Anchor") == std::string::npos) return false;
    }

    // Head must be equality or negated equality
    bool negated = false;
    std::string eqExpr = head;
    if (!head.empty() && head[0] == '!') {
        negated = true;
        eqExpr = head.substr(1);
    }

    std::string headExpr = ce::extractExpression(eqExpr);
    if (headExpr != "=") return false;

    std::vector<std::string> eqArgs = ce::getArgs(eqExpr);
    if (eqArgs.size() != 2) return false;

    // Last chain element must be an operator
    const std::string& lastPremise = std::get<0>(tempChain.back());
    std::string opName = ce::extractExpression(lastPremise);
    auto cfgIt = coreExpressionMap.find(opName);
    if (cfgIt == coreExpressionMap.end()) return false;
    if (cfgIt->second.outputIndices.empty()) return false;  // not an operator

    int outIdx = cfgIt->second.outputIndices[0];  // 0-based
    std::vector<std::string> opArgs = ce::getArgs(lastPremise);
    if (outIdx >= static_cast<int>(opArgs.size())) return false;

    std::string x = opArgs[outIdx];  // the fresh variable in operator

    // Verify: one of the equality args must be x
    std::string a_out;
    if (eqArgs[0] == x) {
        a_out = eqArgs[1];
    } else if (eqArgs[1] == x) {
        a_out = eqArgs[0];
    } else {
        return false;  // x doesn't appear in equality — not our pattern
    }

    // Back-reformulate: replace x with a_out in the operator, drop the equality + inner >
    opArgs[outIdx] = a_out;
    std::string cleanHead = "(" + opName + "[";
    for (size_t i = 0; i < opArgs.size(); ++i) {
        if (i > 0) cleanHead += ",";
        cleanHead += opArgs[i];
    }
    cleanHead += "])";

    // Negate the operator head if the equality was negated
    if (negated) {
        cleanHead = "!" + cleanHead;
    }

    // Rebuild chain without the last element (operator was a premise, now it's the head)
    std::vector<std::string> outerChain;
    for (size_t i = 0; i < tempChain.size() - 1; ++i) {
        outerChain.push_back(std::get<0>(tempChain[i]));
    }

    backReformulated = reconstructImplication(outerChain, cleanHead);
    return true;
}

// analyze_expressions.cpp
// Thread-safe: locks globalTheoremList only when mutating it.
void ExpressionAnalyzer::updateGlobalDirect(const std::string& theorem, int coreId) {
    // 1) Disintegrate to chain + head
    std::vector<std::tuple<std::string, std::vector<std::string>, std::set<std::string>>> tempChain;
    Mail mailOut;

    deactivateUnnecessary(theorem, this->body);  // deactivates all blocks tied to this theorem.

    std::string value = ce::disintegrateImplication(theorem, tempChain, this->coreExpressionMap);

    std::vector<std::string> ky;
    ky.reserve(tempChain.size());
    for (std::size_t i = 0; i < tempChain.size(); ++i) ky.push_back(std::get<0>(tempChain[i]));

    // Save original decomposition before sections 3/4 overwrite ky/value
    const std::vector<std::string> origKy = ky;
    const std::string origValue = value;

    // 2) Original implication broadcast deferred until after back-reformulation check (section 5)
    //    If back-reformulation succeeds, only the back-reformulated version is broadcast.
    bool willBackReformulate = false;

    std::vector<std::pair<std::string, std::string>> outTheorems;
    if (reformulateTheorem(theorem, outTheorems)) {
        for (const auto& pair : outTheorems) {
            const std::string& refTheoremRaw = pair.first;
            const std::string& refTheoremCompiled = pair.second;

            {
                std::lock_guard<std::mutex> lock(this->theoremListMutex);
                this->globalTheoremList.emplace_back(refTheoremCompiled, "reformulated statement", theorem, "-1");
            }
            std::cout << refTheoremCompiled << std::endl;

            if (parameters.trackHistory) {
                ExpressionWithValidity ev(refTheoremCompiled, "main");
                addOrigin(mailOut.exprOriginMap, ev, std::make_pair("theorem", std::vector<ExpressionWithValidity>()), (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
            }

            tempChain.clear();
            std::string valueR = ce::disintegrateImplication(refTheoremCompiled, tempChain, this->coreExpressionMap);

            ky.clear();
            ky.reserve(tempChain.size());
            for (std::size_t i = 0; i < tempChain.size(); ++i) ky.push_back(std::get<0>(tempChain[i]));

            Memory* memoryBlockR = accessMemory(ky, this->body);
            if (memoryBlockR != NULL) {
                std::pair<std::string, std::vector<ExpressionWithValidity>> originR;
                originR.first = "implication";
                originR.second.push_back(ExpressionWithValidity(refTheoremCompiled, "main"));
                for (const std::string& keyElement : ky) {
                    originR.second.emplace_back(keyElement, "main");
                }

                std::set<int> levelsR;
                for (int i = 0; i <= static_cast<int>(ky.size()); ++i) levelsR.insert(i);

                addStatement(valueR, *memoryBlockR, false, levelsR, originR, "main");
            }

            mailOut.implications.insert(
                std::make_tuple(ky, valueR, std::set<std::string>(), std::set<int>(), refTheoremCompiled)
            );

        }

    }

    // 4) Try to create reshuffled/mirrored variant (anchor-first)
    const std::string reshuffledMirrored =
        ce::createReshuffledMirrored(theorem, this->anchorInfo.name, true, this->coreExpressionMap);  // :contentReference[oaicite:3]{index=3}

    if (!reshuffledMirrored.empty()) {
        {
            std::lock_guard<std::mutex> lock(this->theoremListMutex);
            this->globalTheoremList.emplace_back(reshuffledMirrored, "mirrored statement", theorem, "-1");
        }

        if (parameters.trackHistory) {
            ExpressionWithValidity ev(reshuffledMirrored, "main");
            addOrigin(mailOut.exprOriginMap, ev, std::make_pair("theorem", std::vector<ExpressionWithValidity>()), (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
        }

        // rebuild chain for the mirrored expr
        tempChain.clear();
        value = ce::disintegrateImplication(reshuffledMirrored, tempChain, this->coreExpressionMap);

        ky.clear();
        ky.reserve(tempChain.size());
        for (std::size_t i = 0; i < tempChain.size(); ++i) ky.push_back(std::get<0>(tempChain[i]));

        // Queue the mirrored implication (no addStatement — let it fire naturally via hash burst)
        mailOut.implications.insert(
            std::make_tuple(ky, value, std::set<std::string>(), std::set<int>(), reshuffledMirrored)
        );
    }

    // 5) Incubator: back-reformulate operator-equality pattern
    // Must be before sendMail so the back-reformulated implication gets flushed.
    std::string backReformulated;
    willBackReformulate = tryBackReformulateOperatorHead(theorem, backReformulated);
    if (!willBackReformulate) {
        // No back-reformulation: broadcast and record original theorem
        mailOut.implications.insert(std::make_tuple(origKy, origValue, std::set<std::string>(), std::set<int>(), theorem));
        if (parameters.trackHistory) {
            ExpressionWithValidity ev(theorem, "main");
            addOrigin(mailOut.exprOriginMap, ev, std::make_pair("theorem", std::vector<ExpressionWithValidity>()), (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
        }
        {
            std::lock_guard<std::mutex> lock(this->theoremListMutex);
            this->globalTheoremList.emplace_back(theorem, "direct", "-1", "-1");
        }
        std::cout << theorem << std::endl;
    }
    if (willBackReformulate) {
        // Register original as "direct" so it gets a full proof chapter
        mailOut.implications.insert(std::make_tuple(origKy, origValue, std::set<std::string>(), std::set<int>(), theorem));
        if (parameters.trackHistory) {
            ExpressionWithValidity ev(theorem, "main");
            addOrigin(mailOut.exprOriginMap, ev, std::make_pair("theorem", std::vector<ExpressionWithValidity>()), (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
        }
        {
            std::lock_guard<std::mutex> lock(this->theoremListMutex);
            this->globalTheoremList.emplace_back(theorem, "direct", "-1", "-1");
            this->globalTheoremList.emplace_back(
                backReformulated, "incubator back reformulation", theorem, "-1");
        }
        std::cout << theorem << std::endl;
        std::cout << backReformulated << std::endl;

        if (parameters.trackHistory) {
            ExpressionWithValidity ev(backReformulated, "main");
            addOrigin(mailOut.exprOriginMap, ev,
                std::make_pair("theorem", std::vector<ExpressionWithValidity>()),
                (parameters.compressor_mode
                    ? parameters.compressor_max_origins_per_expr
                    : parameters.max_origin_per_expr));
        }

        std::vector<std::tuple<std::string, std::vector<std::string>, std::set<std::string>>> brChain;
        std::string brHead = ce::disintegrateImplication(backReformulated, brChain, coreExpressionMap);

        std::vector<std::string> brKey;
        brKey.reserve(brChain.size());
        for (std::size_t i = 0; i < brChain.size(); ++i)
            brKey.push_back(std::get<0>(brChain[i]));

        // Queue the back-reformulated implication (no addStatement — let it fire naturally via hash burst)
        mailOut.implications.insert(
            std::make_tuple(brKey, brHead, std::set<std::string>(), std::set<int>(), backReformulated)
        );
    }

    // 6) Flush mail
    assert(coreId >= 0);
    sendMail(this->body, mailOut, coreId, index, boxes);
}

//#pragma optimize("", off)


// Helper to recursively deactivate a block and ONLY its induction-related descendants.
// Helper to recursively deactivate blocks starting from the root.
// Traverses the entire tree and deactivates any node where:
// 1. All its children are inactive.
// 2. It has no pending proofs (toBeProved is empty).
void ExpressionAnalyzer::deactivateRecursively() {
    // Recursive lambda for post-order traversal
    std::function<void(Memory*)> traverse =
        [&](Memory* node) {
        if (!node) return;

        // 1. Recurse into children first (Post-order)
        for (auto& kv : node->simpleMap) {
            if (kv.second) {
                traverse(kv.second);
            }
        }

        // 2. Check if all children are deactivated
        bool allChildrenDeactivated = true;
        for (const auto& kv : node->simpleMap) {
            Memory* child = kv.second;
            if (child && child->isActive) {
                allChildrenDeactivated = false;
                break;
            }
        }

        // 3. Deactivate node if conditions are met:
        //    - All children are inactive
        //    - No pending proofs in the current node

        int mainValidityCount = 0;
        for (const auto& kv : node->toBeProved) {
            // kv.first is the EncodedExpression key
            if (kv.first.validityName == "main") {
                mainValidityCount++;
            }
        }

        if (allChildrenDeactivated && mainValidityCount == 0 && !node->primedForContradiction) {
            node->isActive = false;

        }
        };

    // Start traversal from the root of the ExpressionAnalyzer
    traverse(&this->body);
}

//#pragma optimize("", off)


void ExpressionAnalyzer::deactivateUnnecessary(const std::string& theorem, Memory& root) {
    // 1) Collect original indices for this theorem
    std::vector<int> originals;
    originals.reserve(8);
    for (const auto& kv : this->globalDependencies.originalAuxyMap) {
        if (kv.second.expr == theorem) 
            originals.push_back(kv.first);
    }

    // 2) Disintegrate theorem to find its Memory path
    std::vector<std::tuple<std::string, std::vector<std::string>, std::set<std::string>>> tempChain;
    ce::disintegrateImplication(theorem, tempChain, this->coreExpressionMap);
    if (tempChain.empty()) return;

    // Walk down the chain from root, collecting the blocks on the path
    Memory* node = &root;
    std::vector<Memory*> chainBlocks;
    chainBlocks.reserve(tempChain.size());

    for (const auto& t : tempChain) {
        const std::string& key = std::get<0>(t);
        auto it = node->simpleMap.find(key);
        if (it == node->simpleMap.end() || it->second == nullptr) {
            // Path not present (may have been pruned earlier) — nothing to do safely.
            return;
        }
        node = it->second;
        chainBlocks.push_back(node);
    }

    // Names depend only on the last node on the path
    const std::string zeroArgName = this->findZeroArgName(*node);
    const std::string sName = this->findSName(*node);
    if (zeroArgName.empty() || sName.empty()) {
        // Can't construct the expected keys — bail gracefully.
        return;
    }

    // 3) Deactivate per-original auxiliaries under the final node
    for (int originalIndex : originals) {
        auto itIV = this->globalDependencies.originalInductionVariableMap.find(originalIndex);
        if (itIV == this->globalDependencies.originalInductionVariableMap.end()) continue;

        const std::string& indVar = std::get<0>(itIV->second); // e.g. "i2"
        const std::string& recCounter = std::get<1>(itIV->second); // numeric string
        const std::string  recArg = std::string("rec") + recCounter;

        const std::string k_rec_in2 = "(in2[" + recArg + "," + indVar + "," + sName + "])";
        const std::string k_eq_side = "(=[s(" + recArg + ")," + zeroArgName + "])";

        if (auto it1 = node->simpleMap.find(k_rec_in2); it1 != node->simpleMap.end() && it1->second) {
            it1->second->isActive = false;

        }
        if (auto it2 = node->simpleMap.find(k_eq_side); it2 != node->simpleMap.end() && it2->second) {
            it2->second->isActive = false;

        }
        // No asserts: partial deactivation is still useful/safe.
    }

    // 4) Bubble up: if a block has no active children, deactivate it too
    for (auto rit = chainBlocks.rbegin(); rit != chainBlocks.rend(); ++rit) {
        Memory* block = *rit;
        bool anyActiveChild = false;
        for (const auto& kv : block->simpleMap) {
            Memory* child = kv.second;
            if (child && child->isActive) { anyActiveChild = true; break; }
        }
        if (!anyActiveChild && block->toBeProved.size() == 0)
        {
            block->isActive = false;
        }
    }

	deactivateRecursively();  // Further prune the tree globally
}



// analyze_expressions.cpp
// Thread-safe: locks globalDependencies briefly, and locks globalTheoremList only when pushing.
void ExpressionAnalyzer::updateGlobal(int auxyIndex, bool allLevelsInvolved, int coreId) {
    // ---- dependency bookkeeping (short critical section) ----
    int originalIndex = -1;
    std::string expr;
    std::string indVar;
    std::string recCounter;
    bool promote = false;

    {
        std::lock_guard<std::mutex> lock(this->dependenciesMutex);

        // Resolve original and update its bookkeeping
        originalIndex = this->globalDependencies.auxyOriginalMap[auxyIndex];

        auto& origItem = this->globalDependencies.originalAuxyMap[originalIndex];
        origItem.auxies.erase(auxyIndex);
        origItem.allLevelsInvolved = allLevelsInvolved || origItem.allLevelsInvolved;

        // If ready to promote, capture the data we need before releasing the lock
        if (origItem.auxies.empty() && origItem.allLevelsInvolved) {
            promote = true;
            expr = origItem.expr;

            auto itIV = this->globalDependencies.originalInductionVariableMap.find(originalIndex);
            if (itIV != this->globalDependencies.originalInductionVariableMap.end()) {
                indVar = std::get<0>(itIV->second);
                recCounter = std::get<1>(itIV->second);
            }
        }
    }

    if (!promote) return;

    // ---- heavy work without holding dependency lock ----
    Mail mailOut;

    deactivateUnnecessary(expr, this->body);  // deactivates all blocks tied to this theorem.

    // Disintegrate expr -> (chain, head)
    std::vector<std::tuple<std::string, std::vector<std::string>, std::set<std::string>>> tempChain;
    std::string value = ce::disintegrateImplication(expr, tempChain, this->coreExpressionMap);

    std::vector<std::string> ky;
    ky.reserve(tempChain.size());
    for (std::size_t i = 0; i < tempChain.size(); ++i) ky.push_back(std::get<0>(tempChain[i]));

    // Walk to the memory block for this key
    Memory* memoryBlock = accessMemory(ky, this->body);
    if (memoryBlock != NULL) {
        std::map<EncodedExpression, std::tuple<std::set<int>, std::set<std::string>>>::iterator it = memoryBlock->toBeProved.find(EncodedExpression(value, "main"));
        if (it != memoryBlock->toBeProved.end()) {
            // Python asserts this is an empty set
            assert(std::get<std::set<int>>(it->second).empty());
            memoryBlock->toBeProved.erase(it);

            // Build origin and levels
            std::pair<std::string, std::vector<ExpressionWithValidity>> origin;
            origin.first = "implication";
            origin.second.push_back(ExpressionWithValidity(expr, "main"));
            for (const std::string& keyElement : ky) {
                // Construct the object in-place with the key and "main"
                origin.second.emplace_back(keyElement, "main");
            }

            std::set<int> levels;
            for (int i = 0; i <= static_cast<int>(ky.size()); ++i) levels.insert(i);

            // Add the head as a statement at this memory block
            addStatement(value, *memoryBlock, false, levels, origin, "main");

            // Queue the implication in global mail_out
            mailOut.implications.insert(std::make_tuple(ky, value, std::set<std::string>(), std::set<int>(), expr));

            if (parameters.trackHistory) {
                ExpressionWithValidity ev(expr, "main");
                addOrigin(mailOut.exprOriginMap, ev, std::make_pair("theorem", std::vector<ExpressionWithValidity>()), (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
            }

            // ---- record in globalTheoremList (short critical section) ----
            {
                std::lock_guard<std::mutex> lock(this->theoremListMutex);
                this->globalTheoremList.emplace_back(expr, "induction", indVar, recCounter);
            }
            std::cout << expr << std::endl;

            std::vector<std::pair<std::string, std::string>> outTheorems;
            if (reformulateTheorem(expr, outTheorems)) {
                for (const auto& pair : outTheorems) {
                    const std::string& refTheoremRaw = pair.first;
                    const std::string& refTheoremCompiled = pair.second;

                    {
                        std::lock_guard<std::mutex> lock(this->theoremListMutex);
                        this->globalTheoremList.emplace_back(refTheoremCompiled, "reformulated statement", expr, "-1");
                    }
                    std::cout << refTheoremCompiled << std::endl;

                    if (parameters.trackHistory) {
                        ExpressionWithValidity ev(refTheoremCompiled, "main");
                        addOrigin(mailOut.exprOriginMap, ev, std::make_pair("theorem", std::vector<ExpressionWithValidity>()), (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
                    }

                    tempChain.clear();
                    std::string valueR = ce::disintegrateImplication(refTheoremCompiled, tempChain, this->coreExpressionMap);

                    ky.clear();
                    ky.reserve(tempChain.size());
                    for (std::size_t i = 0; i < tempChain.size(); ++i) ky.push_back(std::get<0>(tempChain[i]));

                    Memory* memoryBlockR = accessMemory(ky, this->body);
                    if (memoryBlockR != NULL) {
                        std::pair<std::string, std::vector<ExpressionWithValidity>> originR;
                        originR.first = "implication";
                        originR.second.push_back(ExpressionWithValidity(refTheoremCompiled, "main"));
                        for (const std::string& keyElement : ky) {
                            originR.second.emplace_back(keyElement, "main");
                        }

                        std::set<int> levelsR;
                        for (int i = 0; i <= static_cast<int>(ky.size()); ++i) levelsR.insert(i);

                        addStatement(valueR, *memoryBlockR, false, levelsR, originR, "main");
                    }

                    mailOut.implications.insert(
                        std::make_tuple(ky, valueR, std::set<std::string>(), std::set<int>(), refTheoremCompiled)
                    );
                }
            }


            // Try anchor-first reshuffled / mirrored variant
            const std::string reshuffledMirrored =
                ce::createReshuffledMirrored(expr, this->anchorInfo.name, true, this->coreExpressionMap);  // :contentReference[oaicite:1]{index=1}

            if (!reshuffledMirrored.empty()) {
                {
                    std::lock_guard<std::mutex> lock(this->theoremListMutex);
                    this->globalTheoremList.emplace_back(reshuffledMirrored, "mirrored statement", expr, "-1");
                }

                if (parameters.trackHistory) {
                    ExpressionWithValidity ev(reshuffledMirrored, "main");
                    addOrigin(mailOut.exprOriginMap, ev, std::make_pair("theorem", std::vector<ExpressionWithValidity>()), (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
                }

                // Disintegrate mirrored and try to add its head too
                tempChain.clear();
                std::string valueM =
                    ce::disintegrateImplication(reshuffledMirrored, tempChain, this->coreExpressionMap);

                ky.clear();
                ky.reserve(tempChain.size());
                for (std::size_t i = 0; i < tempChain.size(); ++i) ky.push_back(std::get<0>(tempChain[i]));

                // Queue the mirrored implication (no addStatement — let it fire naturally via hash burst)
                mailOut.implications.insert(
                    std::make_tuple(ky, valueM, std::set<std::string>(), std::set<int>(), reshuffledMirrored)
                );
            }

            // Flush mail
            assert(coreId >= 0);
            sendMail(this->body, mailOut, coreId, index, boxes);  // :contentReference[oaicite:2]{index=2}
        }
    }
}


inline void ExpressionAnalyzer::updateAdmissionMap3(const std::string& expr,
    Memory& memoryBlock,
    int maxAdmissionDepth,
    int maxSecondaryNumber,
    bool partOfRecursion) {
    // 1) Compute digit arguments of the expression (uses analyzer's anchor)
    std::set<std::string> digitArgs = findDigitArgs(expr, this->anchorInfo, this->coreExpressionMap);

    // 2) Walk up the Memory chain to find the first block whose exprKey shares a digit arg
    Memory* tempMb = &memoryBlock;
    while (true) {
        // Build set of arguments from this block's exprKey
        const std::vector<std::string> keyArgsVec = ce::getArgs(tempMb->exprKey);
        std::set<std::string> exprKeyArgs;
        for (std::size_t i = 0; i < keyArgsVec.size(); ++i) {
            exprKeyArgs.insert(keyArgsVec[i]);
        }

        // Check intersection with digitArgs
        bool intersects = false;
        for (std::set<std::string>::const_iterator it = digitArgs.begin();
            it != digitArgs.end(); ++it) {
            if (exprKeyArgs.find(*it) != exprKeyArgs.end()) {
                intersects = true;
                break;
            }
        }

        if (intersects) {
            // Remaining args = digitArgs ∪ { last argument of expr }
            const std::vector<std::string> exprArgs = ce::getArgs(expr);
            if (!exprArgs.empty()) {
                std::set<std::string> remainingArgs = digitArgs;
                remainingArgs.insert(exprArgs.back());

                // Key is just [expr]
                std::vector<std::string> key;
                key.push_back(expr);

                this->updateAdmissionMap(*tempMb,
                    key,
                    remainingArgs,
                    maxAdmissionDepth,
                    maxSecondaryNumber,
                    partOfRecursion,
                    "main");

                
            }
            break;
        }
        else {
            if (tempMb->parentMemory != NULL) {
                tempMb = tempMb->parentMemory;
            }
            else {
                break;
            }
        }
    }
}


std::string ExpressionAnalyzer::negate(std::string expr) {
    if (!expr.empty() && expr.front() == '!') return expr.substr(1);
    return "!" + expr;
}

//#pragma optimize("", off)

void ExpressionAnalyzer::handleAnchor(const std::string& expr, Memory& memoryBlock, const std::set<int>& involvedLevels)
{   
    std::string anchorCheck = "(" + this->anchorInfo.name;

    if (expr.rfind(anchorCheck, 0) == 0)
    {
        if (memoryBlock.exprKey.rfind("(Anchor", 0) != 0)
        {
            return;
        }

        std::vector<std::string> args = ce::getArgs(expr);
        std::string coreExpr = ce::extractExpression(expr);

        std::map<std::string, std::string> replacementMap;
        for (const auto& [slot, pattern] : anchorInfo.definitionSets)
        {
            if (pattern == "(1)")
            {
                int index = std::stoi(slot) - 1;

                if (args[index].rfind("x", 0) == 0)
                {
					return;
                }

                replacementMap.insert(std::make_pair(args[index], "x" + args[index]));
            }
        }

        std::string replacedAnchor = ce::replaceKeysInString(expr, replacementMap);
		EncodedExpression encReplacedAnchor(replacedAnchor, "main");
		ExpressionWithValidity evReplacedAnchor(replacedAnchor, "main");

        memoryBlock.localEncodedStatements.push_back(EncodedExpression(replacedAnchor, "main"));
        memoryBlock.localEncodedStatementsDelta.push_back(EncodedExpression(replacedAnchor, "main"));
        memoryBlock.encodedStatements.push_back(EncodedExpression(replacedAnchor, "main"));
        { IntEncodedExpr ie = encodeExpression(memoryBlock.encodedStatements.back(), memoryBlock.nameMap);
          memoryBlock.intEncodedStatements.push_back(ie);
          memoryBlock.intLocalEncodedStatements.push_back(ie);
          memoryBlock.intLocalEncodedStatementsDelta.push_back(ie); }
        //in3[2,2,2,4] finder:
        if (replacedAnchor == "(in3[2,2,2,4])" && memoryBlock.exprKey == "(AnchorIncubator[1,2,3,4,5,6,7,8,9,10])")
        {
            int test = 0;
            test++;
        }
        memoryBlock.statementLevelsMap[encReplacedAnchor] = involvedLevels;
        memoryBlock.intKnownStatements.insert(packStatementKey(
            memoryBlock.nameMap.encode(encReplacedAnchor.original),
            memoryBlock.nameMap.encode(encReplacedAnchor.validityName)));
        memoryBlock.wholeExpressions.insert(EncodedExpression(replacedAnchor, "main"));

        std::pair<std::string, std::vector<ExpressionWithValidity>> handledOrigin;
        handledOrigin.first = "anchor handling";
		handledOrigin.second.push_back(ExpressionWithValidity(expr, "main"));

        addOrigin(memoryBlock.exprOriginMap, evReplacedAnchor, std::make_pair("anchor handling", std::vector<ExpressionWithValidity>{ ExpressionWithValidity(expr, "main") }), (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));

        memoryBlock.mailOut.statements.insert(std::make_pair(replacedAnchor, involvedLevels));

        addOrigin(memoryBlock.mailOut.exprOriginMap, evReplacedAnchor, handledOrigin, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
    }
}

// Turn optimizations OFF for just this section
//#pragma optimize("", off)
bool ExpressionAnalyzer::allowedForMail(const std::string& expression, Memory& body) {
    static const std::regex reInt(R"(int_lev_\d+_\d+)");
    std::smatch match;

    if (!std::regex_search(expression, match, reInt)) {
        return true;
    }

    if (body.canBeSentSet.find(expression) != body.canBeSentSet.end()) {
        return true;
    }

    std::set<std::string> intVars;
    auto begin = std::sregex_iterator(expression.begin(), expression.end(), reInt);
    auto end = std::sregex_iterator();
    for (std::sregex_iterator i = begin; i != end; ++i) {
        intVars.insert(i->str());
    }

    if (intVars.size() != 1) {
        return false;
    }

    if (expression.rfind("(>", 0) == 0) {
        return false;
    }

    // Check if replaced with marker it is included in Memory::canBeSentMarkerSet
    std::string theVar = *intVars.begin();
    std::map<std::string, std::string> replacementMap;
    replacementMap[theVar] = "marker";

    std::string markerExpr = ce::replaceKeysInString(expression, replacementMap);

    if (body.canBeSentMarkerSet.find(markerExpr) != body.canBeSentMarkerSet.end()) {
        std::string core = ce::extractExpression(expression);
        auto itComp = compiledExpressions.find(core);
        assert(itComp != compiledExpressions.end());
        if (itComp->second.category != "atomic") {
            return true;
        }
    }

    return false;
}

std::set<std::string> ExpressionAnalyzer::extractRemainingArgs(const std::string& str) {
    std::set<std::string> remainingArgs;

    // 1. Check >[...] blocks (Strict Forbidden Zone for "u_")
    {
        const static std::regex re1(R"(>\[([^\]]*)\])");
        std::sregex_iterator it(str.begin(), str.end(), re1);
        std::sregex_iterator end;

        for (; it != end; ++it) {
            const std::string inside = (*it)[1].str();
            std::size_t start = 0;
            while (start <= inside.size()) {
                std::size_t pos = inside.find(',', start);
                std::string token = ce::trimCopy(inside.substr(
                    start, (pos == std::string::npos ? inside.size() : pos) - start));

                if (!token.empty()) {
                    // CRITICAL ASSERTION: No "u_" allowed here at all
                    bool hasUPrefix = startsWith(token, "u_", 2);
                    assert(!hasUPrefix && "Violation: Argument starting with 'u_' found inside >[...] block");
                }

                if (pos == std::string::npos) break;
                start = pos + 1;
            }
        }
    }

    // 2. Collect "u_" args from [...] blocks (NOT preceded by '>')
    {
        const static std::regex re2(R"(\[([^\]]*)\])");
        std::sregex_iterator it(str.begin(), str.end(), re2);
        std::sregex_iterator end;

        for (; it != end; ++it) {
            const std::size_t lpos = static_cast<std::size_t>((*it).position());

            // Skip if preceded by '>' (handled above)
            if (lpos > 0 && str[lpos - 1] == '>') {
                continue;
            }

            const std::string inside = (*it)[1].str();
            std::size_t start = 0;
            while (start <= inside.size()) {
                std::size_t pos = inside.find(',', start);
                std::string token = ce::trimCopy(inside.substr(
                    start, (pos == std::string::npos ? inside.size() : pos) - start));

                // Collect only if it starts with "u_"
                if (startsWith(token, "u_", 2)) {

                    // Count occurrences of "u_" to ensure it only appears once (at the start)
                    int uCount = 0;
                    std::size_t p = 0;
                    while ((p = token.find("u_", p)) != std::string::npos) {
                        uCount++;
                        p += 2;
                    }
                    assert(uCount == 1 && "Argument starting with 'u_' must not contain additional 'u_' substrings");

                    // Insert without "u_" prefix
                    remainingArgs.insert(token.substr(2));
                }

                if (pos == std::string::npos) break;
                start = pos + 1;
            }
        }
    }

    return remainingArgs;
}

bool ExpressionAnalyzer::reformulateTheorem(const std::string& theorem, std::vector<std::pair<std::string, std::string>>& outTheorems) {
    if (parameters.ban_disintegration) return false;
    
    bool foundAny = false;

    // 1. Disintegrate the original theorem
    std::vector<std::tuple<std::string, std::vector<std::string>, std::set<std::string>>> originalChain;
    std::string head = ce::disintegrateImplication(theorem, originalChain, this->coreExpressionMap);

    // Disintegration safety check
    if (originalChain.empty()) {
        return false;
    }

    // 2. Identify indices of all Definitions in the chain
    std::vector<size_t> defIndices;
    for (size_t i = 0; i < originalChain.size(); ++i) {
        std::string core = ce::extractExpressionUniversal(std::get<0>(originalChain[i]));
        auto it = compiledExpressions.find(core);
        if (it != compiledExpressions.end() && !it->second.definedSet.empty()) {
            defIndices.push_back(i);
        }
    }

    if (defIndices.empty()) return false;

    // 3. Iterate through each definition to see if it qualifies for reformulation
    for (size_t targetDefIdx : defIndices) {

        // A. Construct Reshuffled Key: [Others..., TargetDef]
        // We preserve the relative order of the non-target premises
        std::vector<std::string> reshuffledKey;
        reshuffledKey.reserve(originalChain.size());

        for (size_t i = 0; i < originalChain.size(); ++i) {
            if (i != targetDefIdx) {
                reshuffledKey.push_back(std::get<0>(originalChain[i]));
            }
        }
        // Add the target definition at the VERY END
        reshuffledKey.push_back(std::get<0>(originalChain[targetDefIdx]));

        // B. Reconstruct the Full Theorem with the ORIGINAL Head
        // This is crucial: we need the system to calculate the binding scope 
        // for the new order of premises.
        std::string tempTheorem = reconstructImplication(reshuffledKey, head);

        // C. Disintegrate again to inspect the binding of the last link (our TargetDef)
        std::vector<std::tuple<std::string, std::vector<std::string>, std::set<std::string>>> tempChain;
        std::string tempHead = ce::disintegrateImplication(tempTheorem, tempChain, this->coreExpressionMap);

        // Safety checks
        if (tempChain.empty() || tempHead != head) continue;

        // The last link in tempChain corresponds to reshuffledKey.back() (our TargetDef)
        const auto& lastLink = tempChain.back();
        const std::string& lastDefExpr = std::get<0>(lastLink);
        const std::vector<std::string>& boundVars = std::get<1>(lastLink);

        // D. Verify Logic: Bound Variable == Set Argument
        std::string core = ce::extractExpressionUniversal(lastDefExpr);
        auto leIt = compiledExpressions.find(core);
        if (leIt == compiledExpressions.end()) continue;

        // Find the expected argument position from the signature
        std::vector<std::string> sigArgs = ce::getArgs(leIt->second.signature);
        auto itSig = std::find(sigArgs.begin(), sigArgs.end(), leIt->second.definedSet);
        if (itSig == sigArgs.end()) continue;

        size_t argIndex = std::distance(sigArgs.begin(), itSig);
        std::vector<std::string> actualArgs = ce::getArgs(lastDefExpr);
        if (argIndex >= actualArgs.size()) continue;

        std::string expectedArg = actualArgs[argIndex];

        // CHECK: Is this definition acting as the peeling layer?
        // The last implication must bind exactly one variable, and it must be the set argument.
        if (boundVars.size() == 1 && boundVars[0] == expectedArg) {

            // FOUND VALID REFORMULATION!

            // E. Apply Transformation
            // Structure: (Premises... -> !( > [vars] LastDef !Head ))

            // Prepare Negated Head
            std::string negatedHead;
            if (head.size() > 1 && head[0] == '!') {
                negatedHead = head.substr(1);
            }
            else {
                negatedHead = "!" + head;
            }

            // Build the Special Innermost Tail
            std::string varsStr = boundVars[0];

            // 1. Raw Version (Original Code)
            std::string rawInnerTail = "!(>[" + varsStr + "]" + lastDefExpr + negatedHead + ")";

            // 2. Compiled Version (New Logic)
            std::string compiledInnerTail = this->compileCoreExpressionMapCore(rawInnerTail, this->implCounter, this->existenceCounter, this->statementCounter, this->variableCounter);

            // Wrap the remaining premises (Standard Implication)
            std::string currentTailRaw = rawInnerTail;
            std::string currentTailCompiled = compiledInnerTail;

            for (int i = static_cast<int>(tempChain.size()) - 2; i >= 0; --i) {
                const auto& link = tempChain[i];
                const std::string& premise = std::get<0>(link);
                const std::vector<std::string>& vars = std::get<1>(link);

                std::string vStr;
                for (size_t v = 0; v < vars.size(); ++v) {
                    if (v > 0) vStr += ",";
                    vStr += vars[v];
                }

                std::string wrapper = "(>[" + vStr + "]" + premise;
                currentTailRaw = wrapper + currentTailRaw + ")";
                currentTailCompiled = wrapper + currentTailCompiled + ")";
            }

            outTheorems.push_back(std::make_pair(currentTailRaw, currentTailCompiled));
            foundAny = true;
        }
    }

    return foundAny;
}
// Turn optimizations OFF for just this section
//#pragma optimize("", off)

void ExpressionAnalyzer::addEquality(const std::string& expr,
    Memory& memoryBlock,
    bool local,
    const std::set<int>& levels,
    const std::pair<std::string, std::vector<ExpressionWithValidity>>& origin,
    const std::string& validityName)
{


    // Reconstruct these objects locally
    EncodedExpression encodedExpr(expr, validityName);
    ExpressionWithValidity exprWithValidity(expr, validityName);

    if (memoryBlock.wholeExpressions.find(encodedExpr) == memoryBlock.wholeExpressions.end())
    {
        memoryBlock.wholeExpressions.insert(EncodedExpression(expr, validityName));


        // 1. Register the original equality
        memoryBlock.statementLevelsMap[encodedExpr] = levels;
        memoryBlock.intKnownStatements.insert(packStatementKey(
            memoryBlock.nameMap.encode(encodedExpr.original),
            memoryBlock.nameMap.encode(encodedExpr.validityName)));

        if (parameters.trackHistory) {
            addOrigin(memoryBlock.exprOriginMap, exprWithValidity, origin, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
            addOrigin(memoryBlock.mailOut.exprOriginMap, exprWithValidity, origin, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
        }

        memoryBlock.encodedStatements.push_back(encodedExpr);
        { IntEncodedExpr ie = encodeExpression(encodedExpr, memoryBlock.nameMap);
          memoryBlock.intEncodedStatements.push_back(ie);

        if (local) {
            memoryBlock.localEncodedStatements.push_back(encodedExpr);
            memoryBlock.localEncodedStatementsDelta.push_back(encodedExpr);
            memoryBlock.intLocalEncodedStatements.push_back(ie);
            memoryBlock.intLocalEncodedStatementsDelta.push_back(ie);

            if (validityName == "main") {
                memoryBlock.mailOut.statements.insert(std::make_pair(expr, levels));
            }
            if (parameters.trackHistory) {
                addOrigin(memoryBlock.mailOut.exprOriginMap, exprWithValidity, origin, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
            }
        }
        } // close IntEncodedExpr ie scope

        // 2. Register the MIRRORED equality
        std::vector<std::string> args = ce::getArgs(expr);
        assert(args.size() == 2);

        if (args[0] != args[1]) {
        std::string mirrored = "(=[" + args[1] + "," + args[0] + "])";

        EncodedExpression encodedMirrored(mirrored, validityName);
        ExpressionWithValidity mirroredWithValidity(mirrored, validityName);

        assert(memoryBlock.wholeExpressions.find(encodedMirrored) == memoryBlock.wholeExpressions.end());
        memoryBlock.wholeExpressions.insert(EncodedExpression(mirrored, validityName));

        memoryBlock.statementLevelsMap[encodedMirrored] = levels;
        memoryBlock.intKnownStatements.insert(packStatementKey(
            memoryBlock.nameMap.encode(encodedMirrored.original),
            memoryBlock.nameMap.encode(encodedMirrored.validityName)));

        if (parameters.trackHistory) {
            std::pair<std::string, std::vector<ExpressionWithValidity>> mirroredOrigin;
            mirroredOrigin.first = "symmetry of equality";
            mirroredOrigin.second.push_back(exprWithValidity); // Derived from original

            addOrigin(memoryBlock.exprOriginMap, mirroredWithValidity, mirroredOrigin, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
            addOrigin(memoryBlock.mailOut.exprOriginMap, mirroredWithValidity, mirroredOrigin, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
        }

        memoryBlock.encodedStatements.push_back(encodedMirrored);
        { IntEncodedExpr ieM = encodeExpression(encodedMirrored, memoryBlock.nameMap);
          memoryBlock.intEncodedStatements.push_back(ieM);

        if (local) {
            memoryBlock.localEncodedStatements.push_back(encodedMirrored);
            memoryBlock.localEncodedStatementsDelta.push_back(encodedMirrored);
            memoryBlock.intLocalEncodedStatements.push_back(ieM);
            memoryBlock.intLocalEncodedStatementsDelta.push_back(ieM);

            if (validityName == "main") {
                memoryBlock.mailOut.statements.insert(std::make_pair(mirrored, levels));
            }
            // Origin for mirrored already handled in trackHistory block above
        }
        } // close IntEncodedExpr ieM scope
        } // end if args[0] != args[1]
    }
}

//#pragma optimize("", off)

void ExpressionAnalyzer::checkNecessityForEquality(const std::string& inputExprStr, Memory& mb, std::string validityName) {

    if (parameters.ban_disintegration) return;
    
    // Use special function to make all args "u_"
    std::string genericInputStr = prefixArgumentsWithU(inputExprStr);

    // Parse the input expression string
    std::string inputName = ce::extractExpression(genericInputStr);
    std::vector<std::string> inputArgs = ce::getArgs(genericInputStr);

    // Iterate through all original implications stored in localMemory
    for (const std::vector<std::string>& chain : mb.overallHashMemory.originals) {
        if (chain.empty()) continue;

        // "Head" is the last element of the implication chain
        std::string headStr = chain.back();
        std::string headName = ce::extractExpression(headStr);
        std::vector<std::string> headArgs = ce::getArgs(headStr);

        // Structural check: Name and Arity must match the input expression
        if (headName != inputName) continue;
        if (headArgs.size() != inputArgs.size()) continue;

        // Check if head is "Generic except for one arg X"
        // Condition: Exactly one argument in head does NOT start with "u_" (the constant).
        int constantIndex = -1;
        int constantCount = 0;

        for (size_t i = 0; i < headArgs.size(); ++i) {
            // Check if argument starts with "u_"
            bool isVar = (headArgs[i].size() >= 2 && headArgs[i][0] == 'u' && headArgs[i][1] == '_');

            if (!isVar) {
                constantCount++;
                constantIndex = static_cast<int>(i);
            }
        }

        // Requirement: Exactly one arg is not u_
        if (constantCount != 1) continue;

        // NEW: Check that definition set of this arg is (1)
        auto itCfg = this->coreExpressionMap.find(headName);
        if (itCfg == this->coreExpressionMap.end()) continue;
        const auto& cfg = itCfg->second;

        // definitionSets keys are 1-based strings ("1", "2", etc.)
        std::string argPos = std::to_string(constantIndex + 1);
        auto itDS = cfg.definitionSets.find(argPos);

        // FIX: Access .first because definitionSets values are pair<string, bool>
        if (itDS == cfg.definitionSets.end() || itDS->second.first != "(1)") {
            continue;
        }

        // Requirement: Check that head is equal input with exception of one arg
        // We compare all arguments except the constantIndex.
        bool match = true;
        for (size_t i = 0; i < headArgs.size(); ++i) {
            if (static_cast<int>(i) == constantIndex) continue;

            if (headArgs[i] != inputArgs[i]) {
                match = false;
                break;
            }
        }
        if (!match) continue;

        // "Replacement map is not identity. in one arg they r not equal. these two args are use for replacement"
        std::map<std::string, std::string> replacementMap;
        replacementMap[headArgs[constantIndex]] = inputArgs[constantIndex];

        bool conditionMet = false;
        std::string targetVar = inputArgs[constantIndex]; // The variable we are tracking

        // Iterate over premises (all elements except the last one)
        for (size_t k = 0; k < chain.size() - 1; ++k) {
            std::string premise = chain[k];

            // Apply replacement (Replace the constant from Head with the variable from Input)
            std::string instantiatedPremise = ce::replaceKeysInString(premise, replacementMap);

            std::string removed = removeUPrefixFromArguments(instantiatedPremise);
            std::vector<std::string> argsVec = ce::getArgs(removed);
            std::set<std::string> argsSet(argsVec.begin(), argsVec.end());

            prepareIntegration(removed, argsSet, mb, validityName);

            // NEW: Explicitly check inputArgs[constantIndex] occurs 2 times
            // and did NOT occur 2 times before replacement

            // 1. Count in instantiated premise
            std::vector<std::string> pArgs = ce::getArgs(instantiatedPremise);
            int countAfter = 0;
            for (const auto& arg : pArgs) {
                if (arg == targetVar) countAfter++;
            }

            // 2. Count in original premise
            std::vector<std::string> origArgs = ce::getArgs(premise);
            int countBefore = 0;
            for (const auto& arg : origArgs) {
                if (arg == targetVar) countBefore++;
            }

            // Condition: Occurs explicitly 2 times now, but count was different before
            if (countAfter >= 2 && countBefore == 1) {
                conditionMet = true;
                break;
            }
        }

        if (conditionMet) {
            // NEW CHECK: Check if the duplicated var (targetVar) already has another var =[var,var2]
            // We check the equivalence classes for the current validity scope.
            bool hasExistingEquality = false;
            auto itMap = mb.equivalenceClassesMap.find(validityName);
            if (itMap != mb.equivalenceClassesMap.end()) {
                for (const auto& ec : itMap->second) {
                    if (ec.variables.count(targetVar)) {
                        // User Request: assert that an eq class which contains the var has more than 1 element
                        assert(ec.variables.size() > 1);

                        hasExistingEquality = true;
                        break;
                    }
                }
            }

            // If yes, do not add a new copy
            if (hasExistingEquality) {
                continue;
            }

            // "call addExprToMemoryBlock() for '(=[Y,Y_copy])'"

            // Assert that Y is available
            assert(inputArgs.size() > static_cast<size_t>(constantIndex));

            std::string Y = inputArgs[constantIndex]; // Extract the input arg (Y)
            std::string Y_copy = Y + "_copy";
            std::string equalityExpr = "(=[" + Y + "," + Y_copy + "])";

            // Try to inherit levels
            std::set<int> levels;

            // Create origin info
            std::pair<std::string, std::vector<ExpressionWithValidity>> origin;
            origin.first = "necessity for equality";
            origin.second.push_back(ExpressionWithValidity(inputExprStr, "main"));

            std::string finalExpr = removeUPrefixFromArguments(equalityExpr);

            levels.insert(mb.level);

            // Add the new equality
            addExprToMemoryBlock(finalExpr,
                mb,
                -1, // iteration
                0,  // status (Local statement)
                levels,
                origin,
                -1, // coreId
                -1, // auxyIndex
                "main",
                false);
        }
    }
}

// Turn optimizations OFF for just this section
//#pragma optimize("", off)

void ExpressionAnalyzer::addExprToMemoryBlockKernel(const std::string& expr,
    Memory& memoryBlock,
    int status,
    const std::set<int>& involvedLevels,
    const std::pair<std::string, std::vector<ExpressionWithValidity>>& origin,
    const std::string& validityName,
    int coreId,
    int iteration)
{
    bool isLocal = (status == 0 || status == 1);
    ExpressionWithValidity exprVal(expr, validityName);
    if (status != 2) {
        if (this->isEquality(expr)) {
			addEquality(expr, memoryBlock, isLocal, involvedLevels, origin, validityName);
        }

        memoryBlock.wholeExpressions.insert(EncodedExpression(expr, validityName));
    }

    

    // Directly use 'expr' here
    std::vector<std::string> added = this->addStatement(expr,
        memoryBlock,
        isLocal,
        involvedLevels,
        origin,
        validityName);

    std::vector<std::string> newStatements = added;

    std::vector<std::string> sortedNew = newStatements;
    std::sort(sortedNew.begin(), sortedNew.end());
    for (std::size_t idx = 0; idx < sortedNew.size(); ++idx) 
    {
        updateAdmissionMapIntegration(sortedNew[idx], memoryBlock, validityName);
        updateAdmissionMapRecursion(sortedNew[idx], memoryBlock, validityName);

        const std::string& addExpression = sortedNew[idx];
		EncodedExpression encAddExpression(addExpression, validityName);

        std::map<EncodedExpression, std::set<int> >::const_iterator sit =
            memoryBlock.statementLevelsMap.find(encAddExpression);
        assert(sit != memoryBlock.statementLevelsMap.end());
        const std::set<int>& addExpressionLevels = sit->second;

        bool allLevelsInvolved = (addExpressionLevels.size() == static_cast<std::size_t>(memoryBlock.level + 1));
        if ((addExpressionLevels.size() == static_cast<std::size_t>(memoryBlock.level)) && addExpressionLevels.find(0) == addExpressionLevels.end())
        {
			allLevelsInvolved = true;
        }

        if (memoryBlock.isPartOfRecursion)
        {
            std::map<EncodedExpression, std::tuple<std::set<int>, std::set<std::string>> >::iterator itProof =
                memoryBlock.toBeProved.find(EncodedExpression(addExpression, validityName));
            if (itProof != memoryBlock.toBeProved.end() && validityName == "main") {

                if (!parameters.compressor_mode) { // <--- COMPRESSOR BYPASS
                    std::set<int> auxiesCopy = std::get<std::set<int>>(itProof->second);
                    {
                        std::lock_guard<std::mutex> lock(this->updateGlobalMutex);
                        for (std::set<int>::const_iterator ait = auxiesCopy.begin(); ait != auxiesCopy.end(); ++ait) {
                            this->updateGlobalTuples.push_back(std::make_tuple(*ait, allLevelsInvolved, coreId));
                        }
                    }
                    memoryBlock.toBeProved.erase(itProof);
                    memoryBlock.isActive = false;


                    if (startsWithStr(memoryBlock.exprKey, "(in2[rec")) {
                        {
                            std::lock_guard<std::mutex> lock(this->inductionMemoryBlocksMutex);
                            this->inductionMemoryBlocks.push_back(&memoryBlock);
                        }
                    }
                }
            }
        }
        else {
            if (this->isProved(addExpression) && allLevelsInvolved && status != 0 && validityName == "main") {
                std::string fullTheorem;
                if (memoryBlock.primedForContradiction && !memoryBlock.contradictionTheorem.empty()) {
                    fullTheorem = memoryBlock.contradictionTheorem;
                } else {
                    std::vector<std::string> globalKey = this->getGlobalKey(memoryBlock);
                    fullTheorem = this->reconstructImplication(globalKey, addExpression);
                }

                std::map<EncodedExpression, std::tuple<std::set<int>, std::set<std::string>> >::iterator itTBP2 =
                    memoryBlock.toBeProved.find(EncodedExpression(addExpression, validityName));
                if (itTBP2 != memoryBlock.toBeProved.end()) {

                    if (!parameters.compressor_mode) { // <--- COMPRESSOR BYPASS
                        {
                            std::lock_guard<std::mutex> lock(this->updateGlobalDirectMutex);
                            this->updateGlobalDirectTuples.push_back(std::make_tuple(fullTheorem, coreId));
                        }
                        memoryBlock.toBeProved.erase(itTBP2);
                        int changeInStartInt;
                        this->cleanUpIntegrationPreparation(addExpression, memoryBlock);
                    }
                }
            }
        }

        std::map<EncodedExpression, std::tuple<std::set<int>, std::set<std::string>> >::iterator itProof =
            memoryBlock.toBeProved.find(EncodedExpression(addExpression, validityName));
        if (validityName != "main" && status == 1 && (itProof != memoryBlock.toBeProved.end()))
        {
            std::pair<std::string, std::vector<ExpressionWithValidity>> localOrigin;
            localOrigin.first = "validity name";
            localOrigin.second.push_back(ExpressionWithValidity(addExpression, validityName));
            addExprToMemoryBlock(validityName,
                memoryBlock,
                iteration,
                1,
                involvedLevels,
                localOrigin,
                coreId,
                -1,
                "main",
                false);

            int changeInStartInt;

            this->cleanUpIntegrationPreparation(validityName, memoryBlock);
            memoryBlock.validityNamesToFilter.insert(validityName);
            memoryBlock.intValidityNamesToFilter.insert(memoryBlock.nameMap.encode(validityName));


        }
    }
}



bool ExpressionAnalyzer::checkForEquivalence(const std::string& expr,
    const std::string& validityName,
    Memory& memoryBlock)
{
    // 1. Parse arguments
    std::vector<std::string> args = ce::getArgs(expr);

    // 2. Identify Equivalence Classes for each argument
    // possibilities[i] will contain all valid variable substitutions for args[i]
    std::vector<std::vector<std::string>> possibilities;
    possibilities.reserve(args.size());

    // Access the equivalence class list for this validity scope
    auto itClasses = memoryBlock.equivalenceClassesMap.find(validityName);
    assert(itClasses != memoryBlock.equivalenceClassesMap.end());

    const std::vector<EquivalenceClass>& eqClasses = itClasses->second;

    for (const std::string& arg : args) {
        std::vector<std::string> varsForArg;
        bool foundClass = false;

        // Check if this arg belongs to any known equivalence class
        for (const auto& eqClass : eqClasses) {
            if (eqClass.variables.find(arg) != eqClass.variables.end()) {
                // Found: add all variables in this class as possibilities
                varsForArg.assign(eqClass.variables.begin(), eqClass.variables.end());
                foundClass = true;
                break; // Argument is assumed to belong to only one class per validity
            }
        }

        // If not found, it is its own only possibility
        if (!foundClass) {
            varsForArg.push_back(arg);
        }
        possibilities.push_back(std::move(varsForArg));
    }

    // 4. Generate All Variants (Cartesian Product) and Check
    // Using an iterative index counter to handle variable number of arguments
    std::vector<size_t> indices(possibilities.size(), 0);

    while (true) {
        // A. Construct replacement map for current indices
        std::map<std::string, std::string> replacements;
        for (size_t i = 0; i < args.size(); ++i) {
            // Map the original argument to the currently selected substitution
            replacements[args[i]] = possibilities[i][indices[i]];
        }

        // B. Generate variant using string replacement
        std::string variant = ce::replaceKeysInString(expr, replacements);

        // C. Check if this variant exists in wholeExpressions
        if (memoryBlock.wholeExpressions.find(EncodedExpression(variant, validityName)) !=
            memoryBlock.wholeExpressions.end()) {
            return true;
        }

        // D. Advance indices (odometer style)
        int i = static_cast<int>(possibilities.size()) - 1;
        while (i >= 0) {
            indices[i]++;
            if (indices[i] < possibilities[i].size()) {
                break; // Successfully incremented this position
            }
            else {
                indices[i] = 0; // Reset this position and carry over
                i--;
            }
        }
        // If we wrapped around the first index, we are done
        if (i < 0) break;
    }

    return false;
}
// Turn optimizations OFF for just this section
//#pragma optimize("", off)

void ExpressionAnalyzer::updateWeakVariables(Memory& mb, const std::string& validityName, const std::string& eqlty) {
    // 1. Identify variables involved in the equality
    std::vector<std::string> eqVars = ce::getArgs(eqlty);
    if (eqVars.empty()) return;

    // 2. Locate the specific Equivalence Class containing these variables
    // We only look within the provided validityName scope
    auto itClasses = mb.equivalenceClassesMap.find(validityName);
    assert(itClasses != mb.equivalenceClassesMap.end());

    const std::vector<EquivalenceClass>& classes = itClasses->second;
    const EquivalenceClass* targetClass = nullptr;

    for (const auto& ec : classes) {
        // If any variable from the equality is in this class, this is the target.
        for (const std::string& v : eqVars) {
            if (ec.variables.find(v) != ec.variables.end()) {
                targetClass = &ec;
                break;
            }
        }
        if (targetClass) break;
    }

    if (!targetClass) assert(false);

    // 3. Logic to identify Weak Variables (Renamed Candidates)
    // Matches logic: int_ > it_, lexicographical sort.
    static const std::regex PATTERN_INT(R"(int_lev_\d+_\d+)");
    static const std::regex PATTERN_IT(R"(it_\d+_lev_\d+_\d+)");

    std::vector<std::string> intVars;
    std::vector<std::string> itVars;

    for (const std::string& var : targetClass->variables) {
        if (std::regex_match(var, PATTERN_INT)) {
            intVars.push_back(var);
        }
        else if (std::regex_match(var, PATTERN_IT)) {
            itVars.push_back(var);
        }
    }

    // Determine Canonical Variable
    std::string canonicalVar;
    if (!intVars.empty()) {
        std::sort(intVars.begin(), intVars.end());
        canonicalVar = intVars.front();
    }
    else if (!itVars.empty()) {
        std::sort(itVars.begin(), itVars.end());
        canonicalVar = itVars.front();
    }
    else {
        return; // No variables match the patterns, so no "weak" variables to record.
    }

    // 4. Save Weak Variables to mb.weakVariables
    // Weak = Matches pattern AND is not canonical
    auto checkAndAdd = [&](const std::vector<std::string>& vars) {
        for (const auto& var : vars) {
            if (var != canonicalVar) {
                mb.weakVariables.insert(ExpressionWithValidity(var, validityName));
            }
        }
        };

    checkAndAdd(intVars);
    checkAndAdd(itVars);
}

// Turn optimizations OFF for just this section
//#pragma optimize("", off)
void ExpressionAnalyzer::addExprToMemoryBlock(const std::string& expr,
    Memory& memoryBlock,
    int iteration,
    int status,
    const std::set<int>& involvedLevels,
    const std::pair<std::string, std::vector<ExpressionWithValidity>>& origin,
    int coreId,
    int auxyIndex,
    std::string validityName,
    bool doNotDisintegrate) {

    // --- Static (int-based) early checks ---
    // Fast path: duplicate + validity filter with just nm.encode (no regex, no vector alloc)
    int16_t origId = memoryBlock.nameMap.encode(expr);
    int16_t valId = memoryBlock.nameMap.encode(validityName);

    if (!parameters.compressor_mode) {
        if (memoryBlock.intKnownStatements.count(packStatementKey(origId, valId))) return;
        int16_t mainId = memoryBlock.nameMap.encode("main");
        if (memoryBlock.intKnownStatements.count(packStatementKey(origId, mainId))) return;
    }

    if (memoryBlock.intValidityNamesToFilter.count(valId)) return;

    // Passed fast checks — now do full encoding for axed var check + downstream
    EncodedExpression encExpr(expr, validityName);
    IntEncodedExpr ie = encodeExpression(encExpr, memoryBlock.nameMap);

    for (int16_t a = 0; a < ie.arity; ++a) {
        if (memoryBlock.intAxedVariables.count(ie.argFullId[a])) return;
    }

    ExpressionWithValidity exprVal(expr, validityName);


    if (status == 4)
    {
        memoryBlock.localEncodedStatements.push_back(encExpr);
		memoryBlock.localEncodedStatementsDelta.push_back(encExpr);
        memoryBlock.encodedStatements.push_back(encExpr);
        memoryBlock.intEncodedStatements.push_back(ie);
        memoryBlock.intLocalEncodedStatements.push_back(ie);
        memoryBlock.intLocalEncodedStatementsDelta.push_back(ie);
		memoryBlock.statementLevelsMap[encExpr] = involvedLevels;
        memoryBlock.intKnownStatements.insert(packStatementKey(ie.originalId, ie.validityId));
		memoryBlock.wholeExpressions.insert(encExpr);

        return;
    }

    if (memoryBlock.equivalenceClassesMap.find(validityName) == memoryBlock.equivalenceClassesMap.end())
    {
        memoryBlock.equivalenceClassesMap[validityName] = std::vector<EquivalenceClass>();
        assert(memoryBlock.eqClassSttmntIndexMapMap.find(validityName) == memoryBlock.eqClassSttmntIndexMapMap.end());
        memoryBlock.eqClassSttmntIndexMapMap[validityName] = std::map<std::set<std::string>, int>();
    }


    std::string negation = this->negate(expr);
    int16_t negOrigId = memoryBlock.nameMap.encode(negation);
    bool contradictionFound = memoryBlock.intKnownStatements.count(
        packStatementKey(negOrigId, ie.validityId)) > 0;
    if (contradictionFound)
    {
        // Incubator contradiction: record provenance, emit theorem, deactivate
        if (memoryBlock.primedForContradiction) {
            // Record origin for the triggering expression before early return.
            // Without this, buildStack cannot trace it as a contradiction dependency.
            if (parameters.trackHistory) {
                int maxOrig0 = parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr;
                addOrigin(memoryBlock.exprOriginMap, exprVal, origin, maxOrig0);
            }

            std::string theorem = memoryBlock.contradictionTheorem;
            memoryBlock.contradictionTheorem.clear();
            memoryBlock.isActive = false;
            if (!theorem.empty()) {
                // Record contradiction provenance BEFORE broadcast
                if (parameters.trackHistory) {
                    // Extract cleanOp from exprKey: "__contradiction__(op[...])"
                    std::string cleanOp;
                    const std::string prefix = "__contradiction__";
                    if (memoryBlock.exprKey.substr(0, prefix.size()) == prefix) {
                        cleanOp = memoryBlock.exprKey.substr(prefix.size());
                    }
                    std::string negCleanOp = "!" + cleanOp;

                    // The negated expression is proved by contradiction:
                    // expr and negate(expr) both exist in the LB
                    ExpressionWithValidity evResult(negCleanOp, "main");
                    std::vector<ExpressionWithValidity> deps;
                    deps.push_back(ExpressionWithValidity(expr, validityName));
                    deps.push_back(ExpressionWithValidity(negation, validityName));
                    deps.push_back(ExpressionWithValidity(cleanOp, "main"));
                    int maxOrig = parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr;
                    // Add to predecessor (where directStack navigates)
                    Memory* pred = memoryBlock.parentMemory;
                    while (pred != nullptr) {
                        addOrigin(pred->exprOriginMap, evResult,
                            std::make_pair("contradiction", deps), maxOrig);
                        pred = pred->parentMemory;
                    }
                    addOrigin(memoryBlock.exprOriginMap, evResult,
                        std::make_pair("contradiction", deps), maxOrig);
                    addOrigin(memoryBlock.mailOut.exprOriginMap, evResult,
                        std::make_pair("contradiction", deps), maxOrig);
                }
                updateGlobalDirect(theorem, coreId);
            }
            return;
        }
        // CE filter contradiction
        if (memoryBlock.contradictionIndex >= 0)
        {
			this->contradictionTable[memoryBlock.contradictionIndex].successful = true;
            memoryBlock.isActive = false;

            return;
        }
        else
        {
            std::cout << expr << " " << memoryBlock.exprKey << std::endl;
        }
    }

    if (memoryBlock.contradictionIndex >= 0)
    {
        return;
    }



    if (status == 2) {
        std::map<EncodedExpression, std::tuple<std::set<int>, std::set<std::string>> >::iterator itTBP = memoryBlock.toBeProved.find(encExpr);
        if (itTBP != memoryBlock.toBeProved.end()) {
            if (auxyIndex >= 0) std::get<std::set<int>>(itTBP->second).insert(auxyIndex);
        }
        else {
            if (auxyIndex >= 0) {
                std::tuple<std::set<int>, std::set<std::string>> s;
                std::get<std::set<int>>(s).insert(auxyIndex);

                memoryBlock.toBeProved.insert(std::make_pair(encExpr, s));
            }
            else 
            {
                memoryBlock.toBeProved.insert(std::make_pair(encExpr, std::tuple<std::set<int>, std::set<std::string>>()));
            }

			checkNecessityForEquality(expr, memoryBlock, validityName);
        }

        std::vector<std::string> args = ce::getArgs(expr);
		std::set<std::string> remainingArgs(args.begin(), args.end());


        prepareIntegration(expr, remainingArgs, memoryBlock, validityName);
        memoryBlock.canBeSentSet.insert(expr);

        const std::string coreExpr = ce::extractExpression(expr);
        auto it = this->coreExpressionMap.find(coreExpr);

        if (it != this->coreExpressionMap.end() && !it->second.inputIndices.empty()) {
            this->updateAdmissionMap3(expr,
                memoryBlock,
                parameters.inductionMaxAdmissionDepth,
                parameters.inductionMaxSecondaryNumber,
                true);
        }
        return;
    }
    else {
        if (parameters.trackHistory) {
            addOrigin(memoryBlock.exprOriginMap, exprVal, origin, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
            addOrigin(memoryBlock.mailOut.exprOriginMap, exprVal, origin, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
        }

        if (status == 3)
        {
            bool isSimple = ce::expressionIsSimple(expr);
            std::vector<std::string> inputStream;

            if (isSimple) {
                addExprToMemoryBlockKernel(expr, memoryBlock, status, involvedLevels, origin, validityName, coreId, iteration);
            }
        }

        if (status != 3) {
            std::set<std::string> imps;
            std::set<std::string> stmts;
            int newStart = memoryBlock.startInt;




            
            
            if (!doNotDisintegrate && !checkForEquivalence(expr, validityName, memoryBlock))
            {
                std::tuple< std::set<std::string>, std::set<std::string>, int> tpl =
                    this->disintegrateExpr2(expr,
                        memoryBlock,
                        iteration,
                        status == 0,
                        validityName);
                imps = std::get<0>(tpl);
                stmts = std::get<1>(tpl);
                memoryBlock.startInt = std::get<2>(tpl);
            }
            else
            {
				stmts.insert(expr);
            }

            for (std::set<std::string>::const_iterator itImp = imps.begin(); itImp != imps.end(); ++itImp) 
            {
                std::pair<std::string, std::string> kv = ce::extractKeyValue(*itImp, coreExpressionMap);
                const std::string& ky = kv.first;
                const std::string& vlue = kv.second;

                std::set<std::string> remainingArgsKey = extractRemainingArgs(ky);

                std::vector< std::tuple<
                    std::string,                    // leftExpr
                    std::vector<std::string>,       // args of the current implication node
                    std::set<std::string>           // node->left.arguments  (SET, like Python)
                > >tempChain;
                std::string head = ce::disintegrateImplication(*itImp, tempChain, coreExpressionMap);
                std::vector<std::string> chain;
                for (std::size_t i = 0; i < tempChain.size(); ++i) chain.push_back(std::get<0>(tempChain[i]));


                this->addToHashMemory(chain, head, remainingArgsKey,
                    memoryBlock, memoryBlock.overallHashMemory,
                    involvedLevels, *itImp,
                    parameters.standardMaxAdmissionDepth, parameters.standardMaxSecondaryNumber, false,
                    parameters.minNumOperatorsKey, "implication", true, *itImp, "main");
                this->addToHashMemory(chain, head, remainingArgsKey,
                    memoryBlock, memoryBlock.localHashMemory,
                    involvedLevels, *itImp,
                    parameters.standardMaxAdmissionDepth, parameters.standardMaxSecondaryNumber, false,
                    parameters.minNumOperatorsKey, "implication", true, *itImp, "main");
                this->addToHashMemory(chain, head, remainingArgsKey,
                    memoryBlock, memoryBlock.localHashMemoryDelta,
                    involvedLevels, *itImp,
                    parameters.standardMaxAdmissionDepth, parameters.standardMaxSecondaryNumber, false,
                    parameters.minNumOperatorsKey, "implication", true, *itImp, "main");

                if (allowedForMail(*itImp, memoryBlock))
                {
                    std::set< std::tuple< std::vector<std::string>, std::string, std::set<std::string>, std::set<int>, std::string > >& impSet =
                        memoryBlock.mailOut.implications;

                    impSet.insert(std::make_tuple(chain,
                        head,
                        remainingArgsKey,
                        involvedLevels,
                        *itImp));
                }
            }

            if (imps.size() > 0)
            {
                // Check necessity for equality for all pending proofs
                for (auto const& [key, val] : memoryBlock.toBeProved) {
                    this->checkNecessityForEquality(key.original, memoryBlock, key.validityName);
                }
            }


            for (std::set<std::string>::const_iterator itSt = stmts.begin(); itSt != stmts.end(); ++itSt)
            {
                ExpressionWithValidity ev(*itSt, validityName);
                auto itOrigin = memoryBlock.exprOriginMap.find(ev);
                assert(itOrigin != memoryBlock.exprOriginMap.end() && !itOrigin->second.empty());

                addExprToMemoryBlockKernel(*itSt, memoryBlock, status, involvedLevels, itOrigin->second.front(), validityName, coreId, iteration);
            }
        }
    }
}

std::vector<std::string> ExpressionAnalyzer::extractSubstringsForAuxy(const std::string& expr2) {
    // Matches: "(" + (first char not >, (,[ or () ) + "[...]" + ")"
    const std::regex re(R"(\(([^>(\[]+\[[^\]]*\])\))");
    std::sregex_iterator it(expr2.begin(), expr2.end(), re);
    std::sregex_iterator end;
    std::vector<std::string> out;
    for (; it != end; ++it) {
        out.push_back(it->str(0)); // whole match with surrounding "()"
    }
    return out;
}

std::tuple<
    std::string,
    std::vector<std::string>,
    std::string,
    std::set<std::string>,
    std::string,
    std::string
>
ExpressionAnalyzer::createAuxyImplication(const std::string& expr,
    const std::string& arg,
    const std::string& recArg,
    const std::set<std::string>& digitArgs,
    const std::string& anchorName) {
    std::set<std::string> untouchables = digitArgs;
    untouchables.insert(recArg);

    std::vector<std::string> chain;
    std::string zeroArgName;
    std::string sName;

    // replace arg -> recArg in the whole expression
    std::map<std::string, std::string> replMap;
    replMap[arg] = recArg;
    const std::string replExpr = ce::replaceKeysInString(expr, replMap);

    // collect non-top-level subexpressions "(X[...])" with X != '>'
    const std::vector<std::string> subexprs = extractSubstringsForAuxy(replExpr);
    for (std::size_t i = 0; i < subexprs.size(); ++i) {
        const std::string& sub = subexprs[i];
        if (sub.find(anchorName) != std::string::npos) {
            const std::vector<std::string> anchorArgs = ce::getArgs(sub);
            // add all anchor args to untouchables
            for (std::size_t k = 0; k < anchorArgs.size(); ++k) {
                untouchables.insert(anchorArgs[k]);
            }

            zeroArgName = findZeroArgNameFromAnchor(sub);
            sName = findSArgNameFromAnchor(sub);
        }
        else {
            chain.push_back(sub);
        }
    }

    // remaining_args_key: args from all but the last element of chain that intersect untouchables
    std::set<std::string> remainingArgsKey;
    if (chain.size() >= 1U) {
        for (std::size_t elementIndex = 0; elementIndex + 1 < chain.size(); ++elementIndex) {
            const std::vector<std::string> elementArgs = ce::getArgs(chain[elementIndex]);
            for (std::size_t a = 0; a < elementArgs.size(); ++a) {
                if (untouchables.find(elementArgs[a]) != untouchables.end()) {
                    remainingArgsKey.insert(elementArgs[a]);
                }
            }
        }
    }

    // prefix all untouchables with "u_" inside chain
    if (!untouchables.empty()) {
        std::map<std::string, std::string> replMap2;
        for (std::set<std::string>::const_iterator it = untouchables.begin();
            it != untouchables.end(); ++it) {
            replMap2[*it] = std::string("u_") + *it;
        }
        for (std::size_t i = 0; i < chain.size(); ++i) {
            chain[i] = ce::replaceKeysInString(chain[i], replMap2);
        }
    }

    // head is last element; chain are the preceding elements
    assert(!chain.empty());
    const std::string head = chain.back();
    chain.pop_back();

    const std::string implication = reconstructImplication(chain, head);

    return std::make_tuple(implication, chain, head, remainingArgsKey, zeroArgName, sName);
}

// Turn optimizations OFF for just this section
//#pragma optimize("", off)

// Member of gl::ExpressionAnalyzer
void ExpressionAnalyzer::addTheoremToMemory(const std::string& expr,
                                            Memory& memory,
                                            int iteration,
                                            bool proved,
                                            Dependencies& dependencyTable) {
    // 1) Disintegrate implication -> chain (left nodes) + head (rightmost)
    std::vector< std::tuple<
        std::string,                  // left expression
        std::vector<std::string>,     // node args
        std::set<std::string>         // left-node arguments
    > > tempChain;
    const std::string head = ce::disintegrateImplication(expr, tempChain, this->coreExpressionMap);

    std::vector<std::string> chain;
    chain.reserve(tempChain.size());
    for (std::size_t i = 0; i < tempChain.size(); ++i) chain.push_back(std::get<0>(tempChain[i]));

    // 2) Walk/build the memory path for the chain
    Memory* memoryBlock = &memory;
    for (std::size_t index = 0; index < chain.size(); ++index) {
        const std::string& element = chain[index];
        bool isNewBody = false;

        std::map<std::string, Memory*>::iterator it = memoryBlock->simpleMap.find(element);
        if (it != memoryBlock->simpleMap.end() && it->second != NULL) {
            memoryBlock = it->second;
        } else {
            Memory* child = new Memory();
            memoryBlock->simpleMap[element] = child;

            child->parentMemory = memoryBlock;
            child->level = static_cast<int>(index);
            child->exprKey = element;

            this->permanentBodies.push_back(child);

            memoryBlock = child;
            isNewBody = true;

            std::set<int> lv; lv.insert(child->level);
            std::pair<std::string, std::vector<ExpressionWithValidity>> origin;
            origin.first = "task formulation";
            this->addExprToMemoryBlock(element, *child, iteration, 0, lv, origin, -1, -1, "main", false);
        }

        // 3) At the last node: proved or set up recursion/auxiliaries
        if (index == chain.size() - 1) {
            if (proved) {
                this->addExprToMemoryBlock(head, *memoryBlock, iteration, 1, std::set<int>(), std::pair<std::string, std::vector<ExpressionWithValidity>>(), -1, -1, "main", false);

                // 1. Capture the vector returned by getArgs
                std::vector<std::string> argsVec = ce::getArgs(head);
                // 2. Use the range constructor of std::set
                std::set<std::string> argsSet(argsVec.begin(), argsVec.end());
            } else {
                // digits / immutables
                const std::set<std::string> digits = findDigitArgs(expr, this->anchorInfo, coreExpressionMap);
                for (std::set<std::string>::const_iterator dit = digits.begin(); dit != digits.end(); ++dit) {
                    const std::string digitArg = *dit;
                    std::set<std::string> immutables = this->findImmutableArgs(expr, digitArg);

                    // recX label for equality block, generic 'rec' for the auxy implication chain
                    const std::string recArg = std::string("rec") + std::to_string(memoryBlock->recursionCounter);
                    

                    std::string auxyImplication;
                    std::vector<std::string> auxyChain;
                    std::string auxyHead;
                    std::set<std::string> remainingArgsKey;
                    std::string zeroArgName;
                    std::string sName;

                    // createAuxyImplication(expr, digitArg, "rec", immutables, anchorName)
                    {
                        std::tuple<
                            std::string,
                            std::vector<std::string>,
                            std::string,
                            std::set<std::string>,
                            std::string,
                            std::string
                        > tpl = createAuxyImplication(expr, digitArg, "rec", immutables, this->anchorInfo.name);

                        auxyImplication  = std::get<0>(tpl);
                        auxyChain        = std::get<1>(tpl);
                        auxyHead         = std::get<2>(tpl);
                        remainingArgsKey = std::get<3>(tpl);
                        zeroArgName      = std::get<4>(tpl);
                        sName            = std::get<5>(tpl);
                    }

                    // --- Recursion block #1 ----------------------------------------------------
                    const std::string tempExpr  = std::string("(in2[") + recArg + "," + digitArg + "," + sName + "])";
                    const std::string tempExpr2 = std::string("(in2[") + "rec"   + "," + digitArg + "," + sName + "])";

 

                    // ensure unique
                    if (memoryBlock->simpleMap.find(tempExpr) == memoryBlock->simpleMap.end()) {
                        Memory* tempMb = new Memory();
                        memoryBlock->simpleMap[tempExpr] = tempMb;

                        tempMb->level = static_cast<int>(index + 1);
                        tempMb->parentMemory = memoryBlock;
                        tempMb->exprKey = tempExpr;
                        tempMb->isPartOfRecursion = true;



                        // history tag for auxy implication
						const ExpressionWithValidity auxyImplicationVal(auxyImplication, "main");
                        addOrigin(tempMb->exprOriginMap, auxyImplicationVal, std::make_pair("recursion", std::vector<ExpressionWithValidity>()), (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
                        addOrigin(tempMb->mailOut.exprOriginMap, auxyImplicationVal, std::make_pair("recursion", std::vector<ExpressionWithValidity>()), (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));

                        this->permanentBodies.push_back(tempMb);

                        std::set<int> lv;
                        lv.insert(tempMb->level);

                        this->addToHashMemory(auxyChain, auxyHead, remainingArgsKey,
                            *tempMb, tempMb->overallHashMemory, lv,
                            auxyImplication,
                            parameters.inductionMaxAdmissionDepth,
                            parameters.inductionMaxSecondaryNumber,
                            true,
                            parameters.minNumOperatorsKey,
                            "implication", true, auxyImplication, "main");


                        // add (in2[rec,...]) to temp block
                        {
                            std::set<int> lv; lv.insert(tempMb->level);
							std::pair<std::string, std::vector<ExpressionWithValidity>> or1 = std::make_pair("recursion", std::vector<ExpressionWithValidity>());
                            this->addExprToMemoryBlock(tempExpr2, *tempMb, iteration, 0, lv, or1, -1, -1, "main", false);
                        }

                        // queue head as "to be proved" with auxy index
                        this->addExprToMemoryBlock(head,
                                                   *tempMb,
                                                   iteration,
                                                   2,
                                                   std::set<int>(),
                                                   std::pair<std::string, std::vector<ExpressionWithValidity>>(),
							                       -1,     
                                                   dependencyTable.auxyIndex,
                                                   "main",
                                                   false);

                        // update dependency table (original entry collects this auxy)
                        dependencyTable.originalAuxyMap[dependencyTable.originalIndex] = DependencyItem();
                        dependencyTable.originalAuxyMap[dependencyTable.originalIndex].auxies.insert(dependencyTable.auxyIndex);
                        dependencyTable.originalAuxyMap[dependencyTable.originalIndex].expr = expr;
                        dependencyTable.auxyOriginalMap[dependencyTable.auxyIndex] = dependencyTable.originalIndex;
                        dependencyTable.auxyIndex += 1;
                    }

                    // --- Recursion block #2 (equality side-chain) -----------------------------
                    const std::string tempExpr3 = std::string("(=[s(") + recArg + ")," + zeroArgName + "])";
                    const std::string tempExpr4 = std::string("(=[") + digitArg + "," + zeroArgName + "])";

                    if (memoryBlock->simpleMap.find(tempExpr3) == memoryBlock->simpleMap.end()) {
                        Memory* tempMb2 = new Memory();
                        memoryBlock->simpleMap[tempExpr3] = tempMb2;

                        tempMb2->level = static_cast<int>(index + 1);
                        tempMb2->parentMemory = memoryBlock;
                        tempMb2->exprKey = tempExpr4;
                        tempMb2->isPartOfRecursion = true;

                        {
                            std::set<int> lv; lv.insert(tempMb2->level);
							std::pair<std::string, std::vector<ExpressionWithValidity>> or2 = std::make_pair("recursion", std::vector<ExpressionWithValidity>());
                            this->addExprToMemoryBlock(tempExpr4, *tempMb2, iteration, 0, lv, or2, -1, -1, "main", false);
                        }

                        this->addExprToMemoryBlock(head,
                                                   *tempMb2,
                                                   iteration,
                                                   2,
                                                   std::set<int>(),
                                                   std::pair<std::string, std::vector<ExpressionWithValidity>>(),
                                                   -1,
                                                   dependencyTable.auxyIndex,
                                                   "main",
                                                   false);

                        tempMb2->isActive = false;

                        this->permanentBodies.push_back(tempMb2);

                        dependencyTable.auxyOriginalMap[dependencyTable.auxyIndex] = dependencyTable.originalIndex;
                        dependencyTable.originalAuxyMap[dependencyTable.originalIndex].auxies.insert(dependencyTable.auxyIndex);
                        dependencyTable.originalInductionVariableMap[dependencyTable.originalIndex] = std::make_tuple(digitArg, std::to_string(memoryBlock->recursionCounter));
                        dependencyTable.auxyIndex += 1;
                        dependencyTable.originalIndex += 1;
                        memoryBlock->recursionCounter += 1;
                    }
                }

                // finally mark head as "to be proved" at the original block too
                EncodedExpression headEnc(head, "main");
                if (memoryBlock->wholeExpressions.find(headEnc) != memoryBlock->wholeExpressions.end()) {
                    // Head already derived (e.g. from anchor disintegration) — register directly
                    std::string fullTheorem = reconstructImplication(chain, head);
                    {
                        std::lock_guard<std::mutex> lock(this->theoremListMutex);
                        this->globalTheoremList.emplace_back(fullTheorem, "direct", "-1", "-1");
                    }
                    std::cout << fullTheorem << std::endl;
                } else {
                    this->addExprToMemoryBlock(head, *memoryBlock, iteration, 2, std::set<int>(), std::pair<std::string, std::vector<ExpressionWithValidity>>(), -1, -1, "main", false);
                }
            }

            // --- Incubator: create contradiction LB ---
            if (parameters.try_contradiction && !proved) {
                // Trap: reformulated operator conjecture?
                // head = =[x,a], one non-anchor operator premise, equality concerns output var.
                // Create __contradiction__(op[a]) under the ANCHOR (predecessor), seeded with
                // clean operator expression. Any contradiction in that LB -> !(op[a]).
                bool reformulatedOperator = false;
                if (parameters.incubator_mode) {
                    std::string eqExpr = ce::extractExpression(head);
                    if (eqExpr == "=" && chain.size() >= 2) {
                        const std::string& lastPremise = chain.back();
                        std::string opName = ce::extractExpression(lastPremise);
                        auto cfgIt = coreExpressionMap.find(opName);
                        if (cfgIt != coreExpressionMap.end() && !cfgIt->second.outputIndices.empty()) {
                            int outIdx = cfgIt->second.outputIndices[0];
                            std::vector<std::string> opArgs = ce::getArgs(lastPremise);
                            std::vector<std::string> eqArgs = ce::getArgs(head);
                            if (eqArgs.size() == 2 && outIdx < static_cast<int>(opArgs.size())) {
                                std::string x = opArgs[outIdx];
                                std::string a_out;
                                if (eqArgs[0] == x) a_out = eqArgs[1];
                                else if (eqArgs[1] == x) a_out = eqArgs[0];

                                if (!a_out.empty()) {
                                    // Back-reformulate: clean operator with original output var
                                    opArgs[outIdx] = a_out;
                                    std::string cleanOp = "(" + opName + "[";
                                    for (size_t ci = 0; ci < opArgs.size(); ++ci) {
                                        if (ci > 0) cleanOp += ",";
                                        cleanOp += opArgs[ci];
                                    }
                                    cleanOp += "])";

                                    std::vector<std::string> outerChain(chain.begin(), chain.end() - 1);
                                    std::string negCleanOp = "!" + cleanOp;
                                    std::string contrTheorem = reconstructImplication(outerChain, negCleanOp);

                                    // Create __contradiction__(op[a]) under the predecessor (Anchor LB)
                                    Memory* predecessor = memoryBlock->parentMemory;
                                    const std::string contradictionKey = "__contradiction__" + cleanOp;
                                    if (predecessor != nullptr &&
                                        predecessor->simpleMap.find(contradictionKey) == predecessor->simpleMap.end()) {

                                        Memory* contradictionLB = new Memory();
                                        predecessor->simpleMap[contradictionKey] = contradictionLB;

                                        contradictionLB->parentMemory = predecessor;
                                        contradictionLB->level = static_cast<int>(index);
                                        contradictionLB->exprKey = contradictionKey;
                                        contradictionLB->primedForContradiction = true;
                                        contradictionLB->contradictionTheorem = contrTheorem;

                                        this->permanentBodies.push_back(contradictionLB);

                                        // Seed with clean operator expression as fuel
                                        std::set<int> lv; lv.insert(contradictionLB->level);
                                        std::pair<std::string, std::vector<ExpressionWithValidity>> origin;
                                        origin.first = "task formulation";
                                        this->addExprToMemoryBlock(cleanOp, *contradictionLB, iteration, 0, lv, origin, -1, -1, "main", false);

                                    }
                                    reformulatedOperator = true;
                                }
                            }
                        }
                    }
                }

                if (!reformulatedOperator) {
                    const std::string contradictionKey = "__contradiction__" + head;
                    if (memoryBlock->simpleMap.find(contradictionKey) == memoryBlock->simpleMap.end()) {
                        Memory* contradictionLB = new Memory();
                        memoryBlock->simpleMap[contradictionKey] = contradictionLB;

                        contradictionLB->parentMemory = memoryBlock;
                        contradictionLB->level = static_cast<int>(index + 1);
                        contradictionLB->exprKey = contradictionKey;
                        contradictionLB->primedForContradiction = true;
                        contradictionLB->contradictionTheorem = reconstructImplication(chain, negate(head));

                        this->permanentBodies.push_back(contradictionLB);

                        // Add the head expression as fuel (status 0)
                        std::set<int> lv; lv.insert(contradictionLB->level);
                        std::pair<std::string, std::vector<ExpressionWithValidity>> origin;
                        origin.first = "task formulation";
                        this->addExprToMemoryBlock(head, *contradictionLB, iteration, 0, lv, origin, -1, -1, "main", false);

                    }
                }
            }

        }
    }
}
// Turn optimizations OFF for just this section
//#pragma optimize("", off)



void ExpressionAnalyzer::revisitRejected2(const std::string& markedExpr,
    Memory& memoryBlock,
    std::string validityName)
{
    ExpressionWithValidity evKey(markedExpr, validityName);
    auto& rm = memoryBlock.overallHashMemory.rejectedMap;

    // Guard: prevent re-entrant processing of the same marker
    if (memoryBlock.overallHashMemory.revisitInProgress.count(evKey)) {
        return;
    }

    auto itRM = rm.find(evKey);
    if (itRM == rm.end()) {
        return;
    }

    memoryBlock.overallHashMemory.revisitInProgress.insert(evKey);

    // Snapshot the rejected set — the loop below may indirectly insert
    // new entries into rm[evKey] via addExprToMemoryBlock → updateRejectedMap.
    // We process only the entries that existed before this call.
    std::set<RejectedMapValue> snapshot = itRM->second;

    // Remove only the snapshot entries; preserve any newly added ones.
    for (const auto& val : snapshot) {
        itRM->second.erase(val);
    }
    if (itRM->second.empty()) {
        rm.erase(itRM);
    }

    for (const auto& val : snapshot) {
        const std::string& rejectedStmt = val.renamedExpression;
        std::vector<std::string> markedArgs = ce::getArgs(markedExpr);
        std::vector<std::string> rejectedArgs = ce::getArgs(rejectedStmt);

        if (markedArgs.size() != rejectedArgs.size()) {
            continue;
        }

        int markerIndex = -1;
        for (size_t i = 0; i < markedArgs.size(); ++i) {
            if (markedArgs[i] == "marker") {
                markerIndex = static_cast<int>(i);
                break;
            }
        }
        assert(markerIndex != -1);

        const std::string rejectedVariable = rejectedArgs[markerIndex];

        bool argsIdentical = true;
        for (size_t i = 0; i < markedArgs.size(); ++i) {
            if (static_cast<int>(i) == markerIndex) continue;
            if (markedArgs[i] != rejectedArgs[i]) {
                argsIdentical = false;
                break;
            }
        }
        assert(argsIdentical && "Mismatch between marked and rejected args (other than marker)");
        if (!argsIdentical) continue;

        // IMPORTANT:
        // val.expression is the compact existence expression, not the expanded one.
        // We must reconstruct the expanded form and record the missing
        // compact -> expansion -> disintegration history chain.
        const std::string& compactExpr = val.expression;
        const int iteration = val.iteration;

        EncodedExpression encCompactExpr(compactExpr, validityName);
        auto itLev = memoryBlock.statementLevelsMap.find(encCompactExpr);
        assert(itLev != memoryBlock.statementLevelsMap.end());
        std::set<int> levels = itLev->second;

        std::string replExpr = prefixArgumentsWithU(compactExpr);
        Instruction instructions;
        prepareIntegrationCore(replExpr, instructions, memoryBlock, compactExpr);

        auto itInstr = std::find_if(instructions.data.begin(), instructions.data.end(),
            [&](const LogicalEntity& le) { return le.signature == replExpr; });

        if (itInstr == instructions.data.end()) continue;

        const LogicalEntity& ent = *itInstr;
        assert(ent.category == "existence" && "Reprocessed instruction must be existence");
        if (ent.category != "existence") continue;

        std::vector<std::string> removedArgs = listLastRemovedArgsLE(ent);
        if (removedArgs.empty()) continue;

        std::string boundVar = removedArgs[0];
        std::map<std::string, std::string> replacementMap;
        replacementMap[boundVar] = rejectedVariable;

        LogicalEntity modifiedEnt = ent;
        for (std::string& elem : modifiedEnt.elements) {
            elem = ce::replaceKeysInString(elem, replacementMap);
        }

        const std::string expandedExpr = expandSignature(modifiedEnt);

        if (parameters.trackHistory) {
            std::pair<std::string, std::vector<ExpressionWithValidity>> originExpansion;
            originExpansion.first = "expansion";
            originExpansion.second.push_back(ExpressionWithValidity(compactExpr, validityName));

            ExpressionWithValidity expandVal(expandedExpr, validityName);
            addOrigin(memoryBlock.exprOriginMap,
                expandVal,
                originExpansion,
                (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
            addOrigin(memoryBlock.mailOut.exprOriginMap,
                expandVal,
                originExpansion,
                (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
        }

        std::pair<std::string, std::vector<ExpressionWithValidity>> localOrigin =
            std::make_pair("disintegration", std::vector<ExpressionWithValidity>());
        localOrigin.second.push_back(ExpressionWithValidity(expandedExpr, validityName));

        for (const std::string& elem : ent.elements) {
            std::string s = ce::replaceKeysInString(elem, replacementMap);
            std::string removedU = removeUPrefixFromArguments(s);

            addExprToMemoryBlock(removedU,
                memoryBlock,
                iteration,
                0,
                levels,
                localOrigin,
                -1,
                -1,
                validityName,
                false);
        }
    }

    memoryBlock.overallHashMemory.revisitInProgress.erase(evKey);
    cleanAdmissionMap(markedExpr, validityName, memoryBlock);
}





// Remove "u_" prefixes that begin a token inside bracket lists, i.e. after '[' or ','
std::string ExpressionAnalyzer::replaceUSubstrings(const std::string& s) {
    std::string out = s;
    if (out.size() < 2) return out;

    std::size_t i = 1; // we look back at i-1
    while (i + 1 < out.size()) {
        if (out[i] == 'u' && out[i + 1] == '_' &&
            (out[i - 1] == '[' || out[i - 1] == ',')) {
            // erase "u_"
            out.erase(i, 2);
            // do not advance i; check again at the same spot
            continue;
        }
        ++i;
    }
    return out;
}


void ExpressionAnalyzer::updateAdmissionMapRecursion(const std::string& expression,
    Memory& mb,
    const std::string& validityName)
{
    // 1. Initial Guard Clauses: Check Operator
    std::string core = ce::extractExpression(expression);
    if (this->operators.find(core) == this->operators.end()) {
        return;
    }

    auto itComp = this->coreExpressionMap.find(core);
    assert(itComp != this->coreExpressionMap.end());

    const auto& cfg = itComp->second;
    assert(!cfg.outputIndices.empty());

    // 2. Identify Output Argument
    std::vector<std::string> args = ce::getArgs(expression);
    int outIdx = cfg.outputIndices[0];

    assert(outIdx >= 0 && outIdx < static_cast<int>(args.size()));

    std::string inputOutputArg = args[outIdx];

    // 3. Prepare Lookup
    std::map<std::string, std::string> markerMap;
    markerMap[inputOutputArg] = "marker";
    std::string markedExpr = ce::replaceKeysInString(expression, markerMap);

    if (!mb.overallHashMemory.admissionStatusMap[ExpressionWithValidity(markedExpr, validityName)])
    {
        return;
    }

    // 4. Lookup in Admission Map
    ExpressionWithValidity evMarked(markedExpr, validityName);
    auto itAdm = mb.overallHashMemory.admissionMap.find(evMarked);

    // Guard: Return if not found
    if (itAdm == mb.overallHashMemory.admissionMap.end()) {
        return;
    }

    // 5. Iterate over Admission Values (Snapshot)
    std::vector<AdmissionMapValue> values(itAdm->second.begin(), itAdm->second.end());

    for (const auto& val : values) {
        for (const std::string& keyElem : val.key) {

            // --- Flattened Logic Start ---

            std::string kCore = ce::extractExpression(keyElem);

            // Skip non-operators
            if (this->operators.find(kCore) == this->operators.end()) continue;

            auto itKComp = this->coreExpressionMap.find(kCore);
            assert(itKComp != this->coreExpressionMap.end());
            assert(!itKComp->second.outputIndices.empty());

            std::vector<std::string> kArgs = ce::getArgs(keyElem);
            int kOutIdx = itKComp->second.outputIndices[0];

            assert(kOutIdx >= 0 && kOutIdx < static_cast<int>(kArgs.size()));

            std::string kOutputArg = kArgs[kOutIdx];

            // Condition: Output arg must NOT be in remaining args
            if (val.remainingArgs.find(kOutputArg) != val.remainingArgs.end()) continue;

            // Check for Match
            std::map<std::string, std::string> kMarkerMap;
            kMarkerMap[kOutputArg] = "marker";
            std::string testExpr = ce::replaceKeysInString(keyElem, kMarkerMap);

            if (testExpr != markedExpr) continue;

            // --- Match Found ---

            mb.overallHashMemory.productsOfRecursion.insert(inputOutputArg);
            mb.overallHashMemory.productsOfRecursionIds.insert(mb.nameMap.encode(inputOutputArg));

            // 6. Construct New Admission Entry
            std::map<std::string, std::string> replacementMap;
            replacementMap[kOutputArg] = inputOutputArg;

            std::vector<std::string> newKey;
            newKey.reserve(val.key.size());
            for (const std::string& k : val.key) {
                newKey.push_back(ce::replaceKeysInString(k, replacementMap));
            }

            std::set<std::string> newRemainingArgs = val.remainingArgs;
            newRemainingArgs.insert(inputOutputArg);

            // 7. Validate New Key Candidates
            for (const std::string& newKeyElem : newKey) {
                std::string newCore = ce::extractExpression(newKeyElem);

                if (this->operators.find(newCore) == this->operators.end()) continue;

                auto itNewComp = this->coreExpressionMap.find(newCore);
                assert(itNewComp != this->coreExpressionMap.end());
                assert(!itNewComp->second.outputIndices.empty());

                std::vector<std::string> newKArgs = ce::getArgs(newKeyElem);
                int newOutIdx = itNewComp->second.outputIndices[0];

                assert(newOutIdx >= 0 && newOutIdx < static_cast<int>(newKArgs.size()));

                std::string newOutputArg = newKArgs[newOutIdx];

                // Condition: New output NOT in new remaining args
                if (newRemainingArgs.find(newOutputArg) != newRemainingArgs.end()) continue;

                // Condition: All other inputs MUST be in new remaining args
                bool allInputsPresent = true;
                for (int i = 0; i < static_cast<int>(newKArgs.size()); ++i) {
                    if (i == newOutIdx) continue;
                    if (newRemainingArgs.find(newKArgs[i]) == newRemainingArgs.end()) {
                        allInputsPresent = false;
                        break;
                    }
                }

                if (!allInputsPresent) continue;

                // 8. Insert New Rule
                std::map<std::string, std::string> newMarkerMap;
                newMarkerMap[newOutputArg] = "marker";
                std::string newMarkedExpr = ce::replaceKeysInString(newKeyElem, newMarkerMap);

                // COPY matched value 'val' to preserve:
                // - standardMaxAdmissionDepth
                // - standardMaxSecondaryNumber
                // - flag
                AdmissionMapValue newVal = val;

                // Overwrite the specific fields for the new entry
                newVal.key = newKey;
                newVal.remainingArgs = newRemainingArgs;

                mb.overallHashMemory.admissionMap[ExpressionWithValidity(newMarkedExpr, validityName)].insert(newVal);
                mb.overallHashMemory.admissionStatusMap[ExpressionWithValidity(newMarkedExpr, validityName)] = mb.overallHashMemory.admissionStatusMap[ExpressionWithValidity(markedExpr, validityName)];
                this->revisitRejected2(newMarkedExpr, mb, validityName);
            }
        }
    }
}

//#pragma optimize("", off)






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

void ExpressionAnalyzer::releaseCEBatchMemory() {
    // ---- 0) Defensive: clear tiny per-iteration queues (cheap, may carry strings) ----
    inductionMemoryBlocks.clear();
    std::vector<Memory*>().swap(inductionMemoryBlocks);

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

void ExpressionAnalyzer::destroyParentChildrenMap(ParentChildrenMap& m) {
    for (ParentChildrenMap::iterator it = m.begin(); it != m.end(); ++it)
        std::vector<Memory*>().swap(it->second);
    m.clear();
    m.rehash(0);
}

void ExpressionAnalyzer::destroyMailboxes(PerCoreMailboxes& boxes) {
    for (PerCoreMailboxes::iterator it = boxes.begin(); it != boxes.end(); ++it) {
        std::vector<Mail>& slots = it->second;
        for (std::size_t i = 0; i < slots.size(); ++i) {
            Mail& mb = slots[i];
            mb.statements.clear();
            mb.implications.clear();
            mb.exprOriginMap.clear();
            Mail empty; using std::swap; swap(mb, empty); // free capacity
        }
        std::vector<Mail>().swap(slots);
    }
    boxes.clear();
    boxes.rehash(0);
}






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




using gl::ParentChildrenMap;
using gl::PerCoreMailboxes;

ParentChildrenMap ExpressionAnalyzer::buildParentChildrenMap(const std::vector<Memory*>& roots) const {
    // 1) Collect reachable nodes and direct children
    std::unordered_map<Memory*, std::vector<Memory*>> direct;
    std::unordered_set<Memory*> seen;
    std::vector<Memory*> stack;
    for (Memory* r : roots) if (r) stack.push_back(r);

    while (!stack.empty()) {
        Memory* cur = stack.back(); stack.pop_back();
        if (!cur || !seen.insert(cur).second) continue;

        std::vector<Memory*> kids;
        kids.reserve(cur->simpleMap.size());
        for (std::map<std::string, Memory*>::const_iterator it = cur->simpleMap.begin();
            it != cur->simpleMap.end(); ++it) {
            Memory* ch = it->second;
            if (!ch) continue;
            kids.push_back(ch);
            stack.push_back(ch);
        }
        direct[cur] = std::move(kids);
    }

    // 2) Transitive closure with memoization; cycle-safe
    std::unordered_map<Memory*, std::unordered_set<Memory*>> memo;
    std::unordered_set<Memory*> visiting;

    std::function<const std::unordered_set<Memory*>& (Memory*)> dfs =
        [&](Memory* node) -> const std::unordered_set<Memory*>&{
        std::unordered_map<Memory*, std::unordered_set<Memory*>>::iterator mit = memo.find(node);
        if (mit != memo.end()) return mit->second;

        if (!visiting.insert(node).second) {
            static const std::unordered_set<Memory*> empty;
            return empty; // break cycles defensively
        }

        std::unordered_set<Memory*> acc;
        std::vector<Memory*>& kids = direct[node]; // default-constructed empty if missing
        for (std::size_t i = 0; i < kids.size(); ++i) {
            Memory* ch = kids[i];
            if (!ch) continue;
            acc.insert(ch);
            const std::unordered_set<Memory*>& sub = dfs(ch);
            acc.insert(sub.begin(), sub.end());
        }

        visiting.erase(node);
        return memo.emplace(node, std::move(acc)).first->second;
        };

    // Sort direct keys by exprKey for deterministic DFS order
    std::vector<Memory*> directKeys;
    directKeys.reserve(direct.size());
    for (auto& kv : direct) directKeys.push_back(kv.first);
    std::sort(directKeys.begin(), directKeys.end(),
        [](const Memory* a, const Memory* b) { return a->exprKey < b->exprKey; });
    for (Memory* dk : directKeys) dfs(dk);

    // 3) Convert sets to sorted vectors for deterministic order
    ParentChildrenMap out;
    out.reserve(memo.size());
    for (auto& kv : memo) {
        std::vector<Memory*> v(kv.second.begin(), kv.second.end());
        std::sort(v.begin(), v.end(),
            [](const Memory* a, const Memory* b) { return a->exprKey < b->exprKey; });
        out.emplace(kv.first, std::move(v));
    }

    return out;
}


PerCoreMailboxes ExpressionAnalyzer::buildPerCoreMailboxes(const ParentChildrenMap& index) const {
    unsigned cores = logicalCores;

    // Recipients are all values in the index
    std::unordered_set<Memory*> recipients;
    for (const auto& kv : index) 
        for (Memory* ch : kv.second) 
            if (ch) recipients.insert(ch);

    PerCoreMailboxes boxes;
    for (Memory* b : recipients) 
        boxes[b] = std::vector<Mail>(cores);
    return boxes;
}

void ExpressionAnalyzer::smashMail(PerCoreMailboxes& boxes) const{
    // Sort by body exprKey for deterministic processing order
    std::vector<std::pair<Memory*, std::vector<Mail>*>> sortedBoxes;
    sortedBoxes.reserve(boxes.size());
    for (auto& kv : boxes) {
        if (kv.first) sortedBoxes.push_back({kv.first, &kv.second});
    }
    std::sort(sortedBoxes.begin(), sortedBoxes.end(),
        [](const auto& a, const auto& b) { return a.first->exprKey < b.first->exprKey; });

    for (auto& [body, slotsPtr] : sortedBoxes) {
        // Merge statements and implications (set insert — order-independent)
        for (Mail& m : *slotsPtr) {
            body->mailIn.statements.insert(m.statements.begin(), m.statements.end());
            body->mailIn.implications.insert(m.implications.begin(), m.implications.end());
            m.statements.clear();
            m.implications.clear();
        }

        // Merge exprOriginMap: accumulate origins (order-independent union)
        for (Mail& m : *slotsPtr) {
            for (auto& kv2 : m.exprOriginMap) {
                auto& existing = body->mailIn.exprOriginMap[kv2.first];
                for (auto& orig : kv2.second) {
                    if (std::find(existing.begin(), existing.end(), orig) == existing.end())
                        existing.push_back(std::move(orig));
                }
            }
            m.exprOriginMap.clear();
        }
    }
}

void ExpressionAnalyzer::activateZeroCondition(Memory& memoryBlock)
{
    std::vector<std::string> args0 = ce::getArgs(memoryBlock.exprKey);
    std::string zeroArgName = this->findZeroArgName(memoryBlock);
    std::string eqExpr = std::string("(=[s(") + args0[0] + ")," + zeroArgName + "])";
    if (memoryBlock.parentMemory != NULL) {
        std::map<std::string, Memory*>::iterator itChild =
            memoryBlock.parentMemory->simpleMap.find(eqExpr);
        if (itChild != memoryBlock.parentMemory->simpleMap.end() && itChild->second) {
            itChild->second->isActive = true;
        }
    }
}

void ExpressionAnalyzer::proveKernel(const std::vector<Memory*>& bodies,
    const ParentChildrenMap& index,
    PerCoreMailboxes& boxes) {

    if (bodies.empty()) return;

    const unsigned workers = logicalCores;

    std::atomic<std::size_t> next{ 0 };

    auto worker = [this, &bodies, &index, &boxes, &next, workers](unsigned coreId) {
        const unsigned cid = workers ? (coreId % workers) : 0U;
        for (;;) {
            std::size_t i = next.fetch_add(1, std::memory_order_relaxed);
            if (i >= bodies.size()) break;

            Memory* b = bodies[i];
            if (!b || !b->isActive) continue;

            this->performElementaryLogicalStep(*b, cid, index, boxes);
        }
        };

    std::vector<std::thread> pool;
    pool.reserve(workers);
    for (unsigned t = 0; t < workers; ++t) pool.emplace_back(worker, t);
    for (auto& th : pool) th.join();

    this->smashMail(boxes);  // activate recipients for next iteration

    // Sort thread-collected vectors for deterministic processing order
    std::sort(inductionMemoryBlocks.begin(), inductionMemoryBlocks.end(),
        [](const Memory* a, const Memory* b) { return a->exprKey < b->exprKey; });
    std::sort(updateGlobalTuples.begin(), updateGlobalTuples.end());
    std::sort(updateGlobalDirectTuples.begin(), updateGlobalDirectTuples.end());

    for (int blockIndex = 0; blockIndex < inductionMemoryBlocks.size(); blockIndex++)
    {
        Memory* memoryBlock = inductionMemoryBlocks[blockIndex];
        this->activateZeroCondition(*memoryBlock);
    }

    for (const std::tuple<int, bool, int>& t : updateGlobalTuples)
    {
        updateGlobal(std::get<0>(t), std::get<1>(t), std::get<2>(t));
    }

    for (const std::tuple<std::string, int>& t : updateGlobalDirectTuples)
    {
        updateGlobalDirect(std::get<0>(t), std::get<1>(t));
    }



	inductionMemoryBlocks.clear();
	updateGlobalTuples.clear();
	updateGlobalDirectTuples.clear();
}

void ExpressionAnalyzer::removeExpressionFromMemoryBlock(const EncodedExpression& encExpr, Memory& mb, int state) {
    if (state == 0)
    {
        // Erase matching entries from int vectors in parallel with string vectors.
        // Find indices to remove, erase from back to front.
        auto eraseParallel = [](auto& strVec, auto& intVec, const EncodedExpression& target) {
            for (std::size_t i = strVec.size(); i-- > 0; ) {
                if (strVec[i] == target) {
                    strVec.erase(strVec.begin() + static_cast<std::ptrdiff_t>(i));
                    intVec.erase(intVec.begin() + static_cast<std::ptrdiff_t>(i));
                }
            }
        };
        eraseParallel(mb.localEncodedStatements, mb.intLocalEncodedStatements, encExpr);
        eraseParallel(mb.localEncodedStatementsDelta, mb.intLocalEncodedStatementsDelta, encExpr);
        eraseParallel(mb.encodedStatements, mb.intEncodedStatements, encExpr);
    }
    else if (state == 1) {
        // Remove from toBeProved
        assert(mb.toBeProved.count(encExpr) == 1);
        mb.toBeProved.erase(encExpr);
    }
}
// Turn optimizations OFF for just this section
//#pragma optimize("", off)


void ExpressionAnalyzer::cleanUpIntegrationPreparationCore(const Instruction& instructions,
    Memory& mb)
{
    for (const LogicalEntity& le : instructions.data)
    {
        // Case A: Implication with all u_ args
        if (le.category == "implication") {
            std::string cleanSig = removeUPrefixFromArguments(le.signature);
            std::string prefixVar0 = cleanSig + "_var0_";

            // 1. Go through all statements
            std::set<EncodedExpression> stmtsToRemove;
            for (const auto& stmt : mb.encodedStatements) {
                if (stmt.validityName == cleanSig || stmt.validityName.rfind(prefixVar0, 0) == 0) {
                    stmtsToRemove.insert(stmt);
                }
            }
            for (const auto& stmt : stmtsToRemove) {
                removeExpressionFromMemoryBlock(stmt, mb, 0);
            }

            // 2. Go through toBeProved
            std::vector<EncodedExpression> tbpToRemove;
            for (const auto& kv : mb.toBeProved) {
                if (kv.first.validityName == cleanSig) {
                    tbpToRemove.push_back(kv.first);
                }
            }
            for (const auto& stmt : tbpToRemove) {
                removeExpressionFromMemoryBlock(stmt, mb, 1);
            }
        }

        if (le.category == "existence" || le.category == "and")
        {
            std::vector<std::string> key(le.elements.begin(), le.elements.end());
        }
    }
}


// Turn optimizations OFF for just this section
//#pragma optimize("", off)
void ExpressionAnalyzer::cleanUpIntegrationPreparation(const std::string& expression,
    Memory& memoryBlock)
{
    std::string replExpr = prefixArgumentsWithU(expression);
    Instruction instructions;
    prepareIntegrationCore(replExpr, instructions, memoryBlock, expression);

    cleanUpIntegrationPreparationCore(instructions, memoryBlock);
}

std::string ExpressionAnalyzer::prefixArgumentsWithU(const std::string& expr) {
    std::vector<std::string> args = ce::getArgs(expr);
    std::map<std::string, std::string> replacementMap;

    for (const std::string& arg : args) {


        replacementMap[arg] = "u_" + arg;
    }

    return ce::replaceKeysInString(expr, replacementMap);
}

std::string ExpressionAnalyzer::removeUPrefixFromArguments(const std::string& expr) {
    std::vector<std::string> args = ce::getArgs(expr);
    std::map<std::string, std::string> replacementMap;

    for (const std::string& arg : args) {
        // Only remove "u_" if the argument actually starts with it
        if (startsWith(arg, "u_", 2)) {
            replacementMap[arg] = arg.substr(2);
        }
    }

    return ce::replaceKeysInString(expr, replacementMap);
}

std::string ExpressionAnalyzer::addMissingU(const std::string& expr) {
    std::vector<std::string> args = ce::getArgs(expr);
    std::map<std::string, std::string> replacementMap;

    for (const std::string& arg : args) {
        // Check if the argument does NOT start with "u_"
        if (!startsWith(arg, "u_", 2) && arg != "marker") {
            replacementMap[arg] = "u_" + arg;
        }
    }

    // Replace the arguments in the expression string
    return ce::replaceKeysInString(expr, replacementMap);
}

std::vector<std::string> ExpressionAnalyzer::listLastRemovedArgsLE(const LogicalEntity& le) {
    // Check that the category is explicitly "existence"
    assert(le.category == "existence" && "LogicalEntity category must be 'existence'");

    std::set<std::string> uniqueArgs;

    for (const std::string& element : le.elements) {
        std::vector<std::string> args = ce::getArgs(element);
        for (const std::string& arg : args) {
            // Check if arg does NOT start with "u_"
            if (!startsWith(arg, "u_", 2)) {
                uniqueArgs.insert(arg);
            }
        }
    }

    // Convert set back to vector (sorted by default due to std::set)
    return std::vector<std::string>(uniqueArgs.begin(), uniqueArgs.end());
}

// Turn optimizations OFF for just this section
//#pragma optimize("", off)
void ExpressionAnalyzer::makeAdmissionKeys(const std::vector<std::string>& implication,
    const std::string& expression,
    HashMemory& localMemory,
    std::string validityName)
{
    // 1. Check expression arguments
    //    Must have at least one "repl_" and all others must be "u_", "repl_", or "marker"
    std::vector<std::string> exprArgs = ce::getArgs(expression);
    bool hasRepl = false;

    for (const std::string& arg : exprArgs) {
        if (startsWith(arg, "repl_", 5)) {
            hasRepl = true;
        }
        else if (startsWith(arg, "u_", 2)) {
            continue;
        }
        else if (arg == "marker") {
            continue;
        }
        else {
            // Argument is not allowed (must be u_, repl_, or marker)
            return;
        }
    }

    if (!hasRepl) {
        return;
    }

    localMemory.triggersForAdmissionSetIntegration.insert(ExpressionWithValidity(expression, validityName));

    // 2. Validate Head vs Expression
    if (implication.empty()) return;
    const std::string& head = implication.back();

    // A. Check Core Expression Match
    std::string headCore = ce::extractExpressionUniversal(head);
    std::string exprCore = ce::extractExpressionUniversal(expression);

    if (headCore != exprCore) {
        return;
    }

    // B. Check Negation Status Match
    //    Assuming canonical form where negation always puts '!' at the start
    bool headIsNegated = (!head.empty() && head[0] == '!');
    bool exprIsNegated = (!expression.empty() && expression[0] == '!');

    if (headIsNegated != exprIsNegated) {
        return;
    }

    // C. Check Arguments and 1:1 "u_" relation
    std::vector<std::string> headArgs = ce::getArgs(head);
    assert(headArgs.size() == exprArgs.size());

    std::map<std::string, std::string> replacementMap;

    for (size_t i = 0; i < headArgs.size(); ++i) {
        const std::string& hArg = headArgs[i];
        const std::string& eArg = exprArgs[i];

        bool hIsU = startsWith(hArg, "u_", 2);
        bool eIsU = startsWith(eArg, "u_", 2);

        // "Important: head and expression must have all 'u_' args in the same position."
        if (hIsU || eIsU) {
            if (hArg != eArg) {
                return; // Violation of 1:1 relation
            }
        }
        else {
            // Map non-u head arg to expression arg
            replacementMap[hArg] = eArg;
        }
    }

    // 3. Apply Mapping to Implication Body (Head excluded)
    for (size_t i = 0; i < implication.size() - 1; ++i) {
        std::string mappedElement = ce::replaceKeysInString(implication[i], replacementMap);

        // Analyze args of the mapped element
        std::vector<std::string> outArgs = ce::getArgs(mappedElement);
        int markerCount = 0;
        bool argsValid = true;

        for (const std::string& arg : outArgs) {
            if (arg == "marker") {
                markerCount++;
            }
            else if (startsWith(arg, "repl_", 5)) {
                continue;
            }
            else if (startsWith(arg, "u_", 2)) {
                continue;
            }
            else {
                argsValid = false;
                break;
            }
        }

        // Admission Condition: Exactly one "marker" and valid args
        if (argsValid && markerCount == 1) {
            localMemory.admissionSetIntegration.insert(ExpressionWithValidity(removeUPrefixFromArguments(mappedElement), validityName));
        }
    }
}

// Turn optimizations OFF for just this section
//#pragma optimize("", off)


std::string ExpressionAnalyzer::expandSignature(const LogicalEntity& le)
{
    const std::string& category = le.category;
    const std::string& signature = le.signature;
    const std::vector<std::string>& elements = le.elements;

    // 1. Assert that all signature arguments start with 'u_'
    std::vector<std::string> sigArgsVec = ce::getArgs(signature);
    std::set<std::string> sigArgs(sigArgsVec.begin(), sigArgsVec.end());

    for (const auto& arg : sigArgs) {
        bool startsWithU = (arg.size() >= 2 && arg[0] == 'u' && arg[1] == '_');
        assert(startsWithU && "Signature arguments must start with 'u_' as per assumption.");
    }

    std::string result;

    // --- CASE 1: AND (&) ---
    if (category == "and") {
        if (elements.empty()) result = signature;
        else if (elements.size() == 1) result = elements[0];
        else {
            // Build nested hierarchy: (& result next)
            std::string current = elements[0];
            for (size_t i = 1; i < elements.size(); ++i) {
                current = "(&" + current + elements[i] + ")";
            }
            result = current;
        }
    }

    // --- CASE 2: EXISTENCE ---
    else if (category == "existence") {
        // Expecting at least 2 elements: [0]=body/definition, [1]=head/claim
        if (elements.size() < 2) result = signature;
        else {
            std::string body = elements[0];
            std::string head = elements[1];

            // Handle Head Negation: "remove internal negation" -> effectively invert
            std::string negatedHead;
            if (!head.empty() && head[0] == '!') {
                negatedHead = head.substr(1); // Remove existing '!'
            }
            else {
                negatedHead = "!" + head;     // Add '!'
            }

            // Determine Bound Variables: Args in body that are NOT in signature
            std::vector<std::string> bodyArgs = ce::getArgs(body);
            std::set<std::string> boundVars;

            for (const auto& arg : bodyArgs) {
                if (sigArgs.find(arg) == sigArgs.end()) {
                    boundVars.insert(arg);
                }
            }

            // Construct: !(>[bound_vars] body negated_head)
            std::string varsStr;
            for (const auto& v : boundVars) {
                if (!varsStr.empty()) varsStr += ",";
                varsStr += v;
            }

            result = "!(>[" + varsStr + "]" + body + negatedHead + ")";
        }
    }

    // --- CASE 3: IMPLICATION ---
    else if (category == "implication") {
        if (elements.empty()) result = signature;
        else {
            std::vector<std::string> chain = elements;
            std::string head = chain.back();
            chain.pop_back();

            result = this->reconstructImplicationFullBind(chain, head);
        }
    }

    // --- ERROR ---
    else {
        assert(false && "Forbidden category encountered in expandSignature. Must be 'and', 'existence', or 'implication'.");
        return signature;
    }

    // --- FINAL RENAMING (Skipped for Implication) ---
    if (category == "implication") {
        return result;
    }

    // 1. Collect all tokens from the resulting string to identify what needs renaming
    std::set<std::string> tokens;
    ce::TreeNode1* root = ce::parseExpr(result);

    std::vector<ce::TreeNode1*> stack;
    if (root) {
        stack.push_back(root);
    }

    while (!stack.empty()) {
        ce::TreeNode1* curr = stack.back();
        stack.pop_back();

        // Extract arguments from the current node's value (e.g., ">[a,b]" or "Macro[a,b]")
        std::vector<std::string> args = ce::getArgs(curr->value);

        for (const std::string& t : args) {
            // Simple trim (if needed, though parseExpr strips most whitespace)
            size_t first = t.find_first_not_of(" \t\r\n");
            if (first != std::string::npos) {
                size_t last = t.find_last_not_of(" \t\r\n");
                tokens.insert(t.substr(first, (last - first + 1)));
            }
        }

        // Push children to the stack to continue traversal
        if (curr->left) stack.push_back(curr->left);
        if (curr->right) stack.push_back(curr->right);
    }

    // Clean up the allocated syntax tree
    ce::deleteTree(root);

    // 2. Build the Replacement Map based on rules
    std::map<std::string, std::string> renameMap;
    // Regex for it_... and int_... patterns
    const std::regex reIt(R"(^it_\d+_lev_\d+_\d+$)");
    const std::regex reInt(R"(^int_lev_\d+_\d+$)");

    for (const std::string& t : tokens) {
        // Rule A: All u_ args lose u_
        if (t.size() >= 2 && t[0] == 'u' && t[1] == '_') {
            renameMap[t] = t.substr(2);
        }
        // Rule B: All other args...
        else {
            // ...except it_... and int_...
            if (std::regex_match(t, reIt) || std::regex_match(t, reInt)) {
                // Keep as is
            }
            // ...gain c_ prefix
            else {
                renameMap[t] = "c_" + t;
            }
        }
    }

    // 3. Apply Renaming
    return ce::replaceKeysInString(result, renameMap);
}

// Turn optimizations OFF for just this section
//#pragma optimize("", off)

// In GL_Quick_VS/GL_Quick/src/analyze_expressions.cpp

void ExpressionAnalyzer::disintegrateExprCore2(const std::string& expr,
    std::vector<LogicalEntity>& instructions,
    Memory& memoryBlock,
    int iteration,
    std::map<std::string, std::pair<std::set<std::string>, std::set<std::string>>>& collected,
    std::map<std::string, std::vector<std::string>>& newVarMap,
    std::string validityName)
{
    // Capture startInt at the start of core() as reference
    int referenceStartInt = memoryBlock.startInt;

    if (iteration == -1)
    {
        iteration = 0;
    }

    const std::string addedExpression = addMissingU(expr);

    // 1. Find the instruction corresponding to the current expression
    auto it = std::find_if(instructions.begin(), instructions.end(),
        [&](const LogicalEntity& le) { return le.signature == addedExpression; });

    // Determine the current statement signature for the map key
    std::string currentStatement;
    if (it == instructions.end()) {
        currentStatement = removeUPrefixFromArguments(addedExpression);
        collected[currentStatement]; // Ensure entry exists
        return;
    }

    const LogicalEntity& ent = *it;
    currentStatement = removeUPrefixFromArguments(ent.signature);
    collected[currentStatement]; // Ensure entry exists

    // --- LAMBDA: History Tracking ---
    // Handles "expansion" origin for the entity, and optionally "disintegration" origin for its children
    auto trackExpansionHistory = [&](const LogicalEntity& entityToTrack,
        bool trackChildren = false)
        {
            if (!parameters.trackHistory) return;

            // 1. Record Expansion Origin
            std::pair<std::string, std::vector<ExpressionWithValidity>> originExpansion;
            originExpansion.first = "expansion";
            originExpansion.second.push_back(ExpressionWithValidity(removeUPrefixFromArguments(expr), validityName));

            std::string expandedSignature = expandSignature(entityToTrack);
            ExpressionWithValidity expandVal(expandedSignature, validityName);

            // Using the addOrigin helper handles the std::vector insertion 
            // and respects the max_origins limit automatically.
            addOrigin(memoryBlock.exprOriginMap, expandVal, originExpansion, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
            addOrigin(memoryBlock.mailOut.exprOriginMap, expandVal, originExpansion, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));

            // 2. Record Disintegration Origin for Children (if requested)
            if (trackChildren) {
                std::pair<std::string, std::vector<ExpressionWithValidity>> originDisintegration;
                originDisintegration.first = "disintegration";
                originDisintegration.second.push_back(ExpressionWithValidity(expandedSignature, validityName));

                for (const auto& elem : entityToTrack.elements) {
                    ExpressionWithValidity elemVal(removeUPrefixFromArguments(elem), validityName);

                    // addOrigin handles the vector push_back and respects compressor_max_origins_per_expr
                    addOrigin(memoryBlock.exprOriginMap, elemVal, originDisintegration, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
                    addOrigin(memoryBlock.mailOut.exprOriginMap, elemVal, originDisintegration, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
                }
            }
        };
    // --------------------------------

    if (ent.category == "implication") {
        assert(!ent.elements.empty() && "Implication category cannot be empty!");

        std::vector<std::string> key(ent.elements.begin(), ent.elements.end() - 1);
        std::string value = ent.elements.back();

        // Add implication specifically to THIS statement's entry (first set)
        collected[currentStatement].first.insert(reconstructImplicationFullBind(key, value));

        // Use Lambda (Children not tracked for implication in original code)
        trackExpansionHistory(ent, false);
    }
    else if (ent.category == "and") {
        // Insert elements into the second set, removing the "u_" prefix first
        for (const auto& el : ent.elements) {
            collected[currentStatement].second.insert(removeUPrefixFromArguments(el));
        }

        // Recurse on all elements
        for (const auto& elem : ent.elements) {
            disintegrateExprCore2(elem, instructions, memoryBlock, iteration, collected, newVarMap, validityName);
        }

        // Use Lambda (Children tracked for AND)
        trackExpansionHistory(ent, true);
    }
    else if (ent.category == "existence")
    {
        std::vector<std::string> removedArgs = listLastRemovedArgsLE(ent);
        if (removedArgs.empty()) {
            for (const auto& elem : ent.elements) {
                disintegrateExprCore2(elem, instructions, memoryBlock, iteration, collected, newVarMap, validityName);
            }
            return;
        }

        assert(removedArgs.size() == 1);
        std::string boundVar = removedArgs[0];

        // Helper lambda to process a specific path (Unconditional Generation)
        auto processPath = [&](const std::string& newVar, std::vector<LogicalEntity> instructions) {

            // Create Renamed Elements
            std::map<std::string, std::string> replacementMap;
            replacementMap[boundVar] = newVar;

            std::vector<std::string> renamedElements;
            renamedElements.reserve(ent.elements.size());
            for (const auto& elem : ent.elements) {
                renamedElements.push_back(ce::replaceKeysInString(elem, replacementMap));
            }

            // CRUCIAL: Map the new variable to its immediate defining elements
            newVarMap[newVar] = renamedElements;

            std::map<std::string, std::string> replacementMap2;
            assert(!startsWith(boundVar, "u_", 2));
            replacementMap2[boundVar] = "u_" + newVar;

            // Apply replacement map on all signatures and elements of every instruction
            for (LogicalEntity& le : instructions) {
                le.signature = ce::replaceKeysInString(le.signature, replacementMap2);
                for (std::string& elem : le.elements) {
                    elem = ce::replaceKeysInString(elem, replacementMap2);
                }
            }

            // Recurse using the modified instructions
            for (const auto& rel : renamedElements) {
                disintegrateExprCore2(rel, instructions, memoryBlock, iteration, collected, newVarMap, validityName);
            }
            };

        // Check if at least one element is an operator expression
        bool hasOperator = false;
        for (const std::string& el : ent.elements) {
            std::string core = ce::extractExpression(el);
            if (this->operators.find(core) != this->operators.end()) {
                hasOperator = true;
                break;
            }
        }

        // 1. Generate "it_" path (Algebra) - Conditional
        if (hasOperator)
        {
            std::string newVar = "it_" + std::to_string(iteration) +
                "_lev_" + std::to_string(memoryBlock.level) +
                "_" + std::to_string(memoryBlock.startInt);

            memoryBlock.startInt++;

            if (parameters.trackHistory)
            {
                std::map<std::string, std::string> replacementMap3;
                assert(!startsWith(boundVar, "u_", 2));
                replacementMap3[boundVar] = newVar;

                LogicalEntity modifiedEnt = ent;
                for (auto& elem : modifiedEnt.elements) {
                    elem = ce::replaceKeysInString(elem, replacementMap3);
                }

                // Use Lambda (Children tracked for Algebra/Operator path)
                trackExpansionHistory(modifiedEnt, true);
            }

            processPath(newVar, instructions);
        }

        // 2. Unconditionally generate "int_" path (Integration)
        {
            std::string newVar = "int_lev_" + std::to_string(memoryBlock.level) +
                "_" + std::to_string(memoryBlock.startInt);


            memoryBlock.startInt++;

            if (parameters.trackHistory)
            {
                std::map<std::string, std::string> replacementMap3;
                assert(!startsWith(boundVar, "u_", 2));
                replacementMap3[boundVar] = newVar;

                LogicalEntity modifiedEnt = ent;
                for (auto& elem : modifiedEnt.elements) {
                    elem = ce::replaceKeysInString(elem, replacementMap3);
                }

                // Use Lambda (Children tracked for Integration path)
                trackExpansionHistory(modifiedEnt, true);
            }

            processPath(newVar, instructions);
        }
    }
    else {
        // Default fallback
        // currentStatement was already inserted at top
    }
}

// Turn optimizations OFF for just this section
//#pragma optimize("", off)

std::tuple<std::set<std::string>, std::set<std::string>, int>
ExpressionAnalyzer::disintegrateExpr2(const std::string& expr,
    Memory& memoryBlock,
    int iteration,
    bool forceDeep,
    std::string validityName)
{
    int savedStartInt = memoryBlock.startInt;

    std::map<std::string, std::pair<std::set<std::string>, std::set<std::string>>> collected;
    std::map<std::string, std::vector<std::string>> newVarMap;
    Instruction instructions;

    std::string replExpr = prefixArgumentsWithU(expr);
    prepareIntegrationCore(replExpr, instructions, memoryBlock, expr);

    disintegrateExprCore2(replExpr,
        instructions.data,
        memoryBlock,
        iteration,
        collected,
        newVarMap,
        validityName);

    std::set<std::string> finalImplications;
    std::set<std::string> finalStatements;

    std::function<void(const std::string&)> addToFinal;
    addToFinal = [&](const std::string& s) {
        if (finalStatements.count(s)) return;
        finalStatements.insert(s);

        // If key is missing, add it to collected
        if (collected.find(s) == collected.end()) {
            collected[s]; // Creates empty pair
        }

        // Now access the entry (guaranteed to exist)
        auto& pr = collected[s];
        finalImplications.insert(pr.first.begin(), pr.first.end());


        // Recurse for every child
        for (const auto& child : pr.second) {
            addToFinal(child);
        }
    };

    if (forceDeep) {
        static const std::regex reIt(R"(^it_\d+_lev_(\d+)_(\d+)$)");

        // 1. Filter and Collect Phase
        for (auto const& [stmt, pr] : collected) {
            bool allow = true;
            std::vector<std::string> args = ce::getArgs(stmt);

            for (const std::string& arg : args) {
                std::smatch match;
                // Rule: If a statement has a new "it_" variable with level == mb.level, do not allow it.
                if (std::regex_match(arg, match, reIt)) {
                    try {
                        int level = std::stoi(match[1].str());
                        int id = std::stoi(match[2].str());

                        if (id >= savedStartInt && level == memoryBlock.level) {
                            allow = false;
                            break;
                        }
                    }
                    catch (...) {}
                }
            }

            if (allow) {
                addToFinal(stmt);
            }
        }

        memoryBlock.canBeSentSet.insert(finalStatements.begin(), finalStatements.end());
        memoryBlock.canBeSentSet.insert(finalImplications.begin(), finalImplications.end());
        return std::make_tuple(finalImplications, finalStatements, memoryBlock.startInt);
    }

    // --- FIX: Deferred Rejection Structures ---
    struct PendingRejection {
        std::string removedU;
        std::string markedExpr;
    };
    // Map var -> list of potential rejections
    std::map<std::string, std::vector<PendingRejection>> pendingRejections;
    // Track successfully admitted variables in this call
    std::set<std::string> admittedVars;

    static const std::regex reIt(R"(^it_\d+_lev_\d+_\d+$)");
    static const std::regex reInt(R"(^int_lev_\d+_\d+$)");
    std::string topLevelExprClean = removeUPrefixFromArguments(replExpr);

    size_t previousSize = 0;

    previousSize = finalStatements.size();

    // Pass A: Unconditional Statements
    for (auto const& [stmt, impls] : collected) {
        if (finalStatements.find(stmt) != finalStatements.end()) continue;

        std::vector<std::string> args = ce::getArgs(stmt);
        bool hasNewVars = false;
        for (const std::string& arg : args) {
            if (std::regex_match(arg, reIt) || std::regex_match(arg, reInt)) {
                size_t lastUnderscore = arg.rfind('_');
                size_t levPos = arg.find("_lev_"); // Find position of "_lev_" marker

                if (lastUnderscore != std::string::npos && levPos != std::string::npos) {
                    try {
                        // Extract ID (after last underscore)
                        int id = std::stoi(arg.substr(lastUnderscore + 1));

                        // Extract Level (between "_lev_" and last underscore)
                        // "_lev_" has length 5, so the number starts at levPos + 5
                        int varLevel = std::stoi(arg.substr(levPos + 5, lastUnderscore - (levPos + 5)));

                        // Compare both ID and Level
                        if (varLevel == memoryBlock.level && id >= savedStartInt) {
                            hasNewVars = true;
                            break;
                        }
                    }
                    catch (...) {}
                }
            }
        }

        if (!hasNewVars) {
			addToFinal(stmt);
        }
    }

    if (!parameters.compressor_mode && !parameters.incubator_mode) {
        // Pass B: New Variable Admission
        for (const auto& [var, expressions] : newVarMap)
        {
            bool isVarAdmitted = false;

            for (const auto& stmt : expressions) {
                std::string removedU = removeUPrefixFromArguments(stmt);
                std::string markedExpr = makeMarkedExpr(removedU, var);

                // 1. Check for Iteration Variable (it_...)
                if (std::regex_match(var, reIt)) {
                    std::string core = ce::extractExpression(removedU);
                    bool hasOperator = (this->operators.find(core) != this->operators.end());

                    if (hasOperator) {
                        if (isAdmitted(memoryBlock, removedU, var, markedExpr, validityName)) {
                            isVarAdmitted = true;
                            break;
                        }
                        else {
                            // Buffer rejection instead of calling updateRejectedMap immediately
                            pendingRejections[var].push_back({ removedU, markedExpr });
                        }
                    }
                }
                // 2. Check for Integration Variable (int_...)
                else if (std::regex_match(var, reInt)) {
                    if (isAdmittedIntegration(memoryBlock, removedU, var, markedExpr, validityName)) {
                        isVarAdmitted = true;
                        break;
                    }
                    else if (memoryBlock.overallHashMemory.admissionSetIntegration.find(ExpressionWithValidity(markedExpr, validityName)) !=
                        memoryBlock.overallHashMemory.admissionSetIntegration.end()) {
                        cleanAdmissionMap(markedExpr, validityName, memoryBlock);
                        isVarAdmitted = true;
                        break;
                    }
                }
            }

            if (isVarAdmitted) {
                admittedVars.insert(var); // Mark as locally admitted

                for (const auto& stmt : expressions) {
                    std::string cleanStmt = removeUPrefixFromArguments(stmt);
                    addToFinal(cleanStmt);
                }
            }
        }
    }

    // ---------------------------------------------------------
    // Cascade Admission for Dependent Variables (int_ and it_)
    // ---------------------------------------------------------

    std::string sLevel = std::to_string(memoryBlock.level);
    std::string intPrefix = "int_lev_" + sLevel + "_";
    std::string levSubstring = "_lev_" + sLevel + "_";

    while (true) {
        bool hasGrown = false;

        // Iterate through all potential statements in 'collected'
        for (const auto& stmt : finalStatements) 
        {
            std::vector<std::string> args = ce::getArgs(stmt);

            bool hasAdmittedVar = false;
            std::vector<std::string> candidates;

            // First pass: Classify arguments in this statement
            for (const std::string& arg : args) {
                // Check if already admitted
                if (admittedVars.find(arg) != admittedVars.end()) {
                    hasAdmittedVar = true;
                }
                // Check if it is a potential candidate (Target Syntax + New + Not Admitted)
                else {
                    bool isTarget = false;
                    // Check A: int_lev_X_
                    if (arg.rfind(intPrefix, 0) == 0) {
                        isTarget = true;
                    }
                    // Check B: it_..._lev_X_
                    else if (arg.size() > 3 && arg[0] == 'i' && arg[1] == 't' && arg[2] == '_') {
                        if (arg.find(levSubstring) != std::string::npos) {
                            isTarget = true;
                        }
                    }

                    if (isTarget) {
                        // Must be in newVarMap (new) and not yet admitted
                        if (newVarMap.find(arg) != newVarMap.end()) {
                            candidates.push_back(arg);
                        }
                    }
                }
            }

            // Admission Rule: New vars are admitted ONLY if they share an expression with an admitted var
            if (hasAdmittedVar && !candidates.empty()) {
                for (const auto& cand : candidates) {
                    // Double-check to ensure we don't process the same candidate twice in one pass
                    if (admittedVars.find(cand) == admittedVars.end()) {

                        admittedVars.insert(cand);
                        hasGrown = true;

                        // Pull in all defining expressions for this newly admitted variable
                        // Note: We use operator[] safely here because we verified existence in newVarMap above
                        for (const auto& rawExpr : newVarMap.find(cand)->second) {
                            std::string cleanStmt = removeUPrefixFromArguments(rawExpr);
                            addToFinal(cleanStmt);
                        }
                    }
                }
            }
        }

        // Break if no new variables were admitted in this pass
        if (!hasGrown) {
            break;
        }
    }



    // --- FIX: Commit Rejections Only for Never-Admitted Variables ---
    for (const auto& [var, failures] : pendingRejections) {
        if (admittedVars.find(var) == admittedVars.end()) {

            // Variable was truly rejected (dependencies never resolved)
            for (const auto& fail : failures) {
                updateRejectedMap(fail.removedU, fail.markedExpr, topLevelExprClean, memoryBlock.overallHashMemory, iteration, validityName);
            }
        }
    }

    return std::make_tuple(finalImplications, finalStatements, memoryBlock.startInt);
}






void ExpressionAnalyzer::overwriteOrigins(std::map<ExpressionWithValidity, std::vector<std::pair<std::string, std::vector<ExpressionWithValidity>>>>& left,
    const std::map<ExpressionWithValidity, std::vector<std::pair<std::string, std::vector<ExpressionWithValidity>>>>& right,
    int maxOrigins) {
    for (auto it = right.begin(); it != right.end(); ++it) {
        auto& leftVec = left[it->first];
        for (const auto& orig : it->second) {
            if (leftVec.size() < static_cast<size_t>(maxOrigins)) {
                if (std::find(leftVec.begin(), leftVec.end(), orig) == leftVec.end()) {
                    leftVec.push_back(orig);
                }
            }
        }
    }
}




// Turn optimizations OFF for just this section
//#pragma optimize("", off)

void ExpressionAnalyzer::prove(int numberIterations,
    const std::vector<Memory*>& bodies,
    const ParentChildrenMap& index,
    PerCoreMailboxes& boxes) {

    for (int it = 0; it < numberIterations; ++it) {
        // Debug: count total expressions across all active bodies
        std::size_t totalExprs = 0;
        std::size_t activeBodies = 0;
        for (std::size_t bi = 0; bi < bodies.size(); ++bi) {
            if (bodies[bi] && bodies[bi]->isActive) {
                totalExprs += bodies[bi]->encodedStatements.size();
                ++activeBodies;
            }
        }
        std::cout << "Hash burst: " << it
                  << "  active_bodies=" << activeBodies
                  << "  total_exprs=" << totalExprs << std::endl;

        auto burstStart = std::chrono::high_resolution_clock::now();
        proveKernel(bodies, index, boxes);                     // runs, then smashMail(boxes)
        auto burstEnd = std::chrono::high_resolution_clock::now();
        double burstSec = std::chrono::duration<double>(burstEnd - burstStart).count();
        std::cout << "  dt=" << burstSec << "s" << std::endl;
    }
}

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

void ExpressionAnalyzer::saveProvedTheorems() {
    namespace fs = std::filesystem;

    // Resolve path: <repo>/files/theorems/proved_theorems.txt
    const auto theoremsDir =
        fs::path(__FILE__).parent_path().parent_path().parent_path().parent_path()
        / "files" / "theorems";

    const auto filePath = theoremsDir / "proved_theorems.txt";

    // Create directory if it doesn't exist
    std::error_code ec;
    fs::create_directories(theoremsDir, ec);

    // Open in append mode
    std::ofstream ofs(filePath, std::ios::app);
    if (!ofs.is_open()) {
        std::cerr << "Error: Could not open " << filePath << " for appending." << std::endl;
        return;
    }

    int count = 0; // Move counter outside to use it after unlocking

    // START CHANGE: Explicit scope block
    {
        std::lock_guard<std::mutex> lock(this->theoremListMutex);

        for (const auto& entry : this->globalTheoremList) {
            // entry is tuple<theorem, method, var, recCounter>
            const std::string& theorem = std::get<0>(entry);
            const std::string& method = std::get<1>(entry);

            // Filter out debug/internal entries if necessary, or save all.
            // "debug" method entries are usually from findEnds and shouldn't be saved as proved theorems.
            if (method != "debug") {
                ofs << theorem << "\n";
                count++;
            }
        }
    }
    // END CHANGE: lock is released here, before the function exits or prints to console

    std::cout << "Saved " << count << " new proved theorems to file." << std::endl;
}

void ExpressionAnalyzer::saveCompressedTheorems(const std::vector<std::string>& compressedTheorems) {
    namespace fs = std::filesystem;

    const auto theoremsDir =
        fs::path(__FILE__).parent_path().parent_path().parent_path().parent_path()
        / "files" / "theorems";

    // Standard hardcoded file name
    const auto filePath = theoremsDir / "compressed_proved_theorems.txt";

    std::error_code ec;
    fs::create_directories(theoremsDir, ec);

    // IMPORTANT: std::ios::app appends to the file so tags don't overwrite each other
    std::ofstream ofs(filePath, std::ios::app);
    if (!ofs.is_open()) {
        std::cerr << "Error: Could not open " << filePath << " for writing." << std::endl;
        return;
    }

    for (const std::string& theorem : compressedTheorems) {
        ofs << theorem << "\n";
    }

    std::cout << "Appended " << compressedTheorems.size() << " surviving theorems to compressed_proved_theorems.txt" << std::endl;
}



void ExpressionAnalyzer::saveProvedTheoremsFiltered(const std::vector<std::string>& essentialTheorems,
                                                    const std::unordered_set<std::string>& externalTheorems) {
    namespace fs = std::filesystem;

    const auto theoremsDir =
        fs::path(__FILE__).parent_path().parent_path().parent_path().parent_path()
        / "files" / "theorems";

    const auto filePath = theoremsDir / "proved_theorems.txt";

    std::error_code ec;
    fs::create_directories(theoremsDir, ec);

    // Build compact→expanded lookup from globalTheoremList.
    // globalTheoremList stores expanded forms; the compressor may have
    // reverse-compiled existence heads to compact form.
    // We compile each globalTheoremList entry the same way the compressor
    // constructor does, then map back.
    std::unordered_map<std::string, std::string> compactToExpanded;
    for (const auto& tpl : globalTheoremList) {
        const std::string& expanded = std::get<0>(tpl);
        compactToExpanded[expanded] = expanded;  // identity: already expanded

        // Also try reverse-compiling to build compact key
        std::vector<std::tuple<std::string, std::vector<std::string>, std::set<std::string>>> tempChain;
        std::string head = ce::disintegrateImplication(expanded, tempChain, coreExpressionMap);

        if (startsWith(head, "!(>", 3)) {
            std::string compiledHead = compileCoreExpressionMapCore(
                head, implCounter, existenceCounter, statementCounter, variableCounter);
            std::vector<std::string> chain;
            for (const auto& t : tempChain) chain.push_back(std::get<0>(t));
            std::string compact = reconstructImplication(chain, compiledHead);
            compactToExpanded[compact] = expanded;
        }
    }

    std::ofstream ofs(filePath, std::ios::trunc);
    if (!ofs.is_open()) {
        std::cerr << "Error: Could not open " << filePath << " for writing." << std::endl;
        return;
    }

    // Build set of mirrored-of-external theorems to also exclude
    std::unordered_set<std::string> mirroredOfExternal;
    if (!externalTheorems.empty()) {
        for (const auto& tpl : globalTheoremList) {
            const std::string& expr = std::get<0>(tpl);
            const std::string& method = std::get<1>(tpl);
            const std::string& ref = std::get<2>(tpl);
            if (method == "mirrored statement" && externalTheorems.count(ref)) {
                mirroredOfExternal.insert(expr);
            }
        }
    }

    int written = 0;
    for (const std::string& theorem : essentialTheorems) {
        // Skip surviving external theorems and their mirrors — they are not GL-proved
        if (externalTheorems.count(theorem) || mirroredOfExternal.count(theorem)) continue;

        auto it = compactToExpanded.find(theorem);
        if (it != compactToExpanded.end()) {
            ofs << it->second << "\n";
        }
        else {
            // Theorem came from a previous batch (proved_theorems.txt) — already in
            // expanded form, save as-is.
            ofs << theorem << "\n";
        }
        ++written;
    }

    std::cout << "Rewrote proved_theorems.txt with " << written
        << " globally essential theorems (excluded "
        << (externalTheorems.size() + mirroredOfExternal.size())
        << " external/mirrored-of-external)." << std::endl;
}


void ExpressionAnalyzer::broadcastTheorems(const std::vector<std::string>& provedTheorems,
                                           const std::string& originTag) {
    if (provedTheorems.empty()) return;

    Mail broadcastMail;
    for (const std::string& thOriginal : provedTheorems) {

        // 1. Disintegrate the theorem to inspect its head
        std::vector<std::tuple<std::string, std::vector<std::string>, std::set<std::string>>> tempChain;
        std::string head = ce::disintegrateImplication(thOriginal, tempChain, this->coreExpressionMap);

        std::string finalTheorem = thOriginal;
        std::string finalHead = head;
        std::vector<std::string> finalChain;
        finalChain.reserve(tempChain.size());
        for (const auto& t : tempChain) finalChain.push_back(std::get<0>(t));

        // 2. Check if the head is an existence statement (starts with "!(>")
        if (startsWith(head, "!(>", 3)) {
            // Compile the head
            std::string compiledHead = this->compileCoreExpressionMapCore(
                head,
                this->implCounter,
                this->existenceCounter,
                this->statementCounter,
                this->variableCounter
            );

            // Reconstruct the theorem with the original chain and the NEW compiled head
            finalTheorem = this->reconstructImplication(finalChain, compiledHead);

            // Re-disintegrate the NEW theorem to ensure the chain/head structure is perfectly consistent
            // (e.g. reconstructImplication might reorder premises, though typically it preserves them)
            tempChain.clear();
            finalChain.clear();
            finalHead = ce::disintegrateImplication(finalTheorem, tempChain, this->coreExpressionMap);
            for (const auto& t : tempChain) finalChain.push_back(std::get<0>(t));
        }

        // 3. Queue the (potentially modified) theorem for broadcast
        broadcastMail.implications.insert(std::make_tuple(
            finalChain,
            finalHead,
            std::set<std::string>(), // remainingArgsKey (empty for global facts)
            std::set<int>(),         // levels (empty for global facts)
            finalTheorem
        ));

        if (parameters.trackHistory) {
            ExpressionWithValidity ev(finalTheorem, "main");
            auto originBroadcast = std::make_pair(originTag, std::vector<ExpressionWithValidity>());

            addOrigin(broadcastMail.exprOriginMap, ev, originBroadcast, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
        }
    }

    // Distribute to all permanent bodies
    for (Memory* mb : permanentBodies) {
        if (mb == nullptr) continue;

        // Ensure mailbox has slots
        if (boxes[mb].empty()) {
            boxes[mb].resize(logicalCores);
        }

        // Push to slot 0 (will be picked up by the corresponding worker)
        Mail& dest = boxes[mb][0];
        dest.implications.insert(broadcastMail.implications.begin(), broadcastMail.implications.end());

        if (parameters.trackHistory) {
            dest.exprOriginMap.insert(broadcastMail.exprOriginMap.begin(), broadcastMail.exprOriginMap.end());
        }
    }

    std::cout << "Distributed knowledge to " << permanentBodies.size() << " memory blocks." << std::endl;
}

// Turn optimizations OFF for just this section
//#pragma optimize("", off)

void ExpressionAnalyzer::prefillIntegrationMapsRecursive(Memory* mb) {
    if (mb == nullptr) return;
    if (mb->isPartOfRecursion)
    {
        return;
    }

    // Collect expression keys from the current block and all its predecessors
    std::vector<std::string> exprKeys;
    Memory* current = mb;
    while (current != nullptr) {
        if (!current->exprKey.empty()) {
            exprKeys.push_back(current->exprKey);
        }
        current = current->parentMemory;
    }

    // Process all collected expressions for the current block *mb
    for (const std::string& expr : exprKeys) {
        std::string core = ce::extractExpression(expr);
        if (this->operators.find(core) != this->operators.end()) {
            auto it = this->coreExpressionMap.find(core);
            assert(it != this->coreExpressionMap.end());

            auto itComp = this->compiledExpressions.find(core);
            assert(itComp != this->compiledExpressions.end());

            if (itComp->second.category != "atomic") {
                const ce::CoreExpressionConfig& cfg = it->second;
                assert(!cfg.outputIndices.empty());

                std::vector<std::string> args = ce::getArgs(expr);
                // Assuming the operator has one output variable as implied by context
                int outIdx = cfg.outputIndices[0];

                assert(outIdx >= 0 && outIdx < static_cast<int>(args.size()));

                std::string outputArg = args[outIdx];

                std::map<std::string, std::string> replacementMap;
                replacementMap[outputArg] = "marker";
                std::string rplExpr2 = ce::replaceKeysInString(expr, replacementMap);

                std::vector<std::string> tempArgs = ce::getArgs(rplExpr2);
                std::set<std::string> argSet(tempArgs.begin(), tempArgs.end());

                // Per instructions: erase marker from argSet and call prepareIntegration
                assert(argSet.find("marker") != argSet.end());

                argSet.erase("marker");
                prepareIntegration(rplExpr2, argSet, *mb, "main");
            }
        }
    }

    // Recurse into children
    for (std::map<std::string, Memory*>::iterator it = mb->simpleMap.begin(); it != mb->simpleMap.end(); ++it) {
        prefillIntegrationMapsRecursive(it->second);
    }
}

//#pragma optimize("", off)

void ExpressionAnalyzer::disintegrateExprHypothetically(const std::string& expr, Memory& memoryBlock, std::string validityName) {
    if (parameters.ban_disintegration) return;

    
    // 1. Extract input variables
    std::vector<std::string> inputVars = ce::getArgs(expr);

	for (const auto& var : inputVars) {
        if (var == "marker") {
            return; // Invalid variable name
        }
    }

    // 2. Call disintegrateExpr2 FIRST
    // We pass the current validityName. The resulting strings are raw.
    std::tuple<std::set<std::string>, std::set<std::string>, int> result =
        disintegrateExpr2(expr, memoryBlock, -1, true, validityName);

    // Only look at new statements (index 1)
    const std::set<std::string>& newStatements = std::get<1>(result);

    if (newStatements.size() == 1)
    {
        return;
    }

    // 3. Search for defSets for each input variable in the new statements
    std::vector<std::string> targetVars; // Variables with defSet == "(1)"

    for (const std::string& var : inputVars) {
        bool foundDef = false;
        std::string foundDefSetVal;

        for (const std::string& stmt : newStatements) {
            std::string coreName = ce::extractExpression(stmt);
            auto itCfg = this->coreExpressionMap.find(coreName);

            // "search for at least one expression which has entry in coreExpressionMap"
            if (itCfg != this->coreExpressionMap.end()) {
                const auto& cfg = itCfg->second;
                std::vector<std::string> sArgs = ce::getArgs(stmt);

                // Check if 'var' is an argument in this statement
                for (size_t k = 0; k < sArgs.size(); ++k) {
                    if (sArgs[k] == var) {
                        // "there read defSet"
                        std::string argPos = std::to_string(k + 1);
                        auto itDS = cfg.definitionSets.find(argPos);

                        if (itDS != cfg.definitionSets.end()) {
                            foundDefSetVal = itDS->second.first;
                            foundDef = true;
                            goto def_found; // Break out of statement loop
                        }
                    }
                }
            }
        }

    def_found:
        // "assert u found def set for each var"
        assert(foundDef);

        // "if there is at least one eligible var proceed"
        // Eligible means defSet == "(1)" based on context
        if (foundDefSetVal == "(1)") {
            targetVars.push_back(var);
        }
    }

    // "otherwise return"
    if (targetVars.empty()) {
        return;
    }

    // 4. Create a new validity name
    // Format: old + "_var" + index + "_" + variable ... + "_hypo"
    // Example: ..._var0_x_var1_y_hypo
    std::string newValidityName = validityName;
    for (size_t i = 0; i < targetVars.size(); ++i) {
        newValidityName += "_var" + std::to_string(i) + "_" + targetVars[i];
    }
    newValidityName += "_hypo_" + expr;

    // 5. Find levels
    std::set<int> levels;
    EncodedExpression originEnc(expr, validityName);
    auto itLev = memoryBlock.statementLevelsMap.find(originEnc);

    if (itLev != memoryBlock.statementLevelsMap.end()) {
        levels = itLev->second;
    }
    else {
        levels.insert(memoryBlock.level);
    }
    levels.insert(memoryBlock.level);

    // 6. Check Duplication and Add Equalities
    bool anyDuplicationFound = false;

    for (const std::string& var : targetVars) {
        bool foundDuplicationForVar = false;

        // Check duplication in newStatements
        for (const std::string& e : newStatements) {
            std::vector<std::string> eArgs = ce::getArgs(e);
            int count = 0;
            for (const std::string& arg : eArgs) {
                if (arg == var) count++;
            }
            if (count > 1) {
                foundDuplicationForVar = true;
                break;
            }
        }

        if (foundDuplicationForVar) {
            anyDuplicationFound = true;

            // Check if equality exists in validityName (not "main")
            bool hasExistingEquality = false;
            auto itMap = memoryBlock.equivalenceClassesMap.find(validityName);
            if (itMap != memoryBlock.equivalenceClassesMap.end()) {
                for (const auto& ec : itMap->second) {
                    if (ec.variables.count(var) && ec.variables.size() > 1) {
                        hasExistingEquality = true;
                        break;
                    }
                }
            }

            if (!hasExistingEquality) {
                std::string Y = var;
                std::string Y_copy = Y + "_copy";
                std::string equalityExpr = "(=[" + Y + "," + Y_copy + "])";
                std::string finalExpr = removeUPrefixFromArguments(equalityExpr);

                std::pair<std::string, std::vector<ExpressionWithValidity>> origin;
                origin.first = "necessity for equality (hypo)";
                origin.second.push_back(ExpressionWithValidity(expr, validityName));

                addExprToMemoryBlock(finalExpr, memoryBlock, -1, 0, levels, origin, -1, -1, validityName, false);
            }
        }
    }

    // 7. Register NEW STATEMENTS with NEW validity name

    for (const std::string& stmt : newStatements) {
        EncodedExpression enc(stmt, newValidityName);

        if (memoryBlock.wholeExpressions.find(enc) != memoryBlock.wholeExpressions.end()) {
            continue; // Skip if already present
		}

        //in3[2,2,2,4] finder:
        if (enc.original == "(in3[2,2,2,4])" && memoryBlock.exprKey == "(AnchorIncubator[1,2,3,4,5,6,7,8,9,10])")
        {
            int test = 0;
            test++;
        }
        memoryBlock.encodedStatements.push_back(enc);
        memoryBlock.localEncodedStatements.push_back(enc);
        memoryBlock.localEncodedStatementsDelta.push_back(enc);
        { IntEncodedExpr ie = encodeExpression(enc, memoryBlock.nameMap);
          memoryBlock.intEncodedStatements.push_back(ie);
          memoryBlock.intLocalEncodedStatements.push_back(ie);
          memoryBlock.intLocalEncodedStatementsDelta.push_back(ie); }
        memoryBlock.statementLevelsMap[enc] = levels;
        memoryBlock.intKnownStatements.insert(packStatementKey(
            memoryBlock.nameMap.encode(enc.original),
            memoryBlock.nameMap.encode(enc.validityName)));
        memoryBlock.wholeExpressions.insert(enc);
    }
}

//#pragma optimize("", off)

void ExpressionAnalyzer::prehandleAnchor(Memory* mb) {
    if (mb == nullptr) return;
    if (mb->isPartOfRecursion)
    {
        return;
    }

    std::string anchorPrefix = "(" + this->anchorInfo.name;
    bool isAnchorLB = (mb->exprKey.rfind(anchorPrefix, 0) == 0);

    // Skip processing for the Anchor LB itself, but allow recursion
    if (!isAnchorLB) {

        // 1. Trace the hierarchy to find the specific Anchor Key
        std::string anchorExprKey;
        Memory* current = mb;
        std::set<std::string> traceVariables; // Added: Collect variables from trace

        while (current != nullptr) {
            if (!current->exprKey.empty()) {
                // Check if this ancestor is the Anchor LB
                if (current->exprKey.rfind(anchorPrefix, 0) == 0) {
                    anchorExprKey = current->exprKey;
                    break;
                }
                // Added: Collect variables from the current trace element's key
                std::vector<std::string> kArgs = ce::getArgs(current->exprKey);
                traceVariables.insert(kArgs.begin(), kArgs.end());
            }
            current = current->parentMemory;
        }

        // 2. If we found an anchor ancestor, proceed
        if (!anchorExprKey.empty()) {
            std::vector<std::string> args = ce::getArgs(anchorExprKey);
            std::map<std::string, std::string> replacementMap;

            // 3. Create Replacement Map based on definitionSets == "(1)"
            //    and save the new variables to axedVariables
            for (const auto& [slot, pattern] : this->anchorInfo.definitionSets) {
                if (pattern == "(1)") {
                    try {
                        int index = std::stoi(slot) - 1;
                        if (index >= 0 && index < static_cast<int>(args.size())) {
                            std::string originalVar = args[index];

                            // Added: Check if the variable exists in the trace
                            if (traceVariables.find(originalVar) != traceVariables.end()) {
                                // Only apply x-prefix if not already present
                                if (originalVar.rfind("x", 0) != 0) {
                                    std::string xVar = "x" + originalVar;
                                    replacementMap[originalVar] = xVar;

                                    // Save to the new attribute
                                    mb->axedVariables.insert(xVar);
                                    mb->intAxedVariables.insert(mb->nameMap.encode(xVar));
                                }
                            }
                        }
                    }
                    catch (...) {
                        // Ignore malformed slots
                    }
                }
            }

            // 4. Create and Add the Anchor Expression using the map
            if (!replacementMap.empty()) {
                std::string replacedAnchor = ce::replaceKeysInString(anchorExprKey, replacementMap);

                EncodedExpression enc(replacedAnchor, "main");

                // Add only if not already present
                if (mb->wholeExpressions.find(enc) == mb->wholeExpressions.end()) {

                    std::set<int> levels{ 0 };

                    //in3[2,2,2,4] finder:
                    if (enc.original == "(in3[2,2,2,4])" && mb->exprKey == "(AnchorIncubator[1,2,3,4,5,6,7,8,9,10])")
                    {
                        int test = 0;
                        test++;
                    }
                    mb->localEncodedStatements.push_back(enc);
                    mb->localEncodedStatementsDelta.push_back(enc);
                    mb->encodedStatements.push_back(enc);
                    { IntEncodedExpr ie = encodeExpression(enc, mb->nameMap);
                      mb->intEncodedStatements.push_back(ie);
                      mb->intLocalEncodedStatements.push_back(ie);
                      mb->intLocalEncodedStatementsDelta.push_back(ie); }
                    mb->statementLevelsMap[enc] = levels;
                    mb->intKnownStatements.insert(packStatementKey(
                        mb->nameMap.encode(enc.original),
                        mb->nameMap.encode(enc.validityName)));
                    mb->wholeExpressions.insert(enc);

                    std::pair<std::string, std::vector<ExpressionWithValidity>> origin;
                    origin.first = "anchor handling";
                    // Cite the original anchor expression as the source
                    origin.second.push_back(ExpressionWithValidity(anchorExprKey, "main"));

                    ExpressionWithValidity encVal(replacedAnchor, "main");

                    // addOrigin handles the vector push_back and respects compressor_max_origins_per_expr
                    addOrigin(mb->exprOriginMap, encVal, origin, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
                }
            }
        }
    }

    // 5. Recurse into children
    for (auto it = mb->simpleMap.begin(); it != mb->simpleMap.end(); ++it) {
        prehandleAnchor(it->second);
    }
}

// Turn optimizations OFF for just this section
//#pragma optimize("", off)
void ExpressionAnalyzer::analyzeExpressions(const std::vector<std::string>& theorems,
    const std::vector<std::string>& provedTheorems,
    const std::vector<std::string>& externalTheorems) {

    auto t0 = std::chrono::high_resolution_clock::now();
    std::cout << "Counter example based filtering started." << std::endl;

    // 1. CE Filtering
    std::vector<std::string> filteredConjectures = theorems;

    if (!parameters.skip_ce_filter) {
        std::vector<std::vector<std::string>> simpleFactsLists = readSimpleFacts();
        std::cout << "CE filter: " << filteredConjectures.size() << " conjectures, "
                  << simpleFactsLists.size() << " fact files." << std::endl;
        for (const auto& simpleFacts : simpleFactsLists)
        {
            filteredConjectures = filterConjecturesWithCE(filteredConjectures, simpleFacts);
            std::cout << "CE filter: " << filteredConjectures.size() << " survived." << std::endl;
        }
        saveFilteredConjectures(filteredConjectures);
    }

    std::cout << "Counter example based filtering finished." << std::endl;
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dt = t1 - t0;
    std::cout << "Runtime counter example based filtering: " << dt.count() << " seconds\n";

    // 2. Register Theorems to be Proved (Conjectures)
    for (std::size_t ti = 0; ti < filteredConjectures.size(); ++ti) {
        this->addTheoremToMemory(filteredConjectures[ti], this->body, 0, false, this->globalDependencies);
    }

    this->prefillIntegrationMapsRecursive(&this->body);
    this->prehandleAnchor(&this->body);

    // 3. Build Grid / Connectivity
    index = buildParentChildrenMap(permanentBodies);
    boxes = buildPerCoreMailboxes(index);

    // 4. Initial Smash (Prime the system)
    this->smashMail(boxes);

    std::cout << "Prover started." << std::endl;
    t0 = std::chrono::high_resolution_clock::now();

    // -------------------------------------------------------------------------
    // PHASE 1: Run for 2 iterations (or max if less) to let system settle
    // -------------------------------------------------------------------------
    int preIterations = 2;
    if (parameters.maxIterationNumberProof < preIterations) {
        preIterations = parameters.maxIterationNumberProof;
    }

    if (preIterations > 0) {
        std::cout << "Phase 1: Running " << preIterations << " warm-up iterations..." << std::endl;
        this->prove(preIterations, permanentBodies, index, boxes);
    }

    // -------------------------------------------------------------------------
    // PHASE 2: Broadcast Proved Theorems & Run Remaining Iterations
    // -------------------------------------------------------------------------
    int remainingIterations = parameters.maxIterationNumberProof - preIterations;

    if (!provedTheorems.empty()) {
        std::cout << "Injecting " << provedTheorems.size() << " proved theorems via broadcast..." << std::endl;
        this->broadcastTheorems(provedTheorems);

        // IMPORTANT: Move broadcasted messages from 'boxes' to 'mailIn'
        // so they are visible to the threads in the next prove() call.
        this->smashMail(boxes);
    }

    if (!externalTheorems.empty()) {
        std::cout << "Injecting " << externalTheorems.size()
                  << " external theorems via broadcast..." << std::endl;
        this->broadcastTheorems(externalTheorems, "externally provided theorem");
        this->smashMail(boxes);
    }

    if (remainingIterations > 0) {
        std::cout << "Phase 2: Running remaining " << remainingIterations << " iterations..." << std::endl;
        this->prove(remainingIterations, permanentBodies, index, boxes);
    }

    t1 = std::chrono::high_resolution_clock::now();
    dt = std::chrono::duration<double>(t1 - t0);
    std::cout << "Prover finished." << std::endl;
    std::cout << "Runtime prover: " << dt.count() << " seconds" << std::endl;
}











} // namespace gl
