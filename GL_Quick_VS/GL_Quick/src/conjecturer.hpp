
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

#pragma once
#include <string>
#include <vector>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <tuple>
#include <regex>
#include <filesystem>
#include <atomic>
#include <thread>
#include <mutex>
#include <functional>
#include "compiler.hpp"

// #define CONJ_DEBUG_DUMP  // Uncomment to enable debug dumps for comparison with Python

namespace conj {

// ============================================================================
// Int-path constants
// ============================================================================

static constexpr int MAX_CONJ_ARGS   = 32;   // max args in a conjecture's def-set map
static constexpr int MAX_CONJ_BUF    = 256;  // max int16_t words in a flat conjecture
static constexpr int MAX_CONN_MAPS   = 512;  // max connection maps from cartesian product
static constexpr int MAX_DEFSET_GROUPS = 8;  // max distinct def-set types
static constexpr int MAX_PER_GROUP   = 16;   // max args per def-set group

// ============================================================================
// Int-path structs
// ============================================================================

/// Bidirectional name<->int16_t map for expression names and def-set texts.
/// Variables (already numeric in conjecture strings) are NOT in this map —
/// they are stored directly as int16_t.
struct ConjNameMap {
    std::unordered_map<std::string, int16_t> nameToId;
    std::vector<std::string> idToName;
    int16_t nextId = 1;  // 0 reserved for invalid/empty

    ConjNameMap() { idToName.push_back(""); }

    int16_t encode(const std::string& s) {
        auto it = nameToId.find(s);
        if (it != nameToId.end()) return it->second;
        int16_t id = nextId++;
        nameToId[s] = id;
        idToName.push_back(s);
        return id;
    }

    const std::string& decode(int16_t id) const {
        return idToName[static_cast<std::size_t>(id)];
    }

    int16_t lookup(const std::string& s) const {
        auto it = nameToId.find(s);
        return (it != nameToId.end()) ? it->second : 0;
    }
};

/// Flat int16_t buffer for a conjecture expression.
/// Layout: repeating blocks of [boundCount, bv0..bvN, nameId, arity, arg0..argN]
/// Last block is the head (boundCount=0).
struct IntConjBuf {
    int16_t data[MAX_CONJ_BUF];
    int16_t len = 0;
};

/// Argument→def-set mapping as parallel arrays (replaces DefSetMap in hot path).
struct IntDefSetMap {
    int16_t count = 0;
    int16_t argId[MAX_CONJ_ARGS];       // variable ID (1,2,3...)
    int16_t defSetId[MAX_CONJ_ARGS];    // ConjNameMap ID of def-set text
    int16_t combinable[MAX_CONJ_ARGS];  // 0 or 1
    int16_t connectable[MAX_CONJ_ARGS]; // 0 or 1
};

/// Connection map: argId → target argId. Array-indexed, no std::map.
struct IntConnMap {
    int16_t map[MAX_CONJ_ARGS * 2];  // map[argId] = targetArgId; 0 = unmapped
    int16_t maxArg = 0;              // highest argId in use
};

// ============================================================================
// Type aliases
// ============================================================================

// Definition set tuple: (text, combinable_flag, connectable_flag)
using DefSetTuple = std::tuple<std::string, bool, bool>;

// Map: arg_name -> DefSetTuple
using DefSetMap = std::map<std::string, DefSetTuple>;

// Pre-computed injective mappings: size -> {(left,right) -> list of maps}
using MappingsMap = std::map<int, std::map<std::pair<int,int>, std::vector<std::map<int,int>>>>;

// Permutations: chain_length -> list of permutation vectors
using PermutationsMap = std::map<int, std::vector<std::vector<int>>>;

// Binary sequences: length -> list of binary vectors
using BinarySeqsMap = std::map<int, std::vector<std::vector<int>>>;

// ============================================================================
// Configuration structs (mirrors Python's configuration_reader.py)
// ============================================================================

struct ExpressionDescription {
    int arity = 0;
    DefSetMap definition_sets;                   // pos_str -> (text, combinable, connectable)
    std::string full_mpl;
    std::string handle;
    std::string short_mpl_raw;
    std::string short_mpl_normalized;
    int max_count_per_conjecture = 0;
    int max_size_expression = 0;
    int min_size_expression = 1;
    std::vector<std::string> input_args;
    std::vector<std::string> output_args;
    std::vector<int> indices_input_args;         // 0-based positions
    std::vector<int> indices_output_args;         // 0-based positions
};

struct ConfigurationParameters {
    int min_number_simple_expressions = 2;
    int max_number_simple_expressions = 0;
    int max_size_mapping_def_set = 0;
    int max_number_args_expr = 0;
    int operator_threshold = 0;
    std::map<std::string, int> max_values_for_def_sets;
    std::map<std::string, int> max_values_for_uncomb_def_sets;
    std::map<std::string, int> max_values_for_def_sets_prior_connection;
    std::map<std::string, int> max_complexity_if_anchor_parameter_connected;
    int max_size_binary_list = 0;
    std::vector<int> simple_facts_parameters;
    std::vector<std::string> fact_variable_kinds;
    bool incubator_mode = false;
};

struct ConfigurationData {
    std::map<std::string, ExpressionDescription> data;    // expression name -> description
    std::vector<std::string> expressionOrder;              // JSON key insertion order
    ConfigurationParameters parameters;
    std::vector<std::string> patterns_to_exclude_raw;
    std::vector<std::regex> patterns_to_exclude;
    std::vector<std::string> only_in_head_raw;
    std::vector<std::regex> only_in_head_patterns;
    std::vector<std::set<std::string>> prohibited_combinations;
    std::vector<std::string> prohibited_heads;
    std::string theorems_folder;
    std::string background_theorems_folder;
    std::string anchor_name;
    std::string anchor_id;

    // Convenience: get anchor name (matches Python's get_anchor_name)
    std::string getAnchorName() const;
};

// ============================================================================
// Worker result
// ============================================================================

struct WorkerResult {
    // connected_list: (expression, def_set_map) pairs — intermediate connections
    std::vector<std::pair<std::string, DefSetMap>> connected_list;
    // connected_list2: final conjectures (connected to anchor)
    std::vector<std::string> connected_list2;
    // reshuffled: normalized forms
    std::vector<std::string> reshuffled_list;
    // mirrored: mirror variants
    std::vector<std::string> reshuffled_mirrored_list;
};

// ============================================================================
// RAII guard for TreeNode1
// ============================================================================

struct TreeGuard {
    ce::TreeNode1* root;
    explicit TreeGuard(ce::TreeNode1* r) : root(r) {}
    ~TreeGuard() { if (root) ce::deleteTree(root); }
    TreeGuard(const TreeGuard&) = delete;
    TreeGuard& operator=(const TreeGuard&) = delete;
    ce::TreeNode1* get() const { return root; }
    ce::TreeNode1* release() { auto* r = root; root = nullptr; return r; }
};

// ============================================================================
// Free utility functions (no config dependency)
// ============================================================================

bool repetitionsExist(const std::string& s);
std::pair<ce::TreeNode1*, std::vector<int>> parseDefSet(const std::string& s);
std::string treeToStrReorder(const ce::TreeNode1* root);
std::string treeToStr(const ce::TreeNode1* root, int offset, const std::map<int,int>& subMap);
std::pair<int,int> findMinMaxNumbers(const std::string& s);
std::set<int> findAllIds(const std::string& s);
void updateReplacementMap(std::map<int,int>& repMap, int num1, int num2);
std::pair<bool, std::map<int,int>> defSetsEqual(const std::string& ds1, const std::string& ds2);
std::pair<std::string, std::vector<int>> subtractAndReplaceNumbers(const std::string& s, int subtractValue, const std::map<int,int>& m);
std::pair<std::string, std::vector<int>> reorderNumbers(const std::string& s);
void shiftTogether(std::map<std::string,std::string>& argDefSetMap);

// connect_expression_sets returns (common_map, common_set, success, removed_args)
std::tuple<std::map<std::string,std::string>, std::set<std::pair<std::string,std::string>>, int, std::vector<std::string>>
    connectExpressionSets(
        const std::set<std::pair<std::string,std::string>>& set1,
        const std::set<std::pair<std::string,std::string>>& set2,
        const std::string& connectionType,
        int isDefinition,
        const std::set<std::string>& argsToRemove,
        bool afterGrooming);

std::string extractBetweenBrackets(const std::string& s, size_t startIndex = 0);
int findPositionSurrounded(const std::string& text, const std::string& substring);
std::vector<std::string> sortListAccordingToOccurrence(const std::vector<std::string>& lst, const std::string& text);
std::vector<std::string> findOrderedIntegers(const std::vector<std::string>& intStrings, const std::string& bigString);
std::string replaceIntegerInString(const std::string& bigString, const std::string& targetInt, const std::string& replacementInt);
std::string subtractNumberFromInts(const std::string& expr, int number, const std::set<int>& numbersToReplace, bool replaceAll);

int getNumberRemovableArgs(const std::map<std::string,std::string>& mapping);
bool mappingGood(const std::map<std::string,std::string>& mapping);

// ============================================================================
// Main class
// ============================================================================

class Conjecturer {
public:
    explicit Conjecturer(const std::string& anchorId);

    // Main entry point — generates conjectures, writes output files.
    // Equivalent to Python's create_expressions_parallel(config).
    void run();

private:
    // ---- Immutable after construction (thread-safe to share) ----
    ConfigurationData config_;
    ExpressionDescription anchor_;             // the anchor expression description
    MappingsMap mappingsMap_;
    MappingsMap mappingsMapAnchor_;
    BinarySeqsMap binarySeqsMap_;
    PermutationsMap allPermutations_;
    std::vector<std::string> operators_;        // expressions with input_args AND output_args
    std::vector<std::string> relations_;        // expressions with 2 input_args AND no output_args
    // Adapter for shim functions that need CoreExpressionConfig
    std::map<std::string, ce::CoreExpressionConfig> coreExprMap_;

    std::filesystem::path projectRoot_;

    // ---- Int-path data (immutable after construction) ----
    ConjNameMap nameMap_;
    IntConjBuf anchorInt_;                         // encoded anchor expression
    IntDefSetMap anchorDefSetsInt_;                 // encoded anchor def-set map
    std::vector<int16_t> operatorNameIds_;          // nameMap IDs of operator expressions
    std::vector<int16_t> relationNameIds_;          // nameMap IDs of relation expressions
    // Per-expression config indexed by nameId (dense lookup)
    struct IntExprConfig {
        int16_t nameId = 0;
        int16_t arity = 0;
        int16_t maxCountPerConj = 0;
        int16_t handleId = 0;                      // nameMap ID of handle string
        int16_t maxSizeExpr = 0;
        int16_t minSizeExpr = 1;
        int16_t indicesInputArgs[16];
        int16_t numInputArgs = 0;
        int16_t indicesOutputArgs[4];
        int16_t numOutputArgs = 0;
    };
    std::vector<IntExprConfig> intExprConfigs_;     // indexed by nameId
    // Config param limits indexed by defSetId
    std::vector<int16_t> maxForDefSets_;            // indexed by defSetId
    std::vector<int16_t> maxForUncombDefSets_;      // indexed by defSetId
    std::vector<int16_t> maxForDefSetsPrior_;       // indexed by defSetId
    std::vector<int16_t> maxComplexityAnchorConn_;  // indexed by defSetId

    // ---- Configuration loading ----
    ConfigurationData loadConfiguration(const std::string& anchorId);
    void buildCoreExprMapAdapter();

    // ---- Pre-computation ----
    static MappingsMap createMap(int N);
    static MappingsMap createMapAnchor(int leftMax, int rightMax);
    int determineLeftSideBoundary() const;
    int determineRightSideBoundary() const;

    // ---- Expression parsing & arg maps ----
    DefSetMap findArgMap(const std::string& expr) const;
    std::tuple<std::string, DefSetMap, std::map<std::string,std::string>>
        renameVariablesInExpr(const std::string& expr, bool deep) const;

    // ---- Expression connection ----
    std::tuple<bool, std::string, DefSetMap>
        connectExpressions(const std::string& expr1, const std::string& expr2,
                          const DefSetMap& map1, const DefSetMap& map2,
                          const std::map<std::string,std::string>& subMap,
                          const std::vector<int>& binaryList, bool connectToAnchor) const;

    std::vector<std::map<std::string,std::string>>
        makeAllConnectionMaps(const DefSetMap& map1, const DefSetMap& map2,
                             bool withAnchor, const MappingsMap& mappingsMap) const;

    // ---- Validation filters ----
    bool exprGood(const std::string& expr) const;
    bool exprGood2(const std::string& expr, int nse, const DefSetMap& connectedMap) const;
    bool numbersGood(const std::string& expr) const;
    bool checkDefSets(const DefSetMap& argMap) const;
    bool checkComplexityLevelForDefSets(const DefSetMap& argMap, int complexityLevel) const;
    bool qualifiedForEquality(const std::string& expr) const;
    bool evaluateOperatorExprs2(const std::string& expression, bool anchorAttached) const;
    std::vector<std::string> extractOperatorExpressions(const std::string& expr) const;
    bool checkProhibitedCombinations(const std::string& expression) const;
    bool prohibitedHeadsGood(const std::string& conjecture) const;
    bool countArgumentsFilter(const std::string& conjecture) const;
    bool patternInConjecture(const std::string& conjecture) const;
    bool onlyInHeadGood(const std::string& conjecture) const;
    bool checkConjectureComplexityPerOperator(const std::string& conjecture, const std::string& newExpr) const;
    bool checkMinSizeExpression(const std::string& conjecture) const;
    bool checkInputVariablesTheoremOperatorHead(const std::string& theorem) const;
    bool checkInputVariablesOrder(const std::string& theorem) const;
    bool controlEquality(const std::string& conjecture) const;
    bool checkDefSetsPriorToConnection(const DefSetMap& argsStatement, const DefSetMap& argsGrowingTheorem) const;

    // ---- Sub-functions for checkInputVariablesOrder ----
    std::set<std::string> findDigitArgs(const std::string& theorem) const;
    std::pair<std::set<std::string>, std::set<std::string>>
        getLeftRight(const std::vector<std::string>& chain,
                     const std::string& expression,
                     const std::set<std::string>& digits, int counter,
                     std::set<std::string>& visited) const;
    std::vector<std::string>
        getRightChain(const std::vector<std::string>& chain,
                      const std::string& head, std::set<std::string>& visited) const;
    std::pair<std::vector<std::string>, std::vector<std::string>>
        getLeftRightChains(const std::vector<std::string>& chain) const;
    std::string getOperatorId(const std::string& expr) const;
    bool checkInputVariablePosition(const std::vector<std::string>& chain,
                                    const std::set<std::string>& digits) const;
    std::string removeOutputs(const std::vector<std::string>& chain) const;
    bool checkTautology(const std::vector<std::string>& leftChain, const std::vector<std::string>& rightChain) const;
    bool checkFunctions(const std::vector<std::string>& chain) const;
    bool onlyOneOperator(const std::vector<std::string>& chain) const;
    std::set<std::string> findEntryArgs2(
        const std::vector<std::vector<std::string>>& inputArgsList,
        const std::vector<std::vector<std::string>>& outputArgsList,
        int index, std::set<int>& visited) const;
    std::set<std::string> getTertiaries(const std::vector<std::string>& chain) const;
    bool checkTertiaries(const std::vector<std::string>& leftChain, const std::vector<std::string>& rightChain) const;

    // ---- Reshuffling & mirroring ----
    std::tuple<std::string, DefSetMap, std::map<std::string,std::string>>
        reshuffle(const std::string& expr, bool deep) const;
    std::string createReshuffledMirrored(const std::string& expr, bool anchorFirst = false) const;
    int countOperatorOccurrences(const std::string& s) const;
    bool staysOutputVariable(const std::string& fullExpr, const std::string& headExpr) const;

    // ---- Worker functions ----
    WorkerResult singleThreadCalculation(
        const std::string& statement, const std::string& growingTheorem,
        int nseStatement, int nseGrowingTheorem,
        const DefSetMap& argsStatement, const DefSetMap& argsGrowingTheorem) const;

    WorkerResult singleExprAnchorConnection(
        const std::string& expr, const DefSetMap& exprDefSets) const;

    std::string reformulateOperatorHead(const std::string& conjecture) const;

    // ---- Int-path: encode/decode ----
    void buildNameMap();
    void buildIntExprConfigs();
    IntConjBuf encodeExpr(const std::string& expr) const;
    std::string decodeExpr(const IntConjBuf& buf) const;
    IntDefSetMap encodeDefSetMap(const DefSetMap& dsm) const;
    DefSetMap decodeDefSetMap(const IntDefSetMap& idsm) const;

    // ---- Int-path: connection ----
    bool connectExpressionsInt(
        const IntConjBuf& expr1, const IntConjBuf& expr2,
        const IntDefSetMap& map1, const IntDefSetMap& map2,
        const IntConnMap& subMap,
        const int16_t* binaryList, int binaryLen,
        bool connectToAnchor,
        IntConjBuf& outExpr, IntDefSetMap& outMap) const;

    void makeAllConnectionMapsInt(
        const IntDefSetMap& argsMap1, const IntDefSetMap& argsMap2,
        bool withAnchor, const MappingsMap& mappingsMap,
        std::vector<IntConnMap>& outMaps) const;

    // ---- Int-path: filters ----
    bool repetitionsExistInt(const IntConjBuf& buf) const;
    bool numbersGoodInt(const IntConjBuf& buf) const;
    bool checkDefSetsInt(const IntDefSetMap& argMap) const;
    bool checkComplexityLevelInt(const IntDefSetMap& argMap, int complexityLevel) const;
    bool checkDefSetsPriorInt(const IntDefSetMap& argsStmt, const IntDefSetMap& argsGT) const;
    bool exprGood2Int(const IntConjBuf& buf, int nse, const IntDefSetMap& connMap) const;
    bool onlyInHeadGoodInt(const IntConjBuf& buf) const;
    bool prohibitedHeadsGoodInt(const IntConjBuf& buf) const;
    int getNumberRemovableArgsInt(const IntConnMap& connMap) const;
    void sortByOccurrenceInt(const IntConjBuf& expr, const int16_t* removableArgs, int numRemovable,
                             int16_t* sortedOut) const;
    int countOperatorOccurrencesInt(const IntConjBuf& buf) const;
    bool checkComplexityPerOpInt(const IntConjBuf& growingTheorem, const IntConjBuf& statement) const;

    // ---- Int-path: workers ----
    WorkerResult singleThreadCalculationInt(
        const IntConjBuf& intStatement, const IntConjBuf& intGrowingTheorem,
        int nseStatement, int nseGrowingTheorem,
        const IntDefSetMap& intArgsStatement, const IntDefSetMap& intArgsGrowingTheorem) const;

    WorkerResult singleExprAnchorConnectionInt(
        const IntConjBuf& intExpr, const IntDefSetMap& intExprDefSets) const;

    // ---- Debug ----
    void dumpDebug(const std::string& tag, const std::string& data) const;
};

} // namespace conj
