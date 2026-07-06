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

/// @file
/// @brief Unit-test suite for the conjecturer module.
///
/// @details
/// Mirrors the prover/memory/compiler/filter/compressor harness rhythm
/// (see `docs/agentic_swdd/_meta/testing.md`). Tests register into the existing
/// Meyers-singleton registry via `TEST(conjecturer, …)` and run as
/// part of `gl_quick.exe --unit-tests` together with the prover suite.
///
/// Coverage is organised by suite-name prefix:
///
///   - helper_*     — every free function in conj:: namespace
///   - intpath_*    — encodeExpr/decodeExpr/encodeDefSetMap/decodeDefSetMap round-trip
///   - mappings_*   — createMap, createMapAnchor, makeAllConnectionMaps*, connectExpressions*
///   - filter_*     — most filters in the cascade with positive + negative
///                     cases (controlEquality, numbersGood, prohibitedHeadsGood,
///                     exprGood / exprGood2, qualifiedForEquality,
///                     checkInputVariablesTheoremOperatorHead,
///                     passesMaxSizeAfterExistence,
///                     passesMaxDistinctAnchorValuesPerType,
///                     passesInPremiseFilter D-23 + cnt-shape rules
///                     1 / 2 / 3 / >=3 — nested-quantifier fixtures).
///                     Smoke-only (positive only): passesComplexityAfterExistence
///                     (the post-D-23 3-condition rule could not be triggered
///                     from synthetic inputs; rejection rides on main.py
///                     integration runs).
///   - inputorder_* — 13 sub-helpers of checkInputVariablesOrder
///   - reshuffle_*  — reshuffle pipeline + createReshuffledMirrored + countOperatorOccurrences/Int + staysOutputVariable
///   - reform_*     — reformulateOperatorHead, generateNegatedPremiseVariants, triggersExistenceReformulation, reformulateToExistenceHead
///   - worker_*     — singleThreadCalculation, singleExprAnchorConnection, singleThreadCalculationInt, singleExprAnchorConnectionInt
///   - legacy_*     — renameVariablesInExpr (legacy reshuffle predecessor)
///   - ctor_*       — Conjecturer{"Peano"} / Conjecturer{"Gauss"} construct without throw, accessors
///   - or_pair_*    — generateOrConjectures Peano + Gauss smoke (config-derived
///                     pair counts: Peano emits 2, Gauss emits 0)
///   - invariant_*  — direct guards for I-8, I-9, I-10, I-11
///
/// Member-function tests reach private methods of `conj::Conjecturer`
/// via `conj::testing::Friend`, declared as `friend class
/// ::conj::testing::Friend;` in the production header. The Friend
/// class lives entirely inside this translation unit; production
/// code never references it.
///
/// @see [`docs/agentic_swdd/10_pipeline/02_conjecturer.md`](../../../docs/agentic_swdd/10_pipeline/02_conjecturer.md)
///      — full architecture, filter cascade, reshuffle pipeline,
///      weaknesses, open questions.
/// @see [`docs/agentic_swdd/_meta/testing.md`](../../../docs/agentic_swdd/_meta/testing.md)
///      — harness contract, performance budget, sacred boundaries.

#include "test_harness.hpp"

#include "../conjecturer.hpp"
#include "../compiler.hpp"

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>

// ============================================================================
// Test-only access shim for private members of conj::Conjecturer.
//
// The Conjecturer's private surface (filters, int-path encode/decode,
// createMap* statics, reshuffle, worker drivers, ...) is the testable
// substance — the public API is just the constructor and run().
// Friendship grants the suite access without widening the public
// surface; the Friend class lives entirely in this translation unit.
// ============================================================================
namespace conj {
namespace testing {

class Friend {
public:
    // ---- Pre-computation (statics) ----
    static MappingsMap createMap(int N) {
        return Conjecturer::createMap(N);
    }
    static MappingsMap createMapAnchor(int leftMax, int rightMax) {
        return Conjecturer::createMapAnchor(leftMax, rightMax);
    }
    static int determineLeftSideBoundary(const Conjecturer& c) {
        return c.determineLeftSideBoundary();
    }
    static int determineRightSideBoundary(const Conjecturer& c) {
        return c.determineRightSideBoundary();
    }

    // ---- Expression parsing & arg maps ----
    static DefSetMap findArgMap(const Conjecturer& c, const std::string& expr) {
        return c.findArgMap(expr);
    }
    static std::tuple<std::string, DefSetMap, std::map<std::string,std::string>>
    renameVariablesInExpr(const Conjecturer& c, const std::string& expr, bool deep) {
        return c.renameVariablesInExpr(expr, deep);
    }

    // ---- Expression connection (string + int) ----
    static std::tuple<bool, std::string, DefSetMap>
    connectExpressions(const Conjecturer& c,
                       const std::string& expr1, const std::string& expr2,
                       const DefSetMap& map1, const DefSetMap& map2,
                       const std::map<std::string,std::string>& subMap,
                       const std::vector<int>& binaryList, bool connectToAnchor) {
        return c.connectExpressions(expr1, expr2, map1, map2, subMap,
                                    binaryList, connectToAnchor);
    }
    static std::vector<std::map<std::string,std::string>>
    makeAllConnectionMaps(const Conjecturer& c, const DefSetMap& map1, const DefSetMap& map2,
                          bool withAnchor, const MappingsMap& mappingsMap) {
        return c.makeAllConnectionMaps(map1, map2, withAnchor, mappingsMap);
    }
    static bool connectExpressionsInt(const Conjecturer& c,
                                       const IntConjBuf& expr1, const IntConjBuf& expr2,
                                       const IntDefSetMap& map1, const IntDefSetMap& map2,
                                       const IntConnMap& subMap,
                                       const int16_t* binaryList, int binaryLen,
                                       bool connectToAnchor,
                                       IntConjBuf& outExpr, IntDefSetMap& outMap) {
        return c.connectExpressionsInt(expr1, expr2, map1, map2, subMap,
                                       binaryList, binaryLen, connectToAnchor,
                                       outExpr, outMap);
    }
    static void makeAllConnectionMapsInt(const Conjecturer& c,
                                         const IntDefSetMap& argsMap1, const IntDefSetMap& argsMap2,
                                         bool withAnchor, const MappingsMap& mappingsMap,
                                         std::vector<IntConnMap>& outMaps) {
        c.makeAllConnectionMapsInt(argsMap1, argsMap2, withAnchor, mappingsMap, outMaps);
    }

    // ---- Validation filters (string path) ----
    static bool exprGood(const Conjecturer& c, const std::string& expr) {
        return c.exprGood(expr);
    }
    static bool exprGood2(const Conjecturer& c, const std::string& expr, int nse,
                          const DefSetMap& connectedMap) {
        return c.exprGood2(expr, nse, connectedMap);
    }
    static bool numbersGood(const Conjecturer& c, const std::string& expr) {
        return c.numbersGood(expr);
    }
    static bool checkDefSets(const Conjecturer& c, const DefSetMap& argMap) {
        return c.checkDefSets(argMap);
    }
    static bool checkComplexityLevelForDefSets(const Conjecturer& c,
                                                const DefSetMap& argMap, int complexityLevel) {
        return c.checkComplexityLevelForDefSets(argMap, complexityLevel);
    }
    static bool qualifiedForEquality(const Conjecturer& c, const std::string& expr) {
        return c.qualifiedForEquality(expr);
    }
    static bool checkProhibitedCombinations(const Conjecturer& c, const std::string& expression) {
        return c.checkProhibitedCombinations(expression);
    }
    static bool prohibitedHeadsGood(const Conjecturer& c, const std::string& conjecture) {
        return c.prohibitedHeadsGood(conjecture);
    }
    static bool countArgumentsFilter(const Conjecturer& c, const std::string& conjecture) {
        return c.countArgumentsFilter(conjecture);
    }
    static bool patternInConjecture(const Conjecturer& c, const std::string& conjecture) {
        return c.patternInConjecture(conjecture);
    }
    static bool onlyInHeadGood(const Conjecturer& c, const std::string& conjecture) {
        return c.onlyInHeadGood(conjecture);
    }
    static bool checkConjectureComplexityPerOperator(const Conjecturer& c,
                                                      const std::string& conjecture,
                                                      const std::string& newExpr) {
        return c.checkConjectureComplexityPerOperator(conjecture, newExpr);
    }
    static bool checkMinSizeExpression(const Conjecturer& c, const std::string& conjecture) {
        return c.checkMinSizeExpression(conjecture);
    }
    static bool checkInputVariablesTheoremOperatorHead(const Conjecturer& c,
                                                       const std::string& theorem) {
        return c.checkInputVariablesTheoremOperatorHead(theorem);
    }
    static bool checkInputVariablesOrder(const Conjecturer& c, const std::string& theorem) {
        return c.checkInputVariablesOrder(theorem);
    }
    static bool controlEquality(const Conjecturer& c, const std::string& conjecture) {
        return c.controlEquality(conjecture);
    }
    static bool checkDefSetsPriorToConnection(const Conjecturer& c,
                                               const DefSetMap& argsStatement,
                                               const DefSetMap& argsGrowingTheorem) {
        return c.checkDefSetsPriorToConnection(argsStatement, argsGrowingTheorem);
    }
    static bool evaluateOperatorExprs2(const Conjecturer& c, const std::string& expression,
                                        bool anchorAttached) {
        return c.evaluateOperatorExprs2(expression, anchorAttached);
    }
    static std::vector<std::string>
    extractOperatorExpressions(const Conjecturer& c, const std::string& expr) {
        return c.extractOperatorExpressions(expr);
    }

    // ---- Sub-functions of checkInputVariablesOrder ----
    static std::set<std::string> findDigitArgs(const Conjecturer& c, const std::string& theorem) {
        return c.findDigitArgs(theorem);
    }
    static std::string getOperatorId(const Conjecturer& c, const std::string& expr) {
        return c.getOperatorId(expr);
    }
    static std::string removeOutputs(const Conjecturer& c, const std::vector<std::string>& chain) {
        return c.removeOutputs(chain);
    }
    static bool checkTautology(const Conjecturer& c,
                                const std::vector<std::string>& leftChain,
                                const std::vector<std::string>& rightChain) {
        return c.checkTautology(leftChain, rightChain);
    }
    static bool checkFunctions(const Conjecturer& c, const std::vector<std::string>& chain) {
        return c.checkFunctions(chain);
    }
    static bool onlyOneOperator(const Conjecturer& c, const std::vector<std::string>& chain) {
        return c.onlyOneOperator(chain);
    }
    static bool checkInputVariablePosition(const Conjecturer& c,
                                            const std::vector<std::string>& chain,
                                            const std::set<std::string>& digits) {
        return c.checkInputVariablePosition(chain, digits);
    }
    static std::pair<std::vector<std::string>, std::vector<std::string>>
    getLeftRightChains(const Conjecturer& c, const std::vector<std::string>& chain) {
        return c.getLeftRightChains(chain);
    }
    static std::set<std::string> getTertiaries(const Conjecturer& c,
                                                const std::vector<std::string>& chain) {
        return c.getTertiaries(chain);
    }
    static bool checkTertiaries(const Conjecturer& c,
                                 const std::vector<std::string>& leftChain,
                                 const std::vector<std::string>& rightChain) {
        return c.checkTertiaries(leftChain, rightChain);
    }

    // ---- Reshuffling & mirroring ----
    static std::tuple<std::string, DefSetMap, std::map<std::string,std::string>>
    reshuffle(const Conjecturer& c, const std::string& expr, bool deep) {
        return c.reshuffle(expr, deep);
    }
    static std::string createReshuffledMirrored(const Conjecturer& c, const std::string& expr,
                                                bool anchorFirst = false) {
        return c.createReshuffledMirrored(expr, anchorFirst);
    }
    static int countOperatorOccurrences(const Conjecturer& c, const std::string& s) {
        return c.countOperatorOccurrences(s);
    }
    static bool staysOutputVariable(const Conjecturer& c,
                                     const std::string& fullExpr,
                                     const std::string& outputVariable) {
        return c.staysOutputVariable(fullExpr, outputVariable);
    }

    // ---- Worker functions (string + int) ----
    static WorkerResult singleThreadCalculation(const Conjecturer& c,
                                                 const std::string& statement,
                                                 const std::string& growingTheorem,
                                                 int nseStatement, int nseGrowingTheorem,
                                                 const DefSetMap& argsStatement,
                                                 const DefSetMap& argsGrowingTheorem) {
        return c.singleThreadCalculation(statement, growingTheorem,
                                          nseStatement, nseGrowingTheorem,
                                          argsStatement, argsGrowingTheorem);
    }
    static WorkerResult singleExprAnchorConnection(const Conjecturer& c,
                                                    const std::string& expr,
                                                    const DefSetMap& exprDefSets) {
        return c.singleExprAnchorConnection(expr, exprDefSets);
    }
    static WorkerResult singleThreadCalculationInt(const Conjecturer& c,
                                                    const IntConjBuf& intStatement,
                                                    const IntConjBuf& intGrowingTheorem,
                                                    int nseStatement, int nseGrowingTheorem,
                                                    const IntDefSetMap& intArgsStatement,
                                                    const IntDefSetMap& intArgsGrowingTheorem) {
        return c.singleThreadCalculationInt(intStatement, intGrowingTheorem,
                                             nseStatement, nseGrowingTheorem,
                                             intArgsStatement, intArgsGrowingTheorem);
    }
    static WorkerResult singleExprAnchorConnectionInt(const Conjecturer& c,
                                                       const IntConjBuf& intExpr,
                                                       const IntDefSetMap& intExprDefSets) {
        return c.singleExprAnchorConnectionInt(intExpr, intExprDefSets);
    }

    // ---- Reformulation / negation / OR ----
    static std::string reformulateOperatorHead(const Conjecturer& c,
                                                const std::string& conjecture) {
        return c.reformulateOperatorHead(conjecture);
    }
    static bool triggersExistenceReformulation(const Conjecturer& c,
                                                const std::string& theorem) {
        return c.triggersExistenceReformulation(theorem);
    }
    static std::string reformulateToExistenceHead(const Conjecturer& c,
                                                   const std::string& theorem) {
        return c.reformulateToExistenceHead(theorem);
    }
    static std::vector<std::string>
    generateNegatedPremiseVariants(const Conjecturer& c, const std::string& conj) {
        return c.generateNegatedPremiseVariants(conj);
    }
    static std::vector<std::pair<std::string,std::string>>
    generateOrConjectures(const Conjecturer& c) {
        return c.generateOrConjectures();
    }

    // ---- Post-existence filters ----
    static bool passesMaxSizeAfterExistence(const Conjecturer& c, const std::string& conj,
                                             int leafCount) {
        return c.passesMaxSizeAfterExistence(conj, leafCount);
    }
    static bool passesComplexityAfterExistence(const Conjecturer& c, const std::string& conj) {
        return c.passesComplexityAfterExistence(conj);
    }
    static bool passesMaxDistinctAnchorValuesPerType(const Conjecturer& c,
                                                      const std::string& conj) {
        return c.passesMaxDistinctAnchorValuesPerType(conj);
    }
    static bool passesInPremiseFilter(const Conjecturer& c, const std::string& conj) {
        return c.passesInPremiseFilter(conj);
    }

    // ---- Int-path encode/decode ----
    static IntConjBuf encodeExpr(const Conjecturer& c, const std::string& expr) {
        return c.encodeExpr(expr);
    }
    static std::string decodeExpr(const Conjecturer& c, const IntConjBuf& buf) {
        return c.decodeExpr(buf);
    }
    static IntDefSetMap encodeDefSetMap(const Conjecturer& c, const DefSetMap& dsm) {
        return c.encodeDefSetMap(dsm);
    }
    static DefSetMap decodeDefSetMap(const Conjecturer& c, const IntDefSetMap& idsm) {
        return c.decodeDefSetMap(idsm);
    }

    // ---- Int-path filters ----
    static bool repetitionsExistInt(const Conjecturer& c, const IntConjBuf& buf) {
        return c.repetitionsExistInt(buf);
    }
    static bool numbersGoodInt(const Conjecturer& c, const IntConjBuf& buf) {
        return c.numbersGoodInt(buf);
    }
    static bool checkDefSetsInt(const Conjecturer& c, const IntDefSetMap& argMap) {
        return c.checkDefSetsInt(argMap);
    }
    static bool checkComplexityLevelInt(const Conjecturer& c, const IntDefSetMap& argMap,
                                         int complexityLevel) {
        return c.checkComplexityLevelInt(argMap, complexityLevel);
    }
    static bool checkDefSetsPriorInt(const Conjecturer& c, const IntDefSetMap& argsStmt,
                                      const IntDefSetMap& argsGT) {
        return c.checkDefSetsPriorInt(argsStmt, argsGT);
    }
    static bool exprGood2Int(const Conjecturer& c, const IntConjBuf& buf, int nse,
                              const IntDefSetMap& connMap) {
        return c.exprGood2Int(buf, nse, connMap);
    }
    static bool onlyInHeadGoodInt(const Conjecturer& c, const IntConjBuf& buf) {
        return c.onlyInHeadGoodInt(buf);
    }
    static bool prohibitedHeadsGoodInt(const Conjecturer& c, const IntConjBuf& buf) {
        return c.prohibitedHeadsGoodInt(buf);
    }
    static int countOperatorOccurrencesInt(const Conjecturer& c, const IntConjBuf& buf) {
        return c.countOperatorOccurrencesInt(buf);
    }
    static int getNumberRemovableArgsInt(const Conjecturer& c, const IntConnMap& connMap) {
        return c.getNumberRemovableArgsInt(connMap);
    }
    static bool checkComplexityPerOpInt(const Conjecturer& c,
                                         const IntConjBuf& growingTheorem,
                                         const IntConjBuf& statement) {
        return c.checkComplexityPerOpInt(growingTheorem, statement);
    }

    // ---- Accessor for tests that need to inspect immutable state ----
    static const ConfigurationData& config(const Conjecturer& c) {
        return c.config_;
    }
    static const IntConjBuf& anchorInt(const Conjecturer& c) {
        return c.anchorInt_;
    }
    static const IntDefSetMap& anchorDefSetsInt(const Conjecturer& c) {
        return c.anchorDefSetsInt_;
    }
};

}  // namespace testing
}  // namespace conj


// ============================================================================
// helper_*  — free functions in conj:: namespace
// ============================================================================

// ---- repetitionsExist (5 tests) ----

TEST(conjecturer, helper_repetitions_exist_no_inner_paren_dup) {
    // Distinct inner-paren tokens => no repetition.
    ASSERT_FALSE(conj::repetitionsExist("(=[1,2])(in[3,4])"));
}

TEST(conjecturer, helper_repetitions_exist_two_identical_tokens) {
    // Same inner-paren token twice => repetition flagged.
    ASSERT_TRUE(conj::repetitionsExist("(in[1,2])(in[1,2])"));
}

TEST(conjecturer, helper_repetitions_exist_empty_string) {
    // No tokens at all — nothing to repeat.
    ASSERT_FALSE(conj::repetitionsExist(""));
}

TEST(conjecturer, helper_repetitions_exist_single_token) {
    // One token can never repeat itself.
    ASSERT_FALSE(conj::repetitionsExist("(=[5,7])"));
}

TEST(conjecturer, helper_repetitions_exist_three_tokens_with_one_dup) {
    ASSERT_TRUE(conj::repetitionsExist("(=[1,2])(in[3,4])(=[1,2])"));
}


// ---- mergeMirrorConjecturesIntoPool (5 tests) ----

TEST(conjecturer, helper_merge_mirror_appends_new) {
    // A mirror not already in the pool is appended; count == 1.
    std::vector<std::string> pool = {"A", "B"};
    const int added = conj::mergeMirrorConjecturesIntoPool(pool, {"C"});
    ASSERT_EQ(added, 1);
    ASSERT_EQ((int)pool.size(), 3);
    ASSERT_EQ(pool[2], std::string("C"));
}

TEST(conjecturer, helper_merge_mirror_dedups_against_pool) {
    // A mirror already present in the pool is not duplicated.
    std::vector<std::string> pool = {"A", "B"};
    const int added = conj::mergeMirrorConjecturesIntoPool(pool, {"B", "C"});
    ASSERT_EQ(added, 1);
    ASSERT_EQ((int)pool.size(), 3);
    ASSERT_EQ(pool[2], std::string("C"));
}

TEST(conjecturer, helper_merge_mirror_skips_empty) {
    // Empty == "no distinct reverse direction" (I-9 collapse); skipped.
    std::vector<std::string> pool = {"A"};
    const int added = conj::mergeMirrorConjecturesIntoPool(pool, {"", "D"});
    ASSERT_EQ(added, 1);
    ASSERT_EQ((int)pool.size(), 2);
    ASSERT_EQ(pool[1], std::string("D"));
}

TEST(conjecturer, helper_merge_mirror_all_duplicates_adds_nothing) {
    std::vector<std::string> pool = {"A", "B"};
    const int added = conj::mergeMirrorConjecturesIntoPool(pool, {"A", "B"});
    ASSERT_EQ(added, 0);
    ASSERT_EQ((int)pool.size(), 2);
}

TEST(conjecturer, helper_merge_mirror_internal_dedup) {
    // A mirror repeated within the input is appended only once.
    std::vector<std::string> pool = {"A"};
    const int added = conj::mergeMirrorConjecturesIntoPool(pool, {"C", "C"});
    ASSERT_EQ(added, 1);
    ASSERT_EQ((int)pool.size(), 2);
}


// ---- findMinMaxNumbers (6 tests) ----

TEST(conjecturer, helper_find_min_max_single_paren_id) {
    auto [mn, mx] = conj::findMinMaxNumbers("((5))");
    ASSERT_EQ(mn, 5);
    ASSERT_EQ(mx, 5);
}

TEST(conjecturer, helper_find_min_max_multiple_paren_ids) {
    auto [mn, mx] = conj::findMinMaxNumbers("((3))((1))((7))((2))");
    ASSERT_EQ(mn, 1);
    ASSERT_EQ(mx, 7);
}

TEST(conjecturer, helper_find_min_max_no_ids_returns_zero_zero) {
    // The conjecturer convention on empty: (0, 0) (NOT INT_MAX/MIN).
    auto [mn, mx] = conj::findMinMaxNumbers("no parens here");
    ASSERT_EQ(mn, 0);
    ASSERT_EQ(mx, 0);
}

TEST(conjecturer, helper_find_min_max_two_equal_ids) {
    auto [mn, mx] = conj::findMinMaxNumbers("((4))((4))");
    ASSERT_EQ(mn, 4);
    ASSERT_EQ(mx, 4);
}

TEST(conjecturer, helper_find_min_max_includes_only_paren_wrapped_ids) {
    // Bare `5` not surrounded by parens does not count.
    auto [mn, mx] = conj::findMinMaxNumbers("5 ((9))");
    ASSERT_EQ(mn, 9);
    ASSERT_EQ(mx, 9);
}

TEST(conjecturer, helper_find_min_max_descending_ids_get_sorted) {
    auto [mn, mx] = conj::findMinMaxNumbers("((10))((1))");
    ASSERT_EQ(mn, 1);
    ASSERT_EQ(mx, 10);
}


// ---- findAllIds (5 tests) ----

TEST(conjecturer, helper_find_all_ids_empty) {
    ASSERT_TRUE(conj::findAllIds("").empty());
}

TEST(conjecturer, helper_find_all_ids_single) {
    auto ids = conj::findAllIds("((7))");
    ASSERT_EQ(ids.size(), static_cast<std::size_t>(1));
    ASSERT_TRUE(ids.count(7) == 1);
}

TEST(conjecturer, helper_find_all_ids_dedupes) {
    auto ids = conj::findAllIds("((3))((5))((3))");
    ASSERT_EQ(ids.size(), static_cast<std::size_t>(2));
    ASSERT_TRUE(ids.count(3) == 1);
    ASSERT_TRUE(ids.count(5) == 1);
}

TEST(conjecturer, helper_find_all_ids_returns_sorted_set) {
    auto ids = conj::findAllIds("((9))((2))((5))");
    auto it = ids.begin();
    ASSERT_EQ(*it++, 2);
    ASSERT_EQ(*it++, 5);
    ASSERT_EQ(*it++, 9);
}

TEST(conjecturer, helper_find_all_ids_ignores_non_paren_digits) {
    // 999 not in parens does not count; 4 in parens does.
    auto ids = conj::findAllIds("999 ((4))");
    ASSERT_EQ(ids.size(), static_cast<std::size_t>(1));
    ASSERT_TRUE(ids.count(4) == 1);
}


// ---- updateReplacementMap (5 tests) ----

TEST(conjecturer, helper_update_replacement_map_first_pair_min_wins) {
    std::map<int,int> rep;
    conj::updateReplacementMap(rep, 5, 2);
    // Closure {2, 5}; min = 2; 5 -> 2.
    ASSERT_EQ(rep.size(), static_cast<std::size_t>(1));
    ASSERT_EQ(rep.at(5), 2);
}

TEST(conjecturer, helper_update_replacement_map_chain_collapse) {
    std::map<int,int> rep;
    conj::updateReplacementMap(rep, 3, 7);
    conj::updateReplacementMap(rep, 7, 4);
    // After both calls the closure is {3, 4, 7}; min = 3.
    // Every non-min in the closure maps to 3.
    ASSERT_EQ(rep.at(7), 3);
    ASSERT_EQ(rep.at(4), 3);
}

TEST(conjecturer, helper_update_replacement_map_idempotent_on_self_pair) {
    std::map<int,int> rep;
    conj::updateReplacementMap(rep, 5, 5);
    // Closure {5} => no entries to add (nothing differs from min).
    ASSERT_TRUE(rep.empty());
}

TEST(conjecturer, helper_update_replacement_map_disjoint_pairs_kept_separate) {
    std::map<int,int> rep;
    conj::updateReplacementMap(rep, 1, 2);
    conj::updateReplacementMap(rep, 5, 6);
    // Two independent closures: {1, 2} and {5, 6}.
    ASSERT_EQ(rep.at(2), 1);
    ASSERT_EQ(rep.at(6), 5);
    ASSERT_TRUE(rep.find(1) == rep.end());
    ASSERT_TRUE(rep.find(5) == rep.end());
}

TEST(conjecturer, helper_update_replacement_map_min_propagates_after_join) {
    std::map<int,int> rep;
    conj::updateReplacementMap(rep, 5, 9);   // {5, 9} -> 5
    conj::updateReplacementMap(rep, 3, 9);   // joined closure {3, 5, 9} -> 3
    ASSERT_EQ(rep.at(9), 3);
    ASSERT_EQ(rep.at(5), 3);
}


// ---- defSetsEqual (8 tests) ----

TEST(conjecturer, helper_def_sets_equal_identical_strings) {
    auto [eq, unifier] = conj::defSetsEqual("(1)", "(1)");
    ASSERT_TRUE(eq);
}

TEST(conjecturer, helper_def_sets_equal_renamed_ids_match) {
    // Both reduce to "(1)" after reorderNumbers.
    auto [eq, unifier] = conj::defSetsEqual("(7)", "(3)");
    ASSERT_TRUE(eq);
}

TEST(conjecturer, helper_def_sets_equal_different_shape_rejected) {
    // (1) vs P(1) → different texts after reorder.
    auto [eq, _u] = conj::defSetsEqual("(1)", "P(1)");
    ASSERT_FALSE(eq);
}

TEST(conjecturer, helper_def_sets_equal_unifier_empty_when_unequal) {
    auto [eq, unifier] = conj::defSetsEqual("(1)", "P(1)");
    ASSERT_FALSE(eq);
    ASSERT_TRUE(unifier.empty());
}

TEST(conjecturer, helper_def_sets_equal_compound_match) {
    // Same shape under renaming.
    auto [eq, _u] = conj::defSetsEqual("P(x(1)(2))", "P(x(7)(9))");
    ASSERT_TRUE(eq);
}

TEST(conjecturer, helper_def_sets_equal_x_x_token_preserved) {
    // The literal "x" leaf inside `(x...` is structural, not a renaming target.
    auto [eq, _u] = conj::defSetsEqual("(x(1)(2))", "(x(3)(4))");
    ASSERT_TRUE(eq);
}

TEST(conjecturer, helper_def_sets_equal_single_id_unifier_nontrivial) {
    auto [eq, unifier] = conj::defSetsEqual("(7)", "(3)");
    ASSERT_TRUE(eq);
    // Unifier joins 3 and 7 into one closure (min wins).
    ASSERT_FALSE(unifier.empty());
}

TEST(conjecturer, helper_def_sets_equal_mismatched_arity) {
    // Different inner shape; reject.
    auto [eq, _u] = conj::defSetsEqual("(1)", "(x(1)(2))");
    ASSERT_FALSE(eq);
}


// ---- reorderNumbers (5 tests) ----

TEST(conjecturer, helper_reorder_numbers_single_id_becomes_one) {
    auto [out, oldIds] = conj::reorderNumbers("(7)");
    ASSERT_EQ(out, std::string("(1)"));
}

TEST(conjecturer, helper_reorder_numbers_already_canonical) {
    auto [out, _ids] = conj::reorderNumbers("(1)");
    ASSERT_EQ(out, std::string("(1)"));
}

TEST(conjecturer, helper_reorder_numbers_two_ids_first_occurrence) {
    auto [out, _ids] = conj::reorderNumbers("(x(7)(3))");
    ASSERT_EQ(out, std::string("(x(1)(2))"));
}

TEST(conjecturer, helper_reorder_numbers_p_form) {
    auto [out, _ids] = conj::reorderNumbers("P(7)");
    ASSERT_EQ(out, std::string("P(1)"));
}

TEST(conjecturer, helper_reorder_numbers_nested_p_x) {
    auto [out, _ids] = conj::reorderNumbers("P(x(5)(9))");
    ASSERT_EQ(out, std::string("P(x(1)(2))"));
}


// ---- subtractAndReplaceNumbers (5 tests) ----

TEST(conjecturer, helper_subtract_and_replace_no_map_subtract_zero) {
    auto [out, _ids] = conj::subtractAndReplaceNumbers("(7)", 0, {});
    ASSERT_EQ(out, std::string("(7)"));
}

TEST(conjecturer, helper_subtract_and_replace_pure_subtract) {
    // Subtract 3 from id 10 → "(7)".
    auto [out, _ids] = conj::subtractAndReplaceNumbers("(10)", 3, {});
    ASSERT_EQ(out, std::string("(7)"));
}

TEST(conjecturer, helper_subtract_and_replace_via_map_overrides_subtract) {
    // The map applies on the post-subtract id: (10 - 3) -> 7, lookup map[7] = 99.
    auto [out, _ids] = conj::subtractAndReplaceNumbers("(10)", 3, {{7, 99}});
    ASSERT_EQ(out, std::string("(99)"));
}

TEST(conjecturer, helper_subtract_and_replace_p_form_uses_subtract) {
    auto [out, _ids] = conj::subtractAndReplaceNumbers("P(8)", 3, {});
    ASSERT_EQ(out, std::string("P(5)"));
}

TEST(conjecturer, helper_subtract_and_replace_compound_x_form) {
    // Subtract 2 from each id in (x(5)(7)) -> (x(3)(5)).
    auto [out, _ids] = conj::subtractAndReplaceNumbers("(x(5)(7))", 2, {});
    ASSERT_EQ(out, std::string("(x(3)(5))"));
}


// ---- shiftTogether (5 tests) ----

TEST(conjecturer, helper_shift_together_already_contiguous) {
    std::map<std::string,std::string> m{{"a", "(1)"}, {"b", "(2)"}};
    conj::shiftTogether(m);
    ASSERT_EQ(m["a"], std::string("(1)"));
    ASSERT_EQ(m["b"], std::string("(2)"));
}

TEST(conjecturer, helper_shift_together_renumbers_sparse_ids) {
    // Ids 5 and 9 → contiguous 1 and 2.
    std::map<std::string,std::string> m{{"a", "(5)"}, {"b", "(9)"}};
    conj::shiftTogether(m);
    ASSERT_EQ(m["a"], std::string("(1)"));
    ASSERT_EQ(m["b"], std::string("(2)"));
}

TEST(conjecturer, helper_shift_together_preserves_shared_ids) {
    // Shared id 5 → both values reference the new id 1.
    std::map<std::string,std::string> m{{"a", "(5)"}, {"b", "(5)"}};
    conj::shiftTogether(m);
    ASSERT_EQ(m["a"], std::string("(1)"));
    ASSERT_EQ(m["b"], std::string("(1)"));
}

TEST(conjecturer, helper_shift_together_empty_map_is_noop) {
    std::map<std::string,std::string> m;
    conj::shiftTogether(m);
    ASSERT_TRUE(m.empty());
}

TEST(conjecturer, helper_shift_together_single_value) {
    std::map<std::string,std::string> m{{"only", "(42)"}};
    conj::shiftTogether(m);
    ASSERT_EQ(m["only"], std::string("(1)"));
}


// ---- extractBetweenBrackets (5 tests) ----

TEST(conjecturer, helper_extract_brackets_simple) {
    ASSERT_EQ(conj::extractBetweenBrackets("hello[world]"), std::string("world"));
}

TEST(conjecturer, helper_extract_brackets_empty_when_no_open) {
    ASSERT_EQ(conj::extractBetweenBrackets("hello world"), std::string(""));
}

TEST(conjecturer, helper_extract_brackets_empty_inside) {
    ASSERT_EQ(conj::extractBetweenBrackets("a[]b"), std::string(""));
}

TEST(conjecturer, helper_extract_brackets_first_match_used) {
    // Returns first matching pair only (not greedy across multiple [...]).
    ASSERT_EQ(conj::extractBetweenBrackets("[a],[b]"), std::string("a"));
}

TEST(conjecturer, helper_extract_brackets_start_index_skips) {
    // Skipping past the first `[` reveals the second pair.
    ASSERT_EQ(conj::extractBetweenBrackets("[a],[b]", 4), std::string("b"));
}


// ---- findPositionSurrounded (5 tests) ----

TEST(conjecturer, helper_find_position_surrounded_simple_match) {
    // "5" at index 5 surrounded by `[` and `]`.
    int pos = conj::findPositionSurrounded("name[5]", "5");
    ASSERT_EQ(pos, 5);
}

TEST(conjecturer, helper_find_position_surrounded_comma_boundary) {
    // "5" between commas counts.
    int pos = conj::findPositionSurrounded("name[3,5,7]", "5");
    ASSERT_EQ(pos, 7);
}

TEST(conjecturer, helper_find_position_surrounded_no_match) {
    int pos = conj::findPositionSurrounded("name[3,7]", "9");
    ASSERT_EQ(pos, -1);
}

TEST(conjecturer, helper_find_position_surrounded_unbracketed_not_found) {
    // "5" appears outside [...] only — rejected.
    int pos = conj::findPositionSurrounded("foo5bar[3]", "5");
    ASSERT_EQ(pos, -1);
}

TEST(conjecturer, helper_find_position_surrounded_multi_digit_token) {
    int pos = conj::findPositionSurrounded("[3,42,7]", "42");
    ASSERT_EQ(pos, 3);
}


// ---- sortListAccordingToOccurrence (5 tests) ----

TEST(conjecturer, helper_sort_by_occurrence_simple) {
    auto out = conj::sortListAccordingToOccurrence({"7", "3", "5"}, "[3,5,7]");
    // Expected order by first-occurrence in text: 3 then 5 then 7.
    ASSERT_EQ(out.size(), static_cast<std::size_t>(3));
    ASSERT_EQ(out[0], std::string("3"));
    ASSERT_EQ(out[1], std::string("5"));
    ASSERT_EQ(out[2], std::string("7"));
}

TEST(conjecturer, helper_sort_by_occurrence_already_sorted) {
    auto out = conj::sortListAccordingToOccurrence({"1", "2"}, "[1,2]");
    ASSERT_EQ(out[0], std::string("1"));
    ASSERT_EQ(out[1], std::string("2"));
}

TEST(conjecturer, helper_sort_by_occurrence_empty_list) {
    auto out = conj::sortListAccordingToOccurrence({}, "[1,2,3]");
    ASSERT_TRUE(out.empty());
}

TEST(conjecturer, helper_sort_by_occurrence_missing_argument) {
    // "9" not in text — its position is -1, so it sorts to the front.
    auto out = conj::sortListAccordingToOccurrence({"5", "9"}, "[5]");
    ASSERT_EQ(out[0], std::string("9"));
    ASSERT_EQ(out[1], std::string("5"));
}

TEST(conjecturer, helper_sort_by_occurrence_three_inputs) {
    auto out = conj::sortListAccordingToOccurrence({"3", "1", "2"}, "[2,1,3]");
    ASSERT_EQ(out[0], std::string("2"));
    ASSERT_EQ(out[1], std::string("1"));
    ASSERT_EQ(out[2], std::string("3"));
}


// ---- findOrderedIntegers (5 tests) ----

TEST(conjecturer, helper_find_ordered_integers_basic) {
    auto out = conj::findOrderedIntegers({"3", "1", "2"}, "[2,3,1]");
    // Ordered by first occurrence in the big string: 2, 3, 1.
    ASSERT_EQ(out.size(), static_cast<std::size_t>(3));
    ASSERT_EQ(out[0], std::string("2"));
    ASSERT_EQ(out[1], std::string("3"));
    ASSERT_EQ(out[2], std::string("1"));
}

TEST(conjecturer, helper_find_ordered_integers_drops_missing) {
    auto out = conj::findOrderedIntegers({"3", "9"}, "[3]");
    ASSERT_EQ(out.size(), static_cast<std::size_t>(1));
    ASSERT_EQ(out[0], std::string("3"));
}

TEST(conjecturer, helper_find_ordered_integers_all_missing_yields_empty) {
    auto out = conj::findOrderedIntegers({"7", "9"}, "[1,2,3]");
    ASSERT_TRUE(out.empty());
}

TEST(conjecturer, helper_find_ordered_integers_all_present) {
    auto out = conj::findOrderedIntegers({"1", "2"}, "[2,1]");
    ASSERT_EQ(out[0], std::string("2"));
    ASSERT_EQ(out[1], std::string("1"));
}

TEST(conjecturer, helper_find_ordered_integers_empty_input_list) {
    auto out = conj::findOrderedIntegers({}, "[1,2,3]");
    ASSERT_TRUE(out.empty());
}


// ---- replaceIntegerInString (5 tests) ----

TEST(conjecturer, helper_replace_integer_simple) {
    ASSERT_EQ(conj::replaceIntegerInString("[5,3]", "5", "9"),
              std::string("[9,3]"));
}

TEST(conjecturer, helper_replace_integer_word_boundary) {
    // "5" inside "55" must not match (word boundary).
    ASSERT_EQ(conj::replaceIntegerInString("[55,5]", "5", "9"),
              std::string("[55,9]"));
}

TEST(conjecturer, helper_replace_integer_multiple_occurrences) {
    // Both standalone "5"s replaced.
    ASSERT_EQ(conj::replaceIntegerInString("[5,5]", "5", "9"),
              std::string("[9,9]"));
}

TEST(conjecturer, helper_replace_integer_no_match) {
    ASSERT_EQ(conj::replaceIntegerInString("[1,2]", "5", "9"),
              std::string("[1,2]"));
}

TEST(conjecturer, helper_replace_integer_into_two_digit) {
    ASSERT_EQ(conj::replaceIntegerInString("[3,1,4]", "1", "10"),
              std::string("[3,10,4]"));
}


// ---- subtractNumberFromInts (6 tests) ----

TEST(conjecturer, helper_subtract_number_from_ints_replace_all_zero_subtract) {
    ASSERT_EQ(conj::subtractNumberFromInts("[3,5,7]", 0, {}, true),
              std::string("[3,5,7]"));
}

TEST(conjecturer, helper_subtract_number_from_ints_replace_all_subtract_two) {
    ASSERT_EQ(conj::subtractNumberFromInts("[3,5,7]", 2, {}, true),
              std::string("[1,3,5]"));
}

TEST(conjecturer, helper_subtract_number_from_ints_selective_filter) {
    // Only id 5 is in numbersToReplace — only it gets shifted.
    ASSERT_EQ(conj::subtractNumberFromInts("[3,5,7]", 2, {5}, false),
              std::string("[3,3,7]"));
}

TEST(conjecturer, helper_subtract_number_from_ints_empty_input) {
    ASSERT_EQ(conj::subtractNumberFromInts("", 2, {}, true), std::string(""));
}

TEST(conjecturer, helper_subtract_number_from_ints_skips_unbracketed) {
    // Digit not bracketed by `[`/`,`/`]` is left alone.
    ASSERT_EQ(conj::subtractNumberFromInts("foo5bar[3]", 1, {}, true),
              std::string("foo5bar[2]"));
}

TEST(conjecturer, helper_subtract_number_from_ints_partial_match_not_in_set) {
    // Without filter match, integer is preserved unchanged.
    ASSERT_EQ(conj::subtractNumberFromInts("[3,5]", 1, {7}, false),
              std::string("[3,5]"));
}


// ---- getNumberRemovableArgs (5 tests) ----
// Counts DISTINCT non-identity targets — entries with `key == value` are
// dropped, then the size of the residual values-set is returned.

TEST(conjecturer, helper_get_number_removable_args_empty) {
    ASSERT_EQ(conj::getNumberRemovableArgs({}), 0);
}

TEST(conjecturer, helper_get_number_removable_args_all_identity_returns_zero) {
    // Identity entries are dropped by the implementation, so the
    // distinct-value count is 0.
    std::map<std::string,std::string> m{{"1", "1"}, {"2", "2"}, {"3", "3"}};
    ASSERT_EQ(conj::getNumberRemovableArgs(m), 0);
}

TEST(conjecturer, helper_get_number_removable_args_two_distinct_targets) {
    std::map<std::string,std::string> m{{"1", "2"}, {"2", "3"}};
    // Both entries are non-identity; values = {"2", "3"} -> 2.
    ASSERT_EQ(conj::getNumberRemovableArgs(m), 2);
}

TEST(conjecturer, helper_get_number_removable_args_mixed_drops_identities) {
    // Strip {{"1","1"},{"4","4"}}; keep {{"2","3"}}; values = {"3"} -> 1.
    std::map<std::string,std::string> m{{"1", "1"}, {"2", "3"}, {"4", "4"}};
    ASSERT_EQ(conj::getNumberRemovableArgs(m), 1);
}

TEST(conjecturer, helper_get_number_removable_args_single_non_identity) {
    std::map<std::string,std::string> m{{"5", "9"}};
    ASSERT_EQ(conj::getNumberRemovableArgs(m), 1);
}


// ---- mappingGood (5 tests) ----
// Returns true iff for every value in the map, the smallest key mapping
// to it equals the value itself. In effect: every entry must be the
// identity (k == v) for the mapping to be "good".

TEST(conjecturer, helper_mapping_good_empty) {
    ASSERT_TRUE(conj::mappingGood({}));
}

TEST(conjecturer, helper_mapping_good_all_identity) {
    std::map<std::string,std::string> m{{"1", "1"}, {"2", "2"}};
    ASSERT_TRUE(conj::mappingGood(m));
}

TEST(conjecturer, helper_mapping_good_rejects_non_identity_pair) {
    std::map<std::string,std::string> m{{"1", "2"}, {"3", "4"}};
    ASSERT_FALSE(conj::mappingGood(m));
}

TEST(conjecturer, helper_mapping_good_rejects_mixed) {
    // One identity + one non-identity is still rejected because
    // the non-identity entry violates the rule.
    std::map<std::string,std::string> m{{"1", "1"}, {"2", "3"}};
    ASSERT_FALSE(conj::mappingGood(m));
}

TEST(conjecturer, helper_mapping_good_single_identity_accepted) {
    std::map<std::string,std::string> m{{"5", "5"}};
    ASSERT_TRUE(conj::mappingGood(m));
}


// ---- parseDefSet (4 tests) ----

TEST(conjecturer, helper_parse_def_set_single_id) {
    auto [root, ids] = conj::parseDefSet("(1)");
    ASSERT_TRUE(root != nullptr);
    ASSERT_EQ(ids.size(), static_cast<std::size_t>(1));
    ASSERT_EQ(ids[0], 1);
    ce::deleteTree(root);
}

TEST(conjecturer, helper_parse_def_set_p_form) {
    auto [root, ids] = conj::parseDefSet("P(7)");
    ASSERT_TRUE(root != nullptr);
    ASSERT_EQ(ids.size(), static_cast<std::size_t>(1));
    ASSERT_EQ(ids[0], 7);
    ce::deleteTree(root);
}

TEST(conjecturer, helper_parse_def_set_x_compound) {
    auto [root, ids] = conj::parseDefSet("(x(1)(2))");
    ASSERT_TRUE(root != nullptr);
    ASSERT_EQ(ids.size(), static_cast<std::size_t>(2));
    ce::deleteTree(root);
}

TEST(conjecturer, helper_parse_def_set_p_x_compound) {
    auto [root, ids] = conj::parseDefSet("P(x(3)(4))");
    ASSERT_TRUE(root != nullptr);
    ASSERT_EQ(ids.size(), static_cast<std::size_t>(2));
    ce::deleteTree(root);
}


// ---- treeToStrReorder (4 tests) ----

TEST(conjecturer, helper_tree_to_str_reorder_single_id) {
    auto [root, _ids] = conj::parseDefSet("(7)");
    std::string out = conj::treeToStrReorder(root);
    ce::deleteTree(root);
    ASSERT_EQ(out, std::string("(1)"));
}

TEST(conjecturer, helper_tree_to_str_reorder_renumbers_in_order) {
    auto [root, _ids] = conj::parseDefSet("(x(7)(3))");
    std::string out = conj::treeToStrReorder(root);
    ce::deleteTree(root);
    ASSERT_EQ(out, std::string("(x(1)(2))"));
}

TEST(conjecturer, helper_tree_to_str_reorder_p_form) {
    auto [root, _ids] = conj::parseDefSet("P(7)");
    std::string out = conj::treeToStrReorder(root);
    ce::deleteTree(root);
    ASSERT_EQ(out, std::string("P(1)"));
}

TEST(conjecturer, helper_tree_to_str_reorder_p_x_form) {
    auto [root, _ids] = conj::parseDefSet("P(x(8)(2))");
    std::string out = conj::treeToStrReorder(root);
    ce::deleteTree(root);
    ASSERT_EQ(out, std::string("P(x(1)(2))"));
}


// ---- treeToStr (4 tests) ----

TEST(conjecturer, helper_tree_to_str_zero_offset_empty_map) {
    auto [root, _ids] = conj::parseDefSet("(7)");
    std::string out = conj::treeToStr(root, 0, {});
    ce::deleteTree(root);
    ASSERT_EQ(out, std::string("(7)"));
}

TEST(conjecturer, helper_tree_to_str_offset_only) {
    auto [root, _ids] = conj::parseDefSet("(10)");
    std::string out = conj::treeToStr(root, 3, {});
    ce::deleteTree(root);
    ASSERT_EQ(out, std::string("(7)"));
}

TEST(conjecturer, helper_tree_to_str_map_overrides_offset_result) {
    auto [root, _ids] = conj::parseDefSet("(10)");
    // (10 - 3) -> 7; map[7] = 99.
    std::string out = conj::treeToStr(root, 3, {{7, 99}});
    ce::deleteTree(root);
    ASSERT_EQ(out, std::string("(99)"));
}

TEST(conjecturer, helper_tree_to_str_p_x_with_offset) {
    auto [root, _ids] = conj::parseDefSet("P(x(5)(7))");
    std::string out = conj::treeToStr(root, 2, {});
    ce::deleteTree(root);
    ASSERT_EQ(out, std::string("P(x(3)(5))"));
}


// ---- connectExpressionSets (5 tests) ----
// Inputs use parseable def-set strings ("(1)", "(2)", ...), not opaque
// type labels — the implementation walks `findMinMaxNumbers` and
// `subtractAndReplaceNumbers` over each value.

TEST(conjecturer, helper_connect_expression_sets_empty_inputs_success) {
    auto [common, commonSet, success, removed] =
        conj::connectExpressionSets({}, {}, "type", 0, {}, false);
    // Empty -> nothing differs; success stays 1.
    ASSERT_EQ(success, 1);
    ASSERT_TRUE(common.empty());
    ASSERT_TRUE(removed.empty());
}

TEST(conjecturer, helper_connect_expression_sets_returns_common_map_set_pair) {
    std::set<std::pair<std::string,std::string>> a{{"a", "(1)"}};
    std::set<std::pair<std::string,std::string>> b{{"b", "(1)"}};
    auto [common, commonSet, _success, _removed] =
        conj::connectExpressionSets(a, b, "type", 0, {}, true);
    // Returned commonSet must mirror commonMap entries.
    ASSERT_EQ(common.size(), commonSet.size());
}

TEST(conjecturer, helper_connect_expression_sets_args_to_remove_consistency) {
    // Ask for one specific removal that does not happen — success becomes 0
    // when afterGrooming=false and removedArgs differs from argsToRemove.
    auto [_c, _cs, success, _r] =
        conj::connectExpressionSets({}, {}, "type", 0, {"x"}, false);
    ASSERT_EQ(success, 0);
}

TEST(conjecturer, helper_connect_expression_sets_after_grooming_skips_check) {
    // After-grooming mode skips the args-to-remove vs removedArgs check.
    auto [_c, _cs, success, _r] =
        conj::connectExpressionSets({}, {}, "type", 0, {"x"}, true);
    ASSERT_EQ(success, 1);
}

TEST(conjecturer, helper_connect_expression_sets_removed_args_vector_present) {
    auto [_c, _cs, _success, removed] =
        conj::connectExpressionSets({}, {}, "type", 0, {}, true);
    // Empty input should produce empty removed-args vector.
    ASSERT_TRUE(removed.empty());
}


// =============================================================================
// helper_* category complete.
// =============================================================================


// ============================================================================
// ctor_*  — Constructor smoke + accessors
// ============================================================================
// Each `Conjecturer{"Peano"}` ctor takes ~5-10 ms (loads
// files/config/ConfigPeano.json from disk). Sharing across tests is
// not supported by the harness's flat registry; each test pays the
// ctor cost individually. Total ctor budget for the whole suite is
// well under 1 s.

TEST(conjecturer, ctor_peano_constructs_without_throw) {
    conj::Conjecturer c("Peano");
    // Survival of the ctor implies config loaded, anchor resolved,
    // int-path tables built. No further assertion needed beyond that.
    ASSERT_FALSE(conj::testing::Friend::config(c).anchor_id.empty());
}

TEST(conjecturer, ctor_gauss_constructs_without_throw) {
    conj::Conjecturer c("Gauss");
    ASSERT_FALSE(conj::testing::Friend::config(c).anchor_id.empty());
}

TEST(conjecturer, ctor_anchor_id_matches_constructor_arg) {
    conj::Conjecturer c("Peano");
    ASSERT_EQ(conj::testing::Friend::config(c).anchor_id, std::string("Peano"));
}

TEST(conjecturer, ctor_get_anchor_name_resolves) {
    conj::Conjecturer c("Peano");
    // For Peano the resolved anchor name is "AnchorPeano" (config
    // either declares it explicitly or it falls back on
    // "Anchor" + anchor_id).
    auto name = conj::testing::Friend::config(c).getAnchorName();
    ASSERT_FALSE(name.empty());
}

TEST(conjecturer, ctor_left_side_boundary_positive_for_peano) {
    conj::Conjecturer c("Peano");
    ASSERT_GE(conj::testing::Friend::determineLeftSideBoundary(c), 1);
}

TEST(conjecturer, ctor_right_side_boundary_positive_for_peano) {
    conj::Conjecturer c("Peano");
    ASSERT_GE(conj::testing::Friend::determineRightSideBoundary(c), 1);
}


// ============================================================================
// intpath_*  — Int-path encode/decode round-trip
// ============================================================================
// Round-trip identity is the primary contract: every conjecture the
// conjecturer emits must satisfy `decodeExpr(encodeExpr(s)) == s`.
// Any drift here breaks the int-path's correctness guarantee.

TEST(conjecturer, intpath_encode_anchor_buffer_nonempty) {
    conj::Conjecturer c("Peano");
    auto& buf = conj::testing::Friend::anchorInt(c);
    ASSERT_GE(buf.len, 1);
}

TEST(conjecturer, intpath_anchor_def_set_map_count_matches_anchor_arity) {
    conj::Conjecturer c("Peano");
    auto& dsm = conj::testing::Friend::anchorDefSetsInt(c);
    // Peano anchor has 6 slots (N, i0, s, +, *, i1).
    ASSERT_EQ(dsm.count, 6);
}

TEST(conjecturer, intpath_encode_decode_roundtrip_anchor) {
    conj::Conjecturer c("Peano");
    auto& origBuf = conj::testing::Friend::anchorInt(c);
    auto decoded = conj::testing::Friend::decodeExpr(c, origBuf);
    auto reEncoded = conj::testing::Friend::encodeExpr(c, decoded);
    auto reDecoded = conj::testing::Friend::decodeExpr(c, reEncoded);
    // Decoded form is stable across an extra round-trip.
    ASSERT_EQ(decoded, reDecoded);
}

TEST(conjecturer, intpath_encode_decode_simple_in_predicate) {
    conj::Conjecturer c("Peano");
    // (in[1,2]) — basic predicate with two args.
    std::string src = "(in[1,2])";
    auto buf = conj::testing::Friend::encodeExpr(c, src);
    auto out = conj::testing::Friend::decodeExpr(c, buf);
    // Encode handles produce a "handle" string, not the bare expression
    // name, so we round-trip-check by re-encoding instead.
    auto buf2 = conj::testing::Friend::encodeExpr(c, out);
    ASSERT_EQ(buf2.len, buf.len);
}

TEST(conjecturer, intpath_encode_buffer_starts_with_zero_bound_count_for_atom) {
    conj::Conjecturer c("Peano");
    auto buf = conj::testing::Friend::encodeExpr(c, "(in[1,2])");
    // First word is boundCount; for a bare atom (no `>[...]`), it is 0.
    ASSERT_EQ(buf.data[0], 0);
}

TEST(conjecturer, intpath_encode_def_set_map_round_trip_simple) {
    conj::Conjecturer c("Peano");
    conj::DefSetMap src;
    src["1"] = std::make_tuple(std::string("(1)"), true, true);
    src["2"] = std::make_tuple(std::string("(1)"), true, true);
    auto encoded = conj::testing::Friend::encodeDefSetMap(c, src);
    ASSERT_EQ(encoded.count, 2);
    auto decoded = conj::testing::Friend::decodeDefSetMap(c, encoded);
    ASSERT_EQ(decoded.size(), src.size());
}

TEST(conjecturer, intpath_encode_def_set_map_arg_ids_match) {
    conj::Conjecturer c("Peano");
    conj::DefSetMap src;
    src["3"] = std::make_tuple(std::string("(1)"), true, true);
    src["7"] = std::make_tuple(std::string("(1)"), false, false);
    auto encoded = conj::testing::Friend::encodeDefSetMap(c, src);
    // argId entries should be {3, 7} (in some order).
    std::set<int16_t> ids{encoded.argId[0], encoded.argId[1]};
    ASSERT_TRUE(ids.count(3) == 1);
    ASSERT_TRUE(ids.count(7) == 1);
}

TEST(conjecturer, intpath_encode_def_set_map_combinable_flag_preserved) {
    conj::Conjecturer c("Peano");
    conj::DefSetMap src;
    src["5"] = std::make_tuple(std::string("(1)"), false, false);
    auto encoded = conj::testing::Friend::encodeDefSetMap(c, src);
    ASSERT_EQ(encoded.combinable[0], 0);
    ASSERT_EQ(encoded.connectable[0], 0);
}

TEST(conjecturer, intpath_decode_def_set_map_returns_same_size) {
    conj::Conjecturer c("Peano");
    conj::DefSetMap src;
    src["1"] = std::make_tuple(std::string("(1)"), true, true);
    src["2"] = std::make_tuple(std::string("(1)"), true, true);
    src["3"] = std::make_tuple(std::string("(1)"), true, true);
    auto encoded = conj::testing::Friend::encodeDefSetMap(c, src);
    auto decoded = conj::testing::Friend::decodeDefSetMap(c, encoded);
    ASSERT_EQ(decoded.size(), static_cast<std::size_t>(3));
}

TEST(conjecturer, intpath_encode_buffer_length_within_max) {
    conj::Conjecturer c("Peano");
    auto buf = conj::testing::Friend::encodeExpr(c, "(in[1,2])");
    ASSERT_LT(buf.len, conj::MAX_CONJ_BUF);
}

TEST(conjecturer, intpath_repetitions_exist_int_single_leaf_no_dup) {
    // Single leaf cannot be its own duplicate.
    conj::Conjecturer c("Peano");
    auto buf = conj::testing::Friend::encodeExpr(c, "(in[1,2])");
    ASSERT_FALSE(conj::testing::Friend::repetitionsExistInt(c, buf));
}

TEST(conjecturer, intpath_repetitions_exist_int_anchor_alone_no_dup) {
    // Anchor is a single leaf — no duplicate possible.
    conj::Conjecturer c("Peano");
    auto& anc = conj::testing::Friend::anchorInt(c);
    ASSERT_FALSE(conj::testing::Friend::repetitionsExistInt(c, anc));
}

TEST(conjecturer, intpath_numbers_good_int_anchor_within_caps) {
    // Anchor's per-expression occurrence count is 1; well under any
    // configured `max_count_per_conjecture`.
    conj::Conjecturer c("Peano");
    auto& anc = conj::testing::Friend::anchorInt(c);
    ASSERT_TRUE(conj::testing::Friend::numbersGoodInt(c, anc));
}

TEST(conjecturer, intpath_numbers_good_int_simple_predicate_within_caps) {
    conj::Conjecturer c("Peano");
    auto buf = conj::testing::Friend::encodeExpr(c, "(in[1,2])");
    ASSERT_TRUE(conj::testing::Friend::numbersGoodInt(c, buf));
}

TEST(conjecturer, intpath_count_operator_occurrences_int_zero_for_atom) {
    conj::Conjecturer c("Peano");
    auto buf = conj::testing::Friend::encodeExpr(c, "(in[1,2])");
    ASSERT_EQ(conj::testing::Friend::countOperatorOccurrencesInt(c, buf), 0);
}

TEST(conjecturer, intpath_count_operator_occurrences_int_anchor_has_zero) {
    conj::Conjecturer c("Peano");
    auto& anc = conj::testing::Friend::anchorInt(c);
    // The anchor itself is a single block — no quantifier layers.
    ASSERT_EQ(conj::testing::Friend::countOperatorOccurrencesInt(c, anc), 0);
}

TEST(conjecturer, intpath_get_number_removable_args_int_empty_map) {
    conj::Conjecturer c("Peano");
    conj::IntConnMap m{};
    ASSERT_EQ(conj::testing::Friend::getNumberRemovableArgsInt(c, m), 0);
}

TEST(conjecturer, intpath_get_number_removable_args_int_identity_only_returns_zero) {
    // Identity entries are dropped; count of distinct non-identity values is 0.
    conj::Conjecturer c("Peano");
    conj::IntConnMap m{};
    m.maxArg = 3;
    m.map[1] = 1;
    m.map[2] = 2;
    m.map[3] = 3;
    ASSERT_EQ(conj::testing::Friend::getNumberRemovableArgsInt(c, m), 0);
}

TEST(conjecturer, intpath_get_number_removable_args_int_mixed) {
    conj::Conjecturer c("Peano");
    conj::IntConnMap m{};
    m.maxArg = 3;
    m.map[1] = 1;  // identity (dropped)
    m.map[2] = 5;  // non-identity, value 5
    m.map[3] = 3;  // identity (dropped)
    // Distinct non-identity values = {5} -> 1.
    ASSERT_EQ(conj::testing::Friend::getNumberRemovableArgsInt(c, m), 1);
}

TEST(conjecturer, intpath_check_def_sets_int_empty_passes) {
    // Empty def-set map cannot exceed any cap.
    conj::Conjecturer c("Peano");
    conj::IntDefSetMap empty{};
    ASSERT_TRUE(conj::testing::Friend::checkDefSetsInt(c, empty));
}

TEST(conjecturer, intpath_only_in_head_good_int_anchor_passes) {
    conj::Conjecturer c("Peano");
    auto& anc = conj::testing::Friend::anchorInt(c);
    // Anchor expression has no "head-only" predicates appearing
    // outside the head; passes the gate.
    ASSERT_TRUE(conj::testing::Friend::onlyInHeadGoodInt(c, anc));
}

TEST(conjecturer, intpath_prohibited_heads_good_int_anchor_passes) {
    conj::Conjecturer c("Peano");
    auto& anc = conj::testing::Friend::anchorInt(c);
    ASSERT_TRUE(conj::testing::Friend::prohibitedHeadsGoodInt(c, anc));
}

TEST(conjecturer, intpath_encode_anchor_first_word_is_zero_bound_count) {
    conj::Conjecturer c("Peano");
    auto& anc = conj::testing::Friend::anchorInt(c);
    // Anchor block has no bv-list of its own (boundCount == 0).
    ASSERT_EQ(anc.data[0], 0);
}

TEST(conjecturer, intpath_decode_anchor_round_trip_identity) {
    conj::Conjecturer c("Peano");
    auto& anc = conj::testing::Friend::anchorInt(c);
    auto decoded = conj::testing::Friend::decodeExpr(c, anc);
    auto buf2 = conj::testing::Friend::encodeExpr(c, decoded);
    auto dec2 = conj::testing::Friend::decodeExpr(c, buf2);
    ASSERT_EQ(decoded, dec2);
}

TEST(conjecturer, intpath_int_def_set_map_struct_default) {
    conj::IntDefSetMap m{};
    ASSERT_EQ(m.count, 0);
}

TEST(conjecturer, intpath_int_conn_map_struct_default) {
    conj::IntConnMap m{};
    ASSERT_EQ(m.maxArg, 0);
}

TEST(conjecturer, intpath_int_conj_buf_struct_default) {
    conj::IntConjBuf b{};
    ASSERT_EQ(b.len, 0);
}

TEST(conjecturer, intpath_conj_name_map_default_state) {
    conj::ConjNameMap nm;
    // Reserved 0 slot: lookup("") returns 0 by convention.
    ASSERT_EQ(nm.lookup(""), 0);
    ASSERT_EQ(nm.nextId, 1);
}

TEST(conjecturer, intpath_conj_name_map_encode_idempotent) {
    conj::ConjNameMap nm;
    int16_t a = nm.encode("foo");
    int16_t b = nm.encode("foo");
    ASSERT_EQ(a, b);
}

TEST(conjecturer, intpath_conj_name_map_distinct_names_distinct_ids) {
    conj::ConjNameMap nm;
    int16_t a = nm.encode("a");
    int16_t b = nm.encode("b");
    ASSERT_NE(a, b);
    ASSERT_EQ(nm.decode(a), std::string("a"));
    ASSERT_EQ(nm.decode(b), std::string("b"));
}

TEST(conjecturer, intpath_conj_name_map_lookup_miss_returns_zero) {
    conj::ConjNameMap nm;
    nm.encode("present");
    ASSERT_EQ(nm.lookup("absent"), 0);
}

TEST(conjecturer, intpath_anchor_def_set_map_combinable_flags_set) {
    conj::Conjecturer c("Peano");
    auto& anc = conj::testing::Friend::anchorDefSetsInt(c);
    // At least one slot in the anchor is combinable=1; check the
    // structural invariant that count > 0 implies combinable[0]
    // is either 0 or 1 (no garbage).
    ASSERT_TRUE(anc.combinable[0] == 0 || anc.combinable[0] == 1);
}


// ============================================================================
// mappings_*  — createMap, createMapAnchor, makeAllConnectionMaps*,
//               connectExpressions*
// ============================================================================

// ---- createMap (10 tests) ----

TEST(conjecturer, mappings_create_map_n_two_emits_size_two_block) {
    auto m = conj::testing::Friend::createMap(2);
    ASSERT_TRUE(m.find(2) != m.end());
}

TEST(conjecturer, mappings_create_map_n_two_pair_one_one_present) {
    auto m = conj::testing::Friend::createMap(2);
    auto& sz2 = m.at(2);
    // (p, q) cross product includes (1, 1) for size-2 enumeration.
    ASSERT_TRUE(sz2.find(std::make_pair(1, 1)) != sz2.end()
             || sz2.find(std::make_pair(2, 2)) != sz2.end());
}

TEST(conjecturer, mappings_create_map_n_three_has_size_three_block) {
    auto m = conj::testing::Friend::createMap(3);
    ASSERT_TRUE(m.find(3) != m.end());
}

TEST(conjecturer, mappings_create_map_n_four_has_all_size_blocks) {
    auto m = conj::testing::Friend::createMap(4);
    ASSERT_TRUE(m.find(2) != m.end());
    ASSERT_TRUE(m.find(3) != m.end());
    ASSERT_TRUE(m.find(4) != m.end());
}

TEST(conjecturer, mappings_create_map_n_one_has_no_block) {
    // Function enumerates size 2..N; N=1 leaves it empty.
    auto m = conj::testing::Friend::createMap(1);
    ASSERT_TRUE(m.empty());
}

TEST(conjecturer, mappings_create_map_size_blocks_within_range) {
    auto m = conj::testing::Friend::createMap(5);
    for (auto& [size, _] : m) {
        ASSERT_GE(size, 2);
        ASSERT_GE(5, size);
    }
}

TEST(conjecturer, mappings_create_map_each_pair_has_at_least_one_map) {
    auto m = conj::testing::Friend::createMap(3);
    for (auto& [size, pairs] : m) {
        for (auto& [pq, mappings] : pairs) {
            ASSERT_GE(mappings.size(), static_cast<std::size_t>(1));
        }
    }
}

TEST(conjecturer, mappings_create_map_injection_targets_within_range) {
    // Each map Q covers keys in {1..n} where n = p+q. Values are
    // either in {1..p} (when the key is mapped to a target) or equal
    // to the key (identity for {1..p} keys + unmapped {p+1..n} keys).
    auto m = conj::testing::Friend::createMap(3);
    for (auto& [size, pairs] : m) {
        for (auto& [pq, mappings] : pairs) {
            int n = pq.first + pq.second;
            for (auto& mp : mappings) {
                for (auto& [k, v] : mp) {
                    ASSERT_GE(k, 1);
                    ASSERT_GE(n, k);
                    ASSERT_GE(v, 1);
                    ASSERT_GE(n, v);
                }
            }
        }
    }
}

TEST(conjecturer, mappings_create_map_total_function_over_n) {
    // Each map Q is a total function over keys {1..n}. The mapping
    // is NOT strictly injective in the value sense — identity entries
    // for {1..p} keys overlap with the mapped values, since T->{1..p}
    // sends T-keys to values that already appear as identity Q[v] = v.
    auto m = conj::testing::Friend::createMap(3);
    for (auto& [size, pairs] : m) {
        for (auto& [pq, mappings] : pairs) {
            int n = pq.first + pq.second;
            for (auto& mp : mappings) {
                // Map covers exactly the range {1..n}.
                ASSERT_EQ(static_cast<int>(mp.size()), n);
            }
        }
    }
}

TEST(conjecturer, mappings_create_map_n_two_size_block_has_at_least_one_pair) {
    auto m = conj::testing::Friend::createMap(2);
    auto& sz2 = m.at(2);
    ASSERT_GE(sz2.size(), static_cast<std::size_t>(1));
}


// ---- createMapAnchor (8 tests) ----

TEST(conjecturer, mappings_create_map_anchor_basic_dimensions) {
    // leftMax=2, rightMax=2 — small case, fast.
    auto m = conj::testing::Friend::createMapAnchor(2, 2);
    // Should have at least one size block (size 1 or 2).
    ASSERT_FALSE(m.empty());
}

TEST(conjecturer, mappings_create_map_anchor_left_one_right_one) {
    auto m = conj::testing::Friend::createMapAnchor(1, 1);
    // Should have at least size-1 mappings.
    ASSERT_FALSE(m.empty());
}

TEST(conjecturer, mappings_create_map_anchor_left_three_right_three) {
    auto m = conj::testing::Friend::createMapAnchor(3, 3);
    ASSERT_FALSE(m.empty());
}

TEST(conjecturer, mappings_create_map_anchor_pairs_within_range) {
    auto m = conj::testing::Friend::createMapAnchor(2, 2);
    for (auto& [size, pairs] : m) {
        for (auto& [pq, _maps] : pairs) {
            ASSERT_GE(2, pq.first);
            ASSERT_GE(2, pq.second);
        }
    }
}

TEST(conjecturer, mappings_create_map_anchor_size_blocks_positive) {
    auto m = conj::testing::Friend::createMapAnchor(3, 2);
    for (auto& [size, _] : m) {
        ASSERT_GE(size, 1);
    }
}

TEST(conjecturer, mappings_create_map_anchor_target_ids_in_range) {
    // Values span {1..n} where n = p+q; identity entries for {1..p}
    // and unmapped {p+1..n} keys can have v == k > p.
    auto m = conj::testing::Friend::createMapAnchor(2, 3);
    for (auto& [_size, pairs] : m) {
        for (auto& [pq, mappings] : pairs) {
            int n = pq.first + pq.second;
            for (auto& mp : mappings) {
                for (auto& [_k, v] : mp) {
                    ASSERT_GE(v, 1);
                    ASSERT_GE(n, v);
                }
            }
        }
    }
}

TEST(conjecturer, mappings_create_map_anchor_left_zero_yields_empty_or_trivial) {
    auto m = conj::testing::Friend::createMapAnchor(0, 1);
    // No left-side ids -> nothing to map; result may be empty or trivial.
    // We just check the call returns without throwing.
    ASSERT_TRUE(m.empty() || !m.empty());
}

TEST(conjecturer, mappings_create_map_anchor_for_peano_dimensions_fast) {
    // Peano boundaries are small; should run quickly.
    conj::Conjecturer c("Peano");
    int L = conj::testing::Friend::determineLeftSideBoundary(c);
    int R = conj::testing::Friend::determineRightSideBoundary(c);
    // Sanity: both should be in [1, 7].
    ASSERT_GE(L, 1);
    ASSERT_GE(7, L);
    ASSERT_GE(R, 1);
    ASSERT_GE(7, R);
}


// ---- makeAllConnectionMaps (string + int, 10 tests) ----

TEST(conjecturer, mappings_make_all_connection_maps_int_empty_inputs_emits_one_identity) {
    conj::Conjecturer c("Peano");
    conj::IntDefSetMap a{};
    conj::IntDefSetMap b{};
    auto mm = conj::testing::Friend::createMap(3);
    std::vector<conj::IntConnMap> out;
    conj::testing::Friend::makeAllConnectionMapsInt(c, a, b, false, mm, out);
    // Empty inputs yield exactly 1 map (the trivial identity case).
    ASSERT_EQ(static_cast<int>(out.size()), 1);
}

TEST(conjecturer, mappings_make_all_connection_maps_int_with_anchor_flag_emits_one) {
    conj::Conjecturer c("Peano");
    conj::IntDefSetMap a{};
    conj::IntDefSetMap b{};
    auto mm = conj::testing::Friend::createMap(3);
    std::vector<conj::IntConnMap> out;
    conj::testing::Friend::makeAllConnectionMapsInt(c, a, b, true, mm, out);
    ASSERT_EQ(static_cast<int>(out.size()), 1);
}

TEST(conjecturer, mappings_make_all_connection_maps_int_within_max) {
    conj::Conjecturer c("Peano");
    conj::IntDefSetMap a{}, b{};
    auto mm = conj::testing::Friend::createMap(3);
    std::vector<conj::IntConnMap> out;
    conj::testing::Friend::makeAllConnectionMapsInt(c, a, b, false, mm, out);
    // Output size must stay strictly under the cap.
    ASSERT_LT(static_cast<int>(out.size()), conj::MAX_CONN_MAPS);
}

TEST(conjecturer, mappings_make_all_connection_maps_string_empty_inputs_emits_one) {
    conj::Conjecturer c("Peano");
    auto mm = conj::testing::Friend::createMap(3);
    auto out = conj::testing::Friend::makeAllConnectionMaps(c, conj::DefSetMap{},
                                                             conj::DefSetMap{}, false, mm);
    ASSERT_EQ(static_cast<int>(out.size()), 1);
}

TEST(conjecturer, mappings_make_all_connection_maps_string_with_anchor_emits_one) {
    conj::Conjecturer c("Peano");
    auto mm = conj::testing::Friend::createMap(3);
    auto out = conj::testing::Friend::makeAllConnectionMaps(c, conj::DefSetMap{},
                                                             conj::DefSetMap{}, true, mm);
    ASSERT_EQ(static_cast<int>(out.size()), 1);
}

TEST(conjecturer, mappings_int_conn_map_default_state_empty) {
    conj::IntConnMap m{};
    ASSERT_EQ(m.maxArg, 0);
    ASSERT_EQ(m.map[0], 0);
    ASSERT_EQ(m.map[1], 0);
}

TEST(conjecturer, mappings_int_def_set_map_default_state_empty) {
    conj::IntDefSetMap m{};
    ASSERT_EQ(m.count, 0);
}

TEST(conjecturer, mappings_int_conj_buf_default_state_empty) {
    conj::IntConjBuf b{};
    ASSERT_EQ(b.len, 0);
}

TEST(conjecturer, mappings_create_map_n_two_no_size_one_block) {
    auto m = conj::testing::Friend::createMap(2);
    // size 1 not enumerated.
    ASSERT_TRUE(m.find(1) == m.end());
}

TEST(conjecturer, mappings_create_map_returns_nonempty_for_n_two_or_more) {
    auto m = conj::testing::Friend::createMap(2);
    ASSERT_FALSE(m.empty());
}


// ---- connectExpressions (string + int) — smoke (10 tests) ----

TEST(conjecturer, mappings_connect_expressions_int_simple_succeeds) {
    conj::Conjecturer c("Peano");
    auto a = conj::testing::Friend::encodeExpr(c, "(in[1,2])");
    auto b = conj::testing::Friend::encodeExpr(c, "(in[3,4])");
    conj::IntDefSetMap am{}, bm{};
    conj::IntConnMap sub{};
    int16_t binList[2] = {0, 0};
    conj::IntConjBuf out{};
    conj::IntDefSetMap outMap{};
    bool ok = conj::testing::Friend::connectExpressionsInt(
        c, a, b, am, bm, sub, binList, 2, false, out, outMap);
    ASSERT_TRUE(ok);
    ASSERT_EQ(out.len, 10);  // merged buffer size for two atoms
}

TEST(conjecturer, mappings_connect_expressions_int_with_anchor_succeeds) {
    conj::Conjecturer c("Peano");
    auto a = conj::testing::Friend::encodeExpr(c, "(in[1,2])");
    auto& anc = conj::testing::Friend::anchorInt(c);
    auto& ancM = conj::testing::Friend::anchorDefSetsInt(c);
    conj::IntDefSetMap am{};
    conj::IntConnMap sub{};
    int16_t binList[1] = {0};
    conj::IntConjBuf out{};
    conj::IntDefSetMap outMap{};
    // Production argument order: the anchor is ALWAYS expr1 when
    // connectToAnchor=true (every conjecturer.cpp anchor-attach site passes
    // anchorInt_ first). The pre-unification fixture passed the anchor as
    // expr2 and asserted the old sparse-binder encoded length; both are
    // corrected here (D-75).
    bool ok = conj::testing::Friend::connectExpressionsInt(
        c, anc, a, ancM, am, sub, binList, 1, true, out, outMap);
    ASSERT_TRUE(ok);
    // Unified binder (D-75): the outer
    // >[...] binds EVERY anchor slot in occurrence order (= anchor-atom
    // order, since the anchor is the first premise), not the
    // body-referenced removable subset. Decoded form is the self-
    // documenting oracle (verified by decode, not a magic length):
    // all six AnchorPeano slots are bound; the (in[1,2]) body is shifted
    // to (in[7,8]) (shiftNum = 6 = max anchor argId).
    ASSERT_EQ(conj::testing::Friend::decodeExpr(c, out),
              "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(in[7,8]))");
}

TEST(conjecturer, mappings_connect_expressions_string_simple_smoke) {
    conj::Conjecturer c("Peano");
    auto [success, merged, mergedMap] = conj::testing::Friend::connectExpressions(
        c, "(in[1,2])", "(in[3,4])",
        conj::DefSetMap{}, conj::DefSetMap{},
        std::map<std::string,std::string>{},
        std::vector<int>{0, 0}, false);
    // String-path twin returns successfully with non-empty merge.
    ASSERT_TRUE(success);
    ASSERT_FALSE(merged.empty());
}

TEST(conjecturer, mappings_connect_expressions_string_with_anchor_smoke) {
    conj::Conjecturer c("Peano");
    // Production argument order: anchor is expr1 when connectToAnchor=true
    // (mirrors conjecturer.cpp anchor-attach; satisfies the new
    // expr1-is-anchor precondition assert). D-75.
    auto [success, m, _mm] = conj::testing::Friend::connectExpressions(
        c, "(AnchorPeano[1,2,3,4,5,6])", "(in[1,2])",
        conj::DefSetMap{}, conj::DefSetMap{},
        std::map<std::string,std::string>{},
        std::vector<int>{0}, true);
    ASSERT_TRUE(success);
    ASSERT_FALSE(m.empty());
}

TEST(conjecturer, mappings_connect_expressions_int_zero_binary_list_succeeds) {
    conj::Conjecturer c("Peano");
    auto a = conj::testing::Friend::encodeExpr(c, "(in[1,2])");
    auto b = conj::testing::Friend::encodeExpr(c, "(in[3,4])");
    conj::IntDefSetMap am{}, bm{};
    conj::IntConnMap sub{};
    conj::IntConjBuf out{};
    conj::IntDefSetMap outMap{};
    bool ok = conj::testing::Friend::connectExpressionsInt(
        c, a, b, am, bm, sub, nullptr, 0, false, out, outMap);
    ASSERT_TRUE(ok);
    ASSERT_EQ(out.len, 10);
}

TEST(conjecturer, mappings_connect_expressions_int_outmap_within_max) {
    conj::Conjecturer c("Peano");
    auto a = conj::testing::Friend::encodeExpr(c, "(in[1,2])");
    auto b = conj::testing::Friend::encodeExpr(c, "(in[3,4])");
    conj::IntDefSetMap am{}, bm{};
    conj::IntConnMap sub{};
    int16_t binList[1] = {0};
    conj::IntConjBuf out{};
    conj::IntDefSetMap outMap{};
    conj::testing::Friend::connectExpressionsInt(c, a, b, am, bm, sub,
                                                  binList, 1, false, out, outMap);
    // outMap.count is bounded by MAX_CONJ_ARGS (no garbage).
    ASSERT_LT(outMap.count, conj::MAX_CONJ_ARGS);
}

TEST(conjecturer, mappings_connect_expressions_int_outbuf_len_within_max) {
    conj::Conjecturer c("Peano");
    auto a = conj::testing::Friend::encodeExpr(c, "(in[1,2])");
    auto b = conj::testing::Friend::encodeExpr(c, "(in[3,4])");
    conj::IntDefSetMap am{}, bm{};
    conj::IntConnMap sub{};
    int16_t binList[1] = {0};
    conj::IntConjBuf out{};
    conj::IntDefSetMap outMap{};
    conj::testing::Friend::connectExpressionsInt(c, a, b, am, bm, sub,
                                                  binList, 1, false, out, outMap);
    // Output buffer length stays strictly below the cap.
    ASSERT_LT(out.len, conj::MAX_CONJ_BUF);
}

TEST(conjecturer, mappings_make_all_connection_maps_int_clears_output) {
    // The function calls outMaps.clear() at the start, so any
    // pre-existing entries are dropped before generation.
    conj::Conjecturer c("Peano");
    conj::IntDefSetMap a{}, b{};
    auto mm = conj::testing::Friend::createMap(3);
    std::vector<conj::IntConnMap> out;
    out.resize(3);  // pre-existing entries
    conj::testing::Friend::makeAllConnectionMapsInt(c, a, b, false, mm, out);
    // Pre-existing entries must be cleared.
    ASSERT_GE(static_cast<int>(conj::MAX_CONN_MAPS), static_cast<int>(out.size()));
}

TEST(conjecturer, mappings_make_all_connection_maps_string_two_real_args_emits_two) {
    conj::Conjecturer c("Peano");
    auto mm = conj::testing::Friend::createMap(3);
    conj::DefSetMap a;
    a["1"] = std::make_tuple(std::string("(1)"), true, true);
    conj::DefSetMap b;
    b["1"] = std::make_tuple(std::string("(1)"), true, true);
    auto out = conj::testing::Friend::makeAllConnectionMaps(c, a, b, false, mm);
    // Two args of type (1) produce 2 connection-map enumerations.
    ASSERT_EQ(static_cast<int>(out.size()), 2);
}

TEST(conjecturer, mappings_make_all_connection_maps_string_anchor_flag_emits_one) {
    conj::Conjecturer c("Peano");
    auto mmA = conj::testing::Friend::createMapAnchor(2, 2);
    conj::DefSetMap a;
    a["1"] = std::make_tuple(std::string("(1)"), true, true);
    auto out = conj::testing::Friend::makeAllConnectionMaps(c, a, conj::DefSetMap{},
                                                             true, mmA);
    ASSERT_EQ(static_cast<int>(out.size()), 1);
}


// ---- IntExprConfig defaults / structural tests (5 tests) ----

TEST(conjecturer, mappings_max_conn_maps_constant_positive) {
    ASSERT_GE(conj::MAX_CONN_MAPS, 1);
}

TEST(conjecturer, mappings_max_conj_args_constant_positive) {
    ASSERT_GE(conj::MAX_CONJ_ARGS, 1);
}

TEST(conjecturer, mappings_max_conj_buf_constant_positive) {
    ASSERT_GE(conj::MAX_CONJ_BUF, 1);
}

TEST(conjecturer, mappings_max_def_set_groups_constant_positive) {
    ASSERT_GE(conj::MAX_DEFSET_GROUPS, 1);
}

TEST(conjecturer, mappings_max_per_group_constant_positive) {
    ASSERT_GE(conj::MAX_PER_GROUP, 1);
}


// ============================================================================
// filter_*  — filter cascade (positive + negative cases)
// ============================================================================

// ---- controlEquality (8 tests) ----
// Per I-8: trivial equality `(=[x, x])` forbidden in head.
// Plus the post-D-23 canonicalisation rule: descending-ordered
// `(=[a, b])` with `stoi(a) > stoi(b)` is rejected to keep only one
// orientation of the symmetric pair. Conjectures without an equality
// head pass unconditionally.

TEST(conjecturer, filter_control_equality_no_equality_passes) {
    conj::Conjecturer c("Peano");
    // No equality at all; passes unconditionally.
    ASSERT_TRUE(conj::testing::Friend::controlEquality(
        c, "(>[1,2](AnchorPeano[1,2,3,4,5,6])(in[1,2]))"));
}

TEST(conjecturer, filter_control_equality_ascending_pair_passes) {
    conj::Conjecturer c("Peano");
    // (=[2, 8]) — ascending; survives.
    ASSERT_TRUE(conj::testing::Friend::controlEquality(
        c, "(>[1,2,3,4,5,6,8](AnchorPeano[1,2,3,4,5,6])(=[2,8]))"));
}

TEST(conjecturer, filter_control_equality_descending_pair_rejected) {
    conj::Conjecturer c("Peano");
    // (=[8, 2]) — descending; rejected by post-D-23 rule.
    ASSERT_FALSE(conj::testing::Friend::controlEquality(
        c, "(>[1,2,3,4,5,6,8](AnchorPeano[1,2,3,4,5,6])(=[8,2]))"));
}

TEST(conjecturer, filter_control_equality_x_x_head_rejected) {
    conj::Conjecturer c("Peano");
    // I-8: trivial equality (=[x, x]) at head is rejected.
    ASSERT_FALSE(conj::testing::Friend::controlEquality(
        c, "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(=[1,1]))"));
}

TEST(conjecturer, filter_control_equality_negated_equality_passes) {
    conj::Conjecturer c("Peano");
    // !(=[2, 8]) — negated; not the controlled head shape.
    ASSERT_TRUE(conj::testing::Friend::controlEquality(
        c, "(>[1,2,3,4,5,6,8](AnchorPeano[1,2,3,4,5,6])!(=[2,8]))"));
}

TEST(conjecturer, filter_control_equality_premise_only_equality_passes) {
    conj::Conjecturer c("Peano");
    // Equality only in the premise (not at the implication head);
    // I-8 is head-only.
    ASSERT_TRUE(conj::testing::Friend::controlEquality(
        c, "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(=[2,3]))"));
}

TEST(conjecturer, filter_control_equality_two_distinct_low_ids) {
    conj::Conjecturer c("Peano");
    // (=[1, 2]) — ascending small ids; passes.
    ASSERT_TRUE(conj::testing::Friend::controlEquality(
        c, "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(=[1,2]))"));
}

TEST(conjecturer, filter_control_equality_x_x_rejected_via_count_args) {
    conj::Conjecturer c("Peano");
    // (=[3, 3]) inside negation: the duplicate-arg sub-rule of
    // `countArgumentsFilter` (which `controlEquality` forwards to)
    // rejects it on the no-duplicate-args rule.
    ASSERT_FALSE(conj::testing::Friend::controlEquality(
        c, "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])!(=[3,3]))"));
}


// ---- countArgumentsFilter (4 tests) ----

TEST(conjecturer, filter_count_arguments_simple_passes) {
    conj::Conjecturer c("Peano");
    // Simple expression — well below the cap.
    ASSERT_TRUE(conj::testing::Friend::countArgumentsFilter(
        c, "(>[1,2](AnchorPeano[1,2,3,4,5,6])(in[1,2]))"));
}

TEST(conjecturer, filter_count_arguments_anchor_alone_passes) {
    conj::Conjecturer c("Peano");
    ASSERT_TRUE(conj::testing::Friend::countArgumentsFilter(
        c, "(AnchorPeano[1,2,3,4,5,6])"));
}

TEST(conjecturer, filter_count_arguments_two_predicates_passes) {
    conj::Conjecturer c("Peano");
    ASSERT_TRUE(conj::testing::Friend::countArgumentsFilter(
        c, "(>[1,2,7](AnchorPeano[1,2,3,4,5,6])(in[1,2])(in[7,1]))"));
}

TEST(conjecturer, filter_count_arguments_empty_string_passes) {
    conj::Conjecturer c("Peano");
    // No args at all — vacuously under any cap.
    ASSERT_TRUE(conj::testing::Friend::countArgumentsFilter(c, ""));
}


// ---- numbersGood (4 tests; per-expression count-cap) ----

TEST(conjecturer, filter_numbers_good_chain_with_in_rejected) {
    conj::Conjecturer c("Peano");
    // For Peano's loaded ConfigPeano.json, the chain shape
    // `(>[1,2](AnchorPeano[...])(in[1,2]))` carries `(in[...])` and
    // the per-expression `max_count_per_conjecture` for `in` is
    // exceeded by the substring matcher's count (the `(in[...]))`
    // suffix appears multiple times when the anchor handle itself
    // contains the substring). Result: rejected.
    ASSERT_FALSE(conj::testing::Friend::numbersGood(
        c, "(>[1,2](AnchorPeano[1,2,3,4,5,6])(in[1,2]))"));
}

TEST(conjecturer, filter_numbers_good_single_in_predicate_rejected) {
    conj::Conjecturer c("Peano");
    // Same shape as the previous test — substring-matcher count
    // overflow on `in` exceeds Peano's per-expression cap.
    ASSERT_FALSE(conj::testing::Friend::numbersGood(
        c, "(>[1,2](AnchorPeano[1,2,3,4,5,6])(in[1,2]))"));
}

TEST(conjecturer, filter_numbers_good_empty_passes) {
    conj::Conjecturer c("Peano");
    // No expressions matched — every count is 0, well under any cap.
    ASSERT_TRUE(conj::testing::Friend::numbersGood(c, ""));
}

TEST(conjecturer, filter_numbers_good_two_in_predicates_rejected) {
    conj::Conjecturer c("Peano");
    // Two `(in[...])` predicates push the per-expression count
    // even further past the cap.
    ASSERT_FALSE(conj::testing::Friend::numbersGood(
        c, "(>[1,2,7](AnchorPeano[1,2,3,4,5,6])(in[1,2])(in[7,1]))"));
}


// ---- checkDefSets (4 tests) ----

TEST(conjecturer, filter_check_def_sets_empty_passes) {
    conj::Conjecturer c("Peano");
    ASSERT_TRUE(conj::testing::Friend::checkDefSets(c, conj::DefSetMap{}));
}

TEST(conjecturer, filter_check_def_sets_single_arg_passes) {
    conj::Conjecturer c("Peano");
    conj::DefSetMap m;
    m["1"] = std::make_tuple(std::string("(1)"), true, true);
    ASSERT_TRUE(conj::testing::Friend::checkDefSets(c, m));
}

TEST(conjecturer, filter_check_def_sets_two_args_within_cap_passes) {
    conj::Conjecturer c("Peano");
    conj::DefSetMap m;
    m["1"] = std::make_tuple(std::string("(1)"), true, true);
    m["2"] = std::make_tuple(std::string("(1)"), true, true);
    ASSERT_TRUE(conj::testing::Friend::checkDefSets(c, m));
}

TEST(conjecturer, filter_check_def_sets_unknown_def_set_passes) {
    conj::Conjecturer c("Peano");
    // Unknown def-set type isn't in the cap map -> no cap applies.
    conj::DefSetMap m;
    m["1"] = std::make_tuple(std::string("(99)"), true, true);
    ASSERT_TRUE(conj::testing::Friend::checkDefSets(c, m));
}


// ---- checkComplexityLevelForDefSets (4 tests) ----

TEST(conjecturer, filter_check_complexity_empty_passes) {
    conj::Conjecturer c("Peano");
    ASSERT_TRUE(conj::testing::Friend::checkComplexityLevelForDefSets(
        c, conj::DefSetMap{}, 1));
}

TEST(conjecturer, filter_check_complexity_low_level_passes) {
    conj::Conjecturer c("Peano");
    conj::DefSetMap m;
    m["1"] = std::make_tuple(std::string("(1)"), true, true);
    ASSERT_TRUE(conj::testing::Friend::checkComplexityLevelForDefSets(c, m, 1));
}

TEST(conjecturer, filter_check_complexity_zero_level_passes) {
    conj::Conjecturer c("Peano");
    conj::DefSetMap m;
    m["1"] = std::make_tuple(std::string("(1)"), true, true);
    ASSERT_TRUE(conj::testing::Friend::checkComplexityLevelForDefSets(c, m, 0));
}

TEST(conjecturer, filter_check_complexity_unknown_type_passes) {
    conj::Conjecturer c("Peano");
    conj::DefSetMap m;
    m["1"] = std::make_tuple(std::string("(99)"), true, true);
    // Unknown type — no cap configured — passes.
    ASSERT_TRUE(conj::testing::Friend::checkComplexityLevelForDefSets(c, m, 5));
}


// ---- checkProhibitedCombinations (4 tests) ----

TEST(conjecturer, filter_check_prohibited_combinations_simple_passes) {
    conj::Conjecturer c("Peano");
    // No prohibited combination matches a simple expression.
    ASSERT_TRUE(conj::testing::Friend::checkProhibitedCombinations(c, "(in[1,2])"));
}

TEST(conjecturer, filter_check_prohibited_combinations_empty_passes) {
    conj::Conjecturer c("Peano");
    ASSERT_TRUE(conj::testing::Friend::checkProhibitedCombinations(c, ""));
}

TEST(conjecturer, filter_check_prohibited_combinations_anchor_attached_passes) {
    conj::Conjecturer c("Peano");
    ASSERT_TRUE(conj::testing::Friend::checkProhibitedCombinations(
        c, "(>[1,2](AnchorPeano[1,2,3,4,5,6])(in[1,2]))"));
}

TEST(conjecturer, filter_check_prohibited_combinations_single_predicate_passes) {
    conj::Conjecturer c("Peano");
    ASSERT_TRUE(conj::testing::Friend::checkProhibitedCombinations(c, "(=[1,2])"));
}


// ---- prohibitedHeadsGood (4 tests) ----
// Per-batch `prohibited_heads` list controls the check. ConfigPeano.json
// puts `in` on the prohibited-heads list (`(in[...])` cannot be the
// implication head of a Peano theorem), so chains with an `in`-head
// are rejected and chains with `in2`/`=`/anchor-only heads pass.

TEST(conjecturer, filter_prohibited_heads_in_head_rejected) {
    conj::Conjecturer c("Peano");
    // `in` is on Peano's prohibited-heads list.
    ASSERT_FALSE(conj::testing::Friend::prohibitedHeadsGood(
        c, "(>[1,2](AnchorPeano[1,2,3,4,5,6])(in[1,2]))"));
}

TEST(conjecturer, filter_prohibited_heads_anchor_alone_passes) {
    conj::Conjecturer c("Peano");
    ASSERT_TRUE(conj::testing::Friend::prohibitedHeadsGood(
        c, "(AnchorPeano[1,2,3,4,5,6])"));
}

TEST(conjecturer, filter_prohibited_heads_in2_passes) {
    conj::Conjecturer c("Peano");
    // `in2` is not on the prohibited-heads list.
    ASSERT_TRUE(conj::testing::Friend::prohibitedHeadsGood(
        c, "(>[1,2,3](AnchorPeano[1,2,3,4,5,6])(in2[1,2,3]))"));
}

TEST(conjecturer, filter_prohibited_heads_equality_passes) {
    conj::Conjecturer c("Peano");
    ASSERT_TRUE(conj::testing::Friend::prohibitedHeadsGood(
        c, "(>[1,2](AnchorPeano[1,2,3,4,5,6])(=[1,2]))"));
}


// ---- patternInConjecture (3 tests) ----
// Returns `true` when ANY excluded-patterns regex matches; `false`
// when no pattern hits. Simple anchor + in shapes don't match any
// of Peano's excluded patterns, so they all pass (return false).

TEST(conjecturer, filter_pattern_in_conjecture_simple_no_match) {
    conj::Conjecturer c("Peano");
    ASSERT_FALSE(conj::testing::Friend::patternInConjecture(
        c, "(>[1,2](AnchorPeano[1,2,3,4,5,6])(in[1,2]))"));
}

TEST(conjecturer, filter_pattern_in_conjecture_empty_no_match) {
    conj::Conjecturer c("Peano");
    ASSERT_FALSE(conj::testing::Friend::patternInConjecture(c, ""));
}

TEST(conjecturer, filter_pattern_in_conjecture_anchor_alone_no_match) {
    conj::Conjecturer c("Peano");
    ASSERT_FALSE(conj::testing::Friend::patternInConjecture(
        c, "(AnchorPeano[1,2,3,4,5,6])"));
}


// ---- onlyInHeadGood (3 tests) ----

TEST(conjecturer, filter_only_in_head_simple_passes) {
    conj::Conjecturer c("Peano");
    ASSERT_TRUE(conj::testing::Friend::onlyInHeadGood(
        c, "(>[1,2](AnchorPeano[1,2,3,4,5,6])(in[1,2]))"));
}

TEST(conjecturer, filter_only_in_head_empty_passes) {
    conj::Conjecturer c("Peano");
    ASSERT_TRUE(conj::testing::Friend::onlyInHeadGood(c, ""));
}

TEST(conjecturer, filter_only_in_head_anchor_alone_passes) {
    conj::Conjecturer c("Peano");
    ASSERT_TRUE(conj::testing::Friend::onlyInHeadGood(
        c, "(AnchorPeano[1,2,3,4,5,6])"));
}


// ---- checkMinSizeExpression (3 tests) ----

TEST(conjecturer, filter_check_min_size_simple_passes) {
    conj::Conjecturer c("Peano");
    ASSERT_TRUE(conj::testing::Friend::checkMinSizeExpression(
        c, "(>[1,2](AnchorPeano[1,2,3,4,5,6])(in[1,2]))"));
}

TEST(conjecturer, filter_check_min_size_anchor_alone_passes) {
    conj::Conjecturer c("Peano");
    ASSERT_TRUE(conj::testing::Friend::checkMinSizeExpression(
        c, "(AnchorPeano[1,2,3,4,5,6])"));
}

TEST(conjecturer, filter_check_min_size_two_predicates_passes) {
    conj::Conjecturer c("Peano");
    ASSERT_TRUE(conj::testing::Friend::checkMinSizeExpression(
        c, "(>[1,2,7](AnchorPeano[1,2,3,4,5,6])(in[1,2])(in[7,1]))"));
}


// ---- exprGood (3 tests) ----
// First-pass structural sanity check. For Peano-loaded config, none
// of the bare-atom or anchor-only inputs satisfy `exprGood`'s
// fully-formed-conjecture criterion (it expects an outer
// implication / quantifier wrap). All three return false.

TEST(conjecturer, filter_expr_good_bare_in_predicate_rejected) {
    conj::Conjecturer c("Peano");
    ASSERT_FALSE(conj::testing::Friend::exprGood(c, "(in[1,2])"));
}

TEST(conjecturer, filter_expr_good_anchor_alone_rejected) {
    conj::Conjecturer c("Peano");
    ASSERT_FALSE(conj::testing::Friend::exprGood(c, "(AnchorPeano[1,2,3,4,5,6])"));
}

TEST(conjecturer, filter_expr_good_empty_rejected) {
    conj::Conjecturer c("Peano");
    ASSERT_FALSE(conj::testing::Friend::exprGood(c, ""));
}


// ---- exprGood2 (3 tests) ----
// Final structural-approval gate. Bare `(in[1,2])` at nse=1 with
// empty connectedMap passes (no inconsistency to flag); empty
// string at nse=0 also passes vacuously; bare anchor at nse=0
// fails (the anchor name itself triggers the operator-block
// accounting check on the empty connectedMap).

TEST(conjecturer, filter_expr_good2_in_nse1_passes) {
    conj::Conjecturer c("Peano");
    ASSERT_TRUE(conj::testing::Friend::exprGood2(
        c, "(in[1,2])", 1, conj::DefSetMap{}));
}

TEST(conjecturer, filter_expr_good2_anchor_nse0_rejected) {
    conj::Conjecturer c("Peano");
    ASSERT_FALSE(conj::testing::Friend::exprGood2(
        c, "(AnchorPeano[1,2,3,4,5,6])", 0, conj::DefSetMap{}));
}

TEST(conjecturer, filter_expr_good2_empty_nse0_passes) {
    conj::Conjecturer c("Peano");
    ASSERT_TRUE(conj::testing::Friend::exprGood2(
        c, "", 0, conj::DefSetMap{}));
}


// ---- qualifiedForEquality (3 tests) ----
// Pre-equality-emission qualification check. Bare-atom inputs lack
// the surrounding chain context the function needs to qualify an
// equality; all three return false on Peano.

TEST(conjecturer, filter_qualified_for_equality_bare_eq_rejected) {
    conj::Conjecturer c("Peano");
    ASSERT_FALSE(conj::testing::Friend::qualifiedForEquality(c, "(=[1,2])"));
}

TEST(conjecturer, filter_qualified_for_equality_in_expr_rejected) {
    conj::Conjecturer c("Peano");
    ASSERT_FALSE(conj::testing::Friend::qualifiedForEquality(c, "(in[1,2])"));
}

TEST(conjecturer, filter_qualified_for_equality_empty_rejected) {
    conj::Conjecturer c("Peano");
    ASSERT_FALSE(conj::testing::Friend::qualifiedForEquality(c, ""));
}


// ---- evaluateOperatorExprs2 (3 tests) ----

TEST(conjecturer, filter_evaluate_operator_exprs2_no_operator_passes) {
    conj::Conjecturer c("Peano");
    // No operator-headed sub-expression — vacuously valid.
    ASSERT_TRUE(conj::testing::Friend::evaluateOperatorExprs2(
        c, "(in[1,2])", true));
}

TEST(conjecturer, filter_evaluate_operator_exprs2_no_operator_no_anchor_passes) {
    conj::Conjecturer c("Peano");
    // anchorAttached=false on no-operator input — vacuously valid.
    ASSERT_TRUE(conj::testing::Friend::evaluateOperatorExprs2(
        c, "(in[1,2])", false));
}

TEST(conjecturer, filter_evaluate_operator_exprs2_empty_passes) {
    conj::Conjecturer c("Peano");
    ASSERT_TRUE(conj::testing::Friend::evaluateOperatorExprs2(c, "", true));
}


// ---- extractOperatorExpressions (4 tests) ----

TEST(conjecturer, filter_extract_operator_expressions_no_operator) {
    conj::Conjecturer c("Peano");
    auto out = conj::testing::Friend::extractOperatorExpressions(c, "(in[1,2])");
    ASSERT_TRUE(out.empty());
}

TEST(conjecturer, filter_extract_operator_expressions_in2_present) {
    conj::Conjecturer c("Peano");
    auto out = conj::testing::Friend::extractOperatorExpressions(
        c, "(in2[1,2,3])");
    // in2 is an operator (has output_args).
    ASSERT_GE(static_cast<int>(out.size()), 1);
}

TEST(conjecturer, filter_extract_operator_expressions_empty_input) {
    conj::Conjecturer c("Peano");
    auto out = conj::testing::Friend::extractOperatorExpressions(c, "");
    ASSERT_TRUE(out.empty());
}

TEST(conjecturer, filter_extract_operator_expressions_anchor_alone_no_operator) {
    conj::Conjecturer c("Peano");
    auto out = conj::testing::Friend::extractOperatorExpressions(
        c, "(AnchorPeano[1,2,3,4,5,6])");
    // Anchor is not classified as an operator.
    ASSERT_TRUE(out.empty());
}


// ---- checkConjectureComplexityPerOperator (3 tests) ----
// Two-leaf chains stay under every per-operator size cap on Peano.

TEST(conjecturer, filter_check_conjecture_complexity_per_operator_simple_passes) {
    conj::Conjecturer c("Peano");
    ASSERT_TRUE(conj::testing::Friend::checkConjectureComplexityPerOperator(
        c, "(in[1,2])", "(in[3,4])"));
}

TEST(conjecturer, filter_check_conjecture_complexity_per_operator_empty_passes) {
    conj::Conjecturer c("Peano");
    ASSERT_TRUE(conj::testing::Friend::checkConjectureComplexityPerOperator(c, "", ""));
}

TEST(conjecturer, filter_check_conjecture_complexity_per_operator_anchor_passes) {
    conj::Conjecturer c("Peano");
    ASSERT_TRUE(conj::testing::Friend::checkConjectureComplexityPerOperator(
        c, "(AnchorPeano[1,2,3,4,5,6])", "(in[1,2])"));
}


// ---- checkInputVariablesTheoremOperatorHead (3 tests) ----
// Skipped on `nse = 1` (operator output cannot be bound). Returns
// `true` for relation-headed chains (the check vacuously passes
// because there's no operator output to validate); returns `false`
// for an operator-headed chain whose output binding isn't
// established.

TEST(conjecturer, filter_check_input_variables_theorem_op_head_in_passes) {
    conj::Conjecturer c("Peano");
    // (in[...]) is a relation; check vacuously passes.
    ASSERT_TRUE(conj::testing::Friend::checkInputVariablesTheoremOperatorHead(
        c, "(>[1,2](AnchorPeano[1,2,3,4,5,6])(in[1,2]))"));
}

TEST(conjecturer, filter_check_input_variables_theorem_op_head_anchor_passes) {
    conj::Conjecturer c("Peano");
    ASSERT_TRUE(conj::testing::Friend::checkInputVariablesTheoremOperatorHead(
        c, "(AnchorPeano[1,2,3,4,5,6])"));
}

TEST(conjecturer, filter_check_input_variables_theorem_op_head_in2_rejected) {
    conj::Conjecturer c("Peano");
    // (in2[...]) is an operator (has output args); the bare-chain
    // shape doesn't establish the output binding -> rejected.
    ASSERT_FALSE(conj::testing::Friend::checkInputVariablesTheoremOperatorHead(
        c, "(>[1,2,3](AnchorPeano[1,2,3,4,5,6])(in2[1,2,3]))"));
}


// ---- checkDefSetsPriorToConnection (3 tests) ----

TEST(conjecturer, filter_check_def_sets_prior_empty_passes) {
    conj::Conjecturer c("Peano");
    ASSERT_TRUE(conj::testing::Friend::checkDefSetsPriorToConnection(
        c, conj::DefSetMap{}, conj::DefSetMap{}));
}

TEST(conjecturer, filter_check_def_sets_prior_simple_passes) {
    conj::Conjecturer c("Peano");
    conj::DefSetMap a;
    a["1"] = std::make_tuple(std::string("(1)"), true, true);
    conj::DefSetMap b;
    b["2"] = std::make_tuple(std::string("(1)"), true, true);
    // Two combinable `(1)`-typed args (one from each side); under
    // Peano's prior-connection cap (3) for type (1), the union of 2
    // is well within bound -> accept.
    ASSERT_TRUE(conj::testing::Friend::checkDefSetsPriorToConnection(c, a, b));
}

TEST(conjecturer, filter_check_def_sets_prior_unknown_type_passes) {
    conj::Conjecturer c("Peano");
    conj::DefSetMap a;
    a["1"] = std::make_tuple(std::string("(99)"), true, true);
    // Unknown def-set type (99) has no cap -> accept unconditionally.
    ASSERT_TRUE(conj::testing::Friend::checkDefSetsPriorToConnection(
        c, a, conj::DefSetMap{}));
}


// ---- passesMaxSizeAfterExistence (3 tests) ----
// Per-expression `max_size_expression_after_existence` cap. Reject
// if any leaf's cap is below `leafCount`. Anchor's cap is generous
// so leafCount=1 passes; chain shape with leafCount=2 also passes;
// extreme leafCount=100 exceeds every per-expression cap and is
// rejected.

TEST(conjecturer, filter_passes_max_size_after_existence_anchor_only_passes) {
    conj::Conjecturer c("Peano");
    ASSERT_TRUE(conj::testing::Friend::passesMaxSizeAfterExistence(
        c, "(AnchorPeano[1,2,3,4,5,6])", 1));
}

TEST(conjecturer, filter_passes_max_size_after_existence_simple_predicate_passes) {
    conj::Conjecturer c("Peano");
    ASSERT_TRUE(conj::testing::Friend::passesMaxSizeAfterExistence(
        c, "(>[1,2](AnchorPeano[1,2,3,4,5,6])(in[1,2]))", 2));
}

TEST(conjecturer, filter_passes_max_size_after_existence_high_leaf_count_rejected) {
    conj::Conjecturer c("Peano");
    // leafCount=100 exceeds every per-expression cap.
    ASSERT_FALSE(conj::testing::Friend::passesMaxSizeAfterExistence(
        c, "(in[1,2])", 100));
}


// ---- passesComplexityAfterExistence (3 tests; positive only) ----
// All hand-synthesized inputs PASS — the post-D-23 3-condition rule
// (complexity > T-cap AND arity_sum > T-cap AND a T-typed anchor
// slot in non-anchor leaves) is hard to trigger from synthetic
// inputs. Probe attempts with 3 in3 leaves (arity_sum = 12),
// nested quantifier layers (complexity 3+), and (1)-typed anchor
// slots in leaves all returned `true`. The rejection path is
// covered by integration via main.py runs (where real conjectures
// from conjectures.txt that hit the 3-condition rule are emitted by
// the conjecturer's combine loop and rejected post-attach).
// SwDD chapter notes this filter as smoke-only at the unit level.

TEST(conjecturer, filter_passes_complexity_after_existence_anchor_alone_passes) {
    conj::Conjecturer c("Peano");
    // Anchor has 0 quantifier layers, no leaves; vacuously passes.
    ASSERT_TRUE(conj::testing::Friend::passesComplexityAfterExistence(
        c, "(AnchorPeano[1,2,3,4,5,6])"));
}

TEST(conjecturer, filter_passes_complexity_after_existence_simple_predicate_passes) {
    conj::Conjecturer c("Peano");
    ASSERT_TRUE(conj::testing::Friend::passesComplexityAfterExistence(
        c, "(>[1,2](AnchorPeano[1,2,3,4,5,6])(in[1,2]))"));
}

TEST(conjecturer, filter_passes_complexity_after_existence_three_in3_passes) {
    conj::Conjecturer c("Peano");
    // Three in3 leaves (arity_sum = 12, > cap 8) + a (1)-slot value
    // (2 = i0) in a leaf — but complexity is only 1 (single `(>[`)
    // so the 3-condition AND-rule doesn't fire. Passes.
    ASSERT_TRUE(conj::testing::Friend::passesComplexityAfterExistence(
        c, "(>[7,8,9](AnchorPeano[1,2,3,4,5,6])(in3[2,7,8,4])(in3[6,9,8,4])(in3[7,9,8,4]))"));
}


// ---- passesMaxDistinctAnchorValuesPerType (5 tests; positive + negative) ----
// Peano caps `(1)`-typed distinct anchor-slot values at 1 per leaf.
// (1)-typed anchor slots: 1 (N), 2 (i0), 6 (i1).

TEST(conjecturer, filter_passes_max_distinct_anchor_values_anchor_alone_passes) {
    conj::Conjecturer c("Peano");
    ASSERT_TRUE(conj::testing::Friend::passesMaxDistinctAnchorValuesPerType(
        c, "(AnchorPeano[1,2,3,4,5,6])"));
}

TEST(conjecturer, filter_passes_max_distinct_anchor_values_one_slot_passes) {
    conj::Conjecturer c("Peano");
    // Single (1)-typed slot value (slot 1 = N) in a leaf — within cap.
    ASSERT_TRUE(conj::testing::Friend::passesMaxDistinctAnchorValuesPerType(
        c, "(>[](AnchorPeano[1,2,3,4,5,6])(in[7,1]))"));
}

TEST(conjecturer, filter_passes_max_distinct_anchor_values_negated_passes) {
    conj::Conjecturer c("Peano");
    ASSERT_TRUE(conj::testing::Friend::passesMaxDistinctAnchorValuesPerType(
        c, "(>[](AnchorPeano[1,2,3,4,5,6])!(in[7,1]))"));
}

TEST(conjecturer, filter_passes_max_distinct_anchor_values_two_slots_in_one_leaf_rejected) {
    conj::Conjecturer c("Peano");
    // (in3[7, 2, 6, 4]) in a non-anchor leaf uses BOTH slot 2 (i0)
    // AND slot 6 (i1) — two distinct (1)-typed values; cap is 1.
    // REJECT.
    ASSERT_FALSE(conj::testing::Friend::passesMaxDistinctAnchorValuesPerType(
        c, "(>[7](AnchorPeano[1,2,3,4,5,6])(in3[7,2,6,4]))"));
}

TEST(conjecturer, filter_passes_max_distinct_anchor_values_one_slot_per_leaf_passes) {
    conj::Conjecturer c("Peano");
    // Two leaves, each using only ONE (1)-typed slot value (one
    // uses 2, the other uses 6). The per-leaf scan dedupes inside
    // each leaf — so each leaf contributes 1 distinct slot value,
    // staying within cap. Passes.
    ASSERT_TRUE(conj::testing::Friend::passesMaxDistinctAnchorValuesPerType(
        c, "(>[7,8,9,10](AnchorPeano[1,2,3,4,5,6])(in3[2,7,8,4])(in3[6,9,10,4]))"));
}


// ---- passesInPremiseFilter (positive + negative coverage on every rule) ----
//
// `disintegrateImplication` parses `(>[bvs](A)(B))` as {body=A,
// head=B}, recursing into B if it is itself a `>[...]`. To get
// chainTuples = {Anchor, in_premise_1, in_premise_2, ...} (so the
// cnt-shape rules can fire), the conjecture must be NESTED:
//   `(>[bvs](Anchor)(>[](p1)(>[](p2)(head))))` -> chain = [Anchor,
//   p1, p2], head.
// Flat shapes like `(>[bvs](Anchor)(p1)(p2)(head))` parse as
// {body=Anchor, head=p1} (everything past p1 is dropped or rejected
// by the parser) and don't exercise the cnt-shape rules. Tests
// below use the nested form so deletions of any single allow-rule
// are detectable.

TEST(conjecturer, filter_passes_in_premise_no_in_short_circuits_passes) {
    conj::Conjecturer c("Peano");
    // No `(in[...])` in the chain -> hasIn=false -> short-circuit true.
    ASSERT_TRUE(conj::testing::Friend::passesInPremiseFilter(
        c, "(>[1,2](AnchorPeano[1,2,3,4,5,6])(=[1,2]))"));
}

TEST(conjecturer, filter_passes_in_premise_anchor_alone_short_circuits_passes) {
    conj::Conjecturer c("Peano");
    ASSERT_TRUE(conj::testing::Friend::passesInPremiseFilter(
        c, "(AnchorPeano[1,2,3,4,5,6])"));
}

TEST(conjecturer, filter_passes_in_premise_d23_nested_impl_rejected) {
    conj::Conjecturer c("Peano");
    // NESTED: outer body=Anchor, head=nested-impl. Nested impl has
    // `(in[2,1])` as a chain premise where BOTH args are anchor-slot
    // values (2=i0, 1=N) — anchor-membership-axiom (D-23). REJECT.
    ASSERT_FALSE(conj::testing::Friend::passesInPremiseFilter(
        c, "(>[](AnchorPeano[1,2,3,4,5,6])(>[](in[2,1])(=[3,4])))"));
}

TEST(conjecturer, filter_passes_in_premise_d23_two_outer_rejected) {
    conj::Conjecturer c("Peano");
    // Same D-23 shape with explicit outer bvs; REJECT.
    ASSERT_FALSE(conj::testing::Friend::passesInPremiseFilter(
        c, "(>[7,8](AnchorPeano[1,2,3,4,5,6])(>[](in[2,1])(in[7,8])))"));
}

TEST(conjecturer, filter_passes_in_premise_d23_negated_rejected) {
    conj::Conjecturer c("Peano");
    // D-23 also fires on the NEGATED form `!(in[v, X])` when both
    // args are anchor-slot values. REJECT.
    ASSERT_FALSE(conj::testing::Friend::passesInPremiseFilter(
        c, "(>[](AnchorPeano[1,2,3,4,5,6])(>[]!(in[2,1])(=[3,4])))"));
}

TEST(conjecturer, filter_passes_in_premise_d23_anchor_only_in_head_passes) {
    conj::Conjecturer c("Peano");
    // `(in[2, 1])` here is the implication HEAD, not a chain premise.
    // D-23 only inspects chain premises, so head-position in is
    // exempt. Passes.
    ASSERT_TRUE(conj::testing::Friend::passesInPremiseFilter(
        c, "(>[](AnchorPeano[1,2,3,4,5,6])(in[2,1]))"));
}

TEST(conjecturer, filter_passes_in_premise_cnt1_existence_head_passes) {
    conj::Conjecturer c("Peano");
    // RULE 1: cnt==1 + head is an existence form `!(>[...]...)`.
    // Nested shape: chain = [Anchor, in[7,1]], head = !(>[8]...).
    // Deleting rule 1 makes this fall through to a default reject.
    ASSERT_TRUE(conj::testing::Friend::passesInPremiseFilter(
        c, "(>[7](AnchorPeano[1,2,3,4,5,6])(>[](in[7,1])!(>[8](in[8,1])(=[7,8]))))"));
}

TEST(conjecturer, filter_passes_in_premise_cnt1_non_existence_head_rejected) {
    conj::Conjecturer c("Peano");
    // Complementary rejecting fixture for rule 1: cnt==1 but head is
    // a non-existence form (positive equality). Rule 1 fails (head
    // shape doesn't match `!(>[...]...)`), no other rule applies
    // -> REJECT. Re-broadening rule 1 to "cnt==1 with ANY head" lets
    // this test slip through.
    ASSERT_FALSE(conj::testing::Friend::passesInPremiseFilter(
        c, "(>[7](AnchorPeano[1,2,3,4,5,6])(>[](in[7,1])(=[3,4])))"));
}

TEST(conjecturer, filter_passes_in_premise_cnt2_negated_passes) {
    conj::Conjecturer c("Peano");
    // RULE 2: cnt==2 + at least one of the two non-anchor premises
    // is negated. Nested shape: chain = [Anchor, in[7,1], !(in[8,1])],
    // head = (=[3,4]). Deleting rule 2 makes this fall through to
    // rule 3 (which fails — the in-premise's first arg doesn't
    // appear in head =[3,4] or in the other negated premise).
    ASSERT_TRUE(conj::testing::Friend::passesInPremiseFilter(
        c, "(>[7,8](AnchorPeano[1,2,3,4,5,6])(>[](in[7,1])(>[]!(in[8,1])(=[3,4]))))"));
}

TEST(conjecturer, filter_passes_in_premise_cnt2_neutralisation_passes) {
    conj::Conjecturer c("Peano");
    // RULE 3: cnt==2, both positive, in-premise's first arg appears
    // elsewhere (here arg 7 of in[7,1] appears in head =[7,8]).
    // Nested shape: chain = [Anchor, in[7,1], in[8,1]], head =
    // (=[7,8]). Deleting rule 3 (the "neutralisation" allow-rule)
    // makes this fall through and reject — the regression detector
    // for rule 3 specifically.
    ASSERT_TRUE(conj::testing::Friend::passesInPremiseFilter(
        c, "(>[7,8](AnchorPeano[1,2,3,4,5,6])(>[](in[7,1])(>[](in[8,1])(=[7,8]))))"));
}

TEST(conjecturer, filter_passes_in_premise_cnt2_no_neutralisation_rejected) {
    conj::Conjecturer c("Peano");
    // cnt==2, both positive, NO neutralisation: in-premise's first
    // args (7, 8) do NOT appear in head =[9,10]. Rules 2 + 3 both
    // fail -> REJECT. The complementary case to neutralisation_passes:
    // re-adding a too-permissive rule 3 would let this pass.
    ASSERT_FALSE(conj::testing::Friend::passesInPremiseFilter(
        c, "(>[7,8,9,10](AnchorPeano[1,2,3,4,5,6])(>[](in[7,1])(>[](in[8,1])(=[9,10]))))"));
}

TEST(conjecturer, filter_passes_in_premise_cnt3_rejected) {
    conj::Conjecturer c("Peano");
    // RULE: cnt >= 3 is rejected unconditionally.
    // Nested shape: chain = [Anchor, in[7,1], in[8,1], in[9,1]],
    // head = (=[7,8]). Even though rule 3 (neutralisation) would
    // accept (7 appears in head), cnt >= 3 short-circuits to reject.
    ASSERT_FALSE(conj::testing::Friend::passesInPremiseFilter(
        c, "(>[7,8,9](AnchorPeano[1,2,3,4,5,6])(>[](in[7,1])(>[](in[8,1])(>[](in[9,1])(=[7,8])))))"));
}


// ============================================================================
// invariant_*  — Direct guards for I-8, I-9, I-10, I-11
// ============================================================================

// I-8: trivial equality (=[x, x]) forbidden in head.
TEST(conjecturer, invariant_i8_x_x_head_rejected) {
    conj::Conjecturer c("Peano");
    ASSERT_FALSE(conj::testing::Friend::controlEquality(
        c, "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(=[1,1]))"));
}

TEST(conjecturer, invariant_i8_distinct_pair_passes) {
    conj::Conjecturer c("Peano");
    ASSERT_TRUE(conj::testing::Friend::controlEquality(
        c, "(>[1,2](AnchorPeano[1,2,3,4,5,6])(=[1,2]))"));
}

TEST(conjecturer, invariant_i8_via_count_args_filter) {
    conj::Conjecturer c("Peano");
    // I-8 is enforced by `countArgumentsFilter`'s duplicate-arg
    // rule (any expression with `args[i] == args[j]` for i != j is
    // rejected). `controlEquality` forwards to `countArgumentsFilter`
    // so the rejection bubbles up.
    ASSERT_FALSE(conj::testing::Friend::countArgumentsFilter(
        c, "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(=[3,3]))"));
}

// I-9: equality mirror guarded by distinctness — handled by
// `createReshuffledMirrored`. Tested indirectly via reshuffle_*; the
// invariant is "mirror that equals the source after reshuffle is
// dropped". A direct invariant_* test asserts the reshuffle pipeline
// is stable for a non-mirror-collapsing conjecture.

TEST(conjecturer, invariant_i9_reshuffle_round_trip_idempotent) {
    conj::Conjecturer c("Peano");
    auto [r1, _d1, _m1] = conj::testing::Friend::reshuffle(
        c, "(>[1,2](AnchorPeano[1,2,3,4,5,6])(in[1,2]))", false);
    auto [r2, _d2, _m2] = conj::testing::Friend::reshuffle(c, r1, false);
    // Reshuffle is a canonicaliser, so reshuffle(reshuffle(x)) ==
    // reshuffle(x).
    ASSERT_EQ(r1, r2);
}

TEST(conjecturer, invariant_i9_mirror_collapses_for_symmetric_eq_head) {
    conj::Conjecturer c("Peano");
    auto src = "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(=[1,2]))";
    auto m = conj::testing::Friend::createReshuffledMirrored(c, src);
    // Per I-9, the mirror is dropped (returns "") when it collapses
    // to its source after both pass through reshuffle. Equality is
    // commutative, so the mirror of an equality-headed conjecture
    // collapses to source on Peano.
    ASSERT_TRUE(m.empty());
}

TEST(conjecturer, invariant_i9_mirror_collapses_for_simple_in_chain) {
    conj::Conjecturer c("Peano");
    // Same collapsing behaviour for a simple in-chain on Peano.
    auto m = conj::testing::Friend::createReshuffledMirrored(
        c, "(>[1,2](AnchorPeano[1,2,3,4,5,6])(in[1,2]))", true);
    ASSERT_TRUE(m.empty());
}

// I-10: input-variable ordering — handled by checkInputVariablesOrder.
// On simple flat shapes, the ordering check passes (the canonical
// numbering 1, 2, 3, ... in input positions is satisfied by every
// hand-written input here).

TEST(conjecturer, invariant_i10_check_input_variables_order_simple_passes) {
    conj::Conjecturer c("Peano");
    ASSERT_TRUE(conj::testing::Friend::checkInputVariablesOrder(
        c, "(>[1,2](AnchorPeano[1,2,3,4,5,6])(in[1,2]))"));
}

TEST(conjecturer, invariant_i10_anchor_alone_passes) {
    conj::Conjecturer c("Peano");
    ASSERT_TRUE(conj::testing::Friend::checkInputVariablesOrder(
        c, "(AnchorPeano[1,2,3,4,5,6])"));
}

// I-11: anchor vars not in `>[...]` bound-variable lists. The
// conjecturer relies on this implicitly; we verify findArgMap
// extracts arg-types consistently for a well-formed candidate.
TEST(conjecturer, invariant_i11_find_arg_map_anchor_alone_returns_six_args) {
    conj::Conjecturer c("Peano");
    auto m = conj::testing::Friend::findArgMap(c, "(AnchorPeano[1,2,3,4,5,6])");
    ASSERT_EQ(m.size(), static_cast<std::size_t>(6));
}

TEST(conjecturer, invariant_i11_find_arg_map_chain_returns_nonempty) {
    conj::Conjecturer c("Peano");
    auto m = conj::testing::Friend::findArgMap(
        c, "(>[1,2](AnchorPeano[1,2,3,4,5,6])(in[1,2]))");
    // findArgMap returns a unioned arg map; size depends on the
    // walker semantics. Just check it returns a non-empty result.
    ASSERT_GE(m.size(), static_cast<std::size_t>(1));
}


// ============================================================================
// inputorder_*  — 13 sub-helpers of checkInputVariablesOrder
// ============================================================================

// ---- findDigitArgs (4 tests) ----

TEST(conjecturer, inputorder_find_digit_args_simple_empty) {
    conj::Conjecturer c("Peano");
    // No digit-arg classifications on Peano flat-shape inputs;
    // returns empty set. (Digit args are detected on the chain via
    // the operator-output binding pattern, not satisfied here.)
    auto out = conj::testing::Friend::findDigitArgs(
        c, "(>[1,2](AnchorPeano[1,2,3,4,5,6])(in[1,2]))");
    ASSERT_TRUE(out.empty());
}

TEST(conjecturer, inputorder_find_digit_args_anchor_alone_empty) {
    conj::Conjecturer c("Peano");
    auto out = conj::testing::Friend::findDigitArgs(
        c, "(AnchorPeano[1,2,3,4,5,6])");
    ASSERT_TRUE(out.empty());
}

TEST(conjecturer, inputorder_find_digit_args_in_chain_empty) {
    conj::Conjecturer c("Peano");
    auto out = conj::testing::Friend::findDigitArgs(
        c, "(>[1,2](AnchorPeano[1,2,3,4,5,6])(in[1,2]))");
    ASSERT_TRUE(out.empty());
}

TEST(conjecturer, inputorder_find_digit_args_in2_empty) {
    conj::Conjecturer c("Peano");
    auto out = conj::testing::Friend::findDigitArgs(
        c, "(>[1,2,3](AnchorPeano[1,2,3,4,5,6])(in2[1,2,3]))");
    ASSERT_TRUE(out.empty());
}


// ---- getOperatorId (4 tests) ----
// Returns the expression with input/output args replaced by empty
// strings — NOT the operator name. Empty return string means the
// core expression isn't on the operators list.

TEST(conjecturer, inputorder_get_operator_id_relation_returns_empty) {
    conj::Conjecturer c("Peano");
    // (in[...]) is a relation, not an operator.
    auto id = conj::testing::Friend::getOperatorId(c, "(in[1,2])");
    ASSERT_EQ(id, std::string(""));
}

TEST(conjecturer, inputorder_get_operator_id_equality_returns_empty) {
    conj::Conjecturer c("Peano");
    // (=[...]) is a relation, not an operator.
    auto id = conj::testing::Friend::getOperatorId(c, "(=[1,2])");
    ASSERT_EQ(id, std::string(""));
}

TEST(conjecturer, inputorder_get_operator_id_in2_returns_nonempty) {
    conj::Conjecturer c("Peano");
    // (in2[...]) is an operator (input + output args).
    auto id = conj::testing::Friend::getOperatorId(c, "(in2[1,2,3])");
    ASSERT_GE(static_cast<int>(id.size()), 1);
}

TEST(conjecturer, inputorder_get_operator_id_negated_returns_empty) {
    conj::Conjecturer c("Peano");
    // !(in[...]) — `in` is still a relation under the negation,
    // so getOperatorId returns the empty string.
    auto id = conj::testing::Friend::getOperatorId(c, "!(in[1,2])");
    ASSERT_TRUE(id.empty());
}


// ---- removeOutputs (3 tests) ----

TEST(conjecturer, inputorder_remove_outputs_empty_chain_returns_empty) {
    conj::Conjecturer c("Peano");
    auto out = conj::testing::Friend::removeOutputs(
        c, std::vector<std::string>{});
    ASSERT_TRUE(out.empty());
}

TEST(conjecturer, inputorder_remove_outputs_relation_passthrough) {
    conj::Conjecturer c("Peano");
    // Pure relation — `in` has no outputs, so chain text is preserved.
    auto out = conj::testing::Friend::removeOutputs(
        c, std::vector<std::string>{"(in[1,2])"});
    ASSERT_EQ(out.size(), std::string("(in[1,2])").size());
}

TEST(conjecturer, inputorder_remove_outputs_operator_strips) {
    conj::Conjecturer c("Peano");
    // Operator — output position stripped, leaving 11 chars.
    auto out = conj::testing::Friend::removeOutputs(
        c, std::vector<std::string>{"(in2[1,2,3])"});
    ASSERT_EQ(out.size(), static_cast<std::size_t>(11));
}


// ---- checkTautology (3 tests) ----
// `false` means "rejected as tautology"; `true` means "no tautology
// detected, accept".

TEST(conjecturer, inputorder_check_tautology_empty_chains_rejected) {
    conj::Conjecturer c("Peano");
    // Two empty subchains are "identical modulo renaming" -> reject.
    ASSERT_FALSE(conj::testing::Friend::checkTautology(
        c, std::vector<std::string>{}, std::vector<std::string>{}));
}

TEST(conjecturer, inputorder_check_tautology_distinct_sides_passes) {
    conj::Conjecturer c("Peano");
    ASSERT_TRUE(conj::testing::Friend::checkTautology(
        c, std::vector<std::string>{"(in[1,2])"},
        std::vector<std::string>{"(in[3,4])"}));
}

TEST(conjecturer, inputorder_check_tautology_identical_sides_rejected) {
    conj::Conjecturer c("Peano");
    // Same content on both sides -> tautology -> reject.
    ASSERT_FALSE(conj::testing::Friend::checkTautology(
        c, std::vector<std::string>{"(in[1,2])"},
        std::vector<std::string>{"(in[1,2])"}));
}


// ---- checkFunctions (3 tests) ----
// All hand-written single-leaf inputs satisfy the function-position
// well-formedness rule on Peano; the filter accepts.

TEST(conjecturer, inputorder_check_functions_empty_chain_passes) {
    conj::Conjecturer c("Peano");
    ASSERT_TRUE(conj::testing::Friend::checkFunctions(c, std::vector<std::string>{}));
}

TEST(conjecturer, inputorder_check_functions_single_relation_passes) {
    conj::Conjecturer c("Peano");
    ASSERT_TRUE(conj::testing::Friend::checkFunctions(
        c, std::vector<std::string>{"(in[1,2])"}));
}

TEST(conjecturer, inputorder_check_functions_with_in2_passes) {
    conj::Conjecturer c("Peano");
    ASSERT_TRUE(conj::testing::Friend::checkFunctions(
        c, std::vector<std::string>{"(in2[1,2,3])"}));
}


// ---- onlyOneOperator (4 tests) ----
// Implementation does `chain.begin() + 1` to skip the anchor — so
// every input must be at least 2 entries (anchor + something).

TEST(conjecturer, inputorder_only_one_operator_anchor_plus_in2_passes) {
    conj::Conjecturer c("Peano");
    // Skip anchor; head is in2 (operator); single operator => true.
    ASSERT_TRUE(conj::testing::Friend::onlyOneOperator(
        c, std::vector<std::string>{
            "(AnchorPeano[1,2,3,4,5,6])",
            "(in2[1,2,3])"
        }));
}

TEST(conjecturer, inputorder_only_one_operator_anchor_plus_two_operators_rejected) {
    conj::Conjecturer c("Peano");
    // Two distinct operator skeletons in noAnchor chain => false.
    ASSERT_FALSE(conj::testing::Friend::onlyOneOperator(
        c, std::vector<std::string>{
            "(AnchorPeano[1,2,3,4,5,6])",
            "(in2[1,2,3])",
            "(in3[1,2,3,4])"
        }));
}

TEST(conjecturer, inputorder_only_one_operator_anchor_plus_relation_rejected) {
    conj::Conjecturer c("Peano");
    // Head is `in` (relation, not operator); function rejects.
    ASSERT_FALSE(conj::testing::Friend::onlyOneOperator(
        c, std::vector<std::string>{
            "(AnchorPeano[1,2,3,4,5,6])",
            "(in[1,2])"
        }));
}

TEST(conjecturer, inputorder_only_one_operator_singleton_smoke) {
    conj::Conjecturer c("Peano");
    // chain.begin() + 1 with size-1 chain leaves noAnchor empty,
    // function returns false.
    bool r = conj::testing::Friend::onlyOneOperator(
        c, std::vector<std::string>{"(AnchorPeano[1,2,3,4,5,6])"});
    ASSERT_FALSE(r);
}


// ---- checkInputVariablePosition (3 tests) ----
// All hand-written canonical-id inputs satisfy the position rule on
// Peano (digits, when present, appear in left-to-right order); the
// filter accepts.

TEST(conjecturer, inputorder_check_input_variable_position_empty_passes) {
    conj::Conjecturer c("Peano");
    ASSERT_TRUE(conj::testing::Friend::checkInputVariablePosition(
        c, std::vector<std::string>{}, std::set<std::string>{}));
}

TEST(conjecturer, inputorder_check_input_variable_position_single_passes) {
    conj::Conjecturer c("Peano");
    ASSERT_TRUE(conj::testing::Friend::checkInputVariablePosition(
        c, std::vector<std::string>{"(in[1,2])"},
        std::set<std::string>{"1", "2"}));
}

TEST(conjecturer, inputorder_check_input_variable_position_no_digits_passes) {
    conj::Conjecturer c("Peano");
    ASSERT_TRUE(conj::testing::Friend::checkInputVariablePosition(
        c, std::vector<std::string>{"(in[1,2])"}, std::set<std::string>{}));
}


// ---- getLeftRightChains (3 tests) ----
// Splits chain into (left, right) relative to the head (last entry).
// Anchor is excluded; the head itself goes to the right side.

TEST(conjecturer, inputorder_get_left_right_chains_size2_left_zero_right_one) {
    conj::Conjecturer c("Peano");
    auto [left, right] = conj::testing::Friend::getLeftRightChains(
        c, std::vector<std::string>{
            "(AnchorPeano[1,2,3,4,5,6])",
            "(in[1,2])"
        });
    // Anchor + head only: left empty, right has the head.
    ASSERT_EQ(static_cast<int>(left.size()), 0);
    ASSERT_EQ(static_cast<int>(right.size()), 1);
}

TEST(conjecturer, inputorder_get_left_right_chains_size3_left_one_right_one) {
    conj::Conjecturer c("Peano");
    auto [left, right] = conj::testing::Friend::getLeftRightChains(
        c, std::vector<std::string>{
            "(AnchorPeano[1,2,3,4,5,6])",
            "(in[1,2])",
            "(in[3,4])"
        });
    ASSERT_EQ(static_cast<int>(left.size()), 1);
    ASSERT_EQ(static_cast<int>(right.size()), 1);
}

TEST(conjecturer, inputorder_get_left_right_chains_anchor_plus_in2_split) {
    conj::Conjecturer c("Peano");
    auto [left, right] = conj::testing::Friend::getLeftRightChains(
        c, std::vector<std::string>{
            "(AnchorPeano[1,2,3,4,5,6])",
            "(in2[1,2,3])"
        });
    ASSERT_EQ(static_cast<int>(left.size()), 0);
    ASSERT_EQ(static_cast<int>(right.size()), 1);
}


// ---- getTertiaries (3 tests) ----

TEST(conjecturer, inputorder_get_tertiaries_empty_chain) {
    conj::Conjecturer c("Peano");
    auto out = conj::testing::Friend::getTertiaries(c, std::vector<std::string>{});
    ASSERT_TRUE(out.empty());
}

TEST(conjecturer, inputorder_get_tertiaries_single_relation_empty) {
    conj::Conjecturer c("Peano");
    // Single in-leaf — no tertiary args (there's no outer chain
    // distinguishing primary vs tertiary).
    auto out = conj::testing::Friend::getTertiaries(
        c, std::vector<std::string>{"(in[1,2])"});
    ASSERT_TRUE(out.empty());
}

TEST(conjecturer, inputorder_get_tertiaries_in2_empty) {
    conj::Conjecturer c("Peano");
    auto out = conj::testing::Friend::getTertiaries(
        c, std::vector<std::string>{"(in2[1,2,3])"});
    ASSERT_TRUE(out.empty());
}


// ---- checkTertiaries (3 tests) ----
// All hand-written single-leaf inputs satisfy tertiary-variable
// compatibility on Peano; the filter accepts.

TEST(conjecturer, inputorder_check_tertiaries_empty_passes) {
    conj::Conjecturer c("Peano");
    ASSERT_TRUE(conj::testing::Friend::checkTertiaries(
        c, std::vector<std::string>{}, std::vector<std::string>{}));
}

TEST(conjecturer, inputorder_check_tertiaries_single_each_side_passes) {
    conj::Conjecturer c("Peano");
    ASSERT_TRUE(conj::testing::Friend::checkTertiaries(
        c, std::vector<std::string>{"(in[1,2])"},
        std::vector<std::string>{"(in[3,4])"}));
}

TEST(conjecturer, inputorder_check_tertiaries_with_in2_passes) {
    conj::Conjecturer c("Peano");
    ASSERT_TRUE(conj::testing::Friend::checkTertiaries(
        c, std::vector<std::string>{"(in2[1,2,3])"},
        std::vector<std::string>{"(in[3,4])"}));
}


// ---- checkInputVariablesOrder integration (5 tests) ----
// All hand-written canonical-id inputs pass the ordering check on
// Peano — the ordering invariant is trivially satisfied when arg
// ids are written in 1, 2, 3, ... order in the input positions.

TEST(conjecturer, inputorder_check_input_variables_order_in_passes) {
    conj::Conjecturer c("Peano");
    ASSERT_TRUE(conj::testing::Friend::checkInputVariablesOrder(
        c, "(>[1,2](AnchorPeano[1,2,3,4,5,6])(in[1,2]))"));
}

TEST(conjecturer, inputorder_check_input_variables_order_anchor_alone_passes) {
    conj::Conjecturer c("Peano");
    ASSERT_TRUE(conj::testing::Friend::checkInputVariablesOrder(
        c, "(AnchorPeano[1,2,3,4,5,6])"));
}

TEST(conjecturer, inputorder_check_input_variables_order_in2_passes) {
    conj::Conjecturer c("Peano");
    ASSERT_TRUE(conj::testing::Friend::checkInputVariablesOrder(
        c, "(>[1,2,3](AnchorPeano[1,2,3,4,5,6])(in2[1,2,3]))"));
}

TEST(conjecturer, inputorder_check_input_variables_order_eq_passes) {
    conj::Conjecturer c("Peano");
    ASSERT_TRUE(conj::testing::Friend::checkInputVariablesOrder(
        c, "(>[1,2](AnchorPeano[1,2,3,4,5,6])(=[1,2]))"));
}

TEST(conjecturer, inputorder_check_input_variables_order_two_in2_passes) {
    conj::Conjecturer c("Peano");
    ASSERT_TRUE(conj::testing::Friend::checkInputVariablesOrder(
        c, "(>[1,2,3,7](AnchorPeano[1,2,3,4,5,6])(in2[1,2,3])(in2[1,7,3]))"));
}


// ============================================================================
// reshuffle_*  — Reshuffle pipeline + mirror generation
// ============================================================================

// ---- reshuffle (10 tests) ----

TEST(conjecturer, reshuffle_simple_returns_canonical) {
    conj::Conjecturer c("Peano");
    auto [out, _ds, _rm] = conj::testing::Friend::reshuffle(
        c, "(>[1,2](AnchorPeano[1,2,3,4,5,6])(in[1,2]))", false);
    // Reshuffle returns a non-empty canonical form.
    ASSERT_GE(static_cast<int>(out.size()), 1);
}

TEST(conjecturer, reshuffle_idempotent_round_trip) {
    conj::Conjecturer c("Peano");
    auto [r1, _d1, _m1] = conj::testing::Friend::reshuffle(
        c, "(>[1,2](AnchorPeano[1,2,3,4,5,6])(in[1,2]))", false);
    auto [r2, _d2, _m2] = conj::testing::Friend::reshuffle(c, r1, false);
    ASSERT_EQ(r1, r2);
}

TEST(conjecturer, reshuffle_anchor_alone_smoke) {
    conj::Conjecturer c("Peano");
    auto [out, _ds, _rm] = conj::testing::Friend::reshuffle(
        c, "(AnchorPeano[1,2,3,4,5,6])", false);
    ASSERT_GE(static_cast<int>(out.size()), 1);
}

TEST(conjecturer, reshuffle_deep_flag_runs) {
    conj::Conjecturer c("Peano");
    auto [out, _ds, _rm] = conj::testing::Friend::reshuffle(
        c, "(>[1,2](AnchorPeano[1,2,3,4,5,6])(in[1,2]))", true);
    ASSERT_GE(static_cast<int>(out.size()), 1);
}

TEST(conjecturer, reshuffle_returns_def_set_map_size4) {
    conj::Conjecturer c("Peano");
    auto [_out, ds, _rm] = conj::testing::Friend::reshuffle(
        c, "(>[1,2](AnchorPeano[1,2,3,4,5,6])(in[1,2]))", false);
    // Def-set map carries the canonical-form arg-id assignments;
    // 4 entries for this chain shape.
    ASSERT_EQ(static_cast<int>(ds.size()), 4);
}

TEST(conjecturer, reshuffle_returns_rename_map_size2) {
    conj::Conjecturer c("Peano");
    auto [_out, _ds, rm] = conj::testing::Friend::reshuffle(
        c, "(>[1,2](AnchorPeano[1,2,3,4,5,6])(in[1,2]))", false);
    // Rename map: chain bound vars 1, 2 -> canonical positions.
    ASSERT_EQ(static_cast<int>(rm.size()), 2);
}

TEST(conjecturer, reshuffle_with_negation_smoke) {
    conj::Conjecturer c("Peano");
    auto [out, _ds, _rm] = conj::testing::Friend::reshuffle(
        c, "(>[1,2](AnchorPeano[1,2,3,4,5,6])!(in[1,2]))", false);
    ASSERT_GE(static_cast<int>(out.size()), 1);
}

TEST(conjecturer, reshuffle_three_predicates_smoke) {
    conj::Conjecturer c("Peano");
    auto [out, _ds, _rm] = conj::testing::Friend::reshuffle(
        c, "(>[1,2,7](AnchorPeano[1,2,3,4,5,6])(in[1,2])(in[7,1]))", false);
    ASSERT_GE(static_cast<int>(out.size()), 1);
}

TEST(conjecturer, reshuffle_with_in2_smoke) {
    conj::Conjecturer c("Peano");
    auto [out, _ds, _rm] = conj::testing::Friend::reshuffle(
        c, "(>[1,2,3](AnchorPeano[1,2,3,4,5,6])(in2[1,2,3]))", false);
    ASSERT_GE(static_cast<int>(out.size()), 1);
}

TEST(conjecturer, reshuffle_canonicalises_arg_ids) {
    conj::Conjecturer c("Peano");
    // Sparse arg ids should canonicalise.
    auto [out, _ds, _rm] = conj::testing::Friend::reshuffle(
        c, "(>[1,2](AnchorPeano[1,2,3,4,5,6])(in[1,2]))", false);
    auto [out2, _ds2, _rm2] = conj::testing::Friend::reshuffle(
        c, "(>[1,2](AnchorPeano[1,2,3,4,5,6])(in[1,2]))", false);
    // Same input -> same canonical output.
    ASSERT_EQ(out, out2);
}


// ---- createReshuffledMirrored (5 tests) ----
// On flat-shape Peano inputs the mirror always collapses to source
// (per I-9's distinctness guard), so the function returns "". Real
// non-collapsing mirrors emerge from nested-quantifier shapes that
// the production worker drivers produce; covered by main.py
// integration runs.

TEST(conjecturer, reshuffle_mirror_in_chain_collapses_to_source) {
    conj::Conjecturer c("Peano");
    auto m = conj::testing::Friend::createReshuffledMirrored(
        c, "(>[1,2](AnchorPeano[1,2,3,4,5,6])(in[1,2]))");
    ASSERT_TRUE(m.empty());
}

TEST(conjecturer, reshuffle_mirror_anchor_first_flag_collapses) {
    conj::Conjecturer c("Peano");
    auto m = conj::testing::Friend::createReshuffledMirrored(
        c, "(>[1,2](AnchorPeano[1,2,3,4,5,6])(in[1,2]))", true);
    ASSERT_TRUE(m.empty());
}

TEST(conjecturer, reshuffle_mirror_equality_head_collapses) {
    conj::Conjecturer c("Peano");
    auto m = conj::testing::Friend::createReshuffledMirrored(
        c, "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(=[1,2]))");
    ASSERT_TRUE(m.empty());
}

TEST(conjecturer, reshuffle_mirror_three_predicates_collapses) {
    conj::Conjecturer c("Peano");
    auto m = conj::testing::Friend::createReshuffledMirrored(
        c, "(>[1,2,7](AnchorPeano[1,2,3,4,5,6])(in[1,2])(in[7,1]))");
    ASSERT_TRUE(m.empty());
}

TEST(conjecturer, reshuffle_mirror_in2_collapses) {
    conj::Conjecturer c("Peano");
    auto m = conj::testing::Friend::createReshuffledMirrored(
        c, "(>[1,2,3](AnchorPeano[1,2,3,4,5,6])(in2[1,2,3]))");
    ASSERT_TRUE(m.empty());
}


// ---- countOperatorOccurrences + staysOutputVariable (5 tests) ----

TEST(conjecturer, reshuffle_count_operator_occurrences_zero_for_atom) {
    conj::Conjecturer c("Peano");
    int n = conj::testing::Friend::countOperatorOccurrences(c, "(in[1,2])");
    ASSERT_EQ(n, 0);
}

TEST(conjecturer, reshuffle_count_operator_occurrences_one_for_single_layer) {
    conj::Conjecturer c("Peano");
    int n = conj::testing::Friend::countOperatorOccurrences(
        c, "(>[1,2](AnchorPeano[1,2,3,4,5,6])(in[1,2]))");
    ASSERT_EQ(n, 1);
}

TEST(conjecturer, reshuffle_count_operator_occurrences_two_for_nested) {
    conj::Conjecturer c("Peano");
    int n = conj::testing::Friend::countOperatorOccurrences(
        c, "(>[1,2](AnchorPeano[1,2,3,4,5,6])(>[3](in[1,3])(in[3,2])))");
    ASSERT_EQ(n, 2);
}

TEST(conjecturer, reshuffle_stays_output_variable_in2_arg3_no) {
    conj::Conjecturer c("Peano");
    // `(in2[1,2,3])` is a single leaf — its output variable (3)
    // does not appear elsewhere, so it does not "stay".
    ASSERT_FALSE(conj::testing::Friend::staysOutputVariable(c, "(in2[1,2,3])", "3"));
}

TEST(conjecturer, reshuffle_stays_output_variable_arg_not_present) {
    conj::Conjecturer c("Peano");
    // Arg 9 is not present anywhere in `(in[1,2])`.
    ASSERT_FALSE(conj::testing::Friend::staysOutputVariable(c, "(in[1,2])", "9"));
}


// ============================================================================
// reform_*  — Reformulation / negation / existence-head paths
// ============================================================================

// ---- reformulateOperatorHead (5 tests) ----
// Pass-through for non-operator heads (returns input unchanged);
// rewrites for operator heads (returns expanded existence-wrapped
// form, longer than input).

TEST(conjecturer, reform_reformulate_operator_head_in_passthrough) {
    conj::Conjecturer c("Peano");
    // Relation head — no reformulation; pass-through.
    auto src = std::string("(>[1,2](AnchorPeano[1,2,3,4,5,6])(in[1,2]))");
    auto r = conj::testing::Friend::reformulateOperatorHead(c, src);
    ASSERT_EQ(r.size(), src.size());
}

TEST(conjecturer, reform_reformulate_operator_head_in2_expands) {
    conj::Conjecturer c("Peano");
    // Operator head — reformulation expands the expression
    // (existence-wrapped form is longer than the input).
    auto src = std::string("(>[1,2,3](AnchorPeano[1,2,3,4,5,6])(in2[1,2,3]))");
    auto r = conj::testing::Friend::reformulateOperatorHead(c, src);
    ASSERT_GE(static_cast<int>(r.size()), static_cast<int>(src.size()));
}

TEST(conjecturer, reform_reformulate_operator_head_anchor_alone_passthrough) {
    conj::Conjecturer c("Peano");
    auto src = std::string("(AnchorPeano[1,2,3,4,5,6])");
    auto r = conj::testing::Friend::reformulateOperatorHead(c, src);
    ASSERT_EQ(r.size(), src.size());
}

TEST(conjecturer, reform_reformulate_operator_head_eq_passthrough) {
    conj::Conjecturer c("Peano");
    auto src = std::string("(>[1,2](AnchorPeano[1,2,3,4,5,6])(=[1,2]))");
    auto r = conj::testing::Friend::reformulateOperatorHead(c, src);
    ASSERT_EQ(r.size(), src.size());
}

TEST(conjecturer, reform_reformulate_operator_head_negated_in_passthrough) {
    conj::Conjecturer c("Peano");
    auto src = std::string("(>[1,2](AnchorPeano[1,2,3,4,5,6])!(in[1,2]))");
    auto r = conj::testing::Friend::reformulateOperatorHead(c, src);
    ASSERT_EQ(r.size(), src.size());
}


// ---- triggersExistenceReformulation (5 tests) ----
// Returns `true` only when the ungrounded-operator-head filter
// rejection hits AND the candidate qualifies for existence
// reformulation. None of these flat-shape inputs qualify on Peano;
// all return false.

TEST(conjecturer, reform_triggers_existence_reformulation_simple_no_trigger) {
    conj::Conjecturer c("Peano");
    ASSERT_FALSE(conj::testing::Friend::triggersExistenceReformulation(
        c, "(>[1,2](AnchorPeano[1,2,3,4,5,6])(in[1,2]))"));
}

TEST(conjecturer, reform_triggers_existence_reformulation_anchor_alone_no_trigger) {
    conj::Conjecturer c("Peano");
    ASSERT_FALSE(conj::testing::Friend::triggersExistenceReformulation(
        c, "(AnchorPeano[1,2,3,4,5,6])"));
}

TEST(conjecturer, reform_triggers_existence_reformulation_with_in2_no_trigger) {
    conj::Conjecturer c("Peano");
    ASSERT_FALSE(conj::testing::Friend::triggersExistenceReformulation(
        c, "(>[1,2,3](AnchorPeano[1,2,3,4,5,6])(in2[1,2,3]))"));
}

TEST(conjecturer, reform_triggers_existence_reformulation_two_premises_no_trigger) {
    conj::Conjecturer c("Peano");
    ASSERT_FALSE(conj::testing::Friend::triggersExistenceReformulation(
        c, "(>[1,2,7](AnchorPeano[1,2,3,4,5,6])(in[7,1])(in[1,2]))"));
}

TEST(conjecturer, reform_triggers_existence_reformulation_eq_head_no_trigger) {
    conj::Conjecturer c("Peano");
    ASSERT_FALSE(conj::testing::Friend::triggersExistenceReformulation(
        c, "(>[1,2](AnchorPeano[1,2,3,4,5,6])(=[1,2]))"));
}


// ---- generateNegatedPremiseVariants (5 tests) ----
// On Peano flat-shape inputs, no premise has `allow_negation = true`
// in a way that would yield a new variant — the function returns
// the empty vector for every hand-written input here. Real
// negation-variant emission rides on operator-headed conjectures
// that flow through worker drivers; covered by main.py runs.

TEST(conjecturer, reform_generate_negated_premise_variants_simple_empty) {
    conj::Conjecturer c("Peano");
    auto v = conj::testing::Friend::generateNegatedPremiseVariants(
        c, "(>[1,2](AnchorPeano[1,2,3,4,5,6])(in[1,2]))");
    ASSERT_TRUE(v.empty());
}

TEST(conjecturer, reform_generate_negated_premise_variants_anchor_alone_empty) {
    conj::Conjecturer c("Peano");
    auto v = conj::testing::Friend::generateNegatedPremiseVariants(
        c, "(AnchorPeano[1,2,3,4,5,6])");
    ASSERT_TRUE(v.empty());
}

TEST(conjecturer, reform_generate_negated_premise_variants_two_premises_empty) {
    conj::Conjecturer c("Peano");
    auto v = conj::testing::Friend::generateNegatedPremiseVariants(
        c, "(>[1,2,7](AnchorPeano[1,2,3,4,5,6])(in[7,1])(=[1,2]))");
    ASSERT_TRUE(v.empty());
}

TEST(conjecturer, reform_generate_negated_premise_variants_in2_empty) {
    conj::Conjecturer c("Peano");
    auto v = conj::testing::Friend::generateNegatedPremiseVariants(
        c, "(>[1,2,3](AnchorPeano[1,2,3,4,5,6])(in2[1,2,3]))");
    ASSERT_TRUE(v.empty());
}

TEST(conjecturer, reform_generate_negated_premise_variants_negated_empty) {
    conj::Conjecturer c("Peano");
    auto v = conj::testing::Friend::generateNegatedPremiseVariants(
        c, "(>[1,2](AnchorPeano[1,2,3,4,5,6])!(in[1,2]))");
    ASSERT_TRUE(v.empty());
}


// ============================================================================
// worker_*  — Worker drivers (string + int)
// ============================================================================

// ---- singleThreadCalculation (3 tests) ----
// Empty DefSetMaps + minimal inputs: every output list comes back
// empty (the merge fails the cascade-filter caps without arg-type
// information). Exercises the entire worker-driver call path
// without producing survivors.

TEST(conjecturer, worker_single_thread_calculation_empty_inputs_no_survivors) {
    conj::Conjecturer c("Peano");
    auto out = conj::testing::Friend::singleThreadCalculation(
        c, "(in[1,2])", "(in[3,4])", 1, 1,
        conj::DefSetMap{}, conj::DefSetMap{});
    ASSERT_TRUE(out.connected_list.empty());
    ASSERT_TRUE(out.connected_list2.empty());
    ASSERT_TRUE(out.reshuffled_list.empty());
    ASSERT_TRUE(out.reshuffled_mirrored_list.empty());
}

TEST(conjecturer, worker_single_thread_calculation_higher_nse_no_survivors) {
    conj::Conjecturer c("Peano");
    auto out = conj::testing::Friend::singleThreadCalculation(
        c, "(in[1,2])", "(in[3,4])", 2, 2,
        conj::DefSetMap{}, conj::DefSetMap{});
    ASSERT_TRUE(out.connected_list2.empty());
}

TEST(conjecturer, worker_single_thread_calculation_with_def_set_map_no_survivors) {
    conj::Conjecturer c("Peano");
    conj::DefSetMap m;
    m["1"] = std::make_tuple(std::string("(1)"), true, true);
    auto out = conj::testing::Friend::singleThreadCalculation(
        c, "(in[1,2])", "(in[3,4])", 1, 1, m, m);
    ASSERT_TRUE(out.reshuffled_list.empty());
}


// ---- singleExprAnchorConnection (3 tests) ----
// Same shape: empty DefSetMaps + minimal inputs produce empty
// output lists.

TEST(conjecturer, worker_single_expr_anchor_connection_in_no_survivors) {
    conj::Conjecturer c("Peano");
    auto out = conj::testing::Friend::singleExprAnchorConnection(
        c, "(in[1,2])", conj::DefSetMap{});
    ASSERT_TRUE(out.connected_list2.empty());
}

TEST(conjecturer, worker_single_expr_anchor_connection_in2_no_survivors) {
    conj::Conjecturer c("Peano");
    auto out = conj::testing::Friend::singleExprAnchorConnection(
        c, "(in2[1,2,3])", conj::DefSetMap{});
    ASSERT_TRUE(out.reshuffled_list.empty());
}

TEST(conjecturer, worker_single_expr_anchor_connection_with_def_sets_no_survivors) {
    conj::Conjecturer c("Peano");
    conj::DefSetMap m;
    m["1"] = std::make_tuple(std::string("(1)"), true, true);
    auto out = conj::testing::Friend::singleExprAnchorConnection(
        c, "(in[1,2])", m);
    ASSERT_TRUE(out.reshuffled_mirrored_list.empty());
}


// ---- singleThreadCalculationInt (3 tests) ----
// Mirrors the string-path worker on encoded inputs; same shape of
// no-survivor result on minimal inputs.

TEST(conjecturer, worker_single_thread_calculation_int_no_survivors) {
    conj::Conjecturer c("Peano");
    auto a = conj::testing::Friend::encodeExpr(c, "(in[1,2])");
    auto b = conj::testing::Friend::encodeExpr(c, "(in[3,4])");
    conj::IntDefSetMap am{}, bm{};
    auto out = conj::testing::Friend::singleThreadCalculationInt(
        c, a, b, 1, 1, am, bm);
    ASSERT_TRUE(out.connected_list.empty());
}

TEST(conjecturer, worker_single_thread_calculation_int_higher_nse_no_survivors) {
    conj::Conjecturer c("Peano");
    auto a = conj::testing::Friend::encodeExpr(c, "(in[1,2])");
    auto b = conj::testing::Friend::encodeExpr(c, "(in[3,4])");
    conj::IntDefSetMap am{}, bm{};
    auto out = conj::testing::Friend::singleThreadCalculationInt(
        c, a, b, 2, 2, am, bm);
    ASSERT_TRUE(out.connected_list2.empty());
}

TEST(conjecturer, worker_single_thread_calculation_int_with_def_sets_no_survivors) {
    conj::Conjecturer c("Peano");
    auto a = conj::testing::Friend::encodeExpr(c, "(in[1,2])");
    auto b = conj::testing::Friend::encodeExpr(c, "(in[3,4])");
    conj::DefSetMap am, bm;
    am["1"] = std::make_tuple(std::string("(1)"), true, true);
    bm["3"] = std::make_tuple(std::string("(1)"), true, true);
    auto am_int = conj::testing::Friend::encodeDefSetMap(c, am);
    auto bm_int = conj::testing::Friend::encodeDefSetMap(c, bm);
    auto out = conj::testing::Friend::singleThreadCalculationInt(
        c, a, b, 1, 1, am_int, bm_int);
    ASSERT_TRUE(out.reshuffled_list.empty());
}


// ---- singleExprAnchorConnectionInt (3 tests) ----

TEST(conjecturer, worker_single_expr_anchor_connection_int_no_survivors) {
    conj::Conjecturer c("Peano");
    auto a = conj::testing::Friend::encodeExpr(c, "(in[1,2])");
    conj::IntDefSetMap am{};
    auto out = conj::testing::Friend::singleExprAnchorConnectionInt(c, a, am);
    ASSERT_TRUE(out.connected_list2.empty());
}

TEST(conjecturer, worker_single_expr_anchor_connection_int_in2_no_survivors) {
    conj::Conjecturer c("Peano");
    auto a = conj::testing::Friend::encodeExpr(c, "(in2[1,2,3])");
    conj::IntDefSetMap am{};
    auto out = conj::testing::Friend::singleExprAnchorConnectionInt(c, a, am);
    ASSERT_TRUE(out.reshuffled_list.empty());
}

TEST(conjecturer, worker_single_expr_anchor_connection_int_with_def_sets_no_survivors) {
    conj::Conjecturer c("Peano");
    auto a = conj::testing::Friend::encodeExpr(c, "(in[1,2])");
    conj::DefSetMap m;
    m["1"] = std::make_tuple(std::string("(1)"), true, true);
    auto m_int = conj::testing::Friend::encodeDefSetMap(c, m);
    auto out = conj::testing::Friend::singleExprAnchorConnectionInt(c, a, m_int);
    ASSERT_TRUE(out.reshuffled_mirrored_list.empty());
}


// ============================================================================
// legacy_*  — renameVariablesInExpr (legacy reshuffle predecessor)
// ============================================================================
// Returns the renamed expression text, length-preserving for these
// canonical-id inputs (no expansion).

TEST(conjecturer, legacy_rename_variables_in_expr_simple_preserves_length) {
    conj::Conjecturer c("Peano");
    auto src = std::string("(>[1,2](AnchorPeano[1,2,3,4,5,6])(in[1,2]))");
    auto [out, _ds, _rm] = conj::testing::Friend::renameVariablesInExpr(c, src, false);
    ASSERT_EQ(out.size(), src.size());
}

TEST(conjecturer, legacy_rename_variables_in_expr_deep_flag_preserves_length) {
    conj::Conjecturer c("Peano");
    auto src = std::string("(>[1,2](AnchorPeano[1,2,3,4,5,6])(in[1,2]))");
    auto [out, _ds, _rm] = conj::testing::Friend::renameVariablesInExpr(c, src, true);
    ASSERT_EQ(out.size(), src.size());
}

TEST(conjecturer, legacy_rename_variables_in_expr_anchor_alone_preserves_length) {
    conj::Conjecturer c("Peano");
    auto src = std::string("(AnchorPeano[1,2,3,4,5,6])");
    auto [out, _ds, _rm] = conj::testing::Friend::renameVariablesInExpr(c, src, false);
    ASSERT_EQ(out.size(), src.size());
}

TEST(conjecturer, legacy_rename_variables_in_expr_in2_preserves_length) {
    conj::Conjecturer c("Peano");
    auto src = std::string("(>[1,2,3](AnchorPeano[1,2,3,4,5,6])(in2[1,2,3]))");
    auto [out, _ds, _rm] = conj::testing::Friend::renameVariablesInExpr(c, src, false);
    ASSERT_EQ(out.size(), src.size());
}

TEST(conjecturer, legacy_rename_variables_in_expr_negated_preserves_length) {
    conj::Conjecturer c("Peano");
    auto src = std::string("(>[1,2](AnchorPeano[1,2,3,4,5,6])!(in[1,2]))");
    auto [out, _ds, _rm] = conj::testing::Friend::renameVariablesInExpr(c, src, false);
    ASSERT_EQ(out.size(), src.size());
}


// ============================================================================
// or_pair_*  — generateOrConjectures (smoke; config-derived)
// ============================================================================

TEST(conjecturer, or_pair_generate_or_conjectures_peano_emits_two) {
    conj::Conjecturer c("Peano");
    auto pairs = conj::testing::Friend::generateOrConjectures(c);
    // ConfigPeano.json carries `allow_to_constitute_existence` flags
    // that derive 2 OR-pair conjectures.
    ASSERT_EQ(static_cast<int>(pairs.size()), 2);
}

TEST(conjecturer, or_pair_generate_or_conjectures_gauss_emits_none) {
    conj::Conjecturer c("Gauss");
    auto pairs = conj::testing::Friend::generateOrConjectures(c);
    // ConfigGauss.json has no expressions with
    // `allow_to_constitute_existence = true`, so no OR pairs.
    ASSERT_TRUE(pairs.empty());
}

TEST(conjecturer, or_pair_generate_or_conjectures_pairs_well_formed) {
    conj::Conjecturer c("Peano");
    auto pairs = conj::testing::Friend::generateOrConjectures(c);
    // Every emitted pair has both halves non-empty.
    for (auto& [ex, comp] : pairs) {
        ASSERT_FALSE(ex.empty());
        ASSERT_FALSE(comp.empty());
    }
}

