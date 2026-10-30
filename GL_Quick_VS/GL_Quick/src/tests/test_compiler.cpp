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
/// @brief Phase-4 unit tests for the `ce::` namespace declared in compiler.hpp
/// and the bodies in compiler.cpp.
///
/// @details
/// Coverage:
///   - generateBinarySequencesAsLists boundary cases (n = 0, 1, 2, 4)
///     plus negative input handling.
///   - generateAllPermutations: lex order within each fixed-point group.
///   - getArgs on simple, nested, and empty input.
///   - extractExpression / extractExpressionFromNegation /
///     extractExpressionUniversal across the canonical and negated shapes.
///   - makeAnchorSignature shape (with arity = 0, 1, 4).
///   - findAnchorKey on a synthetic CoreExpressionConfig map.
///   - expressionIsSimple discriminating atomic vs compound forms.

#include "test_harness.hpp"

#include "../compiler.hpp"
#include "../compiler/expression_parser.hpp"
#include "../compiler/compilation_normalizer.hpp"
#include "../compiler/compilation_reuse.hpp"
#include "../compiler/definition_expansion.hpp"
#include "../proof/worker_context.hpp"
#include <thread>

#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include "../project_paths.hpp"
#include <utility>
#include <algorithm>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace {
    // 1 MiB pool / 256 KiB block for the scratch-arena twin tests.
    const gl::StaticMemoryConfig kCompilerTestCfg{ 1 << 20, 1 << 18 };
}

/// @brief Verbatim retained oracle for the deleted `ce::extractKeyValue`
///        (pair-producing), still calling the surviving `ce::parseExpr` /
///        `ce::treeToExpr` / `ce::deleteTree`.
///
/// @details
/// Byte-identity reference for `extract_key_value_key_scratch_matches_heap`
/// (per D-193 a prover-only heap `ce::` function is
/// deleted, its Rule-18 oracle retained test-locally). The original's unused
/// `coreExpressionMap` parameter is dropped — the body never referenced it.
///
/// @param expr2 Canonical MPL implication text.
/// @return The `(key, value)` pair; only `.first` is compared by the twin test.
inline std::pair<std::string, std::string>
extractKeyValueOracle(const std::string& expr2) {
    ce::TreeNode1* root = ce::parseExpr(expr2);

    std::string value;
    ce::TreeNode1* node = root;
    while (node != NULL) {
        if (!node->value.empty() && node->value[0] == '>') {
            node = node->right;
        }
        else {
            value = ce::treeToExpr(node);
            break;
        }
    }

    std::string key;
    if (value.empty()) {
        key = expr2;
    }
    else {
        const std::size_t pos = expr2.rfind(value);
        if (pos != std::string::npos) {
            key = expr2.substr(0, pos) + expr2.substr(pos + value.size());
        }
        else {
            key = expr2;
        }
    }

    ce::deleteTree(root);
    return std::make_pair(key, value);
}

TEST(compiler, generate_binary_sequences_zero_has_one_empty) {
    const auto seqs = ce::generateBinarySequencesAsLists(0);
    ASSERT_EQ(seqs.size(),       static_cast<std::size_t>(1));
    ASSERT_EQ(seqs[0].size(),    static_cast<std::size_t>(0));
}

TEST(compiler, generate_binary_sequences_one_has_two) {
    const auto seqs = ce::generateBinarySequencesAsLists(1);
    ASSERT_EQ(seqs.size(),       static_cast<std::size_t>(2));
    ASSERT_EQ(seqs[0].size(),    static_cast<std::size_t>(1));
    ASSERT_EQ(seqs[0][0],        0);
    ASSERT_EQ(seqs[1][0],        1);
}

TEST(compiler, generate_binary_sequences_two) {
    const auto seqs = ce::generateBinarySequencesAsLists(2);
    ASSERT_EQ(seqs.size(),       static_cast<std::size_t>(4));
    // MSB-first: 00, 01, 10, 11.
    ASSERT_EQ(seqs[0][0], 0); ASSERT_EQ(seqs[0][1], 0);
    ASSERT_EQ(seqs[1][0], 0); ASSERT_EQ(seqs[1][1], 1);
    ASSERT_EQ(seqs[2][0], 1); ASSERT_EQ(seqs[2][1], 0);
    ASSERT_EQ(seqs[3][0], 1); ASSERT_EQ(seqs[3][1], 1);
}

TEST(compiler, generate_binary_sequences_four) {
    const auto seqs = ce::generateBinarySequencesAsLists(4);
    ASSERT_EQ(seqs.size(), static_cast<std::size_t>(16));
    for (const auto& row : seqs) {
        ASSERT_EQ(row.size(), static_cast<std::size_t>(4));
    }
}

TEST(compiler, generate_binary_sequences_negative_returns_empty) {
    const auto seqs = ce::generateBinarySequencesAsLists(-1);
    ASSERT_EQ(seqs.size(), static_cast<std::size_t>(0));
}

TEST(compiler, generate_all_permutations_basic) {
    const auto m = ce::generateAllPermutations(3);
    // m[3] holds permutations of {0,1,2}: 3! = 6 entries, lex order.
    ASSERT_TRUE(m.find(3) != m.end());
    const auto& p3 = m.at(3);
    ASSERT_EQ(p3.size(), static_cast<std::size_t>(6));
    // First permutation is identity; last is reverse.
    ASSERT_EQ(p3[0][0], 0); ASSERT_EQ(p3[0][1], 1); ASSERT_EQ(p3[0][2], 2);
    ASSERT_EQ(p3[5][0], 2); ASSERT_EQ(p3[5][1], 1); ASSERT_EQ(p3[5][2], 0);
}

TEST(compiler, getargs_simple) {
    const auto args = ce::getArgs("(p[a,b,c])");
    ASSERT_EQ(args.size(), static_cast<std::size_t>(3));
    ASSERT_EQ(args[0],     std::string("a"));
    ASSERT_EQ(args[1],     std::string("b"));
    ASSERT_EQ(args[2],     std::string("c"));
}

TEST(compiler, getargs_is_shallow_not_bracket_balanced) {
    // getArgs is a flat parser by design. Nested-expression input gets
    // mis-parsed because `find(']', start)` matches the INNER `]`.
    // Pinning this behavior so a future "fix" doesn't silently rewrite
    // every call site that already presupposes shallow input.
    const auto args = ce::getArgs("(p[(q[a,b]),c])");
    ASSERT_EQ(args.size(), static_cast<std::size_t>(2));
    ASSERT_EQ(args[0],     std::string("(q[a"));
    ASSERT_EQ(args[1],     std::string("b"));
    // The trailing ",c]" is dropped because the substring stops at the
    // inner `]`. Callers that need balanced parsing route through
    // `parseExpr` (recursive descent) instead.
}

TEST(compiler, getargs_zero_args) {
    // Predicate with zero args: getArgs returns empty vector.
    const auto args = ce::getArgs("(zero[])");
    ASSERT_EQ(args.size(), static_cast<std::size_t>(0));
}

TEST(compiler, extractexpression_simple) {
    ASSERT_EQ(ce::extractExpression("(=[a,b])"),       std::string("="));
    ASSERT_EQ(ce::extractExpression("(in2[x,y,z])"),   std::string("in2"));
    ASSERT_EQ(ce::extractExpression("(AnchorPeano[1,2,3,4])"),
              std::string("AnchorPeano"));
}

TEST(compiler, extractexpressionfromnegation) {
    ASSERT_EQ(ce::extractExpressionFromNegation("!(=[a,b])"),
              std::string("="));
    ASSERT_EQ(ce::extractExpressionFromNegation("!(in[x,y])"),
              std::string("in"));
}

TEST(compiler, extractexpressionuniversal_dispatches_correctly) {
    // Not-negated dispatches to extractExpression.
    ASSERT_EQ(extractExpressionUniversalOracle("(p[a])"), std::string("p"));
    // Negated dispatches to extractExpressionFromNegation.
    ASSERT_EQ(extractExpressionUniversalOracle("!(p[a])"), std::string("p"));
}

// Row 238: the span twin disintegrateImplicationSpans reproduces the heap
// disintegrateImplication's head and each triple's key (get<0>) + bound vars
// (get<1>) byte-for-byte, across positive/negated premises, nested implications,
// multi/empty bound-var lists, and a bare head. get<2> (leftArgs) is
// intentionally not compared (never read in-tree).
TEST(compiler, disintegrate_implication_spans_matches_heap) {
    ce::CoreExpressionMap cem;  // unused by both iterative walkers
    const std::vector<std::string> cases = {
        "(>[a,b](P[a,b])(Q[a]))",                    // one layer, positive premise
        "(>[x](>[y](R[x,y])(S[y]))(T[x]))",          // nested implication
        "(>[a,b]!(P[a,b])(=[a,b]))",                 // negated premise, equality head
        "(P[a])",                                     // bare head, no layers
        "(>[m,n,o](A[m,n,o])(>[p](B[p])(C[m,p])))",  // multi-bv + nested
        "(>[](Z[])(H[]))",                            // empty bound-var list
    };
    for (const std::string& expr : cases) {
        std::vector<std::tuple<std::string, std::vector<std::string>,
            std::set<std::string>>> oracleChain;
        const std::string oracleHead =
            ce::disintegrateImplication(expr, oracleChain, cem);

        std::vector<std::string> twinKeys;
        std::vector<std::vector<std::string>> twinBvs;
        gl::StrSpan twinHeadSpan;
        ce::disintegrateImplicationSpans(gl::StrSpan(expr), twinHeadSpan,
            [&twinKeys, &twinBvs](gl::StrSpan keySpan,
                const gl::StrSpan* bvSpans, int32_t bvN) {
                twinKeys.push_back(keySpan.toStdString());
                std::vector<std::string> bv;
                for (int32_t k = 0; k < bvN; ++k) bv.push_back(bvSpans[k].toStdString());
                twinBvs.push_back(std::move(bv));
            });

        ASSERT_EQ(twinHeadSpan.toStdString(), oracleHead);
        ASSERT_EQ(twinKeys.size(), oracleChain.size());
        ASSERT_EQ(twinBvs.size(), oracleChain.size());
        for (std::size_t i = 0; i < oracleChain.size(); ++i) {
            ASSERT_EQ(twinKeys[i], std::get<0>(oracleChain[i]));
            ASSERT_EQ(twinBvs[i], std::get<1>(oracleChain[i]));
        }
    }
}

TEST(compiler, extract_key_value_key_scratch_matches_heap) {
    gl::GlobalMemoryManager m;
    m.init(kCompilerTestCfg);
    gl::ScratchArena sa; sa.bind(&m);

    const std::vector<std::string> cases = {
        "(>[a,b](P[a,b])(Q[a]))",                    // single implication
        "(>[x](>[y](R[x,y])(S[y]))(T[x]))",          // multi-rung chain (nested)
        "(>[a,b]!(P[a,b])(=[a,b]))",                 // negated premise
        "(>[m,n,o](A[m,n,o])(>[p](B[p])(C[m,p])))",  // nested consequent
        "(>[a](P[a])(P[a]))",                        // head substring also in a premise -> rfind LAST occurrence
        "(P[a])",                                     // bare non-implication atom -> empty key
    };
    for (const std::string& expr : cases) {
        const std::pair<std::string, std::string> oracle = extractKeyValueOracle(expr);
        gl::ScratchScope scope(sa);
        const gl::ScratchString twin =
            ce::extractKeyValueKeyScratch(gl::StrSpan(expr), sa);
        ASSERT_EQ(gl::StrSpan(twin).toStdString(), oracle.first);
    }
}

TEST(compiler, makeanchorsignature_shape) {
    ASSERT_EQ(ce::makeAnchorSignature("AnchorPeano", 0),
              std::string("(AnchorPeano[])"));
    ASSERT_EQ(ce::makeAnchorSignature("AnchorPeano", 1),
              std::string("(AnchorPeano[1])"));
    ASSERT_EQ(ce::makeAnchorSignature("AnchorPeano", 4),
              std::string("(AnchorPeano[1,2,3,4])"));
}

TEST(compiler, makeanchorsignature_negative_throws) {
    ASSERT_THROW(ce::makeAnchorSignature("X", -1),
                 std::invalid_argument);
}

TEST(compiler, findanchorkey_on_synthetic_map) {
    ce::CoreExpressionMap m;
    m["="]       = ce::CoreExpressionConfig(2, std::string{}, "(=[1,2])");
    m["AnchorX"] = ce::CoreExpressionConfig(3, std::string{}, "(AnchorX[1,2,3])");
    m["in"]      = ce::CoreExpressionConfig(2, std::string{}, "(in[1,2])");
    ASSERT_EQ(ce::findAnchorKey(m), std::string("AnchorX"));
}

TEST(compiler, findanchorkey_returns_empty_when_no_anchor) {
    ce::CoreExpressionMap m;
    m["="]  = ce::CoreExpressionConfig(2, std::string{}, "(=[1,2])");
    m["in"] = ce::CoreExpressionConfig(2, std::string{}, "(in[1,2])");
    ASSERT_EQ(ce::findAnchorKey(m), std::string(""));
}

TEST(compiler, expressionissimple_discriminates_compound) {
    // Atomic predicate is simple.
    ASSERT_TRUE(ce::expressionIsSimple("(p[a,b])"));
    ASSERT_TRUE(ce::expressionIsSimple("(in[x])"));
    // Compound — implication arrow inside.
    ASSERT_FALSE(ce::expressionIsSimple("(>[v](p[v])(q[v]))"));
}

// =============================================================================
// Round 2 — two more tests per documented function. Edge cases, boundary
// conditions, and idempotence properties not exercised above.
// =============================================================================

// ---------- ce::generateBinarySequencesAsLists ----------
TEST(compiler, generate_binary_sequences_three) {
    // n = 3 → 2^3 = 8 sequences, each length 3, MSB-first.
    const auto seqs = ce::generateBinarySequencesAsLists(3);
    ASSERT_EQ(seqs.size(), static_cast<std::size_t>(8));
    // First and last are the all-zeros and all-ones rows.
    ASSERT_EQ(seqs[0][0], 0); ASSERT_EQ(seqs[0][1], 0); ASSERT_EQ(seqs[0][2], 0);
    ASSERT_EQ(seqs[7][0], 1); ASSERT_EQ(seqs[7][1], 1); ASSERT_EQ(seqs[7][2], 1);
}

TEST(compiler, generate_binary_sequences_deterministic) {
    // Two successive calls return byte-identical output — important for
    // reproducibility across LBs and threads.
    const auto a = ce::generateBinarySequencesAsLists(4);
    const auto b = ce::generateBinarySequencesAsLists(4);
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        ASSERT_EQ(a[i].size(), b[i].size());
        for (std::size_t j = 0; j < a[i].size(); ++j) {
            ASSERT_EQ(a[i][j], b[i][j]);
        }
    }
}

// ---------- ce::generateAllPermutations ----------
TEST(compiler, generate_all_permutations_zero_has_only_empty_key) {
    // n = 0 → only key 0 with the empty permutation slot. Per the body,
    // allPermutations[0] is an empty vector (no permutations of size 0).
    const auto m = ce::generateAllPermutations(0);
    ASSERT_TRUE(m.find(0) != m.end());
    ASSERT_EQ(m.at(0).size(), static_cast<std::size_t>(0));
}

TEST(compiler, generate_all_permutations_one) {
    // n = 1 → key 1 carries exactly one permutation, the singleton {0}.
    const auto m = ce::generateAllPermutations(1);
    ASSERT_TRUE(m.find(1) != m.end());
    ASSERT_EQ(m.at(1).size(),     static_cast<std::size_t>(1));
    ASSERT_EQ(m.at(1)[0].size(),  static_cast<std::size_t>(1));
    ASSERT_EQ(m.at(1)[0][0],      0);
}

// ---------- ce::getArgs ----------
TEST(compiler, getargs_no_brackets_returns_empty) {
    // No `[` at all → empty vector.
    const auto args = ce::getArgs("plain_no_brackets");
    ASSERT_EQ(args.size(), static_cast<std::size_t>(0));
}

TEST(compiler, getargs_single_arg) {
    // Single arg, no comma.
    const auto args = ce::getArgs("(p[onlyarg])");
    ASSERT_EQ(args.size(),  static_cast<std::size_t>(1));
    ASSERT_EQ(args[0],      std::string("onlyarg"));
}

// ---------- ce::extractExpression ----------
TEST(compiler, extractexpression_zero_arity) {
    ASSERT_EQ(ce::extractExpression("(zero[])"), std::string("zero"));
}

TEST(compiler, extractexpression_long_name) {
    ASSERT_EQ(ce::extractExpression("(EnumerationSet2[a,b,c])"),
              std::string("EnumerationSet2"));
}

// ---------- ce::extractExpressionFromNegation ----------
TEST(compiler, extractexpressionfromnegation_zero_arity) {
    ASSERT_EQ(ce::extractExpressionFromNegation("!(zero[])"),
              std::string("zero"));
}

TEST(compiler, extractexpressionfromnegation_long_name) {
    ASSERT_EQ(ce::extractExpressionFromNegation("!(EnumerationSet2[a,b])"),
              std::string("EnumerationSet2"));
}

// ---------- extractExpressionUniversalOracle (retired ce:: function) ----------
TEST(compiler, extractexpressionuniversal_zero_arity_negated) {
    ASSERT_EQ(extractExpressionUniversalOracle("!(zero[])"),
              std::string("zero"));
}

TEST(compiler, extractexpressionuniversal_anchor_form) {
    // Anchor expression — non-negated dispatch path.
    ASSERT_EQ(extractExpressionUniversalOracle("(AnchorPeano[1,2,3])"),
              std::string("AnchorPeano"));
}

// ---------- ce::makeAnchorSignature ----------
TEST(compiler, makeanchorsignature_arity_two) {
    ASSERT_EQ(ce::makeAnchorSignature("Op", 2),
              std::string("(Op[1,2])"));
}

TEST(compiler, makeanchorsignature_idempotent_on_round_trip) {
    // The signature can be parsed back: extractExpression reads name,
    // getArgs reads slot list.
    const std::string sig = ce::makeAnchorSignature("AnchorGauss", 3);
    ASSERT_EQ(ce::extractExpression(sig), std::string("AnchorGauss"));
    const auto args = ce::getArgs(sig);
    ASSERT_EQ(args.size(), static_cast<std::size_t>(3));
    ASSERT_EQ(args[0], std::string("1"));
    ASSERT_EQ(args[1], std::string("2"));
    ASSERT_EQ(args[2], std::string("3"));
}

// ---------- ce::findAnchorKey ----------
TEST(compiler, findanchorkey_picks_first_in_lexicographic_order) {
    // std::map iterates sorted; "Anchor1" sorts before "AnchorZ".
    ce::CoreExpressionMap m;
    m["AnchorZ"] = ce::CoreExpressionConfig(2, std::string{}, "(AnchorZ[1,2])");
    m["Anchor1"] = ce::CoreExpressionConfig(2, std::string{}, "(Anchor1[1,2])");
    ASSERT_EQ(ce::findAnchorKey(m), std::string("Anchor1"));
}

TEST(compiler, findanchorkey_ignores_anchor_substring_in_middle) {
    // findAnchorKey requires the prefix to start with "Anchor", not contain it.
    ce::CoreExpressionMap m;
    m["myAnchorThing"] = ce::CoreExpressionConfig(1, std::string{}, "(myAnchorThing[1])");
    ASSERT_EQ(ce::findAnchorKey(m), std::string(""));
}

// ---------- ce::expressionIsSimple ----------
TEST(compiler, expressionissimple_negated_atomic_is_still_simple) {
    // The implementation flags ONLY compound forms `(>...)` and
    // `(&...)` (and their negated counterparts). A negated atomic
    // predicate `!(p[a])` is not a compound and is therefore reported
    // as simple. This pins the per-implementation contract.
    ASSERT_TRUE(ce::expressionIsSimple("!(p[a])"));
}

TEST(compiler, expressionissimple_with_conjunction) {
    // (&...) compound is not simple.
    ASSERT_FALSE(ce::expressionIsSimple("(&(p[a])(q[b]))"));
    // Negated conjunction also not simple.
    ASSERT_FALSE(ce::expressionIsSimple("!(&(p[a])(q[b]))"));
}

TEST(project_paths, deployed_layout_and_dot_components) {
    const auto root = gl::projectRoot();
    const auto native = root / "GL_Quick_VS" / "GL_Quick";
    ASSERT_EQ(gl::projectRootFromExecutable(native / "gl_quick.exe"), root);
    ASSERT_EQ(gl::projectRootFromExecutable(native / "gl_quick"), root);
    ASSERT_EQ(gl::projectRootFromExecutable(native / "unused" / ".." / "gl_quick.exe"), root);
}

TEST(project_paths, runtime_root_owns_definition_paths) {
    ASSERT_TRUE(std::filesystem::equivalent(gl::projectRoot(), std::filesystem::current_path()));
    ASSERT_EQ(ce::DEFINITIONS_FOLDER, gl::projectRoot() / "files" / "definitions");
    ASSERT_TRUE(std::filesystem::is_regular_file(gl::projectRoot() / "files" / "config" / "ConfigPeano.json"));
}

TEST(project_paths, invalid_layout_assertion_probe) {
    // The external relocation test selects each assertion in a child process.
    std::string mode;
#ifdef _WIN32
    char* value = nullptr;
    size_t length = 0;
    const errno_t result = _dupenv_s(&value, &length, "GL_TEST_INVALID_PROJECT_ROOT");
    ASSERT_EQ(result, 0);
    if (value != nullptr) { mode = value; std::free(value); }
#else
    if (const char* value = std::getenv("GL_TEST_INVALID_PROJECT_ROOT")) mode = value;
#endif
    if (mode == "relative") {
        gl::projectRootFromExecutable("GL_Quick_VS/GL_Quick/gl_quick");
        ASSERT_TRUE(false);
    } else if (mode == "layout") {
        gl::projectRootFromExecutable(gl::projectRoot() / "wrong" / "gl_quick");
        ASSERT_TRUE(false);
    } else if (mode == "filename") {
        gl::projectRootFromExecutable(gl::projectRoot() / "GL_Quick_VS" / "GL_Quick" / "renamed");
        ASSERT_TRUE(false);
    } else {
        ASSERT_TRUE(mode.empty());
    }
}

TEST(expression_parser, parses_tree_and_preserves_argument_sets) {
    auto* tree = gl::ExpressionParser::parse(" (>[x] (p[x]) (q[x])) ");
    ASSERT_TRUE(tree != nullptr);
    ASSERT_EQ(ce::treeToExpr(tree), std::string("(>[x](p[x])(q[x]))"));
    ASSERT_TRUE(tree->left->arguments.count("x") == 1);
    ASSERT_TRUE(tree->right->arguments.count("x") == 1);
    ce::deleteTree(tree);
}

TEST(expression_parser, rejects_empty_input) {
    bool rejected = false;
    try { auto* tree = gl::ExpressionParser::parse(""); ce::deleteTree(tree); }
    catch (const std::runtime_error&) { rejected = true; }
    ASSERT_TRUE(rejected);
}

TEST(expression_parser, normalizes_premises_and_splits_only_positive_heads) {
    const std::string input = "(>[x](&(p[x])(q[x]))(r[x]))";
    ASSERT_EQ(gl::ExpressionParser::smoothenExpr(input),
              std::string("(>[x](p[x])(>[](q[x])(r[x])))"));
    const auto heads = gl::ExpressionParser::groomExpr("(>[x](p[x])(&(q[x])(r[x])))");
    ASSERT_EQ(heads.size(), std::size_t(2));
    ASSERT_EQ(heads[0], std::string("(>[x](p[x])(q[x]))"));
    ASSERT_EQ(heads[1], std::string("(>[x](p[x])(r[x]))"));
    const auto negative = gl::ExpressionParser::groomExpr("!(&(p[x])(q[x]))");
    ASSERT_EQ(negative.size(), std::size_t(1));
    ASSERT_EQ(negative[0], std::string("!(&(p[x])(q[x]))"));
}

TEST(expression_parser, reads_and_renames_outer_binders) {
    const auto args = gl::ExpressionParser::listLastRemovedArgs("(>[x,y](p[x])(q[y]))");
    ASSERT_EQ(args.size(), std::size_t(2));
    ASSERT_EQ(args[0], std::string("x"));
    ASSERT_EQ(args[1], std::string("y"));
    ASSERT_TRUE(gl::ExpressionParser::listLastRemovedArgs("(p[x])").empty());
    const auto renamed = gl::ExpressionParser::renameLastRemoved("!(>[x](p[x])!(q[x]))", 7);
    ASSERT_EQ(std::get<0>(renamed), std::string("!(>[comp_7](p[comp_7])!(q[comp_7]))"));
    ASSERT_EQ(std::get<1>(renamed), 8);
    ASSERT_EQ(std::get<2>(renamed), std::string("comp_7"));
}

TEST(compilation_normalizer, preserves_signed_keys_and_argument_modes) {
    const std::vector<gl::EncodedExpression> input{
        gl::EncodedExpression("(p[x,u_y])", "main"),
        gl::EncodedExpression("!(q[x,z])", "main")};
    const auto literal = gl::CompilationNormalizer::makeNormalizedEncodedKey(input, 1);
    ASSERT_EQ(literal.first.numberExpressions, 2);
    ASSERT_TRUE((literal.first.data == std::vector<std::string>{"p","False","1","u_y","q","True","1","2"}));
    ASSERT_EQ(literal.second.at("x"), std::string("1"));
    ASSERT_EQ(literal.second.at("z"), std::string("2"));
    const auto separate = gl::CompilationNormalizer::makeNormalizedEncodedKey(input, 2);
    ASSERT_EQ(separate.first.data[3], std::string("u_1"));
    ASSERT_EQ(separate.second.at("u_y"), std::string("u_1"));
    const auto together = gl::CompilationNormalizer::makeNormalizedEncodedKey(input, 0);
    ASSERT_EQ(together.first.data[3], std::string("2"));
    ASSERT_EQ(together.first.data[7], std::string("3"));
    ASSERT_TRUE(gl::CompilationNormalizer::makeNormalizedEncodedKey({}).first.data.empty());
}

TEST(compilation_normalizer, composes_mappings_and_orders_retained_arguments) {
    const auto composed = gl::CompilationNormalizer::combineUnchangeablePaths(
        {{"x","u_2"},{"y","7"}}, {{"u_2","u_1"}});
    ASSERT_EQ(composed.at("x"), std::string("u_1"));
    ASSERT_EQ(composed.at("y"), std::string("7"));
    const auto reverse = gl::CompilationNormalizer::createReverseUnchangeableMap(
        {{"x","u_3"},{"y","8"}}, {{"x","u_1"},{"y","2"}});
    ASSERT_EQ(reverse.size(), std::size_t(1));
    ASSERT_EQ(reverse.at("u_1"), std::string("u_3"));
    ASSERT_EQ(gl::CompilationNormalizer::createArgList(
        {{"u_alpha","u_2"},{"u_beta","u_1"},{"unused","3"}}),
        std::string("[beta,alpha]"));
    ASSERT_EQ(gl::CompilationNormalizer::createArgList({}), std::string("[]"));
}

namespace {
// A random body over three operators (fixed arities) and six variables, three
// bound and three u_, kept as indices so that it can be renamed exactly.
struct CanonicalTestElement { int name; bool negation; std::vector<int> variables; };

const char* const kCanonicalTestNames[4] = {"p", "q", "r", "s"};
const int kCanonicalTestArity[4] = {2, 3, 1, 2};

std::vector<CanonicalTestElement> canonicalTestBody(std::mt19937& rng, int premises, bool withHead) {
    std::vector<CanonicalTestElement> body;
    for (int i = 0; i < premises + (withHead ? 1 : 0); ++i) {
        CanonicalTestElement element;
        element.name = i < premises ? static_cast<int>(rng() % 3u) : 3;
        element.negation = i < premises && rng() % 4u == 0;
        for (int a = 0; a < kCanonicalTestArity[element.name]; ++a)
            element.variables.push_back(static_cast<int>(rng() % 6u));
        body.push_back(element);
    }
    return body;
}

// The body spelled with the given variable names (0..2 bound, 3..5 u_).
std::vector<gl::EncodedExpression> canonicalTestSpell(const std::vector<CanonicalTestElement>& body,
                                                     const std::vector<std::string>& names) {
    std::vector<gl::EncodedExpression> out;
    for (const auto& element : body) {
        std::string text = std::string(element.negation ? "!(" : "(") + kCanonicalTestNames[element.name] + "[";
        for (std::size_t a = 0; a < element.variables.size(); ++a) {
            if (a) text += ',';
            text += names[static_cast<std::size_t>(element.variables[a])];
        }
        out.emplace_back(text + "])", "main");
    }
    return out;
}

// The comparison sequence of a list in the given order: per element its
// polarity, then per argument its kind (bound 0, u_ 1) and number, read from
// the mode-2 normalized key.
std::vector<std::vector<int>> canonicalTestTokens(const std::vector<gl::EncodedExpression>& list) {
    const auto normalized = gl::CompilationNormalizer::makeNormalizedEncodedKey(list, 2);
    std::vector<std::vector<int>> out;
    std::size_t position = 0;
    for (const auto& expression : list) {
        ++position;   // the operator name
        std::vector<int> token{normalized.first.data[position++] == "True" ? 1 : 0};
        for (std::size_t a = 0; a < expression.arguments.size(); ++a) {
            const std::string& t = normalized.first.data[position++];
            const bool isU = t.rfind("u_", 0) == 0;
            token.push_back(isU ? 1 : 0);
            token.push_back(std::stoi(isU ? t.substr(2) : t));
        }
        out.push_back(token);
    }
    return out;
}

// The oracle: the smallest sequence over every order that keeps the operator
// groups in name order, each group in every one of its permutations.
std::vector<std::vector<int>> canonicalTestBruteForce(const std::vector<gl::EncodedExpression>& tail,
                                                      const std::vector<gl::EncodedExpression>& head) {
    std::map<std::string, std::vector<int>> groups;
    for (std::size_t i = 0; i < tail.size(); ++i) groups[tail[i].name].push_back(static_cast<int>(i));
    std::vector<std::vector<int>> members;
    for (const auto& group : groups) members.push_back(group.second);
    std::vector<std::vector<int>> best;
    std::vector<int> order;
    const auto walk = [&](const auto& self, std::size_t g) -> void {
        if (g == members.size()) {
            std::vector<gl::EncodedExpression> list;
            for (const int index : order) list.push_back(tail[static_cast<std::size_t>(index)]);
            list.insert(list.end(), head.begin(), head.end());
            const auto tokens = canonicalTestTokens(list);
            if (best.empty() || tokens < best) best = tokens;
            return;
        }
        std::vector<int> permutation = members[g];
        std::sort(permutation.begin(), permutation.end());
        do {
            order.insert(order.end(), permutation.begin(), permutation.end());
            self(self, g + 1);
            order.resize(order.size() - permutation.size());
        } while (std::next_permutation(permutation.begin(), permutation.end()));
    };
    walk(walk, 0);
    return best;
}

const std::vector<std::string> kCanonicalTestVariables{"a", "b", "c", "u_x", "u_y", "u_z"};
} // namespace

// The canonical order is the smallest normalized order of the body: compared
// with an exhaustive search over every in-group permutation, with and without
// a head, polarity and repeated variables included.
TEST(compilation_normalizer, canonical_order_is_the_smallest_normalized_ordering) {
    std::mt19937 rng(20261016u);
    int checked = 0;
    for (int trial = 0; trial < 400; ++trial) {
        const auto body = canonicalTestBody(rng, 1 + static_cast<int>(rng() % 6u), trial % 2 == 0);
        const auto spelled = canonicalTestSpell(body, kCanonicalTestVariables);
        const bool withHead = trial % 2 == 0;
        const std::vector<gl::EncodedExpression> tail(spelled.begin(), spelled.end() - (withHead ? 1 : 0));
        const std::vector<gl::EncodedExpression> head(spelled.end() - (withHead ? 1 : 0), spelled.end());
        auto canonical = gl::CompilationNormalizer::canonicalCombination(tail, head);
        ASSERT_EQ(canonical.size(), tail.size());
        std::multiset<std::string> before, after;
        for (const auto& e : tail) before.insert(e.original);
        for (const auto& e : canonical) after.insert(e.original);
        ASSERT_TRUE(before == after);
        canonical.insert(canonical.end(), head.begin(), head.end());
        ASSERT_TRUE(canonicalTestTokens(canonical) == canonicalTestBruteForce(tail, head));
        ++checked;
    }
    ASSERT_EQ(checked, 400);
    ASSERT_TRUE(gl::CompilationNormalizer::canonicalCombination({}, {}).empty());
}

// The repetition-map key of a body does not depend on the order of its
// premises or the names of its variables, and a body already in canonical
// order keeps its order.
TEST(compilation_normalizer, canonical_key_ignores_premise_order_and_variable_names) {
    std::mt19937 rng(20261017u);
    for (int trial = 0; trial < 300; ++trial) {
        const bool withHead = trial % 3 != 0;
        auto body = canonicalTestBody(rng, 1 + static_cast<int>(rng() % 7u), withHead);
        const auto spelled = canonicalTestSpell(body, kCanonicalTestVariables);
        const std::vector<gl::EncodedExpression> tail(spelled.begin(), spelled.end() - (withHead ? 1 : 0));
        const std::vector<gl::EncodedExpression> head(spelled.end() - (withHead ? 1 : 0), spelled.end());
        const auto key = gl::CompilationReuse::canonicalReuseKey(tail, head);

        // Shuffle the premises, rename the bound and the u_ variables among
        // fresh names of their own kind.
        std::vector<int> premiseOrder(tail.size());
        for (std::size_t i = 0; i < premiseOrder.size(); ++i) premiseOrder[i] = static_cast<int>(i);
        std::shuffle(premiseOrder.begin(), premiseOrder.end(), rng);
        std::vector<CanonicalTestElement> shuffled;
        for (const int index : premiseOrder) shuffled.push_back(body[static_cast<std::size_t>(index)]);
        if (withHead) shuffled.push_back(body.back());
        std::vector<std::string> renamed{"d", "e", "f", "u_v", "u_w", "u_t"};
        std::shuffle(renamed.begin(), renamed.begin() + 3, rng);
        std::shuffle(renamed.begin() + 3, renamed.end(), rng);
        const auto respelled = canonicalTestSpell(shuffled, renamed);
        const std::vector<gl::EncodedExpression> tail2(respelled.begin(), respelled.end() - (withHead ? 1 : 0));
        const std::vector<gl::EncodedExpression> head2(respelled.end() - (withHead ? 1 : 0), respelled.end());
        ASSERT_TRUE(gl::CompilationReuse::canonicalReuseKey(tail2, head2).first == key.first);

        // A canonical list comes back in the same order.
        const auto canonical = gl::CompilationNormalizer::canonicalCombination(tail, head);
        const auto again = gl::CompilationNormalizer::canonicalCombination(canonical, head);
        for (std::size_t i = 0; i < canonical.size(); ++i)
            ASSERT_EQ(again[i].original, canonical[i].original);
    }
    // Different structures keep different keys: a chain a-b, b-c against two
    // edges that meet in b.
    const std::vector<gl::EncodedExpression> chain{gl::EncodedExpression("(p[a,b])", "main"),
                                                    gl::EncodedExpression("(p[b,c])", "main")};
    const std::vector<gl::EncodedExpression> meet{gl::EncodedExpression("(p[a,b])", "main"),
                                                   gl::EncodedExpression("(p[c,b])", "main")};
    ASSERT_FALSE(gl::CompilationReuse::canonicalReuseKey(chain, {}).first
        == gl::CompilationReuse::canonicalReuseKey(meet, {}).first);
}

// The head decides between premise orders the premises leave tied; premises
// that stay interchangeable are taken in input order; a same-name group of
// nine premises (above the former seven-premise cap) keys exactly.
TEST(compilation_normalizer, canonical_order_head_ties_and_large_groups) {
    const std::vector<gl::EncodedExpression> premises{gl::EncodedExpression("(in[u_a,u_N])", "main"),
                                                       gl::EncodedExpression("(in[u_b,u_N])", "main")};
    const std::vector<gl::EncodedExpression> head{gl::EncodedExpression("(in3[u_b,u_a,u_c,u_plus])", "main")};
    const auto ordered = gl::CompilationNormalizer::canonicalCombination(premises, head);
    ASSERT_EQ(ordered[0].original, std::string("(in[u_b,u_N])"));
    ASSERT_EQ(ordered[1].original, std::string("(in[u_a,u_N])"));
    const auto key = gl::CompilationReuse::canonicalReuseKey(premises, head);
    ASSERT_TRUE(key.first == std::vector<std::string>({"(in[u_1,u_2])", "(in[u_3,u_2])", "(in3[u_1,u_3,u_4,u_5])"}));
    ASSERT_EQ(key.second.at("u_b"), std::string("u_1"));
    ASSERT_EQ(key.second.at("u_a"), std::string("u_3"));

    // Private variables: either premise may come first; the first is kept.
    const auto alone = gl::CompilationNormalizer::canonicalCombination(premises, {});
    ASSERT_EQ(alone[0].original, std::string("(in[u_a,u_N])"));

    std::vector<gl::EncodedExpression> large;
    std::vector<gl::EncodedExpression> reversed;
    for (int i = 1; i <= 9; ++i)
        large.emplace_back("(in[u_x" + std::to_string(i) + ",u_N])", "main");
    large.emplace_back("(in3[u_x1,u_x2,u_x3,u_plus])", "main");
    for (auto it = large.rbegin(); it != large.rend(); ++it) reversed.push_back(*it);
    const std::vector<gl::EncodedExpression> conclusion{gl::EncodedExpression("(in[u_x3,u_N])", "main")};
    ASSERT_TRUE(gl::CompilationReuse::canonicalReuseKey(large, conclusion).first
        == gl::CompilationReuse::canonicalReuseKey(reversed, conclusion).first);
}

TEST(worker_context, keeps_worker_identity_local_to_each_thread) {
    const int saved = gl::WorkerContext::coreId();
    gl::WorkerContext::setCoreId(9);
    int initial = 0, selected = 0;
    std::thread worker([&] {
        initial = gl::WorkerContext::coreId();
        gl::WorkerContext::setCoreId(17);
        selected = gl::WorkerContext::coreId();
    });
    worker.join();
    ASSERT_EQ(initial, -1);
    ASSERT_EQ(selected, 17);
    ASSERT_EQ(gl::WorkerContext::coreId(), 9);
    gl::WorkerContext::setCoreId(saved);
}

TEST(definition_expansion, reads_definitions_without_prover_state) {
    ce::CoreExpressionMap configurations;
    ce::CoreExpressionConfig predicate;
    predicate.signature = "(R[x])";
    predicate.definition = std::string("(p[x])");
    configurations["R"] = predicate;
    gl::CompiledExpressionMap definitions;
    definitions["and7"] = gl::LogicalEntity("and", {"(p[u_1])", "(q[u_1])"}, "(and7[u_1])", 1);
    gl::DefinitionExpansion expansion(configurations, definitions);
    ASSERT_EQ(expansion.expandExpr("(R[a])"), std::string("(p[a])"));
    ASSERT_EQ(expansion.expandExpr("(unknown[a])"), std::string("(unknown[a])"));
    ASSERT_EQ(expansion.expandToBaseForm("(and7[a])"), std::string("(&(p[a])(q[a]))"));
    ASSERT_EQ(configurations.size(), std::size_t(1));
    ASSERT_EQ(definitions.size(), std::size_t(1));
}

// A conjunction has exactly two children: a compiled and or or with three
// elements expands to two-child conjunctions nested to the left, each or
// disjunct negated with double-negation cancellation; two elements keep
// the single conjunction.
TEST(definition_expansion, base_form_nests_two_child_conjunctions_to_the_left) {
    ce::CoreExpressionMap configurations;
    gl::CompiledExpressionMap definitions;
    definitions["and8"] = gl::LogicalEntity("and", {"(p[u_1])", "(q[u_1])", "(r[u_1])"}, "(and8[u_1])", 1);
    definitions["or5"] = gl::LogicalEntity("or", {"(p[u_1])", "!(q[u_1])", "(r[u_1])"}, "(or5[u_1])", 1);
    definitions["or6"] = gl::LogicalEntity("or", {"(p[u_1])", "(q[u_1])"}, "(or6[u_1])", 1);
    gl::DefinitionExpansion expansion(configurations, definitions);
    ASSERT_EQ(expansion.expandToBaseForm("(and8[a])"), std::string("(&(&(p[a])(q[a]))(r[a]))"));
    ASSERT_EQ(expansion.expandToBaseForm("(or5[a])"), std::string("!(&(&!(p[a])(q[a]))!(r[a]))"));
    ASSERT_EQ(expansion.expandToBaseForm("!(or5[a])"), std::string("(&(&!(p[a])(q[a]))!(r[a]))"));
    ASSERT_EQ(expansion.expandToBaseForm("(or6[a])"), std::string("!(&!(p[a])!(q[a]))"));
}

// Both nestings of a conjunction chain yield the same ordered conjuncts; a
// negated conjunction is one conjunct and a non-conjunction is itself.
TEST(definition_expansion, collects_conjuncts_of_either_nesting) {
    ce::CoreExpressionMap configurations;
    const auto collect = [&](const std::string& expression) {
        std::vector<std::string> conjuncts;
        gl::DefinitionExpansion::collectConjuncts(expression, configurations, conjuncts);
        return conjuncts;
    };
    const std::vector<std::string> three = {"(p[a])", "!(q[a])", "(r[a])"};
    ASSERT_TRUE(collect("(&(&(p[a])!(q[a]))(r[a]))") == three);
    ASSERT_TRUE(collect("(&(p[a])(&!(q[a])(r[a])))") == three);
    ASSERT_TRUE(collect("(&(&(p[a])!(q[a]))(&(r[a])(s[a])))")
        == std::vector<std::string>({"(p[a])", "!(q[a])", "(r[a])", "(s[a])"}));
    ASSERT_TRUE(collect("(&(p[a])!(&(q[a])(r[a])))")
        == std::vector<std::string>({"(p[a])", "!(&(q[a])(r[a]))"}));
    ASSERT_TRUE(collect("(&(p[a])(>[x](q[x])(r[x])))")
        == std::vector<std::string>({"(p[a])", "(>[x](q[x])(r[x]))"}));
    ASSERT_TRUE(collect("(p[a])") == std::vector<std::string>({"(p[a])"}));
}

// The parser reads exactly two children per conjunction, in either nesting.
TEST(expression_parser, reads_two_child_conjunctions_of_either_nesting) {
    for (const std::string& text : {std::string("(&(&(p[a])(q[a]))(r[a]))"),
                                    std::string("(&(p[a])(&(q[a])(r[a])))"),
                                    std::string("!(&(&!(p[a])!(q[a]))!(r[a]))")}) {
        ce::TreeNode1* root = ce::parseExpr(text);
        ASSERT_TRUE(root != NULL && root->left != NULL && root->right != NULL);
        ASSERT_EQ(ce::treeToExpr(root), text);
        ce::deleteTree(root);
    }
}

TEST(definition_expansion, preserves_argument_prefix_and_negation_operations) {
    ASSERT_EQ(gl::DefinitionExpansion::removeUPrefixFromArguments("(p[u_a,b])"), std::string("(p[a,b])"));
    ASSERT_EQ(gl::DefinitionExpansion::addMissingU("(p[a,u_b,marker])"), std::string("(p[u_a,u_b,marker])"));
    ASSERT_EQ(gl::DefinitionExpansion::replaceUSubstrings("(p[u_u_a,u_b])"), std::string("(p[a,b])"));
    ASSERT_EQ(gl::DefinitionExpansion::stripUPrefixAST("(>[x](p[u_a])(q[x]))"), std::string("(>[x](p[a])(q[x]))"));
    const auto values = gl::DefinitionExpansion::extractValues("(>[x](p[x])(q[x]))", {});
    ASSERT_EQ(values.first, std::string("(p[x])"));
    ASSERT_EQ(values.second, std::string("(q[x])"));
    ASSERT_EQ(gl::DefinitionExpansion::negate("!(p[a])"), std::string("(p[a])"));
}

TEST(definition_expansion, scratch_negation_preserves_input_and_arena_lifetimes) {
    gl::GlobalMemoryManager manager;
    manager.init(gl::StaticMemoryConfig{16 << 20, 1 << 18});
    gl::ScratchArena arena(&manager);
    const std::string negative = "!(p[a])";
    const auto before = arena.cursor();
    const gl::StrSpan positive = gl::DefinitionExpansion::negateScratch(arena, gl::StrSpan(negative));
    ASSERT_EQ(positive.toStdString(), std::string("(p[a])"));
    ASSERT_TRUE(positive.ptr == negative.data() + 1);
    ASSERT_TRUE(arena.cursor() == before);
    const gl::StrSpan prefixed = gl::DefinitionExpansion::negateScratch(arena, positive);
    ASSERT_EQ(prefixed.toStdString(), negative);
    ASSERT_EQ(gl::DefinitionExpansion::negateScratch(arena, gl::StrSpan()).toStdString(), std::string("!"));
    ASSERT_EQ(negative, std::string("!(p[a])"));
}

TEST(worker_context, parallel_mutation_flag_is_thread_local) {
    const bool saved = gl::WorkerContext::inParallelWorkerPhase();
    gl::WorkerContext::setParallelWorkerPhase(true);
    ASSERT_TRUE(gl::WorkerContext::inParallelWorkerPhase());
    bool workerStartedOutside = false;
    bool workerEntered = false;
    std::thread worker([&]() {
        workerStartedOutside = !gl::WorkerContext::inParallelWorkerPhase();
        gl::WorkerContext::setParallelWorkerPhase(true);
        workerEntered = gl::WorkerContext::inParallelWorkerPhase();
        gl::WorkerContext::setParallelWorkerPhase(false);
    });
    worker.join();
    ASSERT_TRUE(workerStartedOutside);
    ASSERT_TRUE(workerEntered);
    ASSERT_TRUE(gl::WorkerContext::inParallelWorkerPhase());
    gl::WorkerContext::setParallelWorkerPhase(false);
    ASSERT_FALSE(gl::WorkerContext::inParallelWorkerPhase());
    gl::WorkerContext::setParallelWorkerPhase(saved);
}
