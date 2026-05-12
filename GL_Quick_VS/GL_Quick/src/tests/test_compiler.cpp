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

#include <cstddef>

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
    ASSERT_EQ(ce::extractExpressionUniversal("(p[a])"), std::string("p"));
    // Negated dispatches to extractExpressionFromNegation.
    ASSERT_EQ(ce::extractExpressionUniversal("!(p[a])"), std::string("p"));
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
    std::map<std::string, ce::CoreExpressionConfig> m;
    m["="]       = ce::CoreExpressionConfig(2, std::string{}, "(=[1,2])");
    m["AnchorX"] = ce::CoreExpressionConfig(3, std::string{}, "(AnchorX[1,2,3])");
    m["in"]      = ce::CoreExpressionConfig(2, std::string{}, "(in[1,2])");
    ASSERT_EQ(ce::findAnchorKey(m), std::string("AnchorX"));
}

TEST(compiler, findanchorkey_returns_empty_when_no_anchor) {
    std::map<std::string, ce::CoreExpressionConfig> m;
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

// ---------- ce::extractExpressionUniversal ----------
TEST(compiler, extractexpressionuniversal_zero_arity_negated) {
    ASSERT_EQ(ce::extractExpressionUniversal("!(zero[])"),
              std::string("zero"));
}

TEST(compiler, extractexpressionuniversal_anchor_form) {
    // Anchor expression — non-negated dispatch path.
    ASSERT_EQ(ce::extractExpressionUniversal("(AnchorPeano[1,2,3])"),
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
    std::map<std::string, ce::CoreExpressionConfig> m;
    m["AnchorZ"] = ce::CoreExpressionConfig(2, std::string{}, "(AnchorZ[1,2])");
    m["Anchor1"] = ce::CoreExpressionConfig(2, std::string{}, "(Anchor1[1,2])");
    ASSERT_EQ(ce::findAnchorKey(m), std::string("Anchor1"));
}

TEST(compiler, findanchorkey_ignores_anchor_substring_in_middle) {
    // findAnchorKey requires the prefix to start with "Anchor", not contain it.
    std::map<std::string, ce::CoreExpressionConfig> m;
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
