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
/// @brief Phase-5 unit tests for the CE-filter module.
///
/// @details
/// The CE-filter ExpressionAnalyzer member-function bodies
/// (`readSimpleFacts`, `saveFilteredConjectures`, `loadFactsForCEFiltering`,
/// `addConjectureForCEFiltering`, `releaseCEBatchMemory`) require a fully
/// constructed `ExpressionAnalyzer` plus disk fixtures, which is too heavy
/// for the <30 s test budget. This phase covers the value-only pieces:
/// `ContradictionItem` default + parameterized construction and equality
/// behaviour.
///
/// Instance-method coverage lands in Phase 7+ once `ExpressionAnalyzer`
/// is documented and a synthetic-config scaffold can be wired up.

#include "test_harness.hpp"

#include "../filter.hpp"
#include "../parameters.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

TEST(filter, contradictionitem_default_is_unsuccessful) {
    gl::ContradictionItem ci;
    ASSERT_TRUE(ci.expr.empty());
    ASSERT_FALSE(ci.successful);
}

TEST(filter, contradictionitem_parameterized_construction) {
    gl::ContradictionItem ci("(=[a,b])", 1);
    ASSERT_EQ(ci.expr, std::string("(=[a,b])"));
    ASSERT_TRUE(ci.successful);
}

TEST(filter, contradictionitem_int_to_bool_zero_is_false) {
    // The parameterized constructor accepts int (legacy contract) and
    // converts non-zero to true, zero to false.
    gl::ContradictionItem ci_zero("expr",  0);
    gl::ContradictionItem ci_one ("expr",  1);
    gl::ContradictionItem ci_neg ("expr", -1);
    ASSERT_FALSE(ci_zero.successful);
    ASSERT_TRUE(ci_one.successful);
    ASSERT_TRUE(ci_neg.successful);  // any non-zero is truthy
}

// =============================================================================
// Round 2 — two more tests per documented function.
// =============================================================================

// ---------- ContradictionItem default ctor ----------
TEST(filter, contradictionitem_default_assignable_to_default) {
    // A default-constructed item compares-equal to another default
    // (no ordering operator declared, so we check fields directly).
    gl::ContradictionItem a;
    gl::ContradictionItem b;
    ASSERT_EQ(a.expr,       b.expr);
    ASSERT_EQ(a.successful, b.successful);
}

TEST(filter, contradictionitem_default_in_vector) {
    // Default-constructable means resize() works on a vector — the
    // type ExpressionAnalyzer holds in `contradictionTable`.
    std::vector<gl::ContradictionItem> v;
    v.resize(3);
    ASSERT_EQ(v.size(), static_cast<std::size_t>(3));
    for (const auto& ci : v) {
        ASSERT_TRUE(ci.expr.empty());
        ASSERT_FALSE(ci.successful);
    }
}

// ---------- ContradictionItem parameterized ctor ----------
TEST(filter, contradictionitem_param_with_empty_expr) {
    // Empty expression text is legal — the parameterized ctor does not
    // require non-empty content.
    gl::ContradictionItem ci("", 1);
    ASSERT_TRUE(ci.expr.empty());
    ASSERT_TRUE(ci.successful);
}

TEST(filter, contradictionitem_param_long_expression_text) {
    // No length cap on the expression slot.
    const std::string longExpr =
        "(>[1,2,3,4,5,6,7,8,9,10](AnchorPeano[1,2,3,4,5,6])(in3[7,8,9,4]))";
    gl::ContradictionItem ci(longExpr, 0);
    ASSERT_EQ(ci.expr, longExpr);
    ASSERT_FALSE(ci.successful);
}

// ---------- ContradictionItem int-to-bool conversion ----------
TEST(filter, contradictionitem_int_max_is_truthy) {
    gl::ContradictionItem ci("expr", std::numeric_limits<int>::max());
    ASSERT_TRUE(ci.successful);
}

TEST(filter, contradictionitem_int_min_is_truthy) {
    gl::ContradictionItem ci("expr", std::numeric_limits<int>::min());
    ASSERT_TRUE(ci.successful);   // INT_MIN != 0
}

// =============================================================================
// Mirror refutation (D-229) — the free
// functions loadMirrorPairs / applyMirrorRefutations, plus the config
// default. The flip's call-site gating on parameters.mirror_refutation
// lives in filterConjecturesWithCE, which needs a full ExpressionAnalyzer;
// it is pinned here indirectly by the default-value test and the Doxygen
// contract.
// =============================================================================

namespace {

    std::vector<gl::ContradictionItem> makeTable(
        const std::vector<std::string>& conjectures,
        const std::vector<int>& refuted) {
        std::vector<gl::ContradictionItem> table;
        for (std::size_t i = 0; i < conjectures.size(); ++i) {
            table.emplace_back(conjectures[i], 0);
        }
        for (int idx : refuted) table[idx].successful = true;
        return table;
    }

    gl::MirrorPartnerMap makePartners(const std::string& a, const std::string& b) {
        gl::MirrorPartnerMap partners;
        partners[a].push_back(b);
        partners[b].push_back(a);
        return partners;
    }

} // namespace

TEST(filter, parameters_default_mirror_refutation_true) {
    gl::ProverParameters p;
    ASSERT_TRUE(p.mirror_refutation);
}

TEST(filter, apply_mirror_refutations_flips_partner_of_refuted_source) {
    const std::vector<std::string> conjectures = {"srcA", "mirA", "other"};
    auto table = makeTable(conjectures, {0});
    const auto partners = makePartners("srcA", "mirA");
    const int flipped = gl::applyMirrorRefutations(table, conjectures, partners);
    ASSERT_EQ(flipped, 1);
    ASSERT_TRUE(table[0].successful);
    ASSERT_TRUE(table[1].successful);
    ASSERT_FALSE(table[2].successful);
}

TEST(filter, apply_mirror_refutations_symmetric_mirror_to_source) {
    // Refuting the MIRROR member dooms the source: the map carries both
    // directions.
    const std::vector<std::string> conjectures = {"srcA", "mirA"};
    auto table = makeTable(conjectures, {1});
    const auto partners = makePartners("srcA", "mirA");
    const int flipped = gl::applyMirrorRefutations(table, conjectures, partners);
    ASSERT_EQ(flipped, 1);
    ASSERT_TRUE(table[0].successful);
    ASSERT_TRUE(table[1].successful);
}

TEST(filter, apply_mirror_refutations_no_flip_when_pair_absent) {
    const std::vector<std::string> conjectures = {"srcA", "mirA"};
    auto table = makeTable(conjectures, {0});
    const gl::MirrorPartnerMap empty;
    const int flipped = gl::applyMirrorRefutations(table, conjectures, empty);
    ASSERT_EQ(flipped, 0);
    ASSERT_TRUE(table[0].successful);
    ASSERT_FALSE(table[1].successful);
}

TEST(filter, apply_mirror_refutations_partner_missing_from_batch_is_noop) {
    // The partner was refuted in an earlier fact-file pass and is no longer
    // in this pass's conjecture list — a defined no-op, not a failure.
    const std::vector<std::string> conjectures = {"srcA", "other"};
    auto table = makeTable(conjectures, {0});
    const auto partners = makePartners("srcA", "mirA");
    const int flipped = gl::applyMirrorRefutations(table, conjectures, partners);
    ASSERT_EQ(flipped, 0);
    ASSERT_TRUE(table[0].successful);
    ASSERT_FALSE(table[1].successful);
}

TEST(filter, apply_mirror_refutations_no_cascade_from_flipped_rows) {
    // Chain srcA↔mirA and mirA↔srcB: refuting srcA flips mirA, but the
    // flipped mirA must NOT seed a second-generation flip of srcB — only
    // CE-confirmed refutations seed.
    const std::vector<std::string> conjectures = {"srcA", "mirA", "srcB"};
    auto table = makeTable(conjectures, {0});
    gl::MirrorPartnerMap partners;
    partners["srcA"].push_back("mirA");
    partners["mirA"].push_back("srcA");
    partners["mirA"].push_back("srcB");
    partners["srcB"].push_back("mirA");
    const int flipped = gl::applyMirrorRefutations(table, conjectures, partners);
    ASSERT_EQ(flipped, 1);
    ASSERT_TRUE(table[1].successful);
    ASSERT_FALSE(table[2].successful);
}

TEST(filter, apply_mirror_refutations_already_refuted_partner_not_recounted) {
    // Both members CE-refuted independently: the pass flips nothing and
    // reports zero.
    const std::vector<std::string> conjectures = {"srcA", "mirA"};
    auto table = makeTable(conjectures, {0, 1});
    const auto partners = makePartners("srcA", "mirA");
    const int flipped = gl::applyMirrorRefutations(table, conjectures, partners);
    ASSERT_EQ(flipped, 0);
}

TEST(filter, load_mirror_pairs_parses_both_directions) {
    namespace fs = std::filesystem;
    const fs::path path =
        fs::temp_directory_path() / "gl_test_mirror_pairs_bidir.txt";
    {
        std::ofstream out(path);
        out << "srcA\tmirA\n";
        out << "srcB\tmirB\n";
    }
    const auto partners = gl::loadMirrorPairs(path);
    std::remove(path.string().c_str());
    ASSERT_EQ(partners.size(), static_cast<std::size_t>(4));
    ASSERT_EQ(partners.at("srcA").size(), static_cast<std::size_t>(1));
    ASSERT_EQ(partners.at("srcA")[0], std::string("mirA"));
    ASSERT_EQ(partners.at("mirA")[0], std::string("srcA"));
    ASSERT_EQ(partners.at("mirB")[0], std::string("srcB"));
}

TEST(filter, load_mirror_pairs_strips_cr_and_skips_empty_lines) {
    namespace fs = std::filesystem;
    const fs::path path =
        fs::temp_directory_path() / "gl_test_mirror_pairs_crlf.txt";
    {
        std::ofstream out(path, std::ios::binary);
        out << "srcA\tmirA\r\n";
        out << "\r\n";
        out << "\n";
    }
    const auto partners = gl::loadMirrorPairs(path);
    std::remove(path.string().c_str());
    ASSERT_EQ(partners.size(), static_cast<std::size_t>(2));
    ASSERT_EQ(partners.at("srcA")[0], std::string("mirA"));
}

TEST(filter, load_mirror_pairs_empty_file_yields_empty_map) {
    // The incubator-mode contract: the file exists but carries no rows.
    namespace fs = std::filesystem;
    const fs::path path =
        fs::temp_directory_path() / "gl_test_mirror_pairs_empty.txt";
    { std::ofstream out(path); }
    const auto partners = gl::loadMirrorPairs(path);
    std::remove(path.string().c_str());
    ASSERT_TRUE(partners.empty());
}

TEST(filter, load_mirror_pairs_duplicate_rows_deduplicated) {
    namespace fs = std::filesystem;
    const fs::path path =
        fs::temp_directory_path() / "gl_test_mirror_pairs_dup.txt";
    {
        std::ofstream out(path);
        out << "srcA\tmirA\n";
        out << "srcA\tmirA\n";
    }
    const auto partners = gl::loadMirrorPairs(path);
    std::remove(path.string().c_str());
    ASSERT_EQ(partners.at("srcA").size(), static_cast<std::size_t>(1));
    ASSERT_EQ(partners.at("mirA").size(), static_cast<std::size_t>(1));
}
