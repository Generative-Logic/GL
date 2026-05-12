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
