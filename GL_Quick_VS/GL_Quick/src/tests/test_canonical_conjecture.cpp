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
/// @brief Tests for the canonical spelling of a compiled conjecture
///        (`ProofGridBuilder::canonicalConjecture`,
///        D-493).
///
/// @details
/// The canonical spelling numbers the variables in first-occurrence order over
/// the compiled premise chain and the head and binds each at its first
/// premise — the spelling a theorem's implication compact expands to. The
/// rows are FTA supplied rows: row 119 is canonical, row 120 introduces its
/// two coefficients in the order 15, 14.

#include "../memory.hpp"
#include <string>
#include "test_harness.hpp"
#include "prover_fixture.hpp"

#include "../prover.hpp"

namespace {

    const char* const kAnchor = "(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])";

    /// @brief A theorem under the FTA anchor.
    ///
    /// @param body The implication below the anchor premise.
    /// @return The full theorem text.
    std::string underAnchor(const std::string& body) {
        return std::string(kAnchor) + body + ")";
    }

} // namespace

// A canonical conjecture is returned unchanged, and the result is canonical:
// canonicalizing twice is canonicalizing once.
TEST(canonical_conjecture, canonical_rows_are_unchanged) {
    gl::tests::ProverFixture ana(std::string("FTA"));
    const std::string row119 = underAnchor(
        "(>[9,10,11](in3[9,10,11,5])(>[12,13](in3[11,12,13,4])(>[14,15](bezEq[1,4,5,6,9,14,12,15])"
        "(>[16](in3[10,15,16,5])(>[17](in3[14,16,17,4])(bezEq[1,4,5,6,9,17,13,15]))))))");
    ASSERT_TRUE(ana.proofGridBuilder.canonicalConjecture(row119) == row119);
}

// Row 120 introduces its coefficients as 15, 14; the canonical spelling
// numbers them 14, 15 everywhere — the spelling its compact expands to.
TEST(canonical_conjecture, bound_variables_renumber_in_first_occurrence_order) {
    gl::tests::ProverFixture ana(std::string("FTA"));
    const std::string supplied = underAnchor(
        "(>[9,10,11](in3[9,10,11,5])(>[12,13](in3[11,12,13,4])(>[14,15](bezEq[1,4,5,6,12,15,9,14])"
        "(>[16](in3[10,15,16,5])(>[17](in3[16,14,17,4])(bezEq[1,4,5,6,13,15,9,17]))))))");
    const std::string canonical = underAnchor(
        "(>[9,10,11](in3[9,10,11,5])(>[12,13](in3[11,12,13,4])(>[14,15](bezEq[1,4,5,6,12,14,9,15])"
        "(>[16](in3[10,14,16,5])(>[17](in3[16,15,17,4])(bezEq[1,4,5,6,13,14,9,17]))))))");
    ASSERT_TRUE(ana.proofGridBuilder.canonicalConjecture(supplied) == canonical);
    ASSERT_TRUE(ana.proofGridBuilder.canonicalConjecture(canonical) == canonical);
}

// A numbering gap left by a bound variable compiled into an existence closes,
// and a variable bound before its first use moves to the premise that first
// uses it.
TEST(canonical_conjecture, gaps_close_and_binders_move_to_first_use) {
    gl::tests::ProverFixture ana(std::string("FTA"));
    ASSERT_TRUE(ana.proofGridBuilder.canonicalConjecture(underAnchor(
            "(>[9,10,12](limitSequence[1,4,9,10,12])(in2[9,10,12]))"))
        == underAnchor("(>[9,10,11](limitSequence[1,4,9,10,11])(in2[9,10,11]))"));
    ASSERT_TRUE(ana.proofGridBuilder.canonicalConjecture(underAnchor(
            "(>[9,10](in[9,1])(>[](in[10,1])!(=[9,10])))"))
        == underAnchor("(>[9](in[9,1])(>[10](in[10,1])!(=[9,10])))"));
}
