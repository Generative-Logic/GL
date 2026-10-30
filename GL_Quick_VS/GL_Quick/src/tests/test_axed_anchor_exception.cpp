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
/// @brief Tests for the axed-variable deposit filter's flag-gated
///        positive-anchor exception in `addExprToMemoryBlock`
///        (D-234, `axed_anchor_exception`,
///        default off): with the flag ON, a positive anchor-category
///        statement carrying an axed x-copy name deposits; with the flag
///        at its default OFF, and for every other x-carrying deposit —
///        non-anchor or negated anchor — the filter drops it. Also covers
///        the `prehandleAnchor` walk's recursion-subtree modes
///        (D-272,
///        I-188): set-only containment (axed
///        names minted, no statement), the anchor-numeral exception (a
///        block-#1 root whose goal cites a `(1)`-typed slot value as a
///        combinable argument leaves its subtree untouched; a fixed
///        definition parameter does not count), the `_induction_` block-#2
///        marker override, and full mint+register outside recursion
///        subtrees.

#include "../memory.hpp"
#include <cstddef>
#include <set>
#include <string>
#include <utility>
#include "test_harness.hpp"
#include "prover_fixture.hpp"

#include "../prover.hpp"

namespace {
    const char* kConj =
        "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8](in2[7,8,3])(=[7,8])))";

    // Registers kConj; the premise-LB chain:
    // root -> (AnchorPeano[...]) -> (in2[7,8,3]).
    gl::Memory* innerOf(gl::tests::ProverFixture& ea) {
        ea.proofGridBuilder.addTheoremToMemory(std::string(kConj), ea.body, 0, false,
                              ea.globalDependencies);
        gl::Memory* anchorLB = ea.simpleMapStore.findChild(
            &ea.body, "(AnchorPeano[1,2,3,4,5,6])");
        ASSERT_TRUE(anchorLB != nullptr);
        gl::Memory* innerLB =
            ea.simpleMapStore.findChild(anchorLB, "(in2[7,8,3])");
        ASSERT_TRUE(innerLB != nullptr);
        return innerLB;
    }

    // Mimics prehandleAnchor's axed-variable registration for slot 2.
    void axeX2(gl::Memory& lb) {
        ((lb).intAxedVariables).mint(((lb).nameMap).encode("x2"));
    }

    void deposit(gl::tests::ProverFixture& ea, gl::Memory& lb,
                 const char* expr) {
        const int lvRun[1] = { ((lb).level) };
        const gl::TransientOrigin origin{
            true, gl::OriginTag::taskFormulation, nullptr, 0 };
        ea.statementMemory.addExprToMemoryBlock(
            gl::StrSpan(expr, static_cast<int32_t>(std::strlen(expr))), lb, 0,
            0, lvRun, 1, origin, -1, -1, gl::StrSpan("main", 4), true);
    }

    bool knownAtMain(gl::Memory& lb, const char* expr) {
        const gl::StatementFlags* row = gl::lookupStatementFlags(
            ((lb).intKnownStatements), ((lb).nameMap), std::string(expr),
            std::string("main"));
        return row != nullptr;
    }
}

TEST(axed_anchor_exception, non_anchor_with_axed_arg_stays_filtered) {
    gl::tests::ProverFixture ea(std::string("Peano"));
    gl::Memory* lb = innerOf(ea);
    axeX2(*lb);

    deposit(ea, *lb, "(in[x2,1])");

    ASSERT_FALSE(knownAtMain(*lb, "(in[x2,1])"));
}

TEST(axed_anchor_exception, positive_anchor_with_axed_arg_deposits_when_enabled) {
    gl::tests::ProverFixture ea(std::string("Peano"));
    ea.parameters.axed_anchor_exception = true;
    gl::Memory* lb = innerOf(ea);
    axeX2(*lb);

    deposit(ea, *lb, "(AnchorPeano[1,x2,3,4,5,6])");

    ASSERT_TRUE(knownAtMain(*lb, "(AnchorPeano[1,x2,3,4,5,6])"));
}

TEST(axed_anchor_exception, positive_anchor_with_axed_arg_filtered_by_default) {
    gl::tests::ProverFixture ea(std::string("Peano"));
    gl::Memory* lb = innerOf(ea);
    axeX2(*lb);

    deposit(ea, *lb, "(AnchorPeano[1,x2,3,4,5,6])");

    ASSERT_FALSE(knownAtMain(*lb, "(AnchorPeano[1,x2,3,4,5,6])"));
}

TEST(axed_anchor_exception, negated_anchor_with_axed_arg_stays_filtered) {
    gl::tests::ProverFixture ea(std::string("Peano"));
    ea.parameters.axed_anchor_exception = true;
    gl::Memory* lb = innerOf(ea);
    axeX2(*lb);

    deposit(ea, *lb, "!(AnchorPeano[1,x2,3,4,5,6])");

    ASSERT_FALSE(knownAtMain(*lb, "!(AnchorPeano[1,x2,3,4,5,6])"));
}

TEST(axed_anchor_exception, axed_free_deposits_unaffected) {
    gl::tests::ProverFixture ea(std::string("Peano"));
    gl::Memory* lb = innerOf(ea);
    axeX2(*lb);

    deposit(ea, *lb, "(in[7,1])");

    ASSERT_TRUE(knownAtMain(*lb, "(in[7,1])"));
}

namespace {
    // Three-level premise chains whose keys cite anchor slot value 2 (i0),
    // so prehandleAnchor's trace guard actually mints x2:
    // root -> (AnchorPeano[...]) -> (in2[2,7,3]) -> (in2[7,8,3]).
    // kConjNumeralHead's head cites slot value 2 (anchor-numeral-headed);
    // kConjPlainHead's head cites elements only.
    const char* kConjNumeralHead =
        "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])"
        "(>[7](in2[2,7,3])(>[8](in2[7,8,3])(=[2,8]))))";
    const char* kConjPlainHead =
        "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])"
        "(>[7](in2[2,7,3])(>[8](in2[7,8,3])(=[7,8]))))";

    gl::Memory* middleOf(gl::tests::ProverFixture& ea, const char* conj) {
        ea.proofGridBuilder.addTheoremToMemory(std::string(conj), ea.body, 0, false,
                              ea.globalDependencies);
        gl::Memory* anchorLB = ea.simpleMapStore.findChild(
            &ea.body, "(AnchorPeano[1,2,3,4,5,6])");
        ASSERT_TRUE(anchorLB != nullptr);
        gl::Memory* middleLB =
            ea.simpleMapStore.findChild(anchorLB, "(in2[2,7,3])");
        ASSERT_TRUE(middleLB != nullptr);
        return middleLB;
    }

    gl::Memory* deepOf(gl::tests::ProverFixture& ea, gl::Memory* middleLB) {
        gl::Memory* deepLB =
            ea.simpleMapStore.findChild(middleLB, "(in2[7,8,3])");
        ASSERT_TRUE(deepLB != nullptr);
        return deepLB;
    }

    bool axedHasX2(gl::Memory& lb) {
        return ((lb).intAxedVariables).contains(((lb).nameMap).encode("x2"));
    }
}

TEST(axed_anchor_exception, set_only_recursion_subtree_mints_without_statement) {
    gl::tests::ProverFixture ea(std::string("Peano"));
    gl::Memory* middleLB = middleOf(ea, kConjPlainHead);
    gl::Memory* deepLB = deepOf(ea, middleLB);
    ((*(middleLB)).isPartOfRecursion) = true;

    ea.proofGridBuilder.prehandleAnchor(&ea.body);

    ASSERT_TRUE(axedHasX2(*middleLB));
    ASSERT_TRUE(axedHasX2(*deepLB));
    ASSERT_FALSE(knownAtMain(*middleLB, "(AnchorPeano[1,x2,3,4,5,6])"));
    ASSERT_FALSE(knownAtMain(*deepLB, "(AnchorPeano[1,x2,3,4,5,6])"));
}

TEST(axed_anchor_exception, prehandle_registers_statement_outside_recursion) {
    gl::tests::ProverFixture ea(std::string("Peano"));
    gl::Memory* middleLB = middleOf(ea, kConjNumeralHead);
    gl::Memory* deepLB = deepOf(ea, middleLB);

    ea.proofGridBuilder.prehandleAnchor(&ea.body);

    ASSERT_TRUE(axedHasX2(*middleLB));
    ASSERT_TRUE(knownAtMain(*middleLB, "(AnchorPeano[1,x2,3,4,5,6])"));
    ASSERT_TRUE(knownAtMain(*deepLB, "(AnchorPeano[1,x2,3,4,5,6])"));
}

TEST(axed_anchor_exception, numeral_headed_recursion_root_left_untouched) {
    gl::tests::ProverFixture ea(std::string("Peano"));
    gl::Memory* middleLB = middleOf(ea, kConjNumeralHead);
    gl::Memory* deepLB = deepOf(ea, middleLB);
    ((*(deepLB)).isPartOfRecursion) = true;

    ea.proofGridBuilder.prehandleAnchor(&ea.body);

    ASSERT_FALSE(axedHasX2(*deepLB));
    ASSERT_FALSE(knownAtMain(*deepLB, "(AnchorPeano[1,x2,3,4,5,6])"));
    ASSERT_TRUE(knownAtMain(*middleLB, "(AnchorPeano[1,x2,3,4,5,6])"));
}

TEST(axed_anchor_exception, induction_marker_root_stays_set_only) {
    gl::tests::ProverFixture ea(std::string("Peano"));
    gl::Memory* middleLB = middleOf(ea, kConjNumeralHead);
    gl::Memory* deepLB = deepOf(ea, middleLB);
    ((*(deepLB)).isPartOfRecursion) = true;
    deepLB->setExprKey("(in2[7,8,3])_induction_rec0_");

    ea.proofGridBuilder.prehandleAnchor(&ea.body);

    ASSERT_TRUE(axedHasX2(*deepLB));
    ASSERT_FALSE(knownAtMain(*deepLB, "(AnchorPeano[1,x2,3,4,5,6])"));
}

TEST(axed_anchor_exception, door_armed_on_set_only_recursion_lb) {
    gl::tests::ProverFixture ea(std::string("Peano"));
    gl::Memory* middleLB = middleOf(ea, kConjPlainHead);
    ((*(middleLB)).isPartOfRecursion) = true;

    ea.proofGridBuilder.prehandleAnchor(&ea.body);
    deposit(ea, *middleLB, "(in[x2,1])");

    ASSERT_FALSE(knownAtMain(*middleLB, "(in[x2,1])"));
}

namespace {
    // The FTA anchor's (1)-typed slot values: 0, 1 and 2.
    const std::set<std::string> kFtaSlotValues = { "2", "6", "7" };

    // Premise chains under the FTA anchor whose keys cite slot value 7 (the
    // numeral 2), so prehandleAnchor's trace guard mints x7:
    // root -> (AnchorFTA[...]) -> (preorder[1,4,7,9]) -> (in2[9,10,3]).
    // kFtaFixedHead cites 7 only as nontrivialFactor's fixed lower bound;
    // kFtaValueHead cites it as a compared value of preorder.
    const char* kFtaFixedHead =
        "(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])"
        "(>[9](preorder[1,4,7,9])(>[10](in2[9,10,3])(nontrivialFactor[1,4,5,7,10,9]))))";
    const char* kFtaValueHead =
        "(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])"
        "(>[9](preorder[1,4,7,9])(>[10](in2[9,10,3])(preorder[1,4,7,10]))))";

    gl::Memory* ftaStepOf(gl::tests::ProverFixture& ea, const char* conj) {
        ea.proofGridBuilder.addTheoremToMemory(std::string(conj), ea.body, 0, false,
                              ea.globalDependencies);
        gl::Memory* anchorLB = ea.simpleMapStore.findChild(
            &ea.body, "(AnchorFTA[1,2,3,4,5,6,7,8])");
        ASSERT_TRUE(anchorLB != nullptr);
        gl::Memory* middleLB =
            ea.simpleMapStore.findChild(anchorLB, "(preorder[1,4,7,9])");
        ASSERT_TRUE(middleLB != nullptr);
        gl::Memory* stepLB =
            ea.simpleMapStore.findChild(middleLB, "(in2[9,10,3])");
        ASSERT_TRUE(stepLB != nullptr);
        return stepLB;
    }

    bool axedHasX7(gl::Memory& lb) {
        return ((lb).intAxedVariables).contains(((lb).nameMap).encode("x7"));
    }
}

TEST(axed_anchor_exception, slot_value_counts_only_as_combinable_argument) {
    gl::tests::ProverFixture ea(std::string("FTA"));
    const gl::ProofGridBuilder& grid = ea.proofGridBuilder;

    ASSERT_TRUE(grid.goalCitesSlotValueAsArgument("(preorder[1,4,7,9])", kFtaSlotValues));
    ASSERT_TRUE(grid.goalCitesSlotValueAsArgument("!(=[6,9])", kFtaSlotValues));
    ASSERT_FALSE(grid.goalCitesSlotValueAsArgument(
        "(nontrivialFactor[1,4,5,7,9,10])", kFtaSlotValues));
    ASSERT_FALSE(grid.goalCitesSlotValueAsArgument(
        "!(>[10](in[10,1])!(nontrivialFactor[1,4,5,7,9,10]))", kFtaSlotValues));
    ASSERT_FALSE(grid.goalCitesSlotValueAsArgument("(preorder[1,4,9,10])", kFtaSlotValues));
}

TEST(axed_anchor_exception, compiled_goal_traced_to_its_atoms) {
    gl::tests::ProverFixture ea(std::string("FTA"));
    // Some x is a nontrivial factor of 9: 2 only as the fixed lower bound.
    const std::string fixed = ea.structuralCompiler.compileExpression(
        "!(>[10](in[10,1])!(nontrivialFactor[1,4,5,7,9,10]))");
    // Some x has x * 2 = 9: 2 as a factor, a combinable argument.
    const std::string value = ea.structuralCompiler.compileExpression(
        "!(>[10](in[10,1])!(in3[10,7,9,5]))");
    // Some x has 2 + x = 9 compiles to the configured preorder[1,4,7,9].
    const std::string configured = ea.structuralCompiler.compileExpression(
        "!(>[10](in[10,1])!(in3[7,10,9,4]))");

    ASSERT_TRUE(fixed.find("existence") != std::string::npos);
    ASSERT_TRUE(value.find(">[") == std::string::npos);
    ASSERT_TRUE(configured.find(">[") == std::string::npos);
    ASSERT_FALSE(ea.proofGridBuilder.goalCitesSlotValueAsArgument(fixed, kFtaSlotValues));
    ASSERT_TRUE(ea.proofGridBuilder.goalCitesSlotValueAsArgument(value, kFtaSlotValues));
    ASSERT_TRUE(ea.proofGridBuilder.goalCitesSlotValueAsArgument(configured, kFtaSlotValues));
}

TEST(axed_anchor_exception, fixed_parameter_numeral_root_stays_set_only) {
    gl::tests::ProverFixture ea(std::string("FTA"));
    gl::Memory* stepLB = ftaStepOf(ea, kFtaFixedHead);
    ((*(stepLB)).isPartOfRecursion) = true;

    ea.proofGridBuilder.prehandleAnchor(&ea.body);

    ASSERT_TRUE(axedHasX7(*stepLB));
}

TEST(axed_anchor_exception, value_numeral_root_left_untouched) {
    gl::tests::ProverFixture ea(std::string("FTA"));
    gl::Memory* stepLB = ftaStepOf(ea, kFtaValueHead);
    ((*(stepLB)).isPartOfRecursion) = true;

    ea.proofGridBuilder.prehandleAnchor(&ea.body);

    ASSERT_FALSE(axedHasX7(*stepLB));
}
