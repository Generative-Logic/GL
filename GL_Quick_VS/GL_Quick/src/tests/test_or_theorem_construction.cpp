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
/// @brief Tests for `gl::OrTheoremBuilder::constructOrTheoremsFromPairs` —
///        single-direction OR licensing, mirror-pair dedup, and the
///        pruned-source skip.

#include "../memory.hpp"
#include <cstddef>
#include <utility>
#include "test_harness.hpp"
#include "theorem_publisher_fixture.hpp"
#include "prover_fixture.hpp"
#include "compiled_definition_fixture.hpp"

#include "../prover.hpp"

#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>

// A single proved straightened form licenses its OR: the companion is
// never proved, yet the OR is constructed, registered with method
// "or theorem", and only the proved parent is consumed.
TEST(or_theorem_construction, single_direction_constructs_or) {
    gl::tests::ProverFixture ea(std::string("Peano"));
    const std::string totality =
        "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])"
        "(>[9](in[9,1])(>[10](in[10,1])"
        "(>[]!(preorder[1,4,9,10])(preorder[1,4,10,9])))))";
    const std::string companion = ea.orTheoremBuilder.headSwitchOne(totality);
    ASSERT_TRUE(!companion.empty());

    ASSERT_TRUE(ea.theoremPublisher.appendGlobalTheorem(totality, "single chapter proof", "-1", "-1"));
    ea.orPairsFromHeadSwitch.push_back({totality, companion});

    std::set<std::string> consumed;
    const std::vector<std::string> ors = ea.orTheoremBuilder.constructOrTheoremsFromPairs(consumed);

    ASSERT_EQ(static_cast<int>(ors.size()), 1);
    ASSERT_TRUE(ors[0].find("(or") != std::string::npos);
    ASSERT_EQ(static_cast<int>(consumed.size()), 1);
    ASSERT_TRUE(consumed.count(totality) == 1);
    // Registered in both registries with the "or theorem" method and the
    // pair as parents.
    const auto& last = ea.globalTheoremList.back();
    ASSERT_TRUE(std::get<0>(last) == ors[0]);
    ASSERT_TRUE(std::get<1>(last) == std::string("or theorem"));
    ASSERT_TRUE(std::get<2>(last) == totality);
    ASSERT_TRUE(std::get<3>(last) == companion);
    ASSERT_EQ(static_cast<int>(ea.fullTheoremList.size()), 1);
    ASSERT_TRUE(std::get<0>(ea.fullTheoremList[0]) == ors[0]);
}

// Mirror pairs (x, y) and (y, x) describe the same disjunction: exactly
// one OR is constructed, and both proved parents are consumed.
TEST(or_theorem_construction, mirror_pair_dedup_and_both_consumed) {
    gl::tests::ProverFixture ea(std::string("Peano"));
    const std::string parent =
        "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])"
        "(>[7](in[7,1])(>[]!(=[7,2])(existence3[1,7,3]))))";
    const std::string mirror = ea.orTheoremBuilder.headSwitchOne(parent);
    ASSERT_TRUE(!mirror.empty());

    ASSERT_TRUE(ea.theoremPublisher.appendGlobalTheorem(parent, "single chapter proof", "-1", "-1"));
    ASSERT_TRUE(ea.theoremPublisher.appendGlobalTheorem(mirror, "single chapter proof", "-1", "-1"));
    ea.orPairsFromHeadSwitch.push_back({parent, mirror});
    ea.orPairsFromHeadSwitch.push_back({mirror, parent});

    std::set<std::string> consumed;
    const std::vector<std::string> ors = ea.orTheoremBuilder.constructOrTheoremsFromPairs(consumed);

    ASSERT_EQ(static_cast<int>(ors.size()), 1);
    ASSERT_EQ(static_cast<int>(consumed.size()), 2);
    ASSERT_TRUE(consumed.count(parent) == 1);
    ASSERT_TRUE(consumed.count(mirror) == 1);
}

// A hypothesis chain with TWO binder-free negated premises folds them BOTH
// into the OR head — three disjuncts in chain order (folded premises
// un-negated, head last), no negated premise surviving as a hypothesis,
// and a De Morgan base form of two-child conjunctions nested to the left.
// The two-branch fold left the second negated premise behind as a
// hypothesis (the trichotomy regression). The base form reloads onto the
// same or entity, as a later batch does after loading the registry.
TEST(or_theorem_construction, chain_of_negated_premises_folds_all_branches) {
    gl::tests::ProverFixture ea(std::string("Peano"));
    const std::string trichotomy =
        "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])"
        "(>[9](in[9,1])(>[10](in[10,1])"
        "(>[]!(preorder[1,4,9,10])(>[]!(=[9,10])(preorder[1,4,10,9]))))))";
    const std::string companion = ea.orTheoremBuilder.headSwitchOne(trichotomy);
    ASSERT_TRUE(!companion.empty());

    ASSERT_TRUE(ea.theoremPublisher.appendGlobalTheorem(trichotomy, "single chapter proof", "-1", "-1"));
    ea.orPairsFromHeadSwitch.push_back({trichotomy, companion});

    std::set<std::string> consumed;
    const std::vector<std::string> ors = ea.orTheoremBuilder.constructOrTheoremsFromPairs(consumed);
    ASSERT_EQ(static_cast<int>(ors.size()), 1);

    // The registered or-entity carries all three disjuncts in chain order,
    // u_-canonical, TRUE polarity.
    std::string orName;
    for (const auto& [name, le] : ea.definitions.entries()) {
        if (le.category == "or" && le.elements.size() == 3) {
            ASSERT_TRUE(le.elements[0] == "(preorder[u_1,u_2,u_3,u_4])");
            ASSERT_TRUE(le.elements[1] == "(=[u_3,u_4])");
            ASSERT_TRUE(le.elements[2] == "(preorder[u_1,u_2,u_4,u_3])");
            orName = name;
        }
    }
    ASSERT_TRUE(!orName.empty());

    // The reassembled theorem keeps only the non-negated premises — both
    // negated premises folded into the head.
    const std::string expected =
        "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])"
        "(>[9](in[9,1])(>[10](in[10,1])(" + orName + "[1,4,9,10]))))";
    ASSERT_TRUE(ors[0] == expected);

    // Base-form export expands the head to two-child conjunctions nested
    // to the left.
    const std::string base = ea.definitionExpansion.expandToBaseForm(ors[0]);
    ASSERT_TRUE(base.find(
        "!(&(&!(preorder[1,4,9,10])!(=[9,10]))!(preorder[1,4,10,9]))")
        != std::string::npos);

    // Reload: register the or's reuse identity the way loading a GL binary
    // registers a generated entry (the canonical key of its elements), then
    // compile the base form. The three disjuncts are collected and the
    // registered entity is reused.
    ea.compilationReuse.registerStoredCompilation(orName, false);
    std::string reloaded = base;
    ea.structuralCompiler.precompileStructuralOperators(reloaded);
    ASSERT_TRUE(reloaded == expected);
}

// A conjunction has two children, so an or of three disjuncts nests. The
// left nesting compiles to ONE three-element or (the conjuncts are collected
// flat, in leaf order); once that or is registered the way a later batch's
// GL-binary load registers it (the canonical key of its elements), the right
// nesting compiles to the same or — its collected element list is the same.
TEST(or_theorem_construction, both_nestings_compile_to_one_three_element_or) {
    gl::tests::ProverFixture ea(std::string("Peano"));
    const std::vector<std::string> elements = {
        "(preorder[u_1,u_2,u_3,u_4])", "(=[u_3,u_4])", "(preorder[u_1,u_2,u_4,u_3])"};
    const std::string left = ea.structuralCompiler.compileExpression(
        "!(&(&!(preorder[1,4,9,10])!(=[9,10]))!(preorder[1,4,10,9]))");
    const std::string name = left.substr(1, left.find('[') - 1);
    ASSERT_TRUE(left == "(" + name + "[1,4,9,10])");
    const auto& entity = ea.definitions.entries().at(name);
    ASSERT_TRUE(entity.category == "or");
    ASSERT_TRUE(entity.elements == elements);

    ea.compilationReuse.registerStoredCompilation(name, false);
    const std::string right = ea.structuralCompiler.compileExpression(
        "!(&!(preorder[1,4,9,10])(&!(=[9,10])!(preorder[1,4,10,9])))");
    ASSERT_TRUE(right == left);
}

// The in-run seam folds a freshly proved row into its or theorem,
// registers it with method "or theorem", and queues exactly one
// implication-compact broadcast on pendingCompactionQueue.
TEST(or_theorem_construction, in_run_constructs_registers_and_queues) {
    gl::tests::ProverFixture ea(std::string("Peano"));
    const std::string totality =
        "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])"
        "(>[9](in[9,1])(>[10](in[10,1])"
        "(>[]!(preorder[1,4,9,10])(preorder[1,4,10,9])))))";
    ASSERT_TRUE(ea.theoremPublisher.appendGlobalTheorem(totality, "single chapter proof", "-1", "-1"));

    ea.orTheoremBuilder.constructOrTheoremsInRun();

    // One new "or theorem" row appended behind the source row.
    ASSERT_EQ(static_cast<int>(ea.globalTheoremList.size()), 2);
    const auto& last = ea.globalTheoremList.back();
    ASSERT_TRUE(std::get<1>(last) == std::string("or theorem"));
    ASSERT_TRUE(std::get<2>(last) == totality);
    ASSERT_TRUE(std::get<0>(last).find("(or") != std::string::npos);
    ASSERT_EQ(static_cast<int>(ea.fullTheoremList.size()), 1);
    // The or theorem is implication-shaped, so it is queued for the
    // deferred-compaction broadcast.
    ASSERT_EQ(static_cast<int>(ea.pendingCompactionQueue.size()), 1);
    ASSERT_TRUE(std::get<0>(ea.pendingCompactionQueue[0]) == std::get<0>(last));
}

// The in-run broadcast carries the proved-theorem documentation every other
// broadcast theorem carries: the root's mailOut holds ("theorem", []) at main
// for the or theorem, so receivers can ground it as a chapter leaf.
TEST(or_theorem_construction, in_run_broadcast_carries_theorem_origin) {
    gl::tests::ProverFixture ea(std::string("Peano"));
    const std::string totality =
        "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])"
        "(>[9](in[9,1])(>[10](in[10,1])"
        "(>[]!(preorder[1,4,9,10])(preorder[1,4,10,9])))))";
    ASSERT_TRUE(ea.theoremPublisher.appendGlobalTheorem(totality, "single chapter proof", "-1", "-1"));
    ASSERT_TRUE(((ea.body).mailOut).empty());

    ea.orTheoremBuilder.constructOrTheoremsInRun();

    const std::string orThm = std::get<0>(ea.globalTheoremList.back());
    ASSERT_TRUE(std::get<1>(ea.globalTheoremList.back()) == std::string("or theorem"));
    const gl::Mail mail = gl::routingMailOutToHeap(((ea.body).mailOut), ((ea.body).mailOutInterner));
    ASSERT_EQ(static_cast<int>(((mail).exprOriginMap).size()), 1);
    const auto it = ((mail).exprOriginMap).find(gl::ExpressionWithValidity(orThm, "main"));
    ASSERT_TRUE(it != ((mail).exprOriginMap).end());
    ASSERT_EQ(static_cast<int>(it->second.size()), 1);
    ASSERT_TRUE(it->second[0].first == std::string("theorem"));
    ASSERT_TRUE(it->second[0].second.empty());
    // Documentation only: the broadcast's statements ride the compaction drain.
    ASSERT_TRUE(mail.statements.empty());
}

// Without history tracking the seam records no documentation, exactly like
// updateGlobalSingleChapter; the or theorem is still registered and queued.
TEST(or_theorem_construction, in_run_without_history_adds_no_origin) {
    gl::tests::ProverFixture ea(std::string("Peano"));
    ea.parameters.trackHistory = false;
    const std::string totality =
        "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])"
        "(>[9](in[9,1])(>[10](in[10,1])"
        "(>[]!(preorder[1,4,9,10])(preorder[1,4,10,9])))))";
    ASSERT_TRUE(ea.theoremPublisher.appendGlobalTheorem(totality, "single chapter proof", "-1", "-1"));

    ea.orTheoremBuilder.constructOrTheoremsInRun();

    ASSERT_TRUE(std::get<1>(ea.globalTheoremList.back()) == std::string("or theorem"));
    ASSERT_EQ(static_cast<int>(ea.pendingCompactionQueue.size()), 1);
    ASSERT_TRUE(((ea.body).mailOut).empty());
}

// The documentation door merges a seam's origin lines into the root's
// mailOut unchanged.
TEST(or_theorem_construction, merge_theorem_origin_mail_reaches_root) {
    gl::tests::ProverFixture ea(std::string("Peano"));
    const gl::ExpressionWithValidity theorem(
        "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[9](in[9,1])(or0[9,1])))", "main");
    gl::Mail documentation;
    ((documentation).exprOriginMap)[theorem].push_back(
        std::make_pair(std::string("theorem"), std::vector<gl::ExpressionWithValidity>()));

    ea.theoremPublisher.mergeTheoremOriginMailIntoRoot(documentation);

    const gl::Mail mail = gl::routingMailOutToHeap(((ea.body).mailOut), ((ea.body).mailOutInterner));
    ASSERT_EQ(static_cast<int>(((mail).exprOriginMap).size()), 1);
    ASSERT_EQ(static_cast<int>(((mail).exprOriginMap).at(theorem).size()), 1);
    ASSERT_TRUE(((mail).exprOriginMap).at(theorem)[0].first == std::string("theorem"));
    ASSERT_TRUE(mail.statements.empty());
}

// A window that documented nothing leaves the root's mailOut untouched.
TEST(or_theorem_construction, merge_theorem_origin_mail_empty_is_noop) {
    gl::tests::ProverFixture ea(std::string("Peano"));
    ea.theoremPublisher.mergeTheoremOriginMailIntoRoot(gl::Mail());
    ASSERT_TRUE(((ea.body).mailOut).empty());
    ASSERT_FALSE(((ea.body).mailOutPending));
}

// A second in-run call scans nothing new: every row is in the scanned
// set, so no duplicate or row and no duplicate broadcast entry appears.
TEST(or_theorem_construction, in_run_rescan_is_idempotent) {
    gl::tests::ProverFixture ea(std::string("Peano"));
    const std::string totality =
        "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])"
        "(>[9](in[9,1])(>[10](in[10,1])"
        "(>[]!(preorder[1,4,9,10])(preorder[1,4,10,9])))))";
    ASSERT_TRUE(ea.theoremPublisher.appendGlobalTheorem(totality, "single chapter proof", "-1", "-1"));

    ea.orTheoremBuilder.constructOrTheoremsInRun();
    const std::size_t rows = ea.globalTheoremList.size();
    const std::size_t queued = ea.pendingCompactionQueue.size();

    ea.orTheoremBuilder.constructOrTheoremsInRun();

    ASSERT_EQ(static_cast<int>(ea.globalTheoremList.size()),
              static_cast<int>(rows));
    ASSERT_EQ(static_cast<int>(ea.pendingCompactionQueue.size()),
              static_cast<int>(queued));
}

// The in-run seam records the companion of the or theorem it registers, and
// a later proof of that companion is withheld at the registration drain:
// no global-list row, no broadcast entry.
TEST(or_theorem_construction, in_run_companion_proved_later_is_withheld) {
    gl::tests::ProverFixture ea(std::string("Peano"));
    const std::string parent =
        "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])"
        "(>[9](in[9,1])(>[10](in[10,1])"
        "(>[]!(preorder[1,4,9,10])(preorder[1,4,10,9])))))";
    const std::string companion = ea.orTheoremBuilder.headSwitchOne(parent);
    ASSERT_TRUE(!companion.empty());
    ASSERT_TRUE(ea.theoremPublisher.appendGlobalTheorem(parent, "single chapter proof", "-1", "-1"));

    ea.orTheoremBuilder.constructOrTheoremsInRun();
    ASSERT_EQ(static_cast<int>(ea.orInRunCompanions.count(companion)), 1);
    ASSERT_EQ(static_cast<int>(ea.orInRunCompanions.count(parent)), 0);
    const std::size_t rows = ea.globalTheoremList.size();
    const std::size_t queued = ea.pendingCompactionQueue.size();

    gl::tests::TheoremPublisherFixture::updateGlobalSingleChapter(ea.theoremPublisher, companion, 0, nullptr, true);

    ASSERT_EQ(static_cast<int>(ea.globalTheoremList.size()), static_cast<int>(rows));
    ASSERT_EQ(static_cast<int>(ea.pendingCompactionQueue.size()),
              static_cast<int>(queued));
    for (const auto& tpl : ea.globalTheoremList) {
        ASSERT_FALSE(std::get<0>(tpl) == companion);
    }
}

// The seam retires the logic blocks of the companion it subsumes: both
// contradiction twins at once, the goal block through the settled-goal inbox
// (seed negate(head), as a firing complement twin deposits), so its own drain
// closes the companion's goal and the opposite-head goal; the seam settles the
// prune it owes, and the next sweep deactivates the whole chain.
TEST(or_theorem_construction, in_run_retires_companion_logic_blocks) {
    gl::tests::ProverFixture ea(std::string("Peano"));
    ea.parameters.try_contradiction = true;
    ea.parameters.try_contradiction_negated_head = true;
    const std::string parent =
        "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])"
        "(>[9](in[9,1])(>[10](in[10,1])"
        "(>[]!(preorder[1,4,9,10])(preorder[1,4,10,9])))))";
    const std::string companion = ea.orTheoremBuilder.headSwitchOne(parent);
    ASSERT_TRUE(companion ==
        "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])"
        "(>[9](in[9,1])(>[10](in[10,1])"
        "(>[]!(preorder[1,4,10,9])(preorder[1,4,9,10])))))");
    const std::string opposite =
        "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])"
        "(>[9](in[9,1])(>[10](in[10,1])"
        "(>[]!(preorder[1,4,10,9])!(preorder[1,4,9,10])))))";
    ea.proofGridBuilder.addTheoremToMemory(companion, ea.body, 0, false, ea.globalDependencies);
    ea.proofGridBuilder.addTheoremToMemory(opposite, ea.body, 0, false, ea.globalDependencies);

    gl::Memory* anchorLB =
        ea.simpleMapStore.findChild(&ea.body, "(AnchorPeano[1,2,3,4,5,6])");
    gl::Memory* nineLB = ea.simpleMapStore.findChild(anchorLB, "(in[9,1])");
    gl::Memory* tenLB = ea.simpleMapStore.findChild(nineLB, "(in[10,1])");
    gl::Memory* goalLB = ea.simpleMapStore.findChild(tenLB, "!(preorder[1,4,10,9])");
    ASSERT_TRUE(goalLB != nullptr);
    const std::string head = "(preorder[1,4,9,10])";
    gl::Memory* verbatimLB =
        ea.simpleMapStore.findChild(goalLB, "__contradiction__" + head);
    gl::Memory* complementLB =
        ea.simpleMapStore.findChild(goalLB, "__contradiction__!" + head);
    ASSERT_TRUE(verbatimLB != nullptr);
    ASSERT_TRUE(complementLB != nullptr);
    ASSERT_TRUE(gl::lookupToBeProved(((*(goalLB)).intToBeProved), ((*(goalLB)).nameMap),
                                     head, "main") != 0);
    ASSERT_TRUE(gl::lookupToBeProved(((*(goalLB)).intToBeProved), ((*(goalLB)).nameMap),
                                     "!" + head, "main") != 0);
    ((*(goalLB)).hasWork) = false;

    ASSERT_TRUE(ea.theoremPublisher.appendGlobalTheorem(parent, "single chapter proof", "-1", "-1"));
    ea.orTheoremBuilder.constructOrTheoremsInRun();

    ASSERT_EQ(static_cast<int>(ea.orInRunCompanions.count(companion)), 1);
    ASSERT_FALSE(ea.deactivationSweepOwed);
    ASSERT_FALSE(((*(verbatimLB)).isActive));
    ASSERT_FALSE(((*(complementLB)).isActive));
    // The goals close at the goal block's own drain, so it is still active.
    ASSERT_TRUE(((*(goalLB)).isActive));
    ASSERT_TRUE(((*(goalLB)).hasWork));
    ASSERT_EQ(((*(goalLB)).pendingDisprovedGoals).count(), 1);
    ASSERT_TRUE(((*(goalLB)).pendingDisprovedGoals).view(1).toStdString() == "!" + head);

    ea.goalProcessor.drainDisprovedGoals(*goalLB);
    ASSERT_TRUE(gl::lookupToBeProved(((*(goalLB)).intToBeProved), ((*(goalLB)).nameMap),
                                     head, "main") == 0);
    ASSERT_TRUE(gl::lookupToBeProved(((*(goalLB)).intToBeProved), ((*(goalLB)).nameMap),
                                     "!" + head, "main") == 0);

    ea.scopeLifecycle.deactivateRecursively();
    ASSERT_FALSE(((*(goalLB)).isActive));
    ASSERT_FALSE(((*(anchorLB)).isActive));
}

// A companion that was never scheduled has no logic blocks: the retirement
// is a no-op and owes no prune. An inactive goal block receives no seed.
TEST(or_theorem_construction, retire_companion_without_active_goal_block) {
    gl::tests::ProverFixture ea(std::string("Peano"));
    const std::string companion =
        "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])"
        "(>[9](in[9,1])(>[10](in[10,1])"
        "(>[]!(preorder[1,4,10,9])(preorder[1,4,9,10])))))";

    ea.theoremPublisher.retireSubsumedCompanion(companion);
    ASSERT_FALSE(ea.deactivationSweepOwed);
    ASSERT_TRUE(ea.simpleMapStore.findChild(&ea.body, "(AnchorPeano[1,2,3,4,5,6])")
        == nullptr);

    ea.proofGridBuilder.addTheoremToMemory(companion, ea.body, 0, false, ea.globalDependencies);
    gl::Memory* anchorLB =
        ea.simpleMapStore.findChild(&ea.body, "(AnchorPeano[1,2,3,4,5,6])");
    gl::Memory* nineLB = ea.simpleMapStore.findChild(anchorLB, "(in[9,1])");
    gl::Memory* tenLB = ea.simpleMapStore.findChild(nineLB, "(in[10,1])");
    gl::Memory* goalLB = ea.simpleMapStore.findChild(tenLB, "!(preorder[1,4,10,9])");
    ASSERT_TRUE(goalLB != nullptr);
    ((*(goalLB)).isActive) = false;

    ea.theoremPublisher.retireSubsumedCompanion(companion);
    ASSERT_EQ(((*(goalLB)).pendingDisprovedGoals).count(), 0);
    // The walk still ran; settling its prune is the calling seam's duty.
    ASSERT_TRUE(ea.deactivationSweepOwed);
}

// A mirror row proved LATER reconstructs the same or through the registry
// reuse; the string dedup gates it — one or row, one broadcast entry.
TEST(or_theorem_construction, in_run_mirror_scanned_later_no_double) {
    gl::tests::ProverFixture ea(std::string("Peano"));
    const std::string parent =
        "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])"
        "(>[9](in[9,1])(>[10](in[10,1])"
        "(>[]!(preorder[1,4,9,10])(preorder[1,4,10,9])))))";
    const std::string mirror = ea.orTheoremBuilder.headSwitchOne(parent);
    ASSERT_TRUE(!mirror.empty());

    ASSERT_TRUE(ea.theoremPublisher.appendGlobalTheorem(parent, "single chapter proof", "-1", "-1"));
    ea.orTheoremBuilder.constructOrTheoremsInRun();
    ASSERT_TRUE(ea.theoremPublisher.appendGlobalTheorem(mirror, "single chapter proof", "-1", "-1"));
    ea.orTheoremBuilder.constructOrTheoremsInRun();

    int orRows = 0;
    for (const auto& tpl : ea.globalTheoremList) {
        if (std::get<1>(tpl) == std::string("or theorem")) ++orRows;
    }
    ASSERT_EQ(orRows, 1);
    ASSERT_EQ(static_cast<int>(ea.pendingCompactionQueue.size()), 1);
    ASSERT_EQ(static_cast<int>(ea.fullTheoremList.size()), 1);
}

// After the in-run seam built the or, the export seam still reports the
// or in its return vector and consumes the parents — but registers and
// prints nothing new (idempotence across the two seams).
TEST(or_theorem_construction, export_seam_idempotent_after_in_run) {
    gl::tests::ProverFixture ea(std::string("Peano"));
    const std::string totality =
        "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])"
        "(>[9](in[9,1])(>[10](in[10,1])"
        "(>[]!(preorder[1,4,9,10])(preorder[1,4,10,9])))))";
    const std::string companion = ea.orTheoremBuilder.headSwitchOne(totality);
    ASSERT_TRUE(!companion.empty());

    ASSERT_TRUE(ea.theoremPublisher.appendGlobalTheorem(totality, "single chapter proof", "-1", "-1"));
    ea.orTheoremBuilder.constructOrTheoremsInRun();
    const std::size_t rowsAfterInRun = ea.globalTheoremList.size();
    ASSERT_EQ(static_cast<int>(ea.fullTheoremList.size()), 1);

    ea.orPairsFromHeadSwitch.push_back({totality, companion});
    std::set<std::string> consumed;
    const std::vector<std::string> ors = ea.orTheoremBuilder.constructOrTheoremsFromPairs(consumed);

    // Return vector + subsumption fire; registration does not repeat.
    ASSERT_EQ(static_cast<int>(ors.size()), 1);
    ASSERT_TRUE(consumed.count(totality) == 1);
    ASSERT_EQ(static_cast<int>(ea.globalTheoremList.size()),
              static_cast<int>(rowsAfterInRun));
    ASSERT_EQ(static_cast<int>(ea.fullTheoremList.size()), 1);
}

// Compressor mode is a defined no-op for the in-run seam.
TEST(or_theorem_construction, in_run_compressor_mode_noop) {
    gl::tests::ProverFixture ea(std::string("Peano"));
    ea.parameters.compressor_mode = true;
    const std::string totality =
        "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])"
        "(>[9](in[9,1])(>[10](in[10,1])"
        "(>[]!(preorder[1,4,9,10])(preorder[1,4,10,9])))))";
    ASSERT_TRUE(ea.theoremPublisher.appendGlobalTheorem(totality, "single chapter proof", "-1", "-1"));

    ea.orTheoremBuilder.constructOrTheoremsInRun();

    ASSERT_EQ(static_cast<int>(ea.globalTheoremList.size()), 1);
    ASSERT_EQ(static_cast<int>(ea.pendingCompactionQueue.size()), 0);
}

// findOrMintOrOperator: a fresh element list mints or<orCounter> into BOTH
// registries with the canonical u_ signature and bumps the counter; a
// second call with the identical list reuses the entry (I-23) and leaves
// the counter untouched.
TEST(or_theorem_construction, find_or_mint_reuse_and_mint) {
    gl::tests::ProverFixture ea(std::string("Peano"));
    const int counterBefore = gl::CompiledDefinitionFixture::orCounter(ea.definitions);
    const std::vector<std::string> elems = {
        "(zztesta[u_1,u_2])", "(zztestb[u_2,u_1])" };

    bool minted = false;
    const std::string name = ea.orRuleCompiler.findOrMintOrOperator(elems, 2, &minted);
    ASSERT_TRUE(minted);
    ASSERT_EQ(gl::CompiledDefinitionFixture::orCounter(ea.definitions), counterBefore + 1);
    const auto it = gl::CompiledDefinitionFixture::compiledExpressions(ea.definitions).find(name);
    ASSERT_TRUE(it != gl::CompiledDefinitionFixture::compiledExpressions(ea.definitions).end());
    ASSERT_TRUE(it->second.category == "or");
    ASSERT_TRUE(it->second.elements == elems);
    ASSERT_TRUE(it->second.signature == "(" + name + "[u_1,u_2])");
    ASSERT_TRUE(gl::CompiledDefinitionFixture::coreExpressionMap(ea.definitions).count(name) == 1);

    bool minted2 = true;
    const std::string name2 = ea.orRuleCompiler.findOrMintOrOperator(elems, 2, &minted2);
    ASSERT_TRUE(!minted2);
    ASSERT_TRUE(name2 == name);
    ASSERT_EQ(gl::CompiledDefinitionFixture::orCounter(ea.definitions), counterBefore + 1);
}

// Helper: count or-category entries in the compile registry.
static int countOrEntries(const gl::tests::ProverFixture& ea) {
    int n = 0;
    for (const auto& [name, le] : ea.definitions.entries()) {
        (void)name;
        if (le.category == "or") ++n;
    }
    return n;
}

// preMintReducedOrs on a 3-ary or (one disjunct negated — TRUE polarity
// preserved, I-175): exactly the three 2-ary reductions appear, each
// findable via compiledOrByElements, and a re-run mints nothing
// (idempotence).
TEST(or_theorem_construction, premint_three_ary_closure_and_idempotence) {
    gl::tests::ProverFixture ea(std::string("Peano"));
    // Flush reductions of any pre-registered pool ors first so the count
    // delta below isolates the synthetic entry.
    ea.orRuleCompiler.preMintReducedOrs();

    const std::vector<std::string> base = {
        "(zzalpha[u_1])", "!(zzbeta[u_1])", "(zzgamma[u_1])" };
    ea.orRuleCompiler.findOrMintOrOperator(base, 1);
    const int before = countOrEntries(ea);

    ea.orRuleCompiler.preMintReducedOrs();

    // The three reductions, in leaf order, u_-renumbered (trivially here).
    const std::vector<std::vector<std::string>> expected = {
        { "!(zzbeta[u_1])", "(zzgamma[u_1])" },
        { "(zzalpha[u_1])", "(zzgamma[u_1])" },
        { "(zzalpha[u_1])", "!(zzbeta[u_1])" } };
    for (const auto& redu : expected) {
        gl::StrSpan spans[2] = { gl::StrSpan(redu[0]), gl::StrSpan(redu[1]) };
        ASSERT_TRUE(ea.compilationReuse.compiledOrByElements(spans, 2) != nullptr);
    }
    ASSERT_EQ(countOrEntries(ea), before + 3);

    // A list never registered stays a defined miss.
    const std::string never1 = "(zznever[u_1])";
    const std::string never2 = "(zznever2[u_1])";
    gl::StrSpan missSpans[2] = { gl::StrSpan(never1), gl::StrSpan(never2) };
    ASSERT_TRUE(ea.compilationReuse.compiledOrByElements(missSpans, 2) == nullptr);

    ea.orRuleCompiler.preMintReducedOrs();
    ASSERT_EQ(countOrEntries(ea), before + 3);
}

// preMintReducedOrs on a 4-ary or reaches the single-elimination closure
// through the worklist: 4 ternary + 6 binary reductions (dedup across
// paths), all in one call.
TEST(or_theorem_construction, premint_four_ary_worklist_closure) {
    gl::tests::ProverFixture ea(std::string("Peano"));
    ea.orRuleCompiler.preMintReducedOrs();

    const std::vector<std::string> base = {
        "(zza[u_1])", "(zzb[u_1])", "(zzc[u_1])", "(zzd[u_1])" };
    ea.orRuleCompiler.findOrMintOrOperator(base, 1);
    const int before = countOrEntries(ea);

    ea.orRuleCompiler.preMintReducedOrs();
    ASSERT_EQ(countOrEntries(ea), before + 10);

    // Spot-check one ternary and one binary from different elimination
    // paths land on the SAME registered binary (dedup).
    {
        gl::StrSpan t[3] = { gl::StrSpan(base[0]), gl::StrSpan(base[1]),
                             gl::StrSpan(base[2]) };
        ASSERT_TRUE(ea.compilationReuse.compiledOrByElements(t, 3) != nullptr);
        gl::StrSpan b[2] = { gl::StrSpan(base[0]), gl::StrSpan(base[1]) };
        ASSERT_TRUE(ea.compilationReuse.compiledOrByElements(b, 2) != nullptr);
    }
}

// flattenRegistryOrLeaves expands a nested or-child through the
// signature-to-instance substitution (arg permutation included), and
// preMintReducedOrs selects by FLATTENED leaf count — a 2-element entry
// with an or-child still pre-mints its 3 reductions, reusing the child
// where the reduction equals it.
TEST(or_theorem_construction, premint_nested_or_flattens_and_reuses_child) {
    gl::tests::ProverFixture ea(std::string("Peano"));
    ea.orRuleCompiler.preMintReducedOrs();

    const std::vector<std::string> childElems = {
        "(zzalpha[u_1])", "(zzbeta[u_2])" };
    const std::string child = ea.orRuleCompiler.findOrMintOrOperator(childElems, 2);

    // Parent embeds the child with PERMUTED instance args (u_2, u_1).
    const std::vector<std::string> parentElems = {
        "(zzgamma[u_1])", "(" + child + "[u_2,u_1])" };
    const std::string parent = ea.orRuleCompiler.findOrMintOrOperator(parentElems, 2);

    std::vector<std::string> leaves;
    const auto pit = gl::CompiledDefinitionFixture::compiledExpressions(ea.definitions).find(parent);
    ASSERT_TRUE(pit != gl::CompiledDefinitionFixture::compiledExpressions(ea.definitions).end());
    ea.orRuleCompiler.flattenRegistryOrLeaves(pit->second, leaves);
    ASSERT_EQ(static_cast<int>(leaves.size()), 3);
    ASSERT_TRUE(leaves[0] == "(zzgamma[u_1])");
    ASSERT_TRUE(leaves[1] == "(zzalpha[u_2])");
    ASSERT_TRUE(leaves[2] == "(zzbeta[u_1])");

    const int before = countOrEntries(ea);
    ea.orRuleCompiler.preMintReducedOrs();
    // minus-gamma renumbers to exactly the child's element list (reuse);
    // the other two reductions are fresh.
    ASSERT_EQ(countOrEntries(ea), before + 2);
    {
        gl::StrSpan g1[2] = { gl::StrSpan(childElems[0]),
                              gl::StrSpan(childElems[1]) };
        const std::string* hit = ea.compilationReuse.compiledOrByElements(g1, 2);
        ASSERT_TRUE(hit != nullptr);
        ASSERT_TRUE(*hit == child);
    }
    {
        const std::string r1 = "(zzgamma[u_1])";
        const std::string r2 = "(zzbeta[u_1])";
        gl::StrSpan g2[2] = { gl::StrSpan(r1), gl::StrSpan(r2) };
        ASSERT_TRUE(ea.compilationReuse.compiledOrByElements(g2, 2) != nullptr);
    }
    {
        const std::string r1 = "(zzgamma[u_1])";
        const std::string r2 = "(zzalpha[u_2])";
        gl::StrSpan g3[2] = { gl::StrSpan(r1), gl::StrSpan(r2) };
        ASSERT_TRUE(ea.compilationReuse.compiledOrByElements(g3, 2) != nullptr);
    }
}

// renumberULeaves maps the surviving u_ tokens to u_1..u_m in
// first-appearance order with SIMULTANEOUS substitution (overlapping
// old/new names rewrite correctly) and returns the reduced arity.
TEST(or_theorem_construction, renumber_u_leaves_first_appearance) {
    std::vector<std::string> leaves = {
        "(zzx[u_3,u_4])", "(zzy[u_4,u_1])" };
    const int arity = gl::OrRuleCompiler::renumberULeaves(leaves);
    ASSERT_EQ(arity, 3);
    ASSERT_TRUE(leaves[0] == "(zzx[u_1,u_2])");
    ASSERT_TRUE(leaves[1] == "(zzy[u_2,u_3])");
}

// A pair whose source theorem is not (or no longer) in globalTheoremList —
// compression pruning or vacuity retraction — licenses nothing.
TEST(or_theorem_construction, pruned_source_is_skipped) {
    gl::tests::ProverFixture ea(std::string("Peano"));
    const std::string totality =
        "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])"
        "(>[9](in[9,1])(>[10](in[10,1])"
        "(>[]!(preorder[1,4,9,10])(preorder[1,4,10,9])))))";
    const std::string companion = ea.orTheoremBuilder.headSwitchOne(totality);
    ASSERT_TRUE(!companion.empty());

    // The source is deliberately NOT appended to globalTheoremList.
    ea.orPairsFromHeadSwitch.push_back({totality, companion});

    std::set<std::string> consumed;
    const std::vector<std::string> ors = ea.orTheoremBuilder.constructOrTheoremsFromPairs(consumed);

    ASSERT_EQ(static_cast<int>(ors.size()), 0);
    ASSERT_EQ(static_cast<int>(consumed.size()), 0);
    ASSERT_EQ(static_cast<int>(ea.globalTheoremList.size()), 0);
}

// forEachCompiledOr (the I-137 read fence behind the subset-exclusion
// registry scan): visits or-category entries only, and a true-returning
// visitor stops the scan after that entry.
TEST(or_theorem_construction, for_each_compiled_or_visits_and_stops) {
    gl::tests::ProverFixture ea(std::string("Peano"));
    const std::vector<std::string> parent = {
        "(zzfea[u_1])", "(zzfeb[u_1])" };
    const std::string name = ea.orRuleCompiler.findOrMintOrOperator(parent, 1);

    bool sawMinted = false;
    bool nonOrSeen = false;
    ea.definitions.forEachCompiledOr([&](const std::string& n,
                             const gl::LogicalEntity& le) -> bool {
        if (le.category != "or") nonOrSeen = true;
        if (n == name) sawMinted = true;
        return false;
    });
    ASSERT_TRUE(sawMinted);
    ASSERT_FALSE(nonOrSeen);

    int32_t visited = 0;
    ea.definitions.forEachCompiledOr([&](const std::string&,
                             const gl::LogicalEntity&) -> bool {
        ++visited;
        return true;
    });
    ASSERT_EQ(visited, 1);
}

// A later premise that binds a variable of the head moves the negated head
// right after that premise, so it is never used before its binder (FTA2 row
// 10's shape); the swapped premise's own place is dropped. A later binder
// the head does not mention keeps the swap in place.
TEST(or_theorem_construction, head_switch_moves_the_new_premise_after_the_binder_it_needs) {
    gl::tests::ProverFixture ea(std::string("Peano"));
    const std::string bound =
        "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])"
        "(>[9](in[9,1])(>[]!(=[9,2])"
        "(>[10](in[10,1])!(preorder[1,4,10,9])))))";
    ASSERT_TRUE(ea.orTheoremBuilder.headSwitchOne(bound)
        == "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])"
           "(>[9](in[9,1])(>[10](in[10,1])"
           "(>[](preorder[1,4,10,9])(=[9,2])))))");
    const std::string unrelated =
        "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])"
        "(>[9](in[9,1])(>[]!(=[9,2])"
        "(>[10](in[10,1])(preorder[1,4,2,9])))))";
    ASSERT_TRUE(ea.orTheoremBuilder.headSwitchOne(unrelated)
        == "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])"
           "(>[9](in[9,1])(>[]!(preorder[1,4,2,9])"
           "(>[10](in[10,1])(=[9,2])))))");
}
