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
/// @brief Tests for the incubator mail topology
///        (D-332): the shared anchor-LB
///        predicate `isAnchorLb`, the mail-ancestor collection
///        `collectMailAncestors` (whole chain outside incubator mode, stops at
///        the anchor LB inside it), `incubatorAnchorLb`, and the routing of a
///        `broadcastTheorems` batch through the root's log (anchor LB only)
///        versus the anchor LB's log (anchor LB and every LB below it), and
///        the anchor LB's history relay (`relaysAbsorbedHistory`: the origin
///        rows of an absorbed root batch reach the anchor's `mailOut`, the
///        statements do not).

#include "test_harness.hpp"

#include "../prover.hpp"

namespace {
    const char* kConj =
        "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8](in2[7,8,3])(=[7,8])))";
    const char* kThm2 =
        "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8](in2[7,8,3])(in[8,1])))";

    // Shape-routing fixtures (AnchorPeano[1,2,3,4,5,6]: the slot values 1..6
    // are the anchor's digits for the equality-head rule).
    const char* kSingle =
        "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(in[2,1]))";
    const char* kEvalFact =
        "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7](in2[2,7,3])(=[7,6])))";
    const char* kEvalFactMirror =
        "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7](in2[2,7,3])(=[6,7])))";
    const char* kNestedHead =
        "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7](in2[2,7,3])(in[7,1])))";
    // The equality's other side must be an anchor slot value; an inner-bound
    // variable is not one (any anchor slot counts -- the function-symbol
    // slots are never equated, so the slot values an equality can mention
    // are the digits).
    const char* kNonDigit =
        "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8](in2[2,7,3])(=[7,8])))";
    // A loaded theorem of an earlier incubator batch keeps that batch's anchor
    // (bridged in-run by `current anchor -> previous anchor`); AnchorGauss is
    // not a core expression of the Peano configuration.
    const char* kForeignAnchorFact =
        "(>[1,2,3,4,5,6,7,8](AnchorGauss[1,2,3,4,5,6,7,8])(>[9](in2[2,9,3])(=[9,6])))";
    const char* kForeignAnchorFree =
        "(>[1,2,3,4,5,6,7,8](AnchorGauss[1,2,3,4,5,6,7,8])(>[9,10](in2[2,9,3])(=[9,10])))";
    const char* kThreePremises =
        "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8](in2[7,8,3])(>[](in[7,1])(in[8,1]))))";

    // The premise-LB chain: root -> (AnchorPeano[...], level 0) -> (in2[7,8,3],
    // level 1). Registered outside incubator mode so the grid shape is the
    // plain one; the tests flip `incubator_mode` afterwards.
    void registerChain(gl::ExpressionAnalyzer& ea) {
        ea.parameters.incubator_mode = false;
        ea.parameters.try_contradiction = false;
        ea.parameters.try_contradiction_negated_head = false;
        ea.parameters.compressor_mode = false;
        ea.addTheoremToMemory(std::string(kConj), ea.body, 0, false,
                              ea.globalDependencies);
    }

    gl::Memory* anchorOf(gl::ExpressionAnalyzer& ea) {
        gl::Memory* anchorLB = ea.simpleMapStore.findChild(
            &ea.body, "(AnchorPeano[1,2,3,4,5,6])");
        ASSERT_TRUE(anchorLB != nullptr);
        return anchorLB;
    }

    gl::Memory* innerOf(gl::ExpressionAnalyzer& ea) {
        gl::Memory* innerLB =
            ea.simpleMapStore.findChild(anchorOf(ea), "(in2[7,8,3])");
        ASSERT_TRUE(innerLB != nullptr);
        return innerLB;
    }

    // Registers the three chain LBs with the analyzer's MailLog exactly as
    // buildGrid does (collectMailAncestors -> registerLb).
    void registerMailEdges(gl::ExpressionAnalyzer& ea) {
        std::vector<const gl::Memory*> ancestors;
        gl::Memory* chain[3] = { &ea.body, anchorOf(ea), innerOf(ea) };
        for (gl::Memory* lb : chain) {
            ea.collectMailAncestors(lb, ancestors);
            ea.mailLog.registerLb(lb, ancestors);
        }
    }
}

TEST(incubator_mail_routing, is_anchor_lb_classifies_root_anchor_inner) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    registerChain(ea);
    ASSERT_FALSE(ea.isAnchorLb(ea.body));
    ASSERT_TRUE(ea.isAnchorLb(*anchorOf(ea)));
    ASSERT_FALSE(ea.isAnchorLb(*innerOf(ea)));
}

TEST(incubator_mail_routing, mail_ancestors_whole_chain_outside_incubator_mode) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    registerChain(ea);
    gl::Memory* anchorLB = anchorOf(ea);
    gl::Memory* innerLB = innerOf(ea);
    std::vector<const gl::Memory*> out;

    ea.collectMailAncestors(innerLB, out);
    ASSERT_EQ(out.size(), static_cast<std::size_t>(2));
    ASSERT_TRUE(out[0] == anchorLB);
    ASSERT_TRUE(out[1] == &ea.body);

    ea.collectMailAncestors(anchorLB, out);
    ASSERT_EQ(out.size(), static_cast<std::size_t>(1));
    ASSERT_TRUE(out[0] == &ea.body);

    ea.collectMailAncestors(&ea.body, out);
    ASSERT_TRUE(out.empty());
}

TEST(incubator_mail_routing, mail_ancestors_stop_at_anchor_in_incubator_mode) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    registerChain(ea);
    ea.parameters.incubator_mode = true;
    gl::Memory* anchorLB = anchorOf(ea);
    gl::Memory* innerLB = innerOf(ea);
    std::vector<const gl::Memory*> out;

    // Below the anchor: the anchor LB only, never the root sentinel.
    ea.collectMailAncestors(innerLB, out);
    ASSERT_EQ(out.size(), static_cast<std::size_t>(1));
    ASSERT_TRUE(out[0] == anchorLB);

    // The anchor LB keeps its root edge; the root has none.
    ea.collectMailAncestors(anchorLB, out);
    ASSERT_EQ(out.size(), static_cast<std::size_t>(1));
    ASSERT_TRUE(out[0] == &ea.body);

    ea.collectMailAncestors(&ea.body, out);
    ASSERT_TRUE(out.empty());
}

TEST(incubator_mail_routing, incubator_anchor_lb_is_the_root_anchor_child) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    registerChain(ea);
    ea.parameters.incubator_mode = true;
    ASSERT_TRUE(ea.incubatorAnchorLb() == anchorOf(ea));
}

TEST(incubator_mail_routing, root_log_reaches_only_the_anchor_lb_in_incubator_mode) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    registerChain(ea);
    ea.parameters.incubator_mode = true;
    registerMailEdges(ea);
    gl::Memory* anchorLB = anchorOf(ea);
    gl::Memory* innerLB = innerOf(ea);

    // A prior-batch theorem batch is stored in the root's log.
    ea.broadcastTheorems({ std::string(kConj) }, "broadcast", ea.body, nullptr);
    ASSERT_TRUE(ea.body.hasWork);

    // The root self-injects (it never pulls its own log).
    const gl::Mail rootIn = gl::routingMailInToHeap(ea.body.mailIn);
    ASSERT_EQ(rootIn.statements.size(), static_cast<std::size_t>(1));

    // The anchor LB pulls the batch; the LB below it does not.
    gl::Mail anchorPulled;
    ea.mailLog.pull(anchorLB, anchorPulled);
    ASSERT_EQ(anchorPulled.statements.size(), static_cast<std::size_t>(1));

    gl::Mail innerPulled;
    ea.mailLog.pull(innerLB, innerPulled);
    ASSERT_TRUE(innerPulled.statements.empty());
}

TEST(incubator_mail_routing, anchor_log_reaches_the_anchor_lb_and_every_lb_below) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    registerChain(ea);
    ea.parameters.incubator_mode = true;
    registerMailEdges(ea);
    gl::Memory* anchorLB = anchorOf(ea);
    gl::Memory* innerLB = innerOf(ea);

    // The externals batch of an incubator batch is stored in the anchor LB's
    // log (the externals must reach every LB below the anchor).
    ea.broadcastTheorems({ std::string(kThm2) }, "externally provided theorem",
                         *anchorLB, nullptr);
    ASSERT_TRUE(anchorLB->hasWork);

    // The anchor LB is self-injected ...
    const gl::Mail anchorIn = gl::routingMailInToHeap(anchorLB->mailIn);
    ASSERT_EQ(anchorIn.statements.size(), static_cast<std::size_t>(1));

    // ... the LB below it pulls the batch from the anchor's log ...
    gl::Mail innerPulled;
    ea.mailLog.pull(innerLB, innerPulled);
    ASSERT_EQ(innerPulled.statements.size(), static_cast<std::size_t>(1));

    // ... and the root's log carries nothing for the anchor LB to pull.
    gl::Mail anchorPulled;
    ea.mailLog.pull(anchorLB, anchorPulled);
    ASSERT_TRUE(anchorPulled.statements.empty());
}

TEST(incubator_mail_routing, root_log_reaches_every_lb_outside_incubator_mode) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    registerChain(ea);
    registerMailEdges(ea);
    gl::Memory* anchorLB = anchorOf(ea);
    gl::Memory* innerLB = innerOf(ea);

    ea.broadcastTheorems({ std::string(kConj) }, "broadcast", ea.body, nullptr);

    gl::Mail anchorPulled;
    ea.mailLog.pull(anchorLB, anchorPulled);
    ASSERT_EQ(anchorPulled.statements.size(), static_cast<std::size_t>(1));

    gl::Mail innerPulled;
    ea.mailLog.pull(innerLB, innerPulled);
    ASSERT_EQ(innerPulled.statements.size(), static_cast<std::size_t>(1));
}

TEST(incubator_mail_routing, relays_absorbed_history_only_for_the_incubator_anchor) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    registerChain(ea);
    ASSERT_FALSE(ea.relaysAbsorbedHistory(*anchorOf(ea)));
    ea.parameters.incubator_mode = true;
    ASSERT_TRUE(ea.relaysAbsorbedHistory(*anchorOf(ea)));
    ASSERT_FALSE(ea.relaysAbsorbedHistory(*innerOf(ea)));
    ASSERT_FALSE(ea.relaysAbsorbedHistory(ea.body));
}

TEST(incubator_mail_routing, anchor_absorb_relays_root_history_rows_not_statements) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    registerChain(ea);
    ea.parameters.incubator_mode = true;
    ea.parameters.trackHistory = true;
    registerMailEdges(ea);
    gl::Memory* anchorLB = anchorOf(ea);

    // A root batch: one compact statement + its `compilation` origin row.
    ea.broadcastTheorems({ std::string(kConj) }, "broadcast", ea.body, nullptr);
    ea.mailLog.pull(anchorLB, anchorLB->mailIn);
    const gl::Mail pulled = gl::routingMailInToHeap(anchorLB->mailIn);
    ASSERT_EQ(pulled.statements.size(), static_cast<std::size_t>(1));
    ASSERT_FALSE(pulled.exprOriginMap.empty());
    const gl::ExpressionWithValidity compact = pulled.statements.begin()->first;

    // The anchor LB's pre-burst absorb of its inbox.
    ea.standardProcessing(*anchorLB, &anchorLB->mailIn,
                          anchorLB->nextIterationInternalMail,
                          anchorLB->sameIterationInternalMail, 0);

    // Its mailOut now carries the compact's history rows ...
    const gl::Mail out = gl::routingMailOutToHeap(anchorLB->mailOut,
                                                  anchorLB->mailOutInterner);
    ASSERT_TRUE(out.exprOriginMap.count(compact) == 1);
    bool hasCompilation = false;
    for (const auto& line : out.exprOriginMap.at(compact))
        if (line.first == "compilation") hasCompilation = true;
    ASSERT_TRUE(hasCompilation);
    // ... and not the compact statement itself (history only).
    ASSERT_TRUE(out.statements.count(std::make_pair(compact, std::set<int>())) == 0);
    for (const auto& st : out.statements)
        ASSERT_FALSE(st.first == compact);
}

TEST(incubator_mail_routing, anchor_only_rule_by_shape) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    ASSERT_TRUE(ea.anchorOnlyRule(kSingle));
    ASSERT_TRUE(ea.anchorOnlyRule(kEvalFact));
    ASSERT_TRUE(ea.anchorOnlyRule(kEvalFactMirror));
    ASSERT_FALSE(ea.anchorOnlyRule(kNestedHead));
    ASSERT_FALSE(ea.anchorOnlyRule(kNonDigit));
    ASSERT_FALSE(ea.anchorOnlyRule(kThreePremises));
    ASSERT_FALSE(ea.anchorOnlyRule(kConj));
    ASSERT_TRUE(ea.anchorOnlyRule(kForeignAnchorFact));
    ASSERT_FALSE(ea.anchorOnlyRule(kForeignAnchorFree));
}

TEST(incubator_mail_routing, load_routing_splits_nested_theorems_by_shape) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    registerChain(ea);
    ea.parameters.incubator_mode = true;
    registerMailEdges(ea);
    gl::Memory* anchorLB = anchorOf(ea);
    gl::Memory* innerLB = innerOf(ea);

    // One anchor-only theorem and one nested theorem in one batch: the first
    // is stored in the root's log, the second in the anchor LB's log.
    ea.broadcastTheorems({ std::string(kSingle), std::string(kNestedHead) },
                         "broadcast", ea.body, anchorLB);

    gl::Mail anchorPulled;
    ea.mailLog.pull(anchorLB, anchorPulled);
    ASSERT_EQ(anchorPulled.statements.size(), static_cast<std::size_t>(1));

    const gl::Mail anchorIn = gl::routingMailInToHeap(anchorLB->mailIn);
    ASSERT_EQ(anchorIn.statements.size(), static_cast<std::size_t>(1));

    gl::Mail innerPulled;
    ea.mailLog.pull(innerLB, innerPulled);
    ASSERT_EQ(innerPulled.statements.size(), static_cast<std::size_t>(1));
    ASSERT_FALSE(innerPulled.statements.begin()->first
                 == anchorPulled.statements.begin()->first);
}
