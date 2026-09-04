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
/// @brief Tests for the per-rule `disintegrationAllowed` flag
///        (D-241): the
///        `isOrIntroInstall` shape detector, and the install-time
///        derivation in `addToHashMemory` (integration justification and
///        or-intro shape derive false; ordinary implications derive true).
///        Plus the input-slot demand route
///        (D-288): the
///        `inputSlotDemandSlotQualifies` predicate matrix and the fourth
///        qualification pass's untagged marker install.

#include "test_harness.hpp"

#include "../prover.hpp"

namespace {

    /// @brief Deposit the ES3 description at @p ea's root LB, installing
    ///        the membership rules plus the per-leaf or-intro rules.
    ///
    /// @param ea The analyzer (IncubatorGauss3 config — the ES3 emitter).
    void depositEs3(gl::ExpressionAnalyzer& ea) {
        const std::string es3 = "(EnumerationSet3[2,6,7,10])";
        const int lv[1] = { 0 };
        const gl::TransientOrigin origin{
            true, gl::OriginTag::taskFormulation, nullptr, 0 };
        ea.addExprToMemoryBlock(gl::StrSpan(es3), ea.body, 0, 0, lv, 1,
            origin, -1, -1, gl::StrSpan("main", 4), false);
    }

    /// @brief Find the installed or-intro chain: the or premise (every intro
    ///        rule's head) and one equality leaf (an intro rule's premise).
    ///
    /// @details
    /// Mirrors `test_or_intro_rules`: scans the root LB's decoded rule
    /// chains for the membership main rule `[(or…), (in[1,u_10])]` and one
    /// intro chain `[(=[u_X,1]), (or…)]`.
    ///
    /// @param ea         The analyzer holding the installed rules.
    /// @param orPremise  Out: the or head string.
    /// @param introLeaf  Out: one intro rule's leaf premise string.
    void findIntroShape(gl::ExpressionAnalyzer& ea,
                        std::string& orPremise, std::string& introLeaf) {
        const int32_t n = ea.body.overallHashMemory.originals.count();
        for (int32_t oi = 1; oi <= n; ++oi) {
            const auto c = gl::decodeValueVector(
                ea.body.overallHashMemory.originals.decodeKey(oi).ids,
                ea.body.ruleInterner);
            if (c.size() == 2 && c[1] == "(in[1,u_10])"
                && c[0].rfind("(or", 0) == 0) {
                orPremise = c[0];
            }
            if (c.size() == 2 && c[0].rfind("(=[u_", 0) == 0
                && c[1].rfind("(or", 0) == 0) {
                introLeaf = c[0];
            }
        }
        ASSERT_TRUE(!orPremise.empty());
        ASSERT_TRUE(!introLeaf.empty());
    }

} // namespace

TEST(disintegration_allowed, or_intro_shape_detected) {
    gl::ExpressionAnalyzer ea(std::string("IncubatorGauss3"));
    depositEs3(ea);
    std::string orPremise, introLeaf;
    findIntroShape(ea, orPremise, introLeaf);

    // Positive: one premise, the premise a flattened leaf of the or head.
    const gl::StrSpan leafRun[1] = { gl::StrSpan(introLeaf) };
    ASSERT_TRUE(ea.isOrIntroInstall(leafRun, 1, gl::StrSpan(orPremise)));

    // Negative: the elimination direction — premise not a leaf.
    const std::string elim = "(in[1,u_10])";
    const gl::StrSpan elimRun[1] = { gl::StrSpan(elim) };
    ASSERT_FALSE(ea.isOrIntroInstall(elimRun, 1, gl::StrSpan(orPremise)));

    // Negative: two premises can never be the intro shape.
    const gl::StrSpan pairRun[2] = { gl::StrSpan(introLeaf), gl::StrSpan(elim) };
    ASSERT_FALSE(ea.isOrIntroInstall(pairRun, 2, gl::StrSpan(orPremise)));

    // Negative: a non-or head.
    ASSERT_FALSE(ea.isOrIntroInstall(leafRun, 1, gl::StrSpan(elim)));
}

TEST(disintegration_allowed, intro_installs_false_elimination_true) {
    gl::ExpressionAnalyzer ea(std::string("IncubatorGauss3"));
    depositEs3(ea);

    // Every or-intro install (original `(>[1](=[u_…)(or…))`) derives
    // disintegrationAllowed false; the elimination-direction install
    // (original `(>[1](in[1,u_10])(or…))`) derives true.
    int introSeen = 0, elimSeen = 0;
    auto& m = ea.body.overallHashMemory.encodedMap;
    for (int32_t id = 1; id <= m.count(); ++id) {
        for (int32_t j = 0; j < m.runLen(id); ++j) {
            const gl::LocalMemoryValue rec = m.recordAt(id, j);
            if (rec.isMarker) continue;
            const std::string orig = gl::decodeValueVector(
                std::vector<int32_t>{ rec.originalImplicationId },
                ea.body.ruleInterner)[0];
            if (orig.rfind("(>[1](=[u_", 0) == 0
                && orig.find("(or") != std::string::npos) {
                ASSERT_FALSE(rec.disintegrationAllowed);
                ++introSeen;
            }
            if (orig.rfind("(>[1](in[1,u_10])(or", 0) == 0) {
                ASSERT_TRUE(rec.disintegrationAllowed);
                ++elimSeen;
            }
        }
    }
    ASSERT_TRUE(introSeen >= 3);
    ASSERT_TRUE(elimSeen >= 1);
}

TEST(disintegration_allowed, integration_justification_derives_false) {
    gl::ExpressionAnalyzer ea(std::string("IncubatorGauss3"));

    const std::string chain0 = "(in3[1,2,3,u_4])";
    const std::string implInt = "(>[1,2,3](in3[1,2,3,u_4])(=[1,2]))";
    const std::string implImp = "(>[1,2,3](in3[1,2,3,u_4])(=[1,3]))";
    const std::string headInt = "(=[1,2])";
    const std::string headImp = "(=[1,3])";
    const int lv[1] = { 0 };

    const gl::StrSpan chainRun[1] = { gl::StrSpan(chain0) };
    ea.addToHashMemory(chainRun, 1, gl::StrSpan(headInt), nullptr, 0,
        ea.body, ea.body.overallHashMemory, lv, 1, gl::StrSpan(implInt),
        ea.parameters.maxIterationNumberVariable,
        ea.parameters.standardMaxSecondaryNumber, false,
        ea.parameters.minNumOperatorsKey, gl::StrSpan("integration", 11),
        false, gl::StrSpan(implInt));
    ea.addToHashMemory(chainRun, 1, gl::StrSpan(headImp), nullptr, 0,
        ea.body, ea.body.overallHashMemory, lv, 1, gl::StrSpan(implImp),
        ea.parameters.maxIterationNumberVariable,
        ea.parameters.standardMaxSecondaryNumber, false,
        ea.parameters.minNumOperatorsKey, gl::StrSpan("implication", 11),
        false, gl::StrSpan(implImp));

    int integrationSeen = 0, implicationSeen = 0;
    auto& m = ea.body.overallHashMemory.encodedMap;
    for (int32_t id = 1; id <= m.count(); ++id) {
        for (int32_t j = 0; j < m.runLen(id); ++j) {
            const gl::LocalMemoryValue rec = m.recordAt(id, j);
            if (rec.justification == gl::RuleJustification::integration) {
                ASSERT_FALSE(rec.disintegrationAllowed);
                ++integrationSeen;
            }
            if (rec.justification == gl::RuleJustification::implication) {
                ASSERT_TRUE(rec.disintegrationAllowed);
                ++implicationSeen;
            }
        }
    }
    ASSERT_TRUE(integrationSeen >= 1);
    ASSERT_TRUE(implicationSeen >= 1);
}

// isSubsetExclusionInstall (I-184): a
// one-premise rule whose negated premise, inserted into the reduced
// or-head's leaf list, matches a registered parent or is tagged; or-intro
// shapes, multi-premise rules, and non-or heads are not. Synthetic
// operators guarantee a clean registry corner.
TEST(disintegration_allowed, single_exclusion_shape_detected) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    const std::vector<std::string> parent = {
        "(zzsea[u_1])", "!(zzseb[u_1])", "(zzsec[u_1])" };
    ea.findOrMintOrOperator(parent, 1);
    ea.preMintReducedOrs();

    // Reduced heads for excluding leaf 0 and (negated) leaf 1.
    const std::string r0a = "!(zzseb[u_1])";
    const std::string r0b = "(zzsec[u_1])";
    gl::StrSpan r0[2] = { gl::StrSpan(r0a), gl::StrSpan(r0b) };
    const std::string* red0 = ea.compiledOrByElements(r0, 2);
    ASSERT_TRUE(red0 != nullptr);
    const std::string r1a = "(zzsea[u_1])";
    const std::string r1b = "(zzsec[u_1])";
    gl::StrSpan r1[2] = { gl::StrSpan(r1a), gl::StrSpan(r1b) };
    const std::string* red1 = ea.compiledOrByElements(r1, 2);
    ASSERT_TRUE(red1 != nullptr);

    // Positive: premise = negation of the excluded POSITIVE leaf.
    const std::string prem0 = "!(zzsea[u_1])";
    const std::string head0 = "(" + *red0 + "[u_1])";
    const gl::StrSpan run0[1] = { gl::StrSpan(prem0) };
    ASSERT_TRUE(ea.isSubsetExclusionInstall(run0, 1, gl::StrSpan(head0)));

    // Positive: excluding the NEGATED leaf — the premise is POSITIVE
    // (double-negation cancellation, I-175).
    const std::string prem1 = "(zzseb[u_1])";
    const std::string head1 = "(" + *red1 + "[u_1])";
    const gl::StrSpan run1[1] = { gl::StrSpan(prem1) };
    ASSERT_TRUE(ea.isSubsetExclusionInstall(run1, 1, gl::StrSpan(head1)));

    // Negative: or-INTRO shape (premise is a leaf of the head itself, not
    // an excluded disjunct) — no registered parent contains its negation.
    const gl::StrSpan introRun[1] = { gl::StrSpan(r1a) };
    ASSERT_FALSE(ea.isSubsetExclusionInstall(introRun, 1, gl::StrSpan(head1)));

    // Negative: two premises; negative: non-or head.
    const gl::StrSpan pairRun[2] = { gl::StrSpan(prem0), gl::StrSpan(prem1) };
    ASSERT_FALSE(ea.isSubsetExclusionInstall(pairRun, 2, gl::StrSpan(head0)));
    ASSERT_FALSE(ea.isSubsetExclusionInstall(run0, 1, gl::StrSpan(prem1)));
}

// The instance-argument mapping: a rule installed with concrete instance
// args (the head's signature-to-instance substitution permuting slots)
// still canonicalizes back onto its registered parent.
TEST(disintegration_allowed, single_exclusion_instance_args_canonicalize) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    const std::vector<std::string> parent = {
        "(zzsed[u_1,u_2])", "(zzsee[u_2,u_1])", "(zzsef[u_1,u_2])" };
    ea.findOrMintOrOperator(parent, 2);
    ea.preMintReducedOrs();

    // Reduced-minus-leaf-0 renumbers first-appearance: u_2 -> u_1,
    // u_1 -> u_2.
    const std::string ra = "(zzsee[u_1,u_2])";
    const std::string rb = "(zzsef[u_2,u_1])";
    gl::StrSpan rr[2] = { gl::StrSpan(ra), gl::StrSpan(rb) };
    const std::string* red = ea.compiledOrByElements(rr, 2);
    ASSERT_TRUE(red != nullptr);

    // Parent instance (u_1, u_2) = (7, 9): excluded leaf (zzsed[7,9]),
    // reduced head instance (9, 7).
    const std::string prem = "!(zzsed[7,9])";
    const std::string head = "(" + *red + "[9,7])";
    const gl::StrSpan run[1] = { gl::StrSpan(prem) };
    ASSERT_TRUE(ea.isSubsetExclusionInstall(run, 1, gl::StrSpan(head)));

    // A mismatched instance pairing (args swapped on one side only) has no
    // parent to canonicalize onto.
    const std::string wrong = "!(zzsed[9,7])";
    const gl::StrSpan wrongRun[1] = { gl::StrSpan(wrong) };
    ASSERT_FALSE(ea.isSubsetExclusionInstall(wrongRun, 1, gl::StrSpan(head)));
}

// The j >= 2 generalization (order-free): premises may arrive in any order
// (the mail-compact round-trip stores the first-seen body and dedups
// name-sorted permutation families, I-52), the complement must match as a
// multiset under a consistent binding, and binding injectivity keeps
// degenerate repeated-arg instances untagged.
TEST(disintegration_allowed, subset_exclusion_two_premises_order_free) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    const std::vector<std::string> parent = {
        "(zzsga[u_1])", "(zzsgb[u_1,u_2])", "!(zzsgc[u_2])", "(zzsgd[u_1])" };
    ea.findOrMintOrOperator(parent, 2);
    ea.preMintReducedOrs();

    // The 2-ary reduction {B, D} (leaves 0 and 2 excluded); identity
    // renumbering, so the raw element spans probe it directly.
    const std::string ra = "(zzsgb[u_1,u_2])";
    const std::string rb = "(zzsgd[u_1])";
    gl::StrSpan rr[2] = { gl::StrSpan(ra), gl::StrSpan(rb) };
    const std::string* red = ea.compiledOrByElements(rr, 2);
    ASSERT_TRUE(red != nullptr);

    // Parent instance (u_1, u_2) = (7, 9): head (red[7,9]); the excluded
    // POSITIVE leaf 0 gives premise !(zzsga[7]); the excluded NEGATED
    // leaf 2 gives the positive-core premise (zzsgc[9]) (I-175).
    const std::string head = "(" + *red + "[7,9])";
    const std::string pA = "!(zzsga[7])";
    const std::string pC = "(zzsgc[9])";

    const gl::StrSpan inOrder[2] = { gl::StrSpan(pA), gl::StrSpan(pC) };
    ASSERT_TRUE(ea.isSubsetExclusionInstall(inOrder, 2, gl::StrSpan(head)));

    const gl::StrSpan permuted[2] = { gl::StrSpan(pC), gl::StrSpan(pA) };
    ASSERT_TRUE(ea.isSubsetExclusionInstall(permuted, 2, gl::StrSpan(head)));

    // Complement mismatch: a premise instantiated off the parent's binding
    // (u_1 is pinned to 7 by the head's leaves) does not embed.
    const std::string pWrong = "!(zzsga[9])";
    const gl::StrSpan wrong[2] = { gl::StrSpan(pWrong), gl::StrSpan(pC) };
    ASSERT_FALSE(ea.isSubsetExclusionInstall(wrong, 2, gl::StrSpan(head)));

    // Degenerate repeated-arg instance: injectivity refuses u_1 and u_2
    // binding the same arg — the rule stays untagged.
    const std::string headDeg = "(" + *red + "[7,7])";
    const std::string pCDeg = "(zzsgc[7])";
    const gl::StrSpan degRun[2] = { gl::StrSpan(pA), gl::StrSpan(pCDeg) };
    ASSERT_FALSE(ea.isSubsetExclusionInstall(degRun, 2, gl::StrSpan(headDeg)));

    // Premise count overshooting every registered parent (2 + 3 = 5
    // leaves; this fixture's parent has 4) is a defined miss.
    const std::string pD = "!(zzsgd[7])";
    const gl::StrSpan three[3] = {
        gl::StrSpan(pA), gl::StrSpan(pC), gl::StrSpan(pD) };
    ASSERT_FALSE(ea.isSubsetExclusionInstall(three, 3, gl::StrSpan(head)));
}

namespace {

    /// @brief Span run over caller-stable string storage (a StrSpan over a
    ///        temporary would dangle — 09c pitfall 1).
    struct KeyRun {
        std::vector<std::string> storage;
        std::vector<gl::StrSpan> spans;
        explicit KeyRun(std::vector<std::string> s) : storage(std::move(s)) {
            for (const std::string& e : storage) spans.push_back(gl::StrSpan(e));
        }
        const gl::StrSpan* data() const { return spans.data(); }
        int32_t count() const { return static_cast<int32_t>(spans.size()); }
    };

    /// @brief Count UNTAGGED marker LMVs whose decoded value is an `in3`
    ///        invocation with the literal `marker` at argument position
    ///        @p argPos — the input-slot (1) vs output-slot (2)
    ///        discriminator for the demand-pass tests.
    ///
    /// @param ea     The analyzer whose root-LB encodedMap is scanned.
    /// @param argPos The argument position the marker must occupy.
    /// @return The matching record count.
    int32_t countIn3MarkerAt(gl::ExpressionAnalyzer& ea, int32_t argPos) {
        int32_t hits = 0;
        auto& m = ea.body.overallHashMemory.encodedMap;
        for (int32_t id = 1; id <= m.count(); ++id) {
            for (int32_t j = 0; j < m.runLen(id); ++j) {
                const gl::LocalMemoryValue rec = m.recordAt(id, j);
                if (!rec.isMarker || rec.ordisOnly) continue;
                const std::string v = gl::decodeValueVector(
                    std::vector<int32_t>{ rec.valueId },
                    ea.body.ruleInterner)[0];
                if (v.rfind("(in3[", 0) != 0 || v.back() != ')') continue;
                const std::string inner =
                    v.substr(5, v.size() - 7);   // strip "(in3[" and "])"
                std::vector<std::string> args;
                std::size_t pos = 0;
                while (true) {
                    const std::size_t c = inner.find(',', pos);
                    if (c == std::string::npos) {
                        args.push_back(inner.substr(pos));
                        break;
                    }
                    args.push_back(inner.substr(pos, c - pos));
                    pos = c + 1;
                }
                if (argPos < static_cast<int32_t>(args.size())
                    && args[argPos] == "marker") {
                    ++hits;
                }
            }
        }
        return hits;
    }

} // namespace

// The input-slot demand predicate on the C8 difference-transport shape
// (D-288): the witness slot qualifies; bound,
// anchor, out-of-range, head-starved, and premise-poor slots refuse.
TEST(disintegration_allowed, input_slot_demand_slot_qualification_matrix) {
    gl::ExpressionAnalyzer ea(std::string("FTA"));

    // Row-56 shape: d·x=a, a+b=c, d·y=c, x+z=y ⟹ d·z=b with plain names —
    // 9=d, 10=x, 11=a, 12=b, 13=c, 14=y, 15=z (witness).
    const KeyRun key56({
        "(AnchorFTA[1,2,3,4,5,6,7,8])",
        "(in3[9,10,11,5])",
        "(in3[11,12,13,4])",
        "(in3[9,14,13,5])",
        "(in3[10,15,14,4])" });
    const std::string head56 = "(in3[9,15,12,5])";

    // The witness slot: premise 4 (x+z=y), input position 1 (z) — the one
    // qualifying pair.
    ASSERT_TRUE(ea.inputSlotDemandSlotQualifies(
        key56.data(), key56.count(), 4, 1, gl::StrSpan(head56)));

    // Input position 0 of the same premise (x): the head takes only d and
    // z at its input slots, so condition (1) refuses.
    ASSERT_FALSE(ea.inputSlotDemandSlotQualifies(
        key56.data(), key56.count(), 4, 0, gl::StrSpan(head56)));

    // The anchor premise refuses outright.
    ASSERT_FALSE(ea.inputSlotDemandSlotQualifies(
        key56.data(), key56.count(), 0, 0, gl::StrSpan(head56)));

    // Premise 1 input slot 0 (d, shared with premise 3): confinement
    // refuses.
    ASSERT_FALSE(ea.inputSlotDemandSlotQualifies(
        key56.data(), key56.count(), 1, 0, gl::StrSpan(head56)));

    // Out-of-range input position (in3 declares two inputs).
    ASSERT_FALSE(ea.inputSlotDemandSlotQualifies(
        key56.data(), key56.count(), 4, 2, gl::StrSpan(head56)));

    // A head that does not take the candidate at an INPUT slot (15 at the
    // output slot): condition (1) refuses.
    const std::string headOut = "(in3[9,12,15,5])";
    ASSERT_FALSE(ea.inputSlotDemandSlotQualifies(
        key56.data(), key56.count(), 4, 1, gl::StrSpan(headOut)));

    // Premise-count gate: dropping a+b=c leaves the slot structurally
    // qualified (arg 4 is an anchor-slot name, 10/14 stay bound) but only
    // three non-anchor premises — kInputSlotDemandMinPremises refuses.
    const KeyRun key3({
        "(AnchorFTA[1,2,3,4,5,6,7,8])",
        "(in3[9,10,11,5])",
        "(in3[9,14,13,5])",
        "(in3[10,15,14,4])" });
    const std::string head3 = "(in3[9,15,12,5])";
    ASSERT_FALSE(ea.inputSlotDemandSlotQualifies(
        key3.data(), key3.count(), 3, 1, gl::StrSpan(head3)));
}

// The fourth qualification pass installs an UNTAGGED in3 marker with the
// marker at an INPUT slot for the row-56 shape, and installs none for a
// premise-poor introduction shape — D-288.
TEST(disintegration_allowed, input_slot_demand_pass_installs_untagged_marker) {
    gl::ExpressionAnalyzer ea(std::string("FTA"));

    const std::vector<std::string> key56 = {
        "(AnchorFTA[1,2,3,4,5,6,7,8])",
        "(in3[9,10,11,5])",
        "(in3[11,12,13,4])",
        "(in3[9,14,13,5])",
        "(in3[10,15,14,4])" };
    const std::string head56 = "(in3[9,15,12,5])";
    const std::string impl56 =
        "(>[9,10,11,12,13,14,15](in3[9,10,11,5])(in3[11,12,13,4])"
        "(in3[9,14,13,5])(in3[10,15,14,4])(in3[9,15,12,5]))";

    const int32_t inputBefore = countIn3MarkerAt(ea, 1);
    ea.makeNormalizedKeysForAdmission(key56, ea.body.overallHashMemory,
        ea.body.nameMap, ea.body.ruleInterner, head56,
        ea.parameters.minNumOperatorsKey, impl56, "main", &ea.body);
    ASSERT_TRUE(countIn3MarkerAt(ea, 1) > inputBefore);

    // Control: a single-premise introduction shape (row 55) mints no
    // input-slot marker — the premise-count gate holds.
    gl::ExpressionAnalyzer ea2(std::string("FTA"));
    const std::vector<std::string> key55 = {
        "(AnchorFTA[1,2,3,4,5,6,7,8])",
        "(in3[9,10,11,4])" };
    const std::string head55 = "(preorder[1,4,9,11])";
    const std::string impl55 =
        "(>[9,10,11](in3[9,10,11,4])(preorder[1,4,9,11]))";
    const int32_t ctrlBefore = countIn3MarkerAt(ea2, 1);
    ea2.makeNormalizedKeysForAdmission(key55, ea2.body.overallHashMemory,
        ea2.body.nameMap, ea2.body.ruleInterner, head55,
        ea2.parameters.minNumOperatorsKey, impl55, "main", &ea2.body);
    ASSERT_TRUE(countIn3MarkerAt(ea2, 1) == ctrlBefore);
}
