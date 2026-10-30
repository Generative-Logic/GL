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
///        qualification pass's untagged marker install. Plus the
///        theorem-relation demand route
///        (D-370): the
///        `theoremRelationDemandSlotQualifies` predicate matrix and the
///        fifth qualification pass's untagged marker install.

#include "../memory.hpp"
#include <cstddef>
#include <utility>
#include "test_harness.hpp"
#include "prover_fixture.hpp"

#include "../prover.hpp"

namespace {

    /// @brief Deposit the ES3 description at @p ea's root LB, installing
    ///        the membership rules plus the per-leaf or-intro rules.
    ///
    /// @param ea The analyzer (IncubatorGauss3 config — the ES3 emitter).
    void depositEs3(gl::tests::ProverFixture& ea) {
        const std::string es3 = "(EnumerationSet3[2,6,7,10])";
        const int lv[1] = { 0 };
        const gl::TransientOrigin origin{
            true, gl::OriginTag::taskFormulation, nullptr, 0 };
        ea.statementMemory.addExprToMemoryBlock(gl::StrSpan(es3), ea.body, 0, 0, lv, 1,
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
    void findIntroShape(gl::tests::ProverFixture& ea,
                        std::string& orPremise, std::string& introLeaf) {
        const int32_t n = ((((ea.body).overallHashMemory)).originals).count();
        for (int32_t oi = 1; oi <= n; ++oi) {
            const auto c = gl::decodeValueVector(
                ((((ea.body).overallHashMemory)).originals).decodeKey(oi).ids,
                ((ea.body).ruleInterner));
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
    gl::tests::ProverFixture ea(std::string("IncubatorGauss3"));
    gl::HashRuleClassifier classifier(ea.definitions);
    depositEs3(ea);
    std::string orPremise, introLeaf;
    findIntroShape(ea, orPremise, introLeaf);

    // Positive: one premise, the premise a flattened leaf of the or head.
    const gl::StrSpan leafRun[1] = { gl::StrSpan(introLeaf) };
    ASSERT_TRUE(classifier.isOrIntroInstall(leafRun, 1, gl::StrSpan(orPremise)));

    // Negative: the elimination direction — premise not a leaf.
    const std::string elim = "(in[1,u_10])";
    const gl::StrSpan elimRun[1] = { gl::StrSpan(elim) };
    ASSERT_FALSE(classifier.isOrIntroInstall(elimRun, 1, gl::StrSpan(orPremise)));

    // Negative: two premises can never be the intro shape.
    const gl::StrSpan pairRun[2] = { gl::StrSpan(introLeaf), gl::StrSpan(elim) };
    ASSERT_FALSE(classifier.isOrIntroInstall(pairRun, 2, gl::StrSpan(orPremise)));

    // Negative: a non-or head.
    ASSERT_FALSE(classifier.isOrIntroInstall(leafRun, 1, gl::StrSpan(elim)));
}

TEST(disintegration_allowed, intro_installs_false_elimination_true) {
    gl::tests::ProverFixture ea(std::string("IncubatorGauss3"));
    depositEs3(ea);

    // Every or-intro install (original `(>[1](=[u_…)(or…))`) derives
    // disintegrationAllowed false; the elimination-direction install
    // (original `(>[1](in[1,u_10])(or…))`) derives true.
    int introSeen = 0, elimSeen = 0;
    auto& m = ((((ea.body).overallHashMemory)).encodedMap);
    for (int32_t id = 1; id <= m.count(); ++id) {
        for (int32_t j = 0; j < m.runLen(id); ++j) {
            const gl::LocalMemoryValue rec = m.recordAt(id, j);
            if (rec.isMarker) continue;
            const std::string orig = gl::decodeValueVector(
                std::vector<int32_t>{ rec.originalImplicationId },
                ((ea.body).ruleInterner))[0];
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
    gl::tests::ProverFixture ea(std::string("IncubatorGauss3"));

    const std::string chain0 = "(in3[1,2,3,u_4])";
    const std::string implInt = "(>[1,2,3](in3[1,2,3,u_4])(=[1,2]))";
    const std::string implImp = "(>[1,2,3](in3[1,2,3,u_4])(=[1,3]))";
    const std::string headInt = "(=[1,2])";
    const std::string headImp = "(=[1,3])";
    const int lv[1] = { 0 };

    const gl::StrSpan chainRun[1] = { gl::StrSpan(chain0) };
    ea.hashRuleInstaller.addToHashMemory(chainRun, 1, gl::StrSpan(headInt), nullptr, 0,
        ea.body, ((ea.body).overallHashMemory), lv, 1, gl::StrSpan(implInt),
        ea.parameters.maxIterationNumberVariable,
        ea.parameters.standardMaxSecondaryNumber, false,
        ea.parameters.minNumOperatorsKey, gl::StrSpan("integration", 11),
        false, gl::StrSpan(implInt));
    ea.hashRuleInstaller.addToHashMemory(chainRun, 1, gl::StrSpan(headImp), nullptr, 0,
        ea.body, ((ea.body).overallHashMemory), lv, 1, gl::StrSpan(implImp),
        ea.parameters.maxIterationNumberVariable,
        ea.parameters.standardMaxSecondaryNumber, false,
        ea.parameters.minNumOperatorsKey, gl::StrSpan("implication", 11),
        false, gl::StrSpan(implImp));

    int integrationSeen = 0, implicationSeen = 0;
    auto& m = ((((ea.body).overallHashMemory)).encodedMap);
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
    gl::tests::ProverFixture ea(std::string("Peano"));
    const std::vector<std::string> parent = {
        "(zzsea[u_1])", "!(zzseb[u_1])", "(zzsec[u_1])" };
    ea.orRuleCompiler.findOrMintOrOperator(parent, 1);
    ea.orRuleCompiler.preMintReducedOrs();

    // Reduced heads for excluding leaf 0 and (negated) leaf 1.
    const std::string r0a = "!(zzseb[u_1])";
    const std::string r0b = "(zzsec[u_1])";
    gl::StrSpan r0[2] = { gl::StrSpan(r0a), gl::StrSpan(r0b) };
    const std::string* red0 = ea.compilationReuse.compiledOrByElements(r0, 2);
    ASSERT_TRUE(red0 != nullptr);
    const std::string r1a = "(zzsea[u_1])";
    const std::string r1b = "(zzsec[u_1])";
    gl::StrSpan r1[2] = { gl::StrSpan(r1a), gl::StrSpan(r1b) };
    const std::string* red1 = ea.compilationReuse.compiledOrByElements(r1, 2);
    ASSERT_TRUE(red1 != nullptr);

    // Positive: premise = negation of the excluded POSITIVE leaf.
    const std::string prem0 = "!(zzsea[u_1])";
    const std::string head0 = "(" + *red0 + "[u_1])";
    const gl::StrSpan run0[1] = { gl::StrSpan(prem0) };
    ASSERT_TRUE(ea.hashRuleClassifier.isSubsetExclusionInstall(run0, 1, gl::StrSpan(head0)));

    // Positive: excluding the NEGATED leaf — the premise is POSITIVE
    // (double-negation cancellation, I-175).
    const std::string prem1 = "(zzseb[u_1])";
    const std::string head1 = "(" + *red1 + "[u_1])";
    const gl::StrSpan run1[1] = { gl::StrSpan(prem1) };
    ASSERT_TRUE(ea.hashRuleClassifier.isSubsetExclusionInstall(run1, 1, gl::StrSpan(head1)));

    // Negative: or-INTRO shape (premise is a leaf of the head itself, not
    // an excluded disjunct) — no registered parent contains its negation.
    const gl::StrSpan introRun[1] = { gl::StrSpan(r1a) };
    ASSERT_FALSE(ea.hashRuleClassifier.isSubsetExclusionInstall(introRun, 1, gl::StrSpan(head1)));

    // Negative: two premises; negative: non-or head.
    const gl::StrSpan pairRun[2] = { gl::StrSpan(prem0), gl::StrSpan(prem1) };
    ASSERT_FALSE(ea.hashRuleClassifier.isSubsetExclusionInstall(pairRun, 2, gl::StrSpan(head0)));
    ASSERT_FALSE(ea.hashRuleClassifier.isSubsetExclusionInstall(run0, 1, gl::StrSpan(prem1)));
}

// The instance-argument mapping: a rule installed with concrete instance
// args (the head's signature-to-instance substitution permuting slots)
// still canonicalizes back onto its registered parent.
TEST(disintegration_allowed, single_exclusion_instance_args_canonicalize) {
    gl::tests::ProverFixture ea(std::string("Peano"));
    const std::vector<std::string> parent = {
        "(zzsed[u_1,u_2])", "(zzsee[u_2,u_1])", "(zzsef[u_1,u_2])" };
    ea.orRuleCompiler.findOrMintOrOperator(parent, 2);
    ea.orRuleCompiler.preMintReducedOrs();

    // Reduced-minus-leaf-0 renumbers first-appearance: u_2 -> u_1,
    // u_1 -> u_2.
    const std::string ra = "(zzsee[u_1,u_2])";
    const std::string rb = "(zzsef[u_2,u_1])";
    gl::StrSpan rr[2] = { gl::StrSpan(ra), gl::StrSpan(rb) };
    const std::string* red = ea.compilationReuse.compiledOrByElements(rr, 2);
    ASSERT_TRUE(red != nullptr);

    // Parent instance (u_1, u_2) = (7, 9): excluded leaf (zzsed[7,9]),
    // reduced head instance (9, 7).
    const std::string prem = "!(zzsed[7,9])";
    const std::string head = "(" + *red + "[9,7])";
    const gl::StrSpan run[1] = { gl::StrSpan(prem) };
    ASSERT_TRUE(ea.hashRuleClassifier.isSubsetExclusionInstall(run, 1, gl::StrSpan(head)));

    // A mismatched instance pairing (args swapped on one side only) has no
    // parent to canonicalize onto.
    const std::string wrong = "!(zzsed[9,7])";
    const gl::StrSpan wrongRun[1] = { gl::StrSpan(wrong) };
    ASSERT_FALSE(ea.hashRuleClassifier.isSubsetExclusionInstall(wrongRun, 1, gl::StrSpan(head)));
}

// The j >= 2 generalization (order-free): premises may arrive in any order
// (the mail-compact round-trip stores the first-seen body and dedups
// name-sorted permutation families, I-52), the complement must match as a
// multiset under a consistent binding, and binding injectivity keeps
// degenerate repeated-arg instances untagged.
TEST(disintegration_allowed, subset_exclusion_two_premises_order_free) {
    gl::tests::ProverFixture ea(std::string("Peano"));
    const std::vector<std::string> parent = {
        "(zzsga[u_1])", "(zzsgb[u_1,u_2])", "!(zzsgc[u_2])", "(zzsgd[u_1])" };
    ea.orRuleCompiler.findOrMintOrOperator(parent, 2);
    ea.orRuleCompiler.preMintReducedOrs();

    // The 2-ary reduction {B, D} (leaves 0 and 2 excluded); identity
    // renumbering, so the raw element spans probe it directly.
    const std::string ra = "(zzsgb[u_1,u_2])";
    const std::string rb = "(zzsgd[u_1])";
    gl::StrSpan rr[2] = { gl::StrSpan(ra), gl::StrSpan(rb) };
    const std::string* red = ea.compilationReuse.compiledOrByElements(rr, 2);
    ASSERT_TRUE(red != nullptr);

    // Parent instance (u_1, u_2) = (7, 9): head (red[7,9]); the excluded
    // POSITIVE leaf 0 gives premise !(zzsga[7]); the excluded NEGATED
    // leaf 2 gives the positive-core premise (zzsgc[9]) (I-175).
    const std::string head = "(" + *red + "[7,9])";
    const std::string pA = "!(zzsga[7])";
    const std::string pC = "(zzsgc[9])";

    const gl::StrSpan inOrder[2] = { gl::StrSpan(pA), gl::StrSpan(pC) };
    ASSERT_TRUE(ea.hashRuleClassifier.isSubsetExclusionInstall(inOrder, 2, gl::StrSpan(head)));

    const gl::StrSpan permuted[2] = { gl::StrSpan(pC), gl::StrSpan(pA) };
    ASSERT_TRUE(ea.hashRuleClassifier.isSubsetExclusionInstall(permuted, 2, gl::StrSpan(head)));

    // Complement mismatch: a premise instantiated off the parent's binding
    // (u_1 is pinned to 7 by the head's leaves) does not embed.
    const std::string pWrong = "!(zzsga[9])";
    const gl::StrSpan wrong[2] = { gl::StrSpan(pWrong), gl::StrSpan(pC) };
    ASSERT_FALSE(ea.hashRuleClassifier.isSubsetExclusionInstall(wrong, 2, gl::StrSpan(head)));

    // Degenerate repeated-arg instance: injectivity refuses u_1 and u_2
    // binding the same arg — the rule stays untagged.
    const std::string headDeg = "(" + *red + "[7,7])";
    const std::string pCDeg = "(zzsgc[7])";
    const gl::StrSpan degRun[2] = { gl::StrSpan(pA), gl::StrSpan(pCDeg) };
    ASSERT_FALSE(ea.hashRuleClassifier.isSubsetExclusionInstall(degRun, 2, gl::StrSpan(headDeg)));

    // Premise count overshooting every registered parent (2 + 3 = 5
    // leaves; this fixture's parent has 4) is a defined miss.
    const std::string pD = "!(zzsgd[7])";
    const gl::StrSpan three[3] = {
        gl::StrSpan(pA), gl::StrSpan(pC), gl::StrSpan(pD) };
    ASSERT_FALSE(ea.hashRuleClassifier.isSubsetExclusionInstall(three, 3, gl::StrSpan(head)));
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
    int32_t countIn3MarkerAt(gl::tests::ProverFixture& ea, int32_t argPos) {
        int32_t hits = 0;
        auto& m = ((((ea.body).overallHashMemory)).encodedMap);
        for (int32_t id = 1; id <= m.count(); ++id) {
            for (int32_t j = 0; j < m.runLen(id); ++j) {
                const gl::LocalMemoryValue rec = m.recordAt(id, j);
                if (!rec.isMarker || rec.ordisOnly) continue;
                const std::string v = gl::decodeValueVector(
                    std::vector<int32_t>{ rec.valueId },
                    ((ea.body).ruleInterner))[0];
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

    /// @brief Count UNTAGGED marker LMVs whose decoded value applies @p core
    ///        with the literal `marker` at argument position @p argPos.
    ///
    /// @param ea     The analyzer whose root-LB encodedMap is scanned.
    /// @param core   The operator name of the marked premise.
    /// @param argPos The argument position the marker must occupy.
    /// @return The matching record count.
    int32_t countMarkerOf(gl::tests::ProverFixture& ea, const std::string& core, int32_t argPos) {
        int32_t hits = 0;
        auto& m = ea.body.overallHashMemory.encodedMap;
        const std::string prefix = "(" + core + "[";
        for (int32_t id = 1; id <= m.count(); ++id) {
            for (int32_t j = 0; j < m.runLen(id); ++j) {
                const gl::LocalMemoryValue rec = m.recordAt(id, j);
                if (!rec.isMarker || rec.ordisOnly) continue;
                const std::string v = gl::decodeValueVector(
                    std::vector<int32_t>{ rec.valueId }, ea.body.ruleInterner)[0];
                if (v.rfind(prefix, 0) != 0) continue;
                const std::vector<std::string> args = ce::getArgs(v);
                if (argPos < static_cast<int32_t>(args.size()) && args[argPos] == "marker") ++hits;
            }
        }
        return hits;
    }

} // namespace

// The input-slot demand predicate on the C8 difference-transport shape
// (D-288): the witness slot qualifies; bound,
// anchor, out-of-range, head-starved, and premise-poor slots refuse.
TEST(disintegration_allowed, input_slot_demand_slot_qualification_matrix) {
    gl::tests::ProverFixture ea(std::string("FTA"));

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
    ASSERT_TRUE(ea.admissionPolicy.inputSlotDemandSlotQualifies(
        key56.data(), key56.count(), 4, 1, gl::StrSpan(head56)));

    // Input position 0 of the same premise (x): the head takes only d and
    // z at its input slots, so condition (1) refuses.
    ASSERT_FALSE(ea.admissionPolicy.inputSlotDemandSlotQualifies(
        key56.data(), key56.count(), 4, 0, gl::StrSpan(head56)));

    // The anchor premise refuses outright.
    ASSERT_FALSE(ea.admissionPolicy.inputSlotDemandSlotQualifies(
        key56.data(), key56.count(), 0, 0, gl::StrSpan(head56)));

    // Premise 1 input slot 0 (d, shared with premise 3): confinement
    // refuses.
    ASSERT_FALSE(ea.admissionPolicy.inputSlotDemandSlotQualifies(
        key56.data(), key56.count(), 1, 0, gl::StrSpan(head56)));

    // Out-of-range input position (in3 declares two inputs).
    ASSERT_FALSE(ea.admissionPolicy.inputSlotDemandSlotQualifies(
        key56.data(), key56.count(), 4, 2, gl::StrSpan(head56)));

    // A head that does not take the candidate at an INPUT slot (15 at the
    // output slot): condition (1) refuses.
    const std::string headOut = "(in3[9,12,15,5])";
    ASSERT_FALSE(ea.admissionPolicy.inputSlotDemandSlotQualifies(
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
    ASSERT_FALSE(ea.admissionPolicy.inputSlotDemandSlotQualifies(
        key3.data(), key3.count(), 3, 1, gl::StrSpan(head3)));
}

// The fourth qualification pass installs an UNTAGGED in3 marker with the
// marker at an INPUT slot for the row-56 shape, and installs none for a
// premise-poor introduction shape — D-288.
TEST(disintegration_allowed, input_slot_demand_pass_installs_untagged_marker) {
    gl::tests::ProverFixture ea(std::string("FTA"));

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
    ea.admissionMarkers.makeNormalizedKeysForAdmission(key56, ((ea.body).overallHashMemory),
        ((ea.body).nameMap), ((ea.body).ruleInterner), head56,
        ea.parameters.minNumOperatorsKey, impl56, "main", &ea.body);
    ASSERT_TRUE(countIn3MarkerAt(ea, 1) > inputBefore);

    // Control: a single-premise introduction shape (row 55) mints no
    // input-slot marker — the premise-count gate holds.
    gl::tests::ProverFixture ea2(std::string("FTA"));
    const std::vector<std::string> key55 = {
        "(AnchorFTA[1,2,3,4,5,6,7,8])",
        "(in3[9,10,11,4])" };
    const std::string head55 = "(preorder[1,4,9,11])";
    const std::string impl55 =
        "(>[9,10,11](in3[9,10,11,4])(preorder[1,4,9,11]))";
    const int32_t ctrlBefore = countIn3MarkerAt(ea2, 1);
    ea2.admissionMarkers.makeNormalizedKeysForAdmission(key55, ((ea2.body).overallHashMemory),
        ((ea2.body).nameMap), ((ea2.body).ruleInterner), head55,
        ea2.parameters.minNumOperatorsKey, impl55, "main", &ea2.body);
    ASSERT_TRUE(countIn3MarkerAt(ea2, 1) == ctrlBefore);
}

// Recursion block #1 records its installed induction implication's
// ruleInterner id (`Memory::recursionAuxyImplId`); the firing site marks that
// rule's fired head as the induction hypothesis
// (D-356). Every LB outside a recursion step
// block keeps 0.
TEST(disintegration_allowed, recursion_block_records_auxy_implication_id) {
    gl::tests::ProverFixture ea(std::string("Peano"));
    const std::string conj =
        "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8](in2[7,8,3])(=[7,8])))";
    ea.proofGridBuilder.addTheoremToMemory(conj, ea.body, 0, false, ea.globalDependencies);

    gl::Memory* anchorLB =
        ea.simpleMapStore.findChild(&ea.body, "(AnchorPeano[1,2,3,4,5,6])");
    ASSERT_TRUE(anchorLB != nullptr);
    gl::Memory* innerLB = ea.simpleMapStore.findChild(anchorLB, "(in2[7,8,3])");
    ASSERT_TRUE(innerLB != nullptr);
    // Not recursion step blocks: the id stays 0.
    ASSERT_EQ(((ea.body).recursionAuxyImplId), 0);
    ASSERT_EQ(((*(anchorLB)).recursionAuxyImplId), 0);
    ASSERT_EQ(((*(innerLB)).recursionAuxyImplId), 0);

    // The one digit arg (7 — the in2 INPUT slot; 8 is its output and is
    // removed by findDigitArgs) spawns recursion block #1 under the
    // innermost premise LB as rec0.
    gl::Memory* rec0 = ea.simpleMapStore.findChild(innerLB, "(in2[rec0,7,3])");
    ASSERT_TRUE(rec0 != nullptr);
    ASSERT_TRUE(ea.simpleMapStore.findChild(innerLB, "(in2[rec1,8,3])") == nullptr);
    ASSERT_TRUE(((*(rec0)).isPartOfRecursion));
    ASSERT_TRUE(((*(rec0)).recursionAuxyImplId) != 0);

    // The recorded id decodes to the block's own induction implication: an
    // implication text that cites the substituted induction variable.
    const std::string impl0 =
        ((*(rec0)).ruleInterner).decodeView(((*(rec0)).recursionAuxyImplId)).toStdString();
    ASSERT_TRUE(impl0.rfind("(>[", 0) == 0);
    ASSERT_TRUE(impl0.find("rec") != std::string::npos);
    // And it is exactly one of the installed rules' producer implications.
    bool found = false;
    auto& m = ((((*(rec0)).overallHashMemory)).encodedMap);
    for (int32_t id = 1; id <= m.count() && !found; ++id) {
        for (int32_t j = 0; j < m.runLen(id); ++j) {
            const gl::LocalMemoryValue rec = m.recordAt(id, j);
            if (rec.originalImplicationId == ((*(rec0)).recursionAuxyImplId)) { found = true; break; }
        }
    }
    ASSERT_TRUE(found);
}

// The theorem-relation demand predicate
// (D-370) on the D2 consumer shape
// `a<b ∧ a+k=b → k≠0` (pool row 20): the witness slot qualifies; the
// successor-witness shape (one bound argument), an operator other-premise,
// a negated other-premise, an atomic other-premise, a disintegration
// product (`u_` argument), an integration install, a bound candidate and an
// out-of-range slot refuse.
TEST(disintegration_allowed, theorem_relation_demand_slot_qualification_matrix) {
    gl::tests::ProverFixture ea(std::string("FTA"));

    // Row-20 shape with plain names — 9=a, 10=b, 11=k (witness).
    const KeyRun key20({
        "(AnchorFTA[1,2,3,4,5,6,7,8])",
        "(strictOrder[1,4,9,10])",
        "(in3[9,11,10,4])" });

    // The witness slot: premise 2 (a+k=b), input position 1 (k) — the head
    // (`!(=[2,11])`) is unconstrained on this route.
    ASSERT_TRUE(ea.admissionPolicy.theoremRelationDemandSlotQualifies(
        key20.data(), key20.count(), 2, 1, false));

    // Input position 0 of the same premise (a, bound by the relation):
    // confinement refuses.
    ASSERT_FALSE(ea.admissionPolicy.theoremRelationDemandSlotQualifies(
        key20.data(), key20.count(), 2, 0, false));

    // The anchor premise and the relation premise refuse outright.
    ASSERT_FALSE(ea.admissionPolicy.theoremRelationDemandSlotQualifies(
        key20.data(), key20.count(), 0, 0, false));
    ASSERT_FALSE(ea.admissionPolicy.theoremRelationDemandSlotQualifies(
        key20.data(), key20.count(), 1, 0, false));

    // Out-of-range input position (in3 declares two inputs).
    ASSERT_FALSE(ea.admissionPolicy.theoremRelationDemandSlotQualifies(
        key20.data(), key20.count(), 2, 2, false));

    // An integration install refuses.
    ASSERT_FALSE(ea.admissionPolicy.theoremRelationDemandSlotQualifies(
        key20.data(), key20.count(), 2, 1, true));

    // Successor-witness shape (row 13: s(p)=x ∧ x≤y → p≤y): the marked in2
    // premise has one non-marker argument bound by the relation —
    // kTheoremRelationDemandMinBoundArgs refuses.
    const KeyRun key13({
        "(AnchorFTA[1,2,3,4,5,6,7,8])",
        "(in2[9,10,3])",
        "(preorder[1,4,10,11])" });
    ASSERT_FALSE(ea.admissionPolicy.theoremRelationDemandSlotQualifies(
        key13.data(), key13.count(), 1, 0, false));

    // An operator among the other premises (row 39: a<b ∧ 1≤c ∧ a·c=x ∧
    // b·c=y → x<y, marked at a·c=x input slot 1): condition (3) refuses.
    const KeyRun key39({
        "(AnchorFTA[1,2,3,4,5,6,7,8])",
        "(strictOrder[1,4,9,10])",
        "(preorder[1,4,6,11])",
        "(in3[9,11,12,5])",
        "(in3[10,11,13,5])" });
    ASSERT_FALSE(ea.admissionPolicy.theoremRelationDemandSlotQualifies(
        key39.data(), key39.count(), 3, 1, false));

    // A negated other-premise refuses.
    const KeyRun keyNeg({
        "(AnchorFTA[1,2,3,4,5,6,7,8])",
        "!(strictOrder[1,4,9,10])",
        "(in3[9,11,10,4])" });
    ASSERT_FALSE(ea.admissionPolicy.theoremRelationDemandSlotQualifies(
        keyNeg.data(), keyNeg.count(), 2, 1, false));

    // An atomic other-premise (a typing fact) refuses.
    const KeyRun keyAtom({
        "(AnchorFTA[1,2,3,4,5,6,7,8])",
        "(in[9,1])",
        "(in3[9,11,10,4])" });
    ASSERT_FALSE(ea.admissionPolicy.theoremRelationDemandSlotQualifies(
        keyAtom.data(), keyAtom.count(), 2, 1, false));

    // A disintegration product (a `u_`-prefixed premise argument) is not a
    // received theorem: condition (1) refuses.
    const KeyRun keyU({
        "(AnchorFTA[1,2,3,4,5,6,7,8])",
        "(strictOrder[1,4,u_9,u_10])",
        "(in3[u_9,11,u_10,4])" });
    ASSERT_FALSE(ea.admissionPolicy.theoremRelationDemandSlotQualifies(
        keyU.data(), keyU.count(), 2, 1, false));

    // No anchor element: not a received theorem.
    const KeyRun keyNoAnchor({
        "(strictOrder[1,4,9,10])",
        "(in3[9,11,10,4])" });
    ASSERT_FALSE(ea.admissionPolicy.theoremRelationDemandSlotQualifies(
        keyNoAnchor.data(), keyNoAnchor.count(), 1, 1, false));
}

// The fifth qualification pass installs an UNTAGGED in3 marker with the
// marker at an INPUT slot for the row-20 shape, and installs none for the
// successor-witness shape — D-370.
TEST(disintegration_allowed, theorem_relation_demand_pass_installs_untagged_marker) {
    gl::tests::ProverFixture ea(std::string("FTA"));

    const std::vector<std::string> key20 = {
        "(AnchorFTA[1,2,3,4,5,6,7,8])",
        "(strictOrder[1,4,9,10])",
        "(in3[9,11,10,4])" };
    const std::string head20 = "!(=[2,11])";
    const std::string impl20 =
        "(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](strictOrder[1,4,9,10])"
        "(>[11](in3[9,11,10,4])!(=[2,11]))))";

    const int32_t inputBefore = countIn3MarkerAt(ea, 1);
    ea.admissionMarkers.makeNormalizedKeysForAdmission(key20, ((ea.body).overallHashMemory),
        ((ea.body).nameMap), ((ea.body).ruleInterner), head20,
        ea.parameters.minNumOperatorsKey, impl20, "main", &ea.body);
    ASSERT_TRUE(countIn3MarkerAt(ea, 1) > inputBefore);

    // Control: the successor-witness shape (row 13) mints no input-slot
    // marker — the bound-argument floor holds (in2 markers are counted
    // through the same decoder on the in3 filter: none expected either way,
    // so the control asserts on the in3 count staying flat AND on the
    // predicate itself).
    gl::tests::ProverFixture ea2(std::string("FTA"));
    const std::vector<std::string> key13 = {
        "(AnchorFTA[1,2,3,4,5,6,7,8])",
        "(in2[9,10,3])",
        "(preorder[1,4,10,11])" };
    const std::string head13 = "(preorder[1,4,9,11])";
    const std::string impl13 =
        "(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](in2[9,10,3])"
        "(>[11](preorder[1,4,10,11])(preorder[1,4,9,11]))))";
    const int32_t ctrlBefore = countIn3MarkerAt(ea2, 1);
    ea2.admissionMarkers.makeNormalizedKeysForAdmission(key13, ((ea2.body).overallHashMemory),
        ((ea2.body).nameMap), ((ea2.body).ruleInterner), head13,
        ea2.parameters.minNumOperatorsKey, impl13, "main", &ea2.body);
    ASSERT_TRUE(countIn3MarkerAt(ea2, 1) == ctrlBefore);
}

// The operator-relation route (D-494) on
// the G2 transport rule (supplied row 123): the q*y slot and the t+r slot
// are missing premises whose output alone is unspecified and sits in the
// head; the anchor, the relation, a bound output, an unspecified input, a
// repeated output, a head without the output, too few in3 premises and a
// key without a relation refuse.
TEST(disintegration_allowed, operator_relation_slot_qualification_matrix) {
    gl::tests::ProverFixture ea(std::string("FTA"));
    const gl::AdmissionPolicy& policy = ea.admissionPolicy;
    const std::string anchorFta = "(AnchorFTA[1,2,3,4,5,6,7,8])";
    const std::string head = "(existence26[16,14,4,1,5,6,13,15,9])";
    const gl::StrSpan h(head);
    const KeyRun rule({ anchorFta, "(in3[9,10,11,5])", "(in3[11,12,13,4])",
        "(bezEq[1,4,5,6,12,15,9,14])", "(in3[10,15,16,5])" });

    // Relation premises: a positive compound predicate only.
    ASSERT_TRUE(policy.isRelationPremise(rule.data()[3]));
    ASSERT_FALSE(policy.isRelationPremise(rule.data()[0]));
    ASSERT_FALSE(policy.isRelationPremise(rule.data()[1]));
    const std::string negated = "!(bezEq[1,4,5,6,12,15,9,14])";
    ASSERT_FALSE(policy.isRelationPremise(gl::StrSpan(negated)));
    const std::string membership = "(in[9,1])";
    ASSERT_FALSE(policy.isRelationPremise(gl::StrSpan(membership)));
    const std::string equality = "(=[12,15])";
    ASSERT_FALSE(policy.isRelationPremise(gl::StrSpan(equality)));

    // Row 123: w = q*y (slot 4) and a = t+r (slot 2) qualify; t = b*q
    // (slot 1) has its output t bound by a = t+r; the anchor and the
    // relation are no missing in3.
    ASSERT_TRUE(policy.operatorRelationSlotQualifies(rule.data(), rule.count(), 4, h, 2));
    ASSERT_TRUE(policy.operatorRelationSlotQualifies(rule.data(), rule.count(), 2, h, 2));
    ASSERT_FALSE(policy.operatorRelationSlotQualifies(rule.data(), rule.count(), 1, h, 2));
    ASSERT_FALSE(policy.operatorRelationSlotQualifies(rule.data(), rule.count(), 0, h, 2));
    ASSERT_FALSE(policy.operatorRelationSlotQualifies(rule.data(), rule.count(), 3, h, 2));

    // The missing output must appear in the head.
    const std::string headWithoutOutput = "(existence26[17,14,4,1,5,6,13,15,9])";
    ASSERT_FALSE(policy.operatorRelationSlotQualifies(rule.data(), rule.count(), 4,
        gl::StrSpan(headWithoutOutput), 2));

    // Every other argument must be specified by another premise.
    const KeyRun unspecified({ anchorFta, "(in3[9,10,11,5])", "(in3[11,12,13,4])",
        "(bezEq[1,4,5,6,12,15,9,14])", "(in3[10,18,16,5])" });
    ASSERT_FALSE(policy.operatorRelationSlotQualifies(unspecified.data(),
        unspecified.count(), 4, h, 2));
    const KeyRun repeated({ anchorFta, "(in3[9,10,11,5])", "(in3[11,12,13,4])",
        "(bezEq[1,4,5,6,12,15,9,14])", "(in3[10,16,16,5])" });
    ASSERT_FALSE(policy.operatorRelationSlotQualifies(repeated.data(),
        repeated.count(), 4, h, 2));

    // More than the threshold of in3 premises, and a relation, are needed.
    const KeyRun fewOperators({ anchorFta, "(in3[9,10,11,5])",
        "(bezEq[1,4,5,6,12,15,9,14])", "(in3[10,15,16,5])" });
    ASSERT_FALSE(policy.operatorRelationSlotQualifies(fewOperators.data(),
        fewOperators.count(), 3, h, 2));
    const KeyRun noRelation({ anchorFta, "(in3[9,10,11,5])", "(in3[11,12,13,4])",
        "(=[12,15])", "(in3[10,15,16,5])" });
    ASSERT_FALSE(policy.operatorRelationSlotQualifies(noRelation.data(),
        noRelation.count(), 4, h, 2));

    // Route (D) qualifies the rule for the regular pass; the classic gate
    // stays closed, and both overloads agree.
    ASSERT_FALSE(policy.baselineClassicQualifies(rule.data(), rule.count(), h, 2));
    ASSERT_TRUE(policy.implicationIsQualified(rule.data(), rule.count(), h, 2));
    ASSERT_TRUE(policy.implicationIsQualified(rule.storage, head, 2));
    ASSERT_FALSE(policy.implicationIsQualified(noRelation.data(), noRelation.count(), h, 2));
    ASSERT_FALSE(policy.implicationIsQualified(noRelation.storage, head, 2));
}

// The regular pass installs the operator-relation output marker for row 123
// (an untagged in3 marker at the output slot) and none for the same rule
// whose head carries neither output.
TEST(disintegration_allowed, operator_relation_rule_installs_output_marker) {
    gl::tests::ProverFixture ea(std::string("FTA"));
    const std::vector<std::string> key123 = {
        "(AnchorFTA[1,2,3,4,5,6,7,8])", "(in3[9,10,11,5])", "(in3[11,12,13,4])",
        "(bezEq[1,4,5,6,12,15,9,14])", "(in3[10,15,16,5])" };
    const std::string head123 = "(existence26[16,14,4,1,5,6,13,15,9])";
    const std::string impl123 =
        "(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10,11](in3[9,10,11,5])"
        "(>[12,13](in3[11,12,13,4])(>[15,14](bezEq[1,4,5,6,12,15,9,14])"
        "(>[16](in3[10,15,16,5])(existence26[16,14,4,1,5,6,13,15,9]))))))";
    const int32_t outputBefore = countIn3MarkerAt(ea, 2);
    ea.admissionMarkers.makeNormalizedKeysForAdmission(key123, ((ea.body).overallHashMemory),
        ((ea.body).nameMap), ((ea.body).ruleInterner), head123, 2, impl123, "main",
        &ea.body);
    ASSERT_TRUE(countIn3MarkerAt(ea, 2) > outputBefore);

    gl::tests::ProverFixture ea2(std::string("FTA"));
    const std::string headControl = "(existence26[17,14,4,1,5,6,18,15,9])";
    const std::string implControl =
        "(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10,11](in3[9,10,11,5])"
        "(>[12,13](in3[11,12,13,4])(>[15,14](bezEq[1,4,5,6,12,15,9,14])"
        "(>[16](in3[10,15,16,5])(existence26[17,14,4,1,5,6,18,15,9]))))))";
    const int32_t controlBefore = countIn3MarkerAt(ea2, 2);
    ea2.admissionMarkers.makeNormalizedKeysForAdmission(key123, ((ea2.body).overallHashMemory),
        ((ea2.body).nameMap), ((ea2.body).ruleInterner), headControl, 2, implControl,
        "main", &ea2.body);
    ASSERT_TRUE(countIn3MarkerAt(ea2, 2) == controlBefore);
}

// The compound-operator route (D-569) on FTA2
// row 16: the append slot is the missing compound premise (its output 13
// occurs only in the head); the anchor, the successor, the sequence relation,
// the fold (its output 14 feeds in3) and in3 refuse. A head without the
// output, an unspecified input, a fifth operator premise, row 17's three
// compound operators, an atomic or negated head and FTA pool row 87's two
// missing fold premises refuse too.
TEST(disintegration_allowed, compound_operator_slot_qualification_matrix) {
    gl::tests::ProverFixture ea(std::string("FTA"));
    const gl::AdmissionPolicy& policy = ea.admissionPolicy;
    const std::string anchorFta = "(AnchorFTA[1,2,3,4,5,6,7,8])";
    const std::string head = "(fold[1,3,4,5,13,2,10,15])";
    const gl::StrSpan h(head);
    const KeyRun row16({ anchorFta, "(in2[9,10,3])", "(sequence[1,4,2,9,11])",
        "(appendToSequence[10,12,11,13])", "(fold[1,3,4,5,11,2,9,14])", "(in3[14,12,15,5])" });
    ASSERT_TRUE(policy.compoundOperatorSlotQualifies(row16.data(), row16.count(), 3, h));
    for (const int32_t i : { 0, 1, 2, 4, 5 })
        ASSERT_FALSE(policy.compoundOperatorSlotQualifies(row16.data(), row16.count(), i, h));

    // The missing output must appear in the head.
    const std::string headWithoutOutput = "(fold[1,3,4,5,16,2,10,15])";
    ASSERT_FALSE(policy.compoundOperatorSlotQualifies(row16.data(), row16.count(), 3,
        gl::StrSpan(headWithoutOutput)));

    // Every other argument must be specified by another premise.
    const KeyRun unspecified({ anchorFta, "(in2[9,10,3])", "(sequence[1,4,2,9,11])",
        "(appendToSequence[10,17,11,13])", "(fold[1,3,4,5,11,2,9,14])", "(in3[14,12,15,5])" });
    ASSERT_FALSE(policy.compoundOperatorSlotQualifies(unspecified.data(), unspecified.count(), 3, h));

    // Exactly four operator premises: a fifth one refuses.
    const KeyRun fiveOperators({ anchorFta, "(in2[9,10,3])", "(sequence[1,4,2,9,11])",
        "(appendToSequence[10,12,11,13])", "(fold[1,3,4,5,11,2,9,14])", "(in3[14,12,15,5])",
        "(in3[9,9,16,4])" });
    ASSERT_FALSE(policy.compoundOperatorSlotQualifies(fiveOperators.data(), fiveOperators.count(), 3, h));

    // Exactly two compound operators: row 17 (append and two folds) refuses.
    const std::string head17 = "(in3[14,12,15,5])";
    const KeyRun row17({ anchorFta, "(in2[9,10,3])", "(sequence[1,4,2,9,11])", "(in[12,1])",
        "(appendToSequence[10,12,11,13])", "(fold[1,3,4,5,11,2,9,14])", "(fold[1,3,4,5,13,2,10,15])" });
    for (int32_t i = 0; i < row17.count(); ++i)
        ASSERT_FALSE(policy.compoundOperatorSlotQualifies(row17.data(), row17.count(), i,
            gl::StrSpan(head17)));

    // The head must be a positive compound: row 16's key under an atomic head
    // naming the append output, or under the negated fold head, refuses.
    const std::string atomicHead = "(in3[13,12,15,5])";
    ASSERT_FALSE(policy.compoundOperatorSlotQualifies(row16.data(), row16.count(), 3,
        gl::StrSpan(atomicHead)));
    const std::string negatedHead = "!(fold[1,3,4,5,13,2,10,15])";
    ASSERT_FALSE(policy.compoundOperatorSlotQualifies(row16.data(), row16.count(), 3,
        gl::StrSpan(negatedHead)));

    // Exactly one missing compound premise: FTA pool row 87 (both fold values
    // only its in3 head names) refuses on every slot, and so does its key
    // under a compound head naming both fold values.
    const std::string head87 = "(in3[13,12,14,5])";
    const KeyRun row87({ anchorFta, "(in2[9,10,3])", "(sequence[1,4,2,10,11])", "(in2[10,12,11])",
        "(fold[1,3,4,5,11,2,9,13])", "(fold[1,3,4,5,11,2,10,14])" });
    const std::string compoundHead87 = "(fold[1,3,4,5,11,2,13,14])";
    for (int32_t i = 0; i < row87.count(); ++i) {
        ASSERT_FALSE(policy.compoundOperatorSlotQualifies(row87.data(), row87.count(), i,
            gl::StrSpan(head87)));
        ASSERT_FALSE(policy.compoundOperatorSlotQualifies(row87.data(), row87.count(), i,
            gl::StrSpan(compoundHead87)));
    }

    // Route (E) qualifies row 16 for the regular pass; the classic and the
    // operator-relation gates stay closed, and both overloads agree.
    ASSERT_FALSE(policy.baselineClassicQualifies(row16.data(), row16.count(), h, 2));
    for (int32_t i = 0; i < row16.count(); ++i)
        ASSERT_FALSE(policy.operatorRelationSlotQualifies(row16.data(), row16.count(), i, h, 2));
    ASSERT_TRUE(policy.implicationIsQualified(row16.data(), row16.count(), h, 2));
    ASSERT_TRUE(policy.implicationIsQualified(row16.storage, head, 2));
    ASSERT_FALSE(policy.implicationIsQualified(row17.data(), row17.count(), gl::StrSpan(head17), 2));
    ASSERT_FALSE(policy.implicationIsQualified(row17.storage, head17, 2));
}

// The regular pass installs the compound-operator output marker for row 16:
// an untagged append marker at its output slot
// (D-569).
TEST(disintegration_allowed, compound_operator_rule_installs_output_marker) {
    gl::tests::ProverFixture ea(std::string("FTA"));
    const std::vector<std::string> key16 = { "(AnchorFTA[1,2,3,4,5,6,7,8])", "(in2[9,10,3])",
        "(sequence[1,4,2,9,11])", "(appendToSequence[10,12,11,13])",
        "(fold[1,3,4,5,11,2,9,14])", "(in3[14,12,15,5])" };
    const std::string head16 = "(fold[1,3,4,5,13,2,10,15])";
    const std::string impl16 =
        "(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](in2[9,10,3])"
        "(>[11](sequence[1,4,2,9,11])(>[12,13](appendToSequence[10,12,11,13])"
        "(>[14](fold[1,3,4,5,11,2,9,14])(>[15](in3[14,12,15,5])(fold[1,3,4,5,13,2,10,15])))))))";
    const int32_t before = countMarkerOf(ea, "appendToSequence", 3);
    ea.admissionMarkers.makeNormalizedKeysForAdmission(key16, ea.body.overallHashMemory,
        ea.body.nameMap, ea.body.ruleInterner, head16, ea.parameters.minNumOperatorsKey, impl16,
        "main", &ea.body);
    ASSERT_TRUE(countMarkerOf(ea, "appendToSequence", 3) > before);
}
