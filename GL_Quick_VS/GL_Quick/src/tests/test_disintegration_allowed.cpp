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
