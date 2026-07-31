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
/// @brief Tests for the OR-intro rule emission (D-237):
///        an implication whose premise element is a disjunction also
///        installs one per-leaf intro implication `(>[bound](D_k)(or…))`.
///
/// @details
/// Covers `ExpressionAnalyzer::disintegrateExprCore2`'s implication branch:
/// disintegrating a statement whose compiled form carries an or-shaped
/// premise (the EnumerationSet3 description's membership-intro conjunct)
/// must install, alongside the main rule, one single-premise intro rule per
/// flattened disjunct — each preserving the or premise's argument `u_`
/// status (u_ args free, the changeable element variable bound).

#include "test_harness.hpp"

#include "../prover.hpp"

namespace {

    /// @brief Decode every rule chain in @p mb's overall hash memory into
    ///        string vectors, mirroring the sacred dump's originals section.
    ///
    /// @details
    /// Iterates `overallHashMemory.originals` ids 1..count and decodes each
    /// id-vector key through the LB's `ruleInterner` — the same derived view
    /// `hashburst_dump.cpp` prints, so assertions read the exact installed
    /// chain strings.
    ///
    /// @param mb The memory block whose rules are read.
    /// @return All chains, each as premise strings followed by the head.
    std::vector<std::vector<std::string>> decodeOriginalChains(gl::Memory& mb) {
        std::vector<std::vector<std::string>> chains;
        const int32_t n = mb.overallHashMemory.originals.count();
        chains.reserve(static_cast<std::size_t>(n));
        for (int32_t oi = 1; oi <= n; ++oi) {
            chains.push_back(gl::decodeValueVector(
                mb.overallHashMemory.originals.decodeKey(oi).ids,
                mb.ruleInterner));
        }
        return chains;
    }

    /// @brief Whether @p chains contains a two-element chain
    ///        `[premise, head]`.
    ///
    /// @param chains Decoded rule chains from @ref decodeOriginalChains.
    /// @param premise The exact premise string.
    /// @param head The exact head string.
    /// @return Whether the chain is present.
    bool hasIntroChain(const std::vector<std::vector<std::string>>& chains,
                       const std::string& premise, const std::string& head) {
        for (const auto& c : chains) {
            if (c.size() == 2 && c[0] == premise && c[1] == head) return true;
        }
        return false;
    }

} // namespace

TEST(or_intro_rules, es3_membership_intro_conjunct_emits_per_leaf_intros) {
    gl::ExpressionAnalyzer ea(std::string("IncubatorGauss3"));

    // Disintegrate the ES3 description at the root memory: the `and` branch
    // recurses into both conjuncts; the membership-intro conjunct is an
    // implication whose premise element is the compiled three-way
    // disjunction.
    const std::string es3 = "(EnumerationSet3[2,6,7,10])";
    const int lv[1] = { 0 };
    const gl::TransientOrigin origin{
        true, gl::OriginTag::taskFormulation, nullptr, 0 };
    ea.addExprToMemoryBlock(gl::StrSpan(es3), ea.body, 0, 0, lv, 1, origin,
        -1, -1, gl::StrSpan("main", 4), false);

    const auto chains = decodeOriginalChains(ea.body);

    // Locate the membership-intro main rule `(or<N>[a,1,b,c]) (in[1,u_10])`
    // — its or-premise string is the head every intro rule must state, and
    // its presence proves the implication branch ran on the right conjunct.
    std::string orPremise;
    for (const auto& c : chains) {
        if (c.size() == 2 && c[1] == "(in[1,u_10])"
            && c[0].rfind("(or", 0) == 0) {
            orPremise = c[0];
            break;
        }
    }
    ASSERT_TRUE(!orPremise.empty());

    // The three flattened disjuncts of the ES3 disjunction, instantiated
    // with the conjunct's argument naming: the or premise's signature args
    // are (u_A, 1, u_B, u_C) with `1` the changeable element variable, so
    // the leaves are (=[u_A,1]) / (=[u_B,1]) / (=[u_C,1]).
    std::vector<std::string> orArgs;
    {
        const std::size_t lb = orPremise.find('[');
        const std::size_t rb = orPremise.rfind(']');
        ASSERT_TRUE(lb != std::string::npos && rb != std::string::npos);
        std::string inner = orPremise.substr(lb + 1, rb - lb - 1);
        std::size_t pos = 0;
        while (pos != std::string::npos) {
            const std::size_t comma = inner.find(',', pos);
            orArgs.push_back(inner.substr(pos,
                comma == std::string::npos ? std::string::npos : comma - pos));
            pos = (comma == std::string::npos) ? std::string::npos : comma + 1;
        }
    }
    ASSERT_EQ(orArgs.size(), static_cast<std::size_t>(4));

    // Every intro rule is installed as the two-element chain
    // [ (=[u_X,1]), orPremise ] — the leaf in RAW element form (u_ args
    // free), the head the or premise verbatim, the element variable bound.
    int introCount = 0;
    for (const auto& a : orArgs) {
        if (a == "1") continue; // the bound element variable, not a leaf value
        ASSERT_TRUE(a.rfind("u_", 0) == 0);
        const std::string leaf = "(=[" + a + ",1])";
        ASSERT_TRUE(hasIntroChain(chains, leaf, orPremise));
        ++introCount;
    }
    ASSERT_EQ(introCount, 3);
}

TEST(or_intro_rules, atomic_premises_emit_no_intro) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));

    // A Peano compact with only atomic (non-or) premises: no intro chain of
    // the shape [ single premise, (or...) head ] may appear.
    const std::string nat = "(NaturalNumbers[1,2,3,4,5])";
    const int lv[1] = { 0 };
    const gl::TransientOrigin origin{
        true, gl::OriginTag::taskFormulation, nullptr, 0 };
    ea.addExprToMemoryBlock(gl::StrSpan(nat), ea.body, 0, 0, lv, 1, origin,
        -1, -1, gl::StrSpan("main", 4), false);

    const auto chains = decodeOriginalChains(ea.body);
    ASSERT_TRUE(!chains.empty());
    for (const auto& c : chains) {
        const bool introShaped = c.size() == 2
            && c[1].rfind("(or", 0) == 0;
        ASSERT_FALSE(introShaped);
    }
}
