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
/// @brief Tests for `ExpressionAnalyzer::constructOrEliminationInRun` —
///        the in-run pre-split merge: guard-pair detection, or-theorem
///        licensing, pending-pair retry, dedup, and literal polarity.

#include "test_harness.hpp"

#include "../prover.hpp"

#include <string>
#include <vector>

namespace {

// Peano-anchored pre-split fixture (slots: 1=N 2=i0 3=s 4=+ 5=* 6=i1).
// The bridge is the row-24 shape whose in-run or theorem licenses the
// merge; the variants are the C13 pre-split pair; merged is the derived
// antisymmetry theorem the seam must rebuild canonically.
const char* kBridge =
    "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])"
    "(>[9](in[9,1])(>[]!(=[2,9])(preorder[1,4,6,9]))))";
const char* kVariantA =
    "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])"
    "(>[9,10](preorder[1,5,9,10])(>[](preorder[1,5,10,9])"
    "(>[](preorder[1,4,6,10])(=[9,10])))))";
const char* kVariantB =
    "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])"
    "(>[9,10](preorder[1,5,9,10])(>[](preorder[1,5,10,9])"
    "(>[](=[2,10])(=[9,10])))))";
const char* kMerged =
    "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])"
    "(>[9,10](preorder[1,5,9,10])(>[](preorder[1,5,10,9])(=[9,10]))))";

} // namespace

// The full flow: the bridge's in-run or theorem licenses the variant
// pair; the merged theorem lands as a DERIVED row with method
// "or elimination", both variants as aux citations, the or theorem in
// the citation ledger, and a queued broadcast.
TEST(or_elimination, merges_pair_under_minted_or_license) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    ASSERT_TRUE(ea.appendGlobalTheorem(kBridge, "direct", "-1", "-1"));
    ea.constructOrTheoremsInRun();
    const std::string orThm = std::get<0>(ea.globalTheoremList.back());
    ASSERT_TRUE(std::get<1>(ea.globalTheoremList.back()) == std::string("or theorem"));

    ASSERT_TRUE(ea.appendGlobalTheorem(kVariantA, "direct", "-1", "-1"));
    ASSERT_TRUE(ea.appendGlobalTheorem(kVariantB, "direct", "-1", "-1"));
    const std::size_t compactionsBefore = ea.pendingCompactionQueue.size();

    ea.constructOrEliminationInRun();

    const auto& last = ea.globalTheoremList.back();
    ASSERT_TRUE(std::get<0>(last) == std::string(kMerged));
    ASSERT_TRUE(std::get<1>(last) == std::string("or elimination"));
    ASSERT_TRUE(std::get<2>(last) == std::string(kVariantA));
    ASSERT_TRUE(std::get<3>(last) == std::string(kVariantB));
    ASSERT_TRUE(std::get<0>(ea.fullTheoremList.back()) == std::string(kMerged));
    const auto cit = ea.orElimCitedOrByMerged.find(kMerged);
    ASSERT_TRUE(cit != ea.orElimCitedOrByMerged.end());
    ASSERT_TRUE(cit->second == orThm);
    ASSERT_EQ(static_cast<int>(ea.pendingCompactionQueue.size()),
              static_cast<int>(compactionsBefore) + 1);
    ASSERT_TRUE(ea.orElimPendingPairs.empty());
}

// A pair whose license has not been minted yet parks and merges on a
// LATER seam invocation once the or theorem exists — the retry contract.
TEST(or_elimination, license_pending_until_or_arrives) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    ASSERT_TRUE(ea.appendGlobalTheorem(kVariantA, "direct", "-1", "-1"));
    ASSERT_TRUE(ea.appendGlobalTheorem(kVariantB, "direct", "-1", "-1"));

    ea.constructOrEliminationInRun();
    ASSERT_EQ(static_cast<int>(ea.orElimPendingPairs.size()), 1);
    ASSERT_TRUE(std::get<1>(ea.globalTheoremList.back()) != std::string("or elimination"));

    ASSERT_TRUE(ea.appendGlobalTheorem(kBridge, "direct", "-1", "-1"));
    ea.constructOrTheoremsInRun();
    ea.constructOrEliminationInRun();

    ASSERT_TRUE(ea.orElimPendingPairs.empty());
    const auto& last = ea.globalTheoremList.back();
    ASSERT_TRUE(std::get<0>(last) == std::string(kMerged));
    ASSERT_TRUE(std::get<1>(last) == std::string("or elimination"));
}

// Re-invoking the seam after a merge changes nothing: the scanned-row
// set, the pair ledger, and the append door each block a repeat.
TEST(or_elimination, rescan_is_idempotent) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    ASSERT_TRUE(ea.appendGlobalTheorem(kBridge, "direct", "-1", "-1"));
    ea.constructOrTheoremsInRun();
    ASSERT_TRUE(ea.appendGlobalTheorem(kVariantA, "direct", "-1", "-1"));
    ASSERT_TRUE(ea.appendGlobalTheorem(kVariantB, "direct", "-1", "-1"));
    ea.constructOrEliminationInRun();

    const std::size_t rows = ea.globalTheoremList.size();
    const std::size_t compactions = ea.pendingCompactionQueue.size();
    ea.constructOrEliminationInRun();
    ASSERT_EQ(static_cast<int>(ea.globalTheoremList.size()), static_cast<int>(rows));
    ASSERT_EQ(static_cast<int>(ea.pendingCompactionQueue.size()), static_cast<int>(compactions));
    ASSERT_TRUE(ea.orElimPendingPairs.empty());
}

// If the merged row already exists (proved through another route), the
// append door's string dedup refuses: no mirror row, no citation, no
// broadcast — and the pair is recorded so it is never re-attempted.
TEST(or_elimination, dedup_when_merged_row_preexists) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    ASSERT_TRUE(ea.appendGlobalTheorem(kMerged, "direct", "-1", "-1"));
    ASSERT_TRUE(ea.appendGlobalTheorem(kBridge, "direct", "-1", "-1"));
    ea.constructOrTheoremsInRun();
    ASSERT_TRUE(ea.appendGlobalTheorem(kVariantA, "direct", "-1", "-1"));
    ASSERT_TRUE(ea.appendGlobalTheorem(kVariantB, "direct", "-1", "-1"));
    const std::size_t compactionsBefore = ea.pendingCompactionQueue.size();

    ea.constructOrEliminationInRun();

    ASSERT_TRUE(std::get<1>(ea.globalTheoremList.back()) != std::string("or elimination"));
    ASSERT_TRUE(ea.orElimCitedOrByMerged.find(kMerged) == ea.orElimCitedOrByMerged.end());
    ASSERT_EQ(static_cast<int>(ea.pendingCompactionQueue.size()),
              static_cast<int>(compactionsBefore));
    ASSERT_TRUE(ea.orElimPendingPairs.empty());
    ASSERT_EQ(static_cast<int>(ea.orElimBuiltByPair.size()), 1);
}

// Polarity is literal (I-175): a NEGATED guard does not align with the
// or theorem's un-negated disjunct, so no merge fires and the pair
// stays parked.
TEST(or_elimination, polarity_literal_negated_guard_refused) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    const std::string variantBNeg =
        "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])"
        "(>[9,10](preorder[1,5,9,10])(>[](preorder[1,5,10,9])"
        "(>[]!(=[2,10])(=[9,10])))))";
    ASSERT_TRUE(ea.appendGlobalTheorem(kBridge, "direct", "-1", "-1"));
    ea.constructOrTheoremsInRun();
    ASSERT_TRUE(ea.appendGlobalTheorem(kVariantA, "direct", "-1", "-1"));
    ASSERT_TRUE(ea.appendGlobalTheorem(variantBNeg, "direct", "-1", "-1"));

    ea.constructOrEliminationInRun();

    ASSERT_TRUE(std::get<1>(ea.globalTheoremList.back()) != std::string("or elimination"));
    ASSERT_EQ(static_cast<int>(ea.orElimPendingPairs.size()), 1);
}

// Level-refused closures land in the proved-not-broadcast tier: recorded
// with the marker method, no broadcast, no reformulation, and the or seam
// never folds ors from tier rows (zero circulation).
TEST(or_elimination, level_refused_registration_lands_in_tier) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    const std::size_t compactionsBefore = ea.pendingCompactionQueue.size();
    ea.updateGlobalDirect(kVariantB, /*coreId=*/-1, &ea.body,
                          /*registerGlobally=*/false);
    ASSERT_EQ(static_cast<int>(ea.globalTheoremList.size()), 1);
    const auto& last = ea.globalTheoremList.back();
    ASSERT_TRUE(std::get<0>(last) == std::string(kVariantB));
    ASSERT_TRUE(std::get<1>(last) == std::string("proved not broadcast"));
    ASSERT_EQ(static_cast<int>(ea.pendingCompactionQueue.size()),
              static_cast<int>(compactionsBefore));

    // A tier row carrying a negated premise must not seed the or seam.
    ea.updateGlobalDirect(kBridge, /*coreId=*/-1, &ea.body,
                          /*registerGlobally=*/false);
    const std::size_t rows = ea.globalTheoremList.size();
    ea.constructOrTheoremsInRun();
    ASSERT_EQ(static_cast<int>(ea.globalTheoremList.size()),
              static_cast<int>(rows));
}

// The pre-split merge accepts a tier row as a guard variant: 13a
// first-class, 13b proved-not-broadcast, the or license minted normally —
// the merged theorem lands as a first-class derived row.
TEST(or_elimination, merge_consumes_tier_variant) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    ASSERT_TRUE(ea.appendGlobalTheorem(kBridge, "direct", "-1", "-1"));
    ea.constructOrTheoremsInRun();
    ASSERT_TRUE(ea.appendGlobalTheorem(kVariantA, "direct", "-1", "-1"));
    ea.updateGlobalDirect(kVariantB, /*coreId=*/-1, &ea.body,
                          /*registerGlobally=*/false);

    ea.constructOrEliminationInRun();

    const auto& last = ea.globalTheoremList.back();
    ASSERT_TRUE(std::get<0>(last) == std::string(kMerged));
    ASSERT_TRUE(std::get<1>(last) == std::string("or elimination"));
    ASSERT_TRUE(std::get<2>(last) == std::string(kVariantA));
    ASSERT_TRUE(std::get<3>(last) == std::string(kVariantB));
}

// Tier registration completes dangling proved-theorem citations in the
// producer chain: a cited registered theorem lacking its own history line
// (its broadcast never landed — the LB dies in the closing window) gains
// a `theorem` origin row so the chapter walk terminates. Rule 16:
// completing missing process documentation, never a proof input.
TEST(or_elimination, tier_registration_repairs_dangling_citation) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    ASSERT_TRUE(ea.appendGlobalTheorem(kVariantA, "direct", "-1", "-1"));

    // Seed the producer's history: a derived head citing the registered
    // theorem as an implication dependency — with no history line of the
    // theorem's own.
    gl::addOriginEncoded(ea.body.exprOriginMap, ea.body.originInterner,
        gl::ExpressionWithValidity("(=[7,8])", "main"),
        std::make_pair(std::string("implication"),
            std::vector<gl::ExpressionWithValidity>{
                gl::ExpressionWithValidity(kVariantA, "main")}),
        5);
    {
        int64_t pk = 0;
        const bool hasKey = gl::lookupOriginKey(ea.body.originInterner,
                                                kVariantA, "main", pk);
        ASSERT_TRUE(!hasKey || ea.body.exprOriginMap.lookup(pk) == 0);
    }

    ea.updateGlobalDirect(kVariantB, /*coreId=*/-1, &ea.body,
                          /*registerGlobally=*/false);

    int64_t pk = 0;
    ASSERT_TRUE(gl::lookupOriginKey(ea.body.originInterner, kVariantA,
                                    "main", pk));
    const int32_t oid = ea.body.exprOriginMap.lookup(pk);
    ASSERT_TRUE(oid != 0);
    ASSERT_TRUE(ea.body.exprOriginMap.runLen(oid) > 0);
}

// Compressor mode is a defined no-op, exactly like the or seam: nothing
// is scanned, filed, or merged.
TEST(or_elimination, compressor_mode_noop) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    ASSERT_TRUE(ea.appendGlobalTheorem(kBridge, "direct", "-1", "-1"));
    ea.constructOrTheoremsInRun();
    ASSERT_TRUE(ea.appendGlobalTheorem(kVariantA, "direct", "-1", "-1"));
    ASSERT_TRUE(ea.appendGlobalTheorem(kVariantB, "direct", "-1", "-1"));

    ea.parameters.compressor_mode = true;
    ea.constructOrEliminationInRun();

    ASSERT_TRUE(ea.orElimInRunScannedRows.empty());
    ASSERT_TRUE(ea.orElimPendingPairs.empty());
    ASSERT_TRUE(std::get<1>(ea.globalTheoremList.back()) != std::string("or elimination"));
}
