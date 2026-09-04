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
/// @brief Tests for `ExpressionAnalyzer::constructOrTheoremsFromPairs` —
///        single-direction OR licensing, mirror-pair dedup, and the
///        pruned-source skip.

#include "test_harness.hpp"

#include "../prover.hpp"

#include <set>
#include <string>
#include <vector>

// A single proved straightened form licenses its OR: the companion is
// never proved, yet the OR is constructed, registered with method
// "or theorem", and only the proved parent is consumed.
TEST(or_theorem_construction, single_direction_constructs_or) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    const std::string totality =
        "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])"
        "(>[9](in[9,1])(>[10](in[10,1])"
        "(>[]!(preorder[1,4,9,10])(preorder[1,4,10,9])))))";
    const std::string companion = ea.headSwitchOne(totality);
    ASSERT_TRUE(!companion.empty());

    ASSERT_TRUE(ea.appendGlobalTheorem(totality, "direct", "-1", "-1"));
    ea.orPairsFromHeadSwitch.push_back({totality, companion});

    std::set<std::string> consumed;
    const std::vector<std::string> ors = ea.constructOrTheoremsFromPairs(consumed);

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
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    const std::string parent =
        "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])"
        "(>[7](in[7,1])(>[]!(=[7,2])(existence3[1,7,3]))))";
    const std::string mirror = ea.headSwitchOne(parent);
    ASSERT_TRUE(!mirror.empty());

    ASSERT_TRUE(ea.appendGlobalTheorem(parent, "direct", "-1", "-1"));
    ASSERT_TRUE(ea.appendGlobalTheorem(mirror, "direct", "-1", "-1"));
    ea.orPairsFromHeadSwitch.push_back({parent, mirror});
    ea.orPairsFromHeadSwitch.push_back({mirror, parent});

    std::set<std::string> consumed;
    const std::vector<std::string> ors = ea.constructOrTheoremsFromPairs(consumed);

    ASSERT_EQ(static_cast<int>(ors.size()), 1);
    ASSERT_EQ(static_cast<int>(consumed.size()), 2);
    ASSERT_TRUE(consumed.count(parent) == 1);
    ASSERT_TRUE(consumed.count(mirror) == 1);
}

// A hypothesis chain with TWO binder-free negated premises folds them BOTH
// into the OR head — three disjuncts in chain order (folded premises
// un-negated, head last), no negated premise surviving as a hypothesis,
// and the flat three-conjunct De Morgan base form. The two-branch fold
// left the second negated premise behind as a hypothesis (the trichotomy
// regression).
TEST(or_theorem_construction, chain_of_negated_premises_folds_all_branches) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    const std::string trichotomy =
        "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])"
        "(>[9](in[9,1])(>[10](in[10,1])"
        "(>[]!(preorder[1,4,9,10])(>[]!(=[9,10])(preorder[1,4,10,9]))))))";
    const std::string companion = ea.headSwitchOne(trichotomy);
    ASSERT_TRUE(!companion.empty());

    ASSERT_TRUE(ea.appendGlobalTheorem(trichotomy, "direct", "-1", "-1"));
    ea.orPairsFromHeadSwitch.push_back({trichotomy, companion});

    std::set<std::string> consumed;
    const std::vector<std::string> ors = ea.constructOrTheoremsFromPairs(consumed);
    ASSERT_EQ(static_cast<int>(ors.size()), 1);

    // The registered or-entity carries all three disjuncts in chain order,
    // u_-canonical, TRUE polarity.
    std::string orName;
    for (const auto& [name, le] : ea.compiledExpressions) {
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

    // Base-form export expands the head to the FLAT three-conjunct
    // De Morgan form.
    const std::string base = ea.expandToBaseForm(ors[0]);
    ASSERT_TRUE(base.find(
        "!(&!(preorder[1,4,9,10])!(=[9,10])!(preorder[1,4,10,9]))")
        != std::string::npos);
}

// The in-run seam folds a freshly proved row into its or theorem,
// registers it with method "or theorem", and queues exactly one
// implication-compact broadcast on pendingCompactionQueue.
TEST(or_theorem_construction, in_run_constructs_registers_and_queues) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    const std::string totality =
        "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])"
        "(>[9](in[9,1])(>[10](in[10,1])"
        "(>[]!(preorder[1,4,9,10])(preorder[1,4,10,9])))))";
    ASSERT_TRUE(ea.appendGlobalTheorem(totality, "direct", "-1", "-1"));

    ea.constructOrTheoremsInRun();

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

// A second in-run call scans nothing new: every row is in the scanned
// set, so no duplicate or row and no duplicate broadcast entry appears.
TEST(or_theorem_construction, in_run_rescan_is_idempotent) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    const std::string totality =
        "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])"
        "(>[9](in[9,1])(>[10](in[10,1])"
        "(>[]!(preorder[1,4,9,10])(preorder[1,4,10,9])))))";
    ASSERT_TRUE(ea.appendGlobalTheorem(totality, "direct", "-1", "-1"));

    ea.constructOrTheoremsInRun();
    const std::size_t rows = ea.globalTheoremList.size();
    const std::size_t queued = ea.pendingCompactionQueue.size();

    ea.constructOrTheoremsInRun();

    ASSERT_EQ(static_cast<int>(ea.globalTheoremList.size()),
              static_cast<int>(rows));
    ASSERT_EQ(static_cast<int>(ea.pendingCompactionQueue.size()),
              static_cast<int>(queued));
}

// A mirror row proved LATER reconstructs the same or through the registry
// reuse; the string dedup gates it — one or row, one broadcast entry.
TEST(or_theorem_construction, in_run_mirror_scanned_later_no_double) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    const std::string parent =
        "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])"
        "(>[9](in[9,1])(>[10](in[10,1])"
        "(>[]!(preorder[1,4,9,10])(preorder[1,4,10,9])))))";
    const std::string mirror = ea.headSwitchOne(parent);
    ASSERT_TRUE(!mirror.empty());

    ASSERT_TRUE(ea.appendGlobalTheorem(parent, "direct", "-1", "-1"));
    ea.constructOrTheoremsInRun();
    ASSERT_TRUE(ea.appendGlobalTheorem(mirror, "direct", "-1", "-1"));
    ea.constructOrTheoremsInRun();

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
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    const std::string totality =
        "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])"
        "(>[9](in[9,1])(>[10](in[10,1])"
        "(>[]!(preorder[1,4,9,10])(preorder[1,4,10,9])))))";
    const std::string companion = ea.headSwitchOne(totality);
    ASSERT_TRUE(!companion.empty());

    ASSERT_TRUE(ea.appendGlobalTheorem(totality, "direct", "-1", "-1"));
    ea.constructOrTheoremsInRun();
    const std::size_t rowsAfterInRun = ea.globalTheoremList.size();
    ASSERT_EQ(static_cast<int>(ea.fullTheoremList.size()), 1);

    ea.orPairsFromHeadSwitch.push_back({totality, companion});
    std::set<std::string> consumed;
    const std::vector<std::string> ors = ea.constructOrTheoremsFromPairs(consumed);

    // Return vector + subsumption fire; registration does not repeat.
    ASSERT_EQ(static_cast<int>(ors.size()), 1);
    ASSERT_TRUE(consumed.count(totality) == 1);
    ASSERT_EQ(static_cast<int>(ea.globalTheoremList.size()),
              static_cast<int>(rowsAfterInRun));
    ASSERT_EQ(static_cast<int>(ea.fullTheoremList.size()), 1);
}

// Compressor mode is a defined no-op for the in-run seam.
TEST(or_theorem_construction, in_run_compressor_mode_noop) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    ea.parameters.compressor_mode = true;
    const std::string totality =
        "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])"
        "(>[9](in[9,1])(>[10](in[10,1])"
        "(>[]!(preorder[1,4,9,10])(preorder[1,4,10,9])))))";
    ASSERT_TRUE(ea.appendGlobalTheorem(totality, "direct", "-1", "-1"));

    ea.constructOrTheoremsInRun();

    ASSERT_EQ(static_cast<int>(ea.globalTheoremList.size()), 1);
    ASSERT_EQ(static_cast<int>(ea.pendingCompactionQueue.size()), 0);
}

// findOrMintOrOperator: a fresh element list mints or<orCounter> into BOTH
// registries with the canonical u_ signature and bumps the counter; a
// second call with the identical list reuses the entry (I-23) and leaves
// the counter untouched.
TEST(or_theorem_construction, find_or_mint_reuse_and_mint) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    const int counterBefore = ea.orCounter;
    const std::vector<std::string> elems = {
        "(zztesta[u_1,u_2])", "(zztestb[u_2,u_1])" };

    bool minted = false;
    const std::string name = ea.findOrMintOrOperator(elems, 2, &minted);
    ASSERT_TRUE(minted);
    ASSERT_EQ(ea.orCounter, counterBefore + 1);
    const auto it = ea.compiledExpressions.find(name);
    ASSERT_TRUE(it != ea.compiledExpressions.end());
    ASSERT_TRUE(it->second.category == "or");
    ASSERT_TRUE(it->second.elements == elems);
    ASSERT_TRUE(it->second.signature == "(" + name + "[u_1,u_2])");
    ASSERT_TRUE(ea.coreExpressionMap.count(name) == 1);

    bool minted2 = true;
    const std::string name2 = ea.findOrMintOrOperator(elems, 2, &minted2);
    ASSERT_TRUE(!minted2);
    ASSERT_TRUE(name2 == name);
    ASSERT_EQ(ea.orCounter, counterBefore + 1);
}

// Helper: count or-category entries in the compile registry.
static int countOrEntries(const gl::ExpressionAnalyzer& ea) {
    int n = 0;
    for (const auto& [name, le] : ea.compiledExpressions) {
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
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    // Flush reductions of any pre-registered pool ors first so the count
    // delta below isolates the synthetic entry.
    ea.preMintReducedOrs();

    const std::vector<std::string> base = {
        "(zzalpha[u_1])", "!(zzbeta[u_1])", "(zzgamma[u_1])" };
    ea.findOrMintOrOperator(base, 1);
    const int before = countOrEntries(ea);

    ea.preMintReducedOrs();

    // The three reductions, in leaf order, u_-renumbered (trivially here).
    const std::vector<std::vector<std::string>> expected = {
        { "!(zzbeta[u_1])", "(zzgamma[u_1])" },
        { "(zzalpha[u_1])", "(zzgamma[u_1])" },
        { "(zzalpha[u_1])", "!(zzbeta[u_1])" } };
    for (const auto& redu : expected) {
        gl::StrSpan spans[2] = { gl::StrSpan(redu[0]), gl::StrSpan(redu[1]) };
        ASSERT_TRUE(ea.compiledOrByElements(spans, 2) != nullptr);
    }
    ASSERT_EQ(countOrEntries(ea), before + 3);

    // A list never registered stays a defined miss.
    const std::string never1 = "(zznever[u_1])";
    const std::string never2 = "(zznever2[u_1])";
    gl::StrSpan missSpans[2] = { gl::StrSpan(never1), gl::StrSpan(never2) };
    ASSERT_TRUE(ea.compiledOrByElements(missSpans, 2) == nullptr);

    ea.preMintReducedOrs();
    ASSERT_EQ(countOrEntries(ea), before + 3);
}

// preMintReducedOrs on a 4-ary or reaches the single-elimination closure
// through the worklist: 4 ternary + 6 binary reductions (dedup across
// paths), all in one call.
TEST(or_theorem_construction, premint_four_ary_worklist_closure) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    ea.preMintReducedOrs();

    const std::vector<std::string> base = {
        "(zza[u_1])", "(zzb[u_1])", "(zzc[u_1])", "(zzd[u_1])" };
    ea.findOrMintOrOperator(base, 1);
    const int before = countOrEntries(ea);

    ea.preMintReducedOrs();
    ASSERT_EQ(countOrEntries(ea), before + 10);

    // Spot-check one ternary and one binary from different elimination
    // paths land on the SAME registered binary (dedup).
    {
        gl::StrSpan t[3] = { gl::StrSpan(base[0]), gl::StrSpan(base[1]),
                             gl::StrSpan(base[2]) };
        ASSERT_TRUE(ea.compiledOrByElements(t, 3) != nullptr);
        gl::StrSpan b[2] = { gl::StrSpan(base[0]), gl::StrSpan(base[1]) };
        ASSERT_TRUE(ea.compiledOrByElements(b, 2) != nullptr);
    }
}

// flattenRegistryOrLeaves expands a nested or-child through the
// signature-to-instance substitution (arg permutation included), and
// preMintReducedOrs selects by FLATTENED leaf count — a 2-element entry
// with an or-child still pre-mints its 3 reductions, reusing the child
// where the reduction equals it.
TEST(or_theorem_construction, premint_nested_or_flattens_and_reuses_child) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    ea.preMintReducedOrs();

    const std::vector<std::string> childElems = {
        "(zzalpha[u_1])", "(zzbeta[u_2])" };
    const std::string child = ea.findOrMintOrOperator(childElems, 2);

    // Parent embeds the child with PERMUTED instance args (u_2, u_1).
    const std::vector<std::string> parentElems = {
        "(zzgamma[u_1])", "(" + child + "[u_2,u_1])" };
    const std::string parent = ea.findOrMintOrOperator(parentElems, 2);

    std::vector<std::string> leaves;
    const auto pit = ea.compiledExpressions.find(parent);
    ASSERT_TRUE(pit != ea.compiledExpressions.end());
    ea.flattenRegistryOrLeaves(pit->second, leaves);
    ASSERT_EQ(static_cast<int>(leaves.size()), 3);
    ASSERT_TRUE(leaves[0] == "(zzgamma[u_1])");
    ASSERT_TRUE(leaves[1] == "(zzalpha[u_2])");
    ASSERT_TRUE(leaves[2] == "(zzbeta[u_1])");

    const int before = countOrEntries(ea);
    ea.preMintReducedOrs();
    // minus-gamma renumbers to exactly the child's element list (reuse);
    // the other two reductions are fresh.
    ASSERT_EQ(countOrEntries(ea), before + 2);
    {
        gl::StrSpan g1[2] = { gl::StrSpan(childElems[0]),
                              gl::StrSpan(childElems[1]) };
        const std::string* hit = ea.compiledOrByElements(g1, 2);
        ASSERT_TRUE(hit != nullptr);
        ASSERT_TRUE(*hit == child);
    }
    {
        const std::string r1 = "(zzgamma[u_1])";
        const std::string r2 = "(zzbeta[u_1])";
        gl::StrSpan g2[2] = { gl::StrSpan(r1), gl::StrSpan(r2) };
        ASSERT_TRUE(ea.compiledOrByElements(g2, 2) != nullptr);
    }
    {
        const std::string r1 = "(zzgamma[u_1])";
        const std::string r2 = "(zzalpha[u_2])";
        gl::StrSpan g3[2] = { gl::StrSpan(r1), gl::StrSpan(r2) };
        ASSERT_TRUE(ea.compiledOrByElements(g3, 2) != nullptr);
    }
}

// renumberULeaves maps the surviving u_ tokens to u_1..u_m in
// first-appearance order with SIMULTANEOUS substitution (overlapping
// old/new names rewrite correctly) and returns the reduced arity.
TEST(or_theorem_construction, renumber_u_leaves_first_appearance) {
    std::vector<std::string> leaves = {
        "(zzx[u_3,u_4])", "(zzy[u_4,u_1])" };
    const int arity = gl::ExpressionAnalyzer::renumberULeaves(leaves);
    ASSERT_EQ(arity, 3);
    ASSERT_TRUE(leaves[0] == "(zzx[u_1,u_2])");
    ASSERT_TRUE(leaves[1] == "(zzy[u_2,u_3])");
}

// A pair whose source theorem is not (or no longer) in globalTheoremList —
// compression pruning or vacuity retraction — licenses nothing.
TEST(or_theorem_construction, pruned_source_is_skipped) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    const std::string totality =
        "(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])"
        "(>[9](in[9,1])(>[10](in[10,1])"
        "(>[]!(preorder[1,4,9,10])(preorder[1,4,10,9])))))";
    const std::string companion = ea.headSwitchOne(totality);
    ASSERT_TRUE(!companion.empty());

    // The source is deliberately NOT appended to globalTheoremList.
    ea.orPairsFromHeadSwitch.push_back({totality, companion});

    std::set<std::string> consumed;
    const std::vector<std::string> ors = ea.constructOrTheoremsFromPairs(consumed);

    ASSERT_EQ(static_cast<int>(ors.size()), 0);
    ASSERT_EQ(static_cast<int>(consumed.size()), 0);
    ASSERT_EQ(static_cast<int>(ea.globalTheoremList.size()), 0);
}

// forEachCompiledOr (the I-137 read fence behind the subset-exclusion
// registry scan): visits or-category entries only, and a true-returning
// visitor stops the scan after that entry.
TEST(or_theorem_construction, for_each_compiled_or_visits_and_stops) {
    gl::ExpressionAnalyzer ea(std::string("Peano"));
    const std::vector<std::string> parent = {
        "(zzfea[u_1])", "(zzfeb[u_1])" };
    const std::string name = ea.findOrMintOrOperator(parent, 1);

    bool sawMinted = false;
    bool nonOrSeen = false;
    ea.forEachCompiledOr([&](const std::string& n,
                             const gl::LogicalEntity& le) -> bool {
        if (le.category != "or") nonOrSeen = true;
        if (n == name) sawMinted = true;
        return false;
    });
    ASSERT_TRUE(sawMinted);
    ASSERT_FALSE(nonOrSeen);

    int32_t visited = 0;
    ea.forEachCompiledOr([&](const std::string&,
                             const gl::LogicalEntity&) -> bool {
        ++visited;
        return true;
    });
    ASSERT_EQ(visited, 1);
}
