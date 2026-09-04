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
/// @brief Unit tests for the cross-chapter theorem-usage graph
///        (D-276).
///
/// @details
/// Direct coverage for `gl::TheoremUsageGraph` (`clear` /
/// `addEdgesFromStack` / `reaches`) plus a behavioral `buildStack` test:
/// with a committed edge U -> T and the current chapter theorem T, a
/// `theorem`-tagged origin citing U is inadmissible and an alternative
/// origin wins; with an empty graph — or the check disabled via an empty
/// `exportCurrentChapterTheorem` — the legacy chapter-local behavior is
/// preserved and the theorem leaf is emitted.

#include "test_harness.hpp"

#include "../prover.hpp"

TEST(theorem_usage_graph, add_edges_all_citation_channels) {
    // One synthetic chapter of theorem "T" exercising every citation
    // channel plus a non-citation row that must contribute nothing.
    gl::TheoremUsageGraph g;
    std::vector<std::vector<std::string>> rows;
    rows.push_back({ "(U1)", "main", "theorem" });
    rows.push_back({ "(T)", "main", "reformulated from", "(U2)", "main" });
    rows.push_back({ "(T)", "main", "incubator back reformulation", "(U3)", "main" });
    rows.push_back({ "(T)", "main", "or theorem",
                     "(U4)", "main", "(U5)", "main" });
    rows.push_back({ "(X)", "main", "implication", "(RULE)", "main",
                     "(PREM)", "main" });   // not a commit channel
    g.addEdgesFromStack("(T)", rows);

    const std::set<std::string>& out = g.edges.at("(T)");
    ASSERT_TRUE(out.size() == 5);
    ASSERT_TRUE(out.count("(U1)") == 1);
    ASSERT_TRUE(out.count("(U2)") == 1);
    ASSERT_TRUE(out.count("(U3)") == 1);
    ASSERT_TRUE(out.count("(U4)") == 1);
    ASSERT_TRUE(out.count("(U5)") == 1);
    ASSERT_TRUE(out.count("(RULE)") == 0);
}

TEST(theorem_usage_graph, add_edges_unions_across_chapter_files) {
    // Induction triads call addEdgesFromStack once per file under one
    // theorem node — edges must union, not replace.
    gl::TheoremUsageGraph g;
    g.addEdgesFromStack("(T)", { { "(U1)", "main", "theorem" } });
    g.addEdgesFromStack("(T)", { { "(U2)", "main", "theorem" } });
    ASSERT_TRUE(g.edges.at("(T)").size() == 2);
}

TEST(theorem_usage_graph, reaches_self_direct_transitive_negative) {
    gl::TheoremUsageGraph g;
    g.addEdgesFromStack("(A)", { { "(B)", "main", "theorem" } });
    g.addEdgesFromStack("(B)", { { "(C)", "main", "theorem" } });

    ASSERT_TRUE(g.reaches("(A)", "(A)"));    // self — length-1 cycle probe
    ASSERT_TRUE(g.reaches("(X)", "(X)"));    // self, node without edges
    ASSERT_TRUE(g.reaches("(A)", "(B)"));    // direct
    ASSERT_TRUE(g.reaches("(A)", "(C)"));    // transitive
    ASSERT_FALSE(g.reaches("(C)", "(A)"));   // against edge direction
    ASSERT_FALSE(g.reaches("(B)", "(A)"));
    ASSERT_FALSE(g.reaches("(A)", "(Z)"));   // unknown target
}

TEST(theorem_usage_graph, clear_drops_all_edges) {
    gl::TheoremUsageGraph g;
    g.addEdgesFromStack("(A)", { { "(B)", "main", "theorem" } });
    ASSERT_TRUE(!g.edges.empty());
    g.clear();
    ASSERT_TRUE(g.edges.empty());
    ASSERT_FALSE(g.reaches("(A)", "(B)"));
}

TEST(theorem_usage_graph, buildstack_rejects_cycle_closing_theorem_leaf) {
    // Behavioral test for the cross-chapter admissibility check in
    // `visualizer.cpp::ExpressionAnalyzer::buildStack`, modeled on the
    // originMap synthesis of equi_reshuffle.buildstack_path_refcount_*.
    //
    //   Head A has two implication candidates (D-49 keeps insertion order):
    //     C1: deps = {U}  — U is a proven theorem whose chapter (already
    //                       exported) reaches the current theorem T
    //     C2: deps = {F}  — broadcast foundation, always acyclic
    //   U's sole origin is a `theorem` leaf (empty deps).
    //
    // Phase 1 — check ARMED (current theorem "T", committed edge U -> T):
    //   C1 recurses into U; the theorem leaf is inadmissible
    //   (reaches(U, T)); U has no other origin and no __contradiction__
    //   sibling, so the degraded emit returns false; C1 rolls back and C2
    //   wins. The final stack cites F and never U.
    //
    // Phase 2 — empty graph, same current theorem: the theorem leaf is
    //   admissible and C1 wins (legacy behavior).
    //
    // Phase 3 — edge present but current theorem EMPTY: check disabled,
    //   C1 wins (findEnds / direct-caller behavior).
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    gl::Memory& mb = ana.body;
    const gl::ExpressionWithValidity A("A", "main");
    const gl::ExpressionWithValidity U("U", "main");
    const gl::ExpressionWithValidity F("FOUNDATION", "main");
    const int64_t kA = gl::mintOriginKey(mb.originInterner, A.original, A.validityName);
    const int64_t kU = gl::mintOriginKey(mb.originInterner, U.original, U.validityName);
    const int64_t kF = gl::mintOriginKey(mb.originInterner, F.original, F.validityName);
    mb.exprOriginMap.assignRun(kA, std::vector<gl::IdOrigin>{
        gl::IdOrigin(gl::OriginTag::implication, {kU}),   // C1 — theorem-leaf dep
        gl::IdOrigin(gl::OriginTag::implication, {kF}),   // C2 — foundation dep
    });
    mb.exprOriginMap.assignRun(kU, std::vector<gl::IdOrigin>{
        gl::IdOrigin(gl::OriginTag::theorem, {}),         // the citation leaf
    });
    mb.exprOriginMap.assignRun(kF, std::vector<gl::IdOrigin>{
        gl::IdOrigin(gl::OriginTag::broadcast, {}),       // early-return, no row
    });

    auto runBuild = [&]() {
        std::vector<std::vector<std::string>> stack;
        std::set<gl::ExpressionWithValidity> covered;
        ana.buildStack(mb, A, stack, covered);
        return stack;
    };
    auto hasRow = [](const std::vector<std::vector<std::string>>& stack,
                     const std::string& head, const std::string& dep0) {
        for (const auto& row : stack) {
            if (row.size() >= 4 && row[0] == head && row[3] == dep0) return true;
            if (dep0.empty() && row.size() >= 3 && row[0] == head) return true;
        }
        return false;
    };

    // Phase 1 — armed: U's chapter reaches T, citing U must lose.
    ana.exportCurrentChapterTheorem = "T";
    ana.exportTheoremUsage.clear();
    ana.exportTheoremUsage.addEdgesFromStack("U", { { "T", "main", "theorem" } });
    std::vector<std::vector<std::string>> armed = runBuild();
    ASSERT_TRUE(hasRow(armed, "A", "FOUNDATION"));   // C2 chosen
    ASSERT_FALSE(hasRow(armed, "A", "U"));           // C1 rejected
    ASSERT_FALSE(hasRow(armed, "U", ""));            // no U row at all

    // Phase 2 — empty graph: theorem leaf admissible, C1 wins.
    ana.exportTheoremUsage.clear();
    std::vector<std::vector<std::string>> emptyGraph = runBuild();
    ASSERT_TRUE(hasRow(emptyGraph, "A", "U"));
    ASSERT_TRUE(hasRow(emptyGraph, "U", ""));        // the theorem leaf row

    // Phase 3 — edge present, check disabled by empty current theorem.
    ana.exportTheoremUsage.addEdgesFromStack("U", { { "T", "main", "theorem" } });
    ana.exportCurrentChapterTheorem.clear();
    std::vector<std::vector<std::string>> disabled = runBuild();
    ASSERT_TRUE(hasRow(disabled, "A", "U"));
    ASSERT_TRUE(hasRow(disabled, "U", ""));
}

TEST(theorem_usage_graph, buildstack_contradiction_fallback_requires_twin_record) {
    // Behavioral test for the D-51 fallback resolution guard
    // (`contradictionLbHoldsRecord`): the post-candidate-loop fallback may
    // switch into a `__contradiction__` twin ONLY when the twin actually
    // holds an origin record for the proved expression.
    //
    //   Head (P)'s single candidate cites U, whose sole origin is a `theorem`
    //   leaf made inadmissible by an armed usage graph (U's chapter reaches
    //   the current theorem T) — so (P)'s candidate loop always fails and the
    //   walk reaches the post-loop contradiction fallback. A twin child keyed
    //   "__contradiction__!(P)" hangs off the root LB.
    //
    // Phase 1 — twin WITHOUT a record for (P): the guard refuses the switch;
    //   the walk survives to the degraded front()-emit ((P) citing U) and
    //   returns false. (Pre-guard behavior: switch in, twin entry finds no
    //   origin, the self-guard blocks re-entry, and the no-origin assert
    //   kills the process.)
    //
    // Phase 2 — twin WITH a record for (P) (discharge shape citing a
    //   broadcast-founded fact): the guard admits the switch and the walk
    //   resolves inside the twin, returning true with the twin's row cited.
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    gl::Memory& mb = ana.body;
    const gl::ExpressionWithValidity P("(P)", "main");
    const int64_t kP = gl::mintOriginKey(mb.originInterner, P.original, P.validityName);
    const int64_t kU = gl::mintOriginKey(mb.originInterner, "U", "main");
    mb.exprOriginMap.assignRun(kP, std::vector<gl::IdOrigin>{
        gl::IdOrigin(gl::OriginTag::implication, {kU}),
    });
    mb.exprOriginMap.assignRun(kU, std::vector<gl::IdOrigin>{
        gl::IdOrigin(gl::OriginTag::theorem, {}),
    });
    ana.exportCurrentChapterTheorem = "T";
    ana.exportTheoremUsage.clear();
    ana.exportTheoremUsage.addEdgesFromStack("U", { { "T", "main", "theorem" } });

    gl::Memory* twin = ana.lbStore.create<gl::Memory>();
    twin->parentMemory = &mb;
    twin->setExprKey("__contradiction__!(P)");
    ana.simpleMapStore.linkChild(&mb, "__contradiction__!(P)", twin);

    auto runBuild = [&](bool& ok) {
        std::vector<std::vector<std::string>> stack;
        std::set<gl::ExpressionWithValidity> covered;
        ok = ana.buildStack(mb, P, stack, covered);
        return stack;
    };

    // Phase 1 — no record in the twin: the guard refuses, the walk reaches
    // the degraded front()-emit instead of the twin's no-origin assert.
    bool ok1 = true;
    std::vector<std::vector<std::string>> degraded = runBuild(ok1);
    ASSERT_FALSE(ok1);
    ASSERT_TRUE(!degraded.empty());
    ASSERT_EQ(degraded.front()[0], std::string("(P)"));
    ASSERT_EQ(degraded.front()[2], std::string("implication"));

    // Phase 2 — record present: the guard admits the switch, resolution
    // happens inside the twin ((P) row carries the contradiction tag).
    const int64_t kPt = gl::mintOriginKey(twin->originInterner, P.original, P.validityName);
    const int64_t kFt = gl::mintOriginKey(twin->originInterner, "FOUNDATION", "main");
    twin->exprOriginMap.assignRun(kPt, std::vector<gl::IdOrigin>{
        gl::IdOrigin(gl::OriginTag::contradiction, {kFt}),
    });
    twin->exprOriginMap.assignRun(kFt, std::vector<gl::IdOrigin>{
        gl::IdOrigin(gl::OriginTag::broadcast, {}),
    });
    bool ok2 = false;
    std::vector<std::vector<std::string>> resolved = runBuild(ok2);
    ASSERT_TRUE(ok2);
    bool twinRow = false;
    for (const auto& row : resolved) {
        if (row.size() >= 4 && row[0] == "(P)" && row[2] == "contradiction"
            && row[3] == "FOUNDATION") {
            twinRow = true;
        }
    }
    ASSERT_TRUE(twinRow);

    ana.destroyGrid();  // deletes the twin child + clears the grid links
}


// ---------------------------------------------------------------------------
// buildStack — the last-resort fallback is skipped when its rows cannot survive
// ---------------------------------------------------------------------------
//
// A recursion from a parent's candidate loop has every row it emitted removed
// by that parent's `stack.resize` the instant it returns false, so running the
// degraded front()-emit and walking its whole dependency subtree there is work
// whose only consumer is the truncation. `rowsSurviveOnFailure = false` says
// exactly that, and the fallback is skipped.
//
// Same origin shape as the contradiction-guard test: (P) <- implication <- U,
// and U's only origin is a `theorem` leaf citing a theorem that already reaches
// the current chapter, so U is inadmissible, (P)'s candidate fails, and the
// walk reaches (P)'s own last-resort fallback.

TEST(theorem_usage_graph, fallback_skipped_when_rows_cannot_survive) {
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    gl::Memory& mb = ana.body;
    const gl::ExpressionWithValidity P("(P)", "main");
    const int64_t kP = gl::mintOriginKey(mb.originInterner, P.original, P.validityName);
    const int64_t kU = gl::mintOriginKey(mb.originInterner, "U", "main");
    mb.exprOriginMap.assignRun(kP, std::vector<gl::IdOrigin>{
        gl::IdOrigin(gl::OriginTag::implication, {kU}),
    });
    mb.exprOriginMap.assignRun(kU, std::vector<gl::IdOrigin>{
        gl::IdOrigin(gl::OriginTag::theorem, {}),
    });
    ana.exportCurrentChapterTheorem = "T";
    ana.exportTheoremUsage.clear();
    ana.exportTheoremUsage.addEdgesFromStack("U", { { "T", "main", "theorem" } });

    // Default (rows survive): the degraded row is emitted, verdict false.
    {
        std::vector<std::vector<std::string>> stack;
        std::set<gl::ExpressionWithValidity> covered;
        const bool ok = ana.buildStack(mb, P, stack, covered);
        ASSERT_FALSE(ok);
        ASSERT_TRUE(!stack.empty());
        ASSERT_EQ(stack.front()[0], std::string("(P)"));
        ASSERT_EQ(stack.front()[2], std::string("implication"));
    }

    // Rows cannot survive: same verdict, no row, and `covered` untouched —
    // the caller would have discarded all three anyway.
    {
        std::vector<std::vector<std::string>> stack;
        std::set<gl::ExpressionWithValidity> covered;
        const bool ok = ana.buildStack(mb, P, stack, covered,
                                       /*rowsSurviveOnFailure=*/false);
        ASSERT_FALSE(ok);
        ASSERT_TRUE(stack.empty());
        ASSERT_TRUE(covered.empty());
    }

    // A succeeding walk is unaffected by the flag: give U a broadcast origin
    // so (P)'s candidate closes, and both flag values return true with the
    // same single row.
    mb.exprOriginMap.assignRun(kU, std::vector<gl::IdOrigin>{
        gl::IdOrigin(gl::OriginTag::broadcast, {}),
    });
    for (int pass = 0; pass < 2; ++pass) {
        std::vector<std::vector<std::string>> stack;
        std::set<gl::ExpressionWithValidity> covered;
        const bool ok = ana.buildStack(mb, P, stack, covered,
                                       /*rowsSurviveOnFailure=*/pass == 0);
        ASSERT_TRUE(ok);
        ASSERT_EQ(stack.size(), 1u);
        ASSERT_EQ(stack.front()[0], std::string("(P)"));
    }

    ana.destroyGrid();
}
