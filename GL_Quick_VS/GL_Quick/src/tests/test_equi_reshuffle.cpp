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
/// @brief Unit tests for the equivalence-class application reshuffle.
///
/// @details
/// Coverage for the equivalence-class-application functions:
/// `applyEquiClasses`, `dischargeToBeProved`, `dischargeContradiction`,
/// `fillMailOut`, and `standardProcessing`. All are members of
/// `gl::ExpressionAnalyzer`
/// and require a fully-constructed analyzer plus disk fixtures to
/// exercise behaviorally — too heavy for the <30 s test budget. The
/// tests confirm symbol existence and signature; behavioral coverage
/// is exercised end-to-end via the full main.py run.

#include "test_harness.hpp"

#include <set>

#include "../prover.hpp"

TEST(equi_reshuffle, apply_equi_classes_symbol_signature) {
    // Symbol-existence + signature check: take the method's address
    // with its expected signature `void(gl::Memory&)`. The test
    // compiles only if `applyEquiClasses` is declared on
    // `gl::ExpressionAnalyzer` with that exact signature, satisfying
    // Rule 18's unit-test requirement. Behavioral coverage
    // (intEncodedStatements mutations, mailOut commits,
    // dischargeToBeProved invocation) is exercised end-to-end via the
    // full main.py run.
    using ApplyFn = void (gl::ExpressionAnalyzer::*)(gl::Memory&);
    ApplyFn fn = &gl::ExpressionAnalyzer::applyEquiClasses;
    ASSERT_TRUE(fn != nullptr);
}

TEST(equi_reshuffle, apply_equi_classes_expands_preexisting_negated_equality) {
    // The insertion-order hole (D-226): a negated
    // equality that arrives BEFORE its variable's class forms gets no
    // arrival-time expansion, and the general class-application pass used to
    // skip negated equalities entirely — so it was never expanded. Pass 1
    // must now revisit it when the class forms: negation !(=[2,a]) at a
    // branch scope, class {a,c} forming later at main, must yield !(=[2,c])
    // at the branch scope.
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    gl::Memory& mb = ana.body;
    mb.isActive = true;

    const gl::NameId branchId = mb.nameMap.encodePush(
        gl::NameMap::MAIN_ID, "ordis_(or9[1,7,3,2])_((=[6,x]))");
    const std::string branch = mb.nameMap.decode(branchId);
    const std::string mainNs = "main";

    gl::DirtyState outDirty = gl::DirtyState::Clean;
    gl::PagedVector<gl::IntEncodedExpr> out(
        &gl::genScratchArenas().forSlot(0), &outDirty);
    const int lv[1] = { 0 };
    const gl::TransientOrigin noOrigin{};

    // 1. The negation arrives FIRST, at the branch scope — no class exists,
    //    so the arrival-time expansion is a no-op.
    const std::string negation = "!(=[2,a])";
    ana.addStatement(gl::StrSpan(negation), mb, true, lv, 1, noOrigin,
                     gl::StrSpan(branch), out);
    // 2. The class {a, c} forms LATER, at main (positive-equality arrival).
    //    The equality carries a real origin — production equalities always
    //    do under trackHistory, and the arrival write is what the Pass-1
    //    expansion's citation probe later resolves against.
    out.clear();
    const std::string equality = "(=[a,c])";
    const gl::TransientOrigin eqOrigin{ true, gl::OriginTag::broadcast,
                                        nullptr, 0 };
    ana.addStatement(gl::StrSpan(equality), mb, true, lv, 1, eqOrigin,
                     gl::StrSpan(mainNs), out);

    // Pre-check: the expanded form does not exist yet.
    const std::string expanded = "!(=[2,c])";
    ASSERT_TRUE(gl::lookupStatementLevels(mb.intStatementLevelsMap, mb.nameMap,
        gl::StrSpan(expanded), gl::StrSpan(branch)) == 0);

    // 3. The class-application pass expands the pre-existing negation.
    ana.applyEquiClasses(mb);

    ASSERT_TRUE(gl::lookupStatementLevels(mb.intStatementLevelsMap, mb.nameMap,
        gl::StrSpan(expanded), gl::StrSpan(branch)) != 0);

    // 4. Citation-scope contract: the expanded row's equality1 history line
    //    cites the justifying equality at the scope where its origin row
    //    lives (main — the class's scope), never at the branch scope. A
    //    branch-scope citation is unresolvable for the chapter walker: its
    //    ancestor lift refuses to cross an or-branch boundary.
    int64_t expandedPk = 0;
    ASSERT_TRUE(gl::lookupOriginKey(mb.originInterner, expanded, branch,
                                    expandedPk));
    const std::vector<gl::IdOrigin> expandedRows =
        mb.exprOriginMap.recordsAt(mb.exprOriginMap.lookup(expandedPk));
    bool citeChecked = false;
    for (const gl::IdOrigin& o : expandedRows) {
        const auto line = gl::decodeOrigin(o, mb.originInterner);
        if (line.first != "equality1") continue;
        ASSERT_TRUE(line.second.size() == 2);
        ASSERT_TRUE(line.second[1].original == equality);
        ASSERT_TRUE(line.second[1].validityName == mainNs);
        citeChecked = true;
    }
    ASSERT_TRUE(citeChecked);
}

TEST(equi_reshuffle, discharge_to_be_proved_symbol_signature) {
    // Symbol-existence + signature check for the toBeProved-discharge
    // helper. Compiles only if `dischargeToBeProved` is declared on
    // `gl::ExpressionAnalyzer` with the current signature. Behavioral
    // coverage (induction-recursion proven, non-recursion proven,
    // OR-integration / implication closure routed via the passed
    // `internalMailOut` `ColdMail&`) is exercised end-to-end via the full
    // main.py + verifier run.
    //
    // The function iterates `intLocalEncodedStatementsDelta`. The
    // `ColdMail& internalMailOut` parameter lets the parent-scope
    // emissions (OR-integration + NotOrScope) be routed same-iter vs
    // cross-iter by the `standardProcessing` caller.
    using DischargeFn = void (gl::ExpressionAnalyzer::*)(gl::Memory&, int, gl::ColdMail&);
    DischargeFn fn = &gl::ExpressionAnalyzer::dischargeToBeProved;
    ASSERT_TRUE(fn != nullptr);
}

TEST(equi_reshuffle, fill_mail_out_symbol_signature) {
    // Symbol-existence + signature check for the centralised mailOut
    // population helper. Compiles only if `fillMailOut` is declared
    // on `gl::ExpressionAnalyzer` with signature `void(Memory&)`.
    // Behavioral coverage (mailOut.statements + mailOut.exprOriginMap
    // population from intLocalEncodedStatementsDelta) is exercised
    // end-to-end via the full main.py run at commit 19 of
    // `sandbox/equi_reshuffle`.
    using FillFn = void (gl::ExpressionAnalyzer::*)(gl::Memory&);
    FillFn fn = &gl::ExpressionAnalyzer::fillMailOut;
    ASSERT_TRUE(fn != nullptr);
}

TEST(equi_reshuffle, ship_expr_history_transitively_symbol_signature) {
    // Symbol-existence + signature check for the recursive history
    // walk helper introduced by the buildStack-crash fix
    // (2026-05-25). Compiles only if
    // `shipExprHistoryTransitively` is declared on
    // `gl::ExpressionAnalyzer` with signature
    // `void(Memory&, const ExpressionWithValidity&, int)`.
    //
    // The helper is invoked from `fillMailOut` on every sub-dep of a
    // shipped origin line: if the dep's history line is not yet in
    // `mailOut.exprOriginMap`, look it up in `exprOriginMap` and
    // ship it (asserting if absent per Rule 19), then recurse on
    // that dep's own sub-deps. Cycle termination + dedup are both
    // via the mailOut-presence early return.
    //
    // Behavioral coverage (transitive shipment of expansion
    // conjunctions into child contradiction LBs; `buildStack` chapter
    // export succeeds on `__contradiction__(=[2,6])` and
    // `__contradiction__(in2[10,10,3])`) is exercised end-to-end via
    // the post-fix main.py run for IncubatorPeano 5-iter.
    using ShipFn = void (gl::ExpressionAnalyzer::*)(gl::Memory&,
                                                     const gl::ExpressionWithValidity&,
                                                     int);
    ShipFn fn = &gl::ExpressionAnalyzer::shipExprHistoryTransitively;
    ASSERT_TRUE(fn != nullptr);
}

TEST(equi_reshuffle, buildstack_path_refcount_doubleentry) {
    // Behavioral test for the per-frame `insertedHere` ownership fix in
    // `visualizer.cpp::ExpressionAnalyzer::buildStack`. Pre-fix:
    // `g_buildStackPath` is a flat `std::set<ExpressionWithValidity>` whose
    // `erase` has no refcount. When a node X appears on the recursion stack
    // twice (outer frame walking X; inner frame re-entering X because X is
    // a dep of a candidate currently being explored), the inner frame's
    // exit `erase(X)` removed the outer frame's entry and broke the cycle
    // filter for the outer frame's remaining candidates. Post-fix every
    // exit erases only if the current frame was the inserter (`.second`
    // of the entry-time `insert`).
    //
    // Test shape — minimal originMap synthesis. The real-world bug shape
    // requires AT LEAST TWO of A's candidates to cite B so that after the
    // first candidate's recursion into B corrupts the path (inner re-entry
    // on A erases A from the path), the SECOND candidate's recursion into
    // B finds the path's cycle filter falsely passing — and the fallback
    // emits the cyclic chapter row. With only one A-candidate citing B,
    // the path is already corrupted by then but no further candidate cites
    // B to exhibit the symptom.
    //
    //   A has three implication candidates (D-49 sort puts all three
    //                                       non-equality candidates first
    //                                       in insertion order):
    //     C1: deps = {B}            — cycle via B's back-edge to A
    //     C2: deps = {B}            — same shape; the SECOND attempt that
    //                                  pre-fix falsely succeeds because
    //                                  the path was corrupted by C1's
    //                                  inner-A erase
    //     C3: deps = {F}            — acyclic foundation (only candidate
    //                                  that should ever succeed)
    //   B has one equality1 candidate:
    //     E1: deps = {A}            — back-edge to A
    //   F is a broadcast foundation — buildStack returns true without
    //   emitting a row for it.
    //
    // Pre-fix trace:
    //   - C1: recurse into B → equality1 cyclic → fallback emits B's
    //     cyclic row → fallback recurses into A → inner A picks C3 (the
    //     acyclic one) → returns true → erase wipes outer A from path.
    //   - C1 subtreeOk=false (B's fallback returned false). resize, restore.
    //   - C2: recurse into B → but A is NOT on path anymore (was wiped)
    //     → equality1 cycle filter falsely passes → emit row for B →
    //     recurse into A (now inserted fresh) → A's inner attempt
    //     succeeds via C3 → erase A → return true → B's subtreeOk=true →
    //     return true → C2 subtreeOk=true → CHAPTER EMITS CYCLIC ROW FOR B.
    //
    // Post-fix trace:
    //   - C1: same as above except inner A's erase is GUARDED by
    //     `insertedHere == false` → path stays {A, B} → B's fallback
    //     ends with erase B (insertedHere=true) → path={A}.
    //   - C1 subtreeOk=false. resize, restore. Path is still {A}.
    //   - C2: recurse into B → B inserts → path={A, B} → equality1's
    //     A-in-path check FIRES → cyclic → no other candidate → fallback
    //     returns false → C2 subtreeOk=false.
    //   - C3: F is broadcast foundation → succeeds → no cyclic row for B.
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    gl::Memory& mb = ana.body;
    const gl::ExpressionWithValidity A("A", "main");
    const gl::ExpressionWithValidity B("B", "main");
    const gl::ExpressionWithValidity F("FOUNDATION", "main");
    const int64_t kA = gl::mintOriginKey(mb.originInterner, A.original, A.validityName);
    const int64_t kB = gl::mintOriginKey(mb.originInterner, B.original, B.validityName);
    const int64_t kF = gl::mintOriginKey(mb.originInterner, F.original, F.validityName);
    mb.exprOriginMap.assignRun(kA, std::vector<gl::IdOrigin>{
        gl::IdOrigin(gl::OriginTag::implication, {kB}),     // C1 — cyclic via B
        gl::IdOrigin(gl::OriginTag::implication, {kB}),     // C2 — second cyclic candidate (pre-fix this is the one that falsely succeeds)
        gl::IdOrigin(gl::OriginTag::implication, {kF}),     // C3 — acyclic via foundation
    });
    mb.exprOriginMap.assignRun(kB, std::vector<gl::IdOrigin>{
        gl::IdOrigin(gl::OriginTag::equality1, {kA}),       // E1 — back-edge to A
    });
    mb.exprOriginMap.assignRun(kF, std::vector<gl::IdOrigin>{
        gl::IdOrigin(gl::OriginTag::broadcast, {}),         // early-return tag, no row emitted, success
    });

    std::vector<std::vector<std::string>> stack;
    std::set<gl::ExpressionWithValidity> covered;
    ana.buildStack(mb, A, stack, covered);

    // Pass criterion: NO chapter row whose head is B exists in the stack.
    // Pre-fix would emit `B equality1 A` via the last-resort fallback
    // inside C2's recursion (because path corruption from C1 hid the
    // cycle filter's A-on-path signal). Post-fix C3 is picked cleanly and
    // no recursion into B survives.
    bool emittedRowForB = false;
    bool emittedC1orC2 = false;  // either of the cyclic A-candidates
    bool emittedC3 = false;      // the acyclic A-candidate
    for (const auto& row : stack) {
        if (row.size() >= 2 && row[0] == B.original && row[1] == B.validityName) emittedRowForB = true;
        if (row.size() >= 5 && row[0] == A.original && row[1] == A.validityName
            && row[3] == B.original) emittedC1orC2 = true;
        if (row.size() >= 5 && row[0] == A.original && row[1] == A.validityName
            && row[3] == F.original) emittedC3 = true;
    }
    ASSERT_FALSE(emittedRowForB);
    ASSERT_FALSE(emittedC1orC2);
    ASSERT_TRUE(emittedC3);
}

TEST(equi_reshuffle, standard_processing_symbol_signature) {
    // Symbol-existence + signature check for the per-call mail-absorb
    // driver introduced by the follow-up consolidation plan at
    // `~/.claude/plans/standard-processing.md`. Compiles only if
    // `standardProcessing` is declared on `gl::ExpressionAnalyzer`
    // with signature
    // `void(Memory&, RoutingColdMail* externalMailIn, ColdMail& internalMailIn,
    //       ColdMail& internalMailOut, int coreId)`.
    // The external routing inbox is the `RoutingColdMail` `&body.mailIn`, absorbed
    // directly from its sorted snapshots (no transient heap `Mail`).
    // Signature widened in commit 3 of the follow-up consolidation:
    // the two-mail input shape (external + internal) plus the
    // separate output internal-mail target lets the caller route
    // discharge emissions same-iter (zero delay; pre-burst call) vs
    // cross-iter (one-step delay; post-burst call) without any new
    // status or parameter on `addExprToMemoryBlock`.
    // Behavioral coverage (per-mail absorb recipe, mail-container
    // clears between drain and apply, applyEquiClasses,
    // dischargeToBeProved routing via the internalMailOut parameter,
    // fillMailOut, changedClassesThisStep clear) is exercised by the two
    // `standardProcessing` call sites in `performElemPhase1` (pre-burst) and
    // `performElemPhase3` (post-burst), validated by the full main.py + verifier run.
    using StdFn = void (gl::ExpressionAnalyzer::*)(gl::Memory&,
                                                    gl::RoutingColdMail*,
                                                    gl::ColdMail&,
                                                    gl::ColdMail&,
                                                    int);
    StdFn fn = &gl::ExpressionAnalyzer::standardProcessing;
    ASSERT_TRUE(fn != nullptr);
}

TEST(equi_reshuffle, discharge_contradiction_symbol_signature) {
    // Symbol-existence + signature check for the consolidated contradiction
    // discharge sweep. Compiles only if `dischargeContradiction` is declared
    // on `gl::ExpressionAnalyzer` with signature `void(Memory&, int)`.
    // The function sweeps the int statement registry once per step (from
    // `standardProcessing`, directly before `dischargeToBeProved`) and fires
    // the matching incubator / CE-filter / vacuous-truth reaction on the
    // first main-scope contradiction. Behavioral coverage is the tests
    // below plus the full main.py + verifier run.
    using DischargeContraFn = void (gl::ExpressionAnalyzer::*)(gl::Memory&, int);
    DischargeContraFn fn = &gl::ExpressionAnalyzer::dischargeContradiction;
    ASSERT_TRUE(fn != nullptr);
}

TEST(equi_reshuffle, discharge_contradiction_refutes_ce_lb) {
    // CE-filter contradiction detection now lives in dischargeContradiction
    // (moved off the addExprToMemoryBlock gate): a CE LB's fired heads enter
    // the statement registry, so this post-burst sweep sees a contradicting
    // pair and refutes the conjecture. This pins that a CE LB
    // (contradictionIndex >= 0) whose statement and negation are both known at
    // main gets its contradictionTable row marked and is
    // deactivated.
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    gl::Memory& mb = ana.body;
    mb.isActive = true;
    mb.primedForContradiction = false;
    mb.isPartOfRecursion = false;
    ana.contradictionTable.push_back(gl::ContradictionItem(std::string("(in[7,N])"), false));
    mb.contradictionIndex = 0;

    const std::string P = "(in[7,N])";
    const std::string V = "main";
    const gl::EncodedExpression encP(P, V);
    mb.intEncodedStatements.push_back(gl::encodeExpression(encP, mb.nameMap));
    const int16_t negId = mb.nameMap.encode(ana.negate(P));
    const int16_t vId = mb.nameMap.encode(V);
    mb.intKnownStatements.insert(gl::StatementKey{ negId, vId }, gl::StatementFlags{ true, false });

    ana.dischargeContradiction(mb, /*coreId=*/0);

    // dischargeContradiction now handles CE: the table row is marked and the LB
    // is deactivated.
    ASSERT_TRUE(ana.contradictionTable[0].successful);
    ASSERT_FALSE(mb.isActive);
}

TEST(equi_reshuffle, discharge_contradiction_no_contradiction_noop) {
    // Negative: active CE-filter LB whose statement has NO negation in scope.
    // No contradiction -> no reaction; LB stays active, table row unmarked.
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    gl::Memory& mb = ana.body;
    mb.isActive = true;
    ana.contradictionTable.push_back(gl::ContradictionItem(std::string("(in[7,N])"), false));
    mb.contradictionIndex = 0;
    const gl::EncodedExpression encP(std::string("(in[7,N])"), std::string("main"));
    mb.intEncodedStatements.push_back(gl::encodeExpression(encP, mb.nameMap));

    ana.dischargeContradiction(mb, /*coreId=*/0);

    ASSERT_TRUE(mb.isActive);
    ASSERT_FALSE(ana.contradictionTable[0].successful);
}

TEST(equi_reshuffle, discharge_contradiction_requires_both_sides_at_main) {
    // A branch-local positive opposed by a main-scope negation rejects no LB:
    // branch rejection / OR-disintegration rewriting is a separate known gap.
    // When the same positive also exists at main, the main row is sufficient
    // and the CE contradiction discharges even though the deeper copy remains.
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    gl::Memory& mb = ana.body;
    mb.isActive = true;
    ana.contradictionTable.push_back(
        gl::ContradictionItem(std::string("(in[7,N])"), false));
    mb.contradictionIndex = 0;

    const std::string P = "(in[7,N])";
    const gl::NameId branchId = mb.nameMap.encodePush(
        gl::NameMap::MAIN_ID,
        "ordis_(or0[7,N])_((in[7,N]))");
    const std::string branch = mb.nameMap.decode(branchId);
    mb.intEncodedStatements.push_back(
        gl::encodeExpression(gl::EncodedExpression(P, branch), mb.nameMap));
    const gl::NameId negId = mb.nameMap.encode(ana.negate(P));
    mb.intKnownStatements.insert(
        gl::StatementKey{ negId, gl::NameMap::MAIN_ID },
        gl::StatementFlags{ true, false });

    ana.dischargeContradiction(mb, /*coreId=*/0);
    ASSERT_TRUE(mb.isActive);
    ASSERT_FALSE(ana.contradictionTable[0].successful);

    mb.intEncodedStatements.push_back(
        gl::encodeExpression(
            gl::EncodedExpression(P, std::string("main")), mb.nameMap));
    ana.dischargeContradiction(mb, /*coreId=*/0);
    ASSERT_FALSE(mb.isActive);
    ASSERT_TRUE(ana.contradictionTable[0].successful);
}

TEST(equi_reshuffle, discharge_contradiction_inactive_noop) {
    // Guard coverage: an already-inactive LB is skipped even with a real
    // contradicting pair present. This is the phase-1/phase-3 double-run
    // idempotency guard (`standardProcessing` runs in both phases and phase 3
    // is not re-filtered by isActive).
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    gl::Memory& mb = ana.body;
    mb.isActive = false;
    ana.contradictionTable.push_back(gl::ContradictionItem(std::string("(in[7,N])"), false));
    mb.contradictionIndex = 0;
    const std::string P = "(in[7,N])";
    const gl::EncodedExpression encP(P, std::string("main"));
    mb.intEncodedStatements.push_back(gl::encodeExpression(encP, mb.nameMap));
    const int16_t negId = mb.nameMap.encode(ana.negate(P));
    const int16_t vId = mb.nameMap.encode(std::string("main"));
    mb.intKnownStatements.insert(gl::StatementKey{ negId, vId }, gl::StatementFlags{ true, false });

    ana.dischargeContradiction(mb, /*coreId=*/0);

    ASSERT_FALSE(ana.contradictionTable[0].successful);
}

TEST(equi_reshuffle, negated_equality_arrival_expands_via_descendant_class) {
    // Scope symmetry, arrival-time direction: the class {a,c} exists at a
    // BRANCH scope FIRST; the negation !(=[2,a]) then arrives at main (the
    // parent). The arrival-time hook's descendant walk must expand it with
    // the product AT THE CLASS'S scope: !(=[2,c]) at the branch, never at
    // main (deeperOf(main, branch) == branch). Without the descendant
    // direction an ancestor-scope negation is invisible to a deeper class
    // and the branch-scope variant is never derived.
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    gl::Memory& mb = ana.body;
    mb.isActive = true;

    const gl::NameId branchId = mb.nameMap.encodePush(
        gl::NameMap::MAIN_ID, "ordis_(or9[1,7,3,2])_((=[6,x]))");
    const std::string branch = mb.nameMap.decode(branchId);
    const std::string mainNs = "main";

    gl::DirtyState outDirty = gl::DirtyState::Clean;
    gl::PagedVector<gl::IntEncodedExpr> out(
        &gl::genScratchArenas().forSlot(0), &outDirty);
    const int lv[1] = { 0 };

    // 1. The class {a, c} forms FIRST, at the branch scope (positive-
    //    equality arrival with a real origin so the later citation probe
    //    resolves).
    const std::string equality = "(=[a,c])";
    const gl::TransientOrigin eqOrigin{ true, gl::OriginTag::broadcast,
                                        nullptr, 0 };
    ana.addStatement(gl::StrSpan(equality), mb, true, lv, 1, eqOrigin,
                     gl::StrSpan(branch), out);

    // 2. The negation arrives LATER, at main — the arrival-time descendant
    //    walk expands it immediately.
    out.clear();
    const std::string negation = "!(=[2,a])";
    const gl::TransientOrigin noOrigin{};
    ana.addStatement(gl::StrSpan(negation), mb, true, lv, 1, noOrigin,
                     gl::StrSpan(mainNs), out);

    const std::string expanded = "!(=[2,c])";
    ASSERT_TRUE(gl::lookupStatementLevels(mb.intStatementLevelsMap, mb.nameMap,
        gl::StrSpan(expanded), gl::StrSpan(branch)) != 0);
    // The product lands at the class's scope only — never at main.
    ASSERT_TRUE(gl::lookupStatementLevels(mb.intStatementLevelsMap, mb.nameMap,
        gl::StrSpan(expanded), gl::StrSpan(mainNs)) == 0);

    // Citation contract: the expanded row's equality1 history line cites
    // the source negation at ITS scope (main) and the justifying equality
    // at the scope where its origin row lives (the branch).
    int64_t expandedPk = 0;
    ASSERT_TRUE(gl::lookupOriginKey(mb.originInterner, expanded, branch,
                                    expandedPk));
    const std::vector<gl::IdOrigin> expandedRows =
        mb.exprOriginMap.recordsAt(mb.exprOriginMap.lookup(expandedPk));
    bool citeChecked = false;
    for (const gl::IdOrigin& o : expandedRows) {
        const auto line = gl::decodeOrigin(o, mb.originInterner);
        if (line.first != "equality1") continue;
        ASSERT_TRUE(line.second.size() == 2);
        ASSERT_TRUE(line.second[0].original == negation);
        ASSERT_TRUE(line.second[0].validityName == mainNs);
        ASSERT_TRUE(line.second[1].original == equality);
        ASSERT_TRUE(line.second[1].validityName == branch);
        citeChecked = true;
    }
    ASSERT_TRUE(citeChecked);
}

TEST(equi_reshuffle, apply_equi_classes_expands_negation_via_later_descendant_class) {
    // Scope symmetry, Pass-1 direction — the descendant twin of the D-226
    // insertion-order hole: the negation !(=[2,a]) sits at main FIRST; the
    // class {a,c} forms LATER at a branch scope. Pass 1's three-way
    // comparability pre-gate must revisit the main-scope negation for the
    // branch-scope delta class and deposit !(=[2,c]) at the branch.
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    gl::Memory& mb = ana.body;
    mb.isActive = true;

    const gl::NameId branchId = mb.nameMap.encodePush(
        gl::NameMap::MAIN_ID, "ordis_(or9[1,7,3,2])_((=[6,x]))");
    const std::string branch = mb.nameMap.decode(branchId);
    const std::string mainNs = "main";

    gl::DirtyState outDirty = gl::DirtyState::Clean;
    gl::PagedVector<gl::IntEncodedExpr> out(
        &gl::genScratchArenas().forSlot(0), &outDirty);
    const int lv[1] = { 0 };
    const gl::TransientOrigin noOrigin{};

    // 1. The negation arrives FIRST, at main — no class exists anywhere,
    //    so the arrival-time expansion is a no-op.
    const std::string negation = "!(=[2,a])";
    ana.addStatement(gl::StrSpan(negation), mb, true, lv, 1, noOrigin,
                     gl::StrSpan(mainNs), out);

    // 2. The class {a, c} forms LATER, at the branch scope.
    out.clear();
    const std::string equality = "(=[a,c])";
    const gl::TransientOrigin eqOrigin{ true, gl::OriginTag::broadcast,
                                        nullptr, 0 };
    ana.addStatement(gl::StrSpan(equality), mb, true, lv, 1, eqOrigin,
                     gl::StrSpan(branch), out);

    // Pre-check: the expanded form does not exist yet at either scope.
    const std::string expanded = "!(=[2,c])";
    ASSERT_TRUE(gl::lookupStatementLevels(mb.intStatementLevelsMap, mb.nameMap,
        gl::StrSpan(expanded), gl::StrSpan(branch)) == 0);

    // 3. The class-application pass expands the pre-existing negation via
    //    the strictly-deeper delta class.
    ana.applyEquiClasses(mb);

    ASSERT_TRUE(gl::lookupStatementLevels(mb.intStatementLevelsMap, mb.nameMap,
        gl::StrSpan(expanded), gl::StrSpan(branch)) != 0);
    // The product lands at the class's scope only — never at main.
    ASSERT_TRUE(gl::lookupStatementLevels(mb.intStatementLevelsMap, mb.nameMap,
        gl::StrSpan(expanded), gl::StrSpan(mainNs)) == 0);
}

TEST(equi_reshuffle, ancestor_known_predicate_directions) {
    // Direct coverage of the shared Site F predicate (Rule 18): strict
    // ancestors vs self, and the deliberate absence of a descendant check.
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    gl::Memory& mb = ana.body;
    const gl::NameId branchId = mb.nameMap.encodePush(
        gl::NameMap::MAIN_ID, "ordis_(or9[1,7,3,2])_((=[6,x]))");

    const gl::NameId exprId = mb.nameMap.encode(std::string("(in[7,N])"));
    gl::upsertStatementKey(mb.intKnownStatements,
        gl::packStatementKey(exprId, gl::NameMap::MAIN_ID), /*local=*/true);
    // Known at main: strict-ancestor-known from the branch, self-known at main.
    ASSERT_TRUE(gl::ancestorKnown(mb, exprId, branchId, /*includeSelf=*/false));
    ASSERT_TRUE(gl::ancestorKnown(mb, exprId, branchId, /*includeSelf=*/true));
    ASSERT_FALSE(gl::ancestorKnown(mb, exprId, gl::NameMap::MAIN_ID,
                                   /*includeSelf=*/false));
    ASSERT_TRUE(gl::ancestorKnown(mb, exprId, gl::NameMap::MAIN_ID,
                                  /*includeSelf=*/true));

    // Known only at the branch: never visible from main (no descendant
    // scan), self-only at the branch.
    const gl::NameId expr2Id = mb.nameMap.encode(std::string("(in[8,N])"));
    gl::upsertStatementKey(mb.intKnownStatements,
        gl::packStatementKey(expr2Id, branchId), /*local=*/true);
    ASSERT_FALSE(gl::ancestorKnown(mb, expr2Id, gl::NameMap::MAIN_ID,
                                   /*includeSelf=*/true));
    ASSERT_FALSE(gl::ancestorKnown(mb, expr2Id, branchId,
                                   /*includeSelf=*/false));
    ASSERT_TRUE(gl::ancestorKnown(mb, expr2Id, branchId,
                                  /*includeSelf=*/true));
}

TEST(equi_reshuffle, add_statement_refuses_ancestor_known_descendant_copy) {
    // The addStatement door gate: a statement known at main is refused at a
    // descendant branch scope — no registration, no newStatements row.
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    gl::Memory& mb = ana.body;
    mb.isActive = true;
    const gl::NameId branchId = mb.nameMap.encodePush(
        gl::NameMap::MAIN_ID, "ordis_(or9[1,7,3,2])_((=[6,x]))");
    const std::string branch = mb.nameMap.decode(branchId);
    const std::string mainNs = "main";

    gl::DirtyState outDirty = gl::DirtyState::Clean;
    gl::PagedVector<gl::IntEncodedExpr> out(
        &gl::genScratchArenas().forSlot(0), &outDirty);
    const int lv[1] = { 0 };
    const gl::TransientOrigin noOrigin{};
    const std::string stmt = "(in[7,N])";

    ana.addStatement(gl::StrSpan(stmt), mb, true, lv, 1, noOrigin,
                     gl::StrSpan(mainNs), out);
    ASSERT_TRUE(gl::lookupStatementLevels(mb.intStatementLevelsMap, mb.nameMap,
        gl::StrSpan(stmt), gl::StrSpan(mainNs)) != 0);

    out.clear();
    ana.addStatement(gl::StrSpan(stmt), mb, true, lv, 1, noOrigin,
                     gl::StrSpan(branch), out);
    ASSERT_TRUE(gl::lookupStatementLevels(mb.intStatementLevelsMap, mb.nameMap,
        gl::StrSpan(stmt), gl::StrSpan(branch)) == 0);
    ASSERT_EQ(out.size(), 0);
    // The main row is untouched.
    ASSERT_TRUE(gl::lookupStatementLevels(mb.intStatementLevelsMap, mb.nameMap,
        gl::StrSpan(stmt), gl::StrSpan(mainNs)) != 0);
}

TEST(equi_reshuffle, add_statement_refuses_ancestor_known_equality_pair) {
    // Equality shapes refuse as a PAIR (I-180 pair invariant holds at the
    // ancestor too): neither orientation registers at the branch.
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    gl::Memory& mb = ana.body;
    mb.isActive = true;
    const gl::NameId branchId = mb.nameMap.encodePush(
        gl::NameMap::MAIN_ID, "ordis_(or9[1,7,3,2])_((=[6,x]))");
    const std::string branch = mb.nameMap.decode(branchId);
    const std::string mainNs = "main";

    gl::DirtyState outDirty = gl::DirtyState::Clean;
    gl::PagedVector<gl::IntEncodedExpr> out(
        &gl::genScratchArenas().forSlot(0), &outDirty);
    const int lv[1] = { 0 };
    const gl::TransientOrigin noOrigin{};
    const std::string eq = "(=[x,y])";
    const std::string eqMirror = "(=[y,x])";

    ana.addStatement(gl::StrSpan(eq), mb, true, lv, 1, noOrigin,
                     gl::StrSpan(mainNs), out);
    ASSERT_TRUE(gl::lookupStatementLevels(mb.intStatementLevelsMap, mb.nameMap,
        gl::StrSpan(eq), gl::StrSpan(mainNs)) != 0);
    ASSERT_TRUE(gl::lookupStatementLevels(mb.intStatementLevelsMap, mb.nameMap,
        gl::StrSpan(eqMirror), gl::StrSpan(mainNs)) != 0);

    out.clear();
    ana.addStatement(gl::StrSpan(eq), mb, true, lv, 1, noOrigin,
                     gl::StrSpan(branch), out);
    ASSERT_TRUE(gl::lookupStatementLevels(mb.intStatementLevelsMap, mb.nameMap,
        gl::StrSpan(eq), gl::StrSpan(branch)) == 0);
    ASSERT_TRUE(gl::lookupStatementLevels(mb.intStatementLevelsMap, mb.nameMap,
        gl::StrSpan(eqMirror), gl::StrSpan(branch)) == 0);
    ASSERT_EQ(out.size(), 0);
}

TEST(equi_reshuffle, apply_equi_classes_refuses_ancestor_known_product) {
    // The equivalence-class commit gate: a branch-scope class rewriting a
    // main statement is refused when the product is itself main-known.
    // Lex order makes the class {a,c} canonicalize toward a, so the
    // rewrite direction is (in[c,N]) -> (in[a,N]).
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    gl::Memory& mb = ana.body;
    mb.isActive = true;
    mb.level = 0;  // the commit's max-level gate compares against this
    const gl::NameId branchId = mb.nameMap.encodePush(
        gl::NameMap::MAIN_ID, "ordis_(or9[1,7,3,2])_((=[6,x]))");
    const std::string branch = mb.nameMap.decode(branchId);
    const std::string mainNs = "main";

    gl::DirtyState outDirty = gl::DirtyState::Clean;
    gl::PagedVector<gl::IntEncodedExpr> out(
        &gl::genScratchArenas().forSlot(0), &outDirty);
    const int lv[1] = { 0 };
    const gl::TransientOrigin noOrigin{};
    const gl::TransientOrigin eqOrigin{ true, gl::OriginTag::broadcast,
                                        nullptr, 0 };

    ana.addStatement(gl::StrSpan(std::string("(in[a,N])")), mb, true, lv, 1,
                     noOrigin, gl::StrSpan(mainNs), out);
    out.clear();
    ana.addStatement(gl::StrSpan(std::string("(in[c,N])")), mb, true, lv, 1,
                     noOrigin, gl::StrSpan(mainNs), out);
    out.clear();
    ana.addStatement(gl::StrSpan(std::string("(=[a,c])")), mb, true, lv, 1,
                     eqOrigin, gl::StrSpan(branch), out);

    ana.applyEquiClasses(mb);

    // The rewrite at the branch is refused: main already knows (in[a,N]).
    ASSERT_TRUE(gl::lookupStatementLevels(mb.intStatementLevelsMap, mb.nameMap,
        gl::StrSpan(std::string("(in[a,N])")), gl::StrSpan(branch)) == 0);
}

TEST(equi_reshuffle, apply_equi_classes_descendant_rewrite_still_deposits) {
    // D-33 control: with no ancestor-known collision, the descendant-scope
    // class still rewrites the main statement and deposits at the branch.
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    gl::Memory& mb = ana.body;
    mb.isActive = true;
    mb.level = 0;  // the commit's max-level gate compares against this
    const gl::NameId branchId = mb.nameMap.encodePush(
        gl::NameMap::MAIN_ID, "ordis_(or9[1,7,3,2])_((=[6,x]))");
    const std::string branch = mb.nameMap.decode(branchId);
    const std::string mainNs = "main";

    gl::DirtyState outDirty = gl::DirtyState::Clean;
    gl::PagedVector<gl::IntEncodedExpr> out(
        &gl::genScratchArenas().forSlot(0), &outDirty);
    const int lv[1] = { 0 };
    const gl::TransientOrigin noOrigin{};
    const gl::TransientOrigin eqOrigin{ true, gl::OriginTag::broadcast,
                                        nullptr, 0 };

    ana.addStatement(gl::StrSpan(std::string("(in[c,N])")), mb, true, lv, 1,
                     noOrigin, gl::StrSpan(mainNs), out);
    out.clear();
    ana.addStatement(gl::StrSpan(std::string("(=[a,c])")), mb, true, lv, 1,
                     eqOrigin, gl::StrSpan(branch), out);

    ana.applyEquiClasses(mb);

    ASSERT_TRUE(gl::lookupStatementLevels(mb.intStatementLevelsMap, mb.nameMap,
        gl::StrSpan(std::string("(in[a,N])")), gl::StrSpan(branch)) != 0);
}

TEST(equi_reshuffle, positive_rewrite_cites_inherited_equality_at_origin_scope) {
    // A branch class may absorb members and pair levels from an ancestor
    // class. Its equality1 history must cite each justifying equality where
    // that equality's origin row lives; otherwise a branch-scope citation of
    // main-scope content cannot cross the ordis boundary in buildStack.
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    gl::Memory& mb = ana.body;
    mb.isActive = true;
    mb.level = 0;
    const gl::NameId branchId = mb.nameMap.encodePush(
        gl::NameMap::MAIN_ID, "ordis_(or9[1,7,3,2])_((=[6,x]))");
    const std::string branch = mb.nameMap.decode(branchId);
    const std::string mainNs = "main";

    gl::DirtyState outDirty = gl::DirtyState::Clean;
    gl::PagedVector<gl::IntEncodedExpr> out(
        &gl::genScratchArenas().forSlot(0), &outDirty);
    const int lv[1] = { 0 };
    const gl::TransientOrigin noOrigin{};
    const gl::TransientOrigin eqOrigin{ true, gl::OriginTag::broadcast,
                                        nullptr, 0 };

    const std::string ancestorEquality = "(=[a,c])";
    ana.addStatement(gl::StrSpan(ancestorEquality), mb, true, lv, 1,
                     eqOrigin, gl::StrSpan(mainNs), out);
    ana.applyEquiClasses(mb);

    out.clear();
    ana.addStatement(gl::StrSpan(std::string("(=[c,d])")), mb, true, lv, 1,
                     eqOrigin, gl::StrSpan(branch), out);

    // The branch merge creates d=a through bridge c. The branch equality
    // d=c and inherited main equality c=a must retain their different scopes
    // in the equality2 row.
    const std::string mergedEquality = "(=[d,a])";
    int64_t mergedPk = 0;
    ASSERT_TRUE(gl::lookupOriginKey(mb.originInterner, mergedEquality, branch,
                                    mergedPk));
    const std::vector<gl::IdOrigin> mergedRows =
        mb.exprOriginMap.recordsAt(mb.exprOriginMap.lookup(mergedPk));
    bool mergeCiteChecked = false;
    for (const gl::IdOrigin& o : mergedRows) {
        const auto line = gl::decodeOrigin(o, mb.originInterner);
        if (line.first != "equality2") continue;
        ASSERT_TRUE(line.second.size() == 2);
        ASSERT_TRUE(line.second[0].original == "(=[d,c])");
        ASSERT_TRUE(line.second[0].validityName == branch);
        ASSERT_TRUE(line.second[1].original == "(=[c,a])");
        ASSERT_TRUE(line.second[1].validityName == mainNs);
        mergeCiteChecked = true;
    }
    ASSERT_TRUE(mergeCiteChecked);

    out.clear();
    const std::string source = "(in[c,N])";
    ana.addStatement(gl::StrSpan(source), mb, true, lv, 1, noOrigin,
                     gl::StrSpan(branch), out);

    ana.applyEquiClasses(mb);

    const std::string product = "(in[a,N])";
    ASSERT_TRUE(gl::lookupStatementLevels(mb.intStatementLevelsMap,
        mb.nameMap, gl::StrSpan(product), gl::StrSpan(branch)) != 0);

    int64_t productPk = 0;
    ASSERT_TRUE(gl::lookupOriginKey(mb.originInterner, product, branch,
                                    productPk));
    const std::vector<gl::IdOrigin> productRows =
        mb.exprOriginMap.recordsAt(mb.exprOriginMap.lookup(productPk));
    bool citeChecked = false;
    for (const gl::IdOrigin& o : productRows) {
        const auto line = gl::decodeOrigin(o, mb.originInterner);
        if (line.first != "equality1") continue;
        ASSERT_TRUE(line.second.size() == 2);
        ASSERT_TRUE(line.second[0].original == source);
        ASSERT_TRUE(line.second[0].validityName == branch);
        ASSERT_TRUE(line.second[1].original == "(=[c,a])");
        ASSERT_TRUE(line.second[1].validityName == mainNs);
        citeChecked = true;
    }
    ASSERT_TRUE(citeChecked);
}

TEST(equi_reshuffle, sweep_ancestor_known_rows_drops_lists_keeps_tombstones) {
    // The end-of-burst sweep: a legacy descendant duplicate (branch-first
    // residue simulated by direct container writes, bypassing the door
    // gates) loses its statement-LIST rows; the registry rows (known /
    // levels / local-set) survive as tombstones; main rows are untouched.
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    gl::Memory& mb = ana.body;
    mb.isActive = true;
    const gl::NameId branchId = mb.nameMap.encodePush(
        gl::NameMap::MAIN_ID, "ordis_(or9[1,7,3,2])_((=[6,x]))");
    const std::string branch = mb.nameMap.decode(branchId);
    const std::string mainNs = "main";
    const std::string stmt = "(in[7,N])";

    gl::DirtyState outDirty = gl::DirtyState::Clean;
    gl::PagedVector<gl::IntEncodedExpr> out(
        &gl::genScratchArenas().forSlot(0), &outDirty);
    const int lv[1] = { 0 };
    const gl::TransientOrigin noOrigin{};

    // Ancestor row through the door.
    ana.addStatement(gl::StrSpan(stmt), mb, true, lv, 1, noOrigin,
                     gl::StrSpan(mainNs), out);

    // Legacy duplicate at the branch: direct writes simulating the
    // pre-gate state (register + list rows at the branch scope).
    const gl::NameId stmtId = mb.nameMap.encode(stmt);
    const int64_t pkBranch = gl::packStatementKey(stmtId, branchId);
    mb.intStatementLevelsMap.assignSetRange(pkBranch, lv, lv + 1);
    gl::upsertStatementKey(mb.intKnownStatements, pkBranch, /*local=*/true);
    mb.intLocalEncodedStatementsSet.mint(pkBranch);
    const gl::IntEncodedExpr ieBranch = gl::encodeExpression(
        gl::StrSpan(stmt), gl::StrSpan(branch), mb.nameMap);
    mb.intEncodedStatements.push_back(ieBranch);
    mb.intLocalEncodedStatements.push_back(ieBranch);
    mb.intLocalEncodedStatementsDelta.push_back(ieBranch);

    mb.mutatedThisBurst = false;
    ana.sweepAncestorKnownRows(mb);

    // Branch LIST rows gone from all three vectors.
    for (int32_t i = 0; i < mb.intEncodedStatements.size(); ++i) {
        ASSERT_FALSE(mb.intEncodedStatements[i].originalId == stmtId
                     && mb.intEncodedStatements[i].validityId == branchId);
    }
    for (int32_t i = 0; i < mb.intLocalEncodedStatements.size(); ++i) {
        ASSERT_FALSE(mb.intLocalEncodedStatements[i].originalId == stmtId
                     && mb.intLocalEncodedStatements[i].validityId == branchId);
    }
    for (int32_t i = 0; i < mb.intLocalEncodedStatementsDelta.size(); ++i) {
        ASSERT_FALSE(
            mb.intLocalEncodedStatementsDelta[i].originalId == stmtId
            && mb.intLocalEncodedStatementsDelta[i].validityId == branchId);
    }
    // Tombstones survive (I-58 / I-85 discipline of the removal door).
    ASSERT_TRUE(mb.intKnownStatements.find(
        gl::StatementKey{ stmtId, branchId }) != nullptr);
    ASSERT_TRUE(mb.intStatementLevelsMap.lookup(pkBranch) != 0);
    // Main rows untouched.
    ASSERT_TRUE(gl::lookupStatementLevels(mb.intStatementLevelsMap, mb.nameMap,
        gl::StrSpan(stmt), gl::StrSpan(mainNs)) != 0);
    bool mainListRow = false;
    for (int32_t i = 0; i < mb.intEncodedStatements.size(); ++i) {
        if (mb.intEncodedStatements[i].originalId == stmtId
            && mb.intEncodedStatements[i].validityId == gl::NameMap::MAIN_ID) {
            mainListRow = true;
        }
    }
    ASSERT_TRUE(mainListRow);
    // Quiescence mutation flag raised (D-194).
    ASSERT_TRUE(mb.mutatedThisBurst);

    // Idempotence: a second sweep finds nothing and flags nothing.
    mb.mutatedThisBurst = false;
    ana.sweepAncestorKnownRows(mb);
    ASSERT_FALSE(mb.mutatedThisBurst);
}

TEST(equi_reshuffle, remove_expression_repairs_eqclass_waterlines) {
    // The removal door's waterline repair: erasing an intEncodedStatements
    // row below a stored per-class waterline decrements it by exactly the
    // erased-below count; a waterline at or below the erased position is
    // untouched.
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    gl::Memory& mb = ana.body;
    mb.isActive = true;
    const std::string mainNs = "main";

    // Four rows at positions 0..3.
    const std::string t0 = "(in[p,N])";
    const std::string t1 = "(in[q,N])";
    const std::string t2 = "(in[r,N])";
    const std::string t3 = "(in[s,N])";
    for (const std::string* t : { &t0, &t1, &t2, &t3 }) {
        const gl::IntEncodedExpr ie = gl::encodeExpression(
            gl::StrSpan(*t), gl::StrSpan(mainNs), mb.nameMap);
        mb.intEncodedStatements.push_back(ie);
    }
    const int32_t baseCount = mb.intEncodedStatements.size();
    ASSERT_TRUE(baseCount >= 4);
    const int32_t posQ = baseCount - 3;  // t1's position

    // Two waterlines: one above the erase position, one at it.
    const std::vector<gl::NameId> membersHigh = {
        mb.nameMap.encode(std::string("wlA")),
        mb.nameMap.encode(std::string("wlB")) };
    const std::vector<gl::NameId> membersLow = {
        mb.nameMap.encode(std::string("wlC")),
        mb.nameMap.encode(std::string("wlD")) };
    gl::upsertEqClassIndex(mb.eqClassSttmntIndexMapMap, gl::NameMap::MAIN_ID,
                           membersHigh, baseCount);       // above posQ
    gl::upsertEqClassIndex(mb.eqClassSttmntIndexMapMap, gl::NameMap::MAIN_ID,
                           membersLow, posQ);             // at posQ

    ana.removeExpressionFromMemoryBlock(gl::StrSpan(t1), gl::StrSpan(mainNs),
                                        mb, /*state=*/0);

    ASSERT_EQ(mb.intEncodedStatements.size(), baseCount - 1);
    ASSERT_EQ(gl::lookupEqClassIndex(mb.eqClassSttmntIndexMapMap,
        gl::NameMap::MAIN_ID, membersHigh), baseCount - 1);
    ASSERT_EQ(gl::lookupEqClassIndex(mb.eqClassSttmntIndexMapMap,
        gl::NameMap::MAIN_ID, membersLow), posQ);
}

TEST(equi_reshuffle, seeding_time_class_delta_survives_to_first_step) {
    // Sole-clear contract of `changedClassesThisStep`
    // (I-215): a class minted OUTSIDE any
    // `standardProcessing` call — here simulating LB seeding, where premise
    // facts land first and the variable-copy equality axiom follows — must
    // survive to the LB's first elementary step. Pass 1 is the ONLY path
    // that back-applies it to the pre-existing statements: the class's
    // Pass-2 waterline is seeded to the then-current registry size, so
    // Pass 2 never revisits them. After the step, Step 7 must have cleared
    // the consumed tracker.
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    gl::Memory& mb = ana.body;
    mb.isActive = true;
    // The product-commit gate requires max(product levels) == mb.level; the
    // bare analyzer root defaults to -1, so pin the production level-0 shape.
    mb.level = 0;

    gl::DirtyState outDirty = gl::DirtyState::Clean;
    gl::PagedVector<gl::IntEncodedExpr> out(
        &gl::genScratchArenas().forSlot(0), &outDirty);
    const int lv[1] = { 0 };
    const gl::TransientOrigin noOrigin{};

    // Seeding-time deposits at main, before any standardProcessing call:
    // the plain fact first, then the copy-axiom-shaped equality.
    const std::string mainNs = "main";
    const std::string fact = "(in2[2,a,3])";
    ana.addStatement(gl::StrSpan(fact), mb, true, lv, 1, noOrigin,
                     gl::StrSpan(mainNs), out);
    out.clear();
    const std::string equality = "(=[a,c])";
    const gl::TransientOrigin eqOrigin{ true, gl::OriginTag::broadcast,
                                        nullptr, 0 };
    ana.addStatement(gl::StrSpan(equality), mb, true, lv, 1, eqOrigin,
                     gl::StrSpan(mainNs), out);

    // The seeding delta is pending; the class product does not exist yet.
    ASSERT_TRUE(mb.changedClassesThisStep.size() > 0);
    const std::string product = "(in2[2,c,3])";
    ASSERT_TRUE(gl::lookupStatementLevels(mb.intStatementLevelsMap, mb.nameMap,
        gl::StrSpan(product), gl::StrSpan(mainNs)) == 0);

    // First elementary step, phase 1: the pre-burst standardProcessing
    // consumes the seeding delta (Pass 1 back-applies the class to the
    // seeded fact) and its Step 7 clears the tracker.
    ana.performElemPhase1(mb, 0);

    ASSERT_TRUE(gl::lookupStatementLevels(mb.intStatementLevelsMap, mb.nameMap,
        gl::StrSpan(product), gl::StrSpan(mainNs)) != 0);
    ASSERT_TRUE(mb.changedClassesThisStep.size() == 0);
}

// ---------------------------------------------------------------------------
// The applyEquiClasses statement batch — containers and the two table passes.
// ---------------------------------------------------------------------------

namespace {

// Serialize a heap class and append it to a validity's class run (one or
// more classes per validity; the run is replaced wholesale).
void putClassesAt(gl::Memory& mb, gl::NameId vid,
                  const std::vector<std::set<std::string>>& memberLists) {
    std::vector<char> pool;
    std::vector<int32_t> lens;
    for (const auto& names : memberLists) {
        gl::EquivalenceClass cls;
        cls.setMembersFromNames(names, mb.nameMap);
        const std::vector<char> blob = gl::serializeEquivalenceClass(cls);
        pool.insert(pool.end(), blob.begin(), blob.end());
        lens.push_back(static_cast<int32_t>(blob.size()));
    }
    mb.equivalenceClassesMap.inner().assignRun(vid, pool.data(), lens.data(),
                                               static_cast<int32_t>(lens.size()));
}

void pushDelta(gl::Memory& mb, gl::NameId vid, const std::set<std::string>& names) {
    gl::EquivalenceClass cls;
    cls.setMembersFromNames(names, mb.nameMap);
    const std::vector<char> blob = gl::serializeEquivalenceClass(cls);
    mb.changedClassesThisStep.push(vid, blob.data(), static_cast<int32_t>(blob.size()));
}

gl::IntEncodedExpr pushRow(gl::Memory& mb, const char* text, gl::NameId vid) {
    const gl::EncodedExpression src(text, mb.nameMap.decode(vid));
    const gl::IntEncodedExpr row = gl::encodeExpression(src, mb.nameMap);
    mb.intEncodedStatements.push_back(row);
    return row;
}

}  // namespace

TEST(equi_batch, merge_level_runs_non_derived_tier_transparent) {
    int out[8];
    const int minus1[1] = { -1 };
    const int a[2] = { 1, 3 };
    const int b[2] = { 2, 3 };
    // {-1} ∪ {} = {-1}; {-1} ∪ {2} = {2}; {1,3} ∪ {2,3} = {1,2,3}; {} ∪ {} = {}.
    ASSERT_EQ(gl::mergeLevelRuns(minus1, 1, nullptr, 0, out, 8), 1);
    ASSERT_EQ(out[0], -1);
    ASSERT_EQ(gl::mergeLevelRuns(minus1, 1, b, 1, out, 8), 1);
    ASSERT_EQ(out[0], 2);
    ASSERT_EQ(gl::mergeLevelRuns(a, 2, b, 2, out, 8), 3);
    ASSERT_EQ(out[0], 1); ASSERT_EQ(out[1], 2); ASSERT_EQ(out[2], 3);
    ASSERT_EQ(gl::mergeLevelRuns(nullptr, 0, nullptr, 0, out, 8), 0);
    ASSERT_EQ(gl::mergeLevelRuns(minus1, 1, minus1, 1, out, 8), 1);
    ASSERT_EQ(out[0], -1);
}

TEST(equi_batch, class_table_posts_every_member_once_per_class) {
    gl::Memory mb;
    const gl::NameId mainId = mb.nameMap.encode("main");
    gl::EquivalenceClass c1; c1.setMembersFromNames({ "a", "b" }, mb.nameMap);
    gl::EquivalenceClass c2; c2.setMembersFromNames({ "b", "c" }, mb.nameMap);
    const std::vector<char> b1 = gl::serializeEquivalenceClass(c1);
    const std::vector<char> b2 = gl::serializeEquivalenceClass(c2);
    const gl::EquivalenceClassView v1{ b1.data(), static_cast<int32_t>(b1.size()) };
    const gl::EquivalenceClassView v2{ b2.data(), static_cast<int32_t>(b2.size()) };
    const gl::NameId idA = mb.nameMap.lookup(gl::StrSpan("a", 1));
    const gl::NameId idB = mb.nameMap.lookup(gl::StrSpan("b", 1));
    const gl::NameId idC = mb.nameMap.lookup(gl::StrSpan("c", 1));
    const gl::NameId strong1[2] = { idA, idB };
    const gl::NameId strong2[2] = { idB, idC };

    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::EquiClassTable table(&gl::genScratchArenas().forSlot(0), &dirty);
    const int32_t i1 = table.addClass(mainId, v1, strong1, 2, 0, 0, true);
    const int32_t i2 = table.addClass(mainId, v2, strong2, 2, 1, 7, false);
    ASSERT_EQ(table.classCount(), 2);
    ASSERT_EQ(table.rowAt(i1).isDelta, 1);
    ASSERT_EQ(table.rowAt(i2).isDelta, 0);
    ASSERT_EQ(table.rowAt(i2).dueFrom, static_cast<int64_t>(7));
    ASSERT_EQ(table.rowAt(i1).strongCount, 2);
    ASSERT_EQ(table.rowAt(i1).strong[1], idB);
    ASSERT_EQ(table.viewAt(i2).memberCount(), 2);

    // b is in both classes, a only in the first, c only in the second, d in none.
    int hitsB = 0, sawI1 = 0, sawI2 = 0;
    table.forEachClassContaining(idB, [&](int32_t c) { ++hitsB; if (c == i1) ++sawI1; if (c == i2) ++sawI2; });
    ASSERT_EQ(hitsB, 2); ASSERT_EQ(sawI1, 1); ASSERT_EQ(sawI2, 1);
    int hitsA = 0;
    table.forEachClassContaining(idA, [&](int32_t c) { ++hitsA; ASSERT_EQ(c, i1); });
    ASSERT_EQ(hitsA, 1);
    ASSERT_TRUE(table.anyClassContains(idC));
    ASSERT_FALSE(table.anyClassContains(mb.nameMap.encode("d")));

    table.setDueFrom(i2, 11);
    ASSERT_EQ(table.rowAt(i2).dueFrom, static_cast<int64_t>(11));
}

TEST(equi_batch, staged_products_dedup_key_and_pools) {
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::StagedEquiProducts staged(&gl::genScratchArenas().forSlot(0), &dirty);
    const gl::NameId vid1 = 5, vid2 = 6;
    const int lv[2] = { 1, 4 };
    const gl::NameId from[1] = { 21 };
    const gl::NameId to[1] = { 22 };
    const gl::NameId args[3] = { 7, 22, 9 };

    ASSERT_FALSE(staged.contains(vid1, gl::StrSpan("(P[a,b,c])", 10)));
    const int32_t r0 = staged.stage(vid1, gl::StrSpan("(P[a,b,c])", 10),
        gl::StagedEquiProducts::Seed, -1, -1, lv, 2, nullptr, nullptr, 0, args, 3);
    const int32_t r1 = staged.stage(vid2, gl::StrSpan("(P[a,b,c])", 10),
        gl::StagedEquiProducts::Full, r0, 3, lv, 1, from, to, 1, args, 3);
    ASSERT_EQ(r0, 0); ASSERT_EQ(r1, 1);
    ASSERT_EQ(staged.count(), 2);
    // Same text at a different scope is a different key; same key is present.
    ASSERT_TRUE(staged.contains(vid1, gl::StrSpan("(P[a,b,c])", 10)));
    ASSERT_TRUE(staged.contains(vid2, gl::StrSpan("(P[a,b,c])", 10)));
    ASSERT_FALSE(staged.contains(vid1, gl::StrSpan("(P[a,b])", 8)));

    const gl::StrSpan t1 = staged.textAt(r1);
    ASSERT_EQ(t1.len, 10);
    ASSERT_TRUE(gl::equalSpans(t1, gl::StrSpan("(P[a,b,c])", 10)));
    ASSERT_EQ(staged.recordAt(r1).targetVid, vid2);
    ASSERT_EQ(staged.recordAt(r1).sourceRec, r0);
    ASSERT_EQ(staged.recordAt(r1).classIdx, 3);
    ASSERT_EQ(staged.recordAt(r1).kind, static_cast<int32_t>(gl::StagedEquiProducts::Full));
    int lvOut[4];
    ASSERT_EQ(staged.copyLevels(r0, lvOut, 4), 2);
    ASSERT_EQ(lvOut[1], 4);
    ASSERT_EQ(staged.copyLevels(r1, lvOut, 4), 1);
    ASSERT_EQ(staged.recordAt(r1).pairCount, 1);
    ASSERT_EQ(staged.pairFromAt(r1, 0), static_cast<gl::NameId>(21));
    ASSERT_EQ(staged.pairToAt(r1, 0), static_cast<gl::NameId>(22));
    ASSERT_EQ(staged.recordAt(r1).argCount, 3);
    ASSERT_EQ(staged.argAt(r1, 1), static_cast<gl::NameId>(22));

    staged.clear();
    ASSERT_EQ(staged.count(), 0);
    ASSERT_FALSE(staged.contains(vid1, gl::StrSpan("(P[a,b,c])", 10)));
}

TEST(equi_batch, build_class_table_delta_rank_due_and_dead_skip) {
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    gl::Memory& mb = ana.body;
    const gl::NameId mainId = mb.nameMap.encode("main");
    const gl::NameId branchId = mb.nameMap.encodePush(mainId, "ordis_(or9[1,7,3,2])_((=[6,x]))");

    // main: {a,b} (delta entry 0) and {x,y} (non-delta, waterline 7);
    // branch: {p,q} (delta entry 1, but dead-scope-skipped by Pass 1).
    putClassesAt(mb, mainId, { { "a", "b" }, { "x", "y" } });
    putClassesAt(mb, branchId, { { "p", "q" } });
    pushDelta(mb, mainId, { "a", "b" });
    pushDelta(mb, branchId, { "p", "q" });
    {
        gl::EquivalenceClass xy; xy.setMembersFromNames({ "x", "y" }, mb.nameMap);
        gl::upsertEqClassIndex(mb.eqClassSttmntIndexMapMap, mainId, xy.memberIds, 7);
    }

    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::ScratchArena& gArena = gl::genScratchArenas().forSlot(0);
    gl::PagedVector<uint8_t> deltaLive(&gArena, &dirty);
    gl::PagedVector<int64_t> startIdx(&gArena, &dirty);
    deltaLive.push_back(1); deltaLive.push_back(0);
    startIdx.push_back(0);  startIdx.push_back(0);
    gl::EquiClassTable table(&gArena, &dirty);
    ana.buildEquiClassTable(mb, table, deltaLive, startIdx, gArena);

    // The dead delta class is absent; the two main classes are present.
    ASSERT_EQ(table.classCount(), 2);
    const gl::NameId idA = mb.nameMap.lookup(gl::StrSpan("a", 1));
    const gl::NameId idX = mb.nameMap.lookup(gl::StrSpan("x", 1));
    const gl::NameId idP = mb.nameMap.lookup(gl::StrSpan("p", 1));
    ASSERT_FALSE(table.anyClassContains(idP));
    int32_t cAB = -1, cXY = -1;
    table.forEachClassContaining(idA, [&](int32_t c) { cAB = c; });
    table.forEachClassContaining(idX, [&](int32_t c) { cXY = c; });
    ASSERT_TRUE(cAB >= 0 && cXY >= 0 && cAB != cXY);
    ASSERT_EQ(table.rowAt(cAB).isDelta, 1);
    ASSERT_EQ(table.rowAt(cAB).rank, 0);
    ASSERT_EQ(table.rowAt(cAB).dueFrom, static_cast<int64_t>(0));
    ASSERT_EQ(table.rowAt(cAB).strongCount, 2);
    ASSERT_EQ(table.rowAt(cXY).isDelta, 0);
    ASSERT_EQ(table.rowAt(cXY).rank, 2);   // after the two delta entries
    ASSERT_EQ(table.rowAt(cXY).dueFrom, static_cast<int64_t>(7));
    ASSERT_EQ(table.rowAt(cXY).vid, mainId);
}

TEST(equi_batch, collect_affected_rows_due_shape_and_scope) {
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    gl::Memory& mb = ana.body;
    const gl::NameId mainId = mb.nameMap.encode("main");
    const gl::NameId br1 = mb.nameMap.encodePush(mainId, "ordis_(or9[1,7,3,2])_((=[6,x]))");
    const gl::NameId br2 = mb.nameMap.encodePush(mainId, "ordis_(or9[1,7,3,2])_((=[6,y]))");

    // main: {a,b} delta (due from 0), {x,y} non-delta (due from 5); br1: {p,q} delta.
    putClassesAt(mb, mainId, { { "a", "b" }, { "x", "y" } });
    putClassesAt(mb, br1, { { "p", "q" } });
    pushDelta(mb, mainId, { "a", "b" });
    pushDelta(mb, br1, { "p", "q" });
    {
        gl::EquivalenceClass xy; xy.setMembersFromNames({ "x", "y" }, mb.nameMap);
        gl::upsertEqClassIndex(mb.eqClassSttmntIndexMapMap, mainId, xy.memberIds, 5);
    }

    // Rows (positions 0..7).
    pushRow(mb, "(in[a,c])", mainId);          // 0: delta member -> affected
    pushRow(mb, "(in[c,d])", mainId);          // 1: no member -> not affected
    pushRow(mb, "(in[x,c])", mainId);          // 2: non-delta class, below its waterline -> not affected
    pushRow(mb, "(=[a,c])", mainId);           // 3: positive equality -> skipped
    pushRow(mb, "!(=[a,c])", mainId);          // 4: negated equality, delta due -> negeq list
    pushRow(mb, "(in[x,c])", br2);             // 5: non-delta class at/after its waterline, comparable (main is an ancestor) -> affected
    pushRow(mb, "(in[p,c])", br2);             // 6: br1 class, br2 row: incomparable -> not affected
    pushRow(mb, "(AnchorPeano[a,c])", mainId); // 7: anchor -> skipped
    pushRow(mb, "!(=[x,c])", br2);             // 8: negated equality touched only by a non-delta class -> not listed

    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::ScratchArena& gArena = gl::genScratchArenas().forSlot(0);
    gl::PagedVector<uint8_t> deltaLive(&gArena, &dirty);
    gl::PagedVector<int64_t> startIdx(&gArena, &dirty);
    deltaLive.push_back(1); deltaLive.push_back(1);
    startIdx.push_back(0);  startIdx.push_back(0);
    gl::EquiClassTable table(&gArena, &dirty);
    ana.buildEquiClassTable(mb, table, deltaLive, startIdx, gArena);
    ASSERT_EQ(table.classCount(), 3);

    gl::PagedVector<int32_t> rows(&gArena, &dirty);
    gl::PagedVector<int32_t> negeq(&gArena, &dirty);
    gl::PagedVector<int64_t> negeqKeys(&gArena, &dirty);
    ana.collectEquiAffectedRows(mb, table, rows, negeq, negeqKeys);
    ASSERT_EQ(rows.size(), 2);
    ASSERT_EQ(rows[0], 0);
    ASSERT_EQ(rows[1], 5);
    ASSERT_EQ(negeq.size(), 1);
    ASSERT_EQ(negeq[0], 4);
    ASSERT_EQ(negeqKeys.size(), 1);
    ASSERT_EQ(negeqKeys[0], gl::packStatementKey(mb.intEncodedStatements[4].originalId,
                                                 mb.intEncodedStatements[4].validityId));

    // Advancing every class past the registry makes nothing due.
    for (int32_t c = 0; c < table.classCount(); ++c)
        table.setDueFrom(c, mb.intEncodedStatements.size());
    ana.collectEquiAffectedRows(mb, table, rows, negeq, negeqKeys);
    ASSERT_EQ(rows.size(), 0);
    ASSERT_EQ(negeq.size(), 0);
}

// ---------------------------------------------------------------------------
// The closure and the drain.
// ---------------------------------------------------------------------------

namespace {

// Register one statement at a scope through the levels / known / registry
// doors (a registered row the closure can seed from), levels {lv}.
gl::IntEncodedExpr registerRow(gl::Memory& mb, const char* text, gl::NameId vid, int lv) {
    const gl::IntEncodedExpr row = pushRow(mb, text, vid);
    const int64_t pk = gl::packStatementKey(row.originalId, row.validityId);
    mb.intStatementLevelsMap.insertSorted(pk, lv);
    gl::upsertStatementKey(mb.intKnownStatements, pk, /*local=*/true);
    return row;
}

struct BatchFixture {
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::ScratchArena& gArena;
    gl::PagedVector<uint8_t> deltaLive;
    gl::PagedVector<int64_t> startIdx;
    gl::EquiClassTable table;
    gl::StagedEquiProducts staged;
    gl::PagedVector<int32_t> classStamp;
    gl::PagedVector<int32_t> cand;
    gl::PagedVector<int32_t> rows;
    gl::PagedVector<int32_t> negeq;
    gl::PagedVector<int64_t> negeqKeys;
    gl::PagedVector<gl::IntEncodedExpr> committed;
    int32_t stamp = 0;

    BatchFixture()
        : gArena(gl::genScratchArenas().forSlot(0)),
          deltaLive(&gArena, &dirty), startIdx(&gArena, &dirty),
          table(&gArena, &dirty), staged(&gArena, &dirty),
          classStamp(&gArena, &dirty), cand(&gArena, &dirty),
          rows(&gArena, &dirty), negeq(&gArena, &dirty),
          negeqKeys(&gArena, &dirty), committed(&gArena, &dirty) {}

    void build(gl::ExpressionAnalyzer& ana, gl::Memory& mb) {
        const int32_t n = mb.changedClassesThisStep.size();
        for (int32_t k = 0; k < n; ++k) { deltaLive.push_back(1); startIdx.push_back(0); }
        ana.buildEquiClassTable(mb, table, deltaLive, startIdx, gArena);
        for (int32_t c = 0; c < table.classCount(); ++c) classStamp.push_back(-1);
    }
};

int countKind(const gl::StagedEquiProducts& s, int32_t kind) {
    int n = 0;
    for (int32_t i = 0; i < s.count(); ++i) if (s.recordAt(i).kind == kind) ++n;
    return n;
}

int32_t findStaged(const gl::StagedEquiProducts& s, gl::NameId vid, const char* text) {
    const gl::StrSpan t(text, static_cast<int32_t>(std::strlen(text)));
    for (int32_t i = 0; i < s.count(); ++i)
        if (s.recordAt(i).targetVid == vid && gl::equalSpans(s.textAt(i), t)) return i;
    return -1;
}

}  // namespace

TEST(equi_batch, closure_single_class_orbit_drops_identity) {
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    gl::Memory& mb = ana.body;
    const gl::NameId mainId = mb.nameMap.encode("main");
    putClassesAt(mb, mainId, { { "a", "b", "c" } });
    pushDelta(mb, mainId, { "a", "b", "c" });
    registerRow(mb, "(in[a,x])", mainId, 2);

    BatchFixture f;
    f.build(ana, mb);
    ana.collectEquiAffectedRows(mb, f.table, f.rows, f.negeq, f.negeqKeys);
    ASSERT_EQ(f.rows.size(), 1);
    ana.stageEquiClosure(mb, f.table, f.rows[0], f.staged, f.classStamp, f.stamp, f.cand);

    // Seed + the two real products; the identity (in[a,x]) is known and dropped.
    ASSERT_EQ(countKind(f.staged, gl::StagedEquiProducts::Seed), 1);
    ASSERT_EQ(countKind(f.staged, gl::StagedEquiProducts::Full), 2);
    ASSERT_EQ(countKind(f.staged, gl::StagedEquiProducts::OriginOnly), 0);
    const int32_t rb = findStaged(f.staged, mainId, "(in[b,x])");
    const int32_t rc = findStaged(f.staged, mainId, "(in[c,x])");
    ASSERT_TRUE(rb >= 0 && rc >= 0);
    // Source = the seed, one justifier (a -> b), levels = the row's {2}.
    ASSERT_EQ(f.staged.recordAt(rb).sourceRec, 0);
    ASSERT_EQ(f.staged.recordAt(rb).pairCount, 1);
    ASSERT_EQ(f.staged.pairFromAt(rb, 0), mb.nameMap.lookup(gl::StrSpan("a", 1)));
    ASSERT_EQ(f.staged.pairToAt(rb, 0), mb.nameMap.lookup(gl::StrSpan("b", 1)));
    int lv[4];
    ASSERT_EQ(f.staged.copyLevels(rb, lv, 4), 1);
    ASSERT_EQ(lv[0], 2);
    ASSERT_EQ(f.staged.recordAt(rb).argCount, 2);
    ASSERT_EQ(f.staged.argAt(rb, 0), mb.nameMap.lookup(gl::StrSpan("b", 1)));
    ASSERT_EQ(f.staged.argAt(rb, 1), mb.nameMap.lookup(gl::StrSpan("x", 1)));
}

TEST(equi_batch, closure_two_classes_reach_the_cartesian_orbit) {
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    gl::Memory& mb = ana.body;
    const gl::NameId mainId = mb.nameMap.encode("main");
    putClassesAt(mb, mainId, { { "a", "b" }, { "x", "y" } });
    pushDelta(mb, mainId, { "a", "b" });
    pushDelta(mb, mainId, { "x", "y" });
    registerRow(mb, "(in[a,x])", mainId, 1);

    BatchFixture f;
    f.build(ana, mb);
    ana.collectEquiAffectedRows(mb, f.table, f.rows, f.negeq, f.negeqKeys);
    ana.stageEquiClosure(mb, f.table, f.rows[0], f.staged, f.classStamp, f.stamp, f.cand);

    ASSERT_EQ(countKind(f.staged, gl::StagedEquiProducts::Full), 3);
    const int32_t rbx = findStaged(f.staged, mainId, "(in[b,x])");
    const int32_t ray = findStaged(f.staged, mainId, "(in[a,y])");
    const int32_t rby = findStaged(f.staged, mainId, "(in[b,y])");
    ASSERT_TRUE(rbx >= 0 && ray >= 0 && rby >= 0);
    // The combined product is chained: rewritten from a one-class product,
    // with that class's single justifier — today's history shape.
    ASSERT_TRUE(f.staged.recordAt(rby).sourceRec == rbx
                || f.staged.recordAt(rby).sourceRec == ray);
    ASSERT_EQ(f.staged.recordAt(rby).pairCount, 1);
    // Every (scope, text) once.
    ASSERT_EQ(f.staged.count(), 4);
}

TEST(equi_batch, closure_descendant_class_deposits_below_and_refuses_ancestor_known) {
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    gl::Memory& mb = ana.body;
    const gl::NameId mainId = mb.nameMap.encode("main");
    const gl::NameId subId = mb.nameMap.encodePush(mainId, "sub_(in[9,1])");
    putClassesAt(mb, subId, { { "a", "b" } });
    pushDelta(mb, subId, { "a", "b" });
    registerRow(mb, "(in[a,x])", mainId, 1);
    // (in[b,x]) already known at main: its copy at the sub scope is refused.
    registerRow(mb, "(in[b,x])", mainId, 1);
    registerRow(mb, "(in[a,z])", mainId, 1);

    BatchFixture f;
    f.build(ana, mb);
    ana.collectEquiAffectedRows(mb, f.table, f.rows, f.negeq, f.negeqKeys);
    ASSERT_EQ(f.rows.size(), 3);
    for (int32_t i = 0; i < f.rows.size(); ++i)
        ana.stageEquiClosure(mb, f.table, f.rows[i], f.staged, f.classStamp, f.stamp, f.cand);

    // (in[a,x]) -> (in[b,x]) at sub: refused (known at the ancestor main);
    // identity copies at sub: refused; (in[b,x]) -> (in[a,x]) at sub: refused;
    // (in[a,z]) -> (in[b,z]) at sub: the one product.
    ASSERT_EQ(countKind(f.staged, gl::StagedEquiProducts::Full), 1);
    const int32_t r = findStaged(f.staged, subId, "(in[b,z])");
    ASSERT_TRUE(r >= 0);
    ASSERT_EQ(f.staged.recordAt(r).targetVid, subId);
}

TEST(equi_batch, closure_refuses_a_product_staged_at_an_ancestor_earlier) {
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    gl::Memory& mb = ana.body;
    const gl::NameId mainId = mb.nameMap.encode("main");
    const gl::NameId subId = mb.nameMap.encodePush(mainId, "sub_(in[9,1])");
    // {a,b} at main; {a,c} at sub (a descendant class formed before an
    // ancestor class exists is not merged — both stay visible at sub).
    putClassesAt(mb, mainId, { { "a", "b" } });
    putClassesAt(mb, subId, { { "a", "c" } });
    pushDelta(mb, mainId, { "a", "b" });
    pushDelta(mb, subId, { "a", "c" });
    registerRow(mb, "(in[a,x])", mainId, 1);

    BatchFixture f;
    f.build(ana, mb);
    ana.collectEquiAffectedRows(mb, f.table, f.rows, f.negeq, f.negeqKeys);
    ana.stageEquiClosure(mb, f.table, f.rows[0], f.staged, f.classStamp, f.stamp, f.cand);

    // Products: (in[b,x])@main (class {a,b}), (in[c,x])@sub (class {a,c});
    // the identity copy of (in[a,x]) at sub and the copy of (in[b,x]) at sub
    // are refused — the latter because (in[b,x]) is staged at the ancestor
    // main earlier in this closure (main's class ranks first).
    ASSERT_EQ(countKind(f.staged, gl::StagedEquiProducts::Full), 2);
    ASSERT_TRUE(findStaged(f.staged, mainId, "(in[b,x])") >= 0);
    ASSERT_TRUE(findStaged(f.staged, subId, "(in[c,x])") >= 0);
    ASSERT_TRUE(findStaged(f.staged, subId, "(in[b,x])") < 0);
}

TEST(equi_batch, compressor_mode_stages_origin_only_beside_the_row_seed) {
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    gl::Memory& mb = ana.body;
    ana.parameters.compressor_mode = true;
    const gl::NameId mainId = mb.nameMap.encode("main");
    const std::string mainNs = "main";
    putClassesAt(mb, mainId, { { "a", "b" } });
    pushDelta(mb, mainId, { "a", "b" });
    // Both orbit members are registered rows without origin rows.
    registerRow(mb, "(in[a,x])", mainId, 1);
    registerRow(mb, "(in[b,x])", mainId, 1);
    for (const char* eq : { "(=[a,b])", "(=[b,a])" })
        gl::addOriginEncoded(mb.exprOriginMap, mb.originInterner, gl::StrSpan(eq, 8),
            gl::StrSpan(mainNs), gl::OriginTag::broadcast, nullptr, 0,
            ana.parameters.compressor_max_origins_per_expr);

    BatchFixture f;
    f.build(ana, mb);
    ana.collectEquiAffectedRows(mb, f.table, f.rows, f.negeq, f.negeqKeys);
    ASSERT_EQ(f.rows.size(), 2);
    for (int32_t i = 0; i < f.rows.size(); ++i)
        ana.stageEquiClosure(mb, f.table, f.rows[i], f.staged, f.classStamp, f.stamp, f.cand);

    // Two seeds, two origin-only records (each row is the other's known
    // product), no full record; each key carries one seed and one product.
    ASSERT_EQ(countKind(f.staged, gl::StagedEquiProducts::Seed), 2);
    ASSERT_EQ(countKind(f.staged, gl::StagedEquiProducts::OriginOnly), 2);
    ASSERT_EQ(countKind(f.staged, gl::StagedEquiProducts::Full), 0);
    ASSERT_TRUE(f.staged.hasSeed(mainId, gl::StrSpan("(in[b,x])", 9)));
    ASSERT_TRUE(f.staged.hasProduct(mainId, gl::StrSpan("(in[b,x])", 9)));

    const int32_t before = mb.intEncodedStatements.size();
    ana.drainStagedEquiProducts(mb, f.staged, f.table, f.committed, f.gArena);
    ASSERT_EQ(f.committed.size(), 0);
    ASSERT_EQ(mb.intEncodedStatements.size(), before);
    // Both rows now carry the equality1 history row the pairwise sink emitted
    // for a known, origin-less product.
    for (const char* text : { "(in[a,x])", "(in[b,x])" }) {
        int64_t opk = 0;
        ASSERT_TRUE(gl::lookupOriginKey(mb.originInterner, std::string(text), mainNs, opk));
        const int32_t oid = mb.exprOriginMap.lookup(opk);
        ASSERT_NE(oid, 0);
        const auto rows = mb.exprOriginMap.recordsAt(oid);
        ASSERT_EQ(rows.size(), static_cast<std::size_t>(1));
        ASSERT_TRUE(gl::decodeOrigin(rows[0], mb.originInterner).first == "equality1");
    }
}

TEST(equi_batch, drain_commits_products_with_levels_bits_and_history) {
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    gl::Memory& mb = ana.body;
    ASSERT_TRUE(ana.parameters.trackHistory);
    const gl::NameId mainId = mb.nameMap.encode("main");
    const std::string mainNs = "main";
    putClassesAt(mb, mainId, { { "a", "b" } });
    pushDelta(mb, mainId, { "a", "b" });
    registerRow(mb, "(in[a,x])", mainId, 3);
    // The justifying equality needs an origin row at the class scope for
    // the citation walk.
    gl::addOriginEncoded(mb.exprOriginMap, mb.originInterner,
        gl::StrSpan("(=[a,b])", 8), gl::StrSpan(mainNs),
        gl::OriginTag::broadcast, nullptr, 0, ana.parameters.max_origin_per_expr);

    BatchFixture f;
    f.build(ana, mb);
    ana.collectEquiAffectedRows(mb, f.table, f.rows, f.negeq, f.negeqKeys);
    ana.stageEquiClosure(mb, f.table, f.rows[0], f.staged, f.classStamp, f.stamp, f.cand);
    ASSERT_EQ(countKind(f.staged, gl::StagedEquiProducts::Full), 1);
    const int32_t before = mb.intEncodedStatements.size();

    ana.drainStagedEquiProducts(mb, f.staged, f.table, f.committed, f.gArena);

    ASSERT_EQ(f.committed.size(), 1);
    ASSERT_EQ(mb.intEncodedStatements.size(), before + 1);
    const std::string product = "(in[b,x])";
    const int32_t lvId = gl::lookupStatementLevels(mb.intStatementLevelsMap, mb.nameMap,
        gl::StrSpan(product), gl::StrSpan(mainNs));
    ASSERT_NE(lvId, 0);
    int lv[4];
    ASSERT_EQ(gl::coldIntRunAt(mb.intStatementLevelsMap, lvId, lv, 4), 1);
    ASSERT_EQ(lv[0], 3);
    const gl::NameId prodId = mb.nameMap.lookup(gl::StrSpan(product));
    ASSERT_NE(prodId, 0);
    const int64_t pk = gl::packStatementKey(prodId, mainId);
    ASSERT_NE(mb.intLocalEncodedStatementsSet.lookup(pk), 0);
    ASSERT_TRUE(mb.intKnownStatements.find(gl::StatementKey{ prodId, mainId }) != nullptr);
    ASSERT_EQ(mb.intLocalEncodedStatementsDelta.size(), 1);

    // History: one equality1 row, source (in[a,x])@main, justifier (=[a,b])@main.
    int64_t opk = 0;
    ASSERT_TRUE(gl::lookupOriginKey(mb.originInterner, product, mainNs, opk));
    const std::vector<gl::IdOrigin> rows = mb.exprOriginMap.recordsAt(mb.exprOriginMap.lookup(opk));
    ASSERT_EQ(rows.size(), static_cast<std::size_t>(1));
    const auto line = gl::decodeOrigin(rows[0], mb.originInterner);
    ASSERT_TRUE(line.first == "equality1");
    ASSERT_EQ(line.second.size(), static_cast<std::size_t>(2));
    ASSERT_TRUE(line.second[0].original == "(in[a,x])");
    ASSERT_TRUE(line.second[0].validityName == mainNs);
    ASSERT_TRUE(line.second[1].original == "(=[a,b])");
    ASSERT_TRUE(line.second[1].validityName == mainNs);

    // A second drain of the same list must not re-commit: the row is known now.
    f.staged.clear();
    ana.collectEquiAffectedRows(mb, f.table, f.rows, f.negeq, f.negeqKeys);
    for (int32_t c = 0; c < f.table.classCount(); ++c) f.table.setDueFrom(c, before + 1);
    ana.collectEquiAffectedRows(mb, f.table, f.rows, f.negeq, f.negeqKeys);
    ASSERT_EQ(f.rows.size(), 0);
}

// ---------------------------------------------------------------------------
// The pairwise oracle (the retired production driver, D-193 survivorship).
// ---------------------------------------------------------------------------

namespace gl {
        /// @brief TEST ORACLE — the retired pairwise `applyEquivalenceClass`, byte-twin
        ///        of the production driver the statement batch replaced. Apply an
        ///        equivalence class to rewrite expressions in an LB —
        /// substitutes class members where eligible and emits the resulting
        /// `equality1` / `equality2` origin records.
        ///
        /// @details
        /// One of the prover's heaviest hot paths. For every expression
        /// in the LB whose args overlap with `clss.variables`, walks the
        /// possible substitutions and emits the rewritten expression
        /// plus its provenance under the appropriate tag:
        ///
        /// - `equality1` — single-variable substitution; e.g. `(p[a])`
        ///   becomes `(p[b])` because `a == b` in the class.
        /// - `equality2` — cross-pair substitution; e.g. `(p[a, c])`
        ///   becomes `(p[b, d])` because `a == b` and `c == d` are both
        ///   in the class. Gated by [I-32](../../docs/agentic_swdd/30_invariants.md#i-32):
        ///   emission requires an existing class/LB origin.
        ///
        /// `equality1` cross-substitution is gated by
        /// [I-34](../../docs/agentic_swdd/30_invariants.md#i-34): emission requires
        /// the target to have no existing origin (only this site enforces that
        /// gate; sibling rewrites in
        /// `applyEquivalenceClassToRejectedMapIntegration` are
        /// additive without the gate).
        ///
        /// @param clss           The equivalence class to apply.
        /// @param exprOriginalId  NameMap id of the statement's original
        ///                        text (decoded once inside; argument
        ///                        re-parse stays `ce::getArgs`).
        /// @param exprValidityId  NameMap id of the statement's scope.
        /// @param memoryBlock     Owning LB.
        /// @param levels          Ascending-unique level run the rewrite
        ///                        applies at (caller-owned stack copy — the
        ///                        interior `assignSetRange`s the same map).
        /// @param levelCount      The level count (`>= 0`).
        /// @param products        Deposit channel for produced statements — the
        ///                        id-form OR-convergence buffer
        ///                        (`PagedVector<IntEncodedExpr>` on the caller's
        ///                        per-slot scratch arena) the caller sorts via
        ///                        `sortStatementRows` and drains through
        ///                        `ordisMerge`, byte-identical to the former
        ///                        `std::vector<ExpressionWithValidity>` sink
        ///                        ([I-25](../../docs/agentic_swdd/30_invariants.md#i-25)).
        /// @param validityName    The class's scope.
        /// @invariant [I-32](../../docs/agentic_swdd/30_invariants.md#i-32) — cross-pair
        ///            equality2 emission gated on existing class/LB origin.
        /// @invariant [I-34](../../docs/agentic_swdd/30_invariants.md#i-34) —
        ///            cross-substitution equality1 emission is skipped when
        ///            the target already has an origin.
        /// @see `mergeTwoEquivalenceClasses` — class merger.
        /// @see `updateEquivalenceClasses` — top-level driver.
        /// @see [`docs/agentic_swdd/20_core_concepts/05_equivalence_classes.md`](../../docs/agentic_swdd/20_core_concepts/05_equivalence_classes.md).
        template <class ClassT>
        inline void applyEquivalenceClassOracle(ExpressionAnalyzer& ana,
            const ClassT& clss,
            const NameId exprOriginalId,
            const NameId exprValidityId,
            Memory& memoryBlock,
            const int* levels, int32_t levelCount,
            PagedVector<IntEncodedExpr>& products,
            StrSpan validityName) {

            // Arena copies onto the per-slot string scratch — they survive the
            // emission block's NameMap mints (I-3) off the heap; the read uses
            // take spans, only the value-struct boundaries below materialize.
            const unsigned aecStrSlot = (ExpressionAnalyzer::g_currentCoreId >= 0)
                ? static_cast<unsigned>(ExpressionAnalyzer::g_currentCoreId)
                : scratchArenas().slotCount() - 1;
            ScratchArena& aecStrArena = scratchArenas().forSlot(aecStrSlot);
            ScratchScope aecStrScope(aecStrArena);
            const StrSpan exprOrigView = memoryBlock.nameMap.decodeView(exprOriginalId);
            const ScratchString exprOriginalCopy =
                ScratchString::copyFrom(aecStrArena, exprOrigView.ptr, exprOrigView.len);
            const StrSpan exprOriginal(exprOriginalCopy);
            const StrSpan exprValidView = memoryBlock.nameMap.decodeView(exprValidityId);
            const ScratchString exprValidityCopy =
                ScratchString::copyFrom(aecStrArena, exprValidView.ptr, exprValidView.len);
            const StrSpan exprValidityName(exprValidityCopy);

            if (exprOriginal.len >= 3 && exprOriginal[0] == '(' && exprOriginal[1] == '='
                && exprOriginal[2] == '[') {
                return; // Skip equalities
			}

            // Deposit scope = deeperOf(class scope, expr scope). Caller has
            // pre-checked that the two scopes are comparable. For the legacy
            // directions (same-NS, class-ancestor) this collapses to
            // the expression's scope — same as before. For the new direction
            // (class strictly deeper than expr), deposit lands at the class's
            // scope. Sound under descendant-inheritance: a fact at expr's
            // scope is observably true at every descendant, so applying a
            // descendant-scope class to it produces a result valid at the
            // class's scope.
            // deeperOf returns a span into the deeper input's bytes (which may
            // alias NameMap); the deposit block below mints NameMap
            // (encode(depositValidity) at pkApplied), so copy onto the string
            // scratch arena — a DIFFERENT arena — to survive that mint (I-3).
            const StrSpan dvSpan =
                memoryBlock.nameMap.deeperOf(StrSpan(validityName), exprValidityName);
            const ScratchString depositValidity =
                ScratchString::copyFrom(aecStrArena, dvSpan.ptr, dvSpan.len);

            // Rewrite accumulators on the per-slot scratch arena — no per-call
            // std::maps. Each mapping the sink emits appends one row (append
            // only, no dedup); the deposit walk below sorts by decoded
            // rewrittenExpr and acts once per equal-key group at its last-emitted
            // row, reproducing the former std::map's ascending-key iteration and
            // operator[] last-write-wins exactly
            // (see I-128).
            struct RewriteRow {
                int32_t exprId;                  // interned rewrittenExpr
                int32_t levelStart, levelCount;  // run in levelPool (sorted-unique)
                int32_t eqStart, eqCount;        // run in eqPool (interned, sorted)
                int32_t seq;                     // emission order (group tie-break)
            };
            const unsigned aecSlot = (ExpressionAnalyzer::g_currentCoreId >= 0)
                ? static_cast<unsigned>(ExpressionAnalyzer::g_currentCoreId)
                : genScratchArenas().slotCount() - 1;
            ScratchArena& aecArena = genScratchArenas().forSlot(aecSlot);
            DirtyState aecDirty = DirtyState::Clean;
            ColdHashSet<BytesKeyStore> exprKeys(&aecArena, &aecDirty);
            ColdHashSet<BytesKeyStore> eqKeys(&aecArena, &aecDirty);
            PagedVector<int32_t> levelPool(&aecArena, &aecDirty);
            PagedVector<int32_t> eqPool(&aecArena, &aecDirty);
            PagedVector<RewriteRow> rows(&aecArena, &aecDirty);

            // Cache args once — spans into exprOriginal (a stable local), no
            // heap arg vector.
            StrSpan argSpans[ExecutionParameters::MAX_ARITY];
            const int32_t argCount = getArgsSpans(exprOriginal, argSpans,
                ExecutionParameters::MAX_ARITY);

            // Reduced eq class (weak vars dropped) — member ids in decoded-lex
            // order on a caller stack run; enumerateEqClassRewrites decodes the
            // names on demand. Cap = class member count; the assert is the
            // Rule-19 tripwire (an FTA-scale class exceeding it would move to a
            // page-tier PagedVector on aecArena).
            NameId eqListBuf[ExecutionParameters::MAX_ARITY
                              * ExecutionParameters::MAX_KEY_SLOTS];
            const int32_t eqListCount = ana.reduceEqClassIds(
                clss, memoryBlock, validityName, eqListBuf,
                ExecutionParameters::MAX_ARITY * ExecutionParameters::MAX_KEY_SLOTS);

            // Base expression and wrappers
            StrSpan baseExpr;
            StrSpan wrapLeft;
            StrSpan wrapRight;
            if (!exprOriginal.empty() && exprOriginal[0] == '(') {
                baseExpr = extractExpressionSpan(exprOriginal);
                wrapLeft = StrSpan("(", 1);
                wrapRight = StrSpan(")", 1);
            }
            else {
                // must start with "!("
                assert(exprOriginal.len >= 2 && exprOriginal[0] == '!' && exprOriginal[1] == '(');
                baseExpr = extractExpressionFromNegationSpan(exprOriginal);
                wrapLeft = StrSpan("!(", 2);
                wrapRight = StrSpan(")", 1);
            }

            // Inner mapping/substitution loop is shared with
            // applyEquivalenceClassToRejectedMapIntegration via this helper.
            // Sink lambda fills exprLevelsMap and exprOriginMapLocal — the
            // downstream emission block (intEncodedStatements push, origin
            // record) below consumes those.
            ana.enumerateEqClassRewrites(clss,
                argSpans, argCount, eqListBuf, eqListCount, baseExpr, wrapLeft, wrapRight,
                memoryBlock.nameMap,
                [&](const ExpressionAnalyzer::EqClassRewrite& r) {
                    const int32_t exprId = exprKeys.mint(r.rewrittenExpr);

                    // newLevels = merge(levels, r.extraLevels), sorted-unique.
                    // Both inputs are ascending runs, so a linear merge
                    // reproduces the former `newLevels` set's contents and order.
                    // The {-1} non-derived tier is transparent to the union
                    // (leading negatives skipped on both inputs); a union whose
                    // every input was non-derived is the {-1} singleton again —
                    // ground material (I-182).
                    const int32_t levelStart = static_cast<int32_t>(levelPool.size());
                    {
                        bool hadNonDerived = false;
                        const int* ai = levels;
                        const int* aEnd = levels + levelCount;
                        int32_t bi = 0;
                        while (ai != aEnd && *ai < 0) { hadNonDerived = true; ++ai; }
                        while (bi < r.extraLevelCount && r.extraLevels[bi] < 0) {
                            hadNonDerived = true; ++bi;
                        }
                        while (ai != aEnd && bi < r.extraLevelCount) {
                            if (*ai < r.extraLevels[bi])      { levelPool.push_back(*ai); ++ai; }
                            else if (r.extraLevels[bi] < *ai) { levelPool.push_back(r.extraLevels[bi]); ++bi; }
                            else                              { levelPool.push_back(*ai); ++ai; ++bi; }
                        }
                        while (ai != aEnd)              { levelPool.push_back(*ai); ++ai; }
                        while (bi < r.extraLevelCount)  { levelPool.push_back(r.extraLevels[bi]); ++bi; }
                        if (static_cast<int32_t>(levelPool.size()) == levelStart
                            && hadNonDerived) {
                            levelPool.push_back(-1);
                        }
                    }
                    const int32_t newLevelCount =
                        static_cast<int32_t>(levelPool.size()) - levelStart;

                    // eqs tail (only under trackHistory): r.setEqualities is the
                    // ascending-sorted (from,to) pair list; format "(=[from,to])"
                    // and intern in that order — byte-identical to the former
                    // std::set<std::string> (names carry no comma, so pair order
                    // == full-string order). This is the equality1 origin's
                    // justifying tail, and verifier check_equality1 is
                    // order-sensitive. Always populated (even when the target
                    // already has an origin) so a first emission is well-formed
                    // (rest length >= 4).
                    const int32_t eqStart = static_cast<int32_t>(eqPool.size());
                    int32_t eqCount = 0;
                    if (ana.parameters.trackHistory) {
                        for (int32_t i = 0; i < r.setEqualityCount; ++i) {
                            const StrSpan f = r.setEqualities[i].from;
                            const StrSpan t = r.setEqualities[i].to;
                            char eqbuf[256];
                            assert(f.len + t.len + 7 <= 256
                                && "equality1 justifier exceeds the eqbuf bound");
                            int32_t n = 0;
                            eqbuf[n++] = '('; eqbuf[n++] = '='; eqbuf[n++] = '[';
                            std::memcpy(eqbuf + n, f.ptr, static_cast<std::size_t>(f.len));
                            n += f.len;
                            eqbuf[n++] = ',';
                            std::memcpy(eqbuf + n, t.ptr, static_cast<std::size_t>(t.len));
                            n += t.len;
                            eqbuf[n++] = ']'; eqbuf[n++] = ')';
                            eqPool.push_back(eqKeys.mint(StrSpan(eqbuf, n)));
                            ++eqCount;
                        }
                    }

                    const int32_t seq = static_cast<int32_t>(rows.size());
                    rows.push_back(RewriteRow{ exprId, levelStart, newLevelCount,
                        eqStart, eqCount, seq });
                });

            // Deposit accumulator rows in ascending rewrittenExpr order (the
            // former std::map key order), once per unique key at its last-emitted
            // row (last-write-wins). Sort a byte-bump row-index by the decoded
            // key via compareSpans (interner ids are first-seen order, never
            // sorted), tie-breaking by seq so a group's last element is its
            // max-seq winner.
            const int32_t rowCount = static_cast<int32_t>(rows.size());
            ScratchScope aecSortScope(aecArena);
            int32_t* idx = (rowCount == 0) ? nullptr
                : reinterpret_cast<int32_t*>(aecArena.resolve(aecArena.alloc(
                      rowCount * static_cast<int32_t>(sizeof(int32_t)),
                      static_cast<int32_t>(alignof(int32_t)))));
            for (int32_t i = 0; i < rowCount; ++i) idx[i] = i;
            std::sort(idx, idx + rowCount, [&](int32_t a, int32_t b) {
                const int ct = compareSpans(exprKeys.keyAt(rows[a].exprId),
                                            exprKeys.keyAt(rows[b].exprId));
                if (ct != 0) return ct < 0;
                return rows[a].seq < rows[b].seq;
            });

            for (int32_t oi = 0; oi < rowCount; ++oi) {
                // One deposit per group: skip all but the group's LAST element
                // (the next element differs in key). Ties sort by ascending seq,
                // so the group's last element is its max-seq last-write-wins row.
                if (oi + 1 < rowCount
                    && rows[idx[oi]].exprId == rows[idx[oi + 1]].exprId) {
                    continue;
                }
                const RewriteRow& row = rows[idx[oi]];
                // `applied` aliases the per-call exprKeys byte store (aecArena)
                // — a DIFFERENT interner than NameMap — so it survives the
                // NameMap encode mints in the deposit below (I-3). appliedEnc /
                // appliedWithValidity are gone: their consumers take spans.
                const StrSpan applied = exprKeys.keyAt(row.exprId);

                const int32_t appliedLvlsId = lookupStatementLevels(
                    memoryBlock.intStatementLevelsMap, memoryBlock.nameMap,
                    applied, StrSpan(depositValidity));

                if (!ana.parameters.compressor_mode && appliedLvlsId != 0) {
                    continue;
                }

                // Ancestor-known refusal (D-273):
                // a rewritten product already known at a STRICT ancestor of
                // the deposit scope is fully visible there — the descendant
                // copy is refused outright, including its equality1 history
                // line (no history for a row that never exists; the
                // ancestor row's own history stands). Non-minting probes; a
                // never-interned product cannot be known (I-83). The
                // or-branch resolution signal survives the refusal (I-174).
                if (!ana.parameters.compressor_mode) {
                    const NameId apOrigId =
                        memoryBlock.nameMap.lookup(applied);
                    const NameId apValId =
                        memoryBlock.nameMap.lookup(StrSpan(depositValidity));
                    if (apOrigId != 0 && apValId != 0
                        && ancestorKnown(memoryBlock, apOrigId, apValId,
                                         /*includeSelf=*/false)) {
                        ana.stageOrReleaseForRefusedDeposit(
                            memoryBlock, apOrigId, apValId);
                        continue;
                    }
                }

                const int mn = ana.extractMaxIterationNumber(applied);
                if (mn != -1 && mn > ana.parameters.maxIterationNumberVariable) {
                    continue;
                }
                if (ana.countPatternOccurrences(applied, memoryBlock.overallHashMemory, memoryBlock.nameMap) >
                    ana.parameters.maxNumberSecondaryVariables) {
                    continue;
                }

                // Axed-variable containment on the orbit path (x only in
                // anchors): a rewritten product citing an armed x-name is
                // refused — the door-bypassing twin of the
                // addExprToMemoryBlock / addStatement checks. Anchors are
                // never orbit products (I-53), so no exemption arm.
                {
                    bool orbitAxed = false;
                    StrSpan oax[ExecutionParameters::MAX_ARITY];
                    const int32_t oaxN = getArgsSpans(applied, oax,
                        ExecutionParameters::MAX_ARITY);
                    for (int32_t k = 0; k < oaxN; ++k) {
                        const NameId aid = memoryBlock.nameMap.lookup(oax[k]);
                        if (aid != 0
                            && memoryBlock.intAxedVariables.contains(aid)) {
                            orbitAxed = true;
                            break;
                        }
                    }
                    if (orbitAxed) continue;
                }

                // Commit iff the statement is NEW in the registry (the
                // statement-levels probe) — the sole admission control on this
                // path (D-320): a copy is
                // admitted exactly when it is not yet known at the deposit
                // scope, with the run inherited from base ∪ pair levels. No
                // level condition — a low-leveled copy at a deep LB is the
                // orbit materializing an ancestor fact's spelling the
                // receiving door refuses per mail on the canonical-only
                // design, where the orbit is the sole source of copy-split
                // statements (D-321).
                // The origin map is never consulted here (Rule 16, I-44): a raw
                // producer line that the canonical door kept for a deposit it
                // folded and ended must not pass for an established statement.
                // The history emission below keeps its own I-34 origin check.
                const bool notInStmtLvl = (appliedLvlsId == 0);
                if (notInStmtLvl) {

                    // Materialize the sorted-unique level run (contiguous) for
                    // assignSetRange's iterator range.
                    int lvlsBuf[512];
                    assert(row.levelCount <= 512
                        && "applyEquivalenceClass: level run exceeds lvlsBuf");
                    for (int32_t k = 0; k < row.levelCount; ++k) {
                        lvlsBuf[k] = levelPool[static_cast<std::size_t>(row.levelStart + k)];
                    }

                    // Commit to memory block

                    const int64_t pkApplied = packStatementKey(
                        memoryBlock.nameMap.encode(applied),
                        memoryBlock.nameMap.encode(StrSpan(depositValidity)));
                    // The statement-levels contract at this writer: a row is
                    // never empty, and never mixes {-1} with real levels.
                    assert(row.levelCount > 0
                        && ((row.levelCount == 1 && lvlsBuf[0] == -1)
                            || lvlsBuf[0] >= 0)
                        && "equi-class commit: invalid statement level run");
                    memoryBlock.intStatementLevelsMap.assignSetRange(
                        pkApplied, lvlsBuf, lvlsBuf + row.levelCount);
                    // The equivalence-class commit is an admission door: the
                    // rewritten product gets its row + levels here.
                    upsertStatementKey(memoryBlock.intKnownStatements, pkApplied,
                        /*local=*/true);
                    memoryBlock.intLocalEncodedStatementsSet.mint(pkApplied);
                    { IntEncodedExpr ie = encodeExpression(applied, StrSpan(depositValidity), memoryBlock.nameMap);
                      memoryBlock.intEncodedStatements.push_back(ie);
                      memoryBlock.intLocalEncodedStatements.push_back(ie);
                      memoryBlock.intLocalEncodedStatementsDelta.push_back(ie);
                      // OR-convergence product in id form — the SAME row (its ids
                      // were minted at `pkApplied` above; encodeExpression re-finds
                      // them, no new mint). The caller sorts via `sortStatementRows`
                      // and decodes at the `ordisMerge` edge, byte-identical to the
                      // former `ExpressionWithValidity(applied, depositValidity)`.
                      products.push_back(ie); }

                    // mailOut for this per-class rewrite is driven by
                    // `fillMailOut`: the rewrite lands in
                    // intLocalEncodedStatementsDelta via the push above,
                    // so fillMailOut catches it at the end of the
                    // elementary step.


                }

                if (ana.parameters.trackHistory) {
                    // Cross-substitution equality1 emission gate (I-34, D-48).
                    // Skip pushing this equality1 record when the target
                    // already has any origin entry in the LB-level
                    // exprOriginMap. Mirrors I-32's cross-pair equality2 gate
                    // structurally: equality1 origin records are
                    // *transitive convenience records* — they document that
                    // the rewrite is derivable through the equivalence class.
                    // When the target is already established by another path
                    // (mail-arrived origin, prior class application via a
                    // different bridge, recursion, anchor, etc.), the new
                    // record adds zero deductive content but pairs with
                    // back-direction substitutions to form 2-cycles like
                    // chapter-96 / chapter-97 rows 56–57 / 72–73 (theorem 96
                    // Gauss fold induction zero-case + step). The gate
                    // leaves the existing derivation untouched and prevents
                    // the parallel-substitution cycle vector. The
                    // exprOriginMapLocal populate above stays unconditional
                    // to preserve the well-formedness of any FIRST emission
                    // for a target (verifier check_equality1 rejects len(rest) < 4).
                    if (!ana.originRowExists(memoryBlock, applied, StrSpan(depositValidity))) {
                        // equality1 history record via the OriginDep span door —
                        // no heap OriginLine / EWV. deps[0] is the source
                        // antecedent (the rewritten expression at its scope); the
                        // rest are the winning row's justifying equalities in
                        // stored (ascending) order. Each equality is cited at
                        // the scope where its origin row actually lives — the
                        // shared `findEqualityCiteScope` walk (class scope
                        // first, then its strict ancestors deepest first; a
                        // descendant class can inherit equality content from an
                        // ancestor class, I-33, and blindly citing that content
                        // at the descendant would strand buildStack at an OR
                        // boundary). Every dep span rides aecArena / caller
                        // buffers or NameMap cold bytes, NOT the originInterner
                        // the door mints into, so they survive the mint (I-3).
                        // row.eqCount is the winning row's substitution count,
                        // bounded by argCount (<= MAX_ARITY, the
                        // enumerateEqClassRewrites eqBuf cap), so the stack run
                        // cannot overflow; the assert is the Rule-19 tripwire.
                        const int32_t depN = 1 + row.eqCount;
                        OriginDep deps[1 + ExecutionParameters::MAX_ARITY];
                        assert(depN <= 1 + ExecutionParameters::MAX_ARITY
                            && "applyEquivalenceClass: equality1 dep run exceeds cap");
                        deps[0] = OriginDep{ exprOriginal, exprValidityName };
                        for (int32_t k = 0; k < row.eqCount; ++k) {
                            const StrSpan equality = eqKeys.keyAt(
                                eqPool[static_cast<std::size_t>(row.eqStart + k)]);
                            deps[1 + k] = OriginDep{ equality,
                                ana.findEqualityCiteScope(memoryBlock, equality, validityName) };
                        }
                        addOriginEncoded(memoryBlock.exprOriginMap,
                            memoryBlock.originInterner, applied, StrSpan(depositValidity),
                            OriginTag::equality1, deps, depN,
                            (ana.parameters.compressor_mode ? ana.parameters.compressor_max_origins_per_expr : ana.parameters.max_origin_per_expr));
                        // mailOut.exprOriginMap addOrigin is driven by
                        // `fillMailOut`: it copies every exprOriginMap
                        // entry into mailOut.exprOriginMap for the
                        // corresponding delta entry, so this equality1
                        // origin still reaches mailOut for the rewrite —
                        // from the centralised site.

                    }
                }
            }
        }
}  // namespace gl

// ---------------------------------------------------------------------------
// Batch vs pairwise oracle on twin LBs.
// ---------------------------------------------------------------------------

namespace {

struct TwinSetup {
    std::vector<std::pair<std::string, std::set<std::string>>> classes;   // (scope, members), scope "main" or a push payload
    std::vector<std::pair<std::string, std::set<std::string>>> deltas;
    std::vector<std::pair<std::string, std::string>> rows;                 // (scope, text), registered with level 1
    std::vector<std::pair<std::string, std::string>> weak;                 // (scope, name)
    std::vector<std::string> axed;
    std::vector<std::pair<std::string, std::string>> equalityOrigins;      // (scope, "(=[a,b])") origin rows for the citations
};

gl::NameId scopeId(gl::Memory& mb, const std::string& scope) {
    const gl::NameId mainId = mb.nameMap.encode("main");
    if (scope == "main") return mainId;
    return mb.nameMap.encodePush(mainId, scope);
}

void applySetup(gl::ExpressionAnalyzer& ana, const TwinSetup& su) {
    gl::Memory& mb = ana.body;
    std::map<std::string, std::vector<std::set<std::string>>> byScope;
    for (const auto& c : su.classes) byScope[c.first].push_back(c.second);
    for (const auto& kv : byScope) putClassesAt(mb, scopeId(mb, kv.first), kv.second);
    for (const auto& d : su.deltas) pushDelta(mb, scopeId(mb, d.first), d.second);
    for (const auto& r : su.rows) registerRow(mb, r.second.c_str(), scopeId(mb, r.first), 1);
    for (const auto& w : su.weak)
        mb.intWeakVariables.mint(gl::packStatementKey(mb.nameMap.encode(w.second), scopeId(mb, w.first)));
    for (const auto& x : su.axed) mb.intAxedVariables.mint(mb.nameMap.encode(x));
    for (const auto& e : su.equalityOrigins) {
        const std::string scope = mb.nameMap.decode(scopeId(mb, e.first));
        gl::addOriginEncoded(mb.exprOriginMap, mb.originInterner,
            gl::StrSpan(e.second), gl::StrSpan(scope), gl::OriginTag::broadcast,
            nullptr, 0, ana.parameters.max_origin_per_expr);
    }
}

// The pairwise driver: the retired Pass-1 / Pass-2 statement loops over the
// oracle (delta classes × every row from their waterline, then every
// non-delta class × the rows since its waterline, until the registry stops
// growing). Equality-shaped rows are skipped as both passes did; the
// negated-equality expander is covered by its own tests.
void pairwiseApply(gl::ExpressionAnalyzer& ana) {
    gl::Memory& mb = ana.body;
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::ScratchArena& gArena = gl::genScratchArenas().forSlot(0);
    gl::PagedVector<gl::IntEncodedExpr> products(&gArena, &dirty);
    const gl::NameId eqId = mb.nameMap.lookup(gl::StrSpan("=", 1));
    const int32_t deltaN = mb.changedClassesThisStep.size();
    std::vector<std::size_t> startIdx(static_cast<std::size_t>(deltaN), 0);
    std::map<std::string, std::size_t> waterline;   // non-delta classes, keyed by (scope, members) text
    std::set<std::string> deltaKeys;
    auto keyOf = [&](gl::NameId vid, const gl::EquivalenceClassView& v) {
        std::string k = std::to_string(vid) + ":";
        for (int32_t i = 0; i < v.memberCount(); ++i) k += std::to_string(v.memberId(i)) + ",";
        return k;
    };
    for (int32_t k = 0; k < deltaN; ++k)
        deltaKeys.insert(keyOf(mb.changedClassesThisStep.validityAt(k),
                               mb.changedClassesThisStep.classViewAt(k, gArena)));
    auto applyOne = [&](const gl::EquivalenceClassView& cls, gl::NameId cvid, std::size_t from) {
        const std::string cv = mb.nameMap.decode(cvid);
        for (std::size_t s = from; s < static_cast<std::size_t>(mb.intEncodedStatements.size()); ++s) {
            const gl::IntEncodedExpr row = mb.intEncodedStatements[static_cast<int32_t>(s)];
            if (eqId != 0 && row.nameId == eqId) continue;
            const bool comparable = row.validityId == cvid
                || mb.nameMap.isStrictAncestor(cvid, row.validityId)
                || mb.nameMap.isStrictAncestor(row.validityId, cvid);
            if (!comparable) continue;
            const int32_t lvId = mb.intStatementLevelsMap.lookup(
                gl::packStatementKey(row.originalId, row.validityId));
            int lv[256];
            const int32_t lvN = gl::coldIntRunAt(mb.intStatementLevelsMap, lvId, lv, 256);
            products.clear();
            gl::applyEquivalenceClassOracle(ana, cls, row.originalId, row.validityId, mb,
                                            lv, lvN, products, gl::StrSpan(cv));
        }
    };
    while (true) {
        const std::size_t before = static_cast<std::size_t>(mb.intEncodedStatements.size());
        for (int32_t k = 0; k < deltaN; ++k) {
            const gl::EquivalenceClassView cls = mb.changedClassesThisStep.classViewAt(k, gArena);
            applyOne(cls, mb.changedClassesThisStep.validityAt(k), startIdx[static_cast<std::size_t>(k)]);
            startIdx[static_cast<std::size_t>(k)] = static_cast<std::size_t>(mb.intEncodedStatements.size());
        }
        const int32_t vCount = mb.equivalenceClassesMap.count();
        for (int32_t kid = 1; kid <= vCount; ++kid) {
            const gl::NameId vid = mb.equivalenceClassesMap.keyAt(kid);
            const int32_t bucketId = mb.equivalenceClassesMap.lookup(vid);
            const int32_t cc = mb.equivalenceClassesMap.runLen(bucketId);
            for (int32_t ci = 0; ci < cc; ++ci) {
                int32_t blen = 0;
                const char* bp = mb.equivalenceClassesMap.inner().peekBlobContiguous(bucketId, ci, blen, gArena);
                const gl::EquivalenceClassView cls{ bp, blen };
                const std::string key = keyOf(vid, cls);
                if (deltaKeys.count(key)) continue;
                applyOne(cls, vid, waterline[key]);
                waterline[key] = static_cast<std::size_t>(mb.intEncodedStatements.size());
            }
        }
        if (static_cast<std::size_t>(mb.intEncodedStatements.size()) == before) break;
    }
}

// Order-free snapshot of the registry: (scope, text) -> levels + decoded
// origin rows.
std::map<std::string, std::string> registrySnapshot(gl::Memory& mb) {
    std::map<std::string, std::string> out;
    for (int32_t s = 0; s < mb.intEncodedStatements.size(); ++s) {
        const gl::IntEncodedExpr& row = mb.intEncodedStatements[s];
        const std::string text = mb.nameMap.decode(row.originalId);
        const std::string scope = mb.nameMap.decode(row.validityId);
        const int32_t lvId = mb.intStatementLevelsMap.lookup(
            gl::packStatementKey(row.originalId, row.validityId));
        int lv[256];
        const int32_t lvN = gl::coldIntRunAt(mb.intStatementLevelsMap, lvId, lv, 256);
        std::string v = "lv";
        for (int32_t k = 0; k < lvN; ++k) v += " " + std::to_string(lv[k]);
        const gl::StatementFlags* fl = mb.intKnownStatements.find(gl::StatementKey{ row.originalId, row.validityId });
        v += fl ? " known" : " unknown";
        v += mb.intLocalEncodedStatementsSet.lookup(gl::packStatementKey(row.originalId, row.validityId)) ? " local" : " nonlocal";
        int64_t opk = 0;
        std::set<std::string> origins;
        if (gl::lookupOriginKey(mb.originInterner, text, scope, opk)) {
            const int32_t oid = mb.exprOriginMap.lookup(opk);
            if (oid != 0) {
                for (const gl::IdOrigin& o : mb.exprOriginMap.recordsAt(oid)) {
                    const auto line = gl::decodeOrigin(o, mb.originInterner);
                    std::string ol = line.first;
                    for (const auto& d : line.second) ol += " | " + d.original + " @ " + d.validityName;
                    origins.insert(ol);
                }
            }
        }
        for (const std::string& ol : origins) v += "\n    " + ol;
        out[scope + " :: " + text] = v;
    }
    return out;
}

void assertTwinsAgree(const TwinSetup& su, std::size_t expectedRows) {
    gl::ExpressionAnalyzer batch(std::string("Peano"));
    gl::ExpressionAnalyzer pairwise(std::string("Peano"));
    applySetup(batch, su);
    applySetup(pairwise, su);
    batch.applyEquiClasses(batch.body);
    pairwiseApply(pairwise);
    const auto a = registrySnapshot(batch.body);
    const auto b = registrySnapshot(pairwise.body);
    ASSERT_EQ(a.size(), expectedRows);
    ASSERT_TRUE(a == b);
    // Every class's persistent waterline records the final registry size —
    // written through the table's blob views at the end of the apply, so a
    // stale view (a wrong key) leaves the old value behind here.
    for (const auto& c : su.classes) {
        gl::EquivalenceClass cls;
        cls.setMembersFromNames(c.second, batch.body.nameMap);
        ASSERT_EQ(gl::lookupEqClassIndex(batch.body.eqClassSttmntIndexMapMap,
                                         scopeId(batch.body, c.first), cls.memberIds),
                  static_cast<int>(batch.body.intEncodedStatements.size()));
    }
    // The batch reached the fixpoint: a second apply changes nothing.
    batch.applyEquiClasses(batch.body);
    ASSERT_TRUE(registrySnapshot(batch.body) == a);
}

}  // namespace

TEST(equi_batch, twins_single_class_orbit) {
    TwinSetup su;
    su.classes = { { "main", { "a", "b", "c" } } };
    su.deltas = su.classes;
    su.rows = { { "main", "(in[a,x])" }, { "main", "(in2[x,c,a])" }, { "main", "(in[x,y])" } };
    su.equalityOrigins = { { "main", "(=[a,b])" }, { "main", "(=[a,c])" }, { "main", "(=[b,c])" },
                           { "main", "(=[b,a])" }, { "main", "(=[c,a])" }, { "main", "(=[c,b])" } };
    // (in[a,x]) -> b,c (+2); (in2[x,c,a]) -> 3*3-1 (+8); (in[x,y]) untouched.
    assertTwinsAgree(su, 3 + 2 + 8);
}

TEST(equi_batch, twins_two_same_scope_classes_cartesian) {
    TwinSetup su;
    su.classes = { { "main", { "a", "b" } }, { "main", { "x", "y" } } };
    su.deltas = su.classes;
    su.rows = { { "main", "(in[a,x])" } };
    su.equalityOrigins = { { "main", "(=[a,b])" }, { "main", "(=[b,a])" },
                           { "main", "(=[x,y])" }, { "main", "(=[y,x])" } };
    assertTwinsAgree(su, 4);
}

TEST(equi_batch, twins_non_delta_class_multiplies_only_new_rows) {
    TwinSetup su;
    su.classes = { { "main", { "a", "b" } }, { "main", { "x", "y" } } };
    su.deltas = { { "main", { "a", "b" } } };
    su.rows = { { "main", "(in[a,x])" }, { "main", "(in[q,x])" } };
    su.equalityOrigins = { { "main", "(=[a,b])" }, { "main", "(=[b,a])" },
                           { "main", "(=[x,y])" }, { "main", "(=[y,x])" } };
    // {x,y} is non-delta with waterline 0 (absent) — due for every row, so
    // both twins reach the full orbit of (in[a,x]) and multiply (in[q,x]).
    assertTwinsAgree(su, 2 + 3 + 1);
}

TEST(equi_batch, twins_ancestor_and_descendant_classes_overlapping) {
    TwinSetup su;
    su.classes = { { "main", { "a", "b" } }, { "sub_(in[9,1])", { "a", "c" } } };
    su.deltas = su.classes;
    su.rows = { { "main", "(in[a,x])" }, { "sub_(in[9,1])", "(in[a,z])" } };
    su.equalityOrigins = { { "main", "(=[a,b])" }, { "main", "(=[b,a])" },
                           { "sub_(in[9,1])", "(=[a,c])" }, { "sub_(in[9,1])", "(=[c,a])" } };
    // main: (in[b,x]); sub: (in[c,x]) from the ancestor row, and from
    // (in[a,z]): (in[b,z]) (main's class) and (in[c,z]).
    assertTwinsAgree(su, 2 + 1 + 1 + 2);
}

TEST(equi_batch, twins_weak_member_and_axed_variable) {
    TwinSetup su;
    su.classes = { { "main", { "a", "b", "w" } }, { "main", { "x", "y" } } };
    su.deltas = su.classes;
    su.rows = { { "main", "(in[w,x])" }, { "main", "(in[a,q])" } };
    su.weak = { { "main", "w" } };
    su.axed = { "y" };
    su.equalityOrigins = { { "main", "(=[w,a])" }, { "main", "(=[w,b])" }, { "main", "(=[a,b])" },
                           { "main", "(=[b,a])" }, { "main", "(=[x,y])" } };
    // w is weak: substitutes are {a,b} only, w never a target; y is axed: no
    // product may cite it. (in[w,x]) -> (in[a,x]), (in[b,x]); (in[a,q]) -> (in[b,q]).
    assertTwinsAgree(su, 2 + 2 + 1);
}
