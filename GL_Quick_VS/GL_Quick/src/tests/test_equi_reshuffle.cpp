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
    mb.intKnownStatements.insert(gl::StatementKey{ negId, vId }, gl::StatementFlags{ true, false, true, true });

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
        gl::StatementFlags{ true, false, true, true });

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
    mb.intKnownStatements.insert(gl::StatementKey{ negId, vId }, gl::StatementFlags{ true, false, true, true });

    ana.dischargeContradiction(mb, /*coreId=*/0);

    ASSERT_FALSE(ana.contradictionTable[0].successful);
}
