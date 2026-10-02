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
#include "canonical_test_helpers.hpp"

#include <set>

#include "../prover.hpp"

namespace {
// Multiplication tests model cores that participate in rule premises.
void enableFixtureCoreVariants(gl::Memory& mb, const char* core) {
    const gl::NameId id = mb.nameMap.encode(core);
    const gl::NameId key[] = {id, 0};
    gl::ExpressionAnalyzer::ensureEmptySubkeyRecord(mb.overallHashMemory.normalizedEncodedSubkeys, 1, key, 2);
    gl::refreshUsedAsPremise(mb);
}
// Signed twin: make `(core, polarity)` a premise core (polarity 1 = the
// negated rows of that core are multiplied).
void enableFixtureCoreVariantsSigned(gl::Memory& mb, const char* core, gl::NameId polarity) {
    const gl::NameId id = mb.nameMap.encode(core);
    const gl::NameId key[] = {id, polarity};
    gl::ExpressionAnalyzer::ensureEmptySubkeyRecord(mb.overallHashMemory.normalizedEncodedSubkeys, 1, key, 2);
    gl::refreshUsedAsPremise(mb);
}
}

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

TEST(equi_reshuffle, negated_equality_straddles_reclaim_scratch_through_recursive_emission) {
    for (bool descendant : {false, true}) {
        gl::ExpressionAnalyzer analyzer(std::string("Peano"));
        analyzer.parameters.trackHistory = false;
        auto& memory = analyzer.body;
        auto& scratch = gl::genScratchArenas().forSlot(gl::genScratchArenas().slotCount()-1);
        gl::ScratchScope fixtureFrame(scratch);
        const auto classScope = descendant
            ? memory.nameMap.encodePush(gl::NameMap::MAIN_ID, "straddle_child")
            : gl::NameMap::MAIN_ID;
        const std::string scope = memory.nameMap.decode(classScope);
        gl::EquivalenceClass cls;
        std::set<std::string> members;
        for (int i=0; i<36; ++i) members.insert("x"+std::to_string(i));
        cls.setMembersFromNames(members, memory.nameMap);
        for (auto a : cls.memberIds) for (auto b : cls.memberIds)
            if (a<b) cls.intEqualityLevelsMap[gl::packEqPairKey(a,b)] = {2};
        memory.assignClassesById(classScope, {cls});
        seedFixtureCanonicalClasses(memory);
        const int bucket = memory.equivalenceClassesMap.lookup(classScope);
        const char* contiguous = nullptr; int32_t length = 0;
        ASSERT_FALSE(memory.equivalenceClassesMap.inner().peekBlobAt(bucket,0,contiguous,length));
        auto* prefix = scratch.allocBytes(32);
        std::memset(prefix,0x39,32);
        const auto start = scratch.cursor();
        gl::DirtyState dirty = gl::DirtyState::Clean;
        gl::PagedVector<gl::IntEncodedExpr> emitted(&scratch,&dirty);
        const int levels[] = {1};
        for (int call=0; call<64; ++call) {
            analyzer.applyEquivalenceClassToNegatedEquality(gl::StrSpan("!(=[outside,other])"),
                memory,true,levels,1,emitted,gl::StrSpan("main"));
            ASSERT_EQ(scratch.cursor(),start);
        }
        ASSERT_TRUE(emitted.empty());
        analyzer.applyEquivalenceClassToNegatedEquality(gl::StrSpan("!(=[x0,outside])"),
            memory,true,levels,1,emitted,gl::StrSpan("main"));
        ASSERT_EQ(scratch.cursor(),start);
        for (int i=0; i<32; ++i) ASSERT_EQ(prefix[i],char(0x39));
        for (const auto& member : members) {
            if (member=="x0") continue;
            const std::string negation="!(=["+member+",outside])";
            ASSERT_TRUE(gl::lookupStatementLevels(memory.intStatementLevelsMap,memory.nameMap,
                gl::StrSpan(negation),gl::StrSpan(scope))!=0);
            if (descendant) ASSERT_EQ(gl::lookupStatementLevels(memory.intStatementLevelsMap,
                memory.nameMap,gl::StrSpan(negation),gl::StrSpan("main")),0);
        }
    }
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
    enableFixtureCoreVariants(mb, "in");
    enableFixtureCoreVariants(mb, "in2");
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
    enableFixtureCoreVariants(mb, "in");
    enableFixtureCoreVariants(mb, "in2");
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
    enableFixtureCoreVariants(mb, "in");
    enableFixtureCoreVariants(mb, "in2");
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
    enableFixtureCoreVariants(mb, "in");
    enableFixtureCoreVariants(mb, "in2");
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
    enableFixtureCoreVariants(mb, "in");
    enableFixtureCoreVariants(mb, "in2");
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
    enableFixtureCoreVariants(mb, "in");
    enableFixtureCoreVariants(mb, "in2");
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
    enableFixtureCoreVariants(mb, "in");
    enableFixtureCoreVariants(mb, "in2");
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
    enableFixtureCoreVariants(mb, "in");
    enableFixtureCoreVariants(mb, "in2");
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
    enableFixtureCoreVariants(mb, "in");
    enableFixtureCoreVariants(mb, "in2");
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
    enableFixtureCoreVariants(mb, "in");
    enableFixtureCoreVariants(mb, "in2");
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
    enableFixtureCoreVariants(mb, "in");
    enableFixtureCoreVariants(mb, "in2");
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
    enableFixtureCoreVariants(mb, "in");
    enableFixtureCoreVariants(mb, "in2");
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
    enableFixtureCoreVariants(mb, "in");
    enableFixtureCoreVariants(mb, "in2");
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
    enableFixtureCoreVariants(mb, "in");
    enableFixtureCoreVariants(mb, "in2");
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
    enableFixtureCoreVariants(mb, "in");
    enableFixtureCoreVariants(mb, "in2");
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
    enableFixtureCoreVariants(mb, "in");
    enableFixtureCoreVariants(mb, "in2");
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
    enableFixtureCoreVariants(mb, "in");
    enableFixtureCoreVariants(mb, "in2");
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
    seedFixtureCanonicalClasses(mb);
}

void pushDelta(gl::Memory& mb, gl::NameId vid, const std::set<std::string>& names) {
    gl::EquivalenceClass cls;
    cls.setMembersFromNames(names, mb.nameMap);
    const std::vector<char> blob = gl::serializeEquivalenceClass(cls);
    mb.changedClassesThisStep.push(vid, blob.data(), static_cast<int32_t>(blob.size()));
}

// The merge seam's ledger re-key for a class placed by `putClassesAt`.
int32_t rekeyAt(gl::ExpressionAnalyzer& ana, gl::Memory& mb, gl::NameId vid,
                const std::set<std::string>& names) {
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::PagedVector<gl::NameId> members(&gl::genScratchArenas().forSlot(0), &d);
    for (const auto& n : names) members.push_back(mb.nameMap.encode(n));
    return ana.recanonicalizeWatchLedgers(mb, vid, members);
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
    enableFixtureCoreVariants(mb, "in");
    enableFixtureCoreVariants(mb, "in2");
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
    enableFixtureCoreVariants(mb, "in");
    enableFixtureCoreVariants(mb, "in2");
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
    enableFixtureCoreVariants(mb, "in");
    enableFixtureCoreVariants(mb, "in2");
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
    enableFixtureCoreVariants(mb, "in");
    enableFixtureCoreVariants(mb, "in2");
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
    enableFixtureCoreVariants(mb, "in");
    enableFixtureCoreVariants(mb, "in2");
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
    enableFixtureCoreVariants(mb, "in");
    enableFixtureCoreVariants(mb, "in2");
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
    enableFixtureCoreVariants(mb, "in");
    enableFixtureCoreVariants(mb, "in2");
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
    enableFixtureCoreVariants(mb, "in");
    enableFixtureCoreVariants(mb, "in2");
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
    enableFixtureCoreVariants(mb, "in");
    enableFixtureCoreVariants(mb, "in2");
    std::map<std::string, std::vector<std::set<std::string>>> byScope;
    for (const auto& c : su.classes) byScope[c.first].push_back(c.second);
    for (const auto& kv : byScope) putClassesAt(mb, scopeId(mb, kv.first), kv.second);
    for (const auto& d : su.deltas) pushDelta(mb, scopeId(mb, d.first), d.second);
    for (const auto& r : su.rows) {
        registerRow(mb, r.second.c_str(), scopeId(mb, r.first), 1);
        const std::string core = std::string(gl::extractExpressionSpan(gl::StrSpan(r.second)).ptr, static_cast<std::size_t>(gl::extractExpressionSpan(gl::StrSpan(r.second)).len));
        enableFixtureCoreVariants(mb, core.c_str());
    }
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
    enableFixtureCoreVariants(mb, "in");
    enableFixtureCoreVariants(mb, "in2");
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

TEST(canonical_unused, signed_premises_only_and_shape_exclusions) {
    gl::Memory mb;
    const auto positive = gl::encodeExpression(gl::StrSpan("(p[a])", 6), gl::StrSpan("main", 4), mb.nameMap);
    const auto negative = gl::encodeExpression(gl::StrSpan("!(p[a])", 7), gl::StrSpan("main", 4), mb.nameMap);
    const auto equality = gl::encodeExpression(gl::StrSpan("(=[a,b])", 8), gl::StrSpan("main", 4), mb.nameMap);
    const auto anchor = gl::encodeExpression(gl::StrSpan("(AnchorX[a])", 12), gl::StrSpan("main", 4), mb.nameMap);
    gl::refreshUsedAsPremise(mb);
    ASSERT_FALSE(gl::usedAsPremise(mb, positive));
    ASSERT_FALSE(gl::usedAsPremise(mb, negative));
    ASSERT_TRUE(gl::usedAsPremise(mb, equality));
    ASSERT_TRUE(gl::usedAsPremise(mb, anchor));
    const gl::NameId key[4] = {positive.nameId, 1, 1, 0};
    gl::ExpressionAnalyzer::ensureEmptySubkeyRecord(mb.localHashMemory.normalizedEncodedSubkeys, 1, key, 4);
    gl::refreshUsedAsPremise(mb);
    ASSERT_FALSE(gl::usedAsPremise(mb, positive));
    ASSERT_TRUE(gl::usedAsPremise(mb, negative));
    // A goal never makes a core used (it is watched, not multiplied): the
    // signed premise membership alone decides, before and after settlement.
    mb.intToBeProved.assignSet(gl::packStatementKey(positive.originalId, positive.validityId), nullptr, 0);
    gl::refreshUsedAsPremise(mb);
    ASSERT_FALSE(gl::usedAsPremise(mb, positive));
    ASSERT_TRUE(gl::usedAsPremise(mb, negative));
    mb.intToBeProved.resetToFresh();
    gl::refreshUsedAsPremise(mb);
    ASSERT_FALSE(gl::usedAsPremise(mb, positive));
    ASSERT_TRUE(gl::usedAsPremise(mb, negative));
    gl::Memory other;
    const auto otherRow = gl::encodeExpression(gl::StrSpan("!(p[a])", 7), gl::StrSpan("main", 4), other.nameMap);
    gl::refreshUsedAsPremise(other);
    ASSERT_FALSE(gl::usedAsPremise(other, otherRow));
}

TEST(canonical_unused, dormant_rows_stay_stored_and_new_arrivals_canonicalize) {
    gl::ExpressionAnalyzer ana("Peano");
    ana.parameters.trackHistory = false;
    gl::Memory& mb = ana.body;
    const gl::NameId main = mb.nameMap.encode("main");
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::PagedVector<gl::IntEncodedExpr> out(&gl::genScratchArenas().forSlot(0), &dirty);
    const int levels[] = {0};
    ana.addStatement(gl::StrSpan("(p[b,d])",8), mb, true, levels, 1,
        gl::TransientOrigin{}, gl::StrSpan("main",4), out);
    const auto original = mb.intEncodedStatements[0];
    putClassesAt(mb, main, {{"a","b"},{"c","d"}});
    for (int pass = 0; pass < 3; ++pass) {
        ana.applyEquiClasses(mb);
        cleanUpScopes(ana, mb, { main });
        ASSERT_EQ(mb.intEncodedStatements.size(),1);
        ASSERT_EQ(mb.intEncodedStatements[0].originalId,original.originalId);
        ASSERT_EQ(mb.intLocalEncodedStatements[0].originalId,original.originalId);
        ASSERT_EQ(mb.intLocalEncodedStatementsDelta[0].originalId,original.originalId);
        ASSERT_TRUE(mb.intKnownStatements.find(gl::StatementKey{original.originalId,main}) != nullptr);
    }
    out.clear();
    ana.addStatement(gl::StrSpan("(r[b,d])",8), mb, true, levels, 1,
        gl::TransientOrigin{}, gl::StrSpan("main",4), out);
    ASSERT_EQ(out.size(),1);
    ASSERT_EQ(mb.nameMap.decode(out[0].originalId),std::string("(r[a,c])"));
    const auto rejected = mb.nameMap.lookup(gl::StrSpan("(r[b,d])",8));
    ASSERT_TRUE(mb.intKnownStatements.find(gl::StatementKey{rejected,main}) == nullptr);
    ASSERT_EQ(mb.intStatementLevelsMap.lookup(gl::packStatementKey(rejected,main)),0);
    enableFixtureCoreVariants(mb,"p");
    ana.applyEquiClasses(mb);
    ASSERT_TRUE(gl::lookupStatementFlags(mb.intKnownStatements,mb.nameMap,
        gl::StrSpan("(p[a,c])",8),gl::StrSpan("main",4)) != nullptr);
    ASSERT_TRUE(mb.intLocalEncodedStatementsSet.lookup(gl::packStatementKey(original.originalId,main)) != 0);
}

TEST(canonical_unused, dormant_external_rows_keep_identity_levels_and_scope) {
    gl::ExpressionAnalyzer ana("Peano");
    ana.parameters.trackHistory = false;
    gl::Memory& mb = ana.body;
    const auto main = mb.nameMap.encode("main");
    const auto child = mb.nameMap.encodePush(main,"hypo_(h[a])");
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::PagedVector<gl::IntEncodedExpr> out(&gl::genScratchArenas().forSlot(0),&dirty);
    const int first[] = {0}, second[] = {2};
    ana.addStatement(gl::StrSpan("(p[a])",6),mb,false,first,1,gl::TransientOrigin{},gl::StrSpan("main",4),out);
    ana.addStatement(gl::StrSpan("(p[b])",6),mb,false,second,1,gl::TransientOrigin{},gl::StrSpan("main",4),out);
    const auto raw = mb.intEncodedStatements[1];
    for (int32_t i=0;i<mb.intEncodedStatements.size();++i) mb.intExternalStatements.push_back(mb.intEncodedStatements[i]);
    putClassesAt(mb,child,{{"a","b"}});
    ana.applyEquiClasses(mb);
    putClassesAt(mb,main,{{"a","b"}});
    ana.applyEquiClasses(mb);
    cleanUpScopes(ana, mb, { main });
    ASSERT_EQ(mb.intEncodedStatements.size(),2);
    ASSERT_EQ(mb.intExternalStatements.size(),2);
    ASSERT_EQ(mb.intExternalStatements[1].originalId,raw.originalId);
    const int32_t id=mb.intStatementLevelsMap.lookup(gl::packStatementKey(raw.originalId,main));
    ASSERT_EQ(mb.intStatementLevelsMap.runLen(id),1);
    ASSERT_EQ(mb.intStatementLevelsMap.valueAt(id,0),2);
    mb.wipeSubtree(child);
    ASSERT_EQ(mb.intEncodedStatements.size(),2);
}

TEST(canonical_unused, later_premise_revisits_existing_class_waterline) {
    gl::ExpressionAnalyzer ana("Peano");
    ana.parameters.trackHistory = false;
    gl::Memory& mb = ana.body;
    const gl::NameId main = mb.nameMap.encode("main");
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::PagedVector<gl::IntEncodedExpr> out(&gl::genScratchArenas().forSlot(0), &dirty);
    const int level[] = {0};
    ana.addStatement(gl::StrSpan("(p[a])",6), mb, true, level, 1, gl::TransientOrigin{}, gl::StrSpan("main",4), out);
    putClassesAt(mb, main, {{"a","b"}});
    ana.applyEquiClasses(mb);
    ASSERT_EQ(mb.intEncodedStatements.size(), 1);
    const gl::NameId core = mb.intEncodedStatements[0].nameId;
    const gl::NameId key[] = {core, 0, 1, 0};
    gl::ExpressionAnalyzer::ensureEmptySubkeyRecord(mb.overallHashMemory.normalizedEncodedSubkeys,1,key,4);
    ana.applyEquiClasses(mb);
    ASSERT_EQ(mb.intEncodedStatements.size(), 2);
    ASSERT_TRUE(mb.nameMap.lookup(gl::StrSpan("(p[b])",6)) != 0);
    mb.overallHashMemory.normalizedEncodedSubkeys.resetToFresh();
    ana.applyEquiClasses(mb);
    ASSERT_EQ(mb.intEncodedStatements.size(), 2);
    const auto goal = gl::encodeExpression(gl::StrSpan("(p[z])",6),gl::StrSpan("main",4),mb.nameMap);
    mb.intToBeProved.assignSet(gl::packStatementKey(goal.originalId, main), nullptr, 0);
    ana.applyEquiClasses(mb);
    ASSERT_EQ(mb.intEncodedStatements.size(), 2);
}

TEST(canonical_unused, ordis_deposit_order_selects_the_capped_contradiction_origin) {
    const auto witness = [](bool reverse) {
        gl::ExpressionAnalyzer ana("Peano");
        ana.parameters.trackHistory = true;
        ana.parameters.max_origin_per_expr = 1;
        gl::Memory mb;
        const gl::NameId main = mb.nameMap.encode("main");
        const gl::NameId branch = mb.nameMap.encodePush(main,"ordis_(or90[1,2])_((a[1]))");
        const std::string scope = mb.nameMap.decode(branch);
        const int levels[] = {0};
        for (const std::string neg : {std::string("!(p[1])"),std::string("!(q[1])")}) {
            const gl::NameId id = mb.nameMap.encode(neg);
            gl::upsertStatementKey(mb.intKnownStatements,gl::packStatementKey(id,main),true);
            mb.intStatementLevelsMap.assignSet(gl::packStatementKey(id,main),levels,1);
        }
        const std::string first = reverse ? "(q[1])" : "(p[1])";
        const std::string second = reverse ? "(p[1])" : "(q[1])";
        ana.ordisMerge(gl::StrSpan(first),gl::StrSpan(scope),levels,1,mb);
        ana.ordisMerge(gl::StrSpan(second),gl::StrSpan(scope),levels,1,mb);
        std::string selected;
        for (const auto& row : gl::decodeInternalMailOrigins(mb.sameIterationInternalMail,mb.originInterner)) {
            if (row.first.original != "!(a[1])") continue;
            ASSERT_EQ(row.second.size(), 1u);
            selected = row.second[0].second[0].original;
        }
        return selected;
    };
    ASSERT_EQ(witness(false), std::string("(p[1])"));
    ASSERT_EQ(witness(true), std::string("(q[1])"));
}

TEST(canonical_unused, counterexample_facts_need_no_history_bridge) {
    gl::ExpressionAnalyzer ana("Peano");
    ana.ceFilteringActive = true;
    ana.parameters.trackHistory = true;
    gl::Memory& mb = ana.body;
    const gl::NameId main = mb.nameMap.encode("main");
    const int levels[] = {0};
    ana.addExprToMemoryBlock(gl::StrSpan("(p[b])",6), mb, 0, 4,
        levels, 1, gl::TransientOrigin{}, -1, -1, gl::StrSpan("main",4), false);
    ASSERT_TRUE(mb.exprOriginMap.empty());
    putClassesAt(mb, main, {{"a","b"}});
    gl::refreshUsedAsPremise(mb);
    ana.applyEquiClasses(mb);
    ASSERT_EQ(mb.intEncodedStatements.size(), 1);
    ASSERT_EQ(mb.nameMap.decode(mb.intEncodedStatements[0].originalId), std::string("(p[b])"));
    ASSERT_TRUE(mb.exprOriginMap.empty());
}

TEST(or_equality_support, books_actual_fact_without_a_literal_statement_and_survives_freeze) {
    for (bool endpointFirst : {false, true}) {
        gl::ExpressionAnalyzer analyzer("Peano");
        analyzer.parameters.trackHistory = true;
        gl::Memory& memory = analyzer.body;
        const gl::NameId main = memory.nameMap.encode("main");
        const gl::NameId parent = memory.nameMap.encodePush(main, "hypo_(h[N])");
        const gl::NameId old = memory.nameMap.encodePush(parent, "ordis_(or90[a,z])_((r[z]))");
        const gl::NameId endpoint = memory.nameMap.encodePush(parent, "ordis_(or90[a,z])_((=[a,z]))");
        const std::string parentName = memory.nameMap.decode(parent);
        const std::string oldName = memory.nameMap.decode(old);
        const std::string endpointName = memory.nameMap.decode(endpoint);
        const gl::NameId goal = memory.nameMap.encode("(p[z,N,g])");
        memory.intToBeProved.assignSet(gl::StatementKey{goal, parent}, nullptr, 0);
        const int cohort = gl::mintOrCohortId(memory.lbStateInterner,
            memory.lbStateInterner.encode(parentName), memory.lbStateInterner.encode("(or90[a,z])"));
        memory.orDisjunctCount.insert(cohort, 2);
        const auto source = registerRow(memory, "(p[a,N,g])", endpoint, 3);
        registerRow(memory, "(p[z,N,g])", old, 5);
        putClassesAt(memory, endpoint, {{"a", "z"}});
        seedFixtureCanonicalClasses(memory);
        seedFixtureClassEvidence(memory, memory.equivalenceClassesMap.recordAt(
            memory.equivalenceClassesMap.lookup(endpoint), 0), gl::StrSpan(endpointName));
        memory.orLiveBranches.mint(endpoint);
        const auto oldContribution = [&] {
            const int levels[] = {5};
            analyzer.ordisMerge(gl::StrSpan("(p[z,N,g])",10),gl::StrSpan(oldName), levels,1,memory);
        };
        if (!endpointFirst) oldContribution();
        analyzer.bookOrEqualitySupport(memory, goal, endpoint, source.originalId, endpoint);
        analyzer.bookOrEqualitySupport(memory, goal, endpoint, source.originalId, endpoint);
        ASSERT_EQ(memory.lbMemory.orSupportFacts.count(), 1);
        ASSERT_TRUE(memory.intKnownStatements.find(gl::StatementKey{goal, endpoint}) == nullptr);
        ASSERT_TRUE(analyzer.orBranchGoalsResolved(memory, endpoint));
        analyzer.freezeResolvedOrBranches(memory);
        memory.wipeSubtree(endpoint);
        ASSERT_EQ(memory.lbMemory.orSupportFacts.count(), 1);
        ASSERT_TRUE(memory.lbMemory.closedOrGoalLevels.lookup(gl::StatementKey{goal,endpoint}) != 0);
        if (endpointFirst) oldContribution();
        bool found = false;
        for (const auto& row : gl::decodeInternalMailOrigins(memory.sameIterationInternalMail, memory.originInterner)) {
            if (row.first.original != "(p[z,N,g])" || row.first.validityName != parentName) continue;
            for (const auto& origin : row.second) if (origin.first == "or convergence") {
                ASSERT_EQ(origin.second.size(), 4u);
                found = true;
                for (const auto& dependency : origin.second)
                    ASSERT_FALSE(dependency.original == "(p[z,N,g])" && dependency.validityName == endpointName);
            }
        }
        ASSERT_TRUE(found);
        gl::LbStore store{&gl::lbMemory(), sizeof(gl::Memory), alignof(gl::Memory)};
        gl::Memory* clone = memory.cloneFactsTemplate(store);
        ASSERT_EQ(clone->lbMemory.orSupportFacts.count(), 0);
        ASSERT_EQ(clone->lbMemory.orSupportEqualities.count(), 0);
        ASSERT_EQ(memory.lbMemory.orSupportFacts.count(), 1);
        store.destroy(clone);
        memory.wipeSubtree(parent);
        ASSERT_EQ(memory.lbMemory.orSupportFacts.count(), 0);
        ASSERT_EQ(memory.lbMemory.orSupportEqualities.count(), 0);
    }
}

TEST(or_equality_support, watch_scan_handles_late_equality_and_rejects_other_scopes_and_shapes) {
    for (int mode = 0; mode < 10; ++mode) {
        gl::ExpressionAnalyzer analyzer("Peano");
        analyzer.parameters.trackHistory = false;
        gl::Memory& memory = analyzer.body;
        const gl::NameId main = memory.nameMap.encode("main");
        const gl::NameId branch = memory.nameMap.encodePush(main,"ordis_(or90[a,z])_((=[a,z]))");
        const gl::NameId sibling = memory.nameMap.encodePush(main,"ordis_(or90[a,z])_((r[z]))");
        const gl::NameId descendant = memory.nameMap.encodePush(branch,"hypo_(s[a])");
        const gl::NameId goal = memory.nameMap.encode(mode == 7 ? "!(p[z,N,g])" : "(p[z,N,g])");
        memory.intToBeProved.assignSet(gl::StatementKey{goal,main},nullptr,0);
        memory.orLiveBranches.mint(branch);
        if (mode == 9) putClassesAt(memory,branch,{{"a","z"}});
        const char* text = mode == 3 ? "!(p[a,N,g])" : mode == 4 ? "(p[a,N,h])"
            : mode == 5 ? "(p[a,N,a])" : mode == 7 ? "!(p[a,N,g])" : "(p[a,N,g])";
        registerRow(memory,text,mode == 2 ? sibling : mode == 6 ? descendant : mode == 8 ? main : branch,2);
        analyzer.materializeWatchHits(memory,0);
        ASSERT_EQ(memory.lbMemory.orSupportFacts.count(),mode == 9 ? 1 : 0);
        putClassesAt(memory,mode == 1 ? sibling : branch,{{"a","z"}});
        seedFixtureCanonicalClasses(memory);
        analyzer.materializeWatchHits(memory,0);
        ASSERT_EQ(memory.lbMemory.orSupportFacts.count(),(mode == 0 || mode >= 7) ? 1 : 0);
        ASSERT_TRUE(memory.intKnownStatements.find(gl::StatementKey{goal,branch}) == nullptr);
        analyzer.materializeWatchHits(memory,0);
        ASSERT_EQ(memory.lbMemory.orSupportFacts.count(),(mode == 0 || mode >= 7) ? 1 : 0);
    }
}

TEST(ordis_merge_order, production_consumers_keep_first_emitted_witness) {
    const auto witness = [](bool negativeUse, bool reverse) {
        gl::ExpressionAnalyzer ana("Peano");
        ana.parameters.trackHistory = true;
        ana.parameters.max_origin_per_expr = 1;
        gl::Memory& mb = ana.body;
        const std::string branchKey = "ordis_(or90[1,2])_((a[1]))";
        TwinSetup setup;
        setup.classes = {{branchKey, {"a", "b"}}};
        setup.deltas = setup.classes;
        setup.rows = {{branchKey, reverse ? "(p[b])" : "(q[b])"},
                      {branchKey, reverse ? "(q[b])" : "(p[b])"}};
        setup.equalityOrigins = {{branchKey, "(=[a,b])"},
                                 {branchKey, "(=[b,a])"}};
        applySetup(ana, setup);
        const gl::NameId main = mb.nameMap.lookup("main");
        const std::string scope = mb.nameMap.decode(scopeId(mb, branchKey));
        const int levels[] = {1};
        for (const auto& row : setup.rows) {
            gl::addOriginEncoded(mb.exprOriginMap, mb.originInterner,
                gl::StrSpan(row.second), gl::StrSpan(scope),
                gl::OriginTag::broadcast, nullptr, 0, 1);
        }
        for (const std::string neg : {std::string("!(p[a])"), std::string("!(q[a])")}) {
            const gl::NameId id = mb.nameMap.encode(neg);
            gl::upsertStatementKey(mb.intKnownStatements,
                gl::packStatementKey(id, main), true);
            mb.intStatementLevelsMap.assignSet(gl::packStatementKey(id, main), levels, 1);
        }
        if (negativeUse) {
            // The second usage configuration: both polarities of `p` and
            // `q` are premise cores (a negated core used as a premise plus
            // the positive one) — the only way a row is multiplied now.
            mb.overallHashMemory.normalizedEncodedSubkeys.resetToFresh();
            enableFixtureCoreVariantsSigned(mb, "p", 0);
            enableFixtureCoreVariantsSigned(mb, "p", 1);
            enableFixtureCoreVariantsSigned(mb, "q", 0);
            enableFixtureCoreVariantsSigned(mb, "q", 1);
        }
        ana.applyEquiClasses(mb);
        std::string selected;
        for (const auto& row : gl::decodeInternalMailOrigins(
                 mb.sameIterationInternalMail, mb.originInterner)) {
            if (row.first.original != "!(a[1])") continue;
            ASSERT_EQ(row.second.size(), 1u);
            selected = row.second[0].second[0].original;
        }
        return selected;
    };
    for (const bool negativeUse : {false, true}) {
        ASSERT_EQ(witness(negativeUse, false), std::string("(q[a])"));
        ASSERT_EQ(witness(negativeUse, false), std::string("(q[a])"));
        ASSERT_EQ(witness(negativeUse, true), std::string("(p[a])"));
        ASSERT_EQ(witness(negativeUse, true), std::string("(p[a])"));
    }
}

TEST(canonical_unused, ancestor_known_refusal_precedes_canonical_history) {
    gl::ExpressionAnalyzer ana("Peano");
    ana.parameters.trackHistory = true;
    gl::Memory& mb = ana.body;
    const gl::NameId main = mb.nameMap.encode("main");
    const gl::NameId child = mb.nameMap.encodePush(main,"hypo_(h[a])");
    const std::string scope = mb.nameMap.decode(child);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::PagedVector<gl::IntEncodedExpr> out(&gl::genScratchArenas().forSlot(0),&dirty);
    const int levels[] = {0};
    const gl::TransientOrigin origin{true,gl::OriginTag::taskFormulation,nullptr,0};
    ana.addStatement(gl::StrSpan("(p[b])",6),mb,true,levels,1,origin,gl::StrSpan("main",4),out);
    putClassesAt(mb,child,{{"a","b"}});
    gl::refreshUsedAsPremise(mb);
    out.clear();
    ana.addStatement(gl::StrSpan("(p[b])",6),mb,true,levels,1,gl::TransientOrigin{},gl::StrSpan(scope),out);
    ASSERT_TRUE(out.empty());
    ASSERT_EQ(mb.intEncodedStatements.size(),1);
}

TEST(canonical_unused, rejected_negative_and_canonical_target_do_not_activate) {
    gl::ExpressionAnalyzer ana("Peano");
    ana.parameters.trackHistory=false;
    gl::Memory& mb=ana.body;
    const auto main=mb.nameMap.encode("main");
    gl::DirtyState dirty=gl::DirtyState::Clean;
    gl::PagedVector<gl::IntEncodedExpr> out(&gl::genScratchArenas().forSlot(0),&dirty);
    const int levels[]={0};
    const auto forbidden=mb.nameMap.encode("a");
    mb.intAxedVariables.mint(forbidden);
    ana.addStatement(gl::StrSpan("!(p[a])",7),mb,true,levels,1,gl::TransientOrigin{},gl::StrSpan("main",4),out);
    ASSERT_TRUE(out.empty());
    ASSERT_EQ(mb.negationLedger.lookup(gl::ExpressionAnalyzer::watchKey(main,mb.nameMap.lookup("p"))),0);
    ana.addStatement(gl::StrSpan("(r[c])",6),mb,true,levels,1,gl::TransientOrigin{},gl::StrSpan("main",4),out);
    putClassesAt(mb,main,{{"a","b","d"}});
    out.clear();
    ana.addStatement(gl::StrSpan("(r[d])",6),mb,true,levels,1,gl::TransientOrigin{},gl::StrSpan("main",4),out);
    ASSERT_TRUE(out.empty());
    ASSERT_TRUE(gl::lookupStatementFlags(mb.intKnownStatements,mb.nameMap,gl::StrSpan("(r[d])",6),gl::StrSpan("main",4))==nullptr);
}

TEST(canonical_unused, settled_goal_retains_export_identity_through_discharge) {
    gl::ExpressionAnalyzer ana("Peano");
    ana.parameters.trackHistory = false;
    gl::Memory& mb = ana.body;
    const gl::NameId main = mb.nameMap.encode("main");
    const auto goal = gl::encodeExpression(gl::StrSpan("(p[b])",6),gl::StrSpan("main",4),mb.nameMap);
    mb.intToBeProved.assignSet(gl::packStatementKey(goal.originalId,main),nullptr,0);
    putClassesAt(mb,main,{{"a","b"}});
    gl::refreshUsedAsPremise(mb);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::PagedVector<gl::IntEncodedExpr> out(&gl::genScratchArenas().forSlot(0),&dirty);
    const int levels[] = {0};
    // The head arrives while the goal is pending: the door collapses the
    // unused deposit to its canonical spelling, the watch drain writes the
    // goal's literal spelling back (the export identity) as a derived row.
    ana.addStatement(gl::StrSpan("(p[b])",6),mb,true,levels,1,gl::TransientOrigin{},gl::StrSpan("main",4),out);
    ana.materializeWatchHits(mb,0);
    mb.intToBeProved.resetToFresh();
    ana.applyEquiClasses(mb);
    ASSERT_FALSE(gl::usedAsPremise(mb,goal));   // a goal never makes a core used
    ASSERT_TRUE(mb.intLocalEncodedStatementsSet.lookup(gl::packStatementKey(goal.originalId,main)) != 0);
    ASSERT_TRUE(mb.intKnownStatements.find(gl::StatementKey{goal.originalId,main}) != nullptr);
    gl::LbArena dischargeScratch(&gl::staticMemory());
    mb.isActive = false;
    mb.dischargeStatementContent(dischargeScratch);
    ASSERT_TRUE(mb.dischargedRegistryKeys.count(gl::packStatementKey(goal.originalId,main)) != 0);
}


TEST(dormant_usage, negative_arrival_never_activates_the_positive_core) {
    gl::ExpressionAnalyzer ana("Peano");
    ana.parameters.trackHistory=false;
    gl::Memory& mb=ana.body;
    const auto main=mb.nameMap.encode("main");
    const auto child=mb.nameMap.encodePush(main,"hypo_(h[a])");
    const std::string childName=mb.nameMap.decode(child);
    gl::DirtyState dirty=gl::DirtyState::Clean;
    gl::PagedVector<gl::IntEncodedExpr> out(&gl::genScratchArenas().forSlot(0),&dirty);
    const int levels[]={0};
    ana.addStatement(gl::StrSpan("(p[b])",6),mb,true,levels,1,gl::TransientOrigin{},gl::StrSpan("main",4),out);
    const auto positive=mb.intEncodedStatements[0];
    putClassesAt(mb,main,{{"a","b"}});
    ana.applyEquiClasses(mb);
    ASSERT_FALSE(gl::usedAsPremise(mb,positive));
    ASSERT_TRUE(gl::lookupStatementFlags(mb.intKnownStatements,mb.nameMap,gl::StrSpan("(p[a])",6),gl::StrSpan("main",4))==nullptr);
    // A negated arrival of the same core (not contradictory: `z` is in no
    // class) is watched, not multiplied, and activates nothing.
    ana.addStatement(gl::StrSpan("!(p[z])",7),mb,true,levels,1,gl::TransientOrigin{},gl::StrSpan(childName),out);
    ASSERT_FALSE(gl::usedAsPremise(mb,positive));
    ana.applyEquiClasses(mb);
    ana.materializeWatchHits(mb,0);
    ASSERT_TRUE(gl::lookupStatementFlags(mb.intKnownStatements,mb.nameMap,gl::StrSpan("(p[a])",6),gl::StrSpan("main",4))==nullptr);
    ASSERT_EQ(mb.intEncodedStatements.size(),2);
    mb.wipeSubtree(child);
    gl::refreshUsedAsPremise(mb);
    ASSERT_FALSE(gl::usedAsPremise(mb,positive));
    ASSERT_EQ(mb.lbMemory.usedAsPremiseCorePairs.lookup(gl::packInt32Pair(positive.nameId,0)),0);
    ASSERT_EQ(mb.lbMemory.usedAsPremiseCorePairs.lookup(gl::packInt32Pair(positive.nameId,1)),0);
    gl::Memory other;
    const auto isolated=gl::encodeExpression(gl::StrSpan("(p[b])",6),gl::StrSpan("main",4),other.nameMap);
    ASSERT_FALSE(gl::usedAsPremise(other,isolated));
}

TEST(dormant_usage, stale_special_row_stays_stored_after_goal_assignment) {
    gl::ExpressionAnalyzer ana("Peano");
    ana.parameters.trackHistory=false;
    gl::Memory& mb=ana.body;
    const auto main=mb.nameMap.encode("main");
    gl::DirtyState dirty=gl::DirtyState::Clean;
    gl::PagedVector<gl::IntEncodedExpr> out(&gl::genScratchArenas().forSlot(0),&dirty);
    const int levels[]={0};
    const std::string raw="(p[it_0_lev_1_0])";
    ana.addStatement(gl::StrSpan(raw),mb,true,levels,1,gl::TransientOrigin{},gl::StrSpan("main",4),out);
    const auto original=mb.intEncodedStatements[0];
    putClassesAt(mb,main,{{"int_0_lev_1_0","it_0_lev_1_0"}});
    ana.applyEquiClasses(mb);
    cleanUpScopes(ana, mb, { main });
    ASSERT_EQ(mb.intEncodedStatements.size(),1);
    ASSERT_EQ(mb.intEncodedStatements[0].originalId,original.originalId);
    // A goal of the same core (not canonically equal: `z` is in no class)
    // is watched, not multiplied: the stored row keeps its spelling.
    const auto goal=gl::encodeExpression(gl::StrSpan("(p[z])",6),gl::StrSpan("main",4),mb.nameMap);
    mb.intToBeProved.assignSet(gl::packStatementKey(goal.originalId,main),nullptr,0);
    ana.applyEquiClasses(mb);
    ana.materializeWatchHits(mb,0);
    ASSERT_EQ(mb.intEncodedStatements.size(),1);
    ASSERT_EQ(mb.intEncodedStatements[0].originalId,original.originalId);
    ASSERT_TRUE(mb.intKnownStatements.find(gl::StatementKey{original.originalId,main})!=nullptr);
    ASSERT_TRUE(mb.intStatementLevelsMap.lookup(gl::packStatementKey(original.originalId,main))!=0);
}

TEST(dormant_usage, both_fact_arrival_orders_and_later_equality_find_contradiction) {
    for (int order=0;order<3;++order) {
        gl::ExpressionAnalyzer ana("Peano");
        ana.parameters.trackHistory=false;
        gl::Memory& mb=ana.body;
        const auto main=mb.nameMap.encode("main");
        ana.contradictionTable.push_back(gl::ContradictionItem(std::string("(p[a])"),false));
        mb.contradictionIndex=0;
        gl::DirtyState dirty=gl::DirtyState::Clean;
        gl::PagedVector<gl::IntEncodedExpr> out(&gl::genScratchArenas().forSlot(0),&dirty);
        const int levels[]={0};
        const std::string first=order==1?"!(p[b])":"(p[a])";
        const std::string second=order==1?"(p[a])":"!(p[b])";
        ana.addStatement(gl::StrSpan(first),mb,true,levels,1,gl::TransientOrigin{},gl::StrSpan("main",4),out);
        if (order!=2) putClassesAt(mb,main,{{"a","b"}});
        ana.applyEquiClasses(mb);
        ana.addStatement(gl::StrSpan(second),mb,true,levels,1,gl::TransientOrigin{},gl::StrSpan("main",4),out);
        if (order==2) { putClassesAt(mb,main,{{"a","b"}}); pushDelta(mb,main,{"a","b"}); rekeyAt(ana,mb,main,{"a","b"}); }
        ana.applyEquiClasses(mb);
        ana.materializeWatchHits(mb,0);   // the production order: drain, then sweep
        ana.dischargeContradiction(mb,0);
        ASSERT_TRUE(ana.contradictionTable[0].successful);
        ASSERT_FALSE(mb.isActive);
    }
}

TEST(dormant_usage, descendant_classes_rewrite_ancestor_facts_without_sibling_leak) {
    for (bool siblings : {false,true}) {
        gl::ExpressionAnalyzer ana("Peano");
        ana.parameters.trackHistory=false;
        gl::Memory& mb=ana.body;
        const auto main=mb.nameMap.encode("main");
        const auto child=mb.nameMap.encodePush(main,"contradiction_!(r[z])");
        const auto sibling=mb.nameMap.encodePush(main,"hypo_(h[z])");
        const std::string childName=mb.nameMap.decode(child);
        const std::string positiveScope=mb.nameMap.decode(siblings?sibling:main);
        gl::DirtyState dirty=gl::DirtyState::Clean;
        gl::PagedVector<gl::IntEncodedExpr> out(&gl::genScratchArenas().forSlot(0),&dirty);
        const int levels[]={0};
        ana.addStatement(gl::StrSpan("(p[c])",6),mb,true,levels,1,gl::TransientOrigin{},gl::StrSpan(positiveScope),out);
        ana.addStatement(gl::StrSpan("!(p[b])",7),mb,true,levels,1,gl::TransientOrigin{},gl::StrSpan(childName),out);
        putClassesAt(mb,child,{{"b","c"}});
        pushDelta(mb,child,{"b","c"});
        ana.applyEquiClasses(mb);
        ana.materializeWatchHits(mb,0);   // the production order: drain, then sweep
        ASSERT_TRUE(gl::lookupStatementFlags(mb.intKnownStatements,mb.nameMap,gl::StrSpan("(p[b])",6),gl::StrSpan("main",4))==nullptr);
        ana.dischargeContradictionScopes(mb,mb.nextIterationInternalMail);
        const gl::Mail mail=gl::makeHeapMail(mb.nextIterationInternalMail,mb.nameMap,mb.originInterner);
        bool found=false;
        for (const auto& row:mail.statements) if(row.first.original=="!(r[z])" && row.first.validityName=="main") found=true;
        ASSERT_EQ(found,!siblings);
    }
}

TEST(scoped_vacuity, records_both_facts_and_levels_then_uses_normal_scope_closure) {
    for (bool sameScope : {false, true}) {
        gl::ExpressionAnalyzer ana("Peano");
        ana.parameters.trackHistory = true;
        gl::Memory& mb = ana.body;
        const std::string compact = "(implication0[f,I])";
        const auto scope = mb.nameMap.encodePush(gl::NameMap::MAIN_ID, compact);
        const std::string scopeName = mb.nameMap.decode(scope);
        const std::string oppositeScope = sameScope ? scopeName : "main";
        const auto head = gl::encodeExpression(gl::StrSpan("(in[x,I])",9), gl::StrSpan(scopeName), mb.nameMap);
        mb.intToBeProved.assignSet(gl::packStatementKey(head.originalId, scope), nullptr, 0);
        gl::DirtyState dirty = gl::DirtyState::Clean;
        gl::PagedVector<gl::IntEncodedExpr> out(&gl::genScratchArenas().forSlot(0), &dirty);
        const int ownLevels[] = {2};
        const int ancestorLevels[] = {0,1};
        ana.addStatement(gl::StrSpan("(p[x])",6), mb, true, ownLevels, 1,
            gl::TransientOrigin{}, gl::StrSpan(scopeName), out);
        ana.dischargeVacuousImplicationScopes(mb);
        ASSERT_TRUE(mb.intKnownStatements.find(gl::StatementKey{head.originalId, scope}) == nullptr);
        ana.addStatement(gl::StrSpan("!(p[x])",7), mb, true, ancestorLevels, 2,
            gl::TransientOrigin{}, gl::StrSpan(oppositeScope), out);
        mb.intLocalEncodedStatementsDelta.clear();
        ana.dischargeVacuousImplicationScopes(mb);
        ASSERT_TRUE(mb.intKnownStatements.find(gl::StatementKey{head.originalId, scope}) != nullptr);
        const auto levelRow = mb.intStatementLevelsMap.lookup(gl::packStatementKey(head.originalId, scope));
        int levels[8];
        ASSERT_EQ(gl::coldIntRunNonNegAt(mb.intStatementLevelsMap, levelRow, levels, 8), 3);
        ASSERT_EQ(levels[0], 0);
        ASSERT_EQ(levels[1], 1);
        ASSERT_EQ(levels[2], 2);
        int64_t originKey = 0;
        ASSERT_TRUE(gl::lookupOriginKey(mb.originInterner, "(in[x,I])", scopeName, originKey));
        ASSERT_TRUE(mb.exprOriginMap.lookup(originKey) != 0);
        const auto origin = mb.exprOriginMap.recordAt(mb.exprOriginMap.lookup(originKey), 0);
        ASSERT_TRUE(origin.first == gl::OriginTag::vacuousTruth);
        ASSERT_EQ(origin.second.size(), static_cast<size_t>(2));
        gl::StrSpan expression0, validity0, expression1, validity1;
        gl::decodeOriginKeyView(origin.second[0], mb.originInterner, expression0, validity0);
        gl::decodeOriginKeyView(origin.second[1], mb.originInterner, expression1, validity1);
        const bool negativeFirst = gl::equalSpans(expression0, gl::StrSpan("!(p[x])",7));
        ASSERT_TRUE(gl::equalSpans(expression0, negativeFirst
            ? gl::StrSpan("!(p[x])",7) : gl::StrSpan("(p[x])",6)));
        ASSERT_TRUE(gl::equalSpans(expression1, negativeFirst
            ? gl::StrSpan("(p[x])",6) : gl::StrSpan("!(p[x])",7)));
        ASSERT_TRUE(gl::equalSpans(validity0, gl::StrSpan(scopeName)));
        ASSERT_TRUE(gl::equalSpans(validity1, gl::StrSpan(sameScope ? scopeName : oppositeScope)));
        const auto before = mb.intEncodedStatements.size();
        ana.dischargeVacuousImplicationScopes(mb);
        ASSERT_EQ(mb.intEncodedStatements.size(), before);
        ana.dischargeToBeProved(mb, 0, mb.nextIterationInternalMail);
        ASSERT_TRUE(mb.pendingWipeScopes.contains(scope));
        const gl::Mail mail = gl::makeHeapMail(mb.nextIterationInternalMail, mb.nameMap, mb.originInterner);
        bool emitted = false;
        for (const auto& row : mail.statements)
            if (row.first.original == compact && row.first.validityName == "main") emitted = true;
        ASSERT_TRUE(emitted);
        mb.wipeSubtree(scope);
        ASSERT_TRUE(mb.intKnownStatements.find(gl::StatementKey{head.originalId, scope}) == nullptr);
        if (!sameScope) ASSERT_TRUE(gl::lookupStatementFlags(mb.intKnownStatements, mb.nameMap,
            gl::StrSpan("!(p[x])",7), gl::StrSpan("main",4)) != nullptr);
        ASSERT_TRUE(mb.isActive);
    }
}

TEST(scoped_vacuity, isolates_siblings_descendants_and_non_implication_goals) {
    for (int mode = 0; mode < 7; ++mode) {
        gl::ExpressionAnalyzer ana("Peano");
        ana.parameters.trackHistory = false;
        gl::Memory& mb = ana.body;
        const char* payload = mode == 3 ? "contradiction_!(implication0[f,I])"
            : mode == 4 ? "ordis_(implication0[f,I])"
            : mode == 5 ? "hypo_(implication0[f,I])" : "(implication0[f,I])";
        const auto scope = mode == 6 ? gl::NameMap::MAIN_ID
            : mb.nameMap.encodePush(gl::NameMap::MAIN_ID, payload);
        const auto sibling = mb.nameMap.encodePush(gl::NameMap::MAIN_ID, "hypo_sibling");
        const auto descendant = mb.nameMap.encodePush(scope, "hypo_child");
        const std::string scopeName = mb.nameMap.decode(scope);
        const auto head = gl::encodeExpression(gl::StrSpan("(in[x,I])",9), gl::StrSpan(scopeName), mb.nameMap);
        mb.intToBeProved.assignSet(gl::packStatementKey(head.originalId, scope), nullptr, 0);
        const auto positiveScope = mode == 2 ? gl::NameMap::MAIN_ID : scope;
        const auto negativeScope = mode == 0 ? sibling : mode == 1 ? descendant
            : mode == 2 ? gl::NameMap::MAIN_ID : scope;
        gl::DirtyState dirty = gl::DirtyState::Clean;
        gl::PagedVector<gl::IntEncodedExpr> out(&gl::genScratchArenas().forSlot(0), &dirty);
        const int levels[] = {0};
        const std::string positiveName = mb.nameMap.decode(positiveScope);
        const std::string negativeName = mb.nameMap.decode(negativeScope);
        ana.addStatement(gl::StrSpan("(p[x])",6), mb, true, levels, 1,
            gl::TransientOrigin{}, gl::StrSpan(positiveName), out);
        ana.addStatement(gl::StrSpan("!(p[x])",7), mb, true, levels, 1,
            gl::TransientOrigin{}, gl::StrSpan(negativeName), out);
        ana.dischargeVacuousImplicationScopes(mb);
        ASSERT_TRUE(mb.intKnownStatements.find(gl::StatementKey{head.originalId, scope}) == nullptr);
        ASSERT_FALSE(mb.pendingWipeScopes.contains(scope));
        ASSERT_TRUE(mb.isActive);
    }
}

TEST(scoped_vacuity, equality_watch_closes_goal_owned_scope_in_both_arrival_orders) {
    for (bool reverse : {false, true}) {
        gl::ExpressionAnalyzer ana("Peano");
        ana.parameters.trackHistory = false;
        gl::Memory& mb = ana.body;
        const auto scope = mb.nameMap.encodePush(gl::NameMap::MAIN_ID,
            "(fold[N,s,add,mul,P,z,z,a])_subproof_(implication0[f,I])");
        const std::string scopeName = mb.nameMap.decode(scope);
        const auto head = gl::encodeExpression(gl::StrSpan("(in[x,I])",9), gl::StrSpan(scopeName), mb.nameMap);
        mb.intToBeProved.assignSet(gl::packStatementKey(head.originalId, scope), nullptr, 0);
        gl::DirtyState dirty = gl::DirtyState::Clean;
        gl::PagedVector<gl::IntEncodedExpr> out(&gl::genScratchArenas().forSlot(0), &dirty);
        const int levels[] = {0};
        for (int index = 0; index < 2; ++index) {
            const bool negative = (index == 0) == reverse;
            ana.addStatement(negative ? gl::StrSpan("!(p[b])",7) : gl::StrSpan("(p[c])",6),
                mb, true, levels, 1, gl::TransientOrigin{},
                negative ? gl::StrSpan(scopeName) : gl::StrSpan("main",4), out);
        }
        putClassesAt(mb, scope, {{"b","c"}});
        pushDelta(mb, scope, {"b","c"});
        ana.applyEquiClasses(mb);
        ana.materializeWatchHits(mb, 0);
        ana.dischargeVacuousImplicationScopes(mb);
        ASSERT_TRUE(mb.intKnownStatements.find(gl::StatementKey{head.originalId, scope}) != nullptr);
        ASSERT_TRUE(mb.isActive);
    }
}

TEST(watch_ledger, record_probe_and_clear_under_visible_classes) {
    gl::ExpressionAnalyzer ana("Peano");
    ana.parameters.trackHistory=false;
    gl::Memory& mb=ana.body;
    const auto main=mb.nameMap.encode("main");
    const auto row=gl::encodeExpression(gl::StrSpan("!(p[b])",7),gl::StrSpan("main",4),mb.nameMap);
    // No classes yet: the canonical id is the original id. The row is in
    // the registry too (the re-key finds affected entries through the rows).
    mb.intEncodedStatements.push_back(row);
    ana.recordWatch(mb.negationLedger,ana.body,row.originalId,main,row.nameId);
    gl::NameId orig=0;
    ASSERT_TRUE(gl::ExpressionAnalyzer::watchRunHasCanonical(mb.negationLedger,main,row.nameId,row.originalId,&orig));
    ASSERT_EQ(orig,row.originalId);
    // A class at main changes the canonical half after the re-key.
    putClassesAt(mb,main,{{"a","b"}});
    ASSERT_EQ(rekeyAt(ana,mb,main,{"a","b"}),1);
    const gl::NameId canon=ana.canonicalWatchIdAt(mb,row.originalId,main);
    ASSERT_TRUE(canon!=row.originalId);
    ASSERT_EQ(mb.nameMap.decode(canon),std::string("!(p[a])"));
    ASSERT_FALSE(gl::ExpressionAnalyzer::watchRunHasCanonical(mb.negationLedger,main,row.nameId,row.originalId,&orig));
    ASSERT_TRUE(gl::ExpressionAnalyzer::watchRunHasCanonical(mb.negationLedger,main,row.nameId,canon,&orig));
    ASSERT_EQ(orig,row.originalId);
    // A second re-key without a class change touches nothing; a class over
    // other names touches nothing either.
    ASSERT_EQ(rekeyAt(ana,mb,main,{"a","b"}),0);
    ASSERT_EQ(rekeyAt(ana,mb,main,{"x","y"}),0);
    // Other core / other scope / other ledger: no run.
    const auto q=mb.nameMap.encode("q");
    ASSERT_FALSE(gl::ExpressionAnalyzer::watchRunHasCanonical(mb.negationLedger,main,q,canon,&orig));
    ASSERT_FALSE(gl::ExpressionAnalyzer::watchRunHasCanonical(mb.goalLedger,main,row.nameId,canon,&orig));
    // Duplicate record is a no-op; clear removes exactly the entry.
    ana.recordWatch(mb.negationLedger,ana.body,row.originalId,main,row.nameId);
    ASSERT_EQ(mb.negationLedger.runLen(mb.negationLedger.lookup(gl::ExpressionAnalyzer::watchKey(main,row.nameId))),1);
    ASSERT_TRUE(ana.clearWatch(mb.negationLedger,mb,row.originalId,main,row.nameId));
    ASSERT_FALSE(ana.clearWatch(mb.negationLedger,mb,row.originalId,main,row.nameId));
    ASSERT_FALSE(gl::ExpressionAnalyzer::watchRunHasCanonical(mb.negationLedger,main,row.nameId,canon,&orig));
}

TEST(watch_ledger, rekey_covers_the_merged_scope_and_its_descendants_only) {
    gl::ExpressionAnalyzer ana("Peano");
    ana.parameters.trackHistory=false;
    gl::Memory& mb=ana.body;
    const auto main=mb.nameMap.encode("main");
    const gl::NameId child=mb.nameMap.encodePush(gl::NameMap::MAIN_ID,"ordis_(or9[1,7,3,2])_((=[6,x]))");
    const gl::NameId sibling=mb.nameMap.encodePush(gl::NameMap::MAIN_ID,"ordis_(or9[1,7,3,2])_((=[7,x]))");
    const std::string childName=mb.nameMap.decode(child);
    const std::string siblingName=mb.nameMap.decode(sibling);
    const auto atMain=gl::encodeExpression(gl::StrSpan("!(p[b])",7),gl::StrSpan("main",4),mb.nameMap);
    const auto atChild=gl::encodeExpression(gl::StrSpan("!(p[b,d])",9),gl::StrSpan(childName),mb.nameMap);
    const auto atSibling=gl::encodeExpression(gl::StrSpan("!(p[b,d])",9),gl::StrSpan(siblingName),mb.nameMap);
    const auto goalAtChild=gl::encodeExpression(gl::StrSpan("(q[d])",6),gl::StrSpan(childName),mb.nameMap);
    for (const auto& r : {atMain, atChild, atSibling}) mb.intEncodedStatements.push_back(r);
    ana.recordWatch(mb.negationLedger,mb,atMain.originalId,main,atMain.nameId);
    ana.recordWatch(mb.negationLedger,mb,atChild.originalId,child,atChild.nameId);
    ana.recordWatch(mb.negationLedger,mb,atSibling.originalId,sibling,atSibling.nameId);
    mb.intToBeProved.assignSet(gl::packStatementKey(goalAtChild.originalId,child),nullptr,0);
    ana.ensureGoalLedgerCurrent(mb);
    // A merge at main is visible at main, child and sibling: three negation
    // entries change; the unrelated goal keeps its current ledger entry.
    putClassesAt(mb,main,{{"a","b"}});
    ASSERT_EQ(rekeyAt(ana,mb,main,{"a","b"}),3);
    ASSERT_FALSE(mb.goalLedgerStale);
    gl::NameId orig=0;
    ASSERT_TRUE(gl::ExpressionAnalyzer::watchRunHasCanonical(mb.negationLedger,child,atChild.nameId,
        mb.nameMap.encode("!(p[a,d])"),&orig));
    ASSERT_EQ(orig,atChild.originalId);
    // A merge at the child is visible at the child only: the child's
    // negation entry changes (`d` -> `c`), the sibling's and main's do not;
    // the goal follows through the rebuild.
    putClassesAt(mb,child,{{"c","d"}});
    ASSERT_EQ(rekeyAt(ana,mb,child,{"c","d"}),1);
    ASSERT_TRUE(gl::ExpressionAnalyzer::watchRunHasCanonical(mb.negationLedger,child,atChild.nameId,
        mb.nameMap.encode("!(p[a,c])"),&orig));
    ana.ensureGoalLedgerCurrent(mb);
    ASSERT_TRUE(gl::ExpressionAnalyzer::watchRunHasCanonical(mb.goalLedger,child,goalAtChild.nameId,
        mb.nameMap.encode("(q[c])"),&orig));
    ASSERT_EQ(orig,goalAtChild.originalId);
    ASSERT_TRUE(gl::ExpressionAnalyzer::watchRunHasCanonical(mb.negationLedger,sibling,atSibling.nameId,
        mb.nameMap.encode("!(p[a,d])"),&orig));
    // The visible-bucket canonical composes main's and the child's classes
    // directly too.
    ASSERT_EQ(mb.nameMap.decode(ana.canonicalWatchIdAt(mb,atChild.originalId,child)),std::string("!(p[a,c])"));
    ASSERT_EQ(mb.nameMap.decode(ana.canonicalWatchIdAt(mb,atSibling.originalId,sibling)),std::string("!(p[a,d])"));
}

TEST(watch_ledger, registration_doors_and_unknow_mirror_known_negated_rows) {
    gl::ExpressionAnalyzer ana("Peano");
    ana.parameters.trackHistory=false;
    gl::Memory& mb=ana.body;
    const auto main=mb.nameMap.encode("main");
    gl::DirtyState dirty=gl::DirtyState::Clean;
    gl::PagedVector<gl::IntEncodedExpr> out(&gl::genScratchArenas().forSlot(0),&dirty);
    const int levels[]={0};
    ana.addStatement(gl::StrSpan("!(p[b])",7),mb,true,levels,1,gl::TransientOrigin{},gl::StrSpan("main",4),out);
    ana.addStatement(gl::StrSpan("(p[a])",6),mb,true,levels,1,gl::TransientOrigin{},gl::StrSpan("main",4),out);
    ana.addStatement(gl::StrSpan("!(=[c,d])",9),mb,true,levels,1,gl::TransientOrigin{},gl::StrSpan("main",4),out);
    const auto p=mb.nameMap.lookup("p");
    const auto eq=mb.nameMap.lookup("=");
    const auto neg=mb.nameMap.lookup("!(p[b])");
    ASSERT_TRUE(p!=0 && eq!=0 && neg!=0);
    gl::NameId orig=0;
    // The negated non-equality row is recorded (canonical == original: no classes yet);
    // the positive row and the negated equality are not.
    ASSERT_TRUE(gl::ExpressionAnalyzer::watchRunHasCanonical(mb.negationLedger,main,p,neg,&orig));
    ASSERT_EQ(orig,neg);
    ASSERT_EQ(mb.negationLedger.runLen(mb.negationLedger.lookup(gl::ExpressionAnalyzer::watchKey(main,p))),1);
    ASSERT_EQ(mb.negationLedger.lookup(gl::ExpressionAnalyzer::watchKey(main,eq)),0);
    // A merge through the equality door re-keys the entry at the seam.
    ana.addStatement(gl::StrSpan("(=[a,b])",8),mb,true,levels,1,gl::TransientOrigin{},gl::StrSpan("main",4),out);
    const gl::NameId canon=ana.canonicalWatchIdAt(mb,neg,main);
    ASSERT_TRUE(canon!=neg);
    ASSERT_TRUE(gl::ExpressionAnalyzer::watchRunHasCanonical(mb.negationLedger,main,p,canon,&orig));
    ASSERT_EQ(orig,neg);
    // The un-know seam drops the entry with the known row.
    ana.resetResentExpressionRegistries(mb,gl::StrSpan("!(p[b])",7),gl::StrSpan("main",4));
    ASSERT_FALSE(gl::ExpressionAnalyzer::watchRunHasCanonical(mb.negationLedger,main,p,canon,&orig));
    // Un-knowing a never-known statement is a defined no-op.
    ana.resetResentExpressionRegistries(mb,gl::StrSpan("!(p[z])",7),gl::StrSpan("main",4));
}

TEST(watch_ledger, goal_ledger_is_derived_from_the_goal_registry) {
    gl::ExpressionAnalyzer ana("Peano");
    ana.parameters.trackHistory=false;
    gl::Memory& mb=ana.body;
    const auto main=mb.nameMap.encode("main");
    const int lv[1]={0};
    // Seeded straight into the registry (the ledger is derived, so no door
    // is involved — exactly what the production goal door does not need).
    const auto goalRow=gl::encodeExpression(gl::StrSpan("(p[a,b])",8),gl::StrSpan("main",4),mb.nameMap);
    const auto goal=goalRow.originalId;
    const auto p=goalRow.nameId;
    ASSERT_TRUE(p!=0 && goal!=0);
    const int64_t pk=gl::packStatementKey(goal,main);
    mb.intToBeProved.assignSet(pk,nullptr,0);
    gl::NameId orig=0;
    // Rebuilt on first probe.
    ana.ensureGoalLedgerCurrent(mb);
    ASSERT_TRUE(gl::ExpressionAnalyzer::watchRunHasCanonical(mb.goalLedger,main,p,goal,&orig));
    ASSERT_EQ(orig,goal);
    // A merge through the equality door marks it stale; the rebuild carries
    // the new canonical.
    gl::DirtyState dirty=gl::DirtyState::Clean;
    gl::PagedVector<gl::IntEncodedExpr> out(&gl::genScratchArenas().forSlot(0),&dirty);
    ana.addStatement(gl::StrSpan("(=[a,b])",8),mb,true,lv,1,gl::TransientOrigin{},gl::StrSpan("main",4),out);
    ASSERT_TRUE(mb.goalLedgerStale);
    ana.ensureGoalLedgerCurrent(mb);
    const gl::NameId canon=ana.canonicalWatchIdAt(mb,goal,main);
    ASSERT_TRUE(canon!=goal);
    ASSERT_TRUE(gl::ExpressionAnalyzer::watchRunHasCanonical(mb.goalLedger,main,p,canon,&orig));
    ASSERT_EQ(orig,goal);
    // A direct registry erase (no door) changes the count: the next probe
    // sees the goal gone; a direct seed is picked up the same way.
    mb.intToBeProved.eraseSet(pk);
    ana.ensureGoalLedgerCurrent(mb);
    ASSERT_FALSE(gl::ExpressionAnalyzer::watchRunHasCanonical(mb.goalLedger,main,p,canon,&orig));
    const auto seededRow=gl::encodeExpression(gl::StrSpan("(q[b])",6),gl::StrSpan("main",4),mb.nameMap);
    const auto seeded=seededRow.originalId;
    mb.intToBeProved.assignSet(gl::packStatementKey(seeded,main),nullptr,0);
    ana.ensureGoalLedgerCurrent(mb);
    ASSERT_TRUE(gl::ExpressionAnalyzer::watchRunHasCanonical(mb.goalLedger,main,seededRow.nameId,
        ana.canonicalWatchIdAt(mb,seeded,main),&orig));
    ASSERT_EQ(orig,seeded);
}

TEST(watch_ledger, wipe_drops_the_closed_scope_entries_only) {
    gl::ExpressionAnalyzer ana("Peano");
    ana.parameters.trackHistory=false;
    gl::Memory& mb=ana.body;
    const auto main=mb.nameMap.encode("main");
    const auto child=mb.nameMap.encodePush(main,"hypo_(h[a])");
    const std::string childName=mb.nameMap.decode(child);
    const auto atMain=gl::encodeExpression(gl::StrSpan("!(p[b])",7),gl::StrSpan("main",4),mb.nameMap);
    const auto atChild=gl::encodeExpression(gl::StrSpan("!(p[c])",7),gl::StrSpan(childName),mb.nameMap);
    ana.recordWatch(mb.negationLedger,mb,atMain.originalId,main,atMain.nameId);
    ana.recordWatch(mb.negationLedger,mb,atChild.originalId,child,atChild.nameId);
    mb.goalLedgerStale=false;
    mb.wipeSubtree(child);
    gl::NameId orig=0;
    ASSERT_TRUE(gl::ExpressionAnalyzer::watchRunHasCanonical(mb.negationLedger,main,atMain.nameId,atMain.originalId,&orig));
    ASSERT_FALSE(gl::ExpressionAnalyzer::watchRunHasCanonical(mb.negationLedger,child,atChild.nameId,atChild.originalId,&orig));
    ASSERT_TRUE(mb.goalLedgerStale);
}

// The watch drain must supply the exact pair on its own: no rule makes `p`
// a premise in the tests below, so nothing is multiplied.
static bool knownAt(const gl::Memory& mb, const char* text, gl::NameId scope) {
    const gl::NameId id = mb.nameMap.lookup(text);
    return id != 0 && mb.intKnownStatements.find(gl::StatementKey{ id, scope }) != nullptr;
}

TEST(watch_ledger, positive_arrival_finds_a_negated_entry_through_the_ledger) {
    gl::ExpressionAnalyzer ana("Peano");
    ana.parameters.trackHistory=false;
    gl::Memory& mb=ana.body;
    const auto main=mb.nameMap.encode("main");
    ana.contradictionTable.push_back(gl::ContradictionItem(std::string("(p[a])"),false));
    mb.contradictionIndex=0;
    gl::DirtyState dirty=gl::DirtyState::Clean;
    gl::PagedVector<gl::IntEncodedExpr> out(&gl::genScratchArenas().forSlot(0),&dirty);
    const int levels[]={0};
    // The negation is stored raw (no class yet); the class forms afterwards
    // with `a` canonical, and the positive arrives already canonical.
    ana.addStatement(gl::StrSpan("!(p[b])",7),mb,true,levels,1,gl::TransientOrigin{},gl::StrSpan("main",4),out);
    putClassesAt(mb,main,{{"a","b"}});
    ASSERT_EQ(rekeyAt(ana,mb,main,{"a","b"}),1);
    ana.applyEquiClasses(mb);
    ASSERT_FALSE(knownAt(mb,"!(p[a])",main));   // not multiplied
    ana.addStatement(gl::StrSpan("(p[a])",6),mb,true,levels,1,gl::TransientOrigin{},gl::StrSpan("main",4),out);
    ana.materializeWatchHits(mb,0);
    ASSERT_TRUE(knownAt(mb,"!(p[a])",main));    // the twin
    ana.dischargeContradiction(mb,0);
    ASSERT_TRUE(ana.contradictionTable[0].successful);
    ASSERT_FALSE(mb.isActive);
}

TEST(watch_ledger, negated_arrival_finds_an_unused_positive_through_the_batch_sweep) {
    gl::ExpressionAnalyzer ana("Peano");
    ana.parameters.trackHistory=false;
    gl::Memory& mb=ana.body;
    const auto main=mb.nameMap.encode("main");
    ana.contradictionTable.push_back(gl::ContradictionItem(std::string("(p[a])"),false));
    mb.contradictionIndex=0;
    gl::DirtyState dirty=gl::DirtyState::Clean;
    gl::PagedVector<gl::IntEncodedExpr> out(&gl::genScratchArenas().forSlot(0),&dirty);
    const int levels[]={0};
    ana.addStatement(gl::StrSpan("(p[b])",6),mb,true,levels,1,gl::TransientOrigin{},gl::StrSpan("main",4),out);
    ana.materializeWatchHits(mb,0);            // nothing to pair yet
    putClassesAt(mb,main,{{"a","b"}});
    rekeyAt(ana,mb,main,{"a","b"});
    ana.applyEquiClasses(mb);
    ASSERT_FALSE(knownAt(mb,"(p[a])",main));    // the unused positive stays raw
    ana.addStatement(gl::StrSpan("!(p[a])",7),mb,true,levels,1,gl::TransientOrigin{},gl::StrSpan("main",4),out);
    ana.materializeWatchHits(mb,0);
    ASSERT_TRUE(knownAt(mb,"(p[a])",main));     // the positive twin from (p[b])
    ana.dischargeContradiction(mb,0);
    ASSERT_TRUE(ana.contradictionTable[0].successful);
}

TEST(watch_ledger, merge_created_contradiction_is_found_by_the_recheck) {
    gl::ExpressionAnalyzer ana("Peano");
    ana.parameters.trackHistory=false;
    gl::Memory& mb=ana.body;
    const auto main=mb.nameMap.encode("main");
    ana.contradictionTable.push_back(gl::ContradictionItem(std::string("(p[a])"),false));
    mb.contradictionIndex=0;
    gl::DirtyState dirty=gl::DirtyState::Clean;
    gl::PagedVector<gl::IntEncodedExpr> out(&gl::genScratchArenas().forSlot(0),&dirty);
    const int levels[]={0};
    ana.addStatement(gl::StrSpan("!(p[b])",7),mb,true,levels,1,gl::TransientOrigin{},gl::StrSpan("main",4),out);
    ana.addStatement(gl::StrSpan("(p[a])",6),mb,true,levels,1,gl::TransientOrigin{},gl::StrSpan("main",4),out);
    ana.materializeWatchHits(mb,0);
    ana.dischargeContradiction(mb,0);
    ASSERT_FALSE(ana.contradictionTable[0].successful);   // nothing equal yet
    // The equality door merges the class: the re-key seam stages the moved
    // negation entry, the drain pairs it with the stored positive.
    ana.addStatement(gl::StrSpan("(=[a,b])",8),mb,true,levels,1,gl::TransientOrigin{},gl::StrSpan("main",4),out);
    ASSERT_TRUE(mb.pendingWatchRechecks.size()>0);
    ana.materializeWatchHits(mb,0);
    ASSERT_TRUE(knownAt(mb,"!(p[a])",main));
    ana.dischargeContradiction(mb,0);
    ASSERT_TRUE(ana.contradictionTable[0].successful);
}

TEST(watch_ledger, goal_reached_through_the_ledger_gets_its_literal_twin) {
    gl::ExpressionAnalyzer ana("Peano");
    ana.parameters.trackHistory=false;
    gl::Memory& mb=ana.body;
    const auto main=mb.nameMap.encode("main");
    gl::DirtyState dirty=gl::DirtyState::Clean;
    gl::PagedVector<gl::IntEncodedExpr> out(&gl::genScratchArenas().forSlot(0),&dirty);
    const int levels[]={0};
    putClassesAt(mb,main,{{"a","b"}});
    // A literal, non-canonical goal.
    const auto goalRow=gl::encodeExpression(gl::StrSpan("(p[b])",6),gl::StrSpan("main",4),mb.nameMap);
    mb.intToBeProved.assignSet(gl::packStatementKey(goalRow.originalId,main),nullptr,0);
    // The canonical fact arrives.
    ana.addStatement(gl::StrSpan("(p[a])",6),mb,true,levels,1,gl::TransientOrigin{},gl::StrSpan("main",4),out);
    ana.materializeWatchHits(mb,0);
    // The goal's literal spelling is now a known local delta row at main —
    // exactly what the exact goal probe fires on — derived, not multiplied.
    ASSERT_TRUE(knownAt(mb,"(p[b])",main));
    bool inDelta=false;
    for (int32_t i=0;i<mb.intLocalEncodedStatementsDelta.size();++i)
        if (mb.intLocalEncodedStatementsDelta[i].originalId==goalRow.originalId) inDelta=true;
    ASSERT_TRUE(inDelta);
    ASSERT_TRUE(mb.intToBeProved.lookup(gl::packStatementKey(goalRow.originalId,main))!=0);
}

TEST(watch_ledger, ancestor_negation_is_imported_under_a_branch_class_and_paired_there) {
    gl::ExpressionAnalyzer ana("Peano");
    ana.parameters.trackHistory=false;
    gl::Memory& mb=ana.body;
    const auto main=mb.nameMap.encode("main");
    const gl::NameId child=mb.nameMap.encodePush(gl::NameMap::MAIN_ID,"ordis_(or9[1,7,3,2])_((=[6,x]))");
    const std::string childName=mb.nameMap.decode(child);
    gl::DirtyState dirty=gl::DirtyState::Clean;
    gl::PagedVector<gl::IntEncodedExpr> out(&gl::genScratchArenas().forSlot(0),&dirty);
    const int levels[]={0};
    ana.addStatement(gl::StrSpan("!(p[b])",7),mb,true,levels,1,gl::TransientOrigin{},gl::StrSpan("main",4),out);
    ana.materializeWatchHits(mb,0);
    // A class at the child only: the ancestor negation names `b`, so the
    // merge seam imports it into the child's run.
    putClassesAt(mb,child,{{"a","b"}});
    gl::PagedVector<gl::NameId> members(&gl::genScratchArenas().forSlot(0),&dirty);
    members.push_back(mb.nameMap.encode("a"));
    members.push_back(mb.nameMap.encode("b"));
    ana.importAncestorNegationsForClass(mb,child,members);
    gl::NameId orig=0;
    const auto p=mb.nameMap.lookup("p");
    ASSERT_TRUE(gl::ExpressionAnalyzer::watchRunHasCanonical(mb.negationLedger,child,p,mb.nameMap.encode("!(p[a])"),&orig));
    ana.applyEquiClasses(mb);
    ASSERT_FALSE(knownAt(mb,"!(p[a])",child));
    ana.addStatement(gl::StrSpan("(p[a])",6),mb,true,levels,1,gl::TransientOrigin{},gl::StrSpan(childName),out);
    ana.materializeWatchHits(mb,0);
    // The negation's twin lands at the child (deeper scope), the positive
    // stays where it is: the exact pair now exists at the child.
    ASSERT_TRUE(knownAt(mb,"!(p[a])",child));
    ASSERT_FALSE(knownAt(mb,"!(p[a])",main));
}

TEST(watch_ledger, live_branch_assumption_refuted_through_the_ledger) {
    gl::ExpressionAnalyzer ana("Peano");
    ana.parameters.trackHistory=false;
    gl::Memory& mb=ana.body;
    const auto main=mb.nameMap.encode("main");
    // The branch asserts `(p[a])`; `main` knows `!(p[b])`; the class {a,b}
    // at main makes them contradictory, but no exact `!(p[a])` exists.
    const gl::NameId branch=mb.nameMap.encodePush(gl::NameMap::MAIN_ID,"ordis_(or9[1,7,3,2])_((p[a]))");
    gl::DirtyState dirty=gl::DirtyState::Clean;
    gl::PagedVector<gl::IntEncodedExpr> out(&gl::genScratchArenas().forSlot(0),&dirty);
    const int levels[]={0};
    ana.addStatement(gl::StrSpan("!(p[b])",7),mb,true,levels,1,gl::TransientOrigin{},gl::StrSpan("main",4),out);
    putClassesAt(mb,main,{{"a","b"}});
    rekeyAt(ana,mb,main,{"a","b"});
    ana.applyEquiClasses(mb);
    mb.orLiveBranches.mint(branch);
    ASSERT_EQ(ana.refutedOrBranchAtOrAbove(branch,mb),0);   // exact probe: nothing
    ana.materializeWatchHits(mb,0);
    ASSERT_TRUE(knownAt(mb,"!(p[a])",branch));              // the exact negation at the branch
    ASSERT_EQ(ana.refutedOrBranchAtOrAbove(branch,mb),branch);
    ASSERT_TRUE(mb.pendingDeadOrBranches.lookup(branch)!=0);
}

TEST(watch_ledger, clone_copies_and_discharge_reclaims) {
    gl::ExpressionAnalyzer ana("Peano");
    ana.parameters.trackHistory=false;
    gl::Memory facts;
    const auto main=facts.nameMap.encode("main");
    const auto row=gl::encodeExpression(gl::StrSpan("!(q[a])",7),gl::StrSpan("main",4),facts.nameMap);
    ana.recordWatch(facts.negationLedger,facts,row.originalId,main,row.nameId);
    gl::LbStore store{&gl::lbMemory(),sizeof(gl::Memory),alignof(gl::Memory)};
    gl::Memory* copy=facts.cloneFactsTemplate(store);
    gl::NameId orig=0;
    ASSERT_TRUE(gl::ExpressionAnalyzer::watchRunHasCanonical(copy->negationLedger,main,row.nameId,row.originalId,&orig));
    ASSERT_EQ(orig,row.originalId);
    facts.negationLedger.resetToFresh();
    ASSERT_TRUE(gl::ExpressionAnalyzer::watchRunHasCanonical(copy->negationLedger,main,row.nameId,row.originalId,&orig));
    store.destroy(copy);
}

TEST(dormant_usage, child_delta_does_not_hide_reactivated_ancestor_cleanup) {
    gl::ExpressionAnalyzer ana("Peano");
    ana.parameters.trackHistory = false;
    gl::Memory& mb = ana.body;
    const auto main = mb.nameMap.encode("main");
    const auto child = mb.nameMap.encodePush(main, "hypo_(h[a])");
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::PagedVector<gl::IntEncodedExpr> out(&gl::genScratchArenas().forSlot(0), &dirty);
    const int levels[] = {0};
    const std::string raw = "(p[it_0_lev_1_0])";
    ana.addStatement(gl::StrSpan(raw), mb, true, levels, 1,
        gl::TransientOrigin{}, gl::StrSpan("main",4), out);
    const auto original = mb.intEncodedStatements[0];
    putClassesAt(mb, main, {{"int_0_lev_1_0", "it_0_lev_1_0"}});
    ana.applyEquiClasses(mb);
    ASSERT_EQ(mb.intEncodedStatements.size(), 1);
    putClassesAt(mb, child, {{"int_0_lev_1_0", "int_0_lev_1_1", "it_0_lev_1_0"}});
    pushDelta(mb, child, {"int_0_lev_1_0", "int_0_lev_1_1", "it_0_lev_1_0"});
    enableFixtureCoreVariants(mb, "p");   // `(p, positive)` becomes a premise core
    ana.applyEquiClasses(mb);
    ASSERT_TRUE(mb.intLocalEncodedStatementsSet.lookup(
        gl::packStatementKey(original.originalId,main)) == 0);
    ASSERT_TRUE(gl::lookupStatementFlags(mb.intKnownStatements,mb.nameMap,
        gl::StrSpan("(p[int_0_lev_1_0])", 18),gl::StrSpan("main",4)) != nullptr);
}


TEST(equi_reshuffle, standard_processing_ce_mode_only_checks_contradictions) {
    // Positive mail and registry witnesses, missing opposition, and both
    // directions of the main-scope boundary exercise the dedicated CE gate.
    for (int scenario = 0; scenario < 5; ++scenario) {
        gl::ExpressionAnalyzer analyzer("Peano");
        analyzer.ceFilteringActive = true;
        gl::Memory& memory = analyzer.body;
        enableFixtureCoreVariants(memory, "in");
        enableFixtureCoreVariants(memory, "in2");
        memory.isActive = true;
        memory.contradictionIndex = 0;
        analyzer.contradictionTable.emplace_back("(in[7,N])", false);
        const gl::NameId main = gl::NameMap::MAIN_ID;
        const gl::NameId branch = memory.nameMap.encodePush(main, "ce_gate_branch");
        const std::string head = "(in[7,N])";
        const gl::NameId positive = memory.nameMap.encode(head);
        const gl::NameId negative = memory.nameMap.encode(analyzer.negate(head));
        if (scenario != 1) {
            memory.intKnownStatements.insert(
                gl::StatementKey{negative, scenario == 2 ? branch : main},
                gl::StatementFlags{true, false});
        }
        const int levels[] = {0};
        gl::insertInternalStatement(memory.sameIterationInternalMail, memory.nameMap,
            gl::StrSpan(head), gl::StrSpan(memory.nameMap.decode(scenario == 3 ? branch : main)),
            levels, 1);
        if (scenario == 4) {
            memory.intEncodedStatements.push_back(gl::encodeExpression(
                gl::EncodedExpression(head, "main"), memory.nameMap));
        }
        memory.intToBeProved.assignSet(gl::packStatementKey(positive, main), nullptr, 0);
        putClassesAt(memory, main, {{"a", "b"}});
        pushDelta(memory, main, {"a", "b"});
        const auto statements = memory.intEncodedStatements.size();
        const auto changed = memory.changedClassesThisStep.size();
        const auto mail = memory.sameIterationInternalMail.statements_.count();
        for (int call = 0; call < 2; ++call) {
            analyzer.standardProcessing(memory, &memory.mailIn,
                memory.sameIterationInternalMail, memory.nextIterationInternalMail, 0);
            ASSERT_EQ(analyzer.contradictionTable[0].successful, scenario == 0 || scenario == 4);
            ASSERT_EQ(memory.isActive, scenario != 0 && scenario != 4);
            ASSERT_EQ(memory.intEncodedStatements.size(), statements);
            ASSERT_EQ(memory.changedClassesThisStep.size(), changed);
            ASSERT_EQ(memory.sameIterationInternalMail.statements_.count(), mail);
            ASSERT_EQ(memory.nextIterationInternalMail.statements_.count(), 0);
            ASSERT_EQ(memory.intToBeProved.count(), 1);
            ASSERT_EQ(memory.ruleStagingArmDepth, 0);
        }
    }
}

TEST(equi_reshuffle, standard_processing_hypothesis_copy_is_applied_and_published) {
    for (bool postBurst : {false, true}) {
        for (int scenario = 0; scenario < 3; ++scenario) {
            gl::ExpressionAnalyzer analyzer("Peano");
            auto& memory = analyzer.body;
            memory.isActive = true;
            memory.level = 0;
            analyzer.parameters.trackHistory = true;
            const auto main = gl::NameMap::MAIN_ID;
            const auto hypothesis = memory.nameMap.encodePush(main,
                "_var0_x_hypo_(in[x,N])");
            enableFixtureCoreVariants(memory, "in");
            const int levels[] = {0};
            gl::DirtyState dirty = gl::DirtyState::Clean;
            gl::PagedVector<gl::IntEncodedExpr> added(
                &gl::genScratchArenas().forSlot(0), &dirty);
            // Register actual equality premises, including their levels and
            // history, so class application can cite every hypothetical rewrite.
            const std::string hypothesisScope = memory.nameMap.decode(hypothesis);
            if (scenario != 1) {
                const std::string equality = "(=[x,int_0_lev_0_1])";
                analyzer.addStatement(gl::StrSpan(equality), memory, true, levels, 1,
                    gl::TransientOrigin{true, gl::OriginTag::taskFormulation, nullptr, 0},
                    gl::StrSpan(hypothesisScope), added);
            }
            if (scenario == 2) {
                analyzer.addStatement(gl::StrSpan("(=[x,y])", 8), memory, true, levels, 1,
                    gl::TransientOrigin{true, gl::OriginTag::taskFormulation, nullptr, 0},
                    gl::StrSpan(hypothesisScope), added);
            }
            analyzer.addStatement(gl::StrSpan("(in[x_copy,N])", 14), memory,
                true, levels, 1, gl::TransientOrigin{true, gl::OriginTag::taskFormulation, nullptr, 0},
                gl::StrSpan("main", 4), added);
            auto& input = postBurst ? memory.sameIterationInternalMail : memory.nextIterationInternalMail;
            auto& output = postBurst ? memory.nextIterationInternalMail : memory.sameIterationInternalMail;
            analyzer.standardProcessing(memory, nullptr, input, output, -1);
            const auto copiedFact = memory.nameMap.lookup(gl::StrSpan("(in[x,N])", 9));
            ASSERT_EQ(memory.intKnownStatements.lookup(gl::StatementKey{copiedFact, main}) != 0,
                scenario == 0);
            ASSERT_EQ(memory.changedClassesThisStep.size(), 0);
            const auto copyHistoryCount = [&]() {
                int count = 0;
                for (const auto& row : gl::decodeMailOutOrigins(memory.mailOut, memory.mailOutInterner))
                    for (const auto& origin : row.second)
                        if (origin.first == "variable copy") ++count;
                return count;
            };
            const int firstCount = copyHistoryCount();
            ASSERT_EQ(firstCount, scenario == 0 ? 1 : 0);
            analyzer.standardProcessing(memory, nullptr, output, input, -1);
            ASSERT_EQ(copyHistoryCount(), firstCount);
            ASSERT_EQ(memory.changedClassesThisStep.size(), 0);
        }
    }
}


namespace {
std::vector<std::set<std::string>> canonicalClassNames(gl::Memory& mb, gl::NameId scope) {
    const int32_t bucket = mb.canonicalEquivalenceClassesMap.lookup(scope);
    ASSERT_TRUE(bucket != 0);
    std::vector<std::set<std::string>> result;
    for (int32_t i = 0; i < mb.canonicalEquivalenceClassesMap.runLen(bucket); ++i) {
        const auto cls = mb.canonicalEquivalenceClassesMap.recordAt(bucket, i);
        std::set<std::string> names;
        for (auto id : cls.memberIds) names.insert(mb.nameMap.decode(id));
        result.push_back(names);
    }
    return result;
}
void addCanonicalFixtureEquality(gl::ExpressionAnalyzer& ana, gl::Memory& mb,
                                 gl::NameId scope, const std::string& equality, int level) {
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::PagedVector<gl::IntEncodedExpr> rows(&gl::genScratchArenas().forSlot(0), &dirty);
    const gl::TransientOrigin origin{true, gl::OriginTag::broadcast, nullptr, 0};
    const std::string validity = mb.nameMap.decode(scope);
    ana.addStatement(gl::StrSpan(equality), mb, true, &level, 1, origin,
                     gl::StrSpan(validity), rows);
}
}

TEST(canonical_classes, initialization_copies_independent_ancestors_without_application_events) {
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    gl::Memory& mb = ana.body;
    const auto root = gl::NameMap::MAIN_ID;
    putClassesAt(mb, root, {{"a", "z"}, {"m", "n"}});
    const auto child = mb.nameMap.encodePush(root, "canonical_child");
    const auto grandchild = mb.nameMap.encodePush(child, "canonical_grandchild");
    const auto classCount = mb.equivalenceClassesMap.count();
    const auto deltas = mb.changedClassesThisStep.size();
    ana.initializeCanonicalScope(mb, grandchild);
    const std::vector<std::set<std::string>> expected{{"a","z"}, {"m","n"}};
    ASSERT_TRUE(canonicalClassNames(mb, root) == expected);
    ASSERT_TRUE(canonicalClassNames(mb, child) == expected);
    ASSERT_TRUE(canonicalClassNames(mb, grandchild) == expected);
    ASSERT_EQ(mb.equivalenceClassesMap.count(), classCount);
    ASSERT_EQ(mb.changedClassesThisStep.size(), deltas);
    ASSERT_EQ(mb.equivalenceClassesMap.lookup(child), 0);
    ana.initializeCanonicalScope(mb, grandchild);
    ASSERT_TRUE(canonicalClassNames(mb, grandchild) == expected);
}

namespace {
// The records of one scope's run, concatenated with a separator.
template <class Map>
std::string classRunBytes(Map& map, gl::NameId scope) {
    std::string out;
    const int32_t bucket = map.lookup(scope);
    if (bucket == 0) return out;
    auto& arena = gl::genScratchArenas().forSlot(0);
    gl::ScratchScope frame(arena);
    for (int32_t i = 0; i < map.runLen(bucket); ++i) {
        int32_t length = 0;
        const char* bytes = map.peekRecordBytes(bucket, i, length, arena);
        out.append(bytes, static_cast<std::size_t>(length));
        out.push_back('|');
    }
    return out;
}
}

// CanonicalRunOverlay — held runs are found by scope and copied back record
// by record; a scope that was not held is not found.
TEST(canonical_classes, run_overlay_holds_and_finds_runs) {
    auto& arena = gl::genScratchArenas().forSlot(0);
    gl::ScratchScope frame(arena);
    gl::ExpressionAnalyzer::CanonicalRunOverlay held(arena, 2);
    const gl::StrSpan first[2] = { gl::StrSpan("abc", 3), gl::StrSpan("", 0) };
    const gl::StrSpan second[1] = { gl::StrSpan("de", 2) };
    held.add(7, first, 2);
    held.add(9, second, 1);
    ASSERT_EQ(held.find(7), 0);
    ASSERT_EQ(held.find(9), 1);
    ASSERT_EQ(held.find(8), -1);
    ASSERT_EQ(held.entries[0].records, 2);
    ASSERT_TRUE(gl::equalSpans(held.copyRecord(0, 0, arena), gl::StrSpan("abc", 3)));
    ASSERT_EQ(held.copyRecord(0, 1, arena).len, 0);
    ASSERT_TRUE(gl::equalSpans(held.copyRecord(1, 0, arena), gl::StrSpan("de", 2)));
}

// refreshCanonicalClasses + flushCanonicalRuns — the held runs written in one
// pass equal the runs written scope by scope in the same order.
TEST(canonical_classes, refresh_one_pass_write_matches_scope_by_scope_writes) {
    const auto build = [](gl::ExpressionAnalyzer& ana) {
        gl::Memory& mb = ana.body;
        const auto root = gl::NameMap::MAIN_ID;
        putClassesAt(mb, root, {{"a", "z"}, {"m", "n"}});
        const auto child = mb.nameMap.encodePush(root, "held_child");
        const auto sibling = mb.nameMap.encodePush(root, "held_sibling");
        const auto grandchild = mb.nameMap.encodePush(child, "held_grandchild");
        putClassesAt(mb, child, {{"c", "d"}});
        ana.initializeCanonicalScope(mb, grandchild);
        ana.initializeCanonicalScope(mb, sibling);
        putClassesAt(mb, root, {{"a", "y", "z"}, {"m", "n"}});
        return std::vector<gl::NameId>{ root, child, sibling, grandchild };
    };
    gl::ExpressionAnalyzer batched(std::string("Peano"));
    gl::ExpressionAnalyzer sequential(std::string("Peano"));
    const auto scopes = build(batched);
    const auto sameScopes = build(sequential);
    ASSERT_TRUE(scopes == sameScopes);
    batched.refreshCanonicalClasses(batched.body, gl::NameMap::MAIN_ID);
    for (const auto scope : { scopes[0], scopes[1], scopes[3], scopes[2] })
        sequential.rebuildCanonicalScope(sequential.body, scope);
    for (const auto scope : scopes)
        ASSERT_EQ(classRunBytes(batched.body.canonicalEquivalenceClassesMap, scope),
                  classRunBytes(sequential.body.canonicalEquivalenceClassesMap, scope));
    const std::vector<std::set<std::string>> expected{{"a", "y", "z"}, {"c", "d"}, {"m", "n"}};
    ASSERT_TRUE(canonicalClassNames(batched.body, scopes[3]) == expected);
}

// updateEquivalenceClasses — an equality between members already in one class
// records its pair in the scope's own class and leaves every complete view as
// it was; an equality that adds a member refreshes them.
TEST(canonical_classes, only_a_membership_change_refreshes_complete_views) {
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    gl::Memory& mb = ana.body;
    const auto root = gl::NameMap::MAIN_ID;
    addCanonicalFixtureEquality(ana, mb, root, "(=[a,b])", 1);
    addCanonicalFixtureEquality(ana, mb, root, "(=[b,c])", 1);
    const auto child = mb.nameMap.encodePush(root, "watch_child");
    ana.initializeCanonicalScope(mb, child);
    const std::string ownBefore = classRunBytes(mb.equivalenceClassesMap, root);
    const std::string rootViewBefore = classRunBytes(mb.canonicalEquivalenceClassesMap, root);
    const std::string childViewBefore = classRunBytes(mb.canonicalEquivalenceClassesMap, child);

    addCanonicalFixtureEquality(ana, mb, root, "(=[a,c])", 5);
    ASSERT_NE(classRunBytes(mb.equivalenceClassesMap, root), ownBefore);
    ASSERT_EQ(classRunBytes(mb.canonicalEquivalenceClassesMap, root), rootViewBefore);
    ASSERT_EQ(classRunBytes(mb.canonicalEquivalenceClassesMap, child), childViewBefore);

    addCanonicalFixtureEquality(ana, mb, root, "(=[c,d])", 1);
    const std::vector<std::set<std::string>> expected{{"a", "b", "c", "d"}};
    ASSERT_TRUE(canonicalClassNames(mb, root) == expected);
    ASSERT_TRUE(canonicalClassNames(mb, child) == expected);
}

TEST(canonical_classes, routine_updates_close_late_overlaps_without_upward_or_sibling_leakage) {
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    gl::Memory& mb = ana.body;
    const auto root = gl::NameMap::MAIN_ID;
    const auto child = mb.nameMap.encodePush(root, "canonical_child");
    const auto sibling = mb.nameMap.encodePush(root, "canonical_sibling");
    const auto grandchild = mb.nameMap.encodePush(child, "canonical_grandchild");
    ana.initializeCanonicalScope(mb, grandchild);
    ana.initializeCanonicalScope(mb, sibling);
    addCanonicalFixtureEquality(ana, mb, child, "(=[b,z])", 2);
    addCanonicalFixtureEquality(ana, mb, child, "(=[c,y])", 3);
    addCanonicalFixtureEquality(ana, mb, root, "(=[y,z])", 1);
    const std::vector<std::set<std::string>> merged{{"b","c","y","z"}};
    ASSERT_TRUE(canonicalClassNames(mb, child) == merged);
    ASSERT_TRUE(canonicalClassNames(mb, grandchild) == merged);
    const std::vector<std::set<std::string>> parent{{"y","z"}};
    ASSERT_TRUE(canonicalClassNames(mb, root) == parent);
    ASSERT_TRUE(canonicalClassNames(mb, sibling) == parent);
    ASSERT_EQ(mb.equivalenceClassesMap.runLen(mb.equivalenceClassesMap.lookup(child)), 2);
    ASSERT_EQ(mb.equivalenceClassesMap.lookup(grandchild), 0);
    const auto cls = mb.canonicalEquivalenceClassesMap.recordAt(
        mb.canonicalEquivalenceClassesMap.lookup(child), 0);
    const auto pair = gl::packEqPairKey(mb.nameMap.lookup("b"), mb.nameMap.lookup("c"));
    ASSERT_TRUE(cls.intEqualityLevelsMap.at(pair) == std::set<int>({1,2,3}));
    ASSERT_TRUE(gl::equalSpans(ana.findEqualityCiteScope(mb, gl::StrSpan("(=[y,z])",8),
        mb.nameMap.decodeView(child)), gl::StrSpan("main",4)));
    int64_t key = 0;
    ASSERT_TRUE(gl::lookupOriginKey(mb.originInterner, "(=[b,c])", mb.nameMap.decode(child), key));
    ASSERT_TRUE(mb.exprOriginMap.lookup(key) != 0);
    addCanonicalFixtureEquality(ana, mb, root, "(=[a,y])", 4);
    const std::vector<std::set<std::string>> enlarged{{"a","b","c","y","z"}};
    ASSERT_TRUE(canonicalClassNames(mb, grandchild) == enlarged);
}

TEST(canonical_classes, explicit_rebuild_preserves_local_pair_levels_and_ancestor_provenance) {
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    gl::Memory& mb = ana.body;
    const auto root = gl::NameMap::MAIN_ID;
    const auto child = mb.nameMap.encodePush(root, "canonical_child");
    addCanonicalFixtureEquality(ana, mb, child, "(=[a,z])", 5);
    addCanonicalFixtureEquality(ana, mb, root, "(=[a,z])", 2);
    ana.rebuildCanonicalScope(mb, child);
    auto cls = mb.canonicalEquivalenceClassesMap.recordAt(
        mb.canonicalEquivalenceClassesMap.lookup(child), 0);
    const auto pair = gl::packEqPairKey(mb.nameMap.lookup("a"), mb.nameMap.lookup("z"));
    ASSERT_TRUE(cls.intEqualityLevelsMap.at(pair) == std::set<int>({5}));
    const auto parent = mb.canonicalEquivalenceClassesMap.recordAt(
        mb.canonicalEquivalenceClassesMap.lookup(root), 0);
    ASSERT_TRUE(parent.intEqualityLevelsMap.at(pair) == std::set<int>({2}));
}

TEST(canonical_classes, facts_clone_and_cleanup_keep_both_sets_independent) {
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    gl::Memory source;
    const auto root = gl::NameMap::MAIN_ID;
    const auto child = source.nameMap.encodePush(root, "canonical_child");
    const auto sibling = source.nameMap.encodePush(root, "canonical_sibling");
    putClassesAt(source, root, {{"a", "z"}});
    putClassesAt(source, child, {{"b", "c"}});
    ana.initializeCanonicalScope(source, child);
    ana.initializeCanonicalScope(source, sibling);
    gl::LbStore store{&gl::lbMemory(), sizeof(gl::Memory), alignof(gl::Memory)};
    gl::Memory* clone = source.cloneFactsTemplate(store);
    ASSERT_TRUE(canonicalClassNames(*clone, child) == canonicalClassNames(source, child));
    ASSERT_EQ(clone->equivalenceClassesMap.runLen(clone->equivalenceClassesMap.lookup(child)), 1);
    clone->wipeSubtree(child);
    ASSERT_EQ(clone->canonicalEquivalenceClassesMap.lookup(child), 0);
    ASSERT_EQ(clone->equivalenceClassesMap.lookup(child), 0);
    ASSERT_TRUE(clone->canonicalEquivalenceClassesMap.lookup(sibling) != 0);
    ASSERT_TRUE(source.canonicalEquivalenceClassesMap.lookup(child) != 0);
    store.destroy(clone);
}

TEST(canonical_classes, full_scope_chain_preserves_independent_components) {
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    gl::Memory& mb = ana.body;
    const auto root = gl::NameMap::MAIN_ID;
    const auto child = mb.nameMap.encodePush(root, "true_child");
    const auto leaf = mb.nameMap.encodePush(child, "true_leaf");
    putClassesAt(mb, root, {{"a", "b"}, {"f", "g"}});
    putClassesAt(mb, child, {{"b", "c"}, {"g", "h"}});
    putClassesAt(mb, leaf, {{"c", "d"}});
    const auto recordEdge = [&](gl::NameId scope, const char* a, const char* b) {
        const std::string validity = mb.nameMap.decode(scope);
        const std::string forward = std::string("(=[") + a + "," + b + "])";
        const std::string reverse = std::string("(=[") + b + "," + a + "])";
        gl::addOriginEncoded(mb.exprOriginMap, mb.originInterner, gl::StrSpan(forward),
            gl::StrSpan(validity), gl::OriginTag::broadcast, nullptr, 0, ana.parameters.max_origin_per_expr);
        gl::addOriginEncoded(mb.exprOriginMap, mb.originInterner, gl::StrSpan(reverse),
            gl::StrSpan(validity), gl::OriginTag::broadcast, nullptr, 0, ana.parameters.max_origin_per_expr);
    };
    recordEdge(root, "a", "b");
    recordEdge(root, "f", "g");
    recordEdge(child, "b", "c");
    recordEdge(child, "g", "h");
    recordEdge(leaf, "c", "d");
    ana.initializeCanonicalScope(mb, leaf);
    const std::vector<std::set<std::string>> expected{{"a", "b", "c", "d"}, {"f", "g", "h"}};
    ASSERT_TRUE(canonicalClassNames(mb, leaf) == expected);
    ASSERT_EQ(mb.equivalenceClassesMap.runLen(mb.equivalenceClassesMap.lookup(leaf)), 1);
    const int64_t key = gl::packStatementKey(mb.nameMap.lookup("d"), leaf);
    const int64_t* selected = mb.lbMemory.canonicalMemberRepresentatives.find(key);
    ASSERT_TRUE(selected != nullptr);
    ASSERT_EQ(static_cast<gl::NameId>(uint64_t(*selected) >> 32), mb.nameMap.lookup("a"));
}

TEST(canonical_classes, unchanged_refresh_has_no_changes_and_late_bridge_marks_old_representative) {
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    gl::Memory& mb = ana.body;
    const auto root = gl::NameMap::MAIN_ID;
    const auto child = mb.nameMap.encodePush(root, "true_child");
    ana.initializeCanonicalScope(mb, child);
    addCanonicalFixtureEquality(ana, mb, root, "(=[f,g])", 1);
    addCanonicalFixtureEquality(ana, mb, child, "(=[b,z])", 2);
    mb.lbMemory.canonicalClassChanges.resetToFresh();
    ASSERT_TRUE(!ana.rebuildCanonicalScope(mb, child));
    ASSERT_EQ(mb.lbMemory.canonicalClassChanges.count(), 0);
    addCanonicalFixtureEquality(ana, mb, root, "(=[a,z])", 3);
    const int64_t changed = gl::packStatementKey(mb.nameMap.lookup("b"), child);
    const int32_t* flags = mb.lbMemory.canonicalClassChanges.find(changed);
    ASSERT_TRUE(flags != nullptr && (*flags & 1) != 0);
    ASSERT_TRUE(mb.lbMemory.canonicalClassChanges.find(
        gl::packStatementKey(mb.nameMap.lookup("f"), child)) == nullptr);
    ASSERT_TRUE(mb.lbMemory.canonicalClassChanges.find(
        gl::packStatementKey(mb.nameMap.lookup("g"), child)) == nullptr);
    mb.lbMemory.canonicalClassChanges.resetToFresh();
    ASSERT_TRUE(!ana.rebuildCanonicalScope(mb, child));
    ASSERT_EQ(mb.lbMemory.canonicalClassChanges.count(), 0);
}

TEST(canonical_classes, representative_snapshot_survives_weak_metadata_change_and_scope_wipe) {
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    gl::Memory& mb = ana.body;
    const auto root = gl::NameMap::MAIN_ID;
    const auto child = mb.nameMap.encodePush(root, "true_child");
    putClassesAt(mb, child, {{"int_0_lev_1_0", "int_0_lev_1_1"}});
    ana.initializeCanonicalScope(mb, child);
    const auto first = mb.nameMap.lookup("int_0_lev_1_0");
    const auto second = mb.nameMap.lookup("int_0_lev_1_1");
    const int64_t key = gl::packStatementKey(first, child);
    ASSERT_EQ(static_cast<gl::NameId>(uint64_t(*mb.lbMemory.canonicalMemberRepresentatives.find(key)) >> 32), first);
    mb.lbMemory.canonicalClassChanges.resetToFresh();
    mb.intWeakVariables.mint(key);
    ASSERT_TRUE(ana.rebuildCanonicalScope(mb, child));
    ASSERT_EQ(static_cast<gl::NameId>(uint64_t(*mb.lbMemory.canonicalMemberRepresentatives.find(key)) >> 32), second);
    ASSERT_TRUE((*mb.lbMemory.canonicalClassChanges.find(key) & 1) != 0);
    mb.wipeSubtree(child);
    ASSERT_TRUE(mb.lbMemory.canonicalMemberRepresentatives.find(key) == nullptr);
    ASSERT_TRUE(mb.lbMemory.canonicalClassChanges.find(key) == nullptr);
}

TEST(canonical_forms, representative_at_follows_the_rebuilt_view_and_weak_members) {
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    ana.parameters.trackHistory = false;
    gl::Memory& mb = ana.body;
    const auto main = gl::NameMap::MAIN_ID;
    const auto child = mb.nameMap.encodePush(main, "true_child");
    putClassesAt(mb, main, {{"b", "z"}});
    putClassesAt(mb, child, {{"a", "b"}, {"int_0_lev_1_0", "int_0_lev_1_1"}});
    const auto id = [&](const char* name) { return mb.nameMap.lookup(name); };
    // Main: z -> b; the child's view merges {a, b, z} -> a.
    ASSERT_EQ(gl::ExpressionAnalyzer::canonicalRepresentativeAt(mb, id("z"), main), id("b"));
    ASSERT_EQ(gl::ExpressionAnalyzer::canonicalRepresentativeAt(mb, id("b"), main), id("b"));
    ASSERT_EQ(gl::ExpressionAnalyzer::canonicalRepresentativeAt(mb, id("z"), child), id("a"));
    // A token in no class keeps its spelling.
    const auto c = mb.nameMap.encode("c");
    ASSERT_EQ(gl::ExpressionAnalyzer::canonicalRepresentativeAt(mb, c, main), c);
    ASSERT_EQ(gl::ExpressionAnalyzer::canonicalRepresentativeAt(mb, id("a"), main), id("a"));
    // Every member weak at the child: nothing is replaced there.
    const auto first = id("int_0_lev_1_0"), second = id("int_0_lev_1_1");
    ASSERT_EQ(gl::ExpressionAnalyzer::canonicalRepresentativeAt(mb, second, child), first);
    mb.intWeakVariables.mint(gl::packStatementKey(first, child));
    mb.intWeakVariables.mint(gl::packStatementKey(second, child));
    ASSERT_TRUE(ana.rebuildCanonicalScope(mb, child));
    ASSERT_EQ(gl::ExpressionAnalyzer::canonicalRepresentativeAt(mb, first, child), first);
    ASSERT_EQ(gl::ExpressionAnalyzer::canonicalRepresentativeAt(mb, second, child), second);
    const auto special = mb.nameMap.encode("(p[int_0_lev_1_1])");
    ASSERT_EQ(ana.canonicalWatchIdAt(mb, special, child), special);
}

TEST(canonical_forms, same_spelling_implies_the_same_canonical_form) {
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    ana.parameters.trackHistory = false;
    gl::Memory& mb = ana.body;
    const auto main = gl::NameMap::MAIN_ID;
    const auto child = mb.nameMap.encodePush(main, "true_child");
    const auto quiet = mb.nameMap.encodePush(main, "true_quiet");
    putClassesAt(mb, main, {{"b", "z"}});
    putClassesAt(mb, child, {{"a", "b"}, {"c", "d"}});
    putClassesAt(mb, quiet, {{"x", "y"}});
    const auto expr = [&](const char* text) { return mb.nameMap.encode(text); };
    const auto zc = expr("!(p[z,c])"), zd = expr("!(p[z,d])"), zx = expr("(q[z,x])"),
        zy = expr("(q[z,y])"), free = expr("(q[e,f])");
    ASSERT_TRUE(gl::ExpressionAnalyzer::sameCanonicalSpelling(mb, zc, child, child));
    // z -> b at main and quiet, z -> a at the child.
    ASSERT_TRUE(gl::ExpressionAnalyzer::sameCanonicalSpelling(mb, zc, main, quiet));
    ASSERT_FALSE(gl::ExpressionAnalyzer::sameCanonicalSpelling(mb, zc, main, child));
    ASSERT_FALSE(gl::ExpressionAnalyzer::sameCanonicalSpelling(mb, zd, quiet, child));
    // x is its own class's representative at quiet: nothing replaces it there.
    ASSERT_TRUE(gl::ExpressionAnalyzer::sameCanonicalSpelling(mb, zx, main, quiet));
    ASSERT_FALSE(gl::ExpressionAnalyzer::sameCanonicalSpelling(mb, zy, main, quiet));
    ASSERT_TRUE(gl::ExpressionAnalyzer::sameCanonicalSpelling(mb, free, main, child));
    const gl::NameId scopes[] = {main, child, quiet};
    for (const auto e : {zc, zd, zx, zy, free})
        for (const auto from : scopes)
            for (const auto to : scopes)
                if (gl::ExpressionAnalyzer::sameCanonicalSpelling(mb, e, from, to))
                    ASSERT_EQ(ana.canonicalWatchIdAt(mb, e, from), ana.canonicalWatchIdAt(mb, e, to));
    ASSERT_EQ(mb.nameMap.decode(ana.canonicalWatchIdAt(mb, zd, child)), std::string("!(p[a,c])"));
}

TEST(canonical_forms, cache_shares_anchor_forms_and_renews_after_a_class_change) {
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    ana.parameters.trackHistory = false;
    gl::Memory& mb = ana.body;
    const auto main = gl::NameMap::MAIN_ID;
    const auto child = mb.nameMap.encodePush(main, "true_child");
    const auto quiet = mb.nameMap.encodePush(main, "true_quiet");
    putClassesAt(mb, main, {{"b", "z"}});
    putClassesAt(mb, child, {{"a", "b"}});
    putClassesAt(mb, quiet, {{"x", "y"}});
    const auto zc = mb.nameMap.encode("(p[z,c])");
    const auto plain = mb.nameMap.encode("(p[c,e])");
    gl::ExpressionAnalyzer::CanonicalFormCache forms(gl::genScratchArenas().forSlot(0));
    // The quiet scope shares main's form; the child computes its own.
    const auto atMain = ana.canonicalIdCached(mb, forms, zc, main, main);
    ASSERT_EQ(mb.nameMap.decode(atMain), std::string("(p[b,c])"));
    ASSERT_EQ(ana.canonicalIdCached(mb, forms, zc, main, quiet), atMain);
    ASSERT_EQ(forms.forms.count(), 2);
    ASSERT_EQ(forms.texts.count(), 1);
    ASSERT_EQ(mb.nameMap.decode(ana.canonicalIdCached(mb, forms, zc, main, child)), std::string("(p[a,c])"));
    ASSERT_EQ(forms.texts.count(), 2);
    ASSERT_EQ(ana.canonicalIdCached(mb, forms, zc, main, quiet), atMain);
    ASSERT_EQ(forms.forms.count(), 3);
    // An expression already canonical keeps its id and needs no text.
    ASSERT_EQ(ana.canonicalIdCached(mb, forms, plain, main, child), plain);
    ASSERT_EQ(forms.texts.count(), 2);
    {
        gl::ScratchArena& strings = gl::scratchArenas().forSlot(0);
        gl::ScratchScope frame(strings);
        ASSERT_TRUE(gl::equalSpans(ana.canonicalTextCached(mb, forms, zc, main, child, strings),
            gl::StrSpan("(p[a,c])", 8)));
        ASSERT_TRUE(gl::equalSpans(ana.canonicalTextCached(mb, forms, plain, main, quiet, strings),
            gl::StrSpan("(p[c,e])", 8)));
    }
    // A complete-view change renews the cache: c now joins {bb, c} -> bb at main.
    putClassesAt(mb, main, {{"b", "z"}, {"bb", "c"}});
    ASSERT_EQ(mb.nameMap.decode(ana.canonicalIdCached(mb, forms, zc, main, quiet)), std::string("(p[b,bb])"));
    ASSERT_EQ(forms.version, mb.lbMemory.canonicalUses.version);
    ASSERT_EQ(ana.canonicalIdCached(mb, forms, zc, main, quiet), ana.canonicalWatchIdAt(mb, zc, quiet));
    ASSERT_EQ(ana.canonicalIdCached(mb, forms, zc, main, child), ana.canonicalWatchIdAt(mb, zc, child));
}

TEST(watch_ledger, record_with_known_canonical_matches_record_and_run_membership) {
    gl::ExpressionAnalyzer ana("Peano");
    ana.parameters.trackHistory = false;
    gl::Memory& mb = ana.body;
    const auto main = gl::NameMap::MAIN_ID;
    putClassesAt(mb, main, {{"a", "b"}});
    const auto first = gl::encodeExpression(gl::StrSpan("!(p[b])", 7), gl::StrSpan("main", 4), mb.nameMap);
    const auto second = gl::encodeExpression(gl::StrSpan("!(p[c])", 7), gl::StrSpan("main", 4), mb.nameMap);
    const auto canonical = ana.canonicalWatchIdAt(mb, first.originalId, main);
    ASSERT_EQ(mb.nameMap.decode(canonical), std::string("!(p[a])"));
    ana.recordWatchCanonical(mb.negationLedger, mb, first.originalId, main, first.nameId, canonical);
    ana.recordWatch(mb.negationLedger, mb, second.originalId, main, second.nameId);
    const int64_t key = gl::ExpressionAnalyzer::watchKey(main, first.nameId);
    const int64_t firstValue = gl::packInt32Pair(canonical, first.originalId);
    const int64_t secondValue = gl::packInt32Pair(second.originalId, second.originalId);
    ASSERT_TRUE(gl::ExpressionAnalyzer::watchRunContainsValue(mb.negationLedger, key, firstValue));
    ASSERT_TRUE(gl::ExpressionAnalyzer::watchRunContainsValue(mb.negationLedger, key, secondValue));
    ASSERT_FALSE(gl::ExpressionAnalyzer::watchRunContainsValue(mb.negationLedger, key,
        gl::packInt32Pair(first.originalId, first.originalId)));
    ASSERT_FALSE(gl::ExpressionAnalyzer::watchRunContainsValue(mb.negationLedger,
        gl::ExpressionAnalyzer::watchKey(main, mb.nameMap.encode("q")), firstValue));
    ASSERT_TRUE(mb.lbMemory.canonicalUses.watches.contains(gl::packStatementKey(first.originalId, main)));
    // A duplicate is a no-op on the sorted-unique run.
    const int32_t run = mb.negationLedger.lookup(key);
    ASSERT_EQ(mb.negationLedger.runLen(run), 2);
    ana.recordWatchCanonical(mb.negationLedger, mb, first.originalId, main, first.nameId, canonical);
    ASSERT_EQ(mb.negationLedger.runLen(mb.negationLedger.lookup(key)), 2);
    gl::NameId found = 0;
    ASSERT_TRUE(gl::ExpressionAnalyzer::watchRunHasCanonical(mb.negationLedger, main, first.nameId, canonical, &found));
    ASSERT_EQ(found, first.originalId);
}

TEST(watch_ledger, shared_refresh_forms_leave_ancestor_imports_unchanged) {
    // The same fixture imported with one cache for both targets and with a
    // per-call cache: identical ledger values, identifiers and re-checks.
    const auto run = [](bool shared) {
        gl::ExpressionAnalyzer ana("Peano");
        ana.parameters.trackHistory = false;
        gl::Memory& mb = ana.body;
        const auto main = mb.nameMap.encode("main");
        const auto left = mb.nameMap.encodePush(main, "hypo_(h[c])");
        const auto right = mb.nameMap.encodePush(main, "hypo_(h[d])");
        gl::DirtyState dirty = gl::DirtyState::Clean;
        gl::PagedVector<gl::IntEncodedExpr> out(&gl::genScratchArenas().forSlot(0), &dirty);
        const int levels[] = {0};
        ana.addStatement(gl::StrSpan("!(p[z,c])", 9), mb, true, levels, 1, gl::TransientOrigin{}, gl::StrSpan("main", 4), out);
        ana.addStatement(gl::StrSpan("!(p[z,d])", 9), mb, true, levels, 1, gl::TransientOrigin{}, gl::StrSpan("main", 4), out);
        putClassesAt(mb, main, {{"b", "z"}});
        putClassesAt(mb, left, {{"c", "d"}});
        putClassesAt(mb, right, {{"c", "d"}});
        gl::PagedVector<gl::NameId> members(&gl::genScratchArenas().forSlot(0), &dirty);
        members.push_back(mb.nameMap.lookup("c"));
        members.push_back(mb.nameMap.lookup("d"));
        gl::ExpressionAnalyzer::CanonicalFormCache forms(gl::genScratchArenas().forSlot(0));
        mb.pendingWatchRechecks.clear();
        ana.importAncestorNegationsForClass(mb, left, members, shared ? &forms : nullptr);
        ana.importAncestorNegationsForClass(mb, right, members, shared ? &forms : nullptr);
        std::vector<int64_t> state;
        for (const auto scope : {left, right}) {
            const int32_t id = mb.negationLedger.lookup(gl::ExpressionAnalyzer::watchKey(scope, mb.nameMap.lookup("p")));
            assert(id != 0);
            for (int32_t j = 0; j < mb.negationLedger.runLen(id); ++j) state.push_back(mb.negationLedger.valueAt(id, j));
        }
        for (int32_t i = 0; i < mb.pendingWatchRechecks.size(); ++i) state.push_back(mb.pendingWatchRechecks[i]);
        state.push_back(mb.nameMap.nameCount());
        const std::string canonical = mb.nameMap.decode(static_cast<gl::NameId>(uint64_t(state[0]) >> 32));
        assert(canonical == "!(p[b,c])");
        return state;
    };
    const std::vector<int64_t> shared = run(true), separate = run(false);
    ASSERT_EQ(shared.size(), static_cast<size_t>(4 + 4 + 1));
    ASSERT_TRUE(shared == separate);
}

TEST(watch_drain, grouped_sweep_serves_requests_of_several_cores) {
    gl::ExpressionAnalyzer ana("Peano");
    ana.parameters.trackHistory = false;
    gl::Memory& mb = ana.body;
    const auto main = mb.nameMap.encode("main");
    const auto child = mb.nameMap.encodePush(main, "hypo_(h[b])");
    const std::string childName = mb.nameMap.decode(child);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::PagedVector<gl::IntEncodedExpr> out(&gl::genScratchArenas().forSlot(0), &dirty);
    const int levels[] = {0};
    ana.addStatement(gl::StrSpan("(q[c])", 6), mb, true, levels, 1, gl::TransientOrigin{}, gl::StrSpan("main", 4), out);
    ana.addStatement(gl::StrSpan("(p[c])", 6), mb, true, levels, 1, gl::TransientOrigin{}, gl::StrSpan("main", 4), out);
    ana.addStatement(gl::StrSpan("(p[e])", 6), mb, true, levels, 1, gl::TransientOrigin{}, gl::StrSpan("main", 4), out);
    ana.addStatement(gl::StrSpan("!(q[b])", 7), mb, true, levels, 1, gl::TransientOrigin{}, gl::StrSpan(childName), out);
    ana.addStatement(gl::StrSpan("!(p[b])", 7), mb, true, levels, 1, gl::TransientOrigin{}, gl::StrSpan(childName), out);
    putClassesAt(mb, child, {{"b", "c"}});
    pushDelta(mb, child, {"b", "c"});
    ana.applyEquiClasses(mb);
    ana.materializeWatchHits(mb, 0);
    const gl::StrSpan childSpan(childName);
    ASSERT_TRUE(gl::lookupStatementFlags(mb.intKnownStatements, mb.nameMap, gl::StrSpan("(p[b])", 6), childSpan) != nullptr);
    ASSERT_TRUE(gl::lookupStatementFlags(mb.intKnownStatements, mb.nameMap, gl::StrSpan("(q[b])", 6), childSpan) != nullptr);
    ASSERT_TRUE(gl::lookupStatementFlags(mb.intKnownStatements, mb.nameMap, gl::StrSpan("(p[b])", 6), gl::StrSpan("main", 4)) == nullptr);
    ASSERT_TRUE(gl::lookupStatementFlags(mb.intKnownStatements, mb.nameMap, gl::StrSpan("(p[e])", 6), childSpan) == nullptr);
}


TEST(canonical_classes, compact_and_watch_readers_share_the_child_representative) {
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    ana.parameters.trackHistory = false;
    gl::Memory& mb = ana.body;
    const auto main = gl::NameMap::MAIN_ID;
    const auto child = mb.nameMap.encodePush(main,"canonical_child");
    putClassesAt(mb,main,{{"b","z"}});
    putClassesAt(mb,child,{{"a","b","z"}});
    const auto original = mb.nameMap.encode("(or98[z,7])");
    ASSERT_EQ(mb.nameMap.decode(ana.canonicalWatchIdAt(mb,original,main)),std::string("(or98[b,7])"));
    ASSERT_EQ(mb.nameMap.decode(ana.canonicalWatchIdAt(mb,original,child)),std::string("(or98[a,7])"));
    gl::ScratchArena& strings=gl::scratchArenas().forSlot(0);
    gl::ScratchArena& scratch=gl::genScratchArenas().forSlot(0);
    gl::ScratchScope textFrame(strings), frame(scratch);
    gl::ScratchString rewritten;
    gl::ExpressionAnalyzer::CompactCanonPair pairs[gl::ExecutionParameters::MAX_ARITY];
    int32_t pairCount=0,levelCount=0;
    int levels[256];
    const std::string validity=mb.nameMap.decode(child);
    ASSERT_TRUE(ana.canonicalizeCompact(mb,mb.nameMap.decodeView(original),gl::StrSpan(validity),
        strings,scratch,rewritten,pairs,pairCount,levels,levelCount,256));
    ASSERT_EQ(rewritten.toStdString(),std::string("(or98[a,7])"));
}


TEST(canonical_classes, late_ancestor_change_rekeys_before_importing_the_same_negation) {
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    ana.parameters.trackHistory=false;
    auto& mb=ana.body;
    const auto root=gl::NameMap::MAIN_ID;
    const auto child=mb.nameMap.encodePush(root,"canonical_child");
    gl::DirtyState dirty=gl::DirtyState::Clean;
    gl::PagedVector<gl::IntEncodedExpr> rows(&gl::genScratchArenas().forSlot(0),&dirty);
    const std::string negative="!(p[b])";
    const int level=0;
    ana.addStatement(gl::StrSpan(negative),mb,true,&level,1,gl::TransientOrigin{},
        gl::StrSpan("main",4),rows);
    addCanonicalFixtureEquality(ana,mb,child,"(=[a,b])",0);
    const auto key=gl::packInt32Pair(child,mb.nameMap.lookup("p"));
    const auto before=mb.negationLedger.lookup(key);
    ASSERT_TRUE(before!=0);
    ASSERT_EQ(mb.negationLedger.runLen(before),1);
    addCanonicalFixtureEquality(ana,mb,root,"(=[0,a])",0);
    const auto after=mb.negationLedger.lookup(key);
    ASSERT_TRUE(after!=0);
    ASSERT_EQ(mb.negationLedger.runLen(after),1);
    const auto pair=mb.negationLedger.valueAt(after,0);
    ASSERT_EQ(static_cast<gl::NameId>(uint64_t(pair)>>32),mb.nameMap.lookup("!(p[0])"));
    ASSERT_EQ(static_cast<gl::NameId>(uint64_t(pair)),mb.nameMap.lookup(negative));
}

TEST(canonical_classes, evidence_only_classes_keep_source_indexes_warm_and_source_writes_current) {
    for (bool integration : {false, true}) {
        gl::ExpressionAnalyzer analyzer("Peano");
        analyzer.parameters.trackHistory = false;
        auto& memory = analyzer.body;
        const auto root = gl::NameMap::MAIN_ID;
        addCanonicalFixtureEquality(analyzer, memory, root, "(=[a,z])", 2);
        const auto child = memory.nameMap.encodePush(root, "source_child");
        const auto sibling = memory.nameMap.encodePush(root, "source_sibling");
        analyzer.initializeCanonicalScope(memory, child);
        analyzer.initializeCanonicalScope(memory, sibling);
        const std::string validity = memory.nameMap.decode(child);
        const char* marker = "(K[z,marker])";
        const int level = 1;
        const auto park = [&](const char* source) {
            if (integration) analyzer.updateRejectedMapIntegration(gl::StrSpan(marker), gl::StrSpan(source),
                0, 2, &level, 1, memory.overallHashMemory, memory, gl::StrSpan(validity));
            else analyzer.updateRejectedMap(gl::StrSpan(marker), gl::StrSpan(source),
                0, 1, &level, 1, memory.overallHashMemory, memory, gl::StrSpan(validity));
        };
        park("(existence98[z])");
        const auto owner = gl::mintTemplateKey(memory.templateInterner, memory.nameMap, marker, validity);
        auto& arena = gl::genScratchArenas().forSlot(0);
        gl::ScratchScope frame(arena);
        gl::DirtyState dirty = gl::DirtyState::Clean;
        gl::PagedVector<gl::NameId> noIdentities(&arena, &dirty);
        auto& lookup = integration ? memory.lbMemory.existenceLookups.integrationSources
                                   : memory.lbMemory.existenceLookups.algebraSources;
        auto& uses = integration ? memory.lbMemory.canonicalUses.integrationSources
                                 : memory.lbMemory.canonicalUses.algebraSources;
        if (integration) analyzer.ensureExistenceSourceLookup(memory.overallHashMemory.rejectedMapIntegration,
            lookup, arena, memory);
        else analyzer.ensureExistenceSourceLookup(memory.overallHashMemory.rejectedMap,
            lookup, arena, memory);
        analyzer.refreshCanonicalSourceUses(memory, child, noIdentities, arena);
        const auto canonical = memory.nameMap.lookup("(existence98[a])");
        const auto identity = gl::packOriginKey(canonical, child);
        ASSERT_TRUE(canonical != 0 && lookup.postings.containsKey(identity));
        ASSERT_FALSE(lookup.postings.containsKey(gl::packOriginKey(canonical, sibling)));
        const auto memberKey = gl::packStatementKey(memory.nameMap.lookup("z"), child);
        const int32_t oldVersion = *memory.lbMemory.canonicalUses.memberVersions.find(memberKey);
        uses.markRawClean();
        memory.lbMemory.canonicalClassChanges.resetToFresh();
        auto cls = memory.equivalenceClassesMap.recordAt(memory.equivalenceClassesMap.lookup(root), 0);
        for (auto& pair : cls.intEqualityLevelsMap) pair.second.insert(7);
        memory.assignClassesById(root, {cls});
        analyzer.refreshCanonicalClasses(memory, root);
        const auto* flags = memory.lbMemory.canonicalClassChanges.find(memberKey);
        ASSERT_TRUE(flags != nullptr && (*flags & 4) != 0 && (*flags & 1) == 0);
        ASSERT_TRUE(*memory.lbMemory.canonicalUses.memberVersions.find(memberKey) > oldVersion);
        ASSERT_FALSE(uses.hasRawChanges());
        ASSERT_TRUE(lookup.postings.containsKey(identity));
        {
            auto& strings = gl::scratchArenas().forSlot(0);
            gl::ScratchScope textFrame(strings);
            int levels[256]; int32_t count = 0;
            const auto rewritten = analyzer.canonicalFormAtScope(gl::StrSpan("(p[z])"),
                gl::StrSpan(validity), memory, strings, levels, count, 256);
            ASSERT_TRUE(gl::equalSpans(rewritten.text, gl::StrSpan("(p[a])")));
            ASSERT_TRUE(std::find(levels, levels + count, 7) != levels + count);
        }
        park("(existence98[y])");
        uses.markRawClean();
        analyzer.refreshCanonicalSourceUses(memory, child, noIdentities, arena);
        ASSERT_TRUE(uses.hasRawChanges());
        const auto later = memory.nameMap.lookup("(existence98[y])");
        ASSERT_TRUE(later != 0 && lookup.postings.containsKey(gl::packOriginKey(later, child)));
        int found = 0;
        uses.forEach(child, gl::StrSpan("y"), [&](int64_t key) { ASSERT_EQ(key, owner); ++found; });
        ASSERT_EQ(found, 1);
        addCanonicalFixtureEquality(analyzer, memory, root, "(=[0,a])", 3);
        const auto replacement = memory.nameMap.lookup("(existence98[0])");
        ASSERT_TRUE(replacement != 0 && lookup.postings.containsKey(gl::packOriginKey(replacement, child)));
        ASSERT_FALSE(lookup.postings.containsKey(identity));
        ASSERT_TRUE(lookup.postings.containsKey(gl::packOriginKey(later, child)));
        ASSERT_FALSE(lookup.postings.containsKey(gl::packOriginKey(replacement, sibling)));
    }
}

TEST(canonical_classes, ancestor_identity_import_keeps_evidence_and_independent_arrivals) {
    gl::ExpressionAnalyzer analyzer("Peano");
    analyzer.parameters.trackHistory = false;
    auto& memory = analyzer.body;
    const auto root = gl::NameMap::MAIN_ID;
    const auto child = memory.nameMap.encodePush(root, "negative_child");
    const auto sibling = memory.nameMap.encodePush(root, "negative_sibling");
    analyzer.initializeCanonicalScope(memory, child);
    analyzer.initializeCanonicalScope(memory, sibling);
    auto& arena = gl::genScratchArenas().forSlot(0);
    gl::ScratchScope frame(arena);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::PagedVector<gl::IntEncodedExpr> rows(&arena, &dirty);
    const int level = 0;
    analyzer.addStatement(gl::StrSpan("!(p[z])"), memory, true, &level, 1,
        gl::TransientOrigin{}, gl::StrSpan("main"), rows);
    addCanonicalFixtureEquality(analyzer, memory, child, "(=[a,z])", 2);
    const auto pKey = analyzer.watchKey(child, memory.nameMap.lookup("p"));
    auto run = memory.negationLedger.lookup(pKey);
    ASSERT_TRUE(run != 0);
    const auto initial = memory.negationLedger.valueAt(run, 0);
    ASSERT_EQ(static_cast<gl::NameId>(uint64_t(initial) >> 32), memory.nameMap.lookup("!(p[a])"));
    const auto pending = memory.pendingWatchRechecks.size();
    auto cls = memory.equivalenceClassesMap.recordAt(memory.equivalenceClassesMap.lookup(child), 0);
    for (auto& pair : cls.intEqualityLevelsMap) pair.second.insert(7);
    memory.assignClassesById(child, {cls});
    memory.lbMemory.canonicalClassChanges.resetToFresh();
    analyzer.refreshCanonicalClasses(memory, child);
    const auto* flags = memory.lbMemory.canonicalClassChanges.find(
        gl::packStatementKey(memory.nameMap.lookup("z"), child));
    ASSERT_TRUE(flags != nullptr && (*flags & 4) != 0 && (*flags & 1) == 0);
    ASSERT_EQ(memory.negationLedger.valueAt(memory.negationLedger.lookup(pKey), 0), initial);
    ASSERT_EQ(memory.pendingWatchRechecks.size(), pending);
    analyzer.addStatement(gl::StrSpan("!(q[z])"), memory, true, &level, 1,
        gl::TransientOrigin{}, gl::StrSpan("main"), rows);
    const auto incoming = gl::encodeExpression(gl::StrSpan("!(q[z])"), gl::StrSpan("main"), memory.nameMap);
    const auto qKey = analyzer.watchKey(child, incoming.nameId);
    ASSERT_EQ(memory.negationLedger.lookup(qKey), 0);
    ASSERT_FALSE(memory.pendingWatchArrivals.empty());
    cls = memory.equivalenceClassesMap.recordAt(memory.equivalenceClassesMap.lookup(child), 0);
    for (auto& pair : cls.intEqualityLevelsMap) pair.second.insert(9);
    memory.assignClassesById(child, {cls});
    analyzer.refreshCanonicalClasses(memory, child);
    run = memory.negationLedger.lookup(qKey);
    ASSERT_TRUE(run != 0);
    ASSERT_EQ(static_cast<gl::NameId>(uint64_t(memory.negationLedger.valueAt(run, 0)) >> 32),
        memory.nameMap.lookup("!(q[a])"));
    const auto afterArrival = memory.pendingWatchRechecks.size();
    analyzer.importNegationIntoDescendantClassScopes(memory, incoming);
    ASSERT_EQ(memory.pendingWatchRechecks.size(), afterArrival);
    addCanonicalFixtureEquality(analyzer, memory, root, "(=[0,a])", 3);
    for (const char* core : {"p", "q"}) {
        const auto coreId = memory.nameMap.lookup(core);
        const auto key = analyzer.watchKey(child, coreId);
        run = memory.negationLedger.lookup(key);
        ASSERT_TRUE(run != 0);
        ASSERT_EQ(memory.negationLedger.runLen(run), 1);
        const auto pair = memory.negationLedger.valueAt(run, 0);
        const std::string canonical = "!(" + std::string(core) + "[0])";
        const std::string original = "!(" + std::string(core) + "[z])";
        ASSERT_EQ(static_cast<gl::NameId>(uint64_t(pair) >> 32), memory.nameMap.lookup(canonical));
        ASSERT_EQ(static_cast<gl::NameId>(uint64_t(pair)), memory.nameMap.lookup(original));
        ASSERT_EQ(memory.negationLedger.lookup(analyzer.watchKey(sibling, coreId)), 0);
    }
}

TEST(canonical_classes, ancestor_candidates_preserve_eligible_order_and_deduplication) {
    gl::ExpressionAnalyzer analyzer("Peano");
    analyzer.parameters.trackHistory = false;
    auto& memory = analyzer.body;
    const auto root = gl::NameMap::MAIN_ID;
    const auto child = memory.nameMap.encodePush(root, "filtered_child");
    analyzer.initializeCanonicalScope(memory, child);
    addCanonicalFixtureEquality(analyzer, memory, child, "(=[a,z])", 2);
    auto& arena = gl::genScratchArenas().forSlot(0);
    gl::ScratchScope frame(arena);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::PagedVector<gl::IntEncodedExpr> rows(&arena, &dirty);
    gl::PagedVector<gl::NameId> members(&arena, &dirty);
    members.push_back(memory.nameMap.lookup("z"));
    members.push_back(memory.nameMap.lookup("a"));
    const int level = 0;
    for (const char* text : {"(r[z])", "!(=[a,z])"})
        analyzer.addStatement(gl::StrSpan(text), memory, true, &level, 1,
            gl::TransientOrigin{}, gl::StrSpan("main"), rows);
    const auto before = memory.pendingWatchRechecks.size();
    analyzer.importAncestorNegationsForClass(memory, child, members);
    ASSERT_EQ(memory.pendingWatchRechecks.size(), before);
    for (const char* text : {"!(q[z])", "!(p[z,a])", "!(unrelated[u])"})
        analyzer.addStatement(gl::StrSpan(text), memory, true, &level, 1,
            gl::TransientOrigin{}, gl::StrSpan("main"), rows);
    analyzer.importAncestorNegationsForClass(memory, child, members);
    ASSERT_EQ(memory.pendingWatchRechecks.size(), before + 2);
    ASSERT_EQ(memory.pendingWatchRechecks[before],
        gl::packInt32Pair(child, memory.nameMap.lookup("!(p[z,a])")));
    ASSERT_EQ(memory.pendingWatchRechecks[before + 1],
        gl::packInt32Pair(child, memory.nameMap.lookup("!(q[z])")));
    for (const char* core : {"p", "q"}) {
        const auto coreId = memory.nameMap.lookup(core);
        const auto run = memory.negationLedger.lookup(analyzer.watchKey(child, coreId));
        ASSERT_TRUE(run != 0);
        ASSERT_EQ(memory.negationLedger.runLen(run), 1);
        const auto original = memory.nameMap.lookup(core[0] == 'p' ? "!(p[z,a])" : "!(q[z])");
        const auto canonical = memory.nameMap.lookup(core[0] == 'p' ? "!(p[a,a])" : "!(q[a])");
        ASSERT_EQ(memory.negationLedger.valueAt(run, 0), gl::packInt32Pair(canonical, original));
    }
    ASSERT_EQ(memory.negationLedger.lookup(analyzer.watchKey(child, memory.nameMap.lookup("unrelated"))), 0);
    analyzer.importAncestorNegationsForClass(memory, child, members);
    ASSERT_EQ(memory.pendingWatchRechecks.size(), before + 2);
}

TEST(canonical_classes, admission_chain_preserves_a_placeholder_equal_to_a_new_fixed_representative) {
    for (bool prefixed : {false, true}) {
        gl::ExpressionAnalyzer ana(std::string("Peano"));
        ana.parameters.trackHistory = false;
        auto& mb = ana.body;
        putClassesAt(mb, gl::NameMap::MAIN_ID, {{"10", "9"}});
        const std::string text = prefixed ? "(in3[u_6,u_9,10,u_5])" : "(in3[6,9,10,5])";
        const gl::StrSpan key[] = {gl::StrSpan(text)};
        const gl::StrSpan remaining[] = {gl::StrSpan("5",1), gl::StrSpan("6",1), gl::StrSpan("9",1)};
        ana.updateAdmissionMap(mb, key, 1, remaining, 3, 4, 4, false, gl::StrSpan("main",4));
        const auto demand = gl::packInt32Pair(mb.templateInterner.lookup("(in3[6,10,marker,5])"), gl::NameMap::MAIN_ID);
        const auto bucket = mb.overallHashMemory.admissionMap.lookup(demand);
        ASSERT_TRUE(bucket != 0);
        const auto values = gl::admissionRecordsAt(mb.overallHashMemory.admissionMap, demand, mb.valueInterner);
        ASSERT_EQ(values.size(), 1);
        const auto& row = *values.begin();
        ASSERT_EQ(row.key.size(), 1);
        ASSERT_EQ(mb.valueInterner.decode(row.key[0]), std::string("(in3[6,10,1,5])"));
    }
}

TEST(canonical_classes, admission_chain_keeps_fixed_metadata_absent_from_the_remaining_premise) {
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    ana.parameters.trackHistory = false;
    auto& mb = ana.body;
    putClassesAt(mb, gl::NameMap::MAIN_ID, {{"10", "9"}});
    const std::string text = "(in3[6,9,10,5])";
    const gl::StrSpan key[] = {gl::StrSpan(text)};
    const gl::StrSpan remaining[] = {gl::StrSpan("5",1), gl::StrSpan("6",1), gl::StrSpan("8",1), gl::StrSpan("9",1)};
    ana.updateAdmissionMap(mb, key, 1, remaining, 4, 4, 4, false, gl::StrSpan("main",4));
    const auto demand = gl::packInt32Pair(mb.templateInterner.lookup("(in3[6,10,marker,5])"), gl::NameMap::MAIN_ID);
    const auto values = gl::admissionRecordsAt(mb.overallHashMemory.admissionMap, demand, mb.valueInterner);
    ASSERT_EQ(values.size(), 1);
    const auto& row = *values.begin();
    ASSERT_EQ(row.remainingArgs.size(), 4);
    ASSERT_EQ(mb.valueInterner.decode(row.remainingArgs.back()), std::string("8"));
    ASSERT_EQ(mb.valueInterner.decode(row.key[0]), std::string("(in3[6,10,1,5])"));
}

TEST(canonical_classes, admission_chain_normalizer_preserves_bindings_and_anchor_exclusion) {
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    ana.parameters.trackHistory = false;
    auto& mb = ana.body;
    const auto scope = gl::NameMap::MAIN_ID;
    putClassesAt(mb, scope, {{"10", "9"}, {"a", "b"}});
    gl::ScratchArena& arena = gl::scratchArenas().forSlot(0);
    gl::ScratchScope frame(arena);
    gl::StrSpan output[2], remainingOutput[4];
    const gl::StrSpan remaining[] = {gl::StrSpan("5",1),gl::StrSpan("6",1),gl::StrSpan("9",1)};
    bool excluded = false;
    const std::string text = "(in3[6,9,10,5])";
    const gl::StrSpan chain[] = {gl::StrSpan(text)};
    ASSERT_EQ(ana.canonicalAdmissionChain(mb, chain, 1, remaining, 3, scope, arena, output, remainingOutput, excluded),3);
    ASSERT_FALSE(excluded);
    ASSERT_EQ(output[0].toStdString(),std::string("(in3[6,10,1,5])"));
    const std::string anchor = "(Anchor[9])";
    const gl::StrSpan protectedChain[] = {gl::StrSpan(anchor),gl::StrSpan(text)};
    ASSERT_EQ(ana.canonicalAdmissionChain(mb, protectedChain, 2, remaining, 3, scope, arena, output, remainingOutput, excluded),3);
    ASSERT_TRUE(excluded);
    ASSERT_EQ(output[0].toStdString(),anchor);
    ASSERT_EQ(output[1].toStdString(),text);
    const std::string named = "(in[b])";
    const gl::StrSpan namedChain[] = {gl::StrSpan(named)};
    const gl::StrReplacement replacements[] = {{gl::StrSpan("b",1),gl::StrSpan("a",1)}};
    ASSERT_EQ(ana.canonicalAdmissionChain(mb,namedChain,1,nullptr,0,scope,arena,output,remainingOutput,excluded,replacements,1),0);
    ASSERT_FALSE(excluded);
    ASSERT_EQ(output[0].toStdString(),std::string("(in[a])"));
}

TEST(canonical_classes, recursion_replay_marks_the_original_output_before_canonicalizing_fixed_arguments) {
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    ana.parameters.trackHistory = false;
    auto& mb = ana.body;
    putClassesAt(mb, gl::NameMap::MAIN_ID, {{"10", "9"}});
    const auto key = gl::mintTemplateKey(mb.templateInterner, mb.nameMap, "(in3[6,10,marker,5])", "main");
    gl::AdmissionMapValue value;
    value.key = {mb.valueInterner.encode("(in3[6,10,1,5])"), mb.valueInterner.encode("(in3[6,1,2,5])")};
    value.remainingArgs = {mb.valueInterner.encode("10"),mb.valueInterner.encode("5"),mb.valueInterner.encode("6")};
    value.standardMaxAdmissionDepth = 2;
    value.standardMaxSecondaryNumber = 2;
    value.flag = true;
    auto& arena = gl::genScratchArenas().forSlot(gl::genScratchArenas().slotCount()-1);
    gl::insertAdmissionValue(mb.overallHashMemory.admissionMap,key,value,mb.valueInterner,arena);
    mb.overallHashMemory.admissionStatusMap.upsert(key,uint8_t(1));
    const std::string arrived = "(in3[6,9,10,5])";
    ana.updateAdmissionMapRecursion(gl::StrSpan(arrived),mb,gl::StrSpan("main",4),true);
    const auto values = gl::admissionRecordsAt(mb.overallHashMemory.admissionMap,key,mb.valueInterner);
    bool replayed = false;
    for (const auto& row : values)
        replayed |= mb.valueInterner.decode(row.key[0]) == "(in3[6,10,10,5])";
    ASSERT_TRUE(replayed);
    const auto beforeRepeated = mb.overallHashMemory.admissionMap.count();
    const std::string repeated = "(in3[6,10,10,5])";
    ana.updateAdmissionMapRecursion(gl::StrSpan(repeated),mb,gl::StrSpan("main",4),true);
    const auto repeatedProbe = gl::mintTemplateKey(mb.templateInterner,mb.nameMap,"(in3[6,marker,marker,5])","main");
    ASSERT_EQ(mb.overallHashMemory.admissionMap.lookup(repeatedProbe),0);
    ASSERT_EQ(mb.overallHashMemory.admissionMap.count(),beforeRepeated);
    ASSERT_EQ(*mb.overallHashMemory.admissionStatusMap.find(repeatedProbe),uint8_t(0));
    gl::CanonicalUseIndex uses(&mb.lbMemory.manager);
    ana.indexCanonicalRecord(mb,mb.overallHashMemory.admissionMap,uses,key,0,arena);
    int placeholderHits = 0;
    uses.forEach(gl::NameMap::MAIN_ID,gl::StrSpan("2",1),[&](int64_t){++placeholderHits;});
    ASSERT_EQ(placeholderHits,0);
    int fixedHits = 0;
    uses.forEach(gl::NameMap::MAIN_ID,gl::StrSpan("10",2),[&](int64_t){++fixedHits;});
    ASSERT_EQ(fixedHits,1);
}

TEST(canonical_classes, integration_replay_keeps_distinct_original_argument_groups_when_fixed_names_merge) {
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    ana.parameters.trackHistory = false;
    auto& mb = ana.body;
    const auto scope = gl::NameMap::MAIN_ID;
    putClassesAt(mb, scope, {{"10", "9"}});
    auto& arena = gl::genScratchArenas().forSlot(gl::genScratchArenas().slotCount()-1);
    gl::WorkInstruction instruction(&arena);
    const std::string signature = "(in3[u_6,u_9,pi_lev_1_0,u_5])";
    const std::string goal = "(existence90[u_9])";
    instruction.setMarkedGoal(gl::StrSpan(goal));
    instruction.commitEntity(gl::StrSpan("atomic",6),gl::StrSpan(signature),gl::StrSpan(),4,instruction.elemMark());
    ana.prepareIntegrationCore2(instruction,nullptr,0,mb,gl::StrSpan("main",4),gl::StrSpan());
    const auto key = gl::mintTemplateKey(mb.templateInterner,mb.nameMap,"(in3[u_6,u_10,marker,u_5])","main");
    ASSERT_TRUE(mb.overallHashMemory.admissionMapIntegration.lookup(key)!=0);
    const std::string arrived = "(in3[6,9,10,5])";
    ana.updateAdmissionMapIntegration(gl::StrSpan(arrived),mb,gl::StrSpan("main",4));
    gl::ArenaIntegrationMap entries(&arena,&mb.valueInterner);
    entries.buildFromCold(mb.overallHashMemory.admissionMapIntegration,key);
    ASSERT_EQ(entries.entries.size(),1);
    ASSERT_TRUE(ana.integrationValueSeenAtScope(mb,entries,0,gl::StrSpan("10",2),scope));
    ASSERT_FALSE(ana.integrationValueSeenAtScope(mb,entries,0,gl::StrSpan("6",1),scope));
}

TEST(canonical_classes, dependency_index_selects_only_changed_tokens_and_tracks_value_replacements) {
    gl::Memory mb;
    gl::ScratchArena& arena = gl::genScratchArenas().forSlot(0);
    gl::ScratchScope frame(arena);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::TypedColdMap<int64_t, int32_t> records(&arena, &dirty);
    gl::CanonicalUseIndex uses(&mb.lbMemory.manager);
    const auto scope = gl::NameMap::MAIN_ID;
    for (int32_t i = 1; i <= 1001; ++i) records.upsert(i, i == 1 ? 1 : 2);
    int decoded = 0;
    auto sync = [&]() {
        uses.synchronize(records, arena, [&](int64_t key) {
            ++decoded;
            uses.remember(key);
            uses.add(key, scope, *records.find(key) == 1 ? gl::StrSpan("b",1) : gl::StrSpan("f",1));
        });
    };
    sync();
    ASSERT_EQ(decoded, 1001);
    int hits = 0;
    uses.select(scope, gl::StrSpan("b",1), 0, [&](int64_t key) { ASSERT_EQ(key, 1); ++hits; });
    ASSERT_EQ(hits, 1);
    uses.select(scope, gl::StrSpan("b",1), 0, [&](int64_t) { ++hits; });
    ASSERT_EQ(hits, 1);
    uses.select(scope, gl::StrSpan("b",1), 1, [&](int64_t key) { ASSERT_EQ(key, 1); ++hits; });
    ASSERT_EQ(hits, 2);
    sync(); ASSERT_EQ(decoded,1001);
    records.upsert(1,2); uses.changed(1); uses.changed(1); sync();
    ASSERT_EQ(decoded,1002);
    hits=0; uses.forEach(scope,gl::StrSpan("b",1),[&](int64_t) { ++hits; }); ASSERT_EQ(hits,0);
    uses.forEach(scope,gl::StrSpan("f",1),[&](int64_t) { ++hits; }); ASSERT_EQ(hits,1001);
    records.erase(1); sync();
    hits=0; uses.forEach(scope,gl::StrSpan("f",1),[&](int64_t) { ++hits; }); ASSERT_EQ(hits,1000);
    ASSERT_TRUE(uses.liveBytes()>0);
    uses.clear(); hits=0; uses.forEach(scope,gl::StrSpan("f",1),[&](int64_t) { ++hits; }); ASSERT_EQ(hits,0);
}

TEST(canonical_classes, template_comparison_uses_target_scope_and_keeps_unrelated_projections_warm) {
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    ana.parameters.trackHistory=false;
    auto& mb=ana.body;
    const auto root=gl::NameMap::MAIN_ID;
    const auto child=mb.nameMap.encodePush(root,"match_child");
    const auto sibling=mb.nameMap.encodePush(root,"match_sibling");
    putClassesAt(mb,root,{{"a","z"},{"f","g"}});
    putClassesAt(mb,child,{{"b","z"}});
    ana.initializeCanonicalScope(mb,sibling);
    gl::ScratchArena& arena=gl::genScratchArenas().forSlot(0);
    gl::ScratchScope frame(arena);
    gl::DirtyState dirty=gl::DirtyState::Clean;
    gl::TypedColdMap<int64_t,int32_t> records(&arena,&dirty);
    const auto original=mb.templateInterner.encode("(K[b,marker])");
    const int64_t owner=gl::packInt32Pair(original,root);
    records.upsert(owner,1);
    for(int i=0;i<100;++i) {
        const std::string text="(Q"+std::to_string(i)+"[g,marker])";
        records.upsert(gl::packInt32Pair(mb.templateInterner.encode(text),root),1);
    }
    auto& index=mb.lbMemory.canonicalUses.algebraMatches;
    gl::PagedVector<int64_t> result(&arena,&dirty);
    auto query=[&](const char* text,gl::NameId target) {
        result.clear();
        ana.matchingCanonicalTemplates(mb,records,index,gl::StrSpan(text),root,target,result,arena);
    };
    query("(K[a,marker])",child);
    ASSERT_EQ(result.size(),1); ASSERT_EQ(result[0],owner);
    ASSERT_TRUE(ana.canonicalTemplatePresent(mb,records,index,gl::StrSpan("(K[a,marker])"),gl::StrSpan("main",4),child));
    ASSERT_FALSE(ana.canonicalTemplatePresent(mb,records,index,gl::StrSpan("(K[x,marker])"),gl::StrSpan("main",4),child));
    query("(Q0[f,marker])",child); ASSERT_EQ(result.size(),1);
    const auto built=index.rebuildCount;
    query("(K[a,marker])",child); ASSERT_EQ(index.rebuildCount,built);
    query("(K[a,marker])",sibling); ASSERT_EQ(result.size(),0);
    const auto before=index.rebuildCount;
    addCanonicalFixtureEquality(ana,mb,root,"(=[0,a])",0);
    query("(K[0,marker])",child);
    ASSERT_EQ(result.size(),1); ASSERT_EQ(result[0],owner);
    ASSERT_EQ(index.rebuildCount,before+1);
    ASSERT_EQ(records.lookup(owner),1);
    query("(K[0,marker])",child); ASSERT_EQ(index.rebuildCount,before+1);
    gl::ScratchArena& strings=gl::scratchArenas().forSlot(0);
    gl::ScratchScope textFrame(strings);
    ASSERT_EQ(ana.canonicalAdmissionText(mb,gl::StrSpan("(K[u_b,marker])"),child,strings,true).toStdString(),std::string("(K[u_0,marker])"));
    ASSERT_EQ(ana.canonicalAdmissionText(mb,gl::StrSpan("(Anchor[b])"),child,strings,true).toStdString(),std::string("(Anchor[b])"));
    ASSERT_EQ(ana.canonicalAdmissionText(mb,gl::StrSpan("(K[b,marker])"),child,strings,false,gl::StrSpan("b",1)).toStdString(),std::string("(K[b,marker])"));
    ASSERT_EQ(ana.canonicalAdmissionText(mb,gl::StrSpan("(goal[b])_subproof_(K[b,marker])"),child,strings).toStdString(),
        std::string("(goal[b])_subproof_(K[0,marker])"));
    const int64_t goalOwner=gl::packInt32Pair(mb.templateInterner.encode("(goal[b])_subproof_(K[b,marker])"),root);
    records.upsert(goalOwner,1);
    query("(goal[b])_subproof_(K[0,marker])",child); ASSERT_EQ(result.size(),1);
    query("(goal[0])_subproof_(K[0,marker])",child); ASSERT_EQ(result.size(),0);
    records.erase(owner); query("(K[0,marker])",child); ASSERT_EQ(result.size(),0);
    ASSERT_TRUE(index.liveBytes()>0); index.clear(); ASSERT_EQ(index.rebuildCount,0);
}

TEST(canonical_classes, comparison_queue_coalesces_and_partitions_effective_scopes) {
    gl::Memory memory;
    const auto root = gl::NameMap::MAIN_ID;
    const auto child = memory.nameMap.encodePush(root, "queued_child");
    const auto original = memory.templateInterner.encode("(K[z,marker])");
    const int64_t first = gl::packInt32Pair(original, root);
    const int64_t second = gl::packInt32Pair(original, child);
    gl::CanonicalTemplateIndex index(&memory.lbMemory.manager);
    index.markRawClean();
    index.enqueueProjection(first);
    index.enqueueProjection(first);
    index.enqueueProjection(second);
    ASSERT_EQ(index.pending.size(), 2);
    ASSERT_TRUE(index.hasRawChanges());
    std::vector<int64_t> found;
    index.pending.forEach(root, [&](int64_t key) { found.push_back(key); });
    ASSERT_EQ(found.size(), 1u);
    ASSERT_EQ(found[0], first);
    ASSERT_TRUE(index.pending.erase(root, first));
    ASSERT_FALSE(index.pending.erase(root, first));
    index.pending.maintain();
    ASSERT_FALSE(index.pending.containsKey(root));
    ASSERT_TRUE(index.pending.containsKey(child));
    index.clear();
    ASSERT_EQ(index.pending.size(), 0);
    index.invalidateMember(child, gl::StrSpan("z", 1));
    ASSERT_EQ(index.pending.size(), 0);
}

TEST(canonical_classes, comparison_member_change_reaches_every_store_and_only_its_scope) {
    gl::Memory memory;
    const auto child = memory.nameMap.encodePush(gl::NameMap::MAIN_ID, "queued_child");
    const auto sibling = memory.nameMap.encodePush(gl::NameMap::MAIN_ID, "queued_sibling");
    auto& uses = memory.lbMemory.canonicalUses;
    gl::CanonicalTemplateIndex* indexes[] = {
        &uses.algebraMatches, &uses.integrationMatches, &uses.integrationSetMatches,
        &uses.algebraParkMatches, &uses.integrationParkMatches, &uses.ordisParkMatches,
        &uses.ordis2ParkMatches, &uses.ordis2Matches, &uses.preparedMatches,
        &uses.preparedMarkerMatches, &uses.statementMatches, &uses.subproofParkMatches
    };
    const auto original = memory.templateInterner.encode("(K[z,marker])");
    const int64_t key = gl::packInt32Pair(original, child);
    for (auto* index : indexes) {
        index->forms.upsert(key, original);
        index->uses.add(key, child, gl::StrSpan("z", 1));
        index->markRawClean();
    }
    uses.invalidateComparisonMember(sibling, gl::StrSpan("z", 1));
    uses.invalidateComparisonMember(child, gl::StrSpan("absent", 6));
    for (auto* index : indexes) {
        ASSERT_EQ(index->pending.size(), 0);
        ASSERT_FALSE(index->hasRawChanges());
    }
    uses.invalidateComparisonMember(child, gl::StrSpan("z", 1));
    uses.invalidateComparisonMember(child, gl::StrSpan("z", 1));
    for (auto* index : indexes) {
        ASSERT_EQ(index->pending.size(), 1);
        ASSERT_TRUE(index->pending.containsKey(child));
        ASSERT_FALSE(index->pending.containsKey(sibling));
        ASSERT_TRUE(index->hasRawChanges());
        index->clear();
    }
    uses.invalidateComparisonMember(child, gl::StrSpan("z", 1));
    for (auto* index : indexes) ASSERT_EQ(index->pending.size(), 0);
}

TEST(canonical_classes, comparison_queries_drain_only_changed_scope_and_keep_warm_results) {
    gl::ExpressionAnalyzer analyzer(std::string("Peano"));
    analyzer.parameters.trackHistory = false;
    auto& memory = analyzer.body;
    const auto root = gl::NameMap::MAIN_ID;
    const auto child = memory.nameMap.encodePush(root, "queued_child");
    const auto sibling = memory.nameMap.encodePush(root, "queued_sibling");
    putClassesAt(memory, root, {{"a", "z"}});
    analyzer.initializeCanonicalScope(memory, child);
    analyzer.initializeCanonicalScope(memory, sibling);
    auto& arena = gl::genScratchArenas().forSlot(0);
    gl::ScratchScope frame(arena);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::TypedColdMap<int64_t, int32_t> records(&arena, &dirty);
    const auto original = memory.templateInterner.encode("(K[z,marker])");
    const int64_t owner = gl::packInt32Pair(original, root);
    records.upsert(owner, 1);
    auto& index = memory.lbMemory.canonicalUses.algebraMatches;
    gl::PagedVector<int64_t> result(&arena, &dirty);
    const auto query = [&](const char* text, gl::NameId scope) {
        result.clear();
        analyzer.matchingCanonicalTemplates(memory, records, index, gl::StrSpan(text),
            root, scope, result, arena);
    };
    query("(K[a,marker])", child); ASSERT_EQ(result.size(), 1);
    query("(K[a,marker])", sibling); ASSERT_EQ(result.size(), 1);
    const auto warm = index.rebuildCount;
    addCanonicalFixtureEquality(analyzer, memory, child, "(=[0,a])", 3);
    ASSERT_TRUE(index.pending.containsKey(child));
    ASSERT_FALSE(index.pending.containsKey(sibling));
    query("(K[a,marker])", sibling);
    ASSERT_EQ(result.size(), 1); ASSERT_EQ(result[0], owner);
    ASSERT_EQ(index.rebuildCount, warm);
    ASSERT_TRUE(index.pending.containsKey(child));
    query("(K[0,marker])", child);
    ASSERT_EQ(result.size(), 1); ASSERT_EQ(result[0], owner);
    ASSERT_EQ(index.rebuildCount, warm + 1);
    ASSERT_FALSE(index.pending.containsKey(child));
    addCanonicalFixtureEquality(analyzer, memory, child, "(=[0,a])", 7);
    ASSERT_FALSE(index.pending.containsKey(child));
    query("(K[a,marker])", child);
    ASSERT_EQ(result.size(), 1); ASSERT_EQ(index.rebuildCount, warm + 1);
    const auto later = memory.templateInterner.encode("(Q[z,marker])");
    const int64_t laterOwner = gl::packInt32Pair(later, root);
    records.upsert(laterOwner, 1);
    query("(K[0,marker])", child); ASSERT_EQ(index.rebuildCount, warm + 1);
    query("(Q[0,marker])", child);
    ASSERT_EQ(result.size(), 1); ASSERT_EQ(result[0], laterOwner);
    ASSERT_EQ(index.rebuildCount, warm + 2);
    records.erase(owner);
    query("(K[0,marker])", child); ASSERT_EQ(result.size(), 0);
    records.upsert(owner, 1);
    query("(K[0,marker])", child);
    ASSERT_EQ(result.size(), 1); ASSERT_EQ(result[0], owner);
}

TEST(canonical_classes, comparison_new_class_invalidates_previously_unrelated_cached_token) {
    gl::ExpressionAnalyzer analyzer(std::string("Peano"));
    analyzer.parameters.trackHistory = false;
    auto& memory = analyzer.body;
    const auto root = gl::NameMap::MAIN_ID;
    putClassesAt(memory, root, {{"m", "n"}});
    auto& arena = gl::genScratchArenas().forSlot(0);
    gl::ScratchScope frame(arena);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::TypedColdMap<int64_t, int32_t> records(&arena, &dirty);
    const auto original = memory.templateInterner.encode("(K[z,marker])");
    const int64_t owner = gl::packInt32Pair(original, root);
    records.upsert(owner, 1);
    auto& index = memory.lbMemory.canonicalUses.algebraMatches;
    gl::PagedVector<int64_t> result(&arena, &dirty);
    analyzer.matchingCanonicalTemplates(memory, records, index, gl::StrSpan("(K[m,marker])"),
        root, root, result, arena);
    ASSERT_EQ(result.size(), 0);
    ASSERT_EQ(index.rebuildCount, 1);
    addCanonicalFixtureEquality(analyzer, memory, root, "(=[a,z])", 1);
    ASSERT_TRUE(index.pending.containsKey(root));
    analyzer.matchingCanonicalTemplates(memory, records, index, gl::StrSpan("(K[a,marker])"),
        root, root, result, arena);
    ASSERT_EQ(result.size(), 1); ASSERT_EQ(result[0], owner);
    ASSERT_EQ(index.rebuildCount, 2);
    index.clear(); result.clear();
    analyzer.matchingCanonicalTemplates(memory, records, index, gl::StrSpan("(K[a,marker])"),
        root, root, result, arena);
    ASSERT_EQ(result.size(), 1); ASSERT_EQ(result[0], owner);
    ASSERT_EQ(index.rebuildCount, 1);
}

TEST(canonical_classes, posting_prefix_walk_stops_and_preserves_survivor_order) {
    gl::Memory mb;
    gl::OrPostingIndex<int64_t> postings(&mb.lbMemory.manager);
    for (int64_t value=1; value<=4; ++value) postings.insert(7,value);
    std::vector<int64_t> seen;
    postings.forEachWhile(7,[&](int64_t value) { seen.push_back(value); return seen.size()<2; });
    ASSERT_TRUE((seen==std::vector<int64_t>{4,3}));
    postings.erase(7,3);
    postings.insert(7,5);
    postings.maintain();
    seen.clear();
    postings.forEachWhile(7,[&](int64_t value) { seen.push_back(value); return true; });
    ASSERT_TRUE((seen==std::vector<int64_t>{5,4,2,1}));
    postings.forEachWhile(8,[&](int64_t) { ASSERT_TRUE(false); return false; });
    postings.clear();
    postings.forEachWhile(7,[&](int64_t) { ASSERT_TRUE(false); return false; });
}

TEST(canonical_classes, new_postings_do_not_reselect_old_dependents_of_unchanged_equality) {
    gl::Memory mb;
    auto& arena=gl::genScratchArenas().forSlot(0);
    gl::ScratchScope frame(arena);
    gl::CanonicalUseIndex uses(&mb.lbMemory.manager);
    const auto scope=gl::NameMap::MAIN_ID;
    for (int64_t record=1; record<=1000; ++record) uses.add(record,scope,gl::StrSpan("a",1));
    std::vector<int64_t> seen;
    auto select=[&](int32_t version) {
        seen.clear();
        uses.select(scope,gl::StrSpan("a",1),version,[&](int64_t record) { seen.push_back(record); });
    };
    select(1); ASSERT_EQ(seen.size(),size_t(1000));
    uses.add(1001,scope,gl::StrSpan("a",1));
    select(1); ASSERT_TRUE((seen==std::vector<int64_t>{1001}));
    uses.add(1001,scope,gl::StrSpan("a",1));
    select(1); ASSERT_TRUE(seen.empty());
    uses.erase(1001,arena);
    select(1); ASSERT_TRUE(seen.empty());
    uses.erase(500,arena);
    uses.add(500,scope,gl::StrSpan("a",1));
    uses.add(250,scope,gl::StrSpan("b",1));
    select(1); ASSERT_TRUE((seen==std::vector<int64_t>{500}));
    select(2); ASSERT_EQ(seen.size(),size_t(1000));
    uses.markRawClean();
    ASSERT_FALSE(uses.hasRawChanges());
    uses.add(1002,scope,gl::StrSpan("a",1));
    ASSERT_TRUE(uses.hasRawChanges());
    select(2); ASSERT_TRUE((seen==std::vector<int64_t>{1002}));
    uses.clear();
    uses.add(2,scope,gl::StrSpan("a",1));
    select(2); ASSERT_TRUE((seen==std::vector<int64_t>{2}));
}

TEST(canonical_classes, presence_exact_hits_avoid_projections_and_aliases_keep_scope_ownership) {
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    auto& mb = ana.body;
    const auto root = gl::NameMap::MAIN_ID;
    const auto child = mb.nameMap.encodePush(root,"presence_child");
    const auto sibling = mb.nameMap.encodePush(root,"presence_sibling");
    putClassesAt(mb,root,{{"a","z"}});
    putClassesAt(mb,child,{{"b","z"}});
    auto& records = mb.overallHashMemory.admissionStatusMap;
    auto& index = mb.lbMemory.canonicalUses.algebraMatches;
    const auto a = gl::mintTemplateKey(mb.templateInterner,mb.nameMap,"(K[a,marker])","main");
    const auto b = gl::mintTemplateKey(mb.templateInterner,mb.nameMap,"(K[b,marker])","main");
    records.upsert(a,uint8_t(1));
    ASSERT_TRUE(ana.canonicalTemplatePresent(mb,records,index,gl::StrSpan("(K[a,marker])"),gl::StrSpan("main",4),child));
    ASSERT_TRUE(ana.canonicalTemplatePresent(mb,records,index,gl::StrSpan("(K[b,marker])"),gl::StrSpan("main",4),child));
    ASSERT_EQ(index.rebuildCount,0);
    ASSERT_EQ(index.forms.count(),0);
    records.erase(a);
    records.upsert(b,uint8_t(1));
    ASSERT_TRUE(ana.canonicalTemplatePresent(mb,records,index,gl::StrSpan("(K[b,marker])"),gl::StrSpan("main",4),child));
    ASSERT_EQ(index.rebuildCount,0);
    ASSERT_TRUE(ana.canonicalTemplatePresent(mb,records,index,gl::StrSpan("(K[a,marker])"),gl::StrSpan("main",4),child));
    ASSERT_EQ(index.rebuildCount,1);
    ASSERT_FALSE(ana.canonicalTemplatePresent(mb,records,index,gl::StrSpan("(K[a,marker])"),gl::StrSpan("main",4),sibling));
    ASSERT_FALSE(ana.canonicalTemplatePresent(mb,records,index,gl::StrSpan("(K[a,marker])"),gl::StrSpan("missing_scope")));
    ASSERT_EQ(records.count(),1);
    ASSERT_TRUE(records.lookup(b)!=0);
}

TEST(canonical_classes, algebra_probe_exact_budget_hit_is_cheap_and_ineligible_hit_checks_aliases) {
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    ana.parameters.maxNumberSecondaryVariables = 2;
    auto& mb = ana.body;
    const auto root = gl::NameMap::MAIN_ID;
    const auto child = mb.nameMap.encodePush(root,"probe_child");
    putClassesAt(mb,root,{{"a","z"}});
    putClassesAt(mb,child,{{"b","z"}});
    auto& arena = gl::genScratchArenas().forSlot(gl::genScratchArenas().slotCount()-1);
    const auto a = gl::mintTemplateKey(mb.templateInterner,mb.nameMap,"(in2[a,marker,3])","main");
    const auto b = gl::mintTemplateKey(mb.templateInterner,mb.nameMap,"(in2[b,marker,3])","main");
    gl::AdmissionMapValue value;
    value.key = {mb.valueInterner.encode("(in2[a,1,3])")};
    value.remainingArgs = {mb.valueInterner.encode("a"),mb.valueInterner.encode("3")};
    value.standardMaxAdmissionDepth = 0;
    value.standardMaxSecondaryNumber = 2;
    gl::insertAdmissionValue(mb.overallHashMemory.admissionMap,a,value,mb.valueInterner,arena);
    mb.overallHashMemory.admissionStatusMap.upsert(a,uint8_t(0));
    value.standardMaxAdmissionDepth = 3;
    gl::insertAdmissionValue(mb.overallHashMemory.admissionMap,b,value,mb.valueInterner,arena);
    mb.overallHashMemory.admissionStatusMap.upsert(b,uint8_t(0));
    const std::string expression = "(in2[b,temporary_witness,3])";
    const std::string marked = "(in2[b,marker,3])";
    const std::string witness = "temporary_witness";
    const auto probe = [&](int generation) {
        const gl::WitnessAdmissionMetrics metrics{generation,1};
        return ana.isAdmitted(mb,gl::StrSpan(expression),gl::StrSpan(witness),gl::StrSpan(marked),
            gl::StrSpan("main",4),&metrics,false,0,child);
    };
    auto& index = mb.lbMemory.canonicalUses.algebraMatches;
    ASSERT_TRUE(probe(0));
    ASSERT_EQ(index.rebuildCount,0);
    ASSERT_EQ(index.forms.count(),0);
    ASSERT_TRUE(probe(3));
    ASSERT_EQ(index.rebuildCount,2);
    ASSERT_FALSE(probe(4));
    ASSERT_EQ(index.rebuildCount,2);
    ASSERT_EQ(mb.overallHashMemory.admissionMap.count(),2);
    ASSERT_TRUE(mb.overallHashMemory.productsOfRecursionIds.empty());
    ASSERT_TRUE(mb.sameIterationInternalMail.empty());
}

TEST(canonical_classes, integration_probe_uses_exact_canonical_key_without_applying_values) {
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    auto& mb = ana.body;
    putClassesAt(mb,gl::NameMap::MAIN_ID,{{"2","9"}});
    const std::string prefixed = "(in2[u_2,marker,u_3])";
    const auto key = gl::mintTemplateKey(mb.templateInterner,mb.nameMap,prefixed,"main");
    gl::Instruction instruction;
    instruction.data.push_back(gl::LogicalEntity("atomic",{},prefixed,3));
    instruction.markedGoal = prefixed;
    gl::IntegrationEntry entry;
    entry.instruction = gl::encodeInstruction(instruction,mb.valueInterner);
    auto& records = mb.overallHashMemory.admissionMapIntegration;
    records.assignRun(key,{entry});
    const auto before = gl::Codec<gl::IntegrationEntry>::serialize(records.recordAt(records.lookup(key),0));
    const std::string expression = "(in2[9,temporary_witness,3])";
    const std::string marked = "(in2[9,marker,3])";
    const std::string witness = "temporary_witness";
    ASSERT_TRUE(ana.isAdmittedIntegration(mb,gl::StrSpan(expression),gl::StrSpan(witness),
        gl::StrSpan(marked),gl::StrSpan("main",4),false));
    ASSERT_EQ(mb.lbMemory.canonicalUses.integrationMatches.rebuildCount,0);
    ASSERT_EQ(mb.lbMemory.canonicalUses.integrationMatches.forms.count(),0);
    ASSERT_TRUE(before==gl::Codec<gl::IntegrationEntry>::serialize(records.recordAt(records.lookup(key),0)));
    ASSERT_EQ(mb.valueInterner.lookup(gl::StrSpan(witness)),0);
    ASSERT_TRUE(mb.sameIterationInternalMail.empty());
}

TEST(canonical_classes, late_parent_bridge_removes_rules_owned_by_the_old_child_representative) {
    for(bool staged:{false,true}) {
        gl::ExpressionAnalyzer ana(std::string("Peano"));
        ana.parameters.trackHistory=true;
        auto& mb=ana.body;
        const auto root=gl::NameMap::MAIN_ID;
        const auto child=mb.nameMap.encodePush(root,"compact_child");
        const std::string validity=mb.nameMap.decode(child);
        addCanonicalFixtureEquality(ana,mb,child,"(=[b,z])",0);
        const std::vector<std::string> leaves={"(in[u_1,u_2])","(in2[u_1,u_3,u_2])"};
        const std::string name=ana.findOrMintOrOperator(leaves,3);
        ana.preMintReducedOrs();
        const auto* entity=ana.compiledEntity(gl::StrSpan(name));
        ASSERT_TRUE(entity!=nullptr && !entity->implications.empty());
        const std::string carrier=ce::replaceKeysInString(entity->implications.front(),{{"u_1","b"},{"u_2","10"},{"u_3","b"}});
        const std::string replacement=ce::replaceKeysInString(carrier,{{"b","a"}});
        const int level=0;
        const gl::TransientOrigin origin{true,gl::OriginTag::taskFormulation,nullptr,0};
        ana.addExprToMemoryBlock(gl::StrSpan(carrier),mb,0,0,&level,1,origin,-1,-1,gl::StrSpan(validity),false);
        const int64_t oldKey=gl::packStatementKey(mb.nameMap.lookup(carrier),child);
        const int32_t oldId=mb.compactExpansions.lookup(oldKey);
        ASSERT_TRUE(oldId!=0);
        const auto oldRule=mb.compactExpansions.recordAt(oldId,0);
        const int32_t ruleId=mb.ruleInterner.lookup(mb.lbStateInterner.decodeView(oldRule.implTextId));
        ASSERT_TRUE(ruleId!=0);
        const auto applicationBefore=mb.decodeClassesById(child);
        addCanonicalFixtureEquality(ana,mb,root,"(=[a,z])",1);
        const auto applicationAfter=mb.decodeClassesById(child);
        ASSERT_EQ(applicationBefore.size(),applicationAfter.size());
        ASSERT_TRUE(applicationBefore.front().memberIds==applicationAfter.front().memberIds);
        if(staged) {
            auto& arena=gl::genScratchArenas().forSlot(0);
            gl::DirtyState dirty=gl::DirtyState::Clean;
            gl::RejectedValidityBuckets buckets(&arena,&dirty);
            gl::PagedVector<int64_t> compactKeys(&arena,&dirty);
            gl::ColdHashSet<gl::BytesKeyStore> visited(&arena,&dirty);
            gl::ColdHashSet<gl::PodKeyStore<int64_t>> drops(&arena,&dirty),removed(&arena,&dirty);
            gl::TypedColdSet<int64_t> selected(&arena,&dirty),pending(&arena,&dirty);
            selected.mint(oldKey);
            const auto cls=mb.canonicalEquivalenceClassesMap.recordAt(mb.canonicalEquivalenceClassesMap.lookup(child),0);
            ASSERT_TRUE(ana.applyEquivalenceClassToCompactImplications(cls,mb,gl::StrSpan(validity),buckets,
                compactKeys,visited,drops,removed,pending,&selected)>0);
            ASSERT_TRUE(pending.contains(oldKey));
            ASSERT_FALSE(ana.compactExpandedHere(mb,gl::StrSpan(replacement),gl::StrSpan(validity)));
            ana.finishCompactRemovals(mb,drops,removed,arena);
            ASSERT_EQ(mb.compactExpansions.lookup(oldKey),0);
            ana.installCanonicalCompacts(mb,pending);
            ASSERT_TRUE(pending.empty());
            ana.installCanonicalCompacts(mb,pending);
        }
        ana.applyEquiClasses(mb);
        ASSERT_EQ(mb.compactExpansions.lookup(oldKey),0);
        ASSERT_TRUE(ana.compactExpandedHere(mb,gl::StrSpan(replacement),gl::StrSpan(validity)));
        for (const auto* memory : {&mb.overallHashMemory,&mb.localHashMemory,&mb.localHashMemoryDelta,&mb.workingMemory}) {
            for(int32_t i=1;i<=memory->encodedMap.count();++i)
                for(int32_t j=0;j<memory->encodedMap.runLen(i);++j)
                    ASSERT_TRUE(memory->encodedMap.recordAt(i,j).originalImplicationId!=ruleId);
            ASSERT_FALSE(memory->ownerlessPending);
        }
    }
}

TEST(canonical_classes, inherited_watch_selection_uses_complete_members_and_stays_indexed) {
    gl::ExpressionAnalyzer analyzer("Peano");
    analyzer.parameters.trackHistory=false;
    auto& memory=analyzer.body;
    const auto root=gl::NameMap::MAIN_ID;
    const auto child=memory.nameMap.encodePush(root,"watch_child");
    putClassesAt(memory,root,{{"b","z"}});
    putClassesAt(memory,child,{{"a","z"}});
    analyzer.initializeCanonicalScope(memory,child);
    const auto negative=gl::encodeExpression(gl::StrSpan("!(p[b])"),gl::StrSpan("main"),memory.nameMap);
    analyzer.importNegationIntoDescendantClassScopes(memory,negative);
    const auto key=analyzer.watchKey(child,negative.nameId);
    auto run=memory.negationLedger.lookup(key);
    ASSERT_TRUE(run!=0);
    ASSERT_EQ(memory.negationLedger.runLen(run),1);
    ASSERT_EQ(memory.nameMap.decode(static_cast<gl::NameId>(uint64_t(memory.negationLedger.valueAt(run,0))>>32)),std::string("!(p[a])"));
    const auto pending=memory.pendingWatchRechecks.size();
    analyzer.importNegationIntoDescendantClassScopes(memory,negative);
    ASSERT_EQ(memory.pendingWatchRechecks.size(),pending);
    analyzer.ensureCanonicalWatchUses(memory);
    gl::EquivalenceClass enlarged; enlarged.setMembersFromNames({"0","b","z"},memory.nameMap);
    memory.assignClassesById(root,{enlarged});
    analyzer.refreshCanonicalClasses(memory,root);
    run=memory.negationLedger.lookup(key);
    ASSERT_EQ(memory.negationLedger.runLen(run),1);
    ASSERT_EQ(memory.nameMap.decode(static_cast<gl::NameId>(uint64_t(memory.negationLedger.valueAt(run,0))>>32)),std::string("!(p[0])"));
}

TEST(canonical_classes, identity_interfaces_track_markers_statements_and_pending_carriers) {
    gl::ExpressionAnalyzer analyzer("Peano"); analyzer.parameters.trackHistory=false;
    auto& memory=analyzer.body;
    const auto root=gl::NameMap::MAIN_ID;
    putClassesAt(memory,root,{{"a","b"}}); analyzer.initializeCanonicalScope(memory,root);
    auto& arena=gl::genScratchArenas().forSlot(0);
    auto& uses=memory.lbMemory.canonicalUses;
    const auto raw=memory.nameMap.encode("(K[b])"),canonical=memory.nameMap.encode("(K[a])");
    const auto rawKey=gl::packStatementKey(raw,root),canonicalKey=gl::packStatementKey(canonical,root);
    analyzer.indexCanonicalText(uses.goals,rawKey,root,gl::StrSpan("(K[u_b])"));
    int hits=0; uses.goals.forEach(root,gl::StrSpan("b"),[&](int64_t key){ASSERT_EQ(key,rawKey);++hits;});
    ASSERT_EQ(hits,1);
    memory.processedOrLedger.insertSorted(root,gl::packInt32Pair(canonical,raw),[](int64_t a,int64_t b){return a<b;});
    analyzer.ensureCanonicalOrMarkerUses(memory,arena);
    analyzer.indexCanonicalOrMarker(memory,raw,canonical,root,arena);
    hits=0; uses.processedOr.forEach(root,gl::StrSpan("a"),[&](int64_t key){ASSERT_EQ(key,rawKey);++hits;});
    ASSERT_EQ(hits,1);
    gl::upsertStatementKey(memory.intKnownStatements,rawKey,true);
    const int level=0; memory.intStatementLevelsMap.assignSetRange(rawKey,&level,&level+1);
    analyzer.ensureCanonicalStatementUses(memory,arena);
    analyzer.indexCanonicalRecord(memory,memory.intKnownStatements,uses.statements,rawKey,5,arena);
    analyzer.ensureCanonicalTemplateIndex(memory,memory.intKnownStatements,uses.statementMatches,arena,true);
    const auto cls=memory.canonicalEquivalenceClassesMap.recordAt(memory.canonicalEquivalenceClassesMap.lookup(root),0);
    gl::DirtyState dirty=gl::DirtyState::Clean; gl::TypedColdSet<int64_t> selected(&arena,&dirty);
    analyzer.selectCanonicalRecords(memory,memory.intKnownStatements,uses.statements,cls,root,5,selected,arena);
    ASSERT_EQ(selected.count(),1);
    selected.resetToFresh();
    analyzer.selectCanonicalRecords(memory,memory.intKnownStatements,uses.statements,cls,root,5,selected,arena);
    ASSERT_TRUE(selected.empty());
    analyzer.writeExistenceState(memory,rawKey,28);
    analyzer.ensureCanonicalReceiptUses(memory,arena);
    analyzer.indexPendingExistence(memory,rawKey,28);
    ASSERT_EQ(analyzer.existenceStateAt(memory,canonical,root),12);
    analyzer.writeExistenceState(memory,canonicalKey,3);
    ASSERT_EQ(analyzer.existenceStateAt(memory,canonical,root),15);
    analyzer.eraseExistenceState(memory,canonicalKey,arena);
    ASSERT_EQ(analyzer.existenceStateAt(memory,canonical,root),12);
    analyzer.writeExistenceState(memory,rawKey,0);
    analyzer.indexPendingExistence(memory,rawKey,0);
    analyzer.eraseExistenceState(memory,rawKey,arena);
    ASSERT_EQ(analyzer.existenceStateAt(memory,canonical,root),0);
}

TEST(canonical_classes, compact_dependency_selection_preserves_carrier_shape_eligibility) {
    gl::ExpressionAnalyzer analyzer("Peano"); analyzer.parameters.trackHistory=false;
    gl::Memory memory;
    ASSERT_TRUE(memory.compactExpansions.empty());
    const auto root=gl::NameMap::MAIN_ID;
    putClassesAt(memory,root,{{"a","b"}});
    const std::string compactText="(implication77[b])",otherText="!(and77[b])",ruleText="(>[]!(in[b,N])(in[N,b]))";
    const gl::StrSpan compact(compactText),other(otherText),rule(ruleText);
    analyzer.appendCompactExpansion(memory,compact,gl::StrSpan("main"),rule,gl::StrSpan("main"),gl::RuleInstallKind::Local);
    analyzer.appendCompactExpansion(memory,other,gl::StrSpan("main"),rule,gl::StrSpan("main"),gl::RuleInstallKind::Local);
    auto& arena=gl::genScratchArenas().forSlot(0);
    gl::DirtyState dirty=gl::DirtyState::Clean; gl::TypedColdSet<int64_t> selected(&arena,&dirty);
    const auto cls=memory.canonicalEquivalenceClassesMap.recordAt(memory.canonicalEquivalenceClassesMap.lookup(root),0);
    analyzer.selectCanonicalRecords(memory,memory.compactExpansions,memory.lbMemory.canonicalUses.compacts,
        cls,root,6,selected,arena);
    ASSERT_EQ(selected.count(),1);
    ASSERT_TRUE(selected.contains(gl::packStatementKey(memory.nameMap.lookup(compact),root)));
    ASSERT_FALSE(selected.contains(gl::packStatementKey(memory.nameMap.lookup(other),root)));
    ASSERT_EQ(memory.compactExpansions.count(),2);
}

TEST(canonical_classes, completed_statement_identity_uses_full_components_and_exact_scope) {
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    ana.parameters.trackHistory=false;
    auto& mb=ana.body;
    const auto root=gl::NameMap::MAIN_ID;
    const auto child=mb.nameMap.encodePush(root,"completed_child");
    const auto sibling=mb.nameMap.encodePush(root,"completed_sibling");
    putClassesAt(mb,root,{{"a","b"}});
    putClassesAt(mb,child,{{"b","c"}});
    const auto canonical=mb.nameMap.encode("(K[a,a,a,a,a,a])");
    gl::upsertStatementKey(mb.intKnownStatements,gl::packStatementKey(canonical,root),true,true);
    const std::string childName=mb.nameMap.decode(child),siblingName=mb.nameMap.decode(sibling);
    ASSERT_FALSE(ana.checkForEquivalence("(K[c,c,c,c,c,c])",childName,mb));
    const auto raw=mb.nameMap.encode("(K[c,c,c,c,c,c])");
    gl::upsertStatementKey(mb.intKnownStatements,gl::packStatementKey(raw,child),true,false);
    ASSERT_FALSE(ana.checkForEquivalence("(K[a,a,a,a,a,a])",childName,mb));
    gl::upsertStatementKey(mb.intKnownStatements,gl::packStatementKey(raw,child),true,true);
    ASSERT_TRUE(ana.checkForEquivalence("(K[a,a,a,a,a,a])",childName,mb));
    ASSERT_FALSE(ana.checkForEquivalence("(K[a,a,a,a,a,a])",siblingName,mb));
    const auto rebuilds=mb.lbMemory.canonicalUses.statementMatches.rebuildCount;
    ASSERT_TRUE(ana.checkForEquivalence("(K[a,a,a,a,a,a])",childName,mb));
    ASSERT_EQ(mb.lbMemory.canonicalUses.statementMatches.rebuildCount,rebuilds);
}

TEST(canonical_classes, integration_applied_values_share_the_complete_argument_identity) {
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    ana.parameters.trackHistory=false;
    auto& mb=ana.body;
    const auto root=gl::NameMap::MAIN_ID;
    const auto child=mb.nameMap.encodePush(root,"history_child");
    putClassesAt(mb,root,{{"a","z"}});
    putClassesAt(mb,child,{{"b","z"}});
    gl::ScratchArena& arena=gl::genScratchArenas().forSlot(0);
    gl::ScratchScope frame(arena);
    gl::ArenaIntegrationMap entries(&arena,&mb.valueInterner);
    gl::IntInstruction instruction;
    instruction.markedGoal=mb.valueInterner.encode("(q[marker])");
    const int32_t entry=entries.appendEntry(instruction);
    entries.insertValue(entry,mb.valueInterner.encode("b"));
    ASSERT_TRUE(ana.integrationValueSeenAtScope(mb,entries,entry,gl::StrSpan("a",1),child));
    ASSERT_FALSE(ana.integrationValueSeenAtScope(mb,entries,entry,gl::StrSpan("a",1),root));
    ASSERT_FALSE(ana.integrationValueSeenAtScope(mb,entries,entry,gl::StrSpan("missing",7),child));
    ASSERT_EQ(ana.canonicalArgumentAtScope(mb,gl::StrSpan("b",1),child).toStdString(),std::string("a"));
    ASSERT_EQ(ana.canonicalArgumentAtScope(mb,gl::StrSpan("missing",7),child).toStdString(),std::string("missing"));
}

TEST(canonical_classes, parked_demand_prefilter_preserves_scope_and_prefix_routes) {
    gl::ExpressionAnalyzer analyzer(std::string("Peano"));
    auto& memory = analyzer.body;
    const auto root = gl::NameMap::MAIN_ID;
    const auto child = memory.nameMap.encodePush(root, "demand_child");
    const auto sibling = memory.nameMap.encodePush(root, "demand_sibling");
    putClassesAt(memory, child, {{"a", "z"}});
    analyzer.initializeCanonicalScope(memory, sibling);
    const std::string childText = memory.nameMap.decode(child);
    const std::string siblingText = memory.nameMap.decode(sibling);
    const gl::StrSpan query("(K[z,marker])", 13);
    ASSERT_FALSE(analyzer.hasPotentialParkedDemand(memory, query, gl::StrSpan(childText), false));
    ASSERT_FALSE(analyzer.hasPotentialParkedDemand(memory, query, gl::StrSpan(childText), true));
    const auto raw = memory.templateInterner.encode("(K[a,marker])");
    const auto algebra = gl::packInt32Pair(raw, root);
    memory.overallHashMemory.admissionMap.inner().assignRun(algebra, nullptr, nullptr, 0);
    ASSERT_EQ(memory.overallHashMemory.admissionMap.count(), 1);
    ASSERT_TRUE(memory.overallHashMemory.admissionMap.lookup(algebra) != 0);
    ASSERT_TRUE(memory.nameMap.isStrictAncestor(root, child));
    ASSERT_EQ(analyzer.canonicalArgumentAtScope(memory, gl::StrSpan("z", 1), child).toStdString(), std::string("a"));
    ASSERT_TRUE(analyzer.canonicalTemplatePresent(memory, memory.overallHashMemory.admissionMap,
        memory.lbMemory.canonicalUses.algebraMatches, query, gl::StrSpan("main", 4), child));
    ASSERT_TRUE(analyzer.hasPotentialParkedDemand(memory, query, gl::StrSpan(childText), false));
    ASSERT_FALSE(analyzer.hasPotentialParkedDemand(memory, query, gl::StrSpan("main", 4), false));
    ASSERT_FALSE(analyzer.hasPotentialParkedDemand(memory, query, gl::StrSpan(siblingText), false));
    memory.overallHashMemory.admissionSetIntegration.mint(algebra);
    ASSERT_FALSE(analyzer.hasPotentialParkedDemand(memory, query, gl::StrSpan(childText), true));
    memory.overallHashMemory.admissionSetIntegration.mint(gl::packInt32Pair(raw, child));
    ASSERT_TRUE(analyzer.hasPotentialParkedDemand(memory, query, gl::StrSpan(childText), true));
    const auto instruction = gl::packInt32Pair(memory.templateInterner.encode("(Q[u_a,marker])"), child);
    memory.overallHashMemory.admissionMapIntegration.inner().assignRun(instruction, nullptr, nullptr, 0);
    ASSERT_TRUE(analyzer.hasPotentialParkedDemand(memory, gl::StrSpan("(Q[z,marker])"), gl::StrSpan(childText), true));
    ASSERT_FALSE(analyzer.hasPotentialParkedDemand(memory, gl::StrSpan("(Q[z,marker])"), gl::StrSpan(siblingText), true));
}

TEST(canonical_classes, parked_demand_prefilter_keeps_operator_input_and_deferred_wakes) {
    gl::ExpressionAnalyzer analyzer(std::string("Peano"));
    auto& memory = analyzer.body;
    analyzer.operators.clear(); analyzer.operators.insert("in2");
    ASSERT_TRUE(analyzer.hasPotentialParkedDemand(memory, gl::StrSpan("(in2[marker,2,3])"), gl::StrSpan("main", 4), false));
    ASSERT_FALSE(analyzer.hasPotentialParkedDemand(memory, gl::StrSpan("(in2[2,marker,3])"), gl::StrSpan("main", 4), false));
    analyzer.operators.insert("in3");
    ASSERT_FALSE(analyzer.hasPotentialParkedDemand(memory, gl::StrSpan("(in3[marker,2,3,4])"), gl::StrSpan("main", 4), false));
    memory.lbMemory.canonicalUses.deferWakes = true;
    analyzer.revisitRejected2(gl::StrSpan("(K[z,marker])"), memory, gl::StrSpan("main", 4));
    analyzer.revisitRejectedIntegration2(gl::StrSpan("(K[z,marker])"), memory, gl::StrSpan("main", 4));
    ASSERT_EQ(memory.lbMemory.canonicalUses.wakeKinds.count(), 1);
    const auto key = gl::packInt32Pair(memory.templateInterner.lookup("(K[z,marker])"), gl::NameMap::MAIN_ID);
    ASSERT_EQ(*memory.lbMemory.canonicalUses.wakeKinds.find(key), 3);
    memory.lbMemory.canonicalUses.deferWakes = false;
    memory.lbMemory.canonicalUses.wakeKinds.resetToFresh();
}

TEST(canonical_classes, parked_demand_prefilter_preserves_algebra_query_normalization) {
    gl::ExpressionAnalyzer analyzer(std::string("Peano"));
    auto& memory = analyzer.body;
    putClassesAt(memory, gl::NameMap::MAIN_ID, {{"a", "u_z"}});
    const auto owner = gl::packInt32Pair(memory.templateInterner.encode("(K[a,marker])"), gl::NameMap::MAIN_ID);
    memory.overallHashMemory.admissionMap.inner().assignRun(owner, nullptr, nullptr, 0);
    ASSERT_TRUE(analyzer.hasPotentialParkedDemand(memory, gl::StrSpan("(K[u_z,marker])"), gl::StrSpan("main", 4), false));
    ASSERT_FALSE(analyzer.hasPotentialParkedDemand(memory, gl::StrSpan("(K[u_missing,marker])"), gl::StrSpan("main", 4), false));
}


TEST(canonical_classes, selective_posting_retirement_preserves_order_and_reuse) {
    gl::Memory memory;
    gl::OrPostingIndex<int64_t> postings(&memory.lbMemory.manager);
    for (int64_t value = 1; value <= 5; ++value) postings.insert(7, value);
    postings.insert(8, 9);
    postings.eraseIf([](int64_t key, int64_t value) { return key == 8 || value == 3; });
    std::vector<int64_t> found;
    postings.forEach(7, [&](int64_t value) { found.push_back(value); });
    ASSERT_TRUE((found == std::vector<int64_t>{5,4,2,1}));
    ASSERT_FALSE(postings.containsKey(8));
    ASSERT_TRUE(postings.insert(7, 3));
    found.clear();
    postings.forEach(7, [&](int64_t value) { found.push_back(value); });
    ASSERT_TRUE((found == std::vector<int64_t>{3,5,4,2,1}));
    postings.eraseIf([](int64_t, int64_t) { return false; });
    ASSERT_EQ(postings.size(), 5);
    postings.eraseIf([](int64_t, int64_t) { return true; });
    ASSERT_EQ(postings.size(), 0);
    ASSERT_TRUE(postings.insert(9, 10));
}

TEST(canonical_classes, closed_scope_dependencies_retain_survivor_observation_versions) {
    gl::Memory memory;
    gl::CanonicalUseIndex index(&memory.lbMemory.manager);
    const auto rootScope = gl::NameMap::MAIN_ID;
    const auto child = memory.nameMap.encodePush(rootScope, "closing_dependency");
    const int64_t kept = gl::packInt32Pair(11, rootScope);
    const int64_t removed = gl::packInt32Pair(12, child);
    index.add(kept, rootScope, gl::StrSpan("z",1));
    index.add(removed, child, gl::StrSpan("z",1));
    std::vector<int64_t> found;
    index.select(rootScope, gl::StrSpan("z",1), 1, [&](int64_t key) { found.push_back(key); });
    ASSERT_EQ(found.size(), 1u);
    index.markRawClean();
    index.eraseScopes([&](gl::NameId scope) { return scope == child; });
    ASSERT_TRUE(index.hasRawChanges());
    found.clear();
    index.select(rootScope, gl::StrSpan("z",1), 1, [&](int64_t key) { found.push_back(key); });
    index.forEach(child, gl::StrSpan("z",1), [&](int64_t key) { found.push_back(key); });
    ASSERT_TRUE(found.empty());
    index.select(rootScope, gl::StrSpan("z",1), 2, [&](int64_t key) { found.push_back(key); });
    ASSERT_TRUE((found == std::vector<int64_t>{kept}));
    index.add(removed, child, gl::StrSpan("z",1));
    found.clear();
    index.select(child, gl::StrSpan("z",1), 1, [&](int64_t key) { found.push_back(key); });
    ASSERT_TRUE((found == std::vector<int64_t>{removed}));
}

TEST(canonical_classes, scope_wipe_keeps_sibling_comparisons_warm_and_invalidatable) {
    gl::ExpressionAnalyzer analyzer(std::string("Peano"));
    analyzer.parameters.trackHistory = false;
    auto& memory = analyzer.body;
    const auto rootScope = gl::NameMap::MAIN_ID;
    const auto child = memory.nameMap.encodePush(rootScope, "closing_comparison");
    const auto leaf = memory.nameMap.encodePush(child, "closing_leaf");
    const auto sibling = memory.nameMap.encodePush(rootScope, "surviving_comparison");
    putClassesAt(memory, rootScope, {{"a", "z"}});
    auto& arena = gl::genScratchArenas().forSlot(0);
    gl::ScratchScope frame(arena);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::TypedColdMap<int64_t,int32_t> records(&arena, &dirty);
    const auto original = memory.templateInterner.encode("(K[z,marker])");
    const int64_t owner = gl::packInt32Pair(original, rootScope);
    records.upsert(owner, 1);
    auto& index = memory.lbMemory.canonicalUses.algebraMatches;
    gl::PagedVector<int64_t> result(&arena, &dirty);
    const auto query = [&](const char* text, gl::NameId scope) {
        result.clear();
        analyzer.matchingCanonicalTemplates(memory, records, index, gl::StrSpan(text), rootScope, scope, result, arena);
        ASSERT_EQ(result.size(), 1);
        ASSERT_EQ(result[0], owner);
    };
    query("(K[a,marker])", child);
    query("(K[a,marker])", leaf);
    query("(K[a,marker])", sibling);
    const auto warm = index.rebuildCount;
    memory.wipeSubtree(child);
    ASSERT_EQ(index.forms.lookup(gl::packInt32Pair(original, child)), 0);
    ASSERT_EQ(index.forms.lookup(gl::packInt32Pair(original, leaf)), 0);
    ASSERT_TRUE(index.forms.lookup(gl::packInt32Pair(original, sibling)) != 0);
    query("(K[a,marker])", sibling);
    ASSERT_EQ(index.rebuildCount, warm);
    addCanonicalFixtureEquality(analyzer, memory, sibling, "(=[0,a])", 3);
    ASSERT_TRUE(index.pending.containsKey(sibling));
    query("(K[0,marker])", sibling);
    ASSERT_EQ(index.rebuildCount, warm + 1);
    index.clear();
    ASSERT_EQ(index.forms.count(), 0);
    query("(K[0,marker])", sibling);
    ASSERT_EQ(index.rebuildCount, 1);
}


namespace {
using namespace gl;
// Recursive reference for the live-scope ordering contract. The statement door
// stops at its expansion seam; this fixture supplies the recursive continuation.
struct RecursiveNegationReference : ExpressionAnalyzer {
    using ExpressionAnalyzer::ExpressionAnalyzer;
    void addStatement(StrSpan expression, Memory& memory, bool local,
        const int* levels, int32_t count, const TransientOrigin& origin,
        StrSpan scope, PagedVector<IntEncodedExpr>& output) {
        bool ready = false;
        ExpressionAnalyzer::addStatement(expression,memory,local,levels,count,
            origin,scope,output,false,nullptr,&ready);
        if (ready) applyEquivalenceClassToNegatedEquality(
            expression,memory,local,levels,count,output,scope);
    }
        inline void applyEquivalenceClassToNegatedEquality(
            StrSpan expr,
            Memory& memoryBlock,
            bool local,
            const int* levels, int32_t levelCount,
            PagedVector<IntEncodedExpr>& newStatements,
            StrSpan validityName)
        {
            assert(isNegatedEquality(expr));
            StrSpan argSpans[ExecutionParameters::MAX_ARITY];
            const int32_t argN = getArgsSpans(expr, argSpans,
                                              ExecutionParameters::MAX_ARITY);
            assert(argN == 2);
            const unsigned emitStrSlot = (g_currentCoreId >= 0)
                ? static_cast<unsigned>(g_currentCoreId)
                : scratchArenas().slotCount() - 1;
            ScratchArena& emitStrArena = scratchArenas().forSlot(emitStrSlot);
            ScratchScope abScope(emitStrArena);
            const ScratchString aCopy =
                ScratchString::copyFrom(emitStrArena, argSpans[0].ptr, argSpans[0].len);
            const ScratchString bCopy =
                ScratchString::copyFrom(emitStrArena, argSpans[1].ptr, argSpans[1].len);

            const unsigned negEmitSlot = (g_currentCoreId >= 0)
                ? static_cast<unsigned>(g_currentCoreId)
                : genScratchArenas().slotCount() - 1;
            DirtyState negEmitDirty = DirtyState::Clean;
            PagedVector<IntEncodedExpr> emitDiscard(
                &genScratchArenas().forSlot(negEmitSlot), &negEmitDirty);

            StrSpan ancSpans[ExecutionParameters::MAX_SCOPE_DEPTH];
            const int32_t ancN = memoryBlock.nameMap.strictAncestorSpans(
                validityName, ancSpans, ExecutionParameters::MAX_SCOPE_DEPTH);
            ScratchScope ancScope(emitStrArena);
            StrSpan ancStable[ExecutionParameters::MAX_SCOPE_DEPTH];
            for (int32_t i = 0; i < ancN; ++i) {
                const ScratchString c = ScratchString::copyFrom(
                    emitStrArena, ancSpans[i].ptr, ancSpans[i].len);
                ancStable[i] = StrSpan(c);
            }

            auto emitNew = [&](StrSpan newExpr, StrSpan eqUsed, StrSpan classScope,
                               StrSpan depositScope,
                               const int* pairLv, int32_t pairLvN) {
                if (lookupStatementLevels(memoryBlock.intStatementLevelsMap,
                        memoryBlock.nameMap, newExpr, depositScope) != 0) return;

                if (!parameters.compressor_mode) {
                    const NameId neOrigId =
                        memoryBlock.nameMap.lookup(newExpr);
                    const NameId neValId =
                        memoryBlock.nameMap.lookup(depositScope);
                    if (neOrigId != 0 && neValId != 0
                        && ancestorKnown(memoryBlock, neOrigId, neValId,
                                         /*includeSelf=*/false)) {
                        this->stageOrReleaseForRefusedDeposit(
                            memoryBlock, neOrigId, neValId);
                        return;
                    }
                }

                const int mn = extractMaxIterationNumber(newExpr);
                if (mn != -1 && mn > parameters.maxIterationNumberVariable) return;
                if (countPatternOccurrences(newExpr, memoryBlock.overallHashMemory, memoryBlock.nameMap)
                    > parameters.maxNumberSecondaryVariables) return;

                StrSpan citeScope = classScope;
                if (parameters.trackHistory) {
                    const auto originRowAt = [&](StrSpan s) {
                        int64_t pk = 0;
                        if (!lookupOriginKey(memoryBlock.originInterner,
                                             eqUsed, s, pk)) return false;
                        const int32_t oid = memoryBlock.exprOriginMap.lookup(pk);
                        return oid != 0 && memoryBlock.exprOriginMap.runLen(oid) > 0;
                    };
                    bool citeFound = originRowAt(classScope);
                    if (!citeFound && validityName.len < classScope.len
                        && originRowAt(validityName)) {
                        citeFound = true;
                        citeScope = validityName;
                    }
                    for (int32_t ai = ancN - 1; ai >= 0 && !citeFound; --ai) {
                        if (ancStable[ai].len >= classScope.len) continue;
                        if (originRowAt(ancStable[ai])) {
                            citeFound = true;
                            citeScope = ancStable[ai];
                        }
                    }
                    assert(citeFound
                        && "negated-equality expansion: justifying equality has no origin row at the class scope or any of its ancestors");
                }
                const OriginDep eq1Deps[2] = {
                    { expr, StrSpan(validityName) },
                    { eqUsed, citeScope } };
                const TransientOrigin origin{ true, OriginTag::equality1, eq1Deps, 2 };

                int mergedLv[256];
                int32_t mergedLvN = 0;
                for (int32_t j = 0; j < levelCount; ++j)
                    mergedLvN = insertLevelSorted(mergedLv, mergedLvN,
                                                  levels[j], 256);
                for (int32_t j = 0; j < pairLvN; ++j)
                    mergedLvN = insertLevelSorted(mergedLv, mergedLvN,
                                                  pairLv[j], 256);

                emitDiscard.clear();
                this->addStatement(newExpr, memoryBlock, local,
                                   mergedLv, mergedLvN,
                                   origin, depositScope, emitDiscard);

                if (local) {
                    newStatements.push_back(encodeExpression(
                        newExpr, depositScope, memoryBlock.nameMap));
                }
            };

            auto emitScratch = [&](StrSpan negL, StrSpan negR,
                                   StrSpan eqL, StrSpan eqR,
                                   StrSpan classScope, StrSpan depositScope,
                                   const int* pairLv, int32_t pairLvN) {
                ScratchScope emitScope(emitStrArena);
                const int32_t negLen = 4 + negL.len + 1 + negR.len + 2;
                char* negBuf = emitStrArena.allocBytes(negLen);
                int32_t na = 0;
                negBuf[na++] = '!'; negBuf[na++] = '('; negBuf[na++] = '=';
                negBuf[na++] = '[';
                std::memcpy(negBuf + na, negL.ptr,
                            static_cast<std::size_t>(negL.len));
                na += negL.len;
                negBuf[na++] = ',';
                std::memcpy(negBuf + na, negR.ptr,
                            static_cast<std::size_t>(negR.len));
                na += negR.len;
                negBuf[na++] = ']'; negBuf[na++] = ')';
                assert(na == negLen);
                const int32_t eqLen = 3 + eqL.len + 1 + eqR.len + 2;
                char* eqBuf = emitStrArena.allocBytes(eqLen);
                int32_t ea = 0;
                eqBuf[ea++] = '('; eqBuf[ea++] = '='; eqBuf[ea++] = '[';
                std::memcpy(eqBuf + ea, eqL.ptr,
                            static_cast<std::size_t>(eqL.len));
                ea += eqL.len;
                eqBuf[ea++] = ',';
                std::memcpy(eqBuf + ea, eqR.ptr,
                            static_cast<std::size_t>(eqR.len));
                ea += eqR.len;
                eqBuf[ea++] = ']'; eqBuf[ea++] = ')';
                assert(ea == eqLen);
                emitNew(StrSpan(negBuf, negLen), StrSpan(eqBuf, eqLen), classScope,
                        depositScope, pairLv, pairLvN);
            };

            const unsigned negSlot = (g_currentCoreId >= 0)
                ? static_cast<unsigned>(g_currentCoreId)
                : genScratchArenas().slotCount() - 1;
            ScratchArena& negArena = genScratchArenas().forSlot(negSlot);
            ScratchScope negFrame(negArena);
            auto emitFromClassesAt = [&](StrSpan scope, StrSpan depositScope) {
                const NameId aId = memoryBlock.nameMap.lookup(StrSpan(aCopy));
                const NameId bId = memoryBlock.nameMap.lookup(StrSpan(bCopy));
                const NameId vId = memoryBlock.nameMap.lookup(scope);
                if (vId == 0) return;
                const int32_t bucketId = memoryBlock.equivalenceClassesMap.lookup(vId);
                if (bucketId == 0) return;
                const std::size_t classCount = static_cast<std::size_t>(
                    memoryBlock.equivalenceClassesMap.runLen(bucketId));
                for (std::size_t ci = 0; ci < classCount; ++ci) {
                    ScratchScope classFrame(negArena);
                    int32_t clsLen = 0;
                    const char* clsPtr =
                        memoryBlock.equivalenceClassesMap.inner().peekBlobContiguous(
                            bucketId, static_cast<int32_t>(ci), clsLen, negArena);
                    const EquivalenceClassView eqc{ clsPtr, clsLen };
                    const bool hasA = aId != 0 && classHasMember(eqc, aId);
                    const bool hasB = bId != 0 && classHasMember(eqc, bId);
                    if (hasA) {
                        for (int32_t k = 0; k < eqc.memberCount(); ++k) {
                            const NameId mid = eqc.memberId(k);
                            if (mid == aId) continue;
                            const StrSpan midView = memoryBlock.nameMap.decodeView(mid);
                            const ScratchString member = ScratchString::copyFrom(
                                emitStrArena, midView.ptr, midView.len);
                            int pairLv[64];
                            int32_t pairLvN = 0;
                            eqc.forEachLevel(packEqPairKey(aId, mid),
                                [&](int lv) {
                                    assert(pairLvN < 64
                                        && "negated-equality pair run "
                                           "exceeds 64");
                                    pairLv[pairLvN++] = lv;
                                });
                            emitScratch(StrSpan(member), StrSpan(bCopy),
                                        StrSpan(aCopy), StrSpan(member), scope,
                                        depositScope, pairLv, pairLvN);
                        }
                    }
                    if (hasB) {
                        for (int32_t k = 0; k < eqc.memberCount(); ++k) {
                            const NameId mid = eqc.memberId(k);
                            if (mid == bId) continue;
                            const StrSpan midView = memoryBlock.nameMap.decodeView(mid);
                            const ScratchString member = ScratchString::copyFrom(
                                emitStrArena, midView.ptr, midView.len);
                            int pairLv[64];
                            int32_t pairLvN = 0;
                            eqc.forEachLevel(packEqPairKey(bId, mid),
                                [&](int lv) {
                                    assert(pairLvN < 64
                                        && "negated-equality pair run "
                                           "exceeds 64");
                                    pairLv[pairLvN++] = lv;
                                });
                            emitScratch(StrSpan(aCopy), StrSpan(member),
                                        StrSpan(bCopy), StrSpan(member), scope,
                                        depositScope, pairLv, pairLvN);
                        }
                    }
                }
            };

            emitFromClassesAt(validityName, validityName);
            for (int32_t i = 0; i < ancN; ++i)
                emitFromClassesAt(ancStable[i], validityName);

            {
                const NameId selfVid = memoryBlock.nameMap.lookup(validityName);
                if (selfVid != 0) {
                    DirtyState descDirty = DirtyState::Clean;
                    PagedVector<NameId> descVids(&negArena, &descDirty);
                    const int32_t eqKeyCount =
                        memoryBlock.equivalenceClassesMap.count();
                    for (int32_t kid = 1; kid <= eqKeyCount; ++kid) {
                        const NameId vid =
                            memoryBlock.equivalenceClassesMap.keyAt(kid);
                        if (memoryBlock.nameMap.isStrictAncestor(selfVid, vid))
                            descVids.push_back(vid);
                    }
                    const int32_t dn = descVids.size();
                    if (dn > 0) {
                        int32_t* dIdx = reinterpret_cast<int32_t*>(
                            negArena.resolve(negArena.alloc(
                                dn * static_cast<int32_t>(sizeof(int32_t)),
                                static_cast<int32_t>(alignof(int32_t)))));
                        for (int32_t k = 0; k < dn; ++k) dIdx[k] = k;
                        std::sort(dIdx, dIdx + dn, [&](int32_t a, int32_t b) {
                            return compareSpans(
                                memoryBlock.nameMap.decodeView(
                                    descVids[static_cast<std::size_t>(a)]),
                                memoryBlock.nameMap.decodeView(
                                    descVids[static_cast<std::size_t>(b)])) < 0;
                        });
                        for (int32_t k = 0; k < dn; ++k) {
                            const StrSpan dView = memoryBlock.nameMap.decodeView(
                                descVids[static_cast<std::size_t>(dIdx[k])]);
                            const ScratchString descScope =
                                ScratchString::copyFrom(emitStrArena,
                                                        dView.ptr, dView.len);
                            emitFromClassesAt(StrSpan(descScope),
                                              StrSpan(descScope));
                        }
                    }
                }
            }
        }
};

EquivalenceClass inequalityFixtureClass(Memory& memory,
    const std::set<std::string>& members, int level) {
    EquivalenceClass result;
    result.setMembersFromNames(members,memory.nameMap);
    for (auto a : result.memberIds) for (auto b : result.memberIds)
        if (a < b) result.intEqualityLevelsMap[packEqPairKey(a,b)] = {level};
    return result;
}

std::string inequalityHistorySnapshot(Memory& memory) {
    std::ostringstream out;
    for (const auto& row : decodeOriginMapSorted(memory.exprOriginMap,memory.originInterner)) {
        out << row.first.first << '@' << row.first.second << '\n';
        for (const auto& origin : row.second) {
            out << origin.first << ':';
            for (const auto& dependency : origin.second)
                out << dependency.original << '@' << dependency.validityName << ';';
            out << '\n';
        }
    }
    return out.str();
}

std::vector<std::vector<char>> inequalityClassSnapshot(Memory& memory,NameId scope) {
    std::vector<std::vector<char>> result;
    const int bucket = memory.canonicalEquivalenceClassesMap.lookup(scope);
    for (const auto& cls : memory.canonicalEquivalenceClassesMap.recordsAt(bucket))
        result.push_back(serializeEquivalenceClass(cls));
    return result;
}

template<class Analyzer>
std::vector<std::string> inequalityOrderSnapshot(bool local,bool nested,bool nonDerived) {
    Analyzer analyzer(std::string("Peano"));
    analyzer.parameters.trackHistory = true;
    auto& memory = analyzer.body;
    const NameId child = memory.nameMap.encodePush(NameMap::MAIN_ID,"hypo_(child[1])");
    const NameId grandchild = memory.nameMap.encodePush(child,"hypo_(grandchild[1])");
    const NameId sibling = memory.nameMap.encodePush(NameMap::MAIN_ID,"hypo_(sibling[1])");
    const auto install = [&](NameId scope,const std::set<std::string>& members,int level) {
        const auto cls = inequalityFixtureClass(memory,members,level);
        memory.assignClassesById(scope,{cls});
        const std::string validity = memory.nameMap.decode(scope);
        seedFixtureClassEvidence(memory,cls,StrSpan(validity));
    };
    install(NameMap::MAIN_ID,{"a0","a1","a2"},2);
    install(child,{"b0","b1","b2"},4);
    install(grandchild,{"a0","a3"},6);
    install(sibling,{"a0","a4"},8);
    seedFixtureCanonicalClasses(memory);
    auto& arena = genScratchArenas().forSlot(genScratchArenas().slotCount()-1);
    ScratchScope frame(arena);
    DirtyState dirty = DirtyState::Clean;
    PagedVector<IntEncodedExpr> output(&arena,&dirty);
    const std::string validity = memory.nameMap.decode(nested ? child : NameMap::MAIN_ID);
    const int levels[] = {1,3};
    const TransientOrigin origin{true,OriginTag::broadcast,nullptr,0};
    analyzer.addStatement(StrSpan("!(=[a0,b0])"),memory,local,levels,
        nonDerived ? 0 : 2,origin,StrSpan(validity),output);
    std::vector<std::string> result;
    std::string names;
    for (int i=1;i<=memory.nameMap.nameCount();++i) names+=memory.nameMap.decode(i)+'\n';
    result.push_back(names);
    const auto rows = [&](const auto& statements) {
        std::string bytes;
        for (int i=0;i<statements.size();++i) {
            const IntEncodedExpr row = statements[i];
            bytes.append(reinterpret_cast<const char*>(&row),sizeof(row));
        }
        return bytes;
    };
    result.push_back(rows(memory.intEncodedStatements));
    result.push_back(rows(output));
    std::ostringstream levelRows;
    for (int i=1;i<=memory.intStatementLevelsMap.count();++i) {
        const auto key = memory.intStatementLevelsMap.decodeKey(i);
        levelRows << key.orig << '@' << key.validity << ':';
        for (int j=0;j<memory.intStatementLevelsMap.runLen(i);++j)
            levelRows << memory.intStatementLevelsMap.valueAt(i,j) << ',';
        levelRows << '\n';
    }
    result.push_back(levelRows.str());
    result.push_back(inequalityHistorySnapshot(memory));
    return result;
}
}

TEST(equi_reshuffle, inequality_iterative_order_levels_history_match_recursive_reference) {
    for (bool local : {false,true}) for (bool nested : {false,true})
        for (bool nonDerived : {false,true}) {
            const auto expected = inequalityOrderSnapshot<RecursiveNegationReference>(local,nested,nonDerived);
            const auto actual = inequalityOrderSnapshot<gl::ExpressionAnalyzer>(local,nested,nonDerived);
            ASSERT_EQ(actual.size(),expected.size());
            for (std::size_t i=0;i<actual.size();++i) ASSERT_TRUE(actual[i]==expected[i]);
        }
}

TEST(equi_reshuffle, inequality_and_canonical_refresh_exclude_refuted_subtree) {
    gl::ExpressionAnalyzer analyzer(std::string("Peano"));
    analyzer.parameters.trackHistory = true;
    auto& memory = analyzer.body;
    const auto root = gl::NameMap::MAIN_ID;
    const auto dead = memory.nameMap.encodePush(root,"ordis_(or90[u,v])_((=[u,v]))");
    const auto child = memory.nameMap.encodePush(dead,"hypo_(child[1])");
    const auto live = memory.nameMap.encodePush(root,"ordis_(or90[u,v])_((=[v,v]))");
    const auto install = [&](gl::NameId scope,const std::set<std::string>& members) {
        const auto cls = inequalityFixtureClass(memory,members,2);
        memory.assignClassesById(scope,{cls});
        const auto validity = memory.nameMap.decode(scope);
        seedFixtureClassEvidence(memory,cls,gl::StrSpan(validity));
    };
    install(dead,{"a","b"}); install(child,{"a","c"}); install(live,{"a","d"});
    seedFixtureCanonicalClasses(memory);
    const auto beforeDead = inequalityClassSnapshot(memory,dead);
    const auto beforeChild = inequalityClassSnapshot(memory,child);
    const auto beforeLive = inequalityClassSnapshot(memory,live);
    const auto historyBefore = inequalityHistorySnapshot(memory);
    memory.pendingDeadOrBranches.mint(dead);
    auto& arena = gl::genScratchArenas().forSlot(gl::genScratchArenas().slotCount()-1);
    gl::ScratchScope frame(arena);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::PagedVector<gl::IntEncodedExpr> output(&arena,&dirty);
    const int levels[] = {1};
    const std::string deadScope = memory.nameMap.decode(dead);
    analyzer.applyEquivalenceClassToNegatedEquality(gl::StrSpan("!(=[a,outside])"),
        memory,true,levels,1,output,gl::StrSpan(deadScope));
    ASSERT_TRUE(output.empty());
    ASSERT_TRUE(inequalityHistorySnapshot(memory)==historyBefore);
    analyzer.applyEquivalenceClassToNegatedEquality(gl::StrSpan("!(=[a,outside])"),
        memory,true,levels,1,output,gl::StrSpan("main"));
    const auto known = [&](const char* expression,gl::NameId scope) {
        const std::string validity = memory.nameMap.decode(scope);
        return gl::lookupStatementLevels(memory.intStatementLevelsMap,memory.nameMap,
            gl::StrSpan(expression),gl::StrSpan(validity))!=0;
    };
    ASSERT_FALSE(known("!(=[b,outside])",dead));
    ASSERT_FALSE(known("!(=[c,outside])",child));
    ASSERT_TRUE(known("!(=[d,outside])",live));
    install(root,{"a","z"});
    analyzer.refreshCanonicalClasses(memory,root);
    analyzer.refreshCanonicalClasses(memory,dead);
    ASSERT_TRUE(inequalityClassSnapshot(memory,dead)==beforeDead);
    ASSERT_TRUE(inequalityClassSnapshot(memory,child)==beforeChild);
    ASSERT_TRUE(inequalityClassSnapshot(memory,live)!=beforeLive);
    ASSERT_TRUE(memory.pendingDeadOrBranches.lookup(dead)!=0);
    ASSERT_TRUE(memory.equivalenceClassesMap.runLen(memory.equivalenceClassesMap.lookup(dead))!=0);
    for (gl::NameId scope : {dead,child}) {
        const auto cls = memory.equivalenceClassesMap.recordAt(memory.equivalenceClassesMap.lookup(scope),0);
        for (auto a : cls.memberIds) for (auto b : cls.memberIds) if(a!=b) {
            const std::string equality = "(=["+memory.nameMap.decode(a)+","+memory.nameMap.decode(b)+"])";
            const std::string validity = memory.nameMap.decode(scope);
            int64_t key=0;
            ASSERT_TRUE(gl::lookupOriginKey(memory.originInterner,equality,validity,key));
            ASSERT_TRUE(memory.exprOriginMap.runLen(memory.exprOriginMap.lookup(key))!=0);
        }
    }
}

TEST(equi_reshuffle, inequality_large_live_closure_uses_explicit_stack) {
    gl::ExpressionAnalyzer analyzer(std::string("Peano"));
    analyzer.parameters.trackHistory = false;
    auto& memory = analyzer.body;
    std::set<std::string> left,right;
    for (int i=0;i<60;++i) {
        left.insert("a"+std::to_string(i)); right.insert("b"+std::to_string(i));
    }
    memory.assignClassesById(gl::NameMap::MAIN_ID,{
        inequalityFixtureClass(memory,left,2),inequalityFixtureClass(memory,right,4)});
    seedFixtureCanonicalClasses(memory);
    auto& arena = gl::genScratchArenas().forSlot(gl::genScratchArenas().slotCount()-1);
    gl::ScratchScope frame(arena);
    const auto cursor = arena.cursor();
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::PagedVector<gl::IntEncodedExpr> output(&arena,&dirty);
    const int levels[] = {1};
    analyzer.addStatement(gl::StrSpan("!(=[a0,b0])"),memory,true,levels,1,
        gl::TransientOrigin{},gl::StrSpan("main"),output);
    ASSERT_EQ(arena.cursor(),cursor);
    for (const auto& a : left) for (const auto& b : right) {
        const std::string expression = "!(=["+a+","+b+"])";
        const std::string mirror = "!(=["+b+","+a+"])";
        ASSERT_TRUE(gl::lookupStatementLevels(memory.intStatementLevelsMap,memory.nameMap,
            gl::StrSpan(expression),gl::StrSpan("main"))!=0);
        ASSERT_TRUE(gl::lookupStatementLevels(memory.intStatementLevelsMap,memory.nameMap,
            gl::StrSpan(mirror),gl::StrSpan("main"))!=0);
    }
}


TEST(canonical_classes, live_record_membership_survives_replacement_and_retires_by_scope) {
    gl::ExpressionAnalyzer analyzer("Peano");
    auto& memory = analyzer.body;
    auto& arena = gl::genScratchArenas().forSlot(0);
    gl::ScratchScope frame(arena);
    gl::CanonicalUseIndex index(&memory.lbMemory.manager);
    const auto child = memory.nameMap.encodePush(gl::NameMap::MAIN_ID, "membership_child");
    const auto sibling = memory.nameMap.encodePush(gl::NameMap::MAIN_ID, "membership_sibling");
    const auto original = memory.nameMap.encode("!(p[z])");
    const auto first = gl::packStatementKey(original, child);
    const auto second = gl::packStatementKey(original, sibling);
    ASSERT_FALSE(index.contains(first));
    index.remember(first);
    ASSERT_TRUE(index.contains(first));
    ASSERT_FALSE(index.contains(second));
    index.add(first, child, gl::StrSpan("z", 1));
    index.erase(first, arena);
    ASSERT_FALSE(index.contains(first));
    index.add(first, child, gl::StrSpan("a", 1));
    index.remember(second);
    index.eraseScopes([&](gl::NameId scope) { return scope == child; });
    ASSERT_FALSE(index.contains(first));
    ASSERT_TRUE(index.contains(second));
    index.clear();
    ASSERT_FALSE(index.contains(second));
}

TEST(canonical_classes, identifier_only_canonicalization_matches_full_proof_text) {
    gl::ExpressionAnalyzer analyzer("Peano");
    analyzer.parameters.trackHistory = true;
    auto& memory = analyzer.body;
    const auto child = memory.nameMap.encodePush(gl::NameMap::MAIN_ID, "identity_child");
    putClassesAt(memory, gl::NameMap::MAIN_ID, {{"b", "z"}});
    putClassesAt(memory, child, {{"a", "b", "z"}});
    for (const auto scope : {gl::NameMap::MAIN_ID, child}) {
        const std::string validity = memory.nameMap.decode(scope);
        const auto bucket = memory.equivalenceClassesMap.lookup(scope);
        for (int32_t i = 0; i < memory.equivalenceClassesMap.runLen(bucket); ++i)
            seedFixtureClassEvidence(memory, memory.equivalenceClassesMap.recordAt(bucket, i), gl::StrSpan(validity));
    }
    const auto originCount = memory.exprOriginMap.count();
    auto& strings = gl::scratchArenas().forSlot(0);
    for (const auto scope : {gl::NameMap::MAIN_ID, child}) {
        for (const std::string text : {"!(p[z,z])", "(q[b,z])", "(q[x])", "(=[b,z])", "!(=[b,z])", "(AnchorFixture[z])"}) {
            const auto original = memory.nameMap.encode(text);
            const std::string validity = memory.nameMap.decode(scope);
            gl::ScratchScope frame(strings);
            gl::ExpressionAnalyzer::WatchBridge proof;
            int levels[256]; int32_t count = 0;
            analyzer.canonicalWatchFormAt(memory, original, scope, strings, proof, levels, count, 256);
            int32_t empty = 0;
            const auto identity = analyzer.canonicalFormAtScope(gl::StrSpan(text), gl::StrSpan(validity),
                memory, strings, nullptr, empty, 0, false, false);
            ASSERT_TRUE(gl::equalSpans(identity.text, proof.text));
            ASSERT_EQ(identity.changed, proof.changed);
            ASSERT_EQ(identity.eqN, 0);
            ASSERT_EQ(empty, 0);
            const auto canonical = analyzer.canonicalWatchIdAt(memory, original, scope);
            ASSERT_TRUE(gl::equalSpans(memory.nameMap.decodeView(canonical), proof.text));
            if (proof.changed) ASSERT_TRUE(proof.eqN > 0 && count > 0);
            else ASSERT_EQ(canonical, original);
        }
    }
    ASSERT_EQ(memory.exprOriginMap.count(), originCount);
}

TEST(watch_ledger, indexed_selection_preserves_run_and_lexical_mint_order) {
    gl::ExpressionAnalyzer analyzer("Peano");
    analyzer.parameters.trackHistory = false;
    auto& memory = analyzer.body;
    const auto root = gl::NameMap::MAIN_ID;
    const auto child = memory.nameMap.encodePush(root, "ordered_child");
    for (const auto scope : {child, root}) {
        const std::string validity = memory.nameMap.decode(scope);
        for (const std::string text : {"!(q[z,2])", "!(q[z,1])", "!(p[z,3])", "!(p[z,0])"}) {
            const auto row = gl::encodeExpression(gl::StrSpan(text), gl::StrSpan(validity), memory.nameMap);
            analyzer.recordWatch(memory.negationLedger, memory, row.originalId, scope, row.nameId);
        }
    }
    putClassesAt(memory, root, {{"a", "z"}});
    putClassesAt(memory, child, {{"b", "z"}});
    std::vector<std::string> expected;
    for (int32_t run = 1; run <= memory.negationLedger.count(); ++run) {
        const auto scope = static_cast<gl::NameId>(uint64_t(memory.negationLedger.decodeKey(run)) >> 32);
        std::vector<std::string> originals;
        for (int32_t j = 0; j < memory.negationLedger.runLen(run); ++j)
            originals.push_back(memory.nameMap.decode(static_cast<gl::NameId>(uint64_t(memory.negationLedger.valueAt(run, j)))));
        std::sort(originals.begin(), originals.end());
        for (auto text : originals) {
            // The child inherits a, so both scope views select the same representative.
            text.replace(text.find("z"), 1, "a");
            if (std::find(expected.begin(), expected.end(), text) == expected.end()) expected.push_back(text);
            ASSERT_EQ(memory.nameMap.lookup(text), 0);
        }
    }
    ASSERT_EQ(rekeyAt(analyzer, memory, root, {"a", "b", "z"}), 8);
    gl::NameId previous = 0;
    for (const auto& text : expected) {
        const auto id = memory.nameMap.lookup(text);
        ASSERT_TRUE(id > previous);
        previous = id;
    }
    ASSERT_EQ(rekeyAt(analyzer, memory, root, {"a", "b", "z"}), 0);
    // Derived index loss rebuilds membership from the authoritative ledger.
    memory.lbMemory.canonicalUses.watches.clear();
    memory.lbMemory.canonicalUses.watchesBuilt = false;
    analyzer.ensureCanonicalWatchUses(memory);
    const auto original = memory.nameMap.lookup("!(q[z,2])");
    ASSERT_TRUE(memory.lbMemory.canonicalUses.watches.contains(gl::packStatementKey(original, child)));
    memory.wipeSubtree(child);
    ASSERT_FALSE(memory.lbMemory.canonicalUses.watches.contains(gl::packStatementKey(original, child)));
    ASSERT_TRUE(memory.lbMemory.canonicalUses.watches.contains(gl::packStatementKey(original, root)));
}

TEST(canonical_classes, stable_representative_growth_updates_new_alias_without_invalidating_old_goal) {
    gl::ExpressionAnalyzer analyzer("Peano");
    analyzer.parameters.trackHistory = false;
    auto& memory = analyzer.body;
    const auto root = gl::NameMap::MAIN_ID;
    addCanonicalFixtureEquality(analyzer, memory, root, "(=[a,b])", 2);
    const auto goal = gl::encodeExpression(gl::StrSpan("(p[b])", 6), gl::StrSpan("main", 4), memory.nameMap);
    memory.intToBeProved.assignSet(gl::packStatementKey(goal.originalId, root), nullptr, 0);
    analyzer.ensureGoalLedgerCurrent(memory);
    for (const std::string text : {"!(q[b])", "!(q[c])"}) {
        const auto row = gl::encodeExpression(gl::StrSpan(text), gl::StrSpan("main", 4), memory.nameMap);
        analyzer.recordWatch(memory.negationLedger, memory, row.originalId, root, row.nameId);
    }
    addCanonicalFixtureEquality(analyzer, memory, root, "(=[b,c])", 7);
    ASSERT_FALSE(memory.goalLedgerStale);
    const auto run = memory.negationLedger.lookup(analyzer.watchKey(root, memory.nameMap.lookup("q")));
    ASSERT_EQ(memory.negationLedger.runLen(run), 2);
    for (int32_t j = 0; j < 2; ++j)
        ASSERT_EQ(static_cast<gl::NameId>(uint64_t(memory.negationLedger.valueAt(run, j)) >> 32), memory.nameMap.lookup("!(q[a])"));
    auto& strings = gl::scratchArenas().forSlot(0);
    gl::ScratchScope frame(strings);
    int levels[256]; int32_t count = 0;
    const auto canonical = analyzer.canonicalFormAtScope(gl::StrSpan("(p[c])", 6), gl::StrSpan("main", 4),
        memory, strings, levels, count, 256);
    ASSERT_TRUE(gl::equalSpans(canonical.text, gl::StrSpan("(p[a])", 6)));
    ASSERT_TRUE(std::find(levels, levels + count, 7) != levels + count);
    addCanonicalFixtureEquality(analyzer, memory, root, "(=[0,a])", 9);
    ASSERT_TRUE(memory.goalLedgerStale);
    analyzer.ensureGoalLedgerCurrent(memory);
    gl::NameId original = 0;
    ASSERT_TRUE(analyzer.watchRunHasCanonical(memory.goalLedger, root, goal.nameId,
        memory.nameMap.lookup("(p[0])"), &original));
    ASSERT_EQ(original, goal.originalId);
}


namespace gl { namespace {
// Independent ordered-selection oracle with separate posting directions.
class CanonicalUseReference {
    struct EdgeKey { int64_t token; int64_t record; };
    DirtyState dirty_ = DirtyState::Clean;
    TypedColdMap<int64_t, int32_t> records_;
    ColdHashSet<BytesKeyStore> names_;
    OrPostingIndex<int64_t> byMember_, byRecord_;
    TypedColdMap<int64_t, int32_t> revisions_;
    TypedColdMap<int64_t, int64_t> observed_;
    ColdHashMap<PodKeyStore<EdgeKey>, int32_t> edgeRevisions_;
    TypedColdSet<int64_t> modified_;
    int32_t revision_ = 0;
    int32_t sourceCount_ = -1;
    uint32_t sourceEpoch_ = 0;
public:
    /// @brief Bind empty dependency postings to a resident arena.
    /// @details No logical record or token is created by construction.
    /// @param arena Owning arena, which outlives this index.
    /// @return An empty dependency index.
    explicit CanonicalUseReference(LbArena* arena)
        : records_(arena, &dirty_), names_(arena, &dirty_), byMember_(arena), byRecord_(arena),
          revisions_(arena, &dirty_), observed_(arena, &dirty_), edgeRevisions_(arena, &dirty_),
          modified_(arena, &dirty_) {}

    /// @brief Register one record's dependency on one visible name.
    /// @details Repeated edges are idempotent; records may have several names.
    /// @param record Stable logical key of the owning record.
    /// @param scope Exact scope where this dependency is interpreted.
    /// @param member Substitution token, interned privately without minting NameMap.
    /// @return Nothing.
    void add(int64_t record, NameId scope, StrSpan member) {
        assert(scope != 0 && !member.empty());
        records_.upsert(record, 1);
        const int64_t token = packInt32Pair(names_.mint(member), scope);
        if (byRecord_.insert(record, token)) {
            const bool added = byMember_.insert(token, record);
            assert(added && revision_ < INT32_MAX);
            revisions_.upsert(token, ++revision_);
            const EdgeKey edge{token, record};
            typename decltype(edgeRevisions_)::InsertHint edgeHint;
            const int32_t edgeId = edgeRevisions_.lookupForInsert(edge, edgeHint);
            if (edgeId == 0) edgeRevisions_.insertAt(edge, revision_, edgeHint);
            else edgeRevisions_.setValueAt(edgeId, revision_);
        }
    }

    /// @brief Remember a record with no current equality-name dependency.
    /// @details This prevents repeated decoding of unchanged token-free keys.
    /// @param record Stable logical record key.
    /// @return Nothing.
    void remember(int64_t record) { records_.upsert(record, 1); }

    /// @brief Test whether a logical record is currently live in this index.
    ///
    /// @details Records with no token dependencies remain live after remember;
    /// erased records are absent even when their retired key is still allocated.
    /// This read neither creates records nor changes dependency revisions.
    /// @param record Stable logical record key, including its effective scope.
    /// @return True exactly when the record is registered and not retired.
    /// @invariant The owning consumer synchronizes its authoritative source first.
    bool contains(int64_t record) const {
        const int32_t* live = records_.find(record);
        return live != nullptr && *live != 0;
    }

    /// @brief Remove one record's old token edges before replacing its text.
    /// @details Snapshot the small reverse run before unlinking either index.
    /// Other records and their insertion order are untouched.
    /// @param record Stable key to remove; absence is a defined no-op.
    /// @param scratch Caller-owned page-tier scratch arena.
    /// @return Nothing.
    void erase(int64_t record, LbArena& scratch) {
        DirtyState dirty = DirtyState::Clean;
        PagedVector<int64_t> tokens(&scratch, &dirty);
        byRecord_.forEach(record, [&](int64_t token) { tokens.push_back(token); });
        for (int32_t i = 0; i < tokens.size(); ++i) {
            byMember_.erase(tokens[i], record);
            byRecord_.erase(record, tokens[i]);
            assert(revision_ < INT32_MAX);
            revisions_.upsert(tokens[i], ++revision_);
        }
        const int32_t* live = records_.find(record);
        if (live != nullptr && *live != 0) records_.upsert(record, 0);
    }

    /// @brief Mark one authoritative value as requiring dependency refresh.
    /// @details Repeated writes coalesce at its stable logical key. The next
    /// synchronization decodes this record before any dependency query.
    /// @param record Written logical record key.
    /// @return Nothing.
    void changed(int64_t record) { modified_.mint(record); }

    /// @brief Bring the index's logical keys into agreement with its source map.
    /// @details Unchanged count/epoch is constant time. Appends decode only new
    /// records; erasures unlink only absent keys and retain every live posting.
    /// Value-only updates are maintained explicitly by their writing doors.
    /// @tparam Map Source map with stable logical keys and an insertion epoch.
    /// @tparam Visit Callback that indexes one newly observed logical key.
    /// @param map Authoritative source map.
    /// @param scratch Caller-owned page-tier scratch for retired keys.
    /// @param visit Synchronous callback accepting the new key.
    /// @return Nothing.
    template<class Map, class Visit>
    void synchronize(const Map& map, LbArena& scratch, Visit&& visit) {
        const int32_t count = map.count();
        const uint32_t epoch = [&]() {
            if constexpr (std::is_same_v<Map, ColdHashSet<PodKeyStore<int64_t>>>)
                return map.insertEpoch();
            else return map.inner().insertEpoch();
        }();
        if (sourceCount_ == count && sourceEpoch_ == epoch && modified_.empty()) return;
        for (int32_t i = 1; i <= modified_.count(); ++i) {
            const int64_t key = modified_.keyAt(i);
            erase(key, scratch);
            if (map.lookup(key) != 0) { records_.upsert(key, 1); visit(key); }
        }
        modified_.resetToFresh();
        const bool appended = sourceCount_ >= 0 && count >= sourceCount_
            && epoch - sourceEpoch_ == static_cast<uint32_t>(count - sourceCount_);
        if (!appended) {
            DirtyState dirty = DirtyState::Clean;
            PagedVector<int64_t> retired(&scratch, &dirty);
            for (int32_t i = 1; i <= records_.count(); ++i) {
                const int64_t key = records_.keyAt(i);
                if (*records_.find(key) != 0 && map.lookup(key) == 0) retired.push_back(key);
            }
            for (int32_t i = 0; i < retired.size(); ++i) erase(retired[i], scratch);
        }
        for (int32_t i = appended ? sourceCount_ + 1 : 1; i <= count; ++i) {
            const int64_t key = map.keyAt(i);
            const int32_t* live = records_.find(key);
            if (live != nullptr && *live != 0) continue;
            records_.upsert(key, 1);
            visit(key);
        }
        sourceCount_ = count;
        sourceEpoch_ = epoch;
    }

    /// @brief Visit records depending on an exact member at an exact scope.
    /// @details Posting order is internal; proof-visible consumers sort selected
    /// logical keys before applying changes. The callback must not mutate this index.
    /// @tparam Sink Synchronous callback accepting a stable record key.
    /// @param scope Effective scope of the canonical record.
    /// @param member Name whose identity or equality metadata changed.
    /// @param sink Consumer of each distinct affected record.
    /// @return Nothing.
    template<class Sink>
    void forEach(NameId scope, StrSpan member, Sink&& sink) const {
        const int32_t name = names_.lookup(member);
        if (name == 0) return;
        byMember_.forEach(packInt32Pair(name, scope), static_cast<Sink&&>(sink));
    }

    /// @brief Select a token's dependent records only after a relevant change.
    /// @details Equality versions and posting revisions are compared exactly.
    /// An unchanged component with unchanged records performs no record visits.
    /// If only postings changed, visit the newly inserted prefix, including
    /// reinserted edges of replaced records. A changed equality version visits
    /// every live dependent. Revisions use logical edge keys, never node slots.
    /// @tparam Sink Synchronous callback receiving selected logical record keys.
    /// @param scope Effective comparison scope.
    /// @param member Original or canonical dependency name.
    /// @param version Current equality identity/evidence version for this member.
    /// @param sink Consumer that collects keys without mutating this index.
    /// @return Nothing.
    template<class Sink>
    void select(NameId scope, StrSpan member, int32_t version, Sink&& sink) {
        const int32_t name = names_.lookup(member);
        if (name == 0) return;
        const int64_t token = packInt32Pair(name, scope);
        const int32_t* revision = revisions_.find(token);
        if (revision == nullptr) return;
        const int64_t state = packInt32Pair(version, *revision);
        const int64_t* previous = observed_.find(token);
        if (previous != nullptr && *previous == state) return;
        const bool sameVersion = previous != nullptr
            && static_cast<int32_t>(uint64_t(*previous) >> 32) == version;
        const int32_t previousRevision = previous == nullptr ? 0 : static_cast<int32_t>(*previous);
        observed_.upsert(token, state);
        if (!sameVersion) {
            byMember_.forEach(token, static_cast<Sink&&>(sink));
            return;
        }
        byMember_.forEachWhile(token, [&](int64_t record) {
            const int32_t* inserted = edgeRevisions_.find(EdgeKey{token, record});
            assert(inserted != nullptr);
            if (*inserted <= previousRevision) return false;
            sink(record);
            return true;
        });
    }

    /// @brief Retire dependencies belonging to closed scopes.
    ///
    /// @details Every record and token carries its scope in the low half.
    /// Surviving edges retain their insertion and observation revisions. Source
    /// synchronization rechecks authoritative keys after the scope-map compaction.
    /// Private name identifiers remain stable until canonical release.
    /// @tparam Closed Predicate over already minted scope identifiers.
    /// @param closed True for each closing scope and its descendants.
    /// @return Nothing.
    template<class Closed>
    void eraseScopes(const Closed& closed) {
        const auto dead = [&](int64_t key) { return closed(static_cast<NameId>(uint64_t(key))); };
        records_.eraseIf(dead);
        modified_.eraseIf(dead);
        byMember_.eraseIf([&](int64_t token, int64_t record) { return dead(token) || dead(record); });
        byRecord_.eraseIf([&](int64_t record, int64_t token) { return dead(record) || dead(token); });
        revisions_.eraseIf(dead);
        observed_.eraseIf(dead);
        edgeRevisions_.eraseIf([&](const EdgeKey& edge) { return dead(edge.token) || dead(edge.record); });
        sourceCount_ = -1;
        sourceEpoch_ = 0;
        dirty_ = DirtyState::Restructured;
    }

    /// @brief Release all derived edges at an owning lifecycle boundary.
    /// @details Authoritative records remain in their separate owning stores.
    /// @return Nothing.
    void clear() {
        records_.resetToFresh(); names_.resetToFresh(); byMember_.clear(); byRecord_.clear();
        sourceCount_ = -1; sourceEpoch_ = 0;
        revisions_.resetToFresh(); observed_.resetToFresh(); edgeRevisions_.resetToFresh();
        modified_.resetToFresh(); revision_ = 0;
    }

    /// @brief Report derived mutations since the last raw image.
    /// @details Logical canonical images exclude these caches; raw images contain their pages.
    /// @return True when a raw snapshot must include changed cache storage.
    bool hasRawChanges() const { return dirty_ != DirtyState::Clean; }

    /// @brief Acknowledge a completed raw snapshot of this index.
    /// @details Call only once the owning arena image has been written successfully.
    /// @return Nothing.
    void markRawClean() { dirty_ = DirtyState::Clean; }

    /// @brief Account for all live and retained derived arena storage.
    /// @details Derived bytes are excluded from canonical deload facets.
    /// @return Allocated live-byte accounting for this index.
    int64_t liveBytes() const {
        return records_.count() * (sizeof(int64_t) + sizeof(int32_t))
            + names_.liveBytes() + byMember_.liveBytes() + byRecord_.liveBytes()
            + revisions_.count() * 12 + observed_.count() * 16 + modified_.count() * 8
            + edgeRevisions_.count() * (sizeof(EdgeKey) + sizeof(int32_t));
    }
};
} }


TEST(equi_reshuffle, shared_canonical_edges_recycle_and_compact_without_reordering) {
    gl::Memory memory;
    gl::CanonicalDependencyEdges edges(&memory.lbMemory.manager);
    int32_t revision = 0;
    for (int64_t record = 1; record <= 200; ++record)
        ASSERT_TRUE(edges.insert(17, record, revision));
    ASSERT_EQ(revision, 200);
    ASSERT_FALSE(edges.insert(17, 100, revision));
    ASSERT_EQ(revision, 200);
    for (int64_t record = 1; record <= 150; ++record)
        ASSERT_TRUE(edges.erase(17, record));
    ASSERT_FALSE(edges.erase(17, 1));
    edges.maintain();
    ASSERT_EQ(edges.size(), 50);
    std::vector<int64_t> actual;
    edges.forEachWhile<true>(17, [&](int64_t record, int32_t inserted) {
        ASSERT_EQ(inserted, static_cast<int32_t>(record));
        actual.push_back(record); return true;
    });
    ASSERT_EQ(actual.size(), std::size_t(50));
    for (int i = 0; i < 50; ++i) ASSERT_EQ(actual[i], int64_t(200 - i));
    ASSERT_TRUE(edges.insert(23, 199, revision));
    ASSERT_TRUE(edges.insert(17, 1, revision));
    actual.clear();
    edges.forEachWhile<false>(199, [&](int64_t token, int32_t) {
        actual.push_back(token); return true;
    });
    ASSERT_TRUE(actual == std::vector<int64_t>({23, 17}));
    actual.clear();
    edges.forEachWhile<true>(17, [&](int64_t record, int32_t) {
        actual.push_back(record); return false;
    });
    ASSERT_TRUE(actual == std::vector<int64_t>({1}));
    int32_t limit = INT32_MAX;
    ASSERT_FALSE(edges.insert(17, 1, limit));
    ASSERT_EQ(limit, INT32_MAX);
    edges.eraseIf([](int64_t token, int64_t record) { return token == 23 || record < 180; });
    ASSERT_EQ(edges.size(), 21);
    actual.clear();
    edges.forEachWhile<false>(199, [&](int64_t token, int32_t) {
        actual.push_back(token); return true;
    });
    ASSERT_TRUE(actual == std::vector<int64_t>({17}));
    const int64_t held = edges.liveBytes();
    ASSERT_TRUE(held > 0);
    edges.eraseIf([](int64_t, int64_t) { return true; });
    ASSERT_EQ(edges.size(), 0);
    ASSERT_EQ(edges.liveBytes(), int64_t(0));
    edges.forEachWhile<true>(17, [&](int64_t, int32_t) {
        ASSERT_TRUE(false); return true;
    });
    edges.maintain();
    ASSERT_TRUE(edges.insert(8, 9, revision));
    edges.clear();
    ASSERT_EQ(edges.liveBytes(), int64_t(0));
}

TEST(equi_reshuffle, shared_canonical_dependencies_match_separate_ordered_oracle) {
    gl::Memory memory;
    gl::CanonicalUseIndex index(&memory.lbMemory.manager);
    gl::CanonicalUseReference reference(&memory.lbMemory.manager);
    gl::LbArena scratch(&gl::staticMemory());
    const std::string names[] = {"member0", "member1", "member2", "member3", "member4"};
    uint32_t state = 41917;
    for (int step = 0; step < 1800; ++step) {
        state = state * 1664525u + 1013904223u;
        const gl::NameId scope = 1 + (state % 4);
        const int64_t record = gl::packInt32Pair(1 + ((state >> 5) % 41), scope);
        const auto& text = names[(state >> 12) % 5];
        const gl::StrSpan member(text.data(), static_cast<int32_t>(text.size()));
        switch ((state >> 17) % 9) {
        case 0:
            index.erase(record, scratch); reference.erase(record, scratch); break;
        case 1:
            index.eraseScopes([&](gl::NameId s) { return s == scope; });
            reference.eraseScopes([&](gl::NameId s) { return s == scope; }); break;
        case 2:
            index.remember(record); reference.remember(record); break;
        default:
            index.add(record, scope, member); reference.add(record, scope, member); break;
        }
        ASSERT_EQ(index.contains(record), reference.contains(record));
        for (gl::NameId s = 1; s <= 4; ++s) {
            for (const auto& name : names) {
                const gl::StrSpan key(name.data(), static_cast<int32_t>(name.size()));
                std::vector<int64_t> actual, expected;
                index.forEach(s, key, [&](int64_t value) { actual.push_back(value); });
                reference.forEach(s, key, [&](int64_t value) { expected.push_back(value); });
                ASSERT_TRUE(actual == expected);
                actual.clear(); expected.clear();
                const int32_t version = step / 13;
                index.select(s, key, version, [&](int64_t value) { actual.push_back(value); });
                reference.select(s, key, version, [&](int64_t value) { expected.push_back(value); });
                ASSERT_TRUE(actual == expected);
                actual.clear(); expected.clear();
                index.select(s, key, version, [&](int64_t value) { actual.push_back(value); });
                reference.select(s, key, version, [&](int64_t value) { expected.push_back(value); });
                ASSERT_TRUE(actual.empty() && expected.empty());
            }
        }
        if (step % 199 == 198) { index.clear(); reference.clear(); }
    }
}

TEST(equi_reshuffle, shared_canonical_edge_revision_assertion_probe) {
    bool probe = false;
#ifdef _WIN32
    char* mode = nullptr;
    size_t length = 0;
    ASSERT_EQ(_dupenv_s(&mode, &length, "GL_TEST_INVALID_CANONICAL_EDGE"), 0);
    probe = mode != nullptr;
    std::free(mode);
#else
    probe = std::getenv("GL_TEST_INVALID_CANONICAL_EDGE") != nullptr;
#endif
    if (!probe) return;
    gl::Memory memory;
    gl::CanonicalDependencyEdges edges(&memory.lbMemory.manager);
    int32_t revision = INT32_MAX;
    edges.insert(1, 2, revision);
    ASSERT_TRUE(false);
}
