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
/// @brief Regression test for `gl::tests::ProverFixture::cleanUpExpressions`'s
///        retention contract (I-58 / D-93): a class-canonicalization drop
///        removes a statement from the runtime registries ONLY. Its
///        `intKnownStatements` row and its `intStatementLevelsMap`
///        row are immortal — row presence makes every registration
///        path skip row re-creation, and the negated-equality expansion's
///        emit gate reads the levels row as its permanent already-emitted
///        memory. An erased levels row with a surviving registry row makes
///        the statement re-emittable but never re-registrable, and the
///        class expansion regenerates sibling variants without bound.

#include "../logical_block/statement_registry_operations.hpp"
#include "../memory.hpp"
#include <cstddef>
#include <set>
#include <utility>
#include "test_harness.hpp"
#include "prover_fixture.hpp"
#include "canonical_test_helpers.hpp"

#include "../prover.hpp"

// Two distinct statements under the SAME validity, one kept and one dropped
// by the class-canonical filter. The dropped statement must leave the
// statement registry, but BOTH levels rows must survive the cleanup.
TEST(cleanup_levels_retention, dropped_statement_keeps_its_levels_row) {
    gl::tests::ProverFixture analyzer(std::string("Peano"));
    gl::Memory memory;
    const gl::NameId mainId = ((memory).nameMap).encode("main");

    // One equivalence class at the cleanup validity with two IntLev members;
    // canonical is the decoded-lex first (int_0_lev_1_1), so a statement
    // carrying int_0_lev_1_2 holds a non-canonical member and is dropped.
    gl::EquivalenceClass cls;
    cls.setMembersFromNames({ "int_0_lev_1_1", "int_0_lev_1_2" }, ((memory).nameMap));
    const std::vector<char> blob = gl::serializeEquivalenceClass(cls);
    const int32_t lens[1] = { static_cast<int32_t>(blob.size()) };
    ((memory).equivalenceClassesMap).inner().assignRun(mainId, blob.data(), lens, 1);
    seedFixtureCanonicalClasses(memory);

    const gl::EncodedExpression keptSrc(
        "(=[a,int_0_lev_1_1])", ((memory).nameMap).decode(mainId));
    const gl::EncodedExpression droppedSrc(
        "(=[a,int_0_lev_1_2])", ((memory).nameMap).decode(mainId));
    const gl::IntEncodedExpr kept =
        gl::encodeExpression(keptSrc, ((memory).nameMap));
    const gl::IntEncodedExpr dropped =
        gl::encodeExpression(droppedSrc, ((memory).nameMap));
    ASSERT_EQ(kept.validityId, dropped.validityId);
    ASSERT_NE(kept.originalId, dropped.originalId);

    ((memory).intEncodedStatements).push_back(kept);
    ((memory).intEncodedStatements).push_back(dropped);

    const int64_t keptKey =
        gl::packStatementKey(kept.originalId, kept.validityId);
    const int64_t droppedKey =
        gl::packStatementKey(dropped.originalId, dropped.validityId);
    ((memory).intStatementLevelsMap).insertSorted(keptKey, 3);
    ((memory).intStatementLevelsMap).insertSorted(droppedKey, 5);

    cleanUpScopes(analyzer, memory, { mainId });

    // Registry: only the kept statement remains.
    ASSERT_EQ(static_cast<int>(((memory).intEncodedStatements).size()), 1);
    ASSERT_EQ(((memory).intEncodedStatements)[0].originalId, kept.originalId);

    // Levels map: BOTH rows survive — the dropped statement's row is the
    // immortal already-emitted memory, not runtime state.
    ASSERT_NE(((memory).intStatementLevelsMap).lookup(keptKey), 0);
    ASSERT_NE(((memory).intStatementLevelsMap).lookup(droppedKey), 0);
}

// The class-canonical sweep drops every outgoing delta row that names a
// non-canonical class member and ships its history; a dropped positive
// equality at main also ships as a mail statement with its levels, so the
// merge reaches the descendants. A dropped negated equality and a dropped
// merge of ordinary integration witnesses ship no statement; a row at another
// scope is no merge for the mail (D-544).
TEST(cleanup_levels_retention, dropped_merge_ships_as_mail_statement) {
    gl::tests::ProverFixture analyzer(std::string("Peano"));
    gl::Memory memory;
    const gl::NameId mainId = ((memory).nameMap).encode("main");

    gl::EquivalenceClass witnesses;
    witnesses.setMembersFromNames({ "it_0_lev_1_1", "it_0_lev_1_3" }, ((memory).nameMap));
    gl::EquivalenceClass integration;
    integration.setMembersFromNames({ "int_0_lev_1_1", "int_0_lev_1_2" }, ((memory).nameMap));
    const std::vector<char> bw = gl::serializeEquivalenceClass(witnesses);
    const std::vector<char> bi = gl::serializeEquivalenceClass(integration);
    std::vector<char> concat(bw);
    concat.insert(concat.end(), bi.begin(), bi.end());
    const int32_t lens[2] = { static_cast<int32_t>(bw.size()), static_cast<int32_t>(bi.size()) };
    ((memory).equivalenceClassesMap).inner().assignRun(mainId, concat.data(), lens, 2);
    seedFixtureCanonicalClasses(memory);

    const auto deltaRow = [&](const char* text, int level) {
        const gl::IntEncodedExpr row = gl::encodeExpression(
            gl::EncodedExpression(text, ((memory).nameMap).decode(mainId)), ((memory).nameMap));
        const int64_t key = gl::packStatementKey(row.originalId, row.validityId);
        ((memory).intEncodedStatements).push_back(row);
        ((memory).intLocalEncodedStatements).push_back(row);
        ((memory).intLocalEncodedStatementsDelta).push_back(row);
        ((memory).intLocalEncodedStatementsSet).mint(key);
        ((memory).intStatementLevelsMap).insertSorted(key, level);
        return row;
    };
    deltaRow("(=[it_0_lev_1_1,it_0_lev_1_3])", 2);
    deltaRow("!(=[it_0_lev_1_3,5])", 3);
    deltaRow("(=[int_0_lev_1_1,int_0_lev_1_2])", 4);

    cleanUpScopes(analyzer, memory, { mainId });

    // All three rows leave the delta, as before.
    ASSERT_EQ(static_cast<int>(((memory).intLocalEncodedStatementsDelta).size()), 0);
    // Only the merge of the two it_ names travels as a statement, with its levels.
    const gl::Mail mail = gl::routingMailOutToHeap(((memory).mailOut), ((memory).mailOutInterner));
    ASSERT_EQ(mail.statements.size(), static_cast<std::size_t>(1));
    const auto& shipped = *mail.statements.begin();
    ASSERT_TRUE(shipped.first.original == "(=[it_0_lev_1_1,it_0_lev_1_3])");
    ASSERT_TRUE(shipped.first.validityName == "main");
    ASSERT_TRUE(shipped.second == std::set<int>{ 2 });

    // A row at another scope is no merge for the mail.
    const gl::NameId childId = ((memory).nameMap).encodePush(mainId, std::string("child"));
    const gl::IntEncodedExpr childRow = gl::encodeExpression(
        gl::EncodedExpression("(=[it_0_lev_1_1,it_0_lev_1_3])", ((memory).nameMap).decode(childId)),
        ((memory).nameMap));
    ASSERT_FALSE(analyzer.equalityPropagation.shipDroppedMerge(memory, childRow));
    ASSERT_EQ(gl::routingMailOutToHeap(((memory).mailOut), ((memory).mailOutInterner)).statements.size(),
              static_cast<std::size_t>(1));
}

// A cleanup drop compacts `intEncodedStatements`; every persistent per-class
// waterline (eqClassSttmntIndexMapMap) above the dropped position must slide
// down by the number of dropped positions below it, or the next class
// application resumes past the rows that slid under it.
TEST(cleanup_levels_retention, cleanup_repairs_eq_class_waterlines) {
    gl::tests::ProverFixture analyzer(std::string("Peano"));
    gl::Memory memory;
    const gl::NameId mainId = ((memory).nameMap).encode("main");

    gl::EquivalenceClass cls;
    cls.setMembersFromNames({ "int_0_lev_1_1", "int_0_lev_1_2" }, ((memory).nameMap));
    const std::vector<char> blob = gl::serializeEquivalenceClass(cls);
    const int32_t lens[1] = { static_cast<int32_t>(blob.size()) };
    ((memory).equivalenceClassesMap).inner().assignRun(mainId, blob.data(), lens, 1);
    seedFixtureCanonicalClasses(memory);

    // Rows: kept (0), dropped (1), kept (2), dropped (3), kept (4).
    const char* texts[5] = { "(=[a,int_0_lev_1_1])", "(=[a,int_0_lev_1_2])",
                             "(=[b,int_0_lev_1_1])", "(=[b,int_0_lev_1_2])",
                             "(=[c,int_0_lev_1_1])" };
    for (int i = 0; i < 5; ++i) {
        const gl::EncodedExpression src(texts[i], ((memory).nameMap).decode(mainId));
        const gl::IntEncodedExpr row = gl::encodeExpression(src, ((memory).nameMap));
        ((memory).intEncodedStatements).push_back(row);
        ((memory).intStatementLevelsMap).insertSorted(
            gl::packStatementKey(row.originalId, row.validityId), 1);
    }

    // Five waterline keys (ad-hoc member sets; the index map is independent
    // of the class store) at 0, 1, 2, 4, 5.
    const std::vector<gl::NameId> keys[5] = {
        { ((memory).nameMap).encode("k0") }, { ((memory).nameMap).encode("k1") },
        { ((memory).nameMap).encode("k2") }, { ((memory).nameMap).encode("k4") },
        { ((memory).nameMap).encode("k5") } };
    const int before[5] = { 0, 1, 2, 4, 5 };
    for (int i = 0; i < 5; ++i)
        gl::upsertEqClassIndex(((memory).eqClassSttmntIndexMapMap), mainId, keys[i], before[i]);

    cleanUpScopes(analyzer, memory, { mainId });

    ASSERT_EQ(static_cast<int>(((memory).intEncodedStatements).size()), 3);
    // Dropped positions {1, 3}: below 0 -> 0, below 1 -> 0, below 2 -> 1,
    // below 4 -> 2, below 5 -> 2.
    const int after[5] = { 0, 1, 1, 2, 3 };
    for (int i = 0; i < 5; ++i)
        ASSERT_EQ(gl::lookupEqClassIndex(((memory).eqClassSttmntIndexMapMap), mainId, keys[i]),
                  after[i]);
}

// The repair helper itself: an ascending erased run, counted below each
// waterline by binary search — identical for a C array and a PagedVector.
TEST(cleanup_levels_retention, repair_eq_class_waterlines_counts_below) {
    gl::Memory memory;
    const gl::NameId mainId = ((memory).nameMap).encode("main");
    const std::vector<gl::NameId> keys[5] = {
        { ((memory).nameMap).encode("w0") }, { ((memory).nameMap).encode("w1") },
        { ((memory).nameMap).encode("w4") }, { ((memory).nameMap).encode("w7") },
        { ((memory).nameMap).encode("w10") } };
    const int before[5] = { 0, 1, 4, 7, 10 };
    const int32_t erased[4] = { 1, 3, 4, 8 };
    // Strictly below: 0 -> none, 1 -> none, 4 -> {1,3}, 7 -> {1,3,4},
    // 10 -> {1,3,4,8}.
    const int after[5] = { 0, 1, 2, 4, 6 };

    for (int i = 0; i < 5; ++i)
        gl::upsertEqClassIndex(((memory).eqClassSttmntIndexMapMap), mainId, keys[i], before[i]);
    gl::StatementRegistryOperations::repairEqClassWaterlines(memory, erased, 4);
    for (int i = 0; i < 5; ++i)
        ASSERT_EQ(gl::lookupEqClassIndex(((memory).eqClassSttmntIndexMapMap), mainId, keys[i]),
                  after[i]);

    // PagedVector run, same content, same result from the same start.
    for (int i = 0; i < 5; ++i)
        gl::upsertEqClassIndex(((memory).eqClassSttmntIndexMapMap), mainId, keys[i], before[i]);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::PagedVector<int32_t> run(&gl::genScratchArenas().forSlot(0), &dirty);
    for (int k = 0; k < 4; ++k) run.push_back(erased[k]);
    gl::StatementRegistryOperations::repairEqClassWaterlines(memory, run, 4);
    for (int i = 0; i < 5; ++i)
        ASSERT_EQ(gl::lookupEqClassIndex(((memory).eqClassSttmntIndexMapMap), mainId, keys[i]),
                  after[i]);
}

// A removal batch leaves the three statement lists and the class waterlines
// exactly as the same removals made one by one through
// removeExpressionFromMemoryBlock(state 0): repeated requests, a pair present
// at two scopes, a duplicate row, a never-interned name and rows interleaved
// with survivors. An empty batch changes nothing.
TEST(cleanup_levels_retention, row_removal_batch_matches_one_by_one_removal) {
    gl::tests::ProverFixture analyzer(std::string("Peano"));
    // Rows in list order; the local and delta lists hold every second row
    // so their positions differ from the encoded list's.
    const char* rows[9][2] = {
        { "(=[a,b])", "main" }, { "(=[c,d])", "main" }, { "(=[a,b])", "s1" },
        { "(=[e,f])", "main" }, { "(=[a,b])", "main" }, { "(=[g,h])", "s1" },
        { "(=[e,f])", "s1" },   { "(=[i,j])", "main" }, { "(=[g,h])", "main" } };
    const char* removals[6][2] = {
        { "(=[a,b])", "main" }, { "(=[g,h])", "s1" }, { "(=[a,b])", "main" },
        { "(=[z,z])", "main" }, { "(=[e,f])", "s1" }, { "(=[i,j])", "main" } };
    const int waterlines[6] = { 0, 1, 3, 5, 8, 9 };

    auto build = [&](gl::Memory& m, std::vector<gl::NameId> (&keys)[6]) {
        for (int i = 0; i < 9; ++i) {
            const gl::IntEncodedExpr ie = gl::encodeExpression(
                gl::EncodedExpression(rows[i][0], rows[i][1]), ((m).nameMap));
            ((m).intEncodedStatements).push_back(ie);
            if (i % 2 == 0) ((m).intLocalEncodedStatements).push_back(ie);
            else ((m).intLocalEncodedStatementsDelta).push_back(ie);
        }
        const gl::NameId mainId = ((m).nameMap).encode("main");
        for (int i = 0; i < 6; ++i) {
            keys[i] = { ((m).nameMap).encode("w" + std::to_string(i)) };
            gl::upsertEqClassIndex(((m).eqClassSttmntIndexMapMap), mainId, keys[i], waterlines[i]);
        }
    };
    auto sameList = [](const gl::SharedExpressionList& x, const gl::SharedExpressionList& y) {
        ASSERT_EQ(x.size(), y.size());
        for (int32_t i = 0; i < x.size(); ++i) {
            ASSERT_EQ(x[i].originalId, y[i].originalId);
            ASSERT_EQ(x[i].validityId, y[i].validityId);
        }
    };

    gl::Memory oneByOne;
    gl::Memory batched;
    std::vector<gl::NameId> keysA[6];
    std::vector<gl::NameId> keysB[6];
    build(oneByOne, keysA);
    build(batched, keysB);

    for (int r = 0; r < 6; ++r)
        gl::StatementRegistryOperations::removeExpressionFromMemoryBlock(
            gl::StrSpan(std::string(removals[r][0])), gl::StrSpan(std::string(removals[r][1])), oneByOne, 0);

    gl::ScratchArena& arena = gl::genScratchArenas().forSlot(0);
    {
        gl::StatementRegistryOperations::RowRemovalBatch batch(arena);
        // An empty batch changes nothing.
        batch.apply(batched);
        ASSERT_EQ(((batched).intEncodedStatements).size(), 9);
        for (int r = 0; r < 6; ++r) {
            const std::string text(removals[r][0]);
            const std::string scope(removals[r][1]);
            batch.stage(gl::StrSpan(text), gl::StrSpan(scope), batched);
        }
        // Staging alone removes nothing.
        ASSERT_EQ(((batched).intEncodedStatements).size(), 9);
        batch.apply(batched);
    }

    // Removed: (a,b)@main twice, (g,h)@s1, (e,f)@s1, (i,j)@main; four rows
    // stay in the encoded list.
    ASSERT_EQ(((oneByOne).intEncodedStatements).size(), 4);
    sameList(((oneByOne).intEncodedStatements), ((batched).intEncodedStatements));
    sameList(((oneByOne).intLocalEncodedStatements), ((batched).intLocalEncodedStatements));
    sameList(((oneByOne).intLocalEncodedStatementsDelta), ((batched).intLocalEncodedStatementsDelta));
    const gl::NameId mainA = ((oneByOne).nameMap).encode("main");
    const gl::NameId mainB = ((batched).nameMap).encode("main");
    for (int i = 0; i < 6; ++i)
        ASSERT_EQ(gl::lookupEqClassIndex(((oneByOne).eqClassSttmntIndexMapMap), mainA, keysA[i]),
                  gl::lookupEqClassIndex(((batched).eqClassSttmntIndexMapMap), mainB, keysB[i]));
    // Erased encoded positions {0, 4, 5, 6, 7}: waterline 5 -> 3, 8 -> 3, 9 -> 4.
    ASSERT_EQ(gl::lookupEqClassIndex(((batched).eqClassSttmntIndexMapMap), mainB, keysB[3]), 3);
    ASSERT_EQ(gl::lookupEqClassIndex(((batched).eqClassSttmntIndexMapMap), mainB, keysB[5]), 4);
}

namespace {

// Two scheduled scopes (main and a child), one unscheduled sibling scope with
// classes of its own, rows of all three interleaved in every registry, a
// stale local-set key, and five waterlines into the statement registry.
struct ScopeFixture {
    gl::NameId mainId = 0;
    gl::NameId childId = 0;
    gl::NameId otherId = 0;
    int64_t staleKey = 0;
    std::vector<gl::NameId> waterlineKeys[5];
};

ScopeFixture buildScopeFixture(gl::Memory& memory) {
    ScopeFixture f;
    f.mainId = ((memory).nameMap).encode("main");
    f.childId = ((memory).nameMap).encodePush(f.mainId, std::string("hypo_(h[a])"));
    f.otherId = ((memory).nameMap).encodePush(f.mainId, std::string("hypo_(g[b])"));

    // Canonical members are the decoded-lex first: int_0_lev_1_1 at main,
    // int_0_lev_1_3 at the child, int_0_lev_1_5 at the sibling.
    const auto putClass = [&](gl::NameId scope, const std::set<std::string>& names) {
        gl::EquivalenceClass cls;
        cls.setMembersFromNames(names, ((memory).nameMap));
        const std::vector<char> blob = gl::serializeEquivalenceClass(cls);
        const int32_t lens[1] = { static_cast<int32_t>(blob.size()) };
        ((memory).equivalenceClassesMap).inner().assignRun(scope, blob.data(), lens, 1);
    };
    putClass(f.mainId, { "int_0_lev_1_1", "int_0_lev_1_2" });
    putClass(f.childId, { "int_0_lev_1_3", "int_0_lev_1_4" });
    putClass(f.otherId, { "int_0_lev_1_5", "int_0_lev_1_6" });
    seedFixtureCanonicalClasses(memory);

    const auto row = [&](const char* text, gl::NameId scope) {
        const gl::EncodedExpression src(text, ((memory).nameMap).decode(scope));
        return gl::encodeExpression(src, ((memory).nameMap));
    };
    const gl::NameId m = f.mainId, c = f.childId, o = f.otherId;
    const std::pair<const char*, gl::NameId> local[7] = {
        { "(=[a,int_0_lev_1_1])", m }, { "(=[b,int_0_lev_1_4])", c },
        { "(=[c,int_0_lev_1_6])", o }, { "(=[d,int_0_lev_1_2])", m },
        { "(=[e,int_0_lev_1_3])", c }, { "(=[f,int_0_lev_1_1])", m },
        { "(=[g,int_0_lev_1_4])", c } };
    for (const auto& r : local) {
        const gl::IntEncodedExpr ie = row(r.first, r.second);
        ((memory).intLocalEncodedStatements).push_back(ie);
        ((memory).intLocalEncodedStatementsSet).mint(
            gl::packStatementKey(ie.originalId, ie.validityId));
    }
    const gl::IntEncodedExpr stale = row("(=[z,int_0_lev_1_1])", m);
    f.staleKey = gl::packStatementKey(stale.originalId, stale.validityId);
    ((memory).intLocalEncodedStatementsSet).mint(f.staleKey);

    const std::pair<const char*, gl::NameId> delta[4] = {
        { "(=[h,int_0_lev_1_4])", c }, { "(=[i,int_0_lev_1_2])", m },
        { "(=[j,int_0_lev_1_6])", o }, { "(=[k,int_0_lev_1_3])", c } };
    for (const auto& r : delta)
        ((memory).intLocalEncodedStatementsDelta).push_back(row(r.first, r.second));

    const std::pair<const char*, gl::NameId> stmts[9] = {
        { "(=[l,int_0_lev_1_1])", m }, { "(=[m,int_0_lev_1_2])", m },
        { "(=[n,int_0_lev_1_4])", c }, { "(=[o,int_0_lev_1_6])", o },
        { "(=[p,int_0_lev_1_3])", c }, { "(=[q,int_0_lev_1_2])", m },
        { "(=[r,int_0_lev_1_4])", c }, { "(=[s,int_0_lev_1_6])", o },
        { "(=[t,int_0_lev_1_1])", m } };
    for (const auto& r : stmts)
        ((memory).intEncodedStatements).push_back(row(r.first, r.second));

    const char* keyNames[5] = { "w0", "w2", "w4", "w6", "w9" };
    const int positions[5] = { 0, 2, 4, 6, 9 };
    for (int i = 0; i < 5; ++i) {
        f.waterlineKeys[i] = { ((memory).nameMap).encode(keyNames[i]) };
        gl::upsertEqClassIndex(((memory).eqClassSttmntIndexMapMap), f.mainId,
                               f.waterlineKeys[i], positions[i]);
    }
    return f;
}

int32_t rowsAt(const gl::SharedExpressionList& rows, gl::NameId scope) {
    int32_t n = 0;
    for (int32_t i = 0; i < rows.size(); ++i)
        if (rows[i].validityId == scope) ++n;
    return n;
}

void assertSameRows(const gl::SharedExpressionList& a,
                    const gl::SharedExpressionList& b) {
    ASSERT_EQ(a.size(), b.size());
    for (int32_t i = 0; i < a.size(); ++i) {
        ASSERT_EQ(a[i].originalId, b[i].originalId);
        ASSERT_EQ(a[i].validityId, b[i].validityId);
    }
}

} // namespace

// One cleanup call over several scopes must leave exactly the state of one
// call per scope in the same order: the same kept rows in the same order in
// all three registries, the same re-minted local set (stale key purged), the
// same repaired waterlines — while an unscheduled scope's rows pass through
// although its own classes would drop them.
TEST(cleanup_levels_retention, one_pass_over_scopes_equals_one_pass_per_scope) {
    gl::tests::ProverFixture analyzer(std::string("Peano"));
    gl::Memory together;
    gl::Memory apart;
    const ScopeFixture ft = buildScopeFixture(together);
    const ScopeFixture fa = buildScopeFixture(apart);
    ASSERT_EQ(ft.childId, fa.childId);
    ASSERT_EQ(ft.staleKey, fa.staleKey);

    cleanUpScopes(analyzer, together, { ft.mainId, ft.childId });
    cleanUpScopes(analyzer, apart, { fa.mainId });
    cleanUpScopes(analyzer, apart, { fa.childId });

    assertSameRows(((together).intLocalEncodedStatements), ((apart).intLocalEncodedStatements));
    assertSameRows(((together).intLocalEncodedStatementsDelta),
                   ((apart).intLocalEncodedStatementsDelta));
    assertSameRows(((together).intEncodedStatements), ((apart).intEncodedStatements));

    // Not vacuous: both scheduled scopes dropped rows in every registry
    // (local 7 -> 4, delta 4 -> 2, statements 9 -> 5), the sibling's rows
    // all survived.
    ASSERT_EQ(((together).intLocalEncodedStatements).size(), 4);
    ASSERT_EQ(((together).intLocalEncodedStatementsDelta).size(), 2);
    ASSERT_EQ(((together).intEncodedStatements).size(), 5);
    ASSERT_EQ(rowsAt(((together).intLocalEncodedStatements), ft.otherId), 1);
    ASSERT_EQ(rowsAt(((together).intLocalEncodedStatementsDelta), ft.otherId), 1);
    ASSERT_EQ(rowsAt(((together).intEncodedStatements), ft.otherId), 2);

    // The local set holds exactly the kept local rows in both.
    ASSERT_EQ(((together).intLocalEncodedStatementsSet).count(),
              ((apart).intLocalEncodedStatementsSet).count());
    ASSERT_EQ(((together).intLocalEncodedStatementsSet).count(), 4);
    ASSERT_EQ(((together).intLocalEncodedStatementsSet).lookup(ft.staleKey), 0);
    for (int32_t i = 0; i < ((together).intLocalEncodedStatements).size(); ++i) {
        const int64_t key = gl::packStatementKey(
            ((together).intLocalEncodedStatements)[i].originalId,
            ((together).intLocalEncodedStatements)[i].validityId);
        ASSERT_NE(((together).intLocalEncodedStatementsSet).lookup(key), 0);
        ASSERT_NE(((apart).intLocalEncodedStatementsSet).lookup(key), 0);
    }

    // Dropped statement positions {1, 2, 5, 6}: waterlines 0, 2, 4, 6, 9
    // move to 0, 1, 2, 3, 5 — one repair equals the two per-scope repairs.
    const int expected[5] = { 0, 1, 2, 3, 5 };
    for (int i = 0; i < 5; ++i) {
        ASSERT_EQ(gl::lookupEqClassIndex(((together).eqClassSttmntIndexMapMap),
                                         ft.mainId, ft.waterlineKeys[i]),
                  expected[i]);
        ASSERT_EQ(gl::lookupEqClassIndex(((apart).eqClassSttmntIndexMapMap),
                                         fa.mainId, fa.waterlineKeys[i]),
                  expected[i]);
    }
}

TEST(cleanup_levels_retention, cleanup_preserves_surviving_shared_records) {
    gl::tests::ProverFixture analyzer(std::string("Peano"));
    gl::Memory memory;
    const ScopeFixture f = buildScopeFixture(memory);
    const int localKept[4] = { 0, 2, 4, 5 };
    const int deltaKept[2] = { 2, 3 };
    const int mainKept[5] = { 0, 3, 4, 7, 8 };
    const gl::IntEncodedExpr* localAddresses[4];
    const gl::IntEncodedExpr* deltaAddresses[2];
    const gl::IntEncodedExpr* mainAddresses[5];
    for (int i = 0; i < 4; ++i)
        localAddresses[i] = &((memory).intLocalEncodedStatements)[localKept[i]];
    for (int i = 0; i < 2; ++i)
        deltaAddresses[i] = &((memory).intLocalEncodedStatementsDelta)[deltaKept[i]];
    for (int i = 0; i < 5; ++i)
        mainAddresses[i] = &((memory).intEncodedStatements)[mainKept[i]];

    // A repeated position must remain repeated, and a removed main record
    // must remain available through its independent external owner.
    ((memory).intEncodedStatements).push_back(((memory).intEncodedStatements)[0]);
    ((memory).intExternalStatements).push_back(((memory).intEncodedStatements)[1]);
    const gl::IntEncodedExpr external = ((memory).intExternalStatements)[0];
    for (int pass = 0; pass < 2; ++pass) {
        // The second pass drops nothing but must still purge stale set keys.
        ((memory).intLocalEncodedStatementsSet).mint(f.staleKey);
        cleanUpScopes(analyzer, memory, { f.mainId, f.childId });
        ASSERT_EQ(((memory).intLocalEncodedStatements).size(), 4);
        ASSERT_EQ(((memory).intLocalEncodedStatementsDelta).size(), 2);
        ASSERT_EQ(((memory).intEncodedStatements).size(), 6);
        for (int i = 0; i < 4; ++i)
            ASSERT_TRUE(&((memory).intLocalEncodedStatements)[i] == localAddresses[i]);
        for (int i = 0; i < 2; ++i)
            ASSERT_TRUE(&((memory).intLocalEncodedStatementsDelta)[i] == deltaAddresses[i]);
        for (int i = 0; i < 5; ++i)
            ASSERT_TRUE(&((memory).intEncodedStatements)[i] == mainAddresses[i]);
        ASSERT_TRUE(&((memory).intEncodedStatements)[5] == mainAddresses[0]);
        ASSERT_EQ(((memory).intLocalEncodedStatementsSet).lookup(f.staleKey), 0);
        ASSERT_EQ(((memory).intLocalEncodedStatementsSet).count(), 4);
        ASSERT_EQ(((memory).intExternalStatements).size(), 1);
        ASSERT_TRUE(std::memcmp(&((memory).intExternalStatements)[0], &external,
                               sizeof(external)) == 0);
    }
}

TEST(cleanup_levels_retention, cleanup_empty_lists_preserves_external_owner) {
    gl::tests::ProverFixture analyzer(std::string("Peano"));
    gl::Memory memory;
    const ScopeFixture f = buildScopeFixture(memory);
    const gl::IntEncodedExpr dropped = ((memory).intEncodedStatements)[1];
    ((memory).intEncodedStatements).clear();
    ((memory).intLocalEncodedStatements).clear();
    ((memory).intLocalEncodedStatementsDelta).clear();
    ((memory).intEncodedStatements).push_back(dropped);
    ((memory).intEncodedStatements).push_back(dropped);
    ((memory).intLocalEncodedStatements).push_back(dropped);
    ((memory).intLocalEncodedStatementsDelta).push_back(dropped);
    ((memory).intExternalStatements).push_back(dropped);
    for (int i = 0; i < 5; ++i)
        gl::upsertEqClassIndex(((memory).eqClassSttmntIndexMapMap), f.mainId,
                              f.waterlineKeys[i], 2);

    cleanUpScopes(analyzer, memory, { f.mainId, f.childId });
    ASSERT_TRUE(((memory).intEncodedStatements).empty());
    ASSERT_TRUE(((memory).intLocalEncodedStatements).empty());
    ASSERT_TRUE(((memory).intLocalEncodedStatementsDelta).empty());
    ASSERT_EQ(((memory).intLocalEncodedStatementsSet).count(), 0);
    for (int i = 0; i < 5; ++i)
        ASSERT_EQ(gl::lookupEqClassIndex(((memory).eqClassSttmntIndexMapMap),
                                        f.mainId, f.waterlineKeys[i]), 0);
    ASSERT_EQ(((memory).lbMemory).sharedExpressions.count(), 1);
    ASSERT_EQ(((memory).intExternalStatements).size(), 1);
    ASSERT_TRUE(std::memcmp(&((memory).intExternalStatements)[0], &dropped,
                           sizeof(dropped)) == 0);
    ((memory).intExternalStatements).clear();
    ASSERT_EQ(((memory).lbMemory).sharedExpressions.count(), 0);
}

// Scheduled scopes without classes, and an empty schedule, are defined
// no-ops: no row moves and the local set is not rebuilt (its stale key
// survives), exactly as a per-scope call returned before its sweep.
TEST(cleanup_levels_retention, scopes_without_classes_leave_registries_untouched) {
    gl::tests::ProverFixture analyzer(std::string("Peano"));
    gl::Memory memory;
    const gl::NameId mainId = ((memory).nameMap).encode("main");
    const gl::NameId childId =
        ((memory).nameMap).encodePush(mainId, std::string("hypo_(h[a])"));
    const char* texts[3] = { "(=[a,int_0_lev_1_1])", "(=[b,int_0_lev_1_2])",
                             "(=[c,int_0_lev_1_2])" };
    for (int i = 0; i < 3; ++i) {
        const gl::EncodedExpression src(texts[i],
            ((memory).nameMap).decode(i == 1 ? childId : mainId));
        const gl::IntEncodedExpr ie = gl::encodeExpression(src, ((memory).nameMap));
        ((memory).intLocalEncodedStatements).push_back(ie);
        ((memory).intEncodedStatements).push_back(ie);
    }
    const int64_t staleKey = gl::packStatementKey(
        ((memory).nameMap).encode("(=[z,int_0_lev_1_1])"), mainId);
    ((memory).intLocalEncodedStatementsSet).mint(staleKey);

    cleanUpScopes(analyzer, memory, { mainId, childId });
    cleanUpScopes(analyzer, memory, {});

    ASSERT_EQ(((memory).intLocalEncodedStatements).size(), 3);
    ASSERT_EQ(((memory).intEncodedStatements).size(), 3);
    ASSERT_NE(((memory).intLocalEncodedStatementsSet).lookup(staleKey), 0);
}
