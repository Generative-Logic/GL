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
/// @brief Regression test for `ExpressionAnalyzer::cleanUpExpressions`'s
///        retention contract (I-58 / D-93): a class-canonicalization drop
///        removes a statement from the runtime registries ONLY. Its
///        `intKnownStatements` row and its `intStatementLevelsMap`
///        row are immortal — row presence makes every registration
///        path skip row re-creation, and the negated-equality expansion's
///        emit gate reads the levels row as its permanent already-emitted
///        memory. An erased levels row with a surviving registry row makes
///        the statement re-emittable but never re-registrable, and the
///        class expansion regenerates sibling variants without bound.

#include "test_harness.hpp"

#include "../prover.hpp"

// Two distinct statements under the SAME validity, one kept and one dropped
// by the class-canonical filter. The dropped statement must leave the
// statement registry, but BOTH levels rows must survive the cleanup.
TEST(cleanup_levels_retention, dropped_statement_keeps_its_levels_row) {
    gl::ExpressionAnalyzer analyzer(std::string("Peano"));
    gl::Memory memory;
    const gl::NameId mainId = memory.nameMap.encode("main");

    // One equivalence class at the cleanup validity with two IntLev members;
    // canonical is the decoded-lex first (int_lev_1_1), so a statement
    // carrying int_lev_1_2 holds a non-canonical member and is dropped.
    gl::EquivalenceClass cls;
    cls.setMembersFromNames({ "int_lev_1_1", "int_lev_1_2" }, memory.nameMap);
    const std::vector<char> blob = gl::serializeEquivalenceClass(cls);
    const int32_t lens[1] = { static_cast<int32_t>(blob.size()) };
    memory.equivalenceClassesMap.inner().assignRun(mainId, blob.data(), lens, 1);

    const gl::EncodedExpression keptSrc(
        "(=[a,int_lev_1_1])", memory.nameMap.decode(mainId));
    const gl::EncodedExpression droppedSrc(
        "(=[a,int_lev_1_2])", memory.nameMap.decode(mainId));
    const gl::IntEncodedExpr kept =
        gl::encodeExpression(keptSrc, memory.nameMap);
    const gl::IntEncodedExpr dropped =
        gl::encodeExpression(droppedSrc, memory.nameMap);
    ASSERT_EQ(kept.validityId, dropped.validityId);
    ASSERT_NE(kept.originalId, dropped.originalId);

    memory.intEncodedStatements.push_back(kept);
    memory.intEncodedStatements.push_back(dropped);

    const int64_t keptKey =
        gl::packStatementKey(kept.originalId, kept.validityId);
    const int64_t droppedKey =
        gl::packStatementKey(dropped.originalId, dropped.validityId);
    memory.intStatementLevelsMap.insertSorted(keptKey, 3);
    memory.intStatementLevelsMap.insertSorted(droppedKey, 5);

    analyzer.cleanUpExpressions(memory, gl::StrSpan("main", 4));

    // Registry: only the kept statement remains.
    ASSERT_EQ(static_cast<int>(memory.intEncodedStatements.size()), 1);
    ASSERT_EQ(memory.intEncodedStatements[0].originalId, kept.originalId);

    // Levels map: BOTH rows survive — the dropped statement's row is the
    // immortal already-emitted memory, not runtime state.
    ASSERT_NE(memory.intStatementLevelsMap.lookup(keptKey), 0);
    ASSERT_NE(memory.intStatementLevelsMap.lookup(droppedKey), 0);
}

// A cleanup drop compacts `intEncodedStatements`; every persistent per-class
// waterline (eqClassSttmntIndexMapMap) above the dropped position must slide
// down by the number of dropped positions below it, or the next class
// application resumes past the rows that slid under it.
TEST(cleanup_levels_retention, cleanup_repairs_eq_class_waterlines) {
    gl::ExpressionAnalyzer analyzer(std::string("Peano"));
    gl::Memory memory;
    const gl::NameId mainId = memory.nameMap.encode("main");

    gl::EquivalenceClass cls;
    cls.setMembersFromNames({ "int_lev_1_1", "int_lev_1_2" }, memory.nameMap);
    const std::vector<char> blob = gl::serializeEquivalenceClass(cls);
    const int32_t lens[1] = { static_cast<int32_t>(blob.size()) };
    memory.equivalenceClassesMap.inner().assignRun(mainId, blob.data(), lens, 1);

    // Rows: kept (0), dropped (1), kept (2), dropped (3), kept (4).
    const char* texts[5] = { "(=[a,int_lev_1_1])", "(=[a,int_lev_1_2])",
                             "(=[b,int_lev_1_1])", "(=[b,int_lev_1_2])",
                             "(=[c,int_lev_1_1])" };
    for (int i = 0; i < 5; ++i) {
        const gl::EncodedExpression src(texts[i], memory.nameMap.decode(mainId));
        const gl::IntEncodedExpr row = gl::encodeExpression(src, memory.nameMap);
        memory.intEncodedStatements.push_back(row);
        memory.intStatementLevelsMap.insertSorted(
            gl::packStatementKey(row.originalId, row.validityId), 1);
    }

    // Five waterline keys (ad-hoc member sets; the index map is independent
    // of the class store) at 0, 1, 2, 4, 5.
    const std::vector<gl::NameId> keys[5] = {
        { memory.nameMap.encode("k0") }, { memory.nameMap.encode("k1") },
        { memory.nameMap.encode("k2") }, { memory.nameMap.encode("k4") },
        { memory.nameMap.encode("k5") } };
    const int before[5] = { 0, 1, 2, 4, 5 };
    for (int i = 0; i < 5; ++i)
        gl::upsertEqClassIndex(memory.eqClassSttmntIndexMapMap, mainId, keys[i], before[i]);

    analyzer.cleanUpExpressions(memory, gl::StrSpan("main", 4));

    ASSERT_EQ(static_cast<int>(memory.intEncodedStatements.size()), 3);
    // Dropped positions {1, 3}: below 0 -> 0, below 1 -> 0, below 2 -> 1,
    // below 4 -> 2, below 5 -> 2.
    const int after[5] = { 0, 1, 1, 2, 3 };
    for (int i = 0; i < 5; ++i)
        ASSERT_EQ(gl::lookupEqClassIndex(memory.eqClassSttmntIndexMapMap, mainId, keys[i]),
                  after[i]);
}

// The repair helper itself: an ascending erased run, counted below each
// waterline by binary search — identical for a C array and a PagedVector.
TEST(cleanup_levels_retention, repair_eq_class_waterlines_counts_below) {
    gl::Memory memory;
    const gl::NameId mainId = memory.nameMap.encode("main");
    const std::vector<gl::NameId> keys[5] = {
        { memory.nameMap.encode("w0") }, { memory.nameMap.encode("w1") },
        { memory.nameMap.encode("w4") }, { memory.nameMap.encode("w7") },
        { memory.nameMap.encode("w10") } };
    const int before[5] = { 0, 1, 4, 7, 10 };
    const int32_t erased[4] = { 1, 3, 4, 8 };
    // Strictly below: 0 -> none, 1 -> none, 4 -> {1,3}, 7 -> {1,3,4},
    // 10 -> {1,3,4,8}.
    const int after[5] = { 0, 1, 2, 4, 6 };

    for (int i = 0; i < 5; ++i)
        gl::upsertEqClassIndex(memory.eqClassSttmntIndexMapMap, mainId, keys[i], before[i]);
    gl::repairEqClassWaterlines(memory, erased, 4);
    for (int i = 0; i < 5; ++i)
        ASSERT_EQ(gl::lookupEqClassIndex(memory.eqClassSttmntIndexMapMap, mainId, keys[i]),
                  after[i]);

    // PagedVector run, same content, same result from the same start.
    for (int i = 0; i < 5; ++i)
        gl::upsertEqClassIndex(memory.eqClassSttmntIndexMapMap, mainId, keys[i], before[i]);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::PagedVector<int32_t> run(&gl::genScratchArenas().forSlot(0), &dirty);
    for (int k = 0; k < 4; ++k) run.push_back(erased[k]);
    gl::repairEqClassWaterlines(memory, run, 4);
    for (int i = 0; i < 5; ++i)
        ASSERT_EQ(gl::lookupEqClassIndex(memory.eqClassSttmntIndexMapMap, mainId, keys[i]),
                  after[i]);
}
