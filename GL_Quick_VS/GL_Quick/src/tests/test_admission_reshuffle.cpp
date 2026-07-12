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
/// @brief Unit tests for the admission-tail reshuffle — deferring the algebra
///        `admissionMap` writes out of the hashburst into a post-loop drain.
///
/// @details
/// Coverage for `gl::ExpressionAnalyzer::drainAdmissionKeysAlgebra`:
///   - symbol existence + signature (`void(gl::Memory&)`);
///   - behavioral: a freshly-staged `AdmissionKeyAlgebraRecord` replays all
///     four writes (admissionMap insert, admissionStatusMap = false,
///     varsInAdmissionMapKeys population skipping "marker", revisitRejected2),
///     and a record whose key is already consumed is skipped by the drain's
///     re-applied consumed-key gate.
/// `revisitRejected2` is a no-op for keys absent from `rejectedMap`, so the
/// behavioral test needs no disk fixtures beyond the `ExpressionAnalyzer("Peano")`
/// construction shared with the other member tests.

#include "test_harness.hpp"

#include "../prover.hpp"

TEST(admission_reshuffle, drain_admission_keys_algebra_symbol_signature) {
    // Symbol-existence + signature check: take the drain's address with its
    // expected signature `void(gl::Memory&)`. Compiles only if
    // drainAdmissionKeysAlgebra is declared on gl::ExpressionAnalyzer with
    // that exact signature, satisfying Rule 18's symbol-coverage requirement.
    using DrainFn = void (gl::ExpressionAnalyzer::*)(gl::Memory&);
    DrainFn fn = &gl::ExpressionAnalyzer::drainAdmissionKeysAlgebra;
    ASSERT_TRUE(fn != nullptr);
}

TEST(admission_reshuffle, drain_replays_writes_and_honors_consumed_gate) {
    // Heavy ExpressionAnalyzer construction is unavoidable for member access
    // (mirrors the test_memory.cpp ea("Peano") behavioral tests).
    gl::ExpressionAnalyzer ea("Peano");

    // --- positive: a freshly-staged record (key not consumed, not in
    // rejectedMap) replays all four writes. revisitRejected2 is a no-op for a
    // key absent from rejectedMap, so no further analyzer/disk state is needed.
    gl::Memory m;
    gl::SealedPageSet pages;
    pages.bind(&gl::staticMemory());
    const auto s = [&pages](const std::string& v) {
        return gl::SealedString::copyFrom(pages, v.data(),
                                          static_cast<int32_t>(v.size()));
    };
    gl::ExpressionWithValidity key("(in2[10,marker,3])", "main");
    gl::StagedAdmissionValue value;
    const gl::SealedString valueKeyArr[2] = { s("10"), s("3") };
    value.key = gl::SealedSpan<gl::SealedString>::copyFrom(pages, valueKeyArr, 2);

    m.admissionKeysAlgebra.push_back(gl::AdmissionKeyAlgebraRecord{
        gl::SealedExpressionWithValidity{ s(key.original),
                                          s(key.validityName) },
        value });
    ea.drainAdmissionKeysAlgebra(m);

    // admissionMap got key -> { value } — the drain minted the packed key.
    int32_t keyPk = 0;
    ASSERT_TRUE(gl::lookupTemplateKey(m.templateInterner, m.nameMap,
                                      key.original, key.validityName, keyPk));
    const int32_t admId = m.overallHashMemory.admissionMap.lookup(keyPk);
    ASSERT_TRUE(admId != 0);
    const gl::AdmissionValueSet admVals = gl::admissionRecordsAt(
        m.overallHashMemory.admissionMap, keyPk, m.valueInterner);
    ASSERT_TRUE(admVals.count(
        gl::stagedToIdValue(value, m.valueInterner)) == static_cast<std::size_t>(1));

    // admissionStatusMap[key] == false (cold find returns a pointer; nullptr on
    // a miss, so the probe inserts no spurious entry).
    const uint8_t* itStatus = m.overallHashMemory.admissionStatusMap.find(keyPk);
    ASSERT_TRUE(itStatus != nullptr);
    ASSERT_TRUE(*itStatus == 0);

    // varsInAdmissionMapKeys was populated from the key's args, never "marker".
    ASSERT_FALSE(m.overallHashMemory.varsInAdmissionMapKeys.empty());
    {
        const int16_t markerId = m.templateInterner.lookup("marker");
        ASSERT_TRUE(markerId == 0
                    || !m.overallHashMemory.varsInAdmissionMapKeys.contains(markerId));
    }

    // --- negative: a key already in consumedAdmissionKeys is skipped by the
    // drain's re-applied gate (the byte-identity-critical within-burst
    // consume->skip ordering).
    gl::Memory m2;
    gl::ExpressionWithValidity key2("(in2[7,marker,2])", "main");
    m2.overallHashMemory.consumedAdmissionKeys.mint(gl::mintTemplateKey(
        m2.templateInterner, m2.nameMap, key2.original, key2.validityName));
    gl::StagedAdmissionValue value2;
    const gl::SealedString value2KeyArr[2] = { s("7"), s("2") };
    value2.key = gl::SealedSpan<gl::SealedString>::copyFrom(pages, value2KeyArr, 2);
    m2.admissionKeysAlgebra.push_back(gl::AdmissionKeyAlgebraRecord{
        gl::SealedExpressionWithValidity{ s(key2.original),
                                          s(key2.validityName) },
        value2 });
    ea.drainAdmissionKeysAlgebra(m2);

    int32_t key2Pk = 0;
    ASSERT_TRUE(gl::lookupTemplateKey(m2.templateInterner, m2.nameMap,
                                      key2.original, key2.validityName, key2Pk));
    ASSERT_TRUE(m2.overallHashMemory.admissionMap.lookup(key2Pk) == 0);
    ASSERT_TRUE(m2.overallHashMemory.admissionStatusMap.find(key2Pk) == nullptr);
    // admissionRecordsAt on an absent key returns an empty set (cold miss path).
    ASSERT_TRUE(gl::admissionRecordsAt(m2.overallHashMemory.admissionMap,
                                       key2Pk, m2.valueInterner).empty());

    // Release the views before the pages die (the production clear point).
    m.admissionKeysAlgebra.clear();
    m2.admissionKeysAlgebra.clear();
    pages.seal();
    pages.freePages();
}

// --- resetResentExpressionRegistries (delete -> send -> reabsorb helper) ---
// Wipes a resent, not-yet-local compound from the per-LB dedup registries so the
// plain absorb re-disintegrates it, while preserving exprOriginMap (history).
// See D-106 (Regression / completion).

TEST(resent_reset, reset_resent_expression_registries_symbol_signature) {
    using ResetFn = void (gl::ExpressionAnalyzer::*)(gl::Memory&,
                                                     gl::StrSpan, gl::StrSpan);
    ResetFn fn = &gl::ExpressionAnalyzer::resetResentExpressionRegistries;
    ASSERT_TRUE(fn != nullptr);
}

TEST(resent_reset, erases_dedup_registries_keeps_history_and_other_exprs) {
    gl::ExpressionAnalyzer ea("Peano");
    gl::Memory m;

    const std::string E = "(existence1[1,9,7,5])";
    gl::EncodedExpression enc(E, "main");
    gl::ExpressionWithValidity ewv(E, "main");

    // A second, unrelated expression that must survive the reset.
    const std::string OTHER = "(in2[2,6,3])";

    // Register E in the dedup registries the helper wipes; add a history entry
    // and the unrelated expression.
    gl::upsertStatementKey(m.intKnownStatements,
        gl::packStatementKey(m.nameMap.encode(E), gl::NameMap::MAIN_ID),
        /*local=*/true, /*registered=*/true, /*known=*/true);
    gl::upsertStatementKey(m.intKnownStatements,
        gl::packStatementKey(m.nameMap.encode(OTHER), gl::NameMap::MAIN_ID),
        /*local=*/true, /*registered=*/true, /*known=*/true);
    m.intStatementLevelsMap.insertSorted(gl::packStatementKey(
        m.nameMap.encode(E), gl::NameMap::MAIN_ID), 4);
    m.intEncodedStatements.push_back(gl::encodeExpression(enc, m.nameMap));
    const int64_t pkHist = gl::mintOriginKey(m.originInterner,
        ewv.original, ewv.validityName);
    // history line the helper must preserve (cold blob map — a real line, not an
    // empty operator[] entry, which the cold map does not create)
    m.exprOriginMap.assignRun(pkHist, std::vector<gl::IdOrigin>{
        gl::IdOrigin(gl::OriginTag::broadcast, {}) });

    ea.resetResentExpressionRegistries(m, gl::StrSpan(ewv.original),
                                       gl::StrSpan(ewv.validityName));

    // Positive: E erased from every dedup registry touched — the whole
    // packed-key row goes, both membership bits with it.
    ASSERT_TRUE(gl::lookupStatementFlags(m.intKnownStatements, m.nameMap,
                                         E, "main") == nullptr);
    ASSERT_EQ(m.intStatementLevelsMap.lookup(gl::packStatementKey(
        m.nameMap.encode(E), gl::NameMap::MAIN_ID)), 0);
    ASSERT_TRUE(m.intEncodedStatements.empty());

    // History preserved (Rule 16 / I-44): exprOriginMap untouched.
    ASSERT_TRUE(m.exprOriginMap.lookup(pkHist) != 0);

    // Negative: the unrelated expression is left intact.
    {
        const gl::StatementFlags* otherRow = gl::lookupStatementFlags(
            m.intKnownStatements, m.nameMap, OTHER, "main");
        ASSERT_TRUE(otherRow != nullptr);
        ASSERT_TRUE(otherRow->registered);
    }
}

// --- drainDeferredAncestorAdmissions (I-28 detect-and-defer trial) ---
// updateAdmissionMap3, in a parallel phase-1/phase-3 worker, climbs parentMemory
// to a strict ancestor and would seed that ancestor's admission map inline (a
// cross-LB write). The trial instead seals the resolved updateAdmissionMap onto
// the shared `deferredAncestorPages` SealedPageSet record chain and replays it
// single-threaded here, post-join (index-sorted, then freePages + reset).

TEST(admission_reshuffle, deferred_ancestor_admission_drain_symbol_signature) {
    // Symbol-existence + signature check: take the drain's address with its
    // expected signature `void()`. Compiles only if
    // drainDeferredAncestorAdmissions is declared on gl::ExpressionAnalyzer with
    // that exact signature, satisfying Rule 18's symbol-coverage requirement.
    using DrainFn = void (gl::ExpressionAnalyzer::*)();
    DrainFn fn = &gl::ExpressionAnalyzer::drainDeferredAncestorAdmissions;
    ASSERT_TRUE(fn != nullptr);
}

TEST(admission_reshuffle, deferred_ancestor_admission_drain_replays_and_skips_discharged) {
    // Heavy ExpressionAnalyzer construction is unavoidable for member access and
    // for the prepareIntegration callee tree (mirrors the test_memory.cpp and
    // admission_integration_reshuffle ea("Peano") behavioral tests).
    gl::ExpressionAnalyzer ea("Peano");

    gl::Memory live;
    gl::Memory discharged;
    discharged.dischargedForever = true;

    // Stage the two seeds on the shared per-iteration SealedPageSet exactly as a
    // parallel worker's enqueue does (proveKernel emplaces it before phase-1).
    ea.deferredAncestorPages.emplace();
    ea.deferredAncestorPages->bind(&gl::staticMemory());
    {
        gl::SealedPageSet& ps = *ea.deferredAncestorPages;
        const auto seal = [&ps](const std::string& v) {
            return gl::SealedString::copyFrom(ps, v.data(),
                                              static_cast<int32_t>(v.size()));
        };
        // Positive: a LIVE ancestor record replays updateAdmissionMap. The key
        // has exactly one non-remaining arg ('b'), so updateAdmissionMap reaches
        // its `withoutRemaining.size() == 1` branch and mints (the changeable
        // arg is renamed but the count is preserved, so the branch fires).
        const gl::SealedString liveKey[1] = { seal("(in3[a,b,c,4])") };
        const gl::SealedString liveRem[3] = { seal("4"), seal("a"), seal("c") };  // sorted-unique
        ps.appendRecord(gl::ExpressionAnalyzer::DeferredAncestorAdmission{
            &live,
            gl::SealedSpan<gl::SealedString>::copyFrom(ps, liveKey, 1),
            gl::SealedSpan<gl::SealedString>::copyFrom(ps, liveRem, 3),
            /*maxAdmissionDepth=*/3, /*maxSecondaryNumber=*/3,
            /*partOfRecursion=*/true });
        // Negative: a dischargedForever ancestor must be skipped — a permanently
        // deactivated LB has no next step and must not be resurrected (I-102/I-112).
        const gl::SealedString dKey[1] = { seal("(in3[e,f,g,4])") };
        const gl::SealedString dRem[3] = { seal("4"), seal("e"), seal("g") };
        ps.appendRecord(gl::ExpressionAnalyzer::DeferredAncestorAdmission{
            &discharged,
            gl::SealedSpan<gl::SealedString>::copyFrom(ps, dKey, 1),
            gl::SealedSpan<gl::SealedString>::copyFrom(ps, dRem, 3),
            3, 3, true });
    }

    // The drain's seam door routes each ancestor reload through the steward
    // handshake (claim-correct seam loads); production always has a live
    // steward at the drain (proveKernel's prove scope), so the direct-call
    // test provides one too. Not started: the handshake works standalone.
    ea.steward = std::make_unique<gl::MemorySteward>();
    ea.drainDeferredAncestorAdmissions();
    ea.steward.reset();

    // The drain replayed both records, then sealed + freed + reset the page set.
    ASSERT_FALSE(ea.deferredAncestorPages.has_value());
    // Live ancestor received the admission seed — varsInAdmissionMapKeys was
    // populated from the minted key's args (the same write the reshuffle test
    // checks), proving updateAdmissionMap ran against `live`.
    ASSERT_FALSE(live.overallHashMemory.varsInAdmissionMapKeys.empty());
    // Discharged ancestor was skipped before any write — untouched.
    ASSERT_TRUE(discharged.overallHashMemory.varsInAdmissionMapKeys.empty());
}
