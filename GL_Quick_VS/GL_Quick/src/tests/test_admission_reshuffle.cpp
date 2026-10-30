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
/// Coverage for `gl::AdmissionMemory::drainAdmissionKeysAlgebra`:
///   - symbol existence + signature (`void(gl::Memory&)`);
///   - behavioral: a freshly-staged `AdmissionKeyAlgebraRecord` replays all
///     four writes (admissionMap insert, admissionStatusMap = false,
///     varsInAdmissionMapKeys population skipping "marker", revisitRejected2).
/// `revisitRejected2` is a no-op for keys absent from `rejectedMap`, so the
/// behavioral test needs no disk fixtures beyond the `gl::tests::ProverFixture("Peano")`
/// construction shared with the other member tests.

#include "../memory.hpp"
#include <cstddef>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "test_harness.hpp"
#include "prover_fixture.hpp"

#include "../prover.hpp"

TEST(admission_reshuffle, drain_admission_keys_algebra_symbol_signature) {
    // Symbol-existence + signature check: take the drain's address with its
    // expected signature `void(gl::Memory&)`. Compiles only if
    // drainAdmissionKeysAlgebra is declared on gl::tests::ProverFixture with
    // that exact signature, satisfying Rule 18's symbol-coverage requirement.
    using DrainFn = void (gl::AdmissionMemory::*)(gl::Memory&);
    DrainFn fn = &gl::AdmissionMemory::drainAdmissionKeysAlgebra;
    ASSERT_TRUE(fn != nullptr);
}

TEST(admission_reshuffle, drain_replays_writes_and_honors_consumed_gate) {
    // Heavy gl::tests::ProverFixture construction is unavoidable for member access
    // (mirrors the test_memory.cpp ea("Peano") behavioral tests).
    gl::tests::ProverFixture ea("Peano");

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

    ((m).admissionKeysAlgebra).push_back(gl::AdmissionKeyAlgebraRecord{
        gl::SealedExpressionWithValidity{ s(key.original),
                                          s(key.validityName) },
        value });
    ea.admissionMemory.drainAdmissionKeysAlgebra(m);

    // admissionMap got key -> { value } — the drain minted the packed key.
    int64_t keyPk = 0;
    ASSERT_TRUE(gl::lookupTemplateKey(((m).templateInterner), ((m).nameMap),
                                      key.original, key.validityName, keyPk));
    const int32_t admId = ((((m).overallHashMemory)).admissionMap).lookup(keyPk);
    ASSERT_TRUE(admId != 0);
    const gl::AdmissionValueSet admVals = gl::admissionRecordsAt(
        ((((m).overallHashMemory)).admissionMap), keyPk, ((m).valueInterner));
    ASSERT_TRUE(admVals.count(
        gl::stagedToIdValue(value, ((m).valueInterner))) == static_cast<std::size_t>(1));

    // admissionStatusMap[key] == false (cold find returns a pointer; nullptr on
    // a miss, so the probe inserts no spurious entry).
    const uint8_t* itStatus = ((((m).overallHashMemory)).admissionStatusMap).find(keyPk);
    ASSERT_TRUE(itStatus != nullptr);
    ASSERT_TRUE(*itStatus == 0);

    // varsInAdmissionMapKeys was populated from the key's args, never "marker".
    ASSERT_FALSE(((((m).overallHashMemory)).varsInAdmissionMapKeys).empty());
    {
        const gl::NameId markerId = ((m).templateInterner).lookup("marker");
        ASSERT_TRUE(markerId == 0
                    || !((((m).overallHashMemory)).varsInAdmissionMapKeys).contains(markerId));
    }

    // Release the views before the pages die (the production clear point).
    ((m).admissionKeysAlgebra).clear();
    pages.seal();
    pages.freePages();
}

// --- resetResentExpressionRegistries (delete -> send -> reabsorb helper) ---
// Wipes a resent, not-yet-local compound from the per-LB dedup registries so the
// plain absorb re-disintegrates it, while preserving exprOriginMap (history).
// See D-106 (Regression / completion).

TEST(resent_reset, reset_resent_expression_registries_symbol_signature) {
    using ResetFn = void (gl::EquivalenceAdmissionRewriter::*)(gl::Memory&,
                                                     gl::StrSpan, gl::StrSpan);
    ResetFn fn = &gl::EquivalenceAdmissionRewriter::resetResentExpressionRegistries;
    ASSERT_TRUE(fn != nullptr);
}

TEST(resent_reset, erases_dedup_registries_keeps_history_and_other_exprs) {
    gl::tests::ProverFixture ea("Peano");
    gl::Memory m;

    const std::string E = "(existence1[1,9,7,5])";
    gl::EncodedExpression enc(E, "main");
    gl::ExpressionWithValidity ewv(E, "main");

    // A second, unrelated expression that must survive the reset.
    const std::string OTHER = "(in2[2,6,3])";

    // Register E in the dedup registries the helper wipes; add a history entry
    // and the unrelated expression.
    gl::upsertStatementKey(((m).intKnownStatements),
        gl::packStatementKey(((m).nameMap).encode(E), gl::NameMap::MAIN_ID),
        /*local=*/true);
    gl::upsertStatementKey(((m).intKnownStatements),
        gl::packStatementKey(((m).nameMap).encode(OTHER), gl::NameMap::MAIN_ID),
        /*local=*/true);
    ((m).intStatementLevelsMap).insertSorted(gl::packStatementKey(
        ((m).nameMap).encode(E), gl::NameMap::MAIN_ID), 4);
    ((m).intEncodedStatements).push_back(gl::encodeExpression(enc, ((m).nameMap)));
    const int64_t pkHist = gl::mintOriginKey(((m).originInterner),
        ewv.original, ewv.validityName);
    // history line the helper must preserve (cold blob map — a real line, not an
    // empty operator[] entry, which the cold map does not create)
    ((m).exprOriginMap).assignRun(pkHist, std::vector<gl::IdOrigin>{
        gl::IdOrigin(gl::OriginTag::broadcast, {}) });

    ea.equivalenceAdmissionRewriter.resetResentExpressionRegistries(m, gl::StrSpan(ewv.original),
                                       gl::StrSpan(ewv.validityName));

    // Positive: E erased from every dedup registry touched — the whole
    // packed-key row goes, both membership bits with it.
    ASSERT_TRUE(gl::lookupStatementFlags(((m).intKnownStatements), ((m).nameMap),
                                         E, "main") == nullptr);
    ASSERT_EQ(((m).intStatementLevelsMap).lookup(gl::packStatementKey(
        ((m).nameMap).encode(E), gl::NameMap::MAIN_ID)), 0);
    ASSERT_TRUE(((m).intEncodedStatements).empty());

    // History preserved (Rule 16 / I-44): exprOriginMap untouched.
    ASSERT_TRUE(((m).exprOriginMap).lookup(pkHist) != 0);

    // Negative: the unrelated expression is left intact.
    {
        const gl::StatementFlags* otherRow = gl::lookupStatementFlags(
            ((m).intKnownStatements), ((m).nameMap), OTHER, "main");
        ASSERT_TRUE(otherRow != nullptr);
        ASSERT_TRUE(otherRow != nullptr);
    }
}

// The registry part of the reset erases the known and levels rows and returns
// the packed key, leaving the statement-list and staging rows; the sweep then
// removes the rows of every given key from both lists, survivors in order, and
// an empty key run changes nothing.
TEST(resent_reset, registry_keys_leave_the_lists_and_the_sweep_removes_their_rows) {
    gl::tests::ProverFixture ea("Peano");
    gl::Memory m;
    const std::string E = "(existence1[1,9,7,5])";
    const std::string F = "(in2[2,7,3])";
    const std::string OTHER = "(in2[2,6,3])";
    const gl::IntEncodedExpr rowE = gl::encodeExpression(gl::EncodedExpression(E, "main"), m.nameMap);
    const gl::IntEncodedExpr rowF = gl::encodeExpression(gl::EncodedExpression(F, "main"), m.nameMap);
    const gl::IntEncodedExpr rowOther = gl::encodeExpression(gl::EncodedExpression(OTHER, "main"), m.nameMap);
    const int64_t keyE = gl::packStatementKey(m.nameMap.encode(E), gl::NameMap::MAIN_ID);
    const int64_t keyF = gl::packStatementKey(m.nameMap.encode(F), gl::NameMap::MAIN_ID);
    gl::upsertStatementKey(m.intKnownStatements, keyE, /*local=*/false);
    gl::upsertStatementKey(m.intKnownStatements, keyF, /*local=*/false);
    m.intStatementLevelsMap.insertSorted(keyE, 4);
    m.intStatementLevelsMap.insertSorted(gl::packStatementKey(m.nameMap.encode(OTHER), gl::NameMap::MAIN_ID), 2);
    m.intEncodedStatements.push_back(rowE);
    m.intEncodedStatements.push_back(rowOther);
    m.intEncodedStatements.push_back(rowF);
    m.intEncodedStatements.push_back(rowE);
    m.intExternalStatements.push_back(rowF);
    m.intExternalStatements.push_back(rowOther);
    m.intExternalStatements.push_back(rowE);

    auto& rewriter = ea.equivalenceAdmissionRewriter;
    ASSERT_EQ(rewriter.resetResentRegistryKeys(m, gl::StrSpan(E), gl::StrSpan("main", 4)), keyE);
    ASSERT_TRUE(gl::lookupStatementFlags(m.intKnownStatements, m.nameMap, E, "main") == nullptr);
    ASSERT_EQ(m.intStatementLevelsMap.lookup(keyE), 0);
    ASSERT_EQ(m.intEncodedStatements.size(), 4);
    ASSERT_EQ(m.intExternalStatements.size(), 3);
    // F has no levels row: the probe leaves the levels map as it is.
    const int32_t levelRows = m.intStatementLevelsMap.count();
    ASSERT_EQ(rewriter.resetResentRegistryKeys(m, gl::StrSpan(F), gl::StrSpan("main", 4)), keyF);
    ASSERT_EQ(m.intStatementLevelsMap.count(), levelRows);
    ASSERT_TRUE(gl::lookupStatementFlags(m.intKnownStatements, m.nameMap, F, "main") == nullptr);

    rewriter.eraseResentRows(m, nullptr, 0);
    ASSERT_EQ(m.intEncodedStatements.size(), 4);
    const int64_t keys[3] = { keyF, keyE, keyF };
    rewriter.eraseResentRows(m, keys, 3);
    ASSERT_EQ(m.intEncodedStatements.size(), 1);
    ASSERT_EQ(m.intEncodedStatements[0].originalId, rowOther.originalId);
    ASSERT_EQ(m.intExternalStatements.size(), 1);
    ASSERT_EQ(m.intExternalStatements[0].originalId, rowOther.originalId);
    ASSERT_TRUE(m.intStatementLevelsMap.lookup(
        gl::packStatementKey(m.nameMap.encode(OTHER), gl::NameMap::MAIN_ID)) != 0);
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
    // drainDeferredAncestorAdmissions is declared on gl::tests::ProverFixture with
    // that exact signature, satisfying Rule 18's symbol-coverage requirement.
    using DrainFn = void (gl::AdmissionMemory::*)();
    DrainFn fn = &gl::AdmissionMemory::drainDeferredAncestorAdmissions;
    ASSERT_TRUE(fn != nullptr);
}

TEST(admission_reshuffle, deferred_ancestor_admission_drain_replays_and_skips_discharged) {
    // Heavy gl::tests::ProverFixture construction is unavoidable for member access and
    // for the prepareIntegration callee tree (mirrors the test_memory.cpp and
    // admission_integration_reshuffle ea("Peano") behavioral tests).
    gl::tests::ProverFixture ea("Peano");

    gl::Memory live;
    gl::Memory discharged;
    ((discharged).dischargedForever) = true;

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
        ps.appendRecord(gl::DeferredAncestorAdmission{
            &live,
            gl::SealedSpan<gl::SealedString>::copyFrom(ps, liveKey, 1),
            gl::SealedSpan<gl::SealedString>::copyFrom(ps, liveRem, 3),
            /*maxAdmissionDepth=*/3, /*maxSecondaryNumber=*/3,
            /*partOfRecursion=*/true });
        // Negative: a dischargedForever ancestor must be skipped — a permanently
        // deactivated LB has no next step and must not be resurrected (I-102/I-112).
        const gl::SealedString dKey[1] = { seal("(in3[e,f,g,4])") };
        const gl::SealedString dRem[3] = { seal("4"), seal("e"), seal("g") };
        ps.appendRecord(gl::DeferredAncestorAdmission{
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
    ea.admissionMemory.drainDeferredAncestorAdmissions();
    ea.steward.reset();

    // The drain replayed both records, then sealed + freed + reset the page set.
    ASSERT_FALSE(ea.deferredAncestorPages.has_value());
    // Live ancestor received the admission seed — varsInAdmissionMapKeys was
    // populated from the minted key's args (the same write the reshuffle test
    // checks), proving updateAdmissionMap ran against `live`.
    ASSERT_FALSE(((((live).overallHashMemory)).varsInAdmissionMapKeys).empty());
    // Discharged ancestor was skipped before any write — untouched.
    ASSERT_TRUE(((((discharged).overallHashMemory)).varsInAdmissionMapKeys).empty());
}

// --- updateAdmissionMap sees premises only, never the head ---
// A premise-completion demand asks for a missing premise instance; the head is
// the rule's product and is never demanded. The ¬∃t level of a negated
// remainder existence has a head-only variable (the `×` anchor slot occurs in
// no premise), which used to file the meaningless demand
// `!(in3[9,10,9,marker])` ("some operator in the × slot").

TEST(admission_reshuffle, install_demands_premises_only_never_the_head) {
    gl::tests::ProverFixture ea(std::string("FTA"));
    gl::HashMemory& hm = ((ea.body).overallHashMemory);
    const int lv[1] = { 0 };
    // premises: t + 9 = 9 (bound t), 9 < 10; head: !(9 × 10 = 9).
    const std::string p0 = "(in3[1,u_9,u_9,u_4])";
    const std::string p1 = "(strictOrder[u_1,u_4,u_9,u_10])";
    const std::string head = "!(in3[u_9,u_10,u_9,u_5])";
    const std::string impl =
        "(>[1](in3[1,u_9,u_9,u_4])(>[](strictOrder[u_1,u_4,u_9,u_10])"
        "!(in3[u_9,u_10,u_9,u_5])))";
    const gl::StrSpan chain[2] = { gl::StrSpan(p0), gl::StrSpan(p1) };
    // Remaining args = the premises' u_-stripped variables, sorted-unique in
    // compareSpans order (the install caller's remRun contract).
    const gl::StrSpan rem[4] = { gl::StrSpan("1", 1), gl::StrSpan("10", 2),
                                 gl::StrSpan("4", 1), gl::StrSpan("9", 1) };
    ea.hashRuleInstaller.addToHashMemory(chain, 2, gl::StrSpan(head), rem, 4,
        ea.body, hm, lv, 1, gl::StrSpan(impl),
        ea.parameters.maxIterationNumberVariable,
        ea.parameters.standardMaxSecondaryNumber, false,
        ea.parameters.minNumOperatorsKey, gl::StrSpan("implication", 11),
        /*performAdmissionMapUpdate=*/true, gl::StrSpan(impl));

    const auto& am = ((hm).admissionMap);
    // The bound premise filed its demand (the premise with exactly one
    // non-remaining argument), so the map is not empty ...
    ASSERT_TRUE(am.count() > 0);
    // ... and no key is the head or any negated template.
    int64_t headPk = 0;
    const bool headInterned = gl::lookupTemplateKey(
        ((ea.body).templateInterner), ((ea.body).nameMap),
        gl::StrSpan("!(in3[9,10,9,marker])"), gl::StrSpan("main", 4), headPk);
    ASSERT_TRUE(!headInterned || am.lookup(headPk) == 0);
    for (int32_t id = 1; id <= am.count(); ++id) {
        const std::pair<std::string, std::string> kv = gl::decodeTemplateKey(
            am.keyAt(id), ((ea.body).templateInterner), ((ea.body).nameMap));
        ASSERT_TRUE(!kv.first.empty() && kv.first[0] == '(');
    }
}

// --- Induction demand for one free argument (D-413) ---
// On a recursion-owned chain every positive compound premise with exactly one
// unbound argument files a marker demand and prepares its integration, as
// operator premises always did; two-input relations (preorder, strictOrder),
// ordinary chains and negated premises keep the D-257 skip. Induction variables
// stay `(1)`-typed, and the post-burst drain keeps a template's recursion
// status.

namespace {
    // Packed key of `templ` at `main` in `mb`, or 0 when never interned.
    int64_t propertyDemandKey(gl::Memory& mb, const std::string& templ) {
        int64_t pk = 0;
        if (!gl::lookupTemplateKey(((mb).templateInterner), ((mb).nameMap),
                gl::StrSpan(templ), gl::StrSpan("main", 4), pk))
            return 0;
        return pk;
    }

    // True iff `templ` at `main` carries an algebra admission entry in `mb`.
    bool propertyDemandFiled(gl::Memory& mb, const std::string& templ) {
        const int64_t pk = propertyDemandKey(mb, templ);
        return pk != 0 && ((((mb).overallHashMemory)).admissionMap).lookup(pk) != 0;
    }

    // One updateAdmissionMap call on the FTA analyzer's root LB.
    void runPropertyChain(gl::tests::ProverFixture& ana, const std::string& element,
                          const std::vector<std::string>& remaining, bool partOfRecursion) {
        const gl::StrSpan key[] = { gl::StrSpan(element) };
        std::vector<gl::StrSpan> rem;
        for (const std::string& r : remaining) rem.push_back(gl::StrSpan(r));
        ana.admissionMemory.updateAdmissionMap(ana.body, key, 1, rem.data(),
            static_cast<int32_t>(rem.size()), 4, 4, partOfRecursion,
            gl::StrSpan("main", 4));
    }
}

TEST(property_input_demand, induction_chain_demands_the_unbound_sequence_input) {
    gl::tests::ProverFixture ana(std::string("FTA"));
    ana.parameters.trackHistory = false;
    runPropertyChain(ana, "(sequence[1,4,2,9,11])", { "1", "2", "4", "9" }, true);

    const std::string demand = "(sequence[1,4,2,9,marker])";
    ASSERT_TRUE(propertyDemandFiled(ana.body, demand));
    const uint8_t* status = ((((ana.body).overallHashMemory)).admissionStatusMap).find(
        propertyDemandKey(ana.body, demand));
    ASSERT_TRUE(status != nullptr);
    ASSERT_EQ(static_cast<int>(*status), 1);
    // The integration of the marker form was prepared.
    const gl::NameId tid = ((ana.body).templateInterner).lookup(gl::StrSpan(demand));
    ASSERT_TRUE(tid != 0);
    ASSERT_TRUE(((ana.body).integrationStartIntMap).lookup(tid) != 0);
}

TEST(property_input_demand, ordinary_chain_keeps_the_d257_skip) {
    gl::tests::ProverFixture ana(std::string("FTA"));
    ana.parameters.trackHistory = false;
    runPropertyChain(ana, "(sequence[1,4,2,9,11])", { "1", "2", "4", "9" }, false);
    ASSERT_FALSE(propertyDemandFiled(ana.body, "(sequence[1,4,2,9,marker])"));
}

TEST(property_input_demand, any_single_free_argument_is_demanded) {
    gl::tests::ProverFixture ana(std::string("FTA"));
    ana.parameters.trackHistory = false;
    // The endpoint 9 is the one free argument: it is demanded and prepared.
    runPropertyChain(ana, "(sequence[1,4,2,9,11])", { "1", "11", "2", "4" }, true);
    const std::string demand = "(sequence[1,4,2,marker,11])";
    ASSERT_TRUE(propertyDemandFiled(ana.body, demand));
    const gl::NameId tid = ((ana.body).templateInterner).lookup(gl::StrSpan(demand));
    ASSERT_TRUE(tid != 0);
    ASSERT_TRUE(((ana.body).integrationStartIntMap).lookup(tid) != 0);
    // Two free arguments: no demand.
    runPropertyChain(ana, "(interval[1,4,2,12,13])", { "1", "2", "4" }, true);
    ASSERT_FALSE(propertyDemandFiled(ana.body, "(interval[1,4,2,marker,13])"));
    ASSERT_FALSE(propertyDemandFiled(ana.body, "(interval[1,4,2,12,marker])"));
}

TEST(property_input_demand, relation_keeps_the_d257_skip) {
    gl::tests::ProverFixture ana(std::string("FTA"));
    ana.parameters.trackHistory = false;
    // preorder has two inputs and no output: a relation, never demanded.
    runPropertyChain(ana, "(preorder[1,4,9,11])", { "1", "4", "9" }, true);
    ASSERT_FALSE(propertyDemandFiled(ana.body, "(preorder[1,4,9,marker])"));
}

TEST(property_input_demand, negated_property_keeps_the_d257_skip) {
    gl::tests::ProverFixture ana(std::string("FTA"));
    ana.parameters.trackHistory = false;
    runPropertyChain(ana, "!(sequence[1,4,2,9,11])", { "1", "2", "4", "9" }, true);
    ASSERT_FALSE(propertyDemandFiled(ana.body, "!(sequence[1,4,2,9,marker])"));
    ASSERT_FALSE(propertyDemandFiled(ana.body, "(sequence[1,4,2,9,marker])"));
}

TEST(property_input_demand, f3_induction_variable_is_the_lower_endpoint) {
    gl::tests::ProverFixture ea(std::string("FTA"));
    gl::StrSpan out[gl::ExecutionParameters::MAX_ADMISSION_REM_ARGS];
    // Reverse and forward F3: the sequence endpoint 10 is an input but also
    // the successor output, so the only induction variable is 9.
    const std::string reverseF3 =
        "(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](in2[9,10,3])(>[11](sequence[1,4,2,10,11])(>[12](in2[10,12,11])(>[13](fold[1,3,4,5,11,2,9,13])(>[14](fold[1,3,4,5,11,2,10,14])(in3[13,12,14,5])))))))";
    const std::string forwardF3 =
        "(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](in2[9,10,3])(>[11](sequence[1,4,2,10,11])(>[12](in2[10,12,11])(>[13](fold[1,3,4,5,11,2,9,13])(>[14](in3[13,12,14,5])(fold[1,3,4,5,11,2,10,14])))))))";
    for (const std::string* theorem : { &reverseF3, &forwardF3 }) {
        const int32_t n = ea.scopeTypes.findDigitArgs(*theorem, ea.anchorInfo, out,
            gl::ExecutionParameters::MAX_ADMISSION_REM_ARGS);
        ASSERT_EQ(n, 1);
        ASSERT_EQ(out[0].toStdString(), std::string("9"));
    }
}

namespace {
    // Drain one staged `(in2[10,marker,3])` record with the given recursion
    // flag into `m`, optionally after seeding the template's status.
    void drainStatusRecord(gl::tests::ProverFixture& ea, gl::Memory& m,
                           bool recordFlag, int seededStatus) {
        gl::SealedPageSet pages;
        pages.bind(&gl::staticMemory());
        const auto s = [&pages](const std::string& v) {
            return gl::SealedString::copyFrom(pages, v.data(),
                                              static_cast<int32_t>(v.size()));
        };
        const std::string keyText = "(in2[10,marker,3])";
        if (seededStatus >= 0) {
            const int64_t pk = gl::mintTemplateKey(((m).templateInterner), ((m).nameMap),
                keyText, std::string("main"));
            ((((m).overallHashMemory)).admissionStatusMap).upsert(pk,
                static_cast<uint8_t>(seededStatus));
        }
        gl::StagedAdmissionValue value;
        const gl::SealedString keyArr[2] = { s("10"), s("3") };
        value.key = gl::SealedSpan<gl::SealedString>::copyFrom(pages, keyArr, 2);
        value.flag = recordFlag;
        ((m).admissionKeysAlgebra).push_back(gl::AdmissionKeyAlgebraRecord{
            gl::SealedExpressionWithValidity{ s(keyText), s("main") }, value });
        ea.admissionMemory.drainAdmissionKeysAlgebra(m);
        ((m).admissionKeysAlgebra).clear();
        pages.seal();
        pages.freePages();
    }

    // The drained template's status, -1 when absent.
    int drainedStatus(gl::Memory& m) {
        const int64_t pk = propertyDemandKey(m, "(in2[10,marker,3])");
        if (pk == 0) return -1;
        const uint8_t* status = ((((m).overallHashMemory)).admissionStatusMap).find(pk);
        return status == nullptr ? -1 : static_cast<int>(*status);
    }
}

TEST(admission_reshuffle, drain_keeps_an_existing_recursion_status) {
    gl::tests::ProverFixture ea("Peano");
    gl::Memory m;
    drainStatusRecord(ea, m, /*recordFlag=*/false, /*seededStatus=*/1);
    ASSERT_EQ(drainedStatus(m), 1);
}

TEST(admission_reshuffle, drain_sets_recursion_status_for_a_recursion_record) {
    gl::tests::ProverFixture ea("Peano");
    gl::Memory m;
    drainStatusRecord(ea, m, /*recordFlag=*/true, /*seededStatus=*/-1);
    ASSERT_EQ(drainedStatus(m), 1);
}

TEST(admission_reshuffle, drain_leaves_an_ordinary_template_ordinary) {
    gl::tests::ProverFixture ea("Peano");
    gl::Memory m;
    drainStatusRecord(ea, m, /*recordFlag=*/false, /*seededStatus=*/0);
    ASSERT_EQ(drainedStatus(m), 0);
}

// --- Recursion propagation: one free argument, integration demands ---
// (D-413) updateAdmissionMapRecursion binds
// an operator's output and any argument of a compound that is not a two-input
// relation. A matching statement re-keys a recursion-owned chain; in the
// re-keyed chain an operator with only its output free and a compound with
// exactly one free argument gain a derived demand, and a positive one also has
// its integration prepared.

namespace {
    // Seed a recursion-owned demand: template `templ` at main carrying one
    // flag-1 value (premise chain `chain`, sorted remaining args `remaining`)
    // and status 1. A block holding an induction chain has a level (a fresh
    // Memory carries the unset -1), which the integration a re-key prepares
    // names its witnesses with.
    void seedRecursionChain(gl::Memory& mb, const std::string& templ,
                            const std::vector<std::string>& chain,
                            const std::vector<std::string>& remaining) {
        mb.level = 1;
        const int64_t key = gl::mintTemplateKey(((mb).templateInterner), ((mb).nameMap),
            templ, std::string("main"));
        gl::AdmissionMapValue value;
        for (const std::string& element : chain)
            value.key.push_back(((mb).valueInterner).encode(element));
        for (const std::string& arg : remaining)
            value.remainingArgs.push_back(((mb).valueInterner).encode(arg));
        value.standardMaxAdmissionDepth = 1;
        value.standardMaxSecondaryNumber = 2;
        value.flag = true;
        auto& arena = gl::genScratchArenas().forSlot(gl::genScratchArenas().slotCount() - 1);
        gl::insertAdmissionValue(((((mb).overallHashMemory)).admissionMap), key, value,
                                 ((mb).valueInterner), arena);
        ((((mb).overallHashMemory)).admissionStatusMap).upsert(key, uint8_t(1));
    }

    // The status of `templ` at main, -1 when the template has none.
    int recursionDemandStatus(gl::Memory& mb, const std::string& templ) {
        const int64_t pk = propertyDemandKey(mb, templ);
        if (pk == 0) return -1;
        const uint8_t* status = ((((mb).overallHashMemory)).admissionStatusMap).find(pk);
        return status == nullptr ? -1 : static_cast<int>(*status);
    }

    // True iff the integration of `templ` was prepared (its start-int row).
    bool recursionIntegrationPrepared(gl::Memory& mb, const std::string& templ) {
        const gl::NameId tid = ((mb).templateInterner).lookup(gl::StrSpan(templ));
        return tid != 0 && ((mb).integrationStartIntMap).lookup(tid) != 0;
    }

    // One deposit-door call for `statement` at main; `inductionAllowed` is
    // the door's verdict on the disintegrated original.
    void arriveAtRecursion(gl::tests::ProverFixture& ana, const std::string& statement,
                           bool inductionAllowed = true) {
        ana.admissionMemory.updateAdmissionMapRecursion(gl::StrSpan(statement), ana.body,
                                        gl::StrSpan("main", 4), inductionAllowed);
    }

    // The reverse-F3 hypothesis chain re-keyed with the step endpoint 9,
    // filed under its sequence premise (the run-01 terminal state).
    void seedReverseF3Hypothesis(gl::Memory& mb) {
        seedRecursionChain(mb, "(sequence[1,4,2,9,marker])",
            { "(in2[rec,9,3])", "(sequence[1,4,2,9,11])", "(in2[9,12,11])",
              "(fold[1,3,4,5,11,2,rec,13])", "(fold[1,3,4,5,11,2,9,14])" },
            { "1", "2", "3", "4", "5", "9", "rec" });
    }
}

TEST(recursion_propagation, sequence_statement_rekeys_the_hypothesis_chain) {
    gl::tests::ProverFixture ana(std::string("FTA"));
    ana.parameters.trackHistory = false;
    seedReverseF3Hypothesis(ana.body);
    arriveAtRecursion(ana, "(sequence[1,4,2,9,20])");

    // The value at 9, the fold over [0,rec] and the fold over [0,9] of the
    // bound sequence 20: each an operator with only its output free.
    for (const std::string demand : { std::string("(in2[9,marker,20])"),
                                      std::string("(fold[1,3,4,5,20,2,rec,marker])"),
                                      std::string("(fold[1,3,4,5,20,2,9,marker])") }) {
        ASSERT_TRUE(propertyDemandFiled(ana.body, demand));
        ASSERT_EQ(recursionDemandStatus(ana.body, demand), 1);
        ASSERT_TRUE(recursionIntegrationPrepared(ana.body, demand));
    }
    // The bound premises demand nothing.
    ASSERT_FALSE(propertyDemandFiled(ana.body, "(in2[rec,marker,3])"));
}

// A product of a disintegration whose original carries a non-number argument
// does not move an induction chain (I-285):
// the same arrival that re-keys the chain above derives nothing, prepares no
// integration and leaves the hypothesis template as it was.
TEST(recursion_propagation, product_of_an_original_with_hidden_arguments_moves_no_chain) {
    gl::tests::ProverFixture ana(std::string("FTA"));
    ana.parameters.trackHistory = false;
    seedReverseF3Hypothesis(ana.body);
    const auto before = ((((ana.body).overallHashMemory)).admissionMap).count();
    arriveAtRecursion(ana, "(sequence[1,4,2,9,20])", false);

    for (const std::string demand : { std::string("(in2[9,marker,20])"),
                                      std::string("(fold[1,3,4,5,20,2,rec,marker])"),
                                      std::string("(fold[1,3,4,5,20,2,9,marker])") }) {
        ASSERT_FALSE(propertyDemandFiled(ana.body, demand));
        ASSERT_FALSE(recursionIntegrationPrepared(ana.body, demand));
    }
    ASSERT_EQ(((((ana.body).overallHashMemory)).admissionMap).count(), before);
    ASSERT_EQ(recursionDemandStatus(ana.body, "(sequence[1,4,2,9,marker])"), 1);
}

// A construction of the induction hypothesis' own head moves no induction
// chain (I-286): the arrival that re-keys
// the chain above derives nothing, prepares no integration, enrolls no product
// of recursion and leaves the hypothesis template as it was.
TEST(recursion_propagation, construction_of_the_induction_head_moves_no_chain) {
    gl::tests::ProverFixture ana(std::string("FTA"));
    ana.parameters.trackHistory = false;
    seedReverseF3Hypothesis(ana.body);
    ((ana.body).recursionHeadWitnesses).mint(gl::StrSpan("20", 2));
    const auto before = ((((ana.body).overallHashMemory)).admissionMap).count();
    const auto productsBefore = ((((ana.body).overallHashMemory)).productsOfRecursionIds).count();
    arriveAtRecursion(ana, "(sequence[1,4,2,9,20])");

    for (const std::string demand : { std::string("(in2[9,marker,20])"),
                                      std::string("(fold[1,3,4,5,20,2,rec,marker])"),
                                      std::string("(fold[1,3,4,5,20,2,9,marker])") }) {
        ASSERT_FALSE(propertyDemandFiled(ana.body, demand));
        ASSERT_FALSE(recursionIntegrationPrepared(ana.body, demand));
    }
    ASSERT_EQ(((((ana.body).overallHashMemory)).admissionMap).count(), before);
    ASSERT_EQ(((((ana.body).overallHashMemory)).productsOfRecursionIds).count(), productsBefore);
    ASSERT_EQ(recursionDemandStatus(ana.body, "(sequence[1,4,2,9,marker])"), 1);
}

// A compound element never binds a construction over constructed objects
// (I-292): the sequence arrival that re-keys the
// chain in `sequence_statement_rekeys_the_hypothesis_chain` derives nothing,
// prepares no integration, enrolls no product of recursion and leaves the
// hypothesis template as it was once its bound sequence is registered.
TEST(recursion_propagation, compound_binding_of_a_hidden_argument_witness_moves_no_chain) {
    gl::tests::ProverFixture ana(std::string("FTA"));
    ana.parameters.trackHistory = false;
    seedReverseF3Hypothesis(ana.body);
    ((ana.body).hiddenArgumentWitnesses).mint(gl::StrSpan("20", 2));
    const auto before = ((((ana.body).overallHashMemory)).admissionMap).count();
    const auto productsBefore = ((((ana.body).overallHashMemory)).productsOfRecursionIds).count();
    arriveAtRecursion(ana, "(sequence[1,4,2,9,20])");

    for (const std::string demand : { std::string("(in2[9,marker,20])"),
                                      std::string("(fold[1,3,4,5,20,2,rec,marker])"),
                                      std::string("(fold[1,3,4,5,20,2,9,marker])") }) {
        ASSERT_FALSE(propertyDemandFiled(ana.body, demand));
        ASSERT_FALSE(recursionIntegrationPrepared(ana.body, demand));
    }
    ASSERT_EQ(((((ana.body).overallHashMemory)).admissionMap).count(), before);
    ASSERT_EQ(((((ana.body).overallHashMemory)).productsOfRecursionIds).count(), productsBefore);
    ASSERT_EQ(recursionDemandStatus(ana.body, "(sequence[1,4,2,9,marker])"), 1);
}

// An operator binds only its output, and the hidden-argument registry never
// refuses that binding (I-292): the value arrival
// moves the chain exactly as it does without the registration.
TEST(recursion_propagation, operator_output_binding_of_a_hidden_argument_witness_moves_the_chain) {
    int32_t entries[2] = { 0, 0 };
    for (const bool registered : { false, true }) {
        gl::tests::ProverFixture ana(std::string("FTA"));
        // History off: the re-key prepares the fully bound sequence premise,
        // whose existence/and rule text is built without history too.
        ana.parameters.trackHistory = false;
        seedRecursionChain(ana.body, "(in2[9,marker,20])",
            { "(in2[rec,9,3])", "(sequence[1,4,2,9,20])", "(in2[9,12,20])",
              "(fold[1,3,4,5,20,2,rec,13])", "(fold[1,3,4,5,20,2,9,14])" },
            { "1", "2", "20", "3", "4", "5", "9", "rec" });
        if (registered) ((ana.body).hiddenArgumentWitnesses).mint(gl::StrSpan("21", 2));
        const auto before = ((((ana.body).overallHashMemory)).admissionMap).count();
        arriveAtRecursion(ana, "(in2[9,21,20])");
        entries[registered ? 1 : 0] = ((((ana.body).overallHashMemory)).admissionMap).count();
        ASSERT_TRUE(entries[registered ? 1 : 0] > before);
        ASSERT_TRUE(propertyDemandFiled(ana.body, "(fold[1,3,4,5,20,2,rec,marker])"));
    }
    ASSERT_EQ(entries[0], entries[1]);
}

TEST(recursion_propagation, only_the_operator_output_probe_leaves_an_entry) {
    gl::tests::ProverFixture ana(std::string("FTA"));
    ana.parameters.trackHistory = false;
    const int32_t before = ((((ana.body).overallHashMemory)).admissionStatusMap).count();
    arriveAtRecursion(ana, "(sequence[1,4,2,9,20])");
    ASSERT_EQ(((((ana.body).overallHashMemory)).admissionStatusMap).count(), before);
    ASSERT_EQ(propertyDemandKey(ana.body, "(sequence[1,4,2,9,marker])"), 0);
    // An unmatched operator statement keeps the historical default-false entry
    // of its output probe; its other probes leave nothing.
    arriveAtRecursion(ana, "(in2[9,20,3])");
    ASSERT_EQ(recursionDemandStatus(ana.body, "(in2[9,marker,3])"), 0);
    ASSERT_EQ(((((ana.body).overallHashMemory)).admissionStatusMap).count(), before + 1);
    ASSERT_EQ(propertyDemandKey(ana.body, "(in2[marker,20,3])"), 0);
}

TEST(recursion_propagation, negated_compound_or_atomic_statement_is_no_participant) {
    gl::tests::ProverFixture ana(std::string("FTA"));
    ana.parameters.trackHistory = false;
    seedRecursionChain(ana.body, "!(sequence[1,4,2,9,marker])",
        { "!(sequence[1,4,2,9,11])", "(in2[9,12,11])" }, { "1", "2", "4", "9" });
    seedRecursionChain(ana.body, "(in[marker,1])",
        { "(in[11,1])", "(in2[9,12,11])" }, { "1", "9" });
    arriveAtRecursion(ana, "!(sequence[1,4,2,9,20])");
    arriveAtRecursion(ana, "(in[20,1])");
    ASSERT_FALSE(propertyDemandFiled(ana.body, "(in2[9,marker,20])"));
}

TEST(recursion_propagation, compound_premise_with_one_free_argument_is_demanded_and_prepared) {
    gl::tests::ProverFixture ana(std::string("FTA"));
    ana.parameters.trackHistory = false;
    seedRecursionChain(ana.body, "(in2[rec,marker,3])",
        { "(in2[rec,10,3])", "(sequence[1,4,2,10,11])", "(in2[10,12,11])" },
        { "1", "2", "3", "4", "rec" });
    arriveAtRecursion(ana, "(in2[rec,9,3])");

    const std::string demand = "(sequence[1,4,2,9,marker])";
    ASSERT_TRUE(propertyDemandFiled(ana.body, demand));
    ASSERT_EQ(recursionDemandStatus(ana.body, demand), 1);
    ASSERT_TRUE(recursionIntegrationPrepared(ana.body, demand));
    // An element with two free arguments gets nothing.
    ASSERT_FALSE(propertyDemandFiled(ana.body, "(in2[9,marker,11])"));
}

TEST(recursion_propagation, compounds_bind_any_slot_operators_only_their_output) {
    gl::tests::ProverFixture ana(std::string("FTA"));
    ana.parameters.trackHistory = false;
    // A compound statement binds the argument its template marks, whatever
    // the slot: the arriving limitSequence binds the free bound 13 to 9,
    // leaving the value premise with only its output free.
    seedRecursionChain(ana.body, "(limitSequence[1,4,marker,11,12])",
        { "(limitSequence[1,4,13,11,12])", "(in2[13,14,12])" }, { "1", "11", "12", "4" });
    arriveAtRecursion(ana, "(limitSequence[1,4,9,11,12])");
    ASSERT_TRUE(propertyDemandFiled(ana.body, "(in2[9,marker,12])"));
    ASSERT_TRUE(recursionIntegrationPrepared(ana.body, "(in2[9,marker,12])"));

    // An operator premise whose one free argument is an input is not demanded.
    seedRecursionChain(ana.body, "(in2[rec,marker,3])",
        { "(in2[rec,10,3])", "(in2[11,10,3])" }, { "3", "rec" });
    arriveAtRecursion(ana, "(in2[rec,9,3])");
    ASSERT_FALSE(propertyDemandFiled(ana.body, "(in2[marker,9,3])"));
}

TEST(recursion_propagation, relation_premise_and_statement_are_no_participants) {
    gl::tests::ProverFixture ana(std::string("FTA"));
    ana.parameters.trackHistory = false;
    // A relation premise left with one free argument is not demanded ...
    seedRecursionChain(ana.body, "(in2[rec,marker,3])",
        { "(in2[rec,10,3])", "(preorder[1,4,10,11])" }, { "1", "3", "4", "rec" });
    arriveAtRecursion(ana, "(in2[rec,9,3])");
    ASSERT_FALSE(propertyDemandFiled(ana.body, "(preorder[1,4,9,marker])"));
    // ... and an arriving relation statement binds nothing.
    seedRecursionChain(ana.body, "(preorder[1,4,9,marker])",
        { "(preorder[1,4,9,12])", "(in2[12,13,3])" }, { "1", "3", "4", "9" });
    arriveAtRecursion(ana, "(preorder[1,4,9,20])");
    ASSERT_FALSE(propertyDemandFiled(ana.body, "(in2[20,marker,3])"));
}

TEST(recursion_propagation, negated_operator_premise_is_demanded_without_integration) {
    gl::tests::ProverFixture ana(std::string("FTA"));
    ana.parameters.trackHistory = false;
    seedRecursionChain(ana.body, "(in2[rec,marker,3])",
        { "(in2[rec,10,3])", "!(in2[10,12,4])" }, { "3", "4", "rec" });
    arriveAtRecursion(ana, "(in2[rec,9,3])");

    const std::string demand = "!(in2[9,marker,4])";
    ASSERT_TRUE(propertyDemandFiled(ana.body, demand));
    ASSERT_EQ(recursionDemandStatus(ana.body, demand), 1);
    ASSERT_FALSE(recursionIntegrationPrepared(ana.body, demand));
}

// A re-keyed induction chain prepares the integration of every config-backed
// compound premise it leaves fully bound
// (I-294): the member premise of the
// row-106 hypothesis once its sequence is bound, next to the existing derived
// demand for the fold's free output. The arriving element itself is not
// prepared.
TEST(recursion_propagation, fully_bound_compound_premise_of_a_rekeyed_chain_is_prepared) {
    gl::tests::ProverFixture ana(std::string("FTA"));
    ana.parameters.trackHistory = false;  // the prepared rule text does not depend on history
    seedRecursionChain(ana.body, "(sequence[1,4,2,rec,marker])",
        { "(sequence[1,4,2,rec,10])", "(fold[1,3,4,5,10,2,rec,11])", "(inSequence[1,12,10])" },
        { "1", "12", "2", "3", "4", "5", "rec" });
    arriveAtRecursion(ana, "(sequence[1,4,2,rec,20])");

    ASSERT_TRUE(recursionIntegrationPrepared(ana.body, "(inSequence[1,12,20])"));
    ASSERT_FALSE(propertyDemandFiled(ana.body, "(inSequence[1,12,marker])"));
    ASSERT_TRUE(propertyDemandFiled(ana.body, "(fold[1,3,4,5,20,2,rec,marker])"));
    ASSERT_TRUE(recursionIntegrationPrepared(ana.body, "(fold[1,3,4,5,20,2,rec,marker])"));
    ASSERT_FALSE(recursionIntegrationPrepared(ana.body, "(sequence[1,4,2,rec,20])"));
}

// The same holds for a relation premise: fully bound it is one statement, so
// its integration is prepared (only a marked relation stays excluded).
TEST(recursion_propagation, fully_bound_relation_premise_of_a_rekeyed_chain_is_prepared) {
    gl::tests::ProverFixture ana(std::string("FTA"));
    ana.parameters.trackHistory = false;  // the prepared rule text does not depend on history
    seedRecursionChain(ana.body, "(in2[rec,marker,3])",
        { "(in2[rec,10,3])", "(preorder[1,4,10,11])" }, { "1", "11", "3", "4", "rec" });
    arriveAtRecursion(ana, "(in2[rec,9,3])");

    ASSERT_TRUE(recursionIntegrationPrepared(ana.body, "(preorder[1,4,9,11])"));
    ASSERT_FALSE(propertyDemandFiled(ana.body, "(preorder[1,4,9,marker])"));
}

// Not prepared: a fully bound premise already known at the scope, an atomic
// premise and an operator premise (an operator keeps its output demand).
TEST(recursion_propagation, known_atomic_or_operator_bound_premise_is_not_prepared) {
    gl::tests::ProverFixture ana(std::string("FTA"));
    ana.parameters.trackHistory = false;
    seedRecursionChain(ana.body, "(sequence[1,4,2,rec,marker])",
        { "(sequence[1,4,2,rec,10])", "(inSequence[1,12,10])", "(in2[12,13,10])",
          "(fold[1,3,4,5,10,2,rec,14])" },
        { "1", "12", "13", "14", "2", "3", "4", "5", "rec" });
    gl::upsertStatementKey(ana.body.intKnownStatements,
        gl::packStatementKey(ana.body.nameMap.encode(std::string("(inSequence[1,12,20])")),
                             gl::NameMap::MAIN_ID),
        /*local=*/true);
    arriveAtRecursion(ana, "(sequence[1,4,2,rec,20])");

    ASSERT_FALSE(recursionIntegrationPrepared(ana.body, "(inSequence[1,12,20])"));
    ASSERT_FALSE(recursionIntegrationPrepared(ana.body, "(in2[12,13,20])"));
    ASSERT_FALSE(recursionIntegrationPrepared(ana.body, "(fold[1,3,4,5,20,2,rec,14])"));
}

namespace {
    using Demand = gl::AdmissionMemory::ChainElementDemand;

    // Classify `element` against the sorted-unique remaining run `rem`.
    Demand classify(gl::tests::ProverFixture& ea, const std::string& element,
                    const std::vector<std::string>& rem, bool partOfRecursion,
                    std::vector<std::string>& unbound) {
        std::vector<gl::StrSpan> remSpans;
        for (const std::string& r : rem) remSpans.push_back(gl::StrSpan(r));
        gl::StrSpan out[gl::ExecutionParameters::MAX_ARITY];
        int32_t outN = 0;
        const Demand d = ea.admissionMemory.classifyChainElement(gl::StrSpan(element),
            remSpans.data(), static_cast<int32_t>(remSpans.size()), partOfRecursion, out, outN);
        unbound.clear();
        for (int32_t i = 0; i < outN; ++i) unbound.push_back(out[i].toStdString());
        return d;
    }

    // Every integration demand key of `m` as `template@scope`.
    std::set<std::string> integrationDemandKeys(gl::Memory& m) {
        std::set<std::string> out;
        const auto& keys = m.overallHashMemory.admissionMapIntegration;
        for (int32_t id = 1; id <= keys.count(); ++id) {
            if (keys.lookup(keys.keyAt(id)) == 0) continue;
            const gl::StatementKey key = gl::Codec<gl::StatementKey>::decode(keys.keyAt(id));
            out.insert(m.templateInterner.decode(key.orig) + "@" + m.nameMap.decode(key.validity));
        }
        return out;
    }

    // The marked goals (path labels) of the instructions under integration
    // key `key` at `main`.
    std::set<std::string> instructionLabels(gl::Memory& m, const std::string& key) {
        gl::ScratchArena& arena = gl::genScratchArenas().forSlot(gl::genScratchArenas().slotCount() - 1);
        gl::ScratchScope frame(arena);
        const int64_t pk = gl::mintTemplateKey(m.templateInterner, m.nameMap, key, std::string("main"));
        gl::ArenaIntegrationMap entries(&arena, &m.valueInterner);
        entries.buildFromCold(m.overallHashMemory.admissionMapIntegration, pk);
        std::set<std::string> out;
        for (int32_t i = 0; i < entries.entries.size(); ++i) {
            gl::WorkInstruction wi(&arena);
            entries.loadInstructionInto(i, wi, m.valueInterner);
            out.insert(wi.markedGoal().toStdString());
        }
        return out;
    }

    // Whether an installed rule of `m` (the `originals` chains) mentions
    // `token`.
    bool rulesMention(gl::Memory& m, const std::string& token) {
        const auto& chains = m.overallHashMemory.originals;
        for (int32_t id = 1; id <= chains.count(); ++id) {
            const gl::IdVecKeyView kv = gl::viewIdVecKey(chains.inner().keyAt(id));
            for (int32_t t = 0; t < kv.count(); ++t)
                if (m.ruleInterner.decodeView(kv.idAt(t)).toStdString().find(token) != std::string::npos)
                    return true;
        }
        return false;
    }
}

// The install-time demand kinds of a chain element
// (D-586): one unbound argument takes the
// marker route, two make a positive participant two-unbound, a recursion-owned
// compound relation with none is a fully bound premise; two-input relations,
// negated elements with two unbound arguments, negated compounds and elements
// without configuration have no demand.
TEST(two_marker_hypothesis, classify_chain_element_kinds) {
    gl::tests::ProverFixture ea(std::string("FTA"));
    std::vector<std::string> unbound;

    // primeFactorSeq (inputs k, P, output n) is an operator: the marker route
    // with one unbound argument, two-unbound with two, inside or outside a
    // recursion-owned chain; an operator has no fully bound premise route.
    const std::vector<std::string> anchorRec = { "1", "2", "3", "4", "5", "6", "7", "u_rec" };
    const std::string pfs = "(primeFactorSeq[1,3,4,5,2,6,7,u_rec,10,11])";
    ASSERT_TRUE(classify(ea, pfs, anchorRec, true, unbound) == Demand::twoUnbound);
    ASSERT_TRUE(unbound == (std::vector<std::string>{ "10", "11" }));
    ASSERT_TRUE(classify(ea, pfs, anchorRec, false, unbound) == Demand::twoUnbound);
    ASSERT_TRUE(classify(ea, pfs, { "1", "10", "2", "3", "4", "5", "6", "7", "u_rec" }, true, unbound)
        == Demand::singleMarker);
    ASSERT_TRUE(unbound == (std::vector<std::string>{ "11" }));
    ASSERT_TRUE(classify(ea, "!" + pfs, anchorRec, true, unbound) == Demand::none);
    ASSERT_TRUE(classify(ea, "(primeFactorSeq[1,3,4,5,2,6,7,u_rec,u_P,u_n])",
        { "1", "2", "3", "4", "5", "6", "7", "u_P", "u_n", "u_rec" }, true, unbound) == Demand::none);
    ASSERT_TRUE(classify(ea, "(in2[u_rec,10,3])", { "3", "u_rec" }, false, unbound) == Demand::singleMarker);
    ASSERT_TRUE(classify(ea, "(in3[10,11,12,5])", { "12", "5" }, true, unbound) == Demand::twoUnbound);
    ASSERT_TRUE(classify(ea, "(in3[10,11,12,5])", { "5" }, true, unbound) == Demand::none);
    ASSERT_TRUE(classify(ea, "!(in3[10,11,12,5])", { "12", "5" }, true, unbound) == Demand::none);

    // sequence (input m, no output) is a config-backed compound relation:
    // only a recursion-owned chain demands it, positive, and a fully bound
    // one is a fully bound premise.
    const std::string seq = "(sequence[1,4,2,10,11])";
    ASSERT_TRUE(classify(ea, seq, { "1", "2", "4" }, true, unbound) == Demand::twoUnbound);
    ASSERT_TRUE(unbound == (std::vector<std::string>{ "10", "11" }));
    ASSERT_TRUE(classify(ea, seq, { "1", "2", "4" }, false, unbound) == Demand::none);
    ASSERT_TRUE(classify(ea, seq, { "1", "10", "2", "4" }, true, unbound) == Demand::singleMarker);
    ASSERT_TRUE(classify(ea, "!" + seq, { "1", "2", "4" }, true, unbound) == Demand::none);
    ASSERT_TRUE(classify(ea, seq, { "1", "10", "11", "2", "4" }, true, unbound) == Demand::fullyBoundPremise);
    ASSERT_TRUE(classify(ea, seq, { "1", "10", "11", "2", "4" }, false, unbound) == Demand::none);

    // A two-input relation without output keeps no marker route.
    ASSERT_TRUE(classify(ea, "(preorder[1,4,10,11])", { "1", "4" }, true, unbound) == Demand::none);
    ASSERT_TRUE(classify(ea, "(notConfiguredCore[10])", { }, true, unbound) == Demand::none);
}

// FTA2 row 20 (same endpoint): the weak step block's hypothesis chain has no
// ordinary demand (both primeFactorSeq premises have two unbound arguments),
// so its install files the two-marker start, whose preparation keys the
// sequence part of the bundle with a marker (the key the prefix existences
// park under). Row 20 builds no strong step block: its length sits in an
// operator with a numeric output (D-594).
TEST(two_marker_hypothesis, weak_step_block_files_the_two_marker_start) {
    gl::tests::ProverFixture ea(std::string("FTA"));
    const std::string row20 =
        "(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10,11](primeFactorSeq[1,3,4,5,2,6,7,9,10,11])"
        "(>[12](primeFactorSeq[1,3,4,5,2,6,7,9,12,11])(=[10,12]))))";
    ea.proofGridBuilder.addTheoremToMemory(row20, ea.body, 0, false, ea.globalDependencies);

    gl::Memory* anchorLB = ea.simpleMapStore.findChild(&ea.body, "(AnchorFTA[1,2,3,4,5,6,7,8])");
    ASSERT_TRUE(anchorLB != nullptr);
    gl::Memory* firstLB = ea.simpleMapStore.findChild(anchorLB, "(primeFactorSeq[1,3,4,5,2,6,7,9,10,11])");
    ASSERT_TRUE(firstLB != nullptr);
    gl::Memory* goalLB = ea.simpleMapStore.findChild(firstLB, "(primeFactorSeq[1,3,4,5,2,6,7,9,12,11])");
    ASSERT_TRUE(goalLB != nullptr);
    gl::Memory* weak = ea.simpleMapStore.findChild(goalLB, "(in2[rec0,9,3])");
    gl::Memory* strong = ea.simpleMapStore.findChild(goalLB, "(in2[rec1,9,3])");
    ASSERT_TRUE(weak != nullptr && strong == nullptr);

    const auto sequenceDemand = [](const std::set<std::string>& keys) {
        for (const std::string& k : keys)
            if (k.rfind("(sequence[", 0) == 0 && k.find(",marker])@main") != std::string::npos)
                return true;
        return false;
    };
    ASSERT_TRUE(sequenceDemand(integrationDemandKeys(*weak)));

    // The fold part holds both open arguments, so it keys both paths: path A
    // with `marker` on the sequence P and the wildcard `pi_` on the product n,
    // path B the other way round. Each path's copies carry its label, the
    // goal with `marker1` on the path's argument; the sequence key is path A.
    std::string foldA, foldB, sequenceKey;
    for (const std::string& entry : integrationDemandKeys(*weak)) {
        const std::string key = entry.substr(0, entry.rfind('@'));
        if (key.rfind("(sequence[", 0) == 0 && key.find(",marker])") != std::string::npos) sequenceKey = key;
        if (key.rfind("(fold[", 0) != 0) continue;
        const std::size_t marker = key.find("marker");
        const std::size_t wildcard = key.find("pi_");
        if (marker == std::string::npos || wildcard == std::string::npos) continue;
        ASSERT_TRUE(key.find("pi_lev") == std::string::npos);
        (marker < wildcard ? foldA : foldB) = key;
    }
    ASSERT_FALSE(foldA.empty());
    ASSERT_FALSE(foldB.empty());
    ASSERT_FALSE(sequenceKey.empty());
    const auto labelled = [&](const std::string& key, const std::string& tail) {
        const std::set<std::string> labels = instructionLabels(*weak, key);
        if (labels.empty()) return false;
        for (const std::string& label : labels)
            if (label.size() < tail.size() || label.compare(label.size() - tail.size(), tail.size(), tail) != 0)
                return false;
        return true;
    };
    ASSERT_TRUE(labelled(foldA, ",marker1,marker2])"));
    ASSERT_TRUE(labelled(foldB, ",marker2,marker1])"));
    ASSERT_TRUE(labelled(sequenceKey, ",marker1,marker2])"));
    ASSERT_TRUE(gl::AdmissionMemory::integrationWildcardFiled(*weak));

    // The two-marker goal is a demand, never a hypothesis: no installed rule
    // carries `marker1` or `marker2`.
    ASSERT_FALSE(rulesMention(*weak, "marker1"));
    ASSERT_FALSE(rulesMention(*weak, "marker2"));
}

// The wildcard variant of a marked query replaces exactly the argument at one
// position by `pi_`, a repeated name only there, and keeps the sign
// (D-586).
TEST(two_marker_hypothesis, wildcard_query_variant_replaces_one_argument) {
    gl::ScratchArena& arena = gl::scratchArenas().forSlot(gl::scratchArenas().slotCount() - 1);
    gl::ScratchScope frame(arena);
    const auto variant = [&](const std::string& query, int32_t position) {
        return gl::StrSpan(gl::AdmissionMemory::makeWildcardQueryScratch(gl::StrSpan(query), position, arena))
            .toStdString();
    };
    const std::string query = "(fold[u_1,u_3,u_4,u_5,u_P,u_2,u_rec,marker])";
    ASSERT_TRUE(variant(query, 0) == "(fold[pi_,u_3,u_4,u_5,u_P,u_2,u_rec,marker])");
    ASSERT_TRUE(variant(query, 4) == "(fold[u_1,u_3,u_4,u_5,pi_,u_2,u_rec,marker])");
    ASSERT_TRUE(variant(query, 6) == "(fold[u_1,u_3,u_4,u_5,u_P,u_2,pi_,marker])");
    ASSERT_TRUE(variant("!(in2[u_x,u_x,marker])", 1) == "!(in2[u_x,pi_,marker])");
    ASSERT_TRUE(variant("(in2[marker,u_y,u_z])", 2) == "(in2[marker,u_y,pi_])");
}
