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
///     varsInAdmissionMapKeys population skipping "marker", revisitRejected2).
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
    int64_t keyPk = 0;
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
        const gl::NameId markerId = m.templateInterner.lookup("marker");
        ASSERT_TRUE(markerId == 0
                    || !m.overallHashMemory.varsInAdmissionMapKeys.contains(markerId));
    }

    // Release the views before the pages die (the production clear point).
    m.admissionKeysAlgebra.clear();
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
        /*local=*/true);
    gl::upsertStatementKey(m.intKnownStatements,
        gl::packStatementKey(m.nameMap.encode(OTHER), gl::NameMap::MAIN_ID),
        /*local=*/true);
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
        ASSERT_TRUE(otherRow != nullptr);
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

// --- updateAdmissionMap sees premises only, never the head ---
// A premise-completion demand asks for a missing premise instance; the head is
// the rule's product and is never demanded. The ¬∃t level of a negated
// remainder existence has a head-only variable (the `×` anchor slot occurs in
// no premise), which used to file the meaningless demand
// `!(in3[9,10,9,marker])` ("some operator in the × slot").

TEST(admission_reshuffle, install_demands_premises_only_never_the_head) {
    gl::ExpressionAnalyzer ea(std::string("FTA"));
    gl::HashMemory& hm = ea.body.overallHashMemory;
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
    ea.addToHashMemory(chain, 2, gl::StrSpan(head), rem, 4,
        ea.body, hm, lv, 1, gl::StrSpan(impl),
        ea.parameters.maxIterationNumberVariable,
        ea.parameters.standardMaxSecondaryNumber, false,
        ea.parameters.minNumOperatorsKey, gl::StrSpan("implication", 11),
        /*performAdmissionMapUpdate=*/true, gl::StrSpan(impl));

    const auto& am = hm.admissionMap;
    // The bound premise filed its demand (the premise with exactly one
    // non-remaining argument), so the map is not empty ...
    ASSERT_TRUE(am.count() > 0);
    // ... and no key is the head or any negated template.
    int64_t headPk = 0;
    const bool headInterned = gl::lookupTemplateKey(
        ea.body.templateInterner, ea.body.nameMap,
        gl::StrSpan("!(in3[9,10,9,marker])"), gl::StrSpan("main", 4), headPk);
    ASSERT_TRUE(!headInterned || am.lookup(headPk) == 0);
    for (int32_t id = 1; id <= am.count(); ++id) {
        const std::pair<std::string, std::string> kv = gl::decodeTemplateKey(
            am.keyAt(id), ea.body.templateInterner, ea.body.nameMap);
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
        if (!gl::lookupTemplateKey(mb.templateInterner, mb.nameMap,
                gl::StrSpan(templ), gl::StrSpan("main", 4), pk))
            return 0;
        return pk;
    }

    // True iff `templ` at `main` carries an algebra admission entry in `mb`.
    bool propertyDemandFiled(gl::Memory& mb, const std::string& templ) {
        const int64_t pk = propertyDemandKey(mb, templ);
        return pk != 0 && mb.overallHashMemory.admissionMap.lookup(pk) != 0;
    }

    // One updateAdmissionMap call on the FTA analyzer's root LB.
    void runPropertyChain(gl::ExpressionAnalyzer& ana, const std::string& element,
                          const std::vector<std::string>& remaining, bool partOfRecursion) {
        const gl::StrSpan key[] = { gl::StrSpan(element) };
        std::vector<gl::StrSpan> rem;
        for (const std::string& r : remaining) rem.push_back(gl::StrSpan(r));
        ana.updateAdmissionMap(ana.body, key, 1, rem.data(),
            static_cast<int32_t>(rem.size()), 4, 4, partOfRecursion,
            gl::StrSpan("main", 4));
    }
}

TEST(property_input_demand, induction_chain_demands_the_unbound_sequence_input) {
    gl::ExpressionAnalyzer ana(std::string("FTA"));
    ana.parameters.trackHistory = false;
    runPropertyChain(ana, "(sequence[1,4,2,9,11])", { "1", "2", "4", "9" }, true);

    const std::string demand = "(sequence[1,4,2,9,marker])";
    ASSERT_TRUE(propertyDemandFiled(ana.body, demand));
    const uint8_t* status = ana.body.overallHashMemory.admissionStatusMap.find(
        propertyDemandKey(ana.body, demand));
    ASSERT_TRUE(status != nullptr);
    ASSERT_EQ(static_cast<int>(*status), 1);
    // The integration of the marker form was prepared.
    const gl::NameId tid = ana.body.templateInterner.lookup(gl::StrSpan(demand));
    ASSERT_TRUE(tid != 0);
    ASSERT_TRUE(ana.body.integrationStartIntMap.lookup(tid) != 0);
}

TEST(property_input_demand, ordinary_chain_keeps_the_d257_skip) {
    gl::ExpressionAnalyzer ana(std::string("FTA"));
    ana.parameters.trackHistory = false;
    runPropertyChain(ana, "(sequence[1,4,2,9,11])", { "1", "2", "4", "9" }, false);
    ASSERT_FALSE(propertyDemandFiled(ana.body, "(sequence[1,4,2,9,marker])"));
}

TEST(property_input_demand, any_single_free_argument_is_demanded) {
    gl::ExpressionAnalyzer ana(std::string("FTA"));
    ana.parameters.trackHistory = false;
    // The endpoint 9 is the one free argument: it is demanded and prepared.
    runPropertyChain(ana, "(sequence[1,4,2,9,11])", { "1", "11", "2", "4" }, true);
    const std::string demand = "(sequence[1,4,2,marker,11])";
    ASSERT_TRUE(propertyDemandFiled(ana.body, demand));
    const gl::NameId tid = ana.body.templateInterner.lookup(gl::StrSpan(demand));
    ASSERT_TRUE(tid != 0);
    ASSERT_TRUE(ana.body.integrationStartIntMap.lookup(tid) != 0);
    // Two free arguments: no demand.
    runPropertyChain(ana, "(interval[1,4,2,12,13])", { "1", "2", "4" }, true);
    ASSERT_FALSE(propertyDemandFiled(ana.body, "(interval[1,4,2,marker,13])"));
    ASSERT_FALSE(propertyDemandFiled(ana.body, "(interval[1,4,2,12,marker])"));
}

TEST(property_input_demand, relation_keeps_the_d257_skip) {
    gl::ExpressionAnalyzer ana(std::string("FTA"));
    ana.parameters.trackHistory = false;
    // preorder has two inputs and no output: a relation, never demanded.
    runPropertyChain(ana, "(preorder[1,4,9,11])", { "1", "4", "9" }, true);
    ASSERT_FALSE(propertyDemandFiled(ana.body, "(preorder[1,4,9,marker])"));
}

TEST(property_input_demand, negated_property_keeps_the_d257_skip) {
    gl::ExpressionAnalyzer ana(std::string("FTA"));
    ana.parameters.trackHistory = false;
    runPropertyChain(ana, "!(sequence[1,4,2,9,11])", { "1", "2", "4", "9" }, true);
    ASSERT_FALSE(propertyDemandFiled(ana.body, "!(sequence[1,4,2,9,marker])"));
    ASSERT_FALSE(propertyDemandFiled(ana.body, "(sequence[1,4,2,9,marker])"));
}

TEST(property_input_demand, f3_induction_variable_is_the_lower_endpoint) {
    gl::ExpressionAnalyzer ea(std::string("FTA"));
    gl::StrSpan out[gl::ExecutionParameters::MAX_ADMISSION_REM_ARGS];
    // Reverse and forward F3: the sequence endpoint 10 is an input but also
    // the successor output, so the only induction variable is 9.
    const std::string reverseF3 =
        "(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](in2[9,10,3])(>[11](sequence[1,4,2,10,11])(>[12](in2[10,12,11])(>[13](fold[1,3,4,5,11,2,9,13])(>[14](fold[1,3,4,5,11,2,10,14])(in3[13,12,14,5])))))))";
    const std::string forwardF3 =
        "(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](in2[9,10,3])(>[11](sequence[1,4,2,10,11])(>[12](in2[10,12,11])(>[13](fold[1,3,4,5,11,2,9,13])(>[14](in3[13,12,14,5])(fold[1,3,4,5,11,2,10,14])))))))";
    for (const std::string* theorem : { &reverseF3, &forwardF3 }) {
        const int32_t n = ea.findDigitArgs(*theorem, ea.anchorInfo, out,
            gl::ExecutionParameters::MAX_ADMISSION_REM_ARGS);
        ASSERT_EQ(n, 1);
        ASSERT_EQ(out[0].toStdString(), std::string("9"));
    }
}

namespace {
    // Drain one staged `(in2[10,marker,3])` record with the given recursion
    // flag into `m`, optionally after seeding the template's status.
    void drainStatusRecord(gl::ExpressionAnalyzer& ea, gl::Memory& m,
                           bool recordFlag, int seededStatus) {
        gl::SealedPageSet pages;
        pages.bind(&gl::staticMemory());
        const auto s = [&pages](const std::string& v) {
            return gl::SealedString::copyFrom(pages, v.data(),
                                              static_cast<int32_t>(v.size()));
        };
        const std::string keyText = "(in2[10,marker,3])";
        if (seededStatus >= 0) {
            const int64_t pk = gl::mintTemplateKey(m.templateInterner, m.nameMap,
                keyText, std::string("main"));
            m.overallHashMemory.admissionStatusMap.upsert(pk,
                static_cast<uint8_t>(seededStatus));
        }
        gl::StagedAdmissionValue value;
        const gl::SealedString keyArr[2] = { s("10"), s("3") };
        value.key = gl::SealedSpan<gl::SealedString>::copyFrom(pages, keyArr, 2);
        value.flag = recordFlag;
        m.admissionKeysAlgebra.push_back(gl::AdmissionKeyAlgebraRecord{
            gl::SealedExpressionWithValidity{ s(keyText), s("main") }, value });
        ea.drainAdmissionKeysAlgebra(m);
        m.admissionKeysAlgebra.clear();
        pages.seal();
        pages.freePages();
    }

    // The drained template's status, -1 when absent.
    int drainedStatus(gl::Memory& m) {
        const int64_t pk = propertyDemandKey(m, "(in2[10,marker,3])");
        if (pk == 0) return -1;
        const uint8_t* status = m.overallHashMemory.admissionStatusMap.find(pk);
        return status == nullptr ? -1 : static_cast<int>(*status);
    }
}

TEST(admission_reshuffle, drain_keeps_an_existing_recursion_status) {
    gl::ExpressionAnalyzer ea("Peano");
    gl::Memory m;
    drainStatusRecord(ea, m, /*recordFlag=*/false, /*seededStatus=*/1);
    ASSERT_EQ(drainedStatus(m), 1);
}

TEST(admission_reshuffle, drain_sets_recursion_status_for_a_recursion_record) {
    gl::ExpressionAnalyzer ea("Peano");
    gl::Memory m;
    drainStatusRecord(ea, m, /*recordFlag=*/true, /*seededStatus=*/-1);
    ASSERT_EQ(drainedStatus(m), 1);
}

TEST(admission_reshuffle, drain_leaves_an_ordinary_template_ordinary) {
    gl::ExpressionAnalyzer ea("Peano");
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
    // and status 1.
    void seedRecursionChain(gl::Memory& mb, const std::string& templ,
                            const std::vector<std::string>& chain,
                            const std::vector<std::string>& remaining) {
        const int64_t key = gl::mintTemplateKey(mb.templateInterner, mb.nameMap,
            templ, std::string("main"));
        gl::AdmissionMapValue value;
        for (const std::string& element : chain)
            value.key.push_back(mb.valueInterner.encode(element));
        for (const std::string& arg : remaining)
            value.remainingArgs.push_back(mb.valueInterner.encode(arg));
        value.standardMaxAdmissionDepth = 1;
        value.standardMaxSecondaryNumber = 2;
        value.flag = true;
        auto& arena = gl::genScratchArenas().forSlot(gl::genScratchArenas().slotCount() - 1);
        gl::insertAdmissionValue(mb.overallHashMemory.admissionMap, key, value,
                                 mb.valueInterner, arena);
        mb.overallHashMemory.admissionStatusMap.upsert(key, uint8_t(1));
    }

    // The status of `templ` at main, -1 when the template has none.
    int recursionDemandStatus(gl::Memory& mb, const std::string& templ) {
        const int64_t pk = propertyDemandKey(mb, templ);
        if (pk == 0) return -1;
        const uint8_t* status = mb.overallHashMemory.admissionStatusMap.find(pk);
        return status == nullptr ? -1 : static_cast<int>(*status);
    }

    // True iff the integration of `templ` was prepared (its start-int row).
    bool recursionIntegrationPrepared(gl::Memory& mb, const std::string& templ) {
        const gl::NameId tid = mb.templateInterner.lookup(gl::StrSpan(templ));
        return tid != 0 && mb.integrationStartIntMap.lookup(tid) != 0;
    }

    // One deposit-door call for `statement` at main; `inductionAllowed` is
    // the door's verdict on the disintegrated original.
    void arriveAtRecursion(gl::ExpressionAnalyzer& ana, const std::string& statement,
                           bool inductionAllowed = true) {
        ana.updateAdmissionMapRecursion(gl::StrSpan(statement), ana.body,
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
    gl::ExpressionAnalyzer ana(std::string("FTA"));
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
    gl::ExpressionAnalyzer ana(std::string("FTA"));
    ana.parameters.trackHistory = false;
    seedReverseF3Hypothesis(ana.body);
    const auto before = ana.body.overallHashMemory.admissionMap.count();
    arriveAtRecursion(ana, "(sequence[1,4,2,9,20])", false);

    for (const std::string demand : { std::string("(in2[9,marker,20])"),
                                      std::string("(fold[1,3,4,5,20,2,rec,marker])"),
                                      std::string("(fold[1,3,4,5,20,2,9,marker])") }) {
        ASSERT_FALSE(propertyDemandFiled(ana.body, demand));
        ASSERT_FALSE(recursionIntegrationPrepared(ana.body, demand));
    }
    ASSERT_EQ(ana.body.overallHashMemory.admissionMap.count(), before);
    ASSERT_EQ(recursionDemandStatus(ana.body, "(sequence[1,4,2,9,marker])"), 1);
}

// A construction of the induction hypothesis' own head moves no induction
// chain (I-286): the arrival that re-keys
// the chain above derives nothing, prepares no integration, enrolls no product
// of recursion and leaves the hypothesis template as it was.
TEST(recursion_propagation, construction_of_the_induction_head_moves_no_chain) {
    gl::ExpressionAnalyzer ana(std::string("FTA"));
    ana.parameters.trackHistory = false;
    seedReverseF3Hypothesis(ana.body);
    ana.body.recursionHeadWitnesses.mint(gl::StrSpan("20", 2));
    const auto before = ana.body.overallHashMemory.admissionMap.count();
    const auto productsBefore = ana.body.overallHashMemory.productsOfRecursionIds.count();
    arriveAtRecursion(ana, "(sequence[1,4,2,9,20])");

    for (const std::string demand : { std::string("(in2[9,marker,20])"),
                                      std::string("(fold[1,3,4,5,20,2,rec,marker])"),
                                      std::string("(fold[1,3,4,5,20,2,9,marker])") }) {
        ASSERT_FALSE(propertyDemandFiled(ana.body, demand));
        ASSERT_FALSE(recursionIntegrationPrepared(ana.body, demand));
    }
    ASSERT_EQ(ana.body.overallHashMemory.admissionMap.count(), before);
    ASSERT_EQ(ana.body.overallHashMemory.productsOfRecursionIds.count(), productsBefore);
    ASSERT_EQ(recursionDemandStatus(ana.body, "(sequence[1,4,2,9,marker])"), 1);
}

TEST(recursion_propagation, only_the_operator_output_probe_leaves_an_entry) {
    gl::ExpressionAnalyzer ana(std::string("FTA"));
    ana.parameters.trackHistory = false;
    const int32_t before = ana.body.overallHashMemory.admissionStatusMap.count();
    arriveAtRecursion(ana, "(sequence[1,4,2,9,20])");
    ASSERT_EQ(ana.body.overallHashMemory.admissionStatusMap.count(), before);
    ASSERT_EQ(propertyDemandKey(ana.body, "(sequence[1,4,2,9,marker])"), 0);
    // An unmatched operator statement keeps the historical default-false entry
    // of its output probe; its other probes leave nothing.
    arriveAtRecursion(ana, "(in2[9,20,3])");
    ASSERT_EQ(recursionDemandStatus(ana.body, "(in2[9,marker,3])"), 0);
    ASSERT_EQ(ana.body.overallHashMemory.admissionStatusMap.count(), before + 1);
    ASSERT_EQ(propertyDemandKey(ana.body, "(in2[marker,20,3])"), 0);
}

TEST(recursion_propagation, negated_compound_or_atomic_statement_is_no_participant) {
    gl::ExpressionAnalyzer ana(std::string("FTA"));
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
    gl::ExpressionAnalyzer ana(std::string("FTA"));
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
    gl::ExpressionAnalyzer ana(std::string("FTA"));
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
    gl::ExpressionAnalyzer ana(std::string("FTA"));
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
    gl::ExpressionAnalyzer ana(std::string("FTA"));
    ana.parameters.trackHistory = false;
    seedRecursionChain(ana.body, "(in2[rec,marker,3])",
        { "(in2[rec,10,3])", "!(in2[10,12,4])" }, { "3", "4", "rec" });
    arriveAtRecursion(ana, "(in2[rec,9,3])");

    const std::string demand = "!(in2[9,marker,4])";
    ASSERT_TRUE(propertyDemandFiled(ana.body, demand));
    ASSERT_EQ(recursionDemandStatus(ana.body, demand), 1);
    ASSERT_FALSE(recursionIntegrationPrepared(ana.body, demand));
}
