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
/// @brief Unit tests for `rejectedMapOrdis` — the parked or-cohort map of the
///        admission-based ordis polarity.
///
/// @details
/// Coverage for the `RejectedMapOrdisValue` codec family (byte twins, run
/// order, blob view, snapshot), the map-level scope-wipe erase, and the
/// `applyEquivalenceClassToRejectedMapOrdis` equi-class hook (I-37 drop +
/// re-mail, never a direct re-insert). The mirror contract with
/// `rejectedMap` is the tested property throughout.

#include "../compiler/compact_expression_syntax.hpp"
#include "../memory.hpp"
#include <cstddef>
#include <cstring>
#include <utility>
#include "test_harness.hpp"
#include "prover_fixture.hpp"
#include "disintegration_processor_fixture.hpp"
#include "equality_propagation_fixture.hpp"
#include "equivalence_admission_rewriter_fixture.hpp"

#include "compiled_definition_fixture.hpp"
#include "canonical_test_helpers.hpp"

#include "../prover.hpp"

namespace {
    /// Standalone-map rig: one arena + one value interner, no Memory.
    struct OrdisRig {
        gl::GlobalMemoryManager g;
        struct Init {
            Init(gl::GlobalMemoryManager& m) {
                m.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
            }
        } init{ g };
        gl::DirtyState d = gl::DirtyState::Clean;
        gl::LbArena lb{ &g };
        gl::ColdStringTable valTable{ &lb, &d };
        gl::ValueInterner vi;
        OrdisRig() { vi.bind(&valTable); }
    };
} // namespace

// insertRejectedOrdisIdsBlob == insertRejectedOrdisValue per record byte,
// on twin maps fed the same shuffled sequence (empty levels, multi-level,
// duplicate re-park deduped by the RMW).
TEST(rejected_ordis, ids_blob_matches_value_form) {
    OrdisRig rig;
    gl::TypedColdBlobMap<int64_t, gl::RejectedMapOrdisValue> mapV(&rig.lb, &rig.d);
    gl::TypedColdBlobMap<int64_t, gl::RejectedMapOrdisValue> mapB(&rig.lb, &rig.d);
    gl::ScratchArena& tArena = gl::genScratchArenas().forSlot(
        gl::genScratchArenas().slotCount() - 1);

    const int32_t sZZ = rig.vi.encode("(or0[1,zz,3,2])");
    const int32_t sAA = rig.vi.encode("(or0[1,aa,3,2])");
    struct Rec { int32_t stmt; std::vector<int> levels; };
    const std::vector<Rec> recs = {
        { sZZ, { 1, 4 } },
        { sAA, {} },            // empty levels
        { sAA, { 0, 2, 7 } },
        { sZZ, { 1, 4 } },      // duplicate re-park (deduped)
    };
    const int64_t pk = 11;
    for (const Rec& rc : recs) {
        gl::RejectedMapOrdisValue v(rc.stmt,
            rc.levels.empty() ? nullptr : rc.levels.data(),
            static_cast<int32_t>(rc.levels.size()));
        gl::insertRejectedOrdisValue(mapV, pk, v, rig.vi, tArena);
        gl::insertRejectedOrdisIdsBlob(mapB, pk, rc.stmt,
            rc.levels.empty() ? nullptr : rc.levels.data(),
            static_cast<int32_t>(rc.levels.size()), rig.vi, tArena);
    }
    const int32_t idV = mapV.lookup(pk);
    const int32_t idB = mapB.lookup(pk);
    ASSERT_TRUE(idV != 0 && idB != 0);
    ASSERT_EQ(mapV.runLen(idV), 3);   // duplicate deduped
    ASSERT_EQ(mapV.runLen(idV), mapB.runLen(idB));
    for (int32_t j = 0; j < mapV.runLen(idV); ++j) {
        int32_t lV = 0, lB = 0;
        const char* pV = mapV.peekRecordBytes(idV, j, lV, tArena);
        const std::vector<char> bufV(pV, pV + lV);
        const char* pB = mapB.peekRecordBytes(idB, j, lB, tArena);
        ASSERT_EQ(lV, lB);
        ASSERT_TRUE(std::memcmp(bufV.data(), pB, static_cast<size_t>(lV)) == 0);
    }
}

// The stored run is canonical: anti-lex mint order + shuffled inserts, and
// the run order still equals the DecodedRejectedOrdisValueLess set order.
TEST(rejected_ordis, run_order_matches_decoded_set_oracle) {
    OrdisRig rig;
    gl::TypedColdBlobMap<int64_t, gl::RejectedMapOrdisValue> m(&rig.lb, &rig.d);
    gl::ScratchArena& tArena = gl::genScratchArenas().forSlot(
        gl::genScratchArenas().slotCount() - 1);

    // Anti-lex mints: ids ascend while decoded order descends.
    const int32_t sZZ = rig.vi.encode("(or0[1,zz,3,2])");
    const int32_t sMM = rig.vi.encode("(or0[1,mm,3,2])");
    const int32_t sAA = rig.vi.encode("(or0[1,aa,3,2])");

    const int64_t pk = 5;
    const int lvA[] = { 2 };
    const int lvB[] = { 1, 3 };
    gl::insertRejectedOrdisValue(m, pk,
        gl::RejectedMapOrdisValue(sZZ, lvA, 1), rig.vi, tArena);
    gl::insertRejectedOrdisValue(m, pk,
        gl::RejectedMapOrdisValue(sAA, lvB, 2), rig.vi, tArena);
    gl::insertRejectedOrdisValue(m, pk,
        gl::RejectedMapOrdisValue(sMM, nullptr, 0), rig.vi, tArena);
    gl::insertRejectedOrdisValue(m, pk,
        gl::RejectedMapOrdisValue(sAA, lvA, 1), rig.vi, tArena);

    const gl::RejectedOrdisValueSet oracle =
        gl::rejectedOrdisRecordsAt(m, pk, rig.vi);
    const int32_t id = m.lookup(pk);
    ASSERT_TRUE(id != 0);
    ASSERT_EQ(m.runLen(id), static_cast<int32_t>(oracle.size()));
    int32_t j = 0;
    for (const gl::RejectedMapOrdisValue& o : oracle) {
        const gl::RejectedMapOrdisValue r = m.recordAt(id, j++);
        ASSERT_EQ(r.orStatement, o.orStatement);
        ASSERT_TRUE(r.levels == o.levels);
    }
}

// RejectedOrdisValueBlobView: per-field peeks equal the codec's deserialize
// (empty and populated levels); the framing assert is exercised implicitly
// by every construction.
TEST(rejected_ordis, blob_view_matches_deserialize) {
    const auto check = [](const gl::RejectedMapOrdisValue& v) {
        const std::vector<char> blob =
            gl::Codec<gl::RejectedMapOrdisValue>::serialize(v);
        const gl::RejectedMapOrdisValue back =
            gl::Codec<gl::RejectedMapOrdisValue>::deserialize(
                blob.data(), static_cast<int32_t>(blob.size()));
        const gl::RejectedOrdisValueBlobView view(
            blob.data(), static_cast<int32_t>(blob.size()));
        ASSERT_EQ(view.orStatementId(), back.orStatement);
        ASSERT_EQ(view.levelCount(), static_cast<int32_t>(back.levels.size()));
        int32_t i = 0;
        for (const int lv : back.levels)
            ASSERT_EQ(view.levelAt(i++), lv);
    };
    check(gl::RejectedMapOrdisValue(42, std::set<int>{ 0, 3, 9 }));
    check(gl::RejectedMapOrdisValue(7, std::set<int>{}));
    check(gl::RejectedMapOrdisValue(-3, std::set<int>{ 1 }));
}

// snapshotRejectedOrdisRun reproduces rejectedOrdisRecordsAt field-for-field
// in canonical order; a never-minted key yields count == 0 (defined miss).
TEST(rejected_ordis, snapshot_matches_records_at) {
    OrdisRig rig;
    gl::TypedColdBlobMap<int64_t, gl::RejectedMapOrdisValue> m(&rig.lb, &rig.d);
    gl::ScratchArena& tArena = gl::genScratchArenas().forSlot(
        gl::genScratchArenas().slotCount() - 1);

    ASSERT_EQ(gl::snapshotRejectedOrdisRun(m, 99, tArena).count, 0);

    const int32_t sZZ = rig.vi.encode("(or0[1,zz,3,2])");
    const int32_t sAA = rig.vi.encode("(or0[1,aa,3,2])");
    const int64_t pk = 6;
    const int lv[] = { 1, 4 };
    gl::insertRejectedOrdisValue(m, pk,
        gl::RejectedMapOrdisValue(sZZ, lv, 2), rig.vi, tArena);
    gl::insertRejectedOrdisValue(m, pk,
        gl::RejectedMapOrdisValue(sAA, nullptr, 0), rig.vi, tArena);

    const gl::ArenaOffset mark = tArena.cursor();
    const gl::RejectedOrdisRunSnapshot snap =
        gl::snapshotRejectedOrdisRun(m, pk, tArena);
    const gl::RejectedOrdisValueSet oracle =
        gl::rejectedOrdisRecordsAt(m, pk, rig.vi);
    ASSERT_EQ(snap.count, static_cast<int32_t>(oracle.size()));
    int32_t j = 0;
    for (const gl::RejectedMapOrdisValue& o : oracle) {
        const gl::RejectedOrdisValueBlobView& v = snap.views[j++];
        ASSERT_EQ(v.orStatementId(), o.orStatement);
        ASSERT_EQ(v.levelCount(), static_cast<int32_t>(o.levels.size()));
        int32_t i = 0;
        for (const int lvv : o.levels)
            ASSERT_EQ(v.levelAt(i++), lvv);
    }
    tArena.popTo(mark);
}

// Scope-wipe semantics at the map level: the wipeHashMem enrollment erases
// exactly the keys whose validity half is closed, survivors untouched (the
// same eraseBlobIf predicate shape wipeSubtree applies).
TEST(rejected_ordis, scope_erase_blob_if_drops_closed_validity) {
    gl::tests::ProverFixture ea("Peano");
    gl::Memory m;
    gl::ScratchArena& tArena = gl::genScratchArenas().forSlot(
        gl::genScratchArenas().slotCount() - 1);

    const int64_t pkMain = gl::mintTemplateKey(((m).templateInterner), ((m).nameMap),
        "(in2[marker,mm,3])", "main");
    const int64_t pkSub = gl::mintTemplateKey(((m).templateInterner), ((m).nameMap),
        "(in2[marker,mm,3])", "main_boundary_sub_(h[1])");
    const gl::NameId subVid = gl::Codec<gl::StatementKey>::decode(pkSub).validity;

    const int32_t sMM = ((m).valueInterner).encode("(or0[1,mm,3,2])");
    const int lv[] = { 1 };
    gl::insertRejectedOrdisValue(((((m).overallHashMemory)).rejectedMapOrdis), pkMain,
        gl::RejectedMapOrdisValue(sMM, lv, 1), ((m).valueInterner), tArena);
    gl::insertRejectedOrdisValue(((((m).overallHashMemory)).rejectedMapOrdis), pkSub,
        gl::RejectedMapOrdisValue(sMM, lv, 1), ((m).valueInterner), tArena);

    ((((m).overallHashMemory)).rejectedMapOrdis).eraseBlobIf(
        [subVid](int64_t k) {
            return gl::Codec<gl::StatementKey>::decode(k).validity == subVid;
        });

    ASSERT_TRUE(((((m).overallHashMemory)).rejectedMapOrdis).lookup(pkSub) == 0);
    ASSERT_TRUE(((((m).overallHashMemory)).rejectedMapOrdis).lookup(pkMain) != 0);
}

// applyEquivalenceClassToRejectedMapOrdis — drops the changed key and mails
// the rewritten OR STATEMENT onto sameIterationInternalMail (drop + mail,
// never a direct re-key into rejectedMapOrdis) — I-37 for the parked map.
TEST(prover, apply_equi_rejected_ordis_drops_and_mails) {
    gl::tests::ProverFixture ea("Peano");
    gl::Memory m;
    gl::ScratchArena& tArena = gl::genScratchArenas().forSlot(
        gl::genScratchArenas().slotCount() - 1);

    const int64_t oldPk = gl::mintTemplateKey(((m).templateInterner), ((m).nameMap),
        "(in2[marker,it_5_lev_1_2,3])", "main");
    const int32_t stmtId =
        ((m).valueInterner).encode("(or0[1,it_5_lev_1_2,3,2])");
    const int lv[] = { 0, 4 };
    gl::insertRejectedOrdisValue(((((m).overallHashMemory)).rejectedMapOrdis), oldPk,
        gl::RejectedMapOrdisValue(stmtId, lv, 2), ((m).valueInterner), tArena);

    gl::EquivalenceClass cls;
    cls.setMembersFromNames({ "it_0_lev_1_2", "it_5_lev_1_2" }, ((m).nameMap));

    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::RejectedValidityBuckets buckets(&tArena, &dirty);
    seedFixtureClassEvidence(m,cls,gl::StrSpan("main"));
    ea.equivalenceAdmissionRewriter.applyEquivalenceClassToRejectedMapOrdis(cls, m, gl::StrSpan("main", 4), buckets);

    // Old key dropped; NO direct re-key under the canonical template.
    ASSERT_TRUE(((((m).overallHashMemory)).rejectedMapOrdis).lookup(oldPk) == 0);
    int64_t canonPk = 0;
    if (gl::lookupTemplateKey(((m).templateInterner), ((m).nameMap),
            "(in2[marker,it_0_lev_1_2,3])", "main", canonPk)) {
        ASSERT_TRUE(((((m).overallHashMemory)).rejectedMapOrdis).lookup(canonPk) == 0);
    }

    // Rewritten or statement mailed onto sameIterationInternalMail at "main".
    bool foundMail = false;
    const gl::Mail mailSnap = gl::makeHeapMail(((m).sameIterationInternalMail),
                                               ((m).nameMap), ((m).originInterner));
    for (const auto& pr : mailSnap.statements) {
        if (pr.first.original == "(or0[1,it_0_lev_1_2,3,2])"
            && pr.first.validityName == "main") {
            foundMail = true;
        }
    }
    ASSERT_TRUE(foundMail);
}

TEST(prover, canonical_or_value_rewrite_preserves_other_values_and_ancestor_citations) {
    for(bool second : {false,true}) {
        gl::tests::ProverFixture analyzer("Peano");
        analyzer.parameters.trackHistory=true;
        gl::Memory memory;
        auto& arena=gl::genScratchArenas().forSlot(gl::genScratchArenas().slotCount()-1);
        const auto child=((memory).nameMap).encodePush(gl::NameMap::MAIN_ID,"park_child");
        const std::string validity=((memory).nameMap).decode(child);
        const auto key=gl::mintTemplateKey(((memory).templateInterner),((memory).nameMap),"(in2[c,marker,d])",validity);
        gl::EquivalenceClass cls; cls.setMembersFromNames({"a","b"},((memory).nameMap));
        seedFixtureClassEvidence(memory,cls,gl::StrSpan("main"));
        const int level=2;
        const int changed=((memory).valueInterner).encode("(or0[b,c])");
        const int untouched=((memory).valueInterner).encode("(or0[f,g])");
        const auto check=[&](auto& map) {
            gl::insertRejectedOrdisValue(map,key,gl::RejectedMapOrdisValue(changed,&level,1),((memory).valueInterner),arena);
            gl::insertRejectedOrdisValue(map,key,gl::RejectedMapOrdisValue(untouched,&level,1),((memory).valueInterner),arena);
            gl::DirtyState dirty=gl::DirtyState::Clean;
            gl::RejectedValidityBuckets buckets(&arena,&dirty);
            gl::TypedColdSet<int64_t> selected(&arena,&dirty); selected.mint(key);
            gl::tests::EquivalenceAdmissionRewriterFixture::applyEquivalenceClassToRejectedOrdisCore(analyzer.equivalenceAdmissionRewriter, cls,memory,gl::StrSpan(validity),map,buckets,&selected);
            const int run=map.lookup(key);
            ASSERT_TRUE(run!=0);
            ASSERT_EQ(map.runLen(run),1);
            ASSERT_EQ(map.recordAt(run,0).orStatement,untouched);
        };
        if(second) check(((((memory).overallHashMemory)).rejectedMapOrdis2));
        else check(((((memory).overallHashMemory)).rejectedMapOrdis));
        const auto mail=gl::makeHeapMail(((memory).sameIterationInternalMail),((memory).nameMap),((memory).originInterner));
        ASSERT_EQ(mail.statements.size(),size_t(1));
        bool cited=false;
        for(const auto& item:((mail).exprOriginMap)) for(const auto& row:item.second)
            for(const auto& dependency:row.second)
                if(dependency.original=="(=[b,a])") {
                    ASSERT_EQ(dependency.validityName,std::string("main")); cited=true;
                }
        ASSERT_TRUE(cited);
    }
}

// The parked-or hook reads the per-apply posting index: only keys at the
// class validity whose template carries a non-canonical member are visited
// (a nested template's inner argument counts — the substitution-token rule),
// other-validity keys, member-free keys and keys whose token merely starts
// with a member stay parked, every hit's rewritten statement is mailed, and
// a key parked AFTER the index was built (the map's insert epoch moved) is
// visited by the next call — the index is refreshed, not a snapshot.
TEST(prover, apply_equi_rejected_ordis_uses_token_postings_and_refreshes) {
    gl::tests::ProverFixture ea("Peano");
    gl::Memory m;
    auto& rmo = ((((m).overallHashMemory)).rejectedMapOrdis);
    gl::ScratchArena& tArena = gl::genScratchArenas().forSlot(
        gl::genScratchArenas().slotCount() - 1);
    auto park = [&](const char* tmpl, const char* vld, const char* stmt) {
        const int64_t pk = gl::mintTemplateKey(((m).templateInterner), ((m).nameMap),
                                               tmpl, vld);
        const int32_t stmtId = ((m).valueInterner).encode(stmt);
        const int lv[] = { 0 };
        gl::insertRejectedOrdisValue(rmo, pk,
            gl::RejectedMapOrdisValue(stmtId, lv, 1), ((m).valueInterner), tArena);
        return pk;
    };
    const int64_t hitFlat = park("(in2[marker,it_5_lev_1_2,3])", "main",
                                 "(or0[1,it_5_lev_1_2,3,2])");
    const int64_t hitNested = park("(and2[(f[it_5_lev_1_2,marker]),marker])", "main",
                                   "(or0[1,it_5_lev_1_2,4,2])");
    const int64_t otherScope = park("(in2[marker,it_5_lev_1_2,5])", "main_boundary_z",
                                    "(or0[1,it_5_lev_1_2,5,2])");
    const int64_t noMember = park("(in2[marker,q,3])", "main",
                                  "(or0[1,q,3,2])");
    const int64_t prefixOnly = park("(in2[marker,it_5_lev_1_22,3])", "main",
                                    "(or0[1,it_5_lev_1_22,3,2])");
    // Production statements mint their argument names; the fixture mints the
    // one whose posting must NOT match the member (its token is longer).
    ((m).nameMap).encode(gl::StrSpan("it_5_lev_1_22", 13));

    gl::EquivalenceClass cls;
    cls.setMembersFromNames({ "it_0_lev_1_2", "it_5_lev_1_2" }, ((m).nameMap));

    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::RejectedValidityBuckets buckets(&tArena, &dirty);
    seedFixtureClassEvidence(m,cls,gl::StrSpan("main"));
    ea.equivalenceAdmissionRewriter.applyEquivalenceClassToRejectedMapOrdis(cls, m, gl::StrSpan("main", 4), buckets);

    ASSERT_TRUE(buckets.built());
    ASSERT_EQ(buckets.countAtBuild(), 5);
    ASSERT_TRUE(rmo.lookup(hitFlat) == 0);
    ASSERT_TRUE(rmo.lookup(hitNested) == 0);
    ASSERT_TRUE(rmo.lookup(otherScope) != 0);   // another validity: never visited
    ASSERT_TRUE(rmo.lookup(noMember) != 0);     // same validity, no member: kept
    ASSERT_TRUE(rmo.lookup(prefixOnly) != 0);   // member is a proper prefix: kept
    ASSERT_EQ(rmo.count(), 3);
    {
        bool mailedFlat = false, mailedNested = false;
        const gl::Mail mailSnap = gl::makeHeapMail(((m).sameIterationInternalMail),
                                                   ((m).nameMap), ((m).originInterner));
        for (const auto& pr : mailSnap.statements) {
            if (pr.first.validityName != "main") continue;
            if (pr.first.original == "(or0[1,it_0_lev_1_2,3,2])") mailedFlat = true;
            if (pr.first.original == "(or0[1,it_0_lev_1_2,4,2])") mailedNested = true;
        }
        ASSERT_TRUE(mailedFlat);
        ASSERT_TRUE(mailedNested);
    }

    // A later park moves the map's insert epoch: the next call rebuilds the
    // index over the current key set and visits the new key.
    const int64_t late = park("(in2[marker,it_5_lev_1_2,6])", "main",
                              "(or0[1,it_5_lev_1_2,6,2])");
    seedFixtureClassEvidence(m,cls,gl::StrSpan("main"));
    ea.equivalenceAdmissionRewriter.applyEquivalenceClassToRejectedMapOrdis(cls, m, gl::StrSpan("main", 4), buckets);
    ASSERT_EQ(buckets.countAtBuild(), 4);
    ASSERT_TRUE(rmo.lookup(late) == 0);
    ASSERT_EQ(rmo.count(), 3);

    // No insert since: the index is the cached one, nothing to drop.
    seedFixtureClassEvidence(m,cls,gl::StrSpan("main"));
    ea.equivalenceAdmissionRewriter.applyEquivalenceClassToRejectedMapOrdis(cls, m, gl::StrSpan("main", 4), buckets);
    ASSERT_EQ(buckets.countAtBuild(), 4);
    ASSERT_EQ(rmo.count(), 3);
}

namespace {
    /// Hand-built or instruction: category "or" under the given signature
    /// with the given u_-form disjunct elements (the test_memory bootstrap
    /// rig).
    void buildOrInstruction(gl::WorkInstruction& wi, const char* signature,
                            const std::initializer_list<const char*>& elements) {
        const int32_t mark = wi.elemMark();
        for (const char* element : elements)
            wi.addElement(gl::StrSpan(
                element, static_cast<int32_t>(std::strlen(element))));
        wi.commitEntity(gl::StrSpan("or", 2),
            gl::StrSpan(signature, static_cast<int32_t>(std::strlen(signature))),
            gl::StrSpan(), static_cast<int32_t>(elements.size()), mark);
    }

    /// The or's rules as the disintegrator emits them: a registry or's K
    /// and subset-exclusion rules are products of its or-implication
    /// compacts, which are CHILD statements of the or's key — so the rules
    /// sit one level down, under each compact's key. Visit the key's own
    /// rules, then every child's.
    template <typename Fn>
    void forImplsDeep(const gl::CollectedArena& collected,
                      const std::string& key, Fn fn) {
        collected.forImpls(gl::StrSpan(key), fn);
        collected.forChildren(gl::StrSpan(key), [&](gl::StrSpan child) {
            collected.forImpls(child, fn);
        });
    }
} // namespace

// The two-route or arm, route-neither: with NO product-of-disintegration
// signal and NO admission demand, the cohort PARKS — no cohort rows, no
// release staging, no mail; rejectedMapOrdis gains one entry per
// operator-based product template (the equality disjunct contributes no
// key), value = the clean or statement + the seed level run. A subsequent
// revisitRejectedOrdis on a parked key erases the cohort, resets the
// known-statement registries (the un-know), and re-deposits the statement
// on sameIterationInternalMail. The hypothetical path (allowOrProbe=false)
// neither opens nor parks.
TEST(prover, ordis_park_on_probe_miss_and_revive) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    gl::WorkInstruction wi(&lb);
    buildOrInstruction(wi, "(or98[u_4,u_5])",
                       { "(existence97[u_1,u_4,u_3])", "(=[u_4,u_5])" });

    gl::tests::ProverFixture analyzer("Peano");
    // Artificial isolated registry entry (the test_memory synthetic-seed
    // pattern): the existence-category disjunct must be operator-backed
    // WITHOUT reading the on-disk spontaneous binaries — the unit-test
    // gate runs before any pipeline priming, on a possibly empty
    // files/GL_binaries. Elements use config-level operator cores only;
    // w1 is the single non-u_ witness the product-template route marks.
    gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions)["existence97"] =
        gl::LogicalEntity("existence",
            { "(in[w1,u_1])", "(in2[u_2,w1,u_3])" },
            "(existence97[u_1,u_2,u_3])", 3);
    analyzer.supportingRuleCompiler.compileDisintegrationPackages();
    gl::Memory memory;
    ASSERT_EQ(((memory).nameMap).encode("main"), gl::NameMap::MAIN_ID);
    ((memory).intToBeProved).assignSet(gl::packStatementKey(
        ((memory).nameMap).encode("(freeze_goal[1])"), gl::NameMap::MAIN_ID),
        nullptr, 0);   // no goals, no or branches: keep the cohort opening live
    gl::CollectedArena collected(&lb);
    gl::NewVarStore newVarMap(&lb);
    int seedLv[1] = { 0 };

    // Hypothetical path first: neither open nor park.
    gl::tests::DisintegrationProcessorFixture::disintegrateExprCore2(analyzer.disintegrationProcessor, gl::StrSpan("(or98[u_4,u_5])", 15), wi,
        memory, 0, collected, newVarMap, gl::StrSpan("main", 4),
        /*trackHistoryLocal=*/true, /*allowOrDisintegration=*/false,
        seedLv, 1, /*allowOrProbe=*/false);
    ASSERT_EQ(((((memory).overallHashMemory)).rejectedMapOrdis).count(), 0);

    // Real deposit, no demand: PARK.
    gl::tests::DisintegrationProcessorFixture::disintegrateExprCore2(analyzer.disintegrationProcessor, gl::StrSpan("(or98[u_4,u_5])", 15), wi,
        memory, 0, collected, newVarMap, gl::StrSpan("main", 4),
        /*trackHistoryLocal=*/true, /*allowOrDisintegration=*/false,
        seedLv, 1, /*allowOrProbe=*/true);

    const int32_t cohortId = gl::mintOrCohortId(((memory).lbStateInterner),
        ((memory).lbStateInterner).encode(gl::StrSpan("main", 4)),
        ((memory).lbStateInterner).encode(gl::StrSpan("(or98[4,5])", 11)));
    ASSERT_TRUE(((memory).orDisjunctCount).lookup(cohortId) == 0);
    ASSERT_TRUE(((memory).pendingOrReleases).lookup(cohortId) == 0);
    ASSERT_EQ(gl::decodeInternalMailStatements(
        ((memory).sameIterationInternalMail), ((memory).nameMap)).size(),
        static_cast<std::size_t>(0));

    const int32_t parkedKeys = ((((memory).overallHashMemory)).rejectedMapOrdis).count();
    ASSERT_TRUE(parkedKeys >= 1);
    std::string firstTemplate;
    for (int32_t id = 1; id <= parkedKeys; ++id) {
        const int64_t pk = ((((memory).overallHashMemory)).rejectedMapOrdis).keyAt(id);
        const auto tv = gl::decodeTemplateKey(pk, ((memory).templateInterner),
                                              ((memory).nameMap));
        // Every park key sits at main and no equality-shaped template exists.
        ASSERT_EQ(tv.second, std::string("main"));
        ASSERT_TRUE(tv.first.rfind("(=[", 0) != 0);
        if (id == 1) firstTemplate = tv.first;
        const gl::RejectedOrdisValueSet vals = gl::rejectedOrdisRecordsAt(
            ((((memory).overallHashMemory)).rejectedMapOrdis), pk, ((memory).valueInterner));
        ASSERT_EQ(vals.size(), static_cast<std::size_t>(1));
        ASSERT_EQ(((memory).valueInterner).decode(vals.begin()->orStatement),
                  std::string("(or98[4,5])"));
        ASSERT_TRUE((vals.begin()->levels == std::set<int>{ 0 }));
    }

    // Seed the statement as a KNOWN local statement (the state a real park
    // leaves behind: the or was consumed flat at arrival), then revive and
    // pin the un-know.
    const gl::NameId stmtId = ((memory).nameMap).encode("(or98[4,5])");
    gl::upsertStatementKey(((memory).intKnownStatements),
        gl::packStatementKey(stmtId, gl::NameMap::MAIN_ID),
        /*local=*/true);
    ASSERT_TRUE(((memory).intKnownStatements).find(
        gl::StatementKey{ stmtId, gl::NameMap::MAIN_ID }) != nullptr);

    analyzer.rejectedSourceReplay.revisitRejectedOrdis(gl::StrSpan(firstTemplate),
                                  memory, gl::StrSpan("main", 4));

    // The woken key is erased; the known row is reset; the statement is
    // re-deposited at main with the parked level run.
    {
        int64_t wokenPk = 0;
        ASSERT_TRUE(gl::lookupTemplateKey(((memory).templateInterner),
            ((memory).nameMap), firstTemplate, "main", wokenPk));
        ASSERT_TRUE(((((memory).overallHashMemory)).rejectedMapOrdis).lookup(wokenPk) == 0);
    }
    ASSERT_TRUE(((memory).intKnownStatements).find(
        gl::StatementKey{ stmtId, gl::NameMap::MAIN_ID }) == nullptr);
    const auto mailRows = gl::decodeInternalMailStatements(
        ((memory).sameIterationInternalMail), ((memory).nameMap));
    ASSERT_EQ(mailRows.size(), static_cast<std::size_t>(1));
    ASSERT_TRUE(mailRows[0].first.original == "(or98[4,5])");
    ASSERT_TRUE(mailRows[0].first.validityName == "main");
    ASSERT_EQ(mailRows[0].second, (std::set<int>{ 0 }));
}

// Algebra product-template admission alone leaves a cohort parked, with no
// starter or branch mail. The admission entry remains available to its owner.
TEST(prover, ordis_algebra_admission_alone_keeps_cohort_parked) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    gl::WorkInstruction wi(&lb);
    // Ranking would pick the in3 disjunct (more tokens); the demand admits
    // the in2 disjunct.
    buildOrInstruction(wi, "(or98[u_4,u_5])",
                       { "(in3[u_7,u_8,u_9,4])", "(in2[u_4,u_7,3])" });

    gl::tests::ProverFixture analyzer("Peano");
    gl::Memory memory;
    ASSERT_EQ(((memory).nameMap).encode("main"), gl::NameMap::MAIN_ID);
    ((memory).intToBeProved).assignSet(gl::packStatementKey(
        ((memory).nameMap).encode("(freeze_goal[1])"), gl::NameMap::MAIN_ID),
        nullptr, 0);   // no goals, no or branches: keep the cohort opening live
    gl::ScratchArena& tArena = gl::genScratchArenas().forSlot(
        gl::genScratchArenas().slotCount() - 1);

    // Register the demand: the in2 disjunct's product template (an operator
    // disjunct probes as itself).
    const int64_t demandPk = gl::mintTemplateKey(((memory).templateInterner),
        ((memory).nameMap), "(in2[4,7,3])", "main");
    gl::AdmissionMapValue demand;
    demand.standardMaxAdmissionDepth = 30;
    demand.standardMaxSecondaryNumber = 1;
    demand.ordisOnly = true;   // the probe reads tagged and untagged alike
    gl::insertAdmissionValue(((((memory).overallHashMemory)).admissionMap), demandPk,
                             demand, ((memory).valueInterner), tArena);

    gl::CollectedArena collected(&lb);
    gl::NewVarStore newVarMap(&lb);
    int seedLv[1] = { 0 };
    gl::tests::DisintegrationProcessorFixture::disintegrateExprCore2(analyzer.disintegrationProcessor, gl::StrSpan("(or98[u_4,u_5])", 15), wi,
        memory, 0, collected, newVarMap, gl::StrSpan("main", 4),
        /*trackHistoryLocal=*/true, /*allowOrDisintegration=*/false,
        seedLv, 1, /*allowOrProbe=*/true);

    // The algebra demand does not authorize cohort creation.
    const int32_t cohortId = gl::mintOrCohortId(((memory).lbStateInterner),
        ((memory).lbStateInterner).encode(gl::StrSpan("main", 4)),
        ((memory).lbStateInterner).encode(gl::StrSpan("(or98[4,5])", 11)));
    ASSERT_TRUE(((((memory).overallHashMemory)).rejectedMapOrdis).count() > 0);
    const int32_t cntRow = ((memory).orDisjunctCount).lookup(cohortId);
    ASSERT_TRUE(cntRow == 0);
    ASSERT_TRUE(((memory).pendingOrReleases).lookup(cohortId) == 0);
    ASSERT_TRUE(((memory).orStarterPick).find(cohortId) == nullptr);
    ASSERT_TRUE(((((memory).overallHashMemory)).admissionMap).lookup(demandPk) != 0);

    // No pending cohort means the release drain emits no branch seed.
    analyzer.caseProcessor.drainPendingOrReleases(memory);
    const auto mailRows = gl::decodeInternalMailStatements(
        ((memory).sameIterationInternalMail), ((memory).nameMap));
    ASSERT_EQ(mailRows.size(), static_cast<std::size_t>(0));
    ASSERT_TRUE(((memory).orStarterPick).find(cohortId) == nullptr);
    const int32_t pRow = ((memory).orPendingBranches).lookup(cohortId);
    ASSERT_TRUE(pRow == 0);
}

// Single-exclusion emission (D-269): a
// registry-backed k=3 cohort emits, in ADDITION to the 3 K rules, the 3
// single-exclusion rules !D_i -> or(D_rest) with pre-minted reduced 2-ary
// heads whose instance args map through the parent's signature positions;
// each emitted rule satisfies the install-time park-first detector. A
// k=2 cohort emits K rules only.
TEST(prover, ternary_or_stages_source_without_exclusion_rules) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    gl::WorkInstruction wi(&lb);
    buildOrInstruction(wi, "(or96[u_1,u_2,u_10,u_11])",
                       { "(in[u_1,u_2])", "(in2[u_1,u_10,u_2])",
                         "(in3[u_1,u_2,u_10,u_11])" });

    gl::tests::ProverFixture analyzer("Peano");
    gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions)["or96"] =
        gl::LogicalEntity("or",
            { "(in[u_1,u_2])", "(in2[u_1,u_3,u_2])",
              "(in3[u_1,u_2,u_3,u_4])" },
            "(or96[u_1,u_2,u_3,u_4])", 4);
    // A hand-registered or takes the mint door's K-rule step explicitly
    // (every production mint site does it); the seam adds the rest.
    analyzer.orRuleCompiler.compileOrKRules("or96");
    analyzer.orRuleCompiler.preMintReducedOrs();

    // The three pre-minted reduced operators, keyed by their canonical
    // (renumbered) element lists, in exclusion order.
    const std::string m0a = "(in2[u_1,u_2,u_3])";
    const std::string m0b = "(in3[u_1,u_3,u_2,u_4])";
    gl::StrSpan s0[2] = { gl::StrSpan(m0a), gl::StrSpan(m0b) };
    const std::string* red0 = analyzer.compilationReuse.compiledOrByElements(s0, 2);
    ASSERT_TRUE(red0 != nullptr);
    const std::string m1a = "(in[u_1,u_2])";
    const std::string m1b = "(in3[u_1,u_2,u_3,u_4])";
    gl::StrSpan s1[2] = { gl::StrSpan(m1a), gl::StrSpan(m1b) };
    const std::string* red1 = analyzer.compilationReuse.compiledOrByElements(s1, 2);
    ASSERT_TRUE(red1 != nullptr);
    const std::string m2a = "(in[u_1,u_2])";
    const std::string m2b = "(in2[u_1,u_3,u_2])";
    gl::StrSpan s2[2] = { gl::StrSpan(m2a), gl::StrSpan(m2b) };
    const std::string* red2 = analyzer.compilationReuse.compiledOrByElements(s2, 2);
    ASSERT_TRUE(red2 != nullptr);

    gl::Memory memory;
    ASSERT_EQ(((memory).nameMap).encode("main"), gl::NameMap::MAIN_ID);
    ((memory).intToBeProved).assignSet(gl::packStatementKey(
        ((memory).nameMap).encode("(freeze_goal[1])"), gl::NameMap::MAIN_ID),
        nullptr, 0);   // no goals, no or branches: keep the cohort opening live
    gl::CollectedArena collected(&lb);
    gl::NewVarStore newVarMap(&lb);
    int seedLv[1] = { 0 };
    const std::string orInst = "(or96[u_1,u_2,u_10,u_11])";
    gl::tests::DisintegrationProcessorFixture::disintegrateExprCore2(analyzer.disintegrationProcessor, gl::StrSpan(orInst), wi, memory, 0,
        collected, newVarMap, gl::StrSpan("main", 4),
        /*trackHistoryLocal=*/true, /*allowOrDisintegration=*/false,
        seedLv, 1, /*allowOrProbe=*/false);

    std::set<std::string> impls;
    const std::string key = "(or96[1,2,10,11])";
    forImplsDeep(collected, key,
        [&](gl::StrSpan orig, gl::StrSpan val) {
            ASSERT_TRUE(val.toStdString() == "main");
            impls.insert(orig.toStdString());
        });
    ASSERT_TRUE(impls.empty());
    ASSERT_TRUE(collected.isOrSource(gl::StrSpan(key)));

    // The emitted rules satisfy the install-time detector — the emission /
    // park-first contract (I-184).
    const std::string sePrem = "!(in[u_1,u_2])";
    const std::string seHead = "(" + *red0 + "[u_1,u_10,u_2,u_11])";
    const gl::StrSpan seRun[1] = { gl::StrSpan(sePrem) };
    ASSERT_TRUE(analyzer.hashRuleClassifier.isSubsetExclusionInstall(seRun, 1,
        gl::StrSpan(seHead)));
}

// k = 2: single-exclusion IS the K rule — nothing extra is emitted.
TEST(prover, single_exclusion_not_emitted_for_binary_or) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    gl::WorkInstruction wi(&lb);
    buildOrInstruction(wi, "(or97[u_1,u_2,u_10])",
                       { "(in[u_1,u_2])", "(in2[u_1,u_10,u_2])" });

    gl::tests::ProverFixture analyzer("Peano");
    gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions)["or97"] =
        gl::LogicalEntity("or",
            { "(in[u_1,u_2])", "(in2[u_1,u_3,u_2])" },
            "(or97[u_1,u_2,u_3])", 3);
    analyzer.orRuleCompiler.compileOrKRules("or97");
    analyzer.orRuleCompiler.preMintReducedOrs();

    gl::Memory memory;
    ASSERT_EQ(((memory).nameMap).encode("main"), gl::NameMap::MAIN_ID);
    ((memory).intToBeProved).assignSet(gl::packStatementKey(
        ((memory).nameMap).encode("(freeze_goal[1])"), gl::NameMap::MAIN_ID),
        nullptr, 0);   // no goals, no or branches: keep the cohort opening live
    gl::CollectedArena collected(&lb);
    gl::NewVarStore newVarMap(&lb);
    int seedLv[1] = { 0 };
    const std::string orInst = "(or97[u_1,u_2,u_10])";
    gl::tests::DisintegrationProcessorFixture::disintegrateExprCore2(analyzer.disintegrationProcessor, gl::StrSpan(orInst), wi, memory, 0,
        collected, newVarMap, gl::StrSpan("main", 4),
        /*trackHistoryLocal=*/true, /*allowOrDisintegration=*/false,
        seedLv, 1, /*allowOrProbe=*/false);

    int implCount = 0;
    const std::string key = "(or97[1,2,10])";
    forImplsDeep(collected, key,
        [&](gl::StrSpan orig, gl::StrSpan val) {
            (void)val;
            // No or-compact appears in either rule: both are the K pair
            // over the in/in2 leaves.
            ASSERT_TRUE(orig.toStdString().find("(or") == std::string::npos);
            ++implCount;
        });
    ASSERT_EQ(implCount, 0);
    ASSERT_TRUE(collected.isOrSource(gl::StrSpan(key)));
}

// Subset-exclusion emission at k = 4: in ADDITION to the 4 K rules, the
// 4 single-exclusion (j=1) and 6 double-exclusion (j=2) rules emit —
// 2^4 - 4 - 2 = 10 exclusion rules, excluded index subsets in
// lexicographic order, premises in parent order, reduced heads
// pre-minted. The j=2 rules satisfy the order-free install detector in
// either premise order.
TEST(prover, quaternary_or_stages_source_without_exclusion_rules) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    gl::WorkInstruction wi(&lb);
    buildOrInstruction(wi, "(or98[u_1,u_2,u_10])",
                       { "(ina[u_1])", "(inb[u_2])", "(inc[u_1,u_2])",
                         "(ind[u_2,u_10])" });

    gl::tests::ProverFixture analyzer("Peano");
    gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions)["or98"] =
        gl::LogicalEntity("or",
            { "(ina[u_1])", "(inb[u_2])", "(inc[u_1,u_2])",
              "(ind[u_2,u_3])" },
            "(or98[u_1,u_2,u_3])", 3);
    analyzer.orRuleCompiler.compileOrKRules("or98");
    analyzer.orRuleCompiler.preMintReducedOrs();

    // Two of the pre-minted reduced operators, keyed by their canonical
    // (renumbered) element lists: survivors of excluding {0,1} and {1,3}.
    const std::string r01a = "(inc[u_1,u_2])";
    const std::string r01b = "(ind[u_2,u_3])";
    gl::StrSpan s01[2] = { gl::StrSpan(r01a), gl::StrSpan(r01b) };
    const std::string* red01 = analyzer.compilationReuse.compiledOrByElements(s01, 2);
    ASSERT_TRUE(red01 != nullptr);
    const std::string r13a = "(ina[u_1])";
    const std::string r13b = "(inc[u_1,u_2])";
    gl::StrSpan s13[2] = { gl::StrSpan(r13a), gl::StrSpan(r13b) };
    const std::string* red13 = analyzer.compilationReuse.compiledOrByElements(s13, 2);
    ASSERT_TRUE(red13 != nullptr);

    gl::Memory memory;
    ASSERT_EQ(((memory).nameMap).encode("main"), gl::NameMap::MAIN_ID);
    ((memory).intToBeProved).assignSet(gl::packStatementKey(
        ((memory).nameMap).encode("(freeze_goal[1])"), gl::NameMap::MAIN_ID),
        nullptr, 0);   // no goals, no or branches: keep the cohort opening live
    gl::CollectedArena collected(&lb);
    gl::NewVarStore newVarMap(&lb);
    int seedLv[1] = { 0 };
    const std::string orInst = "(or98[u_1,u_2,u_10])";
    gl::tests::DisintegrationProcessorFixture::disintegrateExprCore2(analyzer.disintegrationProcessor, gl::StrSpan(orInst), wi, memory, 0,
        collected, newVarMap, gl::StrSpan("main", 4),
        /*trackHistoryLocal=*/true, /*allowOrDisintegration=*/false,
        seedLv, 1, /*allowOrProbe=*/false);

    std::set<std::string> impls;
    const std::string key = "(or98[1,2,10])";
    forImplsDeep(collected, key,
        [&](gl::StrSpan orig, gl::StrSpan val) {
            ASSERT_TRUE(val.toStdString() == "main");
            impls.insert(orig.toStdString());
        });
    ASSERT_TRUE(impls.empty());
    ASSERT_TRUE(collected.isOrSource(gl::StrSpan(key)));

    // The install detector accepts the j=2 shape in EITHER premise order
    // (I-184: order-free detection).
    const std::string p0 = "!(ina[u_1])";
    const std::string p1 = "!(inb[u_2])";
    const std::string head01 = "(" + *red01 + "[u_1,u_2,u_10])";
    const gl::StrSpan fwd[2] = { gl::StrSpan(p0), gl::StrSpan(p1) };
    const gl::StrSpan rev[2] = { gl::StrSpan(p1), gl::StrSpan(p0) };
    ASSERT_TRUE(analyzer.hashRuleClassifier.isSubsetExclusionInstall(fwd, 2,
        gl::StrSpan(head01)));
    ASSERT_TRUE(analyzer.hashRuleClassifier.isSubsetExclusionInstall(rev, 2,
        gl::StrSpan(head01)));
}

// ---------------------------------------------------------------------------
// AdmissionMapOrdis2Value family (D-267) — the
// verbatim twin of the RejectedMapOrdisValue family above; same rig, same
// properties.
// ---------------------------------------------------------------------------

// insertAdmissionOrdis2IdsBlob: canonical RMW per record byte on twin maps
// (empty levels, multi-level, duplicate re-demand deduped by the RMW).
TEST(ordis2, ids_blob_rmw_dedups_and_orders) {
    OrdisRig rig;
    gl::TypedColdBlobMap<int64_t, gl::AdmissionMapOrdis2Value> m(&rig.lb, &rig.d);
    gl::ScratchArena& tArena = gl::genScratchArenas().forSlot(
        gl::genScratchArenas().slotCount() - 1);

    // Anti-lex mints: ids ascend while decoded order descends.
    const int32_t rZZ = rig.vi.encode("(>[]!(zz[u_1])(a[u_1]))");
    const int32_t rMM = rig.vi.encode("(>[]!(mm[u_1])(a[u_1]))");
    const int32_t rAA = rig.vi.encode("(>[]!(aa[u_1])(a[u_1]))");

    const int64_t pk = 9;
    const int lvA[] = { 2 };
    const int lvB[] = { 1, 3 };
    gl::insertAdmissionOrdis2IdsBlob(m, pk, rZZ, lvA, 1, rig.vi, tArena);
    gl::insertAdmissionOrdis2IdsBlob(m, pk, rAA, lvB, 2, rig.vi, tArena);
    gl::insertAdmissionOrdis2IdsBlob(m, pk, rMM, nullptr, 0, rig.vi, tArena);
    gl::insertAdmissionOrdis2IdsBlob(m, pk, rZZ, lvA, 1, rig.vi, tArena);  // dup

    const gl::AdmissionOrdis2ValueSet oracle =
        gl::admissionOrdis2RecordsAt(m, pk, rig.vi);
    const int32_t id = m.lookup(pk);
    ASSERT_TRUE(id != 0);
    ASSERT_EQ(m.runLen(id), 3);  // duplicate deduped
    ASSERT_EQ(m.runLen(id), static_cast<int32_t>(oracle.size()));
    int32_t j = 0;
    for (const gl::AdmissionMapOrdis2Value& o : oracle) {
        const gl::AdmissionMapOrdis2Value r = m.recordAt(id, j++);
        ASSERT_EQ(r.sourceImplId, o.sourceImplId);
        ASSERT_TRUE(r.levels == o.levels);
    }
}

// AdmissionOrdis2ValueBlobView: per-field peeks equal the codec's
// deserialize (empty and populated levels).
TEST(ordis2, blob_view_matches_deserialize) {
    const auto check = [](const gl::AdmissionMapOrdis2Value& v) {
        const std::vector<char> blob =
            gl::Codec<gl::AdmissionMapOrdis2Value>::serialize(v);
        const gl::AdmissionMapOrdis2Value back =
            gl::Codec<gl::AdmissionMapOrdis2Value>::deserialize(
                blob.data(), static_cast<int32_t>(blob.size()));
        const gl::AdmissionOrdis2ValueBlobView view(
            blob.data(), static_cast<int32_t>(blob.size()));
        ASSERT_EQ(view.sourceImplId(), back.sourceImplId);
        ASSERT_EQ(view.levelCount(), static_cast<int32_t>(back.levels.size()));
        int32_t i = 0;
        for (const int lv : back.levels)
            ASSERT_EQ(view.levelAt(i++), lv);
    };
    check(gl::AdmissionMapOrdis2Value(42, std::set<int>{ 0, 3, 9 }));
    check(gl::AdmissionMapOrdis2Value(7, std::set<int>{}));
    check(gl::AdmissionMapOrdis2Value(-3, std::set<int>{ 1 }));
}

// Scope-wipe semantics: the wipeHashMem enrollment erases exactly the
// demand keys whose validity half is closed (dead-scope demands must not
// open cohorts later), survivors untouched.
TEST(ordis2, scope_erase_blob_if_drops_closed_validity) {
    gl::tests::ProverFixture ea("Peano");
    gl::Memory m;
    gl::ScratchArena& tArena = gl::genScratchArenas().forSlot(
        gl::genScratchArenas().slotCount() - 1);

    const int64_t pkMain = gl::mintTemplateKey(((m).templateInterner), ((m).nameMap),
        "(strictOrder[1,4,10,12])", "main");
    const int64_t pkSub = gl::mintTemplateKey(((m).templateInterner), ((m).nameMap),
        "(strictOrder[1,4,10,12])", "main_boundary_sub_(h[1])");
    const gl::NameId subVid = gl::Codec<gl::StatementKey>::decode(pkSub).validity;

    const int32_t rMM = ((m).ruleInterner).encode("(>[]!(mm[u_1])(a[u_1]))");
    const int lv[] = { 1 };
    gl::insertAdmissionOrdis2IdsBlob(((((m).overallHashMemory)).admissionMapOrdis2),
        pkMain, rMM, lv, 1, ((m).ruleInterner), tArena);
    gl::insertAdmissionOrdis2IdsBlob(((((m).overallHashMemory)).admissionMapOrdis2),
        pkSub, rMM, lv, 1, ((m).ruleInterner), tArena);

    ((((m).overallHashMemory)).admissionMapOrdis2).eraseBlobIf(
        [subVid](int64_t k) {
            return gl::Codec<gl::StatementKey>::decode(k).validity == subVid;
        });

    ASSERT_TRUE(((((m).overallHashMemory)).admissionMapOrdis2).lookup(pkSub) == 0);
    ASSERT_TRUE(((((m).overallHashMemory)).admissionMapOrdis2).lookup(pkMain) != 0);
}

// applyEquivalenceClassToAdmissionMapOrdis2 — drop-and-rekey: the stale
// demand key leaves, its value blobs land verbatim under the
// canonicalized template at the same validity (the admission-path D-106
// discipline; no mail — a demand is not a statement).
TEST(prover, apply_equi_admission_ordis2_drops_and_rekeys) {
    gl::tests::ProverFixture ea("Peano");
    gl::Memory m;
    ASSERT_EQ(((m).nameMap).encode("main"), gl::NameMap::MAIN_ID);
    gl::ScratchArena& tArena = gl::genScratchArenas().forSlot(
        gl::genScratchArenas().slotCount() - 1);

    const int64_t oldPk = gl::mintTemplateKey(((m).templateInterner), ((m).nameMap),
        "(strictOrder[1,4,it_5_lev_1_2,12])", "main");
    const int32_t srcImpl = ((m).ruleInterner).encode("(>[]b5dummy(=[10,12]))");
    const int lv[] = { 1 };
    gl::insertAdmissionOrdis2IdsBlob(
        ((((m).overallHashMemory)).admissionMapOrdis2), oldPk, srcImpl,
        lv, 1, ((m).ruleInterner), tArena);

    gl::EquivalenceClass cls;
    cls.setMembersFromNames({ "it_0_lev_1_2", "it_5_lev_1_2" }, ((m).nameMap));

    ea.equivalenceAdmissionRewriter.applyEquivalenceClassToAdmissionMapOrdis2(cls, m, gl::StrSpan("main", 4));

    ASSERT_TRUE(((((m).overallHashMemory)).admissionMapOrdis2).lookup(oldPk) == 0);
    int64_t newPk = 0;
    ASSERT_TRUE(gl::lookupTemplateKey(((m).templateInterner), ((m).nameMap),
        "(strictOrder[1,4,it_0_lev_1_2,12])", "main", newPk));
    const gl::AdmissionOrdis2ValueSet vals = gl::admissionOrdis2RecordsAt(
        ((((m).overallHashMemory)).admissionMapOrdis2), newPk, ((m).ruleInterner));
    ASSERT_EQ(vals.size(), static_cast<std::size_t>(1));
    ASSERT_EQ(vals.begin()->sourceImplId, srcImpl);
    ASSERT_TRUE((vals.begin()->levels == std::set<int>{ 1 }));
}

// ---------------------------------------------------------------------------
// rejectedMapOrdis2 (D-267) — the second park
// index; value family REUSED from RejectedMapOrdisValue, keys = eligible
// disjunct clean GROUND texts.
// ---------------------------------------------------------------------------

// The reused codec family works unchanged against the second map instance,
// and the two instances are fully independent.
TEST(rejected_ordis2, reused_family_rmw_on_second_instance) {
    OrdisRig rig;
    gl::Memory m;
    gl::ScratchArena& tArena = gl::genScratchArenas().forSlot(
        gl::genScratchArenas().slotCount() - 1);

    const int64_t pk = gl::mintTemplateKey(((m).templateInterner), ((m).nameMap),
        "(preorder[1,4,10,12])", "main");
    const int32_t stmtId = ((m).valueInterner).encode("(or7[1,4,10,12])");
    const int lv[] = { 0, 2 };
    gl::insertRejectedOrdisIdsBlob(((((m).overallHashMemory)).rejectedMapOrdis2),
        pk, stmtId, lv, 2, ((m).valueInterner), tArena);
    gl::insertRejectedOrdisIdsBlob(((((m).overallHashMemory)).rejectedMapOrdis2),
        pk, stmtId, lv, 2, ((m).valueInterner), tArena);  // dup — RMW dedups

    const gl::RejectedOrdisValueSet vals = gl::rejectedOrdisRecordsAt(
        ((((m).overallHashMemory)).rejectedMapOrdis2), pk, ((m).valueInterner));
    ASSERT_EQ(vals.size(), static_cast<std::size_t>(1));
    ASSERT_EQ(vals.begin()->orStatement, stmtId);
    ASSERT_TRUE((vals.begin()->levels == std::set<int>{ 0, 2 }));
    // Independence: the OLD park index never sees the filing.
    ASSERT_TRUE(((((m).overallHashMemory)).rejectedMapOrdis).lookup(pk) == 0);
}

// Scope-wipe semantics: the wipeHashMem enrollment erases exactly the
// filed keys whose validity half is closed, survivors untouched.
TEST(rejected_ordis2, scope_erase_blob_if_drops_closed_validity) {
    gl::Memory m;
    gl::ScratchArena& tArena = gl::genScratchArenas().forSlot(
        gl::genScratchArenas().slotCount() - 1);

    const int64_t pkMain = gl::mintTemplateKey(((m).templateInterner), ((m).nameMap),
        "(preorder[1,4,10,12])", "main");
    const int64_t pkSub = gl::mintTemplateKey(((m).templateInterner), ((m).nameMap),
        "(preorder[1,4,10,12])", "main_boundary_sub_(h[1])");
    const gl::NameId subVid = gl::Codec<gl::StatementKey>::decode(pkSub).validity;

    const int32_t stmtId = ((m).valueInterner).encode("(or7[1,4,10,12])");
    const int lv[] = { 1 };
    gl::insertRejectedOrdisIdsBlob(((((m).overallHashMemory)).rejectedMapOrdis2),
        pkMain, stmtId, lv, 1, ((m).valueInterner), tArena);
    gl::insertRejectedOrdisIdsBlob(((((m).overallHashMemory)).rejectedMapOrdis2),
        pkSub, stmtId, lv, 1, ((m).valueInterner), tArena);

    ((((m).overallHashMemory)).rejectedMapOrdis2).eraseBlobIf(
        [subVid](int64_t k) {
            return gl::Codec<gl::StatementKey>::decode(k).validity == subVid;
        });

    ASSERT_TRUE(((((m).overallHashMemory)).rejectedMapOrdis2).lookup(pkSub) == 0);
    ASSERT_TRUE(((((m).overallHashMemory)).rejectedMapOrdis2).lookup(pkMain) != 0);
}

// revisitRejectedOrdis2: the wake erases exactly the probed key, resets the
// known-statement registries (the un-know), and re-deposits the statement
// on sameIterationInternalMail with the filed level run. Entries under
// OTHER ordis2 keys and in the OLD rejectedMapOrdis survive untouched.
TEST(prover, revisit_rejected_ordis2_wakes_erases_and_unknows) {
    gl::tests::ProverFixture analyzer("Peano");
    gl::Memory m;
    ASSERT_EQ(((m).nameMap).encode("main"), gl::NameMap::MAIN_ID);
    gl::ScratchArena& tArena = gl::genScratchArenas().forSlot(
        gl::genScratchArenas().slotCount() - 1);

    const int32_t stmtId = ((m).valueInterner).encode("(or7[1,4,10,12])");
    const int lv[] = { 0 };
    const int64_t pkA = gl::mintTemplateKey(((m).templateInterner), ((m).nameMap),
        "(preorder[1,4,10,12])", "main");
    const int64_t pkB = gl::mintTemplateKey(((m).templateInterner), ((m).nameMap),
        "(preorder[1,4,12,10])", "main");
    gl::insertRejectedOrdisIdsBlob(((((m).overallHashMemory)).rejectedMapOrdis2),
        pkA, stmtId, lv, 1, ((m).valueInterner), tArena);
    gl::insertRejectedOrdisIdsBlob(((((m).overallHashMemory)).rejectedMapOrdis2),
        pkB, stmtId, lv, 1, ((m).valueInterner), tArena);
    // An old-map park under a product-template key must not be touched.
    const int64_t pkOld = gl::mintTemplateKey(((m).templateInterner), ((m).nameMap),
        "(in3[10,marker,12,4])", "main");
    gl::insertRejectedOrdisIdsBlob(((((m).overallHashMemory)).rejectedMapOrdis),
        pkOld, stmtId, lv, 1, ((m).valueInterner), tArena);

    // Seed the statement as a KNOWN local statement (the state a real park
    // leaves behind), then wake key A and pin the un-know.
    const gl::NameId sId = ((m).nameMap).encode("(or7[1,4,10,12])");
    gl::upsertStatementKey(((m).intKnownStatements),
        gl::packStatementKey(sId, gl::NameMap::MAIN_ID), /*local=*/true);

    analyzer.rejectedSourceReplay.revisitRejectedOrdis2(gl::StrSpan("(preorder[1,4,10,12])", 21),
                                   m, gl::StrSpan("main", 4));

    ASSERT_TRUE(((((m).overallHashMemory)).rejectedMapOrdis2).lookup(pkA) == 0);
    ASSERT_TRUE(((((m).overallHashMemory)).rejectedMapOrdis2).lookup(pkB) != 0);
    ASSERT_TRUE(((((m).overallHashMemory)).rejectedMapOrdis).lookup(pkOld) != 0);
    ASSERT_TRUE(((m).intKnownStatements).find(
        gl::StatementKey{ sId, gl::NameMap::MAIN_ID }) == nullptr);
    const auto mailRows = gl::decodeInternalMailStatements(
        ((m).sameIterationInternalMail), ((m).nameMap));
    ASSERT_EQ(mailRows.size(), static_cast<std::size_t>(1));
    ASSERT_TRUE(mailRows[0].first.original == "(or7[1,4,10,12])");
    ASSERT_TRUE(mailRows[0].first.validityName == "main");
    ASSERT_EQ(mailRows[0].second, (std::set<int>{ 0 }));
    // A wake on a never-filed text is a defined miss (no crash, no mail).
    analyzer.rejectedSourceReplay.revisitRejectedOrdis2(gl::StrSpan("(neverfiled[1])", 15),
                                   m, gl::StrSpan("main", 4));
    ASSERT_EQ(gl::decodeInternalMailStatements(
        ((m).sameIterationInternalMail), ((m).nameMap)).size(),
        static_cast<std::size_t>(1));
}

// applyEquivalenceClassToRejectedMapOrdis2 — I-37 drop-and-mail on the
// second park index: stale key dropped, NO direct re-key, the rewritten or
// statement mailed onto sameIterationInternalMail.
TEST(prover, apply_equi_rejected_ordis2_drops_and_mails) {
    gl::tests::ProverFixture ea("Peano");
    gl::Memory m;
    gl::ScratchArena& tArena = gl::genScratchArenas().forSlot(
        gl::genScratchArenas().slotCount() - 1);

    const int64_t oldPk = gl::mintTemplateKey(((m).templateInterner), ((m).nameMap),
        "(preorder[1,4,it_5_lev_1_2,12])", "main");
    const int32_t stmtId =
        ((m).valueInterner).encode("(or7[1,4,it_5_lev_1_2,12])");
    const int lv[] = { 0, 4 };
    gl::insertRejectedOrdisValue(((((m).overallHashMemory)).rejectedMapOrdis2), oldPk,
        gl::RejectedMapOrdisValue(stmtId, lv, 2), ((m).valueInterner), tArena);

    gl::EquivalenceClass cls;
    cls.setMembersFromNames({ "it_0_lev_1_2", "it_5_lev_1_2" }, ((m).nameMap));

    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::RejectedValidityBuckets buckets(&tArena, &dirty);
    seedFixtureClassEvidence(m,cls,gl::StrSpan("main"));
    ea.equivalenceAdmissionRewriter.applyEquivalenceClassToRejectedMapOrdis2(cls, m, gl::StrSpan("main", 4), buckets);

    // Old key dropped; NO direct re-key under the canonical text.
    ASSERT_TRUE(((((m).overallHashMemory)).rejectedMapOrdis2).lookup(oldPk) == 0);
    int64_t canonPk = 0;
    if (gl::lookupTemplateKey(((m).templateInterner), ((m).nameMap),
            "(preorder[1,4,it_0_lev_1_2,12])", "main", canonPk)) {
        ASSERT_TRUE(((((m).overallHashMemory)).rejectedMapOrdis2).lookup(canonPk) == 0);
    }

    // Rewritten or statement mailed onto sameIterationInternalMail at "main".
    bool foundMail = false;
    const gl::Mail mailSnap = gl::makeHeapMail(((m).sameIterationInternalMail),
                                               ((m).nameMap), ((m).originInterner));
    for (const auto& pr : mailSnap.statements) {
        if (pr.first.original == "(or7[1,4,it_0_lev_1_2,12])"
            && pr.first.validityName == "main") {
            foundMail = true;
        }
    }
    ASSERT_TRUE(foundMail);
}

// ---------------------------------------------------------------------------
// K4 — park-side producer: ordis2KeyEligible + the dual filing
// (D-267).
// ---------------------------------------------------------------------------

// ordis2KeyEligible matrix: compiled non-atomic 4-arg compacts pass (both
// polarities), equalities / atomic facts / short arities / anchors fail.
TEST(ordis2, key_eligible_matrix) {
    gl::tests::ProverFixture analyzer("Peano");
    // Synthetic isolated registry entries (the test_memory seed pattern —
    // no on-disk binaries at unit-test time).
    gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions)["preorder97"] =
        gl::LogicalEntity("existence",
            { "(in[w1,u_1])", "(in3[u_3,w1,u_4,u_2])" },
            "(preorder97[u_1,u_2,u_3,u_4])", 4);
    gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions)["short97"] =
        gl::LogicalEntity("existence",
            { "(in[w1,u_1])" },
            "(short97[u_1,u_2,u_3])", 3);
    gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions)["atomic97"] =
        gl::LogicalEntity("atomic", {}, "(atomic97[u_1,u_2,u_3,u_4])", 4);
    analyzer.supportingRuleCompiler.compileDisintegrationPackages();

    // Eligible: compiled non-atomic, 4 args, both polarities.
    ASSERT_TRUE(analyzer.admissionPolicy.ordis2KeyEligible(
        gl::StrSpan("(preorder97[1,2,10,12])", 23)));
    ASSERT_TRUE(analyzer.admissionPolicy.ordis2KeyEligible(
        gl::StrSpan("!(preorder97[1,2,10,12])", 24)));
    // Equality: no compiled entity.
    ASSERT_FALSE(analyzer.admissionPolicy.ordis2KeyEligible(gl::StrSpan("(=[10,12])", 10)));
    // Atomic category refuses even at arity 4.
    ASSERT_FALSE(analyzer.admissionPolicy.ordis2KeyEligible(
        gl::StrSpan("(atomic97[1,2,10,12])", 21)));
    // Arity below kOrdis2DemandMinArity refuses.
    ASSERT_FALSE(analyzer.admissionPolicy.ordis2KeyEligible(
        gl::StrSpan("(short97[1,2,10])", 17)));
    // Anchors never enter the key language.
    ASSERT_FALSE(analyzer.admissionPolicy.ordis2KeyEligible(
        gl::StrSpan("(AnchorFTA[1,2,3,4,5,6,7,8])", 28)));
}

// The park files the cohort in BOTH indexes: the old map under product
// templates, the new map under each ELIGIBLE disjunct's clean ground text;
// the equality disjunct is filed in neither.
TEST(prover, ordis2_park_files_disjunct_texts) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    gl::WorkInstruction wi(&lb);
    buildOrInstruction(wi, "(or96[u_4,u_5])",
                       { "(preorder96[u_1,u_2,u_4,u_5])", "(=[u_4,u_5])" });

    gl::tests::ProverFixture analyzer("Peano");
    gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions)["preorder96"] =
        gl::LogicalEntity("existence",
            { "(in[w1,u_1])", "(in3[u_4,w1,u_5,u_2])" },
            "(preorder96[u_1,u_2,u_4,u_5])", 4);
    analyzer.supportingRuleCompiler.compileDisintegrationPackages();
    gl::Memory memory;
    ASSERT_EQ(((memory).nameMap).encode("main"), gl::NameMap::MAIN_ID);
    ((memory).intToBeProved).assignSet(gl::packStatementKey(
        ((memory).nameMap).encode("(freeze_goal[1])"), gl::NameMap::MAIN_ID),
        nullptr, 0);   // no goals, no or branches: keep the cohort opening live
    gl::CollectedArena collected(&lb);
    gl::NewVarStore newVarMap(&lb);
    int seedLv[1] = { 0 };

    // Real deposit, no route: PARK — dual filing.
    gl::tests::DisintegrationProcessorFixture::disintegrateExprCore2(analyzer.disintegrationProcessor, gl::StrSpan("(or96[u_4,u_5])", 15), wi,
        memory, 0, collected, newVarMap, gl::StrSpan("main", 4),
        /*trackHistoryLocal=*/true, /*allowOrDisintegration=*/false,
        seedLv, 1, /*allowOrProbe=*/true);

    // Old index: at least one product-template key (the witness-marker
    // body fact), none of them the disjunct text itself is required —
    // discipline unchanged (I-178).
    ASSERT_TRUE(((((memory).overallHashMemory)).rejectedMapOrdis).count() >= 1);

    // New index: exactly ONE key — the eligible disjunct's clean ground
    // text; the equality disjunct contributes nothing.
    ASSERT_EQ(((((memory).overallHashMemory)).rejectedMapOrdis2).count(), 1);
    int64_t filedPk = 0;
    ASSERT_TRUE(gl::lookupTemplateKey(((memory).templateInterner),
        ((memory).nameMap), "(preorder96[1,2,4,5])", "main", filedPk));
    const gl::RejectedOrdisValueSet vals = gl::rejectedOrdisRecordsAt(
        ((((memory).overallHashMemory)).rejectedMapOrdis2), filedPk,
        ((memory).valueInterner));
    ASSERT_EQ(vals.size(), static_cast<std::size_t>(1));
    ASSERT_EQ(((memory).valueInterner).decode(vals.begin()->orStatement),
              std::string("(or96[4,5])"));
    ASSERT_TRUE((vals.begin()->levels == std::set<int>{ 0 }));
}

// ---------------------------------------------------------------------------
// K5 — demand runtime (D-267): route (c), the
// drain, and the two arrival orders.
// ---------------------------------------------------------------------------

// Demand-first order: a standing demand entry opens the cohort at deposit
// time — no park, full leaf count registered, the demand-matched leaf is
// the release starter, and the demand remains available to other sources.
TEST(prover, ordis2_route_c_opens_on_demand_and_preserves_it) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    gl::WorkInstruction wi(&lb);
    buildOrInstruction(wi, "(or95[u_4,u_5])",
                       { "(preorder95[u_1,u_2,u_4,u_5])", "(=[u_4,u_5])" });

    gl::tests::ProverFixture analyzer("Peano");
    gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions)["preorder95"] =
        gl::LogicalEntity("existence",
            { "(in[w1,u_1])", "(in3[u_4,w1,u_5,u_2])" },
            "(preorder95[u_1,u_2,u_4,u_5])", 4);
    analyzer.supportingRuleCompiler.compileDisintegrationPackages();
    gl::Memory memory;
    ASSERT_EQ(((memory).nameMap).encode("main"), gl::NameMap::MAIN_ID);
    ((memory).intToBeProved).assignSet(gl::packStatementKey(
        ((memory).nameMap).encode("(freeze_goal[1])"), gl::NameMap::MAIN_ID),
        nullptr, 0);   // no goals, no or branches: keep the cohort opening live
    gl::CollectedArena collected(&lb);
    gl::NewVarStore newVarMap(&lb);
    gl::ScratchArena& tArena = gl::genScratchArenas().forSlot(
        gl::genScratchArenas().slotCount() - 1);
    int seedLv[1] = { 0 };

    // The GROUND demand arrives FIRST (the drain's map insert, done
    // directly here).
    const int64_t demandPk = gl::mintTemplateKey(((memory).templateInterner),
        ((memory).nameMap), "(preorder95[1,2,4,5])", "main");
    const int32_t srcImpl = ((memory).ruleInterner).encode("(>[]dummy(d[1]))");
    const int dlv[] = { 0 };
    gl::insertAdmissionOrdis2IdsBlob(
        ((((memory).overallHashMemory)).admissionMapOrdis2), demandPk, srcImpl,
        dlv, 1, ((memory).ruleInterner), tArena);

    // Deposit: route (c) hits -> the cohort OPENS without ever parking.
    gl::tests::DisintegrationProcessorFixture::disintegrateExprCore2(analyzer.disintegrationProcessor, gl::StrSpan("(or95[u_4,u_5])", 15), wi,
        memory, 0, collected, newVarMap, gl::StrSpan("main", 4),
        /*trackHistoryLocal=*/true, /*allowOrDisintegration=*/false,
        seedLv, 1, /*allowOrProbe=*/true);

    const int32_t cohortId = gl::mintOrCohortId(((memory).lbStateInterner),
        ((memory).lbStateInterner).encode(gl::StrSpan("main", 4)),
        ((memory).lbStateInterner).encode(gl::StrSpan("(or95[4,5])", 11)));
    ASSERT_TRUE(((memory).orDisjunctCount).lookup(cohortId) != 0);
    ASSERT_TRUE(((memory).pendingOrReleases).lookup(cohortId) != 0);
    // An ordinary rule's demand: no I-328 exemption.
    ASSERT_TRUE(memory.lbMemory.hypothesisOrCohorts.lookup(cohortId) == 0);
    // No park in either index (demand-first: opened at deposit).
    ASSERT_EQ(((((memory).overallHashMemory)).rejectedMapOrdis2).count(), 0);
    // Starter = the demand-matched disjunct (wrapped payload form).
    {
        const int32_t* starter = ((memory).orStarterPick).find(cohortId);
        ASSERT_TRUE(starter != nullptr);
        ASSERT_TRUE(((memory).lbStateInterner).decode(*starter)
                    == std::string("((preorder95[1,2,4,5]))"));
    }
    // A case split does not establish the demanded fact.
    ASSERT_TRUE(((((memory).overallHashMemory)).admissionMapOrdis2).lookup(demandPk)
                != 0);
}

// Every source sharing a missing leaf can open, including sources already
// queued by the drain when a fresh source reaches the consumer first.
TEST(prover, ordis2_shared_demand_opens_all_sources_in_both_arrival_orders) {
    for (bool parkFirst : { false, true }) {
        for (bool reverseQueued : { false, true }) {
            gl::GlobalMemoryManager g;
            g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
            gl::LbArena lb(&g);
            gl::WorkInstruction first(&lb), second(&lb), fresh(&lb);
            buildOrInstruction(first, "(or95[u_4,u_5])",
                { "(preorder95[u_1,u_2,u_4,u_5])", "(=[u_4,u_5])" });
            buildOrInstruction(second, "(or96[u_4,u_5])",
                { "(preorder95[u_1,u_2,u_4,u_5])", "!(=[u_4,u_5])" });
            buildOrInstruction(fresh, "(or97[u_4,u_5])",
                { "(preorder95[u_1,u_2,u_4,u_5])", "(in[u_4,u_5])" });
            gl::WorkInstruction* instructions[] = { &first, &second, &fresh };
            const char* signatures[] = {
                "(or95[u_4,u_5])", "(or96[u_4,u_5])", "(or97[u_4,u_5])" };
            const char* ground[] = { "(or95[4,5])", "(or96[4,5])", "(or97[4,5])" };

            gl::tests::ProverFixture analyzer("Peano");
            gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions)["preorder95"] = gl::LogicalEntity(
                "existence", { "(in[w1,u_1])", "(in3[u_4,w1,u_5,u_2])" },
                "(preorder95[u_1,u_2,u_4,u_5])", 4);
            gl::Memory memory;
            ASSERT_EQ(((memory).nameMap).encode("main"), gl::NameMap::MAIN_ID);
            ((memory).intToBeProved).assignSet(gl::packStatementKey(
                ((memory).nameMap).encode("(freeze_goal[1])"), gl::NameMap::MAIN_ID),
                nullptr, 0);
            gl::CollectedArena collected(&lb);
            gl::NewVarStore newVarMap(&lb);
            const int levels[] = { 0 };
            const auto deposit = [&](int index) {
                gl::tests::DisintegrationProcessorFixture::disintegrateExprCore2(analyzer.disintegrationProcessor, gl::StrSpan(signatures[index],
                    static_cast<int32_t>(std::strlen(signatures[index]))),
                    *instructions[index], memory, 0, collected, newVarMap,
                    gl::StrSpan("main", 4), true, false, levels, 1, true);
            };
            if (parkFirst) {
                deposit(0);
                deposit(1);
                ASSERT_EQ(((memory).orDisjunctCount).count(), 0);
            }

            gl::SealedPageSet pages;
            pages.bind(&gl::staticMemory());
            const auto seal = [&pages](const char* text) {
                return gl::SealedString::copyFrom(pages, text,
                    static_cast<int32_t>(std::strlen(text)));
            };
            gl::Ordis2DemandRecord record;
            record.expression = seal("(preorder95[1,2,4,5])");
            record.validityName = seal("main");
            record.sourceImplId = ((memory).ruleInterner).encode("(>[]dummy(d[1]))");
            record.levels = gl::SealedSpan<int>::copyFrom(pages, levels, 1);
            ((memory).admissionKeysOrdis2).push_back(record);
            analyzer.admissionMemory.drainAdmissionKeysOrdis2(memory);
            ((memory).admissionKeysOrdis2).clear();
            pages.seal();
            pages.freePages();
            ASSERT_EQ(gl::decodeInternalMailStatements(
                ((memory).sameIterationInternalMail), ((memory).nameMap)).size(),
                static_cast<std::size_t>(parkFirst ? 2 : 0));

            int64_t demand = 0;
            ASSERT_TRUE(gl::lookupTemplateKey(((memory).templateInterner),
                ((memory).nameMap), "(preorder95[1,2,4,5])", "main", demand));
            deposit(2); // A fresh source arrives before the queued sources.
            deposit(reverseQueued ? 1 : 0);
            deposit(reverseQueued ? 0 : 1);
            ASSERT_TRUE(((((memory).overallHashMemory)).admissionMapOrdis2).lookup(demand) != 0);
            ASSERT_EQ(((((memory).overallHashMemory)).rejectedMapOrdis2).count(), 0);
            ASSERT_EQ(((memory).orDisjunctCount).count(), 3);

            // Repeated consumption cannot reset or duplicate any cohort.
            for (int index = 0; index < 3; ++index) {
                deposit(index);
                const int32_t cohort = gl::mintOrCohortId(((memory).lbStateInterner),
                    ((memory).lbStateInterner).encode(gl::StrSpan("main", 4)),
                    ((memory).lbStateInterner).encode(gl::StrSpan(ground[index],
                        static_cast<int32_t>(std::strlen(ground[index])))));
                const int32_t* count = ((memory).orDisjunctCount).find(cohort);
                ASSERT_TRUE(count != nullptr);
                ASSERT_EQ(*count, 2);
                const int32_t pending = ((memory).orPendingBranches).lookup(cohort);
                ASSERT_TRUE(pending != 0);
                ASSERT_EQ(((memory).orPendingBranches).runLen(pending), 2);
                const int32_t* starter = ((memory).orStarterPick).find(cohort);
                ASSERT_TRUE(starter != nullptr);
                ASSERT_EQ(((memory).lbStateInterner).decode(*starter),
                    std::string("((preorder95[1,2,4,5]))"));
            }
            ASSERT_EQ(((memory).orDisjunctCount).count(), 3);
            ASSERT_EQ(((memory).pendingOrReleases).count(), 3);
            ASSERT_TRUE(((((memory).overallHashMemory)).admissionMapOrdis2).lookup(demand) != 0);
        }
    }
}

// drainAdmissionKeysOrdis2: mints the packed demand key, RMW-inserts the
// {sourceImplId, levels} record, and wakes rejectedMapOrdis2 under the SAME
// key — the filed statement is un-known and re-deposited on internal mail.
// The OLD rejectedMapOrdis is NEVER probed (an entry under the same text
// survives untouched).
TEST(prover, drain_admission_keys_ordis2_mints_and_wakes) {
    gl::tests::ProverFixture ea("Peano");
    gl::Memory m;
    ASSERT_EQ(((m).nameMap).encode("main"), gl::NameMap::MAIN_ID);
    gl::ScratchArena& tArena = gl::genScratchArenas().forSlot(
        gl::genScratchArenas().slotCount() - 1);

    // A parked or filed in the NEW index under the text the demand will
    // name — and in the OLD index under the SAME text (an
    // operator-application disjunct's product template IS its text), to
    // pin the no-cross-wake contract.
    const int64_t parkPk = gl::mintTemplateKey(((m).templateInterner), ((m).nameMap),
        "(strictOrder[1,4,10,12])", "main");
    const int32_t stmtId = ((m).valueInterner).encode("(or90[1,4,10,12])");
    const int plv[] = { 2 };
    gl::insertRejectedOrdisIdsBlob(((((m).overallHashMemory)).rejectedMapOrdis2),
        parkPk, stmtId, plv, 1, ((m).valueInterner), tArena);
    gl::insertRejectedOrdisIdsBlob(((((m).overallHashMemory)).rejectedMapOrdis),
        parkPk, stmtId, plv, 1, ((m).valueInterner), tArena);

    // Seed the statement as KNOWN (the state a real park leaves behind).
    const gl::NameId sId = ((m).nameMap).encode("(or90[1,4,10,12])");
    gl::upsertStatementKey(((m).intKnownStatements),
        gl::packStatementKey(sId, gl::NameMap::MAIN_ID), /*local=*/true);

    // Stage one demand record the way applyFiringRecords does.
    gl::SealedPageSet pages;
    pages.bind(&gl::staticMemory());
    const auto s = [&pages](const std::string& v) {
        return gl::SealedString::copyFrom(pages, v.data(),
                                          static_cast<int32_t>(v.size()));
    };
    const int32_t srcImpl = ((m).ruleInterner).encode("(>[]b5dummy(=[10,12]))");
    const int lvArr[2] = { 0, 3 };
    gl::Ordis2DemandRecord rec;
    rec.expression = s("(strictOrder[1,4,10,12])");
    rec.validityName = s("main");
    rec.sourceImplId = srcImpl;
    rec.levels = gl::SealedSpan<int>::copyFrom(pages, lvArr, 2);
    ((m).admissionKeysOrdis2).push_back(rec);

    ea.admissionMemory.drainAdmissionKeysOrdis2(m);

    // The demand entry landed with the staged payload.
    int64_t pk = 0;
    ASSERT_TRUE(gl::lookupTemplateKey(((m).templateInterner), ((m).nameMap),
        "(strictOrder[1,4,10,12])", "main", pk));
    const gl::AdmissionOrdis2ValueSet vals = gl::admissionOrdis2RecordsAt(
        ((((m).overallHashMemory)).admissionMapOrdis2), pk, ((m).ruleInterner));
    ASSERT_EQ(vals.size(), static_cast<std::size_t>(1));
    ASSERT_EQ(vals.begin()->sourceImplId, srcImpl);
    ASSERT_TRUE((vals.begin()->levels == std::set<int>{ 0, 3 }));

    // The wake hit the NEW index only: its key erased, the OLD index's
    // entry under the same text intact, the known row reset, the statement
    // re-deposited with the filed levels.
    ASSERT_TRUE(((((m).overallHashMemory)).rejectedMapOrdis2).lookup(parkPk) == 0);
    ASSERT_TRUE(((((m).overallHashMemory)).rejectedMapOrdis).lookup(parkPk) != 0);
    ASSERT_TRUE(((m).intKnownStatements).find(
        gl::StatementKey{ sId, gl::NameMap::MAIN_ID }) == nullptr);
    const auto mailRows = gl::decodeInternalMailStatements(
        ((m).sameIterationInternalMail), ((m).nameMap));
    ASSERT_EQ(mailRows.size(), static_cast<std::size_t>(1));
    ASSERT_TRUE(mailRows[0].first.original == "(or90[1,4,10,12])");
    ASSERT_TRUE(mailRows[0].first.validityName == "main");
    ASSERT_EQ(mailRows[0].second, (std::set<int>{ 2 }));

    ((m).admissionKeysOrdis2).clear();
    pages.seal();
    pages.freePages();
}

// Or-first order end-to-end: real deposit parks (dual filing), the drain
// wakes the filed head onto internal mail, and the re-deposit (the absorb's
// re-run, done directly here) opens via route (c), retaining the demand.
TEST(prover, ordis2_pair_or_first_park_wake_open) {
    for (bool contradiction : { false, true }) {

        gl::GlobalMemoryManager g;
        g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
        gl::LbArena lb(&g);
        gl::WorkInstruction wi(&lb);
        buildOrInstruction(wi, "(or94[u_4,u_5])",
                           { "(preorder94[u_1,u_2,u_4,u_5])", "(=[u_4,u_5])" });

        gl::tests::ProverFixture analyzer("Peano");
        gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions)["preorder94"] =
            gl::LogicalEntity("existence",
                { "(in[w1,u_1])", "(in3[u_4,w1,u_5,u_2])" },
                "(preorder94[u_1,u_2,u_4,u_5])", 4);
        analyzer.supportingRuleCompiler.compileDisintegrationPackages();
        gl::Memory memory;
        ASSERT_EQ(((memory).nameMap).encode("main"), gl::NameMap::MAIN_ID);
        ((memory).isActive) = true;
        ((memory).primedForContradiction) = contradiction;
        if (!contradiction) {
            ((memory).intToBeProved).assignSet(gl::packStatementKey(
                ((memory).nameMap).encode("(freeze_goal[1])"), gl::NameMap::MAIN_ID),
                nullptr, 0);
        }
        gl::CollectedArena collected(&lb);
        gl::NewVarStore newVarMap(&lb);
        int seedLv[1] = { 0 };

        // 1. Or-first: the deposit parks (no route), dual-filed.
        gl::tests::DisintegrationProcessorFixture::disintegrateExprCore2(analyzer.disintegrationProcessor, gl::StrSpan("(or94[u_4,u_5])", 15), wi,
            memory, 0, collected, newVarMap, gl::StrSpan("main", 4),
            /*trackHistoryLocal=*/true, /*allowOrDisintegration=*/false,
            seedLv, 1, /*allowOrProbe=*/true);
        int64_t filedPk = 0;
        ASSERT_TRUE(gl::lookupTemplateKey(((memory).templateInterner),
            ((memory).nameMap), "(preorder94[1,2,4,5])", "main", filedPk));
        ASSERT_TRUE(((((memory).overallHashMemory)).rejectedMapOrdis2).lookup(filedPk)
                    != 0);

        // 2. The demand fires later; its drain record wakes the filed head.
        gl::SealedPageSet pages;
        pages.bind(&gl::staticMemory());
        const auto s = [&pages](const std::string& v) {
            return gl::SealedString::copyFrom(pages, v.data(),
                                              static_cast<int32_t>(v.size()));
        };
        gl::Ordis2DemandRecord rec;
        rec.expression = s("(preorder94[1,2,4,5])");
        rec.validityName = s("main");
        rec.sourceImplId = ((memory).ruleInterner).encode("(>[]b5dummy(=[4,5]))");
        const int dl[1] = { 0 };
        rec.levels = gl::SealedSpan<int>::copyFrom(pages, dl, 1);
        ((memory).admissionKeysOrdis2).push_back(rec);
        analyzer.admissionMemory.drainAdmissionKeysOrdis2(memory);
        ((memory).admissionKeysOrdis2).clear();
        pages.seal();
        pages.freePages();

        // Wake evidence: ordis2 key erased, statement re-deposited on mail,
        // demand entry standing.
        ASSERT_TRUE(((((memory).overallHashMemory)).rejectedMapOrdis2).lookup(filedPk)
                    == 0);
        ASSERT_EQ(gl::decodeInternalMailStatements(
            ((memory).sameIterationInternalMail), ((memory).nameMap)).size(),
            static_cast<std::size_t>(1));

        // 3. The absorb re-runs the opener (done directly): route (c) hits the
        // standing demand -> OPEN, starter = the demand-matched leaf, demand
        // retained for other eligible sources.
        gl::tests::DisintegrationProcessorFixture::disintegrateExprCore2(analyzer.disintegrationProcessor, gl::StrSpan("(or94[u_4,u_5])", 15), wi,
            memory, 0, collected, newVarMap, gl::StrSpan("main", 4),
            /*trackHistoryLocal=*/true, /*allowOrDisintegration=*/false,
            seedLv, 1, /*allowOrProbe=*/true);
        const int32_t cohortId = gl::mintOrCohortId(((memory).lbStateInterner),
            ((memory).lbStateInterner).encode(gl::StrSpan("main", 4)),
            ((memory).lbStateInterner).encode(gl::StrSpan("(or94[4,5])", 11)));
        ASSERT_TRUE(((memory).orDisjunctCount).lookup(cohortId) != 0);
        {
            const int32_t* starter = ((memory).orStarterPick).find(cohortId);
            ASSERT_TRUE(starter != nullptr);
            ASSERT_TRUE(((memory).lbStateInterner).decode(*starter)
                        == std::string("((preorder94[1,2,4,5]))"));
        }
        ASSERT_TRUE(((((memory).overallHashMemory)).admissionMapOrdis2).lookup(filedPk)
                    != 0);
        analyzer.caseProcessor.drainPendingOrReleases(memory);
        const int32_t pending = ((memory).orPendingBranches).lookup(cohortId);
        ASSERT_TRUE(pending != 0);
        ASSERT_EQ(((memory).orPendingBranches).runLen(pending), 1);
    }
}

// ---------------------------------------------------------------------------
// K6 — the install route (D-267).
// ---------------------------------------------------------------------------

// ordis2DemandSlotQualifies — the sharpened matrix: subset over non-anchor
// others, key-language filters via ordis2KeyEligible; anchors never
// qualify, anchor-only-bound args reject (the A14 shape mints no demand —
// the maintainer's 2026-08-10 filter decision).
TEST(ordis2, slot_qualification_matrix) {
    gl::tests::ProverFixture ea("Peano");
    gl::CompiledDefinitionFixture::compiledExpressions(ea.definitions)["zzcomp"] =
        gl::LogicalEntity("existence",
            { "(in[w1,u_1])", "(in2[u_2,w1,u_3])" },
            "(zzcomp[u_1,u_2,u_3,u_4])", 4);
    gl::CompiledDefinitionFixture::compiledExpressions(ea.definitions)["zzthree"] =
        gl::LogicalEntity("existence", { "(in[w1,u_1])" },
            "(zzthree[u_1,u_2,u_3])", 3);
    ea.supportingRuleCompiler.compileDisintegrationPackages();

    const std::string anchor = "(AnchorPeano[u_1,u_2,u_3,u_4,u_5,u_6])";
    const std::string b1 = "(in3[u_1,u_4,u_10,u_5])";
    const std::string b2 = "(in3[u_12,u_9,u_11,u_5])";
    const std::string b3 = "(in2[u_10,u_12,u_5])";
    const std::string slotOk = "(zzcomp[u_1,u_4,u_10,u_12])";

    {   // Qualifies: all four args bound by the non-anchor binders, and
        // the rule reaches kOrdis2DemandMinPremises non-anchor premises
        // (slot included).
        const gl::StrSpan key[5] = { gl::StrSpan(anchor), gl::StrSpan(b1),
                                     gl::StrSpan(b2), gl::StrSpan(b3),
                                     gl::StrSpan(slotOk) };
        ASSERT_TRUE(ea.admissionPolicy.ordis2DemandSlotQualifies(key, 5, 4));
        // The anchor premise itself never qualifies.
        ASSERT_FALSE(ea.admissionPolicy.ordis2DemandSlotQualifies(key, 5, 0));
        // An atomic function-application premise (in3) never qualifies,
        // even at arity 4.
        ASSERT_FALSE(ea.admissionPolicy.ordis2DemandSlotQualifies(key, 5, 1));
    }
    {   // Below kOrdis2DemandMinPremises non-anchor premises (the former
        // 3-premise shape): rejected — only premise-rich rules mint
        // demand (maintainer-set minimum, 2026-08-10).
        const gl::StrSpan key[4] = { gl::StrSpan(anchor), gl::StrSpan(b1),
                                     gl::StrSpan(b2), gl::StrSpan(slotOk) };
        ASSERT_FALSE(ea.admissionPolicy.ordis2DemandSlotQualifies(key, 4, 3));
    }
    {   // Polarity-transparent: the NEGATED compound qualifies on its core.
        const std::string slotNeg = "!(zzcomp[u_1,u_4,u_10,u_12])";
        const gl::StrSpan key[5] = { gl::StrSpan(anchor), gl::StrSpan(b1),
                                     gl::StrSpan(b2), gl::StrSpan(b3),
                                     gl::StrSpan(slotNeg) };
        ASSERT_TRUE(ea.admissionPolicy.ordis2DemandSlotQualifies(key, 5, 4));
    }
    {   // Arity below kOrdis2DemandMinArity: rejected.
        const std::string slot3 = "(zzthree[u_1,u_4,u_10])";
        const gl::StrSpan key[5] = { gl::StrSpan(anchor), gl::StrSpan(b1),
                                     gl::StrSpan(b2), gl::StrSpan(b3),
                                     gl::StrSpan(slot3) };
        ASSERT_FALSE(ea.admissionPolicy.ordis2DemandSlotQualifies(key, 5, 4));
    }
    {   // An anchor-slot argument is ground by construction and counts as
        // bound (D-287): u_6 appears nowhere non-anchor, but u_10 does, so
        // the slot qualifies. This is B9's `1 <= d` shape, whose numeral 1
        // is an anchor slot.
        const std::string slotAnch = "(zzcomp[u_1,u_4,u_10,u_6])";
        const gl::StrSpan key[5] = { gl::StrSpan(anchor), gl::StrSpan(b1),
                                     gl::StrSpan(b2), gl::StrSpan(b3),
                                     gl::StrSpan(slotAnch) };
        ASSERT_TRUE(ea.admissionPolicy.ordis2DemandSlotQualifies(key, 5, 4));
    }
    {   // Every argument an anchor slot, none substantive: still rejected
        // (the A14 shape) — a slot the anchor alone binds mints no demand.
        // u_2 / u_3 / u_6 are the fixture's anchor-only slots (u_1, u_4,
        // u_5 all appear in the non-anchor binders).
        const std::string slotAllAnch = "(zzcomp[u_2,u_3,u_6,u_2])";
        const gl::StrSpan key[5] = { gl::StrSpan(anchor), gl::StrSpan(b1),
                                     gl::StrSpan(b2), gl::StrSpan(b3),
                                     gl::StrSpan(slotAllAnch) };
        ASSERT_FALSE(ea.admissionPolicy.ordis2DemandSlotQualifies(key, 5, 4));
    }
}

// The justification-split minimum-premise floor: an integration-justified
// rule qualifies only at kOrdis2DemandMinPremisesIntegration (= 6)
// non-anchor premises — the D1 closure rule (three typings + two matched
// conjuncts + the one missing ground compound premise) — a floor ABOVE the
// general one (= 4), so a five-non-anchor key qualifies as a general rule
// but not as an integration rule; below both floors both are refused.
// Every other gate (key language, subset, substantive binder) is shared
// and pinned by slot_qualification_matrix above.
TEST(ordis2, integration_floor_qualifies_six_premise_rule) {
    gl::tests::ProverFixture ea("Peano");
    gl::CompiledDefinitionFixture::compiledExpressions(ea.definitions)["zzcomp"] =
        gl::LogicalEntity("existence",
            { "(in[w1,u_1])", "(in2[u_2,w1,u_3])" },
            "(zzcomp[u_1,u_2,u_3,u_4])", 4);
    ea.supportingRuleCompiler.compileDisintegrationPackages();

    const std::string anchor = "(AnchorPeano[u_1,u_2,u_3,u_4,u_5,u_6])";
    const std::string b1 = "(in3[u_1,u_4,u_10,u_5])";
    const std::string b2 = "(in3[u_12,u_9,u_11,u_5])";
    const std::string b3 = "(in3[u_13,u_9,u_11,u_5])";
    const std::string b4 = "(in3[u_14,u_9,u_11,u_5])";
    const std::string b5 = "(in3[u_15,u_9,u_11,u_5])";
    const std::string slotOk = "(zzcomp[u_1,u_4,u_10,u_12])";

    {   // Six non-anchor premises (five binders + slot): both floors admit.
        const gl::StrSpan key[7] = { gl::StrSpan(anchor), gl::StrSpan(b1),
                                     gl::StrSpan(b2), gl::StrSpan(b3),
                                     gl::StrSpan(b4), gl::StrSpan(b5),
                                     gl::StrSpan(slotOk) };
        ASSERT_TRUE(ea.admissionPolicy.ordis2DemandSlotQualifies(key, 7, 6,
            /*integrationJustified=*/true));
        ASSERT_TRUE(ea.admissionPolicy.ordis2DemandSlotQualifies(key, 7, 6,
            /*integrationJustified=*/false));
    }
    {   // Five non-anchor premises: the general floor admits, the
        // integration floor refuses.
        const gl::StrSpan key[6] = { gl::StrSpan(anchor), gl::StrSpan(b1),
                                     gl::StrSpan(b2), gl::StrSpan(b3),
                                     gl::StrSpan(b4), gl::StrSpan(slotOk) };
        ASSERT_FALSE(ea.admissionPolicy.ordis2DemandSlotQualifies(key, 6, 5,
            /*integrationJustified=*/true));
        ASSERT_TRUE(ea.admissionPolicy.ordis2DemandSlotQualifies(key, 6, 5,
            /*integrationJustified=*/false));
    }
    {   // Three non-anchor premises (two binders + slot): below both floors.
        const gl::StrSpan key[4] = { gl::StrSpan(anchor), gl::StrSpan(b1),
                                     gl::StrSpan(b2), gl::StrSpan(slotOk) };
        ASSERT_FALSE(ea.admissionPolicy.ordis2DemandSlotQualifies(key, 4, 3,
            /*integrationJustified=*/true));
        ASSERT_FALSE(ea.admissionPolicy.ordis2DemandSlotQualifies(key, 4, 3,
            /*integrationJustified=*/false));
    }
}

// The integration floor end-to-end through makeNormalizedKeysForAdmission:
// a five-non-anchor-premise key installs a tagged demand variant as a
// general rule but NOT as an integration-justified one (floor 6 > 4); a
// six-non-anchor key installs on both routes, and the integration-justified
// variant carries justification `integration` on its LMV (the purity
// exemption's carrier, D-357). The
// regular/ordis routes stay off exactly as in
// install_route_installs_verbatim_tagged_variant.
TEST(ordis2, install_route_integration_floor_gates_demand_variant) {
    const std::string anchor = "(AnchorPeano[u_1,u_2,u_3,u_4,u_5,u_6])";
    const std::vector<std::string> binders = {
        "(in3[u_1,u_4,u_10,u_5])", "(in3[u_12,u_9,u_11,u_5])",
        "(in3[u_13,u_9,u_11,u_5])", "(in3[u_14,u_9,u_11,u_5])",
        "(in3[u_15,u_9,u_11,u_5])" };
    const std::string slot = "(zzcomp[u_1,u_4,u_10,u_12])";

    const auto countDemand = [&](int binderN, bool integrationJustified,
                                 int& integrationTagged) -> int {
        gl::tests::ProverFixture ea("Peano");
        ea.parameters.minLenLongKey = 100;
        gl::CompiledDefinitionFixture::compiledExpressions(ea.definitions)["zzcomp"] =
            gl::LogicalEntity("existence",
                { "(in[w1,u_1])", "(in2[u_2,w1,u_3])" },
                "(zzcomp[u_1,u_2,u_3,u_4])", 4);
        ea.supportingRuleCompiler.compileDisintegrationPackages();
        gl::Memory m;
        if (((m).nameMap).encode("main") != gl::NameMap::MAIN_ID) return -1;
        std::vector<std::string> key = { anchor };
        for (int i = 0; i < binderN; ++i) key.push_back(binders[i]);
        key.push_back(slot);
        ea.admissionMarkers.makeNormalizedKeysForAdmission(key, ((m).overallHashMemory), ((m).nameMap),
            ((m).ruleInterner), "(=[u_10,u_12])", /*minNumOperatorsKey=*/100,
            "(>[9]dummyImpl(=[9,9]))", "main", /*mbTrap=*/nullptr,
            integrationJustified);
        int found = 0;
        integrationTagged = 0;
        const auto& em = ((((m).overallHashMemory)).encodedMap);
        for (int32_t id = 1; id <= em.count(); ++id) {
            const std::vector<gl::LocalMemoryValue> run = em.recordsAt(id);
            for (const gl::LocalMemoryValue& lmv : run) {
                if (!lmv.ordis2Demand) continue;
                ++found;
                if (lmv.justification == gl::RuleJustification::integration)
                    ++integrationTagged;
            }
        }
        return found;
    };

    int tagged = 0;
    // Five non-anchor premises: general installs, integration refuses.
    ASSERT_TRUE(countDemand(4, /*integrationJustified=*/false, tagged) >= 1);
    ASSERT_EQ(tagged, 0);
    ASSERT_EQ(countDemand(4, /*integrationJustified=*/true, tagged), 0);
    // Six non-anchor premises: both install; only the integration-justified
    // install stamps `integration` on every demand LMV.
    const int generalSix = countDemand(5, /*integrationJustified=*/false, tagged);
    ASSERT_TRUE(generalSix >= 1);
    ASSERT_EQ(tagged, 0);
    const int integrationSix = countDemand(5, /*integrationJustified=*/true, tagged);
    ASSERT_TRUE(integrationSix >= 1);
    ASSERT_EQ(tagged, integrationSix);
}

// The ordis2-demand install route end-to-end: a B5-shaped rule installs a
// marker-style variant whose LMV carries isMarker + ordis2Demand and whose
// VALUE is the slot verbatim. The regular/ordis routes stay off (head is
// an equality; minNumOperatorsKey set out of reach).
TEST(ordis2, install_route_installs_verbatim_tagged_variant) {
    gl::tests::ProverFixture ea("Peano");
    ea.parameters.minLenLongKey = 100;
    gl::CompiledDefinitionFixture::compiledExpressions(ea.definitions)["zzcomp"] =
        gl::LogicalEntity("existence",
            { "(in[w1,u_1])", "(in2[u_2,w1,u_3])" },
            "(zzcomp[u_1,u_2,u_3,u_4])", 4);
    ea.supportingRuleCompiler.compileDisintegrationPackages();
    // Deliberately NO operators.insert: the demand pass never consults
    // `operators` (the eligibility contract, pinned by
    // ordis2.key_eligible_matrix), and with zzcomp outside `operators`
    // the post-demand fall-through fails both regular and ordis routes
    // and returns before their core-config loops can see the synthetic
    // operator.

    gl::Memory m;
    ASSERT_EQ(((m).nameMap).encode("main"), gl::NameMap::MAIN_ID);
    // Four non-anchor premises reach kOrdis2DemandMinPremises. The anchor is
    // excluded from that eligibility count but retained in the installed
    // derivative key. The raised long-key threshold keeps this synthetic
    // operator outside the unrelated regular route.
    const std::vector<std::string> key = {
        "(AnchorPeano[u_1,u_2,u_3,u_4,u_5,u_6])",
        "(in3[u_1,u_4,u_10,u_5])",
        "(in3[u_12,u_9,u_11,u_5])",
        "(in2[u_10,u_12,u_5])",
        "(zzcomp[u_1,u_4,u_10,u_12])" };
    ea.admissionMarkers.makeNormalizedKeysForAdmission(key, ((m).overallHashMemory), ((m).nameMap),
        ((m).ruleInterner), "(=[u_10,u_12])", /*minNumOperatorsKey=*/100,
        "(>[9]dummyImpl(=[9,9]))", "main");

    int found = 0;
    const auto& em = ((((m).overallHashMemory)).encodedMap);
    for (int32_t id = 1; id <= em.count(); ++id) {
        const std::vector<gl::LocalMemoryValue> run = em.recordsAt(id);
        for (const gl::LocalMemoryValue& lmv : run) {
            if (!lmv.ordis2Demand) continue;
            ++found;
            ASSERT_EQ(em.decodeKey(id).numberExpressions, 4);
            bool hasAnchor = false;
            for (const int32_t keyId : lmv.keyIds) {
                const std::string keyElem = ((m).ruleInterner).decode(keyId);
                if (keyElem.rfind("(Anchor", 0) == 0) hasAnchor = true;
            }
            ASSERT_TRUE(hasAnchor);
            // OWN kind — neither head nor marker (no marker anywhere in
            // the ordis2 machinery, maintainer decision 2026-08-10).
            ASSERT_FALSE(lmv.isMarker);
            // VERBATIM value: all args here are unchangeable u_ tokens, so
            // the normalized-variable decimal rename has nothing to touch —
            // the stored value is the slot byte-for-byte (no marker token).
            // At fire the u_-strip grounds them to the literally-matched
            // bytes, exactly as marker values ground today.
            const std::string val = ((m).ruleInterner).decode(lmv.valueId);
            ASSERT_TRUE(val == "(zzcomp[u_1,u_4,u_10,u_12])");
        }
    }
    ASSERT_TRUE(found >= 1);
}

// No goals, no or branches: an LB with an empty goal registry neither opens
// nor parks a cohort even under route (b); seeding a goal re-enables the open.
TEST(prover, ordis_no_goals_neither_opens_nor_parks) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    gl::WorkInstruction wi(&lb);
    buildOrInstruction(wi, "(or98[u_4,u_5])",
                       { "(existence3[u_1,u_4,u_3])", "(=[u_4,u_5])" });
    gl::tests::ProverFixture analyzer("Peano");
    gl::Memory memory;
    ASSERT_EQ(((memory).nameMap).encode("main"), gl::NameMap::MAIN_ID);
    gl::CollectedArena collected(&lb);
    gl::NewVarStore newVarMap(&lb);
    int seedLv[1] = { 0 };
    const int32_t cohortId = gl::mintOrCohortId(((memory).lbStateInterner),
        ((memory).lbStateInterner).encode(gl::StrSpan("main", 4)),
        ((memory).lbStateInterner).encode(gl::StrSpan("(or98[4,5])", 11)));

    // Empty goal registry, route (b) on, real deposit: neither open nor park.
    gl::tests::DisintegrationProcessorFixture::disintegrateExprCore2(analyzer.disintegrationProcessor, gl::StrSpan("(or98[u_4,u_5])", 15), wi,
        memory, 0, collected, newVarMap, gl::StrSpan("main", 4),
        /*trackHistoryLocal=*/true, /*allowOrDisintegration=*/true,
        seedLv, 1, /*allowOrProbe=*/true);
    ASSERT_TRUE(((memory).orDisjunctCount).lookup(cohortId) == 0);
    ASSERT_TRUE(((memory).orPendingBranches).lookup(cohortId) == 0);
    ASSERT_EQ(((memory).pendingOrReleases).count(), 0);
    ASSERT_EQ(((((memory).overallHashMemory)).rejectedMapOrdis).count(), 0);
    ASSERT_EQ(((((memory).overallHashMemory)).rejectedMapOrdis2).count(), 0);
    ASSERT_EQ(gl::decodeInternalMailStatements(
        ((memory).sameIterationInternalMail), ((memory).nameMap)).size(),
        static_cast<std::size_t>(0));

    // Seed a goal: the same deposit opens the cohort.
    ((memory).intToBeProved).assignSet(gl::packStatementKey(
        ((memory).nameMap).encode("(freeze_goal[1])"), gl::NameMap::MAIN_ID),
        nullptr, 0);
    gl::tests::DisintegrationProcessorFixture::disintegrateExprCore2(analyzer.disintegrationProcessor, gl::StrSpan("(or98[u_4,u_5])", 15), wi,
        memory, 0, collected, newVarMap, gl::StrSpan("main", 4),
        /*trackHistoryLocal=*/true, /*allowOrDisintegration=*/true,
        seedLv, 1, /*allowOrProbe=*/true);
    ASSERT_TRUE(((memory).orDisjunctCount).lookup(cohortId) != 0);
    ASSERT_TRUE(((memory).pendingOrReleases).lookup(cohortId) != 0);
}

TEST(prover, refused_ancestor_known_deposit_still_stages_or_release) {
    // I-174 companion of the ancestor-known dedup: a branch deposit whose
    // expression matches an open parent-chain goal is REFUSED (the text is
    // already known at main), but the sequenced-release resolution signal
    // must survive — the branch still derived the expression, and without
    // the refusal-side probe the cohort would never release its next
    // pending branch.
    gl::tests::ProverFixture analyzer(std::string("Peano"));
    gl::Memory memory;
    ASSERT_EQ(((memory).nameMap).encode("main"), gl::NameMap::MAIN_ID);
    const gl::NameId branchVid = ((memory).nameMap).encodePush(
        gl::NameMap::MAIN_ID, "ordis_(or98[4,5])_((in2[4,7,3]))");
    const std::string branch = ((memory).nameMap).decode(branchVid);

    // Cohort with a queued pending branch (sequenced release live).
    const int32_t cohortId = gl::mintOrCohortId(((memory).lbStateInterner),
        ((memory).lbStateInterner).encode(gl::StrSpan("main", 4)),
        ((memory).lbStateInterner).encode(gl::StrSpan("(or98[4,5])", 11)));
    ((memory).orPendingBranches).insertSorted(cohortId,
        ((memory).lbStateInterner).encode(gl::StrSpan("((in3[7,8,9,4]))", 16)),
        gl::DecodedIdLess{ &((memory).lbStateInterner) });

    // An open goal at main; the same expression is already KNOWN at main,
    // so the branch deposit below is ancestor-known.
    const std::string goal = "(in[7,N])";
    const gl::NameId goalId = ((memory).nameMap).encode(goal);
    const int lv[1] = { 0 };
    ((memory).intToBeProved).assignSetRange(
        gl::packStatementKey(goalId, gl::NameMap::MAIN_ID), lv, lv + 1);
    gl::upsertStatementKey(((memory).intKnownStatements),
        gl::packStatementKey(goalId, gl::NameMap::MAIN_ID), /*local=*/true);
    ((memory).intStatementLevelsMap).assignSetRange(
        gl::packStatementKey(goalId, gl::NameMap::MAIN_ID), lv, lv + 1);

    gl::DirtyState outDirty = gl::DirtyState::Clean;
    gl::PagedVector<gl::IntEncodedExpr> out(
        &gl::genScratchArenas().forSlot(0), &outDirty);
    const gl::TransientOrigin noOrigin{};
    ASSERT_TRUE(((memory).pendingOrReleases).lookup(cohortId) == 0);

    analyzer.statementMemory.addStatement(gl::StrSpan(goal), memory, true, lv, 1, noOrigin,
                          gl::StrSpan(branch), out);

    // Refused: no branch row, no out-param row...
    ASSERT_TRUE(gl::lookupStatementLevels(((memory).intStatementLevelsMap),
        ((memory).nameMap), gl::StrSpan(goal), gl::StrSpan(branch)) == 0);
    ASSERT_EQ(out.size(), 0);
    // ...but the sequenced-release resolution signal survived.
    ASSERT_TRUE(((memory).pendingOrReleases).lookup(cohortId) != 0);
}

// WitnessMetaStore records witness -> (spawning compact, enclosing witness)
// in mint order; selectRelayWitnesses marks the HIGHEST uncovered existence
// groups (D-284): an uncovered group nested inside another
// uncovered group is NOT selected, the top compound has no parent, and a
// covered group is never selected.
TEST(prover, relay_selection_marks_highest_uncovered_group) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);

    gl::WitnessMetaStore meta(lb);
    meta.append(gl::TemporaryWitnessKind::Algebra, 0, 0, 0,
        gl::StrSpan("(existenceA[x])", 15));
    meta.append(gl::TemporaryWitnessKind::Integration, 0, 0, 1,
        gl::StrSpan("(existenceA[x])", 15));
    meta.append(gl::TemporaryWitnessKind::Algebra, 0, 1, 2,
        gl::StrSpan("(existenceC[y])", 15));
    meta.append(gl::TemporaryWitnessKind::Integration, 0, 1, 3,
        gl::StrSpan("(existenceC[y])", 15));
    meta.closeSubtree(4);
    ASSERT_EQ(meta.slot(4).subtreeEnd, 5);
    ASSERT_EQ(meta.count(), 4);
    ASSERT_EQ(meta.lookup(meta.witnessAt(1)), 2);
    ASSERT_TRUE(gl::equalSpans(meta.instanceAt(2), gl::StrSpan("(existenceC[y])", 15)));
    ASSERT_TRUE(meta.parentAt(0).empty());
    ASSERT_TRUE(gl::equalSpans(meta.parentAt(3), meta.token(1)));

    // Groups: w0/w1 share sig 1 (A), w2/w3 share sig 2 (C); w2/w3's parent is
    // witness 0 (A's it_ witness).
    const int32_t sig[4] = { 1, 1, 2, 2 };
    const int32_t par[4] = { -1, -1, 0, 0 };
    bool relay[4];

    // A covered, C uncovered: relay C alone.
    int32_t n = gl::tests::EqualityPropagationFixture::selectRelayWitnesses(sig, par, 4,
        [](int32_t s) { return s == 1; }, relay);
    ASSERT_EQ(n, 2);
    ASSERT_TRUE(!relay[0] && !relay[1] && relay[2] && relay[3]);

    // Both uncovered: relay A alone (C rides inside it).
    n = gl::tests::EqualityPropagationFixture::selectRelayWitnesses(sig, par, 4,
        [](int32_t) { return false; }, relay);
    ASSERT_EQ(n, 2);
    ASSERT_TRUE(relay[0] && relay[1] && !relay[2] && !relay[3]);

    // All covered: nothing to relay.
    n = gl::tests::EqualityPropagationFixture::selectRelayWitnesses(sig, par, 4,
        [](int32_t) { return true; }, relay);
    ASSERT_EQ(n, 0);
    ASSERT_TRUE(!relay[0] && !relay[1] && !relay[2] && !relay[3]);
}

// Memory::stagePendingRelay / erasePendingRelay: fresh stage records stamp +
// level run; restage min-merges the stamp (with -1 losing to any real
// generation) and keeps the first-seen level run; erase drops both sides.
TEST(prover, pending_relay_stage_merge_erase) {
    gl::Memory memory;
    const gl::NameId id =
        ((memory).nameMap).encode("(existence1[1,int_0_lev_1_1,int_0_lev_2_1,4])");
    const int lv[2] = { 1, 2 };
    memory.stagePendingRelay(id, 5, lv, 2);
    memory.stagePendingRelay(id, 3, nullptr, 0);   // min-merge; levels kept
    memory.stagePendingRelay(id, -1, nullptr, 0);  // none loses to real
    ASSERT_EQ(((memory).pendingRelayIter).count(), 1);
    const int32_t vid = ((memory).pendingRelayIter).lookup(static_cast<int32_t>(id));
    ASSERT_TRUE(vid != 0);
    ASSERT_EQ(((memory).pendingRelayIter).valueAt(vid), 3);
    const int32_t lvId =
        ((memory).pendingRelayLevels).lookup(static_cast<int32_t>(id));
    ASSERT_TRUE(lvId != 0);
    ASSERT_EQ(((memory).pendingRelayLevels).runLen(lvId), 2);
    ASSERT_EQ(((memory).pendingRelayLevels).valueAt(lvId, 0), 1);
    ASSERT_EQ(((memory).pendingRelayLevels).valueAt(lvId, 1), 2);
    memory.erasePendingRelay(id);
    ASSERT_EQ(((memory).pendingRelayIter).count(), 0);
    ASSERT_TRUE(((memory).pendingRelayLevels).lookup(static_cast<int32_t>(id)) == 0);
}

// Unconditional-ordis staging (D-528): a restage
// keeps one entry, erasing an OR that is not staged is a no-op, erasing the
// staged OR empties the set.
TEST(prover, pending_unconditional_ordis_stage_erase) {
    gl::Memory memory;
    const gl::NameId id = ((memory).nameMap).encode("(or8[1,4,5,6,9,x,10,y])");
    const gl::NameId other = ((memory).nameMap).encode("(in[x,1])");
    memory.stageUnconditionalOrdis(id);
    memory.stageUnconditionalOrdis(id);
    ASSERT_EQ(((memory).lbMemory.pendingUnconditionalOrdis).count(), 1);
    ASSERT_TRUE(((memory).lbMemory.pendingUnconditionalOrdis).lookup(id) != 0);
    memory.eraseUnconditionalOrdis(other);
    ASSERT_EQ(((memory).lbMemory.pendingUnconditionalOrdis).count(), 1);
    memory.eraseUnconditionalOrdis(id);
    ASSERT_TRUE(((memory).lbMemory.pendingUnconditionalOrdis).empty());
}

// copyOriginRowsToMailOut mirrors one expression's cold origin rows into the
// outgoing mailbox (the extracted per-key half of the fillMailOut origin
// mirror): rows land under the same key with tag + dependencies preserved; a
// key without rows is a defined no-op.
TEST(prover, copy_origin_rows_to_mail_out) {
    gl::tests::ProverFixture analyzer("Peano");
    analyzer.parameters.trackHistory = true;
    gl::Memory memory;
    ASSERT_EQ(((memory).nameMap).encode("main"), gl::NameMap::MAIN_ID);

    const gl::StrSpan key("(in3[2,3,5,4])", 14);
    const gl::StrSpan dep("(in3[3,2,5,4])", 14);
    gl::OriginDep deps[1] = { { dep, gl::StrSpan("main", 4) } };
    gl::addOriginEncoded(((memory).exprOriginMap), ((memory).originInterner),
        key, gl::StrSpan("main", 4), gl::OriginTag::expansion, deps, 1, 8);

    analyzer.proofHistory.copyOriginRowsToMailOut(memory, key, gl::StrSpan("main", 4), 8);
    const gl::Mail m =
        gl::routingMailOutToHeap(((memory).mailOut), ((memory).mailOutInterner));
    const gl::ExpressionWithValidity k("(in3[2,3,5,4])", "main");
    ASSERT_EQ(((m).exprOriginMap).at(k).size(), static_cast<std::size_t>(1));
    ASSERT_TRUE(((m).exprOriginMap).at(k)[0].first == "expansion");
    ASSERT_EQ(((m).exprOriginMap).at(k)[0].second.size(),
              static_cast<std::size_t>(1));
    ASSERT_TRUE(((m).exprOriginMap).at(k)[0].second[0].original
                == "(in3[3,2,5,4])");

    // Absent key: defined no-op (statements channel untouched either way).
    analyzer.proofHistory.copyOriginRowsToMailOut(memory, gl::StrSpan("(in[9,1])", 10),
        gl::StrSpan("main", 4), 8);
    ASSERT_TRUE(((memory).mailOut).statementsEmpty());
}

// Status-5 door (the flag-5 relay receiver half): a positive existence
// compact deposited at status 5 DOES disintegrate (the status-3 rule-carrier
// gate does not apply). An undemanded source allocates no permanent witness;
// later demand replays it through ordinary absorption while the CARRIER
// registers as a NON-local row (absent from the local delta, so it can never
// be re-forwarded); no relay is re-staged from a mailed carrier (no cascade).
TEST(prover, status5_relay_arrival_disintegrates_at_receiver) {
    gl::tests::ProverFixture ea(std::string("Peano"));
    gl::CompiledDefinitionFixture::compiledExpressions(ea.definitions)["existence97"] =
        gl::LogicalEntity("existence",
            { "(in[w1,u_1])", "(in2[u_2,w1,u_3])" },
            "(existence97[u_1,u_2,u_3])", 3);
    ea.supportingRuleCompiler.compileDisintegrationPackages();
    // A receiving LB sits at a positive tree level (the root sentinel's -1
    // would put a '-' into the minted int_lev name, outside the recognized
    // witness pattern — production relay arrivals never land at the root).
    gl::Memory memory;
    ((memory).level) = 3;
    ASSERT_EQ(((memory).nameMap).encode("main"), gl::NameMap::MAIN_ID);

    const std::string compact = "(existence97[1,2,3])";
    const int lv[1] = { 0 };
    const gl::TransientOrigin origin{
        true, gl::OriginTag::taskFormulation, nullptr, 0 };
    const int startIntBefore = ((memory).startInt);
    ea.statementMemory.addExprToMemoryBlock(gl::StrSpan(compact), memory, -1, 5, lv, 1,
        origin, -1, -1, gl::StrSpan("main", 4), false);

    // Carrier registered...
    ASSERT_TRUE(gl::lookupStatementLevels(((memory).intStatementLevelsMap),
        ((memory).nameMap), gl::StrSpan(compact), gl::StrSpan("main", 4)) != 0);
    // ...but NON-local: no local-delta row for it.
    const gl::NameId compactId = ((memory).nameMap).encode(compact);
    bool inDelta = false;
    for (int32_t i = 0; i < ((memory).intLocalEncodedStatementsDelta).size(); ++i)
        if (((memory).intLocalEncodedStatementsDelta)[i].originalId == compactId)
            inDelta = true;
    ASSERT_TRUE(!inDelta);

    // An undemanded source allocates no permanent witness.
    ASSERT_EQ(((memory).startInt), startIntBefore);
    ASSERT_TRUE(((((memory).overallHashMemory)).rejectedMapIntegration).count() >= 1);
    ASSERT_EQ(((memory).pendingRelayIter).count(), 0);

    const std::string marked = "(in2[2,marker,3])";
    const int64_t demand = gl::mintTemplateKey(((memory).templateInterner), ((memory).nameMap),
        marked, std::string("main"));
    ((((memory).overallHashMemory)).admissionSetIntegration).mint(demand);
    ea.rejectedSourceReplay.revisitRejectedIntegration2(gl::StrSpan(marked), memory, gl::StrSpan("main",4));
    ASSERT_EQ(((memory).startInt), startIntBefore);
    ea.logicBlockOperations.standardProcessing(memory, nullptr, ((memory).sameIterationInternalMail),
        ((memory).nextIterationInternalMail), -1);
    ASSERT_EQ(((memory).startInt), startIntBefore + 1);
    const auto* carrier = gl::lookupStatementFlags(((memory).intKnownStatements),
        ((memory).nameMap), gl::StrSpan(compact), gl::StrSpan("main",4));
    ASSERT_TRUE(carrier != nullptr && !carrier->local);
    ASSERT_EQ(((memory).pendingRelayIter).count(), 0);
}

// "Really missing" gate at branch creation
// (D-358): a standing demand whose leaf is
// ALREADY KNOWN at the deposit scope (or an ancestor) is a satisfied demand —
// route (c) opens nothing and the stale demand entry is consumed by the probe.
// The or itself holds through that leaf, so it is not parked either
// (D-438).
TEST(prover, ordis2_route_c_skips_known_leaf_and_consumes_stale_demand) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    gl::WorkInstruction wi(&lb);
    buildOrInstruction(wi, "(or95[u_4,u_5])",
                       { "(preorder95[u_1,u_2,u_4,u_5])", "(=[u_4,u_5])" });

    gl::tests::ProverFixture analyzer("Peano");
    gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions)["preorder95"] =
        gl::LogicalEntity("existence",
            { "(in[w1,u_1])", "(in3[u_4,w1,u_5,u_2])" },
            "(preorder95[u_1,u_2,u_4,u_5])", 4);
    analyzer.supportingRuleCompiler.compileDisintegrationPackages();
    gl::Memory memory;
    ASSERT_EQ(((memory).nameMap).encode("main"), gl::NameMap::MAIN_ID);
    gl::CollectedArena collected(&lb);
    gl::NewVarStore newVarMap(&lb);
    gl::ScratchArena& tArena = gl::genScratchArenas().forSlot(
        gl::genScratchArenas().slotCount() - 1);
    int seedLv[1] = { 0 };

    // A pending goal keeps OR parking enabled under the goal-presence contract.
    ((memory).intToBeProved).assignSet(gl::packStatementKey(
        ((memory).nameMap).encode("(freeze_goal[1])"), gl::NameMap::MAIN_ID),
        nullptr, 0);

    // A standing demand for the eligible disjunct ...
    const int64_t demandPk = gl::mintTemplateKey(((memory).templateInterner),
        ((memory).nameMap), "(preorder95[1,2,4,5])", "main");
    const int32_t srcImpl = ((memory).ruleInterner).encode("(>[]dummy(d[1]))");
    const int dlv[] = { 0 };
    gl::insertAdmissionOrdis2IdsBlob(
        ((((memory).overallHashMemory)).admissionMapOrdis2), demandPk, srcImpl,
        dlv, 1, ((memory).ruleInterner), tArena);
    // ... whose text is ALREADY KNOWN at main.
    ((memory).intKnownStatements).insert(
        gl::StatementKey{ ((memory).nameMap).encode("(preorder95[1,2,4,5])"),
                          gl::NameMap::MAIN_ID },
        gl::StatementFlags{ true, false });

    gl::tests::DisintegrationProcessorFixture::disintegrateExprCore2(analyzer.disintegrationProcessor, gl::StrSpan("(or95[u_4,u_5])", 15), wi,
        memory, 0, collected, newVarMap, gl::StrSpan("main", 4),
        /*trackHistoryLocal=*/true, /*allowOrDisintegration=*/false,
        seedLv, 1, /*allowOrProbe=*/true);

    // No cohort opened on the satisfied demand ...
    const int32_t cohortId = gl::mintOrCohortId(((memory).lbStateInterner),
        ((memory).lbStateInterner).encode(gl::StrSpan("main", 4)),
        ((memory).lbStateInterner).encode(gl::StrSpan("(or95[4,5])", 11)));
    ASSERT_TRUE(((memory).orDisjunctCount).lookup(cohortId) == 0);
    ASSERT_TRUE(((memory).pendingOrReleases).lookup(cohortId) == 0);
    // ... the satisfied or is not parked ...
    ASSERT_EQ(((((memory).overallHashMemory)).rejectedMapOrdis).count(), 0);
    ASSERT_EQ(((((memory).overallHashMemory)).rejectedMapOrdis2).count(), 0);
    // ... and the stale demand entry was consumed.
    ASSERT_TRUE(((((memory).overallHashMemory)).admissionMapOrdis2).lookup(demandPk)
                == 0);
}

// After one opening, only a fact at the demand scope or an ancestor can
// satisfy the shared demand. A sibling's assumed fact cannot consume it.
TEST(prover, ordis2_retained_demand_cleanup_respects_fact_scope) {
    for (int knownScope = 0; knownScope < 3; ++knownScope) {
        gl::GlobalMemoryManager g;
        g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
        gl::LbArena lb(&g);
        gl::WorkInstruction first(&lb), second(&lb);
        buildOrInstruction(first, "(or95[u_4,u_5])",
            { "(preorder95[u_1,u_2,u_4,u_5])", "(=[u_4,u_5])" });
        buildOrInstruction(second, "(or96[u_4,u_5])",
            { "(preorder95[u_1,u_2,u_4,u_5])", "!(=[u_4,u_5])" });
        gl::tests::ProverFixture analyzer("Peano");
        gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions)["preorder95"] = gl::LogicalEntity(
            "existence", { "(in[w1,u_1])", "(in3[u_4,w1,u_5,u_2])" },
            "(preorder95[u_1,u_2,u_4,u_5])", 4);
        gl::Memory memory;
        ASSERT_EQ(((memory).nameMap).encode("main"), gl::NameMap::MAIN_ID);
        const gl::NameId target = ((memory).nameMap).encodePush(
            gl::NameMap::MAIN_ID, "demand_target");
        const gl::NameId sibling = ((memory).nameMap).encodePush(
            gl::NameMap::MAIN_ID, "demand_sibling");
        const std::string validity = ((memory).nameMap).decode(target);
        ((memory).intToBeProved).assignSet(gl::packStatementKey(
            ((memory).nameMap).encode("(freeze_goal[1])"), target), nullptr, 0);
        auto& arena = gl::genScratchArenas().forSlot(
            gl::genScratchArenas().slotCount() - 1);
        const int64_t demand = gl::mintTemplateKey(((memory).templateInterner),
            ((memory).nameMap), "(preorder95[1,2,4,5])", validity);
        const int32_t source = ((memory).ruleInterner).encode("(>[]dummy(d[1]))");
        const int levels[] = { 0 };
        gl::insertAdmissionOrdis2IdsBlob(((((memory).overallHashMemory)).admissionMapOrdis2),
            demand, source, levels, 1, ((memory).ruleInterner), arena);
        gl::CollectedArena collected(&lb);
        gl::NewVarStore newVarMap(&lb);
        gl::tests::DisintegrationProcessorFixture::disintegrateExprCore2(analyzer.disintegrationProcessor, gl::StrSpan("(or95[u_4,u_5])", 15), first,
            memory, 0, collected, newVarMap, gl::StrSpan(validity),
            true, false, levels, 1, true);
        ASSERT_EQ(((memory).orDisjunctCount).count(), 1);
        ASSERT_TRUE(((((memory).overallHashMemory)).admissionMapOrdis2).lookup(demand) != 0);

        const gl::NameId factScope = knownScope == 0 ? target
            : knownScope == 1 ? gl::NameMap::MAIN_ID : sibling;
        const auto fact = gl::packStatementKey(
            ((memory).nameMap).encode("(preorder95[1,2,4,5])"), factScope);
        gl::upsertStatementKey(((memory).intKnownStatements), fact, true);
        ((memory).intStatementLevelsMap).assignSet(fact, levels, 1);
        gl::tests::DisintegrationProcessorFixture::disintegrateExprCore2(analyzer.disintegrationProcessor, gl::StrSpan("(or96[u_4,u_5])", 15), second,
            memory, 0, collected, newVarMap, gl::StrSpan(validity),
            true, false, levels, 1, true);
        const bool stillMissing = knownScope == 2;
        ASSERT_EQ(((((memory).overallHashMemory)).admissionMapOrdis2).lookup(demand) != 0,
            stillMissing);
        ASSERT_EQ(((memory).orDisjunctCount).count(), stillMissing ? 2 : 1);
    }
}

namespace {
    gl::StrSpan orSpan(const char* s) {
        return gl::StrSpan(s, static_cast<int32_t>(std::strlen(s)));
    }
} // namespace

// The or-uniqueness gate's three primitives (record -> probe -> clear):
// the probe matches the recorded text byte-equal and as a same-operator
// equi variant under the door's own canonicalization walk
// (canonicalFormAtScope — the gate is exactly door-consistent, neither
// wider nor narrower), refuses different args, different operators, and
// the digit-prefix near-miss "(or9[" vs "(or98["; the clear is
// idempotent (several park keys of one or each wake once).
TEST(prover, or_unique_ledger_record_probe_clear) {
    gl::tests::ProverFixture analyzer("Peano");
    gl::Memory memory;
    ASSERT_EQ(((memory).nameMap).encode("main"), gl::NameMap::MAIN_ID);

    gl::EquivalenceClass cls;
    cls.setMembersFromNames({ "a", "b" }, ((memory).nameMap));
    memory.assignClassesById(gl::NameMap::MAIN_ID, { cls });
    seedFixtureCanonicalClasses(memory);

    const gl::NameId recId = ((memory).nameMap).encode("(or98[b,7])");
    analyzer.caseLedger.recordProcessedOr(memory, recId, gl::NameMap::MAIN_ID);

    // Byte-equal recorded text.
    ASSERT_TRUE(analyzer.caseLedger.orEquiRepresentativeRecorded(memory,
        orSpan("(or98[b,7])"), orSpan("main")));
    // Equi variant: the recorded spelling canonicalizes (b -> a) to the
    // probe text.
    ASSERT_TRUE(analyzer.caseLedger.orEquiRepresentativeRecorded(memory,
        orSpan("(or98[a,7])"), orSpan("main")));
    // Different argument / different operator / digit-prefix near-miss.
    ASSERT_FALSE(analyzer.caseLedger.orEquiRepresentativeRecorded(memory,
        orSpan("(or98[a,8])"), orSpan("main")));
    ASSERT_FALSE(analyzer.caseLedger.orEquiRepresentativeRecorded(memory,
        orSpan("(or99[a,7])"), orSpan("main")));
    ASSERT_FALSE(analyzer.caseLedger.orEquiRepresentativeRecorded(memory,
        orSpan("(or9[a,7])"), orSpan("main")));
    // A scope with no ledger row is a defined miss.
    ASSERT_FALSE(analyzer.caseLedger.orEquiRepresentativeRecorded(memory,
        orSpan("(or98[a,7])"), orSpan("elsewhere")));

    // Clear: the row leaves; a second clear is the defined no-op.
    analyzer.caseLedger.clearProcessedOr(memory, recId, gl::NameMap::MAIN_ID);
    ASSERT_FALSE(analyzer.caseLedger.orEquiRepresentativeRecorded(memory,
        orSpan("(or98[a,7])"), orSpan("main")));
    analyzer.caseLedger.clearProcessedOr(memory, recId, gl::NameMap::MAIN_ID);
}

// I-219's ledger rows are (canonical id, original id) pairs: a probe is an
// exact id test on either half, and the class-change seam
// (recanonicalizeProcessedOrLedger) is what keeps the canonical half current
// — a class formed AFTER the record is invisible until the seam runs.
TEST(prover, or_unique_ledger_recanonicalizes_on_class_change) {
    gl::tests::ProverFixture analyzer("Peano");
    gl::Memory memory;
    ASSERT_EQ(((memory).nameMap).encode("main"), gl::NameMap::MAIN_ID);

    const gl::NameId recId = ((memory).nameMap).encode("(or98[b,7])");
    analyzer.caseLedger.recordProcessedOr(memory, recId, gl::NameMap::MAIN_ID);   // no classes yet
    {
        const int32_t row = ((memory).processedOrLedger).lookup(gl::NameMap::MAIN_ID);
        ASSERT_NE(row, 0);
        ASSERT_EQ(((memory).processedOrLedger).runLen(row), 1);
        const int64_t v = ((memory).processedOrLedger).valueAt(row, 0);
        ASSERT_EQ(gl::CaseLedger::ledgerOriginalId(v), recId);
        ASSERT_EQ(gl::CaseLedger::ledgerCanonicalId(v), recId);   // nothing substitutes
    }
    ASSERT_TRUE(analyzer.caseLedger.orEquiRepresentativeRecorded(memory,
        orSpan("(or98[b,7])"), orSpan("main")));
    ASSERT_FALSE(analyzer.caseLedger.orEquiRepresentativeRecorded(memory,
        orSpan("(or98[a,7])"), orSpan("main")));

    gl::EquivalenceClass cls;
    cls.setMembersFromNames({ "a", "b" }, ((memory).nameMap));
    memory.assignClassesById(gl::NameMap::MAIN_ID, { cls });
    seedFixtureCanonicalClasses(memory);
    // The class store changed behind the ledger: stale until the seam runs.
    ASSERT_FALSE(analyzer.caseLedger.orEquiRepresentativeRecorded(memory,
        orSpan("(or98[a,7])"), orSpan("main")));
    // The canonical text is minted by this call; look it up afterwards.
    const gl::NameId canonNow =
        analyzer.caseLedger.canonicalOrIdAtScope(memory, recId, orSpan("main"));
    ASSERT_NE(canonNow, 0);
    ASSERT_EQ(canonNow, ((memory).nameMap).lookup(orSpan("(or98[a,7])")));

    analyzer.caseLedger.recanonicalizeProcessedOrLedger(memory, gl::NameMap::MAIN_ID);
    ASSERT_TRUE(analyzer.caseLedger.orEquiRepresentativeRecorded(memory,
        orSpan("(or98[a,7])"), orSpan("main")));
    ASSERT_TRUE(analyzer.caseLedger.orEquiRepresentativeRecorded(memory,
        orSpan("(or98[b,7])"), orSpan("main")));
    ASSERT_FALSE(analyzer.caseLedger.orEquiRepresentativeRecorded(memory,
        orSpan("(or98[a,8])"), orSpan("main")));
    {
        const int32_t row = ((memory).processedOrLedger).lookup(gl::NameMap::MAIN_ID);
        ASSERT_EQ(((memory).processedOrLedger).runLen(row), 1);
        const int64_t v = ((memory).processedOrLedger).valueAt(row, 0);
        ASSERT_EQ(gl::CaseLedger::ledgerOriginalId(v), recId);
        ASSERT_EQ(gl::CaseLedger::ledgerCanonicalId(v),
                  ((memory).nameMap).lookup(orSpan("(or98[a,7])")));
    }
    // Idempotent: a second seam run on unchanged classes rewrites nothing.
    analyzer.caseLedger.recanonicalizeProcessedOrLedger(memory, gl::NameMap::MAIN_ID);
    ASSERT_TRUE(analyzer.caseLedger.orEquiRepresentativeRecorded(memory,
        orSpan("(or98[a,7])"), orSpan("main")));
    // A scope without rows is the defined no-op.
    analyzer.caseLedger.recanonicalizeProcessedOrLedger(memory,
        ((memory).nameMap).encode("elsewhere"));
    // Clearing keys on the original id, canonical half notwithstanding.
    analyzer.caseLedger.clearProcessedOr(memory, recId, gl::NameMap::MAIN_ID);
    ASSERT_FALSE(analyzer.caseLedger.orEquiRepresentativeRecorded(memory,
        orSpan("(or98[a,7])"), orSpan("main")));
    ASSERT_FALSE(analyzer.caseLedger.orEquiRepresentativeRecorded(memory,
        orSpan("(or98[b,7])"), orSpan("main")));
}

// The or-uniqueness gate at the deposit seam: the FIRST representative of
// an or's equi class gets full processing (K compacts, ordis park, the
// processed-or ledger row); an equi variant deposited AFTER the class
// merge stays a PASSIVE statement — registered, but no new park keys, no
// fullyDisintegrated stamp, no ledger row. An unrelated or of the same
// operator still processes fully.
TEST(prover, or_unique_gate_suppresses_equi_variant_deposit) {
    gl::tests::ProverFixture ea("Peano");
    ea.parameters.trackHistory = true;
    gl::Memory& lb = ea.body;
    ASSERT_EQ(((lb).nameMap).encode("main"), gl::NameMap::MAIN_ID);
    // An LB with goals keeps the or consumption live (a goal-less LB
    // neither opens nor parks — I-206 clause 6).
    ((lb).intToBeProved).assignSet(gl::packStatementKey(
        gl::encodeExpression(orSpan("(in3[1,2,3])"), orSpan("main"), ((lb).nameMap)).originalId, gl::NameMap::MAIN_ID),
        nullptr, 0);
    // Hand-registered binary or (the registry-or rig): two
    // operator-application leaves over config-level cores.
    gl::CompiledDefinitionFixture::compiledExpressions(ea.definitions)["or97"] =
        gl::LogicalEntity("or",
            { "(in[u_1,u_2])", "(in2[u_1,u_3,u_2])" },
            "(or97[u_1,u_2,u_3])", 3);
    ea.orRuleCompiler.compileOrKRules("or97");
    ea.orRuleCompiler.preMintReducedOrs();

    const int lv[1] = { 0 };
    const gl::TransientOrigin origin{
        true, gl::OriginTag::taskFormulation, nullptr, 0 };
    const auto orDeposit = [&](const char* expr) {
        ea.statementMemory.addExprToMemoryBlock(orSpan(expr), lb, 0, 1, lv, 1,
            origin, -1, -1, orSpan("main"), /*doNotDisintegrate=*/false,
            /*allowOrDisintegration=*/false, nullptr);
    };

    // First representative: full processing — parked (no route admits)
    // and recorded in the ledger.
    orDeposit("(or97[b,7,9])");
    const gl::StatementFlags* firstRow = gl::lookupStatementFlags(
        ((lb).intKnownStatements), ((lb).nameMap),
        std::string("(or97[b,7,9])"), std::string("main"));
    ASSERT_TRUE(firstRow != nullptr);
    ASSERT_TRUE(firstRow->fullyDisintegrated);
    const int32_t parkedAfterFirst =
        ((((lb).overallHashMemory)).rejectedMapOrdis).count();
    ASSERT_TRUE(parkedAfterFirst >= 1);
    ASSERT_TRUE(ea.caseLedger.orEquiRepresentativeRecorded(lb,
        orSpan("(or97[b,7,9])"), orSpan("main")));

    // The class {a, b} (canonical `a`) lands AFTER the first or was
    // processed — the leak window the gate closes.
    gl::EquivalenceClass cls;
    cls.setMembersFromNames({ "a", "b" }, ((lb).nameMap));
    lb.assignClassesById(gl::NameMap::MAIN_ID, { cls });
    seedFixtureCanonicalClasses(lb);

    // The equi variant: the door leaves it unchanged (`a` is canonical),
    // Site F passes (new text) — only the or-uniqueness gate suppresses.
    orDeposit("(or97[a,7,9])");
    const gl::StatementFlags* variantRow = gl::lookupStatementFlags(
        ((lb).intKnownStatements), ((lb).nameMap),
        std::string("(or97[a,7,9])"), std::string("main"));
    ASSERT_TRUE(variantRow != nullptr);            // registered (passive)
    ASSERT_FALSE(variantRow->fullyDisintegrated);  // never disintegrated
    ASSERT_EQ(((((lb).overallHashMemory)).rejectedMapOrdis).count(),
              parkedAfterFirst);                   // no new park keys
    {
        const int32_t ledgerRow = ((lb).processedOrLedger).lookup(
            gl::NameMap::MAIN_ID);
        ASSERT_TRUE(ledgerRow != 0);
        ASSERT_EQ(((lb).processedOrLedger).runLen(ledgerRow), 1);
    }

    // An unrelated or of the same operator processes fully: new park
    // keys, second ledger row.
    orDeposit("(or97[c,7,9])");
    ASSERT_TRUE(((((lb).overallHashMemory)).rejectedMapOrdis).count()
                > parkedAfterFirst);
    {
        const int32_t ledgerRow = ((lb).processedOrLedger).lookup(
            gl::NameMap::MAIN_ID);
        ASSERT_EQ(((lb).processedOrLedger).runLen(ledgerRow), 2);
    }
}

// lookupOrCohortId — the non-minting twin of mintOrCohortId: recovers a
// minted cohort id byte-exactly, and returns the defined miss 0 for a
// never-minted pair or a zero component.
TEST(prover, symmetric_or_deposits_keep_one_cohort_and_two_facts) {
    for (bool reversed : {false, true}) {
        gl::tests::ProverFixture analyzer("Peano");
        auto& memory = analyzer.body;
        analyzer.parameters.trackHistory = true;
        analyzer.parameters.max_or_depth = 3;
        const std::string name = analyzer.orRuleCompiler.findOrMintOrOperator(
            {"(in2[u_1,u_2,u_3])", "(=[u_1,u_2])", "(in2[u_2,u_1,u_3])"}, 3);
        analyzer.orRuleCompiler.preMintReducedOrs();
        ((memory).intToBeProved).assignSet(gl::packStatementKey(
            ((memory).nameMap).encode("(in[x,N])"), gl::NameMap::MAIN_ID), nullptr, 0);
        const std::string forward = "(" + name + "[a,b,f])";
        const std::string backward = "(" + name + "[b,a,f])";
        const std::string first = reversed ? backward : forward;
        const std::string second = reversed ? forward : backward;
        const int levels[] = {0};
        const gl::TransientOrigin origin{true, gl::OriginTag::taskFormulation, nullptr, 0};
        for (const auto& text : {first, second})
            analyzer.statementMemory.addExprToMemoryBlock(gl::StrSpan(text), memory, 0, 1,
                levels, 1, origin, -1, -1, orSpan("main"), false, true);
        const auto* firstFlags = gl::lookupStatementFlags(((memory).intKnownStatements),
            ((memory).nameMap), first, "main");
        const auto* secondFlags = gl::lookupStatementFlags(((memory).intKnownStatements),
            ((memory).nameMap), second, "main");
        ASSERT_TRUE(firstFlags != nullptr && firstFlags->fullyDisintegrated);
        ASSERT_TRUE(secondFlags != nullptr && !secondFlags->fullyDisintegrated);
        ASSERT_EQ(((memory).orDisjunctCount).count(), 1);
        ASSERT_EQ(((memory).orImplications).sources.count(), 1);
        const auto row = ((memory).processedOrLedger).lookup(gl::NameMap::MAIN_ID);
        ASSERT_NE(row, 0);
        ASSERT_EQ(((memory).processedOrLedger).runLen(row), 1);
        ASSERT_TRUE(analyzer.caseLedger.orEquiRepresentativeRecorded(memory, gl::StrSpan(second), orSpan("main")));
    }
}

TEST(prover, nested_symmetric_or_consumption_marks_only_one_source) {
    gl::tests::ProverFixture analyzer("Peano");
    const std::string name = analyzer.orRuleCompiler.findOrMintOrOperator(
        {"(in2[u_1,u_2,u_3])", "(in2[u_2,u_1,u_3])"}, 3);
    gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions)["and999"] = gl::LogicalEntity("and",
        {"(" + name + "[u_1,u_2,u_3])", "(" + name + "[u_2,u_1,u_3])"},
        "(and999[u_1,u_2,u_3])", 3);
    analyzer.orRuleCompiler.preMintReducedOrs();
    auto& arena = gl::genScratchArenas().forSlot(gl::genScratchArenas().slotCount() - 1);
    gl::DisintProducts products(&arena);
    analyzer.disintegrationProcessor.disintegrateExpr2(orSpan("(and999[a,b,f])"), analyzer.body, 0,
        true, orSpan("main"), products, false, false, nullptr, 0, true);
    ASSERT_TRUE(products.orSources.contains(gl::StrSpan("(" + name + "[a,b,f])"), orSpan("main")));
    ASSERT_FALSE(products.orSources.contains(gl::StrSpan("(" + name + "[b,a,f])"), orSpan("main")));
}

TEST(memory, lookup_or_cohort_id_matches_mint) {
    gl::Memory memory;
    const int32_t p = ((memory).lbStateInterner).encode(gl::StrSpan("main", 4));
    const int32_t s = ((memory).lbStateInterner).encode(orSpan("(or97[b,7,9])"));
    const int32_t minted = gl::mintOrCohortId(((memory).lbStateInterner), p, s);
    ASSERT_EQ(gl::lookupOrCohortId(((memory).lbStateInterner), p, s), minted);
    ASSERT_EQ(gl::lookupOrCohortId(((memory).lbStateInterner), s, p), 0);
    ASSERT_EQ(gl::lookupOrCohortId(((memory).lbStateInterner), 0, s), 0);
    ASSERT_EQ(gl::lookupOrCohortId(((memory).lbStateInterner), p, 0), 0);
}

// Duplicate-or-cohort retirement: two OPEN cohorts at one parent whose
// signatures become equi variants (class {a,b} lands after both opened)
// collapse to the most-advanced one — the keeper has more statements
// registered under its _ordis_ branch scopes. The loser's live branch is
// queued for wipe and leaves the live registry, its scheduling rows and
// ledger row leave; the keeper's rows and the loser's history stay. A
// second drain is the defined no-op.
TEST(prover, retire_duplicate_or_cohorts_keeps_most_advanced) {
    gl::tests::ProverFixture analyzer("Peano");
    gl::Memory memory;
    ASSERT_EQ(((memory).nameMap).encode("main"), gl::NameMap::MAIN_ID);

    const char* textB = "(or97[b,7,9])";
    const char* textA = "(or97[a,7,9])";
    const gl::NameId idB = ((memory).nameMap).encode(textB);
    const gl::NameId idA = ((memory).nameMap).encode(textA);

    // Both ors were fully processed pre-merge: ledger rows + open cohorts.
    analyzer.caseLedger.recordProcessedOr(memory, idB, gl::NameMap::MAIN_ID);
    analyzer.caseLedger.recordProcessedOr(memory, idA, gl::NameMap::MAIN_ID);
    const int32_t parentId =
        ((memory).lbStateInterner).encode(gl::StrSpan("main", 4));
    const int32_t sigB = ((memory).lbStateInterner).encode(orSpan(textB));
    const int32_t sigA = ((memory).lbStateInterner).encode(orSpan(textA));
    const int32_t cidB =
        gl::mintOrCohortId(((memory).lbStateInterner), parentId, sigB);
    const int32_t cidA =
        gl::mintOrCohortId(((memory).lbStateInterner), parentId, sigA);
    ((memory).orDisjunctCount).insert(cidB, 2);
    ((memory).orDisjunctCount).insert(cidA, 2);
    const int32_t wB =
        ((memory).lbStateInterner).encode(orSpan("((in2[b,9,7]))"));
    const int32_t wA =
        ((memory).lbStateInterner).encode(orSpan("((in2[a,9,7]))"));
    ((memory).orPendingBranches).insertSorted(cidB, wB,
        gl::DecodedIdLess{ &((memory).lbStateInterner) });
    ((memory).orPendingBranches).insertSorted(cidA, wA,
        gl::DecodedIdLess{ &((memory).lbStateInterner) });
    ((memory).orPendingLevels).insertSorted(cidB, 0,
        [](int32_t x, int32_t y) { return x < y; });
    ((memory).orPendingLevels).insertSorted(cidA, 0,
        [](int32_t x, int32_t y) { return x < y; });
    ((memory).orStarterPick).upsert(cidB, wB);

    // One live branch each; the keeper (A) carries two registered branch
    // statements, the loser (B) none.
    const std::string branchB =
        std::string("main_boundary_ordis_") + textB + "_(in[b,7])";
    const std::string branchA =
        std::string("main_boundary_ordis_") + textA + "_(in[a,7])";
    const gl::NameId bvB = ((memory).nameMap).encode(branchB);
    const gl::NameId bvA = ((memory).nameMap).encode(branchA);
    ASSERT_EQ(((memory).nameMap).parentOf(bvB), gl::NameMap::MAIN_ID);
    ((memory).orLiveBranches).mint(bvB);
    ((memory).orLiveBranches).mint(bvA);
    for (const char* st : { "(in3[1,2,3])", "(in3[4,5,6])" }) {
        ((memory).intEncodedStatements).push_back(gl::encodeExpression(
            orSpan(st), gl::StrSpan(branchA), ((memory).nameMap)));
    }

    // Pre-merge: two distinct canonical spellings — the drain must not
    // touch anything.
    analyzer.caseProcessor.retireDuplicateOrCohorts(memory);
    ASSERT_NE(((memory).orDisjunctCount).lookup(cidB), 0);
    ASSERT_NE(((memory).orDisjunctCount).lookup(cidA), 0);

    // The merge lands: {a, b}, canonical `a`. `assignClassesById` is the
    // test-only class door; production stores a merged class through
    // `updateEquivalenceClasses`, whose tail runs the I-219 seam that
    // re-keys the ledger rows — run it here for the same effect, since
    // the drain groups by the ledger's canonical half.
    gl::EquivalenceClass cls;
    cls.setMembersFromNames({ "a", "b" }, ((memory).nameMap));
    memory.assignClassesById(gl::NameMap::MAIN_ID, { cls });
    seedFixtureCanonicalClasses(memory);
    analyzer.caseLedger.recanonicalizeProcessedOrLedger(memory, gl::NameMap::MAIN_ID);

    analyzer.caseProcessor.retireDuplicateOrCohorts(memory);

    // Loser B retired: scheduling rows gone, live branch queued for wipe
    // and out of the live registry, ledger row cleared.
    ASSERT_EQ(((memory).orDisjunctCount).lookup(cidB), 0);
    ASSERT_EQ(((memory).orPendingBranches).lookup(cidB), 0);
    ASSERT_EQ(((memory).orPendingLevels).lookup(cidB), 0);
    ASSERT_TRUE(((memory).orStarterPick).find(cidB) == nullptr);
    ASSERT_NE(((memory).pendingWipeScopes).lookup(bvB), 0);
    ASSERT_EQ(((memory).orLiveBranches).lookup(bvB), 0);
    ASSERT_TRUE(((memory).mutatedThisBurst));

    // Keeper A intact.
    ASSERT_NE(((memory).orDisjunctCount).lookup(cidA), 0);
    ASSERT_NE(((memory).orPendingBranches).lookup(cidA), 0);
    ASSERT_NE(((memory).orLiveBranches).lookup(bvA), 0);
    {
        const int32_t row =
            ((memory).processedOrLedger).lookup(gl::NameMap::MAIN_ID);
        ASSERT_NE(row, 0);
        ASSERT_EQ(((memory).processedOrLedger).runLen(row), 1);
        ASSERT_EQ(gl::CaseLedger::ledgerOriginalId(
                      ((memory).processedOrLedger).valueAt(row, 0)), idA);
    }

    // Idempotent: a second drain finds one open cohort per group.
    analyzer.caseProcessor.retireDuplicateOrCohorts(memory);
    ASSERT_NE(((memory).orDisjunctCount).lookup(cidA), 0);
}


namespace {
    struct InheritedOrFixture {
        gl::tests::ProverFixture analyzer{"Peano"};
        gl::Memory memory;
        std::string name, expression;
        const std::string leaf = "(preorder95[1,2,4,5])";

        explicit InheritedOrFixture(int leafCount = 2) {
            analyzer.parameters.trackHistory = true;
            analyzer.parameters.max_origin_per_expr = 16;
            analyzer.parameters.max_or_depth = 2;
            ((memory).isActive) = true;
            ((memory).level) = 1;
            ((memory).parentMemory) = &analyzer.body;
            ASSERT_EQ(((memory).nameMap).encode("main"), gl::NameMap::MAIN_ID);
            gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions)["preorder95"] =
                gl::LogicalEntity("existence",
                    {"(in[w1,u_1])", "(in3[u_3,w1,u_4,u_2])"},
                    "(preorder95[u_1,u_2,u_3,u_4])", 4);
            analyzer.supportingRuleCompiler.compileDisintegrationPackages();
            analyzer.existenceRuleCompiler.compileExistenceImplications("preorder95");
            std::vector<std::string> leaves{
                "(preorder95[u_1,u_2,u_3,u_4])", "(in[u_3,u_4])"};
            if (leafCount == 3) leaves.push_back("(in[u_4,u_3])");
            name = analyzer.orRuleCompiler.findOrMintOrOperator(leaves, 4);
            analyzer.orRuleCompiler.preMintReducedOrs();
            expression = "(" + name + "[1,2,4,5])";
        }

        void goal() {
            const auto encoded = gl::encodeExpression(orSpan("(in[91,92])"),
                orSpan("main"), ((memory).nameMap));
            ((memory).intToBeProved).assignSet(gl::StatementKey{
                encoded.originalId, encoded.validityId}, nullptr, 0);
        }

        int64_t demand() {
            const int64_t key = gl::mintTemplateKey(((memory).templateInterner),
                ((memory).nameMap), leaf, "main");
            const int32_t rule = ((memory).ruleInterner).encode("(>[]dummy(d[1]))");
            const int levels[] = {0};
            auto& arena = gl::genScratchArenas().forSlot(
                gl::genScratchArenas().slotCount() - 1);
            gl::insertAdmissionOrdis2IdsBlob(
                ((((memory).overallHashMemory)).admissionMapOrdis2),
                key, rule, levels, 1, ((memory).ruleInterner), arena);
            return key;
        }

        void deposit(bool suppress = false, bool forceOpen = false) {
            const int levels[] = {0};
            const gl::TransientOrigin origin{
                true, gl::OriginTag::taskFormulation, nullptr, 0};
            analyzer.statementMemory.addExprToMemoryBlock(gl::StrSpan(expression), memory,
                0, 3, levels, 1, origin, -1, -1, orSpan("main"),
                suppress, forceOpen);
        }

        void receive(int32_t flagCode = gl::kMailStatementFlagNone) {
            const int32_t original = gl::mailInterner().mint(gl::StrSpan(expression));
            const int32_t validity = gl::mailInterner().mint(orSpan("main"));
            const int levels[] = {0};
            ((memory).mailIn).insertStatement(original, validity, levels, 1);
            if (flagCode != gl::kMailStatementFlagNone)
                ((memory).mailIn).mergeStatementFlag(gl::packOriginKey(original, validity),
                    gl::packMailStatementFlag(flagCode, -1));
            gl::addMailOriginRecord(((memory).mailIn).origins_,
                gl::packOriginKey(original, validity),
                static_cast<uint8_t>(gl::OriginTag::taskFormulation),
                nullptr, 0, 16);
            analyzer.logicBlockOperations.standardProcessing(memory, &((memory).mailIn),
                ((memory).sameIterationInternalMail),
                ((memory).nextIterationInternalMail), -1);
            // The phase caller clears the external inbox after absorption.
            ((memory).mailIn).clear();
            ASSERT_TRUE(((memory).mailIn).empty());
        }

        int32_t cohort() {
            return gl::lookupOrCohortId(((memory).lbStateInterner),
                ((memory).lbStateInterner).lookup("main"),
                ((memory).lbStateInterner).lookup(expression));
        }

        bool known(const std::string& text, gl::NameId validity) const {
            const gl::NameId original = ((memory).nameMap).lookup(gl::StrSpan(text));
            return original != 0 && ((memory).intKnownStatements).lookup(
                gl::StatementKey{original, validity}) != 0;
        }

        void internal(const std::string& text, gl::NameId validity) {
            const std::string scope = ((memory).nameMap).decode(validity);
            const int levels[] = {0};
            gl::insertInternalStatement(((memory).sameIterationInternalMail),
                ((memory).nameMap), gl::StrSpan(text), gl::StrSpan(scope), levels, 1);
            gl::addInternalMailOrigin(((memory).sameIterationInternalMail),
                ((memory).originInterner), gl::StrSpan(text), gl::StrSpan(scope),
                gl::OriginTag::taskFormulation, nullptr, 0, 16);
        }

        void absorbInternal() {
            analyzer.logicBlockOperations.standardProcessing(memory, nullptr,
                ((memory).sameIterationInternalMail),
                ((memory).nextIterationInternalMail), -1);
            analyzer.logicBlockOperations.standardProcessing(memory, nullptr,
                ((memory).nextIterationInternalMail),
                ((memory).sameIterationInternalMail), -1);
        }
    };
}

TEST(or_mail, recipient_eligibility_and_suppression) {
    for (int mode = 0; mode < 5; ++mode) {
        InheritedOrFixture f;
        if (mode == 1 || mode == 4) f.goal();
        if (mode == 2 || mode == 3) ((f.memory).primedForContradiction) = true;
        if (mode == 3) ((f.memory).isActive) = false;
        f.deposit(mode == 4, true);
        const bool eligible = mode == 1 || mode == 2;
        ASSERT_EQ(((((f.memory).overallHashMemory)).originals).count(), 0);
        ASSERT_EQ(((f.memory).orImplications).sources.count() > 0, eligible);
        ASSERT_EQ(((f.memory).orDisjunctCount).count() > 0, eligible);
        if (!eligible) {
            ASSERT_EQ(((((f.memory).overallHashMemory)).rejectedMapOrdis).count(), 0);
            ASSERT_EQ(((((f.memory).overallHashMemory)).rejectedMapOrdis2).count(), 0);
        }
        ASSERT_TRUE(f.known(f.expression, gl::NameMap::MAIN_ID));
    }
}

TEST(or_mail, external_absorption_opens_demanded_cases_and_keeps_carrier_nonlocal) {
    for (bool contradiction : {false, true}) {
        InheritedOrFixture f;
        if (contradiction) ((f.memory).primedForContradiction) = true;
        else f.goal();
        const int64_t demand = f.demand();
        f.receive();
        ASSERT_TRUE(f.cohort() != 0);
        ASSERT_EQ(*((f.memory).orDisjunctCount).find(f.cohort()), 2);
        ASSERT_TRUE(((((f.memory).overallHashMemory)).admissionMapOrdis2).lookup(demand) != 0);
        const auto* flags = gl::lookupStatementFlags(
            ((f.memory).intKnownStatements), ((f.memory).nameMap), f.expression, "main");
        ASSERT_TRUE(flags != nullptr);
        ASSERT_FALSE(flags->local);
        ASSERT_TRUE(flags->fullyDisintegrated);
        ASSERT_EQ(((((f.memory).workingMemory)).originals).count(), 0);
        ASSERT_EQ(((f.memory).orImplications).sources.count(), 1);
        ASSERT_EQ(((((f.memory).localHashMemory)).originals).count(), 0);
        f.analyzer.caseProcessor.drainPendingOrReleases(f.memory);
        const auto seeds = gl::decodeInternalMailStatements(
            ((f.memory).sameIterationInternalMail), ((f.memory).nameMap));
        ASSERT_EQ(seeds.size(), static_cast<std::size_t>(1));
        ASSERT_EQ(seeds[0].first.original, f.leaf);
        ASSERT_EQ(seeds[0].second, (std::set<int>{-1}));
        const int32_t sourceLevels = f.memory.orPendingLevels.lookup(f.cohort());
        ASSERT_TRUE(sourceLevels != 0);
        ASSERT_EQ(f.memory.orPendingLevels.valueAt(sourceLevels, 0), 0);
        ASSERT_TRUE(seeds[0].first.validityName.find("_ordis_") != std::string::npos);
        const auto history = gl::decodeInternalMailOrigins(
            ((f.memory).sameIterationInternalMail), ((f.memory).originInterner));
        ASSERT_FALSE(history.empty());
        f.receive();
        ASSERT_EQ(((f.memory).orDisjunctCount).count(), 1);
    }
}

// Flag 6 (D-528): an OR arriving with the
// unconditional-ordis flag opens its case split without any demand where the
// receiver has proof work (a goal, or an active primed contradiction block);
// without proof work it stays a passive statement. The carrier is never staged
// for the flag again and never parks.
TEST(or_mail, unconditional_ordis_flag_opens_without_demand_only_with_proof_work) {
    for (int mode = 0; mode < 3; ++mode) {
        InheritedOrFixture f;
        if (mode == 1) f.goal();
        if (mode == 2) ((f.memory).primedForContradiction) = true;
        f.receive(gl::kMailStatementFlagUnconditionalOrdis);
        const bool proofWork = mode != 0;
        ASSERT_EQ(((f.memory).orDisjunctCount).count(), proofWork ? 1 : 0);
        if (proofWork) ASSERT_EQ(*((f.memory).orDisjunctCount).find(f.cohort()), 2);
        ASSERT_EQ(((((f.memory).overallHashMemory)).rejectedMapOrdis).count(), 0);
        ASSERT_EQ(((((f.memory).overallHashMemory)).rejectedMapOrdis2).count(), 0);
        ASSERT_TRUE(((f.memory).lbMemory.pendingUnconditionalOrdis).empty());
        ASSERT_EQ(((f.memory).lbMemory.forcedOrCohorts).count(), proofWork ? 1 : 0);
        ASSERT_TRUE(f.known(f.expression, gl::NameMap::MAIN_ID));
        const auto* flags = gl::lookupStatementFlags(
            ((f.memory).intKnownStatements), ((f.memory).nameMap), f.expression, "main");
        ASSERT_TRUE(flags != nullptr);
        ASSERT_FALSE(flags->local);
    }
}

// Forced case assumptions (D-527): the case a
// forced cohort releases carries the forced signal; a case of a cohort opened
// by demand does not.
TEST(or_mail, release_forces_only_the_cases_of_premise_or_cohorts) {
    for (bool premiseOr : {true, false}) {
        InheritedOrFixture f;
        f.goal();
        if (!premiseOr) f.demand();
        f.receive(premiseOr ? gl::kMailStatementFlagUnconditionalOrdis
                            : gl::kMailStatementFlagNone);
        ASSERT_EQ(((f.memory).orDisjunctCount).count(), 1);
        ASSERT_EQ(((f.memory).lbMemory.forcedOrCohorts).count(), premiseOr ? 1 : 0);
        f.analyzer.caseProcessor.drainPendingOrReleases(f.memory);
        const auto seeds = gl::decodeInternalMailStatements(
            ((f.memory).sameIterationInternalMail), ((f.memory).nameMap));
        ASSERT_EQ(seeds.size(), static_cast<std::size_t>(1));
        const auto signals = gl::decodeInternalMailDisintegrationSignals(
            ((f.memory).sameIterationInternalMail), ((f.memory).nameMap));
        bool forcedSeed = false;
        for (const auto& row : signals)
            if (row.first == seeds[0].first) forcedSeed = row.second.forced;
        ASSERT_EQ(forcedSeed, premiseOr);
    }
}

TEST(or_mail, external_supply_parks_then_demand_wakes_known_statement) {
    InheritedOrFixture f;
    ((f.memory).primedForContradiction) = true;
    f.receive();
    ASSERT_EQ(((f.memory).orDisjunctCount).count(), 0);
    ASSERT_TRUE(((((f.memory).overallHashMemory)).rejectedMapOrdis2).count() > 0);
    ASSERT_TRUE(f.known(f.expression, gl::NameMap::MAIN_ID));
    f.demand();
    f.analyzer.rejectedSourceReplay.revisitRejectedOrdis2(gl::StrSpan(f.leaf),
        f.memory, orSpan("main"));
    ASSERT_FALSE(f.known(f.expression, gl::NameMap::MAIN_ID));
    ASSERT_FALSE(((f.memory).sameIterationInternalMail).empty());
    f.absorbInternal();
    ASSERT_TRUE(f.cohort() != 0);
    ASSERT_EQ(*((f.memory).orDisjunctCount).find(f.cohort()), 2);
}

TEST(or_mail, selected_exclusions_cite_history_only_compacts_including_reduced_ors) {
    for (int variant = 0; variant < 4; ++variant) {
        InheritedOrFixture f(2 + variant % 2);
        f.goal();
        if (variant >= 2) {
            f.internal("(=[5,6])", gl::NameMap::MAIN_ID);
            f.absorbInternal();
            f.expression = "(" + f.name + "[1,2,4,6])";
        }
        f.receive();
        ASSERT_EQ(((f.memory).orImplications).sources.count(), 1);
        ASSERT_EQ(((f.memory).compactExpansions).count(), 0);
        ASSERT_EQ(((((f.memory).overallHashMemory)).originals).count(), 0);
        f.internal("!(in[4,5])", gl::NameMap::MAIN_ID);
        f.absorbInternal();
        const auto origins = gl::decodeOriginMapSorted(
            ((f.memory).exprOriginMap), ((f.memory).originInterner));
        int selected = 0;
        for (const auto& entry : origins) {
            const std::string& compact = entry.first.first;
            if (compact.find("(implication") != 0) continue;
            ASSERT_FALSE(f.known(compact, gl::NameMap::MAIN_ID));
            if (variant >= 2) {
                const auto args = ce::getArgs(compact);
                ASSERT_TRUE(std::find(args.begin(), args.end(), "6") == args.end());
            }
            bool disintegration = false, expansion = false;
            for (const auto& line : entry.second)
                if (line.first == "disintegration") disintegration = true;
            for (const auto& rule : origins) for (const auto& line : rule.second)
                if (line.first == "expansion" && line.second.size() == 1
                    && line.second[0].original == compact
                    && line.second[0].validityName == entry.first.second)
                    expansion = true;
            ASSERT_TRUE(disintegration && expansion);
            ++selected;
        }
        ASSERT_TRUE(selected > 0);
    }
}

TEST(or_mail, depth_limit_keeps_exclusion_source_without_cases_or_parks) {
    InheritedOrFixture f;
    f.goal();
    f.analyzer.parameters.max_or_depth = 0;
    f.demand();
    f.receive();
    ASSERT_EQ(((((f.memory).overallHashMemory)).originals).count(), 0);
    ASSERT_EQ(((f.memory).orImplications).sources.count(), 1);
    ASSERT_EQ(((f.memory).orDisjunctCount).count(), 0);
    ASSERT_EQ(((((f.memory).overallHashMemory)).rejectedMapOrdis).count(), 0);
    ASSERT_EQ(((((f.memory).overallHashMemory)).rejectedMapOrdis2).count(), 0);
}

namespace {
    /// @brief Deposit the fixture's OR at a fresh child scope of `main`.
    ///
    /// @details A status-0 local deposit, optionally flagged as an
    /// integration premise (D-460).
    ///
    /// @param f                  The fixture.
    /// @param payload            The child scope's payload.
    /// @param integrationPremise The door's integration-premise flag.
    /// @return The child scope's validity name.
    std::string depositOrAtChildScope(InheritedOrFixture& f, const char* payload,
                                      bool integrationPremise) {
        const gl::NameId scope = ((f.memory).nameMap).encodePush(gl::NameMap::MAIN_ID,
            gl::StrSpan(payload, static_cast<int32_t>(std::strlen(payload))));
        const std::string scopeName = ((f.memory).nameMap).decode(scope);
        const int levels[] = {0};
        const gl::TransientOrigin origin{
            true, gl::OriginTag::premiseElement, nullptr, 0};
        f.analyzer.statementMemory.addExprToMemoryBlock(gl::StrSpan(f.expression), f.memory,
            -1, 0, levels, 1, origin, -1, -1, gl::StrSpan(scopeName),
            /*doNotDisintegrate=*/false, /*allowOrDisintegration=*/false,
            /*registeredOut=*/nullptr, /*recursionHead=*/false,
            /*convergenceMailOut=*/nullptr, /*replayKinds=*/0,
            /*replayIntegrationDepth=*/-1, integrationPremise);
        return scopeName;
    }

    /// @brief The cohort id of the fixture's OR under `parent`, 0 if none.
    ///
    /// @param f      The fixture.
    /// @param parent The cohort's parent validity name.
    /// @return The interned cohort id, or 0.
    int32_t cohortAt(InheritedOrFixture& f, const std::string& parent) {
        return gl::lookupOrCohortId(((f.memory).lbStateInterner),
            ((f.memory).lbStateInterner).lookup(gl::StrSpan(parent)),
            ((f.memory).lbStateInterner).lookup(gl::StrSpan(f.expression)));
    }
}

TEST(or_mail, integration_premise_or_opens_without_demand) {
    // The same OR parks as an ordinary local deposit and opens its case
    // split, without any demand, as an integration premise.
    InheritedOrFixture f;
    f.goal();
    const std::string plain = depositOrAtChildScope(f, "(implication901[1,2,4,5])", false);
    ASSERT_EQ(cohortAt(f, plain), 0);
    ASSERT_TRUE(((((f.memory).overallHashMemory)).rejectedMapOrdis2).count() > 0);
    const int32_t parked = ((((f.memory).overallHashMemory)).rejectedMapOrdis2).count();

    const std::string premise = depositOrAtChildScope(f, "(implication902[1,2,4,5])", true);
    const int32_t cohort = cohortAt(f, premise);
    ASSERT_TRUE(cohort != 0);
    ASSERT_TRUE(((f.memory).orDisjunctCount).lookup(cohort) != 0);
    ASSERT_TRUE(((f.memory).pendingOrReleases).lookup(cohort) != 0);
    ASSERT_EQ(((((f.memory).overallHashMemory)).rejectedMapOrdis2).count(), parked);
}

TEST(or_mail, depth_counts_only_ordis_scopes) {
    // At max_or_depth 1 a case split opens inside an `_orint_` branch and
    // stays closed inside an `_ordis_` case (D-458).
    InheritedOrFixture f;
    f.goal();
    f.analyzer.parameters.max_or_depth = 1;
    const std::string orint = depositOrAtChildScope(f, "orint_(or77[1,2])_((in[1,2]))", true);
    ASSERT_TRUE(cohortAt(f, orint) != 0);
    const std::string ordis = depositOrAtChildScope(f, "ordis_(or77[1,2])_((in[1,2]))", true);
    ASSERT_EQ(cohortAt(f, ordis), 0);
    ASSERT_EQ(((f.memory).orDisjunctCount).count(), 1);
}

TEST(or_mail, equivalent_inherited_or_is_not_processed_twice) {
    InheritedOrFixture f;
    f.goal();
    f.deposit();
    const int32_t parks = ((((f.memory).overallHashMemory)).rejectedMapOrdis2).count();
    const int32_t rules = ((((f.memory).overallHashMemory)).originals).count();
    gl::EquivalenceClass cls;
    cls.setMembersFromNames({"5", "6"}, ((f.memory).nameMap));
    f.memory.assignClassesById(gl::NameMap::MAIN_ID, {cls});
    seedFixtureCanonicalClasses(f.memory);
    f.expression = "(" + f.name + "[1,2,4,6])";
    f.deposit();
    ASSERT_EQ(((((f.memory).overallHashMemory)).rejectedMapOrdis2).count(), parks);
    ASSERT_EQ(((((f.memory).overallHashMemory)).originals).count(), rules);
}

TEST(or_mail, inherited_cases_refute_then_converge_and_discharge_at_main) {
    InheritedOrFixture f;
    ((f.memory).primedForContradiction) = true;
    f.memory.setExprKey("__contradiction__!(in[91,92])");
    ((f.memory).contradictionTheoremId) = ((f.memory).nameMap).encode(
        "(>[1,2,4,5,91,92]" + f.expression + "(in[91,92]))");
    f.analyzer.updateGlobalSingleChapterPages.emplace();
    f.analyzer.updateGlobalSingleChapterPages->bind(&gl::staticMemory());
    f.internal("!(in[91,92])", gl::NameMap::MAIN_ID);
    f.absorbInternal();
    f.demand();
    f.receive();
    f.analyzer.caseProcessor.drainPendingOrReleases(f.memory);
    auto seeds = gl::decodeInternalMailStatements(
        ((f.memory).sameIterationInternalMail), ((f.memory).nameMap));
    ASSERT_EQ(seeds.size(), static_cast<std::size_t>(1));
    const gl::NameId first = ((f.memory).nameMap).encode(seeds[0].first.validityName);
    f.internal("(in[91,92])", first);
    f.absorbInternal();
    ASSERT_TRUE(((f.memory).isActive));
    ASSERT_TRUE(f.known("!" + f.leaf, gl::NameMap::MAIN_ID));
    const int levels[] = {0};
    const std::string firstScope = ((f.memory).nameMap).decode(first);
    f.analyzer.caseProcessor.ordisMerge(orSpan("(in[91,92])"),
        gl::StrSpan(firstScope), levels, 1, f.memory);
    ASSERT_TRUE(((f.memory).pendingDeadOrBranches).contains(first));
    f.analyzer.caseProcessor.drainDeadOrBranches(f.memory);
    ASSERT_EQ(*((f.memory).orDisjunctCount).find(f.cohort()), 1);
    f.analyzer.caseProcessor.drainPendingOrReleases(f.memory);
    seeds = gl::decodeInternalMailStatements(
        ((f.memory).sameIterationInternalMail), ((f.memory).nameMap));
    bool secondFound = false;
    gl::NameId second = 0;
    for (const auto& seed : seeds) {
        if (seed.first.original == "(in[4,5])") {
            second = ((f.memory).nameMap).encode(seed.first.validityName);
            secondFound = true;
        }
    }
    ASSERT_TRUE(secondFound);
    ASSERT_TRUE(first != second);
    f.internal("(in[91,92])", second);
    f.absorbInternal();
    ASSERT_FALSE(((f.memory).isActive));
    ASSERT_EQ(((f.memory).contradictionTheoremId), 0);
    f.analyzer.updateGlobalSingleChapterPages->seal();
    f.analyzer.updateGlobalSingleChapterPages->freePages();
    f.analyzer.updateGlobalSingleChapterPages.reset();
}


TEST(prover, unresolved_existence_cannot_open_or_park_or) {
    for (bool routeB : {false, true}) {
        gl::tests::ProverFixture analyzer("Peano");
        gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions)["or97"] = gl::LogicalEntity("or",
            {"(in2[u_1,u_2,u_3])", "(=[u_1,u_2])"},
            "(or97[u_1,u_2,u_3])", 3);
        gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions)["existence97"] = gl::LogicalEntity("existence",
            {"(in[w,u_1])", "(or97[w,u_2,u_3])"},
            "(existence97[u_1,u_2,u_3])", 3);
        analyzer.supportingRuleCompiler.compileDisintegrationPackages();
        gl::Memory memory;
        ((memory).level) = 2;
        const auto mainId = ((memory).nameMap).encode("main");
        const int levels[] = {0};
        ((memory).intToBeProved).assignSetRange(gl::packStatementKey(
            ((memory).nameMap).encode("(=[2,3])"), mainId), levels, levels + 1);
        gl::DisintProducts output(&gl::genScratchArenas().forSlot(
            gl::genScratchArenas().slotCount() - 1));
        analyzer.disintegrationProcessor.disintegrateExpr2(gl::StrSpan("(existence97[1,2,3])"), memory,
            analyzer.parameters.maxIterationNumberVariable + 1, false,
            gl::StrSpan("main", 4), output, true, routeB, levels, 1, true);
        ASSERT_EQ(output.orSources.count(), 0);
        ASSERT_TRUE(((memory).pendingOrReleases).empty());
        ASSERT_TRUE(((memory).orPendingBranches).empty());
        ASSERT_TRUE(((memory).processedOrLedger).empty());
        ASSERT_TRUE(((((memory).overallHashMemory)).rejectedMapOrdis).empty());
        ASSERT_TRUE(((((memory).overallHashMemory)).rejectedMapOrdis2).empty());
        output.statements.forEachSorted([&](gl::StrSpan statement, gl::StrSpan) {
            ASSERT_FALSE(gl::CompactExpressionSyntax::isOrCompactSpan(statement));
        });
    }
}

TEST(prover, admitted_existence_releases_only_its_admitted_or_path) {
    for (bool routeB : {false, true}) {
        gl::tests::ProverFixture analyzer("Peano");
        gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions)["or97"] = gl::LogicalEntity("or",
            {"(in2[u_1,u_2,u_3])", "(=[u_1,u_2])"},
            "(or97[u_1,u_2,u_3])", 3);
        gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions)["existence97"] = gl::LogicalEntity("existence",
            {"(in[w,u_1])", "(or97[w,u_2,u_3])"},
            "(existence97[u_1,u_2,u_3])", 3);
        analyzer.supportingRuleCompiler.compileDisintegrationPackages();
        gl::Memory memory;
        ((memory).level) = 2;
        const auto mainId = ((memory).nameMap).encode("main");
        const int levels[] = {0};
        ((memory).intToBeProved).assignSetRange(gl::packStatementKey(
            ((memory).nameMap).encode("(=[2,3])"), mainId), levels, levels + 1);
        gl::DisintProducts output(&gl::genScratchArenas().forSlot(
            gl::genScratchArenas().slotCount() - 1));
        analyzer.disintegrationProcessor.disintegrateExpr2(gl::StrSpan("(existence97[1,2,3])"), memory,
            0, true, gl::StrSpan("main", 4), output, true, routeB, levels, 1, true);
        ASSERT_EQ(output.orSources.count(), 1);
        output.orSources.forEachSorted([&](gl::StrSpan statement, gl::StrSpan) {
            ASSERT_FALSE(gl::containsSpan(statement, gl::StrSpan("it_", 3)));
            ASSERT_TRUE(gl::containsSpan(statement, gl::StrSpan("force_lev_", 10)));
        });
        if (routeB) {
            ASSERT_EQ(((memory).pendingOrReleases).count(), 1);
            ASSERT_TRUE(((((memory).overallHashMemory)).rejectedMapOrdis).empty());
        } else {
            ASSERT_TRUE(((memory).pendingOrReleases).empty());
            ASSERT_TRUE(((((memory).overallHashMemory)).rejectedMapOrdis).count() > 0);
        }
    }
}


// The inner witness's admission-set route admits nothing beyond it
// (D-482): with the outer witness admitted through its own
// demand, the inner one follows and its OR source is processed; without that
// demand neither is minted and no OR source appears.
TEST(prover, inner_witness_follows_its_admitted_outer_witness_before_or_processing) {
    for (const bool outerDemand : { true, false }) {
        gl::tests::ProverFixture analyzer("Peano");
        gl::CompiledDefinitionFixture::operators(analyzer.definitions).clear();
        gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions)["or97"] = gl::LogicalEntity("or",
            {"(in2[u_1,u_2,u_3])", "(=[u_1,u_2])"},
            "(or97[u_1,u_2,u_3])", 3);
        gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions)["existence98"] = gl::LogicalEntity("existence",
            {"(in[v,u_1])", "(or97[v,u_2,u_3])"},
            "(existence98[u_1,u_2,u_3])", 3);
        gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions)["existence97"] = gl::LogicalEntity("existence",
            {"(in[w,u_1])", "(existence98[u_1,u_2,u_3])"},
            "(existence97[u_1,u_2,u_3])", 3);
        analyzer.supportingRuleCompiler.compileDisintegrationPackages();
        gl::Memory memory;
        ((memory).level) = 2;
        const auto mainId = ((memory).nameMap).encode("main");
        const int levels[] = {0};
        ((memory).intToBeProved).assignSetRange(gl::packStatementKey(
            ((memory).nameMap).encode("(=[2,3])"), mainId), levels, levels + 1);
        const auto demand = gl::mintTemplateKey(((memory).templateInterner), ((memory).nameMap),
            "(or97[marker,2,3])", "main");
        ((((memory).overallHashMemory)).admissionSetIntegration).mint(demand);
        if (outerDemand)
            ((((memory).overallHashMemory)).admissionSetIntegration).mint(gl::mintTemplateKey(
                ((memory).templateInterner), ((memory).nameMap), "(in[marker,1])", "main"));
        gl::DisintProducts output(&gl::genScratchArenas().forSlot(
            gl::genScratchArenas().slotCount() - 1));
        analyzer.disintegrationProcessor.disintegrateExpr2(gl::StrSpan("(existence97[1,2,3])"), memory,
            0, false, gl::StrSpan("main", 4), output, true, true, levels, 1, true);
        bool admittedInnerTyping = false;
        bool admittedOuterTyping = false;
        output.statements.forEachSorted([&](gl::StrSpan statement, gl::StrSpan) {
            if (gl::equalSpans(statement, gl::StrSpan("(in[int_0_lev_2_1,1])")))
                admittedInnerTyping = true;
            if (gl::equalSpans(statement, gl::StrSpan("(in[int_0_lev_2_0,1])")))
                admittedOuterTyping = true;
        });
        ASSERT_EQ(admittedInnerTyping, outerDemand);
        ASSERT_EQ(admittedOuterTyping, outerDemand);
        ASSERT_EQ(output.orSources.count(), outerDemand ? 1 : 0);
        ASSERT_EQ(((memory).pendingOrReleases).count(), outerDemand ? 1 : 0);
        ASSERT_TRUE(((((memory).overallHashMemory)).rejectedMapOrdis).empty());
        ASSERT_TRUE(((((memory).overallHashMemory)).rejectedMapOrdis2).empty());
    }
}

// ---------------------------------------------------------------------------
// The induction hypothesis's missing expression as an OR demand
// (D-575).
// ---------------------------------------------------------------------------

namespace {

    /// @brief Span over a NUL-terminated literal (no terminator). `StrSpan`
    ///        has no `const char*` constructor: a literal converts through a
    ///        temporary `std::string`, which dies with its full expression.
    ///
    /// @param text The literal.
    /// @return The span.
    gl::StrSpan literalSpan(const char* text) {
        return gl::StrSpan(text, static_cast<int32_t>(std::strlen(text)));
    }

    /// @brief Register the synthetic 4-argument compound `preorder94` and the
    ///        3-argument `short94` in a Peano fixture (the test_memory seed
    ///        pattern) and compile their packages.
    ///
    /// @param analyzer The fixture.
    void seedHypothesisDemandDefinitions(gl::tests::ProverFixture& analyzer) {
        gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions)["preorder94"] =
            gl::LogicalEntity("existence", { "(in[w1,u_1])", "(in3[u_4,w1,u_5,u_2])" },
                "(preorder94[u_1,u_2,u_4,u_5])", 4);
        gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions)["short94"] =
            gl::LogicalEntity("existence", { "(in[w1,u_1])" }, "(short94[u_1,u_2,u_3])", 3);
        analyzer.supportingRuleCompiler.compileDisintegrationPackages();
    }

    /// @brief The ordis2 demand records of a ground text at a scope; empty
    ///        when the key was never minted.
    ///
    /// @param m     The block.
    /// @param text  The ground text.
    /// @param scope The scope name.
    /// @return The decoded record set.
    gl::AdmissionOrdis2ValueSet hypothesisDemandsAt(const gl::Memory& m, const std::string& text,
            const std::string& scope) {
        int64_t pk = 0;
        if (!gl::lookupTemplateKey(m.templateInterner, m.nameMap, text, scope, pk))
            return gl::AdmissionOrdis2ValueSet(gl::DecodedAdmissionOrdis2ValueLess{ &m.ruleInterner });
        return gl::admissionOrdis2RecordsAt(m.overallHashMemory.admissionMapOrdis2, pk, m.ruleInterner);
    }

} // namespace

// A premise is fully bound when each of its arguments lies in the chain's
// sorted fixed-argument run; one variable makes it unbound.
TEST(ordis2, fully_bound_element_matrix) {
    const gl::StrSpan fixed[5] = { literalSpan("1"), literalSpan("4"), literalSpan("7"), literalSpan("9"),
        literalSpan("int_0_lev_2_6") };
    ASSERT_TRUE(gl::AdmissionMemory::fullyBoundElement(literalSpan("(preorder[1,4,7,int_0_lev_2_6])"), fixed, 5));
    ASSERT_TRUE(gl::AdmissionMemory::fullyBoundElement(literalSpan("!(=[int_0_lev_2_6,9])"), fixed, 5));
    ASSERT_TRUE(gl::AdmissionMemory::fullyBoundElement(literalSpan("(preorder[1,4,7,9])"), fixed, 5));
    ASSERT_FALSE(gl::AdmissionMemory::fullyBoundElement(literalSpan("(in3[12,int_0_lev_2_6,9,4])"), fixed, 5));
    ASSERT_FALSE(gl::AdmissionMemory::fullyBoundElement(literalSpan("(preorder[1,4,7,int_0_lev_2_60])"), fixed, 5));
    ASSERT_FALSE(gl::AdmissionMemory::fullyBoundElement(literalSpan("(preorder[1,4,7,9])"), fixed, 3));
}

// The hypothesis's missing expression is written as a ground demand at its
// scope, with the hypothesis rule as source and no levels; the same premise
// known at the scope or at an ancestor, or outside the park-key language,
// files nothing. Knowledge at a sibling scope does not count.
TEST(prover, hypothesis_or_demand_files_only_a_missing_eligible_premise) {
    gl::tests::ProverFixture analyzer("Peano");
    seedHypothesisDemandDefinitions(analyzer);
    gl::Memory m;
    ASSERT_EQ(m.nameMap.encode("main"), gl::NameMap::MAIN_ID);
    const gl::NameId scopeId = m.nameMap.encodePush(gl::NameMap::MAIN_ID, std::string("is_(in3[marker,5,1,4])_9"));
    const gl::NameId siblingId = m.nameMap.encodePush(gl::NameMap::MAIN_ID, std::string("is_(in3[marker,5,1,4])_8"));
    const std::string scope = m.nameMap.decode(scopeId);
    m.recursionAuxyImplId = m.ruleInterner.encode("(>[11](in3[11,u_5,u_1,u_4])(preorder94[u_1,u_2,11,u_5]))");
    m.recursionProductImplId = m.ruleInterner.encode("(>[12](in3[12,u_5,u_1,u_5])(preorder94[u_1,u_2,12,u_5]))");
    const auto known = [&](const char* text, gl::NameId at) {
        gl::upsertStatementKey(m.intKnownStatements, gl::packStatementKey(m.nameMap.encode(text), at), true);
    };

    analyzer.admissionMemory.fileHypothesisOrDemand(m, literalSpan("(preorder94[1,2,9,5])"),
        gl::StrSpan(scope), m.recursionAuxyImplId);
    const gl::AdmissionOrdis2ValueSet filed = hypothesisDemandsAt(m, "(preorder94[1,2,9,5])", scope);
    ASSERT_EQ(filed.size(), static_cast<std::size_t>(1));
    ASSERT_EQ(filed.begin()->sourceImplId, m.recursionAuxyImplId);
    ASSERT_TRUE(filed.begin()->levels.empty());
    ASSERT_TRUE(hypothesisDemandsAt(m, "(preorder94[1,2,9,5])", "main").empty());
    // A second hypothesis rule adds its own record under the same key.
    analyzer.admissionMemory.fileHypothesisOrDemand(m, literalSpan("(preorder94[1,2,9,5])"),
        gl::StrSpan(scope), m.recursionProductImplId);
    ASSERT_EQ(hypothesisDemandsAt(m, "(preorder94[1,2,9,5])", scope).size(), static_cast<std::size_t>(2));
    ASSERT_EQ(m.overallHashMemory.admissionMapOrdis2.count(), 1);

    known("(preorder94[1,2,9,6])", scopeId);
    known("(preorder94[1,2,9,7])", gl::NameMap::MAIN_ID);
    known("(preorder94[1,2,9,8])", siblingId);
    for (const char* premise : { "(preorder94[1,2,9,6])", "(preorder94[1,2,9,7])", "(short94[1,2,9])",
            "(=[9,5])", "!(=[9,5])" })
        analyzer.admissionMemory.fileHypothesisOrDemand(m, literalSpan(premise), gl::StrSpan(scope),
            m.recursionAuxyImplId);
    ASSERT_EQ(m.overallHashMemory.admissionMapOrdis2.count(), 1);
    analyzer.admissionMemory.fileHypothesisOrDemand(m, literalSpan("(preorder94[1,2,9,8])"),
        gl::StrSpan(scope), m.recursionAuxyImplId);
    ASSERT_EQ(hypothesisDemandsAt(m, "(preorder94[1,2,9,8])", scope).size(), static_cast<std::size_t>(1));
    ASSERT_EQ(m.overallHashMemory.admissionMapOrdis2.count(), 2);
}

// An OR already parked under the missing expression at the demand's scope is
// woken: its park key is erased, the statement un-known and re-deposited on
// the absorb's output channel while an absorb runs. The same OR parked at a
// sibling scope stays parked.
TEST(prover, hypothesis_or_demand_wakes_the_or_parked_at_its_scope_only) {
    gl::tests::ProverFixture analyzer("Peano");
    seedHypothesisDemandDefinitions(analyzer);
    gl::Memory m;
    ASSERT_EQ(m.nameMap.encode("main"), gl::NameMap::MAIN_ID);
    const gl::NameId scopeId = m.nameMap.encodePush(gl::NameMap::MAIN_ID, std::string("is_(in3[marker,5,1,4])_9"));
    const gl::NameId siblingId = m.nameMap.encodePush(gl::NameMap::MAIN_ID, std::string("is_(in3[marker,5,1,4])_8"));
    const std::string scope = m.nameMap.decode(scopeId);
    const std::string sibling = m.nameMap.decode(siblingId);
    analyzer.canonicalizer.initializeCanonicalScope(m, scopeId);
    analyzer.canonicalizer.initializeCanonicalScope(m, siblingId);
    m.recursionAuxyImplId = m.ruleInterner.encode("(>[11](in3[11,u_5,u_1,u_4])(preorder94[u_1,u_2,11,u_5]))");
    gl::ScratchArena& tArena = gl::genScratchArenas().forSlot(gl::genScratchArenas().slotCount() - 1);

    const int32_t orId = m.valueInterner.encode("(or94[9,5])");
    const int lv[] = { 0 };
    const int64_t pkScope = gl::mintTemplateKey(m.templateInterner, m.nameMap, "(preorder94[1,2,9,5])", scope);
    const int64_t pkSibling = gl::mintTemplateKey(m.templateInterner, m.nameMap, "(preorder94[1,2,9,5])", sibling);
    gl::insertRejectedOrdisIdsBlob(m.overallHashMemory.rejectedMapOrdis2, pkScope, orId, lv, 1, m.valueInterner, tArena);
    gl::insertRejectedOrdisIdsBlob(m.overallHashMemory.rejectedMapOrdis2, pkSibling, orId, lv, 1, m.valueInterner, tArena);
    const gl::NameId orName = m.nameMap.encode("(or94[9,5])");
    gl::upsertStatementKey(m.intKnownStatements, gl::packStatementKey(orName, scopeId), true);
    gl::upsertStatementKey(m.intKnownStatements, gl::packStatementKey(orName, siblingId), true);

    m.replayMailDuringAbsorb = &m.nextIterationInternalMail;
    analyzer.admissionMemory.fileHypothesisOrDemand(m, literalSpan("(preorder94[1,2,9,5])"),
        gl::StrSpan(scope), m.recursionAuxyImplId);
    m.replayMailDuringAbsorb = nullptr;

    ASSERT_EQ(hypothesisDemandsAt(m, "(preorder94[1,2,9,5])", scope).size(), static_cast<std::size_t>(1));
    ASSERT_TRUE(m.overallHashMemory.rejectedMapOrdis2.lookup(pkScope) == 0);
    ASSERT_TRUE(m.overallHashMemory.rejectedMapOrdis2.lookup(pkSibling) != 0);
    ASSERT_TRUE(m.intKnownStatements.find(gl::StatementKey{ orName, scopeId }) == nullptr);
    ASSERT_TRUE(m.intKnownStatements.find(gl::StatementKey{ orName, siblingId }) != nullptr);
    ASSERT_TRUE(m.sameIterationInternalMail.statementsEmpty());
    const auto rows = gl::decodeInternalMailStatements(m.nextIterationInternalMail, m.nameMap);
    ASSERT_EQ(rows.size(), static_cast<std::size_t>(1));
    ASSERT_TRUE(rows[0].first.original == "(or94[9,5])");
    ASSERT_TRUE(rows[0].first.validityName == scope);
    ASSERT_EQ(rows[0].second, (std::set<int>{ 0 }));
}

// At main the hypothesis's missing expressions are its premises with only
// fixed arguments: each is filed as a ground demand at main when it can key
// an OR park; a premise with a bound variable, or an equality, files nothing.
TEST(prover, hypothesis_demands_at_main_file_the_fixed_premises) {
    gl::tests::ProverFixture analyzer("Peano");
    seedHypothesisDemandDefinitions(analyzer);
    gl::Memory m;
    ASSERT_EQ(m.nameMap.encode("main"), gl::NameMap::MAIN_ID);
    m.recursionAuxyImplId = m.ruleInterner.encode(
        "(>[11](in3[11,u_5,u_9,u_4])(>[](preorder94[u_1,u_2,11,u_5])(>[]!(=[u_9,u_5])(>[](preorder94[u_1,u_2,u_9,u_5])(d[11])))))");
    const gl::StrSpan premises[4] = { literalSpan("(in3[11,u_5,u_9,u_4])"), literalSpan("(preorder94[u_1,u_2,11,u_5])"),
        literalSpan("!(=[u_9,u_5])"), literalSpan("(preorder94[u_1,u_2,u_9,u_5])") };
    analyzer.admissionMemory.fileHypothesisDemandsAtMain(m, premises, 4, m.recursionAuxyImplId);
    ASSERT_EQ(m.overallHashMemory.admissionMapOrdis2.count(), 1);
    const gl::AdmissionOrdis2ValueSet filed = hypothesisDemandsAt(m, "(preorder94[1,2,9,5])", "main");
    ASSERT_EQ(filed.size(), static_cast<std::size_t>(1));
    ASSERT_EQ(filed.begin()->sourceImplId, m.recursionAuxyImplId);
    ASSERT_TRUE(filed.begin()->levels.empty());
}

// A matched demand belongs to the induction hypothesis when one of its
// records has a hypothesis rule of the block as source; another rule's record
// does not count (D-575).
TEST(prover, induction_hypothesis_demanded_reads_the_record_sources) {
    gl::Memory m;
    ASSERT_EQ(m.nameMap.encode("main"), gl::NameMap::MAIN_ID);
    gl::ScratchArena& tArena = gl::genScratchArenas().forSlot(gl::genScratchArenas().slotCount() - 1);
    const int32_t ordinaryRule = m.ruleInterner.encode("(>[]dummy(d[1]))");
    const int32_t hypothesisRule = m.ruleInterner.encode("(>[]hypothesis(h[1]))");
    m.recursionProductImplId = hypothesisRule;
    const int lv[] = { 0 };
    const int64_t ordinaryKey = gl::mintTemplateKey(m.templateInterner, m.nameMap, "(preorder94[1,2,9,5])", "main");
    const int64_t sharedKey = gl::mintTemplateKey(m.templateInterner, m.nameMap, "(preorder94[1,2,9,6])", "main");
    gl::insertAdmissionOrdis2IdsBlob(m.overallHashMemory.admissionMapOrdis2, ordinaryKey, ordinaryRule, lv, 1,
        m.ruleInterner, tArena);
    gl::insertAdmissionOrdis2IdsBlob(m.overallHashMemory.admissionMapOrdis2, sharedKey, ordinaryRule, lv, 1,
        m.ruleInterner, tArena);
    gl::insertAdmissionOrdis2IdsBlob(m.overallHashMemory.admissionMapOrdis2, sharedKey, hypothesisRule, nullptr, 0,
        m.ruleInterner, tArena);
    gl::ScratchScope frame(tArena);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::PagedVector<int64_t> keys(&tArena, &dirty);
    keys.push_back(ordinaryKey);
    ASSERT_FALSE(gl::CaseProcessor::inductionHypothesisDemanded(m, keys, tArena));
    keys.push_back(sharedKey);
    ASSERT_TRUE(gl::CaseProcessor::inductionHypothesisDemanded(m, keys, tArena));
    m.recursionProductImplId = 0;
    ASSERT_FALSE(gl::CaseProcessor::inductionHypothesisDemanded(m, keys, tArena));
}

// A demand of the induction hypothesis opens its OR at deposit like any
// demand, its disjunct first; the fresh cohort is recorded for the release
// below an unsettled induction scope (D-575).
TEST(prover, ordis2_route_c_marks_a_hypothesis_demanded_cohort) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    gl::WorkInstruction wi(&lb);
    buildOrInstruction(wi, "(or93[u_4,u_5])",
                       { "(preorder93[u_1,u_2,u_4,u_5])", "(=[u_4,u_5])" });

    gl::tests::ProverFixture analyzer("Peano");
    gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions)["preorder93"] =
        gl::LogicalEntity("existence",
            { "(in[w1,u_1])", "(in3[u_4,w1,u_5,u_2])" },
            "(preorder93[u_1,u_2,u_4,u_5])", 4);
    analyzer.supportingRuleCompiler.compileDisintegrationPackages();
    gl::Memory memory;
    ASSERT_EQ(memory.nameMap.encode("main"), gl::NameMap::MAIN_ID);
    memory.intToBeProved.assignSet(gl::packStatementKey(
        memory.nameMap.encode("(freeze_goal[1])"), gl::NameMap::MAIN_ID), nullptr, 0);
    gl::CollectedArena collected(&lb);
    gl::NewVarStore newVarMap(&lb);
    gl::ScratchArena& tArena = gl::genScratchArenas().forSlot(gl::genScratchArenas().slotCount() - 1);
    int seedLv[1] = { 0 };

    memory.recursionProductImplId = memory.ruleInterner.encode(
        "(>[11,12](in3[12,11,u_9,u_5])(>[](preorder93[u_1,u_2,u_4,u_5])(h[11])))");
    const int64_t demandPk = gl::mintTemplateKey(memory.templateInterner, memory.nameMap,
        "(preorder93[1,2,4,5])", "main");
    gl::insertAdmissionOrdis2IdsBlob(memory.overallHashMemory.admissionMapOrdis2, demandPk,
        memory.recursionProductImplId, nullptr, 0, memory.ruleInterner, tArena);

    gl::tests::DisintegrationProcessorFixture::disintegrateExprCore2(analyzer.disintegrationProcessor,
        literalSpan("(or93[u_4,u_5])"), wi, memory, 0, collected, newVarMap, gl::StrSpan("main", 4),
        /*trackHistoryLocal=*/true, /*allowOrDisintegration=*/false, seedLv, 1, /*allowOrProbe=*/true);

    const int32_t cohortId = gl::mintOrCohortId(memory.lbStateInterner,
        memory.lbStateInterner.encode(gl::StrSpan("main", 4)),
        memory.lbStateInterner.encode(literalSpan("(or93[4,5])")));
    ASSERT_TRUE(memory.orDisjunctCount.lookup(cohortId) != 0);
    ASSERT_TRUE(memory.pendingOrReleases.lookup(cohortId) != 0);
    ASSERT_TRUE(memory.lbMemory.hypothesisOrCohorts.lookup(cohortId) != 0);
    const int32_t* starter = memory.orStarterPick.find(cohortId);
    ASSERT_TRUE(starter != nullptr);
    ASSERT_TRUE(memory.lbStateInterner.decode(*starter) == std::string("((preorder93[1,2,4,5]))"));
}
