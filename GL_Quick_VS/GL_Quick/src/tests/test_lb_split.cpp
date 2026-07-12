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
/// @brief Unit tests for the LB-split reference-baseline refactor (commit 1):
///        the `FiringRecord` capture + `applyFiringRecords` canonical sort/apply.
///
/// @details
/// The hashburst's deposit side is captured as `FiringRecord`s and applied in a
/// canonical sorted order so the resulting container state is a function of the
/// firing SET, not the request-generation ORDER — the determinism the LB split
/// (later commits) relies on. The central test pins that order-independence
/// directly: apply two permutations of the same record set to two memory blocks
/// and assert identical deposits (cap-bounded origin selection AND the staged
/// vectors' drain order). The remaining tests pin per-record deposit routing and
/// the changed/new member signatures. See `D-117`.

#include "test_harness.hpp"

#include "../prover.hpp"
#include "../infra/rt_tracker.hpp"

#include <algorithm>
#include <string>
#include <vector>
#include <set>
#include <tuple>
#include <atomic>
#include <cstddef>
#include <cstdint>

namespace {
    /// Standalone-NameMap rig (strings campaign): a NameMap is a facade
    /// over two cold string tables; this binds one to private tables on
    /// the harness pool (declaration order: manager first, so the tables
    /// destruct before it).
    struct NameMapRig {
        gl::LbArena lb{ &gl::staticMemory() };
        gl::DirtyState dirty = gl::DirtyState::Clean;
        gl::ColdStringTable names{ &lb, &dirty };
        gl::ColdStringTable subs{ &lb, &dirty };
        gl::PagedVector<gl::ValidityNode> nodes{ &lb, &dirty };
        gl::NameMap nm;
        NameMapRig() { nm.bind(&names, &subs, &nodes); }
    };
} // namespace


namespace {

    /// Per-test sealed-page rig backing the records this suite builds:
    /// binds to the harness pool, seals and frees in its destructor.
    /// Declare it before the records it backs so the views die first.
    struct RecordRig {
        gl::SealedPageSet pages;
        RecordRig() { pages.bind(&gl::staticMemory()); }
        ~RecordRig() {
            if (pages.filling()) pages.seal();
            pages.freePages();
        }
        gl::SealedString s(const std::string& v) {
            return gl::SealedString::copyFrom(
                pages, v.data(), static_cast<int32_t>(v.size()));
        }
    };

    gl::FiringRecord makeHeadRecord(RecordRig& rig,
                                    const std::string& expr,
                                    const std::string& validity,
                                    const std::string& tag,
                                    const std::string& dep,
                                    bool allGood,
                                    bool alreadyKnown) {
        gl::FiringRecord r;
        r.isMarker = false;
        r.rplExpr2 = rig.s(expr);
        r.validityName = rig.s(validity);
        const int levelsArr[1] = { 0 };
        r.levels = gl::SealedSpan<int>::copyFrom(rig.pages, levelsArr, 1);
        r.originTag = rig.s(tag);
        const gl::SealedExpressionWithValidity depsArr[1] = {
            gl::SealedExpressionWithValidity{ rig.s(dep), rig.s("main") } };
        r.originDeps =
            gl::SealedSpan<gl::SealedExpressionWithValidity>::copyFrom(
                rig.pages, depsArr, 1);
        r.doNotDisintegrate = false;
        r.allowOrDisintegration = false;
        r.allGood = allGood;
        r.alreadyKnown = alreadyKnown;
        return r;
    }

    gl::FiringRecord makeMarkerRecord(RecordRig& rig,
                                      const std::string& expr,
                                      const std::string& validity,
                                      const std::string& keyElem) {
        gl::FiringRecord r;
        r.isMarker = true;
        r.rplExpr2 = rig.s(expr);
        r.validityName = rig.s(validity);
        const gl::SealedString markerArgsArr[1] = { rig.s("a") };
        r.markerArgsSorted = gl::SealedSpan<gl::SealedString>::copyFrom(
            rig.pages, markerArgsArr, 1);
        gl::StagedAdmissionValue admv;
        const gl::SealedString keyArr[1] = { rig.s(keyElem) };
        admv.key = gl::SealedSpan<gl::SealedString>::copyFrom(
            rig.pages, keyArr, 1);
        r.admv = admv;
        r.markerNotAtomic = true;
        return r;
    }

} // namespace

TEST(lb_split, check_local_encoded_memory_static_symbol_signature) {
    // Pins the record-chain capture signature. Compiles only if
    // checkLocalEncodedMemoryStatic appends its firings to the task's sealed
    // page set (no separate out-buffer parameter).
    using ChkFn = void (gl::ExpressionAnalyzer::*)(const gl::StaticRequest&,
                                                   const gl::Memory&,
                                                   unsigned,
                                                   gl::SealedPageSet&);
    ChkFn fn = &gl::ExpressionAnalyzer::checkLocalEncodedMemoryStatic;
    ASSERT_TRUE(fn != nullptr);
}

TEST(lb_split, apply_firing_records_symbol_signature) {
    // Pins the parts-view consumer signature: the LB's sealed part sets in
    // part order, records read off each set's chain.
    using ApplyFn = void (gl::ExpressionAnalyzer::*)(gl::Memory&,
                                                     gl::SealedPageSet* const*,
                                                     std::int32_t);
    ApplyFn fn = &gl::ExpressionAnalyzer::applyFiringRecords;
    ASSERT_TRUE(fn != nullptr);
}

TEST(lb_split, apply_firing_records_head_deposit) {
    // A head record (allGood, not alreadyKnown) deposits the statement, the
    // origin, and the disintegrationSignals entry. The mail-eligibility memo
    // stays empty: the head carries no `int_lev_*`, so `allowedForMail`
    // could never probe an entry for it (insert is gated).
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    ana.parameters.trackHistory = true;
    ana.parameters.compressor_mode = false;
    ana.parameters.max_origin_per_expr = 1;
    RecordRig rig;
    gl::Memory mb;
    rig.pages.appendRecord(
        makeHeadRecord(rig, "(in[2,1])", "main", "implication", "(impA)", true, false));
    gl::SealedPageSet* parts[1] = { &rig.pages };
    ana.applyFiringRecords(mb, parts, 1);

    const gl::ExpressionWithValidity key("(in[2,1])", "main");
    bool found = false;
    const gl::Mail mailSnap = gl::makeHeapMail(mb.sameIterationInternalMail,
                                               mb.nameMap, mb.originInterner);
    for (const auto& st : mailSnap.statements) {
        if (st.first.original == "(in[2,1])" && st.first.validityName == "main") found = true;
    }
    ASSERT_TRUE(found);
    ASSERT_TRUE(mb.canBeSentIds.empty());
    ASSERT_TRUE(mailSnap.exprOriginMap.find(key) != mailSnap.exprOriginMap.end());
    ASSERT_TRUE(mailSnap.disintegrationSignals.find(key)
                != mailSnap.disintegrationSignals.end());
}

TEST(lb_split, apply_firing_records_already_known_skips_deposit) {
    // alreadyKnown gates the statement/origin deposit off, but the
    // mail-eligibility memo is still populated for an `int_lev_*`-carrying
    // allGood head (memo insert is unconditional on alreadyKnown, matching
    // the pre-refactor inline order).
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    ana.parameters.trackHistory = true;
    ana.parameters.compressor_mode = false;
    RecordRig rig;
    gl::Memory mb;
    rig.pages.appendRecord(
        makeHeadRecord(rig, "(in[int_lev_1_2,1])", "main", "implication", "(impA)", true, /*alreadyKnown=*/true));
    gl::SealedPageSet* parts[1] = { &rig.pages };
    ana.applyFiringRecords(mb, parts, 1);
    ASSERT_TRUE(mb.sameIterationInternalMail.statementsEmpty());
    ASSERT_TRUE(mb.sameIterationInternalMail.exprOriginMapEmpty());
    const int16_t headId = mb.nameMap.lookup(std::string("(in[int_lev_1_2,1])"));
    ASSERT_TRUE(headId != 0);
    ASSERT_TRUE(mb.canBeSentIds.contains(headId));
}

TEST(lb_split, apply_firing_records_marker_deposit) {
    // A marker record stages deferredIntegrationPreps + admissionKeysAlgebra and
    // populates canBeSentMarkerIds (non-atomic; marker forms are inserted
    // ungated — the probe key is the marker form itself).
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    RecordRig rig;
    gl::Memory mb;
    rig.pages.appendRecord(
        makeMarkerRecord(rig, "(in3[1,marker,2,3])", "main", "(k1)"));
    gl::SealedPageSet* parts[1] = { &rig.pages };
    ana.applyFiringRecords(mb, parts, 1);
    ASSERT_EQ((int)mb.deferredIntegrationPreps.size(), 1);
    ASSERT_EQ((int)mb.admissionKeysAlgebra.size(), 1);
    const int16_t markerId = mb.nameMap.lookup(std::string("(in3[1,marker,2,3])"));
    ASSERT_TRUE(markerId != 0);
    ASSERT_TRUE(mb.canBeSentMarkerIds.contains(markerId));
}

namespace {
    /// Shared assertion block for the order-independence tests: the two memory
    /// blocks must carry identical deposits (cap-1 origin selection AND the
    /// staged vectors' drain order).
    void assertSameDeposits(gl::Memory& mbA, gl::Memory& mbB) {
        const gl::ExpressionWithValidity key("(in[2,1])", "main");
        const gl::Mail snapA = gl::makeHeapMail(mbA.sameIterationInternalMail,
                                                mbA.nameMap, mbA.originInterner);
        const gl::Mail snapB = gl::makeHeapMail(mbB.sameIterationInternalMail,
                                                mbB.nameMap, mbB.originInterner);
        auto itA = snapA.exprOriginMap.find(key);
        auto itB = snapB.exprOriginMap.find(key);
        ASSERT_TRUE(itA != snapA.exprOriginMap.end());
        ASSERT_TRUE(itB != snapB.exprOriginMap.end());
        ASSERT_EQ((int)itA->second.size(), 1);
        ASSERT_EQ((int)itB->second.size(), 1);
        ASSERT_EQ(itA->second.front().first, itB->second.front().first);
        ASSERT_EQ((int)itA->second.front().second.size(), 1);
        ASSERT_EQ((int)itB->second.front().second.size(), 1);
        ASSERT_EQ(itA->second.front().second.front().original,
                  itB->second.front().second.front().original);

        ASSERT_EQ((int)mbA.deferredIntegrationPreps.size(),
                  (int)mbB.deferredIntegrationPreps.size());
        for (std::size_t i = 0; i < mbA.deferredIntegrationPreps.size(); ++i) {
            ASSERT_EQ(mbA.deferredIntegrationPreps[i].expression.toStdString(),
                      mbB.deferredIntegrationPreps[i].expression.toStdString());
        }
        ASSERT_EQ((int)mbA.admissionKeysAlgebra.size(),
                  (int)mbB.admissionKeysAlgebra.size());
        for (std::size_t i = 0; i < mbA.admissionKeysAlgebra.size(); ++i) {
            ASSERT_EQ(mbA.admissionKeysAlgebra[i].key.original.toStdString(),
                      mbB.admissionKeysAlgebra[i].key.original.toStdString());
        }
    }
} // namespace

TEST(lb_split, apply_firing_records_order_independent) {
    // The load-bearing property for the LB split: applying the SAME records in
    // two different orders yields identical container state. Two head records
    // share a key (cap-1 origin selection is order-sensitive) and two markers
    // share a key (staged-vector drain order is order-sensitive). Both memory
    // blocks must end identical.
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    ana.parameters.trackHistory = true;
    ana.parameters.compressor_mode = false;
    ana.parameters.max_origin_per_expr = 1;
    RecordRig rig;

    gl::FiringRecord h1 = makeHeadRecord(rig,"(in[2,1])", "main", "implication", "(impA)", true, false);
    gl::FiringRecord h2 = makeHeadRecord(rig, "(in[2,1])", "main", "implication", "(impB)", true, false);
    gl::FiringRecord m1 = makeMarkerRecord(rig, "(in3[1,marker,2,3])", "main", "(k1)");
    gl::FiringRecord m2 = makeMarkerRecord(rig, "(in3[1,marker,2,3])", "main", "(k2)");

    // The SAME records (their views all alias `rig`) appended to two chains
    // in opposite orders — the chain twin of the former vector permutations.
    RecordRig rigA;
    rigA.pages.appendRecord(h1);
    rigA.pages.appendRecord(h2);
    rigA.pages.appendRecord(m1);
    rigA.pages.appendRecord(m2);
    RecordRig rigB;
    rigB.pages.appendRecord(m2);
    rigB.pages.appendRecord(m1);
    rigB.pages.appendRecord(h2);
    rigB.pages.appendRecord(h1);

    gl::Memory mbA;
    gl::Memory mbB;
    gl::SealedPageSet* partsA[1] = { &rigA.pages };
    gl::SealedPageSet* partsB[1] = { &rigB.pages };
    ana.applyFiringRecords(mbA, partsA, 1);
    ana.applyFiringRecords(mbB, partsB, 1);
    assertSameDeposits(mbA, mbB);

    // Two-part variant: the records split across TWO page sets handed in both
    // part orders — deposits identical (part-order independence, the property
    // the former merged-vector concatenation gave structurally).
    RecordRig rigP;
    rigP.pages.appendRecord(h1);
    rigP.pages.appendRecord(m1);
    RecordRig rigQ;
    rigQ.pages.appendRecord(h2);
    rigQ.pages.appendRecord(m2);

    gl::Memory mbC;
    gl::Memory mbD;
    gl::SealedPageSet* partsPQ[2] = { &rigP.pages, &rigQ.pages };
    gl::SealedPageSet* partsQP[2] = { &rigQ.pages, &rigP.pages };
    ana.applyFiringRecords(mbC, partsPQ, 2);
    ana.applyFiringRecords(mbD, partsQP, 2);
    assertSameDeposits(mbC, mbD);
    // And the two-part merge equals the one-part merge of the same set.
    assertSameDeposits(mbA, mbC);
}

TEST(lb_split, pointer_index_sort_matches_vector_sort) {
    // The frozen-byte oracle for the C2 index-sort flip: sorting a pointer
    // INDEX under the production comparator's strict total order yields the
    // exact record sequence of an in-place std::sort over a record vector.
    // Adversarial near-ties: same expr different validity; same head
    // different originDeps lengths; equal-content duplicates; marker vs head
    // on one key; markers differing only in markerNotAtomic.
    RecordRig rig;
    std::vector<gl::FiringRecord> recs;
    // Same expr, different validity.
    recs.push_back(makeHeadRecord(rig, "(in[2,1])", "main", "implication", "(impA)", true, false));
    recs.push_back(makeHeadRecord(rig, "(in[2,1])", "main_boundary_(h1)", "implication", "(impA)", true, false));
    // Same head, different originDeps lengths (2 deps vs 1 dep).
    {
        gl::FiringRecord r = makeHeadRecord(rig, "(in[3,1])", "main", "implication", "(impA)", true, false);
        const gl::SealedExpressionWithValidity deps2[2] = {
            gl::SealedExpressionWithValidity{ rig.s("(impA)"), rig.s("main") },
            gl::SealedExpressionWithValidity{ rig.s("(impB)"), rig.s("main") } };
        r.originDeps = gl::SealedSpan<gl::SealedExpressionWithValidity>::copyFrom(
            rig.pages, deps2, 2);
        recs.push_back(r);
    }
    recs.push_back(makeHeadRecord(rig, "(in[3,1])", "main", "implication", "(impA)", true, false));
    // Equal-content duplicates.
    recs.push_back(makeHeadRecord(rig, "(in[4,1])", "main", "implication", "(impC)", true, false));
    recs.push_back(makeHeadRecord(rig, "(in[4,1])", "main", "implication", "(impC)", true, false));
    // Marker vs head on one key.
    recs.push_back(makeHeadRecord(rig, "(in3[1,marker,2,3])", "main", "implication", "(impA)", true, false));
    recs.push_back(makeMarkerRecord(rig, "(in3[1,marker,2,3])", "main", "(k1)"));
    // Markers differing only in markerNotAtomic.
    {
        gl::FiringRecord r = makeMarkerRecord(rig, "(in3[9,marker,2,3])", "main", "(k9)");
        recs.push_back(r);
        r.markerNotAtomic = false;
        recs.push_back(r);
    }
    // Two more for bulk (differing bools on one head key).
    {
        gl::FiringRecord r = makeHeadRecord(rig, "(in[5,1])", "main", "implication", "(impD)", true, false);
        recs.push_back(r);
        r.allGood = false;
        recs.push_back(r);
    }
    ASSERT_EQ((int)recs.size(), 12);

    // VERBATIM replica of the production comparator (memory.cpp
    // applyFiringRecords `recordLess`) — the retained test oracle per the
    // house doctrine; a production comparator edit shows up as a divergence
    // against the full-pipeline gate, a replica edit as a review diff here.
    const auto recordLess = [](const gl::FiringRecord& a, const gl::FiringRecord& b) -> bool {
        using gl::StrSpan;
        using gl::compareSpans;
        int c = compareSpans(StrSpan(a.rplExpr2), StrSpan(b.rplExpr2));
        if (c != 0) return c < 0;
        c = compareSpans(StrSpan(a.validityName), StrSpan(b.validityName));
        if (c != 0) return c < 0;
        if (a.isMarker != b.isMarker) return a.isMarker < b.isMarker; // head before marker
        if (!a.isMarker) {
            c = compareSpans(StrSpan(a.originTag), StrSpan(b.originTag));
            if (c != 0) return c < 0;
            if (a.originDeps.size() != b.originDeps.size())
                return a.originDeps.size() < b.originDeps.size();
            for (std::int32_t i = 0; i < a.originDeps.size(); ++i) {
                c = compareSpans(StrSpan(a.originDeps[i].original),
                                 StrSpan(b.originDeps[i].original));
                if (c != 0) return c < 0;
                c = compareSpans(StrSpan(a.originDeps[i].validityName),
                                 StrSpan(b.originDeps[i].validityName));
                if (c != 0) return c < 0;
            }
            {
                const std::int32_t na = a.levels.size();
                const std::int32_t nb = b.levels.size();
                const std::int32_t mn = na < nb ? na : nb;
                for (std::int32_t i = 0; i < mn; ++i) {
                    if (a.levels[i] != b.levels[i])
                        return a.levels[i] < b.levels[i];
                }
                if (na != nb) return na < nb;
            }
            if (a.doNotDisintegrate != b.doNotDisintegrate)
                return a.doNotDisintegrate < b.doNotDisintegrate;
            if (a.allowOrDisintegration != b.allowOrDisintegration)
                return a.allowOrDisintegration < b.allowOrDisintegration;
            if (a.allGood != b.allGood) return a.allGood < b.allGood;
            if (a.alreadyKnown != b.alreadyKnown) return a.alreadyKnown < b.alreadyKnown;
            return false;
        }
        const auto lexLess3 = [](const gl::SealedSpan<gl::SealedString>& x,
                                 const gl::SealedSpan<gl::SealedString>& y) -> int {
            const std::int32_t n = x.size() < y.size() ? x.size() : y.size();
            for (std::int32_t i = 0; i < n; ++i) {
                const int e = gl::compareSpans(gl::StrSpan(x[i]), gl::StrSpan(y[i]));
                if (e != 0) return e;
            }
            if (x.size() != y.size())
                return x.size() < y.size() ? -1 : 1;
            return 0;
        };
        c = lexLess3(a.admv.key, b.admv.key);
        if (c != 0) return c < 0;
        c = lexLess3(a.markerArgsSorted, b.markerArgsSorted);
        if (c != 0) return c < 0;
        c = lexLess3(a.admv.remainingArgsSorted,
                     b.admv.remainingArgsSorted);
        if (c != 0) return c < 0;
        if (a.markerNotAtomic != b.markerNotAtomic)
            return a.markerNotAtomic < b.markerNotAtomic;
        return false;
    };

    // Oracle: the former shape — an in-place std::sort over a record vector.
    std::vector<gl::FiringRecord> oracle = recs;
    std::sort(oracle.begin(), oracle.end(), recordLess);

    // Candidate: the index-sort shape — stable record addresses, a pointer
    // index sorted by the deref'd comparator.
    std::vector<const gl::FiringRecord*> refs;
    for (const gl::FiringRecord& r : recs) refs.push_back(&r);
    std::vector<std::int32_t> idx(refs.size());
    for (std::int32_t k = 0; k < (std::int32_t)refs.size(); ++k) idx[k] = k;
    std::sort(idx.begin(), idx.end(), [&](std::int32_t x, std::int32_t y) {
        return recordLess(*refs[x], *refs[y]);
    });

    // The deref'd sequence field-equals the oracle sequence element-by-element.
    const auto fieldEquals = [](const gl::FiringRecord& a, const gl::FiringRecord& b) {
        if (a.isMarker != b.isMarker) return false;
        if (a.rplExpr2.toStdString() != b.rplExpr2.toStdString()) return false;
        if (a.validityName.toStdString() != b.validityName.toStdString()) return false;
        if (a.levels.size() != b.levels.size()) return false;
        for (std::int32_t i = 0; i < a.levels.size(); ++i)
            if (a.levels[i] != b.levels[i]) return false;
        if (a.originTag.toStdString() != b.originTag.toStdString()) return false;
        if (a.originDeps.size() != b.originDeps.size()) return false;
        for (std::int32_t i = 0; i < a.originDeps.size(); ++i) {
            if (a.originDeps[i].original.toStdString()
                != b.originDeps[i].original.toStdString()) return false;
            if (a.originDeps[i].validityName.toStdString()
                != b.originDeps[i].validityName.toStdString()) return false;
        }
        if (a.doNotDisintegrate != b.doNotDisintegrate) return false;
        if (a.allowOrDisintegration != b.allowOrDisintegration) return false;
        if (a.allGood != b.allGood) return false;
        if (a.alreadyKnown != b.alreadyKnown) return false;
        if (a.markerArgsSorted.size() != b.markerArgsSorted.size()) return false;
        for (std::int32_t i = 0; i < a.markerArgsSorted.size(); ++i)
            if (a.markerArgsSorted[i].toStdString()
                != b.markerArgsSorted[i].toStdString()) return false;
        if (a.admv.key.size() != b.admv.key.size()) return false;
        for (std::int32_t i = 0; i < a.admv.key.size(); ++i)
            if (a.admv.key[i].toStdString()
                != b.admv.key[i].toStdString()) return false;
        if (a.admv.remainingArgsSorted.size()
            != b.admv.remainingArgsSorted.size()) return false;
        for (std::int32_t i = 0; i < a.admv.remainingArgsSorted.size(); ++i)
            if (a.admv.remainingArgsSorted[i].toStdString()
                != b.admv.remainingArgsSorted[i].toStdString()) return false;
        if (a.markerNotAtomic != b.markerNotAtomic) return false;
        return true;
    };
    ASSERT_EQ((int)oracle.size(), (int)idx.size());
    for (std::size_t i = 0; i < oracle.size(); ++i) {
        ASSERT_TRUE(fieldEquals(oracle[i], *refs[idx[i]]));
    }
}

TEST(lb_split, make_partition_id_packs_and_is_injective) {
    // makePartitionId packs origId in the high 16 bits and scopeVid (as
    // uint16) in the low 16 — the composite id stored in OwnerSet::partitionIds.
    // Layout:
    ASSERT_EQ(gl::makePartitionId(0, 0), (std::int32_t)0);
    ASSERT_EQ(gl::makePartitionId(1, 0), (std::int32_t)(1 << 16));
    ASSERT_EQ(gl::makePartitionId(0, 5), (std::int32_t)5);
    ASSERT_EQ(gl::makePartitionId(7, 3), (std::int32_t)((7 << 16) | 3));
    // Max non-negative int16 inputs (the ids NameMap mints) stay a positive
    // int32 — so id % N is well-defined and non-negative for any N > 0:
    ASSERT_TRUE(gl::makePartitionId(32767, 32767) > 0);
    ASSERT_EQ(gl::makePartitionId(32767, 32767),
              (std::int32_t)((32767 << 16) | 32767));
    // Injective: differing in EITHER half yields a different id — distinct
    // (original, validity) owners never collide.
    ASSERT_TRUE(gl::makePartitionId(3, 4) != gl::makePartitionId(4, 3));
    ASSERT_TRUE(gl::makePartitionId(3, 4) != gl::makePartitionId(3, 5));
    ASSERT_TRUE(gl::makePartitionId(3, 4) != gl::makePartitionId(2, 4));
    // Deterministic: same inputs -> same id, so the wipe-time recompute matches
    // the install-time id (the lockstep-erase contract).
    ASSERT_EQ(gl::makePartitionId(123, 45), gl::makePartitionId(123, 45));
}

TEST(lb_split, partition_accepts_filters_by_residue) {
    // partitionAccepts(set) accepts iff g_splitCount <= 1 (unsplit -> always) or
    // some id in the set has id % g_splitCount == g_splitProcessID. It reads the
    // thread-local executor context, so save/restore it around the checks.
    const int savedPid = gl::g_splitProcessID;
    const int savedN = gl::g_splitCount;

    std::set<std::int32_t> ids = { 3, 7, 12 };

    // N <= 1 is the unsplit identity: always true, set untouched (even empty).
    gl::g_splitCount = 1; gl::g_splitProcessID = 0;
    ASSERT_TRUE(gl::partitionAccepts(ids));
    std::set<std::int32_t> empty;
    ASSERT_TRUE(gl::partitionAccepts(empty));

    // N = 4: residues present are {3%4=3, 7%4=3, 12%4=0} = {0, 3}.
    gl::g_splitCount = 4;
    gl::g_splitProcessID = 0; ASSERT_TRUE(gl::partitionAccepts(ids));   // 12 % 4 == 0
    gl::g_splitProcessID = 3; ASSERT_TRUE(gl::partitionAccepts(ids));   // 3 % 4 == 3
    gl::g_splitProcessID = 1; ASSERT_FALSE(gl::partitionAccepts(ids));  // none == 1
    gl::g_splitProcessID = 2; ASSERT_FALSE(gl::partitionAccepts(ids));  // none == 2

    // Completeness: a non-empty set is claimed by at least one executor.
    int claimedBy = 0;
    for (int n = 0; n < 4; ++n) {
        gl::g_splitProcessID = n;
        if (gl::partitionAccepts(ids)) ++claimedBy;
    }
    ASSERT_TRUE(claimedBy >= 1);

    gl::g_splitProcessID = savedPid;
    gl::g_splitCount = savedN;
}

TEST(lb_split, phase_helpers_exist) {
    // The performElem phase-split helpers exist with their declared member
    // signatures (this compiles only if each is present). Their behaviour is
    // pure extraction, covered by the byte-identical full-pipeline gate.
    auto p1 = &gl::ExpressionAnalyzer::performElemPhase1;
    auto p2 = &gl::ExpressionAnalyzer::performElemPhase2;
    auto p3 = &gl::ExpressionAnalyzer::performElemPhase3;
    auto e2 = &gl::ExpressionAnalyzer::performElem2;
    ASSERT_TRUE(p1 != nullptr);
    ASSERT_TRUE(p2 != nullptr);
    ASSERT_TRUE(p3 != nullptr);
    ASSERT_TRUE(e2 != nullptr);
}

TEST(lb_split, rt_note_iterations_here_no_tracker_is_noop) {
    // The _HERE iteration helper no-ops when no tracker is active on this thread
    // (g_currentThreadTracker == nullptr) — the phase-2 self-gating path. Must
    // not crash with no tracker.
    gl::rt_tracker::rtNoteIterationsHere(1);
    gl::rt_tracker::rtNoteIterationsHere(0);
    ASSERT_TRUE(true);
}

TEST(lb_split, is_straggler_fair_share) {
    // The split TRIGGER: an LB is a straggler iff its total work exceeds the idle-
    // core fair-share (totalWork / cores) AND clears the setup break-even floor. The
    // work is the split-invariant SUM of the LB's parts (D-117). Fair-share is
    // masking-resistant and self-limiting to at most cores-1 LBs; integer arithmetic
    // keeps the verdict deterministic (two runs byte-identical).
    using EA = gl::ExpressionAnalyzer;
    const int cores = 8;
    const int64_t floor = 1000;

    // Below fair-share -> not a straggler. Total 800, fairShare 100; 100 is not > 100.
    ASSERT_FALSE(EA::isStraggler(100, 800, cores, floor));
    // Above fair-share AND above the floor -> straggler. fairShare 1000; 5000 > 1000.
    ASSERT_TRUE(EA::isStraggler(5000, 8000, cores, floor));
    // Above fair-share but BELOW the floor -> not a straggler (setup break-even).
    // fairShare 100; 500 > 100 but 500 < floor 1000.
    ASSERT_FALSE(EA::isStraggler(500, 800, cores, floor));
    // Exactly AT fair-share -> not a straggler (strict >).
    ASSERT_FALSE(EA::isStraggler(1000, 8000, cores, floor));
    // Just above fair-share and above the floor -> straggler.
    ASSERT_TRUE(EA::isStraggler(1001, 8000, cores, /*floor=*/1000));
    // Self-limiting: 5 equal LBs (each 1000, total 5000, C=4, fairShare 1250) ->
    // none exceeds fair-share, a balanced load splits nothing.
    ASSERT_FALSE(EA::isStraggler(1000, 5000, 4, 0));
    // One dominant LB among small ones IS flagged: total 1300, C=4, fairShare 325;
    // the 1000-work LB is > 325.
    ASSERT_TRUE(EA::isStraggler(1000, 1300, 4, 0));
    // Integer division: totalWork 7, cores 2 -> fairShare 3.
    ASSERT_TRUE(EA::isStraggler(4, 7, 2, 0));   // 4 > 3
    ASSERT_FALSE(EA::isStraggler(3, 7, 2, 0));  // 3 not > 3
}

TEST(lb_split, disable_lb_split_defaults_false) {
    // Production safety: the diagnostic split-disable flag must default off so a
    // normal build runs the barriered parallel path unchanged. It is set true
    // only for an RT / profiling run (one part per LB, RT tracker homed in
    // performElem2). See D-110.
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    ASSERT_FALSE(ana.parameters.disable_lb_split);
    ana.parameters.disable_lb_split = true;
    ASSERT_TRUE(ana.parameters.disable_lb_split);
    // split_growth_factor / max_number_splits / split_fallback_ratio /
    // fixed_number_splits / the two caps are now UNUSED no-op fields (the split
    // trigger is the statistics-driven isStraggler). Defaults pinned only to keep
    // config-file parsing stable.
    ASSERT_EQ(ana.parameters.split_growth_factor, 2);
    ASSERT_EQ(ana.parameters.max_number_splits, 32);
}

TEST(lb_split, min_split_work_defaults_positive) {
    // The straggler trigger's only tunable knob: the setup break-even floor below
    // which an LB is never split (per-bucket work would be under the fixed setup
    // cost). Must default positive so a trivially cheap iteration splits nothing.
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    ASSERT_TRUE(ana.parameters.min_split_work > 0);
}

TEST(lb_split, max_split_parts_bounds_the_expression_split) {
    // The one split dimension is the expression/bucket split, fanned to logicalCores
    // buckets, so the machine core count must fit under kMaxSplitParts (the
    // proveKernel entry assert). kStumpsPerBucketTarget is a positive balance factor.
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    ASSERT_TRUE(static_cast<int>(ana.logicalCores)
                <= gl::ExpressionAnalyzer::kMaxSplitParts);
    ASSERT_TRUE(gl::ExpressionAnalyzer::kStumpsPerBucketTarget > 0);
}

TEST(lb_split, name_map_lookup_is_non_minting) {
    // lookup is the read-only counterpart of encode the parallel LB-split
    // executors use on the shared LB (D-116).
    NameMapRig nmRig;
    gl::NameMap& nm = nmRig.nm;

    // "main" is interned at construction -> lookup returns MAIN_ID == encode.
    ASSERT_EQ(nm.lookup("main"), gl::NameMap::MAIN_ID);
    ASSERT_EQ(nm.lookup("main"), nm.encode("main"));

    // A fresh name is absent -> lookup returns the slot-0 sentinel and does NOT
    // mint it: a second lookup still returns 0 (only encode mints).
    const std::string fresh = "(=[42,43])";
    ASSERT_EQ(nm.lookup(fresh), 0);
    ASSERT_EQ(nm.lookup(fresh), 0);

    // encode mints it (non-zero id); lookup now returns that same id.
    const int16_t id = nm.encode(fresh);
    ASSERT_NE(id, 0);
    ASSERT_EQ(nm.lookup(fresh), id);
}

TEST(lb_split, burst_deactivates_symbol_signature) {
    // Pins the predicate signature the phase-2 sink calls per head record.
    using DeactFn = bool (gl::ExpressionAnalyzer::*)(const gl::Memory&,
                                                     const gl::FiringRecord&);
    DeactFn fn = &gl::ExpressionAnalyzer::burstDeactivates;
    ASSERT_TRUE(fn != nullptr);
}

TEST(lb_split, burst_deactivates_contradiction) {
    // primedForContradiction + neg(head) known at an ancestor scope -> doomed.
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    ana.parameters.compressor_mode = false;
    RecordRig rig;
    gl::Memory mb;
    mb.primedForContradiction = true;
    const int16_t negId = mb.nameMap.encode("!(in[2,1])");
    mb.intKnownStatements.insert(gl::StatementKey{ negId, gl::NameMap::MAIN_ID },
                                 gl::StatementFlags{ true, false, true, true });

    gl::FiringRecord fr =
        makeHeadRecord(rig, "(in[2,1])", "main", "implication", "(impA)", true, false);
    ASSERT_TRUE(ana.burstDeactivates(mb, fr));
}

TEST(lb_split, burst_deactivates_requires_known_negation) {
    // primed but neg(head) absent from intKnownStatements -> no contradiction yet.
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    ana.parameters.compressor_mode = false;
    RecordRig rig;
    gl::Memory mb;
    mb.primedForContradiction = true;

    gl::FiringRecord fr =
        makeHeadRecord(rig, "(in[2,1])", "main", "implication", "(impA)", true, false);
    ASSERT_FALSE(ana.burstDeactivates(mb, fr));
}

TEST(lb_split, burst_deactivates_plain_lb_never_fires) {
    // Not primed, not recursion: even with neg(head) known no LB role applies.
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    ana.parameters.compressor_mode = false;
    RecordRig rig;
    gl::Memory mb;
    const int16_t negId = mb.nameMap.encode("!(in[2,1])");
    mb.intKnownStatements.insert(gl::StatementKey{ negId, gl::NameMap::MAIN_ID },
                                 gl::StatementFlags{ true, false, true, true });

    gl::FiringRecord fr =
        makeHeadRecord(rig, "(in[2,1])", "main", "implication", "(impA)", true, false);
    ASSERT_FALSE(ana.burstDeactivates(mb, fr));
}

TEST(lb_split, burst_deactivates_vacuous_truth) {
    // isPartOfRecursion at "main" + neg(head) known -> vacuous-truth contradiction.
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    ana.parameters.compressor_mode = false;
    RecordRig rig;
    gl::Memory mb;
    mb.isPartOfRecursion = true;
    const int16_t negId = mb.nameMap.encode("!(in[2,1])");
    mb.intKnownStatements.insert(gl::StatementKey{ negId, gl::NameMap::MAIN_ID },
                                 gl::StatementFlags{ true, false, true, true });

    gl::FiringRecord fr =
        makeHeadRecord(rig, "(in[2,1])", "main", "implication", "(impA)", true, false);
    ASSERT_TRUE(ana.burstDeactivates(mb, fr));
}

TEST(lb_split, burst_deactivates_tobeproved_reached) {
    // isPartOfRecursion + head matches a main-scope toBeProved goal -> doomed.
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    ana.parameters.compressor_mode = false;
    RecordRig rig;
    gl::Memory mb;
    mb.isPartOfRecursion = true;
    mb.intToBeProved.assignSet(gl::packStatementKey(
        mb.nameMap.encode("(in[2,1])"), gl::NameMap::MAIN_ID), nullptr, 0);

    gl::FiringRecord fr =
        makeHeadRecord(rig, "(in[2,1])", "main", "implication", "(impA)", true, false);
    ASSERT_TRUE(ana.burstDeactivates(mb, fr));
}

TEST(lb_split, burst_deactivates_multigoal_keeps_burst_alive) {
    // Multi-goal induction LB: firing ONE of several toBeProved goals must NOT
    // deactivate. Under LB-split the other goals are owned by sibling parts that
    // have not fired yet; early-exiting here would cut them. Only the sole
    // remaining goal (toBeProved.size()==1) dooms the LB.
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    ana.parameters.compressor_mode = false;
    RecordRig rig;
    gl::Memory mb;
    mb.isPartOfRecursion = true;
    mb.intToBeProved.assignSet(gl::packStatementKey(
        mb.nameMap.encode("(in[2,1])"), gl::NameMap::MAIN_ID), nullptr, 0);
    mb.intToBeProved.assignSet(gl::packStatementKey(
        mb.nameMap.encode("(in[3,1])"), gl::NameMap::MAIN_ID), nullptr, 0);

    gl::FiringRecord fr =
        makeHeadRecord(rig, "(in[2,1])", "main", "implication", "(impA)", true, false);
    ASSERT_FALSE(ana.burstDeactivates(mb, fr));
}

TEST(lb_split, burst_deactivates_direct_lb_sole_goal) {
    // Direct (non-recursion) LB reaching its SOLE toBeProved goal must now
    // deactivate: the "toBeProved reached" early-exit is no longer gated on
    // isPartOfRecursion, so direct proofs are eligible too.
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    ana.parameters.compressor_mode = false;
    RecordRig rig;
    gl::Memory mb;
    mb.isPartOfRecursion = false;   // DIRECT proof, not an induction triad
    mb.intToBeProved.assignSet(gl::packStatementKey(
        mb.nameMap.encode("(in[2,1])"), gl::NameMap::MAIN_ID), nullptr, 0);

    gl::FiringRecord fr =
        makeHeadRecord(rig, "(in[2,1])", "main", "implication", "(impA)", true, false);
    ASSERT_TRUE(ana.burstDeactivates(mb, fr));
}

TEST(lb_split, burst_deactivates_ce_filter) {
    // CE-filter LB (contradictionIndex >= 0) + neg(head) a loaded fact -> the
    // conjecture is refuted, so the burst is doomed and stops early.
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    ana.parameters.compressor_mode = false;
    RecordRig rig;
    gl::Memory mb;
    mb.contradictionIndex = 5;          // CE LB under test; not primed, not recursion
    const int16_t negId = mb.nameMap.encode("!(in[2,1])");
    mb.intKnownStatements.insert(gl::StatementKey{ negId, gl::NameMap::MAIN_ID },
                                 gl::StatementFlags{ true, false, true, true });

    gl::FiringRecord fr =
        makeHeadRecord(rig, "(in[2,1])", "main", "implication", "(impA)", true, false);
    ASSERT_TRUE(ana.burstDeactivates(mb, fr));
}

TEST(lb_split, burst_deactivates_ce_filter_requires_known_negation) {
    // CE LB but neg(head) absent from intKnownStatements -> no refutation yet,
    // the burst continues.
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    ana.parameters.compressor_mode = false;
    RecordRig rig;
    gl::Memory mb;
    mb.contradictionIndex = 5;

    gl::FiringRecord fr =
        makeHeadRecord(rig, "(in[2,1])", "main", "implication", "(impA)", true, false);
    ASSERT_FALSE(ana.burstDeactivates(mb, fr));
}

TEST(lb_split, burst_sink_runs_uncapped_and_gates_early_exit_on_multipart) {
    // BurstSink no longer caps on the submatch count -- bursts run to COMPLETION
    // (a heavy LB is split preemptively next iteration, not truncated). consume
    // still runs the dependency skip (a request element must be a known statement
    // before firing). The early-exit stop is honored ONLY for a SINGLE-part LB
    // (g_isMultiPart == false); a multi-part LB ignores the sibling-set stop and
    // runs every part to completion (I-76).
    gl::ExpressionAnalyzer ana(std::string("Peano"));
    gl::Memory mb;
    RecordRig rig;
    std::atomic<bool> stop{ false };
    const bool savedMulti = gl::g_isMultiPart;
    gl::g_isMultiPart = false;
    gl::BurstSink sink{ &ana, &mb, 0, &rig.pages, &stop,
                        gl::SealedRecordCursor<gl::FiringRecord>(rig.pages) };

    // No cap: canAccept stays true however high the submatch count climbs
    // (single-part, stop clear).
    gl::ExpressionAnalyzer::g_growthMatchCount = 0;
    ASSERT_TRUE(sink.canAccept());
    gl::ExpressionAnalyzer::g_growthMatchCount = 1 << 20;
    ASSERT_TRUE(sink.canAccept());
    gl::ExpressionAnalyzer::g_growthMatchCount = 0;

    // Dependency skip: a 1-element request whose element is absent from
    // intKnownStatements is counted (produced) but not fired.
    gl::IntEncodedExpr ie{};
    ie.originalId = 7;
    ie.validityId = gl::NameMap::MAIN_ID;
    gl::StaticRequest req{};
    req.count = 1;
    req.intExprs[0] = &ie;
    ASSERT_TRUE(sink.consume(req));        // dependency unmet -> skipped, keep going
    ASSERT_EQ((int)sink.produced, 1);
    ASSERT_EQ(rig.pages.recordCount(), 0); // nothing fired

    // Early-exit gate: with the stop set, a SINGLE-part LB closes...
    stop.store(true);
    gl::g_isMultiPart = false;
    ASSERT_FALSE(sink.canAccept());
    // ...but a MULTI-part LB ignores the sibling-set stop and keeps generating.
    gl::g_isMultiPart = true;
    ASSERT_TRUE(sink.canAccept());

    gl::g_isMultiPart = savedMulti;
    stop.store(false);
    gl::ExpressionAnalyzer::g_growthMatchCount = 0;  // reset for later tests
}
