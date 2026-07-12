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
/// @brief Unit tests for the LB deload machinery (`memory_infra/lb_deload.*`)
///        and the `LbMemory` aggregate it serializes.
///
/// @details
/// Pins the file format's determinism contracts: round-trip losslessness,
/// double-dump byte identity, canonical bytes after page-fragmenting
/// churn (the "straightening" property), multi-file splitting, the deload
/// ordinal (injective names, lifetime stability), registry ordering, and
/// the purge. Test files land under
/// `.debug/test_deload_*` directories (gitignored, recreated per run).

#include "test_harness.hpp"

#include "../memory_infra/lb_deload.hpp"
#include "../memory.hpp"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace {

    const gl::StaticMemoryConfig kDeloadTestCfg{ 1 << 20, 1 << 18 };

    /// Patterned statement row; value-initialized first so the arrays
    /// beyond the written slots are deterministic zero (IntEncodedExpr has
    /// no padding holes: 8 + 5*MAX_ARITY int16 fields).
    gl::IntEncodedExpr makeExpr(int16_t seed) {
        gl::IntEncodedExpr e{};
        e.nameId = seed;
        e.negation = static_cast<int16_t>(seed & 1);
        e.arity = 3;
        e.maxIteration = -1;
        e.originalId = static_cast<int16_t>(seed + 1);
        e.validityId = 1;
        e.isHypo = 0;
        e.isAnchor = 0;
        for (int16_t k = 0; k < gl::ExecutionParameters::MAX_ARITY; ++k) {
            e.argId[k] = static_cast<int16_t>(seed + k);
            e.argUnchangeable[k] = static_cast<int16_t>(k & 1);
            e.argIteration[k] = -1;
            e.argLevPlus1[k] = 0;
            e.argFullId[k] = static_cast<int16_t>(seed - k);
        }
        return e;
    }

    bool sameExpr(const gl::IntEncodedExpr& a, const gl::IntEncodedExpr& b) {
        return std::memcmp(&a, &b, sizeof(gl::IntEncodedExpr)) == 0;
    }

    std::vector<char> readFileBytes(const std::filesystem::path& p) {
        std::ifstream in(p, std::ios::binary);
        return std::vector<char>((std::istreambuf_iterator<char>(in)),
                                 std::istreambuf_iterator<char>());
    }

    std::filesystem::path freshDir(const std::string& name) {
        const std::filesystem::path dir =
            std::filesystem::path(".debug") / name;
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
        std::filesystem::create_directories(dir, ec);
        return dir;
    }

}

TEST(static_memory, int_stmt_view_wraps_paged_vector) {
    gl::GlobalMemoryManager g;
    g.init(kDeloadTestCfg);
    gl::LbArena lb(&g);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::PagedVector<gl::IntEncodedExpr> source(&lb, &dirty);
    for (int16_t i = 0; i < 5; ++i) source.push_back(makeExpr(i));
    const gl::IntStmtView view(source);
    ASSERT_EQ(view.size(), 5);
    ASSERT_FALSE(view.empty());
    for (int32_t i = 0; i < view.size(); ++i)
        ASSERT_TRUE(sameExpr(view[i], source[i]));
    // Element pointers alias the viewed storage (the pointer-array
    // plumbing in the request generators relies on this).
    ASSERT_EQ(&view[0], &source[0]);
    const gl::IntStmtView emptyView;
    ASSERT_EQ(emptyView.size(), 0);
    ASSERT_TRUE(emptyView.empty());
}

TEST(lb_deload, file_name_shape) {
    // Base part: lb<ordinal>_<part>_of_<N>.bin (pure ordinal — no chain
    // prefix, no hash; the ordinal is the file's injective identity).
    ASSERT_EQ(gl::lbdeload::deloadFileName(7, 0, 2, 3),
              std::string("lb7_2_of_3.bin"));
    // Tail sets carry a t<k>_ component so they never collide with the
    // base image or each other.
    ASSERT_EQ(gl::lbdeload::deloadFileName(7, 4, 1, 1),
              std::string("lb7_t4_1_of_1.bin"));
}

TEST(lb_deload, file_name_carries_ordinal_not_hash) {
    // Two distinct ordinals yield distinct names — injective by
    // construction (the whole point of replacing the chain hash).
    const std::string a = gl::lbdeload::deloadFileName(5, 0, 1, 1);
    const std::string b = gl::lbdeload::deloadFileName(6, 0, 1, 1);
    ASSERT_EQ(a, std::string("lb5_1_of_1.bin"));
    ASSERT_NE(a, b);
    ASSERT_TRUE(a.find("lb5_") == 0);
}

TEST(lb_deload, memory_deload_reload_preserves_all_four_hashmemories) {
    // Regression for the hc-branch 182-check drop: the two transient HashMemory
    // instances (localHashMemoryDelta / workingMemory) moved onto the cold arena
    // and must now survive an LB's deload/reload round-trip exactly like the
    // persistent pair (overall / local). An evicted LB's next-iteration Batch 5
    // reads the prior cycle's localHashMemoryDelta, so a lossy reload here
    // silently drops firings. Memory-level path (deloadStaticContainers /
    // ensureLoaded); the four HashMemory instances are now LbMemory members,
    // dumped/reloaded through lbMemory.visitContainers (D-147).
    const std::filesystem::path dir = freshDir("test_deload_hashmem_all_four");
    gl::Memory m;
    m.setExprKey("(test_deload_hashmem_lb)");
    for (int16_t i = 0; i < 8; ++i) m.intEncodedStatements.push_back(makeExpr(i));
    int16_t buf[2] = { 7, 9 };
    const gl::NormKey nk{ 1, std::vector<int16_t>(buf, buf + 2) };
    m.overallHashMemory.encodedMap.assignRun(
        nk, std::vector<gl::LocalMemoryValue>{ gl::LocalMemoryValue{} });
    m.localHashMemory.encodedMap.assignRun(
        nk, std::vector<gl::LocalMemoryValue>{ gl::LocalMemoryValue{} });
    m.localHashMemoryDelta.encodedMap.assignRun(
        nk, std::vector<gl::LocalMemoryValue>{ gl::LocalMemoryValue{} });
    m.workingMemory.encodedMap.assignRun(
        nk, std::vector<gl::LocalMemoryValue>{ gl::LocalMemoryValue{} });
    ASSERT_EQ(m.overallHashMemory.encodedMap.count(), 1);
    ASSERT_EQ(m.localHashMemory.encodedMap.count(), 1);
    ASSERT_EQ(m.localHashMemoryDelta.encodedMap.count(), 1);
    ASSERT_EQ(m.workingMemory.encodedMap.count(), 1);

    m.deloadStaticContainers(dir.string());
    ASSERT_FALSE(m.lbMemory.manager.resident());

    m.ensureLoaded(dir.string());
    ASSERT_TRUE(m.lbMemory.manager.resident());
    // All four must come back; the transient pair is the regression target.
    ASSERT_EQ(m.overallHashMemory.encodedMap.count(), 1);
    ASSERT_EQ(m.localHashMemory.encodedMap.count(), 1);
    ASSERT_EQ(m.localHashMemoryDelta.encodedMap.count(), 1);
    ASSERT_EQ(m.workingMemory.encodedMap.count(), 1);
}

TEST(lb_deload, roundtrip_single_file_all_four_containers) {
    gl::GlobalMemoryManager g;
    g.init(kDeloadTestCfg);
    gl::LbMemory src(&g);
    for (int16_t i = 0; i < 100; ++i)
        src.intEncodedStatements.push_back(makeExpr(i));
    for (int16_t i = 0; i < 30; ++i)
        src.intLocalEncodedStatements.push_back(makeExpr(
            static_cast<int16_t>(200 + i)));
    for (int16_t i = 0; i < 7; ++i)
        src.intLocalEncodedStatementsDelta.push_back(makeExpr(
            static_cast<int16_t>(300 + i)));
    for (int16_t i = 0; i < 4; ++i)
        src.intExternalStatements.push_back(makeExpr(
            static_cast<int16_t>(400 + i)));

    const std::filesystem::path dir = freshDir("test_deload_roundtrip");
    const std::string chain = "(=[a,b])__(AnchorPeano[N])";
    const std::vector<std::string> files = gl::lbdeload::dumpLbMemory(
        src, chain, 0, dir, kDeloadTestCfg.blockBytes);
    ASSERT_EQ(files.size(), static_cast<size_t>(1));

    gl::LbMemory dst(&g);
    gl::lbdeload::loadLbMemory(dst, chain, files, dir);
    ASSERT_EQ(dst.intEncodedStatements.size(), 100);
    for (int32_t i = 0; i < 100; ++i)
        ASSERT_TRUE(sameExpr(dst.intEncodedStatements[i],
                             src.intEncodedStatements[i]));
    ASSERT_EQ(dst.intLocalEncodedStatements.size(), 30);
    for (int32_t i = 0; i < 30; ++i)
        ASSERT_TRUE(sameExpr(dst.intLocalEncodedStatements[i],
                             src.intLocalEncodedStatements[i]));
    ASSERT_EQ(dst.intLocalEncodedStatementsDelta.size(), 7);
    for (int32_t i = 0; i < 7; ++i)
        ASSERT_TRUE(sameExpr(dst.intLocalEncodedStatementsDelta[i],
                             src.intLocalEncodedStatementsDelta[i]));
    ASSERT_EQ(dst.intExternalStatements.size(), 4);
    for (int32_t i = 0; i < 4; ++i)
        ASSERT_TRUE(sameExpr(dst.intExternalStatements[i],
                             src.intExternalStatements[i]));
}

TEST(lb_deload, cold_string_table_roundtrip_with_tail) {
    gl::GlobalMemoryManager g;
    g.init(kDeloadTestCfg);
    gl::LbMemory src(&g);
    src.intEncodedStatements.push_back(makeExpr(1));
    const std::string base[] = { "(p[a,marker])",
                                 "(in2[u_x,int_0_1,u_z])",
                                 "(=[v1,zero])" };
    for (const std::string& s : base)
        src.templateStrings.intern(gl::StrSpan(s));

    const std::filesystem::path dir = freshDir("test_deload_cold_strings");
    const std::string chain = "(coldstr[chain])";
    std::vector<std::string> files = gl::lbdeload::dumpLbMemory(
        src, chain, 0, dir, kDeloadTestCfg.blockBytes);

    // Per-tag counts at the base — the production `deloadedCounts`
    // (for the bytes tag this is a BYTE total).
    std::vector<int32_t> counts;
    src.visitContainers(
        [&counts](gl::LbMemory::ContainerTag, const auto& c) {
            counts.push_back(c.size());
        });

    // Appended-only window: one more row, two more strings.
    src.intEncodedStatements.push_back(makeExpr(2));
    const std::string tailA = "(tail[one,marker])";
    const std::string tailB = "(tail[two])";
    src.templateStrings.intern(gl::StrSpan(tailA));
    src.templateStrings.intern(gl::StrSpan(tailB));
    const std::vector<std::string> tailFiles = gl::lbdeload::dumpLbMemoryTail(
        src, chain, 0, counts, 1, dir, kDeloadTestCfg.blockBytes);
    files.insert(files.end(), tailFiles.begin(), tailFiles.end());

    // Base + tail reproduce the table exactly: ids, bytes, lookups.
    gl::LbMemory dst(&g);
    gl::lbdeload::loadLbMemory(dst, chain, files, dir);
    ASSERT_EQ(dst.intEncodedStatements.size(), 2);
    ASSERT_EQ(dst.templateStrings.count(), 5);
    for (int32_t id = 1; id <= 5; ++id) {
        ASSERT_TRUE(gl::equalSpans(dst.templateStrings.view(id),
                                   src.templateStrings.view(id)));
    }
    for (const std::string& s : base) {
        ASSERT_EQ(dst.templateStrings.lookup(gl::StrSpan(s)),
                  src.templateStrings.lookup(gl::StrSpan(s)));
    }
    ASSERT_EQ(dst.templateStrings.lookup(gl::StrSpan(tailA)), 4);
    ASSERT_EQ(dst.templateStrings.lookup(gl::StrSpan(tailB)), 5);
}

TEST(lb_deload, empty_aggregate_roundtrip_writes_header_only_file) {
    gl::GlobalMemoryManager g;
    g.init(kDeloadTestCfg);
    gl::LbMemory src(&g);
    // blockBytes() is read by the dump header, so the manager's global
    // must be initialized - but no block may be consumed by an empty dump.
    const std::filesystem::path dir = freshDir("test_deload_empty");
    const std::vector<std::string> files =
        gl::lbdeload::dumpLbMemory(src, "chain", 0, dir, 1000);
    ASSERT_EQ(files.size(), static_cast<size_t>(1));
    ASSERT_EQ(g.blocksInUse(), static_cast<int64_t>(0));
    gl::LbMemory dst(&g);
    gl::lbdeload::loadLbMemory(dst, "chain", files, dir);
    ASSERT_EQ(dst.intEncodedStatements.size(), 0);
}

TEST(lb_deload, double_dump_is_byte_identical) {
    gl::GlobalMemoryManager g;
    g.init(kDeloadTestCfg);
    gl::LbMemory src(&g);
    for (int16_t i = 0; i < 64; ++i)
        src.intEncodedStatements.push_back(makeExpr(i));
    const std::filesystem::path dirA = freshDir("test_deload_double_a");
    const std::filesystem::path dirB = freshDir("test_deload_double_b");
    const std::string chain = "double";
    const auto filesA =
        gl::lbdeload::dumpLbMemory(src, chain, 0, dirA, 1 << 18);
    const auto filesB =
        gl::lbdeload::dumpLbMemory(src, chain, 0, dirB, 1 << 18);
    ASSERT_EQ(filesA.size(), filesB.size());
    for (size_t i = 0; i < filesA.size(); ++i) {
        ASSERT_EQ(filesA[i], filesB[i]);
        const auto a = readFileBytes(dirA / filesA[i]);
        const auto b = readFileBytes(dirB / filesB[i]);
        ASSERT_EQ(a.size(), b.size());
        ASSERT_TRUE(std::memcmp(a.data(), b.data(), a.size()) == 0);
    }
}

TEST(lb_deload, churned_container_dumps_canonical_bytes) {
    // The "straightening" property: page-fragmenting churn must leave no
    // trace in the dumped bytes - they are a pure function of logical
    // content.
    gl::GlobalMemoryManager g;
    g.init(kDeloadTestCfg);

    gl::LbMemory clean(&g);
    for (int16_t i = 0; i < 50; ++i)
        clean.intEncodedStatements.push_back(makeExpr(i));

    gl::LbMemory churned(&g);
    for (int16_t i = 0; i < 200; ++i)
        churned.intEncodedStatements.push_back(makeExpr(
            static_cast<int16_t>(1000 + i)));
    churned.intEncodedStatements.clear();
    for (int16_t i = 0; i < 50; ++i) {
        churned.intEncodedStatements.push_back(makeExpr(i));
        churned.intEncodedStatements.push_back(makeExpr(
            static_cast<int16_t>(500 + i)));
    }
    for (int32_t i = churned.intEncodedStatements.size(); i-- > 0;)
        if (churned.intEncodedStatements[i].nameId >= 500)
            churned.intEncodedStatements.erase(i);

    const std::filesystem::path dirA = freshDir("test_deload_clean");
    const std::filesystem::path dirB = freshDir("test_deload_churned");
    const std::string chain = "churn";
    const auto filesA =
        gl::lbdeload::dumpLbMemory(clean, chain, 0, dirA, 1 << 18);
    const auto filesB =
        gl::lbdeload::dumpLbMemory(churned, chain, 0, dirB, 1 << 18);
    ASSERT_EQ(filesA.size(), filesB.size());
    for (size_t i = 0; i < filesA.size(); ++i) {
        const auto a = readFileBytes(dirA / filesA[i]);
        const auto b = readFileBytes(dirB / filesB[i]);
        ASSERT_EQ(a.size(), b.size());
        ASSERT_TRUE(std::memcmp(a.data(), b.data(), a.size()) == 0);
    }
}

TEST(lb_deload, multi_file_split_and_roundtrip) {
    gl::GlobalMemoryManager g;
    g.init(kDeloadTestCfg);
    gl::LbMemory src(&g);
    for (int16_t i = 0; i < 10; ++i)
        src.intEncodedStatements.push_back(makeExpr(i));
    // 10 * 176 = 1760 payload bytes at 500 bytes per file = 4 parts.
    const std::filesystem::path dir = freshDir("test_deload_split");
    const std::string chain = "split";
    const std::vector<std::string> files =
        gl::lbdeload::dumpLbMemory(src, chain, 0, dir, 500);
    ASSERT_EQ(files.size(), static_cast<size_t>(4));
    ASSERT_TRUE(files[0].find("_1_of_4.bin") != std::string::npos);
    ASSERT_TRUE(files[3].find("_4_of_4.bin") != std::string::npos);

    gl::LbMemory dst(&g);
    gl::lbdeload::loadLbMemory(dst, chain, files, dir);
    ASSERT_EQ(dst.intEncodedStatements.size(), 10);
    for (int32_t i = 0; i < 10; ++i)
        ASSERT_TRUE(sameExpr(dst.intEncodedStatements[i],
                             src.intEncodedStatements[i]));
}

TEST(lb_deload, build_lb_chain_string_walks_to_root) {
    gl::Memory root; // root sentinel: empty exprKey, null parent
    gl::Memory child;
    child.setExprKey("(=[a,b])");
    child.parentMemory = &root;
    gl::Memory grand;
    grand.setExprKey("(in2[x,N])");
    grand.parentMemory = &child;
    ASSERT_EQ(gl::buildLbChainString(grand),
              std::string("(in2[x,N])__(=[a,b])__"));
    ASSERT_EQ(gl::buildLbChainString(child), std::string("(=[a,b])__"));
    ASSERT_EQ(gl::buildLbChainString(root), std::string(""));
}

TEST(lb_deload, memory_deload_reload_roundtrip) {
    gl::Memory m;
    m.setExprKey("(=[a,b])");
    for (int16_t i = 0; i < 80; ++i)
        m.intEncodedStatements.push_back(makeExpr(i));
    const gl::ExpressionWithValidity outgoing("(in[a,N])", "main");
    m.insertMailOutStatement(outgoing, std::set<int>{ 1, 4 });
    m.addMailOutOrigin(outgoing,
        std::make_pair("disintegration",
            std::vector<gl::ExpressionWithValidity>{
                gl::ExpressionWithValidity("(&[x,y])", "main") }),
        8);
    const int64_t mailBytesBefore = m.mailOutLiveBytes;
    ASSERT_TRUE(m.mailOutPending);
    ASSERT_TRUE(mailBytesBefore > 0);
    ASSERT_TRUE(m.lbMemory.manager.blocksHeld() > 0);

    const std::filesystem::path dir = freshDir("test_deload_memory");
    m.deloadStaticContainers(dir.string());
    ASSERT_FALSE(m.lbMemory.manager.resident());
    ASSERT_EQ(m.lbMemory.manager.blocksHeld(), static_cast<int64_t>(0));
    ASSERT_FALSE(m.deloadFiles.empty());
    // Cold metadata path: size() is residency-asserted, so counts come
    // from the record made at deload.
    ASSERT_EQ(m.intEncodedStatementsCount(), 80);

    m.ensureLoaded(dir.string());
    ASSERT_TRUE(m.lbMemory.manager.resident());
    ASSERT_EQ(m.intEncodedStatements.size(), 80);
    ASSERT_EQ(m.intEncodedStatementsCount(), 80); // resident branch
    for (int16_t i = 0; i < 80; ++i)
        ASSERT_TRUE(sameExpr(m.intEncodedStatements[i], makeExpr(i)));
    ASSERT_TRUE(m.mailOutPending);
    ASSERT_EQ(m.mailOutLiveBytes, mailBytesBefore);
    const gl::Mail reloaded = gl::routingMailOutToHeap(
        m.mailOut, m.mailOutInterner);
    ASSERT_EQ(reloaded.statements.size(), static_cast<std::size_t>(1));
    ASSERT_TRUE(reloaded.statements.count(
        std::make_pair(outgoing, std::set<int>{ 1, 4 })) == 1);
    ASSERT_EQ(reloaded.exprOriginMap.at(outgoing).size(),
              static_cast<std::size_t>(1));
    ASSERT_EQ(reloaded.exprOriginMap.at(outgoing)[0].first,
              std::string("disintegration"));

    // Resident no-op path (the touch points call unconditionally).
    m.ensureLoaded(dir.string());
    ASSERT_EQ(m.intEncodedStatements.size(), 80);
    m.clearMailOut();
    ASSERT_FALSE(m.mailOutPending);
    ASSERT_EQ(m.mailOutLiveBytes, 0);
    ASSERT_TRUE(m.mailOut.empty());
    ASSERT_EQ(m.mailOutInterner.internedCount(), 0);
}

// NameMap validity metadata (the validityNodes parent-pointer forest)
// round-trips through a Memory deload + reload: ids stay stable and the
// ancestor/stack structure reproduces exactly.
TEST(lb_deload, namemap_metadata_roundtrip) {
    gl::Memory m;
    m.setExprKey("(=[a,b])");
    const int16_t mainId = m.nameMap.encode("main");
    const int16_t a = m.nameMap.encodePush(mainId, "a");
    const int16_t b = m.nameMap.encodePush(a, "b");        // main < a < b
    const int16_t c = m.nameMap.encodePush(mainId, "c");   // diverges from a/b
    m.intEncodedStatements.push_back(makeExpr(7));          // hold a block
    ASSERT_TRUE(m.lbMemory.manager.blocksHeld() > 0);

    // Pre-deload structure.
    ASSERT_EQ(m.nameMap.stackSize(), 5);                    // slot0 + main + a,b,c
    ASSERT_EQ(m.nameMap.ancLen(b), 3);                      // {main, a, b}
    ASSERT_TRUE(m.nameMap.comparable(a, b));
    ASSERT_FALSE(m.nameMap.comparable(b, c));               // diverge

    const std::filesystem::path dir = freshDir("test_deload_namemap_meta");
    m.deloadStaticContainers(dir.string());
    ASSERT_FALSE(m.lbMemory.manager.resident());

    m.ensureLoaded(dir.string());
    ASSERT_TRUE(m.lbMemory.manager.resident());

    // Ids stay stable (re-interned in id order) and the metadata reproduces.
    ASSERT_EQ(m.nameMap.lookup("main_boundary_a"), a);
    ASSERT_EQ(m.nameMap.lookup("main_boundary_a_boundary_b"), b);
    ASSERT_EQ(m.nameMap.lookup("main_boundary_c"), c);
    ASSERT_EQ(m.nameMap.stackSize(), 5);
    ASSERT_EQ(m.nameMap.ancLen(b), 3);
    ASSERT_EQ(m.nameMap.ancAt(b, 0), mainId);
    ASSERT_EQ(m.nameMap.ancAt(b, 1), a);
    ASSERT_EQ(m.nameMap.ancAt(b, 2), b);
    ASSERT_EQ(m.nameMap.stackLen(b), 2);
    ASSERT_TRUE(m.nameMap.comparable(a, b));
    int16_t v = 0;
    ASSERT_TRUE(m.nameMap.verdict(a, b, v));
    ASSERT_EQ(v, static_cast<int16_t>(-1));                 // a is ancestor of b
    ASSERT_FALSE(m.nameMap.comparable(b, c));               // still diverge
}

TEST(lb_deload, ensure_loaded_for_read_revives_discharged_lb) {
    // The sanctioned post-prove READ reload (D-158):
    // a discharged-and-drained LB's cold string content comes back for
    // the chapter walk; ensureLoaded itself stays forbidden there.
    gl::Memory m;
    m.setExprKey("__contradiction__(=[10,11])");
    const std::string origin = "!(=[10,11])";
    const std::string scope = "main";
    const int32_t exprId = m.originInterner.encode(origin);
    const int32_t scopeId = m.originInterner.encode(scope);
    m.intEncodedStatements.push_back(makeExpr(7));

    const std::filesystem::path dir = freshDir("test_deload_read_revive");
    m.dischargedForever = true;            // discharged, then drained:
    m.deloadStaticContainers(dir.string());
    ASSERT_FALSE(m.lbMemory.manager.resident());

    m.ensureLoadedForRead(dir.string());
    ASSERT_TRUE(m.lbMemory.manager.resident());
    // Ids and content reproduce exactly — the lookups the export's
    // hasOrigins probe makes now hit.
    ASSERT_EQ(m.originInterner.lookup(origin), exprId);
    ASSERT_EQ(m.originInterner.lookup(scope), scopeId);
    ASSERT_EQ(m.originInterner.decode(exprId), origin);
    // Resident no-op path.
    m.ensureLoadedForRead(dir.string());
    ASSERT_EQ(m.originInterner.internedCount(), 2);
}

TEST(lb_deload, memory_deload_reload_cycle_repeats) {
    gl::Memory m;
    m.setExprKey("(in3[i0,i1,v1,+])");
    for (int16_t i = 0; i < 40; ++i)
        m.intEncodedStatements.push_back(makeExpr(i));
    const std::filesystem::path dir = freshDir("test_deload_cycle");
    for (int round = 0; round < 3; ++round) {
        m.deloadStaticContainers(dir.string());
        m.ensureLoaded(dir.string());
        // Containers keep working after each cycle.
        m.intEncodedStatements.push_back(
            makeExpr(static_cast<int16_t>(100 + round)));
    }
    ASSERT_EQ(m.intEncodedStatements.size(), 43);
    ASSERT_EQ(m.intEncodedStatements[42].nameId,
              static_cast<int16_t>(102));
}

TEST(lb_deload, skip_unchanged_deload_keeps_files_and_content) {
    gl::Memory m;
    m.setExprKey("(in2[a,b,c])");
    for (int16_t i = 0; i < 25; ++i)
        m.intEncodedStatements.push_back(makeExpr(i));
    const std::filesystem::path dir = freshDir("test_deload_skip");

    // Dump 1 (dirty -> writes), reload (RAM == disk, flag cleared).
    m.deloadStaticContainers(dir.string());
    m.ensureLoaded(dir.string());
    const auto writtenAt =
        std::filesystem::last_write_time(dir / m.deloadFiles.front());

    // No mutation: the second deload must SKIP the rewrite — the file's
    // write time stays byte-for-byte the dump-1 stamp.
    m.deloadStaticContainers(dir.string());
    ASSERT_TRUE(std::filesystem::last_write_time(
                    dir / m.deloadFiles.front()) == writtenAt);
    m.ensureLoaded(dir.string());
    ASSERT_EQ(m.intEncodedStatements.size(), 25);
    for (int16_t i = 0; i < 25; ++i)
        ASSERT_TRUE(sameExpr(m.intEncodedStatements[i], makeExpr(i)));

    // Mutation: the next deload must rewrite, and the reload must see
    // the new content.
    m.intEncodedStatements.push_back(makeExpr(99));
    m.deloadStaticContainers(dir.string());
    m.ensureLoaded(dir.string());
    ASSERT_EQ(m.intEncodedStatements.size(), 26);
    ASSERT_TRUE(sameExpr(m.intEncodedStatements[25], makeExpr(99)));
}

TEST(lb_deload, append_only_deload_writes_tail_and_roundtrips) {
    gl::Memory m;
    m.setExprKey("(in3[a,b,c,d])");
    for (int16_t i = 0; i < 40; ++i)
        m.intEncodedStatements.push_back(makeExpr(i));
    const std::filesystem::path dir = freshDir("test_deload_tail");

    // Base dump, reload, append a few rows (AppendedOnly window).
    m.deloadStaticContainers(dir.string());
    ASSERT_EQ(m.deloadFiles.size(), static_cast<size_t>(1));
    m.ensureLoaded(dir.string());
    const auto baseWrittenAt =
        std::filesystem::last_write_time(dir / m.deloadFiles.front());
    for (int16_t i = 0; i < 5; ++i)
        m.intEncodedStatements.push_back(makeExpr(
            static_cast<int16_t>(500 + i)));

    // Tail deload: one new tail file, base file untouched.
    m.deloadStaticContainers(dir.string());
    ASSERT_EQ(m.deloadFiles.size(), static_cast<size_t>(2));
    ASSERT_TRUE(m.deloadFiles[1].find("_t1_") != std::string::npos);
    ASSERT_TRUE(std::filesystem::last_write_time(
                    dir / m.deloadFiles.front()) == baseWrittenAt);

    // Reload = base + tail; content exact.
    m.ensureLoaded(dir.string());
    ASSERT_EQ(m.intEncodedStatements.size(), 45);
    for (int16_t i = 0; i < 40; ++i)
        ASSERT_TRUE(sameExpr(m.intEncodedStatements[i], makeExpr(i)));
    for (int16_t i = 0; i < 5; ++i)
        ASSERT_TRUE(sameExpr(m.intEncodedStatements[40 + i],
                             makeExpr(static_cast<int16_t>(500 + i))));

    // A second appended-only window stacks a second tail.
    m.intEncodedStatements.push_back(makeExpr(600));
    m.deloadStaticContainers(dir.string());
    ASSERT_EQ(m.deloadFiles.size(), static_cast<size_t>(3));
    ASSERT_TRUE(m.deloadFiles[2].find("_t2_") != std::string::npos);
    m.ensureLoaded(dir.string());
    ASSERT_EQ(m.intEncodedStatements.size(), 46);
    ASSERT_TRUE(sameExpr(m.intEncodedStatements[45], makeExpr(600)));
}

TEST(lb_deload, restructure_forces_full_compaction) {
    gl::Memory m;
    m.setExprKey("(in2[x,y,z])");
    for (int16_t i = 0; i < 30; ++i)
        m.intEncodedStatements.push_back(makeExpr(i));
    const std::filesystem::path dir = freshDir("test_deload_compact");
    m.deloadStaticContainers(dir.string());
    m.ensureLoaded(dir.string());
    m.intEncodedStatements.push_back(makeExpr(100));
    m.deloadStaticContainers(dir.string());
    ASSERT_EQ(m.deloadFiles.size(), static_cast<size_t>(2)); // base+tail
    m.ensureLoaded(dir.string());

    // An erase is a restructuring mutation: the next deload compacts to
    // a fresh single base image.
    m.intEncodedStatements.erase(0);
    m.deloadStaticContainers(dir.string());
    ASSERT_EQ(m.deloadFiles.size(), static_cast<size_t>(1));
    m.ensureLoaded(dir.string());
    ASSERT_EQ(m.intEncodedStatements.size(), 30);
    ASSERT_TRUE(sameExpr(m.intEncodedStatements[0], makeExpr(1)));
    ASSERT_TRUE(sameExpr(m.intEncodedStatements[29], makeExpr(100)));
}

TEST(lb_deload, tail_volume_threshold_forces_compaction) {
    gl::Memory m;
    m.setExprKey("(in[q,N])");
    for (int16_t i = 0; i < 20; ++i)
        m.intEncodedStatements.push_back(makeExpr(i));
    const std::filesystem::path dir = freshDir("test_deload_threshold");
    m.deloadStaticContainers(dir.string()); // base of 20 rows
    m.ensureLoaded(dir.string());
    // Appending >= base/4 rows exceeds the compaction fraction: the
    // deload must produce a fresh single base, not a tail.
    for (int16_t i = 0; i < 6; ++i)
        m.intEncodedStatements.push_back(makeExpr(
            static_cast<int16_t>(700 + i)));
    m.deloadStaticContainers(dir.string());
    ASSERT_EQ(m.deloadFiles.size(), static_cast<size_t>(1));
    m.ensureLoaded(dir.string());
    ASSERT_EQ(m.intEncodedStatements.size(), 26);
}

TEST(lb_deload, dump_without_release_keeps_resident_then_pressure_path) {
    gl::Memory m;
    m.setExprKey("(in2[r,s,t])");
    for (int16_t i = 0; i < 33; ++i)
        m.intEncodedStatements.push_back(makeExpr(i));
    const std::filesystem::path dir = freshDir("test_deload_writethrough");

    // Write-through dump: image fresh, LB stays resident with blocks.
    m.dumpStaticContainers(dir.string());
    ASSERT_TRUE(m.lbMemory.manager.resident());
    ASSERT_TRUE(m.lbMemory.manager.blocksHeld() > 0);
    ASSERT_FALSE(m.deloadFiles.empty());
    // ensureLoaded on the resident LB is a no-op; content untouched.
    m.ensureLoaded(dir.string());
    ASSERT_EQ(m.intEncodedStatements.size(), 33);

    // Pressure path: release the (fresh-imaged) blocks, then reload.
    m.releaseStaticBlocks();
    ASSERT_FALSE(m.lbMemory.manager.resident());
    ASSERT_EQ(m.lbMemory.manager.blocksHeld(), static_cast<int64_t>(0));
    m.ensureLoaded(dir.string());
    ASSERT_EQ(m.intEncodedStatements.size(), 33);
    for (int16_t i = 0; i < 33; ++i)
        ASSERT_TRUE(sameExpr(m.intEncodedStatements[i], makeExpr(i)));
}

TEST(lb_deload, purge_empties_and_recreates) {
    const std::filesystem::path dir = freshDir("test_deload_purge");
    std::ofstream(dir / "junk.bin", std::ios::binary) << "x";
    ASSERT_TRUE(std::filesystem::exists(dir / "junk.bin"));
    gl::lbdeload::purgeDeloadDirectory(dir);
    ASSERT_TRUE(std::filesystem::exists(dir));
    ASSERT_TRUE(std::filesystem::is_empty(dir));
}

TEST(lb_deload, registry_sorted_by_ordinal_with_extent_columns) {
    const std::filesystem::path dir = freshDir("test_deload_registry");
    std::map<int64_t, std::string> m;
    m[2] = "chainB";
    m[0] = "chainA";
    m[5] = "chainC";
    // Ordinal 2 owns an extent slab; 0 and 5 are slab-less (v3-only /
    // discharged / extent off) and write the -1/0 placeholder columns.
    std::map<int64_t, gl::SlabAllocation> slabs;
    slabs[2] = gl::SlabAllocation{ 262144, 524288 };
    gl::lbdeload::rewriteRegistry(m, slabs, dir);
    std::ifstream in(dir / "registry.txt");
    std::string line1, line2, line3;
    std::getline(in, line1);
    std::getline(in, line2);
    std::getline(in, line3);
    ASSERT_EQ(line1, std::string("0\t-1\t0\tchainA"));
    ASSERT_EQ(line2, std::string("2\t262144\t524288\tchainB"));
    ASSERT_EQ(line3, std::string("5\t-1\t0\tchainC"));
}

TEST(lb_deload, assign_deload_ordinal_is_monotone_unique) {
    gl::GlobalMemoryManager g;
    g.init(kDeloadTestCfg);
    ASSERT_EQ(g.assignDeloadOrdinal(), static_cast<int64_t>(0));
    ASSERT_EQ(g.assignDeloadOrdinal(), static_cast<int64_t>(1));
    ASSERT_EQ(g.assignDeloadOrdinal(), static_cast<int64_t>(2));
    g.registerDeloadOrdinal(0, "chainA");
    g.registerDeloadOrdinal(2, "chainC");
    ASSERT_EQ(g.deloadRegistry().size(), static_cast<size_t>(2));
    ASSERT_EQ(g.deloadRegistry().at(0), std::string("chainA"));
    ASSERT_EQ(g.deloadRegistry().at(2), std::string("chainC"));
    // resetDeloadRegistry clears the map but NOT the counter (process-
    // monotonic, so a persisting LB never collides a reused number).
    g.resetDeloadRegistry();
    ASSERT_TRUE(g.deloadRegistry().empty());
    ASSERT_EQ(g.assignDeloadOrdinal(), static_cast<int64_t>(3));
}

TEST(lb_deload, deload_ordinal_stable_across_base_and_tail) {
    gl::Memory m;
    m.setExprKey("(ordinal[base,tail])");
    for (int16_t i = 0; i < 40; ++i)
        m.intEncodedStatements.push_back(makeExpr(i));
    const std::filesystem::path dir = freshDir("test_deload_ord_tail");
    m.deloadStaticContainers(dir.string());
    const int64_t ord = m.deloadOrdinal;
    ASSERT_GE(ord, static_cast<int64_t>(0));
    m.ensureLoaded(dir.string());
    for (int16_t i = 0; i < 5; ++i)
        m.intEncodedStatements.push_back(
            makeExpr(static_cast<int16_t>(500 + i)));
    m.deloadStaticContainers(dir.string());  // AppendedOnly -> tail set
    ASSERT_EQ(m.deloadFiles.size(), static_cast<size_t>(2));
    ASSERT_EQ(m.deloadOrdinal, ord);  // stable across base + tail
    const std::string lb = "lb" + std::to_string(ord) + "_";
    for (const std::string& f : m.deloadFiles)
        ASSERT_TRUE(f.find(lb) == 0);
}

TEST(lb_deload, deload_ordinal_assigned_once_across_redump) {
    gl::Memory m;
    m.setExprKey("(ordinal[redump])");
    for (int16_t i = 0; i < 30; ++i)
        m.intEncodedStatements.push_back(makeExpr(i));
    const std::filesystem::path dir = freshDir("test_deload_ord_redump");
    m.deloadStaticContainers(dir.string());
    const int64_t ord = m.deloadOrdinal;
    m.ensureLoaded(dir.string());
    // A restructuring erase forces a full re-dump; the ordinal must NOT
    // change (assigned once, lifetime-stable identity).
    m.intEncodedStatements.erase(0);
    m.deloadStaticContainers(dir.string());
    ASSERT_EQ(m.deloadOrdinal, ord);
    ASSERT_EQ(m.deloadFiles.size(), static_cast<size_t>(1));  // fresh base
    const std::string lb = "lb" + std::to_string(ord) + "_";
    ASSERT_TRUE(m.deloadFiles.front().find(lb) == 0);
}

// The two internal-mail ColdMail channels (tags 455..464 / 505..514) round-trip
// across deload: all three sameIter columns (statements + origins +
// disintegrationSignals — the column RoutingColdMail dropped) and the two nextIter
// columns, reconstructed in canonical order. Covers the "non-empty sameIter at
// the deload seam" reality (sanitizeHashMemory writes sameIter after the
// post-burst clear, so it must survive deload).
TEST(lb_deload, internal_mail_coldmail_roundtrip) {
    gl::GlobalMemoryManager g;
    g.init(kDeloadTestCfg);
    gl::LbMemory src(&g);

    // Raw id-form inserts (no NameMap in this low-level deload test): pack an
    // (originalId, validityId) pair like packOriginKey for the origins / signals
    // key form. The deload streams the raw ids; the round trip must preserve them.
    auto pk = [](int32_t o, int32_t v) {
        return static_cast<int64_t>(
            (static_cast<uint64_t>(static_cast<uint32_t>(o)) << 32)
            | static_cast<uint64_t>(static_cast<uint32_t>(v)));
    };
    src.sameInternalMail.insertStatement(1, 16, std::set<int>{ 1, 2 });
    src.sameInternalMail.insertStatement(2, 20, std::set<int>{ 3 });
    src.sameInternalMail.origins_.assignRun(pk(1, 16),
        std::vector<gl::IntMailOrigin>{
            gl::IntMailOrigin{ 6, std::vector<int64_t>{ pk(50, 16) } } });
    src.sameInternalMail.setDisintegrationSignal(pk(1, 16), /*dnd=*/true,
                                                 /*aod=*/false);
    src.nextInternalMail.insertStatement(1, 16, std::set<int>{ 5 });
    src.nextInternalMail.origins_.assignRun(pk(1, 16),
        std::vector<gl::IntMailOrigin>{
            gl::IntMailOrigin{ 7, std::vector<int64_t>{ pk(51, 16) } } });

    const std::filesystem::path dir = freshDir("test_deload_internal_mail");
    const std::string chain = "(=[a,b])__(AnchorPeano[N])";
    const std::vector<std::string> files = gl::lbdeload::dumpLbMemory(
        src, chain, 0, dir, kDeloadTestCfg.blockBytes);

    gl::LbMemory dst(&g);
    gl::lbdeload::loadLbMemory(dst, chain, files, dir);

    // sameInternalMail: 2 statements, 1 origin run, 1 signal — all round-trip.
    ASSERT_EQ(dst.sameInternalMail.statements_.count(), 2);
    bool sawA = false, sawB = false;
    for (int32_t id = 1; id <= dst.sameInternalMail.statements_.count(); ++id) {
        const gl::IntMailStatementKey k =
            dst.sameInternalMail.statements_.decodeKey(id);
        if (k.originalId == 1 && k.validityId == 16
            && k.levels == std::vector<int32_t>{ 1, 2 }) sawA = true;
        if (k.originalId == 2 && k.validityId == 20
            && k.levels == std::vector<int32_t>{ 3 }) sawB = true;
    }
    ASSERT_TRUE(sawA && sawB);
    ASSERT_EQ(dst.sameInternalMail.origins_.count(), 1);
    {
        const int32_t oid = dst.sameInternalMail.origins_.lookup(pk(1, 16));
        ASSERT_TRUE(oid != 0);
        const std::vector<gl::IntMailOrigin> recs =
            dst.sameInternalMail.origins_.recordsAt(oid);
        ASSERT_TRUE(recs.size() == 1 && recs[0].tag == 6
            && recs[0].deps == std::vector<int64_t>{ pk(50, 16) });
    }
    {
        const std::pair<bool, bool> sig =
            dst.sameInternalMail.getDisintegrationSignal(pk(1, 16));
        ASSERT_TRUE(sig.first == true && sig.second == false);
    }

    // nextInternalMail: 1 statement, 1 origin run, no signals.
    ASSERT_EQ(dst.nextInternalMail.statements_.count(), 1);
    {
        const gl::IntMailStatementKey k =
            dst.nextInternalMail.statements_.decodeKey(1);
        ASSERT_TRUE(k.originalId == 1 && k.validityId == 16
            && k.levels == std::vector<int32_t>{ 5 });
    }
    {
        const int32_t oid = dst.nextInternalMail.origins_.lookup(pk(1, 16));
        ASSERT_TRUE(oid != 0);
        const std::vector<gl::IntMailOrigin> recs =
            dst.nextInternalMail.origins_.recordsAt(oid);
        ASSERT_TRUE(recs.size() == 1 && recs[0].tag == 7);
    }
    ASSERT_TRUE(dst.nextInternalMail.disintegrationSignalsEmpty());
}
