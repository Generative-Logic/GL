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

#include "test_harness.hpp"
#include "../mail_log.hpp"

#include <cstdint>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

    // MailLog uses `const Memory*` purely as a map-key identity and never
    // dereferences it, so fabricated addresses stand in for LBs — keeping the
    // test free of the arena-backed `Memory` constructor (no static-pool setup).
    const gl::Memory* fakeLb(std::uintptr_t id) {
        return reinterpret_cast<const gl::Memory*>(id);
    }

    gl::Mail oneStatement(const std::string& expr) {
        gl::Mail m;
        m.statements.insert(std::make_pair(
            gl::ExpressionWithValidity(expr, "main"), std::set<int>()));
        return m;
    }

    // A self-contained mail arena + log on a PRIVATE mail pool. The statified
    // MailLog draws its pages from the arena; fabricated `Memory*` addresses
    // stand in for LBs (the pointer is pure key identity, never dereferenced).
    // Default config: 1 MiB pool / 256 KiB block / 8 KiB page, PoolKind::Mail.
    struct MailLogFixture {
        gl::GlobalMemoryManager mem;
        gl::LbArena arena;
        gl::DirtyState dirty = gl::DirtyState::Clean;
        gl::MailLog log;
        explicit MailLogFixture(
            gl::StaticMemoryConfig cfg = gl::StaticMemoryConfig{
                1 << 20, 1 << 18, 1 << 13, gl::PoolKind::Mail })
            : mem(), arena(&mem), dirty(gl::DirtyState::Clean),
              log(&arena, &dirty) {
            mem.init(cfg);
        }
    };

}  // namespace

// A child pulls a single ancestor's committed batch.
TEST(mail_log, single_ancestor_delivery) {
    MailLogFixture f;
    gl::MailLog& log = f.log;
    const gl::Memory* root = fakeLb(0x100);
    const gl::Memory* child = fakeLb(0x200);
    log.registerLb(root, {});
    log.registerLb(child, { root });
    log.commit(root, oneStatement("(in[2,1])"));
    gl::Mail inbox;
    log.pull(child, inbox);
    ASSERT_EQ(inbox.statements.size(), static_cast<std::size_t>(1));
}

// A second pull with no new batches delivers nothing — the cursor is monotone.
TEST(mail_log, cursor_monotonic_no_redelivery) {
    MailLogFixture f;
    gl::MailLog& log = f.log;
    const gl::Memory* root = fakeLb(0x100);
    const gl::Memory* child = fakeLb(0x200);
    log.registerLb(root, {});
    log.registerLb(child, { root });
    log.commit(root, oneStatement("(in[2,1])"));
    gl::Mail first;
    log.pull(child, first);
    ASSERT_EQ(first.statements.size(), static_cast<std::size_t>(1));
    gl::Mail second;
    log.pull(child, second);
    ASSERT_TRUE(second.statements.empty());
}

// Telemetry counts exactly the serialized blobs plus one BlobRef per commit;
// the full-history baseline retains every committed byte simultaneously.
TEST(mail_log, telemetry_counts_committed_and_retained_bytes) {
    MailLogFixture f;
    gl::MailLog& log = f.log;
    const gl::Memory* root = fakeLb(0x100);
    log.registerLb(root, {});
    const gl::Mail batch = oneStatement("(in[2,1])");
    const std::uint64_t oneCommit =
        static_cast<std::uint64_t>(gl::Codec<gl::Mail>::serialize(batch).size())
        + sizeof(gl::BlobRef);

    log.commit(root, batch);
    ASSERT_EQ(log.totalCommittedHistoryBytes, oneCommit);
    ASSERT_EQ(log.peakRetainedHistoryBytes, oneCommit);

    log.commit(root, batch);
    ASSERT_EQ(log.totalCommittedHistoryBytes, oneCommit * 2);
    ASSERT_EQ(log.peakRetainedHistoryBytes, oneCommit * 2);

    log.clear();
    ASSERT_EQ(log.totalCommittedHistoryBytes, oneCommit * 2);
    ASSERT_EQ(log.peakRetainedHistoryBytes, oneCommit * 2);
}

// A parked LB that never pulled catches up the whole ancestor log on first pull.
TEST(mail_log, dormant_catch_up) {
    MailLogFixture f;
    gl::MailLog& log = f.log;
    const gl::Memory* root = fakeLb(0x100);
    const gl::Memory* dormant = fakeLb(0x200);
    log.registerLb(root, {});
    log.registerLb(dormant, { root });
    log.commit(root, oneStatement("(in[2,1])"));
    log.commit(root, oneStatement("(in[3,1])"));
    log.commit(root, oneStatement("(in[4,1])"));
    gl::Mail inbox;
    log.pull(dormant, inbox);
    ASSERT_EQ(inbox.statements.size(), static_cast<std::size_t>(3));
}

// Rolling retirement reuses a bounded set of mail-pool blocks while preserving
// the routing graph, cumulative producer count, and recipient cursor.
TEST(mail_log, repeated_retire_pull_cycles_are_bounded) {
    MailLogFixture f;
    gl::MailLog& log = f.log;
    const gl::Memory* root = fakeLb(0x100);
    const gl::Memory* child = fakeLb(0x200);
    log.registerLb(root, {});
    log.registerLb(child, { root });
    const gl::Mail batch = oneStatement("(in[2,1])");
    const std::uint64_t oneCommit =
        static_cast<std::uint64_t>(gl::Codec<gl::Mail>::serialize(batch).size())
        + sizeof(gl::BlobRef);
    std::int64_t steadyPeakBlocks = 0;

    for (std::int32_t cycle = 0; cycle < 64; ++cycle) {
        log.commit(root, batch);
        ASSERT_TRUE(log.mailPeek(child));
        gl::Mail inbox;
        log.pull(child, inbox);
        ASSERT_EQ(inbox.statements.size(), static_cast<std::size_t>(1));
        ASSERT_FALSE(log.mailPeek(child));

        ASSERT_EQ(log.retireDeliveredBatches(), oneCommit);
        ASSERT_TRUE(log.mailBlobPool.empty());
        ASSERT_TRUE(log.mailRefs.empty());
        const gl::MailHead* head = log.mailHeads.find(gl::MailLog::lbKey(root));
        ASSERT_TRUE(head != nullptr);
        ASSERT_EQ(head->lastRef, -1);
        ASSERT_EQ(head->count, cycle + 1);
        const std::int32_t* cursor =
            log.mailCursor.find(gl::MailLog::edgeKey(child, root));
        ASSERT_TRUE(cursor != nullptr);
        ASSERT_EQ(*cursor, cycle + 1);

        if (cycle == 0) steadyPeakBlocks = f.mem.peakBlocksInUse();
        ASSERT_EQ(f.mem.peakBlocksInUse(), steadyPeakBlocks);
    }

    ASSERT_EQ(log.totalCommittedHistoryBytes, oneCommit * 64);
    ASSERT_EQ(log.peakRetainedHistoryBytes, oneCommit);
}

// A grandchild pulls from every ancestor on its chain.
TEST(mail_log, multi_ancestor_union) {
    MailLogFixture f;
    gl::MailLog& log = f.log;
    const gl::Memory* a = fakeLb(0x100);   // root
    const gl::Memory* b = fakeLb(0x200);   // child of A
    const gl::Memory* c = fakeLb(0x300);   // grandchild, chain C -> B -> A
    log.registerLb(a, {});
    log.registerLb(b, { a });
    log.registerLb(c, { b, a });
    log.commit(a, oneStatement("(fromA)"));
    log.commit(b, oneStatement("(fromB)"));
    gl::Mail inbox;
    log.pull(c, inbox);
    ASSERT_EQ(inbox.statements.size(), static_cast<std::size_t>(2));
}

// mailPeek is the fold-free twin of pull: true iff pull would deliver a batch.
TEST(mail_log, mailpeek_true_before_pull_false_after) {
    MailLogFixture f;
    gl::MailLog& log = f.log;
    const gl::Memory* root = fakeLb(0x100);
    const gl::Memory* child = fakeLb(0x200);
    log.registerLb(root, {});
    log.registerLb(child, { root });
    // No mail yet: nothing pending.
    ASSERT_TRUE(!log.mailPeek(child));
    log.commit(root, oneStatement("(in[2,1])"));
    // A committed ancestor batch the child has not seen: peek is true.
    ASSERT_TRUE(log.mailPeek(child));
    gl::Mail inbox;
    log.pull(child, inbox);
    // Cursor caught up: peek is false again (no re-delivery).
    ASSERT_TRUE(!log.mailPeek(child));
}

// The root has no ancestors, so it never has pending mail to poll.
TEST(mail_log, mailpeek_root_has_no_ancestors) {
    MailLogFixture f;
    gl::MailLog& log = f.log;
    const gl::Memory* root = fakeLb(0x100);
    log.registerLb(root, {});
    log.commit(root, oneStatement("(in[2,1])"));
    ASSERT_TRUE(!log.mailPeek(root));
}

// A grandchild polls true when ANY ancestor on its chain has un-ingested mail,
// and only goes false once it has pulled from every ancestor.
TEST(mail_log, mailpeek_any_ancestor_pending) {
    MailLogFixture f;
    gl::MailLog& log = f.log;
    const gl::Memory* a = fakeLb(0x100);   // root
    const gl::Memory* b = fakeLb(0x200);   // child of A
    const gl::Memory* c = fakeLb(0x300);   // grandchild, chain C -> B -> A
    log.registerLb(a, {});
    log.registerLb(b, { a });
    log.registerLb(c, { b, a });
    ASSERT_TRUE(!log.mailPeek(c));
    log.commit(a, oneStatement("(fromA)"));
    ASSERT_TRUE(log.mailPeek(c));   // A pending
    gl::Mail inbox;
    log.pull(c, inbox);
    ASSERT_TRUE(!log.mailPeek(c));
    log.commit(b, oneStatement("(fromB)"));
    ASSERT_TRUE(log.mailPeek(c));   // now B pending
}

// Origin lines union with dedup: identical lines collapse, distinct ones stay.
TEST(mail_log, origin_dedup_union) {
    MailLogFixture f;
    gl::MailLog& log = f.log;
    const gl::Memory* root = fakeLb(0x100);
    const gl::Memory* child = fakeLb(0x200);
    log.registerLb(root, {});
    log.registerLb(child, { root });
    gl::ExpressionWithValidity key("(in[2,1])", "main");
    std::vector<gl::ExpressionWithValidity> dep{
        gl::ExpressionWithValidity("(NaturalNumbers[1,2,3,4,5])", "main") };

    gl::Mail b1;
    b1.exprOriginMap[key].push_back(std::make_pair(std::string("disintegration"), dep));
    log.commit(root, b1);

    gl::Mail b2;
    b2.exprOriginMap[key].push_back(std::make_pair(std::string("disintegration"), dep));
    b2.exprOriginMap[key].push_back(std::make_pair(std::string("expansion"), dep));
    log.commit(root, b2);

    gl::Mail inbox;
    log.pull(child, inbox);
    ASSERT_EQ(inbox.exprOriginMap[key].size(), static_cast<std::size_t>(2));
}

// The PRODUCTION pull overload pull(const Memory*, RoutingColdMail&) delivers
// the SAME content (statements + exprOriginMap, decoded via the global
// mailInterner) as the retained heap oracle pull(const Memory*, Mail&) over the
// same log. This is the first end-to-end unit test of the pool-cursor
// production path (performElemPhase1 -> pull(RoutingColdMail&) -> readBlobInto
// -> deserializeInto(pool)). The origins-only batch is committed LAST so it is
// the newest and therefore pulled FIRST into the fresh RoutingColdMail — its
// deserializeMailBlobInto is then the mailbox's first write, re-covering the
// G-55 ensureArena-before-both-loops path (stCount==0). Origins sit under
// distinct keys (one line each) so the map comparison is order-robust.
TEST(mail_log, pull_routing_cold_mail_matches_heap) {
    MailLogFixture f;
    gl::MailLog& log = f.log;
    const gl::Memory* root = fakeLb(0x100);
    const gl::Memory* child = fakeLb(0x200);
    const gl::Memory* child2 = fakeLb(0x300);
    log.registerLb(root, {});
    log.registerLb(child, { root });
    log.registerLb(child2, { root });

    // Batch 1: statements + an origin line (under key A).
    gl::ExpressionWithValidity keyA("(=[3,4])", "main");
    gl::Mail b1;
    b1.statements.insert(std::make_pair(
        gl::ExpressionWithValidity("(=[3,4])", "main"), std::set<int>{ 1, 5 }));
    b1.statements.insert(std::make_pair(
        gl::ExpressionWithValidity("(in[5,1])", "main"), std::set<int>()));
    b1.exprOriginMap[keyA].push_back(std::make_pair(
        std::string("expansion"),
        std::vector<gl::ExpressionWithValidity>{
            gl::ExpressionWithValidity("(a)", "main"),
            gl::ExpressionWithValidity("(b)", "v2") }));
    log.commit(root, b1);

    // Batch 2: ORIGINS-ONLY (no statements), under a distinct key B. Committed
    // last => newest => pulled first => exercises the G-55 origins-only-first
    // ensureArena path in a fresh RoutingColdMail.
    gl::ExpressionWithValidity keyB("(in[2,1])", "main");
    gl::Mail b2;
    b2.exprOriginMap[keyB].push_back(std::make_pair(
        std::string("disintegration"),
        std::vector<gl::ExpressionWithValidity>{
            gl::ExpressionWithValidity("(NaturalNumbers[1,2,3])", "main") }));
    log.commit(root, b2);

    // Production overload: pull into a RoutingColdMail (global ids), then decode.
    gl::RoutingColdMail routingInbox;
    log.pull(child, routingInbox);
    const gl::Mail routingHeap = gl::routingMailInToHeap(routingInbox);

    // Oracle overload: pull the same two batches into a heap Mail (distinct
    // recipient, so its cursor is fresh).
    gl::Mail heapInbox;
    log.pull(child2, heapInbox);

    ASSERT_TRUE(routingHeap.statements == heapInbox.statements);
    ASSERT_TRUE(routingHeap.exprOriginMap == heapInbox.exprOriginMap);
    // Sanity: both keys and both statements actually arrived.
    ASSERT_EQ(routingHeap.statements.size(), static_cast<std::size_t>(2));
    ASSERT_EQ(routingHeap.exprOriginMap.size(), static_cast<std::size_t>(2));
}

// An LB with no registered ancestors pulls nothing — "no ancestors" is a
// defined no-op, not an error. This covers the root (registered, empty ancestor
// list) and, uniformly, any not-yet-registered LB. (The heap prototype threw
// std::out_of_range for an unregistered recipient; the statified form treats
// no-edges uniformly as no delivery, and the no-mid-run-LB-birth architecture
// guarantees every real pull is on a registered LB. The surviving Rule-19 trap
// is the cursor-cell-consistency assert inside the pull loop.)
TEST(mail_log, no_ancestors_pulls_nothing) {
    MailLogFixture f;
    gl::MailLog& log = f.log;
    const gl::Memory* root = fakeLb(0x100);
    log.registerLb(root, {});               // registered, but no ancestors
    log.commit(root, oneStatement("(in[2,1])"));
    gl::Mail inbox;
    log.pull(root, inbox);                   // the root pulls from nothing
    ASSERT_TRUE(inbox.statements.empty());
    gl::Mail inbox2;
    log.pull(fakeLb(0x999), inbox2);         // never registered — also a no-op
    ASSERT_TRUE(inbox2.statements.empty());
}

// clear() drops every batch and cursor at batch teardown.
TEST(mail_log, clear_resets) {
    MailLogFixture f;
    gl::MailLog& log = f.log;
    const gl::Memory* root = fakeLb(0x100);
    const gl::Memory* child = fakeLb(0x200);
    log.registerLb(root, {});
    log.registerLb(child, { root });
    log.commit(root, oneStatement("(in[2,1])"));
    ASSERT_FALSE(log.empty());   // a batch, an edge, and a cursor cell exist
    log.clear();
    ASSERT_TRUE(log.empty());
}

// Codec<Mail> round-trips statements + exprOriginMap losslessly and drops the
// field the log never carries (disintegrationSignals).
TEST(mail_log, codec_mail_round_trip) {
    gl::Mail m;
    // Two statements, one with a non-empty level set.
    m.statements.insert(std::make_pair(
        gl::ExpressionWithValidity("(in[2,1])", "main"), std::set<int>()));
    m.statements.insert(std::make_pair(
        gl::ExpressionWithValidity("(=[3,4])", "main"), std::set<int>{ 1, 5, 9 }));
    // One origin key with two origin lines, the second carrying two deps at
    // distinct validities.
    gl::ExpressionWithValidity key("(in[2,1])", "main");
    m.exprOriginMap[key].push_back(std::make_pair(
        std::string("disintegration"),
        std::vector<gl::ExpressionWithValidity>{
            gl::ExpressionWithValidity("(NaturalNumbers[1,2,3])", "main") }));
    m.exprOriginMap[key].push_back(std::make_pair(
        std::string("expansion"),
        std::vector<gl::ExpressionWithValidity>{
            gl::ExpressionWithValidity("(a)", "main"),
            gl::ExpressionWithValidity("(b)", "v2") }));
    const std::vector<char> blob = gl::Codec<gl::Mail>::serialize(m);
    const gl::Mail r = gl::Codec<gl::Mail>::deserialize(
        blob.data(), static_cast<std::int32_t>(blob.size()));

    ASSERT_TRUE(r.statements == m.statements);
    ASSERT_TRUE(r.exprOriginMap == m.exprOriginMap);
    ASSERT_TRUE(r.disintegrationSignals.empty());  // dropped by the log
    // serialize is the exact inverse: re-serializing the decode reproduces the blob.
    ASSERT_TRUE(gl::Codec<gl::Mail>::serialize(r) == blob);
}

// An empty batch round-trips to an empty batch (the header-only case).
TEST(mail_log, codec_mail_empty_round_trip) {
    gl::Mail m;
    const std::vector<char> blob = gl::Codec<gl::Mail>::serialize(m);
    const gl::Mail r = gl::Codec<gl::Mail>::deserialize(
        blob.data(), static_cast<std::int32_t>(blob.size()));
    ASSERT_TRUE(r.statements.empty());
    ASSERT_TRUE(r.exprOriginMap.empty());
}

// Codec<CursorKey> is the identity over the 16-byte POD key; the two endpoints
// are not interchangeable (recipient != ancestor when swapped).
TEST(mail_log, codec_cursorkey_identity_round_trip) {
    const gl::CursorKey k{ 0xDEADBEEFull, 0x1234ABCDull };
    const gl::CursorKey e = gl::Codec<gl::CursorKey>::encode(k);
    ASSERT_TRUE(e == k);
    const gl::CursorKey d = gl::Codec<gl::CursorKey>::decode(e);
    ASSERT_TRUE(d == k);
    const gl::CursorKey swapped{ 0x1234ABCDull, 0xDEADBEEFull };
    ASSERT_FALSE(swapped == k);
}

// The identity-key helpers cast the Memory* bit pattern faithfully and the edge
// key is directional.
TEST(mail_log, key_helpers_are_pointer_identity) {
    const gl::Memory* a = fakeLb(0x1234);
    const gl::Memory* b = fakeLb(0x5678);
    ASSERT_EQ(gl::MailLog::lbKey(a), static_cast<std::int64_t>(0x1234));
    ASSERT_EQ(gl::MailLog::lbKey(b), static_cast<std::int64_t>(0x5678));
    const gl::CursorKey k = gl::MailLog::edgeKey(a, b);
    ASSERT_EQ(k.recipient, static_cast<std::uint64_t>(0x1234));
    ASSERT_EQ(k.ancestor, static_cast<std::uint64_t>(0x5678));
    ASSERT_FALSE(gl::MailLog::edgeKey(a, b) == gl::MailLog::edgeKey(b, a));
}

// Regression guard for the bug that broke every earlier mail-statification
// attempt: PagedHashIndex::reset silently wrote past its single directory page
// once the index exceeded dirCap, corrupting neighbouring arena pages. At a
// 256-byte page (dirCap = 64 data-page vids per directory page) a few thousand
// distinct LB keys / edges / cursor cells push the three cold containers' indices
// into the two-level directory — the now-fixed path — and every pull must still
// deliver exactly its ancestors' batches.
TEST(mail_log, forced_two_level_spill_no_corruption) {
    // 16 MiB pool / 4 KiB block / 256 B page.
    MailLogFixture f(gl::StaticMemoryConfig{ 1 << 24, 1 << 12, 256,
                                             gl::PoolKind::Mail });
    gl::MailLog& log = f.log;
    const int N = 2500;   // well past dirCap=64 for all three indices
    auto child = [](int i) {
        return fakeLb(0x100000 + static_cast<std::uintptr_t>(i) * 64);
    };
    const gl::Memory* root = fakeLb(0x10);
    log.registerLb(root, {});
    for (int i = 0; i < N; ++i) log.registerLb(child(i), { root });

    // Root broadcasts one seed; every child commits its own distinct batch.
    log.commit(root, oneStatement("(seed)"));
    for (int i = 0; i < N; ++i) log.commit(child(i), oneStatement("(c)"));

    // Every child pulls exactly the root's seed; a second pull delivers nothing
    // (cursor monotone at scale).
    for (int i = 0; i < N; ++i) {
        gl::Mail inbox;
        log.pull(child(i), inbox);
        ASSERT_EQ(inbox.statements.size(), static_cast<std::size_t>(1));
        gl::Mail again;
        log.pull(child(i), again);
        ASSERT_TRUE(again.statements.empty());
    }

    // A grandchild pulls its parent's batch + the root's seed (2 ancestors),
    // proving the spilled mailBatches index resolves distinct interior keys.
    const gl::Memory* gc = fakeLb(0x999);
    log.registerLb(gc, { child(1234), root });
    gl::Mail gci;
    log.pull(gc, gci);
    ASSERT_EQ(gci.statements.size(), static_cast<std::size_t>(2));  // (c) + (seed)
}

// ===================================================================
//  Mail byte codecs — pure round-trip coverage (no arena). OriginLine is a
//  blob record; serialize/deserialize must be exact inverses.
// ===================================================================

TEST(hotmail_codec, originline_roundtrip) {
    gl::OriginLine line;
    line.first = "implication";
    line.second.emplace_back("(>[v1](in[v1,N])(in[(s[v1]),N]))", "main");
    line.second.emplace_back("(in[a,N])", "main");
    const std::vector<char> blob = gl::Codec<gl::OriginLine>::serialize(line);
    const gl::OriginLine back = gl::Codec<gl::OriginLine>::deserialize(
        blob.data(), static_cast<int32_t>(blob.size()));
    ASSERT_TRUE(back == line);

    // No-antecedent line (e.g. "theorem") round-trips.
    gl::OriginLine bare;
    bare.first = "theorem";
    const std::vector<char> bareBlob =
        gl::Codec<gl::OriginLine>::serialize(bare);
    ASSERT_TRUE(gl::Codec<gl::OriginLine>::deserialize(
        bareBlob.data(), static_cast<int32_t>(bareBlob.size())) == bare);
}

// ===================================================================
//  DeloadableMailOut — outgoing routing mail plus its private interner on the
//  owning Memory's deloadable LB arena. RoutingColdMail remains the mailIn type.
// ===================================================================

namespace {
    gl::ExpressionWithValidity ev(const std::string& expr) {
        return gl::ExpressionWithValidity(expr, "main");
    }
}

TEST(deloadable_mail_out, insert_and_toheapmail) {
    gl::Memory mem;
    ASSERT_TRUE(mem.mailOut.empty());
    ASSERT_FALSE(mem.mailOutPending);
    mem.insertMailOutStatement(ev("(in[a,N])"), std::set<int>{});
    mem.insertMailOutStatement(ev("(in[b,N])"), std::set<int>{ 2, 5 });
    mem.addMailOutOrigin(ev("(in[a,N])"),
        std::make_pair("theorem", std::vector<gl::ExpressionWithValidity>()), 4);
    ASSERT_FALSE(mem.mailOut.empty());
    ASSERT_TRUE(mem.mailOutPending);

    const gl::Mail m = gl::routingMailOutToHeap(
        mem.mailOut, mem.mailOutInterner);
    ASSERT_EQ(m.statements.size(), static_cast<std::size_t>(2));
    ASSERT_EQ(m.exprOriginMap.size(), static_cast<std::size_t>(1));
    ASSERT_EQ(m.exprOriginMap.at(ev("(in[a,N])")).size(),
              static_cast<std::size_t>(1));
    ASSERT_TRUE(m.exprOriginMap.at(ev("(in[a,N])"))[0].first == "theorem");
    ASSERT_TRUE(mem.mailOutLiveBytes > 0);
    mem.clearMailOut();
    ASSERT_TRUE(mem.mailOut.empty());
    ASSERT_FALSE(mem.mailOutPending);
    ASSERT_EQ(mem.mailOutLiveBytes, 0);
    ASSERT_EQ(mem.mailOutInterner.internedCount(), 0);
}

TEST(deloadable_mail_out, statement_multiplicity) {
    // Two members sharing an EWV but differing in levels stay distinct (the
    // whole-pair key reproduces set<pair<EWV,set<int>>> exactly).
    gl::Memory mem;
    mem.insertMailOutStatement(ev("(in[a,N])"), std::set<int>{ 1 });
    mem.insertMailOutStatement(ev("(in[a,N])"), std::set<int>{ 2 });
    mem.insertMailOutStatement(ev("(in[a,N])"), std::set<int>{ 1 });
    const gl::Mail m = gl::routingMailOutToHeap(
        mem.mailOut, mem.mailOutInterner);
    ASSERT_EQ(m.statements.size(), static_cast<std::size_t>(2));  // {1} and {2}
}

TEST(deloadable_mail_out, origin_cap_foundation_displaces_convenience) {
    // D-49 at cap=1: a foundational origin overwrites an equality-convenience
    // slot; mirrors ExpressionAnalyzer::addOrigin's string policy.
    gl::Memory mem;
    mem.addMailOutOrigin(ev("(=[a,b])"),
        std::make_pair("equality1", std::vector<gl::ExpressionWithValidity>()), 1);
    mem.addMailOutOrigin(ev("(=[a,b])"),
        std::make_pair("implication",
            std::vector<gl::ExpressionWithValidity>{ ev("(>[v1]...)") }), 1);
    const gl::Mail m = gl::routingMailOutToHeap(
        mem.mailOut, mem.mailOutInterner);
    ASSERT_EQ(m.exprOriginMap.at(ev("(=[a,b])")).size(),
              static_cast<std::size_t>(1));
    ASSERT_TRUE(m.exprOriginMap.at(ev("(=[a,b])"))[0].first == "implication");
}

// Run-door byte-twin: depositing the same statements through the
// std::set<int> overload (mailbox A) and the (const int*, int32_t) run
// overload (mailbox B) yields the same statement count, string-identical heap
// snapshots, and BYTE-IDENTICAL Codec<Mail>::serialize blobs. Cases: multi
// {1,3,7}, singleton {0}, EMPTY (nullptr, 0), and two statements differing
// only in levels (whole-pair key multiplicity preserved).
TEST(deloadable_mail_out, statement_levels_run_door_twin) {
    gl::Memory viaSet;
    gl::Memory viaRun;

    const int multi[] = { 1, 3, 7 };
    const int one[] = { 0 };
    const int two[] = { 2 };

    viaSet.insertMailOutStatement(ev("(in[a,N])"), std::set<int>{ 1, 3, 7 });
    viaSet.insertMailOutStatement(ev("(in[b,N])"), std::set<int>{ 0 });
    viaSet.insertMailOutStatement(ev("(in[c,N])"), std::set<int>{});
    viaSet.insertMailOutStatement(ev("(in[b,N])"), std::set<int>{ 2 });

    viaRun.insertMailOutStatement(gl::StrSpan("(in[a,N])"), gl::StrSpan("main"), multi, 3);
    viaRun.insertMailOutStatement(gl::StrSpan("(in[b,N])"), gl::StrSpan("main"), one, 1);
    viaRun.insertMailOutStatement(gl::StrSpan("(in[c,N])"), gl::StrSpan("main"), nullptr, 0);
    viaRun.insertMailOutStatement(gl::StrSpan("(in[b,N])"), gl::StrSpan("main"), two, 1);

    const gl::Mail hs =
        gl::routingMailOutToHeap(viaSet.mailOut, viaSet.mailOutInterner);
    const gl::Mail hr =
        gl::routingMailOutToHeap(viaRun.mailOut, viaRun.mailOutInterner);
    ASSERT_EQ(hs.statements.size(), static_cast<std::size_t>(4));
    ASSERT_TRUE(hs.statements == hr.statements);

    const std::vector<char> bs =
        gl::Codec<gl::Mail>::serialize(viaSet.mailOut, viaSet.mailOutInterner);
    const std::vector<char> br =
        gl::Codec<gl::Mail>::serialize(viaRun.mailOut, viaRun.mailOutInterner);
    ASSERT_TRUE(bs == br);   // byte-for-byte
}

TEST(deloadable_mail_out, clear_then_reuse) {
    gl::Memory mem;
    mem.insertMailOutStatement(ev("(in[a,N])"), std::set<int>{ 3 });
    ASSERT_FALSE(mem.mailOut.statementsEmpty());
    mem.clearMailOut();
    ASSERT_TRUE(mem.mailOut.empty());
    mem.insertMailOutStatement(ev("(in[c,N])"), std::set<int>{});
    ASSERT_EQ(gl::routingMailOutToHeap(
        mem.mailOut, mem.mailOutInterner).statements.size(),
              static_cast<std::size_t>(1));
}

// Codec<Mail>::serialize(const DeloadableMailOut&, interner) emits
// BYTE-IDENTICAL bytes to serialize(routingMailOutToHeap(...)).
// the same blob (GLOBAL mailInterner ids for statements) as the heap-Mail path it
// replaces. This pins the determinism contract at the unit level.
TEST(routing_cold_mail_codec, serialize_matches_toheapmail_bytes) {
    gl::Memory mem;
    mem.insertMailOutStatement(ev("(in[b,N])"), std::set<int>{ 2, 5 });
    mem.insertMailOutStatement(ev("(in[a,N])"), std::set<int>{});
    mem.insertMailOutStatement(ev("(in[a,N])"), std::set<int>{ 1 });
    mem.addMailOutOrigin(ev("(in[a,N])"),
        std::make_pair("theorem", std::vector<gl::ExpressionWithValidity>()), 8);
    mem.addMailOutOrigin(ev("(in[a,N])"),
        std::make_pair("disintegration",
            std::vector<gl::ExpressionWithValidity>{ ev("(&[x,y])") }), 8);
    mem.addMailOutOrigin(ev("(=[a,b])"),
        std::make_pair("equality1", std::vector<gl::ExpressionWithValidity>()), 8);

    const std::vector<char> direct =
        gl::Codec<gl::Mail>::serialize(mem.mailOut, mem.mailOutInterner);
    const std::vector<char> viaHeap =
        gl::Codec<gl::Mail>::serialize(gl::routingMailOutToHeap(
            mem.mailOut, mem.mailOutInterner));
    ASSERT_TRUE(direct == viaHeap);   // byte-for-byte

    // ...and the blob decodes back to a Mail that re-serializes identically.
    const gl::Mail r = gl::Codec<gl::Mail>::deserialize(
        direct.data(), static_cast<int32_t>(direct.size()));
    ASSERT_TRUE(gl::Codec<gl::Mail>::serialize(r) == direct);
}

// Codec<Mail>::deserializeInto folds a blob STRAIGHT into a routing mailIn (GLOBAL
// ids) via its write doors. mailOut (private per-LB ids) and mailIn (global ids) use
// different id-spaces, so the CONTENT round-trips at the string level (not the
// raw bytes): routingMailOutToHeap(src) == routingMailInToHeap(inbox).
TEST(routing_cold_mail_codec, deserialize_into_round_trips) {
    gl::Memory mem;
    mem.insertMailOutStatement(ev("(in[b,N])"), std::set<int>{ 2, 5 });
    mem.insertMailOutStatement(ev("(in[a,N])"), std::set<int>{});
    mem.addMailOutOrigin(ev("(in[a,N])"),
        std::make_pair("theorem", std::vector<gl::ExpressionWithValidity>()), 8);
    mem.addMailOutOrigin(ev("(in[b,N])"),
        std::make_pair("disintegration",
            std::vector<gl::ExpressionWithValidity>{ ev("(&[x,y])") }), 8);
    const std::vector<char> blob =
        gl::Codec<gl::Mail>::serialize(mem.mailOut, mem.mailOutInterner);

    // Direct decode into a fresh mailIn (global ids).
    gl::RoutingColdMail inbox;
    gl::Codec<gl::Mail>::deserializeInto(
        blob.data(), static_cast<int32_t>(blob.size()), inbox);
    const gl::Mail srcHeap =
        gl::routingMailOutToHeap(mem.mailOut, mem.mailOutInterner);
    const gl::Mail inboxHeap = gl::routingMailInToHeap(inbox);
    ASSERT_TRUE(srcHeap.statements == inboxHeap.statements);
    ASSERT_TRUE(srcHeap.exprOriginMap == inboxHeap.exprOriginMap);

    // ...and equals the heap reference path deserialize -> mergeBatchIntoMailIn.
    gl::RoutingColdMail refInbox;
    const gl::Mail batch = gl::Codec<gl::Mail>::deserialize(
        blob.data(), static_cast<int32_t>(blob.size()));
    gl::mergeBatchIntoMailIn(batch, refInbox);
    const gl::Mail refHeap = gl::routingMailInToHeap(refInbox);
    ASSERT_TRUE(refHeap.statements == inboxHeap.statements);
    ASSERT_TRUE(refHeap.exprOriginMap == inboxHeap.exprOriginMap);
}

// Codec<Mail>::deserializeInto(pool, start, len, inbox) — the production
// pool-native overload — decodes byte-identically to the retained char*
// overload. A batch is serialized to a blob, that blob is written into a
// PagedVector<char> on a private mail pool, and BOTH overloads decode it into
// fresh mailIns; the resulting heap Mails (statements + exprOriginMap) must be
// equal. The content deliberately mixes an empty-levels statement, an
// empty-antecedent origin line, and a multi-dep origin line. It runs twice:
// once with the default page size (contiguous, no straddle), once with a tiny
// page AND a 2-byte lead pad — every Codec<Mail> field is a 4-byte int32, so a
// 4-divisible page with a 4-aligned start never splits a field; the 2-byte pad
// shifts every field off the page grid, forcing PoolMailSource's mid-field
// straddle path (the reassembly the old std::vector<char> buffer used to hide).
TEST(routing_cold_mail_codec, deserialize_into_pool_matches_char) {
    gl::Memory mem;
    mem.insertMailOutStatement(ev("(in[b,N])"), std::set<int>{ 2, 5 });
    mem.insertMailOutStatement(ev("(in[a,N])"), std::set<int>{});
    mem.addMailOutOrigin(ev("(in[a,N])"),
        std::make_pair("theorem", std::vector<gl::ExpressionWithValidity>()), 8);  // empty antecedent
    mem.addMailOutOrigin(ev("(in[b,N])"),
        std::make_pair("disintegration",
            std::vector<gl::ExpressionWithValidity>{
                ev("(&[x,y])"), ev("(=[p,q])"), ev("(in[c,N])") }), 8);  // multi-dep
    const std::vector<char> blob =
        gl::Codec<gl::Mail>::serialize(mem.mailOut, mem.mailOutInterner);

    // The char* overload — the oracle both pool decodes are compared against.
    gl::RoutingColdMail charInbox;
    gl::Codec<gl::Mail>::deserializeInto(
        blob.data(), static_cast<int32_t>(blob.size()), charInbox);
    const gl::Mail charHeap = gl::routingMailInToHeap(charInbox);

    // Decode from a PagedVector<char> at byte offset `pad` (0 or 2) on a private
    // mail pool with `pageBytes`, and assert equality with the char* oracle.
    const auto poolDecodeEquals = [&](int32_t pageBytes, int32_t pad) {
        gl::GlobalMemoryManager pm;
        pm.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18, pageBytes,
                                        gl::PoolKind::Mail });
        gl::DirtyState d = gl::DirtyState::Clean;
        gl::LbArena arena(&pm);
        gl::PagedVector<char> pool(&arena, &d);
        for (int32_t i = 0; i < pad; ++i) pool.push_back('\0');   // lead pad
        for (const char c : blob) pool.push_back(c);
        gl::RoutingColdMail poolInbox;
        gl::Codec<gl::Mail>::deserializeInto(
            pool, static_cast<std::uint32_t>(pad),
            static_cast<std::uint32_t>(blob.size()), poolInbox);
        const gl::Mail poolHeap = gl::routingMailInToHeap(poolInbox);
        ASSERT_TRUE(poolHeap.statements == charHeap.statements);
        ASSERT_TRUE(poolHeap.exprOriginMap == charHeap.exprOriginMap);
    };

    poolDecodeEquals(1 << 13, 0);   // default 8 KiB page, aligned — contiguous
    poolDecodeEquals(32, 2);        // 32 B page + 2 B pad — forces the straddle
}
