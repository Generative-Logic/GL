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

#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "memory.hpp"

namespace gl {

    /// @brief Key of the statified mail cursor — the (recipient, ancestor) edge
    ///        whose count of already-ingested batches the cursor tracks.
    ///
    /// @details
    /// Both fields are the `uintptr_t` of a `const Memory*` — pure RUNTIME
    /// identity (the pointer is never dereferenced, Rule 12; `exprKey` is not
    /// unique, so the pointer is the only stable LB identity). The mail pool is
    /// never deloaded, so there is no canonical-bytes obligation and
    /// pointer-valued keys are sound — proof output stays content-deterministic
    /// via the downstream set-merge + absorb sort, independent of pointer values.
    /// A 16-byte POD with no padding, so `PodKeyStore<CursorKey>` accepts it
    /// (trivially copyable + unique object representation — its two `static_assert`s).
    ///
    /// @see Codec<CursorKey>, MailLog.
    struct CursorKey {
        /// @brief The pulling LB's identity (uintptr of its `const Memory*`).
        std::uint64_t recipient;
        /// @brief The ancestor LB's identity (uintptr of its `const Memory*`).
        std::uint64_t ancestor;

        /// @brief Value equality (tests + decoded-key comparisons).
        ///
        /// @param o The other key.
        /// @return `true` when both edge endpoints match.
        bool operator==(const CursorKey& o) const {
            return recipient == o.recipient && ancestor == o.ancestor;
        }
    };

    /// @brief Codec for `CursorKey` — the identity over `PodKeyStore<CursorKey>`.
    ///
    /// @details
    /// The 16-byte POD struct IS the engine's stored key (no packing to a
    /// scalar), so encode / decode / view are all the identity. The mail pool is
    /// never deloaded; the raw pointer-byte key needs no canonical form.
    ///
    /// @see CursorKey, IdentityKeyCodec.
    template <>
    struct Codec<CursorKey> : IdentityKeyCodec<CursorKey> {};

    /// @brief Merge one committed mail batch's deliverable content into a
    ///        receiver inbox, matching the retired `smashMail`'s field coverage.
    ///
    /// @details
    /// A stored batch in the [MailLog](#MailLog) carries only `statements` and
    /// `exprOriginMap` (the two fields the old push routing ever folded into a
    /// receiver's `mailIn`). This helper performs exactly the merge `smashMail`
    /// did for one box slot:
    /// - `statements` are set-inserted (commutative; duplicates collapse).
    /// - `exprOriginMap` is a dedup-append union: each origin line is appended to
    ///   the receiver's per-key vector only when not already present (linear
    ///   scan, mirroring `smashMail`'s `std::find` guard).
    ///
    /// `expandedImplications` and `disintegrationSignals` are intentionally NOT
    /// merged. `smashMail` dropped `expandedImplications` (it never reached any
    /// receiver's `mailIn` under the old routing), and `disintegrationSignals`
    /// is only ever written on the per-LB internal channel, never on `mailOut`.
    /// Propagating either here would deliver content cross-LB for the first time
    /// — a semantics change forbidden without approval (Rule 8).
    ///
    /// The per-key origin order produced here is not pinned, and deliberately so:
    /// the receiver's `standardProcessing` absorb sorts every origin vector before
    /// the capped `addOrigin` fold, so the surviving origin under the cap is a
    /// function of the sort, not of the merge order. Two pull orders therefore
    /// yield byte-identical persistent `exprOriginMap` state.
    ///
    /// @param batch  One stored batch (statements + exprOriginMap only).
    /// @param inbox  The receiver's `mailIn`, merged into in place.
    /// @see MailLog::pull — the sole caller in the prover path.
    inline void mergeBatchInto(const Mail& batch, Mail& inbox) {
        inbox.statements.insert(batch.statements.begin(), batch.statements.end());
        for (const std::pair<const ExpressionWithValidity,
                             std::vector<std::pair<std::string,
                                                   std::vector<ExpressionWithValidity>>>>& keyed
             : batch.exprOriginMap) {
            std::vector<std::pair<std::string, std::vector<ExpressionWithValidity>>>& dst =
                inbox.exprOriginMap[keyed.first];
            for (const std::pair<std::string, std::vector<ExpressionWithValidity>>& origin
                 : keyed.second) {
                if (std::find(dst.begin(), dst.end(), origin) == dst.end()) {
                    dst.push_back(origin);
                }
            }
        }
    }

    /// @brief Fold a decoded heap batch into a routing `mailOut` (SENDER ids) —
    ///        statements interned into the producing LB's `NameMap`.
    ///
    /// @details
    /// The heap-`Mail` deposit path into a producer's own outbox (the post-join
    /// `updateGlobal*` / compaction sends — single-threaded, so the `NameMap`
    /// mint is safe). Statements bulk-mint by the whole `(EWV, levels)` pair (set
    /// dedup); origins dedup-APPEND per key with NO cap (`addMailOrigin` at
    /// `INT_MAX` stays in its below-cap dedup-append path). The receiver's absorb
    /// re-sorts origin runs before the capped fold, so merge order is irrelevant.
    ///
    /// @param batch One stored batch (statements + exprOriginMap only).
    /// @param inbox The producer's `mailOut` (a `RoutingColdMail`), merged in place.
    /// @param nm    The producing LB's `NameMap` (statements id-space).
    /// @param oi    The producing LB's `originInterner` (origins id-space).
    inline void mergeBatchIntoMailOut(const Mail& batch, RoutingColdMail& inbox,
                                      NameMap& nm, ValueInterner& oi) {
        for (const std::pair<ExpressionWithValidity, std::set<int>>& st
             : batch.statements) {
            inbox.insertStatement(nm.encode(st.first.original),
                                  nm.encode(st.first.validityName), st.second);
        }
        for (const std::pair<const ExpressionWithValidity,
                 std::vector<OriginLine>>& keyed : batch.exprOriginMap) {
            for (const OriginLine& origin : keyed.second) {
                addRoutingMailOrigin(inbox, oi, keyed.first, origin,
                                     (std::numeric_limits<int>::max)());
            }
        }
    }

    /// @brief Fold a decoded heap batch into a routing `mailIn` (GLOBAL ids) —
    ///        statements + origins interned into the global `mailInterner`.
    ///
    /// @details
    /// The load-time broadcast self-inject into the root's inbox (single-threaded,
    /// so the `mailInterner` mint is safe). Both columns are global-id; the origin
    /// EWVs are packed through the global interner (so `addRoutingMailOrigin`'s
    /// originInterner path does not apply — the record is built directly here).
    ///
    /// @param batch One stored batch (statements + exprOriginMap only).
    /// @param inbox The receiver's `mailIn` (a `RoutingColdMail`), merged in place.
    /// @see mergeBatchIntoMailOut, addMailOriginRecord, MailLog::pull.
    inline void mergeBatchIntoMailIn(const Mail& batch, RoutingColdMail& inbox) {
        inbox.ensureArena();
        const auto globalKey = [](const ExpressionWithValidity& e) {
            return packOriginKey(mailInterner().intern(e.original),
                                 mailInterner().intern(e.validityName));
        };
        for (const std::pair<ExpressionWithValidity, std::set<int>>& st
             : batch.statements) {
            inbox.insertStatement(mailInterner().intern(st.first.original),
                                  mailInterner().intern(st.first.validityName),
                                  st.second);
        }
        for (const std::pair<const ExpressionWithValidity,
                 std::vector<OriginLine>>& keyed : batch.exprOriginMap) {
            const int64_t key = globalKey(keyed.first);
            for (const OriginLine& origin : keyed.second) {
                IntMailOrigin record;
                record.tag =
                    static_cast<uint8_t>(originTagFromString(origin.first));
                record.deps.reserve(origin.second.size());
                for (const ExpressionWithValidity& dep : origin.second)
                    record.deps.push_back(globalKey(dep));
                addMailOriginRecord(inbox.origins_, key, record,
                                    (std::numeric_limits<int>::max)());
            }
        }
    }

    /// @brief One committed batch's location in the shared mail blob pool, plus a
    ///        back-link forming its producing LB's append-only chain.
    ///
    /// @details
    /// A trivially-copyable POD so a `PagedVector<BlobRef>` can hold all LBs' refs
    /// in one shared, append-only column. The `prev` back-link threads each LB's
    /// batches into an independent newest-first chain — the pool equivalent of the
    /// heap prototype's per-LB `vector<Mail>`, so a commit never shifts another
    /// LB's bytes.
    struct BlobRef {
        /// @brief Byte offset of the blob in `mailBlobPool`.
        std::uint32_t start;
        /// @brief Blob length in bytes.
        std::uint32_t len;
        /// @brief Index in `mailRefs` of this LB's previous batch, or -1 for its first.
        std::int32_t prev;
    };

    /// @brief The head of one LB's batch chain — its newest ref and its count.
    ///
    /// @details
    /// The value type of `mailHeads`; a commit reads then overwrites it
    /// single-threaded. Trivially copyable (a single-value cold map column).
    struct MailHead {
        /// @brief Index in `mailRefs` of the LB's newest batch, or -1 if none.
        std::int32_t lastRef;
        /// @brief Total batches the LB has committed this execution batch.
        std::int32_t count;
    };

    /// @brief Per-execution-batch store of each LB's outgoing mail, pulled by
    ///        its descendants — STATIFIED onto the dedicated mail pool.
    ///
    /// @details
    /// `MailLog` replaces the push routing `mailOut → sendMail → boxes →
    /// smashMail → every descendant's mailIn`. Each LB's per-cycle mail is stored
    /// ONCE; a receiver pulls from its ancestors' logs and ingests only the
    /// batches it has not seen, tracked by a per-(recipient, ancestor) cursor — a
    /// dormant LB stays at cursor 0 and catches up its ancestors' whole logs the
    /// moment it wakes. Delivery is the exact dual of the old routing: "pull from
    /// every ancestor on the `parentMemory` chain" reaches the identical
    /// sender→receiver set ([I-57]).
    ///
    /// All storage lives on the dedicated, NEVER-DELOADED mail pool — the
    /// `LbArena` + `DirtyState` handed to the constructor (the
    /// `ExpressionAnalyzer`-owned `mailArena{ &mailMemory() }`, the "local memory
    /// manager"). The five cold containers are pure RUNTIME containers: never
    /// enumerated by `LbMemory::visitContainers`, so never deloaded /
    /// dirty-tracked / reshuffled (the `intToBeProved` precedent); their
    /// `DirtyState` is never read.
    ///
    /// **Per-LB independent logs (the heap-faithful structure).** The batch store
    /// mirrors the heap prototype's `unordered_map<Memory*, vector<Mail>>` — each
    /// LB's batches grow INDEPENDENTLY, so a commit is O(1) and never touches
    /// another LB's bytes:
    /// - **`mailBlobPool`** — `PagedVector<char>`: every batch's `Codec<Mail>`
    ///   blob, APPEND-ONLY (a written blob never moves — no cross-LB shift).
    /// - **`mailRefs`** — `PagedVector<BlobRef>`: one ref per committed batch,
    ///   APPEND-ONLY; each LB's refs form a newest-first back-linked chain (the
    ///   pool equivalent of one LB's `vector<Mail>`).
    /// - **`mailHeads`** — `TypedColdMap<int64, MailHead>`: producing-LB key →
    ///   its chain head (newest ref + batch count). Written single-threaded at
    ///   commit; read at pull.
    ///
    /// (An earlier cut stored the batches in a single `TypedColdBlobMap` whose CSR
    /// concatenates all LBs' runs; a commit to any non-tail LB then `memcpy`-shifts
    /// every later LB's bytes — superlinear, a massive slowdown. The append-only
    /// pool + per-LB chain restores the heap's O(1) commit.)
    ///
    /// The routing index is two more cold containers:
    /// - **`mailEdges`** — `ColdMultiMap<int64, int64>`: recipient key → its
    ///   ancestor-key list, set once at registration (it drives the pull — a hash
    ///   map cannot enumerate the cursor's keys by recipient prefix).
    /// - **`mailCursor`** — `TypedColdMap<CursorKey, int32>`: (recipient, ancestor)
    ///   edge → count of ingested batches; every cell pre-created at registration.
    ///
    /// Keys are the `uintptr_t` of a `const Memory*`, NEVER dereferenced (Rule 12;
    /// `exprKey` is not unique). The mail pool is never deloaded, so pointer-valued
    /// keys and grant-order layout carry NO canonical-bytes obligation — proof
    /// output stays content-deterministic via the downstream set-merge + the
    /// absorb's pre-fold origin sort, exactly as the heap prototype did.
    ///
    /// Lifecycle:
    /// - **Register** at grid build (single-threaded): set each recipient's
    ///   ancestor list (`mailEdges`) and pre-create each (recipient, ancestor)
    ///   cursor cell (`mailCursor`). No LB is born mid-run, so the parallel pull
    ///   only reads frozen state and advances pre-existing cells — never mints.
    /// - **Commit** at the post-join seam (single-threaded): the LB's `mailOut`
    ///   is serialized (`Codec<Mail>`), its bytes appended to `mailBlobPool`, a
    ///   `BlobRef` appended to `mailRefs`, and its `mailHeads` chain head bumped.
    /// - **Pull** in `performElemPhase1` (parallel, per-LB): for each ancestor the
    ///   receiver walks the new tail of that ancestor's ref chain, decodes each
    ///   blob, folds it into `mailIn`, and advances only its own cursor cells
    ///   (`setValueAtRelaxed` — disjoint, no shared dirty write). Race-free:
    ///   commits happen only at the seam (logs frozen), the mail arena is never
    ///   compacted/deloaded (vids resolve stably), every read is pure, and the one
    ///   write per edge is a disjoint no-dirty cursor advance.
    ///
    /// @invariant Commits run only at the single-threaded seam; the parallel
    ///            phase only READS the blob pool / refs / heads / ancestor lists
    ///            and writes the active LB's own (disjoint) cursor cells —
    ///            race-free without locks ([I-94]).
    /// @invariant Every cursor cell is pre-created at registration; the pull
    ///            asserts a found cell (an absent one means an unregistered LB
    ///            reached the pull — a bug, Rule 19).
    /// @see mergeBatchInto — the per-batch field-coverage contract.
    /// @see Codec<Mail> (memory.hpp) — the batch blob codec.
    /// @see docs/agentic_swdd/20_core_concepts/03_mail_system.md
    struct MailLog {
        /// @brief Bind the five cold containers to the mail arena.
        ///
        /// @details
        /// All five draw their pages from `arena` (the `ExpressionAnalyzer`-owned
        /// `mailArena` on the never-deloaded mail pool). `dirty` is the mail dirty
        /// flag, present only to satisfy the container constructors — it is NEVER
        /// read (these containers produce no deload image).
        ///
        /// @param arena The mail arena (on `mailMemory()`).
        /// @param dirty The mail dirty flag (never consulted).
        explicit MailLog(LbArena* arena, DirtyState* dirty)
            : mailBlobPool(arena, dirty),
              mailRefs(arena, dirty),
              mailHeads(arena, dirty),
              mailEdges(arena, dirty),
              mailCursor(arena, dirty) {}

        /// Append-only bytes of every committed batch's `Codec<Mail>` blob.
        PagedVector<char> mailBlobPool;

        /// Append-only refs; each LB's batches form a newest-first back-linked chain.
        PagedVector<BlobRef> mailRefs;

        /// Producing-LB key → its chain head (newest ref + batch count).
        TypedColdMap<int64_t, MailHead> mailHeads;

        /// Recipient key → its ancestor-key list (set once at registration).
        ColdMultiMap<PodKeyStore<int64_t>, int64_t> mailEdges;

        /// (recipient, ancestor) edge → count of batches already ingested.
        TypedColdMap<CursorKey, int32_t> mailCursor;

        /// @brief The runtime identity key of an LB — the `uintptr_t` of its
        ///        `const Memory*`, as an `int64_t` (the cold-map family's POD key).
        ///
        /// @details
        /// The pointer is pure identity, never dereferenced (Rule 12). The mail
        /// pool is never deloaded, so a pointer-valued key needs no canonical form.
        ///
        /// @param lb The LB (may be the root).
        /// @return Its identity key.
        static int64_t lbKey(const Memory* lb) {
            return static_cast<int64_t>(reinterpret_cast<std::uintptr_t>(lb));
        }

        /// @brief The cursor key of a (recipient, ancestor) edge.
        ///
        /// @param recipient The pulling LB.
        /// @param ancestor  An ancestor on the recipient's chain.
        /// @return The packed `CursorKey`.
        static CursorKey edgeKey(const Memory* recipient, const Memory* ancestor) {
            return CursorKey{
                static_cast<std::uint64_t>(
                    reinterpret_cast<std::uintptr_t>(recipient)),
                static_cast<std::uint64_t>(
                    reinterpret_cast<std::uintptr_t>(ancestor)) };
        }

        /// @brief Record an LB's ancestor list and pre-create its cursor cells.
        ///
        /// @details
        /// Called once per LB at grid build, single-threaded. `mailEdges` gets the
        /// recipient's whole ancestor run (appended consecutively — `appendToTail`
        /// only ever sees this brand-new-or-current-last key), and `mailCursor`
        /// gets a zeroed cell per (recipient, ancestor) edge. Pre-creating every
        /// cell is what lets the later parallel pull advance with a non-minting
        /// lookup + a disjoint no-dirty write — never an insert / rehash. The root
        /// passes an empty `ancestors`, so it gets no edges and no cursor cells.
        ///
        /// @param lb         The LB being registered (identity key).
        /// @param ancestors  Every ancestor on `lb`'s chain to root.
        /// @invariant The prover registers each LB exactly once per batch.
        void registerLb(const Memory* lb,
                        const std::vector<const Memory*>& ancestors) {
            const int64_t r = lbKey(lb);
            for (const Memory* anc : ancestors) {
                mailEdges.appendToTail(r, lbKey(anc));
                mailCursor.insert(edgeKey(lb, anc), 0);
            }
        }

        /// @brief Serialize one LB's outgoing mail and append it to its chain.
        ///
        /// @details
        /// Called at the single-threaded post-join seam, once per LB per cycle.
        /// O(blob): `Codec<Mail>` serializes `src` (only `statements` +
        /// `exprOriginMap` — see mergeBatchInto), the bytes are APPENDED to
        /// `mailBlobPool` (never shifting another LB's bytes), one `BlobRef` is
        /// appended to `mailRefs` linking to this LB's previous batch, and the LB's
        /// `mailHeads` chain head is bumped (insert on the first commit, in-place
        /// overwrite after). No cross-LB interaction — the heap prototype's O(1)
        /// per-LB `push_back`, on the pool.
        ///
        /// @param lb   The producing LB.
        /// @param src  The LB's `mailOut` for this cycle (moved-from on return).
        void commit(const Memory* lb, Mail src) {
            const std::vector<char> bytes = Codec<Mail>::serialize(src);
            const std::uint32_t start =
                static_cast<std::uint32_t>(mailBlobPool.size());
            mailBlobPool.appendRun(bytes.data(),
                                   static_cast<int32_t>(bytes.size()));
            const int64_t k = lbKey(lb);
            const MailHead* h = mailHeads.find(k);
            const std::int32_t prev = (h != nullptr) ? h->lastRef : -1;
            const std::int32_t cnt = (h != nullptr) ? h->count : 0;
            const std::int32_t refIdx = mailRefs.size();
            mailRefs.push_back(BlobRef{
                start, static_cast<std::uint32_t>(bytes.size()), prev });
            mailHeads.upsert(k, MailHead{ refIdx, cnt + 1 });
        }

        /// @brief Serialize one LB's outgoing routing mailbox (`RoutingColdMail`) and
        ///        append it to its chain — the direct-from-`RoutingColdMail` twin of
        ///        `commit(const Memory*, Mail)`, with no transient heap `Mail`.
        ///
        /// @details Identical chain bookkeeping to the heap overload; only the
        /// serializer differs (`Codec<Mail>::serialize(const RoutingColdMail&)`, which
        /// emits the same bytes as `serialize(src.toHeapMail())`). Called at the
        /// single-threaded post-join commit barrier + the grid-build startup
        /// commit, once per LB per cycle.
        ///
        /// @param lb   The producing LB.
        /// @param src  The LB's `mailOut` (a `RoutingColdMail`) for this cycle.
        void commit(const Memory* lb, const RoutingColdMail& src) {
            // The blob carries GLOBAL mailInterner ids; serialize decodes src's
            // SENDER NameMap ids via lb->nameMap and re-interns them globally
            // (single-threaded commit seam, so the mint is race-free).
            const std::vector<char> bytes =
                Codec<Mail>::serialize(src, lb->nameMap, lb->originInterner);
            const std::uint32_t start =
                static_cast<std::uint32_t>(mailBlobPool.size());
            mailBlobPool.appendRun(bytes.data(),
                                   static_cast<int32_t>(bytes.size()));
            const int64_t k = lbKey(lb);
            const MailHead* h = mailHeads.find(k);
            const std::int32_t prev = (h != nullptr) ? h->lastRef : -1;
            const std::int32_t cnt = (h != nullptr) ? h->count : 0;
            const std::int32_t refIdx = mailRefs.size();
            mailRefs.push_back(BlobRef{
                start, static_cast<std::uint32_t>(bytes.size()), prev });
            mailHeads.upsert(k, MailHead{ refIdx, cnt + 1 });
        }

        /// @brief Reassemble and decode one blob from the (possibly page-straddling)
        ///        blob pool.
        ///
        /// @details
        /// `mailBlobPool` is a contiguous virtual byte space whose physical pages
        /// may split a blob, so the bytes are copied span-by-span (`contiguousRun`)
        /// into a local buffer before `Codec<Mail>::deserialize`. A pure read of
        /// the frozen pool — safe to call concurrently from the parallel pull. The
        /// transient `Mail` is heap, lives only across the immediate
        /// `mergeBatchInto`, and is discarded; the persistent storage stays on the
        /// pool. The straddle path is exercised by the forced-small-page test.
        ///
        /// @param start Byte offset of the blob in `mailBlobPool`.
        /// @param len   Blob length in bytes.
        /// @return The decoded batch (statements + exprOriginMap only).
        Mail readBlob(std::uint32_t start, std::uint32_t len) const {
            std::vector<char> buf;
            buf.reserve(len);
            std::int32_t i = static_cast<std::int32_t>(start);
            const std::int32_t endByte = static_cast<std::int32_t>(start + len);
            while (i < endByte) {
                std::int32_t runLen = 0;
                const char* p = mailBlobPool.contiguousRun(i, runLen);
                const std::int32_t take =
                    (runLen < endByte - i) ? runLen : (endByte - i);
                buf.insert(buf.end(), p, p + take);
                i += take;
            }
            return Codec<Mail>::deserialize(buf.data(),
                                            static_cast<int32_t>(buf.size()));
        }

        /// @brief Decode one blob STRAIGHT into a `RoutingColdMail` inbox, reading
        ///        it pool-native — the `readBlob` + `mergeBatchInto(const Mail&,
        ///        RoutingColdMail&)` fusion, 0% heap and with no reassembly buffer.
        ///
        /// @details Forwards to the pool-cursor
        /// `Codec<Mail>::deserializeInto(mailBlobPool, start, len, inbox)`, which
        /// reads each field straight off the (possibly page-straddling) blob pool
        /// via a `PoolMailSource` and routes each decoded statement / origin
        /// through the `RoutingColdMail` write doors — NO `std::vector<char>`
        /// span-by-span reassembly and NO transient heap `Mail`. A pure read of
        /// the frozen pool (safe under the parallel pull); the result equals
        /// `mergeBatchInto(readBlob(start, len), inbox)` byte-for-byte (the page
        /// straddle is reproduced by the cursor rather than a buffer copy — twin
        /// `deserialize_into_pool_matches_char`).
        ///
        /// @param start Byte offset of the blob in `mailBlobPool`.
        /// @param len   Blob length in bytes.
        /// @param inbox The receiver's `mailIn` (a `RoutingColdMail`), folded in place.
        void readBlobInto(std::uint32_t start, std::uint32_t len,
                          RoutingColdMail& inbox) const {
            Codec<Mail>::deserializeInto(mailBlobPool, start, len, inbox);
        }

        /// @brief Ingest every un-seen ancestor batch into a receiver's routing
        ///        inbox — the PRODUCTION pull, decoding pool-native (0% heap).
        ///
        /// @details
        /// Called in `performElemPhase1` before the pre-burst `mailIn` absorb. For
        /// each ancestor on the recipient's registered list it walks the NEW tail
        /// of that ancestor's ref chain — the `count − cursor` newest refs, the
        /// batches not yet ingested — newest-first via the `prev` back-link,
        /// decoding each blob STRAIGHT into `inbox` (`readBlobInto` →
        /// `Codec<Mail>::deserializeInto(mailBlobPool, ...)` via a `PoolMailSource`,
        /// no transient heap `Mail`), then advances the cursor cell to the
        /// ancestor's count with the disjoint no-dirty write. The walk is
        /// newest-first while the heap prototype merged oldest-first; the order is
        /// irrelevant — statements merge set-wise and the receiver's absorb sorts
        /// origin vectors before its capped fold. Every access is a pure read of
        /// frozen state except the one cursor write per edge, which touches only
        /// this recipient's own cell.
        ///
        /// The chain walk is DUPLICATED verbatim from the `Mail&` oracle overload
        /// (each self-contained, no shared templated helper), so that only this
        /// overload textually names `readBlobInto` — production reachability of the
        /// heap `readBlob` / `mergeBatchInto` oracle path is thereby severed, and
        /// they leave the `performElemPhase1` inventory tree.
        ///
        /// @param recipient  The pulling LB (must be registered).
        /// @param inbox      The receiver's `mailIn` (a `RoutingColdMail`), folded
        ///                   in place.
        /// @invariant `setValueAtRelaxed` writes only this recipient's disjoint
        ///            cursor cells (race-free across recipients — [I-94]).
        /// @see pull(const Memory*, Mail&) — the heap oracle / test twin;
        ///      readBlobInto; Codec<Mail>::deserializeInto.
        void pull(const Memory* recipient, RoutingColdMail& inbox) {
            const int64_t r = lbKey(recipient);
            const int32_t edgesId = mailEdges.lookup(r);
            if (edgesId == 0) return;   // no ancestors (e.g. the root)
            const int32_t ancCount = mailEdges.runLen(edgesId);
            for (int32_t k = 0; k < ancCount; ++k) {
                const int64_t a = mailEdges.valueAt(edgesId, k);
                const MailHead* h = mailHeads.find(a);
                const int32_t n = (h != nullptr) ? h->count : 0;
                const CursorKey ck{ static_cast<std::uint64_t>(r),
                                    static_cast<std::uint64_t>(a) };
                const int32_t* cp = mailCursor.find(ck);
                assert(cp != nullptr
                    && "MailLog::pull: cursor cell missing — every edge is "
                       "pre-created at registration (Rule 19)");
                const int32_t c = *cp;
                std::int32_t ref = (h != nullptr) ? h->lastRef : -1;
                for (int32_t step = 0; step < n - c; ++step) {
                    assert(ref >= 0
                        && "MailLog::pull: ref chain underran the batch count");
                    const BlobRef br = mailRefs[ref];
                    readBlobInto(br.start, br.len, inbox);
                    ref = br.prev;
                }
                if (n != c) mailCursor.setValueAtRelaxed(ck, n);
            }
        }

        /// @brief Ingest every un-seen ancestor batch into a heap `Mail` receiver —
        ///        the retained oracle / test twin (heap `readBlob` +
        ///        `mergeBatchInto`).
        ///
        /// @details
        /// Byte-identical chain walk to the `RoutingColdMail` production overload,
        /// duplicated verbatim (no shared templated helper), differing ONLY in the
        /// per-batch decode: each blob is decoded to a transient heap `Mail`
        /// (`readBlob`) and folded via `mergeBatchInto`. It has NO production
        /// caller — every prover pull passes a `RoutingColdMail` — so it stays out
        /// of the `performElemPhase1` tree; it survives as the differential-test
        /// oracle and the round-trip reference the unit tests build against.
        ///
        /// @param recipient  The pulling LB (must be registered).
        /// @param inbox      A heap `Mail`, merged into in place.
        /// @invariant `setValueAtRelaxed` writes only this recipient's disjoint
        ///            cursor cells (race-free across recipients — [I-94]).
        /// @see pull(const Memory*, RoutingColdMail&) — the production overload;
        ///      readBlob; mergeBatchInto.
        void pull(const Memory* recipient, Mail& inbox) {
            const int64_t r = lbKey(recipient);
            const int32_t edgesId = mailEdges.lookup(r);
            if (edgesId == 0) return;   // no ancestors (e.g. the root)
            const int32_t ancCount = mailEdges.runLen(edgesId);
            for (int32_t k = 0; k < ancCount; ++k) {
                const int64_t a = mailEdges.valueAt(edgesId, k);
                const MailHead* h = mailHeads.find(a);
                const int32_t n = (h != nullptr) ? h->count : 0;
                const CursorKey ck{ static_cast<std::uint64_t>(r),
                                    static_cast<std::uint64_t>(a) };
                const int32_t* cp = mailCursor.find(ck);
                assert(cp != nullptr
                    && "MailLog::pull: cursor cell missing — every edge is "
                       "pre-created at registration (Rule 19)");
                const int32_t c = *cp;
                std::int32_t ref = (h != nullptr) ? h->lastRef : -1;
                for (int32_t step = 0; step < n - c; ++step) {
                    assert(ref >= 0
                        && "MailLog::pull: ref chain underran the batch count");
                    const BlobRef br = mailRefs[ref];
                    const Mail batch = readBlob(br.start, br.len);
                    mergeBatchInto(batch, inbox);
                    ref = br.prev;
                }
                if (n != c) mailCursor.setValueAtRelaxed(ck, n);
            }
        }

        /// @brief Drop all batches, refs, heads, edges, and cursors at
        ///        execution-batch teardown (`destroyGrid`).
        ///
        /// @details
        /// Empties every cold container and returns its pages to the mail arena
        /// (the arena keeps its blocks for the next batch). The `Memory*` keys are
        /// about to be deleted, so this prevents a stale pointer-keyed entry from
        /// aliasing a reused address in the next batch's grid.
        void clear() {
            mailBlobPool.clear();
            mailRefs.clear();
            mailHeads.resetToFresh();
            mailEdges.resetToFresh();
            mailCursor.resetToFresh();
        }

        /// @brief Whether the log holds nothing — no blobs, refs, heads, edges, or
        ///        cursors.
        ///
        /// @return `true` when all five cold containers are empty (the
        ///         post-`clear` / pre-registration state).
        bool empty() const {
            return mailBlobPool.empty() && mailRefs.empty()
                && mailHeads.empty() && mailEdges.empty()
                && mailCursor.empty();
        }
    };

}  // namespace gl
